"""HTTP client for the GitHub REST and GraphQL APIs.

Respects the documented limits: primary limits are read from ``x-ratelimit-*`` headers, search
endpoints are throttled locally (30/min, semantic search 10/min, code search 10/min), secondary
limits and 5xx responses are retried with backoff, and unchanged resources are revalidated with
ETags (a ``304`` does not count against the primary limit).
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx
from pydantic import SecretStr

DEFAULT_BASE_URL = "https://api.github.com"
API_VERSION = "2022-11-28"

# resource name -> list of (limit, period seconds) windows that must all have room.
_LOCAL_LIMITS: dict[str, list[tuple[int, float]]] = {
    "search": [(30, 60.0)],
    "semantic": [(30, 60.0), (10, 60.0)],
    "code_search": [(10, 60.0)],
}


class GitHubError(Exception):
    """Base error for GitHub API failures. ``status_code`` is the HTTP status, if there was one."""

    status_code: int | None = None

    def with_status(self, status: int) -> GitHubError:
        self.status_code = status
        return self


class GitHubGraphQLError(GitHubError):
    """GitHub answered a GraphQL request with an ``errors`` list."""


class GitHubAuthError(GitHubError):
    """Bad credentials or missing permission."""


class GitHubNotFoundError(GitHubError):
    """The resource does not exist or the token cannot see it."""


class GitHubRateLimitError(GitHubError):
    """A rate limit would require waiting longer than ``max_wait`` seconds."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class _Window:
    """Sliding-window limiter. ``reserve`` books a slot and says how long to wait for it."""

    def __init__(self, limit: int, period: float, clock: Callable[[], float]) -> None:
        self.limit = limit
        self.period = period
        self._clock = clock
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def reserve(self) -> float:
        with self._lock:
            now = self._clock()
            while self._stamps and now - self._stamps[0] >= self.period:
                self._stamps.popleft()
            if len(self._stamps) < self.limit:
                self._stamps.append(now)
                return 0.0
            slot = self._stamps.popleft() + self.period
            self._stamps.append(slot)
            return max(0.0, slot - now)


class _Limiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._windows = {
            name: [_Window(limit, period, clock) for limit, period in specs]
            for name, specs in _LOCAL_LIMITS.items()
        }

    def reserve(self, resource: str) -> float:
        return max((w.reserve() for w in self._windows.get(resource, [])), default=0.0)


_NO_TOKEN_HINT = (
    " You are not signed in, which allows only 60 requests an hour. Run `gh auth login` once"
    " (https://cli.github.com) or set GITHUB_TOKEN."
)


def _gh_cli_token() -> str | None:
    """Token from the GitHub CLI (``gh auth login``), or ``None`` if it is missing or logged out."""
    gh = shutil.which("gh")
    if not gh:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            [gh, "auth", "token"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _resolve_token(token: str | SecretStr | None, base_url: str = DEFAULT_BASE_URL) -> str | None:
    """The explicit token, else ``GITHUB_TOKEN``, ``GH_TOKEN``, then the GitHub CLI's login.

    The CLI's token belongs to github.com, so it is only used for the default API URL.
    """
    if isinstance(token, SecretStr):
        token = token.get_secret_value()
    found = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not found and base_url.rstrip("/") == DEFAULT_BASE_URL:
        found = _gh_cli_token()
    return found or None


def _cache_key(method: str, url: str, params: dict[str, Any] | None) -> str:
    return f"{method} {url} {sorted((params or {}).items())}"


class _Core:
    """State and decisions shared by the sync and async clients."""

    def __init__(
        self,
        token: str | SecretStr | None,
        base_url: str,
        max_retries: int,
        max_wait: float,
        user_agent: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.max_wait = max_wait
        self.limiter = _Limiter()
        self.etags: dict[str, tuple[str, Any]] = {}
        self.requests_made = 0
        self.last_rate_limit: dict[str, int] = {}
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": user_agent,
        }
        resolved = _resolve_token(token, base_url)
        if resolved:
            self._headers["Authorization"] = f"Bearer {resolved}"

    @property
    def authenticated(self) -> bool:
        return "Authorization" in self._headers

    def headers(self, accept: str | None, etag: str | None) -> dict[str, str]:
        out = dict(self._headers)
        if accept:
            out["Accept"] = accept
        if etag:
            out["If-None-Match"] = etag
        return out

    def url(self, path: str) -> str:
        return path if path.startswith("http") else f"{self.base_url}{path}"

    def record(self, resp: httpx.Response) -> None:
        self.requests_made += 1
        h = resp.headers
        if "x-ratelimit-remaining" in h:
            self.last_rate_limit = {
                "remaining": int(h["x-ratelimit-remaining"]),
                "limit": int(h.get("x-ratelimit-limit", 0)),
                "reset": int(h.get("x-ratelimit-reset", 0)),
            }

    def decide(self, resp: httpx.Response, attempt: int) -> float | None:
        """Return seconds to wait before retrying, ``None`` if the response is final.

        Raises for errors that retrying cannot fix.
        """
        status = resp.status_code
        if status < 400 or status == 304:
            return None
        if status == 401:
            raise GitHubAuthError(
                "Bad credentials (401). Check your GitHub token, or run `gh auth login` again."
            ).with_status(401)
        if status == 404:
            raise GitHubNotFoundError(
                f"Not found: {resp.request.url.path}. "
                + (
                    "Check the spelling. If it is a private repository, sign in with "
                    "`gh auth login` (or set GITHUB_TOKEN) so GitHub can show it to you."
                    if not self.authenticated
                    else "Check the spelling, and that your token is allowed to read it."
                )
            ).with_status(404)
        if status in (403, 429):
            wait = self._rate_limit_wait(resp)
            if wait is None:
                raise GitHubAuthError(f"Forbidden (403): {self._message(resp)}").with_status(403)
            if wait > self.max_wait or attempt >= self.max_retries:
                raise GitHubRateLimitError(
                    f"Rate limited; would need to wait {wait:.0f}s ({self._message(resp)})."
                    + ("" if self.authenticated else _NO_TOKEN_HINT),
                    wait,
                ).with_status(status)
            return wait
        if status >= 500 and attempt < self.max_retries:
            return min(2.0**attempt + random.random(), self.max_wait)
        raise GitHubError(f"GitHub returned {status}: {self._message(resp)}").with_status(status)

    def transport_retry(self, exc: httpx.TransportError, attempt: int) -> float:
        """Seconds to wait after a dropped or timed-out connection; raises once retries run out."""
        if attempt >= self.max_retries:
            raise GitHubError(
                f"Could not reach GitHub after {attempt + 1} attempts: {exc!r}"
            ) from exc
        return min(2.0**attempt + random.random(), self.max_wait)

    def _rate_limit_wait(self, resp: httpx.Response) -> float | None:
        h = resp.headers
        if "retry-after" in h:
            return float(h["retry-after"])
        if h.get("x-ratelimit-remaining") == "0":
            return max(0.0, int(h.get("x-ratelimit-reset", 0)) - time.time()) + 1.0
        if "rate limit" in self._message(resp).lower() or resp.status_code == 429:
            return 60.0
        return None

    @staticmethod
    def _message(resp: httpx.Response) -> str:
        try:
            return str(resp.json().get("message", resp.text[:200]))
        except Exception:
            return resp.text[:200]

    @staticmethod
    def check_declared_size(resp: httpx.Response, max_bytes: int | None) -> None:
        declared = int(resp.headers.get("content-length", 0))
        if max_bytes and declared > max_bytes:
            raise GitHubError(
                f"Archive is {declared / 1e6:.0f} MB, above max_archive_mb "
                f"({max_bytes / 1e6:.0f} MB). Narrow the filters or raise the limit."
            )

    @staticmethod
    def check_streamed_size(size: int, max_bytes: int | None) -> None:
        if max_bytes and size > max_bytes:
            raise GitHubError("Archive exceeded max_archive_mb while downloading.")

    @staticmethod
    def json(resp: httpx.Response) -> Any:
        """The body as JSON, or a ``GitHubError`` if it is not JSON (proxy, captive portal)."""
        try:
            return resp.json()
        except ValueError as exc:
            raise GitHubError(
                f"GitHub returned a response that is not JSON (HTTP {resp.status_code}). "
                "A proxy, VPN or captive portal may be in the way. "
                f"Start of the reply: {resp.text[:80]!r}"
            ) from exc

    def parse(self, resp: httpx.Response, key: str, etag: bool) -> Any:
        if resp.status_code == 304:
            return self.etags[key][1]
        data = self.json(resp)
        if etag and "etag" in resp.headers:
            self.etags[key] = (resp.headers["etag"], data)
        return data


def _parse_text(core: _Core, resp: httpx.Response, key: str, etag: bool) -> str:
    if resp.status_code == 304:
        return str(core.etags[key][1])
    if etag and "etag" in resp.headers:
        core.etags[key] = (resp.headers["etag"], resp.text)
    return resp.text


def _next_link(resp: httpx.Response) -> str | None:
    nxt = resp.links.get("next")
    return nxt["url"] if nxt else None


class GitHubClient:
    """Synchronous client."""

    def __init__(
        self,
        token: str | SecretStr | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_wait: float = 120.0,
        user_agent: str = "langchain-github",
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._core = _Core(token, base_url, max_retries, max_wait, user_agent)
        self._http = http or httpx.Client(timeout=timeout, follow_redirects=True)
        self._sleep = sleep

    @property
    def core(self) -> _Core:
        return self._core

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _send(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        resource: str = "core",
        accept: str | None = None,
        etag: bool = False,
    ) -> tuple[httpx.Response, str]:
        c = self._core
        url = c.url(path)
        key = _cache_key(method, url, params)
        cached = c.etags.get(key, (None, None))[0] if etag else None
        attempt = 0
        while True:
            wait = c.limiter.reserve(resource)
            if wait > c.max_wait:
                raise GitHubRateLimitError(f"Local {resource} limit: wait {wait:.0f}s", wait)
            if wait:
                self._sleep(wait)
            try:
                resp = self._http.request(
                    method, url, params=params, json=json, headers=c.headers(accept, cached)
                )
            except httpx.TransportError as exc:
                self._sleep(c.transport_retry(exc, attempt))
                attempt += 1
                continue
            c.record(resp)
            retry = c.decide(resp, attempt)
            if retry is None:
                return resp, key
            self._sleep(retry)
            attempt += 1

    def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        resource: str = "core",
        etag: bool = True,
    ) -> Any:
        resp, key = self._send("GET", path, params=params, resource=resource, etag=etag)
        return self._core.parse(resp, key, etag)

    def get_text(
        self,
        path: str,
        *,
        accept: str,
        params: dict[str, Any] | None = None,
        etag: bool = True,
    ) -> str:
        resp, key = self._send("GET", path, params=params, accept=accept, etag=etag)
        return _parse_text(self._core, resp, key, etag)

    def paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        resource: str = "core",
        items_key: str | None = None,
        max_items: int | None = None,
    ) -> Iterator[Any]:
        params = {"per_page": 100, **(params or {})}
        url: str | None = path
        seen = 0
        while url:
            resp, _ = self._send("GET", url, params=params, resource=resource)
            body = self._core.json(resp)
            for item in body[items_key] if items_key else body:
                yield item
                seen += 1
                if max_items is not None and seen >= max_items:
                    return
            url, params = _next_link(resp), None

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        resp, _ = self._send(
            "POST", "/graphql", json={"query": query, "variables": variables or {}}
        )
        body = self._core.json(resp)
        if body.get("errors"):
            raise GitHubGraphQLError(f"GraphQL error: {body['errors'][0].get('message')}")
        return body["data"]  # type: ignore[no-any-return]

    def download(self, path: str, *, max_bytes: int | None = None) -> bytes:
        """Download a (possibly redirected) binary resource such as a repository tarball.

        Rate limits, 5xx responses and dropped connections are retried like any other request.
        """
        c = self._core
        attempt = 0
        while True:
            wait = c.limiter.reserve("core")
            if wait > c.max_wait:
                raise GitHubRateLimitError(f"Local core limit: wait {wait:.0f}s", wait)
            if wait:
                self._sleep(wait)
            try:
                with self._http.stream(
                    "GET", c.url(path), headers=c.headers("application/vnd.github+json", None)
                ) as resp:
                    c.record(resp)
                    if resp.status_code >= 400:
                        resp.read()
                    retry = c.decide(resp, attempt)
                    if retry is None:
                        c.check_declared_size(resp, max_bytes)
                        buf = bytearray()
                        for chunk in resp.iter_bytes():
                            buf.extend(chunk)
                            c.check_streamed_size(len(buf), max_bytes)
                        return bytes(buf)
            except httpx.TransportError as exc:
                self._sleep(c.transport_retry(exc, attempt))
                attempt += 1
                continue
            self._sleep(retry)
            attempt += 1


class AsyncGitHubClient:
    """Asynchronous client with the same behaviour as :class:`GitHubClient`."""

    def __init__(
        self,
        token: str | SecretStr | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_wait: float = 120.0,
        user_agent: str = "langchain-github",
        http: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._core = _Core(token, base_url, max_retries, max_wait, user_agent)
        self._http = http or httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self._sleep = sleep

    @property
    def core(self) -> _Core:
        return self._core

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncGitHubClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _send(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        resource: str = "core",
        accept: str | None = None,
        etag: bool = False,
    ) -> tuple[httpx.Response, str]:
        c = self._core
        url = c.url(path)
        key = _cache_key(method, url, params)
        cached = c.etags.get(key, (None, None))[0] if etag else None
        attempt = 0
        while True:
            wait = c.limiter.reserve(resource)
            if wait > c.max_wait:
                raise GitHubRateLimitError(f"Local {resource} limit: wait {wait:.0f}s", wait)
            if wait:
                await self._sleep(wait)
            try:
                resp = await self._http.request(
                    method, url, params=params, json=json, headers=c.headers(accept, cached)
                )
            except httpx.TransportError as exc:
                await self._sleep(c.transport_retry(exc, attempt))
                attempt += 1
                continue
            c.record(resp)
            retry = c.decide(resp, attempt)
            if retry is None:
                return resp, key
            await self._sleep(retry)
            attempt += 1

    async def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        resource: str = "core",
        etag: bool = True,
    ) -> Any:
        resp, key = await self._send("GET", path, params=params, resource=resource, etag=etag)
        return self._core.parse(resp, key, etag)

    async def get_text(
        self,
        path: str,
        *,
        accept: str,
        params: dict[str, Any] | None = None,
        etag: bool = True,
    ) -> str:
        resp, key = await self._send("GET", path, params=params, accept=accept, etag=etag)
        return _parse_text(self._core, resp, key, etag)

    async def paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        resource: str = "core",
        items_key: str | None = None,
        max_items: int | None = None,
    ) -> AsyncIterator[Any]:
        params = {"per_page": 100, **(params or {})}
        url: str | None = path
        seen = 0
        while url:
            resp, _ = await self._send("GET", url, params=params, resource=resource)
            body = self._core.json(resp)
            for item in body[items_key] if items_key else body:
                yield item
                seen += 1
                if max_items is not None and seen >= max_items:
                    return
            url, params = _next_link(resp), None

    async def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        resp, _ = await self._send(
            "POST", "/graphql", json={"query": query, "variables": variables or {}}
        )
        body = self._core.json(resp)
        if body.get("errors"):
            raise GitHubGraphQLError(f"GraphQL error: {body['errors'][0].get('message')}")
        return body["data"]  # type: ignore[no-any-return]

    async def download(self, path: str, *, max_bytes: int | None = None) -> bytes:
        c = self._core
        attempt = 0
        while True:
            wait = c.limiter.reserve("core")
            if wait > c.max_wait:
                raise GitHubRateLimitError(f"Local core limit: wait {wait:.0f}s", wait)
            if wait:
                await self._sleep(wait)
            try:
                async with self._http.stream(
                    "GET", c.url(path), headers=c.headers("application/vnd.github+json", None)
                ) as resp:
                    c.record(resp)
                    if resp.status_code >= 400:
                        await resp.aread()
                    retry = c.decide(resp, attempt)
                    if retry is None:
                        c.check_declared_size(resp, max_bytes)
                        buf = bytearray()
                        async for chunk in resp.aiter_bytes():
                            buf.extend(chunk)
                            c.check_streamed_size(len(buf), max_bytes)
                        return bytes(buf)
            except httpx.TransportError as exc:
                await self._sleep(c.transport_retry(exc, attempt))
                attempt += 1
                continue
            await self._sleep(retry)
            attempt += 1
