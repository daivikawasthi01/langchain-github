"""Document loaders for GitHub repositories."""

from __future__ import annotations

import asyncio
import re
import warnings
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pydantic import SecretStr
from typing_extensions import Self

from ._chunk import Chunker, language_for
from ._client import (
    DEFAULT_BASE_URL,
    AsyncGitHubClient,
    GitHubClient,
    GitHubError,
    GitHubNotFoundError,
)
from ._repo import normalize_repo
from ._threads import Thread, fetch_all_issues, fetch_discussions, thread_documents
from ._tree import (
    BlobStore,
    PathFilter,
    TreeEntry,
    decode_blob,
    extract_tarball,
    permalink,
    select_entries,
)

_SHA_MEDIA = "application/vnd.github.sha"
_BLOB_CONCURRENCY = 8


@dataclass(frozen=True)
class RepoSnapshot:
    """A repository at one exact commit, with the files that pass the filters."""

    repo: str
    ref: str
    commit_sha: str
    entries: tuple[TreeEntry, ...]


class GitHubRepositoryLoader(BaseLoader):
    """Load a repository's code and docs as line-anchored chunks at any branch, tag or commit.

    * Works at an exact commit, so every :class:`~langchain_core.documents.Document` carries a
      permalink (``.../blob/<sha>/<path>#L10-L42``) that never drifts.
    * Fetches many files with a single tarball request instead of one request per file, and
      reuses unchanged files across commits via a content-addressed cache.
    * Revalidates the ref with ETags, so asking "has anything changed?" costs no rate limit.

    Args:
        repo: ``"owner/name"``.
        ref: Branch, tag or commit SHA. Defaults to the repository's default branch.
        token: GitHub token. Falls back to ``GITHUB_TOKEN``, ``GH_TOKEN``, then the GitHub CLI
            login (``gh auth login``). Optional for public repos, but unauthenticated requests
            are limited to 60 per hour.
        include: Glob patterns a path must match (e.g. ``["docs/**", "*.py"]``).
        exclude: Glob patterns to drop. Replaces the defaults (``node_modules/``, lock files,
            minified assets, ...) when given.
        extensions: Only load these file extensions.
        max_file_size: Skip files larger than this many bytes.
        split: Split files into chunks of about ``chunk_size`` characters (language-aware).
        cache_dir: Persist downloaded file contents here, keyed by git blob SHA.
        blob_fetch_threshold: Above this many uncached files, use one tarball download.
        max_archive_mb: Refuse tarball downloads larger than this.
    """

    def __init__(
        self,
        repo: str,
        *,
        ref: str | None = None,
        token: str | SecretStr | None = None,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        extensions: Sequence[str] | None = None,
        max_file_size: int = 300_000,
        split: bool = True,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        cache_dir: str | Path | None = None,
        blob_fetch_threshold: int = 25,
        max_archive_mb: int = 200,
        base_url: str = DEFAULT_BASE_URL,
        client: GitHubClient | None = None,
        async_client: AsyncGitHubClient | None = None,
    ) -> None:
        self.repo = normalize_repo(repo)
        self.ref = ref
        self.split = split
        self.blob_fetch_threshold = blob_fetch_threshold
        self.max_archive_bytes = max_archive_mb * 1_000_000
        self._filter = PathFilter(include, exclude, extensions, max_file_size)
        self._chunker = Chunker(chunk_size, chunk_overlap)
        self._store = BlobStore(cache_dir)
        self._token = token
        self._base_url = base_url
        self._client = self._given_client = client
        self._async_client = self._given_async_client = async_client
        self._default_branch: str | None = None

    # -- clients are created lazily so constructing a loader never opens sockets -----------------

    @property
    def client(self) -> GitHubClient:
        if self._client is None:
            self._client = GitHubClient(self._token, base_url=self._base_url)
        return self._client

    @property
    def async_client(self) -> AsyncGitHubClient:
        if self._async_client is None:
            self._async_client = AsyncGitHubClient(self._token, base_url=self._base_url)
        return self._async_client

    def close(self) -> None:
        """Close the connection this loader opened (a client you passed in is left alone)."""
        if self._client is not None and self._client is not self._given_client:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._async_client is not None and self._async_client is not self._given_async_client:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- snapshot ---------------------------------------------------------------------------------

    def snapshot(self) -> RepoSnapshot:
        """Resolve the ref to a commit and list the files that pass the filters."""
        c = self.client
        ref = self.ref or self._default_branch
        if ref is None:
            ref = self._default_branch = c.get_json(f"/repos/{self.repo}")["default_branch"]
        try:
            sha = (
                self._sha_or_none(ref)
                or c.get_text(
                    f"/repos/{self.repo}/commits/{quote(ref, safe='/')}", accept=_SHA_MEDIA
                ).strip()
            )
        except GitHubError as exc:
            raise self._ref_error(ref, exc) from exc
        tree = c.get_json(f"/repos/{self.repo}/git/trees/{sha}", {"recursive": "1"})
        nodes = tree["tree"]
        if tree.get("truncated"):
            nodes = self._walk_truncated(sha)
        return self._snapshot_of(ref, sha, nodes)

    def _snapshot_of(self, ref: str, sha: str, nodes: list[dict[str, Any]]) -> RepoSnapshot:
        entries = tuple(select_entries(nodes, self._filter))
        if not entries and nodes:
            warnings.warn(
                f"No files in {self.repo}@{ref} passed your include/exclude/extensions/"
                f"max_file_size filters, so there is nothing to search.",
                stacklevel=3,
            )
        return RepoSnapshot(self.repo, ref, sha, entries)

    async def asnapshot(self) -> RepoSnapshot:
        c = self.async_client
        ref = self.ref or self._default_branch
        if ref is None:
            info = await c.get_json(f"/repos/{self.repo}")
            ref = self._default_branch = info["default_branch"]
        try:
            sha = (
                self._sha_or_none(ref)
                or (
                    await c.get_text(
                        f"/repos/{self.repo}/commits/{quote(ref, safe='/')}", accept=_SHA_MEDIA
                    )
                ).strip()
            )
        except GitHubError as exc:
            raise self._ref_error(ref, exc) from exc
        tree = await c.get_json(f"/repos/{self.repo}/git/trees/{sha}", {"recursive": "1"})
        nodes = tree["tree"]
        if tree.get("truncated"):
            nodes = await self._awalk_truncated(sha)
        return self._snapshot_of(ref, sha, nodes)

    def _ref_error(self, ref: str, exc: GitHubError) -> GitHubError:
        """GitHub answers an unknown branch/tag/commit with a bare 422; say what it means."""
        if exc.status_code == 422:
            return GitHubNotFoundError(
                f"{self.repo} has no branch, tag or commit named {ref!r}. "
                "Pass an existing name or a full or short commit SHA as ref=."
            ).with_status(422)
        return exc

    @staticmethod
    def _sha_or_none(ref: str) -> str | None:
        return ref if re.fullmatch(r"[0-9a-f]{40}", ref) else None

    def _walk_truncated(self, root_sha: str) -> list[dict[str, Any]]:
        """GitHub truncates huge recursive listings; walk the tree one level at a time."""
        out: list[dict[str, Any]] = []
        stack = [("", root_sha)]
        while stack:
            prefix, sha = stack.pop()
            for node in self.client.get_json(f"/repos/{self.repo}/git/trees/{sha}")["tree"]:
                path = f"{prefix}{node['path']}"
                if node["type"] == "tree":
                    stack.append((f"{path}/", node["sha"]))
                else:
                    out.append({**node, "path": path})
        return out

    async def _awalk_truncated(self, root_sha: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        stack = [("", root_sha)]
        while stack:
            prefix, sha = stack.pop()
            tree = await self.async_client.get_json(f"/repos/{self.repo}/git/trees/{sha}")
            for node in tree["tree"]:
                path = f"{prefix}{node['path']}"
                if node["type"] == "tree":
                    stack.append((f"{path}/", node["sha"]))
                else:
                    out.append({**node, "path": path})
        return out

    # -- content ----------------------------------------------------------------------------------

    def _uncached(self, snap: RepoSnapshot) -> list[TreeEntry]:
        return [e for e in snap.entries if not self._store.get(e.sha)[0]]

    def _remember_archive(self, data: bytes, missing: list[TreeEntry]) -> None:
        found = extract_tarball(data, {e.path: e.sha for e in missing})
        for e in missing:
            self._store.put(e.sha, found.get(e.path))

    def _remember_blob(self, entry: TreeEntry, blob: dict[str, Any]) -> None:
        text = decode_blob(blob["content"]) if blob.get("encoding") == "base64" else blob["content"]
        self._store.put(entry.sha, text)

    def fetch(self, snap: RepoSnapshot) -> None:
        """Download any file contents not already cached."""
        missing = self._uncached(snap)
        if len(missing) > self.blob_fetch_threshold:
            data = self.client.download(
                f"/repos/{self.repo}/tarball/{snap.commit_sha}", max_bytes=self.max_archive_bytes
            )
            self._remember_archive(data, missing)
            return
        for e in missing:
            blob = self.client.get_json(f"/repos/{self.repo}/git/blobs/{e.sha}", etag=False)
            self._remember_blob(e, blob)

    async def afetch(self, snap: RepoSnapshot) -> None:
        missing = self._uncached(snap)
        if len(missing) > self.blob_fetch_threshold:
            data = await self.async_client.download(
                f"/repos/{self.repo}/tarball/{snap.commit_sha}", max_bytes=self.max_archive_bytes
            )
            self._remember_archive(data, missing)
            return
        gate = asyncio.Semaphore(_BLOB_CONCURRENCY)

        async def one(e: TreeEntry) -> None:
            async with gate:
                blob = await self.async_client.get_json(
                    f"/repos/{self.repo}/git/blobs/{e.sha}", etag=False
                )
            self._remember_blob(e, blob)

        await asyncio.gather(*(one(e) for e in missing))

    def documents(self, snap: RepoSnapshot) -> Iterator[Document]:
        """Turn already-fetched files into Documents."""
        for e in snap.entries:
            _, text = self._store.get(e.sha)
            if not text or not text.strip():
                continue
            if self.split:
                yield from self._chunker.split(
                    text,
                    repo=snap.repo,
                    ref=snap.ref,
                    commit_sha=snap.commit_sha,
                    path=e.path,
                    blob_sha=e.sha,
                )
            else:
                end = text.count("\n") + 1
                url = permalink(snap.repo, snap.commit_sha, e.path, 1, end)
                yield Document(
                    page_content=text,
                    metadata={
                        "repo": snap.repo,
                        "ref": snap.ref,
                        "commit_sha": snap.commit_sha,
                        "path": e.path,
                        "blob_sha": e.sha,
                        "language": language_for(e.path),
                        "kind": "file",
                        "start_line": 1,
                        "end_line": end,
                        "url": url,
                        "source": url,
                    },
                )

    # -- BaseLoader API ---------------------------------------------------------------------------

    def lazy_load(self) -> Iterator[Document]:
        snap = self.snapshot()
        self.fetch(snap)
        yield from self.documents(snap)

    async def alazy_load(self) -> AsyncIterator[Document]:
        snap = await self.asnapshot()
        await self.afetch(snap)
        for doc in self.documents(snap):
            yield doc


class GitHubIssuesLoader(BaseLoader):
    """Load a repository's issues and discussions as Documents, one per body or comment.

    Every Document carries its thread's ``state``, ``resolution`` (``fixed``, ``answered``,
    ``wontfix``, ``duplicate`` or ``open``), the merged PRs that closed it (``resolved_by``) and a
    link to the exact comment. Pull requests are skipped. Discussions are included when enabled.
    """

    def __init__(
        self,
        repo: str,
        *,
        token: str | SecretStr | None = None,
        state: str = "all",
        include_discussions: bool = True,
        max_threads: int = 1000,
        base_url: str = DEFAULT_BASE_URL,
        client: GitHubClient | None = None,
    ) -> None:
        repo = normalize_repo(repo)
        if state not in {"all", "open", "closed"}:
            raise ValueError("state must be 'all', 'open' or 'closed'")
        self.repo = repo
        self.state = state
        self.include_discussions = include_discussions
        self.max_threads = max_threads
        self._token, self._base_url, self._client = token, base_url, client

    @property
    def client(self) -> GitHubClient:
        if self._client is None:
            self._client = GitHubClient(self._token, base_url=self._base_url)
        return self._client

    def threads(self) -> list[Thread]:
        out = fetch_all_issues(
            self.client, self.repo, max_threads=self.max_threads, state=self.state
        )
        if self.include_discussions and self.state != "open":
            out.extend(fetch_discussions(self.client, self.repo))
        return out

    def lazy_load(self) -> Iterator[Document]:
        for thread in self.threads():
            yield from thread_documents(thread, self.repo)

    async def alazy_load(self) -> AsyncIterator[Document]:
        for thread in await asyncio.to_thread(self.threads):
            for doc in thread_documents(thread, self.repo):
                yield doc
