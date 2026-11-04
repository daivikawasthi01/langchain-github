import httpx
import pytest
import respx

from langchain_github._client import (
    AsyncGitHubClient,
    GitHubAuthError,
    GitHubClient,
    GitHubError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    _Limiter,
    _resolve_token,
    _Window,
)

API = "https://api.github.com"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_client(**kw):
    slept: list[float] = []
    client = GitHubClient("tok", sleep=slept.append, **kw)
    return client, slept


def test_window_allows_limit_then_makes_callers_wait():
    clock = Clock()
    w = _Window(2, 60.0, clock)
    assert w.reserve() == 0.0
    clock.now += 10
    assert w.reserve() == 0.0
    clock.now += 10  # 20s in, oldest slot frees at t=60
    assert w.reserve() == pytest.approx(40.0)
    # that third slot is booked for t=60; the next one must wait for the second slot (t=70)
    assert w.reserve() == pytest.approx(50.0)


def test_window_recovers_after_period():
    clock = Clock()
    w = _Window(1, 60.0, clock)
    assert w.reserve() == 0.0
    clock.now += 61
    assert w.reserve() == 0.0


def test_semantic_search_respects_both_buckets():
    clock = Clock()
    lim = _Limiter(clock)
    waits = [lim.reserve("semantic") for _ in range(11)]
    assert waits[:10] == [0.0] * 10
    assert waits[10] > 0
    assert _Limiter(clock).reserve("core") == 0.0


@respx.mock
def test_sends_auth_and_version_headers():
    route = respx.get(f"{API}/repos/o/r").mock(return_value=httpx.Response(200, json={"ok": 1}))
    client, _ = make_client()
    assert client.get_json("/repos/o/r") == {"ok": 1}
    sent = route.calls.last.request.headers
    assert sent["authorization"] == "Bearer tok"
    assert sent["x-github-api-version"] == "2022-11-28"


@respx.mock
def test_etag_revalidation_returns_cached_body_on_304():
    route = respx.get(f"{API}/repos/o/r/git/trees/main")
    route.side_effect = [
        httpx.Response(200, json={"sha": "abc"}, headers={"etag": '"v1"'}),
        httpx.Response(304),
    ]
    client, _ = make_client()
    first = client.get_json("/repos/o/r/git/trees/main", {"recursive": "1"})
    second = client.get_json("/repos/o/r/git/trees/main", {"recursive": "1"})
    assert first == second == {"sha": "abc"}
    assert "if-none-match" not in route.calls[0].request.headers
    assert route.calls[1].request.headers["if-none-match"] == '"v1"'


@respx.mock
def test_retry_after_is_honoured_then_succeeds():
    route = respx.get(f"{API}/x")
    route.side_effect = [
        httpx.Response(403, headers={"retry-after": "7"}, json={"message": "secondary rate limit"}),
        httpx.Response(200, json={"ok": True}),
    ]
    client, slept = make_client()
    assert client.get_json("/x") == {"ok": True}
    assert slept == [7.0]
    assert client.core.requests_made == 2


@respx.mock
def test_wait_longer_than_max_wait_raises_instead_of_hanging():
    respx.get(f"{API}/x").mock(
        return_value=httpx.Response(429, headers={"retry-after": "3600"}, json={"message": "slow"})
    )
    client, slept = make_client(max_wait=120)
    with pytest.raises(GitHubRateLimitError) as exc:
        client.get_json("/x")
    assert exc.value.retry_after == 3600
    assert slept == []


@respx.mock
def test_5xx_retried_with_backoff():
    route = respx.get(f"{API}/x")
    route.side_effect = [httpx.Response(502), httpx.Response(200, json=[1])]
    client, slept = make_client()
    assert client.get_json("/x") == [1]
    assert len(slept) == 1 and slept[0] >= 1.0


@respx.mock
def test_5xx_gives_up_after_max_retries():
    respx.get(f"{API}/x").mock(return_value=httpx.Response(503, json={"message": "down"}))
    client, _ = make_client(max_retries=2)
    with pytest.raises(GitHubError):
        client.get_json("/x")


@respx.mock
def test_connect_timeout_is_retried():
    route = respx.get(f"{API}/x")
    route.side_effect = [httpx.ConnectTimeout("timed out"), httpx.Response(200, json=[1])]
    client, slept = make_client()
    assert client.get_json("/x") == [1]
    assert len(slept) == 1


@respx.mock
def test_unreachable_github_raises_github_error_not_httpx():
    respx.get(f"{API}/x").mock(side_effect=httpx.ConnectError("down"))
    client, slept = make_client(max_retries=2)
    with pytest.raises(GitHubError, match="Could not reach GitHub"):
        client.get_json("/x")
    assert len(slept) == 2


@respx.mock
async def test_async_connect_timeout_is_retried():
    route = respx.get(f"{API}/x")
    route.side_effect = [httpx.ConnectTimeout("timed out"), httpx.Response(200, json=[1])]

    async def fake_sleep(s: float) -> None:
        pass

    async with AsyncGitHubClient("tok", sleep=fake_sleep) as client:
        assert await client.get_json("/x") == [1]


@respx.mock
@pytest.mark.parametrize(
    "status,exc", [(401, GitHubAuthError), (404, GitHubNotFoundError), (403, GitHubAuthError)]
)
def test_terminal_errors_are_not_retried(status, exc):
    route = respx.get(f"{API}/x").mock(return_value=httpx.Response(status, json={"message": "no"}))
    client, slept = make_client()
    with pytest.raises(exc):
        client.get_json("/x")
    assert route.call_count == 1 and slept == []


@respx.mock
def test_paginate_follows_link_header_and_respects_max_items():
    respx.get(f"{API}/items", params={"per_page": "100"}).mock(
        return_value=httpx.Response(
            200, json=[1, 2], headers={"link": f'<{API}/items?page=2>; rel="next"'}
        )
    )
    respx.get(f"{API}/items", params={"page": "2"}).mock(return_value=httpx.Response(200, json=[3]))
    client, _ = make_client()
    assert list(client.paginate("/items")) == [1, 2, 3]
    assert list(client.paginate("/items", max_items=2)) == [1, 2]


@respx.mock
def test_graphql_error_raises():
    respx.post(f"{API}/graphql").mock(
        return_value=httpx.Response(200, json={"errors": [{"message": "bad field"}]})
    )
    client, _ = make_client()
    with pytest.raises(GitHubError, match="bad field"):
        client.graphql("{ x }")


@respx.mock
def test_download_refuses_oversized_archive():
    respx.get(f"{API}/repos/o/r/tarball/main").mock(
        return_value=httpx.Response(200, content=b"x" * 10, headers={"content-length": "500000000"})
    )
    client, _ = make_client()
    with pytest.raises(GitHubError, match="max_archive_mb"):
        client.download("/repos/o/r/tarball/main", max_bytes=1_000_000)


def test_token_comes_from_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "envtok")
    assert GitHubClient().core.authenticated
    monkeypatch.delenv("GITHUB_TOKEN")
    assert not GitHubClient().core.authenticated


@respx.mock
async def test_async_client_has_same_retry_and_etag_behaviour():
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    route = respx.get(f"{API}/x")
    route.side_effect = [
        httpx.Response(403, headers={"retry-after": "2"}, json={"message": "secondary rate limit"}),
        httpx.Response(200, json={"v": 1}, headers={"etag": '"e"'}),
        httpx.Response(304),
    ]
    async with AsyncGitHubClient("tok", sleep=fake_sleep) as client:
        assert await client.get_json("/x") == {"v": 1}
        assert await client.get_json("/x") == {"v": 1}
    assert slept == [2.0]
    assert route.calls[2].request.headers["if-none-match"] == '"e"'


@respx.mock
def test_get_text_revalidates_with_etag():
    route = respx.get(f"{API}/repos/o/r/commits/main")
    route.side_effect = [
        httpx.Response(200, text="a" * 40, headers={"etag": '"c1"'}),
        httpx.Response(304),
    ]
    client, _ = make_client()
    sha_media = "application/vnd.github.sha"
    assert client.get_text("/repos/o/r/commits/main", accept=sha_media) == "a" * 40
    assert client.get_text("/repos/o/r/commits/main", accept=sha_media) == "a" * 40
    assert route.calls[1].request.headers["if-none-match"] == '"c1"'
    assert route.calls[1].request.headers["accept"] == sha_media


def test_token_falls_back_to_gh_token_then_gh_cli(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "ghtok")
    assert _resolve_token(None) == "ghtok"
    monkeypatch.delenv("GH_TOKEN")
    monkeypatch.setattr("langchain_github._client._gh_cli_token", lambda: "clitok")
    assert _resolve_token(None) == "clitok"
    assert _resolve_token("explicit") == "explicit"
    # the CLI's github.com login must not be sent to another host
    assert _resolve_token(None, "https://ghe.example.com/api/v3") is None


def test_gh_cli_token_handles_missing_and_logged_out(monkeypatch):
    import subprocess

    from langchain_github import _client

    monkeypatch.setattr(_client.shutil, "which", lambda _: None)
    assert _client._gh_cli_token() is None
    monkeypatch.setattr(_client.shutil, "which", lambda _: "/usr/bin/gh")
    done = subprocess.CompletedProcess([], 0, stdout="abc\n", stderr="")
    monkeypatch.setattr(_client.subprocess, "run", lambda *a, **k: done)
    assert _client._gh_cli_token() == "abc"
    failed = subprocess.CompletedProcess([], 1, stdout="", stderr="not logged in")
    monkeypatch.setattr(_client.subprocess, "run", lambda *a, **k: failed)
    assert _client._gh_cli_token() is None
