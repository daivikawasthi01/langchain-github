"""The signed-in fetch path: issues, comments and closing PRs through GraphQL."""

import pytest
import respx

from langchain_github import GitHubClient, GitHubIssuesLoader, GitHubIssuesRetriever
from langchain_github._client import _Core

from .fake_issues import REPO, FakeIssues


@pytest.fixture
def gql(monkeypatch):
    monkeypatch.setattr(_Core, "graphql_issues", True)
    f = FakeIssues()
    with respx.mock(assert_all_called=False) as router:
        f.mount(router)
        yield f


def threads_by_number(docs):
    return {d.metadata["number"]: d.metadata for d in docs if d.metadata["kind"] == "issue"}


def test_signed_in_index_build_uses_graphql_not_the_rest_listings(gql):
    docs = GitHubIssuesLoader(REPO, token="t").load()
    assert "graphql:issues" in gql.calls
    assert not any(c.endswith(("/issues/comments", "/pulls")) for c in gql.calls)
    meta = threads_by_number(docs)
    assert meta[1]["resolution"] == "fixed" and meta[1]["resolved_by"] == [9]
    assert meta[3]["resolution"] == "duplicate" and meta[3]["duplicate_of"] == 1
    assert meta[4]["resolution"] == "wontfix"
    assert 9 not in meta  # pull requests are not issues


def test_signed_out_uses_rest_and_skips_discussions(gql):
    client = GitHubClient()
    client.core._headers.pop("Authorization", None)
    loader = GitHubIssuesLoader(REPO, client=client)
    with pytest.warns(UserWarning, match="gh auth login"):
        docs = loader.load()
    assert "graphql:issues" not in gql.calls and f"/repos/{REPO}/pulls" in gql.calls
    assert threads_by_number(docs)[1]["resolved_by"] == [9]


def test_search_mode_fetches_all_candidates_in_one_request(gql):
    gql.search_hits = [1, 2, 3, 4]
    r = GitHubIssuesRetriever(repo=REPO, token="t", mode="search", use_github_search=False)
    docs = r.invoke("pool exhausted")
    assert gql.calls.count("graphql:batch") == 1
    assert not any(c.endswith(("/timeline", "/comments")) for c in gql.calls)
    assert {d.metadata["number"] for d in docs} >= {1, 2}


def test_pull_requests_and_missing_numbers_in_search_hits_are_skipped(gql):
    from langchain_github._threads import fetch_threads

    client = GitHubClient("t")
    out = fetch_threads(client, REPO, [1, 9, 404, 2])  # 9 is a PR, 404 does not exist
    assert [t.number for t in out] == [1, 2]
    assert client.core.graphql_issues is True  # one missing number must not disable GraphQL


def test_closing_pr_from_another_repository_is_ignored(gql):
    gql.fork_pr = True
    meta = threads_by_number(GitHubIssuesLoader(REPO, token="t").load())
    assert meta[2]["resolved_by"] == []


def test_falls_back_to_rest_when_github_refuses_the_graphql_query(gql):
    gql.graphql_failures = ["Something went wrong while executing your query"]
    loader = GitHubIssuesLoader(REPO, token="t", include_discussions=False)
    meta = threads_by_number(loader.load())
    assert f"/repos/{REPO}/pulls" in gql.calls
    assert meta[1]["resolved_by"] == [9]
    assert loader.client.core.graphql_issues is False


def test_threads_with_more_comments_than_one_page_are_completed_from_rest(gql):
    gql.overflow = {1}
    meta_docs = GitHubIssuesLoader(REPO, token="t", include_discussions=False).load()
    assert f"/repos/{REPO}/issues/1/comments" in gql.calls
    assert any(d.metadata["number"] == 1 and d.metadata["part"] == "comment" for d in meta_docs)


@pytest.mark.parametrize(
    "base,expected",
    [
        ("https://api.github.com", "https://api.github.com/graphql"),
        ("https://ghe.example.com/api/v3", "https://ghe.example.com/api/graphql"),
    ],
)
def test_graphql_url_for_github_and_enterprise(base, expected):
    assert GitHubClient("t", base_url=base).core.graphql_url == expected


def test_backport_pr_that_github_did_not_link_still_counts_as_the_fix(gql):
    gql.pulls.append(
        {"number": 10, "title": "Backport", "body": "Fixes #2", "merged_at": "x", "linked": False}
    )
    meta = threads_by_number(GitHubIssuesLoader(REPO, token="t").load())
    assert meta[2]["resolved_by"] == [10]
    assert meta[1]["resolved_by"] == [9]
