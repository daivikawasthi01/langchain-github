import pytest
import respx

from langchain_github import GitHubIssuesRetriever, GitHubRepositoryLoader
from langchain_github._client import web_url
from langchain_github._threads import build_issue_thread, thread_documents
from langchain_github._tree import PathFilter

from .fake_github import FakeRepo
from .fake_issues import REPO, FakeIssues, issue


@pytest.mark.parametrize(
    "api,site",
    [
        ("https://api.github.com", "https://github.com"),
        ("https://api.github.com/", "https://github.com"),
        ("https://ghe.corp.example/api/v3", "https://ghe.corp.example"),
    ],
)
def test_web_url_for_github_and_enterprise(api, site):
    assert web_url(api) == site


def test_enterprise_permalinks_point_at_the_enterprise_host():
    api = "https://ghe.corp.example/api/v3"
    repo = FakeRepo("acme/demo", {"a.py": "x = 1\n"})
    with respx.mock(assert_all_called=False) as router:
        # serve the fake under the enterprise API prefix
        import tests.unit_tests.fake_github as fg

        fg.API = api
        try:
            repo.mount(router)
        finally:
            fg.API = "https://api.github.com"
        docs = GitHubRepositoryLoader("acme/demo", token="t", base_url=api).load()
    assert docs[0].metadata["url"].startswith("https://ghe.corp.example/acme/demo/blob/")
    assert docs[0].metadata["source"] == docs[0].metadata["url"]


def test_pull_request_links_use_the_threads_own_host():
    t = build_issue_thread(
        {**issue(1, "t", "b", "closed"), "html_url": "https://ghe.corp.example/acme/demo/issues/1"},
        [],
        [9],
    )
    docs = thread_documents(t, "acme/demo")
    assert docs[0].metadata["resolved_by_urls"] == ["https://ghe.corp.example/acme/demo/pull/9"]


def test_excluded_directories_are_not_walked():
    f = PathFilter()
    assert f.prunes("node_modules") and f.prunes("a/b/node_modules") and f.prunes("dist")
    assert not f.prunes("src") and not f.prunes("docs/node")
    custom = PathFilter(exclude=["vendor/", "*.lock"])
    assert custom.prunes("vendor") and custom.prunes("x/vendor") and not custom.prunes("x.lock2")


def test_empty_issue_query_returns_nothing_without_any_request():
    f = FakeIssues()
    with respx.mock(assert_all_called=False) as router:
        f.mount(router)
        r = GitHubIssuesRetriever(repo=REPO, token="t")
        assert r.invoke("   ") == []
    assert f.calls == []
