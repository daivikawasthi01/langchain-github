import pytest
import respx
from pydantic import ValidationError

from langchain_github import GitHubIssuesRetriever, GitHubRepoRetriever
from langchain_github._repo import normalize_repo
from langchain_github._tree import PathFilter, TreeEntry

from .test_retriever import build


@pytest.mark.parametrize(
    "given",
    [
        "pallets/click",
        "https://github.com/pallets/click",
        "http://www.github.com/pallets/click/",
        "github.com/pallets/click",
        "pallets/click.git",
        " pallets/click/ ",
    ],
)
def test_repo_forms_people_paste_are_accepted(given):
    assert normalize_repo(given) == "pallets/click"
    assert GitHubRepoRetriever(repo=given).repo == "pallets/click"
    assert GitHubIssuesRetriever(repo=given).repo == "pallets/click"


@pytest.mark.parametrize("bad", ["click", "", "a/b/c", "https://github.com/pallets"])
def test_bad_repo_fails_at_construction_with_an_example(bad):
    with pytest.raises(ValidationError, match="owner/name"):
        GitHubRepoRetriever(repo=bad)


def test_misspelled_options_are_rejected():
    with pytest.raises(ValidationError, match="extension"):
        GitHubRepoRetriever(repo="a/b", extension=[".md"])
    with pytest.raises(ValidationError, match="top_k"):
        GitHubIssuesRetriever(repo="a/b", top_k=3)


def test_k_must_be_positive_in_constructor_and_per_call():
    with pytest.raises(ValidationError):
        GitHubRepoRetriever(repo="a/b", k=0)
    _, router = build()
    with router:
        r = GitHubRepoRetriever(repo="acme/demo")
        with pytest.raises(ValueError, match="at least 1"):
            r.invoke("retry", k=0)


def _picked(*patterns):
    f = PathFilter(include=list(patterns))
    paths = ["docs/a.md", "docs/deep/b.md", "src/docs.py", "src/x/docs/c.md", "README.md"]
    return [p for p in paths if f.accepts(TreeEntry(p, "s", 1))]


def test_bare_directory_name_selects_everything_under_it():
    everywhere = ["docs/a.md", "docs/deep/b.md", "src/x/docs/c.md"]
    assert _picked("docs") == everywhere
    assert _picked("docs/") == everywhere
    assert _picked("docs/**") == ["docs/a.md", "docs/deep/b.md"]
    assert _picked("src/x") == ["src/x/docs/c.md"]


def test_filters_that_select_nothing_warn():
    _, router = build()
    with router:
        r = GitHubRepoRetriever(repo="acme/demo", extensions=[".zig"])
        with pytest.warns(UserWarning, match="nothing to search"):
            assert r.invoke("retry") == []


def test_changing_an_option_after_first_use_takes_effect():
    _, router = build()
    with router:
        r = GitHubRepoRetriever(repo="acme/demo", k=5)
        assert any(d.metadata["path"] == "src/http/retry.py" for d in r.invoke("retry limit"))
        r.include = ["docs/**"]
        assert {d.metadata["path"] for d in r.invoke("retry limit")} == {"docs/config.md"}


def test_unknown_ref_says_so_instead_of_a_bare_422():
    import httpx

    from langchain_github import GitHubNotFoundError

    repo, router = build()
    router.get("https://api.github.com/repos/acme/demo/commits/nope").mock(
        return_value=httpx.Response(422, json={"message": "No commit found for SHA: nope"})
    )
    with router:
        r = GitHubRepoRetriever(repo="acme/demo", ref="nope", token="t")
        with pytest.raises(GitHubNotFoundError, match="no branch, tag or commit named 'nope'"):
            r.invoke("retry")


def test_issue_search_translates_a_missing_repo_and_surfaces_rate_limits():
    import httpx

    from langchain_github import GitHubClient, GitHubNotFoundError, GitHubRateLimitError
    from langchain_github._threads import count_threads, search_candidates

    with respx.mock() as router:
        router.get("https://api.github.com/search/issues").mock(
            return_value=httpx.Response(422, json={"message": "Validation Failed"})
        )
        client = GitHubClient("t", sleep=lambda s: None)
        with pytest.raises(GitHubNotFoundError, match="not found, or is private"):
            count_threads(client, "a/b")
        assert search_candidates(client, "a/b", [("hybrid", "x"), ("lexical", "y")]) == []

    with respx.mock() as router:
        router.get("https://api.github.com/search/issues").mock(
            return_value=httpx.Response(
                403, headers={"retry-after": "900"}, json={"message": "rate limit"}
            )
        )
        client = GitHubClient("t", sleep=lambda s: None)
        with pytest.raises(GitHubRateLimitError):
            search_candidates(client, "a/b", [("hybrid", "x")])
