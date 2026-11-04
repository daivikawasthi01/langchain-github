import pytest
import respx
from langchain_core.documents import Document

from langchain_github import GitHubIssuesLoader, GitHubIssuesRetriever
from langchain_github._client import _Core
from langchain_github._index import ThreadIndex
from langchain_github._query import keywords, normalize, search_variants
from langchain_github._threads import (
    build_discussion_thread,
    build_issue_thread,
    closing_references,
    duplicate_target,
    thread_documents,
)

from .fake_issues import HTML, REPO, FakeIssues, comment, issue  # noqa: E402

TRACE = """Traceback (most recent call last):
  File "/home/u/app/main.py", line 10, in <module>
    run()
  File "/usr/lib/python3.12/site-packages/acme/pool.py", line 88, in acquire
    raise PoolExhaustedError("no free connections at 0x7f3a")
acme.errors.PoolExhaustedError: no free connections at 0x7f3a
"""


@pytest.fixture(params=["graphql", "rest"])
def fake(request, monkeypatch):
    """Every test using this runs twice: with the GraphQL fetch path and with REST only."""
    monkeypatch.setattr(_Core, "graphql_issues", request.param == "graphql")
    f = FakeIssues()
    f.path = request.param
    with respx.mock(assert_all_called=False) as router:
        f.mount(router)
        yield f


# -- query handling ---------------------------------------------------------------------------


def test_traceback_collapses_to_exception_and_frames():
    out = normalize(TRACE)
    assert "PoolExhaustedError" in out and "acquire" in out
    assert "0x7f3a" not in out and "/home/u" not in out and "<module>" not in out


def test_plain_question_is_left_alone():
    assert normalize("  why   does it hang? ") == "why does it hang?"


def test_variants_start_with_hybrid_and_shrink():
    v = search_variants("requests hang when many threads exhaust the connection pool")
    assert v[0][0] == "hybrid"
    assert all(t == "lexical" for t, _ in v[1:])
    assert len(v[1][1].split()) > len(v[-1][1].split())
    assert "the" not in keywords("the pool and the threads")


def test_closing_and_duplicate_parsing():
    assert closing_references("Fixes #12, closes acme/demo#7 and resolved #3") == [12, 7, 3]
    assert closing_references("see #4") == []
    assert duplicate_target("Closing: duplicate of #8") == 8
    assert duplicate_target("nothing") is None


# -- thread model -----------------------------------------------------------------------------


def test_resolution_states():
    def mk(**kw):
        return build_issue_thread(issue(1, "t", "b", **kw), [], kw.pop("prs", []))

    assert build_issue_thread(issue(1, "t", "b"), [], []).resolution == "open"
    assert (
        build_issue_thread(issue(1, "t", "b", "closed", "completed"), [], [7]).resolution == "fixed"
    )
    assert (
        build_issue_thread(issue(1, "t", "b", "closed", "completed"), [], []).resolution
        == "answered"
    )
    assert (
        build_issue_thread(issue(1, "t", "b", "closed", "not_planned"), [], []).resolution
        == "wontfix"
    )
    dup = build_issue_thread(
        issue(2, "t", "b", "closed", "duplicate"), [comment(1, 2, "Duplicate of #1.")], []
    )
    assert dup.resolution == "duplicate" and dup.duplicate_of == 1


def test_documents_carry_thread_metadata_and_comment_links():
    t = build_issue_thread(
        issue(1, "Pool exhausted", "body", "closed", "completed"),
        [comment(11, 1, "use finally")],
        [9],
    )
    docs = thread_documents(t, REPO)
    assert [d.metadata["part"] for d in docs] == ["body", "comment"]
    assert docs[1].metadata["url"].endswith("#issuecomment-11")
    assert docs[1].metadata["resolved_by_urls"] == [f"{HTML}/pull/9"]
    assert docs[1].metadata["resolution"] == "fixed"


def test_discussion_answer_is_marked():
    t = build_discussion_thread(
        {
            "number": 5,
            "title": "q",
            "body": "b",
            "url": "u",
            "closed": False,
            "answer": {"id": "A", "url": "ua", "body": "the answer"},
            "comments": {"nodes": []},
        }
    )
    assert t.resolution == "answered"
    assert [d.metadata["part"] for d in thread_documents(t, REPO)] == ["body", "answer"]


def test_long_comment_is_split_into_passages():
    t = build_issue_thread(issue(1, "t", "b"), [comment(1, 1, ("word " * 100 + "\n\n") * 10)], [])
    assert len([d for d in thread_documents(t, REPO) if d.metadata["part"] == "comment"]) > 1


# -- ranking ----------------------------------------------------------------------------------


def _docs():
    out = []
    for t in (
        build_issue_thread(
            issue(1, "Connection pool exhausted", "acquire raises", "closed", "completed"),
            [comment(1, 1, "release sockets in finally")],
            [9],
        ),
        build_issue_thread(issue(2, "Stream timeout ignored", "download never ends"), [], []),
        build_issue_thread(
            issue(3, "Pool exhausted again", "same", "closed", "duplicate"),
            [comment(3, 3, "Duplicate of #1.")],
            [],
        ),
    ):
        out.extend(thread_documents(t, REPO))
    return out


def test_best_passage_per_thread_and_duplicates_resolve_to_canonical():
    idx = ThreadIndex(_docs())
    res = idx.search("pool exhausted", 5)
    assert [d.metadata["number"] for d in res] == [1]  # thread 3 collapsed into its canonical
    assert res[0].metadata["matched_duplicate"] == 3
    assert [d.metadata["number"] for d in idx.search("stream timeout", 5)] == [2]


def test_no_match_returns_nothing():
    assert ThreadIndex(_docs()).search("kubernetes", 5) == []
    assert ThreadIndex([]).search("pool", 5) == []


def test_prior_only_keeps_the_outside_order():
    idx = ThreadIndex(_docs())
    res = idx.search("pool exhausted", 5, prior=[("issue", 2), ("issue", 1)], prior_only=True)
    assert [d.metadata["number"] for d in res][:2] == [2, 1]


def test_embeddings_signal_is_fused():
    docs = _docs()
    vecs = [[1.0, 0.0] if d.metadata["number"] == 2 else [0.0, 1.0] for d in docs]
    res = ThreadIndex(docs, vecs).search("pool exhausted", 3, query_vector=[1.0, 0.0])
    assert 2 in [d.metadata["number"] for d in res]


# -- retriever --------------------------------------------------------------------------------


def make(**kw):
    return GitHubIssuesRetriever(repo=REPO, token="t", **kw)


def test_auto_picks_index_for_small_repo_and_answers_with_resolution(fake):
    r = make(k=3)
    docs = r.invoke("connection pool exhausted when threads pile up")
    assert r.active_mode == "index"
    top = docs[0]
    assert top.metadata["number"] == 1
    assert top.metadata["resolution"] == "fixed"
    assert top.metadata["resolved_by"] == [9]
    assert "pull_request" not in top.metadata
    assert all(d.metadata["number"] != 9 for d in docs)  # PRs are not issues


def test_index_mode_finds_discussions_and_answers(fake):
    docs = make().invoke("where is the retry count set?")
    d = next(d for d in docs if d.metadata["kind"] == "discussion")
    assert d.metadata["resolution"] == "answered"


def test_duplicate_is_replaced_by_canonical(fake):
    docs = make(k=5).invoke("pool exhausted again")
    numbers = [d.metadata["number"] for d in docs if d.metadata["kind"] == "issue"]
    assert 3 not in numbers and numbers[0] == 1
    assert docs[0].metadata["matched_duplicate"] == 3


def test_wontfix_resolution(fake):
    docs = make().invoke("add dark mode")
    assert docs[0].metadata["resolution"] == "wontfix"


def test_unchanged_repo_is_not_refetched(fake):
    r = make()
    r.invoke("pool exhausted")
    fetched = fake.calls.count(f"/repos/{REPO}/issues/comments")
    r.invoke("stream timeout")
    assert fake.calls.count(f"/repos/{REPO}/issues/comments") == fetched
    assert fake.etag_hits == 1


def test_changed_repo_is_refetched(fake):
    r = make()
    r.invoke("pool exhausted")
    fake.updated = "2026-02-02T00:00:00Z"
    fake.issues.append(issue(7, "Kubernetes probes fail", "liveness probe times out"))
    docs = r.invoke("kubernetes liveness probe")
    assert docs[0].metadata["number"] == 7


def test_github_search_order_is_fused_in_index_mode(fake):
    fake.search_hits = [2]
    r = make(k=3)
    r.invoke("pool exhausted")
    assert "search:hybrid" in fake.calls
    off = make(use_github_search=False)
    fake.calls.clear()
    off.invoke("pool exhausted")
    assert "search:hybrid" not in fake.calls


def test_search_mode_keeps_github_order_and_adds_metadata(fake):
    fake.search_hits = [2, 1]
    r = make(mode="search", k=3)
    docs = r.invoke("something about connections")
    assert [d.metadata["number"] for d in docs][:2] == [2, 1]
    assert docs[1].metadata["resolved_by"] == [9]  # from the timeline cross-reference
    assert r.active_mode == "search"


def test_large_repo_switches_to_search_mode(fake):
    fake.search_hits = [1]
    r = make(max_index_threads=1)
    assert r.invoke("pool")[0].metadata["number"] == 1
    assert r.active_mode == "search"


def test_search_mode_with_no_hits_returns_empty(fake):
    fake.search_hits = []
    assert make(mode="search").invoke("zzz") == []


async def test_async_invoke(fake):
    docs = await make().ainvoke("connection pool exhausted")
    assert docs[0].metadata["number"] == 1


def test_k_call_kwarg(fake):
    assert len(make().invoke("pool exhausted timeout dark", k=1)) == 1


# -- loader -----------------------------------------------------------------------------------


def test_loader_yields_issue_and_discussion_documents(fake):
    docs = GitHubIssuesLoader(REPO, token="t").load()
    kinds = {d.metadata["kind"] for d in docs}
    assert kinds == {"issue", "discussion"}
    assert all(isinstance(d, Document) and d.metadata["url"] for d in docs)
    assert not any(d.metadata["number"] == 9 for d in docs if d.metadata["kind"] == "issue")


def test_loader_can_skip_discussions(fake):
    docs = GitHubIssuesLoader(REPO, token="t", include_discussions=False).load()
    assert {d.metadata["kind"] for d in docs} == {"issue"}


def test_loader_validates_arguments():
    with pytest.raises(ValueError):
        GitHubIssuesLoader("nonsense")
    with pytest.raises(ValueError):
        GitHubIssuesLoader(REPO, state="weird")


async def test_loader_async(fake):
    docs = [d async for d in GitHubIssuesLoader(REPO, token="t").alazy_load()]
    assert docs


def _discussion_client(signed_in, graphql):
    from types import SimpleNamespace

    return SimpleNamespace(core=SimpleNamespace(authenticated=signed_in), graphql=graphql)


def test_signed_out_user_is_told_discussions_were_skipped_without_calling_graphql():
    from langchain_github._threads import fetch_discussions

    calls = []
    client = _discussion_client(False, lambda *a, **k: calls.append(1))
    with pytest.warns(UserWarning, match="gh auth login"):
        assert fetch_discussions(client, "o/r") == []
    assert calls == []


def test_graphql_refusal_is_a_warning_but_rate_limits_and_bad_credentials_are_raised():
    from langchain_github._client import GitHubAuthError, GitHubGraphQLError, GitHubRateLimitError
    from langchain_github._threads import fetch_discussions

    def refuse(*a, **k):
        raise GitHubGraphQLError("Could not resolve to a Repository")

    with pytest.warns(UserWarning, match="Could not resolve"):
        assert fetch_discussions(_discussion_client(True, refuse), "o/r") == []

    for exc in (GitHubRateLimitError("slow down", 500), GitHubAuthError("bad token")):

        def boom(*a, exc=exc, **k):
            raise exc

        with pytest.raises(type(exc)):
            fetch_discussions(_discussion_client(True, boom), "o/r")


def test_references_to_other_repositories_are_not_mistaken_for_this_one():
    assert closing_references("fixes psf/requests#12", "acme/demo") == []
    assert closing_references("fixes ACME/Demo#12", "acme/demo") == [12]
    assert closing_references("Fixes #12 and closes other/repo#9", "acme/demo") == [12]
    assert duplicate_target("Duplicate of other/repo#5", "acme/demo") is None
    assert duplicate_target("duplicate of other/repo#5. Duplicate of #6", "acme/demo") == 6


def test_full_issue_urls_are_recognised():
    url = "Closes https://github.com/acme/demo/issues/12"
    assert closing_references(url, "acme/demo") == [12]
    assert closing_references(url, "acme/other") == []
    assert duplicate_target("Duplicate of https://github.com/acme/demo/issues/3", "acme/demo") == 3


def test_prose_that_mentions_a_number_is_not_a_closing_reference():
    for text in ("fix: handle #12 edge case", "prefix #12", "see #12", "unresolved #12"):
        assert closing_references(text) == [], text
