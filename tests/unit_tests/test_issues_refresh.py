"""Keeping the issues index current without rebuilding it."""

import pytest
import respx
from langchain_core.embeddings import Embeddings

from langchain_github import GitHubIssuesRetriever, _threads
from langchain_github._client import _Core

from .fake_issues import REPO, FakeIssues, comment

T0, T1 = "2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"


@pytest.fixture(params=["graphql", "rest"])
def fake(request, monkeypatch):
    monkeypatch.setattr(_Core, "graphql_issues", request.param == "graphql")
    f = FakeIssues()
    f.path = request.param
    f.updated = T0
    with respx.mock(assert_all_called=False) as router:
        f.mount(router)
        yield f


class Counting(Embeddings):
    def __init__(self):
        self.texts: list[str] = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return [[float(len(t) % 7), 1.0] for t in texts]

    def embed_query(self, text):
        return [1.0, 1.0]


def touch_issue_two(fake, text="Pass timeout=30 to http.stream() so a stalled stream is cut off."):
    fake.comments.append(comment(21, 2, text))
    fake.issues[1]["updated_at"] = T1
    fake.updated = T1


def make(**kw):
    return GitHubIssuesRetriever(repo=REPO, token="t", use_github_search=False, **kw)


def test_a_change_fetches_only_the_changed_issue(fake):
    r = make()
    r.invoke("stream timeout")
    touch_issue_two(fake)
    docs = r.invoke("how do I stop a stalled stream with timeout")
    assert docs[0].metadata["number"] == 2 and "timeout=30" in docs[0].page_content
    assert fake.since_log == [T0]  # asked for what changed since the last check
    if fake.path == "graphql":
        assert fake.calls.count("graphql:issues") == 2  # the build, then one small patch
    else:  # REST: the repo-wide comment and PR listings were not repeated
        assert fake.calls.count(f"/repos/{REPO}/issues/comments") == 1
        assert fake.calls.count(f"/repos/{REPO}/pulls") == 1


def test_only_new_text_is_embedded_again(fake):
    emb = Counting()
    r = make(embeddings=emb)
    r.invoke("stream timeout")
    first = len(emb.texts)
    assert first >= 5
    touch_issue_two(fake)
    r.invoke("stream timeout")
    assert len(emb.texts) - first == 1  # the new comment; every unchanged passage is reused


def test_rebuild_after_forces_a_full_rebuild(fake):
    r = make(rebuild_after=0)
    r.invoke("stream timeout")
    touch_issue_two(fake)
    r.invoke("stream timeout")
    assert fake.since_log == []


def test_too_many_changes_fall_back_to_a_rebuild(fake, monkeypatch):
    monkeypatch.setattr(_threads, "MAX_INCREMENTAL", 0)
    r = make()
    r.invoke("stream timeout")
    touch_issue_two(fake)
    docs = r.invoke("stalled stream timeout")
    assert any("timeout=30" in d.page_content for d in docs)


def test_discussion_edits_are_picked_up_when_max_age_passes(fake):
    r = make(max_age=0)
    r.invoke("retry count")
    fake.discussions[0]["answer"]["body"] = "Set MAX_ATTEMPTS in settings.toml."
    fake.discussions[0]["updatedAt"] = "y"
    docs = r.invoke("MAX_ATTEMPTS settings")
    assert any("MAX_ATTEMPTS" in d.page_content for d in docs)


def test_issue_vectors_survive_a_restart_with_cache_dir(fake, tmp_path):
    first = Counting()
    make(embeddings=first, cache_dir=str(tmp_path)).invoke("stream timeout")
    assert len(first.texts) > 5
    second = Counting()
    docs = make(embeddings=second, cache_dir=str(tmp_path)).invoke("stream timeout")
    assert docs
    assert second.texts == ["langchain-github model fingerprint"]  # nothing else re-embedded
