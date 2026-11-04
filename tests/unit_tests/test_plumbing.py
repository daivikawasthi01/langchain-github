import asyncio
import copy
import gc
import pickle

import pytest
import respx

from langchain_github import (
    GitHubClient,
    GitHubIssuesRetriever,
    GitHubRepoRetriever,
    GitHubRepositoryLoader,
)

from .fake_issues import REPO, FakeIssues
from .test_retriever import build


def paths(docs):
    return [(d.metadata["path"], d.metadata["start_line"]) for d in docs]


async def test_concurrent_first_queries_build_the_index_once():
    repo, router = build()
    with router:
        r = GitHubRepoRetriever(repo="acme/demo")
        results = await asyncio.gather(*(r.ainvoke("retry limit") for _ in range(6)))
    assert repo.count("/git/blobs/") == 5  # each file fetched once
    assert all(paths(x) == paths(results[0]) for x in results)


def test_sync_and_async_calls_share_one_index():
    repo, router = build()
    with router:
        r = GitHubRepoRetriever(repo="acme/demo")
        a = r.invoke("retry limit")
        b = asyncio.run(r.ainvoke("retry limit"))
    assert paths(a) == paths(b) and repo.count("/git/blobs/") == 5  # each file fetched once


def test_a_retriever_can_be_pickled_and_copied_and_still_works():
    _, router = build()
    with router:
        r = GitHubRepoRetriever(repo="acme/demo", k=2)
        before = paths(r.invoke("retry limit"))  # now it holds a lock, a client and an index
        for clone in (pickle.loads(pickle.dumps(r)), copy.deepcopy(r)):
            assert clone.k == 2 and clone.repo == "acme/demo"
            assert paths(clone.invoke("retry limit")) == before
    issues = GitHubIssuesRetriever(repo=REPO, token="t", k=3)
    assert pickle.loads(pickle.dumps(issues)).k == 3
    assert copy.deepcopy(issues).repo == REPO


def test_context_manager_closes_connections_and_the_retriever_reconnects():
    _, router = build()
    with router:
        with GitHubRepoRetriever(repo="acme/demo") as r:
            r.invoke("retry limit")
            http = r.loader.client._http
        assert http.is_closed
        assert r.invoke("pool acquire")  # a fresh connection is made on demand


def test_issues_retriever_close_resets_its_client():
    f = FakeIssues()
    with respx.mock(assert_all_called=False) as router:
        f.mount(router)
        with GitHubIssuesRetriever(repo=REPO, token="t") as r:
            r.invoke("pool exhausted")
            http = r.client._http
        assert http.is_closed
        assert r.invoke("pool exhausted")


def test_loader_closes_only_the_client_it_made():
    _, router = build()
    mine = GitHubClient("t")
    with router:
        with GitHubRepositoryLoader("acme/demo", client=mine) as loader:
            loader.load()
        assert not mine._http.is_closed  # the caller's client is theirs to close
        with GitHubRepositoryLoader("acme/demo", token="t") as own:
            own.load()
            http = own.client._http
        assert http.is_closed


def test_an_unclosed_client_is_closed_when_collected():
    http = GitHubClient()._http
    gc.collect()
    assert http.is_closed


def test_check_every_skips_the_branch_check_between_queries():
    repo, router = build()
    with router:
        r = GitHubRepoRetriever(repo="acme/demo", check_every=3600)
        r.invoke("retry limit")
        seen = len(repo.calls)
        repo.push(README_dot_md="# Acme v2\n")
        r.invoke("retry limit")
        assert len(repo.calls) == seen  # no request at all
        r.check_every = 0
        r.invoke("retry limit")
        assert len(repo.calls) > seen  # now it looks, and finds the new commit
    assert r.indexed_commit == repo.sha


def test_loader_options_is_gone_and_explicit_options_replace_it():
    with pytest.raises(ValueError, match="loader_options"):
        GitHubRepoRetriever(repo="a/b", loader_options={"split": False})
    r = GitHubRepoRetriever(repo="a/b", max_archive_mb=5, blob_fetch_threshold=1)
    assert r.loader.max_archive_bytes == 5_000_000 and r.loader.blob_fetch_threshold == 1
