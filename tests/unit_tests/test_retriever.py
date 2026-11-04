import hashlib

import pytest
import respx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_github._index import ChunkIndex, stem, tokenize
from langchain_github.retrievers import GitHubRepoRetriever

from .fake_github import FakeRepo

FILES = {
    "README.md": "# Acme\n\nA small web framework.\n",
    "docs/config.md": "# Configuration\n\nSet RETRY_LIMIT to control how many times requests are retried.\n",
    "src/http/retry.py": (
        "import time\n\n\ndef retry_request(send, retry_limit=3):\n"
        "    for attempt in range(retry_limit):\n        try:\n            return send()\n"
        "        except TimeoutError:\n            time.sleep(2 ** attempt)\n    raise TimeoutError\n"
    ),
    "src/auth/tokens.py": "def refresh_token(session):\n    return session.post('/oauth/token')\n",
    "src/db/pool.py": "class ConnectionPool:\n    def acquire(self):\n        return self._free.pop()\n",
}


def build(files=None):
    repo = FakeRepo("acme/demo", files or FILES)
    router = respx.mock(assert_all_called=False)
    repo.mount(router)
    return repo, router


def doc(path, text, start=1):
    n = text.count("\n")
    return Document(
        page_content=text,
        metadata={"path": path, "start_line": start, "end_line": start + n, "blob_sha": path},
    )


def test_tokenize_keeps_identifiers_and_their_parts():
    assert {"retrylimit", "retry", "limit"} <= set(tokenize("RETRY_LIMIT"))
    assert {"retrylimit", "retry", "limit"} <= set(tokenize("retryLimit"))
    assert {"httpserver", "http", "server"} <= set(tokenize("HTTPServer"))
    assert "the" not in tokenize("the cat") and "where" not in tokenize("where is it", query=True)


def test_get_and_set_are_dropped_from_queries_but_kept_in_code():
    assert "get" in tokenize("def get(self):") and "get" not in tokenize("get user", query=True)


@pytest.mark.parametrize(
    "word,expected",
    [
        ("retries", "retry"),
        ("configured", "configur"),
        ("classes", "class"),
        ("pass", "pass"),
        ("handling", "handl"),
        ("is", "is"),
        ("sessions", "session"),
    ],
)
def test_stem(word, expected):
    assert stem(word) == expected


def test_bm25_prefers_file_named_for_the_topic_and_matches_identifier_spelling():
    idx = ChunkIndex(
        [
            doc("src/a.py", "def helper():\n    log('retry later')\n"),
            doc("src/http/retry.py", "def send(): pass\n"),
            doc("src/b.py", "x = 1\n"),
        ]
    )
    assert idx.search("retry logic", 2)[0].metadata["path"] == "src/http/retry.py"
    idx2 = ChunkIndex([doc("a.py", "retryLimit = 3\n"), doc("b.py", "other = 1\n")])
    assert idx2.search("what is the retry limit", 1)[0].metadata["path"] == "a.py"


def test_overlapping_chunks_of_the_same_file_are_not_returned_twice():
    a = doc("f.py", "retry retry retry\nretry\nretry\n", start=10)
    b = doc("f.py", "retry retry\nretry\nretry\n", start=11)  # mostly the same lines
    c = doc("f.py", "retry elsewhere\n", start=80)
    out = ChunkIndex([a, b, c]).search("retry", 5)
    assert [d.metadata["start_line"] for d in out] == [10, 80]


def test_zero_score_chunks_are_not_padded_into_results():
    idx = ChunkIndex([doc("a.py", "alpha beta\n"), doc("b.py", "gamma\n")])
    assert idx.search("zzzz", 5) == []


class BagEmbeddings(Embeddings):
    """Deterministic fake: hashes words into a small vector, so shared words => similarity."""

    DIM = 64

    def _vec(self, text):
        v = [0.0] * self.DIM
        for w in tokenize(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        return v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_hybrid_ranking_returns_fused_scores():
    docs = [doc("a.py", "alpha token flow\n"), doc("b.py", "beta stuff\n")]
    vecs = BagEmbeddings().embed_documents([d.page_content for d in docs])
    out = ChunkIndex(docs, vecs).search("alpha", 2, BagEmbeddings().embed_query("alpha"))
    assert out[0].metadata["path"] == "a.py" and out[0].metadata["score"] < 1  # RRF-scale score


def test_retriever_returns_cited_chunks_for_a_natural_language_question():
    repo, router = build()
    with router:
        r = GitHubRepoRetriever(repo="acme/demo", k=3)
        out = r.invoke("how many times are requests retried?")
    assert {d.metadata["path"] for d in out[:2]} == {"docs/config.md", "src/http/retry.py"}
    top = out[0]
    assert top.metadata["url"].startswith(f"https://github.com/acme/demo/blob/{repo.sha}/")
    assert top.metadata["rank"] == 1 and "score" in top.metadata


def test_k_works_as_constructor_arg_and_call_kwarg():
    _, router = build()
    with router:
        r = GitHubRepoRetriever(repo="acme/demo", k=1)
        assert len(r.invoke("retry token pool config")) == 1
        assert len(r.invoke("retry token pool config", k=3)) == 3


def test_index_is_built_once_and_rebuilt_only_when_the_branch_moves():
    repo, router = build()
    with router:
        r = GitHubRepoRetriever(repo="acme/demo")
        r.invoke("retry")
        built = r._index
        fetched = repo.count("/git/blobs/")
        r.invoke("token")
        assert r._index is built and repo.count("/git/blobs/") == fetched  # same commit: no work

        repo.push(src__http__retry_dot_py="def backoff_policy():\n    return 'exponential'\n")
        out = r.invoke("exponential backoff policy")
        assert r._index is not built and r.indexed_commit == repo.sha
        assert repo.count("/git/blobs/") == fetched + 1  # only the changed file was fetched
        assert out[0].metadata["commit_sha"] == repo.sha


def test_retriever_with_embeddings_runs_hybrid_and_caches_vectors():
    repo, router = build()
    emb = BagEmbeddings()
    calls = []
    orig = emb.embed_documents
    emb.embed_documents = lambda t: calls.append(len(t)) or orig(t)
    with router:
        r = GitHubRepoRetriever(repo="acme/demo", embeddings=emb)
        assert r.invoke("refresh oauth token")[0].metadata["path"] == "src/auth/tokens.py"
        first = sum(calls)
        repo.push(README_dot_md="# Acme v2\n")
        r.invoke("refresh oauth token")
    assert sum(calls) - first == 1  # only the one changed chunk was re-embedded


async def test_async_matches_sync():
    repo, router = build()
    with router:
        r = GitHubRepoRetriever(repo="acme/demo", k=3)
        sync = r.invoke("connection pool acquire")
        r2 = GitHubRepoRetriever(repo="acme/demo", k=3)
        aout = await r2.ainvoke("connection pool acquire")
    strip = lambda ds: [(d.metadata["path"], d.metadata["start_line"], d.page_content) for d in ds]  # noqa: E731
    assert strip(sync) == strip(aout)


def test_unknown_repo_error_is_clear():
    import httpx

    from langchain_github._client import GitHubNotFoundError

    with respx.mock() as router:
        router.get("https://api.github.com/repos/acme/ghost").mock(
            return_value=httpx.Response(404, json={"message": "Not Found"})
        )
        with pytest.raises(GitHubNotFoundError):
            GitHubRepoRetriever(repo="acme/ghost").invoke("anything")


def test_bm25_idf_stays_positive_for_terms_in_most_documents():
    from langchain_github._index import BM25

    bm = BM25([["retry", "a"], ["retry", "b"], ["retry", "c"]])  # "retry" is in every doc
    assert all(s > 0 for s in bm.get_scores(["retry"]))
    assert bm.get_scores(["missing"]) == [0.0, 0.0, 0.0]
    # a rarer term still outranks a ubiquitous one
    docs = [["retry", "x"], ["retry", "y"], ["retry", "rare"]]
    s = BM25(docs).get_scores(["retry", "rare"])
    assert s[2] == max(s)
