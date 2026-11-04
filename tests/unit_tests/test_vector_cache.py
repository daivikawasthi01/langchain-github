import pytest
from langchain_core.embeddings import Embeddings

from langchain_github import GitHubRepoRetriever
from langchain_github._vectors import _PROBE, VectorCache

from .test_retriever import build


class Model(Embeddings):
    """Deterministic embeddings that remember what they were asked to embed."""

    def __init__(self, salt=1.0, model="toy-model"):
        self.salt, self.model = salt, model
        self.asked: list[str] = []

    def embed_documents(self, texts):
        self.asked.extend(texts)
        return [[float(len(t) % 11) * self.salt, float(sum(map(ord, t)) % 13), 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    @property
    def real(self):
        return [t for t in self.asked if t != _PROBE]


def test_vectors_survive_a_restart(tmp_path):
    first = Model()
    a = VectorCache(first, tmp_path).vectors(["alpha", "beta", "alpha"])
    assert first.real == ["alpha", "beta"]  # the duplicate is embedded once
    second = Model()
    b = VectorCache(second, tmp_path).vectors(["beta", "alpha", "gamma"])
    assert second.real == ["gamma"]  # only the new text
    assert b[0] == pytest.approx(a[1]) and b[1] == pytest.approx(a[0])


def test_another_model_never_reuses_these_vectors(tmp_path):
    VectorCache(Model(salt=1.0), tmp_path).vectors(["alpha"])
    other = Model(salt=5.0)
    out = VectorCache(other, tmp_path).vectors(["alpha"])
    assert other.real == ["alpha"] and out[0][0] == pytest.approx(5.0 * 5)
    assert len(list((tmp_path / "vectors").iterdir())) == 2


def test_a_torn_final_record_is_ignored(tmp_path):
    VectorCache(Model(), tmp_path).vectors(["alpha", "beta"])
    (path,) = (tmp_path / "vectors").iterdir()
    path.write_bytes(path.read_bytes()[:-7])  # crash mid-write
    again = Model()
    VectorCache(again, tmp_path).vectors(["alpha", "beta"])
    assert again.real == ["beta"]


def test_without_cache_dir_nothing_is_written_and_no_probe_is_sent(tmp_path):
    m = Model()
    VectorCache(m).vectors(["alpha"])
    assert m.asked == ["alpha"]


def test_repo_retriever_does_not_re_embed_after_a_restart(tmp_path):
    _, router = build()
    with router:
        a = Model()
        GitHubRepoRetriever(repo="acme/demo", embeddings=a, cache_dir=str(tmp_path)).invoke("retry")
        assert a.real
        b = Model()
        r = GitHubRepoRetriever(repo="acme/demo", embeddings=b, cache_dir=str(tmp_path))
        assert r.invoke("retry limit")
    assert b.real == ["retry limit"]  # just the query


def test_identical_files_at_different_paths_get_their_own_vectors(tmp_path):
    from langchain_core.documents import Document

    from langchain_github._index import embed_text

    def doc(path):
        return Document(page_content="same body", metadata={"path": path})

    assert embed_text(doc("a/x.py")) != embed_text(doc("b/x.py"))
    m = Model()
    VectorCache(m, tmp_path).vectors([embed_text(doc("a/x.py")), embed_text(doc("b/x.py"))])
    assert len(m.real) == 2
