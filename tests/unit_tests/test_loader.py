import pytest
import respx

from langchain_github.loaders import GitHubRepositoryLoader

from .fake_github import FakeRepo

FILES = {
    "README.md": "# Demo\n\nA demo project.\n",
    "src/app.py": "def handler(event):\n    return retry(event)\n\n\ndef retry(e):\n    return e\n",
    "docs/guide.md": "# Guide\n\nConfigure retries with RETRY_LIMIT.\n",
    "node_modules/dep/index.js": "module.exports = 1\n",
    "assets/logo.png": "not really a png",
}


def make(files=None, **kw):
    repo = FakeRepo("acme/demo", files or FILES, **kw)
    router = respx.mock(assert_all_called=False)
    repo.mount(router)
    return repo, router


def test_loads_filtered_files_with_permalinks_at_exact_commit():
    repo, router = make()
    with router:
        docs = list(GitHubRepositoryLoader("acme/demo", token="t").lazy_load())
    paths = sorted({d.metadata["path"] for d in docs})
    assert paths == ["README.md", "docs/guide.md", "src/app.py"]  # node_modules + png dropped
    d = next(d for d in docs if d.metadata["path"] == "src/app.py")
    assert d.metadata["commit_sha"] == repo.sha and d.metadata["ref"] == "main"
    assert d.metadata["url"] == f"https://github.com/acme/demo/blob/{repo.sha}/src/app.py#L1-L6"
    assert d.metadata["source"] == d.metadata["url"]


def test_include_and_extension_filters():
    _, router = make()
    with router:
        docs = list(GitHubRepositoryLoader("acme/demo", include=["docs/**"]).lazy_load())
        assert {d.metadata["path"] for d in docs} == {"docs/guide.md"}


def test_small_loads_use_blob_api_and_large_loads_use_one_tarball():
    repo, router = make()
    with router:
        list(GitHubRepositoryLoader("acme/demo", blob_fetch_threshold=25).lazy_load())
    assert repo.count("/git/blobs/") == 3 and repo.count("/tarball/") == 0

    big, router = make({f"pkg/m{i}.py": f"x = {i}\n" for i in range(40)})
    with router:
        docs = list(GitHubRepositoryLoader("acme/demo", blob_fetch_threshold=25).lazy_load())
    assert len(docs) == 40 and big.count("/tarball/") == 1 and big.count("/git/blobs/") == 0


def test_incremental_reload_only_fetches_changed_files(tmp_path):
    repo, router = make({f"pkg/m{i}.py": f"x = {i}\n" for i in range(40)})
    loader = GitHubRepositoryLoader("acme/demo", cache_dir=tmp_path, blob_fetch_threshold=5)
    with router:
        list(loader.lazy_load())
        assert repo.count("/tarball/") == 1

        list(loader.lazy_load())  # nothing changed: ref + tree are 304/cheap, no downloads
        assert repo.count("/tarball/") == 1 and repo.count("/git/blobs/") == 0
        assert any(c.startswith("304") and "/commits/" in c for c in repo.calls)

        repo.push(pkg__m3_dot_py="x = 'changed'\n")
        docs = list(loader.lazy_load())  # one file changed -> one blob, not a new tarball
    assert repo.count("/tarball/") == 1 and repo.count("/git/blobs/") == 1
    assert any("changed" in d.page_content for d in docs)
    assert all(d.metadata["commit_sha"] == repo.sha for d in docs)


def test_cache_dir_survives_a_new_loader_instance(tmp_path):
    repo, router = make()
    with router:
        list(GitHubRepositoryLoader("acme/demo", cache_dir=tmp_path).lazy_load())
        before = repo.count("/git/blobs/")
        list(GitHubRepositoryLoader("acme/demo", cache_dir=tmp_path).lazy_load())
    assert repo.count("/git/blobs/") == before


def test_commit_sha_ref_skips_commit_lookup():
    repo, router = make()
    with router:
        snap = GitHubRepositoryLoader("acme/demo", ref=repo.sha).snapshot()
    assert snap.commit_sha == repo.sha
    assert not any("/commits/" in c for c in repo.calls)


def test_truncated_tree_is_walked_level_by_level():
    repo, router = make(truncated=True)
    with router:
        snap = GitHubRepositoryLoader("acme/demo").snapshot()
    assert {e.path for e in snap.entries} == {
        "all/README.md",
        "all/src/app.py",
        "all/docs/guide.md",
    }


def test_split_false_returns_one_document_per_file():
    _, router = make()
    with router:
        docs = list(GitHubRepositoryLoader("acme/demo", split=False).lazy_load())
    assert len(docs) == 3 and all(d.metadata["kind"] == "file" for d in docs)


@pytest.mark.parametrize("bad", ["nope", "a/b/c", "", "owner/"])
def test_rejects_malformed_repo(bad):
    with pytest.raises(ValueError):
        GitHubRepositoryLoader(bad)


async def test_async_loader_matches_sync_output():
    repo, router = make({f"pkg/m{i}.py": f"x = {i}\n" for i in range(40)})
    loader = GitHubRepositoryLoader("acme/demo", blob_fetch_threshold=5)
    with router:
        sync_docs = list(loader.lazy_load())
        aloader = GitHubRepositoryLoader("acme/demo", blob_fetch_threshold=100)  # blob path
        async_docs = [d async for d in aloader.alazy_load()]
    key = lambda d: (d.metadata["path"], d.metadata["start_line"])  # noqa: E731
    assert [(key(d), d.page_content) for d in sorted(sync_docs, key=key)] == [
        (key(d), d.page_content) for d in sorted(async_docs, key=key)
    ]
