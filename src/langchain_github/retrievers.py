"""Query-driven retrievers for GitHub."""

from __future__ import annotations

import asyncio
import copy
import threading
import time
import warnings
from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr, field_validator
from typing_extensions import Self

from ._client import DEFAULT_BASE_URL, GitHubClient
from ._index import ChunkIndex, ThreadIndex, embed_text, issue_embed_text
from ._query import normalize, search_variants
from ._repo import normalize_repo
from ._threads import (
    Thread,
    count_threads,
    fetch_all_issues,
    fetch_changed_issues,
    fetch_discussions,
    fetch_threads,
    probe,
    search_candidates,
    thread_documents,
)
from ._vectors import VectorCache
from .loaders import GitHubRepositoryLoader, RepoSnapshot


def _resolve_k(k: int | None, default: int) -> int:
    if k is None:
        return default
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return k


class _Retriever(BaseRetriever):
    """Behaviour both retrievers share: closing, pickling and copying.

    Clients, locks and indexes are rebuilt on demand, so a retriever can be pickled (for
    multiprocessing) or deep-copied, and the copy simply re-fetches what it needs.
    """

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _close(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Close the HTTP connections. The retriever can still be used; it reconnects."""
        with self._lock:
            self._close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __getstate__(self) -> dict[Any, Any]:
        state = super().__getstate__()
        state["__pydantic_private__"] = None  # clients, locks and indexes are rebuilt lazily
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        super().__setstate__(state)
        self.model_post_init(None)  # recreate the private attributes with their defaults

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        new = type(self).__new__(type(self))
        new.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return new


class GitHubRepoRetriever(_Retriever):
    """Answer questions about a repository's code and docs with line-anchored passages.

    The first query indexes the repository (one tarball request for a typical repo). Every query
    then revalidates the ref with an ETag, which costs no rate limit, and re-indexes only if the
    branch moved, downloading just the files that changed. Results are ranked with a code-aware
    BM25, or fused with embedding similarity when ``embeddings`` is given.

    Each result's ``metadata["url"]`` is a permalink to the exact commit and lines.

    Example:
        >>> retriever = GitHubRepoRetriever(repo="pallets/click", k=5)
        >>> retriever.invoke("how are environment variables read for options?")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    repo: str
    """``"owner/name"``. A GitHub URL or ``owner/name.git`` is also accepted."""
    ref: str | None = None
    """Branch, tag or commit. Defaults to the repository's default branch."""
    k: int = Field(default=5, ge=1)
    """Number of passages to return."""
    token: SecretStr | None = None
    """GitHub token; falls back to ``GITHUB_TOKEN``, ``GH_TOKEN``, then ``gh auth login``."""
    include: list[str] | None = None
    exclude: list[str] | None = None
    extensions: list[str] | None = None
    max_file_size: int = 300_000
    chunk_size: int = 1500
    chunk_overlap: int = 150
    embeddings: Embeddings | None = None
    """Optional embedding model; enables hybrid (BM25 + semantic) ranking."""
    cache_dir: str | None = None
    """Persist downloaded files between runs, keyed by git blob SHA."""
    max_archive_mb: int = 200
    """Refuse repository downloads larger than this."""
    blob_fetch_threshold: int = 25
    """Above this many uncached files, download one tarball instead of file by file."""
    check_every: float = Field(default=0.0, ge=0)
    """Seconds to go without asking GitHub whether the branch moved. 0 checks on every query
    (about 0.8 s, and free of rate limit when nothing changed)."""
    base_url: str = DEFAULT_BASE_URL

    _loader: GitHubRepositoryLoader | None = PrivateAttr(default=None)
    _loader_key: tuple[Any, ...] | None = PrivateAttr(default=None)
    _index: ChunkIndex | None = PrivateAttr(default=None)
    _indexed_sha: str | None = PrivateAttr(default=None)
    _checked_at: float = PrivateAttr(default=0.0)
    _vectors: VectorCache | None = PrivateAttr(default=None)

    @field_validator("repo")
    @classmethod
    def _check_repo(cls, value: str) -> str:
        return normalize_repo(value)

    def _close(self) -> None:
        if self._loader is not None:
            self._loader.close()

    def _config_key(self) -> tuple[Any, ...]:
        """Everything the loader is built from, so changing an option after use takes effect."""
        return (
            self.repo,
            self.ref,
            self.token.get_secret_value() if self.token else None,
            repr(self.include),
            repr(self.exclude),
            repr(self.extensions),
            self.max_file_size,
            self.chunk_size,
            self.chunk_overlap,
            self.cache_dir,
            self.base_url,
            self.max_archive_mb,
            self.blob_fetch_threshold,
        )

    @property
    def loader(self) -> GitHubRepositoryLoader:
        key = self._config_key()
        if self._loader is not None and key != self._loader_key:
            self._loader, self._index, self._indexed_sha = None, None, None
            self._checked_at = 0.0
        if self._loader is None:
            self._loader_key = key
            self._loader = GitHubRepositoryLoader(
                self.repo,
                ref=self.ref,
                token=self.token,
                include=self.include,
                exclude=self.exclude,
                extensions=self.extensions,
                max_file_size=self.max_file_size,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                cache_dir=self.cache_dir,
                base_url=self.base_url,
                max_archive_mb=self.max_archive_mb,
                blob_fetch_threshold=self.blob_fetch_threshold,
            )
        return self._loader

    @property
    def indexed_commit(self) -> str | None:
        """The commit SHA the current index was built from."""
        return self._indexed_sha

    # -- index management -------------------------------------------------------------------------

    def _vector_cache(self) -> VectorCache:
        """Vectors by text; kept in ``cache_dir`` too, so a restart does not re-embed."""
        assert self.embeddings is not None
        cache = self._vectors
        if cache is None or cache.embeddings is not self.embeddings:
            cache = self._vectors = VectorCache(self.embeddings, self.cache_dir)
        return cache

    def _sync_index(self, snap: RepoSnapshot) -> None:
        if snap.commit_sha == self._indexed_sha:
            return
        self.loader.fetch(snap)
        docs = list(self.loader.documents(snap))
        vectors = None
        if self.embeddings is not None:
            vectors = self._vector_cache().vectors([embed_text(d) for d in docs])
        self._index, self._indexed_sha = ChunkIndex(docs, vectors), snap.commit_sha

    # -- BaseRetriever API ------------------------------------------------------------------------

    def _run(self, query: str, k: int | None) -> list[Document]:
        k = _resolve_k(k, self.k)
        qvec = self.embeddings.embed_query(query) if self.embeddings else None
        with self._lock:  # one index build at a time, however many threads or tasks ask
            loader = self.loader  # notices options changed since the last query
            now = time.monotonic()
            if self._index is None or now - self._checked_at >= self.check_every:
                self._sync_index(loader.snapshot())
                self._checked_at = now
            assert self._index is not None
            return self._index.search(query, k, qvec)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, k: int | None = None
    ) -> list[Document]:
        return self._run(query, k)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, k: int | None = None
    ) -> list[Document]:
        return await asyncio.to_thread(self._run, query, k)


class GitHubIssuesRetriever(_Retriever):
    """Find the issue, discussion or comment that answers a problem, with how it was resolved.

    Ask in your own words or paste a traceback. Results are the best-matching passage of each
    thread, linked to the exact comment, with ``resolution`` (``fixed``, ``answered``, ``wontfix``,
    ``duplicate``, ``open``) and ``resolved_by`` (merged PR numbers). A thread closed as a
    duplicate is replaced by the thread it points to.

    Two strategies, chosen by ``mode``:

    * ``"index"``: for repositories with at most ``max_index_threads`` issues. All issues,
      comments, closing PRs and discussions are fetched in bulk (a few requests per hundred
      threads when signed in) and ranked locally. Each query makes one cheap check for changes
      and fetches only the issues that changed since, embedding only their new text.
    * ``"search"``: for large repositories. GitHub's search (hybrid plus keyword variants)
      proposes up to ``max_candidates`` threads, which are fetched in full. Their *order* stays
      GitHub's (re-ranking did not beat it in our eval); what you gain is the best passage in
      the thread, a link to that comment, the resolution and the closing PRs.

    ``"auto"`` (default) counts the repository's issues once and picks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    repo: str
    """``"owner/name"``. A GitHub URL or ``owner/name.git`` is also accepted."""
    k: int = Field(default=5, ge=1)
    token: SecretStr | None = None
    mode: Literal["auto", "index", "search"] = "auto"
    max_index_threads: int = 1500
    max_candidates: int = 15
    include_discussions: bool = True
    use_github_search: bool = True
    """Fuse GitHub's own hybrid search order into the ranking (one extra search call per query)."""
    embeddings: Embeddings | None = None
    max_age: float = 3600.0
    """Seconds after which discussions (which the change check cannot see) are re-checked."""
    rebuild_after: float = 86400.0
    """Seconds after which the whole index is rebuilt, which also drops deleted issues."""
    cache_dir: str | None = None
    """Persist embedding vectors here (with ``embeddings``) so a restart does not re-embed."""
    base_url: str = DEFAULT_BASE_URL

    _client: GitHubClient | None = PrivateAttr(default=None)
    _index: ThreadIndex | None = PrivateAttr(default=None)
    _mode: str | None = PrivateAttr(default=None)
    _stamp: str = PrivateAttr(default="")
    _checked_at: float = PrivateAttr(default=0.0)
    _full_at: float = PrivateAttr(default=0.0)
    _threads: dict[tuple[str, int], Thread] = PrivateAttr(default_factory=dict)
    _vectors: VectorCache | None = PrivateAttr(default=None)

    @field_validator("repo")
    @classmethod
    def _check_repo(cls, value: str) -> str:
        return normalize_repo(value)

    def _close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def client(self) -> GitHubClient:
        if self._client is None:
            self._client = GitHubClient(self.token, base_url=self.base_url)
        return self._client

    @property
    def active_mode(self) -> str | None:
        """``"index"`` or ``"search"`` once the first query has decided."""
        return self._mode

    def _decide(self) -> str:
        if self._mode is None:
            if self.mode != "auto":
                self._mode = self.mode
            else:
                n = count_threads(self.client, self.repo)
                self._mode = "index" if n <= self.max_index_threads else "search"
        return self._mode

    def _embed(self, texts: list[str]) -> list[list[float]]:
        assert self.embeddings is not None
        if self._vectors is None or self._vectors.embeddings is not self.embeddings:
            self._vectors = VectorCache(self.embeddings, self.cache_dir)
        return self._vectors.vectors(texts)

    def _build(self, threads: list[Thread]) -> ThreadIndex:
        docs = [d for t in threads for d in thread_documents(t, self.repo)]
        vectors = None
        if self.embeddings is not None and docs:
            vectors = self._embed([issue_embed_text(d) for d in docs])
        return ThreadIndex(docs, vectors)

    def _refresh_index(self) -> None:
        """Keep the index current at the lowest cost.

        One request asks whether anything changed. If so, only the changed issues are fetched and
        only their new text is embedded. Everything is rebuilt after ``rebuild_after`` seconds, or
        when too many issues changed for a patch to be cheaper.
        """
        stamp = probe(self.client, self.repo)
        now = time.monotonic()
        if self._index is not None:
            due = now - self._checked_at >= self.max_age
            if stamp == self._stamp and not due:
                return
            if now - self._full_at < self.rebuild_after and self._patch(stamp):
                self._stamp, self._checked_at = stamp, now
                return
        self._rebuild(stamp, now)

    def _rebuild(self, stamp: str, now: float) -> None:
        threads = fetch_all_issues(self.client, self.repo, max_threads=self.max_index_threads)
        if self.include_discussions:
            threads.extend(fetch_discussions(self.client, self.repo))
        if not threads:
            warnings.warn(
                f"{self.repo} has no issues or discussions to search (issues may be turned off "
                "for this repository).",
                stacklevel=3,
            )
        self._threads = {(t.kind, t.number): t for t in threads}
        self._index = self._build(threads)
        self._stamp, self._checked_at, self._full_at = stamp, now, now

    def _patch(self, stamp: str) -> bool:
        """Apply changes since the last check. ``False`` means a full rebuild is needed."""
        changed: list[Thread] = []
        if stamp != self._stamp:
            if not self._stamp:
                return False
            issues = fetch_changed_issues(self.client, self.repo, self._stamp)
            if issues is None:
                return False
            changed.extend(issues)
        if self.include_discussions and self.client.core.authenticated:
            since = max(
                (t.updated_at for t in self._threads.values() if t.kind == "discussion"), default=""
            )
            changed.extend(fetch_discussions(self.client, self.repo, since=since or None))
        if changed:
            for t in changed:
                self._threads[(t.kind, t.number)] = t
            self._index = self._build(list(self._threads.values()))
        return True

    def _search_mode(self, query: str, k: int) -> list[Document]:
        numbers = search_candidates(self.client, self.repo, search_variants(query))
        prior = [("issue", n) for n in numbers]
        threads = fetch_threads(self.client, self.repo, numbers[: self.max_candidates])
        # Search order is a ranking too: keep it by listing candidates first-seen first.
        docs = [d for t in threads for d in thread_documents(t, self.repo)]
        if not docs:
            return []
        vectors = None
        qvec = None
        if self.embeddings is not None:
            vectors = self._embed([issue_embed_text(d) for d in docs])
            qvec = self.embeddings.embed_query(query)
        return ThreadIndex(docs, vectors).search(normalize(query), k, qvec, prior, prior_only=True)

    def _run(self, query: str, k: int | None) -> list[Document]:
        with self._lock:
            k = _resolve_k(k, self.k)
            if self._decide() == "search":
                return self._search_mode(query, k)
            self._refresh_index()
            assert self._index is not None
            qvec = self.embeddings.embed_query(query) if self.embeddings else None
            prior = None
            if self.use_github_search:
                found = search_candidates(
                    self.client, self.repo, search_variants(query)[:1], per_query=10
                )
                prior = [("issue", n) for n in found]
            return self._index.search(normalize(query), k, qvec, prior)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, k: int | None = None
    ) -> list[Document]:
        return self._run(query, k)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, k: int | None = None
    ) -> list[Document]:
        return await asyncio.to_thread(self._run, query, k)
