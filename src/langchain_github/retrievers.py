"""Query-driven retrievers for GitHub."""

from __future__ import annotations

import asyncio
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

from ._client import DEFAULT_BASE_URL, GitHubClient, GitHubNotFoundError
from ._index import (
    ChunkIndex,
    ThreadIndex,
    aembed_all,
    embed_all,
    embed_all_with,
    issue_embed_text,
)
from ._query import normalize, search_variants
from ._repo import normalize_repo
from ._threads import (
    Thread,
    count_threads,
    fetch_all_issues,
    fetch_discussions,
    fetch_one_issue,
    probe,
    search_candidates,
    thread_documents,
)
from .loaders import GitHubRepositoryLoader, RepoSnapshot


def _resolve_k(k: int | None, default: int) -> int:
    if k is None:
        return default
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return k


class GitHubRepoRetriever(BaseRetriever):
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
    base_url: str = DEFAULT_BASE_URL
    loader_options: dict[str, Any] = Field(default_factory=dict)

    _loader: GitHubRepositoryLoader | None = PrivateAttr(default=None)
    _loader_key: tuple[Any, ...] | None = PrivateAttr(default=None)
    _index: ChunkIndex | None = PrivateAttr(default=None)
    _indexed_sha: str | None = PrivateAttr(default=None)
    _vector_cache: dict[tuple[str, int, int], list[float]] = PrivateAttr(default_factory=dict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @field_validator("repo")
    @classmethod
    def _check_repo(cls, value: str) -> str:
        return normalize_repo(value)

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
            repr(self.loader_options),
        )

    @property
    def loader(self) -> GitHubRepositoryLoader:
        key = self._config_key()
        if self._loader is not None and key != self._loader_key:
            self._loader, self._index, self._indexed_sha = None, None, None
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
                **self.loader_options,
            )
        return self._loader

    @property
    def indexed_commit(self) -> str | None:
        """The commit SHA the current index was built from."""
        return self._indexed_sha

    # -- index management -------------------------------------------------------------------------

    def _vectors_for(self, docs: list[Document], fresh: list[list[float]], todo: list[int]) -> None:
        for i, vec in zip(todo, fresh, strict=True):
            m = docs[i].metadata
            self._vector_cache[(m["blob_sha"], m["start_line"], m["end_line"])] = vec

    def _cached_vectors(self, docs: list[Document]) -> tuple[list[list[float]], list[int]]:
        have: list[list[float]] = [[] for _ in docs]
        todo: list[int] = []
        for i, d in enumerate(docs):
            m = d.metadata
            cached = self._vector_cache.get((m["blob_sha"], m["start_line"], m["end_line"]))
            if cached is None:
                todo.append(i)
            else:
                have[i] = cached
        return have, todo

    def _sync_index(self, snap: RepoSnapshot) -> None:
        with self._lock:
            if snap.commit_sha == self._indexed_sha:
                return
            self.loader.fetch(snap)
            docs = list(self.loader.documents(snap))
            vectors = None
            if self.embeddings is not None:
                vectors, todo = self._cached_vectors(docs)
                if todo:
                    fresh = embed_all(self.embeddings, [docs[i] for i in todo])
                    self._vectors_for(docs, fresh, todo)
                    for i, v in zip(todo, fresh, strict=True):
                        vectors[i] = v
            self._index, self._indexed_sha = ChunkIndex(docs, vectors), snap.commit_sha

    async def _async_index(self, snap: RepoSnapshot) -> None:
        if snap.commit_sha == self._indexed_sha:
            return
        await self.loader.afetch(snap)
        docs = list(self.loader.documents(snap))
        vectors = None
        if self.embeddings is not None:
            vectors, todo = self._cached_vectors(docs)
            if todo:
                fresh = await aembed_all(self.embeddings, [docs[i] for i in todo])
                self._vectors_for(docs, fresh, todo)
                for i, v in zip(todo, fresh, strict=True):
                    vectors[i] = v
        index = await asyncio.to_thread(ChunkIndex, docs, vectors)
        self._index, self._indexed_sha = index, snap.commit_sha

    def _search(
        self, query: str, k: int | None, query_vector: list[float] | None
    ) -> list[Document]:
        assert self._index is not None
        return self._index.search(query, _resolve_k(k, self.k), query_vector)

    # -- BaseRetriever API ------------------------------------------------------------------------

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, k: int | None = None
    ) -> list[Document]:
        self._sync_index(self.loader.snapshot())
        qvec = self.embeddings.embed_query(query) if self.embeddings else None
        return self._search(query, k, qvec)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, k: int | None = None
    ) -> list[Document]:
        await self._async_index(await self.loader.asnapshot())
        qvec = await self.embeddings.aembed_query(query) if self.embeddings else None
        return self._search(query, k, qvec)


class GitHubIssuesRetriever(BaseRetriever):
    """Find the issue, discussion or comment that answers a problem, with how it was resolved.

    Ask in your own words or paste a traceback. Results are the best-matching passage of each
    thread, linked to the exact comment, with ``resolution`` (``fixed``, ``answered``, ``wontfix``,
    ``duplicate``, ``open``) and ``resolved_by`` (merged PR numbers). A thread closed as a
    duplicate is replaced by the thread it points to.

    Two strategies, chosen by ``mode``:

    * ``"index"``: for repositories with at most ``max_index_threads`` issues. All issues,
      comments, closing PRs and discussions are fetched with repo-wide listings (a handful of
      requests per hundred threads) and ranked locally. A free ETag check per query re-fetches
      only when something changed.
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
    """Seconds after which the index is refreshed even if the probe sees no change."""
    base_url: str = DEFAULT_BASE_URL

    _client: GitHubClient | None = PrivateAttr(default=None)
    _index: ThreadIndex | None = PrivateAttr(default=None)
    _mode: str | None = PrivateAttr(default=None)
    _stamp: str = PrivateAttr(default="")
    _built_at: float = PrivateAttr(default=0.0)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @field_validator("repo")
    @classmethod
    def _check_repo(cls, value: str) -> str:
        return normalize_repo(value)

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

    def _build(self, threads: list[Thread]) -> ThreadIndex:
        docs = [d for t in threads for d in thread_documents(t, self.repo)]
        vectors = None
        if self.embeddings is not None and docs:
            vectors = embed_all_with(self.embeddings, [issue_embed_text(d) for d in docs])
        return ThreadIndex(docs, vectors)

    def _refresh_index(self) -> None:
        stamp = probe(self.client, self.repo)
        fresh = time.monotonic() - self._built_at < self.max_age
        if self._index is not None and stamp == self._stamp and fresh:
            return
        threads = fetch_all_issues(self.client, self.repo, max_threads=self.max_index_threads)
        if self.include_discussions:
            threads.extend(fetch_discussions(self.client, self.repo))
        if not threads:
            warnings.warn(
                f"{self.repo} has no issues or discussions to search (issues may be turned off "
                "for this repository).",
                stacklevel=3,
            )
        self._index = self._build(threads)
        self._stamp, self._built_at = stamp, time.monotonic()

    def _search_mode(self, query: str, k: int) -> list[Document]:
        numbers = search_candidates(self.client, self.repo, search_variants(query))
        prior = [("issue", n) for n in numbers]
        threads: list[Thread] = []
        for n in numbers[: self.max_candidates]:
            try:
                t = fetch_one_issue(self.client, self.repo, n)
            except GitHubNotFoundError:  # deleted or transferred since the search
                continue
            if t is not None:
                threads.append(t)
        # Search order is a ranking too: keep it by listing candidates first-seen first.
        docs = [d for t in threads for d in thread_documents(t, self.repo)]
        if not docs:
            return []
        vectors = None
        qvec = None
        if self.embeddings is not None:
            vectors = embed_all_with(self.embeddings, [issue_embed_text(d) for d in docs])
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
