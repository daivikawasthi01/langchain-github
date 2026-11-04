"""In-memory ranking index over repository chunks: code-aware BM25 with optional embeddings."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from langchain_core.documents import Document

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it its of on or that the "
    "this to was what when where which who why with you your we our my me not use used using "
    "get set code file files function class method repo repository project".split()
)
# Words that are stopwords in a question but meaningful tokens inside source code.
_QUERY_ONLY_STOPWORDS = frozenset({"get", "set", "use", "used", "using", "file", "files"})


def stem(word: str) -> str:
    """A deliberately small suffix stripper so ``retries``/``retry`` and ``configured`` meet."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str, *, query: bool = False) -> list[str]:
    """Split text into search tokens, keeping whole identifiers *and* their parts.

    ``retryLimit`` and ``RETRY_LIMIT`` both yield ``retrylimit`` plus ``retry`` and ``limit``,
    so a natural-language query about "retry limit" matches code that spells it either way.
    """
    out: list[str] = []
    for ident in _IDENT.findall(text):
        parts = [p for p in re.split(r"_+", _CAMEL.sub("_", ident).lower()) if p]
        if len(parts) > 1:
            out.append(ident.lower().replace("_", ""))
        for part in parts:
            if part in _STOPWORDS and not (part in _QUERY_ONLY_STOPWORDS and not query):
                continue
            out.append(stem(part))
    return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return fused


def embed_text(doc: Document) -> str:
    """What gets embedded: the chunk prefixed with its path, so file names carry meaning."""
    return f"{doc.metadata['path']}\n{doc.page_content}"


class BM25:
    """Okapi BM25 over an inverted index.

    Uses the Lucene IDF, ``log(1 + (N - n + 0.5) / (n + 0.5))``, which stays positive for terms
    found in most documents. (The textbook form goes negative there, which silently zeroes out
    matches in small repositories.)
    """

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.size = len(corpus)
        self._len = [len(doc) for doc in corpus]
        self._avg = (sum(self._len) / self.size) if self.size else 1.0
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                self._postings.setdefault(term, []).append((i, tf))
        self._idf = {
            t: math.log(1 + (self.size - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self._postings.items()
        }

    def get_scores(self, query: Sequence[str]) -> list[float]:
        scores = [0.0] * self.size
        for term in set(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in self._postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return scores


class ChunkIndex:
    """BM25 over chunks, optionally fused with embedding similarity (reciprocal rank fusion)."""

    def __init__(self, docs: list[Document], vectors: list[list[float]] | None = None) -> None:
        self.docs = docs
        self.vectors = vectors
        corpus = [self._doc_tokens(d) for d in docs]
        self._bm25 = BM25(corpus) if corpus else None

    @staticmethod
    def _doc_tokens(doc: Document) -> list[str]:
        # The path is repeated so a file called ``retry.py`` outranks a passing mention of retry.
        return tokenize(doc.metadata["path"]) * 2 + tokenize(doc.page_content)

    def ranked(
        self, query: str, query_vector: list[float] | None = None
    ) -> list[tuple[int, float]]:
        """All chunks ordered best-first as ``(index, score)``."""
        if not self.docs or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query, query=True))
        lexical = sorted(range(len(scores)), key=lambda i: -scores[i])
        if query_vector is None or self.vectors is None:
            return [(i, float(scores[i])) for i in lexical]
        sims = [_cosine(query_vector, v) for v in self.vectors]
        semantic = sorted(range(len(sims)), key=lambda i: -sims[i])
        fused = _rrf([lexical, semantic])
        return sorted(fused.items(), key=lambda kv: -kv[1])

    def search(self, query: str, k: int, query_vector: list[float] | None = None) -> list[Document]:
        picked: list[Document] = []
        for idx, score in self.ranked(query, query_vector):
            if score <= 0:
                break
            doc = self.docs[idx]
            if any(_overlaps(doc, p) for p in picked):
                continue
            meta = {**doc.metadata, "score": round(score, 4), "rank": len(picked) + 1}
            picked.append(Document(page_content=doc.page_content, metadata=meta))
            if len(picked) >= k:
                break
        return picked


def _overlaps(a: Document, b: Document) -> bool:
    """True if two chunks of the same file cover mostly the same lines."""
    ma, mb = a.metadata, b.metadata
    if ma["path"] != mb["path"]:
        return False
    lo = max(ma["start_line"], mb["start_line"])
    hi = min(ma["end_line"], mb["end_line"])
    if hi < lo:
        return False
    smaller = min(ma["end_line"] - ma["start_line"], mb["end_line"] - mb["start_line"]) + 1
    return bool((hi - lo + 1) / smaller > 0.5)


def issue_embed_text(doc: Document) -> str:
    """What gets embedded for an issue piece: the title plus the passage."""
    return f"{doc.metadata['title']}\n{doc.page_content}"


class ThreadIndex:
    """Ranks issue and discussion threads, returning each thread's best passage.

    Three rankings are fused with reciprocal rank fusion: BM25 over individual passages, BM25 over
    whole threads (so a question phrased like the report can match the discussion that fixes it),
    and, if vectors are given, embedding similarity over passages. A thread closed as a duplicate
    is replaced by the thread it points to when that one is indexed.
    """

    def __init__(self, docs: list[Document], vectors: list[list[float]] | None = None) -> None:
        self.docs = docs
        self.vectors = vectors
        self._titles: dict[tuple[str, int], list[str]] = {}
        self._keys: list[tuple[str, int]] = []
        by_thread: dict[tuple[str, int], list[str]] = {}
        for d in docs:
            key = (d.metadata["kind"], d.metadata["number"])
            self._keys.append(key)
            self._titles.setdefault(key, tokenize(d.metadata["title"]))
            by_thread.setdefault(key, []).extend(tokenize(d.page_content))
        self._thread_keys = list(by_thread)
        self._piece = BM25([self._piece_tokens(d) for d in docs]) if docs else None
        self._thread = (
            BM25([self._titles[k] * 3 + by_thread[k] for k in self._thread_keys]) if docs else None
        )

    def _piece_tokens(self, doc: Document) -> list[str]:
        key = (doc.metadata["kind"], doc.metadata["number"])
        return self._titles[key] * 2 + tokenize(doc.page_content)

    def threads(self) -> int:
        return len(self._thread_keys)

    def _by_key(self) -> dict[tuple[str, int], list[int]]:
        out: dict[tuple[str, int], list[int]] = {}
        for i, key in enumerate(self._keys):
            out.setdefault(key, []).append(i)
        return out

    def search(
        self,
        query: str,
        k: int,
        query_vector: list[float] | None = None,
        prior: Sequence[tuple[str, int]] | None = None,
        prior_only: bool = False,
    ) -> list[Document]:
        """Best passage per thread.

        ``prior`` is an outside ranking (GitHub's own search order) fused in as one more signal.
        With ``prior_only`` the threads keep the prior's order and only the passage shown and the
        metadata come from this index.
        """
        if not self.docs or self._piece is None or self._thread is None:
            return []
        q = tokenize(query, query=True)
        piece_scores = self._piece.get_scores(q)
        thread_scores = self._thread.get_scores(q)
        pieces_of = self._by_key()
        best: dict[tuple[str, int], int] = {
            key: max(idxs, key=lambda i: piece_scores[i]) for key, idxs in pieces_of.items()
        }
        sims = None
        if query_vector is not None and self.vectors is not None:
            sims = [_cosine(query_vector, v) for v in self.vectors]
        thread_score = dict(zip(self._thread_keys, thread_scores, strict=True))
        thread_rank = sorted(self._thread_keys, key=lambda key: -thread_score[key])
        piece_rank = sorted(best, key=lambda key: -piece_scores[best[key]])
        rankings: list[list[tuple[str, int]]] = [
            piece_rank,
            thread_rank,
        ]
        if sims is not None:
            rankings.append(sorted(best, key=lambda key: -max(sims[i] for i in pieces_of[key])))
        if prior:
            rankings.append([key for key in prior if key in pieces_of])
        fused: dict[tuple[str, int], float] = {}
        for ranking in rankings:
            for pos, key in enumerate(ranking):
                fused[key] = fused.get(key, 0.0) + 1.0 / (60 + pos + 1)
        backed = set(prior or ())
        matched = {
            key
            for key in fused
            if sims is not None
            or key in backed
            or piece_scores[best[key]] > 0
            or thread_score[key] > 0
        }
        ordered = [key for key, _ in sorted(fused.items(), key=lambda kv: -kv[1]) if key in matched]
        if prior_only and prior:
            head = [key for key in prior if key in pieces_of]
            ordered = head + [key for key in ordered if key not in head]

        results: list[Document] = []
        seen: set[tuple[str, int]] = set()
        for key in ordered:
            doc = self.docs[best[key]]
            target = doc.metadata.get("duplicate_of")
            meta = dict(doc.metadata)
            content = doc.page_content
            if target is not None and ("issue", target) in pieces_of:
                canon_key = ("issue", target)
                canon = self.docs[best[canon_key]]
                meta = {**canon.metadata, "matched_duplicate": doc.metadata["number"]}
                content, key = canon.page_content, canon_key
            if key in seen:
                continue
            seen.add(key)
            meta["rank"] = len(results) + 1
            results.append(Document(page_content=content, metadata=meta))
            if len(results) >= k:
                break
        return results
