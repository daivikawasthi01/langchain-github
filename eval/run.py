"""Retrieval eval: does GitHubRepoRetriever beat what people use today?

Usage:  python eval/run.py [--systems a,b,...] [--out eval/results/run.json]

Metric: a question is a *hit@k* if any of the first k distinct files returned is in the question's
``expected`` list. MRR is the reciprocal rank of the first expected file (0 if absent). Labels in
``eval/golden.jsonl`` were written from the repositories' file trees and checked against file
contents by ``eval/verify_golden.py``; tests/examples that also answer a question count as misses,
so absolute numbers are conservative and only the *comparison* between systems is meaningful.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_github import GitHubClient, GitHubRepoRetriever, GitHubRepositoryLoader

K = 5
CACHE = Path(".cache/eval_code_search.json")
QUESTION_WORDS = set(
    "a an and are as at be by can do does for from how i if in into is it its of on or that the "
    "this to was what when where which who why with you your we my me so some such there their "
    "then than them these those like use used using get got set also just may might must should "
    "would could will shall one two three any each other same own very too".split()
)


class FastEmbedAdapter(Embeddings):
    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._m = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._m.embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._m.query_embed([text]))).tolist()


def keywords(question: str, n: int = 4) -> str:
    words = [w for w in re.findall(r"[A-Za-z]+", question.lower()) if w not in QUESTION_WORDS]
    return " ".join(sorted(set(words), key=lambda w: (-len(w), w))[:n])


def distinct_paths(docs: list[Document]) -> list[str]:
    seen: list[str] = []
    for d in docs:
        p = d.metadata["path"]
        if p not in seen:
            seen.append(p)
    return seen


# -- systems: each maps (repo, question) -> ordered list of file paths -------------------------


class CodeSearch:
    def __init__(self, mode: str, client: GitHubClient) -> None:
        self.mode, self.client = mode, client
        self.cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def __call__(self, repo: str, question: str) -> list[str]:
        q = question if self.mode == "raw" else keywords(question)
        key = f"{repo}|{q}"
        if key not in self.cache:
            try:
                body = self.client.get_json(
                    "/search/code",
                    {"q": f"{q} repo:{repo}", "per_page": K},
                    resource="code_search",
                    etag=False,
                )
                self.cache[key] = [i["path"] for i in body.get("items", [])]
            except Exception as exc:  # 422 on unusable queries is a legitimate zero-result
                self.cache[key] = []
                print(f"   code search error for {q!r}: {exc}")
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(self.cache))
        return self.cache[key]


class Naive:
    """Plain loader output -> default recursive splitter -> either word-BM25 or dense vectors."""

    def __init__(self, dense: bool, embedder: Embeddings | None, files: dict[str, dict[str, str]]):
        self.dense, self.embedder, self.files = dense, embedder, files
        self.indexes: dict[str, tuple[list[Document], object]] = {}

    def _build(self, repo: str):
        import numpy as np

        from langchain_github._index import BM25

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [
            Document(page_content=c, metadata={"path": p})
            for p, text in self.files[repo].items()
            for c in splitter.split_text(text)
        ]
        if self.dense:
            vecs = np.array(self.embedder.embed_documents([d.page_content for d in docs]))
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            return docs, vecs
        return docs, BM25([re.findall(r"\w+", d.page_content.lower()) for d in docs])

    def __call__(self, repo: str, question: str) -> list[str]:
        import numpy as np

        if repo not in self.indexes:
            self.indexes[repo] = self._build(repo)
        docs, index = self.indexes[repo]
        if self.dense:
            qv = np.array(self.embedder.embed_query(question))
            scores = index @ (qv / np.linalg.norm(qv))
        else:
            scores = index.get_scores(re.findall(r"\w+", question.lower()))
        order = sorted(range(len(docs)), key=lambda i: -scores[i])
        return distinct_paths([docs[i] for i in order[:50]])


class Ours:
    def __init__(self, embedder: Embeddings | None) -> None:
        self.embedder = embedder
        self.retrievers: dict[str, GitHubRepoRetriever] = {}
        self.requests: dict[str, int] = {}

    def __call__(self, repo: str, question: str) -> list[str]:
        if repo not in self.retrievers:
            self.retrievers[repo] = GitHubRepoRetriever(
                repo=repo, k=50, embeddings=self.embedder, cache_dir=".cache/gh"
            )
        r = self.retrievers[repo]
        return distinct_paths(r.invoke(question))


def score(results: dict[str, list[str]], rows: list[dict]) -> dict:
    per_repo: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    misses = []
    for row in rows:
        got = results[row["id"]]
        exp = set(row["expected"])
        rank = next((i + 1 for i, p in enumerate(got) if p in exp), None)
        per_repo[row["repo"]].append(
            (int(rank == 1), int(rank is not None and rank <= K), 1 / rank if rank else 0.0)
        )
        if rank is None or rank > K:
            misses.append({"id": row["id"], "question": row["question"], "top": got[:K]})
    flat = [x for v in per_repo.values() for x in v]
    return {
        "hit@1": round(statistics.mean(x[0] for x in flat), 3),
        "hit@5": round(statistics.mean(x[1] for x in flat), 3),
        "mrr": round(statistics.mean(x[2] for x in flat), 3),
        "by_repo_hit@5": {
            r: round(statistics.mean(x[1] for x in v), 2) for r, v in per_repo.items()
        },
        "misses": misses,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--systems", default="code_raw,code_kw,naive_bm25,naive_vec,ours_bm25,ours_hybrid"
    )
    ap.add_argument("--out", default="eval/results/run.json")
    ap.add_argument("--golden", default="eval/golden.jsonl")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.golden)]
    wanted = args.systems.split(",")

    client = GitHubClient()
    files = {}
    if any(s.startswith("naive") for s in wanted):
        for repo in {r["repo"] for r in rows}:
            ld = GitHubRepositoryLoader(repo, split=False, cache_dir=".cache/gh")
            files[repo] = {d.metadata["path"]: d.page_content for d in ld.lazy_load()}
    embedder = (
        FastEmbedAdapter() if any(s in wanted for s in ("naive_vec", "ours_hybrid")) else None
    )

    systems = {
        "code_raw": CodeSearch("raw", client),
        "code_kw": CodeSearch("kw", client),
        "naive_bm25": Naive(False, None, files),
        "naive_vec": Naive(True, embedder, files),
        "ours_bm25": Ours(None),
        "ours_hybrid": Ours(embedder),
    }
    report = {}
    for name in wanted:
        t0 = time.time()
        fn = systems[name]
        results = {row["id"]: fn(row["repo"], row["question"]) for row in rows}
        report[name] = {
            **score(results, rows),
            "seconds": round(time.time() - t0, 1),
            "top5": {k: v[:K] for k, v in results.items()},
        }
        r = report[name]
        print(
            f"{name:12s} hit@1={r['hit@1']:.2f}  hit@5={r['hit@5']:.2f}  mrr={r['mrr']:.2f}  {r['by_repo_hit@5']}  ({r['seconds']}s)"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
