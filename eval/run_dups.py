"""Real-repo issues eval: does the retriever find the issue a duplicate was closed against?

    GITHUB_TOKEN=... python eval/run_dups.py [--index-repos psf/requests,...]

Rows come from ``eval/build_dup_golden.py``: the question is a duplicate issue's title, the answer
is the canonical issue maintainers pointed to. The duplicate itself (and its own pointer comment)
is removed from every system's results before scoring, so no system can answer by finding the
question's own thread. Systems: GitHub issue search (hybrid, keywords) and ours in *search* mode;
for repositories small enough, also ours in *index* mode with the duplicate thread dropped from
the index.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from langchain_github import GitHubClient, GitHubIssuesRetriever
from langchain_github._index import ThreadIndex
from langchain_github._query import keywords, normalize
from langchain_github._threads import search_candidates

K = 5
HERE = Path(__file__).parent


def score(rows: list[dict], got: dict[str, list[int]]) -> dict[str, float]:
    h1 = h5 = rr = 0.0
    for r in rows:
        ranked = [n for n in got.get(r["id"], []) if n != r["duplicate"]][:K]
        for rank, n in enumerate(ranked, 1):
            if n in r["expected"]:
                rr += 1 / rank
                h5 += 1
                h1 += rank == 1
                break
    n = len(rows) or 1
    return {"n": len(rows), "hit@1": h1 / n, "hit@5": h5 / n, "mrr": rr / n}


def numbers(docs) -> list[int]:
    out: list[int] = []
    for d in docs:
        if d.metadata["kind"] == "issue" and d.metadata["number"] not in out:
            out.append(d.metadata["number"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-repos", default="", help="comma list; evaluates index mode too")
    ap.add_argument("--out", default=str(HERE / "results" / "dups.json"))
    args = ap.parse_args()
    rows = [json.loads(line) for line in (HERE / "golden_dups.jsonl").read_text().splitlines()]
    client = GitHubClient()
    systems: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for r in rows:
        repo, q = r["repo"], r["question"]
        systems["github_keywords"][r["id"]] = search_candidates(
            client, repo, [("lexical", " ".join(keywords(normalize(q), 4)))], per_query=K + 1
        )
        systems["github_hybrid"][r["id"]] = search_candidates(
            client, repo, [("hybrid", normalize(q)[:256])], per_query=K + 1
        )
    retrievers: dict[str, GitHubIssuesRetriever] = {}
    for r in rows:
        ret = retrievers.setdefault(
            r["repo"], GitHubIssuesRetriever(repo=r["repo"], mode="search", k=K + 1)
        )
        systems["ours_search"][r["id"]] = numbers(ret.invoke(r["question"]))
    index_repos = [x for x in args.index_repos.split(",") if x]
    for repo in index_repos:
        ret = GitHubIssuesRetriever(repo=repo, mode="index", max_index_threads=20000, k=K + 1)
        ret.invoke("warm up the index")
        assert ret._index is not None
        for r in [x for x in rows if x["repo"] == repo]:
            docs = [d for d in ret._index.docs if d.metadata["number"] != r["duplicate"]]
            prior = [
                ("issue", n)
                for n in search_candidates(
                    client, repo, [("hybrid", normalize(r["question"])[:256])], per_query=10
                )
            ]
            res = ThreadIndex(docs).search(normalize(r["question"]), K + 1, prior=prior)
            systems["ours_index"][r["id"]] = numbers(res)

    report = {}
    print(f"{'system':18} {'n':>3} {'hit@1':>6} {'hit@5':>6} {'MRR':>6}")
    for name, got in systems.items():
        sub = [r for r in rows if r["id"] in got]
        report[name] = score(sub, got)
        s = report[name]
        print(f"{name:18} {s['n']:>3} {s['hit@1']:>6.2f} {s['hit@5']:>6.2f} {s['mrr']:>6.2f}")
    Path(args.out).write_text(json.dumps({"scores": report, "results": systems}, indent=1))


if __name__ == "__main__":
    main()
