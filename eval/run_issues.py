"""Issues-retriever eval on the synthetic testbed (eval/golden_issues.jsonl).

Usage:  GITHUB_TOKEN=... python eval/run_issues.py [--out eval/results/issues.json]

A question is a *hit@k* if one of the first k distinct threads returned is in ``expected`` (the
canonical issue or any of its duplicates; for discussions, the discussion). ``resolution accuracy``
is measured on questions whose top-1 result is a hit: the result's ``resolution`` must equal the
label and, when the label names a fixing PR, ``resolved_by`` must contain it.

Baselines are GitHub's own issue search. They only index issues, so discussion questions can only
be answered by our retriever; headline numbers are therefore reported for issue questions alone,
with the all-questions figure beside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_github import GitHubClient, GitHubIssuesRetriever
from langchain_github._query import keywords

K = 5
HERE = Path(__file__).parent


def github_search(client: GitHubClient, repo: str, text: str, search_type: str | None) -> list[int]:
    params = {"q": f"repo:{repo} is:issue {text}", "per_page": K}
    if search_type:
        params["search_type"] = search_type
    try:
        body = client.get_json("/search/issues", params, resource="search")
    except Exception:
        return []
    return [i["number"] for i in body.get("items", [])]


def distinct(keys: list[tuple[str, int]]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for key in keys:
        if key not in out:
            out.append(key)
    return out


def score(rows: list[dict], results: dict[str, list[tuple[str, int]]]) -> dict[str, float]:
    h1 = h5 = rr = 0.0
    for row in rows:
        want = {(row["kind"], n) for n in row["expected"]}
        got = distinct(results.get(row["id"], []))[:K]
        for rank, key in enumerate(got, 1):
            if key in want:
                rr += 1 / rank
                h5 += 1
                h1 += rank == 1
                break
    n = len(rows) or 1
    return {"n": len(rows), "hit@1": h1 / n, "hit@5": h5 / n, "mrr": rr / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "issues.json"))
    args = ap.parse_args()
    rows = [json.loads(line) for line in (HERE / "golden_issues.jsonl").read_text().splitlines()]
    repo = rows[0]["repo"]
    client = GitHubClient()

    systems: dict[str, dict[str, list[tuple[str, int]]]] = {
        "github_lexical_raw": {},
        "github_lexical_keywords": {},
        "github_hybrid": {},
        "ours_index": {},
        "ours_search": {},
    }
    for row in rows:
        q = row["question"]
        if row["kind"] == "issue":
            systems["github_lexical_raw"][row["id"]] = [
                ("issue", n) for n in github_search(client, repo, q, None)
            ]
            systems["github_lexical_keywords"][row["id"]] = [
                ("issue", n) for n in github_search(client, repo, " ".join(keywords(q, 4)), None)
            ]
            systems["github_hybrid"][row["id"]] = [
                ("issue", n) for n in github_search(client, repo, q, "hybrid")
            ]
    for name, mode in (("ours_index", "index"), ("ours_search", "search")):
        retriever = GitHubIssuesRetriever(repo=repo, k=K, mode=mode)  # type: ignore[arg-type]
        for row in rows:
            if mode == "search" and row["kind"] == "discussion":
                continue  # search mode proposes candidates from issue search only
            docs = retriever.invoke(row["question"])
            systems[name][row["id"]] = [(d.metadata["kind"], d.metadata["number"]) for d in docs]
            row.setdefault("_docs", {})[name] = docs

    issue_rows = [r for r in rows if r["kind"] == "issue"]
    report = {}
    for name, res in systems.items():
        report[name] = {"issues": score(issue_rows, res)}
        if name.startswith("ours_index"):
            report[name]["all"] = score(rows, res)

    acc = {}
    for name in ("ours_index", "ours_search"):
        ok = total = 0
        for row in issue_rows:
            docs = row.get("_docs", {}).get(name, [])
            if not docs or (docs[0].metadata["kind"], docs[0].metadata["number"]) not in {
                ("issue", n) for n in row["expected"]
            }:
                continue
            total += 1
            m = docs[0].metadata
            good = m["resolution"] == row["resolution"]
            if row["resolved_by"]:
                good = good and row["resolved_by"] in m["resolved_by"]
            ok += good
        acc[name] = {"top1_hits": total, "resolution_accuracy": ok / total if total else 0.0}
    report["resolution_accuracy"] = acc

    misses = {
        name: [r["id"] for r in issue_rows if not any(
            k in {("issue", n) for n in r["expected"]}
            for k in distinct(res.get(r["id"], []))[:K]
        )]
        for name, res in systems.items()
    }
    report["misses"] = misses
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"{'system':28} {'n':>3} {'hit@1':>6} {'hit@5':>6} {'MRR':>6}")
    for name in systems:
        s = report[name]["issues"]
        print(f"{name:28} {s['n']:>3} {s['hit@1']:>6.2f} {s['hit@5']:>6.2f} {s['mrr']:>6.2f}")
    a = report["ours_index"]["all"]
    print(f"{'ours_index (+discussions)':28} {a['n']:>3} {a['hit@1']:>6.2f} {a['hit@5']:>6.2f} {a['mrr']:>6.2f}")
    print("resolution accuracy:", json.dumps(acc))
    print("misses:", json.dumps({k: v for k, v in misses.items() if k.startswith("ours")}))


if __name__ == "__main__":
    main()
