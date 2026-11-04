"""Seed the synthetic testbed repo and write the golden file for the issues retriever.

    python eval/testbed/seed.py --dry-run                 # validate data, print what would happen
    python eval/testbed/seed.py --repo OWNER/NAME         # create everything (uses `gh` auth)

Progress is stored in eval/testbed/manifest.json, so an interrupted run resumes where it stopped.
Writes are spaced out to stay under GitHub's content-creation limits (80/min, 500/hour).
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import seed_data as data  # noqa: E402

MANIFEST = Path(__file__).parent / "manifest.json"
GOLDEN = Path(__file__).parents[1] / "golden_issues.jsonl"
PAUSE = 1.6


def validate() -> None:
    keys = set(data.ISSUES)
    for dup, canon, *_ in data.DUPLICATES:
        assert canon in keys, f"{dup} points at unknown issue {canon}"
    for key, issue in data.ISSUES.items():
        assert issue["resolution"] in {"fixed", "answered", "wontfix", "open"}, key
        if issue["resolution"] == "fixed":
            assert issue["file"] in data.INITIAL_FILES, f"{key}: {issue['file']} not seeded"
            assert f"# known-issue: {key}" in data.INITIAL_FILES[issue["file"]], f"{key}: no marker"
    markers = [
        line.split(": ", 1)[1]
        for text in data.INITIAL_FILES.values()
        for line in text.splitlines()
        if line.startswith("# known-issue: ")
    ]
    fixed = {k for k, v in data.ISSUES.items() if v["resolution"] == "fixed"}
    assert set(markers) == fixed, f"markers/fixed mismatch: {set(markers) ^ fixed}"


class GH:
    def __init__(self, repo: str, dry: bool) -> None:
        self.repo, self.dry = repo, dry
        self.calls = 0

    def api(self, method: str, path: str, body: dict | None = None):
        self.calls += 1
        if self.dry:
            return {"number": 0, "sha": "0" * 40, "content": "", "id": 0}
        cmd = ["gh", "api", "-X", method, path.format(repo=self.repo), "--input", "-"]
        out = subprocess.run(
            cmd, input=json.dumps(body or {}), capture_output=True, text=True, check=False
        )
        if out.returncode:
            raise RuntimeError(f"{method} {path}: {out.stderr.strip()[:300]}")
        time.sleep(PAUSE)
        return json.loads(out.stdout) if out.stdout.strip() else {}

    def graphql(self, query: str, **variables):
        self.calls += 1
        if self.dry:
            return {"id": "X", "number": 0}
        cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
        for k, v in variables.items():
            cmd += ["-f", f"{k}={v}"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if out.returncode:
            raise RuntimeError(f"graphql: {out.stderr.strip()[:300]}")
        time.sleep(PAUSE)
        return json.loads(out.stdout)["data"]


def seed(gh: GH, m: dict) -> None:
    # 1. files on the default branch
    m.setdefault("files", [])
    for path, text in {"README.md": data.README, **data.INITIAL_FILES}.items():
        if path in m["files"]:
            continue
        existing = None if gh.dry else _existing_sha(gh, path)
        body = {"message": f"Add {path}", "content": base64.b64encode(text.encode()).decode()}
        if existing:
            body["sha"] = existing
        gh.api("PUT", f"/repos/{{repo}}/contents/{path}", body)
        m["files"].append(path)
        save(m, gh)
    # 2. issues (canonical first, then duplicates)
    issues = m.setdefault("issues", {})
    for key, issue in data.ISSUES.items():
        if key in issues:
            continue
        number = gh.api("POST", "/repos/{repo}/issues", {"title": issue["title"], "body": issue["body"]})["number"]
        issues[key] = number
        for text in issue["comments"]:
            gh.api("POST", f"/repos/{{repo}}/issues/{number}/comments", {"body": text})
        if issue["resolution"] == "wontfix":
            gh.api("PATCH", f"/repos/{{repo}}/issues/{number}", {"state": "closed", "state_reason": "not_planned"})
        elif issue["resolution"] == "answered":
            gh.api("PATCH", f"/repos/{{repo}}/issues/{number}", {"state": "closed", "state_reason": "completed"})
        save(m, gh)
    for dup, canon, title, body in data.DUPLICATES:
        if dup in issues:
            continue
        number = gh.api("POST", "/repos/{repo}/issues", {"title": title, "body": body})["number"]
        issues[dup] = number
        gh.api("POST", f"/repos/{{repo}}/issues/{number}/comments", {"body": f"Duplicate of #{issues[canon]}."})
        gh.api("PATCH", f"/repos/{{repo}}/issues/{number}", {"state": "closed", "state_reason": "duplicate"})
        save(m, gh)
    # 3. fix PRs: branch -> edit marker line -> PR "Fixes #N" -> squash merge
    prs = m.setdefault("prs", {})
    for key, issue in data.ISSUES.items():
        if issue["resolution"] != "fixed" or key in prs:
            continue
        base = gh.api("GET", "/repos/{repo}/git/ref/heads/main")
        base_sha = base.get("object", {}).get("sha", "0" * 40)
        branch = f"fix/{key}"
        gh.api("POST", "/repos/{repo}/git/refs", {"ref": f"refs/heads/{branch}", "sha": base_sha})
        cur = gh.api("GET", f"/repos/{{repo}}/contents/{issue['file']}?ref=main")
        text = base64.b64decode(cur["content"]).decode() if cur.get("content") else ""
        text = text.replace(f"# known-issue: {key}", f"# fixed: {key} -- {issue['fix']}")
        gh.api("PUT", f"/repos/{{repo}}/contents/{issue['file']}", {
            "message": issue["fix"], "branch": branch, "sha": cur["sha"],
            "content": base64.b64encode(text.encode()).decode(),
        })
        pr = gh.api("POST", "/repos/{repo}/pulls", {
            "title": issue["fix"], "head": branch, "base": "main",
            "body": f"{issue['fix']}.\n\nFixes #{issues[key]}",
        })["number"]
        gh.api("PUT", f"/repos/{{repo}}/pulls/{pr}/merge", {"merge_method": "squash"})
        prs[key] = pr
        save(m, gh)
    # 4. Q&A discussions with accepted answers
    discussions = m.setdefault("discussions", {})
    info = gh.graphql(
        'query($o:String!,$n:String!){repository(owner:$o,name:$n){id discussionCategories(first:20)'
        "{nodes{id name isAnswerable}}}}",
        o=gh.repo.split("/")[0], n=gh.repo.split("/")[1],
    ) if not gh.dry else {}
    repo_id = info.get("repository", {}).get("id", "R")
    cats = info.get("repository", {}).get("discussionCategories", {}).get("nodes", [])
    cat_id = next((c["id"] for c in cats if c["isAnswerable"]), "C")
    for key, d in data.DISCUSSIONS.items():
        if key in discussions:
            continue
        created = gh.graphql(
            "mutation($r:ID!,$c:ID!,$t:String!,$b:String!){createDiscussion(input:{repositoryId:$r,"
            "categoryId:$c,title:$t,body:$b}){discussion{id number}}}",
            r=repo_id, c=cat_id, t=d["title"], b=d["body"],
        )
        disc = created.get("createDiscussion", {}).get("discussion", {"id": "D", "number": 0})
        comment = gh.graphql(
            "mutation($d:ID!,$b:String!){addDiscussionComment(input:{discussionId:$d,body:$b}){comment{id}}}",
            d=disc["id"], b=d["answer"],
        )
        cid = comment.get("addDiscussionComment", {}).get("comment", {"id": "K"})["id"]
        gh.graphql("mutation($i:ID!){markDiscussionCommentAsAnswer(input:{id:$i}){discussion{id}}}", i=cid)
        discussions[key] = disc["number"]
        save(m, gh)


def _existing_sha(gh: GH, path: str) -> str | None:
    try:
        return gh.api("GET", f"/repos/{{repo}}/contents/{path}").get("sha")
    except RuntimeError:
        return None


def save(m: dict, gh: GH) -> None:
    if not gh.dry:
        MANIFEST.write_text(json.dumps(m, indent=1))


def write_golden(m: dict, repo: str) -> int:
    rows = []
    dups: dict[str, list[int]] = {}
    for dup, canon, *_ in data.DUPLICATES:
        dups.setdefault(canon, []).append(m["issues"][dup])
    for key, issue in data.ISSUES.items():
        number = m["issues"][key]
        rows.append({
            "id": f"issue-{key}", "kind": "issue", "repo": repo, "question": issue["query"],
            "expected": [number, *dups.get(key, [])], "canonical": number,
            "resolution": issue["resolution"],
            "resolved_by": m["prs"].get(key),
        })
    for key, d in data.DISCUSSIONS.items():
        rows.append({
            "id": key, "kind": "discussion", "repo": repo, "question": d["query"],
            "expected": [m["discussions"][key]], "canonical": m["discussions"][key],
            "resolution": "answered", "resolved_by": None,
        })
    GOLDEN.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="daivikawasthi01/langchain-github-testbed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    validate()
    m = {} if args.dry_run or not MANIFEST.exists() else json.loads(MANIFEST.read_text())
    gh = GH(args.repo, args.dry_run)
    seed(gh, m)
    n_issues = len(data.ISSUES) + len(data.DUPLICATES)
    n_prs = sum(1 for i in data.ISSUES.values() if i["resolution"] == "fixed")
    print(f"{'DRY RUN: ' if args.dry_run else ''}{n_issues} issues, {n_prs} PRs, "
          f"{len(data.DISCUSSIONS)} discussions, {gh.calls} API calls "
          f"(~{gh.calls * PAUSE / 60:.1f} min with pacing)")
    if not args.dry_run:
        print(f"wrote {write_golden(m, args.repo)} golden rows to {GOLDEN}")


if __name__ == "__main__":
    main()
