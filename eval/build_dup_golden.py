"""Build a real-world golden set from issues maintainers closed as duplicates.

For each issue closed with reason "duplicate", the canonical issue is the single other *issue*
number the closing comments point to (``duplicate of #N`` or an issue URL). The question is the
duplicate's title, i.e. what someone would type before finding that the problem was already
reported. Rows without exactly one unambiguous issue target are dropped.

    GITHUB_TOKEN=... python eval/build_dup_golden.py psf/requests pallets/flask > eval/golden_dups.jsonl
"""

from __future__ import annotations

import json
import re
import sys

from langchain_github import GitHubClient, GitHubError

REF = re.compile(
    r"(?:duplicate\s+of|dupe\s+of|dup\s+of|same\s+as|see|already\s+(?:reported|tracked)\s+in)?\s*(?:https://github\.com/[\w.-]+/[\w.-]+/issues/|#)(\d+)",
    re.I,
)


PER_REPO = 10


def main(repos: list[str]) -> None:
    c = GitHubClient()
    for repo in repos:
        try:
            dups = list(
                c.paginate(
                    "/search/issues",
                    {"q": f"repo:{repo} is:issue is:closed reason:duplicate", "per_page": 100},
                    resource="search",
                    items_key="items",
                    max_items=60,
                )
            )
        except GitHubError as exc:
            print(f"skip {repo}: {exc}", file=sys.stderr)
            continue
        kept = 0
        for d in dups:
            if kept >= PER_REPO or len(d["title"].strip()) < 20 or "spam" in d["title"].lower():
                continue
            cs = list(c.paginate(f"/repos/{repo}/issues/{d['number']}/comments"))
            targets = {
                int(n) for cm in cs for n in REF.findall(cm["body"] or "") if int(n) != d["number"]
            }
            if len(targets) != 1:
                continue
            (t,) = targets
            try:
                other = c.get_json(f"/repos/{repo}/issues/{t}")
            except Exception:
                continue
            if "pull_request" in other:
                continue
            kept += 1
            print(
                json.dumps(
                    {
                        "id": f"{repo.split('/')[1]}-dup{d['number']}",
                        "kind": "issue",
                        "repo": repo,
                        "question": d["title"],
                        "duplicate": d["number"],
                        "expected": [t],
                    }
                )
            )


if __name__ == "__main__":
    main(sys.argv[1:])
