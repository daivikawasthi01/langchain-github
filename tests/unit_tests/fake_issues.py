"""A tiny in-memory GitHub issues API served through respx."""

import json
import re

import httpx

from langchain_github._threads import closing_references

API = "https://api.github.com"
REPO = "acme/demo"
HTML = f"https://github.com/{REPO}"


def issue(n, title, body, state="open", reason=None, updated="2026-01-01T00:00:00Z"):
    return {
        "number": n,
        "title": title,
        "body": body,
        "state": state,
        "state_reason": reason,
        "html_url": f"{HTML}/issues/{n}",
        "labels": [{"name": "bug"}],
        "updated_at": updated,
    }


def comment(cid, n, body):
    return {
        "id": cid,
        "body": body,
        "html_url": f"{HTML}/issues/{n}#issuecomment-{cid}",
        "issue_url": f"{API}/repos/{REPO}/issues/{n}",
    }


class FakeIssues:
    def __init__(self):
        self.updated = "2026-01-01T00:00:00Z"
        self.issues = [
            issue(
                1,
                "Connection pool exhausted under load",
                "acquire() raises when threads pile up",
                "closed",
                "completed",
            ),
            issue(2, "Timeouts ignored when streaming", "stream never times out", "open"),
            issue(3, "Pool exhausted again", "same as before", "closed", "duplicate"),
            issue(4, "Add dark mode", "please", "closed", "not_planned"),
            {**issue(9, "Fix pool", "Fixes #1", "closed"), "pull_request": {}},
        ]
        self.comments = [
            comment(
                11, 1, "Raising pool_size to 50 and releasing sockets in a finally block fixes it."
            ),
            comment(31, 3, "Duplicate of #1."),
            comment(41, 4, "Out of scope for this project."),
        ]
        self.pulls = [{"number": 9, "title": "Fix pool", "body": "Fixes #1", "merged_at": "x"}]
        self.discussions = [
            {
                "number": 5,
                "title": "How do I configure retries?",
                "body": "Where is the retry count set?",
                "url": f"{HTML}/discussions/5",
                "closed": False,
                "updatedAt": "x",
                "answer": {
                    "id": "DC1",
                    "url": f"{HTML}/discussions/5#discussioncomment-1",
                    "body": "Set RETRY_LIMIT.",
                },
                "comments": {"nodes": []},
            }
        ]
        self.graphql_failures: list[str] = []  # GraphQL messages to refuse, one per request
        self.fork_pr = False  # also report a merged PR from another repository as closing #2
        self.since_log: list[str] = []  # the `since` of every incremental issue listing
        self.search_hits: list[int] = []
        self.calls: list[str] = []
        self.etag_hits = 0

    def _gql_issue(self, i):
        n = i["number"]
        text_closers = [
            p for p in self.pulls if n in closing_references(f"{p['title']}\n{p['body']}", REPO)
        ]
        closers = [  # GitHub links only PRs aimed at the default branch
            {"number": p["number"], "merged": True, "repository": {"nameWithOwner": REPO}}
            for p in text_closers
            if p.get("linked", True)
        ]
        events = [
            {
                "source": {
                    "number": p["number"],
                    "merged": True,
                    "title": p["title"],
                    "body": p["body"],
                    "repository": {"nameWithOwner": REPO},
                }
            }
            for p in text_closers
        ]
        if self.fork_pr and n == 2:
            closers.append(
                {"number": 77, "merged": True, "repository": {"nameWithOwner": "fork/demo"}}
            )
        mine = [c for c in self.comments if c["issue_url"].endswith(f"/{n}")]
        return {
            "__typename": "Issue",
            "number": n,
            "title": i["title"],
            "body": i["body"],
            "url": i["html_url"],
            "state": i["state"].upper(),
            "stateReason": i["state_reason"].upper() if i["state_reason"] else None,
            "updatedAt": i["updated_at"],
            "labels": {"nodes": i["labels"]},
            "comments": {
                "pageInfo": {"hasNextPage": n in getattr(self, "overflow", set())},
                "nodes": [
                    {"databaseId": c["id"], "url": c["html_url"], "body": c["body"]} for c in mine
                ],
            },
            "closedByPullRequestsReferences": {"nodes": closers},
            "timelineItems": {"nodes": events},
        }

    def _graphql(self, req):
        query = json.loads(req.content)["query"]
        if self.graphql_failures:
            return httpx.Response(
                200, json={"data": None, "errors": [{"message": self.graphql_failures.pop(0)}]}
            )
        real = [i for i in self.issues if "pull_request" not in i]
        if "issueOrPullRequest" in query:
            self.calls.append("graphql:batch")
            out = {}
            for n in map(int, re.findall(r"i(\d+): issueOrPullRequest", query)):
                found = next((i for i in self.issues if i["number"] == n), None)
                if found is None:
                    out[f"i{n}"] = None
                elif "pull_request" in found:
                    out[f"i{n}"] = {"__typename": "PullRequest"}
                else:
                    out[f"i{n}"] = self._gql_issue(found)
            errors = [{"message": "Could not resolve"}] if None in out.values() else None
            return httpx.Response(200, json={"data": {"repository": out}, "errors": errors})
        if "issues(first" in query:
            self.calls.append("graphql:issues")
            variables = json.loads(req.content)["variables"]
            nodes = [self._gql_issue(i) for i in real]
            if variables.get("since"):
                self.since_log.append(variables["since"])
                nodes = [n for n in nodes if n["updatedAt"] >= variables["since"]]
            if variables.get("states"):
                nodes = [n for n in nodes if n["state"] in variables["states"]]
            page = {"pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": nodes}
            return httpx.Response(200, json={"data": {"repository": {"issues": page}}})
        self.calls.append("graphql:discussions")
        return httpx.Response(
            200,
            json={
                "data": {
                    "repository": {
                        "discussions": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": self.discussions,
                        }
                    }
                }
            },
        )

    def mount(self, router):
        def handle(req: httpx.Request) -> httpx.Response:
            path, params = req.url.path, req.url.params
            self.calls.append(path)
            if path == "/search/issues":
                self.calls.append("search:" + params.get("search_type", "lexical"))
                if "is:issue" in params["q"] and params.get("per_page") == "1":
                    return httpx.Response(
                        200, json={"total_count": len(self.issues) - 1, "items": []}
                    )
                by_number = {i["number"]: i for i in self.issues}
                items = [by_number[n] for n in self.search_hits]
                return httpx.Response(200, json={"total_count": len(items), "items": items})
            if path == f"/repos/{REPO}/issues" and params.get("per_page") == "1":
                tag = f'"{self.updated}"'
                if req.headers.get("if-none-match") == tag:
                    self.etag_hits += 1
                    return httpx.Response(304)
                return httpx.Response(
                    200, json=[{"updated_at": self.updated}], headers={"etag": tag}
                )
            if path == f"/repos/{REPO}/issues":
                if params.get("since"):
                    self.since_log.append(params["since"])
                    return httpx.Response(
                        200, json=[i for i in self.issues if i["updated_at"] >= params["since"]]
                    )
                return httpx.Response(200, json=self.issues)
            if path == f"/repos/{REPO}/issues/comments":
                return httpx.Response(200, json=self.comments)
            if path == f"/repos/{REPO}/pulls":
                return httpx.Response(200, json=self.pulls)
            num = path.rsplit("/", 1)[-1]
            if path.endswith("/timeline"):
                n = int(path.split("/")[-2])
                pr = {
                    "number": 9,
                    "title": "Fix pool",
                    "body": "Fixes #1",
                    "pull_request": {"merged_at": "x"},
                }
                ev = [{"event": "cross-referenced", "source": {"issue": pr}}] if n == 1 else []
                return httpx.Response(200, json=ev)
            if path.endswith("/comments"):
                n = int(path.split("/")[-2])
                return httpx.Response(
                    200, json=[c for c in self.comments if c["issue_url"].endswith(f"/{n}")]
                )
            if num.isdigit():
                return httpx.Response(
                    200, json=next(i for i in self.issues if i["number"] == int(num))
                )
            return httpx.Response(404, json={"message": "nope"})

        router.get(url__regex=rf"{API}/.*").mock(side_effect=handle)
        router.post(f"{API}/graphql").mock(side_effect=self._graphql)
