"""A tiny in-memory GitHub issues API served through respx."""

import httpx

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
        self.search_hits: list[int] = []
        self.calls: list[str] = []
        self.etag_hits = 0

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
                return httpx.Response(200, json=self.issues)
            if path == f"/repos/{REPO}/issues/comments":
                return httpx.Response(200, json=self.comments)
            if path == f"/repos/{REPO}/pulls":
                return httpx.Response(200, json=self.pulls)
            if path == "/graphql":
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
        router.post(f"{API}/graphql").mock(side_effect=handle)
