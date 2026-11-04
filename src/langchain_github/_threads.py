"""Issue and discussion threads: fetching, resolution, and conversion to Documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from ._client import GitHubClient, GitHubError

_CLOSES = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+(?:[\w.-]+/[\w.-]+)?#(\d+)", re.I
)
_DUPLICATE = re.compile(r"\bduplicate\s+of\s+(?:[\w.-]+/[\w.-]+)?#(\d+)", re.I)

_DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 50, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body url closed updatedAt
        answer { id url body }
        comments(first: 30) { nodes { id url body } }
      }
    }
  }
}
"""

MAX_PIECE_CHARS = 1800


@dataclass
class Comment:
    id: str
    url: str
    body: str
    is_answer: bool = False


@dataclass
class Thread:
    kind: str  # "issue" | "discussion"
    number: int
    title: str
    body: str
    url: str
    state: str  # "open" | "closed"
    state_reason: str | None = None
    labels: list[str] = field(default_factory=list)
    comments: list[Comment] = field(default_factory=list)
    resolved_by: list[int] = field(default_factory=list)
    duplicate_of: int | None = None
    updated_at: str = ""

    @property
    def resolution(self) -> str:
        """``fixed`` | ``answered`` | ``wontfix`` | ``duplicate`` | ``open``."""
        if self.kind == "discussion":
            return "answered" if any(c.is_answer for c in self.comments) else "open"
        if self.state == "open":
            return "open"
        if self.duplicate_of is not None or self.state_reason == "duplicate":
            return "duplicate"
        if self.state_reason == "not_planned":
            return "wontfix"
        return "fixed" if self.resolved_by else "answered"


def closing_references(text: str | None) -> list[int]:
    return [int(n) for n in _CLOSES.findall(text or "")]


def duplicate_target(text: str | None) -> int | None:
    m = _DUPLICATE.search(text or "")
    return int(m.group(1)) if m else None


def build_issue_thread(
    issue: dict[str, Any], comments: list[dict[str, Any]], merged_prs: list[int]
) -> Thread:
    dup = None
    if issue.get("state") == "closed":
        for c in comments:
            dup = duplicate_target(c.get("body")) or dup
    return Thread(
        kind="issue",
        number=issue["number"],
        title=issue.get("title") or "",
        body=issue.get("body") or "",
        url=issue["html_url"],
        state=issue.get("state", "open"),
        state_reason=issue.get("state_reason"),
        labels=[
            lbl["name"] if isinstance(lbl, dict) else str(lbl) for lbl in issue.get("labels", [])
        ],
        comments=[Comment(str(c["id"]), c["html_url"], c.get("body") or "") for c in comments],
        resolved_by=sorted(set(merged_prs)),
        duplicate_of=dup,
        updated_at=issue.get("updated_at", ""),
    )


def build_discussion_thread(node: dict[str, Any]) -> Thread:
    answer_id = (node.get("answer") or {}).get("id")
    comments = [
        Comment(c["id"], c["url"], c.get("body") or "", is_answer=c["id"] == answer_id)
        for c in (node.get("comments") or {}).get("nodes", [])
    ]
    ans = node.get("answer")
    if ans and not any(c.is_answer for c in comments):
        comments.append(Comment(ans["id"], ans["url"], ans.get("body") or "", is_answer=True))
    return Thread(
        kind="discussion",
        number=node["number"],
        title=node.get("title") or "",
        body=node.get("body") or "",
        url=node["url"],
        state="closed" if node.get("closed") else "open",
        comments=comments,
        updated_at=node.get("updatedAt", ""),
    )


# -- fetching ---------------------------------------------------------------------------------


def probe(client: GitHubClient, repo: str) -> str:
    """Newest ``updated_at`` across issues and PRs. One request, free when nothing changed."""
    items = client.get_json(
        f"/repos/{repo}/issues",
        {"state": "all", "sort": "updated", "direction": "desc", "per_page": 1},
    )
    return str(items[0]["updated_at"]) if items else ""


def count_threads(client: GitHubClient, repo: str) -> int:
    body = client.get_json(
        "/search/issues", {"q": f"repo:{repo} is:issue", "per_page": 1}, resource="search"
    )
    return int(body.get("total_count", 0))


def merged_pr_closers(client: GitHubClient, repo: str, limit: int = 1000) -> dict[int, list[int]]:
    """Map issue number -> merged PR numbers that say they close it."""
    out: dict[int, list[int]] = {}
    for pr in client.paginate(f"/repos/{repo}/pulls", {"state": "closed"}, max_items=limit):
        if not pr.get("merged_at"):
            continue
        for n in closing_references(f"{pr.get('title', '')}\n{pr.get('body') or ''}"):
            out.setdefault(n, []).append(pr["number"])
    return out


def fetch_all_issues(
    client: GitHubClient, repo: str, *, max_threads: int, state: str = "all"
) -> list[Thread]:
    """Every issue with its comments using repo-wide listings (about N/100 requests each)."""
    issues = [
        i
        for i in client.paginate(
            f"/repos/{repo}/issues", {"state": state}, max_items=max_threads * 2
        )
        if "pull_request" not in i
    ][:max_threads]
    by_number: dict[int, list[dict[str, Any]]] = {}
    wanted = {i["number"] for i in issues}
    for c in client.paginate(f"/repos/{repo}/issues/comments", max_items=max_threads * 40):
        n = int(c["issue_url"].rsplit("/", 1)[-1])
        if n in wanted:
            by_number.setdefault(n, []).append(c)
    closers = merged_pr_closers(client, repo)
    return [
        build_issue_thread(i, by_number.get(i["number"], []), closers.get(i["number"], []))
        for i in issues
    ]


def fetch_discussions(client: GitHubClient, repo: str, limit: int = 500) -> list[Thread]:
    owner, name = repo.split("/", 1)
    out: list[Thread] = []
    after: str | None = None
    try:
        while len(out) < limit:
            data = client.graphql(
                _DISCUSSIONS_QUERY, {"owner": owner, "name": name, "after": after}
            )
            page = data["repository"]["discussions"]
            out.extend(build_discussion_thread(n) for n in page["nodes"])
            if not page["pageInfo"]["hasNextPage"]:
                break
            after = page["pageInfo"]["endCursor"]
    except GitHubError:
        return out  # discussions disabled or token lacks access: issues still work
    return out


def fetch_one_issue(client: GitHubClient, repo: str, number: int) -> Thread | None:
    """Fetch a single thread (issue + comments + merged PRs that reference it)."""
    issue = client.get_json(f"/repos/{repo}/issues/{number}")
    if "pull_request" in issue:
        return None
    comments = list(client.paginate(f"/repos/{repo}/issues/{number}/comments"))
    merged: list[int] = []
    for ev in client.paginate(f"/repos/{repo}/issues/{number}/timeline", max_items=200):
        src = (ev.get("source") or {}).get("issue") or {}
        pr = src.get("pull_request") or {}
        if ev.get("event") == "cross-referenced" and pr.get("merged_at"):
            if number in closing_references(f"{src.get('title', '')}\n{src.get('body') or ''}"):
                merged.append(src["number"])
    return build_issue_thread(issue, comments, merged)


def search_candidates(
    client: GitHubClient, repo: str, variants: list[tuple[str, str]], per_query: int = 10
) -> list[int]:
    """Issue numbers from GitHub search, in the order the variants were tried (deduplicated)."""
    seen: dict[int, None] = {}
    for search_type, text in variants:
        params: dict[str, Any] = {"q": f"repo:{repo} is:issue {text}", "per_page": per_query}
        if search_type != "lexical":
            params["search_type"] = search_type
        try:
            body = client.get_json("/search/issues", params, resource="search")
        except GitHubError:
            continue
        for item in body.get("items", []):
            seen.setdefault(item["number"], None)
    return list(seen)


# -- documents --------------------------------------------------------------------------------


def _pieces(text: str, limit: int = MAX_PIECE_CHARS) -> list[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) > limit:
            out.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
        while len(cur) > limit * 1.5:
            out.append(cur[:limit].strip())
            cur = cur[limit:]
    if cur.strip():
        out.append(cur.strip())
    return out


def thread_documents(thread: Thread, repo: str) -> list[Document]:
    """One Document per body/comment piece, each carrying thread-level metadata."""
    resolved_urls = [f"https://github.com/{repo}/pull/{n}" for n in thread.resolved_by]
    base = {
        "repo": repo,
        "kind": thread.kind,
        "number": thread.number,
        "title": thread.title,
        "state": thread.state,
        "state_reason": thread.state_reason,
        "resolution": thread.resolution,
        "resolved_by": thread.resolved_by,
        "resolved_by_urls": resolved_urls,
        "duplicate_of": thread.duplicate_of,
        "labels": thread.labels,
        "thread_url": thread.url,
    }
    docs = [
        Document(
            page_content=piece,
            metadata={**base, "part": "body", "url": thread.url, "source": thread.url},
        )
        for piece in _pieces(thread.body) or [""]
    ]
    for c in thread.comments:
        for piece in _pieces(c.body):
            part = "answer" if c.is_answer else "comment"
            docs.append(
                Document(
                    page_content=piece,
                    metadata={**base, "part": part, "url": c.url, "source": c.url},
                )
            )
    return docs
