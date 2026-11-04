"""Issue and discussion threads: fetching, resolution, and conversion to Documents."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from ._client import GitHubClient, GitHubError, GitHubGraphQLError, GitHubNotFoundError

# "fixes #12", "closes owner/name#12" and "resolves https://github.com/owner/name/issues/12".
_REF = (
    r"(?:(?P<slug>[\w.-]+/[\w.-]+)#|https?://[^/\s]+/(?P<url>[\w.-]+/[\w.-]+)/issues/|#)"
    r"(?P<num>\d+)"
)
_CLOSES = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+" + _REF, re.I)
_DUPLICATE = re.compile(r"\bduplicate\s+of\s+" + _REF, re.I)
_REPO_IN_URL = re.compile(r"^https?://[^/]+/([^/]+/[^/]+)/(?:issues|pull|discussions)/\d+")


def repo_of_url(url: str) -> str | None:
    """``owner/name`` from an issue, pull request or discussion web URL."""
    m = _REPO_IN_URL.match(url or "")
    return m.group(1) if m else None


def _same_repo(match: re.Match[str], repo: str | None) -> bool:
    """True if the reference points at ``repo`` (references without an owner always do)."""
    named = match.group("slug") or match.group("url")
    return named is None or repo is None or named.lower() == repo.lower()


_DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 50, after: $after, orderBy: {field: UPDATED_AT, direction: DESC}) {
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
        """``fixed`` | ``answered`` | ``closed`` | ``wontfix`` | ``duplicate`` | ``open``.

        ``closed`` is an issue closed as completed with no linked pull request and no comments, so
        nothing says how it was resolved (it may have been fixed by a plain commit).
        """
        if self.kind == "discussion":
            return "answered" if any(c.is_answer for c in self.comments) else "open"
        if self.state == "open":
            return "open"
        if self.duplicate_of is not None or self.state_reason == "duplicate":
            return "duplicate"
        if self.state_reason == "not_planned":
            return "wontfix"
        if self.resolved_by:
            return "fixed"
        return "answered" if self.comments else "closed"


def closing_references(text: str | None, repo: str | None = None) -> list[int]:
    """Issue numbers ``text`` says it closes; with ``repo``, other repositories are ignored."""
    return [int(m.group("num")) for m in _CLOSES.finditer(text or "") if _same_repo(m, repo)]


def duplicate_target(text: str | None, repo: str | None = None) -> int | None:
    """The issue number ``text`` says it duplicates (the first one in ``repo``)."""
    for m in _DUPLICATE.finditer(text or ""):
        if _same_repo(m, repo):
            return int(m.group("num"))
    return None


def _duplicate_in(state: str | None, bodies: list[str | None], repo: str | None) -> int | None:
    """The issue a closed thread says it duplicates (the last such comment wins)."""
    dup = None
    if state == "closed":
        for body in bodies:
            dup = duplicate_target(body, repo) or dup
    return dup


def build_issue_thread(
    issue: dict[str, Any], comments: list[dict[str, Any]], merged_prs: list[int]
) -> Thread:
    repo = repo_of_url(issue.get("html_url", ""))
    dup = _duplicate_in(issue.get("state"), [c.get("body") for c in comments], repo)
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


_ISSUE_FIELDS = """
  number title body url state stateReason updatedAt
  labels(first: 20) { nodes { name } }
  comments(first: 50) { pageInfo { hasNextPage } nodes { databaseId url body } }
  closedByPullRequestsReferences(first: 10, includeClosedPrs: false) {
    nodes { number merged repository { nameWithOwner } }
  }
  timelineItems(first: 20, itemTypes: [CROSS_REFERENCED_EVENT]) {
    nodes {
      ... on CrossReferencedEvent {
        source { ... on PullRequest { number merged title body repository { nameWithOwner } } }
      }
    }
  }
"""

_ISSUES_QUERY = (
    """
query($owner: String!, $name: String!, $after: String, $states: [IssueState!],
      $since: DateTime, $order: IssueOrder!) {
  repository(owner: $owner, name: $name) {
    issues(first: 50, after: $after, states: $states, filterBy: {since: $since},
           orderBy: $order) {
      pageInfo { hasNextPage endCursor }
      nodes {"""
    + _ISSUE_FIELDS
    + """}
    }
  }
}
"""
)

_BATCH_SIZE = 20


def _batch_query(numbers: list[int]) -> str:
    aliases = "\n".join(
        f"i{int(n)}: issueOrPullRequest(number: {int(n)}) "
        f"{{ __typename ... on Issue {{ {_ISSUE_FIELDS} }} }}"
        for n in numbers
    )
    return (
        "query($owner: String!, $name: String!) "
        f"{{ repository(owner: $owner, name: $name) {{ {aliases} }} }}"
    )


def _in_repo(pr: dict[str, Any], repo: str) -> bool:
    return bool((pr.get("repository") or {}).get("nameWithOwner", repo).lower() == repo.lower())


def thread_from_graphql(client: GitHubClient, repo: str, node: dict[str, Any]) -> Thread:
    """Convert a GraphQL ``Issue`` node. Comments beyond the first page come from REST."""
    number = node["number"]
    if node["comments"]["pageInfo"]["hasNextPage"]:
        rest = client.paginate(f"/repos/{repo}/issues/{number}/comments")
        comments = [Comment(str(c["id"]), c["html_url"], c.get("body") or "") for c in rest]
    else:
        comments = [
            Comment(str(c["databaseId"]), c["url"], c.get("body") or "")
            for c in node["comments"]["nodes"]
        ]
    state = str(node["state"]).lower()
    merged = {
        pr["number"]
        for pr in node["closedByPullRequestsReferences"]["nodes"]
        if pr.get("merged") and _in_repo(pr, repo)
    }
    # GitHub only links a closing PR that targets the default branch. A backport that says
    # "Fixes #12" is found from the issue's cross-references instead, as the REST path does.
    for event in node.get("timelineItems", {}).get("nodes", []):
        pr = event.get("source") or {}
        if pr.get("merged") and _in_repo(pr, repo):
            if number in closing_references(f"{pr.get('title', '')}\n{pr.get('body') or ''}", repo):
                merged.add(pr["number"])
    return Thread(
        kind="issue",
        number=number,
        title=node.get("title") or "",
        body=node.get("body") or "",
        url=node["url"],
        state=state,
        state_reason=(node.get("stateReason") or "").lower() or None,
        labels=[lbl["name"] for lbl in node["labels"]["nodes"]],
        comments=comments,
        resolved_by=sorted(merged),
        duplicate_of=_duplicate_in(state, [c.body for c in comments], repo),
        updated_at=node.get("updatedAt", ""),
    )


def _use_graphql(client: GitHubClient) -> bool:
    """GraphQL needs a signed-in user; it also returns closing PRs without extra requests."""
    return client.core.authenticated and client.core.graphql_issues


def _issues_graphql(
    client: GitHubClient, repo: str, max_threads: int, state: str, since: str | None = None
) -> list[Thread]:
    """Issues newest-created first, or (with ``since``) those updated since then."""
    owner, name = repo.split("/", 1)
    states = None if state == "all" else [state.upper()]
    order = {"field": "UPDATED_AT" if since else "CREATED_AT", "direction": "DESC"}
    out: list[Thread] = []
    after: str | None = None
    while len(out) < max_threads:
        data = client.graphql(
            _ISSUES_QUERY,
            {
                "owner": owner,
                "name": name,
                "after": after,
                "states": states,
                "since": since,
                "order": order,
            },
        )
        page = data["repository"]["issues"]
        out.extend(thread_from_graphql(client, repo, n) for n in page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    return out[:max_threads]


def fetch_threads(client: GitHubClient, repo: str, numbers: list[int]) -> list[Thread]:
    """Full threads for these issue numbers, in order. Pull requests and missing numbers are
    skipped. Signed in, up to 20 threads come back in one request."""
    if _use_graphql(client):
        owner, name = repo.split("/", 1)
        by_number: dict[int, Thread] = {}
        try:
            for i in range(0, len(numbers), _BATCH_SIZE):
                chunk = numbers[i : i + _BATCH_SIZE]
                data = client.graphql(
                    _batch_query(chunk), {"owner": owner, "name": name}, partial=True
                )["repository"]
                for n in chunk:
                    node = data.get(f"i{n}")
                    if node and node.get("__typename") == "Issue":
                        by_number[n] = thread_from_graphql(client, repo, node)
            return [by_number[n] for n in numbers if n in by_number]
        except GitHubGraphQLError:
            client.core.graphql_issues = False
    out: list[Thread] = []
    for n in numbers:
        try:
            thread = fetch_one_issue(client, repo, n)
        except GitHubNotFoundError:  # deleted or transferred since the search
            continue
        if thread is not None:
            out.append(thread)
    return out


MAX_INCREMENTAL = 30  # more changed issues than this over REST is cheaper to rebuild in full


def fetch_changed_issues(client: GitHubClient, repo: str, since: str) -> list[Thread] | None:
    """Issues created or updated since ``since`` (an ``updated_at`` stamp), with everything the
    index needs. ``None`` means too many changed: rebuild instead."""
    if _use_graphql(client):
        try:
            limit = 10 * MAX_INCREMENTAL
            changed = _issues_graphql(client, repo, limit, "all", since)
        except GitHubGraphQLError:
            client.core.graphql_issues = False
        else:
            return None if len(changed) >= limit else changed
    listing = client.paginate(
        f"/repos/{repo}/issues",
        {"state": "all", "since": since, "sort": "updated", "direction": "desc"},
        max_items=3 * MAX_INCREMENTAL,
    )
    numbers = [i["number"] for i in listing if "pull_request" not in i]
    if len(numbers) > MAX_INCREMENTAL:
        return None
    return fetch_threads(client, repo, numbers)


def probe(client: GitHubClient, repo: str) -> str:
    """Newest ``updated_at`` across issues and PRs. One request, free when nothing changed."""
    items = client.get_json(
        f"/repos/{repo}/issues",
        {"state": "all", "sort": "updated", "direction": "desc", "per_page": 1},
    )
    return str(items[0]["updated_at"]) if items else ""


def count_threads(client: GitHubClient, repo: str) -> int:
    try:
        body = client.get_json(
            "/search/issues", {"q": f"repo:{repo} is:issue", "per_page": 1}, resource="search"
        )
    except GitHubError as exc:
        if exc.status_code == 422:  # search says this for a repo it cannot see
            raise GitHubNotFoundError(
                f"Repository {repo} was not found, or is private. Check the spelling; for a "
                "private repository sign in with `gh auth login` or set GITHUB_TOKEN."
            ).with_status(404) from exc
        raise
    return int(body.get("total_count", 0))


def merged_pr_closers(client: GitHubClient, repo: str, limit: int = 1000) -> dict[int, list[int]]:
    """Map issue number -> merged PR numbers that say they close it."""
    out: dict[int, list[int]] = {}
    for pr in client.paginate(f"/repos/{repo}/pulls", {"state": "closed"}, max_items=limit):
        if not pr.get("merged_at"):
            continue
        for n in closing_references(f"{pr.get('title', '')}\n{pr.get('body') or ''}", repo):
            out.setdefault(n, []).append(pr["number"])
    return out


def fetch_all_issues(
    client: GitHubClient, repo: str, *, max_threads: int, state: str = "all"
) -> list[Thread]:
    """Every issue with its comments and closing PRs.

    Signed in, this is GraphQL: about N/50 requests for everything. Signed out it uses the REST
    listings of issues, comments and merged pull requests (about 3N/100 requests).
    """
    if _use_graphql(client):
        try:
            return _issues_graphql(client, repo, max_threads, state)
        except GitHubGraphQLError:
            client.core.graphql_issues = False
    issues = [
        i
        for i in client.paginate(
            f"/repos/{repo}/issues", {"state": state}, max_items=max_threads * 2
        )
        if "pull_request" not in i
    ][:max_threads]
    if not issues:
        return []
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


def fetch_discussions(
    client: GitHubClient, repo: str, limit: int = 500, since: str | None = None
) -> list[Thread]:
    """All discussions, or none (with a warning) if they cannot be read.

    GitHub serves discussions only through GraphQL, which needs a signed-in user. Rate-limit and
    credential errors are raised; only GraphQL-level refusals (such as discussions being switched
    off for the repository) are turned into a warning. With ``since``, only discussions updated
    at or after that stamp are returned.
    """
    if not client.core.authenticated:
        warnings.warn(
            "Discussions were skipped because you are not signed in to GitHub. "
            "Run `gh auth login` or set GITHUB_TOKEN to include them.",
            stacklevel=2,
        )
        return []
    owner, name = repo.split("/", 1)
    out: list[Thread] = []
    after: str | None = None
    try:
        while len(out) < limit:
            data = client.graphql(
                _DISCUSSIONS_QUERY, {"owner": owner, "name": name, "after": after}
            )
            page = data["repository"]["discussions"]
            nodes = page["nodes"]
            if since:  # newest-updated first, so stop at the first one that is older
                fresh = [n for n in nodes if n.get("updatedAt", "") >= since]
                out.extend(build_discussion_thread(n) for n in fresh)
                if len(fresh) < len(nodes):
                    break
            else:
                out.extend(build_discussion_thread(n) for n in nodes)
            if not page["pageInfo"]["hasNextPage"]:
                break
            after = page["pageInfo"]["endCursor"]
    except GitHubGraphQLError as exc:
        warnings.warn(f"Discussions of {repo} were skipped: {exc}", stacklevel=2)
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
        if ev.get("event") != "cross-referenced" or not pr.get("merged_at"):
            continue
        if (repo_of_url(src.get("html_url", "")) or repo).lower() != repo.lower():
            continue  # a pull request in a fork or another repository
        text = f"{src.get('title', '')}\n{src.get('body') or ''}"
        if number in closing_references(text, repo):
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
        except GitHubError as exc:
            if exc.status_code == 422:  # this query form was rejected; try the next variant
                continue
            raise
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
    site = re.match(r"https?://[^/]+", thread.url)
    resolved_urls = [
        f"{site.group(0) if site else 'https://github.com'}/{repo}/pull/{n}"
        for n in thread.resolved_by
    ]
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
