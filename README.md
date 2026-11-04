# langchain-github

Retrievers and loaders that put GitHub repositories, issues and discussions into LangChain.

- **`GitHubRepoRetriever`**: ask a question about a repo, get the best code and doc passages, each with a **permalink to the exact commit and lines**. Works on any branch, tag or commit.
- **`GitHubIssuesRetriever`**: describe a problem (or paste a traceback), get the issue, discussion or comment that answers it, with **how it was resolved** (`fixed`, `answered`, `wontfix`, `duplicate`, `open`) and the merged PR that closed it.
- **`GitHubRepositoryLoader`**, **`GitHubIssuesLoader`**: the same data as plain `Document`s for your own pipeline.

A retriever returns passages, not an answer. Pass them to an LLM to get a written answer (see below).

```bash
pip install langchain-github
export GITHUB_TOKEN=...   # any token; public repos need no scopes. Without one you get 60 requests/hour.
```

```python
from langchain_github import GitHubRepoRetriever, GitHubIssuesRetriever

docs = GitHubRepoRetriever(repo="pallets/click", k=3).invoke(
    "how are environment variables read for options?"
)
print(docs[0].metadata["url"])   # https://github.com/pallets/click/blob/<sha>/docs/arguments.md#L146-L192

issues = GitHubIssuesRetriever(repo="owner/name").invoke("requests hang when all sockets are in use")
m = issues[0].metadata
print(m["title"], m["resolution"], m["resolved_by_urls"], m["url"])
```

Async (`ainvoke`) is supported by both. `k` can be set on the retriever or per call (`invoke(q, k=10)`).

## What it adds over GitHub's own search

| | GitHub search | **langchain-github** |
|---|---|---|
| Any branch / tag / commit | default branch only | yes |
| Natural-language questions | code search ANDs every word | ranked (BM25, optional embeddings) |
| Results | whole files | passages with line permalinks |
| Whole-repo cost | n/a | one tarball request |
| Re-sync after a push | n/a | free ETag check, then only changed files |
| Issue resolution, closing PR, accepted answers, discussions | not returned | yes |

## Evidence (read the caveats)

Everything below is reproducible with `eval/`. Questions are paraphrases written without copying identifiers. Labels were checked against file contents (`eval/verify_golden.py`).

**Repo retriever, held-out set** (20 questions on psf/requests and pallets/flask, written after the design was fixed and never tuned against):

| system | hit@1 | hit@5 | MRR |
|---|---|---|---|
| plain BM25 over the same chunks | 0.35 | 0.80 | 0.56 |
| default splitter + dense vectors (bge-small) | 0.65 | 0.90 | 0.76 |
| **ours, BM25 only** | 0.60 | 0.85 | 0.73 |
| **ours, hybrid (same bge-small)** | 0.60 | 0.90 | 0.74 |

First set (30 questions, click / express / cobra): GitHub code search scored 0.00 hit@5 on the raw question and 0.27 with hand-stripped keywords; ours (hybrid) scored 0.97.

Honest reading: the code-aware ranking clearly beats plain BM25 (that is the part that is ours). Against a good dense-vector baseline we **tie**, we do not win; the value over a vector store is the loader (any ref, permalinks, incremental sync, no per-file requests), not the ranker. Samples are small (20 and 30 questions); one question is 3 to 5 points.

**Issues retriever** (26 questions over a synthetic repo with 28 issues, 13 fixing PRs and 6 Q&A discussions; `eval/golden_issues.jsonl`):

| system | hit@1 | hit@5 | MRR |
|---|---|---|---|
| GitHub issue search, keywords | 0.00 | 0.00 | 0.00 |
| GitHub `search_type=hybrid` | 0.85 | 0.90 | 0.88 |
| **ours, index mode** | 0.85 | 1.00 | 0.91 |
| ours, search mode | 0.85 | 0.90 | 0.88 |

Resolution and fixing-PR metadata were correct on every top-1 hit (17 of 17), but that comes from GitHub's data, not from ranking cleverness. Index mode also answers the 6 discussion questions that issue search cannot (0.96 hit@5 over all 26).

Honest reading: on retrieval quality alone, index mode is about even with GitHub's own hybrid search (a gap of two questions in 20 is noise). What you get is discussions, comment-level links, duplicate-to-canonical resolution and resolution metadata. The testbed corpus is synthetic and tiny (see the real-repo table below). In search mode (large repos) we deliberately keep GitHub's order because our re-ranking did not beat it.

**Issues retriever on large real repositories** (41 duplicate-finding questions across requests, pytest, pandas, scikit-learn, langchain and flask, 4k to 30k issues each; `eval/run_dups.py`). The question is a duplicate issue's title, the answer is the issue maintainers closed it against, and the duplicate itself is excluded for every system:

| system | hit@1 | hit@5 | MRR |
|---|---|---|---|
| GitHub issue search, keywords | 0.10 | 0.12 | 0.11 |
| GitHub `search_type=hybrid` | 0.07 | 0.24 | 0.15 |
| **ours, search mode** | 0.20 | 0.34 | 0.26 |
| ours, index mode (14 questions, requests + pytest only) | 0.14 | 0.36 | 0.23 |

Honest reading: search mode found the answer where hybrid did not on 4 questions and lost on 0 (small sample; the gain comes from adding keyword-variant candidates, not from re-ranking). Index mode got 5 of 14 right where hybrid and search mode got 3 of 14. Finding duplicates is hard: even the best system misses about two thirds. Index mode was only run on two repositories, and a few canonical threads may mention their duplicates in comments, which slightly helps every system that reads comments.

## Issues retriever modes

- `index` (repos with up to `max_index_threads=1500` issues): fetches all issues, comments, closing PRs and discussions with repo-wide listings, ranks locally, and fuses GitHub's hybrid order as one more signal. A free ETag probe per query re-fetches only when something changed.
- `search` (larger repos): GitHub search proposes up to `max_candidates=15` threads, which are fetched in full. Order stays GitHub's; you gain the best passage, comment link and resolution. Semantic/hybrid search is limited to 10 requests/minute by GitHub, and the client waits rather than failing.
- `auto` (default) counts issues once and picks.

## Getting a written answer

```python
from langchain_github import GitHubRepoRetriever
retriever = GitHubRepoRetriever(repo="pallets/click", k=4)
docs = retriever.invoke("how are environment variables read for options?")
context = "\n\n".join(f"[{d.metadata['url']}]\n{d.page_content}" for d in docs)
# send `context` and the question to any chat model; ask it to cite the links.
```

`examples/try_it.py <owner/repo> "<question>"` prints retrieved passages without any LLM.

## Options worth knowing

- `embeddings=` any LangChain `Embeddings` turns on hybrid ranking. No embedding dependency is installed for you.
- `cache_dir=` persists downloaded files by git blob SHA, so a new commit refetches only changed files.
- `include` / `exclude` / `extensions` / `max_file_size` filter files (gitignore-style globs). Generated and binary files are skipped by default.
- Requests are throttled to GitHub's documented limits, honour `Retry-After`, retry network errors with backoff, and raise `GitHubRateLimitError` instead of hanging when the wait would exceed `max_wait`.

## Limits

- Read-only. Public repositories need no token scopes; private repositories need a token that can read them.
- Search results come from GitHub, which caps search at 1,000 results and does not index forks by default.
- Very large monorepos: the tarball is capped by `max_archive_mb`; use `include` to narrow it.
- The index lives in memory per retriever instance (rebuilt from the on-disk cache).
- Not an official GitHub product; it uses the public REST and GraphQL APIs under GitHub's terms.

## Development

```bash
pip install -e '.[dev]'
pytest                                   # offline, ~100 tests
GITHUB_LIVE=1 GITHUB_TOKEN=... pytest -m live
python eval/run.py --golden eval/golden_heldout.jsonl --systems naive_bm25,ours_bm25
python eval/run_issues.py
```

MIT licensed.
