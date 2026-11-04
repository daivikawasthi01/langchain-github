# langchain-github: progress

Plan: `/Users/daivikawasthi/.claude/plans/let-s-work-on-zoom-jolly-forest.md` (build order: GitHub, Render, YouTube subset, Reddit conditional).
Venv: `.venv` (Python 3.12). Run: `.venv/bin/pytest tests -q`. Live: `GITHUB_LIVE=1 GITHUB_TOKEN=$(gh auth token) .venv/bin/pytest -m live`.

## Status (2026-11-03)
| CP | State |
|---|---|
| 0 Verify + eval first | Done (golden sets written before tuning; testbed repo seeded) |
| 1 Scaffold + client | Done, incl. retry/backoff on dropped connections |
| 2 Loaders | Done: `GitHubRepositoryLoader`, `GitHubIssuesLoader` |
| 3 Retrievers + gates | Done: `GitHubRepoRetriever`, `GitHubIssuesRetriever` (see results) |
| 4 README + example + install test | README written; wheel builds and works from a clean venv; `examples/try_it.py` |
| 5 Ship | Committed locally with CI + release workflows and `docs/PUBLISHING.md`. **Not pushed or published** (needs user go-ahead) |

## Eval results
Repo retriever, first set (30 q, tuned against once): hybrid hit@5 0.97 vs code search 0.00/0.27, naive vectors 0.77, naive BM25 0.97.
Repo retriever, **held-out** (20 q, requests+flask, `eval/golden_heldout.jsonl`): plain BM25 0.35/0.80/0.56 (hit@1/hit@5/MRR), naive vectors 0.65/0.90/0.76, ours BM25 0.60/0.85/0.73, ours hybrid 0.60/0.90/0.74. Ours beats plain BM25; **ties** vector baseline.
Issues (26 q, synthetic testbed): GitHub keywords 0.00 hit@5; GitHub hybrid 0.85/0.90/0.88; ours index 0.85/1.00/0.91; ours search mode 0.85/0.90/0.88 (keeps GitHub order, because our re-rank was worse: 0.85). Resolution accuracy 17/17. Index mode design change (fusing GitHub's order as a signal) was made after the first run showed 0.85 hit@5; the test set is small and was seen, so treat the 1.00 as optimistic.
Real large repos (41 duplicate-finding q, `eval/run_dups.py`): GitHub hybrid 0.07/0.24/0.15, ours search 0.20/0.34/0.26 (4 wins, 0 losses vs hybrid); ours index 5/14 vs hybrid 3/14 on requests+pytest.
Not measured: private repos, embeddings on issues, repos with discussions at scale.

## Files
- `src/langchain_github/_client.py` sync+async httpx client, local limiters (search 30/min, semantic 10/min, code_search 10/min), Retry-After/reset backoff, ETag cache
- `_tree.py` PathFilter (gitignore-style globs), select_entries, tarball extraction, BlobStore, permalink
- `_chunk.py` language-aware splitting + exact line ranges
- `_index.py` code-aware tokenizer, own BM25 (Lucene IDF), RRF fusion with optional embeddings, overlap de-dup
- `loaders.py` GitHubRepositoryLoader (+ `snapshot()` / `fetch()` / `documents()` building blocks)
- `retrievers.py` GitHubRepoRetriever
- `eval/golden.jsonl` 30 questions / 3 repos, `eval/verify_golden.py`, `eval/run.py`
- tests: `tests/unit_tests` (fake GitHub via respx in `fake_github.py`), `tests/integration_tests/test_standard.py` (langchain-tests; offline + live)

## API facts verified live (2026-11-03)
- `GET /search/issues?search_type=hybrid|semantic` works with a PAT; response includes `search_type`. On "connection pool exhausted under load" in fastapi/fastapi, hybrid found #4231 "socket.accept() out of system resource" which lexical search missed. Result counts are small (1-3), so native hybrid is a **strong baseline** the issues retriever must beat.
- ETag/`If-None-Match` on tree and commit-sha endpoints returns 304 and does **not** consume the primary limit.
- `GET /repos/{r}/commits/{ref}` with `Accept: application/vnd.github.sha` returns the bare SHA.
- Recursive tree API fine for small repos (`truncated: false` for click); truncated fallback is implemented but only unit-tested.
- One `tarball` request fetches a whole repo: pallets/click 172 files -> 1331 chunks in 6 requests / ~3 s; re-run costs 2 free revalidation requests.
- GraphQL `search(type: DISCUSSION)` works and exposes `answer`; cost 1 point.
- GitHub code search: 0.00 hit@5 on natural-language questions (AND semantics), 0.27 with stripped keywords.
- `langchain-tests` has **no** retriever unit-test base; `RetrieversIntegrationTests` needs `invoke(q, None, k=N)` -> `k` must be a call kwarg.
- `rank_bm25` Okapi IDF goes negative for terms in >50% of docs, zeroing results on small repos. Replaced with an own BM25 using Lucene IDF.

## Decisions
- Repo-first (proven demand is repo->RAG, 44 issues on the LlamaIndex repo reader vs 7 on issues reader); issues retriever is secondary and must beat GitHub hybrid search or ship as a loader only.
- No `rank-bm25` dependency. Embeddings are optional (any LangChain `Embeddings`).
- Cache is content-addressed by blob SHA, so new commits only refetch changed files.
- Eval uses `fastembed` (BAAI/bge-small-en-v1.5) as a dev-only dependency, not a package dependency.

## Testbed
`daivikawasthi01/langchain-github-testbed` is seeded (manifest in `eval/testbed/manifest.json`). Re-run `python eval/run_issues.py` to evaluate.

## Open items
1. LICENSE file, CI workflow, git commit/tag, TestPyPI then PyPI, `langchain-ai/docs` PR (all outward-facing; need user go-ahead).
2. (done) search mode measured on large repos; langchain-tests suite added for issues retriever.
3. Next package: langchain-render.
