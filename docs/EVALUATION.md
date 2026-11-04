# Evaluation and design notes

Everything here is reproducible with the scripts in `eval/`. Read the caveats: samples are small.

## Evidence

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

## Keyword mode vs embeddings

Without `embeddings=`, ranking is keyword-based (BM25), so it matches words, not meaning. If your question uses different words than the code and docs, it can miss. Example on `pallets/click`, top 6 results:

| question | keyword only | with embeddings (`bge-small-en-v1.5`) |
|---|---|---|
| "how are **env vars** read for options?" | `shell_completion.py` first (it contains the literal word `env`); `docs/options.md` absent | `docs/options.md` and `docs/arguments.md` in the top 5; shell-completion gone; test files still rank high |
| "**environment variable envvar** option value" | `core.py`, then `docs/options.md` x3 | same pages, slightly better order |

So keyword mode is fine when you use the project's own vocabulary (identifiers, error text, "envvar") and weak on casual paraphrase or abbreviations. For natural-language questions, pass any LangChain `Embeddings`:

```python
from fastembed import TextEmbedding            # any Embeddings works: OpenAI, Google, HuggingFace, ...
from langchain_core.embeddings import Embeddings

class Fast(Embeddings):
    def __init__(self): self.m = TextEmbedding("BAAI/bge-small-en-v1.5")
    def embed_documents(self, texts): return [v.tolist() for v in self.m.embed(texts)]
    def embed_query(self, t): return next(iter(self.m.embed([t]))).tolist()

retriever = GitHubRepoRetriever(repo="pallets/click", embeddings=Fast())
```

Cost to know about: embeddings are computed for every chunk when the index is first built (about 2.5 minutes on a laptop CPU for `pallets/click` with this small local model; a hosted embedding API is usually faster). Queries after that are fast. This is a single-question comparison, not a benchmark; the measured retrieval numbers above are the ones to rely on.
