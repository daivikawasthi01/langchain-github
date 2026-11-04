# langchain-github

LangChain retrievers and loaders for GitHub repositories, issues and discussions.

- **`GitHubRepoRetriever`**: ask a question about a repo and get the best code and doc passages, each with a permalink to the exact commit and lines. Works on any branch, tag or commit.
- **`GitHubIssuesRetriever`**: describe a problem or paste a traceback and get the matching issue, discussion or comment, with how it was resolved (`fixed`, `answered`, `wontfix`, `duplicate`, `open`) and the PR that closed it.
- **`GitHubRepositoryLoader`**, **`GitHubIssuesLoader`**: the same data as plain `Document`s.

## Installation

```bash
pip install -U langchain-github
export GITHUB_TOKEN=...   # optional; public repos need no scopes. Without it: 60 requests/hour.
```

## Usage

```python
from langchain_github import GitHubRepoRetriever, GitHubIssuesRetriever

docs = GitHubRepoRetriever(repo="pallets/click", k=3).invoke(
    "how are environment variables read for options?"
)
print(docs[0].metadata["url"])  # https://github.com/pallets/click/blob/<sha>/docs/options.md#L677-L690

issues = GitHubIssuesRetriever(repo="owner/name").invoke("requests hang when all sockets are in use")
m = issues[0].metadata
print(m["title"], m["resolution"], m["resolved_by_urls"], m["url"])
```

Both retrievers support `ainvoke`, and `k` can be set on the retriever or per call.

## Use with a chat model

```python
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

retriever = GitHubRepoRetriever(repo="pallets/click", k=4)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer from these passages and cite their links.\n\n{context}"),
    ("human", "{question}"),
])
format_docs = lambda docs: "\n\n".join(f"[{d.metadata['url']}]\n{d.page_content}" for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | init_chat_model("anthropic:claude-sonnet-5-5") | StrOutputParser()
)
print(chain.invoke("how are environment variables read for options?"))
```

## Key options

| Option | Effect |
|---|---|
| `ref=` | Branch, tag or commit (repo retriever). |
| `embeddings=` | Any LangChain `Embeddings`; turns on hybrid ranking. Recommended for natural-language questions. |
| `cache_dir=` | Cache downloads by blob SHA so a new commit refetches only changed files. |
| `include` / `exclude` / `extensions` | Filter files with gitignore-style globs. |
| `mode=` | Issues retriever: `index`, `search` or `auto` (default). |

Requests are throttled to GitHub's documented limits, honour `Retry-After`, and raise `GitHubRateLimitError` rather than hang.

## How well does it work?

- **Repo retriever:** clearly better than plain BM25 over the same chunks (hit@5 0.85 vs 0.80, MRR 0.73 vs 0.56 on 20 held-out questions). It ties a good dense-vector baseline; the value is the loader, permalinks and incremental sync.
- **Issues retriever:** about even with GitHub's own hybrid search on a small synthetic repo. On six large real repos, 0.34 hit@5 for duplicate-finding versus 0.24 for GitHub's hybrid search (41 questions).
- **Keyword mode** matches words, not meaning. Pass `embeddings=` if your questions are paraphrases.

Samples are small. Full tables, method and caveats: [docs/EVALUATION.md](https://github.com/daivikawasthi01/langchain-github/blob/main/docs/EVALUATION.md).

## Limits

- Read-only. Private repositories need a token that can read them.
- GitHub search caps at 1,000 results. Very large monorepos are capped by `max_archive_mb`.
- The index is held in memory per retriever instance.
- Not an official GitHub product.

## Development

```bash
pip install -e '.[dev]' && pytest              # offline tests
GITHUB_LIVE=1 GITHUB_TOKEN=... pytest -m live  # live tests
```

MIT licensed. Issues and pull requests welcome at [github.com/daivikawasthi01/langchain-github](https://github.com/daivikawasthi01/langchain-github).
