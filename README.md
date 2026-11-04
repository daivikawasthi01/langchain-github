# langchain-github

LangChain retrievers and loaders for GitHub repositories, issues and discussions.

- **`GitHubRepoRetriever`**: ask a question about a repo and get the best code and doc passages, each with a permalink to the exact commit and lines. Works on any branch, tag or commit.
- **`GitHubIssuesRetriever`**: describe a problem or paste a traceback and get the matching issue, discussion or comment, with how it was resolved (`fixed`, `answered`, `wontfix`, `duplicate`, `open`) and the PR that closed it.
- **`GitHubRepositoryLoader`**, **`GitHubIssuesLoader`**: the same data as plain `Document`s.

## Installation

```bash
pip install -U langchain-github
```

No token is needed to try it, but without signing in GitHub allows only 60 requests an hour. To sign in once, install the [GitHub CLI](https://cli.github.com) and run:

```bash
gh auth login
```

The package picks up that login automatically. It also reads `GITHUB_TOKEN` or `GH_TOKEN`, or you can pass `token=`. Public repos need no special permissions.

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

| Option | Effect | Example | What the example does |
|---|---|---|---|
| `ref=` | Branch, tag or commit (repo retriever). | `GitHubRepoRetriever(repo="pallets/click", ref="8.1.7")` | Searches the code as it was at release 8.1.7, not the latest version. Links point to that exact commit. |
| `embeddings=` | Any LangChain `Embeddings`; turns on hybrid ranking. Recommended for natural-language questions. | `GitHubRepoRetriever(repo="pallets/click", embeddings=my_embeddings)` | Matches on meaning as well as words, so "env vars" also finds text about "environment variables". |
| `cache_dir=` | Cache downloads so a new commit refetches only changed files. | `GitHubRepoRetriever(repo="pallets/click", cache_dir=".cache/gh")` | Saves downloaded files in `.cache/gh`. The next run reuses them and only fetches what changed. |
| `include` / `exclude` / `extensions` | Filter files with gitignore-style globs. | `GitHubRepoRetriever(repo="pallets/click", include=["docs/**"], extensions=[".md"])` | Only searches Markdown files under `docs/`, so code and tests never show up in results. |
| `mode=` | Issues retriever: `index`, `search` or `auto` (default). | `GitHubIssuesRetriever(repo="pallets/click", mode="search")` | Asks GitHub's own search for candidate issues instead of downloading every issue first. |

Requests are throttled to GitHub's documented limits, honour `Retry-After`, and raise `GitHubRateLimitError` rather than hang.

## How well does it work?

- **Repos:** finds the right passages noticeably better than plain keyword search, and about as well as a standard vector store, while also giving you exact line links, any branch or tag, and quick updates when the repo changes.
- **Issues:** about as good as GitHub's own search at finding the right issue, and it adds what search does not: discussions, the closing pull request, and whether the problem was fixed, answered or closed as a duplicate.
- **Wording matters:** by default it matches words, so it works best when your question uses the project's own terms. For casual or paraphrased questions, pass `embeddings=` so it also matches meaning.

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

MIT licensed. 
Issues and pull requests welcome at [github.com/daivikawasthi01/langchain-github](https://github.com/daivikawasthi01/langchain-github).
