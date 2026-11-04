"""Print what the retriever returns for a question.

GITHUB_TOKEN=$(gh auth token) \
    python examples/try_it.py pallets/click "how do options read env vars?"
"""

import sys

from langchain_github import GitHubRepoRetriever

repo, question = sys.argv[1], " ".join(sys.argv[2:])
docs = GitHubRepoRetriever(repo=repo, k=3, cache_dir=".cache/gh").invoke(question)

print(f"Q: {question}\n")
for i, d in enumerate(docs, 1):
    m = d.metadata
    print(f"--- result {i}: {m['path']} lines {m['start_line']}-{m['end_line']}")
    print(m["url"])
    print(d.page_content[:500].rstrip(), "\n")
