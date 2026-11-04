"""Answer a question about a GitHub repo: retrieve passages, then let a chat model cite them.

GITHUB_TOKEN=$(gh auth token) ANTHROPIC_API_KEY=... \
    python examples/answer_with_llm.py pallets/click "how are env vars read for options?"

Any chat model works: set MODEL (default "anthropic:claude-sonnet-5-5"; needs
`pip install langchain` and the provider package, e.g. langchain-anthropic).
"""

from __future__ import annotations

import os
import sys

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

from langchain_github import GitHubRepoRetriever

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only the passages below. Cite the link of every passage you use. "
            "If they do not answer the question, say so.\n\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['url']}]\n{d.page_content}" for d in docs)


def build_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable[str, str]:
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    from langchain.chat_models import init_chat_model

    repo, question = sys.argv[1], " ".join(sys.argv[2:])
    llm = init_chat_model(os.environ.get("MODEL", "anthropic:claude-sonnet-5-5"))
    retriever = GitHubRepoRetriever(repo=repo, k=4, cache_dir=".cache/gh")
    print(build_chain(retriever, llm).invoke(question))
