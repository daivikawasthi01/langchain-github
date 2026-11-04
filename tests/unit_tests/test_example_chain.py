"""The README's retrieve-then-answer pattern, with a fake chat model."""

import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

sys.path.insert(0, str(Path(__file__).parents[2] / "examples"))
from answer_with_llm import build_chain  # noqa: E402


class _Fixed(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [
            Document(page_content="Retries are set in Session.", metadata={"url": "https://x/y#L1"})
        ]


def test_chain_puts_passages_and_links_in_the_prompt():
    seen = []

    class Spy(GenericFakeChatModel):
        def _generate(self, messages, *a, **kw):
            seen.extend(m.content for m in messages)
            return super()._generate(messages, *a, **kw)

    llm = Spy(messages=iter([AIMessage(content="See https://x/y#L1")]))
    assert build_chain(_Fixed(), llm).invoke("where are retries set?") == "See https://x/y#L1"
    assert "[https://x/y#L1]" in seen[0] and "Retries are set in Session." in seen[0]
    assert seen[1] == "where are retries set?"
