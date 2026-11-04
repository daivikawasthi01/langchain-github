"""LangChain's standard retriever conformance tests.

``TestOffline`` runs the suite against an in-memory fake repo (no network, runs on every PR).
``TestLive`` runs the same suite against real GitHub; set ``GITHUB_LIVE=1`` (and optionally
``GITHUB_TOKEN``) to enable it.
"""

import os

import pytest
import respx
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_github import GitHubIssuesRetriever, GitHubRepoRetriever
from tests.unit_tests.fake_github import FakeRepo
from tests.unit_tests.fake_issues import REPO, FakeIssues

FILES = {
    "README.md": "# Acme\n\nA small web framework with retries and sessions.\n",
    "docs/config.md": "# Config\n\nRETRY_LIMIT controls how often requests are retried.\n",
    "src/http/retry.py": "def retry_request(send, retry_limit=3):\n    return send()\n",
    "src/auth/tokens.py": "def refresh_token(session):\n    return session.post('/oauth/token')\n",
    "src/db/pool.py": "class ConnectionPool:\n    def acquire(self):\n        return self._free.pop()\n",
}


class TestOffline(RetrieversIntegrationTests):
    @pytest.fixture(autouse=True)
    def _fake_github(self):
        repo = FakeRepo("acme/demo", FILES)
        with respx.mock(assert_all_called=False) as router:
            repo.mount(router)
            yield

    @property
    def retriever_constructor(self):
        return GitHubRepoRetriever

    @property
    def retriever_constructor_params(self):
        return {"repo": "acme/demo"}

    @property
    def retriever_query_example(self):
        return "retry requests session token connection pool"


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GITHUB_LIVE"), reason="set GITHUB_LIVE=1 to hit GitHub")
class TestLive(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self):
        return GitHubRepoRetriever

    @property
    def retriever_constructor_params(self):
        return {"repo": "pallets/click", "cache_dir": ".cache/gh"}

    @property
    def retriever_query_example(self):
        return "how are environment variables read for options"


class TestIssuesOffline(RetrieversIntegrationTests):
    @pytest.fixture(autouse=True)
    def _fake_github(self):
        fake = FakeIssues()
        with respx.mock(assert_all_called=False) as router:
            fake.mount(router)
            yield

    @property
    def retriever_constructor(self):
        return GitHubIssuesRetriever

    @property
    def retriever_constructor_params(self):
        return {"repo": REPO, "token": "t"}

    @property
    def retriever_query_example(self):
        return "pool exhausted, stream timeout, dark mode, retries"


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GITHUB_LIVE"), reason="set GITHUB_LIVE=1 to hit GitHub")
class TestIssuesLive(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self):
        return GitHubIssuesRetriever

    @property
    def retriever_constructor_params(self):
        return {"repo": "daivikawasthi01/langchain-github-testbed"}

    @property
    def retriever_query_example(self):
        return "my download never gives up even though I set a time limit"
