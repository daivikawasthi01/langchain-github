import pytest


@pytest.fixture(autouse=True)
def _no_gh_cli(monkeypatch):
    """Keep tests independent of whoever is logged in to the GitHub CLI on this machine."""
    monkeypatch.setattr("langchain_github._client.shutil.which", lambda _name: None)
