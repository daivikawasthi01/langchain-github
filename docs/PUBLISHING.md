# Publishing checklist

Nothing here has been run. Each step is outward-facing and needs the maintainer.

## 1. GitHub repo
```bash
gh repo create daivikawasthi01/langchain-github --public --source . --push
```
Then enable Actions. The `live` CI job uses the built-in `GITHUB_TOKEN` (public repos only).

## 2. TestPyPI dry run
```bash
python -m build && twine check dist/*
twine upload --repository testpypi dist/*          # needs a TestPyPI token
python -m venv /tmp/v && /tmp/v/bin/pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ langchain-github
```
Run `examples/try_it.py` from that venv.

## 3. PyPI
1. Check the name `langchain-github` is still free: https://pypi.org/project/langchain-github/ (it was on 2026-11-03).
2. On pypi.org add a *pending trusted publisher*: owner `daivikawasthi01`, repo `langchain-github`, workflow `release.yml`, environment `pypi`.
3. Set `version = "0.1.0"` in `pyproject.toml` and `__init__.py`, commit, tag `v0.1.0`, publish a GitHub release. `release.yml` uploads.

## 4. LangChain docs PR (`langchain-ai/docs`)
Add an integration page for each retriever under the retrievers integrations section, following the template of an existing community retriever page. Content to include:

- Overview table: class, package, serializable (no), JS support (no), downloads/version badges.
- Setup: `pip install -U langchain-github`, `GITHUB_TOKEN`.
- Instantiation and `invoke` examples from the README.
- A "use within a chain" example (retrieved passages passed to a chat model).
- API reference link (the repo README until docs are generated).

Suggested PR title: `docs: add GitHub retriever integration (langchain-github)`.
Body: what it does in two sentences, link to the package and the eval table in the README, state that it is community-maintained and read-only.
Do not open the PR before the package is on PyPI.
