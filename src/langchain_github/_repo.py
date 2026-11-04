"""Parsing the ``repo`` argument."""

from __future__ import annotations

import re

_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
_URL_PREFIX = re.compile(r"^(?:https?://)?(?:www\.)?[\w.-]+\.[a-z]{2,}/", re.I)


def normalize_repo(repo: str) -> str:
    """Accept ``owner/name`` and tidy the forms people paste.

    ``https://github.com/owner/name``, ``github.com/owner/name``, ``owner/name.git`` and
    ``owner/name/`` all become ``owner/name``. Anything else raises ``ValueError`` with an example.
    """
    if not isinstance(repo, str):
        raise ValueError(f"repo must be a string like 'pallets/click', got {repo!r}")
    text = repo.strip()
    text = _URL_PREFIX.sub("", text, count=1).strip("/")
    if text.endswith(".git"):
        text = text[:-4]
    if not _REPO_RE.match(text):
        raise ValueError(
            f"repo must look like 'owner/name' (for example 'pallets/click'), got {repo!r}"
        )
    return text
