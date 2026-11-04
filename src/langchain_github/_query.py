"""Turning a user's question or pasted error into search input."""

from __future__ import annotations

import re

_TRACE = re.compile(r"Traceback \(most recent call last\)|^\s*File \".+\", line \d+", re.M)
_EXC = re.compile(
    r"^\s*((?:[A-Za-z_][\w.]*\.)?[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning|Exit|Timeout)):?\s*(.*)$",
    re.M,
)
_FRAME_FN = re.compile(r"^\s*File \".+\", line \d+, in (\w+)", re.M)
_NOISE = re.compile(r"0x[0-9a-fA-F]+|\b\d{4,}\b|[A-Za-z]:\\[^\s'\"]+|/[\w./-]+\.\w+")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

_STOP = frozenset(
    "the and for are but not you all can had her was one our out has have how its may new now "
    "see him two way who did get got she use that this with what when where which while from "
    "into than then them they will would could should about after again been being below "
    "between both does doing down during each few further here just more most other over same "
    "some such very your my me when why lots at same time left no".split()
)


def is_traceback(text: str) -> bool:
    return bool(_TRACE.search(text))


def normalize(query: str) -> str:
    """Collapse a pasted traceback to the part that identifies it; leave plain questions alone.

    The exception type and message plus the innermost function names are what issue titles and
    reports quote. File paths, line numbers and memory addresses never match anything.
    """
    query = query.strip()
    if not is_traceback(query):
        return " ".join(query.split())
    found = list(_EXC.finditer(query))
    exc = found[-1] if found else None
    parts: list[str] = []
    if exc is not None:
        parts.append(f"{exc.group(1).rsplit('.', 1)[-1]} {_NOISE.sub(' ', exc.group(2))}")
    frames = [f for f in _FRAME_FN.findall(query) if f != "<module>"]
    parts.extend(frames[-3:])
    cleaned = " ".join(" ".join(parts).split())
    return cleaned or " ".join(_NOISE.sub(" ", query).split())[:300]


def keywords(query: str, limit: int = 8) -> list[str]:
    """Distinctive words, longest first, for GitHub's AND-matching lexical search."""
    seen: dict[str, None] = {}
    for w in _WORD.findall(query):
        lw = w.lower()
        if lw not in _STOP:
            seen.setdefault(lw, None)
    return sorted(seen, key=lambda w: (-len(w), w))[:limit]


def search_variants(query: str, max_variants: int = 4) -> list[tuple[str, str]]:
    """``(search_type, text)`` pairs to run against GitHub issue search, best first.

    GitHub's lexical search requires every word to match, so a paraphrased question finds nothing
    until it is cut down to a few distinctive words.
    """
    text = normalize(query)
    out: list[tuple[str, str]] = [("hybrid", text[:256])]
    kws = keywords(text)
    for n in (5, 3):
        if len(kws) >= n:
            out.append(("lexical", " ".join(kws[:n])))
    if len(kws) >= 2:
        out.append(("lexical", " ".join(kws[:2])))
    deduped: list[tuple[str, str]] = []
    for item in out:
        if item not in deduped:
            deduped.append(item)
    return deduped[:max_variants]
