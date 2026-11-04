"""Repository tree selection, file decoding and the content-addressed blob store."""

from __future__ import annotations

import base64
import io
import re
import tarfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

DEFAULT_EXCLUDE: tuple[str, ...] = (
    ".git/",
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "venv/",
    ".tox/",
    ".next/",
    "dist/",
    "*.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "go.sum",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.svg",
    "*.ipynb",
)

BINARY_EXTENSIONS = frozenset(
    ".png .jpg .jpeg .gif .webp .ico .bmp .tiff .pdf .zip .gz .tgz .bz2 .xz .7z .rar .tar "
    ".jar .war .class .pyc .pyo .so .dylib .dll .exe .bin .o .a .woff .woff2 .ttf .otf .eot "
    ".mp3 .mp4 .mov .avi .wav .ogg .webm .parquet .pkl .pickle .npy .npz .onnx .pt .pth .db "
    ".sqlite .wasm".split()
)

_SYMLINK_MODE = "120000"


@dataclass(frozen=True)
class TreeEntry:
    path: str
    sha: str
    size: int


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Translate a gitignore-style glob (``*``, ``**``, ``?``) into an anchored regex."""
    if pattern.endswith("/"):
        pattern += "**"
    out, i = [], 0
    while i < len(pattern):
        ch = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif ch == "*":
            out.append("[^/]*")
            i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(ch))
            i += 1
    return re.compile("".join(out) + r"\Z")


class PathFilter:
    """Include/exclude globs, extension allow-list and size cap for repository files.

    A pattern without a slash (``*.md``, ``docs``) matches a file or directory name at any depth,
    like a ``.gitignore`` line. A pattern ending in ``/`` matches a directory at any depth. A
    pattern naming a directory (``docs`` or ``src/click``) selects everything under it.
    """

    def __init__(
        self,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        extensions: Iterable[str] | None = None,
        max_file_size: int = 300_000,
    ) -> None:
        self._include = [self._compile(p) for p in include or []]
        self._exclude = [
            self._compile(p) for p in (DEFAULT_EXCLUDE if exclude is None else exclude)
        ]
        self._extensions = (
            {e if e.startswith(".") else f".{e}" for e in extensions} if extensions else None
        )
        self.max_file_size = max_file_size

    @staticmethod
    def _compile(pattern: str) -> tuple[re.Pattern[str], bool]:
        anywhere = "/" not in pattern.rstrip("/") and not pattern.startswith("**")
        if pattern.endswith("/") and anywhere:
            pattern = f"**/{pattern}"
            anywhere = False
        return _glob_regex(pattern), anywhere

    @staticmethod
    def _hit(rule: tuple[re.Pattern[str], bool], path: str) -> bool:
        regex, anywhere = rule
        parts = path.split("/")
        if anywhere:
            return any(regex.match(part) for part in parts)
        return any(regex.match("/".join(parts[: i + 1])) for i in range(len(parts)))

    def prunes(self, directory: str) -> bool:
        """True if everything under ``directory`` is excluded, so it need not be listed."""
        probe = f"{directory.rstrip('/')}/\0"
        return any(self._hit(r, probe) for r in self._exclude)

    def accepts(self, entry: TreeEntry) -> bool:
        suffix = Path(entry.path).suffix.lower()
        if suffix in BINARY_EXTENSIONS or entry.size > self.max_file_size:
            return False
        if self._extensions is not None and suffix not in self._extensions:
            return False
        if self._include and not any(self._hit(r, entry.path) for r in self._include):
            return False
        return not any(self._hit(r, entry.path) for r in self._exclude)


def select_entries(tree: Sequence[dict[str, Any]], path_filter: PathFilter) -> list[TreeEntry]:
    """Pick regular files from a git tree listing that pass the filter, sorted by path."""
    picked = [
        TreeEntry(t["path"], t["sha"], t.get("size", 0))
        for t in tree
        if t.get("type") == "blob" and t.get("mode") != _SYMLINK_MODE
    ]
    return sorted((e for e in picked if path_filter.accepts(e)), key=lambda e: e.path)


def decode_text(raw: bytes) -> str | None:
    """Decode file bytes as UTF-8 text; ``None`` for binary content."""
    if b"\x00" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def decode_blob(content_b64: str) -> str | None:
    return decode_text(base64.b64decode(content_b64))


def extract_tarball(data: bytes, wanted: dict[str, str]) -> dict[str, str | None]:
    """Read ``wanted`` paths (path -> blob sha) out of a GitHub tarball.

    Returns path -> text, or ``None`` for files that are not valid text.
    """
    found: dict[str, str | None] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            _, _, path = member.name.partition("/")
            if path in wanted:
                fh = tar.extractfile(member)
                found[path] = decode_text(fh.read()) if fh else None
    return found


def permalink(
    repo: str,
    commit_sha: str,
    path: str,
    start: int | None,
    end: int | None,
    site: str = "https://github.com",
) -> str:
    url = f"{site}/{repo}/blob/{commit_sha}/{quote(path)}"
    if start is None:
        return url
    return f"{url}#L{start}" if end in (None, start) else f"{url}#L{start}-L{end}"


class BlobStore:
    """Content-addressed text cache keyed by git blob SHA.

    A file whose blob SHA is unchanged is never downloaded again, so re-indexing a new commit
    only fetches the files that changed. Binary files are remembered so they are not retried.
    """

    _SKIP = object()

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._mem: dict[str, object] = {}
        self._dir = Path(cache_dir) / "blobs" if cache_dir else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, sha: str) -> tuple[bool, str | None]:
        """Return ``(hit, text)``; ``text`` is ``None`` for a remembered binary file."""
        if sha in self._mem:
            val = self._mem[sha]
            return True, None if val is self._SKIP else str(val)
        if self._dir:
            if (self._dir / f"{sha}.skip").exists():
                self._mem[sha] = self._SKIP
                return True, None
            f = self._dir / sha
            if f.exists():
                text = f.read_text(encoding="utf-8")
                self._mem[sha] = text
                return True, text
        return False, None

    def put(self, sha: str, text: str | None) -> None:
        self._mem[sha] = self._SKIP if text is None else text
        if self._dir:
            if text is None:
                (self._dir / f"{sha}.skip").touch()
            else:
                (self._dir / sha).write_text(text, encoding="utf-8")
