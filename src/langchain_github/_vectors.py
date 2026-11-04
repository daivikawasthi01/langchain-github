"""Embedding vectors cached by text, optionally on disk so a restart does not re-embed."""

from __future__ import annotations

import hashlib
import re
import struct
from array import array
from pathlib import Path

from langchain_core.embeddings import Embeddings

_PROBE = "langchain-github model fingerprint"
_DIGEST = 20  # bytes of SHA-1


def _embed(embeddings: Embeddings, texts: list[str], batch: int = 64) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(embeddings.embed_documents(texts[i : i + batch]))
    return out


class VectorCache:
    """``text -> vector``, in memory and, with ``cache_dir``, in an append-only file.

    The file is named after the embedding model, found by embedding a fixed sentence and hashing
    the result, so switching models never reuses another model's vectors. Records are a 20-byte
    SHA-1 of the text followed by the vector as float32; a torn final record is ignored.
    """

    def __init__(self, embeddings: Embeddings, cache_dir: str | Path | None = None) -> None:
        self.embeddings = embeddings
        self._dir = Path(cache_dir) / "vectors" if cache_dir else None
        self._mem: dict[bytes, list[float]] = {}
        self._ready = self._dir is None
        self._path: Path | None = None
        self._dim = 0

    @staticmethod
    def key(text: str) -> bytes:
        return hashlib.sha1(text.encode("utf-8", "surrogatepass")).digest()

    # -- persistence ------------------------------------------------------------------------------

    def _open(self, probe: list[float]) -> None:
        assert self._dir is not None
        self._dir.mkdir(parents=True, exist_ok=True)
        model = next(
            (
                str(getattr(self.embeddings, attr))
                for attr in ("model", "model_name", "model_id")
                if isinstance(getattr(self.embeddings, attr, None), str)
            ),
            type(self.embeddings).__name__,
        )
        shape = repr([round(x, 3) for x in probe[:64]]) + type(self.embeddings).__qualname__
        fingerprint = hashlib.sha1(shape.encode()).hexdigest()[:12]
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model)[:40]
        self._dim = len(probe)
        self._path = self._dir / f"{slug}-{fingerprint}-{self._dim}.bin"
        self._ready = True
        if self._path.exists():
            self._load(self._path.read_bytes())

    def _load(self, data: bytes) -> None:
        size = _DIGEST + 4 * self._dim
        for off in range(0, len(data) - size + 1, size):
            vec = struct.unpack(f"{self._dim}f", data[off + _DIGEST : off + size])
            self._mem[data[off : off + _DIGEST]] = list(vec)

    def _append(self, keys: list[bytes], vectors: list[list[float]]) -> None:
        if self._path is None:
            return
        with self._path.open("ab") as fh:
            fh.write(
                b"".join(k + array("f", v).tobytes() for k, v in zip(keys, vectors, strict=True))
            )

    # -- lookup -----------------------------------------------------------------------------------

    def _missing(self, keys: list[bytes]) -> list[int]:
        seen: set[bytes] = set()
        todo = []
        for i, key in enumerate(keys):
            if key not in self._mem and key not in seen:
                seen.add(key)
                todo.append(i)
        return todo

    def _remember(self, keys: list[bytes], todo: list[int], fresh: list[list[float]]) -> None:
        new_keys = [keys[i] for i in todo]
        for key, vec in zip(new_keys, fresh, strict=True):
            self._mem[key] = vec
        if self._path is not None and all(len(v) == self._dim for v in fresh):
            self._append(new_keys, fresh)

    def vectors(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, embedding only texts not seen before."""
        if not self._ready:
            self._open(self.embeddings.embed_documents([_PROBE])[0])
        keys = [self.key(t) for t in texts]
        todo = self._missing(keys)
        if todo:
            self._remember(keys, todo, _embed(self.embeddings, [texts[i] for i in todo]))
        return [self._mem[k] for k in keys]
