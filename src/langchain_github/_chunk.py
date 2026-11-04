"""Turn file text into line-anchored chunk Documents."""

from __future__ import annotations

from pathlib import PurePosixPath

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from ._tree import permalink

_LANGUAGE_BY_SUFFIX: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".mjs": Language.JS,
    ".cjs": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".c": Language.C,
    ".h": Language.C,
    ".cc": Language.CPP,
    ".cpp": Language.CPP,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".lua": Language.LUA,
    ".ex": Language.ELIXIR,
    ".exs": Language.ELIXIR,
    ".hs": Language.HASKELL,
    ".sol": Language.SOL,
    ".proto": Language.PROTO,
    ".html": Language.HTML,
    ".md": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
    ".rst": Language.RST,
    ".tex": Language.LATEX,
}


def language_for(path: str) -> str | None:
    lang = _LANGUAGE_BY_SUFFIX.get(PurePosixPath(path).suffix.lower())
    return lang.value if lang else None


class Chunker:
    """Splits files with language-aware separators and records exact line ranges."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitters: dict[Language | None, RecursiveCharacterTextSplitter] = {}

    def _splitter(self, lang: Language | None) -> RecursiveCharacterTextSplitter:
        if lang not in self._splitters:
            if lang:
                self._splitters[lang] = RecursiveCharacterTextSplitter.from_language(
                    lang,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    add_start_index=True,
                )
            else:
                self._splitters[lang] = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    add_start_index=True,
                )
        return self._splitters[lang]

    def split(
        self, text: str, *, repo: str, ref: str, commit_sha: str, path: str, blob_sha: str
    ) -> list[Document]:
        lang_name = language_for(path)
        lang = Language(lang_name) if lang_name else None
        base = {
            "repo": repo,
            "ref": ref,
            "commit_sha": commit_sha,
            "path": path,
            "blob_sha": blob_sha,
            "language": lang_name,
            "kind": "code" if lang and lang not in (Language.MARKDOWN, Language.RST) else "doc",
        }
        docs: list[Document] = []
        for piece in self._splitter(lang).create_documents([text]):
            start_idx = int(piece.metadata.get("start_index", 0))
            start = text.count("\n", 0, start_idx) + 1
            end = start + piece.page_content.count("\n")
            url = permalink(repo, commit_sha, path, start, end)
            docs.append(
                Document(
                    page_content=piece.page_content,
                    metadata={
                        **base,
                        "start_line": start,
                        "end_line": end,
                        "url": url,
                        "source": url,
                    },
                )
            )
        return docs
