import base64
import io
import tarfile

import pytest

from langchain_github._chunk import Chunker, language_for
from langchain_github._tree import (
    BlobStore,
    PathFilter,
    TreeEntry,
    decode_blob,
    decode_text,
    extract_tarball,
    permalink,
    select_entries,
)


def entry(path: str, size: int = 100) -> TreeEntry:
    return TreeEntry(path, "sha-" + path, size)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("README.md", True),
        ("src/app/main.py", True),
        ("node_modules/x/index.js", False),
        ("pkg/node_modules/x.js", False),  # excluded at any depth
        ("assets/logo.png", False),  # binary
        ("poetry.lock", False),
        ("web/app.min.js", False),
        ("huge.py", False),  # over size cap, set below
    ],
)
def test_default_filter(path, expected):
    f = PathFilter(max_file_size=1000)
    size = 5000 if path == "huge.py" else 100
    assert f.accepts(entry(path, size)) is expected


def test_include_globs_and_depth_rules():
    f = PathFilter(include=["docs/**", "*.md"])
    assert f.accepts(entry("docs/guide/intro.rst"))
    assert f.accepts(entry("a/b/c/NOTES.md"))  # slash-less pattern matches at any depth
    assert not f.accepts(entry("src/main.py"))


def test_extension_allow_list_accepts_with_or_without_dot():
    f = PathFilter(extensions=["py", ".md"])
    assert f.accepts(entry("a.py")) and f.accepts(entry("b.MD"))
    assert not f.accepts(entry("c.go"))


def test_custom_exclude_replaces_defaults():
    f = PathFilter(exclude=["tests/"])
    assert not f.accepts(entry("tests/test_a.py"))
    assert not f.accepts(entry("pkg/tests/test_a.py"))
    assert f.accepts(entry("node_modules/x.js"))  # defaults are not applied when overridden


def test_select_entries_skips_dirs_symlinks_and_submodules():
    tree = [
        {"path": "a.py", "type": "blob", "mode": "100644", "sha": "1", "size": 5},
        {"path": "link", "type": "blob", "mode": "120000", "sha": "2", "size": 5},
        {"path": "src", "type": "tree", "mode": "040000", "sha": "3"},
        {"path": "vendor/sub", "type": "commit", "mode": "160000", "sha": "4"},
        {"path": "b.py", "type": "blob", "mode": "100755", "sha": "5", "size": 5},
    ]
    assert [e.path for e in select_entries(tree, PathFilter())] == ["a.py", "b.py"]


def test_decode_text_rejects_binary_and_strips_bom():
    assert decode_text(b"\xef\xbb\xbfhello") == "hello"
    assert decode_text(b"ab\x00cd") is None
    assert decode_text(b"\xff\xfe\xfa") is None
    assert decode_blob(base64.b64encode(b"hi\n").decode()) == "hi\n"


def make_tarball(files: dict[str, bytes], prefix="owner-repo-abc1234") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo(f"{prefix}/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "a.py"
        tar.addfile(link)
    return buf.getvalue()


def test_extract_tarball_strips_prefix_and_only_returns_wanted():
    data = make_tarball({"a.py": b"print(1)\n", "docs/b.md": b"# b", "img.bin": b"\x00\x01"})
    got = extract_tarball(data, {"a.py": "s1", "img.bin": "s2", "link": "s3"})
    assert got == {"a.py": "print(1)\n", "img.bin": None}


def test_permalink_formats():
    assert permalink("o/r", "abc", "a b/c.py", 3, 3).endswith("/blob/abc/a%20b/c.py#L3")
    assert permalink("o/r", "abc", "x.py", 3, 9).endswith("#L3-L9")
    assert permalink("o/r", "abc", "x.py", None, None).endswith("/blob/abc/x.py")


def test_blob_store_memory_and_disk_roundtrip(tmp_path):
    store = BlobStore(tmp_path)
    assert store.get("s1") == (False, None)
    store.put("s1", "text")
    store.put("s2", None)
    fresh = BlobStore(tmp_path)  # new process, same cache dir
    assert fresh.get("s1") == (True, "text")
    assert fresh.get("s2") == (True, None)  # remembered binary file is a hit, not a refetch


SOURCE = "\n".join(
    ["import os", ""]
    + [f"def func_{i}(x):\n    # step {i}\n    return x + {i}\n" for i in range(60)]
)


def test_chunk_line_ranges_map_back_to_the_file():
    chunks = Chunker(chunk_size=400, chunk_overlap=40).split(
        SOURCE, repo="o/r", ref="main", commit_sha="abc", path="pkg/m.py", blob_sha="b1"
    )
    assert len(chunks) > 3
    lines = SOURCE.splitlines()
    for c in chunks:
        a, b = c.metadata["start_line"], c.metadata["end_line"]
        window = "\n".join(lines[a - 1 : b])
        assert c.page_content.strip().splitlines()[0].strip() in window
        assert c.metadata["url"].endswith(f"#L{a}-L{b}") or a == b
        assert c.metadata["commit_sha"] == "abc" and c.metadata["language"] == "python"
        assert c.metadata["kind"] == "code"


def test_markdown_is_doc_kind_and_unknown_suffix_falls_back():
    md = Chunker().split(
        "# T\n\ntext", repo="o/r", ref="m", commit_sha="c", path="README.md", blob_sha="x"
    )
    assert md[0].metadata["kind"] == "doc" and md[0].metadata["start_line"] == 1
    txt = Chunker().split(
        "hello", repo="o/r", ref="m", commit_sha="c", path="LICENSE", blob_sha="y"
    )
    assert txt[0].metadata["language"] is None
    assert language_for("a/b.tsx") == "ts"
