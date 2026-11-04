"""A tiny in-memory GitHub repo served through respx, for loader/retriever tests."""

from __future__ import annotations

import base64
import hashlib
import io
import tarfile

import httpx
import respx

API = "https://api.github.com"


def blob_sha(text: str) -> str:
    data = text.encode()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeRepo:
    def __init__(self, repo: str, files: dict[str, str], *, truncated: bool = False) -> None:
        self.repo = repo
        self.files = dict(files)
        self.truncated = truncated
        self.commit = 0
        self.calls: list[str] = []

    @property
    def sha(self) -> str:
        return hashlib.sha1(f"{self.repo}{self.commit}".encode()).hexdigest()

    def push(self, **changes: str | None) -> None:
        """Create a new commit; ``name=None`` deletes. Keys use ``__`` for ``/``."""
        self.commit += 1
        for key, val in changes.items():
            path = key.replace("__", "/").replace("_dot_", ".")
            if val is None:
                self.files.pop(path, None)
            else:
                self.files[path] = val

    def _tree_nodes(self) -> list[dict]:
        return [
            {
                "path": p,
                "type": "blob",
                "mode": "100644",
                "sha": blob_sha(t),
                "size": len(t.encode()),
            }
            for p, t in self.files.items()
        ]

    def tarball(self) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for path, text in self.files.items():
                data = text.encode()
                info = tarfile.TarInfo(f"{self.repo.replace('/', '-')}-{self.sha[:7]}/{path}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    def mount(self, router: respx.MockRouter) -> None:
        r = self.repo

        def etag_reply(request: httpx.Request, key: str, build) -> httpx.Response:
            tag = f'"{key}"'
            if request.headers.get("if-none-match") == tag:
                self.calls.append(f"304 {request.url.path}")
                return httpx.Response(304)
            self.calls.append(f"200 {request.url.path}")
            return build(tag)

        router.get(f"{API}/repos/{r}").mock(
            side_effect=lambda req: etag_reply(
                req,
                "info",
                lambda t: httpx.Response(200, json={"default_branch": "main"}, headers={"etag": t}),
            )
        )
        router.get(f"{API}/repos/{r}/commits/main").mock(
            side_effect=lambda req: etag_reply(
                req, self.sha, lambda t: httpx.Response(200, text=self.sha, headers={"etag": t})
            )
        )

        def tree(req: httpx.Request) -> httpx.Response:
            self.calls.append(f"200 {req.url.path}")
            if self.truncated and "recursive" in req.url.params:
                return httpx.Response(200, json={"sha": self.sha, "tree": [], "truncated": True})
            if self.truncated:  # level-by-level walk: put everything one level down
                nodes = self._tree_nodes()
                if req.url.path.endswith("/trees/" + self.sha):
                    return httpx.Response(
                        200,
                        json={
                            "tree": [
                                {"path": "all", "type": "tree", "sha": "sub", "mode": "040000"}
                            ]
                        },
                    )
                return httpx.Response(200, json={"tree": nodes})
            return httpx.Response(
                200, json={"sha": self.sha, "tree": self._tree_nodes(), "truncated": False}
            )

        router.get(url__regex=rf"{API}/repos/{r}/git/trees/.+").mock(side_effect=tree)

        def blob(req: httpx.Request) -> httpx.Response:
            self.calls.append(f"200 {req.url.path}")
            sha = req.url.path.rsplit("/", 1)[-1]
            by_sha = {blob_sha(t): t for t in self.files.values()}
            content = base64.b64encode(by_sha[sha].encode()).decode()
            return httpx.Response(200, json={"sha": sha, "encoding": "base64", "content": content})

        router.get(url__regex=rf"{API}/repos/{r}/git/blobs/.+").mock(side_effect=blob)

        def archive(req: httpx.Request) -> httpx.Response:
            self.calls.append(f"200 {req.url.path}")
            return httpx.Response(200, content=self.tarball())

        router.get(url__regex=rf"{API}/repos/{r}/tarball/.+").mock(side_effect=archive)

    def count(self, fragment: str) -> int:
        return sum(1 for c in self.calls if c.startswith("200") and fragment in c)
