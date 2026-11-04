"""Check each golden label against the real file contents (independent of the retriever)."""

import json
import re
import sys

from langchain_github.loaders import GitHubRepositoryLoader

rows = [json.loads(line) for line in open(sys.argv[1] if len(sys.argv) > 1 else "eval/golden.jsonl")]
loaders = {}
bad = 0
for row in rows:
    L = loaders.setdefault(
        row["repo"], GitHubRepositoryLoader(row["repo"], split=False, cache_dir=".cache/gh")
    )
    if not hasattr(L, "_docs"):
        L._docs = {d.metadata["path"]: d.page_content for d in L.lazy_load()}
    missing = [p for p in row["expected"] if p not in L._docs]
    hit = [p for p in row["expected"] if p in L._docs and re.search(row["evidence"], L._docs[p])]
    status = "ok " if not missing and hit else "BAD"
    bad += status == "BAD"
    print(
        f"{status} {row['id']:11s} evidence in {len(hit)}/{len(row['expected'])} files"
        + (f"  MISSING {missing}" if missing else "")
    )
sys.exit(1 if bad else 0)
