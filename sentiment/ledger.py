"""Append-only JSONL decision log with a sha256 hash chain (CONTRACT: Log).

Each line is the canonical JSON of one record (sorted keys, no whitespace, non-JSON values via
str). `prev_hash` is the previous line's `hash` (64 zeros for the first line) and `hash` is the
sha256 of the canonical JSON of the record without `hash`. Editing, deleting or reordering any
line breaks `verify`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GENESIS = "0" * 64


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def record_hash(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(canonical(body).encode()).hexdigest()


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_hash = GENESIS
        if self.path.exists():
            lines = [ln for ln in self.path.read_text().splitlines() if ln.strip()]
            if lines:
                self.last_hash = json.loads(lines[-1])["hash"]

    def append(self, record: dict) -> dict:
        body = {k: v for k, v in record.items() if k not in ("hash", "prev_hash")}
        body["prev_hash"] = self.last_hash
        body = json.loads(canonical(body))            # normalise to what a reader will load back
        body["hash"] = record_hash(body)
        with self.path.open("a") as f:
            f.write(canonical(body) + "\n")
        self.last_hash = body["hash"]
        return body


def verify(path) -> bool:
    prev = GENESIS
    for ln in Path(path).read_text().splitlines():
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            return False
        if rec.get("prev_hash") != prev or rec.get("hash") != record_hash(rec):
            return False
        prev = rec["hash"]
    return True


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(p, "ok" if verify(p) else "BROKEN")
