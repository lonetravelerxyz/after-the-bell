"""Offline replay check for judge_b.sh: rerun logged replays from the cached LLM replies and compare.

  uv run python scripts/judge_replay.py                  # every replay run shipped in out/, full length
  uv run python scripts/judge_replay.py --quick 3        # every shipped replay, first 3 days only
  uv run python scripts/judge_replay.py RUN_DIR [...]    # named run directories

A full replay must end on the same `log_last_hash` as the shipped log (the hash chain covers every
record, so one equal hash means every decision, risk action, fill and mark is identical). A quick replay
covers the first N days of the shipped run and compares each record field by field with the shipped
record at the same position, ignoring only `run_id`, `hash` and `prev_hash` (the run id carries the end
date, so the hashes of a shorter run differ by construction).

Nothing here calls the network: SENTIMENT_OFFLINE=1 is set, and a prompt that is not in
cache/llm/ raises CacheMiss instead of calling the model. Replays are written under
judge-out/, never over the shipped runs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
WORK = ROOT / "judge-out"
DATES = re.compile(r"_(\d{4}-\d\d-\d\d)_(\d{4}-\d\d-\d\d)$")
IGNORE = {"run_id", "hash", "prev_hash"}


def shipped_runs() -> list[Path]:
    """Replay run directories in out/ that have a metrics.json with a logged last hash."""
    out = []
    for d in sorted(OUT.iterdir()) if OUT.exists() else []:
        m = d / "metrics.json"
        if d.is_dir() and d.name != "live" and m.exists() and DATES.search(d.name):
            if json.loads(m.read_text()).get("log_last_hash"):
                out.append(d)
    return out


def replay_cmd(d: Path, end: str | None, dest: Path) -> list[str]:
    m = json.loads((d / "metrics.json").read_text())
    start, logged_end = DATES.search(d.name).groups()
    cmd = [sys.executable, "-m", "sentiment.replay", "--variant", m["variant"], "--start", start,
           "--end", end or logged_end, "--offline", "--risk-version", m.get("risk_version") or "v0",
           "--out", str(dest)]
    if m.get("cards_from"):              # as logged; a pre-split "sentiment/out/<run>" resolves to out/<run>
        cmd += ["--cards-from", m["cards_from"]]
    return cmd


def records(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def check(d: Path, quick_days: int | None) -> bool:
    import pandas as pd
    m = json.loads((d / "metrics.json").read_text())
    start, _ = DATES.search(d.name).groups()
    end = None
    if quick_days:
        end = (pd.Timestamp(start) + pd.Timedelta(days=quick_days - 1)).strftime("%Y-%m-%d")
    dest = WORK / (d.name + (f"_first{quick_days}d" if quick_days else ""))
    t0 = time.time()
    r = subprocess.run(replay_cmd(d, end, dest), cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "SENTIMENT_OFFLINE": "1"})
    took = time.time() - t0
    if r.returncode:
        print(f"FAIL  {d.name}: replay exited {r.returncode} after {took:.0f}s\n{r.stderr[-2000:]}")
        return False
    new = records(dest / "log.jsonl")
    if not quick_days:
        got = json.loads((dest / "metrics.json").read_text())["log_last_hash"]
        shipped_last = records(d / "log.jsonl")[-1]["hash"]          # the shipped log must end where its metrics say
        ok = got == m["log_last_hash"] == shipped_last
        print(f"{'ok  ' if ok else 'FAIL'}  {d.name}: {len(new)} records, last hash {got[:16]}… "
              f"{'==' if ok else '!='} logged {m['log_last_hash'][:16]}… ({took:.0f}s)")
        return ok
    old = records(d / "log.jsonl")[: len(new)]
    diffs = [(i, k) for i, (a, b) in enumerate(zip(new, old))
             for k in sorted((set(a) | set(b)) - IGNORE) if a.get(k) != b.get(k)]
    ok = len(new) == len(old) and len(new) > 0 and not diffs
    print(f"{'ok  ' if ok else 'FAIL'}  {d.name}: first {quick_days} days, {len(new)} records identical to the "
          f"shipped log apart from run_id/hash ({took:.0f}s)" if ok else
          f"FAIL  {d.name}: first {quick_days} days, {len(new)} new vs {len(old)} shipped records; "
          f"first differences {diffs[:5]}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="*", help="run directories (default: every replay run shipped in out/)")
    p.add_argument("--quick", type=int, default=None, metavar="DAYS", help="replay only the first DAYS days")
    a = p.parse_args()
    os.environ["SENTIMENT_OFFLINE"] = "1"
    runs = [Path(r).resolve() for r in a.runs] if a.runs else shipped_runs()
    if not runs:
        sys.exit("no replay run with a logged hash found in out/")
    WORK.mkdir(exist_ok=True)
    results = [check(d, a.quick) for d in runs]
    print(f"{sum(results)} of {len(results)} replays reproduce the shipped log")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
