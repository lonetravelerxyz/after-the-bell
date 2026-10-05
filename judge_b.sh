#!/usr/bin/env bash
# One command, offline, no API keys: reproduce the sentiment agent's shipped evidence.
#   ./judge_b.sh           tests -> snapshot hashes -> log hash chains -> full offline replay of every shipped
#                          replay run (dev and test windows, baseline, LLM and placebos); each must end on its
#                          logged hash -> report regenerated from the shipped runs
#   ./judge_b.sh --quick   same, but each run is replayed over its first 3 days only and compared record by
#                          record with the shipped log
# Runs are replayed under judge-out/, never over the shipped logs.
# Every LLM reply the runs used is in cache/llm/; SENTIMENT_OFFLINE=1 turns a cache miss into an
# error instead of a network call, so this never needs a Qwen key or a Bitget key.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
export SENTIMENT_OFFLINE=1
MODE=()
case "${1:-}" in
  --quick) MODE=(--quick 3) ;;
  "")      ;;
  *) echo "usage: ./judge_b.sh [--quick]"; exit 2 ;;
esac
uv sync -q --locked
echo "== tests";             uv run pytest -q tests
echo "== snapshot hashes";   uv run python -m sentiment.snapshot --check
echo "== log hash chains"
uv run python - <<'PY'
import glob, sys
from sentiment import ledger
logs = sorted(glob.glob("out/*/log.jsonl"))
bad = [p for p in logs if not ledger.verify(p)]
for p in logs:
    print(("BROKEN " if p in bad else "ok     ") + p)
sys.exit(1 if bad or not logs else 0)
PY
echo "== offline replays";   uv run python scripts/judge_replay.py "${MODE[@]+"${MODE[@]}"}"
echo "== report";            uv run python -m sentiment.report > /dev/null
uv run python - <<'PY'
import json
m = json.load(open("report/metrics.json"))
for line in m.get("summary", []):
    print("-", line)
print("full report: report/REPORT.md")
PY
