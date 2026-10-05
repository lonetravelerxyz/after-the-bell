"""Project B constants (spec docs/spec-B-market-sentiment.md, contract docs/CONTRACT.md)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]    # the project dir: b-sentiment-agent/ here, the root of the public repo


def data_dir() -> Path:
    """Raw-data root (gitignored): ROOT/data if it exists, else the research monorepo's shared ROOT.parent/data."""
    return ROOT / "data" if (ROOT / "data").is_dir() or not (ROOT.parent / "data").is_dir() else ROOT.parent / "data"


_MONOREPO = (ROOT.parent / "docs" / "decisions" / "log.md").is_file() and (ROOT.parent / ".beads").is_dir()
RAW = data_dir() / "raw"                     # every data/raw reference goes through this
ENV_FILE = ROOT / ".env" if (ROOT / ".env").exists() or not _MONOREPO else ROOT.parent / ".env"  # ../.env: monorepo only
SNAP = ROOT / "snapshot"
CACHE = ROOT / "cache"
OUT = ROOT / "out"
REPORT_DIR, DOCS, PROMPTS = ROOT / "report", ROOT / "docs", ROOT / "sentiment" / "prompts"
PREREG_FILE, TRIALS_FILE = DOCS / "PREREG.md", ROOT / "trials.log"


def run_dir(p) -> Path:
    """A run dir as a metrics.json or the CLI names it; a pre-split cards_from "sentiment/out/<run>" -> OUT/<run>."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p
    if p.parts[:2] == ("sentiment", "out"):
        return OUT.joinpath(*p.parts[2:])
    return ROOT / p if (ROOT / p).exists() else p

UNIVERSE = ["NVDA", "AAPL", "GOOGL", "META", "AMZN", "TSLA", "MSTR", "COIN", "HOOD", "CRCL"]
INDEX_PERPS = ["SP500USDT", "NDX100USDT"]     # context only in v0, not traded
REPLAY_START, DEV_END, TEST_START, REPLAY_END = "2026-06-22", "2026-08-10", "2026-08-11", "2026-09-22"
DECISION_HOURS = (0, 4, 8, 12, 16, 20)       # UTC
CARD_LOOKBACK_H = 72

TAKER = 0.0006                               # same as A4 src/backtest.py
HALF_SPREAD_FALLBACK = 0.0005
SPREAD_FILE = ROOT / "static" / "quoted_spread.csv"

MAX_W_NAME, MAX_GROSS, MAX_NET = 0.15, 1.0, 0.30
STOP_LOSS_NAME = 0.05                        # v0 flat stop (kept so v0 runs reproduce; unused by v1)
STOP_COOLDOWN_H = 24
DAILY_KILL = 0.03
KILL_COOLDOWN_H = 24
OFF_HOURS_SCALE = 0.5
FUNDING_BLOCK = 0.001
START_EQUITY = 10_000.0

# Risk layer version (sentiment/risk.py). "v0" = flat per-name cap and flat STOP_LOSS_NAME stop, no beta
# step (every run before 2026-09-23). "v1" = vol-scaled per-name cap, vol-based stop, beta neutralisation.
# The v1 parameters below were fixed before any v1 run, not tuned on dev-window P&L.
# The version also picks the prompt (v0: prompts/decide.md, no risk context; v1: prompts/decide_v1.md + RISK
# table) and whether the state carries beta_60d; a v0 run reproduces the records of the runs before it.
# Override without editing code: env SENTIMENT_RISK_VERSION (live loop), or `--risk-version` on replay/live.
RISK_VERSION = "v1"
RISK_VERSION = os.environ.get("SENTIMENT_RISK_VERSION") or RISK_VERSION
VOL_CAP_FLOOR = 0.33                         # cap_i = MAX_W_NAME * clip(median vol_7d / vol_7d_i, VOL_CAP_FLOOR, 1)
STOP_K, STOP_FLOOR, STOP_CAP = 2.0, 0.03, 0.10   # stop at clip(STOP_K * vol_7d at entry, STOP_FLOOR, STOP_CAP)
BETA_NEUTRAL = True                          # v1: project targets onto beta . w = 0 before the net/gross caps
BETA_LOOKBACK_D, BETA_MIN_D = 60, 30         # beta_60d: trailing daily beta to the equal-weight universe basket
LLM_MODEL = "qwen3.8-max"                    # model id in every LLM cache key; env BITGET_QWEN_MODEL overrides


def perp(ticker: str) -> str:
    return f"{ticker}USDT"
