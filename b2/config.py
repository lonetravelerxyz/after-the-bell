"""B2 crisis-derisk module: paths and pre-declared constants (spec docs/spec-B2-crisis-derisk.md)."""
from pathlib import Path

from sentiment.config import data_dir, OUT as _OUT

ROOT = Path(__file__).resolve().parents[1]
RAW = data_dir() / "raw" / "b2"                  # gitignored raw pulls
OUT = _OUT / "b2"                                # run outputs (gitignored; final run force-added)
SYMBOL = "BTCUSDT"
START = "2019-08-01"                             # first Bitget BTCUSDT perp 1H bar (measured 2026-09-26)
STABLES = {"tether": "USDT", "usd-coin": "USDC", "dai": "DAI", "ethena-usde": "USDe", "terrausd": "UST",
           "binance-usd": "BUSD", "frax": "FRAX", "true-usd": "TUSD"}
AUGMENTO_SOURCES = ("twitter", "reddit")
MAJOR_STABLES = ("USDT", "USDC", "DAI", "USDe", "UST")   # v1.1: C3 only on the largest coins (BUSD/FRAX/TUSD wind-downs are not crises)
DVOL_MULT = 1.25            # v1b C5: hourly DVOL close >= DVOL_MULT x trailing-30d median (95th pct of the ratio), with r_24h < 0
CRISIS_TOPICS = ("Panicking", "Fearful/Concerned", "Hacks", "Bad_news", "Selling", "Bearish")

# ---- v0 rules, fixed before the first run (literature values, not tuned) ----
R24_CRASH = -0.10          # C1: 24h return <= -10%
VOL_MULT = 2.0             # C2: 24h realised vol >= VOL_MULT x trailing-90d median, with r_24h < 0
DEPEG = 0.03               # C3: any major stablecoin >= 3% below $1 (daily close)
SOCIAL_Q = 0.90            # C4: crisis-topic share >= trailing-365d 90th percentile, with r_24h < 0
COOLDOWN_H = 7 * 24        # exit lasts at least this long
REENTRY_R24 = 0.0          # after the cooldown, re-enter once r_24h > this
TAKER = 0.0006
HALF_SPREAD = 0.0002
FUNDING_8H = 0.0001        # sensitivity line: 0.01% per 8h while long (~11%/yr); base case 0
MA_DAYS = 200              # trend benchmark
VOL_TARGET = 0.40          # vol-target benchmark (annualised), cap 1.0
