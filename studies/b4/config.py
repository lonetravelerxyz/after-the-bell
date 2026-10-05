"""B4 F&G-scaled MSTR long: paths and constants, fixed before the first run (2026-10-02).

Rule (user-specified, not tuned): long-only weight w = clip((75 - F&G) / 50, 0, 1):
full long at F&G <= 25, flat at F&G >= 75, linear in between. No shorts, no leverage.
"""
from pathlib import Path

from sentiment.config import data_dir, OUT as _OUT

ROOT = Path(__file__).resolve().parents[2]   # studies/b4/config.py -> project root
RAW = data_dir() / "raw" / "b4"                  # gitignored raw pulls
OUT = _OUT / "b4"                                # run outputs (gitignored)

FG_FULL, FG_FLAT = 25, 75                        # w = 1 at <= FG_FULL, 0 at >= FG_FLAT

# long history (daily US closes of the stock): from MicroStrategy's first BTC purchase,
# announced 2020-08-11 (21,454 BTC, $250M); the signal F&G(d) is held close(d) -> close(d+1)
HIST_START = "2020-08-11"
HIST_TICKERS = ("MSTR", "BTC-USD")

# paper log on the rToken: Bitget spot RMSTRUSDT 1h bars from its listing (symbol openTime
# 2026-06-02 06:58 UTC); weekday bars track the stock, weekend bars are the rToken's own tape
SYMBOL = "RMSTRUSDT"
PAPER_START = "2026-06-02 07:00"
START_EQUITY = 10_000.0
FEE = 0.001                                      # Bitget spot taker (symbols endpoint, takerFeeRate)
HALF_SPREAD = 0.0003                             # RMSTRUSDT book 2026-10-02: 165.28 / 165.38 = 6 bp
BAND = 0.05                                      # trade only when |target - current weight| >= 5 pp
FNG_LAG_H = 1                                    # F&G stamped d 00:00 UTC is used from d 01:00 UTC
FNG_LAG_H_SENS = 24                              # sensitivity: B's convention (available d+1 00:00)
