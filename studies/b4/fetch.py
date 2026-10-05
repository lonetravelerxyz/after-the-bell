"""Pull B4 inputs into data/raw/b4/ (public, no keys).

  fng.parquet      Alternative.me crypto fear & greed, daily (date = 00:00 UTC stamp), 2018-02-01 ->
  rtoken.parquet   Bitget spot RMSTRUSDT 1h bars from config.PAPER_START (ts ms = bar open)
  daily.parquet    split-adjusted daily closes of config.HIST_TICKERS (yfinance; run with
                   `uv run --with yfinance python -m studies.b4.fetch daily`)

Run: uv run python -m studies.b4.fetch [fng|rtoken|daily ...]
"""
from __future__ import annotations

import sys
import time

import pandas as pd
import requests

from studies.b4.config import HIST_START, HIST_TICKERS, PAPER_START, RAW, SYMBOL

BITGET = "https://api.bitget.com"
COLS = ["ts", "open", "high", "low", "close", "base_vol", "quote_vol"]


def _get(url: str, **params):
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            raise RuntimeError(f"{r.status_code} {r.text[:120]}")
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))


def fng() -> pd.DataFrame:
    rows = _get("https://api.alternative.me/fng/", limit=0, format="json")["data"]
    df = pd.DataFrame({"date": pd.to_datetime([int(r["timestamp"]) for r in rows], unit="s", utc=True),
                       "value": [int(r["value"]) for r in rows]})
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def rtoken() -> pd.DataFrame:
    start_ms = int(pd.Timestamp(PAPER_START, tz="UTC").timestamp() * 1000)
    end, out = int(time.time() * 1000), []
    while end > start_ms:
        body = _get(BITGET + "/api/v2/spot/market/history-candles", symbol=SYMBOL, granularity="1h",
                    endTime=end, limit=200)
        page = body.get("data") or []
        if not page:
            break
        out += page
        first = int(page[0][0])
        if first >= end or len(page) < 2:
            break
        end = first
        time.sleep(0.12)
    df = pd.DataFrame([r[:7] for r in out], columns=COLS).astype({"ts": "int64"})
    df = df.astype({c: "float64" for c in COLS[1:]}).drop_duplicates("ts").sort_values("ts")
    return df[df.ts >= start_ms].reset_index(drop=True)


def daily() -> pd.DataFrame:
    import yfinance as yf  # optional: not a project dependency
    px = yf.download(list(HIST_TICKERS), start=pd.Timestamp(HIST_START) - pd.Timedelta(days=400),
                     auto_adjust=True, progress=False)["Close"]
    return px.dropna(how="all")


def main(which: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in which or ["fng", "rtoken"]:
        df = {"fng": fng, "rtoken": rtoken, "daily": daily}[name]()
        df.to_parquet(RAW / f"{name}.parquet")
        print(f"{name}: {len(df)} rows -> {RAW / (name + '.parquet')}")


if __name__ == "__main__":
    main(sys.argv[1:])
