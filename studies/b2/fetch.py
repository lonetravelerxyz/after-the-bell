"""Pull the free inputs of the crisis-derisk module into data/raw/b2/ (all public, no keys).

  perp.parquet      Bitget BTCUSDT USDT-FUTURES 1H bars from config.START (ts ms = bar open)
  fng.parquet       Alternative.me fear & greed, daily, 2018-02-01 ->
  stables.parquet   DefiLlama daily stablecoin prices (peggedUSD assets in config.STABLES)
  augmento.parquet  Augmento 24H topic counts for bitcoin, twitter + reddit, 2017-01-01 -> (free: ~31d delay)

Run: uv run python -m studies.b2.fetch [perp|fng|stables|augmento ...]
"""
from __future__ import annotations

import sys
import time

import pandas as pd
import requests

from studies.b2.config import AUGMENTO_SOURCES, RAW, STABLES, START, SYMBOL

BITGET = "https://api.bitget.com"
COLS = ["ts", "open", "high", "low", "close", "base_vol", "quote_vol"]


def _get(url: str, **params):
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            raise RuntimeError(f"{r.status_code} {r.text[:120]}")
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))


def perp() -> pd.DataFrame:
    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end, out = int(time.time() * 1000), []
    while end > start_ms:
        body = _get(f"{BITGET}/api/v2/mix/market/history-candles", symbol=SYMBOL, productType="USDT-FUTURES",
                    granularity="1H", endTime=end, limit=200)
        page = body.get("data") or []
        if not page:
            break
        out.extend(page)
        first = int(page[0][0])
        if first >= end or len(page) < 2:
            break
        end = first
        time.sleep(0.25)
    df = pd.DataFrame([r[:7] for r in out], columns=COLS).astype({"ts": "int64"})
    df = df.astype({c: "float64" for c in COLS[1:]}).drop_duplicates("ts").sort_values("ts")
    df = df[df.ts >= start_ms].reset_index(drop=True)
    df.to_parquet(RAW / "perp.parquet", index=False)
    print(f"perp: {len(df)} bars {pd.to_datetime(df.ts.min(), unit='ms')} -> {pd.to_datetime(df.ts.max(), unit='ms')}")
    return df


def fng() -> pd.DataFrame:
    d = _get("https://api.alternative.me/fng/", limit=0, format="json")["data"]
    df = pd.DataFrame(d)[["timestamp", "value"]].astype({"timestamp": "int64", "value": "int64"})
    df["date"] = pd.to_datetime(df.timestamp, unit="s", utc=True).dt.normalize()
    df = df.sort_values("date")[["date", "value"]].reset_index(drop=True)
    df.to_parquet(RAW / "fng.parquet", index=False)
    print(f"fng: {len(df)} days {df.date.min().date()} -> {df.date.max().date()}")
    return df


def stables() -> pd.DataFrame:
    rows = _get("https://stablecoins.llama.fi/stablecoinprices")
    out = []
    for r in rows:
        for gid, sym in STABLES.items():
            v = (r.get("prices") or {}).get(gid)
            if v is not None:
                out.append({"date": pd.Timestamp(int(r["date"]), unit="s", tz="UTC").normalize(), "sym": sym, "price": float(v)})
    df = pd.DataFrame(out).sort_values(["date", "sym"]).reset_index(drop=True)
    df = df[df.date >= "2018-01-01"]
    df.to_parquet(RAW / "stables.parquet", index=False)
    print(f"stables: {len(df)} rows, {df.date.min().date()} -> {df.date.max().date()}, syms {sorted(df.sym.unique())}")
    return df


def augmento() -> pd.DataFrame:
    topics = _get("https://api.augmento.ai/v0.1/topics")
    names = [topics[str(i)] for i in range(len(topics))]
    frames = []
    for src in AUGMENTO_SOURCES:
        rows, start = [], "2017-01-01T00:00:00Z"
        while True:
            page = _get("https://api.augmento.ai/v0.1/events/aggregated", source=src, coin="bitcoin", bin_size="24H",
                        start_ptr=0, count_ptr=365, start_datetime=start, end_datetime="2026-12-31T00:00:00Z")
            if not page:
                break
            rows.extend(page)
            last = page[-1]["datetime"]
            if len(page) < 365:
                break
            start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            time.sleep(0.3)
        df = pd.DataFrame({"date": pd.to_datetime([r["datetime"] for r in rows], utc=True)})
        counts = pd.DataFrame([r["counts"] for r in rows], columns=names)
        df = pd.concat([df, counts], axis=1)
        df["source"] = src
        frames.append(df)
        print(f"augmento {src}: {len(df)} days {df.date.min().date()} -> {df.date.max().date()}")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(RAW / "augmento.parquet", index=False)
    return out




# ---- Deribit (public, no key): DVOL implied-vol index (2021-03-24 ->) and BTC-PERPETUAL funding (2019-08 ->) ----
DERIBIT = "https://www.deribit.com/api/v2/public"


def _deribit(method: str, **params):
    body = _get(f"{DERIBIT}/{method}", **params)
    if "error" in body:
        raise RuntimeError(str(body["error"])[:200])
    return body["result"]


def dvol() -> pd.DataFrame:
    """Hourly DVOL candles (ts ms, open, high, low, close) for BTC, paged backwards from now."""
    rows, end = [], int(time.time() * 1000)
    start = int(pd.Timestamp("2021-03-01", tz="UTC").timestamp() * 1000)
    while end > start:
        res = _deribit("get_volatility_index_data", currency="BTC", resolution=3600,
                       start_timestamp=max(start, end - 900 * 3600 * 1000), end_timestamp=end)
        data = res.get("data") or []
        if not data:
            break
        rows.extend(data)
        end = int(data[0][0]) - 1
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts").sort_values("ts")
    df.to_parquet(RAW / "dvol.parquet", index=False)
    print(f"dvol: {len(df)} hours {pd.to_datetime(df.ts.min(), unit='ms')} -> {pd.to_datetime(df.ts.max(), unit='ms')}")
    return df


def deribit_funding() -> pd.DataFrame:
    """Hourly funding records of BTC-PERPETUAL (interest_8h = the 8h rate in force, interest_1h = paid that hour)."""
    rows, t = [], int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    now = int(time.time() * 1000)
    step = 30 * 24 * 3600 * 1000
    while t < now:
        res = _deribit("get_funding_rate_history", instrument_name="BTC-PERPETUAL", start_timestamp=t, end_timestamp=min(now, t + step))
        rows.extend(res or [])
        t += step
        time.sleep(0.2)
    df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp").rename(columns={"timestamp": "ts"})
    df.to_parquet(RAW / "deribit_funding.parquet", index=False)
    print(f"deribit funding: {len(df)} hours {pd.to_datetime(df.ts.min(), unit='ms')} -> {pd.to_datetime(df.ts.max(), unit='ms')}")
    return df


def main(argv=None) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    what = (argv or sys.argv[1:]) or ["fng", "stables", "augmento", "perp", "dvol", "deribit_funding"]
    for w in what:
        globals()[w]()


if __name__ == "__main__":
    main()
