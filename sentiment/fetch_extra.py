"""Extra raw inputs for project B: analyst ratings, insider trades, crypto F&G, index perps, top-ups.

MCP (via the read-only `sentiment/mcp_client.py`, vendored from A4): `equity_estimates_price_target` and
`equity_ownership_insider_trading` per UNIVERSE ticker, `crypto_sentiment_crypto_fear_greed`.
Bitget public REST: 1h history-candles of the two index perps (SP500USDT, NDX100USDT) into
`data/raw/perp/{SYM}.parquet` (same format as A4's perp files).

A4's `data/raw/{perp,spot,funding}` for the universe stop at its last fetch (2026-09-19); B needs
bars to REPLAY_END and beyond, so the universe's newer bars/settlements are written to
`data/raw/sentiment/topup/{perp,spot,funding}/` and merged by `sentiment.data` (A4's files are
never rewritten).

Run: uv run python -m sentiment.fetch_extra
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from sentiment.config import INDEX_PERPS, RAW, UNIVERSE, perp
from sentiment.mcp_client import BitgetMCP

OUT = RAW / "sentiment"
TOPUP = OUT / "topup"
BASE = "https://api.bitget.com"
START = pd.Timestamp("2026-06-01", tz="UTC")
COLS = ["ts", "open", "high", "low", "close", "base_vol", "quote_vol"]
_last = [0.0]


def get(path: str, **params) -> list:
    """GET a Bitget public v2 endpoint at <= 2 req/s with retries; returns `data`."""
    for attempt in range(5):
        wait = _last[0] + 0.5 - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        try:
            body = requests.get(BASE + path, params=params, timeout=20).json()
            if body.get("code") == "00000":
                return body["data"]
            raise RuntimeError(f"{body.get('code')} {body.get('msg')}")
        except Exception as e:  # noqa: BLE001 - retry any transport/API error
            if attempt == 4:
                raise RuntimeError(f"GET {path} {params}: {e}") from e
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def candles(kind: str, symbol: str, start_ms: int) -> pd.DataFrame:
    """1h candles from start_ms to now, paging backwards with endTime (A4 format: ts ms bar open)."""
    if kind == "spot":
        path, params = "/api/v2/spot/market/history-candles", {"symbol": symbol, "granularity": "1h"}
    else:
        path = "/api/v2/mix/market/history-candles"
        params = {"symbol": symbol, "productType": "USDT-FUTURES", "granularity": "1H"}
    end, out = int(time.time() * 1000), []
    while end > start_ms:
        page = get(path, **params, endTime=end, limit=200)
        if not page:
            break
        out.extend(page)
        first = int(page[0][0])
        if first >= end or len(page) < 2:
            break
        end = first
    if not out:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame([r[:7] for r in out], columns=COLS)
    df = df.astype({"ts": "int64"}).astype({c: "float64" for c in COLS[1:]})
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df[df.ts >= start_ms].reset_index(drop=True)


def funding(symbol: str, start_ms: int) -> pd.DataFrame:
    """Funding settlements newer than start_ms (history-fund-rate pages newest first)."""
    out = []
    for page in range(1, 20):
        rows = get("/api/v2/mix/market/history-fund-rate", symbol=symbol, productType="USDT-FUTURES",
                   pageSize=100, pageNo=page)
        out += rows
        if len(rows) < 100 or min(int(r["fundingTime"]) for r in rows) < start_ms:
            break
    df = pd.DataFrame(out, columns=["symbol", "fundingRate", "fundingTime"])
    df = df.rename(columns={"fundingRate": "rate", "fundingTime": "ts"}).drop(columns="symbol")
    df = df.astype({"rate": "float64", "ts": "int64"}).drop_duplicates("ts").sort_values("ts")
    return df[df.ts >= start_ms].reset_index(drop=True)


def _save(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.attrs = {}
    df.to_parquet(path, index=False)


def _flat(rows: list[dict]) -> pd.DataFrame:
    """MCP rows -> DataFrame; nested values become strings so parquet accepts them."""
    d = pd.DataFrame(rows)
    for c in d.columns:
        if d[c].map(lambda v: isinstance(v, (dict, list))).any():
            d[c] = d[c].map(lambda v: v if v is None or isinstance(v, str) else str(v))
    return d


def fetch_mcp() -> None:
    m = BitgetMCP()
    ratings, insider = [], []
    for t in UNIVERSE:
        r = m.query("equity_estimates_price_target", symbol=t, limit=1000)
        i = m.query("equity_ownership_insider_trading", symbol=t, limit=500)
        ratings += [{**x, "symbol": t} for x in r]
        insider += [{**x, "symbol": t} for x in i]
        rr = pd.to_datetime(pd.Series([x.get("rating_date") for x in r]), errors="coerce")
        ii = pd.to_datetime(pd.Series([x.get("filing_date") for x in i]), errors="coerce")
        print(f"{t}: ratings {len(r)} ({(rr >= '2026-06-01').sum()} since 6/1, oldest {rr.min()}), "
              f"insider {len(i)} ({(ii >= '2026-06-01').sum()} since 6/1, oldest {ii.min()})", flush=True)
    _save(_flat(ratings), OUT / "ratings.parquet")
    _save(_flat(insider), OUT / "insider.parquet")
    days = (pd.Timestamp.now("UTC") - START).days + 30
    f = _flat(m.query("crypto_sentiment_crypto_fear_greed", limit=days))
    _save(f, OUT / "crypto_fng.parquet")
    print(f"crypto F&G: {len(f)} days {f['date'].min()} -> {f['date'].max()}")


def fetch_bars() -> None:
    start_ms = int(START.timestamp() * 1000)
    for sym in INDEX_PERPS:
        df = candles("perp", sym, start_ms)
        _save(df, RAW / "perp" / f"{sym}.parquet")
        print(f"{sym}: {len(df)} bars {pd.to_datetime(df.ts.min(), unit='ms')} -> {pd.to_datetime(df.ts.max(), unit='ms')}")
    for t in UNIVERSE:
        s = perp(t)
        for kind, sym in (("perp", s), ("spot", "R" + s), ("funding", s)):
            base = RAW / kind / f"{sym}.parquet"
            since = int(pd.read_parquet(base, columns=["ts"]).ts.max()) - 2 * 3_600_000 if base.exists() else start_ms
            df = funding(sym, since) if kind == "funding" else candles(kind, sym, since)
            _save(df, TOPUP / kind / f"{sym}.parquet")
        print(f"top-up {t}: done", flush=True)


def main() -> None:
    fetch_mcp()
    fetch_bars()


if __name__ == "__main__":
    main()
