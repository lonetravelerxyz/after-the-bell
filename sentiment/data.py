"""Point-in-time data layer for project B (contract: sentiment/CONTRACT.md, Data).

`Store` answers every question "as of t": it never returns a bar, settlement, index value or item
whose availability time is after t. Availability rules:
  bar       open ts + 1h (close time)          news     published_at - 8h (stamps are UTC+8, see below)
  funding   settlement ts                      rating   rating_date + 1 day 13:30 UTC
  F&G day d d + 1 day 00:00 UTC                insider  filing_date + 1 day 03:00 UTC (P and S only)

Insider: EDGAR stamps Form 4s accepted up to 22:00 New York time with that day's filing date, which is
up to 02:00 UTC (EDT) or 03:00 UTC (EST) the next day; filing_date + 1 day 03:00 UTC is the first time
every filing of that date is public, so the first decision that can see one is 04:00 UTC.
`next_open` only fills from a bar that opens within one hour of t (a data gap gives None, not the
open of a bar hours later).

Sources: `sentiment/snapshot/` (frozen by `sentiment.snapshot`) when `snapshot=True` and it exists,
else `data/raw/` (A4's perp/spot/funding files merged with `data/raw/sentiment/topup/`).

Note on news timestamps: the MCP stamps `published_at` with a "Z" suffix, but the stamps are Beijing
(UTC+8) wall time. Evidence (2026-09-23): at 10:52 UTC the MCP already served an item stamped 17:01Z,
6.2h in the future; the Bitget UEX Daily stamped 09:20Z on 9/22 says it uses "September 22 morning"
quotes; the 9/17 "[Market Movement Analysis] ... INTC" item stamped 21:40Z (13:40 UTC after the
correction) follows INTC's 13:00 UTC jump. So news is available at published_at - 8h. In live mode the
loop also records when it first saw an item (`first_seen`), and availability is the later of the two.
"""
from __future__ import annotations

import bisect
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment.config import INDEX_PERPS, RAW, SNAP, UNIVERSE, perp

HOUR = pd.Timedelta(hours=1)
EMPTY = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
BAR_COLS = ["open", "high", "low", "close", "quote_vol"]
TOPUP = RAW / "sentiment" / "topup"
RATING_EN = {"买入": "Buy", "强力买进": "Strong Buy", "增持": "Overweight", "跑赢大盘": "Outperform",
             "好于板块": "Sector Outperform", "积极": "Positive", "中性": "Neutral", "持有": "Hold",
             "持股观望": "Hold", "市场持平": "Market Perform", "减持": "Underweight", "逊于大盘": "Underperform",
             "卖出": "Sell"}
ACTION_EN = {"维持": "maintains", "重申": "reiterates", "首次覆盖": "initiates", "调高评级": "upgrades",
             "下调评级": "downgrades", "假设": "assumes", "恢复": "resumes", "暂停评级": "suspends"}


# ---------------------------------------------------------------- loading

def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _bars_raw(kind: str, sym: str) -> pd.DataFrame | None:
    """A4's raw file merged with B's top-up (top-up wins on overlapping ts)."""
    name = ("R" + sym) if kind == "spot" else sym
    parts = [d for d in (_read(RAW / kind / f"{name}.parquet"), _read(TOPUP / kind / f"{name}.parquet")) if d is not None]
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True).drop_duplicates("ts", keep="last").sort_values("ts")
    return d.assign(sym=sym).reset_index(drop=True)


def load_raw() -> dict[str, pd.DataFrame]:
    """Every input from data/raw/ as long frames (bars/funding carry a `sym` = perp symbol column)."""
    syms = [perp(t) for t in UNIVERSE] + INDEX_PERPS
    out: dict[str, pd.DataFrame] = {}
    for kind, names in (("perp", syms), ("spot", [perp(t) for t in UNIVERSE]), ("funding", [perp(t) for t in UNIVERSE])):
        parts = [d for d in (_bars_raw(kind, s) for s in names) if d is not None]
        out[kind] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["sym", "ts"])
    out["news"] = _read(RAW / "news" / "news.parquet")
    for k in ("ratings", "insider", "crypto_fng"):
        out[k] = _read(RAW / "sentiment" / f"{k}.parquet")
    return {k: v for k, v in out.items() if v is not None}


def load_snapshot() -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in sorted(SNAP.glob("*.parquet"))}


# ---------------------------------------------------------------- items

def _ts(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _clean(v):
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _item(kind: str, at: pd.Timestamp, title: str, text: str, tickers, raw: dict) -> dict:
    iid = hashlib.sha1(f"{kind}|{at.isoformat()}|{title}".encode()).hexdigest()
    return {"id": iid, "kind": kind, "available_at": at, "title": title, "text": text, "tickers": tickers,
            "raw": {k: _clean(v) for k, v in raw.items()}}


def _num(v) -> str | None:
    v = _clean(v)
    return None if v is None else f"{float(v):g}"


def rating_available(rating_date) -> pd.Timestamp:
    return _ts(pd.Timestamp(str(rating_date)[:10])) + pd.Timedelta(days=1, hours=13, minutes=30)


INSIDER_LAG = pd.Timedelta(days=1, hours=3)     # 22:00 New York time on the filing date, EST or EDT


def insider_available(filing_date) -> pd.Timestamp:
    return _ts(pd.Timestamp(str(filing_date)[:10])) + INSIDER_LAG


NEWS_STAMP_OFFSET = pd.Timedelta(hours=8)       # MCP published_at is UTC+8 wall time labelled "Z"


def news_available(published_at, first_seen=None) -> pd.Timestamp:
    at = _ts(published_at) - NEWS_STAMP_OFFSET
    return at if first_seen is None or pd.isna(first_seen) else max(at, _ts(first_seen))


def fng_available(date) -> pd.Timestamp:
    return _ts(pd.Timestamp(str(date)[:10])) + pd.Timedelta(days=1)


def build_items(frames: dict[str, pd.DataFrame]) -> list[dict]:
    """News, rating and insider items, deduplicated by id, sorted by (available_at, id)."""
    items: list[dict] = []
    news = frames.get("news")
    if news is not None:
        for r in news.to_dict("records"):
            items.append(_item("news", news_available(r["published_at"], r.get("first_seen")), str(r["title"]), r.get("content") or "", None, r))
    rat = frames.get("ratings")
    if rat is not None:
        for r in rat[rat["symbol"].isin(UNIVERSE)].to_dict("records"):
            firm = r.get("analyst_firm") or r.get("rating_org") or "analyst"
            cur, prev = _clean(r.get("rating_current")), _clean(r.get("rating_previous"))
            cur_en, prev_en = RATING_EN.get(cur, cur), RATING_EN.get(prev, prev)
            act = r.get("action") or ""
            pt, pt0 = _num(r.get("price_target")), _num(r.get("price_target_previous"))
            title = (f"{firm} {ACTION_EN.get(act, act)} {r['symbol']}: "
                     f"{prev_en + ' -> ' if prev_en and prev_en != cur_en else ''}{cur_en or 'n/a'}"
                     f"{', PT ' + (pt0 + ' -> ' if pt0 and pt0 != pt else '') + pt if pt else ''}")
            text = (f"{r['rating_date']} {firm} {act} {r['symbol']} rating {prev} -> {cur}, "
                    f"price target {pt0} -> {pt}")
            items.append(_item("rating", rating_available(r["rating_date"]), title, text, [r["symbol"]], r))
    ins = frames.get("insider")
    if ins is not None:
        ins = ins[ins["symbol"].isin(UNIVERSE) & ins["transaction_type"].isin(["P", "S"])]
        for r in ins.to_dict("records"):
            verb = "buys" if r["transaction_type"] == "P" else "sells"
            qty, px = _num(r.get("securities_transacted")), _num(r.get("transaction_price"))
            title = (f"{r['symbol']} insider {r.get('owner_name')} ({r.get('owner_title') or r.get('ownership_type')}) "
                     f"{verb} {qty} shares @ {px} on {r.get('transaction_date')}")
            text = (f"Form 4 filed {r['filing_date']}: {title}; owns {_num(r.get('securities_owned'))} after. "
                    f"{r.get('filing_url') or ''}").strip()
            items.append(_item("insider", insider_available(r["filing_date"]), title, text, [r["symbol"]], r))
    uniq = {it["id"]: it for it in reversed(items)}           # first occurrence wins
    return sorted(uniq.values(), key=lambda it: (it["available_at"], it["id"]))


# ---------------------------------------------------------------- store

class Store:
    """Point-in-time accessor over the snapshot (default) or raw inputs; see module docstring."""

    def __init__(self, snapshot: bool = True):
        use_snap = snapshot and SNAP.exists() and any(SNAP.glob("*.parquet"))
        self.source = "snapshot" if use_snap else "raw"
        self._init(load_snapshot() if use_snap else load_raw())

    def _init(self, frames: dict[str, pd.DataFrame]) -> None:
        self._perp = self._split(frames.get("perp"))
        self._spot = self._split(frames.get("spot"))
        self._fund: dict[str, pd.Series] = {}
        f = frames.get("funding")
        if f is not None and len(f):
            for sym, g in f.sort_values("ts").groupby("sym"):
                self._fund[sym] = pd.Series(g["rate"].to_numpy(float), index=pd.to_datetime(g["ts"].to_numpy(), unit="ms", utc=True))
        g = frames.get("crypto_fng")
        self._fng = pd.Series(dtype=float)
        if g is not None and len(g):
            s = pd.Series(g["value"].astype(int).to_numpy(), index=[fng_available(d) for d in g["date"]])
            self._fng = s[~s.index.duplicated(keep="last")].sort_index()
        self._items = build_items(frames)
        self._item_at = [it["available_at"] for it in self._items]

    @staticmethod
    def _split(d: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        if d is None or not len(d):
            return out
        for sym, g in d.sort_values("ts").groupby("sym"):
            idx = pd.DatetimeIndex(pd.to_datetime(g["ts"].to_numpy(), unit="ms", utc=True), name="ts")
            out[sym] = pd.DataFrame(g[BAR_COLS].to_numpy(float), index=idx, columns=BAR_COLS)
        return out

    @staticmethod
    def _sym(sym: str) -> str:
        return sym if sym.endswith("USDT") else perp(sym)

    # -- items
    def items(self, t, lookback_h) -> list[dict]:
        t = _ts(t)
        lo = bisect.bisect_right(self._item_at, t - pd.Timedelta(hours=lookback_h))
        hi = bisect.bisect_right(self._item_at, t)
        return self._items[lo:hi]

    def all_items(self) -> list[dict]:
        return list(self._items)

    # -- bars
    def bars(self, sym, t, n) -> pd.DataFrame:
        d = self._perp.get(self._sym(sym))
        if d is None:
            return pd.DataFrame(columns=BAR_COLS, index=pd.DatetimeIndex([], tz="UTC", name="ts"))
        hi = d.index.searchsorted(_ts(t) - HOUR, side="right")        # open + 1h <= t
        return d.iloc[max(0, hi - n):hi]

    def next_open(self, sym, t) -> float | None:
        d = self._perp.get(self._sym(sym))
        if d is None:
            return None
        t = _ts(t)
        i = d.index.searchsorted(t, side="left")
        return float(d["open"].iloc[i]) if i < len(d) and d.index[i] < t + HOUR else None

    # -- funding
    def funding(self, sym, t, n) -> pd.Series:
        s = self._fund.get(self._sym(sym), EMPTY)
        hi = s.index.searchsorted(_ts(t), side="right") if len(s) else 0
        return s.iloc[max(0, hi - n):hi]

    def settlements(self, sym, t0, t1) -> pd.Series:
        s = self._fund.get(self._sym(sym), EMPTY)
        if not len(s):
            return s
        return s.iloc[s.index.searchsorted(_ts(t0), side="right"):s.index.searchsorted(_ts(t1), side="right")]

    # -- context
    def fng(self, t) -> int | None:
        hi = self._fng.index.searchsorted(_ts(t), side="right") if len(self._fng) else 0
        return int(self._fng.iloc[hi - 1]) if hi else None

    def spot_premium(self, sym, t) -> float | None:
        """Latest closed hour (<= t, at most 6h old) that has both an rToken and a perp bar."""
        sym = self._sym(sym)
        p, s = self._perp.get(sym), self._spot.get(sym)
        if p is None or s is None:
            return None
        t = _ts(t)
        s = s.iloc[:s.index.searchsorted(t - HOUR, side="right")]
        common = s.index[s.index.isin(p.index) & (s.index > t - 7 * HOUR)]
        if not len(common):
            return None
        ts = common[-1]
        pc, sc = float(p.at[ts, "close"]), float(s.at[ts, "close"])
        return sc / pc - 1 if pc > 0 else None
