"""Replay execution: a simulated broker on stock perps (CONTRACT: Execution).

Positions are held in contract quantity. `rebalance` turns target weights x equity into quantity
deltas and fills them at the next 1h bar open, moved half a quoted spread against us, plus a taker
fee on the fill notional. `mark` values the book at the last closed bar close and books funding for
every settlement since the previous mark (a long pays a positive rate).

No-churn band (`below_band`, shared with live.plan_orders): a same-side resize is skipped when
|delta weight| < max(MIN_TRADE, REL_BAND x |target weight|), so a slowly decaying target (the baseline's
exp(-age/24) cards lose ~15% per 4h step) is not re-traded at every decision; opens, closes and flips
always clear it. A ticker with no bar opening within 1h of t (a data gap) is not filled and is listed
in `self.skipped`. `rebalance(..., only=names)` trades just those names (hourly risk exits).

Cost accounting (all cumulative, in quote currency):
  fees    = sum of taker fees paid (>= 0)
  spread  = sum of half-spread costs paid (>= 0)
  funding = funding P&L received (signed; negative = net paid)
  gross_equity = equity + fees + spread - funding   (the book before any cost)
so the ladder gross -> after fees -> after spread -> after funding ends at `equity`.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from sentiment.config import HALF_SPREAD_FALLBACK, SPREAD_FILE, START_EQUITY, TAKER, perp

MIN_TRADE = 0.0025          # skip deltas below this fraction of equity (no churn); full closes always trade
REL_BAND = 0.25             # ... or below this fraction of the target weight (same-side resizes)


def below_band(delta_w: float, target_w: float) -> bool:
    """True when a trade of delta_w (fraction of equity) towards target_w is inside the no-churn band."""
    return abs(delta_w) < max(MIN_TRADE, REL_BAND * abs(target_w))


@lru_cache(maxsize=1)
def _spread_table() -> dict[str, float]:
    if not SPREAD_FILE.exists():
        return {}
    s = pd.read_csv(SPREAD_FILE)
    return dict(zip(s.ticker, s.spread_bps.astype(float)))


def half_spread(ticker: str) -> float:
    """Quoted half-spread as a fraction of price (static/quoted_spread.csv, else the fallback)."""
    bps = _spread_table().get(ticker)
    return HALF_SPREAD_FALLBACK if bps is None or pd.isna(bps) else bps / 2 / 1e4


class SimBroker:
    """Replay broker over a Store (only `next_open`, `bars` and `settlements` are used)."""

    def __init__(self, store, start_equity: float = START_EQUITY):
        self.store = store
        self.cash = float(start_equity)
        self.positions: dict[str, float] = {}
        self.last_price: dict[str, float] = {}
        self.fees = self.spread = self.funding = 0.0
        self._funding_t: pd.Timestamp | None = None       # funding is booked up to this time
        self.skipped: list[dict] = []                     # tickers the last rebalance could not fill

    # ---- pricing -------------------------------------------------------------------------------
    def _mark_price(self, tk: str, t: pd.Timestamp) -> float | None:
        b = self.store.bars(perp(tk), t, 1)
        if b is not None and len(b):
            self.last_price[tk] = float(b["close"].iloc[-1])
        return self.last_price.get(tk)

    def _accrue_funding(self, t: pd.Timestamp) -> None:
        """Book funding for settlements in (last booked, t] on the positions held over that span."""
        t0 = self._funding_t
        if t0 is not None and t <= t0:
            return
        self._funding_t = t
        if t0 is None:                                    # first call: nothing held before it
            return
        for tk, qty in self.positions.items():
            if qty == 0:
                continue
            s = self.store.settlements(perp(tk), t0, t)
            for ts, rate in (s.items() if s is not None else []):
                px = self._mark_price(tk, ts)
                if px is None or pd.isna(rate):
                    continue
                pnl = -qty * px * float(rate)
                self.cash += pnl
                self.funding += pnl

    # ---- Broker interface ----------------------------------------------------------------------
    def rebalance(self, t, targets: dict[str, float], equity: float, only=None) -> list[dict]:
        t = pd.Timestamp(t)
        self._accrue_funding(t)                           # funding before t belongs to the old book
        fills = []
        self.skipped = []
        names = set(targets) | {k for k, q in self.positions.items() if q}
        for tk in sorted(names if only is None else names & set(only)):
            mid = self.store.next_open(perp(tk), t)
            if mid is None or pd.isna(mid) or mid <= 0:
                self.skipped.append({"ticker": tk, "reason": "no bar opening within 1h"})
                continue
            mid = float(mid)
            cur = self.positions.get(tk, 0.0)
            tgt = float(targets.get(tk, 0.0)) * equity / mid
            delta = tgt - cur
            closing = tgt == 0 and cur != 0
            flip = cur != 0 and tgt != 0 and (cur > 0) != (tgt > 0)
            if delta == 0 or (not closing and not flip and below_band(delta * mid / equity, float(targets.get(tk, 0.0)))):
                continue
            side = 1 if delta > 0 else -1
            hs = half_spread(tk)
            price = mid * (1 + side * hs)
            notional = abs(delta) * price
            fee = TAKER * notional
            spread_cost = abs(delta) * mid * hs
            self.cash -= delta * price + fee
            self.fees += fee
            self.spread += spread_cost
            self.positions[tk] = 0.0 if closing else cur + delta
            self.last_price.setdefault(tk, mid)
            fills.append({"ticker": tk, "side": "buy" if side > 0 else "sell", "qty": abs(delta),
                          "price": price, "fee": fee, "half_spread_cost": spread_cost,
                          "ts": t.ceil("h"), "order_id": f"sim-{t:%Y%m%dT%H%M}-{tk}"})
        self.positions = {k: q for k, q in self.positions.items() if q != 0}
        return fills

    def mark(self, t) -> dict:
        t = pd.Timestamp(t)
        self._accrue_funding(t)
        value, prices = 0.0, {}
        for tk, qty in self.positions.items():
            px = self._mark_price(tk, t)
            if px is None:
                continue
            prices[tk] = px
            value += qty * px
        equity = self.cash + value
        return {"ts": t, "equity": equity, "positions": dict(self.positions), "cash": self.cash,
                "fees": self.fees, "spread": self.spread, "funding": self.funding,
                "gross_equity": equity + self.fees + self.spread - self.funding, "prices": prices}
