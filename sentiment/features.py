"""Market state at a decision time t (contract: sentiment/CONTRACT.md, Features).

Everything comes from `Store` accessors called with t, so only data available by t is used.
Per ticker: last closed price, 4h/24h/7d returns (close at t vs the last close at or before t-k),
vol_7d (stdev of hourly log returns over the last 7 days, scaled to one day), funding (last
settlement and 7-day mean), rToken premium, whether the US regular session is open; under risk v1 also
beta_60d, the trailing daily beta to the equal-weight UNIVERSE basket (see `betas`), and beta_days, the number
of daily returns it was estimated from (< BETA_MIN_D -> beta 1.0 fallback). Under risk v0 (`version`, default
config.RISK_VERSION) the state has neither, exactly as every run before risk v1 logged it.
A name whose last closed bar is more than STALE_H hours old (a data gap) gets price, returns and vol
None rather than a stale close shown as current.
"_market": crypto F&G value and regime, 24h returns of the SP500 and NDX100 index perps.
"""
from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sentiment import config
from sentiment.config import BETA_LOOKBACK_D, BETA_MIN_D, UNIVERSE, perp

NY = ZoneInfo("America/New_York")
NYSE_HOLIDAYS = {"2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
                 "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"}
FNG_REGIMES = ((25, "extreme_fear"), (46, "fear"), (54, "neutral"), (74, "greed"), (100, "extreme_greed"))
LOOKBACK_BARS = 24 * 7 + 8
STALE_H = 1                # last close more than this many hours before t -> no price/returns


def us_session_open(t: pd.Timestamp) -> bool:
    """NYSE regular session 09:30-16:00 New York time, weekdays, 2026 full holidays excluded."""
    ny = pd.Timestamp(t).tz_convert(NY)
    minute = ny.hour * 60 + ny.minute
    return ny.weekday() < 5 and str(ny.date()) not in NYSE_HOLIDAYS and 570 <= minute < 960


def fng_regime(v: int | None) -> str | None:
    if v is None:
        return None
    return next(name for hi, name in FNG_REGIMES if v <= hi)


def _f(x) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def _closes(store, sym: str, t: pd.Timestamp) -> pd.Series:
    """Closes of the last closed bars, indexed by close time (<= t); empty when the last one is stale."""
    b = store.bars(sym, t, LOOKBACK_BARS)
    c = pd.Series(b["close"].to_numpy(float), index=b.index + pd.Timedelta(hours=1))
    return c.iloc[:0] if len(c) and c.index[-1] < t - pd.Timedelta(hours=STALE_H) else c


def _ret(c: pd.Series, t: pd.Timestamp, hours: int) -> float | None:
    if not len(c):
        return None
    past = c[c.index <= t - pd.Timedelta(hours=hours)]
    if not len(past) or past.iloc[-1] <= 0:
        return None
    return _f(c.iloc[-1] / past.iloc[-1] - 1)


def _ticker_state(store, tk: str, t: pd.Timestamp) -> dict:
    sym = perp(tk)
    c = _closes(store, sym, t)
    week = c[c.index > t - pd.Timedelta(days=7)]
    lr = np.diff(np.log(week.to_numpy())) if len(week) > 2 and (week > 0).all() else np.array([])
    f = store.funding(sym, t, 200)
    f7 = f[f.index > t - pd.Timedelta(days=7)]
    prem = store.spot_premium(sym, t)
    return {
        "price": _f(c.iloc[-1]) if len(c) else None,
        "ret_4h": _ret(c, t, 4), "ret_24h": _ret(c, t, 24), "ret_7d": _ret(c, t, 24 * 7),
        "vol_7d": _f(lr.std(ddof=1) * math.sqrt(24)) if len(lr) > 1 else None,
        "funding_last": _f(f.iloc[-1]) if len(f) else None,
        "funding_7d_mean": _f(f7.mean()) if len(f7) else None,
        "spot_premium": _f(prem) if prem is not None else None,
        "us_session_open": us_session_open(t),
    }


def daily_closes(store, sym: str, t: pd.Timestamp, days: int) -> pd.Series:
    """Close of each COMPLETED UTC day among the last `days` days before t, indexed by the day (00:00 UTC).
    A day's close is the close of its last available 1h bar (bar open in that day, close time <= t);
    the day in progress at t is left out, so the series changes only at 00:00 UTC. Days without a bar
    are absent."""
    last_day = t.floor("D") - pd.Timedelta(days=1)                 # the latest day whose end is <= t
    b = store.bars(sym, t, 24 * (days + 2))
    if not len(b):
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    c = pd.Series(b["close"].to_numpy(float), index=b.index)
    c = c.groupby(c.index.floor("D")).last()
    return c[(c.index <= last_day) & (c.index > last_day - pd.Timedelta(days=days))]


def betas(store, t: pd.Timestamp, lookback_d: int = BETA_LOOKBACK_D, min_d: int = BETA_MIN_D) -> dict[str, float]:
    """Trailing daily beta of each UNIVERSE name to the equal-weight UNIVERSE basket at t (point-in-time).

    Daily closes of the last lookback_d + 1 completed UTC days (`daily_closes`, bars with close time
    <= t) -> simple daily returns over consecutive calendar days (a missing day gives no return for it
    and the next day) -> basket return = mean of the names' returns that day -> beta_i = cov(r_i, r_b) /
    var(r_b) over the days both exist. Fewer than min_d such days, or a flat basket, -> beta 1.0."""
    return beta_stats(store, t, lookback_d, min_d)[0]


def beta_stats(store, t: pd.Timestamp, lookback_d: int = BETA_LOOKBACK_D,
               min_d: int = BETA_MIN_D) -> tuple[dict[str, float], dict[str, int]]:
    """(`betas`, number of daily returns each beta was estimated from). A count below min_d means the
    beta is the 1.0 fallback; below lookback_d, a shorter history than intended."""
    last_day = t.floor("D") - pd.Timedelta(days=1)
    days = pd.date_range(last_day - pd.Timedelta(days=lookback_d), last_day, freq="D")
    closes = pd.DataFrame({tk: daily_closes(store, perp(tk), t, lookback_d + 1) for tk in UNIVERSE}).reindex(days)
    closes = closes.where(closes > 0)
    rets = closes / closes.shift(1) - 1
    basket = rets.mean(axis=1, skipna=True)
    out, n = {}, {}
    for tk in UNIVERSE:
        pair = pd.DataFrame({"r": rets[tk], "b": basket}).dropna()
        n[tk] = len(pair)
        var = float(pair["b"].var(ddof=1)) if len(pair) > 1 else 0.0
        if len(pair) < min_d or not math.isfinite(var) or var <= 0:
            out[tk] = 1.0
        else:
            beta = float(pair["r"].cov(pair["b"], ddof=1)) / var
            out[tk] = beta if math.isfinite(beta) else 1.0
    return out, n


def market_state(store, t, version: str | None = None) -> dict[str, dict]:
    """The state at t (module docstring). `version` = risk version (default config.RISK_VERSION): beta_60d
    and beta_days are added under v1 only."""
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    out = {tk: _ticker_state(store, tk, t) for tk in UNIVERSE}
    if (version or config.RISK_VERSION) == "v1":
        b, n = beta_stats(store, t)
        for tk in UNIVERSE:
            out[tk]["beta_60d"], out[tk]["beta_days"] = b[tk], n[tk]
    fng = store.fng(t)
    out["_market"] = {"fng": fng, "fng_regime": fng_regime(fng),
                      "sp500_ret_24h": _ret(_closes(store, "SP500USDT", t), t, 24),
                      "ndx_ret_24h": _ret(_closes(store, "NDX100USDT", t), t, 24)}
    return out


def main() -> None:
    import json
    import sys

    from sentiment.data import Store
    t = sys.argv[1] if len(sys.argv) > 1 else "2026-08-03 16:00"
    print(json.dumps(market_state(Store(), t), indent=1, default=str))


if __name__ == "__main__":
    main()
