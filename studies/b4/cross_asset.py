"""B4 step 0: crypto F&G as a contrarian signal on BTC and four crypto stocks, 2018-02 -> 2026-10.

Pre-declared before the first run (2026-10-02, studies/b4/trials.log entry 0): F&G(d) (stamp 00:00 UTC d) is held by a
stock over close(d) -> close(d+1) and by BTC over day d+1. Rules, no grid: (a) long-only w = clip((75-FG)/50, 0, 1);
(b) long/short w = clip((50-FG)/50, -1, 1). Benchmarks: buy & hold; the same weight functions on the rank of BTC's
30d return within its trailing 365 days (the momentum F&G is partly built from); the asset's own 200d MA.
10 bp per unit turnover. Periods P1 2018-02..2022-12 and P2 2023-01..2026-10-01, each once. Pass: beat buy & hold
with a block-bootstrap CI excluding 0 in both periods AND beat the momentum twin. Also: F&G quintile -> forward
5/20/60d return with a Newey-West t on the slope, and the same for the 20d return net of 60d beta x BTC.

This file is the committed form of the scratch run of the same day; the only change is the bootstrap
annualisation (365 days for BTC, 252 for stocks; the scratch run used 252 for both, so its BTC intervals were
~17% too narrow). Run: uv run --with yfinance --with statsmodels python -m studies.b4.cross_asset -> out/b4/cross_asset.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from studies.b4.config import OUT, RAW

ASSETS = ("BTC-USD", "MSTR", "COIN", "MARA", "RIOT")
PERIODS = {"P1": ("2018-02-01", "2022-12-31"), "P2": ("2023-01-01", "2026-10-01")}
COST = 0.001
rng = np.random.default_rng(0)


def prices() -> pd.DataFrame:
    f = RAW / "cross.parquet"
    if not f.exists():
        import yfinance as yf
        yf.download(list(ASSETS), start="2017-01-01", auto_adjust=True, progress=False)["Close"].to_parquet(f)
    return pd.read_parquet(f).loc[:"2026-10-01"]


def fng() -> pd.Series:
    s = pd.read_parquet(RAW / "fng.parquet").set_index("date")["value"].astype(float)
    s.index = s.index.tz_localize(None)
    return s.reindex(pd.date_range(s.index[0], s.index[-1])).ffill()


def wa(s): return ((75 - s) / 50).clip(0, 1)
def wb(s): return ((50 - s) / 50).clip(-1, 1)


def aligned(px: pd.DataFrame, a: str, fg: pd.Series, mom: pd.Series):
    s = px[a].dropna()
    r = s.pct_change().iloc[1:]
    sig_dates = r.index - pd.Timedelta(days=1) if a == "BTC-USD" else s.index[:-1]
    sig = lambda x: pd.Series(x.reindex(sig_dates, method="ffill").to_numpy(), index=r.index)  # noqa: E731
    ma = (s > s.rolling(200).mean()).astype(float).shift(1).reindex(r.index)
    return s, r, sig(fg), sig(mom), ma


def run(r: pd.Series, w: pd.Series, cost: float) -> pd.Series:
    w = w.fillna(0)
    return w * r - cost * w.diff().abs().fillna(w.abs())


def stats(net: pd.Series, ann: int) -> dict:
    eq = (1 + net).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"sharpe": float(net.mean() / net.std() * np.sqrt(ann)), "cagr": float(eq.iloc[-1] ** (ann / len(net)) - 1),
            "max_dd": dd}


def boot(a: pd.Series, b: pd.Series, ann: int, block: int = 20, reps: int = 2000) -> list[float]:
    x = pd.concat([a, b], axis=1).dropna().to_numpy()
    n, out = len(x), []
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, n // block + 1)])[:n]
        y = x[idx]
        sh = y.mean(0) / y.std(0)
        out.append(sh[0] - sh[1])
    return [float(v) for v in np.percentile(out, [2.5, 97.5]) * np.sqrt(ann)]


def nw_slope(f: np.ndarray, y: np.ndarray, lags: int) -> tuple[float, float]:
    import statsmodels.api as sm
    m = sm.OLS(y, sm.add_constant(f)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(m.params[1] * 50), float(m.tvalues[1])          # effect of +50 F&G points


def main() -> None:
    px, fg = prices(), fng()
    btc = px["BTC-USD"].dropna()
    rank = btc.pct_change(30).rolling(365, min_periods=180).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)
    mom = rank.reindex(fg.index).ffill()
    res: dict = {"strategies": {}, "slopes": {}, "resid_slopes": {}, "short_leg_mstr": {}}
    for a in ASSETS:
        s, r_all, f_all, m_all, ma_all = aligned(px, a, fg, mom)
        ann = 365 if a == "BTC-USD" else 252
        for pn, (s0, s1) in PERIODS.items():
            r, f, m, ma = (x.loc[s0:s1] for x in (r_all, f_all, m_all, ma_all))
            if len(r) < 250:
                continue
            bh = run(r, pd.Series(1.0, index=r.index), 0.0)
            rules = {"buy_hold": bh, "fg_long": run(r, wa(f), COST), "fg_long_short": run(r, wb(f), COST),
                     "mom_long": run(r, wa(m), COST), "mom_long_short": run(r, wb(m), COST),
                     "ma200": run(r, ma, COST)}
            res["strategies"][f"{a}|{pn}"] = {
                "n_days": len(r), **{k: stats(v, ann) for k, v in rules.items()},
                "ci_fg_long_minus_bh": boot(rules["fg_long"], bh, ann),
                "ci_fg_long_short_minus_bh": boot(rules["fg_long_short"], bh, ann)}
            for h in (5, 20, 60):
                fwd = (s.shift(-h) / s - 1)
                fs = fg.reindex(s.index - pd.Timedelta(days=1)) if a == "BTC-USD" else fg.reindex(s.index, method="ffill")
                d = pd.DataFrame({"f": fs.to_numpy(), "y": fwd.to_numpy()}, index=s.index).loc[s0:s1].dropna()
                b, t = nw_slope(d.f.to_numpy(), d.y.to_numpy(), h)
                res["slopes"][f"{a}|{pn}|{h}d"] = {"per_50_fg": b, "nw_t": t, "n": len(d)}
    for a in ("MSTR", "COIN", "MARA", "RIOT"):
        s = px[a].dropna()
        r = s.pct_change()
        b = btc.reindex(s.index).ffill().pct_change()
        beta = (r.rolling(60).cov(b) / b.rolling(60).var()).shift(1)
        fwd = (r - beta * b)[::-1].rolling(20).sum()[::-1].shift(-1)
        f = fg.reindex(s.index, method="ffill")
        for pn, (s0, s1) in PERIODS.items():
            d = pd.DataFrame({"f": f, "y": fwd}).loc[s0:s1].dropna()
            if len(d) >= 250:
                bb, t = nw_slope(d.f.to_numpy(), d.y.to_numpy(), 20)
                res["resid_slopes"][f"{a}|{pn}"] = {"per_50_fg": bb, "nw_t": t, "n": len(d)}
    _, r, f, _, _ = aligned(px, "MSTR", fg, mom)
    leg = wb(f).clip(upper=0) * r
    for y in range(2018, 2027):
        res["short_leg_mstr"][str(y)] = {"sum_of_daily": float(leg.loc[str(y)].sum()),
                                         "share_days_short": float((wb(f).loc[str(y)] < 0).mean())}
    s = res["strategies"]
    res["summary"] = {
        "n_asset_periods": len(s),
        "fg_long_short_below_bh": sum(v["fg_long_short"]["sharpe"] < v["buy_hold"]["sharpe"] for v in s.values()),
        "fg_long_below_bh": sum(v["fg_long"]["sharpe"] < v["buy_hold"]["sharpe"] for v in s.values()),
        "fg_long_beats_bh_ci_excl_0": sum(v["ci_fg_long_minus_bh"][0] > 0 for v in s.values()),
        "p1_slopes_positive": sum(v["per_50_fg"] > 0 for k, v in res["slopes"].items() if "|P1|" in k),
        "p1_slopes_total": sum(1 for k in res["slopes"] if "|P1|" in k),
        "resid_max_abs_t": max(abs(v["nw_t"]) for v in res["resid_slopes"].values()),
        "verdict": "fail (not claimed): the long-only rule beats buy & hold with a CI excluding 0 in none of the asset-periods "
                   "and the long/short rule is below buy & hold in nearly all; in 2018-22 most slopes of forward return on "
                   "F&G are positive (greed was followed by higher returns), the opposite of the contrarian premise"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cross_asset.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["summary"], indent=1))
    print(pd.DataFrame({k: {c: round(v[c]["sharpe"], 2) for c in ("buy_hold", "fg_long", "fg_long_short", "mom_long", "ma200")}
                        for k, v in s.items()}).T.to_string())
    print({y: round(v["sum_of_daily"], 3) for y, v in res["short_leg_mstr"].items()})


if __name__ == "__main__":
    main()
