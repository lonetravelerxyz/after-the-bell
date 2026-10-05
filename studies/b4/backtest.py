"""B4 long history: F&G-scaled MSTR long on daily US closes from MSTR's first BTC purchase.

F&G stamped d 00:00 UTC is known before the US close of d (~20:00 UTC), so its weight is held
close(d) -> close(d+1). Benchmarks (fixed in trials.log before the run): buy & hold, a constant
weight equal to the rule's mean weight, the same weight function on the rank of BTC's 30d return
(the momentum F&G is partly built from), and MSTR above its 200d MA.

Run: uv run python -m studies.b4.backtest      -> out/b4/history.json, prints the tables
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from studies.b4.config import FEE, FG_FLAT, FG_FULL, HALF_SPREAD, HIST_START, OUT, RAW

COST = FEE + HALF_SPREAD
END = "2026-10-01"
rng = np.random.default_rng(0)


def weight(fg: pd.Series) -> pd.Series:
    return ((FG_FLAT - fg) / (FG_FLAT - FG_FULL)).clip(0, 1)


def run(r: pd.Series, w: pd.Series, cost: float = COST) -> pd.Series:
    w = w.reindex(r.index).fillna(0)
    return w * r - cost * w.diff().abs().fillna(w.abs())


def stats(net: pd.Series, w: pd.Series | None = None) -> dict:
    eq = (1 + net).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    yrs = len(net) / 252
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    return {"total": float(eq.iloc[-1] - 1), "cagr": cagr, "sharpe": float(net.mean() / net.std() * np.sqrt(252)),
            "vol": float(net.std() * np.sqrt(252)), "max_dd": dd, "calmar": cagr / -dd if dd < 0 else None,
            "mean_w": None if w is None else float(w.mean())}


def boot(a: pd.Series, b: pd.Series, block: int = 20, reps: int = 4000) -> list[float]:
    """95% CI of annualised Sharpe(a) - Sharpe(b), circular block bootstrap on paired days."""
    x = pd.concat([a, b], axis=1).dropna().to_numpy()
    n, out = len(x), []
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, n // block + 1)])[:n]
        y = x[idx]
        sh = y.mean(0) / y.std(0)
        out.append(sh[0] - sh[1])
    return [float(v) for v in np.percentile(out, [2.5, 97.5]) * np.sqrt(252)]


def main() -> None:
    px = pd.read_parquet(RAW / "daily.parquet")
    fg = pd.read_parquet(RAW / "fng.parquet").set_index("date")["value"].astype(float)
    fg.index = fg.index.tz_localize(None)
    fg = fg.reindex(pd.date_range(fg.index[0], fg.index[-1])).ffill()
    s = px["MSTR"].dropna().loc[:END]
    r = s.pct_change().loc[HIST_START:].iloc[1:]           # return close(d-1) -> close(d)
    prev = s.index[s.index.get_indexer(r.index) - 1]       # the signal day d-1 of each return
    f = pd.Series(fg.reindex(prev, method="ffill").to_numpy(), index=r.index)

    btc = px["BTC-USD"].dropna()
    m30 = btc.pct_change(30)
    rank = m30.rolling(365, min_periods=180).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)
    mom = pd.Series(rank.reindex(prev, method="ffill").to_numpy(), index=r.index)
    ma = pd.Series((s > s.rolling(200).mean()).astype(float).reindex(prev).to_numpy(), index=r.index)

    w_rule = weight(f)
    rules = {
        "buy_hold": pd.Series(1.0, index=r.index),
        "fg_long": w_rule,
        "constant_mean_w": pd.Series(float(w_rule.mean()), index=r.index),
        "mom_twin": weight(mom),
        "ma200": ma,
    }
    nets = {k: run(r, w, 0.0 if k in ("buy_hold", "constant_mean_w") else COST) for k, w in rules.items()}
    res = {"window": [str(r.index[0].date()), str(r.index[-1].date())], "n_days": len(r), "cost_per_turnover": COST,
           "rules": {k: stats(nets[k], rules[k]) for k in rules},
           "ci_sharpe_fg_minus_buy_hold": boot(nets["fg_long"], nets["buy_hold"]),
           "ci_sharpe_fg_minus_constant": boot(nets["fg_long"], nets["constant_mean_w"]),
           "ci_sharpe_fg_minus_mom_twin": boot(nets["fg_long"], nets["mom_twin"])}
    years = {}
    for y in sorted(set(r.index.year)):
        ix = r.index.year == y
        years[str(y)] = {"mstr": float((1 + r[ix]).prod() - 1), "fg_long": float((1 + nets["fg_long"][ix]).prod() - 1),
                         "mean_w": float(w_rule[ix].mean()), "share_fg_le_25": float((f[ix] <= FG_FULL).mean()),
                         "share_fg_ge_75": float((f[ix] >= FG_FLAT).mean())}
    res["by_year"] = years
    # fear buckets: what MSTR did after each weight level
    b = pd.cut(f, [0, 25, 45, 55, 75, 100], labels=["<=25", "26-45", "46-55", "56-75", ">75"])
    fwd20 = (s.shift(-20) / s - 1).reindex(prev).set_axis(r.index)
    res["fwd20_by_fg_bucket"] = {str(k): {"n_days": int(v.size), "mean_fwd20": float(v.mean())}
                                 for k, v in fwd20.groupby(b, observed=True)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "history.json").write_text(json.dumps(res, indent=1))

    t = pd.DataFrame(res["rules"]).T
    for c in ("total", "cagr", "vol", "max_dd"):
        t[c] = t[c].map(lambda v: f"{v:+.0%}")
    print(f"window {res['window']} ({len(r)} days), cost {COST:.2%} per unit turnover\n")
    print(t.round(2).to_string())
    for k in ("buy_hold", "constant", "mom_twin"):
        print(f"Sharpe fg_long - {k}: 95% CI {np.round(res['ci_sharpe_fg_minus_' + k], 2).tolist()}")
    print("\n" + pd.DataFrame(years).T.map(lambda v: f"{v:+.0%}" if isinstance(v, float) else v).to_string())
    print("\nMSTR 20d forward return by F&G bucket:")
    print(pd.DataFrame(res["fwd20_by_fg_bucket"]).T.to_string())


if __name__ == "__main__":
    main()
