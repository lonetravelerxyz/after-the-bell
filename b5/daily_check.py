"""B5 multi-year check of the vol rule on daily stock closes (pre-declared in b5/trials.log, entry 1).

w_i(d) = (1/N_d) * min(1, median(sd10_i over the trailing 60 trading days) / sd10_i(d)), sd10 = stdev of the
last 10 daily returns, N_d = names listed on d; held close(d) -> close(d+1); 13 bp per unit turnover.
Benchmarks: equal weight 1/N_d (daily rebalanced) and a constant fraction of it equal to the rule's mean
gross (exposure-matched). Data: split-adjusted closes from yfinance, cached in data/raw/b5/daily.parquet.

Run: uv run --with yfinance python -m b5.daily_check     -> out/b5/daily_check.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from b5.run import B5_OUT
from sentiment.config import UNIVERSE, data_dir

RAW = data_dir() / "raw" / "b5"
START, END = "2021-01-01", "2026-10-01"
SD_N, MED_N, COST = 10, 60, 0.0013
rng = np.random.default_rng(0)


def closes() -> pd.DataFrame:
    f = RAW / "daily.parquet"
    if not f.exists():
        import yfinance as yf
        RAW.mkdir(parents=True, exist_ok=True)
        yf.download(UNIVERSE, start="2020-06-01", auto_adjust=True, progress=False)["Close"].to_parquet(f)
    return pd.read_parquet(f)


def net(r: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    w = w.fillna(0.0)
    turn = w.diff().abs().sum(axis=1)
    turn.iloc[0] = w.iloc[0].abs().sum()
    return (w * r.fillna(0.0)).sum(axis=1) - COST * turn


def stats(x: pd.Series) -> dict:
    eq = (1 + x).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    cagr = float(eq.iloc[-1] ** (252 / len(x)) - 1)
    return {"total": float(eq.iloc[-1] - 1), "cagr": cagr, "sharpe": float(x.mean() / x.std() * np.sqrt(252)),
            "vol": float(x.std() * np.sqrt(252)), "max_dd": dd}


def boot(a: pd.Series, b: pd.Series, block: int = 20, reps: int = 4000) -> list[float]:
    x = pd.concat([a, b], axis=1).dropna().to_numpy()
    n, out = len(x), []
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, n // block + 1)])[:n]
        y = x[idx]
        sh = y.mean(0) / y.std(0)
        out.append(sh[0] - sh[1])
    return [float(v) for v in np.percentile(out, [2.5, 97.5]) * np.sqrt(252)]


def main() -> None:
    px = closes()
    r = px.pct_change(fill_method=None)
    sd = r.rolling(SD_N, min_periods=SD_N).std()
    tgt = sd.rolling(MED_N, min_periods=MED_N).median()
    live = sd.notna() & tgt.notna()
    n = live.sum(axis=1).replace(0, np.nan)
    scale = (tgt / sd).clip(upper=1.0).where(live)
    w_rule = scale.div(n, axis=0)                              # known at close(d)
    w_ew = live.astype(float).div(n, axis=0).where(live)
    # weights at close(d) earn the return close(d) -> close(d+1)
    R = r.shift(-1).loc[START:END].iloc[:-1]
    wr, we = w_rule.loc[R.index], w_ew.loc[R.index]
    rule, ew = net(R, wr), net(R, we)
    k = float(wr.sum(axis=1).mean())
    const = net(R, we * k)
    res = {"window": [str(R.index[0].date()), str(R.index[-1].date())], "n_days": len(R), "mean_gross_rule": k,
           "rule": stats(rule), "equal_weight": stats(ew), "const_exposure_matched": stats(const),
           "ci_sharpe_rule_minus_const": boot(rule, const), "ci_sharpe_rule_minus_ew": boot(rule, ew),
           "by_year": {str(y): {"rule": float((1 + rule[rule.index.year == y]).prod() - 1),
                                "const": float((1 + const[const.index.year == y]).prod() - 1),
                                "ew": float((1 + ew[ew.index.year == y]).prod() - 1),
                                "gross_rule": float(wr[wr.index.year == y].sum(axis=1).mean())}
                       for y in sorted(set(R.index.year))},
           "names_first_day": {tk: str(px[tk].first_valid_index().date()) for tk in UNIVERSE}}
    B5_OUT.mkdir(parents=True, exist_ok=True)
    (B5_OUT / "daily_check.json").write_text(json.dumps(res, indent=1) + "\n")
    t = pd.DataFrame({k: res[k] for k in ("rule", "const_exposure_matched", "equal_weight")}).T
    print(f"{res['window']} {len(R)} days, rule mean gross {k:.3f}")
    print(t.map(lambda v: f"{v:+.1%}").assign(sharpe=t.sharpe.round(2)).to_string())
    print("CI rule-const", np.round(res["ci_sharpe_rule_minus_const"], 2), "CI rule-ew", np.round(res["ci_sharpe_rule_minus_ew"], 2))
    print(pd.DataFrame(res["by_year"]).T.map(lambda v: f"{v:+.1%}").to_string())
    print(res["names_first_day"])


if __name__ == "__main__":
    main()
