"""B5 comparison: per window, every variant's metrics and block-bootstrap CIs of Sharpe differences.

Run: uv run python -m b5.compare     -> out/b5/summary.json, prints the table
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from b5.run import B5_OUT
from sentiment import metrics

WINDOWS = {"dev": ("2026-06-22", "2026-08-10"), "test": ("2026-08-11", "2026-09-22"),
           "post": ("2026-09-23", "2026-10-01"), "long": ("2025-11-01", "2026-10-01")}
PAIRS = (("rule", "const"), ("rule", "buyhold"), ("llm", "rule"))
BLOCK, REPS = 5, 4000


def daily(run_dir) -> pd.Series:
    return metrics.complete_day_returns(metrics.load_equity(run_dir)["equity"])


def boot(a: pd.Series, b: pd.Series, seed: int = 0) -> list[float] | None:
    x = pd.concat([a, b], axis=1).dropna().to_numpy()
    n = len(x)
    if n < 2 * BLOCK:
        return None
    rng, out = np.random.default_rng(seed), []
    for _ in range(REPS):
        idx = np.concatenate([np.arange(s, s + BLOCK) % n for s in rng.integers(0, n, n // BLOCK + 1)])[:n]
        y = x[idx]
        sd = y.std(0, ddof=1)
        if (sd > 0).all():
            out.append((y.mean(0) / sd)[0] - (y.mean(0) / sd)[1])
    return [float(v) for v in np.percentile(out, [2.5, 97.5]) * np.sqrt(metrics.ANN)]


def main() -> None:
    res, rows = {}, []
    for wn, (s, e) in WINDOWS.items():
        runs = {v: B5_OUT / f"{v}_{s}_{e}" for v in ("rule", "llm", "const", "buyhold")}
        runs = {v: d for v, d in runs.items() if (d / "metrics.json").exists()}
        res[wn] = {"window": [s, e], "variants": {}, "ci": {}}
        for v, d in runs.items():
            m = json.loads((d / "metrics.json").read_text())
            keep = {k: m.get(k) for k in ("total_return", "sharpe", "max_dd", "trades", "turnover_ann",
                                          "mean_gross_exposure", "const_w", "n_invalid", "log_last_hash")}
            res[wn]["variants"][v] = keep
            rows.append({"window": wn, "variant": v, "return": f"{keep['total_return']:+.2%}",
                         "sharpe": round(keep["sharpe"], 2) if keep["sharpe"] is not None else None,
                         "max_dd": f"{keep['max_dd']:.2%}", "gross": round(keep["mean_gross_exposure"], 3),
                         "trades": keep["trades"]})
        for a, b in PAIRS:
            if a in runs and b in runs:
                res[wn]["ci"][f"{a}-{b}"] = boot(daily(runs[a]), daily(runs[b]))
    (B5_OUT / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))
    for wn, r in res.items():
        print(wn, {k: None if v is None else [round(x, 2) for x in v] for k, v in r["ci"].items()})


if __name__ == "__main__":
    main()
