"""Where does the LLM's knowledge end? (project 2, research probe R1)

Closed-book questions about each month from 2024-07 to 2026-08, built from our own daily closes:
  P1 direction: did <ticker> end <month> higher or lower than it started? Only names that moved >= 5%
     that month, balanced to equal UP and DOWN, so a constant answer scores 50%.
  P2 level:     what was <ticker>'s close on the last trading day of <month>? Scored as |log error|.
Before the cutoff P1 is above 50% and P2 errors are small; after it P1 falls to chance and P2 errors
grow. The month where that happens bounds the clean window for a point-in-time replay.

Run: uv run python -m sentiment.probes.cutoff_probe   (needs BITGET_QWEN_* in .env)
Writes out/cutoff_probe.json (questions, answers, per-month scores); reads data/raw/underlying (config.RAW).
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the project dir, when run as a script
from sentiment.config import ENV_FILE, RAW, ROOT  # noqa: E402
from sentiment.config import OUT as OUT_DIR  # noqa: E402

UND = RAW / "underlying"
OUT = OUT_DIR / "cutoff_probe.json"
NAMES = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "PLTR", "INTC",
         "JPM", "AVGO", "ORCL", "MU", "SMCI", "ARM", "UBER", "BA", "DIS", "MSTR", "HOOD", "GME", "LLY",
         "COST", "CRM", "ADBE", "QCOM", "SHOP", "MARA", "RIOT", "SNOW", "PYPL", "BABA", "NKE", "WMT", "XOM"]
MONTHS = pd.period_range("2024-07", "2026-08", freq="M")
PER_SIDE = 6          # P1: this many UP and this many DOWN questions per month
N_LEVEL = 10          # P2: level questions per month
SYSTEM = ("You answer from memory only. You have no tools and no internet. "
          "If you do not know, still give your single best guess. Reply with JSON only.")


def env() -> dict[str, str]:
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return {k: os.environ[k] for k in ("BITGET_QWEN_API_KEY", "BITGET_QWEN_BASE_URL", "BITGET_QWEN_MODEL")}


def ask(cfg: dict[str, str], prompt: str) -> str:
    body = {"model": cfg["BITGET_QWEN_MODEL"], "temperature": 0, "enable_thinking": False,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request(cfg["BITGET_QWEN_BASE_URL"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {cfg['BITGET_QWEN_API_KEY']}",
                                          "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["choices"][0]["message"]["content"]


def parse(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {}


def monthly(names: list[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        f = UND / f"{n}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)[["date", "open", "close"]]
        d["m"] = pd.to_datetime(d["date"]).dt.to_period("M")
        g = d.groupby("m").agg(first_open=("open", "first"), last_close=("close", "last"),
                               last_date=("date", "last"))
        g["ret"] = g["last_close"] / g["first_open"] - 1
        g["name"] = n
        rows.append(g.reset_index())
    return pd.concat(rows)


def main() -> None:
    cfg = env()
    rng = random.Random(7)
    mo = monthly(NAMES)
    log = []
    for m in MONTHS:
        g = mo[mo["m"] == m]
        up = g[g["ret"] >= 0.05].sample(frac=1, random_state=7).head(PER_SIDE)
        dn = g[g["ret"] <= -0.05].sample(frac=1, random_state=7).head(PER_SIDE)
        k = min(len(up), len(dn))
        p1 = pd.concat([up.head(k), dn.head(k)]).sample(frac=1, random_state=7)
        label = m.strftime("%B %Y")
        q1 = {r["name"]: "UP" if r["ret"] > 0 else "DOWN" for _, r in p1.iterrows()}
        a1 = parse(ask(cfg, f"For each US stock ticker below, did the stock end {label} higher (UP) or lower "
                            f"(DOWN) than where it started {label}? Tickers: {', '.join(q1)}. "
                            'Reply as {"TICKER": "UP" or "DOWN", ...}.'))
        p2 = g.sample(n=min(N_LEVEL, len(g)), random_state=rng.randint(0, 10**6))
        q2 = {r["name"]: (str(r["last_date"])[:10], float(r["last_close"])) for _, r in p2.iterrows()}
        a2 = parse(ask(cfg, "Give the closing price in US dollars of each US stock on the date shown. "
                            + "; ".join(f"{t} on {d}" for t, (d, _) in q2.items())
                            + '. Reply as {"TICKER": number, ...}.'))
        hits = [str(a1.get(t, "")).upper() == v for t, v in q1.items()]
        errs = []
        for t, (_, px) in q2.items():
            try:
                errs.append(abs(np.log(float(a2[t]) / px)))
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                errs.append(np.nan)
        row = {"month": str(m), "p1_n": len(hits), "p1_acc": float(np.mean(hits)) if hits else None,
               "p2_n": int(np.sum(~np.isnan(errs))), "p2_median_abs_log_err": float(np.nanmedian(errs)),
               "q1": q1, "a1": a1, "q2": q2, "a2": a2}
        log.append(row)
        print(f"{row['month']}  P1 {row['p1_acc']!s:>6} (n={row['p1_n']:2d})  "
              f"P2 median |log err| {row['p2_median_abs_log_err']:.3f} (n={row['p2_n']})", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": cfg["BITGET_QWEN_MODEL"], "months": log}, indent=1, default=str))
    print("wrote", OUT.relative_to(ROOT), file=sys.stderr)


if __name__ == "__main__":
    main()
