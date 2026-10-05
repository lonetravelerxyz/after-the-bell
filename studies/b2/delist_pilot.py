"""Delisting pilot: after Binance announces a full token delisting, what does the token do on Bitget?

Events: Binance CMS delisting catalog (catalogId 161), articles titled "Binance Will Delist <A>, <B> ... on <date>";
the announcement's releaseDate (ms) is the exact event time. Prices: Bitget spot 1-minute candles [t-1d, t+2d],
1-hour candles [t-1d, t+31d]. Measured from the close at the announcement minute: returns to +5m/+30m/+1h/+6h/
+24h/+72h/+7d/+30d, the return to the delisting date, the 30-day low, BTC-adjusted.

Run: uv run python -m studies.b2.delist_pilot -> out/b2/delistings.json, delistings.md
"""
from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

from studies.b2 import config as C
from studies.b2.incident_pilot import candles

CMS = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
TITLE = re.compile(r"^Binance Will Delist (.+?) on (\d{4}-\d{2}-\d{2})", re.I)


def articles() -> list[dict]:
    out, page = [], 1
    while True:
        r = requests.get(CMS, params={"type": 1, "pageNo": page, "pageSize": 50, "catalogId": 161}, timeout=30).json()
        arts = r["data"]["catalogs"][0]["articles"]
        if not arts:
            break
        out.extend(arts)
        if len(arts) < 50:
            break
        page += 1
        time.sleep(0.3)
    return out


def events() -> list[dict]:
    evs = []
    for a in articles():
        m = TITLE.match(a.get("title", ""))
        if not m:
            continue
        toks = re.split(r",\s*|\s+and\s+|\s*&\s*", m.group(1))
        toks = [t.strip().upper() for t in toks if re.fullmatch(r"[A-Za-z0-9]{2,10}", t.strip())]
        t = pd.Timestamp(int(a["releaseDate"]), unit="ms", tz="UTC")
        for tok in toks:
            evs.append({"token": tok, "announced": str(t), "delist_date": m.group(2), "title": a["title"]})
    return evs


def study(ev: dict) -> dict:
    t0 = pd.Timestamp(ev["announced"])
    m = candles(ev["token"], "1min", t0 - pd.Timedelta(days=1), t0 + pd.Timedelta(days=2))
    if len(m) < 600 or m.index[0] > t0 - pd.Timedelta(hours=6):
        return {**ev, "skip": f"no 1min data ({len(m)} rows)"}
    h = candles(ev["token"], "1h", t0 - pd.Timedelta(days=1), t0 + pd.Timedelta(days=31))
    bm = candles("BTC", "1min", t0 - pd.Timedelta(days=1), t0 + pd.Timedelta(days=2))
    bh = candles("BTC", "1h", t0 - pd.Timedelta(days=1), t0 + pd.Timedelta(days=31))
    p0 = float(m.asof(t0))
    res = {**ev, "p0": p0, "pre_1h": float(p0 / m.asof(t0 - pd.Timedelta(hours=1)) - 1)}
    for lab, dt in [("5m", pd.Timedelta(minutes=5)), ("30m", pd.Timedelta(minutes=30)), ("1h", pd.Timedelta(hours=1)),
                    ("6h", pd.Timedelta(hours=6)), ("24h", pd.Timedelta(hours=24)), ("72h", pd.Timedelta(hours=72)),
                    ("7d", pd.Timedelta(days=7)), ("30d", pd.Timedelta(days=30))]:
        src, bs = (m, bm) if dt <= pd.Timedelta(hours=24) else (h, bh)
        t1 = t0 + dt
        p1 = float(src.asof(t1)) if t1 <= src.index[-1] else None
        b0 = float(bs.asof(t0)) if len(bs) else None
        b1 = float(bs.asof(t1)) if len(bs) and t1 <= bs.index[-1] else None
        r = None if p1 is None else p1 / p0 - 1
        res[f"r_{lab}"] = r
        res[f"adj_{lab}"] = None if r is None or b0 is None or b1 is None else r - (b1 / b0 - 1)
    td = pd.Timestamp(ev["delist_date"], tz="UTC")
    res["r_to_delist"] = float(h.asof(td) / p0 - 1) if td <= h.index[-1] else None
    after = h[h.index > t0]
    res["min_30d"] = float(after.min() / p0 - 1) if len(after) else None
    return res


def main() -> None:
    C.OUT.mkdir(parents=True, exist_ok=True)
    evs = events()
    print(f"{len(evs)} token delisting events from {len(set(e['title'] for e in evs))} announcements", file=sys.stderr)
    rows = []
    for e in evs:
        try:
            r = study(e)
        except Exception as ex:  # noqa: BLE001
            r = {**e, "skip": f"error {str(ex)[:60]}"}
        rows.append(r)
        print(f"{e['announced'][:16]} {e['token']:8s} " + (r.get("skip") or f"5m {r['r_5m']:+.3f} 1h {r['r_1h']:+.3f} 24h {r['r_24h'] if r['r_24h'] is None else round(r['r_24h'],3)} 7d {r['r_7d'] if r['r_7d'] is None else round(r['r_7d'],3)} min30 {r['min_30d'] if r['min_30d'] is None else round(r['min_30d'],3)}"), file=sys.stderr)
    (C.OUT / "delistings.json").write_text(json.dumps(rows, indent=1, default=str))
    ok = [r for r in rows if "p0" in r]
    df = pd.DataFrame(ok)
    L = ["# Delisting pilot (estimated, Bitget spot candles; event time = Binance announcement release time)\n",
         f"Events: {len(rows)} tokens named in 'Binance Will Delist ...' announcements, {len(ok)} with Bitget 1-minute data around the announcement.\n",
         "| Window after the announcement | Median | Mean | Share < -10% | Median BTC-adjusted |", "|---|---:|---:|---:|---:|"]
    for lab in ("5m", "30m", "1h", "6h", "24h", "72h", "7d", "30d"):
        s = df[f"r_{lab}"].dropna().astype(float); a = df[f"adj_{lab}"].dropna().astype(float)
        if len(s):
            L.append(f"| +{lab} | {s.median():+.1%} | {s.mean():+.1%} | {(s < -0.10).mean():.0%} | {a.median():+.1%} |")
    s = df["r_to_delist"].dropna().astype(float)
    L.append(f"| to the delisting date | {s.median():+.1%} | {s.mean():+.1%} | {(s < -0.10).mean():.0%} | |")
    s = df["min_30d"].dropna().astype(float)
    L.append(f"| 30-day low | {s.median():+.1%} | {s.mean():+.1%} | {(s < -0.10).mean():.0%} | |")
    L.append(f"\nMove in the hour BEFORE the announcement (leak check): median {df['pre_1h'].astype(float).median():+.1%}.\n")
    L.append("| Announced (UTC) | Token | Delist date | pre-1h | +5m | +30m | +1h | +24h | +7d | to delist | 30d low |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    def p(x):
        return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.1%}"
    for r in sorted(ok, key=lambda x: x["announced"]):
        L.append(f"| {r['announced'][:16]} | {r['token']} | {r['delist_date']} | {p(r['pre_1h'])} | {p(r['r_5m'])} | {p(r['r_30m'])} | {p(r['r_1h'])} | {p(r['r_24h'])} | {p(r['r_7d'])} | {p(r['r_to_delist'])} | {p(r['min_30d'])} |")
    (C.OUT / "delistings.md").write_text("\n".join(L))
    print("\n".join(L[:16]))


if __name__ == "__main__":
    main()
