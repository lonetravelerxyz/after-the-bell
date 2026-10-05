"""Incident-exit pilot: after a hack/exploit becomes visible in the price, does the token keep bleeding?

Events: DefiLlama /hacks with amount >= MIN_USD since 2021, mapped to a Bitget spot symbol (protocol symbol
from /protocols, plus manual overrides). Prices: Bitget spot 1-minute candles for [day0-1d, day0+2d] and
1-hour candles for [day0, day0+31d]. "Alert proxy" t_a = the first minute on the event day (UTC) whose
trailing 60-minute return is <= ALERT_DROP: the moment a price-only watcher would notice. Measured: the
token's return from the close at t_a to +1h/+6h/+24h/+72h/+7d/+30d and the 30-day low, and BTC over the
same windows (market-adjusted = token - BTC). Also the part that cannot be avoided: day0 00:00 -> t_a.

Run: uv run python -m b2.incident_pilot  -> out/b2/incidents.json, incidents.md
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import requests

from b2 import config as C

MIN_USD = 20e6
START = pd.Timestamp("2021-01-01", tz="UTC")
ALERT_DROP = -0.05
BITGET = "https://api.bitget.com"
OVERRIDES = {"FTX": "FTT", "Ronin Bridge": "RON", "Ronin Network": "RON", "Harmony Horizon Bridge": "ONE",
             "Multichain": "MULTI", "Bitget": "BGB", "Curve": "CRV", "Curve Finance": "CRV", "Euler": "EUL",
             "Euler Finance": "EUL", "Mango Markets": "MNGO", "KyberSwap": "KNC", "Radiant Capital": "RDNT",
             "Drift Protocol": "DRIFT", "Drift": "DRIFT", "Balancer": "BAL", "Cream Finance": "CREAM",
             "BadgerDAO": "BADGER", "Badger DAO": "BADGER", "Compound": "COMP", "Sushi": "SUSHI", "SushiSwap": "SUSHI",
             "Gala Games": "GALA", "Gala": "GALA", "Hedera": "HBAR", "Alphapo": None, "CoinDCX": None, "WazirX": None,
             "Bybit": None, "Binance Bridge": "BNB", "BNB Bridge": "BNB", "Poly Network": None, "Wormhole": None,
             "Portal": None, "Nomad Bridge": None, "Coinbase": None, "Deribit": None, "Kraken": None, "LuBian": None,
             "DMM Bitcoin": None, "Atomic Wallet": None, "Stake.com": None, "Mixin Network": "XIN", "Poloniex": None,
             "HTX": "HT", "Huobi": "HT", "Heco Bridge": "HT", "Orbit Bridge": "ORC", "Orbit Chain": "ORC",
             "Munchables": None, "Prisma Finance": "PRISMA", "Hedgey Finance": None, "Gamma Strategies": "GAMMA",
             "Sonne Finance": "SONNE", "UwU Lend": None, "Velocore": None, "Li.Fi": None, "LI.FI": None,
             "Penpie": "PNP", "Bybit (Feb 2025)": None, "Infini": None, "Cetus": "CETUS", "Cetus Protocol": "CETUS",
             "Nobitex": None, "GMX": "GMX", "CoinDCX (2025)": None, "BtcTurk": None, "Phemex": None, "Zoth": None,
             "Abracadabra": "SPELL", "Abracadabra Money": "SPELL", "Bunni": None, "Resupply": "RSUP",
             "Cork Protocol": None, "Loopscale": None, "Force Bridge": None, "Yearn": "YFI", "Yearn Finance": "YFI",
             "Shibarium": "BONE", "Shibarium Bridge": "BONE", "Liquid Network": None, "Bitget (2026)": "BGB", "UPCX": None, "DEXX": None}


def _get(url, **params):
    for a in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            raise RuntimeError(r.status_code)
        except Exception:  # noqa: BLE001
            if a == 4:
                raise
            time.sleep(1.5 * (a + 1))


def symbol_map() -> dict[str, str]:
    prot = _get("https://api.llama.fi/protocols")
    m = {}
    for p in prot:
        s = p.get("symbol")
        if s and s != "-":
            m.setdefault(p["name"].lower(), s.upper())
            if p.get("parentProtocol"):
                m.setdefault(p["parentProtocol"].replace("parent#", "").lower(), s.upper())
    return m


def events() -> list[dict]:
    hacks = _get("https://api.llama.fi/hacks")
    smap = symbol_map()
    out = []
    for h in hacks:
        if (h.get("amount") or 0) < MIN_USD:
            continue
        t = pd.Timestamp(int(h["date"]), unit="s", tz="UTC")
        if t < START:
            continue
        name = h["name"]
        if name in OVERRIDES:
            sym = OVERRIDES[name]
        else:
            sym = smap.get(name.lower())
            if sym is None:
                sym = next((v for k, v in smap.items() if len(k) >= 4 and (name.lower() in k or k in name.lower())), None)
            if sym is not None and len(sym) < 2:
                sym = None
        if not sym:
            out.append({"name": name, "date": str(t.date()), "amount": h["amount"], "symbol": None, "skip": "no token"})
            continue
        out.append({"name": name, "date": str(t.date()), "amount": h["amount"], "symbol": sym.upper(),
                    "classification": h.get("classification"), "technique": h.get("technique"),
                    "target": h.get("targetType"), "chain": h.get("chain")})
    return out


def candles(symbol: str, gran: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    rows, e = [], int(end.timestamp() * 1000)
    s_ms = int(start.timestamp() * 1000)
    for _ in range(400):
        body = _get(f"{BITGET}/api/v2/spot/market/history-candles", symbol=f"{symbol}USDT", granularity=gran, endTime=e, limit=200)
        page = body.get("data") or []
        if not page:
            break
        rows.extend(page)
        first = int(page[0][0])
        if first <= s_ms or len(page) < 2:
            break
        e = first
        time.sleep(0.15)
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["ts", "open", "high", "low", "close"]
    df = df.astype({"ts": "int64", "close": "float64"}).drop_duplicates("ts").sort_values("ts")
    s = pd.Series(df.close.to_numpy(), index=pd.to_datetime(df.ts, unit="ms", utc=True))
    return s[(s.index >= start) & (s.index <= end)]


def study(ev: dict, btc_min: pd.Series | None = None) -> dict:
    d0 = pd.Timestamp(ev["date"], tz="UTC")
    m = candles(ev["symbol"], "1min", d0 - pd.Timedelta(days=1), d0 + pd.Timedelta(days=2))
    if len(m) < 600:
        return {**ev, "skip": f"no 1min data ({len(m)} rows)"}
    h = candles(ev["symbol"], "1h", d0 - pd.Timedelta(days=1), d0 + pd.Timedelta(days=31))
    b_m = candles("BTC", "1min", d0 - pd.Timedelta(days=1), d0 + pd.Timedelta(days=2))
    b_h = candles("BTC", "1h", d0 - pd.Timedelta(days=1), d0 + pd.Timedelta(days=31))
    day = m[(m.index >= d0) & (m.index < d0 + pd.Timedelta(days=2))]
    trail = day / m.reindex(day.index - pd.Timedelta(minutes=60), method="nearest").to_numpy() - 1
    hit = trail.index[trail <= ALERT_DROP]
    if not len(hit):
        return {**ev, "skip": "no 5%/60min drop on day0-1", "day0_ret": float(day.iloc[-1] / day.iloc[0] - 1)}
    ta = hit[0]
    pa = float(m.asof(ta))

    def ret(series: pd.Series, t1) -> float | None:
        v = series.asof(t1) if t1 <= series.index[-1] else None
        return None if v is None or pd.isna(v) else float(v)
    res = {**ev, "t_alert": str(ta), "unavoidable_day0_to_alert": float(pa / m.asof(d0) - 1),
           "pre_alert_60m": float(trail.loc[ta])}
    for lab, dt in [("1h", pd.Timedelta(hours=1)), ("6h", pd.Timedelta(hours=6)), ("24h", pd.Timedelta(hours=24)),
                    ("72h", pd.Timedelta(hours=72)), ("7d", pd.Timedelta(days=7)), ("30d", pd.Timedelta(days=30))]:
        src, bsrc = (m, b_m) if dt <= pd.Timedelta(hours=24) else (h, b_h)
        p1, b0, b1 = ret(src, ta + dt), ret(bsrc, ta), ret(bsrc, ta + dt)
        r = None if p1 is None else p1 / pa - 1
        rb = None if b0 is None or b1 is None else b1 / b0 - 1
        res[f"r_{lab}"] = r
        res[f"adj_{lab}"] = None if r is None or rb is None else r - rb
    after = h[h.index > ta]
    res["min_30d"] = float(after.min() / pa - 1) if len(after) else None
    res["t_min_30d"] = str(after.idxmin()) if len(after) else None
    return res


def main() -> None:
    C.OUT.mkdir(parents=True, exist_ok=True)
    evs = [e for e in events()]
    print(f"{len(evs)} events >= ${MIN_USD/1e6:.0f}M since {START.date()}, {sum(1 for e in evs if e.get('symbol'))} with a symbol", file=sys.stderr)
    rows = []
    for e in evs:
        if not e.get("symbol"):
            rows.append(e)
            continue
        try:
            r = study(e)
        except Exception as ex:  # noqa: BLE001
            r = {**e, "skip": f"error {str(ex)[:80]}"}
        rows.append(r)
        tag = r.get("skip") or f"alert {r['t_alert'][:16]} unavoid {r['unavoidable_day0_to_alert']:+.1%} then 24h {r['r_24h'] if r['r_24h'] is None else round(r['r_24h'],3)} 7d {r['r_7d'] if r['r_7d'] is None else round(r['r_7d'],3)} min30 {round(r['min_30d'],3) if r['min_30d'] is not None else None}"
        print(f"{e['date']} {e['name'][:28]:28s} {e['symbol']:6s} {tag}", file=sys.stderr)
    (C.OUT / "incidents.json").write_text(json.dumps(rows, indent=1, default=str))
    ok = [r for r in rows if r.get("t_alert")]
    df = pd.DataFrame(ok)
    L = ["# Incident-exit pilot (estimated, Bitget spot candles)\n",
         f"Events: DefiLlama hacks >= ${MIN_USD/1e6:.0f}M since {START.date()}: {len(rows)} listed, {sum(1 for r in rows if r.get('symbol'))} with a token, "
         f"{len(ok)} with Bitget 1-minute data and a >= 5% trailing-60-minute drop on day 0 or day 1 (the alert proxy).\n",
         "| Window after the alert | Median token return | Mean | Share < -10% | Median BTC-adjusted |", "|---|---:|---:|---:|---:|"]
    for lab in ("1h", "6h", "24h", "72h", "7d", "30d"):
        s = df[f"r_{lab}"].dropna().astype(float)
        a = df[f"adj_{lab}"].dropna().astype(float)
        if len(s):
            L.append(f"| +{lab} | {s.median():+.1%} | {s.mean():+.1%} | {(s < -0.10).mean():.0%} | {a.median():+.1%} |")
    s = df["min_30d"].dropna().astype(float)
    L.append(f"| 30-day low | {s.median():+.1%} | {s.mean():+.1%} | {(s < -0.10).mean():.0%} | |")
    u = df["unavoidable_day0_to_alert"].astype(float)
    L.append(f"\nUnavoidable part (day-0 00:00 UTC to the alert minute): median {u.median():+.1%}, mean {u.mean():+.1%}. "
             f"Pre-alert 60-minute drop at the alert: median {df['pre_alert_60m'].astype(float).median():+.1%}.\n")
    L.append("| Date | Event | Token | Loss | Alert (UTC) | Unavoidable | +1h | +24h | +7d | +30d | 30d low |")
    L.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    def p(x):
        return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.1%}"
    for r in sorted(ok, key=lambda x: x["date"]):
        L.append(f"| {r['date']} | {r['name']} | {r['symbol']} | ${r['amount']/1e6:.0f}M | {r['t_alert'][5:16]} | {p(r['unavoidable_day0_to_alert'])} | "
                 f"{p(r['r_1h'])} | {p(r['r_24h'])} | {p(r['r_7d'])} | {p(r['r_30d'])} | {p(r['min_30d'])} |")
    skipped = [r for r in rows if r.get("skip")]
    L.append(f"\nSkipped ({len(skipped)}): " + "; ".join(f"{r['name']} ({r['skip']})" for r in skipped))
    (C.OUT / "incidents.md").write_text("\n".join(L))
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
