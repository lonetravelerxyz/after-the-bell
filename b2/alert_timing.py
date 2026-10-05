"""Alert-timing experiment: how early did X know, relative to the price and to the official announcement?

Uses twitterapi.io only (env TWITTERAPI_IO_KEY or TWITTERAPIIO_KEY, the Infisical name; $0.15 per 1,000 tweets). Per event it runs a few narrow
searches (T1 security accounts + protocol name, then any account + incident keywords) inside a 3-day window and
keeps the EARLIEST tweet of each kind. Budget guard: at most MAX_PAGES pages of 20 tweets per query and
MAX_TWEETS_TOTAL in all (env overrides ALERT_TIMING_PAGES / ALERT_TIMING_MAX_TWEETS).

Caveat: the API returns newest first, so for a noisy query (any account) the "earliest of the first pages" is biased
late; only the T1 column (few tweets, fully paged) is a real lead-time measurement. The recorded run (2026-09-26)
used 5 pages per query and fetched 2,378 tweets; the defaults below are tighter for reruns.

Inputs: out/b2/incidents.json (hack events with a price alert) and out/b2/delistings.json (Binance
announcements). Output: out/b2/alert_timing.json and alert_timing.md.

Run: infisical run --env dev --path /general -- uv run python -m b2.alert_timing [--dry-run]
     uv run python -m b2.alert_timing --from-json      # rebuild alert_timing.md and the summary from the saved json, no API
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from b2 import config as C

API = "https://api.twitterapi.io/twitter/tweet/advanced_search"
T1 = ["PeckShieldAlert", "CertiKAlert", "SlowMist_Team", "BlockSecTeam", "CyversAlerts", "zachxbt", "lookonchain",
      "AnciliaInc", "Phalcon_xyz", "BeosinAlert", "hexagate_"]
INCIDENT_WORDS = "(exploit OR exploited OR hack OR hacked OR drained OR rug OR rugged OR attacker OR stolen)"
# T1 matches judged by hand after the run (the classifier layer would do this live): excluded from the summary.
MISMATCH = {"FTX": "PeckShieldAlert 2022-11-11 08:28 is about FTX NFTs, not the 2022-11-12 drain"}
RELATED = {"Curve DEX": "CertiKAlert 2023-07-30 14:55 names JPEG'd's pETH-ETH pool, the first victim of the same Vyper bug (entity_match = upstream)"}
MAX_PAGES = int(os.environ.get("ALERT_TIMING_PAGES", "2"))
MAX_TWEETS_TOTAL = int(os.environ.get("ALERT_TIMING_MAX_TWEETS", "1500"))   # hard stop (~$0.23)
NAME_FIX = {"Curve DEX": "Curve", "Cetus CLMM": "Cetus", "Drift Trade": "Drift", "GMX V1 Perps": "GMX", "Balancer V2": "Balancer",
            "Radiant V2": "Radiant", "KyberSwap Elastic": "KyberSwap", "Compound V2": "Compound", "Resolv USR": "Resolv"}


def parse_ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")).tz_convert("UTC")


def search(key: str, query: str, since: pd.Timestamp, until: pd.Timestamp, budget: dict) -> list[dict]:
    q = f"{query} since_time:{int(since.timestamp())} until_time:{int(until.timestamp())}"
    out, cursor = [], ""
    for _ in range(MAX_PAGES):
        if budget.get("tweets", 0) >= MAX_TWEETS_TOTAL:
            budget["capped"] = True
            break
        r = requests.get(API, headers={"X-API-Key": key}, params={"query": q, "queryType": "Latest", "cursor": cursor}, timeout=40)
        if r.status_code != 200:
            budget["errors"] = budget.get("errors", 0) + 1
            break
        body = r.json()
        tw = body.get("tweets") or []
        out.extend(tw)
        budget["tweets"] = budget.get("tweets", 0) + len(tw)
        if not body.get("has_next_page") or not tw:
            break
        cursor = body.get("next_cursor", "")
        time.sleep(0.3)
    return out


def earliest(tweets: list[dict]) -> dict | None:
    rows = []
    for t in tweets:
        try:
            rows.append((parse_ts(t["createdAt"]), t))
        except Exception:  # noqa: BLE001
            continue
    if not rows:
        return None
    ts, t = min(rows, key=lambda x: x[0])
    return {"at": str(ts), "author": (t.get("author") or {}).get("userName"), "text": (t.get("text") or "")[:200], "url": t.get("url")}


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    key = os.environ.get("TWITTERAPI_IO_KEY") or os.environ.get("TWITTERAPIIO_KEY")
    if "--from-json" in argv:
        saved = json.loads((C.OUT / "alert_timing.json").read_text())
        write_md(saved["budget"], saved["rows"])
        return
    if not key and "--dry-run" not in argv:
        print("TWITTERAPI_IO_KEY / TWITTERAPIIO_KEY is not set: run under `infisical run --env dev --path /general` (or --dry-run).", file=sys.stderr)
        sys.exit(2)
    budget: dict = {"pages_per_query": MAX_PAGES, "max_tweets": MAX_TWEETS_TOTAL}
    rows = []
    hacks = [r for r in json.load(open(C.OUT / "incidents.json")) if r.get("t_alert")]
    for r in hacks:
        name = NAME_FIX.get(r["name"], r["name"].split(" ")[0])
        t_alert = pd.Timestamp(r["t_alert"])
        d0 = pd.Timestamp(r["date"], tz="UTC")
        since, until = d0 - pd.Timedelta(days=1), d0 + pd.Timedelta(days=2)
        row = {"kind": "hack", "event": r["name"], "token": r["symbol"], "price_alert": str(t_alert)}
        if key:
            q1 = f"{name} ({' OR '.join('from:' + a for a in T1)})"
            q2 = f"{name} {INCIDENT_WORDS}"
            e1, e2 = earliest(search(key, q1, since, until, budget)), earliest(search(key, q2, since, until, budget))
            row["first_t1"], row["first_any"] = e1, e2
            for k, e in (("t1", e1), ("any", e2)):
                row[f"lead_min_{k}"] = None if e is None else float((t_alert - pd.Timestamp(e["at"])) / pd.Timedelta(minutes=1))
        rows.append(row)
    delist = [r for r in json.load(open(C.OUT / "delistings.json")) if "p0" in r]
    for r in delist:
        t0 = pd.Timestamp(r["announced"])
        row = {"kind": "delist", "event": r["title"], "token": r["token"], "announced": str(t0)}
        if key:
            q = f"{r['token']} (delist OR delisting OR delisted)"
            e = earliest(search(key, q, t0 - pd.Timedelta(hours=6), t0 + pd.Timedelta(hours=1), budget))
            row["first_any"] = e
            row["lead_min_any"] = None if e is None else float((t0 - pd.Timestamp(e["at"])) / pd.Timedelta(minutes=1))
        rows.append(row)
    (C.OUT / "alert_timing.json").write_text(json.dumps({"budget": budget, "rows": rows}, indent=1, default=str))
    write_md(budget, rows)


def summary(rows: list[dict]) -> dict:
    """Lead-time statistics per kind; T1 for hacks (the only unbiased column), any-account for delistings."""
    out = {}
    for kind, col in (("hack", "lead_min_t1"), ("delist", "lead_min_any")):
        v = pd.Series([r[col] for r in rows if r["kind"] == kind and r.get(col) is not None and r["event"] not in MISMATCH], dtype=float)
        n_all = sum(1 for r in rows if r["kind"] == kind)
        out[kind] = {"events": n_all, "with_tweet": int(len(v)), "excluded_mismatch": [e for e in MISMATCH if any(r["event"] == e and r["kind"] == kind for r in rows)], "earlier_than_ref": int((v > 0).sum()),
                     "median_lead_min": None if v.empty else float(v.median()), "min_lead_min": None if v.empty else float(v.min()),
                     "max_lead_min": None if v.empty else float(v.max())}
    return out


def write_md(budget: dict, rows: list[dict]) -> None:
    L = ["# Alert timing (twitterapi.io)\n", f"Tweets fetched: {budget.get('tweets', 0)} (~${budget.get('tweets', 0) * 0.15 / 1000:.2f}), errors {budget.get('errors', 0)}.\n",
         "| Kind | Event | Token | Reference time | First T1 tweet | Lead (min) | First any tweet | Lead (min) |", "|---|---|---|---|---|---:|---|---:|"]
    for r in rows:
        ref = r.get("price_alert") or r.get("announced")
        t1 = r.get("first_t1") or {}
        an = r.get("first_any") or {}
        flag = " (mismatch)" if r["event"] in MISMATCH else " (related)" if r["event"] in RELATED else ""
        L.append(f"| {r['kind']} | {r['event'][:30]} | {r['token']} | {ref[:16]} | {(t1.get('at') or '-')[:16]} {('@' + t1['author']) if t1.get('author') else ''}{flag} | "
                 f"{'' if r.get('lead_min_t1') is None else f'{r['lead_min_t1']:+.0f}'} | {(an.get('at') or '-')[:16]} {('@' + an['author']) if an.get('author') else ''} | "
                 f"{'' if r.get('lead_min_any') is None else f'{r['lead_min_any']:+.0f}'} |")
    L.append("\nLead = reference time minus first tweet time: positive means X was earlier than the price alert / the announcement.")
    S = summary(rows)
    L.append("\n## Summary\n")
    L.append("| Kind | Events | With a tweet | X earlier than the reference | Median lead (min) | Min | Max |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for kind, s in S.items():
        f = lambda x: "-" if x is None else f"{x:+.0f}"
        L.append(f"| {kind} ({'T1 accounts' if kind == 'hack' else 'any account'}) | {s['events']} | {s['with_tweet']} | {s['earlier_than_ref']} | {f(s['median_lead_min'])} | {f(s['min_lead_min'])} | {f(s['max_lead_min'])} |")
    L.append("\nHack reference = first minute the price rule fires (60-min return <= -5% and volume >= 3x the 24h mean); delist reference = Binance announcement time. "
             "The any-account column for hacks is biased late (newest-first paging) and is not used in the summary. "
             + " ".join(f"{k}: {v}." for k, v in list(MISMATCH.items()) + list(RELATED.items())))
    (C.OUT / "alert_timing.md").write_text("\n".join(L))
    (C.OUT / "alert_timing_summary.json").write_text(json.dumps(S, indent=1))
    print("\n".join(L[:6]))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
