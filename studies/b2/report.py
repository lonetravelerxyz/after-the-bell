"""Render out/b2/{metrics.json, events.json, equity.csv, signals.parquet} -> out/b2/REPORT.md + figures.

Run: uv run python -m studies.b2.report
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from studies.b2 import config as C

NAMES = {"buy_hold": "Buy & hold", "ma200": "200-day MA", "voltarget": "40% vol target",
         "module": "Crisis module (v0)", "module_ma200": "Module + 200-day MA"}


def pct(x, nd=1):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:+.{nd}f}%"


def f2(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:.2f}"


def fig_equity(eq: pd.DataFrame, sig: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, lab in NAMES.items():
        ax.plot(eq.index, eq[k], label=lab, lw=1.2 if k in ("module", "buy_hold") else 0.9)
    ax.set_yscale("log")
    out = sig.pos_module == 0
    ax.fill_between(sig.index, 0.5, eq.max().max() * 1.1, where=out, color="grey", alpha=0.15, lw=0, label="module out")
    ax.set_title("BTCUSDT, exposure policies, equity (log), estimated (replay)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.OUT / "equity.png", dpi=130)
    plt.close(fig)


def fig_events(events: list[dict], sig: pd.DataFrame) -> None:
    ev = [e for e in events if e.get("covered")]
    n = len(ev)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(13, 3.2 * ((n + 2) // 3)))
    for ax, e in zip(axes.flat, ev):
        t0 = pd.Timestamp(e["start"], tz="UTC")
        w = sig.loc[t0 - pd.Timedelta(days=7): t0 + pd.Timedelta(days=30)]
        ax.plot(w.index, w.open / float(sig.open.asof(t0)), color="black", lw=1)
        ax.fill_between(w.index, 0, 2, where=w.pos_module == 0, color="red", alpha=0.15, lw=0)
        ax.axvline(t0, color="blue", ls="--", lw=0.8)
        ax.set_ylim(w.open.min() / float(sig.open.asof(t0)) * 0.97, w.open.max() / float(sig.open.asof(t0)) * 1.03)
        tag = f"trig +{e['hours_after_start']:.0f}h {','.join(e['rules'])}" if e.get("triggered") else "NOT triggered"
        ax.set_title(f"{e['id']} {e['name']} ({e['split']}): {tag}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Events: BTC open / price at event start; red = module out of the market", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.OUT / "events.png", dpi=130)
    plt.close(fig)


def v1_sections() -> list[str]:
    grid = json.loads((C.OUT / "v1_grid.json").read_text())
    hold = json.loads((C.OUT / "v1_holdout.json").read_text())
    lead = json.loads((C.OUT / "v1_leadtime.json").read_text()) if (C.OUT / "v1_leadtime.json").exists() else []
    L = ["\n## 7. v1 trials (studies/b2/trials.log): grid on dev 2019-08..2022-12, chosen configuration once on holdout 2023-01..\n",
         "C1 and C3 (major stablecoins only) are always on. Selection on dev: Calmar of the module over its base minus the base's own Calmar, tie-break fewer false alarms.\n",
         "| C2 | C4 | C5 (DVOL) | Re-entry | Base | Dev return | Dev max DD | Dev Calmar | Base Calmar | Increment | Episodes | False alarms | Events hit |",
         "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in grid:
        d = r["dev"]
        L.append(f"| {r['c2'] or 'off'} | {'q95 x 2d' if r['c4'] else 'off'} | {r.get('c5') or 'off'} | {r['reentry']} | {r['base']} | {pct(d['module']['total_return'])} | "
                 f"{pct(d['module']['max_dd'])} | {f2(d['module']['calmar'])} | {f2(d['base']['calmar'])} | {f2(d['calmar_increment'])} | {d['episodes']} | {d['false_alarms']} | {d['events_hit']}/{d['events_n']} |")
    ch, h, refs = hold["chosen"], hold["holdout"], hold["holdout_refs"]
    L.append(f"\n**Chosen on dev**: C2 {ch['c2'] or 'off'}, C4 {'q95 x 2d' if ch['c4'] else 'off'}, C5 {ch.get('c5') or 'off'}, re-entry {ch['reentry']}, base {ch['base']}.\n")
    L.append("| Holdout 2023-01 -> now | Total return | Max DD | Calmar | Sharpe | Time in market | Episodes | False alarms | Events hit |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    L.append(f"| Module (chosen) | {pct(h['module']['total_return'])} | {pct(h['module']['max_dd'])} | {f2(h['module']['calmar'])} | {f2(h['module']['sharpe'])} | {h['module']['time_in_market']*100:.0f}% | {h['episodes']} | {h['false_alarms']} | {h['events_hit']}/{h['events_n']} |")
    for k, lab in (("buy_hold", "Buy & hold"), ("ma200", "200-day MA"), ("voltarget", "40% vol target")):
        m = refs[k]
        L.append(f"| {lab} | {pct(m['total_return'])} | {pct(m['max_dd'])} | {f2(m['calmar'])} | {f2(m['sharpe'])} | {m['time_in_market']*100:.0f}% | | | |")
    if lead:
        keys = ["C1", "C2_v0(2x)", "C2_v1(3x)", "C3", "C4_v0(q90)", "C4_v1(q95x2d)", "C5_dvol"]
        L.append("\n### 7a. Lead time per rule and event (diagnostic)\n")
        L.append("Cell = hours after the event start when the rule first fires (negative = before) and BTC's move from the next open to the 30-day trough, i.e. what an exit at that hour avoids; `*` = fired only after the trough.\n")
        L.append("| Event | BTC start->trough | " + " | ".join(keys) + " |")
        L.append("|---|---:|" + "---:|" * len(keys))
        for r in lead:
            cells = []
            for k in keys:
                v = r.get(k)
                cells.append("-" if v is None else f"{v['lag_h']:+.0f}h {pct(v['avoidable'])}{'*' if v.get('after_trough') else ''}")
            L.append(f"| {r['id']} {r['name']} | {pct(r['btc_start_to_trough'])} | " + " | ".join(cells) + " |")
    L.append("\nReading: detection is not the problem (C2 at 2x fires before or within a day of eight of nine events with 8-48% still to fall); "
             "the false alarms between events and the re-entry rule are. 'recover' (re-enter above the exit price) stays out for the whole 2022 bear and misses 2023-24; "
             "'ma20' re-enters into every bounce. No configuration adds to the 200-day MA on dev; the chosen one fails on holdout.\n")
    return L


def main() -> None:
    m = json.loads((C.OUT / "metrics.json").read_text())
    ev = json.loads((C.OUT / "events.json").read_text())
    eq = pd.read_csv(C.OUT / "equity.csv", index_col=0, parse_dates=True)
    sig = pd.read_parquet(C.OUT / "signals.parquet")
    fig_equity(eq, sig)
    fig_events(ev["events"], sig)
    L = []
    L.append("# B2-C crisis-derisk module v0: report\n")
    L.append(f"Generated by `uv run python -m studies.b2.report` from `out/b2/`. Every number is **{m['label']}**. "
             f"Window {m['window'][0][:10]} -> {m['window'][1][:10]}, {m['bars']} hourly bars ({m['bar_gaps_filled']} gaps carried). "
             "Rules v0 fixed before the run (spec `docs/spec-B2-crisis-derisk.md` section 4), not tuned.\n")
    mets = m["metrics"]
    L.append("## 1. Policies over the full window\n")
    L.append("| Policy | Total return | CAGR | Ann. vol | Sharpe | Max DD | Calmar | Time in market | Position changes | Costs paid |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, lab in NAMES.items():
        x = mets[k]
        L.append(f"| {lab} | {pct(x['total_return'])} | {pct(x['cagr'])} | {pct(x['ann_vol'])} | {f2(x['sharpe'])} | {pct(x['max_dd'])} | "
                 f"{f2(x['calmar'])} | {x['time_in_market']*100:.0f}% | {x['position_changes']} | {pct(x['cost_paid_frac'], 2)} |")
    fs, bf = mets["module_funding_sens"], mets["buy_hold_funding_sens"]
    L.append(f"\nFunding sensitivity (0.01%/8h paid while long): module {pct(fs['total_return'])} (Sharpe {f2(fs['sharpe'])}), "
             f"buy & hold {pct(bf['total_return'])} (Sharpe {f2(bf['sharpe'])}).\n")
    L.append("![equity](equity.png)\n")
    L.append("## 2. Exit episodes and false alarms\n")
    L.append(f"Rule hours true: {m['rule_hours_true']}. Episodes (out-of-market spells): **{m['episodes']}**, first rule of each: "
             f"{m['episodes_first_rule']}. False alarms (BTC did not fall a further 10% within 30 days of the exit): "
             f"**{m['false_alarms']}** ({(m['false_alarm_rate'] or 0)*100:.0f}%), mean BTC move exit->re-entry on false alarms "
             f"{pct(m['false_alarm_missed_mean'])}, on hits {pct(m['hit_missed_mean'])} (negative = the exit avoided a loss).\n")
    L.append("| Exit | Re-entry | Hours out | Rules | BTC min in 30d | BTC exit->re-entry | False alarm | F&G at exit |")
    L.append("|---|---|---:|---|---:|---:|---|---:|")
    for e in ev["episodes"]:
        L.append(f"| {e['exit'][:16]} | {(e['reentry'] or '-')[:16]} | {e['hours_out']} | {','.join(e['rules'])} | {pct(e['min_30d_from_exit'])} | "
                 f"{pct(e['missed'])} | {'yes' if e['false_alarm'] else ''} | {e['fng_at_exit'] if e['fng_at_exit'] is not None else '-'} |")
    L.append("\n## 3. Event study\n")
    L.append(f"Events covered {m['events_covered']}, triggered {m['events_triggered']}.\n")
    L.append("| Event | Split | BTC start->trough (30d) | Triggered | Hours after start | Rules | BTC trigger->trough | Re-entry | BTC trigger->re-entry |")
    L.append("|---|---|---:|---|---:|---|---:|---|---:|")
    for e in ev["events"]:
        if not e.get("covered"):
            L.append(f"| {e['id']} {e['name']} | {e['split']} | not covered | | | | | | |")
            continue
        if e.get("triggered"):
            L.append(f"| {e['id']} {e['name']} | {e['split']} | {pct(e['btc_start_to_trough'])} | yes | {e['hours_after_start']:+.0f} | {','.join(e['rules'])} | "
                     f"{pct(e.get('btc_trigger_to_trough'))} | {(e.get('reentry') or '-')[:16]} | {pct(e.get('btc_trigger_to_reentry'))} |")
        else:
            L.append(f"| {e['id']} {e['name']} | {e['split']} | {pct(e['btc_start_to_trough'])} | **no** | | | | | |")
    L.append("\n![events](events.png)\n")
    L.append("## 4. Year by year\n")
    L.append("| Year | " + " | ".join(NAMES.values()) + " |")
    L.append("|---|" + "---:|" * len(NAMES))
    for row in m["yearly"]:
        L.append(f"| {row['year']} | " + " | ".join(pct(row[k]) for k in NAMES) + " |")
    L.append("\n## 5. Diagnostics: each rule alone and leave-one-out (not used to choose v0)\n")
    L.append("| Variant | Total return | Max DD | Calmar | Sharpe | Time in market | Episodes | False alarms | Events hit (of 9) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in m["ablation"]:
        L.append(f"| {a['variant']} | {pct(a['total_return'])} | {pct(a['max_dd'])} | {f2(a['calmar'])} | {f2(a['sharpe'])} | "
                 f"{a['time_in_market']*100:.0f}% | {a['episodes']} | {a['false_alarms']} | {a['events_hit']} |")
    L.append("\n## 6. Reading against the spec's three criteria\n")
    mod, bh, ma, vt = mets["module"], mets["buy_hold"], mets["ma200"], mets["voltarget"]
    c1 = mod["max_dd"] > bh["max_dd"] and mod["calmar"] > bh["calmar"]
    c2 = mod["calmar"] > max(ma["calmar"], vt["calmar"]) and mod["max_dd"] > min(ma["max_dd"], vt["max_dd"])
    L.append(f"1. Max DD and Calmar better than buy & hold: **{'yes' if c1 else 'no'}** "
             f"(DD {pct(mod['max_dd'])} vs {pct(bh['max_dd'])}; Calmar {f2(mod['calmar'])} vs {f2(bh['calmar'])}).")
    L.append(f"2. Better than the 200-day MA and the 40% vol target on both: **{'yes' if c2 else 'no'}** "
             f"(MA: DD {pct(ma['max_dd'])}, Calmar {f2(ma['calmar'])}; vol target: DD {pct(vt['max_dd'])}, Calmar {f2(vt['calmar'])}).")
    L.append(f"3. False alarms: {m['false_alarms']} of {m['episodes']} episodes; mean missed move on false alarms {pct(m['false_alarm_missed_mean'])} "
             f"vs mean avoided move on hits {pct(m['hit_missed_mean'])}.")
    L.append("\nAll of the above is in-sample: the nine events were known when the rules were written, even though no threshold was tuned on the results.\n")
    if (C.OUT / "v1_holdout.json").exists():
        L += v1_sections()
    (C.OUT / "REPORT.md").write_text("\n".join(L))
    print((C.OUT / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
