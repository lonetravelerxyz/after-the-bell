"""Report generator: empty out/, the existing dev runs (test runs absent -> 'not run yet'), the pre-registered
readings on a stand-in test window, decision chains, live/reconcile, provenance. REPORT.md renders from
metrics.json alone and the report never writes into out/."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from sentiment import metrics, report
from sentiment.config import OUT, ROOT

DEV = ("llm_2026-06-22_2026-08-10", "baseline_2026-06-22_2026-08-10",
       "llm_v1_2026-06-22_2026-07-15", "baseline_v1_2026-06-22_2026-07-15")
TEST = {v: f"{v}_v1_2026-08-11_2026-09-22" for v in ("llm", "baseline", "nonews", "shuffled", "blinded")}
H = pd.Timedelta(hours=1)
T = pd.Timestamp("2026-07-01 00:00", tz="UTC")


def _files(root):
    return sorted((str(p.relative_to(root)), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file())


def _build(tmp_path, out, **kw):
    kw.setdefault("item_ids", None)
    return report.build(out, tmp_path / "report", now="2026-09-23T00:00:00+00:00", **kw)


def _roundtrip(tmp_path):
    rd = tmp_path / "report"
    m = json.loads((rd / "metrics.json").read_text())
    assert report.render(m) == (rd / "REPORT.md").read_text()
    return m, (rd / "REPORT.md").read_text()


# ---- empty out/ -------------------------------------------------------------------------------
def test_empty_out_every_run_not_run_yet(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    m = _build(tmp_path, out, snap=tmp_path / "nosnap", llm_cache=tmp_path / "nocache",
               trials_path=tmp_path / "none.log", diagnosis_path=tmp_path / "none.md")
    assert len(m["runs"]) == len(report.EXPECTED) + 1                              # expected grid + live
    assert all(r["status"] == "not run yet" and "metrics" not in r for r in m["runs"])
    rd = m["readings"]
    assert rd["headline"]["run"] == TEST["llm"] and rd["headline"]["status"] == "not run yet"
    assert rd["increment"]["claim"] is None and rd["increment"]["verdict"].startswith("not run yet")
    assert rd["information"]["claim"] is None and rd["information"]["verdict"].startswith("not run yet")
    assert rd["signal_tests"]["status"] == "not run yet"
    assert [r["status"] for r in rd["signal_tests"]["rows"]] == ["not run yet"] * 5
    assert rd["risk"] == {} and m["chains"]["items"] == [] and m["figures"] == {}
    assert m["live"]["status"] == "not run yet" and m["live"]["reconcile"]["status"] == "not run yet"
    assert m["dev_reference"]["trials"]["status"] == "missing" and not m["dev_reference"]["diagnosis"]["exists"]
    assert all(r["status"] == "not run yet" for r in m["dev_reference"]["runs"])
    assert 1 <= len(m["summary"]) <= 8
    assert "not run yet" in m["summary"][0] and "not run yet" in m["summary"][1]
    _, text = _roundtrip(tmp_path)
    assert text.count("**not run yet**") >= len(m["runs"])
    assert "## 5. Performance net of costs" in text and "### 5c. Dev window (in-sample)" in text and "## 7. Robustness: what else we tried" in text
    assert not list((tmp_path / "report").glob("*.png"))


def test_missing_out_dir_does_not_crash(tmp_path):
    m = _build(tmp_path, tmp_path / "does_not_exist")
    assert all(r["status"] == "not run yet" for r in m["runs"])
    assert len(m["summary"]) <= 8


def test_incomplete_test_run_is_listed_not_dropped(tmp_path):
    out = tmp_path / "out"
    (out / TEST["llm"]).mkdir(parents=True)
    (out / TEST["llm"] / "log.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    m = _build(tmp_path, out)
    r = next(r for r in m["runs"] if r["run_id"] == TEST["llm"])
    assert r["status"] == "incomplete" and r["n_log_records"] == 2 and "in progress" in r["note"]
    assert m["readings"]["headline"]["status"] == "incomplete" and "metrics" not in m["readings"]["headline"]
    assert m["readings"]["runs"]["blinded"]["note"].startswith("optional")


# ---- the existing dev runs (read only; test runs absent) ----------------------------------------
@pytest.fixture
def dev_out(tmp_path):
    if not all((OUT / s / "metrics.json").exists() for s in DEV):
        pytest.skip("dev runs not present in out/")
    out = tmp_path / "out"
    out.mkdir()
    for s in DEV:
        (out / s).symlink_to(OUT / s, target_is_directory=True)
    for f in ("cutoff_probe.json", "signal_tests_dev.json"):
        if (OUT / f).exists():
            shutil.copy(OUT / f, out / f)
    return out


def test_dev_runs_report(tmp_path, dev_out):
    before = _files(OUT / DEV[0]) + _files(OUT / DEV[2])
    m = _build(tmp_path, dev_out, item_ids="auto")
    assert _files(OUT / DEV[0]) + _files(OUT / DEV[2]) == before                    # read only
    runs = {r["run_id"]: r for r in m["runs"]}
    for s in DEV:
        r = runs[s]
        assert r["status"] == "complete" and r["label"] == "estimated (replay)"
        assert r["log_verified"] is True and r["hash_matches_recorded"] is True
        want = metrics.compute(dev_out / s)
        assert r["metrics"]["n_days"] == want["n_days"] and r["metrics"]["total_return"] == pytest.approx(want["total_return"])
        assert r["provenance"]["items_checked"] and not r["provenance"]["stale"]
        assert r["beta"]["status"] == "computed" and r["beta"]["n_days"] == want["n_days"]
    assert runs[DEV[0]]["risk_version"] == "v0" and runs[DEV[2]]["risk_version"] == "v1"
    assert runs[DEV[2]]["version"] == "v1" and runs[DEV[2]]["window"] == "dev-half"
    for name in TEST.values():                                                     # test runs absent
        assert runs[name]["status"] == "not run yet"
    rd = m["readings"]
    assert rd["headline"]["status"] == "not run yet" and rd["risk"] == {}
    assert rd["increment"]["verdict"].startswith("not run yet")

    dev = m["dev_reference"]
    assert [r["status"] for r in dev["runs"]] == ["complete"] * 4
    assert set(dev["increments"]) == {"dev v0", "dev-half v1"}
    for x in dev["increments"].values():                                          # in-sample: no claim sentence
        assert "beats" not in x["reading"] and "trails" not in x["reading"]
    inc = dev["increments"]["dev v0"]["increment"]
    assert inc["n_boot_used"] + inc["n_boot_dropped"] == inc["n_boot"] == 2000
    assert len(dev["diagnosis"]["digest"]) == 5 and dev["diagnosis"]["exists"]
    ex = dev["trials"]["explainability"]
    assert ex and ex["verdict"].startswith("FAIL") and ex["before_after"]["cooldown_rerequests"] == [14, 10]
    if (dev_out / "signal_tests_dev.json").exists():
        assert dev["signal_tests"]["status"] == "complete" and dev["signal_tests"]["window_ok"] is True
        assert [r["id"] for r in dev["signal_tests"]["rows"]] == [i for i, _ in report.SIGNAL_TESTS]

    s = m["summary"]
    assert 1 <= len(s) <= 8
    assert s[0].startswith("Headline") and "not run yet" in s[0]
    assert any("Lost over the full dev window" in x and "v0 LLM" in x for x in s)          # dev loss
    assert any(x.startswith("No sentiment signal in dev") for x in s)                      # no dev signal
    assert any("FAILED its pre-declared explainability" in x for x in s)                   # explainability
    assert any("trials.log entry 5" in x for x in s) and not any("14:40" in x for x in s)   # entry, not the stamp

    ch = m["chains"]
    assert ch["run"] == DEV[2] and len(ch["items"]) == report.N_CHAINS              # v1 half-dev: newest LLM run
    sizes = [c["executed_size"] for c in ch["items"]]
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] > 0
    recs = {r["hash"]: r for r in metrics.load_records(dev_out / DEV[2])}
    for c in ch["items"]:
        r = recs[c["record_hash"]]
        shown = set(r["evidence_ids"])
        eq = r["equity"]
        for row in c["rows"]:
            fl = [f for f in r["fills"] if f["ticker"] == row["ticker"]]
            want = sum(f["qty"] * f["price"] * (1 if f["side"] == "buy" else -1) for f in fl) / eq
            assert row["executed_dw"] == pytest.approx(want)
            assert row["requested"] == pytest.approx(r["decision"]["targets"].get(row["ticker"], 0.0))
            assert row["final_target"] == pytest.approx(r["targets"].get(row["ticker"], 0.0))
            assert [st["rule"] for st in row["risk_steps"]] == [a["rule"] for a in r["risk_actions"]
                                                                if a.get("ticker") == row["ticker"]]
            for k in row.get("cards", []):
                assert k["card_id"] in shown and k["in_prompt_text"]                 # only cards the prompt showed
    _, text = _roundtrip(tmp_path)
    assert "### 8c. Decisions: requested → risk actions → executed → fills" in text
    assert "Event → decision → execution" not in text


def test_every_number_comes_from_metrics_json(tmp_path, dev_out):
    _build(tmp_path, dev_out)
    _roundtrip(tmp_path)


# ---- pre-registered readings on a stand-in test window --------------------------------------------
@pytest.fixture
def standin_out(tmp_path):
    """The v1 half-dev runs copied under test-window names: exercises the readings' mechanics (not results)."""
    if not all((OUT / s / "metrics.json").exists() for s in DEV[2:]):
        pytest.skip("v1 dev runs not present in out/")
    out = tmp_path / "out"
    out.mkdir()
    for v, src in (("llm", DEV[2]), ("baseline", DEV[3]), ("nonews", DEV[2]), ("shuffled", DEV[3])):
        (out / TEST[v]).symlink_to(OUT / src, target_is_directory=True)
    return out


def test_prereg_readings_mechanics(tmp_path, standin_out):
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps({"window": {"start": "2026-08-11", "end": "2026-09-22"}, "tests": {
        "a_ic_4h": {"n": 250, "estimate": 0.01, "t": 0.5}, "a_ic_24h": {"n": 240, "estimate": 0.05, "t": 2.7},
        "a_ic_72h": {"n": 230, "estimate": -0.02, "t": -2.59},
        "b_rating_24h": {"n": 40, "estimate": 0.004, "estimate_bp": 40.0, "t": 1.1}}}))
    m = _build(tmp_path, standin_out, signal=sig)
    rd = m["readings"]
    h = rd["headline"]
    assert h["status"] == "complete" and h["run"] == TEST["llm"] and h["versions_ok"] is True
    assert h["label"] == "estimated (replay)" and h["metrics"]["n_days"] == metrics.compute(standin_out / TEST["llm"])["n_days"]
    inc = rd["increment"]
    assert inc["increment"]["run_a"] == TEST["llm"] and inc["increment"]["run_b"] == TEST["baseline"]
    lo = inc["increment"]["ci95"][0]
    assert inc["claim"] is (lo > 0)
    info = rd["information"]
    assert info["nonews"]["increment"]["sharpe_diff"] == 0                          # the same run: CI includes 0
    assert info["nonews"]["claim"] is False and "not distinguishable" in info["nonews"]["verdict"]
    assert info["claim"] is False and "not attributable to the news" in info["verdict"]
    st = rd["signal_tests"]
    assert st["window_ok"] is True and st["n_significant"] == 1
    by = {r["id"]: r for r in st["rows"]}
    assert by["a_ic_24h"]["significant"] and by["a_ic_24h"]["status"] == "observed (replay prices)"
    assert not by["a_ic_72h"]["significant"] and by["a_ic_72h"]["status"] == "not significant"
    assert by["b_rating_24h"]["estimate"] == 40.0 and by["b_rating_24h"]["estimate_unit"] == "bp"
    assert by["c_news_new_24h"]["status"] == "missing from the file"
    assert set(rd["risk"]) == {TEST[v] for v in ("llm", "baseline", "nonews", "shuffled")}
    for x in rd["risk"].values():
        assert x["beta"]["status"] == "computed" and x["beta"]["within_band"] == (abs(x["beta"]["beta"]) <= 0.03)
        assert "beta_neutral" in x["risk"]["by_rule"] and x["risk"]["effects_priced"]
    assert rd["blinded"]["status"] == "not run yet"
    s = m["summary"]
    assert len(s) <= 8 and "Headline (pre-registered" in s[0] and "reported whatever its sign" in s[0]
    assert inc["verdict"] in s[1] and info["verdict"] in s[2] and "1 of 5 signal tests" in s[2]
    assert m["chains"]["run"] == TEST["llm"]
    _roundtrip(tmp_path)


def test_signal_tests_for_another_window_are_flagged(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"window": {"start": "2026-06-22", "end": "2026-08-10"},
                             "tests": {i: {"t": 0.1, "n": 5} for i, _ in report.SIGNAL_TESTS}}))
    st = report.signal_tests(p, report.WINDOWS["test"])
    assert st["window_ok"] is False and st["n_tests"] == 5 and st["n_significant"] == 0
    p.write_text(json.dumps([{"name": "x", "t_stat": -3.0}]))                        # own ids: shown as given
    st = report.signal_tests(p)
    assert [r["id"] for r in st["rows"]] == ["x"] and st["rows"][0]["significant"]
    assert st["rows"][0]["status"] == "in-sample" and st["n_significant"] == 1       # not the test window


@pytest.mark.parametrize("ci,claim,word", [([0.2, 1.8], True, "claimed: the LLM adds value"),
                                           ([-0.5, 1.8], False, "includes 0"),
                                           ([-2.0, -0.1], False, "trails the baseline")])
def test_increment_claim_rule(ci, claim, word):
    c = report._claim_increment({"increment": {"ci95": ci, "n_days": 40}})
    assert c["claim"] is claim and word in c["verdict"]


def test_information_needs_both_controls(tmp_path):
    comps = {"test v1": {"vs": {o: {"status": "complete", "increment": {"ci95": ci, "sharpe_diff": 1.0}}
                                for o, ci in (("baseline", [0.1, 1]), ("nonews", [0.1, 1]), ("shuffled", [-0.1, 1]))}}}
    sig = {"status": "not run yet", "rows": []}
    rd = report.prereg_readings([], comps, sig, {})
    assert rd["information"]["claim"] is False and "vs shuffled: CI includes 0" in rd["information"]["verdict"]
    comps["test v1"]["vs"]["shuffled"]["increment"]["ci95"] = [0.2, 1.0]
    rd = report.prereg_readings([], comps, sig, {})
    assert rd["information"]["claim"] is True and rd["information"]["verdict"].startswith("claimed: news matters")


# ---- units: chains, prompt parsing, risk effects, beta --------------------------------------------
def rec(t, targets, requested=None, event="decision", actions=(), fills=(), reasons=None, evidence=None,
        shown=(), equity=10_000.0):
    return {"ts": str(t), "event": event, "targets": targets, "risk_actions": list(actions), "fills": list(fills),
            "equity": equity, "variant": "llm",
            "decision": {"targets": requested if requested is not None else targets, "reasons": reasons or {},
                         "evidence": evidence or {}, "confidence": 0.5, "valid": True} if event == "decision" else None,
            "baseline_targets": {}, "state": {}, "evidence_ids": list(shown)}


def _f(tk, side, qty, px):
    return {"ticker": tk, "side": side, "qty": qty, "price": px, "fee": 0.1, "half_spread_cost": 0.01,
            "ts": "x", "order_id": "o"}


def test_chains_rank_by_executed_rebalance_and_show_rules_in_order():
    recs = [rec(T, {"NVDA": 0.05}, fills=[_f("NVDA", "buy", 5, 100.0)]),                     # 0.05
            rec(T + 4 * H, {"NVDA": 0.05, "AAPL": -0.10}, requested={"NVDA": 0.05, "AAPL": -0.20},
                actions=[{"rule": "name_cap", "ticker": "AAPL", "before": -0.20, "after": -0.15},
                         {"rule": "net_cap", "ticker": "AAPL", "before": -0.15, "after": -0.10}],
                fills=[_f("AAPL", "sell", 10, 100.0)], reasons={"AAPL": "bad news"}),           # 0.10
            rec(T + 5 * H, {"AAPL": -0.10}, event="risk_exit", fills=[_f("NVDA", "sell", 50, 100.0)]),
            rec(T + 8 * H, {"AAPL": -0.08}, fills=[_f("AAPL", "buy", 1, 100.0)])]              # 0.01
    ch = report.decision_chains("x", recs, n=2)
    assert [c["ts"] for c in ch] == [str(T + 4 * H), str(T)]                                 # risk exits excluded
    c = ch[0]
    assert c["executed_size"] == pytest.approx(0.10)
    aapl = next(r for r in c["rows"] if r["ticker"] == "AAPL")
    assert aapl["requested"] == -0.20 and aapl["final_target"] == -0.10 and aapl["executed_dw"] == pytest.approx(-0.10)
    assert [s["rule"] for s in aapl["risk_steps"]] == ["name_cap", "net_cap"]
    assert aapl["reason"] == "bad news" and c["focus"] == ["AAPL"]
    assert {r["ticker"] for r in c["rows"]} == {"AAPL"}                                      # NVDA unchanged, no fill


def test_chain_cards_only_from_the_shown_prompt(tmp_path):
    """Cited cards come from the cached prompt text of that decision; a cited id not in the shown set is listed
    apart, and the mutable card cache is never read."""
    from sentiment import agent, llm
    t = T + 16 * H
    mk = lambda cid, tk, stance, summ: {"card_id": cid, "item_id": cid[:-2], "kind": "news",
                                        "available_at": t - 2 * H, "tickers": [tk], "scope": "name", "sector": "",
                                        "stance": stance, "strength": 0.8, "horizon_h": 24, "novelty": "new",
                                        "summary": summ, "quote": "q"}
    shown = [mk("aaaaaaaaaa-0", "NVDA", 1, "NVIDIA demand strong"), mk("bbbbbbbbbb-0", "AAPL", -1, "Apple cut")]
    (tmp_path / "llm").mkdir()
    keys = {}
    for ver, ctx in (("v0", None), ("v1", {"since_card": {}, "caps": {}, "beta": {}})):
        try:
            msgs = agent.build_messages(t, {}, shown, {}, "llm", risk_ctx=ctx)
        except (KeyError, TypeError, AttributeError):
            if ver == "v1":
                continue
            raise
        keys[ver] = llm.cache_key("m", msgs)
        (tmp_path / "llm" / f"{keys[ver]}.json").write_text(json.dumps({"request": {"messages": msgs}, "response": "{}"}))
        cards = report.prompt_cards(keys[ver], tmp_path / "llm")
        assert set(cards) == {"aaaaaaaaaa-0", "bbbbbbbbbb-0"}
        assert cards["aaaaaaaaaa-0"]["stance"] == 1 and cards["aaaaaaaaaa-0"]["summary"] == "NVIDIA demand strong"
        assert cards["bbbbbbbbbb-0"]["subject"] == "AAPL" and cards["aaaaaaaaaa-0"]["age_h"] == 2
    r = rec(t, {"NVDA": 0.1}, reasons={"NVDA": "demand"}, fills=[_f("NVDA", "buy", 10, 100.0)],
            evidence={"NVDA": ["aaaaaaaaaa-0", "bbbbbbbbbb-0", "cccccccccc-0"]},
            shown=["aaaaaaaaaa-0", "bbbbbbbbbb-0"])
    r["llm_request_hash"] = keys["v0"]
    [c] = report.decision_chains("x", [r], n=1, llm_cache=tmp_path / "llm")
    [row] = c["rows"]
    a, b = row["cards"]
    assert a["in_prompt_text"] and a["summary"] == "NVIDIA demand strong" and not a["off_ticker"]
    assert b["off_ticker"] and b["tickers"] == ["AAPL"]
    assert row["cited_not_shown"] == ["cccccccccc-0"]
    assert a["available_at"] == (t - 2 * H).isoformat()
    [c] = report.decision_chains("x", [r], n=1, llm_cache=tmp_path / "none")               # prompt not cached
    assert not c["prompt_cached"] and not c["rows"][0]["cards"][0]["in_prompt_text"]


def test_risk_counts_effects_and_sources():
    bars = {"NVDA": pd.DataFrame({"open": [100.0, 100.0, 100.0, 100.0, 110.0], "close": 100.0},
                                 index=pd.DatetimeIndex([T + i * H for i in range(5)]))}
    recs = [rec(T, {}, actions=[{"rule": "net_cap", "ticker": "NVDA", "before": 0.2, "after": 0.15},
                                {"rule": "off_hours", "ticker": "AAPL", "before": 0.1, "after": 0.075}]),
            rec(T + H, {}, event="risk_exit",
                actions=[{"rule": "daily_kill", "ticker": None, "before": 10000.0, "after": 9690.0},
                         {"rule": "kill", "ticker": "NVDA", "before": 0.15, "after": 0.0}])]
    c = report.risk_counts(recs, bars)["by_rule"]
    assert c["net_cap"]["decision"] == 1 and c["net_cap"]["hourly_check"] == 0
    assert c["daily_kill"]["hourly_check"] == 1 and c["daily_kill"]["weight_moved"] == 0
    assert c["net_cap"]["weight_moved"] == pytest.approx(0.05)
    # cut 0.05 of 10,000 before NVDA's +10% move to the next decision (04:00): -50 USDT
    assert c["net_cap"]["pnl_effect_est"] == pytest.approx(-50.0)
    assert c["kill"]["pnl_effect_est"] == pytest.approx(-150.0)
    assert c["off_hours"]["pnl_effect_est"] is None                               # no AAPL bars: not priced


def test_realised_beta_recovers_a_known_beta(tmp_path):
    days = pd.date_range("2026-07-01", periods=41, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    b = pd.Series(rng.normal(0, 0.02, 40), index=days[1:])
    eq = 10_000 * np.r_[1.0, np.cumprod(1 + 0.5 * b.to_numpy() + rng.normal(0, 1e-4, 40))]
    d = tmp_path / "run"
    d.mkdir()
    pd.DataFrame({"ts": days, "equity": eq, "gross_equity": eq}).to_csv(d / "equity.csv", index=False)
    out = report.realised_beta(d, b)
    assert out["status"] == "computed" and out["n_days"] == 40 and out["beta"] == pytest.approx(0.5, abs=0.01)
    assert out["within_band"] is False


def test_basket_daily_labels_by_closing_midnight():
    idx = pd.date_range("2026-07-01", periods=72, freq="h", tz="UTC")
    bars = {tk: pd.DataFrame({"open": 1.0, "close": np.r_[np.full(24, 100.0), np.full(24, 110.0), np.full(24, k)]},
                             index=idx) for tk, k in (("NVDA", 121.0), ("AAPL", 99.0))}
    s = report.basket_daily(bars)
    assert list(s.index) == [pd.Timestamp("2026-07-03", tz="UTC"), pd.Timestamp("2026-07-04", tz="UTC")]
    assert s.iloc[0] == pytest.approx(0.10) and s.iloc[1] == pytest.approx((0.10 + (99 / 110 - 1)) / 2)


# ---- live, reconcile, labels, partial days ----------------------------------------------------------
def _live_dir(out, mode, hours, records=()):
    d = out / "live"
    d.mkdir(parents=True)
    (d / f"{mode}_state.json").write_text("{}")
    rows = [f"{T + 10 * H + i * H},{10000.0 + i},{10000.0 + i},0,0,0,{10000.0 + i},0,0" for i in range(hours)]
    (d / "equity.csv").write_text("ts,equity,gross_equity,fees,spread,funding,cash,gross_exposure,net_exposure\n"
                                  + "\n".join(rows) + "\n")
    if records:
        from sentiment import ledger
        lg = ledger.Ledger(d / "log.jsonl")
        for r in records:
            lg.append(r)
    return d


@pytest.mark.parametrize("mode,label", [("shadow", "estimated (shadow ledger: simulated fills at live-venue quotes)"),
                                        ("demo", "observed (Bitget Demo fills)")])
def test_live_label_follows_broker_mode(tmp_path, mode, label):
    out = tmp_path / "out"
    _live_dir(out, mode, 2)
    m = _build(tmp_path, out)
    live = next(r for r in m["runs"] if r["run_id"] == "live")
    assert live["label"] == label and m["live"]["label"] == label
    _, text = _roundtrip(tmp_path)
    assert label in text


def test_live_by_risk_version_and_reconcile(tmp_path):
    out = tmp_path / "out"
    recs = [rec(T + 12 * H, {"NVDA": 0.05}, fills=[_f("NVDA", "buy", 5, 100.0)]),
            {**rec(T + 16 * H, {"NVDA": 0.04}), "risk_version": "v1", "prompt_version": "v1"},
            {**rec(T + 17 * H, {}, event="risk_exit"), "risk_version": "v1"}]
    _live_dir(out, "shadow", 9, recs)
    m = _build(tmp_path, out)
    by = m["live"]["by_risk_version"]
    assert by["v0"]["n_decisions"] == 1 and by["v0"]["n_fills"] == 1 and by["v0"]["prompt_versions"] == {"v0": 1}
    assert by["v1"]["n_decisions"] == 1 and by["v1"]["n_risk_exits"] == 1
    assert m["live"]["reconcile"]["status"] == "not run yet"
    rj = tmp_path / "reconcile.json"
    rj.write_text(json.dumps({"summary": {"n_live_records": 3, "n_decisions": 2, "n_identical": 2,
                                          "decision_identity_rate": 1.0, "n_orders": 1, "n_matched": 1,
                                          "fills_by_mode": {"shadow": {"n_orders": 1, "n_matched": 1,
                                                                       "gap_bp_median": -7.0}}}}))
    m = _build(tmp_path, out, reconcile=rj)
    rc = m["live"]["reconcile"]
    assert rc["status"] == "present" and rc["covers_log"] is True and rc["n_identical"] == 2
    assert rc["fills_by_mode"]["shadow"]["label"] == "estimated (shadow ledger: simulated fills at live-venue quotes)"
    assert rc["gap_label"].startswith("estimated (shadow") and rc["identity_label"].startswith("observed")
    assert any("2/2 decisions identical" in s and "fill gap estimated" in s for s in m["summary"])
    assert m["live"]["versions"].startswith("risk v0: 1 records (prompt v0 1)") and "risk v1: 2 records" in m["live"]["versions"]
    _roundtrip(tmp_path)


def test_partial_last_day_is_not_a_daily_return(tmp_path):
    out = tmp_path / "out"
    _live_dir(out, "shadow", 2)                                # 10:00 and 11:00 UTC: no complete day
    m = _build(tmp_path, out)
    x = next(r for r in m["runs"] if r["run_id"] == "live")["metrics"]
    assert x["n_days"] == 0 and x["sharpe"] is None and x["partial_last_day_h"] == 1
    _, text = _roundtrip(tmp_path)
    assert "(+1h partial last day" in text and "not counted" in text


# ---- provenance and code stamps ------------------------------------------------------------------
def _toy_run(d, stamp=None):
    from sentiment import ledger
    d.mkdir(parents=True)
    ts = pd.date_range("2026-08-11", periods=49, freq="h", tz="UTC")
    pd.DataFrame({"ts": ts, "equity": 10_000.0, "gross_equity": 10_000.0, "fees": 0.0, "spread": 0.0,
                  "funding": 0.0}).to_csv(d / "equity.csv", index=False)
    r = ledger.Ledger(d / "log.jsonl").append({"ts": str(ts[0]), "event": "decision", "fills": [],
                                               "evidence_ids": ["abcdef0123-0"], "decision": {"valid": True}})
    rec_ = {"risk_version": "v1", "prompt_version": "v1", "log_last_hash": r["hash"]}
    if stamp:
        rec_["code_stamp"] = stamp
    (d / "metrics.json").write_text(json.dumps(rec_))
    return d


def test_unknown_evidence_items_make_a_run_stale(tmp_path):
    out = tmp_path / "out"
    _toy_run(out / TEST["llm"])
    m = _build(tmp_path, out, item_ids={"zzzzzzzzzz"})
    r = next(r for r in m["runs"] if r["run_id"] == TEST["llm"])
    assert r["status"] == "stale" and "older availability rule" in r["note"]
    assert m["readings"]["headline"]["status"] == "stale" and "metrics" not in m["readings"]["headline"]
    assert any("stale runs" in s for s in m["summary"])
    m = _build(tmp_path, out, item_ids={"abcdef0123"})
    assert next(r for r in m["runs"] if r["run_id"] == TEST["llm"])["status"] == "complete"


def test_legacy_code_hash_mismatch_is_stale(tmp_path):
    out = tmp_path / "out"
    _toy_run(out / TEST["baseline"], stamp="0" * 64)
    m = _build(tmp_path, out)
    r = next(r for r in m["runs"] if r["run_id"] == TEST["baseline"])
    assert r["status"] == "stale" and "code_stamp" in r["note"]
    shutil.rmtree(out)
    _toy_run(out / TEST["baseline"], stamp=report.code_hash())
    m = _build(tmp_path, out)
    assert next(r for r in m["runs"] if r["run_id"] == TEST["baseline"])["status"] == "complete"


def test_git_code_stamp_is_checked_against_the_prereg_freeze():
    from sentiment import replay
    g = report.git_info()
    if not g["head"] or not g["prereg_freeze_commit"]:
        pytest.skip("not a git checkout with the PREREG freeze commit")
    if report._git("cat-file", "-e", g["prereg_freeze_commit"] + "^{commit}") is None:
        pytest.skip("the PREREG freeze commit is not in this checkout (the public repo ships one commit; PROVENANCE names the freeze)")
    stamp = replay.code_stamp()
    assert stamp and stamp.split("-")[0] == g["head"]
    c = report.code_check(g["prereg_freeze_commit"], g["prereg_freeze_commit"])
    assert c["files_differing_from_freeze"] == [] and c["matches_freeze"] is True
    c = report.code_check(g["prereg_freeze_commit"] + "-dirty", g["prereg_freeze_commit"])
    assert c["matches_freeze"] is False and "uncommitted" in c["check"]
    assert "no code_stamp" in report.code_check(None, g["prereg_freeze_commit"])["check"]
    assert report.code_check("0000000", g["prereg_freeze_commit"])["check"].startswith("commit 0000000 not found")


def test_report_never_imports_the_llm_client():
    src = (ROOT / "sentiment" / "report.py").read_text()
    assert "from sentiment import llm" not in src and "llm.complete" not in src and "import llm" not in src


# ---- review fixes: labels, readings, PREREG freeze facts, reproduction commands -----------------------
def test_only_demo_fills_are_observed():
    assert report.live_label(["demo"]) == "observed (Bitget Demo fills)"
    assert report.live_label(["shadow"]).startswith("estimated (shadow ledger")
    assert report.live_label([]).startswith("unverified") and "observed" not in report.live_label([])
    assert report.gap_label({"shadow": {"n_matched": 3}}).startswith("estimated")
    assert report.gap_label({"demo": {"n_matched": 3}}).startswith("observed")
    assert report.gap_label({"demo": {"n_matched": 1}, "shadow": {"n_matched": 1}}).startswith("mixed")
    assert report.gap_label({"shadow": {"n_matched": 0}}).startswith("not computed")


def test_header_says_only_demo_fills_are_observed(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _build(tmp_path, out)
    _, text = _roundtrip(tmp_path)
    assert "only Bitget Demo fills are **observed**" in text and "Live numbers are **observed**" not in text


def _inc(lo, hi, days=40):
    return {"ci95": [lo, hi], "sharpe_diff": (lo + hi) / 2, "n_days": days, "n_boot_used": 2000, "n_boot": 2000}


def test_one_control_alone_never_reads_as_a_news_effect():
    comps = {"test v1": {"vs": {"baseline": {"status": "complete", "increment": _inc(-0.5, 1.5)},
                                "nonews": {"status": "complete", "increment": _inc(0.2, 2.0)},
                                "shuffled": {"status": "complete", "increment": _inc(-0.5, 1.5)}}}}
    rd = report.prereg_readings([], comps, {"status": "not run yet", "rows": []}, {})
    assert rd["information"]["claim"] is False and "not attributable" in rd["information"]["verdict"]
    for other in ("nonews", "shuffled"):
        txt = report.reading(_inc(0.2, 2.0), other)
        assert "does not establish" in txt and "both nonews and shuffled" in txt
        assert "news adds" not in txt and "depends on news timing" not in txt


def test_no_claim_sentence_outside_the_test_window():
    inc = _inc(5.34, 15.72, days=24)
    assert "beats the fixed-rule baseline" in report.reading(inc, "baseline")
    txt = report.reading(inc, "baseline", claim=False)
    assert "beats" not in txt and "no claim" in txt and "[+5.34, +15.72]" in txt


def _git_repo(root):
    import subprocess
    def git(*a):
        return subprocess.run(["git", *a], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    return git


def test_prereg_deviations_are_not_an_integrity_failure(tmp_path, monkeypatch):
    import shutil as _sh
    if not _sh.which("git"):
        pytest.skip("git not available")
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    git = _git_repo(root)
    pre = root / "docs" / "PREREG.md"
    body = "# Pre-registration\n\nStatus: draft.\n\n## Rules\n\n- rule one\n\n## Deviations\n\n(none)\n"
    pre.write_text(body)
    git("add", "-A"); git("commit", "-qm", "draft")
    pre.write_text(body.replace("draft.", "**FROZEN**."))
    git("add", "-A"); git("commit", "-qm", "freeze")
    freeze = git("rev-parse", "--short", "HEAD")
    monkeypatch.setattr(report, "ROOT", root)
    monkeypatch.setattr(report, "PREREG_FILE", pre)
    nofile = tmp_path / "none.json"
    g = report.git_info(nofile)
    assert g["prereg_freeze_commit"] == freeze and g["prereg_freeze_time_utc"].endswith("+00:00")
    assert g["prereg_changed_since_freeze"] is False and g["prereg_deviations_added"] is False
    pre.write_text(pre.read_text().replace("(none)", "2026-09-24: bug X fixed, rerun added."))
    git("add", "-A"); git("commit", "-qm", "deviation")
    g = report.git_info(nofile)
    assert g["prereg_freeze_commit"] == freeze                                   # still the freeze, not the latest
    assert g["prereg_changed_since_freeze"] is False and g["prereg_deviations_added"] is True
    pre.write_text(pre.read_text().replace("rule one", "rule two"))
    assert report.git_info(nofile)["prereg_changed_since_freeze"] is True
    # the exported repo has no research history: PROVENANCE.json carries the freeze facts
    fixed = report.prereg_parts(pre.read_text().replace("rule two", "rule one"))[0]
    pj = tmp_path / "PROVENANCE.json"
    pj.write_text(json.dumps({"prereg_freeze_commit": "abc1234", "prereg_freeze_time": "2026-09-23T21:32:47+08:00",
                              "prereg_fixed_sha256_at_freeze": report._sha(fixed),
                              "prereg_deviations_at_freeze": "(none)", "code_checks": {"x": {"check": "ok"}}}))
    g = report.git_info(pj)
    assert g["prereg_freeze_commit"] == "abc1234" and g["prereg_freeze_time_utc"] == "2026-09-23T13:32:47+00:00"
    assert g["prereg_changed_since_freeze"] is True and g["prereg_deviations_added"] is True
    assert g["export_code_checks"] == {"x": {"check": "ok"}}


def test_code_check_ignores_files_off_the_replay_path():
    assert "sentiment/report.py" not in report.REPLAY_CODE_PATHS
    assert "sentiment/replay.py" in report.REPLAY_CODE_PATHS and "sentiment/prompts" in report.REPLAY_CODE_PATHS
    parent = report._git("rev-parse", "--short", "8fbc622^")
    if not parent:
        pytest.skip("commit 8fbc622 (reconcile.py + report.py only) not in this checkout")
    c = report.code_check("8fbc622", parent)
    assert c["files_differing_from_freeze"] == [] and c["matches_freeze"] is True


def test_code_provenance_and_freeze_rerun(tmp_path):
    runs = [{"run_id": TEST[v], "mode": "replay", "window": "test", "status": "complete", "log_last_hash": h,
             "code": c} for v, h, c in (("baseline", "h1", {"stamp": None, "check": "no code_stamp recorded"}),
                                        ("llm", "h2", {"stamp": "62286d9-dirty", "matches_freeze": False,
                                                       "check": "dirty"}))]
    git = {"prereg_freeze_commit": "62286d9", "source": "export"}
    none = report.freeze_rerun(tmp_path / "missing.json", runs)
    cp = report.code_provenance(runs, git, none)
    assert none["status"] == "not run yet" and cp["n_unverified"] == 2 and "not yet verified" in cp["verdict"]
    f = tmp_path / "freeze_rerun.json"
    f.write_text(json.dumps({"freeze_commit": "62286d9", "runs": {TEST["baseline"]: {"log_last_hash": "h1"},
                                                                  TEST["llm"]: {"log_last_hash": "h2"}}}))
    cp = report.code_provenance(runs, git, report.freeze_rerun(f, runs))
    assert cp["n_unverified"] == 0 and cp["verdict"].startswith("all 2 test runs used")
    f.write_text(json.dumps({"runs": {TEST["llm"]: {"log_last_hash": "other"}}}))
    cp = report.code_provenance(runs, git, report.freeze_rerun(f, runs))
    assert "does NOT reproduce" in cp["verdict"] and TEST["llm"] in cp["verdict"]


def test_deviations_in_summary_and_report(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    m = _build(tmp_path, out)
    dv = m["readings"]["deviations"]
    assert {n["id"] for n in dv["notes"]} >= {"timestamps", "code-after-freeze", "signal-test-c-lookahead"}
    assert any(s.startswith("Deviations and errata") for s in m["summary"]) and len(m["summary"]) <= 8
    _, text = _roundtrip(tmp_path)
    assert "### 9b. Deviations and errata" in text and "13:32:47 UTC" in text


def test_reproduction_commands_never_write_into_shipped_runs():
    runs = [{"run_id": "llm_v1_2026-06-22_2026-07-15", "cards_from": "sentiment/out/llm_2026-06-22_2026-08-10"}]
    rows = report.commands(runs)
    replays = [r["cmd"] for r in rows if "sentiment.replay" in r["cmd"]]
    assert len(replays) == len(report.EXPECTED)
    assert all("--out judge-out/" in c and "--out sentiment/out" not in c and "--out out" not in c for c in replays)
    half = next(c for c in replays if "--variant llm --start 2026-06-22 --end 2026-07-15" in c)
    assert half.endswith("--cards-from out/llm_2026-06-22_2026-08-10")      # pre-split cards_from, new layout
    dev0 = next(c for c in replays if "--variant llm --start 2026-06-22 --end 2026-08-10" in c)
    assert "--risk-version v0" in dev0 and "judge-out/llm_2026-06-22_2026-08-10" in dev0
