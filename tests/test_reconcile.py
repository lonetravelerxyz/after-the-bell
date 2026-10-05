"""Replay-vs-live reconciliation: identity on a relabelled replay, mismatch diagnosis, fill gaps, empty logs."""
import hashlib
import json

import pandas as pd
import pytest

from sentiment import agent, config, evidence, features, ledger, llm, reconcile, replay
from sentiment.config import UNIVERSE, perp
from tests.test_replay import FakeStore

H = pd.Timedelta(hours=1)


@pytest.fixture(autouse=True)
def _risk_v0(monkeypatch):
    """These tests encode the v0 risk rules and record format (flat cap and stop, no beta step, no risk
    context); the v1 counterparts are in test_versions.py, test_agent_v1.py and test_risk_v1.py."""
    monkeypatch.setattr(config, "RISK_VERSION", "v0")


def item_cards(items):
    """Deterministic per item (unlike test_replay.fake_cards, which depends on the list position)."""
    out = []
    for it in items:
        h = int(hashlib.sha256(it["id"].encode()).hexdigest()[:8], 16)
        out.append({"card_id": f"{it['id']}-0", "item_id": it["id"], "kind": "news",
                    "available_at": pd.Timestamp(it["available_at"]), "tickers": [UNIVERSE[h % len(UNIVERSE)]],
                    "scope": "name", "sector": "", "stance": (h % 5) - 2, "strength": 0.8, "horizon_h": 48,
                    "novelty": "new", "summary": f"item {it['id']}", "quote": ""})
    return out


@pytest.fixture
def cached_llm(tmp_path, monkeypatch):
    """LLM cache in tmp; the 'model' answers from a hash of the prompt (no network)."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(llm, "model_name", lambda: "test-model")
    monkeypatch.delenv("SENTIMENT_OFFLINE", raising=False)

    def fake_call(model, messages):
        h = int(hashlib.sha256(messages[-1]["content"].encode()).hexdigest()[:8], 16)
        tg = {tk: round(((h >> i) % 7 - 3) * 0.03, 2) for i, tk in enumerate(UNIVERSE)}
        return json.dumps({"targets": tg, "reasons": {}, "evidence": {}, "confidence": 0.5}), {}
    monkeypatch.setattr(llm, "_call", fake_call)
    return tmp_path


def _run(tmp_path, variant="llm", store=None):
    return replay.run(variant, "2026-07-01", "2026-07-02", store=store or FakeStore(), out_dir=tmp_path / "run",
                      cards_fn=item_cards, quiet=True)


def _relabel(run_dir, dest, edit=None):
    """Replay log -> live-style log in dest (re-chained); `edit(records)` may change records first."""
    recs = reconcile.load_log(run_dir / "log.jsonl", modes=None)
    recs = edit(recs) or recs if edit else recs
    dest.mkdir(parents=True, exist_ok=True)
    lg = ledger.Ledger(dest / "log.jsonl")
    for r in recs:
        lg.append({**r, "mode": "live"})
    (dest / "equity.csv").write_text((run_dir / "equity.csv").read_text())
    return dest


def test_relabelled_replay_reconciles_to_identity_and_zero_gap(cached_llm, monkeypatch):
    run = _run(cached_llm)
    live = reconcile.fabricate(run, cached_llm / "live")
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")                   # any new LLM call would raise
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards)
    assert s["log_verified"] and s["n_decisions"] == 12 and s["n_live_records"] >= 12
    assert s["decision_identity_rate"] == 1.0 and s["prompt_identity_rate"] == 1.0
    assert s["targets_identity_rate"] == 1.0 and s["risk_rebuild_ok_rate"] == 1.0
    assert s["target_l1_max"] < 1e-12 and not s["mismatch_inputs"]
    assert s["n_orders"] > 0 and s["n_matched"] == s["n_orders"] and s["abs_gap_bp_max"] < 1e-9
    csv = pd.read_csv(live / "reconcile.csv")
    assert list(csv.columns) == reconcile.CSV_COLS and len(csv) == s["n_orders"]
    out = json.loads((live / "reconcile.json").read_text())
    assert out["summary"]["n_decisions"] == 12
    d = out["decisions"][0]
    assert d["replay_source"] == "cache (live prompt)" and d["book_source"] == "live prompt"
    assert d["input_identical"] and not d["state_diffs"] and not d["cards_live_only"]


def test_baseline_variant_reconciles(tmp_path):
    run = _run(tmp_path, "baseline")
    s = reconcile.reconcile(reconcile.fabricate(run, tmp_path / "live"), store=FakeStore(), cards_fn=item_cards)
    assert s["decision_identity_rate"] == 1.0 and s["n_matched"] == s["n_orders"] > 0
    assert s["abs_gap_bp_max"] < 1e-9


class LateBarStore(FakeStore):
    """The bar closing at `at` for NVDA has a different close: live saw one version, the Store now has another."""

    def __init__(self, at, bump=1.01):
        super().__init__()
        b = self.b[perp("NVDA")]
        b.loc[at - H, "close"] *= bump


def test_late_bar_shows_as_state_and_prompt_mismatch(cached_llm, monkeypatch):
    run = _run(cached_llm)
    live = reconcile.fabricate(run, cached_llm / "live")
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    t = pd.Timestamp("2026-07-01 16:00", tz="UTC")
    reconcile.reconcile(live, store=LateBarStore(t), cards_fn=item_cards, out_dir=cached_llm / "o")
    dec = {pd.Timestamp(d["ts"]): d for d in json.loads((cached_llm / "o" / "reconcile.json").read_text())["decisions"]}
    d = dec[t]
    assert d["prompt_identical"] is False and d["decision_identical"] is False
    assert "state" in d["mismatch_inputs"]
    fields = {x["field"]: x for x in d["state_diffs"]}
    assert "NVDA.price" in fields and fields["NVDA.price"]["source"] == "perp bars"
    assert fields["NVDA.price"]["replay"] == pytest.approx(fields["NVDA.price"]["live"] * 1.01)
    assert {ln[:5] for ln in d["prompt_diff"]} >= {"-NVDA", "+NVDA"}
    assert d["replay_source"].startswith("not cached")                    # no new LLM call was made
    assert dec[t - 4 * H]["decision_identical"]                           # earlier decisions unaffected


class LateItemStore(FakeStore):
    """One item shows up later in the Store than it did live (e.g. a later first_seen)."""

    def __init__(self, item_id, delay_h=10):
        super().__init__()
        for it in self.items:
            if it["id"] == item_id:
                it["available_at"] = it["available_at"] + delay_h * H


def test_card_availability_mismatch_is_explained(cached_llm, monkeypatch):
    run = _run(cached_llm)
    live = reconcile.fabricate(run, cached_llm / "live")
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    t = pd.Timestamp("2026-07-01 08:00", tz="UTC")                       # it1 is available at 05:00
    reconcile.reconcile(live, store=LateItemStore("it1"), cards_fn=item_cards, out_dir=cached_llm / "o")
    d = next(x for x in json.loads((cached_llm / "o" / "reconcile.json").read_text())["decisions"]
             if pd.Timestamp(x["ts"]) == t)
    assert "evidence" in d["mismatch_inputs"] and not d["prompt_identical"]
    [why] = d["cards_live_only"]
    assert why["card"] == "it1-0" and why["note"].startswith("available after t")
    assert not d["cards_replay_only"] and not d["state_diffs"]


def test_fill_gap_sign_and_live_mid(cached_llm, monkeypatch):
    run = _run(cached_llm)
    edited = {}

    def worse(recs):
        for r in recs:
            for f in r["fills"]:
                if not edited:
                    s = 1 if f["side"] == "buy" else -1
                    f["price"] *= 1 + s * 0.001                             # live paid 10 bp worse
                    f["order_id"] = "shadow-x"
                    edited.update(ticker=f["ticker"], ts=r["ts"], side=f["side"], qty=f["qty"])
        return recs
    live = _relabel(run, cached_llm / "live", worse)
    t, tk = pd.Timestamp(edited["ts"]), edited["ticker"]
    mid = FakeStore().next_open(perp(tk), t) * 1.0002                     # live venue mid 2 bp above the bar open
    (live / "orders.jsonl").write_text(json.dumps({"status": "filled", "live_quote": {"mid": mid},
                                                   "fill": {"order_id": "shadow-x", "qty": edited["qty"]}}) + "\n")
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards)
    csv = pd.read_csv(live / "reconcile.csv")
    row = csv[(pd.to_datetime(csv["ts"]) == t) & (csv["ticker"] == tk)].iloc[0]
    assert row["status"] == "matched" and row["gap_bp"] == pytest.approx(10.0, abs=1e-6)
    sign = 1 if edited["side"] == "buy" else -1
    assert row["mid_gap_bp"] == pytest.approx(sign * 2.0, abs=1e-6)
    assert s["n_matched"] == s["n_orders"] and s["gap_bp_median"] == pytest.approx(0.0, abs=1e-9)
    assert s["abs_gap_bp_max"] == pytest.approx(10.0, abs=1e-6)


def test_unmatched_orders_are_listed(cached_llm, monkeypatch):
    run = _run(cached_llm)

    def drop(recs):
        with_fills = [r for r in recs if r["fills"]]
        r = with_fills[1]                                                   # not the first record: carried book
        r["fills"] = r["fills"][1:]                                         # live missed an order
        return recs
    live = _relabel(run, cached_llm / "live", drop)
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards)
    assert s["n_sim_only"] >= 1 and s["n_matched"] == s["n_orders"] - s["n_sim_only"] - s["n_live_only"]
    assert s["n_positions_breaks"] == 1                     # its post-trade positions no longer follow from its fills


def test_empty_live_dir_does_not_crash(tmp_path):
    s = reconcile.reconcile(tmp_path / "live")                            # no log, no equity, no Store needed
    assert s["n_decisions"] == 0 and s["decision_identity_rate"] is None and s["log_exists"] is False
    assert list(pd.read_csv(tmp_path / "live" / "reconcile.csv").columns) == reconcile.CSV_COLS
    (tmp_path / "live" / "equity.csv").write_text("ts,equity\n2026-09-23 10:00:00+00:00,10000.0\n")
    s = reconcile.reconcile(tmp_path / "live")
    assert s["n_live_records"] == 0 and s["n_orders"] == 0
    reconcile.print_summary(s)


def test_replay_records_are_not_live(tmp_path):
    run = _run(tmp_path, "baseline")
    s = reconcile.reconcile(run, store=FakeStore(), cards_fn=item_cards, write=False)   # mode "replay" only
    assert s["n_live_records"] == 0 and s["n_decisions"] == 0


@pytest.mark.parametrize("variant", ["llm", "blinded"])
def test_prompt_book_roundtrip(variant):
    t = pd.Timestamp("2026-07-01 16:00", tz="UTC")
    state = features.market_state(FakeStore(), t)
    book = {"NVDA": 0.123456, "AAPL": -0.0654, "TSLA": 0.0002, "META": -0.0001}
    msgs = agent.build_messages(t, state, [], book, variant)
    pb = reconcile.prompt_book(msgs, variant == "blinded")
    assert set(pb) == set(book) and pb["NVDA"] == 0.123 and pb["META"] < 0 < pb["TSLA"]
    assert agent.build_messages(t, state, [], pb, variant) == msgs
    assert reconcile.prompt_book(None) is None


def test_card_reader_matches_cards_for_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "CARDS_FILE", tmp_path / "cards.parquet")
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    at = pd.Timestamp("2026-07-01 10:00", tz="UTC")
    news = {"id": "n" * 40, "kind": "news", "available_at": at, "title": "x", "text": "", "tickers": None, "raw": {}}
    gone = {**news, "id": "m" * 40}
    rating = {"id": "r" * 40, "kind": "rating", "available_at": at + H, "title": "r", "text": "", "tickers": ["NVDA"],
              "raw": {"symbol": "NVDA", "action": "调高评级", "rating_previous": "持有", "rating_current": "买入"}}
    card = {"tickers": ["NVDA"], "scope": "name", "sector": "", "stance": 1, "strength": 0.5, "horizon_h": 24,
            "novelty": "new", "summary": "s", "quote": "q"}
    evidence._save(pd.DataFrame(evidence._rows(news, [card], evidence.news_version()), columns=evidence.COLS))
    before = evidence.CARDS_FILE.read_bytes()
    read = reconcile.card_reader()
    got = read([news, gone, rating])
    assert evidence.CARDS_FILE.read_bytes() == before                     # read-only
    assert [it["id"] for it in read.missing] == ["m" * 40]
    assert got == evidence.cards_for([news, rating])                       # same cards as the live path


# ---- regressions: book, mark gap, hourly exits, failed calls, zero-card extractions, broker modes ----------
def _decisions(out_dir):
    return {pd.Timestamp(d["ts"]): d for d in json.loads((out_dir / "reconcile.json").read_text())["decisions"]}


def test_wrong_live_book_is_a_book_mismatch(cached_llm, monkeypatch):
    """A live book that is not what the prompt showed must show in prompt/decision identity, not only in a
    side field (the replay prompt is then built from the reference book)."""
    run = _run(cached_llm)
    recs = reconcile.load_log(run / "log.jsonl", modes=None)
    k = next(i for i, r in enumerate(recs[:-1]) if r["marks"]["positions"])
    t = pd.Timestamp(recs[k + 1]["ts"])

    def triple(rs):
        rs[k]["marks"]["positions"] = {tk: 3 * q for tk, q in rs[k]["marks"]["positions"].items()}
        return rs
    live = _relabel(run, cached_llm / "live", triple)
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards, out_dir=cached_llm / "o")
    d = _decisions(cached_llm / "o")[t]
    assert d["book_check"]["ok"] is False and d["book_check"]["diffs"]
    assert "book" in d["mismatch_inputs"] and d["prompt_identical"] is False
    assert d["decision_identical"] is False and d["book_source"].startswith("replay marks")
    assert s["n_book_mismatch"] >= 1 and s["mismatch_inputs"].get("book", 0) >= 1
    assert s["decision_identity_rate"] < 1.0 and s["n_positions_breaks"] >= 1
    assert "book" in s["identity_scope"]


def _off_hours_record(recs):
    """The decision where off_hours clamps a held short to |b| (book-dependent) in the FakeStore baseline run."""
    for i, r in enumerate(recs):
        for a in r["risk_actions"]:
            if a["rule"] == "off_hours" and abs(abs(a["after"]) - 0.075) > 1e-6:
                return i, a["ticker"]
    raise AssertionError("no book-dependent off_hours clamp in the fixture run")


@pytest.mark.parametrize("logged", [False, True])
def test_book_driven_target_difference(tmp_path, logged):
    """Live marks a few bp off the replay marks move a book-dependent clamp: reported as book-driven (within the
    mark gap) without a logged pre-trade book, exact with one."""
    run = _run(tmp_path, "baseline")
    recs = reconcile.load_log(run / "log.jsonl", modes=None)
    i, tk = _off_hours_record(recs)
    t = pd.Timestamp(recs[i]["ts"])
    e = reconcile.load_equity(run / "equity.csv")[t]
    pos = reconcile.implied_pre_positions(recs[i])
    _, px = reconcile.pre_trade_book(FakeStore(), t, pos, e, {})
    live_px = {**px, tk: px[tk] * 1.0005}                                  # live venue marked tk 5 bp higher

    def live_marks(rs):
        r = rs[i]
        r["pre_trade"] = {"positions": pos, "prices": live_px, "equity": e}
        r["targets"][tk] = reconcile.logged_book(r)[tk]                     # off_hours clamps to the live |b|
        r["marks"]["prices"][tk] *= 1.0005
        if not logged:
            del r["pre_trade"]
        return rs
    live = _relabel(run, tmp_path / "live", live_marks)
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards, out_dir=tmp_path / "o")
    d = _decisions(tmp_path / "o")[t]
    assert d["decision_identical"] and d["targets_match"] and d["risk_rebuild_ok"]
    if logged:
        assert d["targets_identical"] and d["risk_rebuild_exact"] and d["risk_book_source"] == "logged pre-trade book"
        assert s["n_targets_book_driven"] == 0 and s["targets_identity_rate"] == 1.0
    else:
        assert not d["targets_identical"] and not d["risk_rebuild_exact"]
        assert 0 < d["target_l1"] <= d["targets_tol"] < 0.01
        assert s["n_targets_book_driven"] == 1 and s["targets_match_rate"] == 1.0
        assert s["targets_identity_rate"] < 1.0


def test_target_difference_beyond_mark_gap_is_a_mismatch(tmp_path):
    run = _run(tmp_path, "baseline")
    recs = reconcile.load_log(run / "log.jsonl", modes=None)
    i, tk = _off_hours_record(recs)

    def bump(rs):
        rs[i]["targets"][tk] -= 0.01                                        # far more than a mark gap explains
        return rs
    live = _relabel(run, tmp_path / "live", bump)
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards, out_dir=tmp_path / "o")
    d = _decisions(tmp_path / "o")[pd.Timestamp(recs[i]["ts"])]
    assert not d["targets_match"] and not d["risk_rebuild_ok"] and s["targets_match_rate"] < 1.0


class CrashStore(FakeStore):
    """AMZN (long from the 16:00 decision) drops 6% from the bar closing at 18:00: a stop-loss at 18:00."""

    def __init__(self, tk="AMZN", at=pd.Timestamp("2026-07-01 18:00", tz="UTC"), drop=0.06):
        super().__init__()
        b = self.b[perp(tk)]
        b.loc[b.index >= at - H, ["open", "high", "low", "close"]] *= 1 - drop


def test_hourly_exits_are_recomputed(tmp_path):
    run = _run(tmp_path, "baseline", store=CrashStore())
    recs = reconcile.load_log(run / "log.jsonl", modes=None)
    ex = next(i for i, r in enumerate(recs) if r["event"] == "risk_exit")
    assert recs[ex]["risk_actions"][0]["rule"] == "stop_loss"

    s = reconcile.reconcile(reconcile.fabricate(run, tmp_path / "same"), store=CrashStore(), cards_fn=item_cards)
    assert s["n_hourly_checks"] > 20 and s["exit_identity_rate"] == 1.0
    assert s["n_live_only_exits"] == s["n_replay_only_exits"] == 0 and not reconcile.self_check_failures(s)

    live = _relabel(run, tmp_path / "missed", lambda rs: rs[:ex] + rs[ex + 1:])   # live never fired the stop
    s = reconcile.reconcile(live, store=CrashStore(), cards_fn=item_cards)
    assert s["n_replay_only_exits"] >= 1 and s["exit_identity_rate"] < 1.0      # 18:00, and again each hour after
    assert any(f.startswith("exit_identity_rate") for f in reconcile.self_check_failures(s))
    x = next(x for x in json.loads((live / "reconcile.json").read_text())["exits"] if not x["identical"])
    assert pd.Timestamp(x["ts"]) == pd.Timestamp(recs[ex]["ts"])
    assert x["replay_rules"] == ["stop_loss:AMZN"] and x["live_rules"] == []
    assert x["stop_distance"][0]["loss_at_replay_mark"] <= -0.05

    def fake_stop(rs):                                                      # live stopped a name the replay kept
        k = next(i for i, r in enumerate(rs) if r["marks"]["positions"])
        r = rs[k]
        tk = sorted(r["marks"]["positions"])[0]
        rs.insert(k + 1, {**r, "ts": str(pd.Timestamp(r["ts"]) + H), "event": "risk_exit", "decision": None,
                          "state": None, "llm_request_hash": None, "input_hash": None, "evidence_ids": [],
                          "risk_actions": [{"rule": "stop_loss", "ticker": tk, "before": 0.1, "after": 0.0}],
                          "targets": {}, "fills": [],
                          "marks": {**r["marks"], "positions": dict(r["marks"]["positions"])}})
        return rs
    s = reconcile.reconcile(_relabel(run, tmp_path / "extra", fake_stop), store=CrashStore(), cards_fn=item_cards)
    assert s["n_live_only_exits"] == 1


def test_self_check_requires_post_risk_identity(tmp_path, monkeypatch):
    """A broken risk rebuild must fail the self-check even though decisions and fills still match."""
    run = _run(tmp_path, "baseline")
    real = reconcile.risk.apply

    def halved(*a, **k):
        tg, acts = real(*a, **k)
        return {tk: v / 2 for tk, v in tg.items()}, acts
    monkeypatch.setattr(reconcile.risk, "apply", halved)
    s = reconcile.reconcile(reconcile.fabricate(run, tmp_path / "live"), store=FakeStore(), cards_fn=item_cards)
    assert s["decision_identity_rate"] == 1.0 and s["n_matched"] == s["n_orders"]
    fails = reconcile.self_check_failures(s)
    assert any(f.startswith("targets_identity_rate") for f in fails)
    assert any(f.startswith("risk_rebuild_exact_rate") for f in fails)


def test_failed_live_call_is_unverifiable_not_a_mismatch(cached_llm, monkeypatch):
    run = _run(cached_llm)
    t = pd.Timestamp("2026-07-01 16:00", tz="UTC")

    def fail(rs):
        r = next(r for r in rs if pd.Timestamp(r["ts"]) == t)
        r["error"] = "HTTP 500"                                             # the live call failed: hold
        r["llm_request_hash"] = "0" * 64                                    # its prompt was never cached
        r["decision"] = {**r["decision"], "valid": False, "targets": {}}
        return rs
    live = _relabel(run, cached_llm / "live", fail)
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards, out_dir=cached_llm / "o")
    d = _decisions(cached_llm / "o")[t]
    assert d["decision_identical"] is None and d["unverifiable"].startswith("live LLM call failed")
    assert "live prompt not in cache" not in d["mismatch_inputs"]
    assert s["n_unverifiable"] == 1 and s["n_live_errors"] == 1


def test_card_reader_zero_card_extraction_is_not_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "CARDS_FILE", tmp_path / "cards.parquet")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(llm, "model_name", lambda: "test-model")
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    at = pd.Timestamp("2026-07-01 10:00", tz="UTC")
    item = lambda c, title: {"id": c * 40, "kind": "news", "available_at": at, "title": title,  # noqa: E731
                             "text": "NVIDIA said demand is strong.", "tickers": None, "raw": {}}
    empty, full, never = item("e", "macro wrap"), item("f", "NVIDIA demand"), item("g", "not extracted")
    (tmp_path / "llm").mkdir()
    for it, reply in ((empty, {"cards": []}),
                      (full, {"cards": [{"tickers": ["NVDA"], "scope": "name", "stance": 1, "strength": 0.6,
                                         "horizon_h": 24, "novelty": "new", "summary": "demand strong",
                                         "quote": "demand is strong"}]})):
        key = llm.cache_key("test-model", evidence.news_messages(it))
        (tmp_path / "llm" / f"{key}.json").write_text(json.dumps({"response": json.dumps(reply)}))
    read = reconcile.card_reader()
    got = read([empty, full, never])
    assert [it["id"] for it in read.missing] == [never["id"]]              # a zero-card extraction is not missing
    assert not evidence.CARDS_FILE.exists()                                # read-only
    assert got == evidence.cards_for([empty, full])                        # same cards as the live path


def _live_run_id(rs, mode):
    for r in rs:
        r["run_id"] = f"live_baseline_{mode}"
    return rs


def test_fill_status_follows_broker_mode(tmp_path):
    run = _run(tmp_path, "baseline")
    recs = reconcile.load_log(run / "log.jsonl", modes=None)
    r0 = next(r for r in recs if r["fills"])
    f = r0["fills"][0]
    t = pd.Timestamp(r0["ts"])
    bar = FakeStore().next_open(perp(f["ticker"]), t)
    live = _relabel(run, tmp_path / "live", lambda rs: _live_run_id(rs, "shadow"))
    (live / "orders.jsonl").write_text(json.dumps({
        "status": "filled", "mode": "shadow", "fill": {"order_id": f["order_id"], "qty": f["qty"]},
        "live_quote": {"mid": bar * 1.0001}, "demo_quote": {"mid": bar * 1.0021}}) + "\n")
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards)
    assert s["broker_modes"] == ["shadow"] and s["status"].startswith("estimated (shadow")
    assert "observed" not in s["status"] and list(s["fills_by_mode"]) == ["shadow"]
    csv = pd.read_csv(live / "reconcile.csv")
    row = csv[(pd.to_datetime(csv["ts"]) == t) & (csv["ticker"] == f["ticker"])].iloc[0]
    assert row["mode"] == "shadow"
    assert row["demo_live_mid_bp"] == pytest.approx((1.0021 / 1.0001 - 1) * 1e4, abs=1e-6)
    sign = 1 if f["side"] == "buy" else -1
    assert row["demo_mid_gap_bp"] == pytest.approx(sign * 21.0, abs=1e-6)
    assert s["fills_by_mode"]["shadow"]["demo_live_mid_bp_median"] == pytest.approx(row["demo_live_mid_bp"])
    demo = _relabel(run, tmp_path / "demo", lambda rs: _live_run_id(rs, "demo"))
    s = reconcile.reconcile(demo, store=FakeStore(), cards_fn=item_cards, write=False)
    assert s["status"].startswith("observed (Bitget Demo")


def test_broker_mode_switch_does_not_carry_the_book(tmp_path):
    """Shadow records, then demo records that start flat, in one live dir: each mode keeps its own book."""
    a = replay.run("baseline", "2026-07-01", "2026-07-02", store=FakeStore(), out_dir=tmp_path / "a",
                   cards_fn=item_cards, quiet=True)
    b = replay.run("baseline", "2026-07-02", "2026-07-02", store=FakeStore(), out_dir=tmp_path / "b",
                   cards_fn=item_cards, quiet=True)
    cut = pd.Timestamp("2026-07-02", tz="UTC")
    ra = [r for r in reconcile.load_log(a / "log.jsonl", modes=None) if pd.Timestamp(r["ts"]) < cut]
    rb = reconcile.load_log(b / "log.jsonl", modes=None)
    assert ra[-1]["marks"]["positions"]                                     # the shadow book is not flat
    live = tmp_path / "live"
    live.mkdir()
    lg = ledger.Ledger(live / "log.jsonl")
    for r in _live_run_id(ra, "shadow") + _live_run_id(rb, "demo"):
        lg.append({**r, "mode": "live"})
    ea, eb = pd.read_csv(a / "equity.csv"), pd.read_csv(b / "equity.csv")
    pd.concat([ea[pd.to_datetime(ea["ts"]) < cut], eb]).to_csv(live / "equity.csv", index=False)
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards)
    assert s["broker_modes"] == ["demo", "shadow"] and s["status"].startswith("mixed")
    assert s["n_live_only"] == s["n_sim_only"] == 0 and s["n_matched"] == s["n_orders"] > 0
    assert s["n_positions_breaks"] == 0
