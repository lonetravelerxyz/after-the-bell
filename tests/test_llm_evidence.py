"""sentiment.llm cache/offline/retry behaviour, parse_json, and sentiment.evidence cards (no network)."""
from __future__ import annotations

import io
import json
import urllib.error

import pandas as pd
import pytest

from sentiment import evidence as ev
from sentiment import llm

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]


def _resp(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolated cache dirs, a fake model id and no network unless a test stubs `_post`."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(ev, "CARDS_FILE", tmp_path / "cards.parquet")
    monkeypatch.setenv("BITGET_QWEN_MODEL", "test-model")
    monkeypatch.setenv("BITGET_QWEN_API_KEY", "k")
    monkeypatch.setenv("BITGET_QWEN_BASE_URL", "http://invalid.local/v1")
    monkeypatch.delenv("SENTIMENT_OFFLINE", raising=False)
    monkeypatch.setattr(llm, "BACKOFF_S", 0.0)

    def no_net(*a, **k):
        raise AssertionError("network call")
    monkeypatch.setattr(llm, "_post", no_net)
    return tmp_path


# ---------------------------------------------------------------- llm

def test_cache_hit_makes_no_network_call(iso, monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_post", lambda cfg, body: calls.append(body) or _resp('{"a": 1}'))
    assert llm.complete(MSGS, purpose="t") == '{"a": 1}'
    assert calls[0]["temperature"] == 0 and calls[0]["enable_thinking"] is False

    def boom(*a, **k):
        raise AssertionError("network call on cache hit")
    monkeypatch.setattr(llm, "_post", boom)
    assert llm.complete(MSGS, purpose="t") == '{"a": 1}'
    rec = json.loads(next((iso / "llm").glob("*.json")).read_text())
    assert rec["purpose"] == "t" and rec["request"]["messages"] == MSGS and rec["usage"]["total_tokens"] == 15
    assert len(calls) == 1


def test_cache_key_depends_on_model_and_messages():
    k = llm.cache_key("m", MSGS)
    assert k == llm.cache_key("m", [dict(m) for m in MSGS])
    assert k != llm.cache_key("m2", MSGS)
    assert k != llm.cache_key("m", MSGS[:1])


def test_offline_miss_raises(iso, monkeypatch):
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    with pytest.raises(llm.CacheMiss):
        llm.complete(MSGS, purpose="t")


def test_retries_5xx_then_succeeds_and_4xx_fails_fast(iso, monkeypatch):
    n = {"c": 0}

    def flaky(cfg, body):
        n["c"] += 1
        if n["c"] < 3:
            raise urllib.error.HTTPError("u", 503, "busy", {}, None)
        return _resp("ok")
    monkeypatch.setattr(llm, "_post", flaky)
    assert llm.complete(MSGS, purpose="t") == "ok" and n["c"] == 3

    m = {"c": 0}

    def bad(cfg, body):
        m["c"] += 1
        raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"bad request"))
    monkeypatch.setattr(llm, "_post", bad)
    with pytest.raises(llm.LLMError):
        llm.complete(MSGS + [{"role": "user", "content": "x"}], purpose="t")
    assert m["c"] == 1


def test_usage_summary_totals(iso, monkeypatch):
    monkeypatch.setattr(llm, "_post", lambda cfg, body: _resp("x"))
    llm.complete(MSGS, purpose="a")
    llm.complete(MSGS[:1], purpose="b")
    u = llm.usage_summary()
    assert u["calls"] == 2 and u["total_tokens"] == 30 and u["by_purpose"]["a"]["prompt_tokens"] == 10


@pytest.mark.parametrize("text, want", [
    ('{"cards": []}', {"cards": []}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here is the result:\n{"a": {"b": [1, 2]}}\nHope that helps.', {"a": {"b": [1, 2]}}),
    ('<think>maybe {"no": 1}</think>{"a": 2}', {"a": 2}),
    ('{"a": [1, 2,], "b": 3,}', {"a": [1, 2], "b": 3}),
    ('{"ok": True, "x": None}', {"ok": True, "x": None}),
    ('{"q": "a } brace { inside"} trailing', {"q": "a } brace { inside"}),
    ('see [1] below: {"a": 1}', {"a": 1}),
    ('[{"a": 1}]', [{"a": 1}]),
    ('{“a”: “b”}', {"a": "b"}),
])
def test_parse_json_messy(text, want):
    assert llm.parse_json(text) == want


def test_parse_json_raises_without_json():
    with pytest.raises(ValueError):
        llm.parse_json("no json here {unbalanced")


# ---------------------------------------------------------------- deterministic cards

def _rating(**raw) -> dict:
    return {"id": "r" + json.dumps(raw, ensure_ascii=False, sort_keys=True), "kind": "rating",
            "available_at": pd.Timestamp("2026-07-01 13:30", tz="UTC"), "title": "", "text": "",
            "tickers": [raw.get("symbol", "NVDA")], "raw": {"symbol": "NVDA", "analyst_firm": "Firm", **raw}}


@pytest.mark.parametrize("raw, stance, novelty", [
    ({"action": "调高评级", "rating_previous": "持有", "rating_current": "买入", "price_target_previous": 200.0, "price_target": 240.0}, 2, "new"),
    ({"action": "调高评级", "rating_previous": "持有", "rating_current": "买入", "price_target_previous": 200.0, "price_target": 205.0}, 1, "new"),
    ({"action": "下调评级", "rating_previous": "买入", "rating_current": "中性", "price_target_previous": 200.0, "price_target": 190.0}, -1, "new"),
    ({"action": "下调评级", "rating_previous": "买入", "rating_current": "卖出"}, -2, "new"),
    ({"action": "首次覆盖", "rating_current": "增持", "price_target": 300.0}, 1, "new"),
    ({"action": "首次覆盖", "rating_current": "中性", "price_target": 300.0}, 0, "new"),
    ({"action": "首次覆盖", "rating_current": "卖出", "price_target": 100.0}, -1, "new"),
    ({"action": "维持", "rating_previous": "买入", "rating_current": "买入", "price_target_previous": 200.0, "price_target": 220.0}, 1, "new"),
    ({"action": "维持", "rating_previous": "买入", "rating_current": "买入", "price_target_previous": 200.0, "price_target": 260.0}, 2, "new"),
    ({"action": "重申", "rating_previous": "买入", "rating_current": "买入", "price_target_previous": 200.0, "price_target": 180.0}, -1, "new"),
    ({"action": "维持", "rating_previous": "买入", "rating_current": "买入", "price_target_previous": 390.0, "price_target": 390.0}, 0, "recap"),
])
def test_rating_cards(raw, stance, novelty):
    c = ev.rating_card(_rating(**raw))
    assert c["stance"] == stance and c["novelty"] == novelty
    assert c["tickers"] == ["NVDA"] and c["scope"] == "name" and 0 <= c["strength"] <= 1 and len(c["summary"]) <= 200


def test_rating_strength_grows_with_pt_change():
    small = ev.rating_card(_rating(action="维持", rating_current="买入", rating_previous="买入", price_target_previous=200.0, price_target=210.0))
    big = ev.rating_card(_rating(action="维持", rating_current="买入", rating_previous="买入", price_target_previous=200.0, price_target=250.0))
    assert big["strength"] > small["strength"]


def test_rating_outside_universe_has_no_card():
    it = _rating(action="调高评级", rating_current="买入", rating_previous="持有")
    it["tickers"], it["raw"]["symbol"] = ["MSFT"], "MSFT"
    assert ev.rating_card(it) is None


def _insider(typ: str, qty: float, px: float, owned: float = 1e6) -> dict:
    return {"id": f"i{typ}{qty}{px}", "kind": "insider", "available_at": pd.Timestamp("2026-07-02", tz="UTC"),
            "title": "", "text": "", "tickers": ["TSLA"],
            "raw": {"symbol": "TSLA", "transaction_type": typ, "securities_transacted": qty,
                    "transaction_price": px, "securities_owned": owned, "owner_name": "X", "owner_title": "CFO",
                    "filing_date": "2026-07-01"}}


def test_insider_cards():
    buy = ev.insider_card(_insider("P", 10_000, 200.0))          # $2M open-market buy
    assert buy["stance"] == 2 and buy["strength"] >= 0.6
    small_buy = ev.insider_card(_insider("P", 100, 200.0))
    assert small_buy["stance"] == 1
    sell = ev.insider_card(_insider("S", 2_605, 402.0, owned=22_039))  # routine sale
    assert sell["stance"] == -1 and sell["strength"] == pytest.approx(0.1)
    big = ev.insider_card(_insider("S", 100_000, 400.0, owned=1e7))   # $40M
    assert big["stance"] == -1 and big["strength"] == pytest.approx(0.5)
    huge = ev.insider_card(_insider("S", 1e6, 400.0, owned=1e6))      # $400M and half the stake
    assert huge["stance"] == -2
    for typ in ("M", "A", "F", "G"):
        assert ev.insider_card(_insider(typ, 1e6, 23.0)) is None


# ---------------------------------------------------------------- news cards (fake LLM)

def _news(title: str, text: str = "<p>Body mentions NVIDIA and Tesla.</p>") -> dict:
    return {"id": "n-" + title, "kind": "news", "available_at": pd.Timestamp("2026-07-03 12:00", tz="UTC"),
            "title": title, "text": text, "tickers": None, "raw": {}}


FAKE = json.dumps({"cards": [
    {"scope": "name", "tickers": ["NVDA", "TSLA", "MSFT"], "stance": 5, "strength": 1.7, "horizon_h": 48,
     "novelty": "new", "summary": "x" * 300, "quote": "Body mentions ... NVIDIA and Tesla"},
    {"scope": "name", "tickers": ["INTC"], "stance": -1, "strength": 0.3, "horizon_h": 24, "novelty": "new",
     "summary": "no watched ticker", "quote": ""},
    {"scope": "market", "tickers": ["AAPL"], "stance": "-1", "strength": "0.4", "horizon_h": "72",
     "novelty": "weird", "summary": "macro", "quote": "Body mentions NVIDIA"},
]})


def test_news_cards_normalized_and_persisted(iso, monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "complete", lambda messages, *, purpose: calls.append(messages) or "```json\n" + FAKE + "\n```")
    item = _news("Plain headline")
    cards = ev.cards_for([item])
    names = [c for c in cards if c["scope"] == "name"]
    assert sorted(c["tickers"][0] for c in names) == ["NVDA", "TSLA"]          # split, MSFT dropped
    assert all(c["stance"] == 2 and c["strength"] == 1.0 and len(c["summary"]) == 200 for c in names)
    assert names[0]["quote"] == "NVIDIA and Tesla"                             # ellipsis repaired
    sector = [c for c in cards if c["scope"] == "sector"]
    assert sector and sector[0]["tickers"] == [] and sector[0]["sector"] == "other"   # INTC-only name card
    market = next(c for c in cards if c["scope"] == "market")
    assert market["tickers"] == [] and market["stance"] == -1 and market["horizon_h"] == 72 and market["novelty"] == "new"
    assert all(c["item_id"] == item["id"] and c["available_at"] == item["available_at"] for c in cards)
    assert ev.CARDS_FILE.exists() and len(calls) == 1

    again = ev.cards_for([item])                                   # already extracted: no LLM call
    assert len(calls) == 1 and [c["card_id"] for c in again] == [c["card_id"] for c in cards]

    monkeypatch.setattr(ev, "news_version", lambda: "news:changed")   # prompt change -> re-extract, replace rows
    ev.cards_for([item])
    assert len(calls) == 2 and len(ev.load_cards()) == len(cards)


def test_post_hoc_title_forced(iso, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda messages, *, purpose: FAKE)
    for title in ["[Market Movement Analysis] Tesla Surges on September 17: Optimus Orders",
                  "Market Movement Interpretation: Broadcom surges on September 17 as optical ...",
                  "[Market Insight] Corning surges on September 17: Driven by optical communication"]:
        assert {c["novelty"] for c in ev.cards_for([_news(title)])} == {"post_hoc"}
    assert "post_hoc" not in {c["novelty"] for c in ev.cards_for([_news("Bitget UEX Daily | Oil rebounds")])}


def test_unparseable_output_gives_no_cards(iso, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda messages, *, purpose: "I cannot help with that.")
    assert ev.cards_for([_news("Something")]) == []


def test_offline_miss_propagates_from_cards_for(iso, monkeypatch):
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    with pytest.raises(llm.CacheMiss):
        ev.cards_for([_news("Uncached headline")])


def test_prompt_has_no_date_and_no_price_task():
    it = _news("Headline", "<div>Apple ships a product.</div>")
    msgs = ev.news_messages(it)
    blob = json.dumps(msgs)
    assert "2026" not in blob and "07-03" not in blob
    assert "{{UNIVERSE}}" not in msgs[0]["content"] and "CRCL" in msgs[0]["content"]
    assert msgs[1]["content"].startswith("TITLE: Headline") and "Apple ships a product." in msgs[1]["content"]


def test_mixed_items_and_deterministic_via_cards_for(iso, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda messages, *, purpose: '{"cards": []}')
    items = [_rating(action="调高评级", rating_previous="持有", rating_current="买入"),
             _insider("M", 1e6, 23.0), _insider("P", 10_000, 200.0), _news("Irrelevant corn")]
    cards = ev.cards_for(items)
    assert [c["kind"] for c in cards] == ["rating", "insider"]
    assert all(c["available_at"].tzinfo is not None for c in cards)


def test_truncate_keeps_head_and_ticker_paragraphs():
    text = "\n".join(["filler line about nothing"] * 400 + ["Tesla announced a recall of 1M cars."] + ["more filler"] * 50)
    out = ev.truncate(text, max_chars=3000, head=2000)
    assert len(out) <= 3100 and out.startswith("filler") and "Tesla announced a recall" in out


def test_clean_text_strips_html():
    raw = '<p>A &amp; B</p><br><a href="https://x.com/y">link</a><img src="z.png"><li>item</li>'
    assert ev.clean_text(raw) == "A & B\nlink\nitem"


def test_offline_cache_needs_no_env(iso, monkeypatch, tmp_path):
    """Regression: with no BITGET_QWEN_MODEL (a judge without our .env) model_name() raised, so the offline
    replay crashed before looking in the cache. The pinned config.LLM_MODEL is the fallback."""
    from sentiment import agent
    from sentiment.config import LLM_MODEL
    for k in ("BITGET_QWEN_MODEL", "BITGET_QWEN_API_KEY", "BITGET_QWEN_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "ENV_FILE", tmp_path / "nowhere" / ".env")   # no .env
    assert llm.model_name() == LLM_MODEL
    f = llm.CACHE_DIR / f"{llm.cache_key(LLM_MODEL, MSGS)}.json"
    llm._write_atomic(f, {"response": '{"targets": {"NVDA": 0.1}}', "purpose": "t"})
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    assert llm.complete(MSGS, purpose="t") == '{"targets": {"NVDA": 0.1}}'
    with pytest.raises(llm.CacheMiss):                            # not an LLMError about the env
        agent.decide(pd.Timestamp("2026-07-01", tz="UTC"), {}, [], {}, "nonews")


def test_cards_for_raises_on_llm_error_after_saving_the_rest(iso, monkeypatch):
    """Regression: an LLMError on one news item used to be printed and dropped, so a replay finished on
    partial evidence. Now it raises after saving the items that did succeed (retried on the next call)."""
    def flaky(messages, *, purpose):
        if "Broken" in messages[1]["content"]:
            raise llm.LLMError("HTTP 503 simulated")
        return FAKE
    monkeypatch.setattr(llm, "complete", flaky)
    good, bad = _news("Good headline"), _news("Broken headline")
    with pytest.raises(llm.LLMError, match="1 news item"):
        ev.cards_for([good, bad])
    assert set(ev.load_cards()["item_id"]) == {good["id"]}          # the good one was persisted
    errors: list = []                                               # live-loop mode: collect, do not raise
    cards = ev.cards_for([good, bad], errors=errors)
    assert {c["item_id"] for c in cards} == {good["id"]} and len(errors) == 1
    monkeypatch.setattr(llm, "complete", lambda messages, *, purpose: FAKE)    # endpoint back: retried
    assert {c["item_id"] for c in ev.cards_for([good, bad])} == {good["id"], bad["id"]}
