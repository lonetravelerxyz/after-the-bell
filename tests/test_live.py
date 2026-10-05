"""live.py: request signing, contract-size rounding, order planning, shadow-ledger fill arithmetic.

No network: HTTP is replaced by a fake client; the shadow broker must never POST.
"""
import json

import pandas as pd
import pytest

from sentiment import live
from sentiment.config import TAKER

T = pd.Timestamp("2026-09-24 00:00", tz="UTC")
SPEC = {"min_qty": 0.01, "step": "0.01", "volume_place": 2, "price_place": 2, "min_usdt": 5.0,
        "max_market_qty": 60.0, "max_lever": "25", "status": "normal", "taker": "0.0006"}


# ---- signature ---------------------------------------------------------------------------------
def test_sign_known_vectors():
    # expected values computed independently with:
    #   printf '%s' '<prehash>' | openssl dgst -sha256 -hmac 'test-secret' -binary | base64
    assert live.sign("test-secret", "1684814440729", "get", "/api/v2/mix/account/account",
                     "marginCoin=USDT&symbol=BTCUSDT") == "MLqskf1+LxcsIUtLPq+V0vmJ4LlG/rvE4XkJ+o3+CP4="
    assert live.sign("test-secret", "1685013478665", "POST", "/api/v2/mix/order/place-order", "",
                     '{"symbol":"NVDAUSDT","size":"0.5"}') == "IczMzuGeKhuAVkjCYuh/aQZMLh2x0bgdxh9wSe4sZDo="


def test_query_string_sorted_and_drops_none():
    assert live.query_string({"symbol": "BTCUSDT", "marginCoin": "USDT", "x": None}) == "marginCoin=USDT&symbol=BTCUSDT"
    assert live.query_string(None) == ""


def test_client_signs_exactly_what_it_sends(monkeypatch):
    c = live.Client({"api_key": "k", "secret": "s", "passphrase": "p"})
    c._offset_ms = 0
    monkeypatch.setattr(live.time, "time", lambda: 1_700_000_000.0)
    seen = {}

    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"code": "00000", "data": {"ok": 1}}

    def fake(method, url, headers, data, timeout):
        seen.update(method=method, url=url, headers=headers, data=data)
        return R()

    monkeypatch.setattr(c.session, "request", fake)
    assert c.request("POST", "/api/v2/mix/order/place-order", {"b": 2, "a": 1}, {"size": "1"}, auth=True) == {"ok": 1}
    h = seen["headers"]
    assert seen["url"] == "https://api.bitget.com/api/v2/mix/order/place-order?a=1&b=2"
    assert h["paptrading"] == "1" and h["ACCESS-KEY"] == "k" and h["ACCESS-PASSPHRASE"] == "p"
    assert h["ACCESS-TIMESTAMP"] == "1700000000000"
    assert h["ACCESS-SIGN"] == live.sign("s", "1700000000000", "POST", "/api/v2/mix/order/place-order",
                                         "a=1&b=2", seen["data"])
    assert seen["data"] == '{"size":"1"}'
    live_only = {}
    monkeypatch.setattr(c.session, "request", lambda *a, **kw: (live_only.update(kw), R())[1])
    c.request("GET", "/api/v2/mix/market/tickers", {"productType": "USDT-FUTURES"}, demo=False)
    assert "paptrading" not in live_only["headers"] and "ACCESS-SIGN" not in live_only["headers"]


def test_client_raises_on_error_code(monkeypatch):
    c = live.Client()

    class R:
        status_code = 400

        @staticmethod
        def json():
            return {"code": "40009", "msg": "sign signature error"}

    monkeypatch.setattr(c.session, "request", lambda *a, **k: R())
    with pytest.raises(live.BitgetError, match="40009"):
        c.request("GET", "/x")
    with pytest.raises(live.BitgetError, match="not configured"):
        c.request("GET", "/x", auth=True)


# ---- sizing ------------------------------------------------------------------------------------
def test_round_qty_toward_zero_to_step():
    assert live.round_qty(1.23456, SPEC) == 1.23
    assert live.round_qty(-1.239, SPEC) == -1.23
    assert live.round_qty(0.29, SPEC) == 0.29            # float 0.29/0.01 = 28.999..., Decimal keeps it exact
    assert live.round_qty(0.009, SPEC) == 0.0
    idx = {**SPEC, "step": "0.0001", "volume_place": 4}
    assert live.round_qty(0.128749, idx) == 0.1287
    coarse = {**SPEC, "step": "0.5", "volume_place": 1}
    assert live.round_qty(1.99, coarse) == 1.5


def test_legs_split_sign_flip():
    assert live.legs(0, 5) == [("open", 5)]
    assert live.legs(5, 3) == [("close", -2)]
    assert live.legs(5, 0) == [("close", -5)]
    assert live.legs(5, -2) == [("close", -5), ("open", -2)]
    assert live.legs(-1, -4) == [("open", -3)]
    assert live.legs(2, 2) == []


def _book(bid, ask, mark=None, sym="NVDAUSDT"):
    return {sym: {"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "last": ask, "mark": mark or (bid + ask) / 2,
                  "funding": 0.0, "ts": 0}}


def test_plan_orders_minimums_band_flip_and_cap():
    specs = {"NVDAUSDT": SPEC}
    book = _book(199.5, 200.5)
    # 10% of 10k at mid 200 -> 5.00 contracts
    o, s = live.plan_orders({"NVDA": 0.10}, {}, 10_000, book, specs)
    assert [(x["side"], x["qty"], x["kind"], x["hold"]) for x in o] == [("buy", 5.0, "open", "long")]
    # delta below the no-churn band (0.25% of equity = $25 -> 0.125 contracts) is skipped
    o, s = live.plan_orders({"NVDA": 0.101}, {"NVDA": 5.0}, 10_000, book, specs)
    assert o == [] and s[0]["reason"] == "below no-churn band"
    # below contract minimum notional (min_usdt 5) with a tiny equity
    o, s = live.plan_orders({"NVDA": 0.10}, {}, 40, book, specs)
    assert o == [] and s[0]["reason"] in ("below contract minimum", "below no-churn band")
    # flip long 5 -> short 2.5: close 5 (sell), then open 2.5 short (sell)
    o, _ = live.plan_orders({"NVDA": -0.05}, {"NVDA": 5.0}, 10_000, book, specs)
    assert [(x["side"], x["qty"], x["kind"], x["hold"]) for x in o] == [("sell", 5.0, "close", "long"),
                                                                        ("sell", 2.5, "open", "short")]
    # full close always trades, even when tiny
    o, _ = live.plan_orders({}, {"NVDA": 0.02}, 10_000, book, specs)
    assert [(x["side"], x["qty"], x["kind"]) for x in o] == [("sell", 0.02, "close")]
    # 150 contracts with a 60 cap -> 60, 60, 30
    o, _ = live.plan_orders({"NVDA": 3.0}, {}, 10_000, book, specs)
    assert [x["qty"] for x in o] == [60.0, 60.0, 30.0]
    # unknown symbol is reported, not traded
    o, s = live.plan_orders({"ZZZ": 0.1}, {}, 10_000, book, specs)
    assert o == [] and s[0]["reason"] == "no contract or ticker"


# ---- shadow ledger -----------------------------------------------------------------------------
class FakeClient:
    """Answers the public endpoints the shadow broker uses; fails on anything signed or POSTed.
    The live venue quotes bid/ask/mark; the demo book is much wider (195/205), as on 2026-09-23."""

    def __init__(self, bid=199.5, ask=200.5, mark=200.0, funding_rows=(), demo_bid=195.0, demo_ask=205.0):
        self.bid, self.ask, self.mark, self.funding_rows = bid, ask, mark, list(funding_rows)
        self.demo_bid, self.demo_ask = demo_bid, demo_ask
        self.calls = []

    def request(self, method, path, params=None, body=None, auth=False, demo=True):
        self.calls.append((method, path, auth, demo))
        assert method == "GET" and not auth, "shadow broker must not send orders or signed requests"
        if path.endswith("/tickers") and demo:
            return [{"symbol": "NVDAUSDT", "bidPr": str(self.demo_bid), "askPr": str(self.demo_ask),
                     "lastPr": str(self.demo_ask), "markPrice": "200.0", "fundingRate": "0.0001", "ts": "1"}]
        assert not demo or path.endswith("/contracts"), f"shadow broker prices at the live venue: {path}"
        if path.endswith("/contracts"):
            return [{"symbol": "NVDAUSDT", "minTradeNum": "0.01", "sizeMultiplier": "0.01", "volumePlace": "2",
                     "pricePlace": "2", "minTradeUSDT": "5", "maxMarketOrderQty": "60", "maxLever": "25",
                     "symbolStatus": "normal", "takerFeeRate": "0.0006"}]
        if path.endswith("/tickers"):
            return [{"symbol": "NVDAUSDT", "bidPr": str(self.bid), "askPr": str(self.ask), "lastPr": str(self.ask),
                     "markPrice": str(self.mark), "fundingRate": "0.0001", "ts": "1790157872269"}]
        if path.endswith("/history-fund-rate"):
            return self.funding_rows
        raise AssertionError(path)


def _broker(tmp_path, client):
    return live.DemoBroker(dry_run=True, client=client, state_path=tmp_path / "state.json",
                           log_path=tmp_path / "orders.jsonl")


def test_shadow_fill_arithmetic(tmp_path):
    c = FakeClient()
    b = _broker(tmp_path, c)
    assert b.mode == "shadow" and not b.real
    f = b.rebalance(T, {"NVDA": 0.10}, 10_000.0)
    assert len(f) == 1
    f = f[0]
    assert f["side"] == "buy" and f["qty"] == 5.0 and f["price"] == 200.5
    assert f["fee"] == pytest.approx(TAKER * 5 * 200.5)
    assert f["half_spread_cost"] == pytest.approx(5 * 0.5)
    assert f["ts"].tzinfo is not None and f["order_id"].startswith("shadow-")
    assert b.cash == pytest.approx(10_000 - 5 * 200.5 - TAKER * 5 * 200.5)

    c.mark = 201.0
    m = b.mark(T)
    assert m["positions"] == {"NVDA": 5.0}
    assert m["equity"] == pytest.approx(b.cash + 5 * 201.0)
    assert m["gross_equity"] == pytest.approx(m["equity"] + m["fees"] + m["spread"] - m["funding"])
    # gross equity = start + price move from mid (200 -> 201) on 5 contracts
    assert m["gross_equity"] == pytest.approx(10_000 + 5 * 1.0)

    f2 = b.rebalance(T + pd.Timedelta(hours=4), {}, m["equity"])
    assert [(x["side"], x["qty"], x["price"]) for x in f2] == [("sell", 5.0, 199.5)]
    m2 = b.mark(T + pd.Timedelta(hours=4))
    assert m2["positions"] == {}
    # round trip at bid/ask around an unchanged mid: lose the full spread plus two taker fees
    assert m2["equity"] == pytest.approx(10_000 - 5 * 1.0 - TAKER * 5 * (200.5 + 199.5))
    assert m2["fees"] == pytest.approx(TAKER * 5 * (200.5 + 199.5)) and m2["spread"] == pytest.approx(5.0)

    # every intended order is logged with both quotes, and the state survives a restart
    recs = [json.loads(x) for x in (tmp_path / "orders.jsonl").read_text().splitlines()]
    assert [r["status"] for r in recs] == ["filled", "filled"]
    assert recs[0]["live_quote"]["ask"] == 200.5 and recs[0]["demo_quote"]["ask"] == 205.0
    b2 = _broker(tmp_path, c)
    assert b2.cash == pytest.approx(b.cash) and b2.fees == pytest.approx(b.fees) and b2.positions == {}


def test_shadow_funding_accrual(tmp_path):
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    settle = now_ms - 3_600_000
    c = FakeClient(mark=200.0, funding_rows=[{"symbol": "NVDAUSDT", "fundingRate": "0.0002", "fundingTime": str(settle)},
                                             {"symbol": "NVDAUSDT", "fundingRate": "0.0005",
                                              "fundingTime": str(settle - 8 * 3_600_000)}])
    b = _broker(tmp_path, c)
    b.positions = {"NVDA": 5.0}
    b.funding_cursor = {"NVDA": settle - 60_000}         # opened after the older settlement
    cash0 = b.cash
    m = b.mark(T)
    # a long pays a positive rate: only the settlement after the cursor counts
    assert m["funding"] == pytest.approx(-5 * 200.0 * 0.0002)
    assert b.cash == pytest.approx(cash0 - 5 * 200.0 * 0.0002)
    assert b.funding_cursor["NVDA"] == settle
    b.mark(T)                                            # no double counting
    assert b.funding == pytest.approx(-5 * 200.0 * 0.0002)


def test_order_body_one_way_and_hedge(tmp_path):
    b = _broker(tmp_path, FakeClient())
    b._specs = {"NVDAUSDT": SPEC}
    close_long = {"symbol": "NVDAUSDT", "ticker": "NVDA", "side": "sell", "qty": 5.0, "kind": "close", "hold": "long"}
    open_short = {**close_long, "qty": 2.5, "kind": "open", "hold": "short"}
    b._pos_mode = "one_way_mode"
    body = b.order_body(close_long, "x1")
    assert body == {"symbol": "NVDAUSDT", "productType": "USDT-FUTURES", "marginMode": "crossed",
                    "marginCoin": "USDT", "size": "5.00", "orderType": "market", "clientOid": "x1",
                    "side": "sell", "reduceOnly": "YES"}
    assert "tradeSide" not in b.order_body(open_short, "x2")
    b._pos_mode = "hedge_mode"
    body = b.order_body(close_long, "x3")                # v2 hedge: side names the position
    assert body["side"] == "buy" and body["tradeSide"] == "close" and "reduceOnly" not in body
    body = b.order_body(open_short, "x4")
    assert body["side"] == "sell" and body["tradeSide"] == "open" and body["size"] == "2.50"


# ---- regressions -------------------------------------------------------------------------------
def test_shadow_fills_at_live_quotes_not_demo(tmp_path):
    """Regression: shadow fills used the demo book (10-62 bp wide) while the replay charges the live
    venue's quoted half-spread; shadow now fills and marks at live-venue quotes."""
    b = _broker(tmp_path, FakeClient(demo_bid=190.0, demo_ask=210.0))
    f = b.rebalance(T, {"NVDA": 0.10}, 10_000.0)[0]
    assert f["price"] == 200.5 and f["half_spread_cost"] == pytest.approx(5 * 0.5)


def test_mark_never_values_a_held_name_at_zero(tmp_path):
    """Regression: a held name whose ticker row had null bid/ask was dropped and valued at 0."""
    class NoBook(FakeClient):
        def request(self, method, path, params=None, body=None, auth=False, demo=True):
            if path.endswith("/tickers"):
                return [{"symbol": "NVDAUSDT", "bidPr": None, "askPr": None, "lastPr": "181", "markPrice": "181",
                         "fundingRate": "0", "ts": "1"}]
            return super().request(method, path, params, body, auth, demo)
    b = _broker(tmp_path, NoBook())
    b.positions, b.cash = {"NVDA": 8.3}, 8506.0
    m = b.mark(T)
    assert m["equity"] == pytest.approx(8506.0 + 8.3 * 181) and m["prices"] == {"NVDA": 181.0}
    o, s = live.plan_orders({"NVDA": 0.0}, {"NVDA": 8.3}, 10_000, live.tickers(NoBook()), {"NVDAUSDT": SPEC})
    assert o == [] and s[0]["reason"] == "no top of book"

    class Gone(FakeClient):                          # the symbol vanishes from the tickers entirely
        def request(self, method, path, params=None, body=None, auth=False, demo=True):
            return [] if path.endswith("/tickers") else super().request(method, path, params, body, auth, demo)
    b2 = live.DemoBroker(dry_run=True, client=Gone(), state_path=tmp_path / "state.json", log_path=tmp_path / "o.jsonl")
    assert b2.mark(T)["equity"] == pytest.approx(8506.0 + 8.3 * 181)   # last known price, persisted
    b3 = live.DemoBroker(dry_run=True, client=Gone(), state_path=tmp_path / "s3.json", log_path=tmp_path / "o3.jsonl")
    b3.positions = {"NVDA": 1.0}
    assert b3.mark(T)["equity"] != b3.mark(T)["equity"]                   # NaN, never cash-only


class RealClient:
    """Signed demo endpoints for DemoBroker in real mode."""

    def __init__(self, fills=(), detail=None, post_timeout=False, lookup=None):
        self.fills, self.detail, self.post_timeout, self.lookup = list(fills), detail, post_timeout, lookup
        self.posts, self.calls = [], []

    def request(self, method, path, params=None, body=None, auth=False, demo=True):
        self.calls.append((method, path, dict(params or {})))
        if path.endswith("/contracts"):
            return FakeClient().request("GET", path, demo=True)
        if path.endswith("/tickers"):
            return FakeClient().request("GET", path, demo=demo)
        if path.endswith("/accounts"):
            return [{"marginCoin": "USDT", "posMode": "one_way_mode", "usdtEquity": "10000"}]
        if path.endswith("/all-position"):
            return []
        if path.endswith("/place-order"):
            self.posts.append(body)
            if self.post_timeout:
                raise live.BitgetTimeout("POST /place-order: ReadTimeout")
            return {"orderId": "o1", "clientOid": body["clientOid"]}
        if path.endswith("/order/fills"):
            return {"fillList": self.fills}
        if path.endswith("/order/detail"):
            if params.get("clientOid"):
                return self.lookup
            return self.detail
        raise AssertionError(path)


def _real(tmp_path, client, monkeypatch):
    monkeypatch.setattr(live, "load_keys", lambda: {"api_key": "k", "secret": "s", "passphrase": "p"})
    monkeypatch.setattr(live.time, "sleep", lambda s: None)
    return live.DemoBroker(dry_run=False, client=client, state_path=tmp_path / "d.json", log_path=tmp_path / "o.jsonl")


def test_real_fill_from_order_detail_or_pending(tmp_path, monkeypatch):
    """Regression: with /order/fills still empty, _fill_real returned a qty-0 / NaN-price 'filled' fill
    (which crashed metrics.round_trips). It now reads /order/detail, and logs 'pending' if nothing executed."""
    c = RealClient(detail={"orderId": "o1", "baseVolume": "5", "priceAvg": "200.6", "fee": "-0.6", "uTime": "1790000000000",
                           "state": "filled"})
    b = _real(tmp_path, c, monkeypatch)
    f = b.rebalance(T, {"NVDA": 0.10}, 10_000.0)
    assert len(f) == 1 and f[0]["qty"] == 5.0 and f[0]["price"] == 200.6 and f[0]["fee"] == pytest.approx(0.6)
    c2 = RealClient(detail={"orderId": "o1", "baseVolume": "0", "priceAvg": "", "state": "live"})
    b2 = _real(tmp_path / "x", c2, monkeypatch)
    assert b2.rebalance(T, {"NVDA": 0.10}, 10_000.0) == []
    recs = [json.loads(x) for x in (tmp_path / "x" / "o.jsonl").read_text().splitlines()]
    assert [r["status"] for r in recs] == ["pending"] and "o1" in recs[0]["error"]


def test_post_timeout_is_not_resent(tmp_path, monkeypatch):
    """Regression: a place-order POST that timed out was re-sent with the same clientOid; the duplicate was
    rejected and an executed order logged as an error. Now: one POST, then a lookup by clientOid."""
    sess_calls = []

    class Sess:
        def request(self, method, url, headers, data, timeout):
            sess_calls.append(method)
            raise live.requests.ReadTimeout("read timed out")
    cl = live.Client({"api_key": "k", "secret": "s", "passphrase": "p"})
    cl._offset_ms, cl.session = 0, Sess()
    with pytest.raises(live.BitgetTimeout):
        cl.request("POST", "/api/v2/mix/order/place-order", body={"a": 1}, auth=True)
    assert sess_calls == ["POST"]
    c = RealClient(post_timeout=True, lookup={"orderId": "o9", "clientOid": "x"},
                   fills=[{"baseVolume": "5", "price": "200.4", "cTime": "1790000000000",
                           "feeDetail": [{"totalFee": "-0.6"}]}])
    b = _real(tmp_path, c, monkeypatch)
    f = b.rebalance(T, {"NVDA": 0.10}, 10_000.0)
    assert len(c.posts) == 1 and f[0]["order_id"] == "o9" and f[0]["qty"] == 5.0
    c2 = RealClient(post_timeout=True, lookup=None)                    # never reached the server
    b2 = _real(tmp_path / "y", c2, monkeypatch)
    assert b2.rebalance(T, {"NVDA": 0.10}, 10_000.0) == [] and len(c2.posts) == 1


def test_funding_paid_pages_and_filters(tmp_path, monkeypatch):
    """Regression: one page of 100 bills per 30-day window, no businessType filter: trade bills filled the
    page and funding was under-counted."""
    bills = [{"billId": str(i), "businessType": "contract_settle_fee", "amount": "-1"} for i in range(250)]

    class Bills(RealClient):
        def request(self, method, path, params=None, body=None, auth=False, demo=True):
            if path.endswith("/account/bill"):
                self.calls.append(dict(params))
                assert params["businessType"] == "contract_settle_fee"
                start = int(params["idLessThan"]) if params.get("idLessThan") else len(bills)
                page = bills[max(0, start - params["limit"]):start][::-1]
                return {"bills": page, "endId": page[-1]["billId"] if page else None}
            return super().request(method, path, params, body, auth, demo)
    c = Bills()
    b = _real(tmp_path, c, monkeypatch)
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    assert b.funding_paid(now_ms - 86_400_000) == pytest.approx(-250.0)
    assert len(c.calls) == 3 and c.calls[1]["idLessThan"] == "150"


def test_live_step_writes_hash_chained_live_records(tmp_path, monkeypatch):
    """The live loop drives DemoBroker through the replay's decision step and logs mode='live'."""
    from sentiment import ledger
    from tests.test_replay import FakeStore
    store = FakeStore(days=90)
    b = _broker(tmp_path, FakeClient())
    calls = []

    def pol(t, state, cards, book, variant):
        calls.append(t)
        return {"targets": {"NVDA": 0.05}, "reasons": {"NVDA": "r"}, "evidence": {}, "confidence": 0.5,
                "raw": "{}", "valid": True, "shown": [], "llm_request_hash": "h"}
    t = pd.Timestamp("2026-09-24 00:01", tz="UTC")
    r = live.step(t, broker=b, store=store, refresh=False, decide_fn=pol, cards_fn=lambda items, errors=None: [],
                  out_dir=tmp_path / "live")
    assert r["mode"] == "live" and r["event"] == "decision" and r["ts"].startswith("2026-09-24 00:00")
    assert r["fills"] and r["run_id"] == "live_llm_shadow" and ledger.verify(tmp_path / "live" / "log.jsonl")
    assert live.step(t, broker=b, store=store, refresh=False, decide_fn=pol, out_dir=tmp_path / "live") is None
    assert live.step(t + pd.Timedelta(hours=1), broker=b, store=store, refresh=False, decide_fn=pol,
                     out_dir=tmp_path / "live") is None                  # no stop: nothing to log
    assert len(calls) == 1 and len(pd.read_csv(tmp_path / "live" / "equity.csv")) == 2
    st = json.loads((tmp_path / "live" / "runner_state.json").read_text())
    assert st["risk"]["entry"]["NVDA"]["side"] == 1
