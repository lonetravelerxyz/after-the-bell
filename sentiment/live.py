"""Live execution on Bitget Demo Trading (CONTRACT: Execution, `live.DemoBroker`).

Bitget v2 mix API on https://api.bitget.com with header `paptrading: 1` (demo), productType
USDT-FUTURES, marginCoin USDT, crossed margin, market orders. Signing (Bitget "signature" doc):
  ACCESS-SIGN = base64(HMAC-SHA256(secret, timestamp + METHOD + requestPath + ("?" + query) + body))
with the query string sorted by key and sent exactly as signed.

Two modes, same interface as `sim.SimBroker`:
  real   - keys BITGET_DEMO_API_KEY / _SECRET / _PASSPHRASE in .env and dry_run=False: orders go to
           the demo account; `mark` reads positions and equity from the account endpoints. A fill
           is read from /order/fills, else from /order/detail; an order with no executed quantity
           yet is logged as `pending` (never as a qty-0 fill). Orders are POSTed once: after a
           timeout the order is looked up by clientOid instead of being re-sent.
  shadow - keys absent or dry_run=True (spec §8 fallback, "shadow ledger at live quotes"): intended
           orders are filled at the LIVE-venue top of book (buy at ask, sell at bid) plus TAKER
           fee, and the book is marked and charged funding at live-venue prices. Nothing is sent.
Cost regime: the Bitget demo book is much wider than the live venue (2026-09-23 --check: demo spreads
10-62 bp vs live 0.3-2.5 bp, demo mid up to 36 bp off live). Shadow fills therefore use live quotes,
the same regime as the replay's static quoted half-spread, so replay and shadow compare like with
like. Real demo fills pay the demo spread; every order log line carries both the demo and the live
quote so the reconciliation can report slippage against either mid and the demo-vs-live gap.
Both modes append every intended order to `out/live/orders.jsonl` and persist the
book (and the last known price of each held name) to `out/live/{mode}_state.json`, so the
live loop can restart. A held name without a quote is marked at its last known price, never at 0;
with no price at all, equity is NaN and the runner skips the risk update for that hour.

Sizing: target weight x equity / mid -> contract quantity; legs are rounded toward zero to the
contract step (`sizeMultiplier`, `volumePlace`), capped at `maxMarketOrderQty` per order, and
skipped below `minTradeNum` / `minTradeUSDT` or inside the replay's no-churn band
(`sim.below_band`; full closes and flips always trade). A sign flip is two legs: close, then open.

Live loop (spec §2, mode "live"): every hour, top up bars/funding/news (and ratings/insider/F&G at
decision hours) into data/raw, rebuild the point-in-time Store over data/raw, mark, run the hourly
risk check, and at DECISION_HOURS run the same `replay.decision_step` as the replay (same prompt,
cache, risk layer and record format) on the demo broker; records go to the hash-chained
out/live/log.jsonl and the pre-trade hourly marks to out/live/equity.csv.

  uv run python -m sentiment.live --check            public endpoints: demo contracts, demo vs live quotes
  uv run python -m sentiment.live --once [--shadow]  one step for the current hour (cron friendly)
  uv run python -m sentiment.live --run [--shadow]   loop, one step per hour
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import sys
import time
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import pandas as pd
import requests

from sentiment.config import ENV_FILE, INDEX_PERPS, OUT, START_EQUITY, TAKER, UNIVERSE, perp
from sentiment.sim import below_band

BASE = "https://api.bitget.com"
PRODUCT = "USDT-FUTURES"
MARGIN_COIN = "USDT"
MARGIN_MODE = "crossed"
LIVE_DIR = OUT / "live"
UA = "bitget-ai-s2-sentiment/0.1"


class BitgetError(RuntimeError):
    pass


class BitgetTimeout(BitgetError):
    """A non-GET request timed out or lost its connection: the server may or may not have acted on it."""


class BitgetPending(BitgetError):
    """An order was accepted but no executed quantity could be read back yet."""


# ---- signing and transport ---------------------------------------------------------------------
def query_string(params: dict | None) -> str:
    """Bitget query string: keys sorted ascending, values as-is, no URL encoding."""
    if not params:
        return ""
    return "&".join(f"{k}={params[k]}" for k in sorted(params) if params[k] is not None)


def sign(secret: str, timestamp: str, method: str, path: str, query: str = "", body: str = "") -> str:
    prehash = str(timestamp) + method.upper() + path + (f"?{query}" if query else "") + (body or "")
    return base64.b64encode(hmac.new(secret.encode(), prehash.encode(), hashlib.sha256).digest()).decode()


def load_keys() -> dict[str, str] | None:
    """Demo API keys from the environment or .env; None when any of the three is missing."""
    env = ENV_FILE
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    keys = {k: os.environ.get(f"BITGET_DEMO_{k.upper()}", "") for k in ("api_key", "secret", "passphrase")}
    return keys if all(keys.values()) else None


class Client:
    """Thin v2 REST client. `demo=True` adds `paptrading: 1`; `auth=True` signs the request."""

    def __init__(self, keys: dict[str, str] | None = None, timeout: float = 15.0):
        self.keys, self.timeout = keys, timeout
        self.session = requests.Session()
        self._offset_ms: int | None = None

    def _now_ms(self) -> int:
        if self._offset_ms is None:                    # server clock skew; the sign window is 30s
            try:
                srv = int(self.request("GET", "/api/v2/public/time", demo=False)["serverTime"])
                self._offset_ms = srv - int(time.time() * 1000)
            except Exception:  # noqa: BLE001 - fall back to the local clock
                self._offset_ms = 0
        return int(time.time() * 1000) + self._offset_ms

    def request(self, method: str, path: str, params: dict | None = None, body: dict | None = None,
                auth: bool = False, demo: bool = True):
        q = query_string(params)
        data = json.dumps(body, separators=(",", ":")) if body is not None else ""
        h = {"Content-Type": "application/json", "locale": "en-US", "User-Agent": UA}
        if demo:
            h["paptrading"] = "1"
        if auth:
            if not self.keys:
                raise BitgetError("demo API keys are not configured (BITGET_DEMO_* in .env)")
            ts = str(self._now_ms())
            h.update({"ACCESS-KEY": self.keys["api_key"], "ACCESS-PASSPHRASE": self.keys["passphrase"],
                      "ACCESS-TIMESTAMP": ts, "ACCESS-SIGN": sign(self.keys["secret"], ts, method, path, q, data)})
        url = BASE + path + (f"?{q}" if q else "")
        for attempt in range(4):
            try:
                r = self.session.request(method, url, headers=h, data=data or None, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                if method != "GET":                    # never re-send an order: it may have been accepted
                    raise BitgetTimeout(f"{method} {path}: {type(e).__name__}: {e}") from e
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 500 and attempt < 3 and method == "GET":
                time.sleep(2 ** attempt)
                continue
            try:
                j = r.json()
            except ValueError:
                raise BitgetError(f"{method} {path}: HTTP {r.status_code} {r.text[:200]}") from None
            if str(j.get("code")) != "00000":
                raise BitgetError(f"{method} {path}: {j.get('code')} {j.get('msg')}")
            return j.get("data")
        raise BitgetError(f"{method} {path}: retries exhausted")


# ---- market data -------------------------------------------------------------------------------
def contracts(client: Client, demo: bool = True) -> dict[str, dict]:
    """Contract precision per perp symbol: min_qty, step, volume_place, min_usdt, max_market_qty, ..."""
    out = {}
    for c in client.request("GET", "/api/v2/mix/market/contracts", {"productType": PRODUCT}, demo=demo):
        out[c["symbol"]] = {
            "min_qty": float(c["minTradeNum"]), "step": c["sizeMultiplier"], "volume_place": int(c["volumePlace"]),
            "price_place": int(c["pricePlace"]), "min_usdt": float(c.get("minTradeUSDT") or 0),
            "max_market_qty": float(c.get("maxMarketOrderQty") or 0) or None,
            "max_lever": c.get("maxLever"), "status": c.get("symbolStatus"), "taker": c.get("takerFeeRate")}
    return out


def _fnum(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def tickers(client: Client, demo: bool = True) -> dict[str, dict]:
    """Top of book per perp symbol: bid, ask, last, mark, mid, funding (current), ts. A row with an empty
    book keeps bid/ask None and mid = mark (or last); a row with no price at all is dropped."""
    out = {}
    for x in client.request("GET", "/api/v2/mix/market/tickers", {"productType": PRODUCT}, demo=demo):
        bid, ask, last, mark = (_fnum(x.get(k)) for k in ("bidPr", "askPr", "lastPr", "markPrice"))
        mark = mark or last
        mid = (bid + ask) / 2 if bid and ask else mark
        if mid is None:
            continue
        out[x["symbol"]] = {"bid": bid if ask else None, "ask": ask if bid else None, "mid": mid,
                            "last": last or mid, "mark": mark or mid,
                            "funding": float(x.get("fundingRate") or 0), "ts": int(x.get("ts") or 0)}
    return out


def round_qty(qty: float, spec: dict) -> float:
    """Round |qty| toward zero to the contract step and volume precision; sign is kept."""
    step = Decimal(str(spec["step"]))
    q = (Decimal(str(abs(qty))) / step).to_integral_value(rounding=ROUND_DOWN) * step
    q = q.quantize(Decimal(1).scaleb(-int(spec["volume_place"])), rounding=ROUND_DOWN)
    return float(q) if qty >= 0 else -float(q)


def legs(cur: float, tgt: float) -> list[tuple[str, float]]:
    """Signed quantity legs from cur to tgt: ('close', dq) then ('open', dq); a flip gives both."""
    if cur != 0 and (tgt == 0 or (cur > 0) != (tgt > 0)):
        out = [("close", -cur)]
        return out + ([("open", tgt)] if tgt != 0 else [])
    d = tgt - cur
    if d == 0:
        return []
    return [("open" if abs(tgt) > abs(cur) else "close", d)]


def plan_orders(targets: dict[str, float], positions: dict[str, float], equity: float,
                book: dict[str, dict], specs: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Orders to move `positions` (qty) to `targets` (weights). Returns (orders, skipped)."""
    orders, skipped = [], []
    for tk in sorted(set(targets) | {k for k, q in positions.items() if q}):
        sym = perp(tk)
        spec, quote = specs.get(sym), book.get(sym)
        if spec is None or quote is None or quote["mid"] <= 0:
            skipped.append({"ticker": tk, "reason": "no contract or ticker"})
            continue
        if not quote.get("bid") or not quote.get("ask"):
            skipped.append({"ticker": tk, "reason": "no top of book"})
            continue
        mid, cur = quote["mid"], float(positions.get(tk, 0.0))
        tgt_w = float(targets.get(tk, 0.0))
        tgt = round_qty(tgt_w * equity / mid, spec)
        full_close = tgt == 0 and cur != 0
        flip = cur != 0 and tgt != 0 and (cur > 0) != (tgt > 0)
        if not full_close and not flip and below_band((tgt - cur) * mid / equity, tgt_w):
            if tgt != cur:
                skipped.append({"ticker": tk, "reason": "below no-churn band", "delta": tgt - cur})
            continue
        for kind, dq in legs(cur, tgt):
            dq = round_qty(dq, spec)
            closing_all = kind == "close" and abs(dq) >= abs(cur) - 1e-12
            if abs(dq) < spec["min_qty"] or (not closing_all and abs(dq) * mid < spec["min_usdt"]):
                skipped.append({"ticker": tk, "reason": "below contract minimum", "delta": dq})
                continue
            cap = spec.get("max_market_qty") or abs(dq)
            left = abs(dq)
            while left > 1e-12:                        # split above the market-order cap
                q = round_qty(min(left, cap), spec)
                if q <= 0:
                    break
                orders.append({"ticker": tk, "symbol": sym, "side": "buy" if dq > 0 else "sell", "qty": q,
                               "kind": kind, "hold": "long" if (dq > 0) == (kind == "open") else "short",
                               "mid": mid, "bid": quote["bid"], "ask": quote["ask"]})
                left = round(left - q, 12)
    return orders, skipped


# ---- broker ------------------------------------------------------------------------------------
def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class DemoBroker:
    """Broker on Bitget Demo; shadow ledger at live quotes when dry_run or keys are absent (see module doc)."""

    def __init__(self, dry_run: bool = True, client: Client | None = None, state_path: Path | None = None,
                 log_path: Path | None = None, start_equity: float = START_EQUITY):
        keys = load_keys()
        self.real = not dry_run and keys is not None
        self.mode = "demo" if self.real else "shadow"
        self.client = client or Client(keys if self.real else None)
        self.state_path = Path(state_path or LIVE_DIR / f"{self.mode}_state.json")
        self.log_path = Path(log_path or LIVE_DIR / "orders.jsonl")
        self._specs: dict[str, dict] | None = None
        self._pos_mode: str | None = None
        s = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.cash = float(s.get("cash", start_equity))
        self.positions: dict[str, float] = {k: float(v) for k, v in s.get("positions", {}).items()}
        self.fees, self.spread, self.funding = (float(s.get(k, 0.0)) for k in ("fees", "spread", "funding"))
        self.funding_cursor: dict[str, int] = {k: int(v) for k, v in s.get("funding_cursor", {}).items()}
        self.last_price: dict[str, float] = {k: float(v) for k, v in s.get("last_price", {}).items()}
        self.started_ms = int(s.get("started_ms") or _now().timestamp() * 1000)
        self.skipped: list[dict] = []

    # -- helpers
    @property
    def specs(self) -> dict[str, dict]:
        if self._specs is None:
            self._specs = contracts(self.client, demo=True)
        return self._specs

    @property
    def venue_demo(self) -> bool:
        """Quotes used for sizing, shadow fills and shadow marks: the demo book in real mode, live otherwise."""
        return self.real

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "mode": self.mode, "cash": self.cash, "positions": self.positions, "fees": self.fees,
            "spread": self.spread, "funding": self.funding, "funding_cursor": self.funding_cursor,
            "last_price": self.last_price, "started_ms": self.started_ms, "saved_at": _now().isoformat()}, indent=1))

    def _log(self, rec: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def _prices(self, positions: dict[str, float], book: dict[str, dict], marks: dict[str, float] | None = None
                ) -> dict[str, float]:
        """Mark per held ticker: account mark, else the ticker mark, else the last known price."""
        out = {}
        for tk in positions:
            px = (marks or {}).get(tk) or (book.get(perp(tk)) or {}).get("mark") or self.last_price.get(tk)
            if px:
                out[tk] = float(px)
                self.last_price[tk] = float(px)
        return out

    def pos_mode(self) -> str:
        """'one_way_mode' or 'hedge_mode' from the demo account (real mode only)."""
        if self._pos_mode is None:
            acc = [a for a in self.client.request("GET", "/api/v2/mix/account/accounts", {"productType": PRODUCT},
                                                  auth=True) if a.get("marginCoin") == MARGIN_COIN]
            self._pos_mode = acc[0].get("posMode", "one_way_mode") if acc else "one_way_mode"
        return self._pos_mode

    def positions_real(self) -> tuple[dict[str, float], dict[str, float]]:
        """Net contract qty per ticker (long - short) and mark price per ticker, from the demo account."""
        rows = self.client.request("GET", "/api/v2/mix/position/all-position",
                                   {"productType": PRODUCT, "marginCoin": MARGIN_COIN}, auth=True) or []
        pos, marks = {}, {}
        tk_of = {perp(t): t for t in UNIVERSE}
        for r in rows:
            tk = tk_of.get(r["symbol"], r["symbol"].removesuffix("USDT"))
            q = float(r.get("total") or 0) * (-1 if r.get("holdSide") == "short" else 1)
            pos[tk] = pos.get(tk, 0.0) + q
            if r.get("markPrice"):
                marks[tk] = float(r["markPrice"])
        return {k: v for k, v in pos.items() if v}, marks

    def order_body(self, o: dict, client_oid: str) -> dict:
        b = {"symbol": o["symbol"], "productType": PRODUCT, "marginMode": MARGIN_MODE, "marginCoin": MARGIN_COIN,
             "size": f"{o['qty']:.{self.specs[o['symbol']]['volume_place']}f}", "orderType": "market",
             "clientOid": client_oid}
        if self.pos_mode() == "hedge_mode":
            # v2 hedge mode: side names the position (buy = long, sell = short), tradeSide open/close
            b["side"] = "buy" if o["hold"] == "long" else "sell"
            b["tradeSide"] = o["kind"]
        else:
            b["side"] = o["side"]
            if o["kind"] == "close":
                b["reduceOnly"] = "YES"
        return b

    def _fills_of(self, symbol: str, order_id: str, tries: int = 6) -> list[dict]:
        for i in range(tries):
            d = self.client.request("GET", "/api/v2/mix/order/fills",
                                    {"productType": PRODUCT, "symbol": symbol, "orderId": order_id}, auth=True)
            rows = (d or {}).get("fillList") or []
            if rows:
                return rows
            time.sleep(0.5 * (i + 1))
        return []

    def order_detail(self, symbol: str, order_id: str | None = None, client_oid: str | None = None) -> dict | None:
        """/api/v2/mix/order/detail by orderId or clientOid; None when the order does not exist."""
        try:
            d = self.client.request("GET", "/api/v2/mix/order/detail",
                                    {"productType": PRODUCT, "symbol": symbol, "orderId": order_id,
                                     "clientOid": client_oid}, auth=True)
        except BitgetError:
            return None
        d = d[0] if isinstance(d, list) and d else d
        return d if isinstance(d, dict) and (d.get("orderId") or d.get("clientOid")) else None

    # -- Broker interface
    def rebalance(self, t, targets: dict[str, float], equity: float, only=None) -> list[dict]:
        t = pd.Timestamp(t)
        if self.real:
            positions, _ = self.positions_real()
        else:
            self._accrue_funding(_now())
            positions = dict(self.positions)
        demo_q, live_q = tickers(self.client, demo=True), tickers(self.client, demo=False)
        book = demo_q if self.venue_demo else live_q
        if only is not None:
            targets = {k: v for k, v in targets.items() if k in only}
            positions = {k: v for k, v in positions.items() if k in only}
        orders, skipped = plan_orders(targets, positions, equity, book, self.specs)
        self.skipped = skipped
        fills = []
        for i, o in enumerate(orders):
            oid = f"sb{t:%Y%m%d%H%M}{o['ticker'][:5]}{i}"
            rec = {"t": t, "mode": self.mode, "client_oid": oid, **o,
                   "demo_quote": _quote(demo_q.get(o["symbol"])), "live_quote": _quote(live_q.get(o["symbol"]))}
            try:
                f = self._fill_real(o, oid) if self.real else self._fill_shadow(o, oid)
            except BitgetPending as e:
                self._log({**rec, "status": "pending", "error": str(e)})
                continue
            except BitgetError as e:
                self._log({**rec, "status": "error", "error": str(e)})
                continue
            self._log({**rec, "status": "filled", "fill": f})
            fills.append(f)
        for s in skipped:
            self._log({"t": t, "mode": self.mode, "status": "skipped", **s})
        self.positions = {k: q for k, q in self.positions.items() if abs(q) > 1e-12}
        self._save()
        return fills

    def _fill_shadow(self, o: dict, oid: str) -> dict:
        sgn = 1 if o["side"] == "buy" else -1
        price = o["ask"] if sgn > 0 else o["bid"]
        fee = TAKER * o["qty"] * price
        spread_cost = o["qty"] * abs(price - o["mid"])
        cur = self.positions.get(o["ticker"], 0.0)
        if cur == 0:
            self.funding_cursor[o["ticker"]] = int(_now().timestamp() * 1000)
        self.positions[o["ticker"]] = round(cur + sgn * o["qty"], 10)
        self.last_price[o["ticker"]] = o["mid"]
        self.cash -= sgn * o["qty"] * price + fee
        self.fees += fee
        self.spread += spread_cost
        return {"ticker": o["ticker"], "side": o["side"], "qty": o["qty"], "price": price, "fee": fee,
                "half_spread_cost": spread_cost, "ts": _now(), "order_id": f"shadow-{oid}"}

    def _fill_real(self, o: dict, oid: str) -> dict:
        try:
            r = self.client.request("POST", "/api/v2/mix/order/place-order", body=self.order_body(o, oid), auth=True)
            order_id = r["orderId"]
        except BitgetTimeout as e:                     # accepted or not? ask by clientOid, never re-send
            d = self.order_detail(o["symbol"], client_oid=oid)
            if d is None:
                raise BitgetError(f"place-order timed out and no order {oid} exists: {e}") from e
            order_id = d["orderId"]
        rows = self._fills_of(o["symbol"], order_id)
        if rows:
            qty = sum(float(x["baseVolume"]) for x in rows)
            price = sum(float(x["price"]) * float(x["baseVolume"]) for x in rows) / qty if qty else float("nan")
            fee = -sum(float(fd.get("totalFee") or 0) for x in rows for fd in (x.get("feeDetail") or []))
            ts = pd.Timestamp(int(max(int(x["cTime"]) for x in rows)), unit="ms", tz="UTC")
        else:                                          # fills endpoint lagging: the order detail has the totals
            d = self.order_detail(o["symbol"], order_id=order_id) or {}
            qty, price = float(d.get("baseVolume") or 0), float(d.get("priceAvg") or "nan")
            fee = -float(d.get("fee") or 0)
            ts = pd.Timestamp(int(d.get("uTime") or d.get("cTime") or _now().timestamp() * 1000), unit="ms", tz="UTC")
        if not qty > 0 or not math.isfinite(price):
            raise BitgetPending(f"order {order_id} ({oid}) accepted, no executed quantity yet")
        spread_cost = qty * abs(price - o["mid"])
        self.fees += fee
        self.spread += spread_cost
        return {"ticker": o["ticker"], "side": o["side"], "qty": qty, "price": price, "fee": fee,
                "half_spread_cost": spread_cost, "ts": ts, "order_id": order_id}

    def _accrue_funding(self, now: pd.Timestamp) -> None:
        """Shadow only: book live-venue funding settlements since each position's cursor, at the current mark."""
        held = {k: q for k, q in self.positions.items() if q}
        if not held:
            return
        now_ms = int(now.timestamp() * 1000)
        marks = None
        for tk, qty in held.items():
            cur = self.funding_cursor.get(tk, now_ms)
            if now_ms - cur < 60_000:
                continue
            rows = self.client.request("GET", "/api/v2/mix/market/history-fund-rate",
                                       {"productType": PRODUCT, "symbol": perp(tk), "pageSize": 20},
                                       demo=self.venue_demo) or []
            due = [(int(r["fundingTime"]), float(r["fundingRate"])) for r in rows
                   if cur < int(r["fundingTime"]) <= now_ms]
            if not due:
                continue
            marks = marks or tickers(self.client, demo=self.venue_demo)
            px = (marks.get(perp(tk)) or {}).get("mark") or self.last_price.get(tk)
            if px is None:
                continue
            for _, rate in due:
                pnl = -qty * px * rate
                self.cash += pnl
                self.funding += pnl
            self.funding_cursor[tk] = max(ts for ts, _ in due)

    def mark(self, t) -> dict:
        t = pd.Timestamp(t)
        if self.real:
            acc = [a for a in self.client.request("GET", "/api/v2/mix/account/accounts", {"productType": PRODUCT},
                                                  auth=True) if a.get("marginCoin") == MARGIN_COIN]
            equity = float(acc[0].get("usdtEquity") or acc[0]["accountEquity"]) if acc else float("nan")
            positions, marks = self.positions_real()
            self.positions = positions
            prices = self._prices(positions, {}, marks)
            self.funding = self.funding_paid(self.started_ms)
            cash = equity - sum(q * prices.get(tk, 0.0) for tk, q in positions.items())
        else:
            self._accrue_funding(_now())
            positions = dict(self.positions)
            prices = self._prices(positions, tickers(self.client, demo=self.venue_demo))
            cash = self.cash
            missing = [tk for tk in positions if tk not in prices]
            equity = float("nan") if missing else self.cash + sum(q * prices[tk] for tk, q in positions.items())
        self._save()
        return {"ts": t, "equity": equity, "positions": positions, "cash": cash, "fees": self.fees,
                "spread": self.spread, "funding": self.funding,
                "gross_equity": equity + self.fees + self.spread - self.funding, "prices": prices,
                "mode": self.mode}

    # -- reconciliation (real mode; needs keys)
    def funding_paid(self, start_ms: int | None = None) -> float:
        """Signed funding P&L from the demo account bills (businessType contract_settle_fee) since start_ms
        (default 89 days): 30-day windows, each paged back with idLessThan = endId until a short page."""
        end = int(_now().timestamp() * 1000)
        start = start_ms or end - 89 * 86_400_000
        total, s = 0.0, start
        while s < end:
            e = min(s + 30 * 86_400_000, end)
            last = None
            while True:
                d = self.client.request("GET", "/api/v2/mix/account/bill",
                                        {"productType": PRODUCT, "coin": MARGIN_COIN, "businessType": "contract_settle_fee",
                                         "startTime": s, "endTime": e, "limit": 100, "idLessThan": last},
                                        auth=True) or {}
                page = d.get("bills") or []
                total += sum(float(b.get("amount") or 0) for b in page
                             if "settle_fee" in str(b.get("businessType", "contract_settle_fee")))
                if len(page) < 100 or not d.get("endId") or d.get("endId") == last:
                    break
                last = d["endId"]
            s = e
        return total


def _quote(q: dict | None) -> dict | None:
    return None if q is None else {k: q.get(k) for k in ("bid", "ask", "mid", "mark")}


def _utc(x) -> pd.Timestamp:
    x = pd.Timestamp(x)
    return x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")


def fetch_fills(client: Client, start, end, symbol: str | None = None) -> pd.DataFrame:
    """Real demo fills in [start, end) from /order/fill-history (7-day windows, paged) for reconciliation."""
    start, end = _utc(start), _utc(end)
    rows, s = [], start
    while s < end:
        e = min(s + pd.Timedelta(days=7), end)
        last = None
        while True:
            p = {"productType": PRODUCT, "startTime": int(s.timestamp() * 1000), "endTime": int(e.timestamp() * 1000),
                 "limit": 100, "symbol": symbol, "idLessThan": last}
            d = client.request("GET", "/api/v2/mix/order/fill-history", p, auth=True) or {}
            page = d.get("fillList") or []
            rows += page
            if len(page) < 100 or not d.get("endId"):
                break
            last = d["endId"]
        s = e
    if not rows:
        return pd.DataFrame(columns=["ts", "ticker", "side", "trade_side", "qty", "price", "fee", "order_id",
                                     "trade_id", "pos_mode"])
    df = pd.DataFrame({
        "ts": pd.to_datetime([int(x["cTime"]) for x in rows], unit="ms", utc=True),
        "ticker": [x["symbol"].removesuffix("USDT") for x in rows],
        "side": [x.get("side") for x in rows], "trade_side": [x.get("tradeSide") for x in rows],
        "qty": [float(x["baseVolume"]) for x in rows], "price": [float(x["price"]) for x in rows],
        "fee": [-sum(float(f.get("totalFee") or 0) for f in (x.get("feeDetail") or [])) for x in rows],
        "order_id": [x["orderId"] for x in rows], "trade_id": [x.get("tradeId") for x in rows],
        "pos_mode": [x.get("posMode") for x in rows]})
    return df.drop_duplicates("trade_id").sort_values("ts").reset_index(drop=True)


def fetch_orders(client: Client, start, end, symbol: str | None = None) -> pd.DataFrame:
    """Real demo order history in [start, end) from /order/orders-history (7-day windows, paged)."""
    start, end = _utc(start), _utc(end)
    rows, s = [], start
    while s < end:
        e = min(s + pd.Timedelta(days=7), end)
        last = None
        while True:
            p = {"productType": PRODUCT, "startTime": int(s.timestamp() * 1000), "endTime": int(e.timestamp() * 1000),
                 "limit": 100, "symbol": symbol, "idLessThan": last}
            d = client.request("GET", "/api/v2/mix/order/orders-history", p, auth=True) or {}
            page = d.get("entrustedList") or []
            rows += page
            if len(page) < 100 or not d.get("endId"):
                break
            last = d["endId"]
        s = e
    return pd.DataFrame(rows)


# ---- live loop ---------------------------------------------------------------------------------
STEP_LAG_S = 90                      # run each hour's step this long after the bar close
TOPUP_DAYS = 3                       # bars / funding / news re-fetched over this trailing window


def _merge_parquet(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    old = pd.read_parquet(path) if path.exists() else None
    d = new if old is None or not len(old) else pd.concat([old, new], ignore_index=True)
    first = d.groupby(keys)["first_seen"].min() if "first_seen" in d else None   # news: keep the earliest sighting
    d = d.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    if first is not None:
        d["first_seen"] = d.set_index(keys).index.map(first)
    path.parent.mkdir(parents=True, exist_ok=True)
    d.attrs = {}
    d.to_parquet(path, index=False)


def refresh_data(t: pd.Timestamp, decision: bool) -> list[str]:
    """Best-effort top-up of data/raw for the live Store (bars, funding, news; MCP ratings/insider/F&G at
    decision hours). Returns warnings; a failed source leaves the older data in place (features then
    report a stale name's price as None)."""
    from sentiment import fetch_extra, fetch_news
    warn: list[str] = []
    since = int((t - pd.Timedelta(days=TOPUP_DAYS)).timestamp() * 1000)
    jobs = ([("perp", perp(tk)) for tk in UNIVERSE] + [("perp", s) for s in INDEX_PERPS]
            + [("spot", "R" + perp(tk)) for tk in UNIVERSE] + [("funding", perp(tk)) for tk in UNIVERSE])
    for kind, sym in jobs:
        try:
            new = fetch_extra.funding(sym, since) if kind == "funding" else fetch_extra.candles(kind, sym, since)
            if len(new):
                _merge_parquet(fetch_extra.TOPUP / kind / f"{sym}.parquet", new, ["ts"])
        except Exception as e:  # noqa: BLE001 - one source down must not stop the loop
            warn.append(f"{kind} {sym}: {e}"[:200])
    try:
        d = fetch_news.fetch(str((t - pd.Timedelta(days=TOPUP_DAYS)).date()), str((t + pd.Timedelta(days=1)).date()))
        d = d.assign(query_labels=d["query_labels"].map(lambda x: ",".join(map(str, x))), first_seen=t)
        _merge_parquet(fetch_news.OUT, d, ["published_at", "title"])
    except Exception as e:  # noqa: BLE001
        warn.append(f"news: {e}"[:200])
    if decision:
        try:
            fetch_extra.fetch_mcp()
        except Exception as e:  # noqa: BLE001
            warn.append(f"mcp ratings/insider/fng: {e}"[:200])
    return warn


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def step(now=None, *, variant: str = "llm", broker=None, store=None, refresh: bool = True, decide_fn=None,
         cards_fn=None, out_dir: Path | None = None, dry_run: bool = False) -> dict | None:
    """One live step for the hour containing `now` (default: the wall clock). Returns the logged record
    (a decision at DECISION_HOURS, a risk exit when the hourly check binds) or None."""
    from sentiment import agent, baseline, evidence, ledger, replay, risk
    from sentiment.config import DECISION_HOURS
    from sentiment.data import Store
    now = _utc(now if now is not None else _now())
    t = now.floor("h")
    out = Path(out_dir or LIVE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    rs_path = out / "runner_state.json"
    rs = _load_json(rs_path)
    if rs.get("last_t") and pd.Timestamp(rs["last_t"]) >= t:
        print(f"live: {t} already stepped", file=sys.stderr)
        return None
    decision = t.hour in DECISION_HOURS
    warn = refresh_data(t, decision) if refresh else []
    store = store or Store(snapshot=False)
    broker = broker or DemoBroker(dry_run=dry_run)
    lstate = rs.get("risk") or {}
    run_id = f"live_{variant}_{broker.mode}"
    m = broker.mark(t)
    pre_trade = {"positions": dict(m.get("positions") or {}), "prices": dict(m.get("prices") or {}),
                 "equity": m["equity"]}                     # exact book for sentiment.reconcile
    rec = None
    if not math.isfinite(m["equity"]):
        warn.append("mark: equity unavailable (no price for a held name); risk update and decision skipped")
    else:
        row = replay._mark_row(m)
        pd.DataFrame([row]).to_csv(out / "equity.csv", mode="a", index=False, header=not (out / "equity.csv").exists())
        risk.observe(lstate, t, m["equity"])
        if decision:
            errors: list = []
            items = replay._live(store.all_items(), t)            # items available in (t - lookback, t]
            cards = replay._live((cards_fn or evidence.cards_for)(items, errors=errors), t)
            policy = decide_fn or (baseline.decide if variant == "baseline" else agent.decide)
            rec, m = replay.decision_step(t, store=store, broker=broker, m=m, lstate=lstate, shown_cards=cards,
                                          true_cards=cards, variant=variant, policy=policy, run_id=run_id, mode="live")
            rec["evidence_errors"] = [str(e)[:200] for e in errors]
        else:
            rec, m = replay.risk_exit_step(t, broker=broker, m=m, lstate=lstate, variant=variant, run_id=run_id,
                                           mode="live")
    if rec is not None:
        rec.update({"wall_ts": now, "data_warnings": warn, "pre_trade": pre_trade})
        rec = ledger.Ledger(out / "log.jsonl").append(rec)
    for w in warn:
        print(f"live {t:%m-%d %H:%M}: {w}", file=sys.stderr)
    rs_path.write_text(json.dumps({"last_t": t.isoformat(), "variant": variant, "risk": lstate}, default=str, indent=1))
    return rec


def run_loop(variant: str = "llm", dry_run: bool = False) -> None:
    """Step every hour, STEP_LAG_S after the bar close, until interrupted."""
    import traceback
    broker = DemoBroker(dry_run=dry_run)
    print(f"live loop: variant {variant}, broker mode {broker.mode}", file=sys.stderr)
    while True:
        try:
            rec = step(variant=variant, broker=broker)
            if rec is not None:
                print(f"{rec['ts']} {rec['event']} eq {rec['equity']:.2f} fills {len(rec['fills'])}", file=sys.stderr)
        except Exception:  # noqa: BLE001 - keep the loop alive; the next hour retries
            traceback.print_exc()
        now = _now()
        nxt = now.floor("h") + pd.Timedelta(hours=1, seconds=STEP_LAG_S)
        time.sleep(max(1.0, (nxt - now).total_seconds()))


# ---- CLI ---------------------------------------------------------------------------------------
def _spr_bp(q: dict | None) -> float | None:
    return round((q["ask"] - q["bid"]) / q["mid"] * 1e4, 1) if q and q.get("bid") and q.get("ask") else None


def check() -> None:
    c = Client()
    specs = contracts(c, demo=True)
    demo, live = tickers(c, demo=True), tickers(c, demo=False)
    syms = [perp(t) for t in UNIVERSE] + INDEX_PERPS
    rows = []
    for s in syms:
        sp, d, lv = specs.get(s), demo.get(s), live.get(s)
        rows.append({
            "symbol": s, "demo": "yes" if sp else "NO", "status": sp and sp["status"],
            "min_qty": sp and sp["min_qty"], "step": sp and sp["step"], "vol_place": sp and sp["volume_place"],
            "min_usdt": sp and sp["min_usdt"], "max_mkt_qty": sp and sp["max_market_qty"],
            "max_lev": sp and sp["max_lever"],
            "demo_bid": d and d["bid"], "demo_ask": d and d["ask"], "demo_spr_bp": _spr_bp(d),
            "live_mid": lv and lv["mid"], "live_spr_bp": _spr_bp(lv),
            "demo-live_bp": (d and lv) and round((d["mid"] / lv["mid"] - 1) * 1e4, 1),
            "demo_fund": d and d["funding"], "live_fund": lv and lv["funding"]})
    df = pd.DataFrame(rows).set_index("symbol")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.to_string())
    print(f"\nmedian spread (bp): demo {df['demo_spr_bp'].median()}, live {df['live_spr_bp'].median()}; "
          f"shadow fills use live quotes, real demo orders pay the demo spread")
    print(f"demo contracts on {PRODUCT}: {len(specs)}; demo keys configured: {load_keys() is not None}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="print demo contracts and demo vs live tickers")
    ap.add_argument("--once", action="store_true", help="run one live step for the current hour")
    ap.add_argument("--run", action="store_true", help="run the hourly live loop")
    ap.add_argument("--variant", default="llm", choices=("baseline", "llm", "nonews", "blinded"))
    ap.add_argument("--shadow", action="store_true", help="never send orders (shadow ledger at live quotes)")
    ap.add_argument("--risk-version", default=None, choices=("v0", "v1"),
                    help="risk layer + prompt version (default config.RISK_VERSION, env SENTIMENT_RISK_VERSION)")
    a = ap.parse_args(argv)
    if a.risk_version:
        from sentiment import config
        config.RISK_VERSION = a.risk_version
    if a.check:
        check()
    elif a.once:
        rec = step(variant=a.variant, dry_run=a.shadow)
        print(json.dumps(None if rec is None else {k: rec.get(k) for k in ("ts", "event", "equity", "targets", "hash")},
                         default=str, indent=1))
    elif a.run:
        run_loop(a.variant, dry_run=a.shadow)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
