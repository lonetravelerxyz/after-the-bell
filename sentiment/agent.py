"""LLM decision policy (contract: sentiment/CONTRACT.md, Policies; spec §5.2).

`decide(t, state, cards, book, variant)` builds one compact prompt (system prompt
sentiment/prompts/decide.md, user message = market state table + evidence cards + current book),
asks Qwen through the cached `sentiment.llm.complete`, and parses the JSON reply into a Decision.
The LLM is the decision-maker; the risk layer enforces the limits afterwards. Unparseable output or
an endpoint failure gives `valid=False` (the risk layer then holds the book). An endpoint failure also
sets `error`: nothing is cached for that prompt, so the replay runner stops on it (the live runner holds
and logs it); `CacheMiss` in offline mode propagates, because a replay that cannot be reproduced must stop.

Variants:
  llm       full prompt: date and time, tickers, evidence cards;
  nonews    same prompt with no evidence cards (price, funding, premium and F&G only);
  shuffled  as llm; the runner has already moved the cards to wrong dates;
  blinded   tickers replaced by aliases A..J (also inside card summaries, with company names),
            no date or clock time. Card summaries are masked by `blind_summary`: company names ->
            alias; company-unique people/products (IDENTITY) -> [name]; dates, month and weekday
            names, M/D dates and quarters -> [date]; dollar amounts -> [amount]; analyst price
            targets -> % change only; insider cards -> role, side and notional bucket (no owner
            name, share count or price). Sector words (AI, chips, Bitcoin, crypto) are kept: they
            carry the evidence. Replies are mapped back.
The market state is shown as returns, funding and premia only, never price levels.
Extra Decision keys beyond the contract: `shown` (card ids in the prompt), `llm_request_hash`,
`prompt_version`.

Prompt versions. v0 = prompts/decide.md, frozen byte-identical to 94cbe51 (every run before risk v1, and
any process still running that code, reads that file); v1 = prompts/decide_v1.md. The version follows the
risk context: a call without `risk_ctx` (the CONTRACT signature; the runner under risk v0) builds exactly the
v0 prompt, a call with one builds the v1 prompt. The Decision's `prompt_version` says which.

Risk context (PROMPT_VERSION v1). `decide(..., risk_ctx=...)` takes the optional risk context the runner
builds before the risk layer runs (`sentiment.replay.risk_context`, schema there): the per-name cap in
force now, beta, cooldown, P&L since entry and distance to the stop, book net and beta-weighted net,
and for every shown card the move of its tickers since the card became available. It is shown as a
RISK table and a `since_bp` column in the evidence list; the system prompt tells the model that the risk
layer beta-neutralises the book and scales caps by volatility, so it should express relative views.
Only returns, weights, betas and hour counts are shown (no price levels, no dates), so the blinded
variant stays identity-free. Without `risk_ctx` the user message has no RISK block (CONTRACT signature).
"""
from __future__ import annotations

import math
import random
import re

import pandas as pd

from sentiment import llm
from sentiment.config import (CARD_LOOKBACK_H, DAILY_KILL, FUNDING_BLOCK, MAX_GROSS, MAX_NET, MAX_W_NAME, ROOT,
                              UNIVERSE)
from sentiment.evidence import NAMES

PROMPT_FILE = ROOT / "sentiment" / "prompts" / "decide.md"            # v0: frozen, never edit
PROMPT_FILES = {"v0": PROMPT_FILE, "v1": ROOT / "sentiment" / "prompts" / "decide_v1.md"}
PROMPT_VERSION = "v1"      # the current prompt: v0 = decide.md at 94cbe51 (no risk context); v1 adds the RISK table
VARIANTS = ("llm", "nonews", "shuffled", "blinded")
MAX_CARDS = 60             # prompt budget; beyond it post_hoc cards go first, then the oldest
ALIAS = dict(zip(random.Random(20260923).sample(UNIVERSE, len(UNIVERSE)), "ABCDEFGHIJ"))
UNALIAS = {v: k for k, v in ALIAS.items()}
MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_RE = re.compile(rf"\b{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+20\d\d)?\b|\b20\d\d-\d\d-\d\d\b|\b20\d\d\b")
BLIND_DATE_RE = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\b\.?|\b{MONTHS}\b\.?|\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b"
    r"|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b[1-4]Q\b|\bQ[1-4]\b|\bFY\s?\d{2,4}\b")
MONEY_RE = re.compile(r"(?:US)?\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:[KMBT]\b|k\b|bn\b|thousand|million|billion|trillion))?")
HOLDINGS_RE = re.compile(r"~?\b\d[\d,.]*\s?[Kk]?\s+(?=(?:BTC|bitcoins?|Bitcoins?)\b)")   # a treasury's coin count
PT_CHG_RE = re.compile(r"PT\s+[\d.,]+\s*->\s*[\d.,]+\s*\(([+-]?\d+%)\)")
PT_LEVEL_RE = re.compile(r",?\s*\bPT\s+\d[\d.,]*(?:\s*->\s*\d[\d.,]*)?")
AT_PRICE_RE = re.compile(r"(\b(?:at|near|to|from|above|below)\s+|@\s*)\d[\d,]*\.\d+\b(?!\s*%)")
INSIDER_RE = re.compile(r"^(?P<who>.*?)\s+(?P<verb>bought|sold)\s+[\d,]+\s+\S+\s+shares\s+at\s+[\d,.]+\s+\(\$(?P<m>[\d,.]+)M\)")
# company-unique people, products and brands (case-sensitive); masked to [name] in blinded prompts
IDENTITY = {
    "NVDA": ["Jensen Huang", "Huang", "Jensen", "Blackwell", "Rubin", "Hopper", "CUDA", "GeForce", "DGX", "NVLink",
             "H20", "H100", "H200", "B200", "B300", "GB200", "GB300"],
    "AAPL": ["Tim Cook", "Cook", "iPhone", "iPhones", "iPad", "iOS", "macOS", "MacBook", "Mac", "Vision Pro", "Siri",
             "App Store", "WWDC", "AirPods", "Apple Intelligence"],
    "GOOGL": ["Sundar Pichai", "Pichai", "Gemini", "YouTube", "Waymo", "Android", "Chrome", "DeepMind", "TPU", "TPUs",
              "Pixel"],
    "META": ["Mark Zuckerberg", "Zuckerberg", "Instagram", "WhatsApp", "Llama", "Threads", "Reality Labs", "Ray-Ban",
             "Superintelligence Labs"],
    "AMZN": ["Andy Jassy", "Jassy", "Jeff Bezos", "Bezos", "Prime", "Alexa", "Kuiper", "Trainium", "Whole Foods"],
    "TSLA": ["Elon Musk", "Musk", "Elon", "Optimus", "Cybercab", "Cybertruck", "Robotaxi", "robotaxi", "robotaxis",
             "FSD", "Model Y", "Model 3", "Model S", "Model X", "Megapack", "Powerwall", "Gigafactory", "xAI", "SpaceX", "Semi"],
    "MSTR": ["Michael Saylor", "Saylor", "Strategy", "STRK", "STRF", "STRD", "STRC"],
    "COIN": ["Brian Armstrong", "Armstrong", "Deribit"],
    "HOOD": ["Vlad Tenev", "Tenev", "Bitstamp"],
    "CRCL": ["Jeremy Allaire", "Allaire", "USDC", "EURC", "Arc"],
}


def _ts(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _card_id(c: dict) -> str:
    return str(c.get("card_id") or c["item_id"])


# ---------------------------------------------------------------- prompt

def prompt_version(risk_ctx: dict | None) -> str:
    return "v0" if risk_ctx is None else PROMPT_VERSION


def system_prompt(blinded: bool = False, version: str = PROMPT_VERSION) -> str:
    tickers = ", ".join(ALIAS[tk] for tk in sorted(ALIAS, key=ALIAS.get)) if blinded else ", ".join(UNIVERSE)
    subs = {"{{TICKERS}}": tickers, "{{MAX_W}}": f"{MAX_W_NAME:g}", "{{MAX_GROSS}}": f"{MAX_GROSS:g}",
            "{{MAX_NET}}": f"{MAX_NET:g}", "{{FUNDING_BLOCK_BP}}": f"{FUNDING_BLOCK * 1e4:g}",
            "{{DAILY_KILL_PCT}}": f"{DAILY_KILL * 100:g}"}
    text = PROMPT_FILES[version].read_text()
    for k, v in subs.items():
        text = text.replace(k, v)
    return text.strip()


_NAME_RE = [(re.compile(rf"\b{re.escape(a)}\b(?:'s)?"), tk)
            for tk in UNIVERSE for a in sorted([tk, *NAMES.get(tk, [])], key=len, reverse=True)]


_IDENTITY_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(
    {w for ws in IDENTITY.values() for w in ws}, key=len, reverse=True)) + r")\b(?:'s)?")


def blind_text(s: str) -> str:
    """Tickers and company names -> 'Company X'; company-unique names -> '[name]'; dates, months, weekdays,
    M/D dates, quarters and years -> '[date]'; price-target levels -> % change; dollar amounts and
    decimal prices -> '[amount]' / '[price]'."""
    s = PT_CHG_RE.sub(r"PT \1", s)
    s = PT_LEVEL_RE.sub("", s)
    s = MONEY_RE.sub("[amount]", s)
    s = HOLDINGS_RE.sub("[amount] ", s)
    for rx, tk in _NAME_RE:
        s = rx.sub(f"Company {ALIAS[tk]}", s)
    s = _IDENTITY_RE.sub("[name]", s)
    s = DATE_RE.sub("[date]", s)
    s = BLIND_DATE_RE.sub("[date]", s)
    return AT_PRICE_RE.sub(r"\1[price]", s)


def _bucket(musd: float) -> str:
    for hi, label in ((0.1, "under 0.1M USD"), (1, "0.1-1M USD"), (10, "1-10M USD"), (25, "10-25M USD")):
        if musd < hi:
            return label
    return "over 25M USD"


def blind_summary(card: dict) -> str:
    """The card summary as shown in the blinded variant (see module docstring)."""
    s = str(card.get("summary") or "")
    if card.get("kind") == "insider":
        tk = next(iter(card.get("tickers") or []), "")
        m = INSIDER_RE.match(s)
        if m:
            role = re.findall(r"\(([^()]*)\)", m["who"])
            role = role[-1] if role else "insider"
            s = f"An insider ({role}) {m['verb']} {tk} shares, notional {_bucket(float(m['m'].replace(',', '')))}"
        else:
            s = f"Insider transaction in {tk} shares"
    return blind_text(s)


def select_cards(t, cards: list[dict]) -> list[dict]:
    """Cards available in (t - CARD_LOOKBACK_H, t], at most MAX_CARDS, newest first."""
    t = _ts(t)
    lo = t - pd.Timedelta(hours=CARD_LOOKBACK_H)
    live = [c for c in cards if lo < _ts(c["available_at"]) <= t]
    if len(live) > MAX_CARDS:
        live.sort(key=lambda c: (c.get("novelty") != "post_hoc", _ts(c["available_at"]), _card_id(c)), reverse=True)
        live = live[:MAX_CARDS]
    return sorted(live, key=lambda c: (_ts(c["available_at"]), _card_id(c)), reverse=True)


def _pct(x, nd=2) -> str:
    return "na" if x is None or not math.isfinite(x) else f"{x * 100:+.{nd}f}"


def _bp(x, nd=1) -> str:
    return "na" if x is None or not math.isfinite(x) else f"{x * 1e4:+.{nd}f}"


def _hours_left(until, t: pd.Timestamp) -> float | None:
    if until is None:
        return None
    h = (_ts(until) - t).total_seconds() / 3600
    return h if h > 0 else None


def _since(c: dict, risk_ctx: dict | None, name) -> str:
    """The 'since_bp' cell of a card: move of its tickers (market cards: the SP500 perp) since available_at."""
    moves = ((risk_ctx or {}).get("since_card") or {}).get(_card_id(c)) or {}
    cells = [("mkt" if k == "_market" else name(k), v) for k, v in moves.items() if k == "_market" or k in UNIVERSE]
    cells = [f"{lab} {_bp(v, 0)}" for lab, v in cells if v is not None and math.isfinite(v)]
    return ",".join(cells) if cells else "-"


def risk_lines(t, risk_ctx: dict, order: list[str], name) -> list[str]:
    """The RISK block of the user message: book net / beta net, then one row per name.
    Only weights, betas, returns and hour counts: no price levels, no dates (safe for the blinded variant)."""
    t = _ts(t)
    caps, beta = risk_ctx.get("caps") or {}, risk_ctx.get("beta") or {}
    cool, entry = risk_ctx.get("cooldown_until") or {}, risk_ctx.get("entry") or {}
    bnet = risk_ctx.get("book_beta_net")
    lines = ["RISK (applied after you: per-name cap in force now, beta-neutral book, stops; weights, % and hours)",
             f"book net {risk_ctx.get('book_net', 0.0):+.3f}; beta-weighted net "
             f"{'na' if bnet is None else f'{bnet:+.3f}'}"]
    kill = _hours_left(risk_ctx.get("kill_until"), t)
    if kill:
        lines.append(f"DAILY KILL in force for {kill:.0f}h more: every target is held at 0 until then.")
    if risk_ctx.get("us_session_open") is False:
        held = risk_ctx.get("caps_held") or {}
        cells = ", ".join(f"{name(tk)} {held[tk]:.3f}" for tk in order if tk in held)
        lines.append("US session closed: cap limits NEW exposure only. A position already held on the same side is "
                     f"not cut to it: it may stay at its current size up to cap_held ({cells}).")
    lines.append("ticker beta cap cooldown_h pnl_entry to_stop")
    for tk in order:
        b, c, e = beta.get(tk), caps.get(tk), entry.get(tk) or {}
        cd = _hours_left(cool.get(tk), t)
        lines.append(" ".join([name(tk), "na" if b is None else f"{b:.2f}", "na" if c is None else f"{c:.3f}",
                               "-" if cd is None else f"{cd:.0f}",
                               _pct(e.get("pnl")) if e else "-",
                               _pct(e.get("to_stop")).lstrip("+") if e else "-"]))
    return lines


def user_message(t, state: dict, cards: list[dict], book: dict[str, float], variant: str,
                 risk_ctx: dict | None = None) -> str:
    t = _ts(t)
    blinded = variant == "blinded"
    name = (lambda tk: ALIAS[tk]) if blinded else (lambda tk: tk)
    order = sorted(UNIVERSE, key=ALIAS.get) if blinded else list(UNIVERSE)
    session = next((state[tk].get("us_session_open") for tk in UNIVERSE if tk in state), None)
    m = state.get("_market") or {}
    lines = []
    if not blinded:
        lines.append(f"NOW: {t:%Y-%m-%d %H:%M} UTC ({t:%A})")
    lines.append(f"US regular session: {'open' if session else 'closed'}")
    fng = m.get("fng")
    lines.append(f"MARKET: crypto fear&greed {fng if fng is not None else 'na'} ({m.get('fng_regime') or 'na'}); "
                 f"SP500 perp 24h {_pct(m.get('sp500_ret_24h'))}%; NDX100 perp 24h {_pct(m.get('ndx_ret_24h'))}%")
    lines += ["", "NAMES (ret/vol in %, funding/premium in bp, book = current weight)",
              "ticker ret_4h ret_24h ret_7d vol_1d funding funding_7d premium book"]
    for tk in order:
        s = state.get(tk) or {}
        lines.append(" ".join([name(tk), _pct(s.get("ret_4h")), _pct(s.get("ret_24h")), _pct(s.get("ret_7d")),
                               _pct(s.get("vol_7d")).lstrip("+"), _bp(s.get("funding_last"), 2),
                               _bp(s.get("funding_7d_mean"), 2), _bp(s.get("spot_premium")),
                               f"{book.get(tk, 0.0):+.3f}"]))
    if risk_ctx is not None:
        lines += [""] + risk_lines(t, risk_ctx, order, name)
    lines.append("")
    if variant == "nonews":
        lines.append("EVIDENCE: not provided in this run.")
    elif not cards:
        lines.append(f"EVIDENCE (last {CARD_LOOKBACK_H}h): none.")
    else:
        since = risk_ctx is not None
        lines += [f"EVIDENCE (last {CARD_LOOKBACK_H}h, newest first"
                  + ("; since_bp = move of the card's ticker(s) since it became available, mkt = SP500 perp)"
                     if since else ")"),
                  "id | age_h | " + ("since_bp | " if since else "")
                  + "scope subject | stance | strength | horizon_h | novelty | summary"]
        for c in cards:
            age = (t - _ts(c["available_at"])).total_seconds() / 3600
            if c["scope"] == "name":
                subj = "name " + ",".join(name(tk) for tk in c.get("tickers") or [] if tk in UNIVERSE)
            elif c["scope"] == "sector":
                subj = f"sector {c.get('sector') or 'other'}"
            else:
                subj = "market"
            summ = str(c.get("summary") or "").replace("\n", " ").replace("|", "/")
            summ = blind_summary({**c, "summary": summ}) if blinded else summ
            flag = "POST_HOC (move already happened)" if c.get("novelty") == "post_hoc" else c.get("novelty", "new")
            mid = f"{_since(c, risk_ctx, name)} | " if since else ""
            lines.append(f"{_card_id(c)} | {age:.0f} | {mid}{subj} | {int(c['stance']):+d} | "
                         f"{float(c['strength']):.2f} | {int(c.get('horizon_h') or 24)} | {flag} | {summ}")
    held = [f"{name(tk)} {book[tk]:+.3f}" for tk in order if abs(book.get(tk, 0.0)) > 1e-9]
    lines += ["", "CURRENT BOOK: " + (", ".join(held) if held else "flat"),
              "", "Return the JSON decision for every ticker."]
    return "\n".join(lines)


def build_messages(t, state: dict, cards: list[dict], book: dict[str, float], variant: str,
                   risk_ctx: dict | None = None) -> list[dict]:
    """System + user message; the v0 prompt without `risk_ctx`, the v1 prompt with it."""
    return [{"role": "system", "content": system_prompt(variant == "blinded", prompt_version(risk_ctx))},
            {"role": "user", "content": user_message(t, state, cards, book, variant, risk_ctx)}]


# ---------------------------------------------------------------- reply

def _key(k, blinded: bool) -> str:
    k = str(k).upper().strip().replace("USDT", "")
    k = re.sub(r"^COMPANY\s+", "", k)
    return UNALIAS.get(k, k) if blinded else k


def parse_decision(text: str, shown: list[str], blinded: bool = False) -> dict:
    """Model reply -> Decision. Invalid (valid=False, empty targets) if no usable targets object."""
    bad = {"targets": {}, "reasons": {}, "evidence": {}, "confidence": 0.0, "raw": text, "valid": False}
    try:
        obj = llm.parse_json(text)
    except ValueError:
        return bad
    tg = obj.get("targets") if isinstance(obj, dict) else None
    if not isinstance(tg, dict):
        return bad
    try:
        targets = {_key(k, blinded): float(v if v is not None else 0.0) for k, v in tg.items()}
    except (TypeError, ValueError):
        return bad
    if not all(math.isfinite(v) for v in targets.values()):
        return bad
    ok = set(shown)
    reasons = {_key(k, blinded): str(v) for k, v in (obj.get("reasons") or {}).items()} \
        if isinstance(obj.get("reasons"), dict) else {}
    evidence = {}
    if isinstance(obj.get("evidence"), dict):
        for k, v in obj["evidence"].items():
            ids = [v] if isinstance(v, str) else (v if isinstance(v, list) else [])
            evidence[_key(k, blinded)] = [str(i) for i in ids if str(i) in ok]
    try:
        conf = min(1.0, max(0.0, float(obj.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"targets": {k: v for k, v in targets.items() if v != 0.0}, "reasons": reasons, "evidence": evidence,
            "confidence": conf, "raw": text, "valid": True}


def decide(t, state: dict, cards: list[dict], book: dict[str, float], variant: str = "llm", *,
           risk_ctx: dict | None = None) -> dict:
    """CONTRACT policy. `risk_ctx` (optional, from `replay.risk_context`) adds the RISK table and the
    since_bp card column to the prompt; it does not change the parsing or any other logic."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; one of {VARIANTS}")
    shown_cards = [] if variant == "nonews" else select_cards(t, cards)
    shown = [_card_id(c) for c in shown_cards]
    messages = build_messages(t, state, shown_cards, book, variant, risk_ctx)
    req = llm.cache_key(llm.model_name(), messages)
    try:
        text = llm.complete(messages, purpose=f"decide:{variant}")
    except llm.LLMError as e:
        d = parse_decision("", shown)
        d["raw"] = f"LLMError: {e}"
        d["error"] = str(e)
    else:
        d = parse_decision(text, shown, variant == "blinded")
    return {**d, "shown": shown, "llm_request_hash": req, "prompt_version": prompt_version(risk_ctx)}
