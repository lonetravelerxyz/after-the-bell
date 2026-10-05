"""Evidence cards (contract: sentiment/CONTRACT.md, "Evidence"; spec docs/spec-B-market-sentiment.md §5.1).

News items -> one LLM call each (prompt sentiment/prompts/extract.md, cached by sentiment.llm); the
model reads only the title and the cleaned text and is never asked about prices. The prompt injects no
"as of" date, but titles and bodies often carry their own dates (568 of 614 extraction messages contain
a calendar date or year). Each item is extracted in isolation, so those dates only reveal when that item
itself was published; what rules out parametric look-ahead is the model's knowledge cutoff (about
2025-04/05, sentiment/out/cutoff_probe.json), 13+ months before the replay window.
One item may yield several cards: one per watched ticker it actually discusses, up to two sector
cards and one market card. Titles that explain a move already made ("[Market Movement Analysis]",
"X surges on September 17: ...") are forced to novelty "post_hoc" whatever the model says.
Analyst ratings and insider filings -> deterministic cards, no LLM.

Cards persist in sentiment/cache/cards.parquet keyed by item_id; each row carries the extractor
version (prompt hash for news), so a prompt change re-extracts and already-extracted items are skipped.
`cards_for` never returns partial evidence: after saving what succeeded it raises `CacheMiss` (offline
and not cached) or `LLMError` (endpoint failure) if any news item could not be extracted.

Run: uv run python -m sentiment.evidence [n]   -> extracts cards for the first n news items (demo)
"""
from __future__ import annotations

import hashlib
import html
import math
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from sentiment import llm
from sentiment.config import CACHE, RAW, ROOT, UNIVERSE

CARDS_FILE = CACHE / "cards.parquet"
PROMPT_FILE = ROOT / "sentiment" / "prompts" / "extract.md"
MAX_CHARS = 8000          # cleaned text budget per news item (~2k tokens)
HEAD_CHARS = 5000         # always keep the start; fill the rest with paragraphs naming watched companies
RATING_VERSION, INSIDER_VERSION = "rating:v1", "insider:v1"

NAMES = {"NVDA": ["NVIDIA", "Nvidia"], "AAPL": ["Apple"], "GOOGL": ["Alphabet", "Google", "GOOG"],
         "META": ["Meta Platforms", "Meta", "Facebook"], "AMZN": ["Amazon", "AWS"], "TSLA": ["Tesla"],
         "MSTR": ["MicroStrategy", "Strategy Inc"], "COIN": ["Coinbase"], "HOOD": ["Robinhood"],
         "CRCL": ["Circle Internet", "Circle"]}
SECTORS = {"ai_semis", "megacap_tech", "crypto", "fintech", "ev_auto", "other"}
COLS = ["card_id", "item_id", "kind", "available_at", "tickers", "scope", "sector", "stance", "strength",
        "horizon_h", "novelty", "summary", "quote", "extractor"]
POST_HOC = re.compile(
    r"(market\s+move(ment|rs)?\s+(analysis|interpretation)|movement\s+(analysis|interpretation)"
    r"|unusual\s+movement|(anomaly|fluctuation)\s+(analysis|interpretation)"
    r"|\b(surg|soar|jump|plung|tumbl|slump|rall|spik|sink|drop|fall|rise|rose|climb|gain|slid)\w*\s+on\s+"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2})", re.IGNORECASE)


# ---------------------------------------------------------------- news text

def clean_text(raw: str) -> str:
    """HTML news body -> plain text with one paragraph per line, links and images dropped."""
    t = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw or "")
    t = re.sub(r"(?i)<br\s*/?>|</?(p|div|li|h\d|tr|ul|ol)\b[^>]*>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+", "", html.unescape(t))
    t = re.sub(r"[ \t 　]+", " ", t)
    return "\n".join(ln.strip() for ln in t.splitlines() if ln.strip())


def _mentions(s: str) -> bool:
    return any(re.search(rf"\b{re.escape(a)}\b", s) for tk in UNIVERSE for a in [tk, *NAMES.get(tk, [])])


def truncate(text: str, max_chars: int = MAX_CHARS, head: int = HEAD_CHARS) -> str:
    """Keep the head of the text, then later paragraphs that name a watched company, within budget."""
    if len(text) <= max_chars:
        return text
    out = [text[:head].rsplit("\n", 1)[0]]
    used = len(out[0])
    for para in text[len(out[0]):].split("\n"):
        if para and _mentions(para) and used + len(para) + 7 <= max_chars:
            out.append(para)
            used += len(para) + 7
    return "\n[...]\n".join(out)


def _universe_block() -> str:
    return "\n".join(f"- {tk}: {', '.join(NAMES.get(tk, []))}" for tk in UNIVERSE)


def system_prompt() -> str:
    return PROMPT_FILE.read_text().replace("{{UNIVERSE}}", _universe_block()).strip()


def news_version() -> str:
    return "news:" + hashlib.sha256(system_prompt().encode()).hexdigest()[:10]


def news_messages(item: dict) -> list[dict]:
    body = truncate(clean_text(item.get("text") or ""))
    user = f"TITLE: {item.get('title', '').strip()}\n\nTEXT:\n{body or '(no body; title only)'}"
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


def _clip(x, lo, hi, default):
    try:
        return min(hi, max(lo, type(default)(float(x))))
    except (TypeError, ValueError):
        return default


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def fix_quote(quote: str, haystack: str) -> str:
    """Keep the quote verbatim: if the model joined excerpts with ellipses, keep the longest piece
    that occurs in the text; otherwise return it unchanged."""
    q = _ws(quote)
    if not q or q.lower() in haystack.lower():
        return q
    parts = sorted((_ws(x).strip(" .\"'") for x in re.split(r"\.\.\.|\u2026", q)), key=len, reverse=True)
    return next((x for x in parts if len(x) >= 15 and x.lower() in haystack.lower()), q)


def normalize(raw_cards: list, item: dict) -> list[dict]:
    """Validate model cards against the Card schema; drop what cannot be repaired."""
    post_hoc = bool(POST_HOC.search(item.get("title", "")))
    hay = _ws(item.get("title", "") + " " + clean_text(item.get("text") or ""))
    out, seen = [], set()
    for c in raw_cards if isinstance(raw_cards, list) else []:
        if not isinstance(c, dict):
            continue
        tks = c.get("tickers") or []
        tks = [tks] if isinstance(tks, str) else tks
        tks = sorted({str(t).upper().strip().replace("USDT", "") for t in tks} & set(UNIVERSE))
        scope = str(c.get("scope", "")).lower().strip()
        if scope == "name" and not tks:
            scope = "sector"
        if scope not in ("name", "sector", "market"):
            scope = "name" if tks else "market"
        if scope != "name":
            tks = []
        sector = str(c.get("sector") or "").strip().lower() if scope == "sector" else ""
        sector = sector if sector in SECTORS else ("other" if scope == "sector" else "")
        novelty = str(c.get("novelty", "")).lower().strip().replace("-", "_")
        novelty = "post_hoc" if post_hoc else (novelty if novelty in ("new", "recap", "post_hoc") else "new")
        base = {"tickers": tks, "scope": scope, "sector": sector,
                "stance": _clip(c.get("stance"), -2, 2, 0), "strength": round(_clip(c.get("strength"), 0.0, 1.0, 0.0), 3),
                "horizon_h": _clip(c.get("horizon_h"), 1, 720, 24), "novelty": novelty,
                "summary": str(c.get("summary") or "").strip()[:200], "quote": fix_quote(str(c.get("quote") or ""), hay)[:300]}
        if scope == "name":                      # one card per ticker
            for tk in tks:
                if ("name", tk) not in seen:
                    seen.add(("name", tk))
                    out.append({**base, "tickers": [tk]})
        elif (scope, sector) not in seen:
            seen.add((scope, sector))
            out.append(base)
    return out


def _news_cards(item: dict) -> list[dict] | Exception:
    try:
        text = llm.complete(news_messages(item), purpose="evidence")
    except (llm.CacheMiss, llm.LLMError) as e:
        return e
    try:
        parsed = llm.parse_json(text)
    except ValueError:
        print(f"evidence: {item['id'][:10]} unparseable model output: {text[:120]!r}", file=sys.stderr)
        parsed = {"cards": []}
    raw = parsed.get("cards", []) if isinstance(parsed, dict) else parsed
    return normalize(raw, item)


# ---------------------------------------------------------------- deterministic cards

BULL = {"买入", "强力买进", "强力买入", "跑赢大盘", "增持", "积极", "最佳选择", "长期买入", "好于板块"}
BEAR = {"减持", "卖出", "逊于大盘", "谨慎", "消极"}
RATING_EN = {"买入": "Buy", "强力买进": "Strong Buy", "强力买入": "Strong Buy", "跑赢大盘": "Outperform",
             "增持": "Overweight", "积极": "Positive", "最佳选择": "Top Pick", "长期买入": "Long-term Buy",
             "好于板块": "Sector Outperform", "中性": "Neutral", "持有": "Hold", "持股观望": "Hold",
             "市场持平": "Market Perform", "行业一致": "In-Line", "板块表现": "Sector Perform",
             "同业一致": "Peer Perform", "大市表现": "Market Perform", "喜忧参半": "Mixed", "不评级": "Not Rated",
             "减持": "Underweight", "卖出": "Sell", "逊于大盘": "Underperform", "谨慎": "Cautious", "消极": "Negative"}
ACTION_EN = {"维持": "maintains", "重申": "reiterates", "首次覆盖": "initiates", "调高评级": "upgrades",
             "下调评级": "downgrades", "假设": "assumes", "恢复": "resumes", "暂停评级": "suspends"}


def _tier(r) -> int | None:
    return None if not r else (1 if r in BULL else -1 if r in BEAR else 0)


def _num(x) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) and v > 0 else None
    except (TypeError, ValueError):
        return None


def _ticker(item: dict) -> str | None:
    raw = item.get("raw") or {}
    cand = (item.get("tickers") or []) + [raw.get("symbol"), raw.get("ticker")]
    return next((str(t).upper() for t in cand if t and str(t).upper() in UNIVERSE), None)


def rating_card(item: dict) -> dict | None:
    """Upgrade / bullish initiation / PT raise positive; downgrade / bearish initiation / PT cut negative;
    magnitude from the PT change. A maintained rating with an unchanged PT is a stance-0 recap."""
    r, tk = item.get("raw") or {}, _ticker(item)
    if tk is None:
        return None
    action = r.get("action") or r.get("rating_chg_cn") or ""
    cur, prev = r.get("rating_current") or r.get("latest_rating_cn"), r.get("rating_previous") or r.get("pre_rating_cn")
    pt, pt0 = _num(r.get("price_target") or r.get("latest_target_price")), _num(r.get("price_target_previous") or r.get("pre_target_price"))
    chg = pt / pt0 - 1 if pt and pt0 else None
    if action == "暂停评级":
        return None
    t_cur, t_prev = _tier(cur), _tier(prev)
    if action == "调高评级" or (t_cur is not None and t_prev is not None and t_cur > t_prev):
        stance, strength, novelty = 1, 0.5, "new"
        if (t_prev is not None and t_cur is not None and t_cur - t_prev >= 2) or (chg or 0) >= 0.15:
            stance = 2
    elif action == "下调评级" or (t_cur is not None and t_prev is not None and t_cur < t_prev):
        stance, strength, novelty = -1, 0.5, "new"
        if (t_prev is not None and t_cur is not None and t_prev - t_cur >= 2) or (chg or 0) <= -0.15:
            stance = -2
    elif action in ("首次覆盖", "假设", "恢复") or (not action and prev is None and pt0 is None):
        stance, strength, novelty = (t_cur or 0), (0.4 if t_cur else 0.15), "new"
    elif chg is not None and abs(chg) >= 0.02:
        stance, strength, novelty = (1 if chg > 0 else -1), 0.25, "new"
        if abs(chg) >= 0.15 and (t_cur or 0) * chg > 0:
            stance *= 2
    else:
        stance, strength, novelty = 0, 0.1, "recap"
    if chg is not None:
        strength = min(1.0, strength + min(0.4, 1.5 * abs(chg)))
    firm = r.get("analyst_firm") or r.get("rating_org") or "An analyst"
    cur_en, prev_en = RATING_EN.get(cur, cur or "n/a"), RATING_EN.get(prev, prev)
    rating_txt = f"{prev_en} -> {cur_en}" if prev_en and prev_en != cur_en else cur_en
    pt_txt = f", PT {pt0:g} -> {pt:g} ({chg:+.0%})" if chg is not None else (f", PT {pt:g}" if pt else "")
    summary = f"{firm} {ACTION_EN.get(action, action or 'rates')} {tk} {rating_txt}{pt_txt}"[:200]
    quote = " ".join(str(x) for x in [firm, action, prev, "->" if prev else None, cur, pt0, "->" if pt0 else None, pt] if x)
    return {"tickers": [tk], "scope": "name", "sector": "", "stance": int(stance), "strength": round(strength, 3),
            "horizon_h": 120, "novelty": novelty, "summary": summary, "quote": quote[:300]}


def insider_card(item: dict) -> dict | None:
    """Open-market buys (P) positive; sales (S) weak negative unless large ($25M+ or 25%+ of the stake).
    Awards, exercises, tax withholding and gifts carry no card."""
    r, tk = item.get("raw") or {}, _ticker(item)
    typ = str(r.get("transaction_type") or "").upper()
    if tk is None or typ not in ("P", "S"):
        return None
    qty, px, owned = _num(r.get("securities_transacted")), _num(r.get("transaction_price")), _num(r.get("securities_owned"))
    notional = qty * px if qty and px else 0.0
    who = f"{r.get('owner_name') or 'insider'} ({r.get('owner_title') or r.get('ownership_type') or 'insider'})"
    if typ == "P":
        stance = 2 if notional >= 1e6 else 1
        strength = 0.4 + (0.3 if notional >= 1e6 else 0.1 if notional >= 1e5 else 0.0)
        verb = "bought"
    else:
        frac = qty / (qty + (owned or 0)) if qty else 0.0
        large_n, large_f = notional >= 25e6, frac >= 0.25
        stance = -2 if large_n and large_f else -1
        strength = 0.5 if (large_n or large_f) else 0.1
        verb = "sold"
    summary = f"{who} {verb} {qty or 0:,.0f} {tk} shares at {px or 0:,.2f} (${notional / 1e6:,.1f}M)"[:200]
    quote = f"{r.get('filing_date', '')} {typ} {qty or 0:g} @ {px or 0:g}".strip()
    return {"tickers": [tk], "scope": "name", "sector": "", "stance": stance, "strength": round(strength, 3),
            "horizon_h": 168, "novelty": "new", "summary": summary, "quote": quote}


# ---------------------------------------------------------------- persistence + public API

def load_cards() -> pd.DataFrame:
    if not CARDS_FILE.exists():
        return pd.DataFrame(columns=COLS)
    d = pd.read_parquet(CARDS_FILE)
    d["tickers"] = d["tickers"].map(lambda x: [str(t) for t in x] if x is not None else [])
    return d


def _save(d: pd.DataFrame) -> None:
    CARDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CARDS_FILE.parent, suffix=".parquet.tmp")
    os.close(fd)
    d[COLS].reset_index(drop=True).to_parquet(tmp, index=False)
    os.replace(tmp, CARDS_FILE)


def _rows(item: dict, cards: list[dict], version: str) -> list[dict]:
    at = pd.Timestamp(item["available_at"])
    at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
    return [{"card_id": f"{item['id'][:10]}-{i}", "item_id": item["id"], "kind": item["kind"], "available_at": at,
             **c, "extractor": version} for i, c in enumerate(cards)]


def _to_card(row: dict) -> dict:
    c = {k: row[k] for k in COLS if k != "extractor"}
    c["tickers"], c["stance"], c["horizon_h"] = list(c["tickers"]), int(c["stance"]), int(c["horizon_h"])
    c["strength"] = float(c["strength"])
    c["available_at"] = pd.Timestamp(c["available_at"]).tz_convert("UTC")
    return c


def cards_for(items: list[dict], workers: int = 4, save_every: int = 25, errors: list | None = None) -> list[dict]:
    """Cards for these items (contract Card + `card_id`, `sector`), sorted by available_at.
    News uses the LLM (cached); items already in cards.parquet under the current extractor are reused.
    A news item that cannot be extracted raises CacheMiss / LLMError after the rest is saved; with an
    `errors` list (the live loop) LLMErrors are appended there instead and the other cards returned."""
    versions = {"news": news_version(), "rating": RATING_VERSION, "insider": INSIDER_VERSION}
    store = load_cards()
    done = set(zip(store["item_id"], store["extractor"])) if len(store) else set()
    todo = [it for it in items if (it["id"], versions.get(it["kind"])) not in done]
    new_rows: list[dict] = []
    done_ids: set[str] = set()

    def flush() -> None:
        nonlocal store
        if not done_ids:
            return
        keep = store[~store["item_id"].isin(done_ids)]
        new = pd.DataFrame(new_rows, columns=COLS)
        store = new if not len(keep) else keep if not len(new) else pd.concat([keep, new], ignore_index=True)
        _save(store)

    for it in todo:
        if it["kind"] in ("rating", "insider"):
            c = (rating_card if it["kind"] == "rating" else insider_card)(it)
            new_rows += _rows(it, [c] if c else [], versions[it["kind"]])
            done_ids.add(it["id"])
    news = [it for it in todo if it["kind"] == "news"]
    failed: list[Exception] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for n, (it, cards) in enumerate(zip(news, ex.map(_news_cards, news)), 1):
                if isinstance(cards, Exception):         # not marked done: retried on the next call
                    failed.append(cards)
                    print(f"evidence: {it['id'][:10]} {it['title'][:60]!r}: {cards}", file=sys.stderr)
                    continue
                new_rows += _rows(it, cards, versions["news"])
                done_ids.add(it["id"])
                if n % save_every == 0:
                    flush()
    finally:
        flush()
    misses = [e for e in failed if isinstance(e, llm.CacheMiss)]
    if misses:
        raise llm.CacheMiss(f"{len(misses)} news item(s) not in the LLM cache; first: {misses[0]}")
    if failed and errors is None:
        raise llm.LLMError(f"{len(failed)} news item(s) failed extraction (rerun to retry); first: {failed[0]}")
    if errors is not None:
        errors.extend(failed)
    wanted = {it["id"] for it in items}
    sel = store[store["item_id"].isin(wanted) & store["extractor"].isin(set(versions.values()))] if len(store) else store
    return sorted((_to_card(r) for r in sel.to_dict("records")), key=lambda c: (c["available_at"], c["card_id"]))


# ---------------------------------------------------------------- demo

def news_items(path=None) -> list[dict]:
    """Stand-in for sentiment.data.Store.all_items() news items (same Item shape)."""
    d = pd.read_parquet(path or RAW / "news" / "news.parquet")
    out = []
    for r in d.to_dict("records"):
        at = pd.Timestamp(r["published_at"]).tz_convert("UTC")
        iid = hashlib.sha1(f"news|{at.isoformat()}|{r['title']}".encode()).hexdigest()
        out.append({"id": iid, "kind": "news", "available_at": at, "title": r["title"], "text": r["content"] or "",
                    "tickers": None, "raw": r})
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    items = news_items()[:n]
    for c in cards_for(items):
        print(f"{c['available_at']:%m-%d %H:%M} {c['scope']:6} {','.join(c['tickers']) or c['sector'] or '-':8} "
              f"{c['stance']:+d} {c['strength']:.2f} {c['horizon_h']:>4}h {c['novelty']:8} {c['summary']}")
    print(llm.usage_summary())


if __name__ == "__main__":
    main()
