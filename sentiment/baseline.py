"""Fixed-rule baseline policy, no LLM (contract: sentiment/CONTRACT.md, Policies).

Per ticker: score = sum of stance * strength * exp(-age_h / 24) over the name-scoped cards on it,
plus 0.5 * the same sum over market-scoped cards (applied to every name); sector cards and
`post_hoc` cards are skipped. Weight = clip(0.05 * score, +-MAX_W_NAME). The risk layer applies the
remaining limits. Only cards available by t count (the runner passes those; later ones are ignored).
"""
from __future__ import annotations

import math

import pandas as pd

from sentiment.config import MAX_W_NAME, UNIVERSE

K_WEIGHT = 0.05            # weight per unit of score
MARKET_MULT = 0.5          # market cards count half, on every name
DECAY_H = 24.0             # e-folding age of a card in hours


def _ts(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def card_ref(card: dict) -> str:
    return str(card.get("card_id") or card["item_id"])


def scores(t, cards: list[dict]) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Score per ticker and the ids of the cards that contributed to it."""
    t = _ts(t)
    sc = {tk: 0.0 for tk in UNIVERSE}
    used: dict[str, list[str]] = {tk: [] for tk in UNIVERSE}
    for c in cards:
        if c.get("novelty") == "post_hoc":
            continue
        age = (t - _ts(c["available_at"])).total_seconds() / 3600
        if age < 0:                                   # not yet available at t
            continue
        v = int(c["stance"]) * float(c["strength"]) * math.exp(-age / DECAY_H)
        if c.get("scope") == "name":
            names, mult = [tk for tk in c.get("tickers") or [] if tk in sc], 1.0
        elif c.get("scope") == "market":
            names, mult = list(UNIVERSE), MARKET_MULT
        else:
            continue
        for tk in names:
            sc[tk] += mult * v
            if v:
                used[tk].append(card_ref(c))
    return sc, used


def decide(t, state: dict, cards: list[dict], book: dict[str, float], variant: str = "baseline") -> dict:
    sc, used = scores(t, cards)
    targets, reasons, evidence = {}, {}, {}
    for tk in UNIVERSE:
        w = max(-MAX_W_NAME, min(MAX_W_NAME, K_WEIGHT * sc[tk]))
        if abs(w) < 1e-9:
            continue
        targets[tk] = round(w, 6)
        reasons[tk] = f"rule score {sc[tk]:+.3f} from {len(used[tk])} card(s)"
        evidence[tk] = used[tk]
    return {"targets": targets, "reasons": reasons, "evidence": evidence, "confidence": 1.0,
            "raw": "", "valid": True}
