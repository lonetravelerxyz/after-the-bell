"""Deterministic risk layer between the policy and the broker (CONTRACT: Risk).

`apply` enforces the config limits in a fixed order and logs every rule that binds:
invalid -> hold; per-name cap; off-hours scale; funding block; stop-loss/cooldown; daily kill;
[v1: beta neutralisation]; net cap; gross cap. Weights are fractions of equity (+ long, - short). A
ticker missing from the decision's targets means target 0 (the targets are the whole desired book).

Two versions, picked by `version` (default: config.RISK_VERSION, read at call time):
  v0  flat per-name cap MAX_W_NAME; flat stop STOP_LOSS_NAME; no beta step (every run before 2026-09-23).
  v1  - per-name cap_i = MAX_W_NAME * clip(median_vol / vol_i, VOL_CAP_FLOOR, 1), vol = state vol_7d (daily
        vol from hourly returns, point-in-time), median over the universe names with a vol at t. A name
        without a vol at t (data gap) keeps the flat cap. Off-hours scaling multiplies cap_i.
      - stop: a name is flat when its adverse move from the quantity-averaged entry reaches
        stop_i = clip(STOP_K * vol_i, STOP_FLOOR, STOP_CAP), with vol_i = vol_7d AT ENTRY (the decision
        that opened / grew the position; a top-up re-averages stop_i by quantity like the entry price).
        Entry-time vol is point-in-time, lets the hourly `check` work without market state, and keeps the
        stop from widening as the adverse move itself raises the vol. No vol at entry -> the universe
        median vol at entry; none either -> STOP_LOSS_NAME (also for entries recorded without a stop).
      - beta neutralisation (BETA_NEUTRAL, after the kill, before net/gross caps): the minimal-L2 change
        of the weights that makes beta . w = 0 (beta = state beta_60d, missing -> 1.0) while every name
        stays inside the per-name bounds the earlier rules left (cap_i, off-hours, funding block; stopped /
        cooldown / killed names and names without a price are fixed) AND |sum w| <= MAX_NET. Without binding
        bounds this is w' = w - beta * (beta . w) / (beta . beta); with them the same projection with the
        bounds applied (KKT: w_i = clip(w_i - lambda * beta_i - mu, lo_i, hi_i), lambda solved exactly for
        beta . w' = 0, mu = 0 unless the net cap binds, else solved by bisection for sum w' = +-MAX_NET).
        The net cap is enforced inside the projection because scaling one side down afterwards (step 8)
        breaks neutrality whenever the long and short sides have different betas. Logged in two stages,
        one action per changed name each: 'beta_neutral' (w -> the projection without the net cap), then
        'net_cap' (-> the projection with it) when the net cap binds. When the bounds make beta . w = 0
        unreachable, the closest reachable beta exposure is taken; when the net cap cannot be met on
        beta . w = 0 (only with fixed non-zero names), the projection without it is kept and step 8 scales
        the dominant side as in v0 (the net cap is a hard limit, neutrality is best effort). The gross cap
        (step 9) scales every name by one factor, so it keeps beta . w = 0 and |sum w| <= MAX_NET.

`ledger_state` is a dict the runner keeps across calls; `apply` creates and updates its keys:
  equity       current equity (set by the runner before the call; enables the daily kill)
  equity_hist  [(ts, equity), ...] of the last 24h (also fed hourly via `observe`)
  entry        {ticker: {"price", "side"[, "stop"]}} average entry price of each open position (stop-loss
               reference; decision price of the opening trade, re-averaged by quantity whenever the
               position grows on the same side; unchanged when it shrinks); v1 adds the stop threshold
  cooldown     {ticker: ts} name blocked (target 0) until ts after a stop-loss
  kill_until   ts, everything flat until then after the daily kill
Rules that only restrict new exposure (off-hours, funding block) never force a reduction of an
existing position on the same side; the hard rules (caps, stop, kill) do.

`check(t, book, prices, ledger_state)` runs the stop-loss and the daily kill between decisions: the
runners call it every hour after `observe`, so the stop and DAILY_KILL are checked on every hourly
mark, not only every 4h. It returns the forced book (stopped names at 0, or everything flat after a
kill) when a rule binds, else None.
"""
from __future__ import annotations

import math

import pandas as pd

from sentiment import config
from sentiment.config import (DAILY_KILL, FUNDING_BLOCK, KILL_COOLDOWN_H, MAX_GROSS, MAX_NET,
                              MAX_W_NAME, OFF_HOURS_SCALE, STOP_COOLDOWN_H, STOP_LOSS_NAME, UNIVERSE)

EPS = 1e-12
VERSIONS = ("v0", "v1")


def _version(version: str | None) -> str:
    v = version or config.RISK_VERSION
    if v not in VERSIONS:
        raise ValueError(f"unknown risk version {v!r}; one of {VERSIONS}")
    return v


def us_session_open(t: pd.Timestamp) -> bool:
    """US regular session, 13:30-20:00 UTC Mon-Fri (the config's approximation; no holidays)."""
    t = pd.Timestamp(t)
    minute = t.hour * 60 + t.minute
    return t.weekday() < 5 and 13 * 60 + 30 <= minute < 20 * 60


def observe(ledger_state: dict, t, equity: float) -> None:
    """Record an equity point for the 24h high-water mark (call every mark, not just decisions)."""
    t = pd.Timestamp(t)
    ledger_state["equity"] = float(equity)
    hist = ledger_state.setdefault("equity_hist", [])
    if not hist or pd.Timestamp(hist[-1][0]) < t:
        hist.append((t, float(equity)))
    cutoff = t - pd.Timedelta(hours=24)
    ledger_state["equity_hist"] = [(ts, e) for ts, e in hist if pd.Timestamp(ts) >= cutoff]


def _sign(x: float) -> int:
    return (x > EPS) - (x < -EPS)


def _num(x) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _vol(state: dict, tk: str) -> float | None:
    v = _num((state.get(tk) or {}).get("vol_7d")) if state else None
    return v if v is not None and v > 0 else None


def median_vol(state: dict) -> float | None:
    """Cross-sectional median of vol_7d over the universe names that have one at t."""
    vs = sorted(v for tk in UNIVERSE if (v := _vol(state, tk)) is not None)
    if not vs:
        return None
    n = len(vs)
    return vs[n // 2] if n % 2 else (vs[n // 2 - 1] + vs[n // 2]) / 2


def vol_caps(state: dict, names) -> dict[str, float]:
    """v1 per-name cap: MAX_W_NAME * clip(median_vol / vol_i, VOL_CAP_FLOOR, 1); flat cap without a vol."""
    med = median_vol(state)
    out = {}
    for tk in names:
        v = _vol(state, tk)
        k = 1.0 if med is None or v is None else max(config.VOL_CAP_FLOOR, min(1.0, med / v))
        out[tk] = MAX_W_NAME * k
    return out


def _session(t: pd.Timestamp, state: dict, names) -> bool:
    session = state.get(names[0], {}).get("us_session_open") if state else None
    return us_session_open(t) if session is None else bool(session)


def effective_caps(t, state: dict, version: str | None = None) -> dict[str, float]:
    """The per-name cap on NEW exposure `apply` would use at t for each universe name: the per-name cap
    (v1: vol-scaled) times OFF_HOURS_SCALE outside the US session. (A held position above it on the same
    side is kept up to the per-name cap off-hours; see `apply`.) For showing the policy its limits."""
    t, state = pd.Timestamp(t), state or {}
    names = sorted(UNIVERSE)
    caps = vol_caps(state, names) if _version(version) == "v1" else {tk: MAX_W_NAME for tk in names}
    k = 1.0 if _session(t, state, names) else OFF_HOURS_SCALE
    return {tk: caps[tk] * k for tk in UNIVERSE}


def stop_level(state: dict, tk: str) -> float:
    """v1 stop threshold for a position entered at the state's t: clip(STOP_K * vol, STOP_FLOOR, STOP_CAP)."""
    v = _vol(state, tk)
    v = median_vol(state) if v is None else v
    if v is None:
        return STOP_LOSS_NAME
    return max(config.STOP_FLOOR, min(config.STOP_CAP, config.STOP_K * v))


def _stop_of(e: dict, version: str) -> float:
    return STOP_LOSS_NAME if version == "v0" else float(e.get("stop", STOP_LOSS_NAME))


def _neutral_at(w: dict[str, float], beta: dict[str, float], lo: dict[str, float], hi: dict[str, float],
                mu: float = 0.0) -> dict[str, float]:
    """w'_i = clip(w_i - mu - lambda * beta_i, lo_i, hi_i) over the movable names (priced: in beta; lo < hi),
    with lambda solving beta . w' = 0 exactly (g(lambda) = beta . w'(lambda) is piecewise linear and
    non-increasing: solved between its breakpoints), or the closest reachable exposure. mu = 0 is the plain
    beta projection."""
    movable = [tk for tk in w if tk in beta and hi[tk] - lo[tk] > EPS]
    free = [tk for tk in movable if abs(beta[tk]) > EPS]
    v = {tk: w[tk] - mu for tk in movable}

    def at(lam: float) -> dict[str, float]:
        out = dict(w)
        for tk in movable:
            out[tk] = max(lo[tk], min(hi[tk], v[tk] - lam * beta[tk]))
        return out

    if not free:
        return at(0.0)
    base = at(0.0)
    fixed = sum(beta[tk] * base[tk] for tk in w if tk in beta and tk not in free)

    def g(lam: float) -> float:
        x = at(lam)
        return fixed + sum(beta[tk] * x[tk] for tk in free)

    bb = sum(beta[tk] ** 2 for tk in free)
    lam = (fixed + sum(beta[tk] * v[tk] for tk in free)) / bb          # unconstrained projection
    if all(lo[tk] - EPS <= v[tk] - lam * beta[tk] <= hi[tk] + EPS for tk in free):
        return at(lam)
    bps = sorted({(v[tk] - b) / beta[tk] for tk in free for b in (lo[tk], hi[tk])})
    gs = [g(x) for x in bps]
    if gs[0] <= 0:                                  # even the largest reachable exposure is <= 0
        return at(bps[0])
    if gs[-1] >= 0:
        return at(bps[-1])
    for a, b, ga, gb in zip(bps, bps[1:], gs, gs[1:]):
        if ga >= 0 >= gb:
            lam = a if ga == gb else a + (b - a) * ga / (ga - gb)
            return at(lam)
    return at(0.0)                                  # unreachable (g is monotone)


def beta_neutral(w: dict[str, float], beta: dict[str, float], lo: dict[str, float],
                 hi: dict[str, float], max_net: float | None = None) -> dict[str, float]:
    """argmin ||w' - w||_2 s.t. beta . w' = 0, lo <= w' <= hi and, with `max_net`, |sum w'| <= max_net (w
    inside the bounds; names absent from beta, or with lo == hi, stay fixed). KKT: w'_i = clip(w_i - lambda
    * beta_i - mu, lo_i, hi_i). Without a binding net cap mu = 0 and lambda is exact (`_neutral_at`). When
    the plain projection's |net| exceeds max_net the net constraint is active at the optimum, sum w' =
    s * max_net: net(mu) (lambda re-solved at each mu) is non-increasing in mu (the derivative of the
    concave dual), so mu is found by bisection and the side with |net| <= max_net is returned. When no mu
    reaches it (fixed non-zero names), the plain projection is returned."""
    out = _neutral_at(w, beta, lo, hi)
    if max_net is None:
        return out
    net = sum(out.values())
    if abs(net) <= max_net + EPS:
        return out
    s = 1.0 if net > 0 else -1.0

    def ok(mu: float) -> bool:
        return s * sum(_neutral_at(w, beta, lo, hi, mu).values()) <= max_net

    a, b, step = 0.0, None, 1.0
    for _ in range(64):                             # bracket: mu moves the net towards the cap
        if ok(s * step):
            b = s * step
            break
        a, step = s * step, step * 2
    if b is None:
        return out
    for _ in range(200):                            # bisection to float resolution; b always satisfies the cap
        mid = (a + b) / 2
        if mid in (a, b):
            break
        if ok(mid):
            b = mid
        else:
            a = mid
    return _neutral_at(w, beta, lo, hi, b)


def _valid(decision: dict) -> bool:
    if not isinstance(decision, dict) or not decision.get("valid", False):
        return False
    tg = decision.get("targets")
    if not isinstance(tg, dict):
        return False
    try:
        return all(math.isfinite(float(v)) for v in tg.values())
    except (TypeError, ValueError):
        return False


def apply(t, decision: dict, book: dict[str, float], state: dict,
          ledger_state: dict, version: str | None = None) -> tuple[dict[str, float], list[dict]]:
    t = pd.Timestamp(t)
    v = _version(version)
    actions: list[dict] = []
    book = {k: float(x) for k, x in book.items()}
    names = sorted(set(UNIVERSE) | set(book))
    state = state or {}

    def bind(rule: str, tk: str | None, before, after) -> None:
        actions.append({"rule": rule, "ticker": tk, "before": before, "after": after})

    def set_w(rule: str, tk: str, new: float) -> None:
        if abs(new - w[tk]) > EPS:
            bind(rule, tk, w[tk], new)
            w[tk] = new

    # 1. invalid -> hold the current book
    if _valid(decision):
        raw = {k: float(x) for k, x in decision["targets"].items()}
        for tk in sorted(set(raw) - set(names)):
            bind("universe", tk, raw[tk], 0.0)
        w = {tk: raw.get(tk, 0.0) for tk in names}
    else:
        bind("invalid", None, (decision or {}).get("targets") if isinstance(decision, dict) else None,
             dict(book))
        w = {tk: book.get(tk, 0.0) for tk in names}

    # 2. per-name cap (v1: vol-scaled). lo/hi track each name's bounds for the beta step.
    caps = vol_caps(state, names) if v == "v1" else {tk: MAX_W_NAME for tk in names}
    lo = {tk: -caps[tk] for tk in names}
    hi = {tk: caps[tk] for tk in names}
    for tk in names:
        set_w("name_cap", tk, max(-caps[tk], min(caps[tk], w[tk])))

    # 3. off-hours: the per-name cap is scaled for new exposure
    if not _session(t, state, names):
        for tk in names:
            cap, b = caps[tk] * OFF_HOURS_SCALE, book.get(tk, 0.0)
            lim = max(cap, abs(b)) if _sign(b) == _sign(w[tk]) else cap
            lim = min(lim, caps[tk])
            set_w("off_hours", tk, max(-lim, min(lim, w[tk])))
            hi[tk] = min(hi[tk], max(cap, b))
            lo[tk] = max(lo[tk], min(-cap, b))

    # 4. funding block: no new exposure on the paying side
    for tk in names:
        rate = (state.get(tk) or {}).get("funding_last")
        if rate is None or pd.isna(rate) or abs(rate) < FUNDING_BLOCK:
            continue
        b = book.get(tk, 0.0)
        if rate > 0:                                      # longs pay
            if w[tk] > 0:
                set_w("funding_block", tk, min(w[tk], max(b, 0.0)))
            hi[tk] = min(hi[tk], max(b, 0.0))
        else:                                             # shorts pay
            if w[tk] < 0:
                set_w("funding_block", tk, max(w[tk], min(b, 0.0)))
            lo[tk] = max(lo[tk], min(b, 0.0))

    # 5. stop-loss on each open name, then cooldown
    entry = ledger_state.setdefault("entry", {})
    cooldown = ledger_state.setdefault("cooldown", {})
    for tk in names:
        b = book.get(tk, 0.0)
        px = (state.get(tk) or {}).get("price")
        e = entry.get(tk)
        if _sign(b) and e and px and _sign(b) == e["side"]:
            loss = e["side"] * (px / e["price"] - 1)
            if loss <= -_stop_of(e, v):
                cooldown[tk] = t + pd.Timedelta(hours=STOP_COOLDOWN_H)
                bind("stop_loss", tk, w[tk], 0.0)
                w[tk] = lo[tk] = hi[tk] = 0.0
                continue
        until = cooldown.get(tk)
        if until is not None:
            if t < pd.Timestamp(until):
                set_w("cooldown", tk, 0.0)
                lo[tk] = hi[tk] = 0.0
            else:
                cooldown.pop(tk)

    # 6. daily kill: equity down DAILY_KILL from its 24h high -> flat everything for 24h
    if _kill_active(t, ledger_state, bind):
        for tk in names:
            set_w("kill", tk, 0.0)
            lo[tk] = hi[tk] = 0.0

    # 7. v1 beta neutralisation (bounds from steps 2-6; names without a price cannot trade -> fixed)
    if v == "v1" and config.BETA_NEUTRAL:
        beta = {}
        for tk in names:
            s = state.get(tk) or {}
            if tk in UNIVERSE and _num(s.get("price")):
                b = _num(s.get("beta_60d"))
                beta[tk] = 1.0 if b is None else b
            else:
                lo[tk] = hi[tk] = w[tk]
        for tk in names:                                  # w is inside its bounds (float guard)
            lo[tk], hi[tk] = min(lo[tk], w[tk]), max(hi[tk], w[tk])
        w0 = dict(w)
        plain = beta_neutral(w0, beta, lo, hi)
        for tk in names:
            set_w("beta_neutral", tk, plain[tk])
        joint = beta_neutral(w0, beta, lo, hi, max_net=MAX_NET)   # the net cap inside the projection
        for tk in names:
            set_w("net_cap", tk, joint[tk])

    # 8. net cap: scale the dominant side down (v1: binds only when step 7 could not meet it)
    net = sum(w.values())
    if abs(net) > MAX_NET + EPS:
        sg = _sign(net)
        dom = sum(x for x in w.values() if _sign(x) == sg) * sg
        other = sum(x for x in w.values() if _sign(x) == -sg) * -sg
        k = max(0.0, (MAX_NET + other) / dom)
        for tk in names:
            if _sign(w[tk]) == sg:
                set_w("net_cap", tk, w[tk] * k)

    # 9. gross cap: proportional scale-down
    gross = sum(abs(x) for x in w.values())
    if gross > MAX_GROSS + EPS:
        k = MAX_GROSS / gross
        for tk in names:
            set_w("gross_cap", tk, w[tk] * k)

    # entry prices (v1: and stop thresholds) for the resulting book
    for tk in names:
        _update_entry(entry, tk, book.get(tk, 0.0), w[tk], (state.get(tk) or {}).get("price"),
                      stop_level(state, tk) if v == "v1" else None)

    return {tk: x for tk, x in w.items() if abs(x) > EPS}, actions


def _update_entry(entry: dict, tk: str, b: float, w: float, px, stop: float | None = None) -> None:
    """Opening or flipping sets the entry to px; growing on the same side re-averages it by quantity
    (quantities are proportional to weight / price at the common price px); shrinking keeps it.
    `stop` (v1) is the stop threshold at this t; it is set and re-averaged by quantity like the price
    (an entry without one, e.g. opened under v0, takes this one's when it grows)."""
    s, sb = _sign(w), _sign(b)
    if s == 0:
        entry.pop(tk, None)
    elif not px:
        return
    elif s != sb or tk not in entry:
        entry[tk] = {"price": float(px), "side": s}
        if stop is not None:
            entry[tk]["stop"] = float(stop)
    elif abs(w) > abs(b) + EPS:
        q_old, q_add = abs(b) / px, (abs(w) - abs(b)) / px
        e = entry[tk]
        new = {"price": (q_old * e["price"] + q_add * px) / (q_old + q_add), "side": s}
        if stop is not None:
            old_stop = float(e.get("stop", stop))
            new["stop"] = (q_old * old_stop + q_add * stop) / (q_old + q_add)
        entry[tk] = new


def _kill_active(t: pd.Timestamp, ledger_state: dict, bind) -> bool:
    """Trigger the daily kill when equity is DAILY_KILL below its 24h high; expire it after
    KILL_COOLDOWN_H. True while the kill is in force."""
    if "equity" in ledger_state:
        observe(ledger_state, t, ledger_state["equity"])
        hw = max(e for _, e in ledger_state["equity_hist"])
        if ledger_state["equity"] <= hw * (1 - DAILY_KILL) and not ledger_state.get("kill_until"):
            ledger_state["kill_until"] = t + pd.Timedelta(hours=KILL_COOLDOWN_H)
            bind("daily_kill", None, hw, ledger_state["equity"])
    ku = ledger_state.get("kill_until")
    if ku is None:
        return False
    if t < pd.Timestamp(ku):
        return True
    ledger_state["kill_until"] = None
    ledger_state["equity_hist"] = [(t, ledger_state["equity"])] if "equity" in ledger_state else []
    return False


def check(t, book: dict[str, float], prices: dict[str, float],
          ledger_state: dict, version: str | None = None) -> tuple[dict[str, float] | None, list[dict]]:
    """Hourly stop-loss and daily kill between decisions (call after `observe` with the hour's equity).
    `book` = current weights, `prices` = current mark per ticker. Returns (forced targets, actions):
    the book with stopped names at 0 (or {} while the kill is in force), or None when nothing binds."""
    t = pd.Timestamp(t)
    ver = _version(version)
    actions: list[dict] = []

    def bind(rule: str, tk: str | None, before, after) -> None:
        actions.append({"rule": rule, "ticker": tk, "before": before, "after": after})

    w = {tk: float(v) for tk, v in book.items() if abs(v) > EPS}
    if _kill_active(t, ledger_state, bind):
        for tk in sorted(w):
            bind("kill", tk, w[tk], 0.0)
        return ({}, actions) if actions else (None, actions)
    entry = ledger_state.setdefault("entry", {})
    cooldown = ledger_state.setdefault("cooldown", {})
    for tk in sorted(w):
        e, px = entry.get(tk), prices.get(tk)
        if e and px and _sign(w[tk]) == e["side"] and e["side"] * (px / e["price"] - 1) <= -_stop_of(e, ver):
            cooldown[tk] = t + pd.Timedelta(hours=STOP_COOLDOWN_H)
            entry.pop(tk)
            bind("stop_loss", tk, w[tk], 0.0)
            w[tk] = 0.0
    if not actions:
        return None, actions
    return {tk: v for tk, v in w.items() if abs(v) > EPS}, actions
