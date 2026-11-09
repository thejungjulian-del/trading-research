"""The TJR 4-step model as a causal state machine over 5-minute bars.

sweep -> confirmation (BOS/IFVG) -> pullback into FVG/equilibrium -> trigger.
Emits Setup objects; execution and costs live in backtest.py so the same
fill machinery can be reused by the random-entry control.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

import detectors as D


@dataclass
class Params:
    swing_n: int = 2              # fractal width = "significance" of a high/low
    use_session_levels: bool = True   # also sweep PDH/PDL and premarket H/L
    confirm_window: int = 12      # bars allowed between sweep and BOS/IFVG
    pullback_window: int = 12     # bars allowed between confirmation and zone touch
    trigger_window: int = 6       # bars allowed for the trigger after zone touch
    stop_buffer: float = 0.0      # extra distance beyond the swept extreme
    target_mode: str = "R"        # "R" (fixed multiple) or "liquidity"
    target_R: float = 2.0
    min_stop: float = 0.01        # reject degenerate setups
    same_session: bool = True     # setups must complete intraday; False for daily bars
    zone_mode: str = "fvg_eq"     # "fvg_eq" | "order_block" | "both"
    ob_atr: float = 1.5           # impulse size that defines an order block


@dataclass
class Setup:
    i_entry: int                  # bar index of the TRIGGER; fill happens at i_entry+1 open
    side: int                     # +1 long, -1 short
    stop: float
    target: float
    sweep_price: float
    zone: str                     # "fvg" | "eq"


def _levels(df, sess, p):
    """Per-bar arrays of the currently-known significant high and low to sweep."""
    sw = D.swings(df, p.swing_n)
    hi = np.full(len(df), np.nan)
    lo = np.full(len(df), np.nan)
    cur_h = cur_l = np.nan
    by_confirm = {}
    for s in sw:
        by_confirm.setdefault(s.confirm_i, []).append(s)
    for i in range(len(df)):
        for s in by_confirm.get(i, []):
            if s.kind == "high":
                cur_h = s.price if np.isnan(cur_h) else max(cur_h, s.price)
            else:
                cur_l = s.price if np.isnan(cur_l) else min(cur_l, s.price)
        hi[i], lo[i] = cur_h, cur_l
        # a swept level stops being the reference
        if not np.isnan(cur_h) and df["high"].iat[i] > cur_h:
            cur_h = np.nan
        if not np.isnan(cur_l) and df["low"].iat[i] < cur_l:
            cur_l = np.nan
    out = pd.DataFrame({"sig_h": hi, "sig_l": lo}, index=df.index)
    if p.use_session_levels:
        # Session levels must be CONSUMED once swept, exactly like swing levels.
        # Without this, a trending day that closes above PDH re-registers a
        # "sweep" on every subsequent bar and floods the sample with
        # counter-trend setups in whichever instrument trended hardest.
        h_arr, l_arr = df["high"].to_numpy(), df["low"].to_numpy()
        day = df.index.normalize().to_numpy()
        for col, arr, agg, cmp in (("pdh", h_arr, np.fmax, np.greater),
                                   ("pmh", h_arr, np.fmax, np.greater),
                                   ("pdl", l_arr, np.fmin, np.less),
                                   ("pml", l_arr, np.fmin, np.less)):
            lvl = sess[col].to_numpy().copy()
            live = np.full(len(df), np.nan)
            cur_day, spent = None, False
            for i in range(len(df)):
                if day[i] != cur_day:
                    cur_day, spent = day[i], False    # fresh level each session
                cur = np.nan if spent else lvl[i]     # lvl is NaN until knowable
                live[i] = cur
                if not np.isnan(cur) and cmp(arr[i], cur):
                    spent = True                      # swept once -> done for the day
            tgt = "sig_h" if col in ("pdh", "pmh") else "sig_l"
            out[tgt] = agg(out[tgt].to_numpy(), live)
    return out


# Detector results depend only on (frame, swing_n) - or on the frame alone - so a
# sweep over 300+ parameter cells can reuse them.
#
# NB: the key is derived from frame CONTENT, not id(df). CPython reuses memory
# addresses after garbage collection, so an id()-keyed cache silently hands one
# frame's detectors to a different frame - a bug that fabricates results.
_CACHE: dict = {}


def _fid(df: pd.DataFrame):
    return (len(df), df.index[0].value, df.index[-1].value,
            float(df["close"].iat[0]), float(df["close"].iat[-1]))


def _ctx(df: pd.DataFrame, p: Params):
    f = _fid(df)
    k = (f, p.swing_n, p.use_session_levels, p.ob_atr)
    if k not in _CACHE:
        base = (f, "base")
        if base not in _CACHE:
            _CACHE[base] = (D.session_levels(df), D.inverse_fvgs(df), D.fvgs(df))
        sess, iv, gaps = _CACHE[base]
        obk = (f, "ob", p.ob_atr)
        if obk not in _CACHE:
            _CACHE[obk] = D.order_blocks(df, impulse_atr=p.ob_atr)
        _CACHE[k] = (sess, _levels(df, sess, p), D.structure_breaks(df, p.swing_n),
                     iv, gaps, _CACHE[obk])
    return _CACHE[k]


def find_setups(df: pd.DataFrame, p: Params) -> list[Setup]:
    sess, lv, st, iv, gaps, obs = _ctx(df, p)
    gaps_by_bar = {}
    for g in gaps:
        gaps_by_bar.setdefault(g.i, []).append(g)

    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    bos, ifvg = st["bos"].to_numpy(), iv["ifvg"].to_numpy()
    sig_h, sig_l = lv["sig_h"].to_numpy(), lv["sig_l"].to_numpy()
    # On daily bars there is no intraday session, so the same-day constraint
    # would make every setup impossible. Collapse it to a single session.
    day = (df.index.normalize().to_numpy() if p.same_session
           else np.zeros(len(df), dtype=np.int64))

    setups: list[Setup] = []
    i = 0
    n = len(df)
    while i < n - 1:
        # ---- step 1: sweep
        side = 0
        if not np.isnan(sig_h[i]) and h[i] > sig_h[i]:
            side, sweep = -1, h[i]
        elif not np.isnan(sig_l[i]) and l[i] < sig_l[i]:
            side, sweep = 1, l[i]
        if side == 0:
            i += 1
            continue

        # ---- step 2: change in order flow, same day
        j = i + 1
        conf = -1
        while j < n and j - i <= p.confirm_window and day[j] == day[i]:
            if (side == -1 and (bos[j] == -1 or ifvg[j] == -1)) or \
               (side == 1 and (bos[j] == 1 or ifvg[j] == 1)):
                conf = j
                break
            if (side == -1 and h[j] > sweep) or (side == 1 and l[j] < sweep):
                break                                   # sweep extended -> stale
            j += 1
        if conf < 0:
            i += 1
            continue

        # impulse leg = sweep extreme -> extreme reached at confirmation
        leg_far = l[i:conf + 1].min() if side == -1 else h[i:conf + 1].max()
        eq = D.equilibrium(min(sweep, leg_far), max(sweep, leg_far))
        # zones price can retrace into, per the chosen definition
        leg_gaps = []
        if p.zone_mode in ("fvg_eq", "both"):
            leg_gaps += [g for g in
                         (gg for b in range(i, conf + 1) for gg in gaps_by_bar.get(b, []))
                         if (g.bull is False) == (side == -1)]
        leg_obs = []
        if p.zone_mode in ("order_block", "both"):
            # "mark the candle that started the strong move". The origin candle
            # may sit just before the sweep, so the window opens 2 bars early.
            # Causality is enforced at use time: a block counts only once its
            # defining impulse has completed (confirm_i <= the bar being tested).
            leg_obs = [b for b in obs
                       if i - 2 <= b.i <= conf and (b.bull is False) == (side == -1)]
        use_eq = p.zone_mode in ("fvg_eq", "both")

        # ---- step 3 + 4: pullback into zone, then trigger out of it
        k = conf + 1
        touched = -1
        zone = ""
        while k < n and k - conf <= p.pullback_window and day[k] == day[i]:
            if (side == -1 and h[k] > sweep) or (side == 1 and l[k] < sweep):
                break                                   # invalidated
            hit_g = any((l[k] <= g.hi and h[k] >= g.lo) for g in leg_gaps) or \
                    any((l[k] <= b.hi and h[k] >= b.lo) for b in leg_obs
                        if b.confirm_i <= k)
            hit_eq = use_eq and ((h[k] >= eq) if side == -1 else (l[k] <= eq))
            if hit_g or hit_eq:
                touched, zone = k, ("zone" if hit_g else "eq")
                break
            k += 1
        if touched < 0:
            i = conf + 1
            continue

        t = touched
        trig = -1
        while t < n - 1 and t - touched <= p.trigger_window and day[t] == day[i]:
            if (side == -1 and h[t] > sweep) or (side == 1 and l[t] < sweep):
                break
            if (side == -1 and c[t] < o[t]) or (side == 1 and c[t] > o[t]):
                trig = t
                break
            t += 1
        if trig < 0:
            i = conf + 1
            continue

        stop = sweep + side * -1 * p.stop_buffer
        ref = c[trig]
        risk = abs(ref - stop)
        if risk < p.min_stop:
            i = conf + 1
            continue
        if p.target_mode == "R":
            target = ref + side * p.target_R * risk
        else:
            draw = sess["pdl"].iat[trig] if side == -1 else sess["pdh"].iat[trig]
            if np.isnan(draw) or (side == -1 and draw >= ref) or (side == 1 and draw <= ref):
                target = ref + side * p.target_R * risk
            else:
                target = float(draw)
        setups.append(Setup(trig, side, float(stop), float(target), float(sweep), zone))
        i = trig + 1
    return setups
