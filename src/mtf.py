"""Multi-timeframe version of the model, rebuilt to match what the videos teach.

Fixes three things the first implementation got wrong:
  1. ZONES COME FROM A HIGHER TIMEFRAME than the structure break. TradingLab
     says "go to the four hour time frame" to mark supply and demand; TJR runs
     HTF bias -> 5m structure -> LTF entry. v1 marked zones on 5m, so a "supply
     zone" was a 15-minute gap rather than a region where size plausibly traded.
  2. LEVELS THAT HOLD LIQUIDITY. v1 swept 25-55 minute fractals - nobody's stop
     is there. Now: prior day / prior week / 20-day extremes / round numbers.
  3. VOLATILE SINGLE STOCKS, not index ETFs.

Causality rule for the timeframe join: a higher-timeframe zone is usable only
after the HTF bar that defines it has CLOSED. Violating this is the classic
repaint bug and it flatters backtests enormously.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

import detectors as D
import trend as TR


@dataclass
class MtfParams:
    level_kinds: tuple = ("day", "week", "n20")
    zone_tf: str = "1h"
    zone_mode: str = "fvg"        # fvg | order_block | bpr | breaker | all
    smt_filter: bool = False      # require SMT divergence confirming the reversal
    smt_lookback: int = 20
    htf_bias: str = "off"         # off | align | counter
    futures: bool = False         # CME trade dates + true Asia/London sessions
    ob_atr: float = 1.5
    struct_swing_n: int = 3       # swing width for the 5m break of structure
    confirm_window: int = 12      # 5m bars: sweep -> structure break
    pullback_window: int = 24     # 5m bars: structure break -> zone touch
    trigger_window: int = 6       # 5m bars: zone touch -> entry trigger
    target_mode: str = "R"        # "R" | "level"
    target_R: float = 2.0
    same_session: bool = True
    min_stop_bp: float = 5.0      # reject degenerate stops (in bp of price)


@dataclass
class Zone:
    usable_from: pd.Timestamp     # HTF bar close - not before
    lo: float
    hi: float
    bull: bool


def htf_zones(df5: pd.DataFrame, p: MtfParams) -> list[Zone]:
    """Zones marked on the higher timeframe, each stamped with the moment it
    becomes knowable."""
    htf = df5.resample(p.zone_tf, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()
    if len(htf) < 30:
        return []
    step = pd.Timedelta(p.zone_tf)
    out: list[Zone] = []
    if p.zone_mode in ("fvg", "both", "all"):
        for g in D.fvgs(htf):
            # a 3-bar gap is known once bar 3 closes
            out.append(Zone(htf.index[g.i] + step, g.lo, g.hi, g.bull))
    if p.zone_mode in ("order_block", "both", "all"):
        for b in D.order_blocks(htf, impulse_atr=p.ob_atr):
            # known once the impulse that defines it has closed
            if b.confirm_i < len(htf):
                out.append(Zone(htf.index[b.confirm_i] + step, b.lo, b.hi, b.bull))
    if p.zone_mode in ("bpr", "all"):
        for g in D.bpr(htf):
            out.append(Zone(htf.index[g.i] + step, g.lo, g.hi, g.bull))
    if p.zone_mode in ("breaker", "all"):
        for g in D.breakers(htf, p.struct_swing_n):
            out.append(Zone(htf.index[g.i] + step, g.lo, g.hi, g.bull))
    return sorted(out, key=lambda z: z.usable_from)


@dataclass
class Setup:
    i_entry: int
    side: int
    stop: float
    target: float
    sweep_price: float
    level: float


# Keyed on frame CONTENT, never id(df) - CPython reuses addresses after GC and
# an id()-keyed cache silently hands one frame's detectors to another.
_C: dict = {}


def _fid(df):
    return (len(df), df.index[0].value, df.index[-1].value,
            float(df["close"].iat[0]), float(df["close"].iat[-1]))


def _cached(key, fn):
    if key not in _C:
        _C[key] = fn()
    return _C[key]


def htf_bias(df5: pd.DataFrame, p: MtfParams) -> np.ndarray:
    """Higher-timeframe order flow, mapped down to the 5m index.
    TJR's step zero: establish HTF direction before looking for anything."""
    htf = df5.resample(p.zone_tf, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()
    if len(htf) < 30:
        return np.zeros(len(df5), dtype=np.int8)
    st = TR.trend_state(htf, p.struct_swing_n)
    # shift by one HTF bar so a 5m bar never sees the HTF bar it is inside
    st = st.shift(1).reindex(df5.index, method="ffill").fillna(0)
    return st.to_numpy().astype(np.int8)


def find_setups(df5: pd.DataFrame, p: MtfParams,
                ref: pd.DataFrame | None = None) -> list[Setup]:
    f = _fid(df5)
    sw = _cached((f, "sw", p.level_kinds, p.futures),
                 lambda: D.liquidity_sweeps(df5, p.level_kinds, p.futures))
    lvl = _cached((f, "lvl", p.level_kinds, p.futures),
                  lambda: D.liquidity_levels(df5, p.level_kinds, p.futures))
    lvl_hi = lvl[[c for c in lvl.columns if c.endswith("h") and not c.endswith("_l")]].to_numpy()
    lvl_lo = lvl[[c for c in lvl.columns if c.endswith("l") and not c.endswith("_h")]].to_numpy()
    st = _cached((f, "st", p.struct_swing_n), lambda: D.structure_breaks(df5, p.struct_swing_n))
    iv = _cached((f, "iv"), lambda: D.inverse_fvgs(df5))
    zones = _cached((f, "z", p.zone_tf, p.zone_mode, p.ob_atr, p.struct_swing_n),
                    lambda: htf_zones(df5, p))
    smt = None
    if p.smt_filter and ref is None:
        # Silent no-op filters are how you end up believing a result was
        # filtered when it was not. Fail loudly instead.
        raise ValueError(
            "smt_filter=True requires a reference instrument (`ref`). "
            "TJR's SMT pair is NQ vs ES; passing None silently disables the filter.")
    if p.smt_filter and ref is not None:
        smt = _cached((f, "smt", p.smt_lookback), lambda: D.smt_divergence(
            df5, ref, p.smt_lookback)["smt"].reindex(df5.index).fillna(0)
            .to_numpy().astype(np.int8))
    bias = (_cached((f, "bias", p.zone_tf, p.struct_swing_n),
                    lambda: htf_bias(df5, p)) if p.htf_bias != "off" else None)

    o, h, l, c = (df5[k].to_numpy() for k in ("open", "high", "low", "close"))
    sh, sl = sw["swept_h"].to_numpy(), sw["swept_l"].to_numpy()
    bos, ifvg = st["bos"].to_numpy(), iv["ifvg"].to_numpy()
    ts = df5.index
    if not p.same_session:
        day = np.zeros(len(df5), dtype=np.int64)
    elif p.futures:
        day = (ts + pd.Timedelta(hours=6)).normalize().to_numpy()
    else:
        day = ts.normalize().to_numpy()
    n = len(df5)

    # zone lookup: for bar k, only zones with usable_from <= ts[k]
    z_times = np.array([z.usable_from.value for z in zones]) if zones else np.array([])

    setups: list[Setup] = []
    i = 0
    while i < n - 1:
        side = 0
        if not np.isnan(sh[i]):
            side, sweep, level = -1, h[i], sh[i]
        elif not np.isnan(sl[i]):
            side, sweep, level = 1, l[i], sl[i]
        if side == 0:
            i += 1
            continue

        # step 0 - SMT divergence must confirm the reversal.
        # smt_divergence(stock, index) returns -1 when the stock takes a high the
        # index does not, i.e. bearish for the stock - which is `side` for a short.
        if smt is not None:
            w = smt[max(0, i - p.smt_lookback):i + 1]
            if not (w == side).any():
                i += 1
                continue

        # step 0b - higher-timeframe order flow
        if bias is not None and bias[i] != 0:
            if p.htf_bias == "align" and bias[i] != side:
                i += 1
                continue
            if p.htf_bias == "counter" and bias[i] == side:
                i += 1
                continue

        # step 2 - change in order flow on the 5m
        j, conf = i + 1, -1
        while j < n and j - i <= p.confirm_window and day[j] == day[i]:
            if (side == -1 and (bos[j] == -1 or ifvg[j] == -1)) or \
               (side == 1 and (bos[j] == 1 or ifvg[j] == 1)):
                conf = j
                break
            if (side == -1 and h[j] > sweep) or (side == 1 and l[j] < sweep):
                break
            j += 1
        if conf < 0:
            i += 1
            continue

        # step 3 - retrace into a HIGHER-TIMEFRAME zone sitting between the
        # confirmation price and the swept extreme (i.e. between entry and stop)
        ref = c[conf]
        lo_b, hi_b = (min(ref, sweep), max(ref, sweep))
        cand = [z for z in zones
                if (z.bull is False) == (side == -1)
                and z.lo >= lo_b and z.hi <= hi_b * 1.001]
        if not cand:
            i = conf + 1
            continue

        k, touched = conf + 1, -1
        while k < n and k - conf <= p.pullback_window and day[k] == day[i]:
            if (side == -1 and h[k] > sweep) or (side == 1 and l[k] < sweep):
                break
            tk = ts[k].value
            if any(z.usable_from.value <= tk and l[k] <= z.hi and h[k] >= z.lo
                   for z in cand):
                touched = k
                break
            k += 1
        if touched < 0:
            i = conf + 1
            continue

        # step 4 - confirmation out of the zone
        t, trig = touched, -1
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

        stop, ref = sweep, c[trig]
        risk = abs(ref - stop)
        if risk < p.min_stop_bp * 1e-4 * ref:
            i = conf + 1
            continue
        if p.target_mode == "R":
            target = ref + side * p.target_R * risk
        else:
            # "draw on liquidity" = the nearest level in the direction of travel
            # that has NOT yet been taken. For a short that is the closest low
            # below us; for a long, the closest high above.
            row = lvl_lo[trig] if side == -1 else lvl_hi[trig]
            row = row[~np.isnan(row)]
            cand = row[row < ref] if side == -1 else row[row > ref]
            if len(cand):
                draw = float(cand.max() if side == -1 else cand.min())
                # ignore draws so close the trade cannot pay for its own spread
                target = (draw if abs(draw - ref) >= 0.5 * risk
                          else ref + side * p.target_R * risk)
            else:
                target = ref + side * p.target_R * risk
        setups.append(Setup(trig, side, float(stop), float(target),
                            float(sweep), float(level)))
        i = trig + 1
    return setups
