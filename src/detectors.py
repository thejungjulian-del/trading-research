"""Mechanical detectors for the TJR model. See ../STRATEGY.md.

Every detector is causal: a signal at bar i uses only bars <= i.
Swing points are the exception in the obvious way - a swing at bar i is only
*confirmed* (and therefore usable) at bar i+n. Detectors return the confirm index.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- swings
@dataclass(frozen=True)
class Swing:
    i: int          # bar index of the extreme
    confirm_i: int  # first bar at which it is knowable (i + n)
    price: float
    kind: str       # "high" | "low"


def swings(df: pd.DataFrame, n: int = 2) -> list[Swing]:
    """Fractal swings: bar i is a swing high if its high is the strict max of
    [i-n, i+n]. Ties are rejected, so flat tops produce no swing."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    out = []
    for i in range(n, len(df) - n):
        w = slice(i - n, i + n + 1)
        if h[i] == h[w].max() and (h[w] == h[i]).sum() == 1:
            out.append(Swing(i, i + n, float(h[i]), "high"))
        if l[i] == l[w].min() and (l[w] == l[i]).sum() == 1:
            out.append(Swing(i, i + n, float(l[i]), "low"))
    return sorted(out, key=lambda s: (s.confirm_i, s.i))


# ------------------------------------------------------------------- fvg
@dataclass(frozen=True)
class FVG:
    i: int        # index of candle 3 - the bar at which the gap exists
    lo: float     # zone bottom
    hi: float     # zone top
    bull: bool    # True = bullish gap (support), False = bearish gap (resistance)


def fvgs(df: pd.DataFrame) -> list[FVG]:
    """3-bar imbalance. Bullish: bar3.low > bar1.high. Bearish: bar3.high < bar1.low."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    out = []
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:
            out.append(FVG(i, float(h[i - 2]), float(l[i]), True))
        elif h[i] < l[i - 2]:
            out.append(FVG(i, float(h[i]), float(l[i - 2]), False))
    return out


# ------------------------------------------------------------- structure
def structure_breaks(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Break of structure on close, against the most recent *confirmed* swing.

    Returns a frame indexed like df with columns:
      bos          : +1 bullish break, -1 bearish break, 0 none
      last_sh/sl   : the confirmed swing levels in force at that bar
    A level is consumed once broken, so one swing yields at most one BOS.
    """
    sw = swings(df, n)
    by_confirm: dict[int, list[Swing]] = {}
    for s in sw:
        by_confirm.setdefault(s.confirm_i, []).append(s)

    c = df["close"].to_numpy()
    bos = np.zeros(len(df), dtype=np.int8)
    sh = np.full(len(df), np.nan)
    sl = np.full(len(df), np.nan)
    cur_h = cur_l = np.nan

    for i in range(len(df)):
        for s in by_confirm.get(i, []):          # swings become known at bar i
            if s.kind == "high":
                cur_h = s.price
            else:
                cur_l = s.price
        sh[i], sl[i] = cur_h, cur_l
        if not np.isnan(cur_h) and c[i] > cur_h:
            bos[i] = 1
            cur_h = np.nan                        # consumed
        elif not np.isnan(cur_l) and c[i] < cur_l:
            bos[i] = -1
            cur_l = np.nan
    return pd.DataFrame({"bos": bos, "last_sh": sh, "last_sl": sl}, index=df.index)


def inverse_fvgs(df: pd.DataFrame, max_age: int = 50) -> pd.DataFrame:
    """An FVG is 'inverted' when a later bar CLOSES fully through it.

    A bullish gap closed below flips to bearish resistance -> bearish signal (-1).
    A bearish gap closed above flips to bullish support   -> bullish signal (+1).
    Gaps older than max_age bars are dropped.
    """
    c = df["close"].to_numpy()
    sig = np.zeros(len(df), dtype=np.int8)
    live: list[FVG] = []
    gaps_at = {}
    for g in fvgs(df):
        gaps_at.setdefault(g.i, []).append(g)

    for i in range(len(df)):
        live = [g for g in live if i - g.i <= max_age]
        still = []
        for g in live:
            if g.bull and c[i] < g.lo:
                sig[i] = -1
            elif not g.bull and c[i] > g.hi:
                sig[i] = 1
            else:
                still.append(g)
        live = still + gaps_at.get(i, [])
    return pd.DataFrame({"ifvg": sig}, index=df.index)


# ----------------------------------------------------------- equilibrium
def equilibrium(low: float, high: float) -> float:
    """50% of a leg. Above it = premium (sell only), below = discount (buy only)."""
    return (low + high) / 2.0


# ------------------------------------------------------- session levels
SESSIONS = {
    # ET windows. The free QQQ/SPY proxy has no Asia session - see STRATEGY.md.
    "premarket": ("04:00", "09:30"),   # stands in for London
    "rth_am":    ("09:30", "12:00"),
    "rth_pm":    ("12:00", "16:00"),
}


def session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """For each bar, the highs/lows of completed sessions available at that time,
    plus prior-day high/low. All strictly backward-looking."""
    idx = df.index
    day = idx.normalize()
    out = pd.DataFrame(index=idx)

    # prior day extremes
    daily = df.groupby(day).agg(dh=("high", "max"), dl=("low", "min"))
    out["pdh"] = day.map(daily["dh"].shift(1))
    out["pdl"] = day.map(daily["dl"].shift(1))

    # completed premarket of the current day
    t = idx.time
    import datetime as _dt
    pm_mask = (t >= _dt.time(4, 0)) & (t < _dt.time(9, 30))
    pm = df[pm_mask].groupby(day[pm_mask]).agg(h=("high", "max"), l=("low", "min"))
    out["pmh"] = day.map(pm["h"])
    out["pml"] = day.map(pm["l"])
    out.loc[pm_mask, ["pmh", "pml"]] = np.nan       # not complete until 09:30
    return out


# ------------------------------------------------------------------ smt
def smt_divergence(a: pd.DataFrame, b: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Bearish (-1): a makes a new `lookback` high this bar, b does not.
    Bullish (+1): a makes a new low, b does not. Sign is from a's perspective;
    the FAILING symbol is the one to trade (see STRATEGY.md step 0)."""
    a, b = a.align(b, join="inner", axis=0)
    ah = a["high"].rolling(lookback).max().shift(1)
    al = a["low"].rolling(lookback).min().shift(1)
    bh = b["high"].rolling(lookback).max().shift(1)
    bl = b["low"].rolling(lookback).min().shift(1)
    a_new_h, b_new_h = a["high"] > ah, b["high"] > bh
    a_new_l, b_new_l = a["low"] < al, b["low"] < bl
    sig = np.zeros(len(a), dtype=np.int8)
    sig[(a_new_h & ~b_new_h).to_numpy()] = -1
    sig[(a_new_l & ~b_new_l).to_numpy()] = 1
    return pd.DataFrame({"smt": sig}, index=a.index)


# --------------------------------------------------------- order blocks
@dataclass(frozen=True)
class OrderBlock:
    i: int          # the origin candle
    confirm_i: int  # only knowable once the impulse has happened
    lo: float
    hi: float
    bull: bool      # True = demand zone, False = supply zone


def order_blocks(df: pd.DataFrame, impulse_atr: float = 1.5,
                 horizon: int = 3, atr_n: int = 14) -> list[OrderBlock]:
    """'Mark the candle that started a strong move' - the supply/demand zone as
    defined in the TradingLab video.

    Bearish (supply): an UP candle followed within `horizon` bars by a move down
    of at least `impulse_atr` ATRs. Zone = that candle's [low, high].
    Causal: the block is only confirmed at i + horizon, since the impulse that
    defines it has not happened yet at bar i.
    """
    h, l, c, o = (df[k].to_numpy() for k in ("high", "low", "close", "open"))
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(atr_n).mean().to_numpy()
    out = []
    for i in range(atr_n, len(df) - horizon):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        move = c[i + horizon] - c[i]
        if move <= -impulse_atr * atr[i] and c[i] > o[i]:
            out.append(OrderBlock(i, i + horizon, float(l[i]), float(h[i]), False))
        elif move >= impulse_atr * atr[i] and c[i] < o[i]:
            out.append(OrderBlock(i, i + horizon, float(l[i]), float(h[i]), True))
    return out


# ------------------------------------------------- real liquidity levels
def _round_step(px: float) -> float:
    return 1.0 if px < 50 else 5.0 if px < 200 else 10.0


SESSION_WINDOWS = {
    # ET. US stocks genuinely do not trade Asia hours, so "premkt" (04:00-09:30)
    # is the only pre-US session visible in equity data - it overlaps London.
    "premkt": ("04:00", "09:30"),
    "ny_am":  ("09:30", "12:00"),
    "ny_pm":  ("12:00", "16:00"),
}

# CME futures trade ~23h, so TJR's actual session model is only testable here.
# Asia spans midnight and is handled by the wrap-around branch below.
FUTURES_SESSIONS = {
    "asia":   ("18:00", "03:00"),
    "london": ("03:00", "09:30"),
    "ny_am":  ("09:30", "12:00"),
    "ny_pm":  ("12:00", "16:00"),
}


def session_extremes(df: pd.DataFrame, windows=None, cme: bool = False) -> pd.DataFrame:
    """High/low of each COMPLETED session, available only after it closes.

    cme=True groups on the CME trade date (starting 18:00 ET) and handles the
    Asia session wrapping past midnight.
    """
    import datetime as _dt
    idx = df.index
    day = ((idx + pd.Timedelta(hours=6)).normalize() if cme else idx.normalize())
    t = idx.time
    out = pd.DataFrame(index=idx)
    for name, (a, b) in (windows or SESSION_WINDOWS).items():
        ha, ma = map(int, a.split(":")); hb, mb = map(int, b.split(":"))
        ta, tb = _dt.time(ha, ma), _dt.time(hb, mb)
        wraps = ta > tb
        m = ((t >= ta) | (t < tb)) if wraps else ((t >= ta) & (t < tb))
        if not m.any():
            continue
        g = df[m].groupby(day[m]).agg(h=("high", "max"), l=("low", "min"))
        # A session extreme is knowable ONLY from the moment that session ENDS.
        # Mapping it across the whole day and blanking it just *during* the
        # session leaves every EARLIER bar reading a high from later that day -
        # a lookahead that manufactures enormous fake edge.
        # available only once the session has ENDED. For a wrapping session
        # (Asia) that means the window between its end and the next day's start.
        after = ((t >= tb) & (t < ta)) if wraps else (t >= tb)
        out[f"{name}_h"] = np.where(after, day.map(g["h"]), np.nan)
        out[f"{name}_l"] = np.where(after, day.map(g["l"]), np.nan)
    return out


def liquidity_levels(df: pd.DataFrame, kinds=("day", "week", "n20"),
                     futures: bool = False) -> pd.DataFrame:
    """Levels a crowd can actually see, and therefore where stops cluster.

    A 25-minute fractal high has nobody's stop under it. Yesterday's low, last
    week's low, the 20-day high and round numbers do. All strictly backward
    looking: a level is known only from the session AFTER it completed.
    """
    idx = df.index
    day = idx.normalize()
    out = pd.DataFrame(index=idx)
    daily = df.groupby(day).agg(h=("high", "max"), l=("low", "min"))

    if "day" in kinds:
        out["pdh"] = day.map(daily["h"].shift(1))
        out["pdl"] = day.map(daily["l"].shift(1))
    if "week" in kinds:
        out["pwh"] = day.map(daily["h"].shift(1).rolling(5, min_periods=2).max())
        out["pwl"] = day.map(daily["l"].shift(1).rolling(5, min_periods=2).min())
    if "n20" in kinds:
        out["n20h"] = day.map(daily["h"].shift(1).rolling(20, min_periods=5).max())
        out["n20l"] = day.map(daily["l"].shift(1).rolling(20, min_periods=5).min())
    if "round" in kinds:
        po = day.map(daily["h"].shift(1))          # prior day high as the anchor price
        step = np.array([_round_step(v) if not np.isnan(v) else np.nan for v in po])
        c = df["close"].to_numpy()
        out["rnd_h"] = np.ceil(c / step) * step
        out["rnd_l"] = np.floor(c / step) * step
    if "eq" in kinds:
        # relative equal highs/lows - the densest stop clusters
        eh = np.full(len(df), np.nan); el = np.full(len(df), np.nan)
        for e in equal_levels(df):
            arr = eh if e.kind == "high" else el
            if e.confirm_i < len(df):
                arr[e.confirm_i:] = e.price      # live from confirmation onward
        out["eq_h"] = eh
        out["eq_l"] = el
    if "session" in kinds:
        se = (session_extremes(df, FUTURES_SESSIONS, cme=True) if futures
              else session_extremes(df))
        for c_ in se.columns:
            out[c_] = se[c_]
    if "gap" in kinds:
        # NDOG / NWOG edges are imbalances price "wants to seek out to balance"
        gh = np.full(len(df), np.nan); gl = np.full(len(df), np.nan)
        for g in opening_gaps(df):
            gh[g.i:] = g.hi
            gl[g.i:] = g.lo
        out["gap_h"] = gh
        out["gap_l"] = gl
    if "range" in kinds:
        r = ranges(df)
        out["rng_h"] = r["range_h"].where(r["in_range"])
        out["rng_l"] = r["range_l"].where(r["in_range"])
    if "vwap" in kinds:
        vw = vwap(df)
        out["vwap_uh"] = vw["vwap_u2"]
        out["vwap_ll"] = vw["vwap_l2"]
    if "profile" in kinds:
        vp = volume_profile(df)
        out["va_hh"] = vp["va_high"]
        out["va_ll"] = vp["va_low"]
    return out


def liquidity_sweeps(df: pd.DataFrame, kinds=("day", "week", "n20"),
                     futures: bool = False) -> pd.DataFrame:
    """Per-bar sweep detection against those levels.

    Each level is CONSUMED once taken, per session - otherwise a trending day
    that closes above the prior-day high re-registers a fresh 'sweep' on every
    remaining bar, which is the bug that wrecked the first version of this.

    Returns: swept_h / swept_l (the level value taken on that bar, else NaN)
    """
    lv = liquidity_levels(df, kinds, futures)
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    day = ((df.index + pd.Timedelta(hours=6)).normalize().to_numpy() if futures
           else df.index.normalize().to_numpy())
    hi_cols = [c for c in lv.columns if c.endswith(("h", "_h")) and not c.endswith("_l")]
    lo_cols = [c for c in lv.columns if c.endswith(("l", "_l")) and not c.endswith("_h")]
    sw_h = np.full(len(df), np.nan)
    sw_l = np.full(len(df), np.nan)

    for cols, arr, sink, gt in ((hi_cols, h, sw_h, True), (lo_cols, l, sw_l, False)):
        for c in cols:
            vals = lv[c].to_numpy()
            cur_day, spent = None, False
            for i in range(len(df)):
                if day[i] != cur_day:
                    cur_day, spent = day[i], False
                v = vals[i]
                if spent or np.isnan(v):
                    continue
                if (arr[i] > v) if gt else (arr[i] < v):
                    spent = True
                    # keep the most extreme level taken on this bar
                    sink[i] = v if np.isnan(sink[i]) else (max(sink[i], v) if gt
                                                          else min(sink[i], v))
    return pd.DataFrame({"swept_h": sw_h, "swept_l": sw_l}, index=df.index)


# ------------------------------------------- relative equal highs / lows
@dataclass(frozen=True)
class EqualLevel:
    confirm_i: int   # bar at which the cluster reaches min_count and becomes known
    price: float     # the level itself (the extreme of the cluster)
    count: int       # how many swings stack there - this is the "strength"
    kind: str        # "high" | "low"


def equal_levels(df: pd.DataFrame, swing_n: int = 3, tol_bp: float = 15.0,
                 min_count: int = 2, lookback: int = 120) -> list[EqualLevel]:
    """Relative equal highs/lows - the densest liquidity pool TJR describes.

    One high has one set of stops above it. Three highs at the same price have
    three sets stacked in the same band ("four times the amount of orders"), so
    an equal-high cluster is a materially stronger draw than a lone swing high.

    Two swings are "relatively equal" if they sit within tol_bp basis points of
    each other and within `lookback` bars. The level becomes usable only at the
    bar where the cluster reaches min_count - never earlier.
    """
    out: list[EqualLevel] = []
    for kind in ("high", "low"):
        sw = [s for s in swings(df, swing_n) if s.kind == kind]
        clusters: list[list] = []          # each: list of swings deemed equal
        for s in sw:
            placed = False
            for cl in clusters:
                ref = cl[-1]
                if (abs(s.price - ref.price) / ref.price * 1e4 <= tol_bp
                        and s.i - ref.i <= lookback):
                    cl.append(s)
                    if len(cl) >= min_count:
                        px = max(x.price for x in cl) if kind == "high" \
                            else min(x.price for x in cl)
                        out.append(EqualLevel(s.confirm_i, float(px), len(cl), kind))
                    placed = True
                    break
            if not placed:
                clusters.append([s])
    return sorted(out, key=lambda e: e.confirm_i)


# ------------------------------------------ opening gaps (NDOG/NWOG/NCOG)
@dataclass(frozen=True)
class Gap:
    i: int          # first bar of the new period - the gap exists from here
    lo: float
    hi: float
    kind: str       # "ndog" | "nwog" | "ncog"


def opening_gaps(df: pd.DataFrame, kinds=("ndog", "nwog")) -> list[Gap]:
    """TJR's 'advanced imbalance concepts': the untraded space between one
    period's close and the next period's open.

    "There's a small hour where the market is not going to be trading... that
    would be considered a new day opening gap... an area of imbalance price is
    actively going to want to seek out to balance out."

    Causal by construction: the gap is only defined once the new period opens,
    and it is stamped at that bar.
    """
    day = df.index.normalize()
    c, o = df["close"].to_numpy(), df["open"].to_numpy()
    starts = np.flatnonzero(day.to_numpy()[1:] != day.to_numpy()[:-1]) + 1
    out = []
    for s in starts:
        prev_c, new_o = float(c[s - 1]), float(o[s])
        if prev_c == new_o:
            continue
        lo, hi = min(prev_c, new_o), max(prev_c, new_o)
        if "ndog" in kinds:
            out.append(Gap(int(s), lo, hi, "ndog"))
        # a new WEEK opening gap is a new-day gap that crosses a week boundary
        if "nwog" in kinds and df.index[s].weekday() < df.index[s - 1].weekday():
            out.append(Gap(int(s), lo, hi, "nwog"))
    return out


# ------------------------------------------------ BPR (balanced price range)
def bpr(df: pd.DataFrame, max_age: int = 200) -> list[FVG]:
    """Balanced Price Range: where a bullish gap and a bearish gap OVERLAP.

    Two opposing imbalances covering the same prices is a stronger zone than
    either alone. Returned as an FVG whose index is the LATER of the two, so it
    can never be used before both halves exist.
    """
    gs = fvgs(df)
    out = []
    for k, b in enumerate(gs):
        # pair with the NEAREST preceding opposing gap only - a BPR is a specific
        # two-gap overlap, not every combination of gaps in a window
        for a in reversed(gs[:k]):
            if b.i - a.i > max_age:
                break
            if a.bull == b.bull:
                continue
            lo, hi = max(a.lo, b.lo), min(a.hi, b.hi)
            if lo < hi:
                out.append(FVG(b.i, lo, hi, b.bull))
            break
    return out


# ---------------------------------------------------------- breaker blocks
def breakers(df: pd.DataFrame, swing_n: int = 3) -> list[FVG]:
    """A level that was support, got broken, and should now act as resistance
    (or the mirror). Represented as a zone so it plugs into the same machinery.

    Stamped at the bar the break is confirmed - never before.
    """
    st = structure_breaks(df, swing_n)
    bos = st["bos"].to_numpy()
    sh, sl = st["last_sh"].to_numpy(), st["last_sl"].to_numpy()
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    out = []
    for i in range(len(df)):
        if bos[i] == -1 and not np.isnan(sl[i]):
            # broken support -> resistance zone at the broken level
            out.append(FVG(i, float(sl[i]), float(max(sl[i], h[i])), False))
        elif bos[i] == 1 and not np.isnan(sh[i]):
            out.append(FVG(i, float(min(sh[i], l[i])), float(sh[i]), True))
    return out


# --------------------------------------------------- consolidation / range
def ranges(df: pd.DataFrame, window: int = 24, max_atr: float = 4.0,
           atr_n: int = 14) -> pd.DataFrame:
    """Consolidation: a stretch where price is contained in a tight band.
    Its high and low are liquidity - stops sit just outside a visible range.

    in_range / range_h / range_l are all shifted so a bar never sees the range
    it is still forming.
    """
    h, l = df["high"], df["low"]
    prev = df["close"].shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_n).mean()
    rh = h.rolling(window).max()
    rl = l.rolling(window).min()
    tight = (rh - rl) <= max_atr * atr
    return pd.DataFrame({
        "in_range": tight.shift(1).fillna(False),
        "range_h": rh.shift(1),
        "range_l": rl.shift(1),
    }, index=df.index)


# ------------------------- institutional reference levels (VWAP / profile)
def vwap(df: pd.DataFrame, anchor: str = "session") -> pd.DataFrame:
    """Volume-weighted average price - the benchmark institutions are measured
    against. A desk filling a large order is trying to beat VWAP, so it acts as
    a genuine magnet in a way a retail chart level does not.

    Resets each session. Cumulative and therefore causal by construction.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    grp = df.index.normalize() if anchor == "session" else pd.Series(0, index=df.index)
    cum_pv = pv.groupby(grp).cumsum()
    cum_v = df["volume"].groupby(grp).cumsum()
    vw = cum_pv / cum_v.replace(0, np.nan)
    # standard deviation bands - where institutional execution gets stretched
    dev = ((tp - vw) ** 2 * df["volume"]).groupby(grp).cumsum() / cum_v.replace(0, np.nan)
    sd = np.sqrt(dev)
    return pd.DataFrame({"vwap": vw, "vwap_u1": vw + sd, "vwap_l1": vw - sd,
                         "vwap_u2": vw + 2 * sd, "vwap_l2": vw - 2 * sd},
                        index=df.index)


def volume_profile(df: pd.DataFrame, lookback: int = 390, bins: int = 40
                   ) -> pd.DataFrame:
    """Where volume actually transacted, not where the chart looks pretty.

    POC = price level with the most traded volume in the window. HVN/LVN are
    high- and low-volume nodes. Institutions accumulate at high-volume nodes;
    price tends to move fast through low-volume ones.

    Every value is shifted so a bar never sees the profile it is still forming.
    """
    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
    v = df["volume"].to_numpy()
    n = len(df)
    poc = np.full(n, np.nan)
    val = np.full(n, np.nan)   # value-area low
    vah = np.full(n, np.nan)   # value-area high
    for i in range(lookback, n, 5):        # every 5 bars is ample resolution
        p, w = tp[i - lookback:i], v[i - lookback:i]
        if w.sum() <= 0:
            continue
        lo, hi = p.min(), p.max()
        if hi <= lo:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        hist = np.bincount(idx, weights=w, minlength=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        k = int(hist.argmax())
        poc[i:i + 5] = centers[k]
        # value area = smallest contiguous band holding 70% of volume
        order = np.argsort(hist)[::-1]
        tot, acc, taken = hist.sum(), 0.0, []
        for j in order:
            taken.append(j); acc += hist[j]
            if acc >= 0.7 * tot:
                break
        val[i:i + 5] = centers[min(taken)]
        vah[i:i + 5] = centers[max(taken)]
    out = pd.DataFrame({"poc": poc, "va_low": val, "va_high": vah}, index=df.index)
    return out.shift(1)
