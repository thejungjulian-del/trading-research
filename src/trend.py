"""Objective trend labelling and trend-break events.

The question this exists to answer: does anything predict a trend break EARLY?
"Market makers are about to break this trend" is only testable once "trend" and
"break" are defined without hindsight, so both are defined structurally here and
both are causal.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

import detectors as D


def trend_state(df: pd.DataFrame, swing_n: int = 3) -> pd.Series:
    """+1 uptrend, -1 downtrend, 0 undefined.

    Structural definition: an uptrend requires the last two confirmed swing highs
    to be rising AND the last two confirmed swing lows to be rising. State only
    changes on information available at that bar (swings use confirm_i).
    """
    sw = D.swings(df, swing_n)
    by_confirm: dict[int, list] = {}
    for s in sw:
        by_confirm.setdefault(s.confirm_i, []).append(s)
    highs: list[float] = []
    lows: list[float] = []
    out = np.zeros(len(df), dtype=np.int8)
    cur = 0
    for i in range(len(df)):
        for s in by_confirm.get(i, []):
            (highs if s.kind == "high" else lows).append(s.price)
        if len(highs) >= 2 and len(lows) >= 2:
            hh, hl = highs[-1] > highs[-2], lows[-1] > lows[-2]
            lh, ll = highs[-1] < highs[-2], lows[-1] < lows[-2]
            if hh and hl:
                cur = 1
            elif lh and ll:
                cur = -1
        out[i] = cur
    return pd.Series(out, index=df.index, name="trend")


@dataclass(frozen=True)
class Break:
    i: int          # bar the flip is confirmed
    start_i: int    # bar the trend began
    side: int       # the trend that ended: +1 was up, -1 was down
    age: int        # bars the trend lasted


def trend_breaks(state: pd.Series) -> list[Break]:
    """A break = the state flipping to the opposite sign (0 does not count)."""
    s = state.to_numpy()
    out, cur, start = [], 0, 0
    for i in range(len(s)):
        if s[i] != 0 and s[i] != cur:
            if cur != 0:
                out.append(Break(i, start, cur, i - start))
            cur, start = s[i], i
    return out


def label_forward(df: pd.DataFrame, state: pd.Series, horizon: int = 10) -> pd.DataFrame:
    """For every in-trend bar: does the trend break within `horizon` bars?

    This is the supervised target. Predictors get scored on how well they
    separate will-break from will-not-break, WITHOUT seeing the future.
    """
    s = state.to_numpy()
    n = len(s)
    breaks_at = np.zeros(n, dtype=bool)
    for b in trend_breaks(state):
        breaks_at[b.i] = True
    will = np.zeros(n, dtype=bool)
    for i in range(n):
        j = min(i + horizon + 1, n)
        will[i] = breaks_at[i + 1:j].any()
    return pd.DataFrame({"trend": s, "will_break": will}, index=df.index)


# ================= alternative trend definitions (robustness) =============
# Every finding must survive a change of definition. If a result only exists
# under swing-structure labelling, it is a property of my labelling, not of the
# market.

def trend_ma(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Classic: fast EMA above slow EMA and rising."""
    c = df["close"]
    f, s = c.ewm(span=fast).mean(), c.ewm(span=slow).mean()
    up = (f > s) & (s.diff() > 0)
    dn = (f < s) & (s.diff() < 0)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index,
                     name="trend").shift(1).fillna(0).astype(np.int8)


def trend_donchian(df: pd.DataFrame, n: int = 40) -> pd.Series:
    """Breakout state: last N-bar extreme taken decides direction."""
    h, l, c = df["high"], df["low"], df["close"]
    hi = h.rolling(n).max().shift(1)
    lo = l.rolling(n).min().shift(1)
    raw = pd.Series(np.where(c > hi, 1, np.where(c < lo, -1, np.nan)), index=df.index)
    return raw.ffill().fillna(0).shift(1).fillna(0).astype(np.int8).rename("trend")


def trend_regression(df: pd.DataFrame, n: int = 40, min_r2: float = 0.5) -> pd.Series:
    """Slope of a rolling linear fit, gated on the fit actually being good.
    A trend nobody could draw a line through is not a trend."""
    c = np.log(df["close"].to_numpy())
    out = np.zeros(len(c), dtype=np.int8)
    x = np.arange(n)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    for i in range(n, len(c)):
        y = c[i - n:i]
        ym = y.mean()
        b = ((x - xm) * (y - ym)).sum() / sxx
        resid = y - (ym + b * (x - xm))
        ss_tot = ((y - ym) ** 2).sum()
        r2 = 1 - resid.var() * n / ss_tot if ss_tot > 0 else 0
        if r2 >= min_r2:
            out[i] = 1 if b > 0 else -1
    return pd.Series(out, index=df.index, name="trend")


DEFINITIONS = {
    "structure": lambda d: trend_state(d, 3),
    "ma_cross":  trend_ma,
    "donchian":  trend_donchian,
    "regression": trend_regression,
}
