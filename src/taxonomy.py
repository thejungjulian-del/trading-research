"""What do trends LEAD TO? Not "does it break" - what actually happens.

Two halves:
  1. classify each trend by measurable attributes present AT ITS START
  2. measure the OUTCOME DISTRIBUTION for each class

Outcomes are in ATR units so instruments of different price and volatility are
comparable. Nothing here is a guarantee - the output is a probability profile.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

import detectors as D, trend as TR

CLASSIFY_BARS = 10   # bars used to classify; outcomes measured strictly after


@dataclass
class Episode:
    start: int
    end: int
    side: int
    # classification (all measured at or before `start`)
    from_range: bool        # broke out of a consolidation
    from_sweep: bool        # began right after a liquidity sweep
    steep: float            # slope over the first 10 bars, in ATR/bar
    vol_regime: float       # ATR at start / its own 100-bar average
    aligned: int            # +1/-1/0 : was the reference index trending the same way
    hour: int
    # outcomes
    mfe: float              # furthest favourable move, in ATR
    mae: float              # furthest adverse move before the end, in ATR
    length: int             # bars


def episodes(df: pd.DataFrame, ref_state: pd.Series | None = None,
             swing_n: int = 3) -> list[Episode]:
    st = TR.trend_state(df, swing_n)
    rng = D.ranges(df)
    sw = D.liquidity_sweeps(df, ("day", "session", "eq"))
    swept = (sw["swept_h"].notna() | sw["swept_l"].notna()).to_numpy()

    h, l, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    atr_avg = pd.Series(atr).rolling(100).mean().to_numpy()
    in_rng = rng["in_range"].to_numpy()
    hours = df.index.hour.to_numpy()
    ref = ref_state.reindex(df.index).fillna(0).to_numpy() if ref_state is not None \
        else np.zeros(len(df))

    out = []
    for b in TR.trend_breaks(st):
        s, e, side = b.start_i, b.i, b.side
        if s < 120 or e >= len(df) or np.isnan(atr[s]) or atr[s] <= 0:
            continue
        # CLASSIFY on the first CLASSIFY_BARS, MEASURE only after them.
        # Measuring the run over a window that includes the bars used to judge
        # "steepness" is circular - a fast start guarantees a big number.
        k = s + CLASSIFY_BARS
        if k >= e:
            continue                       # trend ended inside the judging window
        steep = ((c[k] - c[s]) / atr[s] / CLASSIFY_BARS) * side
        base = c[k]                        # outcomes are measured from HERE
        seg_h, seg_l = h[k:e + 1], l[k:e + 1]
        if side == 1:
            mfe = (seg_h.max() - base) / atr[s]
            mae = (base - seg_l.min()) / atr[s]
        else:
            mfe = (base - seg_l.min()) / atr[s]
            mae = (seg_h.max() - base) / atr[s]
        out.append(Episode(
            start=s, end=e, side=side,
            from_range=bool(in_rng[max(s - 3, 0):s + 1].any()),
            from_sweep=bool(swept[max(s - 5, 0):s + 1].any()),
            steep=float(steep),
            vol_regime=float(atr[s] / atr_avg[s]) if atr_avg[s] > 0 else np.nan,
            aligned=int(np.sign(ref[s]) * side),
            hour=int(hours[s]),
            mfe=float(mfe), mae=float(mae), length=int(e - k)))
    return out


def profile(eps: list[Episode], mask, label: str) -> dict:
    sel = [e for e, m in zip(eps, mask) if m]
    if len(sel) < 50:
        return None
    mfe = np.array([e.mfe for e in sel])
    mae = np.array([e.mae for e in sel])
    ln = np.array([e.length for e in sel])
    return {
        "class": label, "n": len(sel),
        "median_run_atr": round(float(np.median(mfe)), 2),
        "p75_run": round(float(np.percentile(mfe, 75)), 2),
        "median_pain_atr": round(float(np.median(mae)), 2),
        "run_over_2atr": round(float((mfe > 2).mean()), 3),
        "run_over_4atr": round(float((mfe > 4).mean()), 3),
        "median_bars": int(np.median(ln)),
        "reward_risk": round(float(np.median(mfe) / max(np.median(mae), .01)), 2),
    }
