"""Does anything predict a trend break EARLY?

Each candidate predictor is scored by AUC against the forward label
"this trend breaks within H bars". AUC 0.50 = no information.

The null is NOT a naive label shuffle. Adjacent 5m bars are nearly identical, so
a naive shuffle destroys autocorrelation and makes trivial deviations look
significant. Labels are permuted in CONTIGUOUS DAY-LENGTH BLOCKS instead, which
preserves the serial structure and gives an honest null band.
"""
import numpy as np
import pandas as pd

import detectors as D, trend


def auc(score: np.ndarray, label: np.ndarray) -> float:
    m = ~np.isnan(score)
    s, y = score[m], label[m]
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def features(df: pd.DataFrame, ref: pd.DataFrame | None, state: pd.Series) -> pd.DataFrame:
    c, h, l, v = (df[k] for k in ("close", "high", "low", "volume"))
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    d = pd.DataFrame(index=df.index)

    # how long has this trend been running
    s = state.to_numpy()
    age = np.zeros(len(s))
    run = 0
    for i in range(1, len(s)):
        run = run + 1 if s[i] == s[i - 1] and s[i] != 0 else 0
        age[i] = run
    d["age"] = age

    d["stretch"] = (c - c.ewm(span=20).mean()) / atr * state      # signed with trend
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14).mean()
    rsi = 100 - 100 / (1 + up / dn)
    d["rsi_extreme"] = (rsi - 50) * state                          # signed with trend
    d["vol_expansion"] = atr.rolling(5).mean() / atr.rolling(20).mean()
    d["volume_ratio"] = v / v.rolling(78).mean()
    d["ret_accel"] = (c.pct_change(5) - c.pct_change(20)) * state

    # TJR's signature tell: this instrument makes a new extreme, the reference
    # index does not (or vice versa)
    if ref is not None:
        a, b = df.align(ref, join="inner", axis=0)
        smt = D.smt_divergence(a, b, lookback=20)["smt"]
        d["smt_against_trend"] = smt.reindex(df.index).fillna(0) * -state
    return d


def block_null(score: np.ndarray, label: np.ndarray, block: int = 78,
               iters: int = 200, seed: int = 0) -> np.ndarray:
    """AUC null from permuting labels in contiguous day-length blocks."""
    rng = np.random.default_rng(seed)
    n = len(label)
    nb = n // block
    blocks = [label[i * block:(i + 1) * block] for i in range(nb)]
    out = []
    for _ in range(iters):
        order = rng.permutation(nb)
        shuffled = np.concatenate([blocks[k] for k in order])
        out.append(auc(score[:len(shuffled)], shuffled))
    return np.array([x for x in out if not np.isnan(x)])
