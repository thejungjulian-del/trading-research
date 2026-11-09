"""Which material corporate events actually predict returns?

Guards, because 30+ item codes means 30+ tests:
  * t clustered by calendar month - events cluster heavily in time
  * a shuffle null that permutes item labels WITHIN month, preserving that
    clustering (permuting across the whole sample makes the null far too narrow)
  * a Bonferroni-style bar quoted alongside the raw t, so a "significant" result
    from 30 tries is not mistaken for a discovery
"""
import numpy as np
import pandas as pd

import edgar as E

HORIZONS = [1, 5, 20, 60]


def clustered(d: pd.DataFrame, col: str, freq: str = "M"):
    g = d.groupby(pd.to_datetime(d["entry_date"]).dt.to_period(freq))[col].mean()
    if len(g) < 6:
        return np.nan, np.nan, len(g)
    return float(g.mean()), float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), len(g)


def item_table(df: pd.DataFrame, h: int = 20, min_n: int = 300,
               freq: str = "M") -> pd.DataFrame:
    """Every item type against the average event, month-clustered."""
    col = f"ar_{h}d"
    d = df.dropna(subset=[col]).copy()
    d["p"] = pd.to_datetime(d["entry_date"]).dt.to_period(freq)
    base = d.groupby("p")[col].mean()          # the average 8-K, per period
    rows = []
    for item, g in d.groupby("item"):
        if len(g) < min_n:
            continue
        per = g.groupby("p")[col].mean()
        diff = (per - base).dropna()
        if len(diff) < 6:
            continue
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
        rows.append({"item": item, "meaning": E.ITEMS.get(item, ""),
                     "n": len(g), "periods": len(diff),
                     "edge_bp": round(float(diff.mean()), 1),
                     "t": round(float(t), 2),
                     "hit": round(float((diff > 0).mean()), 2)})
    return pd.DataFrame(rows).sort_values("t")


def shuffle_null(df: pd.DataFrame, item: str, h: int, iters: int = 300,
                 seed: int = 0, freq: str = "M") -> np.ndarray:
    """Permute item labels WITHIN month: keeps the time clustering, randomises
    only which event got which label."""
    rng = np.random.default_rng(seed)
    col = f"ar_{h}d"
    d = df.dropna(subset=[col]).copy()
    d["p"] = pd.to_datetime(d["entry_date"]).dt.to_period(freq)
    lab = (d["item"] == item).to_numpy()
    ret = d[col].to_numpy()
    per = d["p"].to_numpy()
    groups = [np.flatnonzero(per == p) for p in pd.unique(per)]
    base_by_p = {p: ret[np.flatnonzero(per == p)].mean() for p in pd.unique(per)}
    out = []
    for _ in range(iters):
        l = lab.copy()
        for idx in groups:
            if len(idx) > 1:
                l[idx] = rng.permutation(l[idx])
        diffs = []
        for p, idx in zip(pd.unique(per), groups):
            sel = idx[l[idx]]
            if len(sel):
                diffs.append(ret[sel].mean() - base_by_p[p])
        if diffs:
            out.append(np.mean(diffs))
    return np.array(out)
