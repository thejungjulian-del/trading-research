"""Cross-sectional momentum, tested the standard way.

Not conditioned on earnings - every stock, every month. This is the most
replicated anomaly in equities (Jegadeesh & Titman 1993) and it is the one
"recognisable pattern" with a 30-year public track record.

Construction is the textbook one: rank on the trailing 12-month return SKIPPING
the most recent month (the last month reverses, so including it fights you),
rebalance monthly, hold the top decile long and the bottom decile short.
Everything is measured strictly before the holding period begins.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def build(px: pd.DataFrame, n_buckets: int = 5, hold_days: int = 21):
    px = px.copy()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    spy = px["SPY"]
    px = px.drop(columns=["SPY"])

    month_ends = px.resample("ME").last().index
    rows = []
    for d in month_ends:
        j = px.index.searchsorted(d)
        if j < 253 or j + hold_days >= len(px):
            continue
        # signal: return from 252 days ago to 21 days ago (skip last month)
        sig = px.iloc[j - 21] / px.iloc[j - 252] - 1
        sig = sig.dropna()
        if len(sig) < 40:
            continue
        # forward return over the holding period, market-adjusted
        fwd = px.iloc[j + hold_days] / px.iloc[j] - 1
        mkt = spy.iloc[j + hold_days] / spy.iloc[j] - 1
        ar = (fwd - mkt).reindex(sig.index).dropna()
        sig = sig.reindex(ar.index)
        b = pd.qcut(sig.rank(method="first"), n_buckets, labels=False)
        for k in range(n_buckets):
            rows.append({"date": px.index[j], "bucket": k,
                         "ar_bp": 1e4 * float(ar[b == k].mean()),
                         "n": int((b == k).sum())})
    return pd.DataFrame(rows)


def summarise(r: pd.DataFrame, n_buckets: int = 5, lo=None, hi=None):
    d = r.copy()
    if lo is not None:
        d = d[d.date >= lo]
    if hi is not None:
        d = d[d.date < hi]
    w = d.pivot(index="date", columns="bucket", values="ar_bp").dropna()
    if len(w) < 6:
        return None
    spread = w[n_buckets - 1] - w[0]
    t = spread.mean() / (spread.std(ddof=1) / np.sqrt(len(spread)))
    return {"months": len(w),
            "top_bp": round(float(w[n_buckets - 1].mean()), 1),
            "bottom_bp": round(float(w[0].mean()), 1),
            "spread_bp": round(float(spread.mean()), 1),
            "t": round(float(t), 2),
            "hit": round(float((spread > 0).mean()), 2)}
