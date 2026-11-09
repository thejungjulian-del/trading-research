"""Event-agnostic event study.

Give it any list of dated events plus a daily price series; it returns the mean
abnormal return on each day relative to the event, with t-stats, plus a
random-date control run through identical machinery.

Abnormal return = raw return minus the sample's unconditional mean daily return,
so a rising market alone cannot produce a "result".
"""
import numpy as np
import pandas as pd


def daily_returns(px: pd.Series) -> pd.Series:
    return px.pct_change().dropna()


def study(ret: pd.Series, dates, pre: int = 5, post: int = 5) -> pd.DataFrame:
    """ret: daily returns indexed by date. dates: event dates.
    Returns one row per relative day with mean abnormal return and t-stat."""
    idx = ret.index.tz_localize(None) if ret.index.tz is not None else ret.index
    ret = pd.Series(ret.to_numpy(), index=idx)
    mu = ret.mean()
    pos = {d: i for i, d in enumerate(ret.index)}
    rows = []
    for d in pd.to_datetime(pd.Series(list(dates))).dt.normalize():
        # first trading day on or after the event date
        j = ret.index.searchsorted(d)
        if j >= len(ret) or j - pre < 0 or j + post >= len(ret):
            continue
        rows.append(ret.to_numpy()[j - pre: j + post + 1] - mu)
    if not rows:
        return pd.DataFrame()
    A = np.vstack(rows)
    out = []
    for k, rel in enumerate(range(-pre, post + 1)):
        col = A[:, k]
        se = col.std(ddof=1) / np.sqrt(len(col))
        out.append({"rel_day": rel, "n": len(col),
                    "mean_abn_bp": 1e4 * col.mean(),
                    "t": col.mean() / se if se > 0 else np.nan})
    df = pd.DataFrame(out)
    df["cum_abn_bp"] = df["mean_abn_bp"].cumsum()
    return df


def control(ret: pd.Series, n_events: int, pre=5, post=5, iters=500, seed=0) -> pd.DataFrame:
    """Same machinery, random dates. The distribution of outcomes here is the
    yardstick any real result has to beat."""
    rng = np.random.default_rng(seed)
    idx = ret.index.tz_localize(None) if ret.index.tz is not None else ret.index
    lo, hi = pre + 1, len(ret) - post - 1
    peaks = []
    for _ in range(iters):
        d = idx[rng.integers(lo, hi, n_events)]
        s = study(ret, d, pre, post)
        if len(s):
            peaks.append(s["t"].abs().max())
    return pd.Series(peaks)
