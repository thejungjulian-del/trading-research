"""Company research engine: rank stocks for long and short on measurable
catalysts, then test whether the ranking actually predicts forward returns.

Starts with post-earnings announcement drift (PEAD) because it is the most
durably documented single-stock effect in finance (Ball & Brown 1968, and still
partly alive), and because it is the one signal here with NO fundamental
lookahead risk: the announcement timestamp and the surprise are both known at
the moment of announcement.

Timing discipline: earnings print at 16:00 ET, after the close. The first
tradeable price is therefore the NEXT session's open. Every forward return below
is measured from that open, never from the announcement-day close.

Returns are MARKET-ADJUSTED (SPY subtracted) so we are not just measuring beta.
"""
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "research"
HORIZONS = [1, 5, 20, 60]


def fetch_earnings(tickers, limit: int = 24) -> pd.DataFrame:
    """Announcement date, estimate, actual, surprise % - per ticker."""
    rows = []
    for i, t in enumerate(tickers):
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=limit)
        except Exception:
            continue
        if e is None or not len(e):
            continue
        e = e.dropna(subset=["Reported EPS"])
        for ts, r in e.iterrows():
            rows.append({"ticker": t, "announced": ts,
                         "est": r.get("EPS Estimate"),
                         "actual": r.get("Reported EPS"),
                         "surprise_pct": r.get("Surprise(%)")})
        if i % 25 == 24:
            time.sleep(1)
    df = pd.DataFrame(rows).dropna(subset=["surprise_pct"])
    return df.sort_values(["ticker", "announced"]).reset_index(drop=True)


def fetch_prices(tickers, start="2020-01-01") -> pd.DataFrame:
    px = yf.download(list(tickers) + ["SPY"], start=start, auto_adjust=True,
                     progress=False)["Close"]
    return px.dropna(how="all")


def forward_returns(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Market-adjusted returns from the FIRST TRADEABLE OPEN after the print."""
    px = px.copy()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    spy = px["SPY"]
    out = []
    for _, r in ev.iterrows():
        t = r["ticker"]
        if t not in px.columns:
            continue
        s = px[t].dropna()
        d = pd.Timestamp(r["announced"]).tz_localize(None).normalize()
        # announcement is after the close -> entry is the NEXT session
        j = s.index.searchsorted(d + pd.Timedelta(days=1))
        if j >= len(s) - max(HORIZONS) - 1:
            continue
        base, base_m = s.iloc[j], spy.reindex(s.index).iloc[j]
        row = {**r.to_dict(), "entry_date": s.index[j]}
        ok = True
        for h in HORIZONS:
            if j + h >= len(s):
                ok = False
                break
            stock = s.iloc[j + h] / base - 1
            mkt = spy.reindex(s.index).iloc[j + h] / base_m - 1
            row[f"ar_{h}d"] = 1e4 * (stock - mkt)      # abnormal return, bp
        if ok:
            out.append(row)
    return pd.DataFrame(out)


def decile_profile(df: pd.DataFrame, by: str = "surprise_pct", q: int = 5) -> pd.DataFrame:
    """Sort events by the signal and report abnormal returns per bucket."""
    d = df.dropna(subset=[by]).copy()
    d["bucket"] = pd.qcut(d[by].rank(method="first"), q,
                          labels=[f"Q{i+1}" for i in range(q)])
    spec = {"n": (by, "size"), "signal": (by, "median")}
    spec.update({f"ar_{h}d": (f"ar_{h}d", "mean") for h in HORIZONS})
    return d.groupby("bucket", observed=True).agg(**spec).round(1)


def spread_t(df: pd.DataFrame, by: str, h: int, q: int = 5):
    """Top-bucket minus bottom-bucket abnormal return, with a t-stat clustered
    by announcement month (earnings cluster in time; ignoring that inflates t)."""
    d = df.dropna(subset=[by, f"ar_{h}d"]).copy()
    d["bucket"] = pd.qcut(d[by].rank(method="first"), q, labels=False)
    top = d[d.bucket == q - 1][f"ar_{h}d"]
    bot = d[d.bucket == 0][f"ar_{h}d"]
    spread = top.mean() - bot.mean()
    d["m"] = pd.to_datetime(d["entry_date"]).dt.to_period("M")
    per = d[d.bucket.isin([0, q - 1])].groupby(["m", "bucket"])[f"ar_{h}d"].mean().unstack()
    if per.shape[1] < 2:
        return spread, np.nan, 0
    diff = (per[q - 1] - per[0]).dropna()
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 2 else np.nan
    return spread, t, len(diff)


def shuffle_null(df: pd.DataFrame, by: str, h: int, q: int = 5,
                 iters: int = 300, seed: int = 0, within_month: bool = True
                 ) -> np.ndarray:
    """Permute the SIGNAL, keeping returns in place.

    Shuffling across ALL events destroys the fact that earnings cluster heavily
    in time - hundreds land in the same fortnight and share the same market
    conditions. That makes the null far too narrow and stamps "significant" on
    noise. Permuting WITHIN each announcement month preserves the time structure
    and randomises only the cross-sectional ranking, which is the thing being
    tested.
    """
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=[by, f"ar_{h}d"]).copy()
    d["m"] = pd.to_datetime(d["entry_date"]).dt.to_period("M")
    ret = d[f"ar_{h}d"].to_numpy()
    groups = [np.flatnonzero(d["m"].to_numpy() == m) for m in d["m"].unique()]
    sig = d[by].to_numpy()
    out = []
    for _ in range(iters):
        s = sig.copy()
        if within_month:
            for idx in groups:
                if len(idx) > 1:
                    s[idx] = rng.permutation(s[idx])
        else:
            s = rng.permutation(s)
        b = pd.qcut(pd.Series(s).rank(method="first"), q, labels=False).to_numpy()
        out.append(ret[b == q - 1].mean() - ret[b == 0].mean())
    return np.array(out)


# ---------------------------------------------- better surprise measures
def add_signals(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Surprise(%) divides by the estimate, so a $0.01 estimate reporting $0.02
    prints +100% and swamps the ranking with small-denominator noise. Build the
    measures the literature actually uses instead.

      sue        - unexpected EPS standardised by that company's OWN history of
                   surprises (Standardised Unexpected Earnings). Expanding
                   window, so it never sees the future.
      surp_px    - unexpected EPS scaled by share price - denominator can't
                   collapse toward zero.
      gap_ar     - the market's own verdict: the overnight gap from the
                   announcement-day close to the next open, market-adjusted.
                   This is the cleanest modern PEAD conditioning variable.
    """
    px = px.copy()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    spy = px["SPY"]
    d = ev.sort_values(["ticker", "announced"]).copy()
    d["ue"] = d["actual"] - d["est"]

    # expanding std of that ticker's own past unexpected earnings (shifted)
    g = d.groupby("ticker")["ue"]
    d["ue_sd"] = g.transform(lambda s: s.shift(1).expanding(min_periods=4).std())
    d["sue"] = d["ue"] / d["ue_sd"]

    gaps, prices = [], []
    for _, r in d.iterrows():
        t = r["ticker"]
        if t not in px.columns:
            gaps.append(np.nan); prices.append(np.nan); continue
        s = px[t].dropna()
        day = pd.Timestamp(r["announced"]).tz_localize(None).normalize()
        j = s.index.searchsorted(day + pd.Timedelta(days=1))
        if j < 1 or j >= len(s):
            gaps.append(np.nan); prices.append(np.nan); continue
        prev, nxt = s.iloc[j - 1], s.iloc[j]
        m = spy.reindex(s.index)
        gaps.append(1e4 * ((nxt / prev - 1) - (m.iloc[j] / m.iloc[j - 1] - 1)))
        prices.append(float(prev))
    d["gap_ar"] = gaps
    d["price"] = prices
    d["surp_px"] = 1e4 * d["ue"] / d["price"]
    return d


def bucket_vs_rest(df: pd.DataFrame, by: str, h: int, bucket: int = 0, q: int = 5,
                   lo=None, hi=None):
    """One bucket against all the others. Needed because a top-minus-bottom
    spread cancels a U-shaped pattern, where BOTH tails outperform the middle.

    t is clustered by announcement month - earnings cluster in time and treating
    them as independent roughly doubles the apparent significance.
    """
    d = df.dropna(subset=[by, f"ar_{h}d"]).copy()
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    if lo is not None:
        d = d[d.entry_date >= lo]
    if hi is not None:
        d = d[d.entry_date < hi]
    if len(d) < 200:
        return None
    d["b"] = pd.qcut(d[by].rank(method="first"), q, labels=False)
    d["m"] = d["entry_date"].dt.to_period("M")
    sel, rest = d[d.b == bucket], d[d.b != bucket]
    per = d.groupby(["m", d.b == bucket])[f"ar_{h}d"].mean().unstack()
    if per.shape[1] < 2:
        return None
    diff = (per[True] - per[False]).dropna()
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 2 else np.nan
    return {"n": len(sel), "bucket_ar": round(float(sel[f"ar_{h}d"].mean()), 1),
            "rest_ar": round(float(rest[f"ar_{h}d"].mean()), 1),
            "edge_bp": round(float(sel[f"ar_{h}d"].mean() - rest[f"ar_{h}d"].mean()), 1),
            "t": round(float(t), 2), "months": len(diff)}


# ------------------------------------------- long-horizon (overlap-safe) tests
def cluster_t(d: pd.DataFrame, col: str, group_col: str):
    """Mean and t, clustered on `group_col`."""
    per = d.groupby(group_col)[col].mean()
    if len(per) < 3:
        return np.nan, np.nan, len(per)
    m = per.mean()
    t = m / (per.std(ddof=1) / np.sqrt(len(per)))
    return float(m), float(t), len(per)


def long_horizon_test(df: pd.DataFrame, by: str, h: int, bucket: int = 0,
                      q: int = 5, lo=None, hi=None, cluster: str = "Q"):
    """One bucket vs the rest at a long horizon.

    A 60-day window spans about a quarter, so events from the same quarter share
    most of their return window - they are nowhere near independent. Clustering
    by MONTH (fine for a 1-day horizon) badly understates that overlap, so the
    cluster is widened to the horizon's own length: quarters for 60d.
    """
    d = df.dropna(subset=[by, f"ar_{h}d"]).copy()
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    if lo is not None:
        d = d[d.entry_date >= lo]
    if hi is not None:
        d = d[d.entry_date < hi]
    if len(d) < 200:
        return None
    d["b"] = pd.qcut(d[by].rank(method="first"), q, labels=False)
    d["grp"] = d["entry_date"].dt.to_period(cluster)
    per = d.groupby(["grp", d.b == bucket])[f"ar_{h}d"].mean().unstack()
    if per.shape[1] < 2:
        return None
    diff = (per[True] - per[False]).dropna()
    if len(diff) < 3:
        return None
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    sel = d[d.b == bucket]
    return {"n": len(sel), "edge_bp": round(float(diff.mean()), 1),
            "t": round(float(t), 2), "clusters": len(diff),
            "hit": round(float((diff > 0).mean()), 2)}


def add_momentum(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """12-1 month price momentum measured strictly BEFORE the entry date.

    Cross-sectional momentum is the most replicated anomaly in equities, and it
    needs no fundamental data, so there is no point-in-time lookahead risk.
    The most recent month is skipped - the standard construction, because the
    last month reverses.
    """
    px = px.copy()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    out = []
    for _, r in ev.iterrows():
        t = r["ticker"]
        if t not in px.columns:
            out.append(np.nan); continue
        s = px[t].dropna()
        j = s.index.searchsorted(pd.Timestamp(r["entry_date"]))
        if j < 252:
            out.append(np.nan); continue
        out.append(1e4 * (s.iloc[j - 21] / s.iloc[j - 252] - 1))
    ev = ev.copy()
    ev["mom_12_1"] = out
    return ev
