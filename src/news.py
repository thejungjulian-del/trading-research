"""News layer: does media coverage predict returns?

Source-agnostic on purpose - GDELT (free, no key, rate-limited to 1 req/5s) and
Alpaca (free key, Benzinga feed) both plug in behind the same panel interface.

THE DISCIPLINE THAT MATTERS, built in rather than bolted on:

  * News dated day t may only predict returns from day t+1 onward. Never same-day.
  * Every test is re-run with the news DELAYED by extra days. If an edge dies
    when the news arrives later, the edge was the timestamp, not the news. That
    single check invalidates most published news-trading results.
  * Coverage volume is normalised against each ticker's own trailing baseline -
    Apple always has more articles than Ametek, and that difference is not news.
"""
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "news"
UA = {"User-Agent": "Mozilla/5.0"}
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
_last_call = [0.0]


def _throttle(min_gap: float = 6.0):
    wait = min_gap - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def gdelt_timeline(query: str, start: str, end: str, mode: str = "timelinevol",
                   retries: int = 3) -> pd.Series | None:
    """Daily article volume ('timelinevol') or average tone ('timelinetone')."""
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(GDELT, headers=UA, timeout=60, params={
                "query": query, "mode": mode, "format": "json",
                "startdatetime": start, "enddatetime": end, "timelinesmooth": "0"})
        except Exception:
            continue
        if r.status_code == 429:
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code != 200:
            return None
        try:
            tl = r.json().get("timeline")
        except Exception:
            return None
        if not tl:
            return None
        pts = tl[0]["data"]
        idx = pd.to_datetime([p["date"] for p in pts])
        return pd.Series([p["value"] for p in pts], index=idx, name=mode)
    return None


def abnormal(s: pd.Series, window: int = 60) -> pd.Series:
    """Z-score against the ticker's OWN trailing baseline, shifted so a day never
    sees itself. Big names always generate more coverage; only the deviation is
    information."""
    mu = s.rolling(window, min_periods=20).mean().shift(1)
    sd = s.rolling(window, min_periods=20).std().shift(1)
    return ((s - mu) / sd.replace(0, np.nan)).rename(f"abn_{s.name}")


def align_to_returns(feat: pd.Series, px: pd.Series, spy: pd.Series,
                     horizons=(1, 5, 20), delay_days: int = 0) -> pd.DataFrame:
    """Join a daily news feature to FORWARD market-adjusted returns.

    delay_days simulates learning the news later than it broke. An edge that
    survives delay is about content; one that vanishes was about speed - or
    about a timestamp that was never trustworthy.
    """
    px = px.dropna()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    spy = spy.reindex(px.index)
    f = feat.copy()
    f.index = pd.to_datetime(f.index).tz_localize(None).normalize()
    f = f.groupby(level=0).mean()

    rows = []
    for d, v in f.items():
        if np.isnan(v):
            continue
        # news on day d is actionable no earlier than the NEXT session
        j = px.index.searchsorted(d + pd.Timedelta(days=1 + delay_days))
        if j >= len(px) - max(horizons) - 1:
            continue
        row = {"date": d, "feature": float(v), "entry_date": px.index[j]}
        base, base_m = px.iloc[j], spy.iloc[j]
        for h in horizons:
            if j + h >= len(px):
                row = None
                break
            row[f"ar_{h}d"] = 1e4 * ((px.iloc[j + h] / base - 1) -
                                     (spy.iloc[j + h] / base_m - 1))
        if row:
            rows.append(row)
    return pd.DataFrame(rows)
