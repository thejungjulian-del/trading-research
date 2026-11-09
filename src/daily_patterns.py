"""The daily-pattern battery.

Every candidate here is a PUBLISHED calendar effect - the test list is fixed by
the literature, not mined from this sample, so the multiple-testing correction
is over a known, closed list:

  1. overnight vs intraday   (Cooper, Cliff & Gulen 2008 - the "night effect")
  2. day-of-week             (French 1980 - the Monday effect)
  3. turn-of-month           (Lakonishok & Smidt 1988)
  4. month-of-year           (January effect / Sell-in-May)

Discipline: every result is shown in THREE eras, with mean AND median, and a
month-clustered t. A pattern that only lives in one era is a regime, not a rule.
"""
import numpy as np
import pandas as pd


def load(path):
    d = pd.read_parquet(path)
    d.index = pd.to_datetime(d.index).tz_localize(None)
    out = pd.DataFrame(index=d.index)
    out["overnight"] = 1e4 * (d["Open"] / d["Close"].shift(1) - 1)
    out["intraday"] = 1e4 * (d["Close"] / d["Open"] - 1)
    out["full_day"] = 1e4 * (d["Close"] / d["Close"].shift(1) - 1)
    return out.dropna()


def mstats(s: pd.Series):
    """Mean, median, month-clustered t, hit rate, annualised bp."""
    per = s.groupby(s.index.to_period("M")).mean()
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 3 else np.nan
    return {"n": len(s), "mean_bp": round(float(s.mean()), 2),
            "med_bp": round(float(s.median()), 2),
            "t": round(float(t), 2), "hit": round(float((s > 0).mean()), 3),
            "ann_pct": round(float(s.mean() * 252 / 100), 1)}


ERAS = [("1993-2004", None, "2005"), ("2005-2015", "2005", "2016"),
        ("2016-now", "2016", None)]


def by_era(s: pd.Series):
    rows = []
    for lbl, a, b in ERAS:
        x = s
        if a: x = x[x.index >= a]
        if b: x = x[x.index < b]
        if len(x) > 200:
            rows.append({"era": lbl, **mstats(x)})
    return pd.DataFrame(rows)
