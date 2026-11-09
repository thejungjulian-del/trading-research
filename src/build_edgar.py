"""Pull 8-K material events for a universe and align them to forward returns."""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edgar as E, research as R

warnings.filterwarnings("ignore")
HORIZONS = [1, 5, 20, 60]


def build_events(tickers, deep=True):
    m = E.ticker_cik_map()
    rows = []
    miss = 0
    for i, t in enumerate(tickers):
        cik = m.get(t) or m.get(t.replace("-", "."))
        if not cik:
            miss += 1
            continue
        try:
            f = E.filings(cik, deep=deep)
        except Exception:
            continue
        k = E.events_8k(f)
        if len(k):
            k = k.copy()
            k["ticker"] = t
            rows.append(k)
        if i % 50 == 49:
            print(f"  {i+1}/{len(tickers)}  events so far: "
                  f"{sum(len(x) for x in rows):,}", flush=True)
        time.sleep(0.12)
    print(f"  unmapped tickers: {miss}", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def attach_returns(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    px = px.copy()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    spy = px["SPY"]
    ev = ev.copy()
    ev["target"] = E.to_entry_date(ev["accepted"])
    out = []
    for t, grp in ev.groupby("ticker"):
        if t not in px.columns:
            continue
        s = px[t].dropna()
        if len(s) < 300:
            continue
        m = spy.reindex(s.index)
        for _, r in grp.iterrows():
            j = s.index.searchsorted(r["target"])
            if j < 1 or j >= len(s) - max(HORIZONS) - 1:
                continue
            row = {"ticker": t, "item": r["item"], "accepted": r["accepted"],
                   "entry_date": s.index[j]}
            base, bm = s.iloc[j], m.iloc[j]
            ok = True
            for h in HORIZONS:
                if j + h >= len(s):
                    ok = False
                    break
                row[f"ar_{h}d"] = 1e4 * ((s.iloc[j + h] / base - 1) -
                                         (m.iloc[j + h] / bm - 1))
            if ok:
                out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    px = pd.read_parquet(R.DATA / "big_prices.parquet")
    tick = [c for c in px.columns if c != "SPY"][:600]
    print(f"pulling 8-K events for {len(tick)} tickers...", flush=True)
    ev = build_events(tick)
    E.DATA.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(E.DATA / "events_8k_raw.parquet")
    print(f"raw 8-K item rows: {len(ev):,}", flush=True)
    fr = attach_returns(ev, px)
    fr.to_parquet(E.DATA / "events_8k.parquet")
    print(f"usable events with forward windows: {len(fr):,}", flush=True)
    print(f"date range: {fr.entry_date.min().date()} -> {fr.entry_date.max().date()}",
          flush=True)
