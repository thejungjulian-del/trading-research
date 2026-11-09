"""Large-universe momentum test: S&P 500 + 400 + 600, 15 years.

195 stocks over 6.5 years could not resolve momentum (t=0.49) - the published
effect is real but modest, so detecting it needs breadth and history. This is
7x the stocks and 2.5x the span.

SURVIVORSHIP: still a CURRENT-membership universe, so companies that failed or
were acquired are missing. The diagnostic is printed rather than hidden - if the
BOTTOM bucket beats SPY, that gap is the bias showing itself.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import momentum as M
import research as R

OUT = R.DATA / "big_prices.parquet"


def fetch(tickers, start="2010-01-01", chunk=200):
    frames = []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        d = yf.download(part + ["SPY"], start=start, auto_adjust=True,
                        progress=False, threads=True)["Close"]
        frames.append(d)
        print(f"  chunk {i//chunk+1}: {d.shape[1]} symbols", flush=True)
    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]
    # keep names with enough history and real liquidity
    px = px.dropna(axis=1, thresh=int(len(px) * 0.5))
    return px


if __name__ == "__main__":
    tick = [t.strip() for t in open("/tmp/universe_big.txt") if t.strip()]
    print(f"fetching {len(tick)} tickers from 2010...", flush=True)
    px = fetch(tick)
    px.to_parquet(OUT)
    print(f"kept {px.shape[1]} symbols x {px.shape[0]:,} days "
          f"({px.index[0].date()} -> {px.index[-1].date()})", flush=True)

    r = M.build(px, n_buckets=5)
    r.to_parquet(R.DATA / "big_momentum.parquet")
    SPLIT = pd.Timestamp("2019-01-01")
    print()
    print("CROSS-SECTIONAL MOMENTUM - large universe")
    for lbl, kw in [("FULL SAMPLE", {}), ("first half", dict(hi=SPLIT)),
                    ("second half (OOS)", dict(lo=SPLIT))]:
        s = M.summarise(r, 5, **kw)
        if not s:
            continue
        print(f"  {lbl:20s} months {s['months']:3d}  top {s['top_bp']:+7.1f}  "
              f"bottom {s['bottom_bp']:+7.1f}  spread {s['spread_bp']:+7.1f} bp  "
              f"t {s['t']:+5.2f}  hit {s['hit']:.2f}")
    s = M.summarise(r, 5)
    print()
    print(f"  SURVIVORSHIP DIAGNOSTIC: bottom bucket vs SPY = {s['bottom_bp']:+.1f} bp/mo")
    print("  (a bucket of losers beating the market IS the bias, measured)")
