"""Fetch earnings history + prices for a broad universe.

Survivorship note: this is a CURRENT universe, so delisted names are absent.
For PEAD that bias is much milder than for an absolute-return study, because we
compare high-surprise against low-surprise WITHIN the same universe and both
groups carry the same survivorship. It is still a real caveat and stays stated.
"""
import sys
from pathlib import Path
import pandas as pd

import research as R
from universe import TICKERS as VOLATILE

LARGE = """
AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA BRK-B JPM WMT LLY V UNH XOM MA ORCL
COST PG JNJ HD ABBV BAC KO NFLX CRM CVX MRK AMD PEP TMO ADBE LIN CSCO ACN MCD
ABT DIS WFC GE CAT VZ TXN INTU IBM QCOM CMCSA NOW AMGN PFE DHR UNP SPGI PM
NEO RTX HON LOW UPS BKNG ELV SBUX GS BLK MS ISRG PLD AMT SYK MDT ADP GILD
DE TJX VRTX C SCHW LMT MMC CB ZTS MO REGN BSX SO PGR DUK EOG SLB ITW APD
NSC CSX FDX EMR MET AIG PRU ALL TRV AFL WELL EQIX SPG O PSA CCI DLR VICI
KMB CL GIS K HSY SYY KR DG DLTR ROST ORLY AZO YUM CMG DRI MAR HLT RCL CCL
""".split()

if __name__ == "__main__":
    tick = sorted(set(LARGE + VOLATILE))
    R.DATA.mkdir(parents=True, exist_ok=True)
    print(f"universe: {len(tick)} tickers", flush=True)

    print("fetching earnings history...", flush=True)
    ev = R.fetch_earnings(tick, limit=24)
    ev.to_parquet(R.DATA / "earnings.parquet")
    print(f"  {len(ev):,} announcements across {ev.ticker.nunique()} tickers  "
          f"{ev.announced.min().date()} -> {ev.announced.max().date()}", flush=True)

    print("fetching prices...", flush=True)
    px = R.fetch_prices(sorted(ev.ticker.unique()), start="2019-01-01")
    px.to_parquet(R.DATA / "prices.parquet")
    print(f"  {px.shape[0]:,} days x {px.shape[1]} symbols", flush=True)

    print("computing market-adjusted forward returns...", flush=True)
    fr = R.forward_returns(ev, px)
    fr.to_parquet(R.DATA / "events.parquet")
    print(f"  {len(fr):,} usable events with full forward windows", flush=True)
