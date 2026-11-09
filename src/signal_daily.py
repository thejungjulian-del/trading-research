"""The daily driver: everything this project validated, as one evening signal.

Run near the close (cron 15:45 ET). Prints exactly what to hold tonight:
  1. STOCK SLEEVE  - 3x QQQ overnight, ONLY above the 200dma (mined rule,
                     survived sealed train/val/test + replicated on crypto)
  2. CRYPTO SLEEVE - BTC/ETH long above their 200dma, flat below
  3. RISK DIAL     - dealer gamma (free CSV): low GEX => expect ~2x range tonight
  4. CALENDAR      - FOMC eve (hold: +25bp avg) / FOMC day (1.37x vol: size down)

No predictions, no patterns, no discretion. Signals change at most a few times
a year. This is the whole edge; everything else tested to zero.
"""
import datetime as dt
import warnings
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")


def trend(sym):
    d = yf.download(sym, period="400d", progress=False)["Close"]
    d = d.iloc[:, 0] if hasattr(d, "columns") else d
    px, sma = float(d.iloc[-1]), float(d.rolling(200).mean().iloc[-1])
    return px, sma, px > sma


def gex_state():
    try:
        r = requests.get("https://squeezemetrics.com/monitor/static/DIX.csv",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        g = pd.read_csv(pd.io.common.StringIO(r.text), parse_dates=["date"])
        q = g.gex.rolling(252).rank(pct=True).iloc[-1]
        return float(q), g.date.iloc[-1].date()
    except Exception:
        return None, None


def fomc_flags():
    try:
        ev = pd.read_csv("events/fomc.csv", parse_dates=["date"])
        today = pd.Timestamp.now().normalize()
        days = (ev.date.dt.normalize() - today).dt.days
        return bool((days == 1).any() or (days == 3).any()), bool((days == 0).any())
    except Exception:
        return False, False


if __name__ == "__main__":
    print(f"=== NIGHT SIGNAL  {dt.datetime.now():%Y-%m-%d %H:%M} ===")
    px, sma, up = trend("QQQ")
    print(f"QQQ {px:,.2f} vs 200dma {sma:,.2f}  ->  "
          f"{'IN: hold 3x overnight (buy MOC, sell MOO)' if up else 'OUT: flat tonight'}")
    for s in ("BTC-USD", "ETH-USD"):
        p2, s2, u2 = trend(s)
        print(f"{s[:3]} {p2:,.0f} vs 200dma {s2:,.0f}  ->  {'LONG' if u2 else 'FLAT'}")
    q, asof = gex_state()
    if q is not None:
        band = "LOW (short gamma: expect ~2x range tonight - do not oversize)" \
            if q < 1/3 else ("HIGH (long gamma: quiet tape likely)" if q > 2/3 else "MID")
        print(f"GEX percentile {q:.0%} (as of {asof})  ->  {band}")
    eve, day = fomc_flags()
    if eve:
        print("FOMC EVE: historically the best-paid night (+25bp avg) - hold.")
    if day:
        print("FOMC DAY: 1.37x normal range - size down or widen stops.")
    print("Rules: no intraday trades. No exceptions. The day session pays zero.")
