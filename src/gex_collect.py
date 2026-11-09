"""Daily options-chain collector -> build our own dealer-gamma (GEX) history.

Historical options open interest is the one paid dataset blocking the real
"ride the market makers" test (dealer gamma hedging). Current chains are free,
so we record them daily and own the history in a few months - same
forward-collection play as the Polymarket bots.

Per underlying we store every listed strike/expiry with OI, volume, IV, and
compute a naive GEX summary (standard dealer convention: dealers long calls
customers sold... conventions vary - we store RAW data so any convention can be
recomputed later; the summary is just a convenience).
"""
import datetime as dt
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "chains"
UNDERLYINGS = ["SPY", "QQQ", "IWM"]


def snap(sym: str) -> pd.DataFrame | None:
    t = yf.Ticker(sym)
    spot = t.fast_info.get("lastPrice")
    rows = []
    try:
        exps = t.options[:8]                     # near expiries carry the gamma
    except Exception:
        return None
    for e in exps:
        try:
            ch = t.option_chain(e)
        except Exception:
            continue
        for side, df in (("C", ch.calls), ("P", ch.puts)):
            d = df[["strike", "openInterest", "volume",
                    "impliedVolatility", "lastPrice"]].copy()
            d["type"], d["expiry"] = side, e
            rows.append(d)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out["spot"] = spot
    out["asof"] = pd.Timestamp.now(tz="America/New_York").isoformat()
    return out


def gex_summary(d: pd.DataFrame) -> str:
    """Crude same-sign GEX: call OI adds positive dealer gamma, put OI negative
    (one common convention). Raw data is stored, so this is disposable."""
    spot = d["spot"].iloc[0]
    if not spot:
        return "no spot"
    near = d[(d.strike > spot * 0.9) & (d.strike < spot * 1.1)].copy()
    g = near.groupby("type")["openInterest"].sum()
    net = int(g.get("C", 0) - g.get("P", 0))
    pin = near.groupby("strike")["openInterest"].sum().idxmax()
    return f"net C-P OI (±10%): {net:+,}  max-OI strike: {pin}"


def healthy(d: pd.DataFrame) -> bool:
    """yfinance intermittently returns OI=0 across the board (OCC refresh gap).
    A snapshot recording zeros for months would poison the dataset silently -
    refuse to overwrite a healthy file with an unhealthy one."""
    return (d["openInterest"] > 0).mean() > 0.30


if __name__ == "__main__":
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    day = dt.date.today().isoformat()
    for sym in UNDERLYINGS:
        d = snap(sym)
        if d is None or not len(d):
            print(f"{sym}: FETCH FAILED")
            continue
        f = OUT / f"{sym}_{day}.parquet"
        ok = healthy(d)
        if not ok and f.exists():
            print(f"{sym}: UNHEALTHY (OI empty) - keeping earlier healthy file")
            continue
        d.to_parquet(f)
        tag = "ok" if ok else "UNHEALTHY-OI (kept; retry later run)"
        print(f"{sym}: {len(d):5d} contracts, OI>0 on "
              f"{(d.openInterest>0).mean():.0%} [{tag}]  {gex_summary(d)}")
