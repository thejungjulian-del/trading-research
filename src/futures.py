"""CME index futures with TRUE 24-hour session structure.

This is what TJR actually trades, and it is what makes his session model
testable: US equities have no Asia session, but ES/NQ trade nearly around the
clock and yfinance serves them intraday for free.

CME "trade date" starts at 18:00 ET the previous calendar day, so sessions are
grouped on that basis, not on midnight.
"""
import datetime as dt
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent.parent / "data" / "futures"
ET = "America/New_York"

SYMBOLS = {"NQ=F": "Nasdaq", "ES=F": "S&P500", "YM=F": "Dow", "RTY=F": "Russell"}

# ET windows. Asia spans midnight, so it is handled separately below.
SESSIONS = {
    "asia":   (dt.time(18, 0), dt.time(3, 0)),
    "london": (dt.time(3, 0),  dt.time(9, 30)),
    "ny_am":  (dt.time(9, 30), dt.time(12, 0)),
    "ny_pm":  (dt.time(12, 0), dt.time(16, 0)),
}
ORDER = ["asia", "london", "ny_am", "ny_pm"]


def trade_date(idx: pd.DatetimeIndex) -> pd.Series:
    """CME session date: everything from 18:00 ET belongs to the NEXT day."""
    shifted = idx + pd.Timedelta(hours=6)      # 18:00 ET -> 00:00 next day
    return pd.Series(shifted.normalize(), index=idx)


def session_of(idx: pd.DatetimeIndex) -> pd.Series:
    t = idx.time
    out = np.full(len(idx), "", dtype=object)
    for name in ORDER:
        a, b = SESSIONS[name]
        m = ((t >= a) | (t < b)) if a > b else ((t >= a) & (t < b))
        out[m] = name
    return pd.Series(out, index=idx)


def fetch(days: int = 59, interval: str = "5m", tag: str = "") -> dict:
    """5m gives ~59 days; 1h gives ~730. The hourly set is the only one with
    enough history to test anything with statistical power."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for sym in SYMBOLS:
        d = yf.download(sym, period=f"{days}d", interval=interval,
                        progress=False, prepost=True)
        if d is None or not len(d):
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Open", "High", "Low", "Close", "Volume"]]
        d.columns = ["open", "high", "low", "close", "volume"]
        d.index = d.index.tz_convert(ET)
        d.index.name = "ts"
        d = d.dropna()
        d.to_parquet(DATA / f"{sym.replace('=','_')}{tag}.parquet")
        out[sym] = d
    return out


def load(tag: str = "") -> dict:
    pat = f"*{tag}.parquet" if tag else "*_F.parquet"
    return {p.stem.replace(tag, "").replace("_F", "=F"): pd.read_parquet(p)
            for p in sorted(DATA.glob(pat))}


def session_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (trade date, session): OHLC, return, range."""
    d = df.copy()
    d["td"] = trade_date(d.index).to_numpy()
    d["sess"] = session_of(d.index).to_numpy()
    d = d[d["sess"] != ""]
    g = d.groupby(["td", "sess"]).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), vol=("volume", "sum"))
    g["ret_bp"] = 1e4 * (g["close"] / g["open"] - 1)
    g["range_bp"] = 1e4 * (g["high"] / g["low"] - 1)
    return g.reset_index()


def wide(df: pd.DataFrame) -> pd.DataFrame:
    """Sessions as columns, one row per trade date - so you can ask whether
    what Asia did predicts what London or New York does next."""
    t = session_table(df)
    p = t.pivot(index="td", columns="sess")
    p.columns = [f"{b}_{a}" for a, b in p.columns]
    keep = [c for c in p.columns if c.split("_", 1)[0] in ORDER
            or c.rsplit("_", 1)[0] in ORDER]
    return p[[c for c in p.columns]].dropna(how="all")
