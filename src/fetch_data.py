"""Fetch 1-minute bars for QQQ/SPY (free proxy for NQ/ES) and cache to parquet.

yfinance caps 1m history at ~30 days and 8 days per request, so we walk backwards
in 7-day windows and concatenate.
"""
import sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA = Path(__file__).resolve().parent.parent / "data"
ET = "America/New_York"


def fetch_1m(symbol: str, days: int = 29) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    floor = end - timedelta(days=days)
    frames = []
    while end > floor:
        start = max(end - timedelta(days=7), floor)
        df = yf.download(symbol, start=start, end=end, interval="1m",
                         auto_adjust=False, progress=False, prepost=True)
        if df is not None and len(df):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            frames.append(df)
        end = start
        time.sleep(1)
    if not frames:
        raise RuntimeError(f"no data for {symbol}")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out.index = out.index.tz_convert(ET)
    out = out[["Open", "High", "Low", "Close", "Volume"]]
    out.columns = ["open", "high", "low", "close", "volume"]
    out.index.name = "ts"
    return out.dropna()


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return out


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for sym in (sys.argv[1:] or ["QQQ", "SPY"]):
        df = fetch_1m(sym)
        df.to_parquet(DATA / f"{sym}_1m.parquet")
        five = resample(df, "5min")
        five.to_parquet(DATA / f"{sym}_5m.parquet")
        print(f"{sym}: {len(df):>7} 1m bars  {len(five):>6} 5m bars  "
              f"{df.index[0]} -> {df.index[-1]}")


def fetch_5m(symbol: str, days: int = 59) -> pd.DataFrame:
    """yfinance allows ~60 days of 5m history (vs 30 for 1m). Since the model
    now triggers on the 5m close, this is 3x the sample for free."""
    df = yf.download(symbol, period=f"{days}d", interval="5m",
                     auto_adjust=False, progress=False, prepost=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = df.index.tz_convert(ET)
    df.index.name = "ts"
    return df.dropna()
