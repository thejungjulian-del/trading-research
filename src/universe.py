"""Volatile, liquid, retail-heavy single stocks.

Index ETFs were the wrong test bed: they are the most arbitraged instruments
that exist. The premise of the strategy is that retail stops cluster at obvious
levels and that running them produces a reflexive move - that lives in names
with heavy retail participation and real volatility.

Note on survivorship: the window is ~59 days, so almost nothing delists inside
it. The usual survivorship bias of a "current names" list is negligible here.
It would NOT be for a multi-year daily test.
"""
import warnings
from pathlib import Path
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent.parent / "data" / "stocks"
ET = "America/New_York"

TICKERS = """
TSLA NVDA AMD PLTR SOFI RIVN LCID MARA RIOT COIN HOOD GME AMC IONQ RGTI QBTS
SMCI ARM AFRM UPST DKNG ROKU SNAP PINS U RBLX NIO XPEV LI CVNA CHWY DASH ABNB
SHOP PYPL NET DDOG SNOW CRWD ZS MDB OKTA TWLO FSLR ENPH PLUG CHPT NKLA SPCE
JOBY ACHR LUNR RKLB ASTS SOUN AI HIMS OSCR CELH ELF LYFT UBER TTD APP MSTR
CLSK HUT WULF BITF CIFR
""".split()


def fetch(tickers=None, days: int = 59) -> dict:
    tickers = tickers or TICKERS
    DATA.mkdir(parents=True, exist_ok=True)
    raw = yf.download(tickers, period=f"{days}d", interval="5m", auto_adjust=False,
                      progress=False, group_by="ticker", threads=True, prepost=True)
    out = {}
    for t in tickers:
        try:
            x = raw[t].dropna()
        except Exception:
            continue
        x.columns = [c.lower() for c in x.columns]
        if not {"open", "high", "low", "close", "volume"} <= set(x.columns):
            continue
        x = x[["open", "high", "low", "close", "volume"]]
        if len(x) < 2000 or x["volume"].sum() == 0:
            continue
        x.index = x.index.tz_convert(ET)
        x.index.name = "ts"
        x.to_parquet(DATA / f"{t}.parquet")
        out[t] = x
    return out


def load() -> dict:
    return {p.stem: pd.read_parquet(p) for p in sorted(DATA.glob("*.parquet"))}


if __name__ == "__main__":
    d = fetch()
    print(f"{len(d)}/{len(TICKERS)} tickers usable")
    s = pd.Series({t: 1e4 * (x.high / x.low - 1).mean() for t, x in d.items()})
    print(f"median 5m bar range: {s.median():.0f} bp   "
          f"(QQQ for comparison is roughly 8-12 bp)")
    print("most volatile:", ", ".join(s.sort_values(ascending=False).head(6).index))
    any_df = next(iter(d.values()))
    print(f"bars per ticker ~{len(any_df)}  {any_df.index[0].date()} -> {any_df.index[-1].date()}")
