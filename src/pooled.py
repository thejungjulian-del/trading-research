"""Pool setups across correlated index ETFs, with day-clustered standard errors.

QQQ/SPY/IWM/DIA are ~90% correlated. A setup firing on all four at 10:15 is ONE
observation, not four. Treating them as independent inflates the t-stat by ~2x -
the same clustering error that had to be corrected in the Kalshi work.
"""
import sys
import numpy as np
import pandas as pd

import strategy, backtest

SYMBOLS = ["QQQ", "SPY", "IWM", "DIA"]


def clustered_t(R: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """Cluster-robust SE for a sample mean. Returns (mean, t)."""
    m = R.mean()
    e = R - m
    n = len(R)
    ss = sum(e[cluster == g].sum() ** 2 for g in np.unique(cluster))
    se = np.sqrt(ss) / n
    return float(m), (float(m / se) if se > 0 else np.nan)


def collect(p: strategy.Params, costs=backtest.Costs()):
    rows = []
    for sym in SYMBOLS:
        df = pd.read_parquet(f"data/{sym}_5m.parquet")
        for t in backtest.run(df, strategy.find_setups(df, p), costs):
            rows.append({"sym": sym, "ts": df.index[t.i_in], "R": t.R,
                         "side": t.side, "reason": t.reason})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tr = collect(strategy.Params())
    if tr.empty:
        sys.exit("no trades")
    tr["day"] = tr["ts"].dt.normalize()
    R = tr["R"].to_numpy()

    naive_se = R.std(ddof=1) / np.sqrt(len(R))
    m, t_cl = clustered_t(R, tr["day"].to_numpy())

    print(f"pooled trades      : {len(R)}  across {tr['sym'].nunique()} symbols, "
          f"{tr['day'].nunique()} distinct days")
    print(f"win rate           : {(R > 0).mean():.3f}")
    print(f"expectancy         : {m:+.4f} R")
    print(f"t (naive, WRONG)   : {m / naive_se:+.2f}")
    print(f"t (day-clustered)  : {t_cl:+.2f}   <- the honest one")
    print(f"total R            : {R.sum():+.2f}")
    print("\nby symbol:")
    print(tr.groupby("sym")["R"].agg(n="size", mean="mean", total="sum").round(3))
    print("\nby side (1=long, -1=short):")
    print(tr.groupby("side")["R"].agg(n="size", mean="mean", total="sum").round(3))
    print("\nby month:")
    print(tr.groupby(tr["ts"].dt.to_period("M"))["R"]
            .agg(n="size", mean="mean", total="sum").round(3))
