"""The real test: TJR's strategy on TJR's instruments.

ES/NQ futures, CME trade dates, true Asia/London/NY sessions, and SMT measured
on NQ-vs-ES - the pair he actually teaches. Everything before this was measured
on US equities, which have no Asia session and are not what he trades.
"""
import itertools
import numpy as np, pandas as pd
import mtf, backtest, futures as F
from pooled import clustered_t

U = F.load("_1h")   # hourly: 2.4 years, the only set with real power
SPLIT = pd.Timestamp("2025-12-01", tz="America/New_York")
# his SMT pair: each index is referenced against its sibling
PARTNER = {"NQ=F": "ES=F", "ES=F": "NQ=F", "YM=F": "ES=F", "RTY=F": "ES=F"}

CANON = dict(level_kinds=("day", "session", "eq"), zone_mode="fvg", zone_tf="4h",
             smt_filter=True, htf_bias="align", target_mode="level",
             target_R=2.0, futures=True)

GRID = dict(
    level_kinds=[("session",), ("day", "session"), ("day", "session", "eq"),
                 ("day", "session", "eq", "gap", "range")],
    zone_mode=["fvg", "bpr", "all"],
    zone_tf=["4h", "1d"],
    smt_filter=[True, False],
    htf_bias=["align", "off", "counter"],
    target_mode=["level", "R"],
)


def run_cell(kw, costs, lo=None, hi=None, tp1=0.0):
    p = mtf.MtfParams(futures=True, **{k: v for k, v in kw.items() if k != "futures"})
    rows = []
    for sym, df in U.items():
        ref = U.get(PARTNER.get(sym))
        for t in backtest.run(df, mtf.find_setups(df, p, ref), costs,
                              tp1_R=tp1, tp1_frac=0.5):
            ts = df.index[t.i_in]
            if (lo is None or ts >= lo) and (hi is None or ts < hi):
                rows.append((ts.normalize(), t.R))
    if not rows:
        return 0, np.nan, np.nan
    R = np.array([r for _, r in rows])
    m, t_ = clustered_t(R, np.array([d for d, _ in rows]))
    return len(R), m, t_


if __name__ == "__main__":
    # futures cost: ~1 tick round turn on NQ/ES is roughly 1-2 bp of notional
    C = backtest.Costs(slippage_bp=1.0)
    FREE = backtest.Costs(slippage_bp=0.0, slippage=0.0)
    print("=" * 76)
    print("CANONICAL TJR CONFIG - ON HIS ACTUAL INSTRUMENTS")
    print("=" * 76)
    for lbl, tp1 in [("all-or-nothing", 0.0), ("50% off at 1R", 1.0)]:
        ni, mi, ti = run_cell(CANON, C, hi=SPLIT, tp1=tp1)
        no, mo, to = run_cell(CANON, C, lo=SPLIT, tp1=tp1)
        ng, mg, tg = run_cell(CANON, FREE, tp1=tp1)
        print(f"{lbl:16s} IS n={ni:4d} exp={mi:+.3f} t={ti:+.2f} | "
              f"OOS n={no:4d} exp={mo:+.3f} t={to:+.2f} | GROSS n={ng:4d} exp={mg:+.4f}")

    print()
    print("=" * 76)
    print("GRID")
    print("=" * 76)
    keys = list(GRID)
    rows = []
    for vals in itertools.product(*(GRID[k] for k in keys)):
        kw = dict(zip(keys, vals))
        n_is, m_is, _ = run_cell(kw, C, hi=SPLIT)
        n_os, m_os, t_os = run_cell(kw, C, lo=SPLIT)
        n_g, m_g, _ = run_cell(kw, FREE)
        rows.append({**{k: str(v) for k, v in kw.items()}, "n_is": n_is,
                     "exp_is": m_is, "n_os": n_os, "exp_os": m_os,
                     "t_os": t_os, "gross": m_g})
    r = pd.DataFrame(rows)
    r.to_csv("data/futures_sweep.csv", index=False)
    a = r[(r.n_is >= 40) & (r.n_os >= 20)]
    print(f"cells {len(r)}  usable {len(a)}  median trades IS {a.n_is.median():.0f} OOS {a.n_os.median():.0f}")
    print(f"IN-SAMPLE     median {a.exp_is.median():+.3f}  positive {(a.exp_is>0).sum()}/{len(a)}")
    print(f"OUT-OF-SAMPLE median {a.exp_os.median():+.3f}  positive {(a.exp_os>0).sum()}/{len(a)}")
    print(f"GROSS         median {a.gross.median():+.4f}  positive {(a.gross>0).sum()}/{len(a)}")
    print(f"OOS t>2: {(a.t_os>2).sum()}/{len(a)}   IS->OOS spearman "
          f"{a[['exp_is','exp_os']].corr(method='spearman').iloc[0,1]:+.3f}")
