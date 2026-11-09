"""Full-fidelity TJR test. Every concept from FIDELITY.md is available here.

Reports two things:
  1. THE CANONICAL CONFIG - the strategy exactly as he teaches it
  2. a grid around it, so the answer cannot hinge on one lucky setting
"""
import itertools
import numpy as np, pandas as pd
import mtf, backtest
from universe import load
from pooled import clustered_t

SPLIT = pd.Timestamp("2026-07-21", tz="America/New_York")
U = load()
REF = pd.read_parquet("data/SPY_5m.parquet")

# the strategy as taught: his three most-mentioned level types, fair value gaps
# as the entry zone, SMT confirming, HTF order flow aligned, targeting the next
# un-taken draw, scaling half off at 1R.
CANON = dict(level_kinds=("day", "session", "eq"), zone_mode="fvg", zone_tf="1h",
             smt_filter=True, htf_bias="align", target_mode="level", target_R=2.0)

GRID = dict(
    level_kinds=[("day", "session", "eq"),
                 ("day", "session", "eq", "gap", "range"),
                 ("session",), ("eq",)],
    zone_mode=["fvg", "bpr", "all"],
    zone_tf=["1h", "2h"],
    smt_filter=[True, False],
    htf_bias=["align", "off", "counter"],
    target_mode=["level", "R"],
)


def run_cell(kw, costs, lo=None, hi=None, tp1=0.0):
    p = mtf.MtfParams(**kw)
    rows = []
    for sym, df in U.items():
        for t in backtest.run(df, mtf.find_setups(df, p, REF), costs,
                              tp1_R=tp1, tp1_frac=0.5):
            ts = df.index[t.i_in]
            if (lo is None or ts >= lo) and (hi is None or ts < hi):
                rows.append((ts.normalize(), t.R))
    if not rows:
        return 0, np.nan, np.nan
    R = np.array([r for _, r in rows])
    m, t_ = clustered_t(R, np.array([d for d, _ in rows]))
    return len(R), m, t_


def describe(R):
    """Mean is outlier-sensitive; always show the median beside it."""
    return f"mean={R.mean():+.4f} median={np.median(R):+.4f} max={R.max():+.1f}"


if __name__ == "__main__":
    C = backtest.Costs(slippage_bp=5.0)
    print("=" * 74)
    print("THE CANONICAL TJR CONFIGURATION")
    print("=" * 74)
    for lbl, tp1 in [("all-or-nothing", 0.0), ("50% off at 1R", 1.0)]:
        ni, mi, ti = run_cell(CANON, C, hi=SPLIT, tp1=tp1)
        no, mo, to = run_cell(CANON, C, lo=SPLIT, tp1=tp1)
        ng, mg, tg = run_cell(CANON, backtest.Costs(slippage_bp=0.0, slippage=0.0), tp1=tp1)
        print(f"{lbl:16s} IS n={ni:4d} exp={mi:+.3f} t={ti:+.2f} | "
              f"OOS n={no:4d} exp={mo:+.3f} t={to:+.2f} | gross(all) exp={mg:+.4f}")

    print()
    print("=" * 74)
    print("GRID AROUND IT")
    print("=" * 74)
    keys = list(GRID)
    rows = []
    for vals in itertools.product(*(GRID[k] for k in keys)):
        kw = dict(zip(keys, vals))
        n_is, m_is, _ = run_cell(kw, C, hi=SPLIT)
        n_os, m_os, t_os = run_cell(kw, C, lo=SPLIT)
        rows.append({**{k: str(v) for k, v in kw.items()},
                     "n_is": n_is, "exp_is": m_is, "n_os": n_os,
                     "exp_os": m_os, "t_os": t_os})
    r = pd.DataFrame(rows)
    r.to_csv("data/mtf_v4_sweep.csv", index=False)
    a = r[(r.n_is >= 40) & (r.n_os >= 20)]
    print(f"cells {len(r)}   usable {len(a)}   median trades IS {a.n_is.median():.0f} OOS {a.n_os.median():.0f}")
    print(f"IN-SAMPLE     median {a.exp_is.median():+.3f}   positive {(a.exp_is>0).sum()}/{len(a)}")
    print(f"OUT-OF-SAMPLE median {a.exp_os.median():+.3f}   positive {(a.exp_os>0).sum()}/{len(a)}")
    print(f"OOS cells t>2: {(a.t_os>2).sum()}/{len(a)}     IS->OOS spearman "
          f"{a[['exp_is','exp_os']].corr(method='spearman').iloc[0,1]:+.3f}")
    print()
    print("best OOS cells:")
    print(a.sort_values("exp_os", ascending=False).head(5)[
        ["level_kinds","zone_mode","smt_filter","htf_bias","target_mode","n_os","exp_os","t_os"]
    ].to_string(index=False))
