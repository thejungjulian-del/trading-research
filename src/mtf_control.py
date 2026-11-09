"""Random-entry control matched to the real setups.

Same universe, same number of trades, same stop distances (as a fraction of
price), same target multiples, same fill machinery - but random entry bars and
random direction. This is the yardstick: if the strategy's gross expectancy is
not distinguishably better than this, the pattern carries no information.
"""
import numpy as np, pandas as pd
import mtf, backtest
from universe import load

U = load()


def matched_control(p, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    real = {s: mtf.find_setups(df, p) for s, df in U.items()}
    profile = []                                     # (symbol, stop_frac, R_mult)
    for s, setups in real.items():
        df = U[s]
        for st in setups:
            ref = df["close"].iat[st.i_entry]
            profile.append((s, abs(ref - st.stop) / ref,
                            abs(st.target - ref) / max(abs(ref - st.stop), 1e-9)))
    if not profile:
        return None, None
    means = []
    for _ in range(iters):
        rows = []
        for s, sf, rm in profile:
            df = U[s]
            i = int(rng.integers(20, len(df) - 90))
            side = int(rng.choice([-1, 1]))
            ref = df["close"].iat[i]
            fake = mtf.Setup(i, side, ref - side * sf * ref,
                             ref + side * rm * sf * ref, ref, ref)
            rows += [t.R for t in backtest.run(df, [fake], backtest.Costs(slippage=0.0))]
        if rows:
            means.append(np.mean(rows))
    return np.array(means), len(profile)


if __name__ == "__main__":
    free = backtest.Costs(slippage=0.0, slippage_bp=0.0)
    print(f"{'config':28s} {'n':>5} {'strategy':>10} {'random':>10} {'p5..p95 of random':>22}")
    print("-" * 80)
    for name, p in [("1h fvg (default)", mtf.MtfParams()),
                    ("2h fvg", mtf.MtfParams(zone_tf="2h")),
                    ("1h order block", mtf.MtfParams(zone_mode="order_block")),
                    ("1h both, all levels", mtf.MtfParams(
                        zone_mode="both",
                        level_kinds=("day", "week", "n20", "round")))]:
        rows = []
        for s, df in U.items():
            rows += [t.R for t in backtest.run(df, mtf.find_setups(df, p), free)]
        strat = np.mean(rows) if rows else np.nan
        ctrl, n = matched_control(p)
        if ctrl is None:
            continue
        print(f"{name:28s} {n:5d} {strat:+10.4f} {ctrl.mean():+10.4f}   "
              f"[{np.percentile(ctrl,5):+.4f}, {np.percentile(ctrl,95):+.4f}]"
              f"{'  <- outside' if (strat<np.percentile(ctrl,5) or strat>np.percentile(ctrl,95)) else ''}")
