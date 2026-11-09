"""Execution and cost accounting. Deliberately pessimistic:

  * fill at the OPEN of the bar AFTER the trigger, never at the signal price
  * slippage against you on entry and exit
  * if a bar's range contains both the stop and the target, the STOP is taken
  * unresolved positions are closed at the last bar and counted
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Costs:
    slippage: float = 0.01     # $/share each side (QQQ spread is ~1 cent)
    commission: float = 0.0    # $/share round turn
    slippage_bp: float | None = None   # if set, use bp of price each side; 0.0 = truly free

    def slip(self, px: float) -> float:
        # None = "not specified" -> per-share slippage. An explicit 0.0 means FREE.
        # (The old rule `if slippage_bp > 0` silently turned an explicit 0.0 back
        # into 1c/share - the trap behind the v4c "gross" discrepancy.)
        return px * self.slippage_bp * 1e-4 if self.slippage_bp is not None else self.slippage


@dataclass
class Trade:
    i_in: int
    i_out: int
    side: int
    entry: float
    exit: float
    stop: float
    target: float
    pnl: float        # net $/share
    R: float          # net, in units of initial risk
    reason: str


def run(df: pd.DataFrame, setups, costs: Costs = Costs(), max_hold: int = 78,
        tp1_R: float = 0.0, tp1_frac: float = 0.5, min_risk_bp: float = 10.0) -> list[Trade]:
    """tp1_R > 0 enables scaling out: take `tp1_frac` of the position at tp1_R,
    run the remainder to the full target. Both videos teach 50% at a first
    target; modelling everything all-or-nothing understates such a strategy."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(df)
    out = []
    for s in setups:
        i = s.i_entry + 1
        if i >= n:
            continue
        entry = o[i] + s.side * costs.slip(o[i])
        risk = abs(entry - s.stop)
        # The setup screens stop distance at the TRIGGER close, but we fill at
        # the NEXT bar's open, which can land almost on the stop. Dividing P&L by
        # a near-zero risk produces absurd R values - two such trades once
        # supplied an entire fake "edge" (max R = +812 on 1,009 trades).
        # A stop tighter than the spread is not a tradeable stop.
        if risk <= 0 or risk < min_risk_bp * 1e-4 * entry:
            continue
        tp1 = entry + s.side * tp1_R * risk if tp1_R > 0 else None
        booked = 0.0          # R already banked from the scaled-out portion
        frac = 1.0
        j, ex, why = i, None, ""
        while j < n and j - i < max_hold:
            hit_stop = (l[j] <= s.stop) if s.side == 1 else (h[j] >= s.stop)
            hit_tgt = (h[j] >= s.target) if s.side == 1 else (l[j] <= s.target)
            if hit_stop:                       # pessimistic: stop wins ties
                ex, why = s.stop, "stop"
                break
            if tp1 is not None and frac == 1.0:
                hit1 = (h[j] >= tp1) if s.side == 1 else (l[j] <= tp1)
                # When one bar sweeps BOTH tp1 and the full target, the partial
                # must still be taken at tp1 - crediting the whole position with
                # the far target is optimistic (bug #10).
                if hit1:
                    booked = tp1_frac * tp1_R
                    frac = 1.0 - tp1_frac
                    tp1 = None
            if hit_tgt:
                ex, why = s.target, "target"
                break
            j += 1
        if ex is None:
            j = min(j, n - 1)
            ex, why = c[j], "timeout"
        exit_px = ex - s.side * costs.slip(ex)
        pnl = s.side * (exit_px - entry) - costs.commission
        R = booked + frac * (pnl / risk)
        out.append(Trade(i, j, s.side, entry, exit_px, s.stop, s.target,
                         pnl, R, why))
    return out


def stats(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    R = np.array([t.R for t in trades])
    wins = R > 0
    exp = R.mean()
    se = R.std(ddof=1) / np.sqrt(len(R)) if len(R) > 1 else np.nan
    return {
        "n": len(R),
        "win_rate": round(float(wins.mean()), 3),
        "expectancy_R": round(float(exp), 4),
        "median_R": round(float(np.median(R)), 4),
        "stderr_R": round(float(se), 4),
        "t_stat": round(float(exp / se), 2) if se and se > 0 else np.nan,
        "total_R": round(float(R.sum()), 2),
        "worst_R": round(float(R.min()), 2),
        "best_R": round(float(R.max()), 2),
        "timeouts": sum(t.reason == "timeout" for t in trades),
    }


def random_control(df: pd.DataFrame, setups, costs: Costs = Costs(),
                   n_iter: int = 200, seed: int = 0) -> dict:
    """Same fill machinery, same count, same stop distances, random entry bars
    and random direction. Its expectancy should land at roughly -costs.
    If it comes out near zero, the cost model is broken; if positive, the
    exit logic itself is manufacturing edge."""
    from strategy import Setup
    rng = np.random.default_rng(seed)
    n = len(df)
    dists = [abs(df["close"].iat[s.i_entry] - s.stop) for s in setups]
    Rs = []
    for _ in range(n_iter):
        fake = []
        for d in dists:
            i = int(rng.integers(10, n - 100))
            side = int(rng.choice([-1, 1]))
            ref = df["close"].iat[i]
            fake.append(Setup(i, side, ref - side * d, ref + side * 2 * d, ref, "rand"))
        tr = run(df, fake, costs)
        if tr:
            Rs.append(np.mean([t.R for t in tr]))
    Rs = np.array(Rs)
    return {"mean_R": round(float(Rs.mean()), 4),
            "p05": round(float(np.percentile(Rs, 5)), 4),
            "p95": round(float(np.percentile(Rs, 95)), 4)}
