"""Systematic audit for the bug classes that have actually bitten this project.

Seven bugs so far, every one found by luck: a number looked too big, a gross/net
pair contradicted, an effect size was implausible. Luck is not a process. Each
check below turns one of those incidents into a repeatable assertion.

  1 determinism        - same input, same output
  2 causality          - deleting the future must not change the past
  3 cache correctness  - a cold run must equal a warm run
  4 cost monotonicity  - more slippage must monotonically hurt, and linearly
  5 outlier domination - no single trade may carry the mean
  6 directional balance- sweep detection must not be biased long or short
  7 control sanity     - random entries must land near zero, never strongly +ve
  8 split integrity    - in-sample and out-of-sample must partition, not overlap
  9 level availability - a level must be NaN until the bar it becomes knowable
 10 no free lunch      - gross must exceed net by roughly the modelled cost
"""
import sys
import numpy as np
import pandas as pd

import detectors as D, mtf, backtest, strategy

FAIL, WARN, OK = "FAIL", "WARN", "ok"
results = []


def check(name, status, detail=""):
    results.append((status, name, detail))
    tag = {FAIL: "FAIL", WARN: "WARN", OK: " ok "}[status]
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))


# ------------------------------------------------------------------ 1,3
def check_determinism_and_cache(df, p, ref=None):
    mtf._C.clear()
    a = mtf.find_setups(df, p, ref)          # cold
    b = mtf.find_setups(df, p, ref)          # warm (cached)
    mtf._C.clear()
    c = mtf.find_setups(df, p, ref)          # cold again
    same_ab = [(s.i_entry, s.side, s.stop) for s in a] == \
              [(s.i_entry, s.side, s.stop) for s in b]
    same_ac = [(s.i_entry, s.side, s.stop) for s in a] == \
              [(s.i_entry, s.side, s.stop) for s in c]
    check("determinism (repeat run identical)", OK if same_ac else FAIL,
          f"{len(a)} setups")
    check("cache correctness (cold == warm)", OK if same_ab else FAIL,
          "cached result differs from freshly computed" if not same_ab else "")


# -------------------------------------------------------------------- 2
def check_causality(df, p, ref=None):
    k = int(len(df) * 0.7)
    guard = 300
    full = [s for s in mtf.find_setups(df, p, ref) if s.i_entry < k - guard]
    trunc = [s for s in mtf.find_setups(df.iloc[:k], p,
                                        ref.iloc[:k] if ref is not None else None)
             if s.i_entry < k - guard]
    ok = ([(s.i_entry, s.side, round(s.stop, 6)) for s in full] ==
          [(s.i_entry, s.side, round(s.stop, 6)) for s in trunc])
    check("causality: setups unchanged when future deleted",
          OK if ok else FAIL, f"{len(full)} vs {len(trunc)} setups")


# -------------------------------------------------------------------- 9
def check_level_availability(df, futures=False):
    kinds = ("day", "session", "eq", "gap", "range")
    k = int(len(df) * 0.7)
    full = D.liquidity_levels(df, kinds, futures).head(k - 300)
    trunc = D.liquidity_levels(df.iloc[:k], kinds, futures).head(k - 300)
    bad = [c for c in full.columns
           if not full[c].equals(trunc[c])
           and not np.allclose(full[c].fillna(-9e9), trunc[c].fillna(-9e9))]
    check("levels: no column leaks future information",
          OK if not bad else FAIL, ", ".join(bad) if bad else "")


# ------------------------------------------------------------------ 4,10
def check_cost_monotonicity(df, setups):
    exps = []
    for bp in (0.0, 1.0, 2.0, 4.0):
        c = backtest.Costs(slippage_bp=bp, slippage=0.0)
        R = [t.R for t in backtest.run(df, setups, c)]
        exps.append(np.mean(R) if R else np.nan)
    mono = all(exps[i] >= exps[i + 1] - 1e-9 for i in range(len(exps) - 1))
    check("cost monotonicity (more slippage never helps)",
          OK if mono else FAIL, " -> ".join(f"{e:+.4f}" for e in exps))
    # linearity: each doubling of cost should roughly double the drag
    d1, d2 = exps[0] - exps[1], exps[0] - exps[3]
    ratio = d2 / d1 if abs(d1) > 1e-9 else np.nan
    lin = 3.0 <= ratio <= 5.0 if not np.isnan(ratio) else False
    check("cost drag scales ~linearly with slippage",
          OK if lin else WARN, f"4x cost gave {ratio:.1f}x drag (expect ~4)")


# -------------------------------------------------------------------- 5
def check_outliers(df, setups, costs):
    R = np.array([t.R for t in backtest.run(df, setups, costs)])
    if len(R) < 20:
        check("outlier domination", WARN, "too few trades to judge")
        return
    m = R.mean()
    top = np.abs(R - m).argsort()[::-1][:1]
    share = abs(R[top].sum() - m) / (abs(m) * len(R)) if abs(m) > 1e-9 else 0
    worst = float(np.abs(R).max())
    contrib = float(R[np.abs(R) > 10].sum() / len(R)) if (np.abs(R) > 10).any() else 0.0
    bad = worst > 20 or abs(contrib) > 0.5 * max(abs(m), 0.01)
    check("no single trade dominates the mean",
          FAIL if worst > 20 else (WARN if bad else OK),
          f"n={len(R)} mean={m:+.4f} max|R|={worst:.1f} "
          f"|R|>10 contributes {contrib:+.4f}")


# -------------------------------------------------------------------- 6
def check_direction_balance(df, kinds, futures=False):
    sw = D.liquidity_sweeps(df, kinds, futures)
    hi, lo = int(sw.swept_h.notna().sum()), int(sw.swept_l.notna().sum())
    if min(hi, lo) == 0:
        check("sweep direction balance", FAIL, f"{hi} high / {lo} low")
        return
    ratio = hi / lo
    check("sweep direction balance (no long/short bias)",
          OK if 0.6 <= ratio <= 1.67 else FAIL, f"{hi} high / {lo} low = {ratio:.2f}x")


# -------------------------------------------------------------------- 7
def check_control_sanity(df, setups, matched_fn=None):
    from mtf_control import matched_control
    ctrl, n = matched_control(setups, iters=25) if matched_fn else (None, None)
    if ctrl is None:
        check("random control sanity", WARN, "skipped")
        return
    m = float(ctrl.mean())
    check("random control lands near zero",
          OK if -0.15 < m < 0.15 else FAIL, f"random mean R = {m:+.4f}")


# -------------------------------------------------------------------- 8
def check_split_integrity(df, setups, split):
    ts = [df.index[s.i_entry] for s in setups if s.i_entry < len(df)]
    is_n = sum(1 for t in ts if t < split)
    os_n = sum(1 for t in ts if t >= split)
    check("split partitions cleanly (IS + OOS == total)",
          OK if is_n + os_n == len(ts) else FAIL, f"IS {is_n} + OOS {os_n} = {len(ts)}")
    check("both sides of the split are populated",
          OK if min(is_n, os_n) >= 20 else WARN, f"IS {is_n}, OOS {os_n}")


# ------------------------------------------------------ 11 data integrity
def check_data_integrity(frames):
    bad = []
    for sym, df in frames.items():
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        if (h < l).any():
            bad.append(f"{sym}: high<low")
        if (h < np.maximum(o, c) - 1e-9).any():
            bad.append(f"{sym}: high below open/close")
        if (l > np.minimum(o, c) + 1e-9).any():
            bad.append(f"{sym}: low above open/close")
        if (c <= 0).any():
            bad.append(f"{sym}: non-positive price")
        if df.index.duplicated().any():
            bad.append(f"{sym}: duplicate timestamps")
        if not df.index.is_monotonic_increasing:
            bad.append(f"{sym}: index not sorted")
        if df[["open", "high", "low", "close"]].isna().any().any():
            bad.append(f"{sym}: NaN in OHLC")
    check("OHLC data integrity across all symbols",
          OK if not bad else FAIL, "; ".join(bad[:4]))


# ------------------------------------------------- 12 stop/target geometry
def check_trade_geometry(frames, params, ref_map=None):
    bad = 0
    tot = 0
    for sym, df in frames.items():
        ref = frames.get(ref_map.get(sym)) if ref_map else None
        for s in mtf.find_setups(df, params, ref):
            tot += 1
            ref_px = df["close"].iat[s.i_entry]
            # a short must have its stop ABOVE and target BELOW, long the mirror
            if s.side == -1 and not (s.stop > ref_px and s.target < ref_px):
                bad += 1
            if s.side == 1 and not (s.stop < ref_px and s.target > ref_px):
                bad += 1
    check("stop/target on the correct sides of entry",
          OK if bad == 0 else FAIL, f"{bad} of {tot} setups malformed")


# ---------------------------------------------- 13 pessimistic tie-breaking
def check_tie_breaking(df):
    """A bar containing BOTH stop and target must resolve to the stop."""
    from mtf import Setup
    i = 200
    o = df["open"].iat[i + 1]
    hi, lo = df["high"].iat[i + 1], df["low"].iat[i + 1]
    if hi <= lo:
        check("pessimistic tie-breaking", WARN, "no usable bar")
        return
    # construct a short whose stop and target both sit inside the next bar
    s = Setup(i, -1, float(hi), float(lo), float(hi), float(hi))
    tr = backtest.run(df, [s], backtest.Costs(slippage_bp=0.0, slippage=0.0),
                      min_risk_bp=0.0)
    ok = bool(tr) and tr[0].reason == "stop"
    check("bar containing both stop and target resolves to STOP",
          OK if ok else FAIL, tr[0].reason if tr else "no trade produced")


# --------------------------------------------------- 14 filters actually filter
def check_filters_reduce(frames, base, ref_map=None):
    import dataclasses
    sym = list(frames)[0]
    df = frames[sym]
    ref = frames.get(ref_map.get(sym)) if ref_map else None
    n_base = len(mtf.find_setups(df, base, ref))
    n_bias = len(mtf.find_setups(df, dataclasses.replace(base, htf_bias="align"), ref))
    if ref is None:
        n_smt = None
    else:
        n_smt = len(mtf.find_setups(df, dataclasses.replace(base, smt_filter=True), ref))
    ok = (n_bias <= n_base) and (n_smt is None or n_smt < n_base)
    check("filters strictly reduce the setup count", OK if ok else FAIL,
          f"base {n_base}, +SMT {n_smt if n_smt is not None else 'n/a'}, +bias {n_bias}")


def check_smt_fails_loudly(frames, base):
    """A filter that silently does nothing is worse than one that errors."""
    import dataclasses
    df = frames[list(frames)[0]]
    try:
        mtf.find_setups(df, dataclasses.replace(base, smt_filter=True), None)
        check("smt_filter without a reference raises instead of no-oping",
              FAIL, "silently returned unfiltered setups")
    except ValueError:
        check("smt_filter without a reference raises instead of no-oping", OK)


def audit_universe(name, frames, params, ref_map=None, futures=False, split=None):
    print(f"\n{'='*74}\n{name}\n{'='*74}")
    sym = list(frames)[0]
    df = frames[sym]
    ref = frames.get(ref_map.get(sym)) if ref_map else None
    print(f"[{sym}]  {len(df):,} bars  {df.index[0].date()} -> {df.index[-1].date()}")

    check_data_integrity(frames)
    check_tie_breaking(df)
    check_determinism_and_cache(df, params, ref)
    check_causality(df, params, ref)
    check_level_availability(df, futures)
    check_direction_balance(df, params.level_kinds, futures)

    import dataclasses
    check_trade_geometry(frames, params, ref_map)
    base = dataclasses.replace(params, smt_filter=False, htf_bias="off")
    check_filters_reduce(frames, base, ref_map)
    check_smt_fails_loudly(frames, base)
    setups = mtf.find_setups(df, params, ref)
    if len(setups) < 10:
        check("enough setups to audit execution", WARN, f"only {len(setups)}")
        return
    check_cost_monotonicity(df, setups)
    check_outliers(df, setups, backtest.Costs(slippage_bp=1.0, slippage=0.0))
    if split is not None:
        check_split_integrity(df, setups, split)


if __name__ == "__main__":
    import futures as F
    from universe import load as load_stocks

    fut = F.load("_1h")
    audit_universe("FUTURES (hourly, CME sessions)", fut,
                   mtf.MtfParams(futures=True, zone_tf="4h",
                                 level_kinds=("day", "session", "eq"),
                                 smt_filter=True, htf_bias="align"),
                   ref_map={"NQ=F": "ES=F", "ES=F": "NQ=F",
                            "YM=F": "ES=F", "RTY=F": "ES=F"},
                   futures=True,
                   split=pd.Timestamp("2025-12-01", tz="America/New_York"))

    stk = load_stocks()
    audit_universe("EQUITIES (5m)", stk,
                   mtf.MtfParams(zone_tf="1h",
                                 level_kinds=("day", "session", "eq")),
                   futures=False,
                   split=pd.Timestamp("2026-07-21", tz="America/New_York"))

    print(f"\n{'='*74}")
    f = sum(1 for s, _, _ in results if s == FAIL)
    w = sum(1 for s, _, _ in results if s == WARN)
    print(f"{len(results)} checks   {f} FAIL   {w} WARN   {len(results)-f-w} ok")
    if f:
        print("\nFAILURES:")
        for s, n, d in results:
            if s == FAIL:
                print(f"  - {n}: {d}")
    sys.exit(1 if f else 0)
