"""Overnight-premium strategies built from the validated findings.

  A NIGHT-OWL : hold overnight ONLY (buy close, sell next open) when the VIX
                sits below its trailing-252d 67th percentile; else cash.
  B CALM-RIDER: hold CONTINUOUSLY while VIX is calm; cash while stressed.
                Trades only on regime flips -> tiny cost drag.
  Benchmark   : buy & hold.

Causality: the VIX percentile at today's close decides tonight's/tomorrow's
position. Costs are charged per side on auction fills.
"""
import numpy as np
import pandas as pd


def run(px_path, vix, cost_bp=1.0, thresh=2/3):
    d = pd.read_parquet(px_path)
    d.index = pd.to_datetime(d.index).tz_localize(None) if d.index.tz else d.index
    on = (d["Open"] / d["Close"].shift(1) - 1).dropna()      # night t-1 -> t
    intr = (d["Close"] / d["Open"] - 1).reindex(on.index)
    full = (d["Close"] / d["Close"].shift(1) - 1).reindex(on.index)
    q = vix.reindex(on.index).ffill().rolling(252).rank(pct=True)
    calm = (q < thresh).shift(1).fillna(False)               # decided at PRIOR close
    c = cost_bp * 1e-4

    res = {}
    res["buy&hold"] = full
    # A: in the market only for calm nights; 2 auction fills per night held
    res["A night-owl"] = on.where(calm, 0.0) - calm.astype(float) * 2 * c
    # B: hold through calm days+nights; pay costs only when the regime flips
    pos = calm.astype(float)
    flips = pos.diff().abs().fillna(0)
    res["B calm-rider"] = full * pos - flips * c
    return pd.DataFrame(res).dropna(), int(calm.sum()), int(flips.sum())


def stats(r: pd.Series):
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": f"{cagr:+.1%}", "vol": f"{vol:.1%}",
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
            "maxDD": f"{dd:.0%}"}
