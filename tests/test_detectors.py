import sys, pathlib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import detectors as D

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def bars(rows):
    """rows: list of (open, high, low, close)"""
    idx = pd.date_range("2026-01-05 09:30", periods=len(rows), freq="1min", tz="America/New_York")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


# ------------------------------------------------------------------ fvg
def test_bullish_fvg():
    # bar3.low (12) > bar1.high (10) -> bullish gap zone [10, 12]
    df = bars([(9, 10, 8, 9.5), (10, 14, 9.5, 13), (13, 15, 12, 14)])
    g = D.fvgs(df)
    assert len(g) == 1
    assert g[0].bull and g[0].i == 2 and g[0].lo == 10 and g[0].hi == 12


def test_bearish_fvg():
    # bar3.high (8) < bar1.low (10) -> bearish gap zone [8, 10]
    df = bars([(12, 13, 10, 11), (11, 11, 7, 8), (8, 8, 6, 7)])
    g = D.fvgs(df)
    assert len(g) == 1
    assert (not g[0].bull) and g[0].lo == 8 and g[0].hi == 10


def test_no_fvg_when_overlapping():
    df = bars([(9, 11, 8, 10), (10, 12, 9, 11), (11, 13, 10, 12)])
    assert D.fvgs(df) == []


# --------------------------------------------------------------- swings
def test_swing_high_confirm_lag():
    df = bars([(1, 2, 0, 1), (2, 3, 1, 2), (3, 9, 2, 8), (8, 4, 3, 3), (3, 3, 2, 2)])
    sh = [s for s in D.swings(df, n=2) if s.kind == "high"]
    assert len(sh) == 1
    assert sh[0].i == 2 and sh[0].price == 9
    assert sh[0].confirm_i == 4          # not knowable until 2 bars later


def test_flat_top_rejected():
    df = bars([(1, 2, 0, 1), (2, 5, 1, 4), (4, 5, 3, 4), (4, 3, 2, 2), (2, 2, 1, 1)])
    assert [s for s in D.swings(df, n=2) if s.kind == "high" and s.price == 5] == []


# ------------------------------------------------------------------ bos
def test_bos_fires_once_per_swing():
    # swing high of 9 at bar 2 (confirmed bar 4), then two closes above it
    df = bars([(1, 2, 0, 1), (2, 3, 1, 2), (3, 9, 2, 8), (8, 4, 3, 3), (3, 5, 2, 4),
               (4, 10, 4, 9.5), (9.5, 11, 9, 10.5)])
    r = D.structure_breaks(df, n=2)
    assert (r["bos"] == 1).sum() == 1
    assert r["bos"].to_numpy()[5] == 1   # first close above 9, not the second


def test_no_bos_before_confirmation():
    # price closes above the peak BEFORE the swing is confirmed -> no signal
    df = bars([(1, 2, 0, 1), (2, 3, 1, 2), (3, 9, 2, 8), (8, 12, 7, 11), (11, 12, 10, 11)])
    r = D.structure_breaks(df, n=2)
    assert (r["bos"] == 1).sum() == 0


# ----------------------------------------------------------------- ifvg
def test_bullish_gap_inverts_bearish():
    df = bars([(9, 10, 8, 9.5), (10, 14, 9.5, 13), (13, 15, 12, 14),   # bull gap [10,12]
               (14, 14, 9, 9.5)])                                       # closes below 10
    r = D.inverse_fvgs(df)
    assert r["ifvg"].to_numpy()[3] == -1


def test_gap_not_inverted_by_wick():
    df = bars([(9, 10, 8, 9.5), (10, 14, 9.5, 13), (13, 15, 12, 14),
               (14, 14, 9, 11)])                                        # wicks in, closes inside
    assert (D.inverse_fvgs(df)["ifvg"] != 0).sum() == 0


# ------------------------------------------------------------------ smt
def test_smt_bearish():
    a = bars([(1, 5, 0, 4)] * 3 + [(4, 9, 3, 8)])      # a makes a new high
    b = bars([(1, 5, 0, 4)] * 3 + [(4, 5, 3, 4)])      # b does not
    r = D.smt_divergence(a, b, lookback=3)
    assert r["smt"].to_numpy()[3] == -1


# ------------------------------------------------- causality (anti-lookahead)
@pytest.mark.parametrize("sym", ["QQQ", "SPY"])
def test_no_lookahead_on_real_data(sym):
    """Truncating the data must not change any signal on the bars that remain.
    This is the test that catches accidental future peeking."""
    p = DATA / f"{sym}_5m.parquet"
    if not p.exists():
        pytest.skip("run src/fetch_data.py first")
    df = pd.read_parquet(p).head(1200)
    k = 900
    for name, fn in [("bos", lambda d: D.structure_breaks(d, 2)["bos"]),
                     ("ifvg", lambda d: D.inverse_fvgs(d)["ifvg"]),
                     ("sess", lambda d: D.session_levels(d)["pdh"])]:
        full, trunc = fn(df), fn(df.head(k))
        pd.testing.assert_series_equal(
            full.head(k), trunc, check_names=False,
            obj=f"{name} changed when future bars were removed")


# ---------------------------------------------------------- order blocks
def test_order_block_is_causal_and_correct():
    # bar 14 is an UP candle, then price drops hard over the next 3 bars
    rows = [(10, 10.5, 9.5, 10)] * 14 + [(10, 11, 10, 10.9)] + \
           [(10.9, 11, 8, 8.2), (8.2, 8.3, 7, 7.1), (7.1, 7.2, 6, 6.1)] + \
           [(6.1, 6.2, 5.9, 6.0)] * 3
    df = bars(rows)
    obs = [b for b in D.order_blocks(df, impulse_atr=1.0, horizon=3) if not b.bull]
    assert any(b.i == 14 and b.lo == 10 and b.hi == 11 for b in obs)
    ob = [b for b in obs if b.i == 14][0]
    assert ob.confirm_i == 17          # not knowable until the impulse completes


def test_order_block_no_lookahead_on_real_data():
    p = DATA / "QQQ_5m.parquet"
    if not p.exists():
        pytest.skip("run src/fetch_data.py first")
    df = pd.read_parquet(p).head(1200)
    k = 900
    full = [b for b in D.order_blocks(df) if b.confirm_i < k]
    trunc = [b for b in D.order_blocks(df.head(k)) if b.confirm_i < k]
    assert full == trunc
