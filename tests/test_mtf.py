import sys, pathlib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import detectors as D, mtf

STOCKS = pathlib.Path(__file__).resolve().parent.parent / "data" / "stocks"


def _frames(k=3):
    ps = sorted(STOCKS.glob("*.parquet"))[:k]
    if not ps:
        pytest.skip("run src/universe.py first")
    return {p.stem: pd.read_parquet(p) for p in ps}


def test_htf_zone_not_usable_before_its_bar_closes():
    """A 1h zone must not be usable at any 5m bar inside the 1h bar that made it."""
    df = next(iter(_frames(1).values()))
    p = mtf.MtfParams()
    for z in mtf.htf_zones(df, p)[:200]:
        # usable_from must land exactly on an hour boundary at/after the bar close
        assert z.usable_from.minute in (0, 30), z.usable_from
        # and nothing may reference it earlier
        assert z.usable_from > df.index[0]


def test_mtf_setups_are_causal():
    """Truncating future bars must not change any setup that already occurred.
    This is the test that catches multi-timeframe repainting."""
    p = mtf.MtfParams()
    for name, df in _frames(3).items():
        k = int(len(df) * 0.7)
        full = [s for s in mtf.find_setups(df, p) if s.i_entry < k - 30]
        trunc = [s for s in mtf.find_setups(df.iloc[:k], p) if s.i_entry < k - 30]
        assert len(full) == len(trunc), f"{name}: {len(full)} vs {len(trunc)}"
        for a, b in zip(full, trunc):
            assert (a.i_entry, a.side) == (b.i_entry, b.side), name
            assert abs(a.stop - b.stop) < 1e-9 and abs(a.target - b.target) < 1e-9


def test_levels_are_backward_looking():
    """Prior-day high on day D must equal the actual high of day D-1, never D."""
    df = next(iter(_frames(1).values()))
    lv = D.liquidity_levels(df, ("day",))
    daily = df.groupby(df.index.normalize()).agg(h=("high", "max"))
    days = list(daily.index)
    for d0, d1 in zip(days[:-1], days[1:]):
        got = lv.loc[lv.index.normalize() == d1, "pdh"].dropna().unique()
        if len(got):
            assert abs(got[0] - daily.loc[d0, "h"]) < 1e-9


def test_swept_level_is_consumed_once_per_session():
    """The bug from v1: a trending day must not re-register a sweep every bar."""
    df = next(iter(_frames(1).values()))
    sw = D.liquidity_sweeps(df, ("day",))
    per_day = sw.groupby(sw.index.normalize())["swept_h"].apply(lambda s: s.notna().sum())
    assert per_day.max() <= 1, f"prior-day high swept {per_day.max()} times in one session"


# ------------------------------------------- equal highs / lows + sessions
def test_equal_levels_are_causal():
    """An equal-high level must not exist before the cluster reaches its count."""
    df = next(iter(_frames(1).values()))
    k = int(len(df) * 0.6)
    full = [e for e in D.equal_levels(df) if e.confirm_i < k - 200]
    trunc = [e for e in D.equal_levels(df.iloc[:k]) if e.confirm_i < k - 200]
    assert full == trunc


def test_equal_level_cluster_is_within_tolerance():
    df = next(iter(_frames(1).values()))
    for e in D.equal_levels(df, tol_bp=15.0)[:300]:
        assert e.count >= 2
        assert e.kind in ("high", "low")


def test_equal_level_not_live_before_confirmation():
    df = next(iter(_frames(1).values()))
    lv = D.liquidity_levels(df, ("eq",))
    e = D.equal_levels(df)
    if not e:
        pytest.skip("no clusters")
    first = min(x.confirm_i for x in e)
    assert lv["eq_h"].iloc[:first].isna().all() or lv["eq_l"].iloc[:first].isna().all()


def test_session_extremes_not_known_during_session():
    """A session's high cannot be referenced by bars inside that session."""
    import datetime as dt
    df = next(iter(_frames(1).values()))
    se = D.session_extremes(df)
    if "ny_am_h" not in se.columns:
        pytest.skip("no ny_am bars")
    t = df.index.time
    inside = (t >= dt.time(9, 30)) & (t < dt.time(12, 0))
    assert se.loc[inside, "ny_am_h"].isna().all()


def test_session_extremes_not_known_BEFORE_session():
    """The bug that faked a huge edge: a 05:00 premarket bar must not be able to
    read the NY afternoon high from later the same day."""
    import datetime as dt
    df = next(iter(_frames(1).values()))
    se = D.session_extremes(df)
    if "ny_pm_h" not in se.columns:
        pytest.skip("no ny_pm bars")
    t = df.index.time
    before = t < dt.time(12, 0)
    assert se.loc[before, "ny_pm_h"].isna().all(), \
        "session high leaked backwards in time"


def test_every_level_column_is_causal():
    """Blanket guard: truncating the future must not change any level value."""
    df = next(iter(_frames(1).values()))
    k = int(len(df) * 0.7)
    kinds = ("day", "week", "n20", "eq", "session", "gap", "range")
    full = D.liquidity_levels(df, kinds).head(k - 300)
    trunc = D.liquidity_levels(df.iloc[:k], kinds).head(k - 300)
    for c in full.columns:
        pd.testing.assert_series_equal(full[c], trunc[c], check_names=False,
                                       obj=f"level {c} leaked future information")


# --------------------------- advanced imbalance concepts (NDOG/NWOG/BPR/etc)
def test_opening_gap_stamped_at_new_period_open():
    df = next(iter(_frames(1).values()))
    day = df.index.normalize()
    for g in D.opening_gaps(df)[:40]:
        # must sit on the FIRST bar of a session, never inside the prior one
        assert day[g.i] != day[g.i - 1]
        assert g.lo <= g.hi


def test_bpr_uses_later_of_the_two_gaps():
    """A balanced price range cannot be known before BOTH halves exist."""
    df = next(iter(_frames(1).values()))
    gs = D.fvgs(df)
    by_i = {}
    for g in gs:
        by_i.setdefault(g.i, []).append(g)
    for z in D.bpr(df)[:200]:
        assert z.i in by_i, "BPR must be stamped at a bar where a gap completed"
        assert z.lo < z.hi


def test_new_detectors_are_causal():
    """Truncating the future must not change any past zone."""
    df = next(iter(_frames(1).values()))
    k = int(len(df) * 0.7)
    for name, fn in [("opening_gaps", D.opening_gaps),
                     ("bpr", D.bpr),
                     ("breakers", D.breakers)]:
        full = [z for z in fn(df) if z.i < k - 250]
        trunc = [z for z in fn(df.iloc[:k]) if z.i < k - 250]
        assert full == trunc, f"{name} changed when future bars were removed"


def test_ranges_do_not_see_their_own_bar():
    df = next(iter(_frames(1).values()))
    r = D.ranges(df)
    k = int(len(df) * 0.7)
    full, trunc = D.ranges(df), D.ranges(df.iloc[:k])
    pd.testing.assert_series_equal(full["range_h"].head(k - 5), trunc["range_h"].head(k - 5))


# ------------------------------------------------- futures / CME sessions
def _fut():
    import futures as F
    d = F.load()
    if not d:
        pytest.skip("run src/futures.py first")
    return d


def test_asia_session_wraps_midnight_and_is_causal():
    """Asia runs 18:00->03:00 ET. Its extreme must be unavailable DURING the
    session (including past midnight) and available only after 03:00."""
    import datetime as dt
    nq = _fut()["NQ=F"]
    se = D.session_extremes(nq, D.FUTURES_SESSIONS, cme=True)
    t = nq.index.time
    during = (t >= dt.time(18, 0)) | (t < dt.time(3, 0))
    assert se.loc[during, "asia_h"].isna().all(), "Asia high leaked during Asia"
    after = (t >= dt.time(3, 0)) & (t < dt.time(18, 0))
    assert se.loc[after, "asia_h"].notna().any(), "Asia high never became available"


def test_futures_levels_have_no_lookahead():
    nq = _fut()["NQ=F"]
    k = int(len(nq) * 0.7)
    kinds = ("day", "session", "eq")
    full = D.liquidity_levels(nq, kinds, futures=True).head(k - 300)
    trunc = D.liquidity_levels(nq.iloc[:k], kinds, futures=True).head(k - 300)
    for c in full.columns:
        pd.testing.assert_series_equal(full[c], trunc[c], check_names=False,
                                       obj=f"futures level {c} leaked")


def test_cme_trade_date_starts_at_1800():
    nq = _fut()["NQ=F"]
    td = (nq.index + pd.Timedelta(hours=6)).normalize()
    i = nq.index.indexer_between_time("18:00", "18:05")[0]
    assert td[i] > nq.index[i].normalize(), "18:00 ET must roll to the next trade date"
