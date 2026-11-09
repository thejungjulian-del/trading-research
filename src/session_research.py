"""Does one session's action predict the next one's?

Two questions, both straight from TJR's session model:
  A. SPILLOVER  - does Asia's move predict London's, London's predict New York's?
  B. SWEEP-AND-REVERSE - his central claim. When a session takes out the prior
     session's high, does it then REVERSE (the stop-grab) more often than it
     continues? If manipulation is real at session scale, this is where it shows.

Null for B is not a coin flip: it is the SAME statistic with sessions randomly
re-paired, which breaks the temporal link while keeping each session's own
distribution intact.
"""
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ET = "America/New_York"

# hourly-bar friendly windows (09:30 is not an hour boundary)
SESSIONS = {"asia": (18, 3), "london": (3, 9), "ny_am": (9, 12), "ny_pm": (12, 16)}
ORDER = ["asia", "london", "ny_am", "ny_pm"]


def fetch_hourly(sym="NQ=F", period="730d") -> pd.DataFrame:
    d = yf.download(sym, period=period, interval="1h", progress=False, prepost=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close", "Volume"]]
    d.columns = ["open", "high", "low", "close", "volume"]
    d.index = d.index.tz_convert(ET)
    return d.dropna()


def sessionize(df: pd.DataFrame) -> pd.DataFrame:
    h = df.index.hour
    sess = np.full(len(df), "", dtype=object)
    for name, (a, b) in SESSIONS.items():
        m = ((h >= a) | (h < b)) if a > b else ((h >= a) & (h < b))
        sess[m] = name
    td = (df.index + pd.Timedelta(hours=6)).normalize()   # CME trade date
    d = df.assign(sess=sess, td=td)
    d = d[d["sess"] != ""]
    g = d.groupby(["td", "sess"]).agg(
        o=("open", "first"), h=("high", "max"), l=("low", "min"),
        c=("close", "last"), v=("volume", "sum"))
    g["ret"] = 1e4 * (g["c"] / g["o"] - 1)
    w = g.unstack("sess")
    w.columns = [f"{b}_{a}" for a, b in w.columns]
    return w.dropna(subset=[f"{s}_c" for s in ORDER if f"{s}_c" in w.columns])


def spillover(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in [("asia", "london"), ("london", "ny_am"), ("ny_am", "ny_pm"),
                 ("asia", "ny_am"), ("london", "ny_pm")]:
        x, y = w[f"{a}_ret"], w[f"{b}_ret"]
        m = x.notna() & y.notna()
        x, y = x[m], y[m]
        if len(x) < 50:
            continue
        r = np.corrcoef(x, y)[0, 1]
        t = r * np.sqrt((len(x) - 2) / max(1 - r ** 2, 1e-9))
        rows.append({"predictor": a, "target": b, "n": len(x),
                     "corr": round(float(r), 3), "t": round(float(t), 2)})
    return pd.DataFrame(rows)


def sweep_reverse(w: pd.DataFrame, prior: str, nxt: str) -> dict:
    """Of the times `nxt` trades through `prior`'s high, how often does it CLOSE
    back below it (reversal) rather than above (continuation)?"""
    ph, pl = w[f"{prior}_h"], w[f"{prior}_l"]
    nh, nl, nc = w[f"{nxt}_h"], w[f"{nxt}_l"], w[f"{nxt}_c"]
    tookhigh = nh > ph
    tooklow = nl < pl
    rev_h = (nc < ph)[tookhigh]
    rev_l = (nc > pl)[tooklow]
    both = pd.concat([rev_h, rev_l])
    return {"pair": f"{prior}->{nxt}", "n_sweeps": int(len(both)),
            "reversal_rate": round(float(both.mean()), 3) if len(both) else np.nan}


def shuffle_null(w: pd.DataFrame, prior: str, nxt: str,
                 iters: int = 400, seed: int = 0) -> np.ndarray:
    """Null that preserves PRICE GEOMETRY.

    Naively permuting prior sessions across 2.4 years pairs today's session with
    a price level from a year ago, so the level is never reached and the null
    collapses to ~0.02 - which makes any real rate look like overwhelming
    support. Instead: express the prior extreme as a DISTANCE from the next
    session's open, in units of that session's own range, and shuffle the
    DISTANCES. Price level and volatility are preserved; only the identity of
    which prior session supplied the geometry is randomised.
    """
    rng = np.random.default_rng(seed)
    o = w[f"{nxt}_o"].to_numpy()
    nh, nl, nc = (w[f"{nxt}_{k}"].to_numpy() for k in ("h", "l", "c"))
    scale = np.maximum(nh - nl, 1e-9)
    dh = (w[f"{prior}_h"].to_numpy() - o) / scale
    dl = (o - w[f"{prior}_l"].to_numpy()) / scale
    out = []
    for _ in range(iters):
        i1, i2 = rng.permutation(len(o)), rng.permutation(len(o))
        ph = o + dh[i1] * scale
        pl = o - dl[i2] * scale
        th, tl = nh > ph, nl < pl
        vals = np.concatenate([(nc < ph)[th], (nc > pl)[tl]])
        if len(vals):
            out.append(vals.mean())
    return np.array(out)
