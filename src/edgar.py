"""SEC EDGAR events: material corporate disclosures with exact timestamps.

Better than scraped news for this purpose, on three counts:
  * TIMESTAMP is `acceptanceDateTime`, the second the filing hit the wire. News
    APIs return last-updated times that drift, which is how news backtests fake
    edges. Here there is nothing to drift.
  * COMPLETE - companies are legally required to disclose material events.
  * CATEGORISED - 8-K item codes say what actually happened.

Free, no API key, 10 requests/second with a declared User-Agent.
"""
import json, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "edgar"
# SEC asks for a contact in the User-Agent: set SEC_USER_AGENT="your-name your-email"
UA = {"User-Agent": os.environ.get("SEC_USER_AGENT", "trading-research")}

# 8-K item codes worth separating. The two flagged SHORT are classic
# accounting-trouble signals; 2.02 is the routine earnings release.
ITEMS = {
    "1.01": "material agreement",
    "1.03": "bankruptcy",
    "2.02": "results of operations (earnings)",
    "2.05": "costs of exit/disposal",
    "3.01": "delisting notice",
    "4.01": "auditor change            <- SHORT signal candidate",
    "4.02": "non-reliance on financials <- SHORT signal candidate",
    "5.02": "director/officer departure",
    "7.01": "Reg FD disclosure",
    "8.01": "other events",
}


def ticker_cik_map() -> dict:
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=UA, timeout=30)
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in r.json().values()}


def _rows(rec: dict) -> pd.DataFrame:
    n = len(rec.get("form", []))
    return pd.DataFrame({
        "form": rec.get("form", []),
        "filed": rec.get("filingDate", []),
        "accepted": rec.get("acceptanceDateTime", []),
        "items": rec.get("items", [""] * n),
        "accession": rec.get("accessionNumber", []),
    })


def filings(cik: str, deep: bool = True) -> pd.DataFrame:
    """All filings for a CIK. `deep` also pulls the archived older pages."""
    out = []
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                     headers=UA, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    d = r.json()
    out.append(_rows(d["filings"]["recent"]))
    if deep:
        for f in d["filings"].get("files", []):
            time.sleep(0.12)                      # stay under 10 req/s
            rr = requests.get(f"https://data.sec.gov/submissions/{f['name']}",
                              headers=UA, timeout=30)
            if rr.status_code == 200:
                out.append(_rows(rr.json()))
    df = pd.concat(out, ignore_index=True)
    df["accepted"] = pd.to_datetime(df["accepted"], errors="coerce", utc=True)
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    return df.dropna(subset=["accepted"])


def events_8k(df: pd.DataFrame) -> pd.DataFrame:
    """8-K rows exploded to one row per item code."""
    k = df[df["form"].astype(str).str.startswith("8-K")].copy()
    if not len(k):
        return pd.DataFrame()
    k["item_list"] = k["items"].fillna("").astype(str).str.split(",")
    k = k.explode("item_list")
    k["item"] = k["item_list"].str.strip()
    return k[k["item"] != ""][["form", "filed", "accepted", "item", "accession"]]


def to_entry_date(accepted: pd.Series) -> pd.Series:
    """Exact acceptance time -> the first session you could actually trade.

    Filed at or after 16:00 ET (or on a weekend) -> next session. Filed during
    the session -> we still wait for the NEXT open, which is deliberately
    conservative: it forfeits any same-day move rather than claim a fill nobody
    could have got.
    """
    et = accepted.dt.tz_convert("America/New_York")
    return (et + pd.Timedelta(days=1)).dt.normalize().dt.tz_localize(None)
