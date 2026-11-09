"""Scheduled-event calendars. Every date here is published in advance by the
issuing agency, so there is no timestamp-contamination risk - the thing that
makes unscheduled-news backtests untrustworthy.
"""
import re, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "events"
UA = {"User-Agent": "Mozilla/5.0"}


def fomc_dates(start_year: int = 2000, end_year: int = 2026) -> pd.DataFrame:
    """FOMC statement dates, scraped from the press-release URLs the Fed's own
    calendar pages link to (monetaryYYYYMMDDa.htm). Statements are released at
    14:00 ET on the final day of each meeting."""
    urls = ["https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"]
    urls += [f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
             for y in range(start_year, 2021)]
    seen = set()
    for u in urls:
        try:
            r = requests.get(u, headers=UA, timeout=30)
        except Exception:
            continue
        if r.status_code != 200:
            continue
        seen |= {m for m in re.findall(r"monetary(\d{8})", r.text)}
        time.sleep(0.3)
    d = sorted(pd.to_datetime(sorted(seen), format="%Y%m%d"))
    df = pd.DataFrame({"date": d})
    df = df[(df.date.dt.year >= start_year) & (df.date.dt.year <= end_year)]
    df["event"] = "FOMC"
    df["time_et"] = "14:00"
    return df.reset_index(drop=True)


if __name__ == "__main__":
    EV.mkdir(exist_ok=True)
    f = fomc_dates()
    f.to_csv(EV / "fomc.csv", index=False)
    print(f"FOMC: {len(f)} announcements  {f.date.min().date()} -> {f.date.max().date()}")
    print(f.groupby(f.date.dt.year).size().tail(8).to_string())
