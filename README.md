# Trading research

Three market studies that share one codebase and one rule: decide the test before looking at the result, keep data the test hasn't seen, and always run a random-date or shuffled control.

- **Overnight premium.** Hold QQQ (Nasdaq-100) only overnight, and only while it trades above its 200-day average. Over 27 years of daily data, at 3× that returned +21.3% a year at a Sharpe of 0.85, against 0.49 for buy-and-hold. It was the 1 rule out of about 260 mined rules that passed all three test periods, and the same filter improved BTC, ETH and SOL. `src/signal_daily.py` prints tonight's position.
- **Fed day study.** An event-study harness for scheduled news (`src/eventstudy.py`, `src/events.py`). On 132 FOMC announcements from 2010 to 2026, the market didn't move in a predictable direction (−6.1bp, t = −0.46), and the pre-FOMC drift found in older papers is gone. The day's range was 1.37× normal (t = 3.87, against a random-date p95 of 2.79), for about three days. 250,189 SEC 8-K filings showed nothing. Write-up in `EVENTSTUDY.md`.
- **Day-trading course test.** I turned a 6-hour YouTube day-trading course into exact rules (`STRATEGY.md`, `FIDELITY.md`) and tested them on QQQ, SPY, IWM and DIA 5-minute bars, then on futures. All 288 settings made money in-sample. On the held-out month the median was −0.145R, and 0 of 288 were significant. `RESULTS.md` has the final verdict and the list of bugs that faked earlier edges.

## Run it

Python 3 with numpy, pandas, yfinance, requests, pyarrow and pytest. Data isn't included; the fetch scripts in `src/` download it from Yahoo, the Fed's site and SEC EDGAR into `data/`. For EDGAR, set `SEC_USER_AGENT` to your name and email (see `.env.example`), which the SEC asks for.

```
pip install numpy pandas yfinance requests pyarrow pytest
python src/fetch_data.py
python src/signal_daily.py
python src/validate.py
```

`src/validate.py` runs the audit checks (causality, determinism, cost direction, outliers) that exist because each one caught a real bug. `pytest tests` runs the detector tests once data is downloaded.

## Status

Research, not a trading system, and not advice. The overnight rule held up in every test I ran, but it's a backtest on one index and 3× leverage has deep drawdowns. The Fed result is about risk sizing, not a trade. The course rules failed out of sample and are closed.

Portfolio: https://thejungjulian-del.github.io/portfolio/
