# Results — current state (2026-08-24)

Superseded analyses live in `archive/`. Entry points: `src/mtf_futures.py`
(the final strategy verdict), `src/mtf_v4.py` (equities), `src/validate.py`
(run before trusting anything: 30 checks + 31 pytest tests, all green).

## The final verdict on the TJR strategy
Tested at FULL FIDELITY (all 22 concepts, `FIDELITY.md`, 0 TODO) on his actual
instruments (NQ/ES/YM/RTY hourly futures, 2024-04 → 2026-08, CME trade dates,
true Asia/London/NY sessions, SMT on the real NQ↔ES pair):

**GROSS EXPECTANCY = ZERO.** Canonical config gross −0.027R; 254-cell grid gross
median +0.003, 0/254 cells significant OOS. Same answer independently on
equities (gross −0.0019 on 993 trades — reproduces exactly after every bug fix).
The strategy loses transaction costs and nothing more.

Supporting negatives (all well-powered):
- Session spillover Asia→London→NY: nothing (n=597 trade dates)
- Session sweep-and-reverse: no support vs geometry-preserving null
- Trend-break tells: SMT/RSI/stretch are INVERTED (predict continuation)
- 8-K corporate events: nothing across 250,189 events (max |t| 1.87, bar ≈3.2)
- Earnings-surprise direction: nothing; the 1-day "miss bounce" = bid-ask bounce

## What survived every test
1. **The overnight effect** — essentially all equity return accrues close→open;
   the trading session returns ≈ zero. Stable across 3 eras, 3 ETFs, and
   independently in futures (NQ overnight +6.26bp/day t=2.1 vs RTH −0.27 t≈0).
   Best-paid nights: Mon→Tue (+6.6bp, t=4.13, era-stable) and Tue→Wed.
   Practical: buy at the close, not the open; early-week closes are best.
2. **FOMC event-day volatility** — 1.37× baseline range, t=3.87 (reproduces).
   Direction on those days: nothing. Size positions accordingly.
3. **Momentum** — median +40bp/mo across 1,314 stocks × 15yr, but skew −10.8
   (Nov-2020 alone = −142%). Real, crash-prone, not naively tradeable.

## The bug ledger (10) — why the validator exists
1 unconsumed session levels → 5.7:1 sweep bias · 2 id(df) cache collisions ·
3 taxonomy circularity (12x fake effect) · 4 session lookahead (+0.55R fake edge)
· 5 near-zero-risk outliers (R=+812) · 6 silent SMT no-op filter · 7 broken
session-shuffle null · 8 too-narrow earnings null · 9 partial-exit same-bar
optimism · 10 Costs slippage_bp=0.0 fallback trap (API fixed: None vs 0.0).
Standing lessons: a spectacular result is a bug report; never quote a mean alone
(3 fake edges came from means); never claim a data limitation without running
the command.

## Hard limits, stated
- 5m intraday futures: only 53 canonical setups in 59 free days — UNRESOLVED.
- Distress shorts (restatements/bankruptcy): UNTESTABLE on free data —
  survivorship (7 bankruptcies in 250k events; the dead are absent).
- Universe is current index membership; bottom-bucket-beats-SPY (+118bp/mo) is
  the measured size of that bias.
