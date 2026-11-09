# TJR model — mechanical spec
Source: "How To Start Day Trading In 2026 [Full Tutorial]" (TJR, 6h40m, yiuFUp0kFz8)
Transcript: `data/transcript.txt` (10,261 timestamped lines, 72,817 words)

He states the model explicitly at ~06:25:27. Four steps, plus bias filter, stop, target.

## Step 0 — Bias (optional filter)
- Higher timeframe order flow (hourly/4h): are we above/below prior significant highs/lows.
- **SMT divergence** between NQ and ES: one index sweeps a high, the other fails to.
  The one that FAILED is the "leading index" — trade that one, in the direction of the failure.
  (Bearish SMT = NQ takes the high, ES doesn't → short ES.)

## Step 1 — Liquidity sweep ("potential for orders to be filled")
Price trades **above a significant swing high** (for shorts) or **below a significant swing low** (for longs).
- "Significant" = the subjective part. Candidates to formalize: prior day high/low,
  Asia session high/low, London session high/low, N-bar fractal swing on the 15m/1h.

## Step 2 — Change in order flow (confirmation the sweep was real)
On the **5-minute**, after the sweep, require ONE of:
- **Break of structure (BOS)** — close beyond the most recent opposing swing point, or
- **Inverse FVG (IFVG)** — a fair value gap in the sweep leg gets traded fully through and closed beyond.

FVG = 3-candle imbalance: candle1.high < candle3.low (bullish gap) or candle1.low > candle3.high (bearish gap).

## Step 3 — Continuation confluence (the pullback zone)
After the 5m structure break, wait for price to retrace into ONE of:
- an unfilled **FVG** left behind by the impulse leg, or
- **equilibrium** = the 50% level of the impulse leg (swing high → swing low).
  Above EQ = premium (only sell), below EQ = discount (only buy).

## Step 4 — Entry trigger
Do NOT enter on touch of the zone. Require confirmation *out of* the zone:
- **Conservative:** next 5m candle closes in the trade direction.
- **TJR's version:** drop to the **1-minute**. Inside the zone price will typically break 1m structure
  *against* the trade; enter when 1m structure breaks back *with* the trade (BOS or IFVG on 1m).

## Stop / Target
- **Stop:** beyond the Step-1 sweep extreme (above the swept high / below the swept low).
- **Target:** the next "draw on liquidity" — session lows/highs (Asia, London), prior-day extremes,
  hourly swing points. Scaled out across multiple draws (TP1 partial, runner to the far draw).

## Sessions (all US/Eastern)
Asia → London → New York AM → New York PM. RTH open 09:30 ET.
Session highs/lows are both sweep sources (Step 1) and targets.

---
## What is and isn't mechanical

| Step | Codeable? | Note |
|---|---|---|
| FVG / IFVG | yes | exact 3-bar definition |
| Equilibrium (50%) | yes | needs a swing-leg definition |
| BOS | yes | needs a swing-point definition (fractal N) |
| Session high/low, PDH/PDL | yes | exact |
| SMT divergence | yes | needs a lookback window |
| "**significant** high/low" | **no — free parameter** | biggest ambiguity; must be swept as a grid |
| "the next **draw on liquidity**" | **no — free parameter** | target selection changes R:R entirely |
| Position sizing / partials | partial | he teaches fixed-% risk; partials are discretionary |

Two free parameters (swing significance, target choice) control most of the P&L.
Any backtest must sweep both and report the *distribution* of results, not the best one.

## The measurement trap
Entries are 1-minute triggers on NQ/ES. Assuming fills at the trigger price is how the
Polymarket dip-bot produced a fake edge. Requirements for this backtest to mean anything:
- fill at next-bar open (or worse), never at the signal price
- explicit round-turn cost: commission + exchange fees + 1 tick slippage minimum
  (MNQ ~$0.50-1.50 RT commission; 1 tick MNQ = $0.50; NQ 1 tick = $5)
- stop-vs-target ambiguity inside a bar resolved pessimistically (stop first)
- out-of-sample split fixed BEFORE any parameter sweep
