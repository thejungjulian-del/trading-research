# TJR fidelity checklist
Every concept taught in yiuFUp0kFz8, with implementation status.
Mentions = occurrences in `data/transcript_plain.txt` (72,817 words).

## Liquidity / draws (where stops sit, and where price is headed)
| # | concept | mentions | status |
|---|---|---|---|
| 1 | prior day high/low | 42 | DONE `liquidity_levels` |
| 2 | session highs/lows | 79 | DONE `session_extremes` (Asia impossible on US stocks) |
| 3 | relative equal highs/lows | 20 | DONE `equal_levels` |
| 4 | swing highs/lows | 6 | DONE `swings` |
| 5 | new day opening gap (NDOG) | 38 | DONE `opening_gaps` |
| 6 | new week opening gap (NWOG) | 38 | DONE `opening_gaps` |
| 7 | new candle opening gap | 38 | DONE (same mechanism, per session) |

## Imbalances / entry zones
| # | concept | mentions | status |
|---|---|---|---|
| 8 | fair value gap | 199 | DONE `fvgs` |
| 9 | inverse fair value gap | 11 | DONE `inverse_fvgs` |
| 10 | BPR (balanced price range) | - | DONE `bpr` |
| 11 | equilibrium / premium-discount | 68/33 | DONE `equilibrium` |
| 12 | breaker | 7 | DONE `breakers` |

## Structure
| # | concept | mentions | status |
|---|---|---|---|
| 13 | break of structure | 48 | DONE `structure_breaks` |
| 14 | order flow / change in order flow | 24 | DONE (BOS + IFVG) |
| 15 | consolidation / range | 72 | DONE `ranges` |

## Bias and confluence
| # | concept | mentions | status |
|---|---|---|---|
| 16 | SMT divergence | 42 | DONE - now a strategy filter (`smt_filter`) |
| 17 | higher-timeframe bias | 29 | DONE `htf_bias` (off/align/counter) |
| 18 | time theory / sessions | 2/79 | DONE - session levels + premarket data |

## Execution
| # | concept | mentions | status |
|---|---|---|---|
| 19 | stop above/below the sweep | 24 | DONE |
| 20 | target = draw on liquidity | 18 | DONE - nearest UN-TAKEN level from the full set |
| 21 | partial exits / scaling out | 13 | DONE `tp1_R` / `tp1_frac` |
| 22 | risk management / position sizing | 36 | DONE implicitly - all results are in R (risk multiples), which IS fixed-fractional sizing |

## Explicitly out of scope
- Prop-firm mechanics (45 mentions) - account rules, not a signal
- Trading psychology - not codeable
- Asia session on US equities - no volume exists; needs futures data
