# Event study: which scheduled events actually cause a predictable pattern

Harness: `src/eventstudy.py` (event-agnostic) + `src/events.py` (calendars).
Any dated event list + any price series -> mean abnormal return path, t-stats,
and a random-date control run through identical machinery.

Why scheduled events only, for now: the release timestamp is published in
advance by the issuing agency, so there is no way to contaminate the test with
hindsight. Unscheduled news is where every lookahead trap lives (news APIs
return last-*updated* times, articles get backfilled and re-scored).

## Test 1 - FOMC, SPY, 2010-2026 (n=132 announcements)

### Direction: NOTHING

| rel day | mean abnormal | t |
|---|---|---|
| -1 | +4.4 bp | 0.35 |
| **0 (announcement)** | **-6.1 bp** | **-0.46** |
| +1 | -13.7 bp | -0.94 |

Day -4 printed +26.6 bp at t=2.41, which looks tradeable. It is not.
**Random-date control: 14.4% of random 132-date sets produce a |t| >= 2.41
somewhere in an 11-day window** (median max|t| = 1.86, p95 = 2.79). One in seven
by chance. This is the multiple-testing correction measured empirically rather
than assumed, and it is the single most important guard in this whole project.

Also note: the famous **pre-FOMC announcement drift** (Lucca & Moench, J.Finance
2015, ~49bp in the 24h before announcements, sample 1994-2011) is **absent** in
2010-2026. Consistent with the well-documented post-publication decay. Treat any
published anomaly as a decaying asset.

### Volatility: LARGE AND REAL

Baseline SPY intraday range 112 bp, |daily move| 72 bp.

| rel day | range vs baseline | \|move\| vs baseline | t (range) |
|---|---|---|---|
| -1 | 1.07x | 1.12x | +0.64 |
| **0** | **1.37x** | **1.30x** | **+3.87** |
| +1 | 1.28x | 1.44x | +3.07 |
| +2 | 1.20x | 1.21x | +2.17 |

t = 3.87 clears the control's p95 of 2.79, and it lands on the event day itself -
the pre-registered obvious place, not a fished-out one. Realized vol does NOT
build in advance (days -3..-1 are flat); it arrives on the day and decays over
about three days.

**The finding: direction around scheduled events is efficiently priced;
volatility is not.** That is where the predictable pattern lives.

### What this does and does not license
Actionable now, no extra data: on FOMC day you are carrying ~1.37x your usual
risk at the same stop distance. Position sizing and stop width should account
for it.
NOT established: that this is tradeable. Elevated *realized* vol says nothing
about whether *implied* vol already prices it. Testing that needs options data
(the variance risk premium generally favours sellers, but that is a separate
test we have not run).

## Next events to point the harness at
1. CPI and NFP (08:30 ET) - same free-calendar treatment, ~180 events each since 2010
2. Earnings / PEAD - thousands of events, the most robust anomaly in the
   literature, single stocks
3. Intraday response around CPI/NFP - needs bars only in a window around ~180
   events, a far smaller data ask than continuous history
4. Unscheduled headlines - last, and only with a publication-time delay buffer;
   if the edge dies when the news is delayed 30 min, the edge WAS the timestamp

## Standing rules for this project
- Pick events and horizons BEFORE looking
- Every result gets a random-date control through the same machinery
- Report the control's p95, not a nominal p-value
- Volatility findings and direction findings are different claims - do not blur them
