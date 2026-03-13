# Step 1 — Event Definition Catalog
## NQ Futures Event-Driven Research Framework
### Date: 2026-03-03 | Primary Data: 5-min OHLCV | Period: 2008–2026

---

## Research Parameters (from user constraints)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Session | RTH only (09:30–16:00 ET) | Overnight data used for context only |
| Holding period | Intraday swing (30 min – 4 hours) | ~6–48 bars on 5-min |
| Account size | $50,000 (paper → TopStep → live) | NQ = $20/pt, 1 tick = $5 |
| Max drawdown | 5% ($2,500) | TopStep 50k trailing DD |
| Max risk/trade | 1–2% ($500–$1,000) | ~100–200 NQ points stop |
| Data | 5-min OHLCV, 1.2M bars, 17 years | FirstRateData continuous contract |
| Bar timeframe | 5-minute (primary), cross-TF at discretion | |
| Slippage assumption | 1–2 ticks (0.25–0.50 pts = $5–$10) | Conservative for NQ RTH |

---

## TopStep 50k Trading Combine Reference

> **Verify current rules at topstep.com before going live — these change periodically.**

| Rule | Typical Value | Impact on Strategy Design |
|------|--------------|--------------------------|
| Profit target | $3,000 | Need consistent positive EV, not home runs |
| Trailing max drawdown | $2,000 | CRITICAL constraint — tighter than 5% |
| Daily loss limit | $1,000 | Kill switch must trigger at -$1,000/day |
| Max position size | 2 NQ contracts (or 10 MNQ) | Position sizing capped |
| Min trading days | 2 days | Need sufficient frequency |
| Restricted times | Sometimes around major news | Macro filter may be required |

**Key implication**: The $2,000 trailing drawdown is the binding constraint, not the
$2,500 (5%) you mentioned. At $20/point on NQ, a 100-point adverse move = $2,000.
This means **every trade must have a hard stop of ~40-50 points maximum** to leave
room for a sequence of losers. This heavily favors tight, ATR-based stops and
selective entry (high win-rate or favorable R:R).

---

## Event Selection Criteria

Each candidate event must satisfy ALL of the following:

1. **Testable hypothesis** — a structural, behavioral, or statistical reason why
   the event should predict direction
2. **Sufficient frequency** — minimum 50 events/year (ideally 150–500) to build
   statistical significance across the 17-year dataset
3. **Implementable from OHLCV** — no Level 2, tick data, or external feeds required
   for the core definition (supplementary data enhances but isn't required)
4. **Compatible with holding period** — the event should resolve within 30 min to
   4 hours, matching the intraday swing target
5. **No look-ahead** — the event must be detectable in real-time using only
   information available at the close of the triggering bar
6. **Tradeable in RTH** — the event must occur during 09:30–16:00 ET with
   sufficient time remaining to manage the trade

---

## Event Catalog: 10 Candidate Events

---

### EVENT 1: Prior Session High/Low Sweep & Reversal

**Category:** Structural / Liquidity
**Estimated frequency:** 200–400 events/year (both directions combined)
**Recommended priority:** ★★★★★ (TOP PICK — start here)

#### Definition

A "sweep" occurs when price trades beyond the prior RTH session's high or low
by at least `sweep_threshold` points (e.g., 2–5 NQ points), then reverses and
closes back below/above the prior level within `reversal_window` bars.

```
Prior session high: H_prev
Current bar high:   H_curr
Current bar close:  C_curr

BEARISH SWEEP EVENT triggers when:
  (1) H_curr > H_prev + sweep_threshold           # price exceeded prior high
  (2) C_curr < H_prev                              # but closed back below it
  (3) This is the FIRST time today that H_prev was exceeded  # avoid re-triggers

BULLISH SWEEP EVENT (mirror for prior session low):
  (1) L_curr < L_prev - sweep_threshold            # price exceeded prior low
  (2) C_curr > L_prev                              # but closed back above it
  (3) First occurrence today
```

#### Hypothesis

Institutional and retail traders place stop-loss orders beyond prior session
extremes — these are "obvious" levels visible on every chart. Market makers and
algorithmic liquidity providers know these stops exist. The sweep mechanism works
as follows:

1. Price is driven through the prior high/low, triggering clustered stops
2. Stop triggers create a burst of market orders (forced buying above highs,
   forced selling below lows)
3. Sophisticated participants use this liquidity burst to fill large orders
   in the opposite direction
4. Price reverses as the stop-driven flow exhausts and the counter-directional
   orders absorb it

This is a **liquidity-driven mean reversion** hypothesis. It predicts that the
post-sweep direction is OPPOSITE to the sweep direction.

#### Direction

- Sweep of prior session HIGH → **SHORT** (bearish reversal expected)
- Sweep of prior session LOW → **LONG** (bullish reversal expected)

#### Parameters to sweep

| Parameter | Range | Default |
|-----------|-------|---------|
| sweep_threshold | 1–10 NQ points | 3 points |
| require_close_reversal | True/False | True |
| reversal_window | 1–5 bars | 1 bar (same bar) |
| min_time_after_open | 15–60 min | 30 min (avoid opening noise) |
| max_time_before_close | 30–60 min | 60 min (need time for trade) |

#### Known failure modes

- **Strong trend days**: When price sweeps the prior high and KEEPS going.
  These are breakout continuation days. Filter: check trend regime, ADX, or
  whether the sweep is in the direction of the higher-timeframe trend.
- **Double sweeps**: Price sweeps the high, reverses, then sweeps the low (or
  vice versa). These are "liquidation" days. Filter: disqualify if the opposite
  level was already swept today.
- **Low-volume sweeps**: Sweeps that occur on below-average volume may lack the
  stop-trigger mechanism. Filter: require volume on sweep bar > 1.2x rolling avg.
- **Gap days**: If the market gaps above/below the prior level at the open, the
  "sweep" happens differently. Filter: exclude if the open is already beyond the
  prior high/low.

#### Implementation notes for your data

```python
# Prior session high/low: compute from RTH bars (09:30-16:00 ET) of the previous day
# Your data has ET timestamps, so filter:
#   hour >= 9 and (hour > 9 or minute >= 30) and hour < 16
# Group by date, compute daily high/low, shift forward 1 day
```

---

### EVENT 2: Opening Range Breakout (ORB)

**Category:** Session-based / Momentum
**Estimated frequency:** ~250 events/year per direction (up + down)
**Recommended priority:** ★★★★★ (TOP PICK — classic, well-studied)

#### Definition

The Opening Range (OR) is the high and low established during the first N minutes
of RTH. An ORB event triggers when price closes above/below the OR boundary.

```
OR_high = max(high) for bars from 09:30 to 09:30 + OR_period
OR_low  = min(low)  for bars from 09:30 to 09:30 + OR_period

BULLISH ORB triggers when:
  (1) Bar close > OR_high                          # closes above opening range
  (2) Time is after OR_period but before max_entry_time
  (3) This is the FIRST close above OR_high today  # single trigger per direction

BEARISH ORB triggers when:
  (1) Bar close < OR_low
  (2) Time is after OR_period but before max_entry_time
  (3) First close below OR_low today
```

#### Hypothesis

The opening 15–30 minutes of RTH represent the "initial auction" where overnight
order flow, institutional rebalancing, and opening market-on-open orders establish
the session's initial range. This range reflects the market's initial consensus
on value.

A breakout from this range indicates that new information or order flow is pushing
price beyond the initial consensus — this represents **directional conviction**.
The hypothesis is momentum/continuation: the breakout direction predicts the
session's trend.

This is supported by market microstructure theory: the opening auction resolves
the information gap from overnight. Once resolved, a clear directional move
indicates institutional positioning.

#### Direction

- Break above OR → **LONG** (continuation expected)
- Break below OR → **SHORT** (continuation expected)

#### Parameters to sweep

| Parameter | Range | Default |
|-----------|-------|---------|
| OR_period_minutes | 5, 15, 30, 60 | 15 min |
| min_breakout_margin | 0–5 points beyond OR | 2 points |
| max_entry_time | 10:30, 11:00, 12:00 ET | 11:00 ET |
| require_close_beyond | True/False | True |
| min_or_range | 5–20 NQ points | 8 points (filter too narrow ORs) |
| max_or_range | 30–80 NQ points | 50 points (filter too wide ORs) |

#### Known failure modes

- **Wide opening ranges**: If the OR is already 50+ points, a breakout may have
  limited upside remaining in the session. Filter: cap OR width.
- **Both sides break (chop)**: On range-bound days, price breaks above the OR
  then reverses through the bottom (or vice versa). Filter: disable if the
  opposite ORB already triggered today.
- **News-driven opens**: If a major macro release hits at 08:30 (e.g., NFP, CPI),
  the opening range may already embed the full reaction. Filter: macro calendar flag.
- **Late entries**: ORBs that trigger after 11:00 ET leave less time for the trade
  to develop. Filter: max_entry_time parameter.

#### Implementation notes

```python
# OR period: first 3 bars (15min), 6 bars (30min), or 12 bars (60min)
# of RTH session (starting at 09:30 ET)
# Your data timestamps are already in ET — directly filterable
# Key: must track whether ORB already triggered today (per direction)
```

---

### EVENT 3: CUSUM Directional Accumulation Threshold

**Category:** Statistical / Regime Detection
**Estimated frequency:** 100–500 events/year (highly tunable)
**Recommended priority:** ★★★★☆ (novel, statistically rigorous)

#### Definition

The Cumulative Sum (CUSUM) filter detects structural breaks in the price process.
It accumulates signed returns and triggers when the cumulative magnitude exceeds
a threshold. Two variants:

**Variant A: Standard CUSUM (positive and negative accumulators)**
```
S_pos[t] = max(0, S_pos[t-1] + (r[t] - E[r]))    # positive accumulator
S_neg[t] = min(0, S_neg[t-1] + (r[t] - E[r]))    # negative accumulator

where r[t] = log return of bar t
      E[r] = expected return (typically 0 for intraday)

BULLISH EVENT: S_pos[t] > threshold (e.g., 1.5 * ATR_20)
BEARISH EVENT: |S_neg[t]| > threshold

After trigger: reset the accumulator that fired
```

**Variant B: Rolling directional CUSUM (session-anchored)**
```
Reset at session open (09:30 ET)
S[t] = cumsum of log returns since session open
BULLISH EVENT: S[t] crosses above +threshold for the first time today
BEARISH EVENT: S[t] crosses below -threshold for the first time today
```

#### Hypothesis

The CUSUM is derived from sequential analysis (Wald, 1945; Page, 1954) and is
used in industrial quality control to detect when a process has shifted from its
normal mean. Applied to markets:

1. Under the null hypothesis (no trend), returns are noise around zero
2. The CUSUM accumulates evidence against the null
3. When the threshold is breached, we reject the null — the process has shifted
4. The shift represents a genuine change in the supply/demand balance

This is a **trend detection** hypothesis. Unlike moving average crossovers
(which are lagging), CUSUM responds to the CUMULATIVE magnitude of directional
movement, making it more adaptive to the actual rate of price change.

The key advantage: CUSUM naturally adapts to volatility when the threshold is
expressed in ATR units. In high-vol environments, it requires more movement to
trigger; in low-vol environments, less.

#### Direction

- Positive CUSUM breach → **LONG** (uptrend detected)
- Negative CUSUM breach → **SHORT** (downtrend detected)

#### Parameters to sweep

| Parameter | Range | Default |
|-----------|-------|---------|
| threshold (ATR multiple) | 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 | 1.5 |
| ATR lookback | 10, 14, 20 bars | 14 |
| anchor_type | "rolling" vs "session" | "session" |
| E[r] estimation | 0, rolling mean(20), rolling mean(60) | 0 |
| max_triggers_per_day | 1, 2, 3, unlimited | 1 |
| min_time_for_trigger | 09:45, 10:00 ET | 09:45 ET |

#### Known failure modes

- **Choppy accumulation**: In choppy markets, the CUSUM can slowly drift to the
  threshold without a genuine trend. Filter: require the threshold to be hit
  within N bars (e.g., 20 bars) to ensure "sharpness."
- **Late-session triggers**: CUSUM triggering at 15:00 leaves no time for the
  trade. Filter: max time parameter.
- **Whipsaw after trigger**: Price triggers CUSUM then immediately reverses.
  Filter: require confirmation bar (close in direction of trigger).
- **Parameter sensitivity**: The threshold is the critical parameter. Too low =
  too many noisy signals. Too high = too few signals and late entry. This is
  why Step 4 includes a sensitivity sweep.

---

### EVENT 4: Volatility Compression → Expansion Breakout

**Category:** Regime Change
**Estimated frequency:** 150–300 events/year
**Recommended priority:** ★★★★☆ (strong theoretical basis)

#### Definition

Identify periods where short-term volatility contracts significantly below
long-term volatility (compression), then trigger when the first expansion bar
appears.

```
vol_ratio = ATR(fast_window) / ATR(slow_window)

COMPRESSION detected when:
  vol_ratio < compression_threshold (e.g., 0.70)
  AND compression persists for min_compression_bars (e.g., 3+ bars)

EXPANSION EVENT triggers when (while in compression state):
  current bar range > expansion_multiple * ATR(slow_window) (e.g., 1.3x)

Direction determined by expansion bar:
  BULLISH: expansion bar close > open (bullish expansion)
  BEARISH: expansion bar close < open (bearish expansion)
```

#### Hypothesis

Volatility is the most mean-reverting property in financial markets.
Low-volatility periods (compression) represent equilibrium — balanced buying
and selling pressure with no new information being incorporated. This
equilibrium is inherently unstable because:

1. Pending orders accumulate above and below the compressed range
2. The longer compression persists, the more energy builds
3. When a catalyst (news, large order, time-of-day effect) arrives,
   the breakout is amplified by the accumulated orders

The direction of the first expansion bar is informative because it reveals
which side of the equilibrium broke first. This is the principle behind
Bollinger Band squeezes and Keltner Channel breakouts.

#### Direction

- Bullish expansion bar → **LONG**
- Bearish expansion bar → **SHORT**

#### Parameters to sweep

| Parameter | Range | Default |
|-----------|-------|---------|
| fast_atr_window | 3, 5, 7 bars | 5 |
| slow_atr_window | 14, 20, 30 bars | 20 |
| compression_threshold | 0.50, 0.60, 0.70, 0.80 | 0.70 |
| min_compression_bars | 2, 3, 5, 8 | 3 |
| expansion_multiple | 1.2, 1.3, 1.5, 2.0 | 1.3 |
| require_volume_confirmation | True/False | True (vol > 1.2x avg) |

#### Known failure modes

- **False expansion**: A single large bar in compression that doesn't follow
  through. Filter: require 2nd bar to close in same direction (confirmation).
- **News-driven spikes**: Macro releases can cause expansion that reverses
  immediately. Filter: macro calendar flag.
- **End-of-compression drift**: Compression gradually lifts as ATR normalizes
  without a genuine breakout. Filter: require bar range to be significantly
  above the compression-period average, not just above the slow ATR.

---

### EVENT 5: VWAP Reclaim / Rejection

**Category:** Anchored Reference Level
**Estimated frequency:** 300–500 events/year
**Recommended priority:** ★★★★☆ (institutional benchmark level)

#### Definition

VWAP (Volume-Weighted Average Price) is the session-anchored fair value. A
reclaim or rejection event triggers when price makes a decisive cross of VWAP
after being on one side for a sustained period.

```
VWAP[t] = cumulative(TP * volume) / cumulative(volume)
  where TP = (high + low + close) / 3, reset at RTH open

VWAP_distance[t] = (close[t] - VWAP[t]) / ATR(20)

BULLISH RECLAIM triggers when:
  (1) close[t] > VWAP[t]                                  # closes above VWAP
  (2) close[t-1] < VWAP[t-1] or close[t-2] < VWAP[t-2]  # was below recently
  (3) min(VWAP_distance[t-N:t-1]) < -0.5                  # was meaningfully below
  (4) volume[t] > 1.0 * avg_volume(20)                    # decent volume on cross

BEARISH REJECTION triggers when:
  (1) close[t] < VWAP[t]                                  # closes below VWAP
  (2) close[t-1] > VWAP[t-1] or close[t-2] > VWAP[t-2]  # was above recently
  (3) max(VWAP_distance[t-N:t-1]) > 0.5                   # was meaningfully above
  (4) volume[t] > 1.0 * avg_volume(20)
```

#### Hypothesis

VWAP is the most important intraday reference price for institutional traders.
Large orders are frequently benchmarked to VWAP (VWAP algorithms, TWAP
algorithms). This creates self-reinforcing behavior:

1. When price is below VWAP, institutional buyers with VWAP benchmarks see
   an opportunity to buy below their target → buying pressure → reclaim
2. When price reclaims VWAP from below, it signals that the selling pressure
   that pushed it below VWAP has been absorbed
3. Institutions on the wrong side (short below VWAP) may cover → additional
   buying pressure

The reclaim event captures the transition from "below fair value" to
"at or above fair value" — a shift in the balance of power.

#### Direction

- Reclaim from below → **LONG** (buyers absorbed selling, fair value recaptured)
- Rejection from above → **SHORT** (sellers defended fair value)

#### Known failure modes

- **Choppy VWAP oscillation**: On range-bound days, price crosses VWAP 6-10+
  times. Each cross generates noise. Filter: require sustained time below VWAP
  (N bars) before a reclaim qualifies.
- **Late-day VWAP**: VWAP becomes less meaningful late in the session as it's
  heavily anchored to early price action. Filter: weight early-session events
  more heavily, or exclude events after 14:00 ET.
- **Trend days**: On strong trend days, price stays on one side of VWAP all day.
  The "reclaim" events on these days are actually pullback entries — valid but
  a different trade type.

---

### EVENT 6: Opening Gap Fill / Fade

**Category:** Session-based / Mean Reversion
**Estimated frequency:** 200–250 events/year
**Recommended priority:** ★★★☆☆ (well-known, but many participants trade it)

#### Definition

A gap occurs when the RTH open price differs meaningfully from the prior RTH
close. The gap fade event triggers at the open; the gap fill event triggers
when price returns to the prior close.

```
prior_close = close of last bar of prior RTH session (15:55 or 16:00 ET bar)
today_open  = open of first RTH bar today (09:30 ET bar)
gap_size    = today_open - prior_close
gap_pct     = gap_size / prior_close

GAP FADE EVENT (entry at open):
  Triggers when: |gap_pct| > min_gap_pct (e.g., 0.15% ≈ 25 NQ points)
  Direction: OPPOSITE to gap direction
    Gap up   → SHORT (fade the gap, expect fill)
    Gap down → LONG  (fade the gap, expect fill)

GAP FILL EVENT (confirmation):
  Triggers when price touches prior_close after a gap
  This is a SECONDARY event that confirms the fade worked

GAP CONTINUATION EVENT (alternative hypothesis):
  Triggers when: |gap_pct| > large_gap_pct (e.g., 0.5%)
  Direction: SAME as gap direction (large gaps continue)
```

#### Hypothesis

**Small/medium gaps (0.15%–0.5%):** Overnight futures markets are thinner and
more susceptible to sentiment-driven moves. When RTH opens and full liquidity
returns, the overnight move often partially or fully reverts. Statistics show
that NQ gaps fill within the first 2 hours approximately 60–70% of the time
for gaps under 0.5%.

**Large gaps (>0.5%):** Large gaps typically represent genuine price discovery
(earnings, FOMC, geopolitical events). These are less likely to fill and more
likely to continue. The hypothesis reverses for large gaps.

#### Parameters to sweep

| Parameter | Range | Default |
|-----------|-------|---------|
| min_gap_pct | 0.10%, 0.15%, 0.20%, 0.30% | 0.15% |
| large_gap_pct (continuation) | 0.40%, 0.50%, 0.75% | 0.50% |
| max_gap_pct (exclude extreme) | 1.0%, 1.5%, 2.0% | 1.5% |
| entry_delay_bars | 0, 1, 2 bars after open | 1 bar (let open settle) |
| fill_target | 100%, 75%, 50% of gap | 75% |

#### Known failure modes

- **Trend continuation gaps**: Gaps in the direction of a strong multi-day
  trend often don't fill. Filter: trend regime context.
- **Macro event gaps**: CPI, FOMC gaps represent genuine new information.
  Filter: macro calendar.
- **Partial fills**: Price fills 50% of the gap then reverses. This is why
  the fill_target parameter is important — 75% fill is often more realistic.

---

### EVENT 7: Multi-Bar Momentum Exhaustion

**Category:** Microstructure / Mean Reversion
**Estimated frequency:** 100–200 events/year
**Recommended priority:** ★★★★☆ (counter-trend, high R:R potential)

#### Definition

Identify a sequence of N+ consecutive directional bars with signs of exhaustion
on the final bar(s).

```
BEARISH EXHAUSTION (after up-move, signals reversal down):
  (1) consec_up_bars >= N (e.g., 4+ bars closing above open)
  (2) cumulative_move = sum of (close - open) for those bars
  (3) cumulative_move > exhaustion_atr_multiple * ATR(20) (e.g., 1.5x ATR)
  (4) EXHAUSTION SIGNAL on current bar (at least 1 of):
      a) bar_range < 0.6 * avg_range(5)            # shrinking range
      b) upper_wick_ratio > 0.4                     # rejection wick
      c) volume < 0.8 * avg_volume(5)               # declining volume
      d) close_position < 0.3                       # closed in lower 30% of bar

BULLISH EXHAUSTION (after down-move, signals reversal up):
  Mirror of above with inverted conditions
```

#### Hypothesis

Extended unidirectional moves exhaust available supply (for up-moves) or demand
(for down-moves) at current prices. As the move extends:

1. Profit-taking from early participants increases
2. Counter-trend limit orders accumulate
3. The marginal buyer/seller becomes less aggressive
4. Exhaustion signals (declining volume, wicks, smaller bars) indicate the
   move is running out of fuel

The mean-reversion trade enters AGAINST the exhausted move, anticipating that
profit-taking and counter-trend orders will push price back. This is particularly
effective in NQ because index futures attract short-term momentum traders whose
positions must be unwound.

#### Direction

- Exhaustion after UP-move → **SHORT**
- Exhaustion after DOWN-move → **LONG**

#### Parameters to sweep

| Parameter | Range | Default |
|-----------|-------|---------|
| min_consec_bars | 3, 4, 5, 6 | 4 |
| min_atr_move | 1.0, 1.5, 2.0, 2.5 | 1.5 |
| exhaustion_criteria | any 1 of 4, any 2 of 4 | any 2 of 4 |
| volume_decline_threshold | 0.7, 0.8, 0.9 | 0.8 |
| wick_rejection_threshold | 0.3, 0.4, 0.5 | 0.4 |

#### Known failure modes

- **Parabolic trending days**: Rarely, NQ enters a parabolic move where
  exhaustion signals are false — the move accelerates after apparent exhaustion.
  These are the most dangerous days for mean-reversion. Filter: vol regime
  (avoid in "crisis" vol), check if event aligns with a major news catalyst.
- **Partial retracement**: Price retraces 30-50% of the move but then continues.
  Use ATR-based take-profit to capture the retracement rather than expecting
  a full reversal.

---

### EVENT 8: Inside Bar Breakout (Compression Pattern)

**Category:** Microstructure / Breakout
**Estimated frequency:** 150–300 events/year (with filters)
**Recommended priority:** ★★★☆☆ (simple but noisy without context filters)

#### Definition

An inside bar (IB) is a bar whose entire range is contained within the prior
bar's range. The event triggers on the breakout from the inside bar.

```
INSIDE BAR detected when:
  high[t] <= high[t-1]  AND  low[t] >= low[t-1]

BREAKOUT EVENT triggers on bar t+1 (or t+2, etc.) when:
  BULLISH: close[t+k] > high[t-1]    # breaks above the mother bar's high
  BEARISH: close[t+k] < low[t-1]     # breaks below the mother bar's low

FILTERED inside bars (to reduce noise):
  (1) Mother bar range > 1.0 * ATR(20)         # mother bar was meaningful
  (2) Inside bar range < 0.5 * mother bar range # genuine contraction
  (3) Inside bar volume < mother bar volume     # declining participation
  (4) Occurs near a reference level (VWAP, prior session H/L, round number)
```

#### Hypothesis

An inside bar represents a temporary balance between buyers and sellers —
neither side could push price beyond the prior bar's range. This micro-scale
equilibrium, especially after a large mother bar, represents a "pause" in the
directional move while participants evaluate.

The breakout from this equilibrium is informative: the direction of the breakout
indicates which side won the evaluation period. When combined with context
(trend direction, proximity to key levels), inside bar breakouts can be high-
probability continuation or reversal signals.

#### Direction

- Break above mother bar high → **LONG**
- Break below mother bar low → **SHORT**

#### Known failure modes

- **Extremely common without filters**: Raw inside bars on 5-min data occur
  20-40x per day. Without filters, the signal-to-noise ratio is terrible.
- **Multiple inside bars**: Sometimes 3-4 inside bars form in sequence (NR4,
  NR7 patterns). The breakout from the FIRST vs LAST inside bar may differ.
- **Time-of-day dependency**: Inside bars during 12:00-14:00 ET (lunch) are
  much less meaningful than those during the opening or closing cross.

---

### EVENT 9: Initial Balance Extension

**Category:** Session Structure / Market Profile
**Estimated frequency:** ~250 events/year
**Recommended priority:** ★★★★☆ (strong theoretical basis from Market Profile)

#### Definition

The Initial Balance (IB) is the range established during the first hour of RTH
(09:30–10:30 ET = first 12 bars on 5-min). An IB Extension occurs when price
breaks beyond the IB for the first time.

```
IB_high = max(high) for bars 09:30 to 10:30 ET
IB_low  = min(low)  for bars 09:30 to 10:30 ET
IB_range = IB_high - IB_low

IB EXTENSION UP triggers when (after 10:30 ET):
  high[t] > IB_high  AND  this is the FIRST time IB_high was exceeded

IB EXTENSION DOWN triggers when:
  low[t] < IB_low  AND  this is the FIRST time IB_low was breached

ADDITIONAL CONTEXT:
  IB_width_vs_atr = IB_range / ATR(20)
  - Narrow IB (< 0.8x ATR) → more likely to extend (range expansion day)
  - Wide IB (> 1.5x ATR) → less likely to extend, or extension fails
```

#### Hypothesis

From Market Profile theory (Steidlmayer): the Initial Balance represents the
range of prices where "local" (short-timeframe) participants establish value.
An extension beyond the IB indicates the arrival of "other timeframe"
participants (institutional, longer-horizon traders) who are pushing price
beyond the local consensus.

The first IB extension is the most significant because it signals the first
directional commitment of the session by these other-timeframe participants.
Market Profile categorizes days by IB behavior:

- **Normal day**: IB is not extended significantly
- **Normal variation day**: IB extended by ~0.5–1x IB range
- **Trend day**: IB extended by 2x+ — these are the high-profit opportunity days

The hypothesis is **momentum/continuation**: the extension direction predicts
the session's dominant direction.

#### Direction

- Extension above IB → **LONG** (other-timeframe buyers arrived)
- Extension below IB → **SHORT** (other-timeframe sellers arrived)

#### Parameters to sweep

| Parameter | Range | Default |
|-----------|-------|---------|
| IB_duration_minutes | 30, 60, 90 | 60 |
| min_extension_points | 0, 2, 5 NQ points | 2 |
| max_entry_time | 12:00, 13:00, 14:00 ET | 13:00 ET |
| IB_width_filter_low | 0.5, 0.6, 0.8 ATR | None |
| IB_width_filter_high | 1.5, 2.0, 2.5 ATR | 2.0 ATR |

#### Known failure modes

- **Failed extensions**: Price extends IB by a few points then reverses back
  into IB. These are "Normal variation" days. Filter: require extension to
  persist for N bars (e.g., 3 bars beyond IB) or require close beyond IB.
- **Both-side extensions**: Both IB_high and IB_low get breached → range day.
  Filter: disable if opposite side already extended.
- **Wide IB**: If IB is very wide (2x+ ATR), extension is less meaningful.
  Filter: IB width cap.

---

### EVENT 10: Single-Bar Volatility Spike (ATR Z-Score)

**Category:** Volatility Event / Information Arrival
**Estimated frequency:** 50–150 events/year (Z > 2.5)
**Recommended priority:** ★★★☆☆ (lower frequency but high-information content)

#### Definition

A single bar whose True Range is an extreme outlier relative to recent volatility.

```
TR[t] = max(high[t] - low[t], |high[t] - close[t-1]|, |low[t] - close[t-1]|)
ATR_mean = rolling_mean(TR, 20)
ATR_std  = rolling_std(TR, 20)
TR_zscore[t] = (TR[t] - ATR_mean[t]) / ATR_std[t]

VOLATILITY SPIKE EVENT triggers when:
  TR_zscore[t] > spike_threshold (e.g., 2.5)
  AND volume[t] > 1.5 * avg_volume(20)   # confirm it's real, not a data glitch

Direction determined by close position:
  BULLISH SPIKE: close_position = (close - low) / (high - low) > 0.7
    → closed in upper 30% of spike bar → buyers won
  BEARISH SPIKE: close_position < 0.3
    → closed in lower 30% → sellers won
  NEUTRAL SPIKE: close_position between 0.3 and 0.7
    → indeterminate, no event triggered
```

#### Hypothesis

Anomalously large bars indicate the arrival of significant new information
or a large institutional order that temporarily overwhelms the order book.
The close position within the spike bar reveals how the market resolved
this new information:

1. **Close near high** (bullish resolution): The initial shock was absorbed
   by buyers. The large bar attracted buying interest, and the close near the
   high indicates continued upside pressure. Follow-through is expected.
2. **Close near low** (bearish resolution): The shock was resolved to the
   downside. Sellers maintained control through the volatility event.
3. **Close in middle** (unresolved): The spike represents confusion, not
   conviction. No directional signal.

#### Direction

- Spike with close near high → **LONG** (bullish resolution)
- Spike with close near low → **SHORT** (bearish resolution)

#### Known failure modes

- **Data errors / flash crashes**: Ensure volume confirms the spike is real.
- **FOMC / major news**: The first spike bar after news may be followed by
  whipsaw as the market digests the information. Filter: if during the first
  5 minutes after a known macro release, wait for the second spike bar.
- **Stop cascades**: A spike triggered by cascading stops may reverse sharply
  once the stops are exhausted. Close position helps distinguish genuine
  directional moves from cascades.

---

## Recommended Starting Configuration

Based on your constraints (intraday swing, RTH only, 50k account, TopStep
eventual target), I recommend researching these events in this priority order:

### Tier 1 — Start here (highest expected value, best sample size)
1. **EVENT 1: Prior Session High/Low Sweep** — counter-trend, high theoretical
   basis, excellent for intraday swing timeframe
2. **EVENT 2: Opening Range Breakout** — momentum, massive academic and
   practitioner literature, aligns with early-session entries

### Tier 2 — Research next (strong but need more parameter tuning)
3. **EVENT 3: CUSUM Threshold** — statistically novel, less crowded
4. **EVENT 9: Initial Balance Extension** — Market Profile basis, pairs well with ORB
5. **EVENT 4: Volatility Compression → Expansion** — regime-change entry

### Tier 3 — Research if Tier 1-2 show promise (supporting events)
6. **EVENT 5: VWAP Reclaim/Rejection** — may work better as a filter than primary event
7. **EVENT 7: Momentum Exhaustion** — counter-trend, fewer events but high R:R
8. **EVENT 6: Gap Fill/Fade** — well-known edge, possibly decayed

### Deprioritize (noisy or low frequency without enrichment data)
9. **EVENT 8: Inside Bar Breakout** — too noisy without additional context
10. **EVENT 10: Volatility Spike** — low frequency, better as a filter

---

## Supplementary Data Sources

### Free / Low-Cost Sources

| Data Type | Source | Cost | Notes |
|-----------|--------|------|-------|
| Economic Calendar | **FRED API** (fred.stlouisfed.org) | Free | FOMC dates, CPI, NFP, GDP release dates. Python: `fredapi` package. Excellent for macro event flags. |
| Economic Calendar | **Investing.com Economic Calendar** | Free | Scrape or use unofficial API. Covers all major releases with impact ratings. |
| VIX Daily | **CBOE** (cboe.com/tradable_products/vix) | Free | Daily VIX close. Download CSV. Good for daily regime context. |
| VIX Intraday | **Yahoo Finance** (^VIX) | Free | 1-min/5-min VIX via `yfinance` package. Slight lag but usable. |
| FOMC Dates | **Federal Reserve** (federalreserve.gov) | Free | Historical FOMC meeting dates. Critical for filtering. |

### Paid Sources (when ready to upgrade)

| Data Type | Source | Cost | Notes |
|-----------|--------|------|-------|
| Tick data / L2 | **Databento** | ~$50-100/mo | Institutional-grade, CME direct feed. Best quality. |
| Tick data | **Kinetick** (via NinjaTrader) | ~$55/mo | Good for NQ/ES. Real-time + historical. |
| Volume Profile | **Computable from your 5-min data** | Free | Approximate VP using 5-min volume distribution. Not true tick-level VP but functional. |
| Multi-asset | **Polygon.io** | Free tier + $29/mo | Stocks, options, indices. Good for cross-market features. |

### Recommended First Addition

**Start with FRED API for FOMC/CPI/NFP dates.** This is free, takes 30 minutes
to implement, and immediately gives you a critical binary feature
(`is_macro_day`) that filters several events above. I can build this in Step 3.

---

## Next Steps

1. **Select 2-3 events** from the catalog to implement first
2. Move to **Step 2 (Outcome Labeling)** — define double-barrier labels for
   the selected events
3. Move to **Step 3 (Feature Engineering)** — build contextualizing features
   using your existing `feature_engineering.py` as a foundation
4. Implement events in `event_definitions.py` with precise, vectorized pandas logic

---

*Document generated 2026-03-03 for NQ event-driven research framework.*
