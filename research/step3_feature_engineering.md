# Step 3 — Event-Context Feature Engineering
## NQ Futures Event-Driven Research Framework
### Date: 2026-03-03

---

## Why Contextual Features Matter

In Step 2, we saw that the base events with default barrier parameters produce
slightly negative expected values. **This is expected.** A raw event signal
without context is like asking "did the traffic light turn green?" without
looking at the road conditions.

The same event can produce very different outcomes depending on **context**:

- A sweep of the prior session low on a **low-volatility trending day** might
  be a genuine reversal signal (buyers absorb stops and push price higher)
- The same sweep on a **high-volatility crash day** is probably just a
  continuation — price will keep going through the level

Feature engineering answers: **what was the market environment when this event
fired?** Features that consistently separate winners from losers become
the inputs to our predictive model (Step 5).

---

## Feature Design Principles

### 1. No Look-Ahead

Every feature is computed using ONLY data available at or before the event bar.
This means:
- Rolling calculations use `.shift(1)` or are evaluated up to the current bar
- Cross-timeframe features use forward-fill from the most recently **completed** bar
- Session-level stats (VWAP, session high) use cumulative values up to the event bar

### 2. Normalization

Raw price-level features (e.g., "close = 18,500") are useless because NQ's
price level changes over time. Instead, we normalize everything:
- Distances → expressed in ATR units
- Ranges → expressed as ratios to recent average
- Returns → log returns or percentage changes
- Volume → relative to rolling average

### 3. Stationarity

Features should be approximately stationary (same statistical properties over
time). Non-stationary features like raw price or cumulative volume will cause
models to overfit to specific time periods.

### 4. Event-Specific Relevance

Not all features matter equally for every event. A sweep event cares about
how far beyond the level price went; an ORB event cares about opening range
width. We include both **universal features** (apply to all events) and
**event-specific features** (unique to each event type).

---

## Feature Categories

### Category 1: Volatility Context (5 features)

These answer: **"What is the volatility environment?"**

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `atr_14` | ATR(14) at event time | Raw (used as normalizer) |
| `vol_ratio_5_20` | Fast ATR / Slow ATR ratio | Ratio (>1 = expanding) |
| `natr_14` | ATR / Close (normalized ATR) | Fraction of price |
| `vol_regime` | Volatility percentile bucket (0-3) | Ordinal |
| `vol_zscore` | (Short vol - Long vol) / Long vol | Z-score |

**Why these matter:** Events in low-vol environments tend to resolve
differently than in high-vol. A 15-point sweep in a market doing 10-point
ATR bars is a massive move; the same 15-point sweep when ATR is 30 is noise.

### Category 2: Trend & Momentum Context (7 features)

These answer: **"Is the market trending or range-bound? Which direction?"**

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `log_ret_5` | 5-bar (25 min) log return | Log return |
| `log_ret_20` | 20-bar (100 min) log return | Log return |
| `log_ret_60` | 60-bar (5 hour) log return | Log return |
| `rsi_14` | RSI(14) at event time | 0-100 |
| `efficiency_ratio_1d` | Price path efficiency (1 = pure trend) | 0-1 |
| `trend_regime` | Binary: trending (1) or ranging (0) | Binary |
| `dist_ma_20` | Distance from 20-bar MA, normalized | Fraction |

**Why these matter:** Mean-reversion events (sweeps, exhaustion) work best
in range-bound markets. Breakout events (ORB, CUSUM) work best in trending
markets. Knowing the regime helps filter false signals.

### Category 3: Volume & Participation (5 features)

These answer: **"Is the move backed by conviction?"**

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `rvol_ratio_20` | Current volume / 20-bar avg volume | Ratio |
| `rvol_ratio_5` | Current volume / 5-bar avg volume | Ratio |
| `vwap_distance_atr` | Distance from session VWAP in ATR | ATR units |
| `obv_slope_20` | OBV slope over 20 bars | Cumulative |
| `vol_price_diverge` | Volume/price direction disagreement | Binary |

**Why these matter:** A sweep on heavy volume is more likely to be genuine
institutional activity. A breakout on declining volume is more likely a
false breakout. VWAP distance tells us whether institutional flow is with
or against the event.

### Category 4: Microstructure (4 features)

These answer: **"What does the price action look like at the event?"**

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `body_ratio` | Bar body / bar range (0=doji, 1=marubozu) | 0-1 |
| `upper_wick_ratio` | Upper wick / bar range | 0-1 |
| `lower_wick_ratio` | Lower wick / bar range | 0-1 |
| `range_ratio_10` | Current bar range / 10-bar avg range | Ratio |

**Why these matter:** A sweep bar with a long wick (rejection) is a stronger
reversal signal than one with a full body. An ORB breakout bar with a large
body shows conviction; a doji shows indecision.

### Category 5: Temporal Context (5 features)

These answer: **"When in the session did this event fire?"**

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `bar_of_session` | Bar number within RTH (0-77) | Integer |
| `mins_since_open` | Minutes since RTH open | Integer |
| `hour_sin` | Cyclical hour encoding (sin) | -1 to 1 |
| `hour_cos` | Cyclical hour encoding (cos) | -1 to 1 |
| `dow_sin` | Day-of-week cyclical encoding | -1 to 1 |

**Why these matter:** Morning events (first 2 hours) have different dynamics
than afternoon events. Monday mornings differ from Friday afternoons.
Cyclical encoding prevents the model from thinking 15:00 and 09:30 are
"far apart" — they're adjacent in the trading cycle.

### Category 6: Session Reference Levels (5 features)

These answer: **"Where is price relative to key institutional levels?"**

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `dist_prior_high_atr` | Distance from prior session high (ATR) | ATR units |
| `dist_prior_low_atr` | Distance from prior session low (ATR) | ATR units |
| `dist_session_open_atr` | Distance from today's RTH open (ATR) | ATR units |
| `gap_pct` | Opening gap (if applicable) | Fraction |
| `ib_position` | Position within Initial Balance range | 0-1 |

**Why these matter:** Institutional traders use these levels. Events near
key levels have confluence. A sweep of the prior low when price is already
2 ATR below VWAP in a downtrend is very different from a sweep when price
is near VWAP in a range.

### Category 7: Event-Specific Features

These are unique to each event type:

#### Sweep Events (4 features)
| Feature | Description |
|---------|-------------|
| `sweep_depth_atr` | How far beyond the prior level (ATR units) |
| `sweep_bar_body_direction` | Did the sweep bar close bullish or bearish? |
| `bars_since_session_start` | How far into the session is this sweep? |
| `prior_level_distance_atr` | How far was the prior level from today's open? |

#### ORB Events (4 features)
| Feature | Description |
|---------|-------------|
| `or_range_atr` | Opening range width in ATR units |
| `or_range_relative` | OR width vs 20-day average OR width |
| `breakout_strength_atr` | How far beyond the OR boundary |
| `gap_same_direction` | Did the opening gap align with breakout? |

---

## Feature Count Summary

| Category | Features | Purpose |
|----------|----------|---------|
| Volatility Context | 5 | Regime identification |
| Trend & Momentum | 7 | Directional context |
| Volume & Participation | 5 | Conviction / divergence |
| Microstructure | 4 | Price action quality |
| Temporal Context | 5 | Time-of-day patterns |
| Session Reference | 5 | Institutional levels |
| Event-Specific | 4 per type | Unique event context |
| **Total** | **~35** | |

This is a manageable feature set. Too few features and the model can't
learn; too many and it overfits. 30-40 features with 100-500 events per
year gives a reasonable feature-to-sample ratio for tree-based models
(which handle many features well).

---

## Implementation Architecture

### Two-Stage Pipeline

```
Stage 1: Bar-Level Features
    load_ohlcv() → build_features() → DataFrame with 60+ columns
    (runs once on the full dataset, reused for all events)

Stage 2: Event-Context Extraction
    For each event type:
        labeled_events (from Step 2)
        ↓
        extract_features_at_event_time(full_df, events)
        ↓
        add_event_specific_features()
        ↓
        Feature matrix ready for modeling (Step 5)
```

Stage 1 uses the existing `feature_engineering.py` from `research_utils/`.
Stage 2 is the new `event_features.py` module built in this step.

### Output Format

The final output is a single DataFrame per event type with:
- **Index**: event_time (timestamp)
- **Label columns**: barrier_label, exit_type, barrier_return_pts (from Step 2)
- **Feature columns**: ~35 normalized features (from this step)

This DataFrame is the direct input to Step 5 (model training).

---

## Feature Importance Preview

After extraction, we compute preliminary feature importance using:
1. **Correlation with label**: Pearson correlation between each feature and
   the barrier_label (+1/-1/0)
2. **Mean difference**: Average feature value for winners vs losers
3. **Distribution separation**: KS-test statistic between winner/loser
   feature distributions

These are quick checks to verify features have signal before building
models. Features with zero correlation across all events are candidates
for removal.

---

## Cross-Timeframe Features (Stretch Goal)

For Step 3, we use single-timeframe (5-min) features only. In Step 4/5,
we can add:
- 15-min RSI, ATR, efficiency ratio (via `add_cross_timeframe_features`)
- 1-hour trend regime
- Daily-level open interest changes

These require resampling the 5-min data to higher timeframes and using
forward-fill to avoid look-ahead. The infrastructure for this already
exists in `research_utils/feature_engineering.py`.

---

*Document generated 2026-03-03 for NQ event-driven research framework.*
