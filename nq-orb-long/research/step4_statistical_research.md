# Step 4 — Statistical Research
## NQ Futures Event-Driven Research Framework
### Date: 2026-03-03

---

## Purpose

Step 4 is where we answer the critical question: **"Is there a real edge, or
is this noise?"** We apply rigorous statistical methods to separate genuine
predictive signal from randomness.

This step uses an expanded dataset (**2019–2026, ~7 years**) covering multiple
market regimes — pre-COVID, COVID crash, post-COVID bull run, 2022 bear market,
and the 2023-2025 recovery. This diversity is essential for testing robustness.

---

## Analysis Components

### 1. Expected Value with Bootstrap Confidence Intervals

The single most important number in trading is **Expected Value (EV)** — the
average P&L per trade. But a point estimate alone is meaningless without
knowing how confident we are. A strategy with EV = +2.0 pts/trade based on
50 trades could easily be noise; the same EV from 500 trades is more credible.

**Bootstrap method:**
1. Take the actual trade outcomes (barrier_return_pts for each event)
2. Resample with replacement 10,000 times (same sample size)
3. Compute EV for each bootstrap sample
4. The 2.5th and 97.5th percentiles give the 95% confidence interval

**Interpretation:**
- If the 95% CI is entirely above zero → statistically significant positive EV
- If the CI spans zero → cannot reject the null (no edge proven)
- The WIDTH of the CI tells us about estimation precision (more trades = narrower)

### 2. Parameter Sensitivity Sweep

We test how sensitive the results are to barrier parameters:

| Parameter | Values Tested | Purpose |
|-----------|--------------|---------|
| SL multiple | 0.75, 1.0, 1.25, 1.5 | Stop-loss tightness |
| TP multiple | 1.0, 1.5, 2.0, 2.5, 3.0 | Take-profit ambition |
| Max holding | 12, 24, 36, 48 bars | Time patience |
| ATR lookback | 10, 14, 20 | Volatility responsiveness |

**Total combinations:** 4 × 5 × 4 × 3 = 240 per event type

We're looking for **robust regions** — parameter zones where the strategy is
profitable across a range of values, not a single "magic" setting that works
only at one exact point (which would indicate overfitting).

### 3. Regime Segmentation

We test performance in different market environments:

| Regime | Definition | Why It Matters |
|--------|-----------|---------------|
| Volatility regime | Low / Normal / High / Crisis (ATR percentile) | Most strategies fail in crisis vol |
| Trend regime | Trending vs. range-bound (efficiency ratio) | Reversals fail in trends |
| Day-of-week | Mon-Fri | Institutional flow patterns differ |
| Time-of-day | Morning / Midday / Afternoon | Liquidity regimes differ |
| Year | 2019-2025 | Structural market changes |

### 4. Look-Ahead Bias Check

We verify that no future information leaks into features by:
1. Checking that all features are computed from data at or before the event bar
2. Confirming that labels use only the conservative same-bar convention
3. Running a "shuffled label" test — if the model performs equally well on
   randomly shuffled labels, our features are overfit or leaking

### 5. Sample Size Sufficiency

Using the formula for minimum sample size to detect a given effect:
```
n = (z_α/2 + z_β)² × σ² / δ²
```
Where:
- δ = minimum EV we want to detect (e.g., 2 pts/trade)
- σ = standard deviation of trade returns
- z values for 95% confidence / 80% power

This tells us whether we have ENOUGH events to draw reliable conclusions.

---

## Data Coverage

| Period | Years | Regime | Purpose |
|--------|-------|--------|---------|
| 2019-01 to 2019-12 | 1 | Pre-COVID low vol | Baseline calm market |
| 2020-01 to 2020-12 | 1 | COVID crash + recovery | Stress test |
| 2021-01 to 2021-12 | 1 | Post-COVID bull | Trending up |
| 2022-01 to 2022-12 | 1 | Fed tightening bear | Trending down |
| 2023-01 to 2023-12 | 1 | Recovery + AI boom | Mixed |
| 2024-01 to 2024-12 | 1 | Continuation | Recent |
| 2025-01 to 2026-01 | 1 | Most recent | Validation |

**Total expected events per type:** 300-700 over 7 years (vs. 135-174 in 2 years).
This provides much stronger statistical power.

---

*Document generated 2026-03-03 for NQ event-driven research framework.*
