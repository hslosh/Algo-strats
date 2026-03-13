# NQ ORB Long — ML-Enhanced Opening Range Breakout

A machine-learning-filtered Opening Range Breakout strategy for NQ (Nasdaq-100 E-mini) futures, targeting the TopStep $50k combine and eventual live deployment via Interactive Brokers.

---

## Strategy Overview

| Parameter | Value |
|-----------|-------|
| Instrument | NQ Futures (E-mini Nasdaq-100) |
| Direction | Long only |
| Session | RTH — 09:30–16:00 ET |
| Opening Range | First 30 minutes (09:30–10:00 ET) |
| Entry trigger | Close above OR high + 1 tick |
| Entry timing | Open of bar+2 (2-bar confirmation delay) |
| Entry filter | Calibrated ML P(win) ≥ 0.58 |
| Stop Loss | 1.0 × ATR14 below entry |
| Take Profit | 1.5 × ATR14 above entry |
| Timeout | 48 bars (4 hours) — flatten if neither hit |
| EOD flatten | 15:30 ET (30 min before close) |
| Slippage | 1 tick ($5) per fill |
| Commission | $4.50/side ($9.00 round-trip) per contract |
| Account size | $50,000 |
| Max contracts | 2 |

---

## Pipeline (8 Steps)

```
Step 1  event_definitions.py     — ORB breakout event detection
Step 2  outcome_labeling.py      — Triple-barrier labeling (SL / TP / timeout)
Step 3  feature_engineering.py   — 60+ features (momentum, volume, session context)
Step 4  statistical_research.py  — Edge validation, bootstrap EV, CUSUM
Step 5  model_design.py          — Walk-forward logistic regression + Platt calibration
Step 6  strategy_construction.py — Position sizing, Monte Carlo, threshold sensitivity
Step 7  backtest_validation.py   — True bar-by-bar backtest with risk management
Step 8  deployment_checklist.py  — Pre-flight checks, final report
```

---

## Key Design Decisions

**Walk-forward validation** — Expanding window with 6-month test blocks and 5-day embargo. Minimum 200 training events. No data leakage between folds.

**Holdout separation** — Events from 2025-07-01 onward are held out from all WFO training/calibration. A production model is trained on all pre-holdout data, then applied to the holdout set with a calibrator fit on WFO OOS results only.

**Forward-only calibration** — Fold 0 calibrates on the training tail (isotonic). Fold N calibrates on all prior OOS folds (Platt). No future OOS data touches the calibrator.

**Bar-by-bar simulation** — The backtest iterates every 5-minute bar (not just event times). SL/TP/timeout/EOD exits are checked on each bar. SL takes precedence when SL and TP could both trigger on the same bar.

**Risk gates** — Daily loss cap (50 ticks), max 3 trades/day, consecutive loss pause (halts rest of session), max drawdown circuit breaker.

---

## Repository Structure

```
nq-orb-long/
├── research/
│   ├── config.py                  # Canonical constants (threshold, commission, etc.)
│   ├── event_definitions.py       # Step 1 — ORB event detection
│   ├── outcome_labeling.py        # Step 2 — Triple-barrier labels
│   ├── event_features.py          # Step 3 — Feature extraction
│   ├── statistical_research.py    # Step 4 — Edge validation
│   ├── model_design.py            # Step 5 — WFO model + calibration
│   ├── strategy_construction.py   # Step 6 — Simulation + Monte Carlo
│   ├── backtest_validation.py     # Step 7 — Bar-by-bar backtest
│   ├── deployment_checklist.py    # Step 8 — Pre-flight checks
│   ├── live_runner.py             # IB live signal generator
│   ├── orb_long_nq.pine           # PineScript v6 — TradingView visualization
│   └── step[1-8]_*.md             # Step-by-step research notes
├── research_utils/
│   ├── feature_engineering.py     # OHLCV loading + feature building
│   ├── data_pipeline.py           # Data utilities
│   ├── backtest_runner.py         # Backtest helpers
│   └── wfo_and_robustness.py      # WFO split builder + robustness checks
├── NQ_*_sample.csv                # Sample OHLCV data (1min/5min/30min/1h/1d)
└── FIX_PROMPT.md                  # Audit log of pipeline fixes applied
```

---

## Canonical Constants (`research/config.py`)

```python
CANONICAL_THRESHOLD   = 0.58   # Min calibrated P(win) to enter
TICK_SIZE             = 0.25   # NQ tick size (index points)
NQ_MULTIPLIER         = 20     # $20 per point
COMMISSION_PER_SIDE   = 4.50   # Per contract per fill
SLIPPAGE_TICKS        = 1      # Per fill
STOP_LOSS_ATR_MULT    = 1.0
TAKE_PROFIT_ATR_MULT  = 1.5
MAX_POSITION_BARS     = 48
DAILY_LOSS_CAP_TICKS  = 50
MAX_TRADES_PER_DAY    = 3
HOLDOUT_START_DATE_STR = '2025-07-01'
```

---

## Quick Start

```python
# Run the full Step 7 backtest + validation pipeline
from research.backtest_validation import run_full_step7
results = run_full_step7(verbose=True)

# Outputs: trades DataFrame, metrics, Monte Carlo, rolling stability, holdout report
```

```python
# Run Step 6 (strategy construction / simulation)
from research_utils.feature_engineering import load_ohlcv, build_features
from research.strategy_construction import run_full_step6

df = load_ohlcv('nq_continuous_5m_converted.csv')
df = build_features(df)
results = run_full_step6(df)
```

---

## Data

Historical NQ 5-minute continuous data from [FirstRateData](https://firstratedata.com) (2008–2026, ratio-adjusted, US Eastern Time). The full dataset (`nq_continuous_5m_converted.csv`) is excluded from version control due to size. Sample files for each timeframe are included.

---

## Live Deployment

`research/live_runner.py` connects to Interactive Brokers via `ib_insync`, streams live 5-minute bars, and generates entry signals with calibrated P(win) probabilities. The TradingView PineScript (`orb_long_nq.pine`) provides visual confirmation on paper trades.

Target platform: TopStep $50k Combine → funded account → IB live.
