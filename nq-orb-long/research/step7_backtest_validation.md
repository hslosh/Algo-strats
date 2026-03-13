# Step 7 — Backtesting & Validation

**Date:** 2026-03-03
**Status:** Implementation
**Depends on:** Steps 1-6 (Full Pipeline)

---

## Purpose

Validate the complete strategy by running a bar-by-bar simulation that processes
every 5-minute bar sequentially — exactly as it would execute in live trading.

Step 6 used pre-computed event outcomes (event-level simulation). Step 7 adds:
- True bar-by-bar position tracking with mark-to-market
- Realistic execution: slippage + commission modeling
- Daily equity curve with proper time-series risk metrics
- Rolling performance windows to detect regime decay
- Out-of-sample holdout: last 6 months reserved as pure test

---

## Strategy Parameters (from Step 6 Results)

| Parameter | Value | Source |
|---|---|---|
| Event | ORB Long | Step 4-5: best edge, AUC 0.80 |
| Model | Logistic Regression | Step 5: beat RF, linear signal |
| Threshold | 0.58 | Step 6: sweet spot (DD<$2,625, Sharpe 10.7) |
| Stop-loss | 1.0× ATR | Step 2 parameters |
| Take-profit | 1.5× ATR | Step 2 parameters |
| Max hold | 48 bars / session end | Step 2 parameters |
| Risk per trade | 1.5% of equity | Step 6 sizing |
| Max contracts | 2 | Account constraint |
| Max daily loss | -$1,000 | Step 6 risk rules |
| Slippage | 0.50 pts per side (1.0 round-trip) | Conservative NQ estimate |
| Commission | $4.50 per round-trip per contract | Standard NQ rate |

---

## Validation Layers

### Layer 1: Bar-by-Bar Replay
Process each 5-min bar chronologically. When an ORB Long event fires:
1. Check model probability (walk-forward, no look-ahead)
2. Check risk limits (daily loss, drawdown)
3. Enter at next bar's open + slippage
4. Track position: update SL/TP/timeout each bar
5. Exit when barrier hit or time expired

### Layer 2: Transaction Costs
- Entry slippage: 0.50 pts (half the bid-ask spread)
- Exit slippage: 0.50 pts
- Commission: $4.50 per round-trip per contract
- Total cost per trade: ~(1.0 pts × $20 + $4.50) × contracts

### Layer 3: Rolling Stability
- 6-month rolling Sharpe ratio
- 12-month rolling total P&L
- Detect if recent performance degrades below historical average

### Layer 4: Final Holdout
Last 6 months of data reserved — model never trained or calibrated on this period.
This is the ultimate test of forward generalization.

---

## Success Criteria

The backtest PASSES if:
1. Net profit positive after all costs
2. Sharpe ≥ 1.5 (after costs)
3. Max DD < $3,000 (6% — slightly relaxed for costs)
4. Profit factor ≥ 1.3 (after costs)
5. Holdout period profitable (last 6 months)
6. Rolling 6-month Sharpe never below -1.0 (no catastrophic periods)

---

## Implementation

See `backtest_validation.py` for the full implementation.
