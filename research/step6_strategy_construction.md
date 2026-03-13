# Step 6 — Strategy Construction

**Date:** 2026-03-03
**Status:** Implementation
**Depends on:** Steps 1-5 (Event Detection → Model Design)

---

## Purpose

Transform Step 5's calibrated model outputs into a complete, tradeable strategy with:
- Explicit entry/exit rules
- ATR-based position sizing scaled by model confidence
- Daily/weekly risk limits and drawdown circuit breaker
- Event-level simulation using walk-forward OOS predictions
- Monte Carlo robustness testing

This step bridges **research** (Steps 1-5) → **live execution** (Steps 7-8).

---

## Strategy Specification

### Signal: ORB Long (Primary)

**Step 5 determined:**
- Logistic regression (AUC = 0.80) is the best model
- `session_range_position` is the dominant feature (importance 1.12)
- Optimal threshold range: 0.42-0.48 (30 TPY with EV +14-17 pts)

**Entry rules:**
1. ORB Long event detected (first close above opening range high)
2. Model generates calibrated P(win)
3. P(win) ≥ threshold → TRADE; P(win) < threshold → SKIP

**Exit rules (from Step 2):**
- Take-profit: 1.5× ATR above entry
- Stop-loss: 1.0× ATR below entry
- Time stop: 48 bars (4 hours) or session end, whichever first
- R:R = 1.5:1

### Secondary Signal: Sweep Low (Optional Overlay)

Sweep Low has decent AUC (0.71) but unstable features and negative unconditional EV.
**Decision:** Include at reduced size (50%) only when model confidence is high (P(win) > 0.50).
This adds ~10-15 diversification trades per year without materially increasing risk.

---

## Position Sizing

### Base Size
- Account: $50,000
- Max risk per trade: 1.5% = $750
- NQ point value: $20/point
- Base size: `risk_budget / (SL_distance_pts × $20)` → typically 1-2 contracts

### Confidence Scaling
The model's calibrated probability modulates position size:

| P(win) Zone | Size Multiplier | Rationale |
|---|---|---|
| < threshold | 0.0 (NO TRADE) | Below confidence minimum |
| threshold - 0.55 | 0.50× | Moderate confidence |
| 0.55 - 0.65 | 0.75× | Good confidence |
| > 0.65 | 1.00× | High confidence |

This naturally concentrates capital on the highest-probability setups.

---

## Risk Management

### Daily Limits
| Rule | Value | Action |
|---|---|---|
| Max daily loss | -$1,000 (2% of account) | Stop trading for the day |
| Max daily trades | 3 | Prevent over-trading |
| Max concurrent positions | 1 | No stacking (intraday swing) |
| Max consecutive losses before pause | 3 | Cool-off: skip next signal |

### Weekly/Monthly Limits
| Rule | Value | Action |
|---|---|---|
| Max weekly loss | -$1,500 (3%) | Reduce to 50% size for remainder |
| Max drawdown from peak | -$2,500 (5%) | Circuit breaker: stop all trading |

### Recovery Protocol
After circuit breaker triggers:
- Resume at 50% size after 5 trading days
- Return to full size after 3 consecutive winning days at 50%

---

## Event-Level Simulation

The simulation uses OOS predictions from Step 5's walk-forward:

```
For each event in chronological order:
  1. Check risk limits (daily loss, weekly loss, drawdown)
  2. Get calibrated P(win) from model
  3. Apply threshold → trade or skip
  4. If trade:
     a. Compute position size (ATR-based × confidence multiplier)
     b. Record entry at event price
     c. Use pre-computed exit (TP/SL/timeout from Step 2)
     d. Compute P&L = (exit - entry) × contracts × $20
  5. Update account equity, daily P&L, drawdown tracking
```

This is NOT a bar-by-bar simulation — it uses the already-computed outcomes from
the double-barrier labeling. This is appropriate for the research phase because:
- The barriers are computed from actual price paths (no look-ahead)
- Walk-forward ensures the model never sees future data
- Bar-by-bar simulation is Step 7 (validation)

---

## Performance Metrics

### Core Metrics
- **Total P&L** and **P&L per trade** (in points and dollars)
- **Win rate** and **Profit factor**
- **Sharpe ratio** (annualized, daily returns)
- **Sortino ratio** (downside-only volatility)
- **Max drawdown** (peak-to-trough in dollars and percent)
- **Calmar ratio** (annual return / max drawdown)
- **Average winner / average loser** ratio
- **Expectancy** per trade (EV in dollars)

### Robustness Metrics
- **Longest winning/losing streak**
- **Recovery time** from max drawdown (in trading days)
- **Monthly win rate** (% of months profitable)
- **Rolling 3-month Sharpe** (consistency check)

### Monte Carlo Simulation
Randomize trade order 10,000 times to build confidence intervals for:
- Max drawdown distribution (P95, P99)
- Final equity distribution
- Worst-case 12-month return

This answers: "Even with bad luck in trade ordering, does the strategy survive?"

---

## Success Criteria

The strategy PASSES if:
1. Net profit > 0 over the OOS period
2. Sharpe ≥ 1.0 (annualized)
3. Max drawdown < $2,500 (5% of $50k)
4. Profit factor ≥ 1.5
5. Monthly win rate ≥ 55%
6. Monte Carlo P95 max drawdown < $3,500 (7% of $50k)

---

## Implementation

See `strategy_construction.py` for the full implementation with:
- `StrategyConfig` — all parameters in one place
- `compute_position_size()` — ATR-based sizing with confidence scaling
- `simulate_strategy()` — event-level P&L simulation with risk limits
- `compute_performance_metrics()` — complete analytics suite
- `monte_carlo_drawdown()` — randomized trade-order robustness
- `run_full_step6()` — one-call entry point
