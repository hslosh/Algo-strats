# Step 8 — Deployment Checklist & Paper Trading Protocol

## Purpose
Translate the validated backtest (Step 7 PASS) into a concrete operational plan
for paper trading, performance monitoring, and eventual live deployment.

## Strategy Identity Card

| Parameter | Value |
|---|---|
| Strategy name | ORB-Long-NQ |
| Instrument | NQ (Nasdaq-100 E-mini futures) |
| Signal | Opening Range Breakout (long) |
| Model | Logistic regression, walk-forward validated |
| Probability threshold | 0.58 |
| Session | RTH only (09:30–16:00 ET) |
| Holding period | Intraday swing (30 min – 4 hours) |
| Entry | At ORB breakout bar close, if calibrated P(win) ≥ 0.58 |
| Stop loss | 1.0× ATR below entry |
| Take profit | 1.5× ATR above entry |
| Slippage budget | 0.50 pts per side (1.0 pts round-trip) |
| Commission | $4.50 per round-trip per contract |

## Position Sizing Rules

| Calibrated P(win) | Tier | Contracts (50k acct) |
|---|---|---|
| 0.58 – 0.64 | Conservative | 1 |
| 0.65 – 0.74 | Standard | 1–2 |
| 0.75+ | Aggressive | 2 |

Formula: `contracts = floor(equity × risk_pct / (SL_pts × $20))`
Capped at max_contracts = 2 for $50k account.

## Risk Management Rules

### Daily Controls
- Max daily loss: -$1,000 → stop trading for the day
- Max daily trades: 3
- Max concurrent positions: 1
- Consecutive loss pause: 3 losses in a row → sit out remainder of day

### Weekly Controls
- Max weekly loss: -$1,500 → stop trading until Monday

### Account-Level Controls
- Max drawdown: -$2,500 (5% of $50k) → circuit breaker triggered
- Circuit breaker recovery: trade at 50% size for 5 consecutive wins, then resume full size
- If circuit breaker fires twice in 30 days → stop trading for 2 weeks, review model

## Paper Trading Protocol

### Phase 1: Observation (Weeks 1–2)
- Run the model daily but do NOT trade
- Log every signal: timestamp, event, P(win), would-be entry/exit, SL/TP levels
- Compare model predictions to actual market outcomes
- Goal: Verify the model generates signals consistent with backtest frequency (~2/week)

### Phase 2: Paper Trading (Weeks 3–8)
- Execute all qualifying signals on paper/sim account
- Track execution quality: slippage vs. budget, fill times
- Log every trade with full metadata
- Weekly review: compare paper results to backtest expectations

### Phase 3: Evaluation (Week 9–10)
- Compute paper trading performance metrics
- Compare to backtest benchmarks (see pass/fail below)
- Decision: proceed to live, extend paper, or shelve

## Paper Trading Pass/Fail Criteria

Strategy must demonstrate during 6+ weeks of paper trading:

| Metric | Minimum | Backtest Reference |
|---|---|---|
| Trades | ≥ 10 | ~2.2/week |
| Win rate | ≥ 55% | 73% backtest |
| Profit factor | ≥ 1.3 | 4.04× backtest |
| Avg P&L/trade | > $0 | +$591 backtest |
| Max drawdown | < $3,000 | -$2,661 backtest |
| No week > -$1,500 | True | True in backtest |

**Relaxed targets**: Paper results will be worse than backtest (execution friction,
model refit lag, etc.). The targets above are ~50% haircuts from backtest numbers.

## Live Deployment Stages (Post-Paper)

### Stage 1: TopStep 50k Combine
- Same rules as paper trading
- Combine-specific constraints: daily loss limit, trailing drawdown
- Duration: until combine is passed or failed

### Stage 2: Funded Account (Micro)
- Start with 1 contract maximum regardless of signal
- First 30 trades at minimum size
- If first 30 trades profitable: unlock full sizing rules

### Stage 3: Scale
- After 3 consecutive profitable months on funded account
- Review model drift metrics before scaling
- Never exceed 2 contracts on $50k

## Monitoring & Model Maintenance

### Daily Checks
- [ ] Model generates predictions before RTH open
- [ ] Signal count consistent with expectations (~0.3/day)
- [ ] No data feed errors or missing bars
- [ ] Execution slippage within budget (< 1.0 pts RT)

### Weekly Review
- [ ] Win rate trailing 20 trades
- [ ] P&L curve vs. backtest equity curve
- [ ] Rolling 4-week Sharpe > 0
- [ ] Circuit breaker status (not triggered)

### Monthly Review
- [ ] Recalibrate model on latest data (re-run walk-forward)
- [ ] Check feature importance stability
- [ ] Compare monthly return to backtest distribution
- [ ] Review slippage actuals vs. budget

### Quarterly Review
- [ ] Full model refit with new data
- [ ] Run Steps 4–7 on updated dataset
- [ ] Compare OOS AUC to original (0.80) — flag if < 0.70
- [ ] Consider adding Sweep Low if edge stabilizes

## Kill Switches

Immediately halt trading and review if ANY of these occur:

1. **Max DD breached**: Account drops below $47,500 (-$2,500)
2. **Model degradation**: Rolling 30-trade win rate < 45%
3. **Signal drought**: < 1 signal per 2 weeks for 4+ weeks
4. **Regime break**: VIX > 40 sustained for 5+ days (model trained on normal vol)
5. **Execution failure**: Average slippage > 2.0 pts for 10+ trades
6. **Circuit breaker double-fire**: Triggers twice within 30 calendar days

## Data Requirements

### Minimum Data for Model Operation
- 5-minute OHLCV bars, NQ continuous contract
- Must have overnight session for context features
- RTH session for signal generation (09:30–16:00 ET)
- Latency: model prediction must complete before ORB window closes

### Data Sources
- Primary: [user to specify — broker feed, CME direct, etc.]
- Backup: [user to specify]
- Historical: Current dataset through 2026-01-13

## Files & Artifacts

| File | Purpose |
|---|---|
| `feature_engineering.py` | Bar-level features from OHLCV |
| `event_definitions.py` | ORB breakout event detection |
| `outcome_labeling.py` | Double-barrier SL/TP labeling |
| `event_features.py` | Event-level feature matrix |
| `statistical_research.py` | Feature validation & stats |
| `model_design.py` | Walk-forward model training |
| `strategy_construction.py` | Position sizing & simulation |
| `backtest_validation.py` | Cost-inclusive backtest |
| `deployment_checklist.py` | This step — final report generator |
