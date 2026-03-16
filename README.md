# Algo-Strats

Quantitative trading strategies for NQ (Nasdaq-100 E-mini) futures.

## Strategies

| Folder | Instrument | Setup | Edge | Status |
|--------|-----------|-------|------|--------|
| [`nq-orb-long/`](./nq-orb-long/) | NQ Futures | Opening Range Breakout (Long, ML-filtered) | Directional breakout after 30-min opening range | In Development |
| [`nq-meanrev-v4/`](./nq-meanrev-v4/) | NQ Futures | Mean-Reversion (Composite Z-Score) | Adaptive threshold + signal-tier sizing | Validated |

## Strategy Summaries

### `nq-orb-long` — ML-Enhanced Opening Range Breakout
- **Direction:** Long only, RTH (09:30–16:00 ET)
- **Entry:** Close above OR high + 1 tick, filtered by calibrated ML P(win) ≥ 0.58
- **Risk:** 1.0 × ATR14 stop / 1.5 × ATR14 target / 48-bar timeout
- **Pipeline:** 8-step institutional framework (event → label → features → stats → model → construction → backtest → deployment)
- **Live target:** TopStep $50k Combine → IB live

### `nq-meanrev-v4` — Adaptive Mean-Reversion V4
- **Signal:** Composite Z-score (VWAP distance, RSI, % rank, log returns, NATR) over 21-day rolling window
- **Sizing:** 1ct @ sig ≥ 3.0 / 2ct @ sig ≥ 3.3 / 3ct @ sig ≥ 4.0
- **Validated results (OOS 2020–2025):** 96 trades, $32,889 net P&L, Sharpe 3.89, 65.6% WR, max DD -$5,058
- **Bootstrap:** 99.5% P(profitable) across 10,000 resampled paths

## Structure

Each strategy lives in its own self-contained folder:

```
Algo-strats/
├── nq-orb-long/            # NQ ORB ML strategy (Steps 1–8)
│   ├── research/           # Core pipeline (events, model, backtest, live runner)
│   ├── research_utils/     # Data loading, feature engineering, WFO utilities
│   └── README.md
├── nq-meanrev-v4/          # NQ mean-reversion V4 (validated)
│   ├── strategy_v4_production.py
│   ├── strategy_v4.py
│   ├── strategy_v2.py
│   ├── run_v4_refine.py
│   ├── research_utils/     # Feature engineering, WFO, backtest runner
│   └── README.md
└── README.md
```

## Data

Historical NQ 5-minute continuous data from [FirstRateData](https://firstratedata.com) (2008–2026, ~1.2M bars, ratio-adjusted, US Eastern Time). The full dataset is excluded from version control due to size. Sample files for each timeframe are included in each strategy folder.
