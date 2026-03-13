# Algo-Strats

Quantitative trading strategies for futures markets.

## Strategies

| Folder | Instrument | Setup | Status |
|--------|-----------|-------|--------|
| [`nq-orb-long/`](./nq-orb-long/) | NQ Futures | Opening Range Breakout (Long) | In Development |

## Structure

Each strategy lives in its own folder with a self-contained codebase:

```
Algo-strats/
├── nq-orb-long/        # NQ ORB ML strategy (Steps 1–8)
│   ├── research/       # Core pipeline (events, model, backtest, live runner)
│   ├── research_utils/ # Data loading, feature engineering, WFO utilities
│   └── ...
└── README.md
```
