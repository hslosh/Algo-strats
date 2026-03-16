# NQ Mean-Reversion V4

NQ (Nasdaq-100 E-mini) futures mean-reversion strategy, developed on 17 years of 5-minute OHLCV data (2008–2026, ~1.2M bars).

## Best Config (BEST_A + Signal-Tier Sizing)

| Parameter | Value |
|---|---|
| Entry threshold | 3.0 (adaptive, vol_scale=0.8) |
| Sizing | 1ct ≥ 3.0 / 2ct ≥ 3.3 / 3ct ≥ 4.0 |
| Disaster stop | 8.0 ATR |
| Early cut | bars=4, loss=1.5, mfe=0.2 |

## Validated Results (OOS 2020–2025)

| Metric | Value |
|---|---|
| Trades | 96 |
| Net P&L | $32,889 |
| Sharpe | 3.89 |
| Win Rate | 65.6% |
| Max Drawdown | -$5,058 |
| Bootstrap P(profitable) | 99.5% |

## File Structure

```
nq-meanrev-v4/
├── strategy_v4_production.py   # Production-ready class (all 5 validation checks pass)
├── strategy_v4.py              # V4 core strategy (adaptive threshold)
├── strategy_v2.py              # Base class: StrategyConfig, Trade, composite signal
├── run_v4_refine.py            # Canonical make_v4_config() builder
└── research_utils/
    ├── feature_engineering.py  # 88 features, build_features() pipeline
    ├── wfo_and_robustness.py   # Walk-forward validation + Monte Carlo
    ├── backtest_runner.py      # Custom backtest loop
    └── data_pipeline.py        # Data loading/cleaning
```

## Composite Signal (6 features)

`vwap_distance×2.5 + rsi_28×1.5 + pctrank_24×1.0 + log_ret_6×1.0 + log_ret_48×0.5 + natr_12×0.3`

Z-scored over a 21-day rolling window, averaged across available features.

## Key Config Notes

Many `StrategyConfig` defaults are wrong for V4. Always build via `make_v4_config()`:
- `max_efficiency_ratio=1.0` (disables ER filter)
- `adverse_excursion_atr=99.0` (disables adverse exit)
- `time_stop_bars=20`, `max_holding_bars=72`, `atr_target_multiple=10.0`
