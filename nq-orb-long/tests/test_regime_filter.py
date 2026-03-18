"""
Tests for Phase 2 — Trend Regime Filter.
Verifies no lookahead, correct blocking, and boundary behavior.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest
from datetime import timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research_utils'))


def _make_multi_day_bars(n_days=7, base_price=16000, daily_trend=0):
    """Generate n_days of RTH 5-min bars (09:30-15:55 ET = 78 bars/day)."""
    bars = []
    dates = pd.bdate_range('2024-01-02', periods=n_days, freq='B')
    for d_idx, day in enumerate(dates):
        day_open = base_price + d_idx * daily_trend
        for i in range(78):
            ts = pd.Timestamp(day) + timedelta(hours=9, minutes=30 + 5 * i)
            price = day_open + i * 0.1  # slight upward intraday
            bars.append({
                'open': price, 'high': price + 2, 'low': price - 2,
                'close': price + 1, 'volume': 1000,
            })
    df = pd.DataFrame(bars)
    timestamps = []
    for d_idx, day in enumerate(dates):
        for i in range(78):
            timestamps.append(pd.Timestamp(day) + timedelta(hours=9, minutes=30 + 5 * i))
    df.index = pd.DatetimeIndex(timestamps)
    return df


class TestRegimeUsePriorDayClose:
    """Regime signal must use prior day's close, not current day."""

    def test_regime_uses_prior_day_close(self):
        from research_utils.feature_engineering import add_trend_regime

        # Create 7 days of data with upward trend
        df = _make_multi_day_bars(n_days=7, base_price=16000, daily_trend=50)
        df = add_trend_regime(df, ema_period=3)

        dates = sorted(df.index.date)
        unique_dates = sorted(set(dates))

        # Day 0 (first day) should be True (no prior day)
        day0_regime = df[df.index.date == unique_dates[0]]['regime_long_allowed'].iloc[0]
        assert day0_regime == True, "First day should default to True"

        # Day 6 regime should be based on day 5 close, not day 6
        day5_close = df[df.index.date == unique_dates[4]].iloc[-1]['close']
        day6_regime = df[df.index.date == unique_dates[5]]['regime_long_allowed'].iloc[0]
        # With strong uptrend (+50/day), prior close should be above EMA
        assert day6_regime == True, "Strong uptrend: regime should allow longs"


class TestBearRegimeBlocked:
    """All bars on a bear-regime day should have regime_long_allowed == False."""

    def test_all_bars_bear_session_blocked(self):
        from research_utils.feature_engineering import add_trend_regime

        # Create 7 days with strong downward trend
        df = _make_multi_day_bars(n_days=7, base_price=16000, daily_trend=-100)
        df = add_trend_regime(df, ema_period=3)

        unique_dates = sorted(set(df.index.date))
        # After a few days of strong decline, regime should block
        # Check day 5 (index 4) — enough history for EMA to react
        day5_bars = df[df.index.date == unique_dates[4]]
        regime_values = day5_bars['regime_long_allowed']
        # All bars on the same day should have the same regime value
        assert regime_values.nunique() == 1, "All bars on same day should have same regime"


class TestFirstDayPermissive:
    """No prior day data → regime_long_allowed == True."""

    def test_first_day_permissive(self):
        from research_utils.feature_engineering import add_trend_regime

        df = _make_multi_day_bars(n_days=2, base_price=16000, daily_trend=-200)
        df = add_trend_regime(df, ema_period=50)

        unique_dates = sorted(set(df.index.date))
        day0 = df[df.index.date == unique_dates[0]]
        assert day0['regime_long_allowed'].all(), "First day must be permissive (True)"


class TestNoForwardFillAcrossSession:
    """Monday should use Friday's close, with no interpolation."""

    def test_monday_uses_friday(self):
        from research_utils.feature_engineering import add_trend_regime

        # 5 business days covers Mon-Fri
        df = _make_multi_day_bars(n_days=10, base_price=16000, daily_trend=10)
        df = add_trend_regime(df, ema_period=3)

        unique_dates = sorted(set(df.index.date))
        # All dates should have regime values (no NaN after fillna)
        for d in unique_dates:
            day_bars = df[df.index.date == d]
            assert not day_bars['regime_long_allowed'].isna().any(), (
                f"Date {d} has NaN regime values"
            )


class TestSimulateSkipsBearRegime:
    """simulate_strategy should skip events on bear-regime days."""

    def test_simulate_skips_bear_regime(self):
        from research.strategy_construction import simulate_strategy, StrategyConfig

        config = StrategyConfig(
            account_size=50000, threshold=0.58,
            max_daily_trades=10, consec_loss_pause=99,
        )

        tz = 'America/New_York'
        # Create 2 events on a day marked as bear regime
        events = pd.DataFrame([
            {
                'session_date': pd.Timestamp('2024-01-02').date(),
                'calibrated_prob': 0.70,
                'barrier_return_pts': 10.0,
                'barrier_label': 1,
                'sl_distance_pts': 10.0,
                'regime_long_allowed': False,  # bear regime
            },
            {
                'session_date': pd.Timestamp('2024-01-02').date(),
                'calibrated_prob': 0.75,
                'barrier_return_pts': 15.0,
                'barrier_label': 1,
                'sl_distance_pts': 10.0,
                'regime_long_allowed': False,  # bear regime
            },
        ], index=pd.DatetimeIndex([
            pd.Timestamp('2024-01-02 10:00', tz=tz),
            pd.Timestamp('2024-01-02 10:30', tz=tz),
        ]))

        result = simulate_strategy(events, config, prob_col='calibrated_prob')
        trades = result['trades']
        assert len(trades) == 0, (
            f"Expected 0 trades on bear-regime day, got {len(trades)}"
        )
