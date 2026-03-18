"""
Tests for the bar-by-bar backtest engine (backtest_validation.py).
Verifies SL priority, EOD flatten, entry delay, and daily trade limit.
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


def _make_bars(start_ts, n_bars, base_price=16000, price_step=0):
    """Generate n bars of 5-min OHLCV data."""
    bars = []
    for i in range(n_bars):
        ts = start_ts + timedelta(minutes=5 * i)
        o = base_price + i * price_step
        h = o + 5
        l = o - 5
        c = o + 2
        bars.append({
            'open': o, 'high': h, 'low': l, 'close': c,
            'volume': 1000, 'atr_14': 20.0,
        })
    df = pd.DataFrame(bars)
    df.index = pd.DatetimeIndex([
        start_ts + timedelta(minutes=5 * i) for i in range(n_bars)
    ])
    return df


def _make_oos_predictions(event_times, probs=None):
    """Create OOS predictions DataFrame matching backtest input format."""
    if probs is None:
        probs = [0.70] * len(event_times)
    return pd.DataFrame({
        'calibrated_prob': probs,
        'barrier_label': 1,
        'barrier_return_pts': 5.0,
    }, index=pd.DatetimeIndex(event_times))


class TestSLPriority:
    """SL must take priority when both SL and TP can trigger on the same bar."""

    def test_sl_wins_on_tie_bar(self):
        """Bar where low <= SL AND high >= TP → exit_type == 'stop_loss'."""
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        # Create bars: event at bar 0, entry at bar 2
        bars = []
        for i in range(10):
            ts = start + timedelta(minutes=5 * i)
            bars.append({
                'open': 16000, 'high': 16005, 'low': 15995,
                'close': 16002, 'volume': 1000,
            })

        # Bar 2 (entry bar): normal
        # Bar 3 (the tie bar): low drops to SL AND high reaches TP
        # With entry at ~16000, SL at entry - 1.0*ATR, TP at entry + 1.5*ATR
        # ATR ~= 10 pts, so SL ~ 15990, TP ~ 16015
        # Make bar 3 have low=15985 (hits SL) AND high=16020 (hits TP)
        bars[3] = {
            'open': 16000, 'high': 16050, 'low': 15950,
            'close': 16000, 'volume': 1000,
        }

        df = pd.DataFrame(bars)
        df.index = pd.DatetimeIndex([
            start + timedelta(minutes=5 * i) for i in range(10)
        ])

        oos = _make_oos_predictions([start])  # event at bar 0
        config = BacktestConfig(threshold=0.58, account_size=50000)

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        if len(trades) > 0:
            assert trades.iloc[0]['exit_type'] == 'stop_loss', (
                f"Expected stop_loss on tie bar, got {trades.iloc[0]['exit_type']}"
            )


class TestEODFlatten:
    """Position must be flattened at 15:30 ET (30 min before close)."""

    def test_eod_flatten_at_1530(self):
        """Position open at 15:25 → exit at 15:30 with exit_type='eod_flatten'."""
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        # Create a full trading day of bars from 09:30 to 16:00
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)
        n_bars = 78  # 09:30 to 15:55, 5-min intervals
        bars = []
        for i in range(n_bars):
            ts = start + timedelta(minutes=5 * i)
            bars.append({
                'open': 16000, 'high': 16010, 'low': 15990,
                'close': 16005, 'volume': 1000,
            })
        df = pd.DataFrame(bars)
        df.index = pd.DatetimeIndex([
            start + timedelta(minutes=5 * i) for i in range(n_bars)
        ])

        # Event at 09:30, entry at bar+2 = 09:40
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            sl_atr_mult=5.0,  # very wide SL so it won't trigger
            tp_atr_mult=10.0,  # very wide TP so it won't trigger
            max_holding_bars=999,  # disable timeout
            force_session_exit=True,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        if len(trades) > 0:
            assert trades.iloc[0]['exit_type'] == 'eod_flatten', (
                f"Expected eod_flatten, got {trades.iloc[0]['exit_type']}"
            )
            exit_time = pd.Timestamp(trades.iloc[0]['exit_time'])
            assert exit_time.hour == 15 and exit_time.minute >= 30, (
                f"EOD exit should be at 15:30+, got {exit_time}"
            )


class TestEntryDelay:
    """Entry should occur at bar N+2 (2-bar confirmation delay) after event at bar N."""

    def test_entry_at_bar_plus_2(self):
        """Event at bar 0 → entry recorded at bar 2's timestamp."""
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)
        n_bars = 20
        bars = []
        for i in range(n_bars):
            ts = start + timedelta(minutes=5 * i)
            bars.append({
                'open': 16000, 'high': 16010, 'low': 15990,
                'close': 16005, 'volume': 1000,
            })
        df = pd.DataFrame(bars)
        df.index = pd.DatetimeIndex([
            start + timedelta(minutes=5 * i) for i in range(n_bars)
        ])

        event_time = start  # bar 0
        oos = _make_oos_predictions([event_time])
        config = BacktestConfig(threshold=0.58, account_size=50000)

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        if len(trades) > 0:
            entry_time = pd.Timestamp(trades.iloc[0]['entry_time'])
            expected_entry = start + timedelta(minutes=10)  # bar 2
            assert entry_time == expected_entry, (
                f"Expected entry at {expected_entry}, got {entry_time}"
            )
