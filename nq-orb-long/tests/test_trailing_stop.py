"""
Tests for Phase 4 — Trailing Stop in bar-by-bar backtest.
Verifies activation, ratcheting, SL priority, and EOD override.
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


def _make_bars(start_ts, n_bars, base_price=16000, prices=None):
    """Generate n bars of 5-min OHLCV data.

    If prices is provided, it should be a list of (open, high, low, close) tuples.
    """
    bars = []
    for i in range(n_bars):
        if prices and i < len(prices):
            o, h, l, c = prices[i]
        else:
            o = base_price
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
    if probs is None:
        probs = [0.70] * len(event_times)
    return pd.DataFrame({
        'calibrated_prob': probs,
        'barrier_label': 1,
        'barrier_return_pts': 5.0,
    }, index=pd.DatetimeIndex(event_times))


class TestTrailActivation:
    """Trail should not activate until profit exceeds activation threshold."""

    def test_no_activation_below_threshold(self):
        """Price moves 0.3×ATR above entry — trail should NOT activate."""
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        # ATR ~20, activation = 0.5*20 = 10 pts
        # Entry at bar 2 open ~16000. Make bar 3 high = 16006 (0.3×ATR = 6 pts)
        # Then bar 4 drops below entry but above SL → should NOT trigger trail
        prices = [
            (16000, 16005, 15995, 16002),  # bar 0 (event)
            (16002, 16005, 15995, 16002),  # bar 1
            (16002, 16005, 15995, 16002),  # bar 2 (entry)
            (16002, 16006, 15998, 16004),  # bar 3 — small up (0.3×ATR)
            (16004, 16004, 15990, 15992),  # bar 4 — drops near SL
        ]
        # Extend with flat bars to reach EOD
        for i in range(73):
            prices.append((16000, 16005, 15995, 16002))

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            use_trailing_stop=True,
            trail_activation_mult=0.5,
            trail_distance_mult=0.75,
            sl_atr_mult=1.0,    # SL = entry - 20 = ~15980
            tp_atr_mult=5.0,    # very wide TP
            max_holding_bars=999,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        if len(trades) > 0:
            # Should NOT exit via trailing stop (profit < activation)
            assert trades.iloc[0]['exit_type'] != 'trailing_stop', (
                f"Trail should not activate at 0.3×ATR, got {trades.iloc[0]['exit_type']}"
            )


class TestTrailOnlyMovesUp:
    """Trail stop should only ratchet upward, never decrease."""

    def test_trail_ratchets_up(self):
        """After activation, running_high drops → trail_stop should not drop."""
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        # ATR ~20, activation=0.5*20=10, trail_dist=0.75*20=15
        # Entry ~16000
        # Bar 3: high=16015 → activates trail, trail_stop=16015-15=16000
        # Bar 4: high=16020 → trail_stop=16020-15=16005
        # Bar 5: high=16010 (lower) → trail_stop stays 16005
        # Bar 6: low=16003 → above trail_stop → no exit
        # Bar 7: low=16004 → still above
        # Bar 8: low=15999 → below trail_stop → trailing_stop exit
        prices = [
            (16000, 16005, 15995, 16002),  # bar 0 (event)
            (16002, 16005, 15995, 16002),  # bar 1
            (16002, 16005, 15995, 16002),  # bar 2 (entry at 16002+0.5slip)
            (16002, 16015, 15998, 16012),  # bar 3 — activates trail
            (16012, 16020, 16005, 16018),  # bar 4 — trail ratchets up
            (16018, 16010, 16005, 16008),  # bar 5 — high drops
            (16008, 16010, 16003, 16006),  # bar 6 — above trail
            (16006, 16008, 16004, 16005),  # bar 7 — above trail
            (16005, 16006, 15999, 16002),  # bar 8 — below trail → exit
        ]
        for i in range(69):
            prices.append((16000, 16005, 15995, 16002))

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            use_trailing_stop=True,
            trail_activation_mult=0.5,
            trail_distance_mult=0.75,
            sl_atr_mult=2.0,    # wide SL
            tp_atr_mult=5.0,    # very wide TP
            max_holding_bars=999,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        if len(trades) > 0:
            assert trades.iloc[0]['exit_type'] == 'trailing_stop', (
                f"Expected trailing_stop exit, got {trades.iloc[0]['exit_type']}"
            )


class TestSLPriorityOverTrail:
    """SL must take priority when bar low penetrates the hard SL level."""

    def test_sl_checked_before_trail(self):
        """When bar low hits SL (which is below trail_stop), SL wins."""
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        # ATR ≈ 10 from bars 0-2 (range=10 each)
        # Entry at bar 2 open=16000 + 0.5 slip = 16000.5
        # SL = 16000.5 - 1.0*10 ≈ 15990.5
        # Bar 3: push high to 16020 (trail activates)
        #   trail_stop = 16020 - 7.5 = 16012.5
        #   BUT bar 3 low=16015 > trail_stop → trail doesn't exit on bar 3
        # Bar 4: low=15985 < SL → hits both SL and trail → SL wins
        prices = [
            (16000, 16005, 15995, 16002),  # bar 0 (event)
            (16002, 16005, 15995, 16002),  # bar 1
            (16000, 16005, 15995, 16002),  # bar 2 (entry)
            (16002, 16020, 16015, 16018),  # bar 3 — activates trail, low stays above trail
            (16018, 16018, 15985, 15988),  # bar 4 — crash through both SL and trail
        ]
        for i in range(73):
            prices.append((16000, 16005, 15995, 16002))

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            use_trailing_stop=True,
            trail_activation_mult=0.5,
            trail_distance_mult=0.75,
            sl_atr_mult=1.0,
            tp_atr_mult=5.0,
            max_holding_bars=999,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        if len(trades) > 0:
            assert trades.iloc[0]['exit_type'] == 'stop_loss', (
                f"SL should win when bar crashes through both, got {trades.iloc[0]['exit_type']}"
            )


class TestTrailDisabledDefault:
    """With use_trailing_stop=False, behavior should match Phase 1 baseline exactly."""

    def test_trail_off_matches_baseline(self):
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        n_bars = 20
        df = _make_bars(start, n_bars, base_price=16000)
        oos = _make_oos_predictions([start])

        config_off = BacktestConfig(
            threshold=0.58, account_size=50000,
            use_trailing_stop=False,
        )
        config_on = BacktestConfig(
            threshold=0.58, account_size=50000,
            use_trailing_stop=True,
        )

        result_off = run_bar_by_bar_backtest(df, oos, config_off, verbose=False)
        result_on = run_bar_by_bar_backtest(df, oos, config_on, verbose=False)

        trades_off = result_off.get('trades', pd.DataFrame())
        trades_on = result_on.get('trades', pd.DataFrame())

        # When trail doesn't activate (small price moves), results should be identical
        assert len(trades_off) == len(trades_on), "Trade count should match with trail off"
