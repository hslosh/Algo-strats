"""
Tests for Phase A — ORB Short direction support in bar-by-bar backtest.
TDD: These tests are written BEFORE implementation.
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
    """Generate n bars of 5-min OHLCV data."""
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


# =====================================================================
# SHORT ENTRY TESTS
# =====================================================================

class TestShortEntryPrice:
    """Short entry should be open - slippage (not open + slippage)."""

    def test_short_entry_price_below_open(self):
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        prices = [
            (16000, 16005, 15995, 16002),  # bar 0 (event)
            (16002, 16005, 15995, 16002),  # bar 1
            (16000, 16005, 15995, 16002),  # bar 2 (entry)
        ]
        # Extend with bars that hit TP for short (low enough)
        for i in range(75):
            prices.append((15950, 15955, 15945, 15950))

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            direction='short',
            sl_atr_mult=2.0,   # wide SL (not too wide for position sizing)
            tp_atr_mult=1.5,
            max_holding_bars=999,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        assert len(trades) > 0, "Should produce at least one trade"
        entry = trades.iloc[0]['entry_price']
        bar2_open = prices[2][0]
        # Short entry = open - slippage (0.50)
        assert entry < bar2_open, (
            f"Short entry {entry} should be below open {bar2_open}"
        )
        assert abs(entry - (bar2_open - 0.50)) < 0.01


class TestShortSLAboveEntry:
    """Short stop loss should be ABOVE entry price."""

    def test_short_sl_above_entry(self):
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        # ATR ~20, SL mult=1.0 → SL = entry + 20
        # Make bar 3 high go above SL to trigger it
        prices = [
            (16000, 16005, 15995, 16002),  # bar 0 (event)
            (16002, 16005, 15995, 16002),  # bar 1
            (16000, 16005, 15995, 16002),  # bar 2 (entry at 15999.5)
            (16002, 16025, 15995, 16020),  # bar 3 — high hits SL (entry + 20 ≈ 16019.5)
        ]
        for i in range(74):
            prices.append((16000, 16005, 15995, 16002))

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            direction='short',
            sl_atr_mult=1.0,
            tp_atr_mult=5.0,  # very wide TP
            max_holding_bars=999,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        assert len(trades) > 0, "Should produce at least one trade"
        assert trades.iloc[0]['exit_type'] == 'stop_loss', (
            f"Expected stop_loss, got {trades.iloc[0]['exit_type']}"
        )
        # SL exit price should be above entry price (loss for short)
        assert trades.iloc[0]['exit_price'] > trades.iloc[0]['entry_price']


class TestShortTPBelowEntry:
    """Short take profit should be BELOW entry price."""

    def test_short_tp_below_entry(self):
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        # ATR ~20, TP mult=1.5 → TP = entry - 30
        # Entry at bar 2 open - 0.5 = 15999.5, TP ≈ 15969.5
        prices = [
            (16000, 16005, 15995, 16002),  # bar 0 (event)
            (16002, 16005, 15995, 16002),  # bar 1
            (16000, 16005, 15995, 16002),  # bar 2 (entry)
            (16002, 16005, 15965, 15970),  # bar 3 — low hits TP
        ]
        for i in range(74):
            prices.append((16000, 16005, 15995, 16002))

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            direction='short',
            sl_atr_mult=2.0,   # wide SL (not too wide for position sizing)
            tp_atr_mult=1.5,
            max_holding_bars=999,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        assert len(trades) > 0, "Should produce at least one trade"
        assert trades.iloc[0]['exit_type'] == 'take_profit', (
            f"Expected take_profit, got {trades.iloc[0]['exit_type']}"
        )
        # TP exit price should be below entry price (profit for short)
        assert trades.iloc[0]['exit_price'] < trades.iloc[0]['entry_price']


# =====================================================================
# SHORT EXIT DIRECTION TESTS
# =====================================================================

class TestShortSLTriggersOnHigh:
    """Short SL should trigger on bar HIGH (not bar low)."""

    def test_short_sl_trigger_on_high(self):
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        # Entry at bar 2: open=16000, entry=15999.5 (short)
        # SL = 15999.5 + 20 = 16019.5
        # Bar 3: high=16020 (hits SL), low=15990 (would hit TP for long)
        prices = [
            (16000, 16005, 15995, 16002),  # bar 0 (event)
            (16002, 16005, 15995, 16002),  # bar 1
            (16000, 16005, 15995, 16002),  # bar 2 (entry)
            (16002, 16020, 15990, 16010),  # bar 3 — high >= SL
        ]
        for i in range(74):
            prices.append((16000, 16005, 15995, 16002))

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            direction='short',
            sl_atr_mult=1.0,  # SL = entry + 20
            tp_atr_mult=5.0,  # wide TP
            max_holding_bars=999,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        assert len(trades) > 0
        assert trades.iloc[0]['exit_type'] == 'stop_loss'


class TestShortTPTriggersOnLow:
    """Short TP should trigger on bar LOW (not bar high)."""

    def test_short_tp_trigger_on_low(self):
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        # Entry at bar 2: open=16000, entry=15999.5
        # TP = 15999.5 - 30 = 15969.5
        # Bar 3: low=15965 (hits TP), high=16005 (does NOT hit SL)
        prices = [
            (16000, 16005, 15995, 16002),  # bar 0 (event)
            (16002, 16005, 15995, 16002),  # bar 1
            (16000, 16005, 15995, 16002),  # bar 2 (entry)
            (15990, 16005, 15965, 15975),  # bar 3 — low <= TP
        ]
        for i in range(74):
            prices.append((16000, 16005, 15995, 16002))

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            direction='short',
            sl_atr_mult=2.0,   # wide SL so high doesn't hit it
            tp_atr_mult=1.5,   # TP = entry - 30
            max_holding_bars=999,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        assert len(trades) > 0
        assert trades.iloc[0]['exit_type'] == 'take_profit'


class TestShortSLPriorityOnTie:
    """When both SL and TP are hit on same bar, SL wins (conservative)."""

    def test_short_sl_priority_on_tie(self):
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        # Entry at bar 2: open=16000, entry=15999.5
        # SL = 15999.5 + 20 = 16019.5 (ATR=20, mult=1.0)
        # TP = 15999.5 - 30 = 15969.5 (ATR=20, mult=1.5)
        # Bar 3: high=16025 (hits SL), low=15960 (hits TP) → SL wins
        prices = [
            (16000, 16005, 15995, 16002),  # bar 0 (event)
            (16002, 16005, 15995, 16002),  # bar 1
            (16000, 16005, 15995, 16002),  # bar 2 (entry)
            (16000, 16025, 15960, 16000),  # bar 3 — both SL and TP hit
        ]
        for i in range(74):
            prices.append((16000, 16005, 15995, 16002))

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            direction='short',
            sl_atr_mult=1.0,
            tp_atr_mult=1.5,
            max_holding_bars=999,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        assert len(trades) > 0
        assert trades.iloc[0]['exit_type'] == 'stop_loss', (
            f"SL should win on tie, got {trades.iloc[0]['exit_type']}"
        )


# =====================================================================
# SHORT TRAILING STOP
# =====================================================================

class TestShortTrailUsesRunningLow:
    """Short trail should track running_low (min of bar lows), not running_high."""

    def test_short_trail_uses_running_low(self):
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        # ATR ~20, activation=0.5*20=10, trail_dist=0.75*20=15
        # Entry at bar 2: open=16000, entry=15999.5 (short)
        # Bar 3: low=15985 → profit=14.5 > 10 → activates trail
        #   running_low=15985, trail_stop=15985+15=16000
        # Bar 4: low=15980 → running_low=15980, trail_stop=15980+15=15995
        #   (trail ratchets DOWN, not up)
        # Bar 5: low=15990 → running_low stays 15980, trail_stop stays 15995
        # Bar 6: high=15996 → above trail_stop → trail exit
        prices = [
            (16000, 16005, 15995, 16002),  # bar 0 (event)
            (16002, 16005, 15995, 16002),  # bar 1
            (16000, 16005, 15995, 16002),  # bar 2 (entry at 15999.5)
            (15998, 16000, 15985, 15988),  # bar 3 — activates trail
            (15988, 15992, 15980, 15982),  # bar 4 — trail ratchets down
            (15982, 15990, 15990, 15988),  # bar 5 — low doesn't improve
            (15988, 15996, 15985, 15990),  # bar 6 — high=15996 > trail_stop=15995
        ]
        for i in range(71):
            prices.append((16000, 16005, 15995, 16002))

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            direction='short',
            use_trailing_stop=True,
            trail_activation_mult=0.5,
            trail_distance_mult=0.75,
            sl_atr_mult=2.0,    # wide SL
            tp_atr_mult=5.0,    # very wide TP
            max_holding_bars=999,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        assert len(trades) > 0, "Should produce at least one trade"
        assert trades.iloc[0]['exit_type'] == 'trailing_stop', (
            f"Expected trailing_stop exit, got {trades.iloc[0]['exit_type']}"
        )


# =====================================================================
# EOD FLATTEN (same for both directions)
# =====================================================================

class TestShortEODFlatten:
    """Short position at 15:30 ET should flatten (same as long)."""

    def test_short_eod_flatten(self):
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        # Start at 15:00 so bar at 15:30 hits EOD cutoff
        start = pd.Timestamp('2024-01-02 15:00', tz=tz)

        prices = [
            (16000, 16005, 15995, 16002),  # 15:00 (event)
            (16002, 16005, 15995, 16002),  # 15:05
            (16000, 16005, 15995, 16002),  # 15:10 (entry)
            (16002, 16005, 15995, 16002),  # 15:15
            (16002, 16005, 15995, 16002),  # 15:20
            (16002, 16005, 15995, 16002),  # 15:25
            (16002, 16005, 15995, 16002),  # 15:30 — EOD flatten
        ]

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])
        config = BacktestConfig(
            threshold=0.58, account_size=50000,
            direction='short',
            sl_atr_mult=2.0,   # wide SL
            tp_atr_mult=5.0,   # wide TP
            max_holding_bars=999,
            force_session_exit=True,
        )

        result = run_bar_by_bar_backtest(df, oos, config, verbose=False)
        trades = result.get('trades', pd.DataFrame())

        assert len(trades) > 0, "Should produce at least one trade"
        assert trades.iloc[0]['exit_type'] == 'eod_flatten', (
            f"Expected eod_flatten, got {trades.iloc[0]['exit_type']}"
        )


# =====================================================================
# REGRESSION: LONG DIRECTION UNCHANGED
# =====================================================================

class TestLongDirectionUnchanged:
    """Adding direction param should not change existing long behavior."""

    def test_long_direction_unchanged(self):
        from research.backtest_validation import run_bar_by_bar_backtest, BacktestConfig

        tz = 'America/New_York'
        start = pd.Timestamp('2024-01-02 09:30', tz=tz)

        prices = [
            (16000, 16005, 15995, 16002),  # bar 0 (event)
            (16002, 16005, 15995, 16002),  # bar 1
            (16000, 16005, 15995, 16002),  # bar 2 (entry)
            (16002, 16040, 15995, 16035),  # bar 3 — high hits TP
        ]
        for i in range(74):
            prices.append((16000, 16005, 15995, 16002))

        df = _make_bars(start, len(prices), prices=prices)
        oos = _make_oos_predictions([start])

        # Explicit direction='long' should behave identically to default
        config_default = BacktestConfig(
            threshold=0.58, account_size=50000,
            sl_atr_mult=1.0, tp_atr_mult=1.5,
            max_holding_bars=999,
        )
        config_explicit = BacktestConfig(
            threshold=0.58, account_size=50000,
            direction='long',
            sl_atr_mult=1.0, tp_atr_mult=1.5,
            max_holding_bars=999,
        )

        result_default = run_bar_by_bar_backtest(df, oos, config_default, verbose=False)
        result_explicit = run_bar_by_bar_backtest(df, oos, config_explicit, verbose=False)

        trades_d = result_default.get('trades', pd.DataFrame())
        trades_e = result_explicit.get('trades', pd.DataFrame())

        assert len(trades_d) == len(trades_e), "Trade count should match"
        if len(trades_d) > 0:
            assert trades_d.iloc[0]['entry_price'] == trades_e.iloc[0]['entry_price']
            assert trades_d.iloc[0]['exit_type'] == trades_e.iloc[0]['exit_type']
            assert trades_d.iloc[0]['pnl_dollars'] == trades_e.iloc[0]['pnl_dollars']
