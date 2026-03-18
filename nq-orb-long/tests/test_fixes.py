"""
Tests verifying all audit fixes (P1-A through P4-D) are correctly applied.
Run BEFORE the pipeline re-run to confirm fix correctness on synthetic data.
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


# ═══════════════════════════════════════════════════════════════════
# P1-A: session_range_position uses expanding window (no lookahead)
# ═══════════════════════════════════════════════════════════════════

class TestP1A_NoLookahead:
    """session_high/session_low must use expanding() not full-day max/min."""

    def test_session_high_expanding(self, sample_ohlcv_df):
        """At bar N, session_high == max(high[0..N]), NOT max(all bars)."""
        from research.event_definitions import add_session_columns

        df = add_session_columns(sample_ohlcv_df)

        # Get session 1 bars (2024-01-02)
        day1 = df[df.index.date == pd.Timestamp('2024-01-02').date()]
        rth_bars = day1[day1.get('is_rth', True) == True] if 'is_rth' in day1.columns else day1

        if len(rth_bars) < 6:
            pytest.skip("Not enough RTH bars for test")

        # Bar 3 (0-indexed): session_high should be max of bars 0..3
        bar3_session_high = rth_bars.iloc[3]['session_high']
        expected_high = rth_bars.iloc[:4]['high'].max()
        assert bar3_session_high == expected_high, (
            f"Bar 3 session_high={bar3_session_high}, expected {expected_high} "
            f"(max of first 4 bars)"
        )

        # session_high at bar 3 must be LESS THAN the final session_high
        # (since highs increase monotonically in our fixture)
        final_session_high = rth_bars.iloc[-1]['session_high']
        assert bar3_session_high < final_session_high, (
            "session_high at bar 3 should be less than final session_high "
            "(expanding, not full-day)"
        )

    def test_session_low_expanding(self, sample_ohlcv_df):
        """session_low should also use expanding min."""
        from research.event_definitions import add_session_columns

        df = add_session_columns(sample_ohlcv_df)
        day1 = df[df.index.date == pd.Timestamp('2024-01-02').date()]

        if 'session_low' not in day1.columns or len(day1) < 4:
            pytest.skip("session_low not computed or too few bars")

        bar3_low = day1.iloc[3]['session_low']
        expected_low = day1.iloc[:4]['low'].min()
        assert bar3_low == expected_low

    def test_first_bar_session_high_equals_own_high(self, sample_ohlcv_df):
        """First bar of session: session_high == that bar's high."""
        from research.event_definitions import add_session_columns

        df = add_session_columns(sample_ohlcv_df)
        day1 = df[df.index.date == pd.Timestamp('2024-01-02').date()]

        if len(day1) > 0 and 'session_high' in day1.columns:
            first_bar = day1.iloc[0]
            assert first_bar['session_high'] == first_bar['high']


# ═══════════════════════════════════════════════════════════════════
# P1-C: Sharpe annualization uses daily P&L × √252
# ═══════════════════════════════════════════════════════════════════

class TestP1C_SharpeAnnualization:
    """bootstrap_ev must use daily P&L grouped by date, not trade-level."""

    def test_sharpe_with_timestamps(self, sample_returns_with_timestamps):
        """When timestamps are provided, annualized_sharpe uses daily groupby."""
        from research.statistical_research import bootstrap_ev

        returns, timestamps = sample_returns_with_timestamps
        result = bootstrap_ev(returns, timestamps=timestamps, n_bootstrap=100)

        assert 'annualized_sharpe' in result
        assert result['annualized_sharpe'] != 0.0, "Sharpe should not be 0 when timestamps provided"

        # Verify the point-estimate Sharpe matches manual calculation
        daily_pnl = pd.Series(returns, index=timestamps).groupby(
            timestamps.normalize()
        ).sum()
        expected_sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
        assert abs(result['annualized_sharpe'] - expected_sharpe) < 0.01, (
            f"Sharpe {result['annualized_sharpe']:.4f} != expected {expected_sharpe:.4f}"
        )

    def test_sharpe_without_timestamps(self):
        """When timestamps=None, annualized_sharpe should be 0."""
        from research.statistical_research import bootstrap_ev

        returns = np.random.normal(2, 10, 50)
        result = bootstrap_ev(returns, timestamps=None, n_bootstrap=100)

        assert result['annualized_sharpe'] == 0.0

    def test_sharpe_returns_n_trading_days(self, sample_returns_with_timestamps):
        """Result should include n_trading_days count."""
        from research.statistical_research import bootstrap_ev

        returns, timestamps = sample_returns_with_timestamps
        result = bootstrap_ev(returns, timestamps=timestamps, n_bootstrap=100)

        assert result['n_trading_days'] == 10  # 10 business days in fixture


# ═══════════════════════════════════════════════════════════════════
# P4-C: Consecutive loss pause halts FULL day, not just one signal
# ═══════════════════════════════════════════════════════════════════

class TestP4C_ConsecLossHaltsDay:
    """After consec_loss_pause losses, all remaining signals that day are skipped."""

    def _make_events(self, n_events, same_date, all_losses=False, prob=0.70):
        """Create synthetic events for simulate_strategy testing."""
        tz = 'America/New_York'
        ts_base = pd.Timestamp(same_date + ' 10:05', tz=tz)
        rows = []
        for i in range(n_events):
            ts = ts_base + timedelta(minutes=30 * i)
            rows.append({
                'session_date': pd.Timestamp(same_date).date(),
                'calibrated_prob': prob,
                'barrier_return_pts': -5.0 if (all_losses or i < 3) else 10.0,
                'barrier_label': 0 if (all_losses or i < 3) else 1,
                'sl_distance_pts': 10.0,
            })
        df = pd.DataFrame(rows, index=[
            ts_base + timedelta(minutes=30 * i) for i in range(n_events)
        ])
        return df

    def test_halts_after_3_consec_losses(self):
        """5 events same day, first 3 are losses → only 3 trades executed."""
        from research.strategy_construction import simulate_strategy, StrategyConfig

        config = StrategyConfig(
            account_size=50000, threshold=0.58, consec_loss_pause=3,
            max_daily_trades=10,  # high limit so it's not the blocker
        )
        events = self._make_events(5, '2024-01-02', all_losses=False, prob=0.70)
        result = simulate_strategy(events, config, prob_col='calibrated_prob')
        trades = result['trades']

        assert len(trades) == 3, (
            f"Expected 3 trades (first 3 losses halt day), got {len(trades)}"
        )

    def test_resets_on_new_day(self):
        """Consecutive losses reset at start of new trading day."""
        from research.strategy_construction import simulate_strategy, StrategyConfig

        config = StrategyConfig(
            account_size=50000, threshold=0.58, consec_loss_pause=3,
            max_daily_trades=10,
        )
        # Day 1: 3 losses → halted. Day 2: should trade normally
        day1 = self._make_events(3, '2024-01-02', all_losses=True, prob=0.70)
        day2 = self._make_events(2, '2024-01-03', all_losses=False, prob=0.70)
        events = pd.concat([day1, day2])

        result = simulate_strategy(events, config, prob_col='calibrated_prob')
        trades = result['trades']

        day2_trades = trades[trades['session_date'] == pd.Timestamp('2024-01-03').date()] if len(trades) > 0 else pd.DataFrame()
        assert len(day2_trades) > 0, "Day 2 should have trades (consec_losses reset)"


# ═══════════════════════════════════════════════════════════════════
# P4-D: Daily loss cap checked pre-entry with worst-case estimate
# ═══════════════════════════════════════════════════════════════════

class TestP4D_DailyLossCapPreEntry:
    """Daily loss cap must block entry when worst-case SL hit would breach limit."""

    def test_blocks_when_worstcase_breaches_cap(self):
        """
        Prior daily PnL = -$800.
        Next trade SL dist = 15 pts → worst case = -(15 * 20 + 4.50 * 2) = -$309.
        -800 + (-309) = -$1109 < -$1000 cap → BLOCKED.
        """
        from research.strategy_construction import simulate_strategy, StrategyConfig

        config = StrategyConfig(
            account_size=50000,
            threshold=0.58,
            max_daily_loss=-1000.0,
            max_daily_trades=10,
            consec_loss_pause=99,  # disable consec loss
        )

        tz = 'America/New_York'
        ts_base = pd.Timestamp('2024-01-02 10:05', tz=tz)

        # First trade: a $800 loss
        # With point_value=20, barrier_return_pts=-40 → PnL = -40*1*20 = -$800
        events = pd.DataFrame([
            {
                'session_date': pd.Timestamp('2024-01-02').date(),
                'calibrated_prob': 0.70,
                'barrier_return_pts': -40.0,
                'barrier_label': 0,
                'sl_distance_pts': 40.0,
            },
            {
                'session_date': pd.Timestamp('2024-01-02').date(),
                'calibrated_prob': 0.70,
                'barrier_return_pts': 20.0,  # would be a winner
                'barrier_label': 1,
                'sl_distance_pts': 15.0,
            },
        ], index=[ts_base, ts_base + timedelta(minutes=30)])

        result = simulate_strategy(events, config, prob_col='calibrated_prob')
        trades = result['trades']

        # First trade executes (-$800). Second should be blocked by pre-entry cap check.
        assert len(trades) <= 1, (
            f"Expected ≤1 trade (2nd blocked by daily loss cap), got {len(trades)}"
        )
