"""
Tests for Phase 3 — FOMC Calendar Filter.
Verifies FOMC date recognition and trade blocking.
"""
import sys
import os
import datetime
import pandas as pd
import pytest
from datetime import timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research_utils'))


class TestFOMCDatesRecognized:
    """is_high_impact_day() should return True for all FOMC dates."""

    def test_fomc_dates_recognized(self):
        from research.config import is_high_impact_day, FOMC_DATES

        for d in FOMC_DATES:
            assert is_high_impact_day(d), f"FOMC date {d} not recognized"

    def test_fomc_timestamp_input(self):
        """Should handle pd.Timestamp input (not just date)."""
        from research.config import is_high_impact_day

        ts = pd.Timestamp('2024-01-31 10:00', tz='America/New_York')
        assert is_high_impact_day(ts), "Should recognize FOMC from Timestamp"


class TestNormalDayPasses:
    """Non-FOMC dates should return False."""

    def test_normal_day_passes(self):
        from research.config import is_high_impact_day

        normal_day = datetime.date(2024, 2, 15)  # not an FOMC date
        assert not is_high_impact_day(normal_day), "Normal day should not be flagged"

    def test_day_before_fomc(self):
        from research.config import is_high_impact_day

        day_before = datetime.date(2024, 1, 30)  # day before Jan 31 FOMC
        assert not is_high_impact_day(day_before), "Day before FOMC should pass"


class TestSimulateZeroFOMCTrades:
    """simulate_strategy should produce 0 trades on FOMC-only event set."""

    def test_simulate_zero_fomc_trades(self):
        from research.strategy_construction import simulate_strategy, StrategyConfig
        from research.config import FOMC_DATES

        config = StrategyConfig(
            account_size=50000, threshold=0.58,
            max_daily_trades=10, consec_loss_pause=99,
        )

        tz = 'America/New_York'
        fomc_date = sorted(FOMC_DATES)[0]  # first FOMC date

        events = pd.DataFrame([
            {
                'session_date': fomc_date,
                'calibrated_prob': 0.80,
                'barrier_return_pts': 20.0,
                'barrier_label': 1,
                'sl_distance_pts': 10.0,
            },
        ], index=pd.DatetimeIndex([
            pd.Timestamp(fomc_date.isoformat() + ' 10:00', tz=tz),
        ]))

        result = simulate_strategy(events, config, prob_col='calibrated_prob')
        trades = result['trades']
        assert len(trades) == 0, (
            f"Expected 0 trades on FOMC day, got {len(trades)}"
        )
