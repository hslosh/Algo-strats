"""
Tests for Phase B — Gap Continuation Long/Short signal pipeline.
TDD: Tests written BEFORE pipeline integration.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research_utils'))


class TestGapContMutuallyExclusive:
    """gap_cont_long and gap_cont_short should never fire on the same bar."""

    def test_gap_cont_mutually_exclusive(self):
        from research_utils.feature_engineering import load_ohlcv, build_features
        from research.event_definitions import detect_all_events, add_session_columns

        DATA_PATH = os.path.join(PROJECT_ROOT, 'nq_continuous_5m_converted.csv')
        if not os.path.exists(DATA_PATH):
            pytest.skip("Data file not available")

        df = load_ohlcv(DATA_PATH)
        df = df[df.index >= '2020-01-01'].copy()
        df = build_features(df, add_targets_flag=False)
        df = add_session_columns(df)
        df = detect_all_events(df)

        both = df[(df['event_gap_cont_long'] == True) &
                  (df['event_gap_cont_short'] == True)]
        assert len(both) == 0, (
            f"Found {len(both)} bars with both gap_cont_long AND gap_cont_short"
        )


class TestGapContFiresBar1:
    """gap_cont should fire at most once per session (on bar 1)."""

    def test_gap_cont_fires_bar_1_only(self):
        from research_utils.feature_engineering import load_ohlcv, build_features
        from research.event_definitions import detect_all_events, add_session_columns

        DATA_PATH = os.path.join(PROJECT_ROOT, 'nq_continuous_5m_converted.csv')
        if not os.path.exists(DATA_PATH):
            pytest.skip("Data file not available")

        df = load_ohlcv(DATA_PATH)
        df = df[df.index >= '2020-01-01'].copy()
        df = build_features(df, add_targets_flag=False)
        df = add_session_columns(df)
        df = detect_all_events(df)

        # Check that all gap_cont events fire on bar_of_session == 1
        gap_long = df[df['event_gap_cont_long'] == True]
        gap_short = df[df['event_gap_cont_short'] == True]

        if len(gap_long) > 0:
            assert (gap_long['bar_of_session'] == 1).all(), (
                f"gap_cont_long fires on bars other than bar 1: "
                f"{gap_long['bar_of_session'].unique()}"
            )

        if len(gap_short) > 0:
            assert (gap_short['bar_of_session'] == 1).all(), (
                f"gap_cont_short fires on bars other than bar 1: "
                f"{gap_short['bar_of_session'].unique()}"
            )

        # At most one per session per direction
        daily_long = gap_long.groupby(gap_long.index.date).size()
        assert (daily_long <= 1).all(), "gap_cont_long fires >1x per session"

        daily_short = gap_short.groupby(gap_short.index.date).size()
        assert (daily_short <= 1).all(), "gap_cont_short fires >1x per session"


class TestGapContEventCounts:
    """Verify gap_cont event counts are reasonable for WFO."""

    def test_gap_cont_sufficient_events(self):
        from research_utils.feature_engineering import load_ohlcv, build_features
        from research.event_definitions import detect_all_events, add_session_columns

        DATA_PATH = os.path.join(PROJECT_ROOT, 'nq_continuous_5m_converted.csv')
        if not os.path.exists(DATA_PATH):
            pytest.skip("Data file not available")

        df = load_ohlcv(DATA_PATH)
        df = df[df.index >= '2019-01-01'].copy()
        df = build_features(df, add_targets_flag=False)
        df = add_session_columns(df)
        df = detect_all_events(df)

        n_long = df['event_gap_cont_long'].sum()
        n_short = df['event_gap_cont_short'].sum()

        # Need >=200 for WFO minimum train events
        assert n_long >= 200, f"gap_cont_long only has {n_long} events (need 200+)"
        assert n_short >= 200, f"gap_cont_short only has {n_short} events (need 200+)"


class TestGapContTemporallyDistinctFromORB:
    """Gap cont fires bar 1, ORB fires bar 3+ — should be temporally distinct."""

    def test_gap_cont_before_orb(self):
        from research_utils.feature_engineering import load_ohlcv, build_features
        from research.event_definitions import detect_all_events, add_session_columns

        DATA_PATH = os.path.join(PROJECT_ROOT, 'nq_continuous_5m_converted.csv')
        if not os.path.exists(DATA_PATH):
            pytest.skip("Data file not available")

        df = load_ohlcv(DATA_PATH)
        df = df[df.index >= '2020-01-01'].copy()
        df = build_features(df, add_targets_flag=False)
        df = add_session_columns(df)
        df = detect_all_events(df)

        gap_bars = df[df['event_gap_cont_long'] | df['event_gap_cont_short']]
        orb_bars = df[df['event_orb_long'] | df['event_orb_short']]

        if len(gap_bars) > 0:
            gap_bar_nums = gap_bars['bar_of_session'].values
            assert gap_bar_nums.max() <= 2, (
                f"Gap cont should fire on bar 0-1, max was {gap_bar_nums.max()}"
            )

        if len(orb_bars) > 0:
            orb_bar_nums = orb_bars['bar_of_session'].values
            # ORB needs 30 min OR period = 6 bars, so should be bar 6+
            assert orb_bar_nums.min() >= 3, (
                f"ORB should fire on bar 3+, min was {orb_bar_nums.min()}"
            )
