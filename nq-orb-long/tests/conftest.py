"""
Shared fixtures for NQ ORB strategy test suite.
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research_utils'))


@pytest.fixture
def sample_ohlcv_df():
    """
    100 bars of synthetic 5-min NQ data across 2 sessions.
    Session 1: 2024-01-02 09:30-16:00 ET (78 bars)
    Session 2: 2024-01-03 09:30-10:30 ET (partial, 12 bars)
    Prices trend upward within each session.
    """
    tz = 'America/New_York'
    bars = []
    base_price = 16000.0

    # Session 1: 78 RTH bars (09:30 to 15:55 inclusive, 5-min intervals)
    session1_start = pd.Timestamp('2024-01-02 09:30', tz=tz)
    for i in range(78):
        ts = session1_start + timedelta(minutes=5 * i)
        o = base_price + i * 2
        h = o + 5 + i * 0.5  # highs increase to test expanding
        l = o - 3
        c = o + 2
        v = 1000 + i * 10
        bars.append({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v,
                      'timestamp': ts})

    # Session 2: 12 bars (partial day)
    base_price2 = 16200.0
    session2_start = pd.Timestamp('2024-01-03 09:30', tz=tz)
    for i in range(12):
        ts = session2_start + timedelta(minutes=5 * i)
        o = base_price2 + i * 2
        h = o + 4 + i * 0.3
        l = o - 2
        c = o + 1
        v = 800 + i * 5
        bars.append({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v,
                      'timestamp': ts})

    df = pd.DataFrame(bars)
    df.index = pd.DatetimeIndex(df['timestamp'])
    df = df.drop(columns=['timestamp'])
    return df


@pytest.fixture
def sample_returns_with_timestamps():
    """20 trades across 10 trading days with known P&L for Sharpe verification."""
    np.random.seed(42)
    dates = pd.bdate_range('2024-01-02', periods=10, tz='America/New_York')
    timestamps = []
    returns = []
    for d in dates:
        # 2 trades per day
        timestamps.append(d + timedelta(hours=10))
        timestamps.append(d + timedelta(hours=14))
        returns.extend([np.random.normal(2, 10), np.random.normal(2, 10)])
    return np.array(returns), pd.DatetimeIndex(timestamps)


@pytest.fixture
def backtest_config():
    """Standard BacktestConfig for tests."""
    from research.backtest_validation import BacktestConfig
    return BacktestConfig()


@pytest.fixture
def strategy_config():
    """Standard StrategyConfig for tests."""
    from research.strategy_construction import StrategyConfig
    return StrategyConfig()
