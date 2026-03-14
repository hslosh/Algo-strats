"""Shared utility functions for the NQ ORB pipeline."""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Compute True Range for each bar.

    TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.

    Returns:
        pd.Series of TR values (NaN for the first bar).
    """
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def save_pipeline_results(results: dict, stage: str) -> str:
    """
    Save computed results to outputs/ directory.

    Args:
        results: Dict of computed metrics (NOT hardcoded strings)
        stage:   Pipeline stage name (e.g., 'step5_model', 'step6_strategy',
                 'step7_backtest', 'step8_deploy')

    Returns:
        Path to the saved JSON file
    """
    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'outputs/{stage}_{timestamp}.json'

    # Convert any non-serializable types
    def _default(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return str(obj)

    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=_default)

    print(f"[SAVE] Results written to {path}")
    return path
