"""
event_features.py
=================
Event-context feature engineering for NQ event-driven research framework.

This module extracts contextual features AT THE TIME of each event and
packages them alongside the outcome labels (from Step 2) into a single
DataFrame ready for model training (Step 5).

Two-stage architecture:
  Stage 1: Build bar-level features on the full 5-min OHLCV dataset
           (uses research_utils/feature_engineering.py)
  Stage 2: For each event, extract the feature vector at event time
           and add event-specific features

All features use ONLY data available at or before the event bar (no look-ahead).

Usage:
    from event_features import build_model_dataset, build_feature_matrix
    from event_definitions import detect_all_events, add_session_columns
    from research.outcome_labeling import label_event_full
    from feature_engineering import load_ohlcv, build_features

    # Load and prepare data
    df = load_ohlcv('nq_continuous_5m_converted.csv')
    df = detect_all_events(df, include_events=['sweep', 'orb'])

    # Build bar-level features (Stage 1)
    df = build_features(df, add_targets_flag=False)

    # Build complete model dataset for one event type (Stage 1 + 2)
    dataset = build_model_dataset(
        df,
        event_col='sweep_high_first_today',
        direction='short',
    )
    # dataset is a DataFrame: index=event_time, columns=features+labels
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# STAGE 2: EXTRACT FEATURES AT EVENT TIME
# ===========================================================================

# --- Universal feature columns to extract from bar-level DataFrame ---
# These are columns created by feature_engineering.build_features()

VOLATILITY_FEATURES = [
    'atr_5', 'atr_10', 'atr_20',
    'natr_20',
    'vol_ratio_5_20',
    'vol_ratio_10_60',
    'vol_zscore',
    'rvol_cc_5', 'rvol_cc_20',
    'rvol_parkinson_20',
]

TREND_FEATURES = [
    'log_ret_1', 'log_ret_5', 'log_ret_10', 'log_ret_20', 'log_ret_60',
    'mom_sign_5', 'mom_sign_20',
    'mom_accel',
    'rsi_14', 'rsi_28',
    'dist_ma_20', 'dist_ma_60',
    'bb_zscore_20',
    'pctrank_20', 'pctrank_60',
    'efficiency_ratio_1d',
    'trend_regime',
    'consec_up', 'consec_down',
]

VOLUME_FEATURES = [
    'rvol_ratio_5', 'rvol_ratio_20',
    'vwap_distance',
    'vol_price_diverge',
    'obv_slope_20',
    'vol_mom_20',
]

MICROSTRUCTURE_FEATURES = [
    'body_ratio',
    'upper_wick_ratio',
    'lower_wick_ratio',
    'close_position',
    'range_ratio_10', 'range_ratio_20',
    'gap', 'gap_pct',
]

TEMPORAL_FEATURES = [
    'bar_of_session',
    'mins_since_open',
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
]

REGIME_FEATURES = [
    'vol_regime',
    'vol_of_vol',
]

# All universal feature columns to attempt extraction
ALL_UNIVERSAL_FEATURES = (
    VOLATILITY_FEATURES +
    TREND_FEATURES +
    VOLUME_FEATURES +
    MICROSTRUCTURE_FEATURES +
    TEMPORAL_FEATURES +
    REGIME_FEATURES
)


# ORB-specific feature names (added by add_orb_features)
ORB_SPECIFIC_FEATURES = [
    'or_range_atr', 'or_range_relative', 'breakout_strength_atr',
    'gap_alignment', 'vwap_dist_at_breakout_atr',
]

# Session reference feature names (added by add_session_reference_features)
SESSION_REFERENCE_FEATURES = [
    'dist_prior_high_atr', 'dist_prior_low_atr', 'dist_session_open_atr',
    'vwap_dist_atr', 'session_range_position', 'gap_pct', 'ib_position',
]


def get_expected_feature_columns() -> list[str]:
    """
    Return the canonical ordered list of feature column names that
    the model may use. Includes universal features (from build_features),
    ORB-specific features, and session reference features.
    """
    return (
        list(ALL_UNIVERSAL_FEATURES) +
        ORB_SPECIFIC_FEATURES +
        SESSION_REFERENCE_FEATURES
    )


def extract_event_features_row(
    df_bars: pd.DataFrame,
    event_time: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Given a rolling DataFrame of recent bars (OHLCV + DatetimeIndex),
    compute all universal features for the bar at `event_time`.

    Returns a single-row DataFrame with available feature columns, or None
    if insufficient history or event_time not in index.

    Args:
        df_bars:    DataFrame with OHLCV columns and DatetimeIndex.
                    Should contain >= 50 bars for fast features.
        event_time: The timestamp of the event bar.
    """
    from research_utils.feature_engineering import build_features

    if len(df_bars) < 50:
        logger.warning(f"[FEATURES] Insufficient bar history ({len(df_bars)} bars, need >= 50)")
        return None

    if event_time not in df_bars.index:
        logger.warning(f"[FEATURES] event_time {event_time} not in bar buffer index")
        return None

    # Build bar-level features on the available history
    feature_df = build_features(df_bars.copy(), add_targets_flag=False)

    if event_time not in feature_df.index:
        logger.warning(f"[FEATURES] event_time lost after build_features")
        return None

    row = feature_df.loc[[event_time]]

    # Return all available universal features
    available = [c for c in ALL_UNIVERSAL_FEATURES if c in row.columns]
    missing = [c for c in ALL_UNIVERSAL_FEATURES if c not in row.columns]

    if missing:
        logger.debug(f"[FEATURES] {len(missing)} universal features not computed: {missing[:5]}...")

    if not available:
        logger.warning("[FEATURES] No universal features could be computed")
        return None

    return row[available]


def extract_universal_features(
    df: pd.DataFrame,
    event_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Extract universal (bar-level) features at each event time.

    Parameters
    ----------
    df : DataFrame with bar-level features already computed
         (via feature_engineering.build_features)
    event_times : DatetimeIndex of event timestamps

    Returns
    -------
    DataFrame indexed by event_time with one column per available feature
    """
    # Find which columns actually exist in the DataFrame
    available = [c for c in ALL_UNIVERSAL_FEATURES if c in df.columns]
    missing = [c for c in ALL_UNIVERSAL_FEATURES if c not in df.columns]

    if missing:
        print(f"[FEATURES] Note: {len(missing)} universal features not found "
              f"(run build_features first). Missing: {missing[:5]}...")

    if not available:
        print("[FEATURES] WARNING: No universal features found in DataFrame!")
        return pd.DataFrame(index=event_times)

    # Extract feature values at event times using .loc
    # This is efficient because we're selecting specific rows
    valid_times = event_times[event_times.isin(df.index)]
    if len(valid_times) < len(event_times):
        print(f"[FEATURES] Warning: {len(event_times) - len(valid_times)} event times "
              f"not found in DataFrame index")

    features = df.loc[valid_times, available].copy()
    features.index.name = 'event_time'

    return features


# ===========================================================================
# EVENT-SPECIFIC FEATURES
# ===========================================================================

def add_sweep_features(
    df: pd.DataFrame,
    labeled_events: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    """
    Add features specific to session sweep events.

    Parameters
    ----------
    df : Full DataFrame with session columns
    labeled_events : DataFrame from label_events (indexed by event_time)
    direction : 'long' or 'short'

    Returns
    -------
    labeled_events with additional columns
    """
    events = labeled_events.copy()

    for event_time in events.index:
        if event_time not in df.index:
            continue

        row = df.loc[event_time]
        entry_price = events.loc[event_time, 'entry_price']
        atr = events.loc[event_time, 'atr_at_entry']

        if atr <= 0 or np.isnan(atr):
            continue

        # How far beyond the prior level did price go?
        if direction == 'short':
            # Sweep high: how far above prior session high
            prior_level = row.get('prior_session_high', np.nan)
            if not np.isnan(prior_level):
                sweep_depth = row['high'] - prior_level
                events.loc[event_time, 'sweep_depth_atr'] = sweep_depth / atr
                events.loc[event_time, 'dist_prior_level_atr'] = (
                    (prior_level - row.get('session_open', entry_price)) / atr
                )
            else:
                events.loc[event_time, 'sweep_depth_atr'] = np.nan
                events.loc[event_time, 'dist_prior_level_atr'] = np.nan
        else:
            # Sweep low: how far below prior session low
            prior_level = row.get('prior_session_low', np.nan)
            if not np.isnan(prior_level):
                sweep_depth = prior_level - row['low']
                events.loc[event_time, 'sweep_depth_atr'] = sweep_depth / atr
                events.loc[event_time, 'dist_prior_level_atr'] = (
                    (row.get('session_open', entry_price) - prior_level) / atr
                )
            else:
                events.loc[event_time, 'sweep_depth_atr'] = np.nan
                events.loc[event_time, 'dist_prior_level_atr'] = np.nan

        # Bar direction relative to trade direction
        bar_body = row['close'] - row['open']
        if direction == 'long':
            events.loc[event_time, 'sweep_bar_alignment'] = 1 if bar_body > 0 else -1
        else:
            events.loc[event_time, 'sweep_bar_alignment'] = 1 if bar_body < 0 else -1

        # Distance from session open
        session_open = row.get('session_open', np.nan)
        if not np.isnan(session_open):
            events.loc[event_time, 'dist_session_open_atr'] = (
                (entry_price - session_open) / atr
            )
        else:
            events.loc[event_time, 'dist_session_open_atr'] = np.nan

    return events


def add_orb_features(
    df: pd.DataFrame,
    labeled_events: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    """
    Add features specific to Opening Range Breakout events.

    Parameters
    ----------
    df : Full DataFrame with session columns (must have or_high, or_low, or_range)
    labeled_events : DataFrame from label_events
    direction : 'long' or 'short'

    Returns
    -------
    labeled_events with additional columns
    """
    events = labeled_events.copy()

    # Compute 20-day average OR range for relative sizing
    if 'or_range' in df.columns:
        # Get unique daily OR ranges
        rth_mask = df['is_rth'].astype(bool) if 'is_rth' in df.columns else pd.Series(True, index=df.index)
        rth = df[rth_mask]
        daily_or = rth.groupby(rth.index.date)['or_range'].first().dropna()
        or_range_ma20 = daily_or.rolling(20, min_periods=5).mean()
    else:
        or_range_ma20 = None

    for event_time in events.index:
        if event_time not in df.index:
            continue

        row = df.loc[event_time]
        entry_price = events.loc[event_time, 'entry_price']
        atr = events.loc[event_time, 'atr_at_entry']

        if atr <= 0 or np.isnan(atr):
            continue

        # OR width in ATR units
        or_range = row.get('or_range', np.nan)
        if not np.isnan(or_range):
            events.loc[event_time, 'or_range_atr'] = or_range / atr
        else:
            events.loc[event_time, 'or_range_atr'] = np.nan

        # OR width relative to recent average
        event_date = event_time.date()
        if or_range_ma20 is not None and event_date in or_range_ma20.index:
            avg_or = or_range_ma20.loc[event_date]
            if avg_or > 0:
                events.loc[event_time, 'or_range_relative'] = or_range / avg_or
            else:
                events.loc[event_time, 'or_range_relative'] = np.nan
        else:
            events.loc[event_time, 'or_range_relative'] = np.nan

        # Breakout strength: how far beyond the OR boundary
        if direction == 'long':
            or_boundary = row.get('or_high', np.nan)
            if not np.isnan(or_boundary):
                events.loc[event_time, 'breakout_strength_atr'] = (
                    (entry_price - or_boundary) / atr
                )
            else:
                events.loc[event_time, 'breakout_strength_atr'] = np.nan
        else:
            or_boundary = row.get('or_low', np.nan)
            if not np.isnan(or_boundary):
                events.loc[event_time, 'breakout_strength_atr'] = (
                    (or_boundary - entry_price) / atr
                )
            else:
                events.loc[event_time, 'breakout_strength_atr'] = np.nan

        # Did the opening gap align with breakout direction?
        gap = row.get('gap_pct', 0)
        if direction == 'long':
            events.loc[event_time, 'gap_alignment'] = 1 if gap > 0 else (0 if gap == 0 else -1)
        else:
            events.loc[event_time, 'gap_alignment'] = 1 if gap < 0 else (0 if gap == 0 else -1)

        # Distance from VWAP at breakout
        vwap = row.get('vwap', np.nan)
        if not np.isnan(vwap) and atr > 0:
            events.loc[event_time, 'vwap_dist_at_breakout_atr'] = (
                (entry_price - vwap) / atr
            )
        else:
            events.loc[event_time, 'vwap_dist_at_breakout_atr'] = np.nan

    return events


def add_session_reference_features(
    df: pd.DataFrame,
    labeled_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add session-level reference features applicable to all event types.

    Parameters
    ----------
    df : Full DataFrame with session columns
    labeled_events : DataFrame from label_events

    Returns
    -------
    labeled_events with additional columns
    """
    events = labeled_events.copy()

    for event_time in events.index:
        if event_time not in df.index:
            continue

        row = df.loc[event_time]
        entry_price = events.loc[event_time, 'entry_price']
        atr = events.loc[event_time, 'atr_at_entry']

        if atr <= 0 or np.isnan(atr):
            continue

        # Distance from prior session high/low (in ATR)
        psh = row.get('prior_session_high', np.nan)
        psl = row.get('prior_session_low', np.nan)
        if not np.isnan(psh):
            events.loc[event_time, 'dist_prior_high_atr'] = (entry_price - psh) / atr
        if not np.isnan(psl):
            events.loc[event_time, 'dist_prior_low_atr'] = (entry_price - psl) / atr

        # Distance from today's session open
        session_open = row.get('session_open', np.nan)
        if not np.isnan(session_open):
            events.loc[event_time, 'dist_session_open_atr'] = (
                (entry_price - session_open) / atr
            )

        # Distance from VWAP
        vwap = row.get('vwap', np.nan)
        if not np.isnan(vwap):
            events.loc[event_time, 'vwap_dist_atr'] = (entry_price - vwap) / atr

        # Position within today's range so far
        session_high = row.get('session_high', np.nan)
        session_low = row.get('session_low', np.nan)
        if not np.isnan(session_high) and not np.isnan(session_low):
            session_range = session_high - session_low
            if session_range > 0:
                events.loc[event_time, 'session_range_position'] = (
                    (entry_price - session_low) / session_range
                )

        # Gap info
        gap_pct = row.get('gap_pct', np.nan)
        if not np.isnan(gap_pct):
            events.loc[event_time, 'gap_pct'] = gap_pct

        # Position within Initial Balance (if available)
        ib_high = row.get('ib_high', np.nan)
        ib_low = row.get('ib_low', np.nan)
        if not np.isnan(ib_high) and not np.isnan(ib_low):
            ib_range = ib_high - ib_low
            if ib_range > 0:
                events.loc[event_time, 'ib_position'] = (
                    (entry_price - ib_low) / ib_range
                )

    return events


# ===========================================================================
# MAIN PIPELINE: BUILD COMPLETE FEATURE MATRIX
# ===========================================================================

def build_feature_matrix(
    df: pd.DataFrame,
    labeled_events: pd.DataFrame,
    event_type: str = 'generic',
    direction: str = 'long',
) -> pd.DataFrame:
    """
    Build the complete feature matrix for a set of labeled events.

    Combines:
      1. Universal features extracted at event time
      2. Session reference features
      3. Event-specific features (if applicable)
      4. Outcome labels (from Step 2)

    Parameters
    ----------
    df : Full DataFrame with bar-level features (from build_features)
         and session columns (from add_session_columns)
    labeled_events : DataFrame from label_event_full (Step 2)
                     Must have: entry_price, atr_at_entry, barrier_label, etc.
    event_type : 'sweep', 'orb', 'cusum', 'volexp', 'vwap', 'gap',
                 'exhaustion', 'ib_ext', 'volspike', or 'generic'
    direction : 'long' or 'short'

    Returns
    -------
    DataFrame with all features and labels, indexed by event_time.
    Ready for model training (Step 5).
    """
    if len(labeled_events) == 0:
        print(f"[FEATURES] No events to process for {event_type}")
        return pd.DataFrame()

    event_times = labeled_events.index
    print(f"\n[FEATURES] Building feature matrix for {len(event_times)} "
          f"{event_type} events ({direction})...")

    # --- Stage 2a: Universal features ---
    print("[FEATURES]   Extracting universal features...")
    universal = extract_universal_features(df, event_times)

    # --- Stage 2b: Session reference features ---
    print("[FEATURES]   Adding session reference features...")
    events_with_ref = add_session_reference_features(df, labeled_events)

    # --- Stage 2c: Event-specific features ---
    if event_type == 'sweep':
        print("[FEATURES]   Adding sweep-specific features...")
        events_with_ref = add_sweep_features(df, events_with_ref, direction)
    elif event_type == 'orb':
        print("[FEATURES]   Adding ORB-specific features...")
        events_with_ref = add_orb_features(df, events_with_ref, direction)
    # Future: add cusum, volexp, etc. specific features

    # --- Merge everything ---
    print("[FEATURES]   Merging feature matrix...")

    # Start with labeled events (contains labels + trade metadata)
    result = events_with_ref.copy()

    # Merge universal features
    for col in universal.columns:
        if col not in result.columns:
            result[col] = universal[col]

    # Count features vs labels
    label_cols = [
        'direction', 'entry_price', 'atr_at_entry',
        'sl_level', 'tp_level', 'sl_distance_pts', 'tp_distance_pts',
        'barrier_label', 'exit_type', 'exit_price', 'exit_time',
        'barrier_return_pts', 'barrier_return_pct',
        'time_to_barrier', 'mae_pts', 'mfe_pts', 'mae_atr', 'mfe_atr',
    ]
    fwd_cols = [c for c in result.columns if c.startswith('fwd_')]
    label_cols = label_cols + fwd_cols

    feature_cols = [c for c in result.columns if c not in label_cols]

    print(f"[FEATURES]   Done: {len(feature_cols)} features, "
          f"{len([c for c in label_cols if c in result.columns])} label/metadata columns")
    print(f"[FEATURES]   Total columns: {len(result.columns)}")

    # Report NaN percentages for key features
    nan_report = result[feature_cols].isnull().mean().sort_values(ascending=False)
    high_nan = nan_report[nan_report > 0.1]
    if len(high_nan) > 0:
        print(f"[FEATURES]   Features with >10% NaN:")
        for col_name, pct in high_nan.head(10).items():
            print(f"               {col_name}: {pct:.1%} NaN")

    return result


def get_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names from a model dataset.

    Excludes label/metadata columns so only predictive features remain.
    Useful for passing to model training: X = dataset[get_feature_columns(dataset)]
    """
    label_and_meta = {
        'direction', 'entry_price', 'atr_at_entry',
        'sl_level', 'tp_level', 'sl_distance_pts', 'tp_distance_pts',
        'barrier_label', 'exit_type', 'exit_price', 'exit_time',
        'barrier_return_pts', 'barrier_return_pct',
        'time_to_barrier', 'mae_pts', 'mfe_pts', 'mae_atr', 'mfe_atr',
    }
    # Also exclude forward return columns
    fwd_cols = {c for c in dataset.columns if c.startswith('fwd_')}
    exclude = label_and_meta | fwd_cols

    return [c for c in dataset.columns if c not in exclude]


# ===========================================================================
# HIGH-LEVEL: BUILD COMPLETE MODEL DATASET
# ===========================================================================

def build_model_dataset(
    df: pd.DataFrame,
    event_col: str,
    direction: str,
    event_type: str = 'generic',
    sl_atr_multiple: float = 1.0,
    tp_atr_multiple: float = 1.5,
    atr_lookback: int = 14,
    max_holding_bars: int = 48,
    force_session_exit: bool = True,
    fwd_horizons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    One-call pipeline: event detection → labeling → feature extraction.

    This is the top-level function for producing a complete model dataset.
    Assumes df already has:
      - Session columns (from add_session_columns)
      - Event columns (from detect_* functions)
      - Bar-level features (from build_features)

    Parameters
    ----------
    df : Fully prepared DataFrame (OHLCV + session + events + features)
    event_col : Boolean event column name
    direction : 'long' or 'short'
    event_type : Event type for specific feature extraction
    sl_atr_multiple, tp_atr_multiple, etc. : Barrier parameters

    Returns
    -------
    DataFrame ready for model training: features + labels
    """
    # Import here to avoid circular dependency
    from research.outcome_labeling import label_event_full

    # Step 2: Label events
    labeled = label_event_full(
        df, event_col, direction,
        sl_atr_multiple=sl_atr_multiple,
        tp_atr_multiple=tp_atr_multiple,
        atr_lookback=atr_lookback,
        max_holding_bars=max_holding_bars,
        force_session_exit=force_session_exit,
        fwd_horizons=fwd_horizons,
    )

    if len(labeled) == 0:
        return pd.DataFrame()

    # Step 3: Build feature matrix
    dataset = build_feature_matrix(df, labeled, event_type=event_type, direction=direction)

    return dataset


# ===========================================================================
# FEATURE DIAGNOSTICS
# ===========================================================================

def feature_importance_preview(
    dataset: pd.DataFrame,
    label_col: str = 'barrier_label',
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Quick feature importance analysis using correlation and mean differences.

    NOT a substitute for proper model-based importance (Step 5), but useful
    for a quick sanity check that features have some signal.

    Parameters
    ----------
    dataset : Output from build_feature_matrix or build_model_dataset
    label_col : Column to correlate features against
    top_n : Number of top features to display

    Returns
    -------
    DataFrame with importance metrics per feature, sorted by absolute correlation
    """
    feature_cols = get_feature_columns(dataset)
    if not feature_cols:
        print("[IMPORTANCE] No feature columns found!")
        return pd.DataFrame()

    # Only use events with definitive labels (+1 or -1)
    definitive = dataset[dataset[label_col].isin([1, -1])].copy()
    if len(definitive) < 20:
        print(f"[IMPORTANCE] Only {len(definitive)} definitive labels — too few for analysis")
        return pd.DataFrame()

    results = []

    for col in feature_cols:
        vals = definitive[col].dropna()
        labels = definitive.loc[vals.index, label_col]

        if len(vals) < 10:
            continue

        # Pearson correlation with label
        corr = vals.corr(labels)

        # Mean for winners vs losers
        winners = vals[labels == 1]
        losers = vals[labels == -1]

        mean_win = winners.mean() if len(winners) > 0 else np.nan
        mean_loss = losers.mean() if len(losers) > 0 else np.nan
        mean_diff = mean_win - mean_loss if not np.isnan(mean_win) and not np.isnan(mean_loss) else np.nan

        # Effect size (Cohen's d)
        pooled_std = vals.std()
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        results.append({
            'feature': col,
            'correlation': round(corr, 4) if not np.isnan(corr) else 0,
            'abs_correlation': abs(round(corr, 4)) if not np.isnan(corr) else 0,
            'mean_winners': round(mean_win, 4) if not np.isnan(mean_win) else np.nan,
            'mean_losers': round(mean_loss, 4) if not np.isnan(mean_loss) else np.nan,
            'mean_diff': round(mean_diff, 4) if not np.isnan(mean_diff) else np.nan,
            'cohens_d': round(cohens_d, 4),
            'n_valid': len(vals),
            'pct_nan': round(1 - len(vals) / len(definitive), 3),
        })

    importance_df = pd.DataFrame(results).sort_values('abs_correlation', ascending=False)

    # Display
    print(f"\n{'=' * 80}")
    print(f"  FEATURE IMPORTANCE PREVIEW (corr with {label_col})")
    print(f"  {len(definitive)} events with definitive labels (+1/-1)")
    print(f"{'=' * 80}")
    print(f"  {'Feature':<30} {'Corr':>8} {'Cohen d':>8} {'Win Mean':>10} {'Loss Mean':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for _, row in importance_df.head(top_n).iterrows():
        print(f"  {row['feature']:<30} {row['correlation']:>+8.4f} "
              f"{row['cohens_d']:>+8.4f} "
              f"{row['mean_winners']:>10.4f} {row['mean_losers']:>10.4f}")

    # Summarize
    strong_features = importance_df[importance_df['abs_correlation'] > 0.1]
    print(f"\n  Features with |corr| > 0.10: {len(strong_features)}")
    moderate_features = importance_df[importance_df['abs_correlation'] > 0.05]
    print(f"  Features with |corr| > 0.05: {len(moderate_features)}")

    return importance_df


# ===========================================================================
# VALIDATION SCRIPT
# ===========================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../research_utils')
    sys.path.insert(0, '.')

    from feature_engineering import load_ohlcv, build_features
    from event_definitions import add_session_columns, detect_session_sweep, detect_orb
    from research.outcome_labeling import label_event_full, print_label_summary

    print("=" * 70)
    print("EVENT FEATURES — Validation Run")
    print("=" * 70)

    # --- Load data ---
    filepath = '../nq_continuous_5m_converted.csv'
    print(f"\n[LOAD] Loading {filepath}...")
    df = load_ohlcv(filepath)
    df = df['2024-01-01':'2026-01-14']
    print(f"[LOAD] {len(df)} bars loaded ({df.index[0]} to {df.index[-1]})")

    # --- Stage 1: Bar-level features ---
    print("\n[STAGE 1] Building bar-level features...")
    df = build_features(df, add_targets_flag=False)
    print(f"[STAGE 1] {len(df.columns)} columns in DataFrame")

    # --- Detect events ---
    print("\n[EVENTS] Detecting Tier 1 events...")
    df = add_session_columns(df)
    df = detect_session_sweep(df)
    df = detect_orb(df)

    # --- Build model datasets for each Tier 1 event ---
    event_configs = [
        ('sweep_high_first_today', 'short', 'sweep', 'Sweep High → SHORT'),
        ('sweep_low_first_today',  'long',  'sweep', 'Sweep Low → LONG'),
        ('event_orb_long',         'long',  'orb',   'ORB → LONG'),
        ('event_orb_short',        'short', 'orb',   'ORB → SHORT'),
    ]

    all_datasets = {}

    for event_col, direction, event_type, display_name in event_configs:
        if event_col not in df.columns:
            print(f"\n[SKIP] {event_col} not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"  Processing: {display_name}")
        print(f"{'=' * 70}")

        # Full pipeline: label → extract features
        dataset = build_model_dataset(
            df,
            event_col=event_col,
            direction=direction,
            event_type=event_type,
        )

        if len(dataset) > 0:
            all_datasets[event_col] = dataset

            # Print label summary
            print_label_summary(dataset, display_name)

            # Feature importance preview
            feature_importance_preview(dataset, top_n=10)

            # Dataset shape
            feature_cols = get_feature_columns(dataset)
            print(f"\n  Dataset shape: {dataset.shape}")
            print(f"  Feature columns: {len(feature_cols)}")
            print(f"  Label columns: {len(dataset.columns) - len(feature_cols)}")

            # Sample feature vector
            print(f"\n  Sample feature vector (first event):")
            first_event = dataset.iloc[0]
            for col in feature_cols[:10]:
                val = first_event[col]
                print(f"    {col:<30} = {val}")
            if len(feature_cols) > 10:
                print(f"    ... ({len(feature_cols) - 10} more features)")

    print(f"\n{'=' * 70}")
    print(f"  VALIDATION COMPLETE")
    print(f"  Processed {len(all_datasets)} event types")
    for name, ds in all_datasets.items():
        fc = get_feature_columns(ds)
        print(f"    {name}: {len(ds)} events, {len(fc)} features")
    print(f"{'=' * 70}")
