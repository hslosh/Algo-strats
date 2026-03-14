"""
event_definitions.py
====================
Event detection for NQ futures event-driven research framework.

Each event detector is a pure function: DataFrame in → DataFrame with event columns out.
All detectors use ONLY past data (no look-ahead). Events are boolean columns that
can be filtered, counted, and used as row selectors for outcome labeling.

Designed to work with the existing feature_engineering.py pipeline and
FirstRateData 5-min OHLCV format (ET timestamps, no header).

Usage:
    from event_definitions import detect_all_events, detect_session_sweep
    from feature_engineering import load_ohlcv

    df = load_ohlcv('nq_continuous_5m_converted.csv')
    df = detect_all_events(df)
    sweep_events = df[df['event_sweep_high'] | df['event_sweep_low']]
"""

import numpy as np
import pandas as pd
from typing import Optional

# P5-F: Numba acceleration for CUSUM detector
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ===========================================================================
# SESSION UTILITIES (shared across events)
# ===========================================================================

def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RTH session context columns needed by multiple event detectors.
    Assumes DatetimeIndex in US Eastern Time.

    Adds:
        session_date     : date of the RTH session this bar belongs to
        is_rth           : bool, True if bar is within 09:30-16:00 ET
        bar_of_session   : int, sequential bar number within RTH (0-indexed)
        prior_session_high : prior RTH session's high (available at today's open)
        prior_session_low  : prior RTH session's low
        prior_session_close: last RTH bar's close from prior session
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex (use load_ohlcv)")

    hour = df.index.hour
    minute = df.index.minute
    minutes_of_day = hour * 60 + minute

    # RTH: 09:30 (570 min) to 15:55 (955 min) inclusive for 5-min bars
    # Last bar of RTH session starts at 15:55, runs 15:55-15:59
    df['is_rth'] = (minutes_of_day >= 570) & (minutes_of_day < 960)

    # Session date: for RTH bars, use calendar date
    # For overnight bars, they "belong" to the next RTH session (context only)
    df['session_date'] = df.index.date
    # Bars after 16:00 belong to next session's overnight context
    df.loc[minutes_of_day >= 960, 'session_date'] = (
        pd.to_datetime(df.loc[minutes_of_day >= 960].index.date) + pd.Timedelta(days=1)
    ).date

    # RTH-only subset for computing session stats
    rth = df[df['is_rth']].copy()

    # Prior session high, low, close
    session_stats = rth.groupby(rth.index.date).agg(
        session_high=('high', 'max'),
        session_low=('low', 'min'),
        session_close=('close', 'last'),
        session_open=('open', 'first'),
    )
    session_stats.index = pd.to_datetime(session_stats.index)

    # Shift forward by 1 session day so today's bars see YESTERDAY's stats
    session_stats['prior_session_high'] = session_stats['session_high'].shift(1)
    session_stats['prior_session_low'] = session_stats['session_low'].shift(1)
    session_stats['prior_session_close'] = session_stats['session_close'].shift(1)

    # Merge back: map each bar's calendar date to the session stats
    date_index = pd.to_datetime(df.index.date)
    for col in ['prior_session_high', 'prior_session_low', 'prior_session_close',
                'session_high', 'session_low', 'session_open']:
        mapping = session_stats[col] if col in session_stats.columns else None
        if mapping is not None:
            df[col] = date_index.map(mapping).values

    # P1-A FIX: Replace full-day session_high / session_low with expanding
    # (causal) versions so that each bar only sees high/low up to and including
    # itself.  The full-day values were only needed to compute prior_session_*
    # (via shift), which are already stored above.
    rth_mask = df['is_rth']
    df.loc[rth_mask, 'session_high'] = (
        df.loc[rth_mask]
        .groupby(df.loc[rth_mask].index.date)['high']
        .transform(lambda x: x.expanding().max())
    )
    df.loc[rth_mask, 'session_low'] = (
        df.loc[rth_mask]
        .groupby(df.loc[rth_mask].index.date)['low']
        .transform(lambda x: x.expanding().min())
    )

    # Bar number within RTH session (0-indexed)
    df['bar_of_session'] = 0
    if len(rth) > 0:
        rth_bar_num = rth.groupby(rth.index.date).cumcount()
        df.loc[rth.index, 'bar_of_session'] = rth_bar_num.values

    return df


from research_utils.utils import true_range as _true_range


# ===========================================================================
# EVENT 1: PRIOR SESSION HIGH/LOW SWEEP & REVERSAL
# ===========================================================================

def detect_session_sweep(
    df: pd.DataFrame,
    sweep_threshold: float = 3.0,
    require_close_reversal: bool = True,
    min_bars_after_open: int = 6,
    max_bars_before_close: int = 12,
) -> pd.DataFrame:
    """
    Detect sweep of prior session high or low with reversal.

    A sweep occurs when price exceeds the prior session extreme then closes
    back on the original side of the level (stop hunt / liquidity grab).

    Parameters
    ----------
    df : DataFrame with session columns (call add_session_columns first)
    sweep_threshold : minimum points beyond the prior level to qualify
    require_close_reversal : if True, bar must close back inside the level
    min_bars_after_open : ignore events in the first N RTH bars (opening noise)
    max_bars_before_close : ignore events in the last N RTH bars (need time)

    Returns
    -------
    DataFrame with added columns:
        event_sweep_high : bool, True on bars that swept the prior session high
        event_sweep_low  : bool, True on bars that swept the prior session low
        sweep_high_first_today : bool, True only on the FIRST sweep high of the day
        sweep_low_first_today  : bool, True only on the FIRST sweep low of the day
    """
    df = df.copy()

    psh = df['prior_session_high']
    psl = df['prior_session_low']

    # --- Sweep High (bearish signal) ---
    exceeded_high = df['high'] > (psh + sweep_threshold)
    if require_close_reversal:
        closed_below = df['close'] < psh
        df['event_sweep_high'] = exceeded_high & closed_below & df['is_rth']
    else:
        df['event_sweep_high'] = exceeded_high & df['is_rth']

    # --- Sweep Low (bullish signal) ---
    exceeded_low = df['low'] < (psl - sweep_threshold)
    if require_close_reversal:
        closed_above = df['close'] > psl
        df['event_sweep_low'] = exceeded_low & closed_above & df['is_rth']
    else:
        df['event_sweep_low'] = exceeded_low & df['is_rth']

    # --- Time filters ---
    df.loc[df['bar_of_session'] < min_bars_after_open, 'event_sweep_high'] = False
    df.loc[df['bar_of_session'] < min_bars_after_open, 'event_sweep_low'] = False

    # 78 bars in full RTH session (09:30-16:00 at 5-min)
    max_bar = 78 - max_bars_before_close
    df.loc[df['bar_of_session'] > max_bar, 'event_sweep_high'] = False
    df.loc[df['bar_of_session'] > max_bar, 'event_sweep_low'] = False

    # --- First occurrence per day ---
    df['sweep_high_first_today'] = False
    df['sweep_low_first_today'] = False

    sweep_high_dates = df[df['event_sweep_high']].groupby(
        df[df['event_sweep_high']].index.date
    ).head(1).index
    df.loc[sweep_high_dates, 'sweep_high_first_today'] = True

    sweep_low_dates = df[df['event_sweep_low']].groupby(
        df[df['event_sweep_low']].index.date
    ).head(1).index
    df.loc[sweep_low_dates, 'sweep_low_first_today'] = True

    return df


# ===========================================================================
# EVENT 2: OPENING RANGE BREAKOUT (ORB)
# ===========================================================================

def detect_orb(
    df: pd.DataFrame,
    or_period_bars: int = 3,
    min_breakout_margin: float = 2.0,
    max_entry_bar: int = 30,
    min_or_range: float = 8.0,
    max_or_range: float = 50.0,
) -> pd.DataFrame:
    """
    Detect Opening Range Breakout events.

    The Opening Range (OR) is defined by the first `or_period_bars` of RTH.
    A breakout triggers when price closes beyond the OR boundary.

    Parameters
    ----------
    df : DataFrame with session columns
    or_period_bars : number of 5-min bars defining the opening range
                     3 = 15 min, 6 = 30 min, 12 = 60 min
    min_breakout_margin : minimum points beyond OR to qualify as breakout
    max_entry_bar : latest bar_of_session to trigger an ORB (time cutoff)
    min_or_range : minimum OR width in points (filter too-narrow ranges)
    max_or_range : maximum OR width in points (filter too-wide ranges)

    Returns
    -------
    DataFrame with added columns:
        or_high, or_low, or_range : Opening Range boundaries for each day
        event_orb_long  : bool, first close above OR high
        event_orb_short : bool, first close below OR low
    """
    df = df.copy()
    rth = df[df['is_rth']].copy()

    # Compute OR high/low per day
    or_bars = rth[rth['bar_of_session'] < or_period_bars]
    or_stats = or_bars.groupby(or_bars.index.date).agg(
        or_high=('high', 'max'),
        or_low=('low', 'min'),
    )
    or_stats['or_range'] = or_stats['or_high'] - or_stats['or_low']
    or_stats.index = pd.to_datetime(or_stats.index)

    # Map OR stats to all bars
    date_index = pd.to_datetime(df.index.date)
    for col in ['or_high', 'or_low', 'or_range']:
        df[col] = date_index.map(or_stats[col]).values

    # --- ORB Long: close above OR high ---
    above_or = df['close'] > (df['or_high'] + min_breakout_margin)
    after_or_period = df['bar_of_session'] >= or_period_bars
    before_cutoff = df['bar_of_session'] <= max_entry_bar
    range_ok = (df['or_range'] >= min_or_range) & (df['or_range'] <= max_or_range)

    df['_orb_long_raw'] = above_or & after_or_period & before_cutoff & range_ok & df['is_rth']

    # First occurrence per day only
    df['event_orb_long'] = False
    orb_long_first = df[df['_orb_long_raw']].groupby(
        df[df['_orb_long_raw']].index.date
    ).head(1).index
    df.loc[orb_long_first, 'event_orb_long'] = True

    # --- ORB Short: close below OR low ---
    below_or = df['close'] < (df['or_low'] - min_breakout_margin)
    df['_orb_short_raw'] = below_or & after_or_period & before_cutoff & range_ok & df['is_rth']

    df['event_orb_short'] = False
    orb_short_first = df[df['_orb_short_raw']].groupby(
        df[df['_orb_short_raw']].index.date
    ).head(1).index
    df.loc[orb_short_first, 'event_orb_short'] = True

    # Cleanup temp columns
    df = df.drop(columns=['_orb_long_raw', '_orb_short_raw'])

    return df


# ===========================================================================
# EVENT 3: CUSUM DIRECTIONAL THRESHOLD
# ===========================================================================

def _cusum_session_loop_py(log_ret_arr, close_arr, rth_mask_arr, date_ids, n):
    """Pure Python session-anchored CUSUM loop (fallback)."""
    cusum_pos = np.zeros(n, dtype=np.float64)
    cusum_neg = np.zeros(n, dtype=np.float64)
    running_pos = 0.0
    running_neg = 0.0
    prev_date = -1

    for i in range(n):
        if not rth_mask_arr[i]:
            cusum_pos[i] = 0.0
            cusum_neg[i] = 0.0
            continue

        current_date = date_ids[i]
        if current_date != prev_date:
            running_pos = 0.0
            running_neg = 0.0
            prev_date = current_date

        r = log_ret_arr[i]
        if np.isnan(r):
            r = 0.0
        r_points = r * close_arr[i]

        running_pos = max(0.0, running_pos + r_points)
        running_neg = min(0.0, running_neg + r_points)

        cusum_pos[i] = running_pos
        cusum_neg[i] = running_neg

    return cusum_pos, cusum_neg


def _cusum_rolling_loop_py(r_points_arr, n):
    """Pure Python rolling CUSUM loop (fallback)."""
    cusum_pos = np.zeros(n, dtype=np.float64)
    cusum_neg = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        cusum_pos[i] = max(0.0, cusum_pos[i-1] + r_points_arr[i])
        cusum_neg[i] = min(0.0, cusum_neg[i-1] + r_points_arr[i])
    return cusum_pos, cusum_neg


# P5-F: Numba JIT versions (if available)
if HAS_NUMBA:
    _cusum_session_loop = njit(_cusum_session_loop_py)
    _cusum_rolling_loop = njit(_cusum_rolling_loop_py)
else:
    _cusum_session_loop = _cusum_session_loop_py
    _cusum_rolling_loop = _cusum_rolling_loop_py


def detect_cusum(
    df: pd.DataFrame,
    threshold_atr_multiple: float = 1.5,
    atr_lookback: int = 14,
    anchor: str = 'session',
    max_triggers_per_day: int = 1,
    min_bar_of_session: int = 3,
    max_bar_of_session: int = 66,
) -> pd.DataFrame:
    """
    Detect CUSUM directional threshold events.

    The CUSUM accumulates signed log returns and fires when cumulative
    directional movement exceeds a threshold (expressed in ATR units).

    Parameters
    ----------
    df : DataFrame with session columns
    threshold_atr_multiple : threshold as multiple of ATR
    atr_lookback : ATR window for normalizing the threshold
    anchor : 'session' (reset at RTH open) or 'rolling' (never reset)
    max_triggers_per_day : maximum events per day per direction
    min_bar_of_session : earliest bar to trigger
    max_bar_of_session : latest bar to trigger

    Returns
    -------
    DataFrame with added columns:
        cusum_pos, cusum_neg : running CUSUM accumulators
        event_cusum_long  : bool, positive CUSUM threshold breach
        event_cusum_short : bool, negative CUSUM threshold breach
    """
    df = df.copy()

    # Log returns
    log_ret = np.log(df['close'] / df['close'].shift(1))

    # ATR for dynamic threshold
    tr = _true_range(df)
    atr = tr.rolling(atr_lookback).mean()
    threshold = threshold_atr_multiple * atr

    if anchor == 'session':
        # Session-anchored: reset CUSUM at each RTH open
        # P5-F: Use Numba-accelerated loop (or pure Python fallback)
        rth_mask_arr = df['is_rth'].values.astype(np.bool_)
        # Encode dates as integer IDs for Numba compatibility
        dates = df.index.date
        unique_dates = {d: i for i, d in enumerate(sorted(set(dates)))}
        date_ids = np.array([unique_dates[d] for d in dates], dtype=np.int64)

        cusum_pos_arr, cusum_neg_arr = _cusum_session_loop(
            log_ret.values.astype(np.float64),
            df['close'].values.astype(np.float64),
            rth_mask_arr, date_ids, len(df),
        )
        df['cusum_pos'] = cusum_pos_arr
        df['cusum_neg'] = cusum_neg_arr

    else:
        # Rolling: never reset (standard CUSUM)
        # P5-F: Use Numba-accelerated loop (or pure Python fallback)
        r_points = (log_ret * df['close']).fillna(0)
        cusum_pos_arr, cusum_neg_arr = _cusum_rolling_loop(
            r_points.values.astype(np.float64), len(df),
        )
        df['cusum_pos'] = cusum_pos_arr
        df['cusum_neg'] = cusum_neg_arr

    # --- Event detection ---
    time_ok = (
        (df['bar_of_session'] >= min_bar_of_session) &
        (df['bar_of_session'] <= max_bar_of_session) &
        df['is_rth']
    )

    # CUSUM crosses threshold
    crossed_pos = (df['cusum_pos'] > threshold) & time_ok
    crossed_neg = (df['cusum_neg'].abs() > threshold) & time_ok

    # First N per day
    df['event_cusum_long'] = False
    df['event_cusum_short'] = False

    if crossed_pos.any():
        first_pos = df[crossed_pos].groupby(df[crossed_pos].index.date).head(
            max_triggers_per_day
        ).index
        df.loc[first_pos, 'event_cusum_long'] = True

    if crossed_neg.any():
        first_neg = df[crossed_neg].groupby(df[crossed_neg].index.date).head(
            max_triggers_per_day
        ).index
        df.loc[first_neg, 'event_cusum_short'] = True

    return df


# ===========================================================================
# EVENT 4: VOLATILITY COMPRESSION → EXPANSION
# ===========================================================================

def detect_vol_compression_expansion(
    df: pd.DataFrame,
    fast_atr_window: int = 5,
    slow_atr_window: int = 20,
    compression_threshold: float = 0.70,
    min_compression_bars: int = 3,
    expansion_multiple: float = 1.3,
) -> pd.DataFrame:
    """
    Detect volatility compression followed by expansion breakout.

    Parameters
    ----------
    df : DataFrame with OHLCV
    fast_atr_window : short ATR window
    slow_atr_window : long ATR window
    compression_threshold : ratio below which compression is detected
    min_compression_bars : minimum bars of sustained compression
    expansion_multiple : current bar range must exceed this * slow ATR
    """
    df = df.copy()

    tr = _true_range(df)
    atr_fast = tr.rolling(fast_atr_window).mean()
    atr_slow = tr.rolling(slow_atr_window).mean()

    # Volatility ratio
    df['vol_ratio'] = atr_fast / atr_slow.replace(0, np.nan)

    # Compression state: vol_ratio below threshold for N+ bars
    is_compressed = df['vol_ratio'] < compression_threshold
    compression_streak = is_compressed.groupby(
        (~is_compressed).cumsum()
    ).cumcount() + 1
    compression_streak[~is_compressed] = 0
    df['compression_bars'] = compression_streak

    in_compression = df['compression_bars'] >= min_compression_bars

    # Expansion: current bar range exceeds threshold
    bar_range = df['high'] - df['low']
    is_expansion = bar_range > (expansion_multiple * atr_slow)

    # Direction from expansion bar
    bullish_bar = df['close'] > df['open']
    bearish_bar = df['close'] < df['open']

    # Events
    df['event_volexp_long'] = in_compression.shift(1).fillna(False) & is_expansion & bullish_bar & df['is_rth']
    df['event_volexp_short'] = in_compression.shift(1).fillna(False) & is_expansion & bearish_bar & df['is_rth']

    return df


# ===========================================================================
# EVENT 5: VWAP RECLAIM / REJECTION
# ===========================================================================

def detect_vwap_cross(
    df: pd.DataFrame,
    min_bars_on_side: int = 6,
    min_distance_atr: float = 0.3,
    min_bar_of_session: int = 6,
    max_bar_of_session: int = 66,
) -> pd.DataFrame:
    """
    Detect VWAP reclaim (cross from below) and rejection (cross from above).

    A qualified cross requires price to have been on one side of VWAP for
    at least `min_bars_on_side` bars and at a meaningful distance.

    Parameters
    ----------
    df : DataFrame with session columns
    min_bars_on_side : minimum bars on one side before a cross qualifies
    min_distance_atr : minimum distance from VWAP (in ATR units) during the prior period
    min_bar_of_session : earliest bar for event
    max_bar_of_session : latest bar for event
    """
    df = df.copy()

    # Session-anchored VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_vol = typical_price * df['volume']

    rth = df['is_rth']
    dates = df.index.date

    # Compute VWAP per session
    cum_tpv = tp_vol.where(rth, 0).groupby(dates).cumsum()
    cum_vol = df['volume'].where(rth, 0).groupby(dates).cumsum()
    df['vwap'] = cum_tpv / cum_vol.replace(0, np.nan)

    # Distance from VWAP in ATR units
    tr = _true_range(df)
    atr_20 = tr.rolling(20).mean()
    df['vwap_dist_atr'] = (df['close'] - df['vwap']) / atr_20.replace(0, np.nan)

    # Side tracking: was price below VWAP recently?
    below_vwap = df['close'] < df['vwap']
    above_vwap = df['close'] > df['vwap']

    # Count consecutive bars on each side
    below_streak = below_vwap.groupby((~below_vwap).cumsum()).cumcount() + 1
    below_streak[~below_vwap] = 0
    above_streak = above_vwap.groupby((~above_vwap).cumsum()).cumcount() + 1
    above_streak[~above_vwap] = 0

    # Reclaim: was below for N bars, now crosses above
    was_below = below_streak.shift(1) >= min_bars_on_side
    crosses_above = above_vwap & below_vwap.shift(1)

    # Check minimum distance: was meaningfully below
    min_dist_below = df['vwap_dist_atr'].rolling(min_bars_on_side).min().shift(1)
    was_meaningfully_below = min_dist_below < -min_distance_atr

    time_ok = (
        (df['bar_of_session'] >= min_bar_of_session) &
        (df['bar_of_session'] <= max_bar_of_session) &
        df['is_rth']
    )

    df['event_vwap_reclaim'] = was_below & crosses_above & was_meaningfully_below & time_ok

    # Rejection: was above for N bars, now crosses below
    was_above = above_streak.shift(1) >= min_bars_on_side
    crosses_below = below_vwap & above_vwap.shift(1)
    max_dist_above = df['vwap_dist_atr'].rolling(min_bars_on_side).max().shift(1)
    was_meaningfully_above = max_dist_above > min_distance_atr

    df['event_vwap_rejection'] = was_above & crosses_below & was_meaningfully_above & time_ok

    return df


# ===========================================================================
# EVENT 6: OPENING GAP FILL / FADE
# ===========================================================================

def detect_gap(
    df: pd.DataFrame,
    min_gap_pct: float = 0.0015,
    large_gap_pct: float = 0.005,
    max_gap_pct: float = 0.015,
    entry_delay_bars: int = 1,
) -> pd.DataFrame:
    """
    Detect opening gap events.

    Parameters
    ----------
    df : DataFrame with session columns (prior_session_close must exist)
    min_gap_pct : minimum gap as fraction of price (0.0015 = 0.15%)
    large_gap_pct : threshold for "large gap" (continuation hypothesis)
    max_gap_pct : maximum gap to consider (exclude extreme/error gaps)
    entry_delay_bars : bars after open before triggering
    """
    df = df.copy()

    # First bar of each RTH session
    rth = df[df['is_rth']]
    first_bars = rth.groupby(rth.index.date).head(1)

    # Gap calculation
    df['gap_size'] = np.nan
    df['gap_pct'] = np.nan
    df.loc[first_bars.index, 'gap_size'] = (
        df.loc[first_bars.index, 'open'] - df.loc[first_bars.index, 'prior_session_close']
    )
    df.loc[first_bars.index, 'gap_pct'] = (
        df.loc[first_bars.index, 'gap_size'] / df.loc[first_bars.index, 'prior_session_close']
    )

    # Forward-fill gap info to all bars in the session
    df['gap_size'] = df['gap_size'].ffill()
    df['gap_pct'] = df['gap_pct'].ffill()

    # --- Gap Fade events (mean reversion) ---
    abs_gap = df['gap_pct'].abs()
    gap_qualifies = (abs_gap >= min_gap_pct) & (abs_gap < large_gap_pct) & (abs_gap <= max_gap_pct)

    # Trigger on the bar after the entry delay
    entry_bar = df['bar_of_session'] == entry_delay_bars

    df['event_gap_fade_long'] = (df['gap_pct'] < 0) & gap_qualifies & entry_bar & df['is_rth']
    df['event_gap_fade_short'] = (df['gap_pct'] > 0) & gap_qualifies & entry_bar & df['is_rth']

    # --- Large Gap Continuation events ---
    large_gap = abs_gap >= large_gap_pct
    large_qualifies = large_gap & (abs_gap <= max_gap_pct)

    df['event_gap_cont_long'] = (df['gap_pct'] > 0) & large_qualifies & entry_bar & df['is_rth']
    df['event_gap_cont_short'] = (df['gap_pct'] < 0) & large_qualifies & entry_bar & df['is_rth']

    return df


# ===========================================================================
# EVENT 7: MULTI-BAR MOMENTUM EXHAUSTION
# ===========================================================================

def detect_momentum_exhaustion(
    df: pd.DataFrame,
    min_consec_bars: int = 4,
    min_atr_move: float = 1.5,
    atr_lookback: int = 20,
    exhaustion_criteria_needed: int = 2,
    volume_decline_threshold: float = 0.8,
    wick_rejection_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Detect multi-bar momentum exhaustion events.

    Parameters
    ----------
    df : DataFrame with OHLCV
    min_consec_bars : minimum consecutive same-direction bars
    min_atr_move : minimum cumulative move in ATR units
    atr_lookback : ATR calculation window
    exhaustion_criteria_needed : how many exhaustion signals required (out of 4)
    volume_decline_threshold : volume ratio below this = declining
    wick_rejection_threshold : wick ratio above this = rejection
    """
    df = df.copy()

    tr = _true_range(df)
    atr = tr.rolling(atr_lookback).mean()
    bar_range = df['high'] - df['low']
    avg_range = bar_range.rolling(5).mean()
    avg_volume = df['volume'].rolling(5).mean()

    bar_direction = np.sign(df['close'] - df['open'])

    # Consecutive up/down bars
    def _consec(direction_series, target):
        groups = (direction_series != target).cumsum()
        result = direction_series.groupby(groups).cumcount() + 1
        result[direction_series != target] = 0
        return result

    consec_up = _consec(bar_direction, 1)
    consec_down = _consec(bar_direction, -1)

    # Cumulative move over consecutive bars
    bar_body = df['close'] - df['open']

    # Rolling sum of body over last N bars (approximation of move)
    cum_move_up = bar_body.rolling(min_consec_bars).sum()
    cum_move_down = bar_body.rolling(min_consec_bars).sum()

    # Exhaustion signals on current bar
    shrinking_range = bar_range < (0.6 * avg_range)
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    upper_wick_ratio = upper_wick / bar_range.replace(0, np.nan)
    lower_wick_ratio = lower_wick / bar_range.replace(0, np.nan)

    declining_volume = df['volume'] < (volume_decline_threshold * avg_volume)
    close_position = (df['close'] - df['low']) / bar_range.replace(0, np.nan)

    # --- Bearish exhaustion (after up-move) ---
    up_move_long_enough = consec_up >= min_consec_bars
    up_move_big_enough = cum_move_up > (min_atr_move * atr)

    exhaust_signals_bear = (
        shrinking_range.astype(int) +
        (upper_wick_ratio > wick_rejection_threshold).astype(int) +
        declining_volume.astype(int) +
        (close_position < 0.3).astype(int)
    )
    exhaustion_present_bear = exhaust_signals_bear >= exhaustion_criteria_needed

    df['event_exhaustion_short'] = (
        up_move_long_enough & up_move_big_enough &
        exhaustion_present_bear & df['is_rth']
    )

    # --- Bullish exhaustion (after down-move) ---
    down_move_long_enough = consec_down >= min_consec_bars
    down_move_big_enough = cum_move_down.abs() > (min_atr_move * atr)

    exhaust_signals_bull = (
        shrinking_range.astype(int) +
        (lower_wick_ratio > wick_rejection_threshold).astype(int) +
        declining_volume.astype(int) +
        (close_position > 0.7).astype(int)
    )
    exhaustion_present_bull = exhaust_signals_bull >= exhaustion_criteria_needed

    df['event_exhaustion_long'] = (
        down_move_long_enough & down_move_big_enough &
        exhaustion_present_bull & df['is_rth']
    )

    return df


# ===========================================================================
# EVENT 9: INITIAL BALANCE EXTENSION
# ===========================================================================

def detect_ib_extension(
    df: pd.DataFrame,
    ib_period_bars: int = 12,
    min_extension_points: float = 2.0,
    max_entry_bar: int = 54,
    max_ib_width_atr: float = 2.0,
) -> pd.DataFrame:
    """
    Detect Initial Balance (IB) extension events (Market Profile concept).

    IB = range of first `ib_period_bars` of RTH (12 bars = 60 min).
    Extension = first time price exceeds the IB boundary.

    Parameters
    ----------
    df : DataFrame with session columns
    ib_period_bars : number of 5-min bars defining IB (12 = 60 min)
    min_extension_points : minimum points beyond IB to qualify
    max_entry_bar : latest bar to trigger event
    max_ib_width_atr : filter out days with extremely wide IB
    """
    df = df.copy()

    rth = df[df['is_rth']].copy()

    # IB stats per day
    ib_bars = rth[rth['bar_of_session'] < ib_period_bars]
    ib_stats = ib_bars.groupby(ib_bars.index.date).agg(
        ib_high=('high', 'max'),
        ib_low=('low', 'min'),
    )
    ib_stats['ib_range'] = ib_stats['ib_high'] - ib_stats['ib_low']
    ib_stats.index = pd.to_datetime(ib_stats.index)

    # Map to all bars
    date_index = pd.to_datetime(df.index.date)
    for col in ['ib_high', 'ib_low', 'ib_range']:
        df[col] = date_index.map(ib_stats[col]).values

    # ATR for width filter
    tr = _true_range(df)
    atr = tr.rolling(20).mean()
    ib_width_ok = df['ib_range'] < (max_ib_width_atr * atr)

    # Time window
    after_ib = df['bar_of_session'] >= ib_period_bars
    before_cutoff = df['bar_of_session'] <= max_entry_bar
    time_ok = after_ib & before_cutoff & df['is_rth']

    # Extension up
    above_ib = df['high'] > (df['ib_high'] + min_extension_points)
    df['_ib_ext_up_raw'] = above_ib & time_ok & ib_width_ok

    df['event_ib_ext_long'] = False
    if df['_ib_ext_up_raw'].any():
        first_up = df[df['_ib_ext_up_raw']].groupby(
            df[df['_ib_ext_up_raw']].index.date
        ).head(1).index
        df.loc[first_up, 'event_ib_ext_long'] = True

    # Extension down
    below_ib = df['low'] < (df['ib_low'] - min_extension_points)
    df['_ib_ext_dn_raw'] = below_ib & time_ok & ib_width_ok

    df['event_ib_ext_short'] = False
    if df['_ib_ext_dn_raw'].any():
        first_dn = df[df['_ib_ext_dn_raw']].groupby(
            df[df['_ib_ext_dn_raw']].index.date
        ).head(1).index
        df.loc[first_dn, 'event_ib_ext_short'] = True

    # Cleanup
    df = df.drop(columns=['_ib_ext_up_raw', '_ib_ext_dn_raw'])

    return df


# ===========================================================================
# EVENT 10: SINGLE-BAR VOLATILITY SPIKE
# ===========================================================================

def detect_vol_spike(
    df: pd.DataFrame,
    zscore_threshold: float = 2.5,
    atr_lookback: int = 20,
    min_volume_multiple: float = 1.5,
    bullish_close_threshold: float = 0.7,
    bearish_close_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Detect single-bar volatility spike events.

    Parameters
    ----------
    df : DataFrame with OHLCV
    zscore_threshold : TR z-score threshold to qualify as spike
    atr_lookback : window for computing rolling ATR mean and std
    min_volume_multiple : minimum volume relative to rolling average
    bullish_close_threshold : close position above this = bullish resolution
    bearish_close_threshold : close position below this = bearish resolution
    """
    df = df.copy()

    tr = _true_range(df)
    tr_mean = tr.rolling(atr_lookback).mean()
    tr_std = tr.rolling(atr_lookback).std()

    df['tr_zscore'] = (tr - tr_mean) / tr_std.replace(0, np.nan)

    avg_vol = df['volume'].rolling(atr_lookback).mean()
    vol_ok = df['volume'] > (min_volume_multiple * avg_vol)

    bar_range = df['high'] - df['low']
    close_pos = (df['close'] - df['low']) / bar_range.replace(0, np.nan)

    spike = (df['tr_zscore'] > zscore_threshold) & vol_ok & df['is_rth']

    df['event_volspike_long'] = spike & (close_pos > bullish_close_threshold)
    df['event_volspike_short'] = spike & (close_pos < bearish_close_threshold)

    return df


# ===========================================================================
# MASTER PIPELINE
# ===========================================================================

def detect_all_events(
    df: pd.DataFrame,
    include_events: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Run all event detectors with default parameters.

    Parameters
    ----------
    df : DataFrame from load_ohlcv() with DatetimeIndex
    include_events : list of event names to include, or None for all.
        Valid names: 'sweep', 'orb', 'cusum', 'volexp', 'vwap',
                     'gap', 'exhaustion', 'ib_ext', 'volspike'

    Returns
    -------
    DataFrame with all event columns added
    """
    all_events = {
        'sweep', 'orb', 'cusum', 'volexp', 'vwap',
        'gap', 'exhaustion', 'ib_ext', 'volspike',
    }

    if include_events is None:
        include_events = all_events
    else:
        include_events = set(include_events)

    # Session columns are needed by most events
    print("[EVENTS] Adding session columns...")
    df = add_session_columns(df)

    if 'sweep' in include_events:
        print("[EVENTS] Detecting: Prior Session Sweep...")
        df = detect_session_sweep(df)

    if 'orb' in include_events:
        print("[EVENTS] Detecting: Opening Range Breakout...")
        df = detect_orb(df)

    if 'cusum' in include_events:
        print("[EVENTS] Detecting: CUSUM Threshold...")
        df = detect_cusum(df)

    if 'volexp' in include_events:
        print("[EVENTS] Detecting: Vol Compression/Expansion...")
        df = detect_vol_compression_expansion(df)

    if 'vwap' in include_events:
        print("[EVENTS] Detecting: VWAP Reclaim/Rejection...")
        df = detect_vwap_cross(df)

    if 'gap' in include_events:
        print("[EVENTS] Detecting: Opening Gap...")
        df = detect_gap(df)

    if 'exhaustion' in include_events:
        print("[EVENTS] Detecting: Momentum Exhaustion...")
        df = detect_momentum_exhaustion(df)

    if 'ib_ext' in include_events:
        print("[EVENTS] Detecting: IB Extension...")
        df = detect_ib_extension(df)

    if 'volspike' in include_events:
        print("[EVENTS] Detecting: Volatility Spike...")
        df = detect_vol_spike(df)

    # Summary
    event_cols = [c for c in df.columns if c.startswith('event_')]
    print(f"\n[EVENTS] Detection complete. {len(event_cols)} event columns added.")
    for col in event_cols:
        count = df[col].sum()
        years = (df.index[-1] - df.index[0]).days / 365.25
        per_year = count / years if years > 0 else 0
        print(f"  {col}: {int(count)} total ({per_year:.0f}/year)")

    return df


# ===========================================================================
# QUICK VALIDATION SCRIPT
# ===========================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../research_utils')
    from feature_engineering import load_ohlcv

    print("=" * 70)
    print("EVENT DEFINITIONS — Quick Validation")
    print("=" * 70)

    # Load a subset for quick testing
    filepath = '../nq_continuous_5m_converted.csv'
    print(f"\n[LOAD] Loading {filepath}...")
    df = load_ohlcv(filepath)

    # Filter to recent 2 years for quick test
    df = df['2024-01-01':'2026-01-15']
    print(f"[LOAD] Filtered to {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    # Detect events with Tier 1 only for speed
    df = detect_all_events(df, include_events=['sweep', 'orb', 'ib_ext'])

    # Show sample events
    event_cols = [c for c in df.columns if c.startswith('event_')]
    for col in event_cols:
        events = df[df[col]]
        if len(events) > 0:
            print(f"\n[SAMPLE] {col} — first 5 events:")
            print(events[['open', 'high', 'low', 'close', 'volume']].head())
