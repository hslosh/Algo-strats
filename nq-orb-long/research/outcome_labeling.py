"""
outcome_labeling.py
===================
Double-barrier (triple-barrier) outcome labeling for event-driven research.

Implements the labeling methodology from Step 2:
  1. Primary labels: Double-barrier (TP / SL / timeout) → +1 / -1 / 0
  2. Secondary labels: Fixed-horizon forward returns at multiple horizons
  3. Trade metadata: MAE, MFE, time-to-barrier, exit type

Designed to work with event_definitions.py output.

Usage:
    from outcome_labeling import label_events, label_forward_returns
    from event_definitions import detect_all_events
    from feature_engineering import load_ohlcv

    df = load_ohlcv('nq_continuous_5m_converted.csv')
    df = detect_all_events(df, include_events=['sweep', 'orb'])

    # Label sweep-high events (SHORT direction)
    results = label_events(
        df,
        event_col='sweep_high_first_today',
        direction='short',
        sl_atr_multiple=1.0,
        tp_atr_multiple=1.5,
    )
"""

import numpy as np
import pandas as pd
from typing import Optional


# ===========================================================================
# CORE: DOUBLE-BARRIER LABELING
# ===========================================================================

def label_events(
    df: pd.DataFrame,
    event_col: str,
    direction: str,
    sl_atr_multiple: float = 1.0,
    tp_atr_multiple: float = 1.5,
    atr_lookback: int = 14,
    max_holding_bars: int = 48,
    force_session_exit: bool = True,
) -> pd.DataFrame:
    """
    Apply double-barrier labeling to events detected in the DataFrame.

    For each bar where event_col is True, simulate a trade entry at the bar's
    close price and scan forward to determine which barrier is hit first.

    Parameters
    ----------
    df : DataFrame with OHLCV, session columns, and event columns.
         Must have: open, high, low, close, volume, is_rth, bar_of_session
    event_col : name of the boolean event column (e.g., 'sweep_high_first_today')
    direction : 'long' or 'short' — determines barrier orientation
    sl_atr_multiple : stop-loss distance as multiple of ATR at entry
    tp_atr_multiple : take-profit distance as multiple of ATR at entry
    atr_lookback : bars for ATR calculation
    max_holding_bars : maximum bars to hold before forced exit (vertical barrier)
    force_session_exit : if True, exit at end of RTH even if no barrier hit

    Returns
    -------
    DataFrame containing one row per event with columns:
        event_time       : timestamp of the event
        entry_price      : close of the event bar
        atr_at_entry     : ATR(lookback) at event time
        sl_level         : stop-loss price
        tp_level         : take-profit price
        barrier_label    : +1 (TP hit), -1 (SL hit), 0 (timeout)
        exit_type        : 'tp', 'sl', 'timeout', 'session_end'
        exit_price       : price at exit
        exit_time        : timestamp of exit
        barrier_return_pts : P&L in NQ points
        barrier_return_pct : P&L as fraction
        time_to_barrier  : bars from entry to exit
        mae_pts          : Maximum Adverse Excursion (points, always >= 0)
        mfe_pts          : Maximum Favorable Excursion (points, always >= 0)
        mae_atr          : MAE normalized by ATR at entry
        mfe_atr          : MFE normalized by ATR at entry
    """
    if direction not in ('long', 'short'):
        raise ValueError("direction must be 'long' or 'short'")

    # Compute ATR if not already present
    atr_col = f'_atr_{atr_lookback}'
    if atr_col not in df.columns:
        tr = _true_range(df)
        df[atr_col] = tr.rolling(atr_lookback).mean()

    # Get event indices
    event_mask = df[event_col].fillna(False).astype(bool)
    event_indices = df.index[event_mask]

    if len(event_indices) == 0:
        print(f"[LABEL] No events found in column '{event_col}'")
        return pd.DataFrame()

    print(f"[LABEL] Labeling {len(event_indices)} events from '{event_col}' "
          f"(direction={direction}, SL={sl_atr_multiple}x ATR, TP={tp_atr_multiple}x ATR)")

    # Pre-extract arrays for fast scanning
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values
    is_rth = df['is_rth'].values if 'is_rth' in df.columns else np.ones(len(df), dtype=bool)
    bar_of_session = df['bar_of_session'].values if 'bar_of_session' in df.columns else np.zeros(len(df))
    timestamps = df.index
    atrs = df[atr_col].values

    # Positional index lookup
    idx_to_pos = {ts: i for i, ts in enumerate(timestamps)}

    results = []

    for event_ts in event_indices:
        pos = idx_to_pos[event_ts]

        entry_price = closes[pos]
        atr_at_entry = atrs[pos]

        # Skip if ATR is NaN (insufficient warmup data)
        if np.isnan(atr_at_entry) or atr_at_entry <= 0:
            continue

        # Compute barrier levels
        sl_distance = sl_atr_multiple * atr_at_entry
        tp_distance = tp_atr_multiple * atr_at_entry

        if direction == 'long':
            tp_level = entry_price + tp_distance
            sl_level = entry_price - sl_distance
        else:  # short
            tp_level = entry_price - tp_distance
            sl_level = entry_price + sl_distance

        # Determine vertical barrier (max hold or session end)
        max_pos = min(pos + max_holding_bars, len(df) - 1)

        if force_session_exit:
            # Find end of current RTH session
            current_date = timestamps[pos].date()
            session_end_pos = pos
            for j in range(pos + 1, min(pos + max_holding_bars + 10, len(df))):
                if is_rth[j] and timestamps[j].date() == current_date:
                    session_end_pos = j
                elif timestamps[j].date() != current_date or not is_rth[j]:
                    break
            max_pos = min(max_pos, session_end_pos)

        # Scan forward through bars
        label = 0
        exit_type = 'timeout'
        exit_price = closes[max_pos]
        exit_time = timestamps[max_pos]
        time_to_barrier = max_pos - pos
        mae_pts = 0.0  # worst adverse move (always positive number)
        mfe_pts = 0.0  # best favorable move (always positive number)

        for j in range(pos + 1, max_pos + 1):
            bar_high = highs[j]
            bar_low = lows[j]

            # Track MAE and MFE
            if direction == 'long':
                adverse = entry_price - bar_low
                favorable = bar_high - entry_price
            else:
                adverse = bar_high - entry_price
                favorable = entry_price - bar_low

            mae_pts = max(mae_pts, adverse)
            mfe_pts = max(mfe_pts, favorable)

            # Check barrier hits
            if direction == 'long':
                sl_hit = bar_low <= sl_level
                tp_hit = bar_high >= tp_level
            else:
                sl_hit = bar_high >= sl_level
                tp_hit = bar_low <= tp_level

            if sl_hit and tp_hit:
                # Both barriers could have been hit in this bar
                # Conservative convention: assume ADVERSE (SL) hit first
                label = -1
                exit_type = 'sl'
                exit_price = sl_level
                exit_time = timestamps[j]
                time_to_barrier = j - pos
                break
            elif sl_hit:
                label = -1
                exit_type = 'sl'
                exit_price = sl_level
                exit_time = timestamps[j]
                time_to_barrier = j - pos
                break
            elif tp_hit:
                label = 1
                exit_type = 'tp'
                exit_price = tp_level
                exit_time = timestamps[j]
                time_to_barrier = j - pos
                break

        # If loop ended without hitting a barrier
        if label == 0:
            exit_price = closes[max_pos]
            exit_time = timestamps[max_pos]
            time_to_barrier = max_pos - pos
            if force_session_exit and max_pos == session_end_pos:
                exit_type = 'session_end'
            else:
                exit_type = 'timeout'

        # Compute return
        if direction == 'long':
            barrier_return_pts = exit_price - entry_price
        else:
            barrier_return_pts = entry_price - exit_price

        barrier_return_pct = barrier_return_pts / entry_price

        results.append({
            'event_time': event_ts,
            'direction': direction,
            'entry_price': round(entry_price, 2),
            'atr_at_entry': round(atr_at_entry, 2),
            'sl_level': round(sl_level, 2),
            'tp_level': round(tp_level, 2),
            'sl_distance_pts': round(sl_distance, 2),
            'tp_distance_pts': round(tp_distance, 2),
            'barrier_label': label,
            'exit_type': exit_type,
            'exit_price': round(exit_price, 2),
            'exit_time': exit_time,
            'barrier_return_pts': round(barrier_return_pts, 2),
            'barrier_return_pct': round(barrier_return_pct, 6),
            'time_to_barrier': time_to_barrier,
            'mae_pts': round(mae_pts, 2),
            'mfe_pts': round(mfe_pts, 2),
            'mae_atr': round(mae_pts / atr_at_entry, 2) if atr_at_entry > 0 else 0,
            'mfe_atr': round(mfe_pts / atr_at_entry, 2) if atr_at_entry > 0 else 0,
        })

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.set_index('event_time')

    return result_df


# ===========================================================================
# SECONDARY: FIXED-HORIZON FORWARD RETURNS
# ===========================================================================

def label_forward_returns(
    df: pd.DataFrame,
    event_col: str,
    direction: str,
    horizons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute fixed-horizon forward returns for each event.

    Parameters
    ----------
    df : DataFrame with OHLCV data
    event_col : boolean event column name
    direction : 'long' or 'short' — determines return sign convention
    horizons : list of forward bar counts (default: [1, 5, 10, 20, 40])

    Returns
    -------
    DataFrame indexed by event_time with columns per horizon:
        fwd_ret_{h}_pts  : forward return in NQ points
        fwd_ret_{h}_pct  : forward return as fraction
        fwd_dir_{h}      : 1 if profitable, 0 if not
        fwd_mae_{h}_pts  : max adverse excursion over horizon
        fwd_mfe_{h}_pts  : max favorable excursion over horizon
    """
    if horizons is None:
        horizons = [1, 5, 10, 20, 40]

    event_mask = df[event_col].fillna(False).astype(bool)
    event_indices = df.index[event_mask]

    if len(event_indices) == 0:
        return pd.DataFrame()

    # Pre-extract
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    timestamps = df.index
    idx_to_pos = {ts: i for i, ts in enumerate(timestamps)}

    sign = 1 if direction == 'long' else -1
    results = []

    for event_ts in event_indices:
        pos = idx_to_pos[event_ts]
        entry_price = closes[pos]
        row = {'event_time': event_ts, 'entry_price': entry_price}

        for h in horizons:
            end_pos = pos + h
            if end_pos >= len(df):
                # Not enough data for this horizon
                row[f'fwd_ret_{h}_pts'] = np.nan
                row[f'fwd_ret_{h}_pct'] = np.nan
                row[f'fwd_dir_{h}'] = np.nan
                row[f'fwd_mae_{h}_pts'] = np.nan
                row[f'fwd_mfe_{h}_pts'] = np.nan
                continue

            exit_price = closes[end_pos]
            raw_return = exit_price - entry_price
            directed_return = sign * raw_return

            # MAE and MFE over the horizon window
            window_highs = highs[pos + 1:end_pos + 1]
            window_lows = lows[pos + 1:end_pos + 1]

            if direction == 'long':
                mae = entry_price - window_lows.min() if len(window_lows) > 0 else 0
                mfe = window_highs.max() - entry_price if len(window_highs) > 0 else 0
            else:
                mae = window_highs.max() - entry_price if len(window_highs) > 0 else 0
                mfe = entry_price - window_lows.min() if len(window_lows) > 0 else 0

            row[f'fwd_ret_{h}_pts'] = round(directed_return, 2)
            row[f'fwd_ret_{h}_pct'] = round(directed_return / entry_price, 6)
            row[f'fwd_dir_{h}'] = 1 if directed_return > 0 else 0
            row[f'fwd_mae_{h}_pts'] = round(max(0, mae), 2)
            row[f'fwd_mfe_{h}_pts'] = round(max(0, mfe), 2)

        results.append(row)

    result_df = pd.DataFrame(results).set_index('event_time')
    return result_df


# ===========================================================================
# COMBINED LABELING PIPELINE
# ===========================================================================

def label_event_full(
    df: pd.DataFrame,
    event_col: str,
    direction: str,
    sl_atr_multiple: float = 1.0,
    tp_atr_multiple: float = 1.5,
    atr_lookback: int = 14,
    max_holding_bars: int = 48,
    force_session_exit: bool = True,
    fwd_horizons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Full labeling pipeline: double-barrier + forward returns, merged.

    Returns a single DataFrame with all labels and metadata per event.
    """
    # Double-barrier labels
    barrier_df = label_events(
        df, event_col, direction,
        sl_atr_multiple=sl_atr_multiple,
        tp_atr_multiple=tp_atr_multiple,
        atr_lookback=atr_lookback,
        max_holding_bars=max_holding_bars,
        force_session_exit=force_session_exit,
    )

    if len(barrier_df) == 0:
        return barrier_df

    # Forward returns
    fwd_df = label_forward_returns(df, event_col, direction, horizons=fwd_horizons)

    # Merge on event_time index
    if len(fwd_df) > 0:
        # Drop duplicate entry_price column from fwd_df
        fwd_df = fwd_df.drop(columns=['entry_price'], errors='ignore')
        result = barrier_df.join(fwd_df, how='left')
    else:
        result = barrier_df

    return result


# ===========================================================================
# SUMMARY STATISTICS
# ===========================================================================

def print_label_summary(labeled_df: pd.DataFrame, event_name: str = "Event"):
    """
    Print comprehensive summary statistics for labeled events.

    Parameters
    ----------
    labeled_df : output from label_events() or label_event_full()
    event_name : display name for the event
    """
    if len(labeled_df) == 0:
        print(f"[SUMMARY] {event_name}: No events to summarize.")
        return

    n = len(labeled_df)
    direction = labeled_df['direction'].iloc[0]

    # Barrier label distribution
    label_counts = labeled_df['barrier_label'].value_counts().sort_index()
    tp_count = label_counts.get(1, 0)
    sl_count = label_counts.get(-1, 0)
    timeout_count = label_counts.get(0, 0)

    win_rate = tp_count / n * 100
    loss_rate = sl_count / n * 100
    timeout_rate = timeout_count / n * 100

    # Returns
    avg_return_pts = labeled_df['barrier_return_pts'].mean()
    median_return_pts = labeled_df['barrier_return_pts'].median()
    total_return_pts = labeled_df['barrier_return_pts'].sum()
    std_return_pts = labeled_df['barrier_return_pts'].std()

    # Winners vs losers
    winners = labeled_df[labeled_df['barrier_label'] == 1]
    losers = labeled_df[labeled_df['barrier_label'] == -1]

    avg_win = winners['barrier_return_pts'].mean() if len(winners) > 0 else 0
    avg_loss = losers['barrier_return_pts'].mean() if len(losers) > 0 else 0

    # Profit factor
    gross_profit = winners['barrier_return_pts'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['barrier_return_pts'].sum()) if len(losers) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Expected value per trade (in dollars for NQ at $20/point)
    ev_pts = avg_return_pts
    ev_dollars = ev_pts * 20  # NQ multiplier

    # Time to barrier
    avg_time = labeled_df['time_to_barrier'].mean()
    median_time = labeled_df['time_to_barrier'].median()

    # MAE / MFE
    avg_mae = labeled_df['mae_pts'].mean()
    avg_mfe = labeled_df['mfe_pts'].mean()

    # Date range
    first_event = labeled_df.index.min()
    last_event = labeled_df.index.max()
    years = (last_event - first_event).days / 365.25
    events_per_year = n / years if years > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  OUTCOME SUMMARY: {event_name} ({direction.upper()})")
    print(f"{'=' * 70}")
    print(f"  Date range:      {first_event.date()} → {last_event.date()} ({years:.1f} years)")
    print(f"  Total events:    {n} ({events_per_year:.0f}/year)")
    print(f"")
    print(f"  ┌── BARRIER LABELS ─────────────────────────────────────────┐")
    print(f"  │  Winners (TP hit):   {tp_count:>5}  ({win_rate:>5.1f}%)                  │")
    print(f"  │  Losers  (SL hit):   {sl_count:>5}  ({loss_rate:>5.1f}%)                  │")
    print(f"  │  Timeouts:           {timeout_count:>5}  ({timeout_rate:>5.1f}%)                  │")
    print(f"  └────────────────────────────────────────────────────────────┘")
    print(f"")
    print(f"  ┌── RETURNS (NQ points) ────────────────────────────────────┐")
    print(f"  │  Expected Value:   {ev_pts:>+8.2f} pts/trade (${ev_dollars:>+.0f}/trade)  │")
    print(f"  │  Median return:    {median_return_pts:>+8.2f} pts                          │")
    print(f"  │  Std deviation:    {std_return_pts:>8.2f} pts                          │")
    print(f"  │  Avg winner:       {avg_win:>+8.2f} pts                          │")
    print(f"  │  Avg loser:        {avg_loss:>+8.2f} pts                          │")
    print(f"  │  Profit factor:    {profit_factor:>8.2f}x                           │")
    print(f"  │  Total cumulative: {total_return_pts:>+8.1f} pts (${total_return_pts*20:>+,.0f})       │")
    print(f"  └────────────────────────────────────────────────────────────┘")
    print(f"")
    print(f"  ┌── TRADE DYNAMICS ─────────────────────────────────────────┐")
    print(f"  │  Avg time to exit:  {avg_time:>5.1f} bars ({avg_time*5:.0f} min)              │")
    print(f"  │  Med time to exit:  {median_time:>5.1f} bars ({median_time*5:.0f} min)              │")
    print(f"  │  Avg MAE:           {avg_mae:>5.1f} pts (worst drawdown)          │")
    print(f"  │  Avg MFE:           {avg_mfe:>5.1f} pts (best unrealized)         │")
    print(f"  │  MFE/MAE ratio:     {avg_mfe/avg_mae:>5.2f}x                             │" if avg_mae > 0 else "  │  MFE/MAE ratio:     N/A                               │")
    print(f"  └────────────────────────────────────────────────────────────┘")

    # Exit type breakdown
    exit_types = labeled_df['exit_type'].value_counts()
    print(f"\n  Exit type breakdown:")
    for et, count in exit_types.items():
        print(f"    {et:>12s}: {count:>5} ({count/n*100:.1f}%)")

    # Forward returns summary (if available)
    fwd_cols = [c for c in labeled_df.columns if c.startswith('fwd_ret_') and c.endswith('_pts')]
    if fwd_cols:
        print(f"\n  ┌── FORWARD RETURNS (NQ points, by horizon) ────────────────┐")
        for col in fwd_cols:
            h = col.replace('fwd_ret_', '').replace('_pts', '')
            avg = labeled_df[col].mean()
            pct_positive = (labeled_df[col] > 0).mean() * 100
            print(f"  │  {h:>3}-bar: {avg:>+7.2f} pts avg, {pct_positive:>5.1f}% positive              │")
        print(f"  └────────────────────────────────────────────────────────────┘")

    print()


# ===========================================================================
# UTILITIES
# ===========================================================================

from research_utils.utils import true_range as _true_range


# ===========================================================================
# MULTI-EVENT BATCH LABELING
# ===========================================================================

def label_all_tier1_events(
    df: pd.DataFrame,
    sl_atr_multiple: float = 1.0,
    tp_atr_multiple: float = 1.5,
    atr_lookback: int = 14,
    max_holding_bars: int = 48,
) -> dict[str, pd.DataFrame]:
    """
    Label all Tier 1 events (sweep + ORB) with default parameters.

    Returns a dict mapping event names to their labeled DataFrames.
    """
    event_configs = [
        ('sweep_high_first_today', 'short', 'Sweep Prior Session High → SHORT'),
        ('sweep_low_first_today',  'long',  'Sweep Prior Session Low → LONG'),
        ('event_orb_long',         'long',  'Opening Range Breakout → LONG'),
        ('event_orb_short',        'short', 'Opening Range Breakout → SHORT'),
    ]

    results = {}

    for event_col, direction, display_name in event_configs:
        if event_col not in df.columns:
            print(f"[SKIP] {event_col} not found in DataFrame — run detect_all_events first")
            continue

        labeled = label_event_full(
            df, event_col, direction,
            sl_atr_multiple=sl_atr_multiple,
            tp_atr_multiple=tp_atr_multiple,
            atr_lookback=atr_lookback,
            max_holding_bars=max_holding_bars,
        )

        if len(labeled) > 0:
            print_label_summary(labeled, display_name)

        results[event_col] = labeled

    return results


# ===========================================================================
# VALIDATION SCRIPT
# ===========================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../research_utils')
    sys.path.insert(0, '.')

    from feature_engineering import load_ohlcv
    from event_definitions import add_session_columns, detect_session_sweep, detect_orb

    print("=" * 70)
    print("OUTCOME LABELING — Validation Run")
    print("=" * 70)

    filepath = '../nq_continuous_5m_converted.csv'
    print(f"\n[LOAD] Loading {filepath}...")
    df = load_ohlcv(filepath)
    df = df['2024-01-01':'2026-01-14']
    print(f"[LOAD] {len(df)} bars loaded")

    # Detect events
    print("\n[EVENTS] Detecting Tier 1 events...")
    df = add_session_columns(df)
    df = detect_session_sweep(df)
    df = detect_orb(df)

    # Label all Tier 1 events
    print("\n[LABEL] Labeling all Tier 1 events...")
    results = label_all_tier1_events(df)

    print("\n[DONE] Validation complete.")
