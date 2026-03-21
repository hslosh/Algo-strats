"""
Step 7 — Backtesting & Validation
==================================

Bar-by-bar backtest with slippage, commissions, and risk management.
Validates the complete ORB Long strategy in realistic conditions.

Usage
-----
    from backtest_validation import run_full_step7
    results = run_full_step7()
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))  # allow bare imports from research/

from research.event_features import build_model_dataset, get_feature_columns
from research.model_design import (
    select_features, build_walk_forward_splits, prepare_labels,
    train_model, predict_proba, calibrate_probabilities,
    apply_calibration,
)
from research.config import CANONICAL_THRESHOLD, TICK_SIZE, is_high_impact_day
from research.strategy_construction import (
    StrategyConfig, compute_position_size, get_size_multiplier,
    compute_performance_metrics, monte_carlo_drawdown,
    print_performance_report, print_monte_carlo_report, print_pass_fail,
)


# =====================================================================
# 1. BACKTEST CONFIGURATION
# =====================================================================

@dataclass
class BacktestConfig:
    """Bar-by-bar backtest parameters."""

    # Strategy params
    threshold: float = CANONICAL_THRESHOLD
    direction: str = 'long'       # 'long' or 'short'
    sl_atr_mult: float = 1.0
    tp_atr_mult: float = 1.5
    max_holding_bars: int = 48
    force_session_exit: bool = True

    # Execution costs
    slippage_pts: float = 0.50       # per side
    commission_per_rt: float = 4.50  # per round-trip per contract

    # Account
    account_size: float = 50_000.0
    point_value: float = 20.0
    risk_per_trade_pct: float = 0.015
    max_contracts: int = 2

    # Sizing tiers (will be set based on threshold)
    size_tiers: list = field(default_factory=list)

    # Risk limits
    max_daily_loss: float = -1_000.0
    max_daily_trades: int = 3
    consec_loss_pause: int = 3
    max_drawdown: float = -2_500.0

    # Trailing stop
    use_trailing_stop: bool = False
    trail_activation_mult: float = 0.5   # activate after 0.5×ATR profit
    trail_distance_mult: float = 0.75    # trail 0.75×ATR behind running high

    def __post_init__(self):
        if not self.size_tiers:
            th = self.threshold
            self.size_tiers = [
                (0.00, th, 0.0),
                (th, 0.65, 0.50),
                (0.65, 0.75, 0.75),
                (0.75, 1.00, 1.00),
            ]


# =====================================================================
# 2. BAR-BY-BAR BACKTEST ENGINE
# =====================================================================

def _time_subtract(n_minutes, close_hour=16, close_min=0):
    """Return a time object n_minutes before 16:00 ET."""
    from datetime import time as dt_time
    total = close_hour * 60 + close_min - n_minutes
    return dt_time(total // 60, total % 60)


def _get_atr(df_bars, idx, period=14):
    """Compute ATR at bar idx using a backward-looking window (no lookahead)."""
    start = max(0, idx - period)
    window = df_bars.iloc[start:idx + 1]
    trs = np.maximum(
        window['high'] - window['low'],
        np.maximum(
            abs(window['high'] - window['close'].shift(1)),
            abs(window['low']  - window['close'].shift(1))
        )
    ).fillna(window['high'] - window['low'])
    return trs.mean()


def run_bar_by_bar_backtest(df: pd.DataFrame,
                            oos_predictions: pd.DataFrame,
                            config: BacktestConfig,
                            verbose: bool = True) -> dict:
    """
    True bar-by-bar backtest with SL/TP monitoring on every bar.

    Iterates every bar in df (not just event times). Entries use a 2-bar
    confirmation delay after ORB event signal. Exits are checked on each
    bar against stop-loss, take-profit, timeout, and end-of-day constraints.
    SL-first on tie (SL checked before TP).

    Parameters
    ----------
    df : DataFrame
        Full OHLCV data with features (bar-level, all bars), DatetimeIndex
    oos_predictions : DataFrame
        OOS events with calibrated_prob (indexed by event_time)
    config : BacktestConfig

    Returns
    -------
    dict with trades (DataFrame), daily_pnl (Series), final_equity (float), config
    """
    tick = TICK_SIZE                    # NQ tick size in index points
    slippage_pts = config.slippage_pts  # default 0.50 pts per side

    # Sort predictions chronologically
    preds = oos_predictions.sort_index()
    prob_lookup = preds['calibrated_prob'].to_dict()

    # ── Build pending entries: event_bar + 2-bar confirmation delay ──
    # ORB trigger fires at event_bar close; entry at open of event_bar+2
    pending_entries = {}
    for event_time, prob in prob_lookup.items():
        if event_time not in df.index:
            continue
        if prob < config.threshold:
            continue
        event_bar_iloc = df.index.get_loc(event_time)
        if isinstance(event_bar_iloc, slice):
            event_bar_iloc = event_bar_iloc.start
        entry_candidate_iloc = event_bar_iloc + 2
        if entry_candidate_iloc < len(df):
            event_row = preds.loc[event_time] if event_time in preds.index else None
            pending_entries[entry_candidate_iloc] = {
                'prob': prob,
                'event_time': event_time,
                'event_row': event_row,
                'direction': config.direction,
            }

    # ── State tracking ──
    equity = config.account_size
    peak_equity = equity
    trades = []
    daily_pnl = {}
    daily_trade_count = {}
    consec_losses = 0
    circuit_breaker_until = None

    # ── Position state ──
    in_position = False
    entry_price = None
    stop_loss = None
    take_profit = None
    entry_bar_idx = None
    bars_held = 0
    position_info = {}
    running_high = 0.0
    running_low = float('inf')
    trail_stop = 0.0
    trail_active = False

    skipped_days = set()
    eod_cutoff = _time_subtract(30)  # 15:30 ET — flatten 30 min before close

    # ── Main bar loop ──
    for i, (ts, bar) in enumerate(df.iterrows()):
        bar_date = ts.date()
        bar_time = ts.time()

        # Initialize daily tracking
        if bar_date not in daily_pnl:
            daily_pnl[bar_date] = 0.0
            daily_trade_count[bar_date] = 0

        # Daily loss cap: check at bar open BEFORE any entry (P4-D)
        if daily_pnl[bar_date] <= config.max_daily_loss:
            skipped_days.add(bar_date)

        # ==========================================================
        # EXIT CHECK — runs on every bar while in position
        # ==========================================================
        if in_position:
            bars_held += 1
            exit_price = None
            exit_reason = None

            # Update trailing stop state
            if config.use_trailing_stop:
                atr_at_entry = position_info.get('atr_at_entry', 20.0)
                if config.direction == 'long':
                    running_high = max(running_high, bar['high'])
                    profit_from_entry = running_high - entry_price
                else:
                    running_low = min(running_low, bar['low'])
                    profit_from_entry = entry_price - running_low
                if profit_from_entry >= config.trail_activation_mult * atr_at_entry:
                    trail_active = True
                    if config.direction == 'long':
                        new_trail = running_high - config.trail_distance_mult * atr_at_entry
                        trail_stop = max(trail_stop, new_trail)
                    else:
                        new_trail = running_low + config.trail_distance_mult * atr_at_entry
                        trail_stop = min(trail_stop, new_trail)

            # Direction-conditional SL/TP checks
            # SL-first on tie: checked before TP
            if config.direction == 'long':
                sl_hit = bar['low'] <= stop_loss
                tp_hit = bar['high'] >= take_profit
                trail_hit = config.use_trailing_stop and trail_active and bar['low'] <= trail_stop
            else:
                sl_hit = bar['high'] >= stop_loss
                tp_hit = bar['low'] <= take_profit
                trail_hit = config.use_trailing_stop and trail_active and bar['high'] >= trail_stop

            if sl_hit:
                if config.direction == 'long':
                    exit_price = stop_loss - slippage_pts
                else:
                    exit_price = stop_loss + slippage_pts
                exit_reason = 'stop_loss'
            # Trailing stop hit (after SL, before TP)
            elif trail_hit:
                if config.direction == 'long':
                    exit_price = trail_stop - slippage_pts
                else:
                    exit_price = trail_stop + slippage_pts
                exit_reason = 'trailing_stop'
            # TP hit
            elif tp_hit:
                if config.direction == 'long':
                    exit_price = take_profit - slippage_pts
                else:
                    exit_price = take_profit + slippage_pts
                exit_reason = 'take_profit'
            # Timeout: max bars held
            elif bars_held >= config.max_holding_bars:
                exit_price = bar['close'] - slippage_pts
                exit_reason = 'timeout'
            # EOD flatten: 30 min before session close
            elif config.force_session_exit and bar_time >= eod_cutoff:
                exit_price = bar['close'] - slippage_pts
                exit_reason = 'eod_flatten'

            if exit_price is not None:
                contracts = position_info.get('contracts', 1)
                if config.direction == 'long':
                    pnl_pts = exit_price - entry_price
                else:
                    pnl_pts = entry_price - exit_price
                commission = config.commission_per_rt * contracts
                pnl_dollars = (pnl_pts * config.point_value * contracts
                               - commission)

                equity += pnl_dollars
                peak_equity = max(peak_equity, equity)
                daily_pnl[bar_date] += pnl_dollars

                if pnl_dollars < 0:
                    consec_losses += 1
                else:
                    consec_losses = 0

                trades.append({
                    'event_time': position_info.get('event_time', ts),
                    'session_date': bar_date,
                    'entry_time': df.index[entry_bar_idx],
                    'exit_time': ts,
                    'prob': position_info.get('prob', 0),
                    'size_mult': position_info.get('size_mult', 1.0),
                    'contracts': contracts,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'sl_distance_pts': position_info.get('sl_distance_pts', 0),
                    'return_pts': pnl_pts,
                    'return_pts_raw': pnl_pts + slippage_pts * 2,
                    'return_pts_net': pnl_pts,
                    'slippage_pts': slippage_pts * 2,
                    'commission': commission,
                    'pnl_dollars': pnl_dollars,
                    'equity_after': equity,
                    'drawdown': equity - peak_equity,
                    'exit_type': exit_reason,
                    'barrier_label': 1 if pnl_pts > 0 else 0,
                    'bars_held': bars_held,
                })

                in_position = False
                entry_price = None
                stop_loss = None
                take_profit = None
                entry_bar_idx = None
                bars_held = 0
                position_info = {}
                running_high = 0.0
                running_low = float('inf')
                trail_stop = 0.0
                trail_active = False

        # ==========================================================
        # ENTRY CHECK — only when flat, on pending-entry bars
        # ==========================================================
        if (not in_position
                and bar_date not in skipped_days
                and i in pending_entries
                and bar_time < eod_cutoff):

            # ── Risk gates ──
            skip_entry = False

            # Circuit breaker (drawdown cooldown)
            if circuit_breaker_until is not None:
                if bar_date >= circuit_breaker_until:
                    circuit_breaker_until = None
                else:
                    skip_entry = True

            # Drawdown check
            if not skip_entry:
                drawdown = equity - peak_equity
                if drawdown <= config.max_drawdown:
                    try:
                        breaker_date = (pd.Timestamp(bar_date)
                                        + pd.Timedelta(days=8))
                        circuit_breaker_until = breaker_date.date()
                    except Exception:
                        pass
                    skip_entry = True

            # Daily trade limit
            if (not skip_entry
                    and daily_trade_count[bar_date] >= config.max_daily_trades):
                skip_entry = True

            # Consecutive loss pause: halt rest of day (P4-C fix)
            if not skip_entry and consec_losses >= config.consec_loss_pause:
                skipped_days.add(bar_date)
                consec_losses = 0
                skip_entry = True

            # Phase 2: Regime filter (direction-aware)
            if not skip_entry and config.direction == 'long' and 'regime_long_allowed' in df.columns:
                if not bar.get('regime_long_allowed', True):
                    skip_entry = True

            # Phase 3: FOMC filter
            if not skip_entry and is_high_impact_day(bar_date):
                skip_entry = True

            if not skip_entry:
                ev_info = pending_entries[i]

                # Compute ATR for position sizing and SL/TP
                atr_pts = _get_atr(df, i, period=14)
                sl_dist = config.sl_atr_mult * atr_pts

                if sl_dist > 0:
                    # Position sizing via strategy_construction
                    strat_config = StrategyConfig(
                        threshold=config.threshold,
                        size_tiers=config.size_tiers,
                        account_size=config.account_size,
                        point_value=config.point_value,
                        risk_per_trade_pct=config.risk_per_trade_pct,
                        max_contracts=config.max_contracts,
                    )
                    contracts = compute_position_size(
                        strat_config, ev_info['prob'], sl_dist, equity
                    )

                    if contracts > 0:
                        if config.direction == 'long':
                            entry_price = bar['open'] + slippage_pts
                            stop_loss = entry_price - config.sl_atr_mult * atr_pts
                            take_profit = entry_price + config.tp_atr_mult * atr_pts
                        else:  # short
                            entry_price = bar['open'] - slippage_pts
                            stop_loss = entry_price + config.sl_atr_mult * atr_pts
                            take_profit = entry_price - config.tp_atr_mult * atr_pts

                        entry_bar_idx = i
                        bars_held = 0
                        in_position = True
                        daily_trade_count[bar_date] += 1

                        # Initialize trailing stop state
                        running_high = entry_price
                        running_low = entry_price
                        trail_stop = float('inf') if config.direction == 'short' else 0.0
                        trail_active = False

                        position_info = {
                            'prob': ev_info['prob'],
                            'event_time': ev_info.get('event_time'),
                            'contracts': contracts,
                            'size_mult': get_size_multiplier(
                                strat_config, ev_info['prob']
                            ),
                            'sl_distance_pts': sl_dist,
                            'atr_at_entry': atr_pts,
                        }

    # ── Build output ──
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    daily_df = pd.Series(daily_pnl, name='daily_pnl').sort_index()

    if verbose and not trades_df.empty:
        cost_total = (trades_df['slippage_pts'].sum() * config.point_value *
                      trades_df['contracts'].mean() +
                      trades_df['commission'].sum())
        print(f"  Trades: {len(trades_df)}, "
              f"Total slippage: {trades_df['slippage_pts'].sum():.0f} pts, "
              f"Total commission: ${trades_df['commission'].sum():,.0f}")

    return {
        'trades': trades_df,
        'daily_pnl': daily_df,
        'final_equity': equity,
        'config': config,
    }


# =====================================================================
# 3. ROLLING STABILITY ANALYSIS
# =====================================================================

def rolling_stability(trades_df: pd.DataFrame,
                       window_months: int = 6) -> pd.DataFrame:
    """
    Compute rolling performance windows.
    """
    if trades_df.empty:
        return pd.DataFrame()

    trades = trades_df.copy()
    trades['month'] = pd.to_datetime(trades['session_date']).dt.to_period('M')

    # Monthly aggregation
    monthly = trades.groupby('month').agg(
        n=('pnl_dollars', 'count'),
        pnl=('pnl_dollars', 'sum'),
        avg_pnl=('pnl_dollars', 'mean'),
        wr=('barrier_label', lambda x: (x > 0).mean()),
    ).reset_index()

    if len(monthly) < window_months:
        return monthly

    # Rolling metrics
    results = []
    for i in range(window_months, len(monthly) + 1):
        window = monthly.iloc[i - window_months:i]
        total_trades = window['n'].sum()
        total_pnl = window['pnl'].sum()
        avg_monthly_pnl = window['pnl'].mean()
        std_monthly_pnl = window['pnl'].std()
        sharpe_6m = (avg_monthly_pnl / std_monthly_pnl * np.sqrt(12)
                     if std_monthly_pnl > 0 else 0)
        months_positive = (window['pnl'] > 0).mean()

        results.append({
            'end_month': window['month'].iloc[-1],
            'n_trades': total_trades,
            'total_pnl': total_pnl,
            'monthly_sharpe': sharpe_6m,
            'months_positive': months_positive,
        })

    return pd.DataFrame(results)


def print_rolling_stability(rolling_df: pd.DataFrame):
    """Print rolling performance analysis."""
    if rolling_df.empty:
        print("  No rolling data available.")
        return

    print()
    print("  Rolling 6-Month Performance Windows:")
    print(f"  {'End Month':>10}  {'Trades':>6}  {'P&L':>10}  "
          f"{'Sharpe':>7}  {'Mo Pos':>6}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*6}")

    for _, r in rolling_df.iterrows():
        print(f"  {str(r['end_month']):>10}  {r['n_trades']:>6.0f}  "
              f"${r['total_pnl']:>+9,.0f}  {r['monthly_sharpe']:>+7.2f}  "
              f"{r['months_positive']:>5.0%}")

    # Summary
    min_sharpe = rolling_df['monthly_sharpe'].min()
    max_sharpe = rolling_df['monthly_sharpe'].max()
    pct_positive_windows = (rolling_df['total_pnl'] > 0).mean()
    print()
    print(f"  Sharpe range: [{min_sharpe:+.2f}, {max_sharpe:+.2f}]")
    print(f"  {pct_positive_windows:.0%} of 6-month windows profitable")

    worst_ok = min_sharpe > -1.0
    sym = "\u2713" if worst_ok else "\u2717"
    print(f"  {sym} Worst 6m Sharpe > -1.0: {min_sharpe:+.2f}")
    print()


# =====================================================================
# 4. HOLDOUT TEST
# =====================================================================

def holdout_test(trades_df: pd.DataFrame,
                  holdout_start: str,
                  config: BacktestConfig) -> dict:
    """
    Test performance on the final holdout period.
    """
    if trades_df.empty:
        return {'error': 'No trades'}

    cutoff = pd.Timestamp(holdout_start).date()
    holdout = trades_df[trades_df['session_date'] >= cutoff]
    pre_holdout = trades_df[trades_df['session_date'] < cutoff]

    if holdout.empty:
        return {
            'holdout_trades': 0,
            'holdout_pnl': 0,
            'holdout_wr': 0,
            'pre_holdout_trades': len(pre_holdout),
            'pre_holdout_pnl': pre_holdout['pnl_dollars'].sum() if not pre_holdout.empty else 0,
        }

    return {
        'holdout_start': str(cutoff),
        'holdout_trades': len(holdout),
        'holdout_pnl': holdout['pnl_dollars'].sum(),
        'holdout_wr': (holdout['barrier_label'] > 0).mean(),
        'holdout_avg_pnl': holdout['pnl_dollars'].mean(),
        'holdout_pf': (holdout[holdout['pnl_dollars'] > 0]['pnl_dollars'].sum() /
                       abs(holdout[holdout['pnl_dollars'] < 0]['pnl_dollars'].sum())
                       if (holdout['pnl_dollars'] < 0).any() else float('inf')),
        'pre_holdout_trades': len(pre_holdout),
        'pre_holdout_pnl': pre_holdout['pnl_dollars'].sum(),
        'pre_holdout_wr': (pre_holdout['barrier_label'] > 0).mean() if not pre_holdout.empty else 0,
    }


def print_holdout_report(holdout: dict):
    """Print holdout test results."""
    print()
    print("  +-- HOLDOUT TEST (Final 6 Months) ────────────────────────+")

    if holdout.get('holdout_trades', 0) == 0:
        print("  |  No trades in holdout period                              |")
        print("  |  (circuit breaker may have blocked all trades)            |")
    else:
        n = holdout['holdout_trades']
        pnl = holdout['holdout_pnl']
        wr = holdout['holdout_wr']
        avg = holdout['holdout_avg_pnl']
        pf = holdout.get('holdout_pf', 0)
        print(f"  |  Period:         {holdout.get('holdout_start', '?'):>14}+                |")
        print(f"  |  Trades:         {n:>6}                                    |")
        print(f"  |  Total P&L:      ${pnl:>+9,.0f}                             |")
        print(f"  |  Avg P&L/trade:  ${avg:>+9,.0f}                             |")
        print(f"  |  Win rate:       {wr:>6.1%}                                |")
        pf_str = f"{pf:.2f}x" if pf < 100 else "inf"
        print(f"  |  Profit factor:  {pf_str:>6}                                |")

    pre_n = holdout.get('pre_holdout_trades', 0)
    pre_pnl = holdout.get('pre_holdout_pnl', 0)
    print(f"  |                                                          |")
    print(f"  |  Pre-holdout:    {pre_n} trades, ${pre_pnl:>+,.0f}           |")

    ok = holdout.get('holdout_pnl', 0) > 0
    sym = "\u2713" if ok else "\u2717"
    tag = "PASS" if ok else "FAIL"
    print(f"  |                                                          |")
    print(f"  |  Holdout profitable?  {sym} {tag}                          |")
    print("  +----------------------------------------------------------+")
    print()


# =====================================================================
# 5. COST IMPACT ANALYSIS
# =====================================================================

def print_cost_analysis(trades_df: pd.DataFrame, config: BacktestConfig):
    """Show the impact of slippage and commissions."""
    if trades_df.empty:
        return

    n = len(trades_df)
    avg_contracts = trades_df['contracts'].mean()

    total_slip_pts = trades_df['slippage_pts'].sum()
    total_slip_dollars = total_slip_pts * config.point_value * avg_contracts
    total_commission = trades_df['commission'].sum()
    total_cost = total_slip_dollars + total_commission

    gross_pnl = (trades_df['return_pts_raw'] * trades_df['contracts'] *
                 config.point_value).sum()
    net_pnl = trades_df['pnl_dollars'].sum()

    print()
    print("  +-- COST ANALYSIS ────────────────────────────────────────+")
    print(f"  |  Trades:           {n:>6}                                |")
    print(f"  |  Avg contracts:    {avg_contracts:>6.1f}                                |")
    print(f"  |                                                          |")
    print(f"  |  Total slippage:   ${total_slip_dollars:>+9,.0f}  "
          f"({total_slip_pts:.0f} pts)      |")
    print(f"  |  Total commission: ${total_commission:>+9,.0f}                       |")
    print(f"  |  Total costs:      ${total_cost:>+9,.0f}                       |")
    print(f"  |                                                          |")
    print(f"  |  Gross P&L:        ${gross_pnl:>+9,.0f}                       |")
    print(f"  |  Net P&L:          ${net_pnl:>+9,.0f}                       |")
    cost_pct = (total_cost / gross_pnl * 100) if gross_pnl > 0 else 0
    print(f"  |  Cost drag:        {cost_pct:>6.1f}%                            |")
    print("  +----------------------------------------------------------+")
    print()


# =====================================================================
# 6. FINAL PASS/FAIL
# =====================================================================

def print_step7_verdict(metrics: dict, mc: dict, holdout: dict,
                         rolling_df: pd.DataFrame):
    """Print final Step 7 pass/fail."""
    print()
    print("  " + "=" * 56)
    print("    STEP 7 — FINAL BACKTEST VERDICT")
    print("  " + "=" * 56)

    checks = []

    # 1. Net profit > 0 after costs
    ok = metrics.get('total_pnl', 0) > 0
    checks.append(ok)
    sym = "\u2713" if ok else "\u2717"
    print(f"  {sym}  Net profit > 0 (after costs): ${metrics.get('total_pnl', 0):>+,.0f}")

    # 2. Sharpe >= 1.5
    ok = metrics.get('sharpe', 0) >= 1.5
    checks.append(ok)
    sym = "\u2713" if ok else "\u2717"
    print(f"  {sym}  Sharpe >= 1.5:                 {metrics.get('sharpe', 0):.2f}")

    # 3. Max DD < $3,000
    ok = metrics.get('max_drawdown', -9999) > -3_000
    checks.append(ok)
    sym = "\u2713" if ok else "\u2717"
    print(f"  {sym}  Max DD < $3,000:               ${metrics.get('max_drawdown', 0):>+,.0f}")

    # 4. PF >= 1.3
    ok = metrics.get('profit_factor', 0) >= 1.3
    checks.append(ok)
    sym = "\u2713" if ok else "\u2717"
    print(f"  {sym}  Profit factor >= 1.3:          {metrics.get('profit_factor', 0):.2f}x")

    # 5. Holdout profitable
    ok = holdout.get('holdout_pnl', 0) > 0 or holdout.get('holdout_trades', 0) == 0
    checks.append(ok)
    sym = "\u2713" if ok else "\u2717"
    h_pnl = holdout.get('holdout_pnl', 0)
    h_n = holdout.get('holdout_trades', 0)
    print(f"  {sym}  Holdout profitable:            ${h_pnl:>+,.0f} ({h_n} trades)")

    # 6. Rolling Sharpe never < -1.0
    if not rolling_df.empty:
        min_sharpe = rolling_df['monthly_sharpe'].min()
        ok = min_sharpe > -1.0
        checks.append(ok)
        sym = "\u2713" if ok else "\u2717"
        print(f"  {sym}  No catastrophic 6m window:    {min_sharpe:+.2f}")

    print()
    if all(checks):
        print("  >>> VERDICT: PASS — Strategy approved for paper trading <<<")
    else:
        n_fail = sum(1 for c in checks if not c)
        print(f"  >>> VERDICT: FAIL — {n_fail} criterion(s) not met <<<")
        print("       Review failed criteria before deploying.")
    print()


# =====================================================================
# 7. PORTFOLIO BACKTEST WRAPPER
# =====================================================================

def run_portfolio_backtest(signal_specs: list, verbose: bool = True) -> dict:
    """
    Run multiple signal pipelines and merge into a single portfolio.

    Parameters
    ----------
    signal_specs : list of dicts
        Each dict has: {event_col, direction, event_type}
        Example: [
            {'event_col': 'event_orb_long',  'direction': 'long',  'event_type': 'orb'},
            {'event_col': 'event_orb_short', 'direction': 'short', 'event_type': 'orb'},
        ]
    verbose : bool

    Returns
    -------
    dict with per-signal results, combined trades, portfolio metrics, overlap stats
    """
    signal_results = {}
    all_trades = []

    for spec in signal_specs:
        key = f"{spec['event_col']}_{spec['direction']}"
        print(f"\n{'='*72}")
        print(f"SIGNAL: {key}")
        print(f"{'='*72}")

        result = run_full_step7(
            verbose=verbose,
            event_col=spec['event_col'],
            direction=spec['direction'],
            event_type=spec.get('event_type', 'orb'),
        )
        signal_results[key] = result

        if not result['trades'].empty:
            trades = result['trades'].copy()
            trades['signal_type'] = spec['event_col']
            trades['direction'] = spec['direction']
            all_trades.append(trades)

    # Merge all trades
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        combined_trades = combined_trades.sort_values('entry_time').reset_index(drop=True)
    else:
        combined_trades = pd.DataFrame()

    # Trade-day overlap check
    overlap_stats = {}
    keys = list(signal_results.keys())
    for i, key_a in enumerate(keys):
        trades_a = signal_results[key_a]['trades']
        if trades_a.empty:
            continue
        dates_a = set(pd.to_datetime(trades_a['session_date']).dt.date
                      if trades_a['session_date'].dtype != 'O'
                      else trades_a['session_date'])
        for key_b in keys[i+1:]:
            trades_b = signal_results[key_b]['trades']
            if trades_b.empty:
                continue
            dates_b = set(pd.to_datetime(trades_b['session_date']).dt.date
                          if trades_b['session_date'].dtype != 'O'
                          else trades_b['session_date'])
            shared = dates_a & dates_b
            total = dates_a | dates_b
            pct = len(shared) / len(total) if total else 0
            overlap_stats[f"{key_a} vs {key_b}"] = {
                'shared_days': len(shared),
                'total_days': len(total),
                'overlap_pct': pct,
            }
            print(f"\n[OVERLAP] {key_a} vs {key_b}: "
                  f"{len(shared)}/{len(total)} days ({pct:.1%})")

    # Portfolio-level metrics
    # Use the full daily_pnl series from each signal result (which has zero-pnl days
    # for every trading day iterated, not just days with completed trades). This gives
    # a correct all-days Sharpe that accounts for idle days.
    portfolio_metrics = {}
    if not combined_trades.empty:
        # Combine all-day daily_pnl series from each signal
        daily_series_list = []
        for key, res in signal_results.items():
            dpnl = res.get('daily_pnl')
            if dpnl is not None and len(dpnl) > 0:
                daily_series_list.append(dpnl.rename(key))

        if daily_series_list:
            # Outer join so union of all days is covered; NaN → 0 (no trade that day)
            portfolio_daily = pd.concat(daily_series_list, axis=1).fillna(0).sum(axis=1)
            portfolio_daily.index = pd.to_datetime(portfolio_daily.index)
        else:
            # Fallback: trade-days-only (inflated but better than nothing)
            portfolio_daily = combined_trades.groupby('session_date')['pnl_dollars'].sum()
            portfolio_daily.index = pd.to_datetime(portfolio_daily.index)

        avg_daily = portfolio_daily.mean()
        std_daily = portfolio_daily.std()
        n_all_days = len(portfolio_daily)
        # Correct years from actual calendar span, not trade-day count
        span_days = (portfolio_daily.index.max() - portfolio_daily.index.min()).days
        years = max(span_days / 365.25, 0.5)
        sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0
        total_pnl = combined_trades['pnl_dollars'].sum()
        n_trades = len(combined_trades)
        trades_per_yr = n_trades / years if years > 0 else 0
        win_rate = (combined_trades['pnl_dollars'] > 0).mean()
        n_trade_days = combined_trades['session_date'].nunique()

        portfolio_metrics = {
            'n_trades': n_trades,
            'trades_per_yr': trades_per_yr,
            'total_pnl': total_pnl,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'n_trade_days': n_trade_days,
            'n_all_days': n_all_days,
            'years': years,
        }

        print(f"\n{'='*72}")
        print("PORTFOLIO SUMMARY")
        print(f"{'='*72}")
        print(f"  Total trades:     {n_trades}")
        print(f"  Trades/yr:        {trades_per_yr:.1f}  (span={years:.1f} yrs)")
        print(f"  Total P&L:        ${total_pnl:>+,.0f}")
        print(f"  Win rate:         {win_rate:.1%}")
        print(f"  Daily Sharpe:     {sharpe:.2f}  (all {n_all_days} days incl. idle)")

    return {
        'signal_results': signal_results,
        'combined_trades': combined_trades,
        'portfolio_metrics': portfolio_metrics,
        'overlap_stats': overlap_stats,
    }


# =====================================================================
# 8. MASTER PIPELINE
# =====================================================================

def run_full_step7(verbose: bool = True, use_regime_filter: bool = False,
                   event_col: str = 'event_orb_long', direction: str = 'long',
                   event_type: str = 'orb') -> dict:
    """
    One-call entry point for Step 7 bar-by-bar backtest.

    Parameters
    ----------
    event_col : str
        Column name for event signal (e.g. 'event_orb_long', 'event_orb_short')
    direction : str
        'long' or 'short'
    event_type : str
        Event type for feature extraction (e.g. 'orb', 'gap', 'exhaustion')
    """
    from research_utils.feature_engineering import load_ohlcv, build_features, add_trend_regime
    from research.event_definitions import detect_all_events, add_session_columns
    from research.strategy_construction import generate_oos_predictions
    from research.config import REGIME_EMA_PERIOD

    DATA_PATH = os.path.join(os.path.dirname(__file__),
                             '..', 'nq_continuous_5m_converted.csv')

    print("=" * 72)
    print("STEP 7 — BACKTESTING & VALIDATION")
    print("=" * 72)
    print()

    # ── Load & prepare ──
    print("[LOAD] Loading data...")
    df = load_ohlcv(DATA_PATH)
    df = df[df.index >= '2019-01-01'].copy()
    print(f"  {len(df)} bars ({df.index.min()} to {df.index.max()})")

    print("[FEATURES] Building features...")
    df = build_features(df, add_targets_flag=False)
    df = add_session_columns(df)

    if use_regime_filter:
        print("[REGIME] Adding trend regime filter...")
        df = add_trend_regime(df, ema_period=REGIME_EMA_PERIOD)
        regime_days = df.groupby(df.index.date)['regime_long_allowed'].first()
        print(f"  {regime_days.mean()*100:.1f}% of days are bull regime (trade allowed)")

    print("[EVENTS] Detecting events...")
    df = detect_all_events(df)

    # ── Build model dataset ──
    print()
    label = event_col.replace('event_', '').replace('_', ' ').title()
    print(f"[DATASET] Building {label} model dataset...")
    dataset = build_model_dataset(df, event_col, direction,
                                   event_type=event_type)

    # ── P2-B: Split into WFO pool and holdout BEFORE model training ──
    HOLDOUT_START_DATE = pd.Timestamp('2025-07-01')
    wfo_dataset = dataset[dataset.index < HOLDOUT_START_DATE].copy()
    holdout_dataset = dataset[dataset.index >= HOLDOUT_START_DATE].copy()
    print(f"[SPLIT] WFO pool: {len(wfo_dataset)} events  |  "
          f"Holdout (>= {HOLDOUT_START_DATE.date()}): {len(holdout_dataset)} events")

    # ── Generate OOS predictions (P4-B: per-fold feature selection) ──
    print("[MODEL] Generating walk-forward OOS predictions (WFO pool only)...")
    print("  (feature selection runs per-fold on training data only)")
    oos_df, feature_cols = generate_oos_predictions(
        wfo_dataset, feature_cols=None, model_type='logistic',
        min_train_events=200, test_months=6, embargo_days=5,
        verbose=verbose,
    )
    print(f"  Production feature set: {len(feature_cols)} features")

    # ── P2-B: Generate holdout predictions with production model ──
    holdout_oos = pd.DataFrame()
    if len(holdout_dataset) > 0:
        print(f"[HOLDOUT-MODEL] Generating holdout predictions "
              f"({len(holdout_dataset)} events)...")

        # Prepare labels for production model training and holdout scoring
        wfo_prepared, wfo_y = prepare_labels(
            wfo_dataset, label_col='barrier_label',
            return_col='barrier_return_pts', timeout_treatment='proportional')
        holdout_prepared, holdout_y = prepare_labels(
            holdout_dataset, label_col='barrier_label',
            return_col='barrier_return_pts', timeout_treatment='proportional')

        # Train production model on ALL WFO data
        X_wfo_all = wfo_prepared[feature_cols]
        production_model = train_model(X_wfo_all, wfo_y, model_type='logistic')

        # Raw probabilities for holdout
        X_holdout = holdout_prepared[feature_cols]
        holdout_raw_probs = predict_proba(
            production_model, X_holdout, model_type='logistic')

        # Fit calibrator on WFO OOS results — holdout never touches calibration
        wfo_oos_raw = oos_df['raw_prob'].values
        _, wfo_oos_y = prepare_labels(
            oos_df, label_col='barrier_label',
            return_col='barrier_return_pts', timeout_treatment='proportional')
        final_calibrator = calibrate_probabilities(
            wfo_oos_raw, wfo_oos_y.values, method='platt')
        holdout_cal_probs = apply_calibration(
            final_calibrator, holdout_raw_probs, method='platt')

        # Build holdout predictions DataFrame (compatible with oos_df)
        holdout_oos = holdout_prepared.copy()
        holdout_oos['raw_prob'] = holdout_raw_probs
        holdout_oos['calibrated_prob'] = holdout_cal_probs

        # Report holdout model-level metrics
        from sklearn.metrics import roc_auc_score, brier_score_loss
        try:
            h_auc = roc_auc_score(holdout_y.values, holdout_cal_probs)
            h_brier = brier_score_loss(holdout_y.values, holdout_cal_probs)
            print(f"  [HOLDOUT] AUC: {h_auc:.4f}, Brier: {h_brier:.4f}, "
                  f"mean P(win): {holdout_cal_probs.mean():.3f}")
        except Exception as e:
            print(f"  [HOLDOUT] Metrics error: {e}")

        # Combine WFO OOS + holdout for full backtest
        oos_combined = pd.concat([oos_df, holdout_oos], axis=0).sort_index()
        print(f"  Combined predictions: {len(oos_combined)} events")
    else:
        print("[HOLDOUT] No holdout events (data ends before holdout date)")
        oos_combined = oos_df

    # Ensure needed columns exist
    if 'session_date' not in oos_combined.columns:
        oos_combined['session_date'] = pd.to_datetime(oos_combined.index).date

    # ── Configure backtest ──
    config = BacktestConfig(threshold=CANONICAL_THRESHOLD, direction=direction)

    # ── Run bar-by-bar backtest ──
    print()
    print(f"[BACKTEST] Running bar-by-bar simulation "
          f"(threshold={config.threshold}, "
          f"slip={config.slippage_pts} pts, "
          f"comm=${config.commission_per_rt})...")
    bt_result = run_bar_by_bar_backtest(df, oos_combined, config, verbose=verbose)

    trades = bt_result['trades']
    if trades.empty:
        print("  [ERROR] No trades generated. Check threshold/risk settings.")
        return bt_result

    # ── Performance metrics ──
    # Use strategy_construction's metrics (compatible format)
    sim_result = {
        'trades': trades,
        'daily_pnl': bt_result['daily_pnl'],
        'final_equity': bt_result['final_equity'],
        'config': StrategyConfig(
            account_size=config.account_size,
            threshold=config.threshold,
        ),
    }
    metrics = compute_performance_metrics(sim_result)
    print_performance_report(metrics, title=f"{label} Backtest (th={config.threshold}, dir={direction}, with costs)")

    # ── Cost analysis ──
    print_cost_analysis(trades, config)

    # ── Monte Carlo ──
    print("[MC] Running Monte Carlo (10,000 paths)...")
    mc = monte_carlo_drawdown(trades, starting_equity=config.account_size)
    print_monte_carlo_report(mc, StrategyConfig(account_size=config.account_size))

    # ── Rolling stability ──
    print("[ROLLING] Computing 6-month rolling windows...")
    rolling = rolling_stability(trades, window_months=6)
    print_rolling_stability(rolling)

    # ── P2-B: Holdout test using canonical HOLDOUT_START_DATE ──
    holdout_start_str = HOLDOUT_START_DATE.strftime('%Y-%m-%d')
    print(f"[HOLDOUT] Testing final period from {holdout_start_str} (canonical)...")
    holdout = holdout_test(trades, holdout_start_str, config)
    print_holdout_report(holdout)

    # ── Final verdict ──
    print_step7_verdict(metrics, mc, holdout, rolling)

    # ── Monthly P&L table ──
    if not trades.empty:
        print("  Monthly P&L (with costs):")
        trades_m = trades.copy()
        trades_m['month'] = pd.to_datetime(trades_m['session_date']).dt.to_period('M')
        monthly = trades_m.groupby('month').agg(
            n=('pnl_dollars', 'count'),
            pnl=('pnl_dollars', 'sum'),
            wr=('barrier_label', lambda x: (x > 0).mean()),
            cost=('commission', 'sum'),
        )
        print(f"  {'Month':>10}  {'Trades':>6}  {'P&L':>10}  {'WR':>5}  {'Costs':>7}")
        print(f"  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*5}  {'-'*7}")
        for period, row in monthly.iterrows():
            print(f"  {str(period):>10}  {row['n']:>6.0f}  "
                  f"${row['pnl']:>+9,.0f}  {row['wr']:>4.0%}  "
                  f"${row['cost']:>+6,.0f}")
        print()

    print("[DONE] Step 7 complete.")
    print()

    results = {
        'trades': trades,
        'metrics': metrics,
        'monte_carlo': mc,
        'rolling': rolling,
        'holdout': holdout,
        'config': config,
        'oos_df': oos_df,
        'holdout_oos': holdout_oos,
        'oos_combined': oos_combined,
        'oos_predictions': oos_combined,  # alias for portfolio wrapper
        'daily_pnl': bt_result['daily_pnl'],  # full all-days series for portfolio Sharpe
    }

    # P5-E: Save computed results to disk
    from research_utils.utils import save_pipeline_results
    save_metrics = {k: v for k, v in metrics.items()
                    if not isinstance(v, (pd.DataFrame, pd.Series))}
    save_metrics['holdout'] = holdout
    save_pipeline_results(save_metrics, 'step7_backtest')

    return results


# =====================================================================
# STANDALONE EXECUTION
# =====================================================================

if __name__ == '__main__':
    results = run_full_step7(verbose=True)
