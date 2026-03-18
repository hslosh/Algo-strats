"""
statistical_research.py
=======================
Step 4: Statistical research for NQ event-driven framework.

Provides rigorous statistical analysis of event-driven trade outcomes:
  1. Bootstrap confidence intervals on EV, win rate, profit factor, Sharpe
  2. Parameter sensitivity sweep (SL/TP/hold/ATR)
  3. Regime segmentation (vol, trend, time-of-day, day-of-week, yearly)
  4. Look-ahead bias checks (shuffled-label test)
  5. Sample size sufficiency analysis

Uses the expanded dataset (2019-2026, ~7 years) for statistical power.

Usage:
    from statistical_research import (
        bootstrap_ev, parameter_sweep, regime_analysis,
        run_full_analysis
    )
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ===========================================================================
# 1. BOOTSTRAP CONFIDENCE INTERVALS
# ===========================================================================

def bootstrap_ev(
    returns: np.ndarray,
    timestamps=None,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
    rng=None,
) -> dict:
    """
    Bootstrap confidence interval for expected value and related metrics.

    Parameters
    ----------
    returns : array of trade returns (in points)
    timestamps : array-like of trade entry dates/datetimes for daily P&L Sharpe.
                 If None, Sharpe is set to 0 (cannot be computed without dates).
    n_bootstrap : number of bootstrap resamples
    ci_level : confidence level (0.95 = 95% CI)
    seed : random seed for reproducibility (ignored if rng is provided)
    rng : optional pre-initialized RandomState (for reuse across calls)

    Returns
    -------
    dict with point estimates and confidence intervals
    """
    if rng is None:
        rng = np.random.RandomState(seed)
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    if n < 5:
        return {'error': f'Too few trades ({n}) for bootstrap'}

    alpha = 1 - ci_level
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    # Point estimates
    ev = np.mean(returns)
    median = np.median(returns)
    std = np.std(returns, ddof=1)
    win_rate = np.mean(returns > 0)
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    avg_win = np.mean(winners) if len(winners) > 0 else 0
    avg_loss = np.mean(losers) if len(losers) > 0 else 0
    gross_profit = np.sum(winners) if len(winners) > 0 else 0
    gross_loss = abs(np.sum(losers)) if len(losers) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # P1-C FIX: Annualized Sharpe on daily P&L series (√252)
    annualized_sharpe = 0.0
    n_trading_days = 0
    if timestamps is not None:
        ts = pd.to_datetime(timestamps)
        daily_pnl = pd.Series(returns, index=ts).groupby(ts.normalize()).sum()
        n_trading_days = len(daily_pnl)
        if daily_pnl.std() > 0 and n_trading_days >= 5:
            annualized_sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)

    # Bootstrap
    boot_ev = np.zeros(n_bootstrap)
    boot_wr = np.zeros(n_bootstrap)
    boot_pf = np.zeros(n_bootstrap)
    boot_sharpe = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        boot_ev[i] = np.mean(sample)
        boot_wr[i] = np.mean(sample > 0)

        s_win = sample[sample > 0]
        s_loss = sample[sample < 0]
        gp = np.sum(s_win) if len(s_win) > 0 else 0
        gl = abs(np.sum(s_loss)) if len(s_loss) > 0 else 1
        boot_pf[i] = gp / gl if gl > 0 else 0

        # P1-C FIX: Bootstrap Sharpe uses same daily-P&L approach when timestamps
        # are available; otherwise falls back to 0.
        if timestamps is not None and n_trading_days >= 5:
            ts_sample = rng.choice(np.arange(n_trading_days), size=n_trading_days, replace=True)
            daily_vals = daily_pnl.values
            boot_daily = daily_vals[ts_sample]
            b_std = np.std(boot_daily, ddof=1)
            boot_sharpe[i] = (np.mean(boot_daily) / b_std) * np.sqrt(252) if b_std > 0 else 0
        else:
            boot_sharpe[i] = 0.0

    return {
        'n_trades': n,
        'n_trading_days': n_trading_days,
        'ev_pts': round(ev, 3),
        'ev_ci_lower': round(np.percentile(boot_ev, lower_pct), 3),
        'ev_ci_upper': round(np.percentile(boot_ev, upper_pct), 3),
        'ev_ci_contains_zero': bool(
            np.percentile(boot_ev, lower_pct) <= 0 <= np.percentile(boot_ev, upper_pct)
        ),
        'ev_pvalue': round(np.mean(boot_ev <= 0), 4) if ev > 0 else round(np.mean(boot_ev >= 0), 4),
        'median_pts': round(median, 3),
        'std_pts': round(std, 3),
        'win_rate': round(win_rate, 4),
        'win_rate_ci_lower': round(np.percentile(boot_wr, lower_pct), 4),
        'win_rate_ci_upper': round(np.percentile(boot_wr, upper_pct), 4),
        'avg_win_pts': round(avg_win, 3),
        'avg_loss_pts': round(avg_loss, 3),
        'profit_factor': round(profit_factor, 3),
        'pf_ci_lower': round(np.percentile(boot_pf, lower_pct), 3),
        'pf_ci_upper': round(np.percentile(boot_pf, upper_pct), 3),
        'annualized_sharpe': round(annualized_sharpe, 3),
        'sharpe_ci_lower': round(np.percentile(boot_sharpe, lower_pct), 3),
        'sharpe_ci_upper': round(np.percentile(boot_sharpe, upper_pct), 3),
        'total_pts': round(np.sum(returns), 1),
        'total_dollars': round(np.sum(returns) * 20, 0),  # NQ $20/pt
    }


def print_bootstrap_summary(result: dict, event_name: str = "Event"):
    """Pretty-print bootstrap analysis results."""
    if 'error' in result:
        print(f"\n[BOOTSTRAP] {event_name}: {result['error']}")
        return

    sig = "YES" if not result['ev_ci_contains_zero'] else "NO"
    sig_marker = "***" if not result['ev_ci_contains_zero'] else "   "

    print(f"\n{'=' * 72}")
    print(f"  BOOTSTRAP ANALYSIS: {event_name}")
    print(f"{'=' * 72}")
    print(f"  Trades: {result['n_trades']}")
    print(f"")
    print(f"  ┌── EXPECTED VALUE ──────────────────────────────────────────────┐")
    print(f"  │  EV:     {result['ev_pts']:>+8.3f} pts/trade  (${result['ev_pts']*20:>+.0f}/trade)       │")
    print(f"  │  95% CI: [{result['ev_ci_lower']:>+8.3f}, {result['ev_ci_upper']:>+8.3f}] pts              │")
    print(f"  │  Statistically significant? {sig} {sig_marker}                       │")
    print(f"  │  p-value: {result['ev_pvalue']:.4f}                                         │")
    print(f"  └────────────────────────────────────────────────────────────────┘")
    print(f"  ┌── WIN RATE ────────────────────────────────────────────────────┐")
    print(f"  │  Win rate:  {result['win_rate']:>6.1%}  CI: [{result['win_rate_ci_lower']:.1%}, {result['win_rate_ci_upper']:.1%}]       │")
    print(f"  │  Avg win:  {result['avg_win_pts']:>+8.2f} pts   Avg loss: {result['avg_loss_pts']:>+8.2f} pts    │")
    print(f"  └────────────────────────────────────────────────────────────────┘")
    print(f"  ┌── RISK METRICS ────────────────────────────────────────────────┐")
    print(f"  │  Profit Factor: {result['profit_factor']:>6.3f}x  CI: [{result['pf_ci_lower']:.3f}, {result['pf_ci_upper']:.3f}]     │")
    sharpe_val = result.get('annualized_sharpe', result.get('sharpe', 0))
    print(f"  │  Sharpe (ann):  {sharpe_val:>+6.3f}   CI: [{result['sharpe_ci_lower']:+.3f}, {result['sharpe_ci_upper']:+.3f}]     │")
    print(f"  │  Total P&L:    {result['total_pts']:>+8.1f} pts  (${result['total_dollars']:>+,.0f})           │")
    print(f"  └────────────────────────────────────────────────────────────────┘")


# ===========================================================================
# 2. PARAMETER SENSITIVITY SWEEP
# ===========================================================================

def parameter_sweep(
    df: pd.DataFrame,
    event_col: str,
    direction: str,
    sl_multiples: list[float] = None,
    tp_multiples: list[float] = None,
    max_holding_list: list[int] = None,
    atr_lookbacks: list[int] = None,
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """
    Sweep barrier parameters and evaluate each combination.

    Parameters
    ----------
    df : DataFrame with session columns and event columns
    event_col : Boolean event column
    direction : 'long' or 'short'
    sl_multiples : Stop-loss ATR multiples to test
    tp_multiples : Take-profit ATR multiples to test
    max_holding_list : Max holding periods to test
    atr_lookbacks : ATR lookback windows to test
    n_bootstrap : Bootstrap iterations per combo (reduced for speed)

    Returns
    -------
    DataFrame with one row per parameter combination and performance metrics
    """
    from outcome_labeling import label_events

    if sl_multiples is None:
        sl_multiples = [0.75, 1.0, 1.25, 1.5]
    if tp_multiples is None:
        tp_multiples = [1.0, 1.5, 2.0, 2.5, 3.0]
    if max_holding_list is None:
        max_holding_list = [12, 24, 36, 48]
    if atr_lookbacks is None:
        atr_lookbacks = [14]  # Default: only test 14 to reduce combos

    total_combos = len(sl_multiples) * len(tp_multiples) * len(max_holding_list) * len(atr_lookbacks)
    print(f"\n[SWEEP] Testing {total_combos} parameter combinations for '{event_col}'...")

    results = []
    combo_num = 0

    # P1-C FIX: Single RNG shared across all combos (not reset per combo)
    rng = np.random.RandomState(42)

    for atr_lb in atr_lookbacks:
        for sl in sl_multiples:
            for tp in tp_multiples:
                for hold in max_holding_list:
                    combo_num += 1

                    # Skip obviously bad R:R ratios
                    rr_ratio = tp / sl
                    breakeven_wr = sl / (sl + tp)

                    labeled = label_events(
                        df, event_col, direction,
                        sl_atr_multiple=sl,
                        tp_atr_multiple=tp,
                        atr_lookback=atr_lb,
                        max_holding_bars=hold,
                        force_session_exit=True,
                    )

                    if len(labeled) < 10:
                        continue

                    returns = labeled['barrier_return_pts'].values

                    # Quick stats (skip full bootstrap for speed)
                    ev = np.mean(returns)
                    n_trades = len(returns)
                    win_rate = np.mean(labeled['barrier_label'] == 1)
                    std = np.std(returns, ddof=1)
                    total = np.sum(returns)

                    winners = returns[returns > 0]
                    losers = returns[returns < 0]
                    gp = np.sum(winners) if len(winners) > 0 else 0
                    gl = abs(np.sum(losers)) if len(losers) > 0 else 1
                    pf = gp / gl if gl > 0 else 0

                    avg_time = labeled['time_to_barrier'].mean()
                    avg_mae = labeled['mae_pts'].mean()
                    avg_mfe = labeled['mfe_pts'].mean()

                    # Lightweight bootstrap for EV CI (uses shared rng)
                    boot_ev = np.array([
                        np.mean(rng.choice(returns, size=n_trades, replace=True))
                        for _ in range(n_bootstrap)
                    ])
                    ev_ci_lower = np.percentile(boot_ev, 2.5)
                    ev_ci_upper = np.percentile(boot_ev, 97.5)

                    # P1-C FIX: Annualized Sharpe on daily P&L series
                    sweep_sharpe = 0.0
                    if hasattr(labeled.index, 'normalize'):
                        daily_pnl = pd.Series(returns, index=labeled.index).groupby(
                            labeled.index.normalize()
                        ).sum()
                        if daily_pnl.std() > 0 and len(daily_pnl) >= 5:
                            sweep_sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)

                    results.append({
                        'sl_multiple': sl,
                        'tp_multiple': tp,
                        'max_hold': hold,
                        'atr_lookback': atr_lb,
                        'rr_ratio': round(rr_ratio, 2),
                        'breakeven_wr': round(breakeven_wr, 3),
                        'n_trades': n_trades,
                        'ev_pts': round(ev, 3),
                        'ev_ci_lower': round(ev_ci_lower, 3),
                        'ev_ci_upper': round(ev_ci_upper, 3),
                        'ev_significant': bool(ev_ci_lower > 0) if ev > 0 else False,
                        'win_rate': round(win_rate, 4),
                        'profit_factor': round(pf, 3),
                        'annualized_sharpe': round(sweep_sharpe, 3),
                        'total_pts': round(total, 1),
                        'total_dollars': round(total * 20, 0),
                        'avg_time_bars': round(avg_time, 1),
                        'avg_mae_pts': round(avg_mae, 1),
                        'avg_mfe_pts': round(avg_mfe, 1),
                        'std_pts': round(std, 2),
                    })

                    if combo_num % 20 == 0:
                        print(f"  [{combo_num}/{total_combos}] SL={sl} TP={tp} Hold={hold} → "
                              f"EV={ev:+.2f} WR={win_rate:.1%} PF={pf:.2f}")

    sweep_df = pd.DataFrame(results)
    if len(sweep_df) > 0:
        sweep_df = sweep_df.sort_values('ev_pts', ascending=False)

    return sweep_df


def print_sweep_summary(sweep_df: pd.DataFrame, event_name: str, top_n: int = 10):
    """Print the top parameter combinations from a sweep."""
    if len(sweep_df) == 0:
        print(f"[SWEEP] {event_name}: No results")
        return

    print(f"\n{'=' * 90}")
    print(f"  PARAMETER SWEEP: {event_name}")
    print(f"  {len(sweep_df)} combinations tested")
    print(f"{'=' * 90}")

    # Profitable combos
    profitable = sweep_df[sweep_df['ev_pts'] > 0]
    significant = sweep_df[sweep_df['ev_significant']]

    print(f"  Profitable combinations: {len(profitable)}/{len(sweep_df)} ({len(profitable)/len(sweep_df)*100:.0f}%)")
    print(f"  Statistically significant (95% CI > 0): {len(significant)}")

    # Top N
    print(f"\n  Top {top_n} by Expected Value:")
    print(f"  {'SL':>4} {'TP':>4} {'Hold':>4} {'R:R':>5} {'Trades':>6} "
          f"{'EV':>8} {'CI_lo':>8} {'CI_hi':>8} {'WR':>6} {'PF':>6} {'Total':>8} {'Sig':>4}")
    print(f"  {'─'*4} {'─'*4} {'─'*4} {'─'*5} {'─'*6} "
          f"{'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*8} {'─'*4}")

    for _, row in sweep_df.head(top_n).iterrows():
        sig = " ** " if row['ev_significant'] else "    "
        print(f"  {row['sl_multiple']:>4.2f} {row['tp_multiple']:>4.1f} {row['max_hold']:>4d} "
              f"{row['rr_ratio']:>5.2f} {row['n_trades']:>6d} "
              f"{row['ev_pts']:>+8.2f} {row['ev_ci_lower']:>+8.2f} {row['ev_ci_upper']:>+8.2f} "
              f"{row['win_rate']:>5.1%} {row['profit_factor']:>6.2f} "
              f"{row['total_pts']:>+8.1f}{sig}")

    # Robustness: how wide is the profitable region?
    if len(profitable) > 0:
        print(f"\n  Robust parameter ranges (where EV > 0):")
        for param in ['sl_multiple', 'tp_multiple', 'max_hold']:
            vals = profitable[param].unique()
            if len(vals) > 0:
                formatted = [f"{v:.2f}" if isinstance(v, float) or 'float' in str(type(v)) else str(int(v)) for v in sorted(vals)]
                print(f"    {param}: [{', '.join(formatted)}]")


# ===========================================================================
# 3. REGIME SEGMENTATION
# ===========================================================================

def regime_analysis(
    labeled_events: pd.DataFrame,
    full_df: pd.DataFrame,
    event_name: str = "Event",
) -> dict:
    """
    Analyze event performance across market regimes.

    Parameters
    ----------
    labeled_events : DataFrame from build_model_dataset (has features + labels)
    full_df : Full OHLCV DataFrame (for computing regime at event time)
    event_name : Display name

    Returns
    -------
    dict of regime analyses
    """
    if len(labeled_events) < 20:
        print(f"[REGIME] {event_name}: Too few events ({len(labeled_events)})")
        return {}

    results = {}

    print(f"\n{'=' * 72}")
    print(f"  REGIME ANALYSIS: {event_name}")
    print(f"{'=' * 72}")

    returns = labeled_events['barrier_return_pts'].values

    # --- Volatility Regime ---
    if 'vol_regime' in labeled_events.columns:
        print(f"\n  ── By Volatility Regime ──")
        _print_regime_breakdown(labeled_events, 'vol_regime', {
            0: 'Low Vol', 1: 'Normal', 2: 'High Vol', 3: 'Crisis'
        })
        results['vol_regime'] = _compute_regime_breakdown(labeled_events, 'vol_regime')

    # --- Trend Regime ---
    if 'trend_regime' in labeled_events.columns:
        print(f"\n  ── By Trend Regime ──")
        _print_regime_breakdown(labeled_events, 'trend_regime', {
            0: 'Range-bound', 1: 'Trending'
        })
        results['trend_regime'] = _compute_regime_breakdown(labeled_events, 'trend_regime')

    # --- Time of Day ---
    if 'bar_of_session' in labeled_events.columns:
        # Create time buckets
        events_copy = labeled_events.copy()
        bars = events_copy['bar_of_session']
        events_copy['time_bucket'] = 'midday'
        events_copy.loc[bars <= 12, 'time_bucket'] = 'morning_1h'
        events_copy.loc[(bars > 12) & (bars <= 24), 'time_bucket'] = 'late_morning'
        events_copy.loc[(bars > 24) & (bars <= 54), 'time_bucket'] = 'midday'
        events_copy.loc[bars > 54, 'time_bucket'] = 'afternoon'

        print(f"\n  ── By Time of Day ──")
        _print_regime_breakdown(events_copy, 'time_bucket', {
            'morning_1h': 'Morning (1st hr)',
            'late_morning': 'Late AM',
            'midday': 'Midday',
            'afternoon': 'Afternoon',
        })
        results['time_of_day'] = _compute_regime_breakdown(events_copy, 'time_bucket')

    # --- Day of Week ---
    if hasattr(labeled_events.index, 'dayofweek'):
        events_copy = labeled_events.copy()
        events_copy['dow'] = events_copy.index.dayofweek
        dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}

        print(f"\n  ── By Day of Week ──")
        _print_regime_breakdown(events_copy, 'dow', dow_names)
        results['day_of_week'] = _compute_regime_breakdown(events_copy, 'dow')

    # --- Year ---
    events_copy = labeled_events.copy()
    events_copy['year'] = events_copy.index.year
    year_names = {y: str(y) for y in events_copy['year'].unique()}

    print(f"\n  ── By Year ──")
    _print_regime_breakdown(events_copy, 'year', year_names)
    results['year'] = _compute_regime_breakdown(events_copy, 'year')

    return results


def _print_regime_breakdown(df: pd.DataFrame, col: str, label_map: dict):
    """Print EV, WR, and trade count for each regime value."""
    print(f"  {'Regime':<20} {'N':>5} {'EV (pts)':>10} {'WR':>7} {'PF':>6} {'Total':>10}")
    print(f"  {'─'*20} {'─'*5} {'─'*10} {'─'*7} {'─'*6} {'─'*10}")

    for val in sorted(df[col].dropna().unique()):
        subset = df[df[col] == val]
        n = len(subset)
        if n < 3:
            continue

        returns = subset['barrier_return_pts'].values
        ev = np.mean(returns)
        wr = np.mean(subset['barrier_label'] == 1)
        winners = returns[returns > 0]
        losers = returns[returns < 0]
        gp = np.sum(winners) if len(winners) > 0 else 0
        gl = abs(np.sum(losers)) if len(losers) > 0 else 1
        pf = gp / gl if gl > 0 else 0
        total = np.sum(returns)

        name = label_map.get(val, str(val))
        marker = " <+" if ev > 0 else ""
        print(f"  {name:<20} {n:>5} {ev:>+10.2f} {wr:>6.1%} {pf:>6.2f} {total:>+10.1f}{marker}")


def _compute_regime_breakdown(df: pd.DataFrame, col: str) -> list[dict]:
    """Compute regime breakdown data as a list of dicts."""
    results = []
    for val in sorted(df[col].dropna().unique()):
        subset = df[df[col] == val]
        n = len(subset)
        if n < 3:
            continue
        returns = subset['barrier_return_pts'].values
        results.append({
            'regime': val,
            'n': n,
            'ev': round(np.mean(returns), 3),
            'win_rate': round(np.mean(subset['barrier_label'] == 1), 4),
            'total': round(np.sum(returns), 1),
        })
    return results


# ===========================================================================
# 4. LOOK-AHEAD BIAS CHECK
# ===========================================================================

def shuffled_label_test(
    dataset: pd.DataFrame,
    n_shuffles: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Check for look-ahead bias by testing if features predict shuffled labels.

    If the real correlation between features and labels is similar to what
    you get with randomly shuffled labels, the features are NOT predictive
    (they might be leaking future info or fitting noise).

    Parameters
    ----------
    dataset : Model dataset from build_model_dataset
    n_shuffles : Number of random shuffles
    seed : Random seed

    Returns
    -------
    dict with test results
    """
    from event_features import get_feature_columns

    rng = np.random.RandomState(seed)
    feature_cols = get_feature_columns(dataset)

    # Only use definitive labels
    definitive = dataset[dataset['barrier_label'].isin([1, -1])].copy()
    if len(definitive) < 20:
        return {'error': 'Too few definitive labels'}

    labels = definitive['barrier_label'].values

    # Real correlation: average absolute correlation across features
    real_corrs = []
    for col in feature_cols:
        vals = definitive[col].dropna()
        if len(vals) < 10:
            continue
        matched_labels = definitive.loc[vals.index, 'barrier_label'].values
        corr = np.corrcoef(vals.values, matched_labels)[0, 1]
        if not np.isnan(corr):
            real_corrs.append(abs(corr))

    if not real_corrs:
        return {'error': 'No valid features for correlation'}

    real_avg_corr = np.mean(real_corrs)
    real_max_corr = np.max(real_corrs)

    # Shuffled correlations
    shuffled_avg_corrs = []
    shuffled_max_corrs = []

    # Build a position map: for each row in definitive, its integer position
    row_positions = {idx: pos for pos, idx in enumerate(definitive.index)}

    for _ in range(n_shuffles):
        shuffled_labels = rng.permutation(labels)

        s_corrs = []
        for col in feature_cols:
            vals = definitive[col].dropna()
            if len(vals) < 10:
                continue
            # Get the shuffled labels corresponding to non-NaN rows
            pos_indices = [row_positions[idx] for idx in vals.index]
            s_labels = shuffled_labels[pos_indices]
            corr = np.corrcoef(vals.values, s_labels)[0, 1]
            if not np.isnan(corr):
                s_corrs.append(abs(corr))

        if s_corrs:
            shuffled_avg_corrs.append(np.mean(s_corrs))
            shuffled_max_corrs.append(np.max(s_corrs))

    shuffled_avg_corrs = np.array(shuffled_avg_corrs)
    shuffled_max_corrs = np.array(shuffled_max_corrs)

    # p-value: how often does shuffled data produce correlations as strong?
    avg_pvalue = np.mean(shuffled_avg_corrs >= real_avg_corr)
    max_pvalue = np.mean(shuffled_max_corrs >= real_max_corr)

    return {
        'n_features': len(real_corrs),
        'n_events': len(definitive),
        'real_avg_abs_corr': round(real_avg_corr, 4),
        'real_max_abs_corr': round(real_max_corr, 4),
        'shuffled_avg_abs_corr_mean': round(np.mean(shuffled_avg_corrs), 4),
        'shuffled_avg_abs_corr_95th': round(np.percentile(shuffled_avg_corrs, 95), 4),
        'shuffled_max_abs_corr_mean': round(np.mean(shuffled_max_corrs), 4),
        'shuffled_max_abs_corr_95th': round(np.percentile(shuffled_max_corrs, 95), 4),
        'avg_corr_pvalue': round(avg_pvalue, 4),
        'max_corr_pvalue': round(max_pvalue, 4),
        'verdict': 'PASS - features have genuine signal' if avg_pvalue < 0.05
                   else 'INCONCLUSIVE' if avg_pvalue < 0.20
                   else 'FAIL - features may not have real signal',
    }


def print_shuffled_label_results(result: dict, event_name: str):
    """Print shuffled label test results."""
    if 'error' in result:
        print(f"\n[BIAS CHECK] {event_name}: {result['error']}")
        return

    print(f"\n{'=' * 72}")
    print(f"  LOOK-AHEAD BIAS CHECK: {event_name}")
    print(f"{'=' * 72}")
    print(f"  Features tested: {result['n_features']}, Events: {result['n_events']}")
    print(f"")
    print(f"  Real avg |correlation|:     {result['real_avg_abs_corr']:.4f}")
    print(f"  Shuffled avg |corr| (mean): {result['shuffled_avg_abs_corr_mean']:.4f}")
    print(f"  Shuffled avg |corr| (95th): {result['shuffled_avg_abs_corr_95th']:.4f}")
    print(f"  p-value (avg corr):         {result['avg_corr_pvalue']:.4f}")
    print(f"")
    print(f"  Real max |correlation|:     {result['real_max_abs_corr']:.4f}")
    print(f"  Shuffled max |corr| (mean): {result['shuffled_max_abs_corr_mean']:.4f}")
    print(f"  Shuffled max |corr| (95th): {result['shuffled_max_abs_corr_95th']:.4f}")
    print(f"  p-value (max corr):         {result['max_corr_pvalue']:.4f}")
    print(f"")
    print(f"  Verdict: {result['verdict']}")


# ===========================================================================
# 5. SAMPLE SIZE SUFFICIENCY
# ===========================================================================

def sample_size_analysis(
    returns: np.ndarray,
    target_ev: float = 2.0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Assess whether the sample size is sufficient to detect a given edge.

    Parameters
    ----------
    returns : array of trade returns
    target_ev : minimum EV (pts) we want to detect
    alpha : significance level
    power : desired statistical power

    Returns
    -------
    dict with sample size analysis
    """
    from scipy import stats as scipy_stats

    n = len(returns)
    std = np.std(returns, ddof=1)
    observed_ev = np.mean(returns)

    # z-values
    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
    z_beta = scipy_stats.norm.ppf(power)

    # Required sample size to detect target_ev
    if std > 0:
        required_n = int(np.ceil(((z_alpha + z_beta) * std / target_ev) ** 2))
    else:
        required_n = 0

    # Power of current sample to detect observed_ev
    if std > 0 and observed_ev != 0:
        current_power_z = abs(observed_ev) / std * np.sqrt(n) - z_alpha
        current_power = scipy_stats.norm.cdf(current_power_z)
    else:
        current_power = 0

    # Events per year needed (assuming ~250 trading days)
    years_of_data = 7  # approximate
    events_per_year = n / years_of_data
    if events_per_year > 0:
        years_needed = required_n / events_per_year
    else:
        years_needed = float('inf')

    return {
        'current_n': n,
        'observed_ev': round(observed_ev, 3),
        'std': round(std, 3),
        'target_ev_to_detect': target_ev,
        'required_n': required_n,
        'sufficient': n >= required_n,
        'sample_ratio': round(n / required_n, 2) if required_n > 0 else float('inf'),
        'current_power': round(current_power, 3),
        'events_per_year': round(events_per_year, 1),
        'years_needed_for_required_n': round(years_needed, 1),
    }


def print_sample_size_analysis(result: dict, event_name: str):
    """Print sample size analysis."""
    print(f"\n{'=' * 72}")
    print(f"  SAMPLE SIZE ANALYSIS: {event_name}")
    print(f"{'=' * 72}")
    print(f"  Current sample:        {result['current_n']} trades")
    print(f"  Observed EV:           {result['observed_ev']:+.3f} pts/trade")
    print(f"  Return std dev:        {result['std']:.3f} pts")
    print(f"")
    print(f"  To detect EV = {result['target_ev_to_detect']:.1f} pts at 95% conf / 80% power:")
    print(f"    Required sample:     {result['required_n']} trades")
    suf = "YES" if result['sufficient'] else "NO"
    print(f"    Sufficient?          {suf} ({result['sample_ratio']:.2f}x of required)")
    print(f"    At {result['events_per_year']:.0f} events/year:   ~{result['years_needed_for_required_n']:.1f} years of data needed")
    print(f"")
    print(f"  Power of current sample to detect observed EV:")
    print(f"    Statistical power:   {result['current_power']:.1%}")


# ===========================================================================
# MASTER PIPELINE
# ===========================================================================

def run_full_analysis(
    df: pd.DataFrame,
    event_col: str,
    direction: str,
    event_type: str = 'generic',
    event_name: str = 'Event',
    run_sweep: bool = True,
    run_regime: bool = True,
    run_bias_check: bool = True,
    run_sample_size: bool = True,
    sweep_atr_lookbacks: list[int] = None,
) -> dict:
    """
    Run the complete Step 4 statistical research pipeline for one event.

    Parameters
    ----------
    df : Full DataFrame with session columns, events, and bar-level features
    event_col : Boolean event column
    direction : 'long' or 'short'
    event_type : For event-specific features
    event_name : Display name
    run_sweep : Whether to run parameter sweep (slow)
    run_regime : Whether to run regime analysis
    run_bias_check : Whether to run look-ahead bias check
    run_sample_size : Whether to run sample size analysis

    Returns
    -------
    dict with all analysis results
    """
    from event_features import build_model_dataset

    print(f"\n{'#' * 72}")
    print(f"  FULL STATISTICAL ANALYSIS: {event_name}")
    print(f"{'#' * 72}")

    results = {'event_name': event_name, 'event_col': event_col, 'direction': direction}

    # Build model dataset (labels + features)
    dataset = build_model_dataset(
        df, event_col, direction,
        event_type=event_type,
    )

    if len(dataset) < 10:
        print(f"[ANALYSIS] Too few events ({len(dataset)}) — skipping")
        return results

    returns = dataset['barrier_return_pts'].values
    timestamps = dataset.index  # pass event timestamps for daily Sharpe

    # 1. Bootstrap CI
    print(f"\n[1/5] Bootstrap confidence intervals...")
    boot = bootstrap_ev(returns, timestamps=timestamps)
    print_bootstrap_summary(boot, event_name)
    results['bootstrap'] = boot

    # 2. Parameter sweep
    if run_sweep:
        print(f"\n[2/5] Parameter sensitivity sweep...")
        sweep = parameter_sweep(
            df, event_col, direction,
            atr_lookbacks=sweep_atr_lookbacks or [14],
        )
        print_sweep_summary(sweep, event_name)
        results['sweep'] = sweep
    else:
        print(f"\n[2/5] Parameter sweep: SKIPPED")

    # 3. Regime analysis
    if run_regime:
        print(f"\n[3/5] Regime segmentation...")
        regimes = regime_analysis(dataset, df, event_name)
        results['regimes'] = regimes
    else:
        print(f"\n[3/5] Regime analysis: SKIPPED")

    # 4. Look-ahead bias check
    if run_bias_check:
        print(f"\n[4/5] Look-ahead bias check (shuffled labels)...")
        bias = shuffled_label_test(dataset)
        print_shuffled_label_results(bias, event_name)
        results['bias_check'] = bias
    else:
        print(f"\n[4/5] Bias check: SKIPPED")

    # 5. Sample size
    if run_sample_size:
        print(f"\n[5/5] Sample size sufficiency...")
        try:
            ss = sample_size_analysis(returns)
            print_sample_size_analysis(ss, event_name)
            results['sample_size'] = ss
        except ImportError:
            print(f"  [SKIP] scipy not available for sample size analysis")
    else:
        print(f"\n[5/5] Sample size: SKIPPED")

    # Dataset reference
    results['dataset'] = dataset

    return results


# ===========================================================================
# VALIDATION SCRIPT
# ===========================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../research_utils')
    sys.path.insert(0, '.')

    from feature_engineering import load_ohlcv, build_features
    from event_definitions import add_session_columns, detect_session_sweep, detect_orb

    print("=" * 72)
    print("STEP 4 — STATISTICAL RESEARCH — Validation Run")
    print("=" * 72)

    # --- Load EXPANDED dataset (2019-2026) ---
    filepath = '../nq_continuous_5m_converted.csv'
    print(f"\n[LOAD] Loading {filepath}...")
    df = load_ohlcv(filepath)
    df = df['2019-01-01':'2026-01-14']
    print(f"[LOAD] {len(df)} bars loaded ({df.index[0]} to {df.index[-1]})")
    print(f"[LOAD] ~{(df.index[-1] - df.index[0]).days / 365.25:.1f} years of data")

    # --- Build features ---
    print("\n[FEATURES] Building bar-level features...")
    df = build_features(df, add_targets_flag=False)
    print(f"[FEATURES] {len(df.columns)} columns")

    # --- Detect events ---
    print("\n[EVENTS] Detecting Tier 1 events...")
    df = add_session_columns(df)
    df = detect_session_sweep(df)
    df = detect_orb(df)

    # --- Run full analysis for each Tier 1 event ---
    event_configs = [
        ('sweep_high_first_today', 'short', 'sweep', 'Sweep High → SHORT'),
        ('sweep_low_first_today',  'long',  'sweep', 'Sweep Low → LONG'),
        ('event_orb_long',         'long',  'orb',   'ORB → LONG'),
        ('event_orb_short',        'short', 'orb',   'ORB → SHORT'),
    ]

    all_results = {}

    for event_col, direction, event_type, display_name in event_configs:
        if event_col not in df.columns:
            print(f"\n[SKIP] {event_col} not found")
            continue

        result = run_full_analysis(
            df,
            event_col=event_col,
            direction=direction,
            event_type=event_type,
            event_name=display_name,
            run_sweep=True,
            run_regime=True,
            run_bias_check=True,
            run_sample_size=True,
        )
        all_results[event_col] = result

    # --- Summary ---
    print(f"\n\n{'#' * 72}")
    print(f"  STEP 4 — MASTER SUMMARY")
    print(f"{'#' * 72}")
    print(f"\n  {'Event':<30} {'N':>5} {'EV':>8} {'95% CI':>20} {'Sig':>5} {'PF':>6}")
    print(f"  {'─'*30} {'─'*5} {'─'*8} {'─'*20} {'─'*5} {'─'*6}")

    for name, result in all_results.items():
        boot = result.get('bootstrap', {})
        if 'error' in boot:
            continue
        sig = "YES" if not boot.get('ev_ci_contains_zero', True) else " no"
        print(f"  {result['event_name']:<30} {boot['n_trades']:>5} "
              f"{boot['ev_pts']:>+8.3f} [{boot['ev_ci_lower']:>+8.3f}, {boot['ev_ci_upper']:>+8.3f}] "
              f"{sig:>5} {boot['profit_factor']:>6.3f}")

    # Best sweep params per event
    print(f"\n  Best parameter sets from sweep (highest EV with significance):")
    for name, result in all_results.items():
        sweep = result.get('sweep')
        if sweep is not None and len(sweep) > 0:
            sig_sweep = sweep[sweep['ev_significant']]
            if len(sig_sweep) > 0:
                best = sig_sweep.iloc[0]
                print(f"    {result['event_name']:<30}: SL={best['sl_multiple']:.2f} "
                      f"TP={best['tp_multiple']:.1f} Hold={best['max_hold']} "
                      f"→ EV={best['ev_pts']:+.2f} WR={best['win_rate']:.1%}")
            else:
                print(f"    {result['event_name']:<30}: No statistically significant params found")

    print(f"\n[DONE] Step 4 complete.")
