"""
Step 6 — Strategy Construction
===============================

Transforms Step 5's walk-forward model outputs into a complete tradeable
strategy with position sizing, risk management, and Monte Carlo robustness.

Usage
-----
    from strategy_construction import run_full_step6
    results = run_full_step6(df)

Requires: numpy, pandas, scikit-learn (via model_design).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ── Project imports ──────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.config import CANONICAL_THRESHOLD
from research.event_features import build_model_dataset
from research.model_design import (
    select_features, build_walk_forward_splits, prepare_labels,
    train_model, predict_proba, calibrate_probabilities,
    apply_calibration, get_feature_importance,
)


# =====================================================================
# 1. STRATEGY CONFIGURATION
# =====================================================================

@dataclass
class StrategyConfig:
    """All strategy parameters in one place."""

    # ── Account ──
    account_size: float = 50_000.0
    point_value: float = 20.0          # NQ: $20 per point

    # ── Risk per trade ──
    risk_per_trade_pct: float = 0.015  # 1.5% of account
    max_contracts: int = 2             # hard cap

    # ── Confidence-based sizing ──
    threshold: float = CANONICAL_THRESHOLD  # min P(win) to trade
    size_tiers: list = field(default_factory=lambda: [
        # (min_prob, max_prob, multiplier)
        (0.00, CANONICAL_THRESHOLD, 0.0),  # NO TRADE
        (CANONICAL_THRESHOLD, 0.65, 0.50), # reduced size
        (0.65, 0.75, 0.75),               # moderate
        (0.75, 1.00, 1.00),               # full size
    ])

    # ── Daily risk limits ──
    max_daily_loss: float = -1_000.0   # stop trading for the day
    max_daily_trades: int = 3
    max_concurrent: int = 1            # intraday, no stacking
    consec_loss_pause: int = 3         # skip next signal after N losses

    # ── Weekly / drawdown limits ──
    max_weekly_loss: float = -1_500.0  # reduce to 50% size
    max_drawdown: float = -2_500.0     # circuit breaker
    recovery_days: int = 5             # days before resuming after breaker
    recovery_size_mult: float = 0.50   # size during recovery

    # ── Barrier defaults (from Step 2) ──
    sl_atr_mult: float = 1.0
    tp_atr_mult: float = 1.5
    max_holding_bars: int = 48


# =====================================================================
# 2. POSITION SIZING
# =====================================================================

def compute_position_size(config: StrategyConfig, prob: float,
                          sl_distance_pts: float,
                          equity: float) -> int:
    """
    ATR-based position size scaled by model confidence.

    Returns number of contracts (0 = no trade).
    """
    # Find confidence tier
    multiplier = 0.0
    for lo, hi, mult in config.size_tiers:
        if lo <= prob < hi:
            multiplier = mult
            break
    if prob >= config.size_tiers[-1][0]:
        multiplier = config.size_tiers[-1][2]

    if multiplier == 0.0:
        return 0

    # Base risk budget
    risk_budget = equity * config.risk_per_trade_pct * multiplier

    # Contracts from SL distance
    if sl_distance_pts <= 0:
        return 0
    dollar_risk_per_contract = sl_distance_pts * config.point_value
    contracts = int(risk_budget / dollar_risk_per_contract)

    # Clamp
    contracts = max(0, min(contracts, config.max_contracts))
    return contracts


def get_size_multiplier(config: StrategyConfig, prob: float) -> float:
    """Return the sizing multiplier for a given probability."""
    for lo, hi, mult in config.size_tiers:
        if lo <= prob < hi:
            return mult
    return config.size_tiers[-1][2]


# =====================================================================
# 3. EVENT-LEVEL SIMULATION
# =====================================================================

def simulate_strategy(oos_events: pd.DataFrame,
                      config: StrategyConfig,
                      prob_col: str = 'calibrated_prob',
                      verbose: bool = True) -> dict:
    """
    Event-level P&L simulation with full risk management.

    Parameters
    ----------
    oos_events : DataFrame
        Must contain: calibrated_prob, barrier_return_pts, sl_distance_pts,
        barrier_label, exit_type, event_time (index or column),
        session_date (for daily tracking).
    config : StrategyConfig
    prob_col : str

    Returns
    -------
    dict with keys: trades, daily_pnl, equity_curve, metrics
    """
    events = oos_events.sort_index().copy()

    # Ensure we have session_date for daily tracking
    if 'session_date' not in events.columns:
        events['session_date'] = pd.to_datetime(events.index).date

    trades = []
    equity = config.account_size
    peak_equity = equity
    daily_pnl = {}       # date -> cumulative P&L for the day
    daily_trade_count = {}
    consec_losses = 0
    circuit_breaker_until = None
    weekly_reduced = False

    for idx, row in events.iterrows():
        event_date = row['session_date'] if 'session_date' in row.index else idx.date()
        prob = row.get(prob_col, 0.0)

        # ── Initialize daily tracking ──
        if event_date not in daily_pnl:
            daily_pnl[event_date] = 0.0
            daily_trade_count[event_date] = 0

            # Weekly reset check (Monday)
            if hasattr(event_date, 'weekday') and event_date.weekday() == 0:
                weekly_reduced = False

        # ── Risk checks ──

        # Circuit breaker
        if circuit_breaker_until is not None:
            if event_date < circuit_breaker_until:
                continue
            else:
                circuit_breaker_until = None  # resume

        # Drawdown check
        drawdown = equity - peak_equity
        if drawdown <= config.max_drawdown:
            if verbose:
                print(f"  [CIRCUIT BREAKER] DD={drawdown:.0f} on {event_date}")
            # Set breaker for recovery_days
            try:
                breaker_date = pd.Timestamp(event_date) + pd.Timedelta(
                    days=int(config.recovery_days * 1.5))  # calendar days
                circuit_breaker_until = breaker_date.date()
            except Exception:
                circuit_breaker_until = None
            continue

        # Daily loss limit
        if daily_pnl[event_date] <= config.max_daily_loss:
            continue

        # Daily trade count
        if daily_trade_count[event_date] >= config.max_daily_trades:
            continue

        # Consecutive loss pause
        if consec_losses >= config.consec_loss_pause:
            consec_losses = 0  # reset after skipping one
            continue

        # ── Threshold check ──
        if prob < config.threshold:
            continue

        # ── Position sizing ──
        sl_dist = abs(row.get('sl_distance_pts', 0))
        if sl_dist <= 0:
            continue

        # Check weekly reduction
        size_equity = equity
        extra_mult = 1.0
        if weekly_reduced:
            extra_mult = 0.5

        # Recovery mode
        if circuit_breaker_until is not None:
            extra_mult *= config.recovery_size_mult

        contracts = compute_position_size(config, prob, sl_dist, size_equity)
        if extra_mult < 1.0:
            contracts = max(1, int(contracts * extra_mult))
        if contracts == 0:
            continue

        # ── Compute P&L ──
        return_pts = row['barrier_return_pts']
        pnl_dollars = return_pts * contracts * config.point_value

        # Update state
        equity += pnl_dollars
        peak_equity = max(peak_equity, equity)
        daily_pnl[event_date] += pnl_dollars
        daily_trade_count[event_date] += 1

        # Consecutive loss tracking
        if pnl_dollars < 0:
            consec_losses += 1
        else:
            consec_losses = 0

        # Weekly loss check
        # Sum this week's P&L
        week_start = event_date
        if hasattr(event_date, 'weekday'):
            days_since_monday = event_date.weekday()
            week_start = event_date - pd.Timedelta(days=days_since_monday)
        week_pnl = sum(v for d, v in daily_pnl.items()
                       if d >= week_start if hasattr(d, '__ge__'))
        if week_pnl <= config.max_weekly_loss:
            weekly_reduced = True

        trades.append({
            'event_time': idx,
            'session_date': event_date,
            'prob': prob,
            'size_mult': get_size_multiplier(config, prob),
            'contracts': contracts,
            'sl_distance_pts': sl_dist,
            'return_pts': return_pts,
            'pnl_dollars': pnl_dollars,
            'equity_after': equity,
            'drawdown': equity - peak_equity,
            'exit_type': row.get('exit_type', 'unknown'),
            'barrier_label': row.get('barrier_label', 0),
        })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    daily_df = pd.Series(daily_pnl, name='daily_pnl').sort_index()

    return {
        'trades': trades_df,
        'daily_pnl': daily_df,
        'final_equity': equity,
        'config': config,
    }


# =====================================================================
# 4. PERFORMANCE METRICS
# =====================================================================

def compute_performance_metrics(sim_result: dict,
                                 years: float = None) -> dict:
    """
    Comprehensive performance analytics from simulation output.
    """
    trades = sim_result['trades']
    config = sim_result['config']
    daily_pnl = sim_result['daily_pnl']

    if trades.empty:
        return {'n_trades': 0, 'error': 'No trades'}

    n = len(trades)
    winners = trades[trades['pnl_dollars'] > 0]
    losers = trades[trades['pnl_dollars'] < 0]
    scratches = trades[trades['pnl_dollars'] == 0]

    # ── Time span ──
    if years is None:
        dates = trades['session_date']
        span_days = (pd.Timestamp(dates.max()) - pd.Timestamp(dates.min())).days
        years = max(span_days / 365.25, 0.5)

    # ── Basic stats ──
    total_pnl = trades['pnl_dollars'].sum()
    avg_pnl = trades['pnl_dollars'].mean()
    avg_winner = winners['pnl_dollars'].mean() if len(winners) > 0 else 0
    avg_loser = losers['pnl_dollars'].mean() if len(losers) > 0 else 0
    win_rate = len(winners) / n if n > 0 else 0

    gross_profit = winners['pnl_dollars'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl_dollars'].sum()) if len(losers) > 0 else 0.001
    profit_factor = gross_profit / gross_loss

    # ── Points-based stats ──
    total_pts = trades['return_pts'].sum()
    avg_pts = trades['return_pts'].mean()

    # ── Drawdown ──
    equity_curve = trades['equity_after'].values
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - peak
    max_dd = drawdowns.min()
    max_dd_pct = max_dd / config.account_size * 100

    # Find drawdown recovery time
    dd_recovery_days = 0
    if max_dd < 0:
        dd_idx = np.argmin(drawdowns)
        recovery_idx = np.where(equity_curve[dd_idx:] >= peak[dd_idx])[0]
        if len(recovery_idx) > 0:
            dd_recovery_days = recovery_idx[0]
        else:
            dd_recovery_days = n - dd_idx  # never recovered

    # ── Streaks ──
    wins_losses = (trades['pnl_dollars'] > 0).astype(int).values
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    current_type = None
    for wl in wins_losses:
        if wl == current_type:
            current_streak += 1
        else:
            current_type = wl
            current_streak = 1
        if wl == 1:
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)

    # ── Sharpe & Sortino (from daily returns) ──
    daily_returns = daily_pnl[daily_pnl != 0]  # only trading days
    if len(daily_returns) > 5:
        # Annualize: ~252 trading days
        mean_daily = daily_returns.mean()
        std_daily = daily_returns.std()
        sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0

        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() if len(downside) > 2 else std_daily
        sortino = (mean_daily / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    # ── Calmar ratio ──
    annual_return = total_pnl / years
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # ── Monthly stats ──
    trades_with_month = trades.copy()
    trades_with_month['month'] = pd.to_datetime(
        trades_with_month['session_date']).dt.to_period('M')
    monthly_pnl = trades_with_month.groupby('month')['pnl_dollars'].sum()
    monthly_win_rate = (monthly_pnl > 0).mean() if len(monthly_pnl) > 0 else 0

    # ── Trades per year ──
    tpy = n / years

    return {
        # Volume
        'n_trades': n,
        'n_winners': len(winners),
        'n_losers': len(losers),
        'n_scratches': len(scratches),
        'trades_per_year': tpy,
        'years': years,

        # P&L
        'total_pnl': total_pnl,
        'total_pts': total_pts,
        'avg_pnl_trade': avg_pnl,
        'avg_pts_trade': avg_pts,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,

        # Ratios
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'payoff_ratio': abs(avg_winner / avg_loser) if avg_loser != 0 else 0,

        # Risk
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd_pct,
        'dd_recovery_trades': dd_recovery_days,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,

        # Risk-adjusted
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'annual_return': annual_return,

        # Monthly
        'monthly_win_rate': monthly_win_rate,
        'n_months': len(monthly_pnl),
        'best_month': monthly_pnl.max() if len(monthly_pnl) > 0 else 0,
        'worst_month': monthly_pnl.min() if len(monthly_pnl) > 0 else 0,
    }


# =====================================================================
# 5. MONTE CARLO SIMULATION
# =====================================================================

def monte_carlo_drawdown(trades_df: pd.DataFrame,
                          starting_equity: float = 50_000.0,
                          n_simulations: int = 10_000,
                          seed: int = 42) -> dict:
    """
    Randomize trade order to build drawdown confidence intervals.

    Returns distribution of max drawdowns and final equity.
    """
    if trades_df.empty:
        return {'error': 'No trades'}

    rng = np.random.RandomState(seed)
    pnl_array = trades_df['pnl_dollars'].values
    n = len(pnl_array)

    max_drawdowns = np.zeros(n_simulations)
    final_equities = np.zeros(n_simulations)

    for i in range(n_simulations):
        shuffled = rng.permutation(pnl_array)
        equity_curve = starting_equity + np.cumsum(shuffled)
        peak = np.maximum.accumulate(equity_curve)
        dd = equity_curve - peak
        max_drawdowns[i] = dd.min()
        final_equities[i] = equity_curve[-1]

    return {
        'max_dd_mean': np.mean(max_drawdowns),
        'max_dd_median': np.median(max_drawdowns),
        'max_dd_p5': np.percentile(max_drawdowns, 5),
        'max_dd_p1': np.percentile(max_drawdowns, 1),
        'max_dd_worst': np.min(max_drawdowns),
        'final_eq_mean': np.mean(final_equities),
        'final_eq_median': np.median(final_equities),
        'final_eq_p5': np.percentile(final_equities, 5),
        'prob_profitable': np.mean(final_equities > starting_equity),
        'prob_survive_5pct': np.mean(max_drawdowns > -starting_equity * 0.05),
        'prob_survive_7pct': np.mean(max_drawdowns > -starting_equity * 0.07),
        'n_simulations': n_simulations,
        'n_trades': n,
    }


# =====================================================================
# 6. REPORTING
# =====================================================================

def print_performance_report(metrics: dict, title: str = "Strategy"):
    """Print formatted performance report."""
    print()
    print("=" * 72)
    print(f"  {title} — Performance Report")
    print("=" * 72)

    m = metrics
    if m.get('n_trades', 0) == 0:
        print("  No trades to report.")
        return

    print()
    print(f"  Period: {m['years']:.1f} years, {m['n_trades']} trades "
          f"({m['trades_per_year']:.1f}/yr)")
    print()

    print("  +-- P&L SUMMARY ------------------------------------------+")
    print(f"  |  Total P&L:        ${m['total_pnl']:>+10,.0f}  "
          f"({m['total_pts']:>+8.1f} pts)    |")
    print(f"  |  Per trade:         ${m['avg_pnl_trade']:>+10,.0f}  "
          f"({m['avg_pts_trade']:>+8.1f} pts)    |")
    print(f"  |  Annual return:     ${m['annual_return']:>+10,.0f}          "
          f"           |")
    print(f"  |  Avg winner:        ${m['avg_winner']:>+10,.0f}              "
          f"       |")
    print(f"  |  Avg loser:         ${m['avg_loser']:>+10,.0f}              "
          f"       |")
    print("  +----------------------------------------------------------+")
    print()

    print("  +-- RATIOS -----------------------------------------------+")
    wr = m['win_rate']
    pf = m['profit_factor']
    pr = m['payoff_ratio']
    sh = m['sharpe']
    so = m['sortino']
    ca = m['calmar']
    print(f"  |  Win rate:      {wr:>6.1%}                                 |")
    print(f"  |  Profit factor: {pf:>6.2f}x                                |")
    print(f"  |  Payoff ratio:  {pr:>6.2f}x  (avg W / avg L)              |")
    print(f"  |  Sharpe:        {sh:>6.2f}   (annualized)                  |")
    print(f"  |  Sortino:       {so:>6.2f}   (annualized)                  |")
    print(f"  |  Calmar:        {ca:>6.2f}   (return / max DD)             |")
    print("  +----------------------------------------------------------+")
    print()

    dd = m['max_drawdown']
    ddp = m['max_drawdown_pct']
    wst = m['max_win_streak']
    lst = m['max_loss_streak']
    mwr = m['monthly_win_rate']
    print("  +-- RISK -------------------------------------------------+")
    print(f"  |  Max drawdown:  ${dd:>+9,.0f}  ({ddp:>+5.1f}%)               |")
    print(f"  |  DD recovery:   {m['dd_recovery_trades']:>4d} trades                          |")
    print(f"  |  Win streak:    {wst:>4d}    Loss streak: {lst:>4d}             |")
    print(f"  |  Monthly WR:    {mwr:>5.1%}  ({m['n_months']} months)               |")
    print(f"  |  Best month:    ${m['best_month']:>+9,.0f}                        |")
    print(f"  |  Worst month:   ${m['worst_month']:>+9,.0f}                        |")
    print("  +----------------------------------------------------------+")
    print()


def print_monte_carlo_report(mc: dict, config: StrategyConfig):
    """Print Monte Carlo simulation results."""
    print()
    print("  +-- MONTE CARLO ROBUSTNESS --------------------------------+")
    print(f"  |  Simulations:     {mc['n_simulations']:>7,}                          |")
    print(f"  |  Trades shuffled: {mc['n_trades']:>7,}                          |")
    print(f"  |                                                          |")
    print(f"  |  Max DD mean:     ${mc['max_dd_mean']:>+9,.0f}                        |")
    print(f"  |  Max DD median:   ${mc['max_dd_median']:>+9,.0f}                        |")
    print(f"  |  Max DD P5:       ${mc['max_dd_p5']:>+9,.0f}                        |")
    print(f"  |  Max DD P1:       ${mc['max_dd_p1']:>+9,.0f}  (1-in-100)            |")
    print(f"  |  Max DD worst:    ${mc['max_dd_worst']:>+9,.0f}                        |")
    print(f"  |                                                          |")
    eq5 = mc['final_eq_p5']
    print(f"  |  Final equity P5: ${eq5:>+10,.0f}                       |")
    print(f"  |  P(profitable):   {mc['prob_profitable']:>6.1%}                          |")
    surv5 = mc['prob_survive_5pct']
    surv7 = mc['prob_survive_7pct']
    print(f"  |  P(DD < 5%):      {surv5:>6.1%}                          |")
    print(f"  |  P(DD < 7%):      {surv7:>6.1%}                          |")

    # Pass/fail
    max_dd_ok = mc['max_dd_p5'] > -3_500
    print(f"  |                                                          |")
    tag = "PASS" if max_dd_ok else "FAIL"
    sym = "\u2713" if max_dd_ok else "\u2717"
    print(f"  |  MC P95 DD < $3,500?   {sym} {tag}                       |")
    print("  +----------------------------------------------------------+")
    print()


def print_pass_fail(metrics: dict, mc: dict):
    """Print final pass/fail summary against Step 6 criteria."""
    print()
    print("  " + "=" * 56)
    print("    STRATEGY PASS/FAIL CRITERIA")
    print("  " + "=" * 56)

    checks = []

    # 1. Net profit > 0
    ok = metrics['total_pnl'] > 0
    checks.append(ok)
    sym = "\u2713" if ok else "\u2717"
    print(f"  {sym}  Net profit > 0:        ${metrics['total_pnl']:>+,.0f}")

    # 2. Sharpe >= 1.0
    ok = metrics['sharpe'] >= 1.0
    checks.append(ok)
    sym = "\u2713" if ok else "\u2717"
    print(f"  {sym}  Sharpe >= 1.0:         {metrics['sharpe']:.2f}")

    # 3. Max DD < $2,500
    ok = metrics['max_drawdown'] > -2_500
    checks.append(ok)
    sym = "\u2713" if ok else "\u2717"
    print(f"  {sym}  Max DD < $2,500:       ${metrics['max_drawdown']:>+,.0f}")

    # 4. PF >= 1.5
    ok = metrics['profit_factor'] >= 1.5
    checks.append(ok)
    sym = "\u2713" if ok else "\u2717"
    print(f"  {sym}  Profit factor >= 1.5:  {metrics['profit_factor']:.2f}x")

    # 5. Monthly WR >= 55%
    ok = metrics['monthly_win_rate'] >= 0.55
    checks.append(ok)
    sym = "\u2713" if ok else "\u2717"
    print(f"  {sym}  Monthly WR >= 55%:     {metrics['monthly_win_rate']:.1%}")

    # 6. MC P95 DD < $3,500
    if mc and 'max_dd_p5' in mc:
        ok = mc['max_dd_p5'] > -3_500
        checks.append(ok)
        sym = "\u2713" if ok else "\u2717"
        print(f"  {sym}  MC P95 DD < $3,500:   ${mc['max_dd_p5']:>+,.0f}")

    print()
    if all(checks):
        print("  >>> VERDICT: PASS — Strategy is viable for paper trading <<<")
    else:
        n_fail = sum(1 for c in checks if not c)
        print(f"  >>> VERDICT: FAIL — {n_fail} criterion(s) not met <<<")
    print()


# =====================================================================
# 7. THRESHOLD SENSITIVITY
# =====================================================================

def threshold_sensitivity(oos_events: pd.DataFrame,
                          config: StrategyConfig,
                          thresholds: list = None,
                          prob_col: str = 'calibrated_prob') -> pd.DataFrame:
    """
    Run simulation across multiple thresholds to show trade-offs.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.65, 0.02)]

    rows = []
    for th in thresholds:
        cfg = StrategyConfig(
            threshold=th,
            account_size=config.account_size,
            point_value=config.point_value,
            risk_per_trade_pct=config.risk_per_trade_pct,
            max_contracts=config.max_contracts,
            max_daily_loss=config.max_daily_loss,
            max_daily_trades=config.max_daily_trades,
            max_concurrent=config.max_concurrent,
            consec_loss_pause=config.consec_loss_pause,
            max_weekly_loss=config.max_weekly_loss,
            max_drawdown=config.max_drawdown,
            size_tiers=[
                (0.00, th, 0.0),
                (th, min(th + 0.13, 0.55), 0.50),
                (min(th + 0.13, 0.55), min(th + 0.23, 0.65), 0.75),
                (min(th + 0.23, 0.65), 1.00, 1.00),
            ],
        )
        sim = simulate_strategy(oos_events, cfg, prob_col=prob_col,
                                verbose=False)
        m = compute_performance_metrics(sim)
        rows.append({
            'threshold': th,
            'n_trades': m.get('n_trades', 0),
            'tpy': m.get('trades_per_year', 0),
            'total_pnl': m.get('total_pnl', 0),
            'avg_pnl': m.get('avg_pnl_trade', 0),
            'win_rate': m.get('win_rate', 0),
            'pf': m.get('profit_factor', 0),
            'sharpe': m.get('sharpe', 0),
            'max_dd': m.get('max_drawdown', 0),
            'calmar': m.get('calmar', 0),
        })

    return pd.DataFrame(rows)


def print_threshold_sensitivity(sens_df: pd.DataFrame):
    """Print threshold sensitivity table."""
    print()
    print("  Threshold Sensitivity (with risk management applied)")
    print("  " + "-" * 90)
    print(f"  {'Thresh':>6}  {'Trades':>6}  {'TPY':>5}  {'Total P&L':>10}  "
          f"{'$/Trade':>8}  {'WR':>5}  {'PF':>5}  {'Sharpe':>6}  "
          f"{'Max DD':>9}  {'Calmar':>6}")
    print("  " + "-" * 90)

    for _, r in sens_df.iterrows():
        print(f"  {r['threshold']:>6.2f}  {r['n_trades']:>6.0f}  "
              f"{r['tpy']:>5.0f}  ${r['total_pnl']:>+9,.0f}  "
              f"${r['avg_pnl']:>+7,.0f}  {r['win_rate']:>4.0%}  "
              f"{r['pf']:>5.2f}  {r['sharpe']:>+6.2f}  "
              f"${r['max_dd']:>+8,.0f}  {r['calmar']:>+6.2f}")
    print()


# =====================================================================
# 8. GENERATE OOS PREDICTIONS (walk-forward replay)
# =====================================================================

def generate_oos_predictions(dataset: pd.DataFrame,
                              feature_cols: list,
                              model_type: str = 'logistic',
                              min_train_events: int = 200,
                              test_months: int = 6,
                              embargo_days: int = 5,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Re-run walk-forward to produce OOS calibrated probabilities
    for every event that falls in a test fold.

    Returns the dataset augmented with 'raw_prob' and 'calibrated_prob'.
    """
    # Prepare labels
    ds, labels = prepare_labels(dataset, label_col='barrier_label',
                                return_col='barrier_return_pts',
                                timeout_treatment='proportional')

    # Build splits
    splits = build_walk_forward_splits(ds, min_train_events=min_train_events,
                                       test_months=test_months,
                                       embargo_days=embargo_days)

    if verbose:
        print(f"  Walk-forward: {len(splits)} folds, "
              f"{sum(len(s['test_idx']) for s in splits)} total OOS events")

    # Collect OOS predictions with forward-only calibration (P1-B fix)
    all_cal_probs = []
    all_raw_probs = []
    all_indices = []
    all_true_labels = []
    fold_results = []  # store per-fold for forward calibration

    X = ds[feature_cols]
    y = labels

    for fold_i, split in enumerate(splits):
        train_idx = split['train_idx']
        test_idx = split['test_idx']

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[test_idx]

        model = train_model(X_train, y_train, model_type=model_type)
        fold_probs = predict_proba(model, X_test, model_type=model_type)
        fold_labels = y.loc[test_idx].values

        # P1-B: Forward-only calibration — only use folds 0..fold_i-1
        if fold_i == 0:
            # No prior OOS data; calibrate on chronological tail of training set
            n_cal = max(len(train_idx) // 2, 20)
            cal_idx = train_idx[-n_cal:]
            X_cal = X.loc[cal_idx]
            y_cal = y.loc[cal_idx]
            cal_raw = predict_proba(model, X_cal, model_type=model_type)
            calibrator = calibrate_probabilities(cal_raw, y_cal.values, method='isotonic')
            cal_probs_fold = apply_calibration(calibrator, fold_probs, method='isotonic')
        else:
            prior_probs = np.concatenate([r['probs'] for r in fold_results])
            prior_labels = np.concatenate([r['labels'] for r in fold_results])
            calibrator = calibrate_probabilities(prior_probs, prior_labels, method='platt')
            cal_probs_fold = apply_calibration(calibrator, fold_probs, method='platt')

        fold_results.append({
            'probs': fold_probs,
            'cal_probs': cal_probs_fold,
            'labels': fold_labels,
        })
        all_raw_probs.extend(fold_probs)
        all_cal_probs.extend(cal_probs_fold)
        all_indices.extend(test_idx)
        all_true_labels.extend(fold_labels)

    raw_probs = np.array(all_raw_probs)
    cal_probs = np.array(all_cal_probs)

    # Build output
    oos_df = ds.loc[all_indices].copy()
    oos_df['raw_prob'] = raw_probs
    oos_df['calibrated_prob'] = cal_probs

    if verbose:
        print(f"  OOS predictions: {len(oos_df)} events, "
              f"mean P(win) = {cal_probs.mean():.3f}")

    return oos_df


# =====================================================================
# 9. MASTER PIPELINE
# =====================================================================

def run_full_step6(df: pd.DataFrame,
                   threshold: float = CANONICAL_THRESHOLD,
                   verbose: bool = True) -> dict:
    """
    One-call entry point for Step 6.

    1. Build model dataset for ORB Long
    2. Select features + generate walk-forward OOS predictions
    3. Simulate strategy with risk management
    4. Compute performance metrics
    5. Run Monte Carlo
    6. Threshold sensitivity analysis

    Parameters
    ----------
    df : DataFrame with OHLCV + features + events (output of Steps 1-3)
    threshold : float, default 0.42
    verbose : bool

    Returns
    -------
    dict with all results
    """
    from research.event_definitions import detect_all_events, add_session_columns
    from research_utils.feature_engineering import build_features, load_ohlcv

    print("=" * 72)
    print("STEP 6 — STRATEGY CONSTRUCTION — Validation Run")
    print("=" * 72)
    print()

    # ── Load & prepare ──
    print("[LOAD] Preparing data...")
    if 'atr_14' not in df.columns:
        df = build_features(df, add_targets_flag=False)
    if 'session_date' not in df.columns:
        df = add_session_columns(df)
    if 'event_orb_long' not in df.columns:
        df = detect_all_events(df)

    # ── Build model dataset ──
    print("[DATASET] Building ORB Long model dataset...")
    dataset = build_model_dataset(df, 'event_orb_long', 'long',
                                   event_type='orb')

    # ── Feature selection ──
    print("[FEATURES] Selecting features...")
    feature_cols = select_features(dataset, verbose=verbose)

    # ── Generate OOS predictions ──
    print()
    print("[MODEL] Generating walk-forward OOS predictions...")
    oos_df = generate_oos_predictions(
        dataset, feature_cols, model_type='logistic',
        min_train_events=200, test_months=6, embargo_days=5,
        verbose=verbose,
    )

    # Ensure session_date is in oos_df
    if 'session_date' not in oos_df.columns:
        oos_df['session_date'] = pd.to_datetime(oos_df.index).date

    # ── Configure strategy ──
    config = StrategyConfig(threshold=threshold)
    # Adjust size tiers based on chosen threshold
    config.size_tiers = [
        (0.00, threshold, 0.0),
        (threshold, 0.55, 0.50),
        (0.55, 0.65, 0.75),
        (0.65, 1.00, 1.00),
    ]

    # ── Simulate ──
    print()
    print(f"[SIM] Simulating strategy (threshold={threshold:.2f})...")
    sim = simulate_strategy(oos_df, config, verbose=verbose)

    # ── Performance metrics ──
    metrics = compute_performance_metrics(sim)
    print_performance_report(metrics, title=f"ORB Long (threshold={threshold})")

    # ── Monte Carlo ──
    print("[MC] Running Monte Carlo simulation (10,000 paths)...")
    mc = monte_carlo_drawdown(sim['trades'], starting_equity=config.account_size)
    print_monte_carlo_report(mc, config)

    # ── Threshold sensitivity ──
    print("[SENS] Running threshold sensitivity...")
    sens = threshold_sensitivity(oos_df, config)
    print_threshold_sensitivity(sens)

    # ── Pass/fail ──
    print_pass_fail(metrics, mc)

    # ── Trade distribution by month ──
    trades = sim['trades']
    if not trades.empty:
        print("  Monthly P&L:")
        trades_m = trades.copy()
        trades_m['month'] = pd.to_datetime(trades_m['session_date']).dt.to_period('M')
        monthly = trades_m.groupby('month').agg(
            n=('pnl_dollars', 'count'),
            pnl=('pnl_dollars', 'sum'),
            wr=('barrier_label', lambda x: (x > 0).mean()),
        )
        print(f"  {'Month':>10}  {'Trades':>6}  {'P&L':>10}  {'WR':>5}")
        print(f"  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*5}")
        for period, row in monthly.iterrows():
            print(f"  {str(period):>10}  {row['n']:>6.0f}  "
                  f"${row['pnl']:>+9,.0f}  {row['wr']:>4.0%}")
        print()

    print("[DONE] Step 6 complete.")

    return {
        'oos_df': oos_df,
        'sim': sim,
        'metrics': metrics,
        'monte_carlo': mc,
        'threshold_sensitivity': sens,
        'config': config,
    }


# =====================================================================
# STANDALONE VALIDATION
# =====================================================================

if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from research_utils.feature_engineering import load_ohlcv, build_features
    from research.event_definitions import detect_all_events, add_session_columns

    DATA_PATH = os.path.join(os.path.dirname(__file__),
                             '..', 'nq_continuous_5m_converted.csv')

    print(f"[LOAD] Loading {DATA_PATH}...")
    df = load_ohlcv(DATA_PATH)

    # Trim to 2019+ (consistent with Steps 4-5)
    df = df[df.index >= '2019-01-01'].copy()
    print(f"[LOAD] {len(df)} bars ({df.index.min()} to {df.index.max()})")
    print()

    print("[FEATURES] Building bar-level features...")
    df = build_features(df, add_targets_flag=False)
    df = add_session_columns(df)

    print("[EVENTS] Detecting events...")
    df = detect_all_events(df)

    results = run_full_step6(df, threshold=CANONICAL_THRESHOLD, verbose=True)
