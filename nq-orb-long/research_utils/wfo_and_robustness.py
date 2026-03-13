"""
wfo_and_robustness.py

Phase 3–4: Walk-forward optimization, Monte Carlo, and robustness testing.
Supports parameter search, walk-forward windows, and stress testing.

Usage:
    # Phase 2: Random parameter search
    python wfo_and_robustness.py --mode random-search --num-sets 100 \
      --start-date 2022-01-01 --end-date 2023-12-31 \
      --data-file data/nq_continuous_5m.csv \
      --output results/random_search_20260122.json
    
    # Phase 3: Walk-forward optimization
    python wfo_and_robustness.py --mode wfo \
      --start-date 2019-01-01 --end-date 2024-12-31 \
      --is-period-months 24 --oos-period-months 6 --step-months 3 \
      --data-file data/nq_continuous_5m.csv \
      --output results/wfo_20260122.json
    
    # Phase 4: Monte Carlo
    python wfo_and_robustness.py --mode monte-carlo \
      --params-file results/best_oos_params.json \
      --data-file data/nq_continuous_5m.csv \
      --num-sims 1000 --output results/monte_carlo_20260122.json
"""

import argparse
import json
import sys
import random
from datetime import datetime, timedelta
import os

from backtest_runner import BacktestRunner
from data_pipeline import DataLoader
from strategy_family import StrategyFactory


class RandomSearcher:
    """Phase 2: Random parameter search over strategy family."""
    
    def __init__(self, instrument='NQ', initial_cash=10000):
        self.instrument = instrument
        self.initial_cash = initial_cash
        self.factory = StrategyFactory()
        self.runner = BacktestRunner(instrument=instrument, initial_cash=initial_cash)
    
    def run(self, data, num_sets, start_date=None, end_date=None):
        """
        Run random search over parameter space.
        
        Args:
            data (list): OHLCV data from DataLoader
            num_sets (int): Number of random parameter sets to test
            start_date (str): Start date for backtest
            end_date (str): End date for backtest
        
        Returns:
            dict: Results with ranked parameter sets
        """
        ranges = self.factory.get_parameter_ranges()
        results = []
        
        print(f"\n[RANDOM SEARCH] Testing {num_sets} random parameter sets...")
        
        for i in range(num_sets):
            # Generate random params
            params = {}
            for key, values in ranges.items():
                params[key] = random.choice(values)
            
            # Validate
            is_valid, errors = self.factory.validate_params(params)
            if not is_valid:
                print(f"  Set {i+1}: SKIPPED (invalid: {errors})")
                continue
            
            # Run backtest
            backtest_result = self.runner.run(data, params, start_date=start_date, end_date=end_date)
            
            if backtest_result:
                results.append({
                    'rank': 0,  # Will be updated after sorting
                    'params': params,
                    'backtest': backtest_result,
                    'total_return_pct': backtest_result.get('total_return_pct', 0),
                    'trade_count': backtest_result.get('trade_count', 0),
                })
                print(f"  Set {i+1}/{num_sets}: {backtest_result['total_return_pct']:+.1f}% return, {backtest_result['trade_count']} trades")
        
        # Sort by return (simple ranking; can be enhanced)
        results.sort(key=lambda x: x['total_return_pct'], reverse=True)
        for rank, result in enumerate(results, 1):
            result['rank'] = rank
        
        return {
            'timestamp': datetime.now().isoformat(),
            'instrument': self.instrument,
            'search_type': 'random',
            'num_sets_tested': num_sets,
            'num_valid_results': len(results),
            'results': results[:20],  # Top 20
            'date_range': {
                'start': start_date,
                'end': end_date,
            }
        }


class WalkForwardOptimizer:
    """Phase 3: Walk-forward optimization with rolling windows."""
    
    def __init__(self, instrument='NQ', initial_cash=10000):
        self.instrument = instrument
        self.initial_cash = initial_cash
        self.factory = StrategyFactory()
        self.runner = BacktestRunner(instrument=instrument, initial_cash=initial_cash)
    
    def split_data(self, data, start_date_str, end_date_str, is_months=24, oos_months=6, step_months=3):
        """
        Split data into walk-forward windows.
        
        Args:
            data (list): Full OHLCV data
            start_date_str (str): Start date (YYYY-MM-DD)
            end_date_str (str): End date (YYYY-MM-DD)
            is_months (int): In-sample period (months)
            oos_months (int): Out-of-sample period (months)
            step_months (int): Step forward (months)
        
        Returns:
            list: Windows with (IS start, IS end, OOS start, OOS end)
        """
        from datetime import datetime, timedelta
        
        start_dt = datetime.fromisoformat(start_date_str)
        end_dt = datetime.fromisoformat(end_date_str)
        
        windows = []
        current_is_start = start_dt
        
        while current_is_start < end_dt:
            # IS period
            current_is_end = current_is_start + timedelta(days=30 * is_months)
            if current_is_end > end_dt:
                break
            
            # OOS period
            current_oos_start = current_is_end
            current_oos_end = current_oos_start + timedelta(days=30 * oos_months)
            if current_oos_end > end_dt:
                current_oos_end = end_dt
            
            windows.append({
                'is_start': current_is_start.date().isoformat(),
                'is_end': current_is_end.date().isoformat(),
                'oos_start': current_oos_start.date().isoformat(),
                'oos_end': current_oos_end.date().isoformat(),
            })
            
            # Step forward
            current_is_start += timedelta(days=30 * step_months)
        
        return windows
    
    def run(self, data, start_date_str, end_date_str, is_months=24, oos_months=6, step_months=3, num_random_sets=100):
        """
        Run walk-forward optimization.
        
        Args:
            data (list): Full OHLCV data
            start_date_str (str): Start date
            end_date_str (str): End date
            is_months (int): In-sample period
            oos_months (int): Out-of-sample period
            step_months (int): Step size
            num_random_sets (int): Random sets per IS window
        
        Returns:
            dict: WFO results with IS/OOS metrics per window
        """
        windows = self.split_data(data, start_date_str, end_date_str, is_months, oos_months, step_months)
        
        print(f"\n[WFO] Running {len(windows)} walk-forward windows...")
        print(f"  IS period: {is_months} months, OOS period: {oos_months} months, Step: {step_months} months")
        
        wfo_results = []
        
        for window_idx, window in enumerate(windows, 1):
            print(f"\n[WINDOW {window_idx}/{len(windows)}]")
            print(f"  IS: {window['is_start']} to {window['is_end']}")
            print(f"  OOS: {window['oos_start']} to {window['oos_end']}")
            
            # Random search on IS
            searcher = RandomSearcher(instrument=self.instrument, initial_cash=self.initial_cash)
            search_result = searcher.run(data, num_random_sets, start_date=window['is_start'], end_date=window['is_end'])
            
            # Test best IS param set on OOS
            if search_result['results']:
                best_params = search_result['results'][0]['params']
                print(f"  Best IS params selected; testing on OOS...")
                
                oos_result = self.runner.run(data, best_params, start_date=window['oos_start'], end_date=window['oos_end'])
                
                window_summary = {
                    'window_num': window_idx,
                    'is_period': window,
                    'oos_period': window,
                    'best_is_params': best_params,
                    'is_results': search_result['results'][0]['backtest'] if search_result['results'] else None,
                    'oos_results': oos_result,
                }
                
                wfo_results.append(window_summary)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'instrument': self.instrument,
            'wfo_type': 'rolling',
            'num_windows': len(wfo_results),
            'windows': wfo_results,
        }


class MonteCarlo:
    """Phase 4: Monte Carlo simulation for robustness testing."""
    
    def __init__(self, instrument='NQ', initial_cash=10000):
        self.instrument = instrument
        self.initial_cash = initial_cash
        self.runner = BacktestRunner(instrument=instrument, initial_cash=initial_cash)
    
    def run(self, data, params, num_sims=1000, start_date=None, end_date=None):
        """
        Run Monte Carlo boostrap on equity curve.
        
        Placeholder: Full implementation would:
        - Extract trades from backtest
        - Bootstrap trade order
        - Recalculate equity curve
        - Compute statistics (max DD, returns, Sharpe, etc.)
        
        Args:
            data (list): OHLCV data
            params (dict): Strategy parameters
            num_sims (int): Number of simulations
            start_date (str): Start date
            end_date (str): End date
        
        Returns:
            dict: Monte Carlo results
        """
        print(f"\n[MONTE CARLO] Running {num_sims} simulations...")
        
        # Run base case
        base_result = self.runner.run(data, params, start_date=start_date, end_date=end_date)
        
        if base_result is None:
            return None
        
        # Placeholder: Would implement actual bootstrap here
        # For now, return structure for future implementation
        
        return {
            'timestamp': datetime.now().isoformat(),
            'instrument': self.instrument,
            'num_simulations': num_sims,
            'base_result': base_result,
            'status': 'PLACEHOLDER - Full implementation requires trade extraction and bootstrapping',
            'note': 'See Phase 4 in design.md for Monte Carlo methodology',
        }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Phase 2–4: Random search, walk-forward, and Monte Carlo testing'
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['random-search', 'wfo', 'monte-carlo'],
                        default='random-search', help='Analysis mode')
    
    # Data arguments
    parser.add_argument('--data-file', type=str, default='data/nq_continuous_5m.csv',
                        help='Path to OHLCV data CSV file')
    parser.add_argument('--start-date', type=str, default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--instrument', type=str, choices=['NQ', 'MNQ'], default='NQ', help='Instrument')
    
    # Random search arguments
    parser.add_argument('--num-sets', type=int, default=100, help='Number of random sets to test')
    
    # Walk-forward arguments
    parser.add_argument('--is-period-months', type=int, default=24, help='In-sample period (months)')
    parser.add_argument('--oos-period-months', type=int, default=6, help='Out-of-sample period (months)')
    parser.add_argument('--step-months', type=int, default=3, help='Step size (months)')
    parser.add_argument('--num-random-sets', type=int, default=50, help='Random sets per IS window')
    
    # Monte Carlo arguments
    parser.add_argument('--num-sims', type=int, default=1000, help='Number of Monte Carlo simulations')
    parser.add_argument('--params-file', type=str, help='JSON file with best parameters')
    
    # Output arguments
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--initial-cash', type=float, default=10000, help='Starting capital')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load data
    print(f"[LOAD] Loading data from {args.data_file}...")
    loader = DataLoader(instrument=args.instrument)
    
    try:
        data = loader.load_csv(args.data_file, start_date=args.start_date, end_date=args.end_date)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    
    if data is None or len(data) == 0:
        print("[ERROR] No data loaded")
        return 1
    
    print(f"[LOAD] Loaded {len(data)} bars")
    
    # Mode routing
    if args.mode == 'random-search':
        searcher = RandomSearcher(instrument=args.instrument, initial_cash=args.initial_cash)
        results = searcher.run(data, args.num_sets, start_date=args.start_date, end_date=args.end_date)
    
    elif args.mode == 'wfo':
        optimizer = WalkForwardOptimizer(instrument=args.instrument, initial_cash=args.initial_cash)
        results = optimizer.run(data, args.start_date, args.end_date,
                               is_months=args.is_period_months,
                               oos_months=args.oos_period_months,
                               step_months=args.step_months,
                               num_random_sets=args.num_random_sets)
    
    elif args.mode == 'monte-carlo':
        # Load params from file
        if not args.params_file:
            print("[ERROR] --params-file required for monte-carlo mode")
            return 1
        
        with open(args.params_file, 'r') as f:
            params = json.load(f)
        
        mc = MonteCarlo(instrument=args.instrument, initial_cash=args.initial_cash)
        results = mc.run(data, params, num_sims=args.num_sims, start_date=args.start_date, end_date=args.end_date)
    
    else:
        print(f"[ERROR] Unknown mode: {args.mode}")
        return 1
    
    # Export results
    if results is None:
        print("[ERROR] Analysis failed")
        return 1
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) if args.output else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[EXPORT] Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2, default=str))
    
    print("\n[SUCCESS] Analysis completed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
