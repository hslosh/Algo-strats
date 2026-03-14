"""
backtest_runner.py - Working version with margin fix
"""

import argparse
import json
import sys
from datetime import datetime
import os
import backtrader as bt
from data_pipeline import DataLoader
from strategy_family import StrategyFactory


class BacktestRunner:
    def __init__(self, instrument='NQ', initial_cash=10000):
        self.instrument = instrument
        self.initial_cash = initial_cash
        self.loader = DataLoader(instrument=instrument)
        self.factory = StrategyFactory()
        from research.config import COMMISSION_PER_SIDE
        self.commission = COMMISSION_PER_SIDE  # $4.50 per side

    def run(self, data_list, params, start_date=None, end_date=None):
        cerebro = bt.Cerebro()
        strategy_class = self.factory.create_strategy(params)
        cerebro.addstrategy(strategy_class)
        
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        cerebro.broker.set_checksubmit(False)
        
        import pandas as pd
        df = pd.DataFrame(data_list)
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('datetime')
        data_feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data_feed)
        
        print(f"\n[BACKTEST] Running {self.instrument} strategy...")
        print(f"  Initial cash: ${self.initial_cash:,.2f}")
        print(f"  Params: {json.dumps(params, indent=4)}")
        
        strategies = cerebro.run()
        strategy = strategies[0]
        
        final_value = cerebro.broker.getvalue()
        total_return = final_value - self.initial_cash
        total_return_pct = (total_return / self.initial_cash) * 100
        
        result_dict = {
            'timestamp': datetime.now().isoformat(),
            'instrument': self.instrument,
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'trade_count': len(strategy.trade_log) // 2 if hasattr(strategy, 'trade_log') else 0,
            'params': params,
            'trade_log': strategy.trade_log if hasattr(strategy, 'trade_log') else [],
        }
        
        print(f"\n[RESULTS]")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: ${total_return:,.2f} ({total_return_pct:+.2f}%)")
        
        return result_dict

    def export_results(self, results, output_file):
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[EXPORT] Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str)
    parser.add_argument('--start-date', type=str)
    parser.add_argument('--end-date', type=str)
    parser.add_argument('--instrument', type=str, default='NQ')
    parser.add_argument('--initial-cash', type=float, default=50000)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    
    loader = DataLoader(instrument=args.instrument)
    print(f"[LOAD] Loading data from {args.data_file}...")
    data = loader.load_csv(args.data_file, start_date=args.start_date, end_date=args.end_date)
    
    if not data:
        print("[ERROR] No data loaded")
        return 1
    
    print(f"[LOAD] Loaded {len(data)} bars")
    loader.validate_ohlc(data)
    loader.summary_stats(data)
    
    params = {
        'fast_ma_len': 20, 'slow_ma_len': 50, 'atr_len': 14,
        'min_atr_ticks': 5, 'stop_loss_ticks': 12, 'target_ticks': 20,
        'max_trades_per_day': 3, 'max_position_bars': 12,
        'daily_loss_cap_ticks': 50, 'avoid_first_n_minutes': 5,
        'avoid_last_n_minutes': 30, 'use_trend_filter': False,
        'tick_value_dollars': 5.0, 'commission_per_trade': 4.50,
    }
    
    runner = BacktestRunner(instrument=args.instrument, initial_cash=args.initial_cash)
    results = runner.run(data, params, start_date=args.start_date, end_date=args.end_date)
    
    if args.output:
        runner.export_results(results, args.output)
    
    print("\n[SUCCESS] Backtest completed")
    return 0

if __name__ == '__main__':
    sys.exit(main())
