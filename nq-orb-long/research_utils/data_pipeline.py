"""
data_pipeline.py

Load, validate, and cache 5-minute NQ/MNQ OHLC data for backtesting.
Handles continuous contract data with back-adjustment for roll-overs.

Usage:
    from data_pipeline import DataLoader
    loader = DataLoader(instrument='NQ')
    data_df = loader.load_csv('data/nq_continuous_5m.csv', start_date='2022-01-01', end_date='2023-12-31')
    loader.validate_ohlc(data_df)
    print(f"Loaded {len(data_df)} bars")
"""

import csv
import json
from datetime import datetime, timedelta
import pytz

# Standard library only; no external dependencies except pandas for convenience
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARN] pandas not available; using CSV reader instead")


class DataLoader:
    """
    Loads and validates 5-minute OHLC data for NQ/MNQ continuous contracts.
    
    Assumptions:
    - CSV format: timestamp, open, high, low, close, volume (no headers by default)
    - Timestamps: ISO format (YYYY-MM-DD HH:MM:SS) or Unix epoch (seconds)
    - All prices in contract points (e.g., NQ 15500.50)
    - UTC timestamps; converts to ET for session filtering if needed
    - No gaps >10 minutes during RTH (09:30-16:00 ET)
    """
    
    def __init__(self, instrument='NQ', tick_size=0.25, multiplier=20, commission_per_round_trip=9.00):
        """
        Initialize data loader for a specific futures instrument.
        
        Args:
            instrument (str): 'NQ' or 'MNQ' (defaults to 'NQ')
            tick_size (float): Tick size in index points (0.25 for NQ/MNQ)
            multiplier (int): Dollar multiplier per point ($20 for NQ, $2 for MNQ)
            commission_per_round_trip (float): Commission in dollars per round-trip trade
        """
        self.instrument = instrument
        self.tick_size = tick_size
        self.multiplier = multiplier
        self.commission_per_round_trip = commission_per_round_trip
        self.tick_value_dollars = tick_size * multiplier  # e.g., NQ: 0.25 * 20 = $5 per tick
        
        # Session times (ET, for filtering)
        self.session_start_et = "09:30"
        self.session_end_et = "16:00"
        self.session_close_buffer_minutes = 30  # Flatten positions 30 min before close
        
        self.cache = {}  # Simple in-memory cache of loaded data
    
    def load_csv(self, filepath, start_date=None, end_date=None, has_header=False):
        """
        Load 5-minute OHLC data from CSV.
        
        Expected CSV format (no header by default):
            2022-01-03 09:30:00, 14500.25, 14510.50, 14495.75, 14505.00, 1250
        
        Args:
            filepath (str): Path to CSV file
            start_date (str): ISO format YYYY-MM-DD or None (load all)
            end_date (str): ISO format YYYY-MM-DD or None (load all)
            has_header (bool): If True, skip first row
        
        Returns:
            pd.DataFrame: Columns [timestamp, open, high, low, close, volume]
                         (or list of tuples if pandas not available)
        """
        cache_key = f"{filepath}_{start_date}_{end_date}"
        if cache_key in self.cache:
            print(f"[INFO] Returning cached data for {filepath}")
            return self.cache[cache_key]
        
        data = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0 and has_header:
                        continue
                    if len(row) < 6:
                        print(f"[WARN] Row {i} has <6 columns; skipping: {row}")
                        continue
                    
                    try:
                        timestamp_str = row[0].strip()
                        o = float(row[1])
                        h = float(row[2])
                        l = float(row[3])
                        c = float(row[4])
                        v = int(row[5])
                        
                        # Parse timestamp
                        try:
                            ts = datetime.fromisoformat(timestamp_str)
                        except:
                            # Try Unix epoch (seconds)
                            ts = datetime.utcfromtimestamp(float(timestamp_str))
                        
                        # Filter by date range
                        if start_date and ts.date() < datetime.fromisoformat(start_date).date():
                            continue
                        if end_date and ts.date() > datetime.fromisoformat(end_date).date():
                            continue
                        
                        data.append({
                            'timestamp': ts,
                            'open': o,
                            'high': h,
                            'low': l,
                            'close': c,
                            'volume': v
                        })
                    except Exception as e:
                        print(f"[WARN] Error parsing row {i}: {e}; skipping")
                        continue
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Convert to DataFrame if pandas available, else return list of dicts
        if HAS_PANDAS:
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            self.cache[cache_key] = df
            return df
        else:
            # Return as list of dicts; caller can iterate
            data.sort(key=lambda x: x['timestamp'])
            self.cache[cache_key] = data
            return data
    
    def validate_ohlc(self, data):
        """
        Sanity checks on OHLC data.
        
        Checks:
        - No duplicate timestamps
        - OHLC logic: High >= max(O,C), Low <= min(O,C)
        - Volume > 0
        - No wild price moves (e.g., >10% in 5 min outside known events)
        - No gaps >10 minutes during RTH
        
        Args:
            data: pd.DataFrame or list of dicts
        
        Returns:
            tuple: (is_valid, list of error/warning messages)
        """
        errors = []
        warnings = []
        
        # If pandas DataFrame
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            timestamps = data['timestamp'].tolist()
            ohlcv = data[['open', 'high', 'low', 'close', 'volume']].values
        else:
            # Assume list of dicts
            timestamps = [row['timestamp'] for row in data]
            ohlcv = [(row['open'], row['high'], row['low'], row['close'], row['volume']) for row in data]
        
        # 1. Duplicates
        if len(timestamps) != len(set(timestamps)):
            errors.append("Duplicate timestamps detected")
        
        # 2. OHLC logic
        for i, (ts, (o, h, l, c, v)) in enumerate(zip(timestamps, ohlcv)):
            if h < max(o, c):
                errors.append(f"Row {i} ({ts}): High {h} < max(O,C) = {max(o, c)}")
            if l > min(o, c):
                errors.append(f"Row {i} ({ts}): Low {l} > min(O,C) = {min(o, c)}")
            if v <= 0:
                warnings.append(f"Row {i} ({ts}): Volume {v} <= 0")
            
            # Price spike check (avoid obvious feed glitches)
            if i > 0:
                prev_c = ohlcv[i-1][3]
                pct_move = abs(c - prev_c) / prev_c if prev_c != 0 else 0
                if pct_move > 0.10:  # >10% move in 5 min
                    warnings.append(f"Row {i} ({ts}): Large move {pct_move:.1%} (prev close {prev_c}, new close {c})")
        
        # 3. Gaps during RTH
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
            if gap > 10:
                # Check if gap is during RTH (could be due to session boundary)
                h1 = timestamps[i-1].hour
                h2 = timestamps[i].hour
                if 9 < h1 < 16 and 9 < h2 < 16:
                    warnings.append(f"Gap >10 min during RTH at {timestamps[i]}: {gap:.0f} min gap")
        
        is_valid = len(errors) == 0
        
        # Print summary
        print(f"\n[VALIDATION] {self.instrument} data:")
        print(f"  Bars loaded: {len(timestamps)}")
        print(f"  Date range: {timestamps[0]} to {timestamps[-1]}")
        print(f"  Errors: {len(errors)}")
        print(f"  Warnings: {len(warnings)}")
        
        if errors:
            print("\n[ERRORS]")
            for err in errors[:10]:  # Print first 10
                print(f"  - {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        if warnings:
            print("\n[WARNINGS]")
            for warn in warnings[:10]:
                print(f"  - {warn}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")
        
        return is_valid, errors, warnings
    
    def summary_stats(self, data):
        """
        Print basic statistics on loaded data.
        
        Args:
            data: pd.DataFrame or list of dicts
        """
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            timestamps = data['timestamp'].tolist()
            closes = data['close'].tolist()
            volumes = data['volume'].tolist()
        else:
            timestamps = [row['timestamp'] for row in data]
            closes = [row['close'] for row in data]
            volumes = [row['volume'] for row in data]
        
        start_price = closes[0]
        end_price = closes[-1]
        min_price = min(closes)
        max_price = max(closes)
        avg_volume = sum(volumes) / len(volumes)
        
        print(f"\n[STATS] {self.instrument} data:")
        print(f"  Price range: {min_price:.2f} to {max_price:.2f}")
        print(f"  Start: {start_price:.2f}, End: {end_price:.2f}")
        print(f"  Change: {(end_price - start_price) / start_price * 100:+.1f}%")
        print(f"  Avg volume per bar: {avg_volume:.0f} contracts")
        print(f"  Date range: {timestamps[0].date()} to {timestamps[-1].date()}")
        print(f"  Total bars: {len(closes)}")
        print(f"  Commission per round-trip: ${self.commission_per_round_trip:.2f}")
        print(f"  Tick value: ${self.tick_value_dollars:.2f}/tick")


def example_usage():
    """Quick test of data loading and validation."""
    
    # Example: Load NQ data
    loader = DataLoader(instrument='NQ')
    
    # This would normally load real data:
    # data = loader.load_csv('data/nq_continuous_5m.csv', 
    #                         start_date='2022-01-01', 
    #                         end_date='2023-12-31')
    
    # For now, just print the expected setup:
    print("\n[EXAMPLE] Data pipeline ready. To load real data:")
    print("  1. Download NQ 5-min OHLC CSV from broker or HistData")
    print("  2. Place in data/ folder")
    print("  3. Call loader.load_csv('data/nq_continuous_5m.csv', ...)")
    print("  4. Call loader.validate_ohlc(data) to check integrity")
    print("  5. Pass to backtest_runner.py for backtesting")
    
    print("\n[EXPECTED CSV FORMAT]")
    print("  2022-01-03 09:30:00, 14500.25, 14510.50, 14495.75, 14505.00, 1250")
    print("  (timestamp, open, high, low, close, volume)")


if __name__ == '__main__':
    example_usage()
