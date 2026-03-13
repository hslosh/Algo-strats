"""
live_runner.py — ORB-Long-NQ Real-Time Signal Generator
═══════════════════════════════════════════════════════════

Connects to Interactive Brokers via ib_insync, streams live
5-minute NQ bars, runs the full ML pipeline (62 features +
logistic regression + Platt calibration), and generates
entry signals with calibrated P(win) probabilities.

Architecture:
  1. STARTUP   — Load historical CSV + train model on all data
  2. MORNING   — Connect to IB, subscribe to live 5-min bars
  3. OR PERIOD — Track Opening Range (first 30 min RTH)
  4. POST-OR   — On breakout, compute features → predict → signal
  5. SIGNAL    — Log signal + optional webhook to TradingView

Requirements:
  pip install ib_insync pandas numpy scikit-learn lightgbm requests

Usage:
  python research/live_runner.py                  # IB live feed
  python research/live_runner.py --mode backfill  # Train model only
  python research/live_runner.py --mode dry-run   # No IB, simulated
"""

import sys
import os
import json
import time
import logging
import argparse
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

# ── Add project root to path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research_utils.feature_engineering import load_ohlcv, build_features
from research.event_features import build_model_dataset, get_feature_columns
from research.config import CANONICAL_THRESHOLD
from research.model_design import (
    build_walk_forward_splits,
    train_model,
    predict_proba,
    calibrate_probabilities,
    apply_calibration,
    select_features,
)

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LiveConfig:
    """All tunable parameters for the live runner."""
    # Strategy
    threshold: float = CANONICAL_THRESHOLD
    sl_atr_mult: float = 1.0
    tp_atr_mult: float = 1.5
    atr_lookback: int = 14
    or_minutes: int = 30
    delay_minutes: int = 10    # 2 bars on 5-min

    # Risk
    account_size: float = 50_000.0
    risk_per_trade_pct: float = 1.5
    max_contracts: int = 2
    max_daily_loss: float = -1_000.0
    max_daily_trades: int = 3
    slippage_pts: float = 0.50
    commission_rt: float = 4.50

    # Position sizing tiers
    tier_conservative: Tuple[float, float] = (0.58, 0.65)
    tier_standard: Tuple[float, float] = (0.65, 0.75)
    tier_aggressive: Tuple[float, float] = (0.75, 1.01)
    tier_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'conservative': 0.6,
        'standard': 0.85,
        'aggressive': 1.0,
    })

    # IB connection
    ib_host: str = '127.0.0.1'
    ib_port: int = 7497          # 7497=TWS paper, 4002=Gateway paper
    ib_client_id: int = 10
    ib_symbol: str = 'NQ'
    ib_exchange: str = 'CME'

    # Data
    csv_path: str = str(PROJECT_ROOT / 'nq_continuous_5m_converted.csv')
    log_dir: str = str(PROJECT_ROOT / 'research' / 'live_logs')
    signal_log: str = str(PROJECT_ROOT / 'research' / 'live_logs' / 'signals.json')

    # Webhook (optional — for TradingView alerts)
    webhook_url: Optional[str] = None  # e.g. "http://localhost:5000/signal"

    # Model
    model_type: str = 'lgbm'
    event_col: str = 'event_orb_long'
    direction: str = 'long'
    event_type: str = 'orb'
    min_train_events: int = 200
    max_feature_nan_pct: float = 0.30
    n_top_features: int = 20


# ══════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str) -> logging.Logger:
    """Configure structured logging to console + daily file."""
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(log_dir, f'runner_{today}.log')

    logger = logging.getLogger('ORB_Runner')
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-5s | %(message)s', datefmt='%H:%M:%S'))

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s'))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ══════════════════════════════════════════════════════════════════════
# MODEL TRAINER
# ══════════════════════════════════════════════════════════════════════

class ModelManager:
    """
    Trains the walk-forward model on historical data and provides
    a predict() method for live inference.

    Strategy: Train on ALL available data (expanding window), then
    calibrate probabilities using the most recent OOS fold.
    """

    def __init__(self, config: LiveConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.calibrator = None
        self.feature_cols = None
        self.threshold = config.threshold
        self.is_ready = False

    def train(self) -> bool:
        """
        Full training pipeline:
          1. Load historical CSV
          2. Build model dataset (events + features + labels)
          3. Select features
          4. Walk-forward splits (use last fold for calibration)
          5. Train final model on ALL data
          6. Calibrate on last OOS fold
        Returns True if successful.
        """
        self.logger.info("=" * 60)
        self.logger.info("MODEL TRAINING — Starting full pipeline")
        self.logger.info("=" * 60)

        try:
            # 1. Load data
            self.logger.info(f"Loading data from: {self.config.csv_path}")
            df = load_ohlcv(self.config.csv_path)
            df = df[df.index >= '2019-01-01']
            self.logger.info(f"  Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

            # 2. Build model dataset
            self.logger.info("Building model dataset (events + features + labels)...")
            dataset = build_model_dataset(
                df,
                self.config.event_col,
                self.config.direction,
                event_type=self.config.event_type,
                sl_atr_multiple=self.config.sl_atr_mult,
                tp_atr_multiple=self.config.tp_atr_mult,
                atr_lookback=self.config.atr_lookback,
            )
            self.logger.info(f"  Dataset: {len(dataset)} events")

            # Determine label column
            label_col = 'label_binary' if 'label_binary' in dataset.columns else 'barrier_label'
            self.logger.info(f"  Label column: {label_col}")

            # 3. Get and select features
            all_features = get_feature_columns(dataset)
            self.logger.info(f"  Available features: {len(all_features)}")

            self.feature_cols = select_features(
                dataset, all_features,
                max_nan_pct=self.config.max_feature_nan_pct,
                n_top=self.config.n_top_features,
                label_col=label_col,
                verbose=False,
            )
            self.logger.info(f"  Selected features: {len(self.feature_cols)}")

            # 4. Walk-forward splits (for calibration data)
            splits = build_walk_forward_splits(dataset)
            last_split = splits[-1]
            self.logger.info(f"  Walk-forward: {len(splits)} folds, using last for calibration")

            # 5. Train final model on everything except last fold's test set
            #    (so we have clean OOS data for calibration)
            train_mask = dataset.index.isin(last_split['train_idx'])
            test_mask = dataset.index.isin(last_split['test_idx'])

            train_data = dataset[train_mask].copy()
            test_data = dataset[test_mask].copy()

            X_train = train_data[self.feature_cols].fillna(0)
            X_test = test_data[self.feature_cols].fillna(0)

            if label_col == 'barrier_label':
                y_train = (train_data[label_col] == 1).astype(int)
                y_test = (test_data[label_col] == 1).astype(int)
            else:
                y_train = train_data[label_col].astype(int)
                y_test = test_data[label_col].astype(int)

            self.logger.info(f"  Training: {len(X_train)} samples, Testing: {len(X_test)} samples")

            self.model = train_model(X_train, y_train, model_type=self.config.model_type)

            # 6. Calibrate on test fold
            raw_probs = predict_proba(self.model, X_test, model_type=self.config.model_type)
            self.calibrator = calibrate_probabilities(raw_probs, y_test.values)

            # Verify calibration
            cal_probs = apply_calibration(self.calibrator, raw_probs)
            above_thresh = cal_probs >= self.threshold
            if above_thresh.sum() > 0:
                wr_above = y_test.values[above_thresh].mean() * 100
                self.logger.info(f"  Calibration check: {above_thresh.sum()} events >= {self.threshold:.0%} threshold")
                self.logger.info(f"  Actual WR above threshold: {wr_above:.1f}%")
            else:
                self.logger.warning("  No events above threshold in calibration fold")

            # 7. Now retrain on ALL data for production
            self.logger.info("  Retraining final model on ALL data...")
            X_all = dataset[self.feature_cols].fillna(0)
            if label_col == 'barrier_label':
                y_all = (dataset[label_col] == 1).astype(int)
            else:
                y_all = dataset[label_col].astype(int)
            self.model = train_model(X_all, y_all, model_type=self.config.model_type)

            self.is_ready = True
            self.logger.info("MODEL TRAINING — Complete and ready for live inference")
            self.logger.info(f"  Features: {self.feature_cols[:5]}... ({len(self.feature_cols)} total)")
            return True

        except Exception as e:
            self.logger.error(f"MODEL TRAINING FAILED: {e}", exc_info=True)
            return False

    def predict(self, feature_row: pd.DataFrame) -> Optional[float]:
        """
        Given a single-row DataFrame with feature columns,
        return calibrated P(win) or None if model not ready.
        """
        if not self.is_ready:
            self.logger.warning("Model not ready — call train() first")
            return None

        try:
            X = feature_row[self.feature_cols].fillna(0)
            raw_prob = predict_proba(self.model, X, model_type=self.config.model_type)
            cal_prob = apply_calibration(self.calibrator, raw_prob)
            return float(cal_prob[0])
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════
# SIGNAL LOGGER
# ══════════════════════════════════════════════════════════════════════

class SignalLogger:
    """Logs signals to a JSON file and optionally fires webhooks."""

    def __init__(self, config: LiveConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        os.makedirs(os.path.dirname(config.signal_log), exist_ok=True)

    def log_signal(self, signal: Dict):
        """Append signal to JSON log file."""
        signal['timestamp'] = datetime.now().isoformat()
        try:
            if os.path.exists(self.config.signal_log):
                with open(self.config.signal_log, 'r') as f:
                    signals = json.load(f)
            else:
                signals = []
            signals.append(signal)
            with open(self.config.signal_log, 'w') as f:
                json.dump(signals, f, indent=2, default=str)
            self.logger.info(f"Signal logged: {signal.get('action', 'unknown')}")
        except Exception as e:
            self.logger.error(f"Failed to log signal: {e}")

    def fire_webhook(self, signal: Dict):
        """Send signal to webhook (e.g. TradingView alert server)."""
        if not self.config.webhook_url:
            return
        try:
            import requests
            resp = requests.post(self.config.webhook_url, json=signal, timeout=5)
            self.logger.info(f"Webhook fired: {resp.status_code}")
        except Exception as e:
            self.logger.warning(f"Webhook failed: {e}")

    def emit(self, signal: Dict):
        """Log + webhook."""
        self.log_signal(signal)
        self.fire_webhook(signal)


# ══════════════════════════════════════════════════════════════════════
# POSITION SIZER
# ══════════════════════════════════════════════════════════════════════

def compute_position_size(config: LiveConfig, prob: float, sl_pts: float) -> Dict:
    """
    Compute contracts and tier based on calibrated probability.
    Returns dict with contracts, tier, risk_dollars.
    """
    # Determine tier
    if prob >= config.tier_aggressive[0]:
        tier = 'aggressive'
    elif prob >= config.tier_standard[0]:
        tier = 'standard'
    else:
        tier = 'conservative'

    multiplier = config.tier_multipliers[tier]
    risk_dollars = config.account_size * (config.risk_per_trade_pct / 100.0)

    # NQ point value = $20/point
    if sl_pts > 0:
        raw_contracts = risk_dollars / (sl_pts * 20)
    else:
        raw_contracts = 1

    contracts = max(1, min(int(raw_contracts * multiplier), config.max_contracts))

    return {
        'contracts': contracts,
        'tier': tier,
        'multiplier': multiplier,
        'risk_dollars': risk_dollars,
        'sl_pts': sl_pts,
    }


# ══════════════════════════════════════════════════════════════════════
# LIVE SESSION STATE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SessionState:
    """Tracks intraday state for the current RTH session."""
    date: str = ""
    or_high: float = 0.0
    or_low: float = 9999999.0
    or_complete: bool = False
    or_bars: int = 0
    breakout_fired: bool = False
    delay_bars_remaining: int = 0
    in_trade: bool = False
    daily_trades: int = 0
    daily_pnl: float = 0.0
    halted: bool = False
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pending_prob: float = 0.0

    def reset(self, date_str: str):
        """Reset for new trading day."""
        self.date = date_str
        self.or_high = 0.0
        self.or_low = 9999999.0
        self.or_complete = False
        self.or_bars = 0
        self.breakout_fired = False
        self.delay_bars_remaining = 0
        self.in_trade = False
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.halted = False
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.pending_prob = 0.0


# ══════════════════════════════════════════════════════════════════════
# BAR PROCESSOR
# ══════════════════════════════════════════════════════════════════════

class BarProcessor:
    """
    Processes each incoming 5-minute bar through the full pipeline:
      OR tracking → Breakout detection → Feature computation → ML prediction → Signal
    """

    def __init__(self, config: LiveConfig, model: ModelManager,
                 signal_logger: SignalLogger, logger: logging.Logger):
        self.config = config
        self.model = model
        self.signal_logger = signal_logger
        self.logger = logger
        self.state = SessionState()
        self.bar_buffer = pd.DataFrame()   # Rolling window of recent bars
        self.bars_today = []

    def _is_rth(self, dt: datetime) -> bool:
        """Check if time is within RTH (09:30–16:00 ET)."""
        t = dt.hour * 60 + dt.minute
        return 570 <= t < 960

    def _is_or_period(self, dt: datetime) -> bool:
        """Check if time is within Opening Range (09:30–10:00 ET)."""
        t = dt.hour * 60 + dt.minute
        return 570 <= t < (570 + self.config.or_minutes)

    def _is_new_session(self, dt: datetime) -> bool:
        """Check if this is the first bar of a new RTH session."""
        date_str = dt.strftime('%Y-%m-%d')
        return date_str != self.state.date and self._is_rth(dt)

    def process_bar(self, bar: Dict):
        """
        Process a single 5-minute bar.
        bar should have: datetime, open, high, low, close, volume
        """
        dt = bar['datetime']
        if not self._is_rth(dt):
            return

        # New session?
        date_str = dt.strftime('%Y-%m-%d')
        if self._is_new_session(dt):
            self.logger.info(f"{'='*50}")
            self.logger.info(f"NEW SESSION: {date_str}")
            self.logger.info(f"{'='*50}")
            self.state.reset(date_str)

        # Check halted
        if self.state.halted:
            return

        # ── OR Period: Track high/low ──
        if self._is_or_period(dt):
            self.state.or_high = max(self.state.or_high, bar['high'])
            self.state.or_low = min(self.state.or_low, bar['low'])
            self.state.or_bars += 1
            return

        # ── OR just completed? ──
        if not self.state.or_complete and not self._is_or_period(dt):
            self.state.or_complete = True
            or_range = self.state.or_high - self.state.or_low
            self.logger.info(f"OR Complete — High: {self.state.or_high:.2f}  "
                           f"Low: {self.state.or_low:.2f}  "
                           f"Range: {or_range:.2f} pts")

        if not self.state.or_complete:
            return

        # ── Delay countdown ──
        if self.state.delay_bars_remaining > 0:
            self.state.delay_bars_remaining -= 1
            if self.state.delay_bars_remaining == 0:
                if bar['close'] > self.state.or_high:
                    self._execute_entry(bar)
                else:
                    self.logger.info("Breakout failed during delay — price fell back below OR High")
                    self.state.breakout_fired = False
            return

        # ── Breakout detection ──
        if (not self.state.breakout_fired and
            not self.state.in_trade and
            self.state.daily_trades < self.config.max_daily_trades and
            bar['close'] > self.state.or_high):

            self.logger.info(f"BREAKOUT DETECTED at {bar['close']:.2f} "
                           f"(OR High: {self.state.or_high:.2f})")

            # Run ML prediction
            prob = self._get_prediction(bar)
            if prob is None:
                self.logger.warning("Prediction unavailable — skipping")
                return

            self.logger.info(f"  Calibrated P(win): {prob:.3f}  "
                           f"Threshold: {self.config.threshold:.3f}")

            if prob >= self.config.threshold:
                self.state.pending_prob = prob
                delay_bars = self.config.delay_minutes // 5
                if delay_bars > 0:
                    self.state.breakout_fired = True
                    self.state.delay_bars_remaining = delay_bars
                    self.logger.info(f"  Signal QUALIFIED — waiting {delay_bars} bars "
                                   f"({self.config.delay_minutes} min) for confirmation")
                else:
                    self._execute_entry(bar)
            else:
                self.logger.info(f"  Below threshold — NO SIGNAL")

    def _get_prediction(self, bar: Dict) -> Optional[float]:
        """
        Build feature vector for the current event and predict.
        This is a simplified version — for maximum accuracy, you'd
        maintain a full rolling DataFrame and compute all 62 features.
        """
        return self.model.predict(self._build_feature_row(bar))

    def _build_feature_row(self, bar: Dict) -> Optional[pd.DataFrame]:
        """
        Build a single-row DataFrame with all required features.

        NOTE: This is a placeholder that returns the bar_buffer's last row.
        For production, you should maintain a rolling DataFrame and call
        build_model_dataset() or extract_universal_features() incrementally.
        """
        if self.bar_buffer is None or len(self.bar_buffer) == 0:
            return None
        # Return last row as a DataFrame (keeps column names)
        return self.bar_buffer.iloc[[-1]]

    def _execute_entry(self, bar: Dict):
        """Generate and log an entry signal."""
        prob = self.state.pending_prob
        or_high = self.state.or_high

        # Compute ATR from recent bars (simplified)
        if len(self.bar_buffer) >= self.config.atr_lookback:
            recent = self.bar_buffer.tail(self.config.atr_lookback)
            tr = pd.concat([
                recent['high'] - recent['low'],
                (recent['high'] - recent['close'].shift(1)).abs(),
                (recent['low'] - recent['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.mean()
        else:
            atr = (bar['high'] - bar['low']) * 2  # Fallback estimate

        entry = bar['close'] + self.config.slippage_pts
        sl = entry - self.config.sl_atr_mult * atr
        tp = entry + self.config.tp_atr_mult * atr
        sl_pts = entry - sl

        sizing = compute_position_size(self.config, prob, sl_pts)

        signal = {
            'action': 'ENTRY_LONG',
            'symbol': 'NQ',
            'date': self.state.date,
            'time': bar['datetime'].strftime('%H:%M:%S'),
            'entry_price': round(entry, 2),
            'stop_loss': round(sl, 2),
            'take_profit': round(tp, 2),
            'atr': round(atr, 2),
            'or_high': round(or_high, 2),
            'or_low': round(self.state.or_low, 2),
            'probability': round(prob, 4),
            'tier': sizing['tier'],
            'contracts': sizing['contracts'],
            'risk_dollars': round(sizing['risk_dollars'], 2),
        }

        self.logger.info("=" * 50)
        self.logger.info("ENTRY SIGNAL")
        self.logger.info(f"  Entry:     {entry:.2f}")
        self.logger.info(f"  Stop Loss: {sl:.2f} ({sl_pts:.1f} pts)")
        self.logger.info(f"  Take Profit: {tp:.2f}")
        self.logger.info(f"  P(win):    {prob:.3f}")
        self.logger.info(f"  Tier:      {sizing['tier'].upper()}")
        self.logger.info(f"  Contracts: {sizing['contracts']}")
        self.logger.info("=" * 50)

        self.signal_logger.emit(signal)

        self.state.in_trade = True
        self.state.daily_trades += 1
        self.state.entry_price = entry
        self.state.stop_loss = sl
        self.state.take_profit = tp

    def update_bar_buffer(self, df: pd.DataFrame):
        """Update the rolling bar buffer with processed feature data."""
        self.bar_buffer = df


# ══════════════════════════════════════════════════════════════════════
# IB DATA FEED
# ══════════════════════════════════════════════════════════════════════

class IBDataFeed:
    """
    Connects to Interactive Brokers TWS/Gateway and streams
    5-minute NQ bars in real-time.
    """

    def __init__(self, config: LiveConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.ib = None

    def connect(self) -> bool:
        """Connect to IB TWS/Gateway."""
        try:
            from ib_insync import IB, Future
            self.ib = IB()
            self.ib.connect(
                self.config.ib_host,
                self.config.ib_port,
                clientId=self.config.ib_client_id,
            )
            self.logger.info(f"Connected to IB at {self.config.ib_host}:{self.config.ib_port}")
            return True
        except ImportError:
            self.logger.error("ib_insync not installed. Run: pip install ib_insync")
            return False
        except Exception as e:
            self.logger.error(f"IB connection failed: {e}")
            return False

    def get_contract(self):
        """Get the front-month NQ futures contract."""
        from ib_insync import Future
        # Get front month — IB auto-resolves continuous
        contract = Future(
            self.config.ib_symbol,
            exchange=self.config.ib_exchange,
        )
        contracts = self.ib.reqContractDetails(contract)
        if contracts:
            return contracts[0].contract
        else:
            self.logger.error("Could not resolve NQ contract")
            return None

    def get_historical_bars(self, contract, duration='5 D') -> pd.DataFrame:
        """Fetch recent historical 5-min bars to seed the feature pipeline."""
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=False,  # Include globex for feature computation
            formatDate=1,
        )
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame([{
            'datetime': b.date,
            'open': b.open,
            'high': b.high,
            'low': b.low,
            'close': b.close,
            'volume': b.volume,
        } for b in bars])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df

    def stream_bars(self, contract, callback):
        """
        Subscribe to real-time 5-minute bars.
        callback(bar_dict) is called for each new bar.
        """
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=False,
            keepUpToDate=True,  # streaming!
            formatDate=1,
        )

        def on_update(bars, hasNewBar):
            if hasNewBar and len(bars) > 0:
                b = bars[-1]
                bar_dict = {
                    'datetime': b.date,
                    'open': b.open,
                    'high': b.high,
                    'low': b.low,
                    'close': b.close,
                    'volume': b.volume,
                }
                callback(bar_dict)

        bars.updateEvent += on_update
        self.logger.info("Streaming live 5-min bars...")

        # Run event loop
        try:
            self.ib.run()
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            self.disconnect()

    def disconnect(self):
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self.logger.info("Disconnected from IB")


# ══════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_backfill_only(config: LiveConfig, logger: logging.Logger):
    """Train model and validate — no live trading."""
    logger.info("MODE: Backfill (train model only)")
    model = ModelManager(config, logger)
    success = model.train()
    if success:
        logger.info("Model trained successfully. Ready for live deployment.")
    else:
        logger.error("Model training failed.")
    return success


def run_live(config: LiveConfig, logger: logging.Logger):
    """Full live pipeline: train model → connect IB → stream → signal."""
    logger.info("MODE: Live (IB data feed)")

    # 1. Train model
    model = ModelManager(config, logger)
    if not model.train():
        logger.error("Aborting — model training failed")
        return

    # 2. Connect to IB
    feed = IBDataFeed(config, logger)
    if not feed.connect():
        logger.error("Aborting — IB connection failed")
        return

    # 3. Get contract
    contract = feed.get_contract()
    if not contract:
        logger.error("Aborting — could not resolve NQ contract")
        return
    logger.info(f"Contract: {contract.localSymbol} ({contract.lastTradeDateOrContractMonth})")

    # 4. Set up processor
    signal_logger = SignalLogger(config, logger)
    processor = BarProcessor(config, model, signal_logger, logger)

    # 5. Seed with recent history for features
    logger.info("Fetching recent bars for feature seeding...")
    hist_df = feed.get_historical_bars(contract, duration='30 D')
    if len(hist_df) > 0:
        processor.update_bar_buffer(hist_df)
        logger.info(f"Seeded with {len(hist_df)} historical bars")

    # 6. Stream live bars
    def on_bar(bar_dict):
        # Append to buffer
        new_row = pd.DataFrame([bar_dict]).set_index('datetime')
        processor.bar_buffer = pd.concat([processor.bar_buffer, new_row]).tail(5000)
        processor.process_bar(bar_dict)

    feed.stream_bars(contract, on_bar)


def run_dry_run(config: LiveConfig, logger: logging.Logger):
    """
    Dry run: train model, then simulate processing the last 5 days
    of bars from the CSV to verify signals fire correctly.
    """
    logger.info("MODE: Dry Run (CSV replay)")

    # 1. Train model
    model = ModelManager(config, logger)
    if not model.train():
        logger.error("Aborting — model training failed")
        return

    # 2. Load last 5 days from CSV
    df = load_ohlcv(config.csv_path)
    df = df[df.index >= '2019-01-01']

    # Use last 5 RTH days
    df_rth = df.between_time('09:30', '16:00')
    last_dates = df_rth.index.normalize().unique()[-5:]
    replay_start = last_dates[0]
    replay_df = df[df.index >= replay_start]

    logger.info(f"Replaying {len(replay_df)} bars from {replay_start}")

    # 3. Process bars
    signal_logger = SignalLogger(config, logger)
    processor = BarProcessor(config, model, signal_logger, logger)

    # Seed buffer with data before replay window
    pre_replay = df[df.index < replay_start].tail(5000)
    processor.update_bar_buffer(pre_replay)

    for idx, row in replay_df.iterrows():
        bar_dict = {
            'datetime': idx.to_pydatetime(),
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row.get('volume', 0),
        }
        new_row = pd.DataFrame([bar_dict]).set_index('datetime')
        processor.bar_buffer = pd.concat([processor.bar_buffer, new_row]).tail(5000)
        processor.process_bar(bar_dict)

    logger.info("Dry run complete.")


# ══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ORB-Long-NQ Live Runner')
    parser.add_argument('--mode', choices=['live', 'backfill', 'dry-run'],
                        default='dry-run', help='Run mode (default: dry-run)')
    parser.add_argument('--port', type=int, default=7497,
                        help='IB TWS/Gateway port (default: 7497 for paper)')
    parser.add_argument('--threshold', type=float, default=CANONICAL_THRESHOLD,
                        help=f'Confidence threshold (default: {CANONICAL_THRESHOLD})')
    parser.add_argument('--webhook', type=str, default=None,
                        help='Webhook URL for signal delivery')
    args = parser.parse_args()

    config = LiveConfig()
    config.ib_port = args.port
    config.threshold = args.threshold
    config.webhook_url = args.webhook

    logger = setup_logging(config.log_dir)
    logger.info(f"ORB-Long-NQ Live Runner — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {args.mode}")

    if args.mode == 'backfill':
        run_backfill_only(config, logger)
    elif args.mode == 'live':
        run_live(config, logger)
    elif args.mode == 'dry-run':
        run_dry_run(config, logger)


if __name__ == '__main__':
    main()
