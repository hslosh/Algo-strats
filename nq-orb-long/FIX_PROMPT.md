# Fix Specification: nq_backtest_2 Codebase

**Addressed to:** Developer implementing corrections to the NQ E-mini futures ML research pipeline  
**Project root:** `/Users/henrys/nq_backtest_2/`  
**Context:** A full three-phase audit identified 22 issues across 5 priority tiers. This document is self-contained — implement every fix in tier order. Do not begin tier N+1 until tier N is complete and validated.

---

## BACKGROUND

The codebase implements an Opening Range Breakout (ORB) strategy on NQ 5-min OHLC data with a LightGBM + Logistic Regression probability model, walk-forward optimization (WFO), and a live runner via ib_insync. The pipeline flow is:

```
event_definitions.py  →  outcome_labeling.py  →  event_features.py
→  model_design.py  →  statistical_research.py  →  strategy_construction.py
→  backtest_validation.py  →  deployment_checklist.py  →  live_runner.py
```

Support utilities in `research_utils/`: `feature_engineering.py`, `wfo_and_robustness.py`, `data_pipeline.py`, `backtest_runner.py`.

**Canonical constants (use everywhere):**
- Tick size: 0.25 index points  
- Multiplier: $20/point (NQ), $2/point (MNQ)  
- Commission: **$4.50 per contract per side** ($9.00 RT)  
- Slippage: 1 tick = $5 per fill  
- RTH: 09:30–16:00 ET  
- ORB window: first 30 minutes (09:30–10:00 ET close)  
- Triple-barrier: SL = 1.0×ATR14, TP = 1.5×ATR14, timeout = 48 bars; SL-first on tie  
- Entry threshold: **0.58** (canonical; see P2-C)  
- WFO: expanding window, min 200 events train, 6-month test blocks, 5-day embargo  

---

## PRIORITY 1 — Fixes That Invalidate All Current Metrics

**All current performance numbers, IC values, and feature importances are meaningless until these are fixed. Fix these first, then delete and regenerate all cached outputs.**

---

### P1-A — Fix `session_range_position` Lookahead

**File:** `research/event_features.py`  
**Function:** `add_session_reference_features(df)` (approximately lines 88–105)  
**Severity:** CRITICAL — this feature has importance rank #1 (1.12) in the saved model and IC decay of >1.0 in `wf_feature_survival.csv`, which is statistically impossible for genuine OOS data. It encodes future price action into a feature computed at ~10:05 ET.

**Find this code:**
```python
df['session_high'] = df.groupby(df.index.date)['high'].transform('max')
df['session_low']  = df.groupby(df.index.date)['low'].transform('min')
df['session_range_position'] = (
    (df['close'] - df['session_low']) /
    (df['session_high'] - df['session_low'])
)
```

**Replace with:**
```python
# Use only price history up to and including the current bar (no lookahead)
df['session_high'] = df.groupby(df.index.date)['high'].transform(
    lambda x: x.expanding().max()
)
df['session_low'] = df.groupby(df.index.date)['low'].transform(
    lambda x: x.expanding().min()
)
session_range = df['session_high'] - df['session_low']
df['session_range_position'] = np.where(
    session_range > 0,
    (df['close'] - df['session_low']) / session_range,
    0.5  # Default to midpoint if range is zero (first bar of session)
)
```

**After applying this fix:**
1. Delete `research_utils/wf_oos_summary.csv`
2. Delete `research_utils/wf_feature_survival.csv`
3. Delete any cached model files (`.pkl`, `.joblib`, `.lgbm`) in the repo
4. Re-run the full pipeline from `event_features.py` forward
5. Verify: all `ic_decay` values in the regenerated `wf_feature_survival.csv` must be ≤ 1.0. Any value > 1.0 after this fix indicates a second lookahead source that must be found and fixed before proceeding.

---

### P1-B — Fix Calibration Leakage in WFO

**File:** `research/strategy_construction.py`  
**Function:** `generate_oos_predictions()` (approximately lines 180–240)  
**Severity:** CRITICAL — the current implementation accumulates OOS predictions from all WFO folds into one pool and then fits a single Platt calibrator on that pool. This means later folds' predictions are used to train a calibrator that is then applied to earlier folds — temporal leakage.

**Current broken pattern:**
```python
all_oos_probs  = []
all_oos_labels = []
for fold in wfo_splits:
    # ... train model, predict OOS fold ...
    all_oos_probs.extend(fold_probs)
    all_oos_labels.extend(fold_labels)

# BUG: calibrator sees all folds including "future" ones relative to earlier folds
calibrator = calibrate_probabilities(all_oos_probs, all_oos_labels, method='platt')
```

**Replace with a forward-only calibration scheme:**
```python
all_oos_probs   = []
all_oos_labels  = []
fold_results    = []  # store (probs, labels, timestamps) per fold

for fold_idx, fold in enumerate(wfo_splits):
    # Train model on fold's training set
    model = train_model(fold.train_data)
    fold_probs  = model.predict_proba(fold.test_data)[:, 1]
    fold_labels = fold.test_labels
    fold_times  = fold.test_timestamps

    # Calibrate using only folds 0..fold_idx-1 (never the current or future folds)
    if fold_idx == 0:
        # No prior OOS data available; use isotonic regression on a 50/50
        # chronological split of the current fold's own training set.
        n_cal = len(fold.train_data) // 2
        cal_data   = fold.train_data.iloc[-n_cal:]
        cal_labels = fold.train_labels.iloc[-n_cal:]
        cal_probs  = model.predict_proba(cal_data)[:, 1]
        calibrator = calibrate_probabilities(cal_probs, cal_labels, method='isotonic')
    else:
        prior_probs  = np.concatenate([r['probs']  for r in fold_results])
        prior_labels = np.concatenate([r['labels'] for r in fold_results])
        calibrator = calibrate_probabilities(prior_probs, prior_labels, method='platt')

    calibrated_probs = calibrator.predict_proba(fold_probs.reshape(-1, 1))[:, 1]

    fold_results.append({
        'probs':      fold_probs,
        'cal_probs':  calibrated_probs,
        'labels':     fold_labels,
        'timestamps': fold_times,
    })
    all_oos_probs.extend(calibrated_probs)
    all_oos_labels.extend(fold_labels)

# The production calibrator is fitted on all folds EXCEPT the holdout period
# (holdout events must not have contributed any training samples to any calibrator)
production_calibrator = calibrate_probabilities(
    all_oos_probs, all_oos_labels, method='platt'
)
```

**Additionally:** Ensure the final 6-month holdout period is split off from the data **before** `generate_oos_predictions()` is called. Add a `holdout_start_date` parameter and assert that no event with `timestamp >= holdout_start_date` appears in `wfo_splits`.

---

### P1-C — Fix Sharpe Annualization

**File:** `research/statistical_research.py`  
**Function:** `bootstrap_ev()` (approximately lines 70–85)  
**Severity:** CRITICAL — the formula `sharpe = (ev / std) * np.sqrt(min(n, 250))` treats the number of trades `n` as though it were 250 trading days. This inflates the Sharpe ratio and makes it incomparable to any published benchmark.

**Find this code:**
```python
ev  = np.mean(bootstrap_means)
std = np.std(bootstrap_means)
sharpe = (ev / std) * np.sqrt(min(n, 250))
```

**Replace with (add a `timestamps` parameter to `bootstrap_ev`):**
```python
def bootstrap_ev(returns, timestamps, n_bootstrap=10000, confidence=0.95, rng=None):
    """
    Compute bootstrapped EV and annualized Sharpe on daily P&L.
    
    Args:
        returns:    array-like of per-trade P&L in dollars
        timestamps: array-like of trade entry dates (date or datetime)
        ...
    """
    import pandas as pd
    returns    = np.array(returns)
    timestamps = pd.to_datetime(timestamps)

    # Group trade P&L by calendar date to get daily returns
    daily_pnl = pd.Series(returns, index=timestamps).groupby(
        timestamps.normalize()
    ).sum()

    # Annualized Sharpe on daily P&L series
    if daily_pnl.std() == 0:
        annualized_sharpe = 0.0
    else:
        annualized_sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)

    # Bootstrap trade-level EV (unchanged)
    if rng is None:
        rng = np.random.RandomState(42)
    bootstrap_means = [
        np.mean(rng.choice(returns, size=len(returns), replace=True))
        for _ in range(n_bootstrap)
    ]
    ev  = np.mean(bootstrap_means)
    std = np.std(bootstrap_means)
    ci_lo = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    ci_hi = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)

    return {
        'ev': ev,
        'std': std,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'annualized_sharpe': annualized_sharpe,  # replaces old 'sharpe'
        'n_trades': len(returns),
        'n_trading_days': len(daily_pnl),
    }
```

**Also fix the seed reinitialization bug in `parameter_sweep()`:**

Find this pattern inside the parameter sweep loop:
```python
for params in param_grid:
    rng = np.random.RandomState(42)   # BUG: resets seed for every combo
    result = bootstrap_ev(returns, rng=rng)
```

Replace with:
```python
rng = np.random.RandomState(42)   # Single RNG shared across all combos
for params in param_grid:
    result = bootstrap_ev(returns, rng=rng)
```

---

## PRIORITY 2 — Fixes to Make the Backtest Honest

**After completing P1 and regenerating outputs, apply these fixes to ensure the backtest reflects realistic execution.**

---

### P2-A — Rewrite Bar-by-Bar Backtest

**File:** `research/backtest_validation.py`  
**Function:** `run_bar_by_bar_backtest()` (full function)  
**Severity:** HIGH — the current function iterates `event_times` (pre-labeled events) and reads pre-computed `barrier_return_pts`. It is not a bar-by-bar simulation; it is an event-list replay that cannot detect position conflicts, misses intraday stop monitoring, and applies fixed slippage without checking whether the price was ever reachable.

**Replace the entire function body with:**
```python
def run_bar_by_bar_backtest(df_bars, events_df, config, slippage_ticks=1):
    """
    True bar-by-bar backtest.

    Args:
        df_bars:         Full 5-min OHLCV DataFrame, DatetimeIndex, ET-aware.
        events_df:       DataFrame with columns [event_time, signal, prob, entry_price].
                         Only rows where prob >= config.threshold are candidates.
        config:          BacktestConfig with fields: stop_loss_ticks, target_ticks,
                         max_position_bars, daily_loss_cap_ticks, avoid_last_n_minutes,
                         tick_value_dollars, commission_per_trade.
        slippage_ticks:  Slippage per fill in ticks (default 1 tick = $5).

    Returns:
        dict with trade_log, equity_curve, metrics dict
    """
    import pandas as pd
    tick = config.tick_size          # 0.25
    tv   = config.tick_value_dollars # $5
    slippage_pts = slippage_ticks * tick

    trade_log     = []
    equity_curve  = [config.initial_cash]
    cash          = config.initial_cash

    in_position   = False
    entry_price   = None
    stop_loss     = None
    take_profit   = None
    entry_bar_idx = None
    bars_held     = 0
    signal_dir    = None

    # Build a set of confirmed entry bars: event bar + 2-bar delay (P2-A note)
    # The ORB trigger fires at event_bar close; entry at open of event_bar+2
    pending_entries = {}   # bar_idx → (prob, direction)
    for _, ev in events_df[events_df['prob'] >= config.threshold].iterrows():
        event_bar_idx = df_bars.index.get_loc(ev['event_time'])
        entry_candidate_idx = event_bar_idx + 2  # 2-bar confirmation delay
        if entry_candidate_idx < len(df_bars):
            pending_entries[entry_candidate_idx] = {
                'prob': ev['prob'],
                'direction': ev.get('signal', 1),  # 1=long (ORB is long-only)
            }

    daily_pnl    = {}   # date → cumulative P&L for daily loss cap
    skipped_days = set()

    for i, (ts, bar) in enumerate(df_bars.iterrows()):
        bar_date  = ts.date()
        bar_time  = ts.time()

        # --- Daily loss cap: check at bar open ---
        today_pnl = daily_pnl.get(bar_date, 0.0)
        loss_cap_pts = config.daily_loss_cap_ticks * tick
        if today_pnl <= -loss_cap_pts:
            skipped_days.add(bar_date)

        # --- Exit check (must come before entry check) ---
        if in_position:
            bars_held += 1
            exit_price = None
            exit_reason = None

            # SL hit: assume filled at SL price if bar.low <= SL (long position)
            if signal_dir == 1 and bar['low'] <= stop_loss:
                exit_price  = stop_loss - slippage_pts   # slippage against us
                exit_reason = 'stop_loss'
            # TP hit
            elif signal_dir == 1 and bar['high'] >= take_profit:
                exit_price  = take_profit - slippage_pts
                exit_reason = 'take_profit'
            # Time stop: max bars held
            elif bars_held >= config.max_position_bars:
                exit_price  = bar['close'] - slippage_pts
                exit_reason = 'timeout'
            # EOD: avoid_last_n_minutes before 16:00
            elif bar_time >= _time_subtract(config.avoid_last_n_minutes):
                exit_price  = bar['close'] - slippage_pts
                exit_reason = 'eod_flatten'

            if exit_price is not None:
                pnl_pts = (exit_price - entry_price) * signal_dir
                pnl_usd = pnl_pts * (1.0 / tick) * tv - config.commission_per_trade * 2
                cash   += pnl_usd
                daily_pnl[bar_date] = daily_pnl.get(bar_date, 0.0) + pnl_usd
                trade_log.append({
                    'entry_time':   df_bars.index[entry_bar_idx],
                    'exit_time':    ts,
                    'entry_price':  entry_price,
                    'exit_price':   exit_price,
                    'pnl_pts':      pnl_pts,
                    'pnl_usd':      pnl_usd,
                    'reason':       exit_reason,
                    'bars_held':    bars_held,
                })
                in_position   = False
                entry_price   = None
                stop_loss     = None
                take_profit   = None
                entry_bar_idx = None
                bars_held     = 0

        equity_curve.append(cash)

        # --- Entry check ---
        if (not in_position
                and bar_date not in skipped_days
                and i in pending_entries
                and bar_time < _time_subtract(config.avoid_last_n_minutes)):

            ev_info   = pending_entries[i]
            # Confirm price is still valid (close above OR High for longs)
            # If you track or_high, check: bar['close'] >= or_high
            # Here we use open as entry (next bar after confirmation)
            entry_price  = bar['open'] + slippage_pts
            atr_pts      = _get_atr(df_bars, i, period=14)
            stop_loss    = entry_price - (config.stop_loss_ticks * tick)  # or 1×ATR
            take_profit  = entry_price + (config.target_ticks * tick)     # or 1.5×ATR
            signal_dir   = ev_info['direction']
            entry_bar_idx = i
            bars_held    = 0
            in_position  = True
            daily_pnl[bar_date] = daily_pnl.get(bar_date, 0.0) - config.commission_per_trade

    return {
        'trade_log':    trade_log,
        'equity_curve': equity_curve,
        'final_cash':   cash,
        'n_trades':     len(trade_log),
    }


def _time_subtract(n_minutes, close_hour=16, close_min=0):
    """Return a time object n_minutes before 16:00 ET."""
    from datetime import time
    total = close_hour * 60 + close_min - n_minutes
    return time(total // 60, total % 60)


def _get_atr(df_bars, idx, period=14):
    """Compute ATR at bar idx using a backward-looking window."""
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
```

---

### P2-B — Enforce Holdout Separation

**File:** `research/backtest_validation.py` and `research/strategy_construction.py`  
**Severity:** HIGH

In the main pipeline entry point (wherever `generate_oos_predictions()` is called):

```python
# Add this BEFORE calling generate_oos_predictions
HOLDOUT_START_DATE = pd.Timestamp('2025-07-01', tz='America/New_York')

# Split events into WFO pool and holdout
wfo_events     = events_df[events_df.index < HOLDOUT_START_DATE]
holdout_events = events_df[events_df.index >= HOLDOUT_START_DATE]

# Run WFO only on wfo_events
oos_probs, oos_labels, calibrator = generate_oos_predictions(wfo_events, config)

# Apply final calibrator to holdout events (no calibrator fitting on holdout)
holdout_raw_probs  = production_model.predict_proba(holdout_features)[:, 1]
holdout_cal_probs  = calibrator.predict_proba(holdout_raw_probs.reshape(-1, 1))[:, 1]

# Report holdout metrics separately — these are the honest numbers
print(f"\n[HOLDOUT] n={len(holdout_events)} events")
print(f"[HOLDOUT] AUC: {roc_auc_score(holdout_labels, holdout_cal_probs):.4f}")
```

---

### P2-C — Standardize Entry Threshold

**Files:** `research/strategy_construction.py`, `research/backtest_validation.py`, `research/live_runner.py`, `research/deployment_checklist.py`  
**Severity:** MEDIUM — `StrategyConfig.threshold = 0.42` contradicts `BacktestConfig.threshold = 0.58`. The audit cannot determine which is correct without re-running the pipeline, but a single source of truth must exist.

**Create `research/config.py`:**
```python
"""Shared configuration constants for the NQ ORB ML pipeline."""

# --- Canonical trading constants ---
CANONICAL_THRESHOLD    = 0.58    # Minimum calibrated probability to enter
TICK_SIZE              = 0.25    # NQ/MNQ tick size in index points
NQ_MULTIPLIER          = 20      # Dollars per point
MNQ_MULTIPLIER         = 2
COMMISSION_PER_SIDE    = 4.50    # Per contract per fill
SLIPPAGE_TICKS         = 1       # Ticks of slippage per fill
STOP_LOSS_ATR_MULT     = 1.0
TAKE_PROFIT_ATR_MULT   = 1.5
MAX_POSITION_BARS      = 48
AVOID_LAST_N_MINUTES   = 30
DAILY_LOSS_CAP_TICKS   = 50
MAX_TRADES_PER_DAY     = 3
```

**In `strategy_construction.py`:**
```python
from research.config import CANONICAL_THRESHOLD
# Replace: threshold: float = 0.42
# With:
threshold: float = CANONICAL_THRESHOLD
```

**In `backtest_validation.py`:**
```python
from research.config import CANONICAL_THRESHOLD
# Replace: threshold: float = 0.58
# With:
threshold: float = CANONICAL_THRESHOLD
```

**In `live_runner.py`:**
```python
from research.config import CANONICAL_THRESHOLD
# Replace inline threshold values with CANONICAL_THRESHOLD
```

---

## PRIORITY 3 — Fixes Required Before Any Live Deployment

**Do not connect to a live account until all P3 fixes are complete and verified in paper trading.**

---

### P3-A — Implement `_build_feature_row()` in Live Runner

**File:** `research/live_runner.py`  
**Class:** `BarProcessor`  
**Method:** `_build_feature_row(self, bar: Dict) -> Optional[pd.DataFrame]`  
**Severity:** CRITICAL — the current implementation is an explicit placeholder that returns a raw OHLCV row instead of the 62 engineered features the model was trained on. Any signal produced by the current live runner is meaningless.

**Step 1:** Add a helper to `research/event_features.py` that extracts features for a single event bar from a rolling DataFrame:

```python
def extract_event_features_row(df_bars: pd.DataFrame, event_time: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Given a rolling DataFrame of bars (recent history), compute all 62 features
    for the bar at `event_time`. Returns a single-row DataFrame or None if
    insufficient history is available.

    Args:
        df_bars:    DataFrame with OHLCV columns and DatetimeIndex.
                    Must contain at least 252 trading days of bars for all
                    features to be valid (e.g., rvol_1d uses a 252-day window).
                    Minimum viable: 50 bars for fast features only.
        event_time: The timestamp of the ORB event bar (10:00 ET close bar).

    Returns:
        Single-row pd.DataFrame with all feature columns, or None.
    """
    from research_utils.feature_engineering import build_features
    if len(df_bars) < 50:
        return None
    if event_time not in df_bars.index:
        return None

    # Build the full feature matrix on the available history
    feature_df = build_features(df_bars.copy())

    # Extract only the row for this event
    if event_time not in feature_df.index:
        return None
    row = feature_df.loc[[event_time]]

    # Verify all expected columns are present
    expected_cols = get_expected_feature_columns()   # see Step 2
    missing = set(expected_cols) - set(row.columns)
    if missing:
        logger.warning(f"[FEATURES] Missing {len(missing)} features: {sorted(missing)[:5]}...")
        return None

    return row[expected_cols]
```

**Step 2:** Add to `research/event_features.py`:

```python
def get_expected_feature_columns() -> list:
    """Return the canonical ordered list of 62 feature column names."""
    return [
        # Time features
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        # Price/level features
        'or_range_pts', 'or_range_atr', 'breakout_strength',
        'session_range_position',   # Must use expanding version after P1-A fix
        'dist_to_vwap_ticks', 'dist_to_ema20_ticks', 'dist_to_ema50_ticks',
        # ATR / volatility features
        'atr14', 'atr_ratio_5_14', 'atr_ratio_14_50',
        'realized_vol_5', 'realized_vol_20',
        # Volume features
        'rvol_ratio_24', 'rvol_ratio_5d', 'volume_ma20_ratio',
        'or_volume_ratio',
        # Momentum features
        'roc_1', 'roc_5', 'roc_20',
        'rsi14', 'stoch_k', 'stoch_d',
        # VWAP deviation features
        'vwap_dev_1', 'vwap_dev_5',
        # Regime features
        'trend_strength', 'adx14', 'regime_vol',
        # Add all remaining features here to reach 62 total
        # ... (fill in from model_design.py feature list)
    ]
```

**Step 3:** Replace `_build_feature_row()` in `live_runner.py`:

```python
def _build_feature_row(self, bar: Dict) -> Optional[pd.DataFrame]:
    """
    Build the 62-feature row for the current ORB event bar.
    Uses a rolling buffer of recent 5-min bars.
    """
    if self.bar_buffer is None or len(self.bar_buffer) < 50:
        logger.warning("[FEATURES] Insufficient bar history; need >= 50 bars")
        return None

    event_time = pd.Timestamp(bar['datetime'])

    # Import the correct helper (NOT the placeholder path)
    from research.event_features import extract_event_features_row
    feature_row = extract_event_features_row(self.bar_buffer.copy(), event_time)

    if feature_row is None:
        logger.warning(f"[FEATURES] Could not build features for {event_time}")
        return None

    return feature_row
```

**Verification:** After implementing, add a unit test that:
1. Loads 300 bars from the historical CSV
2. Calls `_build_feature_row()` on bar 251
3. Asserts the returned DataFrame has exactly 62 columns
4. Asserts no column is NaN

---

### P3-B — Fix Live Model/Calibrator Mismatch

**File:** `research/live_runner.py`  
**Class:** `ModelManager`  
**Method:** `train()` (approximately lines 250–275)  
**Severity:** HIGH — after calibrating on the WFO OOS predictions, the code retrains the production model on ALL data. This produces a new model with different internal weights, making the calibrator (fitted on the prior model's outputs) invalid.

**Find this block:**
```python
# Retraining final model on ALL data
logger.info("Retraining final model on ALL data for production...")
self.model.fit(X_all, y_all)
```

**Option A (preferred — keep calibrator valid):** Remove the retrain block. Use the model trained on all-but-holdout as the production model. The calibrator remains valid.

```python
# REMOVE the "Retraining final model on ALL data" block entirely.
# The production model is the one trained during the last WFO fold.
# The calibrator was fitted on that model's OOS predictions — it remains valid.
logger.info("Production model: last WFO fold model (calibrator is valid for it)")
```

**Option B (if you must train on all data):** After retraining on all data, re-calibrate using a 20% chronological holdout of the full training set:

```python
# If retraining on all data, re-calibrate on a fresh holdout
n_cal = int(len(X_all) * 0.20)
X_train_final = X_all[:-n_cal]
X_cal_final   = X_all[-n_cal:]
y_train_final = y_all[:-n_cal]
y_cal_final   = y_all[-n_cal:]

self.model.fit(X_train_final, y_train_final)
cal_probs  = self.model.predict_proba(X_cal_final)[:, 1]
self.calibrator = calibrate_probabilities(cal_probs, y_cal_final, method='isotonic')
logger.info(f"Re-calibrated on {n_cal} samples after full retrain")
```

---

### P3-C — Fix IB Contract Specification

**File:** `research/live_runner.py`  
**Class:** `IBDataFeed`  
**Method:** `get_contract()` (approximately lines 85–100)  
**Severity:** MEDIUM — `Future('NQ', exchange='CME')` without `lastTradeDateOrContractMonth` causes IB to return multiple contracts. The code does not handle this case.

**Replace:**
```python
contract = Future('NQ', exchange='CME')
```

**With:**
```python
import datetime

def get_contract(self) -> Contract:
    """Resolve the front-month NQ futures contract via IB qualifyContracts."""
    # Create an underspecified contract; IB will resolve front month
    contract = Future(
        symbol='NQ',
        exchange='CME',
        currency='USD',
    )
    # qualifyContracts fills in expiry, conId, etc.
    try:
        details = self.ib.reqContractDetails(contract)
        if not details:
            raise RuntimeError("No contract details returned for NQ CME")
        if len(details) > 1:
            # Multiple contracts: pick the nearest expiry
            details.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
            logger.info(f"Multiple NQ contracts found; using nearest expiry: "
                        f"{details[0].contract.lastTradeDateOrContractMonth}")
        qualified = details[0].contract
        logger.info(f"[IB] Resolved NQ contract: {qualified.localSymbol} "
                    f"expiry={qualified.lastTradeDateOrContractMonth}")
        return qualified
    except Exception as e:
        logger.error(f"[IB] Contract resolution failed: {e}")
        raise
```

---

### P3-D — Add SL/TP Exit Monitoring to Live Runner

**File:** `research/live_runner.py`  
**Class:** `BarProcessor`  
**Method:** `process_bar()` (approximately lines 480–540)  
**Severity:** CRITICAL — the live runner has no exit logic. Once a position is entered, it is never exited by the bot. This is a live trading deployment blocker.

**Add the following exit check at the TOP of `process_bar()`, before the entry check:**

```python
def process_bar(self, bar: Dict) -> None:
    """Process a new 5-min bar. Check exits first, then entries."""
    ts       = pd.Timestamp(bar['datetime'])
    bar_date = ts.date()
    bar_time = ts.time()

    # === EXIT CHECK (always runs first) ===
    if self.state.in_trade:
        self.state.bars_held += 1
        exit_order = None
        exit_reason = None

        low  = bar['low']
        high = bar['high']

        # Stop loss hit
        if low <= self.state.stop_loss:
            exit_order  = MarketOrder('SELL', self.state.position_size)
            exit_reason = 'stop_loss'
        # Take profit hit
        elif high >= self.state.take_profit:
            exit_order  = LimitOrder('SELL', self.state.position_size, self.state.take_profit)
            exit_reason = 'take_profit'
        # Time stop
        elif self.state.bars_held >= MAX_POSITION_BARS:
            exit_order  = MarketOrder('SELL', self.state.position_size)
            exit_reason = 'timeout'
        # EOD flatten: AVOID_LAST_N_MINUTES before 16:00
        elif bar_time >= _eod_threshold():
            exit_order  = MarketOrder('SELL', self.state.position_size)
            exit_reason = 'eod_flatten'

        if exit_order is not None:
            logger.info(f"[EXIT] {exit_reason} at ~{bar['close']} (bar {ts})")
            trade = self.ib.placeOrder(self.state.contract, exit_order)
            self.state.in_trade       = False
            self.state.stop_loss      = None
            self.state.take_profit    = None
            self.state.position_size  = 0
            self.state.bars_held      = 0
            self._record_exit(trade, exit_reason, ts)
            return   # Do not evaluate entries on the same bar as exit

    # === ENTRY CHECK (only when flat) ===
    # ... existing entry logic follows ...
```

**Add `LiveState` fields if not already present:**
```python
@dataclass
class LiveState:
    in_trade:      bool  = False
    stop_loss:     float = 0.0
    take_profit:   float = 0.0
    position_size: int   = 0
    bars_held:     int   = 0
    entry_price:   float = 0.0
    contract:      object = None
```

---

## PRIORITY 4 — Correctness Fixes for Reporting and Code Quality

---

### P4-A — Replace Hardcoded Metrics in Deployment Checklist

**File:** `research/deployment_checklist.py`  
**Function:** `print_framework_summary()` (approximately lines 378–402)  
**Severity:** HIGH — all metrics are hardcoded strings; the function never reads actual results.

**Replace:**
```python
def print_framework_summary():
    print("Performance: 100 trades, 73% WR, Sharpe 10.65, DD -$2,661")
    # ... etc
```

**With:**
```python
def print_framework_summary(metrics: dict = None):
    """
    Print framework summary. Pass `metrics` dict from the most recent backtest run.
    If metrics is None, prints placeholder text only (no performance claims).
    
    Expected keys in metrics:
        n_trades, win_rate, annualized_sharpe, max_drawdown_usd,
        total_return_pct, avg_trade_pnl, profit_factor, holdout_auc
    """
    print("\n" + "="*60)
    print("NQ ORB ML STRATEGY — FRAMEWORK SUMMARY")
    print("="*60)
    if metrics is None:
        print("[WARNING] No metrics provided. Run the backtest pipeline first.")
        print("          Call print_framework_summary(metrics=result_dict)")
        return

    print(f"  Trades (OOS):        {metrics.get('n_trades', 'N/A')}")
    print(f"  Win Rate:            {metrics.get('win_rate', 0):.1%}")
    print(f"  Annualized Sharpe:   {metrics.get('annualized_sharpe', 0):.2f}")
    print(f"  Max Drawdown:        ${metrics.get('max_drawdown_usd', 0):,.0f}")
    print(f"  Avg Trade P&L:       ${metrics.get('avg_trade_pnl', 0):.2f}")
    print(f"  Profit Factor:       {metrics.get('profit_factor', 0):.2f}x")
    print(f"  Holdout AUC:         {metrics.get('holdout_auc', 0):.4f}")
    print(f"  Total Return (OOS):  {metrics.get('total_return_pct', 0):+.1f}%")
    print("="*60)
```

---

### P4-B — Move Feature Selection Inside WFO Fold Loop

**File:** `research/model_design.py`  
**Function:** wherever `select_features()` is currently called on the full dataset before splitting  
**Severity:** HIGH — feature selection on the full dataset uses OOS data to influence which features are kept, introducing subtle selection bias.

**Find the current pattern:**
```python
# WRONG: runs on full dataset before WFO splits
selected_features = select_features(full_df, labels, method='importance')
for fold in wfo_splits:
    model.fit(fold.train[selected_features], fold.train_labels)
```

**Replace with:**
```python
fold_selected_features = []
for fold in wfo_splits:
    # Feature selection uses ONLY the training fold
    fold_features = select_features(
        fold.train_data, fold.train_labels, method='importance'
    )
    fold_selected_features.append(fold_features)
    model.fit(fold.train_data[fold_features], fold.train_labels)
    # ...

# For production model: take the union of features selected in >=50% of folds
from collections import Counter
feature_counts = Counter(f for fs in fold_selected_features for f in fs)
n_folds = len(wfo_splits)
production_features = [
    f for f, count in feature_counts.items()
    if count >= n_folds * 0.50
]
logger.info(f"Production feature set: {len(production_features)} features "
            f"(union of >=50% fold selections)")
```

---

### P4-C — Fix `consec_losses` Pause Logic

**File:** `research/strategy_construction.py`  
**Function:** `simulate_strategy()` (approximately lines 295–340)  
**Severity:** MEDIUM — after 3 consecutive losses, the current code pauses for exactly one signal, not for the rest of the trading day.

**Find:**
```python
if consec_losses >= config.max_consec_losses:
    consec_losses = 0   # Reset after skipping one signal
    continue
```

**Replace with:**
```python
if consec_losses >= config.max_consec_losses:
    # Halt for the rest of this calendar date
    daily_halted_dates.add(event_date)
    logger.debug(f"[HALT] Day {event_date} halted after {consec_losses} consecutive losses")
    continue

# At the top of the loop, also add:
if event_date in daily_halted_dates:
    continue
```

**Add initialization before the loop:**
```python
daily_halted_dates = set()
consec_losses      = 0
prev_date          = None

# Reset consec_losses at start of each new date
if event_date != prev_date:
    consec_losses = 0
    prev_date     = event_date
```

---

### P4-D — Fix Max Daily Loss Boundary Condition

**File:** `research/strategy_construction.py`  
**Function:** `simulate_strategy()`  
**Severity:** MEDIUM — the daily loss cap is checked after the trade P&L is computed, allowing the last trade to exceed the cap.

**Find:**
```python
pnl = compute_trade_pnl(trade, config)
daily_pnl[date] += pnl
if daily_pnl[date] <= -config.daily_loss_cap:
    halt_day(date)
```

**Replace with (check BEFORE executing the trade):**
```python
# Estimate worst-case P&L for this trade (full SL hit + commission)
worst_case_pnl = -(config.stop_loss_ticks * config.tick_value_dollars
                   + config.commission_per_trade * 2)

# Only enter if even a full SL loss would not breach the daily cap
if daily_pnl.get(date, 0.0) + worst_case_pnl <= -config.daily_loss_cap_usd:
    logger.debug(f"[SKIP] {date}: entering would risk breaching daily loss cap")
    continue

pnl = compute_trade_pnl(trade, config)
daily_pnl[date] = daily_pnl.get(date, 0.0) + pnl
```

---

### P4-E — Remove Global Warning Suppression

**File:** `research/deployment_checklist.py` (line 18)  
**Severity:** LOW

**Remove:**
```python
warnings.filterwarnings('ignore')
```

**Replace with targeted suppression only:**
```python
import warnings
# Suppress only known benign warnings from specific libraries
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
# All other warnings remain visible
```

---

## PRIORITY 5 — Documentation and Consistency

---

### P5-A — Delete Stale CSV Outputs

**After P1-A is applied and the pipeline is re-run:**
```bash
rm research_utils/wf_oos_summary.csv
rm research_utils/wf_feature_survival.csv
# Regenerate by running the full pipeline
python research/model_design.py
```

The new CSVs must show all `ic_decay` values ≤ 1.0.

---

### P5-B — Reconcile Pine Script vs Python

**File:** `research/orb_long_nq.pine` — add to the header comment block:

```
// IMPORTANT: This Pine Script implements a HEURISTIC approximation of the
// strategy concept using 10 scoring rules (0–100 composite score). It is NOT
// equivalent to the Python ML pipeline which uses a 62-feature LightGBM model
// with walk-forward optimization and Platt-scaled probabilities.
//
// Performance metrics shown in this script are for the PINE heuristic only:
//   137 trades, 70.8% WR, PF 2.41×, max DD −$2,156 (TradingView backtest)
//
// Python ML pipeline metrics (post-fix, to be updated after P1 is applied):
//   [re-run pipeline and fill in computed numbers here]
//
// Features unique to Pine script NOT yet in Python:
//   - Trailing stop (0.5×ATR activation, 0.75×ATR trail)
//   - 2-bar entry delay (i_delay_bars = 2)
// These should be added to the Python backtest (see P2-A).
```

---

### P5-C — Deduplicate `true_range()` Function

The function `_true_range()` appears independently in three files:
- `research/event_definitions.py`
- `research/outcome_labeling.py`
- `research_utils/feature_engineering.py`

**Create `research_utils/utils.py`:**
```python
"""Shared utility functions for the NQ ORB pipeline."""
import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Compute True Range for each bar.
    
    TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
    
    Returns:
        pd.Series of TR values (NaN for the first bar).
    """
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr
```

**In each of the three files, replace the local `_true_range()` definition with:**
```python
from research_utils.utils import true_range as _true_range
```

---

### P5-D — Standardize Commission Across All Files

Commission must be `$4.50 per contract per side` everywhere:

| File | Current value | Fix |
|------|--------------|-----|
| `research_utils/data_pipeline.py` | `commission_per_round_trip=12.50` | Change to `9.00` (2 × $4.50) |
| `research_utils/backtest_runner.py` | `self.commission = 6.25` (NQ) | Change to `4.50` |
| `research/strategy_construction.py` | Check all `commission_per_trade` references | Use `4.50` |
| `research/backtest_validation.py` | Check `commission_per_trade` | Use `4.50` |
| `research/live_runner.py` | Check commission references | Use `4.50` |

After P2-C, all files should import `COMMISSION_PER_SIDE` from `research/config.py`.

---

### P5-E — Save Computed Results to Disk

**All three main pipeline scripts must write timestamped JSON output to `outputs/`.**

Add at the end of each main pipeline run function:

```python
import json
import os
from datetime import datetime

def save_pipeline_results(results: dict, stage: str) -> str:
    """
    Save computed results to outputs/ directory.
    
    Args:
        results: Dict of computed metrics (NOT hardcoded strings)
        stage:   Pipeline stage name (e.g., 'step6_backtest', 'step8_deploy')
    
    Returns:
        Path to the saved JSON file
    """
    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'outputs/{stage}_{timestamp}.json'
    
    # Convert any non-serializable types
    def _default(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return str(obj)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=_default)
    
    print(f"[SAVE] Results written to {path}")
    return path
```

Call this at the end of `run_full_step6()`, `run_bar_by_bar_backtest()`, and the deployment checklist.

---

### P5-F — Vectorize CUSUM Detector

**File:** `research/event_definitions.py`  
**Function:** wherever the CUSUM detector iterates over all 1.2M bars with a Python for-loop  
**Severity:** LOW (performance only)

**Replace the Python loop:**
```python
events = []
s_pos, s_neg = 0.0, 0.0
for i, row in df.iterrows():
    s_pos = max(0, s_pos + row['z'] - h)
    s_neg = max(0, s_neg - row['z'] - h)
    if s_pos > threshold or s_neg > threshold:
        events.append(i)
        s_pos, s_neg = 0.0, 0.0
```

**Note:** True CUSUM with reset-on-trigger cannot be fully vectorized with `np.cumsum` alone because each reset depends on prior state. The most practical speedup is to use Numba:

```python
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

@njit
def _cusum_loop(z_arr, h, threshold):
    """Numba-accelerated CUSUM event detector."""
    events = []
    s_pos  = 0.0
    s_neg  = 0.0
    for i in range(len(z_arr)):
        s_pos = max(0.0, s_pos + z_arr[i] - h)
        s_neg = max(0.0, s_neg - z_arr[i] - h)
        if s_pos > threshold or s_neg > threshold:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0
    return events

def detect_cusum_events(df, h=0.5, threshold=1.0):
    z_arr = df['z_score'].values.astype(np.float64)
    if HAS_NUMBA:
        event_indices = _cusum_loop(z_arr, h, threshold)
    else:
        # Fallback: pure Python (slow but correct)
        event_indices = _cusum_loop.py_func(z_arr, h, threshold)
    return df.index[event_indices]
```

---

## VALIDATION CHECKLIST

After completing all Priority 1–3 fixes, run the full pipeline from scratch and verify every item:

### Post-P1 Validation (required before trusting any number)

- [ ] `wf_feature_survival.csv` regenerated — all `ic_decay` values ≤ 1.0
- [ ] `session_range_position` feature importance has decreased from rank #1 (if still rank #1 with a dramatically lower IC, the feature is genuinely informative; if it drops to bottom half, it was mostly noise from lookahead)
- [ ] Model AUC (OOS, post-fix) is documented as the new baseline — do not compare to pre-fix AUC
- [ ] Sharpe ratio is computed as `(mean_daily_pnl / std_daily_pnl) * sqrt(252)` — verify with a worked example from the trade log
- [ ] All performance claims in `deployment_checklist.py` are now computed values (not hardcoded strings)

### Post-P2 Validation (backtest honesty)

- [ ] `run_bar_by_bar_backtest()` correctly processes bars chronologically and the log shows position opens and closes with realistic fills
- [ ] `in_position` flag transitions `False → True` on entry and `True → False` on exit (check trade log continuity)
- [ ] 2-bar confirmation delay is visible in the trade log: entry `bar_idx = event_bar_idx + 2`
- [ ] Holdout metrics are reported separately from WFO OOS metrics
- [ ] Single threshold value (`CANONICAL_THRESHOLD = 0.58`) imported everywhere

### Post-P3 Validation (live deployment safety)

- [ ] `_build_feature_row()` returns a 62-column DataFrame (run the unit test from P3-A)
- [ ] Paper trading session shows valid entry AND exit signals (no positions held overnight without explicit reason)
- [ ] IB contract resolves to a single front-month NQ contract (check logs for `"Resolved NQ contract"`)
- [ ] Stop loss and take profit levels are logged at entry and exit confirmation is logged at the stop or target

### Final Sign-Off Criteria

Do not use this system for live trading until:
1. IC decay ≤ 1.0 for all features (confirms no lookahead)
2. Holdout AUC > 0.52 (confirms the model retains signal after the lookahead fix)
3. Bar-by-bar backtest Sharpe (daily returns basis) > 0.5 over the holdout period
4. Paper trading P&L over ≥ 50 signals matches backtest within 20% (confirms live feature parity)
5. All items in the above checklists are checked off

---

*End of Fix Specification*  
*Generated from three-phase audit of `/Users/henrys/nq_backtest_2` — 2026-03-10*
