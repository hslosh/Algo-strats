"""
Shared configuration constants for the NQ ORB ML pipeline.
============================================================

Single source of truth for all canonical trading and model parameters.
Every module should import from here rather than defining local values.
"""

# --- Canonical trading constants ---
CANONICAL_THRESHOLD    = 0.58    # Minimum calibrated probability to enter
TICK_SIZE              = 0.25    # NQ/MNQ tick size in index points
NQ_MULTIPLIER          = 20      # Dollars per point (NQ full-size)
MNQ_MULTIPLIER         = 2       # Dollars per point (Micro NQ)
COMMISSION_PER_SIDE    = 4.50    # Per contract per fill ($9.00 round-trip)
SLIPPAGE_TICKS         = 1       # Ticks of slippage per fill
STOP_LOSS_ATR_MULT     = 1.0     # SL = entry - 1.0 × ATR14
TAKE_PROFIT_ATR_MULT   = 1.5     # TP = entry + 1.5 × ATR14
MAX_POSITION_BARS      = 48      # Timeout after 48 bars (4 hours at 5-min)
AVOID_LAST_N_MINUTES   = 30      # Flatten 30 min before RTH close (15:30 ET)
DAILY_LOSS_CAP_TICKS   = 50      # Stop trading for the day after 50 ticks loss
MAX_TRADES_PER_DAY     = 3       # Maximum entries per session

# --- Walk-forward optimization ---
WFO_MIN_TRAIN_EVENTS   = 200     # Minimum events in expanding train window
WFO_TEST_MONTHS        = 6       # Test block length
WFO_EMBARGO_DAYS       = 5       # Gap between train and test

# --- Regime filter ---
REGIME_EMA_PERIOD      = 50            # Daily EMA period for trend regime

# --- FOMC dates (2024-2026) ---
import datetime as _dt
FOMC_DATES = {
    # 2024
    _dt.date(2024, 1, 31), _dt.date(2024, 3, 20), _dt.date(2024, 5, 1),
    _dt.date(2024, 6, 12), _dt.date(2024, 7, 31), _dt.date(2024, 9, 18),
    _dt.date(2024, 11, 7), _dt.date(2024, 12, 18),
    # 2025
    _dt.date(2025, 1, 29), _dt.date(2025, 3, 19), _dt.date(2025, 5, 7),
    _dt.date(2025, 6, 18), _dt.date(2025, 7, 30), _dt.date(2025, 9, 17),
    _dt.date(2025, 10, 29), _dt.date(2025, 12, 17),
    # 2026
    _dt.date(2026, 1, 28), _dt.date(2026, 3, 18), _dt.date(2026, 4, 29),
    _dt.date(2026, 6, 17), _dt.date(2026, 7, 29), _dt.date(2026, 9, 16),
    _dt.date(2026, 10, 28), _dt.date(2026, 12, 16),
}

def is_high_impact_day(event_date) -> bool:
    """Check if a date is a high-impact day (FOMC announcement)."""
    if hasattr(event_date, 'date'):
        event_date = event_date.date()
    return event_date in FOMC_DATES

# --- Holdout ---
HOLDOUT_START_DATE_STR = '2025-07-01'  # Canonical holdout separation date
