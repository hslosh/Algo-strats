"""
Feature Engineering Pipeline for NQ Futures
============================================
Transforms raw OHLCV data into statistically testable predictive features.

Feature categories:
  1. Returns & Momentum
  2. Volatility (multiple estimators)
  3. Volume dynamics
  4. Microstructure (candle anatomy)
  5. Mean reversion / stretch
  6. Cross-timeframe (multi-resolution)
  7. Temporal / calendar
  8. Open Interest (daily only)

All features are computed as pure functions of past data (no lookahead).
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# 1. RETURNS & MOMENTUM
# ---------------------------------------------------------------------------

def add_return_features(df: pd.DataFrame, periods: list[int] = None) -> pd.DataFrame:
    """Log returns and momentum over multiple lookback windows."""
    if periods is None:
        periods = [1, 3, 5, 10, 20, 60]

    for p in periods:
        # Log return over p bars
        df[f"log_ret_{p}"] = np.log(df["close"] / df["close"].shift(p))

        # Momentum: sign and magnitude separation
        df[f"mom_sign_{p}"] = np.sign(df[f"log_ret_{p}"])

    # Rate of change of momentum (momentum acceleration)
    if 5 in periods and 20 in periods:
        df["mom_accel"] = df["log_ret_5"] - df["log_ret_5"].shift(5)

    # Consecutive up/down bars
    df["bar_direction"] = np.sign(df["close"] - df["open"])
    df["consec_up"] = _consecutive_count(df["bar_direction"], 1)
    df["consec_down"] = _consecutive_count(df["bar_direction"], -1)

    return df


def _consecutive_count(series: pd.Series, value) -> pd.Series:
    """Count consecutive occurrences of a value."""
    groups = (series != value).cumsum()
    result = series.groupby(groups).cumcount() + 1
    result[series != value] = 0
    return result


# ---------------------------------------------------------------------------
# 2. VOLATILITY
# ---------------------------------------------------------------------------

def add_volatility_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Multiple volatility estimators — each captures different information."""
    if windows is None:
        windows = [5, 10, 20, 60]

    log_ret = np.log(df["close"] / df["close"].shift(1))
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    for w in windows:
        # Close-to-close (standard) realized volatility
        df[f"rvol_cc_{w}"] = log_ret.rolling(w).std()

        # Parkinson (high-low based — more efficient estimator)
        df[f"rvol_parkinson_{w}"] = np.sqrt(
            (1 / (4 * np.log(2))) * (log_hl ** 2).rolling(w).mean()
        )

        # Garman-Klass (uses OHLC — most efficient estimator)
        gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        df[f"rvol_gk_{w}"] = np.sqrt(gk.rolling(w).mean())

        # ATR (Average True Range)
        tr = _true_range(df)
        df[f"atr_{w}"] = tr.rolling(w).mean()

        # ATR normalized by price (comparable across price levels)
        df[f"natr_{w}"] = df[f"atr_{w}"] / df["close"]

    # Volatility ratio: short-term vs long-term (regime detection)
    if 5 in windows and 20 in windows:
        df["vol_ratio_5_20"] = df["rvol_cc_5"] / df["rvol_cc_20"].replace(0, np.nan)

    if 10 in windows and 60 in windows:
        df["vol_ratio_10_60"] = df["rvol_cc_10"] / df["rvol_cc_60"].replace(0, np.nan)

    # Volatility z-score (is current vol high or low relative to recent history?)
    if 20 in windows and 60 in windows:
        df["vol_zscore"] = (
            (df["rvol_cc_20"] - df["rvol_cc_60"])
            / df["rvol_cc_60"].replace(0, np.nan)
        )

    return df


from research_utils.utils import true_range as _true_range


# ---------------------------------------------------------------------------
# 3. VOLUME DYNAMICS
# ---------------------------------------------------------------------------

def add_volume_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Volume-based features — liquidity, participation, divergences."""
    if windows is None:
        windows = [5, 10, 20, 60]

    for w in windows:
        # Relative volume (current bar vs rolling average)
        avg_vol = df["volume"].rolling(w).mean()
        df[f"rvol_ratio_{w}"] = df["volume"] / avg_vol.replace(0, np.nan)

        # Volume momentum (is volume increasing or decreasing?)
        df[f"vol_mom_{w}"] = np.log(
            df["volume"].rolling(w).mean()
            / df["volume"].rolling(w * 2).mean().replace(0, np.nan)
        )

    # VWAP (intraday anchored — resets each session)
    df["vwap"] = _session_vwap(df)
    df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)

    # Volume-price divergence: price up but volume declining
    if 10 in windows:
        price_dir = np.sign(df["close"] - df["close"].shift(10))
        vol_dir = np.sign(df["volume"].rolling(10).mean() - df["volume"].rolling(20).mean())
        df["vol_price_diverge"] = (price_dir != vol_dir).astype(int)

    # On Balance Volume (cumulative)
    direction = np.sign(df["close"] - df["close"].shift(1))
    df["obv"] = (direction * df["volume"]).cumsum()
    # OBV slope
    if 20 in windows:
        df["obv_slope_20"] = df["obv"] - df["obv"].shift(20)

    return df


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP. Resets at the start of each trading day."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    if not isinstance(df.index, pd.DatetimeIndex):
        # If no datetime index, compute global cumulative VWAP
        cum_tpv = tp_vol.cumsum()
        cum_vol = df["volume"].cumsum()
        return cum_tpv / cum_vol.replace(0, np.nan)

    # Group by date for session-anchored VWAP
    dates = df.index.date
    cum_tpv = tp_vol.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 4. MICROSTRUCTURE (Candle Anatomy)
# ---------------------------------------------------------------------------

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bar-level microstructure features from OHLC anatomy."""

    bar_range = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    # Body-to-range ratio (0 = doji, 1 = full body)
    df["body_ratio"] = body / bar_range.replace(0, np.nan)

    # Upper/lower wick ratios (rejection signals)
    df["upper_wick_ratio"] = upper_wick / bar_range.replace(0, np.nan)
    df["lower_wick_ratio"] = lower_wick / bar_range.replace(0, np.nan)

    # Bar range relative to recent average (expansion/contraction)
    df["range_ratio_10"] = bar_range / bar_range.rolling(10).mean().replace(0, np.nan)
    df["range_ratio_20"] = bar_range / bar_range.rolling(20).mean().replace(0, np.nan)

    # Close position within bar (0 = closed at low, 1 = closed at high)
    df["close_position"] = (df["close"] - df["low"]) / bar_range.replace(0, np.nan)

    # Gap: open vs previous close
    df["gap"] = df["open"] - df["close"].shift(1)
    df["gap_pct"] = df["gap"] / df["close"].shift(1)

    return df


# ---------------------------------------------------------------------------
# 5. MEAN REVERSION / STRETCH
# ---------------------------------------------------------------------------

def add_mean_reversion_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Features measuring how stretched price is from equilibrium."""
    if windows is None:
        windows = [10, 20, 60]

    for w in windows:
        ma = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()

        # Distance from moving average (normalized)
        df[f"dist_ma_{w}"] = (df["close"] - ma) / ma.replace(0, np.nan)

        # Bollinger Band z-score
        df[f"bb_zscore_{w}"] = (df["close"] - ma) / std.replace(0, np.nan)

        # Percentile rank of close within rolling window (vectorized)
        df[f"pctrank_{w}"] = df["close"].rolling(w).rank(pct=True)

    # RSI (Relative Strength Index) — classic mean reversion signal
    for period in [14, 28]:
        df[f"rsi_{period}"] = _rsi(df["close"], period)

    return df


def _rsi(prices: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# 6. TEMPORAL / CALENDAR
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-of-day, day-of-week, session classification."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    # Hour of day (cyclical encoding)
    hour = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day of week (cyclical encoding)
    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # Session classification (US Eastern Time assumed)
    hour_raw = df.index.hour
    df["session"] = "overnight"
    df.loc[(hour_raw >= 4) & (hour_raw < 9), "session"] = "premarket"
    df.loc[(hour_raw >= 9) & (hour_raw < 10), "session"] = "open_cross"
    df.loc[(hour_raw >= 10) & (hour_raw < 15), "session"] = "midday"
    df.loc[(hour_raw >= 15) & (hour_raw < 16), "session"] = "close_cross"
    df.loc[(hour_raw == 16), "session"] = "postmarket"

    # Binary flags for key sessions
    df["is_rth"] = ((hour_raw >= 9) & (hour_raw < 16)).astype(int)
    df["is_open_30min"] = ((hour_raw == 9) & (df.index.minute < 30)).astype(int)
    df["is_close_30min"] = ((hour_raw == 15) & (df.index.minute >= 30)).astype(int)

    # Minutes since RTH open (useful for intraday patterns)
    rth_start_minutes = 9 * 60 + 30
    minutes_of_day = hour_raw * 60 + df.index.minute
    df["mins_since_open"] = np.clip(minutes_of_day - rth_start_minutes, 0, None)

    return df


# ---------------------------------------------------------------------------
# 7. OPEN INTEREST (daily data only)
# ---------------------------------------------------------------------------

def add_open_interest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features from open interest — only for daily bars."""
    if "open interest" not in df.columns:
        return df

    oi = df["open interest"]

    # OI change (absolute and percentage)
    df["oi_change"] = oi.diff()
    df["oi_change_pct"] = oi.pct_change()

    # Volume / OI ratio (participation rate)
    df["vol_oi_ratio"] = df["volume"] / oi.replace(0, np.nan)

    # OI momentum
    for w in [3, 5]:
        df[f"oi_ma_{w}"] = oi.rolling(w).mean()
        df[f"oi_above_ma_{w}"] = (oi > df[f"oi_ma_{w}"]).astype(int)

    return df


# ---------------------------------------------------------------------------
# 8. REGIME DETECTION
# ---------------------------------------------------------------------------

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect market regime from price/vol structure.
    Regimes gate the strategy on/off to avoid trading in hostile conditions.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))

    # --- Volatility regime ---
    # Rolling 1-day realized vol (96 bars at 5min)
    rvol_1d = log_ret.rolling(96).std()
    # Rolling 1-month realized vol
    rvol_1m = log_ret.rolling(96 * 21).std()
    # Vol-of-vol (stability of volatility itself)
    df["vol_of_vol"] = rvol_1d.rolling(96 * 5).std()

    # Volatility regime: low / normal / high / crisis
    vol_pctile = rvol_1d.rolling(96 * 252).rank(pct=True)
    df["vol_regime"] = 1  # normal
    df.loc[vol_pctile < 0.25, "vol_regime"] = 0   # low vol
    df.loc[vol_pctile > 0.75, "vol_regime"] = 2   # high vol
    df.loc[vol_pctile > 0.95, "vol_regime"] = 3   # crisis

    # --- Trend regime ---
    # ADX-like measure: ratio of directional move to total distance traveled
    abs_move = (df["close"] - df["close"].shift(96)).abs()
    total_path = log_ret.abs().rolling(96).sum() * df["close"]
    df["efficiency_ratio_1d"] = abs_move / total_path.replace(0, np.nan)

    abs_move_5d = (df["close"] - df["close"].shift(96 * 5)).abs()
    total_path_5d = log_ret.abs().rolling(96 * 5).sum() * df["close"]
    df["efficiency_ratio_5d"] = abs_move_5d / total_path_5d.replace(0, np.nan)

    # Trending when efficiency ratio is high (>0.3 = strong trend)
    df["trend_regime"] = 0  # range-bound
    df.loc[df["efficiency_ratio_1d"] > 0.3, "trend_regime"] = 1   # trending

    # --- Rolling Sharpe (strategy performance regime) ---
    # This tells us whether mean-reversion has been working recently
    # We compute it on the raw RSI-based signal as a proxy
    rsi = df.get("rsi_28")
    if rsi is not None:
        # Simple mean-reversion return proxy: short when RSI high, long when low
        mr_signal = -(rsi - 50) / 50  # normalized to [-1, 1]
        mr_ret = mr_signal.shift(1) * log_ret
        df["rolling_sharpe_mr_21d"] = (
            mr_ret.rolling(96 * 21).mean()
            / mr_ret.rolling(96 * 21).std().replace(0, np.nan)
        ) * np.sqrt(96 * 252)

        df["rolling_sharpe_mr_63d"] = (
            mr_ret.rolling(96 * 63).mean()
            / mr_ret.rolling(96 * 63).std().replace(0, np.nan)
        ) * np.sqrt(96 * 252)

    return df


# ---------------------------------------------------------------------------
# 9. CROSS-TIMEFRAME FEATURES
# ---------------------------------------------------------------------------

def add_cross_timeframe_features(
    df_fast: pd.DataFrame,
    df_slow: pd.DataFrame,
    slow_label: str = "1h"
) -> pd.DataFrame:
    """
    Merge higher-timeframe context into lower-timeframe data.
    Uses forward-fill to avoid lookahead (each fast bar gets the most
    recent completed slow bar's features).
    """
    # Compute some features on the slow timeframe
    slow = df_slow[["close", "high", "low", "volume"]].copy()
    slow[f"ret_1_{slow_label}"] = np.log(slow["close"] / slow["close"].shift(1))
    slow[f"atr_5_{slow_label}"] = _true_range_from_df(slow).rolling(5).mean()
    slow[f"rvol_cc_5_{slow_label}"] = np.log(slow["close"] / slow["close"].shift(1)).rolling(5).std()
    slow[f"rsi_14_{slow_label}"] = _rsi(slow["close"], 14)

    # Keep only the derived features
    merge_cols = [c for c in slow.columns if slow_label in c]
    slow_features = slow[merge_cols]

    # Merge via index (forward-fill = no lookahead)
    df_out = df_fast.join(slow_features, how="left")
    df_out[merge_cols] = df_out[merge_cols].ffill()

    return df_out


_true_range_from_df = _true_range  # Alias for backward compatibility


# ---------------------------------------------------------------------------
# TARGET VARIABLES
# ---------------------------------------------------------------------------

def add_targets(df: pd.DataFrame, horizons: list[int] = None) -> pd.DataFrame:
    """
    Forward-looking return targets at multiple horizons.
    These are what we're trying to predict. Only used for analysis/training,
    never as input features during live trading.
    """
    if horizons is None:
        horizons = [1, 3, 5, 10, 20, 60]

    for h in horizons:
        # Forward log return
        df[f"target_ret_{h}"] = np.log(df["close"].shift(-h) / df["close"])

        # Forward direction (binary classification target)
        df[f"target_dir_{h}"] = (df[f"target_ret_{h}"] > 0).astype(int)

        # Forward max adverse excursion (worst drawdown in next h bars)
        df[f"target_mae_{h}"] = (
            df["low"].rolling(h).min().shift(-h) - df["close"]
        ) / df["close"]

        # Forward max favorable excursion (best upside in next h bars)
        df[f"target_mfe_{h}"] = (
            df["high"].rolling(h).max().shift(-h) - df["close"]
        ) / df["close"]

    return df


# ---------------------------------------------------------------------------
# MASTER PIPELINE
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    add_targets_flag: bool = True,
    return_periods: list[int] = None,
    vol_windows: list[int] = None,
    volume_windows: list[int] = None,
    mr_windows: list[int] = None,
    target_horizons: list[int] = None,
) -> pd.DataFrame:
    """
    Master pipeline: applies all feature groups to a single-timeframe DataFrame.

    Parameters
    ----------
    df : OHLCV DataFrame with DatetimeIndex
    add_targets_flag : whether to add forward-looking targets
    """
    df = df.copy()

    df = add_return_features(df, return_periods)
    df = add_volatility_features(df, vol_windows)
    df = add_volume_features(df, volume_windows)
    df = add_microstructure_features(df)
    df = add_mean_reversion_features(df, mr_windows)
    df = add_temporal_features(df)
    df = add_regime_features(df)
    df = add_open_interest_features(df)

    if add_targets_flag:
        df = add_targets(df, target_horizons)

    return df


# ---------------------------------------------------------------------------
# TREND REGIME FILTER
# ---------------------------------------------------------------------------

def add_trend_regime(df: pd.DataFrame, ema_period: int = 50) -> pd.DataFrame:
    """Add regime_long_allowed column based on prior day's close vs daily EMA.

    Uses RTH close (last bar 15:55-16:00 ET) to compute daily close.
    shift(1) ensures we use prior day's value (no lookahead).
    First day defaults to True (permissive).
    """
    rth_mask = df.index.hour * 60 + df.index.minute
    # RTH = 09:30 to 15:55 (last bar before 16:00)
    is_rth = (rth_mask >= 570) & (rth_mask < 960)  # 9*60+30=570, 16*60=960

    rth_df = df.loc[is_rth].copy()
    # Daily close = last RTH bar's close per date
    daily_close = rth_df.groupby(rth_df.index.date)['close'].last()
    daily_close = daily_close.sort_index()

    # Compute EMA on daily close
    daily_ema = daily_close.ewm(span=ema_period, adjust=False).mean()

    # Regime: close > EMA, shifted by 1 day (use prior day's value)
    regime_signal = (daily_close > daily_ema).shift(1)

    # Map back to 5-min bars by date
    date_to_regime = regime_signal.to_dict()
    df['regime_long_allowed'] = df.index.date
    df['regime_long_allowed'] = df['regime_long_allowed'].map(date_to_regime)

    # First day (no prior) defaults to True
    df['regime_long_allowed'] = df['regime_long_allowed'].fillna(True).astype(bool)

    return df


# ---------------------------------------------------------------------------
# DATA LOADING HELPERS
# ---------------------------------------------------------------------------

def load_ohlcv(filepath: str, has_oi: bool = False) -> pd.DataFrame:
    """Load a FirstRateData CSV into a clean DataFrame with DatetimeIndex.
    Handles both header and headerless formats."""
    # Sniff whether file has a header
    with open(filepath, "r") as f:
        first_line = f.readline().strip()

    has_header = not first_line[0].isdigit()

    if has_header:
        df = pd.read_csv(filepath, parse_dates=[0])
        df.columns = [c.strip().lower() for c in df.columns]
    else:
        # Headerless: timestamp,O,H,L,C,V[,OI]
        col_names = ["timestamp", "open", "high", "low", "close", "volume"]
        if has_oi:
            col_names.append("open interest")
        df = pd.read_csv(filepath, header=None, names=col_names, parse_dates=[0])

    ts_col = df.columns[0] if df.columns[0] in ("timestamp",) else df.columns[0]
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})

    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df
