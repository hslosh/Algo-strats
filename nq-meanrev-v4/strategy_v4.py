"""
NQ Strategy V4 — Adaptive Mean Reversion with Surgical Exit Logic
=================================================================
Improvements over V2_MFE (best result: Sharpe -0.40, -$13K/17yr):

1. ADAPTIVE ENTRY THRESHOLD — Vol-scaled selectivity
   High-vol regimes require stronger signals (fewer bad entries).
   threshold_effective = base_threshold × (1 + vol_premium)
   where vol_premium = max(0, rvol_ratio - 1) × vol_scale_factor

2. EARLY PRE-DISASTER CUT — Catch doomed trades before disaster stop
   If unrealized < -1.5 ATR AND bars_held >= 3 AND MFE < 0.2 ATR → exit.
   Replaces 35 disaster stops (-$61K avg -$1,743) with earlier, smaller losses.

3. PROGRESSIVE TRAILING TIGHTENING — Lock in more as profit grows
   Start at trail_atr_wide, tighten to trail_atr_tight as unrealized grows.
   Captures more profit on strong moves, lets breathing room on early moves.

4. DAILY TRADE LIMIT — Max N losing trades per day
   After max_daily_losers consecutive losing trades, stop for the day.
   Prevents revenge trading in hostile environments.

5. SMARTER SIGNAL DECAY — Only exit if signal flipped, not just decayed
   Instead of exit when signal < 30% of entry, exit when signal has
   actually crossed zero (thesis reversal, not just weakening).

Core architecture from V2 preserved:
  - Composite z-score signal from walk-forward validated features
  - RTH-only trading with late-entry cutoff
  - MFE tracking for thesis-failed detection
  - 6-bar entry cooldown after exit
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from strategy_v2 import StrategyConfig, Trade, NQMeanReversionV2


@dataclass
class V4Config(StrategyConfig):
    """V4 extends StrategyConfig with adaptive parameters."""

    # --- Adaptive threshold ---
    vol_scale_factor: float = 0.5       # how much vol inflates threshold (0=disabled)
    vol_lookback_bars: int = 96         # 1 day for current vol
    vol_baseline_bars: int = 96 * 21    # 21 days for baseline vol

    # --- Early pre-disaster cut ---
    early_cut_bars: int = 4             # check after this many bars
    early_cut_loss_atr: float = 1.5     # loss threshold in ATR
    early_cut_mfe_atr: float = 0.2      # MFE must be below this

    # --- Progressive trailing ---
    trail_atr_wide: float = 2.0         # initial trailing distance
    trail_atr_tight: float = 1.0        # tighten to this at high profit
    trail_tighten_profit_atr: float = 3.0  # profit level to reach full tightening

    # --- Daily trade limit ---
    max_daily_losers: int = 2           # stop after N losing trades in a day

    # --- Signal reversal exit ---
    use_signal_reversal: bool = True    # exit when signal crosses zero while underwater


class NQMeanRevV4(NQMeanReversionV2):
    """V4: Adaptive threshold + progressive trailing + early cut + MFE exits."""

    def __init__(self, config: V4Config = None):
        self.config = config or V4Config()
        self.trades: list[Trade] = []

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        self.trades = []

        signal = self.compute_composite_signal(df)
        atr = df.get("atr_12", pd.Series(np.nan, index=df.index))

        # Pre-compute rolling vol ratio for adaptive threshold
        log_ret = np.log(df["close"] / df["close"].shift(1))
        rvol_fast = log_ret.rolling(cfg.vol_lookback_bars, min_periods=20).std()
        rvol_slow = log_ret.rolling(cfg.vol_baseline_bars, min_periods=96).std()
        vol_ratio = (rvol_fast / rvol_slow.replace(0, np.nan)).fillna(1.0).values

        # Time masks
        hour = np.array(df.index.hour)
        minute = np.array(df.index.minute)
        time_minutes = hour * 60 + minute
        rth_start = cfg.rth_start_hour * 60 + cfg.rth_start_min
        flatten_time = cfg.flatten_hour * 60 + cfg.flatten_min
        last_entry_time = cfg.last_entry_hour * 60 + cfg.last_entry_min

        is_entry_ok = (time_minutes >= rth_start) & (time_minutes <= last_entry_time)
        is_flatten_time = time_minutes >= flatten_time

        # Regime mask
        regime_ok = np.ones(len(df), dtype=bool)
        if cfg.use_regime_filter and "vol_regime" in df.columns:
            regime_ok &= np.asarray(df["vol_regime"] <= cfg.max_vol_regime)
        if cfg.max_efficiency_ratio < 1.0 and "efficiency_ratio_1d" in df.columns:
            er = df["efficiency_ratio_1d"].values
            regime_ok &= (er <= cfg.max_efficiency_ratio) | np.isnan(er)

        n = len(df)
        position = np.zeros(n, dtype=int)
        trade_pnl = np.zeros(n, dtype=float)
        close = df["close"].values
        high_arr = df["high"].values
        low_arr = df["low"].values
        sig = signal.values
        atr_vals = atr.values
        dates = df.index.date

        # State
        pos = 0
        entry_px = 0.0
        disaster_stop = 0.0
        trailing_stop = 0.0
        trailing_active = False
        best_px = 0.0
        bars_held = 0
        day_pnl = 0.0
        cur_date = None
        entry_time_idx = None
        mfe = 0.0
        entry_signal = 0.0
        last_exit_bar = -999
        daily_losers = 0     # count of losing trades today

        for i in range(1, n):
            # Day reset
            if dates[i] != cur_date:
                cur_date = dates[i]
                day_pnl = 0.0
                daily_losers = 0

            if pos != 0:
                bars_held += 1
                exit_reason = ""
                exit_px = 0.0
                direction = 1 if pos > 0 else -1

                # Track MFE
                unrealized = direction * (close[i] - entry_px)
                if unrealized > mfe:
                    mfe = unrealized

                # Update best price
                if direction == 1:
                    if high_arr[i] > best_px:
                        best_px = high_arr[i]
                else:
                    if low_arr[i] < best_px:
                        best_px = low_arr[i]

                cur_atr = atr_vals[i]
                if np.isnan(cur_atr) or cur_atr <= 0:
                    cur_atr = abs(entry_px) * 0.002

                # --- PROGRESSIVE TRAILING ---
                # Compute current trailing ATR multiple based on profit level
                if unrealized > 0 and cur_atr > 0:
                    profit_atr = unrealized / cur_atr
                    # Linear interpolation from wide to tight
                    tighten_frac = min(1.0, profit_atr / cfg.trail_tighten_profit_atr)
                    cur_trail_mult = cfg.trail_atr_wide - tighten_frac * (cfg.trail_atr_wide - cfg.trail_atr_tight)
                else:
                    cur_trail_mult = cfg.trail_atr_wide

                # Trailing activation
                if not trailing_active:
                    if unrealized >= cfg.trailing_activation_atr * cur_atr:
                        trailing_active = True
                        if direction == 1:
                            trailing_stop = best_px - cur_trail_mult * cur_atr
                        else:
                            trailing_stop = best_px + cur_trail_mult * cur_atr

                if trailing_active:
                    if direction == 1:
                        nt = best_px - cur_trail_mult * cur_atr
                        if nt > trailing_stop:
                            trailing_stop = nt
                    else:
                        nt = best_px + cur_trail_mult * cur_atr
                        if nt < trailing_stop:
                            trailing_stop = nt

                # ====== EXIT PRIORITY ======

                # 0. EARLY PRE-DISASTER CUT — catch doomed trades early
                if not exit_reason and bars_held >= cfg.early_cut_bars:
                    if (unrealized < -cfg.early_cut_loss_atr * cur_atr
                            and mfe < cfg.early_cut_mfe_atr * cur_atr):
                        exit_px = close[i]
                        exit_reason = "early_cut"

                # 1. Disaster stop (very wide — last resort)
                if not exit_reason:
                    if direction == 1 and low_arr[i] <= disaster_stop:
                        exit_px = disaster_stop
                        exit_reason = "disaster_stop"
                    elif direction == -1 and high_arr[i] >= disaster_stop:
                        exit_px = disaster_stop
                        exit_reason = "disaster_stop"

                # 2. Trailing stop
                if not exit_reason and trailing_active:
                    if direction == 1 and low_arr[i] <= trailing_stop:
                        exit_px = trailing_stop
                        exit_reason = "trailing_stop"
                    elif direction == -1 and high_arr[i] >= trailing_stop:
                        exit_px = trailing_stop
                        exit_reason = "trailing_stop"

                # 3. Signal exit (mean reversion complete)
                if not exit_reason and not np.isnan(sig[i]):
                    if direction == 1 and sig[i] < cfg.exit_threshold:
                        exit_px = close[i]
                        exit_reason = "signal_exit"
                    elif direction == -1 and sig[i] > -cfg.exit_threshold:
                        exit_px = close[i]
                        exit_reason = "signal_exit"

                # 4. MFE thesis-failed exit
                if not exit_reason and bars_held >= cfg.time_stop_bars:
                    if mfe < 0.3 * cur_atr:
                        exit_px = close[i]
                        exit_reason = "mfe_fail"

                # 5. Signal reversal exit — signal crossed zero while underwater
                if not exit_reason and cfg.use_signal_reversal and bars_held >= 4:
                    if not np.isnan(sig[i]) and unrealized < 0:
                        if direction == 1 and sig[i] < 0:
                            exit_px = close[i]
                            exit_reason = "signal_reversal"
                        elif direction == -1 and sig[i] > 0:
                            exit_px = close[i]
                            exit_reason = "signal_reversal"

                # 6. Max hold
                if not exit_reason and bars_held >= cfg.max_holding_bars:
                    exit_px = close[i]
                    exit_reason = "max_hold"

                # 7. EOD flatten
                if not exit_reason and is_flatten_time[i]:
                    exit_px = close[i]
                    exit_reason = "eod_flatten"

                if exit_reason:
                    abs_c = abs(pos)
                    gross = direction * (exit_px - entry_px) * cfg.point_value * abs_c
                    comm = cfg.commission_per_contract * abs_c
                    slip = cfg.slippage_ticks * cfg.tick_value * abs_c * 2
                    net = gross - comm - slip

                    trade_pnl[i] = net
                    day_pnl += net
                    last_exit_bar = i

                    if net < 0:
                        daily_losers += 1

                    self.trades.append(Trade(
                        entry_time=df.index[entry_time_idx] if entry_time_idx else df.index[i],
                        exit_time=df.index[i],
                        direction=direction,
                        entry_price=entry_px,
                        exit_price=exit_px,
                        contracts=abs_c,
                        exit_reason=exit_reason,
                        pnl_gross=gross,
                        pnl_net=net,
                        commission=comm,
                        slippage=slip,
                        bars_held=bars_held,
                    ))

                    pos = 0
                    trailing_active = False
                    bars_held = 0

            # --- ENTRY ---
            if pos == 0 and is_entry_ok[i] and not is_flatten_time[i] and regime_ok[i]:
                # Daily loss cap
                if day_pnl <= -cfg.daily_loss_cap_dollars:
                    position[i] = 0
                    continue

                # Daily loser cap
                if daily_losers >= cfg.max_daily_losers:
                    position[i] = 0
                    continue

                # Cooldown
                if i - last_exit_bar < 6:
                    position[i] = 0
                    continue

                cur_atr = atr_vals[i]
                if np.isnan(cur_atr) or cur_atr <= 0 or np.isnan(sig[i]):
                    position[i] = 0
                    continue

                # ADAPTIVE THRESHOLD: inflate in high-vol environments
                vr = vol_ratio[i]
                if np.isnan(vr):
                    vr = 1.0
                vol_premium = max(0.0, vr - 1.0) * cfg.vol_scale_factor
                effective_threshold = cfg.entry_threshold * (1.0 + vol_premium)

                nc = cfg.contracts

                if sig[i] >= effective_threshold:
                    pos = nc
                    entry_px = close[i]
                    disaster_stop = entry_px - cur_atr * cfg.atr_disaster_stop
                    trailing_stop = 0.0
                    trailing_active = False
                    best_px = high_arr[i]
                    bars_held = 0
                    entry_time_idx = i
                    mfe = 0.0
                    entry_signal = sig[i]

                elif sig[i] <= -effective_threshold:
                    pos = -nc
                    entry_px = close[i]
                    disaster_stop = entry_px + cur_atr * cfg.atr_disaster_stop
                    trailing_stop = 0.0
                    trailing_active = False
                    best_px = low_arr[i]
                    bars_held = 0
                    entry_time_idx = i
                    mfe = 0.0
                    entry_signal = sig[i]

            position[i] = pos

        result = df.copy()
        result["signal"] = signal
        result["position"] = position
        result["trade_pnl"] = trade_pnl
        result["cum_pnl"] = np.cumsum(trade_pnl)
        return result
