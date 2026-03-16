"""
NQ Strategy V2 — Feature-Driven Intraday Mean Reversion
========================================================
Built from statistically validated features (walk-forward IC analysis).

Core thesis (from data):
  - NQ mean-reverts at the 30min-8hr scale (negative IC on momentum)
  - VWAP distance is the most stable predictor (97% OOS hit rate)
  - Volatility features (ATR/NATR) are the strongest raw predictors
  - The edge is regime-dependent (stronger in range-bound markets)

Strategy logic:
  1. RTH-only (9:35-15:45 ET)
  2. Regime gate — skip only crisis volatility
  3. Composite entry signal from validated features (high threshold)
  4. Signal-based primary exit (mean reversion to zero)
  5. Wide disaster stop (4x ATR) — catastrophic protection only
  6. Trailing stop once in profit to lock in gains
  7. Full transaction cost modeling
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyConfig:
    """All tunable parameters in one place."""

    # --- Session filter ---
    rth_start_hour: int = 9
    rth_start_min: int = 35
    rth_end_hour: int = 15
    rth_end_min: int = 45
    flatten_hour: int = 15
    flatten_min: int = 50

    # --- Signal thresholds ---
    entry_threshold: float = 1.5
    exit_threshold: float = 0.2       # exit when signal reverts past this

    # --- Risk management ---
    atr_disaster_stop: float = 4.0    # wide catastrophic stop only
    atr_target_multiple: float = 5.0  # wide target — let winners run
    trailing_atr_multiple: float = 1.5  # trailing stop once profitable
    trailing_activation_atr: float = 1.0  # activate trailing after 1x ATR profit
    max_holding_bars: int = 60        # 5 hours max hold
    daily_loss_cap_dollars: float = 2000.0

    # --- Time-based stop (key for mean reversion) ---
    time_stop_bars: int = 15          # exit underwater trades after 15 bars (75 min)
    adverse_excursion_atr: float = 2.0  # fast exit if moves 2x ATR against in early bars
    adverse_excursion_bars: int = 6     # check adverse excursion in first 6 bars

    # --- Late entry filter ---
    last_entry_hour: int = 14
    last_entry_min: int = 0           # no new entries after 14:00

    # --- Position sizing ---
    contracts: int = 1               # fixed for now — isolate edge quality

    # --- Regime filter ---
    use_regime_filter: bool = True
    max_vol_regime: int = 2          # skip only crisis (3)
    use_trend_filter: bool = False   # disabled — signal handles direction
    min_rolling_sharpe: float = -2.0
    max_efficiency_ratio: float = 0.55  # skip when market is strongly trending

    # --- Costs ---
    commission_per_contract: float = 6.25
    slippage_ticks: float = 0.5
    tick_size: float = 0.25
    tick_value: float = 5.0
    point_value: float = 20.0


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: int
    entry_price: float
    exit_price: Optional[float]
    contracts: int
    exit_reason: str = ""
    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    bars_held: int = 0


class NQMeanReversionV2:
    """Vectorized intraday mean-reversion strategy for NQ futures."""

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.trades: list[Trade] = []

    def compute_composite_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Composite entry signal from walk-forward validated features.

        Feature signs from OOS IC analysis:
          vwap_distance:  IC = -0.065  -> above VWAP = short
          rsi_28:         IC = -0.034  -> high RSI = short
          pctrank_24:     IC = -0.023  -> high percentile = short
          mom_sign_6:     IC = -0.020  -> 30min up = short
          mom_sign_48:    IC = -0.023  -> 4hr up = short
          natr_12:        IC = +0.036  -> high vol = slightly bullish
          vol_mom_48:     IC = +0.021  -> rising volume = slightly bullish
        """
        signals = pd.DataFrame(index=df.index, dtype=float)

        # Mean reversion (high weight — these have best IC)
        if "vwap_distance" in df.columns:
            signals["vwap_dist"] = -self._zscore(df["vwap_distance"]) * 2.5

        if "rsi_28" in df.columns:
            signals["rsi"] = -self._zscore(df["rsi_28"]) * 1.5

        if "pctrank_24" in df.columns:
            signals["pctrank"] = -self._zscore(df["pctrank_24"]) * 1.0

        # Short-term momentum reversal
        if "log_ret_6" in df.columns:
            signals["ret_6"] = -self._zscore(df["log_ret_6"]) * 1.0

        if "log_ret_48" in df.columns:
            signals["ret_48"] = -self._zscore(df["log_ret_48"]) * 0.5

        # Volatility (supportive)
        if "natr_12" in df.columns:
            signals["natr"] = self._zscore(df["natr_12"]) * 0.3

        n_signals = signals.notna().sum(axis=1).replace(0, np.nan)
        composite = signals.sum(axis=1) / n_signals
        return composite

    def _zscore(self, series: pd.Series, window: int = 96 * 21) -> pd.Series:
        mean = series.rolling(window, min_periods=96).mean()
        std = series.rolling(window, min_periods=96).std().replace(0, np.nan)
        return ((series - mean) / std).clip(-4, 4)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run strategy, return DataFrame with positions and PnL."""
        cfg = self.config
        self.trades = []

        signal = self.compute_composite_signal(df)
        atr = df.get("atr_12", pd.Series(np.nan, index=df.index))

        # Time masks
        hour = np.array(df.index.hour)
        minute = np.array(df.index.minute)
        time_minutes = hour * 60 + minute
        rth_start = cfg.rth_start_hour * 60 + cfg.rth_start_min
        rth_end = cfg.rth_end_hour * 60 + cfg.rth_end_min
        flatten_time = cfg.flatten_hour * 60 + cfg.flatten_min

        is_trading_time = (time_minutes >= rth_start) & (time_minutes <= rth_end)
        is_flatten_time = time_minutes >= flatten_time
        last_entry_time = cfg.last_entry_hour * 60 + cfg.last_entry_min
        is_entry_allowed = (time_minutes >= rth_start) & (time_minutes <= last_entry_time)

        # Regime mask
        regime_ok = np.ones(len(df), dtype=bool)
        if cfg.use_regime_filter and "vol_regime" in df.columns:
            regime_ok &= np.asarray(df["vol_regime"] <= cfg.max_vol_regime)
        if cfg.use_trend_filter and "trend_regime" in df.columns:
            regime_ok &= np.asarray(df["trend_regime"] <= 0)
        if cfg.use_regime_filter and "rolling_sharpe_mr_63d" in df.columns:
            rs = df["rolling_sharpe_mr_63d"].values
            regime_ok &= (rs >= cfg.min_rolling_sharpe) | np.isnan(rs)
        if cfg.max_efficiency_ratio < 1.0 and "efficiency_ratio_1d" in df.columns:
            er = df["efficiency_ratio_1d"].values
            regime_ok &= (er <= cfg.max_efficiency_ratio) | np.isnan(er)

        # Arrays for speed
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
        target_px = 0.0
        trailing_stop = 0.0
        trailing_active = False
        best_px = 0.0  # best price seen since entry (for trailing)
        bars_held = 0
        day_pnl = 0.0
        cur_date = None
        entry_time_idx = None

        for i in range(1, n):
            if dates[i] != cur_date:
                cur_date = dates[i]
                day_pnl = 0.0

            # --- EXIT ---
            if pos != 0:
                bars_held += 1
                exit_reason = ""
                exit_px = 0.0
                direction = 1 if pos > 0 else -1

                # Update trailing stop
                if direction == 1:
                    if high_arr[i] > best_px:
                        best_px = high_arr[i]
                else:
                    if low_arr[i] < best_px:
                        best_px = low_arr[i]

                cur_atr_at_entry = atr_vals[min(i, n-1)]
                if np.isnan(cur_atr_at_entry) or cur_atr_at_entry <= 0:
                    cur_atr_at_entry = abs(entry_px) * 0.002  # fallback

                # Activate trailing once in profit by activation_atr
                if not trailing_active:
                    unrealized = direction * (close[i] - entry_px)
                    if unrealized >= cfg.trailing_activation_atr * cur_atr_at_entry:
                        trailing_active = True
                        if direction == 1:
                            trailing_stop = best_px - cfg.trailing_atr_multiple * cur_atr_at_entry
                        else:
                            trailing_stop = best_px + cfg.trailing_atr_multiple * cur_atr_at_entry

                if trailing_active:
                    if direction == 1:
                        new_trail = best_px - cfg.trailing_atr_multiple * cur_atr_at_entry
                        if new_trail > trailing_stop:
                            trailing_stop = new_trail
                    else:
                        new_trail = best_px + cfg.trailing_atr_multiple * cur_atr_at_entry
                        if new_trail < trailing_stop:
                            trailing_stop = new_trail

                # 0a. Adverse excursion fast exit — bail if trend against us
                if not exit_reason and bars_held <= cfg.adverse_excursion_bars:
                    adverse_move = direction * (close[i] - entry_px)
                    if adverse_move <= -cfg.adverse_excursion_atr * cur_atr_at_entry:
                        exit_px = close[i]
                        exit_reason = "adverse_excursion"

                # 0b. Time stop — if still underwater after N bars, cut the trade
                if not exit_reason and bars_held >= cfg.time_stop_bars:
                    unrealized = direction * (close[i] - entry_px)
                    if unrealized < 0:
                        exit_px = close[i]
                        exit_reason = "time_stop"

                # 1. Disaster stop
                if not exit_reason and direction == 1 and low_arr[i] <= disaster_stop:
                    exit_px = disaster_stop
                    exit_reason = "disaster_stop"
                elif not exit_reason and direction == -1 and high_arr[i] >= disaster_stop:
                    exit_px = disaster_stop
                    exit_reason = "disaster_stop"

                # 2. Trailing stop (only if active)
                if not exit_reason and trailing_active:
                    if direction == 1 and low_arr[i] <= trailing_stop:
                        exit_px = trailing_stop
                        exit_reason = "trailing_stop"
                    elif direction == -1 and high_arr[i] >= trailing_stop:
                        exit_px = trailing_stop
                        exit_reason = "trailing_stop"

                # 3. Take profit
                if not exit_reason:
                    if direction == 1 and high_arr[i] >= target_px:
                        exit_px = target_px
                        exit_reason = "take_profit"
                    elif direction == -1 and low_arr[i] <= target_px:
                        exit_px = target_px
                        exit_reason = "take_profit"

                # 4. Signal exit — primary exit for mean reversion
                if not exit_reason and not np.isnan(sig[i]):
                    if direction == 1 and sig[i] < cfg.exit_threshold:
                        exit_px = close[i]
                        exit_reason = "signal_exit"
                    elif direction == -1 and sig[i] > -cfg.exit_threshold:
                        exit_px = close[i]
                        exit_reason = "signal_exit"

                # 5. Max hold
                if not exit_reason and bars_held >= cfg.max_holding_bars:
                    exit_px = close[i]
                    exit_reason = "max_hold"

                # 6. EOD flatten
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
                    entry_px = 0.0
                    trailing_active = False
                    bars_held = 0

            # --- ENTRY ---
            if pos == 0 and is_entry_allowed[i] and not is_flatten_time[i] and regime_ok[i]:
                if day_pnl <= -cfg.daily_loss_cap_dollars:
                    position[i] = 0
                    continue

                cur_atr = atr_vals[i]
                if np.isnan(cur_atr) or cur_atr <= 0 or np.isnan(sig[i]):
                    position[i] = 0
                    continue

                nc = cfg.contracts

                if sig[i] >= cfg.entry_threshold:
                    pos = nc
                    entry_px = close[i]
                    disaster_stop = entry_px - cur_atr * cfg.atr_disaster_stop
                    target_px = entry_px + cur_atr * cfg.atr_target_multiple
                    trailing_stop = 0.0
                    trailing_active = False
                    best_px = high_arr[i]
                    bars_held = 0
                    entry_time_idx = i

                elif sig[i] <= -cfg.entry_threshold:
                    pos = -nc
                    entry_px = close[i]
                    disaster_stop = entry_px + cur_atr * cfg.atr_disaster_stop
                    target_px = entry_px - cur_atr * cfg.atr_target_multiple
                    trailing_stop = 0.0
                    trailing_active = False
                    best_px = low_arr[i]
                    bars_held = 0
                    entry_time_idx = i

            position[i] = pos

        result = df.copy()
        result["signal"] = signal
        result["position"] = position
        result["trade_pnl"] = trade_pnl
        result["cum_pnl"] = np.cumsum(trade_pnl)
        return result

    def performance_report(self) -> dict:
        """Compute performance statistics from completed trades."""
        if not self.trades:
            return {"error": "No trades"}

        cfg = self.config
        trades_df = pd.DataFrame([{
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "contracts": t.contracts,
            "exit_reason": t.exit_reason,
            "pnl_gross": t.pnl_gross,
            "pnl_net": t.pnl_net,
            "commission": t.commission,
            "slippage": t.slippage,
            "bars_held": t.bars_held,
        } for t in self.trades])

        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_time"]).dt.date
        trades_df["entry_year"] = pd.to_datetime(trades_df["entry_time"]).dt.year

        n_trades = len(trades_df)
        total_gross = trades_df["pnl_gross"].sum()
        total_commission = trades_df["commission"].sum()
        total_slippage = trades_df["slippage"].sum()
        total_net = trades_df["pnl_net"].sum()

        winners = trades_df[trades_df["pnl_net"] > 0]
        losers = trades_df[trades_df["pnl_net"] < 0]
        win_rate = len(winners) / n_trades if n_trades > 0 else 0
        avg_win = winners["pnl_net"].mean() if len(winners) > 0 else 0
        avg_loss = losers["pnl_net"].mean() if len(losers) > 0 else 0
        profit_factor = (
            winners["pnl_net"].sum() / abs(losers["pnl_net"].sum())
            if len(losers) > 0 and losers["pnl_net"].sum() != 0 else float("inf")
        )
        expectancy = trades_df["pnl_net"].mean()

        cum_pnl = trades_df["pnl_net"].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_dd = drawdown.min()

        first_date = trades_df["entry_time"].min()
        last_date = trades_df["exit_time"].max()
        years = (last_date - first_date).days / 365.25
        ann_return = total_net / max(years, 1)

        daily_pnl = trades_df.groupby("entry_date")["pnl_net"].sum()
        daily_sharpe = (
            daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
            if daily_pnl.std() > 0 else 0
        )

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        exit_reasons = trades_df.groupby("exit_reason").agg(
            count=("pnl_net", "count"),
            total_pnl=("pnl_net", "sum"),
            avg_pnl=("pnl_net", "mean"),
            win_rate=("pnl_net", lambda x: (x > 0).mean()),
        )

        yearly = trades_df.groupby("entry_year").agg(
            n_trades=("pnl_net", "count"),
            gross_pnl=("pnl_gross", "sum"),
            net_pnl=("pnl_net", "sum"),
            win_rate=("pnl_net", lambda x: (x > 0).mean()),
            avg_trade=("pnl_net", "mean"),
        )

        longs = trades_df[trades_df["direction"] == 1]
        shorts = trades_df[trades_df["direction"] == -1]

        return {
            "n_trades": n_trades,
            "total_gross_pnl": total_gross,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_net_pnl": total_net,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_drawdown": max_dd,
            "daily_sharpe": daily_sharpe,
            "annualized_return": ann_return,
            "calmar": calmar,
            "years": years,
            "avg_bars_held": trades_df["bars_held"].mean(),
            "n_long": len(longs),
            "n_short": len(shorts),
            "long_pnl": longs["pnl_net"].sum() if len(longs) > 0 else 0,
            "short_pnl": shorts["pnl_net"].sum() if len(shorts) > 0 else 0,
            "exit_reasons": exit_reasons,
            "yearly": yearly,
            "trades_df": trades_df,
        }
