"""
V4 Refinement — Fine-tune around vol_scale=0.8 breakthrough
============================================================
Best so far: vol_scale=0.8, 95 trades, +$17K, Sharpe 2.93

Goals:
  1. Fine-tune vol_scale around 0.6-1.2 to map stability
  2. Test adding early_cut to catch remaining 5 disaster stops
  3. Test adding signal_reversal as safety net
  4. Run bootstrap confidence analysis on best config
  5. Test different base thresholds with scaled vol_scale
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

from feature_engineering import load_ohlcv, build_features
from strategy_v4 import NQMeanRevV4, V4Config
from final_sweep import NQMeanRevV2_MFE
from strategy_v2 import StrategyConfig

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "nq_continuous_5m_converted.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_one(df, strat_cls, cfg, label=""):
    strat = strat_cls(cfg)
    strat.run(df)
    if not strat.trades:
        return None
    rpt = strat.performance_report()
    if "error" in rpt:
        return None
    rpt["label"] = label
    rpt["trades_per_year"] = rpt["n_trades"] / max(rpt["years"], 1)
    yearly = rpt["yearly"]
    rpt["pos_years"] = f"{(yearly['net_pnl'] > 0).sum()}/{len(yearly)}"
    return rpt


def make_v4_config(base_thresh=3.0, vol_scale=0.8, early_cut=False,
                   sig_reversal=False, early_cut_bars=4, early_cut_loss=1.5,
                   early_cut_mfe=0.2, disaster=6.0, trail_wide=2.0,
                   trail_tight=2.0, trail_profit=99.0, daily_losers=99,
                   max_hold=72, activation=1.5):
    return V4Config(
        entry_threshold=base_thresh, exit_threshold=0.2,
        atr_disaster_stop=disaster, atr_target_multiple=10.0,
        trailing_atr_multiple=trail_wide, trailing_activation_atr=activation,
        time_stop_bars=20, adverse_excursion_atr=99.0,
        last_entry_hour=14, last_entry_min=0,
        max_efficiency_ratio=1.0, max_holding_bars=max_hold,
        vol_scale_factor=vol_scale,
        early_cut_bars=early_cut_bars if early_cut else 999,
        early_cut_loss_atr=early_cut_loss if early_cut else 99.0,
        early_cut_mfe_atr=early_cut_mfe if early_cut else 99.0,
        trail_atr_wide=trail_wide, trail_atr_tight=trail_tight,
        trail_tighten_profit_atr=trail_profit,
        use_signal_reversal=sig_reversal,
        max_daily_losers=daily_losers,
        contracts=1,
    )


def bootstrap_analysis(strat_cls, cfg, df, n_boot=1000, seed=42):
    """Bootstrap confidence intervals on strategy performance."""
    strat = strat_cls(cfg)
    strat.run(df)
    if not strat.trades:
        return None

    trades_pnl = np.array([t.pnl_net for t in strat.trades])
    n_trades = len(trades_pnl)

    rng = np.random.RandomState(seed)
    boot_totals = np.zeros(n_boot)
    boot_sharpes = np.zeros(n_boot)
    boot_wrs = np.zeros(n_boot)

    for b in range(n_boot):
        sample = rng.choice(trades_pnl, size=n_trades, replace=True)
        boot_totals[b] = sample.sum()
        boot_wrs[b] = (sample > 0).mean()
        if sample.std() > 0:
            boot_sharpes[b] = sample.mean() / sample.std() * np.sqrt(252 / max(1, n_trades / 17))
        else:
            boot_sharpes[b] = 0

    return {
        "n_trades": n_trades,
        "actual_total": trades_pnl.sum(),
        "actual_mean": trades_pnl.mean(),
        "total_ci_5": np.percentile(boot_totals, 5),
        "total_ci_50": np.percentile(boot_totals, 50),
        "total_ci_95": np.percentile(boot_totals, 95),
        "sharpe_ci_5": np.percentile(boot_sharpes, 5),
        "sharpe_ci_50": np.percentile(boot_sharpes, 50),
        "sharpe_ci_95": np.percentile(boot_sharpes, 95),
        "wr_ci_5": np.percentile(boot_wrs, 5),
        "wr_ci_50": np.percentile(boot_wrs, 50),
        "wr_ci_95": np.percentile(boot_wrs, 95),
        "prob_positive": (boot_totals > 0).mean(),
    }


def main():
    t0 = time.time()
    print("Loading & featurizing...")
    df = load_ohlcv(DATA_PATH)
    df = build_features(
        df, add_targets_flag=True,
        return_periods=[1, 3, 6, 12, 24, 48, 96],
        vol_windows=[12, 24, 48, 96],
        volume_windows=[12, 24, 48, 96],
        mr_windows=[24, 48, 96],
        target_horizons=[12],
    )
    print(f"  {len(df):,} bars  ({time.time()-t0:.1f}s)")

    configs = []

    # ================================================================
    # SECTION 1: Fine-tune vol_scale (stability mapping)
    # ================================================================
    for vs in [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.2]:
        configs.append((
            f"vol_scale={vs}",
            NQMeanRevV4,
            make_v4_config(vol_scale=vs)
        ))

    # ================================================================
    # SECTION 2: vol_scale=0.8 + additional improvements
    # ================================================================
    # A: + early_cut
    for ec_bars, ec_loss, ec_mfe in [(3, 1.5, 0.2), (4, 1.5, 0.2), (4, 2.0, 0.3)]:
        configs.append((
            f"vs=0.8 +earlycut(b={ec_bars},l={ec_loss},m={ec_mfe})",
            NQMeanRevV4,
            make_v4_config(vol_scale=0.8, early_cut=True,
                           early_cut_bars=ec_bars, early_cut_loss=ec_loss,
                           early_cut_mfe=ec_mfe)
        ))

    # B: + signal reversal
    configs.append((
        "vs=0.8 +sig_reversal",
        NQMeanRevV4,
        make_v4_config(vol_scale=0.8, sig_reversal=True)
    ))

    # C: + early_cut + signal reversal
    configs.append((
        "vs=0.8 +earlycut+sigreversal",
        NQMeanRevV4,
        make_v4_config(vol_scale=0.8, early_cut=True,
                       early_cut_bars=4, early_cut_loss=1.5, early_cut_mfe=0.2,
                       sig_reversal=True)
    ))

    # D: + daily loser cap
    configs.append((
        "vs=0.8 +daily_losers=1",
        NQMeanRevV4,
        make_v4_config(vol_scale=0.8, daily_losers=1)
    ))

    # ================================================================
    # SECTION 3: Different base thresholds with scaled vol_scale
    # ================================================================
    # With lower threshold but higher vol_scale → more trades in calm,
    # same selectivity in volatile
    for bt, vs in [(2.5, 1.2), (2.5, 1.5), (2.8, 1.0), (2.8, 0.8),
                    (3.5, 0.5), (3.5, 0.3)]:
        configs.append((
            f"thresh={bt} vs={vs}",
            NQMeanRevV4,
            make_v4_config(base_thresh=bt, vol_scale=vs)
        ))

    # ================================================================
    # SECTION 4: Disaster stop width (now that we have fewer entries)
    # ================================================================
    for ds in [4.0, 5.0, 6.0, 8.0]:
        configs.append((
            f"vs=0.8 disaster={ds}",
            NQMeanRevV4,
            make_v4_config(vol_scale=0.8, disaster=ds)
        ))

    # ================================================================
    # SECTION 5: Trailing activation sensitivity
    # ================================================================
    for act in [0.75, 1.0, 1.5, 2.0]:
        configs.append((
            f"vs=0.8 trail_act={act}",
            NQMeanRevV4,
            make_v4_config(vol_scale=0.8, activation=act)
        ))

    # ================================================================
    # SECTION 6: Best combos
    # ================================================================
    configs.append((
        "BEST_A: vs=0.8+earlycut+wider_disaster",
        NQMeanRevV4,
        make_v4_config(vol_scale=0.8, early_cut=True,
                       early_cut_bars=4, early_cut_loss=1.5, early_cut_mfe=0.2,
                       disaster=8.0)
    ))
    configs.append((
        "BEST_B: vs=0.8+earlycut+sig_rev+daily1",
        NQMeanRevV4,
        make_v4_config(vol_scale=0.8, early_cut=True,
                       early_cut_bars=4, early_cut_loss=1.5, early_cut_mfe=0.2,
                       sig_reversal=True, daily_losers=1)
    ))
    configs.append((
        "BEST_C: vs=0.85+earlycut+sig_rev",
        NQMeanRevV4,
        make_v4_config(vol_scale=0.85, early_cut=True,
                       early_cut_bars=4, early_cut_loss=1.5, early_cut_mfe=0.2,
                       sig_reversal=True)
    ))
    configs.append((
        "BEST_D: thresh=2.8 vs=1.0+earlycut",
        NQMeanRevV4,
        make_v4_config(base_thresh=2.8, vol_scale=1.0, early_cut=True,
                       early_cut_bars=4, early_cut_loss=1.5, early_cut_mfe=0.2)
    ))

    # Run all
    print(f"\nRunning {len(configs)} configs...")
    results = []
    for idx, (label, cls, cfg) in enumerate(configs):
        t1 = time.time()
        r = run_one(df, cls, cfg, label)
        dt = time.time() - t1
        if r:
            results.append(r)
            print(f"  [{idx+1:3d}/{len(configs)}] {label:52s} | "
                  f"N={r['n_trades']:>5,}  Net=${r['total_net_pnl']:>10,.0f}  "
                  f"Sharpe={r['daily_sharpe']:>6.2f}  WR={r['win_rate']:.1%}  "
                  f"PF={r['profit_factor']:.3f}  ({dt:.1f}s)")
        else:
            print(f"  [{idx+1:3d}/{len(configs)}] {label:52s} | NO TRADES ({dt:.1f}s)")

    # Sort and print
    results.sort(key=lambda x: x["daily_sharpe"], reverse=True)
    w = 145
    print(f"\n{'='*w}")
    print(f"  V4 REFINEMENT — ALL RESULTS (sorted by Sharpe)")
    print(f"{'='*w}")
    hdr = (f"  {'Label':52s} {'N':>5s} {'T/Y':>4s} {'Net PnL':>11s} "
           f"{'Sharpe':>7s} {'WR':>5s} {'PF':>6s} {'AvgW':>7s} {'AvgL':>7s} "
           f"{'Expect':>7s} {'MaxDD':>11s} {'+Yr':>5s}")
    print(hdr)
    print(f"  {'-'*(w-2)}")
    for r in results:
        print(f"  {r['label']:52s} {r['n_trades']:>5,} {r['trades_per_year']:>4.0f} "
              f"${r['total_net_pnl']:>10,.0f} {r['daily_sharpe']:>7.3f} "
              f"{r['win_rate']:>4.1%} {r['profit_factor']:>6.3f} "
              f"${r['avg_win']:>6,.0f} ${r['avg_loss']:>6,.0f} "
              f"${r['expectancy']:>6,.0f} ${r['max_drawdown']:>10,.0f} {r['pos_years']:>5s}")

    # Exit breakdown for top 5
    print(f"\n{'='*w}")
    print(f"  EXIT BREAKDOWNS — TOP 5")
    print(f"{'='*w}")
    for r in results[:5]:
        print(f"\n  >>> {r['label']} (Sharpe={r['daily_sharpe']:.3f}, Net=${r['total_net_pnl']:,.0f})")
        for reason, row in r["exit_reasons"].iterrows():
            print(f"      {reason:22s}  N={row['count']:>5.0f}  "
                  f"Total=${row['total_pnl']:>10,.0f}  Avg=${row['avg_pnl']:>7,.0f}  "
                  f"WR={row['win_rate']:>5.1%}")

    # Year-by-year for best
    if results:
        best = results[0]
        print(f"\n{'='*w}")
        print(f"  YEAR-BY-YEAR — BEST: {best['label']}")
        print(f"{'='*w}")
        yearly = best["yearly"]
        print(f"  {'Year':>6s} {'Trades':>7s} {'Gross':>11s} {'Net':>11s} {'WR':>6s} {'Avg':>8s}")
        print(f"  {'-'*55}")
        for yr, row in yearly.iterrows():
            m = " ✓" if row['net_pnl'] > 0 else ""
            print(f"  {yr:>6d} {row['n_trades']:>7.0f} ${row['gross_pnl']:>10,.0f} "
                  f"${row['net_pnl']:>10,.0f} {row['win_rate']:>5.1%} ${row['avg_trade']:>7,.0f}{m}")

    # ================================================================
    # BOOTSTRAP CONFIDENCE ANALYSIS — Top 3 configs
    # ================================================================
    print(f"\n{'='*w}")
    print(f"  BOOTSTRAP CONFIDENCE ANALYSIS (1000 resamples)")
    print(f"{'='*w}")

    for r in results[:3]:
        label = r["label"]
        # Find matching config
        for lbl, cls, cfg in configs:
            if lbl == label:
                boot = bootstrap_analysis(cls, cfg, df)
                if boot:
                    print(f"\n  >>> {label}")
                    print(f"      Trades: {boot['n_trades']}")
                    print(f"      Actual total PnL: ${boot['actual_total']:,.0f}")
                    print(f"      Total PnL CI: [${boot['total_ci_5']:,.0f}, "
                          f"${boot['total_ci_50']:,.0f}, ${boot['total_ci_95']:,.0f}]")
                    print(f"      Sharpe CI:     [{boot['sharpe_ci_5']:.2f}, "
                          f"{boot['sharpe_ci_50']:.2f}, {boot['sharpe_ci_95']:.2f}]")
                    print(f"      Win Rate CI:   [{boot['wr_ci_5']:.1%}, "
                          f"{boot['wr_ci_50']:.1%}, {boot['wr_ci_95']:.1%}]")
                    print(f"      P(profitable): {boot['prob_positive']:.1%}")
                break

    # ================================================================
    # IS/OOS on top 3
    # ================================================================
    print(f"\n{'='*w}")
    print(f"  IS/OOS VALIDATION — TOP 3")
    print(f"{'='*w}")

    is_mask = df.index.year <= 2019
    oos_mask = df.index.year >= 2020
    df_is = df[is_mask].copy()
    df_oos = df[oos_mask].copy()

    for r in results[:3]:
        label = r["label"]
        for lbl, cls, cfg in configs:
            if lbl == label:
                r_is = run_one(df_is, cls, cfg, f"IS: {label}")
                r_oos = run_one(df_oos, cls, cfg, f"OOS: {label}")

                print(f"\n  >>> {label}")
                if r_is:
                    print(f"      IS  (2008-2019): N={r_is['n_trades']:>4}  Net=${r_is['total_net_pnl']:>10,.0f}  "
                          f"Sharpe={r_is['daily_sharpe']:>6.2f}  WR={r_is['win_rate']:.1%}  PY={r_is['pos_years']}")
                else:
                    print(f"      IS  (2008-2019): NO TRADES")
                if r_oos:
                    print(f"      OOS (2020-2025): N={r_oos['n_trades']:>4}  Net=${r_oos['total_net_pnl']:>10,.0f}  "
                          f"Sharpe={r_oos['daily_sharpe']:>6.2f}  WR={r_oos['win_rate']:.1%}  PY={r_oos['pos_years']}")
                else:
                    print(f"      OOS (2020-2025): NO TRADES")
                break

    # Save
    rows = [{k: v for k, v in r.items() if k not in ("exit_reasons", "yearly", "trades_df")} for r in results]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "v4_refine_results.csv"), index=False)
    print(f"\n  Saved v4_refine_results.csv  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
