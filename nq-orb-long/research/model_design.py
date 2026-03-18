"""
Step 5 — Model Design
=====================

Builds a walk-forward validated, probability-calibrated model that filters
ORB Long (and optionally other) events into trade / no-trade decisions.

Usage
-----
    from model_design import run_full_step5
    results = run_full_step5(df, 'event_orb_long', 'long', 'orb', 'ORB → LONG')

Requires: numpy, pandas, scikit-learn.  LightGBM optional (falls back to RF).
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Check for LightGBM availability
try:
    import lightgbm as lgb
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, accuracy_score,
)
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from scipy import stats


# ===========================================================================
# 1. FEATURE SELECTION
# ===========================================================================

def select_features(dataset, feature_cols=None, max_nan_pct=0.30,
                    min_std=1e-8, max_corr=0.85, n_top=20,
                    label_col='barrier_label', verbose=True):
    """
    Reduce features through a multi-stage pipeline.

    Returns
    -------
    list of selected feature column names
    """
    if feature_cols is None:
        from research.event_features import get_feature_columns
        feature_cols = get_feature_columns(dataset)

    if verbose:
        print(f"\n  Feature selection: starting with {len(feature_cols)} features")

    # -- Stage 1: NaN filter --
    nan_pcts = dataset[feature_cols].isna().mean()
    keep = [c for c in feature_cols if nan_pcts[c] <= max_nan_pct]
    dropped_nan = len(feature_cols) - len(keep)
    if verbose and dropped_nan:
        print(f"  [1/4] NaN filter (>{max_nan_pct:.0%}): dropped {dropped_nan}")
    feature_cols = keep

    # -- Stage 2: Variance filter --
    stds = dataset[feature_cols].std()
    keep = [c for c in feature_cols if stds[c] > min_std]
    dropped_var = len(feature_cols) - len(keep)
    if verbose and dropped_var:
        print(f"  [2/4] Variance filter: dropped {dropped_var}")
    feature_cols = keep

    # -- Stage 3: Correlation clustering --
    if len(feature_cols) > 2:
        # Fill NaN for correlation computation
        filled = dataset[feature_cols].fillna(dataset[feature_cols].median())
        corr_matrix = filled.corr(method='spearman').abs()

        # Compute univariate relevance (|corr| with label)
        definitive = dataset[dataset[label_col].isin([1, -1])]
        label_corrs = {}
        for col in feature_cols:
            vals = definitive[col].dropna()
            if len(vals) < 10:
                label_corrs[col] = 0.0
                continue
            matched = definitive.loc[vals.index, label_col]
            c, _ = stats.spearmanr(vals, matched)
            label_corrs[col] = abs(c) if not np.isnan(c) else 0.0

        # Greedy clustering: remove redundant features
        to_remove = set()
        sorted_features = sorted(feature_cols, key=lambda c: label_corrs.get(c, 0), reverse=True)

        for i, f1 in enumerate(sorted_features):
            if f1 in to_remove:
                continue
            for f2 in sorted_features[i+1:]:
                if f2 in to_remove:
                    continue
                if f1 in corr_matrix.columns and f2 in corr_matrix.columns:
                    if corr_matrix.loc[f1, f2] > max_corr:
                        to_remove.add(f2)  # f1 has higher relevance, remove f2

        keep = [c for c in feature_cols if c not in to_remove]
        if verbose and to_remove:
            print(f"  [3/4] Correlation clustering (>{max_corr:.2f}): dropped {len(to_remove)}")
        feature_cols = keep

    # -- Stage 4: Permutation importance (top N) --
    if len(feature_cols) > n_top:
        definitive = dataset[dataset[label_col].isin([1, -1])].copy()
        y = (definitive[label_col] == 1).astype(int)
        X = definitive[feature_cols].fillna(definitive[feature_cols].median())

        if len(y) >= 50:
            try:
                if HAS_LGBM:
                    quick_model = lgb.LGBMClassifier(
                        max_depth=2, n_estimators=50, num_leaves=4,
                        min_child_samples=max(20, int(0.1 * len(y))),
                        learning_rate=0.1, verbose=-1, random_state=42,
                    )
                else:
                    quick_model = RandomForestClassifier(
                        n_estimators=100, max_depth=3,
                        min_samples_leaf=max(20, int(0.1 * len(y))),
                        random_state=42, n_jobs=-1,
                    )
                quick_model.fit(X, y)
                perm = permutation_importance(
                    quick_model, X, y, n_repeats=10, random_state=42, n_jobs=-1,
                )
                imp = pd.Series(perm.importances_mean, index=feature_cols)
                top = imp.nlargest(n_top).index.tolist()
                if verbose:
                    print(f"  [4/4] Permutation importance: kept top {n_top} of {len(feature_cols)}")
                feature_cols = top
            except Exception as e:
                if verbose:
                    print(f"  [4/4] Permutation importance failed ({e}), keeping all {len(feature_cols)}")
        else:
            if verbose:
                print(f"  [4/4] Too few samples ({len(y)}) for importance, keeping all")

    if verbose:
        print(f"  Final features: {len(feature_cols)}")
        # Print top features with label correlation
        definitive = dataset[dataset[label_col].isin([1, -1])]
        rows = []
        for col in feature_cols:
            vals = definitive[col].dropna()
            if len(vals) < 10:
                continue
            matched = definitive.loc[vals.index, label_col]
            c, _ = stats.spearmanr(vals, matched)
            rows.append({'feature': col, 'corr_with_label': round(c, 4) if not np.isnan(c) else 0})
        if rows:
            imp_df = pd.DataFrame(rows).sort_values('corr_with_label', key=abs, ascending=False)
            print(f"\n  {'Feature':<40} {'Corr w/ label':>14}")
            print(f"  {'─'*40} {'─'*14}")
            for _, r in imp_df.head(20).iterrows():
                print(f"  {r['feature']:<40} {r['corr_with_label']:>+14.4f}")

    return feature_cols


# ===========================================================================
# 2. WALK-FORWARD FRAMEWORK
# ===========================================================================

def build_walk_forward_splits(dataset, min_train_events=200,
                               test_months=6, embargo_days=5):
    """
    Build time-series expanding-window walk-forward splits.

    Returns
    -------
    list of dicts with train_idx, test_idx, fold metadata
    """
    idx = dataset.index.sort_values()
    dates = pd.Series(idx)

    splits = []
    fold = 0

    # Find earliest point where we have min_train_events
    if len(idx) < min_train_events + 10:
        print(f"  [WARN] Only {len(idx)} events, need {min_train_events} for training")
        return splits

    first_test_start = idx[min_train_events]

    # Generate test blocks
    current_test_start = first_test_start
    end_date = idx[-1]

    while current_test_start < end_date:
        fold += 1
        test_end = current_test_start + pd.DateOffset(months=test_months)

        # Embargo: train ends embargo_days before test starts
        embargo_delta = pd.Timedelta(days=embargo_days)
        train_end = current_test_start - embargo_delta

        # Get indices
        train_mask = idx <= train_end
        test_mask = (idx >= current_test_start) & (idx < test_end)

        train_idx = idx[train_mask]
        test_idx = idx[test_mask]

        if len(test_idx) < 5:
            break

        splits.append({
            'fold': fold,
            'train_start': train_idx[0],
            'train_end': train_idx[-1],
            'test_start': test_idx[0],
            'test_end': test_idx[-1],
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'train_idx': train_idx,
            'test_idx': test_idx,
        })

        # Step forward
        current_test_start = test_end

    return splits


def print_walk_forward_summary(splits):
    """Print walk-forward split structure."""
    if not splits:
        print("  No splits generated.")
        return

    print(f"\n  Walk-Forward Splits: {len(splits)} folds")
    print(f"  {'Fold':>4} {'Train':>14} {'→':>2} {'':>14} {'N_tr':>6} "
          f"{'Test':>14} {'→':>2} {'':>14} {'N_te':>6}")
    print(f"  {'─'*4} {'─'*14} {'─'*2} {'─'*14} {'─'*6} "
          f"{'─'*14} {'─'*2} {'─'*14} {'─'*6}")

    for s in splits:
        print(f"  {s['fold']:>4} {str(s['train_start'].date()):>14} → "
              f"{str(s['train_end'].date()):>14} {s['n_train']:>6}  "
              f"{str(s['test_start'].date()):>14} → "
              f"{str(s['test_end'].date()):>14} {s['n_test']:>6}")

    total_oos = sum(s['n_test'] for s in splits)
    print(f"\n  Total OOS events: {total_oos}")


# ===========================================================================
# 3. MODEL TRAINING
# ===========================================================================

def prepare_labels(dataset, label_col='barrier_label',
                   return_col='barrier_return_pts',
                   timeout_treatment='proportional'):
    """
    Convert barrier labels to binary (0=loss, 1=win).

    timeout_treatment:
      'exclude' - drop timeout events (label==0)
      'loss'    - treat all timeouts as losses
      'proportional' - timeouts are wins if return > 0, else losses
    """
    df = dataset.copy()

    if timeout_treatment == 'exclude':
        df = df[df[label_col].isin([1, -1])]
        y = (df[label_col] == 1).astype(int)
    elif timeout_treatment == 'loss':
        y = (df[label_col] == 1).astype(int)
    elif timeout_treatment == 'proportional':
        y = pd.Series(0, index=df.index)
        y[df[label_col] == 1] = 1
        y[df[label_col] == -1] = 0
        # Timeouts: positive return → win
        timeout_mask = df[label_col] == 0
        if return_col in df.columns:
            y[timeout_mask & (df[return_col] > 0)] = 1
        # else timeouts stay as 0 (loss)
    else:
        raise ValueError(f"Unknown timeout_treatment: {timeout_treatment}")

    return df, y


def train_model(X_train, y_train, model_type='lgbm', random_state=42):
    """
    Train a model with aggressive regularization.

    Parameters
    ----------
    model_type : 'lgbm', 'rf', or 'logistic'

    Returns
    -------
    fitted model
    """
    n = len(y_train)

    if model_type == 'lgbm':
        if not HAS_LGBM:
            print("  [WARN] LightGBM not available, falling back to RF")
            return train_model(X_train, y_train, 'rf', random_state)

        model = lgb.LGBMClassifier(
            objective='binary',
            max_depth=3,
            num_leaves=8,
            min_child_samples=max(30, int(0.05 * n)),
            learning_rate=0.05,
            n_estimators=100,
            reg_alpha=1.0,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.6,
            verbose=-1,
            random_state=random_state,
        )

        # Early stopping with 20% validation split
        val_size = max(20, int(0.2 * n))
        X_tr = X_train.iloc[:-val_size]
        y_tr = y_train.iloc[:-val_size] if hasattr(y_train, 'iloc') else y_train[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:] if hasattr(y_train, 'iloc') else y_train[-val_size:]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(10, verbose=False), lgb.log_evaluation(-1)],
        )
        return model

    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=max(20, int(0.05 * n)),
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1,
        )
        # Impute NaN for sklearn
        X_filled = X_train.fillna(X_train.median())
        model.fit(X_filled, y_train)
        return model

    elif model_type == 'logistic':
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                penalty='l2', C=0.1, max_iter=1000,
                random_state=random_state,
            )),
        ])
        pipe.fit(X_train, y_train)
        return pipe

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def predict_proba(model, X, model_type='lgbm'):
    """Get P(win) predictions, handling NaN for sklearn models."""
    if model_type == 'lgbm' and HAS_LGBM:
        return model.predict_proba(X)[:, 1]
    else:
        X_filled = X.fillna(X.median())
        return model.predict_proba(X_filled)[:, 1]


def get_feature_importance(model, feature_names, model_type='lgbm'):
    """Extract feature importance from fitted model."""
    if model_type == 'lgbm' and HAS_LGBM:
        imp = model.feature_importances_
    elif model_type == 'rf':
        imp = model.feature_importances_
    elif model_type == 'logistic':
        # Get coefficients from pipeline
        coefs = model.named_steps['clf'].coef_[0]
        imp = np.abs(coefs)
    else:
        return pd.DataFrame()

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': imp,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    return df


# ===========================================================================
# 4. PROBABILITY CALIBRATION
# ===========================================================================

def calibrate_probabilities(raw_probs, true_labels, method='platt'):
    """
    Calibrate raw model probabilities using OOS data.

    method: 'platt' (logistic) or 'isotonic'
    Returns fitted calibrator that has a .predict() method.
    """
    if method == 'isotonic':
        calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        calibrator.fit(raw_probs, true_labels)
    elif method == 'platt':
        # Logistic regression: raw_prob → P(win)
        from sklearn.linear_model import LogisticRegression as LR
        X_cal = raw_probs.reshape(-1, 1) if isinstance(raw_probs, np.ndarray) else np.array(raw_probs).reshape(-1, 1)
        calibrator = LR(max_iter=1000)
        calibrator.fit(X_cal, true_labels)
    else:
        raise ValueError(f"Unknown method: {method}")

    return calibrator


def apply_calibration(calibrator, raw_probs, method='platt'):
    """Apply fitted calibrator to raw probabilities."""
    if method == 'isotonic':
        return calibrator.predict(raw_probs)
    elif method == 'platt':
        X = np.array(raw_probs).reshape(-1, 1)
        return calibrator.predict_proba(X)[:, 1]


def print_calibration_table(calibrated_probs, true_labels, n_bins=8):
    """Print reliability table."""
    bins = np.linspace(0, 1, n_bins + 1)
    print(f"\n  {'Bin':<14} {'Predicted':>10} {'Actual':>8} {'N':>6}")
    print(f"  {'─'*14} {'─'*10} {'─'*8} {'─'*6}")

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (calibrated_probs >= lo) & (calibrated_probs < hi)
        if mask.sum() == 0:
            continue
        pred = calibrated_probs[mask].mean()
        actual = true_labels[mask].mean()
        n = mask.sum()
        print(f"  [{lo:.2f}, {hi:.2f}){'' :>3} {pred:>10.3f} {actual:>8.3f} {n:>6}")


# ===========================================================================
# 5. THRESHOLD OPTIMIZATION
# ===========================================================================

def optimize_threshold(calibrated_probs, returns, labels,
                       min_trades_per_year=30, years=7.0,
                       timestamps=None):
    """
    Sweep threshold to find optimal trade/no-trade cutoff.

    Optimizes utility = EV * sqrt(N).
    """
    thresholds = np.arange(0.30, 0.71, 0.01)
    results = []
    min_total_trades = min_trades_per_year * years

    for thresh in thresholds:
        mask = calibrated_probs >= thresh
        n = mask.sum()

        if n < min_total_trades * 0.5:  # Allow some slack
            continue

        filtered_returns = returns[mask]
        filtered_labels = labels[mask]

        if len(filtered_returns) == 0:
            continue

        ev = filtered_returns.mean()
        wr = filtered_labels.mean() if len(filtered_labels) > 0 else 0
        std = filtered_returns.std()

        # Profit factor
        wins = filtered_returns[filtered_returns > 0]
        losses = filtered_returns[filtered_returns < 0]
        pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 999

        # Sharpe — daily P&L annualized if timestamps available, else trade-level approx
        if timestamps is not None:
            import pandas as pd
            filt_ts = pd.to_datetime(timestamps[mask])
            daily_pnl = pd.Series(filtered_returns, index=filt_ts).groupby(filt_ts.normalize()).sum()
            sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0
        else:
            sharpe = ev / std * np.sqrt(252) if std > 0 else 0

        # Utility: EV * sqrt(N) — balances quality and frequency
        utility = ev * np.sqrt(n)

        # Trades per year
        tpy = n / years

        results.append({
            'threshold': round(thresh, 2),
            'n_trades': n,
            'trades_per_year': round(tpy, 1),
            'ev_pts': round(ev, 3),
            'win_rate': round(wr, 4),
            'profit_factor': round(pf, 3),
            'sharpe': round(sharpe, 3),
            'total_pnl': round(filtered_returns.sum(), 1),
            'utility': round(utility, 2),
        })

    if not results:
        return pd.DataFrame(), 0.50

    df = pd.DataFrame(results)
    best_idx = df['utility'].idxmax()
    best_threshold = df.loc[best_idx, 'threshold']

    return df, best_threshold


def print_threshold_analysis(threshold_df, best_threshold, unconditional_ev):
    """Print threshold sweep results."""
    if threshold_df.empty:
        print("  No valid thresholds found.")
        return

    print(f"\n  ┌── THRESHOLD OPTIMIZATION ─────────────────────────────────────┐")
    print(f"  │  Unconditional EV: {unconditional_ev:>+8.3f} pts/trade                  │")
    print(f"  │  Optimal threshold: {best_threshold:>5.2f}                                │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    # Show a subset of results around the optimal
    print(f"\n  {'Thresh':>6} {'N':>5} {'TPY':>5} {'EV':>8} {'WR':>6} "
          f"{'PF':>6} {'Sharpe':>7} {'Total':>8} {'Utility':>8}")
    print(f"  {'─'*6} {'─'*5} {'─'*5} {'─'*8} {'─'*6} "
          f"{'─'*6} {'─'*7} {'─'*8} {'─'*8}")

    for _, row in threshold_df.iterrows():
        marker = " ◀" if row['threshold'] == best_threshold else ""
        print(f"  {row['threshold']:>6.2f} {row['n_trades']:>5} {row['trades_per_year']:>5.0f} "
              f"{row['ev_pts']:>+8.3f} {row['win_rate']:>6.1%} "
              f"{row['profit_factor']:>6.2f} {row['sharpe']:>+7.2f} "
              f"{row['total_pnl']:>+8.1f} {row['utility']:>+8.1f}{marker}")


# ===========================================================================
# 6. PERFORMANCE EVALUATION
# ===========================================================================

def evaluate_model(oos_df, unconditional_ev, unconditional_n, years=7.0):
    """
    Comprehensive OOS evaluation.

    oos_df columns: y_true, raw_prob, cal_prob, returns, fold
    """
    y = oos_df['y_true'].values
    raw_p = oos_df['raw_prob'].values
    cal_p = oos_df['cal_prob'].values
    returns = oos_df['returns'].values

    metrics = {}

    # Overall metrics
    try:
        metrics['auc'] = round(roc_auc_score(y, raw_p), 4)
    except:
        metrics['auc'] = 0.5
    metrics['brier'] = round(brier_score_loss(y, raw_p), 4)
    try:
        metrics['log_loss'] = round(log_loss(y, np.clip(raw_p, 0.01, 0.99)), 4)
    except:
        metrics['log_loss'] = 999
    metrics['accuracy'] = round(accuracy_score(y, (raw_p >= 0.5).astype(int)), 4)
    metrics['n_oos'] = len(y)
    metrics['oos_win_rate'] = round(y.mean(), 4)
    metrics['oos_ev'] = round(returns.mean(), 3)
    metrics['unconditional_ev'] = unconditional_ev

    # Per-fold stability
    fold_aucs = []
    fold_evs = []
    for fold in oos_df['fold'].unique():
        fold_mask = oos_df['fold'] == fold
        fold_y = y[fold_mask]
        fold_p = raw_p[fold_mask]
        fold_r = returns[fold_mask]

        if len(np.unique(fold_y)) < 2:
            continue
        try:
            fold_aucs.append(roc_auc_score(fold_y, fold_p))
        except:
            fold_aucs.append(0.5)
        fold_evs.append(fold_r.mean())

    metrics['fold_auc_mean'] = round(np.mean(fold_aucs), 4) if fold_aucs else 0.5
    metrics['fold_auc_std'] = round(np.std(fold_aucs), 4) if fold_aucs else 0
    metrics['fold_ev_mean'] = round(np.mean(fold_evs), 3) if fold_evs else 0
    metrics['fold_ev_std'] = round(np.std(fold_evs), 3) if fold_evs else 0

    # Threshold-filtered results
    thresh_df, best_thresh = optimize_threshold(
        cal_p, returns, y, years=years,
        timestamps=oos_df.index,
    )
    metrics['best_threshold'] = best_thresh
    metrics['threshold_df'] = thresh_df

    if not thresh_df.empty:
        best_row = thresh_df[thresh_df['threshold'] == best_thresh]
        if len(best_row) > 0:
            metrics['filtered_ev'] = best_row.iloc[0]['ev_pts']
            metrics['filtered_n'] = best_row.iloc[0]['n_trades']
            metrics['filtered_wr'] = best_row.iloc[0]['win_rate']
            metrics['filtered_pf'] = best_row.iloc[0]['profit_factor']
            metrics['filtered_tpy'] = best_row.iloc[0]['trades_per_year']
        else:
            metrics['filtered_ev'] = metrics['oos_ev']
            metrics['filtered_n'] = metrics['n_oos']
    else:
        metrics['filtered_ev'] = metrics['oos_ev']
        metrics['filtered_n'] = metrics['n_oos']

    # Pass/fail verdict
    ev_improvement = (metrics.get('filtered_ev', 0) - unconditional_ev) / max(abs(unconditional_ev), 0.01)
    auc_pass = metrics['auc'] >= 0.55
    ev_pass = metrics.get('filtered_ev', 0) > unconditional_ev * 1.25 if unconditional_ev > 0 else metrics.get('filtered_ev', 0) > 0
    trades_pass = metrics.get('filtered_tpy', 0) >= 30

    metrics['auc_pass'] = auc_pass
    metrics['ev_pass'] = ev_pass
    metrics['trades_pass'] = trades_pass
    metrics['overall_pass'] = auc_pass and ev_pass and trades_pass

    return metrics


def print_model_report(metrics, event_name="Event"):
    """Print comprehensive model evaluation."""
    print(f"\n{'='*72}")
    print(f"  MODEL EVALUATION: {event_name}")
    print(f"{'='*72}")

    print(f"\n  ┌── DISCRIMINATIVE POWER ──────────────────────────────────────────┐")
    print(f"  │  AUC-ROC:    {metrics['auc']:>6.4f}  (≥0.55 required)  "
          f"{'✓ PASS' if metrics['auc_pass'] else '✗ FAIL':>10}  │")
    print(f"  │  Brier:      {metrics['brier']:>6.4f}  (lower = better)              │")
    print(f"  │  Log loss:   {metrics['log_loss']:>6.4f}                                   │")
    print(f"  │  Accuracy:   {metrics['accuracy']:>6.1%}                                   │")
    print(f"  │  OOS events: {metrics['n_oos']:>6}                                   │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌── FOLD STABILITY ────────────────────────────────────────────────┐")
    print(f"  │  Mean fold AUC:  {metrics['fold_auc_mean']:>6.4f} ± {metrics['fold_auc_std']:.4f}          │")
    print(f"  │  Mean fold EV:   {metrics['fold_ev_mean']:>+7.3f} ± {metrics['fold_ev_std']:.3f} pts      │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌── FILTERED vs UNCONDITIONAL ──────────────────────────────────────┐")
    print(f"  │  Unconditional EV:  {metrics['unconditional_ev']:>+8.3f} pts/trade               │")
    print(f"  │  Filtered EV:       {metrics.get('filtered_ev', 0):>+8.3f} pts/trade "
          f"{'✓ BETTER' if metrics['ev_pass'] else '✗ NO IMPROVEMENT':>18} │")
    filt_wr = metrics.get('filtered_wr', 0)
    filt_pf = metrics.get('filtered_pf', 0)
    filt_tpy = metrics.get('filtered_tpy', 0)
    print(f"  │  Filtered WR:       {filt_wr:>8.1%}                            │")
    print(f"  │  Filtered PF:       {filt_pf:>8.3f}x                           │")
    print(f"  │  Trades/year:       {filt_tpy:>8.1f}  (≥30 required) "
          f"{'✓' if metrics['trades_pass'] else '✗':>5}     │")
    print(f"  │  Best threshold:    {metrics['best_threshold']:>8.2f}                            │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    verdict = "✓ PASS — Model adds value, use filtered signals" if metrics['overall_pass'] \
              else "✗ FAIL — Take every event (model doesn't help enough)"
    print(f"\n  VERDICT: {verdict}")


# ===========================================================================
# 7. MASTER PIPELINE
# ===========================================================================

def run_model_pipeline(dataset, event_name='Event',
                       model_types=None, timeout_treatment='proportional',
                       feature_cols=None, verbose=True):
    """
    Full model pipeline: feature selection → walk-forward → calibration → evaluation.

    Returns dict with all results.
    """
    if model_types is None:
        model_types = ['lgbm', 'rf', 'logistic']
        if not HAS_LGBM:
            model_types = ['rf', 'logistic']

    results = {'event_name': event_name}

    print(f"\n{'#'*72}")
    print(f"  STEP 5 — MODEL PIPELINE: {event_name}")
    print(f"{'#'*72}")

    # --- Prepare labels ---
    print(f"\n[1/6] Preparing labels (timeout={timeout_treatment})...")
    dataset_binary, y_all = prepare_labels(dataset, timeout_treatment=timeout_treatment)
    print(f"  Events: {len(y_all)}, Win rate: {y_all.mean():.1%}")
    results['n_events'] = len(y_all)
    results['win_rate'] = round(y_all.mean(), 4)

    # Unconditional EV
    if 'barrier_return_pts' in dataset_binary.columns:
        unconditional_ev = dataset_binary['barrier_return_pts'].mean()
    else:
        unconditional_ev = 0
    results['unconditional_ev'] = round(unconditional_ev, 3)
    print(f"  Unconditional EV: {unconditional_ev:+.3f} pts")

    # --- Feature selection (P4-B: moved inside WFO fold loop) ---
    print(f"\n[2/6] Feature selection (per-fold on training data only)...")
    if feature_cols is None:
        from research.event_features import get_feature_columns
        all_features = get_feature_columns(dataset_binary)
    else:
        all_features = feature_cols

    # NOTE: select_features is now called per-fold inside the WFO loop below.
    # This avoids selection bias from using OOS data to influence feature choice.
    results['initial_feature_pool'] = all_features
    results['n_initial_features'] = len(all_features)

    # --- Walk-forward splits ---
    print(f"\n[3/6] Building walk-forward splits...")
    years_of_data = (dataset_binary.index[-1] - dataset_binary.index[0]).days / 365.25
    splits = build_walk_forward_splits(dataset_binary, min_train_events=200, test_months=6)
    print_walk_forward_summary(splits)
    results['splits'] = splits

    if not splits:
        print("  [ERROR] No valid walk-forward splits. Aborting.")
        return results

    # P4-B: Per-fold feature selection (cached across model types)
    from collections import Counter
    fold_feature_sets = []
    for fold_i, split in enumerate(splits):
        train_data = dataset_binary.loc[split['train_idx']]
        fold_feats = select_features(train_data, feature_cols=all_features, verbose=False)
        fold_feature_sets.append(fold_feats)
        print(f"    Fold {fold_i}: {len(fold_feats)} features selected on train data")

    # Production feature set: union of features in >=50% of folds
    feature_counts = Counter(f for fs in fold_feature_sets for f in fs)
    n_folds = len(splits)
    selected = sorted([
        f for f, count in feature_counts.items()
        if count >= n_folds * 0.50
    ])
    print(f"  Production feature set: {len(selected)} features "
          f"(>=50% of {n_folds} folds)")
    results['selected_features'] = selected
    results['n_features'] = len(selected)

    # --- Walk-forward training ---
    print(f"\n[4/6] Walk-forward model training...")

    best_model_type = None
    best_auc = 0
    all_model_results = {}

    for mt in model_types:
        print(f"\n  ── Model: {mt.upper()} ──")
        oos_records = []
        fold_importances = []

        for fold_i, split in enumerate(splits):
            # Train/test data
            train_data = dataset_binary.loc[split['train_idx']]
            test_data = dataset_binary.loc[split['test_idx']]

            _, y_train = prepare_labels(train_data, timeout_treatment=timeout_treatment)
            _, y_test = prepare_labels(test_data, timeout_treatment=timeout_treatment)

            # P4-B: Use per-fold feature set (train-only selection)
            fold_selected = fold_feature_sets[fold_i]
            X_train = train_data.loc[y_train.index, fold_selected]
            X_test = test_data.loc[y_test.index, fold_selected]

            # Handle edge cases
            if len(y_train) < 50 or len(y_test) < 5:
                continue
            if y_train.nunique() < 2:
                continue

            # Train
            model = train_model(X_train, y_train, model_type=mt)

            # Predict
            probs = predict_proba(model, X_test, model_type=mt)

            # Feature importance (use this fold's feature set)
            imp = get_feature_importance(model, fold_selected, model_type=mt)
            fold_importances.append(imp)

            # Get returns for test events
            test_returns = test_data.loc[y_test.index, 'barrier_return_pts'].values \
                if 'barrier_return_pts' in test_data.columns else np.zeros(len(y_test))

            # Record OOS predictions
            for i, idx in enumerate(y_test.index):
                oos_records.append({
                    'event_time': idx,
                    'y_true': y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i],
                    'raw_prob': probs[i],
                    'returns': test_returns[i],
                    'fold': split['fold'],
                })

        if not oos_records:
            print(f"    No OOS predictions generated")
            continue

        oos_df = pd.DataFrame(oos_records)

        # Quick AUC
        try:
            auc = roc_auc_score(oos_df['y_true'], oos_df['raw_prob'])
        except:
            auc = 0.5
        print(f"    OOS events: {len(oos_df)}, AUC: {auc:.4f}, "
              f"OOS EV: {oos_df['returns'].mean():+.3f}")

        all_model_results[mt] = {
            'oos_df': oos_df,
            'fold_importances': fold_importances,
            'auc': auc,
        }

        if auc > best_auc:
            best_auc = auc
            best_model_type = mt

    if best_model_type is None:
        print("\n  [ERROR] No model produced valid OOS predictions.")
        return results

    results['all_models'] = {k: v['auc'] for k, v in all_model_results.items()}
    results['best_model'] = best_model_type
    print(f"\n  Best model: {best_model_type.upper()} (AUC={best_auc:.4f})")

    # Use best model's OOS predictions
    oos_df = all_model_results[best_model_type]['oos_df']

    # --- Calibration ---
    print(f"\n[5/6] Probability calibration...")
    calibrator = calibrate_probabilities(
        oos_df['raw_prob'].values,
        oos_df['y_true'].values,
        method='platt',
    )
    cal_probs = apply_calibration(calibrator, oos_df['raw_prob'].values, method='platt')
    oos_df['cal_prob'] = cal_probs

    print_calibration_table(cal_probs, oos_df['y_true'].values)
    results['calibrator'] = calibrator

    # --- Evaluation ---
    print(f"\n[6/6] Model evaluation...")
    metrics = evaluate_model(oos_df, unconditional_ev, len(dataset_binary), years=years_of_data)
    print_model_report(metrics, event_name)

    # Threshold analysis
    if not metrics['threshold_df'].empty:
        print_threshold_analysis(metrics['threshold_df'], metrics['best_threshold'], unconditional_ev)

    results['metrics'] = metrics
    results['oos_df'] = oos_df

    # --- Feature importance stability ---
    fold_imps = all_model_results[best_model_type]['fold_importances']
    if len(fold_imps) >= 2:
        print(f"\n  Feature Importance Stability ({best_model_type.upper()}):")
        # Compare first vs last fold importance rankings
        first_imp = fold_imps[0].set_index('feature')['importance']
        last_imp = fold_imps[-1].set_index('feature')['importance']
        common = first_imp.index.intersection(last_imp.index)
        if len(common) >= 3:
            rank_corr, _ = stats.spearmanr(
                first_imp[common].rank(),
                last_imp[common].rank(),
            )
            print(f"    Rank correlation (fold 1 vs fold {len(fold_imps)}): {rank_corr:.4f}")
            print(f"    {'✓ STABLE' if rank_corr > 0.5 else '✗ UNSTABLE'} "
                  f"(threshold: 0.50)")
            results['importance_stability'] = round(rank_corr, 4)

        # Average importance across folds
        print(f"\n  Top features (avg importance across folds):")
        all_imp = pd.concat([
            imp.set_index('feature')['importance'] for imp in fold_imps
        ], axis=1)
        avg_imp = all_imp.mean(axis=1).sort_values(ascending=False)
        print(f"  {'Feature':<40} {'Avg Importance':>14}")
        print(f"  {'─'*40} {'─'*14}")
        for feat, val in avg_imp.head(15).items():
            print(f"  {feat:<40} {val:>14.4f}")

    return results


def run_full_step5(df, event_col, direction, event_type, event_name,
                   timeout_treatment='proportional'):
    """
    One-call entry point: build dataset → run model pipeline.
    """
    from event_features import build_model_dataset

    print(f"\n[DATASET] Building model dataset for {event_name}...")
    dataset = build_model_dataset(
        df, event_col=event_col, direction=direction, event_type=event_type,
    )
    print(f"[DATASET] {len(dataset)} events, {len(dataset.columns)} columns")

    return run_model_pipeline(
        dataset, event_name=event_name,
        timeout_treatment=timeout_treatment,
    )


# ===========================================================================
# VALIDATION SCRIPT
# ===========================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../research_utils')
    sys.path.insert(0, '.')

    from feature_engineering import load_ohlcv, build_features
    from event_definitions import add_session_columns, detect_orb, detect_session_sweep

    print("=" * 72)
    print("STEP 5 — MODEL DESIGN — Validation Run")
    print("=" * 72)

    # --- Load expanded dataset ---
    filepath = '../nq_continuous_5m_converted.csv'
    print(f"\n[LOAD] Loading {filepath}...")
    df = load_ohlcv(filepath)
    df = df['2019-01-01':'2026-01-14']
    print(f"[LOAD] {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    # --- Build features ---
    print("\n[FEATURES] Building bar-level features...")
    df = build_features(df, add_targets_flag=False)

    # --- Detect events ---
    print("[EVENTS] Detecting events...")
    df = add_session_columns(df)
    df = detect_orb(df)
    df = detect_session_sweep(df)

    # --- Primary candidate: ORB Long ---
    results_orb_long = run_full_step5(
        df, event_col='event_orb_long', direction='long',
        event_type='orb', event_name='ORB → LONG',
    )

    # --- Secondary candidate: Sweep Low Long ---
    print(f"\n\n{'='*72}")
    print("  Now testing secondary candidate...")
    print(f"{'='*72}")

    results_sweep_low = run_full_step5(
        df, event_col='sweep_low_first_today', direction='long',
        event_type='sweep', event_name='Sweep Low → LONG',
    )

    # --- Final Comparison ---
    print(f"\n\n{'#'*72}")
    print(f"  STEP 5 — FINAL COMPARISON")
    print(f"{'#'*72}")
    print(f"\n  {'Event':<25} {'Model':>8} {'AUC':>7} {'Uncond EV':>10} {'Filt EV':>9} {'Filt TPY':>9} {'Verdict':>10}")
    print(f"  {'─'*25} {'─'*8} {'─'*7} {'─'*10} {'─'*9} {'─'*9} {'─'*10}")

    for name, res in [('ORB → LONG', results_orb_long), ('Sweep Low → LONG', results_sweep_low)]:
        m = res.get('metrics', {})
        bm = res.get('best_model', 'n/a')
        print(f"  {name:<25} {bm:>8} {m.get('auc', 0):>7.4f} "
              f"{m.get('unconditional_ev', 0):>+10.3f} "
              f"{m.get('filtered_ev', 0):>+9.3f} "
              f"{m.get('filtered_tpy', 0):>9.1f} "
              f"{'PASS' if m.get('overall_pass') else 'FAIL':>10}")

    print(f"\n[DONE] Step 5 complete.")
