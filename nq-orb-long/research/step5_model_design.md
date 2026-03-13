# Step 5 — Model Design

**Date:** 2026-03-03
**Status:** Implementation
**Depends on:** Steps 1-4 (Event Definitions, Outcome Labeling, Feature Engineering, Statistical Research)

---

## Purpose

Transform the unconditional ORB Long signal (EV +0.72 pts, not statistically significant)
into a **conditional** trade filter that selects only high-probability events. The model's
job is NOT to predict market direction — the event already defines the direction. The model
predicts: **given that an ORB Long event just triggered, how likely is it to hit the
take-profit before the stop-loss?**

## The Honest Question: Does a Model Add Value?

With 655 ORB Long events across 7 years and ~60 features, the risk of overfitting is
real. More features × small sample = data mining danger. We must answer honestly:

- **Null hypothesis:** The model provides no improvement over taking every ORB Long signal.
- **Alternative:** Features can separate high-quality from low-quality events, concentrating
  the noisy unconditional edge into a reliable conditional edge.
- **Escape hatch:** If OOS filtered EV is not ≥25% better than unconditional EV, or OOS AUC
  is below 0.55, we abandon the model and take every event instead.

Step 4 evidence that a model *might* work:
- ORB Long passed the shuffled-label bias check (p=0.001) — features have genuine signal
- 85% of parameter combos are profitable — the edge is broad, not fragile
- Regime analysis shows clear conditional patterns (trending: +11.47 pts, crisis: -6.83)

---

## Model Selection

| Model | Role | Why |
|---|---|---|
| **LightGBM** | Primary | Handles NaN natively, fast, good with small samples when heavily regularized |
| **Random Forest** | Validation | More resistant to overfitting; if LightGBM can't beat RF, the boosting isn't adding value |
| **Logistic Regression** | Baseline floor | If trees can't beat L2-regularized logistic regression, there's no non-linear signal |

**Why NOT deep learning:** 655 samples is 100x too small. Neural nets need thousands minimum.

**Why binary classification, not regression:** Predicting P(win) is more interpretable than
predicting continuous return. "This event has a 68% win probability" is actionable. The
user can understand and trust a probability threshold.

---

## Feature Selection Pipeline

Starting from ~60 features, reduce to 10-20 for the final model:

1. **NaN filter:** Drop features with >30% missing values
2. **Variance filter:** Drop features with near-zero standard deviation
3. **Correlation clustering:** Among features with |corr| > 0.85, keep only the one
   with the highest univariate predictive power (Spearman corr with label)
4. **Permutation importance:** Train an over-regularized LightGBM (depth=2, 50 trees),
   compute permutation importance, select top 20 features

---

## Walk-Forward Cross-Validation

**Mandatory:** No random shuffle, no k-fold with random splits. Time flows forward.

| Parameter | Value | Rationale |
|---|---|---|
| Window type | Expanding | Small sample — can't afford to discard early data |
| Min training | 200 events (~2.1 years) | Minimum for tree models with depth-3 |
| Test blocks | 6 months (~47 events) | Enough for meaningful OOS statistics |
| Step forward | 6 months | Each fold adds the previous test block to training |
| Embargo | 5 trading days | Prevent autocorrelation leak at boundary |
| Expected folds | ~7-8 | Covers the full 2019-2026 period |

---

## Regularization (Anti-Overfitting)

**LightGBM settings (intentionally conservative):**

| Parameter | Value | Default | Why so aggressive? |
|---|---|---|---|
| max_depth | 3 | -1 (unlimited) | Force broad patterns only |
| num_leaves | 8 | 31 | Limits model complexity |
| min_child_samples | 30+ | 20 | At least 5% of training data per leaf |
| learning_rate | 0.05 | 0.1 | Slow learning = better generalization |
| n_estimators | 100 | 100 | With early stopping, patience=10 |
| reg_alpha (L1) | 1.0 | 0.0 | Penalize complexity |
| reg_lambda (L2) | 1.0 | 0.0 | Penalize complexity |
| subsample | 0.8 | 1.0 | Row bagging for variance reduction |
| colsample_bytree | 0.6 | 1.0 | Feature bagging — critical with correlated features |

---

## Probability Calibration

Raw model outputs from tree models are NOT well-calibrated probabilities. Calibration
maps "model says 0.7" to "historically, events scored 0.7 won ~70% of the time."

- **Method:** Platt scaling (logistic regression of raw prob vs label) — preferred for
  small samples over isotonic regression
- **Data:** Calibration is fit on pooled OOS predictions only (never in-sample)
- **Validation:** Reliability table showing predicted vs actual win rates by bin

---

## Threshold System (Trade/No-Trade)

The calibrated probability maps to a trade decision:

| Score Zone | Action | Rationale |
|---|---|---|
| < 0.45 | NO TRADE | Low confidence — skip |
| 0.45 - 0.55 | REDUCED SIZE (50%) | Moderate confidence — trade conservatively |
| > 0.55 | FULL SIZE | High confidence — full position |

Threshold optimization sweeps from 0.30 to 0.70 and selects the value that maximizes
`utility = EV × √N` (balances edge quality vs trade frequency). Minimum constraint:
at least 30 trades per year on average.

---

## Success Criteria

The model PASSES if:
1. OOS AUC ≥ 0.55 (above random)
2. Filtered EV at optimal threshold ≥ 25% better than unconditional EV (+0.72 pts)
3. Feature importance is stable across walk-forward folds (rank correlation > 0.5)
4. At least 30 filtered trades per year (still tradeable)

If ANY criterion fails → take every ORB Long event at reduced size (the raw edge is
nearly positive, and adding a bad model is worse than no model).

---

## Implementation

See `model_design.py` for the full implementation with:
- `select_features()` — feature reduction pipeline
- `build_walk_forward_splits()` — time-respecting cross-validation
- `train_model()` — fit models with proper regularization
- `calibrate_probabilities()` — OOS probability calibration
- `optimize_threshold()` — trade/no-trade threshold selection
- `run_model_pipeline()` — master orchestration
- `run_full_step5()` — one-call entry point
