# Step 2 — Outcome Labeling
## NQ Futures Event-Driven Research Framework
### Date: 2026-03-03

---

## Why Labeling Matters

In supervised machine learning, the **label** (also called the "target" or
"dependent variable") is the thing you're trying to predict. In trading, the
label answers the question: **"After this event fired, what happened next?"**

Most retail quant strategies make a critical mistake here: they use simple
forward returns (e.g., "was the close 20 bars later higher or lower?"). This
ignores the reality of trading:

- You don't hold for exactly 20 bars — you have a **stop-loss** and **take-profit**
- The PATH of price matters, not just the endpoint
- A trade that goes +50 points then comes all the way back to +2 points at bar 20
  would be labeled "winning" by forward returns but was actually a terrible trade
  (you'd have been stopped out or taken profit long before bar 20)

The **double-barrier method** (also called "triple-barrier" when including a time
limit) solves this by labeling based on **which exit condition is hit first**:

```
                    ┌─── Take-Profit barrier (1.5x ATR above entry)
                    │
    Price ──────────┤    ← Entry point (event fires here)
                    │
                    └─── Stop-Loss barrier (1.0x ATR below entry)

    ──────────────────────────────────────────────► Time
                    │                            │
                    └── Trade starts here         └── Vertical barrier
                                                     (max hold or session end)
```

**Label = which barrier gets hit first:**
- Take-profit hit first → **Label = +1** (winner)
- Stop-loss hit first → **Label = -1** (loser)
- Time runs out (vertical barrier) → **Label = 0** (timeout/scratch)

---

## Labeling Scheme: Two Systems

We use TWO labeling systems in parallel. They serve different purposes:

### Primary: Double-Barrier Labels (for model training)

This is what we train models on because it mirrors actual trade outcomes.

| Label | Meaning | How It's Determined |
|-------|---------|-------------------|
| +1 | Winner | Take-profit barrier hit before stop-loss and before time expires |
| -1 | Loser | Stop-loss barrier hit before take-profit and before time expires |
| 0 | Timeout | Neither barrier hit within the maximum holding period |

**Why this is better than simple direction labels:**
- It incorporates your actual risk management (stop and target)
- It accounts for the path dependency of price
- A trade that goes the "right" direction but hits your stop first is correctly
  labeled as a loser — because that's what it would be in real trading
- Timeout trades (label 0) are genuine — sometimes the market just doesn't move

### Secondary: Fixed-Horizon Forward Returns (for analysis)

These are used for **understanding**, not for model training. They answer:
"Regardless of stops, what did price do over the next N bars?"

| Horizon | Bars | Time (5-min) | Use |
|---------|------|-------------|-----|
| 1-bar | 1 | 5 min | Immediate reaction / slippage analysis |
| 5-bar | 5 | 25 min | Short-term direction |
| 10-bar | 10 | 50 min | Near-term momentum |
| 20-bar | 20 | 100 min (~1.7 hrs) | Core intraday swing horizon |
| 40-bar | 40 | 200 min (~3.3 hrs) | Extended hold |

For each horizon we compute:
- **Raw forward return** (in NQ points and in percentage)
- **Forward direction** (binary: 1 if positive, 0 if negative)
- **Maximum Favorable Excursion (MFE)**: best unrealized P&L during the window
- **Maximum Adverse Excursion (MAE)**: worst unrealized drawdown during the window

MFE and MAE are extremely valuable for calibrating barrier levels later (Step 4).

---

## Barrier Calibration

### Stop-Loss Barrier

The stop-loss distance is expressed as a multiple of ATR (Average True Range).
ATR naturally adapts to current volatility:
- In low-vol environments (ATR = 8 points): stop might be 8-12 points
- In high-vol environments (ATR = 25 points): stop might be 25-37 points

This is critical because a fixed 20-point stop would be too tight in high-vol
and too wide in low-vol.

**Default: 1.0x ATR(14)**

Rationale: 1x ATR represents roughly one "normal" bar's worth of movement.
A stop at 1x ATR means you're giving the trade room to breathe through one
bar of normal adverse movement, but exiting if it goes beyond that.

For the TopStep account: ATR(14) on NQ 5-min bars typically ranges from
10-30 points. At 1.0x ATR, stop-loss = 10-30 points = $200-$600 per contract.
This is well within the 1-2% risk budget ($500-$1,000 on a $50k account).

### Take-Profit Barrier

**Default: 1.5x ATR(14)**

Rationale: The take-profit is set wider than the stop-loss to create a
favorable risk-reward ratio. At 1.5x ATR TP / 1.0x ATR SL, you need a
win rate above 40% to be profitable:

```
Break-even win rate = SL / (SL + TP) = 1.0 / (1.0 + 1.5) = 40%
```

If the events have genuine predictive power, they should produce win rates
above 50%, making this R:R highly profitable.

### Vertical Barrier (Time Limit)

**Default: min(48 bars, session end)**

Rationale:
- 48 bars = 4 hours on 5-min data, the upper end of "intraday swing"
- But we also force exit at the end of RTH (16:00 ET) since we're RTH-only
- Whichever comes first becomes the vertical barrier

### Asymmetric Barriers for Event Direction

This is important to understand: the barrier interpretation flips based on
whether the event predicts a LONG or SHORT:

**For LONG events** (sweep low → buy, ORB long → buy):
- Take-profit = entry + TP_distance (price goes UP to hit TP)
- Stop-loss = entry - SL_distance (price goes DOWN to hit SL)

**For SHORT events** (sweep high → sell, ORB short → sell):
- Take-profit = entry - TP_distance (price goes DOWN to hit TP)
- Stop-loss = entry + SL_distance (price goes UP to hit SL)

### Same-Bar Barrier Hits (Conservative Convention)

Sometimes a single 5-min bar is wide enough that BOTH the TP and SL
could theoretically have been touched within the same bar. When this happens:

**Convention: assume the ADVERSE barrier was hit first.**

This is the conservative assumption. With 5-min bars, we can't know the
intra-bar sequence of prices. By assuming the worst case, we avoid over-
fitting to lucky outcomes. This slightly underestimates the strategy's true
performance, which is the right bias to have for out-of-sample robustness.

---

## Additional Metrics Computed Per Event

Beyond the label itself, we compute metadata about each trade outcome
that will be essential for Step 4 (statistical analysis) and Step 6
(strategy construction):

| Metric | Description |
|--------|-------------|
| `barrier_label` | +1, -1, or 0 |
| `barrier_return_pts` | Actual P&L in NQ points at exit |
| `barrier_return_pct` | Actual P&L as percentage |
| `time_to_barrier` | Bars from entry to barrier hit |
| `exit_type` | 'tp', 'sl', or 'timeout' |
| `exit_price` | Approximate exit price |
| `mae_pts` | Maximum Adverse Excursion (points) |
| `mfe_pts` | Maximum Favorable Excursion (points) |
| `mae_atr` | MAE normalized by ATR at entry |
| `mfe_atr` | MFE normalized by ATR at entry |
| `atr_at_entry` | ATR(14) value at the time of the event |
| `entry_price` | Close of the event bar |
| `tp_level` | Take-profit price level |
| `sl_level` | Stop-loss price level |
| `vertical_bar` | Index of the vertical barrier bar |

---

## Parameters Available for Sweep in Step 4

| Parameter | Values to Test | Impact |
|-----------|---------------|--------|
| sl_atr_multiple | 0.75, 1.0, 1.25, 1.5 | Tighter stop = more losers but smaller losses |
| tp_atr_multiple | 1.0, 1.5, 2.0, 2.5, 3.0 | Wider TP = fewer winners but larger wins |
| atr_lookback | 10, 14, 20 | Longer = smoother but less responsive |
| max_holding_bars | 12, 24, 36, 48 | Longer = more time for trade to work |
| force_session_exit | True, False | True = flat by EOD (recommended for TopStep) |

---

*Document generated 2026-03-03 for NQ event-driven research framework.*
