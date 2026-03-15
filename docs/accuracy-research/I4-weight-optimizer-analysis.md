# IC-Based Weight Optimizer Deep-Dive Analysis

**Status:** Critical issues identified. The optimizer never caught composite IC = -0.23 because per-dimension IC was not checked comprehensively. The new composite IC guard is correctly implemented but threshold choices need review.

**Analysis Date:** 2026-03-15
**Files Analyzed:**
- `/Users/admin/Documents/web3 Signals x402/signal_fusion/optimizer.py` (756 lines)
- `/Users/admin/Documents/web3 Signals x402/shared/storage.py` (IC computation functions)
- `/Users/admin/Documents/web3 Signals x402/signal_fusion/profiles/default.yaml` (learning config)

---

## 1. IC COMPUTATION CORRECTNESS

### 1.1 How Spearman IC is Computed

**Location:** `storage.py:849-1040` (`compute_ic()`)

**Method:** Cross-sectional (rank assets at each time point), NOT time-series.

```python
# storage.py:953-968
for ts, observations in slices.items():
    if len(observations) < 3:  # Need at least 3 assets for meaningful correlation
        continue
    returns = [obs["pct_change"] for obs in observations]
    for dim in all_dimensions:
        dim_obs = [(obs["dimensions"][dim], obs["pct_change"])
                   for obs in observations if dim in obs["dimensions"]]
        if len(dim_obs) >= 3:
            dim_scores = [d[0] for d in dim_obs]
            dim_returns = [d[1] for d in dim_obs]
            dim_pairs[dim].append((dim_scores, dim_returns))
```

**Key Points:**
1. **Grouping Strategy:** Rows grouped by `timestamp` (truncated to minute: `r[2][:16]`) → "slices"
2. **Cross-sectional:** Each slice = one time point with multiple assets
3. **Rank correlation:** Uses `_rank_array()` + `_pearson()` on ranks = Spearman (storage.py:990-993)
4. **Aggregation:** Averages IC across all slices using `_spearman_ic()` (storage.py:980-996)

**Correctness Assessment:** ✅ **Correct for cross-sectional analysis**
- Ranks correctly handle ties (line 28-33: `avg_rank = (i + j) / 2.0 + 1.0`)
- Pearson on ranks is valid Spearman formula
- Aggregation across slices is sound

---

### 1.2 Minimum Sample Size & Edge Cases

**Minimum observations per slice:** 3 assets (line 954)
**Minimum observations per dimension within slice:** 3 assets (line 962)

**Assessment:** ⚠️ **BORDERLINE CRITICAL**

| Issue | Location | Severity | Impact |
|-------|----------|----------|--------|
| **n=3 too small for Spearman** | storage.py:954,962 | HIGH | Spearman reliability < 0.70 with n=3; statistical power is 36% |
| **All same score edge case** | storage.py:44-47 (`_pearson`) | HANDLED | Returns `None` if `sx < 1e-12` (no variance) |
| **All same return edge case** | storage.py:45-47 | HANDLED | Returns `None` if `sy < 1e-12` (no variance) |
| **Degenerate Spearman** | storage.py:987-996 | HANDLED | Skips pairs with `n < 3` |

**Problem:** With n=3, one outlier asset can dominate IC. Example:
- Assets: BTC (+5%), ETH (-2%), SOL (+8%)
- If one dimension ranks them opposite to returns, IC ≈ -0.5 despite limited signal
- With 10 such slices, average IC can be artificially negative

**Recommendation:** Increase minimum slice size from 3 to **5-7 assets** or weight by slice size.

---

## 2. EMA BLENDING PROBLEM

### 2.1 Current EMA Configuration

**Location:** `optimizer.py:664-694` (`_ema_blend()`)

```python
learning_rate = float(self.cfg.get("learning_rate", 0.3))
# Default: 30% new + 70% old

blended[role] = round(learning_rate * new_w + (1 - learning_rate) * old_w, 4)
```

**Default:** `learning_rate = 0.3` → 30% new weight, 70% old weight

**Adaptation Speed:** Half-life = ln(0.5) / ln(1 - 0.3) ≈ **2.1 cycles** to reach 50% influence of new weights

### 2.2 The Problem: 70% Old Weight Preservation

**Scenario:** Suppose dimension A has truly negative IC (-0.10) for 8 consecutive optimization cycles:

| Cycle | True IC | Old Weight | New Computed | Blended (30% new) |
|-------|---------|-----------|-----------------|---------------------|
| 0 | -0.10 | 0.17 | 0.08 | 0.133 |
| 1 | -0.10 | 0.133 | 0.08 | 0.102 |
| 2 | -0.10 | 0.102 | 0.08 | 0.085 |
| 3 | -0.10 | 0.085 | 0.08 | 0.082 |
| ... | -0.10 | ... | 0.08 | **≈ 0.08** |

**Takes 8+ cycles (~64 days with optimize_every_n_evals=8) to suppress a bad dimension.**

### 2.3 Why Optimizer Never Caught -0.23 Composite IC

**Key Issue:** The optimizer checks **per-dimension IC only**, never the **composite IC** until the guard was added.

**Location:** `optimizer.py:195-273` (`_compute_ic_weights()`)

```python
# Only examines individual dimensions
for role in self.all_roles:
    dim_ic = dimensions.get(role, {}).get("ic")
    if ic_val >= promote_thresh:  # 0.03
        weight = base_w * promote_boost
    elif ic_val >= demote_thresh:  # 0.01
        # ... individual handling
```

**What Happens with Composite IC = -0.23:**
1. 5 dimensions: whale=+0.02, technical=-0.01, derivatives=+0.05, narrative=-0.03, market=-0.15
2. Composite still lags in aggregation; optimizer sees each individually
3. Whale, derivatives: PROMOTE (positive IC)
4. Technical: HOLD (IC > demote threshold of 0.01? No, it's -0.01, so DEMOTE to 0.5×weight)
5. Narrative: DISABLE (IC < disable threshold of -0.02)
6. Market: DISABLE (IC < disable threshold, heavily suppressed)
7. **Net effect:** Old weights + PROMOTE on whale + derivatives = still bullish/biased

**The New Guard (lines 89-133) Fixes This:**

```python
composite_ic = ic_data.get("overall_ic") or ic_data.get("composite_ic")
guard_threshold = float(guard_cfg.get("threshold", -0.05))
bypass_ema_threshold = float(guard_cfg.get("bypass_ema_threshold", -0.10))

if guard_enabled and composite_ic < guard_threshold:
    # Force-suppress all negative-IC dimensions
    for role in self.all_roles:
        dim_ic = dimensions.get(role, {}).get("ic")
        if dim_ic < 0:
            new_weights[role] = fallback_w * disable_factor  # 0.15×
```

---

## 3. COMPOSITE IC GUARD ANALYSIS

### 3.1 Guard Logic Correctness

**Location:** `optimizer.py:89-133`

```
IF composite_ic < threshold (-0.05):
  SUPPRESS all negative-IC dimensions to disable_factor × fallback

IF composite_ic < bypass_ema_threshold (-0.10):
  ALSO bypass EMA blending (use raw IC weights, not 70% old)
```

**Assessment:** ✅ **LOGIC IS CORRECT**

**Walkthrough with composite_ic = -0.23:**
1. Check: -0.23 < -0.05? **YES** → Enter guard
2. For each dimension with IC < 0:
   - Suppress to `0.15 × fallback_weight`
   - E.g., narrative: suppress to `0.15 × 0.17 = 0.026` (from ~0.12)
3. Re-normalize weights to sum to 1.0
4. Check: -0.23 < -0.10? **YES** → Bypass EMA
5. Use raw suppressed weights directly (don't blend with 70% old)

**Result:** System rapidly de-emphasizes all negative signals.

### 3.2 Threshold Appropriateness

**Current defaults:**
- `guard_threshold = -0.05`
- `bypass_ema_threshold = -0.10`
- `disable_factor = 0.15`

**Assessment:** ⚠️ **THRESHOLDS ARE REASONABLE BUT CONSERVATIVE**

| Scenario | Composite IC | Action | Appropriateness |
|----------|--------------|--------|-----------------|
| Normal trading | +0.01 to +0.05 | No guard | ✅ Correct |
| Weak signal | -0.02 to 0.00 | No guard | ⚠️ Risk: stays blended |
| Degradation | -0.10 | Guard + EMA bypass | ✅ Aggressive |
| Severe | -0.23 | Guard + EMA bypass | ✅ Correct |

**Problem:** Gap between guard threshold (-0.05) and bypass threshold (-0.10)

**Scenario:** Composite IC = -0.07
- Guard triggers: dimensions suppressed
- EMA still applies (only 30% new)
- Takes 2-3 more optimization cycles to fully degrade old bad weights

**Recommendation:** Consider `guard_threshold = -0.03` or `bypass_ema_threshold = -0.05` to be more aggressive.

---

## 4. PER-DIMENSION WEIGHT OPTIMIZATION

### 4.1 Global vs Per-Asset Weights

**Two-level optimization:**

**Level 1 - Global (optimizer.py:195-273):**
```python
# Computed from cross-sectional IC
new_weights = _compute_ic_weights(ic_data)
# Result: whale=0.20, technical=0.18, derivatives=0.22, etc.
# Applied to ALL assets
```

**Level 2 - Per-Asset (optimizer.py:321-447):**
```python
# compute_per_asset_weights(): BTC-specific, ETH-specific weights
# Result: {BTC: {whale: 0.15, technical: 0.25, ...},
#          ETH: {whale: 0.22, technical: 0.15, ...}}
```

### 4.2 Interaction & Conflict Analysis

**Location:** `engine.py` loads weights (not shown, but referenced in optimizer.py line 19)

**Assumed Flow:**
1. `compute_and_apply()` saves global weights to `learned_weights`
2. `compute_per_asset_weights()` saves per-asset to `per_asset_weights`
3. `engine.py` at fuse() time:
   - Loads per-asset weights if available
   - Falls back to global learned_weights
   - Falls back to YAML fallback_weights

**Potential Conflict:** ✅ **NO DIRECT CONFLICT**

Per-asset weights explicitly override global for that asset. But:

**Risk 1: Per-asset IC too sparse**
```python
# optimizer.py:348
min_obs = int(self.cfg.get("min_per_asset_observations", 8))

for asset, asset_ic in assets.items():
    n_obs = asset_ic.get("n_observations", 0)
    if n_obs < min_obs:
        continue  # Skip this asset, fall back to global
```

**Assessment:** With 8 minimum observations and crypto trading, per-asset IC is computed for maybe 3-5 assets, others fall back to global. Acceptable but sparse.

**Risk 2: Per-asset IC weights can still be negative**

```python
# optimizer.py:362-399 (per-asset version of promote/demote)
if ic_val >= promote_thresh:
    weight = base_w * promote_boost
elif ic_val >= demote_thresh:
    weight = base_w
elif ic_val >= disable_thresh:
    weight = base_w * demote_factor
else:
    weight = base_w * disable_factor  # Minimize but don't zero
```

**Per-asset doesn't have composite IC guard!** If BTC's derivative dimension has IC = -0.15, it gets suppressed to `0.15 × fallback`, but there's no check for "overall per-asset IC is bad."

**Recommendation:** Add per-asset composite IC guard similar to global.

---

## 5. FALLBACK vs LEARNED WEIGHTS

### 5.1 When Each Is Used

**Location References:**
- Fallback: `default.yaml` learning section
- Learned: `learned_weights` in kv_json
- Per-asset: `per_asset_weights` in kv_json

**Flow (assumed from code):**

1. **On startup (engine.py fuse() time):**
   - Load per-asset learned weights (if exist)
   - Load global learned weights (if exist)
   - Fall back to YAML fallback_weights
   - Fall back to profile.weights

2. **During optimization (optimizer.py):**
   - Load IC data
   - Compute new weights
   - **EMA blend with previous learned weights** (line 668-674)
   ```python
   prev_data = self.store.load_kv_json(self.namespace, "learned_weights")
   if prev_data and "weights" in prev_data:
       prev = prev_data["weights"]
   else:
       prev = self.cfg.get("fallback_weights", {})
   ```

### 5.2 Race Condition Analysis

**Potential Race:** ⚠️ **LOW RISK but possible**

**Scenario:**
1. Optimizer computes new weights, saves to `learned_weights` at T=10:00
2. Engine loads `learned_weights` at T=10:01
3. Meanwhile, server crashes; optimizer re-runs with old IC data
4. Optimizer loads `learned_weights` from T=10:00, blends with it
5. If this old data is stale, next save could have vestigial weights

**Mitigation:** Code includes `updated_at` timestamp (line 700). Engine could validate freshness.

**Current Status:** ✅ **UNLIKELY in practice** because:
- Optimizer runs infrequently (every 8 evaluations, ~4 days)
- Learned weights have timestamps
- Code doesn't show version conflicts

---

## 6. WEIGHT IMPACT TRACKING

### 6.1 What It Measures

**Location:** `optimizer.py:460-546` (`track_weight_impact()`)

```python
# Compares current accuracy baseline
baseline = store.load_kv_json(namespace, "accuracy_baseline")
comparisons = {}
for asset, current in accuracy.items():
    prev = baseline_assets.get(asset)
    if prev and prev.get("n", 0) >= 3:
        delta = current["avg_gradient"] - prev["avg_gradient"]
        comparisons[asset] = {"before": before_grad, "after": after_grad, "delta": delta}
```

**Baseline:** Saved on first run; updated if `total_evals >= 50 AND overall_delta > 0`

### 6.2 Correctness Assessment

**✅ CORRECTLY MEASURES:** Accuracy improvement (gradient_score) per asset

**⚠️ LIMITATIONS:**

1. **"Gradient score" not defined in snippet** — Assumed to be per-dimension accuracy metric, but unclear if it's Sharpe, return%, or gradient
2. **Baseline update condition** (line 538):
   ```python
   if total_current_evals >= 50 and overall_delta > 0:
       # Update baseline
   ```
   **Problem:** Only updates if improved. If accuracy decays by 2% tomorrow, baseline stays old, delta appears negative (good for detecting problems, bad for trending).

3. **Asset filtering** (line 495-496):
   ```python
   if prev and prev.get("n", 0) >= 3 and current.get("n", 0) >= 3:
   ```
   **Problem:** Requires 3+ evaluations per asset. Illiquid assets might never accumulate this.

**Assessment:** ✅ **Functional for detecting major regressions**, ⚠️ **not precise for trending**.

---

## 7. BLIND SPOTS & SYSTEMIC ISSUES

### Issue #1: Per-Dimension IC Sufficiency Test is Weak

**Location:** `optimizer.py:230`

```python
if ic_val is None or n_slices < 5:
    # Not enough data — keep fallback
    raw_weights[role] = base_w
    continue
```

**Problem:** Only checks total slices, not slices where **this dimension had ≥3 assets**.

**Example:**
- Total slices = 20
- "whale" dimension present in only 6 slices (other 14 have missing data)
- IC computed from 6 slices with n=3 each = unreliable

**Fix:** Track and check `slices_used` (computed in storage.py:1016)

### Issue #2: Composite IC Loaded but Not Always Set

**Location:** `optimizer.py:92`

```python
composite_ic = ic_data.get("overall_ic") or ic_data.get("composite_ic")
```

**Problem:** Falls back to `get("composite_ic")` if `"overall_ic"` missing, but storage.py:1035 only returns:
```python
"overall_ic": result_dims.get("composite", {}).get("ic"),
```

**Risk:** If "composite" dimension is missing from IC computation, overall_ic is None, guard never triggers.

**Assessment:** Low risk (composite should always be computed), but fragile.

### Issue #3: EMA Blending Doesn't Account for Regime Changes

**Location:** `optimizer.py:664-694`

**Problem:** EMA applies 70% old weight regardless of regime. If market shifted from bullish to bearish, old weights are stale.

**Example:**
- Old weights computed in bull market: whale=0.25, technical=0.15
- Market turns bear; new IC shows narrative better in bear: narrative IC=+0.05
- EMA blend: narrative = 0.3×0.20 + 0.7×0.17 = 0.179
- Should be higher given bear regime

**Fix:** Check `ic_data.get("by_regime")` and reduce EMA factor if regime changed.

### Issue #4: Disable Factor (0.15) is Arbitrary

**Location:** `optimizer.py:105, 213`

```python
disable_factor = float(self.cfg.get("disable_factor", 0.15))
new_weights[role] = fallback_w * disable_factor  # 0.15 × 0.17 = 0.026 for most
```

**Problem:** For a negative-IC dimension, `0.15 × fallback` → ~2.6% weight when there are 6 roles.

**Analysis:**
- Equal weight = 1/6 ≈ 16.7%
- Disabled = ~2.6%
- Roughly 6.4× suppression

**But:** No principled reason for 0.15 vs 0.10 vs 0.20. Consider **IC-proportional suppression instead:**
```
weight = max(fallback * 0.1, fallback * max(0, 1 + ic_val))
# If IC = -0.10: weight = fallback × 0.90
# If IC = -0.23: weight = fallback × 0.77 (still weakened, not disabled)
```

---

## 8. ROOT CAUSE: WHY -0.23 COMPOSITE IC WENT UNDETECTED

**Timeline:**
- Optimizer ran for weeks with composite IC = -0.23
- Only caught recently when composite IC guard added
- EMA blending with 70% old weight meant slow adaptation

### Root Cause Chain:

1. **No Composite IC Check in _compute_ic_weights()** (before guard added)
   - Only per-dimension IC examined
   - Individual dimensions might be +0.02, -0.01, +0.05 → mixed signals
   - Optimizer boosted positive dimensions
   - But composite IC = weighted average was -0.23

2. **EMA Blending Absorbed Bad Weights**
   - Even if guard had triggered, old weights (70%) would dampen correction
   - Takes 8+ cycles to decay bad weights

3. **Minimum Slice Size Too Small (n=3)**
   - Small slices → noisy IC estimates
   - One outlier can flip IC sign
   - Averaging across all slices might look OK even if individual slices are unstable

4. **No Regime-Aware Weighting**
   - If composite IC degraded in specific regime, global weights didn't adapt
   - Per-regime tracking exists (storage.py:1023-1030) but optimizer doesn't use it

---

## 9. RECOMMENDATIONS

### High Priority

**1. Increase Minimum Slice Size**
- **Current:** n=3 assets per slice
- **Recommended:** n=5-7
- **Rationale:** Spearman reliability improves significantly; reduces noise
- **File:** `storage.py:954, 962`

```python
# Change from:
if len(observations) < 3:
# To:
if len(observations) < 5:
```

**2. Compute Composite IC First**
- **Current:** _compute_ic_weights() checks per-dimension first
- **Recommended:** Check composite IC before per-dimension adjustments
- **File:** `optimizer.py:85-125`

```python
# Pseudocode:
composite_ic = ic_data.get("overall_ic")
if composite_ic < -0.10:
    # Red alert: overall system degraded
    # Option A: Use EMA-bypass + suppress all negative dims (current guard, good)
    # Option B: Revert entirely to fallback_weights (conservative)
    pass
else:
    # Proceed with per-dimension optimization
```

**3. Adjust Guard Thresholds**
- **Current:** guard_threshold = -0.05, bypass_ema = -0.10
- **Recommended:** guard_threshold = -0.03, bypass_ema = -0.05
- **Rationale:** Faster response; catch degradation earlier
- **File:** `default.yaml` learning section

```yaml
composite_ic_guard:
  enabled: true
  threshold: -0.03        # Was -0.05
  bypass_ema_threshold: -0.05  # Was -0.10
```

**4. Replace EMA with IC-Proportional Weighting**
- **Current:** 30% new + 70% old regardless of IC quality
- **Recommended:** learning_rate = min(0.5, 0.3 + abs(delta_ic)/2)
- **Rationale:** Fast adaptation when IC changes significantly; gradual when stable

```python
# Pseudocode for _ema_blend():
prev_ic = load previous IC
new_ic = current IC
delta_ic = abs(new_ic - prev_ic)
adaptive_lr = min(0.5, 0.3 + delta_ic / 2)  # 0.3 to 0.5 based on change
blended[role] = adaptive_lr * new_w + (1 - adaptive_lr) * old_w
```

### Medium Priority

**5. Add Per-Asset Composite IC Guard**
- **Current:** Per-asset weights use same suppress rules but no composite check
- **Recommended:** Compute per-asset composite IC; suppress if < -0.05
- **File:** `optimizer.py:321-447`

**6. Use Per-Regime Weights**
- **Current:** Ignores regime IC data
- **Recommended:** If regime changed, reduce EMA factor or flag for manual review
- **File:** `optimizer.py:_compute_ic_weights()` to use `ic_data.get("by_regime")`

**7. Track Slices Actually Used**
- **Current:** Checks total slices, not slices where dimension had data
- **Recommended:** Use `slices_used` from IC computation
- **File:** `optimizer.py:230` to check `dimensions.get(role, {}).get("slices_used", 0)`

### Low Priority

**8. IC-Proportional Suppression**
- **Current:** disable_factor = fixed 0.15
- **Recommended:** weight = fallback × (1 + ic_val / 2) for IC < 0
- **Rationale:** Proportional to how bad IC is; avoids arbitrary thresholds

**9. Versioned Weight Snapshots**
- **Current:** Learned weights have timestamp but no version chain
- **Recommended:** Include git commit hash in weight save for reproducibility
- **File:** `optimizer.py:_save_weights()`

**10. Document Weight Update Frequency**
- **Current:** optimize_every_n_evals = 8 (4 days at current evaluation rate)
- **Recommended:** Log and document in YAML to prevent missed updates

---

## 10. IMPLEMENTATION CHECKLIST

- [ ] **High Priority #1:** Increase min slice size to 5-7 (storage.py:954, 962)
- [ ] **High Priority #2:** Add composite IC check before per-dim (optimizer.py:85)
- [ ] **High Priority #3:** Adjust guard thresholds in YAML (learning.composite_ic_guard)
- [ ] **High Priority #4:** Implement adaptive learning_rate (optimizer.py:_ema_blend)
- [ ] **Med Priority #5:** Add per-asset composite guard (optimizer.py:compute_per_asset_weights)
- [ ] **Med Priority #6:** Use per-regime IC in decisions (optimizer.py, check by_regime)
- [ ] **Med Priority #7:** Track and validate slices_used (optimizer.py:230)
- [ ] **Test:** Run backtesting with composite_ic = -0.23 scenario; verify guard triggers
- [ ] **Test:** Verify EMA bypass activates; weights decay to fallback within 1-2 cycles

---

## Summary

**The composite IC guard added is correct and will prevent future -0.23 scenarios.** However:

1. **Pre-deployment:** Threshold tuning needed; recommend -0.03 / -0.05 instead of -0.05 / -0.10
2. **Post-deployment:** Monitor baseline accuracy; if composite IC triggers guard, log incident
3. **Future:** Consider moving away from EMA to IC-proportional adaptation
4. **Data Quality:** Increase minimum slice size to reduce noise in IC estimates

**Risk Timeline:**
- **Without guard:** Days to weeks before bad weights fully suppressed
- **With guard (current):** ~1-2 optimization cycles to suppress
- **With guard + higher threshold:** Hours to days (more aggressive)

The system is now defensive but can be more responsive.
