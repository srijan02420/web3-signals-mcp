# E5: Ensemble Methods & Meta-Learning for Trading Signal Combination

**Research Agent:** E5
**Date:** 2026-03-16
**Scope:** How quant funds combine signals, ensemble methods, meta-learning, calibration techniques, and specific recommendations for replacing the 7-layer weight cascade.

---

## Executive Summary

**The 7-layer weight cascade in `signal_fusion/engine.py` is architecturally flawed.** It applies sequential multiplicative transformations (asymmetric weights -> accuracy scaling -> regime weighting -> F&G regime scoring -> data tier reweighting -> velocity dampening -> abstain check) where each layer can amplify, invert, or nullify the effects of previous layers. This is the opposite of how successful quant systems combine signals.

**Key findings:**

1. **Production quant systems use parallel combination, not sequential cascading.** Signals are calibrated independently, then combined via linear combination, Bayesian averaging, or shallow stacking. Deep sequential cascades are an anti-pattern.

2. **The "forecast-then-adjust" paradigm is wrong.** Our system builds a composite score and then applies 6 layers of post-hoc adjustments. The industry standard is "calibrate-then-combine": each signal is individually calibrated to a common scale (probability or z-score), then combined in a single step.

3. **Simple methods dominate in low-data regimes.** With 5 signals and limited historical data, inverse-variance weighting or equal weighting consistently outperforms complex stacking. Our system is massively overfit to the cascade logic.

4. **The cascade can invert good signals** because multiplicative dampening factors applied sequentially create non-linear interactions. A signal scoring 75 (bullish) can become 48 (slightly bearish) after regime dampening * velocity dampening * trend dampening stack.

**Recommended architecture:** Replace the 7-layer cascade with a 3-phase pipeline:
- Phase 1: Independent signal calibration (Platt scaling or isotonic regression per dimension)
- Phase 2: Single-step combination (inverse-variance weighted average with regime-conditional weights)
- Phase 3: Confidence gating (abstain based on calibrated ensemble uncertainty, not distance-from-center)

---

## Table of Contents

1. [How Quant Funds Actually Combine Signals](#1-how-quant-funds-actually-combine-signals)
2. [Why Sequential Cascades Fail](#2-why-sequential-cascades-fail)
3. [Signal Combination Methods: A Taxonomy](#3-signal-combination-methods-a-taxonomy)
4. [Calibration Techniques](#4-calibration-techniques)
5. [Meta-Learning for Dynamic Signal Weighting](#5-meta-learning-for-dynamic-signal-weighting)
6. [Regime-Aware Signal Aggregation](#6-regime-aware-signal-aggregation)
7. [When Simple Beats Complex](#7-when-simple-beats-complex)
8. [Diagnosing Our Cascade: Specific Failure Modes](#8-diagnosing-our-cascade-specific-failure-modes)
9. [Recommended Architecture](#9-recommended-architecture)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Key References](#11-key-references)

---

## 1. How Quant Funds Actually Combine Signals

### 1.1 The Standard Alpha Combination Framework

The canonical approach used at systematic funds (AQR, Two Sigma, DE Shaw, Man Group, Winton, Citadel Securities' systematic arm) follows a well-established pipeline:

```
Raw Data -> Feature Engineering -> Signal Generation -> Signal Calibration -> Signal Combination -> Risk Sizing -> Execution
```

The critical insight: **signal combination is a single step**, not a multi-layer cascade. Each signal is individually processed to produce a calibrated forecast (typically a z-score or probability), and then signals are combined via one of:

#### Method 1: Linear Combination (Most Common)

```python
# Industry standard: weighted linear combination of calibrated signals
alpha_combined = sum(w_i * signal_i for i in range(n_signals))
# where each signal_i is already calibrated to a common scale
```

**Why it dominates:** Grinold & Kahn's "Fundamental Law of Active Management" shows that the information ratio of a combined signal is:

```
IR_combined = sqrt(sum(IC_i^2 * breadth_i))
```

When signals have low correlation, linear combination is provably optimal under Gaussian assumptions. This is the workhorse at most systematic funds.

#### Method 2: Bayesian Model Averaging (BMA)

```python
# Each model/signal gets a posterior probability weight
P(model_k | data) proportional to P(data | model_k) * P(model_k)
prediction = sum(P(model_k | data) * prediction_k for k in models)
```

Used at firms like Man AHL for combining macro signals with varying reliability. Naturally handles model uncertainty and can down-weight signals that have been historically unreliable.

#### Method 3: Information Coefficient (IC) Weighting

```python
# Weights proportional to rolling IC (signal-return correlation)
weights = rolling_IC / sum(rolling_IC)  # after clipping negatives to 0
alpha = sum(weights * signals)
```

This is close to what our optimizer attempts, but done in a single step without cascading adjustments. The key difference: IC is computed on the raw calibrated signal, not on a post-adjusted composite.

### 1.2 What Successful Funds Do NOT Do

1. **They do not apply sequential multiplicative adjustments** to a composite score. There is no "regime dampening layer" applied after "accuracy scaling layer" applied after "directional weight selection layer."

2. **They do not build a score and then adjust it.** They calibrate inputs independently and combine once.

3. **They do not use the same signal for both scoring and weighting.** Our system uses Fear & Greed to (a) score the market dimension, (b) select asymmetric weights, (c) apply regime weight shifts, and (d) apply score dampening. This quadruple-counting creates feedback loops.

4. **They do not have "inversion" logic embedded in scoring.** Our contrarian scoring rules (where low F&G = bullish) are an implicit forecast that should be a separate signal, not baked into the scoring function of another signal.

### 1.3 The Two Sigma / AQR Approach (Documented in Academic Papers)

Based on published research from these firms:

- **Signal Orthogonalization:** Before combining, signals are orthogonalized (residualized against each other) so that overlapping information is not double-counted. Our system explicitly double-counts (F&G affects market score, regime weights, and dampening simultaneously).

- **Decay-Weighted IC:** Signal weights use exponentially-weighted IC with half-lives of 60-250 days, not the EMA blend with half-life of 2.1 cycles that our optimizer uses.

- **Turnover Constraints:** Combined signals are constrained to limit portfolio turnover, which implicitly smooths regime transitions rather than using abrupt regime-switching logic.

---

## 2. Why Sequential Cascades Fail

### 2.1 The Mathematical Problem

Our 7-layer cascade applies transformations sequentially:

```
score_final = abstain_check(
  velocity_dampen(
    composite(
      tier_reweight(
        fg_regime_shift(
          regime_shift(
            accuracy_scale(
              asymmetric_select(base_weights)
            )
          )
        )
      ) * dimension_scores
    )
  )
)
```

Each layer is a function `f_i` applied to the output of the previous layer. The final output is:

```
y = f_7(f_6(f_5(f_4(f_3(f_2(f_1(x)))))))
```

**Problem 1: Gradient Vanishing / Exploding.** When each `f_i` is a multiplicative dampening factor in [0.3, 1.0], the product of multiple dampening factors can crush legitimate signals:

```
# Example: BTC in fear market, slight downtrend, with accelerating RSI decline
# Starting composite: 68 (moderate buy based on whale + technical signals)
# Layer 4 (F&G regime dampening): factor 0.6 -> distance from 50 * 0.6 = 50 + 18*0.6 = 60.8
# Layer 6 (velocity dampening): factor 0.5 -> distance from 50 * 0.5 = 50 + 10.8*0.5 = 55.4
# Layer 3 (regime weight shift already changed weights): could push further
# Final: 55.4 from 68 = a moderate buy became barely a buy
# Meanwhile the whale + technical signals were CORRECT
```

**Problem 2: Layer Interaction Creates Emergent Behavior.** The combination of asymmetric weight selection (Layer 1) with regime weight shifts (Layer 3 + Layer 4) means the weights applied to a signal depend on the signal's own direction. This creates self-reinforcing or self-canceling loops:

```
# Self-canceling example:
# Raw signals lean bullish -> Layer 1 selects bullish weights (boosting market)
# But F&G is extreme fear -> Layer 4 shifts weights to dampen market
# Net effect: the bullish weight boost and the fear dampening partially cancel
# Result: nearly identical to default weights, but with added noise
```

**Problem 3: Order Dependence.** `f_3(f_2(x)) != f_2(f_3(x))` in general. The order in which regime shifts and accuracy scaling are applied matters, but there's no principled reason for one order over another.

### 2.2 Empirical Evidence from Our System

From the I4 analysis (weight optimizer deep-dive), we know:
- Composite IC = -0.23 (the combined output predicts returns in the WRONG direction)
- Individual dimension ICs vary: some positive, some negative
- The cascade transforms individually-reasonable signals into a collectively-wrong composite

This is the textbook signature of a sequential cascade problem: each layer is locally reasonable but globally destructive.

### 2.3 The Signal Inversion Mechanism

From examining `signal_fusion/engine.py`, the specific inversion path:

1. **Contrarian Scoring (in `_score_market`, `_score_derivatives`):** Low F&G -> high market dimension score (contrarian "buy the fear" logic)
2. **F&G Regime Dampening (lines 419-428):** In fear markets, dampen dimensions with high scores toward 50
3. **Net Effect:** Step 1 creates a bullish signal from bearish data, then Step 2 partially reverses it. The degree of reversal depends on config parameters that were hand-tuned, not learned.

This is equivalent to: `signal = bullish_boost * dampening_factor` where both factors are derived from the same input (F&G). The product is a noisy, attenuated version of what a single well-calibrated signal would produce.

---

## 3. Signal Combination Methods: A Taxonomy

### 3.1 Linear Methods (Recommended Starting Point)

#### Equal Weighting (1/N)

```python
composite = mean(calibrated_signals)
```

**When to use:** When you have < 100 historical data points per signal, or when signal quality is uncertain. DeMiguel, Garlappi & Uppal (2009) showed 1/N outperforms mean-variance optimization in most realistic settings with estimation error.

**Relevance to our system:** With limited backtest history and uncertain IC estimates, equal weighting on well-calibrated signals would likely outperform our current cascade.

#### Inverse-Variance Weighting

```python
# Weight inversely proportional to signal variance
w_i = (1 / var_i) / sum(1 / var_j for j in signals)
composite = sum(w_i * signal_i)
```

**When to use:** When signals have varying reliability/noise levels but you lack enough data for full IC estimation. This is a simple, robust method that naturally down-weights noisy signals.

#### IC-Weighted Combination (Treynor-Black Model Analog)

```python
# Weight proportional to Information Coefficient
w_i = max(IC_i, 0) / sum(max(IC_j, 0) for j in signals)
composite = sum(w_i * signal_i)
```

**When to use:** When you have reliable IC estimates (requires > 200-500 observations per signal). Our optimizer attempts this but applies it through a cascade rather than directly.

### 3.2 Ensemble Methods

#### Stacking (Stacked Generalization)

**Concept:** Train a meta-model to combine base model predictions. Level-0 models produce predictions; Level-1 model learns optimal combination.

```python
# Walk-forward stacking for time series
for each test_fold:
    train Level-0 models on training data
    generate out-of-fold predictions on validation set
    train Level-1 model on (predictions, actual_returns)
    predict on test_fold using Level-0 predictions + Level-1 weights
```

**Critical requirements for trading:**
- **Walk-forward only:** No future leakage. Each fold trains only on past data.
- **Regularized meta-learner:** L1 (Lasso) or L2 (Ridge) regularization on the Level-1 model prevents overfitting to specific signal combinations.
- **Minimum data:** Requires 500+ observations for stable Level-1 estimates with 5 signals.

**Relevance:** Our system does NOT have enough historical data for reliable stacking. The optimizer.py has < 100 evaluated signal-return pairs. Stacking would overfit catastrophically.

#### Boosting (AdaBoost, XGBoost, LightGBM)

**Concept:** Sequential ensemble where each model corrects errors of previous models.

```python
# Gradient boosting for signal combination
model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
model.fit(X=signal_matrix, y=forward_returns,
          eval_set=[(X_val, y_val)], early_stopping_rounds=10)
```

**In trading context:**
- Used at some quant funds for non-linear signal combination
- Captures interaction effects (e.g., "whale accumulation only predicts returns when RSI < 30")
- Requires substantial data (1000+ observations) and careful cross-validation
- Prone to overfitting in non-stationary environments

**Relevance:** Premature for our system. We need to fix the calibration/combination architecture first. Boosting on badly-calibrated inputs will learn the cascade's artifacts.

#### Bagging (Random Forests for Signal Combination)

**Concept:** Train multiple models on bootstrap samples and average predictions.

```python
# Random forest meta-learner
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=3)
rf.fit(signal_matrix, forward_returns)
```

**Advantages:** More robust than single models, handles non-linear interactions, provides feature importance.

**Disadvantages:** Still requires substantial data; may overfit to training regime.

**Relevance:** Could be useful once we have 500+ observations with proper calibration.

### 3.3 Bayesian Methods

#### Bayesian Model Averaging (BMA)

```python
# For K models/signals, posterior model probability:
# P(M_k | D) = P(D | M_k) * P(M_k) / P(D)
# Combined prediction:
# E[y | D] = sum_k P(M_k | D) * E[y | M_k, D]

# In practice with conjugate priors:
# Use BIC approximation for model evidence
BIC_k = -2 * log_likelihood_k + p_k * log(n)
w_k = exp(-0.5 * BIC_k) / sum(exp(-0.5 * BIC_j))
prediction = sum(w_k * prediction_k)
```

**Advantages:**
- Naturally handles model uncertainty
- Down-weights models that have been historically unreliable
- No overfitting risk from the combination step itself
- Theoretically optimal under correct prior specification

**Disadvantages:**
- Prior specification matters (but can use uninformative priors)
- Computational cost for exact inference (but BIC approximation is fast)
- Assumes models are "correct" — doesn't handle model misspecification well

**Relevance:** **Highly recommended for our use case.** BMA with uninformative priors is equivalent to IC-weighting in the Gaussian case, but provides uncertainty estimates that can drive the abstain decision.

#### Bayesian Hierarchical Models

```python
# Signal k at time t:
# signal_kt = alpha_k + beta_k * true_state_t + epsilon_kt
# true_state_t ~ N(mu_regime, sigma_regime)  [regime-dependent]
# alpha_k, beta_k ~ N(mu_alpha, sigma_alpha)  [hierarchical prior]

# Posterior inference via MCMC or variational inference
# Naturally handles: varying signal quality, regime changes, calibration
```

**Advantage:** Unifies calibration, combination, and regime-awareness in a single probabilistic model. No cascading needed.

**Disadvantage:** Complex to implement, requires MCMC or variational inference, computationally expensive.

**Relevance:** The "ideal" long-term solution but significant implementation effort. Recommend as Phase 3 of the roadmap.

---

## 4. Calibration Techniques

**The most important single improvement we can make.** Our current system outputs raw scores on a 0-100 scale where the meaning of "70" varies wildly across dimensions and market conditions. Calibration maps each signal's output to a consistent, interpretable scale.

### 4.1 Platt Scaling (Sigmoid Calibration)

**What it does:** Fits a logistic regression to map raw scores to calibrated probabilities.

```python
# For each dimension's raw score s:
# P(positive_return | s) = 1 / (1 + exp(-(A*s + B)))
# Fit A, B by maximum likelihood on historical (score, actual_outcome) pairs

import numpy as np
from scipy.optimize import minimize

def platt_scaling(raw_scores, binary_outcomes):
    """Fit Platt scaling parameters A, B."""
    def neg_log_likelihood(params):
        A, B = params
        p = 1.0 / (1.0 + np.exp(-(A * raw_scores + B)))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(binary_outcomes * np.log(p) + (1 - binary_outcomes) * np.log(1 - p))

    result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method='L-BFGS-B')
    return result.x  # A, B

def apply_platt(score, A, B):
    """Apply calibration to a new score."""
    return 1.0 / (1.0 + np.exp(-(A * score + B)))
```

**Requirements:**
- 50-200 historical (score, outcome) pairs per dimension
- Binary outcomes (positive/negative return) or continuous returns discretized
- Recalibrate periodically (rolling window of 90-180 days recommended)

**Advantages:**
- Simple (2 parameters per signal)
- Well-understood statistical properties
- Robust with limited data
- Output is a proper probability

**When it fails:**
- Non-monotonic score-outcome relationship (score of 80 is LESS predictive than score of 60)
- This might apply to our contrarian-inverted signals

**Relevance:** **Immediate recommendation for our system.** Apply Platt scaling to each dimension's raw score before combination. This replaces accuracy scaling + regime dampening with a principled calibration.

### 4.2 Isotonic Regression

**What it does:** Fits a non-parametric, monotonically non-decreasing function from raw scores to probabilities.

```python
from sklearn.isotonic import IsotonicRegression

def isotonic_calibration(raw_scores, binary_outcomes):
    """Fit isotonic regression calibrator."""
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(raw_scores, binary_outcomes)
    return ir  # call ir.predict(new_scores) later

# Usage:
calibrated = ir.predict(raw_score)
```

**Advantages over Platt:**
- Handles non-linear calibration curves
- Does not assume sigmoid shape
- Can capture "middle scores are better" or "extreme scores are worse" patterns

**Disadvantages:**
- Requires more data (200+ observations recommended)
- Can overfit with small samples
- Monotonicity constraint may not be appropriate for all signals

**Relevance:** Better than Platt for our contrarian signals where the score-outcome relationship may not be sigmoid-shaped. However, requires more data than we currently have for most dimensions.

### 4.3 Temperature Scaling (Simple, Fast)

```python
# Single parameter T (temperature) learned on validation set:
# calibrated = softmax(logits / T) for classification
# For regression-style signals:
# calibrated = (score - 50) / T + 50  [scale distance from center]

def temperature_scale(scores, outcomes):
    """Find optimal temperature parameter."""
    from scipy.optimize import minimize_scalar

    def loss(T):
        calibrated = (scores - 50) / T + 50
        calibrated = np.clip(calibrated, 0, 100)
        # Use cross-entropy or Brier score as loss
        probs = calibrated / 100.0
        return np.mean((probs - outcomes)**2)  # Brier score

    result = minimize_scalar(loss, bounds=(0.1, 10.0), method='bounded')
    return result.x
```

**Relevance:** The simplest possible calibration. Could replace our entire accuracy_scaling + regime_dampening + fg_regime_scoring stack with a single learned parameter per dimension, per regime.

### 4.4 Venn-ABERS Calibration (State of the Art)

A distribution-free calibration method that provides calibrated probability intervals rather than point estimates:

```python
# For each test point, produces [p_lower, p_upper] instead of single p
# The true probability is guaranteed to lie in this interval
# with frequency validity (not just in expectation)
```

**Advantages:**
- Provides uncertainty bounds on calibrated probabilities
- Distribution-free (no assumptions about score distribution)
- Valid finite-sample coverage guarantees

**Relevance:** The uncertainty bounds naturally support the "abstain" decision: if `p_upper - p_lower > threshold`, the signal is too uncertain to act on. This would replace our distance-from-center abstain logic with something principled.

### 4.5 Calibration for Our Specific System

**Current state:** Our 0-100 scores have no calibration. A score of 70 from the whale dimension means something completely different from 70 from the technical dimension. The cascade tries to compensate with per-dimension weights and dampening, but this is not calibration — it's ad-hoc rescaling.

**Proposed calibration pipeline:**

```python
class SignalCalibrator:
    def __init__(self):
        self.calibrators = {}  # {dimension: {regime: calibrator}}

    def fit(self, dimension, regime, raw_scores, outcomes):
        """Fit calibrator for one dimension in one regime."""
        if len(raw_scores) < 50:
            # Too few samples: use temperature scaling (1 parameter)
            self.calibrators[(dimension, regime)] = TemperatureCalibrator()
        elif len(raw_scores) < 200:
            # Moderate samples: use Platt scaling (2 parameters)
            self.calibrators[(dimension, regime)] = PlattCalibrator()
        else:
            # Sufficient samples: use isotonic regression
            self.calibrators[(dimension, regime)] = IsotonicCalibrator()

        self.calibrators[(dimension, regime)].fit(raw_scores, outcomes)

    def calibrate(self, dimension, regime, raw_score):
        """Return calibrated probability for a raw score."""
        key = (dimension, regime)
        if key not in self.calibrators:
            key = (dimension, 'default')  # fall back to regime-agnostic
        return self.calibrators[key].predict(raw_score)
```

---

## 5. Meta-Learning for Dynamic Signal Weighting

### 5.1 The Meta-Learning Framing

Meta-learning asks: "Given a new market condition, what combination weights should I use based on how signals have performed in similar conditions historically?"

This is fundamentally different from our current approach of hard-coding regime-dependent weight shifts in YAML.

### 5.2 Approaches

#### Regime-Conditional Weighting (Simplest Meta-Learning)

```python
# Learn separate weight sets for each detected regime
# At inference, detect regime, look up weights, apply

regimes = detect_market_regime(current_data)  # e.g., "trending_bull"

# Pre-computed weights per regime (from historical IC analysis)
regime_weights = {
    "trending_bull": {"whale": 0.15, "technical": 0.30, "derivatives": 0.20, ...},
    "trending_bear": {"whale": 0.20, "technical": 0.25, "derivatives": 0.25, ...},
    "ranging": {"whale": 0.25, "technical": 0.15, "derivatives": 0.15, ...},
    "crisis": {"whale": 0.30, "technical": 0.10, "derivatives": 0.30, ...},
}

weights = regime_weights.get(regimes, default_weights)
composite = sum(weights[dim] * calibrated_signals[dim] for dim in dimensions)
```

**This is essentially what our system tries to do**, but our implementation is wrong because:
1. We apply regime shifts as multiplicative adjustments to existing weights rather than looking up pre-computed optimal weights
2. We apply F&G regime AND market regime AND trend regime as separate layers instead of a unified regime classification
3. We don't learn the regime-conditional weights from data; they're hand-tuned in YAML

#### Online Learning / Bandit Approaches

```python
# EXP3 or EXP4 algorithm for adversarial weight selection
# Maintains a distribution over weight vectors
# Updates based on realized loss after each prediction

class Exp3WeightLearner:
    def __init__(self, n_signals, eta=0.1):
        self.weights = np.ones(n_signals) / n_signals
        self.eta = eta

    def get_weights(self):
        # Mix with uniform for exploration
        return (1 - self.eta) * self.weights + self.eta / len(self.weights)

    def update(self, losses):
        # Exponential weight update
        self.weights *= np.exp(-self.eta * losses)
        self.weights /= self.weights.sum()
```

**Advantage:** No regime detection needed; adapts continuously. Regret-bounded performance guarantees.

**Disadvantage:** Slow adaptation (needs many observations); doesn't exploit regime structure.

**Relevance:** Good supplement to regime-conditional weighting. Can detect when regime labels are wrong.

#### Neural Meta-Learner (MAML-style)

```python
# Model-Agnostic Meta-Learning for signal combination
# Train a small network that takes:
#   - Current signal values (5 dimensions)
#   - Recent signal history (last K predictions and outcomes)
#   - Market context features (volatility, regime indicators)
# And outputs:
#   - Optimal combination weights for this specific situation
#   - Confidence estimate

class MetaSignalCombiner(nn.Module):
    def __init__(self, n_signals, context_dim):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.weight_head = nn.Sequential(
            nn.Linear(16 + n_signals, 16),
            nn.ReLU(),
            nn.Linear(16, n_signals),
            nn.Softmax(dim=-1)
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(16 + n_signals, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, signals, context):
        ctx = self.context_encoder(context)
        combined = torch.cat([signals, ctx], dim=-1)
        weights = self.weight_head(combined)
        confidence = self.confidence_head(combined)
        prediction = (weights * signals).sum(dim=-1)
        return prediction, weights, confidence
```

**Advantage:** Can learn complex, non-linear signal interactions conditioned on context.

**Disadvantage:** Requires thousands of training examples; difficult to interpret; can overfit spectacularly.

**Relevance:** Future exploration once we have > 1000 evaluated signal-outcome pairs.

### 5.3 The "Learning to Learn Weights" Problem

A fundamental tension in meta-learning for trading:

- **More parameters = better in-sample fit but worse out-of-sample generalization**
- **Non-stationarity means the "meta-distribution" of tasks is itself changing**
- **Low data volume per regime makes within-regime learning fragile**

**Practical solution:** Use hierarchical Bayesian priors that pool information across regimes:

```python
# Instead of separate weights per regime (fragile with small samples):
# w_regime ~ N(w_global, sigma_regime)
# w_global ~ N(equal_weights, sigma_global)

# This allows regime-specific weights to deviate from the global average
# but shrinks them back when data is scarce (Bayesian shrinkage)
```

This is the single most important insight from meta-learning research for our use case: **pool across regimes with Bayesian shrinkage rather than maintaining independent weight sets per regime.**

---

## 6. Regime-Aware Signal Aggregation

### 6.1 The Core Problem

Market regimes (trending, mean-reverting, crisis) change which signals are informative:
- In trending markets: momentum/technical signals dominate
- In ranging markets: mean-reversion/contrarian signals dominate
- In crisis: whale/flow signals and derivatives signals dominate
- In extreme sentiment: contrarian signals have highest IC but also highest variance

Our system tries to handle this with 3 separate layers (regime weighting, F&G regime scoring, trend override). This triple-regime-adjustment is both redundant and contradictory.

### 6.2 Unified Regime Framework

**Recommended approach:** Single regime classification with learned conditional weights.

```python
class UnifiedRegimeDetector:
    """
    Combines multiple regime indicators into a single regime state.
    Replaces: regime_weighting + fg_regime_scoring + trend_override
    """

    def __init__(self):
        self.regime_features = [
            'btc_distance_from_ma30',    # Trend strength
            'ma7_ma30_alignment',         # Trend confirmation
            'fear_greed_index',           # Sentiment regime
            'realized_volatility_30d',    # Volatility regime
            'btc_rsi_14d',               # Overbought/oversold
            'funding_rate_avg',           # Derivatives sentiment
        ]

    def detect_regime(self, features: dict) -> str:
        """
        Output one of: trending_bull, trending_bear, ranging,
                       fear_capitulation, greed_euphoria, crisis
        """
        fg = features.get('fear_greed_index', 50)
        ma_dist = features.get('btc_distance_from_ma30', 0)
        vol = features.get('realized_volatility_30d', 0)

        # Simple rule-based (replace with learned classifier when data allows)
        if vol > 0.8:  # Annualized vol > 80%
            return 'crisis'
        if fg <= 20 and ma_dist < -0.10:
            return 'fear_capitulation'
        if fg >= 80 and ma_dist > 0.10:
            return 'greed_euphoria'
        if abs(ma_dist) > 0.08:
            return 'trending_bull' if ma_dist > 0 else 'trending_bear'
        return 'ranging'
```

### 6.3 Regime-Conditional Weight Learning

Instead of applying multiplier shifts to base weights, learn weights per regime:

```python
class RegimeConditionalWeights:
    """
    Maintains separate weight estimates per regime with Bayesian shrinkage.
    """

    def __init__(self, n_signals, regimes, prior_strength=10.0):
        self.global_weights = np.ones(n_signals) / n_signals
        self.regime_weights = {r: np.ones(n_signals) / n_signals for r in regimes}
        self.regime_counts = {r: 0 for r in regimes}
        self.prior_strength = prior_strength  # pseudo-observations for prior

    def get_weights(self, regime: str) -> np.ndarray:
        """Return weights with Bayesian shrinkage toward global."""
        n = self.regime_counts.get(regime, 0)
        if n == 0:
            return self.global_weights.copy()

        # Shrinkage factor: how much to trust regime-specific vs global
        shrinkage = self.prior_strength / (self.prior_strength + n)

        return (shrinkage * self.global_weights +
                (1 - shrinkage) * self.regime_weights[regime])

    def update(self, regime: str, signals: np.ndarray, realized_return: float):
        """Update weights based on new observation."""
        self.regime_counts[regime] = self.regime_counts.get(regime, 0) + 1

        # Compute IC contribution of each signal
        # (simplified; full version uses rolling window)
        signal_returns = signals * np.sign(realized_return)

        # Update regime-specific weights toward IC-proportional
        lr = 1.0 / (self.regime_counts[regime] + self.prior_strength)
        self.regime_weights[regime] += lr * (signal_returns - self.regime_weights[regime])

        # Normalize
        self.regime_weights[regime] = np.maximum(self.regime_weights[regime], 0.01)
        self.regime_weights[regime] /= self.regime_weights[regime].sum()
```

### 6.4 Handling Regime Transitions

The most dangerous period is during regime transitions (e.g., ranging -> trending). Our current system handles this poorly: abrupt weight changes when the regime label flips.

**Solutions:**

1. **Soft regime detection:** Instead of hard regime labels, output probabilities per regime and blend weights:

```python
# P(trending) = 0.6, P(ranging) = 0.4
# weights = 0.6 * trending_weights + 0.4 * ranging_weights
```

2. **Transition smoothing:** EMA-blend regime probabilities over time to prevent whipsawing:

```python
regime_prob_smoothed = alpha * regime_prob_current + (1 - alpha) * regime_prob_previous
```

3. **Regime transition signals:** Use the transition itself as a signal (e.g., "transitioning from ranging to trending" might favor momentum).

---

## 7. When Simple Beats Complex

### 7.1 The Bias-Variance Tradeoff in Signal Combination

**Key result (DeMiguel et al. 2009):** With N assets and T time periods, the optimal (mean-variance) portfolio requires estimating N(N+1)/2 parameters. When T/N < some threshold (roughly 100-500), the estimation error dominates, and 1/N (equal weighting) outperforms.

**Applied to our signal combination:**
- We have 5-6 signals (dimensions)
- We have < 100 evaluated signal-return pairs
- Our cascade has > 30 tunable parameters (weights, dampening factors, thresholds, etc.)
- **We are massively in the overfitting regime**

### 7.2 Rules of Thumb from the Literature

| Data Points per Signal | Recommended Method | Rationale |
|---|---|---|
| < 50 | Equal weighting (1/N) | Estimation error dominates |
| 50-200 | Inverse-variance weighting | Can estimate variance but not covariance |
| 200-500 | Platt-calibrated + IC weighting | Can estimate IC reliably |
| 500-2000 | Walk-forward stacking (L1-regularized) | Enough for meta-model training |
| > 2000 | Gradient boosting / neural meta-learner | Can learn non-linear interactions |

**Our system has < 100 data points.** We should be using equal weighting or inverse-variance weighting, not a 30+ parameter cascade.

### 7.3 Empirical Results from "Forecast Combination" Literature

The forecast combination literature (Timmermann 2006, Genre et al. 2013, Smith & Wallis 2009) consistently finds:

1. **Simple average is a robust benchmark.** It's hard to beat with small sample sizes.
2. **Trimmed mean (drop best and worst signal, average the rest) often outperforms all individual signals and most combination methods.**
3. **Inverse-MSE weighting marginally improves on equal weighting** but requires reliable MSE estimates.
4. **Complex combinations (regression, stacking) only outperform simple methods when:**
   - Training data > 200 observations
   - Signal quality varies significantly across dimensions
   - The combination function is stable over time

### 7.4 The "Wisdom of Crowds" Effect

Surowiecki's "Wisdom of Crowds" and its formalization by Page (2007) show that:

```
Collective Error = Average Individual Error - Diversity
```

Where "Diversity" = variance of individual predictions around the group mean.

**Implication:** The value of combining signals comes from their DIVERSITY, not from clever weighting. Our cascade reduces diversity by (a) dampening outlier signals toward 50 and (b) applying the same regime adjustments to all dimensions. This is counterproductive.

### 7.5 Practical Recommendation

**Phase 1 (immediate):** Replace the cascade with calibrated equal-weighting:

```python
# Current: 7-layer cascade with 30+ parameters
# Proposed: 3-step pipeline with ~10 parameters

# Step 1: Calibrate each signal independently
calibrated = {dim: platt_calibrate(raw_scores[dim]) for dim in dimensions}

# Step 2: Equal-weight average
composite = np.mean(list(calibrated.values()))

# Step 3: Abstain if calibrated uncertainty is high
uncertainty = np.std(list(calibrated.values()))
if uncertainty > threshold:
    signal = "INSUFFICIENT EDGE"
```

This will almost certainly outperform the current cascade on out-of-sample data, based on the principle that simple methods dominate in low-data regimes.

---

## 8. Diagnosing Our Cascade: Specific Failure Modes

### 8.1 Failure Mode 1: F&G Quadruple-Counting

Fear & Greed index is used in 4 different layers:

1. **Market dimension scoring** (`_score_market`): F&G directly contributes to the raw score via contrarian logic
2. **F&G regime weight shifts** (`fg_regime_scoring`): Shifts dimension weights based on F&G level
3. **F&G score dampening** (`fg_score_dampening`): Dampens bullish scores in fear markets
4. **Dynamic abstain threshold** (`dynamic_cfg`): Changes the abstain zone width based on F&G

**Net effect:** F&G has ~4x the influence it should have. In extreme fear (F&G=14):
- Market dimension score is very high (contrarian boost)
- F&G regime shifts DOWN-weight market dimension
- F&G dampening pulls market score toward 50
- Wider abstain zone catches more signals

The system is simultaneously BOOSTING market bullishness (via contrarian scoring) and SUPPRESSING it (via regime dampening). These partially cancel, leaving a noisy, attenuated signal.

**Fix:** F&G should influence scoring in exactly ONE way: either as a direct signal input or as a regime indicator for weight selection, but not both.

### 8.2 Failure Mode 2: Velocity Dampening After Composition

Velocity dampening (Layer 6) applies after the composite is already built. It dampens the composite's distance from 50 by a factor of 0.3-1.0. But this means:

- A composite of 70 driven by strong whale + technical signals gets dampened identically to a composite of 70 driven by a single noisy dimension
- The velocity information (RSI accelerating down) is not incorporated into the signal; it's used to suppress the signal
- If velocity is truly informative, it should be a signal dimension, not a dampener

**Fix:** Velocity should be a separate calibrated signal that gets combined with others, not a post-hoc dampener.

### 8.3 Failure Mode 3: Regime Weight Shifts Interact with Accuracy Scaling

Current order: accuracy scaling -> regime weight shifts -> F&G weight shifts -> tier reweighting.

After accuracy scaling, weights are renormalized to sum to 1.0. Then regime shifts are applied multiplicatively. Then F&G shifts are applied multiplicatively. Then tier reweighting adjusts further.

**Problem:** The renormalization after accuracy scaling means that regime shifts operate on different base weights than intended. If accuracy scaling dramatically changes the weight distribution (e.g., from 0.20/0.20/0.20/0.20/0.20 to 0.05/0.10/0.35/0.35/0.15), then the regime shifts intended for the original weight distribution produce unexpected results.

**Fix:** Apply all weight adjustments simultaneously, not sequentially. Or better: learn the final weights directly conditioned on regime.

### 8.4 Failure Mode 4: Abstain Logic is Disconnected from Signal Quality

The abstain check uses `abs(composite - 50) < threshold`. This treats a composite of 51 the same regardless of whether:
- All 5 dimensions unanimously score 51 (high consensus, low conviction)
- Dimensions split 80/80/20/20/55 (high conviction but mixed direction)

The first case is genuinely low-edge; the second is a strong disagreement signal that might warrant a different action (e.g., abstain due to uncertainty, or act on the majority).

**Fix:** Abstain based on calibrated ensemble disagreement (variance of calibrated signals) rather than distance-from-center.

### 8.5 Failure Mode 5: Trend Override as Binary Switch

The trend override (`is_downtrend`) is a binary variable that enables/disables dampening for specific dimensions. This creates a discontinuity: at the threshold boundary, a tiny price change flips dampening on/off, potentially changing the composite by 5-10 points.

**Fix:** Use continuous trend strength as a signal, not a binary switch.

---

## 9. Recommended Architecture

### 9.1 Target Architecture: Calibrate-Combine-Gate

Replace the 7-layer cascade with a 3-phase pipeline:

```
                     Phase 1                    Phase 2                Phase 3
                   CALIBRATE                   COMBINE                 GATE

Raw Whale Score ─────> Calibrated P(up|whale) ──┐
Raw Technical Score ──> Calibrated P(up|tech) ──┤
Raw Derivatives Score > Calibrated P(up|deriv) ─┼──> Weighted Avg ──> Confidence ──> Signal
Raw Narrative Score ──> Calibrated P(up|narr) ──┤    (single step)     Check        (or ABSTAIN)
Raw Market Score ─────> Calibrated P(up|mkt) ───┘

Context Features ─────────────────────────────────> Weight Selection
(regime, volatility,                                (regime-conditional
 F&G, velocity)                                      lookup table)
```

### 9.2 Phase 1: Signal Calibration (Replaces Layers 2, 3, 4)

Each dimension's raw 0-100 score is independently calibrated to a probability P(positive_return | raw_score):

```python
class DimensionCalibrator:
    """
    Calibrates one dimension's raw score to probability.
    Replaces: accuracy_scaling + contrarian_inversion + regime_dampening
    """

    def __init__(self, dimension_name: str):
        self.name = dimension_name
        self.calibrator = None
        self.regime_calibrators = {}

    def fit(self, historical_scores, historical_returns, regimes=None):
        """
        Fit calibrator from historical data.

        Args:
            historical_scores: Raw scores from this dimension
            historical_returns: Actual forward returns
            regimes: Optional regime labels for regime-conditional calibration
        """
        binary_outcomes = (np.array(historical_returns) > 0).astype(float)
        scores = np.array(historical_scores)

        n = len(scores)

        if n < 30:
            # Insufficient data: use identity (no calibration)
            self.calibrator = IdentityCalibrator()
        elif n < 100:
            # Limited data: Platt scaling (2 params)
            self.calibrator = PlattCalibrator()
            self.calibrator.fit(scores, binary_outcomes)
        else:
            # Adequate data: isotonic regression
            self.calibrator = IsotonicCalibrator()
            self.calibrator.fit(scores, binary_outcomes)

        # Regime-conditional calibrators (if enough data per regime)
        if regimes is not None:
            for regime in set(regimes):
                mask = np.array(regimes) == regime
                regime_scores = scores[mask]
                regime_outcomes = binary_outcomes[mask]
                if len(regime_scores) >= 50:
                    rc = PlattCalibrator()
                    rc.fit(regime_scores, regime_outcomes)
                    self.regime_calibrators[regime] = rc

    def calibrate(self, raw_score, regime=None):
        """Return calibrated P(up) for this dimension."""
        if regime and regime in self.regime_calibrators:
            return self.regime_calibrators[regime].predict(raw_score)
        return self.calibrator.predict(raw_score)
```

**Key insight:** The contrarian inversion (low F&G = bullish) should be LEARNED by the calibrator, not hard-coded. If the market dimension correctly predicts returns when F&G is low (i.e., low F&G -> high score -> positive return), the calibrator will learn a positive calibration curve. If the contrarian logic is wrong, the calibrator will learn a flat or negative curve, effectively zero-weighting that dimension.

### 9.3 Phase 2: Single-Step Combination (Replaces Layers 1, 3, 4, 5)

```python
class SingleStepCombiner:
    """
    Combines calibrated signals in one operation.
    Replaces: asymmetric_weights + regime_shifts + fg_shifts + tier_reweight
    """

    def __init__(self, n_signals: int, regimes: list):
        self.weight_table = {}  # {regime: weights}
        self.default_weights = np.ones(n_signals) / n_signals
        self.bayesian_shrinkage = BayesianShrinkage(n_signals, regimes)

    def get_weights(self, regime: str, data_tiers: dict) -> np.ndarray:
        """
        Get combination weights for current regime and data quality.
        Handles tier adjustment in a single step.
        """
        base_weights = self.bayesian_shrinkage.get_weights(regime)

        # Adjust for data quality (tier)
        tier_factors = {'full': 1.0, 'partial': 0.5, 'none': 0.0}
        adjusted = base_weights.copy()
        for i, (dim, tier) in enumerate(data_tiers.items()):
            adjusted[i] *= tier_factors.get(tier, 1.0)

        # Renormalize
        total = adjusted.sum()
        if total > 0:
            adjusted /= total
        else:
            adjusted = self.default_weights.copy()

        return adjusted

    def combine(self, calibrated_signals: dict, regime: str, data_tiers: dict) -> float:
        """Single-step signal combination."""
        signals = np.array(list(calibrated_signals.values()))
        weights = self.get_weights(regime, data_tiers)
        return float(np.dot(signals, weights))
```

### 9.4 Phase 3: Confidence Gating (Replaces Layers 6, 7)

```python
class ConfidenceGate:
    """
    Decides whether to signal or abstain based on ensemble properties.
    Replaces: velocity_dampening + abstain_check
    """

    def __init__(self, min_confidence: float = 0.55):
        self.min_confidence = min_confidence

    def evaluate(self,
                 calibrated_signals: dict,
                 combined_probability: float,
                 velocity_signal: float = None) -> dict:
        """
        Evaluate whether to emit a signal.

        Returns:
            {
                'emit': bool,
                'probability': float,
                'confidence': float,
                'reason': str
            }
        """
        probs = np.array(list(calibrated_signals.values()))

        # Confidence metric 1: How far from 50/50?
        edge = abs(combined_probability - 0.5)

        # Confidence metric 2: Signal agreement (low std = high agreement)
        agreement = 1.0 - np.std(probs)  # 0 to ~0.5 range

        # Confidence metric 3: Velocity confirmation (if available)
        velocity_bonus = 0.0
        if velocity_signal is not None:
            # Positive velocity_signal means velocity confirms direction
            velocity_bonus = max(0, velocity_signal) * 0.1

        # Combined confidence
        confidence = 0.5 * edge + 0.3 * agreement + 0.2 * velocity_bonus

        # Abstain decision
        emit = combined_probability > self.min_confidence or \
               combined_probability < (1 - self.min_confidence)

        if not emit:
            reason = f"Edge too small: P(up)={combined_probability:.2f}, agreement={agreement:.2f}"
        else:
            direction = "BUY" if combined_probability > 0.5 else "SELL"
            reason = f"{direction}: P(up)={combined_probability:.2f}, confidence={confidence:.2f}"

        return {
            'emit': emit,
            'probability': combined_probability,
            'confidence': confidence,
            'reason': reason,
            'signal_std': float(np.std(probs)),
        }
```

### 9.5 The Complete Replacement Pipeline

```python
class CalibratedSignalFusion:
    """
    Replaces the 7-layer cascade with calibrate-combine-gate.
    """

    def __init__(self, profile, store):
        self.dimensions = ['whale', 'technical', 'derivatives', 'narrative', 'market']

        # Phase 1: One calibrator per dimension
        self.calibrators = {dim: DimensionCalibrator(dim) for dim in self.dimensions}

        # Phase 2: Single-step combiner
        self.combiner = SingleStepCombiner(
            n_signals=len(self.dimensions),
            regimes=['trending_bull', 'trending_bear', 'ranging',
                     'fear_capitulation', 'greed_euphoria', 'crisis']
        )

        # Phase 3: Confidence gate
        self.gate = ConfidenceGate(min_confidence=0.55)

        # Regime detector (replaces 3 separate regime systems)
        self.regime_detector = UnifiedRegimeDetector()

    def fuse(self, raw_scores: dict, context: dict) -> dict:
        """
        Main fusion method.

        Args:
            raw_scores: {dimension: raw_0_100_score}
            context: {fear_greed, btc_ma30_distance, volatility, ...}

        Returns:
            Signal dict with calibrated probability, confidence, and direction
        """
        # Detect regime (single unified detection)
        regime = self.regime_detector.detect_regime(context)

        # Phase 1: Calibrate each dimension independently
        calibrated = {}
        for dim in self.dimensions:
            calibrated[dim] = self.calibrators[dim].calibrate(
                raw_scores[dim], regime=regime
            )

        # Phase 2: Combine in single step
        data_tiers = context.get('data_tiers', {dim: 'full' for dim in self.dimensions})
        combined = self.combiner.combine(calibrated, regime, data_tiers)

        # Phase 3: Gate
        velocity_signal = context.get('velocity_signal', None)
        result = self.gate.evaluate(calibrated, combined, velocity_signal)

        result['regime'] = regime
        result['calibrated_signals'] = calibrated
        result['weights_used'] = dict(zip(
            self.dimensions,
            self.combiner.get_weights(regime, data_tiers).tolist()
        ))

        return result
```

---

## 10. Implementation Roadmap

### Phase 1: Quick Win (1-2 days) — Flatten the Cascade

**Goal:** Remove the most harmful cascade interactions without changing the architecture.

**Steps:**
1. Disable F&G regime scoring (`fg_regime_scoring.enabled: false`)
2. Disable velocity dampening (`velocity.enabled: false`)
3. Disable trend override (`trend_override.enabled: false`)
4. Keep only: base weights + data tier reweighting + simple abstain

**Expected impact:** Removes ~4 layers of cascading adjustments. The remaining system is equivalent to a weighted average with data quality adjustment, which is close to the IC-weighted combination used by quant funds.

**Risk:** Lower. We're removing complexity, not adding it.

### Phase 2: Calibration (3-5 days) — Add Platt Scaling

**Goal:** Make signal scores interpretable and comparable across dimensions.

**Steps:**
1. Collect historical (raw_score, actual_return) pairs for each dimension
   - Need minimum 50 pairs; ideally 100+
   - Use data from the existing SQLite storage
2. Fit Platt calibrators for each dimension
3. Modify `_score_dimension` to output both raw score and calibrated probability
4. Replace weighted-average-of-raw-scores with weighted-average-of-calibrated-probabilities
5. Validate calibration quality (reliability diagrams, Brier score)

**Expected impact:** Significant. Calibrated probabilities are the foundation for all subsequent improvements.

### Phase 3: Unified Regime (2-3 days) — Single Regime System

**Goal:** Replace the 3 separate regime systems with one unified detector.

**Steps:**
1. Implement `UnifiedRegimeDetector` that classifies into 4-6 regimes
2. Learn regime-conditional weight tables from historical data (with Bayesian shrinkage)
3. Use soft regime detection (probability outputs) with transition smoothing
4. Remove: `regime_weighting`, `fg_regime_scoring`, `trend_override` sections from YAML

**Expected impact:** Moderate. Eliminates contradictory regime adjustments.

### Phase 4: Confidence-Based Gating (1-2 days) — Better Abstain

**Goal:** Replace distance-from-center abstain with calibrated confidence.

**Steps:**
1. Implement `ConfidenceGate` using signal agreement + edge magnitude
2. Use calibrated probability intervals (from Platt scaling) to quantify uncertainty
3. Remove: `abstain.dynamic` and F&G-based abstain threshold logic

**Expected impact:** Better abstain decisions. Currently we abstain on "close to 50" which misses high-disagreement situations and wrongly abstains on low-disagreement-near-50 situations.

### Phase 5: Meta-Learning (1-2 weeks) — Optional, Data-Dependent

**Goal:** Learn optimal signal combination weights from data.

**Steps:**
1. Implement walk-forward IC computation per regime
2. Implement Bayesian shrinkage weight learner
3. Implement online EXP3 weight adjustment as supplement
4. Add regime transition detection and smoothing
5. Backtest against equal-weighting and current system

**Expected impact:** Uncertain. Depends entirely on data volume. Do NOT implement until Phase 2 calibration is validated.

### Phase 6: Advanced Ensemble (2-4 weeks) — Future

**Goal:** Explore non-linear signal combination once sufficient data exists.

**Steps:**
1. Walk-forward stacking with L1-regularized meta-learner (when > 500 data points)
2. Gradient boosting for signal combination (when > 1000 data points)
3. Neural meta-learner with context conditioning (when > 2000 data points)
4. Continuous A/B testing of combination methods

**Expected impact:** Potentially significant, but requires patience (data accumulation).

---

## 11. Key References

### Academic Papers

1. **Grinold & Kahn (2000).** *Active Portfolio Management.* The foundational text on alpha signal combination. Introduces the Fundamental Law of Active Management showing IC * sqrt(breadth) determines portfolio information ratio.

2. **DeMiguel, Garlappi & Uppal (2009).** "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *Journal of Finance.* Shows equal weighting outperforms optimized weighting in realistic settings with estimation error. Directly relevant to our low-data regime.

3. **Timmermann (2006).** "Forecast Combinations." *Handbook of Economic Forecasting.* Comprehensive survey showing simple forecast combination methods are robust and often optimal.

4. **Platt (1999).** "Probabilistic Outputs for Support Vector Machines." Introduces Platt scaling for probability calibration. Simple, effective, and widely used in practice.

5. **Zadrozny & Elkan (2002).** "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." Extends calibration to multi-class settings; isotonic regression approach.

6. **Vovk, Gammerman & Shafer (2005).** *Algorithmic Learning in a Random World.* Foundation for conformal prediction and Venn-ABERS calibration with distribution-free guarantees.

7. **Genre, Kenny, Meyler & Timmermann (2013).** "Combining Expert Forecasts: Can Anything Beat the Simple Average?" *International Journal of Forecasting.* Empirical study confirming simple average as robust benchmark.

8. **Smith & Wallis (2009).** "A Simple Explanation of the Forecast Combination Puzzle." *Oxford Bulletin of Economics and Statistics.* Explains why simple averages are hard to beat: estimation error in combination weights often exceeds the gain from optimization.

9. **Clemen (1989).** "Combining Forecasts: A Review and Annotated Bibliography." *International Journal of Forecasting.* The canonical review establishing that combined forecasts outperform individual forecasts.

10. **Raftery, Gneiting, Balabdaoui & Polakowski (2005).** "Using Bayesian Model Averaging to Calibrate Forecast Ensembles." *Monthly Weather Review.* BMA for forecast calibration and combination; applicable to trading signals.

### Quant Finance Specific

11. **Kakushadze (2016).** "101 Formulaic Alphas." *Wilmott Magazine.* Documents signal construction at WorldQuant; shows the emphasis on orthogonalization before combination.

12. **Harvey, Liu & Zhu (2016).** "...and the Cross-Section of Expected Returns." *Review of Financial Studies.* Shows that most "alpha" signals are false discoveries; emphasizes the need for rigorous calibration and multiple testing correction.

13. **Gu, Kelly & Xiu (2020).** "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies.* Comprehensive comparison of ML methods for return prediction. Finds that simple methods (penalized linear, including ridge and elastic net) often match complex methods (neural networks, random forests) for signal combination.

14. **Lopez de Prado (2018).** *Advances in Financial Machine Learning.* Chapters on meta-labeling, walk-forward stacking, and combinatorial purged cross-validation for financial ML.

15. **Chincarini & Kim (2006).** *Quantitative Equity Portfolio Management.* Chapter 6 covers alpha signal combination in production settings; advocates for IC-weighted linear combination.

### Open Source Implementations

16. **mlfinlab (Hudson & Thames).** Python library implementing Lopez de Prado's methods including meta-labeling and walk-forward validation. GitHub: `hudson-and-thames/mlfinlab`

17. **skfolio.** Scikit-learn compatible portfolio optimization library with ensemble methods for combining signals. GitHub: `skfolio/skfolio`

18. **Qlib (Microsoft).** ML-based quant investment platform with signal combination pipeline. GitHub: `microsoft/qlib`

19. **Zipline (Quantopian legacy).** Event-driven backtesting with pipeline for signal combination. GitHub: `stefan-jansen/zipline-reloaded`

20. **LOBSTER / FinRL.** Reinforcement learning frameworks for trading that include meta-learning components. GitHub: `AI4Finance-Foundation/FinRL`

---

## Appendix A: Why Our Specific Cascade Fails — Worked Example

### Setup

Consider BTC when Fear & Greed = 18 (extreme fear), RSI = 32 (near oversold), whale accumulation detected, price 8% below MA30.

### Current Cascade (7 layers)

```
Layer 1 (Asymmetric weights): raw_avg < 50 → bearish weights selected
  whale: 0.20, technical: 0.25, derivatives: 0.25, narrative: 0.15, market: 0.15

Layer 2 (Accuracy scaling): bearish direction multipliers applied
  whale *= 0.65 → 0.13, technical *= 0.72 → 0.18, derivatives *= 0.55 → 0.14
  narrative *= 0.40 → 0.06, market *= 0.80 → 0.12
  Renormalized: whale=0.21, technical=0.29, derivatives=0.22, narrative=0.10, market=0.19

Layer 3 (Regime weighting): 8% below MA30 → trending_bear detected
  technical *= 1.3, whale *= 0.8
  Renormalized: whale=0.17, technical=0.36, derivatives=0.21, narrative=0.09, market=0.18

Layer 4 (F&G regime scoring): extreme_fear detected
  weight_shifts: whale *= 1.2, derivatives *= 1.3, market *= 0.6
  Renormalized: whale=0.19, technical=0.34, derivatives=0.26, narrative=0.08, market=0.10
  Also: score_dampening factor 0.5 applied to market + narrative dims above 50

Layer 5 (Data tier): whale has partial data → tier multiplier 0.5
  whale: 0.19 * 0.5 = 0.10, redistribute freed 0.10 to others
  Final: whale=0.10, technical=0.38, derivatives=0.29, narrative=0.09, market=0.11

Raw dimension scores:
  whale: 65 (accumulation detected)
  technical: 72 (RSI oversold = bullish contrarian)
  derivatives: 45 (neutral-bearish)
  narrative: 55 (mildly bullish)
  market: 78 (extreme fear = very bullish contrarian)

But Layer 4 dampened market and narrative:
  market: 50 + (78-50)*0.5 = 64
  narrative: 50 + (55-50)*0.5 = 52.5

Composite = 0.10*65 + 0.38*72 + 0.29*45 + 0.09*52.5 + 0.11*64
          = 6.5 + 27.4 + 13.1 + 4.7 + 7.0
          = 58.7

Layer 6 (Velocity dampening): RSI falling (50→42→35), dampening_factor = 0.45
  Composite = 50 + (58.7 - 50) * 0.45 = 53.9

Layer 7 (Abstain check): |53.9 - 50| = 3.9 < dynamic_threshold(extreme_fear) = 5
  Result: INSUFFICIENT EDGE → ABSTAIN
```

### What Should Have Happened

The whale accumulation was genuine. The technical oversold was a real contrarian signal. The extreme fear was correctly identified. **This was a buy signal that the cascade destroyed.**

### Proposed System (Calibrate-Combine-Gate)

```
Phase 1 (Calibrate): Each signal maps to P(positive_24h_return)
  whale: 65 → P(up) = 0.62 (Platt-calibrated)
  technical: 72 → P(up) = 0.65
  derivatives: 45 → P(up) = 0.42
  narrative: 55 → P(up) = 0.52
  market: 78 → P(up) = 0.58

Phase 2 (Combine): Regime = fear_capitulation, equal weights (low data)
  P(up) = mean([0.62, 0.65, 0.42, 0.52, 0.58]) = 0.558

Phase 3 (Gate): P(up) = 0.558, signal_std = 0.085
  Edge: |0.558 - 0.5| = 0.058 > min_edge(0.05) → EMIT
  Direction: BUY with 56% probability
  Confidence: moderate (mixed signals)

  Result: MODERATE BUY with caveats about derivatives divergence
```

The proposed system preserves the buy signal while correctly noting mixed confidence. The cascade system abstained on a real opportunity.

---

## Appendix B: Quick Reference — Ensemble Methods Comparison

| Method | Parameters | Min Data | Handles Regimes | Handles Non-Linear | Interpretable | Recommended Phase |
|--------|-----------|----------|-----------------|-------------------|---------------|------------------|
| Equal Weight (1/N) | 0 | 0 | No | No | Yes | Phase 1 (immediate) |
| Inverse-Variance | N | 30/signal | No | No | Yes | Phase 1 |
| IC-Weighted | N | 100/signal | No | No | Yes | Phase 2 |
| Platt + Equal | 2N | 50/signal | Via regime-specific calibration | Partially (sigmoid) | Yes | Phase 2 |
| Regime-Conditional IC | N * R | 50/regime | Yes | No | Yes | Phase 3 |
| BMA | 2N | 100/signal | Via hierarchical | No | Yes | Phase 3 |
| Walk-Forward Stacking | N^2 | 500 | Implicitly | With interactions | Moderate | Phase 5 |
| Gradient Boosting | 100+ | 1000 | Yes | Yes | Low | Phase 6 |
| Neural Meta-Learner | 1000+ | 2000+ | Yes | Yes | No | Phase 6 |

**Where we are:** < 100 data points. **Recommended:** Equal weight or Platt + Equal.
**Where we need to be for stacking:** 500+ data points. **Timeline:** ~6 months of data accumulation.

---

## Appendix C: Implementation Checklist

### Immediate (This Sprint)

- [ ] Disable `fg_regime_scoring` (YAML: `fg_regime_scoring.enabled: false`)
- [ ] Disable `velocity` dampening (YAML: `velocity.enabled: false`)
- [ ] Disable `trend_override` (YAML: `trend_override.enabled: false`)
- [ ] Measure baseline accuracy with simplified 2-layer system (weights + tiers only)
- [ ] Begin collecting (raw_score, 24h_return) pairs per dimension for calibration

### Short-term (2-4 Weeks)

- [ ] Implement Platt calibrator class in `signal_fusion/calibration.py`
- [ ] Fit initial calibrators on available historical data
- [ ] Implement `UnifiedRegimeDetector`
- [ ] Implement `ConfidenceGate` based on calibrated signal agreement
- [ ] A/B test calibrated system vs current cascade

### Medium-term (1-3 Months)

- [ ] Accumulate 200+ (score, return) pairs per dimension
- [ ] Implement regime-conditional Platt scaling
- [ ] Implement Bayesian shrinkage weight learner
- [ ] Walk-forward backtest of all combination methods

### Long-term (3-6 Months)

- [ ] Reach 500+ data points for walk-forward stacking feasibility
- [ ] Implement L1-regularized stacking meta-learner
- [ ] Explore gradient boosting for signal combination
- [ ] Continuous calibration monitoring (auto-recalibrate when calibration degrades)

---

*End of E5 Research Report*
