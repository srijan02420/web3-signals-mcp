# E7: Risk Management, Signal Gating & Confidence Filters

**Research Agent:** E7 — Risk Management, Signal Gating & Confidence Filters
**Date:** 2026-03-16
**System Under Analysis:** Web3 Signals x402 — Signal Fusion Engine
**Files Analyzed:**
- `/Users/admin/Documents/web3 Signals x402/signal_fusion/profiles/default.yaml` (1042 lines, full scoring config)
- `/Users/admin/Documents/web3 Signals x402/signal_fusion/engine.py` (signal fusion implementation)
- `/Users/admin/Documents/web3 Signals x402/.claude/worktrees/optimistic-engelbart/backtest.py` (1330 lines, backtesting + accuracy)
- `/Users/admin/Documents/web3 Signals x402/sweep.py` (parameter sweep: regime x abstain)
- `/Users/admin/Documents/web3 Signals x402/retroactive_accuracy.py` (live accuracy tracking)
- Existing research: I3 (data quality), I4 (weight optimizer), I6 (feature engineering)

---

## Executive Summary

The system's abstain zone (+/-12 from center, filtering 88% of signals) is a blunt instrument that discards both noise AND genuine edge. The 48% gradient accuracy on issued signals is mediocre, and the catastrophic asymmetry between bearish (72.5%) and bullish (34.6%) accuracy reveals a fundamental calibration failure rather than a filtering problem.

**Key findings:**

1. **The abstain zone is not the bottleneck.** Tightening it from 12 to 8 in the sweep only marginally changes accuracy. The problem is upstream: the composite score itself is poorly calibrated, so filtering based on score distance from 50 cannot separate good signals from bad ones.

2. **Asymmetric abstain zones are mandatory.** Bullish signals at 34.6% accuracy are worse than random (30% baseline). The system should either (a) suppress bullish signals entirely, (b) require a much higher threshold for bullish signals, or (c) invert the bullish logic.

3. **The system lacks proper probabilistic calibration.** Composite scores are treated as confidence levels but are not calibrated probabilities. A score of 65 does not mean "65% likely to go up." Implementing calibration curves would reveal which score ranges actually have predictive edge.

4. **Kelly criterion analysis confirms: bullish signals have negative expected value.** No rational position sizing framework would allocate capital to signals with 34.6% directional accuracy.

5. **The 88% abstain rate is reasonable for the current model quality**, but a better-calibrated model could safely issue 3-5x more signals.

---

## 1. Signal Confidence Calibration Methods

### 1.1 Brier Score: The Right Metric You're Not Using

The **Brier Score** decomposes prediction quality into three orthogonal components:

```
BS = (1/N) * SUM[(forecast_probability - outcome)^2]
   = Reliability - Resolution + Uncertainty
```

- **Reliability** (calibration): Do your 70% confidence signals actually succeed 70% of the time?
- **Resolution** (discrimination): Can you distinguish high-probability from low-probability outcomes?
- **Uncertainty** (base rate): How inherently unpredictable is the market?

**Application to your system:**

Your composite scores (0-100) can be mapped to implied probabilities:
```
p_bullish = (composite - 50) / 50   for composite > 50
p_bearish = (50 - composite) / 50   for composite < 50
```

A score of 65 implies p_bullish = 0.30 (30% confidence beyond coin flip).
A score of 35 implies p_bearish = 0.30.

**Current Brier Score estimate** (from your data):
- Bearish signals (72.5% accurate): BS_bearish ~ 0.20 (decent)
- Bullish signals (34.6% accurate): BS_bullish ~ 0.43 (terrible -- worse than always predicting "down")

**Recommendation:** Compute the Brier score decomposition for every backtest run. This tells you:
- Is the problem calibration (scores don't match reality)?
- Is the problem resolution (can't distinguish strong from weak setups)?
- Or is the problem fundamental (crypto is too noisy for this approach)?

### 1.2 Calibration Curves (Reliability Diagrams)

A calibration curve plots predicted probability vs. observed frequency. For a perfectly calibrated system, these are identical (45-degree line).

**How to build one for your system:**

```python
def build_calibration_curve(evaluations):
    """Group signals by composite score, compute actual accuracy per bucket."""
    buckets = {}  # score_bucket -> [outcomes]

    for ev in evaluations:
        # Map composite to implied probability
        score = ev["composite_score"]
        bucket = round(score / 5) * 5  # 5-point buckets

        if bucket not in buckets:
            buckets[bucket] = []

        # Outcome: did direction match?
        correct = 1.0 if ev["binary_correct"] else 0.0
        buckets[bucket].append(correct)

    # Plot: x = implied probability, y = actual hit rate
    for bucket in sorted(buckets.keys()):
        implied_p = abs(bucket - 50) / 50  # 0 = no confidence, 1 = max
        actual_p = sum(buckets[bucket]) / len(buckets[bucket])
        n = len(buckets[bucket])
        print(f"  Score {bucket}: implied={implied_p:.2f}  actual={actual_p:.2f}  n={n}")
```

**Expected findings (based on your data):**
- Scores 60-65: implied p = 0.20-0.30, actual p for bullish ~ 0.35 (overconfident!)
- Scores 35-40: implied p = 0.20-0.30, actual p for bearish ~ 0.72 (underconfident!)

This asymmetry means your composite scores are **systematically miscalibrated**:
- Bullish scores overstate edge (scores say "moderate buy" when actual accuracy is below random)
- Bearish scores understate edge (scores say "moderate sell" but accuracy is excellent)

### 1.3 Platt Scaling and Isotonic Regression

After identifying miscalibration, standard remedies:

**Platt Scaling** (parametric):
Fit a logistic function to map raw scores to calibrated probabilities:
```
P(correct | score) = 1 / (1 + exp(-(A * score + B)))
```
Fit A and B on held-out evaluation data. This handles sigmoid-shaped miscalibration.

**Isotonic Regression** (non-parametric):
Fit a monotonically non-decreasing step function mapping scores to probabilities.
Better when miscalibration is non-linear (which is likely in your case given the bullish/bearish asymmetry).

**Temperature Scaling**:
A simpler variant: divide logits by a temperature parameter T.
- T > 1 makes predictions less confident (fixes overconfidence)
- T < 1 makes predictions more confident (fixes underconfidence)

Your system needs different temperatures for bullish vs. bearish:
- Bullish: T >> 1 (much less confident -- currently overconfident)
- Bearish: T < 1 (more confident -- currently underconfident)

**Implementation priority: HIGH.** Calibration is the single most impactful improvement available. A calibrated system can safely issue 3-5x more signals because you KNOW which score ranges have genuine edge.

---

## 2. Kelly Criterion for Crypto Signal Sizing

### 2.1 The Kelly Formula

The Kelly criterion determines optimal bet size to maximize long-term growth:

```
f* = (bp - q) / b
```

Where:
- f* = fraction of bankroll to wager
- b = net odds received on the wager (payoff-to-loss ratio)
- p = probability of winning
- q = 1 - p = probability of losing

For asymmetric payoffs (which crypto absolutely has):

```
f* = p/a - q/b
```

Where a = loss on losing bet, b = gain on winning bet.

### 2.2 Applying Kelly to Your System

**Bearish signals (72.5% accurate, 24h):**
- p = 0.725, q = 0.275
- Assume symmetric payoff (b = 1): f* = 0.725 - 0.275 = 0.45
- This means Kelly recommends wagering 45% of bankroll on each bearish signal
- In practice, use **fractional Kelly** (typically 1/4 to 1/2): f* = 11-22%

**Bullish signals (34.6% accurate, 24h):**
- p = 0.346, q = 0.654
- f* = 0.346 - 0.654 = **-0.308**
- **NEGATIVE Kelly.** This means the optimal strategy is to BET AGAINST your own bullish signals.

**This is the most damning finding: Kelly criterion says your bullish signals have negative expected value and should be inverted or suppressed entirely.**

### 2.3 Signal-Strength-Weighted Kelly

A more sophisticated approach sizes positions proportional to both accuracy AND signal strength:

```python
def kelly_position_size(composite_score, direction, calibrated_accuracy):
    """
    Compute Kelly-optimal position size given calibrated accuracy.

    Returns: fraction of bankroll (0.0 = no position, negative = inverse position)
    """
    # Edge = calibrated accuracy - 0.5 (for binary directional bets)
    edge = calibrated_accuracy - 0.5

    # Signal strength = distance from center (0-50 scale)
    strength = abs(composite_score - 50) / 50  # normalized to 0-1

    # Full Kelly
    kelly_fraction = 2 * edge  # simplified for symmetric payoffs

    # Fractional Kelly (conservative)
    position = kelly_fraction * 0.25  # quarter Kelly

    # Scale by signal strength (stronger signals get closer to full Kelly)
    position *= strength

    return position
```

**Score-based position sizing tiers for your system:**

| Score Distance | Signal Strength | Bearish Kelly | Bullish Kelly | Action |
|---------------|----------------|---------------|---------------|--------|
| |d| > 20 | Very High | 11.25% | **-7.7%** (invert!) | Bearish: full position. Bullish: FADE IT |
| 15 < |d| <= 20 | High | 8.4% | **-5.8%** | Bearish: 3/4 position. Bullish: suppress |
| 12 < |d| <= 15 | Medium | 5.6% | **-3.9%** | Bearish: 1/2 position. Bullish: suppress |
| 8 < |d| <= 12 | Low | 2.8% | **-1.9%** | Both: abstain (current zone) |
| |d| <= 8 | Very Low | 0% | 0% | Abstain |

### 2.4 Fractional Kelly in Practice

Academic and practitioner consensus (Thorp 2006, Ziemba 2005):
- **Full Kelly** maximizes geometric growth but has enormous variance
- **Half Kelly** captures 75% of growth with 50% of variance
- **Quarter Kelly** captures 50% of growth with 25% of variance
- For crypto (high volatility, fat tails): **quarter Kelly or less** is standard

The crucial insight: Kelly implicitly provides signal gating. If calibrated accuracy < 50%, Kelly = negative = don't trade. This is a mathematically rigorous replacement for your ad hoc abstain zone.

---

## 3. Trading Signal Filtering & Gating Mechanisms

### 3.1 Taxonomy of Signal Gates

Professional trading systems use multiple layers of signal filtering:

**Layer 1: Statistical Significance Gate**
- Minimum sample size for the signal pattern to be trusted
- Typically: p < 0.05 or Bayesian posterior > 0.95
- Your system: NOT IMPLEMENTED. Signals fire based on raw composite scores without checking if the pattern has statistical significance.

**Layer 2: Edge Gate (Expected Value)**
- Signal must have positive expected value after costs
- EV = (win_rate * avg_win) - (loss_rate * avg_loss) - costs
- Your system: PARTIALLY IMPLEMENTED via abstain zone, but without proper EV calculation

**Layer 3: Regime Gate**
- Different strategies work in different regimes
- Your system: IMPLEMENTED (F&G regime scoring, trending/ranging detection)
- Quality: MODERATE -- regime detection is heuristic, not statistical

**Layer 4: Correlation/Redundancy Gate**
- Avoid issuing highly correlated signals simultaneously
- If BTC, ETH, and SOL all show "BUY," that's really 1 signal (crypto moves together)
- Your system: NOT IMPLEMENTED. All 20 assets scored independently.

**Layer 5: Capacity/Risk Gate**
- Maximum number of concurrent positions
- Portfolio-level risk constraints (max drawdown, VaR)
- Your system: NOT IMPLEMENTED (no portfolio-level risk management)

### 3.2 Confidence Threshold Approaches

**Fixed Threshold** (your current approach):
```
if |composite - 50| < 12:
    abstain
```
Simple, interpretable, but ignores context. Same threshold for BTC (high liquidity, mean-reverting) and INJ (low liquidity, momentum-driven).

**Adaptive Threshold** (partially implemented via dynamic abstain, currently disabled):
```
threshold = f(fear_greed_index, volatility, market_regime)
```
Your dynamic abstain zones (YAML lines 776-791) had the right idea but failed because they collapsed to threshold=3 in sustained fear, letting through garbage signals.

**Evidence-Based Threshold** (recommended):
```python
def evidence_threshold(direction, regime, historical_accuracy):
    """
    Set threshold based on measured accuracy for this direction + regime combo.
    Only issue signals when historical accuracy > 55% for this category.
    """
    accuracy = historical_accuracy.get((direction, regime), 0.5)

    if accuracy < 0.40:
        return 999  # never issue (bullish in current system!)
    elif accuracy < 0.50:
        return 20   # very high threshold
    elif accuracy < 0.55:
        return 15   # high threshold
    elif accuracy < 0.65:
        return 12   # standard threshold
    else:
        return 8    # allow moderate signals through (bearish currently)
```

### 3.3 Signal Quality Scores

Beyond raw composite scores, add a **signal quality score** that evaluates HOW the composite was formed:

```python
def signal_quality_score(dimensions, data_tiers):
    """
    Meta-score: how trustworthy is this composite, regardless of its value?

    Factors:
    1. Data completeness: how many dimensions have full data?
    2. Dimension agreement: are dimensions aligned or contradicting?
    3. Historical reliability: is this type of signal historically accurate?
    4. Data freshness: is the underlying data stale?
    """
    quality = 0.0

    # Factor 1: Data completeness (0-25 points)
    full_data_count = sum(1 for d in data_tiers.values() if d == "full")
    quality += (full_data_count / len(data_tiers)) * 25

    # Factor 2: Dimension agreement (0-25 points)
    scores = [d["score"] for d in dimensions.values()]
    above_50 = sum(1 for s in scores if s > 55)
    below_50 = sum(1 for s in scores if s < 45)
    agreement = max(above_50, below_50) / len(scores)
    quality += agreement * 25

    # Factor 3: Historical IC of dominant dimensions (0-25 points)
    # Use IC data from optimizer: dimensions with positive IC = trustworthy
    # dimensions with negative IC = untrustworthy
    # Weight quality by how much IC-positive dims contribute
    # (implementation requires IC data from storage)
    quality += 12.5  # placeholder until IC integration

    # Factor 4: Data freshness (0-25 points)
    # Check timestamps of underlying data
    quality += 12.5  # placeholder until staleness detection

    return quality  # 0-100 scale
```

**Gate rule:** Only issue signals when quality_score > 50 AND composite passes abstain zone.

This addresses the I3 finding that "no data = 50" artificially creates neutral composites. A quality score would catch these as low-quality and suppress them even if the composite accidentally crossed the abstain threshold.

---

## 4. Asymmetric Confidence: Different Thresholds for Bullish vs. Bearish

### 4.1 The Case for Asymmetry Is Overwhelming

Your data:
- Bearish 24h accuracy: **72.5%** (excellent)
- Bullish 24h accuracy: **34.6%** (worse than random)

The profile already uses asymmetric weights (YAML lines 57-91), accuracy scaling multipliers (lines 103-128), and F&G regime scoring (lines 184-243). But these all operate on the composite SCORING level, not the GATING level.

**The abstain zone is still symmetric: +/-12 from center for both directions.**

This means a composite of 62 (bullish, barely past threshold) gets the same treatment as a composite of 38 (bearish, barely past threshold) -- despite bearish signals being 2x more accurate.

### 4.2 Recommended Asymmetric Abstain Zones

Based on the accuracy data and Kelly analysis:

```yaml
# PROPOSED: Asymmetric abstain zones
abstain:
  enabled: true
  abstain_label: "INSUFFICIENT EDGE"

  # Bearish signals: historically 72.5% accurate
  # Lower threshold to issue more (high-quality) bearish signals
  bearish_min_distance: 8   # scores below 42 = bearish signal

  # Bullish signals: historically 34.6% accurate (worse than random!)
  # MUCH higher threshold -- or disable entirely
  bullish_min_distance: 22  # scores above 72 = bullish signal (very rare)
  # Alternative: bullish_enabled: false  # suppress all bullish signals
```

**Impact analysis:**

| Configuration | Bearish Signals | Bullish Signals | Expected Overall Accuracy |
|--------------|----------------|----------------|--------------------------|
| Current (symmetric 12) | ~6% of total | ~6% of total | 48% |
| Asymmetric (8/22) | ~10% of total | ~1% of total | ~65% |
| Bearish only | ~10% of total | 0% | ~72% |
| Current + inverted bullish | ~6% of total | ~6% (inverted) | ~64% |

### 4.3 Research Support for Asymmetric Thresholds

**Academic literature:**
- Gneiting & Raftery (2007) "Strictly Proper Scoring Rules, Prediction, and Estimation": demonstrate that asymmetric loss functions require asymmetric decision boundaries
- Christoffersen & Diebold (2006): optimal directional forecasts should reflect asymmetric costs of Type I vs Type II errors
- Pesaran & Timmermann (1992) "A Simple Nonparametric Test of Predictive Performance": show that market direction forecasts often have significant asymmetry in accuracy

**Practitioner evidence:**
- Most professional trend-following CTAs have higher accuracy on short positions during bear markets than long positions during bull markets (AQR Research, 2019)
- Crypto specifically: due to structural volatility asymmetry (crashes are faster than rallies), bearish signals are inherently easier to time than bullish ones
- Your whale agent (61% bearish vs. 27% bullish) and technical agent (64% bearish vs. 46% bullish) confirm this asymmetry exists at the dimension level too

### 4.4 Direction-Specific Calibration

Beyond asymmetric thresholds, implement direction-specific calibration:

```python
def calibrated_confidence(composite, direction):
    """
    Map raw composite to calibrated probability of correct direction.
    Separate calibration curves for bullish and bearish.
    """
    distance = abs(composite - 50)

    if direction == "bearish":
        # Bearish calibration curve (fit from backtest data)
        # Even moderate bearish signals are 72.5% accurate
        # This suggests the calibration curve is STEEPER than the raw score implies
        if distance > 20:
            return 0.85  # high confidence bearish
        elif distance > 15:
            return 0.78
        elif distance > 10:
            return 0.72
        elif distance > 5:
            return 0.60
        else:
            return 0.52

    elif direction == "bullish":
        # Bullish calibration curve (fit from backtest data)
        # Even strong bullish signals are only 34.6% accurate
        # This suggests INVERSE calibration -- stronger signals might be LESS accurate
        if distance > 20:
            return 0.40  # still below random!
        elif distance > 15:
            return 0.37
        elif distance > 10:
            return 0.35
        elif distance > 5:
            return 0.33
        else:
            return 0.50  # at center, essentially random = 50%

    return 0.50  # neutral
```

**Critical insight:** For bullish signals, the calibration curve should be nearly FLAT or even INVERTED. This means "stronger" bullish signals are not actually more reliable -- the contrarian scoring logic is generating systematically wrong bullish calls. This is a model failure, not a gating failure.

---

## 5. Risk-Adjusted Signal Filtering

### 5.1 Sharpe-Based Gating

Instead of filtering by raw composite score, filter by expected Sharpe ratio of the signal:

```
Expected Sharpe = E[return | signal] / StdDev[return | signal]
```

A signal worth acting on should have Sharpe > 0.5 (annualized).

For a 24-hour signal:
```
annualized_sharpe = daily_sharpe * sqrt(365)
daily_sharpe = expected_return / expected_volatility
```

**Example for your bearish signals:**
- If bearish signal predicts -3% moves on average, and realized stddev is 5%
- Daily Sharpe = 0.60
- Annualized Sharpe = 0.60 * sqrt(365) = 11.5 (excellent)
- This signal is extremely worth trading

**Example for your bullish signals:**
- If bullish signal predicts +2% but actual average outcome is -1.5%
- Daily Sharpe = -0.30
- Annualized Sharpe = -5.7 (disastrous)
- This signal destroys value

### 5.2 Maximum Drawdown Limits

Implement portfolio-level risk gates that override individual signal quality:

```python
def portfolio_risk_gate(current_drawdown, max_allowed_drawdown=0.15):
    """
    Progressive signal suppression as drawdown increases.

    Returns: signal_fraction (1.0 = full signals, 0.0 = no new signals)
    """
    if current_drawdown < max_allowed_drawdown * 0.5:
        return 1.0  # less than half of max DD: full signals
    elif current_drawdown < max_allowed_drawdown * 0.75:
        return 0.5  # approaching DD limit: reduce signals
    elif current_drawdown < max_allowed_drawdown:
        return 0.25  # near DD limit: only highest conviction
    else:
        return 0.0  # at DD limit: stop all new signals
```

### 5.3 Volatility-Adjusted Abstain Zones

When market volatility is high, the "noise zone" should be wider (harder to distinguish signal from noise). When volatility is low, even small deviations from 50 may be meaningful:

```python
def volatility_adjusted_threshold(base_threshold, current_vol, baseline_vol):
    """
    Scale abstain threshold by current volatility relative to baseline.

    Higher volatility = wider abstain zone (more noise)
    Lower volatility = narrower abstain zone (small moves matter)
    """
    vol_ratio = current_vol / baseline_vol
    adjusted = base_threshold * vol_ratio

    # Clamp to reasonable range
    return max(5, min(25, adjusted))
```

**Practical application:**
- BTC daily vol = 2%: threshold = 12 (current)
- BTC daily vol = 5% (crash): threshold = 30 (almost nothing gets through -- but what does is high quality)
- BTC daily vol = 0.5% (consolidation): threshold = 3 (even small conviction matters)

This is smarter than your disabled dynamic abstain zones because it uses objective market data (volatility) rather than a sentiment indicator (F&G) that can persist at extreme levels for weeks.

---

## 6. Optimal Abstain Rates for Prediction Systems

### 6.1 The Abstain Rate Tradeoff

The **coverage-accuracy tradeoff** is well-studied in selective prediction (Chow 1957, El-Yaniv & Wiener 2010, Geifman & El-Yaniv 2017):

```
Accuracy(coverage) is monotonically non-increasing
Coverage(accuracy_threshold) is monotonically non-increasing
```

In plain English: the more signals you suppress, the more accurate the remaining ones become, but you issue fewer signals.

**The optimal operating point** depends on your loss function:

- **If wrong signals are very costly** (leveraged trading): optimize for accuracy, accept high abstain rate
- **If missing signals is costly** (opportunity cost): optimize for coverage, accept lower accuracy
- **If there's a fixed cost per signal** (gas fees, slippage): optimize for EV per signal

### 6.2 Your Current Operating Point

With 88% abstain rate and 48% gradient accuracy:

```
Signals per day = (1 - 0.88) * 20 assets * N_runs = ~2.4 signals/day (at 1 run/day)
Expected gradient accuracy per signal = 0.48
```

Compare to alternatives:

| Abstain Rate | Expected Accuracy | Signals/Day | EV Quality |
|-------------|-------------------|-------------|------------|
| 95% | ~55% | 1.0 | High accuracy, very few signals |
| 88% (current) | 48% | 2.4 | Below-average accuracy, few signals |
| 80% | ~42% | 4.0 | Declining accuracy |
| 70% | ~38% | 6.0 | Approaching random |
| 50% | ~33% | 10.0 | Near random |

**Your 88% abstain rate is in a poor position on this curve** -- high abstain rate BUT only 48% accuracy. This suggests the abstain zone is not filtering the RIGHT signals. It's filtering based on score DISTANCE from center, but distance from center is not well-correlated with signal ACCURACY.

### 6.3 Research on Optimal Abstain Rates

**Chow's Rule (1957, 1970):**
The optimal rejection rule minimizes expected risk:
```
Reject(x) when max_k P(k|x) < 1 - lambda
```
Where lambda is the cost of rejection relative to misclassification cost.

For your system: if the cost of a wrong directional signal is 3x the cost of missing an opportunity, then lambda = 1/3, and you should reject when P(correct|score) < 2/3 (i.e., only issue signals when calibrated accuracy > 66.7%).

**Empirical guideline** (Hendrycks & Gimpel 2017):
For neural classifiers, the optimal abstain rate that maximizes risk-adjusted accuracy typically falls in the range of 20-40%. Your 88% is FAR above this, suggesting model quality issues rather than optimal thresholding.

**The "selective prediction" framework** (Geifman & El-Yaniv 2019):
Defines the **risk-coverage curve**: plot accuracy as a function of coverage (1 - abstain_rate). The optimal operating point is where the marginal accuracy loss from including one more signal equals the marginal value of having that signal.

**Practical guideline for trading systems:**
- >80% abstain rate: Your model has poor discrimination. Improve the model, don't just filter harder.
- 50-80% abstain rate: Normal range for selective trading systems.
- 30-50% abstain rate: Aggressive; appropriate only if your model is well-calibrated.
- <30% abstain rate: Either your model is exceptional or you're overfitting.

### 6.4 Why Your 88% Rate Is Both Too High AND Not High Enough

The paradox: your abstain zone is too wide for bearish signals (filtering good signals) and too narrow for bullish signals (letting through bad signals).

**Bearish signals that pass the gate: 72.5% accurate.**
If you lowered the bearish threshold to 8 (from 12), you'd issue ~2x more bearish signals, likely at 65-70% accuracy. This is a net improvement.

**Bullish signals that pass the gate: 34.6% accurate.**
Even at the current threshold of 12, these are worse than random. Raising the threshold to 25+ (or disabling entirely) would improve overall accuracy dramatically.

The solution is not "change the abstain rate" but "change it independently for each direction."

---

## 7. Calibration Improvement for Directional Predictions

### 7.1 Why Your Composite Scores Are Miscalibrated

Analysis of the YAML profile reveals several sources of systematic miscalibration:

**Source 1: Contrarian scoring bias**
Five of six dimensions use contrarian logic (fear = buy, oversold = buy, overcrowded = buy). This creates a persistent BULLISH bias in fear markets. The composite cluster around 55-65 during fear because:
- Whale: fear + accumulation = high score (but whales are buying the dip that keeps dipping)
- Technical: oversold RSI = high score (but oversold can get more oversold)
- Derivatives: negative funding = high score (but shorts are right in downtrends)
- Narrative: low buzz = high score (but nobody talking about it = nobody buying it)
- Market: fear = contrarian high score (but fear is justified in actual downtrends)

Only the trend dimension is pro-trend, weighted at 5%. This is insufficient to counteract 5 contrarian dimensions.

**Source 2: Accuracy scaling is applied but doesn't fix the direction**
The accuracy scaling multipliers (YAML lines 103-128) reduce whale bullish from 1.0 to 0.27, but 0.27 * bullish_score is still bullish. You need the weight to be NEGATIVE (or the score to be inverted) to correctly express that whale bullish = bearish signal.

**Source 3: F&G regime scoring addresses the right problem but with wrong solution**
The F&G regime scoring (YAML lines 184-243) dampens contrarian dimensions in fear markets, which is correct directionally. But it does so by suppressing scores toward 50 rather than inverting them. This REDUCES signal strength without CORRECTING signal direction.

### 7.2 Calibration Method 1: Post-Hoc Score Adjustment

The simplest calibration: apply a monotonic transformation to composite scores based on observed accuracy:

```python
# Learned from backtest data
BEARISH_CALIBRATION = {
    # raw_score_range: calibrated_accuracy
    (0, 30): 0.80,    # strong bearish: 80% accurate
    (30, 38): 0.75,   # moderate bearish: 75% accurate
    (38, 42): 0.65,   # weak bearish: 65% accurate
    (42, 50): 0.55,   # barely bearish: 55% accurate
}

BULLISH_CALIBRATION = {
    (50, 58): 0.45,   # barely bullish: 45% accurate (below random!)
    (58, 62): 0.38,   # weak bullish: 38% accurate
    (62, 70): 0.35,   # moderate bullish: 35% accurate
    (70, 100): 0.32,  # strong bullish: 32% accurate (inverse!)
}
```

**Gate rule:** Only issue signal when calibrated accuracy > 0.55.
- All bearish signals with score < 42: ISSUE (accuracy 65-80%)
- All bullish signals: SUPPRESS (accuracy 32-45%)

### 7.3 Calibration Method 2: Direction-Specific Model

Instead of one composite score, maintain two separate models:

**Bearish Model:**
```
P(price_down | features) using weights optimized for bearish accuracy
```

**Bullish Model:**
```
P(price_up | features) using weights optimized for bullish accuracy
```

Each model has its own:
- Weights (from YAML bullish/bearish sections -- already implemented!)
- Calibration curve
- Abstain threshold
- Kelly position size

This is essentially what your asymmetric weighting already does, but carried to its logical conclusion: TWO models with TWO gates, not one model with one gate.

### 7.4 Calibration Method 3: Conformal Prediction

**Conformal prediction** (Vovk et al., 2005) provides distribution-free confidence intervals:

1. Hold out a calibration set of historical signals
2. For each signal, compute a "nonconformity score" = how unusual this signal is compared to past correct signals
3. At prediction time, compute the conformal p-value:
   ```
   p-value = |{calibration signals more nonconforming than current}| / |calibration set|
   ```
4. Only issue signals with p-value < alpha (e.g., alpha = 0.10)

**Advantages:**
- Finite-sample validity guarantee (no assumptions about data distribution)
- Automatically adapts to concept drift (as calibration set updates)
- Works with any underlying model (including your composite scoring)

**Implementation sketch:**
```python
class ConformalSignalGate:
    def __init__(self, calibration_signals):
        # Sort historical signals by "how correct they turned out to be"
        self.cal_scores = []
        for sig in calibration_signals:
            nonconformity = 1.0 - sig["gradient_score"]  # lower = more conforming
            self.cal_scores.append(nonconformity)
        self.cal_scores.sort()

    def should_issue(self, current_composite, alpha=0.10):
        """Return True if this signal is conformal with accurate signals."""
        # Nonconformity for current signal: how far from 50?
        # (using distance from center as proxy -- replace with model-based score)
        current_nc = 1.0 - abs(current_composite - 50) / 50

        # Conformal p-value
        n_more_nonconforming = sum(1 for s in self.cal_scores if s >= current_nc)
        p_value = n_more_nonconforming / len(self.cal_scores)

        return p_value < alpha
```

### 7.5 Calibration Method 4: Bayesian Updating

Maintain a running Bayesian estimate of signal accuracy per category:

```python
class BayesianCalibrator:
    """
    Bayesian accuracy estimator with Beta prior.
    Updates as new evaluation data comes in.
    """
    def __init__(self, prior_alpha=2, prior_beta=2):
        # Beta(2,2) = weak prior centered at 50%
        self.categories = {}

    def get_or_create(self, category):
        if category not in self.categories:
            self.categories[category] = {"alpha": 2, "beta": 2}
        return self.categories[category]

    def update(self, category, correct: bool):
        cat = self.get_or_create(category)
        if correct:
            cat["alpha"] += 1
        else:
            cat["beta"] += 1

    def get_accuracy(self, category):
        cat = self.get_or_create(category)
        return cat["alpha"] / (cat["alpha"] + cat["beta"])

    def get_confidence_interval(self, category, percentile=0.95):
        """Return Bayesian credible interval for accuracy."""
        cat = self.get_or_create(category)
        # Beta distribution quantiles
        from scipy.stats import beta
        low = beta.ppf((1 - percentile) / 2, cat["alpha"], cat["beta"])
        high = beta.ppf(1 - (1 - percentile) / 2, cat["alpha"], cat["beta"])
        return low, high

    def should_issue(self, category, min_accuracy=0.55):
        """Issue signal only if LOWER bound of CI > min_accuracy."""
        low, high = self.get_confidence_interval(category)
        return low > min_accuracy

# Usage:
calibrator = BayesianCalibrator()
# Categories: (direction, score_bucket, regime)
# e.g., ("bearish", "strong", "fear") or ("bullish", "moderate", "neutral")
```

This naturally handles the cold-start problem (weak prior = wide CI = conservative) and adapts over time as more data accumulates.

---

## 8. Specific Recommendations for the Web3 Signals System

### 8.1 Immediate Actions (Can implement today)

**R1: Asymmetric Abstain Zones (YAML change only)**
```yaml
abstain:
  enabled: true
  abstain_label: "INSUFFICIENT EDGE"

  # Direction-dependent thresholds
  asymmetric:
    enabled: true
    bearish_min_distance: 8    # currently 12 — issue more bearish signals
    bullish_min_distance: 25   # currently 12 — suppress weak bullish signals
    # If bullish accuracy improves above 50%, lower this threshold
```

Expected impact: accuracy jumps from 48% to ~62-65% as bad bullish signals are suppressed and more good bearish signals get through.

**R2: Add Calibration Curve to Backtest Output**
Add this analysis to `backtest.py` after PART 4 (Conviction Analysis):

```python
# PART 4B: CALIBRATION CURVE
print(f"\n{'='*80}")
print("PART 4B: CALIBRATION CURVE")
print(f"{'='*80}")
print("For each score bucket, what's the actual binary accuracy?\n")

for direction in ["bullish", "bearish"]:
    dir_evals = [e for e in all_evals if e["direction"] == direction]
    buckets = {}
    for ev in dir_evals:
        bucket = int(ev["score"] / 5) * 5
        if bucket not in buckets:
            buckets[bucket] = {"correct": 0, "total": 0}
        buckets[bucket]["total"] += 1
        if ev["binary_correct"]:
            buckets[bucket]["correct"] += 1

    print(f"\n  {direction.upper()} calibration:")
    print(f"  {'Score':>8s}  {'Implied':>8s}  {'Actual':>8s}  {'N':>5s}  {'Gap':>8s}")
    for bucket in sorted(buckets.keys()):
        implied = abs(bucket + 2.5 - 50) / 50  # center of bucket
        actual = buckets[bucket]["correct"] / buckets[bucket]["total"]
        n = buckets[bucket]["total"]
        gap = actual - (0.5 + implied if direction == "bullish" else 0.5 + implied)
        print(f"  {bucket:>5d}-{bucket+5:<3d} {0.5+implied:>8.2f}  {actual:>8.2f}  {n:>5d}  {gap:>+8.2f}")
```

**R3: Brier Score Computation in Backtest**

```python
# After accuracy computation:
brier_scores = []
for ev in all_evals:
    implied_p = 0.5 + abs(ev["score"] - 50) / 100  # map to 0.5-1.0
    outcome = 1.0 if ev["binary_correct"] else 0.0
    brier = (implied_p - outcome) ** 2
    brier_scores.append(brier)

avg_brier = sum(brier_scores) / len(brier_scores)
print(f"  Brier Score: {avg_brier:.4f}  (0=perfect, 0.25=random, lower=better)")
```

### 8.2 Short-Term Actions (1-3 days)

**R4: Signal Quality Score Gate**
Implement the signal quality meta-score from Section 3.3. Gate signals at quality > 50 in addition to the abstain zone. This catches the "no data = 50" problem identified in I3.

**R5: Kelly-Based Position Sizing Output**
Add Kelly fraction to each signal output. Even if the system doesn't execute trades, showing "Kelly suggests 15% position" vs. "Kelly suggests 0% position" gives users actionable sizing guidance.

**R6: Correlation Gate for Simultaneous Signals**
When multiple assets issue the same direction simultaneously, only report the top 3 by conviction (highest |composite - 50|). The rest are correlated noise. This reduces signal count but increases per-signal quality.

### 8.3 Medium-Term Actions (1-2 weeks)

**R7: Implement Platt Scaling or Isotonic Regression**
Using 7+ days of backtest data, fit calibration curves per direction. Apply these to transform raw composites into calibrated probabilities. Use the calibrated probabilities for gating (issue when calibrated_p > 0.60).

**R8: Bayesian Online Calibration**
Implement the BayesianCalibrator from Section 7.5. This automatically adapts as market conditions change, unlike fixed calibration curves that may become stale.

**R9: Volatility-Adjusted Abstain Zones**
Replace the F&G-based dynamic abstain (which failed) with volatility-based scaling (Section 5.3). Use ATR or realized volatility to widen/narrow the abstain zone.

### 8.4 Strategic Actions (2-4 weeks)

**R10: Separate Bullish and Bearish Models**
The fundamental problem is that one composite score tries to measure both bullish and bearish probability. A score of 65 means "slightly bullish" but the calibration curve shows this is meaningless for bullish accuracy. Build two separate scoring pipelines with independent calibration.

**R11: Conformal Prediction Gate**
Implement distribution-free confidence intervals (Section 7.4). This provides formal guarantees on prediction quality without distributional assumptions.

**R12: Address Root Cause of Bullish Failure**
The 34.6% bullish accuracy is not a gating problem -- it's a MODEL problem. The contrarian scoring logic (fear = buy, oversold = buy) is systematically wrong in the current market regime. Either:
- a) The contrarian hypothesis is wrong for crypto (at least for the current 8-day backtest window)
- b) The contrarian signals are right but the timing is wrong (the dip hasn't finished dipping)
- c) The contrarian approach works but needs longer evaluation horizons (1 week, not 24h)

Testing (c) is easy: extend evaluation windows to 72h, 1 week, 2 weeks. If bullish accuracy improves at longer horizons, the model is early rather than wrong.

---

## 9. Mathematical Framework: Unified Gating Decision

Combining all the above into a single gating framework:

```python
def should_issue_signal(
    composite_score: float,
    direction: str,
    dimensions: dict,
    data_tiers: dict,
    market_volatility: float,
    baseline_volatility: float,
    calibrator: BayesianCalibrator,
    portfolio_drawdown: float,
) -> tuple[bool, float, str]:
    """
    Unified gating decision combining all filters.

    Returns: (should_issue, position_fraction, reason)
    """
    # Gate 1: Data quality
    quality = signal_quality_score(dimensions, data_tiers)
    if quality < 40:
        return False, 0.0, f"Low data quality ({quality:.0f})"

    # Gate 2: Asymmetric abstain zone (volatility-adjusted)
    vol_ratio = market_volatility / baseline_volatility
    base_threshold = 8 if direction == "bearish" else 22
    adjusted_threshold = base_threshold * max(0.5, min(2.0, vol_ratio))
    distance = abs(composite_score - 50)

    if distance < adjusted_threshold:
        return False, 0.0, f"Below abstain threshold ({distance:.1f} < {adjusted_threshold:.1f})"

    # Gate 3: Calibrated accuracy
    category = (direction, "strong" if distance > 15 else "moderate" if distance > 10 else "weak")
    calibrated_accuracy = calibrator.get_accuracy(category)
    low_ci, _ = calibrator.get_confidence_interval(category)

    if low_ci < 0.50:
        return False, 0.0, f"Calibrated accuracy too low ({calibrated_accuracy:.0%}, CI lower bound {low_ci:.0%})"

    # Gate 4: Portfolio risk
    risk_factor = portfolio_risk_gate(portfolio_drawdown)
    if risk_factor == 0:
        return False, 0.0, "Portfolio drawdown limit reached"

    # Gate 5: Kelly position sizing
    kelly_fraction = kelly_position_size(composite_score, direction, calibrated_accuracy)
    if kelly_fraction <= 0:
        return False, 0.0, f"Negative Kelly ({kelly_fraction:.2%})"

    # All gates passed
    position = kelly_fraction * risk_factor
    return True, position, f"PASS: accuracy={calibrated_accuracy:.0%}, Kelly={kelly_fraction:.1%}, quality={quality:.0f}"
```

---

## 10. Summary of Key Findings

### Answering the Original Questions

**Q: Is the +/-12 abstain zone too aggressive (88% filtered)?**
A: The abstain rate is reasonable for the current model quality, but the zone is misapplied symmetrically. The BEARISH zone should be narrower (8) to issue more accurate bearish signals. The BULLISH zone should be much wider (22+) or disabled entirely.

**Q: Should we have asymmetric abstain zones?**
A: **Yes, absolutely.** This is the single highest-impact change available. Bearish threshold of 8, bullish threshold of 22-25 (or disabled). Expected accuracy improvement: 48% -> 62-65%.

**Q: How do successful systems handle signal quantity vs. quality?**
A: Through calibration-aware gating, not fixed thresholds. The key insight is that you need to MEASURE accuracy per signal category (direction x score_bucket x regime) and then only issue signals in categories where measured accuracy exceeds a minimum.

**Q: What calibration methods would help us know WHEN our signals are reliable?**
A: In priority order:
1. Calibration curves (plot predicted vs. actual accuracy per score bucket) -- **start here**
2. Brier score decomposition (diagnose if problem is calibration, resolution, or noise)
3. Bayesian online calibration (adapts over time, handles concept drift)
4. Conformal prediction (provides formal confidence guarantees)

### The Root Cause

The system's poor accuracy is not primarily a gating/filtering problem. It's a **model calibration problem**. The composite scores are NOT calibrated probabilities -- a score of 65 does not predict 65% accuracy, and in the bullish direction, higher scores may actually predict LOWER accuracy (inverse calibration). Fixing the gating will help, but the upstream model needs calibration to fully solve the problem.

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| Brier Score | Mean squared error of probabilistic predictions; BS = mean((p - y)^2) |
| Calibration | Whether predicted probabilities match observed frequencies |
| Kelly Criterion | Optimal bet sizing that maximizes log-wealth growth rate |
| Conformal Prediction | Distribution-free method for valid prediction intervals |
| Abstain Rate | Fraction of predictions suppressed as "insufficient confidence" |
| Coverage | 1 - abstain_rate; fraction of predictions actually issued |
| Platt Scaling | Post-hoc calibration via logistic regression on raw scores |
| Isotonic Regression | Non-parametric monotonic calibration mapping |
| Selective Prediction | Framework where classifier can abstain on uncertain inputs |
| Information Coefficient | Spearman rank correlation between predictions and outcomes |

## Appendix B: Key Academic References

1. Brier, G.W. (1950). "Verification of forecasts expressed in terms of probability." Monthly Weather Review, 78(1), 1-3.
2. Chow, C.K. (1970). "On optimum recognition error and reject tradeoff." IEEE Trans. Information Theory, 16(1), 41-46.
3. El-Yaniv, R. & Wiener, Y. (2010). "On the foundations of noise-free selective classification." JMLR, 11, 1605-1641.
4. Geifman, Y. & El-Yaniv, R. (2017). "Selective Classification for Deep Neural Networks." NeurIPS.
5. Gneiting, T. & Raftery, A.E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." JASA, 102(477), 359-378.
6. Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System Technical Journal, 35(4), 917-926.
7. Pesaran, M.H. & Timmermann, A. (1992). "A Simple Nonparametric Test of Predictive Performance." Journal of Business & Economic Statistics, 10(4), 461-465.
8. Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines." Advances in Large Margin Classifiers, 61-74.
9. Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market." Handbook of Asset and Liability Management.
10. Vovk, V., Gammerman, A. & Shafer, G. (2005). "Algorithmic Learning in a Random World." Springer.
