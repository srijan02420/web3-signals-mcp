# Fusion Engine Pipeline Deep-Dive Analysis (I1)

**Date:** 2026-03-15
**Agent:** I1 — Fusion Engine Pipeline Deep-Dive

## Executive Summary

The signal fusion engine transforms raw dimension scores through **7 distinct transformation layers**. The system is heavily contrarian (5 out of 6 dimensions fight the trend).

**KEY FINDING**: No wholesale signal inversion. Instead, **multi-layered suppression** crushes bullish signals:
- A strong contrarian buy (raw score=84) gets dampened to 49 through THREE mechanisms
- Trend dampening (0.7x) + Velocity dampening (0.4x) + Accuracy scaling (0.46x)
- Bearish signals are PRESERVED; bullish signals are SUPPRESSED

## Transformation Layers

### Layer 1: Raw Dimension Scoring
- WHALE: Ratio-based (accumulate/sell), exchange flow → 0-100
- TECHNICAL: RSI, MACD, MA, trend → CONTRARIAN (oversold=high, overbought=low)
- DERIVATIVES: L/S ratio, funding, OI → ~30-70 centered
- NARRATIVE: Volume (inverted), LLM, community → 0-100
- MARKET: Price change (contrarian), volume, F&G → 50-centered
- TREND: MA alignment, RSI momentum (PRO-TREND) → 50-centered

### Layer 2: Data Tier Detection
- full/partial/none classification per agent

### Layer 3: Direction-Aware Weight Selection
- raw_avg > 50 → bullish weights; < 50 → bearish weights

### Layer 4: Accuracy Scaling
- Multiplies weights by historical directional accuracy
- Whale bullish: ×0.27, Technical bullish: ×0.46

### Layer 5: Regime Weighting
- Trending: suppress whale ×0.3, suppress trend ×0.5
- Ranging: boost whale ×1.3, boost narrative ×1.2

### Layer 6: F&G Regime Scoring
- Extreme Fear: market ×1.3, suppress whale ×0.15, suppress trend ×0.15
- Greed: whale ×1.3, suppress market ×0.7

### Layer 7: Data Tier Multipliers + Score Dampening
- 7B-i: Trend dampening (30% toward 50 in downtrends)
- 7B-ii: F&G score dampening (suppress contrarian in fear)
- 7B-iii: Velocity dampening (reduce distance from 50 when indicators accelerate)

### Layer 8: Abstain Check (±12 from center)

## Example Trace: Bullish Signal Destruction

**Scenario:** RSI=28 (oversold), MACD bearish, below MA30 → Strong contrarian buy

1. Raw technical score: **84.33**
2. Accuracy scaling (bullish ×0.46): weight reduced
3. Trend dampening (BTC 8% below MA30): 84→74 (−10 pts)
4. Velocity dampening (RSI falling): 74→59.6 (−14 pts)
5. **Final contribution: ~4.7 points** to composite despite strongest possible signal

## Key Conclusions

1. No wholesale signal inversion — direction preserved
2. Sophisticated suppression prevents false contrarian calls in downtrends
3. Asymmetric: bearish PRESERVED, bullish SUPPRESSED
4. Abstain ±12 is empirically validated
5. Conviction boost correctly disabled (41.7% vs 51.1%)
6. Anti-predictive dims (whale IC=-0.08, trend IC=-0.13) heavily discounted
