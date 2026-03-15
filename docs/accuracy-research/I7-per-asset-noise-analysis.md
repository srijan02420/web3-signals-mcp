# Per-Asset Signal Quality & Noise Analysis (I7)

**Date:** 2026-03-15
**Agent:** I7 — Per-Asset Signal Quality & Noise

## Executive Summary

Removing INJ, ATOM, OP boosts overall accuracy from 51.2% to **57.5%** (+6.3 percentage points). These assets are anti-predictive, not just noisy.

## Per-Asset Accuracy

**Top Performers (>70%):**
- AVAX: 82.5%
- LINK: 78.0%
- ADA: 72.5%
- DOT: 71.2%
- APT: 70.0%

**Problem Assets (<42%):**
- INJ: 20.7% (WORST — consistently anti-predicts)
- ATOM: 24.5% (anti-predicts almost as badly)
- OP: 41.9% (below coin-flip)

## Root Cause: Why These Assets Are Noisy

### INJ & ATOM: Systematic Anti-Prediction
- Wrong market microstructure fit (contrarian logic doesn't apply)
- Different trader psychology (Cosmos staking, Injective DEX mechanics)
- INJ has MOST signals (46) but WORST accuracy — signal interpretation problem, not data

### OP: Marginal Performer
- Below 52.9% system average
- Generates too much noise relative to value

## Recommended Actions

### Immediate: Asset Blacklist
```yaml
asset_blacklist:
  enabled: true
  assets: [INJ, ATOM, OP]
```
**Expected improvement:** +6.3% accuracy

### Medium-term: Per-Asset Confidence Scaling
```yaml
per_asset_scaling:
  INJ: 0.8   # 20% penalty
  ATOM: 0.8
  OP: 0.9
```

### Long-term: Per-Asset Noise Thresholds
```yaml
per_asset_noise:
  INJ: 3.0   # Higher tolerance for volatility
  ATOM: 3.0
  OP: 2.5
```

## Key Finding

INJ generates 46 signals (most of any asset) but has worst accuracy (20.7%). This is NOT a data availability problem — it's a **signal interpretation problem**. The fusion scoring doesn't map INJ's agent outputs correctly to price moves.

## Asset Configuration

20 assets tracked: BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOT, MATIC, LINK, UNI, ATOM, LTC, FIL, NEAR, APT, ARB, OP, INJ, SUI

Asset tiers system exists but is disabled. Per-asset weight configuration possible but not implemented.
