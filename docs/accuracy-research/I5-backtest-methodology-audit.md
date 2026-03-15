# Backtest Methodology & Validation Audit (I5)

**Date:** 2026-03-15
**Agent:** I5 — Backtest Methodology Audit

## Executive Summary

**CRITICAL: Look-ahead bias vulnerability in backtest.py.** The backtest lacks temporal gating that retroactive_accuracy.py correctly implements. Gradient scoring thresholds are not asset-calibrated. Risk metrics are missing.

## 1. Look-Ahead Bias — CRITICAL

**backtest.py** (lines 1339-1346): No check to ensure target_time is in the past.

**retroactive_accuracy.py** does it right (lines 257-310):
```python
if target_24h <= now_ts:  # Only score if 24h has actually passed
    price_at_24h = find_price_at_time(prices, target_24h)
```

**Impact:** Reported accuracies from backtest.py may be unreliable.

**Fix:** Add temporal gate:
```python
now_ts = datetime.now(timezone.utc).timestamp()
if target_time.timestamp() > now_ts:
    continue
```

## 2. Deduplication — CORRECT

1-signal-per-12h via AM/PM bucket key. Applied consistently in both files.
Minor caveat: boundary artifacts possible at 11:50 AM / 12:10 PM.

## 3. Price Selection — CORRECT

Both use closest price within tolerance window. Uses hourly candle closes.
Discrepancy: backtest uses 6h tolerance, retroactive uses 2h.
**Recommendation:** Reduce backtest tolerance to 2h for consistency.

## 4. Gradient Scoring — UNCALIBRATED

Fixed 2%/5% thresholds not asset-specific:
- BTC: 2% move = ~0.8 std devs (meaningful)
- INJ: 2% move = ~0.25 std devs (noise)

**Recommendation:** Asset-specific thresholds:
- BTC: noise=0.5%, strong=2.5%
- ETH: noise=1.0%, strong=4.0%
- Alts: noise=2.0%, strong=5.0%

## 5. Missing Metrics

| Metric | backtest.py | retroactive_accuracy.py |
|--------|-------------|-------------------------|
| Gradient accuracy | Yes | Yes |
| Binary accuracy | Yes | No |
| IC (Spearman) | Yes | No |
| Sharpe ratio | **No** | No |
| Max drawdown | **No** | No |
| Profit factor | **No** | No |

## 6. Statistical Significance

- Adequate for 5-asset analysis
- Marginal for 20-asset analysis (~5-10 per asset)
- With look-ahead bias, all numbers are suspect

## Audit Summary

| Area | Status | Severity |
|------|--------|----------|
| Look-ahead bias | FAIL | CRITICAL |
| Deduplication | PASS | LOW |
| Price selection | PASS | NONE |
| Gradient calibration | WARN | MEDIUM |
| Evaluation metrics | WARN | MEDIUM |
| Data sufficiency | PASS | NONE |
| File consistency | FAIL | HIGH |
