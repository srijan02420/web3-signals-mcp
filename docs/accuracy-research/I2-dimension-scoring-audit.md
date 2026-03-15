# Dimension Scoring Quality Audit (I2)

**Date:** 2026-03-15
**Agent:** I2 — Dimension Scoring Quality Audit

## Executive Summary

Critical scoring bugs and design conflicts identified across all dimensions. The root cause of 34.6% bullish accuracy (worse than random) is a **double-inversion**: contrarian scoring + accuracy scaling together crush bullish signals.

## Critical Findings

### 1. Technical (IC=-0.46) — Asymmetric Scoring Bug

RSI scoring awards 30 points for oversold (<30) but only 10 for overbought (>70). The LABELS are correct (oversold=bullish opportunity) but the scoring creates a bearish bias:
- When RSI oversold → score high → predict "buy"
- But in bearish market, price continues down → prediction wrong → IC negative

The technical condition requires BOTH 30D trend bullish AND 7D trend bullish, but scorers apply contrarian logic (oversold=good). **Conflicting philosophies within same dimension.**

### 2. Whale (IC=-0.53) — Exchange Flow Interpretation Issues

Exchange inflow marked as "sell" (bearish) — conceptually correct but:
- Inflows can precede bounces (shorts covering, liquidations clearing)
- Exchange flow is DISABLED by default (flow_cfg.enabled=false)
- Most backtest signals ignore inflow/outflow distinction entirely

### 3. Trend (IC=-0.13) — Pro-Trend in Contrarian System

Trend dimension is explicitly PRO-TREND (momentum-following), not contrarian. Suppressed to 0.15 in both directions. Should be:
- In TRENDING market: full weight (momentum works)
- In RANGING market: suppress (contrarian works)
- Currently: suppressed everywhere → useless

### 4. Root Cause: Bearish 72.5% >> Bullish 34.6%

The backtest uses CONTRARIAN scoring across all dimensions:
- Technical: oversold=high score (contrarian buy)
- Whale: accumulation=bullish
- Market: fear=bullish, greed=bearish
- Narrative: peak crowded=bearish

In BEARISH markets: contrarian signals are RIGHT (buy dips works)
In BULLISH markets: contrarian signals are WRONG (signals noise)

**Accuracy scaling amplifies the problem:**
- Technical bullish: ×0.46 (crippled)
- Whale bullish: ×0.27 (almost zero!)
- Market bullish: ×0.65 (boosted, good)

**You can't have BOTH accuracy scaling AND contrarian scoring.** Pick one:
- Option A: Neutral scoring + accuracy scaling
- Option B: Contrarian scoring + NO accuracy scaling

Currently doing both = double-penalty on bullish signals.

### 5. Complete Contrarian Inversion List

| Dimension | Logic | Label Issue | IC Impact |
|-----------|-------|-------------|-----------|
| Technical RSI | Oversold=+30 | Correct label | Wrong in bearish window |
| Technical MA | Below=0, Above=10 | Contradicts oversold | Conflicting signals |
| Whale Accum | from_exchange=bullish | Correct | IC negative due to timing |
| Whale Inflow | to_exchange=bearish | Correct | Inflows at bottoms |
| Narrative Vol | Inverted if enabled | Contrarian | Correct for euphoria |
| Market Price | Strong positive=-5 | Contrarian | IC=+0.31 (works alone) |

## Recommendations

1. **Disable accuracy scaling** OR adopt neutral scoring + keep scaling
2. **Fix technical dimension** — choose fully contrarian OR fully momentum
3. **Disable whale inflow logic** when exchange_flow already disabled
4. **Extend backtest window** to include UP markets (currently 8 days bearish-biased)
5. Move to **regime-aware multi-strategy** (contrarian in fear, momentum in greed)
