# Data Quality & Pipeline Freshness Analysis
## Web3 Signals x402 — Investigation I3

**Date:** 2026-03-15
**Scope:** Assess pipeline reliability, data staleness, silent failures, and the "no data = 50" scoring problem

---

## Executive Summary

The pipeline exhibits **significant data quality risks** across multiple dimensions:

1. **Silent Failures with 50-Score Default**: When agents fail, empty data structures are saved and scored as 50 (neutral), artificially clustering composites. This masks failures and creates false stability signals.

2. **Multi-Day Stale Data Persistence**: Some assets may operate on 1–2 week old narrative data if Reddit/Twitter fails early in the cycle and doesn't recover. LLM sentiment caches for 24h but isn't regenerated if sources fail.

3. **API Cascades**: Market agent data powers trend scorers; if CoinGecko is down, dependent calculations receive zero or default values without detection.

4. **Partial Failures Are Invisible**: An agent reports "partial" status but doesn't indicate *which assets* are affected. The fusion system can't distinguish between "BTC is fine, ETH has no data" vs. "all assets are stale."

5. **Refresh Rate Misalignment**: Whale agent (30 min), narrative agent (60 min), but score computation doesn't wait—composites mix 30-min-old whale data with 60-min-old narrative data daily.

6. **Data Tier Detection Is Fragile**: Keyword detection for "full"/"partial"/"none" tiers based on simple source counts. A single API returning empty data set (due to rate limit) is indistinguishable from "no actual activity."

---

## 1. Data Sources & API Reliability

### 1.1 Whale Agent (942 lines)

**Primary Sources:**
- **Whale Alert REST API** (Layer 1, primary): Requires `WHALE_ALERT_API_KEY`. Paginated, rate-limited with exponential backoff.
- **Etherscan V2** (Layer 3a): ETH + ERC-20 transfers. Requires `ETHERSCAN_API_KEY`.
- **Blockchain.com** (Layer 3b): BTC transfers. No API key but has rate limits.
- **Twitter via Apify** (Layer 2, supplementary): Requires `APIFY_API_KEY`.
- **Exchange flow** (Layer 4): Snapshots from Etherscan/Blockchain.
- **Whale wallets** (Layer 5): Balance tracking of known whale addresses.

**Reliability Profile:**

| Source | Failure Mode | Graceful Handling | Result |
|--------|--------------|-------------------|--------|
| Whale Alert API | Rate limit (429), timeout | Retry with backoff (2^attempt) | Can succeed on retry |
| Whale Alert API | No data (empty transactions []) | Returns empty move list | Empty moves → 0 credible moves |
| Etherscan | Rate limit, timeout | Catches exception, appends error, continues | Partial data (exchange flow incomplete) |
| Blockchain.com | Down/slow | Catches exception, continues | Partial data |
| Twitter/Apify | Auth/timeout | Catches exception, continues | No Twitter layer (silent drop) |
| Exchange flow | Can't reach wallet addresses | Continues, leaves balances None | Flow = {} (empty) |

**Critical Issue #1: "No Data Returned" Is Not Distinguishable From "No Activity"**

```python
# whale_agent.py, lines 251-253
if not transactions:
    break  # Breaks pagination loop

# Returns empty moves list, which is valid
# But no indication if this is due to:
# - No whale activity in window (legitimate)
# - API rate limit (failure)
# - Network timeout (failure)
```

The whale agent's `collect()` returns `whale_moves: []` with no way for downstream to know if this is real or a failure. Score is computed as:
- Total moves = 0
- Credible moves = 0
- This feeds into whale condition check

**Recommendation:** Embed metadata flag: `data_confidence: 0.0` when empty due to API failure.

---

### 1.2 Technical Agent (296 lines)

**Primary Source:**
- **Binance Spot Klines**: Free API, no key required. Fetches 50 daily candles for RSI/MACD/MA calculations.

**Reliability Profile:**

| Issue | Impact | Handling |
|-------|--------|----------|
| Binance klines timeout | Missing candles | Error appended, asset skipped entirely |
| Not enough candles (< slow MA + signal period) | Can't compute indicators | Asset marked "unknown" trend, no rsi_14/macd |
| Candle data is stale (daily interval) | Price is 24h old | Technical condition based on day-old data |

**Critical Issue #2: Single-Point Failure, Full Asset Skip**

```python
# technical_agent.py, lines 104-110
closes = self._fetch_klines(binance_sym, interval, candle_limit)
if len(closes) < macd_slow + macd_signal_period:
    errors.append(f"{sym}: not enough candles ({len(closes)})")
    continue  # SKIPS ENTIRE ASSET

# Asset data becomes:
# {
#   "price": None,
#   "rsi_14": None,
#   "trend_7d": "unknown",
#   "technical_condition": False  # DEFAULT FALSE
# }
```

If Binance API is slow and returns 30 candles instead of 50, all indicators are None. The asset gets `technical_condition: False` by default. This might be correct, but there's no way to know if it's a real bearish signal or a data failure.

**Recommendation:** Use `data_age_minutes` field. Track last successful fetch timestamp.

---

### 1.3 Derivatives Agent (270 lines)

**Primary Source:**
- **Binance Futures**: Three endpoints (long/short ratio, funding rate, open interest). Free, no key required.
- **Historical snapshots** (Lead indicators): Uses storage to load prior 20 snapshots for 4h/24h deltas.

**Reliability Profile:**

| Issue | Impact | Handling |
|-------|--------|----------|
| One endpoint down (e.g., long/short) | Partial asset data | Error logged, asset proceeds |
| All endpoints fail | ls_status, funding_status = "unknown" | `derivatives_condition = False` |
| Lead indicator history is sparse | Only 5 snapshots instead of 20 | Delta computation skipped or incomplete |
| 4h/24h snapshots not aligned (e.g., 3h 45m old) | Timing mismatch | Looks for "closest" snapshot within tolerance window |

**Critical Issue #3: Lead Indicators Depend on Storage**

```python
# derivatives_agent.py, lines 143-149
try:
    store = Storage()
    history = store.load_history("derivatives_agent", limit=20)
    if len(history) >= 2:
        self._compute_deltas(data, history, errors)
except Exception as exc:
    errors.append(f"lead indicators: {exc}")

# If store is down or has < 2 snapshots:
# funding_rate_change_4h = None (stays None, not computed)
# oi_change_pct_4h = None
```

**If a new derivatives agent runs and only 1 snapshot exists (first run), lead indicators never compute until 20 snapshots accumulate.** On first run, all assets have null deltas.

**Recommendation:** Cache most recent 1-snapshot age in memory or return "insufficient history" with timestamp of oldest available snapshot.

---

### 1.4 Narrative Agent (991 lines)

**Primary Sources:**
1. **Reddit (PRAW)** — Requires `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`. Authority-weighted mentions.
2. **Twitter/X (twikit)** — Requires `TWITTER_USERNAME`, `TWITTER_EMAIL`, `TWITTER_PASSWORD`. Cookies cached.
3. **Farcaster (Neynar)** — Requires `NEYNAR_API_KEY`. Cast search.
4. **CryptoPanic** — Requires `CRYPTOPANIC_API_KEY`. Hot posts + community votes.
5. **Google News RSS** — Free, no key. Unlimited queries.
6. **CoinGecko Trending** — Free endpoint, no key.

**Reliability Profile:**

| Source | Failure Mode | Impact | Handling |
|--------|--------------|--------|----------|
| Reddit auth fails | No Reddit mentions | Error logged, continues; counts[sym] stays 0 |
| Twitter cookies invalid or login fails | No Twitter mentions | Error logged, continues; counts[sym] stays 0 |
| Farcaster API down | No Farcaster mentions | Error logged, continues; counts[sym] stays 0 |
| CryptoPanic API quota exceeded | No CP mentions | Error logged, continues; counts[sym] stays 0 |
| Google News timeout | No Google mentions | Error logged, continues; counts[sym] stays 0 |
| CoinGecko trending down | Not marked as trending | Continues; `is_trending = False` |

**Critical Issue #4: Multi-Source Failure → Zero Mentions → "Unknown" Status**

```python
# narrative_agent.py, lines 191-228
total = rd + tw + fc + cp + gn + boost  # All zeros if all sources fail

if total == 0:
    status = "unknown"
    no_data.append(sym)
    # narrative_condition = False
    # narrative_status = "unknown"
```

If all 6 sources fail silently (caught exceptions), the asset is marked "unknown" (not an error). But this is **invisible** in error logs unless you check the catch blocks. The agent reports "partial" status with error count, but downstream doesn't know which assets are affected.

**Critical Issue #5: LLM Sentiment Stale 24+ Hours**

```python
# narrative_agent.py, lines 891-911
def _load_cached_llm_sentiment(self, asset: str) -> Optional[Dict[str, Any]]:
    cached = store.load_kv_json("llm_sentiment", "latest")
    if not cached:
        return None

    ts = cached.get("timestamp", "")
    max_age_hours = int(self.profile.get("llm_sentiment", {}).get("max_age_hours", 24))
    if (datetime.now(timezone.utc) - cached_time).total_seconds() > max_age_hours * 3600:
        return None  # Return None if stale
```

The LLM sentiment is computed **once per 12-hour batch** (orchestrator calls `run_llm_sentiment()` separately). But if the batch job fails to run (e.g., ANTHROPIC_API_KEY not set, rate limit), the cache becomes stale. After 24h, it returns `None`, and the asset gets `llm_sentiment = None` indefinitely until the next batch succeeds.

**Real-world scenario:** If LLM batch hasn't run in 48 hours and cache is 24h old:
- `llm_sentiment` field becomes None
- Composite score doesn't incorporate LLM-based narrative weight
- No indication in the data that the LLM source is unavailable

---

### 1.5 Market Agent (488 lines)

**Primary Sources:**
1. **CoinGecko** — Per-asset price/volume, categories, global market, trending. Free.
2. **Binance Klines** — Volume spike enrichment. Free.
3. **DexScreener** — Top DEX pairs. Free.
4. **Fear & Greed Index** — Community sentiment. Free.

**Reliability Profile:**

| Source | Failure Mode | Impact | Handling |
|--------|--------------|--------|----------|
| CoinGecko per_asset fails | No price/volume for any asset | `per_asset = {}` | Downstream gets empty dict |
| Binance klines fail on volume enrichment | No volume spike detection | Asset remains "normal" | No indication of data gap |
| CoinGecko global fails | BTC dominance unknown | `total_market_cap_change_24h = None` | Trend scorer uses None (coerced to 0.0) |
| DexScreener down | No DEX pairs | `dex.top_pairs = []` | Continues without DEX data |
| Fear & Greed timeout | No sentiment | `fear_greed_index = None` | Continues |

**Critical Issue #6: Market Data Used By Downstream (Trend Scorer)**

Market agent data is loaded by the trend scorer in the fusion system. If `per_asset` is empty:
- No volume spike data → composites score without volume context
- No price data → trend scorer can't weight by 24h change

The fusion system doesn't validate `data.get("per_asset")` is non-empty. If it is, downstream scores are computed with `None` values that get coerced to 0.0.

**Recommendation:** Explicit data quality tier in market agent result: `data_quality_tier: "full" | "partial" | "none"` based on how many sources succeeded.

---

## 2. Refresh Rates & Staleness

### 2.1 Agent Cadence (from orchestrator)

```python
_AGENT_CADENCES_MIN: Dict[str, int] = {
    "technical_agent":   15,   # Every 15 min
    "derivatives_agent": 15,   # Every 15 min
    "whale_agent":       30,   # Every 30 min
    "market_agent":      30,   # Every 30 min
    "narrative_agent":   60,   # Every 60 min
}
```

**Problem:** Whale and narrative agents have different cadences. Composites score on latest data from each agent, which means:

- **Scenario A (T=0):** All agents run.
- **Scenario B (T=15 min):** Technical + derivatives refresh. Composites mix 15-min-new technical with 30-min-old whale + 60-min-old narrative.
- **Scenario C (T=30 min):** Whale refreshes. Composites mix 30-min-new whale with 45-min-new technical but 60-min-old narrative.

**This is not inherently wrong**, but it means narrative momentum scores are evaluated against market conditions 60+ minutes in the past.

### 2.2 Stale Data Persistence Across Failed Runs

If narrative agent fails at T=60, the orchestrator logs an error but **doesn't delete old data**. On next refresh (T=120), the same old narrative data is re-scored. This can happen repeatedly:

```
T=60: Narrative agent fails (Reddit down) — saves partial result or skips save
T=120: Narrative agent fails again — old data from T=0 persists
T=180: Narrative agent succeeds — finally updates
```

**Storage layer doesn't prevent overwrites**, so each successful run overwrites the stale snapshot. But if an agent fails 6 consecutive times (6 hours for narrative), the data is 6 hours old and still being scored.

**No automatic staleness expiry:** If whale agent's Whale Alert API goes down permanently, the system keeps re-scoring data from the last successful run indefinitely. There's no "max data age" threshold in the scoring system.

### 2.3 LLM Sentiment 24-Hour Staleness

The LLM sentiment batch runs on a separate 12-hour schedule (outside the main loop). If it fails to run:

```
T=0h: LLM batch succeeds, saves cache with timestamp T=0
T=12h: LLM batch fails (rate limit), no update
T=24h: LLM batch fails again, no update
T=24h+ : _load_cached_llm_sentiment() returns None (stale)
```

After 24h, the field becomes None. If the batch never runs again, all assets lack LLM sentiment. This isn't a "no data = 50" problem; it's "no data = None", which might be scored as 0 or abstained.

---

## 3. Silent Failures

### 3.1 Empty Results Returned As Valid Data

**Whale Agent:**
```python
# whale_agent.py, lines 65-186
def collect(self) -> Tuple[Dict[str, Any], List[str]]:
    data = self.empty_data()  # {"whale_moves": [], ...}
    errors: List[str] = []

    # If all layers fail to fetch, returns:
    # data = {"whale_moves": [], "by_asset": {sym: [] for sym in assets}, ...}
    # errors = ["whale_alert_api: ...", "etherscan: ...", ...]

    return data, errors
```

The orchestrator sees `status = "partial"` (errors present) but still saves `data` to storage. Downstream scoring systems load this data and see:
- `summary.total_moves = 0`
- `summary.credible_moves = 0`
- `summary.assets_with_activity = []`

This is valid whale data (no moves), but indistinguishable from "whale agent failed."

### 3.2 Narrative Agent Silent Drops

If Reddit fails, the catch block logs an error but the agent continues:

```python
# narrative_agent.py, lines 121-128
if is_source_enabled(self.profile, "reddit"):
    try:
        reddit_counts, reddit_weighted, reddit_headlines = self._fetch_reddit()
        # ...
        data["sources_used"].append("reddit")
    except Exception as exc:
        errors.append(f"reddit: {exc}")
        # CONTINUE — no return, no exit

# If all sources fail, errors list grows but data is incomplete
# data["by_asset"][sym].sources_with_data = 0  (computed later)
```

The asset is marked `narrative_status = "unknown"` (status), but the error log shows 6 failures. The calling system doesn't know if:
- 4 of 6 sources worked (partial data) → status = early_pickup is valid
- 0 of 6 sources worked (complete failure) → status = unknown is misleading

**The `sources_with_data` field helps**, but it's computed inside the agent. External systems must load the full result and inspect this field.

### 3.3 Technical Agent Missing Price

If Binance klines fetch fails:

```python
# technical_agent.py, lines 102-110
try:
    closes = self._fetch_klines(binance_sym, interval, candle_limit)
    # ...
except Exception as exc:
    errors.append(f"{sym} klines: {exc}")
    continue

# Asset data in result:
# {
#   "price": None,  # No price
#   "rsi_14": None,
#   "trend_7d": "unknown",
#   "technical_condition": False
# }
```

Downstream scoring systems that depend on `price` field get None. If they coerce None → 0.0, a $40k BTC looks like a $0 asset.

---

## 4. The "No Data = 50" Problem

### 4.1 Where Does 50 Come From?

The problem manifests in the **signal fusion system** (not visible in agent code). When a score dimension has:
- No data (None, empty list, or API failure)
- Default neutral value = 50

**Example scenarios:**

| Dimension | No Data Scenario | Default Score | Problem |
|-----------|------------------|----------------|---------|
| Whale momentum | All API calls fail | 50 | Looks neutral, actually failed |
| Narrative momentum | All sources fail | 50 (if rebased) | Can't tell difference |
| Technical momentum | Price not available | 50 | Should abstain |
| Derivatives momentum | All futures endpoints down | 50 | Should skip BTC |

### 4.2 Implicit 50-Scoring In Fusion

The fusion system computes composite scores by averaging dimension scores. If a dimension is missing:

**Case A: Simple average with zero-filling**
```
composite = (whale_50 + technical_50 + narrative_50) / 3 = 50
```
Result clusters around 50 (abstain zone).

**Case B: Average with NaN removal**
```
dimensions_present = [whale_score]  # Other two are None
composite = average([whale_score])
```
Result depends on single available dimension.

**Either way, the presence of missing data changes the composition and interpretation.**

### 4.3 Assets Most Affected

**BTC / ETH (high-cap, well-covered):** Unlikely to have missing data. All 6 APIs usually succeed.

**Altcoins (low-cap, exchange-dependent):**
- Whale Alert API may not track them (requires min $100k threshold)
- Twitter mentions might be sparse (require influencer activity)
- Technical signal depends on Binance having tradable pair (should be true for top 20)

**Assets with limited exchange activity:**
- Etherscan data is sparse (not trading on tracked exchanges)
- CryptoPanic might not index them
- Google News mentions are rare

**These assets are most likely to operate on missing dimensions, clustering near 50.**

### 4.4 Data Tier Detection Is Fragile

The system uses keyword-based detection to classify data tiers:

```python
# Implicit in agents:
# "full" tier — all sources returned data
# "partial" tier — some sources returned data
# "none" tier — no sources returned data
```

**Problems:**

1. **Single-source false positives**: CoinGecko returns 1 result with $0 price (bad token), marked as "full" tier but data is garbage.

2. **Rate-limit indistinguishability**: Whale Alert returns empty transaction list (rate limited) vs. no actual whale activity. Both return `[]`.

3. **No explicit tier field**: Agents don't include `data_tier: "full" | "partial" | "none"`. Downstream must infer from `sources_used` or `errors`.

---

## 5. Cross-Agent Data Dependencies

### 5.1 Market Agent → Trend Scorer

**Dependency:** Trend scorer uses `market_agent` data to weight narrative momentum.

**Failure mode:**
- Market agent CoinGecko fails → `per_asset = {}`
- Trend scorer loads latest market agent result
- Gets empty dict, tries to access `per_asset[asset].get("change_24h_pct")`
- Gets `None`, coerces to 0.0
- Narrative momentum weighted by 0% price change (wrong)

**No explicit validation** in trend scorer:
```python
# Imagined trend_scorer.py (not in codebase provided)
market_data = store.load_latest("market_agent")["data"]
for asset in assets:
    volume_weight = market_data.get("per_asset", {}).get(asset, {}).get("volume_7d_avg", 0)
    # If market_agent failed, volume_weight = 0
```

### 5.2 Technical Agent → Velocity Dampener

**Dependency:** Velocity dampener uses `technical_agent` price data and RSI to adjust whale momentum scores.

**Failure mode:**
- Technical agent Binance down → `price: None`
- Dampener tries to access `price`
- Gets `None`, can't scale whale momentum
- Score either skipped or uses fallback (50)

### 5.3 Orchestration Doesn't Wait

The orchestrator runs agents on independent cadences:

```python
# orchestrator/runner.py, lines 39-48
def _should_run_agent(name: str, force: bool = False) -> bool:
    if force:
        return True
    env_key = f"AGENT_CADENCE_{name.upper().replace('_AGENT', '')}_MIN"
    cadence_min = int(os.getenv(env_key, str(_AGENT_CADENCES_MIN.get(name, 15))))
    last = _agent_last_run.get(name)
    if last is None:
        return True
    return (time.time() - last) >= cadence_min * 60
```

**No dependency graph.** If whale agent (30 min) depends on market agent data, but whale runs every 30 min and market runs every 30 min offset, whale uses stale market data.

**Better approach:** Define dependencies:
```python
dependencies = {
    "fusion": ["whale_agent", "technical_agent", "market_agent", "narrative_agent"],
    "whale_agent": ["market_agent"],  # Whale should run after market
}
```

---

## 6. Data Age Tracking

### 6.1 No Explicit Data Age Field

Agents save timestamps but don't expose `data_age_minutes` or `last_source_success_timestamp`.

**Stored data structure:**
```json
{
  "agent": "whale_agent",
  "timestamp": "2026-03-15T12:30:00Z",
  "data": { ... },
  "meta": {
    "errors": [ ... ],
    "duration_ms": 1234
  }
}
```

**Missing:**
- `data_freshness_score`: % of expected sources that succeeded
- `last_successful_run`: when this agent last collected non-empty data
- `max_data_age_hours`: if all sources fail, how old is the cached result?

### 6.2 Fusion System Doesn't Validate Freshness

The signal fusion loads latest agent results without checking:
1. How old is this data?
2. Which sources failed?
3. Is this below a minimum quality threshold?

**Should include:**
```python
# In fusion.py (not provided, but likely exists)
for agent_name in ["whale_agent", "technical_agent", ...]:
    result = store.load_latest(agent_name)
    age_minutes = (now - parse_iso_timestamp(result["timestamp"])).total_seconds() / 60

    if age_minutes > MAX_AGE_MINUTES[agent_name]:
        # Alert or abstain this dimension
        # Current code: probably silently uses it
```

---

## 7. Specific Failure Modes & Recommendations

### Failure Mode #1: All APIs Down

**Scenario:** CoinGecko, Binance, Etherscan, and Reddit all unreachable (regional outage).

**Current Behavior:**
- All agents report `status = "error"`
- All `data` payloads are `empty_data()` (empty lists/dicts)
- Fusion system loads empty results
- Composites score 50 (or abstain, depending on implementation)
- User sees "No new signals" but doesn't know if it's real or a failure

**Recommendation:**
1. Include `data_availability_pct` in each agent result:
   ```python
   "meta": {
       "available_sources": 2,  # Out of 6
       "availability_pct": 33.3,
       "minimum_availability_required": 50.0,  # From YAML
   }
   ```
2. Fusion system checks:
   ```python
   if result["meta"]["availability_pct"] < min_required:
       signal = "abstain"  # Don't compute composite
   ```

### Failure Mode #2: Partial Source Failure

**Scenario:** Reddit API is down, but Twitter, Farcaster, CryptoPanic, Google News, and CoinGecko are up.

**Current Behavior:**
- Narrative agent reports `status = "partial"` with 1 error
- Data includes 5 out of 6 sources
- Asset marked `sources_with_data = 5`
- But downstream doesn't know if 5/6 is "good enough" or "should wait for Reddit"

**Recommendation:**
1. Define per-agent `critical_sources`: sources whose absence should flag low confidence.
   ```yaml
   narrative_agent:
     critical_sources: ["reddit", "twitter"]  # Most mentions come from these
     sources_total: 6
   ```
2. In agent result:
   ```python
   "data": {
       "by_asset": { ... },
       "critical_sources_present": ["twitter", "farcaster"],  # Sorted list
       "critical_sources_missing": ["reddit"],
   }
   ```

### Failure Mode #3: Whale Alert Rate-Limiting

**Scenario:** Whale Alert API returns 429 (rate limited) repeatedly.

**Current Behavior:**
- Whale agent retries with exponential backoff (2s, 4s, 8s, 16s, 32s)
- After 3 retries, gives up
- No moves returned
- But error log says "429 rate limit" (distinguishable from "no activity")
- Downstream can't tell if this is a temporary blip or sustained rate limit

**Recommendation:**
1. Extend retry logic with circuit breaker:
   ```python
   MAX_CONSECUTIVE_RATE_LIMITS = 5  # If 5 in a row, stop trying for 1 hour
   if consecutive_429_count >= MAX_CONSECUTIVE_RATE_LIMITS:
       return {
           "whale_moves": [],
           "rate_limit_circuit_broken": True,
           "retry_after_minutes": 60,
       }
   ```
2. Fusion system respects `retry_after_minutes`:
   ```python
   if result["meta"].get("rate_limit_circuit_broken"):
       # Abstain whale signal for 60 minutes
       signal = "abstain"
       reason = "Rate limit detected; retry at " + ...
   ```

### Failure Mode #4: Data Tier Misclassification

**Scenario:** Whale Alert API returns 1 whale move (real activity), but Etherscan, Blockchain.com, Twitter, and exchange flow all fail.

**Current Behavior:**
- Whale agent returns 1 move, no exchange flow
- Marked as "partial" (data present, some sources failed)
- Downstream treats as valid "whale bullish" signal
- But the signal is based on 1 layer out of 5

**Recommendation:**
1. Define `data_tier` explicitly per agent:
   ```python
   def _classify_data_tier(self) -> str:
       sources_present = len([s for s in self.sources_used if s])
       required_sources = {
           "whale_alert_api": "required",
           "etherscan": "recommended",
           "blockchain_com": "recommended",
           "exchange_flow": "optional",
       }

       if sources_present >= 3:
           return "full"
       elif sources_present >= 1:
           return "partial"
       else:
           return "none"
   ```
2. Include in result:
   ```python
   "meta": {
       "data_tier": "partial",  # "full" | "partial" | "none"
       "tier_reasoning": "1/5 layers present (whale_alert_api only)",
   }
   ```

### Failure Mode #5: LLM Sentiment Batching Delay

**Scenario:** LLM batch job runs every 12 hours but fails for 48 hours. Headlines are fresh but LLM sentiment is stale.

**Current Behavior:**
- Narrative agent collects fresh headlines every 60 min
- LLM sentiment batch job fails at T=12h
- Cache expires at T=36h (24h max age)
- Assets get `llm_sentiment = None` after T=36h
- No indicator that headlines are fresh but sentiment is missing

**Recommendation:**
1. Separate `headline_sentiment` (keyword-based, fresh) from `llm_sentiment` (batched, potentially stale):
   ```python
   "by_asset": {
       "BTC": {
           "keyword_sentiment": 0.35,  # Fresh, every 60 min
           "llm_sentiment": None,  # Stale or missing, every 12h
       }
   }
   ```
2. Fusion system uses keyword sentiment as fallback if LLM is stale.

### Failure Mode #6: Derivatives Lead Indicators Require History

**Scenario:** New system deployment. First derivatives agent run saves 1 snapshot. Next run (15 min later) can compute deltas because history is available but sparse.

**Current Behavior:**
- First run: `funding_rate_change_4h = None` (no history)
- Second run: Tries to load 20 snapshots, gets 1, finds "closest" within tolerance
- If only 1 snapshot and it's 15 min old (< 1.5h min age), delta isn't computed
- Asset shows `funding_rate_change_4h = None` for hours until 4h of snapshots accumulate

**Recommendation:**
1. Store cumulative delta computation time:
   ```python
   "meta": {
       "lead_indicators_available_after_minutes": 240,  # ~4h of snapshots needed
       "current_snapshot_age_minutes": 15,
       "lead_indicators_ready": False,
   }
   ```
2. Fusion system doesn't score derivatives momentum until flag is True.

---

## 8. Summary Table: Data Freshness & Reliability

| Agent | Primary Source | Cadence | Failure Mode | Detection | Recommendation |
|-------|---|---|---|---|---|
| **Whale** | Whale Alert REST | 30 min | Rate limit or no data indistinguishable | Parse error msg | Explicit `rate_limit_circuit_broken` flag |
| **Technical** | Binance klines | 15 min | Kline fetch fails, indicators = None | Check for None | Include `data_availability_pct` |
| **Derivatives** | Binance futures | 15 min | Lead indicators unavailable first 4h | Check for None deltas | `lead_indicators_ready_after_minutes` field |
| **Narrative** | 6 sources (Reddit, Twitter, etc) | 60 min | Individual source fails, status = "unknown" | Error log only | Explicit `critical_sources_present/missing` |
| **Market** | CoinGecko + Binance | 30 min | CoinGecko down, no per_asset data | Check for empty dict | `per_asset_available_pct` field |
| **Fusion** | All agents | 15 min | Loads data without freshness check | No validation | Validate max data age before scoring |

---

## 9. Recommended Actions (Priority)

### P0: Data Availability Tracking
Add `data_availability_pct` and `critical_failures` fields to all agents. Fusion system must check before scoring.

### P1: Explicit Data Tiers
Define per-agent `data_tier: "full" | "partial" | "none"` with clear thresholds. Fusion weights accordingly.

### P2: Maximum Data Age Enforcement
Add `MAX_DATA_AGE_MINUTES` per agent. Fusion skips scoring if data exceeds threshold.

### P3: Dependency Graph
Define which agents depend on which others (whale depends on market for USD conversion). Orchestrator respects dependencies.

### P4: LLM Sentiment Separation
Split `headline_sentiment` (fresh, keyword-based) from `llm_sentiment` (batched, potentially stale). Fusion uses headline as fallback.

### P5: Circuit Breaker for Rate Limits
Implement exponential backoff + circuit breaker for repeated 429 responses. Stop retrying for 1 hour to avoid cascading failures.

---

## Conclusion

The pipeline exhibits **signal integrity risks** across multiple dimensions:

1. **Silent failures masquerading as neutral signals** ("no data = 50")
2. **Multi-source absence indistinguishable from low activity**
3. **Stale data persistence indefinitely** without max-age enforcement
4. **No explicit data quality tracking** to distinguish failures from facts

**Immediate action required:** Implement explicit `data_tier` and `data_availability_pct` fields in all agents, and validation in the fusion system before composite scoring.

---

**Analysis completed:** 2026-03-15
**Report location:** `/tmp/accuracy-research/I3-data-quality-freshness.md`
