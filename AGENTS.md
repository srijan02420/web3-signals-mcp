# Web3 Signals Agent

## Identity
- **Name**: Web3 Signals Agent
- **Description**: Multi-agent crypto market intelligence system. Five specialized AI agents analyze whale activity, derivatives positioning, technical indicators, narrative momentum, and market data — fused into scored signals for 20 crypto assets with regime-aware scoring and LLM insights.
- **Version**: 0.3.0
- **Provider**: Web3 Signals

## Architecture
Web3 Signals is a **multi-agent market intelligence system** that fuses 5 independent data dimensions into actionable scored signals (0-100).

### Signal Generation (every 15 minutes)
1. **5 specialized agents** independently score each asset (0-100) with **continuous proportional scoring** — no flat buckets, indicators like RSI, funding rate, and F&G produce proportional scores across their full range
2. **Direction-aware asymmetric weighting** — different weight sets for bullish vs bearish leans, based on per-dimension accuracy data
3. **Direction gating** — zero out dimensions with historically bad accuracy in specific directions (e.g. whale data gated in bullish direction due to 16-27% accuracy)
4. **Dynamic data reweighting** — agents with missing/partial data get reduced weight, redistributed to agents with full data
5. **Velocity overlay** — computes rate-of-change of RSI, MACD, F&G, and funding rate across 1h/4h/24h windows. When indicators are accelerating against the signal (e.g. RSI still falling while system says BUY), dampens the signal by 30-70%. Prevents premature contrarian calls.
6. **Trend override** — in confirmed BTC downtrends (price >5% below 30D MA), contrarian boost on market/derivatives is dampened by 30%, allowing bearish signals to emerge
7. **Dynamic abstain zone** — threshold adjusts based on Fear & Greed: extreme conditions narrow the band (more signals), neutral markets widen it (fewer, better signals)

### Scoring Philosophy
**Contrarian / mean-reversion**: Fear = buying opportunity, greed = danger. When the crowd panics, the system looks for value. When euphoria peaks, it signals caution.

### Performance Tracking
- **Gradient accuracy scoring** (0.0-1.0) — not binary hit/miss
- **24h and 48h evaluation windows**
- Abstained (neutral) signals are NOT counted — only directional calls
- Rolling 30-day accuracy with per-asset and per-timeframe breakdowns

## Protocols
- **REST API**: OpenAPI-documented endpoints at /docs
- **MCP**: Model Context Protocol server (SSE transport at /mcp/sse)
- **A2A**: Agent-to-Agent discovery card at /.well-known/agent.json
- **x402**: HTTP 402 micropayments on Base mainnet (USDC) — payment is auth, no API keys needed

## Endpoints
| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| /signal | GET | All 20 asset signals with portfolio summary + LLM insights | x402 $0.001 |
| /signal/{asset} | GET | Single asset signal (e.g. /signal/BTC) | x402 $0.001 |
| /performance/reputation | GET | 30-day rolling accuracy score with methodology | x402 $0.001 |
| /performance/{asset} | GET | Per-asset accuracy breakdown | Free |
| /health | GET | Agent status, data freshness, uptime | Free |
| /dashboard | GET | Live signal intelligence dashboard (browser) | Free |
| /analytics | GET | API usage analytics — request trends, client types | Free |
| /api/history | GET | Historical signal runs (paginated) | Free |
| /docs | GET | OpenAPI interactive documentation | Free |

## Assets Covered
BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOT, MATIC, LINK, UNI, ATOM, LTC, FIL, NEAR, APT, ARB, OP, INJ, SUI

## Data Sources
1. **Whale tracking** — on-chain accumulate/sell ratio, exchange flows, whale wallet signals
2. **Technical analysis** — RSI, MACD, moving averages (7D/30D) via Binance
3. **Derivatives positioning** — funding rate, open interest, long/short ratio via Binance Futures
4. **Narrative momentum** — Reddit, News, CoinGecko trending, LLM sentiment analysis (12h cycle)
5. **Market data** — price, volume, Fear & Greed Index, BTC dominance via CoinGecko

## Update Frequency
- Signals refresh every **15 minutes**
- LLM sentiment analysis every **12 hours**
- Performance evaluation every **4 hours**

## Pricing
- **$0.001/call** USDC on Base mainnet via x402 protocol
- Payment IS authentication — no API keys, no signup, no accounts
- Free endpoints: /health, /dashboard, /analytics, /docs
- Facilitator: CDP (Coinbase Developer Platform)

## Integration Examples

### For AI Agents (x402)
```
GET /signal
Header: X-PAYMENT (x402 payment receipt)
```
The x402 protocol returns HTTP 402 with payment instructions. Your agent pays $0.001 USDC on Base and retries with the payment receipt.

### For MCP Clients
Connect to `/mcp/sse` for real-time tool access from Claude, GPT, or any MCP-compatible agent.

### For REST Clients
Free dashboard at `/dashboard`. API docs at `/docs`.

## Contact
- API Docs: /docs
- Dashboard: /dashboard
