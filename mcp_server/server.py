"""
Web3 Signals — MCP Server (Intelligent Edition)

Exposes crypto signal intelligence as MCP tools for Claude Desktop, Cursor,
and other MCP-compatible AI assistants.

Tools:
    get_market_briefing   Executive summary — regime, risk, top movers, actionable calls
    get_all_signals       Full fusion: portfolio + 20 scored signals + LLM insights
    get_asset_signal      Single asset signal with 6-dimension breakdown
    compare_assets        Side-by-side comparison of 2-5 assets
    get_health            Agent status, last run, uptime
    get_performance       Signal accuracy tracking (30-day rolling)
    get_asset_performance Per-asset accuracy breakdown
    get_analytics         API usage analytics — requests, clients, trends
    get_x402_stats        x402 micropayment analytics — revenue, paid calls, conversion

Run:
    python -m mcp_server.server          # stdio mode (default)
    python -m mcp_server.server --sse    # SSE mode for remote connections
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

# Add project root to path so we can import shared modules
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.storage import Storage
from signal_fusion.engine import SignalFusion

# ---------------------------------------------------------------------------
# MCP Server setup
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Web3 Signals — AgentMarketSignal",
    instructions=(
        "AgentMarketSignal: Production AI-powered crypto signal intelligence. "
        "6 specialized AI agents analyze 20 assets every 15 minutes across "
        "whale activity, derivatives positioning, technical analysis, narrative "
        "momentum, market sentiment, and trend direction. Signals are fused "
        "into composite scores (0-100) with regime-aware weighting and "
        "accuracy-scaled dimension influence.\n\n"
        "QUICK START:\n"
        "- Market overview? → get_market_briefing (executive summary with actionable calls)\n"
        "- Specific asset? → get_asset_signal('BTC') (6-dimension breakdown)\n"
        "- Compare assets? → compare_assets('BTC,ETH,SOL') (side-by-side)\n"
        "- Full raw data? → get_all_signals (complete JSON for all 20 assets)\n"
        "- How accurate? → get_performance (30-day rolling accuracy)\n"
        "- Who's using this? → get_analytics (request stats, client types)\n"
        "- Payment revenue? → get_x402_stats (micropayment analytics)\n"
        "- System healthy? → get_health (agent status, uptime)\n\n"
        "SCORING: 0-30 = bearish, 30-40 = weak sell, 40-60 = neutral, "
        "60-70 = weak buy, 70-100 = bullish. Signals below 38 or above 62 "
        "are high-conviction (outside abstain zone).\n\n"
        "DATA FRESHNESS: Signals update every 15 min. Narrative LLM analysis "
        "runs every 12 hours. Accuracy uses 30-day rolling gradient scoring."
    ),
)

# Globals (lazy-initialized on first tool call)
_store: Storage | None = None
_fusion: SignalFusion | None = None


def _get_store() -> Storage:
    global _store
    if _store is None:
        _store = Storage()
    return _store


def _get_fusion() -> SignalFusion:
    global _fusion
    if _fusion is None:
        _fusion = SignalFusion()
    return _fusion


# ---------------------------------------------------------------------------
# Tool: get_market_briefing — Executive summary with actionable intelligence
# ---------------------------------------------------------------------------
@mcp.tool(annotations={"readOnlyHint": True})
def get_market_briefing() -> str:
    """Get executive market intelligence briefing with top buy/sell signals and regime context. Returns market regime (TRENDING/RANGING), risk level, top 3 bullish and bearish signals ranked by composite score, high-conviction positions outside the 38-62 abstain zone, and regime-specific trading context. Best starting point for portfolio decisions. For raw per-asset data, use get_all_signals."""
    fusion = _get_fusion()
    result = fusion.fuse()

    portfolio = result.get("data", {}).get("portfolio_summary", {})
    signals = result.get("data", {}).get("signals", {})
    timestamp = result.get("timestamp", "unknown")

    # Sort assets by score
    scored = []
    for asset, sig in signals.items():
        score = sig.get("composite_score", 50)
        scored.append((asset, score, sig))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Top buys (highest scores) and sells (lowest scores)
    top_buys = scored[:3]
    top_sells = scored[-3:]
    top_sells.reverse()  # lowest first

    # High conviction signals
    high_conviction_bullish = [
        (a, s, sig) for a, s, sig in scored if s >= 62
    ]
    high_conviction_bearish = [
        (a, s, sig) for a, s, sig in scored if s <= 38
    ]
    neutral_count = len([1 for _, s, _ in scored if 38 < s < 62])

    def _signal_summary(asset, score, sig):
        """Build a concise summary for one asset."""
        direction = sig.get("direction", "?")
        label = sig.get("label", "?")
        dims = sig.get("dimensions", {})
        # Find strongest dimension
        dim_scores = {}
        for dim_name, dim_data in dims.items():
            if isinstance(dim_data, dict):
                dim_scores[dim_name] = dim_data.get("score", 50)
        if dim_scores:
            strongest_dim = max(dim_scores, key=dim_scores.get)
            weakest_dim = min(dim_scores, key=dim_scores.get)
        else:
            strongest_dim = weakest_dim = "unknown"
        return {
            "asset": asset,
            "score": score,
            "direction": direction,
            "label": label,
            "strongest_dimension": strongest_dim,
            "weakest_dimension": weakest_dim,
        }

    regime = portfolio.get("market_regime", "unknown")
    risk = portfolio.get("risk_level", "unknown")

    briefing = {
        "briefing_type": "market_intelligence",
        "timestamp": timestamp,
        "market_regime": regime,
        "risk_level": risk,
        "signal_momentum": portfolio.get("signal_momentum", "unknown"),
        "regime_context": (
            f"Market is in {regime} regime with {risk} risk. "
            + ("Trend-following signals are amplified. " if regime == "TRENDING" else "")
            + ("Mean-reversion plays are favored. " if regime == "RANGING" else "")
            + f"{len(high_conviction_bullish)} bullish and "
            f"{len(high_conviction_bearish)} bearish high-conviction signals. "
            f"{neutral_count} assets in neutral/abstain zone."
        ),
        "top_buys": [_signal_summary(*x) for x in top_buys],
        "top_sells": [_signal_summary(*x) for x in top_sells],
        "high_conviction_count": {
            "bullish": len(high_conviction_bullish),
            "bearish": len(high_conviction_bearish),
            "neutral": neutral_count,
        },
        "total_assets_tracked": len(signals),
        "data_freshness": timestamp,
    }

    return json.dumps(briefing, indent=2)


# ---------------------------------------------------------------------------
# Tool: get_all_signals
# ---------------------------------------------------------------------------
@mcp.tool(annotations={"readOnlyHint": True})
def get_all_signals() -> str:
    """Export complete signal fusion for all 20 crypto assets (BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOT, MATIC, LINK, UNI, ATOM, LTC, FIL, NEAR, APT, ARB, OP, INJ, SUI). Returns portfolio_summary (regime, risk, top_buys, top_sells), per-asset composite_score (0-100) with 6-dimension breakdown (whale/technical/derivatives/narrative/market/trend), momentum deltas, and LLM cross-dimensional insights. Refreshes every 15 minutes. Use get_asset_signal for single-asset queries."""
    fusion = _get_fusion()
    result = fusion.fuse()
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool: get_asset_signal
# ---------------------------------------------------------------------------
VALID_ASSETS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOT",
    "MATIC", "LINK", "UNI", "ATOM", "LTC", "FIL", "NEAR", "APT",
    "ARB", "OP", "INJ", "SUI",
]


@mcp.tool(annotations={"readOnlyHint": True})
def get_asset_signal(asset: str) -> str:
    """Get composite signal for a single cryptocurrency. Takes one ticker (case-insensitive) and returns composite_score (0-100), direction label (STRONG BUY to STRONG SELL), all 6 dimension scores with labels, momentum vs previous cycle, and market context (regime/risk). Valid: BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOT, MATIC, LINK, UNI, ATOM, LTC, FIL, NEAR, APT, ARB, OP, INJ, SUI. Invalid tickers return error with valid list."""
    asset = asset.upper().strip()
    if asset not in VALID_ASSETS:
        return json.dumps({
            "error": f"Invalid asset '{asset}'. Valid: {VALID_ASSETS}"
        })

    fusion = _get_fusion()
    result = fusion.fuse()

    signals = result.get("data", {}).get("signals", {})
    sig = signals.get(asset, {})
    portfolio = result.get("data", {}).get("portfolio_summary", {})

    return json.dumps({
        "asset": asset,
        "timestamp": result.get("timestamp"),
        "signal": sig,
        "market_context": {
            "regime": portfolio.get("market_regime"),
            "risk_level": portfolio.get("risk_level"),
            "signal_momentum": portfolio.get("signal_momentum"),
        },
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: compare_assets — Side-by-side comparison
# ---------------------------------------------------------------------------
@mcp.tool(annotations={"readOnlyHint": True})
def compare_assets(assets: str) -> str:
    """Rank and compare 2-5 crypto assets side-by-side. Input: comma-separated tickers (e.g. "BTC,ETH,SOL"). Returns ranked comparison with composite_score, direction, all 6 dimensions, momentum, and verdict (strongest/weakest). Sorted by score descending. Use for portfolio rebalancing or multi-asset conviction comparison. Constraints: 2-5 assets; invalid tickers return error with valid list."""
    asset_list = [a.strip().upper() for a in assets.split(",") if a.strip()]
    if len(asset_list) < 2:
        return json.dumps({"error": "Need at least 2 assets to compare. Example: 'BTC,ETH,SOL'"})
    if len(asset_list) > 5:
        return json.dumps({"error": "Maximum 5 assets per comparison."})

    invalid = [a for a in asset_list if a not in VALID_ASSETS]
    if invalid:
        return json.dumps({"error": f"Invalid assets: {invalid}. Valid: {VALID_ASSETS}"})

    fusion = _get_fusion()
    result = fusion.fuse()
    signals = result.get("data", {}).get("signals", {})
    portfolio = result.get("data", {}).get("portfolio_summary", {})

    comparison = []
    for asset in asset_list:
        sig = signals.get(asset, {})
        score = sig.get("composite_score", 50)
        dims = sig.get("dimensions", {})
        dim_summary = {}
        for dim_name, dim_data in dims.items():
            if isinstance(dim_data, dict):
                dim_summary[dim_name] = {
                    "score": dim_data.get("score", 50),
                    "label": dim_data.get("label", "?"),
                }
        comparison.append({
            "asset": asset,
            "composite_score": score,
            "direction": sig.get("direction", "?"),
            "label": sig.get("label", "?"),
            "dimensions": dim_summary,
            "momentum": sig.get("momentum", {}),
            "confidence": sig.get("confidence", {}),
        })

    # Rank by score
    comparison.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, c in enumerate(comparison):
        c["rank"] = i + 1

    return json.dumps({
        "comparison": comparison,
        "market_context": {
            "regime": portfolio.get("market_regime"),
            "risk_level": portfolio.get("risk_level"),
        },
        "timestamp": result.get("timestamp"),
        "verdict": (
            f"Strongest: {comparison[0]['asset']} ({comparison[0]['composite_score']}/100 — "
            f"{comparison[0]['direction']}). "
            f"Weakest: {comparison[-1]['asset']} ({comparison[-1]['composite_score']}/100 — "
            f"{comparison[-1]['direction']})."
        ),
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: get_health
# ---------------------------------------------------------------------------
@mcp.tool(annotations={"readOnlyHint": True})
def get_health() -> str:
    """Check real-time health of all 5 signal agents and fusion pipeline. Returns per-agent (whale, technical, derivatives, narrative, market) status, last_run timestamp, duration_ms, error count, plus fusion status and storage backend type. Use to diagnose pipeline issues, verify data freshness, or detect stuck agents."""
    store = _get_store()
    agent_names = [
        "technical_agent", "derivatives_agent", "market_agent",
        "narrative_agent", "whale_agent",
    ]
    agent_status: dict[str, Any] = {}

    for name in agent_names:
        latest = store.load_latest(name)
        if latest:
            agent_status[name] = {
                "status": latest.get("status", "unknown"),
                "last_run": latest.get("timestamp"),
                "duration_ms": latest.get("meta", {}).get("duration_ms"),
                "errors": len(latest.get("meta", {}).get("errors", [])),
            }
        else:
            agent_status[name] = {"status": "no_data", "last_run": None}

    fusion_latest = store.load_latest("signal_fusion")
    fusion_status = {
        "status": fusion_latest.get("status") if fusion_latest else "no_data",
        "last_run": fusion_latest.get("timestamp") if fusion_latest else None,
    }

    return json.dumps({
        "status": "healthy",
        "storage_backend": store.backend,
        "agents": agent_status,
        "fusion": fusion_status,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: get_performance
# ---------------------------------------------------------------------------
@mcp.tool(annotations={"readOnlyHint": True})
def get_performance() -> str:
    """Get rolling 30-day signal accuracy and reputation score. Returns reputation_score (0-100), accuracy_30d (%), signals_evaluated, breakdown by timeframe (24h/48h) and per-asset, plus scoring methodology. If <24h of history: returns status="collecting_data". Use to assess signal reliability and track improvement over time."""
    store = _get_store()

    stats = store.load_accuracy_stats(days=30)
    total_snapshots = store.count_snapshots(days=30)

    if stats["total"] == 0:
        return json.dumps({
            "status": "collecting_data",
            "message": "Performance tracking is active. Accuracy data will appear after 24h of signal history.",
            "snapshots_collected": total_snapshots,
        })

    accuracy = round(stats["hits"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0

    return json.dumps({
        "status": "active",
        "reputation_score": int(round(accuracy)),
        "accuracy_30d": accuracy,
        "signals_evaluated": stats["total"],
        "signals_correct": stats["hits"],
        "by_timeframe": stats["by_timeframe"],
        "by_asset": stats["by_asset"],
        "snapshots_collected_30d": total_snapshots,
        "methodology": {
            "direction_extraction": "score >60 = bullish, <40 = bearish, 40-60 = neutral",
            "neutral_threshold": "price move <=2% = correct for neutral signals",
            "scoring": "binary (hit/miss)",
            "window": "30-day rolling",
            "timeframes": ["24h", "48h"],
            "price_source": "CoinGecko",
        },
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: get_asset_performance
# ---------------------------------------------------------------------------
@mcp.tool(annotations={"readOnlyHint": True})
def get_asset_performance(asset: str) -> str:
    """Get accuracy metrics for a specific asset over the rolling 30-day window. Takes one ticker (case-insensitive). Returns asset accuracy_30d (%), overall_accuracy_30d, reputation_score, and last_updated. If no data: returns status="collecting_data". Invalid asset returns error with valid list. Use to identify which assets have strongest signal accuracy."""
    asset = asset.upper().strip()
    if asset not in VALID_ASSETS:
        return json.dumps({
            "error": f"Invalid asset '{asset}'. Valid: {VALID_ASSETS}"
        })

    store = _get_store()
    stats = store.load_accuracy_stats(days=30)

    if stats["total"] == 0:
        return json.dumps({
            "status": "collecting_data",
            "message": "Performance tracking is active. Check back after 24h.",
        })

    asset_accuracy = stats["by_asset"].get(asset)
    if asset_accuracy is None:
        return json.dumps({"error": f"No accuracy data for '{asset}'"})

    overall = round(stats["hits"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0

    return json.dumps({
        "asset": asset,
        "accuracy_30d": asset_accuracy,
        "overall_accuracy_30d": overall,
        "reputation_score": int(round(overall)),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: get_analytics — API usage analytics
# ---------------------------------------------------------------------------
@mcp.tool(annotations={"readOnlyHint": True})
def get_analytics(days: int = 7) -> str:
    """Monitor API usage patterns over configurable window (1-90 days, default 7). Returns total_requests, unique_clients, avg_response_ms, breakdowns by endpoint and client_type (Claude, OpenAI, Python, browser), requests_per_day, and top_user_agents. Use for adoption monitoring, AI agent pattern analysis, or performance regression detection."""
    if days < 1:
        days = 1
    if days > 90:
        days = 90

    store = _get_store()
    stats = store.load_api_analytics(days=days)

    return json.dumps({
        "window_days": days,
        "total_requests": stats["total_requests"],
        "unique_clients": stats["unique_ips"],
        "avg_response_ms": stats["avg_duration_ms"],
        "by_endpoint": stats["by_endpoint"],
        "by_client_type": stats["by_user_agent_type"],
        "requests_per_day": stats["requests_per_day"],
        "top_user_agents": stats["top_user_agents"][:10],
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: get_x402_stats — x402 micropayment analytics
# ---------------------------------------------------------------------------
@mcp.tool(annotations={"readOnlyHint": True})
def get_x402_stats(days: int = 30) -> str:
    """Get x402 HTTP 402 micropayment analytics (1-90 days, default 30). Returns total_paid_calls, estimated_revenue_usdc ($0.001/call on Base L2), total_402_challenges, payment_failures, conversion_rate_pct, breakdown by endpoint and client_type, paid_per_day timeline, avg_paid_latency_ms. x402 is Coinbase's HTTP micropayment protocol for agent-to-agent commerce."""
    if days < 1:
        days = 1
    if days > 90:
        days = 90

    store = _get_store()
    stats = store.load_x402_analytics(days=days)
    total_challenges = stats["total_402_challenges"]
    total_paid = stats["total_paid_calls"]
    conversion = (
        round(total_paid / total_challenges * 100, 1)
        if total_challenges > 0 else 0
    )

    return json.dumps({
        "window_days": days,
        "price_per_call": "$0.001 USDC",
        "network": "Base (eip155:8453)",
        "total_paid_calls": total_paid,
        "estimated_revenue_usdc": stats["estimated_revenue_usdc"],
        "total_402_challenges": total_challenges,
        "total_payment_failures": stats["total_payment_failures"],
        "conversion_rate_pct": conversion,
        "by_endpoint": stats["by_endpoint"],
        "by_client_type": stats["by_client_type"],
        "paid_per_day": stats["paid_per_day"],
        "avg_paid_latency_ms": stats["avg_paid_latency_ms"],
    }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Run the MCP server."""
    transport = "stdio"
    if "--sse" in sys.argv:
        transport = "sse"

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
