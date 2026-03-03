"""
Velocity (rate-of-change) analyzer for signal fusion.

Computes how fast key indicators are moving by comparing current values
to historical agent runs. When indicators are accelerating in one direction,
dampens the contrarian signal. When stabilizing or reversing, preserves it.

Problem solved: In sustained fear (F&G=14), the contrarian system screams BUY
on 18/20 assets. But RSI was 50 → 42 → 35 across runs — accelerating downward.
Buying into accelerating fear scores 0.0 on 24h/48h evaluation.

Integration: Phase 4.5 — after composite is built, before abstain check.
Dampens the composite's distance from 50 by a factor of 0.3 to 1.0.

All configuration lives in YAML (velocity section).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VelocityAnalyzer:
    """
    Computes rate-of-change for key indicators across historical agent runs.

    When indicators are accelerating in one direction (e.g. RSI falling fast),
    dampens contrarian signals. When indicators stabilize or reverse,
    preserves the contrarian signal.
    """

    def __init__(self, store: Any, profile: Dict[str, Any]) -> None:
        self.store = store
        self.cfg = profile.get("velocity", {})
        self.enabled = self.cfg.get("enabled", False)
        self.agent_names = profile.get("agent_names", {})
        self._history_cache: Dict[str, List[Dict[str, Any]]] = {}

    def is_enabled(self) -> bool:
        return self.enabled

    def preload_history(self) -> List[str]:
        """
        Load historical data for all needed agents. Called once before asset loop.
        Returns list of debug/error messages.
        """
        errors: List[str] = []
        if not self.enabled:
            return errors

        lookback_days = int(self.cfg.get("lookback_days", 1))
        indicators = self.cfg.get("indicators", {})

        # Determine which agents we need history for
        needed_agents = set()
        for ind_cfg in indicators.values():
            needed_agents.add(ind_cfg.get("agent", ""))

        for role in needed_agents:
            if not role:
                continue
            agent_name = self.agent_names.get(role, f"{role}_agent")
            try:
                history = self.store.load_recent(agent_name, days=lookback_days)
                self._history_cache[role] = history
                errors.append(
                    f"velocity: loaded {len(history)} runs for {role} "
                    f"(lookback={lookback_days}d)"
                )
            except Exception as exc:
                errors.append(f"velocity: failed to load {role} history: {exc}")
                self._history_cache[role] = []

        return errors

    def compute_asset_velocity(
        self, asset: str, composite: float
    ) -> Optional[Dict[str, Any]]:
        """
        Compute velocity-based dampening for a single asset.

        Args:
            asset: Asset symbol (e.g. "BTC")
            composite: Current composite score (0-100)

        Returns:
            Dict with dampening_factor, velocities, data_points — or None
            if insufficient data.
        """
        if not self.enabled:
            return None

        min_data_points = int(self.cfg.get("min_data_points", 8))
        indicators_cfg = self.cfg.get("indicators", {})
        windows_cfg = self.cfg.get("windows", {"short": 4, "medium": 16, "long": 96})
        window_weights = self.cfg.get(
            "window_weights", {"short": 0.50, "medium": 0.30, "long": 0.20}
        )

        indicator_velocities: Dict[str, Dict[str, Any]] = {}

        for ind_name, ind_cfg in indicators_cfg.items():
            agent_role = ind_cfg.get("agent", "")
            field_path = ind_cfg.get("field", "")
            is_global = ind_cfg.get("global", False)
            invert = ind_cfg.get("invert", False)

            if not agent_role or not field_path:
                continue

            history = self._history_cache.get(agent_role, [])
            if not history:
                continue

            # Extract time series for this indicator
            if is_global:
                # Global indicator (e.g. F&G) — same for all assets
                series = self._extract_global_series(history, field_path)
            else:
                # Per-asset indicator
                series = self._extract_asset_series(history, asset, field_path)

            if len(series) < min_data_points:
                continue

            # Compute multi-window velocity
            velocity_result = self._compute_multi_window_velocity(
                series, windows_cfg, window_weights
            )

            if velocity_result is not None:
                if invert:
                    velocity_result["weighted_velocity"] *= -1.0
                    for wname in velocity_result.get("per_window", {}):
                        velocity_result["per_window"][wname] *= -1.0

                indicator_velocities[ind_name] = {
                    **velocity_result,
                    "weight": float(ind_cfg.get("weight", 0.1)),
                    "threshold": float(ind_cfg.get("threshold", 5.0)),
                }

        if not indicator_velocities:
            return None

        # Compute dampening factor from all indicator velocities
        dampening = self._compute_dampening(composite, indicator_velocities)

        total_points = sum(
            v.get("data_points", 0) for v in indicator_velocities.values()
        )

        return {
            "dampening_factor": dampening,
            "velocities": {
                k: {
                    "weighted_velocity": round(v["weighted_velocity"], 3),
                    "data_points": v.get("data_points", 0),
                    "signal": self._classify_velocity(
                        v["weighted_velocity"], v["threshold"]
                    ),
                }
                for k, v in indicator_velocities.items()
            },
            "data_points": total_points,
        }

    # ================================================================ #
    #  Internal helpers
    # ================================================================ #

    def _extract_asset_series(
        self, history: List[Dict[str, Any]], asset: str, field_path: str
    ) -> List[float]:
        """
        Extract a time series for a per-asset indicator.
        History is ordered newest-first.
        Returns list of float values (newest first).
        """
        series: List[float] = []
        for snapshot in history:
            data = snapshot.get("data", {})
            # Navigate: by_asset.{ASSET}.{field}
            by_asset = data.get("by_asset", {})
            asset_data = by_asset.get(asset, {})
            value = asset_data.get(field_path)
            if value is not None:
                try:
                    series.append(float(value))
                except (ValueError, TypeError):
                    continue
        return series

    def _extract_global_series(
        self, history: List[Dict[str, Any]], field_path: str
    ) -> List[float]:
        """
        Extract a time series for a global indicator (e.g. sentiment.fear_greed_index).
        History is ordered newest-first.
        """
        series: List[float] = []
        parts = field_path.split(".")
        for snapshot in history:
            data = snapshot.get("data", {})
            value = data
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            if value is not None:
                try:
                    series.append(float(value))
                except (ValueError, TypeError):
                    continue
        return series

    def _compute_multi_window_velocity(
        self,
        series: List[float],
        windows: Dict[str, int],
        window_weights: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """
        Compute velocity across multiple time windows.
        Series is ordered newest-first.

        Returns weighted velocity and per-window details, or None if insufficient data.
        """
        per_window: Dict[str, float] = {}

        for window_name, n_points in windows.items():
            n_points = int(n_points)
            if len(series) < n_points or n_points < 2:
                continue

            newest = series[0]
            oldest = series[n_points - 1]

            # Percentage change from oldest to newest
            if abs(oldest) > 1e-10:
                pct_change = (newest - oldest) / abs(oldest) * 100.0
            elif abs(newest) > 1e-10:
                # oldest is ~0 but newest isn't — large change
                pct_change = 100.0 if newest > 0 else -100.0
            else:
                pct_change = 0.0

            per_window[window_name] = pct_change

        if not per_window:
            return None

        # Weighted average across windows (recent weighted more)
        weighted_sum = 0.0
        total_weight = 0.0
        for wname, wweight in window_weights.items():
            if wname in per_window:
                weighted_sum += per_window[wname] * float(wweight)
                total_weight += float(wweight)

        weighted_velocity = weighted_sum / total_weight if total_weight > 0 else 0.0

        return {
            "weighted_velocity": weighted_velocity,
            "per_window": per_window,
            "data_points": len(series),
        }

    def _compute_dampening(
        self,
        composite: float,
        indicator_velocities: Dict[str, Dict[str, Any]],
    ) -> float:
        """
        Compute dampening factor (0.3 to 1.0) based on velocity vs signal direction.

        Logic:
        - Bullish composite (>50) + negative velocity (indicators falling) = dampen
        - Bullish composite + positive velocity (stabilizing) = preserve
        - Bearish composite (<50) + positive velocity (indicators rising) = dampen
        - Bearish composite + negative velocity (topping out) = preserve
        """
        min_damp = float(self.cfg.get("min_dampening_factor", 0.3))
        max_damp = float(self.cfg.get("max_dampening_factor", 1.0))

        is_bullish = composite > 50.0

        # Compute per-indicator contradiction scores
        # -1.0 = strongly contradicting signal, +1.0 = confirming signal
        contradiction_scores: List[Tuple[float, float]] = []  # (score, weight)

        for ind_name, vel_data in indicator_velocities.items():
            wv = vel_data["weighted_velocity"]
            threshold = vel_data["threshold"]
            weight = vel_data["weight"]

            if threshold <= 0:
                threshold = 1.0

            if is_bullish:
                # Bullish signal: negative velocity = still dropping = contradicts
                if wv < -threshold:
                    cscore = -1.0
                elif wv < 0:
                    cscore = -abs(wv) / threshold
                elif wv > threshold:
                    cscore = 1.0  # confirming reversal
                else:
                    cscore = wv / threshold
            else:
                # Bearish signal: positive velocity = still rising = contradicts
                if wv > threshold:
                    cscore = -1.0
                elif wv > 0:
                    cscore = -wv / threshold
                elif wv < -threshold:
                    cscore = 1.0  # confirming reversal
                else:
                    cscore = -wv / threshold

            contradiction_scores.append((cscore, weight))

        if not contradiction_scores:
            return max_damp

        # Weighted average of contradiction scores
        total_weight = sum(w for _, w in contradiction_scores)
        if total_weight <= 0:
            return max_damp

        avg_contradiction = sum(s * w for s, w in contradiction_scores) / total_weight

        # Map avg_contradiction [-1, 1] to dampening factor [min_damp, max_damp]
        if avg_contradiction >= 0:
            # Velocity confirms or is neutral → no dampening
            return max_damp
        else:
            # Velocity contradicts → dampen proportionally
            # avg_contradiction = -1.0 → min_damp
            # avg_contradiction = 0.0 → max_damp
            return max(min_damp, max_damp + avg_contradiction * (max_damp - min_damp))

    @staticmethod
    def _classify_velocity(velocity: float, threshold: float) -> str:
        """Classify velocity into a human-readable signal."""
        if abs(velocity) < threshold * 0.3:
            return "stable"
        elif velocity > threshold:
            return "accelerating_up"
        elif velocity > 0:
            return "rising"
        elif velocity < -threshold:
            return "accelerating_down"
        else:
            return "declining"
