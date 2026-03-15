"""
Platt Scaling Calibrator for Signal Fusion (Phase D1).

Transforms raw 0-100 dimension scores into calibrated probabilities
using logistic regression (Platt scaling). Each dimension gets its own
calibrator fitted on historical (score, actual_return) pairs.

Usage:
    calibrator = SignalCalibrator()
    calibrator.fit_from_file("/tmp/calibration_data.json")
    prob = calibrator.calibrate("derivatives", 38.5)  # → 0.72 (P(correct direction))
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


class SignalCalibrator:
    """
    Platt scaling calibrator for dimension scores.

    For each dimension, learns parameters (A, B) such that:
        P(price moves in predicted direction | raw_score) = sigmoid(A * score + B)

    The raw score is the distance from center (0-50 scale, where 50 = maximum conviction).
    """

    def __init__(self) -> None:
        # {dimension_name: (A, B)} — Platt parameters
        self.params: Dict[str, Tuple[float, float]] = {}
        # Metadata about fit quality
        self.fit_stats: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Core Platt scaling
    # ------------------------------------------------------------------ #

    @staticmethod
    def sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            ez = math.exp(x)
            return ez / (1.0 + ez)

    def calibrate(self, dimension: str, raw_score: float) -> float:
        """
        Calibrate a raw dimension score (0-100) to a probability.

        Returns P(signal is directionally correct | raw_score).
        Score > 50 = bullish, Score < 50 = bearish.
        Distance from 50 = conviction.

        Returns:
            float in [0, 1]: calibrated probability of correct direction
        """
        if dimension not in self.params:
            # Fallback: linear mapping (distance from center / 50)
            return min(1.0, max(0.0, abs(raw_score - 50) / 50))

        A, B = self.params[dimension]
        # Use distance from center as input (0 = neutral, 50 = max conviction)
        distance = abs(raw_score - 50)
        return self.sigmoid(A * distance + B)

    def calibrate_all(
        self, dimension_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Calibrate all dimension scores at once."""
        return {
            dim: self.calibrate(dim, score)
            for dim, score in dimension_scores.items()
        }

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def fit_dimension(
        self,
        dimension: str,
        distances: np.ndarray,
        outcomes: np.ndarray,
        min_samples: int = 30,
    ) -> bool:
        """
        Fit Platt scaling for one dimension.

        Args:
            dimension: Name (e.g., "derivatives")
            distances: Array of |score - 50| values (0-50 scale)
            outcomes: Array of 0/1 (1 = direction was correct)
            min_samples: Minimum required samples

        Returns:
            True if fit succeeded
        """
        if len(distances) < min_samples:
            return False

        # Platt scaling: minimize negative log-likelihood
        def neg_log_likelihood(params: np.ndarray) -> float:
            A, B = params
            z = A * distances + B
            # Numerically stable log-sigmoid
            p = np.where(
                z >= 0,
                1.0 / (1.0 + np.exp(-z)),
                np.exp(z) / (1.0 + np.exp(z)),
            )
            p = np.clip(p, 1e-10, 1 - 1e-10)
            nll = -np.sum(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p))
            return nll

        result = minimize(
            neg_log_likelihood,
            x0=np.array([0.05, -0.5]),  # Initial: slight positive slope, negative intercept
            method="L-BFGS-B",
            bounds=[(-2.0, 2.0), (-10.0, 10.0)],
        )

        if result.success:
            A, B = result.x
            self.params[dimension] = (float(A), float(B))

            # Compute fit statistics
            z = A * distances + B
            p = np.where(
                z >= 0,
                1.0 / (1.0 + np.exp(-z)),
                np.exp(z) / (1.0 + np.exp(z)),
            )
            brier = float(np.mean((p - outcomes) ** 2))
            accuracy = float(np.mean((p > 0.5) == outcomes))

            self.fit_stats[dimension] = {
                "n_samples": int(len(distances)),
                "A": float(A),
                "B": float(B),
                "brier_score": round(brier, 4),
                "accuracy": round(accuracy, 4),
                "mean_prob": round(float(np.mean(p)), 4),
                "mean_outcome": round(float(np.mean(outcomes)), 4),
            }
            return True

        return False

    def fit_from_training_data(
        self,
        training_data: List[Dict[str, Any]],
        min_samples: int = 30,
    ) -> Dict[str, Any]:
        """
        Fit all calibrators from training data.

        training_data: List of dicts, each with:
            - "dimension_scores": {dim: score} (raw 0-100)
            - "direction": "bullish" or "bearish"
            - "pct_change": actual price change %
            - "asset": ticker

        Returns summary of fit results.
        """
        # Organize by dimension
        dim_data: Dict[str, Tuple[List[float], List[float]]] = {}

        for sample in training_data:
            scores = sample.get("dimension_scores", {})
            direction = sample.get("direction", "")
            pct_change = sample.get("pct_change")

            if pct_change is None or not direction:
                continue

            # Determine if direction was correct
            correct = (
                (direction == "bullish" and pct_change > 0)
                or (direction == "bearish" and pct_change < 0)
            )

            for dim, score in scores.items():
                if dim not in dim_data:
                    dim_data[dim] = ([], [])
                distance = abs(score - 50)
                dim_data[dim][0].append(distance)
                dim_data[dim][1].append(1.0 if correct else 0.0)

        # Fit each dimension
        results = {}
        for dim, (distances, outcomes) in dim_data.items():
            dist_arr = np.array(distances)
            out_arr = np.array(outcomes)
            success = self.fit_dimension(dim, dist_arr, out_arr, min_samples)
            results[dim] = {
                "fitted": success,
                "n_samples": len(distances),
                "base_rate": round(float(np.mean(out_arr)), 4) if len(out_arr) > 0 else 0,
            }
            if success:
                results[dim].update(self.fit_stats.get(dim, {}))

        return results

    def fit_from_file(self, filepath: str) -> Dict[str, Any]:
        """Load training data from JSON file and fit calibrators."""
        with open(filepath) as f:
            data = json.load(f)
        return self.fit_from_training_data(data)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, filepath: str) -> None:
        """Save fitted parameters to JSON."""
        data = {
            "params": {k: list(v) for k, v in self.params.items()},
            "fit_stats": self.fit_stats,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> bool:
        """Load fitted parameters from JSON."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            self.params = {
                k: tuple(v) for k, v in data.get("params", {}).items()
            }
            self.fit_stats = data.get("fit_stats", {})
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    # ------------------------------------------------------------------ #
    # Confidence-based gating (replaces distance-from-center)
    # ------------------------------------------------------------------ #

    def compute_signal_confidence(
        self,
        dimension_scores: Dict[str, float],
        dimension_weights: Dict[str, float],
        data_tiers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute signal confidence metrics for gating decisions.

        Returns:
            dict with:
                - calibrated_prob: float (0-1, probability of correct direction)
                - dimension_agreement: int (how many dims agree on direction)
                - data_quality: float (0-1, fraction of dims with full data)
                - kelly_edge: float (calibrated_prob - 0.5, positive = has edge)
                - conviction: str ("high", "medium", "low", "none")
                - signal_strength: str ("strong", "moderate", "weak")
        """
        calibrated = self.calibrate_all(dimension_scores)

        # Weighted calibrated probability
        total_weight = 0.0
        weighted_prob = 0.0
        for dim, prob in calibrated.items():
            w = dimension_weights.get(dim, 0.1)
            weighted_prob += prob * w
            total_weight += w
        avg_prob = weighted_prob / total_weight if total_weight > 0 else 0.5

        # Dimension agreement: how many dims lean the same direction?
        bullish_count = sum(1 for s in dimension_scores.values() if s > 55)
        bearish_count = sum(1 for s in dimension_scores.values() if s < 45)
        agreement = max(bullish_count, bearish_count)

        # Data quality
        data_quality = 1.0
        if data_tiers:
            full_count = sum(1 for t in data_tiers.values() if t == "full")
            data_quality = full_count / max(len(data_tiers), 1)

        # Kelly edge
        kelly_edge = avg_prob - 0.5

        # Conviction mapping
        if kelly_edge > 0.15:
            conviction = "high"
        elif kelly_edge > 0.08:
            conviction = "medium"
        elif kelly_edge > 0.03:
            conviction = "low"
        else:
            conviction = "none"

        # Signal strength (combines conviction + agreement + data quality)
        strength_score = (
            kelly_edge * 40  # edge contributes most
            + (agreement / max(len(dimension_scores), 1)) * 30  # agreement
            + data_quality * 30  # data quality
        )
        if strength_score > 20:
            signal_strength = "strong"
        elif strength_score > 10:
            signal_strength = "moderate"
        else:
            signal_strength = "weak"

        return {
            "calibrated_prob": round(avg_prob, 4),
            "dimension_agreement": agreement,
            "data_quality": round(data_quality, 2),
            "kelly_edge": round(kelly_edge, 4),
            "conviction": conviction,
            "signal_strength": signal_strength,
            "per_dimension": {dim: round(p, 4) for dim, p in calibrated.items()},
        }

    def should_emit_signal(
        self,
        confidence: Dict[str, Any],
        min_edge: float = 0.03,
        min_agreement: int = 2,
        min_data_quality: float = 0.3,
    ) -> Tuple[bool, str]:
        """
        Confidence-based gating: should we emit this signal?

        Returns:
            (should_emit, reason)
        """
        edge = confidence["kelly_edge"]
        agreement = confidence["dimension_agreement"]
        dq = confidence["data_quality"]

        if edge <= min_edge:
            return False, f"insufficient edge ({edge:.4f} < {min_edge})"
        if agreement < min_agreement:
            return False, f"low agreement ({agreement} < {min_agreement})"
        if dq < min_data_quality:
            return False, f"poor data quality ({dq:.2f} < {min_data_quality})"
        return True, "signal approved"
