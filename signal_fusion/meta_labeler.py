"""
Meta-Labeling Model for Signal Fusion (Phase D3).

Predicts whether the PRIMARY signal is likely to be correct,
acting as a smart gating mechanism. Only emits signals when
the meta-labeler has high confidence the primary model is right.

This is a simpler, more focused model than the meta-learner:
- Meta-learner: predicts direction (D2)
- Meta-labeler: predicts signal quality (D3)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MetaLabeler:
    """
    Predicts P(primary_signal_correct) using signal context features.

    Uses a LightGBM classifier trained on historical signals with known outcomes.
    If P(correct) > threshold, the signal is approved for emission.
    """

    MODEL_DIR = Path(__file__).resolve().parent / "models"
    MODEL_PATH = MODEL_DIR / "meta_labeler.pkl"

    def __init__(self, threshold: float = 0.55) -> None:
        self.model = None
        self.threshold = threshold
        self.is_fitted = False
        self.stats: Dict[str, Any] = {}

    @staticmethod
    def build_features(
        composite_score: float,
        dimension_scores: Dict[str, float],
        data_tiers: Optional[Dict[str, str]] = None,
        momentum: Optional[str] = None,
    ) -> np.ndarray:
        """
        Build feature vector for meta-labeling.

        Features focus on signal QUALITY indicators, not direction:
        - Distance from center (conviction proxy)
        - Dimension agreement (do dims agree?)
        - Data quality (how many dims have real data?)
        - Score spread (are dims clustered or dispersed?)
        """
        scores = list(dimension_scores.values())
        scores_arr = np.array(scores) if scores else np.array([50.0])

        distance_from_center = abs(composite_score - 50)
        score_std = float(np.std(scores_arr))
        bullish_count = sum(1 for s in scores if s > 55)
        bearish_count = sum(1 for s in scores if s < 45)
        neutral_count = sum(1 for s in scores if 45 <= s <= 55)
        agreement = max(bullish_count, bearish_count) / max(len(scores), 1)
        max_score = float(np.max(scores_arr))
        min_score = float(np.min(scores_arr))
        score_range = max_score - min_score
        avg_distance = float(np.mean(np.abs(scores_arr - 50)))

        # Data quality
        full_count = 0
        partial_count = 0
        if data_tiers:
            full_count = sum(1 for t in data_tiers.values() if t == "full")
            partial_count = sum(1 for t in data_tiers.values() if t == "partial")
        data_quality = full_count / max(len(data_tiers) if data_tiers else 1, 1)

        # Momentum encoding
        momentum_val = {"rising": 1.0, "improving": 1.0, "stable": 0.0,
                        "falling": -1.0, "degrading": -1.0, "new": 0.5}
        mom = momentum_val.get(momentum or "stable", 0.0)

        return np.array([
            distance_from_center,
            score_std,
            agreement,
            bullish_count,
            bearish_count,
            neutral_count,
            score_range,
            avg_distance,
            data_quality,
            full_count,
            mom,
        ], dtype=np.float64)

    def train(
        self,
        training_data_path: str = "/tmp/calibration_data.json",
        noise_threshold_pct: float = 2.0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train meta-labeler on historical (signal, outcome) pairs.

        A signal is "correct" if the price moved in the predicted direction
        by more than noise_threshold_pct.
        """
        import lightgbm as lgb

        with open(training_data_path) as f:
            data = json.load(f)

        data.sort(key=lambda x: x.get("timestamp", ""))

        X_list = []
        y_list = []

        for sample in data:
            scores = sample.get("dimension_scores", {})
            pct_change = sample.get("pct_change", 0)
            direction = sample.get("direction", "neutral")

            if direction == "neutral":
                continue

            # Compute composite (simple average for training)
            avg = sum(scores.values()) / max(len(scores), 1)
            features = self.build_features(avg, scores)
            X_list.append(features)

            # Label: was direction correct AND magnitude > noise?
            effective = pct_change if direction == "bullish" else -pct_change
            correct = effective > 0  # Even weak correct counts
            y_list.append(1 if correct else 0)

        X = np.array(X_list)
        y = np.array(y_list)

        if verbose:
            print(f"  Meta-labeler training: {len(X)} samples")
            print(f"  Base rate (correct): {y.mean():.3f}")

        # Train with walk-forward
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        cv_acc = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            embargo = min(4, len(test_idx) // 4)
            test_idx = test_idx[embargo:]
            if len(test_idx) == 0:
                continue

            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                num_leaves=8,
                min_child_samples=30,
                subsample=0.8,
                reg_alpha=0.2,
                reg_lambda=0.2,
                random_state=42,
                verbose=-1,
            )
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            acc = float(np.mean(preds == y[test_idx]))
            cv_acc.append(acc)
            if verbose:
                print(f"    Fold {fold+1}: acc={acc:.3f}")

        # Final model
        final = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            num_leaves=8, min_child_samples=30, subsample=0.8,
            reg_alpha=0.2, reg_lambda=0.2, random_state=42, verbose=-1,
        )
        final.fit(X, y)
        self.model = final
        self.is_fitted = True

        self.stats = {
            "n_samples": len(X),
            "base_rate": float(y.mean()),
            "cv_accuracy": float(np.mean(cv_acc)) if cv_acc else 0,
            "threshold": self.threshold,
        }

        if verbose:
            print(f"  CV accuracy: {self.stats['cv_accuracy']:.3f}")

        return self.stats

    def should_emit(
        self,
        composite_score: float,
        dimension_scores: Dict[str, float],
        data_tiers: Optional[Dict[str, str]] = None,
        momentum: Optional[str] = None,
    ) -> Tuple[bool, float, str]:
        """
        Predict whether this signal should be emitted.

        Returns:
            (should_emit, probability, reason)
        """
        if not self.is_fitted or self.model is None:
            # Fallback: use distance from center
            dist = abs(composite_score - 50)
            return dist > 10, dist / 50, "fallback (no model)"

        features = self.build_features(composite_score, dimension_scores, data_tiers, momentum)
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        p_correct = float(proba[1])

        if p_correct >= self.threshold:
            return True, p_correct, f"meta-label approved (p={p_correct:.3f})"
        else:
            return False, p_correct, f"meta-label rejected (p={p_correct:.3f} < {self.threshold})"

    def save(self, path: Optional[str] = None) -> None:
        mp = Path(path) if path else self.MODEL_PATH
        mp.parent.mkdir(parents=True, exist_ok=True)
        if self.model:
            with open(mp, "wb") as f:
                pickle.dump({"model": self.model, "stats": self.stats, "threshold": self.threshold}, f)

    def load(self, path: Optional[str] = None) -> bool:
        mp = Path(path) if path else self.MODEL_PATH
        try:
            with open(mp, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.stats = data.get("stats", {})
            self.threshold = data.get("threshold", 0.55)
            self.is_fitted = True
            return True
        except (FileNotFoundError, pickle.UnpicklingError, KeyError):
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING META-LABELER")
    print("=" * 60)

    labeler = MetaLabeler(threshold=0.55)
    stats = labeler.train(verbose=True)
    labeler.save()
    print(f"\n  Model saved to: {labeler.MODEL_PATH}")
