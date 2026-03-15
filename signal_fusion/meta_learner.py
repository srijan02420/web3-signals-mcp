"""
LightGBM Meta-Learner for Signal Fusion (Phase D2).

Learns non-linear combinations of dimension scores to predict
price direction. Replaces the hand-tuned weighted average with
a data-driven model that captures interaction effects.

Usage:
    learner = MetaLearner()
    learner.train("/tmp/calibration_data.json")
    prob = learner.predict(dimension_scores, extra_features)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MetaLearner:
    """
    LightGBM-based meta-learner that predicts price direction
    from dimension scores and contextual features.
    """

    MODEL_DIR = Path(__file__).resolve().parent / "models"
    MODEL_PATH = MODEL_DIR / "meta_learner.pkl"
    STATS_PATH = MODEL_DIR / "meta_learner_stats.json"

    # Feature names in order
    DIMENSION_FEATURES = [
        "whale", "technical", "derivatives", "narrative", "market", "trend"
    ]
    EXTRA_FEATURES = [
        "score_std",           # Std of dimension scores (agreement measure)
        "bullish_count",       # How many dims > 55
        "bearish_count",       # How many dims < 45
        "max_score",           # Highest dimension score
        "min_score",           # Lowest dimension score
        "score_range",         # max - min (spread)
        "avg_distance",        # Average |score - 50|
        "fear_greed",          # F&G index (0-100, or -1 if unavailable)
    ]

    def __init__(self) -> None:
        self.model = None
        self.stats: Dict[str, Any] = {}
        self.is_fitted = False

    # ------------------------------------------------------------------ #
    # Feature engineering
    # ------------------------------------------------------------------ #

    @classmethod
    def build_features(
        cls,
        dimension_scores: Dict[str, float],
        fear_greed: Optional[float] = None,
    ) -> np.ndarray:
        """
        Build feature vector from dimension scores and context.

        Returns 1D array of shape (n_features,).
        """
        # Dimension scores (in fixed order)
        dim_values = [dimension_scores.get(d, 50.0) for d in cls.DIMENSION_FEATURES]

        # Engineered features
        scores_arr = np.array(dim_values)
        score_std = float(np.std(scores_arr))
        bullish_count = sum(1 for s in dim_values if s > 55)
        bearish_count = sum(1 for s in dim_values if s < 45)
        max_score = float(np.max(scores_arr))
        min_score = float(np.min(scores_arr))
        score_range = max_score - min_score
        avg_distance = float(np.mean(np.abs(scores_arr - 50)))
        fg = fear_greed if fear_greed is not None else -1.0

        extra = [
            score_std, bullish_count, bearish_count,
            max_score, min_score, score_range, avg_distance, fg,
        ]

        return np.array(dim_values + extra, dtype=np.float64)

    @classmethod
    def build_feature_matrix(
        cls, training_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build feature matrix X, binary target y, and continuous target y_pct
        from training data.

        Returns:
            X: (n_samples, n_features)
            y: (n_samples,) binary (1 = direction correct, 0 = wrong)
            y_pct: (n_samples,) signed pct change
        """
        X_list = []
        y_list = []
        y_pct_list = []

        for sample in training_data:
            scores = sample.get("dimension_scores", {})
            fg = sample.get("fear_greed")
            correct = sample.get("correct", False)
            pct_change = sample.get("pct_change", 0.0)
            direction = sample.get("direction", "neutral")

            if direction == "neutral":
                continue

            features = cls.build_features(scores, fg)
            X_list.append(features)
            y_list.append(1 if correct else 0)

            # Signed return relative to direction
            effective_return = pct_change if direction == "bullish" else -pct_change
            y_pct_list.append(effective_return)

        return (
            np.array(X_list),
            np.array(y_list),
            np.array(y_pct_list),
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        training_data_path: str = "/tmp/calibration_data.json",
        n_splits: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train LightGBM meta-learner with walk-forward cross-validation.

        Returns training stats including out-of-sample accuracy.
        """
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit

        # Load data
        with open(training_data_path) as f:
            data = json.load(f)

        # Sort by timestamp for proper temporal ordering
        data.sort(key=lambda x: x.get("timestamp", ""))

        X, y, y_pct = self.build_feature_matrix(data)

        if verbose:
            print(f"  Training data: {len(X)} samples, {X.shape[1]} features")
            print(f"  Base rate (correct direction): {y.mean():.3f}")

        # Walk-forward cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_accuracies = []
        cv_brier = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Embargo: skip first 48h equivalent (4 samples at 12h intervals)
            embargo_size = min(4, len(test_idx) // 4)
            test_idx = test_idx[embargo_size:]
            if len(test_idx) == 0:
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20),
                    lgb.log_evaluation(period=0),
                ],
            )

            proba = model.predict_proba(X_test)[:, 1]
            preds = (proba > 0.5).astype(int)
            accuracy = float(np.mean(preds == y_test))
            brier = float(np.mean((proba - y_test) ** 2))
            cv_accuracies.append(accuracy)
            cv_brier.append(brier)

            if verbose:
                print(f"    Fold {fold+1}: acc={accuracy:.3f}, brier={brier:.4f}, n_test={len(y_test)}")

        # Train final model on all data
        final_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        final_model.fit(X, y)
        self.model = final_model
        self.is_fitted = True

        # Feature importance
        feature_names = self.DIMENSION_FEATURES + self.EXTRA_FEATURES
        importances = final_model.feature_importances_
        importance_dict = dict(zip(feature_names, importances.tolist()))

        # Save stats
        self.stats = {
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "base_rate": float(y.mean()),
            "cv_accuracy_mean": float(np.mean(cv_accuracies)) if cv_accuracies else 0,
            "cv_accuracy_std": float(np.std(cv_accuracies)) if cv_accuracies else 0,
            "cv_brier_mean": float(np.mean(cv_brier)) if cv_brier else 0,
            "feature_importances": importance_dict,
        }

        if verbose:
            print(f"\n  CV Accuracy: {self.stats['cv_accuracy_mean']:.3f} ± {self.stats['cv_accuracy_std']:.3f}")
            print(f"  CV Brier: {self.stats['cv_brier_mean']:.4f}")
            print(f"\n  Feature importances:")
            for name, imp in sorted(importance_dict.items(), key=lambda x: -x[1]):
                print(f"    {name:20s}: {imp}")

        return self.stats

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(
        self,
        dimension_scores: Dict[str, float],
        fear_greed: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Predict using trained meta-learner.

        Returns:
            dict with:
                - probability: float (0-1, P(price moves in predicted direction))
                - composite_score: float (0-100, probability mapped to score scale)
                - direction: "bullish" or "bearish"
                - confidence: "high" / "medium" / "low"
        """
        if not self.is_fitted or self.model is None:
            # Fallback: weighted average
            scores = list(dimension_scores.values())
            avg = sum(scores) / len(scores) if scores else 50.0
            return {
                "probability": abs(avg - 50) / 50,
                "composite_score": avg,
                "direction": "bullish" if avg > 50 else "bearish" if avg < 50 else "neutral",
                "confidence": "low",
                "fallback": True,
            }

        features = self.build_features(dimension_scores, fear_greed)
        proba = self.model.predict_proba(features.reshape(1, -1))[0]

        # proba[1] = P(direction is correct)
        p_correct = float(proba[1])

        # Determine direction from raw scores
        raw_avg = sum(dimension_scores.get(d, 50) for d in self.DIMENSION_FEATURES) / len(self.DIMENSION_FEATURES)
        direction = "bullish" if raw_avg > 50 else "bearish" if raw_avg < 50 else "neutral"

        # Map probability to composite score:
        # P(correct) > 0.5 → push score further from center
        # P(correct) < 0.5 → pull score toward center (or flip)
        distance = abs(raw_avg - 50)
        if p_correct > 0.5:
            # Confident in direction: scale distance by confidence
            adjusted_distance = distance * (0.5 + p_correct)
        else:
            # Not confident: shrink distance
            adjusted_distance = distance * p_correct * 2

        composite = 50 + adjusted_distance if direction == "bullish" else 50 - adjusted_distance
        composite = max(0, min(100, composite))

        # Confidence based on probability
        if p_correct > 0.65:
            confidence = "high"
        elif p_correct > 0.55:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "probability": round(p_correct, 4),
            "composite_score": round(composite, 1),
            "direction": direction,
            "confidence": confidence,
            "raw_avg": round(raw_avg, 1),
            "fallback": False,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, model_path: Optional[str] = None, stats_path: Optional[str] = None) -> None:
        """Save trained model and stats."""
        mp = Path(model_path) if model_path else self.MODEL_PATH
        sp = Path(stats_path) if stats_path else self.STATS_PATH
        mp.parent.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            with open(mp, "wb") as f:
                pickle.dump(self.model, f)

        with open(sp, "w") as f:
            json.dump(self.stats, f, indent=2)

    def load(self, model_path: Optional[str] = None, stats_path: Optional[str] = None) -> bool:
        """Load trained model and stats."""
        mp = Path(model_path) if model_path else self.MODEL_PATH
        sp = Path(stats_path) if stats_path else self.STATS_PATH

        try:
            with open(mp, "rb") as f:
                self.model = pickle.load(f)
            self.is_fitted = True

            if sp.exists():
                with open(sp) as f:
                    self.stats = json.load(f)
            return True
        except (FileNotFoundError, pickle.UnpicklingError):
            return False


# ------------------------------------------------------------------ #
# CLI: Train and save model
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/calibration_data.json"

    print("=" * 60)
    print("TRAINING LIGHTGBM META-LEARNER")
    print("=" * 60)

    learner = MetaLearner()
    stats = learner.train(data_path, verbose=True)
    learner.save()

    print(f"\n  Model saved to: {learner.MODEL_PATH}")
    print(f"  Stats saved to: {learner.STATS_PATH}")
