"""
XGBoost classifier for directional probability prediction.
Walk-forward training with expanding window validation.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from loguru import logger

from app.core.config import get_settings
from app.ml.features.engineering import FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS

settings = get_settings()


class XGBoostSwingModel:
    """
    Binary classifier predicting P(positive return over next N days).
    Uses expanding-window walk-forward validation.
    """

    MODEL_FILENAME = "xgb_swing_model.joblib"
    SCALER_FILENAME = "xgb_scaler.joblib"
    META_FILENAME = "xgb_meta.json"

    def __init__(self) -> None:
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: list[str] = []
        self.metrics: dict = {}
        self.last_trained: Optional[datetime] = None

    # ── Training ──────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        feature_cols: list[str] | None = None,
        val_ratio: float = 0.2,
    ) -> dict:
        """
        Train with walk-forward split (no shuffling).
        df rows must be chronologically ordered.
        """
        self.feature_cols = feature_cols or self._detect_features(df)
        logger.info(f"Training XGBoost on {len(self.feature_cols)} features, {len(df)} samples")

        # Align labels
        valid_mask = labels.notna()
        X = df.loc[valid_mask, self.feature_cols].values
        y = labels.loc[valid_mask].values

        # Walk-forward split: train on first (1-val_ratio), validate on last val_ratio
        split_idx = int(len(X) * (1 - val_ratio))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Replace any remaining NaN/inf
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Class weight for imbalanced labels
        pos_ratio = y_train.sum() / len(y_train)
        scale_pos = (1 - pos_ratio) / (pos_ratio + 1e-10)

        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False,
        )

        # Evaluate on validation set
        y_pred = self.model.predict(X_val_scaled)
        y_proba = self.model.predict_proba(X_val_scaled)[:, 1]

        self.metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_val, y_proba)),
            "sample_count": len(y_val),
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "pos_ratio_train": float(pos_ratio),
        }
        self.last_trained = datetime.utcnow()

        logger.info(f"XGBoost val metrics: {self.metrics}")
        return self.metrics

    def walk_forward_evaluate(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        n_splits: int = 5,
        feature_cols: list[str] | None = None,
    ) -> list[dict]:
        """Expanding-window walk-forward cross-validation."""
        feature_cols = feature_cols or self._detect_features(df)
        valid_mask = labels.notna()
        X = df.loc[valid_mask, feature_cols].values
        y = labels.loc[valid_mask].values

        total = len(X)
        min_train = int(total * 0.4)
        step = (total - min_train) // n_splits

        results = []
        for i in range(n_splits):
            train_end = min_train + i * step
            val_end = min(train_end + step, total)
            if train_end >= total or val_end > total:
                break

            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]

            scaler = StandardScaler()
            X_tr = np.nan_to_num(scaler.fit_transform(X_train))
            X_v = np.nan_to_num(scaler.transform(X_val))

            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                eval_metric="logloss", early_stopping_rounds=30,
                random_state=42, n_jobs=-1, tree_method="hist",
            )
            model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)

            y_pred = model.predict(X_v)
            y_proba = model.predict_proba(X_v)[:, 1]

            results.append({
                "fold": i,
                "train_size": len(y_train),
                "val_size": len(y_val),
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "recall": float(recall_score(y_val, y_pred, zero_division=0)),
                "auc_roc": float(roc_auc_score(y_val, y_proba)) if len(np.unique(y_val)) > 1 else 0.0,
            })

        logger.info(f"Walk-forward results: {results}")
        return results

    # ── Inference ─────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability of positive 5-day return for each row."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained or loaded.")

        X = df[self.feature_cols].values
        X_scaled = np.nan_to_num(self.scaler.transform(X))
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_single(self, features: dict) -> float:
        """Predict for a single observation."""
        row = pd.DataFrame([features])
        return float(self.predict_proba(row)[0])

    # ── Feature importance ────────────────────────────────

    def get_feature_importance(self, top_n: int = 20) -> list[dict]:
        if self.model is None:
            return []
        imp = self.model.feature_importances_
        pairs = sorted(zip(self.feature_cols, imp), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": float(i)} for f, i in pairs[:top_n]]

    # ── Persistence ───────────────────────────────────────

    def save(self, directory: Path | None = None) -> None:
        d = directory or settings.model_dir
        d.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, d / self.MODEL_FILENAME)
        joblib.dump(self.scaler, d / self.SCALER_FILENAME)

        meta = {
            "feature_cols": self.feature_cols,
            "metrics": self.metrics,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
        }
        (d / self.META_FILENAME).write_text(json.dumps(meta, indent=2))
        logger.info(f"XGBoost model saved to {d}")

    def load(self, directory: Path | None = None) -> bool:
        d = directory or settings.model_dir
        model_path = d / self.MODEL_FILENAME
        if not model_path.exists():
            logger.warning(f"No XGBoost model at {model_path}")
            return False

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(d / self.SCALER_FILENAME)

        meta = json.loads((d / self.META_FILENAME).read_text())
        self.feature_cols = meta["feature_cols"]
        self.metrics = meta.get("metrics", {})
        self.last_trained = (
            datetime.fromisoformat(meta["last_trained"]) if meta.get("last_trained") else None
        )
        logger.info(f"XGBoost model loaded from {d}")
        return True

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _detect_features(df: pd.DataFrame) -> list[str]:
        available = set(df.columns)
        cols = [c for c in FEATURE_COLUMNS if c in available]
        cols += [c for c in MARKET_FEATURE_COLUMNS if c in available]
        return cols
