"""
Ensemble model combining XGBoost and LSTM probabilities.
Signal generation with probability thresholding.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.config import get_settings
from app.ml.models.xgboost_model import XGBoostSwingModel
from app.ml.models.lstm_model import LSTMSwingModel
from app.ml.features.engineering import (
    compute_all_features, create_labels, FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS,
)

settings = get_settings()


class EnsemblePredictor:
    """
    Weighted ensemble of XGBoost (primary) and LSTM (secondary).
    Produces calibrated probability of positive 5-day return.
    """

    def __init__(self) -> None:
        self.xgb_model = XGBoostSwingModel()
        self.lstm_model = LSTMSwingModel()
        self.xgb_weight = settings.ml_xgboost_weight
        self.lstm_weight = settings.ml_lstm_weight

    # ── Training ──────────────────────────────────────────

    def train_all(
        self,
        stock_data: dict[str, pd.DataFrame],
        index_df: pd.DataFrame | None = None,
    ) -> dict:
        """
        Train both models on pooled multi-stock data.
        stock_data: {symbol: OHLCV DataFrame}
        """
        logger.info(f"Training ensemble on {len(stock_data)} stocks")

        all_features = []
        all_labels = []

        for symbol, raw_df in stock_data.items():
            try:
                feat_df = compute_all_features(raw_df, index_df)
                labels = create_labels(feat_df, settings.ml_forward_days)

                # Drop rows where label is NaN (future not available)
                valid = labels.notna()
                feat_df = feat_df[valid]
                labels = labels[valid]

                all_features.append(feat_df)
                all_labels.append(labels)
            except Exception as e:
                logger.warning(f"Feature engineering failed for {symbol}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid training data generated")

        combined_df = pd.concat(all_features, ignore_index=True)
        combined_labels = pd.concat(all_labels, ignore_index=True)

        logger.info(f"Combined training set: {len(combined_df)} samples")

        # Train XGBoost
        xgb_metrics = self.xgb_model.train(combined_df, combined_labels)

        # Train LSTM
        lstm_metrics = self.lstm_model.train(combined_df, combined_labels)

        # Save both models
        self.xgb_model.save()
        self.lstm_model.save()

        return {
            "xgboost": xgb_metrics,
            "lstm": lstm_metrics,
            "total_samples": len(combined_df),
            "stocks_used": len(stock_data),
        }

    # ── Inference ─────────────────────────────────────────

    def predict(self, df: pd.DataFrame, index_df: pd.DataFrame | None = None) -> dict:
        """
        Generate ensemble prediction for a single stock's current state.
        df: recent OHLCV data (enough for feature warm-up, ~250+ rows).
        Returns dict with probabilities.
        """
        feat_df = compute_all_features(df, index_df)

        if len(feat_df) == 0:
            return {"error": "Insufficient data for features"}

        # XGBoost probability (last row = current)
        xgb_proba = float(self.xgb_model.predict_proba(feat_df.tail(1))[0])

        # LSTM probability
        lstm_proba = np.nan
        try:
            lstm_probas = self.lstm_model.predict_proba(feat_df)
            if not np.isnan(lstm_probas[-1]):
                lstm_proba = float(lstm_probas[-1])
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")

        # Weighted ensemble
        if np.isnan(lstm_proba):
            ensemble_proba = xgb_proba
        else:
            ensemble_proba = (
                self.xgb_weight * xgb_proba + self.lstm_weight * lstm_proba
            )

        # Feature snapshot for logging
        last_row = feat_df.iloc[-1]
        return {
            "probability": round(ensemble_proba, 4),
            "xgb_probability": round(xgb_proba, 4),
            "lstm_probability": round(lstm_proba, 4) if not np.isnan(lstm_proba) else None,
            "atr": round(float(last_row.get("atr_14", 0)), 4),
            "rsi": round(float(last_row.get("rsi_14", 0)), 2),
            "regime": _regime_label(last_row),
            "volatility": round(float(last_row.get("volatility_20", 0)), 4),
            "features": {
                col: round(float(last_row[col]), 4)
                for col in feat_df.columns
                if col in FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS
            },
        }

    def predict_batch(
        self,
        stock_data: dict[str, pd.DataFrame],
        index_df: pd.DataFrame | None = None,
    ) -> dict[str, dict]:
        """Generate predictions for multiple stocks."""
        results = {}
        for symbol, df in stock_data.items():
            try:
                results[symbol] = self.predict(df, index_df)
            except Exception as e:
                logger.warning(f"Prediction failed for {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        return results

    # ── Model loading ─────────────────────────────────────

    def load_models(self) -> bool:
        xgb_ok = self.xgb_model.load()
        lstm_ok = self.lstm_model.load()
        if not xgb_ok:
            logger.error("XGBoost model not found — training required")
        return xgb_ok  # LSTM is optional

    def get_model_metrics(self) -> dict:
        return {
            "xgboost": self.xgb_model.metrics,
            "lstm": self.lstm_model.metrics,
            "xgb_last_trained": (
                self.xgb_model.last_trained.isoformat() if self.xgb_model.last_trained else None
            ),
            "lstm_last_trained": (
                self.lstm_model.last_trained.isoformat() if self.lstm_model.last_trained else None
            ),
        }


# ═══════════════════════════════════════════════════════════
#  Signal generation
# ═══════════════════════════════════════════════════════════

class SignalGenerator:
    """
    Converts raw probability into actionable trade signals
    with confidence levels and risk parameters.
    """

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold or settings.ml_probability_threshold

    def generate_signal(
        self,
        symbol: str,
        prediction: dict,
        current_price: float,
        signal_date: date | None = None,
    ) -> dict | None:
        """
        Generate signal only if probability exceeds threshold.
        Returns None if no signal (below threshold).
        """
        prob = prediction.get("probability", 0)
        if prob < self.threshold:
            return None  # No signal — insufficient edge

        atr = prediction.get("atr", 0)
        regime = prediction.get("regime", "unknown")

        # ATR-based stop loss (2x ATR below entry for longs)
        stop_loss = round(current_price - 2.0 * atr, 2) if atr > 0 else round(current_price * 0.95, 2)

        # Target based on risk:reward minimum
        risk = current_price - stop_loss
        take_profit = round(current_price + risk * settings.risk_reward_min, 2)

        # Confidence bucketing
        confidence = _confidence_level(prob)

        # Risk classification
        volatility = prediction.get("volatility", 0)
        risk_level = _risk_classification(volatility, regime)

        return {
            "symbol": symbol,
            "signal_date": signal_date or date.today(),
            "probability": prob,
            "xgb_probability": prediction.get("xgb_probability"),
            "lstm_probability": prediction.get("lstm_probability"),
            "confidence_level": confidence,
            "direction": "LONG" if prob > 0.5 else "SHORT",
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_level": risk_level,
            "atr": atr,
            "regime": regime,
        }


def _regime_label(row: pd.Series) -> str:
    if row.get("regime_bull", 0) == 1:
        return "BULL"
    elif row.get("regime_bear", 0) == 1:
        return "BEAR"
    return "SIDEWAYS"


def _confidence_level(prob: float) -> str:
    if prob >= 0.75:
        return "HIGH"
    elif prob >= 0.65:
        return "MEDIUM"
    return "LOW"


def _risk_classification(volatility: float, regime: str) -> str:
    if volatility > 0.5 or regime == "BEAR":
        return "HIGH"
    elif volatility > 0.3:
        return "MEDIUM"
    return "LOW"
