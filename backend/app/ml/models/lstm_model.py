"""
LSTM model for sequential price pattern recognition.
Secondary model in the ensemble.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger

from app.core.config import get_settings

settings = get_settings()


# ═══════════════════════════════════════════════════════════
#  PyTorch Dataset
# ═══════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 20):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]
        label = self.y[idx + self.seq_len]
        return x_seq, label


# ═══════════════════════════════════════════════════════════
#  LSTM Network
# ═══════════════════════════════════════════════════════════

class LSTMNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.bn(last_hidden)
        out = self.dropout(self.relu(self.fc1(out)))
        out = torch.sigmoid(self.fc2(out))
        return out.squeeze(-1)


# ═══════════════════════════════════════════════════════════
#  LSTM Swing Model Wrapper
# ═══════════════════════════════════════════════════════════

class LSTMSwingModel:
    """LSTM-based binary classifier for swing trading probability."""

    MODEL_FILENAME = "lstm_swing_model.pt"
    SCALER_FILENAME = "lstm_scaler.npy"
    META_FILENAME = "lstm_meta.json"

    # Sequence features — a subset focused on price action
    SEQUENCE_FEATURES = [
        "ret_1d", "ret_5d", "rsi_14", "macd_hist",
        "bb_pct", "atr_pct", "volume_spike",
        "volatility_ratio", "momentum_score", "channel_position",
        "candle_body_pct", "dist_ema_21",
    ]

    def __init__(self, seq_len: int = 20, hidden_size: int = 128, num_layers: int = 2):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network: Optional[LSTMNetwork] = None
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
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> dict:
        self.feature_cols = feature_cols or [
            c for c in self.SEQUENCE_FEATURES if c in df.columns
        ]
        logger.info(f"Training LSTM on {len(self.feature_cols)} features, seq_len={self.seq_len}")

        # Align
        valid_mask = labels.notna()
        X_all = df.loc[valid_mask, self.feature_cols].values
        y_all = labels.loc[valid_mask].values

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_all)
        X_scaled = np.nan_to_num(X_scaled)

        # Walk-forward split
        split_idx = int(len(X_scaled) * (1 - val_ratio))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_all[:split_idx], y_all[split_idx:]

        train_ds = SequenceDataset(X_train, y_train, self.seq_len)
        val_ds = SequenceDataset(X_val, y_val, self.seq_len)

        if len(train_ds) == 0 or len(val_ds) == 0:
            logger.warning("Not enough data for LSTM training")
            return {}

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Model
        self.network = LSTMNetwork(
            input_size=len(self.feature_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            # Train
            self.network.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.network(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validate
            self.network.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = self.network(x_batch)
                    val_loss += criterion(pred, y_batch).item()
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())

            avg_val_loss = val_loss / max(len(val_loader), 1)
            scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"LSTM early stopping at epoch {epoch}")
                    break

        # Final metrics
        preds = np.array(all_preds)
        labels_arr = np.array(all_labels)
        binary_preds = (preds > 0.5).astype(int)

        self.metrics = {
            "accuracy": float(accuracy_score(labels_arr, binary_preds)),
            "auc_roc": float(
                roc_auc_score(labels_arr, preds)
                if len(np.unique(labels_arr)) > 1 else 0.0
            ),
            "val_loss": float(best_val_loss),
            "epochs_trained": epoch + 1,
            "sample_count": len(labels_arr),
        }
        self.last_trained = datetime.utcnow()

        logger.info(f"LSTM val metrics: {self.metrics}")
        return self.metrics

    # ── Inference ─────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability for each row (needs seq_len preceding rows as context)."""
        if self.network is None or self.scaler is None:
            raise RuntimeError("LSTM not trained or loaded.")

        X = df[self.feature_cols].values
        X_scaled = np.nan_to_num(self.scaler.transform(X))

        self.network.eval()
        probas = []
        with torch.no_grad():
            for i in range(self.seq_len, len(X_scaled)):
                seq = torch.FloatTensor(X_scaled[i - self.seq_len : i]).unsqueeze(0).to(self.device)
                p = self.network(seq).cpu().item()
                probas.append(p)

        # Pad beginning with NaN
        result = np.full(len(df), np.nan)
        result[self.seq_len:] = probas
        return result

    def predict_single_sequence(self, sequence: np.ndarray) -> float:
        """Predict from a pre-scaled sequence of shape (seq_len, n_features)."""
        if self.network is None:
            raise RuntimeError("LSTM not trained or loaded.")
        self.network.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            return float(self.network(x).cpu().item())

    # ── Persistence ───────────────────────────────────────

    def save(self, directory: Path | None = None) -> None:
        d = directory or settings.model_dir
        d.mkdir(parents=True, exist_ok=True)

        torch.save({
            "state_dict": self.network.state_dict(),
            "input_size": len(self.feature_cols),
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "seq_len": self.seq_len,
        }, d / self.MODEL_FILENAME)

        np.save(d / self.SCALER_FILENAME, {
            "mean": self.scaler.mean_,
            "scale": self.scaler.scale_,
        })

        meta = {
            "feature_cols": self.feature_cols,
            "metrics": self.metrics,
            "seq_len": self.seq_len,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
        }
        (d / self.META_FILENAME).write_text(json.dumps(meta, indent=2))
        logger.info(f"LSTM model saved to {d}")

    def load(self, directory: Path | None = None) -> bool:
        d = directory or settings.model_dir
        model_path = d / self.MODEL_FILENAME
        if not model_path.exists():
            logger.warning(f"No LSTM model at {model_path}")
            return False

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.seq_len = checkpoint["seq_len"]

        meta = json.loads((d / self.META_FILENAME).read_text())
        self.feature_cols = meta["feature_cols"]
        self.metrics = meta.get("metrics", {})
        self.last_trained = (
            datetime.fromisoformat(meta["last_trained"]) if meta.get("last_trained") else None
        )

        self.network = LSTMNetwork(
            input_size=checkpoint["input_size"],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.eval()

        scaler_data = np.load(d / self.SCALER_FILENAME, allow_pickle=True).item()
        self.scaler = StandardScaler()
        self.scaler.mean_ = scaler_data["mean"]
        self.scaler.scale_ = scaler_data["scale"]
        self.scaler.var_ = scaler_data["scale"] ** 2
        self.scaler.n_features_in_ = len(self.feature_cols)

        logger.info(f"LSTM model loaded from {d}")
        return True
