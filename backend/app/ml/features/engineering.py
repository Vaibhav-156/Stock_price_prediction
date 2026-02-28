"""
Advanced feature engineering for swing trading.
All features use only past data — no future leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_all_features(df: pd.DataFrame, index_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Master feature builder.  Expects OHLCV DataFrame with columns:
    [open, high, low, close, adj_close, volume] indexed by date (ascending).

    Returns DataFrame with original columns + all engineered features.
    All NaN rows at the head (warm-up period) are dropped.
    """
    df = df.copy().sort_index()

    # ── Technical features ────────────────────────────────
    df = _add_rsi(df)
    df = _add_macd(df)
    df = _add_bollinger(df)
    df = _add_ema_sma_crossovers(df)
    df = _add_atr(df)
    df = _add_rolling_volatility(df)
    df = _add_volume_spike(df)
    df = _add_momentum_score(df)
    df = _add_breakout_detection(df)
    df = _add_candle_features(df)

    # ── Market context features ───────────────────────────
    if index_df is not None and len(index_df) > 0:
        df = _add_market_context(df, index_df)

    # ── Regime detection ──────────────────────────────────
    df = _add_regime(df)

    # Drop warm-up NaNs
    df = df.dropna()
    return df


# ═══════════════════════════════════════════════════════════
#  Technical indicators
# ═══════════════════════════════════════════════════════════

def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # RSI divergence proxy
    df["rsi_slope_5"] = df["rsi_14"].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    return df


def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_slope"] = df["macd_hist"].diff(3)
    return df


def _add_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_upper"] = sma + num_std * std
    df["bb_lower"] = sma - num_std * std
    df["bb_mid"] = sma
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma + 1e-10)
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    return df


def _add_ema_sma_crossovers(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()

    # Crossover signals (1 = bullish, -1 = bearish, 0 = neutral)
    df["ema_cross_9_21"] = np.sign(df["ema_9"] - df["ema_21"])
    df["sma_cross_50_200"] = np.sign(df["sma_50"] - df["sma_200"])

    # Distance from key MAs (normalized)
    df["dist_ema_21"] = (df["close"] - df["ema_21"]) / (df["ema_21"] + 1e-10)
    df["dist_sma_50"] = (df["close"] - df["sma_50"]) / (df["sma_50"] + 1e-10)
    df["dist_sma_200"] = (df["close"] - df["sma_200"]) / (df["sma_200"] + 1e-10)
    return df


def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.ewm(span=period, adjust=False).mean()
    df["atr_pct"] = df["atr_14"] / (df["close"] + 1e-10)
    return df


def _add_rolling_volatility(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["volatility_5"] = log_ret.rolling(5).std() * np.sqrt(252)
    df["volatility_20"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["volatility_ratio"] = df["volatility_5"] / (df["volatility_20"] + 1e-10)
    return df


def _add_volume_spike(df: pd.DataFrame) -> pd.DataFrame:
    vol_sma_20 = df["volume"].rolling(20).mean()
    df["volume_spike"] = df["volume"] / (vol_sma_20 + 1e-10)
    df["volume_trend"] = df["volume"].rolling(5).mean() / (vol_sma_20 + 1e-10)

    # On-balance volume trend
    obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
    df["obv_slope_10"] = obv.rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False
    )
    return df


def _add_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_10d"] = df["close"].pct_change(10)
    df["ret_20d"] = df["close"].pct_change(20)

    # Rate of change
    df["roc_10"] = (df["close"] - df["close"].shift(10)) / (df["close"].shift(10) + 1e-10) * 100

    # Composite momentum
    df["momentum_score"] = (
        0.4 * df["ret_5d"].rank(pct=True) +
        0.3 * df["ret_10d"].rank(pct=True) +
        0.3 * df["ret_20d"].rank(pct=True)
    )
    return df


def _add_breakout_detection(df: pd.DataFrame) -> pd.DataFrame:
    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()
    df["high_50"] = df["high"].rolling(50).max()

    df["breakout_20_high"] = (df["close"] >= df["high_20"].shift(1)).astype(int)
    df["breakdown_20_low"] = (df["close"] <= df["low_20"].shift(1)).astype(int)

    # Channel position
    channel_range = df["high_20"] - df["low_20"]
    df["channel_position"] = (df["close"] - df["low_20"]) / (channel_range + 1e-10)
    return df


def _add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    body = df["close"] - df["open"]
    full_range = df["high"] - df["low"] + 1e-10
    df["candle_body_pct"] = body / full_range
    df["upper_shadow_pct"] = (df["high"] - df[["close", "open"]].max(axis=1)) / full_range
    df["lower_shadow_pct"] = (df[["close", "open"]].min(axis=1) - df["low"]) / full_range
    return df


# ═══════════════════════════════════════════════════════════
#  Market context features
# ═══════════════════════════════════════════════════════════

def _add_market_context(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    idx = index_df[["close"]].rename(columns={"close": "index_close"}).copy()
    idx["index_ret_5d"] = idx["index_close"].pct_change(5)
    idx["index_ret_20d"] = idx["index_close"].pct_change(20)
    idx["index_momentum"] = idx["index_close"].ewm(span=10).mean() / (
        idx["index_close"].ewm(span=30).mean() + 1e-10
    ) - 1

    # Merge on date index
    df = df.join(idx[["index_ret_5d", "index_ret_20d", "index_momentum"]], how="left")

    # Rolling correlation with index
    if "ret_1d" not in df.columns:
        df["ret_1d"] = df["close"].pct_change(1)

    idx_ret = index_df["close"].pct_change(1).reindex(df.index)
    df["corr_to_index_20"] = df["ret_1d"].rolling(20).corr(idx_ret)
    df["corr_to_index_60"] = df["ret_1d"].rolling(60).corr(idx_ret)

    # Relative strength vs index
    stock_cum = (1 + df["ret_1d"]).rolling(20).apply(np.prod, raw=True)
    index_cum = (1 + idx_ret).rolling(20).apply(np.prod, raw=True)
    df["relative_strength_20"] = stock_cum / (index_cum + 1e-10)

    return df


# ═══════════════════════════════════════════════════════════
#  Regime detection
# ═══════════════════════════════════════════════════════════

def _add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple regime classifier based on SMA slope and volatility.
    0 = sideways, 1 = bull, -1 = bear
    """
    if "sma_50" not in df.columns:
        df["sma_50"] = df["close"].rolling(50).mean()
    if "volatility_20" not in df.columns:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["volatility_20"] = log_ret.rolling(20).std() * np.sqrt(252)

    sma_slope = df["sma_50"].pct_change(10)
    vol_median = df["volatility_20"].rolling(60).median()

    conditions = [
        (sma_slope > 0.01) & (df["close"] > df["sma_50"]),
        (sma_slope < -0.01) & (df["close"] < df["sma_50"]),
    ]
    choices = [1, -1]
    df["regime"] = np.select(conditions, choices, default=0)

    # Encode as separate columns for ML
    df["regime_bull"] = (df["regime"] == 1).astype(int)
    df["regime_bear"] = (df["regime"] == -1).astype(int)
    df["regime_sideways"] = (df["regime"] == 0).astype(int)

    return df


# ═══════════════════════════════════════════════════════════
#  Label generation (for training only)
# ═══════════════════════════════════════════════════════════

def create_labels(df: pd.DataFrame, forward_days: int = 5) -> pd.Series:
    """
    Binary label: 1 if close price is higher after `forward_days`, else 0.
    Uses .shift(-forward_days) so last `forward_days` rows will be NaN.
    """
    future_ret = df["close"].shift(-forward_days) / df["close"] - 1
    return (future_ret > 0).astype(int)


# ═══════════════════════════════════════════════════════════
#  Feature column list (excludes raw OHLCV & label)
# ═══════════════════════════════════════════════════════════

FEATURE_COLUMNS = [
    "rsi_14", "rsi_slope_5",
    "macd", "macd_signal", "macd_hist", "macd_hist_slope",
    "bb_width", "bb_pct",
    "ema_cross_9_21", "sma_cross_50_200",
    "dist_ema_21", "dist_sma_50", "dist_sma_200",
    "atr_14", "atr_pct",
    "volatility_5", "volatility_20", "volatility_ratio",
    "volume_spike", "volume_trend", "obv_slope_10",
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "roc_10", "momentum_score",
    "breakout_20_high", "breakdown_20_low", "channel_position",
    "candle_body_pct", "upper_shadow_pct", "lower_shadow_pct",
    "regime_bull", "regime_bear", "regime_sideways",
]

MARKET_FEATURE_COLUMNS = [
    "index_ret_5d", "index_ret_20d", "index_momentum",
    "corr_to_index_20", "corr_to_index_60", "relative_strength_20",
]
