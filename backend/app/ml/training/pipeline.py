"""
Training pipeline — standalone script for model training.
Can be run independently or triggered via API.
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path

from loguru import logger

from app.core.config import get_settings
from app.services.market_data import MarketDataService
from app.ml.inference.ensemble import EnsemblePredictor

settings = get_settings()


async def run_training_pipeline(
    symbols: list[str] | None = None,
    lookback_days: int | None = None,
) -> dict:
    """
    Full training pipeline:
    1. Fetch historical data for universe
    2. Compute features
    3. Train XGBoost + LSTM
    4. Save models
    """
    data_service = MarketDataService()
    ensemble = EnsemblePredictor()

    symbols = symbols or data_service.NIFTY50[:30]  # Use top 30 for faster training
    lookback = lookback_days or settings.ml_lookback_days
    start = date.today() - timedelta(days=lookback)

    logger.info(f"Training pipeline: {len(symbols)} symbols, lookback={lookback}d from {start}")

    # Fetch data
    stock_data = await data_service.fetch_multiple(symbols, start=start)
    index_df = await data_service.fetch_index(start=start)

    if len(stock_data) < 5:
        raise ValueError(f"Only {len(stock_data)} stocks fetched. Need at least 5.")

    logger.info(f"Fetched {len(stock_data)} stocks, training models...")

    # Train ensemble
    results = ensemble.train_all(stock_data, index_df)

    logger.info(f"Training complete: {results}")
    return results


if __name__ == "__main__":
    asyncio.run(run_training_pipeline())
