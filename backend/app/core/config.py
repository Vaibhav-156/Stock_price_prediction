from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_env: str = Field("development", alias="APP_ENV")
    app_debug: bool = Field(True, alias="APP_DEBUG")
    app_secret_key: str = Field("change-me", alias="APP_SECRET_KEY")

    # PostgreSQL
    postgres_host: str = Field("localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    postgres_db: str = Field("swing_trading", alias="POSTGRES_DB")
    postgres_user: str = Field("postgres", alias="POSTGRES_USER")
    postgres_password: str = Field("postgres", alias="POSTGRES_PASSWORD")

    # Redis
    redis_host: str = Field("localhost", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    redis_db: int = Field(0, alias="REDIS_DB")
    redis_password: str = Field("", alias="REDIS_PASSWORD")

    # Yahoo Finance
    yf_rate_limit: int = Field(8, alias="YF_RATE_LIMIT_PER_SECOND")

    # ML
    ml_model_dir: str = Field("./saved_models", alias="ML_MODEL_DIR")
    ml_retrain_interval_hours: int = Field(24, alias="ML_RETRAIN_INTERVAL_HOURS")
    ml_lookback_days: int = Field(504, alias="ML_LOOKBACK_DAYS")
    ml_forward_days: int = Field(5, alias="ML_FORWARD_DAYS")
    ml_probability_threshold: float = Field(0.65, alias="ML_PROBABILITY_THRESHOLD")
    ml_xgboost_weight: float = Field(0.6, alias="ML_XGBOOST_WEIGHT")
    ml_lstm_weight: float = Field(0.4, alias="ML_LSTM_WEIGHT")

    # Risk
    risk_per_trade_pct: float = Field(1.0, alias="RISK_PER_TRADE_PCT")
    risk_reward_min: float = Field(1.5, alias="RISK_REWARD_MIN")
    max_portfolio_exposure_pct: float = Field(80.0, alias="MAX_PORTFOLIO_EXPOSURE_PCT")
    max_correlated_positions: int = Field(3, alias="MAX_CORRELATED_POSITIONS")
    max_drawdown_pct: float = Field(15.0, alias="MAX_DRAWDOWN_PCT")
    correlation_threshold: float = Field(0.7, alias="CORRELATION_THRESHOLD")

    # Backtesting
    backtest_slippage_bps: int = Field(10, alias="BACKTEST_SLIPPAGE_BPS")
    backtest_commission_bps: int = Field(10, alias="BACKTEST_COMMISSION_BPS")
    backtest_initial_capital: float = Field(7_500_000, alias="BACKTEST_INITIAL_CAPITAL")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def model_dir(self) -> Path:
        p = Path(self.ml_model_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
