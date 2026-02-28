from __future__ import annotations

from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Price / OHLCV ──────────────────────────────────────────

class PriceBarOut(BaseModel):
    bar_date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int

    class Config:
        from_attributes = True


# ── Signal ─────────────────────────────────────────────────

class SignalOut(BaseModel):
    symbol: str
    signal_date: date
    probability: float = Field(..., description="Ensemble probability of positive 5-day return")
    xgb_probability: Optional[float] = None
    lstm_probability: Optional[float] = None
    confidence_level: str
    direction: str
    suggested_position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_level: Optional[str] = None
    atr: Optional[float] = None
    regime: Optional[str] = None

    class Config:
        from_attributes = True


class SignalRequest(BaseModel):
    symbols: list[str] = Field(..., description="List of stock symbols to analyze")
    threshold: Optional[float] = Field(None, description="Probability threshold override")


class TrainRequest(BaseModel):
    symbols: Optional[list[str]] = Field(None, description="Symbols to train on (defaults to Nifty 50)")


# ── Risk ───────────────────────────────────────────────────

class PositionSizeOut(BaseModel):
    symbol: str
    entry_price: float
    stop_loss: float
    position_size_shares: int
    position_value: float
    risk_amount: float
    risk_reward_ratio: float


class PortfolioRiskOut(BaseModel):
    total_capital: float
    cash: float
    exposure_pct: float
    current_drawdown_pct: float
    peak_capital: float
    open_positions: int
    max_new_positions: int


# ── Backtest ───────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbols: list[str]
    start_date: date
    end_date: date
    initial_capital: float = 7_500_000
    threshold: float = 0.65
    slippage_bps: int = 10
    commission_bps: int = 10


class BacktestMetrics(BaseModel):
    total_return_pct: float
    cagr: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    expectancy: float
    total_trades: int
    avg_holding_days: float


class BacktestOut(BaseModel):
    run_name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    metrics: BacktestMetrics
    equity_curve: list[dict]
    trades: list[dict]


# ── Stock ──────────────────────────────────────────────────

class StockOut(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    is_active: bool

    class Config:
        from_attributes = True


class StockCreate(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None


# ── Model Performance ─────────────────────────────────────

class ModelMetricsOut(BaseModel):
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    sample_count: int
    last_trained: Optional[datetime] = None
