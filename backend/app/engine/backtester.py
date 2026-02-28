"""
Realistic backtesting engine with walk-forward validation.
Includes slippage, commissions, delayed execution, and comprehensive metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.config import get_settings
from app.engine.risk_engine import RiskEngine, PortfolioState, Position
from app.ml.features.engineering import compute_all_features, create_labels
from app.ml.models.xgboost_model import XGBoostSwingModel
from app.ml.models.lstm_model import LSTMSwingModel

settings = get_settings()


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000
    threshold: float = 0.65
    slippage_bps: int = 10
    commission_bps: int = 10
    forward_days: int = 5
    max_holding_days: int = 10
    xgb_weight: float = 0.6
    lstm_weight: float = 0.4


class BacktestEngine:
    """
    Walk-forward backtesting engine.
    - Trains model on expanding window
    - Generates signals for out-of-sample period
    - Simulates execution with slippage + commission
    - Tracks equity curve & metrics
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig(
            initial_capital=settings.backtest_initial_capital,
            threshold=settings.ml_probability_threshold,
            slippage_bps=settings.backtest_slippage_bps,
            commission_bps=settings.backtest_commission_bps,
        )
        self.risk_engine = RiskEngine()

    def run(
        self,
        stock_data: dict[str, pd.DataFrame],
        index_df: pd.DataFrame | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        walk_forward_periods: int = 4,
    ) -> dict:
        """
        Execute walk-forward backtest.

        1. Split time into periods
        2. For each period: train on all prior data, test on period
        3. Simulate trades with risk management
        """
        logger.info(f"Starting backtest: {len(stock_data)} stocks, WF periods={walk_forward_periods}")

        # Compute features for all stocks
        featured: dict[str, pd.DataFrame] = {}
        for symbol, raw_df in stock_data.items():
            try:
                feat_df = compute_all_features(raw_df, index_df)
                if len(feat_df) > 100:
                    featured[symbol] = feat_df
            except Exception as e:
                logger.warning(f"Skipping {symbol}: {e}")

        if not featured:
            raise ValueError("No valid stock data for backtesting")

        # Determine date range
        all_dates = sorted(set().union(
            *[set(df.index.date if hasattr(df.index, 'date') else df.index) for df in featured.values()]
        ))
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        if len(all_dates) < 100:
            raise ValueError("Insufficient date range for backtesting")

        # Walk-forward splits
        total_days = len(all_dates)
        min_train_days = int(total_days * 0.5)
        test_period = (total_days - min_train_days) // walk_forward_periods

        # Initialize portfolio
        portfolio = PortfolioState(
            total_capital=self.config.initial_capital,
            cash=self.config.initial_capital,
        )
        equity_curve = []
        all_trades = []

        for period_idx in range(walk_forward_periods):
            train_end_idx = min_train_days + period_idx * test_period
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + test_period, total_days)

            if test_start_idx >= total_days:
                break

            train_end_date = all_dates[train_end_idx - 1]
            test_dates = all_dates[test_start_idx:test_end_idx]

            logger.info(
                f"WF Period {period_idx + 1}: "
                f"Train up to {train_end_date}, Test {test_dates[0]} to {test_dates[-1]}"
            )

            # Train XGBoost on expanding window
            xgb_model = XGBoostSwingModel()
            train_features, train_labels = [], []
            for sym, feat_df in featured.items():
                mask = feat_df.index.date <= train_end_date if hasattr(feat_df.index, 'date') else feat_df.index <= pd.Timestamp(train_end_date)
                train_chunk = feat_df[mask]
                if len(train_chunk) > 50:
                    labels = create_labels(train_chunk, self.config.forward_days)
                    valid = labels.notna()
                    train_features.append(train_chunk[valid])
                    train_labels.append(labels[valid])

            if not train_features:
                continue

            combined_feat = pd.concat(train_features, ignore_index=True)
            combined_labels = pd.concat(train_labels, ignore_index=True)
            xgb_model.train(combined_feat, combined_labels)

            # Simulate test period day by day
            for test_date in test_dates:
                # Check stops first
                current_prices = {}
                for sym, feat_df in featured.items():
                    mask = feat_df.index.date == test_date if hasattr(feat_df.index, 'date') else feat_df.index == pd.Timestamp(test_date)
                    day_data = feat_df[mask]
                    if len(day_data) > 0:
                        current_prices[sym] = float(day_data.iloc[0]["close"])

                self.risk_engine.check_stops(portfolio, current_prices, test_date)

                # Close positions past max holding
                for sym in list(portfolio.positions.keys()):
                    pos = portfolio.positions[sym]
                    if (test_date - pos.entry_date).days >= self.config.max_holding_days:
                        if sym in current_prices:
                            exit_price = self._apply_slippage(current_prices[sym], "sell")
                            exit_price = self._apply_commission(exit_price, "sell")
                            self.risk_engine.close_position(
                                portfolio, sym, exit_price, test_date, "max_hold"
                            )

                # Generate signals and enter new positions
                for sym, feat_df in featured.items():
                    if sym in portfolio.positions:
                        continue

                    # Get data up to test_date
                    mask = feat_df.index.date <= test_date if hasattr(feat_df.index, 'date') else feat_df.index <= pd.Timestamp(test_date)
                    available = feat_df[mask]
                    if len(available) < 50:
                        continue

                    try:
                        proba = float(xgb_model.predict_proba(available.tail(1))[0])
                    except Exception:
                        continue

                    if proba < self.config.threshold:
                        continue

                    entry_price = current_prices.get(sym)
                    if entry_price is None:
                        continue

                    # Delayed execution: use next candle (simulated by adding slippage)
                    entry_price = self._apply_slippage(entry_price, "buy")
                    entry_price = self._apply_commission(entry_price, "buy")

                    # ATR stop
                    atr = float(available.iloc[-1].get("atr_14", entry_price * 0.02))
                    stop_loss = entry_price - 2.0 * atr
                    take_profit = entry_price + 2.0 * atr * self.config.threshold

                    # Validate
                    validation = self.risk_engine.validate_trade(
                        portfolio, sym, entry_price, stop_loss, take_profit
                    )
                    if not validation["approved"]:
                        continue

                    sizing = validation["position_sizing"]
                    shares = sizing["shares"]
                    if shares <= 0:
                        continue

                    self.risk_engine.open_position(
                        portfolio, sym, entry_price, shares,
                        stop_loss, take_profit, test_date
                    )

                # Record equity
                equity = portfolio.cash + sum(
                    current_prices.get(s, p.entry_price) * p.shares
                    for s, p in portfolio.positions.items()
                )
                portfolio.peak_capital = max(portfolio.peak_capital, equity)
                equity_curve.append({
                    "date": str(test_date),
                    "equity": round(equity, 2),
                    "cash": round(portfolio.cash, 2),
                    "positions": len(portfolio.positions),
                    "drawdown_pct": round(
                        max(0, (portfolio.peak_capital - equity) / portfolio.peak_capital * 100), 2
                    ),
                })

        # Close remaining positions at last known price
        for sym in list(portfolio.positions.keys()):
            if sym in current_prices:
                exit_price = self._apply_slippage(current_prices[sym], "sell")
                self.risk_engine.close_position(
                    portfolio, sym, exit_price, all_dates[-1], "backtest_end"
                )

        # Calculate metrics
        metrics = self._compute_metrics(
            portfolio, equity_curve, self.config.initial_capital, all_dates
        )

        return {
            "run_name": f"WF_{walk_forward_periods}p_{start_date}_{end_date}",
            "start_date": str(all_dates[min_train_days] if min_train_days < len(all_dates) else all_dates[0]),
            "end_date": str(all_dates[-1]),
            "initial_capital": self.config.initial_capital,
            "final_capital": round(portfolio.current_equity, 2),
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trades": portfolio.trade_history,
        }

    # ── Slippage & Commission ─────────────────────────────

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = price * self.config.slippage_bps / 10000
        return price + slip if side == "buy" else price - slip

    def _apply_commission(self, price: float, side: str) -> float:
        comm = price * self.config.commission_bps / 10000
        return price + comm if side == "buy" else price - comm

    # ── Metrics ───────────────────────────────────────────

    def _compute_metrics(
        self,
        portfolio: PortfolioState,
        equity_curve: list[dict],
        initial_capital: float,
        all_dates: list,
    ) -> dict:
        trades = portfolio.trade_history
        if not trades:
            return {
                "total_return_pct": 0, "cagr": 0, "sharpe_ratio": 0,
                "max_drawdown_pct": 0, "win_rate": 0, "profit_factor": 0,
                "expectancy": 0, "total_trades": 0, "avg_holding_days": 0,
            }

        # Total return
        final = equity_curve[-1]["equity"] if equity_curve else initial_capital
        total_return = (final - initial_capital) / initial_capital * 100

        # CAGR
        if len(all_dates) > 1:
            years = (all_dates[-1] - all_dates[0]).days / 365.25
            cagr = ((final / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100
        else:
            cagr = 0

        # Sharpe ratio (daily returns)
        if len(equity_curve) > 1:
            equities = [e["equity"] for e in equity_curve]
            daily_returns = pd.Series(equities).pct_change().dropna()
            sharpe = (
                daily_returns.mean() / (daily_returns.std() + 1e-10) * math.sqrt(252)
            )
        else:
            sharpe = 0

        # Max drawdown
        max_dd = max((e["drawdown_pct"] for e in equity_curve), default=0)

        # Win rate
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / (gross_loss + 1e-10)

        # Expectancy
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0
        win_prob = len(wins) / len(trades) if trades else 0
        expectancy = (win_prob * avg_win) - ((1 - win_prob) * avg_loss)

        # Average holding days
        avg_hold = np.mean([t.get("holding_days", 0) for t in trades])

        return {
            "total_return_pct": round(total_return, 2),
            "cagr": round(cagr, 2),
            "sharpe_ratio": round(float(sharpe), 3),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 3),
            "expectancy": round(float(expectancy), 2),
            "total_trades": len(trades),
            "avg_holding_days": round(float(avg_hold), 1),
        }
