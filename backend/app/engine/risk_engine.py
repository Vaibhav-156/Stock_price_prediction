"""
Risk management engine.
Handles position sizing, exposure limits, drawdown tracking, and correlation filtering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.config import get_settings

settings = get_settings()


@dataclass
class Position:
    symbol: str
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    entry_date: date
    direction: str = "LONG"

    @property
    def position_value(self) -> float:
        return self.entry_price * self.shares

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.shares

    def pnl(self, current_price: float) -> float:
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.shares
        return (self.entry_price - current_price) * self.shares

    def should_stop(self, current_price: float) -> bool:
        if self.direction == "LONG":
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def should_take_profit(self, current_price: float) -> bool:
        if self.direction == "LONG":
            return current_price >= self.take_profit
        return current_price <= self.take_profit


@dataclass
class PortfolioState:
    total_capital: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    peak_capital: float = 0.0
    trade_history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.peak_capital == 0:
            self.peak_capital = self.total_capital

    @property
    def exposure_value(self) -> float:
        return sum(p.position_value for p in self.positions.values())

    @property
    def exposure_pct(self) -> float:
        return (self.exposure_value / self.total_capital * 100) if self.total_capital > 0 else 0

    @property
    def current_equity(self) -> float:
        return self.cash + self.exposure_value

    @property
    def drawdown_pct(self) -> float:
        if self.peak_capital <= 0:
            return 0
        return max(0, (self.peak_capital - self.current_equity) / self.peak_capital * 100)


class RiskEngine:
    """
    Core risk management:
    - Fixed fractional position sizing (risk X% of capital per trade)
    - ATR-based stop losses
    - Maximum portfolio exposure cap
    - Correlation filtering
    - Drawdown circuit breaker
    """

    def __init__(self) -> None:
        self.risk_per_trade = settings.risk_per_trade_pct / 100.0
        self.risk_reward_min = settings.risk_reward_min
        self.max_exposure = settings.max_portfolio_exposure_pct
        self.max_correlated = settings.max_correlated_positions
        self.max_drawdown = settings.max_drawdown_pct
        self.correlation_threshold = settings.correlation_threshold

    # ── Position sizing ───────────────────────────────────

    def calculate_position_size(
        self,
        portfolio: PortfolioState,
        entry_price: float,
        stop_loss: float,
        direction: str = "LONG",
    ) -> dict:
        """
        Fixed fractional position sizing.
        Risk = (risk_per_trade * total_capital) / per-share risk.
        """
        # Per-share risk
        per_share_risk = abs(entry_price - stop_loss)
        if per_share_risk <= 0:
            return {"shares": 0, "error": "Invalid stop loss (zero risk per share)"}

        # Amount we can risk on this trade
        risk_budget = portfolio.total_capital * self.risk_per_trade
        shares = int(risk_budget / per_share_risk)

        # Cap by available cash
        max_shares_by_cash = int(portfolio.cash / entry_price)
        shares = min(shares, max_shares_by_cash)

        # Cap by exposure limit
        current_exposure = portfolio.exposure_value
        remaining_exposure = (
            portfolio.total_capital * self.max_exposure / 100.0 - current_exposure
        )
        max_shares_by_exposure = int(max(0, remaining_exposure) / entry_price)
        shares = min(shares, max_shares_by_exposure)

        if shares <= 0:
            return {"shares": 0, "error": "Position size zero after constraints"}

        position_value = shares * entry_price
        risk_amount = shares * per_share_risk
        rr_ratio = abs(entry_price - stop_loss)  # Placeholder; actual target needed

        return {
            "shares": shares,
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_pct_of_capital": round(risk_amount / portfolio.total_capital * 100, 3),
            "exposure_after": round(
                (current_exposure + position_value) / portfolio.total_capital * 100, 2
            ),
        }

    # ── Pre-trade validation ──────────────────────────────

    def validate_trade(
        self,
        portfolio: PortfolioState,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        returns_matrix: pd.DataFrame | None = None,
    ) -> dict:
        """
        Run all risk checks before entering a trade.
        Returns {approved: bool, reasons: [...]}
        """
        reasons = []
        approved = True

        # 1. Drawdown circuit breaker
        if portfolio.drawdown_pct >= self.max_drawdown:
            reasons.append(
                f"BLOCKED: Drawdown {portfolio.drawdown_pct:.1f}% >= max {self.max_drawdown}%"
            )
            approved = False

        # 2. Exposure cap
        if portfolio.exposure_pct >= self.max_exposure:
            reasons.append(
                f"BLOCKED: Exposure {portfolio.exposure_pct:.1f}% >= max {self.max_exposure}%"
            )
            approved = False

        # 3. Already in position
        if symbol in portfolio.positions:
            reasons.append(f"BLOCKED: Already holding {symbol}")
            approved = False

        # 4. Risk:Reward check
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr = reward / risk if risk > 0 else 0
        if rr < self.risk_reward_min:
            reasons.append(
                f"BLOCKED: R:R {rr:.2f} < minimum {self.risk_reward_min}"
            )
            approved = False

        # 5. Correlation filter
        if returns_matrix is not None and len(portfolio.positions) > 0:
            correlated = self._count_correlated(symbol, portfolio, returns_matrix)
            if correlated >= self.max_correlated:
                reasons.append(
                    f"BLOCKED: {correlated} correlated positions >= max {self.max_correlated}"
                )
                approved = False

        # 6. Sufficient cash
        sizing = self.calculate_position_size(portfolio, entry_price, stop_loss)
        if sizing.get("shares", 0) <= 0:
            reasons.append(f"BLOCKED: Cannot size position — {sizing.get('error', 'unknown')}")
            approved = False

        if approved:
            reasons.append("APPROVED")

        return {
            "approved": approved,
            "reasons": reasons,
            "position_sizing": sizing if approved else None,
            "risk_reward": round(rr, 2),
        }

    # ── Correlation filtering ─────────────────────────────

    def _count_correlated(
        self,
        symbol: str,
        portfolio: PortfolioState,
        returns_matrix: pd.DataFrame,
    ) -> int:
        """Count how many current positions are highly correlated with the new symbol."""
        if symbol not in returns_matrix.columns:
            return 0
        count = 0
        for pos_symbol in portfolio.positions:
            if pos_symbol in returns_matrix.columns:
                corr = returns_matrix[symbol].corr(returns_matrix[pos_symbol])
                if abs(corr) >= self.correlation_threshold:
                    count += 1
        return count

    def compute_correlation_matrix(
        self, price_data: dict[str, pd.DataFrame], window: int = 60
    ) -> pd.DataFrame:
        """Build returns correlation matrix from recent price data."""
        returns = {}
        for sym, df in price_data.items():
            if "close" in df.columns and len(df) >= window:
                returns[sym] = df["close"].pct_change().tail(window)
        if not returns:
            return pd.DataFrame()
        return pd.DataFrame(returns).corr()

    # ── Portfolio updates ─────────────────────────────────

    def open_position(
        self,
        portfolio: PortfolioState,
        symbol: str,
        entry_price: float,
        shares: int,
        stop_loss: float,
        take_profit: float,
        entry_date: date,
        direction: str = "LONG",
    ) -> PortfolioState:
        cost = entry_price * shares
        portfolio.cash -= cost
        portfolio.positions[symbol] = Position(
            symbol=symbol,
            entry_price=entry_price,
            shares=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_date=entry_date,
            direction=direction,
        )
        logger.info(
            f"Opened {direction} {shares} {symbol} @ {entry_price}, "
            f"SL={stop_loss}, TP={take_profit}"
        )
        return portfolio

    def close_position(
        self,
        portfolio: PortfolioState,
        symbol: str,
        exit_price: float,
        exit_date: date,
        reason: str = "manual",
    ) -> PortfolioState:
        if symbol not in portfolio.positions:
            return portfolio

        pos = portfolio.positions.pop(symbol)
        proceeds = exit_price * pos.shares
        portfolio.cash += proceeds
        pnl = pos.pnl(exit_price)

        # Update peak
        equity = portfolio.current_equity
        portfolio.peak_capital = max(portfolio.peak_capital, equity)

        # Record trade
        portfolio.trade_history.append({
            "symbol": symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "shares": pos.shares,
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / pos.position_value * 100, 2),
            "entry_date": str(pos.entry_date),
            "exit_date": str(exit_date),
            "holding_days": (exit_date - pos.entry_date).days,
            "reason": reason,
        })

        logger.info(
            f"Closed {symbol} @ {exit_price}, PnL={pnl:.2f}, reason={reason}"
        )
        return portfolio

    def check_stops(
        self,
        portfolio: PortfolioState,
        current_prices: dict[str, float],
        current_date: date,
    ) -> list[str]:
        """Check all positions for stop loss / take profit hits."""
        closed = []
        for symbol in list(portfolio.positions.keys()):
            price = current_prices.get(symbol)
            if price is None:
                continue
            pos = portfolio.positions[symbol]
            if pos.should_stop(price):
                self.close_position(portfolio, symbol, price, current_date, "stop_loss")
                closed.append(symbol)
            elif pos.should_take_profit(price):
                self.close_position(portfolio, symbol, price, current_date, "take_profit")
                closed.append(symbol)
        return closed

    # ── Risk metrics ──────────────────────────────────────

    def portfolio_risk_summary(self, portfolio: PortfolioState) -> dict:
        return {
            "total_capital": round(portfolio.total_capital, 2),
            "current_equity": round(portfolio.current_equity, 2),
            "cash": round(portfolio.cash, 2),
            "exposure_pct": round(portfolio.exposure_pct, 2),
            "drawdown_pct": round(portfolio.drawdown_pct, 2),
            "peak_capital": round(portfolio.peak_capital, 2),
            "open_positions": len(portfolio.positions),
            "max_new_positions": self._max_new_positions(portfolio),
            "positions": {
                sym: {
                    "entry_price": p.entry_price,
                    "shares": p.shares,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "direction": p.direction,
                    "entry_date": str(p.entry_date),
                }
                for sym, p in portfolio.positions.items()
            },
        }

    def _max_new_positions(self, portfolio: PortfolioState) -> int:
        if portfolio.drawdown_pct >= self.max_drawdown:
            return 0
        remaining_exposure = (
            portfolio.total_capital * self.max_exposure / 100.0 - portfolio.exposure_value
        )
        if remaining_exposure <= 0:
            return 0
        # Rough estimate: average position ~5% of capital
        avg_pos_size = portfolio.total_capital * 0.05
        return max(0, int(remaining_exposure / avg_pos_size))
