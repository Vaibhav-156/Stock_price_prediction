"""
API routes for the trading platform.
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from loguru import logger

from app.core.config import get_settings
from app.schemas.trading import (
    SignalRequest, SignalOut, BacktestRequest, BacktestOut,
    StockOut, StockCreate, PortfolioRiskOut, ModelMetricsOut,
    PriceBarOut, TrainRequest,
)
from app.services.market_data import MarketDataService
from app.ml.inference.ensemble import EnsemblePredictor, SignalGenerator
from app.ml.training.pipeline import run_training_pipeline
from app.engine.risk_engine import RiskEngine, PortfolioState
from app.engine.backtester import BacktestEngine, BacktestConfig
from app.cache.redis_cache import cache
from app.services.news_service import fetch_stock_news, fetch_market_sentiment

settings = get_settings()
router = APIRouter()

# ── Singletons ────────────────────────────────────────────
data_service = MarketDataService()
ensemble = EnsemblePredictor()
signal_gen = SignalGenerator()
risk_engine = RiskEngine()

# In-memory portfolio (for demo — production would persist to DB)
_portfolio = PortfolioState(
    total_capital=settings.backtest_initial_capital,
    cash=settings.backtest_initial_capital,
)


# ═══════════════════════════════════════════════════════════
#  Signals
# ═══════════════════════════════════════════════════════════

@router.post("/signals", response_model=list[SignalOut])
async def generate_signals(req: SignalRequest):
    """Generate trade signals for given symbols."""
    import numpy as np
    import time as _time

    _t0 = _time.time()

    # Check if trained models are available
    models_ready = False
    if ensemble.xgb_model.model is None:
        models_ready = ensemble.load_models()
    else:
        models_ready = True

    threshold = req.threshold or settings.ml_probability_threshold

    logger.info(f"[SIGNAL TIMING] model check: {_time.time()-_t0:.2f}s")

    # ── 1. Check cache first — skip download for cached symbols ──
    signals = []
    uncached_symbols: list[str] = []
    for symbol in req.symbols:
        try:
            cache_key = cache.signal_key(symbol, str(date.today()))
            cached = await cache.get_json(cache_key)
            if cached:
                signals.append(SignalOut(**cached))
                continue
        except Exception:
            pass
        uncached_symbols.append(symbol)

    logger.info(f"[SIGNAL TIMING] cache check: {_time.time()-_t0:.2f}s, cached={len(signals)}, uncached={len(uncached_symbols)}")

    if not uncached_symbols:
        return signals  # All cached — instant response

    # ── 2. Fetch only uncached symbols ──
    signal_lookback = 120 if not models_ready else settings.ml_lookback_days
    start = date.today() - timedelta(days=signal_lookback)

    all_syms = uncached_symbols + [data_service.INDEX_SYMBOL]
    _t1 = _time.time()
    all_data = await data_service.fetch_multiple(all_syms, start=start)
    logger.info(f"[SIGNAL TIMING] batch download: {_time.time()-_t1:.2f}s for {len(all_syms)} symbols")
    index_df = all_data.pop(data_service.INDEX_SYMBOL, pd.DataFrame())
    stock_data = all_data

    for symbol, df in stock_data.items():
        try:
            if len(df) < 20:
                continue

            current_price = float(df.iloc[-1]["close"])

            # ── Fetch news sentiment for this stock ──
            sent_score = None
            sent_label = None
            try:
                news_data = await fetch_stock_news(symbol, max_articles=10)
                if news_data and news_data.get("article_count", 0) > 0:
                    sent_score = news_data["sentiment_score"]
                    sent_label = news_data["sentiment_label"]
                    logger.info(f"[SENTIMENT] {symbol}: score={sent_score}, label={sent_label}")
            except Exception as e:
                logger.debug(f"[SENTIMENT] Could not fetch for {symbol}: {e}")

            if models_ready:
                # ── Use trained ML models (sentiment-aware) ──
                prediction = ensemble.predict(df, index_df, sentiment_score=sent_score)
                if "error" in prediction:
                    continue

                sig = signal_gen.generate_signal(symbol, prediction, current_price)
                if sig is None:
                    sig = {
                        "symbol": symbol,
                        "signal_date": date.today(),
                        "probability": prediction["probability"],
                        "xgb_probability": prediction.get("xgb_probability"),
                        "lstm_probability": prediction.get("lstm_probability"),
                        "sentiment_score": sent_score,
                        "sentiment_label": sent_label,
                        "confidence_level": "LOW",
                        "direction": "NEUTRAL",
                        "atr": prediction.get("atr"),
                        "regime": prediction.get("regime"),
                    }
                else:
                    sig["sentiment_score"] = sent_score
                    sig["sentiment_label"] = sent_label
                    sizing = risk_engine.calculate_position_size(
                        _portfolio, current_price, sig["stop_loss"]
                    )
                    sig["suggested_position_size"] = sizing.get("shares", 0)
            else:
                # ── Fallback: Technical-analysis-based signals ──
                closes = df["close"].values.astype(float)
                highs = df["high"].values.astype(float)
                lows = df["low"].values.astype(float)

                sma20 = float(np.mean(closes[-20:]))
                sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else sma20

                # RSI-14
                deltas = np.diff(closes[-15:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
                avg_loss = float(np.mean(losses)) if len(losses) > 0 else 1e-9
                rs = avg_gain / max(avg_loss, 1e-9)
                rsi = 100.0 - (100.0 / (1.0 + rs))

                # ATR-14
                tr_vals = []
                for i in range(-14, 0):
                    h, l, pc = highs[i], lows[i], closes[i - 1]
                    tr_vals.append(max(h - l, abs(h - pc), abs(l - pc)))
                atr_val = float(np.mean(tr_vals)) if tr_vals else 0.0

                # Composite score → probability
                score = 0.5
                if current_price > sma20:
                    score += 0.10
                if current_price > sma50:
                    score += 0.08
                if sma20 > sma50:
                    score += 0.07
                if rsi < 35:
                    score += 0.10  # oversold bounce
                elif rsi > 65:
                    score -= 0.10  # overbought risk
                # Recent momentum (5-day return)
                ret5 = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0.0
                score += np.clip(ret5 * 2, -0.10, 0.10)
                probability = float(np.clip(score, 0.10, 0.95))

                if probability >= threshold:
                    direction = "LONG"
                    conf = "HIGH" if probability >= 0.75 else "MEDIUM"
                elif probability < (1 - threshold):
                    direction = "SHORT"
                    conf = "HIGH" if probability <= 0.25 else "MEDIUM"
                else:
                    direction = "NEUTRAL"
                    conf = "LOW"

                sl_dist = atr_val * 2.0
                tp_dist = atr_val * 3.0
                stop_loss = current_price - sl_dist if direction == "LONG" else current_price + sl_dist
                take_profit = current_price + tp_dist if direction == "LONG" else current_price - tp_dist

                regime = "BULLISH" if sma20 > sma50 else "BEARISH"

                # Position sizing
                risk_per_trade = _portfolio.total_capital * 0.01  # 1% risk
                shares = int(risk_per_trade / max(sl_dist, 0.01))

                risk_level = "HIGH" if atr_val / current_price > 0.03 else (
                    "MEDIUM" if atr_val / current_price > 0.015 else "LOW"
                )

                # Blend sentiment into the fallback score too
                if sent_score is not None:
                    sent_weight = settings.ml_sentiment_weight
                    sent_proba = max(0.0, min(1.0, (sent_score + 1) / 2))
                    probability = float(
                        (1 - sent_weight) * probability + sent_weight * sent_proba
                    )
                    probability = float(np.clip(probability, 0.10, 0.95))

                sig = {
                    "symbol": symbol,
                    "signal_date": date.today(),
                    "probability": probability,
                    "xgb_probability": None,
                    "lstm_probability": None,
                    "sentiment_score": sent_score,
                    "sentiment_label": sent_label,
                    "confidence_level": conf,
                    "direction": direction,
                    "atr": atr_val,
                    "regime": regime,
                    "stop_loss": round(stop_loss, 2) if direction != "NEUTRAL" else None,
                    "take_profit": round(take_profit, 2) if direction != "NEUTRAL" else None,
                    "suggested_position_size": shares if direction != "NEUTRAL" else 0,
                    "risk_level": risk_level,
                }

            signal_out = SignalOut(**sig)
            signals.append(signal_out)

            # Cache result
            try:
                await cache.cache_json(cache_key, sig, ttl_seconds=14400)  # 4 hours
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            continue

    logger.info(f"[SIGNAL TIMING] TOTAL: {_time.time()-_t0:.2f}s, signals={len(signals)}")
    return signals


@router.get("/signals/{symbol}", response_model=SignalOut)
async def get_signal(symbol: str):
    """Get signal for a single stock."""
    result = await generate_signals(SignalRequest(symbols=[symbol.upper()]))
    if not result:
        raise HTTPException(404, f"Could not generate signal for {symbol}")
    return result[0]


# ═══════════════════════════════════════════════════════════
#  Prices / Data
# ═══════════════════════════════════════════════════════════


@router.get("/quotes")
async def get_live_quotes(symbols: str = Query(None)):
    """Return current price + change for a batch of symbols (fast endpoint)."""
    import numpy as np
    from datetime import datetime
    import pytz

    sym_list = symbols.split(",") if symbols else data_service.ALL_STOCKS
    sym_list = [s.strip().upper() for s in sym_list if s.strip()]

    # Check IST market hours: Mon-Fri, 9:15 AM - 3:30 PM
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)
    weekday = now_ist.weekday()  # 0=Mon, 6=Sun
    market_open_time = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_time = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    is_market_open = 0 <= weekday <= 4 and market_open_time <= now_ist <= market_close_time

    # Use yfinance download for batch — much faster than individual fetches
    try:
        import yfinance as yf
        tickers_str = " ".join(sym_list)
        data = await asyncio.to_thread(
            yf.download,
            tickers_str,
            period="5d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        quotes = []
        for sym in sym_list:
            try:
                if len(sym_list) == 1:
                    close_col = data["Close"]
                else:
                    close_col = data["Close"][sym]
                closes = close_col.dropna()
                if len(closes) < 2:
                    continue
                current = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                change = current - prev
                change_pct = (change / prev) * 100 if prev else 0
                quotes.append({
                    "symbol": sym,
                    "price": round(current, 2),
                    "prev_close": round(prev, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                })
            except Exception:
                continue

        return {
            "quotes": quotes,
            "market_open": is_market_open,
            "timestamp": now_ist.isoformat(),
            "count": len(quotes),
        }
    except Exception as e:
        logger.error(f"Batch quote fetch failed: {e}")
        raise HTTPException(500, f"Quote fetch failed: {e}")


@router.get("/prices/{symbol}")
async def get_prices(
    symbol: str,
    days: int = Query(90, ge=1, le=1000),
):
    """Fetch OHLCV price data."""
    start = date.today() - timedelta(days=days)
    df = await data_service.fetch_stock_history(symbol.upper(), start=start)
    if df.empty:
        raise HTTPException(404, f"No price data for {symbol}")

    records = []
    for idx, row in df.iterrows():
        records.append({
            "date": str(idx.date() if hasattr(idx, "date") else idx),
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": int(row["volume"]),
        })
    return {"symbol": symbol.upper(), "data": records}


@router.get("/stock-detail/{symbol}")
async def stock_detail(symbol: str):
    """Return computed statistics for a stock (no ML needed)."""
    import numpy as np
    start = date.today() - timedelta(days=365)
    df = await data_service.fetch_stock_history(symbol.upper(), start=start)
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")

    close = df["close"]
    current = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else current
    change = current - prev_close
    change_pct = (change / prev_close) * 100 if prev_close else 0

    high_52w = float(close.max())
    low_52w = float(close.min())
    avg_volume = int(df["volume"].tail(20).mean()) if "volume" in df.columns else 0

    # Simple moving averages
    sma_20 = float(close.tail(20).mean()) if len(close) >= 20 else current
    sma_50 = float(close.tail(50).mean()) if len(close) >= 50 else current
    sma_200 = float(close.tail(200).mean()) if len(close) >= 200 else current

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float(100 - (100 / (1 + rs.iloc[-1]))) if not np.isnan(rs.iloc[-1]) else 50.0

    # Volatility (20-day annualized)
    returns = close.pct_change().dropna()
    vol_20 = float(returns.tail(20).std() * np.sqrt(252) * 100) if len(returns) >= 20 else 0

    # Daily returns for mini-chart
    daily_returns = returns.tail(30).tolist()

    # Performance periods
    perf_1w = float((current / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0
    perf_1m = float((current / close.iloc[-22] - 1) * 100) if len(close) > 21 else 0
    perf_3m = float((current / close.iloc[-66] - 1) * 100) if len(close) > 65 else 0
    perf_1y = float((current / close.iloc[0] - 1) * 100) if len(close) > 200 else 0

    # Trend signal (SMA based)
    trend = "BULLISH" if current > sma_50 > sma_200 else "BEARISH" if current < sma_50 < sma_200 else "NEUTRAL"

    # ── 3-Day Price Predictions ─────────────────────────────────
    predictions_3d = _predict_3_days(close, df, current, sma_20, sma_50, rsi, vol_20)

    return {
        "symbol": symbol.upper(),
        "current_price": round(current, 2),
        "prev_close": round(prev_close, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "avg_volume_20d": avg_volume,
        "sma_20": round(sma_20, 2),
        "sma_50": round(sma_50, 2),
        "sma_200": round(sma_200, 2),
        "rsi_14": round(rsi, 1),
        "volatility_20d": round(vol_20, 1),
        "daily_returns_30d": [round(r * 100, 2) for r in daily_returns],
        "perf_1w": round(perf_1w, 1),
        "perf_1m": round(perf_1m, 1),
        "perf_3m": round(perf_3m, 1),
        "perf_1y": round(perf_1y, 1),
        "trend": trend,
        "data_points": len(df),
        "predictions_3d": predictions_3d,
    }


def _predict_3_days(close: pd.Series, df: pd.DataFrame, current: float,
                     sma_20: float, sma_50: float, rsi: float, vol_20: float) -> list[dict]:
    """
    Generate 3-day-ahead price forecasts using a blend of:
      1. Linear regression on the last 20 closes (trend extrapolation)
      2. Mean-reversion pull toward SMA-20
      3. Momentum from recent returns
      4. ATR-based high/low range
    Confidence degrades with each day out.
    """
    import numpy as np
    from datetime import timedelta as td

    n = min(20, len(close))
    recent = close.tail(n).values.astype(float)

    # --- Linear regression slope (daily trend) ---
    x = np.arange(n)
    slope, intercept = np.polyfit(x, recent, 1) if n > 1 else (0.0, current)

    # --- Mean-reversion pull toward SMA-20 ---
    mr_strength = 0.15  # how much to pull toward SMA-20 per day
    gap_to_sma = sma_20 - current

    # --- Momentum (weighted avg of last 5 daily returns) ---
    rets = close.pct_change().dropna().tail(5).values.astype(float)
    weights = np.array([1, 2, 3, 4, 5][-len(rets):], dtype=float)
    momentum = float(np.average(rets, weights=weights)) if len(rets) > 0 else 0.0

    # --- ATR for high/low range ---
    if all(c in df.columns for c in ["high", "low", "close"]):
        high = df["high"].tail(14).values.astype(float)
        low = df["low"].tail(14).values.astype(float)
        cl = df["close"].tail(14).values.astype(float)
        prev_cl = np.roll(cl, 1)
        prev_cl[0] = cl[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_cl), np.abs(low - prev_cl)))
        atr = float(tr.mean())
    else:
        atr = current * (vol_20 / 100) / np.sqrt(252)

    # --- RSI bias: overbought → slight drag, oversold → slight lift ---
    rsi_bias = 0.0
    if rsi > 70:
        rsi_bias = -0.002 * (rsi - 70) / 30  # up to -0.2% per day
    elif rsi < 30:
        rsi_bias = 0.002 * (30 - rsi) / 30   # up to +0.2% per day

    # --- Build predictions ---
    last_date = pd.Timestamp(close.index[-1])
    predictions = []
    prev_predicted = current

    for day in range(1, 4):
        # Combine signals
        trend_move = slope  # daily trend from regression
        mr_move = gap_to_sma * mr_strength * (1 / day)  # mean-reversion decays
        momentum_move = prev_predicted * momentum * (0.7 ** (day - 1))  # momentum decays
        rsi_move = prev_predicted * rsi_bias * (0.8 ** (day - 1))

        predicted_close = prev_predicted + trend_move + mr_move * 0.3 + momentum_move + rsi_move

        # Range from ATR (widen each day)
        day_atr = atr * (day ** 0.5)
        predicted_high = predicted_close + day_atr * 0.6
        predicted_low = predicted_close - day_atr * 0.6

        # Ensure high > low > 0
        predicted_high = max(predicted_high, predicted_close * 1.001)
        predicted_low = min(predicted_low, predicted_close * 0.999)
        predicted_low = max(predicted_low, 0.01)

        # Confidence: starts ~75%, drops ~10% per day
        confidence = max(0.40, 0.78 - 0.12 * (day - 1))

        # Next trading day (skip weekends)
        next_date = last_date + pd.tseries.offsets.BDay(day)

        change_pct = ((predicted_close - current) / current) * 100

        predictions.append({
            "day": day,
            "date": next_date.strftime("%Y-%m-%d"),
            "predicted_close": round(float(predicted_close), 2),
            "predicted_high": round(float(predicted_high), 2),
            "predicted_low": round(float(predicted_low), 2),
            "change_pct": round(float(change_pct), 2),
            "confidence": round(float(confidence), 2),
        })

        prev_predicted = predicted_close

    return predictions


# ═══════════════════════════════════════════════════════════
#  Intraday Signals — Buy / Sell with Stop Loss & Target
# ═══════════════════════════════════════════════════════════

@router.get("/intraday-signals")
async def intraday_signals():
    """
    Compute intraday buy/sell recommendation for all stocks.
    Uses VWAP-like pivot analysis, RSI, Bollinger Bands & momentum.
    Returns current price, intraday target, stop loss, action.
    """
    import numpy as np
    import time as _time

    t0 = _time.time()
    all_symbols = data_service.ALL_STOCKS

    # Fetch last 30 days of data in batch for intraday analysis
    start = date.today() - timedelta(days=45)
    try:
        stock_data = await data_service.fetch_multiple(all_symbols, start=start)
    except Exception as e:
        logger.error(f"Intraday batch fetch failed: {e}")
        raise HTTPException(500, "Failed to fetch market data")

    results = []

    for symbol in all_symbols:
        try:
            df = stock_data.get(symbol)
            if df is None or df.empty or len(df) < 10:
                continue

            close = df["close"].values.astype(float)
            high = df["high"].values.astype(float) if "high" in df.columns else close
            low = df["low"].values.astype(float) if "low" in df.columns else close
            volume = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(close))

            current_price = float(close[-1])
            prev_close = float(close[-2]) if len(close) > 1 else current_price

            # ── Pivot Points (Classic) ──
            prev_high = float(high[-1])
            prev_low = float(low[-1])
            pivot = (prev_high + prev_low + current_price) / 3
            r1 = 2 * pivot - prev_low
            s1 = 2 * pivot - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)

            # ── ATR (14-day) ──
            n_atr = min(14, len(close) - 1)
            tr_vals = []
            for i in range(-n_atr, 0):
                tr_vals.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
            atr = float(np.mean(tr_vals)) if tr_vals else current_price * 0.015

            # ── RSI (14) ──
            diffs = np.diff(close[-15:])
            gains = diffs[diffs > 0]
            losses = -diffs[diffs < 0]
            avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
            avg_loss = float(np.mean(losses)) if len(losses) > 0 else 1e-9
            rs = avg_gain / max(avg_loss, 1e-9)
            rsi = 100.0 - (100.0 / (1.0 + rs))

            # ── Bollinger Bands (20, 2) ──
            bb_n = min(20, len(close))
            bb_mean = float(np.mean(close[-bb_n:]))
            bb_std = float(np.std(close[-bb_n:])) if bb_n > 1 else 0
            bb_upper = bb_mean + 2 * bb_std
            bb_lower = bb_mean - 2 * bb_std

            # ── VWAP approximation ──
            n_vwap = min(10, len(close))
            vwap = float(np.sum(close[-n_vwap:] * volume[-n_vwap:]) / max(np.sum(volume[-n_vwap:]), 1))

            # ── Momentum (5-day return) ──
            ret5 = (close[-1] / close[-6] - 1) if len(close) >= 6 else 0.0
            ret1 = (close[-1] / close[-2] - 1) if len(close) >= 2 else 0.0

            # ── SMA 9 & 21 (short-term for intraday bias) ──
            sma9 = float(np.mean(close[-min(9, len(close)):])) 
            sma21 = float(np.mean(close[-min(21, len(close)):]))

            # ═══ Intraday Scoring ═══
            score = 0.0

            # Price vs VWAP
            if current_price > vwap:
                score += 0.15
            else:
                score -= 0.15

            # Price vs Pivot
            if current_price > pivot:
                score += 0.10
            elif current_price < pivot:
                score -= 0.10

            # SMA crossover (fast vs slow)
            if sma9 > sma21:
                score += 0.12
            else:
                score -= 0.12

            # RSI zones
            if rsi < 30:
                score += 0.15  # oversold → buy
            elif rsi < 40:
                score += 0.08
            elif rsi > 70:
                score -= 0.15  # overbought → sell
            elif rsi > 60:
                score -= 0.08

            # Bollinger Band position
            bb_range = bb_upper - bb_lower if bb_upper > bb_lower else 1
            bb_position = (current_price - bb_lower) / bb_range
            if bb_position < 0.2:
                score += 0.12  # near lower band → buy
            elif bb_position > 0.8:
                score -= 0.12  # near upper band → sell

            # Momentum boost
            score += float(np.clip(ret5 * 1.5, -0.10, 0.10))
            score += float(np.clip(ret1 * 3.0, -0.08, 0.08))

            # ── Decision ──
            if score >= 0.12:
                action = "BUY"
                # For BUY: target = R1 or current + 1.5*ATR, stop = S1 or current - 1*ATR
                intraday_target = round(max(r1, current_price + atr * 1.5), 2)
                intraday_stop = round(max(s1, current_price - atr * 1.0), 2)
            elif score <= -0.12:
                action = "SELL"
                # For SELL: target = S1 or current - 1.5*ATR, stop = R1 or current + 1*ATR
                intraday_target = round(min(s1, current_price - atr * 1.5), 2)
                intraday_stop = round(min(r1, current_price + atr * 1.0), 2)
            else:
                action = "HOLD"
                intraday_target = round(r1, 2)
                intraday_stop = round(s1, 2)

            # Ensure stop loss makes sense
            if action == "BUY" and intraday_stop >= current_price:
                intraday_stop = round(current_price - atr * 0.8, 2)
            elif action == "SELL" and intraday_stop <= current_price:
                intraday_stop = round(current_price + atr * 0.8, 2)

            # Confidence from score magnitude
            confidence = min(0.95, 0.50 + abs(score) * 1.2)

            # Risk-Reward
            if action == "BUY":
                reward = abs(intraday_target - current_price)
                risk = abs(current_price - intraday_stop)
            elif action == "SELL":
                reward = abs(current_price - intraday_target)
                risk = abs(intraday_stop - current_price)
            else:
                reward = abs(intraday_target - current_price)
                risk = abs(current_price - intraday_stop)

            rr_ratio = round(reward / max(risk, 0.01), 2)

            change_from_prev = round(((current_price - prev_close) / prev_close) * 100, 2)

            results.append({
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "prev_close": round(prev_close, 2),
                "change_pct": change_from_prev,
                "action": action,
                "intraday_target": intraday_target,
                "intraday_stop": intraday_stop,
                "rr_ratio": rr_ratio,
                "confidence": round(confidence, 2),
                "rsi": round(rsi, 1),
                "pivot": round(pivot, 2),
                "vwap": round(vwap, 2),
                "atr": round(atr, 2),
            })

        except Exception as e:
            logger.warning(f"Intraday signal failed for {symbol}: {e}")
            continue

    # Sort: BUY first (by confidence desc), then SELL, then HOLD
    action_order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    results.sort(key=lambda x: (action_order.get(x["action"], 3), -x["confidence"]))

    logger.info(f"[INTRADAY] Generated {len(results)} signals in {_time.time()-t0:.2f}s")

    return {
        "signals": results,
        "count": len(results),
        "buy_count": sum(1 for r in results if r["action"] == "BUY"),
        "sell_count": sum(1 for r in results if r["action"] == "SELL"),
        "hold_count": sum(1 for r in results if r["action"] == "HOLD"),
        "timestamp": pd.Timestamp.now("Asia/Kolkata").isoformat(),
    }


# ═══════════════════════════════════════════════════════════
#  News & Sentiment
# ═══════════════════════════════════════════════════════════

@router.get("/news/{symbol}")
async def stock_news(symbol: str):
    """Fetch latest news articles + VADER sentiment for a stock."""
    sym = symbol.upper()
    # Ensure .NS suffix for Indian equities (not index symbols like ^NSEI)
    if not sym.startswith("^") and not sym.endswith(".NS"):
        sym = sym + ".NS"
    try:
        data = await fetch_stock_news(sym)
        return data
    except Exception as e:
        logger.error(f"[NEWS] {sym} error: {e}")
        raise HTTPException(500, f"Failed to fetch news for {sym}: {str(e)}")


@router.get("/market-sentiment")
async def market_sentiment():
    """Aggregate news sentiment across Nifty index + top stocks."""
    try:
        data = await fetch_market_sentiment()
        return data
    except Exception as e:
        logger.error(f"[MARKET-SENTIMENT] error: {e}")
        raise HTTPException(500, f"Failed to compute market sentiment: {str(e)}")


@router.get("/stocks")
async def list_stocks():
    """Return the trading universe."""
    return {
        "universe": data_service.ALL_STOCKS,
        "count": len(data_service.ALL_STOCKS),
        "index": data_service.INDEX_SYMBOL,
    }


# ═══════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════

# Track training state globally
_training_state = {
    "status": "idle",  # idle | training | completed | failed
    "last_trained": None,
    "last_error": None,
    "next_scheduled": None,
}


@router.post("/train")
async def train_models(
    background_tasks: BackgroundTasks,
    req: TrainRequest = TrainRequest(),
):
    """Trigger model training (runs in background)."""
    if _training_state["status"] == "training":
        return {"status": "already_training", "message": "Training already in progress"}
    _training_state["status"] = "training"
    background_tasks.add_task(_train_task, req.symbols)
    return {"status": "training_started", "message": "Training running in background"}


@router.get("/train/status")
async def train_status():
    """Get current training status."""
    return _training_state


async def _train_task(symbols: list[str] | None = None):
    from datetime import datetime
    try:
        result = await run_training_pipeline(symbols)
        _training_state["status"] = "completed"
        _training_state["last_trained"] = datetime.now().isoformat()
        _training_state["last_error"] = None
        # Reload models into the singleton ensemble
        ensemble.load_models()
        logger.info(f"Training completed: {result}")
    except Exception as e:
        _training_state["status"] = "failed"
        _training_state["last_error"] = str(e)
        logger.error(f"Training failed: {e}")


@router.get("/model/metrics")
async def model_metrics():
    """Get current model performance metrics."""
    if ensemble.xgb_model.model is None:
        loaded = ensemble.load_models()
        if not loaded:
            # Return placeholder metrics so the UI isn't empty
            return {
                "xgboost": {
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "auc_roc": None,
                    "sample_count": 0,
                    "status": "not_trained",
                },
                "lstm": {
                    "accuracy": None,
                    "auc_roc": None,
                    "val_loss": None,
                    "sample_count": 0,
                    "status": "not_trained",
                },
                "xgb_last_trained": None,
                "lstm_last_trained": None,
                "training_state": _training_state,
            }
    metrics = ensemble.get_model_metrics()
    metrics["training_state"] = _training_state
    return metrics


@router.get("/model/features")
async def model_features():
    """Get feature importance from XGBoost model."""
    if ensemble.xgb_model.model is None:
        loaded = ensemble.load_models()
        if not loaded:
            # Return common trading features as placeholder
            placeholders = [
                "rsi_14", "macd", "bb_width", "sma_20_50_cross", "atr_14",
                "volume_ratio", "price_momentum_5d", "volatility_20d",
                "close_vs_sma200", "obv_slope", "stoch_k", "adx_14",
                "vwap_distance", "rsi_divergence", "earnings_proximity",
            ]
            return [{"feature": f, "importance": 0.0} for f in placeholders]
    return ensemble.xgb_model.get_feature_importance(top_n=30)


# ═══════════════════════════════════════════════════════════
#  Backtesting
# ═══════════════════════════════════════════════════════════

@router.post("/backtest")
async def run_backtest(req: BacktestRequest):
    """Run walk-forward backtest."""
    config = BacktestConfig(
        initial_capital=req.initial_capital,
        threshold=req.threshold,
        slippage_bps=req.slippage_bps,
        commission_bps=req.commission_bps,
    )
    engine = BacktestEngine(config)

    # Fetch data
    stock_data = await data_service.fetch_multiple(req.symbols, start=req.start_date, end=req.end_date)
    index_df = await data_service.fetch_index(start=req.start_date, end=req.end_date)

    if len(stock_data) < 2:
        raise HTTPException(400, "Need at least 2 stocks with data for backtesting")

    try:
        result = engine.run(stock_data, index_df, req.start_date, req.end_date)
        return result
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(500, f"Backtest error: {str(e)}")


# ═══════════════════════════════════════════════════════════
#  Risk / Portfolio
# ═══════════════════════════════════════════════════════════

@router.get("/portfolio/risk")
async def portfolio_risk():
    """Get current portfolio risk summary."""
    return risk_engine.portfolio_risk_summary(_portfolio)


@router.get("/portfolio/correlation")
async def portfolio_correlation(
    symbols: str = Query(..., description="Comma-separated symbols"),
):
    """Get correlation matrix for given symbols."""
    sym_list = [s.strip().upper() for s in symbols.split(",")]
    start = date.today() - timedelta(days=120)
    stock_data = await data_service.fetch_multiple(sym_list, start=start)
    corr_matrix = risk_engine.compute_correlation_matrix(stock_data)
    return {"symbols": sym_list, "matrix": corr_matrix.to_dict()}


# ═══════════════════════════════════════════════════════════
#  Screening
# ═══════════════════════════════════════════════════════════

@router.get("/screen")
async def screen_universe(
    threshold: float = Query(0.65, ge=0.5, le=0.95),
    top_n: int = Query(10, ge=1, le=50),
):
    """Screen the full universe and return top signals."""
    if ensemble.xgb_model.model is None:
        if not ensemble.load_models():
            raise HTTPException(503, "Models not trained")

    symbols = data_service.ALL_STOCKS
    start = date.today() - timedelta(days=settings.ml_lookback_days)

    stock_data = await data_service.fetch_multiple(symbols, start=start)
    index_df = await data_service.fetch_index(start=start)

    # Fetch sentiment for all screened stocks
    sentiment_scores: dict[str, float | None] = {}
    for sym in list(stock_data.keys()):
        try:
            news_data = await fetch_stock_news(sym, max_articles=5)
            if news_data and news_data.get("article_count", 0) > 0:
                sentiment_scores[sym] = news_data["sentiment_score"]
        except Exception:
            pass

    predictions = ensemble.predict_batch(stock_data, index_df, sentiment_scores=sentiment_scores)

    scored = []
    for sym, pred in predictions.items():
        if "error" in pred:
            continue
        prob = pred.get("probability", 0)
        scored.append({
            "symbol": sym,
            "probability": prob,
            "confidence": "HIGH" if prob >= 0.75 else "MEDIUM" if prob >= 0.65 else "LOW",
            "regime": pred.get("regime", "UNKNOWN"),
            "rsi": pred.get("rsi", 0),
            "volatility": pred.get("volatility", 0),
        })

    scored.sort(key=lambda x: x["probability"], reverse=True)
    above_threshold = [s for s in scored if s["probability"] >= threshold]

    return {
        "threshold": threshold,
        "total_screened": len(scored),
        "signals_above_threshold": len(above_threshold),
        "top_signals": above_threshold[:top_n],
        "all_scores": scored,
    }
