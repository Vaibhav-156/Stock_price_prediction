"""
Data ingestion from Yahoo Finance with rate limiting and error handling.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings

settings = get_settings()


class MarketDataService:
    """Fetches and normalises OHLCV data via Yahoo Finance."""

    # Default universe — Nifty 50 (NSE India)
    NIFTY50: list[str] = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
        "LICI.NS", "LT.NS", "HCLTECH.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "TRENT.NS",
        "BAJAJFINSV.NS", "NTPC.NS", "TATASTEEL.NS", "POWERGRID.NS", "WIPRO.NS",
        "M&M.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "ONGC.NS", "JSWSTEEL.NS",
        "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "GRASIM.NS", "TECHM.NS",
        "BAJAJ-AUTO.NS", "INDUSINDBK.NS", "BRITANNIA.NS", "HINDALCO.NS", "CIPLA.NS",
        "DRREDDY.NS", "DIVISLAB.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS",
        "SBILIFE.NS", "HDFCLIFE.NS", "TATACONSUM.NS", "BPCL.NS", "SHRIRAMFIN.NS",
    ]

    # ETFs & high-momentum / profitable stocks
    EXTRA_STOCKS: list[str] = [
        # Silver / Gold / Commodity ETFs
        "TATAGOLD.NS", "SILVERBEES.NS", "GOLDBEES.NS",
        "SILVERETF.NS", "SILVER1.NS", "SILVERM.NS",
        "GOLDSHARE.NS", "CPSEETF.NS", "BANKETF.NS", "ITBEES.NS",
        "NIFTYBEES.NS", "JUNIORBEES.NS", "LIQUIDBEES.NS",
        # Popular mid/small cap performers
        "IRFC.NS", "ZOMATO.NS", "JIOFIN.NS", "HAL.NS",
        "BEL.NS", "IRCTC.NS", "PAYTM.NS", "DMART.NS",
        "TATAELXSI.NS", "POLYCAB.NS", "PERSISTENT.NS", "COFORGE.NS",
        "DEEPAKNTR.NS", "PIIND.NS", "ABCAPITAL.NS", "CANBK.NS",
        "PNB.NS", "IOB.NS", "RECLTD.NS", "PFC.NS",
        "NHPC.NS", "SJVN.NS", "TATAPOWER.NS", "ADANIGREEN.NS",
        "ADANIPOWER.NS", "SUZLON.NS", "IDEA.NS",
    ]

    ALL_STOCKS: list[str] = NIFTY50 + EXTRA_STOCKS

    INDEX_SYMBOL: str = "^NSEI"  # Nifty 50

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(settings.yf_rate_limit)

    # ── Public API ────────────────────────────────────────

    async def fetch_stock_history(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame indexed by date."""
        return await asyncio.to_thread(
            self._fetch_sync, symbol, start, end, period
        )

    async def fetch_multiple(
        self,
        symbols: list[str],
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch history for multiple symbols using a single batch download."""
        return await asyncio.to_thread(
            self._batch_download_sync, symbols, start, end
        )

    def _batch_download_sync(
        self,
        symbols: list[str],
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Download all symbols in one yf.download() call — much faster."""
        s = str(start or date.today() - timedelta(days=settings.ml_lookback_days))
        e = str(end or date.today())

        try:
            raw = yf.download(
                symbols, start=s, end=e,
                auto_adjust=False, threads=True, group_by="ticker",
            )
        except Exception as exc:
            logger.error(f"Batch download failed: {exc}")
            return {}

        if raw.empty:
            return {}

        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                if len(symbols) == 1:
                    df = raw.copy()
                else:
                    df = raw[sym].copy()

                df = df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
                })
                cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
                df = df[cols]
                if "adj_close" not in df.columns:
                    df["adj_close"] = df["close"]
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                if len(df) > 0:
                    df.index = pd.to_datetime(df.index).date
                    df.index = pd.DatetimeIndex(df.index)
                    df.index.name = "date"
                    out[sym] = df
            except Exception as ex:
                logger.debug(f"Skipping {sym} in batch: {ex}")
        return out

    async def fetch_index(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        return await self.fetch_stock_history(self.INDEX_SYMBOL, start, end)

    async def get_stock_info(self, symbol: str) -> dict:
        """Return basic stock metadata."""
        return await asyncio.to_thread(self._info_sync, symbol)

    # ── Internal sync wrappers (run in thread) ────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _fetch_sync(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        kwargs: dict = {}
        if period:
            kwargs["period"] = period
        else:
            kwargs["start"] = str(start or date.today() - timedelta(days=settings.ml_lookback_days))
            kwargs["end"] = str(end or date.today())

        hist = ticker.history(**kwargs, auto_adjust=False)
        if hist.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Normalise column names
        hist = hist.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })

        # Keep only needed columns
        cols = ["open", "high", "low", "close", "adj_close", "volume"]
        available = [c for c in cols if c in hist.columns]
        hist = hist[available]

        if "adj_close" not in hist.columns:
            hist["adj_close"] = hist["close"]

        # Clean
        hist = hist.replace([np.inf, -np.inf], np.nan).dropna()
        hist.index = pd.to_datetime(hist.index).date
        hist.index = pd.DatetimeIndex(hist.index)
        hist.index.name = "date"
        return hist

    def _info_sync(self, symbol: str) -> dict:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "name": info.get("longName") or info.get("shortName", symbol),
                "sector": info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to get info for {symbol}: {e}")
            return {"symbol": symbol, "name": symbol, "sector": "Unknown", "market_cap": 0}
