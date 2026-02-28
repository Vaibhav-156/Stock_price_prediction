"""
news_service.py
Fetches latest news for NSE stocks and runs VADER sentiment analysis.
Results are cached in Redis for 30 minutes.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..cache.redis_cache import cache

logger = logging.getLogger(__name__)
_analyzer = SentimentIntensityAnalyzer()

# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_text(text: str) -> dict:
    """Run VADER on a single text string."""
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "BULLISH"
    elif compound <= -0.05:
        label = "BEARISH"
    else:
        label = "NEUTRAL"
    return {"score": round(compound, 4), "label": label}


def _parse_article(article: dict) -> dict | None:
    """Convert a raw yfinance news dict into our standard format.

    Supports both the legacy flat format (yfinance < 1.0) and the new
    nested ``content`` format (yfinance >= 1.0).
    """
    try:
        # ── yfinance >= 1.0: news items are wrapped in a 'content' key ──
        inner = article.get("content") or article

        title = inner.get("title") or article.get("title") or ""
        if not title:
            return None

        # Publisher — new format nests under content.provider.displayName
        provider = inner.get("provider") or {}
        publisher = (
            provider.get("displayName")
            or article.get("publisher")
            or article.get("source")
            or "Unknown"
        )

        # Published time — new format uses ISO string 'pubDate', old uses unix ts
        pub_date_str = inner.get("pubDate") or inner.get("displayTime")
        ts_unix = article.get("providerPublishTime") or 0

        if pub_date_str:
            # ISO 8601 string (e.g. "2026-02-26T04:00:10Z")
            pub_dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            published_at = pub_dt.isoformat()
            age_hours = (time.time() - pub_dt.timestamp()) / 3600
        elif ts_unix:
            pub_dt = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
            published_at = pub_dt.isoformat()
            age_hours = (time.time() - ts_unix) / 3600
        else:
            published_at = None
            age_hours = 999

        # URL — new format nests under canonicalUrl / clickThroughUrl
        url = ""
        for url_key in ("canonicalUrl", "clickThroughUrl"):
            url_obj = inner.get(url_key)
            if isinstance(url_obj, dict) and url_obj.get("url"):
                url = url_obj["url"]
                break
        if not url:
            url = article.get("link") or article.get("url") or ""

        # Summary / description for richer sentiment
        summary = inner.get("summary") or inner.get("description") or ""
        content_text = f"{title}. {summary}" if summary else title

        sentiment = _score_text(content_text)

        return {
            "title": title,
            "publisher": publisher,
            "published_at": published_at,
            "age_hours": round(age_hours, 1),
            "url": url,
            "sentiment_score": sentiment["score"],
            "sentiment_label": sentiment["label"],
        }
    except Exception as e:
        logger.debug(f"Error parsing article: {e}")
        return None


# ── Core functions ─────────────────────────────────────────────────────────────

async def fetch_stock_news(symbol: str, max_articles: int = 10) -> dict:
    """
    Fetch recent news for a single stock and compute aggregate sentiment.
    Returns cached result (TTL = 30 min) to avoid hammering yfinance.
    """
    cache_key = f"news:{symbol}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker(symbol)
        raw_news: list[dict] = ticker.news or []
    except Exception as e:
        logger.warning(f"[NEWS] yfinance fetch failed for {symbol}: {e}")
        raw_news = []

    # Parse and filter
    articles = []
    for item in raw_news[:max_articles]:
        parsed = _parse_article(item)
        if parsed:
            articles.append(parsed)

    # Aggregate sentiment
    if articles:
        scores = [a["sentiment_score"] for a in articles]
        avg_score = round(sum(scores) / len(scores), 4)
        bullish = sum(1 for a in articles if a["sentiment_label"] == "BULLISH")
        bearish = sum(1 for a in articles if a["sentiment_label"] == "BEARISH")
        neutral = sum(1 for a in articles if a["sentiment_label"] == "NEUTRAL")
        if avg_score >= 0.05:
            overall_label = "BULLISH"
        elif avg_score <= -0.05:
            overall_label = "BEARISH"
        else:
            overall_label = "NEUTRAL"
    else:
        avg_score = 0.0
        overall_label = "NEUTRAL"
        bullish = bearish = neutral = 0

    result = {
        "symbol": symbol,
        "articles": articles,
        "article_count": len(articles),
        "sentiment_score": avg_score,
        "sentiment_label": overall_label,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    # Cache for 30 minutes
    await cache.cache_json(cache_key, result, ttl_seconds=1800)
    return result


async def fetch_market_sentiment(top_symbols: list[str] | None = None) -> dict:
    """
    Compute overall market sentiment by aggregating news across
    the Nifty index symbol plus the top most-active stocks.
    """
    cache_key = "news:market_sentiment"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached

    # Always include Nifty + a core basket
    core_symbols = ["^NSEI", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
                    "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS",
                    "HINDUNILVR.NS", "LT.NS", "AXISBANK.NS"]
    if top_symbols:
        core_symbols = list(dict.fromkeys(core_symbols + top_symbols[:10]))

    all_articles: list[dict] = []
    symbol_sentiments: list[dict] = []

    for sym in core_symbols:
        try:
            data = await fetch_stock_news(sym, max_articles=5)
            all_articles.extend(data["articles"])
            if data["article_count"] > 0:
                symbol_sentiments.append({
                    "symbol": sym,
                    "score": data["sentiment_score"],
                    "label": data["sentiment_label"],
                    "count": data["article_count"],
                })
        except Exception as e:
            logger.debug(f"[MARKET-SENTIMENT] Skip {sym}: {e}")

    # Aggregate all
    if all_articles:
        scores = [a["sentiment_score"] for a in all_articles]
        market_score = round(sum(scores) / len(scores), 4)
        bullish = sum(1 for a in all_articles if a["sentiment_label"] == "BULLISH")
        bearish = sum(1 for a in all_articles if a["sentiment_label"] == "BEARISH")
        neutral = len(all_articles) - bullish - bearish
    else:
        market_score = 0.0
        bullish = bearish = neutral = 0

    if market_score >= 0.05:
        market_label = "BULLISH"
        market_description = "Overall market news flow is positive"
    elif market_score <= -0.05:
        market_label = "BEARISH"
        market_description = "Overall market news flow is negative"
    else:
        market_label = "NEUTRAL"
        market_description = "Market news flow is mixed/neutral"

    result = {
        "market_score": market_score,
        "market_label": market_label,
        "market_description": market_description,
        "total_articles": len(all_articles),
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "symbol_breakdown": symbol_sentiments,
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    # Cache for 30 minutes
    await cache.cache_json(cache_key, result, ttl_seconds=1800)
    return result
