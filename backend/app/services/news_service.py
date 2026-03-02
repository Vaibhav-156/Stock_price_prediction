"""
news_service.py
Fetches latest news for NSE stocks and runs VADER sentiment analysis.
Applies exponential recency decay so fresh news weighs more than stale news.
Falls back to Google News RSS for ETFs / tickers with no yfinance coverage.
Results are cached in Redis for 30 minutes.
"""
from __future__ import annotations

import logging
import math
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..cache.redis_cache import cache

logger = logging.getLogger(__name__)
_analyzer = SentimentIntensityAnalyzer()

# ── Recency decay ──────────────────────────────────────────────────────────────
# Weight = exp(-λ * age_hours).  Half-life ≈ 24 h  →  λ = ln(2)/24
_DECAY_LAMBDA = math.log(2) / 24.0   # ~0.0289

# ── ETF / index ticker → Google News search query mapping ─────────────────────
# When yfinance has no news for a symbol, we fall back to Google News RSS
# using a more descriptive search query for the underlying asset.
_TICKER_SEARCH_MAP: dict[str, str] = {
    # Gold ETFs
    "GOLDBEES.NS":      "Gold ETF India NSE price",
    "GOLDIETF.NS":      "Gold ETF India NSE price",
    "HDFCGOLD.NS":      "HDFC Gold ETF India",
    "AXISGOLD.NS":      "Axis Gold ETF India",
    "NIPPONIGOLD.NS":   "Nippon Gold ETF India",
    "KOTAKGOLD.NS":     "Kotak Gold ETF India",
    # Silver ETFs
    "SILVERBEES.NS":    "Silver ETF India NSE price",
    "SILVERIETF.NS":    "Silver ETF India NSE",
    "KOTAKSILVE.NS":    "Kotak Silver ETF India",
    # Nifty / broad market ETFs
    "NIFTYBEES.NS":     "Nifty 50 ETF India market",
    "JUNIORBEES.NS":    "Nifty Next 50 ETF India",
    "MOM100.NS":        "Nifty Momentum 100 ETF India",
    "NV20.NS":          "Nifty50 Value 20 ETF India",
    "SETFNIF50.NS":     "SBI Nifty 50 ETF India",
    # Bank / sector ETFs
    "BANKBEES.NS":      "Bank Nifty ETF India NSE",
    "ICICIB22.NS":      "ICICI Bank Nifty ETF India",
    "SETFNN50.NS":      "SBI Nifty Next 50 ETF India",
    "PSUBNKBEES.NS":    "PSU Bank ETF India NSE",
    "ITBEES.NS":        "Nifty IT ETF India NSE",
    "PHARMABEES.NS":    "Pharma ETF India NSE",
    # Index symbols
    "^NSEI":            "Nifty 50 index India stock market",
    "^NSEBANK":         "Bank Nifty index India",
    "^BSESN":           "Sensex BSE India stock market",
}


def _recency_weight(age_hours: float) -> float:
    """Exponential decay weight; articles older than 7 days get near-zero weight."""
    if age_hours > 168:   # 7 days — discard
        return 0.0
    return math.exp(-_DECAY_LAMBDA * max(age_hours, 0.0))


# ── Google News RSS fallback ──────────────────────────────────────────────────

def _google_news_rss(query: str, max_results: int = 15, max_age_days: int = 30) -> list[dict]:
    """
    Fetch articles from Google News RSS feed for a search query.
    Returns a list of parsed article dicts (same schema as _parse_article).

    max_age_days: articles older than this are discarded (default 30 for fallback
    sources like ETFs that have sparser news coverage than individual stocks).
    """
    articles: list[dict] = []
    max_age_hours = max_age_days * 24
    try:
        # Add time filter to query to bias towards recent results
        timed_query = f"{query} when:30d"
        encoded = urllib.parse.quote(timed_query)
        url = (f"https://news.google.com/rss/search?q={encoded}"
               "&hl=en-IN&gl=IN&ceid=IN:en")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (StockSentimentBot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()

        root = ET.fromstring(raw)
        channel = root.find("channel")
        if channel is None:
            return articles

        for item in channel.findall("item")[:max_results * 2]:
            try:
                title = (item.findtext("title") or "").strip()
                # Google wraps titles like "Title - Source"
                title = re.sub(r'\s+-\s+[^-]+$', '', title).strip() or title
                if not title:
                    continue

                link  = item.findtext("link") or ""
                pub_date_str = item.findtext("pubDate") or ""
                source_el = item.find("{http://purl.org/dc/elements/1.1/}publisher")
                source_tag = item.find("source")
                publisher = (
                    (source_el.text if source_el is not None else None)
                    or (source_tag.text if source_tag is not None else None)
                    or "Google News"
                )

                if pub_date_str:
                    pub_dt = parsedate_to_datetime(pub_date_str)
                    published_at = pub_dt.astimezone(timezone.utc).isoformat()
                    age_hours = (time.time() - pub_dt.timestamp()) / 3600
                else:
                    published_at = None
                    age_hours = 999

                if age_hours > max_age_hours:   # skip articles older than max_age_days
                    continue

                description = item.findtext("description") or ""
                # Strip HTML tags from description
                description = re.sub(r'<[^>]+>', '', description).strip()
                content_text = f"{title}. {description}" if description else title
                sentiment = _score_text(content_text)

                articles.append({
                    "title":           title,
                    "publisher":       publisher,
                    "published_at":    published_at,
                    "age_hours":       round(age_hours, 1),
                    "url":             link,
                    "sentiment_score": sentiment["score"],
                    "sentiment_label": sentiment["label"],
                })
            except Exception:
                continue

    except Exception as e:
        logger.debug(f"[GNEWS-RSS] Failed for query '{query}': {e}")

    return sorted(articles, key=lambda a: a["age_hours"])[:max_results]


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

async def fetch_stock_news(symbol: str, max_articles: int = 15) -> dict:
    """
    Fetch recent news for a single stock and compute both plain and
    recency-weighted aggregate sentiment.

    recency_weighted_score applies exponential decay (half-life = 24 h) so
    articles published in the last few hours carry the most influence on the
    model prediction.

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

    # Parse, filter and sort newest-first
    articles = []
    for item in raw_news[:max_articles * 2]:   # over-fetch, then filter
        parsed = _parse_article(item)
        if parsed:
            articles.append(parsed)

    # If yfinance returned nothing (ETFs, indices, low-coverage tickers),
    # fall back to Google News RSS using the mapped query or the symbol itself.
    used_fallback = False
    if not articles:
        used_fallback = True
        query = _TICKER_SEARCH_MAP.get(symbol)
        if not query:
            # Auto-build a query from the symbol: "SILVERBEES.NS" -> "SILVERBEES NSE India stock"
            base = symbol.replace(".NS", "").replace(".BO", "").replace("^", "")
            query = f"{base} NSE India stock news"
        logger.info(f"[NEWS] yfinance empty for {symbol}, trying Google News RSS: '{query}'")
        articles = _google_news_rss(query, max_results=max_articles, max_age_days=30)

    # Age cutoff: 7 days for yfinance articles, 30 days for Google News fallback
    max_age_h = 720 if used_fallback else 168
    articles = [a for a in articles if a["age_hours"] <= max_age_h]
    articles = sorted(articles, key=lambda a: a["age_hours"])[:max_articles]

    # Aggregate sentiment (plain + recency-weighted)
    if articles:
        scores = [a["sentiment_score"] for a in articles]
        weights = [_recency_weight(a["age_hours"]) for a in articles]
        total_weight = sum(weights) or 1.0

        avg_score = round(sum(scores) / len(scores), 4)
        weighted_score = round(
            sum(s * w for s, w in zip(scores, weights)) / total_weight, 4
        )

        bullish = sum(1 for a in articles if a["sentiment_label"] == "BULLISH")
        bearish = sum(1 for a in articles if a["sentiment_label"] == "BEARISH")
        neutral = sum(1 for a in articles if a["sentiment_label"] == "NEUTRAL")

        # Freshness: fraction of total weight in articles < 24 h
        fresh_weight = sum(
            w for a, w in zip(articles, weights) if a["age_hours"] < 24
        )
        freshness = round(fresh_weight / total_weight, 4)

        if weighted_score >= 0.05:
            overall_label = "BULLISH"
        elif weighted_score <= -0.05:
            overall_label = "BEARISH"
        else:
            overall_label = "NEUTRAL"
    else:
        avg_score = weighted_score = 0.0
        overall_label = "NEUTRAL"
        bullish = bearish = neutral = 0
        freshness = 0.0

    result = {
        "symbol": symbol,
        "articles": articles,
        "article_count": len(articles),
        "sentiment_score": avg_score,
        "recency_weighted_score": weighted_score,   # ← used by prediction engine
        "sentiment_label": overall_label,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "freshness": freshness,   # 0–1: 1 = all articles from last 24 h
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
                    "score": data["recency_weighted_score"],
                    "label": data["sentiment_label"],
                    "count": data["article_count"],
                    "freshness": data.get("freshness", 0.0),
                })
        except Exception as e:
            logger.debug(f"[MARKET-SENTIMENT] Skip {sym}: {e}")

    # Aggregate all using recency weights
    if all_articles:
        weights = [_recency_weight(a["age_hours"]) for a in all_articles]
        total_weight = sum(weights) or 1.0
        market_score = round(
            sum(a["sentiment_score"] * w for a, w in zip(all_articles, weights)) / total_weight, 4
        )
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
