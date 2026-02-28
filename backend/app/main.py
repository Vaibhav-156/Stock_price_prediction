"""
FastAPI application entry point.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import get_settings
from app.cache.redis_cache import cache
from app.api.routes import router as api_router, _training_state
from app.ml.training.pipeline import run_training_pipeline
from app.ml.inference.ensemble import EnsemblePredictor

settings = get_settings()

# Global scheduler
scheduler = AsyncIOScheduler()


async def scheduled_training_job():
    """Run model training — called automatically by APScheduler."""
    logger.info("⏰ Scheduled auto-training triggered")
    _training_state["status"] = "training"
    try:
        result = await run_training_pipeline()
        _training_state["status"] = "completed"
        _training_state["last_trained"] = datetime.now().isoformat()
        _training_state["last_error"] = None
        # Reload models into the singleton ensemble
        ensemble = EnsemblePredictor()
        ensemble.load_models()
        logger.info(f"✅ Scheduled training completed: {result}")
    except Exception as e:
        _training_state["status"] = "failed"
        _training_state["last_error"] = str(e)
        logger.error(f"❌ Scheduled training failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Swing Trading Platform...")
    try:
        await cache.connect()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis not available: {e} — running without cache")

    # ── Auto-training scheduler ───────────────────────────────
    # Train daily at 16:00 IST (10:30 UTC) — right after Indian market close (15:30 IST).
    # Also run a second training at 08:00 IST (02:30 UTC) to incorporate overnight global data.
    scheduler.add_job(
        scheduled_training_job,
        CronTrigger(hour=10, minute=30, timezone="UTC"),  # 16:00 IST (post-market)
        id="daily_post_market_train",
        replace_existing=True,
    )
    scheduler.add_job(
        scheduled_training_job,
        CronTrigger(hour=2, minute=30, timezone="UTC"),  # 08:00 IST (pre-market)
        id="daily_pre_market_train",
        replace_existing=True,
    )
    scheduler.start()

    next_run = scheduler.get_jobs()[0].next_run_time
    _training_state["next_scheduled"] = next_run.isoformat() if next_run else None
    logger.info(f"📅 Auto-training scheduled. Next run: {next_run}")

    yield

    # Shutdown
    scheduler.shutdown(wait=False)
    await cache.disconnect()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Swing Trading Intelligence Platform",
    description="AI-powered swing trading system with probability-based signals",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
