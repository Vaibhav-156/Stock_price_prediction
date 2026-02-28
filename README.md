# Swing Trading Intelligence Platform — Indian Market (NSE)

Production-grade AI-powered swing trading system for the Indian stock market (Nifty 50) with probability-based signal generation, risk management, and realistic backtesting.

## Architecture

```
├── backend/                    # Python FastAPI service
│   ├── app/
│   │   ├── api/               # REST API routes
│   │   ├── cache/             # Redis caching layer
│   │   ├── core/              # Config, database
│   │   ├── engine/            # Risk engine, backtester
│   │   ├── ml/
│   │   │   ├── features/      # Feature engineering (40+ features)
│   │   │   ├── models/        # XGBoost, LSTM
│   │   │   ├── training/      # Training pipeline
│   │   │   └── inference/     # Ensemble predictor, signal generator
│   │   ├── models/            # SQLAlchemy ORM models
│   │   ├── schemas/           # Pydantic schemas
│   │   └── services/          # Market data ingestion
│   ├── migrations/            # Alembic migrations
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                   # Next.js dashboard
│   ├── src/
│   │   ├── app/               # App router pages
│   │   ├── components/        # React components
│   │   ├── lib/               # API client
│   │   └── types/
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
└── .env.example
```

## Core Capabilities

**ML Models:**
- XGBoost classifier (primary) — directional probability prediction
- LSTM (secondary) — sequential pattern recognition
- Weighted ensemble output with configurable weights

**Risk Engine:**
- Fixed fractional position sizing (1% capital risk per trade)
- ATR-based stop losses
- Minimum 1.5 risk:reward enforcement
- Portfolio exposure caps (80% max)
- Correlation filtering across positions
- Drawdown circuit breaker (15% max)

**Backtesting:**
- Walk-forward validation (no lookahead bias)
- Slippage modeling (10bps default)
- Commission modeling (10bps default)
- Delayed execution simulation
- Full performance metrics: Sharpe, CAGR, max drawdown, profit factor, expectancy

**Feature Engineering (40+ features):**
- RSI, MACD, Bollinger Bands, EMA/SMA crossovers
- ATR, rolling volatility, volume spike ratio
- Momentum scores, breakout detection
- Market regime detection (bull/bear/sideways)
- Index correlation, sector relative strength

## Quick Start

### Option 1: Docker (recommended)

```bash
# Clone and configure
cp .env.example .env

# Start all services
docker compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:3000
# API docs: http://localhost:8000/docs
```

### Option 2: Manual Setup

**Prerequisites:** Python 3.11+, Node.js 20+, PostgreSQL 16+, Redis 7+

#### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt

# Create database
createdb swing_trading

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/stocks` | List trading universe |
| GET | `/api/v1/prices/{symbol}?days=90` | OHLCV price data |
| POST | `/api/v1/signals` | Generate trade signals |
| GET | `/api/v1/signals/{symbol}` | Single stock signal |
| POST | `/api/v1/train` | Trigger model training |
| GET | `/api/v1/model/metrics` | Model performance |
| GET | `/api/v1/model/features` | Feature importance |
| POST | `/api/v1/backtest` | Run walk-forward backtest |
| GET | `/api/v1/portfolio/risk` | Portfolio risk summary |
| GET | `/api/v1/portfolio/correlation?symbols=RELIANCE.NS,TCS.NS` | Correlation matrix |
| GET | `/api/v1/screen?threshold=0.65&top_n=10` | Screen universe |

## Usage Workflow

1. **Train models** — POST `/api/v1/train` or click "Train Models" in dashboard
2. **Generate signals** — POST `/api/v1/signals` with your symbol list
3. **Review signals** — Only act on signals with probability > threshold (default 0.65)
4. **Validate with backtest** — POST `/api/v1/backtest` to check historical performance
5. **Size positions** — Use suggested position sizes from risk engine
6. **Monitor risk** — Track drawdown, exposure, and correlation

## Configuration

All configuration via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_PROBABILITY_THRESHOLD` | 0.65 | Min probability for signal generation |
| `ML_XGBOOST_WEIGHT` | 0.6 | XGBoost weight in ensemble |
| `ML_LSTM_WEIGHT` | 0.4 | LSTM weight in ensemble |
| `RISK_PER_TRADE_PCT` | 1.0 | % of capital risked per trade |
| `RISK_REWARD_MIN` | 1.5 | Minimum risk:reward ratio |
| `MAX_PORTFOLIO_EXPOSURE_PCT` | 80.0 | Max portfolio exposure % |
| `MAX_DRAWDOWN_PCT` | 15.0 | Drawdown circuit breaker % |
| `CORRELATION_THRESHOLD` | 0.7 | Max correlation between positions |
| `BACKTEST_SLIPPAGE_BPS` | 10 | Slippage in basis points |
| `BACKTEST_COMMISSION_BPS` | 10 | Commission in basis points |

## Deployment

### Backend (Railway / Render)

1. Push backend directory to a Git repo
2. Connect to Railway/Render
3. Set environment variables from `.env.example`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add PostgreSQL and Redis add-ons

### Frontend (Vercel)

1. Push frontend directory to a Git repo
2. Import project on Vercel
3. Set `NEXT_PUBLIC_API_URL` to your backend URL
4. Framework preset: Next.js
5. Deploy

## Disclaimer

This system is designed for quantitative research and informed decision-making. It does not guarantee profits. Past performance does not indicate future results. Always manage risk appropriately and never risk capital you cannot afford to lose.
