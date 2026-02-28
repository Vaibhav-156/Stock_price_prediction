const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Signal {
  symbol: string;
  signal_date: string;
  probability: number;
  xgb_probability: number | null;
  lstm_probability: number | null;
  confidence_level: string;
  direction: string;
  suggested_position_size: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  risk_level: string | null;
  atr: number | null;
  regime: string | null;
}

export interface PriceData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PortfolioRisk {
  total_capital: number;
  current_equity: number;
  cash: number;
  exposure_pct: number;
  drawdown_pct: number;
  peak_capital: number;
  open_positions: number;
  max_new_positions: number;
  positions: Record<string, any>;
}

export interface BacktestResult {
  run_name: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_capital: number;
  metrics: {
    total_return_pct: number;
    cagr: number;
    sharpe_ratio: number;
    max_drawdown_pct: number;
    win_rate: number;
    profit_factor: number;
    expectancy: number;
    total_trades: number;
    avg_holding_days: number;
  };
  equity_curve: Array<{ date: string; equity: number; drawdown_pct: number }>;
  trades: Array<any>;
}

export interface ScreenResult {
  threshold: number;
  total_screened: number;
  signals_above_threshold: number;
  top_signals: Array<{
    symbol: string;
    probability: number;
    confidence: string;
    regime: string;
    rsi: number;
    volatility: number;
  }>;
  all_scores: Array<any>;
}

export interface StockDetail {
  symbol: string;
  current_price: number;
  prev_close: number;
  change: number;
  change_pct: number;
  high_52w: number;
  low_52w: number;
  avg_volume_20d: number;
  sma_20: number;
  sma_50: number;
  sma_200: number;
  rsi_14: number;
  volatility_20d: number;
  daily_returns_30d: number[];
  perf_1w: number;
  perf_1m: number;
  perf_3m: number;
  perf_1y: number;
  trend: string;
  data_points: number;
  predictions_3d: PricePrediction[];
}

export interface PricePrediction {
  day: number;
  date: string;
  predicted_close: number;
  predicted_high: number;
  predicted_low: number;
  change_pct: number;
  confidence: number;
}

export interface LiveQuote {
  symbol: string;
  price: number;
  prev_close: number;
  change: number;
  change_pct: number;
}

export interface QuotesResponse {
  quotes: LiveQuote[];
  market_open: boolean;
  timestamp: string;
  count: number;
}

export interface IntradaySignal {
  symbol: string;
  current_price: number;
  prev_close: number;
  change_pct: number;
  action: "BUY" | "SELL" | "HOLD";
  intraday_target: number;
  intraday_stop: number;
  rr_ratio: number;
  confidence: number;
  rsi: number;
  pivot: number;
  vwap: number;
  atr: number;
}

export interface IntradayResponse {
  signals: IntradaySignal[];
  count: number;
  buy_count: number;
  sell_count: number;
  hold_count: number;
  timestamp: string;
}

export interface NewsItem {
  title: string;
  publisher: string;
  published_at: string | null;
  age_hours: number;
  url: string;
  sentiment_score: number;
  sentiment_label: "BULLISH" | "BEARISH" | "NEUTRAL";
}

export interface StockSentiment {
  symbol: string;
  articles: NewsItem[];
  article_count: number;
  sentiment_score: number;
  sentiment_label: "BULLISH" | "BEARISH" | "NEUTRAL";
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  fetched_at: string;
}

export interface SymbolSentiment {
  symbol: string;
  score: number;
  label: "BULLISH" | "BEARISH" | "NEUTRAL";
  count: number;
}

export interface MarketSentiment {
  market_score: number;
  market_label: "BULLISH" | "BEARISH" | "NEUTRAL";
  market_description: string;
  total_articles: number;
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  symbol_breakdown: SymbolSentiment[];
  fetched_at: string;
}

/** Check if Indian market (NSE) is currently open: Mon-Fri, 9:15-15:30 IST */
export function isIndianMarketOpen(): boolean {
  const now = new Date();
  // Convert to IST (UTC+5:30)
  const utc = now.getTime() + now.getTimezoneOffset() * 60000;
  const ist = new Date(utc + 5.5 * 3600000);
  const day = ist.getDay(); // 0=Sun, 6=Sat
  if (day === 0 || day === 6) return false;
  const mins = ist.getHours() * 60 + ist.getMinutes();
  return mins >= 555 && mins <= 930; // 9:15=555, 15:30=930
}

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  // Health
  health: () => fetchAPI<{ status: string }>("/health"),

  // Signals
  getSignals: (symbols: string[], threshold = 0.55) =>
    fetchAPI<Signal[]>("/api/v1/signals", {
      method: "POST",
      body: JSON.stringify({ symbols, threshold }),
    }),

  getSignal: (symbol: string) =>
    fetchAPI<Signal>(`/api/v1/signals/${symbol}`),

  // Prices
  getPrices: (symbol: string, days = 90) =>
    fetchAPI<{ symbol: string; data: PriceData[] }>(
      `/api/v1/prices/${symbol}?days=${days}`
    ),

  // Stocks
  getStocks: () =>
    fetchAPI<{ universe: string[]; count: number }>("/api/v1/stocks"),

  // Training
  trainModels: (symbols?: string[]) =>
    fetchAPI<{ status: string; message: string }>("/api/v1/train", {
      method: "POST",
      body: JSON.stringify(symbols ? { symbols } : {}),
    }),

  getTrainStatus: () => fetchAPI<any>("/api/v1/train/status"),
  getModelMetrics: () => fetchAPI<any>("/api/v1/model/metrics"),
  getFeatureImportance: () => fetchAPI<any[]>("/api/v1/model/features"),

  // Portfolio
  getPortfolioRisk: () => fetchAPI<PortfolioRisk>("/api/v1/portfolio/risk"),

  getCorrelation: (symbols: string[]) =>
    fetchAPI<any>(`/api/v1/portfolio/correlation?symbols=${symbols.join(",")}`),

  // Screening
  screenUniverse: (threshold = 0.65, topN = 10) =>
    fetchAPI<ScreenResult>(
      `/api/v1/screen?threshold=${threshold}&top_n=${topN}`
    ),

  // Backtest
  runBacktest: (params: {
    symbols: string[];
    start_date: string;
    end_date: string;
    initial_capital?: number;
    threshold?: number;
  }) =>
    fetchAPI<BacktestResult>("/api/v1/backtest", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  // Stock detail
  getStockDetail: (symbol: string) =>
    fetchAPI<StockDetail>(`/api/v1/stock-detail/${symbol}`),

  // Live quotes (batch)
  getQuotes: (symbols?: string[]) => {
    const param = symbols ? `?symbols=${symbols.join(",")}` : "";
    return fetchAPI<QuotesResponse>(`/api/v1/quotes${param}`);
  },

  // Intraday signals
  getIntradaySignals: () =>
    fetchAPI<IntradayResponse>("/api/v1/intraday-signals"),

  // News & Sentiment
  getStockNews: (symbol: string) =>
    fetchAPI<StockSentiment>(`/api/v1/news/${symbol}`),

  getMarketSentiment: () =>
    fetchAPI<MarketSentiment>("/api/v1/market-sentiment"),
};
