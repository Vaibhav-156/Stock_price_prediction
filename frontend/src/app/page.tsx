"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { api, Signal, PriceData, StockDetail, LiveQuote, IntradaySignal, StockSentiment, isIndianMarketOpen } from "@/lib/api";
import CandlestickChart from "@/components/CandlestickChart";
import SignalDashboard from "@/components/SignalDashboard";
import RiskPanel from "@/components/RiskPanel";
import BacktestPanel from "@/components/BacktestPanel";
import ModelPanel from "@/components/ModelPanel";
import StockDetailPanel from "@/components/StockDetailPanel";
import IntradayPanel from "@/components/IntradayPanel";

type Tab = "dashboard" | "signals" | "backtest" | "model";

const NIFTY50 = [
  "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
  "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
  "LT.NS", "HCLTECH.NS", "KOTAKBANK.NS", "AXISBANK.NS", "ASIANPAINT.NS",
  "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "TRENT.NS", "BAJAJFINSV.NS",
  "NTPC.NS", "TATASTEEL.NS", "POWERGRID.NS", "WIPRO.NS", "M&M.NS",
  "ULTRACEMCO.NS", "NESTLEIND.NS", "ONGC.NS", "JSWSTEEL.NS", "ADANIENT.NS",
  "ADANIPORTS.NS", "COALINDIA.NS", "GRASIM.NS", "TECHM.NS", "BAJAJ-AUTO.NS",
  "INDUSINDBK.NS", "BRITANNIA.NS", "HINDALCO.NS", "CIPLA.NS", "DRREDDY.NS",
  "DIVISLAB.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS",
  "SBILIFE.NS", "HDFCLIFE.NS", "TATACONSUM.NS", "BPCL.NS", "SHRIRAMFIN.NS",
  "LICI.NS",
];

const EXTRA_STOCKS = [
  // Silver / Gold / Commodity ETFs
  "TATAGOLD.NS", "SILVERBEES.NS", "GOLDBEES.NS",
  "SILVERETF.NS", "SILVER1.NS", "SILVERM.NS",
  "GOLDSHARE.NS", "CPSEETF.NS", "BANKETF.NS", "ITBEES.NS",
  "NIFTYBEES.NS", "JUNIORBEES.NS", "LIQUIDBEES.NS",
  // Popular mid/small cap performers
  "IRFC.NS", "ZOMATO.NS", "JIOFIN.NS", "HAL.NS",
  "BEL.NS", "IRCTC.NS", "PAYTM.NS", "DMART.NS",
  "TATAELXSI.NS", "POLYCAB.NS", "PERSISTENT.NS", "COFORGE.NS",
  "DEEPAKNTR.NS", "PIIND.NS", "ABCAPITAL.NS", "CANBK.NS",
  "PNB.NS", "IOB.NS", "RECLTD.NS", "PFC.NS",
  "NHPC.NS", "SJVN.NS", "TATAPOWER.NS", "ADANIGREEN.NS",
  "ADANIPOWER.NS", "SUZLON.NS", "IDEA.NS",
];

const ALL_STOCKS = [...NIFTY50, ...EXTRA_STOCKS];

export default function Home() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [symbols] = useState<string[]>(ALL_STOCKS);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [priceData, setPriceData] = useState<PriceData[]>([]);
  const [stockDetail, setStockDetail] = useState<StockDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [riskData, setRiskData] = useState<any>(null);
  const [modelData, setModelData] = useState<any>(null);
  const [features, setFeatures] = useState<any[]>([]);
  const [backtestResult, setBacktestResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);

  const [liveQuotes, setLiveQuotes] = useState<Record<string, LiveQuote>>({});
  const [quotesLoading, setQuotesLoading] = useState(false);
  const [marketOpen, setMarketOpen] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<string>("");
  const refreshTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  // Intraday signals
  const [intradaySignals, setIntradaySignals] = useState<IntradaySignal[]>([]);
  const [intradayLoading, setIntradayLoading] = useState(false);
  const [intradayMeta, setIntradayMeta] = useState<{
    timestamp?: string;
    buyCount: number;
    sellCount: number;
    holdCount: number;
  }>({ buyCount: 0, sellCount: 0, holdCount: 0 });

  // News sentiment per selected stock
  const [stockSentiment, setStockSentiment] = useState<StockSentiment | null>(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);

  // Fetch live quotes for all stocks
  const fetchQuotes = useCallback(async () => {
    setQuotesLoading(true);
    try {
      const resp = await api.getQuotes(ALL_STOCKS);
      const map: Record<string, LiveQuote> = {};
      for (const q of resp.quotes) {
        map[q.symbol] = q;
      }
      setLiveQuotes(map);
      setMarketOpen(resp.market_open);
      setLastRefresh(
        new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
      );
    } catch (e) {
      console.error("Quote fetch failed:", e);
    } finally {
      setQuotesLoading(false);
    }
  }, []);

  // Fetch intraday signals
  const fetchIntraday = useCallback(async () => {
    setIntradayLoading(true);
    try {
      const resp = await api.getIntradaySignals();
      setIntradaySignals(resp.signals);
      setIntradayMeta({
        timestamp: resp.timestamp,
        buyCount: resp.buy_count,
        sellCount: resp.sell_count,
        holdCount: resp.hold_count,
      });
    } catch (e) {
      console.error("Intraday signals failed:", e);
    } finally {
      setIntradayLoading(false);
    }
  }, []);

  // Health check + initial quotes fetch + intraday signals
  useEffect(() => {
    api.health()
      .then(() => { setApiOnline(true); fetchQuotes(); fetchIntraday(); })
      .catch(() => setApiOnline(false));
  }, [fetchQuotes, fetchIntraday]);

  // Auto-refresh every 5s (quotes) + 30s (intraday) during Indian market hours
  const intradayTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    function startRefresh() {
      if (refreshTimer.current) clearInterval(refreshTimer.current);
      if (intradayTimer.current) clearInterval(intradayTimer.current);
      if (isIndianMarketOpen()) {
        refreshTimer.current = setInterval(() => {
          fetchQuotes();
        }, 5000);
        // Refresh intraday signals every 30s during market hours
        intradayTimer.current = setInterval(() => {
          fetchIntraday();
        }, 30000);
      }
    }

    startRefresh();

    // Re-check market hours every 60s (to start/stop refresh at open/close)
    const marketCheck = setInterval(() => {
      const open = isIndianMarketOpen();
      setMarketOpen(open);
      if (open && !refreshTimer.current) {
        startRefresh();
      } else if (!open && refreshTimer.current) {
        clearInterval(refreshTimer.current);
        refreshTimer.current = null;
        if (intradayTimer.current) { clearInterval(intradayTimer.current); intradayTimer.current = null; }
      }
    }, 60000);

    return () => {
      if (refreshTimer.current) clearInterval(refreshTimer.current);
      if (intradayTimer.current) clearInterval(intradayTimer.current);
      clearInterval(marketCheck);
    };
  }, [fetchQuotes, fetchIntraday]);

  // Fetch sentiment for a symbol
  const fetchSentiment = useCallback(async (symbol: string) => {
    setSentimentLoading(true);
    setStockSentiment(null);
    try {
      const data = await api.getStockNews(symbol);
      setStockSentiment(data);
    } catch (e) {
      console.error("Sentiment fetch failed:", e);
    } finally {
      setSentimentLoading(false);
    }
  }, []);

  // Fetch price + detail when symbol selected
  useEffect(() => {
    if (!selectedSymbol) return;
    setDetailLoading(true);
    setPriceData([]);
    setStockDetail(null);

    Promise.allSettled([
      api.getPrices(selectedSymbol, 180),
      api.getStockDetail(selectedSymbol),
    ]).then(([priceRes, detailRes]) => {
      if (priceRes.status === "fulfilled") setPriceData(priceRes.value.data);
      if (detailRes.status === "fulfilled") setStockDetail(detailRes.value);
      setDetailLoading(false);
    });

    // Fetch news sentiment in parallel
    fetchSentiment(selectedSymbol);
  }, [selectedSymbol, fetchSentiment]);

  // Auto-select first stock
  useEffect(() => {
    if (!selectedSymbol) setSelectedSymbol(ALL_STOCKS[0]);
  }, [selectedSymbol]);

  // Fetch signals
  const fetchSignals = useCallback(async () => {
    setLoading(true);
    setError(null);
    setStatus("Generating signals...");
    try {
      const result = await api.getSignals(symbols.slice(0, 15));
      setSignals(result);
      setStatus(`${result.length} signals generated`);
    } catch (e: any) {
      setError(e.message);
      setStatus("");
    } finally {
      setLoading(false);
    }
  }, [symbols]);

  // Auto-fetch signals on mount so accuracy panel populates immediately
  const signalsFetchedRef = useRef(false);
  useEffect(() => {
    if (!signalsFetchedRef.current) {
      signalsFetchedRef.current = true;
      fetchSignals();
    }
  }, [fetchSignals]);

  // Risk
  useEffect(() => {
    api.getPortfolioRisk().then(setRiskData).catch(() => {});
  }, [signals]);

  // Train
  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    setStatus("Training models — this may take a few minutes...");
    try {
      const res = await api.trainModels(symbols.slice(0, 30));
      if (res.status === "already_training") {
        setStatus("Training already in progress. Please wait...");
      } else {
        setStatus("Training started in background. Polling for completion...");
        // Poll training status every 15s
        const poll = setInterval(async () => {
          try {
            const st = await api.getTrainStatus();
            if (st.status === "completed") {
              clearInterval(poll);
              setStatus("✅ Training completed! Refreshing model data...");
              await fetchModelData();
            } else if (st.status === "failed") {
              clearInterval(poll);
              setStatus(`❌ Training failed: ${st.last_error || "unknown error"}`);
            }
          } catch {
            // ignore polling errors
          }
        }, 15_000);
        // Safety: stop polling after 20 minutes
        setTimeout(() => clearInterval(poll), 20 * 60 * 1000);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // Model metrics
  const fetchModelData = async () => {
    try {
      const [metrics, feats] = await Promise.all([
        api.getModelMetrics(),
        api.getFeatureImportance(),
      ]);
      setModelData(metrics);
      setFeatures(feats);
    } catch (e) {
      console.error(e);
    }
  };

  // Backtest
  const handleBacktest = async () => {
    setLoading(true);
    setStatus("Running walk-forward backtest...");
    try {
      const result = await api.runBacktest({
        symbols: symbols.slice(0, 10),
        start_date: "2022-01-01",
        end_date: "2025-12-31",
        initial_capital: 7500000,
        threshold: 0.65,
      });
      setBacktestResult(result);
      setStatus("Backtest complete");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // Search filtered stocks
  const filteredStocks = useMemo(() => {
    if (!searchQuery.trim()) return ALL_STOCKS;
    const q = searchQuery.toUpperCase();
    return ALL_STOCKS.filter((s) => s.replace(".NS", "").includes(q));
  }, [searchQuery]);

  const selectedSignal = signals.find((s) => s.symbol === selectedSymbol) || null;

  // Top picks: high-probability non-NEUTRAL signals sorted by probability
  const topPicks = useMemo(() => {
    return signals
      .filter((s) => s.direction !== "NEUTRAL" && s.probability >= 0.6 && s.take_profit && s.stop_loss)
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5);
  }, [signals]);

  return (
    <div className="min-h-screen flex flex-col">
      {/* ═══ HEADER ═══ */}
      <header className="border-b border-card-border bg-card-bg/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-4 py-3 flex items-center justify-between gap-4">
          {/* Brand — Logo only */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="relative w-9 h-9">
              {/* Outer glow ring */}
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-blue-500 via-cyan-400 to-emerald-400 opacity-30 blur-[6px]" />
              {/* Logo container */}
              <div className="relative w-9 h-9 bg-gradient-to-br from-blue-600 via-blue-500 to-cyan-400 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                {/* SVG chart icon */}
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-white drop-shadow-sm">
                  <path d="M3 20L8 15L12 18L16 10L21 4" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M16 4H21V9" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
                  <circle cx="8" cy="15" r="1.5" fill="currentColor" opacity="0.5"/>
                  <circle cx="12" cy="18" r="1.5" fill="currentColor" opacity="0.5"/>
                </svg>
              </div>
            </div>
          </div>

          {/* Search bar */}
          <div className="flex-1 max-w-md relative">
            <div className="flex items-center bg-slate-800 border border-card-border rounded-lg overflow-hidden">
              <svg className="w-4 h-4 text-slate-400 ml-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setSearchOpen(true);
                }}
                onFocus={() => setSearchOpen(true)}
                onBlur={() => setTimeout(() => setSearchOpen(false), 200)}
                className="flex-1 bg-transparent px-3 py-2 text-sm focus:outline-none placeholder-slate-500"
                placeholder="Search stocks..."
              />
              {searchQuery && (
                <button
                  onClick={() => { setSearchQuery(""); setSearchOpen(false); }}
                  className="px-2 text-slate-400 hover:text-slate-200"
                >
                  ✕
                </button>
              )}
            </div>

            {/* Search dropdown */}
            {searchOpen && searchQuery && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-slate-800 border border-card-border rounded-lg shadow-xl max-h-64 overflow-y-auto z-50">
                {filteredStocks.length > 0 ? (
                  filteredStocks.map((sym) => (
                    <button
                      key={sym}
                      onMouseDown={() => {
                        setSelectedSymbol(sym);
                        setSearchQuery("");
                        setSearchOpen(false);
                        setTab("dashboard");
                      }}
                      className="w-full text-left px-4 py-2.5 text-sm hover:bg-slate-700 flex items-center justify-between"
                    >
                      <span className="font-semibold">{sym.replace(".NS", "")}</span>
                      <span className="text-xs text-slate-400">NSE</span>
                    </button>
                  ))
                ) : (
                  <div className="px-4 py-3 text-sm text-slate-400">No stocks found</div>
                )}
              </div>
            )}
          </div>

          {/* Status indicator */}
          <div className="flex items-center gap-3 shrink-0">
            <div className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 border ${
              marketOpen
                ? "bg-profit/10 border-profit/30"
                : "bg-red-500/10 border-red-500/30"
            }`}>
              <div className={`w-1.5 h-1.5 rounded-full ${marketOpen ? "bg-profit animate-pulse" : "bg-red-500"}`} />
              <span className={`text-xs font-medium ${marketOpen ? "text-profit" : "text-red-400"}`}>
                {marketOpen ? "Market OPEN" : "Market CLOSED"}
              </span>
            </div>
            <div className={`w-2 h-2 rounded-full ${
              apiOnline === true ? "bg-profit" : apiOnline === false ? "bg-loss" : "bg-yellow-500"
            }`} />
            <span className="text-xs text-slate-400 hidden sm:inline">
              {apiOnline === true ? "API Online" : apiOnline === false ? "API Offline" : "Checking..."}
            </span>
          </div>
        </div>
      </header>

      {/* ═══ CONTROLS BAR ═══ */}
      <div className="border-b border-card-border bg-slate-900/50 px-4 py-2.5">
        <div className="max-w-[1600px] mx-auto flex flex-wrap items-center gap-3">
          {/* Tabs */}
          <div className="flex gap-1 bg-slate-800 rounded-lg p-0.5">
            {(["dashboard", "signals", "model"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => {
                  setTab(t);
                  if (t === "model") fetchModelData();
                  if (t === "signals" && signals.length === 0) fetchSignals();
                }}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  tab === t ? "bg-blue-600 text-white" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          <div className="flex-1" />

          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={fetchSignals}
              disabled={loading}
              className="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
            >
              {loading && tab === "signals" ? "Loading..." : "Get Signals"}
            </button>
            <button
              onClick={handleTrain}
              disabled={loading}
              className="px-4 py-1.5 bg-purple-600 hover:bg-purple-700 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
            >
              Train Models
            </button>
            {tab === "backtest" && (
              <button
                onClick={handleBacktest}
                disabled={loading}
                className="px-4 py-1.5 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
              >
                {loading ? "Running..." : "Run Backtest"}
              </button>
            )}
          </div>
        </div>

        {/* Status / error */}
        {(status || error) && (
          <div className="max-w-[1600px] mx-auto mt-2">
            {error && (
              <div className="text-xs text-loss bg-loss/10 border border-loss/20 rounded px-3 py-1.5">
                {error}
              </div>
            )}
            {status && !error && <div className="text-xs text-blue-400">{status}</div>}
          </div>
        )}
      </div>

      {/* ═══ MAIN CONTENT ═══ */}
      <main className="flex-1 max-w-[1600px] mx-auto w-full px-4 py-4">
        {/* ─── Dashboard Tab ─── */}
        {tab === "dashboard" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            {/* Stock list (left sidebar) */}
            <div className="lg:col-span-2">
              <div className="card p-3 max-h-[calc(100vh-200px)] overflow-y-auto">
                <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-2 px-1">
                  Stocks ({filteredStocks.length})
                </h4>
                <div className="space-y-0.5">
                  {filteredStocks.map((sym) => (
                    <button
                      key={sym}
                      onClick={() => setSelectedSymbol(sym)}
                      className={`w-full text-left px-2 py-1.5 rounded-md text-sm transition-colors ${
                        selectedSymbol === sym
                          ? "bg-blue-600/20 text-blue-400 font-semibold"
                          : "text-slate-300 hover:bg-slate-800"
                      }`}
                    >
                      {sym.replace(".NS", "")}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Chart + details (center) */}
            <div className="lg:col-span-7 space-y-4">
              {selectedSymbol && priceData.length > 0 && (
                <CandlestickChart
                  data={priceData}
                  symbol={selectedSymbol}
                  stopLoss={selectedSignal?.stop_loss}
                  takeProfit={selectedSignal?.take_profit}
                  detail={stockDetail}
                  showPrediction={true}
                />
              )}

              {selectedSymbol && priceData.length === 0 && !detailLoading && (
                <div className="card text-center py-12">
                  <p className="text-slate-400">
                    {apiOnline === false
                      ? "Start the backend API to load data"
                      : "Loading chart data..."}
                  </p>
                </div>
              )}

              {detailLoading && priceData.length === 0 && (
                <div className="card h-[460px] animate-pulse flex items-center justify-center">
                  <div className="text-slate-500">Loading chart...</div>
                </div>
              )}

              {/* Signal detail (if available) */}
              {selectedSignal && selectedSignal.direction !== "NEUTRAL" && (
                <div className="card grid grid-cols-2 md:grid-cols-4 gap-4">
                  <DetailItem label="Ensemble Probability" value={`${(selectedSignal.probability * 100).toFixed(1)}%`} />
                  <DetailItem label="XGBoost" value={selectedSignal.xgb_probability ? `${(selectedSignal.xgb_probability * 100).toFixed(1)}%` : "—"} />
                  <DetailItem label="LSTM" value={selectedSignal.lstm_probability ? `${(selectedSignal.lstm_probability * 100).toFixed(1)}%` : "—"} />
                  <DetailItem label="ATR" value={selectedSignal.atr ? `₹${selectedSignal.atr.toFixed(2)}` : "—"} />
                  <DetailItem label="Stop Loss" value={selectedSignal.stop_loss ? `₹${selectedSignal.stop_loss.toFixed(2)}` : "—"} />
                  <DetailItem label="Take Profit" value={selectedSignal.take_profit ? `₹${selectedSignal.take_profit.toFixed(2)}` : "—"} />
                  <DetailItem label="Position Size" value={`${selectedSignal.suggested_position_size ?? 0} shares`} />
                  <DetailItem label="Regime" value={selectedSignal.regime ?? "—"} />
                </div>
              )}

              {/* Intraday Signals — Buy/Sell/Hold with Target & Stop Loss */}
              <IntradayPanel
                signals={intradaySignals}
                loading={intradayLoading}
                onSelectSymbol={setSelectedSymbol}
                timestamp={intradayMeta.timestamp}
                buyCount={intradayMeta.buyCount}
                sellCount={intradayMeta.sellCount}
                holdCount={intradayMeta.holdCount}
              />
            </div>

            {/* Stock detail panel (right sidebar) */}
            <div className="lg:col-span-3 space-y-4">
              <StockDetailPanel
                detail={stockDetail}
                loading={detailLoading}
                liveQuote={selectedSymbol ? liveQuotes[selectedSymbol] ?? null : null}
                signal={selectedSignal ?? null}
                sentiment={stockSentiment}
                sentimentLoading={sentimentLoading}
                onRefreshSentiment={() => selectedSymbol && fetchSentiment(selectedSymbol)}
              />

              {/* Top Picks — high-probability signals */}
              {topPicks.length > 0 && (
                <div className="card p-3">
                  <h4 className="text-xs text-amber-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <span>🎯</span> Top Picks — Invest
                  </h4>
                  <div className="space-y-2">
                    {topPicks.map((s) => {
                      const priceDelta = s.take_profit && s.stop_loss ? ((s.take_profit - (s.stop_loss + (s.take_profit - s.stop_loss) * 0.4)) / (s.stop_loss + (s.take_profit - s.stop_loss) * 0.4) * 100) : null;
                      return (
                        <button
                          key={s.symbol}
                          onClick={() => setSelectedSymbol(s.symbol)}
                          className="w-full text-left bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/40 rounded-lg px-3 py-2.5 transition-colors"
                        >
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="text-sm font-semibold text-white">{s.symbol.replace('.NS', '')}</span>
                            <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${
                              s.direction === 'LONG' ? 'bg-green-900/40 text-green-400' : 'bg-red-900/40 text-red-400'
                            }`}>
                              {s.direction}
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-slate-400">Prob</span>
                            <span className="font-mono text-blue-400 font-semibold">{(s.probability * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex items-center justify-between text-xs mt-1">
                            <span className="text-green-400/70">Target</span>
                            <span className="font-mono text-green-400">{s.take_profit ? `₹${s.take_profit.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : '—'}</span>
                          </div>
                          <div className="flex items-center justify-between text-xs mt-0.5">
                            <span className="text-red-400/70">Stop Loss</span>
                            <span className="font-mono text-red-400">{s.stop_loss ? `₹${s.stop_loss.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : '—'}</span>
                          </div>
                          {s.take_profit && s.stop_loss && (() => {
                            const approxPrice = s.stop_loss + (s.take_profit - s.stop_loss) * 0.4;
                            const reward = Math.abs(s.take_profit - approxPrice);
                            const risk = Math.abs(approxPrice - s.stop_loss);
                            const rr = risk > 0 ? (reward / risk).toFixed(1) : '—';
                            return (
                              <div className="flex items-center justify-between text-xs mt-1 pt-1 border-t border-slate-700/30">
                                <span className="text-slate-500">R:R</span>
                                <span className="font-mono text-amber-400">1:{rr}</span>
                              </div>
                            );
                          })()}
                          <div className="flex items-center justify-between text-xs mt-0.5">
                            <span className="text-slate-500">Confidence</span>
                            <span className={`font-semibold ${
                              s.confidence_level === 'HIGH' ? 'text-green-400' : s.confidence_level === 'MEDIUM' ? 'text-yellow-400' : 'text-slate-400'
                            }`}>{s.confidence_level}</span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ─── Signals Tab ─── */}
        {tab === "signals" && (
          <div className="space-y-4">
              <SignalDashboard
                signals={signals}
                onSelect={(sym) => {
                  setSelectedSymbol(sym);
                  setTab("dashboard");
                }}
                selectedSymbol={selectedSymbol}
              />
          </div>
        )}

        {/* ─── Backtest Tab ─── */}
        {tab === "backtest" && (
          <div className="space-y-4">
            <BacktestPanel
              equityCurve={backtestResult?.equity_curve || []}
              metrics={backtestResult?.metrics || null}
              trades={backtestResult?.trades || []}
            />
          </div>
        )}

        {/* ─── Model Tab ─── */}
        {tab === "model" && (
          <ModelPanel modelData={modelData} features={features} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-card-border py-3 px-4 text-center text-xs text-slate-500">
        NSE India &mdash; Not financial advice
      </footer>
    </div>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-slate-400 mb-0.5">{label}</div>
      <div className="text-sm font-semibold font-mono">{value}</div>
    </div>
  );
}
