"use client";

import type { StockDetail, LiveQuote, Signal, PricePrediction, StockSentiment } from "@/lib/api";
import clsx from "clsx";
import SentimentPanel from "./SentimentPanel";

interface Props {
  detail: StockDetail | null;
  loading: boolean;
  liveQuote?: LiveQuote | null;
  signal?: Signal | null;
  sentiment?: StockSentiment | null;
  sentimentLoading?: boolean;
  onRefreshSentiment?: () => void;
}

export default function StockDetailPanel({ detail, loading, liveQuote, signal, sentiment, sentimentLoading, onRefreshSentiment }: Props) {
  if (loading) {
    return (
      <div className="card animate-pulse space-y-4">
        <div className="h-6 bg-slate-700 rounded w-48" />
        <div className="grid grid-cols-2 gap-3">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="h-10 bg-slate-700 rounded" />
          ))}
        </div>
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="card text-center py-12">
        <div className="text-3xl mb-3">📊</div>
        <p className="text-slate-400 text-sm">Select a stock to view real-time data</p>
      </div>
    );
  }

  return (
    <div className="card space-y-4">
      {/* Price header — prefer live quote for real-time numbers */}
      {(() => {
        const price = liveQuote?.price ?? detail.current_price;
        const change = liveQuote?.change ?? detail.change;
        const changePct = liveQuote?.change_pct ?? detail.change_pct;
        const up = change >= 0;
        return (
          <div className="flex items-start justify-between">
            <div>
              <h3 className="text-lg font-bold">{detail.symbol.replace(".NS", "")}</h3>
              <p className="text-xs text-slate-400">NSE • Nifty 50{liveQuote ? " • Live" : ""}</p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold font-mono">
                ₹{price.toLocaleString("en-IN", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
              </div>
              <div className={clsx("text-sm font-semibold font-mono", up ? "text-profit" : "text-loss")}>
                {up ? "+" : ""}₹{change.toFixed(2)} ({up ? "+" : ""}{changePct.toFixed(2)}%)
              </div>
              {liveQuote && (
                <div className="flex items-center justify-end gap-1 mt-1">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
                  </span>
                  <span className="text-[10px] text-slate-400">Real-time</span>
                </div>
              )}
            </div>
          </div>
        );
      })()}

      {/* Trend badge */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className={clsx(
          "badge",
          detail.trend === "BULLISH" ? "badge-bull" : detail.trend === "BEARISH" ? "badge-bear" : "badge-sideways"
        )}>
          {detail.trend}
        </span>
        <span className={clsx(
          "badge",
          detail.rsi_14 > 70 ? "badge-bear" : detail.rsi_14 < 30 ? "badge-bull" : "badge-sideways"
        )}>
          RSI {detail.rsi_14.toFixed(0)}
        </span>
        <span className="badge badge-low">Vol {detail.volatility_20d.toFixed(1)}%</span>
        {signal && (
          <span className={clsx(
            "badge",
            signal.confidence_level === "HIGH" ? "badge-bull" : signal.confidence_level === "MEDIUM" ? "badge-sideways" : "badge-low"
          )}>
            {signal.confidence_level} Conf
          </span>
        )}
      </div>

      {/* Signal: Target & Stop Loss */}
      {signal && signal.direction !== "NEUTRAL" && (
        <div className="space-y-2">
          <h4 className="text-xs text-slate-400 uppercase tracking-wider">Signal — {signal.direction}</h4>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-green-900/20 border border-green-700/30 rounded-lg px-3 py-2">
              <div className="text-[10px] text-green-400/70 uppercase tracking-wider">Target</div>
              <div className="text-sm font-mono font-semibold text-green-400">
                {signal.take_profit ? `₹${signal.take_profit.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "—"}
              </div>
              {signal.take_profit && detail && (
                <div className="text-[10px] text-green-400/50 font-mono">
                  +{(((signal.take_profit - detail.current_price) / detail.current_price) * 100).toFixed(1)}%
                </div>
              )}
            </div>
            <div className="bg-red-900/20 border border-red-700/30 rounded-lg px-3 py-2">
              <div className="text-[10px] text-red-400/70 uppercase tracking-wider">Stop Loss</div>
              <div className="text-sm font-mono font-semibold text-red-400">
                {signal.stop_loss ? `₹${signal.stop_loss.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "—"}
              </div>
              {signal.stop_loss && detail && (
                <div className="text-[10px] text-red-400/50 font-mono">
                  {(((signal.stop_loss - detail.current_price) / detail.current_price) * 100).toFixed(1)}%
                </div>
              )}
            </div>
          </div>
          {signal.take_profit && signal.stop_loss && detail && (() => {
            const reward = Math.abs(signal.take_profit - detail.current_price);
            const risk = Math.abs(detail.current_price - signal.stop_loss);
            const rr = risk > 0 ? (reward / risk).toFixed(1) : "—";
            return (
              <div className="flex items-center justify-between text-xs px-1">
                <span className="text-slate-500">R:R Ratio</span>
                <span className="font-mono text-slate-300">1:{rr}</span>
              </div>
            );
          })()}
        </div>
      )}

      {signal && signal.direction === "NEUTRAL" && (
        <div className="bg-slate-800/50 border border-slate-700/30 rounded-lg px-3 py-2 text-center">
          <span className="text-xs text-slate-400">NEUTRAL — No active signal</span>
        </div>
      )}

      {/* 3-Day Price Forecast */}
      {detail.predictions_3d && detail.predictions_3d.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs text-slate-400 uppercase tracking-wider flex items-center gap-2">
            <span>📈 3-Day Price Forecast</span>
          </h4>
          <div className="space-y-1.5">
            {detail.predictions_3d.map((pred) => {
              const up = pred.change_pct >= 0;
              const confPct = Math.round(pred.confidence * 100);
              const dayLabel = pred.day === 1 ? "Tomorrow" : `Day ${pred.day}`;
              const dateStr = new Date(pred.date + "T00:00:00").toLocaleDateString("en-IN", {
                weekday: "short",
                month: "short",
                day: "numeric",
              });
              return (
                <div
                  key={pred.day}
                  className={clsx(
                    "rounded-lg px-3 py-2 border",
                    up
                      ? "bg-green-900/10 border-green-700/20"
                      : "bg-red-900/10 border-red-700/20"
                  )}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div>
                      <span className="text-xs font-semibold text-slate-200">{dayLabel}</span>
                      <span className="text-[10px] text-slate-500 ml-1.5">{dateStr}</span>
                    </div>
                    <span
                      className={clsx(
                        "text-[10px] font-mono px-1.5 py-0.5 rounded",
                        confPct >= 70
                          ? "bg-green-900/40 text-green-400"
                          : confPct >= 55
                          ? "bg-yellow-900/40 text-yellow-400"
                          : "bg-slate-700/40 text-slate-400"
                      )}
                    >
                      {confPct}% conf
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-mono font-semibold text-slate-100">
                      ₹{pred.predicted_close.toLocaleString("en-IN", {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}
                    </div>
                    <div
                      className={clsx(
                        "text-xs font-mono font-semibold",
                        up ? "text-profit" : "text-loss"
                      )}
                    >
                      {up ? "▲" : "▼"} {up ? "+" : ""}
                      {pred.change_pct.toFixed(2)}%
                    </div>
                  </div>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-[10px] text-slate-500">
                      H:{" "}
                      <span className="text-green-400/80 font-mono">
                        ₹{pred.predicted_high.toLocaleString("en-IN", {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 2,
                        })}
                      </span>
                    </span>
                    <span className="text-[10px] text-slate-500">
                      L:{" "}
                      <span className="text-red-400/80 font-mono">
                        ₹{pred.predicted_low.toLocaleString("en-IN", {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 2,
                        })}
                      </span>
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
          <p className="text-[9px] text-slate-600 text-center">
            Based on trend, momentum, mean-reversion & volatility analysis
          </p>
        </div>
      )}

      {/* Key stats grid */}
      <div className="grid grid-cols-2 gap-2">
        <StatItem label="52W High" value={`₹${detail.high_52w.toLocaleString("en-IN")}`}
          highlight={detail.current_price >= detail.high_52w * 0.95} highlightColor="text-profit" />
        <StatItem label="52W Low" value={`₹${detail.low_52w.toLocaleString("en-IN")}`}
          highlight={detail.current_price <= detail.low_52w * 1.05} highlightColor="text-loss" />
        <StatItem label="SMA 20" value={`₹${detail.sma_20.toLocaleString("en-IN")}`}
          highlight={detail.current_price > detail.sma_20} highlightColor="text-profit" />
        <StatItem label="SMA 50" value={`₹${detail.sma_50.toLocaleString("en-IN")}`}
          highlight={detail.current_price > detail.sma_50} highlightColor="text-profit" />
        <StatItem label="SMA 200" value={`₹${detail.sma_200.toLocaleString("en-IN")}`}
          highlight={detail.current_price > detail.sma_200} highlightColor="text-profit" />
        <StatItem label="Avg Vol (20d)" value={formatVolume(detail.avg_volume_20d)} />
      </div>

      {/* Separator */}
      <div className="border-t border-card-border" />

      {/* Performance table */}
      <div>
        <h4 className="text-xs text-slate-400 mb-2 uppercase tracking-wider">Performance</h4>
        <div className="grid grid-cols-4 gap-1">
          <PerfItem label="1W" value={detail.perf_1w} />
          <PerfItem label="1M" value={detail.perf_1m} />
          <PerfItem label="3M" value={detail.perf_3m} />
          <PerfItem label="1Y" value={detail.perf_1y} />
        </div>
      </div>

      {/* 30-day returns mini chart */}
      {detail.daily_returns_30d.length > 0 && (
        <div>
          <h4 className="text-xs text-slate-400 mb-2 uppercase tracking-wider">30-Day Returns</h4>
          <div className="flex items-end gap-[2px] h-12">
            {detail.daily_returns_30d.map((r, i) => {
              const maxAbs = Math.max(...detail.daily_returns_30d.map(Math.abs), 0.01);
              const height = Math.max((Math.abs(r) / maxAbs) * 100, 4);
              return (
                <div
                  key={i}
                  className={clsx(
                    "flex-1 rounded-sm min-w-[2px]",
                    r >= 0 ? "bg-profit/70" : "bg-loss/70"
                  )}
                  style={{ height: `${height}%` }}
                  title={`${r >= 0 ? "+" : ""}${r.toFixed(2)}%`}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* Separator before sentiment */}
      {(sentiment || sentimentLoading) && (
        <div className="border-t border-card-border" />
      )}

      {/* News Sentiment */}
      <SentimentPanel
        sentiment={sentiment ?? null}
        loading={sentimentLoading ?? false}
        onRefresh={onRefreshSentiment}
      />
    </div>
  );
}

function StatItem({
  label,
  value,
  highlight,
  highlightColor,
}: {
  label: string;
  value: string;
  highlight?: boolean;
  highlightColor?: string;
}) {
  return (
    <div className="bg-slate-800/50 rounded-lg px-3 py-2">
      <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
      <div className={clsx("text-sm font-mono font-medium", highlight && highlightColor ? highlightColor : "text-slate-200")}>
        {value}
      </div>
    </div>
  );
}

function PerfItem({ label, value }: { label: string; value: number }) {
  const isPos = value >= 0;
  return (
    <div className="text-center bg-slate-800/50 rounded-lg py-2">
      <div className="text-[10px] text-slate-500">{label}</div>
      <div className={clsx("text-sm font-mono font-semibold", isPos ? "text-profit" : "text-loss")}>
        {isPos ? "+" : ""}{value.toFixed(1)}%
      </div>
    </div>
  );
}

function formatVolume(v: number): string {
  if (v >= 10000000) return `${(v / 10000000).toFixed(1)} Cr`;
  if (v >= 100000) return `${(v / 100000).toFixed(1)} L`;
  if (v >= 1000) return `${(v / 1000).toFixed(1)} K`;
  return v.toString();
}
