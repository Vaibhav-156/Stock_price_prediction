"use client";

import { useState, useMemo } from "react";
import type { IntradaySignal } from "@/lib/api";
import clsx from "clsx";

interface Props {
  signals: IntradaySignal[];
  loading: boolean;
  onSelectSymbol?: (symbol: string) => void;
  timestamp?: string;
  buyCount?: number;
  sellCount?: number;
  holdCount?: number;
  marketOpen?: boolean;
  realtimeCount?: number;
  nextRefreshSecs?: number;
}

type FilterAction = "ALL" | "BUY" | "SELL" | "HOLD";

export default function IntradayPanel({
  signals,
  loading,
  onSelectSymbol,
  timestamp,
  buyCount = 0,
  sellCount = 0,
  holdCount = 0,
  marketOpen = false,
  realtimeCount = 0,
  nextRefreshSecs = 30,
}: Props) {
  const [filter, setFilter] = useState<FilterAction>("ALL");
  const [sortBy, setSortBy] = useState<"confidence" | "change" | "rr">("confidence");
  const [showTopPicks, setShowTopPicks] = useState(true);

  // Top 5 BUY picks by confidence
  const topBuys = useMemo(() =>
    signals
      .filter((s) => s.action === "BUY")
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5),
  [signals]);

  const filtered = useMemo(() => {
    let list = filter === "ALL" ? signals : signals.filter((s) => s.action === filter);
    list = [...list].sort((a, b) => {
      if (sortBy === "confidence") return b.confidence - a.confidence;
      if (sortBy === "change") return Math.abs(b.change_pct) - Math.abs(a.change_pct);
      if (sortBy === "rr") return b.rr_ratio - a.rr_ratio;
      return 0;
    });
    return list;
  }, [signals, filter, sortBy]);

  if (loading && signals.length === 0) {
    return (
      <div className="card animate-pulse space-y-3 p-4">
        <div className="h-6 bg-slate-700 rounded w-56" />
        <div className="space-y-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-14 bg-slate-700/60 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (signals.length === 0) {
    return (
      <div className="card text-center py-10">
        <div className="text-3xl mb-2">⚡</div>
        <p className="text-slate-400 text-sm">No intraday signals yet</p>
        <p className="text-slate-500 text-xs mt-1">Signals will appear once market data is loaded</p>
      </div>
    );
  }

  const totalSecs = marketOpen ? 30 : 180;
  const progressPct = Math.round(((totalSecs - nextRefreshSecs) / totalSecs) * 100);

  return (
    <div className="card p-0 overflow-hidden">
      {/* ── Header ───────────────────────────────────────────────── */}
      <div className="px-4 pt-4 pb-3 border-b border-card-border">
        {/* Title row */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-lg">⚡</span>
            <h3 className="text-lg font-bold">Intraday Signals</h3>

            {/* Live / Closed badge */}
            {marketOpen ? (
              <span className="flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-green-900/40 text-green-400 border border-green-500/30">
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                LIVE
              </span>
            ) : (
              <span className="flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-700/60 text-slate-400 border border-slate-600/30">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400" />
                CLOSED
              </span>
            )}

            {/* Realtime data count badge */}
            {realtimeCount > 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-900/40 text-blue-400 border border-blue-500/20">
                {realtimeCount} 5m feeds
              </span>
            )}
          </div>

          {/* Action counters */}
          <div className="flex items-center gap-2">
            <span className="text-xs px-2 py-0.5 rounded bg-green-900/30 text-green-400 font-semibold">{buyCount} Buy</span>
            <span className="text-xs px-2 py-0.5 rounded bg-red-900/30 text-red-400 font-semibold">{sellCount} Sell</span>
            <span className="text-xs px-2 py-0.5 rounded bg-slate-700/50 text-slate-400 font-semibold">{holdCount} Hold</span>
          </div>
        </div>

        {/* Refresh countdown bar */}
        <div className="flex items-center gap-2 mb-2">
          <div className="flex-1 h-1 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={clsx(
                "h-full rounded-full transition-all duration-1000",
                marketOpen ? "bg-green-500" : "bg-blue-500"
              )}
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <span className="text-[10px] text-slate-500 tabular-nums w-20 text-right">
            {loading ? "Refreshing..." : `Next in ${nextRefreshSecs}s`}
          </span>
          {timestamp && (
            <span className="text-[10px] text-slate-500">
              · {new Date(timestamp).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })} IST
            </span>
          )}
        </div>

        {/* Filter + sort */}
        <div className="flex items-center gap-2">
          <div className="flex gap-0.5 bg-slate-800 rounded-lg p-0.5">
            {(["ALL", "BUY", "SELL", "HOLD"] as FilterAction[]).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={clsx(
                  "px-2.5 py-1 rounded-md text-xs font-medium transition-colors",
                  filter === f
                    ? f === "BUY"
                      ? "bg-green-600 text-white"
                      : f === "SELL"
                      ? "bg-red-600 text-white"
                      : f === "HOLD"
                      ? "bg-slate-600 text-white"
                      : "bg-blue-600 text-white"
                    : "text-slate-400 hover:text-slate-200"
                )}
              >
                {f}
              </button>
            ))}
          </div>
          <div className="flex-1" />
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="text-xs bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-slate-300"
          >
            <option value="confidence">Sort: Confidence</option>
            <option value="change">Sort: Change %</option>
            <option value="rr">Sort: R:R Ratio</option>
          </select>
        </div>
      </div>

      {/* ── Model Top Picks (always visible) ─────────────────────── */}
      {topBuys.length > 0 && (
        <div className="border-b border-card-border">
          <button
            onClick={() => setShowTopPicks((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-2 bg-gradient-to-r from-amber-950/30 to-transparent hover:from-amber-900/30 transition-colors"
          >
            <span className="text-xs font-semibold text-amber-400 flex items-center gap-1.5">
              <span>🎯</span>
              Model Top BUY Picks — Intraday
              <span className="text-slate-500 font-normal">(by confidence)</span>
            </span>
            <svg
              className={clsx("w-3.5 h-3.5 text-slate-500 transition-transform", showTopPicks ? "rotate-0" : "-rotate-90")}
              fill="none" viewBox="0 0 24 24" stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {showTopPicks && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2 px-4 pb-3 pt-1">
              {topBuys.map((sig) => {
                const confPct = Math.round(sig.confidence * 100);
                const upside = (((sig.intraday_target - sig.current_price) / sig.current_price) * 100).toFixed(1);
                return (
                  <button
                    key={sig.symbol}
                    onClick={() => onSelectSymbol?.(sig.symbol)}
                    className="bg-slate-800/60 hover:bg-slate-700/70 border border-green-500/20 rounded-lg px-3 py-2.5 text-left transition-colors group"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-bold text-white group-hover:text-green-300 transition-colors">
                        {sig.symbol.replace(".NS", "")}
                      </span>
                      <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">
                        BUY
                      </span>
                    </div>
                    <div className="text-xs font-mono text-slate-300 mb-0.5">
                      ₹{sig.current_price.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className="flex items-center justify-between text-[10px] mb-1">
                      <span className="text-slate-500">Target</span>
                      <span className="text-green-400 font-mono">+{upside}%</span>
                    </div>
                    <div className="flex items-center justify-between text-[10px] mb-1.5">
                      <span className="text-slate-500">R:R</span>
                      <span className={clsx("font-mono", sig.rr_ratio >= 2 ? "text-green-400" : "text-yellow-400")}>
                        1:{sig.rr_ratio.toFixed(1)}
                      </span>
                    </div>
                    {/* Confidence bar */}
                    <div className="relative">
                      <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className={clsx(
                            "h-full rounded-full",
                            confPct >= 70 ? "bg-green-400" : confPct >= 55 ? "bg-yellow-400" : "bg-slate-500"
                          )}
                          style={{ width: `${confPct}%` }}
                        />
                      </div>
                      <span className="text-[9px] text-slate-500 mt-0.5 block text-right">{confPct}% conf</span>
                    </div>
                    {/* Realtime badge */}
                    {sig.data_source === "realtime_5m" && (
                      <span className="text-[8px] text-blue-400/70 mt-0.5 block">● 5m live</span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* ── Signal rows table ─────────────────────────────────────── */}
      <div className="max-h-[460px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-slate-900/95 backdrop-blur">
            <tr className="text-xs text-slate-500 uppercase tracking-wider">
              <th className="text-left px-4 py-2">Stock</th>
              <th className="text-center px-2 py-2">Action</th>
              <th className="text-right px-2 py-2">CMP</th>
              <th className="text-right px-2 py-2">Target</th>
              <th className="text-right px-2 py-2">Stop Loss</th>
              <th className="text-right px-2 py-2">R:R</th>
              <th className="text-right px-2 py-2">RSI</th>
              <th className="text-right px-4 py-2">Conf</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((sig) => {
              const up = sig.change_pct >= 0;
              const confPct = Math.round(sig.confidence * 100);
              const isRealtime = sig.data_source === "realtime_5m";
              return (
                <tr
                  key={sig.symbol}
                  onClick={() => onSelectSymbol?.(sig.symbol)}
                  className="border-t border-slate-800/50 hover:bg-slate-800/40 cursor-pointer transition-colors"
                >
                  {/* Symbol + Change */}
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-1.5">
                      <div>
                        <div className="font-semibold text-slate-100">{sig.symbol.replace(".NS", "")}</div>
                        <div className={clsx("text-[10px] font-mono", up ? "text-profit" : "text-loss")}>
                          {up ? "+" : ""}{sig.change_pct.toFixed(2)}%
                        </div>
                      </div>
                      {isRealtime && (
                        <span className="text-[8px] text-blue-400/60 leading-none">●5m</span>
                      )}
                    </div>
                  </td>

                  {/* Action Badge */}
                  <td className="text-center px-2 py-2.5">
                    <span
                      className={clsx(
                        "inline-block px-2.5 py-1 rounded-md text-xs font-bold tracking-wide",
                        sig.action === "BUY"
                          ? "bg-green-500/20 text-green-400 border border-green-500/30"
                          : sig.action === "SELL"
                          ? "bg-red-500/20 text-red-400 border border-red-500/30"
                          : "bg-slate-600/20 text-slate-400 border border-slate-600/30"
                      )}
                    >
                      {sig.action}
                    </span>
                  </td>

                  {/* Current Market Price */}
                  <td className="text-right px-2 py-2.5 font-mono text-slate-200">
                    ₹{sig.current_price.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>

                  {/* Target */}
                  <td className="text-right px-2 py-2.5">
                    <div className={clsx("font-mono text-sm", sig.action === "SELL" ? "text-red-400" : "text-green-400")}>
                      ₹{sig.intraday_target.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono">
                      {sig.action === "SELL" ? "" : "+"}{(
                        ((sig.intraday_target - sig.current_price) / sig.current_price) * 100
                      ).toFixed(1)}%
                    </div>
                  </td>

                  {/* Stop Loss */}
                  <td className="text-right px-2 py-2.5">
                    <div className={clsx("font-mono text-sm", sig.action === "SELL" ? "text-green-400" : "text-red-400")}>
                      ₹{sig.intraday_stop.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono">
                      {(((sig.intraday_stop - sig.current_price) / sig.current_price) * 100).toFixed(1)}%
                    </div>
                  </td>

                  {/* R:R Ratio */}
                  <td className="text-right px-2 py-2.5">
                    <span className={clsx(
                      "font-mono text-xs",
                      sig.rr_ratio >= 2 ? "text-green-400" : sig.rr_ratio >= 1.5 ? "text-yellow-400" : "text-slate-400"
                    )}>
                      1:{sig.rr_ratio.toFixed(1)}
                    </span>
                  </td>

                  {/* RSI */}
                  <td className="text-right px-2 py-2.5">
                    <span className={clsx(
                      "font-mono text-xs",
                      sig.rsi < 30 ? "text-green-400" : sig.rsi > 70 ? "text-red-400" : "text-slate-400"
                    )}>
                      {sig.rsi.toFixed(0)}
                    </span>
                  </td>

                  {/* Confidence */}
                  <td className="text-right px-4 py-2.5">
                    <div className="flex items-center justify-end gap-1.5">
                      <div className="w-12 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className={clsx(
                            "h-full rounded-full",
                            confPct >= 70 ? "bg-green-400" : confPct >= 55 ? "bg-yellow-400" : "bg-slate-500"
                          )}
                          style={{ width: `${confPct}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono text-slate-300 w-8 text-right">{confPct}%</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div className="text-center py-8 text-sm text-slate-500">
            No {filter} signals found
          </div>
        )}
      </div>

      {/* ── Footer ───────────────────────────────────────────────── */}
      <div className="px-4 py-2 border-t border-card-border bg-slate-900/50 flex items-center justify-between">
        <p className="text-[9px] text-slate-600">
          Pivot Points · VWAP · RSI · Bollinger Bands · MACD · Momentum · Sentiment — Not financial advice
        </p>
        <span className="text-[9px] text-slate-600">
          {marketOpen
            ? `Live 5m data · auto-refreshes every 30s`
            : `Last known data · auto-refreshes every 3min`}
        </span>
      </div>
    </div>
  );
}
