"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType } from "lightweight-charts";
import type { PriceData, StockDetail } from "@/lib/api";

interface Props {
  data: PriceData[];
  symbol: string;
  stopLoss?: number | null;
  takeProfit?: number | null;
  detail?: StockDetail | null;
  showPrediction?: boolean;
}

export default function CandlestickChart({
  data,
  symbol,
  stopLoss,
  takeProfit,
  detail,
  showPrediction,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 420,
      layout: {
        background: { type: ColorType.Solid, color: "#1e293b" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "#334155" },
        horzLines: { color: "#334155" },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: "#334155" },
      timeScale: { borderColor: "#334155", timeVisible: false },
    });

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderDownColor: "#ef4444",
      borderUpColor: "#22c55e",
      wickDownColor: "#ef4444",
      wickUpColor: "#22c55e",
    });

    const candleData = data.map((d) => ({
      time: d.date as any,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));
    candleSeries.setData(candleData);

    // Volume
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
    });
    chart.priceScale("vol").applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });

    const volData = data.map((d) => ({
      time: d.date as any,
      value: d.volume,
      color: d.close >= d.open ? "#22c55e40" : "#ef444440",
    }));
    volumeSeries.setData(volData);

    // SMA lines (calculated from price data)
    if (data.length >= 20) {
      const sma20Data = calcSMA(data, 20);
      const sma20Series = chart.addLineSeries({
        color: "#f59e0b",
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      sma20Series.setData(sma20Data);
    }

    if (data.length >= 50) {
      const sma50Data = calcSMA(data, 50);
      const sma50Series = chart.addLineSeries({
        color: "#3b82f6",
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      sma50Series.setData(sma50Data);
    }

    // Prediction zone overlay
    if (showPrediction && data.length > 20) {
      const lastPrice = data[data.length - 1].close;
      const recentPrices = data.slice(-20);
      const avgReturn =
        recentPrices.reduce((sum, d) => sum + (d.close - d.open) / d.open, 0) / recentPrices.length;
      const volatility = Math.sqrt(
        recentPrices.reduce(
          (sum, d) => sum + Math.pow((d.close - d.open) / d.open - avgReturn, 2),
          0
        ) / recentPrices.length
      );

      // Predicted 5-day path using momentum extrapolation
      const predCenter = lastPrice * (1 + avgReturn * 5);
      const predUpper = lastPrice * (1 + avgReturn * 5 + volatility * 2 * Math.sqrt(5));
      const predLower = lastPrice * (1 + avgReturn * 5 - volatility * 2 * Math.sqrt(5));

      candleSeries.createPriceLine({
        price: predCenter,
        color: "#8b5cf6",
        lineWidth: 2,
        lineStyle: 2,
        axisLabelVisible: true,
        title: `5D Pred: ₹${predCenter.toFixed(0)}`,
      });
      candleSeries.createPriceLine({
        price: predUpper,
        color: "#8b5cf640",
        lineWidth: 1,
        lineStyle: 3,
        axisLabelVisible: false,
        title: "",
      });
      candleSeries.createPriceLine({
        price: predLower,
        color: "#8b5cf640",
        lineWidth: 1,
        lineStyle: 3,
        axisLabelVisible: false,
        title: "",
      });
    }

    // Stop loss / take profit lines
    if (stopLoss) {
      candleSeries.createPriceLine({
        price: stopLoss,
        color: "#ef4444",
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
        title: "Stop Loss",
      });
    }
    if (takeProfit) {
      candleSeries.createPriceLine({
        price: takeProfit,
        color: "#22c55e",
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
        title: "Take Profit",
      });
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [data, stopLoss, takeProfit, detail, showPrediction]);

  // Last bar stats
  const lastBar = data.length > 0 ? data[data.length - 1] : null;
  const prevBar = data.length > 1 ? data[data.length - 2] : null;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-semibold">{symbol.replace(".NS", "")} Chart</h3>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <span className="w-3 h-[2px] bg-yellow-500 inline-block" /> SMA 20
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-[2px] bg-blue-500 inline-block" /> SMA 50
            </span>
            {showPrediction && (
              <span className="flex items-center gap-1">
                <span className="w-3 h-[2px] bg-purple-500 inline-block" /> Prediction
              </span>
            )}
          </div>
        </div>
        <span className="text-sm text-slate-400">{data.length} bars</span>
      </div>
      <div ref={containerRef} />

      {/* OHLCV bar */}
      {lastBar && (
        <div className="mt-3 flex flex-wrap gap-4 text-xs text-slate-400">
          <span>
            O: <span className="text-slate-200 font-mono">₹{lastBar.open.toFixed(2)}</span>
          </span>
          <span>
            H: <span className="text-profit font-mono">₹{lastBar.high.toFixed(2)}</span>
          </span>
          <span>
            L: <span className="text-loss font-mono">₹{lastBar.low.toFixed(2)}</span>
          </span>
          <span>
            C: <span className="text-slate-200 font-mono">₹{lastBar.close.toFixed(2)}</span>
          </span>
          <span>
            Vol: <span className="text-blue-400 font-mono">{formatVol(lastBar.volume)}</span>
          </span>
          {prevBar && (
            <span>
              Chg:{" "}
              <span
                className={
                  lastBar.close >= prevBar.close ? "text-profit font-mono" : "text-loss font-mono"
                }
              >
                {lastBar.close >= prevBar.close ? "+" : ""}
                {(((lastBar.close - prevBar.close) / prevBar.close) * 100).toFixed(2)}%
              </span>
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function calcSMA(data: PriceData[], period: number): { time: any; value: number }[] {
  const result: { time: any; value: number }[] = [];
  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1);
    const avg = slice.reduce((sum, d) => sum + d.close, 0) / period;
    result.push({ time: data[i].date as any, value: avg });
  }
  return result;
}

function formatVol(v: number): string {
  if (v >= 10000000) return `${(v / 10000000).toFixed(1)} Cr`;
  if (v >= 100000) return `${(v / 100000).toFixed(1)} L`;
  if (v >= 1000) return `${(v / 1000).toFixed(1)} K`;
  return v.toString();
}
