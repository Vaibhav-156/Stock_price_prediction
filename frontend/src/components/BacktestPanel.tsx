"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType } from "lightweight-charts";

interface EquityPoint {
  date: string;
  equity: number;
  drawdown_pct: number;
}

interface Metrics {
  total_return_pct: number;
  cagr: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  win_rate: number;
  profit_factor: number;
  expectancy: number;
  total_trades: number;
  avg_holding_days: number;
}

interface Props {
  equityCurve: EquityPoint[];
  metrics: Metrics | null;
  trades: any[];
}

export default function BacktestPanel({ equityCurve, metrics, trades }: Props) {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartRef.current || equityCurve.length === 0) return;

    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 300,
      layout: {
        background: { type: ColorType.Solid, color: "#1e293b" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "#334155" },
        horzLines: { color: "#334155" },
      },
      rightPriceScale: { borderColor: "#334155" },
      timeScale: { borderColor: "#334155" },
    });

    const series = chart.addAreaSeries({
      lineColor: "#3b82f6",
      topColor: "#3b82f640",
      bottomColor: "#3b82f605",
      lineWidth: 2,
    });

    series.setData(
      equityCurve.map((e) => ({
        time: e.date as any,
        value: e.equity,
      }))
    );

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartRef.current) {
        chart.applyOptions({ width: chartRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [equityCurve]);

  return (
    <div className="card space-y-4">
      <h3 className="text-lg font-semibold">Backtest Results</h3>

      {/* Metrics grid */}
      {metrics && (
        <div className="grid grid-cols-3 sm:grid-cols-5 gap-3">
          <MetricCard label="Return" value={`${metrics.total_return_pct.toFixed(1)}%`}
            color={metrics.total_return_pct >= 0 ? "text-profit" : "text-loss"} />
          <MetricCard label="CAGR" value={`${metrics.cagr.toFixed(1)}%`}
            color={metrics.cagr >= 0 ? "text-profit" : "text-loss"} />
          <MetricCard label="Sharpe" value={metrics.sharpe_ratio.toFixed(2)}
            color={metrics.sharpe_ratio >= 1 ? "text-profit" : "text-yellow-400"} />
          <MetricCard label="Max DD" value={`${metrics.max_drawdown_pct.toFixed(1)}%`}
            color={metrics.max_drawdown_pct > 15 ? "text-loss" : "text-yellow-400"} />
          <MetricCard label="Win Rate" value={`${metrics.win_rate.toFixed(1)}%`}
            color={metrics.win_rate >= 55 ? "text-profit" : "text-loss"} />
          <MetricCard label="Profit Factor" value={metrics.profit_factor.toFixed(2)}
            color={metrics.profit_factor >= 1.5 ? "text-profit" : "text-yellow-400"} />
          <MetricCard label="Expectancy" value={`₹${metrics.expectancy.toFixed(0)}`}
            color={metrics.expectancy > 0 ? "text-profit" : "text-loss"} />
          <MetricCard label="Total Trades" value={metrics.total_trades.toString()}
            color="text-blue-400" />
          <MetricCard label="Avg Hold" value={`${metrics.avg_holding_days.toFixed(1)}d`}
            color="text-slate-300" />
        </div>
      )}

      {/* Equity chart */}
      {equityCurve.length > 0 && (
        <div>
          <h4 className="text-sm text-slate-400 mb-2">Equity Curve</h4>
          <div ref={chartRef} />
        </div>
      )}

      {/* Trade log */}
      {trades.length > 0 && (
        <div>
          <h4 className="text-sm text-slate-400 mb-2">
            Recent Trades ({trades.length} total)
          </h4>
          <div className="overflow-x-auto max-h-60 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card-bg">
                <tr className="text-slate-500 border-b border-card-border">
                  <th className="text-left py-1.5 px-1">Symbol</th>
                  <th className="text-left py-1.5 px-1">Entry</th>
                  <th className="text-left py-1.5 px-1">Exit</th>
                  <th className="text-left py-1.5 px-1">PnL</th>
                  <th className="text-left py-1.5 px-1">Return</th>
                  <th className="text-left py-1.5 px-1">Hold</th>
                  <th className="text-left py-1.5 px-1">Reason</th>
                </tr>
              </thead>
              <tbody>
                {trades.slice(-50).reverse().map((t, i) => (
                  <tr key={i} className="border-b border-card-border/30">
                    <td className="py-1 px-1 font-semibold">{t.symbol}</td>
                    <td className="py-1 px-1 font-mono">₹{t.entry_price?.toFixed(2)}</td>
                    <td className="py-1 px-1 font-mono">₹{t.exit_price?.toFixed(2)}</td>
                    <td className={`py-1 px-1 font-mono ${t.pnl >= 0 ? "text-profit" : "text-loss"}`}>
                      ₹{t.pnl?.toFixed(0)}
                    </td>
                    <td className={`py-1 px-1 font-mono ${t.return_pct >= 0 ? "text-profit" : "text-loss"}`}>
                      {t.return_pct?.toFixed(1)}%
                    </td>
                    <td className="py-1 px-1">{t.holding_days}d</td>
                    <td className="py-1 px-1 text-slate-400">{t.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-slate-800/50 rounded-lg p-2.5 text-center">
      <div className="text-xs text-slate-400 mb-1">{label}</div>
      <div className={`text-sm font-semibold font-mono ${color}`}>{value}</div>
    </div>
  );
}
