"use client";

import clsx from "clsx";

interface RiskData {
  total_capital: number;
  current_equity: number;
  cash: number;
  exposure_pct: number;
  drawdown_pct: number;
  peak_capital: number;
  open_positions: number;
  max_new_positions: number;
}

interface Props {
  data: RiskData | null;
  loading: boolean;
}

export default function RiskPanel({ data, loading }: Props) {
  if (loading) {
    return (
      <div className="card animate-pulse">
        <div className="h-6 bg-slate-700 rounded w-40 mb-4" />
        <div className="space-y-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-4 bg-slate-700 rounded w-full" />
          ))}
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold mb-2">Risk Analytics</h3>
        <p className="text-slate-400">No portfolio data available</p>
      </div>
    );
  }

  const metrics = [
    {
      label: "Total Capital",
      value: formatCurrency(data.total_capital),
      color: "text-slate-200",
    },
    {
      label: "Current Equity",
      value: formatCurrency(data.current_equity),
      color: data.current_equity >= data.total_capital ? "text-profit" : "text-loss",
    },
    {
      label: "Cash Available",
      value: formatCurrency(data.cash),
      color: "text-blue-400",
    },
    {
      label: "Exposure",
      value: `${data.exposure_pct.toFixed(1)}%`,
      color: data.exposure_pct > 70 ? "text-loss" : data.exposure_pct > 50 ? "text-yellow-400" : "text-profit",
      bar: data.exposure_pct,
      barMax: 100,
      barColor: data.exposure_pct > 70 ? "bg-loss" : data.exposure_pct > 50 ? "bg-yellow-500" : "bg-profit",
    },
    {
      label: "Drawdown",
      value: `${data.drawdown_pct.toFixed(2)}%`,
      color: data.drawdown_pct > 10 ? "text-loss" : data.drawdown_pct > 5 ? "text-yellow-400" : "text-profit",
      bar: data.drawdown_pct,
      barMax: 20,
      barColor: data.drawdown_pct > 10 ? "bg-loss" : data.drawdown_pct > 5 ? "bg-yellow-500" : "bg-profit",
    },
    {
      label: "Peak Capital",
      value: formatCurrency(data.peak_capital),
      color: "text-slate-300",
    },
    {
      label: "Open Positions",
      value: data.open_positions.toString(),
      color: "text-blue-400",
    },
    {
      label: "Max New Entries",
      value: data.max_new_positions.toString(),
      color: data.max_new_positions === 0 ? "text-loss" : "text-profit",
    },
  ];

  return (
    <div className="card">
      <h3 className="text-lg font-semibold mb-4">Risk Analytics</h3>
      <div className="space-y-3">
        {metrics.map((m) => (
          <div key={m.label}>
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-400">{m.label}</span>
              <span className={clsx("font-mono text-sm font-semibold", m.color)}>
                {m.value}
              </span>
            </div>
            {m.bar !== undefined && (
              <div className="mt-1 w-full bg-slate-700 rounded-full h-1.5">
                <div
                  className={clsx("h-1.5 rounded-full transition-all", m.barColor)}
                  style={{ width: `${Math.min((m.bar / (m.barMax || 1)) * 100, 100)}%` }}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function formatCurrency(n: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(n);
}
