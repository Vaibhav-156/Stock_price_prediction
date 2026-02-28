"use client";

import type { Signal } from "@/lib/api";
import clsx from "clsx";

interface Props {
  signals: Signal[];
  onSelect: (symbol: string) => void;
  selectedSymbol: string | null;
}

export default function SignalDashboard({ signals, onSelect, selectedSymbol }: Props) {
  const sorted = [...signals].sort((a, b) => b.probability - a.probability);

  return (
    <div className="card">
      <h3 className="text-lg font-semibold mb-4">Signal Dashboard</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-slate-400 border-b border-card-border">
              <th className="text-left py-2 px-2">Symbol</th>
              <th className="text-left py-2 px-2">Probability</th>
              <th className="text-left py-2 px-2">Confidence</th>
              <th className="text-left py-2 px-2">Direction</th>
              <th className="text-left py-2 px-2">Stop Loss</th>
              <th className="text-left py-2 px-2">Position</th>
              <th className="text-left py-2 px-2">Risk</th>
              <th className="text-left py-2 px-2">Regime</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => (
              <tr
                key={s.symbol}
                className={clsx(
                  "border-b border-card-border/50 cursor-pointer transition-colors",
                  selectedSymbol === s.symbol
                    ? "bg-blue-500/10"
                    : "hover:bg-slate-800/50"
                )}
                onClick={() => onSelect(s.symbol)}
              >
                <td className="py-2.5 px-2 font-semibold">{s.symbol}</td>
                <td className="py-2.5 px-2">
                  <div className="flex items-center gap-2">
                    <div className="w-20 bg-slate-700 rounded-full h-2">
                      <div
                        className={clsx("prob-bar", probColor(s.probability))}
                        style={{ width: `${s.probability * 100}%` }}
                      />
                    </div>
                    <span className={clsx("font-mono text-xs", probTextColor(s.probability))}>
                      {(s.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                </td>
                <td className="py-2.5 px-2">
                  <span className={clsx("badge", badgeClass(s.confidence_level))}>
                    {s.confidence_level}
                  </span>
                </td>
                <td className="py-2.5 px-2">
                  <span
                    className={clsx(
                      "font-semibold",
                      s.direction === "LONG"
                        ? "text-profit"
                        : s.direction === "SHORT"
                        ? "text-loss"
                        : "text-neutral"
                    )}
                  >
                    {s.direction}
                  </span>
                </td>
                <td className="py-2.5 px-2 font-mono text-xs">
                  {s.stop_loss ? `₹${s.stop_loss.toFixed(2)}` : "—"}
                </td>
                <td className="py-2.5 px-2 font-mono text-xs">
                  {s.suggested_position_size ?? "—"} shares
                </td>
                <td className="py-2.5 px-2">
                  {s.risk_level && (
                    <span
                      className={clsx(
                        "badge",
                        s.risk_level === "LOW"
                          ? "badge-high"
                          : s.risk_level === "MEDIUM"
                          ? "badge-medium"
                          : "badge-bear"
                      )}
                    >
                      {s.risk_level}
                    </span>
                  )}
                </td>
                <td className="py-2.5 px-2">
                  {s.regime && (
                    <span
                      className={clsx(
                        "badge",
                        s.regime === "BULL"
                          ? "badge-bull"
                          : s.regime === "BEAR"
                          ? "badge-bear"
                          : "badge-sideways"
                      )}
                    >
                      {s.regime}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {signals.length === 0 && (
        <p className="text-slate-400 text-center py-8">
          No signals generated. Run model training first or adjust threshold.
        </p>
      )}
    </div>
  );
}

function probColor(p: number): string {
  if (p >= 0.75) return "bg-profit";
  if (p >= 0.65) return "bg-yellow-500";
  if (p >= 0.55) return "bg-blue-500";
  return "bg-slate-500";
}

function probTextColor(p: number): string {
  if (p >= 0.75) return "text-profit";
  if (p >= 0.65) return "text-yellow-400";
  return "text-slate-400";
}

function badgeClass(level: string): string {
  switch (level) {
    case "HIGH":
      return "badge-high";
    case "MEDIUM":
      return "badge-medium";
    default:
      return "badge-low";
  }
}
