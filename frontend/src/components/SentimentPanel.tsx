"use client";

import { useEffect, useRef } from "react";
import type { StockSentiment, NewsItem } from "@/lib/api";
import clsx from "clsx";

interface Props {
  sentiment: StockSentiment | null;
  loading: boolean;
  onRefresh?: () => void;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function sentimentColor(label: string) {
  if (label === "BULLISH") return "text-emerald-400";
  if (label === "BEARISH") return "text-red-400";
  return "text-yellow-400";
}

function sentimentBg(label: string) {
  if (label === "BULLISH") return "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
  if (label === "BEARISH") return "bg-red-500/20 text-red-300 border border-red-500/40";
  return "bg-yellow-500/20 text-yellow-300 border border-yellow-500/40";
}

function sentimentIcon(label: string) {
  if (label === "BULLISH") return "▲";
  if (label === "BEARISH") return "▼";
  return "◆";
}

function formatAge(hours: number): string {
  if (hours < 1) return `${Math.round(hours * 60)}m ago`;
  if (hours < 24) return `${Math.round(hours)}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function ScoreBar({ score }: { score: number }) {
  // score is -1 to +1; map to 0-100 for width, center at 50
  const pct = Math.round((score + 1) / 2 * 100);
  const barColor = score >= 0.05 ? "bg-emerald-500" : score <= -0.05 ? "bg-red-500" : "bg-yellow-500";
  return (
    <div className="relative w-full h-2 bg-slate-700 rounded-full overflow-hidden">
      {/* Center marker */}
      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-500" />
      {/* Fill from center */}
      {score >= 0 ? (
        <div
          className={clsx("absolute h-full rounded-full", barColor)}
          style={{ left: "50%", width: `${(pct - 50)}%` }}
        />
      ) : (
        <div
          className={clsx("absolute h-full rounded-full", barColor)}
          style={{ left: `${pct}%`, width: `${50 - pct}%` }}
        />
      )}
    </div>
  );
}

function BreakdownBar({ bullish, bearish, neutral, total }: {
  bullish: number; bearish: number; neutral: number; total: number;
}) {
  if (total === 0) return null;
  const bPct = (bullish / total) * 100;
  const rPct = (bearish / total) * 100;
  const nPct = (neutral / total) * 100;
  return (
    <div className="flex rounded-full overflow-hidden h-2">
      {bPct > 0 && <div className="bg-emerald-500" style={{ width: `${bPct}%` }} />}
      {nPct > 0 && <div className="bg-yellow-500" style={{ width: `${nPct}%` }} />}
      {rPct > 0 && <div className="bg-red-500" style={{ width: `${rPct}%` }} />}
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function SentimentPanel({ sentiment, loading, onRefresh }: Props) {
  if (loading) {
    return (
      <div className="card animate-pulse space-y-3">
        <div className="h-4 bg-slate-700 rounded w-40" />
        <div className="h-3 bg-slate-700 rounded" />
        {[1, 2, 3].map(i => (
          <div key={i} className="h-12 bg-slate-700 rounded" />
        ))}
      </div>
    );
  }

  if (!sentiment) return null;

  const { articles, sentiment_score, sentiment_label,
          bullish_count, bearish_count, neutral_count, article_count } = sentiment;

  return (
    <div className="card space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
          <span>📰</span> News Sentiment
        </h4>
        <div className="flex items-center gap-2">
          <span className={clsx("text-xs font-mono", sentimentColor(sentiment_label))}>
            {score2str(sentiment_score)}
          </span>
          <span className={clsx("text-xs px-2 py-0.5 rounded-full font-semibold", sentimentBg(sentiment_label))}>
            {sentimentIcon(sentiment_label)} {sentiment_label}
          </span>
          {onRefresh && (
            <button onClick={onRefresh} title="Refresh news" className="text-slate-500 hover:text-slate-300 transition">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Score bar */}
      <div className="space-y-1">
        <ScoreBar score={sentiment_score} />
        <div className="flex justify-between text-[10px] text-slate-500">
          <span>BEARISH</span>
          <span>NEUTRAL</span>
          <span>BULLISH</span>
        </div>
      </div>

      {/* Breakdown */}
      {article_count > 0 && (
        <div className="space-y-1">
          <BreakdownBar bullish={bullish_count} bearish={bearish_count} neutral={neutral_count} total={article_count} />
          <div className="flex gap-3 text-[10px] text-slate-400">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
              {bullish_count} bullish
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-yellow-500 inline-block" />
              {neutral_count} neutral
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
              {bearish_count} bearish
            </span>
            <span className="ml-auto">{article_count} articles</span>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <p className="text-[10px] text-slate-500 italic border-l-2 border-slate-600 pl-2">
        Sentiment is for informational purposes only. Not financial advice. Always verify with
        technical analysis before making trading decisions.
      </p>

      {/* News list */}
      {articles.length === 0 ? (
        <p className="text-xs text-slate-500 text-center py-2">No recent news found</p>
      ) : (
        <div className="space-y-2 max-h-64 overflow-y-auto pr-1 scrollbar-thin">
          {articles.map((article, idx) => (
            <NewsCard key={idx} article={article} />
          ))}
        </div>
      )}
    </div>
  );
}

function score2str(score: number): string {
  return (score >= 0 ? "+" : "") + score.toFixed(3);
}

function NewsCard({ article }: { article: NewsItem }) {
  const { title, publisher, age_hours, url, sentiment_label, sentiment_score } = article;

  return (
    <a
      href={url || "#"}
      target={url ? "_blank" : undefined}
      rel="noopener noreferrer"
      className="block group rounded-lg bg-slate-800/60 border border-slate-700/50 p-2.5 hover:border-slate-500/60 transition-colors"
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs text-slate-200 group-hover:text-white transition leading-snug line-clamp-2 flex-1">
          {title}
        </p>
        <span className={clsx("shrink-0 text-[10px] px-1.5 py-0.5 rounded font-semibold", sentimentBg(sentiment_label))}>
          {sentimentIcon(sentiment_label)}
        </span>
      </div>
      <div className="flex items-center gap-2 mt-1.5">
        <span className="text-[10px] text-slate-500 truncate max-w-[120px]">{publisher}</span>
        <span className="text-[10px] text-slate-600">•</span>
        <span className="text-[10px] text-slate-500">{formatAge(age_hours)}</span>
        <span className="ml-auto text-[10px] font-mono text-slate-500">{score2str(sentiment_score)}</span>
      </div>
    </a>
  );
}
