"use client";

interface ModelData {
  xgboost: {
    accuracy?: number | null;
    precision?: number | null;
    recall?: number | null;
    auc_roc?: number | null;
    sample_count?: number;
    status?: string;
  };
  lstm: {
    accuracy?: number | null;
    auc_roc?: number | null;
    val_loss?: number | null;
    sample_count?: number;
    status?: string;
  };
  xgb_last_trained: string | null;
  lstm_last_trained: string | null;
  training_state?: {
    status: string;
    last_trained: string | null;
    last_error: string | null;
    next_scheduled: string | null;
  };
}

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface Props {
  modelData: ModelData | null;
  features: FeatureImportance[];
}

function TrainingStatusBadge({ state }: { state?: ModelData["training_state"] }) {
  if (!state) return null;
  const colors: Record<string, string> = {
    idle: "bg-slate-600 text-slate-300",
    training: "bg-yellow-600 text-yellow-100 animate-pulse",
    completed: "bg-green-600 text-green-100",
    failed: "bg-red-600 text-red-100",
  };
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <span className={`text-xs px-2 py-0.5 rounded-full ${colors[state.status] || colors.idle}`}>
        {state.status === "training" ? "⏳ Training…" : state.status.toUpperCase()}
      </span>
      {state.last_trained && (
        <span className="text-xs text-slate-400">
          Last trained: {formatDate(state.last_trained)}
        </span>
      )}
      {state.next_scheduled && (
        <span className="text-xs text-slate-400">
          Next: {formatDate(state.next_scheduled)}
        </span>
      )}
      {state.last_error && (
        <span className="text-xs text-red-400 truncate max-w-xs" title={state.last_error}>
          Error: {state.last_error.slice(0, 80)}
        </span>
      )}
    </div>
  );
}

export default function ModelPanel({ modelData, features }: Props) {
  const notTrained =
    modelData?.xgboost?.status === "not_trained" &&
    modelData?.lstm?.status === "not_trained";

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between gap-4">
        <h3 className="text-lg font-semibold">Model Performance</h3>
        <TrainingStatusBadge state={modelData?.training_state} />
      </div>

      {!modelData && (
        <p className="text-slate-400 text-sm">
          No model metrics available. Train models first.
        </p>
      )}

      {notTrained && (
        <div className="bg-yellow-900/30 border border-yellow-700/40 rounded-lg p-4">
          <p className="text-yellow-300 text-sm font-medium">Models not yet trained</p>
          <p className="text-yellow-400/70 text-xs mt-1">
            Click &quot;Train Models&quot; to start. Training fetches 2 years of Nifty 50
            data and builds XGBoost + LSTM models. It usually takes 3–8 minutes.
            Auto-training is scheduled daily after market close (4:00 PM IST).
          </p>
        </div>
      )}

      {modelData && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* XGBoost */}
          <div className="bg-slate-800/50 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-blue-400 mb-3">
              XGBoost (Primary)
            </h4>
            <div className="space-y-2 text-sm">
              <Row label="Accuracy" value={pct(modelData.xgboost.accuracy)} />
              <Row label="Precision" value={pct(modelData.xgboost.precision)} />
              <Row label="Recall" value={pct(modelData.xgboost.recall)} />
              <Row label="AUC-ROC" value={modelData.xgboost.auc_roc?.toFixed(3)} />
              <Row label="Samples" value={modelData.xgboost.sample_count?.toLocaleString()} />
              <Row label="Last Trained" value={formatDate(modelData.xgb_last_trained)} />
            </div>
          </div>

          {/* LSTM */}
          <div className="bg-slate-800/50 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-purple-400 mb-3">
              LSTM (Secondary)
            </h4>
            <div className="space-y-2 text-sm">
              <Row label="Accuracy" value={pct(modelData.lstm.accuracy)} />
              <Row label="AUC-ROC" value={modelData.lstm.auc_roc?.toFixed(3)} />
              <Row label="Val Loss" value={modelData.lstm.val_loss?.toFixed(4)} />
              <Row label="Samples" value={modelData.lstm.sample_count?.toLocaleString()} />
              <Row label="Last Trained" value={formatDate(modelData.lstm_last_trained)} />
            </div>
          </div>
        </div>
      )}

      {/* Feature importance */}
      {features.length > 0 && (
        <div>
          <h4 className="text-sm text-slate-400 mb-2">
            Top Feature Importance (XGBoost)
          </h4>
          <div className="space-y-1.5">
            {features.slice(0, 15).map((f) => (
              <div key={f.feature} className="flex items-center gap-2">
                <span className="text-xs text-slate-400 w-36 truncate">
                  {f.feature}
                </span>
                <div className="flex-1 bg-slate-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{
                      width: `${(f.importance / (features[0]?.importance || 1)) * 100}%`,
                    }}
                  />
                </div>
                <span className="text-xs font-mono text-slate-400 w-12 text-right">
                  {(f.importance * 100).toFixed(1)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value?: string | number | null }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-400">{label}</span>
      <span className="font-mono text-slate-200">{value ?? "—"}</span>
    </div>
  );
}

function pct(v?: number | null): string {
  return v != null ? `${(v * 100).toFixed(1)}%` : "—";
}

function formatDate(iso: string | null): string {
  if (!iso) return "Never";
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
