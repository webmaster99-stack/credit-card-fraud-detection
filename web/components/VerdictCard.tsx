import type { PredictResult } from "@/lib/types";
import { ReasonList } from "./ReasonList";

function percent(p: number): string {
  return `${(p * 100).toFixed(p < 0.1 ? 2 : 1)}%`;
}

export function VerdictCard({ result }: { result: PredictResult }) {
  return (
    <div
      className={`rounded-lg border p-4 ${
        result.flagged
          ? "border-orange-400 bg-orange-50 dark:bg-orange-950/30"
          : "border-slate-300 bg-slate-50 dark:bg-slate-800/50"
      }`}
    >
      <h3 className="text-lg font-semibold">
        {result.flagged ? "Flagged as likely fraud" : "Not flagged"}
      </h3>
      <p className="mt-1 text-sm">
        Fraud probability <strong>{percent(result.fraud_probability)}</strong> (flag threshold{" "}
        {percent(result.threshold)})
      </p>
      <p className="mt-1 text-xs text-slate-500">
        Scored by {result.model_name} v{result.model_version}, pipeline {result.pipeline_version}.
        Request {result.request_id}
      </p>
      <div className="mt-4">
        <h4 className="mb-2 text-sm font-medium">Why this score (largest effect first)</h4>
        <ReasonList reasons={result.reasons} />
      </div>
    </div>
  );
}
