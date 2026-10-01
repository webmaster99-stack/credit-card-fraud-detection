"use client";

import { FormEvent, useState } from "react";

import { problemsFrom } from "@/lib/messages";
import type { BatchResult } from "@/lib/types";
import { useSlowLoading } from "@/lib/useSlowLoading";

function percent(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

function resultsToCsv(result: BatchResult): string {
  const header = "row,request_id,fraud_probability,flagged,top_reason";
  const lines = result.results.map((r) =>
    [r.row, r.request_id, r.fraud_probability, r.flagged, r.top_reason ?? ""]
      .map((v) => `"${String(v).replace(/"/g, '""')}"`)
      .join(","),
  );
  return [header, ...lines].join("\n");
}

function downloadCsv(result: BatchResult) {
  const blob = new Blob([resultsToCsv(result)], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "fraud_scores.csv";
  a.click();
  URL.revokeObjectURL(url);
}

export default function BatchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<BatchResult | null>(null);
  const [problems, setProblems] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);
  const slow = useSlowLoading(loading);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setResult(null);
    setProblems(null);
    const form = new FormData();
    form.append("file", file, file.name);
    try {
      const res = await fetch("/api/predict/batch", { method: "POST", body: form });
      const body = await res.json().catch(() => null);
      if (!res.ok || body === null) {
        setProblems(problemsFrom(body, res.status));
      } else {
        setResult(body as BatchResult);
      }
    } catch {
      setProblems(["Could not reach the API. Is it running and is FRAUD_API_URL set?"]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Batch CSV</h1>
        <p className="mt-1 text-sm text-slate-500">
          Upload a CSV with columns <code>trans_ts, amt, category, gender, state, city_pop, dob,
          lat, long, merch_lat, merch_long, card_id</code>. History for a batch comes only from
          other rows in the same file, sorted by time - not from any card&apos;s stored history.
        </p>
      </div>

      <form onSubmit={onSubmit} className="flex items-center gap-3">
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="text-sm"
        />
        <button
          type="submit"
          disabled={!file || loading}
          className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
        >
          {loading ? "Scoring..." : "Score file"}
        </button>
      </form>

      {slow && (
        <p className="text-sm text-amber-700 dark:text-amber-300">
          This is taking a while - the free-tier model service sleeps when idle and can take
          about a minute to wake. Your request is still running.
        </p>
      )}

      {problems && (
        <div className="rounded border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200">
          <p className="font-medium">Could not score this file</p>
          <ul className="mt-1 list-inside list-disc">
            {problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      )}

      {result && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm">
              Scored <strong>{result.n_rows.toLocaleString()}</strong> transactions;{" "}
              <strong>{result.n_flagged.toLocaleString()}</strong> flagged at threshold{" "}
              {percent(result.threshold)}. Model {result.model_name} v{result.model_version}.
            </p>
            <button
              onClick={() => downloadCsv(result)}
              className="rounded border border-slate-300 px-3 py-1 text-sm dark:border-slate-700"
            >
              Download CSV
            </button>
          </div>
          <div className="overflow-x-auto rounded border border-slate-200 dark:border-slate-800">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-100 dark:bg-slate-800">
                <tr>
                  <th className="px-3 py-2">Row</th>
                  <th className="px-3 py-2">Fraud probability</th>
                  <th className="px-3 py-2">Flagged</th>
                  <th className="px-3 py-2">Top reason</th>
                </tr>
              </thead>
              <tbody>
                {result.results
                  .slice()
                  .sort((a, b) => b.fraud_probability - a.fraud_probability)
                  .slice(0, 500)
                  .map((r) => (
                    <tr key={r.request_id} className="border-t border-slate-200 dark:border-slate-800">
                      <td className="px-3 py-1">{r.row}</td>
                      <td className="px-3 py-1">{percent(r.fraud_probability)}</td>
                      <td className="px-3 py-1">{r.flagged ? "yes" : "no"}</td>
                      <td className="px-3 py-1">{r.top_reason ?? ""}</td>
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
