import type { Reason } from "@/lib/types";

/** Reason bars, largest contribution first - a lighter-weight stand-in for the Gradio demo's SHAP
 * bar chart (no charting library, just relative-width bars). */
export function ReasonList({ reasons }: { reasons: Reason[] }) {
  if (reasons.length === 0) {
    return <p className="text-sm text-slate-500">No reasons returned.</p>;
  }
  const max = Math.max(...reasons.map((r) => Math.abs(r.contribution)));

  return (
    <ul className="space-y-2">
      {reasons.map((r) => {
        const width = max > 0 ? (Math.abs(r.contribution) / max) * 100 : 0;
        const positive = r.direction === "raises";
        return (
          <li key={r.feature} className="text-sm">
            <div className="flex justify-between gap-2">
              <span>{r.text}</span>
            </div>
            <div className="mt-1 h-2 w-full rounded bg-slate-200 dark:bg-slate-700">
              <div
                className={`h-2 rounded ${positive ? "bg-orange-500" : "bg-blue-500"}`}
                style={{ width: `${width}%` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
