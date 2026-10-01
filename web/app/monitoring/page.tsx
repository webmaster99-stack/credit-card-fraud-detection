import { apiMonitoring } from "@/lib/api";
import type {
  FeatureDrift,
  MonitoringReport,
  NightlySummary,
  ReplaySummary,
} from "@/lib/types";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

async function load<T>(kind: "nightly" | "replay"): Promise<MonitoringReport<T> | null | "error"> {
  try {
    return await apiMonitoring<T>(kind);
  } catch {
    return "error";
  }
}

function pct(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined ? "n/a" : `${(x * 100).toFixed(digits)}%`;
}

function DriftTable({ features }: { features: FeatureDrift[] }) {
  return (
    <table className="w-full text-left text-sm">
      <thead>
        <tr className="text-slate-500">
          <th className="py-1 pr-4 font-medium">Feature</th>
          <th className="py-1 pr-4 font-medium">PSI</th>
          <th className="py-1 font-medium">Status</th>
        </tr>
      </thead>
      <tbody>
        {features.map((f) => (
          <tr key={f.feature} className="border-t border-slate-200 dark:border-slate-800">
            <td className="py-1 pr-4 font-mono">{f.feature}</td>
            <td className="py-1 pr-4 font-mono">{f.psi.toFixed(3)}</td>
            <td className={f.drifted ? "py-1 font-medium text-red-600" : "py-1 text-slate-500"}>
              {f.drifted ? "drifted" : "stable"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Notice({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-slate-500">{children}</p>;
}

export default async function MonitoringPage() {
  const [nightly, replay] = await Promise.all([
    load<NightlySummary>("nightly"),
    load<ReplaySummary>("replay"),
  ]);

  return (
    <div className="space-y-8">
      <h1 className="text-xl font-semibold">Monitoring</h1>

      <section className="space-y-3">
        <h2 className="font-semibold">Nightly report</h2>
        {nightly === "error" && <Notice>Could not reach the API.</Notice>}
        {nightly === null && <Notice>The nightly job has not run yet.</Notice>}
        {nightly && nightly !== "error" && <Nightly report={nightly} />}
      </section>

      <section className="space-y-3">
        <h2 className="font-semibold">Drift replay</h2>
        <Notice>
          Held-out transactions replayed through the live API, once as-is and once with shifted
          amounts (<code>python -m monitoring.replay [--shift]</code>).
        </Notice>
        {replay === "error" && <Notice>Could not reach the API.</Notice>}
        {replay === null && <Notice>No replay has been run yet.</Notice>}
        {replay && replay !== "error" && <Replay report={replay} />}
      </section>
    </div>
  );
}

function Nightly({ report }: { report: MonitoringReport<NightlySummary> }) {
  const s = report.summary;
  const latest = s.daily.at(-1);
  return (
    <div className="space-y-4">
      <Notice>
        Generated {new Date(report.created_at).toUTCString()} from {s.n_predictions} logged
        predictions.
      </Notice>
      {s.alerts.length === 0 ? (
        <p className="rounded border border-green-300 bg-green-50 p-3 text-sm text-green-800 dark:border-green-800 dark:bg-green-950/40 dark:text-green-200">
          No alerts.
        </p>
      ) : (
        <ul className="space-y-1 rounded border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200">
          {s.alerts.map((a) => (
            <li key={a.layer + a.message}>
              <strong>{a.layer}</strong>: {a.message}
            </li>
          ))}
        </ul>
      )}
      <table className="w-full text-left text-sm">
        <tbody>
          {[
            ["Requests", String(s.service.requests)],
            ["Error rate (5xx)", pct(s.service.error_rate)],
            ["Invalid inputs (422)", pct(s.service.invalid_fraction, 2)],
            ["p95 latency", s.service.p95_latency_ms === null ? "n/a" : `${s.service.p95_latency_ms} ms`],
            [
              "Recall on labelled",
              s.performance.enough_labels
                ? `${pct(s.performance.recall)} (${s.performance.n_labelled} labels)`
                : `not enough labels yet (${s.performance.n_labelled})`,
            ],
          ].map(([label, value]) => (
            <tr key={label} className="border-t border-slate-200 dark:border-slate-800">
              <td className="py-2 pr-4 font-medium text-slate-500">{label}</td>
              <td className="py-2 font-mono">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {latest?.features && (
        <div className="space-y-1">
          <h3 className="text-sm font-medium">Input drift, {latest.day}</h3>
          <DriftTable features={latest.features} />
        </div>
      )}
      <a href="/api/monitoring/report" className="text-sm underline">
        Full Evidently report
      </a>
    </div>
  );
}

function Replay({ report }: { report: MonitoringReport<ReplaySummary> }) {
  const s = report.summary;
  return (
    <div className="space-y-3">
      <p className="text-sm">
        {new Date(report.created_at).toUTCString()}:{" "}
        <strong>
          {s.shift
            ? `shifted run (${s.shift.category ?? "all"} amounts x${s.shift.amount_factor})`
            : "clean run"}
        </strong>
        , {s.n_rows} rows from the {s.split} split. {s.n_drifted} feature(s) drifted; flag rate{" "}
        {s.flag.outside_band ? "outside" : "inside"} the expected band.
      </p>
      <DriftTable features={s.features} />
    </div>
  );
}
