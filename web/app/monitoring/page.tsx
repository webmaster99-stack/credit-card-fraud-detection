export default function MonitoringPage() {
  return (
    <div className="space-y-3">
      <h1 className="text-xl font-semibold">Monitoring</h1>
      <p className="text-sm text-slate-500">
        This page will embed the latest Evidently drift report and the drift-replay results once
        Phase 6 (Monitoring, see <code>docs/plan.md</code>) builds them. Nothing is wired up here
        yet - the nightly Evidently job, the report it publishes and the drift-replay script all
        come from that phase, not this one.
      </p>
    </div>
  );
}
