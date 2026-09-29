import { apiModelInfo } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function ModelInfoPage() {
  let info;
  let error: string | null = null;
  try {
    info = await apiModelInfo();
  } catch (err) {
    error = err instanceof Error ? err.message : "Could not reach the API.";
  }

  if (error || !info) {
    return (
      <div className="rounded border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200">
        Could not load model info: {error}
      </div>
    );
  }

  const rows: [string, string][] = [
    ["Registered model", `${info.model_name} v${info.model_version} (alias ${info.alias})`],
    ["Algorithm", `${info.step}, ${info.calibration}-calibrated`],
    ["Feature pipeline", `${info.pipeline_version} (${info.feature_set})`],
    ["Dataset", `${info.dataset_name} ${info.dataset_version}`],
    ["Split", info.split_spec],
    ["Decision threshold", info.threshold.toFixed(6)],
    ["Git commit", info.git_commit],
    ["Validation precision", info.validation.precision.toFixed(3)],
    ["Validation recall", info.validation.recall.toFixed(3)],
    ["Precision budget", `>= ${info.min_precision.toFixed(2)}`],
  ];

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Model info</h1>
      <table className="w-full text-left text-sm">
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label} className="border-t border-slate-200 dark:border-slate-800">
              <td className="py-2 pr-4 font-medium text-slate-500">{label}</td>
              <td className="py-2 font-mono">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-sm text-slate-500">
        This model uses a card&apos;s stored transaction history (velocity and behavioural
        features). See the model card on the{" "}
        <a
          href="https://huggingface.co/ilian-hadzhidimitrov/fraud-classifier/tree/champion"
          className="underline"
        >
          HF Hub
        </a>{" "}
        for the full write-up, including the test-set evaluation and its limitations.
      </p>
    </div>
  );
}
