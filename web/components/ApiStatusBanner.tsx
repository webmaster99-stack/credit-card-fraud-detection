"use client";

import { useEffect, useState } from "react";

import type { HealthInfo } from "@/lib/types";

// Render's free tier sleeps after 15 min idle and takes about a minute to wake (docs/infra.md).
// Poll every 5s while degraded so the banner clears itself once the service answers.
const POLL_MS = 5000;

export function ApiStatusBanner() {
  const [health, setHealth] = useState<HealthInfo | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        const body = (await res.json()) as HealthInfo;
        if (!cancelled) setHealth(body);
      } catch {
        if (!cancelled) setHealth({ status: "unreachable", model_loaded: false, database_ok: false });
      }
      if (!cancelled) timer = setTimeout(poll, POLL_MS);
    }
    poll();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  if (!health || health.status === "ok") return null;

  return (
    <div className="border-b border-amber-300 bg-amber-50 px-4 py-2 text-center text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-200">
      Waking up the model service - the free-tier API sleeps when idle and can take about a minute
      to respond to its first request.
    </div>
  );
}
