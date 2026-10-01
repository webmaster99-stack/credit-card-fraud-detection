import "server-only";

import type {
  BatchResult,
  HealthInfo,
  ModelInfo,
  MonitoringReport,
  PredictResult,
  Transaction,
} from "./types";
import { WAKING_MESSAGE } from "./messages";

const BASE_URL = process.env.FRAUD_API_URL;
const API_KEY = process.env.FRAUD_API_KEY;

/** A non-2xx response from the FastAPI backend, with its `detail` normalized to a string list. */
export class FraudApiError extends Error {
  constructor(
    public status: number,
    public problems: string[],
  ) {
    super(problems.join(" "));
    this.name = "FraudApiError";
  }
}

function settings(): { baseUrl: string; apiKey: string } {
  if (!BASE_URL || !API_KEY) {
    throw new Error("FRAUD_API_URL and FRAUD_API_KEY must be set (see web/.env.example).");
  }
  return { baseUrl: BASE_URL, apiKey: API_KEY };
}

/** Longer than the ~53 s a sleeping Render free-tier service took to wake when measured, and
 * shorter than the routes' `maxDuration` (60 s), so a wake-up that is not finished yet becomes a
 * clear "try again" error here instead of the platform killing the function. */
const API_TIMEOUT_MS = 55_000;

async function call(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = API_TIMEOUT_MS,
): Promise<Response> {
  const { baseUrl, apiKey } = settings();
  const headers = new Headers(init.headers);
  headers.set("X-API-Key", apiKey);
  try {
    return await fetch(`${baseUrl}${path}`, {
      ...init,
      headers,
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    if (err instanceof Error && (err.name === "TimeoutError" || err.name === "AbortError")) {
      throw new FraudApiError(504, [WAKING_MESSAGE]);
    }
    throw err;
  }
}

async function callJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  const res = await call(path, { ...init, headers });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body?.detail;
    const problems = Array.isArray(detail) ? detail : [detail ?? `HTTP ${res.status}`];
    throw new FraudApiError(res.status, problems.map(String));
  }
  return body as T;
}

export function apiPredict(txn: Transaction): Promise<PredictResult> {
  return callJson<PredictResult>("/v1/predict", {
    method: "POST",
    body: JSON.stringify(txn),
  });
}

export function apiPredictBatchJson(rows: Transaction[]): Promise<BatchResult> {
  return callJson<BatchResult>("/v1/predict/batch", {
    method: "POST",
    body: JSON.stringify(rows),
  });
}

/** Forwards an uploaded CSV to the backend's multipart batch path unparsed - the backend already
 * owns CSV validation (row cap, column checks); duplicating it here would just be another place
 * for the two to drift apart. */
export async function apiPredictBatchCsv(file: File): Promise<BatchResult> {
  const form = new FormData();
  form.append("file", file, file.name);
  const res = await call("/v1/predict/batch", { method: "POST", body: form });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body?.detail;
    const problems = Array.isArray(detail) ? detail : [detail ?? `HTTP ${res.status}`];
    throw new FraudApiError(res.status, problems.map(String));
  }
  return body as BatchResult;
}

export function apiModelInfo(): Promise<ModelInfo> {
  return callJson<ModelInfo>("/v1/model");
}

/** The newest stored monitoring summary of a kind, or null if that job has not run yet. */
export async function apiMonitoring<T>(
  kind: "nightly" | "replay",
): Promise<MonitoringReport<T> | null> {
  const res = await call(`/v1/monitoring/${kind}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new FraudApiError(res.status, [`HTTP ${res.status}`]);
  return (await res.json()) as MonitoringReport<T>;
}

/** The stored Evidently HTML report, or null when none exists. */
export async function apiMonitoringReportHtml(): Promise<string | null> {
  const res = await call("/v1/monitoring/nightly/report");
  return res.ok ? await res.text() : null;
}

/** Never throws: a health check that itself fails just means "not ok" (e.g. the Render free-tier
 * service is asleep and hasn't answered yet), which is exactly what the caller wants to show. The
 * short timeout matters: without it a sleeping service made this call hang until it woke, so the
 * "waking up" banner only appeared once there was nothing left to wait for. */
const HEALTH_TIMEOUT_MS = 3_000;

export async function apiHealth(): Promise<HealthInfo> {
  try {
    const res = await call("/health", {}, HEALTH_TIMEOUT_MS);
    const body = await res.json();
    return body as HealthInfo;
  } catch {
    return { status: "unreachable", model_loaded: false, database_ok: false };
  }
}
