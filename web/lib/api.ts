import "server-only";

import type {
  BatchResult,
  HealthInfo,
  ModelInfo,
  PredictResult,
  Transaction,
} from "./types";

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

async function call(path: string, init: RequestInit = {}): Promise<Response> {
  const { baseUrl, apiKey } = settings();
  const headers = new Headers(init.headers);
  headers.set("X-API-Key", apiKey);
  return fetch(`${baseUrl}${path}`, { ...init, headers, cache: "no-store" });
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

/** Never throws: a health check that itself fails just means "not ok" (e.g. the Render free-tier
 * service is asleep and hasn't answered yet), which is exactly what the caller wants to show. */
export async function apiHealth(): Promise<HealthInfo> {
  try {
    const res = await call("/health");
    const body = await res.json();
    return body as HealthInfo;
  } catch {
    return { status: "unreachable", model_loaded: false, database_ok: false };
  }
}
