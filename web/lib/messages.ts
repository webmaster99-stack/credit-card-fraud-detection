/** Shown when the backend does not answer in time or a gateway gives up on it. Free-tier Render
 * services sleep when idle and take about a minute to wake (docs/infra.md); the wake-up keeps
 * going after a request gives up, so a retry shortly after usually succeeds. */
export const WAKING_MESSAGE =
  "The model service is still waking up (the free tier sleeps when idle and takes about a minute). " +
  "Wait a moment and try again.";

/** How long the browser waits before the submit pages explain why a request is slow. */
export const SLOW_HINT_MS = 4000;

/** Turn a failed `/api/*` response into the list of problems the pages display. The body may not
 * be JSON at all (a gateway timeout returns an HTML or plain-text error page). */
export function problemsFrom(body: unknown, status: number): string[] {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (Array.isArray(detail)) return detail.map(String);
  if (detail) return [String(detail)];
  if (status === 502 || status === 503 || status === 504) return [WAKING_MESSAGE];
  return [`The service returned HTTP ${status}.`];
}
