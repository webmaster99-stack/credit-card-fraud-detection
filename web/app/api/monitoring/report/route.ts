import { gzipSync } from "node:zlib";

import { apiMonitoringReportHtml } from "@/lib/api";

export const dynamic = "force-dynamic";

/** Serves the stored Evidently report; the API key stays server-side like every other route.
 *
 * The report is ~4 MB of HTML (mostly Evidently's embedded JS bundle), near Vercel's 4.5 MB cap on
 * a function response, so it is sent gzipped (~1.2 MB) and the browser inflates it. */
export async function GET() {
  const html = await apiMonitoringReportHtml();
  if (html === null) {
    return new Response("No Evidently report has been generated yet.", { status: 404 });
  }
  return new Response(gzipSync(html), {
    headers: { "Content-Type": "text/html; charset=utf-8", "Content-Encoding": "gzip" },
  });
}
