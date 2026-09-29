import { apiMonitoringReportHtml } from "@/lib/api";

export const dynamic = "force-dynamic";

/** Serves the stored Evidently report; the API key stays server-side like every other route. */
export async function GET() {
  const html = await apiMonitoringReportHtml();
  if (html === null) {
    return new Response("No Evidently report has been generated yet.", { status: 404 });
  }
  return new Response(html, { headers: { "Content-Type": "text/html; charset=utf-8" } });
}
