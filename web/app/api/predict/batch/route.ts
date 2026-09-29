import { NextRequest, NextResponse } from "next/server";

import { apiPredictBatchCsv, apiPredictBatchJson, FraudApiError } from "@/lib/api";
import type { Transaction } from "@/lib/types";

export async function POST(request: NextRequest) {
  const contentType = request.headers.get("content-type") ?? "";
  try {
    if (contentType.startsWith("multipart/form-data")) {
      const form = await request.formData();
      const file = form.get("file");
      if (!(file instanceof File)) {
        return NextResponse.json({ detail: ["No file uploaded under 'file'."] }, { status: 422 });
      }
      const result = await apiPredictBatchCsv(file);
      return NextResponse.json(result);
    }
    const rows = (await request.json()) as Transaction[];
    const result = await apiPredictBatchJson(rows);
    return NextResponse.json(result);
  } catch (err) {
    if (err instanceof FraudApiError) {
      return NextResponse.json({ detail: err.problems }, { status: err.status });
    }
    throw err;
  }
}
