import { NextRequest, NextResponse } from "next/server";

import { apiPredict, FraudApiError } from "@/lib/api";
import type { Transaction } from "@/lib/types";

// Room for a sleeping free-tier API to wake (see API_TIMEOUT_MS in lib/api.ts).
export const maxDuration = 60;

export async function POST(request: NextRequest) {
  const txn = (await request.json()) as Transaction;
  try {
    const result = await apiPredict(txn);
    return NextResponse.json(result);
  } catch (err) {
    if (err instanceof FraudApiError) {
      return NextResponse.json({ detail: err.problems }, { status: err.status });
    }
    throw err;
  }
}
