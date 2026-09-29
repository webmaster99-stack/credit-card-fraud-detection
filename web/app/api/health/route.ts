import { NextResponse } from "next/server";

import { apiHealth } from "@/lib/api";

export async function GET() {
  return NextResponse.json(await apiHealth());
}
