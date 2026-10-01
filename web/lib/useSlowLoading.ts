"use client";

import { useEffect, useState } from "react";

import { SLOW_HINT_MS } from "./messages";

/** True once `loading` has stayed true for `SLOW_HINT_MS`, so a page can explain a slow request
 * (typically the free-tier API waking up) without flashing the hint on every fast one. */
export function useSlowLoading(loading: boolean): boolean {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    if (!loading) {
      setSlow(false);
      return;
    }
    const timer = setTimeout(() => setSlow(true), SLOW_HINT_MS);
    return () => clearTimeout(timer);
  }, [loading]);
  return slow;
}
