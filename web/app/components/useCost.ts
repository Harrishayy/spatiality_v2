// Polls /api/trace/:scene_id/cost for the live aggregated dollar / token /
// call total. Cheaper than `useTrace` (which pulls the entire span tree) so
// the always-visible header CostBadge can refresh without thrashing.

"use client";

import { useEffect, useRef, useState } from "react";

import { fetchCost } from "@/lib/api";
import type { CostAggregate, ManifestStatus } from "@/lib/types";

interface State {
  data: CostAggregate | null;
  loading: boolean;
}

const POLL_PROCESSING_MS = 5_000;
const POLL_TERMINAL_MS = 60_000;

export function useCost(sceneId: string, manifestStatus: ManifestStatus | null) {
  const [state, setState] = useState<State>({ data: null, loading: true });
  const versionRef = useRef(0);

  useEffect(() => {
    versionRef.current += 1;
    const myVersion = versionRef.current;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
      try {
        const data = await fetchCost(sceneId);
        if (cancelled || versionRef.current !== myVersion) return;
        setState({ data, loading: false });
      } catch {
        // Soft fail — the badge just stays at its last known value rather
        // than rendering an error in the header chrome.
        if (cancelled || versionRef.current !== myVersion) return;
        setState((s) => ({ ...s, loading: false }));
      }
      if (cancelled || versionRef.current !== myVersion) return;
      const interval =
        manifestStatus === "processing" || manifestStatus === "queued"
          ? POLL_PROCESSING_MS
          : POLL_TERMINAL_MS;
      timer = setTimeout(tick, interval);
    };

    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [sceneId, manifestStatus]);

  return state;
}
