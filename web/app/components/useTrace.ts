// Polls /api/trace/:scene_id for the live span tree. Polls every 10s while
// the manifest is processing (so the drawer streams new spans into view as
// the pipeline runs); falls back to a one-shot fetch + 30s manual refresh
// once the scene reaches a terminal state.

"use client";

import { useEffect, useRef, useState } from "react";

import { fetchTrace } from "@/lib/api";
import type { ManifestStatus, TraceResponse } from "@/lib/types";

interface State {
  data: TraceResponse | null;
  loading: boolean;
  error: string | null;
}

const POLL_PROCESSING_MS = 10_000;
const POLL_TERMINAL_MS = 30_000;

export function useTrace(sceneId: string, manifestStatus: ManifestStatus | null) {
  const [state, setState] = useState<State>({
    data: null,
    loading: true,
    error: null,
  });
  // Track the latest scene_id + status so a stale fetch doesn't overwrite a
  // newer one (race when the user navigates between scenes).
  const versionRef = useRef(0);

  useEffect(() => {
    versionRef.current += 1;
    const myVersion = versionRef.current;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
      try {
        const data = await fetchTrace(sceneId);
        if (cancelled || versionRef.current !== myVersion) return;
        setState({ data, loading: false, error: null });
      } catch (e) {
        if (cancelled || versionRef.current !== myVersion) return;
        setState((s) => ({
          ...s,
          loading: false,
          error: e instanceof Error ? e.message : String(e),
        }));
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
