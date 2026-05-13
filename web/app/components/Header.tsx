"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { Manifest } from "@/lib/types";
import { useCost } from "./useCost";

interface Props {
  manifest?: Manifest;
}

export function Header({ manifest }: Props) {
  const status = manifest?.status ?? "queued";
  const sceneId = manifest?.scene_id;
  return (
    <header className="lp-app-header">
      <Link href="/" className="lp-app-brand" aria-label="spatiality_v2 — home">
        <span className="lp-app-brand-mark" />
        <div className="lp-app-brand-meta">
          <span className="lp-app-brand-title">spatiality_v2</span>
          <span className="lp-app-brand-id">phone to spatial 3D</span>
        </div>
      </Link>
      <div className="lp-app-header-center">
        {sceneId && <SceneNameField sceneId={sceneId} />}
      </div>
      <div className="lp-app-header-meta">
        {sceneId && <CostBadge sceneId={sceneId} status={status} />}
        <StatusBadge status={status} />
      </div>
    </header>
  );
}

function CostBadge({
  sceneId,
  status,
}: {
  sceneId: string;
  status: Manifest["status"];
}) {
  const { data } = useCost(sceneId, status);
  // Don't render until we've heard back from the trace endpoint with at
  // least one model call. Avoids the "$0 · CALLS" empty-state flash that
  // prompted this whole rewrite.
  if (!data || data.call_count === 0) return null;
  const usd = data.total_usd;
  const pretty =
    usd >= 0.01 ? `$${usd.toFixed(3)}` : usd > 0 ? `$${usd.toFixed(5)}` : "$0";
  return (
    <span
      className="lp-status-pill"
      title="Total model spend on this scene — labeler plus chat."
    >
      {pretty} · {data.call_count} call{data.call_count === 1 ? "" : "s"}
    </span>
  );
}

// Pre-baked scenes (e.g. the hosted demo) get a humanised default name so
// the header doesn't read "Untitled scene" before the user has typed
// anything. Real upload-flow scenes still start blank.
const SCENE_DEFAULT_NAMES: Record<string, string> = {
  demo_piece: "Demo Piece",
};

function SceneNameField({ sceneId }: { sceneId: string }) {
  // TODO(swap): persist via /agent PATCH onto manifest.json instead of localStorage.
  const storageKey = `spatiality.sceneName.${sceneId}`;
  const defaultName = SCENE_DEFAULT_NAMES[sceneId] ?? "";
  const [name, setName] = useState("");
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const stored =
      typeof window !== "undefined"
        ? window.localStorage.getItem(storageKey)
        : null;
    setName(stored ?? defaultName);
    setHydrated(true);
  }, [storageKey, defaultName]);

  function commit(next: string) {
    const trimmed = next.trim().slice(0, 64);
    setName(trimmed);
    if (typeof window === "undefined") return;
    if (trimmed) window.localStorage.setItem(storageKey, trimmed);
    else window.localStorage.removeItem(storageKey);
  }

  if (!hydrated) return null;

  return (
    <input
      className="lp-app-scene-name"
      value={name}
      onChange={(e) => setName(e.target.value)}
      onBlur={(e) => commit(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === "Escape") {
          (e.currentTarget as HTMLInputElement).blur();
        }
      }}
      placeholder="Untitled scene"
      spellCheck={false}
      maxLength={64}
      aria-label="Scene name"
    />
  );
}

function StatusBadge({ status }: { status: Manifest["status"] }) {
  // Steady "ready" is the silent default — no pill in the corner once the
  // scene loads. Only surface the badge while something is in-flight or has
  // gone wrong, where the user actually needs the signal.
  if (status === "ready") return null;
  const { pillMod, dotMod, label } =
    status === "failed"
      ? { pillMod: "lp-status-pill--err", dotMod: "lp-status-dot--err", label: "Failed" }
      : { pillMod: "lp-status-pill--warn", dotMod: "lp-status-dot--warn", label: "Loading" };
  return (
    <span className={`lp-status-pill ${pillMod}`}>
      <span className={`lp-status-dot ${dotMod}`} />
      {label}
    </span>
  );
}
