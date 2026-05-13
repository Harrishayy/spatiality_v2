"use client";

/**
 * Top-down capture-map preview card (Stage 4).
 *
 * Surfaces the 2D footprint of what was captured (above-floor density
 * heatmap). Answers "what's in the room and how much of it did we
 * cover?" rather than "where could a humanoid stand?", which the data
 * rarely supports for handheld captures.
 *
 * Pinned to the top-left of the viewer so it doesn't fight with the
 * right-anchored side panel or the bottom controls cluster. Toggled by
 * the "Capture map" button in the viewer toolbar.
 */

import { useEffect, useState } from "react";
import type { Manifest } from "@/lib/types";
import { getArtifactUrl } from "@/lib/api";

/** Metadata JSON the backend writes alongside capture_map.png. We read
 *  the summary stats; the full encoded `density_b64` grid is reserved
 *  for a future in-3D overlay path. */
interface CaptureMapMeta {
  cell_size_m: number;
  grid_shape: [number, number];
  /** Dimensions (width_m, height_m) of the captured surfaces *without* the
   *  grid's breathing-room margin. Use this for the displayed "extent"
   *  rather than `grid_shape × cell_size_m`. */
  tight_extent_m?: [number, number];
  floor_height_world: number;
  stats: {
    coverage_cells: number;
    coverage_m2: number;
    n_frames: number;
  };
}

export function CaptureMapCard({
  sceneId,
  manifest,
}: {
  sceneId: string;
  manifest: Manifest;
}) {
  const pngArtifact = manifest.artifacts?.capture_map_png;
  const jsonArtifact = manifest.artifacts?.capture_map_json;
  const pngUrl = pngArtifact ? getArtifactUrl(sceneId, pngArtifact) : null;

  const [meta, setMeta] = useState<CaptureMapMeta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);

  useEffect(() => {
    if (!jsonArtifact) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(getArtifactUrl(sceneId, jsonArtifact));
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const body = (await res.json()) as CaptureMapMeta;
        if (!cancelled) setMeta(body);
      } catch (err) {
        if (!cancelled) {
          setMetaError(err instanceof Error ? err.message : String(err));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sceneId, jsonArtifact]);

  if (!pngUrl) {
    return (
      <div className="pointer-events-auto absolute left-3 top-3 max-w-[280px] rounded-xl border border-ink-700/70 bg-ink-900/85 p-3 font-mono text-[11px] text-ink-200 backdrop-blur">
        <div className="text-accent-300">Estimated capture map</div>
        <div className="mt-1 text-ink-400">
          Stage 4 hasn&rsquo;t run for this scene. Re-run the pipeline to
          generate the capture map.
        </div>
      </div>
    );
  }

  const grid = meta?.grid_shape;
  // Prefer the tight (no-margin) extent for the displayed "extent" — it's
  // the actual captured footprint, not the grid's render-padded size.
  const widthM = meta?.tight_extent_m?.[0]
    ?? (grid && meta ? grid[1] * meta.cell_size_m : null);
  const depthM = meta?.tight_extent_m?.[1]
    ?? (grid && meta ? grid[0] * meta.cell_size_m : null);

  return (
    <div className="pointer-events-auto absolute left-3 top-3 w-[280px] rounded-xl border border-ink-700/70 bg-ink-900/85 p-3 font-mono text-[11px] text-ink-200 backdrop-blur">
      <div className="flex items-baseline justify-between gap-2 whitespace-nowrap">
        <div className="text-accent-300">Estimated capture map</div>
        <div className="text-[10px] uppercase tracking-wider text-ink-400">
          stage 4
        </div>
      </div>

      <div className="mt-2 overflow-hidden rounded-md border border-ink-700/60 bg-ink-950">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={pngUrl}
          alt="Top-down capture map: density of captured surfaces"
          className="block w-full"
          // Hint browser at expected aspect; backend upscales each cell 6×.
          width={grid ? grid[1] * 6 : undefined}
          height={grid ? grid[0] * 6 : undefined}
        />
      </div>

      <ul className="mt-2 space-y-0.5 text-[10px] leading-snug text-ink-300">
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-sm bg-[#ffc484]" />
          <span className="text-ink-200">Observed surfaces</span>
          {meta && (
            <span className="ml-auto text-ink-400">
              ~{meta.stats.coverage_m2.toFixed(2)} m²
            </span>
          )}
        </li>
      </ul>

      {meta && widthM && depthM && (
        <div className="mt-2 grid grid-cols-2 gap-1 border-t border-ink-700/60 pt-2 text-[10px] text-ink-400">
          <div>
            <div className="text-ink-500">extent</div>
            <div className="text-ink-200">
              ~{widthM.toFixed(1)} × {depthM.toFixed(1)} m
            </div>
          </div>
          <div>
            <div className="text-ink-500">frames</div>
            <div className="text-ink-200">{meta.stats.n_frames}</div>
          </div>
        </div>
      )}

      {metaError && (
        <div className="mt-2 text-[10px] text-accent-400">
          stats unavailable: {metaError}
        </div>
      )}
    </div>
  );
}
