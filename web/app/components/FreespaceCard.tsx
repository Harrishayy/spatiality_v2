"use client";

/**
 * Top-down free-space / traversability preview card.
 *
 * Stage 5 of the pipeline produces a 2D occupancy grid telling a humanoid
 * which floor cells it can stand on without colliding with anything between
 * ankle and head height. The backend renders that grid into a polished PNG
 * with the camera track overlaid; this card surfaces it next to the 3D
 * viewer so a reviewer can see geometry (point cloud) and locomotion
 * planning data (traversability) side-by-side.
 *
 * Toggled by the "Free space" button in the viewer toolbar. Hidden by
 * default — reviewers opt in once they've explored the cloud, so the first
 * impression of the scene is uncluttered.
 */

import { useEffect, useState } from "react";
import type { Manifest } from "@/lib/types";
import { getArtifactUrl } from "@/lib/api";

/** Metadata JSON the backend writes alongside traversability.png. We only
 *  read the summary stats; the full encoded `cells_b64` grid is reserved
 *  for the future 3D-overlay path. */
interface FreespaceMeta {
  cell_size_m: number;
  grid_shape: [number, number];
  floor_height_world: number;
  robot_radius_m: number;
  stats: {
    traversable_cells: number;
    obstacle_cells: number;
    unknown_cells: number;
    traversable_m2: number;
    obstacle_m2: number;
  };
}

export function FreespaceCard({
  sceneId,
  manifest,
}: {
  sceneId: string;
  manifest: Manifest;
}) {
  const pngArtifact = manifest.artifacts?.traversability_png;
  const jsonArtifact = manifest.artifacts?.traversability_json;
  const pngUrl = pngArtifact ? getArtifactUrl(sceneId, pngArtifact) : null;

  const [meta, setMeta] = useState<FreespaceMeta | null>(null);
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
        const body = (await res.json()) as FreespaceMeta;
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
      <div className="pointer-events-auto absolute bottom-3 right-3 max-w-[280px] rounded-xl border border-ink-700/70 bg-ink-900/85 p-3 font-mono text-[11px] text-ink-200 backdrop-blur">
        <div className="text-accent-300">Free space</div>
        <div className="mt-1 text-ink-400">
          Stage 5 hasn&rsquo;t run for this scene. Re-run the pipeline to
          generate a traversability grid.
        </div>
      </div>
    );
  }

  const grid = meta?.grid_shape;
  const widthM = grid && meta ? grid[1] * meta.cell_size_m : null;
  const depthM = grid && meta ? grid[0] * meta.cell_size_m : null;

  return (
    <div className="pointer-events-auto absolute bottom-3 right-3 w-[280px] rounded-xl border border-ink-700/70 bg-ink-900/85 p-3 font-mono text-[11px] text-ink-200 backdrop-blur">
      <div className="flex items-baseline justify-between">
        <div className="text-accent-300">Free space</div>
        <div className="text-[10px] uppercase tracking-wider text-ink-400">
          stage 5 · top-down
        </div>
      </div>

      <div className="mt-2 overflow-hidden rounded-md border border-ink-700/60 bg-ink-950">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={pngUrl}
          alt="Top-down traversability grid for this scene"
          className="block w-full"
          // Hint browser at expected aspect; backend upscales each cell 6×.
          width={grid ? grid[1] * 6 : undefined}
          height={grid ? grid[0] * 6 : undefined}
        />
      </div>

      <ul className="mt-2 space-y-0.5 text-[10px] leading-snug text-ink-300">
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-sm bg-[#60c080]" />
          <span className="text-ink-200">Traversable</span>
          {meta && (
            <span className="ml-auto text-ink-400">
              {meta.stats.traversable_m2.toFixed(2)} m²
            </span>
          )}
        </li>
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-sm bg-[#ff6b4a]" />
          <span className="text-ink-200">Obstacle</span>
          {meta && (
            <span className="ml-auto text-ink-400">
              {meta.stats.obstacle_m2.toFixed(2)} m²
            </span>
          )}
        </li>
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-sm bg-[#332622]" />
          <span className="text-ink-200">Unknown</span>
        </li>
      </ul>

      {meta && widthM && depthM && (
        <div className="mt-2 grid grid-cols-2 gap-1 border-t border-ink-700/60 pt-2 text-[10px] text-ink-400">
          <div>
            <div className="text-ink-500">cell</div>
            <div className="text-ink-200">
              {(meta.cell_size_m * 100).toFixed(0)} cm
            </div>
          </div>
          <div>
            <div className="text-ink-500">extent</div>
            <div className="text-ink-200">
              {widthM.toFixed(1)} × {depthM.toFixed(1)} m
            </div>
          </div>
          <div>
            <div className="text-ink-500">robot r</div>
            <div className="text-ink-200">
              {(meta.robot_radius_m * 100).toFixed(0)} cm
            </div>
          </div>
          <div>
            <div className="text-ink-500">floor h</div>
            <div className="text-ink-200">
              {meta.floor_height_world.toFixed(2)}
            </div>
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
