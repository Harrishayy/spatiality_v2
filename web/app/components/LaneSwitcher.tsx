"use client";

import { useUI } from "@/store/ui";
import type { Lane } from "@/lib/types";

const LANES: Array<{ id: Lane; label: string; tooltip: string }> = [
  {
    id: "b",
    label: "B · VLM-verified",
    tooltip:
      "Claude labels each tracked object via orbital novel-view renders, then SAM 3.1 grounds them back into the keyframe to verify.",
  },
  {
    id: "e",
    label: "E · Scene graph",
    tooltip:
      "ConceptGraphs-style spatial relationships between objects (on, under, next-to, …) layered over Lane B's labels.",
  },
  {
    id: "f",
    label: "F · SpatialLM",
    tooltip:
      "SpatialLM (NeurIPS '25) layout-only output: walls, doors, and windows of the room.",
  },
];

/** Compact 3-button switch for the active labeling lane. Sits in the
 *  scene viewer header so the user can A/B/F-compare the three pipelines
 *  on the same scene. */
export function LaneSwitcher() {
  const lane = useUI((s) => s.lane);
  const setLane = useUI((s) => s.setLane);

  return (
    <div
      role="tablist"
      aria-label="Labeling lane"
      className="flex items-center gap-1 rounded-lg border border-ink-700/70 bg-ink-900/70 p-0.5 backdrop-blur"
    >
      {LANES.map((l) => {
        const active = l.id === lane;
        return (
          <button
            key={l.id}
            type="button"
            role="tab"
            aria-selected={active}
            title={l.tooltip}
            onClick={() => setLane(l.id)}
            className={[
              "rounded-md px-3 py-1 text-[11px] font-medium tracking-wide transition-colors",
              active
                ? "bg-accent-500/20 text-accent-200 shadow-inner shadow-black/10"
                : "text-ink-400 hover:bg-ink-800/60 hover:text-ink-100",
            ].join(" ")}
          >
            {l.label}
          </button>
        );
      })}
    </div>
  );
}
