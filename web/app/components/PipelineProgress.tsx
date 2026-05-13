"use client";

import type { Manifest, Stage, StageStatus } from "@/lib/types";
import { useUI } from "@/store/ui";

type VisibleStage = "capture" | "poses" | "segmentation";

// Visible stages in the panel. The internal "splat" manifest stage is a
// no-op placeholder (points.ply doubles as the splat source); we hide it
// from the UI. The pipeline manifest still tracks it for backwards-compat.
const STAGE_ORDER: VisibleStage[] = ["capture", "poses", "segmentation"];

const LABEL: Record<keyof Manifest["stages"], string> = {
  capture: "Capture",
  poses: "Reconstruction (VGGT)",
  splat: "Cloud",
  segmentation: "Segmentation",
};

function formatPointCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)} M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)} K`;
  return n.toLocaleString();
}

export function PipelineProgress({ manifest }: { manifest: Manifest }) {
  const cloudStats = useUI((s) => s.cloudStats);
  return (
    <div className="flex flex-col gap-3">
      <div className="lp-card">
        <div className="lp-card-head">
          <div className="lp-card-head-l">
            <h3 className="lp-card-title">Pipeline</h3>
            <span className="lp-side-section-accent">live</span>
          </div>
          <span className="lp-side-section-id">{manifest.scene_id}</span>
        </div>
        <ol className="lp-stage-list">
          {STAGE_ORDER.map((key) => {
            const stage = manifest.stages[key];
            return (
              <li key={key}>
                <div className="lp-stage-row">
                  <StageDot status={stage.status} />
                  <div className="lp-stage-meta">
                    <span className="lp-stage-label">{LABEL[key]}</span>
                  </div>
                  <DurationOrExtra stage={stage} />
                </div>
              </li>
            );
          })}
        </ol>
      </div>
      <Stats manifest={manifest} cloudStats={cloudStats} />
    </div>
  );
}

function StageDot({ status }: { status: StageStatus }) {
  const mod =
    status === "complete"
      ? "lp-stage-dot--complete"
      : status === "running"
        ? "lp-stage-dot--running"
        : status === "failed"
          ? "lp-stage-dot--failed"
          : "";
  return <span className={`lp-stage-dot ${mod}`} />;
}

function DurationOrExtra({ stage }: { stage: Stage }) {
  const parts: string[] = [];
  if (typeof stage.duration_s === "number") {
    parts.push(`${stage.duration_s.toFixed(1)}s`);
  }
  if (typeof stage["gaussian_count"] === "number") {
    parts.push(`${formatPointCount(stage["gaussian_count"] as number)} clusters`);
  }
  if (typeof stage["object_count"] === "number") {
    parts.push(`${stage["object_count"]} obj`);
  }
  return <span className="lp-stage-dur">{parts.join(" · ") || "—"}</span>;
}

function Stats({
  manifest,
  cloudStats,
}: {
  manifest: Manifest;
  cloudStats: { count: number; sizeMb: number } | null;
}) {
  // `captured` is the ffmpeg-extracted frame count (pre-blur-filter); set on
  // the capture stage and mirrored to stats. `used` is what survived the
  // blur filter and went into VGGT. We display "used / captured" only when
  // both are known and different — otherwise fall back to whichever number
  // we have, so older scenes that recorded only one don't show "720 / 0".
  const captureRaw = manifest.stages.capture?.["frame_count"];
  const captured =
    typeof captureRaw === "number" && captureRaw > 0
      ? captureRaw
      : manifest.stats.frame_count > 0
        ? manifest.stats.frame_count
        : null;
  const usedRaw = manifest.stages.poses["frame_count"];
  const used = typeof usedRaw === "number" ? usedRaw : null;
  let frameValue: string;
  if (used !== null && captured !== null && used !== captured) {
    frameValue = `${used} / ${captured}`;
  } else if (used !== null) {
    frameValue = `${used}`;
  } else if (captured !== null) {
    frameValue = `${captured}`;
  } else {
    frameValue = "—";
  }

  const manifestPointsRaw = manifest.stages.splat["gaussian_count"];
  const manifestPoints =
    typeof manifestPointsRaw === "number" ? manifestPointsRaw : null;
  let cloudLabel: string;
  let cloudValue: string;
  if (cloudStats) {
    cloudLabel = "points";
    cloudValue = `${formatPointCount(cloudStats.count)}`;
  } else if (manifestPoints !== null) {
    cloudLabel = "points";
    cloudValue = `${formatPointCount(manifestPoints)}`;
  } else {
    cloudLabel = "cloud";
    cloudValue = `${manifest.stats.splat_size_mb.toFixed(0)} MB`;
  }

  return (
    <div className="lp-hero-stats lp-hero-stats--side">
      <Cell label="frames" value={frameValue} />
      <Cell label="objects" value={manifest.stats.object_count} />
      <Cell label={cloudLabel} value={cloudValue} />
    </div>
  );
}

function Cell({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="lp-stat lp-stat-compact">
      <div className="lp-stat-label">{label}</div>
      <div className="lp-stat-value">{value}</div>
    </div>
  );
}
