"use client";

import type { Manifest, Stage, StageStatus } from "@/lib/types";
import { useUI, type DrillStage } from "@/store/ui";

// Visible stages in the panel. The internal "splat" stage (voxel-downsample
// of VGGT surfels into the small splat.ply that segmentation clusters over)
// is hidden — the viewer doesn't render that output and it's confusing to
// surface to a user. Pipeline manifest still tracks it; we just don't show it.
const STAGE_ORDER: DrillStage[] = ["capture", "poses", "segmentation"];

const LABEL: Record<keyof Manifest["stages"], string> = {
  capture: "Capture",
  poses: "Reconstruction (VGGT)",
  splat: "Splat (cluster)",
  segmentation: "Segmentation",
};

function formatPointCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)} M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)} K`;
  return n.toLocaleString();
}

export function PipelineProgress({ manifest }: { manifest: Manifest }) {
  const cloudStats = useUI((s) => s.cloudStats);
  const setOpenStage = useUI((s) => s.setOpenStage);
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
                <button
                  type="button"
                  onClick={() => setOpenStage(key)}
                  className="lp-stage-row lp-stage-row--btn"
                  title="View live trace for this stage"
                >
                  <StageDot status={stage.status} />
                  <div className="lp-stage-meta">
                    <span className="lp-stage-label">{LABEL[key]}</span>
                    <span className="lp-stage-trace">view trace ↗</span>
                  </div>
                  <DurationOrExtra stage={stage} />
                </button>
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
  const captured = manifest.stats.frame_count;
  const usedRaw = manifest.stages.poses["frame_count"];
  const used = typeof usedRaw === "number" ? usedRaw : null;
  const frameValue =
    used !== null && used !== captured
      ? `${used} / ${captured}`
      : `${captured}`;

  const splatGaussiansRaw = manifest.stages.splat["gaussian_count"];
  const splatGaussians =
    typeof splatGaussiansRaw === "number" ? splatGaussiansRaw : null;
  let cloudLabel: string;
  let cloudValue: string;
  if (cloudStats) {
    cloudLabel = "points";
    cloudValue = `${formatPointCount(cloudStats.count)}`;
  } else if (splatGaussians !== null) {
    cloudLabel = "points";
    cloudValue = `${formatPointCount(splatGaussians)}`;
  } else {
    cloudLabel = "splat";
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
