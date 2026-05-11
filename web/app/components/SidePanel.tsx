"use client";

import { useState, type ReactNode } from "react";

import type {
  Annotation,
  DiscardReason,
  DiscardStage,
  DiscardedAnnotation,
  Manifest,
  StageStatus,
} from "@/lib/types";
import { useUI } from "@/store/ui";
import { AnnotationEvidencePanel } from "./AnnotationEvidencePanel";
import { PipelineProgress } from "./PipelineProgress";

export type SceneSection = "pipeline" | "objects" | "evidence";

interface ColumnProps {
  manifest: Manifest;
  annotations: Annotation[];
  messages: import("@/lib/types").ChatMessage[];
  onSend: (text: string) => void;
  loading: boolean;
  openSection: SceneSection | null;
  onToggleSection: (s: SceneSection) => void;
}

/**
 * Right-side column on the scenes page: a vertical icon rail (Pipeline /
 * Objects / Evidence). The actual section content renders in a separate
 * <SceneDrawerOverlay> that floats over the 3D canvas — this column is
 * what stays put.
 */
export function SceneSideColumn({
  manifest: _manifest,
  annotations,
  messages: _messages,
  onSend: _onSend,
  loading: _loading,
  openSection,
  onToggleSection,
}: ColumnProps) {
  const selectedId = useUI((s) => s.selectedId);
  const selected =
    selectedId == null ? null : annotations.find((a) => a.id === selectedId) ?? null;

  return (
    <aside className="lp-scene-aside">
      <div className="lp-rail">
        <RailButton
          label="Pipeline"
          icon="◐"
          active={openSection === "pipeline"}
          onClick={() => onToggleSection("pipeline")}
        />
        <RailButton
          label="Objects"
          icon="⊟"
          badge={annotations.length || undefined}
          active={openSection === "objects"}
          onClick={() => onToggleSection("objects")}
        />
        <RailButton
          label="Evidence"
          icon="◳"
          active={openSection === "evidence"}
          disabled={!selected}
          onClick={() => onToggleSection("evidence")}
        />
      </div>
    </aside>
  );
}

interface DrawerProps {
  // URL-param scene id — source of truth for artifact paths. Don't fall
  // back to manifest.scene_id: renamed/imported scenes (e.g. demo_piece)
  // keep the original Modal job id in their manifest, so trusting it
  // routes evidence URLs to a stale, nonexistent directory.
  sceneId: string;
  manifest: Manifest;
  annotations: Annotation[];
  discarded: DiscardedAnnotation[];
  segStatus: StageStatus;
  openSection: SceneSection | null;
  onClose: () => void;
}

/**
 * Floating drawer that slides in from the right edge of the canvas. Renders
 * the content for the currently-open section (Pipeline / Objects / Evidence)
 * over the 3D viewer; chat stays untouched in the persistent column.
 */
export function SceneDrawerOverlay({
  sceneId,
  manifest,
  annotations,
  discarded,
  segStatus,
  openSection,
  onClose,
}: DrawerProps) {
  const isolatedIds = useUI((s) => s.isolatedIds);
  const clearIsolated = useUI((s) => s.clearIsolated);
  const selectedId = useUI((s) => s.selectedId);
  const selected =
    selectedId == null ? null : annotations.find((a) => a.id === selectedId) ?? null;

  const title =
    openSection === "pipeline"
      ? { name: "Pipeline", accent: "live" }
      : openSection === "objects"
        ? { name: "Objects", accent: "scene" }
        : openSection === "evidence"
          ? { name: "Evidence", accent: selected?.label ?? "" }
          : null;

  let body: ReactNode = null;
  if (openSection === "pipeline") {
    body = <PipelineProgress manifest={manifest} />;
  } else if (openSection === "objects") {
    body = (
      <>
        <ObjectsTabs
          annotations={annotations}
          discarded={discarded}
          segStatus={segStatus}
        />
        {isolatedIds.size > 0 && (
          <button
            onClick={clearIsolated}
            className="lp-btn lp-btn-ghost lp-btn-sm self-start"
          >
            ↺ Clear isolation ({isolatedIds.size})
          </button>
        )}
      </>
    );
  } else if (openSection === "evidence" && selected) {
    body = (
      <AnnotationEvidencePanel
        sceneId={sceneId}
        annotation={selected}
      />
    );
  } else if (openSection === "evidence") {
    body = (
      <p className="lp-modal-hint">
        Click an object marker on the scene (or open the Objects panel) to see
        its evidence frames here.
      </p>
    );
  }

  return (
    <div
      className={["lp-drawer", openSection ? "lp-drawer--open" : ""].join(" ")}
      aria-hidden={!openSection}
    >
      <div className="lp-drawer-head">
        <span className="lp-drawer-title">
          <span className="lp-drawer-title-name">{title?.name ?? ""}</span>
          {title?.accent && (
            <span className="lp-drawer-title-accent">{title.accent}</span>
          )}
        </span>
        <button
          type="button"
          className="lp-drawer-close"
          onClick={onClose}
          aria-label="Close"
          title="Close"
        >
          ×
        </button>
      </div>
      <div className="lp-drawer-body">{body}</div>
    </div>
  );
}

function RailButton({
  label,
  icon,
  badge,
  active,
  disabled,
  onClick,
}: {
  label: string;
  icon: string;
  badge?: number;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={[
        "lp-rail-btn",
        active ? "lp-rail-btn--on" : "",
      ].join(" ")}
      disabled={disabled}
      onClick={onClick}
      aria-pressed={active}
      title={disabled ? `${label} (select an object first)` : label}
      style={disabled ? { opacity: 0.4, cursor: "not-allowed" } : undefined}
    >
      <span className="lp-rail-btn-icon" aria-hidden>
        {icon}
      </span>
      <span className="lp-rail-btn-label">{label}</span>
      {badge != null && badge > 0 && (
        <span className="lp-rail-btn-badge">{badge > 99 ? "99+" : badge}</span>
      )}
    </button>
  );
}

function ObjectsList({
  annotations,
  segStatus,
}: {
  annotations: Annotation[];
  segStatus: StageStatus;
}) {
  const selectedId = useUI((s) => s.selectedId);
  const setSelected = useUI((s) => s.setSelected);
  const isolatedIds = useUI((s) => s.isolatedIds);
  const toggleIsolated = useUI((s) => s.toggleIsolated);

  if (annotations.length === 0) {
    let label: string;
    if (segStatus === "running") label = "Segmenting…";
    else if (segStatus === "pending") label = "Segmentation pending.";
    else if (segStatus === "failed") label = "Segmentation failed.";
    else label = "No objects found.";
    return (
      <div className="lp-objects-empty">
        {segStatus === "running" && (
          <span className="lp-status-dot lp-status-dot--warn" />
        )}
        <span>{label}</span>
      </div>
    );
  }
  return (
    <div className="lp-objects-list">
      {annotations.map((a) => {
        const selected = selectedId === a.id;
        const isolated = isolatedIds.has(a.id);
        return (
          <div
            key={a.id}
            className={[
              "lp-objects-row",
              selected ? "lp-objects-row--selected" : "",
              isolated ? "lp-objects-row--isolated" : "",
            ].join(" ")}
            onClick={() => setSelected(selected ? null : a.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                setSelected(selected ? null : a.id);
              }
            }}
          >
            <span
              className="lp-objects-dot"
              style={{ backgroundColor: a.color }}
            />
            <span className="lp-objects-label">{a.label}</span>
            <span className="lp-objects-conf">
              {(a.confidence * 100).toFixed(0)}%
            </span>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                toggleIsolated(a.id);
              }}
              title={isolated ? "Show all" : "Isolate"}
              className={[
                "lp-objects-iso",
                isolated ? "lp-objects-iso--on" : "",
              ].join(" ")}
              aria-label={isolated ? "Show all" : "Isolate"}
            >
              ◉
            </button>
          </div>
        );
      })}
    </div>
  );
}

const DISCARD_REASON_LABEL: Record<DiscardReason, string> = {
  short_tracklet: "short tracklet",
  multiview_filter: "multi-view inconsistent",
  "3d_coherence": "3D incoherent",
  reprojection: "reprojection failed",
  merged_3d: "merged in 3D",
  scene_label: "scene label",
  low_confidence: "low confidence",
  oversize: "oversize",
  merged_duplicate: "merged duplicate",
};

const STAGE_LABEL: Record<DiscardStage, string> = {
  gdino: "Detection",
  lift: "3D Lift",
  postprocess: "Postprocess",
};

const STAGE_BLURB: Record<DiscardStage, string> = {
  gdino: "Cut after Grounding-DINO + IoU tracklets — too few frames to trust.",
  lift: "Cut during 3D lifting — geometry rejected the track.",
  postprocess: "Cut after VLM labelling — unsupported label, low confidence, or duplicate.",
};

const STAGE_ORDER: DiscardStage[] = ["gdino", "lift", "postprocess"];

type ObjectsTab = "confirmed" | "discarded";

function ObjectsTabs({
  annotations,
  discarded,
  segStatus,
}: {
  annotations: Annotation[];
  discarded: DiscardedAnnotation[];
  segStatus: StageStatus;
}) {
  const [tab, setTab] = useState<ObjectsTab>("confirmed");
  return (
    <div className="lp-objects-tabs">
      <div className="lp-objects-tabbar" role="tablist" aria-label="Objects">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "confirmed"}
          className={[
            "lp-objects-tab",
            tab === "confirmed" ? "lp-objects-tab--on" : "",
          ].join(" ")}
          onClick={() => setTab("confirmed")}
        >
          Confirmed
          <span className="lp-objects-tab-count">{annotations.length}</span>
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "discarded"}
          className={[
            "lp-objects-tab",
            tab === "discarded" ? "lp-objects-tab--on" : "",
          ].join(" ")}
          onClick={() => setTab("discarded")}
        >
          Discarded
          <span className="lp-objects-tab-count">{discarded.length}</span>
        </button>
      </div>
      {tab === "confirmed" ? (
        <ObjectsList annotations={annotations} segStatus={segStatus} />
      ) : (
        <DiscardedList discarded={discarded} segStatus={segStatus} />
      )}
    </div>
  );
}

function DiscardedList({
  discarded,
  segStatus,
}: {
  discarded: DiscardedAnnotation[];
  segStatus: StageStatus;
}) {
  if (discarded.length === 0) {
    let label: string;
    if (segStatus === "running") label = "Segmenting…";
    else if (segStatus === "pending") label = "Segmentation pending.";
    else if (segStatus === "failed") label = "Segmentation failed.";
    else label = "Nothing was discarded.";
    return (
      <div className="lp-objects-empty">
        {segStatus === "running" && (
          <span className="lp-status-dot lp-status-dot--warn" />
        )}
        <span>{label}</span>
      </div>
    );
  }

  const byStage = new Map<DiscardStage, DiscardedAnnotation[]>();
  for (const a of discarded) {
    const arr = byStage.get(a.stage) ?? [];
    arr.push(a);
    byStage.set(a.stage, arr);
  }

  return (
    <div className="lp-objects-discarded">
      {STAGE_ORDER.map((stage) => {
        const items = byStage.get(stage);
        if (!items || items.length === 0) return null;
        return (
          <section key={stage} className="lp-objects-discarded-section">
            <header className="lp-objects-discarded-section-head">
              <span className="lp-objects-discarded-section-name">
                {STAGE_LABEL[stage]}
              </span>
              <span className="lp-objects-discarded-section-count">
                {items.length}
              </span>
            </header>
            <p className="lp-objects-discarded-section-blurb">
              {STAGE_BLURB[stage]}
            </p>
            <div className="lp-objects-list">
              {items.map((a) => (
                <DiscardedRow key={a.id} a={a} />
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}

function DiscardedRow({ a }: { a: DiscardedAnnotation }) {
  const reason = DISCARD_REASON_LABEL[a.discard_reason] ?? a.discard_reason;
  const conf = a.confidence != null ? `${(a.confidence * 100).toFixed(0)}%` : null;
  const frames = a.n_frames ?? a.frame_ids?.length ?? null;
  return (
    <div
      className="lp-objects-row lp-objects-row--discarded"
      title={a.discard_detail ?? ""}
    >
      <span
        className="lp-objects-dot"
        style={{ backgroundColor: a.color ?? "#555", opacity: 0.5 }}
      />
      <div className="lp-objects-discarded-meta">
        <span className="lp-objects-label">{a.label || "unknown"}</span>
        <span className="lp-objects-discarded-reason">{reason}</span>
        {a.discard_detail && (
          <span className="lp-objects-discarded-detail">{a.discard_detail}</span>
        )}
      </div>
      <span className="lp-objects-conf">
        {conf ?? (frames != null ? `${frames}f` : "—")}
      </span>
    </div>
  );
}
