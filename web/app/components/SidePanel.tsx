"use client";

import type { ReactNode } from "react";

import type { Annotation, Manifest, StageStatus } from "@/lib/types";
import { useUI } from "@/store/ui";
import { AnnotationEvidencePanel } from "./AnnotationEvidencePanel";
import { ChatPanel } from "./ChatPanel";
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
 * Objects / Evidence) plus the persistent chat panel. The actual section
 * content renders in a separate <SceneDrawerOverlay> that floats over the
 * 3D canvas — this column is what stays put.
 */
export function SceneSideColumn({
  manifest,
  annotations,
  messages,
  onSend,
  loading,
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

      <div className="lp-chat-column">
        <div className="lp-chat-column-head">
          <span className="lp-chat-column-title">
            Chat <em>ask the scene</em>
          </span>
        </div>
        <div className="lp-chat-column-body">
          <ChatPanel
            sceneId={manifest.scene_id}
            messages={messages}
            onSend={onSend}
            disabled={loading}
          />
        </div>
      </div>
    </aside>
  );
}

interface DrawerProps {
  manifest: Manifest;
  annotations: Annotation[];
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
  manifest,
  annotations,
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
        <ObjectsList annotations={annotations} segStatus={segStatus} />
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
        sceneId={manifest.scene_id}
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
