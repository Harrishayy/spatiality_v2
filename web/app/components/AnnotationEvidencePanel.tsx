// "What the model saw" — for the selected annotation, render each keyframe
// where SAM 3.1 produced a mask, with that mask overlaid in the
// annotation's color. The masks themselves live under
// `artifacts/scenes/<id>/masks/<annotation_id>/<frame_stem>.png` (written
// by `segmentation.lift_masks._write_cluster_masks`); we bind them as CSS
// `mask-image` on a colored div so we never have to load mask pixels into
// JS — the browser composites in C.

"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { evidenceFrameUrl, maskUrl } from "@/lib/api";
import type { Annotation } from "@/lib/types";

interface Props {
  sceneId: string;
  annotation: Annotation;
}

export function AnnotationEvidencePanel({ sceneId, annotation }: Props) {
  const frames = (annotation.frame_ids ?? []).slice(0, 6);
  const [openFrame, setOpenFrame] = useState<string | null>(null);

  if (frames.length === 0) {
    return null;
  }

  return (
    <div className="lp-card">
      <div className="lp-card-head">
        <div className="lp-card-head-l">
          <h3 className="lp-card-title">Evidence</h3>
          <span className="lp-side-section-accent">{annotation.label}</span>
        </div>
        <span className="lp-side-section-id">
          {frames.length} keyframe{frames.length === 1 ? "" : "s"}
        </span>
      </div>
      <div className="lp-evidence-grid">
        {frames.map((frameName) => (
          <EvidenceTile
            key={frameName}
            sceneId={sceneId}
            annotationId={annotation.id}
            frameName={frameName}
            tint={annotation.color}
            onOpen={() => setOpenFrame(frameName)}
          />
        ))}
      </div>
      {openFrame && (
        <EvidenceLightbox
          sceneId={sceneId}
          annotationId={annotation.id}
          frameName={openFrame}
          tint={annotation.color}
          label={annotation.label}
          onClose={() => setOpenFrame(null)}
        />
      )}
    </div>
  );
}

function EvidenceTile({
  sceneId,
  annotationId,
  frameName,
  tint,
  onOpen,
}: {
  sceneId: string;
  annotationId: string;
  frameName: string;
  tint: string;
  onOpen: () => void;
}) {
  const frameSrc = evidenceFrameUrl(sceneId, frameName);
  const maskSrc = maskUrl(sceneId, annotationId, frameName);
  const [frameError, setFrameError] = useState(false);
  const [maskError, setMaskError] = useState(false);

  if (frameError) {
    return (
      <div
        className="lp-evidence-tile"
        title={frameName}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "var(--mono)",
          fontSize: 10,
          color: "var(--ink-500)",
        }}
      >
        no frame
      </div>
    );
  }

  // CSS mask: the colored overlay is a div whose alpha is sourced from the
  // mask PNG (white pixels become opaque tint, black pixels become
  // transparent). `mix-blend-mode: screen` brightens the masked region of
  // the image without obscuring detail. If the mask fails to load,
  // `maskError` zeroes the overlay opacity so the bare frame stays useful.
  const maskCss = `url(${maskSrc})`;
  const overlayStyle: React.CSSProperties = maskError
    ? { opacity: 0 }
    : {
        backgroundColor: tint,
        WebkitMaskImage: maskCss,
        maskImage: maskCss,
        WebkitMaskSize: "100% 100%",
        maskSize: "100% 100%",
        WebkitMaskRepeat: "no-repeat",
        maskRepeat: "no-repeat",
        opacity: 0.55,
      };

  return (
    <figure
      className="lp-evidence-tile"
      onClick={onOpen}
      role="button"
      tabIndex={0}
      title={frameName}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen();
        }
      }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={frameSrc}
        alt={`${annotationId} · ${frameName}`}
        className="lp-evidence-tile-img"
        loading="lazy"
        onError={() => setFrameError(true)}
      />
      <div
        aria-hidden
        className="lp-evidence-tile-mask"
        style={overlayStyle}
      />
      {/* Hidden img to detect mask 404 — CSS mask-image fails silently
          otherwise. The browser caches both this and the mask-image
          fetch under one URL, so it's free. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={maskSrc}
        alt=""
        className="hidden"
        onError={() => setMaskError(true)}
      />
    </figure>
  );
}

function EvidenceLightbox({
  sceneId,
  annotationId,
  frameName,
  tint,
  label,
  onClose,
}: {
  sceneId: string;
  annotationId: string;
  frameName: string;
  tint: string;
  label: string;
  onClose: () => void;
}) {
  const frameSrc = evidenceFrameUrl(sceneId, frameName);
  const maskSrc = maskUrl(sceneId, annotationId, frameName);
  const [maskError, setMaskError] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose]);

  // Portal target: only available in the browser. Bail on SSR.
  if (typeof document === "undefined") return null;

  const maskCss = `url(${maskSrc})`;
  const overlayStyle: React.CSSProperties = maskError
    ? { opacity: 0 }
    : {
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        mixBlendMode: "screen",
        backgroundColor: tint,
        WebkitMaskImage: maskCss,
        maskImage: maskCss,
        WebkitMaskSize: "100% 100%",
        maskSize: "100% 100%",
        WebkitMaskRepeat: "no-repeat",
        maskRepeat: "no-repeat",
        opacity: 0.55,
      };

  return createPortal(
    <div
      className="lp-evidence-lightbox-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <button
        type="button"
        className="lp-evidence-lightbox-close"
        onClick={onClose}
        aria-label="Close"
        title="Close"
      >
        ×
      </button>
      <div
        className="lp-evidence-lightbox-stage"
        role="dialog"
        aria-modal="true"
        aria-label={`${label} · ${frameName}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="lp-evidence-lightbox-frame">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={frameSrc}
            alt={`${annotationId} · ${frameName}`}
            className="lp-evidence-lightbox-img"
          />
          <div aria-hidden style={overlayStyle} />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={maskSrc}
            alt=""
            className="hidden"
            onError={() => setMaskError(true)}
          />
        </div>
        <div className="lp-evidence-lightbox-caption">
          <span className="lp-evidence-lightbox-label">{label}</span>
          <span className="lp-evidence-lightbox-frameid">{frameName}</span>
        </div>
      </div>
    </div>,
    document.body,
  );
}
