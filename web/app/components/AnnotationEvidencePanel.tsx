// "What the model saw" — for the selected annotation, render each keyframe
// where SAM 3.1 produced a mask, with that mask overlaid in the
// annotation's color. The masks themselves live under
// `artifacts/scenes/<id>/masks/<annotation_id>/<frame_stem>.png` (written
// by `segmentation.lift_masks._write_cluster_masks`); we bind them as CSS
// `mask-image` on a colored div so we never have to load mask pixels into
// JS — the browser composites in C.

"use client";

import { useState } from "react";

import { evidenceFrameUrl, maskUrl } from "@/lib/api";
import type { Annotation } from "@/lib/types";

interface Props {
  sceneId: string;
  annotation: Annotation;
}

export function AnnotationEvidencePanel({ sceneId, annotation }: Props) {
  const frames = (annotation.frame_ids ?? []).slice(0, 6);
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
          />
        ))}
      </div>
    </div>
  );
}

function EvidenceTile({
  sceneId,
  annotationId,
  frameName,
  tint,
}: {
  sceneId: string;
  annotationId: string;
  frameName: string;
  tint: string;
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
    <figure className="lp-evidence-tile">
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
