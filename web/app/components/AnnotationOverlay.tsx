"use client";

/**
 * 3D-anchored HTML billboards. We avoid CSS2DRenderer because the point
 * cloud viewer manages its own renderer and adding another DOM-mounted
 * Three.js object can fight for the same canvas. Instead: project each
 * annotation's centroid to screen space every frame and absolute-position
 * a div.
 */

import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { Annotation } from "@/lib/types";
import { useUI } from "@/store/ui";

interface Props {
  annotations: Annotation[];
  getCamera: () => THREE.Camera | null;
  containerRef: React.RefObject<HTMLDivElement>;
}

export function AnnotationOverlay({
  annotations,
  getCamera,
  containerRef,
}: Props) {
  const pinsRef = useRef<Record<string, HTMLDivElement | null>>({});
  const isolatedIds = useUI((s) => s.isolatedIds);
  const selectedId = useUI((s) => s.selectedId);
  const setSelected = useUI((s) => s.setSelected);
  const toggleIsolated = useUI((s) => s.toggleIsolated);

  useEffect(() => {
    let raf = 0;
    const tmp = new THREE.Vector3();
    // Distance at which pills render at their natural CSS size. Closer than
    // this clamps to MAX_SCALE; farther shrinks linearly down to MIN_SCALE.
    const REF_DIST = 1;
    const MIN_SCALE = 0.35;
    const MAX_SCALE = 1.0;
    // Pixel-size gates: an object whose bbox projects below FADE_PX starts
    // fading; below HIDE_PX it disappears entirely. Stops dense clusters
    // from carpeting the canvas when the camera pulls back.
    const FADE_PX = 10;
    const HIDE_PX = 4;
    // Hard "vicinity" gate in world units. Past FADE_DIST the label starts
    // fading; past HIDE_DIST it's gone regardless of how big the object is.
    const FADE_DIST = 3;
    const HIDE_DIST = 5;
    const tick = () => {
      const cam = getCamera();
      const container = containerRef.current;
      if (cam && container) {
        const rect = container.getBoundingClientRect();
        const persp = cam as THREE.PerspectiveCamera;
        const fovRad = ((persp.fov ?? 50) * Math.PI) / 180;
        const pxPerWorldAtUnitDist = rect.height / (2 * Math.tan(fovRad / 2));
        annotations.forEach((a) => {
          const el = pinsRef.current[a.id];
          if (!el) return;
          tmp.set(a.centroid[0], a.centroid[1], a.centroid[2]);
          tmp.project(cam);
          const x = (tmp.x * 0.5 + 0.5) * rect.width;
          const y = (-tmp.y * 0.5 + 0.5) * rect.height;
          const inFrustum =
            tmp.z > -1 && tmp.z < 1 && x > -40 && x < rect.width + 40 && y > -40 && y < rect.height + 40;
          const cz = persp.position;
          const dist = Math.hypot(
            a.centroid[0] - cz.x,
            a.centroid[1] - cz.y,
            a.centroid[2] - cz.z,
          );
          const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, REF_DIST / Math.max(0.5, dist)));
          el.style.transform = `translate(-50%, -100%) translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) scale(${scale.toFixed(3)})`;
          const [bmin, bmax] = a.bbox;
          const bboxDiag = Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]);
          const sizePx = (bboxDiag * pxPerWorldAtUnitDist) / Math.max(0.5, dist);
          const sizeFade =
            sizePx >= FADE_PX ? 1 : sizePx <= HIDE_PX ? 0 : (sizePx - HIDE_PX) / (FADE_PX - HIDE_PX);
          const distFade =
            dist <= FADE_DIST ? 1 : dist >= HIDE_DIST ? 0 : (HIDE_DIST - dist) / (HIDE_DIST - FADE_DIST);
          const farFade = Math.min(1, Math.max(0.25, REF_DIST / Math.max(0.5, dist)));
          const visible = inFrustum && sizeFade > 0 && distFade > 0;
          el.style.pointerEvents = visible ? "auto" : "none";
          el.style.opacity = `${visible ? farFade * sizeFade * distFade : 0}`;
        });
      }
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [annotations, getCamera, containerRef]);

  return (
    <>
      {annotations.map((a) => {
        const dim = isolatedIds.size > 0 && !isolatedIds.has(a.id);
        const selected = selectedId === a.id;
        return (
          <div
            key={a.id}
            ref={(el) => {
              pinsRef.current[a.id] = el;
            }}
            className="absolute left-0 top-0 select-none transition-opacity duration-200"
            style={{ willChange: "transform, opacity" }}
          >
            <button
              onClick={() => setSelected(selected ? null : a.id)}
              onDoubleClick={() => toggleIsolated(a.id)}
              className={[
                "lp-anno-pill",
                selected ? "lp-anno-pill--selected" : "",
                dim ? "lp-anno-pill--dim" : "",
              ].join(" ")}
              aria-label={`Annotation: ${a.label}`}
            >
              <span
                className="lp-anno-pill-dot"
                style={{ backgroundColor: a.color }}
              />
              <span className="lp-anno-pill-id">
                {a.id.replace("obj_", "#")}
              </span>
              <span>{a.label}</span>
              <span className="lp-anno-pill-conf">
                {(a.confidence * 100).toFixed(0)}%
              </span>
            </button>
          </div>
        );
      })}
    </>
  );
}
