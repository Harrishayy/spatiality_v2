"use client";

/**
 * 3D-anchored HTML billboards. We avoid CSS2DRenderer because the splat
 * viewer manages its own renderer and adding another DOM-mounted Three.js
 * object can fight for the same canvas. Instead: project each annotation's
 * centroid to screen space every frame and absolute-position a div.
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
    const tick = () => {
      const cam = getCamera();
      const container = containerRef.current;
      if (cam && container) {
        const rect = container.getBoundingClientRect();
        annotations.forEach((a) => {
          const el = pinsRef.current[a.id];
          if (!el) return;
          tmp.set(a.centroid[0], a.centroid[1], a.centroid[2]);
          tmp.project(cam);
          const x = (tmp.x * 0.5 + 0.5) * rect.width;
          const y = (-tmp.y * 0.5 + 0.5) * rect.height;
          const visible = tmp.z > -1 && tmp.z < 1 && x > -40 && x < rect.width + 40 && y > -40 && y < rect.height + 40;
          el.style.transform = `translate(-50%, -100%) translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
          el.style.pointerEvents = visible ? "auto" : "none";
          const cz = (cam as THREE.PerspectiveCamera).position;
          const dist = Math.hypot(
            a.centroid[0] - cz.x,
            a.centroid[1] - cz.y,
            a.centroid[2] - cz.z,
          );
          const farFade = Math.min(1, Math.max(0.25, 4 / Math.max(0.5, dist)));
          el.style.opacity = `${visible ? farFade : 0}`;
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
