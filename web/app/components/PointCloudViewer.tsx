"use client";

/**
 * Point cloud viewer: renders the per-pixel VGGT cloud (`points.ply`, ~12 M
 * raw coloured points) directly via three.js `Points`. Matches the aesthetic
 * of VGGT's reference `demo_viser.py` — crisp opaque coloured pixels, no
 * blending, surface impression emerges from per-pixel density.
 *
 * Rationale: we considered anisotropic Gaussian splatting via @sparkjsdev/spark
 * but it forces a 2 M-Gaussian budget (each Gaussian = 17 floats = 68 B in GPU
 * memory), which would mean voxel-downsampling the raw VGGT cloud 10× and then
 * over-blending the survivors into surface goo. Points use 6 B each
 * (3 float32 xyz + 3 uint8 rgb), so the full 12 M cloud fits in a 76 MB GPU
 * buffer with no blending.
 */

import { useEffect, useRef, useState } from "react";
// Named imports (instead of `import * as THREE`) so Turbopack tree-shakes
// three.js properly — bundler memory during /scenes/[id] compile drops
// dramatically (the namespace import was forcing every three module to be
// held in memory, ~600KB minified).
import {
  AmbientLight,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  Camera,
  Color,
  DirectionalLight,
  EdgesGeometry,
  GridHelper,
  LineBasicMaterial,
  LineSegments,
  PerspectiveCamera,
  Points,
  Raycaster,
  Scene,
  ShaderMaterial,
  Spherical,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import type { Annotation } from "@/lib/types";
import { useUI } from "@/store/ui";
import { AnnotationOverlay } from "./AnnotationOverlay";

// "end_header\n" — used as a needle in the streaming parser to detect when the
// PLY ASCII header is fully received and we can transition to body parsing.
const END_HEADER_NEEDLE = new TextEncoder().encode("end_header\n");

/** Find the offset just past `end_header\n` in a possibly-incomplete byte
 *  buffer. Returns -1 if the marker isn't present yet (caller should keep
 *  reading more chunks). Used by the streaming PLY parser. */
function findEndHeader(buf: Uint8Array): number {
  const n = END_HEADER_NEEDLE.length;
  outer: for (let i = 0; i + n <= buf.length; i++) {
    for (let j = 0; j < n; j++) {
      if (buf[i + j] !== END_HEADER_NEEDLE[j]) continue outer;
    }
    return i + n;
  }
  return -1;
}

/** Parse the ASCII header of a binary little-endian PLY file. Returns the
 *  total vertex `count`, the body `stride` (bytes per vertex), and `layout`
 *  — the byte offsets within each vertex record for x/y/z (float32) and
 *  r/g/b (uchar). Handles the dialect of points.ply that we produce. */
function parsePlyHeader(headerStr: string): {
  count: number;
  stride: number;
  layout: {
    x: number;
    y: number;
    z: number;
    r: number;
    g: number;
    b: number;
    /** Offset of the per-point float32 `confidence` field, or null if the
     *  PLY does not include one. Present in the depth-derived points.ply
     *  emitted by inference/inference/poses.py; absent in older bundles. */
    conf: number | null;
  };
} {
  if (!headerStr.includes("format binary_little_endian 1.0")) {
    throw new Error("PLY: only binary_little_endian 1.0 supported");
  }

  let count = 0;
  const props: { name: string; type: string }[] = [];
  for (const line of headerStr.split("\n")) {
    if (line.startsWith("element vertex ")) {
      count = parseInt(line.split(/\s+/)[2], 10);
    } else if (line.startsWith("property ")) {
      const parts = line.split(/\s+/);
      props.push({ type: parts[1], name: parts[2] });
    }
  }
  if (!count) throw new Error("PLY: vertex count = 0");

  // Only float32 + uchar are supported (covers every PLY dialect we emit).
  const sizeOf = (t: string) =>
    t === "float" || t === "float32"
      ? 4
      : t === "uchar" || t === "uint8"
        ? 1
        : (() => {
            throw new Error(`PLY: unsupported property type ${t}`);
          })();
  let stride = 0;
  const offsetByName: Record<string, { off: number; type: string }> = {};
  for (const p of props) {
    offsetByName[p.name] = { off: stride, type: p.type };
    stride += sizeOf(p.type);
  }

  if (!offsetByName.x || !offsetByName.y || !offsetByName.z) {
    throw new Error("PLY: missing x/y/z");
  }
  if (
    !offsetByName.red ||
    !offsetByName.green ||
    !offsetByName.blue ||
    (offsetByName.red.type !== "uchar" && offsetByName.red.type !== "uint8")
  ) {
    throw new Error("PLY: expected uchar red/green/blue (points.ply layout)");
  }

  // confidence is optional — points.ply written by poses.py since the depth
  // pipeline lands includes it; older bundles do not.
  const confEntry = offsetByName.confidence;
  const confOff =
    confEntry && (confEntry.type === "float" || confEntry.type === "float32")
      ? confEntry.off
      : null;

  return {
    count,
    stride,
    layout: {
      x: offsetByName.x.off,
      y: offsetByName.y.off,
      z: offsetByName.z.off,
      r: offsetByName.red.off,
      g: offsetByName.green.off,
      b: offsetByName.blue.off,
      conf: confOff,
    },
  };
}

type DebugState = {
  status: "idle" | "fetching" | "parsing" | "started" | "error";
  url?: string;
  fetchBytes?: number;
  fetchTotal?: number;
  fetchMs?: number;
  parseMs?: number;
  startedMs?: number;
  sceneCount?: number;
  webglOk?: boolean;
  containerSize?: [number, number];
  error?: string;
  errorStack?: string;
  log: string[];
};


interface Props {
  pointsUrl: string;
  annotations: Annotation[];
  /** Set true when points.ply is empty/0-vertex; we'll show a placeholder. */
  emptyCloud?: boolean;
}

export function PointCloudViewer({ pointsUrl, annotations, emptyCloud }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<{
    cancel: () => void;
    camera: Camera | null;
  }>({ cancel: () => undefined, camera: null });
  const setCamera = useUI((s) => s.setCamera);
  const setCloudStats = useUI((s) => s.setCloudStats);
  const selectedId = useUI((s) => s.selectedId);
  const renderMode = useUI((s) => s.renderMode);
  const showAnnotations = useUI((s) => s.showAnnotations);
  const [debug, setDebug] = useState<DebugState>({ status: "idle", log: [] });
  const debugRef = useRef(debug);
  debugRef.current = debug;
  const pushDebug = (patch: Partial<DebugState>, line?: string) => {
    const next: DebugState = {
      ...debugRef.current,
      ...patch,
      log: line
        ? [...debugRef.current.log, `${new Date().toISOString().slice(11, 23)} ${line}`].slice(-30)
        : debugRef.current.log,
    };
    debugRef.current = next;
    setDebug(next);
    if (line) console.log(`[PointCloudViewer] ${line}`, patch);
  };

  // Refs so the heavy effect can read the latest annotations + setCamera
  // WITHOUT taking them as dependencies. annotations is a fresh array on
  // every parent render (page does `annotations.data ?? []`) — if we put it
  // in deps the viewer remounts on every 2s manifest poll, each in-flight
  // load rejects with "Scene disposed", and the dispose path stomps on DOM
  // nodes mid-load. Critical fix.
  const annotationsRef = useRef<Annotation[]>(annotations);
  useEffect(() => {
    annotationsRef.current = annotations;
  }, [annotations]);
  const setCameraRef = useRef(setCamera);
  useEffect(() => {
    setCameraRef.current = setCamera;
  }, [setCamera]);
  const setCloudStatsRef = useRef(setCloudStats);
  useEffect(() => {
    setCloudStatsRef.current = setCloudStats;
  }, [setCloudStats]);

  // Fly-to request: set externally (annotation click, preset button), the
  // render loop interpolates `target` and `radius` toward it each frame.
  const flyToRef = useRef<{
    target: [number, number, number];
    radius: number;
  } | null>(null);

  // Bridges from the component scope into the heavy useEffect. The effect
  // mounts once per scene; everything below changes far more often (mode
  // toggles) and we don't want to remount on those.
  const shaderMatRef = useRef<ShaderMaterial | null>(null);
  const cloudRef = useRef<Points | null>(null);
  const markDirtyRef = useRef<(() => void) | null>(null);
  // Sibling click handler reads the latest store values via these refs so it
  // doesn't need to re-bind on every state change.
  const renderModeRef = useRef(renderMode);
  useEffect(() => { renderModeRef.current = renderMode; }, [renderMode]);

  // Imperative API the surrounding UI (zoom buttons, minimap) calls into.
  type ViewerApi = {
    zoom: (factor: number) => void;
    setTarget: (xyz: [number, number, number], radius?: number) => void;
    reset: () => void;
  };
  const apiRef = useRef<ViewerApi | null>(null);

  // Subset of points for the minimap (downsampled, kept in React state so
  // the minimap re-renders when the cloud loads). Camera position comes from
  // the existing zustand store, so the minimap re-renders ~60Hz for free.
  const [miniPoints, setMiniPoints] = useState<{
    xz: Float32Array; // (M*2,) flattened (x, z) pairs
    rgb: Float32Array; // (M*3,)
    bounds: { minX: number; maxX: number; minZ: number; maxZ: number };
  } | null>(null);
  // Render mode → shader uMode. RGB = 0 / depth = 1 / confidence = 2.
  useEffect(() => {
    const mat = shaderMatRef.current;
    if (!mat) return;
    mat.uniforms.uMode.value =
      renderMode === "depth" ? 1 : renderMode === "confidence" ? 2 : 0;
    markDirtyRef.current?.();
  }, [renderMode]);


  useEffect(() => {
    if (!selectedId) return;
    const a = annotationsRef.current.find((x) => x.id === selectedId);
    if (!a) return;
    const [lo, hi] = a.bbox;
    const ext = Math.max(
      hi[0] - lo[0],
      hi[1] - lo[1],
      hi[2] - lo[2],
    );
    flyToRef.current = {
      target: a.centroid as [number, number, number],
      // Pull camera in proportional to the object's extent — small things =
      // close-up, big things = farther back. Multiplier kept under 1 so the
      // camera lands inside/at the surface of the bbox; the previous 2.2×
      // floated too far out to inspect interiors.
      radius: Math.max(0.06, ext * 0.85),
    };
  }, [selectedId]);

  useEffect(() => {
    if (!containerRef.current) return;
    let disposed = false;

    // Snapshot annotations once for initial camera framing. Live updates after
    // mount go through AnnotationOverlay (it re-renders on prop change).
    const initial = initialView(annotationsRef.current);

    const useViewer = !emptyCloud;
    let raf = 0;

    const cleanup: (() => void)[] = [];

    const c0 = containerRef.current;
    pushDebug(
      {
        status: "idle",
        url: pointsUrl,
        containerSize: [c0.clientWidth, c0.clientHeight],
      },
      `mount: useViewer=${useViewer} url=${pointsUrl} container=${c0.clientWidth}x${c0.clientHeight}`,
    );

    if (useViewer) {
      // WebGL probe — confirms the canvas can get a context at all.
      try {
        const probe = document.createElement("canvas");
        const ctx =
          probe.getContext("webgl2") || probe.getContext("webgl");
        pushDebug({ webglOk: !!ctx }, `webgl: ${ctx ? "OK" : "MISSING"}`);
      } catch (e) {
        pushDebug({ webglOk: false }, `webgl probe threw: ${e}`);
      }

      const container = containerRef.current;
      // Perf knobs for 12 M-point clouds on a laptop GPU:
      //   antialias: false — at 12 M points, MSAA fragment cost dominates
      //     interactivity. The visual gain on dense pixel grain is invisible.
      //   pixelRatio: min(DPR, 1.5) — Retina detail without paying 4× fragment
      //     cost. With opaque points there's nothing to anti-alias anyway.
      const renderer = new WebGLRenderer({ antialias: false, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setClearColor(0x0a0a0c, 1);
      container.appendChild(renderer.domElement);

      const scene = new Scene();
      const camera = new PerspectiveCamera(
        50,
        container.clientWidth / container.clientHeight,
        0.05,
        100,
      );
      camera.position.set(...initial.position);
      camera.lookAt(...initial.lookAt);
      sceneRef.current.camera = camera;

      // Light scene scaffolding so the cloud has spatial context.
      scene.add(new GridHelper(8, 16, 0x3a3a46, 0x1a1a20));

      // Camera controls:
      //   left-drag   → orbit around `target` (theta/phi)
      //   right-drag  → pan `target` in screen space (move into the cloud)
      //   shift+drag  → also pan (for trackpad users without right-button)
      //   wheel       → dolly (radius); preventDefault on passive:false so the
      //                  browser doesn't scroll the page while you zoom
      //   keys WASD   → pan target horizontally; QE → pan vertically
      const target = new Vector3(...initial.lookAt);
      const sph = new Spherical();
      sph.setFromVector3(camera.position.clone().sub(target));
      let dragging: "orbit" | "pan" | null = null;
      let lastX = 0;
      let lastY = 0;
      const onDown = (e: PointerEvent) => {
        dragging =
          e.button === 2 || e.shiftKey || e.metaKey || e.ctrlKey
            ? "pan"
            : "orbit";
        lastX = e.clientX;
        lastY = e.clientY;
        (e.target as Element)?.setPointerCapture?.(e.pointerId);
      };
      const onMove = (e: PointerEvent) => {
        if (!dragging) return;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;
        if (dragging === "orbit") {
          sph.theta -= dx * 0.005;
          sph.phi = Math.max(0.05, Math.min(Math.PI - 0.05, sph.phi - dy * 0.005));
        } else {
          // Pan in camera-screen space scaled by radius so it feels
          // proportional at any zoom level.
          const right = new Vector3();
          const up = new Vector3();
          camera.matrixWorld.extractBasis(right, up, new Vector3());
          const k = sph.radius * 0.0015;
          target.addScaledVector(right, -dx * k);
          target.addScaledVector(up, dy * k);
        }
      };
      const onUp = () => {
        dragging = null;
      };
      const onWheel = (e: WheelEvent) => {
        e.preventDefault();
        const factor = Math.exp(e.deltaY * 0.0015);
        sph.radius = Math.max(0.05, Math.min(40, sph.radius * factor));
      };
      const onContextMenu = (e: MouseEvent) => e.preventDefault();
      // Double-click to recenter the orbit target on the picked point. We
      // raycast against the streaming Points cloud; the threshold is tuned
      // generously vs. the rendered point size so picking works even when
      // the cursor lands between sparse pixels.
      const raycaster = new Raycaster();
      const ndc = new Vector2();
      raycaster.params.Points = { threshold: 0.02 };
      const onDoubleClick = (e: MouseEvent) => {
        const cloud = cloudRef.current;
        if (!cloud) return;
        const rect = dom.getBoundingClientRect();
        ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(ndc, camera);
        // Scale the picking threshold with the current orbit radius so the
        // user gets a similar hit-area whether zoomed in or out.
        raycaster.params.Points!.threshold = Math.max(0.005, sph.radius * 0.01);
        const hits = raycaster.intersectObject(cloud, false);
        if (hits.length === 0) return;
        const p = hits[0].point;
        flyToRef.current = {
          target: [p.x, p.y, p.z],
          // Preserve current zoom — only the pivot moves.
          radius: sph.radius,
        };
      };
      // True when focus is in a text field — chat input, modal, etc. WASD
      // and friends would otherwise scroll the camera every time the user
      // typed an "a" or "s".
      const isTypingTarget = (e: KeyboardEvent): boolean => {
        const t = e.target as HTMLElement | null;
        if (!t) return false;
        if (t.isContentEditable) return true;
        const tag = t.tagName;
        return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
      };
      const onKey = (e: KeyboardEvent) => {
        if (isTypingTarget(e)) return;
        const k = sph.radius * 0.04;
        const right = new Vector3();
        const up = new Vector3();
        const fwd = new Vector3();
        camera.matrixWorld.extractBasis(right, up, fwd);
        fwd.negate();
        switch (e.key.toLowerCase()) {
          case "w": target.addScaledVector(fwd, k); break;
          case "s": target.addScaledVector(fwd, -k); break;
          case "a": target.addScaledVector(right, -k); break;
          case "d": target.addScaledVector(right, k); break;
          case "q": target.addScaledVector(up, -k); break;
          case "e": target.addScaledVector(up, k); break;
          case "r":
            target.set(...initial.lookAt);
            sph.setFromVector3(
              new Vector3(...initial.position).sub(target),
            );
            break;
          default: return;
        }
      };
      // Render-on-demand: only redraw when something changed. Each input
      // event flips `dirty` true; the tick clears it after rendering. Idle
      // GPU usage drops to zero when nothing is moving.
      let dirty = true;
      const markDirty = () => {
        dirty = true;
      };
      // Expose markDirty to sibling effects (mode toggles etc.) so they can
      // schedule a redraw without owning the render loop.
      markDirtyRef.current = markDirty;
      cleanup.push(() => {
        if (markDirtyRef.current === markDirty) markDirtyRef.current = null;
      });

      const dom = renderer.domElement;
      dom.addEventListener("pointerdown", onDown);
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      dom.addEventListener("wheel", onWheel, { passive: false });
      dom.addEventListener("contextmenu", onContextMenu);
      dom.addEventListener("dblclick", onDoubleClick);
      window.addEventListener("keydown", onKey);
      // Any user interaction → schedule a render.
      dom.addEventListener("pointerdown", markDirty);
      window.addEventListener("pointermove", markDirty);
      dom.addEventListener("wheel", markDirty);
      // Skip key-driven redraws when the user is typing in an input — those
      // keystrokes don't change the camera (see isTypingTarget guard above).
      const markDirtyKey = (e: KeyboardEvent) => {
        if (isTypingTarget(e)) return;
        markDirty();
      };
      window.addEventListener("keydown", markDirtyKey);
      window.addEventListener("resize", markDirty);
      // Pause completely when tab is hidden; redraw once when it comes back.
      const onVisibility = () => {
        if (document.visibilityState === "visible") dirty = true;
      };
      document.addEventListener("visibilitychange", onVisibility);

      // Imperative API for outside UI (zoom buttons, minimap).
      apiRef.current = {
        zoom: (factor) => {
          sph.radius = Math.max(0.05, Math.min(40, sph.radius * factor));
        },
        setTarget: (xyz, radius) => {
          flyToRef.current = {
            target: xyz,
            radius: radius ?? sph.radius,
          };
        },
        reset: () => {
          target.set(...initial.lookAt);
          sph.setFromVector3(
            new Vector3(...initial.position).sub(target),
          );
          flyToRef.current = null;
        },
      };
      cleanup.push(() => {
        apiRef.current = null;
      });

      const onResize = () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      };
      window.addEventListener("resize", onResize);

      // Render-on-demand at 30 fps cap. Pre-allocated working vectors to
      // avoid per-frame GC churn. Each tick:
      //   1. Skip entirely if the tab is hidden.
      //   2. Compute camera target/offset; if anything moved, mark dirty.
      //   3. Push camera state + render only when dirty (or while a fly-to
      //      is in flight).
      const FRAME_INTERVAL_MS = 1000 / 30;
      const MOVE_EPS = 1e-6;
      const tmpVec = new Vector3();
      const tmpDir = new Vector3();
      const tmpFlyDst = new Vector3();
      let lastRender = 0;
      let lastTheta = sph.theta;
      let lastPhi = sph.phi;
      let lastRadius = sph.radius;
      let lastTargetX = target.x;
      let lastTargetY = target.y;
      let lastTargetZ = target.z;
      const tick = (now: number) => {
        if (document.visibilityState !== "visible") {
          raf = requestAnimationFrame(tick);
          return;
        }
        const fly = flyToRef.current;
        if (fly) {
          tmpFlyDst.set(fly.target[0], fly.target[1], fly.target[2]);
          target.lerp(tmpFlyDst, 0.12);
          sph.radius += (fly.radius - sph.radius) * 0.12;
          if (target.distanceToSquared(tmpFlyDst) < 1e-5 && Math.abs(sph.radius - fly.radius) < 1e-3) {
            flyToRef.current = null;
          }
          dirty = true;
        }
        if (
          Math.abs(sph.theta - lastTheta) > MOVE_EPS ||
          Math.abs(sph.phi - lastPhi) > MOVE_EPS ||
          Math.abs(sph.radius - lastRadius) > MOVE_EPS ||
          Math.abs(target.x - lastTargetX) > MOVE_EPS ||
          Math.abs(target.y - lastTargetY) > MOVE_EPS ||
          Math.abs(target.z - lastTargetZ) > MOVE_EPS
        ) {
          dirty = true;
        }

        if (dirty && now - lastRender >= FRAME_INTERVAL_MS) {
          tmpVec.setFromSpherical(sph);
          camera.position.copy(target).add(tmpVec);
          camera.lookAt(target);
          camera.getWorldDirection(tmpDir);
          setCameraRef.current(
            [camera.position.x, camera.position.y, camera.position.z],
            [tmpDir.x, tmpDir.y, tmpDir.z],
          );
          renderer.render(scene, camera);
          lastRender = now;
          lastTheta = sph.theta;
          lastPhi = sph.phi;
          lastRadius = sph.radius;
          lastTargetX = target.x;
          lastTargetY = target.y;
          lastTargetZ = target.z;
          dirty = false;
        }
        raf = requestAnimationFrame(tick);
      };
      tick(performance.now());

      // Three.js Points renderer over the parsed cloud. Coordinate frame:
      // PLY xyz passes through unchanged so annotation centroids (recorded in
      // the same frame upstream) overlay correctly without a transform.
      // We use a custom ShaderMaterial (not PointsMaterial) so a uniform
      // can switch between RGB / depth-heatmap / confidence-heatmap modes
      // without re-uploading any per-vertex data.
      let pointGeo: BufferGeometry | null = null;
      let pointMat: ShaderMaterial | null = null;
      let cloud: Points | null = null;
      // Group of three.js objects added to the scene that need disposing.
      const ownedObjects: { dispose: () => void }[] = [];
      const abortCtl = new AbortController();

      cleanup.push(() => {
        try {
          abortCtl.abort();
        } catch {
          /* abort can throw if already aborted */
        }
        try {
          if (cloud) scene.remove(cloud);
          pointGeo?.dispose();
          pointMat?.dispose();
        } catch {
          /* race */
        }
        // Clear cloud stats so the pipeline panel doesn't carry stale numbers
        // into the next scene.
        try {
          setCloudStatsRef.current(null);
        } catch {
          /* race on unmount */
        }
        // Drop any objects registered after parse completes. Each entry
        // is responsible for disposing its own three.js geometry/material.
        for (const obj of ownedObjects) {
          try {
            obj.dispose();
          } catch {
            /* race */
          }
        }
        ownedObjects.length = 0;
        cancelAnimationFrame(raf);
        dom.removeEventListener("pointerdown", onDown);
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        dom.removeEventListener("wheel", onWheel);
        dom.removeEventListener("contextmenu", onContextMenu);
        dom.removeEventListener("dblclick", onDoubleClick);
        window.removeEventListener("keydown", onKey);
        window.removeEventListener("resize", onResize);
        dom.removeEventListener("pointerdown", markDirty);
        window.removeEventListener("pointermove", markDirty);
        dom.removeEventListener("wheel", markDirty);
        window.removeEventListener("keydown", markDirtyKey);
        window.removeEventListener("resize", markDirty);
        document.removeEventListener("visibilitychange", onVisibility);
        try {
          renderer.dispose();
          renderer.forceContextLoss();
          dom.remove();
        } catch {
          /* renderer disposal may race */
        }
      });
      const fetchStart = performance.now();
      pushDebug(
        { status: "fetching", error: undefined, errorStack: undefined },
        `fetching ${pointsUrl}`,
      );
      // Streaming progressive load. Bytes arrive in chunks; we parse complete
      // vertices and grow the Three.js geometry as they come in. Cloud
      // appears progressively instead of after a 30–60 s blocking download +
      // parse cycle. Two perf wins:
      //   1. UX: first points visible within a second of the header arriving.
      //   2. Memory: colors live in a Uint8Array (3 B/vertex, normalized=true
      //      on the BufferAttribute) instead of Float32 (12 B/vertex). Saves
      //      ~113 MB of JS heap at 12.6 M vertices.
      fetch(pointsUrl, { signal: abortCtl.signal })
        .then(async (res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
          if (!res.body) throw new Error("response has no body — streaming unsupported");
          const total = Number(res.headers.get("content-length") || 0);
          pushDebug(
            { fetchTotal: total, status: "fetching" },
            `fetch ok status=${res.status} content-length=${total}; streaming`,
          );

          const reader = res.body.getReader();
          let tail = new Uint8Array(0);
          let headerDone = false;
          let count = 0;
          let stride = 0;
          let layout: {
            x: number;
            y: number;
            z: number;
            r: number;
            g: number;
            b: number;
            conf: number | null;
          } | null = null;
          let positions: Float32Array | null = null;
          let colors: Uint8Array | null = null;
          let confidences: Float32Array | null = null;
          let pointsParsed = 0;
          let bytesRead = 0;
          let lastFlush = performance.now();
          const FLUSH_INTERVAL_MS = 200;
          // Live-tracked AABB in viewer (post-flip) coords. Used at end of
          // stream to set the depth colormap normalization range.
          let minX = Infinity, maxX = -Infinity;
          let minY = Infinity, maxY = -Infinity;
          let minZ = Infinity, maxZ = -Infinity;

          const flush = () => {
            if (!pointGeo) return;
            (pointGeo.attributes.position as BufferAttribute).needsUpdate = true;
            (pointGeo.attributes.color as BufferAttribute).needsUpdate = true;
            const ac = pointGeo.attributes.aConf as BufferAttribute | undefined;
            if (ac) ac.needsUpdate = true;
            pointGeo.setDrawRange(0, pointsParsed);
            dirty = true;
            const pct = total > 0 ? Math.round((bytesRead / total) * 100) : 0;
            pushDebug(
              { fetchBytes: bytesRead, sceneCount: pointsParsed },
              `streaming: ${pointsParsed.toLocaleString()}/${count.toLocaleString()} pts (${pct}%)`,
            );
          };

          while (true) {
            if (disposed) return;
            const { done, value } = await reader.read();
            if (done) break;
            bytesRead += value.length;

            // Append new chunk to tail.
            const next = new Uint8Array(tail.length + value.length);
            next.set(tail);
            next.set(value, tail.length);
            tail = next;

            // Phase 1: find end-of-header.
            if (!headerDone) {
              const headerEnd = findEndHeader(tail);
              if (headerEnd < 0) continue;
              const headerStr = new TextDecoder("ascii").decode(tail.subarray(0, headerEnd));
              const info = parsePlyHeader(headerStr);
              count = info.count;
              stride = info.stride;
              layout = info.layout;
              positions = new Float32Array(count * 3);
              colors = new Uint8Array(count * 3);
              confidences = info.layout.conf != null ? new Float32Array(count) : null;

              // Build geometry now (empty draw range) and add to scene so the
              // user sees the cloud start to fill rather than a blank canvas.
              pointGeo = new BufferGeometry();
              pointGeo.setAttribute("position", new BufferAttribute(positions, 3));
              // normalized=true → Uint8 [0,255] → vec3 [0,1] in shader for free.
              pointGeo.setAttribute("color", new BufferAttribute(colors, 3, true));
              if (confidences) {
                pointGeo.setAttribute("aConf", new BufferAttribute(confidences, 1));
              }
              pointGeo.setDrawRange(0, 0);
              // Custom shader material — equivalent to PointsMaterial in RGB
              // mode, but drives all three render modes (RGB / depth / conf)
              // through a single uMode uniform. Distance is computed in the
              // vertex shader from world position so we don't need an extra
              // per-point attribute. Confidence comes from the optional aConf
              // attribute (only present when points.ply has a `confidence`
              // property; we fall back to 1.0 otherwise).
              pointMat = new ShaderMaterial({
                uniforms: {
                  uMode: { value: 0 }, // 0=rgb 1=depth 2=confidence
                  uPointSize: { value: 0.0035 },
                  uOpacity: { value: 1.0 },
                  uMinDist: { value: 0.0 },
                  uMaxDist: { value: 1.0 },
                  uHasConf: { value: confidences ? 1 : 0 },
                },
                vertexShader: `
                  attribute vec3 color;
                  ${confidences ? "attribute float aConf;" : ""}
                  uniform float uPointSize;
                  varying vec3 vColor;
                  varying float vDist;
                  varying float vConf;
                  void main() {
                    vColor = color;
                    vDist = length(position);
                    ${confidences ? "vConf = aConf;" : "vConf = 1.0;"}
                    vec4 mv = modelViewMatrix * vec4(position, 1.0);
                    // Perspective point-size attenuation matching three.js
                    // PointsMaterial(sizeAttenuation: true).
                    gl_PointSize = uPointSize * (300.0 / -mv.z);
                    gl_Position = projectionMatrix * mv;
                  }
                `,
                fragmentShader: `
                  precision mediump float;
                  uniform int uMode;
                  uniform float uOpacity;
                  uniform float uMinDist;
                  uniform float uMaxDist;
                  uniform int uHasConf;
                  varying vec3 vColor;
                  varying float vDist;
                  varying float vConf;
                  // Turbo-like colormap (Mikhail Bessmeltsev's approximation,
                  // public domain). Cheap polynomial fit, no LUT needed.
                  vec3 turbo(float t) {
                    t = clamp(t, 0.0, 1.0);
                    vec3 a = vec3(0.13572138, 4.61539260, -42.66032258);
                    vec3 b = vec3(-152.94239396, 459.30493940, -700.74832297);
                    vec3 c = vec3(564.13539948, -198.21580373, -23.96587314);
                    float t2 = t * t;
                    float t3 = t2 * t;
                    return clamp(a + b * t + c * t2 + vec3(
                      53.97720359, -16.20987022, 12.02644739) * t3, 0.0, 1.0);
                  }
                  void main() {
                    vec3 rgb;
                    if (uMode == 1) {
                      float t = (vDist - uMinDist) / max(uMaxDist - uMinDist, 1e-3);
                      rgb = turbo(t);
                    } else if (uMode == 2) {
                      rgb = uHasConf == 1 ? turbo(vConf) : vec3(0.5);
                    } else {
                      rgb = vColor;
                    }
                    gl_FragColor = vec4(rgb, uOpacity);
                  }
                `,
                transparent: true,
                depthWrite: true,
                depthTest: true,
              });
              cloud = new Points(pointGeo, pointMat);
              cloud.frustumCulled = false; // bounds grow during stream; skip culling
              scene.add(cloud);
              shaderMatRef.current = pointMat;
              cloudRef.current = cloud;
              cleanup.push(() => {
                if (shaderMatRef.current === pointMat) shaderMatRef.current = null;
                if (cloudRef.current === cloud) cloudRef.current = null;
              });

              tail = tail.subarray(headerEnd);
              headerDone = true;
              // Broadcast the target cloud count immediately so the pipeline
              // panel matches the streaming overlay (otherwise it shows a
              // stale count from the manifest and disagrees with what's
              // actually being rendered).
              setCloudStatsRef.current({
                count,
                sizeMb: total / (1024 * 1024),
              });
              pushDebug(
                { sceneCount: 0, status: "parsing" },
                `header parsed: count=${count.toLocaleString()} stride=${stride}; streaming body`,
              );
            }

            // Phase 2: parse complete vertices in tail.
            // Coordinate frame: VGGT outputs OpenCV-convention world coords
            // (+Y down, +Z forward). Three.js wants +Y up, +Z toward camera.
            // Negate Y and Z (i.e. rotate 180° around X) on every position so
            // the cloud renders right-side-up. Annotation centroids/bboxes
            // get the matching flip in `web/app/hooks/useScene.ts` so they
            // stay co-registered with the cloud.
            if (positions && colors && layout) {
              const verts = Math.min(Math.floor(tail.length / stride), count - pointsParsed);
              if (verts > 0) {
                const view = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
                const lx = layout.x, ly = layout.y, lz = layout.z;
                const lr = layout.r, lg = layout.g, lb = layout.b;
                const lc = layout.conf;
                for (let v = 0; v < verts; v++) {
                  const base = v * stride;
                  const i = pointsParsed + v;
                  const px = view.getFloat32(base + lx, true);
                  const py = -view.getFloat32(base + ly, true);
                  const pz = -view.getFloat32(base + lz, true);
                  positions[i * 3] = px;
                  positions[i * 3 + 1] = py;
                  positions[i * 3 + 2] = pz;
                  colors[i * 3] = tail[base + lr];
                  colors[i * 3 + 1] = tail[base + lg];
                  colors[i * 3 + 2] = tail[base + lb];
                  if (confidences && lc != null) {
                    confidences[i] = view.getFloat32(base + lc, true);
                  }
                  if (px < minX) minX = px;
                  if (px > maxX) maxX = px;
                  if (py < minY) minY = py;
                  if (py > maxY) maxY = py;
                  if (pz < minZ) minZ = pz;
                  if (pz > maxZ) maxZ = pz;
                }
                pointsParsed += verts;
                tail = tail.subarray(verts * stride);
              }
            }

            // Throttle GPU re-uploads to avoid thrashing on every chunk.
            const now = performance.now();
            if (now - lastFlush >= FLUSH_INTERVAL_MS) {
              flush();
              lastFlush = now;
            }
          }

          // Final flush so the last sliver of points is visible.
          flush();
          const fetchMs = Math.round(performance.now() - fetchStart);
          pushDebug(
            {
              fetchBytes: bytesRead,
              fetchMs,
              sceneCount: pointsParsed,
              status: "started",
              startedMs: fetchMs,
            },
            `done: ${pointsParsed.toLocaleString()} points in ${fetchMs}ms`,
          );

          // Broadcast actual rendered stats so the pipeline panel can show
          // them instead of a stale manifest count.
          setCloudStatsRef.current({
            count: pointsParsed,
            sizeMb: bytesRead / (1024 * 1024),
          });

          // Build minimap from the now-complete cloud.
          if (positions && colors && pointsParsed > 0) {
            const 
            M = Math.min(8000, pointsParsed);
            const mstride = Math.max(1, Math.floor(pointsParsed / M));
            const xz = new Float32Array(Math.ceil(pointsParsed / mstride) * 2);
            const rgb = new Float32Array(Math.ceil(pointsParsed / mstride) * 3);
            let mi = 0;
            let minX = Infinity,
              maxX = -Infinity,
              minZ = Infinity,
              maxZ = -Infinity;
            for (let i = 0; i < pointsParsed; i += mstride) {
              const x = positions[i * 3];
              // positions[i*3+2] is already flipped (negated) at parse-time.
              // For top-down minimap, undo so the orientation matches the
              // user's mental "north = original +Z forward".
              const z = -positions[i * 3 + 2];
              xz[mi * 2] = x;
              xz[mi * 2 + 1] = z;
              rgb[mi * 3] = colors[i * 3] / 255;
              rgb[mi * 3 + 1] = colors[i * 3 + 1] / 255;
              rgb[mi * 3 + 2] = colors[i * 3 + 2] / 255;
              if (x < minX) minX = x;
              if (x > maxX) maxX = x;
              if (z < minZ) minZ = z;
              if (z > maxZ) maxZ = z;
              mi++;
            }
            setMiniPoints({
              xz: xz.subarray(0, mi * 2),
              rgb: rgb.subarray(0, mi * 3),
              bounds: { minX, maxX, minZ, maxZ },
            });
          }

          // Compute a real bounding sphere now that all points are loaded.
          // Without this, three.js's Points.raycast() early-outs against a
          // stale empty sphere computed during streaming and click-to-measure
          // returns zero hits even though the cloud is clearly visible.
          if (pointGeo) {
            pointGeo.computeBoundingSphere();
          }

          // Depth colormap normalization: use the cloud's AABB diagonal so
          // the gradient spans the actual captured volume cleanly.
          if (pointsParsed > 0 && Number.isFinite(minX)) {
            const extX = maxX - minX;
            const extY = maxY - minY;
            const extZ = maxZ - minZ;
            const diag = Math.sqrt(extX * extX + extY * extY + extZ * extZ);
            if (pointMat) {
              pointMat.uniforms.uMinDist.value = 0;
              pointMat.uniforms.uMaxDist.value = Math.max(diag, 0.01);
            }
            markDirty();
          }
        })
        .catch((err: unknown) => {
          const e = err as Error;
          // Don't surface AbortError — that's the cleanup path on unmount.
          if (e?.name === "AbortError" || disposed) return;
          pushDebug(
            {
              status: "error",
              error: e?.message ?? String(err),
              errorStack: e?.stack,
            },
            `FAILED: ${e?.message ?? err}`,
          );
          console.error("[PointCloudViewer] load failed:", pointsUrl, err);
        });
    } else {
      // Placeholder Three.js scene — friendly grid + ambient backdrop so the
      // annotation overlay still has spatial context to live in.
      const container = containerRef.current;
      const renderer = new WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setClearColor(0x0a0a0c, 1);
      container.appendChild(renderer.domElement);

      const scene = new Scene();
      const camera = new PerspectiveCamera(
        45,
        container.clientWidth / container.clientHeight,
        0.05,
        100,
      );
      camera.position.set(...initial.position);
      camera.lookAt(...initial.lookAt);

      // Subtle radial glow + grid + bbox wireframes for each annotation
      const grid = new GridHelper(8, 16, 0x3a3a46, 0x1a1a20);
      grid.position.y = 0;
      scene.add(grid);

      const ambient = new AmbientLight(0xffffff, 0.5);
      scene.add(ambient);
      const dir = new DirectionalLight(0x9b85ff, 0.8);
      dir.position.set(2, 3, 1);
      scene.add(dir);

      annotationsRef.current.forEach((a) => {
        const [lo, hi] = a.bbox;
        const size = new Vector3(
          hi[0] - lo[0],
          hi[1] - lo[1],
          hi[2] - lo[2],
        );
        const center = new Vector3(
          (lo[0] + hi[0]) / 2,
          (lo[1] + hi[1]) / 2,
          (lo[2] + hi[2]) / 2,
        );
        const geo = new BoxGeometry(size.x, size.y, size.z);
        const edges = new EdgesGeometry(geo);
        const mat = new LineBasicMaterial({
          color: new Color(a.color),
          transparent: true,
          opacity: 0.85,
        });
        const wire = new LineSegments(edges, mat);
        wire.position.copy(center);
        wire.userData.annotationId = a.id;
        scene.add(wire);
        geo.dispose();
      });

      // Light orbit controls — drag to rotate around lookAt.
      const target = new Vector3(...initial.lookAt);
      const sph = new Spherical();
      sph.setFromVector3(camera.position.clone().sub(target));
      let dragging = false;
      let lastX = 0;
      let lastY = 0;
      const onDown = (e: PointerEvent) => {
        dragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
      };
      const onMove = (e: PointerEvent) => {
        if (!dragging) return;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;
        sph.theta -= dx * 0.005;
        sph.phi = Math.max(0.05, Math.min(Math.PI - 0.05, sph.phi - dy * 0.005));
      };
      const onUp = () => {
        dragging = false;
      };
      const onWheel = (e: WheelEvent) => {
        sph.radius = Math.max(0.5, Math.min(20, sph.radius + e.deltaY * 0.002));
      };
      const dom = renderer.domElement;
      dom.addEventListener("pointerdown", onDown);
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      dom.addEventListener("wheel", onWheel, { passive: true });

      const onResize = () => {
        if (!container) return;
        const w = container.clientWidth;
        const h = container.clientHeight;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      };
      window.addEventListener("resize", onResize);

      const tick = () => {
        const offset = new Vector3().setFromSpherical(sph);
        camera.position.copy(target.clone().add(offset));
        camera.lookAt(target);

        const dirVec = new Vector3();
        camera.getWorldDirection(dirVec);
        setCameraRef.current(
          [camera.position.x, camera.position.y, camera.position.z],
          [dirVec.x, dirVec.y, dirVec.z],
        );

        renderer.render(scene, camera);
        raf = requestAnimationFrame(tick);
      };
      sceneRef.current.camera = camera;
      tick();

      cleanup.push(() => {
        cancelAnimationFrame(raf);
        dom.removeEventListener("pointerdown", onDown);
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        dom.removeEventListener("wheel", onWheel);
        window.removeEventListener("resize", onResize);
        try {
          renderer.dispose();
          renderer.forceContextLoss();
          dom.remove();
        } catch {
          /* renderer disposal may race */
        }
      });
    }

    return () => {
      disposed = true;
      cleanup.forEach((fn) => {
        try {
          fn();
        } catch {
          /* viewer.dispose() can throw if a load is still in flight */
        }
      });
      // Defensive: renderer.dispose() does not always remove DOM children
      // added during streaming. Clear anything left so React doesn't trip
      // over foreign nodes on re-render. Each removeChild is wrapped
      // because the cleanup path may race with an in-flight load.
      const c = containerRef.current;
      if (c) {
        while (c.firstChild) {
          try {
            c.removeChild(c.firstChild);
          } catch {
            break;
          }
        }
      }
    };
  }, [pointsUrl, emptyCloud]);

  return (
    <div className="relative h-full w-full">
      <div
        ref={containerRef}
        className="absolute inset-0 overflow-hidden bg-ink-950"
      />
      <div
        ref={overlayRef}
        className="pointer-events-none absolute inset-0 overflow-hidden"
      >
        {showAnnotations && (
          <AnnotationOverlay
            annotations={annotations}
            getCamera={() => sceneRef.current.camera}
            containerRef={overlayRef}
          />
        )}
      </div>
      <ViewerToolbar />
      <DebugHUD debug={debug} />
      <ControlsHint
        annotations={annotations}
        onSelect={(id) => useUI.getState().setSelected(id)}
        onZoomIn={() => apiRef.current?.zoom(0.7)}
        onZoomOut={() => apiRef.current?.zoom(1.4)}
        onReset={() => apiRef.current?.reset()}
      />
      {miniPoints && (
        <Minimap
          points={miniPoints}
          // Minimap reports clicks in PLY-frame z (matches the points it
          // draws); setTarget expects renderer-frame z (negated). Flip here
          // so click-to-pan moves the camera target to the spot the user
          // actually pointed at, instead of mirroring across the Z axis.
          onPan={(x, z) => apiRef.current?.setTarget([x, 0.1, -z])}
        />
      )}
    </div>
  );
}

/** Floating toolbar for view-mode toggles. Lives top-center so it's
 *  unmistakably the "demo controls" rather than a side widget. */
function ViewerToolbar() {
  const renderMode = useUI((s) => s.renderMode);
  const cycleRenderMode = useUI((s) => s.cycleRenderMode);
  const showAnnotations = useUI((s) => s.showAnnotations);
  const toggleAnnotations = useUI((s) => s.toggleAnnotations);
  const showFreespace = useUI((s) => s.showFreespace);
  const toggleFreespace = useUI((s) => s.toggleFreespace);
  const modeLabel =
    renderMode === "depth" ? "Depth" : renderMode === "confidence" ? "Confidence" : "RGB";

  return (
    <div className="pointer-events-auto absolute left-1/2 top-3 flex -translate-x-1/2 gap-1.5">
      <button
        onClick={cycleRenderMode}
        className="rounded-md border border-ink-700/70 bg-ink-900/85 px-2.5 py-1 font-mono text-[11px] text-ink-100 backdrop-blur hover:border-accent-400/60 hover:text-accent-200"
        title="Cycle point cloud coloring (RGB → Depth → Confidence)"
      >
        Mode: <span className="text-accent-300">{modeLabel}</span>
      </button>
      <button
        onClick={toggleAnnotations}
        className={`rounded-md border px-2.5 py-1 font-mono text-[11px] backdrop-blur ${
          showAnnotations
            ? "border-accent-400/80 bg-accent-500/15 text-accent-100"
            : "border-ink-700/70 bg-ink-900/85 text-ink-100 hover:border-accent-400/60 hover:text-accent-200"
        }`}
        title="Show/hide object markers and labels"
      >
        {showAnnotations ? "● Annotations" : "Annotations"}
      </button>
      <button
        onClick={toggleFreespace}
        className={`rounded-md border px-2.5 py-1 font-mono text-[11px] backdrop-blur ${
          showFreespace
            ? "border-accent-400/80 bg-accent-500/15 text-accent-100"
            : "border-ink-700/70 bg-ink-900/85 text-ink-100 hover:border-accent-400/60 hover:text-accent-200"
        }`}
        title="Show/hide the humanoid traversability grid (Stage 5)"
      >
        {showFreespace ? "● Free space" : "Free space"}
      </button>
    </div>
  );
}

function ControlsHint({
  annotations,
  onSelect,
  onZoomIn,
  onZoomOut,
  onReset,
}: {
  annotations: Annotation[];
  onSelect: (id: string) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      {/* Help panel + toggle stays bottom-left. */}
      <div className="pointer-events-auto absolute bottom-3 left-3 flex flex-col items-start gap-2">
        {open && (
          <div className="rounded-md border border-ink-700/60 bg-ink-900/85 px-3 py-2 text-[11px] text-ink-200 backdrop-blur">
            <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-ink-400">
              controls
            </div>
            <ul className="space-y-0.5">
              <li><kbd className="font-mono opacity-80">drag</kbd> orbit</li>
              <li><kbd className="font-mono opacity-80">dbl-click</kbd> recenter on point</li>
              <li><kbd className="font-mono opacity-80">shift+drag</kbd> / right-drag — pan</li>
              <li><kbd className="font-mono opacity-80">wheel</kbd> zoom</li>
              <li><kbd className="font-mono opacity-80">W A S D</kbd> move target</li>
              <li><kbd className="font-mono opacity-80">Q E</kbd> up/down</li>
              <li><kbd className="font-mono opacity-80">R</kbd> reset</li>
            </ul>
            {annotations.length > 0 && (
              <>
                <div className="mt-2 mb-1 font-mono text-[10px] uppercase tracking-wider text-ink-400">
                  fly to
                </div>
                <div className="flex flex-wrap gap-1">
                  {annotations.slice(0, 8).map((a) => (
                    <button
                      key={a.id}
                      onClick={() => onSelect(a.id)}
                      className="rounded border border-ink-700/70 bg-ink-800/80 px-1.5 py-0.5 text-[10px] hover:border-accent-400/60 hover:text-accent-300"
                      title={a.label}
                    >
                      {a.label.split(" ").slice(0, 2).join(" ")}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
        <button
          onClick={() => setOpen((v) => !v)}
          className="pointer-events-auto rounded-full border border-ink-700/70 bg-ink-900/80 px-2.5 py-1 font-mono text-[10px] text-ink-200 backdrop-blur hover:border-accent-400/60 hover:text-accent-300"
        >
          {open ? "× hide" : "? controls"}
        </button>
      </div>

      {/* Zoom + reset on the right edge, just below the minimap. */}
      <div className="pointer-events-auto absolute right-3 top-[210px] flex flex-col gap-1.5">
        <button
          onClick={onZoomIn}
          className="size-9 rounded-md border border-ink-700/70 bg-ink-900/80 font-mono text-base text-ink-200 backdrop-blur hover:border-accent-400/60 hover:text-accent-300"
          title="Zoom in"
          aria-label="Zoom in"
        >
          +
        </button>
        <button
          onClick={onZoomOut}
          className="size-9 rounded-md border border-ink-700/70 bg-ink-900/80 font-mono text-base text-ink-200 backdrop-blur hover:border-accent-400/60 hover:text-accent-300"
          title="Zoom out"
          aria-label="Zoom out"
        >
          −
        </button>
        <button
          onClick={onReset}
          className="size-9 rounded-md border border-ink-700/70 bg-ink-900/80 font-mono text-sm text-ink-200 backdrop-blur hover:border-accent-400/60 hover:text-accent-300"
          title="Reset view (R)"
          aria-label="Reset view"
        >
          ↺
        </button>
      </div>
    </>
  );
}

function Minimap({
  points,
  onPan,
}: {
  points: {
    xz: Float32Array;
    rgb: Float32Array;
    bounds: { minX: number; maxX: number; minZ: number; maxZ: number };
  };
  onPan: (x: number, z: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const camera = useUI((s) => s.camera);
  const SIZE = 180;
  const PAD = 12;

  const { minX, maxX, minZ, maxZ } = points.bounds;
  const spanX = Math.max(1e-3, maxX - minX);
  const spanZ = Math.max(1e-3, maxZ - minZ);
  const scale = (SIZE - 2 * PAD) / Math.max(spanX, spanZ);
  const cx = (minX + maxX) / 2;
  const cz = (minZ + maxZ) / 2;

  const worldToCanvas = (x: number, z: number): [number, number] => [
    SIZE / 2 + (x - cx) * scale,
    // Invert z so "north" is up in the minimap.
    SIZE / 2 - (z - cz) * scale,
  ];
  const canvasToWorld = (px: number, py: number): [number, number] => [
    (px - SIZE / 2) / scale + cx,
    -(py - SIZE / 2) / scale + cz,
  ];

  // Draw point cloud once (it's static after load).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = "#0a0a0c";
    ctx.fillRect(0, 0, SIZE, SIZE);
    const xz = points.xz;
    const rgb = points.rgb;
    const m = xz.length / 2;
    for (let i = 0; i < m; i++) {
      const [px, py] = worldToCanvas(xz[i * 2], xz[i * 2 + 1]);
      ctx.fillStyle = `rgb(${(rgb[i * 3] * 255) | 0},${(rgb[i * 3 + 1] * 255) | 0},${(rgb[i * 3 + 2] * 255) | 0})`;
      ctx.fillRect(px | 0, py | 0, 1, 1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points]);

  // Camera marker — drawn on top each render via a sibling canvas overlay.
  // The minimap stores points in PLY-frame z (re-flipped from the renderer's
  // negated-z when miniPoints was built), but `camera.position` from the
  // store is in renderer-frame z. Negate to bring the marker into the same
  // frame as the points; otherwise the marker shows up mirrored on Z.
  const [camPx, camPy] = worldToCanvas(camera.position[0], -camera.position[2]);
  // Direction was already PLY-frame: `-camera.direction[2]` flips renderer-z
  // → PLY-z so the arrow points the correct way relative to the (PLY-frame)
  // canvas without further adjustment.
  const dirAngle = Math.atan2(-camera.direction[2], camera.direction[0]);

  return (
    <div className="pointer-events-auto absolute right-3 top-3 select-none">
      <div className="relative" style={{ width: SIZE, height: SIZE }}>
        <canvas
          ref={canvasRef}
          width={SIZE}
          height={SIZE}
          className="rounded-lg border border-ink-700/70 bg-ink-950 shadow-lg"
          onClick={(e) => {
            const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
            const px = e.clientX - rect.left;
            const py = e.clientY - rect.top;
            const [wx, wz] = canvasToWorld(px, py);
            onPan(wx, wz);
          }}
          title="Click to pan camera target there"
        />
        {/* Camera marker */}
        <svg
          width={SIZE}
          height={SIZE}
          className="pointer-events-none absolute inset-0"
        >
          <g transform={`translate(${camPx}, ${camPy}) rotate(${(-dirAngle * 180) / Math.PI})`}>
            <polygon
              points="0,-7 5,5 0,2 -5,5"
              fill="#a78bfa"
              stroke="#fff"
              strokeWidth="1"
            />
          </g>
        </svg>
        <div className="absolute bottom-1 left-1 font-mono text-[9px] uppercase tracking-wider text-ink-500">
          world · top-down
        </div>
      </div>
    </div>
  );
}

function DebugHUD({ debug }: { debug: DebugState }) {
  // Production-quiet: the HUD is a logging surface for failures only. When
  // nothing's wrong, render nothing — no pill in the corner, no chrome.
  // On error, auto-expand so the user (and we) can see what broke.
  const errored = debug.status === "error" || !!debug.error;
  if (!errored) return null;

  const dot =
    debug.status === "started"
      ? "bg-emerald-400"
      : debug.status === "error"
        ? "bg-red-400"
        : debug.status === "idle"
          ? "bg-ink-500"
          : "bg-accent-400 animate-[pulse_900ms_ease-in-out_infinite]";
  return (
    <div className="pointer-events-auto absolute left-3 top-3 max-w-md rounded-md border border-ink-700/70 bg-ink-900/85 px-3 py-2 font-mono text-[10px] text-ink-200 backdrop-blur">
      <div className="mb-1 flex items-center gap-2">
        <span className={`size-2 rounded-full ${dot}`} />
        <span className="uppercase tracking-wider">{debug.status}</span>
        {debug.containerSize && (
          <span className="opacity-70">
            container {debug.containerSize[0]}×{debug.containerSize[1]}
          </span>
        )}
        <span className="opacity-70">webgl: {debug.webglOk ? "✓" : debug.webglOk === false ? "✗" : "?"}</span>
      </div>
      {debug.url && <div className="truncate opacity-70">url: {debug.url}</div>}
      <div className="opacity-70">
        fetch: {debug.fetchBytes != null ? `${debug.fetchBytes} B` : "—"}
        {debug.fetchMs != null ? ` (${debug.fetchMs}ms)` : ""}
        {debug.fetchTotal ? ` / ${debug.fetchTotal}` : ""}
      </div>
      <div className="opacity-70">
        parse: {debug.parseMs != null ? `${debug.parseMs}ms` : "—"}
        {" · "}sceneCount: {debug.sceneCount ?? "—"}
        {debug.startedMs != null ? ` · started: ${debug.startedMs}ms` : ""}
      </div>
      {debug.error && (
        <div className="mt-1 max-h-24 overflow-auto whitespace-pre-wrap rounded bg-red-500/10 px-2 py-1 text-red-200">
          ERROR: {debug.error}
          {debug.errorStack ? `\n${debug.errorStack.split("\n").slice(0, 4).join("\n")}` : ""}
        </div>
      )}
      <details className="mt-1">
        <summary className="cursor-pointer opacity-60">log ({debug.log.length})</summary>
        <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap text-[9px] leading-tight opacity-80">
          {debug.log.join("\n")}
        </pre>
      </details>
    </div>
  );
}

function initialView(annotations: Annotation[]): {
  position: [number, number, number];
  lookAt: [number, number, number];
} {
  if (annotations.length === 0) {
    return { position: [2, 1.6, 2], lookAt: [0, 0.8, 0] };
  }
  // Center on annotation centroid average, pull the camera back along +Z+Y.
  const c = annotations.reduce<[number, number, number]>(
    (acc, a) => [
      acc[0] + a.centroid[0],
      acc[1] + a.centroid[1],
      acc[2] + a.centroid[2],
    ],
    [0, 0, 0],
  );
  const n = annotations.length;
  const center: [number, number, number] = [c[0] / n, c[1] / n, c[2] / n];
  return {
    position: [center[0] + 1.5, center[1] + 0.6, center[2] + 1.5],
    lookAt: center,
  };
}