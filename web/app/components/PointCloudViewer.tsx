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

import { useEffect, useMemo, useRef, useState } from "react";
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
  Matrix3,
  Matrix4,
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
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Annotation, BBox, Vec3 } from "@/lib/types";
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
  /** Per-frame camera centers (cameras.json) in viewer frame — already
   *  through the OpenCV→three.js (x,-y,-z) flip. When present, the initial
   *  camera spawns at the median of these (≈ where the user stood) rather
   *  than at an arbitrary offset from the annotation centroid. */
  cameraCenters?: Vec3[];
  /** Set true when points.ply is empty/0-vertex; we'll show a placeholder. */
  emptyCloud?: boolean;
  /** URL to capture_map.json (Stage 4 output). When present, the viewer
   *  uses its `up_axis_world` / `u_axis_world` / `v_axis_world` /
   *  `floor_height_world` fields to rotate the cloud so gravity aligns with
   *  +Y and the floor sits at Y=0. Missing or 404 → no re-orientation. */
  captureMapUrl?: string;
}

/** Subset of capture_map.json fields the viewer reads. The backend writes
 *  these in OpenCV world coords (+Y down, +Z forward, first-camera origin). */
interface CaptureMapMeta {
  up_axis_world: [number, number, number];
  u_axis_world: [number, number, number];
  v_axis_world: [number, number, number];
  floor_height_world: number;
}

/** Build the 4×4 viewer-frame transform that levels the cloud:
 *  rotate so the Stage-4 gravity vector aligns with +Y, then translate so
 *  the estimated floor height lands at Y=0.
 *
 *  The basis vectors in `meta` are in OpenCV world coords; the cloud and
 *  annotations both receive the (x, -y, -z) flip when entering viewer
 *  space, so we apply the same flip to the basis vectors here. Dot
 *  products are preserved by that flip, so `floor_height_world` (a scalar)
 *  doesn't need re-projection. */
function buildLevelMatrix(meta: CaptureMapMeta): Matrix4 {
  const up = new Vector3(meta.up_axis_world[0], -meta.up_axis_world[1], -meta.up_axis_world[2]).normalize();
  const u = new Vector3(meta.u_axis_world[0], -meta.u_axis_world[1], -meta.u_axis_world[2]).normalize();
  // Re-orthonormalize u against up to absorb any float drift in the stored
  // basis, then derive v = u × up so (u, up, v) is a RIGHT-handed basis.
  // The backend's v_axis_world is computed as up × u (see capture_map.py)
  // which yields a left-handed triple — taking it verbatim makes the basis
  // matrix a reflection (det = -1), and the result is a mirrored scene
  // (left/right swap). Recomputing v with the correct sign avoids that.
  const uOrtho = u.clone().addScaledVector(up, -u.dot(up)).normalize();
  const vOrtho = new Vector3().crossVectors(uOrtho, up).normalize();
  // makeBasis sets columns. The matrix [u | up | v] maps floor-frame axes
  // into viewer coords; we want the inverse (viewer → floor) = transpose
  // for an orthonormal basis.
  const basis = new Matrix4().makeBasis(uOrtho, up, vOrtho);
  basis.invert();
  const translate = new Matrix4().makeTranslation(0, -meta.floor_height_world, 0);
  return new Matrix4().multiplyMatrices(translate, basis);
}

/** Refine a base level matrix by fitting a plane to the lowest slice of the
 *  parsed cloud and applying a small correction rotation so the plane
 *  normal aligns with +Y exactly.
 *
 *  Why: the backend's gravity estimate is the (trimmed) mean of per-frame
 *  camera-up vectors. On handheld captures it's typically within a few
 *  degrees of true gravity, but those few degrees are visible when you
 *  orbit the scene. The cloud itself contains a strong horizontal signal
 *  (the floor); fitting a plane to its bottom slice cancels the residual.
 *
 *  Returns the composed matrix, or the input unchanged if the refinement
 *  would have been too small to matter, too large to trust (often means
 *  the bottom slice isn't actually the floor — slanted scenes, mostly-air
 *  reconstructions), or numerically ill-conditioned. */
function refineLevelMatrix(positions: Float32Array, count: number, base: Matrix4): Matrix4 {
  const stride = Math.max(1, Math.floor(count / 12000));
  const tmp = new Vector3();
  // Pass 1: find the Y range so we can pick a "floor band" relative to it.
  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < count; i += stride) {
    tmp.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]).applyMatrix4(base);
    if (tmp.y < minY) minY = tmp.y;
    if (tmp.y > maxY) maxY = tmp.y;
  }
  const range = maxY - minY;
  if (range < 0.2) return base; // tiny vertical extent → not enough signal
  const yCutoff = minY + range * 0.15;
  // Pass 2: accumulate normal-equation moments for the floor candidates.
  let n = 0;
  let sxx = 0, sxz = 0, sx = 0, szz = 0, sz = 0, sxy = 0, szy = 0, sy = 0;
  for (let i = 0; i < count; i += stride) {
    tmp.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]).applyMatrix4(base);
    if (tmp.y > yCutoff) continue;
    const x = tmp.x, y = tmp.y, z = tmp.z;
    sxx += x * x; sxz += x * z; sx += x;
    szz += z * z; sz += z;
    sxy += x * y; szy += z * y; sy += y;
    n++;
  }
  if (n < 200) return base;
  // Solve A · [a; b; c] = rhs with A symmetric 3×3 and rhs = [sxy; szy; sy].
  // Coefficients (a, b) of  Y = a·X + b·Z + c  describe the floor's tilt.
  const A = new Matrix3().set(sxx, sxz, sx, sxz, szz, sz, sx, sz, n);
  if (Math.abs(A.determinant()) < 1e-6) return base;
  const Ainv = A.clone().invert();
  const e = Ainv.elements; // column-major
  const a = e[0] * sxy + e[3] * szy + e[6] * sy;
  const b = e[1] * sxy + e[4] * szy + e[7] * sy;
  // Floor plane normal in current leveled frame.
  const normal = new Vector3(-a, 1, -b).normalize();
  const dot = Math.max(-1, Math.min(1, normal.y));
  const angle = Math.acos(dot);
  if (angle < 0.003) return base; // ~0.17°: already aligned, skip
  if (angle > 0.18) return base; // >10°: suspicious — likely not the floor
  const axis = new Vector3().crossVectors(normal, new Vector3(0, 1, 0));
  if (axis.lengthSq() < 1e-10) return base;
  axis.normalize();
  const correction = new Matrix4().makeRotationAxis(axis, angle);
  return new Matrix4().multiplyMatrices(correction, base);
}

/** Apply a 4×4 matrix to a 3-vector annotation field. */
function applyMatrixToVec3(M: Matrix4, p: Vec3): Vec3 {
  const v = new Vector3(p[0], p[1], p[2]).applyMatrix4(M);
  return [v.x, v.y, v.z];
}

/** Apply a 4×4 matrix to an axis-aligned bbox. After rotation the original
 *  corners no longer form an AABB; re-derive a tight AABB by transforming
 *  all 8 corners. */
function applyMatrixToBBox(M: Matrix4, b: BBox): BBox {
  const [lo, hi] = b;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const v = new Vector3();
  for (let i = 0; i < 8; i++) {
    v.set(
      i & 1 ? hi[0] : lo[0],
      i & 2 ? hi[1] : lo[1],
      i & 4 ? hi[2] : lo[2],
    ).applyMatrix4(M);
    if (v.x < minX) minX = v.x;
    if (v.y < minY) minY = v.y;
    if (v.z < minZ) minZ = v.z;
    if (v.x > maxX) maxX = v.x;
    if (v.y > maxY) maxY = v.y;
    if (v.z > maxZ) maxZ = v.z;
  }
  return [[minX, minY, minZ], [maxX, maxY, maxZ]];
}

export function PointCloudViewer({ pointsUrl, annotations, cameraCenters, emptyCloud, captureMapUrl }: Props) {
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
  // The matrix that levels the scene (set once per scene after we fetch
  // capture_map.json; null when Stage 4 hasn't run yet). Drives both the
  // per-vertex transform inside the PLY parser AND the annotation overlay
  // through `transformedAnnotations` below. Wrapped in a wrapper object so
  // identity changes trigger the useMemo even though Matrix4 itself is
  // mutable — we never mutate after assigning.
  const [levelMatrix, setLevelMatrix] = useState<Matrix4 | null>(null);
  // Mirror into a ref so the heavy useEffect (which runs once per scene)
  // can also pick the matrix up after the async fetch resolves.
  const levelMatrixRef = useRef<Matrix4 | null>(null);
  useEffect(() => {
    levelMatrixRef.current = levelMatrix;
  }, [levelMatrix]);

  // Fetch capture_map.json in its own effect so the viewer doesn't
  // remount when Stage 4 finishes mid-session. When the matrix arrives, we
  // also apply it to the already-loaded cloud (if any) via cloudRef so the
  // re-orientation is immediate, no re-stream.
  useEffect(() => {
    if (!captureMapUrl) {
      setLevelMatrix(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(captureMapUrl);
        if (!res.ok) return;
        const meta = (await res.json()) as CaptureMapMeta;
        if (cancelled) return;
        if (
          !Array.isArray(meta.up_axis_world) ||
          !Array.isArray(meta.u_axis_world) ||
          !Array.isArray(meta.v_axis_world) ||
          typeof meta.floor_height_world !== "number"
        ) {
          return;
        }
        const M = buildLevelMatrix(meta);
        setLevelMatrix(M);
        const cloud = cloudRef.current;
        if (cloud) {
          cloud.matrix.copy(M);
          cloud.matrixAutoUpdate = false;
          markDirtyRef.current?.();
        }
        // Re-frame the camera onto the now-leveled scene. Defer one frame
        // so transformedAnnotations (the useMemo result) has time to flush
        // into annotationsRef.current — reset() reads from that ref to
        // compute the new initialView.
        requestAnimationFrame(() => apiRef.current?.reset());
      } catch {
        /* silent: missing capture map = render unleveled (no Stage 4) */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [captureMapUrl]);
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
  // Annotations in viewer space (after the OpenCV→three.js flip applied in
  // useScene.ts) further transformed by the Stage-4 level matrix when it's
  // available, so they stay co-registered with the leveled cloud.
  const transformedAnnotations = useMemo<Annotation[]>(() => {
    if (!levelMatrix) return annotations;
    return annotations.map((a) => ({
      ...a,
      centroid: applyMatrixToVec3(levelMatrix, a.centroid),
      bbox: applyMatrixToBBox(levelMatrix, a.bbox),
    }));
  }, [annotations, levelMatrix]);

  const annotationsRef = useRef<Annotation[]>(transformedAnnotations);
  useEffect(() => {
    annotationsRef.current = transformedAnnotations;
  }, [transformedAnnotations]);

  // Camera centers run through the same level matrix the cloud + annotations
  // see, so `initialView` (which consumes them) gets viewer-frame leveled
  // positions — otherwise the spawn would land under the floor on scenes
  // where the floor isn't already at y=0.
  const transformedCameraCenters = useMemo<Vec3[]>(() => {
    if (!cameraCenters || cameraCenters.length === 0) return [];
    if (!levelMatrix) return cameraCenters;
    return cameraCenters.map((c) => applyMatrixToVec3(levelMatrix, c));
  }, [cameraCenters, levelMatrix]);
  const cameraCentersRef = useRef<Vec3[]>(transformedCameraCenters);
  // Track whether we've ever seen non-empty centers. If they arrive *after*
  // the viewer has already mounted (typical — useQuery resolves async), we
  // want a single reset() so the spawn snaps from the legacy diagonal-offset
  // fallback to the proper in-room median. Subsequent updates (e.g. level
  // matrix arrival re-transforming them) don't re-fire — by then the user
  // may have panned, and yanking the camera back would be jarring.
  const cameraCentersResetFiredRef = useRef(false);
  useEffect(() => {
    cameraCentersRef.current = transformedCameraCenters;
    if (
      !cameraCentersResetFiredRef.current &&
      transformedCameraCenters.length > 0 &&
      apiRef.current
    ) {
      cameraCentersResetFiredRef.current = true;
      // Defer one frame so any concurrent annotation/level updates flush
      // into the refs that reset() reads.
      requestAnimationFrame(() => apiRef.current?.reset());
    }
  }, [transformedCameraCenters]);
  const setCameraRef = useRef(setCamera);
  useEffect(() => {
    setCameraRef.current = setCamera;
  }, [setCamera]);
  const setCloudStatsRef = useRef(setCloudStats);
  useEffect(() => {
    setCloudStatsRef.current = setCloudStats;
  }, [setCloudStats]);

  // Fly-to request: set externally (annotation click, preset button), the
  // render loop interpolates `target` (OrbitControls pivot) and `position`
  // (camera location) toward it each frame. Keeping both ends of the
  // viewing ray independent lets us tween view changes (top/front/side) as
  // well as pivot-only changes (annotation click) through one path.
  const flyToRef = useRef<{
    target: [number, number, number];
    position: [number, number, number];
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
  type PresetView = "top" | "front" | "side" | "reset";
  type ViewerApi = {
    zoom: (factor: number) => void;
    setTarget: (xyz: [number, number, number], radius?: number) => void;
    reset: () => void;
    preset: (view: PresetView) => void;
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


  // Tracks the previous selection so we can detect a "deselect" transition
  // and explicitly release the orbit target — otherwise OrbitControls keeps
  // pivoting around the last-clicked object even after the pill is dropped,
  // which reads as the camera being "locked" to it.
  const prevSelectedRef = useRef<string | null>(null);
  useEffect(() => {
    if (selectedId) {
      const a = annotationsRef.current.find((x) => x.id === selectedId);
      if (a) {
        const [lo, hi] = a.bbox;
        const ext = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
        // Pull camera in proportional to the object's extent — small things =
        // close-up, big things = farther back. apiRef.setTarget preserves the
        // current viewing direction at the requested distance.
        apiRef.current?.setTarget(a.centroid as [number, number, number], Math.max(0.06, ext * 0.85));
      }
    } else if (prevSelectedRef.current) {
      // Deselect: shift the orbit pivot back to the scene's bounding-sphere
      // center, keeping the same view angle + distance. Without this the
      // camera keeps spinning around the previously-selected object.
      const cloud = cloudRef.current;
      const sphere = cloud?.geometry?.boundingSphere ?? null;
      if (sphere) {
        const c = new Vector3().copy(sphere.center).applyMatrix4(cloud!.matrix);
        apiRef.current?.setTarget([c.x, c.y, c.z]);
      } else {
        apiRef.current?.reset();
      }
    }
    prevSelectedRef.current = selectedId;
  }, [selectedId]);

  useEffect(() => {
    if (!containerRef.current) return;
    let disposed = false;

    // Recomputed at mount and on every reset(), so when the level matrix
    // arrives mid-session and the annotations transform under it, the next
    // reset frames the leveled scene rather than the pre-level snapshot.
    const computeInitial = () =>
      initialView(annotationsRef.current, cameraCentersRef.current);
    let initial = computeInitial();

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

      // Three.js's built-in OrbitControls — battle-tested, supports touch &
      // pinch gestures out of the box, and adds two consumer-grade niceties
      // that the previous hand-rolled spherical code lacked:
      //   • enableDamping: drag/scroll inputs ease in/out instead of snapping
      //   • zoomToCursor:  wheel zoom moves toward the cursor, not the orbit
      //     pivot — feels right when inspecting a particular shelf or wall
      // Pan defaults to screen-space (translation along the camera plane),
      // which keeps everything intuitive after the Stage-4 levelling makes
      // "up" actually point up in world space.
      const dom = renderer.domElement;
      const controls = new OrbitControls(camera, dom);
      controls.target.set(...initial.lookAt);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.zoomToCursor = true;
      controls.screenSpacePanning = true;
      controls.minDistance = 0.3;
      controls.maxDistance = 40;
      // Block flipping below the floor: after levelling, +Y is gravity-up
      // and the floor sits at Y=0, so we cap the polar angle just shy of
      // the equator (looking horizontally out from the ground).
      controls.maxPolarAngle = Math.PI - 0.05;
      controls.update();

      // True when focus is in a text field — chat input, modal, etc. The
      // R-reset key would otherwise fire while the user typed in an input.
      const isTypingTarget = (e: KeyboardEvent): boolean => {
        const t = e.target as HTMLElement | null;
        if (!t) return false;
        if (t.isContentEditable) return true;
        const tag = t.tagName;
        return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
      };

      // Render-on-demand: redraw whenever OrbitControls reports a change
      // (drag/wheel/pinch). Idle GPU usage stays at zero.
      let dirty = true;
      const markDirty = () => {
        dirty = true;
      };
      markDirtyRef.current = markDirty;
      cleanup.push(() => {
        if (markDirtyRef.current === markDirty) markDirtyRef.current = null;
      });
      controls.addEventListener("change", markDirty);
      controls.addEventListener("start", markDirty);
      controls.addEventListener("end", markDirty);

      // Double-click to recenter the orbit target on the picked point. We
      // raycast against the streaming Points cloud and use the world-space
      // hit (three.js inverse-transforms the ray into local space when the
      // cloud's Object3D matrix is non-identity — important now that the
      // Stage-4 level transform lives on cloud.matrix).
      const raycaster = new Raycaster();
      const ndc = new Vector2();
      raycaster.params.Points = { threshold: 0.02 };
      // Deferred deselect — fires on a stationary single click that isn't
      // followed by a dblclick within DBLCLICK_GUARD_MS. Lets the user
      // escape the "camera locked to selected annotation" state by clicking
      // empty space.
      let deselectTimer: ReturnType<typeof setTimeout> | null = null;
      const clearDeselectTimer = () => {
        if (deselectTimer) {
          clearTimeout(deselectTimer);
          deselectTimer = null;
        }
      };
      const onDoubleClick = (e: MouseEvent) => {
        // Cancel any pending deselect — the user is recentering, not
        // clearing the selection.
        clearDeselectTimer();
        const cloud = cloudRef.current;
        if (!cloud) return;
        const rect = dom.getBoundingClientRect();
        ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(ndc, camera);
        const dist = camera.position.distanceTo(controls.target);
        raycaster.params.Points!.threshold = Math.max(0.005, dist * 0.01);
        const hits = raycaster.intersectObject(cloud, false);
        if (hits.length === 0) return;
        const p = hits[0].point;
        // Preserve the current camera offset relative to the target so the
        // viewing angle stays put — only the pivot moves.
        const offset = camera.position.clone().sub(controls.target);
        flyToRef.current = {
          target: [p.x, p.y, p.z],
          position: [p.x + offset.x, p.y + offset.y, p.z + offset.z],
        };
      };
      dom.addEventListener("dblclick", onDoubleClick);

      // Drag-vs-click discrimination: track pointer-down position so we only
      // deselect on stationary clicks (orbit drags must not deselect).
      const CLICK_MOVE_PX = 5;
      const CLICK_MAX_MS = 400;
      const DBLCLICK_GUARD_MS = 230;
      let downX = 0;
      let downY = 0;
      let downAt = 0;
      const onPointerDown = (e: PointerEvent) => {
        if (e.button !== 0) return;
        downX = e.clientX;
        downY = e.clientY;
        downAt = e.timeStamp;
      };
      const onPointerUp = (e: PointerEvent) => {
        if (e.button !== 0) return;
        const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
        const elapsed = e.timeStamp - downAt;
        if (moved > CLICK_MOVE_PX || elapsed > CLICK_MAX_MS) return;
        clearDeselectTimer();
        deselectTimer = setTimeout(() => {
          deselectTimer = null;
          if (useUI.getState().selectedId) {
            useUI.getState().setSelected(null);
          }
        }, DBLCLICK_GUARD_MS);
      };
      dom.addEventListener("pointerdown", onPointerDown);
      dom.addEventListener("pointerup", onPointerUp);

      const onKey = (e: KeyboardEvent) => {
        if (isTypingTarget(e)) return;
        if (e.key.toLowerCase() === "r") {
          apiRef.current?.preset("reset");
        }
      };
      window.addEventListener("keydown", onKey);

      // Pause render-on-demand when tab is hidden; redraw once when back.
      const onVisibility = () => {
        if (document.visibilityState === "visible") dirty = true;
      };
      document.addEventListener("visibilitychange", onVisibility);

      // Imperative API for outside UI (zoom buttons, minimap, presets).
      // All re-framing operations push a target+position pair onto
      // flyToRef so the tick loop interpolates smoothly to the destination
      // rather than snapping — snapping bypasses the rest of OrbitControls'
      // damping and feels jarring against the otherwise-smooth navigation.
      const computePresetView = (view: PresetView): { target: Vec3; position: Vec3 } => {
        if (view === "reset") {
          // Re-derive from current annotations so a reset that lands after
          // the level matrix arrived frames the leveled scene.
          initial = computeInitial();
          return { target: initial.lookAt, position: initial.position };
        }
        // Frame the scene by its bounding sphere (computed once parse
        // completes). Before that, fall back to the initial-view scale so
        // pre-load button mashes don't fire to (0, 0, 0).
        const cloud = cloudRef.current;
        const sphere = cloud?.geometry?.boundingSphere ?? null;
        const center = sphere
          ? new Vector3().copy(sphere.center).applyMatrix4(cloud!.matrix)
          : new Vector3(...initial.lookAt);
        const radius = sphere ? Math.max(sphere.radius, 0.5) : 2.5;
        // Position the camera just inside the bounding sphere so the user
        // ends up inside the room shell looking across, rather than way
        // outside seeing the back of the walls. zoomToCursor + scroll lets
        // them tighten or back off from there.
        const t: Vec3 = [center.x, center.y, center.z];
        if (view === "top") {
          // Top is well inside the bounding sphere so the user lands
          // already 'in the room' looking down, instead of hovering above
          // the ceiling shell. Tiny z offset avoids the gimbal
          // singularity directly overhead.
          const topY = center.y + radius * 0.385;
          return { target: t, position: [center.x, topY, center.z + 0.001] };
        }
        if (view === "front") {
          const d = radius * 0.55;
          return { target: t, position: [center.x, center.y + d * 0.2, center.z + d] };
        }
        // side — tightened a bit more than front so the user is just past
        // the wall surface looking across.
        const dSide = radius * 0.4;
        return { target: t, position: [center.x + dSide, center.y + dSide * 0.2, center.z] };
      };

      apiRef.current = {
        zoom: (factor) => {
          const off = camera.position.clone().sub(controls.target);
          const newLen = Math.max(
            controls.minDistance,
            Math.min(controls.maxDistance, off.length() * factor),
          );
          off.setLength(newLen);
          camera.position.copy(controls.target).add(off);
          controls.update();
          markDirty();
        },
        setTarget: (xyz, radius) => {
          // Keep the current view direction; rescale the camera offset to
          // the requested distance (or preserve current distance if none).
          const offset = camera.position.clone().sub(controls.target);
          if (typeof radius === "number" && offset.lengthSq() > 1e-9) {
            offset.setLength(Math.max(controls.minDistance, Math.min(controls.maxDistance, radius)));
          }
          flyToRef.current = {
            target: xyz,
            position: [xyz[0] + offset.x, xyz[1] + offset.y, xyz[2] + offset.z],
          };
        },
        reset: () => {
          initial = computeInitial();
          flyToRef.current = {
            target: initial.lookAt,
            position: initial.position,
          };
        },
        preset: (view) => {
          flyToRef.current = computePresetView(view);
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
        markDirty();
      };
      window.addEventListener("resize", onResize);

      // Render-on-demand at 30 fps cap. Each tick calls
      // OrbitControls.update() so damping continues to ease the camera
      // after the user releases input. Renders only when something has
      // actually changed (input, fly-to in flight, or external markDirty).
      const FRAME_INTERVAL_MS = 1000 / 30;
      const tmpDir = new Vector3();
      const tmpFlyTgt = new Vector3();
      const tmpFlyPos = new Vector3();
      let lastRender = 0;
      const tick = (now: number) => {
        if (document.visibilityState !== "visible") {
          raf = requestAnimationFrame(tick);
          return;
        }
        const fly = flyToRef.current;
        if (fly) {
          tmpFlyTgt.set(fly.target[0], fly.target[1], fly.target[2]);
          tmpFlyPos.set(fly.position[0], fly.position[1], fly.position[2]);
          controls.target.lerp(tmpFlyTgt, 0.18);
          camera.position.lerp(tmpFlyPos, 0.18);
          if (
            controls.target.distanceToSquared(tmpFlyTgt) < 1e-5 &&
            camera.position.distanceToSquared(tmpFlyPos) < 1e-5
          ) {
            flyToRef.current = null;
          }
          dirty = true;
        }
        // OrbitControls.update() returns true while damping is still
        // settling — flush the result into the render.
        if (controls.update()) dirty = true;

        if (dirty && now - lastRender >= FRAME_INTERVAL_MS) {
          camera.getWorldDirection(tmpDir);
          setCameraRef.current(
            [camera.position.x, camera.position.y, camera.position.z],
            [tmpDir.x, tmpDir.y, tmpDir.z],
          );
          renderer.render(scene, camera);
          lastRender = now;
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
        controls.dispose();
        dom.removeEventListener("dblclick", onDoubleClick);
        dom.removeEventListener("pointerdown", onPointerDown);
        dom.removeEventListener("pointerup", onPointerUp);
        clearDeselectTimer();
        window.removeEventListener("keydown", onKey);
        window.removeEventListener("resize", onResize);
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
              // If Stage-4 levelling data is already known by the time we
              // create the cloud (common: capture_map.json is 7 KB so it
              // resolves long before the multi-GB PLY does), apply the
              // rotation+translation to the Points object's transform. Stored
              // vertices stay in local frame; the renderer multiplies by
              // cloud.matrix at draw time, and raycasting transparently
              // inverse-transforms world rays into local space.
              {
                const lm = levelMatrixRef.current;
                if (lm) {
                  cloud.matrix.copy(lm);
                  cloud.matrixAutoUpdate = false;
                }
              }
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

          // Refine the level matrix against the actual cloud now that we
          // have all the points: fit a plane to the bottom 15% and apply
          // a small correction so the floor is exactly horizontal. The
          // Stage-4 estimate (per-frame camera-up average) is reliable to
          // within a few degrees; this step takes off that residual.
          if (positions && pointsParsed > 0) {
            const base = levelMatrixRef.current;
            if (base) {
              const refined = refineLevelMatrix(positions, pointsParsed, base);
              if (refined !== base) {
                // Update the ref synchronously so the minimap build below
                // uses the refined frame; React state update is for the
                // annotation overlay (and any other consumer of the memo).
                levelMatrixRef.current = refined;
                setLevelMatrix(refined);
                if (cloud) {
                  cloud.matrix.copy(refined);
                  cloud.matrixAutoUpdate = false;
                }
                // Re-frame the camera so the now-refined scene fills the
                // viewport. Deferred so transformedAnnotations has time to
                // flush before initialView re-reads annotationsRef.
                requestAnimationFrame(() => apiRef.current?.reset());
              }
            }
          }

          // Build minimap from the now-complete cloud. When the Stage-4
          // level matrix is available, sample points pass through it so the
          // top-down preview matches the leveled 3D view (otherwise the
          // minimap would show the tilted PLY frame while the viewer shows
          // it upright).
          if (positions && colors && pointsParsed > 0) {
            const targetCount = Math.min(8000, pointsParsed);
            const mstride = Math.max(1, Math.floor(pointsParsed / targetCount));
            const xz = new Float32Array(Math.ceil(pointsParsed / mstride) * 2);
            const rgb = new Float32Array(Math.ceil(pointsParsed / mstride) * 3);
            let mi = 0;
            let mmMinX = Infinity,
              mmMaxX = -Infinity,
              mmMinZ = Infinity,
              mmMaxZ = -Infinity;
            const lm = levelMatrixRef.current;
            const tmpP = new Vector3();
            for (let i = 0; i < pointsParsed; i += mstride) {
              tmpP.set(
                positions[i * 3],
                positions[i * 3 + 1],
                positions[i * 3 + 2],
              );
              if (lm) tmpP.applyMatrix4(lm);
              const x = tmpP.x;
              // Negate world Z so "north" reads as upward in the minimap,
              // matching the user's mental model of looking forward into
              // the captured space.
              const z = -tmpP.z;
              xz[mi * 2] = x;
              xz[mi * 2 + 1] = z;
              rgb[mi * 3] = colors[i * 3] / 255;
              rgb[mi * 3 + 1] = colors[i * 3 + 1] / 255;
              rgb[mi * 3 + 2] = colors[i * 3 + 2] / 255;
              if (x < mmMinX) mmMinX = x;
              if (x > mmMaxX) mmMaxX = x;
              if (z < mmMinZ) mmMinZ = z;
              if (z > mmMaxZ) mmMaxZ = z;
              mi++;
            }
            setMiniPoints({
              xz: xz.subarray(0, mi * 2),
              rgb: rgb.subarray(0, mi * 3),
              bounds: { minX: mmMinX, maxX: mmMaxX, minZ: mmMinZ, maxZ: mmMaxZ },
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
            annotations={transformedAnnotations}
            getCamera={() => sceneRef.current.camera}
            containerRef={overlayRef}
          />
        )}
      </div>
      <ViewerToolbar />
      <DebugHUD debug={debug} />
      <ControlsHint
        annotations={transformedAnnotations}
        onSelect={(id) => useUI.getState().setSelected(id)}
        onZoomIn={() => apiRef.current?.zoom(0.7)}
        onZoomOut={() => apiRef.current?.zoom(1.4)}
        onReset={() => apiRef.current?.reset()}
        onPreset={(view) => apiRef.current?.preset(view)}
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
  const showCaptureMap = useUI((s) => s.showCaptureMap);
  const toggleCaptureMap = useUI((s) => s.toggleCaptureMap);
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
        onClick={toggleCaptureMap}
        className={`rounded-md border px-2.5 py-1 font-mono text-[11px] backdrop-blur ${
          showCaptureMap
            ? "border-accent-400/80 bg-accent-500/15 text-accent-100"
            : "border-ink-700/70 bg-ink-900/85 text-ink-100 hover:border-accent-400/60 hover:text-accent-200"
        }`}
        title="Show/hide the top-down capture map (Stage 4)"
      >
        {showCaptureMap ? "● Capture map" : "Capture map"}
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
  onPreset,
}: {
  annotations: Annotation[];
  onSelect: (id: string) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  onPreset: (view: "top" | "front" | "side" | "reset") => void;
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
              <li><kbd className="font-mono opacity-80">wheel</kbd> zoom toward cursor</li>
              <li><kbd className="font-mono opacity-80">pinch</kbd> zoom (trackpad / touch)</li>
              <li><kbd className="font-mono opacity-80">R</kbd> reset view</li>
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

      {/* Right-edge column: zoom + reset (top), preset views (bottom). The
          presets give one-click access to the canonical orientations users
          normally have to compose by hand (top-down for floor planning,
          front/side for elevation views). */}
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
        <div className="mt-2 flex flex-col gap-1 rounded-md border border-ink-700/70 bg-ink-900/80 p-1.5 font-mono text-[10px] backdrop-blur">
          <button
            onClick={() => onPreset("top")}
            className="rounded px-2 py-1 text-ink-200 hover:bg-accent-500/15 hover:text-accent-200"
            title="Top-down floor plan view"
          >
            top
          </button>
          <button
            onClick={() => onPreset("front")}
            className="rounded px-2 py-1 text-ink-200 hover:bg-accent-500/15 hover:text-accent-200"
            title="Front elevation"
          >
            front
          </button>
          <button
            onClick={() => onPreset("side")}
            className="rounded px-2 py-1 text-ink-200 hover:bg-accent-500/15 hover:text-accent-200"
            title="Side elevation"
          >
            side
          </button>
        </div>
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
  // SVG `rotate(α)` (CW) maps the arrow's natural "up" vector to
  // (sin α, -cos α). The arrow points in renderer XZ space projected to
  // canvas: canvas-right = +x_renderer, canvas-up = -z_renderer (since
  // canvas-y = SIZE/2 - (z_ply - cz)·scale and z_ply = -z_renderer).
  // So we need (sin α, -cos α) = (dx, dz) → α = atan2(dx, -dz).
  const dirAngleDeg =
    (Math.atan2(camera.direction[0], -camera.direction[2]) * 180) / Math.PI;

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
          <g transform={`translate(${camPx}, ${camPy}) rotate(${dirAngleDeg})`}>
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

/** Per-axis median — robust to outlier frames at the start/end of capture
 *  where the user is fumbling the phone. Mean would get yanked by those. */
function medianVec3(pts: Vec3[]): Vec3 {
  const xs = pts.map((p) => p[0]).sort((a, b) => a - b);
  const ys = pts.map((p) => p[1]).sort((a, b) => a - b);
  const zs = pts.map((p) => p[2]).sort((a, b) => a - b);
  const mid = (xs.length - 1) / 2;
  const lo = Math.floor(mid);
  const hi = Math.ceil(mid);
  return [(xs[lo] + xs[hi]) / 2, (ys[lo] + ys[hi]) / 2, (zs[lo] + zs[hi]) / 2];
}

function initialView(
  annotations: Annotation[],
  cameraCenters: Vec3[] = [],
): {
  position: [number, number, number];
  lookAt: [number, number, number];
} {
  // Annotation-centroid average is what we look at — the densest cluster of
  // labelled stuff is the most natural "scene center."
  const annoCenter: Vec3 =
    annotations.length === 0
      ? [0, 0.8, 0]
      : (() => {
          const s = annotations.reduce<Vec3>(
            (acc, a) => [
              acc[0] + a.centroid[0],
              acc[1] + a.centroid[1],
              acc[2] + a.centroid[2],
            ],
            [0, 0, 0],
          );
          const n = annotations.length;
          return [s[0] / n, s[1] / n, s[2] / n];
        })();

  // Prefer the median capture position — that's literally "where the user
  // was standing in the room" — so the viewer spawns inside the scene at
  // eye level rather than offset along an arbitrary diagonal.
  if (cameraCenters.length > 0) {
    const pos = medianVec3(cameraCenters);
    return { position: pos, lookAt: annoCenter };
  }
  if (annotations.length === 0) {
    return { position: [2, 1.6, 2], lookAt: [0, 0.8, 0] };
  }
  // Fallback for old scenes with no cameras.json — keep the legacy offset.
  return {
    position: [annoCenter[0] + 1.5, annoCenter[1] + 0.6, annoCenter[2] + 1.5],
    lookAt: annoCenter,
  };
}