"use client";

/**
 * Splat viewer: renders the per-pixel VGGT cloud (`points.ply`, ~12 M raw
 * coloured points) directly via three.js `Points`. Matches the aesthetic of
 * VGGT's reference `demo_viser.py` — crisp opaque coloured pixels, no
 * blending, surface impression emerges from per-pixel density.
 *
 * Rationale: anisotropic Gaussian splatting via @sparkjsdev/spark forces a
 * 2 M-Gaussian budget (each Gaussian = 17 floats = 68 B in GPU memory), which
 * meant voxel-downsampling the raw VGGT cloud 10× and then over-blending the
 * survivors into surface goo. Points use 6 B each (3 float32 xyz + 3 uint8
 * rgb), so the full 12 M cloud fits in a 76 MB GPU buffer with no blending.
 *
 * The companion `splat.ply` (1.24 M anisotropic Gaussians, INRIA layout) is
 * still produced server-side but only consumed by the segmentation pipeline
 * (`segmentation/splat_io.py:load_centers`); the viewer ignores it.
 */

import { useEffect, useRef, useState } from "react";
// Named imports (instead of `import * as THREE`) so Turbopack tree-shakes
// three.js properly — bundler memory during /scenes/[id] compile drops
// dramatically (the namespace import was forcing every three module to be
// held in memory, ~600KB minified).
import {
  AmbientLight,
  Box3,
  Box3Helper,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  Camera,
  Color,
  DirectionalLight,
  EdgesGeometry,
  GridHelper,
  Group,
  Line,
  LineBasicMaterial,
  LineSegments,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Raycaster,
  Scene,
  ShaderMaterial,
  Spherical,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import type { Annotation, Vec3 } from "@/lib/types";
import { useUI } from "@/store/ui";
import { AnnotationOverlay } from "./AnnotationOverlay";

// "end_header\n" — used as a needle in the streaming parser to detect when the
// PLY ASCII header is fully received and we can transition to body parsing.
const END_HEADER_NEEDLE = new TextEncoder().encode("end_header\n");

/** Compute a rectilinear room footprint from the cloud's XZ projection.
 *
 *  Pipeline:
 *   1. Bin the cloud's XZ projection into a 5cm grid → per-cell point counts.
 *   2. **Smooth the counts** with a 3×3 box blur. This is the static-snapshot
 *      analog of a Kalman filter: each cell's "is-occupied" belief is updated
 *      from its neighbours' beliefs, suppressing isolated outlier counts the
 *      same way a Kalman update suppresses spurious observations. (A literal
 *      Kalman filter would need a temporal sequence; we have one frame.)
 *   3. Threshold to a binary occupancy mask.
 *   4. **Morphological closing** (dilate ×2, erode ×2) bridges hairline gaps
 *      between nearby wall fragments so the room reads as a single enclosed
 *      region even when inference noise breaks a wall.
 *   5. **Flood-fill from the grid border**. Anything not reached by the flood
 *      is interior — mark it occupied. This guarantees the ground plan has
 *      no holes (a real floor plan is always simply-connected).
 *   6. Walk the cell-by-cell boundary of the filled mask and emit
 *      axis-aligned line segments — every edge is purely along X or Z.
 *
 *  Floor/ceiling Y come from the 5th/95th percentile of the Y histogram so
 *  stragglers below the floor or above the ceiling don't stretch the box.
 */
function computeRoomFootprint(
  positions: Float32Array,
  pointCount: number,
  rawBounds: { min: [number, number, number]; max: [number, number, number] },
): {
  bounds: { min: [number, number, number]; max: [number, number, number] };
  floorY: number;
  ceilingY: number;
  /** Flat triplets [x0,y,z0, x1,y,z1, ...] describing line-segment pairs
   *  for the floor outline (every segment is purely along X or along Z). */
  floorOutline: Float32Array;
  ceilingOutline: Float32Array;
} {
  const CELL = 0.05; // 5cm grid — fine enough to hug furniture, coarse enough to dedupe noise
  const Y_BINS = 200;
  const [minX, minY, minZ] = rawBounds.min;
  const [maxX, maxY, maxZ] = rawBounds.max;
  const gridW = Math.max(1, Math.ceil((maxX - minX) / CELL) + 1);
  const gridD = Math.max(1, Math.ceil((maxZ - minZ) / CELL) + 1);
  const cells = gridW * gridD;
  const occ = new Uint32Array(cells);
  const yHist = new Uint32Array(Y_BINS);
  const yRange = Math.max(maxY - minY, 1e-3);

  for (let i = 0; i < pointCount; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    const cx = ((x - minX) / CELL) | 0;
    const cz = ((z - minZ) / CELL) | 0;
    if (cx >= 0 && cx < gridW && cz >= 0 && cz < gridD) {
      occ[cz * gridW + cx]++;
    }
    const yb = (((y - minY) / yRange) * Y_BINS) | 0;
    if (yb >= 0 && yb < Y_BINS) yHist[yb]++;
  }

  // Floor / ceiling = 5th / 95th percentile of Y.
  let cum = 0;
  let floorBin = 0;
  let ceilingBin = Y_BINS - 1;
  let foundFloor = false;
  for (let i = 0; i < Y_BINS; i++) {
    cum += yHist[i];
    if (!foundFloor && cum >= pointCount * 0.05) {
      floorBin = i;
      foundFloor = true;
    }
    if (cum >= pointCount * 0.95) {
      ceilingBin = i;
      break;
    }
  }
  const floorY = minY + (floorBin / Y_BINS) * yRange;
  const ceilingY = minY + ((ceilingBin + 1) / Y_BINS) * yRange;

  // Step 2 — smooth counts (3×3 box blur). Cheap proxy for a Gaussian
  // belief refinement: each cell's confidence in occupancy borrows from
  // its neighbours, suppressing isolated noise spikes.
  const blur = new Float32Array(cells);
  for (let cz = 0; cz < gridD; cz++) {
    for (let cx = 0; cx < gridW; cx++) {
      let s = 0;
      let n = 0;
      for (let dz = -1; dz <= 1; dz++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = cx + dx;
          const nz = cz + dz;
          if (nx < 0 || nx >= gridW || nz < 0 || nz >= gridD) continue;
          s += occ[nz * gridW + nx];
          n++;
        }
      }
      blur[cz * gridW + cx] = n > 0 ? s / n : 0;
    }
  }

  // Step 3 — threshold to binary. The threshold scales with average cell
  // density so it adapts to both dense and sparse clouds.
  const avg = pointCount / Math.max(1, cells);
  const occThreshold = Math.max(3, Math.floor(avg * 0.2));
  let mask: Uint8Array = new Uint8Array(cells);
  for (let i = 0; i < cells; i++) mask[i] = blur[i] > occThreshold ? 1 : 0;

  // Step 4 — morphological closing (dilate then erode). 2-cell dilate
  // bridges gaps up to ~10cm; the matching 2-cell erode keeps the wall
  // thickness honest. With closing, the flood-fill in step 5 reliably
  // sees the room interior as enclosed.
  const dilate = (src: Uint8Array): Uint8Array => {
    const dst = new Uint8Array(cells);
    for (let cz = 0; cz < gridD; cz++) {
      for (let cx = 0; cx < gridW; cx++) {
        const i = cz * gridW + cx;
        if (
          src[i] ||
          (cx > 0 && src[i - 1]) ||
          (cx < gridW - 1 && src[i + 1]) ||
          (cz > 0 && src[i - gridW]) ||
          (cz < gridD - 1 && src[i + gridW])
        ) {
          dst[i] = 1;
        }
      }
    }
    return dst;
  };
  const erode = (src: Uint8Array): Uint8Array => {
    const dst = new Uint8Array(cells);
    for (let cz = 0; cz < gridD; cz++) {
      for (let cx = 0; cx < gridW; cx++) {
        const i = cz * gridW + cx;
        if (!src[i]) continue;
        if (cx > 0 && !src[i - 1]) continue;
        if (cx < gridW - 1 && !src[i + 1]) continue;
        if (cz > 0 && !src[i - gridW]) continue;
        if (cz < gridD - 1 && !src[i + gridW]) continue;
        dst[i] = 1;
      }
    }
    return dst;
  };
  mask = dilate(dilate(mask));
  mask = erode(erode(mask));

  // Step 5 — flood-fill from the border. Any empty cell reachable from
  // outside the grid is "exterior". Cells that aren't exterior and aren't
  // already in the mask are interior holes — flip them to occupied.
  // Iterative DFS via a Uint32Array stack to avoid O(n) recursion.
  const exterior = new Uint8Array(cells);
  const stack = new Uint32Array(cells);
  let top = 0;
  const seed = (i: number) => {
    if (!mask[i] && !exterior[i]) {
      exterior[i] = 1;
      stack[top++] = i;
    }
  };
  for (let cx = 0; cx < gridW; cx++) {
    seed(cx);
    seed((gridD - 1) * gridW + cx);
  }
  for (let cz = 0; cz < gridD; cz++) {
    seed(cz * gridW);
    seed(cz * gridW + gridW - 1);
  }
  while (top > 0) {
    const i = stack[--top];
    const cx = i % gridW;
    if (cx > 0) seed(i - 1);
    if (cx < gridW - 1) seed(i + 1);
    if (i >= gridW) seed(i - gridW);
    if (i < cells - gridW) seed(i + gridW);
  }
  for (let i = 0; i < cells; i++) {
    if (!exterior[i]) mask[i] = 1;
  }

  // Tight bounds from the filled mask.
  let tightMinCx = gridW;
  let tightMaxCx = -1;
  let tightMinCz = gridD;
  let tightMaxCz = -1;
  for (let cz = 0; cz < gridD; cz++) {
    for (let cx = 0; cx < gridW; cx++) {
      if (mask[cz * gridW + cx]) {
        if (cx < tightMinCx) tightMinCx = cx;
        if (cx > tightMaxCx) tightMaxCx = cx;
        if (cz < tightMinCz) tightMinCz = cz;
        if (cz > tightMaxCz) tightMaxCz = cz;
      }
    }
  }
  const usable = tightMaxCx >= 0;
  const tightMinX = usable ? minX + tightMinCx * CELL : minX;
  const tightMaxX = usable ? minX + (tightMaxCx + 1) * CELL : maxX;
  const tightMinZ = usable ? minZ + tightMinCz * CELL : minZ;
  const tightMaxZ = usable ? minZ + (tightMaxCz + 1) * CELL : maxZ;

  // Step 6 — extract axis-aligned boundary segments from the filled mask.
  const isOcc = (cx: number, cz: number) =>
    cx >= 0 && cx < gridW && cz >= 0 && cz < gridD && mask[cz * gridW + cx] === 1;
  const segs: number[] = [];
  for (let cz = 0; cz < gridD; cz++) {
    for (let cx = 0; cx < gridW; cx++) {
      if (!isOcc(cx, cz)) continue;
      const x0 = minX + cx * CELL;
      const x1 = x0 + CELL;
      const z0 = minZ + cz * CELL;
      const z1 = z0 + CELL;
      if (!isOcc(cx, cz - 1)) segs.push(x0, z0, x1, z0);
      if (!isOcc(cx, cz + 1)) segs.push(x0, z1, x1, z1);
      if (!isOcc(cx - 1, cz)) segs.push(x0, z0, x0, z1);
      if (!isOcc(cx + 1, cz)) segs.push(x1, z0, x1, z1);
    }
  }

  const segCount = segs.length / 4;
  const floorOutline = new Float32Array(segCount * 6);
  const ceilingOutline = new Float32Array(segCount * 6);
  for (let s = 0; s < segCount; s++) {
    const e = s * 4;
    const ax = segs[e];
    const az = segs[e + 1];
    const bx = segs[e + 2];
    const bz = segs[e + 3];
    floorOutline[s * 6 + 0] = ax;
    floorOutline[s * 6 + 1] = floorY;
    floorOutline[s * 6 + 2] = az;
    floorOutline[s * 6 + 3] = bx;
    floorOutline[s * 6 + 4] = floorY;
    floorOutline[s * 6 + 5] = bz;
    ceilingOutline[s * 6 + 0] = ax;
    ceilingOutline[s * 6 + 1] = ceilingY;
    ceilingOutline[s * 6 + 2] = az;
    ceilingOutline[s * 6 + 3] = bx;
    ceilingOutline[s * 6 + 4] = ceilingY;
    ceilingOutline[s * 6 + 5] = bz;
  }

  return {
    bounds: {
      min: [tightMinX, floorY, tightMinZ],
      max: [tightMaxX, ceilingY, tightMaxZ],
    },
    floorY,
    ceilingY,
    floorOutline,
    ceilingOutline,
  };
}

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
 *  r/g/b (uchar). points.ply is the only PLY dialect we read here; the
 *  Gaussian splat.ply with f_dc_* SH coefficients is not consumed by the
 *  viewer anymore. */
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

/** Parse a binary-LE PLY (xyz float32 + rgb uchar layout, optional
 *  confidence) and return just the xyz positions as a Float32Array of
 *  length 3N. Used to load `wireframe.ply` produced by the backend
 *  segmentation stage. The full streaming parser (used for the much
 *  larger points.ply) is not reused here because the wireframe artifact
 *  is small (~10–20 k points) and a one-shot fetch + parse is simpler.
 *
 *  Mirrors the negate-Y/Z transform applied to points.ply so the
 *  wireframe lives in the same rendered coordinate frame as the cloud
 *  it replaces. */
function parseWireframePLY(buf: ArrayBuffer): Float32Array | null {
  const bytes = new Uint8Array(buf);
  const headerEnd = findEndHeader(bytes);
  if (headerEnd < 0) return null;
  const headerStr = new TextDecoder("ascii").decode(bytes.subarray(0, headerEnd));
  let info: ReturnType<typeof parsePlyHeader>;
  try {
    info = parsePlyHeader(headerStr);
  } catch {
    return null;
  }
  const { count, stride, layout } = info;
  const body = bytes.subarray(headerEnd);
  if (body.length < count * stride) return null;
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const base = i * stride;
    out[i * 3] = view.getFloat32(base + layout.x, true);
    out[i * 3 + 1] = -view.getFloat32(base + layout.y, true);
    out[i * 3 + 2] = -view.getFloat32(base + layout.z, true);
  }
  return out;
}

/** Build the wireframe-mode geometry.
 *
 *  Two source paths:
 *   - **Backend artifact** — when the segmentation pipeline emitted
 *     `wireframe.ply`, the caller passes its parsed positions as
 *     `precomputed`. We skip the client-side voxel + per-object sampling
 *     and go straight to kNN. This is the preferred path: the backend
 *     uses the real SAM 3.1 masks (via splat-Gaussian → points.ply
 *     spatial proximity) so per-object density traces actual segmented
 *     surfaces, not just a bounding-box approximation.
 *   - **Client fallback** — no artifact available. Voxel-downsample the
 *     live full cloud at 5 cm and bbox-sample each annotation for dense
 *     per-object detail.
 *
 *  Both paths converge on a kNN graph (k=5) over the resulting point
 *  set, built with a uniform grid hash sized to MAX_EDGE_LEN so
 *  disconnected objects don't bridge across empty space.
 *
 *  Adds two children to `group`: a `Points` (monochrome accent dots) and
 *  a `LineSegments` (faint accent edges). Materials are owned by the
 *  group's children and disposed via the existing `disposeChildren` path.
 */
function buildWireframeGeometry(
  group: Group,
  cloud: Points,
  annotations: Annotation[],
  precomputed?: Float32Array,
): void {
  const VOXEL = 0.05; // 5 cm
  const PER_OBJECT_SAMPLE = 500;
  const K = 5;
  const MAX_EDGE_LEN = 0.30;
  const ACCENT_POINT = 0xfcd9b8; // sand
  const ACCENT_EDGE = 0xffb347; // dusk apricot

  // ── Step A: assemble the wireframe point set (`wirePos`) ───────────────
  let wirePos: Float32Array;
  if (precomputed && precomputed.length >= 3) {
    // Backend artifact already encodes voxel + per-object density.
    wirePos = precomputed;
  } else {
    wirePos = sampleWireframeFromCloud(cloud, annotations, VOXEL, PER_OBJECT_SAMPLE);
  }
  const N = (wirePos.length / 3) | 0;
  if (N === 0) return;

  buildAndAttachWireframe(group, wirePos, N, K, MAX_EDGE_LEN, ACCENT_POINT, ACCENT_EDGE);
}

/** Client-side fallback sampler: voxel-downsample the live cloud +
 *  bbox-sample each annotation. */
function sampleWireframeFromCloud(
  cloud: Points,
  annotations: Annotation[],
  VOXEL: number,
  PER_OBJECT_SAMPLE: number,
): Float32Array {
  const positions = cloud.geometry.attributes.position as BufferAttribute;
  const total = positions.count;
  if (total === 0) return new Float32Array(0);

  // Step 1: voxel downsample. Pack the integer cell key into a single
  // number using bit-shifts so we can use a Map<number, number> instead
  // of a string-keyed Set (10× faster on V8 for the typical scene size).
  // Each axis is offset to be non-negative before packing; 11 bits per
  // axis covers ±51.2 m which is well beyond any single capture.
  const SHIFT_BITS = 11;
  const HALF = 1 << (SHIFT_BITS - 1);
  const MASK = (1 << SHIFT_BITS) - 1;
  const voxelKeep = new Map<number, number>();
  for (let i = 0; i < total; i++) {
    const x = positions.getX(i);
    const y = positions.getY(i);
    const z = positions.getZ(i);
    const cx = Math.floor(x / VOXEL) + HALF;
    const cy = Math.floor(y / VOXEL) + HALF;
    const cz = Math.floor(z / VOXEL) + HALF;
    if (cx < 0 || cy < 0 || cz < 0 || cx > MASK || cy > MASK || cz > MASK) continue;
    const key = (cx << (SHIFT_BITS * 2)) | (cy << SHIFT_BITS) | cz;
    if (!voxelKeep.has(key)) voxelKeep.set(key, i);
  }
  const voxelIndices = Array.from(voxelKeep.values());

  // Step 2: per-annotation dense sampling. The PLY parser flips Y/Z when
  // uploading positions to the GPU, but annotation bboxes are still in
  // PLY frame, so we apply the same flip to the bbox before testing.
  const objectIndices: number[] = [];
  for (const a of annotations) {
    const [lo, hi] = a.bbox;
    const minX = lo[0];
    const maxX = hi[0];
    // After the parser's negate-Y/Z, the rendered "min Y" is -hi[1] and
    // the rendered "max Y" is -lo[1]; same for Z.
    const minY = -hi[1];
    const maxY = -lo[1];
    const minZ = -hi[2];
    const maxZ = -lo[2];
    const inBox: number[] = [];
    for (let i = 0; i < total; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      const z = positions.getZ(i);
      if (x < minX || x > maxX) continue;
      if (y < minY || y > maxY) continue;
      if (z < minZ || z > maxZ) continue;
      inBox.push(i);
    }
    // Random sample without replacement (Fisher–Yates partial shuffle).
    if (inBox.length <= PER_OBJECT_SAMPLE) {
      for (const idx of inBox) objectIndices.push(idx);
    } else {
      for (let s = 0; s < PER_OBJECT_SAMPLE; s++) {
        const r = s + Math.floor(Math.random() * (inBox.length - s));
        const tmp = inBox[s];
        inBox[s] = inBox[r];
        inBox[r] = tmp;
        objectIndices.push(inBox[s]);
      }
    }
  }

  // Combine voxel + object indices, then materialize a flat Float32Array.
  const allIndices = voxelIndices.concat(objectIndices);
  const N = allIndices.length;
  const wirePos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const src = allIndices[i];
    wirePos[i * 3] = positions.getX(src);
    wirePos[i * 3 + 1] = positions.getY(src);
    wirePos[i * 3 + 2] = positions.getZ(src);
  }
  return wirePos;
}

/** Build a kNN graph over `wirePos` and attach Points + LineSegments to
 *  the group. Uniform grid hash sized to MAX_EDGE_LEN so neighbour search
 *  is O(N·k); long edges (across empty space) are excluded by the same
 *  cutoff. */
function buildAndAttachWireframe(
  group: Group,
  wirePos: Float32Array,
  N: number,
  K: number,
  MAX_EDGE_LEN: number,
  ACCENT_POINT: number,
  ACCENT_EDGE: number,
): void {
  const SHIFT_BITS = 11;
  const HALF = 1 << (SHIFT_BITS - 1);

  const grid = new Map<number, number[]>();
  const cellOf = (x: number, y: number, z: number): number => {
    const cx = Math.floor(x / MAX_EDGE_LEN) + HALF;
    const cy = Math.floor(y / MAX_EDGE_LEN) + HALF;
    const cz = Math.floor(z / MAX_EDGE_LEN) + HALF;
    return (cx << (SHIFT_BITS * 2)) | (cy << SHIFT_BITS) | cz;
  };
  for (let i = 0; i < N; i++) {
    const c = cellOf(wirePos[i * 3], wirePos[i * 3 + 1], wirePos[i * 3 + 2]);
    let bucket = grid.get(c);
    if (!bucket) {
      bucket = [];
      grid.set(c, bucket);
    }
    bucket.push(i);
  }
  const edgeIndex: number[] = [];
  const seenEdge = new Set<number>();
  const maxEdgeSq = MAX_EDGE_LEN * MAX_EDGE_LEN;
  // Pre-allocated per-iteration neighbour scratch — k closest.
  const bestDist = new Float64Array(K);
  const bestIdx = new Int32Array(K);
  for (let i = 0; i < N; i++) {
    const x = wirePos[i * 3];
    const y = wirePos[i * 3 + 1];
    const z = wirePos[i * 3 + 2];
    for (let k = 0; k < K; k++) {
      bestDist[k] = Infinity;
      bestIdx[k] = -1;
    }
    const ix = Math.floor(x / MAX_EDGE_LEN);
    const iy = Math.floor(y / MAX_EDGE_LEN);
    const iz = Math.floor(z / MAX_EDGE_LEN);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const cx = ix + dx + HALF;
          const cy = iy + dy + HALF;
          const cz = iz + dz + HALF;
          const c = (cx << (SHIFT_BITS * 2)) | (cy << SHIFT_BITS) | cz;
          const bucket = grid.get(c);
          if (!bucket) continue;
          for (const j of bucket) {
            if (j === i) continue;
            const ex = wirePos[j * 3] - x;
            const ey = wirePos[j * 3 + 1] - y;
            const ez = wirePos[j * 3 + 2] - z;
            const d2 = ex * ex + ey * ey + ez * ez;
            if (d2 > maxEdgeSq) continue;
            // Insert into k-best.
            for (let k = 0; k < K; k++) {
              if (d2 < bestDist[k]) {
                for (let m = K - 1; m > k; m--) {
                  bestDist[m] = bestDist[m - 1];
                  bestIdx[m] = bestIdx[m - 1];
                }
                bestDist[k] = d2;
                bestIdx[k] = j;
                break;
              }
            }
          }
        }
      }
    }
    for (let k = 0; k < K; k++) {
      const j = bestIdx[k];
      if (j < 0) continue;
      // Dedup undirected edges.
      const a = i < j ? i : j;
      const b = i < j ? j : i;
      const eKey = a * N + b;
      if (seenEdge.has(eKey)) continue;
      seenEdge.add(eKey);
      edgeIndex.push(a, b);
    }
  }

  // Build the three.js objects. Points: small monochrome dots. Edges:
  // one LineSegments mesh sharing the same position buffer (indexed).
  const wireGeo = new BufferGeometry();
  wireGeo.setAttribute("position", new BufferAttribute(wirePos, 3));

  const pointMat = new PointsMaterial({
    color: new Color(ACCENT_POINT),
    size: 0.012,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.7,
    depthWrite: false,
  });
  const points = new Points(wireGeo, pointMat);
  group.add(points);

  // Edges share the same position buffer to avoid duplicating ~N×12 bytes;
  // a separate index attribute defines the line topology.
  const edgeGeo = new BufferGeometry();
  edgeGeo.setAttribute("position", new BufferAttribute(wirePos, 3));
  edgeGeo.setIndex(new BufferAttribute(new Uint32Array(edgeIndex), 1));
  const lineMat = new LineBasicMaterial({
    color: new Color(ACCENT_EDGE),
    transparent: true,
    opacity: 0.55,
  });
  const lines = new LineSegments(edgeGeo, lineMat);
  group.add(lines);
}

interface Props {
  splatUrl: string;
  annotations: Annotation[];
  /** Set true when the splat file is empty/0-vertex; we'll show a placeholder. */
  emptySplat?: boolean;
  /** Optional pre-baked wireframe.ply URL. When undefined the viewer skips the
   *  artifact fetch entirely and uses the client-side voxel sampler. Gated by
   *  the manifest so we don't 404 on scenes that never produced one. */
  wireframeUrl?: string;
}

export function SplatViewer({ splatUrl, annotations, emptySplat, wireframeUrl }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<{
    cancel: () => void;
    camera: Camera | null;
  }>({ cancel: () => undefined, camera: null });
  const setCamera = useUI((s) => s.setCamera);
  const setCloudStats = useUI((s) => s.setCloudStats);
  const setBounds = useUI((s) => s.setBounds);
  const selectedId = useUI((s) => s.selectedId);
  const renderMode = useUI((s) => s.renderMode);
  const showAnnotations = useUI((s) => s.showAnnotations);
  const schematicMode = useUI((s) => s.schematicMode);
  const wireframeMode = useUI((s) => s.wireframeMode);
  const measureMode = useUI((s) => s.measureMode);
  const measurements = useUI((s) => s.measurements);
  const pendingPoint = useUI((s) => s.pendingPoint);
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
    if (line) console.log(`[SplatViewer] ${line}`, patch);
  };

  // Refs so the heavy effect can read the latest annotations + setCamera
  // WITHOUT taking them as dependencies. annotations is a fresh array on
  // every parent render (page does `annotations.data ?? []`) — if we put it
  // in deps the splat viewer remounts on every 2s manifest poll, each
  // in-flight addSplatScene rejects with "Scene disposed", and the dispose
  // path stomps on DOM nodes mid-load. Critical fix.
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
  // toggles, new measurements) and we don't want to remount on those.
  const shaderMatRef = useRef<ShaderMaterial | null>(null);
  const schematicGroupRef = useRef<Group | null>(null);
  const measurementGroupRef = useRef<Group | null>(null);
  const cloudRef = useRef<Points | null>(null);
  const markDirtyRef = useRef<(() => void) | null>(null);
  // Wireframe mode owns its own three.js group (lazy-built on first
  // activation), a DOM overlay layer for billboarded labels, and a
  // per-frame label-update hook the render loop calls so labels follow
  // the camera as it orbits.
  const wireframeGroupRef = useRef<Group | null>(null);
  const wireframeBuiltRef = useRef(false);
  const wireframeLabelLayerRef = useRef<HTMLDivElement | null>(null);
  const wireframeLabelDivsRef = useRef<HTMLDivElement[]>([]);
  const wireframeUpdateLabelsRef = useRef<(() => void) | null>(null);
  // Sibling click handler reads the latest store values via these refs so it
  // doesn't need to re-bind on every state change.
  const measureModeRef = useRef(measureMode);
  const pendingPointRef = useRef(pendingPoint);
  const renderModeRef = useRef(renderMode);
  const schematicModeRef = useRef(schematicMode);
  const wireframeModeRef = useRef(wireframeMode);
  useEffect(() => { measureModeRef.current = measureMode; }, [measureMode]);
  useEffect(() => { pendingPointRef.current = pendingPoint; }, [pendingPoint]);
  useEffect(() => { renderModeRef.current = renderMode; }, [renderMode]);
  useEffect(() => { schematicModeRef.current = schematicMode; }, [schematicMode]);
  useEffect(() => { wireframeModeRef.current = wireframeMode; }, [wireframeMode]);

  // Bounds (set by parser when stream completes) used to size + label the
  // always-on AABB wireframe and the schematic helpers.
  const boundsRef = useRef<{ min: Vec3; max: Vec3 } | null>(null);

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

  // Schematic mode → cloud opacity + populate ground grid + cross sections.
  // Re-runs when bounds become known so the helpers can size to the cloud.
  const bounds = useUI((s) => s.bounds);
  useEffect(() => {
    const mat = shaderMatRef.current;
    if (mat) {
      mat.uniforms.uOpacity.value = schematicMode ? 0.25 : 1.0;
      mat.transparent = schematicMode;
    }
    const group = schematicGroupRef.current;
    if (!group) return;
    // Wipe any previous helpers.
    while (group.children.length) {
      const c = group.children[0];
      group.remove(c);
      const o = c as unknown as {
        geometry?: { dispose: () => void };
        material?: { dispose: () => void };
      };
      o.geometry?.dispose?.();
      o.material?.dispose?.();
    }
    if (!schematicMode || !bounds) {
      group.visible = false;
      markDirtyRef.current?.();
      return;
    }
    group.visible = true;
    const [minX, minY, minZ] = bounds.min;
    const [maxX, maxY, maxZ] = bounds.max;
    const extX = maxX - minX;
    const extY = maxY - minY;
    const extZ = maxZ - minZ;
    const footprint = Math.max(extX, extZ);

    // Ground grid: anchored at minY (post-flip floor), sized to cloud.
    const grid = new GridHelper(
      footprint * 1.4,
      Math.max(8, Math.round(footprint * 8)),
      0x6f5cff,
      0x2a2740,
    );
    grid.position.set((minX + maxX) / 2, minY, (minZ + maxZ) / 2);
    (grid.material as LineBasicMaterial).transparent = true;
    (grid.material as LineBasicMaterial).opacity = 0.4;
    group.add(grid);

    // Three horizontal cross-section frames at 25 / 50 / 75% height. Each
    // frame is the AABB top-face outline rendered at that y, hinting at
    // how the captured volume reads at different elevations.
    const sliceColor = new Color(0xffd166);
    const slices = [0.25, 0.5, 0.75];
    for (const t of slices) {
      const y = minY + extY * t;
      const verts = new Float32Array([
        minX, y, minZ,  maxX, y, minZ,
        maxX, y, minZ,  maxX, y, maxZ,
        maxX, y, maxZ,  minX, y, maxZ,
        minX, y, maxZ,  minX, y, minZ,
      ]);
      const geo = new BufferGeometry();
      geo.setAttribute("position", new BufferAttribute(verts, 3));
      const lineMat = new LineBasicMaterial({
        color: sliceColor,
        transparent: true,
        opacity: 0.55,
      });
      group.add(new LineSegments(geo, lineMat));
    }
    markDirtyRef.current?.();
  }, [schematicMode, bounds]);

  // Wireframe mode → reduced point cloud + kNN edges + DOM labels.
  //
  // Lifecycle: lazy-built on first activation from the streaming PLY's
  // position buffer (via `cloudRef.current.geometry`). The full cloud's
  // GPU buffers stay resident for instant flip-back; we just toggle
  // `cloud.visible`. Subsequent toggles never rebuild the geometry.
  //
  // The label overlay is a sibling DOM layer to `containerRef`; the
  // render loop calls `wireframeUpdateLabelsRef.current()` each frame to
  // project annotation centroids → CSS pixels.
  useEffect(() => {
    const group = wireframeGroupRef.current;
    const cloud = cloudRef.current;
    if (!group) return;

    // Always-up-to-date cloud reveal. If wireframe just turned off, the
    // toggle below restores `cloud.visible = true` even if the geometry
    // was never built (e.g., user toggled before the PLY finished).
    const setCloudVisible = (visible: boolean) => {
      if (cloud) cloud.visible = visible;
    };

    if (!wireframeMode) {
      group.visible = false;
      setCloudVisible(true);
      // Hide label overlay + clear the per-frame updater so the render
      // loop stops touching DOM nodes that may have been unmounted.
      const layer = wireframeLabelLayerRef.current;
      if (layer) layer.style.display = "none";
      wireframeUpdateLabelsRef.current = null;
      markDirtyRef.current?.();
      return;
    }

    // Lazy build on first activation. Requires the PLY to have finished
    // streaming so positions are populated; if the user toggled too
    // early we just hide the cloud and wait — bail-out leaves group empty.
    if (!cloud) {
      // Nothing to build from yet — the streaming load will populate
      // cloudRef.current later; the user can toggle off+on to retry.
      return;
    }

    if (!wireframeBuiltRef.current) {
      // Build is async because we first try to fetch the backend
      // artifact (`wireframe.ply`); fall back to the client-side voxel
      // sampler if it's missing or fails to parse. We mark `built`
      // synchronously to avoid re-entry on rapid toggles.
      wireframeBuiltRef.current = true;
      // Only hit the network when the manifest advertises wireframe.ply —
      // otherwise the 404 noise drowns the console and the voxel sampler is
      // already a fine fallback.
      const tryArtifact = wireframeUrl
        ? fetch(wireframeUrl)
            .then((r) => (r.ok ? r.arrayBuffer() : null))
            .then((b) => (b ? parseWireframePLY(b) : null))
            .catch(() => null)
        : Promise.resolve(null);
      tryArtifact.then((precomputed) => {
        // If the user toggled off before the fetch finished, leave the
        // group empty — they can flip back on to retry.
        if (!wireframeGroupRef.current) return;
        buildWireframeGeometry(
          group,
          cloud,
          annotationsRef.current,
          precomputed ?? undefined,
        );
        markDirtyRef.current?.();
      });
    }

    group.visible = true;
    setCloudVisible(false);

    // Build the DOM label layer the first time the user activates.
    // Mounted as a sibling to containerRef so it sits on top of the
    // canvas without intercepting pointer events.
    const container = containerRef.current;
    if (container && !wireframeLabelLayerRef.current) {
      const layer = document.createElement("div");
      layer.className = "pointer-events-none absolute inset-0 overflow-hidden";
      layer.style.position = "absolute";
      layer.style.inset = "0";
      layer.style.pointerEvents = "none";
      container.appendChild(layer);
      wireframeLabelLayerRef.current = layer;
    }

    // (Re)populate label divs to match the current annotation list.
    const layer = wireframeLabelLayerRef.current;
    if (layer) {
      layer.style.display = "block";
      // Tear down old divs — annotations may have changed between toggles.
      for (const d of wireframeLabelDivsRef.current) d.remove();
      wireframeLabelDivsRef.current = [];
      for (const a of annotationsRef.current) {
        const d = document.createElement("div");
        d.textContent = a.label;
        d.style.position = "absolute";
        d.style.transform = "translate(-50%, -100%)";
        d.style.padding = "2px 6px";
        d.style.fontSize = "10px";
        d.style.fontFamily = "ui-monospace, monospace";
        d.style.color = "#fcd9b8";
        d.style.background = "rgba(20, 16, 28, 0.78)";
        d.style.border = "1px solid rgba(255, 179, 71, 0.45)";
        d.style.borderRadius = "3px";
        d.style.whiteSpace = "nowrap";
        d.style.opacity = "0";
        layer.appendChild(d);
        wireframeLabelDivsRef.current.push(d);
      }
    }

    // Per-frame projection hook the render loop calls. Reads the live
    // camera + container size every tick so labels stay glued to their
    // 3D anchor as the user orbits.
    wireframeUpdateLabelsRef.current = () => {
      const layer = wireframeLabelLayerRef.current;
      const cam = sceneRef.current.camera;
      if (!layer || !cam) return;
      const w = layer.clientWidth;
      const h = layer.clientHeight;
      const tmp = new Vector3();
      const anns = annotationsRef.current;
      const divs = wireframeLabelDivsRef.current;
      for (let i = 0; i < divs.length && i < anns.length; i++) {
        const c = anns[i].centroid;
        // Annotation centroids are recorded in PLY frame; the parser
        // negates Y/Z when uploading point positions (see line ~1314).
        // Apply the same flip here so labels anchor to the rendered cloud.
        tmp.set(c[0], -c[1], -c[2]);
        tmp.project(cam);
        // Behind camera (z >= 1) → hide.
        if (tmp.z >= 1 || tmp.z <= -1) {
          divs[i].style.opacity = "0";
          continue;
        }
        const px = (tmp.x * 0.5 + 0.5) * w;
        const py = (-tmp.y * 0.5 + 0.5) * h;
        divs[i].style.left = `${px}px`;
        divs[i].style.top = `${py - 6}px`;
        divs[i].style.opacity = "1";
      }
    };

    markDirtyRef.current?.();
  }, [wireframeMode, annotations, wireframeUrl]);

  // Tear down wireframe DOM on unmount.
  useEffect(() => {
    return () => {
      for (const d of wireframeLabelDivsRef.current) d.remove();
      wireframeLabelDivsRef.current = [];
      const layer = wireframeLabelLayerRef.current;
      if (layer) {
        layer.remove();
        wireframeLabelLayerRef.current = null;
      }
      wireframeUpdateLabelsRef.current = null;
    };
  }, []);

  // Measurements → 3D line segments. (HTML labels for distances are rendered
  // by <MeasurementOverlay/>; this effect just maintains the in-scene lines
  // and pending-point marker.)
  useEffect(() => {
    const group = measurementGroupRef.current;
    if (!group) return;
    while (group.children.length) {
      const c = group.children[0];
      group.remove(c);
      const o = c as unknown as {
        geometry?: { dispose: () => void };
        material?: { dispose: () => void };
      };
      o.geometry?.dispose?.();
      o.material?.dispose?.();
    }
    const lineColor = new Color(0xffe066);
    for (const m of measurements) {
      const verts = new Float32Array([
        m.a[0], m.a[1], m.a[2],
        m.b[0], m.b[1], m.b[2],
      ]);
      const geo = new BufferGeometry();
      geo.setAttribute("position", new BufferAttribute(verts, 3));
      const mat = new LineBasicMaterial({ color: lineColor });
      group.add(new Line(geo, mat));
    }
    markDirtyRef.current?.();
  }, [measurements]);

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
      // close-up, big things = farther back.
      radius: Math.max(0.18, ext * 2.2),
    };
  }, [selectedId]);

  useEffect(() => {
    if (!containerRef.current) return;
    let disposed = false;

    // Snapshot annotations once for initial camera framing. Live updates after
    // mount go through AnnotationOverlay (it re-renders on prop change).
    const initial = initialView(annotationsRef.current);

    const useViewer = !emptySplat;
    let raf = 0;

    const cleanup: (() => void)[] = [];

    const c0 = containerRef.current;
    pushDebug(
      {
        status: "idle",
        url: splatUrl,
        containerSize: [c0.clientWidth, c0.clientHeight],
      },
      `mount: useViewer=${useViewer} url=${splatUrl} container=${c0.clientWidth}x${c0.clientHeight}`,
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

      // Schematic + measurement groups exist from the start (empty), get
      // populated lazily by sibling effects once bounds are known and the
      // user toggles modes / places measurements.
      const schematicGroup = new Group();
      schematicGroup.visible = false;
      scene.add(schematicGroup);
      schematicGroupRef.current = schematicGroup;
      const measurementGroup = new Group();
      scene.add(measurementGroup);
      measurementGroupRef.current = measurementGroup;
      const wireframeGroup = new Group();
      wireframeGroup.visible = false;
      scene.add(wireframeGroup);
      wireframeGroupRef.current = wireframeGroup;
      cleanup.push(() => {
        if (schematicGroupRef.current === schematicGroup) {
          schematicGroupRef.current = null;
        }
        if (measurementGroupRef.current === measurementGroup) {
          measurementGroupRef.current = null;
        }
        if (wireframeGroupRef.current === wireframeGroup) {
          wireframeGroupRef.current = null;
        }
        wireframeBuiltRef.current = false;
        scene.remove(schematicGroup);
        scene.remove(measurementGroup);
        scene.remove(wireframeGroup);
        // Children may own geometries/materials — dispose those.
        const disposeChildren = (g: Group) => {
          g.traverse((obj) => {
            const o = obj as unknown as {
              geometry?: { dispose: () => void };
              material?: { dispose: () => void };
            };
            o.geometry?.dispose?.();
            o.material?.dispose?.();
          });
        };
        disposeChildren(schematicGroup);
        disposeChildren(measurementGroup);
        disposeChildren(wireframeGroup);
      });

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

      // Click-to-measure raycasting. Distinguishes click from drag by
      // tracking down/up pointer position; only fires when measureMode is
      // active so it doesn't interfere with orbit/pan.
      const raycaster = new Raycaster();
      // THREE.Points raycasting picks the first point whose center is within
      // `threshold` world units of the click ray. Our cloud is at metre
      // scale, so 5cm reliably picks any visible cluster while keeping the
      // nearest hit sensible.
      raycaster.params.Points = { threshold: 0.05 };
      let clickStart: { x: number; y: number; t: number } | null = null;
      const onClickDown = (e: PointerEvent) => {
        if (e.button !== 0 || e.shiftKey || e.metaKey || e.ctrlKey) return;
        clickStart = { x: e.clientX, y: e.clientY, t: performance.now() };
      };
      const onClickUp = (e: PointerEvent) => {
        const start = clickStart;
        clickStart = null;
        if (!start) return;
        if (!measureModeRef.current) return;
        if (Math.abs(e.clientX - start.x) + Math.abs(e.clientY - start.y) > 4) return;
        if (performance.now() - start.t > 600) return;
        const c = cloudRef.current;
        if (!c) return;
        const rect = dom.getBoundingClientRect();
        const ndc = new Vector2(
          ((e.clientX - rect.left) / rect.width) * 2 - 1,
          -((e.clientY - rect.top) / rect.height) * 2 + 1,
        );
        raycaster.setFromCamera(ndc, camera);
        const hits = raycaster.intersectObject(c, false);
        if (hits.length === 0) return;
        // IMPORTANT: hits[0].point is the projection of the matched point
        // onto the click ray, which is offset from the real surface point
        // by up to `threshold` (5cm). For an accurate measurement we need
        // the matched point's actual world position from the geometry.
        const hit = hits[0];
        const posAttr = c.geometry.attributes.position as BufferAttribute;
        const idx = hit.index ?? 0;
        const xyz: Vec3 = [
          posAttr.getX(idx),
          posAttr.getY(idx),
          posAttr.getZ(idx),
        ];
        const ui = useUI.getState();
        if (ui.pendingPoint) ui.finishMeasurement(xyz);
        else ui.beginMeasurement(xyz);
        markDirty();
      };
      dom.addEventListener("pointerdown", onClickDown);
      window.addEventListener("pointerup", onClickUp);
      cleanup.push(() => {
        dom.removeEventListener("pointerdown", onClickDown);
        window.removeEventListener("pointerup", onClickUp);
      });

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
          // Wireframe-mode floating labels: project annotation centroids
          // each frame so DOM tags follow the camera as the user orbits.
          // No-op when the mode is off (the hook is null).
          wireframeUpdateLabelsRef.current?.();
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
      // Always-on AABB wireframe + per-axis HTML dimension labels. Built
      // once when the stream completes; not present until then.
      let bboxHelper: Box3Helper | null = null;
      // Group of three.js objects added to the scene that need disposing.
      // Populated as features (AABB, schematic, measurements) come online.
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
        // Drop the AABB / dimension labels / schematic helpers / measurement
        // lines registered after parse completes. Each entry is responsible
        // for disposing its own three.js geometry/material.
        for (const obj of ownedObjects) {
          try {
            obj.dispose();
          } catch {
            /* race */
          }
        }
        ownedObjects.length = 0;
        boundsRef.current = null;
        try {
          setBounds(null);
        } catch {
          /* race */
        }
        cancelAnimationFrame(raf);
        dom.removeEventListener("pointerdown", onDown);
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        dom.removeEventListener("wheel", onWheel);
        dom.removeEventListener("contextmenu", onContextMenu);
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
        `fetching ${splatUrl}`,
      );
      // Streaming progressive load. Bytes arrive in chunks; we parse complete
      // vertices and grow the Three.js geometry as they come in. Cloud
      // appears progressively instead of after a 30–60 s blocking download +
      // parse cycle. Two perf wins:
      //   1. UX: first points visible within a second of the header arriving.
      //   2. Memory: colors live in a Uint8Array (3 B/vertex, normalized=true
      //      on the BufferAttribute) instead of Float32 (12 B/vertex). Saves
      //      ~113 MB of JS heap at 12.6 M vertices.
      fetch(splatUrl, { signal: abortCtl.signal })
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
          // Live-tracked AABB in viewer (post-flip) coords. Promoted to
          // boundsRef + setBounds(...) once the stream completes so the
          // overlay/AABB always reflects the final volume rather than
          // chasing the partial cloud during streaming.
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
              // panel matches the streaming overlay (otherwise it shows the
              // splat.ply Gaussian count from the manifest — a different
              // file — and disagrees with what's actually being rendered).
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
          // them instead of (the much smaller) splat.ply Gaussian count.
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

          // Build the room footprint: a 2D occupancy grid over the floor
          // projection, walked to produce axis-aligned boundary segments.
          // Replaces the loose AABB with: (a) a tight room-shaped polygon at
          // floor + ceiling height, and (b) a tighter dimension box derived
          // from the occupied cells so the W/H/D labels reflect the actual
          // captured volume rather than the worst-case point spread.
          if (pointsParsed > 0 && Number.isFinite(minX) && positions) {
            const room = computeRoomFootprint(
              positions,
              pointsParsed,
              { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
            );
            const [tMinX, tMinY, tMinZ] = room.bounds.min;
            const [tMaxX, tMaxY, tMaxZ] = room.bounds.max;
            const extX = tMaxX - tMinX;
            const extY = tMaxY - tMinY;
            const extZ = tMaxZ - tMinZ;

            // Dimension box: render at the tight room bounds.
            const pad = Math.max(extX, extY, extZ) * 0.005;
            const box = new Box3(
              new Vector3(tMinX - pad, tMinY - pad, tMinZ - pad),
              new Vector3(tMaxX + pad, tMaxY + pad, tMaxZ + pad),
            );
            const helperColor = new Color(0x9b85ff);
            bboxHelper = new Box3Helper(box, helperColor);
            const lineMat = (bboxHelper as unknown as { material?: LineBasicMaterial })
              .material;
            if (lineMat) {
              lineMat.transparent = true;
              lineMat.opacity = 0.45;
            }
            scene.add(bboxHelper);
            ownedObjects.push({
              dispose: () => {
                if (bboxHelper) scene.remove(bboxHelper);
              },
            });

            // Footprint polygon: floor outline + ceiling outline. Both are
            // axis-aligned line segments hugging the actual room walls (and
            // any furniture clusters that sit on the floor band). Always
            // visible so even in RGB mode the room shape is unmistakable.
            const outlineGroup = new Group();
            const mkOutline = (verts: Float32Array, color: number, opacity: number) => {
              const geo = new BufferGeometry();
              geo.setAttribute("position", new BufferAttribute(verts, 3));
              const mat = new LineBasicMaterial({
                color: new Color(color),
                transparent: true,
                opacity,
              });
              return new LineSegments(geo, mat);
            };
            outlineGroup.add(mkOutline(room.floorOutline, 0x9b85ff, 0.85));
            outlineGroup.add(mkOutline(room.ceilingOutline, 0x9b85ff, 0.45));
            scene.add(outlineGroup);
            ownedObjects.push({
              dispose: () => {
                outlineGroup.traverse((obj) => {
                  const o = obj as unknown as {
                    geometry?: { dispose: () => void };
                    material?: { dispose: () => void };
                  };
                  o.geometry?.dispose?.();
                  o.material?.dispose?.();
                });
                scene.remove(outlineGroup);
              },
            });

            // Depth colormap normalization: use the tight room diagonal so
            // the gradient spans the actual captured volume cleanly.
            const diag = Math.sqrt(extX * extX + extY * extY + extZ * extZ);
            if (pointMat) {
              pointMat.uniforms.uMinDist.value = 0;
              pointMat.uniforms.uMaxDist.value = Math.max(diag, 0.01);
            }

            const finalBounds = {
              min: [tMinX - pad, tMinY - pad, tMinZ - pad] as Vec3,
              max: [tMaxX + pad, tMaxY + pad, tMaxZ + pad] as Vec3,
            };
            boundsRef.current = finalBounds;
            setBounds(finalBounds);
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
          console.error("[SplatViewer] load failed:", splatUrl, err);
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
      // Defensive: viewer.dispose() and renderer.dispose() do not always
      // remove DOM children added by the splat library. Clear anything left
      // so React doesn't trip over foreign nodes on re-render. Each
      // removeChild is wrapped because gaussian-splats-3d may have already
      // detached some nodes inside its own dispose path.
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
  }, [splatUrl, emptySplat]);

  return (
    <div
      className="relative h-full w-full"
      style={{ cursor: measureMode ? "crosshair" : undefined }}
    >
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
        <DimensionOverlay
          getCamera={() => sceneRef.current.camera}
          containerRef={overlayRef}
        />
        <MeasurementOverlay
          getCamera={() => sceneRef.current.camera}
          containerRef={overlayRef}
        />
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

/** Format a metric distance for on-screen labels: cm under 1 m, m above. */
function fmtDist(m: number): string {
  if (!Number.isFinite(m)) return "—";
  if (m < 1) return `${(m * 100).toFixed(1)} cm`;
  return `${m.toFixed(2)} m`;
}

/** Project a world-space point into the overlay container. Returns
 *  null when the point is behind the camera. */
function project3D(
  worldX: number,
  worldY: number,
  worldZ: number,
  camera: Camera,
  width: number,
  height: number,
): { x: number; y: number } | null {
  const v = new Vector3(worldX, worldY, worldZ).project(camera);
  // v.z > 1 means behind near plane / off-screen depth; skip.
  if (!Number.isFinite(v.x) || v.z > 1.5) return null;
  return {
    x: (v.x * 0.5 + 0.5) * width,
    y: (-v.y * 0.5 + 0.5) * height,
  };
}

/** W/H/D dimension labels pinned to the AABB edge midpoints. Always on
 *  once bounds are known — instantly communicates real-world scale. */
function DimensionOverlay({
  getCamera,
  containerRef,
}: {
  getCamera: () => Camera | null;
  containerRef: React.RefObject<HTMLDivElement | null>;
}) {
  const bounds = useUI((s) => s.bounds);
  const displayScale = useUI((s) => s.displayScale);
  // Subscribe to camera so we re-render on every dirty tick.
  useUI((s) => s.camera);
  const cam = getCamera();
  const container = containerRef.current;
  if (!bounds || !cam || !container) return null;
  const w = container.clientWidth;
  const h = container.clientHeight;
  const [minX, minY, minZ] = bounds.min;
  const [maxX, maxY, maxZ] = bounds.max;
  const extX = (maxX - minX) * displayScale;
  const extY = (maxY - minY) * displayScale;
  const extZ = (maxZ - minZ) * displayScale;
  // Midpoints of three orthogonal AABB edges. Pick edges that face the
  // camera-facing front face so labels don't end up behind geometry.
  const wMid = project3D((minX + maxX) / 2, minY, minZ, cam, w, h);
  const hMid = project3D(minX, (minY + maxY) / 2, minZ, cam, w, h);
  const dMid = project3D(minX, minY, (minZ + maxZ) / 2, cam, w, h);

  const labelStyle = (
    p: { x: number; y: number } | null,
  ): React.CSSProperties => ({
    position: "absolute",
    left: p ? `${p.x}px` : "0",
    top: p ? `${p.y}px` : "0",
    transform: "translate(-50%, -50%)",
    visibility: p ? "visible" : "hidden",
  });

  return (
    <>
      <div
        className="rounded border border-accent-400/60 bg-ink-950/85 px-1.5 py-0.5 font-mono text-[10px] text-accent-200 shadow-md backdrop-blur"
        style={labelStyle(wMid)}
      >
        W {fmtDist(extX)}
      </div>
      <div
        className="rounded border border-accent-400/60 bg-ink-950/85 px-1.5 py-0.5 font-mono text-[10px] text-accent-200 shadow-md backdrop-blur"
        style={labelStyle(hMid)}
      >
        H {fmtDist(extY)}
      </div>
      <div
        className="rounded border border-accent-400/60 bg-ink-950/85 px-1.5 py-0.5 font-mono text-[10px] text-accent-200 shadow-md backdrop-blur"
        style={labelStyle(dMid)}
      >
        D {fmtDist(extZ)}
      </div>
    </>
  );
}

/** HTML label for each placed measurement (drawn at the midpoint of the
 *  segment) plus a small marker on the pending first point. */
function MeasurementOverlay({
  getCamera,
  containerRef,
}: {
  getCamera: () => Camera | null;
  containerRef: React.RefObject<HTMLDivElement | null>;
}) {
  const measurements = useUI((s) => s.measurements);
  const pendingPoint = useUI((s) => s.pendingPoint);
  const displayScale = useUI((s) => s.displayScale);
  useUI((s) => s.camera);
  const cam = getCamera();
  const container = containerRef.current;
  if (!cam || !container) return null;
  const w = container.clientWidth;
  const h = container.clientHeight;
  return (
    <>
      {measurements.map((m) => {
        const mid = project3D(
          (m.a[0] + m.b[0]) / 2,
          (m.a[1] + m.b[1]) / 2,
          (m.a[2] + m.b[2]) / 2,
          cam,
          w,
          h,
        );
        if (!mid) return null;
        return (
          <div
            key={m.id}
            className="rounded border border-yellow-300/70 bg-ink-950/90 px-1.5 py-0.5 font-mono text-[11px] text-yellow-200 shadow-md"
            style={{
              position: "absolute",
              left: `${mid.x}px`,
              top: `${mid.y}px`,
              transform: "translate(-50%, -50%)",
            }}
          >
            {fmtDist(m.distance * displayScale)}
          </div>
        );
      })}
      {pendingPoint && (() => {
        const p = project3D(pendingPoint[0], pendingPoint[1], pendingPoint[2], cam, w, h);
        if (!p) return null;
        return (
          <div
            className="size-3 rounded-full border-2 border-yellow-300 bg-yellow-400/40"
            style={{
              position: "absolute",
              left: `${p.x}px`,
              top: `${p.y}px`,
              transform: "translate(-50%, -50%)",
            }}
          />
        );
      })()}
    </>
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