import { create } from "zustand";
import type { Vec3 } from "@/lib/types";

interface CameraState {
  position: Vec3;
  direction: Vec3;
}

interface CloudStats {
  /** Number of points actually rendered (parsed from points.ply). */
  count: number;
  /** Bytes downloaded for the cloud (Content-Length of the streamed PLY). */
  sizeMb: number;
}

/** Axis-aligned bounding box in viewer (Three.js) coordinates — set by the
 *  PLY parser once the stream completes. Used to drive the always-on
 *  dimension overlay and the schematic mode helpers. */
export interface SceneBounds {
  min: Vec3;
  max: Vec3;
}

/** Render-coloring mode for the point cloud.
 *  - "rgb"        : per-point RGB from points.ply (default).
 *  - "depth"      : turbo-colormap by distance from scene center (LiDAR look).
 *  - "confidence" : turbo-colormap by per-point VGGT confidence. */
export type RenderMode = "rgb" | "depth" | "confidence";

/** Pipeline stages the drill-down drawer can open into. Matches the
 *  stage keys on Manifest.stages; "splat" is intentionally absent because
 *  it's hidden in the UI (PipelineProgress.STAGE_ORDER). */
export type DrillStage = "capture" | "poses" | "segmentation";

/** A user-placed measurement: two world-space points and the measured
 *  distance in metres. Drawn as a line + label in the viewer. */
export interface Measurement {
  id: string;
  a: Vec3;
  b: Vec3;
  distance: number;
}

interface UIState {
  selectedId: string | null;
  isolatedIds: Set<string>;
  camera: CameraState;
  /** Live stats from the SplatViewer about the cloud actually rendered.
   *  Distinct from manifest.stages.splat.gaussian_count (which is splat.ply,
   *  ~10× smaller, used by segmentation for clustering). */
  cloudStats: CloudStats | null;
  bounds: SceneBounds | null;
  renderMode: RenderMode;
  schematicMode: boolean;
  /** When true, the viewer hides the full point cloud and renders a
   *  reduced wireframe (voxel-downsampled background + per-object dense
   *  points connected by kNN edges, monochrome, with floating labels at
   *  annotation centroids). The full cloud's GPU buffers stay resident
   *  so toggling back is instant. Mirrors `schematicMode` semantics. */
  wireframeMode: boolean;
  /** When true, viewer clicks place measurement endpoints instead of doing
   *  the default action. Toggled via the "Measure" toolbar button. */
  measureMode: boolean;
  measurements: Measurement[];
  /** Click-to-measure state. While `pendingPoint` is set, the next click
   *  finishes a measurement instead of starting one. */
  pendingPoint: Vec3 | null;
  /** Multiplier applied to every displayed distance / dimension. VGGT
   *  outputs are metric-consistent but not always calibrated to true
   *  metres; this is what the Calibrate UX writes (real / measured) so
   *  every label reflects real-world units. 1.0 = uncalibrated. */
  displayScale: number;
  /** Pipeline drill-down drawer — null when closed. */
  openStage: DrillStage | null;
  setOpenStage: (stage: DrillStage | null) => void;
  setSelected: (id: string | null) => void;
  toggleIsolated: (id: string) => void;
  clearIsolated: () => void;
  setCamera: (pos: Vec3, dir: Vec3) => void;
  setCloudStats: (stats: CloudStats | null) => void;
  setBounds: (bounds: SceneBounds | null) => void;
  setRenderMode: (mode: RenderMode) => void;
  cycleRenderMode: () => void;
  toggleSchematic: () => void;
  toggleWireframe: () => void;
  toggleMeasureMode: () => void;
  beginMeasurement: (p: Vec3) => void;
  finishMeasurement: (p: Vec3) => void;
  cancelMeasurement: () => void;
  clearMeasurements: () => void;
  setDisplayScale: (s: number) => void;
  /** Use the most recent measurement's raw distance and the user-supplied
   *  real value to derive the new displayScale. No-op if there are no
   *  measurements yet or the input isn't a positive finite number. */
  calibrateFromLastMeasurement: (realMetres: number) => void;
}

const RENDER_MODE_ORDER: RenderMode[] = ["rgb", "depth", "confidence"];

export const useUI = create<UIState>((set) => ({
  selectedId: null,
  isolatedIds: new Set<string>(),
  camera: { position: [0, 0, 0], direction: [0, 0, -1] },
  cloudStats: null,
  bounds: null,
  renderMode: "rgb",
  schematicMode: false,
  wireframeMode: false,
  measureMode: false,
  measurements: [],
  pendingPoint: null,
  displayScale: 1,
  openStage: null,
  setOpenStage: (openStage) => set({ openStage }),
  setSelected: (id) => set({ selectedId: id }),
  toggleIsolated: (id) =>
    set((s) => {
      const next = new Set(s.isolatedIds);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { isolatedIds: next };
    }),
  clearIsolated: () => set({ isolatedIds: new Set() }),
  setCamera: (position, direction) =>
    set(() => ({ camera: { position, direction } })),
  setCloudStats: (cloudStats) => set({ cloudStats }),
  setBounds: (bounds) => set({ bounds }),
  setRenderMode: (renderMode) => set({ renderMode }),
  cycleRenderMode: () =>
    set((s) => ({
      renderMode:
        RENDER_MODE_ORDER[
          (RENDER_MODE_ORDER.indexOf(s.renderMode) + 1) % RENDER_MODE_ORDER.length
        ],
    })),
  toggleSchematic: () => set((s) => ({ schematicMode: !s.schematicMode })),
  toggleWireframe: () => set((s) => ({ wireframeMode: !s.wireframeMode })),
  toggleMeasureMode: () =>
    set((s) => ({
      measureMode: !s.measureMode,
      // Cancel any in-flight first click when leaving measure mode.
      pendingPoint: s.measureMode ? null : s.pendingPoint,
    })),
  beginMeasurement: (p) => set({ pendingPoint: p }),
  finishMeasurement: (p) =>
    set((s) => {
      if (!s.pendingPoint) return { pendingPoint: p };
      const dx = p[0] - s.pendingPoint[0];
      const dy = p[1] - s.pendingPoint[1];
      const dz = p[2] - s.pendingPoint[2];
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const m: Measurement = {
        id: `m_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
        a: s.pendingPoint,
        b: p,
        distance,
      };
      return { measurements: [...s.measurements, m], pendingPoint: null };
    }),
  cancelMeasurement: () => set({ pendingPoint: null }),
  clearMeasurements: () => set({ measurements: [], pendingPoint: null }),
  setDisplayScale: (displayScale) =>
    set({ displayScale: Number.isFinite(displayScale) && displayScale > 0 ? displayScale : 1 }),
  calibrateFromLastMeasurement: (realMetres) =>
    set((s) => {
      const last = s.measurements[s.measurements.length - 1];
      if (!last || last.distance <= 0) return {};
      if (!Number.isFinite(realMetres) || realMetres <= 0) return {};
      // Calibration is anchored to the *raw* (uncalibrated) measurement
      // distance so re-calibrating a second time doesn't compound. We
      // recover the raw distance by undoing the current scale.
      const rawDistance = last.distance;
      return { displayScale: realMetres / rawDistance };
    }),
}));
