import { create } from "zustand";
import type { Lane, Vec3 } from "@/lib/types";

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

/** Render-coloring mode for the point cloud.
 *  - "rgb"        : per-point RGB from points.ply (default).
 *  - "depth"      : turbo-colormap by distance from scene center (LiDAR look).
 *  - "confidence" : turbo-colormap by per-point VGGT confidence. */
export type RenderMode = "rgb" | "depth" | "confidence";

interface UIState {
  selectedId: string | null;
  isolatedIds: Set<string>;
  camera: CameraState;
  /** Live stats from the point cloud viewer about the cloud actually
   *  rendered (parsed from the streaming points.ply). */
  cloudStats: CloudStats | null;
  renderMode: RenderMode;
  /** When true, the viewer hides the full point cloud and renders a
   *  reduced wireframe (voxel-downsampled background + per-object dense
   *  points connected by kNN edges, monochrome, with floating labels at
   *  annotation centroids). The full cloud's GPU buffers stay resident
   *  so toggling back is instant. */
  wireframeMode: boolean;
  /** When true, AnnotationOverlay (object markers + labels) is shown over
   *  the 3D scene. Toggled via the "Annotations" button in the toolbar. */
  showAnnotations: boolean;
  toggleAnnotations: () => void;
  /** Active labeling lane — controls which annotations.*.json the viewer
   *  reads. "b" is the default VLM-verified labels lane. */
  lane: Lane;
  setLane: (lane: Lane) => void;
  setSelected: (id: string | null) => void;
  toggleIsolated: (id: string) => void;
  clearIsolated: () => void;
  setCamera: (pos: Vec3, dir: Vec3) => void;
  setCloudStats: (stats: CloudStats | null) => void;
  setRenderMode: (mode: RenderMode) => void;
  cycleRenderMode: () => void;
  toggleWireframe: () => void;
}

const RENDER_MODE_ORDER: RenderMode[] = ["rgb", "depth", "confidence"];

export const useUI = create<UIState>((set) => ({
  selectedId: null,
  isolatedIds: new Set<string>(),
  camera: { position: [0, 0, 0], direction: [0, 0, -1] },
  cloudStats: null,
  renderMode: "rgb",
  wireframeMode: false,
  showAnnotations: true,
  toggleAnnotations: () => set((s) => ({ showAnnotations: !s.showAnnotations })),
  lane: "b",
  setLane: (lane) => set({ lane }),
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
  setRenderMode: (renderMode) => set({ renderMode }),
  cycleRenderMode: () =>
    set((s) => ({
      renderMode:
        RENDER_MODE_ORDER[
          (RENDER_MODE_ORDER.indexOf(s.renderMode) + 1) % RENDER_MODE_ORDER.length
        ],
    })),
  toggleWireframe: () => set((s) => ({ wireframeMode: !s.wireframeMode })),
}));
