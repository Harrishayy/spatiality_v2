export type Vec3 = [number, number, number];
export type BBox = [Vec3, Vec3];

export interface Annotation {
  id: string;
  label: string;
  centroid: Vec3;
  bbox: BBox;
  color: string;
  confidence: number;
  alternatives?: string[];
  cluster_gaussian_indices?: number[];
  provenance?: string[];
  frame_ids?: string[];
}

/** Pipeline stage that produced a discard record. */
export type DiscardStage = "gdino" | "lift" | "postprocess";

/** Reasons a track can be cut. The set is a flat union across stages —
 *  `stage` plus `discard_reason` together identifies the cause. */
export type DiscardReason =
  // GDINO (Stage 3.2 — detection + IoU tracklets)
  | "short_tracklet"
  // 3D lift (Stage 3.5)
  | "multiview_filter"
  | "3d_coherence"
  | "reprojection"
  | "merged_3d"
  // Lane B postprocess (Stage 3.6 cleanup)
  | "scene_label"
  | "low_confidence"
  | "oversize"
  | "merged_duplicate";

/** A track the pipeline considered but dropped at some stage. Geometry
 *  fields are optional — only postprocess-stage discards (which made it
 *  through 3D lift + Lane B labelling) carry a centroid/bbox/color. */
export interface DiscardedAnnotation {
  id: string;
  label: string;
  stage: DiscardStage;
  discard_reason: DiscardReason;
  discard_detail?: string;
  // Postprocess-stage extras — present when the track was lifted.
  centroid?: Vec3;
  bbox?: BBox;
  color?: string;
  confidence?: number;
  alternatives?: string[];
  frame_ids?: string[];
  merged_into?: string;
  // Earlier-stage extras.
  n_frames?: number;
  source?: string;
}

/** Which labeling lane the user is currently viewing.
 *  - "b": VLM-verified labels (Claude over orbital novel-view renders).
 *  - "e": ConceptGraphs-style scene graph (objects + relationship edges).
 *  - "f": SpatialLM layout-only (walls / doors / windows). */
export type Lane = "b" | "e" | "f";

/** Spatial relationship edge between two annotations (Lane E). */
export interface SceneEdge {
  from: string;
  to: string;
  relation:
    | "on" | "under" | "next-to" | "contains" | "supports"
    | "behind" | "in-front-of";
  confidence: number;
}

/** SpatialLM-derived axis-aligned room layout (Lane F).
 *  Coordinates are in the same frame as Annotation.centroid (OpenCV on
 *  disk; the frontend's existing y/z flip applies symmetrically). */
export interface SpatialLayout {
  walls: Array<{ a: Vec3; b: Vec3; height?: number }>;
  doors: Array<{ center: Vec3; extent: Vec3 }>;
  windows: Array<{ center: Vec3; extent: Vec3 }>;
}

/** Full payload of any lane's annotations file. Lane B writes a bare array;
 *  Lane E adds edges; Lane F adds a layout. fetchLanePayload normalises. */
export interface LanePayload {
  annotations: Annotation[];
  edges?: SceneEdge[];
  layout?: SpatialLayout;
}

export type StageStatus = "pending" | "running" | "complete" | "failed";
export type ManifestStatus = "queued" | "processing" | "ready" | "failed";

export interface Stage {
  status: StageStatus;
  duration_s?: number;
  method?: string;
  iterations?: number;
  object_count?: number;
  frame_count?: number;
  gaussian_count?: number;
}

export interface Manifest {
  scene_id: string;
  created_at: string;
  status: ManifestStatus;
  stages: {
    capture: Stage;
    poses: Stage;
    splat: Stage;
    segmentation: Stage;
  };
  artifacts?: {
    splat_ply?: string;
    annotations_json?: string;
    thumbnail_jpg?: string;
    cameras_json?: string;
    capture_map_json?: string;
    capture_map_png?: string;
  };
  stats: {
    frame_count: number;
    object_count: number;
    splat_size_mb: number;
  };
  errors?: string[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "agent";
  text: string;
  pending?: boolean;
  frames_used?: string[];
  tools_called?: string[];
}

export type VlmModelId =
  | "gemini-2.5-flash"
  | "gemini-2.5-flash-lite"
  | "claude-haiku-4-5"
  | "claude-sonnet-4-6"
  | "claude-opus-4-7";

export interface JobSettings {
  max_frames: number;
  target_long_side: number;
  segment: boolean;
  keyframes: number;
  vlm_model: VlmModelId;
}

export interface CostAggregate {
  total_usd: number;
  call_count: number;
  by_span?: Array<{
    span_name: string;
    usd: number;
    tokens_in: number;
    tokens_out: number;
  }>;
}
