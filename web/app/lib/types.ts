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
    wireframe_ply?: string;
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

export type VlmModelId = "claude-haiku-4-5" | "claude-sonnet-4-6" | "claude-opus-4-7";

export interface JobSettings {
  fps: number;
  max_frames: number;
  target_long_side: number;
  segment: boolean;
  keyframes: number;
  vlm_model: VlmModelId;
}

export const VLM_MODEL_OPTIONS = [
  { id: "claude-haiku-4-5" as VlmModelId, label: "Haiku 4.5", cost: "$0.001" },
  { id: "claude-sonnet-4-6" as VlmModelId, label: "Sonnet 4.6", cost: "$0.003" },
  { id: "claude-opus-4-7" as VlmModelId, label: "Opus 4.7", cost: "$0.015" },
] as const;

export interface GatewayHealth {
  ok: boolean;
  key_set: boolean;
  region: "eu" | "us" | "unknown";
  probe_status: number | null;
  latency_ms: number;
}

export interface TraceTreeNode {
  span_id: string;
  parent_span_id: string | null;
  span_name: string;
  start_timestamp: string;
  end_timestamp: string;
  /** Span wall-clock duration. The agent sends seconds in `duration` —
   *  `duration_ms` lingers from older builds where milliseconds was the
   *  only field; both are accepted via `nodeDurationS`. */
  duration?: number;
  duration_ms?: number;
  trace_id: string;
  attributes: Record<string, unknown>;
  children: TraceTreeNode[];
}

/** Resolve a span node's duration in seconds. Returns null when neither
 *  `duration` (seconds) nor `duration_ms` (milliseconds) is a finite
 *  number — so the UI can render `—` instead of `NaN s`. */
export function nodeDurationS(node: {
  duration?: number;
  duration_ms?: number;
}): number | null {
  if (typeof node.duration === "number" && Number.isFinite(node.duration)) {
    return node.duration;
  }
  if (typeof node.duration_ms === "number" && Number.isFinite(node.duration_ms)) {
    return node.duration_ms / 1000;
  }
  return null;
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

export interface TraceResponse {
  scene_id: string;
  span_count: number;
  tree: TraceTreeNode[];
  cost: CostAggregate;
}
