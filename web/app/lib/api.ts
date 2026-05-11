import type {
  Annotation,
  ChatMessage,
  CostAggregate,
  DiscardedAnnotation,
  GatewayHealth,
  JobSettings,
  Lane,
  LanePayload,
  Manifest,
  TraceResponse,
} from "./types";

export function getArtifactUrl(sceneId: string, artifact: string): string {
  return `/artifacts/scenes/${encodeURIComponent(sceneId)}/${artifact}`;
}

export class HttpError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "HttpError";
    this.status = status;
  }
}

async function unwrap<T>(res: Response, label: string): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `${label}: ${res.status}` }));
    throw new HttpError(err.error ?? `${label}: ${res.status}`, res.status);
  }
  return res.json();
}

export async function submitJob(params: {
  scene_id: string;
  upload_path: string;
  settings: JobSettings;
}): Promise<{ status: string; scene_id: string }> {
  const res = await fetch("/api/jobs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
  return unwrap(res, "submitJob");
}

export async function fetchManifest(sceneId: string): Promise<Manifest> {
  const res = await fetch(`/api/jobs/${sceneId}`);
  return unwrap(res, "fetchManifest");
}

/** Lane → artifact filename. Lane B is the default (it's the VLM-verified
 *  labels everyone falls back to); the legacy "annotations.json" path is
 *  kept as the absolute fallback for old scenes. */
function laneArtifact(lane?: Lane): string {
  switch (lane) {
    case "b":
      return "annotations.b.json";
    case "e":
      return "annotations.e.json";
    case "f":
      return "annotations.f.json";
    default:
      return "annotations.json";
  }
}

/** Normalise the per-lane payload — Lane B is a bare Annotation[], Lanes E
 *  and F wrap it in `{annotations, edges?, layout?}`. */
export async function fetchLanePayload(
  sceneId: string,
  lane?: Lane,
): Promise<LanePayload> {
  let res = await fetch(getArtifactUrl(sceneId, laneArtifact(lane)));
  // Fall back to the legacy filename if the lane file isn't present (e.g.
  // an older scene was rendered before the lane refactor).
  if (!res.ok && lane && lane !== "b") {
    res = await fetch(getArtifactUrl(sceneId, "annotations.json"));
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `fetchLanePayload: ${res.status}` }));
    throw new Error(err.error ?? `fetchLanePayload: ${res.status}`);
  }
  const body = await res.json();
  if (Array.isArray(body)) return { annotations: body };
  return {
    annotations: body.annotations ?? [],
    edges: body.edges,
    layout: body.layout,
  };
}

/** Tracks that Lane B labelled but the postprocess dropped (or merged).
 *  Returns [] if the artifact isn't present (older scenes pre-discarded
 *  feature, or scenes where Lane B was skipped). */
export async function fetchDiscardedAnnotations(
  sceneId: string,
): Promise<DiscardedAnnotation[]> {
  const res = await fetch(getArtifactUrl(sceneId, "annotations.b.discarded.json"));
  if (!res.ok) return [];
  const body = await res.json();
  return Array.isArray(body) ? (body as DiscardedAnnotation[]) : [];
}

export async function fetchAnnotations(
  sceneId: string,
  lane?: Lane,
): Promise<Annotation[]> {
  const payload = await fetchLanePayload(sceneId, lane);
  return payload.annotations;
}

export async function fetchPointsUrl(sceneId: string): Promise<string> {
  // The viewer parser is hard-coded for points.ply (xyz + uchar rgb +
  // optional confidence) — the dense colour point cloud produced by
  // FlashVGGT at the end of the poses stage.
  return getArtifactUrl(sceneId, "points.ply");
}

/**
 * Frame URL for the evidence gallery. The segmentation lift writes a
 * per-(track, frame) JPG cropped to the GDINO bbox (with padding) and
 * downsized to ~384 px under `evidence/<annotation_id>/<frame_stem>.jpg`.
 * The full-resolution `frames/` directory is intentionally NOT pulled
 * back to the local disk anymore — these crops are the only frame
 * imagery the UI ever needs.
 *
 * Annotation.frame_ids comes through with the `.png` extension (e.g.
 * `0001.png`) for backwards compatibility, so we strip it before
 * appending `.jpg`.
 */
export function evidenceFrameUrl(
  sceneId: string,
  annotationId: string,
  frameName: string,
): string {
  const stem = frameName.replace(/\.[^./]+$/, "");
  return getArtifactUrl(sceneId, `evidence/${annotationId}/${stem}.jpg`);
}

/**
 * Mask URL for the evidence gallery. Written by `segmentation.lift._write_track_evidence`
 * as binary PNGs cropped to the same bbox+pad region (and resized to
 * the same dimensions) as the matching evidence JPG, so the CSS
 * `mask-image` overlay lines up 1:1. SAM 2.1 mask where the lift had
 * one; bbox-fill rectangle on grid-fallback frames.
 */
export function maskUrl(
  sceneId: string,
  annotationId: string,
  frameName: string,
): string {
  const stem = frameName.replace(/\.[^./]+$/, "");
  return getArtifactUrl(sceneId, `masks/${annotationId}/${stem}.png`);
}

export async function postLocate(params: {
  scene_id: string;
  camera_pos: [number, number, number];
  camera_dir: [number, number, number];
  nearby: Array<{ id: string; label: string; centroid: [number, number, number] }>;
}): Promise<{
  text: string;
  primary_object_id: string | null;
  latency_ms: number;
  model: string;
}> {
  const res = await fetch("/api/agent/locate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
  return unwrap(res, "postLocate");
}

export async function postChat(params: {
  scene_id: string;
  message: string;
  camera_pos: [number, number, number];
}): Promise<ChatMessage> {
  const res = await fetch("/api/agent/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
  return unwrap(res, "postChat");
}

export async function fetchTrace(sceneId: string): Promise<TraceResponse> {
  const res = await fetch(`/api/trace/${sceneId}`);
  return unwrap(res, "fetchTrace");
}

/** Lightweight per-scene cost — same backend cache as `fetchTrace`, just
 *  the aggregated dollar / token / call totals. Drives the header
 *  CostBadge so it can refresh without pulling the full span tree. */
export async function fetchCost(sceneId: string): Promise<CostAggregate> {
  const res = await fetch(`/api/trace/${sceneId}/cost`);
  return unwrap(res, "fetchCost");
}

export interface SceneSummary {
  scene_id: string;
  status?: string;
  created_at?: string;
  stats?: {
    frame_count?: number;
    object_count?: number;
    splat_size_mb?: number;
  };
  thumbnail?: string;
  total_duration_s?: number;
}

export async function fetchScenes(): Promise<SceneSummary[]> {
  const res = await fetch("/api/scenes", { cache: "no-store" });
  if (!res.ok) throw new Error(`fetchScenes: ${res.status}`);
  return res.json();
}

export async function fetchGatewayHealth(): Promise<GatewayHealth | null> {
  try {
    const res = await fetch("/api/gateway/health");
    if (!res.ok) return null;
    const health = (await res.json()) as GatewayHealth;
    return health.ok ? health : null;
  } catch {
    return null;
  }
}
