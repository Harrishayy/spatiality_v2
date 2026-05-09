import { useQuery } from "@tanstack/react-query";
import { fetchLanePayload, fetchManifest, fetchPointsUrl, HttpError } from "@/lib/api";
import { useUI } from "@/store/ui";
import type { Annotation, BBox, LanePayload, Lane, SceneEdge, SpatialLayout, Vec3 } from "@/lib/types";

const POLL_MS = 2000;

// VGGT outputs OpenCV-convention world coords (+Y down, +Z forward). The
// SplatViewer parser flips Y and Z on every point so the cloud renders
// right-side-up in Three.js (+Y up, +Z toward camera). Annotation centroids
// and bboxes — and now Lane E edges and Lane F walls / doors / windows —
// are produced in the same world frame upstream, so they need the same flip.
function flipPoint(p: Vec3): Vec3 {
  return [p[0], -p[1], -p[2]];
}
function flipBBox(b: BBox): BBox {
  // After negating Y and Z, the original lo/hi corners swap on those axes.
  // Re-establish the lo[i] <= hi[i] invariant by mixing components.
  return [
    [b[0][0], -b[1][1], -b[1][2]],
    [b[1][0], -b[0][1], -b[0][2]],
  ];
}
function flipAnnotation(a: Annotation): Annotation {
  return { ...a, centroid: flipPoint(a.centroid), bbox: flipBBox(a.bbox) };
}
function flipLayout(layout: SpatialLayout): SpatialLayout {
  return {
    walls: layout.walls.map((w) => ({
      a: flipPoint(w.a),
      b: flipPoint(w.b),
      height: w.height,
    })),
    doors: layout.doors.map((d) => ({ center: flipPoint(d.center), extent: d.extent })),
    windows: layout.windows.map((w) => ({ center: flipPoint(w.center), extent: w.extent })),
  };
}
function flipPayload(payload: LanePayload): LanePayload {
  return {
    annotations: payload.annotations.map(flipAnnotation),
    edges: payload.edges,
    layout: payload.layout ? flipLayout(payload.layout) : undefined,
  };
}

export function useScene(sceneId: string) {
  const lane: Lane = useUI((s) => s.lane);

  const manifest = useQuery({
    queryKey: ["manifest", sceneId],
    queryFn: () => fetchManifest(sceneId),
    // 404 = scene_id is stale (e.g. localStorage points at a deleted job).
    // Don't burn retries or keep polling — the page reads `manifest.error`
    // and bounces the user back to the landing page.
    retry: (_n, err) => !(err instanceof HttpError && err.status === 404),
    refetchInterval: (q) => {
      const err = q.state.error;
      if (err instanceof HttpError && err.status === 404) return false;
      const m = q.state.data;
      if (!m) return POLL_MS;
      const top = m.status;
      const seg = m.stages.segmentation.status;
      const topDone = top === "ready" || top === "failed";
      const segDone = seg === "complete" || seg === "failed";
      return topDone && segDone ? false : POLL_MS;
    },
  });

  const splatReady = manifest.data?.stages.splat.status === "complete";
  const segReady = manifest.data?.stages.segmentation.status === "complete";

  // Lane is part of the cache key so switching lanes triggers a refetch.
  // The fetcher applies the y/z flip once; downstream consumers see the
  // viewer-frame payload directly.
  const lanePayload = useQuery({
    queryKey: ["lane-payload", sceneId, lane],
    queryFn: async () => flipPayload(await fetchLanePayload(sceneId, lane)),
    enabled: segReady,
  });

  // Same cache entry, surfaced via `select` so the existing `annotations.data`
  // call sites still work and we don't fake-cast a UseQueryResult shape.
  const annotations = useQuery({
    queryKey: ["lane-payload", sceneId, lane],
    queryFn: async () => flipPayload(await fetchLanePayload(sceneId, lane)),
    enabled: segReady,
    select: (p: LanePayload): Annotation[] => p.annotations,
  });

  const edges: SceneEdge[] | undefined = lanePayload.data?.edges;
  const layout: SpatialLayout | undefined = lanePayload.data?.layout;

  const splatUrl = useQuery({
    queryKey: ["splatUrl", sceneId],
    queryFn: () => fetchPointsUrl(sceneId),
    enabled: splatReady,
  });

  return {
    manifest,
    annotations,
    edges,
    layout,
    splatUrl,
    splatReady,
    segReady,
    ready: splatReady,
    lane,
  };
}
