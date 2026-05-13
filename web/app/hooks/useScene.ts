import { useQuery } from "@tanstack/react-query";
import {
  fetchCameraCenters,
  fetchDiscardedAnnotations,
  fetchLanePayload,
  fetchManifest,
  fetchPointsUrl,
  HttpError,
} from "@/lib/api";
import { useUI } from "@/store/ui";
import type {
  Annotation,
  BBox,
  DiscardedAnnotation,
  Lane,
  Vec3,
} from "@/lib/types";

const POLL_MS = 2000;

// VGGT outputs OpenCV-convention world coords (+Y down, +Z forward). The
// point cloud viewer's parser flips Y and Z on every point so the cloud
// renders right-side-up in Three.js (+Y up, +Z toward camera). Annotation
// centroids and bboxes are produced in the same world frame upstream, so
// they need the same flip.
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

  const pointsReady = manifest.data?.stages.splat.status === "complete";
  const segReady = manifest.data?.stages.segmentation.status === "complete";

  // Lane is part of the cache key so switching lanes triggers a refetch.
  // The fetcher applies the y/z flip once; downstream consumers see the
  // viewer-frame payload directly.
  const annotations = useQuery({
    queryKey: ["lane-payload", sceneId, lane],
    queryFn: async () => {
      const p = await fetchLanePayload(sceneId, lane);
      return p.annotations.map(flipAnnotation);
    },
    enabled: segReady,
  });

  const pointsUrl = useQuery({
    queryKey: ["pointsUrl", sceneId],
    queryFn: () => fetchPointsUrl(sceneId),
    enabled: pointsReady,
  });

  // Postprocess-dropped Lane B tracks (scene labels, low-conf, oversize,
  // duplicates). Same y/z flip as the kept annotations so any future viewer
  // overlay using the centroid lines up.
  const discarded = useQuery({
    queryKey: ["discarded", sceneId],
    queryFn: async (): Promise<DiscardedAnnotation[]> => {
      const raw = await fetchDiscardedAnnotations(sceneId);
      // Only postprocess-stage discards carry geometry; upstream stages
      // (gdino, lift) drop before the centroid/bbox exist, so leave those
      // untouched rather than flipping undefined values.
      return raw.map((a) => ({
        ...a,
        centroid: a.centroid ? flipPoint(a.centroid) : undefined,
        bbox: a.bbox ? flipBBox(a.bbox) : undefined,
      }));
    },
    enabled: segReady,
  });

  // Camera centers from cameras.json — used by the viewer to spawn the
  // initial camera inside the room (where the user was actually standing
  // when capturing), rather than 1.5m offset from the annotation centroid.
  // Available as soon as poses are written; doesn't depend on segmentation.
  const posesReady = manifest.data?.stages.poses.status === "complete";
  const cameraCenters = useQuery({
    queryKey: ["cameraCenters", sceneId],
    queryFn: () => fetchCameraCenters(sceneId),
    enabled: posesReady,
    staleTime: Infinity,
  });

  return {
    manifest,
    annotations,
    discarded,
    pointsUrl,
    cameraCenters,
    pointsReady,
    segReady,
  };
}
