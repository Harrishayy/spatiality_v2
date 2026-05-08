import { useQuery } from "@tanstack/react-query";
import { fetchAnnotations, fetchManifest, fetchPointsUrl } from "@/lib/api";
import type { Annotation, BBox, Vec3 } from "@/lib/types";

const POLL_MS = 2000;

// VGGT outputs OpenCV-convention world coords (+Y down, +Z forward). The
// SplatViewer parser flips Y and Z on every point so the cloud renders
// right-side-up in Three.js (+Y up, +Z toward camera). Annotation centroids
// and bboxes are produced in the same world frame upstream, so they need
// the same flip — otherwise labels would float above/behind the wrong
// objects.
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
  const manifest = useQuery({
    queryKey: ["manifest", sceneId],
    queryFn: () => fetchManifest(sceneId),
    // Poll until both top-level pipeline AND segmentation reach a terminal
    // state. We can't stop on status="ready" alone — segmentation may still
    // be running in the background after splat completes.
    refetchInterval: (q) => {
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

  const annotations = useQuery({
    queryKey: ["annotations", sceneId],
    queryFn: async () => (await fetchAnnotations(sceneId)).map(flipAnnotation),
    enabled: segReady,
  });

  const splatUrl = useQuery({
    queryKey: ["splatUrl", sceneId],
    queryFn: () => fetchPointsUrl(sceneId),
    enabled: splatReady,
  });

  // Backward-compat: `ready` used to mean "everything ready". Keep it as the
  // narrower "splat is renderable" signal — that's what its only consumer
  // (the page) actually needed.
  return { manifest, annotations, splatUrl, splatReady, segReady, ready: splatReady };
}
