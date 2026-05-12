/** @type {import('next').NextConfig} */
const nextConfig = {
  // Off in dev: PointCloudViewer owns an imperative Three.js WebGL context
  // whose dispose() does not fully tear down its DOM children. Strict mode's
  // double-invoked effect would leave orphan canvases in the container, which
  // then trips React's removeChild on the next render.
  reactStrictMode: false,

  // tanstack-query is a barrel index — the optimizePackageImports flag lets
  // Turbopack ship only the symbols actually used. PointCloudViewer already
  // uses named imports from `three`, so three doesn't need the flag (named
  // imports tree-shake natively, and the flag would force an extra index
  // walk that defeats the point on cold compile).
  experimental: {
    optimizePackageImports: ["@tanstack/react-query"],
  },

  async rewrites() {
    // The local backend is `uvicorn backend.main:app --port 8765` from the
    // repo root. Override with SPATIALITY_API_URL only if you've actually
    // moved it elsewhere — e.g. a tunnel for someone testing remotely.
    const backend = process.env.SPATIALITY_API_URL ?? "http://localhost:8765";

    // The full demo_piece scene (1.3 GB points.ply + ~20 MB of evidence/
    // masks/JSONs) is NOT committed to the repo — it lives in a Cloudflare
    // R2 bucket. Set `NEXT_PUBLIC_DEMO_CDN_URL` to the bucket's public URL
    // (e.g. `https://<id>.r2.dev`) and the rewrites below proxy the
    // viewer's manifest + artefact fetches straight to R2.
    //
    // Bucket layout (rooted at the URL):
    //     manifest.json
    //     points.ply
    //     cameras.json
    //     annotations.b.json
    //     annotations.c.json
    //     traversability.{json,png}
    //     evidence/<obj>/<frame>.jpg
    //     masks/<obj>/<frame>.png
    //
    // When the env var is unset, the demo rewrites are skipped and the
    // demo_piece URLs fall through to the local FastAPI catch-all — so
    // a clone that's also running uvicorn against `backend/data/outputs/
    // demo_piece/` (e.g. after unzipping `demo_piece_full.zip`) still
    // sees the demo. Reviewers without uvicorn need the R2 URL set.
    const demoCdn = (process.env.NEXT_PUBLIC_DEMO_CDN_URL ?? "").replace(/\/$/, "");
    const demoRewrites = demoCdn
      ? [
          {
            source: "/api/jobs/demo_piece",
            destination: `${demoCdn}/manifest.json`,
          },
          {
            source: "/artifacts/scenes/demo_piece/:path*",
            destination: `${demoCdn}/:path*`,
          },
        ]
      : [];

    return [
      // Order matters: specific demo rewrites (when configured) must come
      // BEFORE the catch-all that proxies the rest of /api/* and
      // /artifacts/* to the local FastAPI. Next.js matches top-to-bottom.
      ...demoRewrites,
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      { source: "/artifacts/:path*", destination: `${backend}/artifacts/:path*` },
    ];
  },
};

export default nextConfig;
