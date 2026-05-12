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
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      { source: "/artifacts/:path*", destination: `${backend}/artifacts/:path*` },
    ];
  },
};

export default nextConfig;
