/** @type {import('next').NextConfig} */
const nextConfig = {
  // Off in dev: SplatViewer wraps an imperative WebGL viewer
  // (@mkkellogg/gaussian-splats-3d) whose dispose() does not fully tear down
  // its DOM children. Strict mode's double-invoked effect leaves orphan canvases
  // in the container, which then trips React's removeChild on the next render.
  reactStrictMode: false,

  // tanstack-query is a barrel index — the optimizePackageImports flag lets
  // Turbopack ship only the symbols actually used. SplatViewer already uses
  // named imports from `three`, so three doesn't need the flag (named
  // imports tree-shake natively, and the flag would force an extra index
  // walk that defeats the point on cold compile).
  experimental: {
    optimizePackageImports: ["@tanstack/react-query"],
  },

  async rewrites() {
    // The local backend is `uvicorn backend.main:app --port 8000` from the
    // repo root. Override with SPATIALITY_API_URL only if you've actually
    // moved it elsewhere — e.g. a tunnel for someone testing remotely.
    const backend = process.env.SPATIALITY_API_URL ?? "http://localhost:8000";
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      { source: "/artifacts/:path*", destination: `${backend}/artifacts/:path*` },
    ];
  },
};

export default nextConfig;
