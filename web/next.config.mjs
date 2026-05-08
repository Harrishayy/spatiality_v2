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
    const agent = process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8000";
    return [
      { source: "/api/:path*", destination: `${agent}/api/:path*` },
      { source: "/artifacts/:path*", destination: `${agent}/artifacts/:path*` },
    ];
  },
};

export default nextConfig;
