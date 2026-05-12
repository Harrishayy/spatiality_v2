/**
 * Deployment-mode flags read from public env vars.
 *
 * Public env vars (NEXT_PUBLIC_*) are inlined at build time, so these
 * helpers can be called from either server or client components and the
 * result is identical to a top-level literal.
 */

/**
 * "Demo-only" mode — the hosted (Vercel) build where uploads are disabled
 * and the only viewable scene is the pre-baked `demo_piece` served from
 * Cloudflare R2 (see `NEXT_PUBLIC_DEMO_CDN_URL` and the rewrites in
 * `web/next.config.mjs`).
 *
 * Set `NEXT_PUBLIC_DEMO_ONLY=1` (or any truthy string) in the deploy
 * environment to flip the UI into read-only mode. Local clones leave it
 * unset → upload card + full pipeline UI stays available.
 */
export function isDemoOnly(): boolean {
  const v = process.env.NEXT_PUBLIC_DEMO_ONLY ?? "";
  return v !== "" && v !== "0" && v.toLowerCase() !== "false";
}

/** Where to send users who try to upload in demo-only mode. */
export const DEMO_SCENE_PATH = "/scenes/demo_piece";
