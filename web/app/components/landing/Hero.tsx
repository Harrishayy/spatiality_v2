import Link from "next/link";
import { DEMO_SCENE_PATH, isDemoOnly } from "@/lib/env";
import { MeshHero } from "./MeshHero";

export function LandingHero() {
  // In demo-only mode (the hosted Vercel build), there's no upload
  // backend — flip the CTA to send visitors straight into the pre-baked
  // demo scene served from R2.
  const demoOnly = isDemoOnly();

  return (
    <section className="lp-hero lp-hero--home">
      <div className="lp-hero-canvas">
        <MeshHero />
      </div>
      <div className="lp-hero-grid" aria-hidden="true" />
      <div className="lp-hero-vignette" aria-hidden="true" />

      <div className="lp-hero-inner lp-hero-inner--centered">
        <h1 className="lp-hero-title lp-hero-title--home">
          {demoOnly ? (
            <>
              <span className="lp-serif">A phone video</span>
              <br />
              <span className="lp-sans">turned into a 3D </span>
              <span className="lp-serif lp-serif-accent">point cloud.</span>
            </>
          ) : (
            <>
              <span className="lp-serif">Capture a scene</span>
              <br />
              <span className="lp-sans">and turn it into a </span>
              <span className="lp-serif lp-serif-accent">point cloud.</span>
            </>
          )}
        </h1>

        <div className="lp-hero-cta lp-hero-cta--centered">
          <Link
            className="lp-btn lp-btn-primary lp-btn-lg"
            href={demoOnly ? DEMO_SCENE_PATH : "/upload"}
          >
            {demoOnly ? "View demo scene" : "Capture a scene"}
            <span className="lp-btn-arrow">→</span>
          </Link>
        </div>
      </div>

      <div className="lp-hero-fade" aria-hidden="true" />
    </section>
  );
}
