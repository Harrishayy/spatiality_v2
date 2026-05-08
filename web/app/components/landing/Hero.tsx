import Link from "next/link";
import { MeshHero } from "./MeshHero";

export function LandingHero() {
  return (
    <section className="lp-hero lp-hero--home">
      <div className="lp-hero-canvas">
        <MeshHero />
      </div>
      <div className="lp-hero-grid" aria-hidden="true" />
      <div className="lp-hero-vignette" aria-hidden="true" />

      <div className="lp-hero-inner lp-hero-inner--centered">
        <h1 className="lp-hero-title lp-hero-title--home">
          <span className="lp-serif">Capture a scene</span>
          <br />
          <span className="lp-sans">and turn it into a </span>
          <span className="lp-serif lp-serif-accent">point cloud.</span>
        </h1>

        <div className="lp-hero-cta lp-hero-cta--centered">
          <Link className="lp-btn lp-btn-primary lp-btn-lg" href="/upload">
            Capture a scene
            <span className="lp-btn-arrow">→</span>
          </Link>
        </div>
      </div>

      <div className="lp-hero-fade" aria-hidden="true" />
    </section>
  );
}
