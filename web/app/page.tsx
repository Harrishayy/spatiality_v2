import Link from "next/link";
import { LandingHero } from "@/components/landing/Hero";
import { TypingTitle } from "@/components/landing/TypingTitle";

export default function Home() {
  return (
    <div className="lp-home">
      <header className="lp-home-header">
        <Link className="lp-home-brand" href="/">
          <span className="lp-home-brand-mark" aria-hidden="true" />
          <TypingTitle />
        </Link>
      </header>
      <LandingHero />
    </div>
  );
}
