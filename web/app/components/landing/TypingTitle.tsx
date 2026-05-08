"use client";

import { useEffect, useState } from "react";

const TARGET = "spatiality_v2";
const CHAR_MS = 95;
const START_DELAY_MS = 240;

export function TypingTitle() {
  const [typed, setTyped] = useState("");

  useEffect(() => {
    let i = 0;
    const start = window.setTimeout(() => {
      const id = window.setInterval(() => {
        i++;
        setTyped(TARGET.slice(0, i));
        if (i >= TARGET.length) window.clearInterval(id);
      }, CHAR_MS);
      cleanup = () => window.clearInterval(id);
    }, START_DELAY_MS);

    let cleanup = () => window.clearTimeout(start);
    return () => cleanup();
  }, []);

  const done = typed.length === TARGET.length;

  return (
    <span className="lp-home-brand-title" aria-label={TARGET}>
      <span aria-hidden="true">{typed}</span>
      <span
        className={`lp-home-brand-caret${done ? " lp-home-brand-caret--blink" : ""}`}
        aria-hidden="true"
      >
        _
      </span>
    </span>
  );
}
