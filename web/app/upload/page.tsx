"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

// Pipeline settings are intentionally hidden in the upload UI; the run uses
// DEFAULT_SETTINGS and the user gets a stage-by-stage explanation instead.
import { DEFAULT_SETTINGS } from "@/components/SettingsPanel";
import { PipelineOverview } from "@/components/PipelineOverview";
import { Uploader } from "@/components/Uploader";
import { TypingTitle } from "@/components/landing/TypingTitle";
import { useUpload } from "@/hooks/useUpload";
import { submitJob } from "@/lib/api";
import { DEMO_SCENE_PATH, isDemoOnly } from "@/lib/env";

export default function UploadPage() {
  const router = useRouter();
  const { state, start, reset } = useUpload();
  const settings = DEFAULT_SETTINGS;
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // In demo-only mode the hosted site has no FastAPI to receive uploads,
  // so we replace the dropzone with a notice + redirect to the pre-baked
  // demo scene. Local clones (env var unset) keep the full upload flow.
  const demoOnly = isDemoOnly();

  const canSubmit = state.status === "done" && !!state.result && !submitting;

  async function onStart() {
    if (!state.result) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      await submitJob({
        scene_id: state.result.sceneId,
        upload_path: state.result.uploadPath,
        settings,
      });
      router.push(`/scenes/${state.result.sceneId}`);
    } catch (err) {
      setSubmitError(String((err as Error).message ?? err));
      setSubmitting(false);
    }
  }

  return (
    <div className="lp-home flex h-[100dvh] w-full max-w-[100vw] flex-col overflow-hidden">
      <header className="lp-home-header lp-home-header--row shrink-0">
        <Link className="lp-home-brand" href="/">
          <span className="lp-home-brand-mark" aria-hidden="true" />
          <TypingTitle />
        </Link>
      </header>

      <main className="mx-auto grid min-h-0 w-full max-w-[1400px] flex-1 grid-cols-1 gap-5 overflow-hidden px-6 pb-5 pt-2 max-md:px-4 lg:grid-cols-[minmax(280px,360px)_1fr]">
        <div className="flex min-h-0 min-w-0 flex-col gap-4">
          {demoOnly ? (
            <DemoOnlyCard />
          ) : (
            <>
              <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                <Uploader
                  state={state}
                  onPick={(file) => start(file)}
                  onReset={reset}
                />
              </div>
              <button
                type="button"
                onClick={onStart}
                disabled={!canSubmit}
                className="lp-cta"
              >
                {submitting
                  ? "Queueing…"
                  : state.status === "uploading"
                    ? `Uploading ${(state.progress * 100).toFixed(0)}%`
                    : state.status === "done"
                      ? "Start pipeline"
                      : "Pick a video first"}
                <span className="lp-btn-arrow">→</span>
              </button>
              {submitError && (
                <span className="lp-field-help text-[#ffb6a3]">{submitError}</span>
              )}
            </>
          )}
        </div>

        <div className="flex min-h-0 min-w-0 flex-col overflow-hidden">
          <PipelineOverview />
        </div>
      </main>
    </div>
  );
}

/** Replaces the dropzone + Start button on the hosted (demo-only) build.
 *
 *  The hosted site doesn't run the FastAPI orchestrator, so an upload
 *  can't actually go anywhere. Rather than show a broken dropzone we
 *  explain what the right CTA is and point the visitor at the demo scene.
 *  The pipeline-overview cards on the right still render as-is — they're
 *  the architectural explainer, useful whether or not you can upload. */
function DemoOnlyCard() {
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-4">
      <div className="rounded-2xl border border-ink-700/70 bg-ink-900/85 p-5 backdrop-blur">
        <div className="font-mono text-[11px] uppercase tracking-wider text-accent-300">
          Hosted demo
        </div>
        <h2 className="mt-2 font-serif text-2xl text-ink-100">
          Uploads are disabled on the live site.
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-ink-300">
          The hosted build serves a single pre-baked scene streamed from a
          public CDN bucket. To run the pipeline on your own phone video,
          clone the repo and follow the &ldquo;Run it locally&rdquo;
          instructions in the README.
        </p>
        <Link
          href={DEMO_SCENE_PATH}
          className="lp-cta mt-4 inline-flex w-full items-center justify-center"
        >
          View demo scene
          <span className="lp-btn-arrow">→</span>
        </Link>
      </div>
      <a
        href="https://github.com/harrishayyanar/spatiality_v2"
        target="_blank"
        rel="noreferrer"
        className="rounded-xl border border-ink-700/60 bg-ink-900/60 p-4 font-mono text-[11px] text-ink-300 backdrop-blur hover:border-accent-400/60 hover:text-accent-200"
      >
        <span className="text-accent-300">Run it yourself ↗</span>
        <span className="ml-2">github.com/harrishayyanar/spatiality_v2</span>
      </a>
    </div>
  );
}
