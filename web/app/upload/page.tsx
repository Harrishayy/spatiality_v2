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

export default function UploadPage() {
  const router = useRouter();
  const { state, start, reset } = useUpload();
  const settings = DEFAULT_SETTINGS;
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

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
        </div>

        <div className="flex min-h-0 min-w-0 flex-col overflow-hidden">
          <PipelineOverview />
        </div>
      </main>
    </div>
  );
}
