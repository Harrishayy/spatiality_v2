"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

import { SettingsPanel, DEFAULT_SETTINGS } from "@/components/SettingsPanel";
import { Uploader } from "@/components/Uploader";
import { TypingTitle } from "@/components/landing/TypingTitle";
import { useUpload } from "@/hooks/useUpload";
import { submitJob } from "@/lib/api";

export default function UploadPage() {
  const router = useRouter();
  const { state, start, reset } = useUpload();
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
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
    <div className="lp-home w-full max-w-[100vw] overflow-x-hidden">
      <header className="lp-home-header lp-home-header--row">
        <Link className="lp-home-brand" href="/">
          <span className="lp-home-brand-mark" aria-hidden="true" />
          <TypingTitle />
        </Link>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-col gap-12 px-10 pb-16 pt-4 max-md:px-5">
        <div className="grid w-full grid-cols-1 gap-8 md:grid-cols-[1.05fr_0.95fr]">
          <div className="min-w-0">
            <Uploader
              state={state}
              onPick={(file) => start(file)}
              onReset={reset}
            />
          </div>

          <div className="flex min-w-0 flex-col gap-5">
            <SettingsPanel
              value={settings}
              onChange={setSettings}
              durationS={state.durationS}
            />
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
        </div>
      </main>
    </div>
  );
}
