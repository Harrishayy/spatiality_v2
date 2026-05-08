"use client";

import { useCallback, useRef, useState } from "react";

import type { UploadState } from "@/hooks/useUpload";

interface Props {
  state: UploadState;
  onPick: (file: File) => void;
  onReset: () => void;
}

export function Uploader({ state, onPick, onReset }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const f = files[0];
      if (!f.type.startsWith("video/")) return;
      onPick(f);
    },
    [onPick],
  );

  if (state.status !== "idle") {
    return <ActiveUpload state={state} onReset={onReset} />;
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      className={`lp-dropzone ${dragOver ? "lp-dropzone--over" : ""}`}
    >
      <span className="lp-dropzone-mark" aria-hidden>
        ↑
      </span>
      <div className="flex flex-col items-center gap-1.5">
        <span className="lp-dropzone-title">Drop a video to start</span>
        <span className="lp-dropzone-sub">MP4, MOV or WebM — up to 2 GB</span>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}

function ActiveUpload({
  state,
  onReset,
}: {
  state: UploadState;
  onReset: () => void;
}) {
  const { file, progress, status, error, durationS } = state;
  const pct = Math.round(progress * 100);
  return (
    <div className="lp-surface" style={{ gap: 16 }}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <span className="truncate text-[15px] font-semibold text-ink-100">
            {file?.name ?? "—"}
          </span>
          <span className="lp-field-help">
            {file ? `${(file.size / 1_000_000).toFixed(1)} MB` : ""}
            {durationS ? ` · ${durationS.toFixed(1)} s` : ""}
            {status === "uploading" ? ` · ${pct}%` : ""}
            {status === "done" ? " · uploaded" : ""}
            {status === "error" ? " · failed" : ""}
          </span>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="lp-btn lp-btn-ghost lp-btn-sm"
        >
          {status === "uploading" ? "Cancel" : "Remove"}
        </button>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-[rgba(255,235,220,0.06)]">
        <div
          className={`h-full transition-all ${
            status === "error"
              ? "bg-accent-500"
              : status === "done"
                ? "bg-emerald"
                : "bg-accent-400"
          }`}
          style={{ width: `${Math.max(pct, status === "uploading" ? 4 : 0)}%` }}
        />
      </div>
      {error && <span className="lp-field-help text-[#ffb6a3]">{error}</span>}
    </div>
  );
}
