"use client";

import { useCallback, useRef, useState } from "react";

export type UploadStatus = "idle" | "uploading" | "done" | "error";

export interface UploadResult {
  sceneId: string;
  uploadPath: string;
}

export interface UploadState {
  status: UploadStatus;
  progress: number; // 0..1
  error?: string;
  file?: File;
  durationS?: number;
  result?: UploadResult;
}

const initial: UploadState = { status: "idle", progress: 0 };

export function useUpload() {
  const [state, setState] = useState<UploadState>(initial);
  const xhrRef = useRef<XMLHttpRequest | null>(null);

  const reset = useCallback(() => {
    xhrRef.current?.abort();
    xhrRef.current = null;
    setState(initial);
  }, []);

  const start = useCallback(async (file: File) => {
    setState({ status: "uploading", progress: 0, file });

    // Best-effort duration probe; failure is OK.
    probeDuration(file).then((d) => {
      if (d != null) setState((s) => ({ ...s, durationS: d }));
    });

    try {
      const fd = new FormData();
      fd.append("video", file, file.name);
      const out = await xhrPostMultipart("/api/uploads/local", fd, (p) => {
        setState((s) => ({ ...s, progress: p }));
      }, xhrRef);
      setState((s) => ({
        ...s,
        status: "done",
        progress: 1,
        result: { sceneId: out.scene_id, uploadPath: out.upload_path },
      }));
    } catch (err) {
      setState((s) => ({ ...s, status: "error", error: String((err as Error).message ?? err) }));
    }
  }, []);

  return { state, start, reset };
}

function xhrPostMultipart(
  url: string,
  body: FormData,
  onProgress: (p: number) => void,
  ref: React.MutableRefObject<XMLHttpRequest | null>,
): Promise<{ scene_id: string; upload_path: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    ref.current = xhr;
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      try {
        const parsed = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) {
          onProgress(1);
          resolve(parsed);
        } else {
          reject(new Error(parsed.error ?? `${xhr.status}`));
        }
      } catch {
        reject(new Error(`bad response: ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error("network error"));
    xhr.onabort = () => reject(new Error("aborted"));
    xhr.send(body);
  });
}

function probeDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(Number.isFinite(v.duration) ? v.duration : null);
    };
    v.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(null);
    };
    v.src = url;
  });
}
