"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useParams, useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { Header } from "@/components/Header";
import { PipelineProgress } from "@/components/PipelineProgress";
import {
  SceneDrawerOverlay,
  SceneSideColumn,
  type SceneSection,
} from "@/components/SidePanel";
import { useChat } from "@/hooks/useChat";
import { useScene } from "@/hooks/useScene";
import { HttpError } from "@/lib/api";
import { useUI } from "@/store/ui";
import type { Manifest, StageStatus } from "@/lib/types";

const PointCloudViewer = dynamic(
  () => import("@/components/PointCloudViewer").then((m) => m.PointCloudViewer),
  { ssr: false },
);

export default function ScenePage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const sceneId = params?.id ?? "";
  const { manifest, annotations, discarded, pointsUrl, pointsReady, segReady } = useScene(sceneId);
  const { messages, send } = useChat(sceneId);
  const selectedId = useUI((s) => s.selectedId);
  const [openSection, setOpenSection] = useState<SceneSection | null>(null);

  // Stale scene_id (deleted on disk, often comes from localStorage pointing
  // at an old job) — bounce to the landing page instead of polling 404s
  // forever. useScene already disables retry+refetch on 404.
  const notFound =
    manifest.error instanceof HttpError && manifest.error.status === 404;
  useEffect(() => {
    if (notFound) router.replace("/");
  }, [notFound, router]);

  // When the user picks an object marker, swing the Evidence drawer open.
  // When they deselect, fold Evidence back away — but leave Pipeline /
  // Objects alone so the user's choice isn't fought over.
  useEffect(() => {
    if (selectedId) setOpenSection("evidence");
    else setOpenSection((s) => (s === "evidence" ? null : s));
  }, [selectedId]);

  const m = manifest.data;
  // Stable reference: react-query gives us the same array across renders when
  // data is unchanged; the ?? [] fallback used to mint a fresh array each
  // render, which would tear down the viewer on every poll.
  const annos = useMemo(() => annotations.data ?? [], [annotations.data]);
  const discardedAnnos = useMemo(() => discarded.data ?? [], [discarded.data]);
  const emptyCloud = (m?.stats.splat_size_mb ?? 0) <= 0.001;
  const segStatus: StageStatus = m?.stages.segmentation.status ?? "pending";
  const failed = m?.status === "failed";

  return (
    <div className="flex h-screen w-screen flex-col bg-ink-950">
      <Header manifest={m} />

      <main className="relative flex min-h-0 flex-1">
        <section className="relative min-h-0 flex-1">
          {pointsReady && pointsUrl.data ? (
            <PointCloudViewer
              pointsUrl={pointsUrl.data}
              annotations={annos}
              emptyCloud={emptyCloud}
            />
          ) : (
            <PipelinePending manifest={m} failed={failed} />
          )}

          {pointsReady && !segReady && segStatus !== "failed" && (
            <SegmentingBanner status={segStatus} />
          )}

          {pointsReady && segStatus === "failed" && (
            <FailedBanner
              title="Segmentation failed"
              detail={m?.errors?.[m.errors.length - 1]}
            />
          )}

          {m && (
            <SceneDrawerOverlay
              sceneId={sceneId}
              manifest={m}
              annotations={annos}
              discarded={discardedAnnos}
              segStatus={segStatus}
              openSection={openSection}
              onClose={() => setOpenSection(null)}
            />
          )}
        </section>

        {m && (
          <SceneSideColumn
            manifest={m}
            annotations={annos}
            messages={messages}
            onSend={send}
            loading={!segReady}
            openSection={openSection}
            onToggleSection={(s) =>
              setOpenSection((prev) => (prev === s ? null : s))
            }
          />
        )}
      </main>
    </div>
  );
}

function PipelinePending({
  manifest,
  failed,
}: {
  manifest?: Manifest;
  failed: boolean;
}) {
  return (
    <div className="flex h-full w-full items-center justify-center p-6">
      <div className="flex w-full max-w-md flex-col items-stretch gap-4">
        <span
          className={`lp-status-pill ${failed ? "lp-status-pill--err" : "lp-status-pill--warn"} self-start`}
        >
          <span
            className={`lp-status-dot ${failed ? "lp-status-dot--err" : "lp-status-dot--warn"}`}
          />
          {failed ? "pipeline failed" : "running pipeline"}
        </span>
        {manifest && <PipelineProgress manifest={manifest} />}
        {failed && manifest?.errors?.length ? (
          <pre className="max-h-32 overflow-auto rounded-lg border border-accent-500/40 bg-accent-500/10 p-3 font-mono text-[11px] text-ink-200">
            {manifest.errors[manifest.errors.length - 1]}
          </pre>
        ) : (
          <p className="text-center text-xs text-ink-500">
            The point cloud will appear as soon as reconstruction completes —
            segmentation continues in the background.
          </p>
        )}
      </div>
    </div>
  );
}

function SegmentingBanner({ status }: { status: StageStatus }) {
  const label =
    status === "running"
      ? "Segmentation in progress — annotations will appear shortly."
      : "Annotations not yet generated. Segmentation pending.";
  return (
    <DismissibleBanner anchor="right" tone="warn">
      <span className="lp-banner-dot" />
      <div className="lp-banner-body">
        <span>{label}</span>
      </div>
    </DismissibleBanner>
  );
}

function FailedBanner({
  title,
  detail,
}: {
  title: string;
  detail?: string;
}) {
  return (
    <DismissibleBanner anchor="right" tone="err">
      <span className="lp-banner-dot" />
      <div className="lp-banner-body">
        <span className="lp-banner-title">{title}</span>
        {detail && <span className="lp-banner-detail">{detail}</span>}
      </div>
    </DismissibleBanner>
  );
}

function DismissibleBanner({
  anchor,
  tone,
  children,
}: {
  anchor: "left" | "right";
  tone?: "warn" | "err";
  children: ReactNode;
}) {
  const [hidden, setHidden] = useState(false);
  if (hidden) return null;
  const toneCls = tone === "warn"
    ? "lp-banner--warn"
    : tone === "err"
      ? "lp-banner--err"
      : "";
  return (
    <div
      className={[
        "pointer-events-none absolute top-3",
        anchor === "right" ? "right-3" : "left-3",
      ].join(" ")}
    >
      <div className={`lp-banner ${toneCls}`}>
        {children}
        <button
          type="button"
          onClick={() => setHidden(true)}
          className="lp-banner-close"
          title="Dismiss"
          aria-label="Dismiss"
        >
          ×
        </button>
      </div>
    </div>
  );
}
