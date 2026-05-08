"use client";

import { useId } from "react";

import { VLM_MODEL_OPTIONS, type JobSettings, type VlmModelId } from "@/lib/types";

export type { JobSettings };

export const DEFAULT_SETTINGS: JobSettings = {
  fps: 2.0,
  max_frames: 400,
  target_long_side: 1920,
  segment: true,
  keyframes: 5,
  vlm_model: "claude-haiku-4-5",
};

interface Props {
  value: JobSettings;
  onChange: (next: JobSettings) => void;
  durationS?: number;
}

export function SettingsPanel({ value, onChange, durationS }: Props) {
  const set = <K extends keyof JobSettings>(k: K, v: JobSettings[K]) =>
    onChange({ ...value, [k]: v });

  const projectedFrames = durationS
    ? Math.min(Math.ceil(durationS * value.fps), value.max_frames)
    : null;

  return (
    <section className="lp-surface">
      <header className="lp-surface-head">
        <div>
          <h2 className="lp-surface-title">Pipeline settings</h2>
          <p className="lp-surface-sub">Tune capture and segmentation before queueing.</p>
        </div>
        {projectedFrames != null && (
          <span className="lp-eyebrow-mono">≈ {projectedFrames} frames</span>
        )}
      </header>

      <Slider
        label="Frames per second"
        suffix="fps"
        min={0.5}
        max={10}
        step={0.5}
        value={value.fps}
        onChange={(v) => set("fps", v)}
      />
      <Slider
        label="Max frames total"
        min={50}
        max={800}
        step={10}
        value={value.max_frames}
        onChange={(v) => set("max_frames", v)}
      />
      <Slider
        label="Target long side"
        suffix="px"
        min={720}
        max={3840}
        step={20}
        value={value.target_long_side}
        onChange={(v) => set("target_long_side", v)}
      />

      <Field
        label="Run segmentation"
        help="Detect each object in the scene and label it."
      >
        <button
          type="button"
          onClick={() => set("segment", !value.segment)}
          className={`lp-segmented-row ${value.segment ? "lp-segmented-row--on" : ""}`}
        >
          <span>{value.segment ? "Enabled" : "Disabled"}</span>
          <span className="lp-segmented-meta">{value.segment ? "on" : "off"}</span>
        </button>
      </Field>

      {value.segment && (
        <Slider
          label="Segmentation keyframes"
          min={2}
          max={20}
          step={1}
          value={value.keyframes}
          onChange={(v) => set("keyframes", v)}
        />
      )}

      {value.segment && (
        <Field
          label="Labeler"
          help="Cost is per scene."
        >
          <div className="lp-segmented">
            {VLM_MODEL_OPTIONS.map((opt) => {
              const active = (value.vlm_model ?? "claude-haiku-4-5") === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => set("vlm_model", opt.id as VlmModelId)}
                  className={`lp-segmented-row ${active ? "lp-segmented-row--on" : ""}`}
                >
                  <span>{opt.label}</span>
                  <span className="lp-segmented-meta">{opt.cost}</span>
                </button>
              );
            })}
          </div>
        </Field>
      )}
    </section>
  );
}

function Field({
  label,
  help,
  children,
}: {
  label: string;
  help?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="lp-field">
      <div className="lp-field-row">
        <span className="lp-field-label">{label}</span>
      </div>
      {help && <span className="lp-field-help">{help}</span>}
      {children}
    </div>
  );
}

function Slider({
  label,
  min,
  max,
  step,
  value,
  onChange,
  suffix,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  suffix?: string;
}) {
  const id = useId();
  return (
    <label htmlFor={id} className="lp-field">
      <div className="lp-field-row">
        <span className="lp-field-label">{label}</span>
        <span className="lp-field-value">
          {value}
          {suffix ? ` ${suffix}` : ""}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="lp-range"
      />
    </label>
  );
}
