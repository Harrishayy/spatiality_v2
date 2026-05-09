"use client";

import { useId } from "react";

import type { JobSettings } from "@/lib/types";

export type { JobSettings };

export const DEFAULT_SETTINGS: JobSettings = {
  max_frames: 400,
  target_long_side: 1920,
  segment: true,
  keyframes: 5,
  vlm_model: "gemini-2.5-flash",
};

interface Props {
  value: JobSettings;
  onChange: (next: JobSettings) => void;
  durationS?: number;
}

export function SettingsPanel({ value, onChange, durationS: _durationS }: Props) {
  const set = <K extends keyof JobSettings>(k: K, v: JobSettings[K]) =>
    onChange({ ...value, [k]: v });

  return (
    <section className="lp-surface min-w-0">
      <header className="lp-surface-head">
        <div>
          <h2 className="lp-surface-title">Pipeline settings</h2>
          <p className="lp-surface-sub">Tune capture and segmentation before queueing.</p>
        </div>
        <span className="lp-eyebrow-mono">≤ {value.max_frames} frames</span>
      </header>

      <Slider
        label="Max frames total"
        min={50}
        max={1000}
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
        <Field label="Labeler">
          <div className="lp-segmented-row lp-segmented-row--on" aria-disabled>
            <span>Gemini 2.5 Flash</span>
            <span className="lp-segmented-meta">VLM</span>
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
