"use client";

import { useEffect, useRef, useState } from "react";

import { useUI } from "@/store/ui";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function CalibrateModal({ open, onClose }: Props) {
  const measurements = useUI((s) => s.measurements);
  const displayScale = useUI((s) => s.displayScale);
  const calibrate = useUI((s) => s.calibrateFromLastMeasurement);

  const last = measurements[measurements.length - 1] ?? null;
  const currentDisplayed = last ? last.distance * displayScale : 0;

  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Reset + focus when the modal opens.
  useEffect(() => {
    if (!open) return;
    setError(null);
    setValue(last ? currentDisplayed.toFixed(2) : "");
    const t = setTimeout(() => inputRef.current?.focus(), 30);
    return () => clearTimeout(t);
  }, [open, last, currentDisplayed]);

  // Esc to close.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const submit = () => {
    if (!last) return;
    const real = parseFloat(value);
    if (!Number.isFinite(real) || real <= 0) {
      setError("Enter a positive number in metres (e.g. 0.8 for an 80 cm door).");
      return;
    }
    calibrate(real);
    onClose();
  };

  return (
    <div
      className="lp-modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="lp-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="calibrate-title"
      >
        <div className="lp-modal-head">
          <div>
            <h2 id="calibrate-title" className="lp-modal-title">
              Calibrate <em>scale</em>
            </h2>
            {last ? (
              <p className="lp-modal-sub">
                Tell us the real-world length of the segment you just measured.
                Every dimension in the viewer will rescale to match.
              </p>
            ) : (
              <p className="lp-modal-sub">
                Take a measurement first — pick two points on something of
                known size (a doorway is ~2.0 m tall; an interior door is
                ~80 cm wide), then come back here.
              </p>
            )}
          </div>
          <button
            type="button"
            className="lp-drawer-close"
            onClick={onClose}
            aria-label="Close"
            title="Close"
          >
            ×
          </button>
        </div>

        {last ? (
          <div className="lp-modal-body">
            <div className="lp-modal-row">
              <span className="lp-modal-label">Currently shown</span>
              <p className="lp-modal-hint">
                {currentDisplayed.toFixed(3)} m
                {displayScale !== 1 && (
                  <> · current scale ×{displayScale.toFixed(3)}</>
                )}
              </p>
            </div>
            <div className="lp-modal-row">
              <label className="lp-modal-label" htmlFor="calibrate-input">
                Real-world length (metres)
              </label>
              <input
                ref={inputRef}
                id="calibrate-input"
                className="lp-modal-input"
                type="number"
                inputMode="decimal"
                step="0.01"
                min="0"
                value={value}
                onChange={(e) => {
                  setValue(e.target.value);
                  if (error) setError(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") submit();
                }}
                placeholder="e.g. 0.80"
              />
            </div>
            {error && <p className="lp-modal-error">{error}</p>}
          </div>
        ) : (
          <div className="lp-modal-body">
            <p className="lp-modal-hint">
              Click <strong>Measure</strong> in the toolbar, click two points
              on the cloud, then re-open Calibrate.
            </p>
          </div>
        )}

        <div className="lp-modal-foot">
          <button
            type="button"
            className="lp-btn lp-btn-ghost lp-btn-sm"
            onClick={onClose}
          >
            Cancel
          </button>
          {last && (
            <button
              type="button"
              className="lp-btn lp-btn-primary lp-btn-sm"
              onClick={submit}
            >
              Apply scale
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
