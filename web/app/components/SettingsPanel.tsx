import type { JobSettings } from "@/lib/types";

export type { JobSettings };

export const DEFAULT_SETTINGS: JobSettings = {
  max_frames: 400,
  target_long_side: 1920,
  segment: true,
  keyframes: 5,
  vlm_model: "gemini-2.5-flash",
};
