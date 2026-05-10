"""Detector-agnostic track dataclasses shared between segmentation backends.

Lives outside any backend-specific module (`gdino.py`, …) so ``lift.py`` and
downstream consumers can import the type contract without forcing a particular
detector's heavy dependencies (transformers, …).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrackFrame:
    frame_id: str
    score: float                          # detector confidence at this frame
    bbox_2d: tuple[int, int, int, int]    # (x0, y0, x1, y1)


@dataclass
class Track:
    track_id: str
    frames: list[TrackFrame] = field(default_factory=list)
    text_prompt: str | None = None
    source: str = "text"
