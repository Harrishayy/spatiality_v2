"""CLI driver for the spatiality_v2 pipeline.

Usage:
    .venv/bin/python scripts/run_pipeline_cli.py <scene_id> [--frames 500]

Assumes ``backend/data/inputs/<scene_id>/source.<ext>`` already exists.
Reuses the helpers in ``backend.main`` so the path matches the FastAPI
orchestrator exactly (ffmpeg extract -> push -> inference -> pull poses ->
segmentation -> pull all).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_id")
    parser.add_argument("--frames", type=int, default=500)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.main import (  # noqa: E402
        _bump_manifest,
        _extract_frames,
        _pull_outputs_from_modal,
        _push_inputs_to_modal,
        _recompute_stats,
        _scene_input_dir,
        _scene_output_dir,
        _seed_manifest,
    )

    scene_id = args.scene_id
    n_frames = args.frames

    in_dir = _scene_input_dir(scene_id)
    if not any(in_dir.glob("source.*")):
        print(f"error: no source.* file in {in_dir}", file=sys.stderr)
        return 2

    t0 = time.time()
    print(f"[cli] scene_id={scene_id} frames={n_frames}")
    print(f"[cli] seed manifest")
    _seed_manifest(scene_id)

    print(f"[cli] extract frames ({n_frames})")
    info = _extract_frames(scene_id, n_frames)
    print(f"[cli]   frames={info['frame_count']} duration={info['duration_s']:.1f}s")

    print(f"[cli] push inputs volume")
    _push_inputs_to_modal(scene_id)

    import modal  # noqa: E402

    print(f"[cli] modal: spatiality-inference.run_inference_one")
    _bump_manifest(scene_id, "poses", "running")
    infer_fn = modal.Function.from_name("spatiality-inference", "run_inference_one")
    res = infer_fn.remote(scene_id, frames_max=n_frames)
    print(f"[cli]   inference result: {res}")

    print(f"[cli] pull poses artifacts")
    _pull_outputs_from_modal(scene_id, exclude={"manifest.json"})
    _recompute_stats(scene_id)

    print(f"[cli] modal: spatiality-segmentation.run_segmentation_one")
    _bump_manifest(scene_id, "segmentation", "running")
    seg_fn = modal.Function.from_name("spatiality-segmentation", "run_segmentation_one")
    res = seg_fn.remote(scene_id)
    print(f"[cli]   segmentation result: {res}")

    print(f"[cli] final pull")
    _pull_outputs_from_modal(scene_id)
    _recompute_stats(scene_id)

    manifest_path = _scene_output_dir(scene_id) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    print(f"[cli] done in {time.time() - t0:.1f}s")
    print(f"[cli] manifest:")
    print(json.dumps(manifest, indent=2))

    out_dir = _scene_output_dir(scene_id)
    print(f"[cli] artifacts in {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.name:32s} {size_mb:8.2f} MB")
        else:
            print(f"  {p.name}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
