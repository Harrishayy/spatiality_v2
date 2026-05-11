"""Re-run only the segmentation stage for an existing scene.

Inference / FlashVGGT artefacts are assumed to already live on the
spatiality-outputs Modal volume (poses stage already completed). This
driver just invokes ``spatiality-segmentation.run_segmentation_one`` and
pulls results back.

Usage:
    .venv/bin/python scripts/run_segmentation_cli.py <scene_id>
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
    args = parser.parse_args()
    scene_id = args.scene_id

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.main import (  # noqa: E402
        _bump_manifest,
        _pull_outputs_from_modal,
        _recompute_stats,
        _scene_output_dir,
    )

    import modal  # noqa: E402

    t0 = time.time()
    print(f"[cli] scene_id={scene_id} (segmentation re-run)", flush=True)
    _bump_manifest(scene_id, "segmentation", "running")

    print(f"[cli] modal: spatiality-segmentation.run_segmentation_one", flush=True)
    seg_fn = modal.Function.from_name("spatiality-segmentation", "run_segmentation_one")
    res = seg_fn.remote(scene_id)
    print(f"[cli]   segmentation result: {res}", flush=True)

    print(f"[cli] pull outputs", flush=True)
    _pull_outputs_from_modal(scene_id)
    _recompute_stats(scene_id)

    manifest_path = _scene_output_dir(scene_id) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    print(f"[cli] done in {time.time() - t0:.1f}s", flush=True)
    print(f"[cli] manifest:", flush=True)
    print(json.dumps(manifest, indent=2))

    out_dir = _scene_output_dir(scene_id)
    print(f"[cli] artifacts in {out_dir}:", flush=True)
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.name:32s} {size_mb:8.2f} MB", flush=True)
        else:
            print(f"  {p.name}/", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
