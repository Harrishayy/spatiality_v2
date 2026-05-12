"""Bake demo_piece into a shippable bundle — no downsampling, nothing in git.

The demo scene is hosted entirely off-repo:

  - **Live demo**: full scene uploaded to a Cloudflare R2 (or any) public
    bucket. The deployed website routes ``/api/jobs/demo_piece`` and
    ``/artifacts/scenes/demo_piece/*`` to that bucket via the rewrites in
    ``web/next.config.mjs`` gated on ``NEXT_PUBLIC_DEMO_CDN_URL``.

  - **Local use**: reviewers download ``dist/demo_piece_full.zip``,
    extract it to ``backend/data/outputs/demo_piece/``, run uvicorn, and
    view the scene at full quality through the local FastAPI orchestrator.

This script produces both deliverables from the source scene at
``backend/data/outputs/demo_piece/``:

  - ``dist/demo_piece_full.zip``  — bundle for the local-download path.
  - ``dist/demo_piece_r2/``       — R2-ready directory (flat layout matching
                                    what the rewrites expect at the bucket
                                    root). Upload this whole directory to
                                    your bucket with `rclone copy`,
                                    `aws s3 sync`, or the R2 web console.

Nothing is written to ``web/public/`` — the live demo data does not live in
the repo. If you want the demo to work on a clone that has no R2 access,
extract the zip locally and run the FastAPI orchestrator.

Usage
-----
::

    python scripts/build_demo_scene.py
    python scripts/build_demo_scene.py --no-zip       # skip the 1.3 GB zip write
    python scripts/build_demo_scene.py --no-r2        # skip the R2 staging dir

Dependencies: numpy + Pillow (same set Stage 5 uses).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SRC_SCENE = REPO / "backend" / "data" / "outputs" / "demo_piece"
DIST_DIR = REPO / "dist"
DST_ZIP = DIST_DIR / "demo_piece_full.zip"
DST_R2 = DIST_DIR / "demo_piece_r2"


# Files the viewer actually fetches. Anything not listed here is
# pipeline-internal (discards, raw lane outputs, scout/track JSONs,
# checkpoints) and stays out of both deliverables.
_USER_FACING_FILES = (
    "manifest.json",
    "points.ply",
    "cameras.json",
    "annotations.b.json",
    "annotations.c.json",
    "annotations.b.discarded.json",
    "traversability.json",
    "traversability.png",
)
_USER_FACING_DIRS = (
    "evidence",
    "masks",
)


def _trim_manifest_text(src_manifest: Path) -> str:
    """Return a trimmed manifest JSON string — viewer-relevant artefacts only.

    Strips artefact entries for files we don't ship in the bundle (raw lane
    outputs, scout phrases, track lists) so the viewer doesn't try to GET
    them and 404 against the bucket / local server.
    """
    m = json.loads(src_manifest.read_text())
    keep = {
        "splat_ply", "cameras_json", "annotations_json",
        "annotations_b_json", "annotations_c_json",
        "traversability_json", "traversability_png",
        "thumbnail_jpg",
    }
    a = m.get("artifacts", {}) or {}
    m["artifacts"] = {k: v for k, v in a.items() if k in keep}
    m["artifacts"].setdefault("splat_ply", "points.ply")
    return json.dumps(m, indent=2)


def _ensure_stage5() -> None:
    """Run Stage 5 on the source scene if traversability.json isn't present.

    The committed demo_piece pre-dates Stage 5, so this nearly always runs.
    Idempotent — compute_freespace overwrites cleanly.
    """
    if (
        (SRC_SCENE / "traversability.json").exists()
        and (SRC_SCENE / "traversability.png").exists()
    ):
        return
    print("[demo] Stage 5 artefacts missing in source; running compute_freespace …")
    sys.path.insert(0, str(REPO / "backend" / "src"))
    from spatiality.nav.freespace import compute_freespace  # noqa: E402

    compute_freespace(SRC_SCENE)


def _iter_payload() -> list[tuple[Path, str]]:
    """List of (source_path, archive_relative_path) for the payload.

    Single source of truth for what goes into both the zip and the R2 dir.
    Archive paths inside the zip are prefixed with ``demo_piece/`` so the
    unzip lands as ``backend/data/outputs/demo_piece/...`` cleanly; for R2
    the same files are placed at the bucket root (no ``demo_piece/`` prefix).
    """
    items: list[tuple[Path, str]] = []
    for name in _USER_FACING_FILES:
        src = SRC_SCENE / name
        if src.exists():
            items.append((src, name))
    for d in _USER_FACING_DIRS:
        src_dir = SRC_SCENE / d
        if not src_dir.is_dir():
            continue
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                items.append((p, p.relative_to(SRC_SCENE).as_posix()))
    return items


def _build_zip(items: list[tuple[Path, str]], trimmed_manifest: str) -> None:
    """Write dist/demo_piece_full.zip.

    Stored uncompressed — the 1.3 GB PLY is already incompressible binary
    (float32 + uint8) and DEFLATE on a file that size is multi-minute
    cost for ~0 gain. ZIP_STORED writes at disk speed.
    """
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if DST_ZIP.exists():
        DST_ZIP.unlink()

    print(f"[demo] writing {DST_ZIP} (stored / uncompressed) …")
    t0 = time.time()
    with zipfile.ZipFile(DST_ZIP, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for src, rel in items:
            if rel == "manifest.json":
                # Use the trimmed manifest, not the raw one — keeps
                # reviewers' viewer from chasing artefacts that don't ship.
                zf.writestr(f"demo_piece/{rel}", trimmed_manifest)
            else:
                zf.write(src, arcname=f"demo_piece/{rel}")
    size_mb = DST_ZIP.stat().st_size / 1e6
    print(f"[demo]   wrote {DST_ZIP} ({size_mb:,.0f} MB in {time.time()-t0:.1f}s)")


def _build_r2_dir(items: list[tuple[Path, str]], trimmed_manifest: str) -> None:
    """Stage every R2 artefact under dist/demo_piece_r2/ in bucket layout.

    The directory mirrors exactly what the bucket root should contain, so
    a single `rclone copy dist/demo_piece_r2/ r2:<bucket>/` (or
    `aws s3 sync ... s3://...`) ships the whole demo.
    """
    if DST_R2.exists():
        shutil.rmtree(DST_R2)
    DST_R2.mkdir(parents=True, exist_ok=True)

    print(f"[demo] staging R2 upload directory at {DST_R2}")
    t0 = time.time()
    for src, rel in items:
        dst = DST_R2 / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel == "manifest.json":
            dst.write_text(trimmed_manifest)
        else:
            # Hardlink when possible so we don't double the 1.3 GB PLY on
            # disk. Falls back to copy across filesystems.
            try:
                if dst.exists():
                    dst.unlink()
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
    total_mb = sum(p.stat().st_size for p in DST_R2.rglob("*") if p.is_file()) / 1e6
    print(f"[demo]   staged {total_mb:,.0f} MB in {time.time()-t0:.1f}s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-zip", action="store_true",
                   help="Skip building dist/demo_piece_full.zip (saves a multi-minute write).")
    p.add_argument("--no-r2", action="store_true",
                   help="Skip staging dist/demo_piece_r2/ for the CDN upload.")
    args = p.parse_args()

    if not SRC_SCENE.exists():
        print(f"error: no source scene at {SRC_SCENE}", file=sys.stderr)
        return 2

    _ensure_stage5()

    items = _iter_payload()
    trimmed_manifest = _trim_manifest_text(SRC_SCENE / "manifest.json")
    total_mb = sum(src.stat().st_size for src, _ in items) / 1e6
    print(f"[demo] payload: {len(items)} files, {total_mb:,.0f} MB")

    if not args.no_r2:
        _build_r2_dir(items, trimmed_manifest)
    if not args.no_zip:
        _build_zip(items, trimmed_manifest)

    print(
        "\n[demo] DONE.\n\n"
        "Upload to Cloudflare R2 (live demo):\n"
        "  1. Create a public R2 bucket.\n"
        f"  2. Sync the staged directory to the bucket root:\n"
        f"        rclone copy {DST_R2}/ r2:<bucket>/\n"
        "     (or `aws s3 sync` against the R2 S3-compatible endpoint).\n"
        "  3. Note the bucket's public URL (e.g. https://<id>.r2.dev).\n"
        "  4. In your Vercel project settings, set\n"
        "        NEXT_PUBLIC_DEMO_CDN_URL=https://<id>.r2.dev\n"
        "     for Production + Preview. The rewrites in next.config.mjs\n"
        "     will route demo_piece fetches to the bucket.\n\n"
        "Attach the local-download zip to a GitHub Release:\n"
        f"  - File: {DST_ZIP}\n"
        "  - Paste the release URL into README → 'Download the full demo scene'.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
