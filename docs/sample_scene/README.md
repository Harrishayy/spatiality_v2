# Sample scene

This directory is reserved for a small pre-computed scene that a reviewer can drop into `backend/data/outputs/<scene_id>/` and view in the web UI **without** Modal access or a GPU.

To keep the git repo small, the actual artefacts are hosted off-repo. Drop them in once you have a public URL and update this file.

## Recommended layout (when populated)

```
docs/sample_scene/
  manifest.json           # trimmed pipeline state
  points.ply              # downsampled to ~500k points (~15–20 MB)
  cameras.json            # full poses
  annotations.c.json      # final labels
  annotations.b.json      # raw Lane B labels (fallback)
  traversability.json     # Stage 5 occupancy grid
  traversability.png      # Stage 5 top-down preview
```

Stick to ≤ 25 MB total so this stays cloneable on a phone tether. If you keep a full 50 M-point cloud, host it externally and point the README at the URL.

## How a reviewer uses it (no Modal needed)

```bash
# copy the sample artefacts into the output dir the orchestrator expects
mkdir -p backend/data/outputs/sample_room
cp docs/sample_scene/* backend/data/outputs/sample_room/

# the FastAPI orchestrator serves any directory it finds under outputs/
uvicorn backend.main:app --port 8765 &

cd web && pnpm dev
# then open http://localhost:3000/scenes/sample_room
```

The web viewer reads the manifest directly; no upload, no Modal call, no Gemini key required to *view* a pre-computed scene.

## How to generate one

After running a scene end-to-end:

```bash
# downsample points.ply to ~500k points (use a tool of your choice;
# Open3D `voxel_down_sample` at 2–4 cm produces clean results)
python -c "
import open3d as o3d
p = o3d.io.read_point_cloud('backend/data/outputs/<id>/points.ply')
p = p.voxel_down_sample(voxel_size=0.03)
o3d.io.write_point_cloud('docs/sample_scene/points.ply', p, write_ascii=False)
"

cp backend/data/outputs/<id>/{cameras,annotations.c,annotations.b,traversability}.json docs/sample_scene/
cp backend/data/outputs/<id>/traversability.png docs/sample_scene/
```

Then trim `manifest.json` to point at these filenames and commit. Total payload should land between 15 and 25 MB.
