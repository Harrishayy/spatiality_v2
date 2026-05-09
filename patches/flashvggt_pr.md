# FlashVGGT — fix package discovery so subpackages get installed

## Problem

`pip install git+https://github.com/wzpscott/FlashVGGT.git` reports a successful install of `flashvggt-0.1.0`, but at runtime:

```python
>>> import flashvggt.models.flashvggt
ModuleNotFoundError: No module named 'flashvggt.models'
```

Inspecting the installed wheel shows only the empty top-level `flashvggt/` and `flashvggt_stream/` directories — none of `models/`, `utils/`, `heads/`, `layers/`, `dependency/` (and their nested `track_modules/`) are packaged.

## Root cause

`pyproject.toml`:

```toml
[tool.setuptools.packages.find]
where = ["."]
exclude = ["outputs*", "ckpts*"]
include = ["flashvggt", "flashvggt_stream"]
```

Two bugs combine to skip everything:

1. **`include` uses literal names, not globs.** `find_packages(include=["flashvggt"])` matches the `flashvggt` package only, not `flashvggt.models`, `flashvggt.utils`, etc. Need `["flashvggt*", "flashvggt_stream*"]`.
2. **No `__init__.py` files.** `flashvggt/` and every subdir contains source files but no `__init__.py`, so setuptools' default `find_packages` (which requires `__init__.py`) discovers nothing. Need namespace-package discovery (`namespaces = true`).

## Verification

With both fixes applied, `setuptools.find_namespace_packages(include=["flashvggt*", "flashvggt_stream*"])` returns all 16 expected packages:

```
flashvggt
flashvggt.dependency
flashvggt.dependency.track_modules
flashvggt.heads
flashvggt.heads.track_modules
flashvggt.layers
flashvggt.models
flashvggt.utils
flashvggt_stream
flashvggt_stream.dependency
flashvggt_stream.dependency.track_modules
flashvggt_stream.heads
flashvggt_stream.heads.track_modules
flashvggt_stream.layers
flashvggt_stream.models
flashvggt_stream.utils
```

After the patch, `import flashvggt.models.flashvggt` succeeds and downstream code paths (e.g. `flashvggt.utils.pose_enc.pose_encoding_to_extri_intri`) resolve correctly.

## Patch

See `patches/flashvggt_minimal.patch`. Three-line diff:

```diff
[tool.setuptools.packages.find]
+namespaces = true
 where = ["."]
 exclude = [
-    "outputs*", 
+    "outputs*",
     "ckpts*",
 ]
-include = ["flashvggt", "flashvggt_stream"]
+include = ["flashvggt*", "flashvggt_stream*"]
```

(The `outputs*` line edit is just trailing-whitespace cleanup — drop it if you want the PR to touch even less.)

## To submit

```bash
gh repo fork wzpscott/FlashVGGT --clone=true --remote=true
cd FlashVGGT
git checkout -b fix/package-discovery
git apply /path/to/spatiality_v2/patches/flashvggt_minimal.patch
git commit -am "Fix package discovery: enable namespace packages + glob include"
git push -u origin fix/package-discovery
gh pr create --title "Fix package discovery so subpackages get installed" \
  --body-file /path/to/spatiality_v2/patches/flashvggt_pr.md
```
