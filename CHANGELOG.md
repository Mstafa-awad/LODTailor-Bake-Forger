# Changelog

All notable changes to the LODTailor Bake Forger project will be documented in this file.

## [v1.2.0] - Unreleased

### ⚡ Performance / Speed Optimizations

- Optimized the bake pipeline to reduce unnecessary repeated work.
- Combined the previous two separate probe/BVH passes into one shared probe pass:
  - high-poly BVH is built once instead of twice,
  - low-poly surface sampling is done once instead of twice,
  - adaptive reach expansion and extended-ray fallback decisions now reuse the same probe data.
- Reduced Blender startup/background overhead by disabling unnecessary editor features during background baking where possible.
- Subprocess output is now written to a log file instead of being fully captured in memory, reducing Python-side overhead during long bakes.
- Base color brightness boost can now be applied during final array processing, avoiding an extra full-image read/write pass in some cases.
- Image write path now avoids unnecessary array duplication when the pixel buffer is already contiguous float32.

### 🧠 Memory / OOM Reduction

- Reworked the bake flow to process maps sequentially instead of allocating all texture maps at once.
- Only the currently baking map and its optional fallback image are kept alive during a bake pass.
- Temporary bake images are removed immediately after their PNG output is saved.
- Final GLB export material is rebuilt from saved PNG files instead of keeping all high-memory float buffers alive until export.
- Reduced peak NumPy memory usage during coverage checks and texel finalization.
- Normal-map coverage validation is now processed in chunks, avoiding large temporary vector arrays for full 4K/8K/16K images.
- Fallback merge operations release intermediate arrays earlier and clean up fallback images immediately after merging.
- Orphan mesh/material/image datablocks are purged after imports and before export.
- High-poly object data is deleted before final low-poly GLB export to reduce memory pressure during export.
- Added optional low-memory ComfyUI output behavior via `load_output_images`:
  - when enabled, baked PNGs are loaded into IMAGE outputs as before,
  - when disabled, IMAGE outputs use lightweight placeholders while the baked PNG/GLB files remain full quality on disk.

### 💾 Storage / Output Handling

- Bake output is now placed under a more visible ComfyUI output location instead of an anonymous system temp folder, making cleanup easier.
- Added support for a storage-safe fixed output workflow:
  - output can be written to a standardized `LODTailorBakeForger/latest` folder,
  - previous bake results can be cleared before the next bake,
  - this prevents many orphaned bake folders from accumulating and filling the SSD.
- Recommended setup for limited-storage systems:
  - keep only the latest bake result,
  - manually copy/save bake results if you want to preserve them,
  - disable unused map outputs if disk space is extremely limited.

### 🔒 Quality Preservation

- No bake-quality parameters were altered by the optimization pass.
- Texture resolution remains fully user-controlled.
- Bake sample count remains unchanged.
- AO minimum sample behavior remains unchanged.
- Cage extrusion, max ray distance, fallback mode, probe thresholds, coverage threshold, and normal tolerance remain unchanged.
- Color-space handling remains the same:
  - Normal, Roughness, and Metallic remain Non-Color data maps.
  - Base Color, Emission, and AO remain sRGB color maps.
- PNG output remains the same final baked output format.
- The optimizations target memory lifecycle, cleanup, and execution efficiency only; they do not intentionally reduce visual quality.

### ⚠️ Behavior Notes

- If using the fixed `latest` output folder mode, previous bake outputs will be replaced by the newest bake.
- If `load_output_images` is disabled, the IMAGE sockets return placeholder tensors, but the saved PNG/GLB files remain full quality.
- For systems with very limited storage, keeping all 8192 maps enabled can still generate large files per bake.

## [v1.1.0] - 2026-09-19

### 🚀 New Features

- Full PBR Metallic Baking Support: The node now fully supports baking the Metallic channel, completing the standard PBR Metal-Roughness texture set required for game engines (Unreal Engine, Unity, etc.).
- New Node Inputs:
  - `bake_metallic_texture` (Boolean): Toggle to enable/disable metallic baking (Default: True).
  - `metallic_resolution` (Integer): Set the resolution for the metallic map (Default: 8192).
  - `material_metallic` (Float): Fallback fill value for uncovered UV texels (Default: 0.0).
- New Node Output: Added a dedicated `metallic` IMAGE output socket to the ComfyUI node.

### 🛠️ Technical Improvements & Under-the-Hood Fixes

- The "Emission Swap" Trick: Because Blender Cycles lacks a native `METALLIC` bake pass, the background script now uses a clever workaround. It temporarily reroutes the high-poly's Metallic input into an Emission shader, runs a standard `EMIT` bake pass to capture the data, and then flawlessly restores the original material nodes.
- Correct Color Space & Buffer: The metallic texture is now allocated with a 32-bit float buffer and "Non-Color" color space (matching Roughness and Normal maps) to prevent gamma clipping and ensure raw data accuracy.
- Low-Poly Material Wiring: The baked metallic map is now automatically linked directly to the `Metallic` input socket of the low-poly's Principled BSDF shader inside the exported GLB.
- Coverage & Fallback Handling: Updated the internal semantic map and texel-filling logic to correctly identify and fill the Metallic channel during two-pass fallback bakes.

### 📦 Export & Pipeline Updates

- The metallic map is now saved as `<model_name>_metallic.png` in the temporary output directory.
- The metallic PNG is loaded as a PyTorch tensor and correctly returned to the ComfyUI workflow for downstream masking or previewing.
- Updated `result.json` payload to include the `metallic_png` file path.