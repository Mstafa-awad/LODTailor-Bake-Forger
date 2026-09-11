# LODTailor: Bake Forger

A focused **high-to-low texture baking node for ComfyUI**.

Bake Forger takes an existing high-poly GLB and an already-prepared low-poly GLB with UVs, runs Blender in the background, and transfers the high-poly surface information onto the low-poly target.

It is built for the part of the asset pipeline that happens **after** your low-poly mesh is ready: baking clean game-ready texture maps and exporting the baked GLB.

## Why Bake Forger exists

Bake Forger was **enhanced and substantially reworked from the baking portion of** the MIT-licensed `hp_to_lp_bake.py` from [mdj128/aeon-unity-tools](https://github.com/mdj128/aeon-unity-tools/blob/main/hp_to_lp_bake.py).

The original script was a useful proof of the baking workflow, but it was built as a Blender Text Editor script with fixed assumptions. Bake Forger turns that baking stage into a configurable ComfyUI node and focuses heavily on the failure cases that matter in an automated high-to-low pipeline.

The upstream MIT license and attribution are preserved in `licenses/AEON-UNITY-TOOLS-MIT.txt`.

## What was enhanced

### Adaptive cage and ray coverage

Bake Forger can inspect the spatial relationship between the high-poly and low-poly surfaces before baking. It uses a surface-coverage probe to decide whether the normal tight bake reach is enough and can automatically expand the allowed reach when more coverage is needed.

### Smart fallback baking

The fallback system supports three modes:

- `AUTO` — probe first and use the wider fallback only when useful.
- `ALWAYS` — always use the two-pass recovery workflow.
- `NEVER` — use only the tight bake.

This makes the node much less dependent on one hard-coded cage distance.

### Missing-texel detection and repair

Instead of simply accepting whatever Blender leaves in the texture, Bake Forger analyzes the resulting map and identifies texels that are not covered by a valid bake.

When a fallback pass is available, missing areas are recovered from that pass. Remaining uncovered areas are filled with semantic defaults rather than being left as obvious black or invalid data.

Normal, roughness, AO, emission, and base color each have their own handling.

### Material-aware base-color baking

Metallic surfaces can cause a diffuse-color bake to come out unexpectedly dark or black. Bake Forger temporarily disables the high-poly material's metallic contribution while the base-color bake runs, then restores the original material inputs and links afterward.

The source material is not permanently rewritten by this operation.

### Multiple map types

Bake Forger can independently bake/export:

- Base Color
- Tangent Normal
- Roughness
- Emission
- Ambient Occlusion

Each map has its own resolution control, so you can avoid spending the same amount of memory and bake time on every texture.

### Better texture encoding

The node treats maps according to their intended data type:

- Base Color uses sRGB handling.
- Normal uses non-color data and a neutral normal fallback.
- Roughness uses non-color data and a configurable material roughness fallback.
- AO and emission get suitable semantic defaults.
- Automatic bake margins scale with the requested texture resolution.

### GPU / CPU baking control

Blender's Cycles backend can try supported GPU compute backends and can optionally include the CPU in a hybrid setup. When no usable GPU device is detected, the worker falls back to CPU baking.

### ComfyUI-native 3D input handling

Bake Forger accepts ComfyUI `FILE_3D_GLB` inputs and contains handling for normal filesystem paths, File3D-style objects, byte streams, and save/export methods.

The Blender work is performed in a separate background process, keeping the actual bake outside the main ComfyUI Python process.

### More useful outputs

The node returns the baked GLB plus the individual image maps and a structured info string containing the bake result and warnings.

## Simple workflow

```text
        HIGH-POLY GLB
              │
              │
              ├───────────────┐
              │               │
              ▼               ▼
        High-poly source   Low-poly target
                            + UVs already ready
              │               │
              └───────┬───────┘
                      ▼
             LODTailor: Bake Forger
                      │
        ┌─────────────┼──────────────┐
        ▼             ▼              ▼
     Textures       Baked GLB     Bake info
```

### Important

Bake Forger is a **baking tool**, not the low-poly generator.

Your low-poly input should already be:

- the topology you want to ship
- positioned correctly against the high-poly source
- UV unwrapped

The high-poly mesh supplies the detail. The low-poly mesh receives it.

## Texture outputs

The node exposes separate outputs for:

- `base_color`
- `normal`
- `roughness`
- `emission`
- `ao`
- `baked_glb`
- `info`

Maps can be individually enabled or disabled.

## Installation

Place the folder in:

```text
ComfyUI/custom_nodes/LODTailor-Bake-Forger
```

Restart ComfyUI and search for:

**LOD Tailor: Bake Forger**

Set `blender_path` to `blender` when Blender is available on your system PATH. Otherwise specify the full path to `blender.exe`.

## Requirements

- ComfyUI
- Blender installed separately
- PyTorch supplied by the ComfyUI environment
- NumPy supplied by the ComfyUI environment
- Pillow available in the ComfyUI Python environment

No additional pip package is required by the node package itself.

## Recommended starting settings

For a normal high-to-low asset bake:

```text
Prefer GPU Baking       = True
Hybrid CPU/GPU          = True
Fallback Bake Mode      = AUTO
Auto Bake Margin        = True
```

Then enable only the texture maps you actually need and choose their resolutions.

## License

**LODTailor: Bake Forger** is released under the **GNU GPL v3 or later**.

The project contains adapted/enhanced baking work originating from the MIT-licensed `hp_to_lp_bake.py` in Aeon Unity Tools. The upstream MIT attribution and license are preserved in the `licenses/` directory.

GPLv3 permits commercial use, modification, redistribution, and sale, subject to the GPL's terms. The upstream MIT component is also permissive for commercial use, provided its notice is preserved.

External software is not bundled with this repository. When you install or redistribute external components such as ComfyUI, Blender, PyTorch, NumPy, or Pillow, their respective licenses continue to apply to those components.

## Third-party licenses

- `licenses/AEON-UNITY-TOOLS-MIT.txt` — upstream MIT license and attribution
- `licenses/NUMPY-BSD-3-CLAUSE.txt` — NumPy notice
- `licenses/PYTORCH-BSD-3-CLAUSE.txt` — PyTorch notice
- `licenses/PILLOW-MIT-CMU.txt` — Pillow notice
- `licenses/COMFYUI-GPL-3.0-NOTICE.txt` — ComfyUI notice
- `NOTICE.txt` — concise project/third-party overview

## Credits

### Upstream baking reference

**Martin Johnson / mdj128 — Aeon Unity Tools**

`hp_to_lp_bake.py`

https://github.com/mdj128/aeon-unity-tools/blob/main/hp_to_lp_bake.py

Bake Forger uses that work as a reference/source for the original baking stage, then extends the workflow for ComfyUI automation, adaptive coverage, fallback recovery, map management, material handling, and production-oriented file I/O.
