# 🌀 ComfyUI-Trellis2-GGUF — Local ROCm Fork (0xra0)

> **Base:** [Aero-Ex/ComfyUI-Trellis2-GGUF](https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF)  
> **Branch:** `local-rocm` — AMD ROCm / gfx1030 (RX 6800) patches on top of upstream.

ComfyUI custom node wrapping [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) with GGUF quantization support.  
This fork adds ROCm-specific stability fixes, Pixal3D MultiView projected-conditioning, safe bake, sparse VAE fixes, and a cascade coordinate scaling fix — all targeting 16 GB VRAM on AMD gfx1030.

---

## What's Different from Upstream

### Commit: `local-rocm-20261015` — Local ROCm fixes

Large patch covering the full ROCm compatibility layer:

| File | Change |
|---|---|
| `__init__.py` | ROCm-safe initialisation guards |
| `model_manager.py` | GGUF loader fixes for ROCm device placement |
| `nodes.py` | Pixal3D MultiView loader node; ROCm-safe node registrations |
| `rocm_safe_bake.py` | *(new)* Safe texture bake path that avoids ROCm allocator crashes |
| `trellis2_gguf/leak_probe.py` | *(new)* Debug utility to probe VRAM leak sources |
| `trellis2_gguf/models/__init__.py` | Model registry fixes |
| `trellis2_gguf/models/sc_vaes/fdg_vae.py` | FDG VAE dtype/device fixes |
| `trellis2_gguf/models/sc_vaes/sparse_unet_vae.py` | Sparse UNet VAE stability on ROCm |
| `trellis2_gguf/modules/attention/full_attn.py` | SDPA fallback for ROCm (no Flash Attention) |
| `trellis2_gguf/modules/sparse/attention/full_attn.py` | Sparse full attention ROCm fix |
| `trellis2_gguf/modules/sparse/attention/proj_attention.py` | Fix `proj_context=None` / `ProjectAttention` crash |
| `trellis2_gguf/modules/sparse/conv/conv_flex_gemm.py` | FlexGEMM conv ROCm path |
| `trellis2_gguf/modules/sparse/spatial/spatial2channel.py` | Spatial-to-channel op fix |
| `trellis2_gguf/pipelines/samplers/flow_euler.py` | Euler sampler stability |
| `trellis2_gguf/pipelines/trellis2_image_to_3d.py` | Pixal3D MultiView camera-projected conditioning; NAF tiled path; sparse-only feature tensors |
| `trellis2_gguf/utils/uv_rasterizer.py` | *(new)* UV rasterizer utility |

### Commit: `f90452a` — Cascade coordinate scaling fix

Fixes 1024/1536/2048/4096 cascade pipelines on Pixal3D:

> Pixal3D's sparse structure is 64³, so upsampled LR coords span 0..1023, but the cascade divided them by a hardcoded `lr_resolution=512`. Every HR coord doubled past the 64³ lattice (max 126), and the dual-grid mesher found almost no faces (182k vertices, 488 faces → a 150-face fragment after cleanup). Fix: use the already-computed `sparse_ratio` and pass `sparse_structure_resolution` from `run()`.

**Unstaged additions** (also active in the working tree):

- `nodes.py` — additional MultiView node fixes
- `trellis2_gguf/leak_probe.py` — extended VRAM leak probe
- `trellis2_gguf/pipelines/trellis2_image_to_3d.py` — further MultiView conditioning tweaks

---

## Hardware Target

| Component | Spec |
|---|---|
| GPU | AMD Radeon RX 6800 (gfx1030) |
| VRAM | 16 GB |
| ROCm | 6.x / 7.x |
| Python | 3.12 |
| PyTorch | ROCm build |

---

## Installation

### Prerequisites

- ComfyUI installed and running (see [0xra0/ComfyUI](https://github.com/0xra0/ComfyUI) for the matching ROCm-tuned base)
- ROCm 6+ drivers
- Python 3.12 environment used for ComfyUI

### 1. Clone into custom_nodes

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/0xra0/ComfyUI-Trellis2-GGUF.git
cd ComfyUI-Trellis2-GGUF
git checkout local-rocm
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

> [!NOTE]
> This node requires the TRELLIS.2 dependencies. On ROCm, Flash Attention is not available — the node falls back to SDPA automatically.

### 3. Download models

**GGUF Pixal3D weights** (place under `ComfyUI/models/Pixal3D-GGUF/`):
- Pixal3D-GGUF Q8_0 weights (from the upstream node's model page)

**DINOv3** (required by Trellis2):
```bash
# Clone into ComfyUI/models/facebook/
git clone https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m \
  /path/to/ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m
```

---

## Usage

### Pixal3D MultiView (ROCm-fixed path)

This fork's key addition is the **camera-projected MultiView conditioning** for Pixal3D-GGUF.

Recommended node settings:

| Setting | Value |
|---|---|
| Model | Pixal3D-GGUF |
| Format | GGUF Q8\_0 |
| Backend | `sdpa` |
| Device | `cuda` (ROCm) |
| `low_vram` | `true` |
| Views | front, back, left, right |
| `pipeline_type` | `512` |
| `sparse_structure_resolution` | `32` |
| Sparse steps | 4 |
| Shape steps | 4 |

Expected log markers confirming the fixed path is active:
```
[Pixal3D-GGUF MV] Using camera-projected conditioning
[Pixal3D MV] views=['front', 'left', 'back', 'right']
[Sparse] occupied tokens after shell=...
```

> [!TIP]
> Keep `generate_texture_slat = false` on first tests to save VRAM. Enable once sparse generation looks correct.

### Safe Bake

`rocm_safe_bake.py` provides a texture bake path that avoids the ROCm allocator crashes seen with the upstream bake on gfx1030. It is called automatically when baking is requested on a ROCm device.

---

## Rebasing onto New Upstream

This branch is designed to stay rebaseable:

```bash
git remote add upstream https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF.git
git fetch upstream
git rebase upstream/main
```

The commit message notes: *"Rebase onto new upstream with: `git pull` (pull.rebase=true)."*

---

## Relationship to Upstream

This is a personal hardware-specific fork of [Aero-Ex/ComfyUI-Trellis2-GGUF](https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF).  
Bug reports for the ROCm patches belong here. Everything else belongs upstream.

---

## License

Same license as upstream — see [LICENSE](LICENSE) if present, otherwise the upstream repository's license applies.
