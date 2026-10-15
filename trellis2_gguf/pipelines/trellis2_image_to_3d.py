from typing import *
import torch

# Detect AMD GPU for NAF fallback
is_amd_gpu = torch.cuda.is_available() and 'AMD' in torch.cuda.get_device_name(0)
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline, _find_sdnq_model_dir
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel

from .. import models

import gc
import os
import folder_paths
import trimesh
import o_voxel
import cumesh
import nvdiffrast.torch as dr
import cv2
import flex_gemm
from flex_gemm.ops.grid_sample import grid_sample_3d

import random
import math

from comfy.utils import ProgressBar

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pixal3d_rocm_limit_hr_coords(
    coords: torch.Tensor,
    grid_res: int,
    max_tokens: int,
) -> torch.Tensor:
    """
    Deterministically reduce an oversized Pixal3D HR sparse lattice on ROCm.

    The 1024 shape flow uses full sparse self-attention.  On gfx1030 the
    42k-token lattice seen in the real MV run is far outside the token range
    that has completed reliably on this machine (~8k-10k).  Keep surface
    voxels first, then fill any remaining budget with spatially distributed
    interior voxels.  The returned coordinates remain unique and ordered.

    This is a correctness guard, not the normal path: it is only called for
    Pixal3D MV + ROCm when the 1024 lattice exceeds the configured budget.
    """
    max_tokens = int(max_tokens)
    if max_tokens <= 0 or int(coords.shape[0]) <= max_tokens:
        return coords

    src_device = coords.device
    src_dtype = coords.dtype
    c = coords.detach().to(device='cpu', dtype=torch.int64)
    grid_res = int(grid_res)

    def _spread_select(rows: torch.Tensor, count: int) -> torch.Tensor:
        count = int(count)
        if count <= 0 or rows.numel() == 0:
            return rows[:0]
        if int(rows.shape[0]) <= count:
            return rows
        xyz = rows[:, 1:]
        key = xyz[:, 0] * (grid_res * grid_res) + xyz[:, 1] * grid_res + xyz[:, 2]
        order = torch.argsort(key, stable=True)
        rows = rows[order]
        idx = torch.linspace(0, rows.shape[0] - 1, steps=count).round().long()
        return rows[idx]

    batches = torch.unique(c[:, 0], sorted=True)
    total = int(c.shape[0])
    remaining = max_tokens
    selected = []
    shell_total = 0

    for bi, batch_id in enumerate(batches.tolist()):
        rows = c[c[:, 0] == batch_id]
        xyz = rows[:, 1:].clamp(0, grid_res - 1)

        occ = torch.zeros(
            (grid_res, grid_res, grid_res),
            dtype=torch.bool,
            device='cpu',
        )
        occ[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True

        interior_grid = torch.zeros_like(occ)
        if grid_res > 2:
            interior_grid[1:-1, 1:-1, 1:-1] = (
                occ[1:-1, 1:-1, 1:-1]
                & occ[:-2, 1:-1, 1:-1]
                & occ[2:, 1:-1, 1:-1]
                & occ[1:-1, :-2, 1:-1]
                & occ[1:-1, 2:, 1:-1]
                & occ[1:-1, 1:-1, :-2]
                & occ[1:-1, 1:-1, 2:]
            )

        is_shell = ~interior_grid[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        shell = rows[is_shell]
        inside = rows[~is_shell]
        shell_total += int(shell.shape[0])

        # Proportional batch budget; the last batch receives the remainder.
        if bi == len(batches) - 1:
            budget = remaining
        else:
            budget = max(1, round(max_tokens * int(rows.shape[0]) / total))
            budget = min(budget, remaining)
        remaining -= budget

        if int(shell.shape[0]) >= budget:
            chosen = _spread_select(shell, budget)
        else:
            fill = budget - int(shell.shape[0])
            chosen = torch.cat([shell, _spread_select(inside, fill)], dim=0)
        selected.append(chosen)

    out = torch.cat(selected, dim=0)
    # Stable canonical order keeps runs deterministic and makes diagnostics easy.
    key = (
        out[:, 0] * (grid_res ** 3)
        + out[:, 1] * (grid_res ** 2)
        + out[:, 2] * grid_res
        + out[:, 3]
    )
    out = out[torch.argsort(key, stable=True)]

    print(
        '[Pixal3D MV ROCm HR token guard] '
        f'grid={grid_res}^3, input={int(coords.shape[0]):,}, '
        f'surface_candidates={shell_total:,}, output={int(out.shape[0]):,}, '
        f'limit={max_tokens:,}'
    )

    return out.to(device=src_device, dtype=src_dtype)

import contextlib
@contextlib.contextmanager
def hide_src_module():
    import sys
    import os
    import torch
    
    # Patch valeoai_NAF_main to be a regular package so it takes precedence over other nodes
    hub_dir = torch.hub.get_dir()
    naf_dir = os.path.join(hub_dir, "valeoai_NAF_main")
    if os.path.exists(naf_dir):
        for d in ["src", "src/model"]:
            init_path = os.path.join(naf_dir, d, "__init__.py")
            if os.path.exists(os.path.join(naf_dir, d)) and not os.path.exists(init_path):
                open(init_path, 'w').close()

    old_src_modules = {}
    for k in list(sys.modules.keys()):
        if k == 'src' or k.startswith('src.'):
            old_src_modules[k] = sys.modules.pop(k)
    try:
        yield
    finally:
        for k, v in old_src_modules.items():
            sys.modules[k] = v


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    # model_names_to_load = [
        # 'sparse_structure_flow_model',
        # 'sparse_structure_decoder',
        # 'shape_slat_flow_model_512',
        # 'shape_slat_flow_model_1024',
        # 'shape_slat_decoder',
        # 'tex_slat_flow_model_512',
        # 'tex_slat_flow_model_1024',
        # 'tex_slat_decoder',
    # ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self._low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.precision = None
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'
        
        self.PIXAL3D_IMAGE_COND_CONFIGS = {
            "ss": {
                "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "image_size": 512,
                "grid_resolution": 16,
            },
            "shape_512": {
                "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "image_size": 512,
                "grid_resolution": 32,
                "use_naf_upsample": True,
                "naf_target_size": 512,
            },
            "shape_1024": {
                "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "image_size": 1024,
                "grid_resolution": 64,
                "use_naf_upsample": True,
                "naf_target_size": 512,
            },
            "tex_1024": {
                "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "image_size": 1024,
                "grid_resolution": 64,
                "use_naf_upsample": True,
                "naf_target_size": 1024,
            },
        }

        # Camera-projected Pixal3D MultiView conditioning.  The denoiser input
        # shape remains identical to single-view when fusion=average, so this
        # also works as a compatibility path with existing Pixal3D GGUF weights.
        self.PIXAL3D_MV_IMAGE_COND_CONFIGS = {
            key: {**cfg, "multiview_fusion": "average"}
            for key, cfg in self.PIXAL3D_IMAGE_COND_CONFIGS.items()
        }

    @property
    def low_vram(self) -> bool:
        return self._low_vram

    @low_vram.setter
    def low_vram(self, value: bool):
        self._low_vram = value
        for m in self.models.values():
            if hasattr(m, 'low_vram'):
                m.low_vram = value
        if hasattr(self, 'image_cond_model') and hasattr(self.image_cond_model, 'low_vram'):
            self.image_cond_model.low_vram = value

    def _cond_to(self, cond: dict, device: torch.device) -> dict:
        # Move only tensors; keep other items unchanged
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond.items()}

    def _cond_cpu(self, cond: dict) -> dict:
        return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in cond.items()}

    def _cleanup_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _get_sampler(self, sampler_name: Optional[str], default_sampler: samplers.Sampler, **kwargs) -> samplers.Sampler:
        """
        Get a sampler by name, or return the default sampler.
        """
        if sampler_name is None or sampler_name == "" or sampler_name == "default":
            return default_sampler
        
        # Determine the base class name based on the default sampler
        base_class = default_sampler.__class__.__name__
        
        # Map simple names to class prefixes
        name_map = {
            "euler": "FlowEuler",
            "heun": "FlowHeun",
            "rk4": "FlowRK4",
            "rk5": "FlowRK5",
        }
        prefix = name_map.get(sampler_name.lower(), "FlowEuler")
        
        # Construct the target class name by replacing the prefix
        # e.g., FlowEulerGuidanceIntervalSampler -> FlowRK4GuidanceIntervalSampler
        if base_class.startswith("FlowEuler"):
            target_class_name = base_class.replace("FlowEuler", prefix)
        elif base_class.startswith("FlowHeun"):
            target_class_name = base_class.replace("FlowHeun", prefix)
        elif base_class.startswith("FlowRK4"):
            target_class_name = base_class.replace("FlowRK4", prefix)
        elif base_class.startswith("FlowRK5"):
            target_class_name = base_class.replace("FlowRK5", prefix)
        else:
            # Fallback if the base class doesn't match expected patterns
            target_class_name = prefix + "Sampler"
            
        target_class = getattr(samplers, target_class_name, None)
        if target_class is None:
            print(f"[Trellis2] Warning: Sampler {target_class_name} not found, using default {base_class}")
            return default_sampler
            
        # Initialize the new sampler with parameters from the default one if possible
        # Most samplers in this repo take sigma_min in __init__
        try:
            init_args = {'sigma_min': getattr(default_sampler, 'sigma_min', 1e-3)}
            # If the default sampler has other attributes like resolution, copy them
            for attr in ['resolution']:
                if hasattr(default_sampler, attr):
                    init_args[attr] = getattr(default_sampler, attr)
            # Override with explicitly passed kwargs
            init_args.update(kwargs)
            return target_class(**init_args)
        except Exception as e:
            print(f"[Trellis2] Warning: Could not initialize {target_class_name}: {e}")
            return default_sampler

    def GetSamplerName(self, sampler_name: Optional[str]) -> str:
        """
        Helper to normalize sampler names.
        """
        if sampler_name is None or sampler_name == "" or sampler_name == "default":
            return "Euler"
        name_map = {
            "euler": "Euler",
            "heun": "Heun",
            "rk4": "RK4",
            "rk5": "RK5",
        }
        return name_map.get(sampler_name.lower(), "Euler")

    def move_all_to_cpu(self):
        """Move all models in the pipeline to CPU."""
        print("[Trellis2] Offloading all models to CPU...")
        for name, model in self.models.items():
            if model is not None:
                model.cpu()
        if self.image_cond_model is not None:
            self.image_cond_model.cpu()
        if self.rembg_model is not None:
            self.rembg_model.cpu()
        self._cleanup_cuda()

    def _sdnq_remap(self, path: str) -> str:
        """Remap a bf16 model path to its SDNQ directory if enable_sdnq is set."""
        if not getattr(self, 'enable_sdnq', False):
            return path
        sdnq_dir = _find_sdnq_model_dir(path, svd_rank=getattr(self, 'sdnq_svd_rank', 32))
        if sdnq_dir:
            print(f'[Trellis2-SDNQ] {os.path.basename(path)} → SDNQ: {os.path.basename(sdnq_dir)}')
            return sdnq_dir
        print(f'[Trellis2-SDNQ] WARNING: No SDNQ dir for {os.path.basename(path)}, using original')
        return path

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json", keep_models_loaded = True,
                        enable_gguf: bool = False, gguf_quant: str = "Q8_0", precision: str = None,
                        enable_sdnq: bool = False, sdnq_use_quantized_matmul: bool = True,
                        sdnq_torch_compile: bool = False,
                        sdnq_svd_rank: int = 32,
                        isPixal3D: bool = False,
                        isPixal3DMV: bool = False) -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        if isPixal3DMV:
            config_file = "pipeline_mv.json"

        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        #pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        #pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.image_cond_model = None
        pipeline.rembg_model = None
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'
        pipeline.path = path
        pipeline.keep_models_loaded = keep_models_loaded
        pipeline.last_processing = ''
        pipeline.enable_gguf = enable_gguf
        pipeline.gguf_quant = gguf_quant
        pipeline.precision = precision
        pipeline.enable_sdnq = enable_sdnq
        pipeline.sdnq_use_quantized_matmul = sdnq_use_quantized_matmul
        pipeline.sdnq_torch_compile = sdnq_torch_compile
        pipeline.sdnq_svd_rank = sdnq_svd_rank
        pipeline.isPixal3D = isPixal3D
        pipeline.isPixal3DMV = isPixal3DMV

        pipeline._pretrained_args['models']['sparse_structure_decoder'] = os.path.join(folder_paths.models_dir,"Trellis2","decoders","Stage1","ss_dec_conv3d_16l8_fp16")
        # Check both the new consolidated location and the old legacy location for DINOv3
        dinov3_new = os.path.join(folder_paths.models_dir,"Trellis2","dinov3","facebook","dinov3-vitl16-pretrain-lvd1689m")
        dinov3_old = os.path.join(folder_paths.models_dir,"Aero-Ex","Dinov3","facebook","dinov3-vitl16-pretrain-lvd1689m")
        dinov3_user = os.path.join(folder_paths.models_dir,"facebook","dinov3-vitl16-pretrain-lvd1689m")
        if os.path.exists(dinov3_new):
            facebook_model_path = dinov3_new
        elif os.path.exists(dinov3_user):
            facebook_model_path = dinov3_user
        else:
            facebook_model_path = dinov3_old
        pipeline._pretrained_args['image_cond_model']['args']['model_name'] = facebook_model_path
        
        for k in pipeline.PIXAL3D_IMAGE_COND_CONFIGS:
            pipeline.PIXAL3D_IMAGE_COND_CONFIGS[k]["model_name"] = facebook_model_path
        for k in pipeline.PIXAL3D_MV_IMAGE_COND_CONFIGS:
            pipeline.PIXAL3D_MV_IMAGE_COND_CONFIGS[k]["model_name"] = facebook_model_path

        return pipeline
        
    def load_moge_model(self):
        if hasattr(self,'moge_model') and self.moge_model is not None:
            return self.moge_model
        
        model_name = "Ruicheng/moge-2-vitl"
        moge_model_path = os.path.join(folder_paths.models_dir, "Ruicheng","moge-2-vitl")
        
        if not os.path.exists(moge_model_path):
            print(f"Downloading MoGe model to: {moge_model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_name,
                local_dir=moge_model_path,
                local_dir_use_symlinks=False,
            )
        
        moge_model_path = os.path.join(moge_model_path,'model.pt')
        
        print('Loading MoGe model ...')
        try:
            from moge.model.v2 import MoGeModel
        except ImportError:
            # Fallback for relative import if running inside the GGUF module
            import sys
            trellis2_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "ComfyUI-Trellis2"))
            if trellis2_path not in sys.path:
                sys.path.append(trellis2_path)
            from moge.model.v2 import MoGeModel
            
        self.moge_model = MoGeModel.from_pretrained(moge_model_path).to(self.device)
        self.moge_model.eval()

    def unload_moge_model(self):
        if hasattr(self,'moge_model') and self.moge_model is not None:
            del self.moge_model
            self.moge_model = None
            self._cleanup_cuda() 

    def get_moge_camera_config(self, image):
        try:
            from trellis2.utils.camera import get_camera_params_wild_moge
        except ImportError:
            import importlib.util
            import os
            camera_py_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "ComfyUI-Trellis2", "trellis2", "utils", "camera.py"))
            spec = importlib.util.spec_from_file_location("trellis2_camera", camera_py_path)
            trellis2_camera = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(trellis2_camera)
            get_camera_params_wild_moge = trellis2_camera.get_camera_params_wild_moge
            
        camera_config = get_camera_params_wild_moge(image, self.moge_model)
        print(camera_config)
        return camera_config

    def _get_build_pixal3d_image_cond_model(self):
        import sys
        for name, module in list(sys.modules.items()):
            if name.startswith('torch.') or name.startswith('torchvision.'): continue
            try:
                if hasattr(module, 'build_pixal3d_image_cond_model'):
                    func = getattr(module, 'build_pixal3d_image_cond_model')
                    if callable(func):
                        return func
            except Exception:
                pass
        raise ImportError("Could not find build_pixal3d_image_cond_model in loaded modules. Make sure ComfyUI-Trellis2 is updated and installed.")

    def _get_build_pixal3d_mv_image_cond_model(self):
        """Load the official Pixal3D multi-view projected DINO extractor.

        Reuse the implementation from the sibling ComfyUI-Trellis2 checkout.  The
        GGUF fork keeps the denoisers/loaders; only the camera-aware conditioning
        frontend comes from the upstream implementation.
        """
        import sys
        import inspect

        def _import_cls():
            from trellis2.trainers.flow_matching.mixins.image_conditioned_proj import (
                DinoV3ProjMultiViewFeatureExtractor,
            )
            return DinoV3ProjMultiViewFeatureExtractor

        try:
            cls = _import_cls()
        except Exception:
            custom_nodes = os.path.abspath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'
            ))
            sibling = os.path.join(custom_nodes, 'ComfyUI-Trellis2')
            if os.path.isdir(sibling) and sibling not in sys.path:
                sys.path.insert(0, sibling)
            try:
                cls = _import_cls()
            except Exception as e:
                raise ImportError(
                    'Pixal3D MultiView conditioning requires an up-to-date '
                    'ComfyUI-Trellis2 sibling checkout containing '
                    'DinoV3ProjMultiViewFeatureExtractor.'
                ) from e

        def _build(config):
            model = cls(**config)
            model.eval()

            # v4.12: never silently run the dedicated Pixal3D MV checkpoints
            # against a stale projected-DINO implementation.  That failure is
            # particularly dangerous because every tensor remains finite while
            # geometry collapses into disconnected fragments.
            try:
                cls_file = os.path.realpath(inspect.getfile(cls))
            except Exception:
                cls_file = "<unknown>"
            try:
                extract_src = inspect.getsource(cls.extract_features)
            except Exception:
                extract_src = ""

            native_official_norm = (
                "_pixal3d_official_final_norm" in extract_src
                and getattr(model, "_pixal3d_official_final_norm", False)
            )
            print(
                "[Pixal3D MV DINO VERIFY] source=" + cls_file
                + "; official_parameterless_norm="
                + ("ACTIVE" if native_official_norm else "MISSING")
            )
            if not native_official_norm:
                raise RuntimeError(
                    "[Pixal3D MV DINO VERIFY] stale/incomplete ComfyUI-Trellis2 "
                    "projected-DINO source is loaded. The official MV checkpoints "
                    "require parameterless F.layer_norm, but the loaded extractor "
                    "does not advertise the v4.11+ official-normalization path. "
                    f"Loaded class source: {cls_file}. Install the bundled v4.12 "
                    "ComfyUI-Trellis2 file before running generation."
                )
            return model

        return _build

    def _load_pixal3d_mv_image_cond(self, stage: str):
        attr = f'pixal3d_mv_image_cond_{stage}'
        model = getattr(self, attr, None)
        if model is not None:
            return model
        print(f'[Pixal3D MV] Loading projected image conditioner: {stage}')
        build = self._get_build_pixal3d_mv_image_cond_model()
        model = build(self.PIXAL3D_MV_IMAGE_COND_CONFIGS[stage])
        setattr(self, attr, model)
        return model

    def _unload_pixal3d_mv_image_cond(self, stage: str):
        attr = f'pixal3d_mv_image_cond_{stage}'
        model = getattr(self, attr, None)
        if model is not None:
            try:
                model.cpu()
            except Exception:
                pass
            setattr(self, attr, None)
            self._cleanup_cuda()

    def load_pixal3d_mv_image_cond_ss(self):
        return self._load_pixal3d_mv_image_cond('ss')

    def load_pixal3d_mv_image_cond_shape_512(self):
        return self._load_pixal3d_mv_image_cond('shape_512')

    def load_pixal3d_mv_image_cond_shape_1024(self):
        return self._load_pixal3d_mv_image_cond('shape_1024')

    def load_pixal3d_mv_image_cond_tex_1024(self):
        return self._load_pixal3d_mv_image_cond('tex_1024')

    def unload_pixal3d_mv_image_cond_ss(self):
        self._unload_pixal3d_mv_image_cond('ss')

    def unload_pixal3d_mv_image_cond_shape_512(self):
        self._unload_pixal3d_mv_image_cond('shape_512')

    def unload_pixal3d_mv_image_cond_shape_1024(self):
        self._unload_pixal3d_mv_image_cond('shape_1024')

    def unload_pixal3d_mv_image_cond_tex_1024(self):
        self._unload_pixal3d_mv_image_cond('tex_1024')

    @staticmethod
    def _pixal3d_c2w(azimuth_deg: float, elevation_deg: float, distance: float):
        """Blender/NeRF c2w convention used by Pixal3D training renders."""
        az = math.radians(float(azimuth_deg))
        el = math.radians(float(elevation_deg))
        eye = np.array([
            math.cos(el) * math.sin(az),
            -math.cos(el) * math.cos(az),
            math.sin(el),
        ], dtype=np.float32) * float(distance)
        back = eye / (np.linalg.norm(eye) + 1e-12)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(back, world_up))) > 0.999:
            world_up = np.array([0.0, 1.0 if back[2] > 0 else -1.0, 0.0], dtype=np.float32)
        right = np.cross(world_up, back)
        right = right / (np.linalg.norm(right) + 1e-12)
        up = np.cross(back, right)
        up = up / (np.linalg.norm(up) + 1e-12)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = back
        c2w[:3, 3] = eye
        return c2w

    @staticmethod
    def _pixal3d_mv_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
        """Resize and alpha-premultiply one view without per-view cropping."""
        im = image.convert('RGBA').resize(
            (int(image_size), int(image_size)), Image.Resampling.LANCZOS
        )
        arr = np.asarray(im, dtype=np.float32) / 255.0
        rgb = torch.from_numpy(arr[..., :3]).permute(2, 0, 1).float()
        alpha = torch.from_numpy(arr[..., 3]).float().unsqueeze(0)
        return rgb * alpha

    def _build_pixal3d_mv_views(
        self,
        front: Image.Image,
        back: Image.Image = None,
        left: Image.Image = None,
        right: Image.Image = None,
        camera_angle_x_deg: float = 20.0,
        camera_distance: float = 3.119205,
        mesh_scale: float = 1.0,
    ) -> dict:
        """Build the camera bundle expected by official Pixal3D MV.

        Official Pixal3D defines azim000 at (0,-d,0), azim090 at
        (+d,0,0), azim180 at (0,+d,0), and azim270 at (-d,0,0).
        For conventional front/back/left/right view names this maps the
        RIGHT-side camera to +X/90 and the LEFT-side camera to -X/270.
        Earlier versions accidentally assigned those two side images to the
        opposite camera transforms, which poisons camera-projected fusion.
        """
        _legacy_lr = (
            os.environ.get(
                "TRELLIS2_PIXAL3D_MV_LEGACY_LR_CAMERA",
                "0",
            ) == "1"
        )
        if _legacy_lr:
            entries = [('front', front, 0.0)]
            if left is not None:
                entries.append(('left', left, 90.0))
            if back is not None:
                entries.append(('back', back, 180.0))
            if right is not None:
                entries.append(('right', right, 270.0))
            print(
                "[Pixal3D MV camera] LEGACY side mapping: "
                "left=az90(+X), right=az270(-X)"
            )
        else:
            entries = [('front', front, 0.0)]
            if right is not None:
                entries.append(('right', right, 90.0))
            if back is not None:
                entries.append(('back', back, 180.0))
            if left is not None:
                entries.append(('left', left, 270.0))
            print(
                "[Pixal3D MV camera] official/conventional side mapping: "
                "right=az90(+X), left=az270(-X)"
            )

        names = [x[0] for x in entries]
        images = [x[1] for x in entries]
        azimuths = [x[2] for x in entries]
        V = len(entries)
        if V < 2:
            raise ValueError('Pixal3D MultiView needs at least two views.')

        # ---------------------------------------------------------
        # Uncalibrated-render camera correction (v4.12).
        #
        # Official Pixal3D MV takes real per-view c2w transforms and
        # explicitly supports a different camera radius per view.  A
        # set of independently rendered / generated turntable PNGs
        # often has the same canvas size but NOT the same framing.
        # Feeding those views with one hard-coded camera radius makes
        # ProjGridMV sample different physical points in each view and
        # their averaged projected features cancel one another.
        #
        # Do not rescale the pixels.  Instead:
        #   1. translate the RGBA silhouette so its alpha bbox centre
        #      is at the principal point;
        #   2. keep the front view as the metric anchor;
        #   3. infer each other camera radius from the ratio of alpha
        #      bbox heights (height is invariant under pure yaw for a
        #      rigid upright object).
        #
        # d_i = d_front * h_front / h_i
        #
        # This is disabled for genuinely calibrated official datasets
        # and enabled explicitly for loose front/back/left/right PNGs.
        # ---------------------------------------------------------
        _auto_calibrate = (
            os.environ.get(
                "TRELLIS2_PIXAL3D_MV_AUTO_CALIBRATE", "0"
            ) == "1"
        )

        _view_distances = [float(camera_distance)] * V

        if _auto_calibrate:
            _alpha_stats = []
            for _name, _im in zip(names, images):
                _rgba = _im.convert("RGBA")
                _alpha = np.asarray(
                    _rgba.getchannel("A"), dtype=np.uint8
                )
                _ys, _xs = np.nonzero(_alpha > 128)
                if len(_xs) == 0:
                    raise RuntimeError(
                        "[Pixal3D MV auto-calibrate] "
                        f"{_name} has an empty alpha mask"
                    )
                _x0, _x1 = int(_xs.min()), int(_xs.max())
                _y0, _y1 = int(_ys.min()), int(_ys.max())
                _cx = 0.5 * (_x0 + _x1)
                _cy = 0.5 * (_y0 + _y1)
                _alpha_stats.append({
                    "bbox": (_x0, _y0, _x1, _y1),
                    "height": float(_y1 - _y0 + 1),
                    "cx": _cx,
                    "cy": _cy,
                })

            # View 0 is always the front/main view.  Keep its radius
            # unchanged so calc_mat_0 remains the official anchor.
            _front_h = _alpha_stats[0]["height"]
            if _front_h <= 0:
                raise RuntimeError(
                    "[Pixal3D MV auto-calibrate] invalid front height"
                )

            _calibrated_images = []
            _calibrated_distances = []

            for _name, _im, _st in zip(
                names, images, _alpha_stats
            ):
                _rgba = _im.convert("RGBA")
                _target_cx = 0.5 * (_rgba.width - 1)
                _target_cy = 0.5 * (_rgba.height - 1)
                _dx = int(round(_target_cx - _st["cx"]))
                _dy = int(round(_target_cy - _st["cy"]))

                if _dx != 0 or _dy != 0:
                    _centered = Image.new(
                        "RGBA", _rgba.size, (0, 0, 0, 0)
                    )
                    _centered.alpha_composite(
                        _rgba, dest=(_dx, _dy)
                    )
                else:
                    _centered = _rgba

                _distance_i = (
                    float(camera_distance)
                    * _front_h
                    / max(_st["height"], 1.0)
                )

                # Reject implausible calibration rather than silently
                # creating a wildly different camera rig.
                _ratio = _distance_i / float(camera_distance)
                if not (0.70 <= _ratio <= 1.30):
                    raise RuntimeError(
                        "[Pixal3D MV auto-calibrate] "
                        f"{_name} inferred distance ratio "
                        f"{_ratio:.4f} is outside [0.70, 1.30]. "
                        "The supplied views are not compatible with "
                        "a cardinal turntable camera rig."
                    )

                _calibrated_images.append(_centered)
                _calibrated_distances.append(_distance_i)

                print(
                    "[Pixal3D MV auto-calibrate] "
                    f"{_name:5s}: bbox={_st['bbox']}, "
                    f"height={_st['height']:.1f}, "
                    f"shift=({_dx:+d},{_dy:+d}), "
                    f"distance={_distance_i:.6f}"
                )

            images = _calibrated_images
            _view_distances = _calibrated_distances
            print(
                "[Pixal3D MV auto-calibrate] front anchored at "
                f"distance={float(camera_distance):.6f}, "
                f"height={_front_h:.1f}; pixels were translated "
                "only, never rescaled"
            )

        c2w = np.stack([
            self._pixal3d_c2w(a, 0.0, d)
            for a, d in zip(azimuths, _view_distances)
        ], axis=0)
        transforms = torch.from_numpy(c2w)[None]
        fov = math.radians(float(camera_angle_x_deg))
        camera_angle_x = torch.full((1, V), fov, dtype=torch.float32)
        camera_distance_t = torch.norm(transforms[:, :, :3, 3], dim=-1)
        bundle_images = {
            size: torch.stack([
                self._pixal3d_mv_tensor(im, size) for im in images
            ], dim=0)[None]
            for size in (512, 1024)
        }

        _distance_desc = ",".join(
            f"{n}:{float(d):.4f}"
            for n, d in zip(names, _view_distances)
        )
        print(
            f"[Pixal3D MV] views={names}, fov={camera_angle_x_deg:.2f}deg, "
            f"distances=[{_distance_desc}], mesh_scale={mesh_scale:.3f}"
        )
        return {
            'images': bundle_images,
            'camera_angle_x': camera_angle_x,
            'camera_distance': camera_distance_t,
            'transform_matrix': transforms,
            'mesh_scale': float(mesh_scale),
            'view_names': names,
        }

    @staticmethod
    def _mv_image_for(views: dict, image_cond_model: nn.Module) -> torch.Tensor:
        size = int(image_cond_model.image_size)
        if size not in views['images']:
            raise KeyError(f'No Pixal3D MV images at {size}px')
        return views['images'][size]

    def _run_mv_extractor(self, image_cond_model: nn.Module, views: dict):
        device = self.device
        image = self._mv_image_for(views, image_cond_model).to(device)
        cam_angle = views['camera_angle_x'].to(device=device, dtype=torch.float32)
        dist = views['camera_distance'].to(device=device, dtype=torch.float32)
        tm = views['transform_matrix'].to(device=device, dtype=torch.float32)
        scale = torch.as_tensor(
            views['mesh_scale'], dtype=torch.float32, device=device
        ).reshape(-1)
        with hide_src_module():
            return image_cond_model(
                image,
                camera_angle_x=cam_angle,
                distance=dist,
                mesh_scale=scale,
                transform_matrix=tm,
            )

    @torch.no_grad()
    def get_proj_cond_ss_mv(self, views: dict, image_cond_model=None) -> dict:
        """Dense 16^3 projected condition for Pixal3D sparse structure."""
        if image_cond_model is None:
            image_cond_model = self.load_pixal3d_mv_image_cond_ss()
        image_cond_model.to(self.device)
        z_global, z_proj = self._run_mv_extractor(image_cond_model, views)
        if self.low_vram:
            image_cond_model.cpu()
        return {
            'cond': {'global': z_global, 'proj': z_proj},
            'neg_cond': {
                'global': torch.zeros_like(z_global),
                'proj': torch.zeros_like(z_proj),
            },
        }

    @torch.no_grad()
    def get_proj_cond_shape_mv(
        self,
        image_cond_model: nn.Module,
        views: dict,
        coords: torch.Tensor,
        grid_resolution_override: int = None,
    ) -> dict:
        """Sparse-only camera projection for Pixal3D MultiView shape/texture.

        Upstream projects a complete R^3 grid and gathers sparse rows afterwards.
        On 16 GB ROCm that creates a huge transient at R=64/96.  Here we temporarily
        replace ProjGridMV.grid_points with only the occupied sparse coordinates, so
        DINO/NAF project exactly N sparse points instead of R^3 points.  This is the
        multi-view equivalent of the existing single-view sparse-projection patch.
        """
        device = self.device
        image_cond_model.to(device)

        grid_res = int(
            grid_resolution_override
            if grid_resolution_override is not None
            else image_cond_model.grid_resolution
        )
        if grid_res <= 1:
            raise ValueError(f'Invalid Pixal3D MV grid resolution: {grid_res}')

        coords_dev = coords.to(device=device, dtype=torch.int32).contiguous()
        xyz = coords_dev[:, 1:4].float()
        xyz = xyz / float(grid_res - 1) * 2.0 - 1.0
        rotation = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float32, device=device)
        sparse_grid_points = xyz @ rotation.T

        proj_grid = image_cond_model.proj_grid
        old_grid_points = proj_grid.grid_points
        old_pg_res = getattr(proj_grid, 'grid_resolution', None)
        old_model_res = image_cond_model.grid_resolution
        old_tile = getattr(image_cond_model, 'naf_tile_factor', 1)

        # 4x4 NAF streaming works well on the user's 16 GB ROCm target and the
        # sparse point list makes the projected output scale with N, not R^3.
        if getattr(image_cond_model, 'use_naf_upsample', False):
            image_cond_model.naf_tile_factor = 4 if self.low_vram else 2

        try:
            proj_grid._buffers['grid_points'] = sparse_grid_points
            proj_grid.grid_resolution = grid_res
            image_cond_model.grid_resolution = grid_res
            z_global, z_proj = self._run_mv_extractor(image_cond_model, views)
        finally:
            proj_grid._buffers['grid_points'] = old_grid_points
            if old_pg_res is not None:
                proj_grid.grid_resolution = old_pg_res
            image_cond_model.grid_resolution = old_model_res
            image_cond_model.naf_tile_factor = old_tile

        if z_proj.ndim != 3 or z_proj.shape[0] != 1:
            raise RuntimeError(
                f'Unexpected Pixal3D MV projected feature shape: {tuple(z_proj.shape)}'
            )
        if z_proj.shape[1] != coords_dev.shape[0]:
            raise RuntimeError(
                'Pixal3D MV sparse projection count mismatch: '
                f'{z_proj.shape[1]} features for {coords_dev.shape[0]} coords.'
            )

        z_proj_sparse = z_proj[0].contiguous()
        z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords_dev)
        if self.low_vram:
            image_cond_model.cpu()

        return {
            'cond': {'global': z_global, 'proj': z_proj_st},
            'neg_cond': {
                'global': torch.zeros_like(z_global),
                'proj': SparseTensor(
                    feats=torch.zeros_like(z_proj_sparse),
                    coords=coords_dev,
                ),
            },
        }

    def get_proj_cond_ss(
        self,
        image: list,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
        image_cond_model = None
    ) -> dict:
        """
        Get proj conditioning for sparse structure stage.

        Args:
            image: List of PIL images.
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': ...}
        """
        print('Getting Proj Image Cond ...')
        device = self.device
        # Use MoGe to estimate camera params from the image when available
        try:
            self.load_moge_model()
            cfg = self.get_moge_camera_config(image[0] if isinstance(image, list) else image)
            camera_angle_x = cfg['camera_angle_x']
            distance = cfg['distance']
            mesh_scale = cfg['mesh_scale']
            if self.low_vram:
                self.unload_moge_model()
        except Exception as e:
            print(f"[MoGe] Falling back to defaults: {e}")
        # Unconditional: the pixal3d proj-cond models are built lazily, after the loader's
        # pipeline.to(device) has already run, so nothing else ever moves them. Under
        # low_vram=False they stayed on CPU while extract_features force-moved only the
        # DINOv3 backbone, leaving proj_grid on CPU -> "two devices, cuda:0 and cpu".
        image_cond_model.to(device)
        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        with hide_src_module():
            z_global, z_proj = image_cond_model(
                image, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor,
            )
        if self.low_vram:
            image_cond_model.cpu()
        return {
            'cond': {'global': z_global, 'proj': z_proj},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': torch.zeros_like(z_proj)},
        }

    @torch.no_grad()
    def get_proj_cond_shape(
        self,
        image_cond_model: nn.Module,
        image: list,
        coords: torch.Tensor,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
        grid_resolution_override: int = None,
    ) -> dict:
        """
        Get proj conditioning for shape/texture stages (sparse-token aligned).

        Args:
            image_cond_model: The proj image cond model for this stage.
            image: List of PIL images.
            coords: Sparse structure coordinates [N, 4] (batch_idx, x, y, z).
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.
            grid_resolution_override: Override the grid resolution if not None.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': SparseTensor}
        """
        # =====================================================
        # Official Pixal3D MV:
        #
        # IMPORTANT:
        # Do not use the generic dense get_proj_cond_shape path.
        #
        # get_proj_cond_shape_mv already projects ONLY the sparse
        # occupied coords and was successfully used earlier in
        # run_pixal3d_multiview.
        # =====================================================
        if (
            getattr(self, "isPixal3DMV", False)
            and isinstance(image, dict)
        ):
            print(
                "[Pixal3D MV get_proj_cond_shape] "
                "delegating to sparse MV conditioner"
            )

            return self.get_proj_cond_shape_mv(
                image_cond_model,
                image,
                coords,
            )

        # =====================================================
        # Official Pixal3D MultiView bundle.
        #
        # The MV pipeline passes a dictionary containing:
        #
        #   images[size]      -> [B,V,3,H,W]
        #   camera_angle_x    -> [B,V]
        #   camera_distance   -> [B,V]
        #   transform_matrix  -> [B,V,4,4]
        #   mesh_scale
        #
        # NEVER send this through the old single-view GGUF path:
        #   - no MoGe
        #   - no PIL-list conversion
        #   - no optimized_forward_instance 4D monkeypatch
        # =====================================================
        if (
            getattr(self, "isPixal3DMV", False)
            and isinstance(image, dict)
            and "images" in image
            and "transform_matrix" in image
        ):
            print(
                "[Pixal3D MV get_proj_cond_shape] "
                "using dedicated camera-aware MV path"
            )

            views = image
            device = self.device

            if self.low_vram:
                image_cond_model.to(device)

            # gfx1030: never activate the custom NAF tiler.
            if (
                getattr(torch.version, "hip", None)
                is not None
                and os.environ.get(
                    "TRELLIS2_MV_NAF_TILING",
                    "0",
                ) == "0"
            ):
                image_cond_model.naf_tile_factor = 1

            orig_grid_res = (
                image_cond_model.grid_resolution
            )

            override = (
                grid_resolution_override is not None
                and grid_resolution_override
                != orig_grid_res
            )

            try:
                if override:
                    image_cond_model.grid_resolution = (
                        int(grid_resolution_override)
                    )

                    image_cond_model.proj_grid = (
                        image_cond_model
                        .proj_grid
                        .__class__(
                            grid_resolution=int(
                                grid_resolution_override
                            ),
                            image_resolution=(
                                image_cond_model
                                .proj_grid
                                .image_resolution
                            ),
                        )
                        .to(device)
                    )

                target_size = int(
                    image_cond_model.image_size
                )

                image_map = views["images"]

                if target_size in image_map:
                    mv_image = image_map[target_size]
                elif str(target_size) in image_map:
                    mv_image = image_map[
                        str(target_size)
                    ]
                else:
                    raise KeyError(
                        "[Pixal3D MV] no conditioning image "
                        f"for size={target_size}; "
                        f"available={list(image_map.keys())}"
                    )

                mv_image = mv_image.to(
                    device=device,
                    dtype=torch.float32,
                )

                if mv_image.ndim != 5:
                    raise RuntimeError(
                        "[Pixal3D MV] expected images "
                        "[B,V,C,H,W], got "
                        f"{tuple(mv_image.shape)}"
                    )

                cam_angle = torch.as_tensor(
                    views.get(
                        "camera_angle_x",
                        camera_angle_x,
                    ),
                    dtype=torch.float32,
                    device=device,
                )

                mv_distance = views.get(
                    "camera_distance",
                    views.get(
                        "distance",
                        distance,
                    ),
                )

                dist_tensor = torch.as_tensor(
                    mv_distance,
                    dtype=torch.float32,
                    device=device,
                )

                scale_tensor = torch.as_tensor(
                    views.get(
                        "mesh_scale",
                        mesh_scale,
                    ),
                    dtype=torch.float32,
                    device=device,
                ).reshape(-1)

                transform_matrix = torch.as_tensor(
                    views["transform_matrix"],
                    dtype=torch.float32,
                    device=device,
                )

                print(
                    "[Pixal3D MV get_proj_cond_shape] "
                    f"image={tuple(mv_image.shape)}, "
                    f"grid={image_cond_model.grid_resolution}"
                )

                print(
                    "[Pixal3D MV get_proj_cond_shape] "
                    f"camera_angle_x="
                    f"{tuple(cam_angle.shape)}, "
                    f"distance={tuple(dist_tensor.shape)}, "
                    f"transform="
                    f"{tuple(transform_matrix.shape)}, "
                    f"mesh_scale="
                    f"{tuple(scale_tensor.shape)}"
                )

                # Call the actual
                # DinoV3ProjMultiViewFeatureExtractor.
                #
                # This is the same argument layout used by
                # official Pixal3D MV.
                with hide_src_module():
                    z_global, z_proj = (
                        image_cond_model(
                            mv_image,
                            camera_angle_x=cam_angle,
                            distance=dist_tensor,
                            mesh_scale=scale_tensor,
                            transform_matrix=(
                                transform_matrix
                            ),
                        )
                    )

                B = int(z_global.shape[0])
                grid_res = int(
                    image_cond_model.grid_resolution
                )

                expected_dense = (
                    B
                    * grid_res
                    * grid_res
                    * grid_res
                )

                if z_proj.numel() == 0:
                    raise RuntimeError(
                        "[Pixal3D MV] empty projected "
                        "conditioning"
                    )

                print(
                    "[Pixal3D MV get_proj_cond_shape] "
                    f"z_global={tuple(z_global.shape)}, "
                    f"z_proj={tuple(z_proj.shape)}, "
                    f"grid={grid_res}"
                )

                z_proj_grid = z_proj.reshape(
                    B,
                    grid_res,
                    grid_res,
                    grid_res,
                    -1,
                )

                coords_device = coords.to(
                    z_proj.device
                )

                b_idx = (
                    coords_device[:, 0]
                    .long()
                )
                x_idx = (
                    coords_device[:, 1]
                    .long()
                )
                y_idx = (
                    coords_device[:, 2]
                    .long()
                )
                z_idx = (
                    coords_device[:, 3]
                    .long()
                )

                xyz = coords_device[:, 1:]

                xyz_min = (
                    xyz.min(dim=0).values
                    .detach().cpu().tolist()
                )
                xyz_max = (
                    xyz.max(dim=0).values
                    .detach().cpu().tolist()
                )

                if (
                    int(xyz.min()) < 0
                    or int(xyz.max()) >= grid_res
                ):
                    raise RuntimeError(
                        "[Pixal3D MV] coords outside "
                        f"{grid_res}^3 grid: "
                        f"min={xyz_min}, max={xyz_max}"
                    )

                z_proj_sparse = z_proj_grid[
                    b_idx,
                    x_idx,
                    y_idx,
                    z_idx,
                ].to(device)

                z_proj_st = SparseTensor(
                    feats=z_proj_sparse,
                    coords=coords,
                )

                # Validate BEFORE returning this to the
                # 1024 flow model.
                for name, tensor in (
                    ("global", z_global),
                    (
                        "proj",
                        z_proj_sparse,
                    ),
                ):
                    finite = torch.isfinite(
                        tensor
                    )

                    print(
                        "[Pixal3D MV get_proj_cond_shape] "
                        f"{name}: "
                        f"shape={tuple(tensor.shape)}, "
                        f"dtype={tensor.dtype}, "
                        f"finite="
                        f"{int(finite.sum().item())}/"
                        f"{tensor.numel()}, "
                        f"min="
                        f"{float(tensor.min().item()):.6f}, "
                        f"max="
                        f"{float(tensor.max().item()):.6f}"
                    )

                    if not bool(
                        finite.all().item()
                    ):
                        raise RuntimeError(
                            "[Pixal3D MV] non-finite "
                            f"{name} conditioning"
                        )

                print(
                    "[Pixal3D MV get_proj_cond_shape] "
                    f"coords min={xyz_min}, "
                    f"max={xyz_max}"
                )

                cond_out = {
                    "cond": {
                        "global": z_global,
                        "proj": z_proj_st,
                    },
                    "neg_cond": {
                        "global": (
                            torch.zeros_like(
                                z_global
                            )
                        ),
                        "proj": SparseTensor(
                            feats=torch.zeros_like(
                                z_proj_sparse
                            ),
                            coords=coords,
                        ),
                    },
                }

            finally:
                if override:
                    image_cond_model.grid_resolution = (
                        orig_grid_res
                    )

                    image_cond_model.proj_grid = (
                        image_cond_model
                        .proj_grid
                        .__class__(
                            grid_resolution=(
                                orig_grid_res
                            ),
                            image_resolution=(
                                image_cond_model
                                .proj_grid
                                .image_resolution
                            ),
                        )
                        .to(device)
                    )

            if self.low_vram:
                image_cond_model.cpu()

            return cond_out

        print('Getting Projected Image Cond ...')
        device = self.device
        # Use MoGe to estimate camera params from the image when available
        try:
            self.load_moge_model()
            cfg = self.get_moge_camera_config(image[0] if isinstance(image, list) else image)
            camera_angle_x = cfg['camera_angle_x']
            distance = cfg['distance']
            mesh_scale = cfg['mesh_scale']
            if self.low_vram:
                self.unload_moge_model()
        except Exception as e:
            print(f"[MoGe] Falling back to defaults: {e}")
        # Unconditional move, for the same reason as get_proj_cond_ss: nothing else ever
        # moves the lazily-built proj-cond models onto the compute device. Offloading back
        # to CPU at the end of this function stays gated on low_vram.
        # (Both branches of the old texture-stage split did the same two things, so the
        # is_texture_stage test that selected between them has been dropped.)
        image_cond_model.to(device)
        image_cond_model.naf_tile_factor = 4

        orig_grid_res = image_cond_model.grid_resolution
        orig_proj_grid = image_cond_model.proj_grid
        target_grid_res = orig_grid_res if grid_resolution_override is None else grid_resolution_override

        # Project only the occupied voxels, not the full target_grid_res**3 dense grid.
        # ProjGrid returns one feature row per entry of its `grid_points` buffer, so swapping
        # that buffer for the subset addressed by `coords` gives exactly the rows the caller
        # gathers anyway - the dense grid is built and thrown away otherwise.
        # At 1536_cascade (96**3 = 884k points, 1024 ch) the dense path costs ~3.4 GB per
        # feature map, x2 for lr+hr and again for the cpu copy + concat: ~13.5 GB host RAM
        # and ~7 GB VRAM, which OOMs a 16 GB / 32 GB machine.
        coords_xyz = coords[:, 1:].long()
        max_coord = int(coords_xyz.max().item()) if coords_xyz.numel() else -1
        if max_coord >= target_grid_res:
            raise ValueError(
                f"coords contain index {max_coord} outside the {target_grid_res}**3 proj grid"
            )
        sub_proj_grid = orig_proj_grid.__class__(
            grid_resolution=target_grid_res,
            image_resolution=orig_proj_grid.image_resolution,
        )
        R = target_grid_res
        # Flat index must match the [R, R, R] meshgrid('ij') ordering ProjGrid builds.
        flat_idx = (coords_xyz[:, 0] * R + coords_xyz[:, 1]) * R + coords_xyz[:, 2]
        sub_proj_grid.grid_points = sub_proj_grid.grid_points[flat_idx.to(sub_proj_grid.grid_points.device)]
        print(f"  [proj cond] grid {R}^3: projecting {coords.shape[0]:,} occupied voxels "
              f"({coords.shape[0] / R ** 3 * 100:.1f}% of dense)")
        image_cond_model.grid_resolution = target_grid_res
        image_cond_model.proj_grid = sub_proj_grid.to(device)

        B = 1
        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        # monkeypatch the forward of this specific image_cond_model instance
        # or its class if it hasn't been monkeypatched yet!
        if not hasattr(image_cond_model, '_monkeypatched_for_cpu'):
            original_forward = image_cond_model.forward.__func__ if hasattr(image_cond_model.forward, '__func__') else image_cond_model.forward
            
            def optimized_forward_instance(self_instance, image, camera_angle_x=None, distance=None, mesh_scale=None, transform_matrix=None):
                if self_instance.grid_resolution < 32:
                    return original_forward(self_instance, image, camera_angle_x, distance, mesh_scale, transform_matrix)
                
                import torch
                import numpy as np
                from PIL import Image

                if isinstance(image, torch.Tensor):
                    assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
                elif isinstance(image, list):
                    assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
                    image = [i.resize((self_instance.image_size, self_instance.image_size), Image.LANCZOS) for i in image]
                    image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
                    image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
                    image = torch.stack(image).cuda()
                else:
                    raise ValueError(f"Unsupported type of image: {type(image)}")
                
                B = image.shape[0]
                
                if self_instance.use_naf_upsample:
                    image_for_naf = image.clone()
                
                image = self_instance.transform(image)
                
                with torch.no_grad():
                    z = self_instance.extract_features(image)
                    
                    z_clstoken = z[:, 0:1]
                    num_reg = getattr(self_instance.model.config, 'num_register_tokens', 4)
                    z_regtokens = z[:, 1:1+num_reg]
                    z_patchtokens = z[:, 1+num_reg:]
                    
                    z_patchtokens_spatial = z_patchtokens.reshape(
                        B, self_instance.patch_number, self_instance.patch_number, -1
                    )
                    
                    if camera_angle_x is None or distance is None or mesh_scale is None:
                        raise ValueError("camera_angle_x, distance, and mesh_scale must be provided")
                    
                    z_proj_lr = self_instance.proj_grid(
                        z_patchtokens_spatial, 
                        camera_angle_x, 
                        distance, 
                        mesh_scale,
                        transform_matrix
                    )
                    
                    if self_instance.use_naf_upsample:
                        self_instance._load_naf()
                        lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)

                        K = getattr(self_instance, 'naf_tile_factor', 2) or 2
                        if K <= 1:
                            hr_features = self_instance.naf_model(
                                image_for_naf, lr_features_bchw, self_instance.naf_target_size
                            )
                            z_proj_hr = self_instance.proj_grid(
                                hr_features,
                                camera_angle_x,
                                distance,
                                mesh_scale,
                                transform_matrix,
                                BHWC=False
                            )
                            del hr_features
                        else:
                            z_proj_hr = self_instance._proj_naf_tiled(
                                image_for_naf, lr_features_bchw,
                                camera_angle_x, distance, mesh_scale, transform_matrix,
                                tile_factor=int(K),
                            )

                        # Copy each half straight into the output buffer. Staging both
                        # halves on the host and then concatenating holds 2x the final
                        # tensor in RAM at the peak; this holds 1x plus one half.
                        n_dim = z_proj_lr.shape[-1]
                        z_proj = torch.empty(
                            *z_proj_lr.shape[:-1], n_dim + z_proj_hr.shape[-1],
                            dtype=z_proj_lr.dtype, device='cpu',
                        )
                        z_proj[..., :n_dim].copy_(z_proj_lr)
                        del z_proj_lr
                        z_proj[..., n_dim:].copy_(z_proj_hr)
                        del z_proj_hr
                        torch.cuda.empty_cache()
                    else:
                        z_proj = z_proj_lr.cpu()
                        del z_proj_lr
                        torch.cuda.empty_cache()
                    
                    global_features = torch.cat([z_clstoken, z_regtokens], dim=1)
                    return global_features, z_proj

            import types
            image_cond_model.forward = types.MethodType(optimized_forward_instance, image_cond_model)
            image_cond_model._monkeypatched_for_cpu = True
            print("[Trellis2-GGUF] Successfully applied instance monkeypatch to image_cond_model!")

        with hide_src_module():
            z_global, z_proj = image_cond_model(
                image, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor,
            )
        # z_proj is already sparse-aligned: row i is the feature for coords[i].
        if z_proj.shape[0] == 1:
            z_proj_sparse = z_proj[0].to(device)
        else:
            batch_indices = coords[:, 0].long().to(z_proj.device)
            row_indices = torch.arange(coords.shape[0], device=z_proj.device)
            z_proj_sparse = z_proj[batch_indices, row_indices].to(device)
        z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)
        del z_proj
        torch.cuda.empty_cache()

        image_cond_model.grid_resolution = orig_grid_res
        image_cond_model.proj_grid = orig_proj_grid

        if self.low_vram:
            image_cond_model.cpu()
        return {
            'cond': {'global': z_global, 'proj': z_proj_st},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords)},
        }

    def load_pixal3d_image_cond_ss(self):    
        if hasattr(self,'pixal3d_image_cond_ss') and self.pixal3d_image_cond_ss is not None:
            return self.pixal3d_image_cond_ss
        print('Loading Pixal3D Image Cond SS Model ...')
        build_func = self._get_build_pixal3d_image_cond_model()
        model = build_func(self.PIXAL3D_IMAGE_COND_CONFIGS["ss"])
        self.pixal3d_image_cond_ss = model
        return model
        
    def unload_pixal3d_image_cond_ss(self):
        if hasattr(self,'pixal3d_image_cond_ss') and self.pixal3d_image_cond_ss is not None:
            del self.pixal3d_image_cond_ss
            self.pixal3d_image_cond_ss = None
            self._cleanup_cuda() 
            
    def load_pixal3d_image_cond_shape_512(self):    
        if hasattr(self,'pixal3d_image_cond_shape_512') and self.pixal3d_image_cond_shape_512 is not None:
            return self.pixal3d_image_cond_shape_512
        print('Loading Pixal3D Image Cond Shape 512 Model ...')
        build_func = self._get_build_pixal3d_image_cond_model()
        model = build_func(self.PIXAL3D_IMAGE_COND_CONFIGS["shape_512"])
        self.pixal3d_image_cond_shape_512 = model
        return model
        
    def unload_pixal3d_image_cond_shape_512(self):
        if hasattr(self,'pixal3d_image_cond_shape_512') and self.pixal3d_image_cond_shape_512 is not None:
            del self.pixal3d_image_cond_shape_512
            self.pixal3d_image_cond_shape_512 = None
            self._cleanup_cuda() 
            
    def load_pixal3d_image_cond_shape_1024(self):    
        if hasattr(self,'pixal3d_image_cond_shape_1024') and self.pixal3d_image_cond_shape_1024 is not None:
            return self.pixal3d_image_cond_shape_1024
        print('Loading Pixal3D Image Cond Shape 1024 Model ...')
        build_func = self._get_build_pixal3d_image_cond_model()
        model = build_func(self.PIXAL3D_IMAGE_COND_CONFIGS["shape_1024"])
        self.pixal3d_image_cond_shape_1024 = model
        return model
        
    def unload_pixal3d_image_cond_shape_1024(self):
        if hasattr(self,'pixal3d_image_cond_shape_1024') and self.pixal3d_image_cond_shape_1024 is not None:
            del self.pixal3d_image_cond_shape_1024
            self.pixal3d_image_cond_shape_1024 = None
            self._cleanup_cuda() 

    def load_pixal3d_image_cond_tex_1024(self):    
        if hasattr(self,'pixal3d_image_cond_tex_1024') and self.pixal3d_image_cond_tex_1024 is not None:
            return self.pixal3d_image_cond_tex_1024
        print('Loading Pixal3D Image Cond Tex 1024 Model ...')
        build_func = self._get_build_pixal3d_image_cond_model()
        model = build_func(self.PIXAL3D_IMAGE_COND_CONFIGS["tex_1024"])
        self.pixal3d_image_cond_tex_1024 = model
        return model
        
    def unload_pixal3d_image_cond_tex_1024(self):
        if hasattr(self,'pixal3d_image_cond_tex_1024') and self.pixal3d_image_cond_tex_1024 is not None:
            del self.pixal3d_image_cond_tex_1024
            self.pixal3d_image_cond_tex_1024 = None
            self._cleanup_cuda()

    def load_sparse_structure_model(self):        
        if self.models['sparse_structure_flow_model'] is None:
            print('Loading Sparse Structure model ...')
            _path = self._sdnq_remap(os.path.join(self.path, self._pretrained_args['models']['sparse_structure_flow_model']))
            self.models['sparse_structure_flow_model'] = models.from_pretrained(
                _path,
                enable_gguf=getattr(self, 'enable_gguf', False),
                gguf_quant=getattr(self, 'gguf_quant', 'Q8_0'),
                precision=getattr(self, 'precision', None),
                enable_sdnq=getattr(self, 'enable_sdnq', False),
                sdnq_use_quantized_matmul=getattr(self, 'sdnq_use_quantized_matmul', True),
                sdnq_torch_compile=getattr(self, 'sdnq_torch_compile', False),
                isPixal3D=getattr(self, 'isPixal3D', False),
            )
            self.models['sparse_structure_flow_model'].eval()
            self.models['sparse_structure_flow_model'].to(self._device)
        
        if self.models['sparse_structure_decoder'] is None:            
            self.models['sparse_structure_decoder'] = models.from_pretrained(self._pretrained_args['models']['sparse_structure_decoder'], isPixal3D=getattr(self, 'isPixal3D', False))
            self.models['sparse_structure_decoder'].eval()
            if (
                is_amd_gpu
                and os.environ.get(
                    "TRELLIS2_ROCM_SS_DECODER_CPU", "1"
                ) == "1"
            ):
                self.models['sparse_structure_decoder'].cpu()
                print(
                    "[Sparse decoder CPU/FP32] load placement: "
                    "keeping Stage-1 occupancy decoder on CPU"
                )
            else:
                self.models['sparse_structure_decoder'].to(self._device)
            if hasattr(self.models['sparse_structure_decoder'], 'low_vram'):
                self.models['sparse_structure_decoder'].low_vram = self.low_vram
    
    def unload_sparse_structure_model(self):
        if self.models['sparse_structure_flow_model'] is not None:
            del self.models['sparse_structure_flow_model']
            self.models['sparse_structure_flow_model'] = None            
            
        if self.models['sparse_structure_decoder'] is not None:
            del self.models['sparse_structure_decoder']
            self.models['sparse_structure_decoder'] = None
        
        self._cleanup_cuda()
            
    def load_image_cond_model(self):
        if self.image_cond_model is None:
            print('Loading Image Cond model ...')
            self.image_cond_model = getattr(image_feature_extractor, self._pretrained_args['image_cond_model']['name'])(**self._pretrained_args['image_cond_model']['args'])
            self.image_cond_model.to(self._device)
            
    def unload_image_cond_model(self):
        if self.image_cond_model is not None:
            del self.image_cond_model
            self.image_cond_model = None            
            self._cleanup_cuda()
            
    def load_shape_slat_flow_model_512(self):        
        if self.models['shape_slat_flow_model_512'] is None:
            print('Loading Shape Slat Flow 512 model ...')
            _path = self._sdnq_remap(os.path.join(self.path, self._pretrained_args['models']['shape_slat_flow_model_512']))
            self.models['shape_slat_flow_model_512'] = models.from_pretrained(
                _path,
                enable_gguf=getattr(self, 'enable_gguf', False),
                gguf_quant=getattr(self, 'gguf_quant', 'Q8_0'),
                precision=getattr(self, 'precision', None),
                enable_sdnq=getattr(self, 'enable_sdnq', False),
                sdnq_use_quantized_matmul=getattr(self, 'sdnq_use_quantized_matmul', True),
                sdnq_torch_compile=getattr(self, 'sdnq_torch_compile', False),
                isPixal3D=getattr(self, 'isPixal3D', False),
            )
            self.models['shape_slat_flow_model_512'].eval()
            self.models['shape_slat_flow_model_512'].to(self._device)
            
    def unload_shape_slat_flow_model_512(self):
        if self.models['shape_slat_flow_model_512'] is not None:
            del self.models['shape_slat_flow_model_512']
            self.models['shape_slat_flow_model_512'] = None
            self._cleanup_cuda()
            
    def load_tex_slat_flow_model_512(self):
        if 'tex_slat_flow_model_512' not in self._pretrained_args.get('models', {}):
            return
        if self.models.get('tex_slat_flow_model_512') is None:
            print('Loading Texture Slat Flow 512 model ...')
            _path = self._sdnq_remap(os.path.join(self.path, self._pretrained_args['models']['tex_slat_flow_model_512']))
            self.models['tex_slat_flow_model_512'] = models.from_pretrained(
                _path,
                enable_gguf=getattr(self, 'enable_gguf', False),
                gguf_quant=getattr(self, 'gguf_quant', 'Q8_0'),
                precision=getattr(self, 'precision', None),
                enable_sdnq=getattr(self, 'enable_sdnq', False),
                sdnq_use_quantized_matmul=getattr(self, 'sdnq_use_quantized_matmul', True),
                sdnq_torch_compile=getattr(self, 'sdnq_torch_compile', False),
                isPixal3D=getattr(self, 'isPixal3D', False),
            )
            self.models['tex_slat_flow_model_512'].eval()
            self.models['tex_slat_flow_model_512'].to(self._device)          

    def unload_tex_slat_flow_model_512(self):
        if self.models.get('tex_slat_flow_model_512') is not None:
            del self.models['tex_slat_flow_model_512']
            self.models['tex_slat_flow_model_512'] = None
            self._cleanup_cuda()

    def load_tex_slat_decoder(self):        
        if self.models['tex_slat_decoder'] is None:
            print('Loading Texture Slat decoder model ...')
            self.models['tex_slat_decoder'] = models.from_pretrained(
                os.path.join(self.path, self._pretrained_args['models']['tex_slat_decoder']),
                precision=getattr(self, 'precision', None),
                isPixal3D=getattr(self, 'isPixal3D', False)
            )
            self.models['tex_slat_decoder'].eval()
            self.models['tex_slat_decoder'].to(self._device)
            if hasattr(self.models['tex_slat_decoder'], 'low_vram'):
                self.models['tex_slat_decoder'].low_vram = self.low_vram

    def unload_tex_slat_decoder(self):
        if self.models['tex_slat_decoder'] is not None:
            del self.models['tex_slat_decoder']
            self.models['tex_slat_decoder'] = None
            self._cleanup_cuda()
            
    def load_shape_slat_decoder(self):        
        if self.models['shape_slat_decoder'] is None:
            print('Loading Shape Slat decoder model ...')
            self.models['shape_slat_decoder'] = models.from_pretrained(
                os.path.join(self.path, self._pretrained_args['models']['shape_slat_decoder']),
                precision=getattr(self, 'precision', None),
                isPixal3D=getattr(self, 'isPixal3D', False)
            )
            self.models['shape_slat_decoder'].eval()
            self.models['shape_slat_decoder'].to(self._device)
            if hasattr(self.models['shape_slat_decoder'], 'low_vram'):
                self.models['shape_slat_decoder'].low_vram = self.low_vram

    def unload_shape_slat_decoder(self):
        if self.models['shape_slat_decoder'] is not None:
            del self.models['shape_slat_decoder']
            self.models['shape_slat_decoder'] = None
            self._cleanup_cuda()

    def load_shape_slat_flow_model_1024(self):        
        if self.models['shape_slat_flow_model_1024'] is None:
            print('Loading Shape Slat Flow 1024 model ...')
            _path = self._sdnq_remap(os.path.join(self.path, self._pretrained_args['models']['shape_slat_flow_model_1024']))
            self.models['shape_slat_flow_model_1024'] = models.from_pretrained(
                _path,
                enable_gguf=getattr(self, 'enable_gguf', False),
                gguf_quant=getattr(self, 'gguf_quant', 'Q8_0'),
                precision=getattr(self, 'precision', None),
                enable_sdnq=getattr(self, 'enable_sdnq', False),
                sdnq_use_quantized_matmul=getattr(self, 'sdnq_use_quantized_matmul', True),
                sdnq_torch_compile=getattr(self, 'sdnq_torch_compile', False),
                isPixal3D=getattr(self, 'isPixal3D', False),
            )
            self.models['shape_slat_flow_model_1024'].eval()
            self.models['shape_slat_flow_model_1024'].to(self._device)           

    def unload_shape_slat_flow_model_1024(self):
        if self.models['shape_slat_flow_model_1024'] is not None:
            del self.models['shape_slat_flow_model_1024']
            self.models['shape_slat_flow_model_1024'] = None
            self._cleanup_cuda()

    def load_tex_slat_flow_model_1024(self):        
        if self.models['tex_slat_flow_model_1024'] is None:
            print('Loading Texture Slat Flow 1024 model ...')
            _path = self._sdnq_remap(os.path.join(self.path, self._pretrained_args['models']['tex_slat_flow_model_1024']))
            self.models['tex_slat_flow_model_1024'] = models.from_pretrained(
                _path,
                enable_gguf=getattr(self, 'enable_gguf', False),
                gguf_quant=getattr(self, 'gguf_quant', 'Q8_0'),
                precision=getattr(self, 'precision', None),
                enable_sdnq=getattr(self, 'enable_sdnq', False),
                sdnq_use_quantized_matmul=getattr(self, 'sdnq_use_quantized_matmul', True),
                sdnq_torch_compile=getattr(self, 'sdnq_torch_compile', False),
                isPixal3D=getattr(self, 'isPixal3D', False),
            )
            self.models['tex_slat_flow_model_1024'].eval()
            self.models['tex_slat_flow_model_1024'].to(self._device)                   

    def unload_tex_slat_flow_model_1024(self):
        if self.models['tex_slat_flow_model_1024'] is not None:
            del self.models['tex_slat_flow_model_1024']
            self.models['tex_slat_flow_model_1024'] = None
            self._cleanup_cuda()

    def load_shape_slat_encoder(self):        
        if self.models['shape_slat_encoder'] is None:
            print('Loading Shape Slat Encoder model ...')
            self.models['shape_slat_encoder'] = models.from_pretrained(f"{self.path}/ckpts/shape_enc_next_dc_f16c32_fp16", isPixal3D=getattr(self, 'isPixal3D', False))
            self.models['shape_slat_encoder'].eval()
            self.models['shape_slat_encoder'].to(self._device)
            if hasattr(self.models['shape_slat_encoder'], 'low_vram'):
                self.models['shape_slat_encoder'].low_vram = self.low_vram

    def unload_shape_slat_encoder(self):
        if self.models['shape_slat_encoder'] is not None:
            del self.models['shape_slat_encoder']
            self.models['shape_slat_encoder'] = None
            self._cleanup_cuda()      

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            if self.image_cond_model is not None:
                self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        resolution: int,
        include_neg_cond: bool = True,
        *,
        fusion_mode: str = "concat",   # "concat" or "mean"
        max_views: int = 4,            # safety cap for 3090
    ) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image: One of:
                - PIL Image
                - list[PIL Image] (multi-view)
                - torch.Tensor batch (B,H,W,C) from ComfyUI
            resolution: Conditioning resolution (e.g. 512 or 1024)
            include_neg_cond: Whether to include negative conditioning
            fusion_mode: "concat" (recommended) or "mean"
            max_views: Max number of views to fuse when list/batch is provided

        Returns:
            dict with keys: cond (+ neg_cond if include_neg_cond)
        """
        self.image_cond_model.image_size = resolution

        # ---- Normalize input into what image_cond_model expects ----
        # Most implementations expect PIL image or list[PIL images].
        if isinstance(image, torch.Tensor):
            # Expect ComfyUI IMAGE tensor: (B,H,W,C) float in [0,1]
            if image.ndim == 4:
                # Lazy import to avoid circulars if tensor2pil is in nodes/utils
                from .nodes import tensor2pil 
                images = [tensor2pil(image[i]) for i in range(min(int(image.shape[0]), max_views))]
            else:
                raise ValueError(f"Expected image tensor with shape (B,H,W,C), got {tuple(image.shape)}")
        elif isinstance(image, Image.Image):
            images = [image]
        elif isinstance(image, (list, tuple)):
            # list of PIL images
            images = list(image)[:max_views]
            if not images:
                raise ValueError("Empty image list provided to get_cond().")
            if not all(isinstance(im, Image.Image) for im in images):
                raise TypeError("get_cond() received a list/tuple but not all elements are PIL Images.")
        else:
            raise TypeError(f"Unsupported image type for get_cond(): {type(image)}")

        if self.low_vram:
            self.image_cond_model.to(self.device)

        # ---- Extract per-view conditioning ----
        with hide_src_module():
            cond = self.image_cond_model(images)

        # Normalize shapes:
        # Common outputs:
        #   - (V, N, D) for multi-view
        #   - (1, N, D) for single-view list length=1
        #   - (N, D) for some single-image extractors
        if cond.ndim == 2:
            # (N, D) -> (1, N, D)
            cond = cond.unsqueeze(0)
        elif cond.ndim != 3:
            raise RuntimeError(f"Unexpected cond ndim={cond.ndim}, shape={tuple(cond.shape)}")

        # If we passed multiple views, fuse them into one conditioning sequence
        if cond.shape[0] > 1:
            if fusion_mode == "concat":
                # (V, N, D) -> (1, V*N, D)
                cond = cond.reshape(1, -1, cond.shape[-1])
            elif fusion_mode == "mean":
                # (V, N, D) -> (1, N, D)
                cond = cond.mean(dim=0, keepdim=True)
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        if self.low_vram:
            self.image_cond_model.cpu()

        if not include_neg_cond:
            return {"cond": cond}

        neg_cond = torch.zeros_like(cond)
        return {"cond": cond, "neg_cond": neg_cond}

    def _decode_sparse_structure_rocm_cpu(self, decoder, z_s):
        """Decode the Stage-1 occupancy VAE on CPU/FP32 on AMD ROCm.

        The Stage-1 decoder is a dense Conv3d network, not a GGUF model.  On
        gfx1030 we have observed the same deterministic sparse-flow latent
        produce materially different decoded occupancy statistics across
        process runs while the decoder ran in FP16 on ROCm.  Geometry is
        already decided at this point, so later finite-value checks cannot
        repair a corrupted occupancy mask.

        Keep the flow sampler on GPU, but move only this small 16^3 -> 64^3
        dense decoder to CPU/FP32.  The returned logits stay on CPU; the
        caller thresholds/argwhere's them there and later moves sparse
        coordinates to the device as needed.
        """
        import hashlib

        enabled = (
            is_amd_gpu
            and os.environ.get(
                "TRELLIS2_ROCM_SS_DECODER_CPU", "1"
            ) == "1"
        )
        if not enabled:
            return decoder(z_s)

        # Surface an earlier asynchronous GPU failure before changing devices.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        z_cpu = (
            z_s.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        z_hash = hashlib.sha256(
            z_cpu.numpy().tobytes()
        ).hexdigest()

        try:
            previous_device = next(decoder.parameters()).device
        except StopIteration:
            previous_device = torch.device("cpu")

        previous_dtype = getattr(
            decoder, "dtype", torch.float32
        )
        previous_use_fp16 = bool(
            getattr(decoder, "use_fp16", False)
        )
        previous_use_fp8 = bool(
            getattr(decoder, "use_fp8", False)
        )

        print(
            "[Sparse decoder CPU/FP32] gfx1030 correctness mode: "
            "Stage-1 occupancy decoder -> CPU FP32; "
            f"z_s sha256={z_hash[:16]}"
        )

        try:
            decoder.eval()

            # SparseStructureDecoder.convert_to_fp32() updates its internal
            # activation dtype.  .float() additionally converts the input and
            # output Conv3d modules, which convert_to_fp32() intentionally
            # leaves untouched in the upstream implementation.
            if hasattr(decoder, "convert_to_fp32"):
                decoder.convert_to_fp32()
            if hasattr(decoder, "use_fp16"):
                decoder.use_fp16 = False
            if hasattr(decoder, "use_fp8"):
                decoder.use_fp8 = False
            if hasattr(decoder, "dtype"):
                decoder.dtype = torch.float32
            decoder.float().cpu()

            with torch.inference_mode():
                decoded_cpu = decoder(z_cpu).float().contiguous()

            finite = torch.isfinite(decoded_cpu)
            finite_count = int(finite.sum().item())
            bad = decoded_cpu.numel() - finite_count
            if bad:
                raise RuntimeError(
                    "[Sparse decoder CPU/FP32] Stage-1 decoder produced "
                    f"{bad} NaN/Inf values"
                )

            occ = decoded_cpu > 0
            occ_count = int(occ.sum().item())
            out_hash = hashlib.sha256(
                decoded_cpu.numpy().tobytes()
            ).hexdigest()

            print(
                "[Sparse decoder CPU/FP32] decoded finite="
                f"{finite_count}/{decoded_cpu.numel()}, "
                f"occupied={occ_count:,}, "
                f"logits sha256={out_hash[:16]}"
            )

            # Optional one-run determinism proof.  This deliberately checks
            # both logits and the geometry-defining threshold mask.
            if os.environ.get(
                "TRELLIS2_ROCM_SS_DECODER_VERIFY", "0"
            ) == "1":
                with torch.inference_mode():
                    decoded_verify = (
                        decoder(z_cpu).float().contiguous()
                    )

                verify_bad = (
                    decoded_verify.numel()
                    - int(torch.isfinite(decoded_verify).sum().item())
                )
                if verify_bad:
                    raise RuntimeError(
                        "[Sparse decoder CPU/FP32 verify] second decode "
                        f"contains {verify_bad} NaN/Inf values"
                    )

                max_abs_diff = float(
                    (decoded_verify - decoded_cpu)
                    .abs().max().item()
                )
                mask_equal = bool(
                    torch.equal(
                        decoded_verify > 0,
                        occ,
                    )
                )
                verify_occ = int(
                    (decoded_verify > 0).sum().item()
                )

                print(
                    "[Sparse decoder CPU/FP32 verify] "
                    f"max_abs_diff={max_abs_diff:.9g}, "
                    f"mask_equal={int(mask_equal)}, "
                    f"occupied={verify_occ:,}"
                )

                if not mask_equal:
                    raise RuntimeError(
                        "[Sparse decoder CPU/FP32 verify] repeated decode "
                        "changed the Stage-1 occupancy mask"
                    )

                del decoded_verify

            del finite, occ
            return decoded_cpu

        finally:
            # Restore model precision metadata for pipeline reuse.  In
            # low-VRAM mode it should stay on CPU; otherwise put it back on
            # its original device.  The decoded logits returned above are an
            # independent CPU tensor.
            try:
                if previous_use_fp8 and hasattr(
                    decoder, "convert_to_fp8"
                ):
                    decoder.convert_to_fp8()
                elif previous_use_fp16 and hasattr(
                    decoder, "convert_to_fp16"
                ):
                    decoder.convert_to_fp16()
                elif hasattr(decoder, "convert_to_fp32"):
                    decoder.convert_to_fp32()

                if hasattr(decoder, "dtype"):
                    decoder.dtype = previous_dtype
                if hasattr(decoder, "use_fp16"):
                    decoder.use_fp16 = previous_use_fp16
                if hasattr(decoder, "use_fp8"):
                    decoder.use_fp8 = previous_use_fp8

                if self.low_vram:
                    decoder.cpu()
                else:
                    decoder.to(previous_device)
            except Exception as restore_exc:
                print(
                    "[Sparse decoder CPU/FP32] warning: failed to "
                    f"restore decoder placement/precision: {restore_exc!r}"
                )


    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        sampler: str = None,
        fill_holes: bool = True,
        hole_structure: int = 1,
        hole_iterations: int = 1,
        hole_fill_algorithm: str = "remove_small_holes",
        keep_only_shell: bool = True,
        verbose: bool = True,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        if self.low_vram:
            cond = self._cond_to(cond, self.device)                         
        # Sample sparse structure latent
        
        if isinstance(cond, dict) and 'cond' in cond:
            c = cond['cond']
            if isinstance(c, dict):
                print("COND global:", c['global'].abs().mean().item(), "proj:", c['proj'].abs().mean().item())
            else:
                print("COND:", c.abs().mean().item())

        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        
        active_sampler = self._get_sampler(sampler, self.sparse_structure_sampler)
        # Debug: log conditioning shapes and stats
        _c = cond.get('cond', cond)
        if isinstance(_c, dict):
            for k, v in _c.items():
                if isinstance(v, torch.Tensor):
                    print(f"  [SS cond '{k}'] shape={v.shape}, dtype={v.dtype}, mean={v.float().mean():.4f}, std={v.float().std():.4f}")
        _nc = cond.get('neg_cond', None)
        if _nc is not None and isinstance(_nc, dict):
            for k, v in _nc.items():
                if isinstance(v, torch.Tensor):
                    print(f"  [SS neg_cond '{k}'] shape={v.shape}, dtype={v.dtype}, mean={v.float().mean():.4f}")
        from .. import leak_probe
        leak_probe.fingerprint("SS", flow_model, noise, cond)
        leak_probe.watch_first_nan(flow_model, "SS step0")
        z_s = active_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        _ss_cpu_mode = (
            is_amd_gpu
            and os.environ.get(
                "TRELLIS2_ROCM_SS_DECODER_CPU", "1"
            ) == "1"
        )
        if self.low_vram and not _ss_cpu_mode:
            decoder.to(self.device)
        print("z_s stats:", z_s.shape, z_s.max().item(), z_s.min().item(), z_s.mean().item(), z_s.dtype)
        tmp_decoded = self._decode_sparse_structure_rocm_cpu(
            decoder, z_s
        )
        print("decoded stats:", tmp_decoded.shape, tmp_decoded.max().item(), tmp_decoded.min().item(), tmp_decoded.mean().item(), tmp_decoded.dtype)
        decoded = tmp_decoded > 0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            current_res = int(decoded.shape[2])
            if resolution < current_res and current_res % resolution == 0:
                ratio = current_res // resolution
                decoded = torch.nn.functional.max_pool3d(
                    decoded.float(), ratio, ratio, 0
                ) > 0.5
            else:
                decoded = torch.nn.functional.interpolate(
                    decoded.float(),
                    size=(resolution, resolution, resolution),
                    mode='nearest',
                ) > 0.5

        # Official Pixal3D MultiView must preserve the decoded occupancy mask
        # exactly as produced by the sparse-structure decoder.  The shell
        # pruning below was added for generic TRELLIS.2 runs to control
        # pathological nearly-solid masks, but Pixal3D was not trained with
        # that postprocessing and can collapse to disconnected fragments.
        is_pixal3d_mv = bool(
            getattr(self, "isPixal3DMV", False)
        )

        occupied_before_shell = int(decoded.sum().item())
        print(
            f"[Sparse] occupied before shell="
            f"{occupied_before_shell:,}"
        )

        if keep_only_shell and not is_pixal3d_mv:
            # Fast GPU binary erosion: preserve two surface layers, remove deep
            # interior voxels before sparse SLat.  This prevents pathological
            # 10-15 minute *per step* runs from nearly-solid voxel fields.
            def _erode3(mask):
                inv = (~mask).float()
                inv = torch.nn.functional.pad(
                    inv,
                    (1, 1, 1, 1, 1, 1),
                    mode="constant",
                    value=1.0,
                )
                return ~torch.nn.functional.max_pool3d(
                    inv, 3, 1, 0
                ).bool()

            eroded1 = _erode3(decoded)
            eroded2 = _erode3(eroded1)
            decoded = decoded & ~eroded2

            occupied_after_shell = int(decoded.sum().item())
            print(
                f"[Sparse] occupied after shell="
                f"{occupied_after_shell:,}"
            )

        elif keep_only_shell and is_pixal3d_mv:
            print(
                "[Pixal3D MV Sparse] official mode: "
                "shell pruning BYPASSED"
            )

        coords = torch.argwhere(
            decoded
        )[:, [0, 2, 3, 4]].int()

        token_count = int(coords.shape[0])

        if is_pixal3d_mv:
            print(
                f"[Pixal3D MV Sparse] official occupied coords="
                f"{token_count:,}"
            )
            if token_count == 0:
                raise RuntimeError(
                    "[Pixal3D MV Sparse] Stage-1 occupancy is empty. "
                    "Stopping before projected conditioning instead of "
                    "falling through to an unrelated torch.cat() error. "
                    "For official MV keep the original equal-size camera "
                    "canvases and strict framing enabled "
                    "(TRELLIS2_PIXAL3D_MV_STRICT_FRAMING=1, "
                    "TRELLIS2_MV_RIG_NORMALIZE=0)."
                )
        else:
            print(
                f"[Sparse] occupied tokens="
                f"{token_count:,}"
            )

        # Generic TRELLIS guard only.  Official Pixal3D MV has its own
        # downstream max_num_tokens handling and should not be clipped by
        # this custom fast-run limit.
        if is_pixal3d_mv:
            speed_limit = 0
        else:
            speed_limit = int(
                kwargs.get(
                    "max_sparse_tokens",
                    18000 if int(resolution) <= 32 else 32768,
                )
            )

        if speed_limit > 0 and token_count > speed_limit:
            raise RuntimeError(
                f"Sparse structure has {token_count:,} tokens, "
                f"above fast-run limit {speed_limit:,}. "
                "Aborting before Shape SLat; try another seed or "
                "lower sparse_structure_resolution."
            )

        coords = coords.cpu()
        del decoded
        del z_s
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()
        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        sampler: str = None,
        verbose: bool = True,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        proj_image_cond_model=None,
        proj_images=None,
        **kwargs,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        if self.low_vram:
            cond = self._cond_to(cond, self.device)

        coords_dev = coords.to(self.device)
        # Rebuild proj cond with coords if proj model is provided
        if proj_image_cond_model is not None and proj_images is not None:
            grid_res = int(coords[:, 1:].max().item()) + 1
            if isinstance(proj_images, dict) and 'transform_matrix' in proj_images:
                cond = self.get_proj_cond_shape_mv(
                    proj_image_cond_model, proj_images, coords,
                    grid_resolution_override=grid_res
                )
            else:
                cond = self.get_proj_cond_shape(
                    proj_image_cond_model, proj_images, coords,
                    grid_resolution_override=grid_res
                )
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        active_sampler = self._get_sampler(sampler, self.shape_slat_sampler)
        slat = active_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                                

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()

        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        sparse_structure_resolution: int = 32,
        sampler: str = None,
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        proj_image_cond_model=None,
        proj_images=None,
        **kwargs,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR

        if self.low_vram:
            lr_cond = self._cond_to(lr_cond, self.device)
            cond = self._cond_to(cond, self.device)

        coords_dev = coords.to(self.device)                         
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
            
        active_sampler = self._get_sampler(sampler, self.shape_slat_sampler)
        slat = active_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (LR)",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
            self._cleanup_cuda()                                
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        _diag_f = slat.feats.detach().float()
        _diag_total = int(_diag_f.numel())
        _diag_finite = int(torch.isfinite(_diag_f).sum().item())

        print(
            f"[Shape LR latent] "
            f"tokens={int(slat.coords.shape[0]):,}, "
            f"channels={int(slat.feats.shape[-1])}, "
            f"finite={_diag_finite:,}/{_diag_total:,}, "
            f"min={float(_diag_f.min().item()):.6f}, "
            f"max={float(_diag_f.max().item()):.6f}, "
            f"mean={float(_diag_f.mean().item()):.6f}, "
            f"std={float(_diag_f.std(unbiased=False).item()):.6f}"
        )

        if slat.coords.numel() > 0:
            _diag_c = slat.coords[:, 1:].detach().cpu()
            print(
                f"[Shape LR coords] "
                f"min={_diag_c.min(dim=0).values.tolist()}, "
                f"max={_diag_c.max(dim=0).values.tolist()}"
            )
            del _diag_c

        if _diag_finite != _diag_total:
            raise RuntimeError(
                "[Shape LR latent] contains NaN/Inf"
            )

        del _diag_f
        
        del coords_dev
        if self.low_vram:
            lr_cond = self._cond_cpu(lr_cond)
            self._cleanup_cuda()

        # Upsample       
        self.load_shape_slat_decoder()
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        
        if not self.keep_models_loaded:
            self.unload_shape_slat_decoder()

        # Sparse structure can be generated at a grid other than 32.  Preserve
        # that scale when quantizing the upsampled LR coordinates to the HR grid.
        sparse_ratio = float(sparse_structure_resolution) / 32.0
        
        while True:
            # ==================================================
            # Official Pixal3D cascade coordinate conversion.
            #
            # 1024 -> 64^3 sparse latent lattice.
            #
            # Official implementation uses:
            #
            #   (grid_res - 1)
            #   round()
            #
            # NOT:
            #
            #   grid_res
            #   int()
            # ==================================================
            grid_res = hr_resolution // 16

            quant_coords = torch.cat([
                hr_coords[:, :1],
                (
                    (hr_coords[:, 1:] + 0.5)
                    # LR coords live on a (16 * sparse_structure_resolution) grid:
                    # 512 for a 32^3 SS, 1024 for Pixal3D's 64^3 SS.  Dividing by a
                    # bare 512 doubled every HR coord past the 64^3 lattice
                    # (max 126) and the mesher found no faces.
                    / (lr_resolution * sparse_ratio)
                    * (grid_res - 1)
                ).round().int(),
            ], dim=1)

            coords = quant_coords.unique(dim=0)

            # gfx1030 correctness guard for the 1024 Pixal3D-MV shape flow.
            # The user's successful historical HR runs were around 8k-10k
            # tokens; the current official-cascade run expanded to 42,546
            # tokens and returned 100% NaN at the very first model forward.
            # Keep the official coordinate conversion, but reduce only an
            # oversized ROCm lattice before rebuilding its projected cond.
            if (
                getattr(self, 'isPixal3DMV', False)
                and is_amd_gpu
                and hr_resolution == 1024
            ):
                _rocm_hr_limit = int(os.environ.get(
                    'TRELLIS2_ROCM_HR_MAX_TOKENS',
                    '0',  # 0 = off: subsampling the shell shreds the mesh (2026-09-23)
                ))
                _rocm_hr_limit = min(
                    _rocm_hr_limit,
                    int(max_num_tokens),
                ) if _rocm_hr_limit > 0 else 0
                coords = _pixal3d_rocm_limit_hr_coords(
                    coords,
                    grid_res,
                    _rocm_hr_limit,
                )

            num_tokens = coords.shape[0]

            if (
                num_tokens < max_num_tokens
                or hr_resolution == 1024
            ):
                if hr_resolution != resolution:
                    print(
                        "Due to the limited number of tokens, "
                        f"the resolution is reduced to "
                        f"{hr_resolution}."
                    )
                break

            hr_resolution -= 128

        if getattr(self, "isPixal3DMV", False):
            self._pixal3d_mv_active_grid_res = (
                hr_resolution // 16
            )

            _mv_xyz = coords[:, 1:]

            print(
                "[Pixal3D MV OFFICIAL CASCADE] "
                f"resolution={hr_resolution}, "
                f"grid_res="
                f"{self._pixal3d_mv_active_grid_res}, "
                f"tokens={coords.shape[0]}, "
                f"min={_mv_xyz.min(0).values.tolist()}, "
                f"max={_mv_xyz.max(0).values.tolist()}"
            )

        coords_dev = coords.to(self.device)

        # Free LR proj cond memory before HR proj cond
        if self.low_vram:
            self._cleanup_cuda()
        # Rebuild proj cond with HR coords if proj model is provided
        # Rebuild projected conditioning on the SAME
        # lattice used by the HR flow model.
        if (
            proj_image_cond_model is not None
            and proj_images is not None
        ):
            if getattr(self, "isPixal3DMV", False):
                #
                # Critical:
                #
                # Never derive the Pixal3D projection-grid size
                # from max(occupied_coord)+1.
                #
                # At resolution 1024 the model lattice is 64^3
                # even when the object only occupies e.g. 3..59.
                #
                hr_grid_res = hr_resolution // 16

                print(
                    "[Pixal3D MV OFFICIAL CASCADE] "
                    f"HR conditioning grid_res={hr_grid_res}"
                )
            else:
                hr_grid_res = (
                    int(coords[:, 1:].max().item()) + 1
                )

            cond = self.get_proj_cond_shape(
                proj_image_cond_model,
                proj_images,
                coords,
                grid_resolution_override=hr_grid_res,
            )

            if getattr(self, "isPixal3DMV", False):
                # Verify the actual data handed to the HR model.
                for _branch in ("cond", "neg_cond"):
                    _branch_dict = cond.get(
                        _branch,
                        {},
                    )

                    for _name, _value in _branch_dict.items():
                        _tensor = (
                            _value.feats
                            if hasattr(_value, "feats")
                            else _value
                        )

                        if not torch.is_tensor(_tensor):
                            continue

                        _finite = torch.isfinite(_tensor)

                        print(
                            "[Pixal3D MV HR COND] "
                            f"{_branch}.{_name}: "
                            f"shape={tuple(_tensor.shape)}, "
                            f"dtype={_tensor.dtype}, "
                            f"finite="
                            f"{int(_finite.sum().item())}/"
                            f"{_tensor.numel()}, "
                            f"min="
                            f"{float(_tensor.min().item()):.6f}, "
                            f"max="
                            f"{float(_tensor.max().item()):.6f}"
                        )

                        if not bool(_finite.all().item()):
                            raise RuntimeError(
                                "[Pixal3D MV] HR projected "
                                "conditioning already contains "
                                "NaN/Inf before flow sampling: "
                                f"{_branch}.{_name}"
                            )

                if os.environ.get(
                    "TRELLIS2_MV_CASCADE_DEBUG_STOP",
                    "0",
                ) == "1":
                    raise RuntimeError(
                        "[Pixal3D MV OFFICIAL CASCADE] "
                        "debug stop before HR flow sampler"
                    )

            if self.low_vram:
                self._cleanup_cuda()

        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
            
        active_sampler = self._get_sampler(sampler, self.shape_slat_sampler)
        slat = active_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                                

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        _diag_f = slat.feats.detach().float()
        _diag_total = int(_diag_f.numel())
        _diag_finite = int(torch.isfinite(_diag_f).sum().item())

        print(
            f"[Shape HR latent] "
            f"tokens={int(slat.coords.shape[0]):,}, "
            f"channels={int(slat.feats.shape[-1])}, "
            f"finite={_diag_finite:,}/{_diag_total:,}, "
            f"min={float(_diag_f.min().item()):.6f}, "
            f"max={float(_diag_f.max().item()):.6f}, "
            f"mean={float(_diag_f.mean().item()):.6f}, "
            f"std={float(_diag_f.std(unbiased=False).item()):.6f}"
        )

        if slat.coords.numel() > 0:
            _diag_c = slat.coords[:, 1:].detach().cpu()
            print(
                f"[Shape HR coords] "
                f"min={_diag_c.min(dim=0).values.tolist()}, "
                f"max={_diag_c.max(dim=0).values.tolist()}"
            )
            del _diag_c

        if _diag_finite != _diag_total:
            raise RuntimeError(
                "[Shape HR latent] contains NaN/Inf"
            )

        del _diag_f
        
        del coords_dev
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()

        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
        use_tiled: bool = True,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        
        self.load_shape_slat_decoder()

        if os.environ.get(
            "TRELLIS2_FINAL_DECODER_LOW_VRAM", "0"
        ) == "1":
            self.models["shape_slat_decoder"].to(
                self.device
            )
            print(
                "[Final decoder] decoder moved to device "
                f"{self.device}"
            )

        decoder = self.models['shape_slat_decoder']
        decoder.set_resolution(resolution)

        _is_rocm = (
            getattr(torch.version, "hip", None)
            is not None
        )

        _old_safe_env = os.environ.get(
            "TRELLIS2_ROCM_SAFE_DECODER"
        )

        _fg_config = None
        _old_algo = None
        _old_hashmap_ratio = None

        if self.low_vram:
            decoder.to(self.device)

        if _is_rocm:
            # The flow model has already been offloaded in low-VRAM mode.
            # For the decoder itself, chunked execution + repeated allocator
            # flushing is less stable on gfx1030 than running the moderate
            # sparse token set normally.
            decoder.low_vram = False

            os.environ[
                "TRELLIS2_ROCM_SAFE_DECODER"
            ] = "1"

            from ..modules.sparse.conv import (
                config as _fg_config
            )

            _old_algo = _fg_config.FLEX_GEMM_ALGO
            _old_hashmap_ratio = (
                _fg_config.FLEX_GEMM_HASHMAP_RATIO
            )

            _fg_config.FLEX_GEMM_ALGO = (
                os.environ.get(
                    "TRELLIS2_ROCM_DECODER_ALGO",
                    "implicit_gemm_splitk",  # same result, 1/9 the VRAM of explicit_gemm (2026-09-23)
                )
            )

            _fg_config.FLEX_GEMM_HASHMAP_RATIO = float(
                os.environ.get(
                    "TRELLIS2_ROCM_DECODER_HASHMAP_RATIO",
                    "4.0",
                )
            )

            print(
                "[Trellis2 Decoder] ROCm safe mode: "
                f"algo={_fg_config.FLEX_GEMM_ALGO}, "
                f"hashmap_ratio="
                f"{_fg_config.FLEX_GEMM_HASHMAP_RATIO}, "
                "neighbor_cache=off, "
                "decoder_low_vram=off"
            )
        else:
            decoder.low_vram = bool(self.low_vram)

        try:
            if (
                getattr(self, "isPixal3DMV", False)
                and os.environ.get(
                    "TRELLIS2_MV_OFFICIAL_DECODER",
                    "1",
                ) == "1"
            ):
                print(
                    "[Pixal3D MV decoder] OFFICIAL non-tiled decode "
                    "(useTiled disabled)"
                )

                # Match official Pixal3D:
                #
                #   shape_slat_decoder(
                #       slat,
                #       return_subs=True,
                #   )
                #
                # Do NOT route official MV through the ComfyUI
                # Fast Tiled Decoding path.
                ret = decoder(
                    slat,
                    return_subs=True,
                )
            else:
                ret = decoder(
                    slat,
                    return_subs=True,
                    useTiled=use_tiled,
                )

            # Make the decoder boundary synchronous so a bad HIP kernel is
            # reported here instead of at a later unrelated tensor cast.
            if _is_rocm:
                torch.cuda.synchronize()

        finally:
            if _is_rocm and _fg_config is not None:
                _fg_config.FLEX_GEMM_ALGO = _old_algo
                _fg_config.FLEX_GEMM_HASHMAP_RATIO = (
                    _old_hashmap_ratio
                )

                if _old_safe_env is None:
                    os.environ.pop(
                        "TRELLIS2_ROCM_SAFE_DECODER",
                        None,
                    )
                else:
                    os.environ[
                        "TRELLIS2_ROCM_SAFE_DECODER"
                    ] = _old_safe_env

            if self.low_vram:
                decoder.low_vram = False
                decoder.cpu()
                self._cleanup_cuda()

        if not self.keep_models_loaded:        
            self.unload_shape_slat_decoder()
            
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
        sampler: str = None,
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        proj_image_cond_model=None,
        proj_images=None,
        **kwargs,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        if self.low_vram:
            cond = self._cond_to(cond, self.device)
        # Rebuild proj cond with shape_slat coords if proj model provided
        if proj_image_cond_model is not None and proj_images is not None:
            tex_coords = shape_slat.coords.cpu()
            # flow_model.resolution is sometimes the voxel size (1024) and sometimes already the
            # latent grid (64); only divide the former, and never go below the coords' extent.
            flow_res = int(getattr(flow_model, 'resolution', 1024))
            tex_grid_res = flow_res // 16 if flow_res > 128 else flow_res
            tex_grid_res = max(tex_grid_res, int(tex_coords[:, 1:].max().item()) + 1)
            if isinstance(proj_images, dict) and 'transform_matrix' in proj_images:
                cond = self.get_proj_cond_shape_mv(
                    proj_image_cond_model, proj_images, tex_coords,
                    grid_resolution_override=tex_grid_res
                )
            else:
                cond = self.get_proj_cond_shape(
                    proj_image_cond_model, proj_images, tex_coords,
                    grid_resolution_override=tex_grid_res
                )

        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)

        active_sampler = self._get_sampler(sampler, self.tex_slat_sampler)
        slat = active_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                    

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()                         
        return slat

    def decode_tex_slat(self, slat: SparseTensor, subs: torch.Tensor = None) -> SparseTensor:
        """
        Decode the structured latent to texture voxels.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        # Comprehensive offload to ensure maximum VRAM for decoder
        self.move_all_to_cpu()
        
        self.load_tex_slat_decoder()
        
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
            self.models['tex_slat_decoder'].low_vram = True                                               

        # Same ROCm safe mode the shape decoder uses (see decode_shape_slat): chunked
        # low-VRAM execution and the default flex_gemm algo hit illegal memory accesses
        # on gfx1030 once the texture decoder sees a few thousand coarse voxels.
        _rocm_cfg = None
        if getattr(torch.version, 'hip', None):
            from ..modules.sparse.conv import config as _rocm_cfg
            _old_rocm = (_rocm_cfg.FLEX_GEMM_ALGO, _rocm_cfg.FLEX_GEMM_HASHMAP_RATIO)
            _rocm_cfg.FLEX_GEMM_ALGO = os.environ.get("TRELLIS2_ROCM_DECODER_ALGO", "implicit_gemm_splitk")
            _rocm_cfg.FLEX_GEMM_HASHMAP_RATIO = float(os.environ.get("TRELLIS2_ROCM_DECODER_HASHMAP_RATIO", "4.0"))
            self.models['tex_slat_decoder'].low_vram = False
            _old_safe_env = os.environ.get("TRELLIS2_ROCM_SAFE_DECODER")
            os.environ["TRELLIS2_ROCM_SAFE_DECODER"] = "1"
            print(f"[Trellis2 Tex Decoder] ROCm safe mode: algo={_rocm_cfg.FLEX_GEMM_ALGO}, decoder_low_vram=off")
        try:
            ret = self._decode_tex_slat_inner(slat, subs)
        finally:
            if _rocm_cfg is not None:
                _rocm_cfg.FLEX_GEMM_ALGO, _rocm_cfg.FLEX_GEMM_HASHMAP_RATIO = _old_rocm
                if _old_safe_env is None:
                    os.environ.pop("TRELLIS2_ROCM_SAFE_DECODER", None)
                else:
                    os.environ["TRELLIS2_ROCM_SAFE_DECODER"] = _old_safe_env

        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
            self.models['tex_slat_decoder'].low_vram = False
            self._cleanup_cuda()
        
        if not self.keep_models_loaded:
            self.unload_tex_slat_decoder()
        
        return ret

    def _decode_tex_slat_inner(self, slat, subs):
        if subs is None:
            if getattr(self, 'use_tiled_decoder_for_texture', False):
                tile_overlap = getattr(self, 'tiled_decoder_overlap', 48)
                tile_size = getattr(self, 'tiled_decoder_size', 120)
                ret = self.models['tex_slat_decoder']._tiled_forward(slat, tile_size=tile_size, overlap=tile_overlap) * 0.5 + 0.5
            else:
                ret = self.models['tex_slat_decoder'](slat) * 0.5 + 0.5
        else:
            slat.clear_spatial_cache()
            if getattr(self, 'use_tiled_decoder_for_texture', False):
                tile_overlap = getattr(self, 'tiled_decoder_overlap', 48)
                tile_size = getattr(self, 'tiled_decoder_size', 120)
                ret = self.models['tex_slat_decoder']._tiled_forward(slat, guide_subs=subs, tile_size=tile_size, overlap=tile_overlap) * 0.5 + 0.5
            else:
                ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
        use_tiled: bool = True,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        def strip_ggml_sparse(st):
            if st is None: return None
            feats = torch.empty(st.feats.shape, dtype=st.feats.dtype, device=st.feats.device).copy_(st.feats)
            coords = torch.empty(st.coords.shape, dtype=st.coords.dtype, device=st.coords.device).copy_(st.coords)
            return st.replace(feats=feats, coords=coords)

        shape_slat = strip_ggml_sparse(shape_slat)
        tex_slat = strip_ggml_sparse(tex_slat)

        if (
            torch.version.hip is not None
            and os.environ.get(
                "TRELLIS2_FINAL_DECODER_LOW_VRAM", "0"
            ) == "1"
        ):
            print(
                "[Final decoder] pre-decode model offload"
            )

            # Offload lazy Pixal3D/DINO/NAF conditioners too;
            # these are not necessarily members of self.models.
            for _name, _obj in list(vars(self).items()):
                if (
                    "image_cond_model" in _name
                    and isinstance(_obj, torch.nn.Module)
                ):
                    try:
                        _obj.cpu()
                    except Exception:
                        pass

            # Flow models are no longer needed for geometry decode.
            self.move_all_to_cpu()
            self._cleanup_cuda()

        meshes, subs = self.decode_shape_slat(shape_slat, resolution, use_tiled=use_tiled)
        
        if subs is not None:
            subs = [strip_ggml_sparse(sub) for sub in subs]

        if self.low_vram:
            self._cleanup_cuda()                                                         
            
        if tex_slat is None:
            if self.low_vram:
                self._cleanup_cuda()                                                         
            out_mesh = []
            for m in meshes:
                out_mesh.append(
                    MeshWithVoxel(
                        m.vertices, m.faces,
                        origin = [-0.5, -0.5, -0.5],
                        voxel_size = 1 / resolution,
                        coords = None,
                        attrs = None,
                        voxel_shape = None,
                        layout=self.pbr_attr_layout
                    )
                )
            return out_mesh
        
        else:    
            tex_voxels = self.decode_tex_slat(tex_slat, subs)
            if self.low_vram:
                self._cleanup_cuda()                                                         
            out_mesh = []
            for m, v in zip(meshes, tex_voxels):
                out_mesh.append(
                    MeshWithVoxel(
                        m.vertices.float(), m.faces.int(),
                        origin = [-0.5, -0.5, -0.5],
                        voxel_size = 1 / resolution,
                        coords = v.coords[:, 1:],
                        attrs = v.feats.float(),
                        voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                        layout=self.pbr_attr_layout
                    )
                )
            return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = False,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        sparse_structure_resolution: int = 32,
        max_views: int = 4,
        generate_texture_slat = True,
        use_tiled: bool = True,
        pbar = None,
        sampler: str = None,
        sparse_structure_sampler: str = None,
        shape_sampler: str = None,
        tex_sampler: str = None,
        verbose: bool = False,
        fill_holes: bool = True,
        hole_iterations: int = 1,
        dino_lock: float = 0.00,
        dino_substeps: int = 4,
        hole_fill_algorithm: str = "remove_small_holes",
        dino_foundation_cap: float = 0.92,
        keep_only_shell: bool = True,
        **kwargs,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        # if pipeline_type == '512':
            # assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        # elif pipeline_type == '1024':
            # assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        # elif pipeline_type == '1024_cascade':
            # assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            # assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        # elif pipeline_type == '1536_cascade':
            # assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            # assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        # else:
            # raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        # Accept either a single PIL image or a list of PIL images (multi-view)
        if isinstance(image, (list, tuple)):
            images = list(image)
        else:
            images = [image]

        if preprocess_image:
            images = [self.preprocess_image(im) for im in images]
            
        seed_all(seed)
        
        # Load sparse structure model early so we can check image_attn_mode
        self.load_sparse_structure_model()
        ss_model = self.models['sparse_structure_flow_model']
        ss_attn_mode = getattr(ss_model, 'image_attn_mode', None)

        # Get Image Cond
        self.load_image_cond_model()
        # Use proj conditioning if sparse structure model requires it
        if ss_attn_mode == 'proj':
            proj_cond_model = self.load_pixal3d_image_cond_ss()
            cond_512 = self.get_proj_cond_ss(images, image_cond_model=proj_cond_model)
        else:
            # Multi-view conditioning happens inside get_cond()
            cond_512 = self.get_cond(images, 512, max_views = max_views)
        cond_1024 = self.get_cond(images, 1024, max_views = max_views) if pipeline_type != '512' else None
        
        if pbar is not None:
            pbar.update(1)
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()
        
        #ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        
        # Sampling Sparse Structure
        self.load_sparse_structure_model()        
        coords = self.sample_sparse_structure(
            cond_512, sparse_structure_resolution,
            num_samples, sparse_structure_sampler_params,
            sampler=sparse_structure_sampler or sampler
        )
        
        if pbar is not None:
            pbar.update(1)
        
        if not self.keep_models_loaded:
            self.unload_sparse_structure_model()
        
        # Sampling Shape
        # Build proj conds for shape/tex stages if needed (Pixal3D models require proj conditioning)
        shape_512_model = None
        shape_1024_model = None
        tex_1024_model = None
        if ss_attn_mode == 'proj':
            # coords are in sparse_structure_resolution space; proj grid must match
            coords_grid_res = (
                32
                if getattr(self, "isPixal3DMV", False)
                else int(coords[:, 1:].max().item()) + 1
            )
            shape_512_model = self.load_pixal3d_image_cond_shape_512()
            cond_512 = self.get_proj_cond_shape(shape_512_model, images, coords,
                                                grid_resolution_override=coords_grid_res)
            if pipeline_type != '512':
                shape_1024_model = self.load_pixal3d_image_cond_shape_1024()
                cond_1024 = self.get_proj_cond_shape(shape_1024_model, images, coords,
                                                     grid_resolution_override=coords_grid_res)
                tex_1024_model = self.load_pixal3d_image_cond_tex_1024()

        if pipeline_type == '512':            
            self.unload_shape_slat_flow_model_1024()
            self.load_shape_slat_flow_model_512()            
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_512_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
            
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_1024()
                self.load_tex_slat_flow_model_512()
                tex_slat = self.sample_tex_slat(
                    cond_512, self.models['tex_slat_flow_model_512'],
                    shape_slat, tex_slat_sampler_params,
                    sampler=tex_sampler or sampler,
                    proj_image_cond_model=shape_512_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )
                
                if pbar is not None:
                    pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_512()
            
            res = 512
        elif pipeline_type == '1024':
            self.unload_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_1024_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_1024()
            
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=tex_sampler or sampler,
                    proj_image_cond_model=tex_1024_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )
                
                if pbar is not None:
                    pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
                
            res = 1024
        elif pipeline_type == '1024_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()            
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                sparse_structure_resolution=sparse_structure_resolution,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_1024_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
            # Free memory before texture stage
            if ss_attn_mode == 'proj' and self.low_vram:
                # These locals are the last references keeping the finished proj-cond
                # models (a separate 1.2 GB DINOv3 each) alive: unload_* only clears the
                # pipeline attribute, so without dropping them nothing is actually
                # released until run() returns. tex_1024_model is still needed below.
                proj_cond_model = shape_512_model = shape_1024_model = None
                self.unload_pixal3d_image_cond_ss()
                self.unload_pixal3d_image_cond_shape_512()
                self.unload_pixal3d_image_cond_shape_1024()
                self._cleanup_cuda()

            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=tex_sampler or sampler,
                    proj_image_cond_model=tex_1024_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )
                
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
        elif pipeline_type == '2048_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 2048,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                sparse_structure_resolution=sparse_structure_resolution,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_1024_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
                
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=tex_sampler or sampler,
                    proj_image_cond_model=tex_1024_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )
                
                if pbar is not None:
                    pbar.update(1)
                
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
        elif pipeline_type == '4096_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 4096,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                sparse_structure_resolution=sparse_structure_resolution,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_1024_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
                
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=tex_sampler or sampler,
                    proj_image_cond_model=tex_1024_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )
                        
                if pbar is not None:
                    pbar.update(1)  
                
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
        elif pipeline_type == '1536_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()                
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                sparse_structure_resolution=sparse_structure_resolution,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_1024_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()

            # Free memory before texture stage (the 1024_cascade branch above does the
            # same; 1536 is the heavier path and was missing it entirely).
            if ss_attn_mode == 'proj' and self.low_vram:
                proj_cond_model = shape_512_model = shape_1024_model = None
                self.unload_pixal3d_image_cond_ss()
                self.unload_pixal3d_image_cond_shape_512()
                self.unload_pixal3d_image_cond_shape_1024()
                self._cleanup_cuda()

            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=tex_sampler or sampler,
                    proj_image_cond_model=tex_1024_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )

                if pbar is not None:
                    pbar.update(1)

            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()

        # Clean up proj cond models when not keeping loaded. Dropping the locals matters
        # most here: decode_latent below is the heaviest stage, and without this the four
        # DINOv3 instances (~4.8 GB) stay resident through it.
        if not self.keep_models_loaded and ss_attn_mode == 'proj':
            proj_cond_model = shape_512_model = shape_1024_model = tex_1024_model = None
            self.unload_pixal3d_image_cond_ss()
            self.unload_pixal3d_image_cond_shape_512()
            self.unload_pixal3d_image_cond_shape_1024()
            self.unload_pixal3d_image_cond_tex_1024()
            gc.collect()
        torch.cuda.empty_cache()
        if generate_texture_slat:
            out_mesh = self.decode_latent(shape_slat, tex_slat, res, use_tiled=use_tiled)
        else:
            out_mesh = self.decode_latent(shape_slat, None, res, use_tiled=use_tiled)
        torch.cuda.empty_cache()
        if pbar is not None:
            pbar.update(1)
        if return_latent:
            if generate_texture_slat:
                return out_mesh, (shape_slat, tex_slat, res)
            else:
                return out_mesh, (shape_slat, None, res)
        else:
            return out_mesh

    @torch.no_grad()
    def run_pixal3d_multiview(
        self,
        front: Image.Image,
        back: Image.Image = None,
        left: Image.Image = None,
        right: Image.Image = None,
        seed: int = 42,
        pipeline_type: str = None,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        max_num_tokens: int = 49152,
        sparse_structure_resolution: int = 32,
        generate_texture_slat: bool = True,
        use_tiled: bool = True,
        return_latent: bool = False,
        pbar: ProgressBar = None,
        sampler: str = None,
        sparse_structure_sampler: str = None,
        shape_sampler: str = None,
        tex_sampler: str = None,
        fill_holes: bool = True,
        hole_iterations: int = 1,
        hole_fill_algorithm: str = 'remove_small_holes',
        keep_only_shell: bool = True,
        **kwargs,
    ):
        """Camera-projected Pixal3D MultiView path for GGUF denoisers.

        This intentionally does NOT use Flow*MultiView samplers.  Pixal3D fuses
        front/back/left/right in the projected DINO/NAF condition, then runs the
        normal Pixal3D denoisers exactly as the official inference_mv pipeline does.
        """
        pipeline_type = pipeline_type or self.default_pipeline_type
        seed_all(seed)

        views = self._build_pixal3d_mv_views(
            front, back=back, left=left, right=right,
            camera_angle_x_deg=float(kwargs.get('pixal3d_fov_deg', 20.0)),
            camera_distance=float(kwargs.get('pixal3d_camera_distance', 3.119205)),
            mesh_scale=float(kwargs.get('pixal3d_mesh_scale', 1.0)),
        )

        if pipeline_type == '512' and generate_texture_slat:
            raise ValueError(
                'Pixal3D official MultiView has no 512 texture flow. Use '
                'generate_texture_slat=False for 512 drafts, or 1024_cascade.'
            )

        print(
            '[Pixal3D-GGUF MV] Using camera-projected conditioning '
            f'({"dedicated *_mv weights" if getattr(self, "isPixal3DMV", False) else "base GGUF compatibility weights"}).'
        )

        # Sparse structure -------------------------------------------------
        self.load_sparse_structure_model()
        ss_model = self.models['sparse_structure_flow_model']
        if getattr(ss_model, 'image_attn_mode', None) != 'proj':
            raise RuntimeError(
                'Loaded model is not a Pixal3D projected-attention model. '
                'Select Pixal3D-GGUF in LoadModel.'
            )

        ss_cond_model = self.load_pixal3d_mv_image_cond_ss()
        cond_ss = self.get_proj_cond_ss_mv(views, ss_cond_model)
        coords = self.sample_sparse_structure(
            cond_ss,
            sparse_structure_resolution,
            sampler_params=sparse_structure_sampler_params,
            sampler=sparse_structure_sampler or sampler,
            fill_holes=fill_holes,
            hole_iterations=hole_iterations,
            hole_fill_algorithm=hole_fill_algorithm,
            keep_only_shell=keep_only_shell,
            max_sparse_tokens=min(int(max_num_tokens), 18000)
                if int(sparse_structure_resolution) <= 32
                else min(int(max_num_tokens), 32768),
        )
        print(
            "[Pixal3D MV SS checkpoint] "
            f"resolution={int(sparse_structure_resolution)}, tokens={int(coords.shape[0])}"
        )
        if os.environ.get("TRELLIS2_MV_SS_DEBUG_STOP", "0") == "1":
            raise RuntimeError(
                "[Pixal3D MV SS checkpoint] debug stop after sparse-structure "
                f"generation; tokens={int(coords.shape[0])}"
            )
        del cond_ss
        self.unload_pixal3d_mv_image_cond_ss()
        if not self.keep_models_loaded:
            self.unload_sparse_structure_model()
        if pbar is not None:
            pbar.update(1)

        # Shape conditioning -----------------------------------------------
        grid_lr = int(sparse_structure_resolution)
        shape512_cond_model = self.load_pixal3d_mv_image_cond_shape_512()
        cond_512 = self.get_proj_cond_shape_mv(
            shape512_cond_model, views, coords,
            grid_resolution_override=grid_lr,
        )

        shape1024_cond_model = None
        cond_1024 = None
        if pipeline_type != '512':
            shape1024_cond_model = self.load_pixal3d_mv_image_cond_shape_1024()
            cond_1024 = self.get_proj_cond_shape_mv(
                shape1024_cond_model, views, coords,
                grid_resolution_override=grid_lr,
            )

        # 512 conditioner no longer needed after its sparse condition exists.
        self.unload_pixal3d_mv_image_cond_shape_512()

        # Shape sampling ---------------------------------------------------
        if pipeline_type == '512':
            self.load_shape_slat_flow_model_512()
            shape_slat = self.sample_shape_slat(
                cond_512,
                self.models['shape_slat_flow_model_512'],
                coords,
                shape_slat_sampler_params,
                sampler=shape_sampler or sampler,
            )
            res = 512
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()

        elif pipeline_type == '1024':
            self.load_shape_slat_flow_model_1024()
            shape_slat = self.sample_shape_slat(
                cond_1024,
                self.models['shape_slat_flow_model_1024'],
                coords,
                shape_slat_sampler_params,
                sampler=shape_sampler or sampler,
            )
            res = 1024
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_1024()

        elif pipeline_type in ('1024_cascade', '1536_cascade'):
            target_res = 1024 if pipeline_type == '1024_cascade' else 1536
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512,
                cond_1024,
                self.models['shape_slat_flow_model_512'],
                self.models['shape_slat_flow_model_1024'],
                512,
                target_res,
                coords,
                shape_slat_sampler_params,
                max_num_tokens,
                sparse_structure_resolution=sparse_structure_resolution,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape1024_cond_model,
                proj_images=views,
            )
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
        else:
            raise ValueError(f'Unsupported Pixal3D MultiView pipeline_type: {pipeline_type}')

        del cond_512
        if cond_1024 is not None:
            del cond_1024
        self.unload_pixal3d_mv_image_cond_shape_1024()
        if pbar is not None:
            pbar.update(1)

        # Texture ----------------------------------------------------------
        tex_slat = None
        if generate_texture_slat:
            self.load_tex_slat_flow_model_1024()
            tex_cond_model = self.load_pixal3d_mv_image_cond_tex_1024()
            tex_grid_res = max(1, int(res) // 16)
            tex_cond = self.get_proj_cond_shape_mv(
                tex_cond_model,
                views,
                shape_slat.coords,
                grid_resolution_override=tex_grid_res,
            )
            tex_slat = self.sample_tex_slat(
                tex_cond,
                self.models['tex_slat_flow_model_1024'],
                shape_slat,
                tex_slat_sampler_params,
                sampler=tex_sampler or sampler,
            )
            del tex_cond
            self.unload_pixal3d_mv_image_cond_tex_1024()
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
            if pbar is not None:
                pbar.update(1)

        self._cleanup_cuda()
        out_mesh = self.decode_latent(
            shape_slat, tex_slat if generate_texture_slat else None,
            res, use_tiled=use_tiled,
        )
        self._cleanup_cuda()
        if pbar is not None:
            pbar.update(1)

        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        return out_mesh

    @torch.no_grad()
    def run_multiview(
        self,
        front: Image.Image,
        back: Image.Image = None,
        left: Image.Image = None,
        right: Image.Image = None,
        seed: int = 42,
        pipeline_type: str = None,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        max_num_tokens: int = 49152,
        sparse_structure_resolution: int = 32,
        generate_texture_slat: bool = True,
        use_tiled: bool = True,
        return_latent: bool = False,
        pbar: ProgressBar = None,
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sampler: str = None,
        sparse_structure_sampler: str = None,
        shape_sampler: str = None,
        tex_sampler: str = None,
        verbose: bool = False,
        fill_holes: bool = True,
        hole_iterations: int = 1,
        dino_lock: float = 0.00,
        dino_substeps: int = 4,
        hole_fill_algorithm: str = "remove_small_holes",
        dino_foundation_cap: float = 0.92,
        keep_only_shell: bool = True,
        **kwargs,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline with named multi-view images and spatial blending.
        """
        if pipeline_type is None:
            pipeline_type = self.default_pipeline_type

        # Pixal3D MultiView is NOT the generic TRELLIS per-view blending path.
        # Fuse camera-projected DINO/NAF features first, then run ordinary Pixal3D
        # denoisers.  This fixes proj_context=None with Pixal3D-GGUF.
        if getattr(self, 'isPixal3D', False):
            return self.run_pixal3d_multiview(
                front=front, back=back, left=left, right=right,
                seed=seed, pipeline_type=pipeline_type,
                sparse_structure_sampler_params=sparse_structure_sampler_params,
                shape_slat_sampler_params=shape_slat_sampler_params,
                tex_slat_sampler_params=tex_slat_sampler_params,
                max_num_tokens=max_num_tokens,
                sparse_structure_resolution=sparse_structure_resolution,
                generate_texture_slat=generate_texture_slat,
                use_tiled=use_tiled, return_latent=return_latent, pbar=pbar,
                sampler=sampler,
                sparse_structure_sampler=sparse_structure_sampler,
                shape_sampler=shape_sampler, tex_sampler=tex_sampler,
                fill_holes=fill_holes, hole_iterations=hole_iterations,
                hole_fill_algorithm=hole_fill_algorithm,
                keep_only_shell=keep_only_shell,
                **kwargs,
            )
        
        seed_all(seed)
        
        # Collect views
        views_dict = {'front': front}
        if back is not None: views_dict['back'] = back
        if left is not None: views_dict['left'] = left
        if right is not None: views_dict['right'] = right
        
        views_list = list(views_dict.keys())

        # 1. Conditioning
        # Calculate conditioning per view
        conds = {}     # 1024 or None (if 512)
        lr_conds = {}  # 512 (for cascade)
        conds_512 = {} # Explicit 512 storage for structure sampling
        conds_1024 = {}
        
        self.load_image_cond_model()
        
        if pipeline_type == '512':
             for v, img in views_dict.items():
                c = self.get_cond([img], 512)
                conds[v] = c
                conds_512[v] = c
                
        elif pipeline_type == '1024':
             for v, img in views_dict.items():
                c1024 = self.get_cond([img], 1024)
                conds[v] = c1024
                conds_1024[v] = c1024
                # Does 1024 pipeline use 512 for structure? 
                # run() says: cond_512 = get_cond(..., 512). So yes.
                conds_512[v] = self.get_cond([img], 512)
                
        elif 'cascade' in pipeline_type:
            # 1024_cascade or 1536_cascade
             for v, img in views_dict.items():
                c512 = self.get_cond([img], 512)
                c1024 = self.get_cond([img], 1024)
                lr_conds[v] = c512
                conds[v] = c1024
                conds_512[v] = c512
                conds_1024[v] = c1024
                
        if not self.keep_models_loaded:
            self.unload_image_cond_model()

        if pbar is not None:
            pbar.update(1)
        
        self.load_sparse_structure_model()          
        coords = self.sample_sparse_structure_multiview(
            conds_512, 
            views_list,
            sparse_structure_resolution,
            sampler_params=sparse_structure_sampler_params,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            sampler=sparse_structure_sampler or sampler,
        )
        
        if not self.keep_models_loaded:
            self.unload_sparse_structure_model()        
            
        if pbar is not None:
            pbar.update(1)

        # 3. Shape Slat MultiView
        shape_slat = None
        res = 0
        
        if pipeline_type == '1024_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat = self.sample_shape_slat_cascade_multiview(
                lr_conds, conds, views_list,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sampler=shape_sampler or sampler
            )
            res = 1024
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
                
        elif pipeline_type == '1536_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat = self.sample_shape_slat_cascade_multiview(
                lr_conds, conds, views_list,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sampler=shape_sampler or sampler
            )
            res = 1536
             
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
        
        elif pipeline_type == '512': # Single stage
             self.load_shape_slat_flow_model_512()
             shape_slat = self.sample_shape_slat_multiview(
                 conds, views_list,
                 self.models['shape_slat_flow_model_512'],
                 coords, shape_slat_sampler_params,
                 front_axis=front_axis,
                 blend_temperature=blend_temperature,
                 sampler=shape_sampler or sampler
             )
             res = 512
             if not self.keep_models_loaded:
                 self.unload_shape_slat_flow_model_512()
        
        elif pipeline_type == '1024': # Single stage
             self.load_shape_slat_flow_model_1024()
             shape_slat = self.sample_shape_slat_multiview(
                 conds, views_list,
                 self.models['shape_slat_flow_model_1024'],
                 coords, shape_slat_sampler_params,
                 front_axis=front_axis,
                 blend_temperature=blend_temperature,
                 sampler=shape_sampler or sampler
             )
             res = 1024
             if not self.keep_models_loaded:
                 self.unload_shape_slat_flow_model_1024()

        if pbar is not None:
            pbar.update(1)

        # Texture Slat MultiView
        tex_slat = None
        if generate_texture_slat:
            tex_model_key = 'tex_slat_flow_model_1024'
            if pipeline_type == '512':
                tex_model_key = 'tex_slat_flow_model_512'
                self.load_tex_slat_flow_model_512()
                flow_model = self.models['tex_slat_flow_model_512']
                tex_conds = conds_512
            else:
                self.load_tex_slat_flow_model_1024()
                flow_model = self.models['tex_slat_flow_model_1024']
                tex_conds = conds_1024
            
            tex_slat = self.sample_tex_slat_multiview(
                tex_conds, views_list,
                shape_slat=shape_slat, 
                flow_model=flow_model,
                sampler_params=tex_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sampler=tex_sampler or sampler
            )  
             
            if not self.keep_models_loaded:
                if pipeline_type == '512':
                    self.unload_tex_slat_flow_model_512()
                else:
                    self.unload_tex_slat_flow_model_1024()

            if pbar is not None:
                pbar.update(1)
                 
        torch.cuda.empty_cache()
        if generate_texture_slat:
            out_mesh = self.decode_latent(shape_slat, tex_slat, res, use_tiled=use_tiled)
        else:
            out_mesh = self.decode_latent(shape_slat, None, res, use_tiled=use_tiled)            
        torch.cuda.empty_cache()
        
        if pbar is not None:
             pbar.update(1)
             
        if return_latent:
             return out_mesh, (shape_slat, tex_slat, res)
        else:
             return out_mesh

    def sample_sparse_structure_multiview(
        self,
        conds: dict,
        views: list,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sampler: str = None,
        fill_holes: bool = True,
        hole_structure: int = 1,
        hole_iterations: int = 1,
        hole_fill_algorithm: str = "remove_small_holes",
        keep_only_shell: bool = True,
        verbose: bool = True,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample sparse structures with multi-view blending.
        """
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_to(conds[v], self.device)
                
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        
        default_sampler = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution
        )
        active_sampler = self._get_sampler(sampler, default_sampler)
        
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        
        if self.low_vram:
            flow_model.to(self.device)
            
        z_s = active_sampler.sample(
            flow_model,
            noise,
            conds=conds,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure (MultiView)",
        ).samples
        
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()
            
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
            
        # Standard decoding logic from sample_sparse_structure
        decoded = decoder(z_s) > 0
        
        if self.low_vram:
            decoder.cpu()
            self._cleanup_cuda()
            
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        
        # Extract coordinates (N, 4) -> (b, d, h, w)
        # argwhere returns (b, c, d, h, w), so we want [0, 2, 3, 4]
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        
        coords = coords.cpu()
        del decoded
        del z_s
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_cpu(conds[v])
            self._cleanup_cuda()
            
        return coords

    def sample_shape_slat_multiview(
        self,
        conds: dict,
        views: list,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sampler: str = None,
    ) -> SparseTensor:
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_to(conds[v], self.device)

        coords_dev = coords.to(self.device)                         
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        
        default_sampler = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution,
        )
        active_sampler = self._get_sampler(sampler, default_sampler)
        
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        
        if self.low_vram:
            flow_model.to(self.device)
            
        slat = active_sampler.sample(
            flow_model,
            noise,
            conds=conds,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (MultiView)",
        ).samples
        
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                                

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_cpu(conds[v])
            self._cleanup_cuda()

        return slat

    def sample_shape_slat_cascade_multiview(
        self,
        lr_conds: dict,
        conds: dict,
        views: list,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sampler: str = None,
    ) -> SparseTensor:
        # LR
        if self.low_vram:
            for v in lr_conds:
                lr_conds[v] = self._cond_to(lr_conds[v], self.device)
            for v in conds:
                conds[v] = self._cond_to(conds[v], self.device)

        coords_dev = coords.to(self.device)                         
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels, device=self.device),
            coords=coords_dev,
        )
        
        default_sampler_lr = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model_lr.resolution,
        )
        active_sampler_lr = self._get_sampler(sampler, default_sampler_lr)
        sampler_params_combined = {**self.shape_slat_sampler_params, **sampler_params}
        slat = active_sampler_lr.sample(
            flow_model_lr,
            noise,
            conds=lr_conds,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            **sampler_params_combined,
            verbose=True,
            tqdm_desc="Sampling shape SLat (MultiView LR)",
        ).samples
        
        if self.low_vram:
            flow_model_lr.cpu()
            self._cleanup_cuda()                                
        
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        del coords_dev
        
        # Upsample logic
        self.load_shape_slat_decoder()
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
            
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                hr_resolution = 512
                break

        # HR
        default_sampler_hr = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution,
        )
        active_sampler_hr = self._get_sampler(sampler, default_sampler_hr)
        
        d_slat = active_sampler_hr.sample(
            flow_model,
            noise,
            conds=conds,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            **sampler_params_combined,
            verbose=True,
            tqdm_desc="Sampling shape SLat (MultiView HR)",
        ).samples
        
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()
            
        slat = d_slat * std + mean
        
        if self.low_vram:
            for v in lr_conds:
                lr_conds[v] = self._cond_cpu(lr_conds[v])
            for v in conds:
                conds[v] = self._cond_cpu(conds[v])
            self._cleanup_cuda()
            
        return slat


    def sample_tex_slat_multiview(
        self,
        conds: dict,
        views: list,
        shape_slat: SparseTensor,
        flow_model,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sampler: str = None,
    ) -> SparseTensor:
        """
        Sample structured latent for texture with multi-view blending.
        """
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_to(conds[v], self.device)

        # Normalize shape slat for conditioning
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat_normalized = (shape_slat - mean) / std

        #coords = shape_slat.coords
        #coords_dev = coords.to(self.device)
        
        # Calculate noise channels: total input - concat cond channels
        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise_channels = in_channels - shape_slat.feats.shape[1]
        
        # noise = SparseTensor(
            # feats=torch.randn(coords.shape[0], noise_channels, device=self.device),
            # coords=coords_dev,
        # )
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        
        default_sampler_tex = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution,
        )
        active_sampler_tex = self._get_sampler(sampler, default_sampler_tex)
        
        if self.low_vram:
            flow_model.to(self.device)
            
        slat = active_sampler_tex.sample(
            flow_model,
            noise,
            conds=conds,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            concat_cond=shape_slat_normalized,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat (MultiView)",
        ).samples
        
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        #del coords_dev
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_cpu(conds[v])
            self._cleanup_cuda()
            
        return slat

    def sample_shape_slat_cascade_advanced(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        low_res_sampler_params: dict = {},
        high_res_sampler_params: dict = {},
        max_num_tokens: int = 999999,
        sparse_structure_resolution: int = 32,
        low_res_sampler_name: str = 'euler',
        high_res_sampler_name: str = 'euler',
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR

        if self.low_vram:
            lr_cond = self._cond_to(lr_cond, self.device)
            cond = self._cond_to(cond, self.device)

        coords_dev = coords.to(self.device)                         
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_params = {**self.shape_slat_sampler_params, **low_res_sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)            
        
        active_sampler = self._get_sampler(
            low_res_sampler_name, 
            self.shape_slat_sampler, 
            steps=sampler_params.get("steps", 12)
        )
            
        slat = active_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (LR)",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
            self._cleanup_cuda()                                
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            lr_cond = self._cond_cpu(lr_cond)
            self._cleanup_cuda()

        # Upsample       
        self.load_shape_slat_decoder()
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        
        if not self.keep_models_loaded:
            self.unload_shape_slat_decoder()
            
        ratio = (sparse_structure_resolution / 32)
        
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / (lr_resolution * ratio) * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                print(f"Num Tokens: {num_tokens}")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 512
                break
        
        coords_dev = coords.to(self.device)                                           
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )        
        
        sampler_params = {**self.shape_slat_sampler_params, **high_res_sampler_params}
        
        active_sampler_hr = self._get_sampler(
            high_res_sampler_name, 
            self.shape_slat_sampler, 
            steps=sampler_params.get("steps", 12)
        )
        
        if self.low_vram:
            flow_model.to(self.device)
        slat = active_sampler_hr.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                                

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()

        return slat, hr_resolution   

    def sample_tex_slat_advanced(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
        sampler_name: str = 'euler',
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        proj_image_cond_model=None,
        proj_images=None,
        **kwargs,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        if self.low_vram:
            cond = self._cond_to(cond, self.device)
        # Rebuild proj cond with shape_slat coords if proj model provided
        if proj_image_cond_model is not None and proj_images is not None:
            tex_coords = shape_slat.coords.cpu()
            tex_grid_res = int(tex_coords[:, 1:].max().item()) + 1
            cond = self.get_proj_cond_shape(proj_image_cond_model, proj_images, tex_coords,
                                            grid_resolution_override=tex_grid_res)
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        
        active_sampler = self._get_sampler(
            sampler_name, 
            self.tex_slat_sampler, 
            steps=sampler_params.get("steps", 12)
        )
        
        if self.low_vram:
            flow_model.to(self.device)
        slat = active_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                    

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()                         
        return slat        
            
    @torch.no_grad()
    def run_cascade(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        low_res_shape_slat_sampler_params: dict = {},
        high_res_shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        pipeline_type: str = '1024_cascade',
        max_num_tokens: int = 999999,
        sparse_structure_resolution: int = 32,
        generate_texture_slat = True,
        use_tiled: bool = True,
        pbar = None,
        sparse_structure_sampler = 'euler',
        low_res_shape_sampler = 'euler',
        high_res_shape_sampler = 'euler',
        tex_sampler = 'euler',
        max_views: int = 4
    ) -> List[MeshWithVoxel]:
        
        if isinstance(image, (list, tuple)):
            images = list(image)
        else:
            images = [image]
            
        seed_all(seed)
        
        # Load sparse structure model early so we can check image_attn_mode
        self.load_sparse_structure_model()
        ss_model = self.models['sparse_structure_flow_model']
        ss_attn_mode = getattr(ss_model, 'image_attn_mode', None)

        # Get Image Cond
        self.load_image_cond_model()
        # Use proj conditioning if sparse structure model requires it
        if ss_attn_mode == 'proj':
            proj_cond_model = self.load_pixal3d_image_cond_ss()
            cond_512 = self.get_proj_cond_ss(images, image_cond_model=proj_cond_model)
        else:
            # Multi-view conditioning happens inside get_cond()
            cond_512 = self.get_cond(images, 512, max_views = max_views)
        cond_1024 = self.get_cond(images, 1024, max_views = max_views) if pipeline_type != '512' else None
        
        if pbar is not None:
            pbar.update(1)
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()       
                
        # Sampling Sparse Structure
        self.load_sparse_structure_model()        
        coords = self.sample_sparse_structure(
            cond_512, sparse_structure_resolution,
            num_samples, sparse_structure_sampler_params,
            sampler=sparse_structure_sampler
        )
        
        if pbar is not None:
            pbar.update(1)
        
        if not self.keep_models_loaded:
            self.unload_sparse_structure_model()
        
        # Sampling Shape
        if pipeline_type == '1024_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()            
            shape_slat, res = self.sample_shape_slat_cascade_advanced(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, low_res_shape_slat_sampler_params, high_res_shape_slat_sampler_params,
                max_num_tokens,
                sparse_structure_resolution,
                low_res_shape_sampler, high_res_shape_sampler
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
            
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat_advanced(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, tex_sampler
                )
                
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
        elif pipeline_type == '1536_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()            
            shape_slat, res = self.sample_shape_slat_cascade_advanced(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, low_res_shape_slat_sampler_params, high_res_shape_slat_sampler_params,
                max_num_tokens,
                sparse_structure_resolution,
                low_res_shape_sampler, high_res_shape_sampler
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
            
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat_advanced(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, tex_sampler
                )
                
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()         
            
        torch.cuda.empty_cache()
        if generate_texture_slat:
            out_mesh = self.decode_latent(shape_slat, tex_slat, res, use_tiled=use_tiled)
        else:
            out_mesh = self.decode_latent(shape_slat, None, res, use_tiled=use_tiled)
        torch.cuda.empty_cache()
        pbar.update(1)              

        return out_mesh

    def preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Preprocess the input mesh.
        """
        mesh = mesh.copy()
        vertices = mesh.vertices.copy()
        
        vertices_min = vertices.min(axis=0)
        vertices_max = vertices.max(axis=0)
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale
        tmp = vertices[:, 1].copy()
        vertices[:, 1] = -vertices[:, 2]
        vertices[:, 2] = tmp
        assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), 'vertices out of range'
        
        mesh.vertices = vertices
        return mesh

    def encode_shape_slat(
        self,
        mesh: trimesh.Trimesh,
        resolution: int = 1024,
        use_tiled_encoder: bool = False,
        encoder_tile_size: int = 512,
        encoder_overlap: int = 24,
    ) -> SparseTensor:
        """
        Encode the meshes to structured latent.

        Args:
            mesh (trimesh.Trimesh): The mesh to encode.
            resolution (int): The resolution of mesh
            use_tiled_encoder (bool): Whether to use spatial chunking during encoding to save VRAM.
            encoder_tile_size (int): The size of the spatial chunks.
            encoder_overlap (int): The overlap between spatial chunks.
        
        Returns:
            SparseTensor: The encoded structured latent.
        """
        print('Converting mesh to flexible dual grid ...')
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
            
        vertices = SparseTensor(
            feats=dual_vertices * resolution - voxel_indices,
            coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)
        ).to(self.device)
        intersected = vertices.replace(intersected).to(self.device)
            
        self.load_shape_slat_encoder()
            
        if self.low_vram:
            self.models['shape_slat_encoder'].to(self.device)
            
        if use_tiled_encoder:
            print(f"Encoding shape slat with tiles (size: {encoder_tile_size}, overlap: {encoder_overlap})...")
            
        import inspect
        encoder_forward = getattr(self.models['shape_slat_encoder'], 'forward', None)
        has_use_tiled = False
        if encoder_forward is not None:
            try:
                sig = inspect.signature(encoder_forward)
                has_use_tiled = 'use_tiled' in sig.parameters
            except Exception:
                pass

        if has_use_tiled:
            shape_slat = self.models['shape_slat_encoder'](
                vertices, intersected, 
                use_tiled=use_tiled_encoder, 
                tile_size=encoder_tile_size, 
                overlap=encoder_overlap
            )
        else:
            shape_slat = self.models['shape_slat_encoder'](vertices, intersected)
        
        if self.low_vram:
            self.models['shape_slat_encoder'].cpu()
            
        if not self.keep_models_loaded:
            self.unload_shape_slat_encoder()
            
        return shape_slat

    def postprocess_mesh(
        self,
        mesh: trimesh.Trimesh,
        pbr_voxel: SparseTensor,
        resolution: int = 1024,
        texture_size: int = 1024,
        texture_alpha_mode = 'OPAQUE',
        double_side_material = True,
        bake_on_vertices = False,
        use_custom_normals = False,
        uv_unwrap_method = 'Xatlas',
        mesh_cluster_threshold_cone_half_angle_rad = 60.0,
        inpainting = 'telea'
    ):        
        vertices = mesh.vertices
        faces = mesh.faces
        normals = np.asarray(mesh.vertex_normals).copy()
        
        vertices_torch = torch.from_numpy(vertices).float().cuda()
        faces_torch = torch.from_numpy(faces).int().cuda()
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = mesh.visual.uv.copy()
            uvs[:, 1] = 1 - uvs[:, 1]
            uvs_torch = torch.from_numpy(uvs).float().cuda()
        else:
            if self.low_vram:
                print("[Trellis2] Low VRAM: Offloading voxel grid to CPU for mesh unwrapping...")
                pbr_voxel = pbr_voxel.cpu()
                self._cleanup_cuda()

            _cumesh = cumesh.CuMesh()
            _cumesh.init(vertices_torch, faces_torch)
            print('Unwrapping mesh ...')
            
            if uv_unwrap_method == 'Blender':
                from ..utils.unwrap_utils import blender_unwrap_glb, check_bpy_available
                if not check_bpy_available():
                    print("[Trellis2] WARNING: bpy not installed, falling back to Xatlas.")
                    uv_unwrap_method = 'Xatlas'
                    
            if uv_unwrap_method == 'Blender':
                _out_verts, _out_faces = _cumesh.read()
                new_verts, new_faces, new_uvs, vmap_blender = blender_unwrap_glb(_out_verts.cpu().numpy(), _out_faces.cpu().numpy())
                if new_verts is None:
                    print("[Trellis2] ERROR: Blender unwrap failed, falling back to Xatlas.")
                    uv_unwrap_method = 'Xatlas'
                else:
                    vertices_torch = torch.from_numpy(new_verts).cuda().float()
                    faces_torch = torch.from_numpy(new_faces).cuda().int()
                    uvs_torch = torch.from_numpy(new_uvs).cuda().float()
                    vmap = torch.from_numpy(vmap_blender).cuda().long()
                    
            if uv_unwrap_method == 'Smart':
                from ..utils.unwrap_utils import python_smart_unwrap_glb
                _out_verts, _out_faces = _cumesh.read()
                new_verts, new_faces, new_uvs, vmap_blender = python_smart_unwrap_glb(
                    _out_verts.cpu().numpy(), 
                    _out_faces.cpu().numpy(),
                    angle_limit=np.radians(mesh_cluster_threshold_cone_half_angle_rad)
                )
                if new_verts is None:
                    print("[Trellis2] ERROR: Smart unwrap failed, falling back to Xatlas.")
                    uv_unwrap_method = 'Xatlas'
                else:
                    vertices_torch = torch.from_numpy(new_verts).cuda().float()
                    faces_torch = torch.from_numpy(new_faces).cuda().int()
                    uvs_torch = torch.from_numpy(new_uvs).cuda().float()
                    vmap = torch.from_numpy(vmap_blender).cuda().long()
                    
            if uv_unwrap_method == 'Xatlas':
                vertices_torch, faces_torch, uvs_torch, vmap = _cumesh.uv_unwrap(
                    compute_charts_kwargs={
                        "threshold_cone_half_angle_rad": np.radians(mesh_cluster_threshold_cone_half_angle_rad),
                        "refine_iterations": 0,
                        "global_iterations": 1,
                        "smooth_strength": 1,
                    },
                    return_vmaps=True,
                    verbose=True,
                )
            
            del _cumesh
            gc.collect()      

            if self.low_vram:
                print("[Trellis2] Low VRAM: Moving voxel grid back to GPU...")
                pbr_voxel = pbr_voxel.to(self.device)
                self._cleanup_cuda()
            
            vertices_torch = vertices_torch.cuda()
            faces_torch = faces_torch.cuda()
            uvs_torch = uvs_torch.cuda()
            vertices = vertices_torch.cpu().numpy()
            faces = faces_torch.cpu().numpy()
            uvs = uvs_torch.cpu().numpy()
            normals = normals[vmap.cpu().numpy()]

        # --- Branch: Bake On Vertices (skip UV unwrapping and texture creation) ---
        if bake_on_vertices:
            aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
            
            # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
            if isinstance(aabb, (list, tuple)):
                aabb = np.array(aabb)
            if isinstance(aabb, np.ndarray):
                aabb = torch.tensor(aabb, dtype=torch.float32, device=pbr_voxel.coords.device)

            voxel_size = 1 / resolution

            # Calculate grid dimensions based on AABB and voxel size                
            if voxel_size is not None:
                if isinstance(voxel_size, float):
                    voxel_size = [voxel_size, voxel_size, voxel_size]
                if isinstance(voxel_size, (list, tuple)):
                    voxel_size = np.array(voxel_size)
                if isinstance(voxel_size, np.ndarray):
                    voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=pbr_voxel.coords.device)
                grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
            else:
                if isinstance(grid_size, int):
                    grid_size = [grid_size, grid_size, grid_size]
                if isinstance(grid_size, (list, tuple)):
                    grid_size = np.array(grid_size)
                if isinstance(grid_size, np.ndarray):
                    grid_size = torch.tensor(grid_size, dtype=torch.int32, device=pbr_voxel.coords.device)
                voxel_size = (aabb[1] - aabb[0]) / grid_size
            
            print('Baking colors on vertices...')
            out_vertices = vertices_torch
            out_faces = faces_torch
            out_normals = normals           
            
            # Sample attributes directly at vertex positions from the voxel grid
            # No BVH mapping needed - the voxel grid contains all the color information
            vertex_attrs = grid_sample_3d(
                pbr_voxel.feats,
                pbr_voxel.coords,
                shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
                grid=((out_vertices - aabb[0]) / voxel_size).reshape(1, -1, 3),
                mode='trilinear',
            )
            
            # Extract base color and alpha per vertex (vertex_attrs shape: N_vertices x C)
            base_color_idx = self.pbr_attr_layout['base_color']
            alpha_idx = self.pbr_attr_layout['alpha']
            
            # Get RGB values and squeeze any extra dimensions to get (N, 3)
            vertex_colors_rgb = vertex_attrs[..., base_color_idx].cpu().numpy()
            vertex_colors_rgb = np.squeeze(vertex_colors_rgb)  # Remove batch dims if any
            if vertex_colors_rgb.ndim == 1:
                vertex_colors_rgb = vertex_colors_rgb[None, :]  # Ensure at least 2D
            vertex_colors_rgb = np.clip(vertex_colors_rgb * 255, 0, 255).astype(np.uint8)
            
            # Handle alpha based on texture_alpha_mode
            if texture_alpha_mode == "OPAQUE":
                # For OPAQUE mode, use full alpha (255)
                vertex_alpha = np.full((vertex_colors_rgb.shape[0], 1), 255, dtype=np.uint8)
            else:
                vertex_alpha = vertex_attrs[..., alpha_idx].cpu().numpy()
                vertex_alpha = np.squeeze(vertex_alpha)  # Remove batch dims if any
                vertex_alpha = np.clip(vertex_alpha * 255, 0, 255).astype(np.uint8)
                # Ensure alpha is 2D with shape (N, 1)
                if vertex_alpha.ndim == 1:
                    vertex_alpha = vertex_alpha[:, None]
            
            # Combine into RGBA
            vertex_colors_rgba = np.concatenate([vertex_colors_rgb, vertex_alpha], axis=-1)
            
            print("Finalizing mesh with vertex colors...")
            
            vertices_np = out_vertices.cpu().numpy()
            faces_np = out_faces.cpu().numpy()
            normals_np = out_normals
            
            # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
            vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
            normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
            
            # Create mesh with vertex colors using ColorVisuals
            if use_custom_normals:
                textured_mesh = trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_normals=normals_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )
            else:
                textured_mesh = trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )                
            
            # Return empty placeholder textures for vertex color mode
            placeholder_texture = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            return (textured_mesh, placeholder_texture, placeholder_texture,)
                
        # rasterize (pure-torch UV rasterizer — nvdiffrast's coarse raster is a no-op on AMD/gfx1030)
        print('Finalizing mesh ...')
        import importlib.util as _ilu
        _uv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "uv_rasterizer.py")
        _uv_spec = _ilu.spec_from_file_location("_uv_rasterizer_mod", _uv_path)
        _uv_mod = _ilu.module_from_spec(_uv_spec)
        _uv_spec.loader.exec_module(_uv_mod)
        rasterize_uv = _uv_mod.rasterize_uv
        face_id, bary = rasterize_uv(uvs_torch, faces_torch, texture_size)

        torch.cuda.synchronize()

        mask = face_id > 0
        # Interpolate vertex positions using barycentric coordinates
        # pos must stay (H, W, 3) so the downstream `pos[mask]` keeps working;
        # write only the covered texels (zeros elsewhere are never read).
        pos = torch.zeros(texture_size, texture_size, 3, device=mask.device, dtype=torch.float32)
        if mask.any():
            face_indices = (face_id[mask] - 1).long()
            face_verts = vertices_torch[faces_torch[face_indices]]      # (N, 3, 3)
            bary_masked = bary[mask].float()                            # (N, 3)
            pos[mask] = (face_verts.float() * bary_masked.unsqueeze(-1)).sum(dim=1)
        
        attrs = torch.zeros(texture_size, texture_size, pbr_voxel.shape[1], device=self.device)
        attrs[mask] = flex_gemm.ops.grid_sample.grid_sample_3d(
            pbr_voxel.feats,
            pbr_voxel.coords,
            shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
            grid=((pos[mask] + 0.5) * resolution).reshape(1, -1, 3),
            mode='trilinear',
        ).float()
        
        torch.cuda.synchronize()
        
        # construct mesh
        mask = mask.cpu().numpy()
        base_color = np.clip(attrs[..., self.pbr_attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., self.pbr_attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., self.pbr_attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., self.pbr_attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        
        # extend
        if inpainting == 'telea':
            inpainting_algo = cv2.INPAINT_TELEA
        else:
            inpainting_algo = cv2.INPAINT_NS
            
        mask = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask, 3, inpainting_algo)
        metallic = cv2.inpaint(metallic, mask, 1, inpainting_algo)[..., None]
        roughness = cv2.inpaint(roughness, mask, 1, inpainting_algo)[..., None]
        alpha = cv2.inpaint(alpha, mask, 1, inpainting_algo)[..., None]
        
        baseColorTexture = Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
        metallicRoughnessTexture = Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1))
        
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=baseColorTexture,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=metallicRoughnessTexture,
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=texture_alpha_mode,
            doubleSided=True,
        )

        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
        normals[:, 1], normals[:, 2] = normals[:, 2], -normals[:, 1]
        uvs[:, 1] = 1 - uvs[:, 1] # Flip UV V-coordinate
        
        if use_custom_normals:
            textured_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=normals,
                process=False,
                visual=trimesh.visual.TextureVisuals(uv=uvs, material=material)
            )
        else:
            textured_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False,
                visual=trimesh.visual.TextureVisuals(uv=uvs, material=material)
            )
            
        return textured_mesh, baseColorTexture, metallicRoughnessTexture

    @torch.no_grad()
    def texture_mesh(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        resolution: int = 1024,
        texture_size: int = 2048,
        texture_alpha_mode = 'OPAQUE',
        double_side_material = True,
        max_views = 4,
        bake_on_vertices = False,
        use_custom_normals = False,
        uv_unwrap_method: str = 'Xatlas',
        mesh_cluster_threshold_cone_half_angle_rad=60.0,
        use_tiled_encoder: bool = False,
        encoder_tile_size: int = 512,
        encoder_overlap: int = 24,
        use_tiled_decoder: bool = False,
        decoder_tile_size: int = 120,
        decoder_overlap: int = 48,
        sampler: str = None,
        inpainting: str = 'telea',
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        **kwargs,
    ):
        
        self.use_tiled_decoder_for_texture = use_tiled_decoder
        self.tiled_decoder_size = decoder_tile_size
        self.tiled_decoder_overlap = decoder_overlap
        
        mesh = self.preprocess_mesh(mesh)
        seed_all(seed)
        
        # Accept either a single PIL image or a list of PIL images (multi-view)
        if isinstance(image, (list, tuple)):
            images = list(image)
        else:
            images = [image]
        
        self.load_image_cond_model()        
        cond_resolution = resolution
        if cond_resolution>1024:
            cond_resolution = 1024
            
        cond = self.get_cond(images, cond_resolution, max_views = max_views)
        proj_model = self.load_pixal3d_image_cond_tex_1024()
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()
        
        shape_slat = self.encode_shape_slat(
            mesh, 
            resolution,
            use_tiled_encoder=use_tiled_encoder,
            encoder_tile_size=encoder_tile_size,
            encoder_overlap=encoder_overlap
        )
        
        if resolution==512:
            self.unload_tex_slat_flow_model_1024()
            self.load_tex_slat_flow_model_512()
            tex_model = self.models['tex_slat_flow_model_512']
            
            tex_slat = self.sample_tex_slat(
                cond, tex_model,
                shape_slat, tex_slat_sampler_params,
                sampler=sampler,
                proj_image_cond_model=proj_model,
                proj_images=images
            )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_512()
        else:
            self.unload_tex_slat_flow_model_512()
            self.load_tex_slat_flow_model_1024()
            tex_model = self.models['tex_slat_flow_model_1024']
            
            tex_slat = self.sample_tex_slat(
                cond, tex_model,
                shape_slat, tex_slat_sampler_params, sampler=sampler,
                proj_image_cond_model=proj_model,
                proj_images=images
            )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()

        if not self.keep_models_loaded:
            self.unload_pixal3d_image_cond_tex_1024()
        torch.cuda.empty_cache()
        pbr_voxel = self.decode_tex_slat(tex_slat)
        torch.cuda.empty_cache()
        
        out_mesh, baseColorTexture, metallicRoughnessTexture = self.postprocess_mesh(mesh, pbr_voxel, resolution, texture_size, texture_alpha_mode, double_side_material, bake_on_vertices, use_custom_normals, uv_unwrap_method, mesh_cluster_threshold_cone_half_angle_rad, inpainting)
        return out_mesh, baseColorTexture, metallicRoughnessTexture
        
    @torch.no_grad()
    def texture_mesh_multiview(
        self,
        mesh: trimesh.Trimesh,
        front: Image.Image,
        back: Image.Image,
        left: Image.Image,
        right: Image.Image,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        resolution: int = 1024,
        texture_size: int = 2048,
        texture_alpha_mode = 'OPAQUE',
        double_side_material = True,
        bake_on_vertices = False,
        use_custom_normals = False,
        uv_unwrap_method: str = 'Xatlas',
        mesh_cluster_threshold_cone_half_angle_rad=60.0,
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        use_tiled_encoder: bool = False,
        encoder_tile_size: int = 512,
        encoder_overlap: int = 24,
        use_tiled_decoder: bool = False,
        decoder_tile_size: int = 120,
        decoder_overlap: int = 48,
        sampler: str = None,
        inpainting: str = 'telea',
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        **kwargs,
    ):
        
        self.use_tiled_decoder_for_texture = use_tiled_decoder
        self.tiled_decoder_size = decoder_tile_size
        self.tiled_decoder_overlap = decoder_overlap
        
        mesh = self.preprocess_mesh(mesh)
        seed_all(seed)
        
        self.load_image_cond_model()        
        # Collect views
        views_dict = {'front': front}
        if back is not None: views_dict['back'] = back
        if left is not None: views_dict['left'] = left
        if right is not None: views_dict['right'] = right
        
        views_list = list(views_dict.keys())

        # 1. Conditioning
        # Calculate conditioning per view
        conds = {}
        
        self.load_image_cond_model()
        
        if resolution == 512:
             for v, img in views_dict.items():
                c = self.get_cond([img], 512)
                conds[v] = c
        else:
             for v, img in views_dict.items():
                c = self.get_cond([img], 1024)
                conds[v] = c
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()
        
        shape_slat = self.encode_shape_slat(
            mesh, 
            resolution,
            use_tiled_encoder=use_tiled_encoder,
            encoder_tile_size=encoder_tile_size,
            encoder_overlap=encoder_overlap
        )
        
        if resolution==512:
            self.unload_tex_slat_flow_model_1024()
            self.load_tex_slat_flow_model_512()
            tex_model = self.models['tex_slat_flow_model_512']
            
            tex_slat = self.sample_tex_slat_multiview(
                conds, views_list,
                shape_slat=shape_slat, 
                flow_model=tex_model,
                sampler_params=tex_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sampler=sampler,
            )            
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_512()
        else:
            self.unload_tex_slat_flow_model_512()
            self.load_tex_slat_flow_model_1024()
            tex_model = self.models['tex_slat_flow_model_1024']
            
            tex_slat = self.sample_tex_slat_multiview(
                conds, views_list,
                shape_slat=shape_slat, 
                flow_model=tex_model,
                sampler_params=tex_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sampler=sampler,
            )                          
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
                
        torch.cuda.empty_cache()
        pbr_voxel = self.decode_tex_slat(tex_slat)
        torch.cuda.empty_cache()
        
        out_mesh, baseColorTexture, metallicRoughnessTexture = self.postprocess_mesh(mesh, pbr_voxel, resolution, texture_size, texture_alpha_mode, double_side_material, bake_on_vertices, use_custom_normals, uv_unwrap_method, mesh_cluster_threshold_cone_half_angle_rad, inpainting)
        return out_mesh, baseColorTexture, metallicRoughnessTexture        
    
    def get_coords_from_trimesh(self, mesh, resolution):
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
        
        coords = torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)                
        coords = coords.cpu()
        
        #print(coords)
        
        del voxel_indices
        del dual_vertices
        del intersected
        
        if self.low_vram:
            self._cleanup_cuda() 
        
        return coords;
        
    def sample_mesh_slat(
        self,
        mesh_slat,
        cond: dict,
        flow_model,
        resolution: int,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        downsampling = 16,
        use_tiled_upsample: bool = False,
        upsample_tile_size: int = 16,
        upsample_overlap: int = 2,
        sampler: str = None,
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        proj_image_cond_model=None,
        proj_images=None,
        **kwargs,
    ) -> SparseTensor:
        # Upsample
        self.load_shape_slat_decoder()
        print('Decoding mesh slat ...')
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        decoder = self.models['shape_slat_decoder']
        # hasattr guard for the same reason encode_shape_slat checks the encoder's signature:
        # the checkpoint's model classes come from whatever version of the vendored
        # trellis2_gguf tree is installed, which may predate tiled upsampling.
        if use_tiled_upsample and hasattr(decoder, '_tiled_upsample'):
            print(f'[Trellis2] Tiled upsample (tile={upsample_tile_size}, overlap={upsample_overlap}) ...')
            hr_coords = decoder._tiled_upsample(mesh_slat, upsample_times=4,
                                                tile_size=upsample_tile_size,
                                                overlap=upsample_overlap)
        else:
            if use_tiled_upsample:
                print('[Trellis2] Decoder has no _tiled_upsample; using whole-grid upsample')
            hr_coords = decoder.upsample(mesh_slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        
        if not self.keep_models_loaded:
            self.unload_shape_slat_decoder()
        
        #downsampling = 16
        lr_resolution = resolution
        # if hr_resolution == 512:
            # downsampling = 16
        # elif hr_resolution == 1024:
            # downsampling = 32
        # elif hr_resolution == 1536:
            # downsampling = 32
        
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // downsampling)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                hr_resolution = 512
                break
        
        coords_dev = coords.to(self.device)

        # Rebuild proj cond on the coords we are about to sample, exactly as
        # sample_shape_slat does. A proj-mode flow model wants (global, proj); the plain
        # get_cond() conditioning refine_mesh starts from is only the global half.
        if proj_image_cond_model is not None and proj_images is not None:
            if self.low_vram:
                self._cleanup_cuda()
            hr_grid_res = int(coords[:, 1:].max().item()) + 1
            cond = self.get_proj_cond_shape(proj_image_cond_model, proj_images, coords,
                                            grid_resolution_override=hr_grid_res)
            if self.low_vram:
                self._cleanup_cuda()

        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        active_sampler = self._get_sampler(sampler, self.shape_slat_sampler)
        slat = active_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                                

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()

        return slat, hr_resolution        
    
    def _proj_cond_model_for(self, flow_model, resolution: int, stage: str):
        """
        The pixal3d proj-cond model a flow model needs, or None if it takes plain
        cross-attention conditioning.

        Mirrors the per-stage choice run() makes: there is no tex_512 proj cond model, so
        the 512 path reuses the shape one for texture.
        """
        if getattr(flow_model, 'image_attn_mode', None) != 'proj':
            return None
        if resolution == 512:
            return self.load_pixal3d_image_cond_shape_512()
        if stage == 'tex':
            return self.load_pixal3d_image_cond_tex_1024()
        return self.load_pixal3d_image_cond_shape_1024()

    @torch.no_grad()
    def refine_mesh(
        self,
        mesh: trimesh.Trimesh,
        image,
        seed: int = 42,
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        resolution: int = 1024,
        max_num_tokens = 50000,
        generate_texture_slat = True,
        return_latent = False,
        downsampling = 16,
        use_tiled: bool = True,
        max_views: int = 4,
        use_tiled_encoder: bool = False,
        encoder_tile_size: int = 512,
        encoder_overlap: int = 24,
        use_tiled_decoder: bool = False,
        decoder_tile_size: int = 120,
        decoder_overlap: int = 48,
        use_tiled_upsample: bool = False,
        upsample_tile_size: int = 16,
        upsample_overlap: int = 2,
        sampler: str = None,
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        **kwargs,
    ):
        self.use_tiled_decoder_for_texture = use_tiled_decoder
        self.tiled_decoder_size = decoder_tile_size
        self.tiled_decoder_overlap = decoder_overlap
        mesh = self.preprocess_mesh(mesh)
        seed_all(seed)
        
        self.load_image_cond_model()
        
        if isinstance(image, (list, tuple)):
            images = list(image)
        else:
            images = [image]        
        
        if resolution == 512:
            cond = self.get_cond(images, 512, max_views = max_views)
        else:
            cond = self.get_cond(images, 1024, max_views = max_views)
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()        
        
        mesh_slat = self.encode_shape_slat(
            mesh, 
            resolution,
            use_tiled_encoder=use_tiled_encoder,
            encoder_tile_size=encoder_tile_size,
            encoder_overlap=encoder_overlap
        )
        
        if resolution==512:
            self.unload_shape_slat_flow_model_1024()
            self.load_shape_slat_flow_model_512()
            shape_flow_model = self.models['shape_slat_flow_model_512']
            proj_shape_model = self._proj_cond_model_for(shape_flow_model, 512, 'shape')
            shape_slat, res = self.sample_mesh_slat(
                mesh_slat,
                cond,
                shape_flow_model,
                512,
                shape_slat_sampler_params,
                max_num_tokens,
                downsampling,
                use_tiled_upsample=use_tiled_upsample,
                upsample_tile_size=upsample_tile_size,
                upsample_overlap=upsample_overlap,
                sampler=sampler,
                proj_image_cond_model=proj_shape_model,
                proj_images=images if proj_shape_model is not None else None,
            )

            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()

            if generate_texture_slat:
                self.unload_tex_slat_flow_model_1024()
                self.load_tex_slat_flow_model_512()
                tex_flow_model = self.models['tex_slat_flow_model_512']
                proj_tex_model = self._proj_cond_model_for(tex_flow_model, 512, 'tex')
                tex_slat = self.sample_tex_slat(
                    cond, tex_flow_model,
                    shape_slat, tex_slat_sampler_params,
                    sampler=sampler,
                    proj_image_cond_model=proj_tex_model,
                    proj_images=images if proj_tex_model is not None else None,
                )

            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_512()
        elif resolution == 1024:
            self.unload_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_flow_model = self.models['shape_slat_flow_model_1024']
            proj_shape_model = self._proj_cond_model_for(shape_flow_model, 1024, 'shape')
            shape_slat, res = self.sample_mesh_slat(
                mesh_slat,
                cond,
                shape_flow_model,
                1024,
                shape_slat_sampler_params,
                max_num_tokens,
                downsampling,
                use_tiled_upsample=use_tiled_upsample,
                upsample_tile_size=upsample_tile_size,
                upsample_overlap=upsample_overlap,
                sampler=sampler,
                proj_image_cond_model=proj_shape_model,
                proj_images=images if proj_shape_model is not None else None,
            )

            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_1024()

            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_flow_model = self.models['tex_slat_flow_model_1024']
                proj_tex_model = self._proj_cond_model_for(tex_flow_model, 1024, 'tex')
                tex_slat = self.sample_tex_slat(
                    cond, tex_flow_model,
                    shape_slat, tex_slat_sampler_params, sampler=sampler,
                    proj_image_cond_model=proj_tex_model,
                    proj_images=images if proj_tex_model is not None else None,
                )

            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
        elif resolution == 1536:
            self.unload_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_flow_model = self.models['shape_slat_flow_model_1024']
            proj_shape_model = self._proj_cond_model_for(shape_flow_model, 1536, 'shape')
            shape_slat, res = self.sample_mesh_slat(
                mesh_slat,
                cond,
                shape_flow_model,
                1536,
                shape_slat_sampler_params,
                max_num_tokens,
                downsampling,
                use_tiled_upsample=use_tiled_upsample,
                upsample_tile_size=upsample_tile_size,
                upsample_overlap=upsample_overlap,
                sampler=sampler,
                proj_image_cond_model=proj_shape_model,
                proj_images=images if proj_shape_model is not None else None,
            )

            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_1024()

            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_flow_model = self.models['tex_slat_flow_model_1024']
                proj_tex_model = self._proj_cond_model_for(tex_flow_model, 1536, 'tex')
                tex_slat = self.sample_tex_slat(
                    cond, tex_flow_model,
                    shape_slat, tex_slat_sampler_params, sampler=sampler,
                    proj_image_cond_model=proj_tex_model,
                    proj_images=images if proj_tex_model is not None else None,
                )

            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()

        # decode_latent below is the heaviest stage; the proj cond models are ~1.2 GB of
        # DINOv3 each and unload_* only clears the pipeline attribute, so the locals have
        # to go too (see the same pattern at the end of run()).
        if not self.keep_models_loaded:
            shape_flow_model = tex_flow_model = None
            proj_shape_model = proj_tex_model = None
            self.unload_pixal3d_image_cond_shape_512()
            self.unload_pixal3d_image_cond_shape_1024()
            self.unload_pixal3d_image_cond_tex_1024()
            gc.collect()
        torch.cuda.empty_cache()
        if generate_texture_slat:
            out_mesh = self.decode_latent(shape_slat, tex_slat, res, use_tiled=use_tiled)
        else:
            out_mesh = self.decode_latent(shape_slat, None, res, use_tiled=use_tiled)
        torch.cuda.empty_cache()
        
        if return_latent:
            if generate_texture_slat:
                return out_mesh, (shape_slat, tex_slat, res)
            else:
                return out_mesh, (shape_slat, None, res)
        else:
            return out_mesh      
