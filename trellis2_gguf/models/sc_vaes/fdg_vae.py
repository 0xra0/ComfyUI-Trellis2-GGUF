from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .sparse_unet_vae import (
    SparseResBlock3d,
    SparseConvNeXtBlock3d,
    
    SparseResBlockDownsample3d,
    SparseResBlockUpsample3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
)
from .sparse_unet_vae import (
    SparseUnetVaeEncoder,
    SparseUnetVaeDecoder,
    chunked_apply,
)
from ...representations import Mesh
from o_voxel.convert import flexible_dual_grid_to_mesh, tiled_flexible_dual_grid_to_mesh


class FlexiDualGridVaeEncoder(SparseUnetVaeEncoder):
    def __init__(
        self,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        use_fp8: bool = False,
    ):
        super().__init__(
            6,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            down_block_type,
            block_args,
            use_fp16,
            use_fp8,
        )
        
    def forward(self, vertices: sp.SparseTensor, intersected: sp.SparseTensor, sample_posterior=False, return_raw=False):
        x = vertices.replace(torch.cat([
            vertices.feats - 0.5,
            intersected.feats.float() - 0.5,
        ], dim=1))
        return super().forward(x, sample_posterior, return_raw)
    
    
class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
        use_fp8: bool = False,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin
        
        super().__init__(
            7,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
            use_fp16=use_fp16,
            use_fp8=use_fp8,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution
        
    def forward(self, x: sp.SparseTensor, gt_intersected: sp.SparseTensor = None, useTiled: bool = True, **kwargs):
        # Clear stale spatial caches from any previous run.
        # Since `replace()` propagates `_spatial_cache` by reference, neighbor maps
        # built during run 1 would otherwise be reused (with wrong indices) in run 2,
        # causing corrupted vertex positions ("infinite vertical lines" artifact).
        x.clear_spatial_cache()
        decoded = super().forward(x, **kwargs)
        if self.training:
            h, subs_gt, subs = decoded
            if self.low_vram:
                vertices_feats = chunked_apply(lambda t: (1 + 2 * self.voxel_margin) * F.sigmoid(t[..., 0:3]) - self.voxel_margin, h.feats, self.chunk_size)
                vertices = h.replace(vertices_feats)
                intersected_logits = h.replace(h.feats[..., 3:6])
                quad_lerp_feats = chunked_apply(lambda t: F.softplus(t[..., 6:7]), h.feats, self.chunk_size)
                quad_lerp = h.replace(quad_lerp_feats)
            else:
                vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
                intersected_logits = h.replace(h.feats[..., 3:6])
                quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = [Mesh(flexible_dual_grid_to_mesh(
                h.coords[:, 1:], v.feats, i.feats, q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=True
            )) for v, i, q in zip(vertices, gt_intersected, quad_lerp)]
            return mesh, vertices, intersected_logits, subs_gt, subs
        else:
            out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
            h = out_list[0]
            if self.low_vram:
                vertices_feats = chunked_apply(lambda t: (1 + 2 * self.voxel_margin) * F.sigmoid(t[..., 0:3]) - self.voxel_margin, h.feats, self.chunk_size)
                vertices = h.replace(vertices_feats)
                intersected = h.replace(h.feats[..., 3:6] > 0)
                quad_lerp_feats = chunked_apply(lambda t: F.softplus(t[..., 6:7]), h.feats, self.chunk_size)
                quad_lerp = h.replace(quad_lerp_feats)
            else:
                vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
                intersected = h.replace(h.feats[..., 3:6] > 0)
                quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
                
            def _valid_mesh_result(result):
                if not isinstance(result, (tuple, list)):
                    return False

                if len(result) < 2:
                    return False

                verts = result[0]
                faces = result[1]

                if verts is None or faces is None:
                    return False

                if not torch.is_tensor(verts) or not torch.is_tensor(faces):
                    return False

                if verts.numel() == 0 or faces.numel() == 0:
                    return False

                if not bool(torch.isfinite(verts).all()):
                    return False

                return True


            mesh = []

            for batch_idx, (v, i, q) in enumerate(
                zip(vertices, intersected, quad_lerp)
            ):
                common_args = dict(
                    coords=v.coords[:, 1:],
                    dual_vertices=v.feats,
                    intersected_flag=i.feats,
                    split_weight=q.feats,
                    aabb=[
                        [-0.5, -0.5, -0.5],
                        [ 0.5,  0.5,  0.5],
                    ],
                    grid_size=self.resolution,
                    train=False,
                )

                result = None

                # ------------------------------------------------------
                # Fast path: tiled o_voxel conversion
                # ------------------------------------------------------
                if useTiled:
                    try:
                        result = tiled_flexible_dual_grid_to_mesh(
                            **common_args,
                            tile_size=min(
                                128,
                                int(self.resolution)
                            ),
                        )
                    except Exception as e:
                        print(
                            "[Trellis2 Decoder] tiled mesher failed: "
                            f"{type(e).__name__}: {e}"
                        )
                        result = None

                    if not _valid_mesh_result(result):
                        print(
                            "[Trellis2 Decoder] tiled conversion "
                            f"returned no mesh for batch {batch_idx}; "
                            "retrying non-tiled conversion..."
                        )

                # ------------------------------------------------------
                # Reliable fallback
                # ------------------------------------------------------
                if not _valid_mesh_result(result):
                    try:
                        result = flexible_dual_grid_to_mesh(
                            **common_args
                        )
                    except Exception as e:
                        raise RuntimeError(
                            "Both tiled and non-tiled dual-grid "
                            f"conversion failed: {e}"
                        ) from e

                # ------------------------------------------------------
                # Model generated no extractable surface.
                # Do not turn this into a misleading NoneType crash.
                # ------------------------------------------------------
                if not _valid_mesh_result(result):
                    token_count = int(v.coords.shape[0])

                    flags = i.feats

                    if torch.is_tensor(flags):
                        positive = int(
                            flags.bool().sum().item()
                        )
                        total = int(flags.numel())
                    else:
                        positive = -1
                        total = -1

                    raise RuntimeError(
                        "Shape decoder produced no extractable surface. "
                        f"batch={batch_idx}, "
                        f"resolution={self.resolution}, "
                        f"tokens={token_count}, "
                        f"intersections={positive}/{total}. "
                        "The Pixal3D pipeline completed, but this "
                        "shape sample contains no valid dual-grid mesh. "
                        "Try shape_steps=8, another seed, or official "
                        "Pixal3D *_mv flow weights."
                    )

                print(
                    "[Trellis2 Decoder] mesh extracted: "
                    f"{result[0].shape[0]:,} vertices, "
                    f"{result[1].shape[0]:,} faces"
                )

                mesh.append(Mesh(*result))
                    
            out_list[0] = mesh
            return out_list[0] if len(out_list) == 1 else tuple(out_list)
