"""
Sparse View-Aligned Projection Attention Module for TRELLIS2

Sparse versions of ProjectAttention and GatedProjectAttention.

Supports two modes:
- "proj": Standard projection (DINOv3 only)
- "gated_proj": Gated fusion of DINOv3 (semantic) + VAE (color) features
"""

from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor, VarLenTensor


def _is_sparse(x) -> bool:
    return isinstance(x, SparseTensor) or (x is not None and x.__class__.__name__ == 'SparseTensor')



def _align_by_coords(x_coords, proj_coords, proj_feats):
    """Gather proj_feats rows whose (z, y, x) coords match each row of x_coords; rows
    without a match get zeros.

    A sort + binary-search join: O(N log N) memory and time in the number of voxels.
    The previous dense lookup table had (max_coord + 1)**3 rows, i.e. 15.6 GiB for a
    long object reaching far into a 1024-res grid even with ~1k occupied voxels.
    """
    m = int(torch.maximum(x_coords.max(), proj_coords.max()).item()) + 1
    key = lambda c: (c[:, 0] * m + c[:, 1]) * m + c[:, 2]
    proj_key, order = torch.sort(key(proj_coords))
    x_key = key(x_coords)
    pos = torch.searchsorted(proj_key, x_key).clamp_(max=proj_key.numel() - 1)
    hit = proj_key[pos] == x_key
    out = proj_feats[order[pos]]
    out[~hit] = 0
    return out


class SparseProjectAttention(nn.Module):
    """
    Sparse Projection-based Attention Module with per-block proj_linear.
    """
    def __init__(self, cross_attn_block: nn.Module, channels: int, proj_in_channels: int):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(proj_in_channels, channels, bias=True)
        
    def forward(
        self, 
        x: SparseTensor, 
        context: Union[Dict[str, Union[torch.Tensor, VarLenTensor, SparseTensor]], 
                       Tuple[Union[torch.Tensor, VarLenTensor], SparseTensor]]
    ) -> SparseTensor:
        if isinstance(context, dict):
            global_context = context['global']
            proj_context = context['proj']
        else:
            global_context, proj_context = context
        
        global_out = self.cross_attn_block(x, global_context)
        
        if _is_sparse(proj_context):
            # Align proj_context coords to x coords - they may differ in ordering or device
            device = global_out.feats.device
            x_coords_key = x.coords[:, 1:].long().to(device)
            proj_coords_key = proj_context.coords[:, 1:].long().to(device)
            proj_feats_dev = proj_context.feats.to(device)
            aligned_feats = _align_by_coords(x_coords_key, proj_coords_key, proj_feats_dev)
            proj_feats = self.proj_linear(aligned_feats)
            combined_feats = proj_feats + global_out.feats
        else:
            proj_feats = self.proj_linear(proj_context)
            combined_feats = proj_feats + global_out.feats
        
        return global_out.replace(combined_feats)


class SparseGatedProjectAttention(nn.Module):
    """
    Sparse Concat-Projection Attention Module for DINOv3 + VAE features.
    
    Concatenates DINOv3 and VAE projected features and applies a single linear
    projection to model_channels. Zero-initialized for stable training.
    
    Context dict must contain:
    - 'global': Global image features for cross-attention
    - 'proj_semantic': DINOv3 projected features (SparseTensor or Tensor)
    - 'proj_color': VAE projected features (SparseTensor or Tensor)
    """
    def __init__(
        self,
        cross_attn_block: nn.Module,
        channels: int,
        dino_in_channels: int,
        vae_in_channels: int,
    ):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(dino_in_channels + vae_in_channels, channels, bias=True)
        # Zero-init: at start, fused=0, only global cross-attn contributes
        nn.init.zeros_(self.proj_linear.weight)
        nn.init.zeros_(self.proj_linear.bias)

    def _get_feats(self, t):
        return t.feats if _is_sparse(t) else t

    def forward(
        self,
        x: SparseTensor,
        context: Union[Dict[str, Union[torch.Tensor, VarLenTensor, SparseTensor]], Tuple],
    ) -> SparseTensor:
        if isinstance(context, dict):
            global_context = context['global']
            proj_semantic = context['proj_semantic']
            proj_color = context['proj_color']
        else:
            global_context, proj_semantic, proj_color = context

        global_out = self.cross_attn_block(x, global_context)

        fused = self.proj_linear(torch.cat([
            self._get_feats(proj_semantic),
            self._get_feats(proj_color),
        ], dim=-1))
        combined_feats = fused + global_out.feats

        return global_out.replace(combined_feats)
