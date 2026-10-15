from typing import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm import tqdm
from ...modules.utils import convert_module_to_f16, convert_module_to_f32, zero_module
from ...modules import sparse as sp
from ...modules.norm import LayerNorm32


def _trellis2_safe_empty_cache():
    """
    Avoid releasing allocator blocks while the ROCm sparse decoder is
    executing. FlexGEMM launches asynchronously and gfx1030 is sensitive
    to allocator/cache churn during this stage.
    """
    import os

    if (
        getattr(torch.version, "hip", None) is not None
        and os.environ.get(
            "TRELLIS2_ROCM_SAFE_DECODER",
            "0",
        ) == "1"
    ):
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def chunked_apply(module: nn.Module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size <= 0 or x.shape[0] <= chunk_size:
        return module(x)
    
    # Process first chunk to determine output shape and dtype
    out_0 = module(x[0:chunk_size])
    out_shape = (x.shape[0],) + out_0.shape[1:]
    out = torch.empty(out_shape, device=x.device, dtype=out_0.dtype)
    out[0:chunk_size] = out_0
    
    # Process remaining chunks
    for i in range(chunk_size, x.shape[0], chunk_size):
        out[i:i+chunk_size] = module(x[i:i+chunk_size])
    return out


def inplace_skip_add(h: sp.SparseTensor, skip_feats: torch.Tensor, chunk_size: int) -> sp.SparseTensor:
    """
    Perform in-place addition of skip connection features to reduce peak memory.
    Processes in chunks if low_vram mode is needed.
    """
    if chunk_size <= 0 or h.feats.shape[0] <= chunk_size:
        h.feats.add_(skip_feats)
    else:
        for i in range(0, h.feats.shape[0], chunk_size):
            end = min(i + chunk_size, h.feats.shape[0])
            h.feats[i:end].add_(skip_feats[i:end])
    del skip_feats
    return h


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        resample_mode: Literal['nearest', 'spatial2channel'] = 'nearest',
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.resample_mode = resample_mode
        self.use_checkpoint = use_checkpoint
        self.low_vram = False
        self.chunk_size = 65536
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        if resample_mode == 'nearest':
            self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        elif resample_mode =='spatial2channel' and self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        if resample_mode == 'nearest':
            self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        elif resample_mode =='spatial2channel' and self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        self.updown = None
        if self.downsample:
            if resample_mode == 'nearest':
                self.updown = sp.SparseDownsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseSpatial2Channel(2)
        elif self.upsample:
            self.to_subdiv = sp.SparseLinear(channels, 8)
            if resample_mode == 'nearest':
                self.updown = sp.SparseUpsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseChannel2Spatial(2)

    def _updown(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.downsample:
            x = self.updown(x)
        elif self.upsample:
            x = self.updown(x, subdiv.replace(subdiv.feats > 0))
        return x

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.upsample:
            subdiv = self.to_subdiv(x)
        if self.low_vram:
            h = x.replace(chunked_apply(self.norm1, x.feats, self.chunk_size))
            h = h.replace(chunked_apply(F.silu, h.feats, self.chunk_size))
        else:
            h = x.replace(self.norm1(x.feats))
            h = h.replace(F.silu(h.feats))
        if self.resample_mode == 'spatial2channel':
            h = self.conv1(h)
        h = self._updown(h, subdiv)
        x = self._updown(x, subdiv)
        if self.resample_mode == 'nearest':
            h = self.conv1(h)
        if self.low_vram:
            h = h.replace(chunked_apply(self.norm2, h.feats, self.chunk_size))
            h = h.replace(chunked_apply(F.silu, h.feats, self.chunk_size))
        else:
            h = h.replace(self.norm2(h.feats))
            h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        if self.low_vram:
            skip = self.skip_connection(x)
            h = inplace_skip_add(h, skip.feats if hasattr(skip, 'feats') else skip, self.chunk_size)
        else:
            h = h + self.skip_connection(x)
        if self.upsample:
            return h, subdiv
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockDownsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        low_vram: bool = False,
        chunk_size: int = 65536,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.low_vram = low_vram
        self.chunk_size = chunk_size
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = sp.SparseDownsample(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.low_vram:
            h = x.replace(chunked_apply(self.norm1, x.feats, self.chunk_size))
            h = h.replace(chunked_apply(F.silu, h.feats, self.chunk_size))
        else:
            h = x.replace(self.norm1(x.feats))
            h = h.replace(F.silu(h.feats))
        h = self.updown(h)
        x = self.updown(x)
        h = self.conv1(h)
        if self.low_vram:
            h = h.replace(chunked_apply(self.norm2, h.feats, self.chunk_size))
            h = h.replace(chunked_apply(F.silu, h.feats, self.chunk_size))
        else:
            h = h.replace(self.norm2(h.feats))
            h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        if self.low_vram:
            skip = self.skip_connection(x)
            h = inplace_skip_add(h, skip.feats if hasattr(skip, 'feats') else skip, self.chunk_size)
        else:
            h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockUpsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
        low_vram: bool = False,
        chunk_size: int = 65536,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        self.low_vram = low_vram
        self.chunk_size = chunk_size
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        if self.pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseUpsample(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        if self.low_vram:
            h = x.replace(chunked_apply(self.norm1, x.feats, self.chunk_size))
            h = h.replace(chunked_apply(F.silu, h.feats, self.chunk_size))
        else:
            h = x.replace(self.norm1(x.feats))
            h = h.replace(F.silu(h.feats))
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = self.conv1(h)
        if self.low_vram:
            h = h.replace(chunked_apply(self.norm2, h.feats, self.chunk_size))
            h = h.replace(chunked_apply(F.silu, h.feats, self.chunk_size))
        else:
            h = h.replace(self.norm2(h.feats))
            h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        if self.low_vram:
            skip = self.skip_connection(x)
            h = inplace_skip_add(h, skip.feats if hasattr(skip, 'feats') else skip, self.chunk_size)
        else:
            h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockS2C3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        low_vram: bool = False,
        chunk_size: int = 65536,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.low_vram = low_vram
        self.chunk_size = chunk_size
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        self.updown = sp.SparseSpatial2Channel(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.low_vram:
            h = x.replace(chunked_apply(self.norm1, x.feats, self.chunk_size))
            h = h.replace(chunked_apply(F.silu, h.feats, self.chunk_size))
        else:
            h = x.replace(self.norm1(x.feats))
            h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = self.updown(h)
        x = self.updown(x)
        if self.low_vram:
            h = h.replace(chunked_apply(self.norm2, h.feats, self.chunk_size))
            h = h.replace(chunked_apply(F.silu, h.feats, self.chunk_size))
        else:
            h = h.replace(self.norm2(h.feats))
            h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        if self.low_vram:
            skip = self.skip_connection(x)
            h = inplace_skip_add(h, skip.feats if hasattr(skip, 'feats') else skip, self.chunk_size)
        else:
            h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockC2S3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
        low_vram: bool = False,
        chunk_size: int = 65536,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        self.low_vram = low_vram
        self.chunk_size = chunk_size
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        if pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseChannel2Spatial(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        if self.low_vram:
            h = x.replace(chunked_apply(self.norm1, x.feats, self.chunk_size))
            h = h.replace(chunked_apply(F.silu, h.feats, self.chunk_size))
        else:
            h = x.replace(self.norm1(x.feats))
            h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        if self.low_vram:
            h = h.replace(chunked_apply(self.norm2, h.feats, self.chunk_size))
            h = h.replace(chunked_apply(F.silu, h.feats, self.chunk_size))
        else:
            h = h.replace(self.norm2(h.feats))
            h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        if self.low_vram:
            skip = self.skip_connection(x)
            h = inplace_skip_add(h, skip.feats if hasattr(skip, 'feats') else skip, self.chunk_size)
        else:
            h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, subdiv, use_reentrant=False)
        else:
            return self._forward(x, subdiv)
        
    
class SparseConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        self.low_vram = False
        self.chunk_size = 65536
        
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = sp.SparseConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.SiLU(),
            zero_module(nn.Linear(int(channels * mlp_ratio), channels)),
        )

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.conv(x)
        if self.low_vram:
            h = h.replace(chunked_apply(self.norm, h.feats, self.chunk_size))
            h = h.replace(chunked_apply(self.mlp, h.feats, self.chunk_size))
        else:
            h = h.replace(self.norm(h.feats))
            h = h.replace(self.mlp(h.feats))
        return h + x
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseUnetVaeEncoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.use_fp8 = use_fp8
        self.dtype = torch.bfloat16 if use_fp8 else (torch.float16 if use_fp16 else torch.float32)

        self.input_layer = sp.SparseLinear(in_channels, model_channels[0])
        self.to_latent = sp.SparseLinear(model_channels[-1], 2 * latent_channels)
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[down_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        **block_args[i],
                    )
                )
                
        self.initialize_weights()
        self._low_vram = False
        self.chunk_size = 65536

        if use_fp8:
            self.convert_to_fp8()
        elif use_fp16:
            self.convert_to_fp16()

    @property
    def low_vram(self) -> bool:
        return self._low_vram
    
    @low_vram.setter
    def low_vram(self, value: bool):
        self._low_vram = value
        for m in self.modules():
            if hasattr(m, 'low_vram') and m is not self:
                m.low_vram = value

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.use_fp8 = False
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp8(self) -> None:
        self.use_fp16 = False
        self.use_fp8 = True
        self.dtype = torch.bfloat16
        from ...modules.utils import convert_module_to_fp8
        self.blocks.apply(convert_module_to_fp8)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.use_fp8 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, sample_posterior=False, return_raw=False):
        h = self.input_layer(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                h = block(h)
        
        if self.low_vram:
            def fused_finalize(t):
                t = t.type(x.dtype)
                t = F.layer_norm(t, (t.shape[-1],))
                t = F.linear(t, self.to_latent.weight, self.to_latent.bias)
                return t
            h = h.replace(chunked_apply(fused_finalize, h.feats, self.chunk_size))
        else:
            h = h.type(x.dtype)
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
            h = self.to_latent(h)
        
        # Sample from the posterior distribution
        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)
            
        if return_raw:
            return z, mean, logvar
        else:
            return z
    
    
class SparseUnetVaeDecoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.use_fp8 = use_fp8
        self.pred_subdiv = pred_subdiv
        self.dtype = torch.bfloat16 if use_fp8 else (torch.float16 if use_fp16 else torch.float32)
        self.low_vram = False
        
        self.output_layer = sp.SparseLinear(model_channels[-1], out_channels)
        self.from_latent = sp.SparseLinear(latent_channels, model_channels[0])
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[up_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        pred_subdiv=pred_subdiv,
                        **block_args[i],
                    )
                )
                    
        self.initialize_weights()
        self._low_vram = False
        self.chunk_size = 65536
        if use_fp8:
            self.convert_to_fp8()
        elif use_fp16:
            self.convert_to_fp16()

    @property
    def low_vram(self) -> bool:
        return self._low_vram
    
    @low_vram.setter
    def low_vram(self, value: bool):
        self._low_vram = value
        for m in self.modules():
            if hasattr(m, 'low_vram') and m is not self:
                m.low_vram = value
            
    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.use_fp8 = False
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp8(self) -> None:
        self.use_fp16 = False
        self.use_fp8 = True
        self.dtype = torch.bfloat16
        from ...modules.utils import convert_module_to_fp8
        self.blocks.apply(convert_module_to_fp8)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.use_fp8 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, guide_subs: Optional[List[sp.SparseTensor]] = None, return_subs: bool = False) -> sp.SparseTensor:
        assert guide_subs is None or self.pred_subdiv == False, "Only decoders with pred_subdiv=False can be used with guide_subs"
        assert return_subs == False or self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with return_subs"

        # Final 1024 geometry decode on 16GB gfx1030 expands to
        # millions of sparse voxels. LayerNorm32 otherwise casts
        # the entire feature tensor to FP32 at once.
        _auto_final_low_vram = (
            os.environ.get(
                "TRELLIS2_FINAL_DECODER_LOW_VRAM", "0"
            ) == "1"
        )

        _restore_low_vram = False

        if _auto_final_low_vram and not self.low_vram:
            _restore_low_vram = True

            _chunk = max(
                1024,
                int(
                    os.environ.get(
                        "TRELLIS2_FINAL_DECODER_CHUNK",
                        "65536",
                    )
                ),
            )

            self.chunk_size = _chunk
            self.low_vram = True

            # low_vram setter propagates the boolean; propagate
            # chunk size as well to LayerNorm32/residual blocks.
            for _m in self.modules():
                if hasattr(_m, "chunk_size"):
                    _m.chunk_size = _chunk

            print(
                f"[Final decoder] chunked low-VRAM mode: "
                f"chunk={_chunk:,}, "
                f"input_tokens={int(x.coords.shape[0]):,}"
            )
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        subs_gt = []
        subs = []
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        if self.training:
                            subs_gt.append(h.get_spatial_cache('subdivision'))

                        # Final Shape-SLat decoder diagnostics.
                        # Probe subdivision logits before SparseChannel2Spatial
                        # can turn an all-negative mask into an empty tensor.
                        if hasattr(block, "to_subdiv"):
                            probe = block.to_subdiv(h)
                            pf = probe.feats.detach()

                            parents = int(pf.shape[0])
                            total = int(pf.numel())

                            if total == 0:
                                raise RuntimeError(
                                    f"[Final decode subdiv {i}] "
                                    f"EMPTY logits; parents={parents}"
                                )

                            finite = torch.isfinite(pf)
                            positive = pf > 0

                            finite_n = int(finite.sum().item())
                            positive_n = int(positive.sum().item())
                            active_parents = int(
                                positive.any(dim=-1).sum().item()
                            )

                            pf32 = pf.float()

                            print(
                                f"[Final decode subdiv {i}] "
                                f"parents={parents:,}, "
                                f"active_parents={active_parents:,}, "
                                f"positive_children="
                                f"{positive_n:,}/{total:,}, "
                                f"finite={finite_n:,}/{total:,}, "
                                f"logits[min={float(pf32.min().item()):.6f}, "
                                f"max={float(pf32.max().item()):.6f}, "
                                f"mean={float(pf32.mean().item()):.6f}, "
                                f"std={float(pf32.std(unbiased=False).item()):.6f}]"
                            )

                            if finite_n != total:
                                raise RuntimeError(
                                    f"[Final decode subdiv {i}] "
                                    "contains NaN/Inf"
                                )

                            if positive_n == 0:
                                raise RuntimeError(
                                    f"[Final decode subdiv {i}] "
                                    "ZERO positive subdivision children. "
                                    "Stopping before empty SparseTensor."
                                )

                            del probe, pf, pf32, finite, positive

                        h, sub = block(h)

                        print(
                            f"[Final decode subdiv {i}] "
                            f"output_coords="
                            f"{int(h.coords.shape[0]):,}"
                        )

                        subs.append(sub)
                    else:
                        h = block(h, subdiv=guide_subs[i] if guide_subs is not None else None)
                else:
                    h = block(h)
                # More frequent cache clearing in low_vram mode to reclaim memory between blocks
                if self.low_vram:
                    _trellis2_safe_empty_cache()
            _trellis2_safe_empty_cache()                        
        if self.low_vram:
            def fused_finalize(t):
                t = t.type(x.dtype)
                t = F.layer_norm(t, (t.shape[-1],))
                # Manually call linear to avoid creating intermediate specific type tensor if possible, 
                # but utilizing the weights of output_layer
                t = F.linear(t, self.output_layer.weight, self.output_layer.bias)
                return t
            h = h.replace(chunked_apply(fused_finalize, h.feats, self.chunk_size))
        else:
            h = h.type(x.dtype)
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
            h = self.output_layer(h)
        if self.training and self.pred_subdiv:
            _result = (h, subs_gt, subs)
        elif return_subs:
            _result = (h, subs)
        else:
            _result = h

        if _restore_low_vram:
            self.low_vram = False

        return _result
    
    def upsample(self, x: sp.SparseTensor, upsample_times: int) -> torch.Tensor:
        assert self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with upsampling"

        cached = self._cached_upsample_coords(x, upsample_times)
        if cached is not None:
            return cached

        h = self.from_latent(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            if i == upsample_times:
                return h.coords
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:

                    # Probe subdivision logits BEFORE SparseChannel2Spatial.
                    # This catches the real failure instead of crashing later
                    # when an empty SparseTensor reaches spatial_shape.max().
                    if hasattr(block, "to_subdiv"):
                        probe = block.to_subdiv(h)
                        pf = probe.feats.detach()

                        parents = int(pf.shape[0])
                        total = int(pf.numel())

                        if total == 0:
                            raise RuntimeError(
                                f"[Shape upsample {i}] subdivision logits "
                                f"are already empty; parents={parents}"
                            )

                        finite = torch.isfinite(pf)
                        finite_n = int(finite.sum().item())

                        positive = pf > 0
                        positive_children = int(
                            positive.sum().item()
                        )
                        active_parents = int(
                            positive.any(dim=-1).sum().item()
                        )

                        pmin = float(pf.min().item())
                        pmax = float(pf.max().item())
                        pmean = float(pf.float().mean().item())

                        print(
                            f"[Shape upsample {i}] "
                            f"parents={parents:,}, "
                            f"active_parents={active_parents:,}, "
                            f"positive_children="
                            f"{positive_children:,}/{total:,}, "
                            f"finite={finite_n:,}/{total:,}, "
                            f"logits[min={pmin:.6f}, "
                            f"max={pmax:.6f}, "
                            f"mean={pmean:.6f}]"
                        )

                        if finite_n != total:
                            raise RuntimeError(
                                f"[Shape upsample {i}] subdivision logits "
                                "contain NaN/Inf"
                            )

                        if positive_children == 0:
                            raise RuntimeError(
                                f"[Shape upsample {i}] ZERO positive "
                                "subdivision children. Stopping before "
                                "creating an empty sparse tensor."
                            )

                        del probe, pf, finite, positive

                    h, sub = block(h)

                    print(
                        f"[Shape upsample {i}] "
                        f"output_coords={int(h.coords.shape[0]):,}"
                    )

                else:
                    h = block(h)
        return h.coords

    def _upsample_factors(self, upsample_times: int) -> Optional[List[int]]:
        """
        The spatial factor each of the first `upsample_times` levels multiplies coords by,
        or None if the model has fewer up blocks than that.
        """
        factors = []
        for i in range(upsample_times):
            if i >= len(self.blocks) - 1:       # last level has no up block
                return None
            factor = getattr(getattr(self.blocks[i][-1], 'updown', None), 'factor', None)
            if factor is None:
                return None
            factors.append(factor)
        return factors

    def _cached_upsample_coords(self, x: sp.SparseTensor, upsample_times: int) -> Optional[torch.Tensor]:
        """
        The coords `upsample` would return, read off `x`'s spatial caches instead of computed.

        A latent handed straight from a matching encoder (what the mesh refiner does) still
        carries the `channel2spatial_{factor}` caches that encoder's downsamples wrote. Every
        `SparseChannel2Spatial` on the way back up prefers that cache over the subdivision the
        blocks predict, so with a full set of them the returned coords are already decided
        before a single block runs -- replaying the encoder's structure exactly. Walking the
        caches gives the identical answer for none of the compute or memory. A gap at any
        level means the blocks really do choose the structure there, so return None and let
        the caller run them.
        """
        if upsample_times <= 0:
            return x.coords
        factors = self._upsample_factors(upsample_times)
        if factors is None:
            return None

        scale = x._scale
        coords = x.coords
        for factor in factors:
            # Same lookup SparseChannel2Spatial does: keyed by the tensor's scale.
            entry = x._spatial_cache.get(str(scale), {}).get(f'channel2spatial_{factor}')
            if entry is None:
                return None
            coords = entry[0]
            scale = tuple([s / factor for s in scale])
        return coords.clone()

    @torch.no_grad()
    def _tiled_upsample(
        self,
        x: sp.SparseTensor,
        upsample_times: int,
        tile_size: int = 16,
        overlap: int = 2,
    ) -> torch.Tensor:
        """
        `upsample` with the peak memory of one tile instead of the whole latent.

        The input voxels are cut into `tile_size` cubes (in input-grid units) and upsampled
        one cube at a time. Each tile is padded with an `overlap` halo of neighbouring input
        voxels so its sparse convolutions still see context across the cut; the halo is
        dropped on merge. Every output voxel descends from exactly one input voxel
        (`out_coords // prod(factors)`), so keeping the ones whose ancestor sits in the tile's
        core partitions the output -- no duplicates, no gaps.

        Tiling is an approximation: a voxel whose receptive field reaches past the halo sees
        fewer neighbours than in a whole-grid pass, so predicted subdivisions near a cut can
        differ. Raise `overlap` to narrow that gap at the cost of recomputed voxels.
        """
        assert self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with upsampling"

        # Costs nothing and is exact, so it wins over tiling whenever it is available.
        cached = self._cached_upsample_coords(x, upsample_times)
        if cached is not None:
            return cached

        if tile_size is None or tile_size <= 0 or upsample_times <= 0:
            return self.upsample(x, upsample_times)
        factors = self._upsample_factors(upsample_times)
        if factors is None:
            return self.upsample(x, upsample_times)

        scale = 1
        for factor in factors:
            scale *= factor
        overlap = max(int(overlap), 0)
        tile_size = int(tile_size)

        coords = x.coords
        feats = x.feats
        device = coords.device
        batch = coords[:, 0].long()
        xyz = coords[:, 1:].long()

        # Plan every tile up front so the progress bar is honest about the total.
        plan = []
        for b in batch.unique().tolist():
            in_batch = (batch == b).nonzero(as_tuple=False).squeeze(1)
            bxyz = xyz[in_batch]
            lo = bxyz.amin(dim=0).tolist()
            hi = (bxyz.amax(dim=0) + 1).tolist()
            for ox in range(lo[0], hi[0], tile_size):
                for oy in range(lo[1], hi[1], tile_size):
                    for oz in range(lo[2], hi[2], tile_size):
                        plan.append((b, in_batch, bxyz, (ox, oy, oz)))

        out_chunks = []
        for b, in_batch, bxyz, origin in tqdm(plan, desc='Tiled upsample'):
            core_lo = torch.tensor(origin, device=device, dtype=torch.long)
            core_hi = core_lo + tile_size
            in_core = ((bxyz >= core_lo) & (bxyz < core_hi)).all(dim=1)
            if not bool(in_core.any()):
                continue
            in_halo = ((bxyz >= core_lo - overlap) & (bxyz < core_hi + overlap)).all(dim=1)
            sel = in_batch[in_halo.nonzero(as_tuple=False).squeeze(1)]

            tile_coords = coords[sel]           # advanced indexing already copies
            tile_coords[:, 0] = 0
            tile_out = self.upsample(sp.SparseTensor(feats=feats[sel], coords=tile_coords), upsample_times)

            ancestor = tile_out[:, 1:].long() // scale
            keep = ((ancestor >= core_lo) & (ancestor < core_hi)).all(dim=1)
            if not bool(((ancestor >= core_lo - overlap) & (ancestor < core_hi + overlap)).all()):
                # Coords grew by something other than prod(factors), so the merge above would
                # silently drop or duplicate voxels. Whole-grid pass is the safe answer.
                print('[tiled upsample] unexpected coordinate scaling, falling back to whole-grid upsample')
                return self.upsample(x, upsample_times)
            kept = tile_out[keep]
            kept[:, 0] = b
            out_chunks.append(kept.cpu())

            del tile_out, ancestor, keep, kept
            if self.low_vram:
                _trellis2_safe_empty_cache()

        if len(out_chunks) == 0:
            return coords[:0].clone()
        return torch.cat(out_chunks, dim=0).to(device)
