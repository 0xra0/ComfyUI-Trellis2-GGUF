"""CPU fallbacks for vertex-color baking on ROCm.

The HIP builds of flex_gemm's grid_sample_3d hashmap kernels, cuBVH queries and
cumesh normal computation hung the GPU during "Baking colors on vertices".
These pure-PyTorch versions run on CPU and match flex_gemm's semantics
(voxel i is centred at i + 0.5; trilinear weights are renormalised over the
neighbours that exist in the sparse grid).
"""
import os
import torch


def use_safe_vertex_bake() -> bool:
    flag = os.environ.get("TRELLIS2_SAFE_VERTEX_BAKE")
    if flag is not None:
        return flag not in ("0", "false", "False")
    return getattr(torch.version, "hip", None) is not None


def sparse_trilinear_cpu(feats, coords_xyz, grid_size, query, chunk=1 << 18):
    """feats [N,C], coords_xyz [N,3] int voxel coords, grid_size (W,H,D),
    query [M,3] in voxel units. Returns [M,C] float32 on CPU."""
    feats = feats.detach().float().cpu()
    c = coords_xyz.detach().long().cpu()
    W, H, D = [int(v) for v in grid_size]
    keys = (c[:, 0] * H + c[:, 1]) * D + c[:, 2]
    keys, order = torch.sort(keys)
    feats = feats[order]

    q = query.detach().float().cpu()
    offs = torch.tensor([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    out = torch.zeros(q.shape[0], feats.shape[1])
    for s in range(0, q.shape[0], chunk):
        qc = q[s:s + chunk]
        n = torch.floor(qc[:, None, :] + offs[None]).long()             # [m,8,3]
        inb = ((n >= 0) & (n < torch.tensor([W, H, D]))).all(-1)
        k = (n[..., 0] * H + n[..., 1]) * D + n[..., 2]
        pos = torch.searchsorted(keys, k.clamp_min(0)).clamp_max(keys.numel() - 1)
        valid = inb & (keys[pos] == k)
        w = torch.prod(1 - (n.float() + 0.5 - qc[:, None, :]).abs(), dim=-1) * valid
        f = (w[..., None] * feats[pos]).sum(1)
        out[s:s + chunk] = f / w.sum(1, keepdim=True).clamp_min(1e-12)
    return out


def vertex_normals_cpu(vertices, faces):
    v = vertices.detach().float().cpu()
    f = faces.detach().long().cpu()
    fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1)  # area-weighted
    vn = torch.zeros_like(v)
    for i in range(3):
        vn.index_add_(0, f[:, i], fn)
    return torch.nn.functional.normalize(vn, dim=1, eps=1e-12)
