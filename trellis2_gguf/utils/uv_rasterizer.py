"""
UV-space rasterizer used for texture baking.

nvdiffrast's rasterizer is CUDA-only; every AMD/HIP port of it currently
mis-rasterizes (dropped tiles, out-of-bounds writes), so on ROCm we rasterize
the UV atlas ourselves. The atlas case needs no depth test, no antialiasing and
no mipmaps, so a direct barycentric rasterizer is exact here.

Pixel mapping matches nvdiffrast: texel (row, col) samples at
u = (col + 0.5) / W, v = (row + 0.5) / H, with row 0 at v ~ 0.
"""

from typing import Tuple

import torch


# Cap on candidate (pixel, face) pairs held in memory at once. Faces are
# processed in chunks small enough to stay under this.
_MAX_PAIRS_PER_CHUNK = 8_000_000


def _face_bboxes(px, py, faces, W, H):
    f0, f1, f2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
    ax, ay = px[f0], py[f0]
    bx, by = px[f1], py[f1]
    cx, cy = px[f2], py[f2]

    # Signed 2x area; degenerate faces cover nothing.
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)

    vx = torch.stack([ax, bx, cx], 1)
    vy = torch.stack([ay, by, cy], 1)
    xmin = vx.min(1).values.floor().long().clamp(0, W - 1)
    xmax = vx.max(1).values.ceil().long().clamp(0, W - 1)
    ymin = vy.min(1).values.floor().long().clamp(0, H - 1)
    ymax = vy.max(1).values.ceil().long().clamp(0, H - 1)
    return (ax, ay, bx, by, cx, cy), denom, xmin, xmax, ymin, ymax


def rasterize_uv(
    uvs: torch.Tensor,
    faces: torch.Tensor,
    texture_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize a UV atlas.

    Args:
        uvs: (V, 2) float, UV coordinates in [0, 1].
        faces: (F, 3) int vertex indices.
        texture_size: output height == width.

    Returns:
        face_id: (H, W) int32, 1-based face index, 0 where nothing was hit.
        bary: (H, W, 3) float32 barycentric weights for faces[:, 0], [1], [2].
    """
    H = W = int(texture_size)
    device = uvs.device

    px = uvs[:, 0].float() * W
    py = uvs[:, 1].float() * H

    (ax, ay, bx, by, cx, cy), denom, xmin, xmax, ymin, ymax = _face_bboxes(px, py, faces, W, H)

    bbox_w = xmax - xmin + 1
    bbox_h = ymax - ymin + 1
    area = (bbox_w * bbox_h).clamp(min=0)
    keep = (denom.abs() > 1e-8) & (area > 0)

    face_id = torch.zeros(H * W, dtype=torch.int32, device=device)
    bary = torch.zeros(H * W, 3, dtype=torch.float32, device=device)

    fis = torch.nonzero(keep, as_tuple=False).flatten()
    if fis.numel() == 0:
        return face_id.view(H, W), bary.view(H, W, 3)

    # Chunk over faces so the candidate-pair enumeration stays within budget.
    # Ascending face order is preserved across chunks, so a later face still
    # wins a contested texel, matching a single-pass scatter.
    chunk_area = area[fis]
    bounds = torch.cumsum(chunk_area, 0)
    total = int(bounds[-1].item())
    n_chunks = max(1, (total + _MAX_PAIRS_PER_CHUNK - 1) // _MAX_PAIRS_PER_CHUNK)
    for fi in torch.chunk(fis, n_chunks):
        n_leaf = area[fi]
        n_pairs = int(n_leaf.sum().item())
        if n_pairs == 0:
            continue

        face_idx = torch.repeat_interleave(fi, n_leaf)

        # Position within each face's bounding box.
        starts = torch.cumsum(n_leaf, 0) - n_leaf
        local = torch.arange(n_pairs, device=device) - torch.repeat_interleave(starts, n_leaf)
        bw = bbox_w[face_idx]
        gx = (xmin[face_idx] + local % bw).float() + 0.5
        gy = (ymin[face_idx] + local // bw).float() + 0.5

        _ax, _ay = ax[face_idx], ay[face_idx]
        _bx, _by = bx[face_idx], by[face_idx]
        _cx, _cy = cx[face_idx], cy[face_idx]
        d = denom[face_idx]

        w0 = ((_by - _cy) * (gx - _cx) + (_cx - _bx) * (gy - _cy)) / d
        w1 = ((_cy - _ay) * (gx - _cx) + (_ax - _cx) * (gy - _cy)) / d
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not bool(inside.any()):
            continue

        flat = gy[inside].long() * W + gx[inside].long()
        hit = face_idx[inside]

        # Sort by texel; hit is already ascending, so a stable sort puts the
        # highest face index last within each texel's run. Keep only that last
        # row per texel and assign it. Going through unique indices matters:
        # a plain scatter_ resolves duplicate indices independently per column,
        # which can pair w0 from one face with w1 from another on texels that
        # two triangles share (an edge), producing invalid barycentrics.
        order = torch.argsort(flat, stable=True)
        flat = flat[order]
        hit = hit[order]
        w = torch.stack([w0[inside], w1[inside], w2[inside]], dim=1)[order]

        winner = torch.ones_like(flat, dtype=torch.bool)
        winner[:-1] = flat[1:] != flat[:-1]
        flat = flat[winner]

        face_id[flat] = (hit[winner] + 1).to(torch.int32)
        bary[flat] = w[winner]

    return face_id.view(H, W), bary.view(H, W, 3)


def interpolate_uv(
    attr: torch.Tensor,
    face_id: torch.Tensor,
    bary: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate per-vertex attributes over a rasterized atlas.

    Args:
        attr: (V, C) per-vertex attributes.
        face_id: (H, W) int32 from rasterize_uv (1-based, 0 = background).
        bary: (H, W, 3) from rasterize_uv.
        faces: (F, 3) int vertex indices.

    Returns:
        (H, W, C), zero at background texels.
    """
    H, W = face_id.shape
    out = torch.zeros(H, W, attr.shape[1], device=attr.device, dtype=torch.float32)

    mask = face_id > 0
    if not bool(mask.any()):
        return out

    tri = faces[(face_id[mask].long() - 1)].long()
    w = bary[mask]
    out[mask] = (
        attr[tri[:, 0]] * w[:, 0:1]
        + attr[tri[:, 1]] * w[:, 1:2]
        + attr[tri[:, 2]] * w[:, 2:3]
    )
    return out
