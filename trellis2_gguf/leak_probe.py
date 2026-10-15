"""TRELLIS2_LEAK_PROBE=1: diagnostics for the "only the first run after a ComfyUI start is
valid" bug (later runs: sparse-structure sampler gives NaN at step 0 with finite cond).

fingerprint() prints process-global state that could differ between run 1 and run 2;
watch_first_nan() hooks every module of a model and reports the first one whose output
goes non-finite during the next forward call, in execution order.
"""
import os
import hashlib
import torch

ENABLED = os.environ.get("TRELLIS2_LEAK_PROBE", "0") == "1"
_run = [0]


def _tstat(t):
    t = t.detach()
    f = t.float() if not t.is_complex() else torch.view_as_real(t).float()
    bad = int((~torch.isfinite(f)).sum())
    h = hashlib.sha256(f.cpu().contiguous().numpy().tobytes()).hexdigest()[:12]
    return f"{tuple(t.shape)} {t.dtype} {t.device} bad={bad} absmax={float(f.abs().max()):.6g} sha={h}"


def _param_digest(model, limit=None):
    """sha256 over every parameter/buffer's raw bytes (quantized storage as stored)."""
    h = hashlib.sha256()
    n, meta = 0, []
    tensors = [(f"{mn}.{pn}", t) for mn, mod in model.named_modules()
               for pn, t in list(mod._parameters.items()) + list(mod._buffers.items()) if t is not None]
    for name, t in tensors[:limit]:
        d = t.detach()
        if d.is_meta:
            meta.append(name)
            continue
        if d.is_complex():
            d = torch.view_as_real(d)
        h.update(name.encode())
        h.update(torch.Tensor.cpu(d).contiguous().view(torch.uint8).numpy().tobytes())
        n += 1
    return f"{h.hexdigest()[:16]} over {n} tensors; {len(meta)} meta: {meta[:12]}"


def fingerprint(tag, model=None, noise=None, cond=None):
    if not ENABLED:
        return
    _run[0] += 1
    from .modules.attention import config as dense_cfg
    from .modules.sparse import config as sparse_cfg
    p = lambda s: print(f"[LeakProbe #{_run[0]} {tag}] {s}", flush=True)
    p(f"dense BACKEND={dense_cfg.BACKEND} sparse ATTN={getattr(sparse_cfg, 'ATTN', None)} "
      f"CONV={getattr(sparse_cfg, 'CONV', None)} env ATTN_BACKEND={os.environ.get('ATTN_BACKEND')}")
    p(f"grad={torch.is_grad_enabled()} inference={torch.is_inference_mode_enabled()} "
      f"default_dtype={torch.get_default_dtype()} autocast={torch.is_autocast_enabled()} "
      f"tf32 matmul={torch.backends.cuda.matmul.allow_tf32} cudnn={torch.backends.cudnn.allow_tf32} "
      f"bench={torch.backends.cudnn.benchmark} prec={torch.get_float32_matmul_precision()} "
      f"sdp flash={torch.backends.cuda.flash_sdp_enabled()} mem={torch.backends.cuda.mem_efficient_sdp_enabled()} "
      f"math={torch.backends.cuda.math_sdp_enabled()} det={torch.are_deterministic_algorithms_enabled()}")
    p(f"F.sdpa={torch.nn.functional.scaled_dot_product_attention!r}")
    p(f"env TRELLIS2_*: " + " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("TRELLIS2")))
    if model is not None:
        p(f"model id={id(model):#x} cls={type(model).__name__} dtype={getattr(model, 'dtype', None)} "
          f"training={model.training} device={next(model.parameters()).device}")
        rp = getattr(model, "rope_phases", None)
        if rp is not None:
            p(f"rope_phases {_tstat(rp)}")
        try:
            p(f"params {_param_digest(model)}")
        except Exception as e:  # the probe must never break a run
            p(f"params digest failed: {e!r}")
        fwd = model.__dict__.get("forward")
        p(f"instance forward override={fwd!r} class forward={type(model).forward!r}")
    if noise is not None:
        p(f"noise {_tstat(noise)}")
    if isinstance(cond, dict):
        for k, v in cond.items():
            if isinstance(v, torch.Tensor):
                p(f"cond[{k}] {_tstat(v)}")
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, torch.Tensor):
                        p(f"cond[{k}][{kk}] {_tstat(vv)}")


def watch_first_nan(model, tag, max_reports=6, calls=2):
    """Hook every submodule; report the first non-finite outputs (and their inputs) for
    the next `calls` forward passes of `model` (CFG runs cond and neg_cond separately),
    then remove the hooks."""
    if not ENABLED:
        return
    reports, handles, seen = [], [], [0]

    def _first_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (tuple, list)):
            for y in x:
                t = _first_tensor(y)
                if t is not None:
                    return t
        if hasattr(x, "feats"):
            return x.feats
        return None

    def hook(mod, inp, out, name):
        if len(reports) >= max_reports:
            return
        t = _first_tensor(out)
        if t is None or not t.is_floating_point():
            return
        if bool(torch.isfinite(t).all()):
            return
        ins = [_tstat(i) for i in (inp if isinstance(inp, tuple) else (inp,))
               if isinstance(i, torch.Tensor)]
        own = {n: _tstat(p) for n, p in list(mod.named_parameters(recurse=False))[:3]}
        reports.append(f"call{seen[0]} {name} ({type(mod).__name__}) out={_tstat(t)} in={ins} params={own}")

    for name, mod in model.named_modules():
        handles.append(mod.register_forward_hook(lambda m, i, o, n=name or "<root>": hook(m, i, o, n)))

    def done(_mod, _inp, _out):
        seen[0] += 1
        if seen[0] < calls:
            return
        for h in handles:
            h.remove()
        root.remove()
        print(f"[LeakProbe #{_run[0]} {tag}] first non-finite modules (execution order): "
              + ("none" if not reports else ""), flush=True)
        for r in reports:
            print(f"    {r}", flush=True)

    root = model.register_forward_hook(done)
