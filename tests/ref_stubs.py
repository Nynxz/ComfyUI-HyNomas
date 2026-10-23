import diffusers, diffusers.models  # noqa: F401  (probe optional packages before the stubs exist)
"""Stub the reference's CUDA-only imports so its DiT runs on stock torch (flex backend)."""
import sys, types, logging
import torch
import torch.nn.functional as F

class _Missing:
    def __init__(self, name): self.name = name
    def __call__(self, *a, **k): raise RuntimeError(f"stub {self.name} called")
    def __getattr__(self, n): return _Missing(self.name + "." + n)


def _mod(name, **attrs):
    import importlib.machinery as _im
    m = types.ModuleType(name); m.__dict__.update(attrs); m.__path__ = []; m.__spec__ = _im.ModuleSpec(name, None, is_package=True)
    m.__getattr__ = lambda n, name=name: _Missing(f"{name}.{n}") if not n.startswith("__") else (_ for _ in ()).throw(AttributeError(n))
    sys.modules[name] = m; return m


import importlib.abc, importlib.machinery
STUB_ROOTS = {"flash_attn", "flash_attn_interface", "flash_attn_3", "magi_attention", "deep_gemm", "loguru", "flashinfer",
              "flash_attn_cute", "cutlass", "liger_kernel", "apex", "xfuser", "yunchang", "flex_block_attn", "kernels", "spas_sage_attn"}

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in STUB_ROOTS and fullname not in sys.modules:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
    def create_module(self, spec): return _mod(spec.name)
    def exec_module(self, module): pass

sys.meta_path.insert(0, _Finder())

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale, is_causal=causal)
    return o.transpose(1, 2)

def flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    out = torch.empty_like(q)
    for i in range(len(cu_q) - 1):
        a, b, c, d = int(cu_q[i]), int(cu_q[i + 1]), int(cu_k[i]), int(cu_k[i + 1])
        out[a:b] = flash_attn_func(q[None, a:b], k[None, c:d], v[None, c:d], softmax_scale=softmax_scale, causal=causal)[0]
    return out

def unpad_input(x, mask):
    m = mask.bool()
    seqlens = m.sum(-1, dtype=torch.int32)
    indices = torch.nonzero(m.flatten(), as_tuple=False).flatten()
    cu = F.pad(torch.cumsum(seqlens, 0, dtype=torch.int32), (1, 0))
    return x.flatten(0, 1)[indices], indices, cu, int(seqlens.max()), seqlens


def pad_input(x, indices, batch, seqlen):
    out = x.new_zeros(batch * seqlen, *x.shape[1:])
    out[indices] = x
    return out.view(batch, seqlen, *x.shape[1:])


fa = _mod("flash_attn", flash_attn_func=flash_attn_func, flash_attn_varlen_func=flash_attn_varlen_func, __version__="2.8.0")
_mod("flash_attn.flash_attn_interface", flash_attn_func=flash_attn_func, flash_attn_varlen_func=flash_attn_varlen_func,
     _flash_attn_forward=None)
_mod("flash_attn.bert_padding", pad_input=pad_input, unpad_input=unpad_input)
logger = logging.getLogger("ref")
_mod("loguru", logger=logger)
