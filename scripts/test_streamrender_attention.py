"""Check Ada-sized FlexAttention tiles against masked SDPA on real CUDA."""
import argparse
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
args = p.parse_args()
sys.path.insert(0, str(args.source / 'render/vendor'))
import torch
from torch.nn.attention.flex_attention import create_block_mask
from streamrender_portable import install_attention
import dev.yanzuolu.utils.flex_attn as attention
from dev.yanzuolu.utils.flash_attn import _sdpa_varlen as original_packed
from streamrender_portable import packed_sdpa
from torch.nn.attention import sdpa_kernel, SDPBackend

install_attention()
torch.manual_seed(3)
reports = []
for length in (257, 1024):
    q, k, v = [torch.randn(1, 2, length, 256, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    def causal(b, h, i, j):
        return i >= j
    mask = create_block_mask(causal, 1, 1, length, length, device='cuda')
    with torch.no_grad():
        actual = attention.flex_attention(q, k, v, block_mask=mask)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    error = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
    reports.append(dict(length=length, head_dim=256, maximum_absolute_error=error))
q = torch.randn(12, 4, 256, device='cuda', dtype=torch.bfloat16)
k = torch.randn(18, 4, 256, device='cuda', dtype=torch.bfloat16)
v = torch.randn_like(k)
for causal, window in [(False, (-1, -1)), (True, (-1, -1)), (True, (3, 0))]:
    kw = dict(q_bounds=[0, 5, 12], k_bounds=[0, 8, 18], dropout_p=0.,
              softmax_scale=.0625, q_scale=.8, causal=causal, window_size=window)
    actual, expected = packed_sdpa(q, k, v, **kw), original_packed(q, k, v, **kw)
    torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
    reports.append(dict(packed=True, causal=causal, window=window,
                        maximum_absolute_error=(actual-expected).abs().max().item()))
q = torch.randn(4096, 12, 256, device='cuda', dtype=torch.bfloat16)
k = torch.randn(8192, 12, 256, device='cuda', dtype=torch.bfloat16)
v = torch.randn_like(k)
torch.cuda.reset_peak_memory_stats()
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    actual = packed_sdpa(q, k, v, q_bounds=[0, 4096], k_bounds=[0, 8192],
                         dropout_p=0., softmax_scale=None, q_scale=None,
                         causal=False, window_size=(-1, -1))
torch.cuda.synchronize()
reports.append(dict(forced_flash=True, shape=list(actual.shape),
                    peak_GiB=torch.cuda.max_memory_allocated()/2**30))
print(json.dumps(reports, indent=2))
