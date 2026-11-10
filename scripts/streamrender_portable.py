"""Memory and decoder adapters for an unchanged StreamRender student on 4x48GB.

Imported only by the isolated reproduction, not by existing ComfyUI jobs.
"""
import json
from pathlib import Path

import torch

from dev.yanzuolu.common.model.weights import DefaultWeightLoader


class StrictIndexedWeights(DefaultWeightLoader):
    def __call__(self, state):
        path = Path(self.config.path)
        index = json.loads(path.read_text())
        expected = set(state['model'].state_dict())
        actual = set(self._remap_keys(state,index['weight_map'],log_decision=False))
        if expected != actual:
            raise ValueError(f'Checkpoint keys differ: missing={sorted(expected-actual)[:5]}, extra={sorted(actual-expected)[:5]}')
        return super().__call__(state)


class ExistingQwenWeights(StrictIndexedWeights):
    """Use existing upstream safetensors instead of a duplicate DCP download."""
    def _remap_keys(self, state, values, *, log_decision):
        keep = state['model'].should_materialize_checkpoint_weight
        return {name:value for name,value in values.items() if keep(name)}


class ReferenceDecoder:
    """Same distilled causal ViT24 weights and timeline, using PyTorch SDPA."""
    def __init__(self, vit):
        self.vit = vit
        self.state = None

    def init_state(self):
        self.state = self.vit.init_state()

    def decode_chunk(self, latents):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return self.vit.decode_chunk(latents,self.state)


def install_decoder():
    from causal_vit_codec import CausalViTVideoDecoder

    def reference_for(self, height, width, device):
        key = (height,width,str(device))
        if key not in self._fast:
            self.vit.to(device)
            self._fast[key] = ReferenceDecoder(self.vit)
        return self._fast[key]

    CausalViTVideoDecoder.fast_for = reference_for


def install_attention():
    import dev.yanzuolu.utils.flex_attn as attention
    import dev.yanzuolu.utils.flash_attn as flash
    compiled = attention.flex_attention

    def ada_attention(*args, **kwargs):
        kwargs['kernel_options'] = {'BLOCK_M': 32, 'BLOCK_N': 32, 'num_stages': 2}
        return compiled(*args, **kwargs)

    attention.flex_attention = ada_attention
    flash._sdpa_varlen = packed_sdpa


def packed_sdpa(q, k, v, *, q_bounds, k_bounds, dropout_p, softmax_scale,
                q_scale, causal, window_size):
    """Preserve packed boundaries while making fused 4D SDPA available."""
    from dev.yanzuolu.utils.flash_attn import _bottom_right_mask
    out = torch.empty_like(q)
    for qs, qe, ks, ke in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]):
        if qs == qe:
            continue
        query = q[qs:qe].transpose(0, 1).unsqueeze(0)
        if q_scale is not None:
            query = query * q_scale
        mask = _bottom_right_mask(qe-qs, ke-ks, causal=causal,
                                  window_size=window_size, device=q.device)
        result = torch.nn.functional.scaled_dot_product_attention(
            query, k[ks:ke].transpose(0, 1).unsqueeze(0),
            v[ks:ke].transpose(0, 1).unsqueeze(0), attn_mask=mask,
            dropout_p=dropout_p, is_causal=causal and mask is None,
            scale=softmax_scale)
        out[qs:qe].copy_(result[0].transpose(0, 1))
    return out


def make_config(source, weights, output):
    config = json.loads((source/'configs/runtime.json').read_text())
    models = config['inference']['models']
    models['backbone']['weight'] = {'module':'streamrender_portable','class_name':'StrictIndexedWeights',
                                    'path':str(weights/'backbone/model.safetensors.index.json'),
                                    'assign':True}
    models['backbone']['placement'] = {'fsdp':{
        'wrap_modules':['CausalMiniMaxH3DiTBlock'], 'param_dtype':'bfloat16',
        'reduce_dtype':'float32','shard_size':4}}
    qwen = models['text_encoder']
    qwen['weight'] = {'module':'streamrender_portable','class_name':'ExistingQwenWeights',
        'path':'/mnt/hdd/MiniMax-H3/Ref2VA/text_encoder/model.safetensors.index.json','assign':True}
    qwen['placement'].pop('plugins',None)
    config['warmup_rounds'] = 0
    config['inference']['distributed']['up_size'] = [1,4]
    config['output_dir'] = str(output/'runtime')
    assets = json.loads((source/'configs/assets.example.json').read_text())
    assets['assets'].update({
        'backbone':str(weights/'backbone/model.safetensors.index.json'),
        'qwen_model':'/mnt/hdd/MiniMax-H3/Ref2VA/text_encoder',
        'qwen_dcp':'/mnt/hdd/MiniMax-H3/Ref2VA/text_encoder',
        'qwen_processor':'/mnt/hdd/MiniMax-H3/Ref2VA/processor',
        'tokenizer':'/mnt/hdd/MiniMax-H3/Ref2VA/tokenizer',
        'tae':str(weights/'tae/taeh3_official.pth'),
        'causal_decoder':str(weights/'causal_decoder/causal_vit24_step6000.safetensors'),
        'audio_vae':'/mnt/hdd/MiniMax-H3/Ref2VA/audio_vae/model.safetensors',
    })
    return config,assets
