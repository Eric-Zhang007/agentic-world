"""Real four-GPU forward check for the release's sharded causal DiT block."""
import argparse
import json
import os
from pathlib import Path
import sys


def main(source, output):
    sys.path.insert(0,str(source/'render/vendor'))
    import torch
    import torch.distributed as dist
    from dev.yanzuolu.common.config import CfgNode
    from dev.yanzuolu.common.distributed.init import configure_distributed,destroy_distributed
    from dev.yanzuolu.common.model.placement import DefaultPlacement
    from dev.yanzuolu.projects.minimax_h3.modeling.transformer.config import MiniMaxH3DiTArchConfig
    from dev.yanzuolu.projects.minimax_h3.modeling.transformer.causal_model import CausalMiniMaxH3DiTBlock
    cfg=CfgNode({'seed':123,'distributed':{'timeout_minutes':3}})
    configure_distributed(cfg)
    try:
        arch=MiniMaxH3DiTArchConfig(hidden_size=128,num_layers=1,num_attention_heads=2,
            attention_head_dim=64,ffn_hidden_size=256,adaln_out_features=2304,
            time_embed_dim=64,time_embed_hidden_size=128,rope_inv_freq_len=8)
        block=CausalMiniMaxH3DiTBlock(arch,None,prefix='test',layer_idx=0).eval().requires_grad_(False)
        # Exercise the real block via a small root so nested FSDP hooks are covered.
        class Root(torch.nn.Module):
            def __init__(self, child):super().__init__();self.block=child
            def forward(self,x,**kwargs):return self.block(x,**kwargs)
        model=Root(block)
        placement=DefaultPlacement(CfgNode({'fsdp':{'wrap_modules':['CausalMiniMaxH3DiTBlock'],
            'param_dtype':'bfloat16','reduce_dtype':'float32','shard_size':4}}))
        model=placement({'model':model,'name':'test'})['model']
        x=torch.randn(8,128,device='cuda',dtype=torch.bfloat16)
        print('SHARD_CONSTRUCTED',dist.get_rank(),flush=True)
        # The complete-model smoke will supply native rotary metadata. Here the
        # existing block's actual MLP verifies all-gather and reshard execution.
        def forward_mlp(self,x,**kwargs):return self.mlp(x)
        import types
        block.forward=types.MethodType(forward_mlp,block)
        with torch.no_grad():
            for _ in range(2):
                y=model(x)
                assert torch.isfinite(y).all()
        if dist.get_rank()==0:
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps({'passed':True,'gpus':4,'scope':'Actual causal block parameter sharding and MLP forward twice; not full attention/inference',
                'shape':list(y.shape),'peak_GiB':torch.cuda.max_memory_allocated()/2**30},indent=2))
    finally:destroy_distributed()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.output)
