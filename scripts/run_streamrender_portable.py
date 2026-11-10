"""Offline semantic replay through the original two-step streaming H3 pipeline."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time


def main(args):
    sys.path[:0] = [str(args.source),str(args.source/'render/vendor'),str(args.source/'render/decoder')]
    import torch
    import torch.distributed as dist
    from dev.yanzuolu.common.config import CfgNode
    from dev.yanzuolu.common.distributed.init import configure_distributed, destroy_distributed
    from dev.yanzuolu.common.distributed.unified_parallel import init_unified_parallel,use_unified_parallel
    from dev.yanzuolu.common.phase import ExecutionPhase,execution_phase
    from dev.yanzuolu.common.model import build_models
    from runtime.config import load_config
    from streamrender_portable import install_decoder, install_attention

    settings,assets = load_config(args.config,args.assets)
    cfg = CfgNode(settings['inference'])
    configure_distributed(cfg)
    init_unified_parallel((1,4),1)
    rank = dist.get_rank()
    torch.set_num_threads(4)
    install_decoder()
    install_attention()
    from render.pipeline import StreamingPipeline
    from PIL import Image
    import numpy as np
    import av
    start = time.monotonic()
    models = build_models(cfg.models)
    # FSDP materializes BF16 parameters without downcasting coordinate inputs.
    models['backbone'].forward = torch.autocast('cuda', dtype=torch.bfloat16)(models['backbone'].forward)
    pipe = StreamingPipeline(cfg,models,settings)
    if rank == 0:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'run.json').write_text(json.dumps({
            'scope':'Original StreamRender student; BF16 FSDP4, native PyTorch decoder, no FA4 speed claim',
            'source':str(args.source),'config':str(args.config),'assets':str(args.assets),
            'video':str(args.video),'rounds':args.rounds,'torch':torch.__version__,
            'load_seconds':time.monotonic()-start},indent=2))
    try:
        with execution_phase(ExecutionPhase.VALIDATION),use_unified_parallel(4),torch.no_grad():
            pipe.start(prompt=assets['prompt'],picture=assets['reference_image'],seed=cfg.seed)
            reader = av.open(str(args.video))
            frames = iter(reader.decode(video=0))
            writer = None
            if rank == 0:
                writer = av.open(str(args.output/'generated.mp4'),'w')
                stream = writer.add_stream('libx264',rate=24)
                stream.width=settings['output']['width'];stream.height=settings['output']['height']
                stream.pix_fmt='yuv420p';stream.options={'crf':'18'}
            try:
                for index in range(args.rounds):
                    _,count = pipe.next_counts()
                    batch=[]
                    for _ in range(count):
                        frame=next(frames,None)
                        if frame is None:raise ValueError('Semantic source ends before requested complete chunk')
                        batch.append(frame.to_ndarray(format='rgb24'))
                    rgb=torch.from_numpy(np.stack(batch)).to(pipe.device).permute(3,0,1,2).float().div_(255)
                    rgb=torch.nn.functional.interpolate(rgb.permute(1,0,2,3),
                        size=(settings['reference']['height'],settings['reference']['width']),
                        mode='nearest').permute(1,0,2,3).contiguous()
                    generated,metrics=pipe.render(rgb)
                    if rank==0:
                        print(json.dumps(metrics),flush=True)
                        with (args.output/'timings.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
                        if generated is not None:
                            array=generated[0].permute(1,2,3,0).clamp(0,1).mul(255).byte().cpu().numpy()
                            for pixels in array:
                                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')):writer.mux(packet)
                            Image.fromarray(array[-1]).save(args.output/f'chunk_{index:03d}_last.png')
                if rank == 0:
                    final = pipe.decoder.flush().video
                    count = 0 if final is None else int(final.shape[2])
                    if final is not None:
                        array=final[0].permute(1,2,3,0).clamp(0,1).mul(255).byte().cpu().numpy()
                        for pixels in array:
                            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')):writer.mux(packet)
            finally:
                reader.close()
                if writer:
                    for packet in stream.encode():writer.mux(packet)
                    writer.close()
            if rank == 0:
                (args.output/'completion.json').write_text(json.dumps({'flush_frames':count,'rounds':args.rounds}))
    finally:
        destroy_distributed()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','config','assets','video','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rounds',type=int,default=4)
    logging.basicConfig(level=logging.INFO)
    main(p.parse_args())
