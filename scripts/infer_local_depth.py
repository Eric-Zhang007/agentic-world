"""Fresh relative-depth inference using an explicit existing native checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--encoder',choices=['vits','vitb'],default='vits')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--input-size',type=int,default=518)
    a=p.parse_args()
    if not a.checkpoint.is_file() or not (a.repo/'depth_anything_v2/dpt.py').is_file():
        raise ValueError('Explicit local source and checkpoint required')
    if a.gpu<0 or a.input_size<14:raise ValueError('Invalid GPU or input size')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(a.repo.resolve()))
    started=time.perf_counter()
    import cv2
    import numpy as np
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2
    if not torch.cuda.is_available():raise RuntimeError('Selected CUDA device unavailable')
    configs={'vits':dict(features=64,out_channels=[48,96,192,384]),
             'vitb':dict(features=128,out_channels=[96,192,384,768])}
    model=DepthAnythingV2(encoder=a.encoder,**configs[a.encoder])
    model.load_state_dict(torch.load(str(a.checkpoint),map_location='cpu',weights_only=True))
    model=model.to('cuda').eval()
    torch.cuda.synchronize()
    load_seconds=time.perf_counter()-started
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(cached_predictions=False,checkpoint=str(a.checkpoint.resolve()),
        source_repo=str(a.repo.resolve()),encoder=a.encoder,input_size=a.input_size,
        device=torch.cuda.get_device_name(0),torch_version=torch.__version__,
        load_seconds=load_seconds,frames=[],world_committed=False,
        scope='Fresh relative-depth model output, not metric depth, 3D reconstruction or world-state acceptance.')
    for i,path in enumerate(a.image):
        image=cv2.imread(str(path))
        if image is None:raise ValueError('Cannot read image: '+str(path))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();begin=time.perf_counter()
        with torch.inference_mode():prediction=model.infer_image(image,a.input_size)
        torch.cuda.synchronize();seconds=time.perf_counter()-begin
        if prediction.shape!=image.shape[:2] or not np.isfinite(prediction).all():
            raise ValueError('Invalid model prediction')
        folder=a.output/f'{i:02d}_{path.parent.name}';folder.mkdir()
        np.save(folder/'relative_depth.npy',prediction)
        low,high=float(prediction.min()),float(prediction.max())
        if high<=low:raise ValueError('Model returned constant depth')
        normalized=np.round((prediction-low)/(high-low)*255).astype(np.uint8)
        preview=cv2.applyColorMap(normalized,cv2.COLORMAP_TURBO)
        cv2.imwrite(str(folder/'depth.png'),preview)
        cv2.imwrite(str(folder/'comparison.jpg'),np.hstack([image,preview]))
        report['frames'].append(dict(image=str(path.resolve()),output=str(folder.resolve()),
            seconds=seconds,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            minimum=low,maximum=high,shape=list(prediction.shape)))
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
    del model
    torch.cuda.empty_cache()
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
