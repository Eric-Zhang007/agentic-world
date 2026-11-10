"""Fresh native SAM2 masks from explicit agent box prompts, not detection."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','checkpoint','requests','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--dependencies',type=Path)
    p.add_argument('--model-config',default='configs/sam2/sam2_hiera_t.yaml')
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args()
    if not a.checkpoint.is_file() or not (a.repo/'sam2/build_sam.py').is_file():
        raise ValueError('Existing native source and weights required')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES=str(a.gpu))
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(a.repo.resolve()))
    if a.dependencies:sys.path.insert(0,str(a.dependencies.resolve()))
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not torch.cuda.is_available():raise RuntimeError('Selected CUDA device unavailable')
    requests=json.loads(a.requests.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter()
    model=build_sam2(a.model_config,str(a.checkpoint),device='cuda',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model)
    torch.cuda.synchronize()
    report=dict(model='SAM2 native',checkpoint=str(a.checkpoint.resolve()),
        cached_predictions=False,load_seconds=time.perf_counter()-begin,images=[],
        world_committed=False,scope='Agent-specified regions and categories; no automatic discovery, identity matching, or metric depth.')
    for index,request in enumerate(requests['images']):
        image=cv2.imread(request['image'])
        if image is None:raise ValueError('Cannot read '+request['image'])
        h,w=image.shape[:2]
        boxes=np.asarray([r['bbox_xyxy'] for r in request['objects']],dtype=float)
        if boxes.ndim!=2 or boxes.shape[1]!=4 or not np.isfinite(boxes).all():
            raise ValueError('Expected finite nonempty XYXY box prompts')
        if np.any(boxes[:,2:]<=boxes[:,:2]) or np.any(boxes<0) or np.any(boxes[:,[0,2]]>w) or np.any(boxes[:,[1,3]]>h):
            raise ValueError('Prompt boxes must be inside image')
        out=a.output/f'{index:02d}';out.mkdir()
        cv2.imwrite(str(out/'input.png'),image)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            torch.cuda.synchronize();encode_seconds=time.perf_counter()-begin
            masks=[];scores=[];begin=time.perf_counter()
            for box in boxes:
                mask,score,_=predictor.predict(box=box,multimask_output=False)
                masks.append(mask[0].astype(bool));scores.append(float(score[0]))
            torch.cuda.synchronize();decode_seconds=time.perf_counter()-begin
        overlay=image.copy();observations=[]
        for i,(row,mask,score) in enumerate(zip(request['objects'],masks,scores)):
            if mask.shape!=(h,w):raise ValueError('Mask dimensions differ from source')
            path=out/f'mask_{i:03d}.png'
            cv2.imwrite(str(path),mask.astype(np.uint8)*255)
            color=np.array([(53+i*67)%255,(180+i*41)%255,(90+i*83)%255])
            overlay[mask]=(overlay[mask]*.6+color*.4).astype(np.uint8)
            x1,y1,x2,y2=map(int,row['bbox_xyxy'])
            cv2.rectangle(overlay,(x1,y1),(x2,y2),color.tolist(),2)
            cv2.putText(overlay,row['id'],(x1,max(15,y1)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            observations.append(dict(observation_id=row['id'],track_id=None,category=row['category'],
                bbox_xyxy=row['bbox_xyxy'],mask_path=str(path.resolve()),mask_area=int(mask.sum()),
                mask_predicted_iou=score,confidence=None,category_source='AGENT_BOX_PROMPT',
                frame_id=0,timestamp=None,embedding_path=None,depth=None,world_position=None,
                source='native-sam2-hiera-tiny+agent-box'))
        cv2.imwrite(str(out/'overlay.png'),overlay)
        result=dict(image=request['image'],image_size=[w,h],observations=observations,
            evidence=request['evidence'],encode_seconds=encode_seconds,decode_seconds=decode_seconds,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,cached_predictions=False)
        (out/'result.json').write_text(json.dumps(result,indent=2))
        report['images'].append(result)
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        predictor.reset_predictor()
    del predictor,model
    torch.cuda.empty_cache()
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
