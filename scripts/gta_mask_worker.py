"""Persistent offline SAM2/DINOv2 worker for reviewed boxes; JSON lines over SSH."""
import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel, Sam2Model, Sam2Processor


def main():
    torch.set_num_threads(4)
    models = json.loads(Path('/mnt/hdd/agentic-world-models/models.json').read_text())
    sp = Sam2Processor.from_pretrained(models['segmenter']['path'],local_files_only=True)
    sam = Sam2Model.from_pretrained(models['segmenter']['path'],local_files_only=True).to('cuda').eval()
    ep = AutoImageProcessor.from_pretrained(models['embedding']['path'],local_files_only=True)
    encoder = AutoModel.from_pretrained(models['embedding']['path'],local_files_only=True).to('cuda').eval()
    print('GTA_READY',flush=True)
    for line in sys.stdin:
        try:
            job = json.loads(line)
            image = Image.open(io.BytesIO(base64.b64decode(job['image']))).convert('RGB')
            boxes = job['boxes']
            inputs = sp(images=image,input_boxes=[boxes],return_tensors='pt').to('cuda')
            with torch.inference_mode():
                out = sam(**inputs,multimask_output=False)
            masks = sp.post_process_masks(out.pred_masks.cpu(),inputs['original_sizes'].cpu())[0][:,0].numpy().astype(bool)
            scores = out.iou_scores.cpu().flatten().tolist()
            crops = []
            result = []
            for box,mask,score in zip(boxes,masks,scores):
                data = np.array(image)
                data[~mask] = 127
                crop = Image.fromarray(data).crop(tuple(map(int,box)))
                crops.append(crop)
                buffer = io.BytesIO()
                Image.fromarray(mask.astype(np.uint8)*255).save(buffer,format='PNG')
                result.append({'mask_png':base64.b64encode(buffer.getvalue()).decode(),
                               'predicted_iou':score,'area':int(mask.sum())})
            features = ep(images=crops,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                embedding = torch.nn.functional.normalize(encoder(**features).last_hidden_state[:,0].float(),dim=-1).cpu().numpy()
            for item,vector in zip(result,embedding):
                item['embedding'] = vector.tolist()
            print('GTA_RESULT '+json.dumps({'objects':result}),flush=True)
        except Exception as error:
            print('GTA_RESULT '+json.dumps({'error':type(error).__name__}),flush=True)


if __name__ == '__main__':
    main()
