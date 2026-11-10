import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image


class CropEncoder:
    def __init__(self, device="cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = device
        manifest = json.loads(Path("/mnt/hdd/agentic-world-models/models.json").read_text())
        path = manifest["embedding"]["path"]
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval().to(device)

    def encode_object_crop(self, image, mask, bbox):
        image = image.convert("RGB")
        if mask is not None:
            pixels = np.array(image)
            pixels[~np.asarray(mask, bool)] = 127
            image = Image.fromarray(pixels)
        x0, y0, x1, y1 = bbox
        bbox = (max(0, int(x0)), max(0, int(y0)),
                min(image.width, int(np.ceil(x1))), min(image.height, int(np.ceil(y1))))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        inputs = self.processor(images=image.crop(bbox), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            feature = self.model(**inputs).last_hidden_state[:, 0].float()
            feature = self.torch.nn.functional.normalize(feature, dim=-1)
        return feature[0].cpu().numpy()

    def close(self):
        del self.model, self.processor
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
