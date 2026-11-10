from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from perception.embeddings import CropEncoder

encoder = CropEncoder()
feature = encoder.encode_object_crop(
    Image.open("outputs/h3_smoke/proxy/reference.png"), None, [200, 240, 400, 320])
assert feature.shape == (384,) and np.isfinite(feature).all()
assert abs(np.linalg.norm(feature)-1) < 1e-5
encoder.close()
print("DINOv2 masked-crop interface: 384 dimensions, finite, unit norm")
