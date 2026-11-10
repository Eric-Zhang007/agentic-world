"""Keep screen overlays out of evidence; inpainted pixels are never observations."""
import json
from pathlib import Path

import cv2
import numpy as np


def load_profile(path):
    data = json.loads(Path(path).read_text())
    for box in data["rectangles_normalized"]:
        if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError("HUD rectangles must be ordered normalized xyxy")
    return data


def invalid_mask(shape, profile, margin=0):
    h, w = shape[:2]
    mask = np.zeros((h,w), np.uint8)
    for x0,y0,x1,y1 in profile["rectangles_normalized"]:
        a,b = np.floor([x0*w,y0*h]).astype(int)
        c,d = np.ceil([x1*w,y1*h]).astype(int)
        mask[max(0,b-margin):min(h,d+margin),max(0,a-margin):min(w,c+margin)] = 255
    return mask


def clean_reference(image, mask):
    """Visual fill only; the separately persisted invalid mask remains authoritative."""
    return cv2.inpaint(np.asarray(image, np.uint8), mask, 5, cv2.INPAINT_TELEA)


def clean_canny(image, mask, low=100, high=200):
    # Compute before masking to avoid introducing a rectangular synthetic boundary.
    edge = cv2.Canny(np.asarray(image, np.uint8), low, high)
    expanded = cv2.dilate(mask, np.ones((7,7),np.uint8))
    edge[expanded>0] = 0
    return np.repeat(edge[...,None],3,axis=2)


def overlap_fraction(bbox, mask, object_mask=None):
    if object_mask is not None:
        pixels = np.asarray(object_mask)>0
        return float(np.mean(mask[pixels]>0)) if pixels.any() else 1.0
    x0,y0,x1,y1 = np.asarray(bbox,float)
    h,w = mask.shape
    a,b = np.maximum(np.floor([x0,y0]).astype(int),0)
    c,d = np.minimum(np.ceil([x1,y1]).astype(int),[w,h])
    return float(np.mean(mask[b:d,a:c]>0)) if c>a and d>b else 1.0
