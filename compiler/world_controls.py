"""Coarse world-derived controls. Unknown background is not reconstructed geometry."""
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from compiler.projection import project_object
from state.models import WorldObject


COLORS = [(255, 0, 0), (0, 180, 0), (0, 0, 255), (255, 180, 0), (180, 0, 180)]


def coarse_object(obj):
    if obj.bbox_3d is not None:
        return obj
    sizes = {"person": [.6, .45, 1.75], "vehicle": [1.8, 4.2, 1.5]}
    if obj.category not in sizes:
        return None
    return replace(obj, bbox_3d=sizes[obj.category], attributes={**obj.attributes,
        "geometry_provenance": "INFERRED_category_size_not_measured"})


def render_controls(objects, camera, ids, near=.5, far=30.0):
    """Ray-box depth in camera-z units; fixed clip range avoids temporal pumping."""
    w,h = camera["width"],camera["height"]
    layout = np.full((h,w,3),255,np.uint8)
    proxy = np.zeros((h,w,3),np.uint8)
    depth = np.full((h,w),np.inf,np.float32)
    projections = {}
    transform = np.asarray(camera["world_to_camera"],float)
    inverse = np.linalg.inv(transform)
    origin = inverse[:3,3]
    kinv = np.linalg.inv(camera["intrinsics"])
    for original in objects:
        if not original.state.get("matchable",True) or not original.state.get("visible",True):
            continue
        if not original.state.get("control_enabled",True) or original.attributes.get("control_level")=="identity_only":
            continue
        obj = coarse_object(original)
        if obj is None:
            continue
        p = project_object(obj,camera)
        if not p or not p.get("visible") or p.get("bbox") is None:
            continue
        projections[obj.object_id] = {**p,"bbox_provenance":obj.attributes.get("geometry_provenance","world_geometry")}
        box = p["bbox"]
        x0,y0 = max(0,int(np.floor(box[0]))),max(0,int(np.floor(box[1])))
        x1,y1 = min(w,int(np.ceil(box[2]))),min(h,int(np.ceil(box[3])))
        if x1<=x0 or y1<=y0:
            continue
        color = COLORS[ids.index(obj.object_id)%len(COLORS)]
        cv2.rectangle(layout,(x0,y0),(x1-1,y1-1),color,2)
        yy,xx = np.mgrid[y0:y1,x0:x1]
        rays = np.stack((xx+.5,yy+.5,np.ones_like(xx)),axis=-1) @ kinv.T
        rotation = Rotation.from_euler("xyz",obj.rotation or [0,0,0]).as_matrix()
        local_rays = rays @ inverse[:3,:3].T @ rotation
        local_origin = (origin-np.asarray(obj.position)) @ rotation
        half = np.asarray(obj.bbox_3d)*np.asarray(obj.scale or [1,1,1])/2
        safe = np.where(np.abs(local_rays)<1e-10,1e-10,local_rays)
        a,b = (-half-local_origin)/safe,(half-local_origin)/safe
        entry,exit = np.minimum(a,b).max(-1),np.maximum(a,b).min(-1)
        z = np.where(entry>near,entry,exit)
        valid = (exit>=entry)&(z>near)&(z<far)
        tile = depth[y0:y1,x0:x1]
        update = valid & (z<tile)
        tile[update] = z[update]
        proxy[y0:y1,x0:x1][update] = color
    known = np.isfinite(depth)
    depth_rgb = np.zeros((h,w,3),np.uint8)
    values = np.zeros((h,w),np.uint8)
    values[known] = np.clip(255*(1/depth[known]-1/far)/(1/near-1/far),1,255).astype(np.uint8)
    depth_rgb[:] = values[...,None]
    edges = np.repeat(cv2.Canny(proxy,80,160)[...,None],3,axis=2)
    gray = np.repeat(cv2.cvtColor(proxy,cv2.COLOR_RGB2GRAY)[...,None],3,axis=2)
    return {"layout":layout,"depth":depth_rgb,"proxy_canny":edges,"gray":gray}, projections, depth, known


def compile_timeline(timeline, output, frames=73, fps=24):
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    source_times = np.array([r["server_time_sec"] for r in timeline])
    if len(timeline)<2 or np.any(np.diff(source_times)<=0):
        raise ValueError("Timeline must have increasing timestamps")
    target = source_times[0]+np.arange(frames)/fps
    if target[-1]>source_times[-1]+1e-6:
        raise ValueError("Timeline does not cover requested duration; no padding")
    ids = sorted({o["object_id"] for r in timeline for o in r["objects"]})
    rendered = {kind:[] for kind in ("layout","depth","proxy_canny","gray")}
    expected=[];depths=[];known=[]
    for index,time in enumerate(target):
        # Previous-frame hold is explicit; camera/world share the same source time.
        row = timeline[max(0,int(np.searchsorted(source_times,time,side="right")-1))]
        objects=[WorldObject(**o) for o in row["objects"]]
        maps,projections,depth,mask=render_controls(objects,row["camera"],ids)
        for kind,value in maps.items():rendered[kind].append(value)
        depths.append(depth);known.append(mask)
        expected.append({"frame_id":index,"timestamp":index/fps,"source_time":row["server_time_sec"],
                         "camera":row["camera"],"objects":row["objects"],"projections":projections})
    from PIL import Image
    for kind,values in rendered.items():
        np.savez_compressed(output/f"{kind}.npz",rgb=np.stack(values))
        Image.fromarray(values[len(values)//2]).save(output/f"{kind}.png")
    np.savez_compressed(output/"geometry.npz",depth=np.stack(depths),known=np.stack(known))
    manifest={"fps":fps,"frames":frames,"source":"persistent_world_timeline",
        "unknown_background":"unconstrained; black depth background is a model-format approximation, not measured far depth",
        "geometry":"oriented coarse cuboids, person dimensions inferred when absent",
        "depth_encoding":"fixed inverse camera-z, near=.5 far=30; not a monocular depth calibration",
        "layout_encoding":"VACE-style 2-pixel RGB box outlines on white; stable object colors, not semantic-ID labels",
        "pose":"unavailable: no measured skeleton; not fabricated",
        "render_only_freedom":["texture","lighting","cloth","fine limb motion","unrecorded static background"],
        "expected":expected}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest
