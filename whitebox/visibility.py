"""Coarse ray/box visibility evidence against opaque static proxy boxes."""
import numpy as np
from whitebox.image_alignment import pixel_rays
from whitebox.asset_geometry import local_vertices


def box_intersections(eye,rays,entity,near=.05):
    """First positive ray parameter for a Y-rotated proxy box, or infinity."""
    center=np.asarray(entity['position'],float).copy()
    yaw=entity.get('rotation_y',0);c,s=np.cos(yaw),np.sin(yaw)
    rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    vertices=local_vertices(entity)
    low_corner,high_corner=vertices.min(axis=0),vertices.max(axis=0)
    size=high_corner-low_corner
    center += rotation @ ((low_corner+high_corner)/2)
    origin=(eye-center)@rotation
    direction=np.asarray(rays)@rotation
    parallel=np.abs(direction)<1e-12
    outside=np.any(parallel & (np.abs(origin)>size/2),axis=1)
    low=np.full_like(direction,-np.inf);high=np.full_like(direction,np.inf)
    np.divide(-size/2-origin,direction,out=low,where=~parallel)
    np.divide(size/2-origin,direction,out=high,where=~parallel)
    entry=np.maximum(np.minimum(low,high).max(axis=1),near)
    exit=np.maximum(low,high).min(axis=1)
    valid=(~outside)&(exit>=entry)&np.isfinite(direction).all(axis=1)
    return np.where(valid,entry,np.inf)


def inspect_visibility(scene,world_id,mask,stride=4):
    result = inspect_group_visibility(scene, [world_id], mask, stride)
    result['world_id'] = world_id
    return result


def inspect_group_visibility(scene,world_ids,mask,stride=4):
    """Only test observed pixels that also intersect target proxy volume.

    Not exact mesh rendering: actor articulation, terrain and non-box blockers
    are excluded. Reports blocker identities rather than auto-editing them.
    """
    if type(stride)is not int or stride<1:raise ValueError('Positive integer stride required')
    if not isinstance(world_ids, list) or not world_ids or len(set(world_ids)) != len(world_ids):
        raise ValueError('Nonempty unique target member IDs required')
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2:raise ValueError('2D mask required')
    h,w=mask.shape
    y,x=np.where(mask[::stride,::stride])
    pixels=np.column_stack([x*stride+.5,y*stride+.5])
    entities={e['id']:e for e in scene['entities']}
    targets=[entities[identity] for identity in world_ids]
    if any(target['kind'] not in ('actor','box') for target in targets):
        return dict(status='UNSUPPORTED_TARGET',member_ids=world_ids)
    eye,rays=pixel_rays(pixels,scene['camera'],w,h)
    near=scene['camera'].get('near',.05)
    distance=np.minimum.reduce([box_intersections(eye,rays,target,near) for target in targets])
    comparable=np.isfinite(distance)
    nearest=distance.copy();blocker=np.full(len(pixels),'',dtype=object)
    for entity in scene['entities']:
        if entity['id'] in world_ids or entity['kind']!='box':continue
        hit=box_intersections(eye,rays,entity,near)
        closer=comparable & (hit<nearest-1e-6)
        nearest[closer]=hit[closer];blocker[closer]=entity['id']
    blocked=blocker!=''
    names,counts=np.unique(blocker[blocked],return_counts=True)
    return dict(status='MEASURED_COARSE_VISIBILITY' if comparable.any() else 'NO_TARGET_OVERLAP',
        member_ids=list(world_ids),sampled_pixels=len(pixels),comparable_pixels=int(comparable.sum()),
        blocked_pixels=int(blocked.sum()),
        blocked_fraction=float(blocked.sum()/comparable.sum()) if comparable.any() else None,
        blockers={str(k):int(v) for k,v in zip(names,counts)},
        blocked_image_pixels=pixels[blocked].tolist(),
        scope='Observed mask pixels overlapping coarse target volume; opaque static boxes only. Not exact mesh or full-world visibility.')
