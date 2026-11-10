"""Image-space bounds diagnostics for explicitly associated world entities."""
import numpy as np
from whitebox.image_alignment import project_points
from whitebox.asset_geometry import local_vertices


def projected_bounds(entity,camera,width,height):
    if entity['kind'] not in ('box','actor','stairs'):
        return None
    corners=local_vertices(entity)
    yaw=entity.get('rotation_y',0);c,s=np.cos(yaw),np.sin(yaw)
    rotation=np.asarray([[c,0,s],[0,1,0],[-s,0,c]])
    pixels,valid=project_points(corners@rotation.T+entity['position'],camera,width,height)
    if not valid.all():return None
    return np.r_[pixels.min(axis=0),pixels.max(axis=0)]


def compare_mask(entity,camera,mask):
    return _compare_bounds(mask,lambda w,h:projected_bounds(entity,camera,w,h),
                           dict(world_id=entity['id']))


def compare_group_mask(entities,camera,mask):
    entities=list(entities)
    member_ids=[entity.get('id') for entity in entities]
    if not member_ids or any(not isinstance(value,str) or not value.strip() for value in member_ids):
        raise ValueError('Group members require nonempty entity IDs')
    if len(set(member_ids))!=len(member_ids):
        raise ValueError('Group member IDs must be unique')

    def bounds(w,h):
        projected=[projected_bounds(entity,camera,w,h) for entity in entities]
        if any(value is None for value in projected):return None
        return np.r_[np.min([value[:2] for value in projected],axis=0),
                     np.max([value[2:] for value in projected],axis=0)]

    report=_compare_bounds(mask,bounds,dict(member_ids=member_ids))
    report['member_ids']=member_ids
    return report


def _compare_bounds(mask,bounds,identity):
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2:raise ValueError('Mask must be 2D')
    h,w=mask.shape
    y,x=np.where(mask)
    if not len(x):return dict(status='EMPTY_MASK')
    observed=np.array([x.min(),y.min(),x.max()+1,y.max()+1],float)
    predicted=bounds(w,h)
    report=dict(status='MEASURED_2D_BOUNDS',**identity,
        observed_bbox=observed.tolist(),mask_area=int(mask.sum()),
        cropped_edges=dict(left=bool(x.min()==0),right=bool(x.max()==w-1),
                           top=bool(y.min()==0),bottom=bool(y.max()==h-1)),
        scope='Visible mask bbox versus unoccluded coarse entity bounds. Not silhouette IoU, pose correctness, or identity matching.')
    if predicted is None:
        report['status']='UNAVAILABLE_PROJECTION';return report
    visible=predicted.copy();visible[[0,2]]=np.clip(visible[[0,2]],0,w);visible[[1,3]]=np.clip(visible[[1,3]],0,h)
    intersection=np.maximum(0,np.minimum(observed[2:],visible[2:])-np.maximum(observed[:2],visible[:2])).prod()
    union=np.prod(observed[2:]-observed[:2])+np.prod(visible[2:]-visible[:2])-intersection
    report.update(projected_bbox=predicted.tolist(),clipped_projected_bbox=visible.tolist(),
        bbox_iou=float(intersection/union) if union>0 else 0.,
        center_error_px=float(np.linalg.norm((observed[:2]+observed[2:]-visible[:2]-visible[2:])/2)))
    return report
