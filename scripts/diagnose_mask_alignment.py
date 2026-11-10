"""Inspect explicit mask/world associations without changing either world."""
import argparse
import json
from pathlib import Path
import sys
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from whitebox.mask_alignment import compare_mask, compare_group_mask
from whitebox.visibility import inspect_visibility, inspect_group_visibility
from whitebox.motion import validate_geometry


def diagnose(scene, perception, mapping, bundle_dir):
    """Report explicit associations; unreviewed entities are not deemed missing."""
    def index(rows, key, label):
        indexed = {}
        for row in rows:
            identity = row[key]
            if identity in indexed:
                raise ValueError(f'Duplicate {label} ID: {identity!r}')
            indexed[identity] = row
        return indexed

    entities=index(scene['entities'], 'id', 'world')
    observations=index(perception['observations'], 'observation_id', 'observation')
    groups = {}
    for obs_id,value in mapping.items():
        if obs_id not in observations:
            raise ValueError(f'Unknown mapping observation ID: {obs_id!r}')
        members = [value] if isinstance(value, str) else value
        if not isinstance(members, list) or not members or any(not isinstance(m,str) for m in members) or len(set(members)) != len(members):
            raise ValueError('Mapping requires a world ID or nonempty unique member-ID list')
        for world_id in members:
            if world_id not in entities:
                raise ValueError(f'Unknown mapping world ID: {world_id!r}')
        groups[obs_id] = members
    bundle_dir=Path(bundle_dir)
    image=cv2.imread(str(bundle_dir/'input.png'))
    if image is None:raise ValueError('Perception source image missing')
    results=[]
    for obs_id,members in groups.items():
        row=observations[obs_id]
        # Local bundles retain their mask basenames across Windows/WSL hosts.
        path=bundle_dir/Path(row['mask_path']).name
        mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape!=image.shape[:2]:raise ValueError('Mask/image mismatch')
        if len(members)==1:
            report=compare_mask(entities[members[0]],scene['camera'],mask>0)
            report['visibility']=inspect_visibility(scene,members[0],mask>0)
        else:
            report=compare_group_mask([entities[m] for m in members],scene['camera'],mask>0)
            report['visibility']=inspect_group_visibility(scene,members,mask>0)
        report['member_ids']=list(members)
        report.update(observation_id=obs_id,association_source='explicit_agent_mapping')
        results.append(report)
        for key,color in [('observed_bbox',(0,255,0)),('clipped_projected_bbox',(0,0,255))]:
            if key not in report:continue
            x1,y1,x2,y2=map(round,report[key]);cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
            cv2.putText(image,obs_id+' '+key,(x1,max(15,y1)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    geometry_errors=validate_geometry(scene)
    unmapped=[identity for identity in observations if identity not in mapping]
    reviewed={member for members in groups.values() for member in members}
    unreviewed=[identity for identity in entities if identity not in reviewed]
    return dict(results=results,geometry_errors=geometry_errors,world_committed=False,
                unmapped_observation_ids=unmapped,unreviewed_world_ids=unreviewed,
                counts=dict(observations=len(observations),world_entities=len(entities),
                            mapped_observations=len(mapping),reviewed_world_entities=len(reviewed),
                            unmapped_observations=len(unmapped),unreviewed_world_entities=len(unreviewed))),image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','perception','mapping','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    scene=json.loads(a.scene.read_text());perception=json.loads(a.perception.read_text())
    mapping=json.loads(a.mapping.read_text())
    report,image=diagnose(scene,perception,mapping,a.perception.parent)
    a.output.mkdir(parents=True,exist_ok=False)
    cv2.imwrite(str(a.output/'overlay.jpg'),image)
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    summary=[]
    for row in report['results']:
        summary.append({**row,'visibility':{k:v for k,v in row['visibility'].items() if k!='blocked_image_pixels'}})
    print(json.dumps({**report,'results':summary},indent=2))


if __name__=='__main__':main()
