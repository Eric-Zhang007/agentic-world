"""Lift an annotated upright box using an explicit horizontal support prior."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from whitebox.image_alignment import pixels_to_plane
from whitebox.scene_refinement import apply_proposal
from scripts.fit_scene_landmarks import fit_scene


def propose(scene, annotation):
    """Produce a compatible inferred-static proposal, never an automatic commit."""
    if not annotation.get('evidence') or not annotation.get('uncertainty'):
        raise ValueError('Image lifting requires evidence and explicit uncertainty')
    identity=annotation['id']
    if any(e['id']==identity for e in scene['entities']):
        raise ValueError('Image extension must use a new identity')
    root,valid=pixels_to_plane([annotation['base_pixel']],scene['camera'],
        annotation['width'],annotation['height'],offset=-annotation['support_height'])
    if not valid[0]:raise ValueError('No forward ground intersection')
    size=np.asarray(annotation['size_prior'],float)
    if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):
        raise ValueError('Positive dimensions required')
    position=root[0]+[0,size[1]/2,0]
    entity=dict(id=identity,kind='box',category=annotation['category'],
        position=position.tolist(),size=size.tolist(),rotation_y=annotation.get('rotation_y',0),
        color=annotation.get('color',[.6,.6,.6]),provenance='INFERRED',
        reason='Annotated RGB primitive with explicit support-plane and depth priors')
    candidate=deepcopy(scene)
    candidate['entities'].append(entity)
    data=dict(width=annotation['width'],height=annotation['height'],
        support_heights={identity:annotation['support_height']},
        landmarks=[dict(id=identity,**row) for row in annotation['landmarks']],
        parameters=[dict(id=identity,**row) for row in annotation['parameters']])
    fitted,report=fit_scene(candidate,data)
    addition=fitted['entities'][-1]
    proposal=dict(evidence=annotation['evidence'],uncertainty=annotation['uncertainty'],
                  additions=[dict(entity=addition,evidence=annotation['evidence'])])
    checked=apply_proposal(scene,proposal)
    if checked['entities'][:-1]!=scene['entities'] or checked['camera']!=scene['camera']:
        raise AssertionError('Primitive proposal changed preexisting geometry or camera')
    report.update(world_committed=False,source_annotation=annotation,
        prior_support_height=annotation['support_height'],
        fitted_bottom_height=addition['position'][1]-addition['size'][1]/2,
        limitation='Single-view box proxy; annotations fit only visible anchors. Depth, occlusion, contact and historical consistency need independent review.')
    return checked,proposal,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','annotation','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    scene=json.loads(args.scene.read_text())
    annotation=json.loads(args.annotation.read_text())
    started=time.perf_counter()
    candidate,proposal,report=propose(scene,annotation)
    report['total_compute_seconds']=time.perf_counter()-started
    args.output.mkdir(parents=True,exist_ok=False)
    for name,value in [('scene',candidate),('proposal',proposal),('report',report)]:
        (args.output/(name+'.json')).write_text(json.dumps(value,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
