"""Portable mixed scene/object reference data and bounded no-replacement sampling."""
from collections import defaultdict, Counter
import json
from pathlib import Path
import random

import cv2
import numpy as np


def balanced_round(rows, fractions, seed=0, max_scene_per_site=2):
    """Use the smallest task's capacity, never fabricate balance by duplication.

    Within a task, cycle over instance/site groups before revisiting a group.
    A round is deliberately small when the background pool is small.
    """
    if not fractions or any(v<=0 for v in fractions.values()) or abs(sum(fractions.values())-1)>1e-6:
        raise ValueError('Task fractions must be positive and sum to one')
    pool=defaultdict(list)
    for row in rows:
        pool[row['task']].append(row)
    missing=set(fractions)-set(pool)
    if missing:raise ValueError('Empty requested task pools: '+str(sorted(missing)))
    rng=random.Random(seed)
    if max_scene_per_site<1:raise ValueError('Scene site cap must be positive')
    scene_tasks={'scene_background','scene_passage','scene_nearby'} & pool.keys()
    sites=defaultdict(lambda:defaultdict(list))
    for task in sorted(scene_tasks):
        for row in pool[task]:sites[row['balance_group']][task].append(row)
        pool[task]=[]
    # Share the site budget across scene subtypes, retaining subtype variety.
    for groups in sites.values():
        for group in groups.values():rng.shuffle(group)
        remaining=max_scene_per_site
        while remaining and any(groups.values()):
            tasks=[task for task,group in groups.items() if group];rng.shuffle(tasks)
            for task in tasks:
                pool[task].append(groups[task].pop());remaining-=1
                if not remaining:break
    if any(not pool[k] for k in fractions):
        raise ValueError('Site cap leaves an empty requested task pool')
    size=int(min(len(pool[k])/p for k,p in fractions.items()))
    quotas={k:int(size*p) for k,p in fractions.items()}
    for k in sorted(fractions,key=lambda k:size*fractions[k]-quotas[k],reverse=True):
        if sum(quotas.values())>=size:break
        if quotas[k]<len(pool[k]):quotas[k]+=1
    selected=[]
    for task,quota in quotas.items():
        groups=defaultdict(list)
        for row in pool[task]:groups[row['balance_group']].append(row)
        for values in groups.values():rng.shuffle(values)
        while quota:
            keys=[k for k,v in groups.items() if v];rng.shuffle(keys)
            if not keys:raise ValueError('Insufficient unique samples')
            for key in keys:
                selected.append(groups[key].pop());quota-=1
                if not quota:break
    rng.shuffle(selected)
    if len({(r['dataset'],r['sample_id']) for r in selected})!=len(selected):
        raise ValueError('Duplicate samples in one round')
    return selected


def video(path, indices):
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);frames=[];i=0
    wanted=set(indices)
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if i in wanted:frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            i+=1
    finally:cap.release()
    if i!=120 or len(frames)!=len(indices):raise ValueError('Invalid target video/frame selection')
    return np.stack(frames)


class MixedMemoryDataset:
    def __init__(self, manifest, roots, temporal_indices=None):
        self.rows=[json.loads(line) for line in Path(manifest).read_text(encoding='utf-8').splitlines() if line]
        self.roots={k:Path(v) for k,v in roots.items()}
        self.indices=list(range(120)) if temporal_indices is None else list(temporal_indices)
        if not self.indices or self.indices!=sorted(set(self.indices)) or min(self.indices)<0 or max(self.indices)>=120:
            raise ValueError('Indices must be unique increasing values within 0..119')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];root=self.roots[row['dataset']]/row['sample_root']
        sample=json.loads((root/'sample.json').read_text(encoding='utf-8'))
        conditions=sample['conditions'];refs=[]
        pairs=conditions.get('reference_pairs') or [conditions['reference_pair']]
        for i,pair in enumerate(pairs):
            value={'reference_id':pair.get('reference_id',row['sample_id']+':scene_ref')}
            for key in ('rgb','whitebox'):
                frame=cv2.imread(str(root/pair[key]))
                if frame is None:raise ValueError('Missing reference image')
                value[key]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            refs.append(value)
        rgb=video(root/sample['supervision']['target_rgb'],self.indices)
        whitebox=video(root/conditions['target_whitebox'],self.indices)
        if rgb.shape!=whitebox.shape:raise ValueError('Video shapes differ')
        valid_path=sample['supervision'].get('valid_pixels')
        valid=cv2.imread(str(root/valid_path),0) if valid_path else np.full(rgb.shape[1:3],255,np.uint8)
        if valid is None or valid.shape!=rgb.shape[1:3]:raise ValueError('Invalid loss mask')
        edges=[]
        for ref in refs:ref['rgb'][valid==0]=0
        for edge in sample.get('object_references',[]):
            clean={k:edge[k] for k in ('object_id','category','reference_id','reference_bbox_xyxy','observed_at_ns','used_at_ns','used_until_ns')}
            clean['target_projected_bboxes']=[edge['target_projected_bboxes'][i] for i in self.indices]
            mask=cv2.imread(str(root/edge['reference_mask']),0)
            if mask is None or mask.shape!=valid.shape:raise ValueError('Invalid object mask')
            clean['reference_mask']=(mask>0)&(valid>0);edges.append(clean)
        # QA, future lifecycle data, labels, and target RGB captions never enter conditions.
        return {'sample_id':row['sample_id'],'task':row['task'],
                'conditions':{'prompt':(root/conditions['prompt']).read_text(encoding='utf-8'),
                              'reference_pairs':refs,'target_whitebox':whitebox,'object_references':edges},
                'supervision':{'target_rgb':rgb,'valid_pixels':valid.astype(np.float32)/255},
                'temporal_indices':self.indices,'source_fps':sample.get('fps',24)}
