"""Portable object-memory dataset reader. Target RGB is supervision only."""
import json
from pathlib import Path

import cv2
import numpy as np


def safe_path(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Dataset path escapes sample root')
    return path


def read_video(path):
    cap = cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    frames=[]
    try:
        while True:
            ok, frame=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    if not frames:raise ValueError(f'Empty video: {path}')
    return np.stack(frames)


def checked_boxes(value, count, width, height):
    try:
        boxes = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid reference/projection boxes') from error
    if (boxes.shape != (count, 4) or not np.isfinite(boxes).all()
            or (boxes[:, :2] < 0).any()
            or (boxes[:, 2:] > [width, height]).any()
            or (boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError('Invalid reference/projection boxes')
    return boxes


class GTAObjectMemoryDataset:
    def __init__(self,root,split='train'):
        self.root=Path(root).resolve()
        with (self.root/'splits'/f'{split}.jsonl').open(encoding='utf-8') as source:
            self.samples=[json.loads(line) for line in source if line.strip()]

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        item=self.samples[index]
        root=safe_path(self.root,item['sample_root'])
        sample=json.loads((root/'sample.json').read_text(encoding='utf-8'))
        conditions=sample['conditions'];supervision=sample['supervision']
        whitebox=read_video(safe_path(root,conditions['target_whitebox']))
        rgb=read_video(safe_path(root,supervision['target_rgb']))
        if (rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.shape!=whitebox.shape
                or len(rgb)!=sample['frame_count'] or not len(rgb)):
            raise ValueError('Paired video shape/count mismatch')
        height, width = rgb.shape[1:3]
        mask=cv2.imread(str(safe_path(root,supervision['valid_pixels'])),cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape!=rgb.shape[1:3] or not mask.any():raise ValueError('Invalid pixel mask')
        refs=[]
        by_reference = {}
        reference_paths = {}
        for pair_index, pair in enumerate(conditions['reference_pairs']):
            reference_id = pair.get('reference_id', f"{sample['sample_id']}:ref{pair_index}")
            if not isinstance(reference_id, str) or not reference_id or reference_id in by_reference:
                raise ValueError('Invalid or duplicate reference ID')
            frames={'reference_id': reference_id}
            for kind in ('rgb','whitebox'):
                image=cv2.imread(str(safe_path(root,pair[kind])))
                if image is None or image.shape != rgb.shape[1:]:raise ValueError('Missing or wrong-shaped reference frame')
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                if kind=='rgb':image[mask==0]=0
                frames[kind]=image
            refs.append(frames)
            by_reference[reference_id] = frames
            reference_paths[reference_id] = {kind: safe_path(root, pair[kind]) for kind in ('rgb', 'whitebox')}
        if not refs:
            raise ValueError('No reference pairs')
        frame_times = None
        if sample.get('frame_state_index'):
            rows = json.loads(safe_path(root, sample['frame_state_index']).read_text(encoding='utf-8'))
            frame_times = [row['server_time_estimate_ns'] for row in rows]
            if (len(frame_times) != len(rgb) + 1
                    or any(type(t) is not int for t in frame_times)
                    or any(a >= b for a, b in zip(frame_times, frame_times[1:]))):
                raise ValueError('Invalid frame-state timeline')
        objects = []
        auxiliary = []
        seen = set()
        fields = ('object_id', 'category', 'reference_id', 'reference_bbox_xyxy',
                  'target_projected_bboxes', 'observed_at_ns', 'used_at_ns', 'used_until_ns')
        for edge in sample['object_references']:
            reference_id = edge['reference_id']
            if reference_id not in by_reference:
                raise ValueError('Object references unknown reference ID')
            for kind in ('rgb', 'whitebox'):
                if edge.get('reference_' + kind) is not None:
                    if safe_path(root, edge['reference_' + kind]) != reference_paths[reference_id][kind]:
                        raise ValueError('Object/reference-pair asset mismatch')
            object_id = edge['object_id']
            if not isinstance(object_id, str) or not object_id or (object_id, reference_id) in seen:
                raise ValueError('Invalid or duplicate object reference')
            seen.add((object_id, reference_id))
            if edge['category'] is not None and not isinstance(edge['category'], str):
                raise ValueError('Invalid object category')
            times = [edge[k] for k in ('observed_at_ns', 'used_at_ns', 'used_until_ns')]
            if (any(type(t) is not int for t in times) or not times[0] < times[1] <= times[2]
                    or (len(rgb) > 1 and times[1] == times[2])):
                raise ValueError('Invalid object reference timeline')
            if frame_times is not None and times != [frame_times[0], frame_times[1], frame_times[-1]]:
                raise ValueError('Object reference/frame-state timeline mismatch')
            box = checked_boxes([edge['reference_bbox_xyxy']], 1, width, height)[0]
            target_boxes = checked_boxes(edge['target_projected_bboxes'], len(rgb), width, height)
            object_mask = cv2.imread(str(safe_path(root, edge['reference_mask'])), cv2.IMREAD_GRAYSCALE)
            if object_mask is None or object_mask.shape != (height, width):
                raise ValueError('Missing or wrong-shaped object mask')
            object_mask = (object_mask > 0) & (mask > 0)
            x0, y0 = np.floor(box[:2]).astype(int)
            x1, y1 = np.ceil(box[2:]).astype(int)
            if not object_mask[y0:y1, x0:x1].any():
                raise ValueError('No valid object-mask pixels in reference box')
            # Only this explicit schema is allowed into training conditions.
            clean = {key: edge[key] for key in fields}
            clean['reference_bbox_xyxy'] = box.tolist()
            clean['target_projected_bboxes'] = target_boxes.tolist()
            clean['reference_mask'] = object_mask
            clean['reference_crop'] = by_reference[reference_id]['rgb'][y0:y1, x0:x1].copy()
            objects.append(clean)
            if edge.get('embedding'):
                embedding = np.load(safe_path(root, edge['embedding']), allow_pickle=False)
                if embedding.shape != (384,) or not np.isfinite(embedding).all():
                    raise ValueError('Invalid auxiliary object embedding')
                auxiliary.append({'object_id': object_id, 'reference_id': reference_id,
                                  'embedding': embedding})
        if not objects:
            raise ValueError('No object references')
        return {'sample_id':sample['sample_id'],
                'conditions':{'prompt':safe_path(root,conditions['prompt']).read_text(encoding='utf-8'),
                              'reference_pairs':refs,'target_whitebox':whitebox,
                              'object_references':objects},
                'auxiliary_object_features': auxiliary,
                'supervision':{'target_rgb':rgb,'valid_pixels':mask.astype(np.float32)/255.0}}
