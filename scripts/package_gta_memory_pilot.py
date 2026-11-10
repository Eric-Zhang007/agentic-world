"""Apply an explicit visual review and package a small portable memory sample."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import cv2

from scripts.prepare_gta_object_memory import read, write
from memory.gta_validation import validate_reference_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', required=True, type=Path)
    parser.add_argument('--review', required=True, type=Path)
    args = parser.parse_args()
    root = args.sample
    archive = root.with_suffix('.zip')
    if archive.exists():
        raise ValueError('Refusing to modify an already archived package')
    package = read(root/'package.json')
    review = read(args.review)
    observations = read(root/'observations.json')
    assert all(package[k] == review[k] for k in ['episode_id', 'agent_id', 'segment'])
    assert observations[0]['proxy_frame_index'] == review['reference_frame']
    target_indices = [o['proxy_frame_index'] for o in observations[1:]]
    assert target_indices == list(range(review['target_frames'][0],review['target_frames'][1]+1))
    oid = review['dataset_entity_id']
    entities = validate_reference_identity(observations, oid)
    # Lifecycle identity is checked independently of model identity.
    import pyarrow.parquet as pq
    lifecycle = []
    path = Path(package['source_root'])/'server/entities.parquet'
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
        for row in batch.to_pylist():
            payload = json.loads(row['payload_json'])
            if str(payload.get('dataset_entity_id')) == oid:
                lifecycle.append(payload)
    assert len(lifecycle) == 1
    life = lifecycle[0]
    ticks = [o['frame_metadata']['world_capture_tick'] for o in observations]
    assert life['spawn_sample_seq'] <= min(ticks)
    assert life.get('despawn_sample_seq') is None or max(ticks) < life['despawn_sample_seq']
    image = cv2.imread(str(root/'references/rgb.png'))
    x0,y0,x1,y1 = review['reference_rgb_bbox_xyxy']
    assert 0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]
    cv2.imwrite(str(root/'references/object_crop.png'),image[y0:y1,x0:x1])
    write(root/'visual_review.json', review)
    write(root/'object_lifecycle.json', life)
    key = f'{package["episode_id"]}:entity:{oid}'
    edge = {'object_id': key, 'reference_id': 'reference_000',
        'reference_rgb': 'references/rgb.png', 'reference_whitebox': 'references/whitebox.png',
        'reference_crop': 'references/object_crop.png',
        'reference_bbox_xyxy': review['reference_rgb_bbox_xyxy'],
        'observed_at_ns': package['reference_time_ns'],
        'used_at_ns': package['target_start_time_ns'],
        'used_by': 'target_000', 'appearance_revision': 'pilot_observed_appearance_000',
        'revision_scope': 'visible exterior only; pose follows target world records',
        'state_compatibility': 'visible appearance reviewed; unobserved state unknown',
        'target_frame_range_inclusive': review['target_frames']}
    assert edge['observed_at_ns'] < edge['used_at_ns']
    category = review.get('category', entities[0]['entity_type'])
    prompt = (f'Object {key} is the {category} identified by the corresponding entity in the target whitebox. '
        'Use Reference 1 RGB and its paired whitebox frame to associate that object with its visible appearance. '
        f'The RGB region is {review["reference_rgb_bbox_xyxy"]} in the {image.shape[1]}x{image.shape[0]} reference. '
        f'{review["visible_appearance"]} '
        'Keep this exterior identity while following target whitebox camera and object motion. '
        'The historical pose and screen position are not target pose constraints. '
        f'{review.get("generation_note", "")} '
        'No object-specific historical appearance constraint is supplied for background objects. '
        'The target RGB video is supervision only and must never be supplied as a memory reference.')
    (root/'prompt.txt').write_text(prompt,encoding='utf-8')
    package.update({'status':'VISUALLY_REVIEWED_OBJECT_MEMORY_PILOT',
        'training_ready':False,
        'not_training_ready_reason':'H3 loader and clip-length formatting are out of scope; no full alignment/mask validation',
        'reference_edges':[edge], 'review':'visual_review.json'})
    write(root/'package.json',package)
    sample = {'sample_id': 'target_000', 'object_references':[edge],
        'conditions': {'prompt':'prompt.txt','reference_pairs':[{'rgb':'references/rgb.png','whitebox':'references/whitebox.png'}],
                       'target_whitebox':'target_whitebox.mp4'},
        'supervision': {'target_rgb':'target_rgb.mp4'},
        'frame_state_index':'observations.json','camera_records':'camera_samples.json',
        'world_records':'world_states.json','review':'visual_review.json',
        'split':'pilot_unassigned_not_for_evaluation', 'training_ready':False}
    sample['invalid_rgb_regions_xyxy'] = review.get('invalid_rgb_regions_xyxy', [])
    (root/'samples.jsonl').write_text(json.dumps(sample,ensure_ascii=False)+'\n',encoding='utf-8')
    archive = root.with_suffix('.zip')
    if archive.exists():
        raise ValueError('Refusing to overwrite existing archive')
    shutil.make_archive(str(root), 'zip', root)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert 'samples.jsonl' in zipped.namelist()
    print(archive)


if __name__ == '__main__':
    main()
