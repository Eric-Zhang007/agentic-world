"""Convert real VGGT + SAM results to a reviewable whitebox scene proposal."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.geometry_assets import compile_assets, transform_points
from whitebox.geometry_regions import deduplicate_masks, fit_facade_boxes
from whitebox.geometry_ground import complete_ground, ground_mesh


GEOMETRY_CATEGORIES = {
    **dict.fromkeys(('road', 'ground', 'sidewalk'), 'ground'),
    **dict.fromkeys(('building', 'wall'), 'facade'),
    **dict.fromkeys(('person', 'pedestrian'), 'actor'),
    **dict.fromkeys(('car', 'truck', 'bus', 'motorcycle', 'vehicle', 'bicycle',
                     'tree', 'pole', 'dog', 'traffic light', 'bench'), 'entity'),
    **dict.fromkeys(('sky', 'unknown'), 'excluded'),
}


def geometry_review_mapping(scene, perception):
    observations=[row['observation_id'] for row in perception['observations']]
    identities=[entity['id'] for entity in scene['entities']]
    if len(set(observations))!=len(observations) or len(set(identities))!=len(identities):
        raise ValueError('Duplicate observation or entity IDs')
    mapping={}
    for entity in scene['entities']:
        source=entity.get('geometry_evidence',{}).get('source_observation_id')
        if source is None:
            continue
        if source not in observations:
            raise ValueError('Unknown source observation ID: '+str(source))
        mapping.setdefault(source,[]).append(entity['id'])
    return mapping


def map_geometry_category(observation):
    """Exact taxonomy membership, never substring/token-union classification.

    Detector text_labels are decoded token phrases, not class IDs. A score is
    detection confidence, not evidence that a composite phrase is one class.
    Multiword taxonomy entries are valid; all other phrases remain unresolved.
    """
    raw = observation.get('category')
    normalized = ' '.join(raw.casefold().strip().rstrip('.').split()) if isinstance(raw, str) else None
    role = GEOMETRY_CATEGORIES.get(normalized, 'unresolved')
    accepted = role not in ('unresolved', 'excluded')
    return {'raw_label': raw, 'normalized_label': normalized,
            'canonical_category': normalized if accepted else None,
            'geometry_role': role, 'accepted': accepted,
            'mapping_method': 'exact_taxonomy_v1',
            'reason': ('exact_supported_category' if accepted else
                       'non_geometric_category' if role == 'excluded' else
                       'no_unique_supported_category')}


def partition_geometry_masks(masks):
    """Separate compile inputs from non-solid semantic candidates and blockers."""
    accepted, blockers, candidates, mappings = [], [], [], []
    for item in masks:
        mapping = map_geometry_category(item)
        row = {'source_mask_id': item['id'], **mapping}
        mappings.append(row)
        if mapping['accepted']:
            accepted.append({**item, 'category': mapping['canonical_category'],
                             'category_mapping': row})
        else:
            # Unknown support is not empty space: preserve its silhouette as a
            # blocker for ground completion, but never fit it as solid geometry.
            blockers.append({**item, 'category': 'unknown'})
            candidates.append({**row, 'category': item.get('category'),
                               'status': 'UNRESOLVED_SEMANTICS' if mapping['geometry_role'] == 'unresolved'
                                         else 'EXCLUDED_NON_GEOMETRIC',
                               'solid': False, 'accepted': False,
                               'confidence': item.get('confidence'),
                               'source_observation_id': item.get('source_observation_id'),
                               'source_mask_path': item.get('source_mask_path'),
                               'confident_pixels': int(np.count_nonzero(item['mask'])),
                               'raw_pixels': int(np.count_nonzero(item.get('raw_mask', item['mask']))),
                               'mask_artifact': 'semantic_candidate_masks.npz',
                               'mask_key': item['id'], 'raw_mask_key': item['id'] + '__raw',
                               'note': 'No solid box or ground inferred from unresolved/excluded semantics'})
    return accepted, blockers, candidates, mappings


def ground_triangles(points, masks, matrix, stride=4):
    selected = [m['mask'] for m in masks if m['category'] in ('road', 'ground', 'sidewalk')]
    mask = np.logical_or.reduce(selected)
    # No fill outside observed ground; do not bridge silhouettes or depth jumps.
    for item in masks:
        if item['category'] not in ('road', 'ground', 'sidewalk'):
            mask &= ~item['mask']
    world = transform_points(points, matrix)
    triangles = []
    height, width = mask.shape
    for y in range(0, height-stride, stride):
        for x in range(0, width-stride, stride):
            if not mask[y:y+stride+1, x:x+stride+1].all():
                continue
            pixels = [(y, x), (y, x+stride), (y+stride, x), (y+stride, x+stride)]
            p = np.array([world[a, b] for a, b in pixels])
            original = np.array([points[a, b] for a, b in pixels])
            radii = np.linalg.norm(original, axis=1)
            if radii.max() > 1.2*radii.min():
                continue
            for inds in ([0, 1, 2], [1, 3, 2]):
                triangle = p[inds]
                if np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])[1] < 0:
                    triangle = triangle[[0, 2, 1]]
                triangles.append(triangle)
    if not triangles:
        raise ValueError('No connected observed ground patches')
    return np.asarray(triangles, dtype='<f4')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reconstruction', type=Path, required=True)
    parser.add_argument('--perception', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--scale-source', default='uncalibrated model units')
    parser.add_argument('--actor-height-prior', type=float,
                        help='Optional assumed height of largest visible person, in meters; not a measurement')
    parser.add_argument('--confidence-quantile', type=float, default=.2)
    parser.add_argument('--plane-threshold', type=float, default=.02)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    if not 0 <= args.confidence_quantile < 1:
        parser.error('confidence-quantile must be in [0,1)')
    with np.load(args.reconstruction, allow_pickle=False) as data:
        points = data['points'][args.frame]
        image = data['images'][args.frame].transpose(1, 2, 0)
        conf = data['depth_conf'][args.frame].squeeze()
        extrinsic, intrinsic = data['extrinsic'][args.frame], data['intrinsic'][args.frame]
        metadata = json.loads(str(data['metadata']))
    perception = json.loads(args.perception.read_text(encoding='utf-8'))
    height, width = points.shape[:2]
    record = metadata['preprocessing']['images'][args.frame]
    matrix = np.asarray(record['original_to_processed_pixel_centers'])
    sam_image = args.perception.parent/'input.png'
    with Image.open(sam_image) as source:
        source = source.convert('RGB').resize(tuple(record['resized_wh']), Image.Resampling.BICUBIC)
        top = record['crop_top']
        source = source.crop((0, top, source.width, min(source.height, top+height)))
        canvas = Image.new('RGB', (width, height), 'white')
        left, pad_top = record['padding_ltrb'][:2]
        canvas.paste(source, (left, pad_top))
    if np.max(abs(np.asarray(canvas)/255.-image)) > .01:
        raise ValueError('SAM and reconstruction RGB differ after preprocessing; refusing misaligned masks')
    finite = np.isfinite(points).all(axis=2) & np.isfinite(conf)
    finite &= conf >= np.quantile(conf[finite], args.confidence_quantile)
    masks = []
    for index, raw in enumerate(perception['observations']):
        path = Path(raw['mask_path'])
        if not path.is_file():
            path = args.perception.parent / path.name
        mask = np.asarray(Image.open(path).convert('L')) > 0
        source_size = tuple(perception['image_size'])
        if source_size != tuple(record['original_wh']):
            raise ValueError('SAM input size differs from original VGGT image; use the same source RGB')
        mask = cv2.warpPerspective(mask.astype(np.uint8), matrix, (width, height),
                                   flags=cv2.INTER_NEAREST)
        raw_mask = mask > 0
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
        masks.append({'id': f'geo_{index:03d}', 'category': raw.get('category'),
                      'source_observation_id': raw.get('observation_id'),
                      'source_mask_path': str(path.resolve()),
                      'confidence': raw['confidence'], 'raw_mask': raw_mask, 'mask': mask & finite})
    masks, semantic_blockers, source_candidates, category_mappings = partition_geometry_masks(masks)
    # Persist semantics before geometric fitting so failed ground fits still
    # leave an inspectable candidate list, without emitting a partial scene.
    args.output.mkdir(parents=True)
    (args.output/'category_mapping.json').write_text(json.dumps({
        'schema_version': 1, 'taxonomy': GEOMETRY_CATEGORIES,
        'observations': category_mappings,
        'policy': 'Exact supported categories only; no composite-label guessing or default solid box',
    }, indent=2), encoding='utf-8')
    (args.output/'source_candidates.json').write_text(json.dumps(source_candidates, indent=2), encoding='utf-8')
    if semantic_blockers:
        arrays = {}
        for item in semantic_blockers:
            arrays[item['id']] = item['mask']
            arrays[item['id']+'__raw'] = item['raw_mask']
        np.savez_compressed(args.output/'semantic_candidate_masks.npz', **arrays)
    masks, deduplication = deduplicate_masks(masks)
    scene, report = compile_assets(points, image, masks, extrinsic, intrinsic,
                                  scale=args.scale, scale_source=args.scale_source,
                                  plane_threshold=args.plane_threshold)
    if args.actor_height_prior is not None:
        if args.actor_height_prior <= 0 or not np.isfinite(args.actor_height_prior):
            raise ValueError('Actor height prior must be positive and finite')
        player = next((e for e in scene['entities'] if e['id'] == report['player_candidate']), None)
        if player is None:
            raise ValueError('No person found for optional height prior')
        scale = args.scale * args.actor_height_prior/player['size'][1]
        scene, report = compile_assets(points, image, masks, extrinsic, intrinsic,
                                      scale=scale,
                                      scale_source=f'assumed visible player height {args.actor_height_prior}m',
                                      plane_threshold=args.plane_threshold)
    world_points = transform_points(points, np.asarray(report['model_to_scene']))
    by_id = {m['id']: m for m in masks}
    compiled, facades = [], []
    for entity in scene['entities']:
        entity['geometry_evidence']['category_mapping'] = by_id[entity['id']]['category_mapping']
        entity['geometry_evidence']['source_observation_id'] = by_id[entity['id']].get('source_observation_id')
        if by_id[entity['id']]['category_mapping']['geometry_role'] != 'facade':
            compiled.append(entity)
            continue
        item = by_id[entity['id']]
        cloud = world_points[item['mask']]
        fit = fit_facade_boxes(cloud)
        source_candidates.append({'source_mask_id': entity['id'], 'category': entity['category'],
                                  'status': 'INCOMPLETE_VISIBLE_GEOMETRY',
                                  'original_box_proposal': entity,
                                  'unassigned_point_indices': fit['unassigned_indices'],
                                  'mask_pixel_order': 'row-major nonzero source mask',
                                  'point_coverage': fit['coverage'],
                                  'note': 'Unfitted geometry is unknown, not a deleted building'})
        facades.append({'source_mask_id': entity['id'], 'parts': len(fit['boxes']),
                        'coverage': fit['coverage'], 'unassigned_points': len(fit['unassigned_indices']),
                        'parameters': fit['parameters']})
        for index, part in enumerate(fit['boxes']):
            indices = np.asarray(part.pop('point_indices'))
            eid = entity['id'] if index == 0 else f"{entity['id']}__part{index+1:02d}"
            compiled.append({**entity, **part, 'id': eid,
                             'color': np.median(image[item['mask']][indices], axis=0).tolist(),
                             'reason': 'Locally supported visible facade; hidden building thickness unknown',
                             'geometry_evidence': {**part['geometry_evidence'],
                                                   'category_mapping': item['category_mapping'],
                                                   'source_mask_id': entity['id'],
                                                   'source_observation_id': item.get('source_observation_id'),
                                                   'scale_source': report['scale_source']}})
    scene['entities'] = compiled
    report.update(mask_deduplication=deduplication, facade_regions=facades,
                  category_mapping={'schema_version': 1, 'artifact': 'category_mapping.json',
                                    'accepted_observations': sum(m['accepted'] for m in category_mappings),
                                    'unresolved_observations': sum(m['geometry_role'] == 'unresolved' for m in category_mappings),
                                    'excluded_observations': sum(m['geometry_role'] == 'excluded' for m in category_mappings)})
    from whitebox.godot.render import validate
    validate(scene, None)
    (args.output/'source_candidates.json').write_text(json.dumps(source_candidates, indent=2), encoding='utf-8')
    completion = complete_ground(points, masks + semantic_blockers,
                                 extrinsic, intrinsic, np.asarray(report['model_to_scene']), valid=finite)
    triangles, inferred_triangles = ground_mesh(completion)
    if not len(triangles):
        raise ValueError('No supported ground surface')
    colors = np.full_like(triangles, .55)
    colors[inferred_triangles] = [.23, .65, .7]
    np.concatenate([triangles, colors], axis=2).astype('<f4').tofile(args.output/'ground.bin')
    triangles.tofile(args.output/'ground_collision.bin')
    ground_ids = [e['id'] for e in scene['entities'] if e['category'] in ('road', 'ground', 'sidewalk')]
    scene['entities'] = [e for e in scene['entities'] if e['id'] not in ground_ids]
    scene['static_geometry'] = {'path': 'ground.bin', 'collision_path': 'ground_collision.bin'}
    report['ground_surface'] = {'triangles': len(triangles), 'replaces_entities': ground_ids,
                                'source_mask_ids': [m['id'] for m in masks
                                                    if m['category_mapping']['geometry_role'] == 'ground'],
                                'source_categories': sorted({m['category'] for m in masks
                                                            if m['category_mapping']['geometry_role'] == 'ground'}),
                                'visible_pixels': int(completion.visible_mask.sum()),
                                'inferred_triangles': int(inferred_triangles.sum()),
                                'inferred_pixels': int(completion.inferred_mask.sum()),
                                'completion_evidence': completion.evidence,
                                'coverage': 'Observed ground plus explicitly inferred dynamic-occlusion patches'}
    Image.fromarray(completion.inferred_mask.astype('uint8')*255).save(args.output/'inferred_ground_mask.png')
    np.savez_compressed(args.output/'ground_evidence.npz', visible_mask=completion.visible_mask,
                        inferred_mask=completion.inferred_mask, inferred_triangles=inferred_triangles)
    report.update(reconstruction=str(args.reconstruction.resolve()),
                  perception=str(args.perception.resolve()), frame=args.frame,
                  confidence_quantile=args.confidence_quantile,
                  entity_count=len(scene['entities']), mask_count=len(masks))
    for name, value in [('scene', scene), ('report', report)]:
        (args.output / (name+'.json')).write_text(json.dumps(value, indent=2), encoding='utf-8')
    (args.output/'review_mapping.json').write_text(json.dumps(geometry_review_mapping(scene,perception),indent=2),encoding='utf-8')
    Image.fromarray((image*255).clip(0, 255).astype('uint8')).save(args.output/'input.png')
    overlay = image.copy()
    for index, item in enumerate(masks):
        color = np.asarray([(index*71+50)%255, (index*39+120)%255, (index*99+80)%255])/255
        overlay[item['mask']] = .55*overlay[item['mask']] + .45*color
    out = Image.fromarray((overlay*255).clip(0, 255).astype('uint8'))
    draw = ImageDraw.Draw(out)
    for item in masks:
        yy, xx = np.where(item['mask'])
        if len(xx):
            draw.text((int(np.median(xx)), int(np.median(yy))), item['id']+' '+item['category'], fill='white')
    out.save(args.output/'masks.png')
    np.savez_compressed(args.output/'masks.npz', **{m['id']: m['mask'] for m in masks})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
