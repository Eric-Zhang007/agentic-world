"""Explicit revisions of inferred geometry, separate from events in world time."""
from copy import deepcopy

from whitebox.camera_reconcile import validate_camera, update_camera
from whitebox.motion import validate_geometry
from whitebox.world import put_entity, scene_from_db, validate_scene


def apply_proposal(scene, proposal):
    if set(proposal)-{'edits','additions','camera','evidence','uncertainty'}:
        raise ValueError('Unknown scene refinement fields')
    if not proposal.get('evidence') or not proposal.get('uncertainty'):
        raise ValueError('Inferred geometry refinement needs evidence and uncertainty')
    candidate = deepcopy(scene)
    entities = {e['id']:e for e in candidate['entities']}
    for addition in proposal.get('additions',[]):
        if set(addition)!={'entity','evidence'} or not addition.get('evidence'):
            raise ValueError('New inferred geometry requires explicit evidence')
        entity=deepcopy(addition['entity'])
        if (entity['id'] in entities or entity.get('kind') not in {'box','stairs'} or
                entity.get('provenance')!='INFERRED'):
            raise ValueError('Only unique inferred static primitives can be proposed')
        entities[entity['id']]=entity
        candidate['entities'].append(entity)
    seen = set()
    for edit in proposal.get('edits',[]):
        if set(edit)-{'id','position','size','rotation_y','evidence'} or not edit.get('evidence'):
            raise ValueError('Only evidence-backed static transforms are supported')
        identity = edit['id']
        if identity not in entities or identity in seen:
            raise ValueError('Refinement needs unique existing IDs')
        entity = entities[identity]
        if entity['kind'] not in {'box','stairs'} or entity['provenance'] != 'INFERRED':
            raise ValueError('This operation only revises inferred static geometry')
        seen.add(identity)
        for key in ('position','size','rotation_y'):
            if key in edit:
                entity[key] = deepcopy(edit[key])
    if proposal.get('camera') is not None:
        if set(proposal['camera'])-{'position','target','fov'}:
            raise ValueError('Unknown camera fields')
        candidate['camera'] = update_camera(candidate['camera'], proposal['camera'])
    validate_camera(candidate['camera'])
    validate_scene(candidate)
    errors = validate_geometry(candidate)
    if errors:
        raise ValueError('Refined scene intersects current actors: '+str(errors))
    return candidate


def historical_geometry_checks(candidate, history):
    """Check every archived actor sample against revised obstacles without editing it."""
    from pathlib import Path
    import json
    errors = []
    for index, record in enumerate(history):
        path = record.get('artifacts',{}).get('corrected_timeline')
        if not path or not Path(path).is_file():
            raise ValueError('Historical geometry cannot be checked without archived timeline')
        timeline = json.loads(Path(path).read_text())
        for frame in timeline['frames']:
            state = deepcopy(candidate)
            dynamic = {e['id']:e for e in frame['entities'] if e['kind'] in {'actor','staff'}}
            state['entities'] = [deepcopy(dynamic.get(e['id'],e)) for e in state['entities']]
            for error in validate_geometry(state):
                errors.append({'history_index':index,'time':frame['time'],**error})
    return errors


def commit_refinement(db, before, proposal, expected_revision, validation, artifacts):
    """Append an epistemic revision; do not claim scenery moved or rewrite history."""
    with db.transaction():
        if not db.connection.in_transaction:
            db.connection.execute('BEGIN IMMEDIATE')
        if db.get_meta('whitebox_revision') != expected_revision or scene_from_db(db) != before:
            raise ValueError('World changed during refinement')
        candidate = apply_proposal(before,proposal)
        if (validation.get('current',{}).get('usable_for_camera_alignment') is not True or
                validation.get('joint',{}).get('consistent') is not True or
                not validation.get('joint',{}).get('evidence') or
                validation.get('history',{}).get('consistent') is not True or
                not validation.get('history',{}).get('evidence')):
            raise ValueError('Static, actor and historical visual evidence must all pass')
        errors = historical_geometry_checks(candidate,db.get_meta('whitebox_history',[]))
        if errors:
            raise ValueError('Revised obstacles contradict archived actor paths: '+str(errors[:8]))
        old = {e['id']:e for e in before['entities']}
        for entity in candidate['entities']:
            if entity != old.get(entity['id']):
                put_entity(db,entity,'Explicit inferred-geometry revision; not an in-world movement')
        db.set_meta('whitebox_entity_order',[e['id'] for e in candidate['entities']])
        db.set_meta('whitebox_header',{k:deepcopy(v) for k,v in candidate.items() if k!='entities'})
        revisions = db.get_meta('whitebox_representation_revisions',[])
        revisions.append({'revision_before':expected_revision,'revision_after':expected_revision+1,
            'world_time':db.get_meta('tick'),'kind':'INFERRED_GEOMETRY_REVISION',
            'semantics':'New interpretation of existing evidence; archived states remain unchanged.',
            'before':before,'after':candidate,'proposal':proposal,'validation':validation,'artifacts':artifacts})
        db.set_meta('whitebox_representation_revisions',revisions)
        db.set_meta('whitebox_revision',expected_revision+1)
    return candidate
