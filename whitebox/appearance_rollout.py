"""Accepted-video appearance memory shared by planning and H3 rendering."""
import json
from pathlib import Path

from agent.action_planner import json_call
from whitebox.appearance import AppearanceMemory
from whitebox.world import scene_from_db


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def capture_accepted(db, config, output):
    """Enrich only the latest committed video; never accept a pending rollout."""
    history = db.get_meta('whitebox_history', [])
    if not history or history[-1].get('provenance') != 'AGENT_ACCEPTED_VIDEO_INTERPRETATION':
        return {'status': 'NO_ACCEPTED_VIDEO', 'references': []}
    record = history[-1]
    video = record['artifacts']['video']
    if video != db.get_meta('whitebox_last_video'):
        raise ValueError('Appearance source is not the current accepted video')
    capture_id = str(record['revision_after']) + ':' + str(Path(video).resolve())
    captures = db.get_meta('appearance_captures', {})
    if capture_id in captures:
        return captures[capture_id]
    output = Path(output).resolve()
    manifest = {'capture_id':capture_id,'video':str(Path(video).resolve())}
    if output.exists():
        if not (output/'source.json').is_file() or json.loads((output/'source.json').read_text()) != manifest:
            raise ValueError('Capture directory belongs to another or unbound source; choose a new directory')
    else:
        output.mkdir(parents=True)
        save(output/'source.json',manifest)
    from perception.frame_sampler import sample_frames
    frames = sample_frames(video, output/'frames', interval=3600.)
    scene = scene_from_db(db)
    proposal = json.loads((output/'proposal.json').read_text(encoding='utf-8')) if (output/'proposal.json').exists() else json_call(config,
        'Select useful APPEARANCE references from an already accepted video. Return JSON '
        '{references:[{object_id:string,sample_index:int,bbox_normalized:[left,top,right,bottom],'
        'view:front|back|left|right|unknown,prompt:string,evidence:string,quality:number}],'
        'uncertainties:[string]}. Images are chronological samples numbered from zero. '
        'Use only supplied existing IDs. Match each identity against description and visible evidence; '
        'omit uncertain, hidden, severely blurred or occluded objects. Do not invent an unseen view. '
        'At most two distinct useful views per object, eight references total. '
        'Boxes enclose the visible object in this exact image with coordinates in [0,1]. '
        'Describe stable visible appearance, never motion, location, ownership or inferred hidden detail. '
        'quality is 0..1. Video acceptance does not make every generated detail authoritative. '
        'These references cannot establish new world facts or override accepted state. '
        'Prioritize actors, held props and distinctive scene landmarks. No duplicate same-object same-view crops.',
        {'entities': [{k: e[k] for k in ('id','kind','category','description','holder') if k in e}
                      for e in scene['entities']],
         'samples': [{'sample_index': i, 'timestamp': f['timestamp']} for i,f in enumerate(frames)]},
        [f['path'] for f in frames])
    save(output/'proposal.json', proposal)
    try:
        result = store_proposal(db, record, capture_id, frames, proposal)
    except (ValueError,KeyError,TypeError) as exc:
        index=0
        while (output/f'proposal_rejected_{index:03d}.json').exists():index+=1
        (output/'proposal.json').rename(output/f'proposal_rejected_{index:03d}.json')
        save(output/f'proposal_rejected_{index:03d}_error.json',{'error':str(exc)})
        raise
    save(output/'result.json', result)
    return result


def store_proposal(db, record, capture_id, frames, proposal):
    """Validate the whole batch and commit metadata atomically, without world edits."""
    history = db.get_meta('whitebox_history', [])
    if not history or history[-1] != record or record.get('provenance') != 'AGENT_ACCEPTED_VIDEO_INTERPRETATION':
        raise ValueError('Memory requires an unchanged accepted history record')
    rows = proposal.get('references')
    if not isinstance(rows, list) or len(rows) > 8:
        raise ValueError('Expected at most eight appearance references')
    ids = {e['id'] for e in scene_from_db(db)['entities']}
    seen, counts = set(), {}
    for row in rows:
        identity = row['object_id']; index = row['sample_index']; view = row['view']
        if identity not in ids or type(index) is not int or not 0 <= index < len(frames):
            raise ValueError('Unknown identity or image index')
        if view not in {'front','back','left','right','unknown'} or (identity,view) in seen:
            raise ValueError('Invalid or duplicate view')
        seen.add((identity,view)); counts[identity] = counts.get(identity,0)+1
        if counts[identity] > 2 or not isinstance(row['prompt'],str) or not row['prompt'].strip() or not row.get('evidence'):
            raise ValueError('Appearance needs visible evidence and at most two views per object')
        quality = AppearanceMemory._number(row['quality'], 'quality')
        box = row['bbox_normalized']
        if not 0 <= quality <= 1 or not isinstance(box,list) or len(box) != 4:
            raise ValueError('Invalid quality or box')
        left,top,right,bottom = [AppearanceMemory._number(x,'box') for x in box]
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError('Invalid normalized box')
    memory = AppearanceMemory(db)
    refs = []
    # The accepted final frame maps to record.end for both supported video
    # conventions. Video zero can be the boundary OR the first simulated tick.
    video_zero_world_time = record['end'] - frames[-1]['timestamp']
    try:
        with db.transaction():
            if not db.connection.in_transaction:
                db.connection.execute('BEGIN IMMEDIATE')
            if db.get_meta('whitebox_history', [])[-1:] != [record]:
                raise ValueError('Accepted history changed during appearance extraction')
            captures = db.get_meta('appearance_captures', {})
            if capture_id in captures:
                return captures[capture_id]
            for row in rows:
                frame = frames[row['sample_index']]
                refs.append(memory.add(row['object_id'], frame['path'], row['prompt'],
                    view=row['view'], state='identity', source='ACCEPTED_GENERATED_VIDEO', accepted=True,
                    bbox_normalized=row['bbox_normalized'], quality=row['quality'],
                    timestamp=video_zero_world_time+frame['timestamp'],
                    evidence={'capture_id': capture_id, 'revision': record['revision_after'],
                              'video': record['artifacts']['video'], 'frame_id': frame['frame_id'],
                              'video_timestamp':frame['timestamp'],
                              'evidence': row['evidence']}))
            result = {'status': 'CAPTURED', 'capture_id': capture_id,
                      'references': [r['id'] for r in refs], 'uncertainties': proposal.get('uncertainties', [])}
            captures[capture_id] = result
            db.set_meta('appearance_captures', captures)
    except BaseException:
        for ref in refs:
            if 'reference_path' in ref:Path(ref['reference_path']).unlink(missing_ok=True)
        raise
    return result


def retrieve(db, output, limit=6, requests=None, text_object_ids=None):
    """Bounded object crops; historical appearance never supplies current state."""
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    memory = AppearanceMemory(db)
    missing_views = []
    entities = sorted(scene_from_db(db)['entities'], key=lambda e: (e['kind'] not in {'actor','staff'},e['id']))
    if type(limit) is not int or limit < 0:
        raise ValueError('Memory image limit must be a nonnegative integer')
    if text_object_ids is not None and (not isinstance(text_object_ids, list) or
            any(not isinstance(identity, str) for identity in text_object_ids)):
        raise ValueError('Text anchor object IDs must be a list of strings')
    text_object_ids = list(text_object_ids or [])
    if len(text_object_ids) != len(set(text_object_ids)):
        raise ValueError('Text anchor object IDs must be unique')
    known_ids = {e['id'] for e in entities}
    if any(identity not in known_ids for identity in text_object_ids):
        raise ValueError('Text anchor object ID is not in the world')
    if requests is None:
        priority = [memory.select(e['id'],state='identity',limit=2) for e in entities if e['kind'] in {'actor','staff'}]
        scenery = [memory.select(e['id'],state='identity',limit=2) for e in entities if e['kind'] not in {'actor','staff'}]
        selected = [group[i] for groups in (priority,scenery) for i in range(2)
                    for group in groups if len(group)>i][:limit]
        missing = []
    else:
        if not isinstance(requests, list) or len(requests) > limit:
            raise ValueError('Explicit memory requests must fit the image budget')
        known = {e['id'] for e in entities}
        seen, groups, missing = set(), [], []
        for request in requests:
            identity = request['object_id']
            view = request.get('view', 'unknown')
            if identity not in known or identity in seen:
                raise ValueError('Memory requests need unique existing object IDs')
            if view not in {'front', 'back', 'left', 'right', 'unknown'}:
                raise ValueError('Invalid requested memory view')
            seen.add(identity)
            group = memory.select(identity, view=view, state='identity', limit=2)
            groups.append(group)
            if not group:
                missing.append(identity)
        # One reference for each requested object before extra views. A requested
        # landmark cannot be displaced by an unrelated actor's second crop.
        selected = [group[i] for i in range(2) for group in groups if len(group)>i][:limit]
        missing_views = [{'object_id': row['object_id'], 'view': row['view']} for row in requests
                         if row.get('view', 'unknown') != 'unknown' and not any(
                             ref['object_id'] == row['object_id'] and ref['view'] == row['view']
                             for ref in selected)]
    images, refs = [], []
    lines = ['<Picture 1> is the current accepted boundary. Additional pictures are historical '
             'identity references ONLY. Do not copy their pose, location, ownership, camera or background. '
             'Multiple views depict the same object, not additional actors. '
             'Never add an offscreen object merely because a reference exists. Current world state wins.']
    for i, ref in enumerate(selected, 2):
        image = memory._image(ref).convert('RGB'); image.thumbnail((512,512))
        path = output/f'picture_{i:02d}.png'; image.save(path)
        images.append(str(path))
        refs.append({**ref, 'role': 'identity_only', 'picture': i, 'crop_path': str(path)})
        lines.append(f"<Picture {i}>: object {ref['object_id']}, view {ref['view']}; {ref['prompt']}")
    requested_ids = [row['object_id'] for row in requests] if requests is not None else []
    identities = list(dict.fromkeys(text_object_ids + requested_ids +
                                    [ref['object_id'] for ref in selected]))
    anchors = [anchor for identity in identities if (anchor := memory.identity_anchor(identity)) is not None]
    if anchors:
        lines.append('Fixed identity text anchors, separate from the selected images. '
                     'These are historical descriptions, not newly supplied views; current state governs pose and ownership.')
        lines.extend(f"Object {r['object_id']}; anchor {r['id']}: {r['prompt']}" for r in anchors)
    result = {'prompt': '\n'.join(lines), 'images': images, 'refs': refs,
              'identity_anchors': anchors,
              'world_revision': db.get_meta('whitebox_revision'),
              'retrieval_requests': requests, 'text_anchor_object_ids': text_object_ids,
              'missing_requested_objects': missing,
              'missing_requested_views': missing_views}
    if missing or missing_views:
        result['prompt'] += '\nMissing evidence (identity fallback is not this view): ' + json.dumps(
            {'objects': missing, 'views': missing_views})
    save(output/'request.json',result)
    if images:
        from PIL import Image, ImageDraw
        columns=min(3,len(images));rows=(len(images)+columns-1)//columns
        sheet=Image.new('RGB',(columns*320,rows*360),'white');draw=ImageDraw.Draw(sheet)
        for i,(path,ref) in enumerate(zip(images,refs)):
            x=(i%columns)*320;y=(i//columns)*360
            with Image.open(path) as crop:
                crop.thumbnail((304,308));sheet.paste(crop,(x+8,y+40))
            draw.text((x+8,y+5),f"Picture {ref['picture']}: {ref['object_id']}",fill='black')
            draw.text((x+8,y+21),f"{ref['view']} / identity only",fill='black')
        sheet.save(output/'contact_sheet.png')
    return result


def retrieve_for_task(db, config, output, task, reference=None, limit=6):
    """Let the agent query a bounded identity/landmark catalog before loading crops."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = scene_from_db(db)
    by_id = {e['id']: e for e in scene['entities']}
    catalog = {identity: {
        'entity': {k: entity[k] for k in ('id', 'kind', 'category', 'description', 'position', 'holder') if k in entity},
        'views': set(), 'reference_count': 0} for identity, entity in by_id.items()}
    for ref in db.get_meta(AppearanceMemory.META_KEY, []):
        if ref['object_id'] not in by_id or not (ref['original'] or (ref['accepted'] and ref['state']=='identity')):
            continue
        entry = catalog[ref['object_id']]
        entry['views'].add(ref['view'])
        entry['reference_count'] += 1
    if not any(row['reference_count'] for row in catalog.values()):
        result = retrieve(db, output, limit=limit, requests=[])
        result['selection_missing_evidence'] = ['No eligible historical appearance references exist.']
        result['prompt'] += '\n' + result['selection_missing_evidence'][0]
        save(output/'request.json', result)
        return result
    catalog = [{**row, 'views': sorted(row['views'])} for row in catalog.values()]
    request = {'task': task, 'camera': scene['camera'], 'catalog': catalog,
               'world_revision': db.get_meta('whitebox_revision'), 'image_budget': limit}
    save(output/'selection_context.json', request)
    selection = json_call(config,
        'Choose historical appearance evidence for this task. Return JSON '
        '{requests:[{object_id:string,view:front|back|left|right|unknown}],'
        'text_object_ids:[string],reason:string,missing_evidence:[string]}. Select unique IDs from the catalog; '
        'text_object_ids are stable identity anchors and may include relevant objects even when no image slot is allocated. '
        'Only requests is limited to image_budget entries, ordered by priority. '
        'Text anchors have no image-slot cost. Consider the whole planned camera '
        'path, including objects that will reappear, not only current visibility. '
        'Select relevant fixed scene landmarks as well as the protagonist and held props. '
        'Do not spend images on unrelated offscreen characters. Request view=unknown '
        'when the intended view cannot be determined. Every selected object gets one '
        'image before second views are added; request fewer objects when multiple views '
        'are necessary. Missing views or landmarks are limitations, never invent evidence. '
        'These images go to your reasoning context, NOT automatically into H3.',
        request, [reference] if reference else [])
    save(output/'selection.json', selection)
    requests = selection.get('requests')
    text_object_ids = selection.get('text_object_ids', [row.get('object_id') for row in requests or []])
    if not isinstance(requests, list):
        raise ValueError('Agent must explicitly select memory requests')
    if not isinstance(text_object_ids, list):
        raise ValueError('Memory selector must report text_object_ids')
    if any(row.get('object_id') not in {e['entity']['id'] for e in catalog} for row in requests):
        raise ValueError('Agent selected an identity outside the memory catalog')
    missing = selection.get('missing_evidence')
    if not isinstance(missing,list) or any(not isinstance(item,str) for item in missing):
        raise ValueError('Memory selector must report missing_evidence strings')
    result = retrieve(db, output, limit=limit, requests=requests, text_object_ids=text_object_ids)
    result['selection_missing_evidence'] = missing
    if missing:
        result['prompt'] += '\nTask evidence limitations: ' + json.dumps(missing)
    save(output/'request.json', result)
    return result
