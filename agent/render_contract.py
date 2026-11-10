"""One compact contract for geometry, prompt compilation and later review."""
from pathlib import Path
import re
from agent.action_planner import json_call

SKILL=Path(__file__).resolve().parents[1]/'skills/render-rollout/SKILL.md'


def boundary_actor(entity,entities):
    """Expose consequential state, not incidental procedural rig parameters."""
    result={k:entity[k] for k in ('id','category','description','position','rotation_y',
                                'posture_weights','support_contacts','support_state','gait','motion') if k in entity}
    result['held_objects']=[{k:e[k] for k in ('id','category','description','holder') if k in e}
                            for e in entities if e.get('holder')==entity['id']]
    return result


def renderer_references(memory, reference_ids=()):
    """Agent evidence is not automatically a renderer condition."""
    available = {r['id']: r for r in memory['refs']} if memory else {}
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError('Duplicate renderer reference IDs')
    refs = []
    for identity in reference_ids:
        if identity not in available:
            raise ValueError('Renderer reference was not retrieved: '+identity)
        ref = available[identity]
        if not (ref.get('accepted') or ref.get('original')):
            raise ValueError('Renderer reference lacks accepted/original provenance')
        refs.append(ref)
    if len(refs) > 8:
        raise ValueError('At most eight supplementary renderer references')
    lines = ['<Picture 1> is the current accepted video boundary.']
    for i, ref in enumerate(refs, 2):
        lines.append(f"<Picture {i}>: historical appearance only, object {ref['object_id']}; "
                     f"{ref['prompt']}. Current state governs pose, ownership and visibility.")
    return {'reference_ids': list(reference_ids), 'images': [r['crop_path'] for r in refs],
            'prompt': '\n'.join(lines),
            'selection': 'explicit supplementary references' if refs else 'accepted full-frame boundary only'}


def identity_text(memory, object_ids):
    """Keep retrieved visual detail after the agent selects this shot's identities."""
    if not isinstance(object_ids, list) or any(not isinstance(x, str) for x in object_ids):
        raise ValueError('Expected appearance_object_ids list')
    available = ({r['object_id'] for r in memory['refs']} |
                 {r['object_id'] for r in memory.get('identity_anchors', [])}) if memory else set()
    if len(object_ids) != len(set(object_ids)) or not set(object_ids).issubset(available):
        raise ValueError('Appearance objects must be unique retrieved identities')
    descriptions = []
    anchors = {r['object_id']: r for r in memory.get('identity_anchors', [])} if memory else {}
    for identity in object_ids:
        refs = ([anchors[identity]] if identity in anchors else
                [r for r in memory['refs'] if r['object_id'] == identity])
        seen = set()
        for ref in refs:
            if not (ref.get('original') or ref.get('accepted')):
                raise ValueError('Identity text lacks original/accepted provenance')
            if ref['prompt'] not in seen:
                descriptions.append(f"Object {identity}: {ref['prompt']}")
                seen.add(ref['prompt'])
    if not descriptions:
        return ''
    return ('Stable appearance descriptions from accepted video memory. Preserve these visible '
            'identity details; historical wording does not override current pose, ownership or location.\n'
            + '\n'.join(descriptions))


def compile_prompt(config,scene,action,timeline,reference,output,prefix_frames=1,memory=None,
                   renderer_reference_ids=(),reference_video=None):
    import json
    actor_id=action['actor_id']
    before=next(e for e in scene['entities'] if e['id']==actor_id)
    after=next(e for e in timeline['frames'][-1]['entities'] if e['id']==actor_id)
    contract={'action':action,'start':boundary_actor(before,scene['entities']),
              'end':boundary_actor(after,timeline['frames'][-1]['entities']),'camera_start':scene['camera'],
              'camera_end':timeline['frames'][-1]['camera'],
              'new_world_seconds':len(timeline['frames'])/timeline['fps'],
              'historical_prefix_frames':prefix_frames if prefix_frames>1 else 0,
              'immutable_entity_ids':[e['id'] for e in scene['entities']],
              'static_layout': [e for e in scene['entities'] if e['kind'] not in {'actor','staff'}]}
    skill=SKILL.read_text(encoding='utf-8')
    selected = renderer_references(memory, renderer_reference_ids)
    contract['renderer_references'] = selected
    contract['available_appearance_object_ids'] = sorted(
        {r['object_id'] for r in memory.get('refs', [])} |
        {r['object_id'] for r in memory.get('identity_anchors', [])}) if memory else []
    if reference_video is not None:
        contract['native_video_reference'] = {
            'tag':'<Video 1>', 'path':str(reference_video),
            'content':'RGB whitebox animation of this planned scene and camera timeline',
            'role':'Motion and layout reference; Picture 1 supplies actual visual identity and materials',
            'appearance_is_proxy':True}
    if memory:
        contract['identity_anchors'] = memory.get('identity_anchors', [])
        contract['appearance_evidence_limits'] = {key: memory.get(key, []) for key in
            ('missing_requested_objects','missing_requested_views','selection_missing_evidence')}
        contract['text_anchor_object_ids'] = memory.get('text_anchor_object_ids', [])
        contract['appearance_memory'] = [
            {'agent_image_index': i, 'object_id': r['object_id'], 'view': r['view'],
             'description': r['prompt'], 'reference_id': r['id']}
            for i,r in enumerate(memory['refs'], 2)]
        contract['appearance_reference_ids'] = [r['id'] for r in memory['refs']]
        skill += ('\nImages supplied to YOU: first the accepted boundary, then the retrieved memory crops '
                  'in appearance_memory order. These are reasoning evidence, NOT automatically images '
                  'supplied to H3. Only renderer_references lists actual H3 Picture tags. '
                  'Use memory to describe stable appearance in words; never emit a Picture tag for an '
                  'unselected memory image. Preserve current pose, location, ownership and visibility. '
                  'Do not mention an offscreen object simply because a memory image shows it. '
                  'Return appearance_object_ids selecting only retrieved identities visible or intended '
                  'to become visible in this shot. Their accepted appearance descriptions will be '
                  'appended intact; do not discard distinctive silhouette, clothing or prop details.')
        skill += (' Fixed identity_anchors are independent of selected image slots. '
                  'Use their descriptions for persistent identity; selected crops provide view evidence. '
                  'Do not merge contradictory newer appearance into the established identity. '
                  'Report conflicts as uncertainties. Anchor text is not an image of an unseen side; '
                  'accepted current state still governs actual changes, pose and ownership.')
    result=json_call(config,skill+'\nReturn JSON {prompt:string,required_outcomes:[string],'
                     'allowed_elaborations:[string],uncertainties:[string],appearance_object_ids:[string]}. '
                     'Write an English video prompt. appearance_object_ids must be a subset of '
                     'available_appearance_object_ids; return [] when that list is empty.',
                     contract,[reference]+(memory['images'] if memory else []))
    if not isinstance(result.get('prompt'),str) or not result['prompt'].strip():raise ValueError('Missing render prompt')
    selected_object_ids=result.get('appearance_object_ids',None if memory and memory['refs'] else [])
    descriptions=identity_text(memory,selected_object_ids)
    contract['rendered_appearance_object_ids']=selected_object_ids
    contract['rendered_identity_descriptions']=descriptions
    contract['rendered_identity_anchor_ids']=[r['id'] for r in memory.get('identity_anchors', [])
                                            if r['object_id'] in selected_object_ids] if memory else []
    tags = re.findall(r'<Picture\s+(\d+)>', result['prompt']+'\n'+selected['prompt']+'\n'+descriptions)
    if any(not 1 <= int(tag) <= 1+len(selected['images']) for tag in tags):
        raise ValueError('Render prompt references an image not selected for H3')
    for name,data in [('render_contract.json',contract),('prompt_compilation.json',result),
                      ('renderer_references.json',selected)]:
        (output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'render_skill_used.md').write_text(skill,encoding='utf-8')
    boundary='起始和结束状态应与给定条件一致，中间按动作描述自然丰富；收势是在目标终态稳定下来，不意味着回到起始位置。'
    prompt=result['prompt']+'\n'+boundary+'\nIf the endpoint is moving, continue through it; recovery does not mean stopping.'
    prompt += '\n' + selected['prompt']
    if descriptions:
        prompt += '\n' + descriptions
    (output/'render_prompt.txt').write_text(prompt,encoding='utf-8')
    return prompt
