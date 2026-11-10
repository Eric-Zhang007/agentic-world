"""Reuse WorldDB; Godot Y-up is an explicit boundary, not a new world store."""
from copy import deepcopy
import math
import numpy as np
from state.models import WorldObject


KINDS = {'actor', 'box', 'staff', 'stairs'}


def terrain_height(terrain, x, z):
    seed = terrain['seed']
    a = terrain['height_amplitude']
    blend = max(0., min(1., (math.hypot(x, z)-terrain.get('flatten_radius', 5.))/3.))
    return a*blend*(.5*math.sin(.31*x+seed*.013)*math.cos(.27*z-seed*.017)
                    +.25*math.sin(.61*(x+z)+seed*.007))


def finite_vector(value, length=3):
    return isinstance(value, list) and len(value)==length and all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in value)


def validate_scene(scene):
    if not isinstance(scene,dict):raise ValueError('Scene must be an object')
    if scene.get('schema_version') != 1:
        raise ValueError('Expected whitebox schema_version=1')
    t = scene['terrain']
    if not finite_vector(t['size'], 2) or min(t['size']) <= 0:
        raise ValueError('Terrain must have positive X/Z dimensions')
    if not isinstance(t['seed'], int) or not 0 <= t['height_amplitude'] <= 3:
        raise ValueError('Invalid terrain seed/amplitude')
    if type(t['resolution']) is not int or not 4 <= t['resolution'] <= 256 or not 0 <= t.get('flatten_radius',5) <= min(t['size'])/2:
        raise ValueError('Invalid terrain sampling/flat area')
    entities = scene['entities']; ids = [e['id'] for e in entities]
    if any(not isinstance(identity, str) or not identity.strip() for identity in ids):
        raise ValueError('Persistent IDs must be nonempty strings')
    if not entities or len(ids)!=len(set(ids)):
        raise ValueError('Scene must have unique persistent IDs')
    for e in entities:
        if e['kind'] not in KINDS or not finite_vector(e['position']) or not finite_vector(e['size']) or min(e['size'])<.02:
            raise ValueError('Invalid entity geometry: '+e['id'])
        if not math.isfinite(e.get('rotation_y',0)) or not finite_vector(e['color']) or not all(0<=c<=1 for c in e['color']):
            raise ValueError('Invalid orientation/color')
        if e.get('provenance') not in {'INFERRED','AGENT_CREATED','OBSERVED'}:
            raise ValueError('Explicit scene provenance required')
        if not e.get('reason'):
            raise ValueError('Every entity needs an evidence/creation reason')
        if e['kind']=='actor':
            if e.get('support_state','grounded')!='grounded':
                raise ValueError('Airborne segment endpoints are not supported; finish landing before the boundary')
            elevation=e.get('weapon_elevation_deg',0.)
            if (isinstance(elevation,bool) or not isinstance(elevation,(int,float))
                    or not math.isfinite(elevation) or not -90<=elevation<=90):
                raise ValueError('Invalid weapon elevation in degrees')
            if 'motion' in e:
                motion=e['motion']
                if (not isinstance(motion,dict) or not finite_vector(motion.get('velocity'))
                        or abs(motion['velocity'][1])>1e-6 or np.linalg.norm(motion['velocity'])>4
                        or motion.get('phase') not in {'locomotion','stationary'}):
                    raise ValueError('Invalid coarse actor motion state')
            from whitebox.posture import validate as validate_posture
            validate_posture(e)
            behavior = e.get('behavior',{})
            if not isinstance(behavior,dict):raise ValueError('Actor behavior must be an object')
            if behavior.get('type') not in {'player','guard','patrol'}:
                raise ValueError('Every actor needs an executable behavior')
            if behavior['type']=='guard' and behavior.get('target') not in ids:
                raise ValueError('Guard needs a known target')
            if behavior['type']=='patrol' and (len(behavior.get('waypoints',[]))<2 or
                not all(finite_vector(p) for p in behavior['waypoints'])):
                raise ValueError('Patrol needs two or more world waypoints')
            speed=behavior.get('speed',.6)
            if not isinstance(speed,(float,int)) or not math.isfinite(speed) or not 0<=speed<=4:
                raise ValueError('Invalid behavior speed')
            if behavior['type']=='guard' and (not isinstance(behavior.get('distance',2.4),(int,float)) or
                not math.isfinite(behavior.get('distance',2.4)) or behavior.get('distance',2.4)<.5):
                raise ValueError('Invalid guard distance')
        if e['kind']=='staff' and e.get('holder') not in ids:
            raise ValueError('Staff requires holder')
        if e['kind']=='staff' and (not isinstance(e.get('length',e['size'][1]),(int,float)) or
            not math.isfinite(e.get('length',e['size'][1])) or e.get('length',e['size'][1])<=0):
            raise ValueError('Invalid staff length')
        if e.get('appearance_bbox') is not None:
            b=e['appearance_bbox']
            if not finite_vector(b,4) or not 0<=b[0]<b[2]<=1 or not 0<=b[1]<b[3]<=1:
                raise ValueError('Invalid appearance crop')
        if e['kind']=='stairs' and not 1 <= e.get('steps',5) <= 30:
            raise ValueError('Invalid stair steps')
    kinds={e['id']:e['kind'] for e in entities}
    holders=[]
    for e in entities:
        if e['kind']=='staff':
            if kinds[e['holder']]!='actor' or e['holder'] in holders:raise ValueError('This rig supports one held staff per actor')
            holders.append(e['holder'])
        if e['kind']=='actor' and e['behavior']['type']=='guard':
            target=e['behavior']['target']
            if kinds[target]!='actor' or target==e['id']:raise ValueError('Guard target must be another actor')
    if not any(e['kind']=='actor' and e['behavior']['type']=='player' for e in entities):
        raise ValueError('A player actor is required')
    for key in ('position','target'):
        if not finite_vector(scene['camera'][key]):raise ValueError('Invalid camera')
    from whitebox.godot.render import camera as validate_camera
    validate_camera(scene['camera'])
    if 'motion' in scene['camera']:
        motion=scene['camera']['motion']
        if (not isinstance(motion,dict) or not finite_vector(motion.get('translation_velocity')) or
                any(not isinstance(motion.get(k),(int,float)) or not math.isfinite(motion[k])
                    for k in ('orbit_velocity_deg','distance_velocity'))):
            raise ValueError('Invalid camera motion')
    if np.linalg.norm(np.subtract(scene['camera']['position'],scene['camera']['target']))<.1:
        raise ValueError('Camera has no direction')
    return scene


def put_entity(db, entity, reason):
    e=deepcopy(entity);x,y,z=e.pop('position');yaw=e.pop('rotation_y',0)
    size=e['size'];identity=e.pop('id');provenance=e.pop('provenance')
    old=db.get_object(identity)
    db.save(WorldObject(object_id=identity,category=e.get('category',e['kind']),
        position=[x,-z,y],rotation=[0,0,yaw],bbox_3d=[size[0],size[2],size[1]],
        attributes={'whitebox':e},state=old.state if old else {},provenance=provenance),reason)


def initialize(db, scene):
    validate_scene(scene)
    if db.query_world():raise ValueError('Do not overwrite an existing world')
    with db.transaction():
        db.set_meta('whitebox_header',{k:deepcopy(v) for k,v in scene.items() if k!='entities'})
        db.set_meta('whitebox_revision',0);db.set_meta('tick',0.)
        db.set_meta('whitebox_entity_order',[e['id'] for e in scene['entities']])
        db.set_meta('whitebox_history',[])
        for e in scene['entities']:put_entity(db,e,'Initialize whitebox: '+e['reason'])


def scene_from_db(db):
    scene=deepcopy(db.get_meta('whitebox_header'))
    if scene is None:raise ValueError('Not a whitebox world')
    scene['entities']=[]
    for obj in db.query_world():
        e=deepcopy(obj.attributes['whitebox']);x,y,z=obj.position
        e.update(id=obj.object_id,position=[x,z,-y],rotation_y=obj.rotation[2],provenance=obj.provenance)
        scene['entities'].append(e)
    order=db.get_meta('whitebox_entity_order',[])
    scene['entities'].sort(key=lambda e:order.index(e['id']) if e['id'] in order else len(order))
    return scene


def in_camera(point, camera):
    eye=np.array(camera['position'],float);direction=np.array(camera['target'],float)-eye
    direction/=np.linalg.norm(direction);delta=np.array(point,float)-eye
    depth=float(delta@direction)
    if depth<=0:return False
    return np.linalg.norm(delta-depth*direction) <= depth*math.tan(math.radians(55))


def reconcile(db, edits, reason):
    """Conservative historical gate; unsupported rewrites return to regeneration."""
    if not reason:raise ValueError('Reconciliation needs a causal explanation')
    scene=scene_from_db(db);history=db.get_meta('whitebox_history',[])
    existing={e['id'] for e in scene['entities']}
    candidates=deepcopy(scene)
    for edit in edits:
        if edit.get('op')!='spawn':
            raise ValueError('Historical transform/terrain edits need rollback; use REGENERATE here')
        e=deepcopy(edit['entity'])
        if e['id'] in existing:raise ValueError('Cannot replace an established entity')
        if e.get('provenance')!='AGENT_CREATED':raise ValueError('Creative additions are AGENT_CREATED, not observed facts')
        cameras=[scene['camera']]+[c for h in history for c in h['cameras']]
        radius=max(e['size'])/2
        points=[e['position']]+[(np.array(e['position'])+offset).tolist() for offset in
            ([radius,0,0],[-radius,0,0],[0,radius,0],[0,0,radius],[0,0,-radius])]
        if history and any(in_camera(p,c) for c in cameras for p in points):
            raise ValueError('Addition may contradict accepted camera coverage; regenerate or explicit historical rollback')
        for prior in history:
            for sample in prior['actor_samples']:
                if math.dist([e['position'][0],e['position'][2]],[sample[0],sample[2]]) < radius+.5:
                    raise ValueError('Addition intersects an accepted actor path')
        candidates['entities'].append(e);existing.add(e['id'])
    validate_scene(candidates)
    from whitebox.motion import validate_geometry
    errors=validate_geometry(candidates)
    if errors:raise ValueError('Invalid addition: '+str(errors))
    with db.transaction():
        for edit in edits:put_entity(db,edit['entity'],reason)
        db.set_meta('whitebox_entity_order',db.get_meta('whitebox_entity_order',[])+[e['entity']['id'] for e in edits])
        db.set_meta('whitebox_revision',db.get_meta('whitebox_revision')+1)
        revisions=db.get_meta('whitebox_reconciliations',[])
        revisions.append({'time':db.get_meta('tick'),'reason':reason,'edits':edits})
        db.set_meta('whitebox_reconciliations',revisions)
