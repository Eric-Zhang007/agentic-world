"""Deterministic coarse articulation, NPC updates and proxy-world checks."""
from copy import deepcopy
import math
import numpy as np
from whitebox.world import terrain_height, validate_scene, put_entity
from whitebox.posture import POSTURES,weights,blend,contacts,anchors,body_size
from whitebox.surfaces import surface_world, floor_height
from whitebox.locomotion import GRAVITY,GAIT_SPEEDS,check_gait,jump_geometry,jump_height


def rotation(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def smooth(t):
    t=max(0.,min(1.,t));return t*t*(3-2*t)


def motion_curve(origin, delta, start_velocity, end_velocity, duration, fraction):
    """Cubic Hermite root path with explicit world-space boundary velocities."""
    u=fraction; p=np.asarray(origin,float); d=np.asarray(delta,float)
    v0=np.asarray(start_velocity,float); v1=np.asarray(end_velocity,float)
    position=p+(-2*u**3+3*u*u)*d+(u**3-2*u*u+u)*duration*v0+(u**3-u*u)*duration*v1
    velocity=(-6*u*u+6*u)*d/duration+(3*u*u-4*u+1)*v0+(3*u*u-2*u)*v1
    return position,velocity


def blocked(position, actor, scene, entities):
    size=body_size(actor)
    surfaces=surface_world(scene)
    if surfaces and surfaces.body_blocked(position,size,foot_clearance=.005 if actor.get('support_state')=='airborne' else .35):return 'static_surface'
    radius=actor['size'][0]/2
    x,_,z=position;w,d=scene['terrain']['size']
    if abs(x)+radius>w/2 or abs(z)+radius>d/2:return 'terrain_boundary'
    for obj in entities:
        if obj['id']==actor['id'] or obj['kind']=='staff' or not obj.get('solid',True):continue
        delta=np.array(position)-obj['position']
        actor_bottom=position[1];actor_top=actor_bottom+size[1]
        other_bottom=obj['position'][1] if obj['kind']=='actor' else obj['position'][1]-obj['size'][1]/2
        other_top=other_bottom+(body_size(obj)[1] if obj['kind']=='actor' else obj['size'][1])
        if actor_top<=other_bottom or actor_bottom>=other_top:continue
        if obj['kind']=='actor':
            if np.linalg.norm(delta[[0,2]])<radius+obj['size'][0]/2:return obj['id']
        else:
            local=rotation(-obj.get('rotation_y',0))@delta
            if abs(local[0])<obj['size'][0]/2+radius and abs(local[2])<obj['size'][2]/2+radius:
                return obj['id']
    return None


def validate_geometry(scene):
    errors=[]
    for actor in scene['entities']:
        if actor['kind']!='actor':continue
        obstacle=blocked(actor['position'],actor,scene,scene['entities'])
        if obstacle:errors.append({'actor':actor['id'],'collision':obstacle})
        floor=floor_height(scene,actor['position'][0],actor['position'][2],actor['position'][1])
        if floor is None or abs(actor['position'][1]-floor)>.08:errors.append({'actor':actor['id'],'unsupported_height':None if floor is None else actor['position'][1]-floor})
    return errors


def bend(a,b,length,pole):
    """Equal-length two-bone IK; finite reach failure is reported, not hidden."""
    a=np.array(a);b=np.array(b);d=b-a;distance=np.linalg.norm(d)
    direction=d/max(distance,1e-9)
    p=np.array(pole);p=p-direction*(p@direction)
    if np.linalg.norm(p)<1e-6:p=np.cross(direction,[0,0,1])
    p/=max(np.linalg.norm(p),1e-9)
    return ((a+b)/2+p*math.sqrt(max(0,length**2-(distance/2)**2))).tolist(),distance>2*length+.005


def rig(actor, time, moved, phase=None, progress=0., held=None, terrain=None, scene=None):
    root=np.array(actor['position'],float);h=actor['size'][1];yaw=actor.get('rotation_y',0)
    r=rotation(yaw);pose={};errors=[]
    def ground_at(p):
        if actor.get('support_state')=='airborne':return root[1]
        ground=floor_height(scene,p[0],p[2],root[1]) if scene else (terrain_height(terrain,p[0],p[2]) if terrain else root[1])
        if ground is None:
            errors.append({'type':'unsupported_contact','actor':actor['id'],'position':list(p)})
            return root[1]
        return ground
    def point(x,y,z):return (root+r@np.array([x*h,y*h,z*h])).tolist()
    pose.update(pelvis=point(0,.49,0),chest=point(0,.74,0),head=point(0,.93,0))
    gait=0. if actor.get('asset_id')=='gta_human' else math.sin(time*7)*min(1,moved*24/.6)
    for side,sign in [('left',-1),('right',1)]:
        pose[side+'_shoulder']=point(sign*.13,.75,0)
        pose[side+'_hip']=point(sign*.09,.49,0)
        foot=point(sign*.12,0,gait*sign*.09)
        ground=ground_at(foot)
        foot[1]=ground+h*(.044+max(0,gait*sign)*.055)
        pose[side+'_foot']=foot
        pose[side+'_hand']=point(sign*.18,.54,.13)
    custom=any(weights(actor).get(name,0)>0 for name in ('crouched','low_guard'))
    center_local=(0,.62,.20)
    if custom:
        def anchor(key,sign=1):
            v=sum((np.array(anchors(name)[key])*weight for name,weight in weights(actor).items()),np.zeros(3))
            v[0]*=sign
            return point(*v)
        for key in ('pelvis','chest','head'):pose[key]=anchor(key)
        for side,sign in [('left',-1),('right',1)]:
            for key in ('shoulder','hip','foot','hand'):pose[side+'_'+key]=anchor(key,sign)
            for key in ('foot','hand'):
                p=pose[side+'_'+key]
                p[1]+=ground_at(p)-root[1]
        center_local=sum((np.array(anchors(name)['weapon_center'])*weight for name,weight in weights(actor).items()),np.zeros(3))
    weapon=None
    if held:
        if weights(actor).get('low_guard',0)>0:raise ValueError('Four-contact posture cannot also hold this staff')
        if phase and phase['type']=='sweep':
            angle=math.radians(phase['start_angle_deg']+(phase['end_angle_deg']-phase['start_angle_deg'])*smooth(progress))
        elif phase and phase['type']=='recover':
            begin=actor.get('weapon_angle',55.)
            angle=math.radians(begin+(55.-begin)*smooth(progress))
        else:angle=math.radians(actor.get('weapon_angle',55.))
        elevation=math.radians(actor.get('weapon_elevation_deg',0.))
        direction=r@np.array([math.sin(angle)*math.cos(elevation),math.sin(elevation),
                              math.cos(angle)*math.cos(elevation)])
        center=np.array(point(*center_local))
        for side,sign in [('left',-1),('right',1)]:
            if actor.get('grip','both') in ('both',side):pose[side+'_hand']=(center+direction*sign*.18).tolist()
        length=held.get('length',held['size'][1])
        weapon={'a':(center-direction*length*.28).tolist(),'b':(center+direction*length*.72).tolist()}
    for side,sign in [('left',-1),('right',1)]:
        elbow,bad=bend(pose[side+'_shoulder'],pose[side+'_hand'],h*.19,r@np.array([sign,0,-.5]))
        knee,badleg=bend(pose[side+'_hip'],pose[side+'_foot'],h*.255,r@np.array([0,0,1]))
        pose[side+'_elbow']=elbow;pose[side+'_knee']=knee
        # Rigid dataset actors have no articulated leg-length contract.
        if bad or (badleg and actor.get('asset_id')!='gta_human'):
            errors.append({'actor':actor['id'],'unreachable_limb':side})
    return pose,weapon,errors


def validate_action(scene, action):
    if not isinstance(action,dict):raise ValueError('Action must be an object')
    ids={e['id']:e for e in scene['entities']}
    actor=ids.get(action.get('actor_id'))
    if not actor or actor['kind']!='actor':raise ValueError('Known actor required')
    policies=action.get('npc_behaviors',[])
    if not isinstance(policies,list):raise ValueError('npc_behaviors must be a list')
    policy_scene=deepcopy(scene);policy_ids=set()
    for change in policies:
        identity=change.get('id')
        if (identity not in ids or ids[identity]['kind']!='actor' or identity==action['actor_id']
                or identity in policy_ids or not change.get('reason')
                or set(change)-{'id','behavior','reason'}):
            raise ValueError('NPC policy needs a unique non-primary actor and causal planning reason')
        policy_ids.add(identity)
        behavior=change.get('behavior')
        if not isinstance(behavior,dict) or behavior.get('type') not in {'patrol','guard'}:
            raise ValueError('NPC policy must be executable patrol or guard behavior')
        next(e for e in policy_scene['entities'] if e['id']==identity)['behavior']=deepcopy(behavior)
    if policies:validate_scene(policy_scene)
    if not action.get('phases'):raise ValueError('Action phases required')
    weapon_angle=actor.get('weapon_angle',55.)
    velocity=np.asarray(actor.get('motion',{}).get('velocity',[0,0,0]),float)
    posture=deepcopy(weights(actor))
    for phase in action['phases']:
        if not isinstance(phase,dict):raise ValueError('Phase must be an object')
        if phase.get('type') not in {'move','jump','sweep','recover','wait','posture'}:raise ValueError('Unknown executable action')
        if phase['type']=='posture':
            if phase.get('target_posture') not in POSTURES:raise ValueError('Unknown target posture')
            if phase['target_posture']=='low_guard' and any(e.get('holder')==actor['id'] for e in ids.values()):
                raise ValueError('Held staff prevents four-contact posture')
            posture={phase['target_posture']:1.}
        duration=phase['duration_sec']
        if not isinstance(duration,(int,float)) or not math.isfinite(duration) or not .5<=duration<=8:raise ValueError('Invalid duration')
        if not phase.get('criterion'):raise ValueError('Phase criterion required')
        if np.linalg.norm(velocity)>1e-8 and (phase['type'] not in {'move','jump'} or 'end_velocity' not in phase):
            raise ValueError('Moving boundary requires a move with explicit end_velocity; decelerate before waiting')
        if 'end_velocity' in phase:
            velocity=np.asarray(phase['end_velocity'],float)
            if (phase['type'] not in {'move','jump'} or velocity.shape!=(3,) or not np.isfinite(velocity).all()
                    or abs(velocity[1])>1e-6 or np.linalg.norm(velocity)>4):
                raise ValueError('end_velocity must be a bounded horizontal move velocity in m/s')
        camera_delta=np.array(phase.get('camera_delta',[0,0,0]),float)
        if 'camera_end_motion' in phase:
            camera_motion=phase['camera_end_motion']
            if not isinstance(camera_motion,dict) or set(camera_motion)!={'translation_velocity','orbit_velocity_deg','distance_velocity'}:
                raise ValueError('camera_end_motion requires translation_velocity, orbit_velocity_deg, distance_velocity')
            translation=np.asarray(camera_motion['translation_velocity'],float)
            if translation.shape!=(3,) or not np.isfinite(translation).all():raise ValueError('Invalid camera translation velocity')
            if not all(isinstance(camera_motion[k],(int,float)) and math.isfinite(camera_motion[k]) for k in ['orbit_velocity_deg','distance_velocity']):
                raise ValueError('Invalid camera angular/distance velocity')
            if 'camera_follow_actor' in phase and np.linalg.norm(translation)>1e-8:
                raise ValueError('Following camera uses actor translation; do not specify additional translation velocity')
        if 'camera_orbit_yaw_deg' in phase:
            angle=phase['camera_orbit_yaw_deg']
            if (isinstance(angle,bool) or not isinstance(angle,(int,float)) or
                    not math.isfinite(angle) or abs(angle)>180 or 'camera_delta' in phase):
                raise ValueError('Camera orbit requires a finite yaw of -180..180 degrees and excludes camera_delta')
            if np.linalg.norm(np.subtract(scene['camera']['position'],scene['camera']['target'])[[0,2]])<1e-6:
                raise ValueError('Camera orbit needs a nonzero horizontal radius')
        if 'camera_follow_actor' in phase:
            follow=ids.get(phase['camera_follow_actor'])
            if not follow or follow['kind']!='actor' or 'camera_delta' in phase:
                raise ValueError('Camera follow requires an existing actor and excludes camera_delta')
        distance_delta=phase.get('camera_distance_delta',0.)
        if not isinstance(distance_delta,(int,float)) or not math.isfinite(distance_delta) or not 0<=distance_delta<=5:
            raise ValueError('Camera distance delta must be a finite outward dolly of 0..5m')
        if camera_delta.shape!=(3,) or not np.isfinite(camera_delta).all() or np.linalg.norm(camera_delta)>5:
            raise ValueError('Invalid camera displacement')
        if phase['type'] in {'move','jump'}:
            delta=np.array(phase['delta'],float)
            if delta.shape!=(3,) or not np.isfinite(delta).all() or abs(delta[1])>1e-6 or np.linalg.norm(delta)>4:
                raise ValueError('Move must be bounded horizontal motion')
            check_gait(phase,posture)
        if phase['type']=='jump':
            if not .5<=duration<=1.2 or 'end_velocity' not in phase:
                raise ValueError('Jump requires 0.5..1.2s flight duration and end_velocity')
            if not math.isclose(posture.get('standing',0),1.):raise ValueError('Jump requires standing takeoff')
        if phase['type']=='sweep':
            if not any(e['kind']=='staff' and e.get('holder')==actor['id'] for e in ids.values()):raise ValueError('Sweep needs held staff')
            angles=[phase.get('start_angle_deg'),phase.get('end_angle_deg')]
            if not all(isinstance(x,(int,float)) and math.isfinite(x) and abs(x)<=150 for x in angles):raise ValueError('Invalid sweep angles')
            if not 45<=abs(angles[1]-angles[0])<=200:raise ValueError('Sweep arc must be 45..200 degrees')
            if not math.isclose(angles[0],weapon_angle,abs_tol=1e-6):raise ValueError('Sweep must start at committed weapon angle')
            weapon_angle=angles[1]
        if phase['type']=='recover':weapon_angle=55.
    return action


def point_segment_distance(point,a,b):
    a=np.array(a);b=np.array(b);p=np.array(point);d=b-a
    t=np.clip((p-a)@d/max(d@d,1e-9),0,1)
    return float(np.linalg.norm(p-a-t*d))


def prepare_scene(scene,time=0.):
    scene=deepcopy(scene)
    for actor in scene['entities']:
        if actor['kind']!='actor':continue
        held=next((e for e in scene['entities'] if e['kind']=='staff' and e.get('holder')==actor['id']),None)
        pose,weapon,errors=rig(actor,time,0.,held=held,terrain=scene['terrain'],scene=scene)
        if errors:raise ValueError('Initial rig invalid: '+str(errors))
        actor['pose']=pose
        if held:
            held['weapon']=weapon;held['position']=((np.array(weapon['a'])+weapon['b'])/2).tolist()
    return scene


def npc_proposal(entity, index, dt):
    """One policy clock for planned actions and keyboard-controlled previews."""
    old=np.array(entity['position'],float);proposed=old.copy();behavior=entity['behavior']
    if weights(entity).get('low_guard',0)>1e-8:return proposed
    if behavior['type']=='guard':
        offset=np.array(index[behavior['target']]['position'])-old;offset[1]=0
        dist=np.linalg.norm(offset);desired=behavior.get('distance',2.4)
        direction=offset/max(dist,1e-9)
        if abs(dist-desired)>.2:proposed+=direction*np.sign(dist-desired)*min(abs(dist-desired),behavior.get('speed',.6)*dt)
        elif behavior.get('strafe',True):proposed+=np.array([direction[2],0,-direction[0]])*behavior.get('speed',.6)*.35*dt
        entity['rotation_y']=math.atan2(offset[0],offset[2])
    elif behavior['type']=='patrol':
        waypoints=behavior['waypoints'];n=entity.get('patrol_index',behavior.get('patrol_index',0))%len(waypoints)
        delta=np.array(waypoints[n])-old;delta[1]=0;dist=np.linalg.norm(delta)
        if dist<.12:entity['patrol_index']=(n+1)%len(waypoints)
        elif dist>0:
            proposed+=delta/dist*min(dist,behavior.get('speed',.6)*dt)
            entity['rotation_y']=math.atan2(delta[0],delta[2])
    return proposed


def simulate(scene, action, start_time=0., fps=24):
    if action.get('type')=='controls':
        from whitebox.controls import simulate_controls
        return simulate_controls(scene,action,start_time,fps)
    validate_scene(scene);validate_action(scene,action)
    if validate_geometry(scene):raise ValueError('Initial geometry invalid: '+str(validate_geometry(scene)))
    entities=deepcopy(scene['entities']);index={e['id']:e for e in entities};frames=[];events=[];violations=[]
    # Policies are prospective intentions; accepted state remains untouched until commit.
    for change in action.get('npc_behaviors',[]):
        index[change['id']]['behavior']=deepcopy(change['behavior'])
        index[change['id']].pop('patrol_index',None)
    primary=index[action['actor_id']];time=start_time;phase_reports=[]
    previous_weapon={};registered=set()
    for phase_index,phase in enumerate(action['phases']):
        posture_start=deepcopy(weights(primary))
        origin=np.array(primary['position']);count=round(phase['duration_sec']*fps);phase_events=[]
        start_velocity=primary.get('motion',{}).get('velocity',[0,0,0])
        landing=None;launch=None
        if phase['type']=='jump':
            landing,launch=jump_geometry(scene,origin,phase['delta'],count/fps)
            event={'type':'planned_takeoff','actor':primary['id'],'time':time,
                   'position':origin.tolist(),'scope':'planned geometry; requires video confirmation'}
            events.append(event);phase_events.append(event)
        camera_start=deepcopy(scene['camera']) if not frames else deepcopy(frames[-1]['camera'])
        camera_motion_start=camera_start.get('motion',{'translation_velocity':[0,0,0],'orbit_velocity_deg':0.,'distance_velocity':0.})
        if 'camera_end_motion' not in phase and (np.linalg.norm(camera_motion_start['translation_velocity'])>1e-8 or abs(camera_motion_start['orbit_velocity_deg'])>1e-8 or abs(camera_motion_start['distance_velocity'])>1e-8):
            raise ValueError('Moving camera requires explicit camera_end_motion; no implicit stop at a segment boundary')
        follow_origin=np.array(index[phase['camera_follow_actor']]['position']) if 'camera_follow_actor' in phase else None
        for i in range(count):
            local=(i+1)/count;progress=min(1.,local/.75);motion={}
            if phase['type']=='posture':
                primary['posture_weights']=blend(posture_start,phase['target_posture'],smooth(local))
                primary['support_contacts']=contacts(primary)
            for entity in entities:
                if entity['kind']!='actor':continue
                old=np.array(entity['position'],float);proposed=old.copy();behavior=entity['behavior']
                supported_low=any(weights(entity).get(name,0)>0 for name in ('crouched','low_guard'))
                if entity['id']==primary['id']:
                    if phase['type'] in {'move','jump'}:
                        if supported_low and phase.get('gait')!='crouch':raise ValueError('Low movement requires explicit crouch gait')
                        if 'end_velocity' in phase:
                            proposed,velocity=motion_curve(origin,phase['delta'],start_velocity,
                                phase['end_velocity'],count/fps,local)
                            primary['motion']={'velocity':velocity.tolist(),'phase':'locomotion' if np.linalg.norm(velocity)>1e-8 else 'stationary',
                                               'provenance':'PLANNED','scope':'coarse horizontal root velocity'}
                            limit=GAIT_SPEEDS.get(phase.get('gait'),4.)
                            if np.linalg.norm(velocity[[0,2]])>limit+1e-6:
                                raise ValueError('Trajectory exceeds gait speed; increase duration or reduce displacement')
                        else:proposed=origin+smooth(progress)*np.array(phase['delta'])
                        if phase.get('gait'):entity['gait']=phase['gait']
                        if phase['type']=='jump':
                            proposed[1]=jump_height(origin[1],launch,(i+1)/fps)
                            entity['support_state']='airborne'
                            entity['vertical_velocity']=launch-GRAVITY*(i+1)/fps
                            entity['support_contacts']=[]
                elif behavior['type'] in {'guard','patrol'} and not supported_low:
                    proposed=npc_proposal(entity,index,1/fps)
                obstacle=None
                for alpha in np.linspace(0,1,max(2,math.ceil(np.linalg.norm(proposed-old)/.01)+1))[1:]:
                    sample=old+alpha*(proposed-old)
                    airborne=entity['id']==primary['id'] and phase['type']=='jump'
                    if airborne:
                        sample[1]=jump_height(origin[1],launch,(i+alpha)/fps)
                        floor=sample[1]
                        terrain=surface_world(scene)
                        if terrain is None and sample[1]<terrain_height(scene['terrain'],sample[0],sample[2])-1e-5:
                            obstacle='terrain_intersection'
                            break
                    else:
                        floor=floor_height(scene,sample[0],sample[2],old[1])
                        if floor is None:
                            obstacle='unsupported_surface'
                            break
                        sample[1]=floor
                    obstacle=blocked(sample,entity,scene,entities)
                    if obstacle:break
                if not obstacle:proposed[1]=floor
                if obstacle:
                    if entity['id']==primary['id'] and phase['type']=='jump':
                        raise ValueError('Jump trajectory intersects '+obstacle+
                            '; phase='+str(phase_index)+'; flight_time='+str((i+alpha)/fps)+
                            '; position='+str(sample.tolist()))
                    if 'motion' in entity:entity['motion']={'velocity':[0.,0.,0.],'phase':'stationary','provenance':'PLANNED','scope':'blocked by geometry'}
                    if entity['id']==primary['id'] and phase['type'] in {'move','posture'}:
                        violations.append({'type':'blocked_player','phase':phase_index,'time':time,'by':obstacle})
                else:entity['position']=proposed.tolist()
                if entity['id']==primary['id'] and phase['type']=='jump' and i==count-1:
                    entity['position']=landing.tolist()
                    entity['support_state']='grounded';entity['vertical_velocity']=0.
                    entity['support_contacts']=contacts(entity)
                    radius=entity['size'][0]/4
                    for dx,dz in ((0,0),(-radius,0),(radius,0),(0,-radius),(0,radius)):
                        support=floor_height(scene,landing[0]+dx,landing[2]+dz,landing[1])
                        if support is None or abs(support-landing[1])>.15:
                            raise ValueError('Jump landing footprint has insufficient support')
                    event={'type':'planned_landing','actor':entity['id'],'time':time+1/fps,
                        'position':landing.tolist(),'scope':'planned geometry; requires video confirmation'}
                    events.append(event);phase_events.append(event)
                motion[entity['id']]=float(np.linalg.norm(np.array(entity['position'])-old))
            for entity in entities:
                if entity['kind']!='actor':continue
                held=next((e for e in entities if e['kind']=='staff' and e.get('holder')==entity['id']),None)
                pose,weapon,errors=rig(entity,time+1/fps,motion[entity['id']],phase if entity['id']==primary['id'] else None,progress,held,scene['terrain'],scene)
                entity['pose']=pose;violations.extend(errors)
                if held:
                    held['weapon']=weapon;held['position']=((np.array(weapon['a'])+weapon['b'])/2).tolist()
                    if phase['type']=='sweep' and entity['id']==primary['id']:
                        oldweapon=previous_weapon.get(held['id'],weapon)
                        for target in entities:
                            if target['kind']!='actor' or target['id']==entity['id']:continue
                            hit=False
                            for alpha in np.linspace(0,1,5):
                                a=np.array(oldweapon['a'])*(1-alpha)+np.array(weapon['a'])*alpha
                                b=np.array(oldweapon['b'])*(1-alpha)+np.array(weapon['b'])*alpha
                                for y in (.3,.55,.8):
                                    p=np.array(target['position'])+[0,target['size'][1]*y,0]
                                    hit|=point_segment_distance(p,a,b)<target['size'][0]/2+held['size'][0]/2
                            key=(phase_index,held['id'],target['id'])
                            if hit and key not in registered:
                                event={'type':'proxy_contact','weapon':held['id'],'target':target['id'],'time':time+1/fps,
                                       'scope':'approximate sampled geometry; no damage/real-world hit inferred'}
                                events.append(event);phase_events.append(event);registered.add(key)
                        previous_weapon[held['id']]=deepcopy(weapon)
            time+=1/fps
            camera=deepcopy(camera_start)
            delta=np.array(phase.get('camera_delta',[0,0,0]),float)*smooth(progress)
            camera_end_motion=phase.get('camera_end_motion')
            orbit=math.radians(phase.get('camera_orbit_yaw_deg',0))*smooth(local)
            distance=phase.get('camera_distance_delta',0)*smooth(local)
            if camera_end_motion is not None:
                delta,translation_velocity=motion_curve([0,0,0],phase.get('camera_delta',[0,0,0]),
                    camera_motion_start['translation_velocity'],camera_end_motion['translation_velocity'],count/fps,local)
                orbit,orbit_velocity=motion_curve(0.,phase.get('camera_orbit_yaw_deg',0),
                    camera_motion_start['orbit_velocity_deg'],camera_end_motion['orbit_velocity_deg'],count/fps,local)
                distance,distance_velocity=motion_curve(0.,phase.get('camera_distance_delta',0),
                    camera_motion_start['distance_velocity'],camera_end_motion['distance_velocity'],count/fps,local)
                orbit=math.radians(float(orbit))
                camera['motion']={'translation_velocity':translation_velocity.tolist(),
                    'orbit_velocity_deg':float(orbit_velocity),'distance_velocity':float(distance_velocity)}
            if follow_origin is not None:
                delta=np.array(index[phase['camera_follow_actor']]['position'])-follow_origin
            camera['position']=(np.array(camera_start['position'])+delta).tolist()
            camera['target']=(np.array(camera_start['target'])+delta).tolist()
            if 'camera_orbit_yaw_deg' in phase or camera_end_motion is not None:
                offset=np.subtract(camera_start['position'],camera_start['target'])
                offset=rotation(orbit)@offset
                camera['position']=(np.array(camera['target'])+offset).tolist()
                if 'up' in camera_start:camera['up']=(rotation(orbit)@np.asarray(camera_start['up'])).tolist()
            if phase.get('camera_distance_delta',0) or camera_end_motion is not None:
                basis=camera if 'camera_orbit_yaw_deg' in phase or camera_end_motion is not None else camera_start
                outward=np.subtract(basis['position'],basis['target'])
                outward=outward/np.linalg.norm(outward)
                camera['position']=(np.array(camera['position'])+outward*distance).tolist()
            frames.append({'time':time,'phase':phase_index,'entities':deepcopy(entities),'camera':camera})
        if phase['type']=='sweep':primary['weapon_angle']=phase['end_angle_deg']
        if phase['type']=='recover':primary['weapon_angle']=55.
        frames[-1]['entities']=deepcopy(entities)
        phase_reports.append({'index':phase_index,'type':phase['type'],'criterion':phase['criterion'],
                              'requested_duration':phase['duration_sec'],'executed_duration':count/fps,
                              'start':time-count/fps,'end':time,'proxy_contacts':phase_events})
    end=deepcopy(scene);end['entities']=deepcopy(entities);end['camera']=deepcopy(frames[-1]['camera'])
    grip_errors=[];support_clearances=[]
    for frame in frames:
        actors={e['id']:e for e in frame['entities'] if e['kind']=='actor'}
        for actor in actors.values():
            if actor.get('support_state')=='airborne':continue
            foot_clearances=[]
            for p in [actor['pose']['left_foot'],actor['pose']['right_foot']]:
                floor=floor_height(scene,p[0],p[2],actor['position'][1])
                if floor is not None:foot_clearances.append(p[1]-actor['size'][1]*.044-floor)
            if foot_clearances:support_clearances.append(min(foot_clearances))
            if weights(actor).get('low_guard',0)>1-1e-8:
                for side in ('left','right'):
                    p=actor['pose'][side+'_hand']
                    floor=floor_height(scene,p[0],p[2],actor['position'][1])
                    if floor is not None:support_clearances.append(p[1]-actor['size'][1]*.044-floor)
        for staff in frame['entities']:
            if staff['kind']!='staff':continue
            for side in ('left','right'):
                if actors[staff['holder']].get('grip','both') not in ('both',side):continue
                grip_errors.append(point_segment_distance(actors[staff['holder']]['pose'][side+'_hand'],staff['weapon']['a'],staff['weapon']['b']))
    if grip_errors and max(grip_errors)>1e-6:violations.append({'type':'detached_grip'})
    if min(support_clearances,default=0)<-1e-6 or max(support_clearances,default=0)>.02:
        violations.append({'type':'unsupported_or_penetrating_feet'})
    report={'valid':not violations,'violations':violations[:50],'violation_count':len(violations),
            'max_grip_axis_error':max(grip_errors,default=0),
            'support_sole_clearance_range':[min(support_clearances,default=0),max(support_clearances,default=0)],
            'phases':phase_reports,'events':events,'entity_ids_preserved':set(index)=={e['id'] for e in end['entities']},
            'npc_displacements':{e['id']:float(np.linalg.norm(np.array(e['position'])-next(x['position'] for x in scene['entities'] if x['id']==e['id'])))
                                 for e in entities if e['kind']=='actor' and e['behavior']['type']!='player'},
            'terrain_unchanged':end['terrain']==scene['terrain'],
            'npc_samples':[{'time':f['time'],'actors':[{'id':e['id'],'position':e['position'],
                            'patrol_index':e.get('patrol_index'),'behavior':e['behavior']['type']} for e in f['entities']
                            if e['kind']=='actor' and e['behavior']['type']!='player']}
                           for f in (frames[0],frames[len(frames)//2],frames[-1])],
            'limitation':'Procedural gait/IK and sampled proxy contacts; not a full rigid-body or biomechanics simulator'}
    return {'fps':fps,'frames':frames},end,report


def commit(db, candidate_scene, timeline, report, visual_review, action, expected_revision):
    with db.transaction():
        return _commit(db,candidate_scene,timeline,report,visual_review,action,expected_revision)


def _commit(db, candidate_scene, timeline, report, visual_review, action, expected_revision):
    if not db.connection.in_transaction:db.connection.execute('BEGIN IMMEDIATE')
    if not report['valid'] or visual_review.get('decision')!='ACCEPT' or not visual_review.get('evidence'):
        raise ValueError('Numerical AND visual acceptance required')
    if db.get_meta('whitebox_revision')!=expected_revision:raise ValueError('Stale candidate')
    from whitebox.world import scene_from_db
    before=scene_from_db(db)
    verified_timeline,verified_end,verified_report=simulate(before,action,db.get_meta('tick'),timeline['fps'])
    if candidate_scene!=verified_end or timeline!=verified_timeline or not verified_report['valid']:
        raise ValueError('Candidate does not match deterministic execution of the committed world')
    if before['terrain']!=candidate_scene['terrain']:raise ValueError('Action cannot rewrite terrain')
    old={e['id']:e for e in before['entities']};new={e['id']:e for e in candidate_scene['entities']}
    if set(old)!=set(new):raise ValueError('Action cannot create/remove entities')
    for identity,e in old.items():
        if e['kind'] not in {'actor','staff'} and e!=new[identity]:raise ValueError('Static entity changed')
    if not math.isclose(timeline['frames'][0]['time'],db.get_meta('tick')+1/timeline['fps'],abs_tol=1e-6):
        raise ValueError('Timeline does not continue committed time')
    with db.transaction():
        for e in candidate_scene['entities']:put_entity(db,e,'Committed numerically and visually reviewed proxy action')
        db.set_meta('whitebox_header',{k:v for k,v in candidate_scene.items() if k!='entities'})
        db.set_meta('tick',timeline['frames'][-1]['time'])
        db.set_meta('whitebox_revision',expected_revision+1)
        history=db.get_meta('whitebox_history',[])
        history.append({'action':action,'start':timeline['frames'][0]['time'],'end':db.get_meta('tick'),
                        'events':report['events'],'visual_review':visual_review,
                        'cameras':[f['camera'] for f in timeline['frames']],
                        'actor_samples':[e['position'] for f in timeline['frames'] for e in f['entities'] if e['kind']=='actor']})
        db.set_meta('whitebox_history',history)
