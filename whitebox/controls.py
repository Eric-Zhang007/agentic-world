"""Recorded keyboard/mouse input, deterministic at 24 Hz, never a world commit."""
from copy import deepcopy
import math
import numpy as np

from whitebox.world import validate_scene
from whitebox.posture import weights, contacts, blend
from whitebox.surfaces import floor_height
from whitebox.locomotion import GRAVITY, GAIT_SPEEDS

KEYS = {'w', 'a', 's', 'd', 'shift', 'space', 'c', 'stand', 'attack'}
FPS = 24


def input_sample(value):
    keys = value.get('keys', [])
    mouse = value.get('mouse', [0., 0.])
    if not isinstance(keys, list) or any(k not in KEYS for k in keys):
        raise ValueError('Unsupported control key (prone and firearms are not implemented)')
    if not isinstance(mouse, list) or len(mouse) != 2 or not all(
            type(x) in (int, float) and math.isfinite(x) and abs(x) <= 1000 for x in mouse):
        raise ValueError('Mouse requires two finite pixel deltas, bounded at 1000 per tick')
    return {'keys': sorted(set(keys)), 'mouse': list(mouse)}


class Controller:
    def __init__(self, scene, actor_id, start_time=0.):
        from whitebox.motion import prepare_scene, validate_geometry
        validate_scene(scene)
        if validate_geometry(scene):
            raise ValueError('Initial scene intersects geometry or has no support')
        self.scene = prepare_scene(scene)
        self.actor = next(e for e in self.scene['entities'] if e['id'] == actor_id)
        if self.actor['kind'] != 'actor' or self.actor['behavior']['type'] != 'player':
            raise ValueError('Keyboard control requires a player actor')
        self.actor_id = actor_id
        self.time = start_time
        self.events = []
        self.previous_keys = set(scene.get('control_state', {}).get('held_keys', []))
        self.attack_time = None
        self.attack_angles = (55.,-55.)
        self.hit_targets = set()
        self.last_blocker = None
        self.posture_target = 'crouched' if weights(self.actor).get('crouched', 0) > .5 else 'standing'
        self.posture_origin = deepcopy(weights(self.actor))
        self.posture_elapsed = .3
        camera = self.scene['camera']
        offset = np.subtract(camera['position'], camera['target'])
        self.distance = float(np.linalg.norm(offset))
        self.yaw = math.atan2(offset[0], offset[2])
        self.pitch = math.asin(offset[1] / self.distance)
        self.target_offset = np.subtract(camera['target'], self.actor['position'])

    def event(self, kind, **fields):
        self.events.append({'type': kind, 'time': self.time, 'actor': self.actor_id,
                            'scope': 'planned proxy geometry; requires video confirmation', **fields})

    def ground_move(self, entity, delta):
        from whitebox.motion import blocked
        origin = np.array(entity['position'], float)
        last = origin.copy()
        obstacle = None
        for alpha in np.linspace(0, 1, max(2, math.ceil(np.linalg.norm(delta) / .01) + 1))[1:]:
            p = origin + alpha * delta
            floor = floor_height(self.scene, p[0], p[2], last[1])
            if floor is None:
                obstacle = 'unsupported_surface'
                break
            p[1] = floor
            obstacle = blocked(p, entity, self.scene, self.scene['entities'])
            if obstacle:
                break
            last = p
        entity['position'] = last.tolist()
        return obstacle

    def step(self, raw):
        from whitebox.motion import blocked, rig, point_segment_distance, npc_proposal
        sample = input_sample(raw)
        keys = set(sample['keys']); pressed = keys - self.previous_keys
        dt = 1 / FPS
        actor = self.actor
        old = np.array(actor['position'], float)
        old_camera = deepcopy(self.scene['camera'])
        self.yaw -= sample['mouse'][0] * .0025
        self.pitch = float(np.clip(self.pitch + sample['mouse'][1] * .0025, -.25, 1.2))
        grounded = actor.get('support_state', 'grounded') == 'grounded'
        if grounded and ('c' in pressed or 'stand' in pressed):
            self.posture_target = ('standing' if 'stand' in pressed or self.posture_target == 'crouched' else 'crouched')
            self.posture_origin = deepcopy(weights(actor)); self.posture_elapsed = 0.
        if grounded and self.posture_elapsed < .3:
            prior = deepcopy(weights(actor))
            self.posture_elapsed = min(.3, self.posture_elapsed + dt)
            u = self.posture_elapsed / .3
            actor['posture_weights'] = blend(self.posture_origin, self.posture_target, u*u*(3-2*u))
            if blocked(actor['position'], actor, self.scene, self.scene['entities']):
                actor['posture_weights'] = prior
                self.posture_elapsed -= dt
                if self.last_blocker != 'posture_clearance':self.event('blocked_posture')
                self.last_blocker = 'posture_clearance'
        crouched = weights(actor).get('crouched', 0) > 1e-8
        actor['gait'] = 'crouch' if crouched else ('run' if 'shift' in keys else 'walk')
        forward = np.array([-math.sin(self.yaw), 0., -math.cos(self.yaw)])
        right = np.array([math.cos(self.yaw), 0., -math.sin(self.yaw)])
        direction = forward * (('w' in keys) - ('s' in keys)) + right * (('d' in keys) - ('a' in keys))
        norm = np.linalg.norm(direction)
        desired = direction / max(1., norm) * GAIT_SPEEDS[actor['gait']]
        velocity = np.array(actor.get('motion', {}).get('velocity', [0., 0., 0.]), float)
        change = desired - velocity
        next_velocity = velocity + change * min(1., 6 * dt / max(np.linalg.norm(change), 1e-9))
        delta = (velocity + next_velocity) * .5 * dt
        obstacle = None
        if grounded and 'space' in pressed and not crouched:
            actor['support_state'] = 'airborne'; actor['vertical_velocity'] = 4.4
            actor['support_contacts'] = []; grounded = False
            self.event('planned_takeoff', position=old.tolist())
        if grounded:
            obstacle = self.ground_move(actor, delta)
            if obstacle:
                # Tangential movement is allowed; normal movement remains blocked.
                for axis in (0, 2):
                    slide = np.zeros(3); slide[axis] = old[axis] + delta[axis] - actor['position'][axis]
                    self.ground_move(actor, slide)
                next_velocity = (np.array(actor['position']) - old) / dt
                next_velocity[1] = 0.
        else:
            vy = actor['vertical_velocity']
            delta[1] = vy * dt - .5 * GRAVITY * dt * dt
            last = old.copy()
            steps = max(2, math.ceil(np.linalg.norm(delta) / .01) + 1)
            landed = False
            for alpha in np.linspace(0, 1, steps)[1:]:
                p = old + alpha * delta
                floor = floor_height(self.scene, p[0], p[2], last[1])
                if vy - GRAVITY * dt * alpha <= 0 and floor is not None and p[1] <= floor <= last[1] + 1e-6:
                    p[1] = floor
                    candidate = dict(actor, support_state='grounded')
                    if not blocked(p, candidate, self.scene, self.scene['entities']):
                        last = p; landed = True; break
                hit = blocked(p, actor, self.scene, self.scene['entities'])
                if hit:
                    obstacle = hit
                    vertical = np.array([last[0],p[1],last[2]])
                    if blocked(vertical, actor, self.scene, self.scene['entities']):break
                    p = vertical
                last = p
            actor['position'] = last.tolist()
            actor['vertical_velocity'] = 0. if landed else vy - GRAVITY * dt
            if obstacle and vy > 0 and last[1] < old[1]+delta[1]-.005:
                actor['vertical_velocity'] = 0.
            if obstacle:
                next_velocity = (last-old)/dt; next_velocity[1] = 0.
            if landed:
                actor['support_state'] = 'grounded'
                self.event('planned_landing', position=last.tolist())
        if obstacle and obstacle != self.last_blocker:self.event('movement_blocked', by=obstacle)
        self.last_blocker = obstacle
        actor['motion'] = {'velocity': next_velocity.tolist(), 'phase': 'locomotion' if np.linalg.norm(next_velocity)>1e-6 else 'stationary'}
        if norm > 1e-8:
            target = math.atan2(direction[0], direction[2])
            turn = (target - actor.get('rotation_y',0) + math.pi) % (2*math.pi) - math.pi
            actor['rotation_y'] = actor.get('rotation_y',0) + float(np.clip(turn, -6*dt, 6*dt))
        actor['support_contacts'] = contacts(actor)
        index = {e['id']: e for e in self.scene['entities']}
        for entity in self.scene['entities']:
            if entity['kind'] != 'actor' or entity is actor:continue
            origin = np.array(entity['position'], float)
            proposed = npc_proposal(entity, index, dt)
            self.ground_move(entity, proposed - origin)
        held = next((e for e in self.scene['entities'] if e['kind']=='staff' and e.get('holder')==self.actor_id), None)
        if 'attack' in pressed:
            if held and self.attack_time is None:
                self.attack_time = 0.; self.hit_targets.clear(); self.event('planned_attack')
                start = actor.get('weapon_angle',55.)
                self.attack_angles = (start, -55. if start >= 0 else 55.)
            elif not held:self.event('unsupported_attack', reason='No held staff; firearms not implemented')
        phase = None; progress = 0.
        if self.attack_time is not None:
            self.attack_time = min(.65, self.attack_time + dt)
            progress = self.attack_time / .65
            phase = {'type':'sweep', 'start_angle_deg':self.attack_angles[0], 'end_angle_deg':self.attack_angles[1]}
        for entity in self.scene['entities']:
            if entity['kind'] != 'actor':continue
            item = next((e for e in self.scene['entities'] if e['kind']=='staff' and e.get('holder')==entity['id']), None)
            previous = deepcopy(item.get('weapon')) if item else None
            pose, weapon, errors = rig(entity, self.time+dt, 0., phase if entity is actor else None,
                progress, item, self.scene['terrain'], self.scene)
            if errors:raise ValueError('Control rig invalid: '+str(errors))
            entity['pose'] = pose
            if not item:continue
            item['weapon'] = weapon; item['position'] = ((np.array(weapon['a'])+weapon['b'])/2).tolist()
            if entity is actor and phase:
                for target in self.scene['entities']:
                    if target['kind']!='actor' or target is actor or target['id'] in self.hit_targets:continue
                    hit = False
                    for a in np.linspace(0,1,5):
                        start = np.array(weapon['a']) if previous is None else np.array(previous['a'])*(1-a)+np.array(weapon['a'])*a
                        end = np.array(weapon['b']) if previous is None else np.array(previous['b'])*(1-a)+np.array(weapon['b'])*a
                        for height in (.3,.55,.8):
                            point = np.array(target['position'])+[0,target['size'][1]*height,0]
                            hit |= point_segment_distance(point,start,end)<target['size'][0]/2+item['size'][0]/2
                    if hit:
                        self.event('proxy_contact', weapon=item['id'], target=target['id'], damage_inferred=False)
                        self.hit_targets.add(target['id'])
        if self.attack_time is not None and self.attack_time >= .65:
            actor['weapon_angle'] = self.attack_angles[1]; self.attack_time = None
        camera = self.scene['camera']
        target = np.array(actor['position']) + self.target_offset
        offset = self.distance*np.array([math.sin(self.yaw)*math.cos(self.pitch), math.sin(self.pitch), math.cos(self.yaw)*math.cos(self.pitch)])
        camera['target'] = target.tolist(); camera['position'] = (target+offset).tolist()
        camera['up'] = [0.,1.,0.]
        camera['motion'] = {'translation_velocity':((target-np.array(old_camera['target']))/dt).tolist(),
            'orbit_velocity_deg': -sample['mouse'][0]*.0025/dt*180/math.pi, 'distance_velocity':0.}
        self.time += dt; self.previous_keys = keys
        self.scene['control_state'] = {'held_keys': sorted(keys)}
        return {'time':self.time,'phase':0,'entities':deepcopy(self.scene['entities']),'camera':deepcopy(camera)}


def simulate_controls(scene, action, start_time=0., fps=FPS):
    if fps != FPS:raise ValueError('Control traces use exactly 24 Hz')
    samples = action.get('inputs')
    if not isinstance(samples,list) or not samples:raise ValueError('Nonempty control input trace required')
    controller = Controller(scene, action['actor_id'], start_time)
    frames = [controller.step(s) for s in samples]
    if controller.actor.get('support_state','grounded')!='grounded':
        raise ValueError('Control trace ends airborne; finish landing before exporting')
    if controller.attack_time is not None:
        raise ValueError('Finish the active staff sweep before exporting')
    if controller.posture_elapsed < .3:
        raise ValueError('Finish the posture transition before exporting')
    end = deepcopy(controller.scene)
    validate_scene(end)
    report = {'valid':True,'violations':[],'events':controller.events,
              'limitation':'Coarse occupancy, sampled contact, no damage or firearm simulation'}
    return {'fps':FPS,'frames':frames}, end, report
