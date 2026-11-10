"""Coarse action semantics; no skeletal animation or inferred contacts."""
import math
import numpy as np
from whitebox.surfaces import floor_height

GRAVITY=9.81
GAIT_SPEEDS={'walk':2.2,'run':4.,'crouch':1.5}


def jump_geometry(scene,origin,delta,duration):
    target=np.asarray(origin,float)+np.asarray(delta,float)
    floor=floor_height(scene,target[0],target[2],origin[1])
    if floor is None:raise ValueError('Jump landing has no reachable support surface')
    target[1]=floor
    launch=(target[1]-origin[1])/duration+GRAVITY*duration/2
    if not 0<launch<GRAVITY*duration:
        raise ValueError('Jump must ascend then descend onto its landing surface')
    return target,float(launch)


def jump_height(origin_y,launch,time):
    return origin_y+launch*time-.5*GRAVITY*time*time


def check_gait(phase,posture):
    gait=phase.get('gait')
    if gait is None:return
    if gait not in GAIT_SPEEDS:raise ValueError('Unknown gait')
    if 'end_velocity' not in phase:raise ValueError('Semantic gait requires explicit end_velocity')
    if gait=='crouch':
        if not math.isclose(posture.get('crouched',0),1.):raise ValueError('Crouch movement requires completed crouched posture')
    elif not math.isclose(posture.get('standing',0),1.):
        raise ValueError('Walk/run requires standing posture')
