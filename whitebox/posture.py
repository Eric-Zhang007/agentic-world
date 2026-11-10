"""Small persistent pose vocabulary; weights describe transitions, not fitted joints."""
import math
import json
from pathlib import Path

POSTURES=('standing','crouched','low_guard')
HEIGHT_FACTORS=json.loads((Path(__file__).resolve().parents[1]/'assets/gta/catalog.json').read_text())['height_factors']


def body_size(actor):
    size=list(actor['size'])
    size[1]*=sum(HEIGHT_FACTORS[name]*amount for name,amount in weights(actor).items())
    return size


def weights(actor):
    return actor.get('posture_weights',{'standing':1.,'crouched':0.,'low_guard':0.})


def blend(origin,target,amount):
    if target not in POSTURES:raise ValueError('Unknown coarse posture')
    return {name:(1-amount)*origin.get(name,0.)+amount*(name==target) for name in POSTURES}


def contacts(actor):
    if actor.get('support_state')=='airborne':return []
    return ['left_foot','right_foot']+(['left_hand','right_hand'] if weights(actor).get('low_guard',0.)>1-1e-8 else [])


def validate(actor):
    w=weights(actor)
    if (not isinstance(w,dict) or set(w)-set(POSTURES) or
        any(type(x) not in (float,int) or not math.isfinite(x) or not 0<=x<=1 for x in w.values()) or
        not math.isclose(sum(w.values()),1.,abs_tol=1e-8)):
        raise ValueError('Posture weights must be finite nonnegative and sum to one')
    if actor.get('grip','both') not in {'both','left','right'}:raise ValueError('Unknown held-staff grip')
    if 'support_contacts' in actor and actor['support_contacts']!=contacts(actor):
        raise ValueError('Support contacts must agree with coarse posture')


def anchors(name):
    if name=='crouched':
        return {'pelvis':(0,.30,-.05),'chest':(0,.49,.07),'head':(0,.68,.10),
                'shoulder':(.13,.50,.07),'hip':(.09,.30,-.05),'foot':(.15,.044,0),
                'hand':(.20,.32,.22),'weapon_center':(0,.40,.24)}
    if name=='low_guard':
        return {'pelvis':(0,.30,-.10),'chest':(0,.27,.22),'head':(0,.36,.39),
                'shoulder':(.13,.28,.22),'hip':(.09,.30,-.10),'foot':(.16,.044,-.24),
                'hand':(.20,.044,.43),'weapon_center':(0,.25,.30)}
    return {'pelvis':(0,.49,0),'chest':(0,.74,0),'head':(0,.93,0),
            'shoulder':(.13,.75,0),'hip':(.09,.49,0),'foot':(.12,.044,0),
            'hand':(.18,.54,.13),'weapon_center':(0,.62,.20)}
