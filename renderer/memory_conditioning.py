"""Explicit H3 training-style image roles shared by prompt and tensor bindings."""
import re


def image_bindings(first_frame, history_pair=None):
    if history_pair is None:
        return {'ref_image_1': first_frame}
    return {'ref_image_1': history_pair, 'ref_image_2': first_frame}


def memory_prompt(action, duration, has_history):
    if re.search(r'<(?:Picture|Video|Object)\s+\d+>', action):
        raise ValueError('Action text must not declare media tags; bindings are tool-owned')
    first = '<Picture 2>' if has_history else '<Picture 1>'
    history = ('<Object 1> is the historical appearance-to-geometry correspondence in '
               '<Picture 1>: one horizontal pair, historical RGB on the left and its '
               'corresponding whitebox view on the right. These are the same scene, '
               'camera and timestamp, not two scenes.\n') if has_history else ''
    return (
        'object_definitions:\n'+history+
        f'{first} is the exact first RGB frame at 0.00 seconds and defines current state.\n'
        f'<Video 1> is the target whitebox camera and motion reference over {duration:.3f} seconds.\n\n'
        'summary:\n'
        f'Generate a continuous {duration:.3f}-second video of this same world.\n\n'
        'attention_analysis:\n'+
        ('<Object 1>: preserve historical scene appearance when those same surfaces reappear. '
         'Do not restore an obsolete object position, pose or ownership.\n' if has_history else '')+
        f'{first}: preserve the starting composition, identities, materials and actual current state.\n'
        '<Video 1>: follow camera trajectory, relative placement, world-facing directions, '
        'movement timing and occlusion. Do not copy proxy colors or mannequin geometry.\n\n'
        'detailed_description:\n'+action.strip()+'\n'
        'Start and end states must match the supplied conditions; enrich intermediate motion '
        'naturally. Recovery means stabilizing at the target state, not returning to the start. '
        'For moving endpoints, preserve motion through the segment boundary without forced braking.\n\n'
        'overall_soundscape:\nNatural scene ambience.\n\nnon_diegetic_music:\nNone.'
    )
