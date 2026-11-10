"""Exact temporal bookkeeping for native historical-prefix generation."""
from pathlib import Path
import json
import numpy as np


def window(total_frames=124,prefix_frames=22,fps=24):
    if total_frames%17!=5 or prefix_frames<5 or prefix_frames%17!=5 or prefix_frames>=total_frames:
        raise ValueError('Native total and prefix must both be 17*k+5; prefix must leave future frames')
    return {'total_frames':total_frames,'prefix_frames':prefix_frames,'trim_start':prefix_frames-1,
            'new_ticks':total_frames-prefix_frames,'new_seconds':(total_frames-prefix_frames)/fps,
            'continuation_frames':total_frames-prefix_frames+1,'fps':fps}


def read_video(path):
    import av
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        decoded=[(float(f.time),f.to_ndarray(format='rgb24')) for f in container.decode(stream)]
    times=np.asarray([x[0] for x in decoded])
    if not len(times) or not np.allclose(times-times[0],np.arange(len(times))/24,atol=1e-4,rtol=0):
        raise ValueError('Expected native contiguous 24 fps video')
    return np.stack([x[1] for x in decoded])


def save_video(rgb,path):
    from scripts.run_comfy_control import save_video as encode
    encode(rgb.astype(np.float32)/255,Path(path))


def historical_prefix(db,output,count=22):
    """Read accepted tail and matching corrected states; never edit the archive."""
    output.mkdir(parents=True,exist_ok=False)
    history=db.get_meta('whitebox_history',[])
    record=history[-1]
    timeline=json.loads(Path(record['artifacts']['corrected_timeline']).read_text())
    if timeline['fps']!=24 or abs(timeline['frames'][-1]['time']-db.get_meta('tick'))>1e-6:
        raise ValueError('Historical prefix must end at the current world clock at 24 fps')
    from whitebox.world import scene_from_db
    current=scene_from_db(db)
    tail=timeline['frames'][-count:]
    if len(tail)!=count or tail[-1]['entities']!=current['entities'] or tail[-1]['camera']!=current['camera']:
        raise ValueError('Historical tail does not match current accepted state')
    pixels=read_video(record['artifacts']['video'])
    if len(pixels) not in (len(timeline['frames']),len(timeline['frames'])+1):
        raise ValueError('Historical video and corrected timeline disagree')
    prefix=output/'prefix.mp4';save_video(pixels[-count:],prefix)
    scene={**current,'entities':tail[0]['entities'],'camera':tail[0]['camera']}
    for name,value in [('scene.json',scene),('timeline.json',{'fps':24,'frames':tail[1:]}),
                       ('source.json',{'video':record['artifacts']['video'],'world_times':[f['time'] for f in tail]})]:
        (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    return prefix


def trim_generated(video,output,schedule):
    pixels=read_video(video)
    if len(pixels)!=schedule['total_frames']:raise ValueError('Generated length differs from declared window')
    save_video(pixels[schedule['trim_start']:],output)
    Path(output).with_suffix('.alignment.json').write_text(json.dumps({**schedule,'raw_video':str(video),
        'historical_overlap_advances_world':False},indent=2))
    return Path(output)
