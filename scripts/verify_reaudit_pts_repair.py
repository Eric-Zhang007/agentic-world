"""Audit-only lossless CFR candidate; require exact decoded-pixel preservation."""
import argparse
from fractions import Fraction
import io
import json
from pathlib import Path
import zipfile

import av
import numpy as np


def candidate(data, path, fps):
    with av.open(io.BytesIO(data)) as source, av.open(str(path),'w') as output:
        original=source.streams.video[0]
        stream=output.add_stream('libx264',rate=fps)
        stream.width=original.width
        stream.height=original.height
        stream.pix_fmt=original.codec_context.format.name
        stream.options={'crf':'0','preset':'fast','bf':'0'}
        clock=Fraction(1,fps)
        stream.time_base=clock
        count=0
        for frame in source.decode(original):
            frame.pts=count
            frame.time_base=clock
            for packet in stream.encode(frame):
                output.mux(packet)
            count+=1
        for packet in stream.encode():
            output.mux(packet)
    return count


def compare(data,path,expected,fps):
    times=[]
    count=0
    with av.open(io.BytesIO(data)) as source, av.open(str(path)) as repaired:
        a=iter(source.decode(video=0))
        b=iter(repaired.decode(video=0))
        while True:
            fa,fb=next(a,None),next(b,None)
            if fa is None or fb is None:
                if fa is not None or fb is not None:
                    raise ValueError('Frame count changed')
                break
            if not np.array_equal(fa.to_ndarray(format='rgb24'),fb.to_ndarray(format='rgb24')):
                raise ValueError(f'Decoded pixels changed at frame {count}')
            times.append(float(fb.pts*fb.time_base))
            count+=1
    if count!=expected or not np.allclose(times,np.arange(count)/fps,rtol=0,atol=1e-6):
        raise ValueError('Repaired count or cadence does not match declared CFR')
    return {'decoded_frames':count,'all_decoded_pixels_identical':True,
            'cfr_timestamps_verified':True,'timestamps':times}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,default=Path('E:/GTA_555_complete_rebuild_20261025/unified_delivery/GTA_memory_training_unified.zip'))
    p.add_argument('--sample',default='ep_20260921_124218_450d_agent_000_segment_00000_w003')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/dataset_reaudit/pts_candidate')
    args=p.parse_args()
    out=args.output/args.sample
    out.mkdir(parents=True,exist_ok=True)
    report={'sample_id':args.sample,'source_archive':str(args.archive),'original_archive_modified':False,
            'method':'Decode presentation order and losslessly encode at declared CFR; B-frame packet order is not used',
            'scope':'Container cadence repair only; source world/RGB synchronization unchanged','media':{}}
    with zipfile.ZipFile(args.archive) as bundle:
        root=f'samples/{args.sample}'
        sample=json.loads(bundle.read(root+'/sample.json'))
        for name in ('target_rgb','target_whitebox'):
            data=bundle.read(root+'/'+name+'.mp4')
            path=out/(name+'.mp4')
            if path.exists():
                raise FileExistsError(path)
            packets=candidate(data,path,sample['fps'])
            report['media'][name]={'encoded_frames':packets,'path':str(path),
                **compare(data,path,sample['frame_count'],sample['fps'])}
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{n:v for n,v in r.items() if n!='timestamps'} for k,r in report['media'].items()},indent=2))


if __name__=='__main__':
    main()
