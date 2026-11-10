"""Replace only exports with incorrect container timing, from original frames."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import os

from scripts.finalize_gta_release import video_shape


def repair_segment(args):
    release,segment=args
    windows={w['sample_id']:w for w in map(json.loads,(segment/'windows.jsonl').read_text().splitlines())}
    repaired=[]
    for sample_id,window in windows.items():
        root=release/'dataset/samples'/sample_id
        if not (root/'sample.json').exists():continue
        sample=json.loads((root/'sample.json').read_text())
        bad=[]
        for index,name in enumerate(('target_rgb.mp4','target_whitebox.mp4')):
            result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                'stream=duration,avg_frame_rate','-of','json',str(root/name)],capture_output=True,text=True,check=True)
            stream=json.loads(result.stdout)['streams'][0]
            a,b=map(int,stream['avg_frame_rate'].split('/'))
            if abs(float(stream['duration'])-120/sample['fps'])>.01 or abs(a/b-sample['fps'])>.01:bad.append((index,name))
        if not bad:continue
        manifest=json.loads(Path(window['source_manifest']).read_text())
        for index,name in bad:
            temporary=root/(name+'.repair.mp4')
            start=window['target_rgb_start'] if index==0 else window['target_start_frame']
            kind='rgb_video' if index==0 else 'proxy_video'
            fps=sample['fps']
            subprocess.run([os.environ.get('FFMPEG_BINARY','ffmpeg'),'-v','error','-y',
                '-hwaccel','cuda','-threads','1','-i',str(Path(window['episode_root'])/manifest[kind]),
                '-an','-vf',f'trim=start_frame={start}:end_frame={start+120},settb=expr=1/{fps},setpts=N',
                '-r',str(fps),'-vsync','1','-c:v','h264_nvenc','-preset','slow','-rc','vbr','-cq','18','-b:v','0',
                '-bf','0',str(temporary)],check=True,timeout=300)
            if video_shape(temporary)[0]!=120:raise ValueError('Repair has wrong frame count')
            temporary.replace(root/name)
        sample['media_timeline']='constant frame rate by paired decoder index; original capture times retained in frame_state_index'
        (root/'sample.json').write_text(json.dumps(sample,indent=2),encoding='utf-8')
        repaired.append({'sample_id':sample_id,'files':[name for _,name in bad]})
    print('TIMING_CHECK',segment.name,'repaired',len(repaired),flush=True)
    return repaired


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    args=parser.parse_args()
    if not (args.release/'export_complete.json').exists():raise ValueError('Finish export before timing repair')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(repair_segment,[(args.release,p.parent) for p in (args.release/'qualification').glob('*/windows.jsonl')]))
    log=args.release/'timing_repairs.json'
    previous=json.loads(log.read_text()) if log.exists() else []
    by_sample={row['sample_id']:row for row in previous}
    by_sample.update({row['sample_id']:row for rows in results for row in rows})
    log.write_text(json.dumps(list(by_sample.values()),indent=2),encoding='utf-8')


if __name__=='__main__':main()
