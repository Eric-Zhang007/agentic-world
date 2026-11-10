"""Export the delivered scene pairs and recorded camera tracks for local review."""
import json
import io
from pathlib import Path
from collections import defaultdict
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('E:/GTA_555_complete_rebuild_20261025')
OUT = Path('C:/Users/zjc/Desktop/GTA相机轨迹与帧对')


def compact(p):
    return dict(id=p['point_id'], episode=p['episode_id'], agent=p['agent_id'],
                clock=p['server_clock_domain'], client=p.get('client_clock_domain', ''),
                t=p['time_ns']/1e9, xyz=p['camera_position'], forward=p['forward'])


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'pairs').mkdir(exist_ok=True)
    (OUT/'examples').mkdir(exist_ok=True)
    points = {}
    for source in ('available_source_training', 'additional_local_training'):
        for line in (BASE/source/'trajectory_points.jsonl').read_bytes().splitlines():
            p = json.loads(line)
            points[p['point_id']] = compact(p)
    samples = []
    chosen_images = {}
    with zipfile.ZipFile(BASE/'unified_delivery/GTA_memory_training_unified.zip') as z:
        names = z.namelist()
        manifests = [n for n in names if n.endswith(tuple('manifests/'+s+'.jsonl' for s in ('train','validation','test')))]
        rows = [json.loads(line) for n in manifests for line in z.read(n).splitlines() if line]
        scenes = [r for r in rows if r['task']=='scene_background']
        assert len(scenes) == 2101
        root = next(n[:-len('sample.json')] for n in names if n.endswith('/'+scenes[0]['sample_id']+'/sample.json'))
        prefix = root.split('samples/')[0]
        for i, row in enumerate(scenes):
            sid = row['sample_id']
            folder = prefix+'samples/'+sid+'/'
            meta = json.loads(z.read(folder+'sample.json'))
            ref = compact(json.loads(z.read(folder+'reference_state.json')))
            target = compact(json.loads(z.read(folder+'target_state.json')))
            points.setdefault(ref['id'], ref)
            points.setdefault(target['id'], target)
            assert ref['episode']==target['episode'] and ref['clock']==target['clock'] and ref['t']<target['t']
            delta = np.asarray(target['xyz'])-ref['xyz']
            pair = dict(id=sid, episode=row['episode_id'], kind=row['kind'], split=row['split'],
                        ref=ref, target=target, age=target['t']-ref['t'],
                        distance=float(np.linalg.norm(delta)), xy=float(np.linalg.norm(delta[:2])),
                        dz=float(delta[2]), angle=meta['reference']['view_angle_deg'],
                        image='pairs/'+sid+'.jpg')
            sheet = Image.new('RGB', (960, 600), 'white')
            draw = ImageDraw.Draw(sheet)
            for j, name in enumerate(('reference_rgb.png','reference_whitebox.png','target_rgb.png','target_whitebox.png')):
                im = Image.open(io.BytesIO(z.read(folder+name))).convert('RGB')
                im.thumbnail((480,270))
                x,y=(j%2)*480,(j//2)*300
                sheet.paste(im,(x+(480-im.width)//2,y+25+(270-im.height)//2))
                draw.text((x+8,y+6),name.replace('.png',''),fill='black')
            sheet.save(OUT/pair['image'],quality=85)
            samples.append(pair)
            if i%300==0: print('paired frames',i,flush=True)
        # Select reproducible examples across relation types, times and viewpoints.
        selected=[]
        for kind in ('same_camera_return','cross_camera_passage','nearby_history'):
            pool=[s for s in samples if s['kind']==kind]
            for key in ('age','angle','distance'):
                p=max(pool,key=lambda s:s[key])
                if p not in selected:selected.append(p)
            p=sorted(pool,key=lambda s:s['age'])[len(pool)//2]
            if p not in selected:selected.append(p)
        for s in selected:
            dest=OUT/'examples'/s['id']
            dest.mkdir(exist_ok=True)
            folder=prefix+'samples/'+s['id']+'/'
            for name in ('reference_rgb.png','reference_whitebox.png','target_rgb.png','target_whitebox.png',
                         'reference_state.json','target_state.json','sample.json'):
                (dest/name).write_bytes(z.read(folder+name))
    tracks=defaultdict(list)
    for p in points.values():
        tracks[(p['episode'],p['clock'],p['agent'],p['client'])].append(p)
    chunks=[]
    for key, group in sorted(tracks.items()):
        group.sort(key=lambda p:(p['t'],p['id']))
        part=[]
        for p in group:
            if part and (p['t']-part[-1]['t']>7.5 or p['t']<=part[-1]['t']):
                chunks.append(dict(episode=key[0],clock=key[1],agent=key[2],points=part));part=[]
            part.append(p)
        if part:chunks.append(dict(episode=key[0],clock=key[1],agent=key[2],points=part))
    for index,s in enumerate(selected):
        fig,ax=plt.subplots(figsize=(8,7),dpi=140)
        rp,tp=np.array(s['ref']['xyz']),np.array(s['target']['xyz'])
        radius=max(30,float(np.linalg.norm(tp[:2]-rp[:2]))*1.5)
        center=(rp[:2]+tp[:2])/2
        for tr in chunks:
            if tr['episode']!=s['episode'] or tr['clock']!=s['ref']['clock']:continue
            pts=np.array([p['xyz'][:2] for p in tr['points']])
            ax.plot(pts[:,0],pts[:,1],color='#aab6bd',lw=.7,alpha=.6)
            active=[p for p in tr['points'] if s['ref']['t']<=p['t']<=s['target']['t'] and p['agent'] in (s['ref']['agent'],s['target']['agent'])]
            if active:
                xy=np.array([p['xyz'][:2] for p in active]);ax.plot(xy[:,0],xy[:,1],color='#536e93',lw=1.5)
        ax.plot([rp[0],tp[0]],[rp[1],tp[1]],'--',color='#a2448c',lw=2,label='Reference-to-target link (not motion)')
        for p,color,label in ((s['ref'],'#16876c','R: historical reference'),(s['target'],'#d85832','T: target start')):
            x,y=p['xyz'][:2];fx,fy=p['forward'][:2]
            ax.scatter(x,y,c=color,s=65,label=label,zorder=5)
            ax.arrow(x,y,fx*radius*.2,fy*radius*.2,color=color,width=radius*.006,length_includes_head=True,zorder=4)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),xlabel='World X',ylabel='World Y')
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set_title(f"{s['kind']} | dt={s['age']:.1f}s | XY={s['xy']:.1f} | dz={s['dz']:.1f}\nview angle={s['angle']:.1f} deg | {s['ref']['agent']} -> {s['target']['agent']}",fontsize=10)
        fig.tight_layout(); path=OUT/'examples'/s['id']/'trajectory.png';fig.savefig(path);plt.close(fig)
        plot=Image.open(path).convert('RGB');pair=Image.open(OUT/s['image']).convert('RGB')
        plot.thumbnail((800,700));pair.thumbnail((960,700))
        board=Image.new('RGB',(1760,750),'white');board.paste(plot,(0,30));board.paste(pair,(800,60))
        ImageDraw.Draw(board).text((10,10),s['id'],fill='black')
        board.save(OUT/'examples'/f'{index+1:02d}_trajectory_and_pair.jpg',quality=92)
    data=dict(samples=samples,tracks=chunks,selected=[s['id'] for s in selected])
    (OUT/'data.json').write_text(json.dumps(data),encoding='utf-8')
    template=Path(__file__).with_name('camera_pairs_viewer.html').read_text(encoding='utf-8')
    template=template.replace('Math.floor(Math.log10(view.r/3))','Math.ceil(Math.log10(view.r/3))')
    angle_script=Path(__file__).with_name('camera_pairs_angle_filter.js').read_text(encoding='utf-8')
    (OUT/'index.html').write_text(template.replace('__DATA__',json.dumps(data)).replace('</html>','<script>'+angle_script+'</script></html>'),encoding='utf-8')
    report=dict(scene_pairs=len(samples),trajectory_points=len(points),trajectory_chunks=len(chunks),examples=len(selected),
                source='Final uploaded unified archive; background scene records only',
                gap_rule='Separate episode, server/client clock and camera; break gaps above 7.5 sec',
                coordinates='Source world XY, equal scale; arrows use horizontal component of recorded forward vector')
    (OUT/'REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report,flush=True)


if __name__=='__main__':main()
