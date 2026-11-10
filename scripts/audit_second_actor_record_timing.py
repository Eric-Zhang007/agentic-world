"""Independent entity identity and record-clock audit; no media trajectory writes."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import numpy as np
from memory.gta_release import Records


def main():
    out=ROOT/'outputs/geometry_source_audit_20261103'
    old=json.loads((out/'second_actor_audit.json').read_text())
    frame_path=Path(old['source_frames_jsonl'])
    def rows(name):
        return [json.loads(s) for s in (frame_path.parent/name).read_text().splitlines()]
    raw,aligned,frames=rows('source_frames.jsonl'),rows('source_alignment.jsonl'),rows('frames.jsonl')
    method=Path('E:/GTA_555_complete_rebuild_20261025/source/method/source/src')
    sys.path.insert(0,str(method))
    from wm_engine.processing.align import fit_affine
    from wm_engine.ingestor.finalize import _best_clock_pairs
    descriptor=Path('E:/GTA_555_complete_rebuild_20261025/source/data/09182/agents/agent_000/segment_00000.json')
    desc=json.loads(descriptor.read_text())
    pairs=_best_clock_pairs(desc['clock_sync'])
    alpha,beta=fit_affine(pairs)
    residuals=np.array([(y-(alpha*x+beta))/1e6 for x,y in pairs])
    fit_evidence={
        'descriptor':str(descriptor),'sync_points_total':len(desc['clock_sync']),
        'selected_points':len(pairs),'selection':'lowest RTT half, then one affine least-squares fit',
        'alpha':alpha,'beta_ns':beta,
        'matches_published_coefficients':alpha==aligned[480]['clock_alpha'] and beta==aligned[480]['clock_beta_ns'],
        'selected_residual_ms':{'min':float(residuals.min()),'max':float(residuals.max()),
                                'rmse':float(np.sqrt(np.mean(residuals**2)))},
        'selected_sync_rows':[p for p in desc['clock_sync'] if (p['sidecar_monotonic_ns'],p['server_monotonic_ns']) in pairs],
        'selection_source':str(method/'wm_engine/ingestor/finalize.py'),
        'fit_source':str(method/'wm_engine/processing/align.py'),
        'frame480_local_ns':raw[480]['sidecar_monotonic_ns'],
        'frame480_reproduced_server_ns':round(alpha*raw[480]['sidecar_monotonic_ns']+beta),
        'interpretation':'Published alignment is exactly reproduced but has large residual against its own selected clock observations. This proves poor fit consistency, not the correct RGB capture timestamp or sole visual-mismatch cause.',
    }
    receipt=json.loads(Path('E:/GTA_555_complete_rebuild_20261025/record_adapter/ep_20260918_132805_7503.json').read_text())
    rec=Records(Path(receipt['index']))
    entity=old['entity_id']
    streams={}
    try:
        for agent in ('agent_000','agent_001'):
            domain=rec.db.execute('select distinct domain from camera where agent=?',(agent,)).fetchone()[0]
            samples=[]
            for seq in range(350,551):
                c=rec.camera(agent,domain,seq)
                e=c['subject_observation'] if agent=='agent_001' else next(e for e in c['scene_observation']['entities'] if e.get('dataset_entity_id')==entity)
                samples.append({'seq':seq,'time_ns':c['server_time_estimate_ns'],
                                'client_monotonic_ms':c['client_monotonic_ms'],
                                'render_frame_id':c['render_frame_id'],'uncertainty_ns':c.get('uncertainty_ns'),
                                'dataset_entity_id':e['dataset_entity_id'],'network_id':e['network_id'],
                                'model_hash':e['model_hash'],'position':e['position'],
                                'client_has_control':e.get('client_has_control'),
                                'velocity':e.get('linear_velocity'),
                                'clock_calibration':c.get('client_clock_calibration')})
            streams[agent]={'domain':domain,'source':receipt['sources'][agent]['path'],'samples':samples}
        a,b=[streams[k]['samples'] for k in ('agent_000','agent_001')]
        ta=np.array([r['time_ns'] for r in a],dtype=np.float64)/1e9
        tb=np.array([r['time_ns'] for r in b],dtype=np.float64)/1e9
        pa=np.array([r['position'] for r in a]);pb=np.array([r['position'] for r in b])
        fits=[]
        # Compare positions on a shared server clock, on the same support for every lag.
        for start,end in [(141.,148.),(141.,144.),(144.,148.)]:
            valid=(ta>=max(start,tb[0]+.6))&(ta<=min(end,tb[-1]-.6))
            candidates=[]
            for shift in np.arange(-.6,.6001,.005):
                interpolated=np.column_stack([np.interp(ta[valid]+shift,tb,pb[:,axis]) for axis in range(3)])
                error=np.linalg.norm(pa[valid]-interpolated,axis=1)
                candidates.append({'self_time_minus_observer_time_ms':round(float(shift*1000),3),
                                   'rmse_world_units':float(np.sqrt(np.mean(error**2))),
                                   'median_world_units':float(np.median(error))})
            fits.append({'interval_server_seconds':[start,end],'samples':int(valid.sum()),
                         'best':min(candidates,key=lambda v:v['rmse_world_units']),
                         'zero_lag':min(candidates,key=lambda v:abs(v['self_time_minus_observer_time_ms']))})
        identities={agent:{'entity_ids':sorted({r['dataset_entity_id'] for r in s['samples']}),
                           'network_ids':sorted({r['network_id'] for r in s['samples']}),
                           'model_hashes':sorted({r['model_hash'] for r in s['samples']}),
                           'unchanged_position_steps':sum(x['position']==y['position'] for x,y in zip(s['samples'],s['samples'][1:])),
                           'nonincreasing_time_steps':sum(x['time_ns']>=y['time_ns'] for x,y in zip(s['samples'],s['samples'][1:]))}
                    for agent,s in streams.items()}
        clocks=[]
        for i in (360,420,450,470,480,490,510,540,599):
            clocks.append({'frame_index':i,'raw':raw[i],'aligned':aligned[i],
                           'proxy_server_time_ns':frames[i]['server_time_estimate_ns'],
                           'alignment_minus_raw_ms':(aligned[i]['server_time_estimate_ns']-raw[i]['server_time_estimate_ns'])/1e6})
        report={'entity_id':entity,'identity_checks':identities,'record_position_lag_fits':fits,
                'clock_fit_reproduction':fit_evidence,
                'lag_definition':'Compare observer_position(t) with self_position(t+shift). Positive shift means observer resembles later self state; not RGB-to-record delay.',
                'frame_clock_evidence':clocks,'alignment_provenance':json.loads((frame_path.parent/'alignment_provenance.json').read_text()),
                'all_frame_ids_preserved':all(r['frame_index']==i and s['frame_index']==i and f['proxy_frame_index']==i and f['rgb_decode_index']==i for i,(r,s,f) in enumerate(zip(raw,aligned,frames))),
                'frame_counts':[len(raw),len(aligned),len(frames)],
                'all_proxy_times_equal_final_alignment':all(f['server_time_estimate_ns']==s['server_time_estimate_ns'] for f,s in zip(frames,aligned)),
                'streams':streams,
                'limits':['Stable record identity is not independent identification of the RGB person.',
                          'Source timestamp is ffmpeg_showinfo_receipt, not capture callback; clock fit is not visual synchronization proof.',
                          'Observer versus self trajectory delay is distinct from RGB versus record delay; no media trajectory measured here.']}
        (out/'second_actor_record_timing.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('streams','frame_clock_evidence')},indent=2))
        print('CLOCK_DELTAS',[(x['frame_index'],x['alignment_minus_raw_ms']) for x in clocks])
    finally:rec.close()


if __name__=='__main__':main()
