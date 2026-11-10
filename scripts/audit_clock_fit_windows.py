"""Reproduce final clock fit and compare time subsets without changing data."""
import sys,json
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
BASE=Path('E:/GTA_555_complete_rebuild_20261025/source')
sys.path.insert(0,str(BASE/'method/source/src'))
from wm_engine.processing.align import fit_affine
from wm_engine.ingestor.finalize import _best_clock_pairs


def main():
    agent=BASE/'data/09182/agents/agent_000'
    desc=json.loads((agent/'segment_00000.json').read_text())
    sync=desc['clock_sync'];epoch=sync[0]['sidecar_monotonic_ns']
    target=desc['frames'][480]['sidecar_monotonic_ns']
    def fit(name,points,select=True):
        pairs=_best_clock_pairs(points) if select else [(p['sidecar_monotonic_ns'],p['server_monotonic_ns']) for p in points]
        if len(pairs)<2:return {'name':name,'insufficient_points':len(pairs)}
        a,b=fit_affine(pairs)
        residual=np.array([(y-a*x-b)/1e6 for x,y in pairs])
        return {'name':name,'input_count':len(points),'selected_count':len(pairs),'alpha':a,'beta_ns':b,
                'selected_sync_indices':[i for i,p in enumerate(sync) if (p['sidecar_monotonic_ns'],p['server_monotonic_ns']) in pairs],
                'local_span_seconds':(max(x for x,y in pairs)-min(x for x,y in pairs))/1e9,
                'residual_rmse_ms':float(np.sqrt(np.mean(residual**2))),
                'frame480_mapped_server_ns':round(a*target+b),
                'frame480_is_extrapolated':not min(x for x,y in pairs)<=target<=max(x for x,y in pairs)}
    fits=[fit('published_descriptor_low_RTT_half',sync),fit('descriptor_all_points',sync,False),
          fit('first_10_points_before_late_discontinuity',sync[:10]),
          fit('first_15_points_before_large_gap',sync[:15]),
          fit('after_large_gap',sync[16:]),
          fit('without_last_selected_point',sync[:24]+sync[25:])]
    for seconds in (5,10,15,20):
        fits.append(fit(f'frame480_plus_minus_{seconds}s',[p for p in sync if abs(p['sidecar_monotonic_ns']-target)<=seconds*1e9]))
    full=[json.loads(s) for s in (agent/'clock_sync.jsonl').read_text().splitlines()]
    matching=[p for p in full if p.get('sidecar_clock_domain')==sync[0]['sidecar_clock_domain'] and p.get('server_clock_domain')==sync[0]['server_clock_domain']]
    fits.append(fit('whole_episode_same_domains',matching))
    changes=[]
    for i,(a,b) in enumerate(zip(sync,sync[1:]),1):
        dx=(b['sidecar_monotonic_ns']-a['sidecar_monotonic_ns'])/1e9
        dy=(b['server_monotonic_ns']-a['server_monotonic_ns'])/1e9
        changes.append({'index':i,'sidecar_elapsed_s':dx,'endpoint_server_elapsed_s':dy,
                        'endpoint_offset_change_ms':(dy-dx)*1000,'rtt_ms':b['round_trip_ms']})
    result={'descriptor':str(agent/'segment_00000.json'),'sync_source':str(agent/'clock_sync.jsonl'),
            'descriptor_sync_count':len(sync),'episode_same_domain_sync_count':len(matching),
            'descriptor_sync_time_span_s':(sync[-1]['sidecar_monotonic_ns']-epoch)/1e9,
            'fits':fits,'successive_sync_changes':changes,
            'causal_reading':'Finalization refits the entire descriptor cohort, not the online first-two-point model. A time-varying endpoint offset is represented as clock rate by a single affine model. Window fits are diagnostics, not repaired timestamps; no trusted absolute capture-time anchor is supplied.'}
    path=Path(__file__).resolve().parents[1]/'outputs/geometry_source_audit_20261103/clock_fit_windows.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
