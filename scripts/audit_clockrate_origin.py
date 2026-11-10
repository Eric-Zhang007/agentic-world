"""Compare published clock rate to independent client calibration and video evidence."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import numpy as np
from memory.gta_release import Records


def fit(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    x=x-x[0];y=y-y[0]
    a=float(np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean()))
    residual=y-(a*x+(y.mean()-a*x.mean()))
    return {'slope':a,'residual_rmse_ms':float(np.sqrt(np.mean(residual**2))*1000),
            'duration_x_seconds':float(x[-1]),'duration_y_seconds':float(y[-1])}


def main():
    out=ROOT/'outputs/geometry_source_audit_20261103'
    receipt=json.loads(Path('E:/GTA_555_complete_rebuild_20261025/record_adapter/ep_20260918_132805_7503.json').read_text())
    records=Records(Path(receipt['index']))
    report={'record_index':receipt['index'],'clients':{}}
    try:
        for agent in ('agent_000','agent_001'):
            domain=records.db.execute('select distinct domain from camera where agent=?',(agent,)).fetchone()[0]
            samples=[];calibrations={}
            for seq in range(0,1801,10):
                c=records.camera(agent,domain,seq)
                t=c['server_time_estimate_ns']
                if not 125e9<=t<=190e9:continue
                samples.append({'seq':seq,'client_monotonic_ms':c['client_monotonic_ms'],'server_time_ns':t})
                cal=c.get('client_clock_calibration')
                if cal and cal.get('midpoint_client_ms') is not None:
                    calibrations[cal['midpoint_client_ms']]=cal
            cal=sorted(calibrations.values(),key=lambda r:r['midpoint_client_ms'])
            report['clients'][agent]={'source':receipt['sources'][agent]['path'],'domain':domain,
                'record_time_fit':fit([r['client_monotonic_ms']/1000 for r in samples],[r['server_time_ns']/1e9 for r in samples]),
                'direct_calibration_fit':fit([r['midpoint_client_ms']/1000 for r in cal],[r['server_ns']/1e9 for r in cal]),
                'unique_calibrations':cal,'record_samples':samples}
    finally:records.close()
    timeline=json.loads((out/'capture_frame_timeline.json').read_text())
    alpha=timeline['full_segment_clock_mapping']['published_sidecar_to_server_alpha']
    motion_path=ROOT/'outputs/dataset_reaudit/actor_sequence/motion_timing.json'
    motion=json.loads(motion_path.read_text())['affine_timing']
    report['video_validation']={'source':str(motion_path),**motion}
    report['clock_rate_comparison']={'published_alpha':alpha,'inverse_published_alpha':1/alpha,
        'independent_video_scale':motion['scale'],
        'relative_scale_difference_percent':100*(motion['scale']/(1/alpha)-1),
        'fit_scope':'Video affine fit and direct client calibration are independent evidence types; offsets and residual actor errors remain separate.'}
    report['implementation_sources']={
        'sync_endpoint':'E:/GTA_555_complete_rebuild_20261025/source/method/source/src/wm_engine/ingestor/app.py:145',
        'clock_storage':'E:/GTA_555_complete_rebuild_20261025/source/method/source/src/wm_engine/ingestor/store.py:91',
        'final_fit':'E:/GTA_555_complete_rebuild_20261025/source/method/source/src/wm_engine/ingestor/finalize.py:46',
        'affine_math':'E:/GTA_555_complete_rebuild_20261025/source/method/source/src/wm_engine/processing/align.py:10'}
    report['causal_boundary']='Direct client calibration tests server/client clock rate independently of the ingestor endpoint. Endpoint extrapolation omits upstream snapshot age; lack of original ingestor per-batch receipt times prevents attributing the exact bias to network versus batching versus scheduling. No data repaired.'
    (out/'clockrate_origin.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'clients':{k:{q:v for q,v in r.items() if q not in ('unique_calibrations','record_samples')} for k,r in report['clients'].items()},'comparison':report['clock_rate_comparison']},indent=2))


if __name__=='__main__':main()
