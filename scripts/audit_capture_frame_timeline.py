"""Check actual MKV presentation order against descriptor and receipt clocks."""
import json
import subprocess
from pathlib import Path
import numpy as np


def main():
    root=Path('E:/GTA_555_complete_rebuild_20261025/source/data/09182')
    agent=root/'agents/agent_000'
    segment=root/'derived/segments/agent_000/segment_00000'
    video=agent/'video_00000.mkv'
    desc=json.loads((agent/'segment_00000.json').read_text())
    def read_rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
    frames=read_rows(segment/'frames.jsonl')
    source=read_rows(segment/'source_frames.jsonl')
    aligned=read_rows(segment/'source_alignment.jsonl')
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
                           '-show_entries','frame=best_effort_timestamp_time,pkt_dts_time,pict_type,coded_picture_number',
                           '-of','json',str(video)],capture_output=True,text=True,check=True)
    decoded=json.loads(result.stdout)['frames']
    pts=np.array([round(float(f['best_effort_timestamp_time'])*1e9) for f in decoded],dtype=np.int64)
    original_pts=np.array([r['video_pts_ns'] for r in source],dtype=np.int64)
    encoded=np.array([r['rgb_encoded_pts_ns'] for r in source],dtype=np.int64)
    local=np.array([r['sidecar_monotonic_ns'] for r in source],dtype=np.int64)
    final=np.array([r['server_time_estimate_ns'] for r in aligned],dtype=np.int64)
    old_root=Path('E:/GTA_555/09182')
    comparisons={}
    for relative in [Path('derived/segments/agent_000/segment_00000')/name for name in
                     ('frames.jsonl','source_frames.jsonl','source_alignment.jsonl','alignment_provenance.json','compiler_manifest.json')]+[
                         Path('agents/agent_000')/name for name in
                         ('segment_00000.json','clock_sync.jsonl','frame_timestamps.jsonl','frame_timestamps.aligned.jsonl')]:
        comparisons[str(relative)]={'original':str(old_root/relative),'rebuild':str(root/relative),
                                    'byte_equal':(old_root/relative).read_bytes()==(root/relative).read_bytes()}
    # Relative receipt latency is observable; absolute capture-to-receipt latency is not.
    region=slice(360,601)
    receipt_offset=(local[region]-local[360])-(original_pts[region]-original_pts[360])
    mkv_offset=(local[region]-local[360])-(pts[region]-pts[360])
    def slope(x,y):
        x=(x-x[0]).astype(float)/1e9;y=(y-y[0]).astype(float)/1e9
        return float(np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean()))
    sync=desc['clock_sync']
    near=[p for p in sync if local[360]-3e9<=p['sidecar_monotonic_ns']<=local[600]+3e9]
    motion_path=Path(__file__).resolve().parents[1]/'outputs/dataset_reaudit/actor_sequence/motion_timing.json'
    motion=json.loads(motion_path.read_text())
    selected=[]
    for i in range(420,601,12):
        selected.append({'file_decode_index':i,'actual_mkv_pts_ns':int(pts[i]),
                         'descriptor_frame_index':desc['frames'][i]['frame_index'],
                         'original_showinfo_pts_ns':int(original_pts[i]),
                         'sidecar_receipt_ns':int(local[i]),'raw_server_time_ns':source[i]['server_time_estimate_ns'],
                         'aligned_server_time_ns':int(final[i]),
                         'receipt_elapsed_minus_pts_elapsed_ms':float(((local[i]-local[420])-(pts[i]-pts[420]))/1e6),
                         'aligned_elapsed_minus_pts_elapsed_ms':float(((final[i]-final[420])-(pts[i]-pts[420]))/1e6),
                         'nearest_client_sample_seq':frames[i]['camera_interpolation']['nearest_client_sample_seq']})
    all_receipt=[]
    with (agent/'frame_timestamps.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if r.get('published_segment')=='segment_00000':all_receipt.append(r)
            elif len(all_receipt)>=len(source):break
    report={'video':str(video),'descriptor':str(agent/'segment_00000.json'),
            'original_vs_rebuild':comparisons,
            'full_segment_clock_mapping':{
                'server_vs_original_rgb_pts_slope':slope(original_pts,final),
                'published_sidecar_to_server_alpha':aligned[0]['clock_alpha'],
                'all_frames_same_alpha':all(r['clock_alpha']==aligned[0]['clock_alpha'] for r in aligned),
                'key_frames':[{'frame':i,'rgb_pts_ns':int(original_pts[i]),'server_time_ns':int(final[i]),
                               'server_delta_from_frame0_ns':int(final[i]-final[0]),
                               'raw_server_time_ns':source[i].get('server_time_estimate_ns')}
                              for i in (0,360,480,600,1320,1439)],
                'attribution':'The same scaled mapping already exists byte-for-byte in the original E:/GTA_555 release and rebuild. It was not introduced by prepare_geometry_trial.py or by copying this release into rebuild. The original capture estimates also used an evolving clock fit; they are not independently verified true capture time.',
            },
            'receipt_diagnostic_360_600':{
                'receipt_minus_showinfo_pts_relative_ms_range':[float(receipt_offset.min()/1e6),float(receipt_offset.max()/1e6)],
                'receipt_minus_showinfo_pts_relative_end_ms':float(receipt_offset[-1]/1e6),
                'receipt_minus_mkv_pts_relative_ms_range':[float(mkv_offset.min()/1e6),float(mkv_offset.max()/1e6)],
                'receipt_vs_showinfo_pts_slope':slope(original_pts[region],local[region]),
                'receipt_vs_mkv_pts_slope':slope(pts[region],local[region]),
                'final_server_vs_mkv_pts_slope':slope(pts[region],final[region]),
                'receipt_elapsed_s':float((local[600]-local[360])/1e9),
                'mkv_elapsed_s':float((pts[600]-pts[360])/1e9),
                'final_server_elapsed_s':float((final[600]-final[360])/1e9),
                'nearby_sync_rows':near,
                'interpretation':'Relative receipt lag versus showinfo PTS measures queue variation, not absolute capture latency. Compare measured slopes before attributing the affine time compression to stderr backlog.',
            },
            'external_motion_evidence':{'path':str(motion_path),'fits':motion['fits'],'ownership':'Read-only main-thread result; not recomputed here'},
            'timestamp_implementation':{
                'receipt':'capture/sidecar.py: stderr line is read, SHOWINFO regex matches, then perf_counter_ns is sampled; scale precedes showinfo in ffmpeg filter chain.',
                'sync_endpoint':'ingestor/app.py: server estimate = latest received world server timestamp + elapsed since its ingestor receipt; upstream transport age of that world snapshot is not added.',
                'sync_storage':'processing/clock.py: sidecar HTTP midpoint paired with endpoint estimate; stored uncertainty_ns is null for these sync points.',
                'fit':'ingestor/finalize.py and processing/align.py select low-RTT half and fit one affine mapping; low endpoint RTT does not measure upstream world ingestion delay.',
                'limits':'Endpoint design can carry variable world-ingestion delay into sync points. Actual per-batch transport delay is not present in the supplied clock_sync rows, so that mechanism is not quantitatively proven for this episode.'},
            'actual_decoded_frame_count':len(decoded),'descriptor_frame_count':len(desc['frames']),
            'actual_pts_equal_recorded_encoded_pts':bool(np.array_equal(pts,encoded)),
            'max_actual_pts_minus_encoded_pts_ns':int(np.max(np.abs(pts-encoded))),
            'file_indices_equal_frame_and_decode_indices':all(r['frame_index']==i and r['rgb_decode_index']==i and frames[i]['rgb_segment_frame_index']==i for i,r in enumerate(source)),
            'source_frames_equal_descriptor_clock_rows':all(all(r[k]==desc['frames'][i][k] for k in ('frame_index','video_pts_ns','sidecar_monotonic_ns')) for i,r in enumerate(source)),
            'capture_timestamp_rows_for_segment':len(all_receipt),
            'capture_receipts_match_source':len(all_receipt)==len(source) and all(r['sidecar_monotonic_ns']==s['sidecar_monotonic_ns'] and r['video_pts_ns']==s['video_pts_ns'] for r,s in zip(all_receipt,source)),
            'showinfo_pts_minus_mkv_pts_ms_range':[float((original_pts-pts).min()/1e6),float((original_pts-pts).max()/1e6)],
            'selected_frame_timeline':selected,
            'interval_420_600':{'mkv_elapsed_s':float((pts[600]-pts[420])/1e9),
                                    'receipt_elapsed_s':float((local[600]-local[420])/1e9),
                                    'aligned_elapsed_s':float((final[600]-final[420])/1e9)},
            'conclusion':'File presentation index equals descriptor/source frame index and recorded encoded PTS are independently checked. Receipt-time/affine-clock mapping is distinct from actual capture time; changing relative elapsed time can produce phase drift. No fixed lag or corrected pairing is asserted.'}
    out=Path(__file__).resolve().parents[1]/'outputs/geometry_source_audit_20261103/capture_frame_timeline.json'
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
