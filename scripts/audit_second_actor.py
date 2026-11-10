"""Read-only source investigation of the second actor; write audit artifacts only."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import numpy as np
import cv2
from scripts.audit_geometry_trial import read_json, read_image, save_image, bbox
from memory.gta_release import Records


def main():
    out = ROOT / 'outputs/geometry_source_audit_20261103'
    audit = read_json(out / 'audit.json')
    target = audit['pairs']['target']
    row = target['frame']
    at = row['server_time_estimate_ns']
    method = Path('E:/GTA_555_complete_rebuild_20261025/source/method/source/src')
    sys.path.insert(0, str(method))
    from wm_engine.proxy.geometry import Camera, Rasterizer, box_vertices, quaternion_matrix, euler_matrix
    from wm_engine.proxy.humanoid import humanoid_parts
    from wm_engine.processing.client_samples import _interpolate_camera
    records = Records(Path(audit['record_receipt']['index']))
    entity_id = 274783417663530
    try:
        brackets = [records.camera('agent_000', row['client_clock_domain'], seq) for seq in (451, 452)]
        cam = Camera(_interpolate_camera(brackets[0]['camera'], brackets[1]['camera'],
                                        row['camera_interpolation']['alpha']), 1280, 720)
        def project(entity):
            rotation = quaternion_matrix(entity['orientation_quaternion']) if entity.get('orientation_quaternion') else euler_matrix(entity['rotation'])
            envelope = box_vertices(entity['position'], [-.325,-.20,-.95], [.325,.20,.85], rotation)
            parts = humanoid_parts(envelope)
            vertices = np.concatenate([v for _, v in parts])
            pixels = cam.project(cam.transform(vertices))
            root = cam.project(cam.transform([entity['position']]))[0]
            return {'root_pixel_xy': root.tolist(),
                    'rigid_projected_bbox': np.r_[pixels.min(0),pixels.max(0)].tolist()}, parts
        def observation(c, subject=False):
            e = c['subject_observation'] if subject else next(e for e in c['scene_observation']['entities'] if e.get('dataset_entity_id') == entity_id)
            if e['dataset_entity_id'] != entity_id:
                raise ValueError('Other-agent identity mismatch')
            return {'client_sample_seq': c['client_sample_seq'], 'agent_id': c['agent_id'],
                    'client_clock_domain': c['client_clock_domain'], 'server_clock_domain': c['server_clock_domain'],
                    'server_time_estimate_ns': c['server_time_estimate_ns'],
                    'delta_from_rgb_ns': c['server_time_estimate_ns'] - at,
                    'uncertainty_ns': c.get('uncertainty_ns'), 'render_frame_id': c['render_frame_id'],
                    'entity': e, 'projection_in_agent000_camera': project(e)[0]}
        observer = [observation(c) for c in brackets]
        domain = records.db.execute('select distinct domain from camera where agent=?', ('agent_001',)).fetchall()
        if len(domain) != 1:
            raise ValueError('Ambiguous other-agent clock domain')
        domain = domain[0][0]
        # Binary search the indexed sequence by recorded server time, then retain both brackets.
        lo, hi = records.db.execute('select min(seq),max(seq) from camera where agent=? and domain=?', ('agent_001',domain)).fetchone()
        while lo < hi:
            mid = (lo + hi) // 2
            if records.camera('agent_001', domain, mid)['server_time_estimate_ns'] < at:
                lo = mid + 1
            else:
                hi = mid
        own = [records.camera('agent_001',domain,seq) for seq in (lo-1,lo)]
        if not own[0]['server_time_estimate_ns'] <= at <= own[1]['server_time_estimate_ns']:
            raise ValueError('Other-agent time not bracketed')
        self_observations = [observation(c, True) for c in own]
        world = records.world(row['server_clock_domain'], row['world_capture_tick'])
        server_entity = next(e for e in world['agents'] if e['dataset_entity_id'] == entity_id)
        sample_dir = Path(audit['sample_json']).parent
        rgb = read_image(sample_dir / 'target_rgb.png')
        proxy = read_image(sample_dir / 'target_whitebox.png')
        b,g,r = proxy.astype(float).transpose(2,0,1)
        red = (r>70)&(r>1.8*(g+1))&(r>1.8*(b+1))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype('uint8'))
        index = 1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        red = labels==index
        perception = ROOT / 'outputs/geometry_init/urban/perception_target'
        people = [o for o in read_json(perception/'result.json')['observations'] if o['category']=='person']
        second = next(o for o in people if o['observation_id']=='smoke_001')
        rgb_bbox = second['bbox_xyxy']
        raster = Rasterizer(cam)
        for _, vertices in project(observer[1]['entity'])[1]:
            raster.box(vertices,'other_agent',human_facing_cues=True)
        rb,rg,rr = raster.image.astype(float).transpose(2,0,1)
        render_mask = (rr>70)&(rr>1.8*(rg+1))&(rr>1.8*(rb+1))
        # Reproduce the foreground subject's occlusion before comparing visible red pixels.
        for _, vertices in project(brackets[1]['subject_observation'])[1]:
            raster.box(vertices,'agent',human_facing_cues=True)
        ob,og,orr = raster.image.astype(float).transpose(2,0,1)
        occluded_mask = (orr>70)&(orr>1.8*(og+1))&(orr>1.8*(ob+1))
        history=[]
        for seq in range(442,462):
            c=records.camera('agent_000',row['client_clock_domain'],seq)
            history.append(observation(c))
        report = {
            'frame_index':480,'rgb_server_time_ns':at,'entity_id':entity_id,'agent_id':'agent_001',
            'source_frames_jsonl':audit['frames_jsonl'],'record_sources':audit['record_receipt']['sources'],
            'proxy_red_bbox_xyxy_exclusive':bbox(red),'rgb_second_person_bbox':rgb_bbox,
            'rgb_detection_source':str(perception/'result.json'),'rgb_detection_id':'smoke_001',
            'rgb_identity_note':'Second visible person selected from existing detection; model bbox, not manual GT identity annotation.',
            'proxy_minus_rgb_bbox_center_xy':((np.array(bbox(red)[:2])+bbox(red)[2:]-np.array(rgb_bbox[:2])-rgb_bbox[2:])/2).tolist(),
            'observer_brackets':observer,'other_agent_self_brackets':self_observations,
            'server_world':{'capture_tick':row['world_capture_tick'],'time_fields':{k:v for k,v in world.items() if 'time' in k or 'clock' in k},'entity':server_entity,'projection':project(server_entity)[0]},
            'observer_record_render_bbox':bbox(render_mask),
            'red_vs_record_render_mask_iou':float(np.count_nonzero(red&render_mask)/np.count_nonzero(red|render_mask)),
            'record_render_with_foreground_occlusion_bbox':bbox(occluded_mask),
            'red_vs_occlusion_aware_render_iou':float(np.count_nonzero(red&occluded_mask)/np.count_nonzero(red|occluded_mask)),
            'observer_history':history,
            'temporal_diagnostic': {
                'sample_443': {'delta_from_rgb_ms':-360.567783,'root_pixel_x':784.18},
                'sample_444': {'delta_from_rgb_ms':-320.567783,'root_pixel_x':763.28},
                'interpretation':'Earlier recorded roots project around the current RGB second-person bbox center. This is a single-frame temporal-offset hypothesis, not proof: root and animated silhouette center are different measurements, and no cross-frame delay has been fitted or applied.',
            },
            'source_policy':'compiler.dynamic_boxes selects identity-matched observer scene entity for other_agent; camera is interpolated, nonvehicle actor pose is nearest observation.',
            'conclusion': {
                'confirmed':'Second visible RGB person and red proxy have a large spatial mismatch. The observer-record projection reproduces the red proxy horizontal location; the prior foreground-subject conclusion does not apply.',
                'source_difference':'Observer agent_000 remote entity, agent_001 self-observation, and server state differ. Both client samples are fresh records by their timestamps; client_has_control=false alone does not prove stale underlying network state.',
                'not_sufficient':'Replacing the observer remote position by the near-time self observation or server state still projects near x=633 or x=637, far from RGB second-person bbox center x=772.77. The measured cross-source difference alone does not explain the full mismatch.',
                'unresolved':'Root cause among RGB-to-record temporal correspondence, rendered remote pose versus recorded entity transform, and other capture/identity effects is not established by these records. No pose, timestamp, or image correction applied.',
            },
        }
        canvas=np.full((760,2560,3),255,dtype='uint8')
        for i,(im,box,title) in enumerate([(rgb,rgb_bbox,'RGB second person: existing detector'),(proxy,bbox(red),'Original proxy: red other_agent')]):
            marked=im.copy()
            x1,y1,x2,y2=np.rint(box).astype(int)
            cv2.rectangle(marked,(x1,y1),(x2,y2),(0,255,255),2)
            canvas[40:,i*1280:(i+1)*1280]=marked
            cv2.putText(canvas,title,(i*1280+10,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),1)
        save_image(out/'second_actor_audit.jpg',canvas)
        save_image(out/'second_actor_record_render.png',raster.image)
        (out/'second_actor_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        bounds=audit['delivery_impact']['source_camera_vs_visual_truth']
        bounds['actor_difference']='FOREGROUND BLUE SUBJECT ONLY: bbox center agreement of 1.99 px and fixed-pose difference. This does not apply to the second red other_agent; see second_actor_audit.json.'
        bounds['verified']='Frame 480 PNG source equality verified. Foreground BLUE SUBJECT ONLY: record render reproduces proxy bbox within one pixel and mask IoU 0.9807914834529045; not a conclusion about the red other_agent.'
        audit['second_actor_audit']=str(out/'second_actor_audit.json')
        (out/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('observer_history','record_sources')},indent=2))
    finally:
        records.close()


if __name__=='__main__':
    main()
