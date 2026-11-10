"""Use a validated recording prefix without extrapolating through gaps or lifecycles."""
import json
from pathlib import Path

import numpy as np

from compiler.calibration import calibrated_camera
from state.models import WorldObject


def bracket(times, timestamp, max_gap=.25):
    if timestamp<times[0] or timestamp>times[-1]:
        raise ValueError("Requested timestamp outside recorded support")
    j=min(max(1,int(np.searchsorted(times,timestamp))),len(times)-1)
    if times[j]-times[j-1]>max_gap:
        raise ValueError("Telemetry gap exceeds interpolation tolerance")
    return j-1,j,float((timestamp-times[j-1])/(times[j]-times[j-1]))


class EpisodeAlignment:
    def __init__(self, evidence_path, calibration_path, agent):
        evidence=json.loads(Path(evidence_path).read_text())
        self.calibration=json.loads(Path(calibration_path).read_text())[agent]
        if self.calibration["status"]!="validated":
            raise ValueError(f"{agent}: calibration is {self.calibration['status']}")
        self.agent=agent;self.stream=evidence["agents"][agent];self.world=evidence["world"]
        self.wt=np.array([r["server_monotonic_ns"]/1e9 for r in self.world])
        self.ct=np.array([r["time"] for r in self.stream["cameras"]])
        self.angles=np.degrees(np.unwrap(np.radians([
            r["camera"]["camera_rotation"] for r in self.stream["cameras"]]),axis=0))
        mapping=self.stream["frame_mapping"]
        self.fps=mapping["fps"];self.segments=mapping["segments"]
        self.origin=self.stream["video_server_origin_sec"]
        self.offset=self.calibration["parameters"]["video_to_telemetry_offset_sec"]

    def frame_time(self, segment, local_frame):
        prefix=0
        for item in self.segments:
            if item["name"]==segment:
                if not isinstance(local_frame,int) or not 0<=local_frame<item["frames"]:
                    raise ValueError("Frame outside verified segment")
                return self.origin+(prefix+local_frame)/self.fps+self.offset
            prefix+=item["frames"]
        raise ValueError("Segment not in verified decoded prefix")

    def at_frame(self, segment, local_frame, width=1280, height=720):
        time=self.frame_time(segment,local_frame)
        i,j,a=bracket(self.ct,time)
        c0,c1=(self.stream["cameras"][k]["camera"] for k in (i,j))
        if c0["render_resolution"]!=c1["render_resolution"]:
            raise ValueError("Camera viewport changed across interpolation")
        record=dict(c0)
        record["camera_position"]=((1-a)*np.array(c0["camera_position"])+a*np.array(c1["camera_position"])).tolist()
        record["camera_rotation"]=((1-a)*self.angles[i]+a*self.angles[j]).tolist()
        record["camera_fov"]=(1-a)*c0["camera_fov"]+a*c1["camera_fov"]
        camera=calibrated_camera(record,width,height,self.calibration,time)
        i,j,a=bracket(self.wt,time)
        first,second=self.world[i],self.world[j]
        objects=[]
        for kind in ("agents","vehicles","peds","objects"):
            right={e["dataset_entity_id"]:e for e in second[kind]}
            for entity in first[kind]:
                other=right.get(entity["dataset_entity_id"])
                if other is None:
                    continue  # Do not interpolate across despawn/ID changes.
                category="person" if kind in ("agents","peds") else "vehicle" if kind=="vehicles" else "unknown"
                pos=((1-a)*np.array(entity["position"])+a*np.array(other["position"])).tolist()
                attributes={"dataset_entity_id":str(entity["dataset_entity_id"]),
                            "original_agent_id":entity.get("agent_id"),"world_completeness":"partial",
                            "interpolation_source_sequences":[first["sample_seq"],second["sample_seq"]]}
                # This is a matching anchor, never a change to physical world position.
                if category=="person":
                    attributes["projection_anchor_offset_xyz"]=[0,0,self.calibration["parameters"]["entity_center_z_offset"]]
                    attributes["projection_anchor_provenance"]="INFERRED_bbox_center_calibration"
                objects.append(WorldObject(
                    object_id=f"{first['episode_id']}:{entity['dataset_entity_id']}",
                    category=category,position=pos,attributes=attributes,
                    state={"timestamp":time,"matchable":entity.get("agent_id")!=self.agent},
                    provenance="OBSERVED"))
        return objects,camera,{"server_time_sec":time,"source_segment":segment,
                              "local_frame":local_frame,"world_completeness":"partial",
                              "calibration_id":self.calibration["calibration_id"]}
