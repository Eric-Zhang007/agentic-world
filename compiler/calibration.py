"""Explicit GTA camera convention and quality-gated calibrated projection."""
import numpy as np


def gta_basis(euler_degrees):
    x,y,z=np.radians(euler_degrees)
    cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z)
    rot=(np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]) @
         np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]) @
         np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]))
    return np.array([rot[:,0],-rot[:,2],rot[:,1]])


def camera_matrix(record, width, height, parameters, fov_axis="vertical"):
    angle=np.asarray(record["camera_rotation"],float).copy()
    angle[0]+=parameters.get("pitch_bias_deg",0)
    angle[2]+=parameters.get("yaw_bias_deg",0)
    basis=gta_basis(angle)
    rw,rh=record["render_resolution"]
    focal=(rh if fov_axis=="vertical" else rw)/(2*np.tan(np.radians(record["camera_fov"])/2))
    fx=focal*width/rw*parameters.get("focal_scale",1)
    fy=focal*height/rh*parameters.get("focal_scale",1)
    k=np.array([[fx,0,width/2],[0,fy,height/2],[0,0,1]])
    transform=np.eye(4);transform[:3,:3]=basis
    transform[:3,3]=-basis@np.asarray(record["camera_position"])
    return k,transform


def calibrated_camera(record, width, height, calibration, timestamp):
    if calibration.get("status")!="validated":
        raise ValueError("Projection calibration has not passed held-out validation")
    start,end=calibration["valid_server_time_range"]
    if not start<=timestamp<=end:
        raise ValueError("Projection requested outside validated calibration interval")
    k,t=camera_matrix(record,width,height,calibration["parameters"],calibration["fov_axis"])
    return {"width":width,"height":height,"intrinsics":k.tolist(),"world_to_camera":t.tolist(),
            "calibration_status":"validated","calibration_id":calibration["calibration_id"],
            "timestamp":timestamp}
