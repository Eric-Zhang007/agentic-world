"""Check that camera calibration changes no action/entity and leaves DB intact."""
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np


def main(root):
    root=root.resolve()
    project=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"manifest.json").read_text())
    candidate=Path(manifest["attempt"])
    if not candidate.is_absolute():candidate=project/candidate
    calibration=json.loads((candidate/"calibration.json").read_text())
    source=Path(calibration["source_attempt"])
    if not source.is_absolute():source=project/source
    old=json.loads((source/"scene.json").read_text());new=json.loads((candidate/"scene.json").read_text())
    old.pop("camera");new.pop("camera");assert old==new,"Calibration changed authoritative scene content"
    old_frames=json.loads((source/"timeline.json").read_text())["frames"]
    new_frames=json.loads((candidate/"timeline.json").read_text())["frames"]
    for frame in old_frames+new_frames:frame.pop("camera",None)
    assert old_frames==new_frames,"Calibration changed the executed action"
    assert json.loads((source/"action.json").read_text())==json.loads((candidate/"action.json").read_text())
    with np.load(root/"control.npz") as data:
        rgb=data["rgb"]
        assert rgb.shape==(73,768,1024,3) and rgb.dtype==np.uint8
        assert np.array_equal(rgb[...,0],rgb[...,1]) and np.array_equal(rgb[...,0],rgb[...,2])
        assert float(rgb.std())>10 and np.any(rgb[0]!=rgb[-1])
    parent=source.parent
    with sqlite3.connect((parent/"world.sqlite").as_uri()+"?mode=ro",uri=True) as db:
        meta={k:json.loads(v) for k,v in db.execute("SELECT key,value FROM world_meta")}
        entities={}
        for identity,attributes,position,rotation,provenance in db.execute(
            "SELECT object_id,attributes,position,rotation,provenance FROM objects"
        ):
            x,y,z=json.loads(position)
            entities[identity]={**json.loads(attributes)["whitebox"],"id":identity,
                                "position":[x,z,-y],"rotation_y":json.loads(rotation)[2],
                                "provenance":provenance}
    snapshot=json.loads((parent/"world_current.json").read_text())
    assert entities=={e["id"]:e for e in snapshot["entities"]},"DB entities differ from committed snapshot"
    assert meta["whitebox_header"]=={k:v for k,v in snapshot.items() if k!="entities"}
    assert meta["whitebox_revision"]==2 and abs(meta["tick"]-5.625)<1e-8
    result={"scene_unchanged_except_experimental_camera":True,"action_and_timeline_unchanged":True,
            "depth_rgb_shape":list(rgb.shape),"depth_std":float(rgb.std()),
            "database_matches_committed_snapshot":True,"world_revision":meta["whitebox_revision"],
            "world_time":meta["tick"],"history_entries":len(meta["whitebox_history"]),
            "entity_count":len(entities)}
    (root/"input_verification.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);main(p.parse_args().root)
