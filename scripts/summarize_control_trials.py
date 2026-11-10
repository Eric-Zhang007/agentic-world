"""Summarize actual measured timings without confusing conditional error and recall."""
import argparse
import json
from pathlib import Path


def main(root):
    root=Path(root);results=[]
    for name,relative in (("legacy_filled","world_controls_20261008/render"),
                          ("legacy_filled_triton","world_controls_20261008/triton"),
                          ("legacy_filled_anchor","world_controls_20261008/anchor"),
                          ("outline_anchor_triton","world_controls_outline_20261008/render")):
        folder=root/relative
        if not (folder/"metadata.json").exists():continue
        metadata=json.loads((folder/"metadata.json").read_text())
        evaluations={r["label"]:r["summary"] for r in json.loads((folder/"evaluation.json").read_text())} if (folder/"evaluation.json").exists() else {}
        for row in metadata["runs"]:
            results.append({"group":name,"label":row["label"],"sample_seconds":row["sample_seconds"],
                "sample_decode_save_seconds":row["seconds"],"cache_skips":row.get("cache_skipped_evaluations"),
                "first_control_layer_calls":row["control_calls"].get("0",0),"steps":row["steps"],
                "metrics":evaluations.get(row["label"]),"path":str(folder/f"{row['label']}.mp4")})
    (root/"control_trials_summary.json").write_text(json.dumps(results,indent=2))
    print("group | case | sample_s | sample_decode_save_s | missing | center_error_px")
    for row in results:
        m=row["metrics"] or {}
        print(f"{row['group']} | {row['label']} | {row['sample_seconds']:.2f} | {row['sample_decode_save_seconds']:.2f} | {m.get('missing_ratio')} | {m.get('mean_projection_error')}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);main(p.parse_args().root)
