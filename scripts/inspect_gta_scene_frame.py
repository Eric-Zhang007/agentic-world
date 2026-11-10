import argparse
from pathlib import Path
import orjson
from memory.gta_scene_memory import inspect_frame


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--frame',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=inspect_frame(args.manifest,args.frame)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(orjson.dumps(result,option=orjson.OPT_INDENT_2))
    print({k:v for k,v in result.items() if k not in ('placements','renderer_entity_ids')})


if __name__=='__main__':main()
