import argparse
from pathlib import Path
from memory.gta_scene_memory import index_segment

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    print(index_segment(args.manifest,args.output))
