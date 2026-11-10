"""Extract bounded calibration evidence without altering the source archive."""
import argparse
from pathlib import Path
import shutil
import zipfile


def main(archive, output):
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename
            selected = name.endswith(("clock_sync.jsonl", "frame_timestamps.jsonl",
                                      "frame_timestamps.aligned.jsonl", "client.jsonl"))
            if name.endswith(".mkv"):
                selected = int(Path(name).stem.split("_")[-1]) < 10
            if not selected or info.is_dir():
                continue
            target = (output / name).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            with z.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(name, info.file_size)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.archive, a.output)
