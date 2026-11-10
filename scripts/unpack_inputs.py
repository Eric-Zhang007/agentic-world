"""Extract source/data only; never restore bundled environments or local credentials."""
import argparse
from pathlib import Path
import zipfile


def unpack(archive, destination):
    destination = Path(destination).resolve()
    skipped = {".venv", "node_modules", "__MACOSX", "__pycache__", ".git", ".pytest_cache"}
    count = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path = Path(info.filename)
            if info.is_dir() or any(p in skipped for p in path.parts):
                continue
            if path.name.startswith(".env") or "settings.local" in path.name or path.name == ".DS_Store":
                continue
            target = (destination / path).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(info.filename)
            z.extract(info, destination)
            count += 1
    print(f"Extracted {count} files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("destination")
    args = parser.parse_args()
    unpack(args.archive, args.destination)
