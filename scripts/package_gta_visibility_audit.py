"""Bundle visibility diagnostics, with explicit evidence scope."""
from pathlib import Path
import zipfile


def main():
    root = Path('E:/GTA_memory_prepared')
    project = Path(__file__).resolve().parents[1]
    output = root/'visibility_diagnostics_20261018.zip'
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in ('visibility_evidence_20261018_v2', 'near_camera_diagnostic_20261018'):
            for path in sorted((root/name).rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
        archive.write(project/'configs/gta_near_camera_review_20261018.json', 'manual_review.json')
        archive.write(project/'docs/gta_visibility_evidence.md', 'README.md')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive integrity failure')
    print(output, output.stat().st_size)


if __name__ == '__main__':
    main()
