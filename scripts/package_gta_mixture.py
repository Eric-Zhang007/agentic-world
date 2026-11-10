from pathlib import Path
import shutil
import zipfile
import orjson


def main():
    root=Path('D:/GTA_memory_mixture_20261019')
    delivery=Path('E:/GTA_scene_pairs_20261019')
    code=root/'processing_code';code.mkdir(exist_ok=True)
    for name in ('scripts/build_gta_mixture.py','scripts/prepare_gta_scene_training.py',
                 'scripts/finalize_gta_scene_review.py','scripts/test_gta_mixed_dataset.py',
                 'configs/gta_scene_review_20261019.json'):
        target=code/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    archive=delivery/'mixed_training_manifests.zip'
    if archive.exists():raise ValueError('Existing delivery is not overwritten')
    with zipfile.ZipFile(archive.with_suffix('.partial'),'w',zipfile.ZIP_DEFLATED) as z:
        for path in root.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive.with_suffix('.partial')) as z:
        if z.testzip():raise ValueError('Invalid archive')
    archive.with_suffix('.partial').rename(archive)
    for name in ('distribution_report.json','loader_verification.json'):
        shutil.copy2(root/name,delivery/name)
    (delivery/'DELIVERY.json').write_bytes(orjson.dumps({
        'new_scene_archive':'scene_pairs.zip','mixed_training_manifest_archive':archive.name,
        'existing_object_archive_directory':'E:/GTA_memory_delivery_20261019',
        'old_samples_preserved':4984,'scene_training_eligible':38,'scene_diagnostics_only':7,
        'paid_api_calls_this_work':0,'training_run_performed':False,
        'important_limit':'Small pilot background pool; no background validation samples in inherited split.',
        'local_gallery':'D:/GTA_scene_pairs_complete_20261019/dataset/index.html'
    },option=orjson.OPT_INDENT_2))
    print(str(archive),archive.stat().st_size)


if __name__=='__main__':main()
