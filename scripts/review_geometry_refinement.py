"""Show the original source mismatch and geometry refinements without hiding holes."""
from pathlib import Path
import json
from PIL import Image, ImageDraw


def main():
    project = Path(__file__).resolve().parents[1]
    root = project/'outputs/geometry_init/urban'
    panels = [
        ('Actual target RGB', root/'input/target_rgb.png'),
        ('Original dataset proxy: NOT aligned visual GT', root/'evaluation/target_whitebox.png'),
        ('Recorded camera + collision geometry re-render', project/'outputs/geometry_source_audit_20261103/target_region_recordcamera.png'),
        ('Before: whole-object boxes, observed ground only', root/'multiview_assets/camera.png'),
        ('After: facade patches; cyan = inferred ground', root/'refined_assets/camera.png'),
        ('3 -> 4 frames: local static-region consistency', root/'local_review_three_to_four/regional_review.png'),
    ]
    board = Image.new('RGB', (1920, 800), '#202326'); draw = ImageDraw.Draw(board)
    for i, (label, path) in enumerate(panels):
        im = Image.open(path).convert('RGB'); im.thumbnail((640, 360))
        x, y = i%3*640, i//3*400
        board.paste(im, (x+(640-im.width)//2, y+(360-im.height)//2))
        draw.text((x+8, y+369), label, fill='white')
    board.save(root/'refinement_comparison.jpg', quality=95)
    report = json.loads((root/'refined_assets/report.json').read_text())
    data = {'geometry': {'entity_count': report['entity_count'],
                         'facade_patches': sum(r['parts'] for r in report['facade_regions']),
                         'deduplicated_masks': len(report['mask_deduplication']['suppressed']),
                         'inferred_ground_pixels': report['ground_surface']['inferred_pixels']},
            'old_to_new_region_counts': json.loads((root/'local_review_three_to_four/report.json').read_text())['counts'],
            'unresolved': ['Partial facades do not reconstruct entire buildings',
                           'Actor heading and NPC support not fully established',
                           'Source proxy uses collision geometry, not verified visual ground truth'],
            'world_modified': False}
    (root/'refinement_summary.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(root/'refinement_comparison.jpg')


if __name__ == '__main__':
    main()
