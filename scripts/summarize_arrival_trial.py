"""Make arrival endpoints and history checks inspectable without editing the world."""
import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.interaction import arrival_distance
from scripts.evaluate_motion_ablation import project
from scripts.summarize_video_reconcile import main as summarize_video


def main(root):
    summarize_video(root)
    contract = json.loads((root/'interaction_contract.json').read_text())
    initial = json.loads((root/'world_initial.json').read_text())
    target = np.asarray(contract['target'])
    ring = [target + [contract['radius']*np.cos(t), 0, contract['radius']*np.sin(t)]
            for t in np.linspace(0, 2*np.pi, 40)]
    pixels = project(ring, initial['camera'], 1024, 768)
    rows = []
    for segment in sorted(root.glob('segment_*')):
        if not (segment/'review/status.json').is_file():
            continue
        status = json.loads((segment/'review/status.json').read_text())
        review = json.loads((segment/'review/review.json').read_text())
        metadata = json.loads((segment/'render/metadata.json').read_text())
        with av.open(str(segment/'render'/(metadata['runs'][0]['label']+'.mp4'))) as video:
            first = last = None
            for frame in video.decode(video=0):
                last = frame.to_image()
                if first is None:
                    first = last
        panels = []
        for image in (first, last):
            image = image.resize((1024, 768))
            draw = ImageDraw.Draw(image)
            draw.line([tuple(p) for p in pixels], fill='yellow', width=3)
            panels.append(image)
        sheet = Image.new('RGB', (1536, 600), 'black')
        draw = ImageDraw.Draw(sheet)
        for index, image in enumerate(panels):
            sheet.paste(image.resize((768, 576)), (index*768, 24))
        draw.text((8, 5), 'Actual first frame | fixed target annotation, not an object', fill='white')
        draw.text((776, 5), 'Actual last frame | '+status['status'], fill='white')
        sheet.save(root/(segment.name+'_arrival_endpoints.jpg'))
        row = {'segment': segment.name, 'status': status['status'],
               'agent_action_complete': review.get('action_complete'),
               'visual_assessment': review.get('interaction_assessment'),
               'planned_distance': arrival_distance(json.loads((segment/'plan/candidate_world.json').read_text()), contract)}
        if status['status'] == 'COMMITTED':
            row['accepted_distance'] = arrival_distance(json.loads((segment/'review/world_after.json').read_text()), contract)
            prior = json.loads((segment/'review/history_before.json').read_text())
            after = json.loads((segment/'review/history_after.json').read_text())
            assert after[:-1] == prior
            row['history_prefix_unchanged'] = True
        rows.append(row)
    result = {'contract': contract, 'segments': rows,
              'chain': json.loads((root/'chain_verification.json').read_text()),
              'scope': 'Video-reviewed coarse arrival; not metric 3D measurement or contact validation'}
    if not any(row['status'] == 'COMMITTED' for row in rows):
        result['rejection_preserves_initial_scene'] = json.loads((root/'world_current.json').read_text()) == initial
        assert result['rejection_preserves_initial_scene']
        result['chain_scope_note'] = ('No new accepted endpoint in this trial. The legacy chain field '
            'current_world_matches_last_accepted_endpoint=false is not evidence of drift from the inherited parent.')
    (root/'arrival_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    main(p.parse_args().root.resolve())
