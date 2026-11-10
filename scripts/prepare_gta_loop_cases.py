"""Prepare RGB-only candidate sheets; keep paired geometry separate for evaluation."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('E:/GTA_555_complete_rebuild_20261025')
OUT = ROOT/'outputs/gta_pi_loop_20261102'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = []
    episodes = set()
    for source in ['available_source_training', 'additional_local_training']:
        for sample in sorted((BASE/source/'training/samples').glob('*/sample.json')):
            if sample.parent.name.split('_agent_')[0] in episodes:
                continue
            record = json.loads(sample.read_text(encoding='utf-8'))
            episode = record.get('episode_id')
            if not episode:
                continue
            if episode in episodes:
                continue
            if not (sample.parent/'target_rgb.png').exists():
                continue
            episodes.add(episode)
            selected.append({'index': len(selected), 'sample': str(sample), 'episode': episode,
                             'rgb': str(sample.parent/'target_rgb.png'),
                             'whitebox': str(sample.parent/'target_whitebox.png'),
                             'state': str(sample.parent/'target_state.json')})
    for start in range(0, len(selected), 9):
        sheet = Image.new('RGB', (1440, 900), '#202020')
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(selected[start:start+9]):
            image = Image.open(row['rgb']).convert('RGB')
            image.thumbnail((480, 270))
            x, y = n%3*480, n//3*300
            sheet.paste(image, (x,y))
            draw.text((x+4,y+273), f"{row['index']:02d} {row['episode']}", fill='white')
        sheet.save(OUT/f'candidates_{start//9:02d}.jpg', quality=94)
    (OUT/'candidates.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'candidate_count':len(selected)}))


if __name__ == '__main__':
    main()
