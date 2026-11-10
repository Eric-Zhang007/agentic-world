"""Prepare the existing GTA garage proxy as an explicit out-of-domain probe."""
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.data / 'render_color/model.mp4'
    with av.open(str(source)) as container:
        fps = float(container.streams.video[0].average_rate)
        frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    assert len(frames)/fps >= 124/24, 'Need at least 124 target frames'
    with av.open(str(args.output/'condition.mp4'), 'w') as writer:
        stream = writer.add_stream('libx264', rate=24)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = 'yuv444p'
        stream.options = {'crf': '0'}
        for index in range(124):
            pixels = frames[min(int(index*fps/24), len(frames)-1)]
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format='rgb24')):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    original = np.asarray(Image.open(args.data/'frames/000.jpg').convert('RGB'))
    Image.fromarray(original).save(args.output/'reference_original.png')
    height, width = original.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    mask[int(.79*height):, :int(.19*width)] = 255
    reference = cv2.inpaint(original, mask, 3, cv2.INPAINT_TELEA)
    Image.fromarray(reference).save(args.output/'reference.png')
    Image.fromarray(mask).save(args.output/'reference_invalid_mask.png')
    prompt = '''subject_definitions:
<Picture 1> shows the appearance reference for a GTA-style third-person scene inside a cramped vehicle repair workshop. Preserve the two human identities, clothing, concrete walls, workbench, tool cabinets and indoor lighting.
<Video 1> is a coarse animated 3D proxy of the same scene. Red shaded geometry denotes the player, blue shaded geometry denotes the other human, and gray geometry denotes the floor, walls, workbench and props. These colors identify objects and are not target appearance colors. This proxy is approximate; use it for the player movement, body orientation and following-camera motion. Both colored figures are articulated adult HUMANS walking on two legs. No colored region denotes a vehicle.
<Subject 1> is the man wearing a red-and-black horizontally striped short-sleeve shirt, dark trousers and an olive scarf, as shown in <Picture 1>. Preserve his human face, hair and clothing throughout.
<Subject 2> is the man in a blue shirt with yellow sleeves, dark trousers and sneakers standing by the workbench, as shown in <Picture 1>.

scene_description:
A small realistic GTA repair workshop with stained concrete block walls, a cluttered workbench, red tool drawers, hanging tools, bottles and a cylinder. The setting remains indoors throughout. Retain the reference image's worn materials and lighting.

video_description:
<Subject 1> takes several natural walking steps near <Subject 2> and the workbench and turns his body as indicated by the red proxy. The third-person camera follows and rotates with the player as indicated by <Video 1>. <Subject 2> stays near the workbench with subtle idle movement. Preserve two human bodies and their distinct clothes, plausible foot placement, scene layout and occlusion. Enrich the coarse motions naturally. Start and end states should agree with the given conditions; settling at the target end state does not mean returning to the initial position. There are no racing cars, roads or race tracks in this indoor scene. No HUD, minimap, subtitles or added overlays.

overall_soundscape:
Quiet indoor workshop ambience and human footsteps. No racing engines or dialogue.

non_diegetic_music:
N/A'''
    assets = json.loads(args.assets.read_text())
    assets['assets']['reference_image'] = str(args.output/'reference.png')
    assets['assets']['prompt'] = prompt
    assets['default_example'] = 'gta_garage_two_humans_proxy_ood'
    (args.output/'assets.json').write_text(json.dumps(assets, indent=2))
    (args.output/'prompt.txt').write_text(prompt)
    metadata = dict(source_proxy=str(source), source_reference=str(args.data/'frames/000.jpg'),
                    source_fps=fps, source_frames=len(frames), output_fps=24, output_frames=124,
                    resampling='previous-frame hold, no motion interpolation',
                    reference_edit='Telea fill lower-left HUD; invalid mask saved; not observed evidence',
                    limitations=['Proxy geometry/camera differs visibly from RGB reference',
                                 'Red/blue human identities are outside the published vehicle palette',
                                 'Proxy has shading, not flat per-class semantics',
                                 'Single seed and scene; not a benchmark or a GTA driving test'])
    (args.output/'input_manifest.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'assets', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
