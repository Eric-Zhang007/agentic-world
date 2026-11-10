"""Generate a text-only scene with existing H3, exporting lossless frames."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=109)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, '/home/star/agentic-world-comfy/ComfyUI')
    sys.argv = [sys.argv[0], '--disable-dynamic-vram', '--use-pytorch-cross-attention',
                '--enable-triton-backend']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import folder_paths
    import nodes
    import comfy.model_management as mm
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift
    from PIL import Image
    from run_comfy_control import save_video

    torch.set_num_threads(8)
    for kind, path in [('diffusion_models', '/mnt/hdd/MiniMax-H3-Quant/diffusion_models'),
                       ('text_encoders', '/mnt/hdd/MiniMax-H3-Quant/text_encoders'),
                       ('vae', '/mnt/hdd/MiniMax-H3/Ref2VA/video_vae/source')]:
        folder_paths.add_model_folder_path(kind, path)
    prompt = (
        'A crisp realistic high-end Chinese fantasy action game scene inspired by Black Myth Wukong. '
        'A monkey warrior with detailed brown fur and layered worn bronze armor holds one long wooden '
        'staff in both hands, seen in a full-body three-quarter rear view in the left foreground. '
        'His entire body, feet and staff are inside the image with comfortable margins. '
        'He faces a large muscular anthropomorphic orange tiger warrior with black stripes, a clearly '
        'readable feline face, leather belt and cloth trousers, standing in the right middle ground. '
        'Both are in balanced combat-ready poses, separated by four meters of clear traversable stone '
        'courtyard. Ancient Chinese temple carved stone steps and weathered statues in the background. '
        'Bright overcast daylight, natural neutral colors, clear silhouettes, finely resolved fur, '
        'armor engravings and stone texture. Deep focus with both fighters clearly sharp. '
        'Eye-level medium-wide gameplay camera, stationary camera, minimal idle breathing only. '
        'No attack, no motion blur, no fog, no depth-of-field blur, no HUD, no text, no watermark.'
    )
    record = dict(prompt=prompt, width=1024, height=768, frames=22, steps=40, seed=args.seed,
                  cache_threshold=0, kernel_backend='triton', source='H3_TEXT_GENERATED',
                  provenance='RENDER_ONLY', reference_image=None,
                  limitation='Short video sampled for a still; not real gameplay or authoritative world evidence.')
    (args.output/'metadata.json').write_text(json.dumps(record, indent=2))
    start = time.monotonic()
    with torch.inference_mode():
        print('load text encoder', flush=True)
        clip = nodes.CLIPLoader().load_clip('qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'minimax')[0]
        positive, latent = MiniMaxH3ReferenceToVideo.execute(
            clip, prompt, 1024, 768, 22, 'match').result
        mm.unload_all_models()
        del clip
        gc.collect()
        mm.soft_empty_cache()
        print('load H3 and sample', flush=True)
        model = nodes.UNETLoader().load_unet('minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'default')[0]
        model = MiniMaxH3SigmaShift.execute(model, 12.0, 3.0).result[0]
        sampled = nodes.KSampler().sample(model, args.seed, 40, 1.0, 'euler', 'simple', positive, positive, latent)[0]
        mm.unload_all_models()
        vae_path = Path('/mnt/hdd/MiniMax-H3/Ref2VA/video_vae/source')
        files = list(vae_path.glob('*.safetensors'))
        if len(files) != 1:
            raise RuntimeError('Expected exactly one existing H3 VAE')
        vae = nodes.VAELoader().load_vae(files[0].name)[0]
        images = nodes.VAEDecode().decode(vae, sampled)[0]
        if not torch.isfinite(images).all():
            raise RuntimeError('Nonfinite output')
        pixels = images.cpu().float().numpy()
        save_video(pixels, args.output/'generated.mp4')
        for index in (0, len(pixels)//2, len(pixels)-1):
            Image.fromarray((pixels[index].clip(0, 1)*255).round().astype('uint8')).save(args.output/f'frame_{index:03d}.png')
        record.update(seconds=time.monotonic()-start, actual_frames=len(pixels))
        (args.output/'metadata.json').write_text(json.dumps(record, indent=2))
        mm.unload_all_models()
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
