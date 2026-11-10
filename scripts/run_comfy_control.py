"""Run real native ComfyUI H3 nodes headlessly, with measured control injection."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import time
import subprocess
from fractions import Fraction


def read_prefix(path, total_frames, width, height):
    """Decode a native guide batch without temporal resampling or truncation."""
    import av
    import numpy as np

    images = []
    first_time = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != Fraction(24, 1):
            raise ValueError("Prefix video must have exact 24 fps")
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Prefix video requires contiguous timestamps")
            timestamp = frame.pts * frame.time_base
            if first_time is None:
                first_time = timestamp
            if timestamp - first_time != Fraction(len(images), 24):
                raise ValueError("Prefix video requires contiguous exact 24 fps timestamps")
            images.append(np.asarray(frame.to_image().convert("RGB").resize((width, height))))
            if len(images) >= total_frames:
                raise ValueError("Prefix frame count must be less than total frames")
    if len(images) < 5 or len(images) % 17 != 5:
        raise ValueError("Prefix frame count must be 17*k+5 and at least 5; no native cropping")
    return np.stack(images)


def add_start_guide(node, positive, latent, vae, reference, anchor_first, prefix=None):
    if prefix is not None or anchor_first:
        image = prefix if prefix is not None else reference
        return node.execute(positive, latent, 0, vae=vae, image=image).result[0]
    return positive


def save_video(images, path):
    import av
    import numpy as np
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=24)
        stream.width, stream.height = images.shape[2], images.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for image in images:
            pixels = (np.clip(image, 0, 1) * 255).round().astype("uint8")
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def encode_references(node, clip, prompt, width, height, frames, vae, images, video=None):
    kwargs = {'vae': vae, 'ref_images': images}
    if video is not None:
        kwargs['ref_videos'] = {'ref_video_1': video}
    return node.execute(clip, prompt, width, height, frames, 'match', **kwargs).result


def main(args):
    if args.frames < 5 or args.frames % 17 != 5:
        raise ValueError("Frame count must be 17*n+5; no implicit control padding")
    if min(args.width, args.height) < 32 or args.width % 32 or args.height % 32:
        raise ValueError("Canvas dimensions must be positive multiples of 32")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if not args.source and not args.reference_image:
        raise ValueError("A source video or reference image is required")
    prefix_path = getattr(args, "prefix_video", None)
    prefix_pixels = (read_prefix(prefix_path, args.frames, args.width, args.height)
                     if prefix_path is not None else None)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.comfy.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # Keep Comfy's argument parser separate from this runner's interface.
    sys.argv = [sys.argv[0], "--disable-dynamic-vram", "--use-pytorch-cross-attention"]
    if args.kernel_backend == "triton":
        sys.argv.append("--enable-triton-backend")
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    import folder_paths
    import nodes
    import comfy.model_management as mm
    from comfy_extras.nodes_minimax_h3 import (
        MiniMaxH3ReferenceToVideo, MiniMaxH3FunControlNetApply, MiniMaxH3SigmaShift, MiniMaxH3AddGuide)
    from comfy_extras.nodes_model_patch import ModelPatchLoader
    from comfy_extras.nodes_easycache import EasyCacheNode, EasyCacheHolder
    from renderer.reference_video import read_reference
    from perception.hud import load_profile, invalid_mask, clean_reference, clean_canny
    from renderer.temporal_control import temporal_schedule, gate_tensors

    torch.set_num_threads(8)
    folder_paths.add_model_folder_path("diffusion_models", "/mnt/hdd/MiniMax-H3-Quant/diffusion_models")
    folder_paths.add_model_folder_path("text_encoders", "/mnt/hdd/MiniMax-H3-Quant/text_encoders")
    folder_paths.add_model_folder_path("model_patches", "/mnt/hdd/agentic-world-models/model_patches")
    folder_paths.add_model_folder_path("vae", "/mnt/hdd/MiniMax-H3/Ref2VA/video_vae/source")
    started = time.monotonic()
    metadata = {"backend": "ComfyUI native nodes, headless execution", "request": vars(args).copy(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "base_variant": "ref2va_pruned_int8_convrot", "runs": []}
    metadata["comfy_commit"] = subprocess.check_output(
        ["git", "-C", str(args.comfy), "rev-parse", "HEAD"], text=True).strip()
    metadata["request"] = {k: str(v) if isinstance(v, Path) else v
                           for k, v in metadata["request"].items()}
    metadata['request']['appearance_image'] = [str(p) for p in getattr(args,'appearance_image',[]) or []]
    metadata["prefix_video"] = str(Path(prefix_path).resolve()) if prefix_path is not None else None
    metadata["prefix_frame_count"] = len(prefix_pixels) if prefix_pixels is not None else 0
    cache_resets=[]
    original_cache_reset=EasyCacheHolder.reset
    def measured_cache_reset(holder):
        cache_resets.append(getattr(holder,"total_steps_skipped",0))
        return original_cache_reset(holder)
    EasyCacheHolder.reset=measured_cache_reset

    def event(stage, **extra):
        message = {"stage": stage, "elapsed": time.monotonic() - started, **extra}
        print(json.dumps(message), flush=True)
        with (out / "events.jsonl").open("a") as f:
            f.write(json.dumps(message) + "\n")

    if args.source:
        frames, sampling = read_reference(str(args.source), args.frames, 24, args.width, args.height)
        rgb = np.stack([np.asarray(frame) for frame in frames])
    else:
        still = np.asarray(Image.open(args.reference_image).convert("RGB").resize((args.width,args.height)))
        rgb = np.repeat(still[None],args.frames,axis=0)
        sampling = {"kind":"static_reference_only", "invented_source_motion":False}

    def read_control(path):
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path,allow_pickle=False) as data:
                pixels = data["rgb"]
                if pixels.shape != (args.frames,args.height,args.width,3) or pixels.dtype != np.uint8:
                    raise ValueError(f"Control shape/dtype mismatch: {pixels.shape}, {pixels.dtype}")
                return pixels.copy()
        values,_ = read_reference(str(path),args.frames,24,args.width,args.height)
        return np.stack([np.asarray(frame) for frame in values])
    hud_mask = None
    if args.hud_profile:
        profile = load_profile(args.hud_profile)
        hud_mask = invalid_mask(rgb.shape[1:], profile, margin=3)
        Image.fromarray(hud_mask).save(out / "invalid_pixels.png")
        metadata["hud"] = {"profile": profile, "excluded_fraction": float((hud_mask>0).mean()),
                           "inpainted_pixels_are_evidence": False}
    if args.control_video:
        control = read_control(args.control_video)
        metadata["control_input"] = {"kind": "external_RGB_control_video",
                                     "path": str(args.control_video)}
    else:
        control = np.stack([clean_canny(frame, hud_mask) if hud_mask is not None else
                            np.repeat(cv2.Canny(frame, 100, 200)[..., None], 3, axis=2) for frame in rgb])
        metadata["control_input"] = {"kind": "Canny_from_source_RGB", "thresholds": [100, 200]}
    save_video(rgb / 255.0, out / "source_resampled.mp4")
    save_video(control / 255.0, out / ("control_input.mp4" if args.control_video else "control_canny.mp4"))
    if args.reference_image:
        reference_pixels = np.asarray(Image.open(args.reference_image).convert("RGB").resize(
            (args.width, args.height)))
    else:
        reference_pixels = rgb[0]
    if hud_mask is not None:
        reference_pixels = clean_reference(reference_pixels, hud_mask)
        if not args.control_video:
            metadata["hud"]["control_nonzero_inside_invalid"] = int(np.count_nonzero(control[:,hud_mask>0]))
        else:
            metadata["hud"]["external_world_control_not_derived_from_HUD"] = True
    Image.fromarray(reference_pixels).save(out / "reference.png")
    metadata["source_sampling"] = sampling
    reference = torch.from_numpy(reference_pixels[None].copy()).float() / 255
    control_tensor = torch.from_numpy(control.copy()).float() / 255

    with torch.inference_mode():
        event("load_vae")
        vae_files = list(Path("/mnt/hdd/MiniMax-H3/Ref2VA/video_vae/source").glob("*.safetensors"))
        if len(vae_files) != 1:
            raise ValueError(f"Expected one VAE file, found {vae_files}")
        vae = nodes.VAELoader().load_vae(vae_files[0].name)[0]
        vae_config = json.loads((vae_files[0].parent.parent / "config.json").read_text())
        for name in ("latents_mean", "latents_std"):
            actual = getattr(vae.first_stage_model, name).cpu()
            expected = torch.tensor(vae_config[name], dtype=actual.dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        metadata["vae_normalization_verified_against_original_config"] = True
        event("load_text_encoder")
        clip = nodes.CLIPLoader().load_clip("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax")[0]
        event("encode_reference_and_prompt")
        from renderer.memory_conditioning import image_bindings
        history_tensor = None
        if args.history_pair:
            if args.appearance_image:
                raise ValueError('Paired history cannot be mixed with legacy appearance crops')
            with Image.open(args.history_pair) as image:
                history_pixels = np.asarray(image.convert('RGB')).copy()
            history_tensor = torch.from_numpy(history_pixels[None]).float()/255
        ref_images = image_bindings(reference, history_tensor)
        metadata['image_roles'] = ({'1':'historical RGB left / whitebox right','2':'current RGB first frame'}
                                   if history_tensor is not None else {'1':'current RGB first frame'})
        appearance_paths = getattr(args,'appearance_image',[]) or []
        if len(appearance_paths)>8:raise ValueError('At most eight additional appearance images')
        for i,path in enumerate(appearance_paths,2):
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB')).copy()
            ref_images[f'ref_image_{i}'] = torch.from_numpy(pixels[None]).float()/255
        metadata['appearance_images'] = [str(p) for p in appearance_paths]
        reference_video = None
        reference_video_path = getattr(args, 'reference_video', None)
        if reference_video_path is not None:
            video_frames, video_sampling = read_reference(
                str(reference_video_path), args.frames, 24, args.width, args.height)
            reference_video = torch.from_numpy(np.stack([np.asarray(f) for f in video_frames])).float()/255
            metadata['native_video_reference'] = video_sampling
            save_video(reference_video.numpy(), out / 'native_video_reference.mp4')
        positive, latent = encode_references(
            MiniMaxH3ReferenceToVideo, clip, args.prompt, args.width, args.height,
            args.frames, vae, ref_images, reference_video)
        ref_blocks = positive[0][1].get('minimax_refs', [])
        metadata['reference_block_kinds'] = [block['kind'] for block in ref_blocks]
        if reference_video is not None and 'video' not in metadata['reference_block_kinds']:
            raise RuntimeError('Requested video reference did not reach native conditioning')
        (out / 'conditioning_metadata.json').write_text(json.dumps(metadata, indent=2))
        prefix_tensor = (torch.from_numpy(prefix_pixels.copy()).float() / 255
                         if prefix_pixels is not None else None)
        positive = add_start_guide(MiniMaxH3AddGuide, positive, latent, vae, reference,
                                   args.anchor_first, prefix_tensor)
        metadata["first_frame_guide"] = bool(args.anchor_first and prefix_pixels is None)
        if args.last_frame:
            last_pixels=np.asarray(Image.open(args.last_frame).convert('RGB').resize((args.width,args.height)))
            last_tensor=torch.from_numpy(last_pixels[None].copy()).float()/255
            positive=MiniMaxH3AddGuide.execute(positive,latent,-1,vae=vae,image=last_tensor).result[0]
            Image.fromarray(last_pixels).save(out/'last_frame_guide.png')
        metadata['last_frame_guide']=str(args.last_frame) if args.last_frame else None
        metadata['resolved_guide_indices']=[k['resolved_frame_index'] for k in positive[0][1].get('minimax_keyframes',[])]
        metadata["reference_blocks"] = len(positive[0][1].get("minimax_refs", []))
        mm.unload_all_models()
        del clip
        gc.collect()
        mm.soft_empty_cache()
        event("load_dit")
        model = nodes.UNETLoader().load_unet(
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "default")[0]
        if args.lora:
            from renderer.h3_lora import apply_lora
            model, metadata['lora'] = apply_lora(model, args.lora)
            event('lora_loaded', **metadata['lora'])
        model = MiniMaxH3SigmaShift.execute(model, 12.0, 3.0).result[0]
        event("load_controlnet")
        patch = ModelPatchLoader().load_model_patch(
            "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors")[0]
        metadata["injection_layers"] = list(patch.model.injection_layers)
        counts = Counter()
        residuals = {}
        original_step = patch.model.step
        original_init_stream = patch.model.init_stream
        schedule = None
        row_weights = None

        def gated_init_stream(h, control_latent, layout, t_emb):
            nonlocal row_weights
            if schedule is not None:
                control_latent, row_weights = gate_tensors(schedule, control_latent, layout)
            else:
                row_weights = None
            return original_init_stream(h, control_latent, layout, t_emb)

        patch.model.init_stream = gated_init_stream

        def measured_step(index, *a, **kw):
            c, skip = original_step(index, *a, **kw)
            if row_weights is not None:
                skip = skip * row_weights.to(device=skip.device, dtype=skip.dtype)[:, None]
            counts[index] += 1
            if index not in residuals:
                if not torch.isfinite(skip).all():
                    raise RuntimeError("Nonfinite ControlNet residual")
                residuals[index] = float(skip.float().abs().mean().item())
            return c, skip

        patch.model.step = measured_step
        cases = json.loads(args.cases.read_text()) if args.cases else [
            {"label":"controlled" if s else "baseline", "strength":s,
             "cache_threshold":args.cache_threshold} for s in ([1.0,0.0] if args.compare else [args.strength])]
        labels = [case["label"] for case in cases]
        if len(set(labels)) != len(labels) or any(not label.replace("_", "").isalnum() for label in labels):
            raise ValueError("Case labels must be unique safe names")
        for case in cases:
            label = case["label"]
            if (out / f"{label}.mp4").exists():
                raise FileExistsError(f"Refusing to overwrite {label}.mp4")
            counts.clear()
            residuals.clear()
            cache_resets.clear()
            strength = float(case.get("strength",args.strength))
            controls = case.get("controls", [{"strength":strength}])
            schedule = temporal_schedule(args.frames, case["temporal_keyframes"]) if "temporal_keyframes" in case else None
            if schedule is not None and len(controls) != 1:
                raise ValueError("Temporal experiment requires exactly one control stream")
            current = model
            for condition in controls:
                pixels = read_control(condition["path"]) if "path" in condition else control
                if schedule is not None:
                    pixels = pixels.copy()
                    inactive = np.ones(args.frames, dtype=bool)
                    inactive[schedule["active_frame_indices"]] = False
                    pixels[inactive] = 0
                    save_video(pixels / 255.0, out / f"{label}_control.mp4")
                weight = float(condition.get("strength",strength))
                start = float(condition.get("start",0.0))
                end = float(condition.get("end",args.control_end))
                if not 0 <= start < end <= 1 or not 0 <= weight <= 2:
                    raise ValueError("Invalid control weight or sampling interval")
                current = MiniMaxH3FunControlNetApply.execute(
                    current, patch, vae, weight, start, end,
                    control_video=torch.from_numpy(pixels.copy()).float()/255).result[0]
            cache_threshold = float(case.get("cache_threshold",args.cache_threshold))
            if cache_threshold < 0:
                raise ValueError("cache threshold must be nonnegative")
            if cache_threshold:
                current = EasyCacheNode.execute(current,cache_threshold,0.15,0.95,False).result[0]
            label = case["label"]
            steps = int(case.get("steps",args.steps))
            event("sample", label=label, strength=strength, steps=steps, cache_threshold=cache_threshold)
            torch.cuda.reset_peak_memory_stats()
            run_started = time.monotonic()
            sampled = nodes.KSampler().sample(
                current, int(case.get("seed",args.seed)), steps, 1.0, "euler", "simple",
                positive, positive, latent)[0]
            torch.cuda.synchronize()
            sample_seconds = time.monotonic()-run_started
            event("decode", strength=strength, control_calls=dict(counts))
            images = nodes.VAEDecode().decode(vae, sampled)[0]
            if not torch.isfinite(images).all():
                raise RuntimeError("Nonfinite decoded video")
            pixels = images.cpu().float().numpy()
            save_video(pixels, out / f"{label}.mp4")
            for index in (0, len(pixels)//2, len(pixels)-1):
                Image.fromarray((pixels[index].clip(0, 1)*255).astype("uint8")).save(
                    out / f"{label}_{index:03d}.png")
            run = {"label": label, "strength": strength, "frames": len(pixels),
                   "seed":int(case.get("seed",args.seed)),
                   "temporal_schedule":schedule,
                   "temporal_method":"blank unselected input; zero unselected latent hints and direct residual rows" if schedule else None,
                   "seconds": time.monotonic() - run_started,
                   "sample_seconds":sample_seconds,"steps":steps,"cache_threshold":cache_threshold,
                   "cache_skipped_evaluations":max(cache_resets,default=0),
                   "controls":controls,"peak_allocated_gb":torch.cuda.max_memory_allocated()/1e9,
                   "control_calls": dict(counts), "first_residual_abs_mean": dict(residuals),
                   "pixel_std": float(pixels.std()), "pixel_mean": float(pixels.mean()),
                   "mean_frame_change": float(np.abs(np.diff(pixels, axis=0)).mean())}
            if any(c.get("strength",strength)>0 for c in controls) and (set(counts) != set(range(10)) or not any(residuals.values())):
                raise RuntimeError("Control branch was not exercised")
            metadata["runs"].append(run)
            (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
            event("saved", **run)
            del sampled, images, pixels, current
            gc.collect()
        mm.unload_all_models()
        event("complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--comfy", type=Path, required=True)
    p.add_argument('--lora', type=Path)
    p.add_argument('--history-pair', type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--control-video", type=Path)
    p.add_argument("--reference-image", type=Path)
    p.add_argument("--reference-video", type=Path, help="Native <Video 1> reference; independent of ControlNet and prefix guides")
    p.add_argument('--appearance-image',type=Path,action='append',default=[])
    p.add_argument("--last-frame",type=Path)
    p.add_argument("--prefix-video", type=Path)
    p.add_argument("--hud-profile", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frames", type=int, default=73)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--cases",type=Path)
    p.add_argument("--strength",type=float,default=1.0)
    p.add_argument("--control-end",type=float,default=1.0)
    p.add_argument("--cache-threshold",type=float,default=0.0)
    p.add_argument("--kernel-backend",choices=("default","triton"),default="default")
    p.add_argument("--anchor-first",action="store_true")
    p.add_argument("--prompt", default=(
        "A realistic video game scene matching <Picture 1>. A man wearing a blue and "
        "yellow shirt stands inside an industrial garage beside a red tool cabinet. "
        "The garage door is open, daylight enters from the left. Preserve the man's "
        "appearance, the workbench, concrete walls, and the camera composition. "
        "Subtle natural body movement."))
    main(p.parse_args())
