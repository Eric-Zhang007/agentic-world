# Native whitebox video reference

`ComfyH3Renderer.render(..., reference_video=...)` now forwards a distinct
`--reference-video` argument to the installed native MiniMax H3 reference node.
It can be combined with `first_frame`, `anchor_first` and optional ControlNet
conditions. Existing callers retain their previous behavior until migrated.

The installed node presents images as `<Picture 1>` and video as `<Video 1>`.
It samples the reference video for Qwen at 2 fps with timestamps and also encodes
the video with the VAE into a `minimax_refs` video block. This is reference
conditioning, not a guarantee of exact trajectory reproduction.

The interfaces have different meanings:

| Argument | Role |
| --- | --- |
| `reference_video` | Native video reference through Qwen and VAE |
| `first_frame` | Appearance reference image |
| `anchor_first=True` | Also apply image as frame-zero guide |
| `prefix_video` | Guide accepted historical frames at the output start |
| `source_video` | Legacy source for frame extraction and default Canny |
| `control_video` | External RGB-valued ControlNet condition, e.g. depth |

Use `controls=[]` to disable ControlNet explicitly; simply omitting depth can
activate legacy Canny derivation. Do not pass the future whitebox animation as
an accepted-history prefix guide.

The diagnostic entry point is `scripts/run_native_whitebox_trial.py`. It uses
an existing saved scene and timeline without world edits, renders a 124-frame
whitebox video at the output canvas, and appends explicit modality-role text.
It is an interface validation, not a prompt-identical ablation. Full acceptance
and performance still depend on reviewing the real generated output.

Runtime artifacts include `native_video_reference.mp4`, sampling metadata and
`reference_block_kinds`. A requested native video with no encoded video block
raises an error before sampling. The first trial is stored separately under
`outputs/native_whitebox_reference_20261013`.
