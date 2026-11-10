# Coarse motion conditions

The GTA compiler now uses the dataset's fixed 13-part humanoid asset, exported
from its original source. It renders neither procedural joint spheres nor limb
animation. Root position, heading, overall size and held objects remain
available; coarse posture scales its overall height without joint animation. The earlier capsule
probe remains an isolated historical experiment. See `gta_asset_alignment.md`
for the imported resource catalog and regional collision-surface previews.

The dataset has recorded game cameras/transforms and collision surfaces for
buildings and terrain. The single-image prototype instead has inferred camera,
block-composed buildings and a height field. Removing detailed actor rendering
does not close that background geometry gap.

## Motion across clips

Use `end_velocity` on continuing move phases. The next segment starts with the
accepted actor velocity. Legacy actions without this field still reproduce
their earlier easing/hold behavior for history replay; new agent instructions
explicitly select ongoing motion instead of adding a wait to every clip.

For camera motion, use:

```json
{
  "camera_orbit_yaw_deg": 9,
  "camera_end_motion": {
    "translation_velocity": [0, 0, 0],
    "orbit_velocity_deg": 3,
    "distance_velocity": 0
  }
}
```

Rates are world units/second and degrees/second. Hermite trajectories use the
previous camera motion and requested ending rates. With actor-following,
translation follows the actor root; explicit camera translation rate is zero.
The camera motion is saved in the endpoint camera. An explicitly moving camera
cannot silently revert to the old stationary-boundary path. Exact H3 timing is
required: alignment cannot append a settling pause to these camera actions.
Old plans retain their old interpolation, preserving reproducible history.

Tests check constant-speed translation and orbit across two 3-second segments,
and reject automatic settle-padding for moving camera plans. The existing
actor-continuity and video-reconciliation suite passes alongside these tests.

## What the models receive

H3 receives the current accepted RGB first frame, native whitebox RGB video,
depth-derived ControlNet frames and a textual action/camera prompt. Current
depth strength is 0.5 over the whole clip, encoded white at 2 world units and
black at 18; no separate pose, segmentation or optical-flow ControlNet is used.
RGB and depth are produced from the same coarse geometry and timeline.

The Pi reviewer currently receives sampled images, not an MP4 video payload:
normally a 0.5-second contact sheet and full opening/ending frames, with optional
extra frame reads. Dense temporal crops help check motion but do not constitute
native video perception. Coarse displacement and natural gait are separate
review findings.

## Live probe

`scripts/probe_coarse_continuation.py` builds two simulated 3-second plans with
ongoing actor and follow/orbit-camera motion, then compiles and generates the
first one. The second plan verifies planned boundary continuity only: it is not
a video-reconciled next segment. No authoritative world is modified or accepted.
Artifacts are under `outputs/coarse_continuation_20261102`.
