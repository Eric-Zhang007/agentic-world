# Fresh SAM2 and image-to-whitebox checks

## What ran

Local work spanning November 9-10 used the existing native SAM2 tiny checkpoint
and source tree, not cached predictions. Agent-authored bounding boxes select
the regions. Categories and object associations therefore come from the agent;
SAM is not being credited with discovering or identifying them.

`outputs/local_fresh_sam_20261109/` contains eight fresh masks, source images,
overlays and reports for an urban street and garage. All masks were visually
inspected. People and bollards have useful coarse outlines. Architectural masks
can include adjacent structures or occluders, so they are not complete building
instances. The garage player's feet leave the image: its lower mask edge is not
a ground-contact observation.

Actual timing with RTX 3060 Laptop and the existing torch 2.0.1 CUDA environment:

| Stage | Time |
| --- | --- |
| Model construction/load, excluding imports | 3.827 s |
| First urban image encoding | 0.509 s |
| Four urban prompts | 0.286 s |
| Subsequent garage image encoding | 0.112 s |
| Four garage prompts | 0.028 s |

Torch peak allocated memory was 556-559 MiB, not total GPU residency. Inference
used bfloat16 autocast. Optional connected-component postprocessing was disabled
because no compiled SAM extension is installed. The original source/weights were
not modified. Reported mask IoU scores are model predictions, not measured IoU.

## Isolated dependencies

Installed only into `.runtime/sam2-deps` using the Tsinghua pip mirror:
`hydra-core==1.3.2`, `omegaconf==2.3.0`, `antlr4-python3-runtime==4.9.3`.
Existing Torch/torchvision and other environments were not upgraded or changed.
No model weight download or paid model API call occurred.

From the project directory in WSL:

```bash
/home/zjc/.venvs/cross-fact-gpu/bin/python -B scripts/segment_local_boxes.py \
  --repo /mnt/d/fact-stack/repos/sam2 \
  --checkpoint /mnt/d/fact-stack/models/sam2/sam2_hiera_tiny.pt \
  --dependencies .runtime/sam2-deps \
  --requests configs/local_sam_box_requests.json \
  --output outputs/new_sam_run --gpu 0
```

## Does the tool expose actual construction errors?

Yes. `scripts/diagnose_mask_alignment.py` compares mask bounds to projected coarse
entity bounds, using explicit agent-provided mappings. It does not perform object
matching. Cropped edges are recorded, and behind-camera bounds are unavailable.
The result is bbox IoU, not silhouette IoU or a quality acceptance score.

Actual reports are under `outputs/local_mask_alignment_20261109/`:

- Garage player center error: 238.81 pixels; bbox overlap zero.
- Garage NPC center error: 173.95 pixels; bbox overlap zero.
- Garage left facade is projected on the opposite side of the image.
- Urban player center error: 25.26 pixels; parking booth: 80.96 pixels.

The garage failure is primarily coordinate layout/camera alignment, not insufficient
human asset detail. A running NPC's silhouette and a standing proxy need not have
the same width. The whole lamp mask also includes the separately modeled base.
These distinctions matter before deciding what to edit.

Example from Windows project root, after SAM outputs exist:

```powershell
python scripts/diagnose_mask_alignment.py --scene outputs/local_garage_agent_20261109/scene_01.json --perception outputs/local_fresh_sam_20261109/01/result.json --mapping configs/garage_mask_world_mapping.json --output outputs/new_mask_check
```

## Depth order as additional evidence

`perception/depth_order.py` compares relative depth in explicit visible patches
against scene-origin camera depth. Both margins retain their separate units.
The four urban patches agree on all six pairwise orders. This is predicted-order
agreement, not metric accuracy; projected location and scale can still be wrong.
See `docs/local_depth_order.md`. The existing urban failure illustrates why an
ordering check cannot replace image alignment.

No authoritative world was initialized or changed. The next integration step is
to use these observations to revise a candidate, rerender it, and recheck anchors
not used during fitting; merely collecting green diagnostic results is not enough.
