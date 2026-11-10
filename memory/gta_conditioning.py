"""Deterministic memory correspondence prompts without target-RGB annotations."""


def reference_prompt(edges):
    lines=[
        'Render the target whitebox camera and coarse world trajectory as realistic GTA gameplay.',
        'Use the historical RGB and paired whitebox frame only for the named objects\' visible exterior identity.',
        'Historical pose and screen position do not constrain target motion. Follow target whitebox motion, allowing natural limb articulation.'
    ]
    for edge in edges:
        lines.append(f"Object {edge['object_id']} ({edge['category']}) corresponds to reference {edge['reference_id']}, RGB box {edge['reference_bbox_xyxy']}, and its paired whitebox projection. Preserve its visible exterior; do not copy its historical pose.")
    lines.append('No object-specific historical appearance constraint is supplied for unreferenced background objects.')
    return '\n'.join(lines)
