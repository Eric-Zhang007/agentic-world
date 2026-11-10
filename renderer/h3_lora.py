"""Apply all PEFT LoRA pairs explicitly; never silently drop unmatched tensors."""
def apply_lora(model, path, strength=1.0):
    import comfy.utils
    import comfy.lora
    weights = comfy.utils.load_torch_file(str(path), safe_load=True)
    mapping = comfy.lora.model_lora_keys_unet(model.model, {})
    state = model.model.state_dict()
    for key, value in weights.items():
        if '.lora_A.default.weight' not in key:
            continue
        stem = key.replace('.lora_A.default.weight', '')
        target = mapping.get(stem)
        other = weights[stem + '.lora_B.default.weight']
        if target is None or tuple(state[target].shape) != (other.shape[0], value.shape[1]):
            raise ValueError(f'LoRA incompatible with base: {stem}, target={target}')
    patches = comfy.lora.load_lora(weights, mapping)
    if len(patches)*2 != len(weights):
        raise ValueError(f'Incomplete LoRA mapping: {len(patches)} patches / {len(weights)} tensors')
    model = model.clone()
    accepted = model.add_patches(patches, strength)
    if len(accepted) != len(patches):
        raise ValueError('Some LoRA patches were rejected')
    return model, {'path': str(path), 'strength': strength, 'patches': len(accepted)}
