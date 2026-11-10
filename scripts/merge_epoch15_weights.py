"""Reassemble the seven supplied binary safetensors parts without changing them."""
import json
import math
from pathlib import Path
import shutil
import struct


def main():
    root = Path('C:/Users/zjc/Downloads')
    parts = [root / f'epoch-15.part-{i:03d}.mp4' for i in range(7)]
    output = root / 'epoch-15.safetensors'
    temporary = root / 'epoch-15.safetensors.partial'
    if output.exists() or temporary.exists():
        raise FileExistsError('Output or partial file already exists')
    with parts[0].open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    cursor = 0
    for name, tensor in sorted(tensors.items(), key=lambda item: item[1]['data_offsets'][0]):
        start, end = tensor['data_offsets']
        if tensor['dtype'] != 'BF16':
            raise ValueError(f'Unexpected dtype: {name}')
        if start != cursor or end - start != math.prod(tensor['shape']) * 2:
            raise ValueError(f'Invalid tensor shape or data offsets: {name}')
        cursor = end
    expected = 8 + header_size + cursor
    if sum(p.stat().st_size for p in parts) != expected:
        raise ValueError('Parts do not match declared length')
    with temporary.open('xb') as destination:
        for part in parts:
            with part.open('rb') as source:
                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
    if temporary.stat().st_size != expected:
        raise ValueError('Incomplete assembled file')
    with temporary.open('rb') as assembled:
        for part in parts:
            with part.open('rb') as source:
                while block := source.read(8 * 1024 * 1024):
                    if assembled.read(len(block)) != block:
                        raise ValueError(f'Byte comparison failed: {part.name}')
        if assembled.read(1):
            raise ValueError('Unexpected trailing bytes')
    loader_check = 'not installed; header, shapes, offsets and every source byte verified'
    try:
        from safetensors import safe_open
    except ImportError:
        pass
    else:
        with safe_open(str(temporary), framework='numpy') as weights:
            if set(weights.keys()) != set(tensors):
                raise ValueError('Safetensors key mismatch')
            for name, tensor in tensors.items():
                if weights.get_slice(name).get_shape() != tensor['shape']:
                    raise ValueError('Safetensors shape mismatch')
        loader_check = 'safe_open passed for all tensor keys and shapes'
    temporary.rename(output)
    print(json.dumps({'path': str(output), 'bytes': expected, 'tensor_count': len(tensors),
                      'dtype': 'BF16', 'source_byte_comparison': 'passed',
                      'safetensors_loader': loader_check, 'parts_preserved': True}, indent=2))


if __name__ == '__main__':
    main()
