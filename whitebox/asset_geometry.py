"""Local asset vertices using the same scaling and origins as Godot's catalog."""
from itertools import product
import json
from pathlib import Path
import numpy as np
from whitebox.posture import body_size

CATALOG = json.loads((Path(__file__).resolve().parents[1]/'assets/gta/catalog.json').read_text())['assets']


def local_vertices(entity):
    kind = entity['kind']
    size = np.asarray(body_size(entity) if kind=='actor' else entity['size'],float)
    asset_id = 'gta_human' if kind=='actor' else entity.get('asset_id')
    if asset_id in CATALOG and kind in ('actor','box'):
        asset = CATALOG[asset_id]
        vertices = np.asarray([v for part in asset['parts'] for v in part['vertices']],float)
        vertices = vertices * (size/np.asarray(asset['size']))
        if kind=='box':
            vertices[:,1] -= size[1]/2
        return vertices
    y = (0,1) if kind=='actor' else (-.5,.5)
    return np.asarray(list(product((-.5,.5),y,(-.5,.5)))) * size
