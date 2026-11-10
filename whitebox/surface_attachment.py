"""Explicit static-box face attachments, independent of inference and storage."""
from copy import deepcopy
import math


def _axes(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return ((c, 0, -s), (0, 1, 0), (s, 0, c))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def apply_surface_attachments(scene, relations):
    """Return a deep copy with child centers projected onto explicit box faces.

    Tangential coordinates and all fields except child position are preserved.
    Rotation angles are in radians; box sizes are full extents.
    """
    candidate = deepcopy(scene)
    entities = {e['id']: e for e in candidate['entities']}
    relations = list(relations)
    children, parents = set(), set()
    for relation in relations:
        if not isinstance(relation, dict) or set(relation) != {'child', 'parent', 'axis', 'sign', 'gap'}:
            raise ValueError('Attachment requires child, parent, axis, sign and gap')
        child, parent = relation['child'], relation['parent']
        if (not isinstance(child, str) or not isinstance(parent, str)
                or child not in entities or parent not in entities or child == parent):
            raise ValueError('Attachment requires distinct known IDs')
        if child in children:
            raise ValueError('A child may have only one attachment')
        if any(entities[key].get('kind') != 'box' for key in (child, parent)):
            raise ValueError('Attachments support static boxes only')
        if type(relation['axis']) is not int or relation['axis'] not in (0, 1, 2):
            raise ValueError('Invalid attachment axis')
        if type(relation['sign']) is not int or relation['sign'] not in (-1, 1):
            raise ValueError('Invalid attachment sign')
        gap = relation['gap']
        if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap < 0:
            raise ValueError('Attachment gap must be finite and nonnegative')
        children.add(child)
        parents.add(parent)
    if children & parents:
        raise ValueError('Chained or cyclic attachments are not supported')
    for relation in relations:
        child, parent = entities[relation['child']], entities[relation['parent']]
        axis = relation['axis']
        normal = tuple(relation['sign'] * x for x in _axes(parent.get('rotation_y', 0))[axis])
        support = sum(abs(_dot(normal, direction)) * size / 2
                      for direction, size in zip(_axes(child.get('rotation_y', 0)), child['size']))
        target = _dot(parent['position'], normal) + parent['size'][axis] / 2 + relation['gap'] + support
        shift = target - _dot(child['position'], normal)
        child['position'] = [p + shift * n for p, n in zip(child['position'], normal)]
    return candidate
