"""単体テスト共通の小道具。"""
import hashlib

import numpy as np

from tree_forge import preset_io, schema
from tree_forge.core import pipeline


def species_params(sid, **ov):
    d = preset_io.builtin_overrides(sid)
    d.update(ov)
    return schema.make_params(**d)


def gen(sid="keyaki", **ov):
    return pipeline.generate(species_params(sid, **ov))


def mesh_hash(tree) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tree.mesh.co).tobytes())
    h.update(np.ascontiguousarray(tree.mesh.loop_vert).tobytes())
    return h.hexdigest()


def axis_rings(tree, axis_id=0):
    """軸 axis_id のリング頂点 (n_rings, V, 3)。頂点の並びは「リング頂点 n*V」→（グループ末尾に）先端中心、幹は根元中心も。"""
    pa = tree.mesh.point_attrs
    sel = np.nonzero(pa["branch_id"] == axis_id)[0]
    ax = tree.skeleton.axes[axis_id]
    extra = 2 if ax.parent < 0 else 1
    V = (len(sel) - extra) // ax.n
    return tree.mesh.co[sel[: ax.n * V]].reshape(ax.n, V, 3).astype(float), V, sel


def ring_radii(rings):
    c = rings.mean(axis=1, keepdims=True)
    return np.linalg.norm(rings - c, axis=2)
