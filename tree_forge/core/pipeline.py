"""パラメータ → 木のジオメトリ（bpy 非依存）。Blender 側はこの結果を Mesh に書くだけ。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import damage, leaf_points, radius, skeleton, tube_mesher
from .rng import make_streams


@dataclass
class Tree:
    params: object
    skeleton: skeleton.Skeleton
    mesh: tube_mesher.MeshData
    leaves: leaf_points.LeafData


def generate(p) -> Tree:
    st = make_streams(p.seed)
    sk = skeleton.build(p, st)
    radius.compute_radii(p, sk)         # 健全な状態の太さ
    damage.apply_damage(p, sk, st)      # 折れ・垂れ下がり
    skeleton.enforce_budget(p, sk, st.skel)
    skeleton.fit_width(p, sk)           # 失われた枝先の分だけ樹冠幅を合わせ直す
    mesh = tube_mesher.build_mesh(p, sk, st)
    leaves = leaf_points.generate(p, sk, st)
    return Tree(p, sk, mesh, leaves)


def measure(tree: Tree) -> tuple[float, float]:
    """生成結果のメッシュから (樹高, 樹冠幅) を測る。"""
    co = tree.mesh.co
    return float(co[:, 2].max()), float(0.5 * (np.ptp(co[:, 0]) + np.ptp(co[:, 1])))
