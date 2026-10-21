"""葉・花の配置点（位置・回転・スケール）。Blender 側が GN の Instance on Points に渡す。

葉は「末端の軸」の先端寄りに並べる。枯れ枝・折れ口より先には付けない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import allometry
from .skeleton import Skeleton
from .util import frame_matrices, mat_to_quat, normalize, perp_basis, smoothstep


@dataclass
class LeafData:
    pos: np.ndarray        # (N,3)
    quat: np.ndarray       # (N,4) w,x,y,z
    scale: np.ndarray      # (N,3)
    is_leaf: np.ndarray    # (N,) bool
    is_flower: np.ndarray  # (N,) bool
    var: np.ndarray        # (N,) 0..1 色の個体差
    leaf_frac: float = 1.0
    flower_frac: float = 0.0
    color_t: float = 0.0   # 0 緑 → 1 紅葉

    @property
    def n(self) -> int:
        return len(self.pos)


def season_state(season: float, evergreen: bool, flowering: bool) -> tuple[float, float, float]:
    """(葉の存在率, 花の存在率, 紅葉度)。0 冬 → .15 開花 → .3 新緑 → .85 紅葉 → 1 落葉。"""
    s = float(season)
    flower = 0.0
    if flowering:
        flower = float(smoothstep((s - 0.06) / 0.07) * (1.0 - smoothstep((s - 0.20) / 0.08)))
    if evergreen:
        return 1.0, flower, 0.0
    start = 0.25 if flowering else 0.12
    grow = float(smoothstep((s - start) / 0.18))
    fall = 1.0 - float(smoothstep((s - 0.90) / 0.10))
    color = float(smoothstep((s - 0.78) / 0.14))
    return grow * fall, flower, color


def _bearing_axes(sk: Skeleton):
    live = [a for a in sk.axes if not a.removed]
    top = max(a.level for a in live)
    has_child = {a.parent for a in live if a.parent >= 0}
    out = []
    for a in live:
        if a.dead or a.level == 0:
            continue
        thin = a.radius is not None and a.radius[-1] < 0.008
        if a.level == top or (a.id not in has_child and thin):
            out.append(a)
    return out


def _place(axes, per_m: float, u0: float, rng, frac_scale: float, theta_mean: float, theta_sd: float):
    """軸に沿って per_m 枚/m で点を置く。位置・接線・弧上位置を返す。"""
    P, Tn = [], []
    for a in axes:
        n_use = a.neck + 1 if a.neck >= 0 else a.n
        if n_use < 2:
            continue
        pos = a.pos[:n_use]
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        L = s[-1]
        if L < 0.02:
            continue
        s0 = u0 * L
        cnt = per_m * (L - s0) * frac_scale
        k = int(cnt + rng.random())
        if k <= 0:
            continue
        sa = s0 + (L - s0) * np.sort(rng.random(k))
        idx = np.clip(np.searchsorted(s, sa, side="right") - 1, 0, n_use - 2)
        f = ((sa - s[idx]) / np.maximum(seg[idx], 1e-9))[:, None]
        pt = pos[idx] * (1 - f) + pos[idx + 1] * f
        tg = normalize(pos[idx + 1] - pos[idx])
        P.append(pt)
        Tn.append(tg)
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(P), np.concatenate(Tn)


def _orient(T: np.ndarray, rng, theta_mean: float, theta_sd: float, hint_up: float):
    n = len(T)
    e1, e2 = perp_basis(T)
    phi = (np.arange(n) * allometry.golden_angle()) + rng.uniform(-0.5, 0.5, n)
    radial = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
    th = np.clip(rng.normal(theta_mean, theta_sd, n), 0.1, 1.5)
    fwd = normalize(T * np.cos(th)[:, None] + radial * np.sin(th)[:, None])
    hint = np.array([0.0, 0.0, 1.0]) * hint_up + 0.3 * radial
    roll = rng.normal(0.0, 0.3, n)
    # 前方軸まわりに法線ヒントを回す
    c, s = np.cos(roll)[:, None], np.sin(roll)[:, None]
    side = np.cross(fwd, hint)
    hint = hint * c + side * s
    return fwd, hint, radial


MAX_BOOST = 2.5          # 上限に達したとき、1枚あたりを大きくする倍率の上限
_AUTO_START = {"NEEDLE_SPIRAL": 0.1, "SCALE_TWIG": 0.1, "NEEDLE_FASCICLE": 0.5}     # それ以外は 0.2


def _sub_levels(sk: Skeleton, depth: int, exclude: set):
    """末端レベルから depth 段手前（>=1）の、枯れていない軸。exclude（すでに葉を付けた軸）は除く。"""
    live = [a for a in sk.axes if not a.removed]
    lv = max(a.level for a in live) - depth
    return [a for a in live if not a.dead and a.level == lv and a.level >= 1 and a.id not in exclude]


def generate(p, sk: Skeleton, st) -> LeafData:
    rng = st.leaves
    flowering = p.flower_density > 0.0
    leaf_frac, flower_frac, color_t = season_state(p.season, p.evergreen, flowering)
    cap = p.leaf_max_instances or allometry.quality_table(p.quality)["leaf_cap"]
    axes = _bearing_axes(sk)

    parts_pos, parts_quat, parts_scale, parts_leaf, parts_var = [], [], [], [], []

    # --- 葉 ---
    if leaf_frac > 0.01 and p.foliage_amount > 0.0:
        theta = {"BROAD": 0.95, "FAN": 0.95, "SCALE_TWIG": 0.7, "NEEDLE_FASCICLE": 0.5, "NEEDLE_SPIRAL": 0.12}[p.leaf_type]
        per_m = p.leaves_per_m * p.foliage_amount
        start = p.leaf_start if p.leaf_start >= 0.0 else _AUTO_START.get(p.leaf_type, 0.2)
        done = {a.id for a in axes}
        pts, tg = _place(axes, per_m, start, rng, leaf_frac, theta, 0.25)
        reach = p.leaf_reach
        if reach == "AUTO":
            reach = "OUTER" if p.leaf_type in ("NEEDLE_SPIRAL", "SCALE_TWIG", "NEEDLE_FASCICLE") else "TERMINAL"
        extra = []
        if reach in ("OUTER", "ALL"):                      # 末端の一段手前の枝の外側半分（量感）
            sub1 = _sub_levels(sk, 1, done)
            done |= {a.id for a in sub1}
            extra.append(_place(sub1, per_m * 0.5, 0.45, rng, leaf_frac, theta, 0.25))
        if reach == "ALL":                                  # さらに手前の枝の外側 40%
            extra.append(_place(_sub_levels(sk, 2, done), per_m * 0.25, 0.6, rng, leaf_frac, theta, 0.25))
        # 幹の上部（樹冠の頂部）。幹は _bearing_axes から外れるため、ここで先端寄りだけに付ける
        live = [a for a in sk.axes if not a.removed]
        extra.append(_place([a for a in live if a.level == 0 and not a.dead], per_m * 0.5, 0.93, rng, leaf_frac, theta, 0.25))
        for e_pts, e_tg in extra:
            pts, tg = np.concatenate([pts, e_pts]), np.concatenate([tg, e_tg])
        n = len(pts)
        boost = 1.0
        if n > cap:
            keep = np.sort(rng.choice(n, cap, replace=False))
            pts, tg = pts[keep], tg[keep]
            boost = min(MAX_BOOST, math.sqrt(n / cap))
            n = len(pts)
        if n:
            fwd, hint, _ = _orient(tg, rng, theta, 0.25, 1.0)
            R = frame_matrices(fwd, hint)
            sc = (1.0 + rng.uniform(-p.leaf_scale_var, p.leaf_scale_var, n)) * boost * p.leaf_scale * (0.6 + 0.4 * np.sqrt(leaf_frac))
            parts_pos.append(pts)
            parts_quat.append(mat_to_quat(R))
            parts_scale.append(np.repeat(sc[:, None], 3, axis=1))
            parts_leaf.append(np.ones(n, bool))
            parts_var.append(rng.random(n))

    # --- 花（L2 以降の細い軸に）---
    if flower_frac > 0.01 and p.flower_density > 0.0:
        f_axes = [a for a in sk.axes if not a.removed and not a.dead and a.level >= 2]
        pts, tg = _place(f_axes, p.leaves_per_m * 1.2 * p.flower_density, 0.15, rng, flower_frac, 0.8, 0.3)
        n = len(pts)
        if n > cap // 2:
            keep = np.sort(rng.choice(n, cap // 2, replace=False))
            pts, tg = pts[keep], tg[keep]
            n = len(pts)
        if n:
            fwd, hint, radial = _orient(tg, rng, 0.9, 0.3, 0.6)
            hint = normalize(radial + np.array([0.0, 0.0, 0.5]))
            R = frame_matrices(fwd, hint)
            sc = (1.0 + rng.uniform(-0.2, 0.2, n)) * p.leaf_scale
            parts_pos.append(pts)
            parts_quat.append(mat_to_quat(R))
            parts_scale.append(np.repeat(sc[:, None], 3, axis=1))
            parts_leaf.append(np.zeros(n, bool))
            parts_var.append(rng.random(n))

    if not parts_pos:
        z3, z4 = np.zeros((0, 3)), np.zeros((0, 4))
        return LeafData(z3, z4, z3.copy(), np.zeros(0, bool), np.zeros(0, bool), np.zeros(0),
                        leaf_frac, flower_frac, color_t)
    is_leaf = np.concatenate(parts_leaf)
    return LeafData(
        pos=np.concatenate(parts_pos), quat=np.concatenate(parts_quat), scale=np.concatenate(parts_scale),
        is_leaf=is_leaf, is_flower=~is_leaf, var=np.concatenate(parts_var),
        leaf_frac=leaf_frac, flower_frac=flower_frac, color_t=color_t,
    )
