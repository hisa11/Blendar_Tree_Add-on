"""骨格 → チューブメッシュ（頂点・面・UV・属性）。

リング頂点数が同じ軸（= 同じレベル）をまとめ、(軸, 節, 頂点) の3次元配列で一気に作る。
  * 断面フレームは平行移動（回転最小）で捻れを避ける
  * 断面の歪み・根張り・こぶ・縦筋は cross_section.py
  * 折れ口は最後のリングを不揃いに前へ押し出し、中心を凹ませた蓋をする（材質スロット 7）
  * UV: u = φ/2π × k（k=整数に丸めたタイル数）、v = 弧長 / タイル高。k が変わる節で継ぎ目が入る
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import allometry, cross_section
from .radius import collar_factor
from .skeleton import Skeleton
from .util import normalize, perp_basis, rotate_about

PART_TRUNK, PART_BRANCH, PART_TWIG, PART_LEAF, PART_FLOWER, PART_FRUIT, PART_ROOT = range(7)
SLOT_BROKEN = 7
TWIG_RADIUS = 0.012          # 付け根の半径がこれ未満の枝は「小枝」


@dataclass
class MeshData:
    co: np.ndarray               # (V,3) float32
    loop_vert: np.ndarray        # (L,) int32
    face_size: np.ndarray        # (P,) int32
    uv: np.ndarray               # (L,2) float32
    material_index: np.ndarray   # (P,) int32
    face_attrs: dict             # name -> (P,) int32
    point_attrs: dict            # name -> (V,) or (V,3)

    @property
    def tri_count(self) -> int:
        return int(np.sum(self.face_size - 2))


def _pad(axs, getter, M, fill_last=True):
    out = np.empty((len(axs), M) + getter(axs[0]).shape[1:])
    for i, a in enumerate(axs):
        v = getter(a)
        out[i, : len(v)] = v
        out[i, len(v):] = v[-1]
    return out


def _frames(T: np.ndarray):
    """平行移動フレーム。T (A,M,3) → U, V (A,M,3)。"""
    A, M, _ = T.shape
    U = np.empty_like(T)
    U[:, 0], _ = perp_basis(T[:, 0])
    for k in range(1, M):
        a, b = T[:, k - 1], T[:, k]
        ax = np.cross(a, b)
        sn = np.linalg.norm(ax, axis=1)
        ok = sn > 1e-9
        axn = np.where(ok[:, None], ax / np.maximum(sn, 1e-12)[:, None], 0.0)
        ang = np.arctan2(sn, np.sum(a * b, axis=1))
        U[:, k] = np.where(ok[:, None], rotate_about(U[:, k - 1], axn, ang), U[:, k - 1])
    return U, np.cross(T, U)


def _group(p, axs, V, dist, dmax, rng, h_root):
    A = len(axs)
    n = np.array([a.n for a in axs])
    M = int(n.max())
    kk = np.arange(M)
    valid = kk[None, :] < n[:, None]
    last = n - 1
    ia = np.arange(A)

    pos = _pad(axs, lambda a: a.pos, M)
    rad = _pad(axs, lambda a: a.radius, M)
    dst = _pad(axs, lambda a: dist[a.id], M)
    seg = np.linalg.norm(np.diff(pos, axis=1), axis=2)
    s = np.concatenate([np.zeros((A, 1)), np.cumsum(seg, axis=1)], axis=1)
    L = s[:, -1]

    T = normalize(pos[:, np.minimum(kk + 1, M - 1)] - pos[:, np.maximum(kk - 1, 0)])
    T = T[ia[:, None], np.minimum(kk[None, :], last[:, None])]
    U, W = _frames(T)

    level = np.array([a.level for a in axs])
    is_root = np.array([a.parent < 0 for a in axs])
    r0 = rad[:, 0]
    r_eff = np.where(is_root[:, None], rad, rad * collar_factor(s, r0[:, None]))
    for i, a in enumerate(axs):
        if 0 <= a.neck < a.n:
            r_eff[i, a.neck] *= 0.55
            if a.neck + 1 < a.n:
                r_eff[i, a.neck + 1] *= 0.8

    irr = np.where(level == 0, p.trunk_irregularity, p.branch_irregularity) * math.sqrt(p.age_factor)
    phi = 2 * math.pi * np.arange(V) / V
    F, off = cross_section.section_factors(rng, p, s=s, z=pos[..., 2], r=r_eff, L=L, level=level,
                                           is_root=is_root, phi=phi, irr=irr, valid=valid)
    for i, a in enumerate(axs):
        if 0 <= a.neck < a.n:
            F[i, a.neck] *= rng.uniform(0.6, 1.0, V)

    ph = rng.uniform(0, 2 * math.pi, A)[:, None] + phi[None, :]
    radial = np.cos(ph)[:, None, :, None] * U[:, :, None, :] + np.sin(ph)[:, None, :, None] * W[:, :, None, :]
    ctr = pos + off[..., 0:1] * U + off[..., 1:2] * W
    co = ctr[:, :, None, :] + (r_eff[:, :, None] * F)[..., None] * radial

    # --- 折れ口: 最後のリングを不揃いに前へ ---
    broken = np.array([a.broken for a in axs])
    bro = np.nonzero(broken)[0]
    if len(bro):
        ke = last[bro]
        t_e = T[bro, ke]
        r_e = r_eff[bro, ke]
        ring = co[bro, ke]
        c_e = ctr[bro, ke]
        inward = rng.uniform(0.65, 1.0, (len(bro), V))
        jag = rng.random((len(bro), V)) ** 1.3
        for _ in range(2 if V >= 6 else 1):                       # 繊維が長く残るささくれ
            jag[np.arange(len(bro)), rng.integers(0, V, len(bro))] *= 2.4
        shift = (0.15 + 1.2 * jag) * r_e[:, None]
        co[bro, ke] = c_e[:, None, :] + (ring - c_e[:, None, :]) * inward[..., None] + t_e[:, None, :] * shift[..., None]
        co[bro, ke - 1] += t_e[:, None, :] * (0.18 * shift)[..., None]

    # --- 頂点配列（軸ごとに連続: 節 k, 頂点 j の順）---
    ring_co = co[valid].reshape(-1, 3)
    counts = n * V
    off_v = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_ring = int(counts.sum())
    root_ids = np.nonzero(is_root)[0]
    tip_idx = n_ring + ia
    base_idx = n_ring + A + np.arange(len(root_ids))

    r_end = r_eff[ia, last]
    t_end = T[ia, last]
    tip_c = ctr[ia, last] + t_end * (0.7 * r_end)[:, None]
    if len(bro):
        tip_c[bro] = co[bro, last[bro]].mean(axis=1) - t_end[bro] * (0.3 * r_end[bro])[:, None]
    base_c = ctr[root_ids, 0] - T[root_ids, 0] * (0.3 * r_eff[root_ids, 0])[:, None]
    co_all = np.concatenate([ring_co, tip_c, base_c], axis=0)

    # --- 頂点属性 ---
    def pa(arr):  # (A,M) or (A,M,3)
        full = np.broadcast_to(arr[:, :, None, ...], arr.shape[:2] + (V,) + arr.shape[2:])
        ring_v = full[valid].reshape((-1,) + arr.shape[2:])
        return np.concatenate([ring_v, arr[ia, last], arr[root_ids, 0]], axis=0)

    axis_id = np.array([a.id for a in axs])
    phase = np.array([a.phase for a in axs])
    blend_len = max(p.junction_blend_len, 1e-6)
    blend = np.where(is_root[:, None], 0.0, np.clip(1.0 - s / (blend_len * r0[:, None]), 0.0, 1.0))
    pivot = np.broadcast_to(pos[:, :1, :], pos.shape)
    point_attrs = {
        "tree_level": pa(np.broadcast_to(np.minimum(level, 3)[:, None], (A, M)).astype(np.int32)),
        "dist_root": pa(dst),
        "dist_root_n": pa(np.clip(dst / dmax, 0.0, 1.0)),
        "branch_t": pa(np.clip(s / np.maximum(L, 1e-9)[:, None], 0.0, 1.0)),
        "branch_phase": pa(np.broadcast_to(phase[:, None], (A, M)).copy()),
        "branch_pivot": pa(pivot),
        "branch_id": pa(np.broadcast_to(axis_id[:, None], (A, M)).astype(np.int32)),
        "junction_blend": pa(blend),
    }

    # --- 側面の四角形 ---
    S = M - 1
    kmask = np.arange(S)[None, :] < (n - 1)[:, None]
    a_i, k_i = np.nonzero(kmask)
    j = np.arange(V)
    b0 = off_v[a_i] + k_i * V
    c00 = b0[:, None] + j[None, :]
    c01 = b0[:, None] + (j + 1) % V
    quads = np.stack([c00, c01, c01 + V, c00 + V], axis=-1).reshape(-1, 4)

    r_mid = 0.5 * (r_eff[:, :-1] + r_eff[:, 1:])[a_i, k_i]
    tiles = 2 * math.pi * r_mid / p.bark_tile_w_m
    tiles = np.maximum(1, np.rint(tiles)) if p.uv_integer_wrap else np.maximum(tiles, 0.25)
    v_off = np.where(is_root, 0.0, rng.random(A) * 7.0)
    v0 = (v_off[a_i] + s[a_i, k_i]) / p.bark_tile_h_m
    v1 = (v_off[a_i] + s[a_i, k_i + 1]) / p.bark_tile_h_m
    u0 = (j[None, :] / V) * tiles[:, None]
    u1 = ((j[None, :] + 1) / V) * tiles[:, None]
    V0, V1 = np.broadcast_to(v0[:, None], u0.shape), np.broadcast_to(v1[:, None], u0.shape)
    uv_quads = np.stack([np.stack([u0, V0], -1), np.stack([u1, V0], -1), np.stack([u1, V1], -1),
                         np.stack([u0, V1], -1)], axis=2).reshape(-1, 2)

    # 面の属性（軸×区間 → 各四角形に V 個ずつ）
    part_axis = np.where(level == 0, PART_TRUNK, np.where(r0 < TWIG_RADIUS, PART_TWIG, PART_BRANCH))
    part_seg = np.broadcast_to(part_axis[:, None], (A, S)).copy()
    z_mid = 0.5 * (pos[:, :-1, 2] + pos[:, 1:, 2])
    part_seg[is_root[:, None] & (z_mid < h_root)] = PART_ROOT
    brk_seg = np.zeros((A, S), bool)
    for i, a in enumerate(axs):
        if a.broken and a.n >= 2:
            brk_seg[i, a.n - 2] = True
        if 0 <= a.neck < a.n - 1:
            brk_seg[i, a.neck] = True
    part_q = np.repeat(part_seg[a_i, k_i], V)
    brk_q = np.repeat(brk_seg[a_i, k_i], V)

    # --- 先端の蓋（全軸）と根元の蓋（幹のみ）---
    lb = off_v + last * V
    t0 = lb[:, None] + j[None, :]
    t1 = lb[:, None] + (j + 1) % V
    tip_tris = np.stack([t0, t1, np.broadcast_to(tip_idx[:, None], t0.shape)], axis=-1).reshape(-1, 3)
    b_off = off_v[root_ids]
    bb0 = b_off[:, None] + j[None, :]
    bb1 = b_off[:, None] + (j + 1) % V
    base_tris = np.stack([bb1, bb0, np.broadcast_to(base_idx[:, None], bb0.shape)], axis=-1).reshape(-1, 3)

    ang = 2 * math.pi * j / V
    ang1 = 2 * math.pi * (j + 1) / V

    def polar(a0, a1):
        c = np.array([0.5, 0.5])
        return np.stack([0.5 + 0.5 * np.stack([np.cos(a0), np.sin(a0)], -1),
                         0.5 + 0.5 * np.stack([np.cos(a1), np.sin(a1)], -1),
                         np.broadcast_to(c, (V, 2))], axis=1)           # (V,3,2)

    uv_tip = np.tile(polar(ang, ang1), (A, 1, 1)).reshape(-1, 2)
    uv_base = np.tile(polar(ang1, ang), (len(root_ids), 1, 1)).reshape(-1, 2)
    part_tip = np.repeat(part_axis, V)
    brk_tip = np.repeat(broken, V)
    part_base = np.repeat(part_axis[root_ids], V)

    face_vert = np.concatenate([quads.reshape(-1), tip_tris.reshape(-1), base_tris.reshape(-1)])
    face_size = np.concatenate([np.full(len(quads), 4), np.full(len(tip_tris), 3), np.full(len(base_tris), 3)])
    uv = np.concatenate([uv_quads, uv_tip, uv_base], axis=0)
    part = np.concatenate([part_q, part_tip, part_base])
    brk = np.concatenate([brk_q, brk_tip, np.zeros(len(base_tris), bool)])
    return dict(co=co_all, face_vert=face_vert, face_size=face_size, uv=uv, part=part, brk=brk,
                point_attrs=point_attrs, n_verts=len(co_all))


def build_mesh(p, sk: Skeleton, st) -> MeshData:
    qual = allometry.quality_table(p.quality)
    rng = st.bark
    dist: list[np.ndarray] = []
    for ax in sk.axes:
        base = 0.0 if ax.parent < 0 else float(dist[ax.parent][min(ax.parent_node, len(dist[ax.parent]) - 1)])
        dist.append(base + ax.arc())
    dmax = max(float(d[-1]) for d in dist) or 1.0
    h_root = p.root_flare_height * p.height_m

    groups = []
    for lv in range(4):
        axs = [a for a in sk.axes if min(a.level, 3) == lv]
        if axs:
            groups.append(_group(p, axs, max(3, qual["ring"][lv]), dist, dmax, rng, h_root))

    cos, fv, fs, uvs, parts, brks = [], [], [], [], [], []
    pas: dict[str, list] = {}
    base = 0
    for g in groups:
        cos.append(g["co"])
        fv.append(g["face_vert"] + base)
        fs.append(g["face_size"])
        uvs.append(g["uv"])
        parts.append(g["part"])
        brks.append(g["brk"])
        for k, v in g["point_attrs"].items():
            pas.setdefault(k, []).append(v)
        base += g["n_verts"]
    part = np.concatenate(parts).astype(np.int32)
    brk = np.concatenate(brks)
    return MeshData(
        co=np.concatenate(cos).astype(np.float32),
        loop_vert=np.concatenate(fv).astype(np.int32),
        face_size=np.concatenate(fs).astype(np.int32),
        uv=np.concatenate(uvs).astype(np.float32),
        material_index=np.where(brk, SLOT_BROKEN, part).astype(np.int32),
        face_attrs={"tree_part": part, "tree_break": brk.astype(np.int32)},
        point_attrs={k: np.concatenate(v).astype(np.int32 if k in ("tree_level", "branch_id") else np.float32)
                     for k, v in pas.items()},
    )
