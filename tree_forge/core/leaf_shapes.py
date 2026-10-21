"""葉・花のインスタンス元ジオメトリ（三角形のみ、単位は m）。

ローカル座標: 葉は +Y が先端方向（付け根が原点）、+X が幅、+Z が表の法線。
返り値は (verts (n,3), tris (m,3), uv (n,2))。
"""
from __future__ import annotations

import math

import numpy as np

from .util import normalize, perp_basis


def _strip(widths: np.ndarray, length: float, fold: float, curl: float):
    """中央の葉脈に沿った帯。widths[i] は行 i の半幅。両端の行は 1 頂点（付け根・先端）。"""
    n = len(widths)
    t = np.linspace(0.0, 1.0, n)
    y = t * length
    zc = -curl * length * t ** 2
    verts = [(0.0, 0.0, 0.0)]
    uv = [(0.5, 0.0)]
    rows = []
    wmax = max(float(widths.max()), 1e-9)
    for i in range(1, n - 1):
        w = widths[i]
        base = len(verts)
        verts += [(-w, y[i], zc[i] + fold * w), (0.0, y[i], zc[i]), (w, y[i], zc[i] + fold * w)]
        uv += [(0.5 - 0.5 * w / wmax, t[i]), (0.5, t[i]), (0.5 + 0.5 * w / wmax, t[i])]
        rows.append(base)
    tip = len(verts)
    verts.append((0.0, y[-1], zc[-1]))
    uv.append((0.5, 1.0))
    tris = []
    r0 = rows[0]
    tris += [(0, r0, r0 + 1), (0, r0 + 1, r0 + 2)]
    for a, b in zip(rows[:-1], rows[1:]):
        tris += [(a, b, b + 1), (a, b + 1, a + 1), (a + 1, b + 1, b + 2), (a + 1, b + 2, a + 2)]
    rl = rows[-1]
    tris += [(rl, tip, rl + 1), (rl + 1, tip, rl + 2)]
    return np.array(verts, float), np.array(tris, np.int32), np.array(uv, float)


def broad(length: float, width_ratio: float, n_rows: int = 7):
    t = np.linspace(0.0, 1.0, n_rows)
    w = 0.5 * length * width_ratio * np.sin(np.pi * t ** 0.85) ** 0.9
    return _strip(w, length, fold=0.22, curl=0.12)


def scale_twig(length: float, width_ratio: float, n_rows: int = 9):
    t = np.linspace(0.0, 1.0, n_rows)
    zig = 0.55 + 0.45 * (np.arange(n_rows) % 2)
    w = 0.5 * length * width_ratio * (1.0 - 0.55 * t) * zig * np.sin(np.pi * np.clip(t, 0, 1) ** 0.5) ** 0.5
    return _strip(w, length, fold=0.10, curl=0.08)


def card(length: float, width_ratio: float):
    w = 0.5 * length * width_ratio
    verts = np.array([(-w, 0, 0), (w, 0, 0), (w, length, 0), (-w, length, 0)], float)
    uv = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)
    return verts, np.array([(0, 1, 2), (0, 2, 3)], np.int32), uv


def fan(radius: float, width_ratio: float, n_arc: int = 11):
    """イチョウ: 扇形（中央に切れ込み）。width_ratio は使わず、半角 70°。"""
    ang = np.linspace(-math.radians(70), math.radians(70), n_arc)
    r = radius * (1.0 - 0.28 * np.exp(-((ang / 0.12) ** 2)))
    verts = [(0.0, 0.0, 0.0)] + [(float(r[i] * math.sin(a)), float(r[i] * math.cos(a)), 0.05 * radius * (1 - math.cos(a)))
                                  for i, a in enumerate(ang)]
    uv = [(0.5, 0.0)] + [(0.5 + 0.5 * math.sin(a), 0.5 + 0.5 * math.cos(a)) for a in ang]
    tris = [(0, i, i + 1) for i in range(1, n_arc)]
    return np.array(verts, float), np.array(tris, np.int32), np.array(uv, float)


def _needle(base: np.ndarray, d: np.ndarray, facing: np.ndarray, length: float, width: float):
    side = normalize(np.cross(d, facing))
    return [base - side * width * 0.5, base + side * width * 0.5, base + d * length]


def needle_fascicle(length: float, count: int):
    """針葉束: count 本の針が付け根から扇状に出る。"""
    verts, uv, tris = [], [], []
    w = max(length * 0.025, 0.0008)
    for i in range(count):
        a = (i - (count - 1) / 2.0) * math.radians(9.0)
        d = np.array([math.sin(a), math.cos(a), 0.08])
        d = d / np.linalg.norm(d)
        b = len(verts)
        verts += _needle(np.zeros(3), d, np.array([0.0, 0.0, 1.0]), length, w)
        uv += [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        tris.append((b, b + 1, b + 2))
    return np.array(verts, float), np.array(tris, np.int32), np.array(uv, float)


def fascicle_tuft(length: float, count: int, n: int):
    """クロマツの枝先の房: n 組の針束が先端方向の円錐状に広がる。"""
    verts, uv, tris = [], [], []
    golden = math.radians(137.5)
    w = max(length * 0.025, 0.0008)
    for i in range(n):
        phi = i * golden
        polar = math.radians(15.0 + 55.0 * math.sqrt((i + 0.5) / n))
        axis = normalize(np.array([math.sin(polar) * math.cos(phi), math.cos(polar), math.sin(polar) * math.sin(phi)]))
        side0 = normalize(np.cross(axis, np.array([0.0, 0.0, 1.0])))
        for j in range(count):
            a = (j - (count - 1) / 2.0) * math.radians(9.0)
            d = normalize(axis * math.cos(a) + side0 * math.sin(a))
            b = len(verts)
            verts += _needle(np.zeros(3), d, normalize(np.cross(d, side0)), length, w)
            uv += [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
            tris.append((b, b + 1, b + 2))
    return np.array(verts, float), np.array(tris, np.int32), np.array(uv, float)


def needle_sprig(needle_len: float, n_needles: int = 16):
    """針葉のらせん: 短い枝に針がらせん状に付く。"""
    verts, uv, tris = [], [], []
    golden = math.radians(137.5)
    step = 0.2 * needle_len
    w = max(needle_len * 0.07, 0.0006)
    for i in range(n_needles):
        phi = i * golden
        radial = np.array([math.cos(phi), 0.0, math.sin(phi)])
        axis = np.array([0.0, 1.0, 0.0])
        d = normalize(axis * math.cos(math.radians(35)) + radial * math.sin(math.radians(35)))
        base = axis * i * step
        b = len(verts)
        verts += _needle(base, d, radial, needle_len, w)
        uv += [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        tris.append((b, b + 1, b + 2))
    return np.array(verts, float), np.array(tris, np.int32), np.array(uv, float)


def flower(radius: float):
    """5弁花。中心が原点、+Z が表。"""
    verts = [(0.0, 0.0, 0.0)]
    uv = [(0.5, 0.5)]
    tris = []
    for i in range(5):
        th = 2 * math.pi * i / 5
        for a, rr, zz in ((th - 0.5, 0.55, 0.05), (th, 1.0, 0.18), (th + 0.5, 0.55, 0.05)):
            verts.append((radius * rr * math.cos(a), radius * rr * math.sin(a), radius * zz))
            uv.append((0.5 + 0.5 * rr * math.cos(a), 0.5 + 0.5 * rr * math.sin(a)))
        b = 1 + 3 * i
        tris += [(0, b, b + 1), (0, b + 1, b + 2)]
    return np.array(verts, float), np.array(tris, np.int32), np.array(uv, float)


def _merge(parts):
    verts, tris, uv, base = [], [], [], 0
    for v, t, u in parts:
        verts.append(v)
        tris.append(t + base)
        uv.append(u)
        base += len(v)
    return np.vstack(verts), np.vstack(tris).astype(np.int32), np.vstack(uv)


def _xform(v, yaw, pitch, scale, offset):
    cz, sz, cx, sx = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (v * scale) @ (Rz @ Rx).T + offset


def cluster(single, n: int, length: float, rosette: bool = False):
    """葉 single (verts, tris, uv) を n 枚まとめた「葉つきの小枝」。+Y 方向に伸びる。決定的（乱数は固定シード）。"""
    rng = np.random.default_rng(7)
    v0, t0, u0 = single
    parts = []
    for i in range(n):
        if rosette:                                   # 短枝の束生（イチョウ）: 付け根から放射状
            yaw = (i - (n - 1) / 2.0) * math.radians(150.0 / max(n - 1, 1))
            off = np.zeros(3)
            sc, pitch = 1.0 - 0.05 * (i % 2), math.radians(rng.uniform(0, 12))
        else:                                         # 互生: 左右交互に開きながら先へ進む
            side = 1.0 if i % 2 == 0 else -1.0
            yaw = side * math.radians(rng.uniform(32, 58))
            off = np.array([side * 0.04 * length, i * 0.36 * length, 0.0])
            sc, pitch = max(0.55, 1.0 - 0.07 * i), math.radians(rng.uniform(-8, 14))
        parts.append((_xform(v0, yaw, pitch, sc, off), t0, u0))
    return _merge(parts)


def leaf_source(p):
    """パラメータに合う葉（または葉つきの小枝）のジオメトリ。"""
    L = p.leaf_length_cm * 0.01
    t = p.leaf_type
    n = p.leaf_cluster
    if n > 1 and t in ("BROAD", "FAN"):
        if t == "FAN":
            return cluster(fan(L, p.leaf_width_ratio, n_arc=7), n, L, rosette=True)
        single = card(L, p.leaf_width_ratio) if p.leaf_geom == "CARD" else broad(L, p.leaf_width_ratio, n_rows=5)
        return cluster(single, n, L)
    if p.leaf_geom == "CARD" and t in ("BROAD", "FAN", "SCALE_TWIG"):
        return card(L, p.leaf_width_ratio)
    if t == "BROAD":
        return broad(L, p.leaf_width_ratio)
    if t == "FAN":
        return fan(L, p.leaf_width_ratio)
    if t == "SCALE_TWIG":
        return scale_twig(L, p.leaf_width_ratio)
    if t == "NEEDLE_FASCICLE":
        return fascicle_tuft(L, p.fascicle_count, n) if n > 1 else needle_fascicle(L, p.fascicle_count)
    if t == "NEEDLE_SPIRAL":
        return needle_sprig(L)
    raise ValueError(t)


def flower_source(p):
    """花の房: 4輪が小さな球状にまとまる（遠景でも塊として見える）。"""
    r = 0.015 + 0.002 * p.leaf_length_cm
    base = flower(r)
    rng = np.random.default_rng(11)
    parts = []
    for i in range(4):
        th = 2 * math.pi * i / 4 + rng.uniform(-0.3, 0.3)
        off = np.array([math.cos(th), math.sin(th), rng.uniform(-0.2, 0.5)]) * r * 1.1
        parts.append((_xform(base[0], th, rng.uniform(-0.4, 0.4), rng.uniform(0.85, 1.1), off), base[1], base[2]))
    return _merge(parts)
