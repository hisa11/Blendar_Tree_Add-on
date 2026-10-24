"""葉・花のインスタンス元ジオメトリ（三角形のみ、単位は m）。

ローカル座標: 葉は +Y が先端方向（付け根が原点）、+X が幅、+Z が表の法線。
返り値は (verts (n,3), tris (m,3), uv (n,2))。
"""
from __future__ import annotations

import math

import numpy as np

from .util import normalize


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
    w = 0.5 * length * width_ratio * np.sin(np.pi * t ** 0.75) ** 0.9          # 最大幅は中央よりやや付け根寄り（卵形）
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
    """イチョウ: 長い葉柄と、くさび形の付け根から広がる扇形（外縁は波打ち、中央に浅い切れ込み）。width_ratio は使わない。

    radius は葉柄の付け根から葉先までの長さ。葉身は半径 0.42〜1.0 の範囲で、付け根側は角度を狭めたくさび形。
    """
    r_in = 0.42
    ang = np.linspace(-math.radians(42), math.radians(42), n_arc)
    r_out = radius * (1.0 + 0.035 * np.sin(ang * 9.0) - 0.16 * np.exp(-((ang / 0.10) ** 2)))        # 波打つ外縁と中央の浅い切れ込み
    r_base = radius * r_in
    verts = [(0.0, 0.0, 0.0)]                                                                          # 葉柄の付け根
    uv = [(0.5, 0.0)]
    for k, a in enumerate(ang):                                                                        # 葉身の付け根側（角度を 0.3 倍に狭めた弧）
        ab = 0.3 * a
        verts.append((float(r_base * math.sin(ab)), float(r_base * math.cos(ab)), 0.0))
        uv.append((0.5 + 0.25 * math.sin(ab), 0.25 + 0.0 * k))
    for a, r in zip(ang, r_out):                                                                       # 外縁
        verts.append((float(r * math.sin(a)), float(r * math.cos(a)), 0.05 * radius * (1 - math.cos(a))))
        uv.append((0.5 + 0.5 * math.sin(a), 0.5 + 0.5 * math.cos(a)))
    lo, hi = 1 + (n_arc - 1) // 2, 1 + n_arc // 2                                                      # 葉柄: 付け根から葉身の付け根の中央へ細く
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    tris = [(0, lo, hi)]
    inner, outer = 1, 1 + n_arc
    for i in range(n_arc - 1):
        tris += [(inner + i, outer + i, outer + i + 1), (inner + i, outer + i + 1, inner + i + 1)]
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


def needle_sprig(needle_len: float, n_needles: int = 24):
    """針葉のらせん: 短い枝に針がらせん状に付く。"""
    verts, uv, tris = [], [], []
    golden = math.radians(137.5)
    step = 0.14 * needle_len
    w = max(needle_len * 0.13, 0.0006)
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


def needle_spray(needle_len: float, n: int):
    """スギの枝葉: 針をらせん状に付けた小枝 n 本が先へ扇状に広がり、先が少し垂れる（遠景で「塊」に見える）。"""
    rng = np.random.default_rng(5)
    base = needle_sprig(needle_len)
    parts = []
    for i in range(n):
        f = (i - (n - 1) / 2.0) / max((n - 1) / 2.0, 1.0)                  # -1 〜 1
        yaw = f * math.radians(55.0)
        pitch = math.radians(-10.0 - 12.0 * abs(f) + rng.uniform(-4, 4))
        scale = 1.0 - 0.22 * abs(f) + rng.uniform(-0.05, 0.05)
        off = np.array([0.0, 0.2 * needle_len * abs(f), 0.0])
        parts.append((_xform(base[0], yaw, pitch, scale, off), base[1], base[2]))
    return _merge(parts)


def flower(radius: float):
    """5弁花（サクラ）。中心が原点、+Z が表。花びらは丸みのある倒卵形で、先端に浅い切れ込みがある。

    隣の花びらと少し重なる（1 枚あたり 76°）。重なる所は z をずらして面が重ならないようにする。
    """
    prof = ((-1.0, 0.93), (-0.6, 1.0), (-0.25, 0.95), (0.0, 0.80), (0.25, 0.95), (0.6, 1.0), (1.0, 0.93))   # (花びらの中心からの角度の割合, 半径)。先端は深く、花びらの境は浅く凹む
    half = 0.66                                                                          # 花びらの半角 [rad]
    verts = [(0.0, 0.0, 0.0)]
    uv = [(0.5, 0.5)]
    tris = []
    for i in range(5):
        th = 2 * math.pi * i / 5
        zo = 0.012 * radius * (i % 2)
        b = len(verts)
        for u, rr in prof:
            a = th + u * half
            verts.append((radius * rr * math.cos(a), radius * rr * math.sin(a), radius * (0.10 * rr * rr) + zo))
            uv.append((0.5 + 0.5 * rr * math.cos(a), 0.5 + 0.5 * rr * math.sin(a)))
        tris += [(0, b + k, b + k + 1) for k in range(len(prof) - 1)]
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


def _stem(length: float, width: float):
    """葉つきの小枝の軸（細い帯）。原点から +Y へ length。"""
    w = 0.5 * width
    return (np.array([(-w, 0.0, 0.0), (w, 0.0, 0.0), (w, length, 0.0), (-w, length, 0.0)], float),
            np.array([(0, 1, 2), (0, 2, 3)], np.int32), np.array([(0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (0.5, 1.0)], float))


def cluster(single, n: int, length: float, rosette: bool = False, staggered: bool = False):
    """葉 single (verts, tris, uv) を n 枚まとめた「葉つきの小枝」。+Y 方向に伸びる。決定的（乱数は固定シード）。

    staggered: 細い軸に沿って、左右交互に間をあけて付ける（サクラ・ケヤキなどの互生）。
    指定なしの互生は、付け根の近くに重ねて開く（葉が幅広のカエデが、掌状に開いた葉に見える）。
    """
    rng = np.random.default_rng(7)
    v0, t0, u0 = single
    parts = []
    if staggered:
        step = 0.5 * length
        parts.append(_stem(0.1 * length + (n - 1) * step + 0.2 * length, 0.03 * length))
    for i in range(n):
        if staggered:                                 # 軸に沿って左右交互に、前向きに開く
            side = 1.0 if i % 2 == 0 else -1.0
            yaw = side * math.radians(rng.uniform(38, 55))
            off = np.array([0.0, 0.1 * length + i * step, 0.0])
            sc, pitch = max(0.6, 1.0 - 0.08 * i), math.radians(rng.uniform(-8, 14))
        elif rosette:                                   # 短枝の束生（イチョウ）: 付け根から放射状
            yaw = (i - (n - 1) / 2.0) * math.radians(150.0 / max(n - 1, 1))
            off = np.array([-math.sin(yaw), math.cos(yaw), 0.0]) * 0.12 * length                              # 付け根を少しずつ放射状にずらして、重なりを減らす
            sc, pitch = 1.0 - 0.05 * (i % 2), math.radians((1.0 if i % 2 else -1.0) * rng.uniform(2, 6))
        else:                                         # 互生（付け根の近くに重ねて開く）: 左右交互に開きながら先へ進む
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
        return cluster(single, n, L, staggered=p.leaf_width_ratio < 0.9)
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
        return needle_spray(L, n) if n > 1 else needle_sprig(L)
    raise ValueError(t)


def flower_source(p):
    """花の房: 4輪が小さな球状にまとまる（遠景でも塊として見える）。"""
    r = 0.011 + 0.0008 * p.leaf_length_cm                    # 花の半径（葉の長さ 10 cm で 1.9 cm = 直径約 4 cm）
    base = flower(r)
    rng = np.random.default_rng(11)
    parts = []
    for i in range(4):
        th = 2 * math.pi * i / 4 + rng.uniform(-0.3, 0.3)
        off = np.array([math.cos(th), math.sin(th), rng.uniform(-0.2, 0.5)]) * r * 1.1
        parts.append((_xform(base[0], th, rng.uniform(-0.4, 0.4), rng.uniform(0.85, 1.1), off), base[1], base[2]))
    return _merge(parts)
