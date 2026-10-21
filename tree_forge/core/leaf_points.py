"""葉・花の配置点（位置・回転・スケール）。Blender 側が GN の Instance on Points に渡す。

葉は「末端の軸」の先端寄りに並べる。枯れ枝・折れ口より先には付けない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import allometry, envelope, leaf_shapes
from .skeleton import Skeleton, _center_fn
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
    color = float(smoothstep((s - 0.62) / 0.20))
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


def _place(axes, per_m: float, u0: float, rng, frac_scale: float, theta_mean: float, theta_sd: float,
           surface: bool = False):
    """軸に沿って per_m 枚/m で点を置く。位置・接線を返す。surface=True は点を軸の半径ぶん外（樹皮の表面）へずらす（太い幹用）。"""
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
        if surface and a.radius is not None:
            r = np.interp(sa, s, a.radius[:n_use])
            e1, e2 = perp_basis(tg)
            phi = rng.random(len(sa)) * (2.0 * math.pi)
            pt = pt + (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2) * (0.9 * r)[:, None]
        P.append(pt)
        Tn.append(tg)
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(P), np.concatenate(Tn)


def _orient(T: np.ndarray, rng, theta_mean: float, theta_sd: float, hint_up: float, roll_sd: float = 0.3):
    n = len(T)
    e1, e2 = perp_basis(T)
    phi = (np.arange(n) * allometry.golden_angle()) + rng.uniform(-0.5, 0.5, n)
    radial = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
    th = np.clip(rng.normal(theta_mean, theta_sd, n), 0.1, 1.5)
    fwd = normalize(T * np.cos(th)[:, None] + radial * np.sin(th)[:, None])
    hint = np.array([0.0, 0.0, 1.0]) * hint_up + 0.3 * radial
    roll = rng.normal(0.0, roll_sd, n)
    # 前方軸まわりに法線ヒントを回す
    c, s = np.cos(roll)[:, None], np.sin(roll)[:, None]
    side = np.cross(fwd, hint)
    hint = hint * c + side * s
    return fwd, hint, radial


MAX_BOOST = 2.5          # 上限に達したとき、1枚あたりを大きくする倍率の上限
_AUTO_START = {"NEEDLE_SPIRAL": 0.1, "SCALE_TWIG": 0.1, "NEEDLE_FASCICLE": 0.5}     # それ以外は 0.2
_AUTO_SCATTER = {"NEEDLE_SPIRAL": 0.7}                                              # それ以外は 0（従来どおり枝に沿う）
SCATTER_POS = 0.35       # 位置の散らばり: 房の大きさに対する、枝からの最大のずれの割合（scatter 1 のとき）
SCATTER_THETA = 0.55     # 向きの散らばり: 枝の接線からの角度の平均に足す量 [rad]（scatter 1 のとき）
SCATTER_THETA_SD = 0.25  # 同・ばらつきに足す量 [rad]
SCATTER_ROLL_SD = 1.0    # 同・房の面の向きのばらつき [rad]


def _sub_levels(sk: Skeleton, depth: int, exclude: set):
    """末端レベルから depth 段手前（>=1）の、枯れていない軸。exclude（すでに葉を付けた軸）は除く。"""
    live = [a for a in sk.axes if not a.removed]
    lv = max(a.level for a in live) - depth
    return [a for a in live if not a.dead and a.level == lv and a.level >= 1 and a.id not in exclude]


VIRTUAL_TARGET = 0.8     # 高品質（HERO）が予算内で持てる L3 の本数の見積りに対する割合（HERO の実測は見積りの 0.8〜1.5 倍）
VIRTUAL_DEADBAND = 0.9   # すでにある葉が目標のこの割合に届いていれば足さない
VIRTUAL_BOOST = 1.8      # 足す葉の上限は「上限枚数 × この倍率² × foliage_amount」。MAX_BOOST までの余地を、葉の量の調整用に残す


def _hero_twigs(p, sk: Skeleton, l2: list) -> tuple[float, float]:
    """高品質（HERO）が同じ木に付ける葉の量を、L3 の細枝の本数に換算して見積もる。(その本数, L3 1 本の平均の長さ [m])。

    enforce_budget の高品質版: 三角形の予算（1.2M）の中で、L0・L1 の次に L2 を入れ、残りを L3 に回す。
    L2 が全部入るなら L3 の本数（上限 l3_cap）、入りきらない大木では L3 が無くなり L2 が自分で葉を付けるので、
    その長さを細枝の本数に直す。
    """
    per = {"SPIRAL": 1, "OPPOSITE": 2, "WHORL": p.l3_whorl_count}[p.l3_phyllotaxis]
    hq = allometry.quality_table("HERO")
    live = [a for a in sk.axes if not a.removed]
    stems = [a for a in live if a.level == 0 and a.radius is not None]
    r_stem = max((a.radius[0] for a in stems), default=0.0) * p.root_flare_scale
    ring = [max(3, allometry.ring_verts("HERO", lv, r_stem)) for lv in range(4)]

    def cost(a):
        return (int(allometry.n_segments(np.array([a.length()]), a.level, "HERO")[0]) * 2 + 1) * ring[a.level]

    fixed = sum(cost(a) for a in live if a.level <= 1 or (a.level == 2 and a.dead))     # 葉の付かない L2 は間引かれない
    c2 = np.array([cost(a) for a in l2], float)
    lens = np.array([a.length() for a in l2])
    # L3 の平均の長さ。長い L2 ほど L3 も多く長いので、L2 の長さで重みを付ける。0.575 は位置 u に応じた係数 max(.15, 1-.75u) と乱数の平均
    twig_len = float((lens ** 2).sum() / lens.sum()) * p.l3_length_ratio * 0.575
    room = hq["tri_cap"] * 0.97 - fixed
    if room >= c2.sum():                                                   # L2 は全部入る
        c3 = (int(allometry.n_segments(np.array([twig_len]), 3, "HERO")[0]) * 2 + 1) * ring[3]
        expected = p.l3_density * 0.85 * lens.sum() / per
        return min(expected, hq["l3_cap"], (room - c2.sum()) / c3), twig_len
    return lens.sum() * max(room, 0.0) / c2.sum() / twig_len, twig_len


def virtual_twig_points(p, sk: Skeleton, rng, per_m: float, start: float, leaf_frac: float, n_existing: int,
                        max_points: int):
    """低い品質（L3 を間引いた・作らない）で、L3 の細枝があるはずの場所に葉の点だけを足す。

    枝は作らない。細い枝（太さ数 mm）は遠目には見えず、葉の量だけが品質で変わってしまうのを防ぐ。
    L2（予算で間引いたものを含む）の途中に、_child_specs と同じ分布（位置 u・角度・長さ）で仮想の細枝を立て、その上に葉の点を置く。
    すでにある葉の点（n_existing）を細枝何本ぶんかに換算して差し引き、足りない分だけ足す。
    HERO は基準なので何も足さない。(位置, 接線) を返す。
    """
    empty = (np.zeros((0, 3)), np.zeros((0, 3)))
    if p.quality == "HERO":
        return empty
    l2 = [a for a in sk.axes if not a.removed and a.level == 2 and not a.dead and a.n >= 2]
    l2 += [a for a in sk.ghosts if a.n >= 2]             # 予算で間引いた L2 にも、あったはずの細枝（の葉）を立てる
    trunk = next((a for a in sk.axes if a.level == 0 and a.parent < 0 and not a.removed), None)
    if not l2 or trunk is None:
        return empty
    target, twig_len = _hero_twigs(p, sk, l2)
    target *= VIRTUAL_TARGET
    leaves_per_twig = max(per_m * leaf_frac * twig_len * (1.0 - start), 0.5)
    have = n_existing / leaves_per_twig                  # すでにある葉の点を、細枝何本ぶんかに換算
    if have >= VIRTUAL_DEADBAND * target:
        return empty
    cum, cat_p, starts, ends, lens = [], [], [], [], []
    off, k = 0.0, 0
    for a in l2:                                          # 全 L2 の節を 1 本の配列にして、位置の探索を一括でする
        n_use = a.neck + 1 if a.neck >= 0 else a.n
        if n_use < 2:
            continue
        P = a.pos[:n_use]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        if s[-1] < 0.05:
            continue
        cum.append(s + off)
        cat_p.append(P)
        starts.append(k)
        ends.append(k + n_use)
        lens.append(s[-1])
        off += s[-1] + 1.0
        k += n_use
    if not lens:
        return empty
    G, PC = np.concatenate(cum), np.concatenate(cat_p)
    starts, ends, lens = np.array(starts), np.array(ends), np.array(lens)
    offs = G[starts]
    missing = min(int(target - have), int(max_points / leaves_per_twig))
    if missing <= 0:
        return empty
    w = lens / lens.sum()
    par = rng.choice(len(lens), missing, p=w)
    u = rng.uniform(0.12, 0.97, missing)
    key = offs[par] + u * lens[par]
    kk = np.clip(np.searchsorted(G, key, side="right") - 1, starts[par], ends[par] - 2)
    f = ((key - G[kk]) / np.maximum(G[kk + 1] - G[kk], 1e-9))[:, None]
    base = PC[kk] * (1 - f) + PC[kk + 1] * f
    t = normalize(PC[kk + 1] - PC[kk])
    e1, e2 = perp_basis(t)
    ang = np.clip(math.radians(p.l3_angle_deg) + rng.normal(0.0, math.radians(p.l3_angle_var_deg), missing),
                  math.radians(5), math.radians(140))
    phi = rng.uniform(0.0, 2.0 * math.pi, missing)
    d = np.cos(ang)[:, None] * t + np.sin(ang)[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
    length = lens[par] * p.l3_length_ratio * np.maximum(0.15, 1.0 - 0.75 * u) * rng.uniform(0.8, 1.15, missing)
    s0 = start * length
    cnt = per_m * leaf_frac * (length - s0)
    m = (cnt + rng.random(missing)).astype(int)
    tw = np.repeat(np.arange(missing), m)
    if len(tw) == 0:
        return empty
    sa = s0[tw] + (length[tw] - s0[tw]) * rng.random(len(tw))
    sag = 0.3 * p.l3_droop
    z = np.array([0.0, 0.0, 1.0])
    pt = base[tw] + d[tw] * sa[:, None] - z * (sag * sa ** 2 / length[tw])[:, None]
    tg = normalize(d[tw] - z * (2.0 * sag * sa / length[tw])[:, None])
    cx, cy = _center_fn(trunk)(pt[:, 2])                                   # 樹冠のエンベロープの外と地面の下は除く
    inside = (np.hypot(pt[:, 0] - cx, pt[:, 1] - cy) <= envelope.reach(p, pt[:, 2]) * 1.05) & (pt[:, 2] > 0.05)
    return pt[inside], tg[inside]


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
        # 幹の上部（樹冠の頂部）。幹は _bearing_axes から外れるため、ここで先端寄りだけに付ける。
        # 幹の中心線上に置くと太い部分では樹皮に埋もれるので、半径ぶん外（表面）へずらす
        live = [a for a in sk.axes if not a.removed]
        s_pts, s_tg = _place([a for a in live if a.level == 0 and not a.dead], per_m * 1.0, 0.82, rng, leaf_frac, theta, 0.25,
                             surface=True)
        above = s_pts[:, 2] >= p.crown_base_ratio * p.height_m          # 幹分かれの下の幹（樹冠より下）には付けない
        extra.append((s_pts[above], s_tg[above]))
        # 足りない細枝の葉は、末端の枝の葉（pts）だけを基準に足す。「葉を付ける枝」で足す内側の枝の葉は、その上にさらに加わる
        v_pts, v_tg = virtual_twig_points(p, sk, rng, per_m, start, leaf_frac, len(pts),
                                          int(cap * VIRTUAL_BOOST ** 2 * p.foliage_amount))
        pts, tg = np.concatenate([pts, v_pts]), np.concatenate([tg, v_tg])
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
            scatter = p.leaf_scatter if p.leaf_scatter >= 0.0 else _AUTO_SCATTER.get(p.leaf_type, 0.0)
            theta_sd = 0.25
            roll_sd = 0.3
            if scatter > 0.0:
                # 枝に沿って並べるだけだと、房が枝の向きにそろって枝先ごとの固まり（芋虫状）になる。
                # 房を枝のまわりの円盤状に散らし、向きも枝からばらけさせて、葉が全体に均一に広がるようにする
                src = leaf_shapes.leaf_source(p)[0]
                size = float(np.ptp(src, axis=0).max()) * p.leaf_scale * boost
                e1, e2 = perp_basis(tg)
                phi = rng.uniform(0.0, 2.0 * math.pi, n)
                rad = scatter * SCATTER_POS * size * np.sqrt(rng.random(n))
                pts = pts + (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2) * rad[:, None]
                theta = theta + scatter * SCATTER_THETA
                theta_sd += scatter * SCATTER_THETA_SD
                roll_sd += scatter * SCATTER_ROLL_SD
            fwd, hint, _ = _orient(tg, rng, theta, theta_sd, 1.0, roll_sd)
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
