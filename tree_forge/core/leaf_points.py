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


SHOOT_SPACING = 0.12     # [m] 側枝（葉の房を付ける小さな枝）の間隔
SHOOT_LEN_RATIO = 1.1    # 側枝の長さ = 葉 1 房の大きさ × この倍率


def _place_shoots(axes, per_m: float, u0: float, rng, frac_scale: float, inst_size: float, surface: bool = False):
    """軸から側枝を出し、その上に葉を置く。位置・接線（側枝の向き）を返す。

    軸の線上に葉を並べると、細い枝が葉の鞘に包まれた毛虫になる。本物の枝は小枝が分かれ、その先に葉が付き、小枝の間に隙間がある。
    軸に沿って SHOOT_SPACING ごとに側枝（長さは葉 1 房の大きさ程度）を出し、1 本の側枝に葉を並べる。
    軸 1 m あたりの葉の数は _place と同じ per_m（側枝 1/SHOOT_SPACING 本 × 1 本あたり per_m × SHOOT_SPACING 枚）。
    """
    P, Tn = [], []
    n_inst = max(1, int(round(per_m * SHOOT_SPACING)))
    shoot_len = float(np.clip(SHOOT_LEN_RATIO * inst_size, 0.05, 0.30))
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
        n_sh = int((L - s0) / SHOOT_SPACING * frac_scale + rng.random())
        if n_sh <= 0:
            continue
        sa = s0 + (L - s0) * (np.arange(n_sh) + rng.random(n_sh)) / n_sh
        idx = np.clip(np.searchsorted(s, sa, side="right") - 1, 0, n_use - 2)
        f = ((sa - s[idx]) / np.maximum(seg[idx], 1e-9))[:, None]
        base = pos[idx] * (1 - f) + pos[idx + 1] * f
        tg = normalize(pos[idx + 1] - pos[idx])
        e1, e2 = perp_basis(tg)
        phi = rng.uniform(0.0, 2.0 * math.pi) + np.arange(n_sh) * allometry.golden_angle() + rng.normal(0.0, 0.3, n_sh)
        alpha = np.clip(rng.normal(0.95, 0.25, n_sh), 0.35, 1.4)
        d = normalize(np.cos(alpha)[:, None] * tg + np.sin(alpha)[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
        ln = shoot_len * rng.uniform(0.7, 1.3, n_sh)
        if surface and a.radius is not None:               # 太い幹では、側枝を樹皮の表面から出す（幹の中心から出すと樹皮に埋もれる）
            radial = normalize(d - np.sum(d * tg, axis=1)[:, None] * tg)
            base = base + radial * (0.9 * np.interp(sa, s, a.radius[:n_use]))[:, None]
        sh = np.repeat(np.arange(n_sh), n_inst)
        w = 0.25 + 0.75 * (np.tile(np.arange(n_inst), n_sh) + rng.random(len(sh))) / n_inst       # 側枝の上の位置（付け根の少し先から先端まで）
        P.append(base[sh] + d[sh] * (ln[sh] * w)[:, None])
        Tn.append(d[sh])
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
_AUTO_SHOOTS = {"BROAD": 1.0, "FAN": 1.0, "SCALE_TWIG": 1.0}                                         # 側枝に葉を付ける割合（針葉は従来どおり枝に沿う）
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
VIRTUAL_SPREAD = 0.22     # 仮想細枝の葉を、軸のまわりに広げる半径（細枝の長さに対する割合）
VIRTUAL_BOOST = 2.3      # 足す葉の上限は「上限枚数 × この倍率² × foliage_amount」。MAX_BOOST までの余地を、葉の量の調整用に残す


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

    def cost(lens, level):                                                  # 軸ごとの三角形数の見積り（長さの配列から一括で）
        return (allometry.n_segments(np.asarray(lens, float), level, "HERO").astype(float) * 2 + 1) * ring[level]

    fixed = 0.0                                                             # 葉の付かない L2 は間引かれない
    for lv in (0, 1, 2):
        ls = [a.length() for a in live if a.level == lv and (lv <= 1 or a.dead)]
        if ls:
            fixed += float(cost(ls, lv).sum())
    lens = np.array([a.length() for a in l2])
    c2 = cost(lens, 2)
    # L3 の平均の長さ。長い L2 ほど L3 も多く長いので、L2 の長さで重みを付ける。0.575 は位置 u に応じた係数 max(.15, 1-.75u) と乱数の平均
    twig_len = float((lens ** 2).sum() / lens.sum()) * p.l3_length_ratio * 0.575
    room = hq["tri_cap"] * 0.97 - fixed
    if room >= c2.sum():                                                   # L2 は全部入る
        c3 = (int(allometry.n_segments(np.array([twig_len]), 3, "HERO")[0]) * 2 + 1) * ring[3]
        expected = p.l3_density / allometry.size_ratio(p) * 0.85 * lens.sum()           # 節あたり per 本で、節の数は密度 / per なので、本数は per に依らない
        return min(expected, hq["l3_cap"], (room - c2.sum()) / c3), twig_len
    return lens.sum() * max(room, 0.0) / c2.sum() / twig_len, twig_len


def virtual_twig_points(p, sk: Skeleton, rng, per_m: float, start: float, leaf_frac: float, n_existing: int,
                        max_points: int, l2_only: bool = False):
    """低い品質（L3 を間引いた・作らない）で、L3 の細枝があるはずの場所に葉の点だけを足す。

    枝は作らない。細い枝（太さ数 mm）は遠目には見えず、葉の量だけが品質で変わってしまうのを防ぐ。
    L2（予算で間引いたものを含む）の途中に、_child_specs と同じ分布（位置 u・角度・長さ）で仮想の細枝を立て、その上に葉の点を置く。
    すでにある葉の点（n_existing）を細枝何本ぶんかに換算して差し引き、足りない分だけ足す。
    l2_only は、L3 が 1 本も無く L2 が末端の枝のとき（プレビュー、予算で L3 が消えた大木）。L2 に沿って葉を並べると枝の向きにそろった
    長い棒（芋虫状）になるので、L2 には葉を並べず、この仮想細枝だけで葉を置く（枝のまわりに広がる房になる）。
    L3 がある HERO は基準なので何も足さない。(位置, 接線) を返す。
    """
    empty = (np.zeros((0, 3)), np.zeros((0, 3)))
    if p.quality == "HERO" and not l2_only:
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
    # 全 L2 の節を 1 本の配列にして、位置の探索を一括でする（軸ごとに numpy を呼ぶと遅いので、弧長も一括で計算する）
    P_list, sizes = [], []
    for a in l2:
        n_use = a.neck + 1 if a.neck >= 0 else a.n
        if n_use < 2:
            continue
        P_list.append(a.pos[:n_use])
        sizes.append(n_use)
    if not P_list:
        return empty
    sizes = np.array(sizes)
    ends = np.cumsum(sizes)
    starts = ends - sizes
    PC = np.concatenate(P_list)
    seg = np.concatenate([[0.0], np.linalg.norm(PC[1:] - PC[:-1], axis=1)])
    seg[starts] = 0.0                                                      # 軸の始まりは弧長 0
    cs = np.cumsum(seg)
    lens = cs[ends - 1] - cs[starts]
    ok = lens >= 0.05
    if not ok.any():
        return empty
    if not ok.all():
        node_ok = np.repeat(ok, sizes)
        PC, cs = PC[node_ok], cs[node_ok]
        sizes, lens = sizes[ok], lens[ok]
        ends = np.cumsum(sizes)
        starts = ends - sizes
        base_cs = cs[starts]
    else:
        base_cs = cs[starts]
    off_axis = np.concatenate([[0.0], np.cumsum(lens + 1.0)[:-1]])          # 軸ごとの区切り（隣の軸と探索がまざらない）
    axis_of_node = np.repeat(np.arange(len(sizes)), sizes)
    G = cs - base_cs[axis_of_node] + off_axis[axis_of_node]
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
    sag = 0.3 * p.l3_droop + 0.9 * max(p.l3_droop - 0.7, 0.0)         # 垂れが強い樹種（シダレヤナギ）の細枝は、ほぼ真下へ下がる
    z = np.array([0.0, 0.0, 1.0])
    pt = base[tw] + d[tw] * sa[:, None] - z * (sag * sa ** 2 / length[tw])[:, None]
    tg = normalize(d[tw] - z * (2.0 * sag * sa / length[tw])[:, None])
    # 細枝の線上に並べると、枝の向きにそろった長い棒になる。本物の細枝は側枝を出して 3 次元の房になるので、軸のまわりへ広げる
    q1, q2 = perp_basis(tg)
    ph = rng.uniform(0.0, 2.0 * math.pi, len(tw))
    rd = VIRTUAL_SPREAD * length[tw] * np.sqrt(rng.random(len(tw)))
    pt = pt + (np.cos(ph)[:, None] * q1 + np.sin(ph)[:, None] * q2) * rd[:, None]
    if l2_only:                                            # L2 が末端なら、その先端 1/4 にも広げた葉を置く（先端が裸の棒にならない）
        kt = (per_m * leaf_frac * 0.25 * lens + rng.random(len(lens))).astype(int)
        pa = np.repeat(np.arange(len(lens)), kt)
        if len(pa):
            key2 = offs[pa] + rng.uniform(0.75, 1.0, len(pa)) * lens[pa]
            k2 = np.clip(np.searchsorted(G, key2, side="right") - 1, starts[pa], ends[pa] - 2)
            f2 = ((key2 - G[k2]) / np.maximum(G[k2 + 1] - G[k2], 1e-9))[:, None]
            tp = PC[k2] * (1 - f2) + PC[k2 + 1] * f2
            tt = normalize(PC[k2 + 1] - PC[k2])
            r1, r2 = perp_basis(tt)
            ph2 = rng.uniform(0.0, 2.0 * math.pi, len(pa))
            rd2 = 0.06 * lens[pa] * np.sqrt(rng.random(len(pa)))
            tp = tp + (np.cos(ph2)[:, None] * r1 + np.sin(ph2)[:, None] * r2) * rd2[:, None]
            pt, tg = np.concatenate([pt, tp]), np.concatenate([tg, tt])
    cx, cy = _center_fn(trunk)(pt[:, 2])                                   # 樹冠のエンベロープの外と地面の下は除く
    inside = (np.hypot(pt[:, 0] - cx, pt[:, 1] - cy) <= envelope.reach(p, pt[:, 2]) * 1.05) & (pt[:, 2] > 0.05)
    return pt[inside], tg[inside]


FLOWER_PER_M = 0.7      # 花の房の数 / 枝 1 m（葉の密度 × この倍率 × 花の密度）。1 房 = 3 輪
CORE_FRAC = 0.5          # 針葉樹の幹まわりの芯: 樹冠の半径に対する幅
CORE_TARGET = 2.0        # 芯の葉の密度を、樹冠全体の平均密度のこの割合まで満たす
CORE_MAX = 0.8          # 足す葉の数の上限（既にある葉の数に対する割合）


def core_fill_points(p, sk: Skeleton, pts: np.ndarray, rng):
    """針葉樹の幹まわり（樹冠の芯）で、枝と枝のあいだに幹が丸見えになるのを、小さな房で埋める。

    本物の針葉樹は枝が密に重なり、幹は外から見えない。ここでは枝の本数が足りない（梢に近いほど枝が短く少ない・大木で枝が減る）ぶん、
    樹冠全体の平均密度（葉の数 ÷ 樹冠の体積）に届くまで、幹から離れた円錐状の芯（半径 = 樹冠の半径 × CORE_FRAC）に房を足す。
    幹の表面には置かない（葉が幹に張り付いた毛虫状にならない）。(位置, 向き) を返す。
    """
    empty = (np.zeros((0, 3)), np.zeros((0, 3)))
    trunk = next((a for a in sk.axes if a.level == 0 and a.parent < 0 and not a.removed), None)
    if trunk is None or trunk.radius is None or len(pts) < 100:
        return empty
    H = p.height_m
    zb = p.crown_base_ratio * H
    zs = np.linspace(zb, 0.985 * H, 49)
    zc, dz = 0.5 * (zs[1:] + zs[:-1]), np.diff(zs)
    R = envelope.reach(p, zc)
    cx, cy = _center_fn(trunk)(pts[:, 2])
    rho = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    inside = pts[:, 2] >= zb
    dens = inside.sum() / max(float((np.pi * R ** 2 * dz).sum()), 1e-9)
    bins = np.clip(np.searchsorted(zs, pts[:, 2]) - 1, 0, len(zc) - 1)
    in_core = inside & (rho <= CORE_FRAC * R[bins])
    have = np.bincount(bins[in_core], minlength=len(zc))
    want = CORE_TARGET * dens * np.pi * (CORE_FRAC * R) ** 2 * dz
    n_fill = np.clip(want - have, 0.0, None)
    total = float(n_fill.sum())
    if total < 1.0:
        return empty
    n_fill *= min(1.0, CORE_MAX * len(pts) / total)
    cnt = (n_fill + rng.random(len(zc))).astype(int)
    b = np.repeat(np.arange(len(zc)), cnt)
    if len(b) == 0:
        return empty
    z = zs[b] + dz[b] * rng.random(len(b))
    zt = np.maximum.accumulate(trunk.pos[:, 2])
    r_trunk = np.interp(z, zt, trunk.radius)
    cx, cy = _center_fn(trunk)(z)
    outer = CORE_FRAC * envelope.reach(p, z)
    r0 = 1.3 * r_trunk
    rad = r0 + np.maximum(outer - r0, 0.0) * np.sqrt(rng.random(len(b)))
    phi = rng.uniform(0.0, 2.0 * math.pi, len(b))
    radial = np.stack([np.cos(phi), np.sin(phi), np.zeros(len(b))], axis=1)
    pos = np.stack([cx, cy, z], axis=1) + radial * rad[:, None]
    tg = normalize(radial + np.array([0.0, 0.0, 0.25]) + rng.normal(0.0, 0.2, (len(b), 3)))
    return pos, tg


def generate(p, sk: Skeleton, st) -> LeafData:
    rng = st.leaves
    flowering = p.flower_density > 0.0
    leaf_frac, flower_frac, color_t = season_state(p.season, p.evergreen, flowering)
    cap = p.leaf_max_instances or allometry.quality_table(p.quality)["leaf_cap"]
    axes = _bearing_axes(sk)
    live_ax = [a for a in sk.axes if not a.removed]
    n_l2 = sum(1 for a in live_ax if a.level == 2 and not a.dead)
    n_l3 = sum(1 for a in live_ax if a.level == 3 and not a.dead)
    # 末端が L2 のとき（プレビュー）や、L3 が L2 の半分も無いとき（予算で L3 が消えた大木）は、L2 に沿って葉を並べると
    # 枝の向きにそろった長い棒になる。L2 には並べず、仮想細枝の房にする
    l2_only = any(a.level == 2 for a in axes) and (max(a.level for a in live_ax) == 2 or n_l3 < 0.5 * n_l2)
    if l2_only:
        axes = [a for a in axes if a.level != 2]

    parts_pos, parts_quat, parts_scale, parts_leaf, parts_var = [], [], [], [], []

    # --- 葉 ---
    if leaf_frac > 0.01 and p.foliage_amount > 0.0:
        theta = {"BROAD": 0.95, "FAN": 0.95, "SCALE_TWIG": 0.7, "NEEDLE_FASCICLE": 0.5, "NEEDLE_SPIRAL": 0.12}[p.leaf_type]
        if p.leaf_type == "BROAD" and p.leaf_width_ratio < 0.25:
            theta = 0.4                                         # 細長い葉（ヤナギ）は枝を横切らず、小枝に沿って並んで垂れる
        per_m = p.leaves_per_m * p.foliage_amount * allometry.size_ratio(p)
        start = p.leaf_start if p.leaf_start >= 0.0 else _AUTO_START.get(p.leaf_type, 0.2)
        done = {a.id for a in axes}
        shoots = p.leaf_shoots if p.leaf_shoots >= 0.0 else _AUTO_SHOOTS.get(p.leaf_type, 0.0)
        inst_size = float(np.ptp(leaf_shapes.leaf_source(p)[0], axis=0).max()) * p.leaf_scale if shoots > 0.0 else 0.0

        def place(axs, pm, u0):
            """枝に沿って並べる葉と、側枝に付ける葉（割合 shoots）に分けて置く。"""
            if shoots <= 0.0:
                return _place(axs, pm, u0, rng, leaf_frac, theta, 0.25)
            b = _place_shoots(axs, pm * shoots, u0, rng, leaf_frac, inst_size)
            if shoots >= 1.0:
                return b
            a = _place(axs, pm * (1.0 - shoots), u0, rng, leaf_frac, theta, 0.25)
            return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])

        pts, tg = place(axes, per_m, start)
        reach = p.leaf_reach
        if reach == "AUTO":
            reach = "OUTER" if p.leaf_type in ("NEEDLE_SPIRAL", "SCALE_TWIG", "NEEDLE_FASCICLE") else "TERMINAL"
        extra = []
        if reach in ("OUTER", "ALL"):                      # 末端の一段手前の枝の外側半分（量感）
            sub1 = _sub_levels(sk, 1, done)
            done |= {a.id for a in sub1}
            extra.append(place(sub1, per_m * 0.5, 0.45))
        if reach == "ALL":                                  # さらに手前の枝の外側 40%
            extra.append(place(_sub_levels(sk, 2, done), per_m * 0.25, 0.6))
        # 幹の上部（樹冠の頂部）。幹は _bearing_axes から外れるため、ここで先端寄りだけに付ける。
        # 幹の中心線上に置くと太い部分では樹皮に埋もれるので、半径ぶん外（表面）へずらす
        live = [a for a in sk.axes if not a.removed]
        forked = {a.parent for a in live if a.level == 0 and a.parent >= 0 and a.parent_node >= sk.axes[a.parent].n - 1}   # 先端で幹が分かれる幹の上端は、分岐部（頂部ではない）。途中から側枝が出る主幹は頂部を持つ
        tops = [a for a in live if a.level == 0 and not a.dead and a.id not in forked]
        u_top = {"NEEDLE_FASCICLE": 0.93, "NEEDLE_SPIRAL": 0.9, "SCALE_TWIG": 0.9}.get(p.leaf_type, 0.82)   # マツの針の房は細い枝先だけ。針葉樹は幹の肌に付けるのを梢の近くだけに（下は芯の房で埋める）
        if shoots > 0.0:
            s_pts, s_tg = _place_shoots(tops, per_m * 1.0 * shoots, u_top, rng, leaf_frac, inst_size, surface=True)
            if shoots < 1.0:
                l_pts, l_tg = _place(tops, per_m * 1.0 * (1.0 - shoots), u_top, rng, leaf_frac, theta, 0.25, surface=True)
                s_pts, s_tg = np.concatenate([s_pts, l_pts]), np.concatenate([s_tg, l_tg])
        else:
            s_pts, s_tg = _place(tops, per_m * 1.0, u_top, rng, leaf_frac, theta, 0.25, surface=True)
        above = s_pts[:, 2] >= p.crown_base_ratio * p.height_m          # 幹分かれの下の幹（樹冠より下）には付けない
        extra.append((s_pts[above], s_tg[above]))
        # 梢: 幹の先端の細い部分は、側枝が無いと葉のない棒として樹冠から突き出す。先端に葉の房を足して覆う
        tip = float(np.clip(0.07 * p.height_m, 0.5, 2.5))
        for a in tops:
            L = a.length()
            if L < 3.0 * tip:
                continue
            pm_tip = max(per_m * 0.8, 25.0 / tip)                                                                 # 先端 1 本あたり 25 房以上（多いと梢だけ濃い筒になる）
            if shoots > 0.0:
                t_pts, t_tg = _place_shoots([a], pm_tip, 1.0 - tip / L, rng, leaf_frac, inst_size)
            else:
                t_pts, t_tg = _place([a], pm_tip, 1.0 - tip / L, rng, leaf_frac, theta, 0.25)
            if len(t_pts):
                e1, e2 = perp_basis(t_tg)
                phi = rng.uniform(0.0, 2.0 * math.pi, len(t_pts))
                # 広がりは樹冠の半径まで（円錐の先端で、円筒のブラシにならない）。先へ行くほど細くなる
                s_up = np.clip((t_pts[:, 2] - (a.pos[-1, 2] - tip)) / tip, 0.0, 1.0)
                rmax = np.minimum(0.35 * tip, 0.9 * envelope.reach(p, t_pts[:, 2])) * (1.0 - 0.6 * s_up)
                rad = rmax * np.sqrt(rng.random(len(t_pts)))
                t_pts = t_pts + (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2) * rad[:, None]
                extra.append((t_pts, t_tg))
        # 足りない細枝の葉は、末端の枝の葉（pts）だけを基準に足す。「葉を付ける枝」で足す内側の枝の葉は、その上にさらに加わる
        v_pts, v_tg = virtual_twig_points(p, sk, rng, per_m, start, leaf_frac, len(pts),
                                          int(cap * VIRTUAL_BOOST ** 2 * p.foliage_amount), l2_only)
        pts, tg = np.concatenate([pts, v_pts]), np.concatenate([tg, v_tg])
        for e_pts, e_tg in extra:
            pts, tg = np.concatenate([pts, e_pts]), np.concatenate([tg, e_tg])
        if p.leaf_type in ("NEEDLE_SPIRAL", "SCALE_TWIG"):      # 針葉樹: 幹まわりの隙間を、幹から離れた房で埋める
            c_pts, c_tg = core_fill_points(p, sk, pts, rng)
            pts, tg = np.concatenate([pts, c_pts]), np.concatenate([tg, c_tg])
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
        # 花の房は、枝の上に間をあけて付ける（1 m あたり FLOWER_PER_M × 葉の密度）。房ごとに枝のまわりの向きをばらす。
        # 側枝の上に置くと、近くで見たとき枝から浮いて見えるので、柄（房の形状に含む）で枝に付ける
        pts, tg = _place(f_axes, p.leaves_per_m * FLOWER_PER_M * p.flower_density * allometry.size_ratio(p), 0.15, rng, flower_frac, 0.8, 0.3)
        n = len(pts)
        if n > cap // 2:
            keep = np.sort(rng.choice(n, cap // 2, replace=False))
            pts, tg = pts[keep], tg[keep]
            n = len(pts)
        if n:
            fwd, hint, radial = _orient(tg, rng, 0.9, 0.3, 0.6, roll_sd=3.0)
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
