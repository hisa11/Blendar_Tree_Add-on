"""損傷: 折れ枝・垂れ下がり折れ・頂部折れ。

半径は「折れる前の健全な枝」として先に決めてあるので、折れ口は元の太さのまま残る
（途中で細くなって尖るのではなく、断ち切られた断面になる）。
"""
from __future__ import annotations

import numpy as np

from .skeleton import Axis, Skeleton
from .util import Z, normalize, perp_basis, rotation_between

LEVEL_WEIGHT = {1: 1.0, 2: 0.35, 3: 0.1}      # 太い枝ほど折れ口が目立つ（細い枝は「折れ」より「枯れ」）
SAG_PER_NODE = 0.12      # 垂れ下がり部で、節ごとに下向きへ寄せる量
GROUND_Z = 0.03          # 垂れ下がった枝は地面（z=0）に当たったら止まる。これより下へは行かない


def _apply_rigid(sk: Skeleton, cm: dict, root: Axis, R: np.ndarray, old_o: np.ndarray, new_o: np.ndarray) -> None:
    """root とその子孫の全節を、old_o 周りの回転 R のあと new_o へ移す。"""
    stack = [root]
    while stack:
        a = stack.pop()
        a.pos = (a.pos - old_o) @ R.T + new_o
        a.pos[:, 2] = np.maximum(a.pos[:, 2], GROUND_Z)
        stack.extend(c for c in cm.get(a.id, []) if not c.removed)


def _mark_dead(cm: dict, root: Axis) -> None:
    stack = [root]
    while stack:
        a = stack.pop()
        a.dead = True
        stack.extend(c for c in cm.get(a.id, []) if not c.removed)


HANG_MAX_FRAC = 0.2      # 垂れ下がる部分の長さの上限（樹高比）。これより先は落ちて無くなる（糸のように地面まで届かない）
HANG_MIN_RADIUS = 0.012  # 折れ口の半径 [m]。これより細い枝は垂れ下がらせず、その場で折る（細い針金が宙に垂れた形になる）


def _hang(sk: Skeleton, ax: Axis, kb: int, cm: dict, rng: np.random.Generator, max_len: float = float("inf")) -> None:
    """節 kb で折れ、先が下へ垂れ下がる。折れ口は繋がったまま。垂れ下がる部分は max_len [m] まで。"""
    cum = np.cumsum(np.linalg.norm(np.diff(ax.pos[kb:], axis=0), axis=1))
    keep = max(2, int(np.searchsorted(cum, max_len, side="right")))
    if kb + 1 + keep < ax.n:
        sk.truncate(ax, kb + 1 + keep, cm)
    P = ax.pos.copy()
    n = len(P)
    d_break = normalize(P[kb + 1] - P[kb])
    horiz = d_break.copy()
    horiz[2] = 0.0
    horiz = horiz / max(np.linalg.norm(horiz), 1e-9) if np.linalg.norm(horiz) > 1e-6 else np.array([1.0, 0.0, 0.0])
    e1, _ = perp_basis(horiz)
    tgt = normalize(0.35 * horiz - 0.95 * Z + rng.normal(0.0, 0.12) * e1)
    R0 = rotation_between(d_break, tgt)

    old_dirs = np.diff(P[kb:], axis=0)                      # (n-1-kb, 3)
    seg_len = np.linalg.norm(old_dirs, axis=1)
    new_dirs = normalize(old_dirs @ R0.T)
    sag = SAG_PER_NODE * (np.arange(len(new_dirs)) + 1)
    new_dirs = normalize(new_dirs - sag[:, None] * Z)
    new_P = np.empty_like(P[kb:])
    new_P[0] = P[kb]
    new_P[1:] = P[kb] + np.cumsum(new_dirs * seg_len[:, None], axis=0)

    # 子孫を節ごとの剛体変換で追従させる
    old_t = np.vstack([normalize(P[min(k + 1, n - 1)] - P[k - 1]) for k in range(kb + 1, n)])
    new_t = np.vstack([normalize(new_P[min(k - kb + 1, n - 1 - kb)] - new_P[k - kb - 1]) for k in range(kb + 1, n)])
    for c in cm.get(ax.id, []):
        if c.removed or c.parent_node <= kb:
            continue
        i = c.parent_node - (kb + 1)
        R = rotation_between(old_t[i], new_t[i])
        _apply_rigid(sk, cm, c, R, P[c.parent_node], new_P[c.parent_node - kb])
        _mark_dead(cm, c)
    new_P[:, 2] = np.maximum(new_P[:, 2], GROUND_Z)
    ax.pos[kb:] = new_P
    ax.neck = kb


DEAD_FLOOR_FRAC = 0.08     # 枯れた細い枝（L2・L3）が垂れ下がってよい下限の高さ（樹高比）。これより下は切り落とす（糸が地面まで届かない）
DEAD_FLOOR_MIN = 1.0       # [m]


def trim_dead_near_ground(p, sk: Skeleton, cm: dict) -> None:
    """樹冠から垂れ下がった枯れ枝（L2 以降）が、細い針金のように地面まで届いていた。地面から樹高の 8%（最低 1 m）より下は無くす。

    枝垂れの樹冠は、枝先が地面近くまで垂れるのが本来の姿なので対象外。乱数は引かない。
    """
    if p.crown_shape == "weeping":
        return
    floor = max(DEAD_FLOOR_MIN, DEAD_FLOOR_FRAC * p.height_m)
    for ax in list(sk.axes):
        if ax.removed or ax.level < 2 or not ax.dead:
            continue
        low = np.nonzero(ax.pos[:, 2] < floor)[0]
        if len(low) == 0:
            continue
        k = int(low[0])
        if k < 2:
            sk.remove_subtree(ax.id, cm)
        else:
            sk.truncate(ax, k, cm)


STUB_PROTRUDE = (0.3, 1.5)      # 幹の表面からスタブが突き出す長さ [m]
STUB_RADIUS_FRAC = (0.05, 0.16)  # スタブ付け根の半径 / その高さの幹の半径
STUB_Z_MIN = 1.2                 # これより低い所には付けない（根張りの高さのほうが高ければそちらまで）


def add_trunk_stubs(p, sk: Skeleton, rng: np.random.Generator) -> None:
    """樹冠より下の裸の幹に、枯れて折れた枝の根元（スタブ）を足す。

    健全な骨格・半径・他の損傷の乱数には触れない（apply_damage の最後に、同じ rng の続きで引く）。
    ただし三角形の予算は共有するので、予算で間引かれる木ではスタブの分だけ細い枝が少し減る。
    また枝の本数が変わると、メッシャーの樹皮ノイズの乱数の割り当てが変わる（形・太さ・葉は変わらない）。
    根張りの高さまでは付けない（根張りで幹の実際の半径が大きくなり、スタブが埋まるため）。
    """
    dens = p.trunk_stub_density
    hb = p.crown_base_ratio * p.height_m
    z_lo = max(STUB_Z_MIN, p.root_flare_height * p.height_m)
    if not p.trunk_stubs or dens <= 0.0 or hb <= z_lo + 1.0:
        return
    for trunk in [a for a in sk.axes if a.level == 0 and a.parent < 0 and not a.removed and a.radius is not None]:
        z_hi = min(hb * 0.97, float(trunk.pos[-1, 2]))
        nn = int(dens * (z_hi - z_lo) + rng.random())
        if nn <= 0:
            continue
        zs = np.sort(rng.uniform(z_lo, z_hi, nn))
        phis = rng.uniform(0.0, 2.0 * np.pi) + np.arange(nn) * 2.399963 + rng.normal(0.0, 0.5, nn)
        for z, phi in zip(zs, phis):
            k = int(np.clip(np.searchsorted(trunk.pos[:, 2], z), 1, trunk.n - 2))
            r_t = float(trunk.radius[k])
            t = normalize(trunk.pos[k + 1] - trunk.pos[k - 1])
            e1, e2 = perp_basis(t[None, :])
            radial = np.cos(phi) * e1[0] + np.sin(phi) * e2[0]
            a = np.radians(rng.uniform(78.0, 105.0))              # 幹の軸からの角度（ほぼ水平、やや垂れる）
            d = normalize(np.cos(a) * t + np.sin(a) * radial)
            length = r_t + rng.uniform(*STUB_PROTRUDE)
            m = max(3, int(length / 0.25))
            s = np.linspace(0.0, length, m + 1)
            pos = trunk.pos[k] + d[None, :] * s[:, None]
            pos[:, 2] -= 0.10 * (s / length) ** 2 * length * 0.5      # 先端ほどわずかに垂れる
            r0 = r_t * rng.uniform(*STUB_RADIUS_FRAC)
            radius = r0 * (1.0 - 0.25 * s / length)                  # 折れ口は健全時の太さのまま残る
            horiz = radial.copy()
            horiz[2] = 0.0
            ax = sk.add(1, trunk.id, k, pos, normalize(horiz), dead=True, phase=float(rng.random()))
            ax.radius = np.maximum(radius, p.min_radius_mm * 1e-3)
            ax.broken = True


def apply_damage(p, sk: Skeleton, st) -> None:
    rng = st.damage
    cm = sk.children_map()

    # 頂部折れ（単幹のときだけ）
    if p.top_break_height > 0.0 and p.trunk_split_count < 2:
        trunk = sk.axes[0]
        k = int(np.searchsorted(trunk.pos[:, 2], p.top_break_height * p.height_m))
        if 3 <= k < trunk.n:
            sk.truncate(trunk, k + 1, cm)
            trunk.broken = True

    base = p.broken_branch_ratio * p.age_factor
    if base > 0.0:
        for ax in list(sk.axes):
            if ax.removed or ax.level == 0 or ax.n < 4:
                continue
            if rng.random() >= base * LEVEL_WEIGHT.get(ax.level, 0.3):
                continue
            b = rng.uniform(p.broken_pos_min, max(p.broken_pos_max, p.broken_pos_min))
            kb = int(np.clip(round(b * (ax.n - 1)), 2, ax.n - 2))
            hang = ax.level <= 2 and rng.random() < p.broken_hang_ratio
            if hang and ax.radius is not None and ax.radius[kb] < HANG_MIN_RADIUS:
                hang = False
            if hang:
                _hang(sk, ax, kb, cm, rng, HANG_MAX_FRAC * p.height_m)
            else:
                sk.truncate(ax, kb + 1, cm)
                ax.broken = True
    trim_dead_near_ground(p, sk, cm)
    add_trunk_stubs(p, sk, rng)
    sk.compact()
