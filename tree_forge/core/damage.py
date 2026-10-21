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


def _apply_rigid(sk: Skeleton, cm: dict, root: Axis, R: np.ndarray, old_o: np.ndarray, new_o: np.ndarray) -> None:
    """root とその子孫の全節を、old_o 周りの回転 R のあと new_o へ移す。"""
    stack = [root]
    while stack:
        a = stack.pop()
        a.pos = (a.pos - old_o) @ R.T + new_o
        stack.extend(c for c in cm.get(a.id, []) if not c.removed)


def _mark_dead(cm: dict, root: Axis) -> None:
    stack = [root]
    while stack:
        a = stack.pop()
        a.dead = True
        stack.extend(c for c in cm.get(a.id, []) if not c.removed)


def _hang(sk: Skeleton, ax: Axis, kb: int, cm: dict, rng: np.random.Generator) -> None:
    """節 kb で折れ、先が下へ垂れ下がる。折れ口は繋がったまま。"""
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
    ax.pos[kb:] = new_P
    ax.neck = kb


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
            if hang:
                _hang(sk, ax, kb, cm, rng)
            else:
                sk.truncate(ax, kb + 1, cm)
                ax.broken = True
    sk.compact()
