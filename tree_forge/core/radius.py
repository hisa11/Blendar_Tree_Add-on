"""半径の決定（パイプモデル + レオナルドの規則）。

各節に「その先が支える量」q を割り当てる。
    q(節 i) = q_tip + λ·(先端までの弧長) + Σ(i 以降の節から出る子の付け根の q)
λ は「枝分かれしない軸が先細りする量」(taper) から決まる暗黙の負荷。
半径は q^(1/Δ) に比例する（Δ=レオナルド指数）。つまり親の節で
    r_親^Δ ≈ r_親(先)^Δ + Σ r_子^Δ
が成り立つ。幹だけは太さプロファイル（AUTO/CUSTOM）で絶対値を与え、枝はその比で決める。
"""
from __future__ import annotations

import numpy as np

from . import allometry
from .skeleton import Skeleton

COLLAR = 0.35        # 枝の付け根の膨らみ（半径に対する最大の割増）
COLLAR_LEN = 1.5     # 膨らみの減衰長（付け根半径に対する倍率）


def _level_taper(p, level: int) -> float:
    if level == 0:
        return 0.10
    return getattr(p, f"l{level}_taper")


def compute_radii(p, sk: Skeleton) -> None:
    """Axis.radius / Axis.q を埋める。"""
    delta = p.leonardo_exponent
    inv = 1.0 / delta
    cm = sk.children_map()
    r_min = p.min_radius_mm * 1e-3
    trunk_fn = allometry.trunk_radius_fn(p)

    # --- 下から上（先端から根元）へ q を集計。子は親より後ろにあるので逆順で処理する ---
    for ax in reversed(sk.axes):
        if ax.removed:
            continue
        s = ax.arc()
        L = max(s[-1], 1e-9)
        taper = max(_level_taper(p, ax.level), 1e-3)
        lam = ((1.0 / taper) ** delta - 1.0) / L
        c = np.zeros(ax.n)
        for ch in cm.get(ax.id, []):
            c[min(ch.parent_node, ax.n - 1)] += ch.q[0]
        ax.q = 1.0 + lam * (L - s) + np.cumsum(c[::-1])[::-1]

    # --- 上から下（根元から先端）へ半径を決める ---
    for ax in sk.axes:
        if ax.removed:
            continue
        s = ax.arc()
        L = max(s[-1], 1e-9)
        if ax.parent < 0:
            r = trunk_fn(ax.pos[:, 2])
        else:
            par = sk.axes[ax.parent]
            r_par = par.radius[ax.parent_node]
            mode = getattr(p, f"l{ax.level}_radius_mode") if ax.level >= 1 else "LEONARDO"
            if mode == "RATIO":
                r0 = r_par * getattr(p, f"l{ax.level}_radius_ratio")
                r = r0 * (1.0 - (1.0 - max(_level_taper(p, ax.level), 0.02)) * s / L)
            else:
                r0 = r_par * (ax.q[0] / par.q[ax.parent_node]) ** inv
                r = r0 * (ax.q / ax.q[0]) ** inv
            # 子は親より太くならない
            r = np.minimum(r, 0.92 * r_par)
        ax.radius = np.maximum(r, r_min)


def collar_factor(s: np.ndarray, r0) -> np.ndarray:
    """付け根の膨らみ係数 (>=1)。s は軸に沿った弧長、r0 は付け根半径（スカラーまたは s にブロードキャスト可）。"""
    return 1.0 + COLLAR * np.exp(-s / (COLLAR_LEN * np.maximum(r0, 1e-4)))
