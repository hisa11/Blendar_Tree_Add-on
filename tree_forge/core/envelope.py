"""樹冠エンベロープ。t=0（枝下）〜1（樹頂）での相対半径 e(t)∈[0,1]。"""
from __future__ import annotations

import numpy as np

_N = 257
MIN_REACH = 0.30        # [m] 樹冠内の最小到達半径。円錐の先端でも短い枝が残る


def _raw(shape: str, t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    if shape == "conical":
        return (1.0 - t) ** 0.95 * (0.96 + 0.04 * t) + 0.03
    if shape == "columnar":
        top = np.sqrt(np.clip(1 - ((t - 0.82) / 0.18) ** 2, 0, 1))
        return np.where(t < 0.82, 1.0, top) * (0.55 + 0.45 * np.minimum(1.0, t / 0.08))
    if shape == "ovoid":
        return np.sin(np.pi * np.clip(t, 1e-6, 1) ** 0.72) ** 0.85 + 0.10
    if shape == "spherical":
        return np.sqrt(np.clip(1 - (2 * t - 1) ** 2, 0, 1)) * 0.92 + 0.12
    if shape == "vase":  # 根元で細く、上部で広がる
        return (0.22 + 0.78 * t ** 0.75) * np.sqrt(np.clip(1 - t ** 5, 0, 1))
    if shape == "umbrella":  # 低い枝は短く、上部が平たく広い
        return (0.30 + 0.70 * t ** 0.55) * np.sqrt(np.clip(1 - t ** 9, 0, 1))
    if shape == "weeping":  # 低い位置まで広く、枝が垂れ下がる。頂部は丸いドーム（細いと、幹の先が裸で突き出る）
        return (np.sin(np.pi * np.clip(t, 1e-6, 1) ** 0.75) ** 0.55) * 0.9 + 0.2 * (1 - t)
    if shape == "irregular":
        base = np.sqrt(np.clip(1 - (2 * t - 1) ** 2, 0, 1)) + 0.12
        return base * (0.78 + 0.22 * np.sin(5.3 * t + 1.1) * np.cos(2.1 * t))
    raise ValueError(f"unknown crown shape: {shape}")


_TABLES: dict[str, np.ndarray] = {}
_XS = np.linspace(0.0, 1.0, _N)       # 表の横軸（毎回作ると枝ごとの呼び出しで遅い）


def _table(shape: str) -> np.ndarray:
    tb = _TABLES.get(shape)
    if tb is None:
        raw = _raw(shape, np.linspace(0.0, 1.0, _N))
        raw[-1] = 0.0 if shape not in ("conical",) else raw[-1] * 0.2
        tb = raw / raw.max()
        _TABLES[shape] = tb
    return tb


def shape_value(shape: str, t: np.ndarray) -> np.ndarray:
    """t∈[0,1] での正規化エンベロープ（最大値 1）。t は範囲外ならクランプ。"""
    tb = _table(shape)
    return np.interp(np.clip(t, 0.0, 1.0), _XS, tb)


def reach(p, z: np.ndarray) -> np.ndarray:
    """高さ z [m] における樹冠の最大半径 [m]。枝下は地面に向かって線形に 0 へ。"""
    z = np.asarray(z, dtype=float)
    H = p.height_m
    hb = p.crown_base_ratio * H
    half = 0.5 * p.crown_width_m
    t = (z - hb) / max(H - hb, 1e-6)
    e = shape_value(p.crown_shape, t)
    below = np.where(z < hb, shape_value(p.crown_shape, np.array(0.0)) * np.clip(z / max(hb, 1e-6), 0.0, 1.0), 0.0)
    out = np.where(z < hb, below, e)
    r = half * out
    inside_crown = (z >= hb) & (z <= 0.985 * H)
    r = np.where(inside_crown, np.maximum(r, MIN_REACH), r)
    return np.where((z > H) | (z < 0), 0.0, r)
