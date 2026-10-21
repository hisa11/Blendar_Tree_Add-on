"""乱数ストリームと滑らかなノイズ。

サブシステムごとにストリームを分け、葉の密度を変えても枝の形が変わらないようにする。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def make_streams(seed: int) -> SimpleNamespace:
    def g(offset: int) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(int(seed) + offset))

    return SimpleNamespace(skel=g(0), leaves=g(1000), bark=g(2000), damage=g(3000))


def smooth_sample(table: np.ndarray, u: np.ndarray) -> np.ndarray:
    """table (...,T) を u∈[0,1] (...,) で滑らかに補間する（smoothstep 補間の値ノイズ）。"""
    T = table.shape[-1]
    x = np.clip(u, 0.0, 1.0) * (T - 1)
    i = np.minimum(np.floor(x).astype(np.int64), T - 2)
    f = x - i
    s = f * f * (3 - 2 * f)
    a = np.take_along_axis(table, i[..., None], axis=-1)[..., 0]
    b = np.take_along_axis(table, (i + 1)[..., None], axis=-1)[..., 0]
    return a + (b - a) * s


def smooth_sample_1d(table: np.ndarray, u: np.ndarray) -> np.ndarray:
    """1次元の table (T,) を u (M,) ∈ [0,1] で滑らかに補間する。"""
    T = len(table)
    x = np.clip(u, 0.0, 1.0) * (T - 1)
    i = np.minimum(np.floor(x).astype(np.int64), T - 2)
    f = x - i
    s = f * f * (3 - 2 * f)
    return table[i] + (table[i + 1] - table[i]) * s
