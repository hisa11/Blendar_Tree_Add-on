"""断面の形。正円にせず、高さ（弧長）ごとに少しずつ歪ませる。

断面の半径は  r(φ, s) = r_nominal · F(φ, s)  で、F は平均がほぼ 1 になる係数。
  * 調和項 k=2..6: 振幅は軸に沿って滑らかに増減し、位相はゆっくり回る
        → 楕円っぽい断面の向きが高さとともにねじれ、ところどころ角張る
  * 偏心: 断面の中心が軸から少しずれる
  * 根張り・こぶ・樹皮の縦筋: 幹のみ
どれも乱数ストリーム(bark)から決まるので seed で再現できる。
"""
from __future__ import annotations

import math

import numpy as np

from .rng import smooth_sample, smooth_sample_1d

HARMONICS = (2, 3, 4, 5, 6)
HARMONIC_W = (1.6, 1.0, 0.7, 0.5, 0.35)     # 2次(楕円)を強める
AMP0 = 0.18                                  # irregularity=1 のときの基準振幅
ECCENTRIC = 0.12                             # 偏心の最大量（半径比 × irregularity）
TWIST_SD = 0.35                              # 位相の回転 [rad/m] の標準偏差


def section_factors(rng, p, *, s, z, r, L, level, is_root, phi, irr, valid):
    """断面係数と中心のずれを返す。

    s,z,r,valid: (A,M)  L,irr,level,is_root: (A,)  phi: (V,)
    返り値: F (A,M,V), off_uv (A,M,2)  ── off_uv は断面フレームの (u,v) 方向のずれ量[m]
    """
    A, M = s.shape
    V = len(phi)
    sf = np.clip(s / np.maximum(L, 1e-9)[:, None], 0.0, 1.0)
    F = np.ones((A, M, V))
    age = p.age_factor

    # --- 調和項 ---
    for k, w in zip(HARMONICS, HARMONIC_W):
        if k > V // 2:
            continue
        amp = AMP0 * w * irr * rng.uniform(0.6, 1.2, A)                       # (A,)
        mod = 0.35 + 0.65 * (0.5 + 0.5 * smooth_sample(rng.uniform(-1, 1, (A, 6))[:, None, :], sf))
        th0 = rng.uniform(0.0, 2 * math.pi, A)
        tw = rng.normal(0.0, TWIST_SD, A)
        theta = th0[:, None] + tw[:, None] * s                               # (A,M)
        F += (amp[:, None] * mod)[:, :, None] * np.cos(k * phi[None, None, :] - theta[:, :, None])

    # --- 偏心 ---
    n1 = smooth_sample(rng.uniform(-1, 1, (A, 6))[:, None, :], sf)
    n2 = smooth_sample(rng.uniform(-1, 1, (A, 6))[:, None, :], sf)
    off = ECCENTRIC * irr[:, None, None] * r[:, :, None] * np.stack([n1, n2], axis=-1)

    # --- 幹だけの特徴 ---
    root_idx = np.nonzero(is_root)[0]
    for a in root_idx:
        _trunk_features(rng, p, F[a], s[a], z[a], r[a], float(L[a]), phi, age)

    return np.clip(F, 0.5, 1.9), off


def _trunk_features(rng, p, F, s, z, r, L, phi, age):
    """F (M,V) を直接更新する。根張り・こぶ・縦筋。"""
    H = p.height_m
    V = len(phi)
    # 根張り
    h_f = p.root_flare_height * H
    if h_f > 1e-6 and p.root_flare_scale > 1.0:
        w = np.clip(1.0 - z / h_f, 0.0, 1.0)[:, None]                          # (M,1)
        extra = (p.root_flare_scale - 1.0) * (0.6 + 0.4 * age)
        F *= 1.0 + 0.6 * extra * w ** 2
        if p.root_count > 0:
            ph_r = rng.uniform(0.0, 2 * math.pi)
            lobes = np.cos(0.5 * p.root_count * (phi - ph_r)) ** 2 - 0.5
            F *= 1.0 + extra * w ** 1.5 * lobes[None, :]
    # こぶ
    nb = int(rng.poisson(p.burl_density * L * 0.6 * (0.5 + 0.5 * age)))
    if nb > 0:
        sb = rng.uniform(0.15 * L, 0.9 * L, nb)
        pb = rng.uniform(0.0, 2 * math.pi, nb)
        ss = rng.uniform(0.15, 0.45, nb)
        amp = rng.uniform(0.08, 0.22, nb) * (0.6 + 0.4 * age)
        for i in range(nb):
            dphi = (phi[None, :] - pb[i] + math.pi) % (2 * math.pi) - math.pi
            sig_phi = np.clip(ss[i] / np.maximum(r, 1e-3), 0.15, 0.9)[:, None]
            F += amp[i] * np.exp(-(((s - sb[i]) / ss[i]) ** 2)[:, None] - (dphi / sig_phi) ** 2)
    # 縦筋（リングの分割が十分なときだけ）
    if p.bark_ridge_mm > 0.0 and V >= 12:
        rel = np.clip(p.bark_ridge_mm * 1e-3 / np.maximum(r, 1e-3), 0.0, 0.25)[:, None]
        k1, k2 = max(3, V // 3), max(2, V // 4)
        sf = np.clip(s / max(L, 1e-9), 0.0, 1.0)
        psi1 = rng.uniform(0, 2 * math.pi) + 2.5 * smooth_sample_1d(rng.uniform(-1, 1, 8), sf)
        psi2 = rng.uniform(0, 2 * math.pi) + 2.5 * smooth_sample_1d(rng.uniform(-1, 1, 8), sf)
        ridge = 0.6 * np.sin(k1 * phi[None, :] + psi1[:, None]) + 0.4 * np.sin(k2 * phi[None, :] + psi2[:, None])
        F += rel * ridge
