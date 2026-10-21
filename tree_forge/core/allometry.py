"""幹の太さプロファイル、品質テーブルなど寸法まわりの導出。

幹の太さは2通り:
  AUTO   胸高直径 + テーパー指数から決まる曲線
  CUSTOM 「樹高比 t のとき直径 d cm」の点列を PCHIP（単調3次エルミート）で補間
どちらも radius_fn(z) -> 半径[m] として core の他の部分から同じように使える。
"""
from __future__ import annotations

import math

import numpy as np

BREAST_HEIGHT_M = 1.3


def pchip(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson の単調3次エルミート補間。x は狭義単調増加。範囲外は端の値。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xq = np.asarray(xq, float)
    n = len(x)
    if n == 1:
        return np.full_like(xq, y[0])
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(n)
    if n == 2:
        m[:] = d[0]
    else:
        for i in range(1, n - 1):
            if d[i - 1] * d[i] > 0:
                w1 = 2 * h[i] + h[i - 1]
                w2 = h[i] + 2 * h[i - 1]
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
        m[0] = _end_slope(h[0], h[1], d[0], d[1])
        m[-1] = _end_slope(h[-1], h[-2], d[-1], d[-2])
    xc = np.clip(xq, x[0], x[-1])
    i = np.clip(np.searchsorted(x, xc, side="right") - 1, 0, n - 2)
    t = (xc - x[i]) / h[i]
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t)
    h11 = t * t * (t - 1)
    return h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1]


def _end_slope(h0, h1, d0, d1):
    s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if s * d0 <= 0:
        return 0.0
    if d0 * d1 <= 0 and abs(s) > 3 * abs(d0):
        return 3 * d0
    return s


def tip_radius(p) -> float:
    """幹頂の半径[m]。"""
    return max(p.min_radius_mm * 1e-3 * 2.0, 0.004)


def clean_profile(profile) -> list[tuple[float, float]]:
    """点列を高さ比で整列し、同じ位置の点は後勝ち。範囲外は [0,1] にクランプ。"""
    pts = {}
    for t, d in profile:
        t = min(1.0, max(0.0, float(t)))
        pts[round(t, 6)] = max(float(d), 0.1)
    return sorted(pts.items())


def trunk_radius_fn(p):
    """z[m]（地上高）→ 幹の半径[m]（根張りは含まない。根張りはメッシュ化で掛ける）。"""
    H = p.height_m
    tip = tip_radius(p)
    pts = clean_profile(p.profile) if p.trunk_profile_mode == "CUSTOM" else []
    if pts:
        ts = [t for t, _ in pts]
        rs = [d / 200.0 for _, d in pts]
        if ts[0] > 0.0:
            ts.insert(0, 0.0)
            rs.insert(0, rs[0])
        if ts[-1] < 1.0:
            ts.append(1.0)
            rs.append(min(tip, rs[-1]))
        ts_a, rs_a = np.array(ts), np.array(rs)

        def fn(z):
            return np.maximum(pchip(ts_a, rs_a, np.asarray(z, float) / H), tip)

        return fn

    r_dbh = p.dbh_cm / 200.0
    e = p.trunk_taper_exp
    t_dbh = min(BREAST_HEIGHT_M / H, 0.9)

    def fn(z):
        t = np.clip(np.asarray(z, float) / H, 0.0, 1.0)
        return np.maximum(r_dbh * ((1.0 - t) / (1.0 - t_dbh)) ** e, tip)

    return fn


def auto_profile_points(p, n: int = 5) -> list[tuple[float, float]]:
    """AUTO プロファイルを n 点で標本化して (高さ比, 直径cm) にする。CUSTOM の初期値用。"""
    q = type("Q", (), dict(vars(p)))()
    q.trunk_profile_mode = "AUTO"
    fn = trunk_radius_fn(q)
    ts = [0.0] + [round(float(t), 3) for t in np.linspace(0.0, 1.0, n + 1)[1:-1] ** 1.0] + [1.0]
    ts = sorted(set(ts))
    return [(t, round(float(fn(t * p.height_m)) * 200.0, 1)) for t in ts]


def profile_dbh_cm(p) -> float:
    """現在のモードでの胸高(1.3 m)直径 [cm]。"""
    return float(trunk_radius_fn(p)(BREAST_HEIGHT_M)) * 200.0


# (リング頂点数: 幹, L1, L2, L3) / 長さ方向の目標分割長[m] / 本数の上限
_QUALITY = {
    "PREVIEW": dict(max_level=2, ring=(12, 6, 4, 3), seg_len=(0.50, 0.50, 0.25, 0.12), seg_max=(60, 24, 10, 5),
                    l3_cap=0, leaf_cap=20_000, tri_cap=30_000),
    "MEDIUM": dict(max_level=3, ring=(24, 10, 6, 4), seg_len=(0.25, 0.25, 0.12, 0.06), seg_max=(120, 48, 24, 8),
                   l3_cap=2200, leaf_cap=150_000, tri_cap=200_000),
    "HERO": dict(max_level=3, ring=(48, 16, 8, 5), seg_len=(0.125, 0.125, 0.06, 0.03), seg_max=(240, 96, 48, 16),
                 l3_cap=14_000, leaf_cap=1_500_000, tri_cap=1_200_000),
}


# 幹（レベル0）のリング頂点数: 太い木では断面の弧長が一定以下になるまで増やす（樹皮の縦筋を表現するため）
_RING_SPACING = {"PREVIEW": None, "MEDIUM": 0.12, "HERO": 0.03}       # [m]
_RING_CAP = {"PREVIEW": 12, "MEDIUM": 96, "HERO": 256}


def ring_verts(quality: str, level: int, r_max: float = 0.0) -> int:
    """レベル level の軸のリング頂点数。r_max はそのレベルの最大の半径[m]（幹のみ使う）。"""
    base = _QUALITY[quality]["ring"][min(level, 3)]
    spacing = _RING_SPACING[quality]
    if level != 0 or spacing is None:
        return base
    v = int(round(2 * math.pi * r_max / spacing / 4)) * 4
    return max(base, min(_RING_CAP[quality], v))


def quality_table(quality: str) -> dict:
    return _QUALITY[quality]


def n_segments(length: np.ndarray, level: int, quality: str) -> np.ndarray:
    q = _QUALITY[quality]
    lv = min(level, 3)
    n = np.ceil(np.asarray(length, float) / q["seg_len"][lv]).astype(np.int64)
    return np.clip(n, 3, q["seg_max"][lv])


def golden_angle() -> float:
    return math.radians(360.0 * (1.0 - 2.0 / (1.0 + math.sqrt(5.0))))
