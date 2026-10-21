"""ベクトル・回転の小道具（すべて numpy、最後の軸が xyz）。"""
from __future__ import annotations

import numpy as np

Z = np.array([0.0, 0.0, 1.0])
EPS = 1e-12


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS)


def perp_basis(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """単位ベクトル t (...,3) に直交する右手系の基底 (e1, e2)。"""
    ref = np.where(np.abs(t[..., 2:3]) < 0.9, Z, np.array([1.0, 0.0, 0.0]))
    e1 = normalize(np.cross(ref, t))
    e2 = np.cross(t, e1)
    return e1, e2


def rotate_about(v: np.ndarray, axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """ロドリゲスの回転。axis は単位ベクトル、ang は [rad]（ブロードキャスト可）。"""
    ang = np.asarray(ang)[..., None]
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.sum(axis * v, axis=-1, keepdims=True) * (1 - c)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """単位ベクトル a→b の最小回転行列 (3,3)。"""
    a = a / max(np.linalg.norm(a), EPS)
    b = b / max(np.linalg.norm(b), EPS)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-9:  # 逆向き: a に直交する任意軸で 180°
        e1, _ = perp_basis(a)
        return 2 * np.outer(e1, e1) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def mat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """(N,3,3) → (N,3) Blender の XYZ オイラー（R = Rz·Ry·Rx）。"""
    sy = np.clip(-R[:, 2, 0], -1.0, 1.0)
    b = np.arcsin(sy)
    cb = np.cos(b)
    ok = cb > 1e-6
    a = np.where(ok, np.arctan2(R[:, 2, 1], R[:, 2, 2]), np.arctan2(-R[:, 1, 2], R[:, 1, 1]))
    c = np.where(ok, np.arctan2(R[:, 1, 0], R[:, 0, 0]), 0.0)
    return np.stack([a, b, c], axis=-1)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(N,3,3) → (N,4) 四元数 (w,x,y,z)。Blender の QUATERNION 属性の並びに合わせる。"""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    # 4通りの分岐を全要素で計算し、最大の対角成分で選ぶ（Shepperd）。選ばれない分岐の 0 割りは無視する
    k = np.argmax(np.stack([tr, m00, m11, m22], axis=1), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s0 = np.sqrt(np.maximum(tr + 1.0, 0.0)) * 2
        q0 = np.stack([0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0], axis=1)
        s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)) * 2
        q1 = np.stack([(m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1], axis=1)
        s2 = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 0.0)) * 2
        q2 = np.stack([(m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2], axis=1)
        s3 = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 0.0)) * 2
        q3 = np.stack([(m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3], axis=1)
    q = np.choose(k[:, None], [q0, q1, q2, q3])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def frame_matrices(forward: np.ndarray, normal_hint: np.ndarray) -> np.ndarray:
    """葉のローカル軸 (x=幅, y=前, z=法線) → ワールドの回転行列 (N,3,3)。列ベクトルが各軸。"""
    f = normalize(forward)
    n = normal_hint - f * np.sum(normal_hint * f, axis=-1, keepdims=True)
    bad = np.linalg.norm(n, axis=-1) < 1e-6
    if bad.any():
        e1, _ = perp_basis(f[bad])
        n[bad] = e1
    n = normalize(n)
    x = np.cross(f, n)
    return np.stack([x, f, n], axis=-1)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)
