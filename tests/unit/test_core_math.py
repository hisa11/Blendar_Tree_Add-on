import numpy as np
import pytest

from tree_forge import preset_io, schema
from tree_forge.core import allometry, envelope, leaf_shapes, util


def test_pchip_passes_through_points_and_is_monotone():
    x = np.array([0, 0.1, 0.5, 1.0])
    y = np.array([100, 70, 30, 3.0])
    assert np.allclose(allometry.pchip(x, y, x), y)
    q = np.linspace(0, 1, 400)
    v = allometry.pchip(x, y, q)
    assert np.all(np.diff(v) <= 1e-9)          # 単調減少の点列は補間も単調（オーバーシュートしない）
    assert v.min() >= 3.0 - 1e-9 and v.max() <= 100 + 1e-9


def test_auto_profile_hits_dbh():
    p = schema.make_params(height_m=20, dbh_cm=80)
    fn = allometry.trunk_radius_fn(p)
    assert fn(1.3) * 200 == pytest.approx(80.0, rel=1e-6)
    assert fn(0.0) > fn(1.3) > fn(10.0) > fn(19.0)


def test_custom_profile_gives_requested_diameter_at_heights():
    """「どの高さでどれだけの太さ」を指定できる（ユーザー要件）。"""
    pts = [(0.0, 100), (0.1, 70), (0.5, 30), (1.0, 3)]
    p = schema.make_params(height_m=20, trunk_profile_mode="CUSTOM", profile=pts)
    fn = allometry.trunk_radius_fn(p)
    for t, d in pts:
        assert fn(t * 20) * 200 == pytest.approx(d, rel=1e-6)
    assert allometry.profile_dbh_cm(p) == pytest.approx(float(fn(1.3)) * 200)


def test_custom_profile_unsorted_and_partial_ranges():
    p = schema.make_params(height_m=10, trunk_profile_mode="CUSTOM", profile=[(0.5, 20), (0.2, 40)])
    fn = allometry.trunk_radius_fn(p)
    assert fn(2.0) * 200 == pytest.approx(40)                # 0.2 → 40 cm
    assert fn(0.0) * 200 == pytest.approx(40)                # 下は最初の点を延長
    assert fn(10.0) * 200 < 20                               # 上は先端へ細る
    # 空の点列は AUTO と同じ
    p2 = schema.make_params(height_m=10, trunk_profile_mode="CUSTOM", profile=[])
    assert allometry.trunk_radius_fn(p2)(1.3) * 200 == pytest.approx(p2.dbh_cm)


def test_auto_profile_points_reproduce_curve():
    p = schema.make_params()
    pts = allometry.auto_profile_points(p, 5)
    assert pts[0][0] == 0.0 and pts[-1][0] == 1.0 and len(pts) == 6
    q = schema.make_params(trunk_profile_mode="CUSTOM", profile=pts)
    a, b = allometry.trunk_radius_fn(p), allometry.trunk_radius_fn(q)
    z = np.linspace(0, 20, 50)
    assert np.allclose(a(z), b(z), rtol=0.06, atol=0.01)   # 6点なら元の曲線に近い（先端の細い部分は絶対誤差 1 cm）


@pytest.mark.parametrize("shape", schema.CROWN_SHAPES)
def test_envelope_shapes_normalised(shape):
    t = np.linspace(0, 1, 513)
    v = envelope.shape_value(shape, t)
    assert v.max() == pytest.approx(1.0, abs=1e-9) and v.min() >= 0
    assert v[-1] < 0.05                                      # 樹頂で閉じる


def test_envelope_reach_ground_and_top():
    p = schema.make_params(height_m=20, crown_width_m=10, crown_base_ratio=0.25, crown_shape="spherical")
    assert envelope.reach(p, np.array(0.0)) == 0.0
    assert envelope.reach(p, np.array(21.0)) == 0.0
    assert envelope.reach(p, np.array(12.5)) == pytest.approx(5.0, rel=0.02)


def test_quaternion_and_euler_roundtrip():
    rng = np.random.default_rng(3)
    fwd = rng.normal(size=(200, 3))
    hint = rng.normal(size=(200, 3))
    R = util.frame_matrices(fwd, hint)
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3), atol=1e-9)        # 直交
    assert np.allclose(np.linalg.det(R), 1.0)
    q = util.mat_to_quat(R)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    w, x, y, z = q.T
    R2 = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], axis=1)
    assert np.allclose(R2, R, atol=1e-9)
    e = util.mat_to_euler_xyz(R)
    a, b, c = e.T
    Rx = np.stack([np.stack([np.ones_like(a), 0 * a, 0 * a], -1), np.stack([0 * a, np.cos(a), -np.sin(a)], -1),
                   np.stack([0 * a, np.sin(a), np.cos(a)], -1)], 1)
    Ry = np.stack([np.stack([np.cos(b), 0 * b, np.sin(b)], -1), np.stack([0 * b, np.ones_like(b), 0 * b], -1),
                   np.stack([-np.sin(b), 0 * b, np.cos(b)], -1)], 1)
    Rz = np.stack([np.stack([np.cos(c), -np.sin(c), 0 * c], -1), np.stack([np.sin(c), np.cos(c), 0 * c], -1),
                   np.stack([0 * c, 0 * c, np.ones_like(c)], -1)], 1)
    assert np.allclose(Rz @ Ry @ Rx, R, atol=1e-9)


def test_rotation_between():
    a = np.array([0.0, 0.0, 1.0])
    for b in (np.array([1.0, 0, 0]), np.array([0, 0, -1.0]), np.array([0.3, -0.5, 0.2])):
        b = b / np.linalg.norm(b)
        assert np.allclose(util.rotation_between(a, b) @ a, b, atol=1e-9)


@pytest.mark.parametrize("leaf_type", schema.LEAF_TYPES)
@pytest.mark.parametrize("cluster", [1, 5])
def test_leaf_sources_valid(leaf_type, cluster):
    p = schema.make_params(leaf_type=leaf_type, leaf_cluster=cluster, leaf_length_cm=5)
    v, t, uv = leaf_shapes.leaf_source(p)
    assert len(v) == len(uv) and t.max() < len(v) and t.min() >= 0
    tri = v[t]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    assert (area > 0).all(), "退化三角形がない"
    assert np.isfinite(v).all() and np.isfinite(uv).all()
    assert np.abs(v).max() < 0.5                                         # 単位は m、5 cm の葉が数十 cm 以内


def test_flower_source_valid():
    v, t, uv = leaf_shapes.flower_source(schema.make_params())
    assert t.max() < len(v) and len(t) == 40


def test_ginkgo_leaf_has_a_petiole_and_a_notched_fan_blade():
    """イチョウの葉: 付け根の細い葉柄、くさび形の付け根から広がる扇形の葉身、外縁の中央の浅い切れ込み（パックマン形ではない）。"""
    R = 0.08
    v, t, uv = leaf_shapes.fan(R, 0.5, 11)
    assert t.min() >= 0 and t.max() < len(v) and len(uv) == len(v)
    r = np.hypot(v[:, 0], v[:, 1])
    assert r.max() <= 1.05 * R
    assert np.ptp(v[r < 0.1 * R][:, 0]) < 0.01 * R                            # 付け根は 1 点（葉柄の先は細い）
    blade = v[r > 0.8 * R]
    assert np.ptp(blade[:, 0]) > 1.0 * R                                      # 葉身は幅広の扇形（半角 42° の外縁）
    outer = v[len(v) - 11:]
    assert outer[5, 1] < 0.9 * outer[:, 1].max() or r[len(v) - 11 + 5] < 0.95 * r[len(v) - 11:].max()   # 中央が浅く凹む
    inner = v[1:12]
    assert np.hypot(inner[:, 0], inner[:, 1]).max() < 0.5 * R                 # 葉身の付け根はくさび形（半径 0.42R）


def test_broad_clusters_are_staggered_on_a_stem_except_round_maple_leaves():
    """サクラ・ケヤキは 1 点から放射状に開くとカエデの掌状葉に見える。軸に沿って左右交互に付ける。幅広の丸い葉（カエデ）は従来の並び。"""
    n = 4
    single = len(leaf_shapes.broad(0.1, 0.5, n_rows=5)[1])
    cherry = schema.make_params(**preset_io.builtin_overrides("someiyoshino"))
    maple = schema.make_params(**preset_io.builtin_overrides("iroha_momiji"))
    v, t, _ = leaf_shapes.leaf_source(cherry)
    assert len(t) == cherry.leaf_cluster * single + 2                         # 葉 + 軸の帯（2 三角形）
    assert np.ptp(v[:, 1]) > 1.8 * cherry.leaf_length_cm * 0.01              # 軸に沿って間をあけて付く
    vm, tm, _ = leaf_shapes.leaf_source(maple)
    assert len(tm) == maple.leaf_cluster * len(leaf_shapes.broad(0.07, 1.0, n_rows=5)[1])
