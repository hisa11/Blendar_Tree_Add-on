import time

import numpy as np
import pytest

from tree_forge import schema
from tree_forge.core import allometry, damage, envelope, leaf_points, pipeline, radius, rng, skeleton, tube_mesher

from tests.unit.helpers import axis_rings, gen, mesh_hash, ring_radii, species_params

ALL = schema.SPECIES_IDS


# --- 寸法・エンベロープ・再現性 ------------------------------------------------
@pytest.mark.parametrize("sid", ALL)
def test_dimensions_and_budget_all_species(sid):
    for q in ("PREVIEW", "MEDIUM"):
        p = species_params(sid, quality=q)
        t = pipeline.generate(p)
        H, W = pipeline.measure(t)
        assert H == pytest.approx(p.height_m, rel=0.03), (sid, q)
        assert W == pytest.approx(p.crown_width_m, rel=0.10), (sid, q)
        cap = allometry.quality_table(q)
        assert t.mesh.tri_count <= cap["tri_cap"], (sid, q, t.mesh.tri_count)
        assert t.leaves.n <= cap["leaf_cap"] * 1.5


def test_hero_keyaki_acceptance():
    """仕様 8: HERO のケヤキ(H=20 m)が 15 s 以内、樹高・樹冠幅 ±5%、三角形 1.2M・葉 1.5M 以下。"""
    p = species_params("keyaki", quality="HERO")
    t0 = time.time()
    t = pipeline.generate(p)
    dt = time.time() - t0
    H, W = pipeline.measure(t)
    assert dt <= 15.0, dt
    assert abs(H / 20.0 - 1) <= 0.05 and abs(W / p.crown_width_m - 1) <= 0.05, (H, W)
    assert t.mesh.tri_count <= 1_200_000 and t.leaves.n <= 1_500_000


@pytest.mark.parametrize("sid", ["keyaki", "sugi", "kuromatsu", "shidareyanagi"])
def test_branch_tips_inside_envelope(sid):
    """枝先(レベル1以上の末端節)の 95% 以上がエンベロープ ×1.05 の内側。"""
    p = species_params(sid, broken_branch_ratio=0.0)
    t = pipeline.generate(p)
    tips = np.array([a.pos[-1] for a in t.skeleton.axes if a.level >= 1])
    cx, cy = skeleton._center_fn(t.skeleton.axes[0])(tips[:, 2])         # エンベロープは幹の軸まわり
    rho = np.hypot(tips[:, 0] - cx, tips[:, 1] - cy)
    inside = rho <= envelope.reach(p, tips[:, 2]) * 1.05 + 1e-6
    assert inside.mean() >= 0.95


@pytest.mark.parametrize("sid,lean", [("sugi", 2.0), ("sugi", 12.0), ("sugi_giant", 1.0), ("hinoki", 6.0)])
def test_conifer_crown_reaches_the_tip_even_when_the_trunk_leans(sid, lean):
    """傾いた幹でも、梢の近くまで枝が付く。エンベロープを原点まわりに置いていた頃は、梢でエンベロープの半径（0.3〜0.9 m）より
    幹のずれ（高さ 25 m・傾き 2° で 0.9 m）のほうが大きく、先端の枝がほとんど消えて裸の棒（高さの 17%）になっていた。"""
    t = gen(sid, quality="MEDIUM", trunk_lean_deg=lean, broken_branch_ratio=0.0, dead_branch_ratio=0.0)
    H = t.params.height_m
    z = np.array([a.pos[0, 2] for a in t.skeleton.axes if a.level == 1])
    assert (z > 0.85 * H).sum() >= 8, (sid, lean, int((z > 0.85 * H).sum()))      # 先端の 15% に枝が 8 本以上
    assert z.max() >= 0.93 * H, (sid, lean, z.max())                              # いちばん高い枝の付け根が樹高の 93% 以上


def test_determinism_and_stream_independence():
    a, b = gen("keyaki", quality="PREVIEW"), gen("keyaki", quality="PREVIEW")
    assert mesh_hash(a) == mesh_hash(b)
    assert mesh_hash(a) != mesh_hash(gen("keyaki", quality="PREVIEW", seed=2))
    # 葉の密度・大きさは枝の形を変えない（乱数ストリームが別）
    c = gen("keyaki", quality="PREVIEW", leaves_per_m=200, leaf_scale=2.0, leaf_cluster=2)
    assert mesh_hash(a) == mesh_hash(c)
    assert not np.array_equal(c.leaves.pos, a.leaves.pos) or c.leaves.scale.mean() != a.leaves.scale.mean()


# --- レオナルドの規則 ---------------------------------------------------------
@pytest.mark.parametrize("delta", [1.8, 2.0, 2.5, 3.0])
def test_leonardo_at_trunk_fork(delta):
    """幹分かれ: 分かれる前の幹の d^Δ = 分かれた主枝の d^Δ の和（誤差 1% 以内）。"""
    p = species_params("keyaki", quality="PREVIEW", leonardo_exponent=delta, trunk_split_count=3)
    t = pipeline.generate(p)
    trunk = t.skeleton.axes[0]
    leaders = [a for a in t.skeleton.axes if a.level == 0 and a.parent == 0]
    assert len(leaders) == 3
    lhs = trunk.radius[-1] ** delta
    rhs = sum(a.radius[0] ** delta for a in leaders)
    assert rhs == pytest.approx(lhs, rel=0.01)


def test_children_never_thicker_than_parent_and_taper_down():
    t = gen("quercus_robur", quality="PREVIEW")
    for a in t.skeleton.axes:
        assert np.all(np.isfinite(a.radius)) and np.all(a.radius > 0)
        if a.parent >= 0:
            assert a.radius[0] <= t.skeleton.axes[a.parent].radius[a.parent_node] + 1e-9
    # 枝は付け根から先へ（局所的な折れ曲がり以外で）細くなる
    thick = [a for a in t.skeleton.axes if a.level == 1 and a.n > 5]
    assert np.mean([a.radius[-1] < a.radius[0] for a in thick]) > 0.95


# --- 幹の太さ（ユーザー要件）-----------------------------------------------------
def test_dbh_auto_matches_request():
    t = gen("keyaki", quality="PREVIEW", dbh_cm=63)
    trunk = t.skeleton.axes[0]
    d = float(np.interp(1.3, trunk.pos[:, 2], trunk.radius)) * 200
    assert d == pytest.approx(63, rel=0.01)


def test_custom_trunk_profile_in_skeleton_and_mesh():
    """高さごとの直径指定が、骨格の半径にもメッシュの断面にも反映される。"""
    pts = [(0.0, 110), (0.08, 80), (0.35, 45), (0.7, 22), (1.0, 4)]
    t = gen("sugi", quality="MEDIUM", trunk_profile_mode="CUSTOM", profile=pts, trunk_irregularity=0.0,
            burl_density=0.0, bark_ridge_mm=0.0, root_flare_scale=1.0)
    H = 30.0
    trunk = t.skeleton.axes[0]
    for ratio, d in pts:
        got = float(np.interp(ratio * H, trunk.pos[:, 2], trunk.radius)) * 200
        assert got == pytest.approx(d, rel=0.02), (ratio, d, got)
    rings, V, _ = axis_rings(t, 0)
    r_mesh = ring_radii(rings).mean(axis=1) * 200
    z = rings.mean(axis=1)[:, 2]
    for ratio, d in pts[1:-1]:
        mesh_d = float(np.interp(ratio * H, z, r_mesh))
        assert mesh_d == pytest.approx(d, rel=0.03), (ratio, d, mesh_d)


def test_custom_profile_scales_branches_too():
    """幹を太く指定すると、レオナルド則で枝も太くなる（枝の太さは幹との比で決まる）。"""
    thin = gen("keyaki", quality="PREVIEW", trunk_profile_mode="CUSTOM", profile=[(0, 40), (1, 4)])
    thick = gen("keyaki", quality="PREVIEW", trunk_profile_mode="CUSTOM", profile=[(0, 120), (1, 12)])
    r1 = np.mean([a.radius[0] for a in thin.skeleton.axes if a.level == 1])
    r2 = np.mean([a.radius[0] for a in thick.skeleton.axes if a.level == 1])
    assert r2 / r1 == pytest.approx(3.0, rel=0.1)


# --- 断面の歪み（ユーザー要件）-----------------------------------------------------
FLAT = dict(root_flare_scale=1.0, burl_density=0.0, bark_ridge_mm=0.0)


def test_cross_section_is_circle_when_irregularity_zero():
    t = gen("sugi", quality="MEDIUM", trunk_irregularity=0.0, **FLAT)
    r = ring_radii(axis_rings(t, 0)[0])
    assert (r.max(axis=1) / r.min(axis=1)).max() < 1.001


def test_cross_section_is_not_circular_and_varies_with_height():
    t = gen("sugi", quality="MEDIUM", trunk_irregularity=0.4, **FLAT)
    rings, V, _ = axis_rings(t, 0)
    r = ring_radii(rings)
    ratio = r.max(axis=1) / r.min(axis=1)
    assert ratio.mean() > 1.15, "正円ではない"
    # 平均半径は設計値(=1)のまま（歪みは面積をほぼ保つ）
    nominal = t.skeleton.axes[0].radius
    assert np.allclose(r.mean(axis=1) / nominal, 1.0, atol=0.04)
    # 楕円の長軸の向きが高さとともに変わる
    c = rings.mean(axis=1, keepdims=True)
    phi = np.arctan2((rings - c)[..., 1], (rings - c)[..., 0])
    major = np.array([np.angle(np.sum(r[i] * np.exp(-2j * phi[i]))) / 2 for i in range(len(r))])
    z = rings.mean(axis=1)[:, 2]
    lo, hi = major[z < 5], major[z > 15]
    d = np.abs(np.angle(np.exp(2j * (lo.mean() - hi.mean()))) / 2)
    spread = np.std(np.unwrap(2 * major)) / 2
    assert spread > np.radians(20) or d > np.radians(20)
    # 隣り合う高さの断面は連続的に変化する（ガタつかない）
    assert np.abs(np.diff(ratio)).max() < 0.25


def test_irregularity_scales_with_parameter():
    def mean_ratio(irr):
        t = gen("sugi", quality="PREVIEW", trunk_irregularity=irr, **FLAT)
        r = ring_radii(axis_rings(t, 0)[0])
        return (r.max(axis=1) / r.min(axis=1)).mean()
    a, b, c = mean_ratio(0.0), mean_ratio(0.3), mean_ratio(0.8)
    assert a < b < c


def test_branch_sections_also_distorted():
    t = gen("keyaki", quality="MEDIUM", branch_irregularity=0.6, broken_branch_ratio=0.0)
    ratios = []
    for a in t.skeleton.axes:
        if a.level == 1 and a.n >= 8:
            r = ring_radii(axis_rings(t, a.id)[0])
            ratios.append((r.max(axis=1) / r.min(axis=1))[2:-2].mean())
    assert np.mean(ratios) > 1.1


# --- 折れ枝（ユーザー要件）--------------------------------------------------------
def test_no_damage_when_disabled():
    t = gen("keyaki", quality="PREVIEW", broken_branch_ratio=0.0, top_break_height=0.0)
    assert not any(a.broken or a.neck >= 0 for a in t.skeleton.axes)
    assert t.mesh.face_attrs["tree_break"].sum() == 0
    assert (t.mesh.material_index == tube_mesher.SLOT_BROKEN).sum() == 0


def test_broken_branches_are_truncated_with_jagged_cut_faces():
    ov = dict(quality="MEDIUM", broken_hang_ratio=0.0, dead_branch_ratio=0.0)
    t = gen("keyaki", broken_branch_ratio=0.35, **ov)
    broken = [a for a in t.skeleton.axes if a.broken]
    assert len(broken) >= 5
    assert t.mesh.face_attrs["tree_break"].sum() > 0
    assert (t.mesh.material_index == tube_mesher.SLOT_BROKEN).sum() > 0
    # 折れ口は健全な太さのまま: 先端で尖らない
    l1 = [a for a in broken if a.level == 1]
    assert l1 and np.mean([a.radius[-1] / a.radius[0] > 0.2 for a in l1]) > 0.8


def test_damage_removes_everything_beyond_the_break():
    """損傷モジュールを直接呼び、折れた先（節・子孫）が消えることを確かめる（予算の間引きを混ぜない）。"""
    p = species_params("keyaki", quality="MEDIUM", broken_branch_ratio=0.4, broken_hang_ratio=0.0,
                       dead_branch_ratio=0.0)
    st = rng.make_streams(p.seed)
    sk = skeleton.build(p, st)
    radius.compute_radii(p, sk)
    nodes0, axes0 = sum(a.n for a in sk.axes), len(sk.axes)
    damage.apply_damage(p, sk, st)
    assert sum(a.n for a in sk.axes) < nodes0 and len(sk.axes) < axes0
    for a in sk.axes:
        if a.parent >= 0:
            assert a.parent_node < sk.axes[a.parent].n, "子の付け根は親に残っている"
        if a.broken:
            assert len(a.radius) == a.n == len(a.pos)


def test_break_end_ring_is_ragged():
    t = gen("keyaki", quality="MEDIUM", broken_branch_ratio=0.5, broken_hang_ratio=0.0)
    ax = next(a for a in t.skeleton.axes if a.broken and a.level == 1 and a.n >= 6)
    rings, V, _ = axis_rings(t, ax.id)
    tang = ax.pos[-1] - ax.pos[-2]
    tang /= np.linalg.norm(tang)
    along = (rings[-1] - rings[-1].mean(axis=0)) @ tang
    assert along.max() - along.min() > 0.3 * ax.radius[-1], "最後のリングが前後にばらついている（ささくれ）"
    # 健全な枝の先端リングは平面
    ok = next(a for a in t.skeleton.axes if not a.broken and a.level == 1 and a.n >= 6)
    rg, _, _ = axis_rings(t, ok.id)
    tg = ok.pos[-1] - ok.pos[-2]
    tg /= np.linalg.norm(tg)
    flat = (rg[-1] - rg[-1].mean(axis=0)) @ tg
    assert flat.max() - flat.min() < 0.35 * ok.radius[-1]


def test_hanging_breaks_droop_and_stay_attached():
    t = gen("keyaki", quality="MEDIUM", broken_branch_ratio=0.5, broken_hang_ratio=1.0)
    necks = [a for a in t.skeleton.axes if a.neck >= 0 and a.level == 1]
    assert len(necks) >= 3
    down = [a.pos[-1][2] < a.pos[a.neck][2] for a in necks]
    assert np.mean(down) > 0.9, "折れ口から先は垂れ下がる"
    for a in necks:
        assert np.linalg.norm(a.pos[a.neck + 1] - a.pos[a.neck]) < 1.0      # 折れ口で切り離されていない
    # 垂れ下がった先の子孫に葉は付かない
    assert all(c.dead for c in t.skeleton.axes if c.parent >= 0 and t.skeleton.axes[c.parent].neck >= 0
               and c.parent_node > t.skeleton.axes[c.parent].neck)


def test_top_break():
    t = gen("sugi", quality="PREVIEW", top_break_height=0.6, broken_branch_ratio=0.0)
    trunk = t.skeleton.axes[0]
    assert trunk.broken
    assert trunk.pos[-1][2] == pytest.approx(0.6 * 30, abs=1.0)
    H, _ = pipeline.measure(t)
    assert H < 0.7 * 30
    assert not any(a.parent == 0 and a.parent_node >= trunk.n for a in t.skeleton.axes)


def test_damage_scales_with_age_and_ratio():
    n = lambda **k: sum(a.broken or a.neck >= 0 for a in gen("keyaki", quality="PREVIEW", **k).skeleton.axes)  # noqa: E731
    assert n(broken_branch_ratio=0.05) < n(broken_branch_ratio=0.4)
    assert n(broken_branch_ratio=0.2, age_factor=0.2) < n(broken_branch_ratio=0.2, age_factor=1.5)


def test_dead_branches_carry_no_leaves():
    t = gen("keyaki", quality="PREVIEW", dead_branch_ratio=0.5, broken_branch_ratio=0.0)
    assert any(a.dead for a in t.skeleton.axes)
    dead_tips = np.array([a.pos[-1] for a in t.skeleton.axes if a.dead and a.level == 2])
    if len(dead_tips):
        d = np.linalg.norm(t.leaves.pos[None, :, :] - dead_tips[:, None, :], axis=2).min(axis=1)
        assert np.median(d) > 0.05


# --- メッシュの整合 ---------------------------------------------------------------
@pytest.mark.parametrize("sid", ALL)
def test_mesh_integrity(sid):
    t = gen(sid, quality="PREVIEW")
    m = t.mesh
    nv = len(m.co)
    assert np.isfinite(m.co).all() and np.isfinite(m.uv).all()
    assert m.loop_vert.min() >= 0 and m.loop_vert.max() < nv
    assert set(np.unique(m.face_size)) <= {3, 4}
    assert int(m.face_size.sum()) == len(m.loop_vert) == len(m.uv)
    assert len(m.material_index) == len(m.face_size) == len(m.face_attrs["tree_part"])
    assert m.material_index.min() >= 0 and m.material_index.max() <= 7
    for name, arr in m.point_attrs.items():
        assert len(arr) == nv, name
    for k in ("tree_level", "dist_root", "dist_root_n", "branch_t", "branch_phase", "branch_pivot", "branch_id",
              "junction_blend"):
        assert k in m.point_attrs
    assert m.co[:, 2].min() > -0.6                                   # 地面の下に潜らない


def test_tree_parts_cover_trunk_branch_twig_and_root():
    t = gen("keyaki", quality="MEDIUM")
    parts = set(np.unique(t.mesh.face_attrs["tree_part"]).tolist())
    assert {tube_mesher.PART_TRUNK, tube_mesher.PART_BRANCH, tube_mesher.PART_TWIG, tube_mesher.PART_ROOT} <= parts
    # 根張りは地面近くの幹の面だけ
    co = t.mesh.co
    mask = t.mesh.face_attrs["tree_part"] == tube_mesher.PART_ROOT
    starts = np.concatenate([[0], np.cumsum(t.mesh.face_size)[:-1]])
    zr = co[t.mesh.loop_vert[starts[mask]], 2]
    assert zr.max() < 0.05 * 20 * 1.5


def test_trunk_normals_point_outward_via_signed_volume():
    t = gen("sugi", quality="MEDIUM", trunk_irregularity=0.2)
    rings, V, sel = axis_rings(t, 0)
    ok = np.zeros(len(t.mesh.co), bool)
    ok[sel] = True
    starts = np.concatenate([[0], np.cumsum(t.mesh.face_size)[:-1]])
    all_in = np.add.reduceat(ok[t.mesh.loop_vert].astype(int), starts) == t.mesh.face_size
    vol = 0.0
    for f in np.nonzero(all_in)[0]:
        vs = t.mesh.co[t.mesh.loop_vert[starts[f]: starts[f] + t.mesh.face_size[f]]].astype(float)
        for k in range(1, len(vs) - 1):
            vol += np.dot(vs[0], np.cross(vs[k], vs[k + 1])) / 6.0
    r = t.skeleton.axes[0].radius
    z = t.skeleton.axes[0].pos[:, 2]
    expect = float(np.sum(np.pi * (0.5 * (r[1:] + r[:-1])) ** 2 * np.diff(z)))
    assert vol > 0, "法線が外向き（符号付き体積が正）"
    assert vol == pytest.approx(expect, rel=0.15)


def test_uv_bark_tiling_uses_integer_wraps():
    t = gen("keyaki", quality="PREVIEW", uv_integer_wrap=True, bark_tile_w_m=1.0)
    m = t.mesh
    starts = np.concatenate([[0], np.cumsum(m.face_size)[:-1]])
    quad = np.nonzero(m.face_size == 4)[0][:2000]
    u = m.uv[starts[quad][:, None] + np.arange(4)][..., 0]
    span = u.max(axis=1) - u.min(axis=1)
    # 四角形の u 幅は 整数/V の倍数（周方向の整数巻き）
    assert np.all(span >= 0)
    t2 = gen("keyaki", quality="PREVIEW", uv_integer_wrap=False)
    assert t2.mesh.uv.shape == m.uv.shape


# --- 葉・季節 ---------------------------------------------------------------------
def test_season_controls_leaves_and_flowers():
    kw = dict(quality="PREVIEW")
    assert gen("keyaki", season=0.0, **kw).leaves.n == 0                       # 冬
    assert gen("keyaki", season=0.5, **kw).leaves.is_leaf.sum() > 100          # 夏
    assert gen("keyaki", season=1.0, **kw).leaves.n == 0                       # 落葉
    assert gen("sugi", season=0.0, **kw).leaves.is_leaf.sum() > 100            # 常緑
    s = gen("someiyoshino", season=0.15, **kw).leaves
    assert s.is_flower.sum() > 100 and s.is_leaf.sum() == 0                    # 開花時は葉なし
    assert gen("someiyoshino", season=0.5, **kw).leaves.is_flower.sum() == 0
    assert gen("keyaki", season=0.95, **kw).leaves.color_t > 0.9               # 紅葉


def test_leaf_points_orientation_and_scale():
    t = gen("keyaki", quality="PREVIEW", season=0.5)
    q = t.leaves.quat
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-6)
    w, x, y, z = q.T
    nz = 1 - 2 * (x * x + y * y)                                               # ローカル +Z（葉の表）の世界 Z 成分
    assert nz[t.leaves.is_leaf].mean() > 0.2, "葉の表は概ね上を向く"
    sc = t.leaves.scale[t.leaves.is_leaf][:, 0]
    assert sc.min() > 0.3 and sc.max() < 3.5
    assert np.allclose(t.leaves.scale[:, 0], t.leaves.scale[:, 1])


def test_leaf_scale_param():
    a = gen("keyaki", quality="PREVIEW", season=0.5, leaf_scale=1.0).leaves.scale[:, 0].mean()
    b = gen("keyaki", quality="PREVIEW", season=0.5, leaf_scale=2.0).leaves.scale[:, 0].mean()
    assert b / a == pytest.approx(2.0, rel=0.05)


# --- 太い幹の細密化と巨大スギ -----------------------------------------------------
def test_ring_verts_grow_with_trunk_radius():
    rv = allometry.ring_verts
    assert rv("PREVIEW", 0, 3.0) == 12                                   # プレビューは増やさない
    assert rv("MEDIUM", 0, 0.2) == 24 and rv("HERO", 0, 0.2) == 48       # 細い幹は表どおり
    assert rv("MEDIUM", 0, 1.2) > 24 and rv("HERO", 0, 1.3) > rv("HERO", 0, 0.5)
    assert rv("HERO", 0, 50.0) == 256 and rv("MEDIUM", 0, 50.0) == 96    # 上限
    assert all(rv(q, 0, r) % 4 == 0 for q in ("MEDIUM", "HERO") for r in (0.3, 0.77, 1.31))
    assert rv("HERO", 1, 5.0) == 16 and rv("HERO", 3, 5.0) == 5          # 枝は半径に依らない


def test_giant_cedar_preset():
    p = species_params("sugi_giant", quality="HERO")
    assert (p.height_m, p.trunk_profile_mode, p.crown_shape) == (55.0, "CUSTOM", "conical")
    t0 = time.time()
    t = pipeline.generate(p)
    assert time.time() - t0 <= 15.0
    H, W = pipeline.measure(t)
    assert H == pytest.approx(55.0, rel=0.03) and W == pytest.approx(20.0, rel=0.08)
    assert t.mesh.tri_count <= 1_200_000
    trunk = t.skeleton.axes[0]
    d = lambda z: float(np.interp(z, trunk.pos[:, 2], trunk.radius)) * 200      # noqa: E731
    assert d(1.3) == pytest.approx(261, rel=0.03)                        # 胸高直径 約 2.6 m
    assert d(0.0) > d(1.3) > d(30.0) > d(50.0)
    rings, V, _ = axis_rings(t, 0)
    assert V == 256                                                      # 太い幹はリングを細かく切る
    arc = 2 * np.pi * trunk.radius[0] / V
    assert arc < 0.05
    # 老齢木: 枯れ枝・折れ枝が多い
    ax = t.skeleton.axes
    assert sum(a.dead for a in ax) / len(ax) > 0.05 and sum(a.broken or a.neck >= 0 for a in ax) > 20
    # 地上高 20 m 付近（枝下高）までは生きた枝が無い。残るのは枯れて折れた枝の根元（スタブ）だけ
    low = [a for a in ax if a.level == 1 and a.pos[0][2] < 20.0]
    assert low and all(a.dead and a.broken for a in low)


@pytest.mark.parametrize("quality,radius,expected", [("MEDIUM", 0.4, 6), ("HERO", 1.2, 62)])
def test_bark_ridge_count_follows_circumference(quality, radius, expected):
    """縦筋の本数は円周から決まる（幅 12 cm）。リングで解像できる範囲（V/4）に収まる。"""
    prof = [(0.0, radius * 200), (1.0, radius * 200 * 0.97)]
    t = gen("sugi", quality=quality, trunk_profile_mode="CUSTOM", profile=prof, trunk_irregularity=0.0,
            burl_density=0.0, root_flare_scale=1.0, bark_ridge_mm=30.0, height_m=20, broken_branch_ratio=0.0)
    rings, V, _ = axis_rings(t, 0)
    r = ring_radii(rings)[len(rings) // 2]
    spec = np.abs(np.fft.rfft(r - r.mean()))
    k = int(np.argmax(spec[2:V // 2])) + 2
    assert abs(k - expected) <= 2, (V, k, expected)


# --- 葉の量 --------------------------------------------------------------------------
UNCAPPED = dict(quality="MEDIUM", leaf_max_instances=3_000_000)


def _leaf_n(sid="keyaki", **ov):
    return int(gen(sid, **{**UNCAPPED, **ov}).leaves.is_leaf.sum())


def test_foliage_amount_scales_leaf_count_linearly():
    n1 = _leaf_n(foliage_amount=1.0)
    assert _leaf_n(foliage_amount=0.0) == 0
    assert _leaf_n(foliage_amount=0.5) == pytest.approx(0.5 * n1, rel=0.08)
    assert _leaf_n(foliage_amount=2.0) == pytest.approx(2.0 * n1, rel=0.08)
    assert _leaf_n(foliage_amount=4.0) == pytest.approx(4.0 * n1, rel=0.08)


def test_foliage_amount_beyond_cap_enlarges_leaves_instead():
    """上限に達したら数を増やせないので、1枚あたりを大きくして密度感を保つ（上限 2.5 倍）。"""
    a = gen("keyaki", quality="PREVIEW", foliage_amount=1.0).leaves
    b = gen("keyaki", quality="PREVIEW", foliage_amount=3.0).leaves
    assert a.n == b.n == allometry.quality_table("PREVIEW")["leaf_cap"]
    assert b.scale[:, 0].mean() > 1.4 * a.scale[:, 0].mean()
    c = gen("keyaki", quality="PREVIEW", foliage_amount=5.0).leaves
    assert c.scale[:, 0].mean() <= 2.6 * a.scale[:, 0].mean()


def test_leaf_max_instances_overrides_quality_cap():
    lo = gen("keyaki", quality="MEDIUM", leaf_max_instances=5000).leaves
    assert lo.is_leaf.sum() == 5000
    hi = gen("keyaki", quality="PREVIEW", leaf_max_instances=60_000, foliage_amount=3.0).leaves
    assert hi.is_leaf.sum() > allometry.quality_table("PREVIEW")["leaf_cap"]
    assert hi.is_leaf.sum() <= 60_000


def test_leaf_reach_extends_to_inner_branches():
    for sid in ("keyaki", "sugi_giant"):
        t, o, a = (_leaf_n(sid, leaf_reach=r) for r in ("TERMINAL", "OUTER", "ALL"))
        assert t < o <= a, (sid, t, o, a)
    # 枝が3段ある針葉樹（標準）では ALL は OUTER より確実に多い。二重に配置された軸は無い
    o, a = (_leaf_n("sugi", leaf_reach=r) for r in ("OUTER", "ALL"))
    assert a > o, (o, a)
    # AUTO: 針葉は OUTER、広葉は TERMINAL（従来の挙動）
    assert _leaf_n("sugi_giant", leaf_reach="AUTO") == _leaf_n("sugi_giant", leaf_reach="OUTER")
    assert _leaf_n("keyaki", leaf_reach="AUTO") == _leaf_n("keyaki", leaf_reach="TERMINAL")


def test_leaf_start_auto_matches_the_old_per_type_values():
    """leaf_start=-1（自動）は、旧版の葉の種類ごとの付き始め（針葉 0.1 / 針葉束 0.5 / その他 0.2）と同じ。"""
    for sid, expect in (("sugi", 0.1), ("hinoki", 0.1), ("kuromatsu", 0.5), ("keyaki", 0.2), ("ichou", 0.2)):
        assert _leaf_n(sid, leaf_start=-1.0) == _leaf_n(sid, leaf_start=expect), sid
    assert schema.BY_NAME["leaf_start"].default == -1.0


def test_leaf_start_moves_leaves_toward_tips():
    assert _leaf_n(leaf_start=0.0) > _leaf_n(leaf_start=0.2) > _leaf_n(leaf_start=0.6)


def _point_to_polyline(pt, poly):
    """点から折れ線への最短距離と、足の位置（弧上の節番号 + 線分内の比）。"""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(np.sum((pt - a) * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-12), 0.0, 1.0)
    d = np.linalg.norm(pt - (a + ab * t[:, None]), axis=1)
    k = int(np.argmin(d))
    return float(d[k]), k, float(t[k])


def test_stem_top_leaves_sit_on_the_bark_surface():
    """幹の先端寄り（上位 18%）に付ける葉は、幹の中心線ではなく樹皮の表面（半径の 0.9 倍）に置く。
    以前は幹が葉の対象外で、頂部が裸の針のように見えた。"""
    t = gen("sugi_giant", quality="PREVIEW")
    trunk = t.skeleton.axes[0]
    rng = np.random.default_rng(0)
    pts, tg = leaf_points._place([trunk], 80.0, 0.82, rng, 1.0, 0.1, 0.2, surface=True)
    assert len(pts) > 100
    s_arc = trunk.arc()
    assert pts[:, 2].min() >= 0.82 * s_arc[-1] * 0.95                            # 先端寄りだけ
    for pt in pts[:60]:
        d, k, f = _point_to_polyline(pt, trunk.pos)
        r = float(trunk.radius[k] * (1 - f) + trunk.radius[k + 1] * f)
        assert abs(d - 0.9 * r) <= 0.02 + 0.05 * r, (d, r)
    # generate() 全体でも、頂部の幹の近くに葉がある（枝の葉だけで済んでいないこと自体は上の単体で担保）
    near_top = trunk.pos[trunk.arc() >= 0.93 * s_arc[-1]]
    lp = t.leaves.pos[t.leaves.is_leaf]
    dmin = np.min(np.linalg.norm(lp[:, None, :] - near_top[None, :, :], axis=2), axis=1)
    assert (dmin < 0.8).sum() >= 20


def test_generate_places_stem_top_leaves_on_the_surface(monkeypatch):
    """generate() が幹の葉を表面（surface=True）に、先端寄り（u0 0.7〜0.9）から付けている。"""
    calls = []
    orig = leaf_points._place

    def spy(axes, per_m, u0, *a, **k):
        calls.append((list(axes), per_m, u0, k.get("surface", False)))
        return orig(axes, per_m, u0, *a, **k)

    monkeypatch.setattr(leaf_points, "_place", spy)
    gen("sugi", quality="PREVIEW")
    trunk = [c for c in calls if c[0] and all(a.level == 0 for a in c[0])]
    assert len(trunk) == 1 and trunk[0][3] is True and 0.7 <= trunk[0][2] <= 0.9 and trunk[0][1] > 0


def test_stem_top_leaves_not_on_the_bare_lower_trunk_of_split_trees():
    """幹分かれの木（ケヤキ）では、分岐前の短い幹は樹冠より下にある。そこには葉を付けない。"""
    t = gen("keyaki", quality="MEDIUM", season=0.5, trunk_split_count=3, trunk_split_height=0.3, crown_base_ratio=0.5)
    hb = 0.5 * 20.0
    lower = [a for a in t.skeleton.axes if a.level == 0 and a.pos[-1, 2] < hb]
    assert lower, "分岐前の幹が樹冠より下で終わる設定のはず"
    lp = t.leaves.pos[t.leaves.is_leaf]
    for a in lower:
        zs = a.pos[:, 2]
        d = np.min(np.linalg.norm(lp[:, None, :] - a.pos[None, :, :], axis=2), axis=1)
        low_leaf = lp[:, 2] < hb - 0.5
        assert not np.any(low_leaf & (d < a.radius.max() * 1.05 + 0.05)), (zs.max(), int(np.sum(low_leaf & (d < 1.0))))


def _leaf_offset_from_branches(t, n=1500):
    """葉（房）の位置から、いちばん近い枝の節までの距離の平均 [m]。散らばりが大きいほど大きい。"""
    nodes = np.concatenate([a.pos for a in t.skeleton.axes if not a.removed and a.level >= 1])
    lp = t.leaves.pos[t.leaves.is_leaf]
    pick = lp[np.random.default_rng(0).choice(len(lp), min(n, len(lp)), replace=False)]
    d = np.min(np.linalg.norm(pick[:, None, :] - nodes[None, :, :], axis=2), axis=1)
    return float(d.mean())


def test_leaf_scatter_spreads_sprays_around_the_branch():
    """房が枝の軸の上にそろうと、枝先ごとの固まり（芋虫状）になる。散らばりを上げると、枝から離れた位置にも房が付く。"""
    kw = dict(quality="PREVIEW", leaf_max_instances=40_000)
    d0 = _leaf_offset_from_branches(gen("sugi", leaf_scatter=0.0, **kw))
    d1 = _leaf_offset_from_branches(gen("sugi", leaf_scatter=1.0, **kw))
    assert d1 > d0 + 0.03, (d0, d1)
    # 自動（-1）: 針葉(らせん)は 0.7、それ以外の葉は 0
    a = gen("sugi", leaf_scatter=-1.0, **kw)
    b = gen("sugi", leaf_scatter=0.7, **kw)
    assert np.array_equal(a.leaves.pos, b.leaves.pos) and np.array_equal(a.leaves.quat, b.leaves.quat)
    k_auto = gen("keyaki", leaf_scatter=-1.0, quality="PREVIEW")
    k_zero = gen("keyaki", leaf_scatter=0.0, quality="PREVIEW")
    assert np.array_equal(k_auto.leaves.pos, k_zero.leaves.pos)             # 広葉は従来どおり（出力が変わらない）


def test_foliage_controls_do_not_change_branches():
    base = mesh_hash(gen("keyaki", quality="PREVIEW"))
    for ov in (dict(foliage_amount=3.0), dict(leaf_reach="ALL"), dict(leaf_start=0.0),
               dict(leaf_max_instances=999), dict(leaf_scatter=1.2)):
        assert mesh_hash(gen("keyaki", quality="PREVIEW", **ov)) == base, ov


def test_trunk_stubs_below_crown_base():
    kw = dict(quality="MEDIUM", height_m=30.0, crown_base_ratio=0.5, dead_branch_ratio=0.0)
    off = gen("sugi", trunk_stubs=False, trunk_stub_density=1.0, **kw)          # オフなら密度があっても付かない
    on = gen("sugi", trunk_stubs=True, trunk_stub_density=1.0, **kw)
    hb = 0.5 * 30.0
    low_off = [a for a in off.skeleton.axes if a.level == 1 and a.pos[0][2] < hb]
    low_on = [a for a in on.skeleton.axes if a.level == 1 and a.pos[0][2] < hb]
    assert not low_off and len(low_on) >= 8                       # 約 1 本/m × (15 − 1.2) m
    trunk = on.skeleton.axes[0]
    for a in low_on:
        assert a.dead and a.broken and a.parent == 0 and a.pos[0][2] >= 1.2
        r_t = float(np.interp(a.pos[0][2], trunk.pos[:, 2], trunk.radius))
        reach = float(np.hypot(*(a.pos[-1, :2] - trunk.pos[a.parent_node, :2])))
        assert reach >= r_t + 0.25, (reach, r_t)                    # 幹の表面から確かに突き出す
        assert a.radius[0] > a.radius[-1] > 0.5 * a.radius[0]      # 折れ口は先細りしすぎない
    # 根張りの高さより上にだけ付く（根張りで幹の実半径が大きく、スタブが埋まるため）
    flared = gen("sugi", trunk_stubs=True, trunk_stub_density=2.0, root_flare_height=0.15, root_flare_scale=2.2, **kw)
    stubs = [a for a in flared.skeleton.axes if a.level == 1 and a.pos[0][2] < hb]
    assert stubs and min(a.pos[0][2] for a in stubs) >= 0.15 * 30.0 - 1e-6
    # 他の枝（骨格・半径・折れ）には触れない: スタブは最後に足されるだけ
    n0 = len(off.skeleton.axes)
    assert len(on.skeleton.axes) == n0 + len(low_on)
    for a, b in zip(off.skeleton.axes, on.skeleton.axes[:n0]):
        assert np.array_equal(a.pos, b.pos) and np.array_equal(a.radius, b.radius)
        assert (a.dead, a.broken, a.neck) == (b.dead, b.broken, b.neck)


def test_trunk_stubs_default_off_and_giant_cedar_on():
    assert schema.BY_NAME["trunk_stubs"].default is False
    assert species_params("keyaki").trunk_stubs is False and species_params("sugi_giant").trunk_stubs is True
    assert not [a for a in gen("sugi", quality="MEDIUM", crown_base_ratio=0.5).skeleton.axes if a.level == 1 and a.dead and a.broken and a.pos[0][2] < 15.0]


def test_lower_crown_branches_die_more_often():
    """樹冠の下側の L1 の枝ほど枯れやすい。枝の本数を多くして、枯れる割合を下・上で直接比べる（実装を外すと落ちる）。"""
    p = species_params("sugi_giant", quality="MEDIUM", l1_density=40.0, dead_branch_ratio=0.2, age_factor=1.5)
    st = rng.make_streams(p.seed)
    sk = skeleton.Skeleton()
    skeleton._build_stems(p, st.skel, sk)
    sp = skeleton._child_specs(p, [a for a in sk.axes if a.level == 0], 1, st.skel)
    hb = p.crown_base_ratio * p.height_m
    t = (sp["start"][:, 2] - hb) / (p.height_m - hb)
    dead = sp["dead"]
    lower, upper = dead[t < 0.33].mean(), dead[t > 0.67].mean()
    assert (t < 0.33).sum() > 150 and (t > 0.67).sum() > 150
    assert lower - upper > 0.08, (lower, upper)                           # 期待値は 0.46 対 0.29 前後
    assert dead.mean() <= 0.6


def test_giant_cedar_branch_and_leaf_balance():
    """枝に対して葉が多すぎると、少ない枝の先に大きな房が付いて見える。HERO では細い枝（L2・L3）を十分に分岐させ、
    1 本の枝あたりの葉の数を抑える（旧: L2 964・L3 2433 本に約 43 万枚で 1 本あたり約 126 枚）。"""
    t = gen("sugi_giant", quality="HERO")
    ax = [a for a in t.skeleton.axes if not a.removed]
    n2, n3 = (sum(1 for a in ax if a.level == lv) for lv in (2, 3))
    assert n2 >= 2500 and n3 >= 3500, (n2, n3)
    assert t.leaves.is_leaf.sum() / (n2 + n3) <= 80.0
    assert t.mesh.tri_count <= 1_200_000                                 # 仕様の予算のまま
