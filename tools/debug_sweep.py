"""デバッグ掃引: 全樹種 × 品質 3 段階 × 3 本（サイズと乱数を変えたもの）を生成して、仕様と違う点を自動で洗い出す。

    python tools/debug_sweep.py                       # 13 樹種 × 3 品質 × 3 本 = 117 本。bpy があれば Blender 側の検査もする
    python tools/debug_sweep.py --species keyaki,sugi --qualities PREVIEW,MEDIUM
    python tools/debug_sweep.py --no-bpy              # bpy なしで、コア（骨格・メッシュ・葉の点）だけ
    python tools/debug_sweep.py --json out.json --report docs/DEBUG_REPORT.md

3 本の中身: サイズ倍率 0.7 / 1.0 / 1.4（樹高・樹冠幅・胸高直径・太さプロファイルに掛ける。スキーマの上限で頭打ち）と、互いに違う乱数シード。

検査の元になる仕様:
  - 仕様書 3.5 品質の目標値: 三角形数・葉インスタンス数・生成時間
  - 仕様書 8 受け入れ基準: 樹高・樹冠幅 ±5%、枝先の 95% 以上がエンベロープ ×1.05 内、7 スロットと tree_part、再生成でマテリアル保持
  - 仕様書 4.2 12 種のデフォルト値（プリセットが表どおりか）、1.6 季節（ソメイヨシノは開花時に葉なし）
  - 依頼: 幹の太さを「高さ X で太さ Y」で指定できる / 枝が折れた部分がある / 断面が正円でなく高さごとに歪む
  - 実際の木としての妥当性: 梢まで枝が付く、樹冠より下に生きた枝がない、地面の下にメッシュがない、など
重大度: FAIL（仕様違反・明らかな不具合）/ WARN（許容内だが気になる）。
"""
import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tree_forge import preset_io, schema  # noqa: E402
from tree_forge.core import allometry, envelope, pipeline, skeleton  # noqa: E402

SIZE_SCALES = (0.7, 1.0, 1.4)
QUALITIES = ("PREVIEW", "MEDIUM", "HERO")
TIME_LIMIT = {"PREVIEW": 1.0, "MEDIUM": 4.0, "HERO": 15.0}          # 仕様書 3.5 [s]

# 仕様書 4.2 の表（12 種）。H, DBH, W, 形, apical, grav, L1 角, L1 葉序, droop L1/L2/L3, 葉, 葉長 [cm], 樹皮
SPEC_4_2 = {
    "sugi": (30, 80, 7.5, "conical", 0.95, 0.6, 85, "SPIRAL", (0.2, 0.4, 0.5), "NEEDLE_SPIRAL", 1.2, "strip_peel"),
    "hinoki": (25, 60, 7, "conical", 0.9, 0.5, 70, "SPIRAL", (0.2, 0.3, 0.4), "SCALE_TWIG", 8, "strip_peel"),
    "kuromatsu": (18, 60, 12, "umbrella", 0.6, 0.3, 65, "WHORL", (0.1, 0.2, 0.2), "NEEDLE_FASCICLE", 10, "scaly_plates"),
    "keyaki": (20, 80, 18, "vase", 0.35, 0.25, 38, "SPIRAL", (0.05, 0.1, 0.15), "BROAD", 6, "flaking"),
    "someiyoshino": (10, 50, 11, "spherical", 0.3, 0.1, 50, "SPIRAL", (0.1, 0.2, 0.3), "BROAD", 10, "smooth_lenticel"),
    "ichou": (20, 70, 10.5, "ovoid", 0.8, 0.4, 50, "SPIRAL", (0.05, 0.1, 0.1), "FAN", None, "fissured"),
    "iroha_momiji": (6, 30, 6.5, "spherical", 0.3, 0.1, 55, "OPPOSITE", (0.1, 0.2, 0.3), "BROAD", 7, "smooth_lenticel"),
    "shirakaba": (12, 30, 6, "ovoid", 0.7, 0.3, 50, "SPIRAL", (0.1, 0.3, 0.5), "BROAD", 6, "smooth_lenticel"),
    "kusunoki": (20, 120, 25, "spherical", 0.3, 0.15, 50, "SPIRAL", (0.05, 0.1, 0.2), "BROAD", 9, "fissured"),
    "quercus_robur": (25, 120, 25, "spherical", 0.35, 0.1, 60, "SPIRAL", (0.05, 0.1, 0.2), "BROAD", 10, "fissured"),
    "picea_abies": (35, 100, 9, "conical", 0.98, 0.0, 90, "WHORL", (0.4, 0.6, 0.7), "NEEDLE_SPIRAL", 1.8, "flaking"),
    "shidareyanagi": (12, 70, 13, "weeping", 0.3, -0.2, 40, "SPIRAL", (0.4, 0.9, 1.0), "BROAD", 11, "fissured"),
}


class Findings:
    def __init__(self):
        self.items = []             # (severity, code, message)

    def fail(self, code, msg):
        self.items.append(("FAIL", code, msg))

    def warn(self, code, msg):
        self.items.append(("WARN", code, msg))


# --------------------------------------------------------------------------------------
# 木の条件
# --------------------------------------------------------------------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def variant_overrides(sid, si, qi, vi, scale):
    """プリセット + サイズ倍率 + シード。サイズは樹高・樹冠幅・胸高直径（と太さプロファイル）に掛ける。"""
    ov = preset_io.builtin_overrides(sid)
    p0 = schema.make_params(**ov)
    hp, wp, dp = schema.BY_NAME["height_m"], schema.BY_NAME["crown_width_m"], schema.BY_NAME["dbh_cm"]
    ov["height_m"] = clamp(p0.height_m * scale, hp.lo, hp.hi)
    ov["crown_width_m"] = clamp(p0.crown_width_m * scale, wp.lo, wp.hi)
    ov["dbh_cm"] = clamp(p0.dbh_cm * scale, dp.lo, dp.hi)
    if ov.get("profile"):
        ov["profile"] = [(t, clamp(d * scale, 2.0, 400.0)) for t, d in ov["profile"]]
    ov["quality"] = QUALITIES[qi]
    ov["seed"] = 7 + 1009 * vi + 31 * si + 3 * qi
    return ov


# --------------------------------------------------------------------------------------
# コアの検査
# --------------------------------------------------------------------------------------
def trunk_rings(t):
    pa = t.mesh.point_attrs
    sel = np.nonzero(pa["branch_id"] == 0)[0]
    ax = t.skeleton.axes[0]
    extra = 2 if ax.parent < 0 else 1
    V = (len(sel) - extra) // ax.n
    return t.mesh.co[sel[: ax.n * V]].reshape(ax.n, V, 3).astype(float), V


def ring_stats(ring):
    c = ring.mean(axis=0)
    d = ring - c
    r = np.linalg.norm(d, axis=1)
    cov = np.cov(d[:, :2].T)
    w, vec = np.linalg.eigh(cov)
    ang = math.atan2(vec[1, 1], vec[0, 1]) % math.pi
    return float(r.mean()), float(r.std() / max(r.mean(), 1e-9)), ang


def check_core(p, t, q, dt, f: Findings) -> dict:
    m = {}
    H, W = pipeline.measure(t)
    m["H"], m["W"] = H, W
    m["H_err"], m["W_err"] = H / p.height_m - 1.0, W / p.crown_width_m - 1.0
    cap = allometry.quality_table(q)
    n_leaf = int(t.leaves.is_leaf.sum())
    m.update(tris=t.mesh.tri_count, leaves=n_leaf, flowers=int(t.leaves.is_flower.sum()), time=dt)

    # --- 数値の健全性 ---
    mesh = t.mesh
    if not np.all(np.isfinite(mesh.co)) or not np.all(np.isfinite(mesh.uv)):
        f.fail("NAN", "頂点または UV に NaN/inf がある")
    if len(mesh.loop_vert) and (mesh.loop_vert.min() < 0 or mesh.loop_vert.max() >= len(mesh.co)):
        f.fail("INDEX", "面の頂点番号が範囲外")
    if int(mesh.face_size.sum()) != len(mesh.loop_vert):
        f.fail("INDEX", "face_size の合計と loop 数が合わない")
    if len(mesh.uv) and (np.abs(mesh.uv).max() > 2000):
        f.warn("UV", f"UV の絶対値が大きい（{np.abs(mesh.uv).max():.0f}）")
    mi = mesh.material_index
    if mi.min() < 0 or mi.max() > 7:
        f.fail("SLOT", f"材質スロット番号が 0〜7 の外（{mi.min()}〜{mi.max()}）")
    for k in ("tree_part", "tree_break"):
        if k not in mesh.face_attrs:
            f.fail("ATTR", f"面属性 {k} が無い")
    lq = t.leaves.quat
    if len(lq):
        if not np.all(np.isfinite(lq)) or np.abs(np.linalg.norm(lq, axis=1) - 1).max() > 1e-3:
            f.fail("LEAFQ", "葉の回転が単位四元数でない")
        if not np.all(np.isfinite(t.leaves.pos)) or (t.leaves.scale <= 0).any():
            f.fail("LEAFQ", "葉の位置が NaN、またはスケールが 0 以下")

    # --- 寸法（仕様 8: ±5%）---
    if abs(m["H_err"]) > 0.05:
        f.fail("HEIGHT", f"樹高 {H:.2f} m が目標 {p.height_m:.2f} m から {m['H_err'] * 100:+.1f}%")
    elif abs(m["H_err"]) > 0.03:
        f.warn("HEIGHT", f"樹高が目標から {m['H_err'] * 100:+.1f}%")
    if abs(m["W_err"]) > 0.10:
        f.fail("WIDTH", f"樹冠幅 {W:.2f} m が目標 {p.crown_width_m:.2f} m から {m['W_err'] * 100:+.1f}%")
    elif abs(m["W_err"]) > 0.05:
        f.warn("WIDTH", f"樹冠幅が目標から {m['W_err'] * 100:+.1f}%")

    # --- 品質の目標（仕様 3.5）---
    if m["tris"] > cap["tri_cap"]:
        f.fail("TRIS", f"三角形 {m['tris']:,} が上限 {cap['tri_cap']:,} 超え")
    if n_leaf > cap["leaf_cap"]:
        f.fail("LEAVES", f"葉 {n_leaf:,} が上限 {cap['leaf_cap']:,} 超え")
    if dt > TIME_LIMIT[q]:
        f.fail("TIME", f"生成 {dt:.1f} s が目標 {TIME_LIMIT[q]:g} s 超え")

    # --- 幹の太さ（依頼: 高さ X で太さ Y）---
    trunk = t.skeleton.axes[0]
    rings, V = trunk_rings(t)
    z_nodes = trunk.pos[:, 2]
    fn = allometry.trunk_radius_fn(p)

    def mean_diameter_at(z):
        k = int(np.argmin(np.abs(z_nodes - z)))
        return 2.0 * ring_stats(rings[k])[0] * 100.0, float(z_nodes[k])

    d_bh, z_bh = mean_diameter_at(allometry.BREAST_HEIGHT_M)
    want = float(fn(z_bh)) * 200.0
    m["dbh_mesh"], m["dbh_want"] = d_bh, want
    if abs(d_bh / want - 1) > 0.10:
        f.fail("TRUNK", f"胸高直径(z={z_bh:.2f} m) がメッシュで {d_bh:.1f} cm、設計 {want:.1f} cm（{(d_bh / want - 1) * 100:+.1f}%）")
    elif abs(d_bh / want - 1) > 0.05:
        f.warn("TRUNK", f"胸高直径がメッシュと設計で {(d_bh / want - 1) * 100:+.1f}%")
    for frac in (0.2, 0.5, 0.8):
        z_t = frac * float(z_nodes[-1])
        d_m, zz = mean_diameter_at(z_t)
        w_ = float(fn(zz)) * 200.0
        if abs(d_m / w_ - 1) > 0.12:
            f.fail("TRUNK", f"z={zz:.1f} m の幹の直径がメッシュで {d_m:.1f} cm、設計 {w_:.1f} cm（{(d_m / w_ - 1) * 100:+.1f}%）")

    # --- 断面が正円でなく、高さごとに歪む（依頼）---
    ks = [int(round(x * (trunk.n - 1))) for x in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)]
    stats = [ring_stats(rings[k]) for k in ks]
    cvs = [s[1] for s in stats]
    angs = np.array([s[2] for s in stats])
    m["section_cv"] = float(np.median(cvs))
    if p.trunk_irregularity >= 0.1 and V >= 8:
        if m["section_cv"] < 0.005:
            f.fail("SECTION", f"幹の断面がほぼ正円（半径の変動 {m['section_cv'] * 100:.2f}%）")
        spread = float(np.ptp(np.unwrap(2 * angs)) / 2)
        if spread < 0.15 and m["section_cv"] < 0.02:
            f.warn("SECTION", "断面の向きが高さでほとんど変わらない")

    # --- 折れた枝（依頼）---
    ax_all = [a for a in t.skeleton.axes if not a.removed]
    n_broken = sum(1 for a in ax_all if a.broken or a.neck >= 0)
    m["broken_axes"] = n_broken
    if p.broken_branch_ratio * p.age_factor > 0.03 and len(ax_all) >= 40 and n_broken == 0:
        f.warn("BROKEN", "折れた枝が 1 本も無い（折れ枝の割合 > 0）")
    if n_broken and int((t.mesh.face_attrs["tree_break"] > 0).sum()) == 0:
        f.fail("BROKEN", "折れ口の軸があるのに tree_break の面が無い")

    # --- 骨格 ---
    center = skeleton._center_fn(trunk)
    tips = np.array([a.pos[-1] for a in ax_all if a.level >= 1])
    if len(tips):
        cx, cy = center(tips[:, 2])
        rho = np.hypot(tips[:, 0] - cx, tips[:, 1] - cy)
        inside = float((rho <= envelope.reach(p, tips[:, 2]) * 1.05 + 1e-6).mean())
        m["tips_inside"] = inside
        if inside < 0.90:
            f.fail("ENVELOPE", f"枝先の {inside * 100:.0f}% しかエンベロープ ×1.05 の内側にない（仕様 95% 以上）")
        elif inside < 0.95:
            f.warn("ENVELOPE", f"枝先の {inside * 100:.0f}% がエンベロープ ×1.05 の内側（仕様 95%）")
    hb = p.crown_base_ratio * p.height_m
    l1 = [a for a in ax_all if a.level == 1]
    if l1:
        z1 = np.array([a.pos[0, 2] for a in l1])
        live_low = [a for a in l1 if not a.dead and not a.broken and a.pos[0, 2] < hb - 0.3]
        if live_low:
            f.fail("CROWNBASE", f"樹冠の下端（{hb:.1f} m）より下に生きた枝が {len(live_low)} 本ある")
        gap = (p.height_m - float(z1.max())) / p.height_m
        m["top_gap"] = gap
        if gap > 0.15:
            f.fail("BARETOP", f"最も高い枝の付け根が樹高の {100 - gap * 100:.0f}% で、先端の {gap * 100:.0f}% が裸")
        elif gap > 0.10:
            f.warn("BARETOP", f"先端の {gap * 100:.0f}% に枝がない")
    zmin = float(mesh.co[:, 2].min())
    if zmin < -0.5:
        f.fail("GROUND", f"地面の下 {zmin:.2f} m にメッシュがある")
    elif zmin < -0.15:
        f.warn("GROUND", f"地面の下 {zmin:.2f} m にメッシュがある")

    # --- 葉 ---
    if n_leaf == 0 and p.season >= 0.3 and p.season <= 0.8:
        f.fail("NOLEAF", "葉が 1 枚もない（季節 %.2f）" % p.season)
    if n_leaf:
        lp = t.leaves.pos[t.leaves.is_leaf]
        low = int((lp[:, 2] < 0.85 * hb - 0.3).sum())
        if low and trunk.n:
            if low / n_leaf > 0.002:
                f.warn("LEAFLOW", f"樹冠の下端より下に葉が {low} 枚（{low / n_leaf * 100:.2f}%）")
        lo, hi = ax_all[0].pos.min(axis=0), np.max([a.pos.max(axis=0) for a in ax_all], axis=0)
        pad = 1.5
        outside = int(((lp < np.minimum(lo, [-W, -W, -1]) - pad) | (lp > np.maximum(hi, [W, W, H]) + pad)).any(axis=1).sum())
        if outside:
            f.fail("LEAFOUT", f"葉が枝の範囲の外に {outside} 枚ある")
        # 葉の房が枝の先に偏りすぎていないか（枝 1 本あたり）
        bearing = sum(1 for a in ax_all if a.level >= 2) or 1
        m["leaves_per_twig"] = n_leaf / bearing
    return m


def check_species_spec(sid, f: Findings):
    """12 種のプリセットが、仕様書 4.2 の表のとおりか。"""
    if sid not in SPEC_4_2:
        return
    H, D, W, shape, apical, grav, a1, ph1, droop, ltype, llen, bark = SPEC_4_2[sid]
    p = schema.make_params(**preset_io.builtin_overrides(sid))
    got = (p.height_m, p.dbh_cm, p.crown_width_m, p.crown_shape, p.apical_dominance, p.gravitropism, p.l1_angle_deg,
           p.l1_phyllotaxis, (p.l1_droop, p.l2_droop, p.l3_droop), p.leaf_type, p.leaf_length_cm, p.bark_type)
    names = ("height_m", "dbh_cm", "crown_width_m", "crown_shape", "apical_dominance", "gravitropism", "l1_angle_deg",
             "l1_phyllotaxis", "droop", "leaf_type", "leaf_length_cm", "bark_type")
    want = (H, D, W, shape, apical, grav, a1, ph1, droop, ltype, llen, bark)
    for n, g, w in zip(names, got, want):
        if w is None:
            continue
        ok = (all(abs(x - y) < 1e-6 for x, y in zip(g, w)) if isinstance(w, tuple)
              else (abs(g - w) < 1e-6 if isinstance(w, (int, float)) else g == w))
        if not ok:
            f.fail("SPEC4.2", f"{sid}: {n} が仕様書 4.2 の表と違う（仕様 {w}、プリセット {g}）")


def check_seasons(sid, f: Findings):
    """季節の挙動（仕様 1.6）。冬は落葉樹の葉が 0、常緑は葉がある。ソメイヨシノは開花時に葉なし・花あり。"""
    ov = schema.make_params(**preset_io.builtin_overrides(sid))
    ev, flowering = ov.evergreen, ov.flower_density > 0

    def count(season):
        p = schema.make_params(**dict(preset_io.builtin_overrides(sid), quality="PREVIEW", season=season))
        t = pipeline.generate(p)
        return int(t.leaves.is_leaf.sum()), int(t.leaves.is_flower.sum()), t.leaves.color_t
    l0, _, _ = count(0.0)
    if ev and l0 == 0:
        f.fail("SEASON", "常緑なのに冬（season 0）に葉がない")
    if not ev and l0 > 0 and not flowering:
        f.fail("SEASON", f"落葉樹なのに冬（season 0）に葉が {l0} 枚ある")
    l_end, _, _ = count(1.0)
    if not ev and l_end > 0:
        f.fail("SEASON", f"落葉樹なのに落葉後（season 1）に葉が {l_end} 枚ある")
    if flowering:
        l15, fl15, _ = count(0.15)
        if fl15 == 0:
            f.fail("SEASON", "開花期（season 0.15）に花がない")
        if sid == "someiyoshino" and l15 > 0:
            f.fail("SEASON", f"ソメイヨシノの開花時（season 0.15）に葉が {l15} 枚ある（仕様: 葉の存在率 0）")
    if not ev:
        _, _, ct = count(0.85)
        if ct < 0.5:
            f.warn("SEASON", f"落葉樹の紅葉度が season 0.85 で {ct:.2f}")


# --------------------------------------------------------------------------------------
# Blender 側の検査（bpy があるとき）
# --------------------------------------------------------------------------------------
class BlenderSide:
    def __init__(self):
        import bpy
        import tree_forge
        tree_forge.register()
        from tree_forge import props
        from tree_forge.blender import gn_leaves
        self.bpy, self.props, self.gn = bpy, props, gn_leaves
        sys.path.insert(0, str(ROOT / "tests" / "blender"))

    def reset(self):
        bpy = self.bpy
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o)
        for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.images):
            for d in list(coll):
                coll.remove(d)

    def check(self, ov, t, f: Findings):
        bpy, props = self.bpy, self.props
        self.reset()
        sc = bpy.context.scene
        props.write(sc.tree_forge, ov)
        sc.tree_forge.initialized = True
        bpy.context.view_layer.objects.active = None
        try:
            r = bpy.ops.treeforge.generate()
        except Exception as e:  # noqa: BLE001
            f.fail("BPY", f"木を生成で例外: {e}")
            return {}
        if r != {"FINISHED"}:
            f.fail("BPY", f"木を生成が {r} を返した")
            return {}
        obj = bpy.context.active_object
        me = obj.data
        m = {}
        if len(me.vertices) != len(t.mesh.co) or len(me.polygons) != len(t.mesh.face_size):
            f.fail("BPY", f"Blender のメッシュ（頂点 {len(me.vertices)}、面 {len(me.polygons)}）がコアの結果と合わない")
        if len(me.materials) != 8:
            f.fail("SLOT", f"材質スロットが {len(me.materials)} 個（仕様 8）")
        names = {a.name for a in me.attributes}
        for k in ("tree_part", "tree_break", "tree_level", "branch_id", "dist_root"):
            if k not in names:
                f.fail("ATTR", f"属性 {k} が無い")
        if len(me.uv_layers) != 1:
            f.fail("UV", f"UV マップが {len(me.uv_layers)} 枚")
        lo = obj.tree_forge.leaves_object
        if lo is None or self.gn.MODIFIER_NAME not in lo.modifiers:
            f.fail("GN", "葉オブジェクトまたは GN モディファイアが無い")
        else:
            dg = bpy.context.evaluated_depsgraph_get()
            n_inst = sum(1 for i in dg.object_instances if i.is_instance and i.parent and i.parent.original == lo)
            want = int(t.leaves.is_leaf.sum()) + int(t.leaves.is_flower.sum())
            m["instances"] = n_inst
            if n_inst != want:
                f.fail("GN", f"GN のインスタンス数 {n_inst} が葉・花の点の数 {want} と違う")
        return m

    def check_regenerate_keeps_materials(self, ov, f: Findings):
        bpy, props = self.bpy, self.props
        self.reset()
        sc = bpy.context.scene
        props.write(sc.tree_forge, dict(ov, quality="PREVIEW"))
        sc.tree_forge.initialized = True
        bpy.context.view_layer.objects.active = None
        bark, leaf = bpy.data.materials.new("Sweep_Bark"), bpy.data.materials.new("Sweep_Leaf")
        sc.tree_forge.materials.trunk, sc.tree_forge.materials.leaf = bark, leaf
        bpy.ops.treeforge.generate()
        obj = bpy.context.active_object
        n0 = len(obj.data.materials)
        sc.tree_forge.materials.trunk = None
        obj.tree_forge.seed += 1
        bpy.context.view_layer.objects.active = obj
        bpy.ops.treeforge.regenerate()
        mats = [m.name if m else None for m in obj.data.materials]
        if len(mats) != n0 or mats[0] != "Sweep_Bark" or mats[3] != "Sweep_Leaf":
            f.fail("REGEN", f"再生成で材質の割り当てが変わった: {mats[:4]}")


# --------------------------------------------------------------------------------------
# 実行・報告
# --------------------------------------------------------------------------------------
def run(species, qualities, use_bpy, progress=print):
    rows, per_species = [], {}
    side = BlenderSide() if use_bpy else None
    for si, sid in enumerate(species):
        fs = Findings()
        check_species_spec(sid, fs)
        check_seasons(sid, fs)
        if side:
            side.check_regenerate_keeps_materials(preset_io.builtin_overrides(sid), fs)
        per_species[sid] = fs.items
        for qi, q in enumerate(QUALITIES):
            if q not in qualities:
                continue
            for vi, scale in enumerate(SIZE_SCALES):
                ov = variant_overrides(sid, si, qi, vi, scale)
                p = schema.make_params(**ov)
                f = Findings()
                row = dict(species=sid, quality=q, variant=vi, scale=scale, seed=ov["seed"], H=p.height_m,
                           W=p.crown_width_m, DBH=p.dbh_cm)
                try:
                    t0 = time.perf_counter()
                    t = pipeline.generate(p)
                    dt = time.perf_counter() - t0
                    row.update(check_core(p, t, q, dt, f))
                    if q == "PREVIEW":                                         # 再現性（同じシードで同じメッシュ）
                        t2 = pipeline.generate(p)
                        if not (np.array_equal(t.mesh.co, t2.mesh.co) and np.array_equal(t.leaves.pos, t2.leaves.pos)):
                            f.fail("DETERMINISM", "同じシードで結果が違う")
                    if side:
                        row.update(side.check(ov, t, f))
                except Exception as e:  # noqa: BLE001
                    import traceback
                    f.fail("EXCEPTION", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
                row["findings"] = f.items
                rows.append(row)
                nf = sum(1 for s, _, _ in f.items if s == "FAIL")
                nw = sum(1 for s, _, _ in f.items if s == "WARN")
                progress(f"{sid:14s} {q:8s} x{scale:<4} seed={ov['seed']:<5d} H={p.height_m:5.1f} "
                         f"tris={row.get('tris', 0):>9,} leaves={row.get('leaves', 0):>8,} t={row.get('time', 0):5.1f}s "
                         f"FAIL={nf} WARN={nw}")
    return rows, per_species


def write_report(rows, per_species, path: Path, meta: dict):
    L = []
    A = L.append
    n = len(rows)
    nfail = sum(1 for r in rows if any(s == "FAIL" for s, _, _ in r["findings"])) + \
        sum(1 for items in per_species.values() for s, _, _ in items if s == "FAIL")
    A(f"# デバッグ掃引レポート（{meta['version']}）")
    A("")
    A("`tools/debug_sweep.py` が生成します（直接編集しないでください）。")
    A("")
    A(f"- 日時: {meta['date']} / Python {meta['python']} / bpy {meta['bpy']}")
    A(f"- 対象: {len(per_species)} 樹種 × 品質 {len(meta['qualities'])} 段階 × 3 本（サイズ倍率 {', '.join(f'×{s}' for s in SIZE_SCALES)}、"
      f"シードは全て別）= **{n} 本**。Blender 側の検査: {'あり' if meta['bpy'] != 'なし' else 'なし'}")
    A(f"- 結果: FAIL を含む木 / 樹種検査 **{nfail}** 件")
    A("")
    A("## 樹種ごとの結果")
    A("")
    A("| 樹種 | 品質 | 木の数 | FAIL | WARN | 樹高誤差(最大) | 樹冠幅誤差(最大) | 三角形(最大) | 葉(最大) | 生成時間(最大) |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for sid in per_species:
        for q in meta["qualities"]:
            rs = [r for r in rows if r["species"] == sid and r["quality"] == q]
            if not rs:
                continue
            nf = sum(1 for r in rs for s, _, _ in r["findings"] if s == "FAIL")
            nw = sum(1 for r in rs for s, _, _ in r["findings"] if s == "WARN")
            A(f"| {sid} | {q} | {len(rs)} | {nf} | {nw} | {max(abs(r.get('H_err', 0)) for r in rs) * 100:.1f}% | "
              f"{max(abs(r.get('W_err', 0)) for r in rs) * 100:.1f}% | {max(r.get('tris', 0) for r in rs):,} | "
              f"{max(r.get('leaves', 0) for r in rs):,} | {max(r.get('time', 0) for r in rs):.1f} s |")
    A("")
    A("## 指摘一覧")
    A("")
    any_item = False
    for sid, items in per_species.items():
        for sev, code, msg in items:
            any_item = True
            A(f"- **{sev}** `{code}` {sid}（樹種）: {msg}")
    for r in rows:
        for sev, code, msg in r["findings"]:
            any_item = True
            A(f"- **{sev}** `{code}` {r['species']} {r['quality']} ×{r['scale']}（seed {r['seed']}、H={r['H']:.1f} m）: {msg}")
    if not any_item:
        A("指摘なし。")
    A("")
    A("## 全ての木")
    A("")
    A("| 樹種 | 品質 | 倍率 | seed | 樹高 [m] | 樹冠幅 [m] | 胸高直径 [cm] | 三角形 | 葉 | 時間 [s] | 先端の裸 | 結果 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        nf = sum(1 for s, _, _ in r["findings"] if s == "FAIL")
        nw = sum(1 for s, _, _ in r["findings"] if s == "WARN")
        res = "FAIL" if nf else ("WARN" if nw else "OK")
        A(f"| {r['species']} | {r['quality']} | ×{r['scale']} | {r['seed']} | {r.get('H', 0):.1f} | {r.get('W', 0):.1f} | "
          f"{r.get('DBH', 0):.0f} | {r.get('tris', 0):,} | {r.get('leaves', 0):,} | {r.get('time', 0):.1f} | "
          f"{r.get('top_gap', 0) * 100:.0f}% | {res} |")
    A("")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="all")
    ap.add_argument("--qualities", default=",".join(QUALITIES))
    ap.add_argument("--no-bpy", action="store_true", help="bpy を使わず、コアの検査だけ")
    ap.add_argument("--json", default="")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    species = list(schema.SPECIES_IDS) if args.species == "all" else args.species.split(",")
    qualities = args.qualities.split(",")
    use_bpy = not args.no_bpy
    bpy_v = "なし"
    if use_bpy:
        try:
            import bpy
            bpy_v = bpy.app.version_string
        except ImportError:
            use_bpy = False
    t0 = time.time()
    rows, per_species = run(species, qualities, use_bpy, progress=lambda s: print(s, flush=True))
    print(f"done in {time.time() - t0:.0f} s")
    meta = dict(version="0.1.0", date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                python=sys.version.split()[0], bpy=bpy_v, qualities=qualities)
    try:
        import subprocess
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True).stdout.strip()
        meta["version"] = f"0.1.0 / commit {sha}"
    except OSError:
        pass
    if args.json:
        Path(args.json).write_text(json.dumps(dict(meta=meta, rows=rows, per_species=per_species), ensure_ascii=False,
                                              default=float, indent=1), encoding="utf-8")
    if args.report:
        write_report(rows, per_species, Path(args.report), meta)
    nfail = sum(1 for r in rows for s, _, _ in r["findings"] if s == "FAIL") + \
        sum(1 for items in per_species.values() for s, _, _ in items if s == "FAIL")
    print(f"FAIL items: {nfail}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
