"""Blender 内での結合テスト。

    python tests/blender/test_addon.py                      # bpy モジュール
    blender -b --factory-startup --python-exit-code 1 --python tests/blender/test_addon.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS, check, register_addon, reset_scene, run_all  # noqa: E402

import bpy  # noqa: E402
import numpy as np  # noqa: E402

register_addon()
from tree_forge import compat, preset_io, props, schema  # noqa: E402
from tree_forge.blender import gn_leaves, materials  # noqa: E402
from tree_forge.core import allometry  # noqa: E402


def new_tree(species="keyaki", **ov):
    reset_scene()
    sc = bpy.context.scene
    d = preset_io.builtin_overrides(species)
    d.update(ov)
    props.write(sc.tree_forge, d)
    sc.tree_forge.initialized = True
    sc.tree_forge.quality = d.get("quality", "PREVIEW")
    assert bpy.ops.treeforge.generate() == {"FINISHED"}
    return bpy.context.active_object


def eval_instances(leaves_obj):
    dg = bpy.context.evaluated_depsgraph_get()
    return sum(1 for i in dg.object_instances if i.is_instance and i.parent and i.parent.original == leaves_obj)


@check("register/unregister が往復できる")
def _():
    import tree_forge
    tree_forge.unregister()
    tree_forge.register()
    assert hasattr(bpy.types.Object, "tree_forge") and hasattr(bpy.types.Scene, "tree_forge")


@check("生成: 8スロット・属性・UV・葉のインスタンスが評価される")
def _():
    obj = new_tree()
    me = obj.data
    assert [m.name for m in me.materials][:3] == ["TF_Bark", "TF_Branch", "TF_Twig"] and len(me.materials) == 8
    names = {a.name for a in me.attributes}
    for k in ("tree_part", "tree_break", "tree_level", "dist_root", "dist_root_n", "branch_t", "branch_phase",
              "branch_pivot", "branch_id", "junction_blend"):
        assert k in names, k
    assert me.attributes["tree_part"].domain == "FACE" and me.attributes["branch_pivot"].data_type == "FLOAT_VECTOR"
    assert len(me.uv_layers) == 1 and len(me.polygons) > 1000
    idx = np.zeros(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", idx)
    tp = np.zeros(len(me.polygons), np.int32)
    me.attributes["tree_part"].data.foreach_get("value", tp)
    brk = np.zeros(len(me.polygons), np.int32)
    me.attributes["tree_break"].data.foreach_get("value", brk)
    assert len(np.unique(idx)) >= 3, "スロット割り当てが面に残っている"
    assert np.array_equal(idx, np.where(brk > 0, 7, tp)), "material_index は tree_part（折れ口は7）と一致する"
    lo = obj.tree_forge.leaves_object
    assert lo.parent == obj and lo.modifiers[gn_leaves.MODIFIER_NAME].node_group is not None
    n_leaf = len(lo.data.vertices)
    assert n_leaf > 500
    assert eval_instances(lo) == int(np.sum([v.value for v in lo.data.attributes["is_leaf"].data]))


@check("葉の点群属性（回転は四元数・単位長）")
def _():
    obj = new_tree(season=0.5)
    me = obj.tree_forge.leaves_object.data
    q = np.zeros(len(me.vertices) * 4, np.float32)
    me.attributes["leaf_rot"].data.foreach_get("value", q)
    q = q.reshape(-1, 4)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-4)
    assert me.attributes["leaf_rot"].data_type == "QUATERNION"


@check("モディファイア入力を使わない（5.2 の API 変更に非依存）")
def _():
    obj = new_tree()
    mod = obj.tree_forge.leaves_object.modifiers[gn_leaves.MODIFIER_NAME]
    ng = mod.node_group
    ins = [i for i in ng.interface.items_tree if i.item_type == "SOCKET" and i.in_out == "INPUT"]
    assert [i.socket_type for i in ins] == ["NodeSocketGeometry"], "入力は Geometry のみ"
    try:
        keys = list(mod.keys())          # 5.1 まで: modifier["Socket_n"]。5.2 以降は IDProperties を持たない
    except TypeError:
        keys = []
    assert not [k for k in keys if k.startswith("Socket_")]


@check("12樹種すべてが PREVIEW で例外なく生成できる")
def _():
    for sid in schema.SPECIES_IDS:
        obj = new_tree(sid, season=0.5)
        assert len(obj.data.polygons) > 500, sid
        assert obj.data.attributes["tree_part"] is not None


@check("再生成: オブジェクト・モディファイア・材質割り当て・トランスフォームが保持される")
def _():
    obj = new_tree()
    pg = obj.tree_forge
    custom = bpy.data.materials.new("MyBark")
    pg.materials.trunk = custom
    pg.materials.leaf = bpy.data.materials.new("MyLeaf")
    obj.location = (3, 2, 1)
    materials.apply_all(obj, pg, props.read(pg))
    lo = pg.leaves_object
    mod_before = lo.modifiers[gn_leaves.MODIFIER_NAME]
    pg.seed = 777
    assert bpy.ops.treeforge.regenerate() == {"FINISHED"}
    assert bpy.context.active_object == obj and tuple(obj.location) == (3, 2, 1)
    assert pg.leaves_object == lo and lo.modifiers[gn_leaves.MODIFIER_NAME].name == mod_before.name
    assert obj.data.materials[0] == custom and obj.data.materials[6] == custom        # 根張り→幹の代替
    assert obj.data.materials[3].name == "MyLeaf"
    ng = lo.modifiers[gn_leaves.MODIFIER_NAME].node_group
    leaf_set = [n for n in ng.nodes if n.bl_idname == "GeometryNodeSetMaterial"]
    assert any(n.inputs["Material"].default_value.name == "MyLeaf" for n in leaf_set)
    assert len(bpy.data.objects) == len({o.name for o in bpy.data.objects})
    assert eval_instances(lo) > 0


@check("再生成でメッシュが増殖しない（古い Mesh を削除）")
def _():
    obj = new_tree()
    before = len(bpy.data.meshes)
    for s in (2, 3, 4):
        obj.tree_forge.seed = s
        bpy.ops.treeforge.regenerate()
    assert len(bpy.data.meshes) == before


@check("幹の太さ: CUSTOM プロファイルが Blender のメッシュに反映される")
def _():
    pts = [(0.0, 100.0), (0.1, 70.0), (0.5, 30.0), (1.0, 4.0)]
    obj = new_tree("sugi", trunk_profile_mode="CUSTOM", profile=pts, trunk_irregularity=0.0, burl_density=0.0,
                   bark_ridge_mm=0.0, root_flare_scale=1.0)
    pg = obj.tree_forge
    assert pg.trunk_profile_mode == "CUSTOM" and [(round(p.height_ratio, 3), p.diameter_cm) for p in pg.profile] == pts
    bid = np.zeros(len(obj.data.vertices), np.int32)
    obj.data.attributes["branch_id"].data.foreach_get("value", bid)
    co = np.zeros(len(obj.data.vertices) * 3, np.float32)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    trunk = co[bid == 0]
    H = pg.height_m
    for ratio, d in pts[1:3]:
        band = trunk[np.abs(trunk[:, 2] - ratio * H) < 0.35]
        r = np.linalg.norm(band[:, :2] - band[:, :2].mean(axis=0), axis=1).mean() * 200
        assert abs(r - d) / d < 0.08, (ratio, d, r)
    # UI 操作: 点の追加・削除・初期化
    n = len(pg.profile)
    bpy.ops.treeforge.profile_add()
    assert len(pg.profile) == n + 1
    bpy.ops.treeforge.profile_remove()
    assert len(pg.profile) == n
    bpy.ops.treeforge.profile_reset()
    assert len(pg.profile) == 6 and pg.trunk_profile_mode == "CUSTOM" or len(pg.profile) == 6


@check("幹の太さ: AUTO→CUSTOM 切替で現在の曲線から初期点ができる")
def _():
    obj = new_tree()
    pg = obj.tree_forge
    assert len(pg.profile) == 0
    pg.trunk_profile_mode = "CUSTOM"
    assert len(pg.profile) == 6
    d0 = pg.profile[0].diameter_cm
    ns = props.read(pg)
    assert abs(allometry.trunk_radius_fn(ns)(0.0) * 200 - d0) < 1.0


@check("折れ枝: 折れ口スロット(7)の面ができ、損傷0なら無い")
def _():
    obj = new_tree(broken_branch_ratio=0.5, quality="MEDIUM")
    idx = np.zeros(len(obj.data.polygons), np.int32)
    obj.data.polygons.foreach_get("material_index", idx)
    assert (idx == 7).sum() > 0
    brk = np.zeros(len(obj.data.polygons), np.int32)
    obj.data.attributes["tree_break"].data.foreach_get("value", brk)
    assert brk.sum() == (idx == 7).sum()
    obj2 = new_tree(broken_branch_ratio=0.0, top_break_height=0.0)
    idx = np.zeros(len(obj2.data.polygons), np.int32)
    obj2.data.polygons.foreach_get("material_index", idx)
    assert (idx == 7).sum() == 0


@check("プリセット: 樹種を変えると値が入れ替わる / 保存→読込で太さ点列も戻る")
def _():
    obj = new_tree()
    pg = obj.tree_forge
    pg.species = "sugi"
    assert pg.height_m == 30 and pg.crown_shape == "conical" and pg.leaf_type == "NEEDLE_SPIRAL"
    pg.trunk_profile_mode = "CUSTOM"
    props.set_profile(pg, [(0, 90), (0.5, 33), (1, 5)])
    pg.height_m = 27.5
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.json"
        assert bpy.ops.treeforge.preset_export(filepath=str(path)) == {"FINISHED"}
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1 and data["params"]["height_m"] == 27.5
        pg.species = "keyaki"
        assert pg.height_m == 20 and len(pg.profile) == 0
        assert bpy.ops.treeforge.preset_load(filepath=str(path)) == {"FINISHED"}
    assert pg.height_m == 27.5 and pg.trunk_profile_mode == "CUSTOM"
    assert [(round(p.height_ratio, 2), p.diameter_cm) for p in pg.profile] == [(0, 90), (0.5, 33), (1, 5)]
    assert bpy.context.active_object == obj and obj.data.polygons


@check("マテリアル: 名前から自動割り当て")
def _():
    obj = new_tree()
    for n in ("Oak_Bark_01", "Oak_Branch", "Oak_Leaf", "Pink_Blossom", "Red_Berry", "Unrelated"):
        bpy.data.materials.new(n)
    found = materials.auto_assign(obj.tree_forge)
    assert found["trunk"] == "Oak_Bark_01" and found["branch"] == "Oak_Branch" and found["leaf"] == "Oak_Leaf"
    assert found["flower"] == "Pink_Blossom" and found["fruit"] == "Red_Berry"


def make_texture(path, color=(0.5, 0.5, 0.5, 1.0)):
    img = bpy.data.images.new(os.path.basename(path), 4, 4, alpha=True)
    img.pixels = list(color) * 16
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


@check("マテリアル: フォルダから構築（法線 DirectX 切替・ARM・変位・不透明度）")
def _():
    new_tree()
    with tempfile.TemporaryDirectory() as d:
        for n in ("Bark_BaseColor", "Bark_nor_dx", "Bark_ARM", "Bark_Displacement", "Bark_Opacity", "Bark_AO"):
            make_texture(os.path.join(d, n + ".png"))
        mat, found = materials.build_from_folder(d)
        assert set(found) == {"albedo", "normal", "arm", "displacement", "opacity", "ao"}
        types = [n.bl_idname for n in mat.node_tree.nodes]
        assert types.count("ShaderNodeTexImage") == 6
        nm = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeNormalMap")
        assert nm.convention == "DIRECTX"
        disp = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeDisplacement")
        out = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
        assert out.inputs["Displacement"].is_linked and disp.outputs["Displacement"].links
        bsdf = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
        assert bsdf.inputs["Alpha"].is_linked and bsdf.inputs["Roughness"].is_linked and bsdf.inputs["Metallic"].is_linked
        imgs = {i.name: i.colorspace_settings.name for i in bpy.data.images}
        assert imgs["Bark_BaseColor.png"] == "sRGB" and imgs["Bark_nor_dx.png"] == "Non-Color"
    with tempfile.TemporaryDirectory() as d:
        make_texture(os.path.join(d, "x_diff.png"))
        make_texture(os.path.join(d, "x_normal_gl.png"))
        mat, _ = materials.build_from_folder(d)
        nm = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeNormalMap")
        assert nm.convention == "OPENGL"
    with tempfile.TemporaryDirectory() as d:
        make_texture(os.path.join(d, "readme.png"))
        try:
            materials.build_from_folder(d)
        except ValueError:
            pass
        else:
            raise AssertionError("アルベドが無ければエラー")


@check("葉マテリアル: 5.2 以降は Thin Wall、それ以前は Translucent を混ぜる")
def _():
    new_tree()
    mat = materials.leaf_material("T_leaf", (0.1, 0.3, 0.05))
    bsdf = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    kinds = [n.bl_idname for n in mat.node_tree.nodes]
    if compat.has_thin_wall(bsdf):
        assert bsdf.inputs["Thin Wall"].default_value is True and "ShaderNodeBsdfTranslucent" not in kinds
    else:
        assert "ShaderNodeBsdfTranslucent" in kinds and "ShaderNodeMixShader" in kinds


@check("書き出し用に実体化: 葉が実メッシュになり、元の木は変わらない")
def _():
    obj = new_tree(season=0.5)
    n_obj = len(bpy.data.objects)
    assert bpy.ops.treeforge.realize_for_export() == {"FINISHED"}
    made = [o for o in bpy.data.objects if o.name.endswith("_export") or o.name.endswith("_leaves_export")]
    assert len(made) == 2 and len(bpy.data.objects) == n_obj + 2
    leaves = next(o for o in made if o.name.endswith("_leaves_export"))
    assert len(leaves.data.polygons) > 1000 and len(leaves.data.materials) >= 1
    assert not leaves.modifiers
    assert obj.tree_forge.leaves_object.modifiers[gn_leaves.MODIFIER_NAME].node_group.name.endswith(obj.name)


@check("花: 開花期は花のインスタンスが評価される")
def _():
    obj = new_tree("someiyoshino", season=0.15)
    lo = obj.tree_forge.leaves_object
    isf = np.array([v.value for v in lo.data.attributes["is_flower"].data])
    assert isf.sum() > 100
    assert eval_instances(lo) == int(isf.sum())
    assert obj.tree_forge.flower_src is not None


@check("ひな形の初期化: 起動直後でもケヤキのプリセットが入る")
def _():
    reset_scene()
    sc = bpy.context.scene
    sc.tree_forge.initialized = False
    sc.tree_forge.species = "keyaki"
    props.ensure_template(sc)
    assert sc.tree_forge.height_m == 20 and sc.tree_forge.trunk_split_count == 3


@check("葉のまとまり・拡大率: 葉ソースの三角形数と点の拡大率に反映される")
def _():
    a = new_tree(season=0.5, leaf_cluster=1)
    n1 = len(a.tree_forge.leaf_src.data.polygons)
    b = new_tree(season=0.5, leaf_cluster=5)
    n5 = len(b.tree_forge.leaf_src.data.polygons)
    assert n5 >= 2.5 * n1
    sc = np.zeros(len(b.tree_forge.leaves_object.data.vertices) * 3, np.float32)
    b.tree_forge.leaves_object.data.attributes["leaf_scale"].data.foreach_get("vector", sc)
    assert abs(sc.reshape(-1, 3)[:, 0].mean() - 1.0) < 0.2


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
