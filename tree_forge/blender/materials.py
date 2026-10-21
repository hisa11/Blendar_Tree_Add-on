"""部位別マテリアル。

スロット順は固定: 0 幹 1 枝 2 小枝 3 葉 4 花 5 果実 6 根張り 7 折れ口。
面の整数属性 tree_part も持つので、スロット順が崩れても復元できる。
未指定の部位は同梱の簡易マテリアルで埋める（根張り→幹、小枝→枝は指定済みの材質を優先して流用）。
"""
from __future__ import annotations

import os
import re

import bpy

from .. import compat
from ..core import texture_names
from ..props import PARTS

KEYWORDS = {
    "trunk": ("bark", "trunk", "樹皮", "幹"),
    "branch": ("branch", "枝"),
    "twig": ("twig", "小枝"),
    "leaf": ("leaf", "leaves", "foliage", "葉"),
    "flower": ("flower", "blossom", "petal", "花"),
    "fruit": ("fruit", "berry", "実"),
    "root_flare": ("root", "根"),
    "broken": ("broken", "wood", "cut", "折れ"),
}
FALLBACK = {"root_flare": "trunk", "twig": "branch"}


def _new(name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    compat.ensure_node_tree(mat)
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (400, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, bsdf, out


def _rgba(c, a=1.0):
    return (c[0], c[1], c[2], a)


def _bark(name, base, dark, scale):
    mat, nt, bsdf, _ = _new(name)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-800, 0)
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.location = (-600, 0)
    mp.inputs["Scale"].default_value = (scale, scale * 0.35, 1.0)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-400, 0)
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-150, 100)
    ramp.color_ramp.elements[0].color = _rgba(dark)
    ramp.color_ramp.elements[1].color = _rgba(base)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (150, -200)
    bump.inputs["Strength"].default_value = 0.6
    nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def _plain(name, color, rough=0.7):
    mat, nt, bsdf, _ = _new(name)
    bsdf.inputs["Base Color"].default_value = _rgba(color)
    bsdf.inputs["Roughness"].default_value = rough
    return mat


def leaf_material(name, color, card_mask=False, translucent=True):
    """葉: インスタンスごとの乱数 leaf_var で明暗をばらす。5.2 以降は Thin Wall、それ以前は Translucent を混ぜる。"""
    mat, nt, bsdf, out = _new(name)
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.location = (-500, 100)
    attr.attribute_type = "INSTANCER"
    attr.attribute_name = "leaf_var"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-250, 100)
    ramp.color_ramp.elements[0].color = _rgba([c * 0.65 for c in color])
    ramp.color_ramp.elements[1].color = _rgba([min(1.0, c * 1.3) for c in color])
    nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.5
    if card_mask:                              # UV の楕円で葉の輪郭を切り抜く（CARD 用）
        tc = nt.nodes.new("ShaderNodeTexCoord")
        tc.location = (-700, -250)
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.location = (-500, -250)
        mp.inputs["Location"].default_value = (-0.5, -0.5, 0.0)
        mp.inputs["Scale"].default_value = (2.0, 2.0, 1.0)
        gr = nt.nodes.new("ShaderNodeTexGradient")
        gr.location = (-300, -250)
        gr.gradient_type = "SPHERICAL"
        th = nt.nodes.new("ShaderNodeMath")
        th.location = (-100, -250)
        th.operation = "GREATER_THAN"
        th.inputs[1].default_value = 0.18
        nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], gr.inputs["Vector"])
        nt.links.new(gr.outputs["Fac"], th.inputs[0])
        nt.links.new(th.outputs["Value"], bsdf.inputs["Alpha"])
    if translucent:
        if compat.has_thin_wall(bsdf):
            bsdf.inputs["Thin Wall"].default_value = True
            if "Transmission Weight" in bsdf.inputs:
                bsdf.inputs["Transmission Weight"].default_value = 0.35
        else:
            tr = nt.nodes.new("ShaderNodeBsdfTranslucent")
            tr.location = (400, -250)
            nt.links.new(ramp.outputs["Color"], tr.inputs["Color"])
            mix = nt.nodes.new("ShaderNodeMixShader")
            mix.location = (600, 0)
            mix.inputs["Fac"].default_value = 0.3
            nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
            nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
            nt.links.remove(out.inputs["Surface"].links[0])
            nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
            out.location = (850, 0)
    return mat


def default_material(part, ns=None, tree_name="", leaf_color=None):
    """同梱の簡易マテリアル。木ごとに色が変わる葉・花は木の名前つき。"""
    if part in ("trunk", "root_flare"):
        return _bark("TF_Bark", (0.22, 0.15, 0.10), (0.09, 0.06, 0.04), 6.0)
    if part == "branch":
        return _bark("TF_Branch", (0.20, 0.14, 0.09), (0.08, 0.055, 0.035), 9.0)
    if part == "twig":
        return _bark("TF_Twig", (0.17, 0.12, 0.08), (0.07, 0.05, 0.03), 12.0)
    if part == "broken":
        return _plain("TF_BrokenWood", (0.62, 0.50, 0.32), 0.8)
    if part == "fruit":
        return _plain("TF_Fruit", (0.55, 0.05, 0.04), 0.4)
    if part == "flower":
        col = tuple(ns.flower_color) if ns is not None else (0.96, 0.72, 0.80)
        return leaf_material(f"TF_Flower.{tree_name}" if tree_name else "TF_Flower", col)
    if part == "leaf":
        col = leaf_color or (tuple(ns.leaf_color) if ns is not None else (0.10, 0.28, 0.05))
        card = ns is not None and ns.leaf_geom == "CARD"
        return leaf_material(f"TF_Leaf.{tree_name}" if tree_name else "TF_Leaf", col, card_mask=card)
    raise ValueError(part)


def season_color(ns, color_t):
    a, b = ns.leaf_color, ns.autumn_color
    return tuple(a[i] + (b[i] - a[i]) * color_t for i in range(3))


def resolve(pg, part):
    """ユーザー指定の材質（なければ代替部位の指定）。どちらも無ければ None。"""
    m = getattr(pg.materials, part)
    if m is None and part in FALLBACK:
        m = getattr(pg.materials, FALLBACK[part])
    return m


def apply_all(obj, pg, ns, color_t=0.0):
    """スロット 0〜7 を作り直す。葉・花の GN も同じ材質に更新する。"""
    from . import gn_leaves
    mats = {}
    for part in PARTS:
        m = resolve(pg, part)
        if m is None:
            leaf_col = season_color(ns, color_t) if part == "leaf" else None
            m = default_material(part, ns, obj.name, leaf_col)
        mats[part] = m
    me = obj.data
    # clear() すると全ポリゴンの material_index が 0 に戻るので、スロットは消さずにインデックスで差し替える
    while len(me.materials) < len(PARTS):
        me.materials.append(None)
    while len(me.materials) > len(PARTS):
        me.materials.pop(index=len(me.materials) - 1)
    for i, part in enumerate(PARTS):
        me.materials[i] = mats[part]
    leaves = pg.leaves_object
    if leaves is not None:
        mod = leaves.modifiers.get(gn_leaves.MODIFIER_NAME)
        if mod is not None and mod.node_group is not None:
            gn_leaves.set_materials(mod.node_group, mats["leaf"], mats["flower"])
            leaves.data.materials.clear()
            leaves.data.materials.append(mats["leaf"])
    return mats


def auto_assign(pg) -> dict:
    """マテリアル名のキーワード照合で各部位に割り当てる。一致数が最大の材質を採用。"""
    found = {}
    cands = [m for m in bpy.data.materials if not m.name.startswith("TF_")]
    for part in PARTS:
        best, best_n = None, 0
        for m in cands:
            low = m.name.lower()
            n = sum(1 for k in KEYWORDS[part] if k in low)
            if n > best_n:
                best, best_n = m, n
        if best is not None:
            setattr(pg.materials, part, best)
            found[part] = best.name
    return found


def append_from_blend(path, names=None, link=False):
    """.blend から Material を取り込む。names が None なら全て。取り込んだ材質のリストを返す。"""
    with bpy.data.libraries.load(path, link=link) as (src, dst):
        pick = [n for n in src.materials if names is None or n in names]
        dst.materials = pick
    return [m for m in dst.materials if m is not None]


def _tex(nt, path, x, y, srgb, label):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.location = (x, y)
    n.image = bpy.data.images.load(path, check_existing=True)
    n.image.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
    n.label = label
    return n


def build_from_folder(folder, name=None, unknown_normal_is_directx=False):
    """フォルダ内のテクスチャをファイル名で判別して Principled BSDF を組む。(Material, 判別結果) を返す。"""
    files = sorted(f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    found = texture_names.scan(files)
    if "albedo" not in found:
        raise ValueError("アルベド(diffuse/albedo/basecolor/color)に当たるファイルが見つかりません")
    name = name or re.sub(r"[_\-. ]+(diff|diffuse|albedo|basecolor|color|col)\b.*$", "", found["albedo"][0],
                          flags=re.I).strip() or os.path.basename(folder.rstrip("/\\"))
    mat, nt, bsdf, out = _new(name)
    p = lambda k: os.path.join(folder, found[k][0])      # noqa: E731
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-900, 0)
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.location = (-700, 0)
    nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])

    cache = {}

    def tex(kind, x, y, srgb):
        """同じ種別は1枚だけ読み込む（ARM を AO と粗さ・金属で共有するため）。"""
        if kind not in cache:
            t = _tex(nt, p(kind), x, y, srgb, kind)
            nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
            cache[kind] = t
        return cache[kind]

    base = tex("albedo", -450, 300, True)
    color_out = base.outputs["Color"]
    if "ao" in found or "arm" in found:
        ao_t = tex("ao", -450, 0, False) if "ao" in found else None
        arm_t = tex("arm", -450, -300, False) if "arm" in found else None
        mul = nt.nodes.new("ShaderNodeMix")
        mul.data_type = "RGBA"
        mul.blend_type = "MULTIPLY"
        mul.location = (-150, 300)
        mul.inputs["Factor"].default_value = 1.0
        a_in = [s for s in mul.inputs if s.name == "A" and s.type == "RGBA"][0]
        b_in = [s for s in mul.inputs if s.name == "B" and s.type == "RGBA"][0]
        nt.links.new(color_out, a_in)
        if ao_t is not None:
            nt.links.new(ao_t.outputs["Color"], b_in)
        else:
            sep = nt.nodes.new("ShaderNodeSeparateColor")
            sep.location = (-150, -300)
            nt.links.new(arm_t.outputs["Color"], sep.inputs["Color"])
            nt.links.new(sep.outputs["Red"], b_in)
        color_out = [s for s in mul.outputs if s.type == "RGBA"][0]
    nt.links.new(color_out, bsdf.inputs["Base Color"])
    if "roughness" in found:
        nt.links.new(tex("roughness", -450, -150, False).outputs["Color"], bsdf.inputs["Roughness"])
    elif "arm" in found:
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-150, -400)
        arm_t = tex("arm", -450, -300, False)
        nt.links.new(arm_t.outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    if "metallic" in found:
        nt.links.new(tex("metallic", -450, -600, False).outputs["Color"], bsdf.inputs["Metallic"])
    if "normal" in found:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (100, -200)
        conv = found["normal"][1]
        directx = (conv == "DX") or (conv is None and unknown_normal_is_directx)
        if not compat.set_normal_map_directx(nm, directx) and directx:
            # convention プロパティが無い版: G を反転して OpenGL 相当にする
            sep = nt.nodes.new("ShaderNodeSeparateColor")
            comb = nt.nodes.new("ShaderNodeCombineColor")
            inv = nt.nodes.new("ShaderNodeMath")
            inv.operation = "SUBTRACT"
            inv.inputs[0].default_value = 1.0
            t = tex("normal", -450, -750, False)
            nt.links.new(t.outputs["Color"], sep.inputs["Color"])
            nt.links.new(sep.outputs["Red"], comb.inputs["Red"])
            nt.links.new(sep.outputs["Green"], inv.inputs[1])
            nt.links.new(inv.outputs["Value"], comb.inputs["Green"])
            nt.links.new(sep.outputs["Blue"], comb.inputs["Blue"])
            nt.links.new(comb.outputs["Color"], nm.inputs["Color"])
        else:
            nt.links.new(tex("normal", -450, -750, False).outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if "displacement" in found:
        disp = nt.nodes.new("ShaderNodeDisplacement")
        disp.location = (400, -400)
        disp.inputs["Scale"].default_value = 0.02
        nt.links.new(tex("displacement", -450, -900, False).outputs["Color"], disp.inputs["Height"])
        nt.links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
    if "opacity" in found:
        nt.links.new(tex("opacity", -450, -1050, False).outputs["Color"], bsdf.inputs["Alpha"])
    if "translucency" in found and "Transmission Weight" in bsdf.inputs:
        nt.links.new(tex("translucency", -450, -1200, False).outputs["Color"], bsdf.inputs["Transmission Weight"])
    return mat, {k: v[0] for k, v in found.items()}
