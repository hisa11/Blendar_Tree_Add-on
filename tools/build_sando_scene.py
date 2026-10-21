"""参道（神社の表参道）の横に立つ巨大スギのシーンを作って描画する。

    python tools/build_sando_scene.py --out sando.png                       # bpy モジュールで
    blender -b --factory-startup --python tools/build_sando_scene.py -- --out sando.png

  --view wide|low|crown   wide: 参道の入口から見上げる全景（縦長） / low: 幹の根元近くから見上げる / crown: 樹冠の拡大
  --seed N            巨大スギの乱数シード
  --samples N         Cycles のサンプル数（既定 96）   --scale 0.5  解像度の倍率
  --save-blend path   .blend も保存する

シーン（単位 m）: 参道は +Y 方向に伸びる幅 約 6 m（中央に石畳、両脇に砂利と縁石）。主役の巨大スギは参道の右（x=+7）。
両側に杉並木、石灯籠、遠景に鳥居、霧、人物（スケール確認用）。
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np

import bmesh  # noqa: E402  (bpy より後に import する必要がある)
from mathutils import Euler, Matrix, Vector

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "tree_forge").is_dir() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------------------
# アドオンの読み込み（開発ツリーでも、拡張機能としてインストール済みでも動く）
# --------------------------------------------------------------------------------------
def load_addon():
    if not hasattr(bpy.types.Scene, "tree_forge"):
        import tree_forge
        tree_forge.register()
    pkg = next(n[: -len(".props")] for n in sys.modules if n.endswith("tree_forge.props"))
    props = sys.modules[pkg + ".props"]
    preset_io = sys.modules[pkg + ".preset_io"]
    return props, preset_io


# --------------------------------------------------------------------------------------
# マテリアル用の小道具
# --------------------------------------------------------------------------------------
class Mat:
    """ノードを並べて作るための薄いラッパー。"""

    def __init__(self, name):
        self.mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        if self.mat.node_tree is None:
            self.mat.use_nodes = True
        self.nt = self.mat.node_tree
        self.nt.nodes.clear()
        self.out = self.nt.nodes.new("ShaderNodeOutputMaterial")
        self.out.location = (900, 0)
        self._x = -1000

    def n(self, idname, x=None, y=0, **attrs):
        node = self.nt.nodes.new(idname)
        self._x += 200
        node.location = (self._x if x is None else x, y)
        for k, v in attrs.items():
            setattr(node, k, v)
        return node

    def principled(self, color=None, rough=0.8, x=500, y=0):
        b = self.nt.nodes.new("ShaderNodeBsdfPrincipled")
        b.location = (x, y)
        if color is not None:
            b.inputs["Base Color"].default_value = (*color, 1.0)
        b.inputs["Roughness"].default_value = rough
        return b

    def link(self, a, b):
        self.nt.links.new(a, b)

    def mix_shader(self, fac, s1, s2, x=700, y=0):
        m = self.nt.nodes.new("ShaderNodeMixShader")
        m.location = (x, y)
        self.link(fac, m.inputs["Fac"])
        self.link(s1.outputs["BSDF"], m.inputs[1])
        self.link(s2.outputs["BSDF"], m.inputs[2])
        return m


def set_in(node, name, value):
    node.inputs[name].default_value = value


def ramp(m, colors, fac=None, x=None, y=0):
    """カラーランプ。colors = [(pos, (r,g,b)), ...]"""
    r = m.n("ShaderNodeValToRGB", x=x, y=y)
    els = r.color_ramp.elements
    while len(els) < len(colors):
        els.new(1.0)
    for e, (pos, c) in zip(els, colors):
        e.position = pos
        e.color = (*c, 1.0)
    if fac is not None:
        m.link(fac, r.inputs["Fac"])
    return r


def noise(m, scale, detail=6.0, rough=0.55, vec=None, x=None, y=0, w=None):
    n = m.n("ShaderNodeTexNoise", x=x, y=y)
    set_in(n, "Scale", scale)
    set_in(n, "Detail", detail)
    set_in(n, "Roughness", rough)
    if vec is not None:
        m.link(vec, n.inputs["Vector"])
    return n


def mapping(m, vec, scale, x=None, y=0):
    mp = m.n("ShaderNodeMapping", x=x, y=y)
    set_in(mp, "Scale", scale)
    m.link(vec, mp.inputs["Vector"])
    return mp


def smooth_range(m, value, lo, hi, x=None, y=0, clamp=True):
    r = m.n("ShaderNodeMapRange", x=x, y=y)
    r.clamp = clamp
    r.interpolation_type = "SMOOTHSTEP"
    set_in(r, "From Min", lo)
    set_in(r, "From Max", hi)
    m.link(value, r.inputs["Value"])
    return r.outputs["Result"]


def sugi_bark():
    """スギの樹皮: 縦に長い繊維状の剥離と深い縦溝、根元は苔。UV は u=周方向(1 m タイル×整数)、v=幹に沿って 1 m。"""
    m = Mat("Sando_SugiBark")
    uv = m.n("ShaderNodeTexCoord", y=0).outputs["UV"]
    obj = m.n("ShaderNodeTexCoord", y=-500).outputs["Object"]
    fine = noise(m, 1.0, 10.0, 0.62, mapping(m, uv, (34.0, 1.6, 1.0)).outputs["Vector"], y=200)
    coarse = noise(m, 1.0, 6.0, 0.55, mapping(m, uv, (7.0, 0.55, 1.0)).outputs["Vector"], y=-100)
    groove = ramp(m, [(0.38, (0.0, 0, 0)), (0.62, (1.0, 1, 1))], coarse.outputs["Fac"], y=-100)
    col = ramp(m, [(0.0, (0.030, 0.014, 0.010)), (0.45, (0.17, 0.075, 0.042)), (1.0, (0.33, 0.17, 0.095))],
               fine.outputs["Fac"], y=250)
    bark = m.principled(rough=0.96, y=250)
    cm = m.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY", y=250)
    set_in(cm, "Factor", 1.0)
    a_in = [s for s in cm.inputs if s.name == "A" and s.type == "RGBA"][0]
    b_in = [s for s in cm.inputs if s.name == "B" and s.type == "RGBA"][0]
    m.link(col.outputs["Color"], a_in)
    gr = ramp(m, [(0.0, (0.35, 0.35, 0.35)), (1.0, (1.0, 1.0, 1.0))], groove.outputs["Color"], y=-250)
    m.link(gr.outputs["Color"], b_in)
    m.link([s for s in cm.outputs if s.type == "RGBA"][0], bark.inputs["Base Color"])
    # 凹凸: 溝 + 繊維
    hmix = m.n("ShaderNodeMath", operation="ADD", y=-300)
    m.link(groove.outputs["Color"], hmix.inputs[0])
    m.link(fine.outputs["Fac"], hmix.inputs[1])
    bump = m.n("ShaderNodeBump", y=-300)
    set_in(bump, "Strength", 1.4)
    set_in(bump, "Distance", 0.06)
    m.link(hmix.outputs["Value"], bump.inputs["Height"])
    m.link(bump.outputs["Normal"], bark.inputs["Normal"])
    # 根元の苔: 高さの減衰 × まだらなノイズ × 縦溝に溜まる × 木ごとのランダム（量と高さがばらつく）
    rnd = m.n("ShaderNodeObjectInfo", y=-600).outputs["Random"]
    sep = m.n("ShaderNodeSeparateXYZ", y=-550)
    m.link(obj, sep.inputs["Vector"])
    reach = m.n("ShaderNodeMath", operation="MULTIPLY_ADD", y=-600)          # 苔の届く高さ 1.6〜4.4 m
    m.link(rnd, reach.inputs[0])
    set_in(reach, 1, 2.8)
    set_in(reach, 2, 1.6)
    hdiv = m.n("ShaderNodeMath", operation="DIVIDE", y=-600)
    m.link(sep.outputs["Z"], hdiv.inputs[0])
    m.link(reach.outputs["Value"], hdiv.inputs[1])
    fall = smooth_range(m, hdiv.outputs["Value"], 0.15, 1.0, y=-600)         # 0(地面)→1(上)
    inv = m.n("ShaderNodeMath", operation="SUBTRACT", y=-600)
    set_in(inv, 0, 1.0)
    m.link(fall, inv.inputs[1])
    patch = noise(m, 1.4, 7.0, 0.6, obj, y=-750)
    fine_m = noise(m, 11.0, 5.0, 0.55, obj, y=-900)
    ingroove = m.n("ShaderNodeMath", operation="SUBTRACT", y=-800)           # 溝（groove=0 の所）ほど 1
    set_in(ingroove, 0, 1.0)
    m.link(groove.outputs["Color"], ingroove.inputs[1])                      # Color は RGB。Math へは明度として入る
    w1 = m.n("ShaderNodeMath", operation="MULTIPLY", y=-750)
    m.link(patch.outputs["Fac"], w1.inputs[0])
    set_in(w1, 1, 0.55)
    w2 = m.n("ShaderNodeMath", operation="MULTIPLY_ADD", y=-850)
    m.link(fine_m.outputs["Fac"], w2.inputs[0])
    set_in(w2, 1, 0.25)
    m.link(w1.outputs["Value"], w2.inputs[2])
    w3 = m.n("ShaderNodeMath", operation="MULTIPLY_ADD", y=-900)
    m.link(ingroove.outputs["Value"], w3.inputs[0])
    set_in(w3, 1, 0.20)
    m.link(w2.outputs["Value"], w3.inputs[2])
    amt = m.n("ShaderNodeMath", operation="MULTIPLY_ADD", y=-950)            # 木ごとの量 0.55〜1.2
    m.link(rnd, amt.inputs[0])
    set_in(amt, 1, 0.65)
    set_in(amt, 2, 0.55)
    raw = m.n("ShaderNodeMath", operation="MULTIPLY", y=-950)
    m.link(inv.outputs["Value"], raw.inputs[0])
    m.link(w3.outputs["Value"], raw.inputs[1])
    scaled = m.n("ShaderNodeMath", operation="MULTIPLY", y=-1000)
    m.link(raw.outputs["Value"], scaled.inputs[0])
    m.link(amt.outputs["Value"], scaled.inputs[1])
    moss_mask = smooth_range(m, scaled.outputs["Value"], 0.30, 0.46, y=-1000)
    mcol = ramp(m, [(0.0, (0.030, 0.060, 0.010)), (0.5, (0.050, 0.095, 0.018)), (1.0, (0.085, 0.100, 0.030))],
                fine_m.outputs["Fac"], y=-300)
    moss = m.principled(rough=1.0, y=-250, x=500)
    m.link(mcol.outputs["Color"], moss.inputs["Base Color"])
    m.link(bump.outputs["Normal"], moss.inputs["Normal"])
    mix = m.mix_shader(moss_mask, bark, moss, y=0)
    m.link(mix.outputs["Shader"], m.out.inputs["Surface"])
    return m.mat


def gravel():
    m = Mat("Sando_Gravel")
    pos = m.n("ShaderNodeTexCoord", y=0).outputs["Object"]
    vor = m.n("ShaderNodeTexVoronoi", y=0)
    set_in(vor, "Scale", 55.0)
    m.link(pos, vor.inputs["Vector"])
    bw = m.n("ShaderNodeRGBToBW", y=200)
    m.link(vor.outputs["Color"], bw.inputs["Color"])
    col = ramp(m, [(0.0, (0.20, 0.19, 0.17)), (0.5, (0.42, 0.40, 0.36)), (1.0, (0.62, 0.59, 0.54))],
               bw.outputs["Val"], y=200)
    fine = noise(m, 220.0, 2.0, 0.5, pos, y=-200)
    bump = m.n("ShaderNodeBump", y=-100)
    set_in(bump, "Strength", 0.9)
    set_in(bump, "Distance", 0.03)
    hh = m.n("ShaderNodeMath", operation="ADD", y=-100)
    m.link(vor.outputs["Distance"], hh.inputs[0])
    m.link(fine.outputs["Fac"], hh.inputs[1])
    set_in(hh, 1, 0.0)
    m.link(hh.outputs["Value"], bump.inputs["Height"])
    b = m.principled(rough=0.85)
    m.link(col.outputs["Color"], b.inputs["Base Color"])
    m.link(bump.outputs["Normal"], b.inputs["Normal"])
    m.link(b.outputs["BSDF"], m.out.inputs["Surface"])
    return m.mat


def ground():
    m = Mat("Sando_Ground")
    pos = m.n("ShaderNodeTexCoord", y=0).outputs["Object"]
    mid = noise(m, 2.2, 6.0, 0.6, pos, y=0)
    fine = noise(m, 40.0, 4.0, 0.5, pos, y=-200)
    col = ramp(m, [(0.0, (0.055, 0.075, 0.025)), (0.45, (0.085, 0.11, 0.035)), (0.62, (0.13, 0.085, 0.045)),
                   (1.0, (0.20, 0.12, 0.06))], mid.outputs["Fac"], y=200)
    b = m.principled(rough=0.95)
    m.link(col.outputs["Color"], b.inputs["Base Color"])
    bump = m.n("ShaderNodeBump", y=-200)
    set_in(bump, "Strength", 0.6)
    set_in(bump, "Distance", 0.08)
    m.link(fine.outputs["Fac"], bump.inputs["Height"])
    m.link(bump.outputs["Normal"], b.inputs["Normal"])
    m.link(b.outputs["BSDF"], m.out.inputs["Surface"])
    return m.mat


def stone(name="Sando_Stone", moss_amount=0.5, use_tint=True):
    """御影石・安山岩: 個体ごとの濃淡（頂点属性 tint）、上向きの面に苔。"""
    m = Mat(name)
    pos = m.n("ShaderNodeTexCoord", y=0).outputs["Object"]
    geo = m.n("ShaderNodeNewGeometry", y=-400)
    fine = noise(m, 60.0, 5.0, 0.55, pos, y=200)
    mid = noise(m, 5.0, 6.0, 0.6, pos, y=0)
    if use_tint:
        tint = m.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="tint", y=400)
        tint_out = tint.outputs["Fac"]
    else:
        tint_out = fine.outputs["Fac"]
    col = ramp(m, [(0.0, (0.20, 0.20, 0.19)), (0.5, (0.34, 0.33, 0.31)), (1.0, (0.46, 0.45, 0.42))], tint_out, y=350)
    b = m.principled(rough=0.82, y=250)
    m.link(col.outputs["Color"], b.inputs["Base Color"])
    bump = m.n("ShaderNodeBump", y=-200)
    set_in(bump, "Strength", 0.5)
    set_in(bump, "Distance", 0.01)
    m.link(fine.outputs["Fac"], bump.inputs["Height"])
    m.link(bump.outputs["Normal"], b.inputs["Normal"])
    sep = m.n("ShaderNodeSeparateXYZ", y=-400)
    m.link(geo.outputs["Normal"], sep.inputs["Vector"])
    up = smooth_range(m, sep.outputs["Z"], 0.55, 0.95, y=-400)
    prod = m.n("ShaderNodeMath", operation="MULTIPLY", y=-450)
    m.link(up, prod.inputs[0])
    nm = m.n("ShaderNodeMath", operation="ADD", y=-500)
    m.link(mid.outputs["Fac"], nm.inputs[0])
    set_in(nm, 1, moss_amount - 0.5)
    m.link(smooth_range(m, nm.outputs["Value"], 0.42, 0.6, y=-500), prod.inputs[1])
    moss = m.principled((0.05, 0.11, 0.022), 1.0, y=-100)
    mix = m.mix_shader(prod.outputs["Value"], b, moss)
    m.link(mix.outputs["Shader"], m.out.inputs["Surface"])
    return m.mat


def paint(name, color, rough=0.45):
    m = Mat(name)
    b = m.principled(color, rough)
    m.link(b.outputs["BSDF"], m.out.inputs["Surface"])
    return m.mat


# --------------------------------------------------------------------------------------
# ジオメトリ
# --------------------------------------------------------------------------------------
def new_collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def obj_from_bm(name, bm, mat, col, loc=(0, 0, 0), rot=(0, 0, 0), smooth=False):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        me.shade_smooth()
    o = bpy.data.objects.new(name, me)
    o.location = loc
    o.rotation_euler = rot
    if mat is not None:
        me.materials.append(mat)
    col.objects.link(o)
    return o


def box(bm, loc, size, rot=(0.0, 0.0, 0.0)):
    return bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.LocRotScale(Vector(loc), Euler(rot), Vector(size)))["verts"]


def cone(bm, loc, r1, r2, depth, segments=8, rot=(0.0, 0.0, 0.0)):
    return bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments, radius1=r1, radius2=r2,
                                 depth=depth, matrix=Matrix.LocRotScale(Vector(loc), Euler(rot), Vector((1, 1, 1))))["verts"]


def sphere(bm, loc, r):
    return bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=r,
                                     matrix=Matrix.Translation(Vector(loc)))["verts"]


def build_path(col, length=150.0):
    """石畳（中央）+ 砂利（両側）+ 縁石。"""
    rng = np.random.default_rng(3)
    y0, y1 = -70.0, -70.0 + length
    mats = {"stone": stone("Sando_Slab", 0.35), "gravel": gravel()}

    # 砂利: 幅 6.2 m
    bm = bmesh.new()
    box(bm, (0, (y0 + y1) / 2, -0.04), (6.2, length, 0.08))
    obj_from_bm("Sando_Gravel", bm, mats["gravel"], col)

    # 石畳: 幅 2.4 m の 2 列。長さ・高さ・向きを少しずつ崩す
    bm = bmesh.new()
    tint = bm.verts.layers.float.new("tint")
    y = y0 + 0.4
    while y < y1 - 1.0:
        L = float(rng.uniform(0.70, 1.00))
        for cx in (-0.60, 0.60):
            w = 1.16 + float(rng.uniform(-0.03, 0.03))
            vs = box(bm, (cx + rng.uniform(-0.015, 0.015), y + L / 2, 0.012 + rng.uniform(-0.012, 0.012)),
                     (w, L, 0.12),
                     (math.radians(rng.normal(0, 0.35)), math.radians(rng.normal(0, 0.35)), math.radians(rng.normal(0, 0.8))))
            t = float(rng.uniform(0.0, 1.0))
            for v in vs:
                v[tint] = t
        y += L + 0.05
    obj_from_bm("Sando_Slabs", bm, mats["stone"], col)

    # 縁石
    bm = bmesh.new()
    tint = bm.verts.layers.float.new("tint")
    for sx in (-3.15, 3.15):
        y = y0 + 0.2
        while y < y1 - 1.6:
            L = float(rng.uniform(1.2, 1.7))
            vs = box(bm, (sx, y + L / 2, 0.06), (0.26, L, 0.2), (0, 0, math.radians(rng.normal(0, 0.5))))
            t = float(rng.uniform(0.0, 1.0))
            for v in vs:
                v[tint] = t
            y += L + 0.03
    obj_from_bm("Sando_Curb", bm, mats["stone"], col)


def build_ground(col):
    bm = bmesh.new()
    box(bm, (0, 0, -0.10), (400, 400, 0.12))
    obj_from_bm("Sando_Ground", bm, ground(), col)


def lantern(col, mat, loc, rot_z):
    """石灯籠（春日灯籠風）: 基礎・竿・中台・火袋・笠・宝珠。高さ約 1.9 m。"""
    bm = bmesh.new()
    cone(bm, (0, 0, 0.07), 0.36, 0.36, 0.14, 6)                       # 基礎
    cone(bm, (0, 0, 0.14 + 0.38), 0.12, 0.12, 0.76, 8)                # 竿
    cone(bm, (0, 0, 0.90 + 0.05), 0.32, 0.27, 0.10, 6)                # 中台
    cone(bm, (0, 0, 1.00 + 0.19), 0.23, 0.23, 0.38, 6)                # 火袋
    for k in range(6):                                                # 火袋の窓（小さな出っ張りで表現）
        a = math.radians(30 + 60 * k)
        box(bm, (0.205 * math.cos(a), 0.205 * math.sin(a), 1.19), (0.05, 0.12, 0.20), (0, 0, a))
    cone(bm, (0, 0, 1.38 + 0.03), 0.34, 0.34, 0.06, 6)                # 笠の下端
    cone(bm, (0, 0, 1.44 + 0.14), 0.62, 0.10, 0.28, 6)                # 笠
    sphere(bm, (0, 0, 1.86 - 0.12), 0.09)                             # 宝珠
    return obj_from_bm("Lantern", bm, mat, col, loc, (0, 0, rot_z), smooth=False)


def build_lanterns(col):
    mat = stone("Sando_Lantern", 0.75, use_tint=False)
    for i, y in enumerate((-30.0, -14.0, 2.0, 18.0, 34.0)):
        for sx in (-4.1, 4.1):
            lantern(col, mat, (sx, y + (1.5 if sx > 0 else 0.0), 0.0), math.radians(15 * i + (30 if sx > 0 else 0)))


def build_torii(col, y=40.0):
    red = paint("Sando_ToriiRed", (0.50, 0.045, 0.02), 0.42)
    black = paint("Sando_ToriiBlack", (0.015, 0.014, 0.014), 0.5)
    bm = bmesh.new()
    for sx in (-2.7, 2.7):
        cone(bm, (sx, y, 2.5), 0.30, 0.24, 5.0, 16)                   # 柱（わずかに細る）
        cone(bm, (sx, y, 0.12), 0.34, 0.34, 0.24, 16)                  # 台座
    box(bm, (0, y, 4.05), (6.8, 0.24, 0.30))                           # 貫
    box(bm, (0, y, 3.55), (0.30, 0.20, 0.5))                           # 額束
    obj_from_bm("Torii_Red", bm, red, col, smooth=True)
    bm = bmesh.new()
    box(bm, (0, y, 5.12), (8.4, 0.52, 0.30))                           # 笠木
    box(bm, (0, y, 4.94), (7.6, 0.40, 0.22))                           # 島木
    obj_from_bm("Torii_Black", bm, black, col)


def build_person(col, loc, yaw=0.0):
    """スケール確認用の人影（約 1.7 m）。"""
    coat = paint("Sando_PersonCoat", (0.025, 0.03, 0.05), 0.8)
    skin = paint("Sando_PersonSkin", (0.55, 0.38, 0.30), 0.6)
    bm = bmesh.new()
    cone(bm, (0, 0, 0.45), 0.13, 0.12, 0.9, 12)
    cone(bm, (0, 0, 1.18), 0.19, 0.17, 0.62, 12)
    obj_from_bm("Person", bm, coat, col, loc, (0, 0, yaw), smooth=True)
    bm = bmesh.new()
    sphere(bm, (0, 0, 1.62), 0.105)
    obj_from_bm("Person_Head", bm, skin, col, loc, (0, 0, yaw), smooth=True)


# --------------------------------------------------------------------------------------
# 木
# --------------------------------------------------------------------------------------
def make_tree(props, preset_io, sc, species, name, loc, yaw, bark, **ov):
    d = preset_io.builtin_overrides(species)
    d.update(ov)
    props.write(sc.tree_forge, d)
    sc.tree_forge.initialized = True
    sc.tree_forge.materials.trunk = bark
    sc.tree_forge.materials.root_flare = bark
    bpy.context.view_layer.objects.active = None
    assert bpy.ops.treeforge.generate() == {"FINISHED"}
    o = bpy.context.active_object
    o.name = name
    o.location = loc
    o.rotation_euler = (0, 0, yaw)
    return o


def build_trees(props, preset_io, sc, hero_seed, hero_quality, n_bg, n_forest=0):
    bark = sugi_bark()
    hero = make_tree(props, preset_io, sc, "sugi_giant", "Cedar_Hero", (6.6, 0.0, 0.0), 0.0, bark,
                     quality=hero_quality, seed=hero_seed)
    rng = np.random.default_rng(11)
    sides = [-1, 1]
    ys = np.linspace(-62, 66, max(n_bg // 2, 2))
    k = 0
    for y in ys:
        for s in sides:
            if s > 0 and abs(y) < 9.0:        # 主役の周り
                continue
            x = s * rng.uniform(8.0, 12.5)
            y_ = float(y + rng.uniform(-3, 3))
            H = float(rng.uniform(33, 47))
            near = abs(y_) < 34
            make_tree(props, preset_io, sc, "sugi", f"Cedar_{k:02d}", (float(x), y_, 0.0),
                      float(rng.uniform(0, 6.28)), bark,
                      quality="MEDIUM" if near else "PREVIEW", seed=int(100 + k), height_m=H,
                      dbh_cm=float(rng.uniform(70, 115)), crown_width_m=float(rng.uniform(6.5, 9.5)),
                      crown_base_ratio=float(rng.uniform(0.38, 0.5)), age_factor=float(rng.uniform(0.9, 1.3)),
                      leaf_scale=5.0, leaf_cluster=5, leaves_per_m=60.0, foliage_amount=2.2)
            k += 1
    # 遠景の森: 参道の両側とその奥。低品質＋葉の数を絞って軽くする
    rng = np.random.default_rng(23)
    j = 0
    while j < n_forest:
        s = 1 if rng.random() < 0.5 else -1
        x = s * float(rng.uniform(15.0, 70.0))
        y = float(rng.uniform(-28.0, 150.0))            # カメラ(y=-44)の近くには置かない
        if abs(x) < 17:
            continue
        make_tree(props, preset_io, sc, "sugi", f"Forest_{j:02d}", (x, y, 0.0), float(rng.uniform(0, 6.28)), bark,
                  quality="PREVIEW", seed=int(500 + j), height_m=float(rng.uniform(28, 48)),
                  dbh_cm=float(rng.uniform(55, 100)), crown_width_m=float(rng.uniform(6.0, 9.5)),
                  crown_base_ratio=float(rng.uniform(0.35, 0.5)), leaf_scale=6.0, leaf_cluster=5, leaves_per_m=60.0,
                  foliage_amount=2.0, leaf_max_instances=6000)
        j += 1
    return hero


def build_fog(col, density):
    """参道一帯を覆う霧（箱のボリューム）。地面付近ほど濃く、高さ 45 m ごとに 1/e。
    世界全体に霧をかけると無限遠まで積分されて空が黒くなるため、有限の箱にする。"""
    m = Mat("Sando_Fog")
    obj = m.n("ShaderNodeTexCoord", y=0).outputs["Object"]
    sep = m.n("ShaderNodeSeparateXYZ", y=0)
    m.link(obj, sep.inputs["Vector"])
    k = m.n("ShaderNodeMath", operation="MULTIPLY", y=0)
    m.link(sep.outputs["Z"], k.inputs[0])
    set_in(k, 1, -1.0 / 45.0)
    ex = m.n("ShaderNodeMath", operation="EXPONENT", y=0)
    m.link(k.outputs["Value"], ex.inputs[0])
    dens = m.n("ShaderNodeMath", operation="MULTIPLY", y=0)
    m.link(ex.outputs["Value"], dens.inputs[0])
    set_in(dens, 1, density * 1.6)
    vs = m.n("ShaderNodeVolumeScatter", y=0)
    set_in(vs, "Color", (0.86, 0.92, 1.0, 1.0))
    set_in(vs, "Anisotropy", 0.45)
    m.link(dens.outputs["Value"], vs.inputs["Density"])
    m.link(vs.outputs["Volume"], m.out.inputs["Volume"])
    bm = bmesh.new()
    box(bm, (0, 0, 60), (330, 330, 120))
    o = obj_from_bm("Sando_Fog", bm, m.mat, col)
    o.display_type = "WIRE"
    o.visible_shadow = False


# --------------------------------------------------------------------------------------
# 空・光・カメラ・レンダ設定
# --------------------------------------------------------------------------------------
def build_world_and_light(sc):
    w = bpy.data.worlds.new("Sando_World")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (600, 0)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.location = (300, 0)
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.location = (0, 0)
    types = [i.identifier for i in sky.bl_rna.properties["sky_type"].enum_items]
    for t in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "NISHITA"):
        if t in types:
            sky.sky_type = t
            break
    elev = math.radians(24)
    to_sun = Vector((-0.74 * math.cos(elev), 0.67 * math.cos(elev), math.sin(elev)))     # 左手前から低く差し込む
    az = math.atan2(to_sun.x, to_sun.y)
    sky.sun_elevation, sky.sun_rotation = elev, az
    if hasattr(sky, "sun_disc"):
        sky.sun_disc = False
    bg.inputs["Strength"].default_value = 0.42
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 3.2
    sun.color = (1.0, 0.93, 0.82)
    sun.angle = math.radians(1.2)
    so = bpy.data.objects.new("Sun", sun)
    sc.collection.objects.link(so)
    so.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()      # 光は太陽から原点へ進む


def build_camera(sc, view, hero_height):
    cam = bpy.data.cameras.new("Cam")
    co = bpy.data.objects.new("Cam", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    if view == "wide":
        cam.lens = 24
        loc, tgt = Vector((-3.2, -43.0, 1.5)), Vector((5.4, 0.0, hero_height * 0.42))
        res = (1100, 1600)
    elif view == "crown":
        cam.lens = 50
        loc, tgt = Vector((-4.0, -30.0, 22.0)), Vector((7.4, 0.0, 37.0))
        res = (1300, 1000)
    else:
        cam.lens = 22
        loc, tgt = Vector((0.6, -9.5, 1.3)), Vector((6.6, -0.8, 9.0))
        res = (1500, 1000)
    co.location = loc
    co.rotation_euler = (tgt - loc).to_track_quat("-Z", "Y").to_euler()
    cam.clip_end = 600
    return res


def setup_render(sc, res, scale, samples, denoise=True):
    addon_enable()
    sc.render.engine = "CYCLES"
    sc.render.resolution_x, sc.render.resolution_y = int(res[0] * scale), int(res[1] * scale)
    cy = sc.cycles
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.02
    cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces, cy.transmission_bounces = 6, 3, 2, 3
    cy.volume_bounces = 1
    cy.volume_step_rate = 2.0
    cy.sample_clamp_indirect = 8.0
    if denoise:
        try:
            cy.use_denoising = True
            cy.denoiser = "OPENIMAGEDENOISE"
        except (TypeError, AttributeError):
            cy.use_denoising = False
    looks = [i.identifier for i in sc.view_settings.bl_rna.properties["look"].enum_items]
    for cand in ("AgX - High Contrast", "High Contrast", "AgX - Medium High Contrast", "Medium High Contrast"):
        if cand in looks:
            sc.view_settings.look = cand
            break
    sc.view_settings.exposure = -0.2


def addon_enable():
    import addon_utils
    addon_utils.enable("cycles", default_set=False)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sando.png")
    ap.add_argument("--view", choices=("wide", "low", "crown"), default="wide")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--fog", type=float, default=0.0004)
    ap.add_argument("--bg-trees", type=int, default=16)
    ap.add_argument("--forest", type=int, default=36, help="遠景の森の本数")
    ap.add_argument("--hero-quality", default="HERO", choices=("PREVIEW", "MEDIUM", "HERO"))
    ap.add_argument("--save-blend", default="")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-denoise", action="store_true")
    args = ap.parse_args(argv)

    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    props, preset_io = load_addon()
    sc = bpy.context.scene

    env = new_collection("Sando")
    build_ground(env)
    build_path(env)
    build_lanterns(env)
    build_torii(env)
    build_person(env, (-1.6, 11.0, 0.0), 0.4)
    build_person(env, (4.6, -6.6, 0.0), 2.6)
    hero = build_trees(props, preset_io, sc, args.seed, args.hero_quality, args.bg_trees, args.forest)
    hero_h = props.read(hero.tree_forge).height_m
    build_world_and_light(sc)
    if args.fog > 0:
        build_fog(env, args.fog)
    res = build_camera(sc, args.view, hero_h)
    setup_render(sc, res, args.scale, args.samples, denoise=not args.no_denoise)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
    if not args.no_render:
        sc.render.filepath = str(Path(args.out).resolve())
        sc.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(write_still=True)
        print("saved", sc.render.filepath)
    print("hero:", hero.tree_forge.info)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)
