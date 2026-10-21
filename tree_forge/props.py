"""PropertyGroup。schema.py の表から生成する。

パラメータは Object.tree_forge に保存し（再編集できる）、未生成のときの入力欄は Scene.tree_forge を使う。
属性アクセスだけを使い、dict 形式のアクセスはしない（Blender 5.0 以降で不可）。
"""
from __future__ import annotations

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy.app.handlers import persistent
from bpy.types import PropertyGroup

from . import preset_io, schema

PARTS = ("trunk", "branch", "twig", "leaf", "flower", "fruit", "root_flare", "broken")
PART_LABELS = {"trunk": "幹", "branch": "枝", "twig": "小枝", "leaf": "葉", "flower": "花", "fruit": "果実",
               "root_flare": "根張り", "broken": "折れ口"}

_applying = False       # プリセット適用中は update から再生成しない


def _regen_if_live(pg) -> None:
    owner = pg.id_data
    if not isinstance(owner, bpy.types.Object):          # シーンのひな形は再生成しない
        return
    tf = owner.tree_forge                                  # pg は太さ点などの子グループのこともある
    if tf.is_tree and tf.live_update and tf.quality != "HERO":
        from .blender import build_mesh
        build_mesh.regenerate(bpy.context, owner)


def _on_update(self, context):
    if not _applying:
        _regen_if_live(self)


def _on_species(self, context):
    if _applying:
        return
    if self.species != "custom":
        write(self, preset_io.builtin_overrides(self.species))
    _regen_if_live(self)


def _on_mode(self, context):
    """CUSTOM に切り替えたとき、点が無ければ今の自動プロファイルから初期点を作る。"""
    if _applying:
        return
    if self.trunk_profile_mode == "CUSTOM" and len(self.profile) == 0:
        reset_profile(self)
    _regen_if_live(self)


class TF_ProfilePoint(PropertyGroup):
    height_ratio: FloatProperty(name="高さ比", description="樹高に対する地上高の比（0 根元 〜 1 樹頂）",
                                default=0.0, min=0.0, max=1.0, update=_on_update)
    diameter_cm: FloatProperty(name="直径 [cm]", description="その高さでの幹の直径", default=30.0,
                               min=0.1, max=1000.0, update=_on_update)


class TF_PartMaterials(PropertyGroup):
    trunk: PointerProperty(type=bpy.types.Material, name="幹",
          description="幹の材質（スロット 0）")
    branch: PointerProperty(type=bpy.types.Material, name="枝",
           description="枝の材質（スロット 1）")
    twig: PointerProperty(type=bpy.types.Material, name="小枝",
         description="小枝の材質（スロット 2）。未指定なら枝の材質を使う")
    leaf: PointerProperty(type=bpy.types.Material, name="葉",
         description="葉の材質（スロット 3。葉は Geometry Nodes のインスタンスに付く）")
    flower: PointerProperty(type=bpy.types.Material, name="花",
           description="花の材質（スロット 4。Geometry Nodes のインスタンスに付く）")
    fruit: PointerProperty(type=bpy.types.Material, name="果実",
          description="果実の材質（スロット 5）")
    root_flare: PointerProperty(type=bpy.types.Material, name="根張り",
               description="根張りの材質（スロット 6）。未指定なら幹の材質を使う")
    broken: PointerProperty(type=bpy.types.Material, name="折れ口",
           description="枝の折れ口の材質（スロット 7）。淡い木質色にすると折れ口が白っぽく見える")
    auto_assign: BoolProperty(name="再生成時に自動割り当て", default=False,
                              description="再生成のたびに、材質名のキーワード（bark, leaf, 枝 など）で各部位へ自動で割り当てる")
    flip_normal_green: BoolProperty(name="法線Gを反転(DirectX)", default=False,
                                    description="フォルダから構築するとき、規約が不明な法線マップを DirectX として扱う")


def _make_prop(p: schema.Param):
    name = p.label + (f" [{p.unit}]" if p.unit else "")
    update = _on_update
    if p.name == "species":
        update = _on_species
    elif p.name == "trunk_profile_mode":
        update = _on_mode
    elif not p.update_regen:
        update = None
    kw = dict(name=name, description=p.desc)
    if update:
        kw["update"] = update
    if p.kind == "float":
        return FloatProperty(default=p.default, min=p.lo, max=p.hi, **kw)
    if p.kind == "int":
        return IntProperty(default=p.default, min=int(p.lo), max=int(p.hi), **kw)
    if p.kind == "bool":
        return BoolProperty(default=p.default, **kw)
    if p.kind == "enum":
        return EnumProperty(items=[(i, lbl, "") for i, lbl in p.items], default=p.default, **kw)
    if p.kind == "color":
        return FloatVectorProperty(subtype="COLOR", size=3, min=0.0, max=1.0, default=p.default, **kw)
    raise ValueError(p.kind)


def _build_group_class():
    ann = {p.name: _make_prop(p) for p in schema.PARAMS}
    ann.update(
        profile=CollectionProperty(type=TF_ProfilePoint),
        profile_index=IntProperty(default=0),
        leaf_object=PointerProperty(type=bpy.types.Object, name="葉オブジェクト",
                                    description="葉の表現が「任意オブジェクト」のときのインスタンス元"),
        materials=PointerProperty(type=TF_PartMaterials),
        is_tree=BoolProperty(default=False),
        live_update=BoolProperty(name="自動再生成", default=False,
                                 description="値を変えるたびに再生成（高品質では無効）"),
        advanced=BoolProperty(name="詳細モード", default=False,
                              description="全パラメータのパネル（枝 L1〜L3 の細目、葉の細目、折れ位置など）を表示する"),
        initialized=BoolProperty(default=False, description="樹種プリセットを一度適用したか"),
        leaves_object=PointerProperty(type=bpy.types.Object),
        leaf_src=PointerProperty(type=bpy.types.Object),
        flower_src=PointerProperty(type=bpy.types.Object),
        info=StringProperty(default=""),
    )
    return type("TF_TreeParams", (PropertyGroup,), {"__annotations__": ann})


TF_TreeParams = _build_group_class()
CLASSES = (TF_ProfilePoint, TF_PartMaterials, TF_TreeParams)


# --------------------------------------------------------------------------------------
# 読み書き
# --------------------------------------------------------------------------------------
def read(pg):
    """PropertyGroup → core が使う SimpleNamespace。"""
    ns = schema.make_params()
    for p in schema.PARAMS:
        v = getattr(pg, p.name)
        setattr(ns, p.name, tuple(v) if p.kind == "color" else v)
    ns.profile = [(pt.height_ratio, pt.diameter_cm) for pt in pg.profile]
    return ns


def write(pg, overrides: dict, reset: bool = True) -> None:
    """辞書（preset_io.overrides_from_dict の形）を PropertyGroup へ書く。reset なら先に既定値へ戻す。"""
    global _applying
    ns = schema.make_params(**overrides)           # 検証とクランプ
    _applying = True
    try:
        if reset:
            base = schema.make_params()
            for p in schema.PARAMS:
                setattr(pg, p.name, getattr(base, p.name))
            pg.profile.clear()
        for k in overrides:
            if k in schema.BY_NAME:
                setattr(pg, k, getattr(ns, k))
        if "profile" in overrides:
            set_profile(pg, ns.profile)
    finally:
        _applying = False


def set_profile(pg, points) -> None:
    global _applying
    prev = _applying
    _applying = True
    try:
        pg.profile.clear()
        for t, d in sorted(points):
            pt = pg.profile.add()
            pt.height_ratio, pt.diameter_cm = float(t), float(d)
        pg.profile_index = 0
    finally:
        _applying = prev


def reset_profile(pg) -> None:
    """現在の AUTO 曲線から CUSTOM の初期点を作る。"""
    from .core import allometry
    ns = read(pg)
    set_profile(pg, allometry.auto_profile_points(ns, 5))


def ensure_template(scene) -> None:
    """シーンのひな形に、選択中の樹種のプリセットを一度だけ適用する。"""
    pg = scene.tree_forge
    if not pg.initialized:
        if pg.species != "custom":
            write(pg, preset_io.builtin_overrides(pg.species))
        pg.initialized = True


def _init_templates():
    if not hasattr(bpy.types.Scene, "tree_forge"):       # 起動直後にアドオンが無効化された場合
        return None
    for sc in bpy.data.scenes:
        ensure_template(sc)
    return None


@persistent
def _on_load_post(_):
    bpy.app.timers.register(_init_templates, first_interval=0.1)


def copy_group(src, dst) -> None:
    ov = {p.name: (tuple(getattr(src, p.name)) if p.kind == "color" else getattr(src, p.name))
          for p in schema.PARAMS}
    ov["profile"] = [(pt.height_ratio, pt.diameter_cm) for pt in src.profile]
    write(dst, ov, reset=False)
    for part in PARTS:                                   # 割り当て済みの材質・葉オブジェクトも引き継ぐ
        setattr(dst.materials, part, getattr(src.materials, part))
    dst.materials.auto_assign = src.materials.auto_assign
    dst.materials.flip_normal_green = src.materials.flip_normal_green
    dst.leaf_object = src.leaf_object
    dst.initialized = True


def active_group(context):
    """(PropertyGroup, 木オブジェクトまたは None)。木が選択されていればそのパラメータ、なければシーンのひな形。"""
    obj = context.active_object
    if obj is not None and obj.tree_forge.is_tree:
        return obj.tree_forge, obj
    obj = context.object
    if obj is not None and obj.parent is not None and obj.parent.tree_forge.is_tree:
        return obj.parent.tree_forge, obj.parent
    return context.scene.tree_forge, None


def register() -> None:
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Object.tree_forge = PointerProperty(type=TF_TreeParams)
    bpy.types.Scene.tree_forge = PointerProperty(type=TF_TreeParams)
    bpy.app.handlers.load_post.append(_on_load_post)
    bpy.app.timers.register(_init_templates, first_interval=0.1)


def unregister() -> None:
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    del bpy.types.Scene.tree_forge
    del bpy.types.Object.tree_forge
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
