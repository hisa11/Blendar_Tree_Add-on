"""core の結果を Blender のオブジェクト（幹メッシュ + 葉の点群 + GN）にする。

再生成では Mesh データブロックだけを差し替え、オブジェクト・モディファイア・トランスフォームは保持する。
"""
from __future__ import annotations

import numpy as np

import bpy

from .. import props
from ..core import leaf_shapes, pipeline
from . import gn_leaves, materials

SOURCES_COLLECTION = "TG_Sources"

# 面・頂点属性 → (Blender の型, データのメンバ名)
_ATTR_KIND = {np.dtype(np.int32): ("INT", "value"), np.dtype(np.float32): ("FLOAT", "value"),
              np.dtype(np.bool_): ("BOOLEAN", "value")}


def _add_attr(me, name, arr, domain):
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 3:
        a = me.attributes.new(name, "FLOAT_VECTOR", domain)
        a.data.foreach_set("vector", arr.astype(np.float32).ravel())
    elif arr.ndim == 2 and arr.shape[1] == 4:
        a = me.attributes.new(name, "QUATERNION", domain)
        a.data.foreach_set("value", arr.astype(np.float32).ravel())
    else:
        kind, member = _ATTR_KIND[arr.dtype]
        a = me.attributes.new(name, kind, domain)
        a.data.foreach_set(member, arr.ravel())


def fill_tree_mesh(me, md) -> None:
    """MeshData を空の Mesh に書き込む。"""
    me.vertices.add(len(md.co))
    me.vertices.foreach_set("co", md.co.ravel())
    me.loops.add(len(md.loop_vert))
    me.loops.foreach_set("vertex_index", md.loop_vert)
    starts = np.zeros(len(md.face_size), np.int32)
    starts[1:] = np.cumsum(md.face_size)[:-1]
    me.polygons.add(len(starts))
    me.polygons.foreach_set("loop_start", starts)
    me.update(calc_edges=True)
    me.polygons.foreach_set("material_index", md.material_index)
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", md.uv.ravel())
    for k, v in md.point_attrs.items():
        _add_attr(me, k, v, "POINT")
    for k, v in md.face_attrs.items():
        _add_attr(me, k, v, "FACE")
    me.shade_smooth()


def fill_point_cloud(me, lf) -> None:
    n = lf.n
    me.vertices.add(n)
    if n:
        me.vertices.foreach_set("co", lf.pos.astype(np.float32).ravel())
        _add_attr(me, "leaf_rot", lf.quat, "POINT")
        _add_attr(me, "leaf_scale", lf.scale, "POINT")
        _add_attr(me, "is_leaf", lf.is_leaf, "POINT")
        _add_attr(me, "is_flower", lf.is_flower, "POINT")
        _add_attr(me, "leaf_var", lf.var.astype(np.float32), "POINT")
        _add_attr(me, "tree_part", np.where(lf.is_leaf, 3, 4).astype(np.int32), "POINT")
    me.update()


def fill_source(me, verts, tris, uv) -> None:
    """葉・花のインスタンス元（三角形のみ）。"""
    m = len(tris)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", np.asarray(verts, np.float32).ravel())
    me.loops.add(m * 3)
    me.loops.foreach_set("vertex_index", np.asarray(tris, np.int32).ravel())
    me.polygons.add(m)
    me.polygons.foreach_set("loop_start", np.arange(m, dtype=np.int32) * 3)
    me.update(calc_edges=True)
    layer = me.uv_layers.new(name="UVMap")
    layer.data.foreach_set("uv", np.asarray(uv, np.float32)[np.asarray(tris).ravel()].ravel())
    me.shade_smooth()


def _replace_mesh(obj, new_me) -> None:
    old = obj.data
    obj.data = new_me
    if old is not None and old.users == 0:
        bpy.data.meshes.remove(old)


def _sources_collection(context):
    col = bpy.data.collections.get(SOURCES_COLLECTION)
    if col is None:
        col = bpy.data.collections.new(SOURCES_COLLECTION)
    if col.name not in context.scene.collection.children:
        context.scene.collection.children.link(col)
    return col


def _hide(obj) -> None:
    obj.hide_render = True
    try:
        obj.hide_set(True)          # 目のアイコン。非表示でも GN の Object Info からは評価される
    except RuntimeError:
        pass


ROLE_KEY = "tf_role"


def _adopt(obj, ref, role):
    """obj が所有する子オブジェクトを返す。ref（保存済みの参照）が自分の子ならそれ、
    複製された木などで参照が他の木を指していれば、自分の子から role の一致するものを探す。"""
    if ref is not None and ref.parent is obj:
        return ref
    for c in obj.children:
        if c.get(ROLE_KEY) == role:
            return c
    return None


def _ensure_source(context, obj, role, name, verts, tris, uv, existing):
    """生成した葉・花のインスタンス元オブジェクト（木の子・非表示）を作る／更新する。"""
    me = bpy.data.meshes.new(name)
    fill_source(me, verts, tris, uv)
    src = _adopt(obj, existing, role)
    if src is None:
        src = bpy.data.objects.new(name, me)
        src[ROLE_KEY] = role
        _sources_collection(context).objects.link(src)
        src.parent = obj
        _hide(src)
        return src
    src.name = name
    _replace_mesh(src, me)
    return src


def create_tree_object(context, name="Tree") -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, me)
    context.collection.objects.link(obj)
    obj.location = context.scene.cursor.location
    obj.tree_forge.is_tree = True
    return obj


def _ensure_leaves_object(context, obj, pg):
    lo = _adopt(obj, pg.leaves_object, "leaves")
    if lo is None:
        lo = bpy.data.objects.new(f"{obj.name}_Leaves", bpy.data.meshes.new(f"{obj.name}_Leaves"))
        lo[ROLE_KEY] = "leaves"
        for col in obj.users_collection:
            col.objects.link(lo)
        lo.parent = obj
    pg.leaves_object = lo
    return lo


def build(context, obj, ns=None):
    """obj のパラメータ（または ns）で木を作り直す。生成結果 (pipeline.Tree) を返す。"""
    pg = obj.tree_forge
    ns = ns if ns is not None else props.read(pg)
    tree = pipeline.generate(ns)

    # 幹・枝メッシュ
    me = bpy.data.meshes.new(obj.name)
    fill_tree_mesh(me, tree.mesh)
    _replace_mesh(obj, me)

    # 葉の点群
    lo = _ensure_leaves_object(context, obj, pg)
    lo.name = f"{obj.name}_Leaves"
    pc = bpy.data.meshes.new(lo.name)
    fill_point_cloud(pc, tree.leaves)
    _replace_mesh(lo, pc)

    # インスタンス元
    if ns.leaf_geom == "CUSTOM_OBJECT" and pg.leaf_object is not None:
        leaf_src = pg.leaf_object
    else:
        v, t, uv = leaf_shapes.leaf_source(ns)
        leaf_src = _ensure_source(context, obj, "leaf_src", f"{obj.name}_LeafSrc", v, t, uv, pg.leaf_src)
        pg.leaf_src = leaf_src
    flower_src = None
    if ns.flower_density > 0.0:
        v, t, uv = leaf_shapes.flower_source(ns)
        flower_src = _ensure_source(context, obj, "flower_src", f"{obj.name}_FlowerSrc", v, t, uv, pg.flower_src)
        pg.flower_src = flower_src

    leaf_mat = materials.resolve(pg, "leaf") or materials.default_material(
        "leaf", ns, obj.name, materials.season_color(ns, tree.leaves.color_t))
    flower_mat = (materials.resolve(pg, "flower") or materials.default_material("flower", ns, obj.name)) \
        if flower_src is not None else None
    ng = gn_leaves.build_group(f"TG_Leaves_{obj.name}", leaf_src, leaf_mat, flower_src, flower_mat)
    gn_leaves.attach(lo, ng)

    if pg.materials.auto_assign:
        materials.auto_assign(pg)
    materials.apply_all(obj, pg, ns, tree.leaves.color_t)

    H, W = pipeline.measure(tree)
    pg.info = (f"樹高 {H:.1f} m / 樹冠幅 {W:.1f} m / 幹・枝 {tree.mesh.tri_count:,} 三角形 / "
               f"葉 {int(tree.leaves.is_leaf.sum()):,} 花 {int(tree.leaves.is_flower.sum()):,}")
    pg.is_tree = True
    return tree


def create(context, template, name=None):
    """シーンのひな形（template）から新しい木を作る。"""
    obj = create_tree_object(context, name or f"Tree_{template.species}")
    props.copy_group(template, obj.tree_forge)
    obj.tree_forge.is_tree = True
    build(context, obj)
    return obj


def regenerate(context, obj):
    return build(context, obj)
