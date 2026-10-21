"""葉・花を Instance on Points で置く固定構成の Geometry Nodes。

モディファイアの入力（modifier["Socket_x"] / modifier.properties）は一切使わない。
インスタンス元オブジェクトとマテリアルはノードの default_value に直接入れる。
そのため 5.1 と 5.2 以降で同じコードが動く。木ごとに専用のノードグループを持つ。

  点群メッシュの属性: is_leaf / is_flower (BOOLEAN), leaf_rot (QUATERNION), leaf_scale (FLOAT_VECTOR)
  [Group Input] → Instance on Points(葉) ┐
                  Instance on Points(花) ┴→ Join Geometry → [Group Output]
  インスタンス元 = Object Info → Set Material → Instance
"""
from __future__ import annotations

import bpy

MODIFIER_NAME = "TG_Leaves"


def _node(ng, bl_idname, x, y, label=None):
    n = ng.nodes.new(bl_idname)
    n.location = (x, y)
    if label:
        n.label = label
    return n


def _named_attr(ng, name, data_type, x, y):
    n = _node(ng, "GeometryNodeInputNamedAttribute", x, y, name)
    n.data_type = data_type
    n.inputs["Name"].default_value = name
    return n.outputs["Attribute"]


def _instancer(ng, flag_attr, src_obj, mat, x, y):
    """flag_attr の点だけに src_obj を置く。(Instances 出力ソケット) を返す。"""
    sel = _named_attr(ng, flag_attr, "BOOLEAN", x, y + 220)
    rot = _named_attr(ng, "leaf_rot", "QUATERNION", x, y + 120)
    scl = _named_attr(ng, "leaf_scale", "FLOAT_VECTOR", x, y + 20)
    info = _node(ng, "GeometryNodeObjectInfo", x, y - 180, "Source")
    info.transform_space = "ORIGINAL"
    info.inputs["Object"].default_value = src_obj
    info.inputs["As Instance"].default_value = False
    setm = _node(ng, "GeometryNodeSetMaterial", x + 220, y - 180)
    if mat is not None:
        setm.inputs["Material"].default_value = mat
    ng.links.new(info.outputs["Geometry"], setm.inputs["Geometry"])
    iop = _node(ng, "GeometryNodeInstanceOnPoints", x + 440, y)
    ng.links.new(sel, iop.inputs["Selection"])
    ng.links.new(setm.outputs["Geometry"], iop.inputs["Instance"])
    ng.links.new(rot, iop.inputs["Rotation"])
    ng.links.new(scl, iop.inputs["Scale"])
    return iop


def ensure_group(name: str):
    ng = bpy.data.node_groups.get(name)
    if ng is None:
        ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
        ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    else:
        ng.nodes.clear()
    return ng


def build_group(name: str, leaf_src, leaf_mat, flower_src=None, flower_mat=None):
    ng = ensure_group(name)
    gin = _node(ng, "NodeGroupInput", -300, 0)
    gout = _node(ng, "NodeGroupOutput", 1100, 0)
    join = _node(ng, "GeometryNodeJoinGeometry", 900, 0)
    leaves = _instancer(ng, "is_leaf", leaf_src, leaf_mat, 0, 200)
    ng.links.new(gin.outputs[0], leaves.inputs["Points"])
    ng.links.new(leaves.outputs["Instances"], join.inputs["Geometry"])
    if flower_src is not None:
        flowers = _instancer(ng, "is_flower", flower_src, flower_mat, 0, -450)
        ng.links.new(gin.outputs[0], flowers.inputs["Points"])
        ng.links.new(flowers.outputs["Instances"], join.inputs["Geometry"])
    ng.links.new(join.outputs["Geometry"], gout.inputs[0])
    return ng


def attach(leaves_obj, ng) -> None:
    mod = leaves_obj.modifiers.get(MODIFIER_NAME)
    if mod is None:
        mod = leaves_obj.modifiers.new(MODIFIER_NAME, "NODES")
    mod.node_group = ng


def set_materials(ng, leaf_mat, flower_mat) -> None:
    """既存グループの Set Material を差し替える（再生成せずにマテリアルだけ変える用）。"""
    setm = [n for n in ng.nodes if n.bl_idname == "GeometryNodeSetMaterial"]
    setm.sort(key=lambda n: -n.location.y)         # 上 = 葉、下 = 花
    for n, m in zip(setm, (leaf_mat, flower_mat)):
        if m is not None:
            n.inputs["Material"].default_value = m


def realize_copy(ng):
    """書き出し用: Join の後ろに Realize Instances を挟んだ複製ノードグループを返す。"""
    cp = ng.copy()
    cp.name = ng.name + "_realize"
    join = next(n for n in cp.nodes if n.bl_idname == "GeometryNodeJoinGeometry")
    gout = next(n for n in cp.nodes if n.bl_idname == "NodeGroupOutput")
    real = cp.nodes.new("GeometryNodeRealizeInstances")
    real.location = (1000, 0)
    cp.links.new(join.outputs["Geometry"], real.inputs["Geometry"])
    cp.links.new(real.outputs["Geometry"], gout.inputs[0])
    return cp
