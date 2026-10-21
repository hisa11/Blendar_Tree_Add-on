"""M0: Blender API の実在確認。結果を JSON で標準出力に出す。

実行例（bpy モジュール版）:
    python tests/blender/smoke_api.py
実行例（blender 本体）:
    blender -b --factory-startup --python-exit-code 1 --python tests/blender/smoke_api.py
"""
import json
import sys

import bpy
import numpy as np

result = {"blender": bpy.app.version_string, "python": sys.version.split()[0], "numpy": np.__version__}


def probe(name, fn):
    try:
        result[name] = fn()
    except Exception as e:  # noqa: BLE001 - 失敗内容そのものを記録するのが目的
        result[name] = f"ERROR {type(e).__name__}: {e}"


# --- Mesh API ---------------------------------------------------------------
def mesh_api():
    me = bpy.data.meshes.new("smoke")
    me.vertices.add(4)
    me.vertices.foreach_set("co", np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], np.float32))
    me.loops.add(4)
    me.loops.foreach_set("vertex_index", np.arange(4, dtype=np.int32))
    me.polygons.add(1)
    me.polygons.foreach_set("loop_start", np.array([0], np.int32))
    out = {"loop_total_after": me.polygons[0].loop_total}
    me.update(calc_edges=True)
    out["edges"] = len(me.edges)
    out["polys"] = len(me.polygons)
    out["has_shade_smooth"] = hasattr(me, "shade_smooth")
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", np.zeros(8, np.float32))
    out["uv_layers"] = [u.name for u in me.uv_layers]
    return out


probe("mesh_api", mesh_api)


def attr_types():
    me = bpy.data.meshes.new("smoke_attr")
    me.vertices.add(3)
    res = {}
    for t in ("INT", "FLOAT", "FLOAT_VECTOR", "BOOLEAN", "QUATERNION", "FLOAT_COLOR", "INT32_2D"):
        try:
            me.attributes.new(f"a_{t}", t, "POINT")
            res[t] = "ok"
        except Exception as e:  # noqa: BLE001
            res[t] = f"ERROR {type(e).__name__}"
    a = me.attributes["a_INT"]
    a.data.foreach_set("value", np.array([1, 2, 3], np.int32))
    res["foreach_int"] = list(a.data.foreach_get and [v.value for v in a.data])
    return res


probe("attr_types", attr_types)


# --- Geometry Nodes ---------------------------------------------------------
def gn_api():
    tree = bpy.data.node_groups.new("smoke_gn", "GeometryNodeTree")
    out = {"interface_new_socket": hasattr(tree.interface, "new_socket")}
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    for bl in (
        "NodeGroupInput", "NodeGroupOutput", "GeometryNodeInstanceOnPoints", "GeometryNodeObjectInfo",
        "GeometryNodeInputNamedAttribute", "GeometryNodeSetMaterial", "GeometryNodeJoinGeometry",
        "GeometryNodeRealizeInstances",
    ):
        n = tree.nodes.new(bl)
        out[bl] = {
            "in": [(s.name, s.identifier, s.type) for s in n.inputs],
            "out": [(s.name, s.identifier, s.type) for s in n.outputs],
        }
        if bl == "GeometryNodeInputNamedAttribute":
            out[bl]["data_types"] = [i.identifier for i in n.bl_rna.properties["data_type"].enum_items]
    return out


probe("gn_api", gn_api)


# --- Shader nodes -----------------------------------------------------------
def shader_api():
    mat = bpy.data.materials.new("smoke_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    out = {}
    pb = nt.nodes.get("Principled BSDF") or nt.nodes.new("ShaderNodeBsdfPrincipled")
    out["principled_inputs"] = [s.name for s in pb.inputs]
    nm = nt.nodes.new("ShaderNodeNormalMap")
    out["normal_map_props"] = [p.identifier for p in nm.bl_rna.properties if not p.is_readonly]
    out["normal_map_inputs"] = [s.name for s in nm.inputs]
    out["has_thin_wall"] = "Thin Wall" in pb.inputs
    for bl in ("ShaderNodeDisplacement", "ShaderNodeBsdfTranslucent", "ShaderNodeMix", "ShaderNodeAttribute"):
        nt.nodes.new(bl)
    return out


probe("shader_api", shader_api)


# --- Extensions / 環境 ------------------------------------------------------
probe("has_extension_path_user", lambda: hasattr(bpy.utils, "extension_path_user"))
probe("has_app_cachedir", lambda: hasattr(bpy.app, "cachedir"))
probe("cycles_addon", lambda: "cycles" in bpy.context.preferences.addons.keys() or "available")
probe("engines", lambda: [e.identifier for e in bpy.types.RenderEngine.__subclasses__()] or "n/a")
probe("scene_engines", lambda: [i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items])

print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
