"""オペレータ。パラメータは Object.tree_forge（なければ Scene.tree_forge のひな形）から読む。"""
from __future__ import annotations

import json
from pathlib import Path

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from .. import compat, preset_io, props
from ..core import allometry
from . import build_mesh, gn_leaves, materials


def _tree_obj(context):
    pg, obj = props.active_group(context)
    return obj


class _TreeOperator(Operator):
    @classmethod
    def poll(cls, context):
        return _tree_obj(context) is not None


class TREEFORGE_OT_generate(Operator):
    bl_idname = "treeforge.generate"
    bl_label = "木を生成"
    bl_description = "現在の設定で新しい木を3Dカーソルの位置に作る"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props.ensure_template(context.scene)
        pg, obj = props.active_group(context)
        template = obj.tree_forge if obj is not None else pg
        new = build_mesh.create(context, template)
        for o in context.selected_objects:
            o.select_set(False)
        new.select_set(True)
        context.view_layer.objects.active = new
        self.report({"INFO"}, new.tree_forge.info)
        return {"FINISHED"}


class TREEFORGE_OT_regenerate(_TreeOperator):
    bl_idname = "treeforge.regenerate"
    bl_label = "再生成"
    bl_description = "同じオブジェクトのまま、メッシュだけ作り直す（モディファイア・材質割り当ては保持）"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = _tree_obj(context)
        build_mesh.regenerate(context, obj)
        self.report({"INFO"}, obj.tree_forge.info)
        return {"FINISHED"}


class TREEFORGE_OT_randomize_seed(Operator):
    bl_idname = "treeforge.randomize_seed"
    bl_label = "シードを変える"
    bl_description = "シードを 0〜99999 の乱数に変える。再生成すると別の形の木になる"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        import random
        pg, obj = props.active_group(context)
        pg.seed = random.randint(0, 99999)
        return {"FINISHED"}


# --- 幹の太さプロファイル ---------------------------------------------------
class TREEFORGE_OT_profile_add(Operator):
    bl_idname = "treeforge.profile_add"
    bl_label = "点を追加"
    bl_description = "選択中の点と次の点の中間に、現在の曲線上の太さで点を足す"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        pg, obj = props.active_group(context)
        pts = sorted((pt.height_ratio, pt.diameter_cm) for pt in pg.profile)
        ns = props.read(pg)
        if not pts:
            props.reset_profile(pg)
            return {"FINISHED"}
        gaps = [(b[0] - a[0], (a[0] + b[0]) / 2) for a, b in zip(pts, pts[1:])]
        t = max(gaps)[1] if gaps else min(1.0, pts[-1][0] + 0.1)
        d = float(allometry.trunk_radius_fn(ns)(t * ns.height_m)) * 200.0
        pt = pg.profile.add()
        pt.height_ratio, pt.diameter_cm = t, d
        pg.profile_index = len(pg.profile) - 1
        return {"FINISHED"}


class TREEFORGE_OT_profile_remove(Operator):
    bl_idname = "treeforge.profile_remove"
    bl_label = "点を削除"
    bl_description = "選択中の太さプロファイルの点を削除する"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        pg, _ = props.active_group(context)
        return len(pg.profile) > 0

    def execute(self, context):
        pg, _ = props.active_group(context)
        pg.profile.remove(pg.profile_index)
        pg.profile_index = max(0, min(pg.profile_index, len(pg.profile) - 1))
        return {"FINISHED"}


class TREEFORGE_OT_profile_reset(Operator):
    bl_idname = "treeforge.profile_reset"
    bl_label = "胸高直径から初期化"
    bl_description = "胸高直径とテーパー指数から作った曲線を6点に標本化して置き直す"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        pg, _ = props.active_group(context)
        ns = props.read(pg)
        ns.trunk_profile_mode = "AUTO"
        props.set_profile(pg, allometry.auto_profile_points(ns, 5))
        return {"FINISHED"}


class TREEFORGE_OT_profile_sort(Operator):
    bl_idname = "treeforge.profile_sort"
    bl_label = "高さ順に並べる"
    bl_description = "太さプロファイルの点を、高さ比の小さい順に並べ直す"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        pg, _ = props.active_group(context)
        props.set_profile(pg, [(pt.height_ratio, pt.diameter_cm) for pt in pg.profile])
        return {"FINISHED"}


# --- マテリアル -------------------------------------------------------------
class TREEFORGE_OT_apply_materials(_TreeOperator):
    bl_idname = "treeforge.apply_materials"
    bl_label = "材質を適用"
    bl_description = "各部位の材質をスロットと葉・花の GN に反映する"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = _tree_obj(context)
        pg = obj.tree_forge
        materials.apply_all(obj, pg, props.read(pg))
        return {"FINISHED"}


class TREEFORGE_OT_auto_assign(Operator):
    bl_idname = "treeforge.auto_assign"
    bl_label = "名前から自動割り当て"
    bl_description = "マテリアル名のキーワード（bark, leaf, 枝 など）で各部位に割り当てる"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        pg, obj = props.active_group(context)
        found = materials.auto_assign(pg)
        if obj is not None:
            materials.apply_all(obj, pg, props.read(pg))
        self.report({"INFO"}, ", ".join(f"{props.PART_LABELS[k]}={v}" for k, v in found.items()) or "一致なし")
        return {"FINISHED"}


class TREEFORGE_OT_build_from_folder(Operator):
    bl_idname = "treeforge.build_from_folder"
    bl_label = "フォルダからマテリアル構築"
    bl_description = "フォルダ内のテクスチャをファイル名で判別して Principled BSDF を組み、選んだ部位に割り当てる"
    bl_options = {"REGISTER", "UNDO"}

    directory: StringProperty(subtype="DIR_PATH")
    part: bpy.props.EnumProperty(name="部位", items=[(k, v, "") for k, v in props.PART_LABELS.items()],
                                 default="trunk")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        pg, obj = props.active_group(context)
        try:
            mat, found = materials.build_from_folder(self.directory,
                                                     unknown_normal_is_directx=pg.materials.flip_normal_green)
        except (OSError, ValueError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        setattr(pg.materials, self.part, mat)
        if obj is not None:
            materials.apply_all(obj, pg, props.read(pg))
        self.report({"INFO"}, f"{mat.name}: " + ", ".join(sorted(found)))
        return {"FINISHED"}


class TREEFORGE_OT_import_blend_materials(Operator):
    bl_idname = "treeforge.import_blend_materials"
    bl_label = ".blend からマテリアルを取り込む"
    bl_description = "別の .blend のマテリアルを取り込み、名前のキーワードで各部位へ割り当てて木に反映する"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        got = materials.append_from_blend(self.filepath)
        pg, obj = props.active_group(context)
        if pg.materials.auto_assign or got:
            materials.auto_assign(pg)
        if obj is not None:
            materials.apply_all(obj, pg, props.read(pg))
        self.report({"INFO"}, f"{len(got)} 個のマテリアルを取り込みました")
        return {"FINISHED"}


# --- プリセット -------------------------------------------------------------
def _user_dir():
    return compat.user_preset_dir()


def _apply_preset_dict(context, d):
    pg, obj = props.active_group(context)
    props.write(pg, preset_io.overrides_from_dict(d))
    if obj is not None:
        build_mesh.regenerate(context, obj)


class TREEFORGE_OT_preset_save(Operator):
    bl_idname = "treeforge.preset_save"
    bl_label = "プリセットを保存"
    bl_description = "現在のパラメータ（幹の太さ点列を含む）をユーザープリセットとして保存する"

    name: StringProperty(name="名前", default="my_tree")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        pg, _ = props.active_group(context)
        d = preset_io.dump_user_preset(props.read(pg), self.name)
        path = _user_dir() / f"{self.name}.json"
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        self.report({"INFO"}, f"保存: {path}")
        return {"FINISHED"}


class TREEFORGE_OT_preset_load(Operator):
    bl_idname = "treeforge.preset_load"
    bl_label = "プリセットを読み込む"
    bl_description = "保存したプリセット（JSON）を読み込み、今の設定を置き換える。木を選択中なら再生成する"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        self.filepath = str(_user_dir()) + "/"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            d = json.loads(Path(self.filepath).read_text(encoding="utf-8"))
            _apply_preset_dict(context, d)
        except (OSError, ValueError, KeyError) as e:
            self.report({"ERROR"}, f"読み込めません: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}


class TREEFORGE_OT_preset_export(Operator):
    bl_idname = "treeforge.preset_export"
    bl_label = "プリセットを書き出す"
    bl_description = "JSON 1ファイルで共有できる形で書き出す"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        self.filepath = "tree_preset.json"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        pg, _ = props.active_group(context)
        d = preset_io.dump_user_preset(props.read(pg), Path(self.filepath).stem)
        Path(self.filepath).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"FINISHED"}


# --- 書き出し ---------------------------------------------------------------
class TREEFORGE_OT_realize_for_export(_TreeOperator):
    bl_idname = "treeforge.realize_for_export"
    bl_label = "書き出し用に実体化"
    bl_description = "葉・花のインスタンスを実メッシュにした複製を作る（FBX/glTF/USD 書き出し用）。元の木は変更しない"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = _tree_obj(context)
        pg = obj.tree_forge
        made = []
        trunk = bpy.data.objects.new(f"{obj.name}_export", obj.data.copy())
        trunk.matrix_world = obj.matrix_world
        context.collection.objects.link(trunk)
        made.append(trunk)
        lo = pg.leaves_object
        mod = lo.modifiers.get(gn_leaves.MODIFIER_NAME) if lo else None
        if mod is not None and len(lo.data.vertices) > 0:
            cp = gn_leaves.realize_copy(mod.node_group)
            tmp = bpy.data.objects.new("tf_tmp", lo.data.copy())
            tmp.matrix_world = obj.matrix_world
            context.collection.objects.link(tmp)
            tmp.modifiers.new(gn_leaves.MODIFIER_NAME, "NODES").node_group = cp
            dg = context.evaluated_depsgraph_get()
            me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg), depsgraph=dg)
            me.name = f"{obj.name}_leaves_export"
            leaves = bpy.data.objects.new(me.name, me)
            leaves.matrix_world = obj.matrix_world
            context.collection.objects.link(leaves)
            made.append(leaves)
            bpy.data.objects.remove(tmp)
            bpy.data.node_groups.remove(cp)
        for o in context.selected_objects:
            o.select_set(False)
        for o in made:
            o.select_set(True)
        context.view_layer.objects.active = made[0]
        self.report({"INFO"}, f"{len(made)} 個のオブジェクトを作りました（書き出しはこの複製を選んで行ってください）")
        return {"FINISHED"}


CLASSES = (
    TREEFORGE_OT_generate, TREEFORGE_OT_regenerate, TREEFORGE_OT_randomize_seed,
    TREEFORGE_OT_profile_add, TREEFORGE_OT_profile_remove, TREEFORGE_OT_profile_reset, TREEFORGE_OT_profile_sort,
    TREEFORGE_OT_apply_materials, TREEFORGE_OT_auto_assign, TREEFORGE_OT_build_from_folder,
    TREEFORGE_OT_import_blend_materials,
    TREEFORGE_OT_preset_save, TREEFORGE_OT_preset_load, TREEFORGE_OT_preset_export,
    TREEFORGE_OT_realize_for_export,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
