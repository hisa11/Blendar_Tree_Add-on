"""N パネル「TreeForge」。"""
from __future__ import annotations

import bpy
from bpy.types import Panel, UIList

from .. import props, schema
from ..core import allometry

CATEGORY = "TreeForge"


def _group(context):
    return props.active_group(context)


def _draw_params(layout, pg, group, skip=()):
    for p in schema.PARAMS:
        if p.group == group and p.name not in skip:
            layout.prop(pg, p.name)


class _Panel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY


class TREEFORGE_UL_profile(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "height_ratio", text="高さ比")
        row.label(text=f"{item.height_ratio * data.height_m:5.2f} m")
        row.prop(item, "diameter_cm", text="直径 cm")


class TREEFORGE_PT_main(_Panel):
    bl_idname = "TREEFORGE_PT_main"
    bl_label = "TreeForge"

    def draw(self, context):
        pg, obj = _group(context)
        lay = self.layout
        lay.prop(pg, "species")
        row = lay.row(align=True)
        row.prop(pg, "seed")
        row.operator("treeforge.randomize_seed", text="", icon="FILE_REFRESH")
        lay.prop(pg, "quality")
        col = lay.column(align=True)
        col.prop(pg, "height_m")
        col.prop(pg, "crown_width_m")
        if pg.trunk_profile_mode == "AUTO":
            col.prop(pg, "dbh_cm")
        else:
            ns = props.read(pg)
            col.label(text=f"胸高直径(1.3 m): {allometry.profile_dbh_cm(ns):.1f} cm（太さプロファイルから）")
        lay.prop(pg, "crown_shape")
        lay.prop(pg, "season", slider=True)
        lay.prop(pg, "foliage_amount", slider=True)
        lay.prop(pg, "age_factor", slider=True)
        lay.prop(pg, "leaf_geom")
        row = lay.row(align=True)
        row.scale_y = 1.4
        if obj is None:
            row.operator("treeforge.generate", icon="OUTLINER_OB_FORCE_FIELD")
        else:
            row.operator("treeforge.regenerate", icon="FILE_REFRESH")
            row.operator("treeforge.generate", text="新規", icon="ADD")
            lay.prop(pg, "live_update")
            if pg.info:
                lay.label(text=pg.info)
        lay.prop(pg, "advanced", icon="PREFERENCES")


class TREEFORGE_PT_trunk(_Panel):
    bl_idname = "TREEFORGE_PT_trunk"
    bl_label = "幹の太さ"
    bl_parent_id = "TREEFORGE_PT_main"

    def draw(self, context):
        pg, _ = _group(context)
        lay = self.layout
        lay.prop(pg, "trunk_profile_mode", expand=True)
        if pg.trunk_profile_mode == "AUTO":
            lay.prop(pg, "trunk_taper_exp")
            lay.operator("treeforge.profile_reset", icon="CURVE_PATH")
            return
        lay.label(text="高さ比 t（0 根元〜1 樹頂）での直径を指定。間は滑らかに補間")
        row = lay.row()
        row.template_list("TREEFORGE_UL_profile", "", pg, "profile", pg, "profile_index", rows=5)
        col = row.column(align=True)
        col.operator("treeforge.profile_add", text="", icon="ADD")
        col.operator("treeforge.profile_remove", text="", icon="REMOVE")
        col.separator()
        col.operator("treeforge.profile_sort", text="", icon="SORTSIZE")
        lay.operator("treeforge.profile_reset", icon="CURVE_PATH")


class TREEFORGE_PT_natural(_Panel):
    bl_idname = "TREEFORGE_PT_natural"
    bl_label = "いびつさ・損傷"
    bl_parent_id = "TREEFORGE_PT_main"

    def draw(self, context):
        pg, _ = _group(context)
        lay = self.layout
        col = lay.column(align=True)
        col.label(text="断面の歪み")
        col.prop(pg, "trunk_irregularity", slider=True)
        col.prop(pg, "branch_irregularity", slider=True)
        col = lay.column(align=True)
        col.label(text="損傷（樹齢感で増減）")
        col.prop(pg, "broken_branch_ratio", slider=True)
        col.prop(pg, "broken_hang_ratio", slider=True)
        col.prop(pg, "dead_branch_ratio", slider=True)
        col.prop(pg, "trunk_stubs")
        sub = col.column(align=True)
        sub.active = pg.trunk_stubs                                 # オフのときは数の欄を薄くする
        sub.prop(pg, "trunk_stub_density")
        if pg.advanced:
            col.prop(pg, "broken_pos_min")
            col.prop(pg, "broken_pos_max")
            col.prop(pg, "top_break_height")


class TREEFORGE_PT_advanced_global(_Panel):
    bl_idname = "TREEFORGE_PT_advanced_global"
    bl_label = "詳細: 全体・幹"
    bl_parent_id = "TREEFORGE_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return _group(context)[0].advanced

    def draw(self, context):
        pg, _ = _group(context)
        _draw_params(self.layout, pg, "general",
                     skip=("species", "seed", "height_m", "dbh_cm", "crown_width_m", "crown_shape", "quality",
                           "age_factor", "season"))
        self.layout.separator()
        _draw_params(self.layout, pg, "trunk",
                     skip=("trunk_profile_mode", "trunk_taper_exp", "trunk_irregularity", "branch_irregularity"))


def _level_panel(n):
    class _P(_Panel):
        bl_idname = f"TREEFORGE_PT_level{n}"
        bl_label = f"詳細: 枝 L{n}"
        bl_parent_id = "TREEFORGE_PT_main"
        bl_options = {"DEFAULT_CLOSED"}

        @classmethod
        def poll(cls, context):
            return _group(context)[0].advanced

        def draw(self, context):
            pg, _ = _group(context)
            _draw_params(self.layout, pg, f"level{n}")

    _P.__name__ = f"TREEFORGE_PT_level{n}"
    return _P


class TREEFORGE_PT_leaf(_Panel):
    bl_idname = "TREEFORGE_PT_leaf"
    bl_label = "葉・花"
    bl_parent_id = "TREEFORGE_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        pg, _ = _group(context)
        hidden = ("foliage_amount",)                       # 主パネルにある
        if not pg.advanced:
            hidden += ("fascicle_count", "leaf_scale_var", "leaf_width_ratio", "bark_type", "leaf_reach",
                       "leaf_start", "leaf_max_instances")
        _draw_params(self.layout, pg, "leaf", skip=hidden)
        if pg.leaf_geom == "CUSTOM_OBJECT":
            self.layout.prop(pg, "leaf_object")


class TREEFORGE_PT_materials(_Panel):
    bl_idname = "TREEFORGE_PT_materials"
    bl_label = "マテリアル"

    def draw(self, context):
        pg, obj = _group(context)
        lay = self.layout
        col = lay.column(align=True)
        for part in props.PARTS:
            col.prop(pg.materials, part, text=props.PART_LABELS[part])
        lay.prop(pg.materials, "auto_assign")
        row = lay.row(align=True)
        row.operator("treeforge.auto_assign", icon="MATERIAL")
        row.operator("treeforge.apply_materials", icon="CHECKMARK")
        lay.operator("treeforge.build_from_folder", icon="FILE_FOLDER")
        lay.prop(pg.materials, "flip_normal_green")
        lay.operator("treeforge.import_blend_materials", icon="APPEND_BLEND")
        col = lay.column(align=True)
        col.prop(pg, "bark_tile_w_m")
        col.prop(pg, "bark_tile_h_m")
        lay.prop(pg, "uv_integer_wrap")
        lay.prop(pg, "junction_blend_len")


class TREEFORGE_PT_presets(_Panel):
    bl_idname = "TREEFORGE_PT_presets"
    bl_label = "プリセット・書き出し"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        lay = self.layout
        col = lay.column(align=True)
        col.operator("treeforge.preset_save", icon="FILE_TICK")
        col.operator("treeforge.preset_load", icon="FILE_FOLDER")
        col.operator("treeforge.preset_export", icon="EXPORT")
        lay.separator()
        lay.operator("treeforge.realize_for_export", icon="MOD_PARTICLES")


CLASSES = (
    TREEFORGE_UL_profile, TREEFORGE_PT_main, TREEFORGE_PT_trunk, TREEFORGE_PT_natural,
    TREEFORGE_PT_advanced_global, _level_panel(1), _level_panel(2), _level_panel(3), TREEFORGE_PT_leaf,
    TREEFORGE_PT_materials, TREEFORGE_PT_presets,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
