"""UI の描画コードをモックのレイアウトで通し、プロパティ名・オペレータ名の綴りミスを検出する。

ヘッドレスでは実際の描画ができないため、layout.prop / operator / template_list の引数だけを検証する。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import check, register_addon, reset_scene, run_all  # noqa: E402

import bpy  # noqa: E402

register_addon()
from tree_forge import preset_io, props  # noqa: E402
from tree_forge.blender import ui  # noqa: E402


class FakeLayout:
    def __init__(self, log):
        self.log = log
        self.scale_y = 1.0
        self.enabled = True
        self.active = True

    def __getattr__(self, name):
        def call(*a, **k):
            self.log.append((name, a, k))
            return FakeLayout(self.log)
        return call


def draw_all(context):
    log = []
    drawn = []
    for cls in ui.CLASSES:
        if not issubclass(cls, bpy.types.Panel):
            continue
        if hasattr(cls, "poll") and not cls.poll(context):
            continue
        self = type("P", (), {"layout": FakeLayout(log)})()
        cls.draw(self, context)
        drawn.append(cls.bl_idname)
    return log, drawn


def validate(log):
    for name, args, kw in log:
        if name == "prop":
            data, prop = args[0], args[1]
            assert prop in data.bl_rna.properties, f"{type(data).__name__}.{prop} が無い"
        elif name == "operator":
            op = args[0]
            mod, fn = op.split(".")
            assert hasattr(getattr(bpy.ops, mod), fn), f"オペレータ {op} が無い"
        elif name == "template_list":
            assert hasattr(bpy.types, args[0]), f"UIList {args[0]} が無い"
            data, prop = args[2], args[3]
            assert prop in data.bl_rna.properties


def setup(tree: bool, **ov):
    reset_scene()
    sc = bpy.context.scene
    props.write(sc.tree_forge, dict(preset_io.builtin_overrides("keyaki"), **ov))
    sc.tree_forge.initialized = True
    if tree:
        bpy.ops.treeforge.generate()
    return bpy.context


@check("UI: 木が無いとき（ひな形）の全パネル")
def _():
    ctx = setup(False)
    for adv in (False, True):
        ctx.scene.tree_forge.advanced = adv
        log, drawn = draw_all(ctx)
        validate(log)
        assert "TREEFORGE_PT_main" in drawn and "TREEFORGE_PT_materials" in drawn
        assert ("TREEFORGE_PT_level1" in drawn) == adv


@check("UI: 木を選択中の全パネル（AUTO / CUSTOM）")
def _():
    ctx = setup(True)
    pg = ctx.active_object.tree_forge
    pg.advanced = True
    for mode in ("AUTO", "CUSTOM"):
        pg.trunk_profile_mode = mode
        log, drawn = draw_all(ctx)
        validate(log)
        assert any(n == "template_list" for n, _, _ in log) == (mode == "CUSTOM")
    assert sum(1 for n, a, _ in log if n == "operator" and a[0] == "treeforge.regenerate") == 1


@check("UI: 太さ点の行描画")
def _():
    ctx = setup(True, trunk_profile_mode="CUSTOM", profile=[(0, 80), (1, 5)])
    pg = ctx.active_object.tree_forge
    lay_log = []
    item = pg.profile[0]
    ui.TREEFORGE_UL_profile.draw_item(type("U", (), {})(), ctx, FakeLayout(lay_log), pg, item, 0, pg, "profile_index", 0)
    validate(lay_log)
    assert any(n == "label" and "m" in k.get("text", "") for n, _, k in lay_log)


def drawn_props(ctx, idname):
    """パネル idname が layout.prop で描いたプロパティ名の集合。"""
    log = []
    cls = next(c for c in ui.CLASSES if getattr(c, "bl_idname", "") == idname)
    cls.draw(type("P", (), {"layout": FakeLayout(log)})(), ctx)
    return {a[1] for n, a, _ in log if n == "prop"}


@check("UI: 葉の量は主パネル、付き始め・最大数・付ける枝は詳細モードのときだけ（docs/PARAMETERS.md の記述どおり）")
def _():
    ctx = setup(False)
    pg = ctx.scene.tree_forge
    pg.advanced = False
    assert "foliage_amount" in drawn_props(ctx, "TREEFORGE_PT_main")
    simple_leaf = drawn_props(ctx, "TREEFORGE_PT_leaf")
    assert {"leaves_per_m", "leaf_cluster", "leaf_scale"} <= simple_leaf
    assert not ({"leaf_start", "leaf_max_instances", "leaf_reach", "foliage_amount"} & simple_leaf)
    pg.advanced = True
    adv_leaf = drawn_props(ctx, "TREEFORGE_PT_leaf")
    assert {"leaf_start", "leaf_max_instances", "leaf_reach"} <= adv_leaf and "foliage_amount" not in adv_leaf
    assert {"broken_pos_min", "top_break_height"} <= drawn_props(ctx, "TREEFORGE_PT_natural")
    pg.advanced = False
    assert not ({"broken_pos_min", "top_break_height"} & drawn_props(ctx, "TREEFORGE_PT_natural"))


@check("UI: 葉の量を変えて再生成すると、葉インスタンス数が増える")
def _():
    ctx = setup(True, leaf_max_instances=2_000_000, season=0.5)
    obj = ctx.active_object
    lo = obj.tree_forge.leaves_object
    n1 = len(lo.data.vertices)
    obj.tree_forge.foliage_amount = 2.0
    bpy.ops.treeforge.regenerate()
    n2 = len(obj.tree_forge.leaves_object.data.vertices)
    assert 1.8 < n2 / n1 < 2.2, (n1, n2)


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
