"""木を「全体像」と「引き」の 2 つの視点で描画する（見た目の確認・文書の画像用）。

    python tools/render_views.py --out views                                  # 13 樹種 × 3 品質 × 3 本 × 2 視点
    python tools/render_views.py --species keyaki --qualities MEDIUM --variants 1 --out views
    python tools/render_views.py --species sugi --views full --res 720 --spp 48 --out views   # 1 本を大きく

  full: 全体像。木全体を画面いっぱいに収める（枝ぶり・樹冠の形・幹）。
  far : 引き。人の背丈（1.7 m）の目線で、樹高の約 8 倍離れて見る。地平線が入り、遠目の樹形のまとまり・葉の密度・シルエットが分かる。
        足元に 1.7 m の人影を置いて大きさの目安にする。

木は tools/debug_sweep.py と同じ条件（サイズ倍率 0.7 / 1.0 / 1.4、別々のシード）で作る。出力は {樹種}_{品質}_v{番号}_{視点}.png。
bpy が要る（bpy モジュール、または blender -b --python）。Cycles CPU。
"""
import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tree_forge  # noqa: E402
from tree_forge import props, schema  # noqa: E402

VIEWS = ("full", "far")


def _sweep():
    spec = importlib.util.spec_from_file_location("debug_sweep", ROOT / "tools" / "debug_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.images, bpy.data.lights, bpy.data.cameras):
        for d in list(coll):
            coll.remove(d)
    for w in list(bpy.data.worlds):
        bpy.data.worlds.remove(w)


def make_tree(ov):
    sc = bpy.context.scene
    props.write(sc.tree_forge, ov)
    sc.tree_forge.initialized = True
    bpy.context.view_layer.objects.active = None
    assert bpy.ops.treeforge.generate() == {"FINISHED"}
    return bpy.context.active_object


def build_environment(H, ground_size):
    sc = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=ground_size, location=(0, 0, -0.03))
    gm = bpy.data.materials.new("ground")
    gm.use_nodes = True
    gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.22, 0.28, 0.12, 1)
    gm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.95
    bpy.context.active_object.data.materials.append(gm)
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.65, 0.9, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.5
    so = bpy.data.objects.new("sun", sun)
    sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(55), 0, math.radians(35))


def add_person(loc):
    """大きさの目安の人影（1.7 m）。"""
    m = bpy.data.materials.new("person")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.9, 0.35, 0.1, 1)
    for z, r, h in ((0.45, 0.13, 0.9), (1.2, 0.18, 0.6)):
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=(loc[0], loc[1], z))
        bpy.context.active_object.data.materials.append(m)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, location=(loc[0], loc[1], 1.62))
    bpy.context.active_object.data.materials.append(m)


def aim(cam_obj, loc, tgt):
    cam_obj.location = loc
    cam_obj.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()


def render_views(ov, out_dir: Path, stem: str, views=VIEWS, res=360, spp=10):
    """1 本を作って、指定の視点で描く。(樹高, 樹冠幅, 情報) を返す。"""
    clear_scene()
    obj = make_tree(ov)
    p = props.read(obj.tree_forge)
    H, W = p.height_m, p.crown_width_m
    build_environment(H, ground_size=max(H, W) * 60)
    sc = bpy.context.scene
    import addon_utils
    addon_utils.enable("cycles", default_set=False)
    sc.render.engine = "CYCLES"
    sc.cycles.samples = spp
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except (TypeError, AttributeError):
        sc.cycles.use_denoising = False
    sc.view_settings.view_transform = "Standard"
    cam = bpy.data.cameras.new("cam")
    cam.lens = 50
    co = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    info = obj.tree_forge.info
    for view in views:
        if view == "full":
            sc.render.resolution_x, sc.render.resolution_y = res, int(res * 1.25)
            vfov = 2 * math.atan(18.0 / cam.lens)                       # 縦長: 長辺（縦）にセンサー幅 36 mm
            hfov = 2 * math.atan(18.0 * sc.render.resolution_x / sc.render.resolution_y / cam.lens)
            d = max(H * 0.56 / math.tan(vfov / 2), W * 0.60 / math.tan(hfov / 2))
            aim(co, (d * 0.6, -d * 0.8, H * 0.42), (0, 0, H * 0.48))
        else:
            sc.render.resolution_x, sc.render.resolution_y = int(res * 4 / 3), int(res * 0.75)
            vfov = 2 * math.atan(18.0 * sc.render.resolution_y / sc.render.resolution_x / cam.lens)
            d = H / (0.30 * 2 * math.tan(vfov / 2))                   # 樹高が画面の約 30%
            aim(co, (0.3 * d, -0.95 * d, 1.7), (0, 0, H * 0.32))
            add_person((W * 0.35 + 1.5, -1.0, 0))
        sc.render.filepath = str(out_dir / f"{stem}_{view}.png")
        bpy.ops.render.render(write_still=True)
    return H, W, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="all")
    ap.add_argument("--qualities", default="PREVIEW,MEDIUM,HERO")
    ap.add_argument("--variants", default="0,1,2")
    ap.add_argument("--views", default="full,far")
    ap.add_argument("--out", default="views")
    ap.add_argument("--res", type=int, default=360, help="全体像の横幅 [px]（縦は 1.25 倍。引きは横 4/3 倍・縦 0.75 倍）")
    ap.add_argument("--spp", type=int, default=10)
    args = ap.parse_args()
    sweep = _sweep()
    tree_forge.register()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    species = list(schema.SPECIES_IDS) if args.species == "all" else args.species.split(",")
    for sid in species:
        si = list(schema.SPECIES_IDS).index(sid)
        for q in args.qualities.split(","):
            qi = sweep.QUALITIES.index(q)
            for vi in map(int, args.variants.split(",")):
                ov = sweep.variant_overrides(sid, si, qi, vi, sweep.SIZE_SCALES[vi])
                t0 = time.time()
                H, W, info = render_views(ov, out, f"{sid}_{q}_v{vi}", args.views.split(","), args.res, args.spp)
                print(f"{sid} {q} v{vi} x{sweep.SIZE_SCALES[vi]} H={H:.1f} W={W:.1f} | {info} | {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
