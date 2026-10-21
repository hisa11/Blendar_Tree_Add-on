"""README の画像（docs/images/）を作り直す。参道の 3 枚は tools/build_sando_scene.py で別に描く。

    python tools/render_docs_images.py                 # 全部
    python tools/render_docs_images.py --only species,lineup
    項目: species / lineup / near / trunk / sections / broken / foliage

  species : 13 樹種の全体像（標準品質・既定サイズ）を並べた表   → species.jpg
  lineup  : 13 樹種を 1 つの景色に並べた引きの絵（大きさの比較） → species_far.jpg
  near    : 13 樹種の寄り（樹冠の枝と葉の付き方、高品質）         → species_near.jpg
  trunk   : 幹の太さプロファイル 3 通り                        → trunk_profiles.jpg
  sections: 幹の断面を高さ別に重ねた図（いびつさ 0 / 0.35 / 0.7）→ sections.png
  broken  : 折れた枝（冬のケヤキ）                              → broken_branches.jpg
  foliage : 葉の量（foliage_amount 0.5 / 1 / 2 / 4）           → foliage_amount.jpg
bpy と PIL と matplotlib が要る。Cycles CPU。
"""
import argparse
import importlib.util
import math
import subprocess
import sys
import time
from pathlib import Path

import bpy
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "images"

import tree_forge  # noqa: E402
from tree_forge import preset_io, props, schema  # noqa: E402


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rv = _tool("render_views")
FONTS = ("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")


def font(size):
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def setup_render(w, h, spp):
    import addon_utils
    sc = bpy.context.scene
    addon_utils.enable("cycles", default_set=False)
    sc.render.engine = "CYCLES"
    sc.cycles.samples = spp
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except (TypeError, AttributeError):
        sc.cycles.use_denoising = False
    sc.view_settings.view_transform = "Standard"
    sc.render.resolution_x, sc.render.resolution_y = w, h
    return sc


def new_camera(sc, lens=50.0, ortho_scale=None):
    cam = bpy.data.cameras.new("cam")
    cam.lens = lens
    if ortho_scale:
        cam.type = "ORTHO"
        cam.ortho_scale = ortho_scale
    co = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    return co


def render_to(path):
    t = time.time()
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print(f"  {Path(path).name}: {time.time() - t:.0f} s", flush=True)


def tree_overrides(sid, **ov):
    d = preset_io.builtin_overrides(sid)
    d.update(ov)
    return d


# --------------------------------------------------------------------------------------
def make_species(tmp: Path):
    """13 樹種の全体像（標準品質・既定サイズ）。5 列 × 3 行。"""
    tiles = []
    for i, sid in enumerate(schema.SPECIES_IDS):
        rv.clear_scene()
        obj = rv.make_tree(tree_overrides(sid, quality="MEDIUM", seed=1 + i))
        p = props.read(obj.tree_forge)
        H, W = p.height_m, p.crown_width_m
        rv.build_environment(H, max(H, W) * 60)
        sc = setup_render(480, 600, 32)
        co = new_camera(sc)
        vfov = 2 * math.atan(18.0 / 50.0)
        hfov = 2 * math.atan(18.0 * 480 / 600 / 50.0)
        d = max(H * 0.56 / math.tan(vfov / 2), W * 0.60 / math.tan(hfov / 2))
        rv.aim(co, (d * 0.6, -d * 0.8, H * 0.42), (0, 0, H * 0.48))
        f = tmp / f"sp_{sid}.png"
        render_to(f)
        tiles.append((sid, f, H, p.dbh_cm))
    w, h = 480, 600
    cols, rows = 5, 3
    img = Image.new("RGB", (w * cols, h * rows), (235, 240, 248))
    d = ImageDraw.Draw(img)
    for i, (sid, f, H, dbh) in enumerate(tiles):
        x, y = (i % cols) * w, (i // cols) * h
        img.paste(Image.open(f).convert("RGB"), (x, y))
        d.rectangle([x, y, x + w, y + 30], fill=(0, 0, 0))
        d.text((x + 8, y + 5), f"{schema.SPECIES_NAMES[sid]}  {H:g} m", fill=(255, 255, 255), font=font(18))
    img = img.resize((img.width * 2 // 3, img.height * 2 // 3), Image.LANCZOS)
    img.save(OUT / "species.jpg", quality=88)


def make_lineup(tmp: Path):
    """13 樹種を高さ順に 1 つの景色へ。引きの絵（大きさの比較）。"""
    rv.clear_scene()
    species = sorted(schema.SPECIES_IDS, key=lambda s: schema.make_params(**preset_io.builtin_overrides(s)).height_m)
    x, placed = 0.0, []
    for i, sid in enumerate(species):
        obj = rv.make_tree(tree_overrides(sid, quality="MEDIUM", seed=11 + i))
        p = props.read(obj.tree_forge)
        x += p.crown_width_m * 0.5
        obj.location = (x, 0.0, 0.0)
        placed.append((sid, x, p.height_m))
        x += p.crown_width_m * 0.5 + 2.0
    total = x
    rv.build_environment(60, 4000)
    rv.add_person((-3.0, -2.0, 0))                       # 大きさの目安の人影（1.7 m）
    sc = setup_render(2000, 640, 32)
    co = new_camera(sc, ortho_scale=total + 6.0)
    # 横長（2000×640）。画面の幅 = ortho_scale。中心を木々の中央、高さ方向は地面が下から約 8% の位置
    view_h = (total + 6.0) * 640 / 2000
    rv.aim(co, (total / 2, -300.0, view_h * 0.42), (total / 2, 0, view_h * 0.42))
    f = tmp / "lineup.png"
    render_to(f)
    img = Image.open(f).convert("RGB")
    d = ImageDraw.Draw(img)
    scale = 2000 / (total + 6.0)
    for sid, xc, h in placed:
        px = (xc + 3.0) * scale
        d.text((px - 30, 640 - 22), schema.SPECIES_NAMES[sid][:6], fill=(255, 255, 255), font=font(13), anchor=None)
    img = img.resize((1600, 512), Image.LANCZOS)
    img.save(OUT / "species_far.jpg", quality=88)


def make_near(tmp: Path):
    """13 樹種の寄り（高品質）。樹冠の中ほどの外側へ 2.5〜7 m まで近づいて、枝の分かれ方と葉の付き方を見る。"""
    from tree_forge.core import envelope
    import numpy as np
    from mathutils import Vector
    w, h = 480, 360
    tiles = []
    for i, sid in enumerate(schema.SPECIES_IDS):
        rv.clear_scene()
        obj = rv.make_tree(tree_overrides(sid, quality="HERO", seed=1 + i))
        p = props.read(obj.tree_forge)
        H = p.height_m
        rv.build_environment(H, max(H, p.crown_width_m) * 60)
        sc = setup_render(w, h, 40)
        co = new_camera(sc)
        zt = (p.crown_base_ratio + 0.55 * (1.0 - p.crown_base_ratio)) * H
        reach = float(envelope.reach(p, np.array([zt]))[0])
        d = min(7.0, max(2.5, 0.12 * H))
        tgt = Vector((0.0, -0.7 * reach, zt))
        rv.aim(co, tgt + Vector((0.3, -1.0, 0.12)).normalized() * d, tgt)
        f = tmp / f"near_{sid}.png"
        render_to(f)
        tiles.append((sid, f))
    cols, rows = 5, 3
    img = Image.new("RGB", (w * cols, h * rows), (235, 240, 248))
    d = ImageDraw.Draw(img)
    for i, (sid, f) in enumerate(tiles):
        x, y = (i % cols) * w, (i // cols) * h
        img.paste(Image.open(f).convert("RGB"), (x, y))
        d.rectangle([x, y, x + w, y + 28], fill=(0, 0, 0))
        d.text((x + 8, y + 4), schema.SPECIES_NAMES[sid], fill=(255, 255, 255), font=font(17))
    img = img.resize((img.width * 2 // 3, img.height * 2 // 3), Image.LANCZOS)
    img.save(OUT / "species_near.jpg", quality=88)


def make_trunk(tmp: Path):
    """同じ木を、幹の太さプロファイルだけ変えて 3 本並べる。"""
    rv.clear_scene()
    variants = [
        ("AUTO (DBH 40 cm)", dict(trunk_profile_mode="AUTO", dbh_cm=40)),
        ("CUSTOM: 根元が太い", dict(trunk_profile_mode="CUSTOM", profile=[(0, 150), (0.04, 85), (0.2, 55), (0.6, 28), (1, 4)])),
        ("CUSTOM: 細く、途中で膨らむ", dict(trunk_profile_mode="CUSTOM", profile=[(0, 60), (0.25, 36), (0.55, 52), (0.8, 26), (1, 4)])),
    ]
    base = dict(preset_io.builtin_overrides("shirakaba"), height_m=12, crown_width_m=7, season=0.5, quality="MEDIUM", seed=4)
    for i, (label, ov) in enumerate(variants):
        o = rv.make_tree(dict(base, **ov))
        o.location = ((i - 1) * 9.0, 0, 0)
    rv.build_environment(12, 800)
    sc = setup_render(1500, 700, 32)
    co = new_camera(sc, 38.0)
    rv.aim(co, (0, -30, 5.5), (0, 0, 5.2))
    f = tmp / "trunk.png"
    render_to(f)
    img = Image.open(f).convert("RGB")
    d = ImageDraw.Draw(img)
    for i, (label, _) in enumerate(variants):
        d.text((i * 500 + 12, 10), label, fill=(0, 0, 0), font=font(22))
    img.resize((1100, 513), Image.LANCZOS).save(OUT / "trunk_profiles.jpg", quality=88)


def make_broken(tmp: Path):
    """折れた枝。冬（葉なし）のケヤキ。"""
    rv.clear_scene()
    rv.make_tree(tree_overrides("keyaki", quality="MEDIUM", season=0.0, seed=4, age_factor=1.4,
                                broken_branch_ratio=0.12, broken_hang_ratio=0.3, height_m=12, crown_width_m=10, dbh_cm=40))
    rv.build_environment(12, 800)
    sc = setup_render(700, 572, 48)
    co = new_camera(sc, 50.0)
    rv.aim(co, (11, -17, 5.0), (0, 0, 5.2))
    f = tmp / "broken.png"
    render_to(f)
    Image.open(f).convert("RGB").save(OUT / "broken_branches.jpg", quality=88)


def make_foliage(tmp: Path):
    """葉の量 foliage_amount 0.5 / 1 / 2 / 4（巨大スギの樹冠）。"""
    tiles = []
    for amount in (0.5, 1.0, 2.0, 4.0):
        rv.clear_scene()
        rv.make_tree(tree_overrides("sugi_giant", quality="MEDIUM", seed=7, foliage_amount=amount, leaf_max_instances=600000))
        rv.build_environment(55, 3000)
        sc = setup_render(520, 520, 24)
        co = new_camera(sc, 50.0)
        rv.aim(co, (30, -42, 33), (0, 0, 33))
        f = tmp / f"fol_{amount}.png"
        render_to(f)
        tiles.append((amount, f))
    img = Image.new("RGB", (1040, 1040))
    d = ImageDraw.Draw(img)
    for i, (a, f) in enumerate(tiles):
        x, y = (i % 2) * 520, (i // 2) * 520
        img.paste(Image.open(f).convert("RGB"), (x, y))
        d.rectangle([x, y, x + 190, y + 28], fill=(0, 0, 0))
        d.text((x + 8, y + 4), f"foliage_amount = {a:g}", fill=(255, 255, 255), font=font(16))
    img.save(OUT / "foliage_amount.jpg", quality=88)


def make_sections(tmp: Path):
    subprocess.run([sys.executable, str(ROOT / "tools" / "plot_sections.py"), str(OUT / "sections.png")], check=True)


MAKERS = {"species": make_species, "lineup": make_lineup, "near": make_near, "trunk": make_trunk, "sections": make_sections,
          "broken": make_broken, "foliage": make_foliage}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default=",".join(MAKERS))
    ap.add_argument("--tmp", default=str(ROOT / "dist" / "_tmp_images"))
    args = ap.parse_args()
    tree_forge.register()
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for name in args.only.split(","):
        print(f"== {name}", flush=True)
        MAKERS[name](tmp)


if __name__ == "__main__":
    main()
