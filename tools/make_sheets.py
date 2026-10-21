"""render_views.py の出力から、樹種ごとの見本シートを作る（行: 品質 PREVIEW / MEDIUM / HERO、列: サイズ倍率 ×0.7 / ×1.0 / ×1.4）。

    python tools/make_sheets.py views sheets       # views/*.png → sheets/<樹種>_{full,far,near,base}.jpg
PIL が要る。ラベルは日本語（IPAゴシックがあれば使う）。
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tree_forge import schema  # noqa: E402

QUALITIES = ("PREVIEW", "MEDIUM", "HERO")
QUALITY_JA = {"PREVIEW": "プレビュー", "MEDIUM": "標準", "HERO": "高品質"}
SCALES = ("×0.7", "×1.0", "×1.4")
VIEW_JA = {"full": "全体像", "far": "引き", "near": "寄り（樹冠）", "base": "根元"}
FONTS = ("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")


def font(size):
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sheet(views_dir: Path, sid: str, view: str):
    tiles = {}
    for q in QUALITIES:
        for vi in range(3):
            f = views_dir / f"{sid}_{q}_v{vi}_{view}.png"
            if f.exists():
                tiles[(q, vi)] = Image.open(f).convert("RGB")
    if not tiles:
        return None
    w, h = next(iter(tiles.values())).size
    bar = 26
    img = Image.new("RGB", (w * 3 + 90, bar + (h + 1) * 3), "white")
    d = ImageDraw.Draw(img)
    d.text((6, 4), f"{schema.SPECIES_NAMES.get(sid, sid)}（{sid}）— {VIEW_JA[view]}", fill=(0, 0, 0), font=font(16))
    for ri, q in enumerate(QUALITIES):
        y = bar + ri * (h + 1)
        d.text((4, y + h // 2 - 8), QUALITY_JA[q], fill=(0, 0, 0), font=font(14))
        for vi in range(3):
            if (q, vi) in tiles:
                img.paste(tiles[(q, vi)], (90 + vi * w, y))
                d.text((94 + vi * w, y + 3), f"{QUALITY_JA[q]} {SCALES[vi]}", fill=(255, 255, 255), font=font(13))
    return img


def main(views_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sid in schema.SPECIES_IDS:
        for view in VIEW_JA:
            img = sheet(Path(views_dir), sid, view)
            if img is not None:
                img.save(out / f"{sid}_{view}.jpg", quality=88)
                print("wrote", out / f"{sid}_{view}.jpg", img.size)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "views", sys.argv[2] if len(sys.argv) > 2 else "sheets")
