"""テクスチャのファイル名 → マップ種別（bpy 非依存）。

区切りは _ - . 空白、大文字小文字は無視。法線は規約(OpenGL/DirectX)も拾う。
各配布サイトの実際の命名は一次確認していない。ここは一般的な慣習に基づく設計。
"""
from __future__ import annotations

import re

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".webp", ".bmp", ".tga"}

_ALBEDO = {"diff", "diffuse", "albedo", "basecolor", "color", "col"}
_NORMAL = {"normal", "nor", "nrm", "nml", "normalgl", "normaldx", "norgl", "nordx", "nrmgl", "nrmdx"}
_GL = {"gl", "opengl", "ogl"}
_DX = {"dx", "directx"}
_ROUGH = {"rough", "roughness"}
_AO = {"ao", "ambientocclusion", "occlusion"}
_DISP = {"disp", "displacement", "height"}
_OPACITY = {"opacity", "alpha", "mask"}
_TRANSL = {"translucency", "transmission"}
_METAL = {"metal", "metallic", "metalness"}

KINDS = ("albedo", "normal", "roughness", "arm", "ao", "displacement", "opacity", "translucency", "metallic")


def tokens(filename: str) -> list[str]:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    stem = re.sub(r"([a-z])([A-Z])", r"\1_\2", stem)          # BaseColor → Base_Color
    parts = [t for t in re.split(r"[_\-. ]+", stem.lower()) if t]
    # 隣り合う語を連結した候補（base + color → basecolor, nor + gl → norgl）も見る
    joined = [a + b for a, b in zip(parts, parts[1:])]
    return parts + joined


def classify(filename: str) -> tuple[str | None, str | None]:
    """(種別, 法線規約)。種別は KINDS のどれかまたは None。規約は 'GL' / 'DX' / None（不明）。"""
    t = set(tokens(filename))
    if t & _NORMAL:
        conv = "DX" if (t & _DX or t & {"normaldx", "nordx", "nrmdx"}) else \
               "GL" if (t & _GL or t & {"normalgl", "norgl", "nrmgl"}) else None
        return "normal", conv
    if "arm" in t:
        return "arm", None
    if t & _ROUGH:
        return "roughness", None
    if t & _AO:
        return "ao", None
    if t & _DISP:
        return "displacement", None
    if t & _OPACITY:
        return "opacity", None
    if t & _TRANSL:
        return "translucency", None
    if t & _METAL:
        return "metallic", None
    if t & _ALBEDO:
        return "albedo", None
    return None, None


def scan(names: list[str]) -> dict[str, tuple[str, str | None]]:
    """ファイル名の一覧 → {種別: (ファイル名, 法線規約)}。同じ種別が複数なら辞書順で最初。"""
    out: dict[str, tuple[str, str | None]] = {}
    for n in sorted(names):
        if "." + n.rsplit(".", 1)[-1].lower() not in IMAGE_EXT:
            continue
        kind, conv = classify(n)
        if kind and kind not in out:
            out[kind] = (n, conv)
    return out
