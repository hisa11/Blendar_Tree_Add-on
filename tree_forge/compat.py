"""Blender のバージョン差を吸収する（機能検出のみ。バージョン番号では分岐しない）。"""
from __future__ import annotations

from pathlib import Path

import bpy


def has_thin_wall(principled) -> bool:
    """Principled BSDF に Thin Wall 入力があるか（Blender 5.2 以降）。"""
    return "Thin Wall" in principled.inputs


def set_normal_map_directx(node, directx: bool) -> bool:
    """Normal Map ノードの法線規約を切り替える。プロパティが無い版では False を返す。"""
    prop = node.bl_rna.properties.get("convention")
    if prop is None:
        return False
    ids = {i.identifier for i in prop.enum_items}
    want = "DIRECTX" if directx else "OPENGL"
    if want not in ids:
        return False
    node.convention = want
    return True


def ensure_node_tree(mat) -> None:
    if mat.node_tree is None:
        mat.use_nodes = True


def user_preset_dir(package: str | None) -> Path:
    """ユーザープリセットの保存先。拡張機能として入っていればその専用領域、そうでなければ scripts 配下。"""
    try:
        return Path(bpy.utils.extension_path_user(package, path="presets", create=True))
    except Exception:  # noqa: BLE001 - 拡張機能でない読み込み（開発時・テスト）
        d = Path(bpy.utils.user_resource("SCRIPTS", path="presets/tree_forge", create=True))
        d.mkdir(parents=True, exist_ok=True)
        return d
