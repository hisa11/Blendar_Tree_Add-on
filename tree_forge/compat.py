"""Blender のバージョン差を吸収する（機能検出のみ。バージョン番号では分岐しない）。"""
from __future__ import annotations

from pathlib import Path

import bpy


def build_label() -> str:
    """このビルドの表示用バージョン（zip 化したときに入る build_info.json から。無ければ開発版）。

    リリース版（安定版）: 0.1.0 安定版（stable）。開発版（デバッグ中）: 0.1.0+dev.93.d4453be 開発版（93 コミット目、コミット d4453be）。
    未コミットの変更を含むと末尾に .dirty。"""
    import json
    try:
        info = json.loads((Path(__file__).parent / "build_info.json").read_text(encoding="utf-8"))
        if info.get("release"):
            return f'{info["base_version"]} 安定版（{info["channel"]}、{info["built_at"]}）'
        return f'{info["version"]} 開発版（{info["built_at"]}）'
    except (OSError, ValueError, KeyError):
        return "開発版（zip 化していない）"


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


def user_preset_dir(package: str | None = None) -> Path:
    """ユーザープリセットの保存先。拡張機能として入っていればその専用領域、そうでなければ scripts 配下。

    package の既定はこのアドオンのルートパッケージ（拡張機能なら bl_ext.<repo>.tree_forge）。
    """
    try:
        return Path(bpy.utils.extension_path_user(package or __package__, path="presets", create=True))
    except Exception:  # noqa: BLE001 - 拡張機能でない読み込み（開発時・テスト）
        d = Path(bpy.utils.user_resource("SCRIPTS", path="presets/tree_forge", create=True))
        d.mkdir(parents=True, exist_ok=True)
        return d
