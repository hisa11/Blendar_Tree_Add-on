"""Blender 側テスト共通。bpy（モジュール版または blender 本体）の中で実行する。"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402

RESULTS = []


def check(name):
    """テスト関数用デコレータ。失敗しても続行し、最後にまとめて報告する。"""
    def deco(fn):
        fn.__test_name__ = name
        RESULTS.append(fn)
        return fn
    return deco


def reset_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.images):
        for d in list(coll):
            coll.remove(d)


def run_all():
    failed = 0
    for fn in RESULTS:
        try:
            fn()
            print(f"PASS {fn.__test_name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__test_name__}")
            traceback.print_exc()
    print(f"--- Blender {bpy.app.version_string}: {len(RESULTS) - failed}/{len(RESULTS)} passed")
    return failed


def register_addon():
    import tree_forge
    tree_forge.register()
    return tree_forge
