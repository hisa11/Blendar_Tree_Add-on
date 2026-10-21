"""全テストを順に実行する。

    python tests/run_all.py
    blender -b --factory-startup --python-exit-code 1 --python tests/run_all.py

- 単体テスト（bpy 不要）: pytest があれば pytest、なければ unittest 互換の最小ランナーは無いので失敗扱い
- Blender テスト: bpy が import できる（= bpy モジュール or blender 本体の中）ときだけ実行
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    rc = 0
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest が無いので単体テストをスキップ（pip install pytest）")
        rc = 1
    else:
        rc |= subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests" / "unit"), "-q"], cwd=ROOT)
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("bpy が無いので Blender テストをスキップ")
    else:
        for t in ("test_addon.py", "test_ui.py", "test_sando_scene.py"):
            rc |= subprocess.call([sys.executable, str(ROOT / "tests" / "blender" / t)], cwd=ROOT)
    return rc


if __name__ == "__main__":
    sys.exit(main())
