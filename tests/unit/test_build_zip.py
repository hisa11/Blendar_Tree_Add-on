"""zip のバージョン表記（どのビルドか分かるようにするための、semver のビルドメタデータ）。"""
import importlib.util
import json
import re
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Blender の拡張機能の検証（bl_pkg/cli/blender_ext.py の RE_MANIFEST_SEMVER）と同じ形
SEMVER = re.compile(
    r'^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)'
    r'(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?'
    r'(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
)


def _tool():
    spec = importlib.util.spec_from_file_location("build_zip", ROOT / "tools" / "build_zip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_version_string_is_valid_semver_with_build_metadata():
    t = _tool()
    v = t.make_version("0.1.0", "93", "d4453be", False)
    assert v == "0.1.0+93.d4453be" and SEMVER.match(v)
    d = t.make_version("0.1.0", "93", "d4453be", True)
    assert d == "0.1.0+93.d4453be.dirty" and SEMVER.match(d)
    assert SEMVER.match(t.make_version("0.1.0", "0", "nogit", False))


def test_zip_carries_the_build_version(tmp_path):
    out = _tool().main(tmp_path)
    base = tomllib.loads((ROOT / "tree_forge" / "blender_manifest.toml").read_text(encoding="utf-8"))["version"]
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        manifest = tomllib.loads(z.read("blender_manifest.toml").decode("utf-8"))
        info = json.loads(z.read("build_info.json").decode("utf-8"))
    assert SEMVER.match(manifest["version"]) and manifest["version"].startswith(base + "+")
    assert manifest["version"] == info["version"] and info["base_version"] == base
    assert out.name == f"tree_forge-{manifest['version']}.zip"
    assert names.count("blender_manifest.toml") == 1 and names.count("build_info.json") == 1
    assert not any("__pycache__" in n or n.endswith(".pyc") or n.startswith(("tests/", "tools/")) for n in names)
    # リポジトリ側のマニフェストはビルドで書き換えない
    assert tomllib.loads((ROOT / "tree_forge" / "blender_manifest.toml").read_text(encoding="utf-8"))["version"] == base
