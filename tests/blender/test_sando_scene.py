"""参道シーン（tools/build_sando_scene.py）で、御神木に足したものを個別に切れること。

シーンはプロセスごとに作り直す（コレクション名が重複しないように）ため、子プロセスで実行して結果行を読む。
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, check, run_all  # noqa: E402

SCRIPT = ROOT / "tools" / "build_sando_scene.py"
EXTRA_OBJECTS = {"shimenawa": "Shimenawa", "shide": "Shide", "mound": "Hero_Mound", "litter": "Hero_Litter"}
BASE = ["--no-render", "--hero-quality", "MEDIUM", "--bg-trees", "2", "--forest", "0"]


def child(flags):
    spec = importlib.util.spec_from_file_location("sando", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import bpy
    mod.main(BASE + flags)
    hero = bpy.data.objects["Cedar_Hero"]
    print("RESULT " + json.dumps({
        "objects": sorted(o.name for o in bpy.data.objects if o.name in EXTRA_OBJECTS.values()),
        "collections": sorted(c.name for c in bpy.data.collections if c.name.startswith("Hero_")),
        "bark": [m.name for m in hero.data.materials if m][0],
        "stubs": hero.tree_forge.trunk_stubs,
    }))


def run(flags):
    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", *flags],
                         capture_output=True, text=True, timeout=600)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, out.stdout[-2000:] + out.stderr[-2000:]
    return json.loads(lines[-1][len("RESULT "):])


@check("参道: 既定では御神木の飾りがすべて入る")
def test_all_on():
    r = run([])
    assert r["objects"] == sorted(EXTRA_OBJECTS.values())
    assert r["collections"] == ["Hero_Extras", "Hero_Litter", "Hero_Mound", "Hero_Shide", "Hero_Shimenawa"]
    assert r["bark"] == "Sando_SugiBark_Hero" and r["stubs"] is True


@check("参道: --no-extras で飾りが全部消え、元の木だけになる")
def test_no_extras():
    r = run(["--no-extras"])
    assert r["objects"] == [] and r["collections"] == []
    assert r["bark"].endswith("_NoMoss") and r["stubs"] is False


@check("参道: 項目ごとに個別に切れる")
def test_individual():
    r = run(["--no-shimenawa", "--no-litter"])
    assert r["objects"] == ["Hero_Mound", "Shide"] and "Hero_Shimenawa" not in r["collections"]
    r = run(["--no-shide", "--no-mound", "--no-bark-moss", "--no-stubs"])
    assert r["objects"] == ["Hero_Litter", "Shimenawa"]
    assert r["bark"] == "Sando_SugiBark_Hero_NoMoss" and r["stubs"] is False


@check("参道: 粗い品質でも（リング間隔が広い PREVIEW、シード 2）注連縄の位置決めが落ちない")
def test_preview_hero_does_not_crash():
    r = run(["--hero-quality", "PREVIEW", "--seed", "2"])
    assert r["objects"] == sorted(EXTRA_OBJECTS.values())


if __name__ == "__main__":
    if "--child" in sys.argv:
        child(sys.argv[sys.argv.index("--child") + 1:])
    else:
        sys.exit(1 if run_all() else 0)
