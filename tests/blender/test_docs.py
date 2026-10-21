"""docs/PARAMETERS.md と docs/PRESET_VALUES.md が、コードとプリセットから生成した内容と一致していること。

設定項目を足したり説明を変えたりしたのに文書を作り直し忘れると落ちる。直し方: python tools/dump_params.py
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, check, register_addon, run_all  # noqa: E402

register_addon()


def _tool():
    spec = importlib.util.spec_from_file_location("dump_params", ROOT / "tools" / "dump_params.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same(name, generated):
    on_disk = (ROOT / "docs" / name).read_text(encoding="utf-8")
    # dump_params は "\n".join(...) で書き、末尾に改行を付けないので、そのまま比べる
    assert on_disk == generated, f"docs/{name} が古い。`python tools/dump_params.py` で作り直す"


@check("文書: PARAMETERS.md が最新（コードから再生成した内容と一致）")
def test_parameters_md_is_current():
    _same("PARAMETERS.md", _tool().make_parameters_md())


@check("文書: PRESET_VALUES.md が最新")
def test_preset_values_md_is_current():
    _same("PRESET_VALUES.md", _tool().make_preset_values_md())


@check("文書: 全パラメータに説明があり、PARAMETERS.md の全項目表に載っている")
def test_every_param_documented():
    from tree_forge import schema
    doc = (ROOT / "docs" / "PARAMETERS.md").read_text(encoding="utf-8")
    missing = [p.name for p in schema.PARAMS if not p.desc.strip()]
    assert not missing, f"説明が空: {missing}"
    absent = [p.name for p in schema.PARAMS if f"`{p.name}` |" not in doc.split("## 2. パラメータ全項目")[1]]
    assert not absent, f"全項目表に無い: {absent}"


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
