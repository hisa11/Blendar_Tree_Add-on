"""全設定項目の一覧（docs/PARAMETERS.md、docs/PRESET_VALUES.md）をコードから生成する。

    python tools/dump_params.py                                   # bpy モジュールが入った Python で
    blender -b --factory-startup --python tools/dump_params.py    # Blender 本体でも可

手で書かずにコードから作る（実装とずれない）。何を読んでいるか:
  - パラメータの名前・単位・範囲・既定値・説明  ← tree_forge/schema.py
  - どのパネルに出るか・どの条件で出るか      ← パネルの draw() を、設定を変えながら実際に呼んで記録（ui.py）
  - ボタン（オペレータ）の名前と説明           ← tree_forge/blender/operators.py
  - UI 専用の設定                              ← tree_forge/props.py の PropertyGroup
  - 品質レベル                                 ← tree_forge/core/allometry.py
  - 樹種プリセットの値                         ← tree_forge/presets/*.json
  - 参道シーンのオプション                     ← tools/build_sando_scene.py の build_parser()
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import bpy
except ImportError:
    sys.exit("bpy が import できません。bpy モジュールを入れた Python か、blender -b --python で実行してください。")

import tree_forge  # noqa: E402
from tree_forge import preset_io, props, schema  # noqa: E402
from tree_forge.blender import operators, ui  # noqa: E402
from tree_forge.core import allometry  # noqa: E402

GROUPS = [
    ("general", "全体"), ("trunk", "幹・太さ・断面"), ("damage", "損傷（折れ・枯れ）"),
    ("level1", "枝 L1（主枝）"), ("level2", "枝 L2"), ("level3", "枝 L3（小枝）"),
    ("leaf", "葉・花・針"), ("material", "樹皮の UV・材質"),
]


# --------------------------------------------------------------------------------------
# 値の整形
# --------------------------------------------------------------------------------------
def fmt_value(p, v):
    if p.kind == "enum":
        return dict(p.items).get(v, str(v))
    if p.kind == "bool":
        return "オン" if v else "オフ"
    if p.kind == "color":
        return "(" + ", ".join(f"{c:.2f}" for c in v) + ")"
    return f"{v:g}" if isinstance(v, float) else str(v)


def fmt_range(p):
    if p.kind in ("float", "int"):
        def f(x):
            return f"{int(x):,}" if float(x).is_integer() else f"{x:g}"
        return f"{f(p.lo)} 〜 {f(p.hi)}"
    if p.kind == "enum":
        return " / ".join(lbl for _, lbl in p.items)
    return ""


def esc(s):
    return str(s).replace("|", "/").replace("\n", " ")


# --------------------------------------------------------------------------------------
# UI の走査: パネルの draw() を、条件を変えながら呼んで、何がいつ出るかを記録する
# --------------------------------------------------------------------------------------
class FakeLayout:
    def __init__(self, log):
        self.log = log
        self.scale_y = 1.0
        self.enabled = True
        self.active = True

    def __getattr__(self, name):
        def call(*a, **k):
            self.log.append((name, a, k))
            return FakeLayout(self.log)
        return call


# 画面の表示を変える条件（次元）。(木を選択中か, 詳細モード, 太さの指定方法, 葉の表現)
DIMS = ("tree", "advanced", "profile_mode", "leaf_geom")
DIM_TEXT = {
    "tree": {False: "木を選んでいないとき", True: "木を選択中"},
    "advanced": {True: "詳細モード", False: "詳細モードがオフ"},
    "profile_mode": {"CUSTOM": "太さが「指定」のとき", "AUTO": "太さが「自動」のとき"},
    "leaf_geom": {"CUSTOM_OBJECT": "葉の表現が「任意オブジェクト」のとき", "CUTOUT_MESH": "葉の表現が「任意オブジェクト」以外のとき"},
}


def scan_ui():
    sc = bpy.context.scene
    panels = [c for c in ui.CLASSES if issubclass(c, bpy.types.Panel)]
    items = {}                                       # (panel_id, kind, ident) -> {"label","present"}
    panel_present = {c.bl_idname: set() for c in panels}
    combos = []
    for tree in (False, True):
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o)
        props.write(sc.tree_forge, dict(preset_io.builtin_overrides("keyaki"), quality="PREVIEW"))
        sc.tree_forge.initialized = True
        if tree:
            assert bpy.ops.treeforge.generate() == {"FINISHED"}
        ctx = bpy.context
        pg, _ = props.active_group(ctx)
        for adv in (False, True):
            for mode in ("AUTO", "CUSTOM"):
                for geom in ("CUTOUT_MESH", "CUSTOM_OBJECT"):
                    pg.advanced, pg.trunk_profile_mode, pg.leaf_geom = adv, mode, geom
                    combo = (tree, adv, mode, geom)
                    combos.append(combo)
                    for cls in panels:
                        if hasattr(cls, "poll") and not cls.poll(ctx):
                            continue
                        panel_present[cls.bl_idname].add(combo)
                        log = []
                        cls.draw(type("P", (), {"layout": FakeLayout(log)})(), ctx)
                        for name, a, k in log:
                            rec = describe_call(name, a, k)
                            if rec:
                                kind, ident, label = rec
                                d = items.setdefault((cls.bl_idname, kind, ident), {"label": label, "present": set()})
                                d["present"].add(combo)
    return panels, items, panel_present, combos


def describe_call(name, a, k):
    if name == "prop":
        data, prop = a[0], a[1]
        owner = type(data).__name__
        if owner.startswith("TF_TreeParams"):
            ident = prop
            label = schema.BY_NAME[prop].label if prop in schema.BY_NAME else data.bl_rna.properties[prop].name
        else:
            label = data.bl_rna.properties[prop].name
            ident = f"{'materials' if 'PartMaterials' in owner else owner}.{prop}"
        text = k.get("text")
        return "prop", ident, (text if isinstance(text, str) and text else label)
    if name == "operator":
        idname = a[0]
        rna = getattr(getattr(bpy.ops, idname.split(".")[0]), idname.split(".")[1]).get_rna_type()
        text = k.get("text")
        return "operator", idname, (text if isinstance(text, str) and text else rna.name)
    if name == "template_list":
        return "list", a[3], "太さプロファイルの点の一覧（高さ比・直径 cm）"
    return None


def condition_text(present, combos):
    """出ている条件の組み合わせから、「常に」「詳細モード」などの説明を作る。"""
    allc = set(combos)
    if present == allc:
        return "常に"
    constraints = []
    for i, dim in enumerate(DIMS):
        vals = {c[i] for c in present}
        if len(vals) == 1:
            constraints.append((i, next(iter(vals))))
    expected = {c for c in allc if all(c[i] == v for i, v in constraints)}
    if expected == present and constraints:
        return " かつ ".join(DIM_TEXT[DIMS[i]][v] for i, v in constraints)
    return "条件つき（" + "、".join(sorted({" / ".join(str(x) for x in c) for c in present})[:3]) + " ほか）"


def where_text(param_name, items, panels_by_id, combos):
    """パラメータが出るパネル（と条件）の短い説明。"""
    out = []
    for (pid, kind, ident), d in items.items():
        if kind == "prop" and ident == param_name:
            cond = condition_text(d["present"], combos)
            lab = panels_by_id[pid].bl_label
            out.append(f"{lab}" + ("" if cond == "常に" else f"（{cond}）"))
    return "、".join(out) if out else "—（画面には出ない。プリセット・書き出し用）"


# --------------------------------------------------------------------------------------
# 文書の生成
# --------------------------------------------------------------------------------------
def operator_list():
    out = []
    for c in operators.CLASSES:
        mod, fn = c.bl_idname.split(".")
        rna = getattr(getattr(bpy.ops, mod), fn).get_rna_type()
        out.append((c.bl_idname, rna.name, rna.description))
    return out


def tree_level_props():
    hide = {"rna_type", "name", "profile", "materials", "is_tree", "initialized", "leaves_object", "leaf_src", "flower_src", "info", "profile_index"}
    rows = []
    for pr in props.TF_TreeParams.bl_rna.properties:
        if pr.identifier in schema.BY_NAME or pr.identifier in hide:
            continue
        rows.append((pr.identifier, pr.name, pr.description))
    return rows


def sub_props(cls):
    return [(pr.identifier, pr.name, pr.description) for pr in cls.bl_rna.properties if pr.identifier not in ("rna_type", "name")]


def make_parameters_md():
    panels, items, panel_present, combos = scan_ui()
    panels_by_id = {c.bl_idname: c for c in panels}
    L = []
    A = L.append
    A("# 設定項目一覧（全項目）")
    A("")
    A("`tools/dump_params.py` が、コードとプリセット JSON から生成します（**直接編集しないでください**）。")
    A("パネルの表示は、画面の描画コードを設定を変えながら実際に呼んで記録した結果です。")
    A("")
    A(f"- パラメータ {len(schema.PARAMS)} 項目（第 2 章）、UI 専用の設定、ボタン {len(operator_list())} 個、品質 3 段階、樹種プリセット {len(schema.SPECIES_IDS)} 種、参道シーンのオプション。")
    A("- 樹種ごとの全項目の値は **[PRESET_VALUES.md](PRESET_VALUES.md)** にあります。")
    A("- 木を選んでいる間は、その木の設定を表示・編集します。選んでいないときは、次に「木を生成」で作る木の設定（ひな形）です。")
    A("")
    A("**目次**: [1 画面のどこにあるか](#1-画面のどこにあるか) / [2 パラメータ全項目](#2-パラメータ全項目) / "
      "[3 UI 専用の設定](#3-ui-専用の設定) / [4 ボタン](#4-ボタン操作) / [5 品質レベル](#5-品質レベル) / "
      "[6 プリセットの書式](#6-プリセットの書式) / [7 参道シーンのオプション](#7-参道シーンのオプション)")
    A("")

    # ---- 1. 画面 ----
    A("## 1. 画面のどこにあるか")
    A("")
    A("3D ビューの `N` キー → 「TreeForge」タブ。パネルは上から次の順です（入れ子は主パネルの下に並びます）。")
    A("")
    for cls in panels:
        pid = cls.bl_idname
        head = "  " * (1 if getattr(cls, "bl_parent_id", "") else 0)
        closed = "（既定で閉じている）" if "DEFAULT_CLOSED" in getattr(cls, "bl_options", set()) else ""
        cond = condition_text(panel_present[pid], combos)
        A(f"### {cls.bl_label}{closed}")
        A("")
        A(f"- パネルが出る条件: {cond}")
        A("")
        rows = [(kind, ident, d) for (p, kind, ident), d in items.items() if p == pid]
        if not rows:
            A("（項目なし）")
            A("")
            continue
        A("| 種類 | 項目 | 内部名 | 出る条件 |")
        A("|---|---|---|---|")
        for kind, ident, d in rows:
            kind_t = {"prop": "設定", "operator": "ボタン", "list": "一覧"}[kind]
            A(f"| {kind_t} | {esc(d['label'])} | `{ident}` | {condition_text(d['present'], combos)} |")
        A("")
        del head
    # ---- 2. パラメータ ----
    A("## 2. パラメータ全項目")
    A("")
    A("「表示場所」は、第 1 章のパネル名です。`—` は画面に出さず、プリセット（JSON）と書き出しでだけ扱う項目です。")
    A("")
    for key, title in GROUPS:
        ps = [p for p in schema.PARAMS if p.group == key]
        if not ps:
            continue
        A(f"### {title}（{len(ps)} 項目）")
        A("")
        A("| 項目 | 内部名 | 単位 | 範囲・選択肢 | 既定 | 表示場所 | 説明 |")
        A("|---|---|---|---|---|---|---|")
        for p in ps:
            A(f"| {esc(p.label)} | `{p.name}` | {p.unit} | {esc(fmt_range(p))} | {esc(fmt_value(p, p.default))} | "
              f"{esc(where_text(p.name, items, panels_by_id, combos))} | {esc(p.desc)} |")
        A("")
    # ---- 3. UI 専用 ----
    A("## 3. UI 専用の設定")
    A("")
    A("パラメータ表（第 2 章）にない、画面の動作や材質の割り当てのための設定です。")
    A("")
    A("| 項目 | 内部名 | 説明 |")
    A("|---|---|---|")
    for ident, name, desc in tree_level_props():
        A(f"| {esc(name)} | `{ident}` | {esc(desc)} |")
    A("")
    A("**部位別の材質**（「マテリアル」パネル。`materials.*`）")
    A("")
    A("| 項目 | 内部名 | 説明 |")
    A("|---|---|---|")
    for ident, name, desc in sub_props(props.TF_PartMaterials):
        A(f"| {esc(name)} | `materials.{ident}` | {esc(desc)} |")
    A("")
    A("**太さプロファイルの点**（太さが「指定」のとき。`profile[]`）")
    A("")
    A("| 項目 | 内部名 | 説明 |")
    A("|---|---|---|")
    for ident, name, desc in sub_props(props.TF_ProfilePoint):
        A(f"| {esc(name)} | `profile[].{ident}` | {esc(desc)} |")
    A("")
    # ---- 4. ボタン ----
    A("## 4. ボタン（操作）")
    A("")
    A("| ボタン | 内部名 | 説明 | 置かれているパネル |")
    A("|---|---|---|---|")
    for idname, name, desc in operator_list():
        pans = sorted({panels_by_id[p].bl_label for (p, kind, ident) in items if kind == "operator" and ident == idname})
        A(f"| {esc(name)} | `{idname}` | {esc(desc)} | {esc('、'.join(pans)) or '—'} |")
    A("")
    # ---- 5. 品質 ----
    A("## 5. 品質レベル")
    A("")
    A("「品質」（`quality`）で、メッシュの細かさと各種の上限が決まります。上限を超える場合は、細い枝（L3 → L2）から間引きます。")
    A("")
    A("| 項目 | プレビュー | 標準 | 高品質 |")
    A("|---|---|---|---|")
    qt = {q: allometry.quality_table(q) for q in schema.QUALITY}
    names = {"max_level": "枝の最大段数", "tri_cap": "幹・枝の三角形の上限", "leaf_cap": "葉インスタンスの上限",
             "l3_cap": "L3 の枝の本数の上限"}
    for key, label in names.items():
        A(f"| {label} | " + " | ".join(f"{qt[q][key]:,}" for q in schema.QUALITY) + " |")
    for lv in range(4):
        lvl = "幹（太いほど増える。これは最小値）" if lv == 0 else f"L{lv}"
        A(f"| 断面の頂点数（{lvl}） | " + " | ".join(str(qt[q]["ring"][lv]) for q in schema.QUALITY) + " |")
    for lv in range(4):
        lvl = "幹" if lv == 0 else f"L{lv}"
        A(f"| 節の刻み（{lvl}） [m] | " + " | ".join(f"{qt[q]['seg_len'][lv]:g}" for q in schema.QUALITY) + " |")
    A("| 幹の断面の弧長の目標 [m] | " + " | ".join("—" if allometry._RING_SPACING[q] is None else f"{allometry._RING_SPACING[q]:g}" for q in schema.QUALITY) + " |")
    A("| 幹の断面の頂点数の上限 | " + " | ".join(str(allometry._RING_CAP[q]) for q in schema.QUALITY) + " |")
    A("")
    A("高品質は三角形の予算を、細い枝の節の刻みではなく枝の本数に回しています。そのため L2・L3 の節の刻みは、標準より粗くなっています"
      "（枝 1 本あたりの滑らかさより、枝の本数を優先）。")
    A("")
    # ---- 6. プリセットの書式 ----
    A("## 6. プリセットの書式")
    A("")
    A("同梱プリセット（`tree_forge/presets/*.json`）は節ごとに分けた形式で、読み込み時に上の内部名へ変換されます。"
      "ユーザープリセット（「プリセットを保存」）は、内部名をそのまま `params` に持つ平坦な形式です。")
    A("")
    A("| JSON のキー | 内部名 | 項目 |")
    A("|---|---|---|")
    for sec, mapping in preset_io._SECTION_KEYS.items():
        for k, name in mapping.items():
            A(f"| `{sec}.{k}` | `{name}` | {esc(schema.BY_NAME[name].label)} |")
    A("| `growth.*` / `trunk.*` / `damage.*` / `material.*` | 内部名そのまま | 第 2 章 |")
    A("| `levels[i].*`（i = 1, 2, 3） | `l{i}_*` | 枝 L1〜L3 |")
    A("| `profile` | `profile` | 太さプロファイル（[高さ比, 直径 cm] の配列） |")
    A("| `_src` | — | 値の出所の注記（M: 図鑑・DB の記載値 / E: 推定値）。読み込みでは使わない |")
    A("")
    A("樹種プリセット: " + "、".join(f"{schema.SPECIES_NAMES[s]}（`{s}`）" for s in schema.SPECIES_IDS) + "。")
    A("")
    # ---- 7. 参道シーン ----
    A("## 7. 参道シーンのオプション")
    A("")
    A("`tools/build_sando_scene.py`（アドオン本体には入っていません）。`python tools/build_sando_scene.py --out sando.png` または "
      "`blender -b --factory-startup --python tools/build_sando_scene.py -- --out sando.png`。")
    A("")
    spec = importlib.util.spec_from_file_location("sando", ROOT / "tools" / "build_sando_scene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    A("| オプション | 既定 | 選択肢 | 説明 |")
    A("|---|---|---|---|")
    for act in mod.build_parser()._actions:
        if not act.option_strings or "-h" in act.option_strings:
            continue
        flag = ", ".join(act.option_strings)
        is_flag = act.nargs == 0
        default = "オフ（付けたときだけ有効）" if is_flag else ("" if act.default in ("", None) else str(act.default))
        choices = " / ".join(map(str, act.choices)) if act.choices else ""
        A(f"| `{flag}` | {esc(default)} | {esc(choices)} | {esc(act.help or '')} |")
    A("")
    A("御神木に足したもの（注連縄・紙垂・土盛り・落ち葉）は、`.blend` では `Hero_Extras` の下に項目ごとの子コレクションで入るので、アウトライナーでも切れます。")
    A("")
    return "\n".join(L)


def make_preset_values_md():
    cols = list(schema.SPECIES_IDS)
    raw = {s: preset_io.builtin_overrides(s) for s in cols}
    made = {s: schema.make_params(**raw[s]) for s in cols}
    L = []
    A = L.append
    A("# 樹種プリセットの値（全項目）")
    A("")
    A("`tools/dump_params.py` が `tree_forge/presets/*.json` から生成します（直接編集しないでください）。")
    A("**太字**は、その項目の既定値と違う値（その樹種で調整してある項目）です。項目の意味・範囲は [PARAMETERS.md](PARAMETERS.md) を見てください。")
    A("値の出所（図鑑・DB の記載値か推定値か）は、各 JSON の `_src` に書いてあります。**分枝角・頂芽優勢・重力屈性・垂れ・L2/L3 の値はすべて推定値**です。")
    A("")
    head = "| 項目 | 内部名 | " + " | ".join(schema.SPECIES_NAMES[s] for s in cols) + " |"
    sep = "|---|---|" + "---|" * len(cols)
    for key, title in GROUPS:
        ps = [p for p in schema.PARAMS if p.group == key and p.name != "species"]
        if not ps:
            continue
        A(f"## {title}")
        A("")
        A(head)
        A(sep)
        for p in ps:
            cells = []
            for s in cols:
                v = getattr(made[s], p.name)
                txt = esc(fmt_value(p, v))
                cells.append(f"**{txt}**" if v != p.default else txt)
            A(f"| {esc(p.label)} | `{p.name}` | " + " | ".join(cells) + " |")
        A("")
    A("## 太さプロファイル（指定している樹種）")
    A("")
    A("`trunk_profile_mode` が「指定」の樹種は、高さ比 t（0 根元〜1 樹頂）と直径 [cm] の点列で幹の太さを決めます。間は滑らかに補間します。")
    A("")
    any_prof = False
    for s in cols:
        prof = raw[s].get("profile")
        if prof and made[s].trunk_profile_mode == "CUSTOM":
            any_prof = True
            A(f"- **{schema.SPECIES_NAMES[s]}**: " + "、".join(f"t={t:g}: {d:g} cm" for t, d in prof))
    if not any_prof:
        A("（なし）")
    A("")
    return "\n".join(L)


def main():
    tree_forge.register()
    out_dir = ROOT / "docs"
    (out_dir / "PARAMETERS.md").write_text(make_parameters_md(), encoding="utf-8")
    (out_dir / "PRESET_VALUES.md").write_text(make_preset_values_md(), encoding="utf-8")
    print("wrote docs/PARAMETERS.md, docs/PRESET_VALUES.md", f"({len(schema.PARAMS)} params)")


if __name__ == "__main__":
    main()
