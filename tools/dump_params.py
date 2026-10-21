"""全設定項目の一覧（docs/PARAMETERS.md）を schema.py から生成する。

    python tools/dump_params.py            # docs/PARAMETERS.md を書き出す
UI の表示名・内部名・単位・範囲・既定値・説明をグループごとに並べる。表は schema から作るので実装とずれない。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tree_forge import schema  # noqa: E402

GROUPS = [
    ("general", "全体"), ("trunk", "幹・太さ・断面"), ("damage", "損傷（折れ・枯れ）"),
    ("level1", "枝 L1（主枝）"), ("level2", "枝 L2"), ("level3", "枝 L3（小枝）"),
    ("leaf", "葉・花・針"), ("material", "樹皮の UV・材質"),
]


def fmt_default(p):
    v = p.default
    if p.kind == "enum":
        return dict(p.items).get(v, v)
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


def main(out: Path):
    lines = [
        "# 設定項目一覧", "",
        "`tools/dump_params.py` が `tree_forge/schema.py` から生成します（直接編集しないでください）。",
        "樹種プリセットを選ぶと、これらの値がまとめて入れ替わります（`tree_forge/presets/*.json`）。", "",
        "## どこに表示されるか", "",
        "- **N パネル「TreeForge」（主パネル）**: 樹種、シード、品質、樹高、樹冠幅、胸高直径（自動のとき）、樹冠形、季節、"
        "**葉の量**、樹齢感、葉の表現。",
        "- **サブパネル**: 「幹の太さ」（自動/指定、テーパー、太さの点列）、「いびつさ・損傷」（断面の歪み、折れ枝、垂れ下がり折れ、枯れ枝、枝下の枝痕）、"
        "「葉・花」。",
        "- **「詳細モード」をオンにしたときだけ**: 折れ位置・頂部折れ、葉の細目（針束の本数・大きさのばらつき・幅・"
        "**葉を付ける枝・葉の付き始め・葉の最大数**・樹皮タイプ）、「詳細: 全体・幹」、「詳細: 枝 L1 / L2 / L3」。",
        "- **「マテリアル」パネル**: 部位別の材質、樹皮タイルの実寸、UV の整数巻き、分岐ブレンド長。", "",
        "このほかに、太さプロファイルの点列（高さ比・直径 cm）、葉オブジェクト（任意オブジェクト用）、部位別マテリアル（幹・枝・小枝・葉・花・果実・根張り・折れ口）、",
        "「自動再生成」があります。", "",
    ]
    for key, title in GROUPS:
        ps = [p for p in schema.PARAMS if p.group == key]
        if not ps:
            continue
        lines += [f"## {title}", "", "| 項目 | 内部名 | 単位 | 範囲・選択肢 | 既定 | 説明 |", "|---|---|---|---|---|---|"]
        for p in ps:
            lines.append(f"| {p.label} | `{p.name}` | {p.unit} | {fmt_range(p)} | {fmt_default(p)} | "
                         f"{p.desc.replace('|', '/')} |")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out, f"({len(schema.PARAMS)} params)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "PARAMETERS.md")
