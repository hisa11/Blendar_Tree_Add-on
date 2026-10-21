"""パラメータ定義（bpy 非依存）。

props.py はこの表から PropertyGroup を生成し、core/ は make_params() が返す
SimpleNamespace を受け取る。単位・範囲・既定値はここ一箇所で持つ。
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

SPECIES_IDS = (
    "sugi", "sugi_giant", "hinoki", "kuromatsu", "keyaki", "someiyoshino", "ichou",
    "iroha_momiji", "shirakaba", "kusunoki", "quercus_robur", "picea_abies", "shidareyanagi",
)
SPECIES_NAMES = {
    "sugi": "スギ", "sugi_giant": "巨大スギ（御神木）", "hinoki": "ヒノキ", "kuromatsu": "クロマツ", "keyaki": "ケヤキ",
    "someiyoshino": "ソメイヨシノ", "ichou": "イチョウ", "iroha_momiji": "イロハモミジ",
    "shirakaba": "シラカバ", "kusunoki": "クスノキ", "quercus_robur": "ヨーロッパナラ",
    "picea_abies": "ヨーロッパトウヒ", "shidareyanagi": "シダレヤナギ", "custom": "カスタム",
}

CROWN_SHAPES = ("conical", "columnar", "ovoid", "spherical", "vase", "umbrella", "weeping", "irregular")
CROWN_SHAPE_NAMES = {
    "conical": "円錐", "columnar": "円柱", "ovoid": "卵形", "spherical": "球形",
    "vase": "ほうき形(vase)", "umbrella": "傘形", "weeping": "枝垂れ", "irregular": "不整形",
}
PHYLLOTAXIS = ("SPIRAL", "OPPOSITE", "WHORL")
PHYLLOTAXIS_NAMES = {"SPIRAL": "互生(らせん)", "OPPOSITE": "対生", "WHORL": "輪生"}
QUALITY = ("PREVIEW", "MEDIUM", "HERO")
LEAF_TYPES = ("BROAD", "NEEDLE_FASCICLE", "NEEDLE_SPIRAL", "SCALE_TWIG", "FAN")
LEAF_TYPE_NAMES = {
    "BROAD": "広葉", "NEEDLE_FASCICLE": "針葉束", "NEEDLE_SPIRAL": "針葉(らせん)",
    "SCALE_TWIG": "鱗片葉の小枝", "FAN": "扇形(イチョウ)",
}
LEAF_GEOM = ("CARD", "CUTOUT_MESH", "CUSTOM_OBJECT")
LEAF_GEOM_NAMES = {"CARD": "カード", "CUTOUT_MESH": "切り抜きメッシュ", "CUSTOM_OBJECT": "任意オブジェクト"}
PROFILE_MODES = ("AUTO", "CUSTOM")
RADIUS_MODES = ("LEONARDO", "RATIO")
BARK_TYPES = ("fissured", "scaly_plates", "strip_peel", "flaking", "smooth_lenticel")

LEVELS = (1, 2, 3)
SEED_MAX = 2**31 - 1


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                     # float / int / bool / enum / color
    default: object
    lo: float | None = None
    hi: float | None = None
    unit: str = ""
    label: str = ""
    desc: str = ""
    group: str = "general"
    simple: bool = False          # 簡易モードで表示
    items: tuple = ()             # enum: ((id, label), ...)
    update_regen: bool = True     # live_update で再生成を起こすか


def _enum(ids, names):
    return tuple((i, names.get(i, i)) for i in ids)


# 枝レベルごとの既定値（L1 は樹冠エンベロープに対する長さ倍率が length_ratio）
LEVEL_DEFAULTS = {
    1: dict(density=3.0, phyllotaxis="SPIRAL", divergence_deg=137.5, whorl_count=4, angle_deg=50.0,
            angle_var_deg=8.0, length_ratio=1.0, droop=0.05, wobble=0.25, taper=0.15,
            radius_mode="LEONARDO", radius_ratio=0.5),
    2: dict(density=2.5, phyllotaxis="SPIRAL", divergence_deg=137.5, whorl_count=4, angle_deg=55.0,
            angle_var_deg=10.0, length_ratio=0.5, droop=0.10, wobble=0.30, taper=0.12,
            radius_mode="LEONARDO", radius_ratio=0.5),
    3: dict(density=4.0, phyllotaxis="SPIRAL", divergence_deg=137.5, whorl_count=4, angle_deg=55.0,
            angle_var_deg=12.0, length_ratio=0.4, droop=0.15, wobble=0.35, taper=0.10,
            radius_mode="LEONARDO", radius_ratio=0.5),
}


def _level_params(n: int) -> list[Param]:
    d = LEVEL_DEFAULTS[n]
    g = f"level{n}"
    pre = f"l{n}_"
    t = f"枝L{n}"
    return [
        Param(pre + "density", "float", d["density"], 0.1, 20.0, "本/m", f"{t} 密度", "親の単位長あたりの本数", g),
        Param(pre + "phyllotaxis", "enum", d["phyllotaxis"], label=f"{t} 葉序", desc="枝の付き方", group=g,
              items=_enum(PHYLLOTAXIS, PHYLLOTAXIS_NAMES)),
        Param(pre + "divergence_deg", "float", d["divergence_deg"], 0.0, 180.0, "°", f"{t} 開度", "らせん互生の開度", g),
        Param(pre + "whorl_count", "int", d["whorl_count"], 2, 8, "", f"{t} 輪生数", "1節あたりの本数", g),
        Param(pre + "angle_deg", "float", d["angle_deg"], 5.0, 120.0, "°", f"{t} 分枝角", "親の接線からの角度", g),
        Param(pre + "angle_var_deg", "float", d["angle_var_deg"], 0.0, 30.0, "°", f"{t} 角度ばらつき", "", g),
        Param(pre + "length_ratio", "float", d["length_ratio"], 0.05, 1.2, "", f"{t} 長さ比",
              "L1: エンベロープに対する倍率 / L2,L3: 親の長さに対する比 γ", g),
        Param(pre + "droop", "float", d["droop"], 0.0, 1.0, "", f"{t} 垂れ", "先端ほど下に曲がる量", g),
        Param(pre + "wobble", "float", d["wobble"], 0.0, 1.0, "", f"{t} くねり", "ランダムな曲がり", g),
        Param(pre + "taper", "float", d["taper"], 0.02, 1.0, "", f"{t} 先細り", "枝先/付け根の半径比（枝分かれなしの場合）", g),
        Param(pre + "radius_mode", "enum", d["radius_mode"], label=f"{t} 太さの決め方",
              desc="LEONARDO: レオナルドの規則 / RATIO: 親の半径に対する固定比", group=g,
              items=(("LEONARDO", "レオナルド則"), ("RATIO", "固定比"))),
        Param(pre + "radius_ratio", "float", d["radius_ratio"], 0.05, 0.95, "", f"{t} 半径比",
              "RATIO モードのとき、親の半径に対する付け根半径の比", g),
    ]


def _build() -> list[Param]:
    P = Param
    ps: list[Param] = [
        # --- 全体 ---
        P("species", "enum", "keyaki", label="樹種", desc="プリセット（変更すると値を上書き）", group="general",
          simple=True, items=_enum(SPECIES_IDS + ("custom",), SPECIES_NAMES), update_regen=False),
        P("seed", "int", 1, 0, SEED_MAX, "", "シード", "乱数シード", "general", True),
        P("height_m", "float", 20.0, 1.0, 70.0, "m", "樹高", "", "general", True),
        P("dbh_cm", "float", 80.0, 2.0, 400.0, "cm", "胸高直径", "地上1.3 mの直径（AUTO プロファイル時）", "general", True),
        P("crown_width_m", "float", 18.0, 0.5, 40.0, "m", "樹冠幅", "", "general", True),
        P("crown_shape", "enum", "vase", label="樹冠形", desc="エンベロープ", group="general", simple=True,
          items=_enum(CROWN_SHAPES, CROWN_SHAPE_NAMES)),
        P("crown_base_ratio", "float", 0.25, 0.0, 0.9, "", "枝下高比", "枝下高 ÷ 樹高", "general"),
        P("quality", "enum", "MEDIUM", label="品質", desc="解像度", group="general", simple=True,
          items=(("PREVIEW", "プレビュー"), ("MEDIUM", "標準"), ("HERO", "高品質"))),
        P("age_factor", "float", 1.0, 0.2, 1.5, "", "樹齢感", "大きいほど曲がり・損傷・根張りが増える", "general", True),
        P("season", "float", 0.4, 0.0, 1.0, "", "季節",
          "0 冬 → 0.15 開花 → 0.3 新緑 → 0.85 紅葉 → 1 落葉", "general", True),
        P("leonardo_exponent", "float", 2.0, 1.8, 3.0, "", "レオナルド指数 Δ", "d^Δ = Σ d_i^Δ", "general"),
        P("apical_dominance", "float", 0.35, 0.0, 1.0, "", "頂芽優勢", "高いほど側枝がまっすぐ、低いほど屈曲する", "general"),
        P("gravitropism", "float", 0.25, -1.0, 1.0, "", "重力屈性", "正で枝先が上向き", "general"),
        P("phototropism", "float", 0.2, 0.0, 1.0, "", "光屈性", "樹冠の外側へ向かう傾き", "general"),
        P("min_radius_mm", "float", 1.0, 0.2, 10.0, "mm", "最小半径", "枝先の半径の下限", "general"),
        # --- 幹 ---
        P("trunk_profile_mode", "enum", "AUTO", label="幹の太さ", group="trunk", simple=True,
          desc="AUTO: 胸高直径とテーパーから自動 / CUSTOM: 高さごとの直径を指定",
          items=(("AUTO", "自動(DBH)"), ("CUSTOM", "高さごとに指定"))),
        P("trunk_taper_exp", "float", 0.8, 0.3, 2.0, "", "テーパー指数", "AUTO プロファイルの半径の減り方", "trunk"),
        P("trunk_lean_deg", "float", 2.0, 0.0, 30.0, "°", "傾き", "", "trunk"),
        P("trunk_wobble", "float", 0.15, 0.0, 1.0, "", "曲がり", "", "trunk"),
        P("trunk_split_height", "float", 0.35, 0.05, 0.95, "", "幹分かれ高", "幹分かれ高 ÷ 樹高", "trunk"),
        P("trunk_split_count", "int", 1, 1, 5, "", "幹分かれ本数", "2以上で指定高さで主幹が分かれる", "trunk"),
        P("root_flare_height", "float", 0.05, 0.0, 0.15, "", "根張り高", "根張り高 ÷ 樹高", "trunk"),
        P("root_flare_scale", "float", 1.4, 1.0, 2.5, "", "根張り倍率", "根元半径の倍率", "trunk"),
        P("root_count", "int", 6, 0, 12, "", "根張り数", "根張りの峰の数", "trunk"),
        P("burl_density", "float", 0.2, 0.0, 2.0, "個/m", "こぶ密度", "", "trunk"),
        P("bark_ridge_mm", "float", 8.0, 0.0, 50.0, "mm", "樹皮の凹凸", "ジオメトリとしての縦筋の深さ", "trunk"),
        P("trunk_irregularity", "float", 0.35, 0.0, 1.0, "", "幹断面のいびつさ",
          "断面を正円からずらす量。高さごとに向きと大きさが変わる", "trunk", True),
        P("branch_irregularity", "float", 0.25, 0.0, 1.0, "", "枝断面のいびつさ", "", "trunk"),
        # --- 損傷 ---
        P("dead_branch_ratio", "float", 0.05, 0.0, 0.5, "", "枯れ枝の割合", "葉がつかない枝", "damage", True),
        P("broken_branch_ratio", "float", 0.08, 0.0, 0.6, "", "折れ枝の割合",
          "途中で折れた枝（折れ口がささくれる）。樹齢感で増減", "damage", True),
        P("broken_hang_ratio", "float", 0.3, 0.0, 1.0, "", "垂れ下がり折れ", "折れ枝のうち、先が垂れ下がって残る割合", "damage"),
        P("broken_pos_min", "float", 0.15, 0.05, 0.95, "", "折れ位置(最小)", "枝長に対する折れ位置の下限", "damage"),
        P("broken_pos_max", "float", 0.8, 0.05, 0.98, "", "折れ位置(最大)", "枝長に対する折れ位置の上限", "damage"),
        P("top_break_height", "float", 0.0, 0.0, 0.98, "", "頂部折れ高", "0で無効。単幹のとき主幹をこの高さ比で折る", "damage"),
        # --- 葉 ---
        P("leaf_type", "enum", "BROAD", label="葉の種類", group="leaf", items=_enum(LEAF_TYPES, LEAF_TYPE_NAMES)),
        P("leaf_length_cm", "float", 6.0, 0.1, 30.0, "cm", "葉身長", "", "leaf", True),
        P("leaf_width_ratio", "float", 0.45, 0.05, 2.0, "", "幅/長さ", "", "leaf"),
        P("leaves_per_m", "float", 60.0, 1.0, 400.0, "枚/m", "葉の密度", "末端枝1 mあたり", "leaf"),
        P("fascicle_count", "int", 2, 1, 5, "", "針葉束の本数", "", "leaf"),
        P("leaf_cluster", "int", 1, 1, 12, "", "葉のまとまり",
          "1インスタンスに含める葉の枚数（葉つきの小枝）。多いほど密に見えるが三角形数が増える", "leaf"),
        P("leaf_scale", "float", 1.0, 0.2, 10.0, "", "葉の拡大率",
          "葉・針・花の見た目のサイズ倍率。針葉は実寸だと遠景で見えないため、針葉樹の同梱プリセットは拡大している", "leaf"),
        P("foliage_amount", "float", 1.0, 0.0, 5.0, "", "葉の量",
          "葉・針の数の倍率（1 = 既定、0 = 葉なし）。上限に達したら、数の代わりに1枚あたりを大きくして密度感を保つ",
          "leaf", True),
        P("leaf_reach", "enum", "AUTO", label="葉を付ける枝", group="leaf",
          desc="どの枝に葉を付けるか。AUTO は針葉・鱗片葉なら外側まで、広葉なら末端のみ",
          items=(("AUTO", "自動"), ("TERMINAL", "末端の枝のみ"), ("OUTER", "一段手前の外側まで"),
                 ("ALL", "さらに手前の枝の外側まで"))),
        P("leaf_start", "float", 0.2, 0.0, 0.9, "", "葉の付き始め",
          "末端の枝の付け根から、この割合より先に葉を付ける（0 で付け根から）", "leaf"),
        P("leaf_max_instances", "int", 0, 0, 5_000_000, "", "葉の最大数",
          "葉インスタンス数の上限。0 は品質ごとの上限（プレビュー 2万 / 標準 15万 / 高品質 150万）", "leaf"),
        P("leaf_geom", "enum", "CUTOUT_MESH", label="葉の表現", group="leaf", simple=True,
          items=_enum(LEAF_GEOM, LEAF_GEOM_NAMES)),
        P("leaf_scale_var", "float", 0.15, 0.0, 0.5, "", "葉サイズのばらつき", "", "leaf"),
        P("flower_density", "float", 0.0, 0.0, 1.0, "", "花の密度", "", "leaf"),
        P("evergreen", "bool", False, label="常緑", desc="季節で葉が落ちない", group="leaf"),
        P("leaf_color", "color", (0.10, 0.28, 0.05), label="葉の色", group="leaf"),
        P("autumn_color", "color", (0.75, 0.40, 0.05), label="紅葉の色", group="leaf"),
        P("flower_color", "color", (0.96, 0.72, 0.80), label="花の色", group="leaf"),
        P("bark_type", "enum", "fissured", label="樹皮タイプ", group="leaf", update_regen=False,
          items=tuple((b, b) for b in BARK_TYPES)),
        # --- 材質・UV ---
        P("bark_tile_w_m", "float", 1.0, 0.05, 5.0, "m", "樹皮タイル幅", "テクスチャ1枚の実寸（周方向）", "material"),
        P("bark_tile_h_m", "float", 1.0, 0.05, 5.0, "m", "樹皮タイル高", "テクスチャ1枚の実寸（長さ方向）", "material"),
        P("uv_integer_wrap", "bool", True, label="UVを整数回で巻く", group="material",
          desc="周方向のタイル数を整数に丸め、継ぎ目を消す"),
        P("junction_blend_len", "float", 1.5, 0.0, 4.0, "", "分岐ブレンド長", "子の半径に対する倍率", "material"),
    ]
    for n in LEVELS:
        ps += _level_params(n)
    return ps


PARAMS: tuple[Param, ...] = tuple(_build())
BY_NAME: dict[str, Param] = {p.name: p for p in PARAMS}

# Blender 側（props.py）が別途持つ非スカラー: trunk_profile（点のリスト）, leaf_object, live_update
DEFAULT_PROFILE_POINTS = 5


def make_params(**overrides) -> SimpleNamespace:
    """既定値の SimpleNamespace を作る。overrides は検証・クランプして上書き。"""
    ns = SimpleNamespace(**{p.name: p.default for p in PARAMS})
    ns.profile = []                 # [(height_ratio, diameter_cm), ...]
    ns.name_ja = ""
    ns.latin = ""
    for k, v in overrides.items():
        if k == "profile":
            ns.profile = [(float(a), float(b)) for a, b in v]
        elif k in BY_NAME:
            setattr(ns, k, coerce(BY_NAME[k], v))
        else:
            raise KeyError(f"unknown parameter: {k}")
    return ns


def coerce(p: Param, v):
    """型変換と範囲クランプ。enum は不正値なら例外。"""
    if p.kind == "float":
        v = float(v)
        if p.lo is not None:
            v = max(p.lo, v)
        if p.hi is not None:
            v = min(p.hi, v)
        return v
    if p.kind == "int":
        v = int(round(float(v)))
        return int(min(p.hi, max(p.lo, v)))
    if p.kind == "bool":
        return bool(v)
    if p.kind == "enum":
        ids = [i for i, _ in p.items]
        if v not in ids:
            raise ValueError(f"{p.name}: {v!r} not in {ids}")
        return v
    if p.kind == "color":
        c = tuple(float(x) for x in v)
        if len(c) != 3:
            raise ValueError(f"{p.name}: need 3 components")
        return c
    raise ValueError(p.kind)


def params_to_dict(ns) -> dict:
    d = {p.name: getattr(ns, p.name) for p in PARAMS}
    d["profile"] = [list(map(float, pt)) for pt in ns.profile]
    return d


def level_attr(ns, level: int, key: str):
    return getattr(ns, f"l{level}_{key}")
