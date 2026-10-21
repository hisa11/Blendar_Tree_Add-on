"""同梱プリセット JSON (tree_forge/presets/*.json) を生成する。

値の出所は "_src" に M（図鑑・DB の記載値）/ E（推定値）で残す。
分枝角・頂芽優勢・重力屈性・垂れ・L2/L3 の値は全て推定値 (E) で、参照写真との比較調整が前提。
実行: python tools/make_presets.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tree_forge" / "presets"


def lv(angle, droop, density, ratio=None, phyl="SPIRAL", whorl=4, wobble=None, var=None, taper=None):
    d = {"phyllotaxis": phyl, "angle_deg": angle, "droop": droop, "density": density}
    if phyl == "WHORL":
        d["whorl_count"] = whorl
    if ratio is not None:
        d["length_ratio"] = ratio
    if wobble is not None:
        d["wobble"] = wobble
    if var is not None:
        d["angle_var_deg"] = var
    if taper is not None:
        d["taper"] = taper
    return d


def sp(id, name_ja, latin, H, D, W, shape, base, apical, grav, levels, leaf, bark, src,
       split=1, split_h=0.3, trunk=None, damage=None, growth=None, photo=0.2, profile=None):
    g = {"apical_dominance": apical, "gravitropism": grav, "phototropism": photo,
         "trunk_split_count": split, "trunk_split_height": split_h}
    g.update(growth or {})
    d = {
        "schema_version": 1, "id": id, "name_ja": name_ja, "latin": latin,
        "measured": {"height_m": H, "dbh_cm": D, "crown_width_m": W},
        "crown": {"shape": shape, "base_ratio": base},
        "growth": g,
        "levels": levels,
        "leaves": leaf,
        "bark": bark,
        "_src": src,
    }
    if trunk:
        d["trunk"] = trunk
    if damage:
        d["damage"] = damage
    if profile:
        d["profile"] = profile
    return d


GREEN_DARK = [0.05, 0.16, 0.06]
GREEN = [0.10, 0.28, 0.05]
GREEN_LIGHT = [0.16, 0.36, 0.08]

PRESETS = [
    sp("sugi", "スギ", "Cryptomeria japonica", 30, 80, 7.5, "conical", 0.12, 0.95, 0.6,
       [lv(85, 0.20, 6.0, 1.0, wobble=0.12), lv(70, 0.40, 5.0, 0.45), lv(60, 0.50, 7.0, 0.40)],
       {"type": "NEEDLE_SPIRAL", "length_cm": 1.2, "per_m": 70, "cluster": 5, "scale": 4.0, "scatter": 1.0, "amount": 2.0,
        "geom": "CUTOUT_MESH", "evergreen": True,
        "color": [0.09, 0.24, 0.09], "autumn_color": [0.30, 0.20, 0.06]},
       {"type": "strip_peel", "displace_mm": 14},
       {"height_m": "M:自生50m(conifers.org)/園芸15-18m(MBG)の中間", "dbh_cm": "M:最大300cm の範囲内", "crown_width_m": "M:MBG 6-9m",
        "growth": "E", "levels": "E", "leaves.length_cm": "M:ICRAF 3-25mm"}),
    sp("sugi_giant", "巨大スギ（御神木）", "Cryptomeria japonica", 55, 260, 20, "conical", 0.38, 0.85, 0.4,
       [lv(80, 0.45, 5.0, 1.0, wobble=0.30, var=14), lv(65, 0.55, 5.5, 0.45, wobble=0.35),
        lv(55, 0.60, 7.0, 0.40)],
       {"type": "NEEDLE_SPIRAL", "length_cm": 1.2, "per_m": 30, "cluster": 7, "scale": 5.5, "scatter": 0.3, "start": 0.05, "amount": 3.0,
        "reach": "ALL", "geom": "CUTOUT_MESH",
        "evergreen": True, "color": [0.08, 0.23, 0.09], "autumn_color": [0.30, 0.20, 0.06]},
       {"type": "strip_peel", "displace_mm": 60},
       {"height_m": "M:自生50(-65)m(conifers.org)の範囲内", "dbh_cm": "M:最大300cm(conifers.org)の範囲内",
        "crown_width_m": "E:老齢木、枝下が高い(自然落枝: ICRAF)", "crown.base_ratio": "E:自然落枝で幹が長く裸になる(ICRAF)",
        "growth": "E", "levels": "E", "profile": "E:ほぼ円筒で根元が張り出す老齢木の形"},
       trunk={"trunk_profile_mode": "CUSTOM", "trunk_wobble": 0.08, "trunk_lean_deg": 1.0, "root_flare_height": 0.05,
              "root_flare_scale": 1.4, "root_count": 6, "burl_density": 0.6, "trunk_irregularity": 0.5,
              "branch_irregularity": 0.3},
       damage={"age_factor": 1.5, "dead_branch_ratio": 0.12, "broken_branch_ratio": 0.10, "broken_hang_ratio": 0.35,
               "trunk_stubs": True, "trunk_stub_density": 0.5},
       profile=[[0.0, 300], [0.012, 272], [0.03, 258], [0.12, 244], [0.40, 214], [0.70, 150], [0.90, 70], [1.0, 8]]),
    sp("hinoki", "ヒノキ", "Chamaecyparis obtusa", 25, 60, 7, "conical", 0.10, 0.9, 0.5,
       [lv(70, 0.20, 5.0, 1.0, wobble=0.15), lv(60, 0.30, 3.0, 0.45), lv(55, 0.40, 4.0, 0.40)],
       {"type": "SCALE_TWIG", "length_cm": 8, "width_ratio": 0.35, "per_m": 40, "scale": 1.6, "geom": "CUTOUT_MESH", "evergreen": True,
        "color": [0.07, 0.20, 0.07], "autumn_color": [0.30, 0.20, 0.06]},
       {"type": "strip_peel", "displace_mm": 12},
       {"height_m": "M:最大35m(Wikipedia)", "dbh_cm": "E", "crown_width_m": "E", "growth": "E", "levels": "E",
        "leaves.length_cm": "E"}),
    sp("kuromatsu", "クロマツ", "Pinus thunbergii", 18, 60, 12, "umbrella", 0.35, 0.6, 0.3,
       [lv(65, 0.10, 2.5, 1.0, phyl="WHORL", whorl=4, wobble=0.35), lv(60, 0.20, 2.5, 0.45, wobble=0.4),
        lv(55, 0.20, 4.0, 0.40)],
       {"type": "NEEDLE_FASCICLE", "length_cm": 10, "fascicle": 2, "per_m": 12, "cluster": 10, "scale": 3.0, "geom": "CUTOUT_MESH", "evergreen": True,
        "color": [0.07, 0.20, 0.06], "autumn_color": [0.30, 0.20, 0.06]},
       {"type": "scaly_plates", "displace_mm": 18},
       {"height_m": "M:園芸6-18m(MBG)/最大40m", "dbh_cm": "E", "crown_width_m": "E", "growth": "E", "levels": "E",
        "leaves.length_cm": "M:7-12cm"},
       trunk={"trunk_wobble": 0.35, "trunk_lean_deg": 6.0}),
    sp("keyaki", "ケヤキ", "Zelkova serrata", 20, 80, 18, "vase", 0.28, 0.35, 0.25,
       [lv(38, 0.05, 2.2, 1.0, wobble=0.3), lv(45, 0.10, 2.5, 0.50), lv(50, 0.15, 4.0, 0.40)],
       {"type": "BROAD", "length_cm": 7, "width_ratio": 0.45, "per_m": 90, "cluster": 5, "geom": "CUTOUT_MESH",
        "color": GREEN, "autumn_color": [0.70, 0.38, 0.06]},
       {"type": "flaking", "displace_mm": 8},
       {"height_m": "M:MBG 15-24m", "dbh_cm": "E", "crown_width_m": "M:MBG 15-24m", "growth": "E", "levels": "E",
        "leaves.length_cm": "M:3.8-12cm"},
       split=3, split_h=0.30),
    sp("someiyoshino", "ソメイヨシノ", "Prunus x yedoensis", 10, 50, 11, "spherical", 0.20, 0.3, 0.1,
       [lv(50, 0.10, 2.0, 1.0, wobble=0.35), lv(55, 0.20, 2.5, 0.50), lv(55, 0.30, 4.0, 0.40)],
       {"type": "BROAD", "length_cm": 10, "width_ratio": 0.5, "per_m": 35, "cluster": 4, "scale": 1.6, "geom": "CUTOUT_MESH", "flower_density": 0.8,
        "color": GREEN_LIGHT, "autumn_color": [0.78, 0.30, 0.05], "flower_color": [0.97, 0.78, 0.85]},
       {"type": "smooth_lenticel", "displace_mm": 3},
       {"height_m": "M:MBG 9-12m", "dbh_cm": "E", "crown_width_m": "M:MBG 9-12m", "growth": "E", "levels": "E",
        "leaves.length_cm": "M:最大12.7cm"},
       split=3, split_h=0.25, trunk={"trunk_wobble": 0.25}),
    sp("ichou", "イチョウ", "Ginkgo biloba", 20, 70, 10.5, "ovoid", 0.20, 0.8, 0.4,
       [lv(50, 0.05, 3.0, 1.0, wobble=0.2), lv(55, 0.10, 2.5, 0.45), lv(55, 0.10, 4.0, 0.40)],
       {"type": "FAN", "length_cm": 3.5, "per_m": 50, "cluster": 4, "geom": "CUTOUT_MESH",
        "color": [0.14, 0.34, 0.07], "autumn_color": [0.95, 0.78, 0.08]},
       {"type": "fissured", "displace_mm": 9},
       {"height_m": "M:MBG 15-24m", "dbh_cm": "E", "crown_width_m": "M:MBG 9-12m", "growth": "E", "levels": "E",
        "leaves.length_cm": "M:幅5-8cm(Flora of China)→扇の半径3.5cm"}),
    sp("iroha_momiji", "イロハモミジ", "Acer palmatum", 6, 30, 6.5, "spherical", 0.15, 0.3, 0.1,
       [lv(55, 0.10, 3.0, 1.0, phyl="OPPOSITE", wobble=0.35), lv(55, 0.20, 3.0, 0.50, phyl="OPPOSITE"),
        lv(55, 0.30, 4.0, 0.40, phyl="OPPOSITE")],
       {"type": "BROAD", "length_cm": 7, "width_ratio": 1.0, "per_m": 55, "cluster": 4, "geom": "CUTOUT_MESH",
        "color": [0.12, 0.32, 0.07], "autumn_color": [0.78, 0.07, 0.03]},
       {"type": "smooth_lenticel", "displace_mm": 2},
       {"height_m": "M:MBG 3-7.6m", "dbh_cm": "E", "crown_width_m": "M:MBG 3-7.6m", "growth": "E", "levels": "E",
        "leaves.length_cm": "M:5-13cm"},
       split=3, split_h=0.20, trunk={"trunk_wobble": 0.3}),
    sp("shirakaba", "シラカバ", "Betula platyphylla", 12, 30, 6, "ovoid", 0.25, 0.7, 0.3,
       [lv(50, 0.10, 3.0, 1.0, wobble=0.25), lv(55, 0.30, 3.0, 0.50), lv(55, 0.50, 4.0, 0.40)],
       {"type": "BROAD", "length_cm": 6, "width_ratio": 0.8, "per_m": 55, "cluster": 5, "geom": "CUTOUT_MESH",
        "color": [0.14, 0.34, 0.07], "autumn_color": [0.92, 0.72, 0.08]},
       {"type": "smooth_lenticel", "displace_mm": 2},
       {"height_m": "M:MBG 9-12m", "dbh_cm": "E", "crown_width_m": "M:MBG 4.6-7.6m", "growth": "E", "levels": "E",
        "leaves.length_cm": "M:3-9cm"}),
    sp("kusunoki", "クスノキ", "Cinnamomum camphora", 20, 120, 25, "spherical", 0.15, 0.3, 0.15,
       [lv(50, 0.05, 1.8, 1.0, wobble=0.3), lv(55, 0.10, 2.5, 0.50), lv(55, 0.20, 4.0, 0.40)],
       {"type": "BROAD", "length_cm": 9, "width_ratio": 0.4, "per_m": 50, "cluster": 5, "geom": "CUTOUT_MESH", "evergreen": True,
        "color": [0.08, 0.24, 0.06], "autumn_color": [0.45, 0.30, 0.06]},
       {"type": "fissured", "displace_mm": 14},
       {"height_m": "M:最大30m(Flora of China)の範囲内", "dbh_cm": "M:最大300cmの範囲内", "crown_width_m": "E", "growth": "E",
        "levels": "E", "leaves.length_cm": "M:6-12cm"},
       split=3, split_h=0.20, trunk={"trunk_wobble": 0.25}),
    sp("quercus_robur", "ヨーロッパナラ", "Quercus robur", 25, 120, 25, "spherical", 0.25, 0.35, 0.1,
       [lv(60, 0.05, 1.8, 1.0, wobble=0.4), lv(60, 0.10, 2.5, 0.50, wobble=0.4), lv(60, 0.20, 4.0, 0.40)],
       {"type": "BROAD", "length_cm": 10, "width_ratio": 0.5, "per_m": 45, "cluster": 4, "geom": "CUTOUT_MESH",
        "color": [0.10, 0.27, 0.05], "autumn_color": [0.55, 0.35, 0.07]},
       {"type": "fissured", "displace_mm": 20},
       {"height_m": "M:Van den Berk 25-30m", "dbh_cm": "E", "crown_width_m": "M:25-30m", "growth": "E", "levels": "E",
        "leaves.length_cm": "M:5-14cm"},
       split=3, split_h=0.30, trunk={"trunk_wobble": 0.3}),
    sp("picea_abies", "ヨーロッパトウヒ", "Picea abies", 35, 100, 9, "conical", 0.05, 0.98, 0.0,
       [lv(90, 0.40, 6.0, 1.0, phyl="WHORL", whorl=5, wobble=0.08), lv(70, 0.60, 3.0, 0.45), lv(60, 0.70, 4.0, 0.40)],
       {"type": "NEEDLE_SPIRAL", "length_cm": 1.8, "per_m": 120, "scale": 4.0, "geom": "CUTOUT_MESH", "evergreen": True,
        "color": [0.07, 0.21, 0.09], "autumn_color": [0.25, 0.18, 0.05]},
       {"type": "flaking", "displace_mm": 10},
       {"height_m": "M:conifers.org 40-50m", "dbh_cm": "M:100-150cm", "crown_width_m": "M:Van den Berk 8-10m", "growth": "E",
        "levels": "E", "leaves.length_cm": "M:1-2.5cm"},
       trunk={"trunk_wobble": 0.06}),
    sp("shidareyanagi", "シダレヤナギ", "Salix babylonica", 12, 70, 13, "weeping", 0.22, 0.3, -0.2,
       [lv(40, 0.40, 2.0, 1.0, wobble=0.3), lv(45, 0.90, 3.0, 0.60), lv(45, 1.00, 4.0, 0.45)],
       {"type": "BROAD", "length_cm": 11, "width_ratio": 0.12, "per_m": 40, "cluster": 6, "geom": "CUTOUT_MESH",
        "color": [0.18, 0.38, 0.09], "autumn_color": [0.90, 0.78, 0.15]},
       {"type": "fissured", "displace_mm": 12},
       {"height_m": "M:UF/IFAS 9-15m", "dbh_cm": "E", "crown_width_m": "M:9-15m以上", "growth": "E", "levels": "E",
        "leaves.length_cm": "M:7-15cm"},
       split=3, split_h=0.30, trunk={"trunk_wobble": 0.3}),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for d in PRESETS:
        with open(OUT / f"{d['id']}.json", "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(f"{len(PRESETS)} presets -> {OUT}")


if __name__ == "__main__":
    main()
