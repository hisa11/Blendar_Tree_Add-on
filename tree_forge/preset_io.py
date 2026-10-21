"""プリセット JSON の読み書き（bpy 非依存）。

同梱プリセットは節ごとに分けた形式（schema_version 1）:
    measured / crown / growth / trunk / damage / levels[] / leaves / bark / _src
ユーザープリセットは "params" に schema の名前をそのまま持つ平坦な形式。
どちらも overrides_from_dict() で make_params(**overrides) に渡せる辞書へ変換できる。
"""
from __future__ import annotations

import json
from pathlib import Path

from . import schema

SCHEMA_VERSION = 1
PRESET_DIR = Path(__file__).parent / "presets"

_SECTION_KEYS = {
    "measured": {"height_m": "height_m", "dbh_cm": "dbh_cm", "crown_width_m": "crown_width_m"},
    "crown": {"shape": "crown_shape", "base_ratio": "crown_base_ratio"},
    "leaves": {"type": "leaf_type", "length_cm": "leaf_length_cm", "width_ratio": "leaf_width_ratio",
               "per_m": "leaves_per_m", "fascicle": "fascicle_count", "geom": "leaf_geom",
               "scale_var": "leaf_scale_var", "flower_density": "flower_density", "evergreen": "evergreen",
               "color": "leaf_color", "autumn_color": "autumn_color", "flower_color": "flower_color"},
    "bark": {"type": "bark_type", "displace_mm": "bark_ridge_mm"},
}
_DIRECT_SECTIONS = ("growth", "trunk", "damage", "material")      # キーは schema の名前そのまま


def migrate(d: dict) -> dict:
    v = d.get("schema_version", 1)
    if v > SCHEMA_VERSION:
        raise ValueError(f"このプリセットは新しい形式です (schema_version={v})")
    return d


def overrides_from_dict(d: dict) -> dict:
    """プリセット辞書 → make_params に渡せる上書き辞書（検証は make_params 側）。"""
    d = migrate(d)
    out: dict = {}
    if "params" in d:                                   # 平坦形式
        out.update(d["params"])
        if "profile" in d:
            out["profile"] = d["profile"]
    for sec, mapping in _SECTION_KEYS.items():
        for k, v in d.get(sec, {}).items():
            if k not in mapping:
                raise KeyError(f"{sec}.{k}: 未知のキー")
            out[mapping[k]] = v
    for sec in _DIRECT_SECTIONS:
        out.update(d.get(sec, {}))
    for i, lv in enumerate(d.get("levels", []), start=1):
        for k, v in lv.items():
            out[f"l{i}_{k}"] = v
    if "profile" in d and "params" not in d:
        out["profile"] = d["profile"]
    if d.get("id") and "species" not in out:
        out["species"] = d["id"] if d["id"] in schema.SPECIES_IDS else "custom"
    return out


def load_builtin(species_id: str) -> dict:
    path = PRESET_DIR / f"{species_id}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def builtin_overrides(species_id: str) -> dict:
    d = load_builtin(species_id)
    ov = overrides_from_dict(d)
    ov["species"] = species_id
    return ov


def dump_user_preset(ns, name: str) -> dict:
    """現在のパラメータをユーザープリセットの辞書にする。"""
    params = schema.params_to_dict(ns)
    profile = params.pop("profile")
    params["species"] = "custom"
    return {"schema_version": SCHEMA_VERSION, "id": name, "name_ja": name, "params": params, "profile": profile}
