import json
import tomllib
from pathlib import Path

import pytest

from tree_forge import preset_io, schema

ROOT = Path(__file__).resolve().parents[2]


def test_defaults_are_valid():
    ns = schema.make_params()
    for p in schema.PARAMS:
        v = getattr(ns, p.name)
        if p.kind in ("float", "int"):
            assert p.lo <= v <= p.hi, p.name
        if p.kind == "enum":
            assert v in [i for i, _ in p.items], p.name


def test_make_params_clamps_and_rejects():
    ns = schema.make_params(height_m=1000, dbh_cm=-5, seed=5)
    assert ns.height_m == schema.BY_NAME["height_m"].hi
    assert ns.dbh_cm == schema.BY_NAME["dbh_cm"].lo
    with pytest.raises(KeyError):
        schema.make_params(nonexistent=1)
    with pytest.raises(ValueError):
        schema.make_params(crown_shape="cube")


def test_level_params_expanded():
    for n in schema.LEVELS:
        for k in ("density", "angle_deg", "droop", "taper", "radius_mode"):
            assert f"l{n}_{k}" in schema.BY_NAME


@pytest.mark.parametrize("sid", schema.SPECIES_IDS)
def test_builtin_presets_load_and_have_provenance(sid):
    d = preset_io.load_builtin(sid)
    assert d["schema_version"] == 1 and d["id"] == sid
    assert d["_src"], "M/E の出所メモがある"
    ns = schema.make_params(**preset_io.builtin_overrides(sid))
    assert ns.species == sid
    assert 1.0 <= ns.height_m <= 70.0


def test_all_species_present():
    """仕様の 12 種 + 巨大スギ（御神木）。JSON と SPECIES_IDS が一致する。"""
    files = {p.stem for p in (ROOT / "tree_forge" / "presets").glob("*.json")}
    assert files == set(schema.SPECIES_IDS) and len(files) == 13
    assert "sugi_giant" in files


def test_preset_values_match_spec_table():
    """仕様 4.2 の表の主要値と一致する。"""
    spec = {"sugi": (30, 80, 7.5, "conical"), "keyaki": (20, 80, 18, "vase"),
            "picea_abies": (35, 100, 9, "conical"), "shidareyanagi": (12, 70, 13, "weeping"),
            "kusunoki": (20, 120, 25, "spherical"), "ichou": (20, 70, 10.5, "ovoid")}
    for sid, (h, d, w, shape) in spec.items():
        ns = schema.make_params(**preset_io.builtin_overrides(sid))
        assert (ns.height_m, ns.dbh_cm, ns.crown_width_m, ns.crown_shape) == (h, d, w, shape), sid


def test_user_preset_roundtrip_keeps_profile():
    ns = schema.make_params(height_m=12.5, trunk_profile_mode="CUSTOM", profile=[(0, 60), (0.5, 25), (1, 3)])
    d = json.loads(json.dumps(preset_io.dump_user_preset(ns, "mine")))
    back = schema.make_params(**preset_io.overrides_from_dict(d))
    assert back.height_m == 12.5 and back.trunk_profile_mode == "CUSTOM"
    assert [tuple(p) for p in back.profile] == [(0, 60), (0.5, 25), (1, 3)]


def test_preset_errors():
    with pytest.raises(ValueError):
        preset_io.overrides_from_dict({"schema_version": 99})
    with pytest.raises(KeyError):
        preset_io.overrides_from_dict({"crown": {"bogus": 1}})


def test_manifest():
    m = tomllib.loads((ROOT / "tree_forge" / "blender_manifest.toml").read_text(encoding="utf-8"))
    for k in ("schema_version", "id", "version", "name", "tagline", "maintainer", "type", "blender_version_min",
              "license"):
        assert k in m, k
    assert m["id"] == "tree_forge" and m["type"] == "add-on"
    assert m["blender_version_min"] == "5.1.0" and m["blender_version_max"] == "5.3.0"
    assert m["license"] == ["SPDX:GPL-3.0-or-later"]
    assert set(m["permissions"]) == {"files"}
    assert not m["permissions"]["files"].endswith(".")        # 末尾に句点を付けない
    assert not m["tagline"].endswith(".")


@pytest.mark.parametrize("sid", schema.SPECIES_IDS)
def test_builtin_presets_set_the_reference_height_and_bark_colours(sid):
    """同梱プリセットの樹高が、枝葉の密度の基準（ref_height_m）。幹・枝の色も樹種ごとに持つ。"""
    p = schema.make_params(**preset_io.builtin_overrides(sid))
    assert p.ref_height_m == p.height_m > 0
    d = preset_io.load_builtin(sid)
    assert len(d["bark"]["color"]) == 3 and len(d["bark"]["branch_color"]) == 3
    assert tuple(p.bark_color) == tuple(d["bark"]["color"])


def test_birch_trunk_is_white_and_dark_barked_species_are_dark():
    def lightness(sid):
        return sum(schema.make_params(**preset_io.builtin_overrides(sid)).bark_color) / 3.0
    assert lightness("shirakaba") > 0.6                                    # シラカバは白い幹
    assert lightness("kuromatsu") < 0.15                                   # クロマツは灰黒色
    assert all(lightness(s) < 0.45 for s in schema.SPECIES_IDS if s != "shirakaba")


def test_user_preset_keeps_the_reference_height():
    ov = preset_io.builtin_overrides("sugi")
    ov["height_m"] = 20.0                                                  # 樹高を変えても基準は樹種の既定のまま
    p = schema.make_params(**ov)
    assert p.ref_height_m == 30.0
    ns = type("NS", (), dict(vars(p)))()
    d = preset_io.dump_user_preset(ns, "mine")
    assert d["params"]["ref_height_m"] == 30.0
    assert schema.make_params(**preset_io.overrides_from_dict(d)).ref_height_m == 30.0
