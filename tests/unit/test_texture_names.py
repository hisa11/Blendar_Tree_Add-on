import pytest

from tree_forge.core import texture_names as T

CASES = [
    ("Bark_001_BaseColor.png", ("albedo", None)), ("bark_diff_4k.jpg", ("albedo", None)),
    ("bark_Albedo.png", ("albedo", None)), ("x_col.png", ("albedo", None)), ("BarkColor.png", ("albedo", None)),
    ("bark_diffuse.tif", ("albedo", None)), ("bark_base_color.png", ("albedo", None)),
    ("bark_Normal_GL.jpg", ("normal", "GL")), ("bark_normal_gl.png", ("normal", "GL")),
    ("bark_NormalGL.png", ("normal", "GL")), ("bark_nor_gl_2k.png", ("normal", "GL")),
    ("bark_nor_dx_2k.png", ("normal", "DX")), ("Bark_normalDX.png", ("normal", "DX")),
    ("bark_normal_directx.png", ("normal", "DX")), ("bark_normal_opengl.png", ("normal", "GL")),
    ("bark_normal.png", ("normal", None)), ("x_nrm.png", ("normal", None)), ("x_nor.png", ("normal", None)),
    ("x_nml.png", ("normal", None)),
    ("bark_Roughness.png", ("roughness", None)), ("bark_rough.png", ("roughness", None)),
    ("bark_ARM.png", ("arm", None)), ("bark_arm_2k.jpg", ("arm", None)),
    ("bark_AO.png", ("ao", None)), ("bark_ambientocclusion.png", ("ao", None)), ("bark_occlusion.png", ("ao", None)),
    ("bark_Displacement.exr", ("displacement", None)), ("bark_disp.png", ("displacement", None)),
    ("bark_Height.png", ("displacement", None)),
    ("leaf_Opacity.png", ("opacity", None)), ("leaf_alpha.png", ("opacity", None)), ("leaf_mask.png", ("opacity", None)),
    ("leaf_Translucency.png", ("translucency", None)), ("leaf_transmission.png", ("translucency", None)),
    ("x_Metalness.png", ("metallic", None)), ("x_metallic.png", ("metallic", None)),
    ("readme.txt", (None, None)), ("preview.png", (None, None)),
]


@pytest.mark.parametrize("name,expected", CASES)
def test_classify(name, expected):
    assert T.classify(name) == expected


def test_case_count():
    assert len(CASES) >= 30


def test_scan_picks_images_only_and_first_sorted():
    got = T.scan(["b_diff.png", "a_diff.png", "notes.txt", "x_nor_dx.png", "x_rough.jpg"])
    assert got["albedo"][0] == "a_diff.png"
    assert got["normal"] == ("x_nor_dx.png", "DX")
    assert "notes.txt" not in {v[0] for v in got.values()}
