"""拡張機能の zip を作る（blender --command extension build の代わりに使える、標準ライブラリだけの簡易版）。

    python tools/build_zip.py            # dist/tree_forge-<version>.zip
正式な検証・ビルドは Blender 本体で:
    blender --command extension validate tree_forge
    blender --command extension build --source-dir tree_forge --output-dir dist
"""
import os
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tree_forge"


def main() -> Path:
    m = tomllib.loads((SRC / "blender_manifest.toml").read_text(encoding="utf-8"))
    out = ROOT / "dist" / f"{m['id']}-{m['version']}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, dn, fn in os.walk(SRC):
            dn[:] = sorted(d for d in dn if d != "__pycache__")
            for f in sorted(fn):
                if not f.endswith(".pyc"):
                    full = Path(dp) / f
                    z.write(full, full.relative_to(SRC))
    print(out)
    return out


if __name__ == "__main__":
    main()
