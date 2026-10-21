"""拡張機能の zip を作る（blender --command extension build の代わりに使える、標準ライブラリだけの簡易版）。

    python tools/build_zip.py            # dist/tree_forge-<version>+<コミット数>.<コミット>.zip

どのビルドか分かるよう、zip の中の blender_manifest.toml のバージョンに、semver の「ビルドメタデータ」として
git のコミット数と短いコミット ID を付ける（例: 0.1.0+93.d4453be）。作業ツリーに未コミットの変更があれば末尾に .dirty が付く。
リポジトリの blender_manifest.toml は 0.1.0 のまま（ビルドのたびに書き換えない）。
同じ内容を build_info.json にも入れ、アドオンのパネルの下端に表示する（Blender の拡張機能一覧の表示と照らし合わせられる）。

正式な検証・ビルドは Blender 本体で:
    blender --command extension validate tree_forge
    blender --command extension build --source-dir tree_forge --output-dir dist
"""
import datetime
import json
import os
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tree_forge"
_VERSION_LINE = re.compile(r'^version\s*=\s*"[^"]*"', re.MULTILINE)


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def make_version(base: str, count: str, sha: str, dirty: bool) -> str:
    """0.1.0 + コミット数 + コミット ID → 0.1.0+93.d4453be（semver のビルドメタデータ。Blender の検証を通る形）。"""
    return f"{base}+{count}.{sha}" + (".dirty" if dirty else "")


def build_info(base: str) -> dict:
    count = _git("rev-list", "--count", "HEAD") or "0"
    sha = _git("rev-parse", "--short=7", "HEAD") or "nogit"
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {
        "base_version": base,
        "version": make_version(base, count, sha, dirty),
        "commit": sha,
        "commit_count": int(count),
        "dirty": dirty,
        "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def main(out_dir: Path | None = None) -> Path:
    manifest_text = (SRC / "blender_manifest.toml").read_text(encoding="utf-8")
    m = tomllib.loads(manifest_text)
    info = build_info(m["version"])
    patched, n = _VERSION_LINE.subn(f'version = "{info["version"]}"', manifest_text, count=1)
    assert n == 1, "blender_manifest.toml に version 行が見つからない"
    out = (out_dir or ROOT / "dist") / f"{m['id']}-{info['version']}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, dn, fn in os.walk(SRC):
            dn[:] = sorted(d for d in dn if d != "__pycache__")
            for f in sorted(fn):
                if f.endswith(".pyc") or f in ("blender_manifest.toml", "build_info.json"):
                    continue
                full = Path(dp) / f
                z.write(full, full.relative_to(SRC))
        z.writestr("blender_manifest.toml", patched)
        z.writestr("build_info.json", json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    print(out)
    return out


if __name__ == "__main__":
    main()
