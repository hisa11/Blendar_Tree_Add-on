"""拡張機能の zip を作る（blender --command extension build の代わりに使える、標準ライブラリだけの簡易版）。

    python tools/build_zip.py            # 開発版: dist/tree_forge-0.1.0+dev.<コミット数>.<コミット>.zip
    python tools/build_zip.py --release  # リリース版: dist/tree_forge-0.1.0+stable.zip（未コミットの変更があると作らない）

どのビルドか分かるよう、zip の中の blender_manifest.toml のバージョンに、semver の「ビルドメタデータ」を付ける。
  開発版（デバッグ中）: +dev.<git のコミット数>.<短いコミット ID>（例: 0.1.0+dev.93.d4453be）。未コミットの変更があれば末尾に .dirty。
  リリース版（安定版）: +stable（Ubuntu の LTS のような印。例: 0.1.0+stable）。
リポジトリの blender_manifest.toml は 0.1.0 のまま（ビルドのたびに書き換えない）。
同じ内容を build_info.json にも入れ、アドオンのパネルの下端に表示する（Blender の拡張機能一覧の表示と照らし合わせられる）。

正式な検証・ビルドは Blender 本体で:
    blender --command extension validate tree_forge
    blender --command extension build --source-dir tree_forge --output-dir dist
"""
import argparse
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
RELEASE_TAG = "stable"      # リリース版の印。開発版は dev.<コミット数>.<コミット ID>


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def make_version(base: str, count: str, sha: str, dirty: bool, release: bool = False) -> str:
    """0.1.0 → 開発版 0.1.0+dev.93.d4453be / リリース版 0.1.0+stable（semver のビルドメタデータ。Blender の検証を通る形）。"""
    if release:
        return f"{base}+{RELEASE_TAG}"
    return f"{base}+dev.{count}.{sha}" + (".dirty" if dirty else "")


def build_info(base: str, release: bool = False) -> dict:
    count = _git("rev-list", "--count", "HEAD") or "0"
    sha = _git("rev-parse", "--short=7", "HEAD") or "nogit"
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    if release and dirty:
        raise SystemExit("未コミットの変更があるので、リリース版は作れません（コミットしてから --release）")
    return {
        "base_version": base,
        "version": make_version(base, count, sha, dirty, release),
        "channel": RELEASE_TAG if release else "dev",
        "release": release,
        "commit": sha,
        "commit_count": int(count),
        "dirty": dirty,
        "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def main(out_dir: Path | None = None, release: bool = False) -> Path:
    manifest_text = (SRC / "blender_manifest.toml").read_text(encoding="utf-8")
    m = tomllib.loads(manifest_text)
    info = build_info(m["version"], release)
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", action="store_true", help=f"リリース版（+{RELEASE_TAG}）を作る。未コミットの変更があると失敗する")
    main(release=ap.parse_args().release)
