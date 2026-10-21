"""TreeForge — 寸法指定でリアルな樹木を作る Blender 拡張機能。

bpy に依存するモジュールは register() の中で読み込む（core/ と schema は bpy なしでテストできる）。
"""

_modules: list = []


def register():
    from . import props
    from .blender import operators, ui

    props.register()
    operators.register()
    ui.register()
    _modules[:] = [props, operators, ui]


def unregister():
    for m in reversed(_modules):
        m.unregister()
    _modules.clear()
