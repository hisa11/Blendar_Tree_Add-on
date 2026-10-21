"""幹の断面を高さ別に重ねて描く（いびつさ 0 / 0.35 / 0.7）。python tools/plot_sections.py docs/images/sections.png"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tree_forge import preset_io, schema  # noqa: E402
from tree_forge.core import pipeline  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "sections.png"
heights = [0.5, 3.0, 6.0, 10.0, 14.0, 18.0]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, irr in zip(axs, (0.0, 0.35, 0.7)):
    ov = dict(preset_io.builtin_overrides("sugi"), quality="HERO", trunk_irregularity=irr,
              root_flare_scale=1.0, burl_density=0.0, bark_ridge_mm=0.0, seed=3)
    p = schema.make_params(**ov)
    t = pipeline.generate(p)
    pa = t.mesh.point_attrs
    sel = np.nonzero(pa["branch_id"] == 0)[0]
    n = t.skeleton.axes[0].n
    V = (len(sel) - 2) // n
    rings = t.mesh.co[sel[: n * V]].reshape(n, V, 3).astype(float)
    z = rings.mean(axis=1)[:, 2]
    for h, col in zip(heights, plt.cm.viridis(np.linspace(0.05, 0.9, len(heights)))):
        i = int(np.argmin(np.abs(z - h)))
        c = rings[i].mean(axis=0)
        xy = (rings[i] - c)[:, :2] * 100
        xy = np.vstack([xy, xy[:1]])
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.6, label=f"z={z[i]:.1f} m")
    ax.set_aspect("equal")
    ax.set_title(f"trunk_irregularity = {irr}")
    ax.set_xlabel("cm")
    ax.grid(alpha=0.25)
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
axs[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Trunk cross-sections at different heights (sugi preset, same axes)", fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("saved", out)
