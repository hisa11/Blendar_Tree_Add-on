"""骨格（枝の中心線）の生成。Weber–Penn 系の階層分枝。

幹（と幹分かれの主枝）= レベル0、枝 L1〜L3 = レベル1〜3。
レベルごとにまとめて numpy で成長させる（Python ループは「成長ステップ」と「親ごとの子の配置」だけ）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import allometry, envelope
from .rng import smooth_sample
from .util import Z, normalize, perp_basis

# 成長ステップ1回あたりの曲げ量の係数（軸全体での総量が下の値 × パラメータになる）
K_WOBBLE = 2.5
K_UP = 1.8
K_DROOP = 3.0
K_PHOTO = 1.0


@dataclass
class Axis:
    id: int
    level: int
    parent: int                 # 親の Axis.id（幹は -1）
    parent_node: int            # 親の何番目の節から出ているか
    pos: np.ndarray             # (n,3)
    out_dir: np.ndarray         # 外向き水平方向（光屈性・葉の向きに使う）
    dead: bool = False
    broken: bool = False        # 先端が折れ口
    neck: int = -1              # 垂れ下がり折れの折れ口の節番号
    removed: bool = False
    radius: np.ndarray | None = None
    q: np.ndarray | None = None
    phase: float = 0.0

    @property
    def n(self) -> int:
        return len(self.pos)

    def arc(self) -> np.ndarray:
        seg = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    def length(self) -> float:
        return float(self.arc()[-1])


class Skeleton:
    def __init__(self) -> None:
        self.axes: list[Axis] = []

    def add(self, level, parent, parent_node, pos, out_dir, dead=False, phase=0.0) -> Axis:
        ax = Axis(len(self.axes), level, parent, parent_node, pos, out_dir, dead=dead, phase=phase)
        self.axes.append(ax)
        return ax

    def children_map(self) -> dict[int, list[Axis]]:
        m: dict[int, list[Axis]] = {}
        for ax in self.axes:
            if ax.parent >= 0 and not ax.removed:
                m.setdefault(ax.parent, []).append(ax)
        return m

    def remove_subtree(self, axis_id: int, cm: dict | None = None) -> None:
        cm = cm if cm is not None else self.children_map()
        stack = [axis_id]
        while stack:
            a = stack.pop()
            self.axes[a].removed = True
            stack.extend(c.id for c in cm.get(a, []))

    def truncate(self, ax: Axis, new_n: int, cm: dict, tip: np.ndarray | None = None) -> None:
        """ax を new_n 節に切り詰め、それより先に出ていた子孫を消す。tip があれば末尾に節を足す。"""
        for c in cm.get(ax.id, []):
            if c.parent_node >= new_n:
                self.remove_subtree(c.id, cm)
        ax.pos = ax.pos[:new_n]
        if ax.radius is not None:
            ax.radius = ax.radius[:new_n]
        if ax.q is not None:
            ax.q = ax.q[:new_n]
        if tip is not None:
            ax.pos = np.vstack([ax.pos, tip])

    def compact(self) -> None:
        """removed を取り除き、id と親参照を詰め直す（親は必ず子より前にある）。"""
        keep = [a for a in self.axes if not a.removed]
        new_id = {a.id: i for i, a in enumerate(keep)}
        for a in keep:
            a.parent = new_id[a.parent] if a.parent >= 0 else -1
        for i, a in enumerate(keep):
            a.id = i
        self.axes = keep

    def live(self) -> list[Axis]:
        return [a for a in self.axes if not a.removed]

    def all_nodes(self) -> np.ndarray:
        return np.concatenate([a.pos for a in self.axes if not a.removed], axis=0)


# --------------------------------------------------------------------------------------
# 成長
# --------------------------------------------------------------------------------------
def grow(start, dir0, length, nseg, rng, *, up_pull, droop, wobble, kink, photo, out_dir, ground=0.05):
    """A 本の軸を同時に成長させる。各引数は (A,) または (A,3)。pos の list を返す。"""
    A = len(length)
    M = int(nseg.max())
    pos = np.empty((A, M + 1, 3))
    pos[:, 0] = start
    d = normalize(np.array(dir0, float))
    nf = nseg.astype(float)
    seg = length / nf
    tab1 = rng.uniform(-1.0, 1.0, (A, 6))
    tab2 = rng.uniform(-1.0, 1.0, (A, 6))
    psi = rng.uniform(0.0, 2 * math.pi, (A, M))
    period = np.maximum(2, nseg // 4)
    c_w = (wobble * K_WOBBLE / nf)[:, None]
    c_up = (up_pull * K_UP / nf)[:, None]
    c_dr = droop * K_DROOP / nf
    c_ph = (photo * K_PHOTO / nf)[:, None]
    photo_dir = out_dir + 0.5 * Z
    for i in range(M):
        act = i < nseg
        u = np.minimum((i + 0.5) / nf, 1.0)
        e1, e2 = perp_basis(d)
        steer = c_w * (smooth_sample(tab1, u)[:, None] * e1 + smooth_sample(tab2, u)[:, None] * e2)
        kk = ((i > 0) & (i % period == 0))[:, None]
        kv = kink[:, None] * kk * (np.cos(psi[:, i])[:, None] * e1 + np.sin(psi[:, i])[:, None] * e2)
        d2 = normalize(d + steer + c_up * Z - (c_dr * u)[:, None] * Z + c_ph * photo_dir + kv)
        d = np.where(act[:, None], d2, d)
        p_new = pos[:, i] + d * (seg * act)[:, None]
        p_new[:, 2] = np.where(act, np.maximum(p_new[:, 2], ground), p_new[:, 2])
        pos[:, i + 1] = p_new
    return [pos[a, : nseg[a] + 1].copy() for a in range(A)]


def _horizontal(d: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    h = d.copy()
    h[..., 2] = 0.0
    n = np.linalg.norm(h, axis=-1, keepdims=True)
    return np.where(n > 1e-6, h / np.maximum(n, 1e-12), fallback)


def _ray_exit(p, start: np.ndarray, d: np.ndarray, lmax: float) -> np.ndarray:
    """直線で伸ばしたとき、樹冠エンベロープの外へ出るまでの距離。"""
    ls = np.linspace(0.0, lmax, 40)
    pts = start[:, None, :] + d[:, None, :] * ls[None, :, None]
    rho = np.hypot(pts[..., 0], pts[..., 1])
    out = rho > envelope.reach(p, pts[..., 2]) + 1e-9
    out[:, 0] = False
    first = np.where(out.any(axis=1), out.argmax(axis=1), len(ls) - 1)
    return ls[first]


# --------------------------------------------------------------------------------------
# 幹と幹分かれ
# --------------------------------------------------------------------------------------
def _build_stems(p, rng, sk: Skeleton) -> None:
    H = p.height_m
    age = p.age_factor
    lean = math.radians(p.trunk_lean_deg)
    az = rng.uniform(0, 2 * math.pi)
    d0 = np.array([[math.sin(lean) * math.cos(az), math.sin(lean) * math.sin(az), math.cos(lean)]])
    split = p.trunk_split_count >= 2
    h_split = H * p.trunk_split_height
    length = (h_split if split else H) / max(math.cos(lean), 0.5)
    nseg = np.maximum(allometry.n_segments(np.array([length]), 0, p.quality), 10)
    wob = p.trunk_wobble * (0.6 + 0.4 * age)
    zeros = np.zeros(1)
    out0 = _horizontal(d0, np.array([[1.0, 0.0, 0.0]]))
    pos = grow(np.zeros((1, 3)), d0, np.array([length]), nseg, rng, up_pull=np.array([1.2]), droop=zeros,
               wobble=np.array([wob]), kink=zeros, photo=zeros, out_dir=out0)[0]
    trunk = sk.add(0, -1, 0, pos, out0[0], phase=rng.random())
    if not split:
        return

    n = p.trunk_split_count
    top = trunk.pos[-1]
    base_tilt = {"vase": 30.0, "spherical": 24.0, "ovoid": 20.0, "umbrella": 34.0, "weeping": 26.0,
                 "irregular": 26.0, "columnar": 12.0, "conical": 14.0}[p.crown_shape]
    az0 = rng.uniform(0, 2 * math.pi)
    az_i = az0 + (np.arange(n) + rng.uniform(-0.2, 0.2, n)) * (2 * math.pi / n)
    tilt = np.radians(base_tilt * rng.uniform(0.7, 1.2, n))
    t_end = normalize(trunk.pos[-1] - trunk.pos[-2])
    e1, e2 = perp_basis(t_end)
    dirs = (np.cos(tilt)[:, None] * t_end + np.sin(tilt)[:, None] *
            (np.cos(az_i)[:, None] * e1 + np.sin(az_i)[:, None] * e2))
    lens = (H - top[2]) / np.maximum(np.cos(tilt), 0.5) * rng.uniform(0.92, 1.06, n)
    lens = np.maximum(lens, 0.5)
    nseg_l = np.maximum(allometry.n_segments(lens, 0, p.quality), 10)
    outs = _horizontal(dirs, np.array([1.0, 0.0, 0.0]))
    pos_l = grow(np.repeat(top[None], n, 0), dirs, lens, nseg_l, rng, up_pull=np.full(n, 0.9), droop=np.zeros(n),
                 wobble=np.full(n, 0.25 * (0.7 + 0.3 * age)), kink=np.full(n, (1 - p.apical_dominance) * 0.15),
                 photo=np.zeros(n), out_dir=outs)
    for i in range(n):
        sk.add(0, trunk.id, trunk.n - 1, pos_l[i], outs[i], phase=rng.random())


# --------------------------------------------------------------------------------------
# 子の配置
# --------------------------------------------------------------------------------------
def _child_specs(p, sk: Skeleton, parents: list[Axis], level: int, rng) -> dict | None:
    pre = f"l{level}_"
    density = getattr(p, pre + "density")
    phyl = getattr(p, pre + "phyllotaxis")
    div = math.radians(getattr(p, pre + "divergence_deg"))
    wcount = getattr(p, pre + "whorl_count")
    ang = math.radians(getattr(p, pre + "angle_deg"))
    var = math.radians(getattr(p, pre + "angle_var_deg"))
    ratio = getattr(p, pre + "length_ratio")
    per = {"SPIRAL": 1, "OPPOSITE": 2, "WHORL": wcount}[phyl]
    hb = p.crown_base_ratio * p.height_m
    cols = {k: [] for k in ("parent", "node", "start", "dir", "plen", "u", "out", "dead")}
    for ax in parents:
        n = ax.n
        if n < 3:
            continue
        P = ax.pos
        s = ax.arc()
        L = s[-1]
        if L < 0.05:
            continue
        k0 = 1
        if level == 1:
            zi = np.nonzero(P[:, 2] >= hb)[0]
            if len(zi) == 0:
                continue
            k0 = max(1, int(zi[0]))
            s0, s1 = s[k0], 0.97 * L
            if s1 <= s0:
                continue
        else:
            s0, s1 = 0.12 * L, 0.97 * L
        dens = density * (0.4 if ax.dead else 1.0)
        nn = int(dens * (s1 - s0) / per + rng.random())
        if nn <= 0:
            continue
        sa = s0 + (s1 - s0) * (np.arange(nn) + 0.5 + rng.uniform(-0.35, 0.35, nn)) / nn
        k = np.clip(np.rint(np.interp(sa, s, np.arange(n))).astype(int), k0, n - 1)
        kk = np.repeat(k, per)
        node_i = np.repeat(np.arange(nn), per)
        j = np.tile(np.arange(per), nn)
        az0 = rng.uniform(0, 2 * math.pi)
        if phyl == "SPIRAL":
            phi = az0 + node_i * div
        elif phyl == "OPPOSITE":
            phi = az0 + node_i * (math.pi / 2) + j * math.pi
        else:
            phi = az0 + node_i * (math.pi / per + rng.uniform(-0.2, 0.2)) + j * (2 * math.pi / per)
        phi = phi + rng.normal(0.0, 0.12, len(phi))
        t = normalize(P[np.minimum(kk + 1, n - 1)] - P[kk - 1])
        e1, e2 = perp_basis(t)
        a = np.clip(ang + rng.normal(0.0, var, len(kk)), math.radians(5), math.radians(140))
        d = np.cos(a)[:, None] * t + np.sin(a)[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
        u = s[kk] / L
        cols["parent"].append(np.full(len(kk), ax.id))
        cols["node"].append(kk)
        cols["start"].append(P[kk])
        cols["dir"].append(d)
        cols["plen"].append(np.full(len(kk), L))
        cols["u"].append(u)
        cols["out"].append(np.broadcast_to(ax.out_dir, (len(kk), 3)))
        cols["dead"].append(np.full(len(kk), ax.dead))
    if not cols["parent"]:
        return None
    sp = {k: np.concatenate(v) for k, v in cols.items()}
    m = len(sp["parent"])
    if level == 1:
        lmax = max(p.height_m, p.crown_width_m) * 0.9
        exit_len = _ray_exit(p, sp["start"], normalize(sp["dir"]), lmax)
        length = exit_len * ratio * rng.uniform(0.85, 1.25, m)
        length = np.maximum(length, max(0.25, 0.02 * p.height_m))
        sp["out"] = _horizontal(sp["dir"], sp["out"])
    else:
        f = np.maximum(0.15, 1.0 - 0.75 * sp["u"])
        length = sp["plen"] * ratio * f * rng.uniform(0.8, 1.15, m)
    sp["length"] = length
    ratio_dead = p.dead_branch_ratio * (0.5 + 0.5 * p.age_factor)
    if level <= 2:
        sp["dead"] = sp["dead"] | (rng.random(m) < ratio_dead)
    return sp


def _grow_level(p, sk: Skeleton, sp: dict, level: int, rng) -> list[Axis]:
    pre = f"l{level}_"
    m = len(sp["length"])
    nseg = allometry.n_segments(sp["length"], level, p.quality)
    age = p.age_factor
    wob = getattr(p, pre + "wobble") * (0.7 + 0.3 * age)
    kink = (1.0 - p.apical_dominance) * (0.35 if level == 1 else 0.25)
    up = p.gravitropism * (1.0 if level == 1 else 0.8)
    pos = grow(sp["start"], sp["dir"], sp["length"], nseg, rng, up_pull=np.full(m, up),
               droop=np.full(m, getattr(p, pre + "droop")), wobble=np.full(m, wob), kink=np.full(m, kink),
               photo=np.full(m, p.phototropism), out_dir=sp["out"])
    out = []
    phases = rng.random(m)
    for i in range(m):
        out.append(sk.add(level, int(sp["parent"][i]), int(sp["node"][i]), pos[i], sp["out"][i],
                          dead=bool(sp["dead"][i]), phase=float(phases[i])))
    return out


# --------------------------------------------------------------------------------------
# 寸法合わせ・エンベロープでの刈り込み
# --------------------------------------------------------------------------------------
def measure(sk: Skeleton) -> tuple[float, float]:
    """(樹高, 樹冠幅)。樹冠幅は xy 外接矩形の幅と奥行きの平均。"""
    pts = sk.all_nodes()
    return float(pts[:, 2].max()), float(0.5 * (np.ptp(pts[:, 0]) + np.ptp(pts[:, 1])))


def fit_dimensions(p, sk: Skeleton, lo=0.6, hi=1.6) -> None:
    h, w = measure(sk)
    sz = min(hi, max(lo, p.height_m / max(h, 1e-6)))
    sxy = min(hi, max(lo, p.crown_width_m / max(w, 1e-6)))
    scale = np.array([sxy, sxy, sz])
    for a in sk.axes:
        if not a.removed:
            a.pos = a.pos * scale


def clip_to_envelope(p, sk: Skeleton, tol: float = 1.0) -> None:
    """エンベロープ（×tol）の外へ出た最初の節で枝を切る。幹・幹分かれの主枝は対象外。"""
    cm = sk.children_map()
    for a in sk.axes:
        if a.removed or a.level == 0:
            continue
        P = a.pos
        rho = np.hypot(P[:, 0], P[:, 1])
        gap = rho - envelope.reach(p, P[:, 2]) * tol          # 正ならエンベロープの外
        out = gap > 1e-9
        out[0] = False
        if out.any():
            m = int(out.argmax())
            if m < 2:
                sk.remove_subtree(a.id, cm)
                continue
            # 外へ出る区間は境界との交点で切り、枝先が境界まで届くようにする
            f = gap[m - 1] / (gap[m - 1] - gap[m])
            f = float(np.clip(f, 0.0, 1.0)) if np.isfinite(f) else 0.0
            tip = (P[m - 1] + (P[m] - P[m - 1]) * f)[None, :] if f > 0.25 else None
            sk.truncate(a, m, cm, tip)
    sk.compact()


def enforce_budget(p, sk: Skeleton, rng) -> None:
    """三角形数の見積りが品質の上限を超えるときは、細い枝（L3 → L2）からランダムに間引く。"""
    q = allometry.quality_table(p.quality)
    ring = q["ring"]
    target = q["tri_cap"] * 0.97

    def cost(a: Axis) -> int:
        return ((a.n - 1) * 2 + 1) * ring[min(a.level, 3)]

    for level in (3, 2):
        for _ in range(4):
            axes = [a for a in sk.axes if not a.removed]
            total = sum(cost(a) for a in axes)
            if total <= target:
                break
            lv = [a for a in axes if a.level == level]
            if not lv:
                break
            drop = min(1.0, (total - target) / max(sum(cost(a) for a in lv), 1) * 1.05)
            cm = sk.children_map()
            for a in lv:
                if rng.random() < drop:
                    sk.remove_subtree(a.id, cm)
            sk.compact()


def build(p, st) -> Skeleton:
    """骨格を作る。半径とダメージはこの後の段階で付く。"""
    rng = st.skel
    qual = allometry.quality_table(p.quality)
    sk = Skeleton()
    _build_stems(p, rng, sk)
    parents = [a for a in sk.axes if a.level == 0]
    for level in (1, 2, 3):
        if level > qual["max_level"] or not parents:
            break
        sp = _child_specs(p, sk, parents, level, rng)
        if sp is None:
            break
        if level == 3:
            cap = qual["l3_cap"]
            m = len(sp["length"])
            if m > cap:
                keep = rng.random(m) < cap / m
                sp = {k: v[keep] for k, v in sp.items()}
        if len(sp["length"]) == 0:
            break
        parents = _grow_level(p, sk, sp, level, rng)
    fit_dimensions(p, sk)
    clip_to_envelope(p, sk, 1.0)
    # 刈り込みで縮んだ分を取り戻す。許容 4% の刈り込みと合わせて数回で収束する
    for _ in range(3):
        fit_dimensions(p, sk, 0.7, 1.4)
        clip_to_envelope(p, sk, 1.04)
    return sk
