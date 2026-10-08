"""[ours] Geometric completion of what the camera never saw (CPU, a few ms). ARKit world frame: y = up (gravity).

fill_floor(v, col, fy, must)  — floor holes
  Input : background-mesh vertices N×3 m and colours N×3 (0..1 or 0..255), floor height fy (m),
          `must` = list of (lo, hi) xz boxes that are certainly floor (object footprints)
  Output: (V K×3, T M×3, C K×3 in the input colour scale, stats) — 1 cm floor quads at fy for every empty cell that is
          enclosed by observed floor or lies under an object; colour = nearest observed floor cell.

complete_object(v, col, fy) — one object
  Input : the object's TSDF vertices N×3 m + colours, support height fy
  Output: {"shape": "box" | "cylinder" | "heightmap", "dims_mm", "rms_mm", "V", "T", "C"} — a closed solid down to the
          support. Box: minimum-area rectangle of the top face × height; cylinder: circle fit of the top-face outline;
          the one whose surface fits the observed vertices within FIT_TOL_M wins, otherwise a 2.5-D height map with a
          mirrored underside (round things close round). All assume no overhangs (a mug handle is filled in).
"""
import time

import cv2
import numpy as np
from scipy import ndimage

FLOOR_CELL_M, FLOOR_BAND_M = 0.01, 0.01
FIT_TOL_M = 0.0035
HM_CELL_M = 0.004
TOP_BAND_M = 0.005
HYBRID_GAP_M = 0.004
MIRROR = True                  # False = plain columns down to the support (the previous version, for comparison)
CYL_MARGIN = 0.8


def _cscale(col):
    return 255.0 if len(col) and col.max() > 1.5 else 1.0


def fill_floor(v, col, fy, must=()):
    t0 = time.time()
    v, col = np.asarray(v, float), np.asarray(col, float)
    fl = np.abs(v[:, 1] - fy) < FLOOR_BAND_M
    if fl.sum() < 50:
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3)), {"cells": 0, "area_m2": 0.0, "ms": 0}
    xz = v[fl][:, [0, 2]]
    lo = xz.min(axis=0) - 2 * FLOOR_CELL_M
    ij = np.floor((xz - lo) / FLOOR_CELL_M).astype(int)
    shape = tuple(ij.max(axis=0) + 3)
    occ = np.zeros(shape, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    csum = np.zeros(shape + (3,))
    cnt = np.zeros(shape)
    np.add.at(csum, (ij[:, 0], ij[:, 1]), col[fl])
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
    closed = ndimage.binary_closing(occ, np.ones((3, 3)), iterations=2)
    holes = ndimage.binary_fill_holes(closed) & ~occ                # enclosed by observed floor
    for blo, bhi in must:                                            # under objects: always floor
        a = np.clip(np.floor((np.asarray(blo)[[0, 2]] - lo) / FLOOR_CELL_M).astype(int), 0, np.array(shape) - 1)
        b = np.clip(np.ceil((np.asarray(bhi)[[0, 2]] - lo) / FLOOR_CELL_M).astype(int), 0, np.array(shape) - 1)
        holes[a[0]:b[0] + 1, a[1]:b[1] + 1] |= ~occ[a[0]:b[0] + 1, a[1]:b[1] + 1]
    hi_, hj = np.nonzero(holes)
    if not len(hi_):
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3)), {"cells": 0, "area_m2": 0.0, "ms": round((time.time() - t0) * 1000, 1)}
    _, (ni, nj) = ndimage.distance_transform_edt(~occ, return_indices=True)
    cc = csum[ni[hi_, hj], nj[hi_, hj]] / np.maximum(cnt[ni[hi_, hj], nj[hi_, hj]], 1)[:, None]
    x0, z0 = lo[0] + hi_ * FLOOR_CELL_M, lo[1] + hj * FLOOR_CELL_M
    y = fy + 0.0005
    s = FLOOR_CELL_M
    V = np.stack([np.c_[x0, np.full_like(x0, y), z0], np.c_[x0 + s, np.full_like(x0, y), z0],
                  np.c_[x0 + s, np.full_like(x0, y), z0 + s], np.c_[x0, np.full_like(x0, y), z0 + s]], 1).reshape(-1, 3)
    b = np.arange(len(x0)) * 4
    T = np.r_[np.c_[b, b + 2, b + 1], np.c_[b, b + 3, b + 2]]          # normals +y
    C = np.repeat(cc, 4, axis=0)
    return V, T, C, {"cells": int(len(x0)), "area_m2": round(len(x0) * s * s, 4), "ms": round((time.time() - t0) * 1000, 1)}


# ---------------------------------------------------------------- objects
def _box_mesh(c, u, w, d, y0, y1, colfn):
    """Closed box: centre c (x,z), unit axis u (x,z), widths w along u and d across, from y0 to y1. 4 verts per face."""
    n = np.array([-u[1], u[0]])
    corners = [c + sa * w / 2 * u + sb * d / 2 * n for sa, sb in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    P = lambda k, y: [corners[k][0], y, corners[k][1]]
    faces = [[P(0, y1), P(3, y1), P(2, y1), P(1, y1)], [P(0, y0), P(1, y0), P(2, y0), P(3, y0)]]
    for k in range(4):
        a, b = k, (k + 1) % 4
        faces.append([P(a, y0), P(a, y1), P(b, y1), P(b, y0)])
    V, T, C = [], [], []
    for f in faces:
        f = np.array(f, float)
        nrm = np.cross(f[1] - f[0], f[2] - f[0])
        if nrm @ (f.mean(0) - np.array([c[0], (y0 + y1) / 2, c[1]])) < 0:
            f = f[::-1]                                              # outward winding
        base = len(V)
        V += f.tolist()
        T += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
        C += [colfn(f.mean(0))] * 4
    return np.array(V), np.array(T), np.array(C)


def _cyl_mesh(c, r, y0, y1, colfn, seg=40):
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]
    V, T, C = [], [], []
    for k in range(seg):
        p, q = ring[k], ring[(k + 1) % seg]
        base = len(V)
        V += [[p[0], y0, p[1]], [q[0], y0, q[1]], [q[0], y1, q[1]], [p[0], y1, p[1]]]
        T += [[base, base + 2, base + 1], [base, base + 3, base + 2]]
        C += [colfn(np.array([(p[0] + q[0]) / 2, (y0 + y1) / 2, (p[1] + q[1]) / 2]))] * 4
    for y, up in ((y1, True), (y0, False)):                          # caps (fan)
        base = len(V)
        V.append([c[0], y, c[1]])
        V += [[p[0], y, p[1]] for p in ring]
        C += [colfn(np.array([c[0], y, c[1]]))] * (seg + 1)
        for k in range(seg):
            i, j = base + 1 + k, base + 1 + (k + 1) % seg
            T.append([base, j, i] if up else [base, i, j])
    return np.array(V), np.array(T), np.array(C)


def _box_rms(v, c, u, w, d, y1):
    n = np.array([-u[1], u[0]])
    rel = v[:, [0, 2]] - c
    lu, ln = rel @ u, rel @ n
    # distance to the nearest of the five visible faces (top + 4 sides); the bottom is never seen
    dist = np.min(np.c_[np.abs(np.abs(lu) - w / 2), np.abs(np.abs(ln) - d / 2), np.abs(v[:, 1] - y1)], axis=1)
    return float(np.sqrt(np.mean(dist ** 2)))


def _circle_fit(p):
    A = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:2]
    return c, float(np.sqrt(max(x[2] + c @ c, 1e-8)))


def _heightmap_mesh(v, col, fy, near_col):
    """Closed 2.5-D solid with mirror symmetry about the object's mid-height (y_m = (support + top) / 2).

    Per top-view cell (HM_CELL_M) the highest point seen gives the upper surface t; the unseen lower surface is its
    mirror b = 2·y_m − t (clamped to the support). A lying carrot, an apple or a banana then closes with a round
    underside (cross-sections mirror their tops) instead of vertical walls; a flat-topped solid still reaches the support.
    Cells whose top lies below y_m (a blade beside a handle, the lower flank of a bulge) are thin slabs from the
    support up to their top; walls join the two surfaces along the outline."""
    from scipy.spatial import cKDTree
    xz = v[:, [0, 2]]
    lo = xz.min(axis=0) - HM_CELL_M
    ij = np.floor((xz - lo) / HM_CELL_M).astype(int) + 1
    sh = tuple(ij.max(axis=0) + 2)
    top = np.full(sh, -np.inf)
    np.maximum.at(top, (ij[:, 0], ij[:, 1]), v[:, 1])
    seen = np.isfinite(top)
    foot = ndimage.binary_fill_holes(ndimage.binary_closing(seen, np.ones((3, 3))))
    _, (ni, nj) = ndimage.distance_transform_edt(~seen, return_indices=True)
    top = np.where(seen, top, top[ni, nj])
    top = np.where(foot, ndimage.median_filter(np.where(foot, top, fy), size=3), fy)
    ym = (fy + float(top[foot].max())) / 2
    # an open container (pot, mug) looks like a high rim around a low inside from above: its underside is not the
    # mirror of that, keep plain walls down to the support
    inner = ndimage.binary_erosion(foot, np.ones((3, 3)), iterations=max(1, int(0.25 * min(foot.sum(0).max(), foot.sum(1).max()))))
    rim = foot & ~ndimage.binary_erosion(foot, np.ones((3, 3)), iterations=2)
    hmax = float(top[foot].max()) - fy
    container = (inner.sum() >= 4 and np.median(top[inner]) - fy < 0.5 * hmax and np.median(top[rim]) - fy > 0.6 * hmax)
    # (an open container whose inside was not seen looks like a flat lump from above and cannot be told apart:
    #  a pot with an unseen inside closes with a lid at rim height — measured 15 mm mean error for the virtual pot)
    mirror = MIRROR and not container
    cell = foot          # cells below the mid-height (a knife blade beside its handle) stay as thin slabs on the support
    H, W = sh
    pad = np.pad(np.where(cell, top, -np.inf), 1, constant_values=-np.inf)
    corner = np.maximum.reduce([pad[:-1, :-1], pad[1:, :-1], pad[:-1, 1:], pad[1:, 1:]])   # (H+1)×(W+1)
    used = np.isfinite(corner)
    cid = -np.ones(corner.shape, int)
    cid[used] = np.arange(used.sum())
    ci, cj = np.nonzero(used)
    Vt = np.c_[lo[0] + (ci - 1) * HM_CELL_M, corner[used], lo[1] + (cj - 1) * HM_CELL_M]
    Vb = Vt.copy()
    if mirror:                                                                          # mirrored underside
        Vb[:, 1] = np.where(Vt[:, 1] > ym, np.clip(2 * ym - Vt[:, 1], fy, Vt[:, 1]), fy)
    else:
        Vb[:, 1] = fy
    nb = len(Vt)
    fi, fj = np.nonzero(cell)
    a, b_, c, d = cid[fi, fj], cid[fi + 1, fj], cid[fi + 1, fj + 1], cid[fi, fj + 1]
    tris = [np.c_[a, d, c], np.c_[a, c, b_], np.c_[a, c, d] + nb, np.c_[a, b_, c] + nb]       # top (+y) and bottom
    for di, dj, e0, e1 in ((-1, 0, a, d), (1, 0, c, b_), (0, -1, b_, a), (0, 1, d, c)):      # outline walls
        out = ~cell[np.clip(fi + di, 0, H - 1), np.clip(fj + dj, 0, W - 1)]
        p, q = e0[out], e1[out]
        tris += [np.c_[p, q + nb, q], np.c_[p, p + nb, q + nb]]
    V = np.vstack([Vt, Vb])
    T = np.vstack(tris)
    _, k = cKDTree(v).query(V)
    return V, T, col[k]


def hybrid(raw_v, raw_t, raw_c, comp_v, comp_t, comp_c, gap=HYBRID_GAP_M):
    """[ours] Observed surface kept as it is; the completion only fills where nothing was observed: completion triangles
    whose centre lies more than `gap` from every observed vertex are added to the observed mesh. Seen faces keep their
    ~1–2 mm TSDF accuracy, unseen ones come from the shape prior."""
    from scipy.spatial import cKDTree
    if not len(raw_t):
        return comp_v, comp_t, comp_c
    cen = comp_v[comp_t].mean(axis=1)
    d, _ = cKDTree(raw_v).query(cen)
    keep = comp_t[d > gap]
    used, inv = np.unique(keep, return_inverse=True)
    V = np.vstack([raw_v, comp_v[used]])
    T = np.vstack([raw_t, inv.reshape(-1, 3) + len(raw_v)]) if len(keep) else raw_t
    C = np.vstack([raw_c, comp_c[used]])
    return V, T, C


CONTAINERS = {"cooking pot", "saucepan", "pot", "coffee mug", "mug", "cup", "bowl", "frying pan"}
WALL_M = 0.004


def complete_container(v, col, fy, seg=48):
    """[ours] Category prior for open containers (pot, mug, cup, bowl): an open (possibly tapered) cylinder shell —
    outer wall, rim, inner wall, inner floor and outer bottom, wall WALL_M thick. Radius from the rim (top band), the
    bottom radius from the lower outside points when seen. Handles are not part of the prior: hybrid() keeps the
    observed surface where it sticks out of the template."""
    y1 = float(np.percentile(v[:, 1], 99))
    h = y1 - fy
    rim = v[v[:, 1] > y1 - 0.006]
    c, _ = _circle_fit(rim[:, [0, 2]])
    # radius per 10° sector, then the median over sectors: handles at rim height occupy a few sectors only
    # (a plain percentile over all rim points put a 201 mm pot at 246 mm — measured)
    def sector_radius(P, q):
        d = P[:, [0, 2]] - c
        r, ang = np.linalg.norm(d, axis=1), np.arctan2(d[:, 1], d[:, 0])
        b = np.floor((ang + np.pi) / (np.pi / 18)).astype(int)
        per = [np.percentile(r[b == i], q) for i in range(36) if (b == i).sum() >= 3]
        return float(np.median(per)) if per else float(np.percentile(r, q))
    r_top = sector_radius(rim, 80)
    allr = np.linalg.norm(v[:, [0, 2]] - c, axis=1)
    low = (v[:, 1] < fy + 0.3 * h) & (allr < r_top + 0.01)
    r_bot = sector_radius(v[low], 80) if low.sum() > 50 else r_top
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = lambda r, y: np.c_[c[0] + r * np.cos(a), np.full(seg, y), c[1] + r * np.sin(a)]
    rings = [ring(r_top, y1), ring(r_bot, fy), ring(r_top - WALL_M, y1), ring(max(r_bot - WALL_M, 0.002), fy + WALL_M)]
    V = np.vstack(rings + [[[c[0], fy + WALL_M, c[1]], [c[0], fy, c[1]]]])
    ci, cb = 4 * seg, 4 * seg + 1
    T = []
    idx = lambda r, k: r * seg + k % seg
    for k in range(seg):
        for r0, r1, flip in ((0, 1, False), (2, 0, False), (3, 2, False)):         # outer wall, rim, inner wall
            a0, a1, b0, b1 = idx(r0, k), idx(r0, k + 1), idx(r1, k), idx(r1, k + 1)
            T += [[a0, b0, a1], [a1, b0, b1]] if not flip else [[a0, a1, b0], [a1, b1, b0]]
        T += [[ci, idx(3, k), idx(3, k + 1)], [cb, idx(1, k + 1), idx(1, k)]]     # inner floor (up), outer bottom (down)
    T = np.array(T)
    # outward orientation check per face against the shell's mid-surface (flip faces pointing the wrong way)
    P = V[T]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    m = P.mean(axis=1)
    radial = np.c_[m[:, 0] - c[0], np.zeros(len(m)), m[:, 2] - c[1]]
    inner = np.isin(T, np.arange(2 * seg, 4 * seg + 1)).all(axis=1)
    want = np.where(inner[:, None], -radial, radial)
    flat = np.abs(n[:, 1]) > 0.9 * np.linalg.norm(n, axis=1)
    want[flat] = np.where(np.isin(T[flat], [cb]).any(axis=1)[:, None], [0, -1, 0], [0, 1, 0])
    bad = (n * want).sum(axis=1) < 0
    T[bad] = T[bad][:, ::-1]
    from scipy.spatial import cKDTree
    _, k_ = cKDTree(v).query(V)
    C = col[k_]
    # fit: radial residual of the observed points within the shell (handles excluded)
    yy = np.clip((v[:, 1] - fy) / max(h, 1e-6), 0, 1)
    r_at = r_bot + (r_top - r_bot) * yy
    near = np.abs(allr - r_at) < 0.012
    rms = float(np.sqrt(np.mean(np.minimum(np.abs(allr - r_at), np.abs(allr - (r_at - WALL_M)))[near] ** 2))) if near.any() else float("nan")
    # sanity: an open container's middle (inside 60 % of the radius) is lower than its rim, or not seen at all;
    # a domed top (an apple labelled "cup") rises above it → the prior does not apply
    mid = allr < 0.6 * r_top
    plausible = bool(mid.sum() < 30 or np.median(v[mid, 1]) < y1 - 0.25 * h)
    return {"shape": "container", "dims_mm": [round(2 * r_top * 1000), round(2 * r_bot * 1000), round(h * 1000)],
            "rms_mm": round(rms * 1000, 2), "fits_mm": {}, "V": V, "T": T, "C": C, "ms": 0, "plausible": plausible}


SYM_PREFER_M = 0.001
MIRROR_GROW = 1.3
ELONGATED = 1.8               # top-view aspect ratio above which an object is mirrored rather than revolved


def _surface_rms(v, V, T, n=4000):
    """RMS distance of the observed points to a candidate surface (how well the completion explains what was seen)."""
    import trimesh
    from scipy.spatial import cKDTree
    S = trimesh.Trimesh(V, T, process=False).sample(n)
    return float(np.sqrt(np.mean(cKDTree(S).query(v)[0] ** 2)))


def _revolve(v, col, fy, seg=48, dy=0.003):
    """Surface of revolution about a vertical axis: the axis from a circle fit to the side points (an arc is enough,
    so a half-seen apple still gets its true centre), the profile r(y) = outer radius per 3 mm height band."""
    from scipy.spatial import cKDTree
    y1 = float(v[:, 1].max())
    h = y1 - fy
    band = v[(v[:, 1] > fy + 0.2 * h) & (v[:, 1] < fy + 0.8 * h)]
    c, _ = _circle_fit((band if len(band) > 20 else v)[:, [0, 2]])
    r = np.linalg.norm(v[:, [0, 2]] - c, axis=1)
    ys = np.arange(fy, y1 + dy, dy)
    prof = np.full(len(ys), np.nan)
    b = np.clip(((v[:, 1] - fy) / dy).astype(int), 0, len(ys) - 1)
    d = v[:, [0, 2]] - c
    sec = np.floor((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (np.pi / 9)).astype(int)     # 20° sectors
    for i in range(len(ys)):
        sel = b == i
        if sel.sum() >= 3:
            # outer radius per sector, median over sectors: a pot's handles touch a few sectors only
            per = [np.percentile(r[sel & (sec == q)], 85) for q in range(18) if (sel & (sec == q)).sum() >= 2]
            prof[i] = np.median(per) if len(per) >= 3 else np.percentile(r[sel], 85)
    ok = np.isfinite(prof)
    if ok.sum() < 3:
        raise ValueError("profile")
    prof = np.interp(ys, ys[ok], prof[ok])
    prof = np.convolve(np.pad(prof, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    prof[-1] = max(prof[-1] * 0.3, 0.001)                                # close at the top
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    V = np.vstack([np.c_[c[0] + rr * np.cos(a), np.full(seg, y), c[1] + rr * np.sin(a)] for y, rr in zip(ys, prof)]
                  + [[[c[0], fy, c[1]], [c[0], y1, c[1]]]])
    nb, nt = len(ys) * seg, len(ys) * seg + 1
    T = []
    for i in range(len(ys) - 1):
        for k in range(seg):
            p0, p1, q0, q1 = i * seg + k, i * seg + (k + 1) % seg, (i + 1) * seg + k, (i + 1) * seg + (k + 1) % seg
            T += [[p0, q0, p1], [p1, q0, q1]]
    for k in range(seg):
        T += [[nb, k, (k + 1) % seg], [nt, (len(ys) - 1) * seg + (k + 1) % seg, (len(ys) - 1) * seg + k]]
    T = np.array(T)
    _, kk = cKDTree(v).query(V)
    return V, T, col[kk]


def _mirror_heightmap(v, col, fy, near_col):
    """Mirror the observed points about a vertical plane along the object's long axis, through its ridge (the highest
    point of each cross-section — the plane through the footprint's centre is biased toward the seen side), then the
    height map of the union: the unseen back half becomes the reflection of the front."""
    xz = v[:, [0, 2]]
    m = xz.mean(axis=0)
    w, U = np.linalg.eigh(np.cov((xz - m).T))
    u, n = U[:, -1], U[:, 0]
    s_, t_ = (xz - m) @ u, (xz - m) @ n
    bins = np.linspace(s_.min(), s_.max(), 12)
    ridge = []
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        sel = (s_ >= lo_) & (s_ < hi_)
        if sel.sum() >= 5:
            ridge.append(t_[sel][np.argmax(v[sel, 1])])
    off = float(np.median(ridge)) if ridge else 0.0
    t2 = 2 * off - t_
    xz2 = m + np.outer(s_, u) + np.outer(t2, n)
    # a curved object (banana) is not mirror-symmetric about a straight plane: the reflection adds a second crescent.
    # Footprint area before/after (4 mm cells) tells: a straight carrot barely grows, a banana grows a lot
    cell = lambda P: len(np.unique(np.floor(P / 0.004).astype(np.int64), axis=0))
    if cell(np.vstack([xz, xz2])) > MIRROR_GROW * cell(xz):
        raise ValueError("not mirror-symmetric")
    v2 = np.c_[xz2[:, 0], v[:, 1], xz2[:, 1]]
    return _heightmap_mesh(np.vstack([v, v2]), np.vstack([col, col]), fy, near_col)


CARVE_CELL_M, CARVE_PAD_M, CARVE_TAU_M = 0.004, 0.03, 0.006


def _free_space(g, kfs, tau):
    free = np.zeros(len(g), bool)
    for k in kfs:
        Tcw = np.linalg.inv(k["T"])
        pc = g @ Tcw[:3, :3].T + Tcw[:3, 3]
        z = pc[:, 2]
        K = k["K"]
        ok = z > 0.05
        u = np.round(K[0, 0] * pc[:, 0] / np.where(ok, z, 1) + K[0, 2]).astype(np.int64)
        w = np.round(K[1, 1] * pc[:, 1] / np.where(ok, z, 1) + K[1, 2]).astype(np.int64)
        dep = k["depth"]
        h, wd = dep.shape
        ok &= (u >= 0) & (u < wd) & (w >= 0) & (w < h)
        d = np.zeros(len(g))
        d[ok] = dep[w[ok], u[ok]]
        free |= ok & (d > 0.05) & (z < d - tau)
    return free


def carve_object(v, col, sup, kfs):
    """[ours, Space Carving (Kutulakos & Seitz 2000) with depth] The object is all the space no keyframe saw through:
    a 4 mm grid over the object's box (observed extent + CARVE_PAD_M, from the support up) is projected into every
    keyframe, and a cell lying more than CARVE_TAU_M in front of the measured depth is free. What survives is solid
    from the inside, hollow where views looked into it (a mug's inside), and the unseen back is filled only as far as
    the other views allow. Surface: marching cubes on the smoothed occupancy."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    lo = v.min(axis=0) - CARVE_PAD_M
    hi = v.max(axis=0) + CARVE_PAD_M
    lo[1] = sup
    hi[1] = v[:, 1].max() + CARVE_CELL_M
    shape = np.maximum(np.ceil((hi - lo) / CARVE_CELL_M).astype(int), 2)
    g = np.stack(np.meshgrid(*[lo[i] + (np.arange(shape[i]) + 0.5) * CARVE_CELL_M for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    free = np.zeros(len(g), bool)
    for k in kfs:
        Tcw = np.linalg.inv(k["T"])
        pc = g @ Tcw[:3, :3].T + Tcw[:3, 3]
        z = pc[:, 2]
        K = k["K"]
        ok = z > 0.05
        u = np.round(K[0, 0] * pc[:, 0] / np.where(ok, z, 1) + K[0, 2]).astype(np.int64)
        w = np.round(K[1, 1] * pc[:, 1] / np.where(ok, z, 1) + K[1, 2]).astype(np.int64)
        dep = k["depth"]
        h, wd = dep.shape
        ok &= (u >= 0) & (u < wd) & (w >= 0) & (w < h)
        d = np.zeros(len(g))
        d[ok] = dep[w[ok], u[ok]]
        free |= ok & (d > 0.05) & (z < d - CARVE_TAU_M)
    occ = (~free).reshape(shape).astype(float)
    occ[:, 0, :] = 1.0 * occ[:, 0, :]                      # the support plane closes the bottom
    occ = np.pad(occ, 1)                                   # closed surface at the box edges
    lab, n = ndi.label(occ > 0.5)
    if n > 1:                                              # keep the solid that holds the observed surface
        idx = np.floor((v - lo) / CARVE_CELL_M).astype(int) + 1
        idx = np.clip(idx, 0, np.array(occ.shape) - 1)
        hit = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
        keep = np.bincount(hit[hit > 0], minlength=n + 1).argmax() if (hit > 0).any() else 1
        occ = (lab == keep).astype(float)
    occ = ndi.gaussian_filter(occ, 0.8)
    V, F, _, _ = marching_cubes(occ, 0.5, spacing=(CARVE_CELL_M,) * 3)
    V = V + lo - CARVE_CELL_M
    V[:, 1] = np.maximum(V[:, 1], sup)
    _, kk = cKDTree(v).query(V)
    return V, F[:, ::-1].astype(np.int64), col[kk]


def fuse_volume(raw_v, raw_t, raw_c, comp_v, comp_t, comp_c, cell=0.002, band=0.003, kfs=None, tau=0.003):
    """[ours] One seamless surface instead of two stitched meshes (the stitch showed as a jagged seam across an apple):
    occupancy on a 3 mm grid = inside the completed solid (ray-parity occupancy) OR within `band` of the observed
    surface, lightly smoothed, then marching cubes. Observed detail survives through the band, the unseen part comes
    from the completion. (Open3D's Poisson aborted the process on some inputs — measured — so it is not used.)"""
    import open3d as o3d
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    lo = np.minimum(raw_v.min(0), comp_v.min(0)) - 3 * cell
    hi = np.maximum(raw_v.max(0), comp_v.max(0)) + 3 * cell
    shape = np.ceil((hi - lo) / cell).astype(int)
    g = np.stack(np.meshgrid(*[lo[i] + (np.arange(shape[i]) + 0.5) * cell for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(comp_v.astype(np.float32)), o3d.core.Tensor(comp_t.astype(np.uint32)))
    inside = sc.compute_occupancy(o3d.core.Tensor(g.astype(np.float32))).numpy() > 0.5
    near = cKDTree(raw_v).query(g, distance_upper_bound=band)[0] < np.inf
    occ = inside | near
    if kfs:                     # what the views saw through is empty: trims a completion that bulges past the observed
        occ &= ~_free_space(g, kfs, tau)                       # surface (the union alone came out 2–3 mm too fat)
    occ = occ.reshape(shape).astype(float)
    occ = ndi.gaussian_filter(np.pad(occ, 1), 0.5)
    V, F, _, _ = marching_cubes(occ, 0.5, spacing=(cell,) * 3)
    V = V + lo - cell
    _, k = cKDTree(np.vstack([raw_v, comp_v])).query(V)
    return V, F[:, ::-1].astype(np.int64), np.vstack([raw_c, comp_c])[k]


def complete_object(v, col, fy):
    t0 = time.time()
    v, col = np.asarray(v, float), np.asarray(col, float)
    sc = _cscale(col)
    y1 = float(np.percentile(v[:, 1], 98))
    xz = v[:, [0, 2]]
    # outline from the top face: seen square-on from above, while the TSDF smears side faces seen at a slant
    top = v[:, 1] > y1 - TOP_BAND_M
    xz_out = xz[top] if top.sum() >= 30 else xz
    near_col = lambda p: col[np.argmin(np.linalg.norm(v - p, axis=1))] if len(v) else np.full(3, 0.5 * sc)
    out = []
    # box: minimum-area rectangle of the top view
    (cx, cz), (w, d), ang = cv2.minAreaRect(xz_out.astype(np.float32))
    u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    out.append(("box", _box_rms(v, np.array([cx, cz]), u, w, d, y1), lambda: _box_mesh(np.array([cx, cz]), u, w, d, fy, y1, near_col),
                [round(w * 1000), round(d * 1000), round((y1 - fy) * 1000)]))
    # cylinder: circle through the outline of the top view
    hull = cv2.convexHull(xz_out.astype(np.float32)).reshape(-1, 2).astype(float)
    if len(hull) >= 5:
        c, r = _circle_fit(hull)
        rr = np.linalg.norm(xz - c, axis=1)
        dist = np.minimum(np.abs(rr - r), np.abs(v[:, 1] - y1) + np.maximum(rr - r, 0))
        out.append(("cylinder", float(np.sqrt(np.mean(dist ** 2))), lambda: _cyl_mesh(c, r, fy, y1, near_col),
                    [round(2 * r * 1000), round((y1 - fy) * 1000)]))
    # a cylinder must fit clearly better than the box (a small box's rounded TSDF corners fit a circle almost as well)
    pick = out[1] if len(out) > 1 and out[1][1] < CYL_MARGIN * out[0][1] else out[0]
    shape, rms, build, dims = pick
    if rms > FIT_TOL_M:
        # [ours] symmetry completion of what was never seen (the far side): surface of revolution for round
        # top views, mirror about the ridge plane for elongated ones; the one closest to the observed points wins,
        # the plain height map is the fallback
        cands = []
        pc2 = xz - xz.mean(axis=0)
        ev = np.linalg.eigvalsh(np.cov(pc2.T)) if len(pc2) > 3 else np.array([1.0, 1.0])
        aspect = float(np.sqrt(ev[-1] / max(ev[0], 1e-12)))
        for name, fn in ((("revolve", lambda: _revolve(v, col, fy)),) if aspect < ELONGATED else ()) + \
                        (("mirror", lambda: _mirror_heightmap(v, col, fy, near_col)),):
            try:
                cands.append((name,) + fn())
            except Exception:
                pass
        cands.append(("heightmap",) + _heightmap_mesh(v, col, fy, near_col))
        scored = [(_surface_rms(v, c[1], c[2]), c) for c in cands]
        best = min(r_ for r_, _ in scored)
        # the observed points cannot reward a completed far side: a symmetric candidate within SYM_PREFER_M of the
        # best fit is preferred over the plain height map (which explains the seen side just as well and cuts the rest)
        sym = [x for x in scored if x[1][0] != "heightmap" and x[0] <= best + SYM_PREFER_M]
        rms, (shape, V, T, C) = min(sym or scored, key=lambda x: x[0])
        dims = [round(np.ptp(V[:, 0]) * 1000), round(np.ptp(V[:, 2]) * 1000), round((V[:, 1].max() - fy) * 1000)]
        out += [(c[0], r_, None, None) for r_, c in scored]
    else:
        V, T, C = build()
    return {"shape": shape, "dims_mm": dims, "rms_mm": None if np.isnan(rms) else round(rms * 1000, 2),
            "fits_mm": {k: round(r * 1000, 2) for k, r, _, _ in out}, "V": V, "T": np.asarray(T).astype(np.int64), "C": C,
            "ms": round((time.time() - t0) * 1000, 1)}
