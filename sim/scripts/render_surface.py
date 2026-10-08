"""MPM 입자 기록을 실제 물체처럼 다시 그린다. 칼(손잡이 포함)·손·나무 도마와 함께 그림자·광택 있는 PBR 렌더러
(Open3D)로 그린다.

--mode mesh(기본): 식재료의 처음 메쉬를 칼질 평면(조각 라벨이 뒤집힌 입자로 어느 조각을 나눌지 정함)으로 미리 나누고,
  조각마다 단면을 새로 삼각분할해, 매 프레임 그 조각 입자의 움직임(강체 맞춤 + 남는 변형 보간)으로 옮긴다. 단면이 평평하고
  모서리가 날카롭다. 껍질은 꼭짓점 색 + 작은 돌기, 단면은 고해상도 질감(과육·씨)이다.
--mode recon: 매 프레임 입자에서 겉면을 재구성(pysplashsurf, 조각 라벨별). 모서리가 둥글고 표면이 울퉁불퉁하다.

python scripts/render_surface.py <run> [--cams persp,side,top,face] [--ss 2] [--mode mesh|recon]
입력: <run>/frames.npz (01_cut_primitive.py --save_frames 25), 결과: <run>/render/<cam>.mp4, render/first_<cam>.png
--ss: 이 배율로 크게 그린 뒤 줄여서 계단 현상을 없앤다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import numpy as np
import open3d as o3d
import pysplashsurf
import trimesh
from open3d.visualization import rendering
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from cutsim.assets.knife import blade_mesh


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mix(c0, c1, w):
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    c0, c1 = (c0[None] if c0.ndim == 1 else c0), (c1[None] if c1.ndim == 1 else c1)
    return c0 * (1 - w[:, None]) + c1 * w[:, None]


def hash01(*ints):
    """정수 격자 좌표 → 0~1 의사 난수(무늬용)."""
    h = np.zeros_like(ints[0], dtype=np.uint64)
    for k, v in zip((73856093, 19349663, 83492791), ints):
        h ^= (v.astype(np.int64).astype(np.uint64) * np.uint64(k))
    h = (h ^ (h >> np.uint64(13))) * np.uint64(0x5bd1e995)
    return ((h ^ (h >> np.uint64(15))) & np.uint64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


def vnoise(p, scale):
    """부드러운 3D 값 잡음(-1~1). p (M,3) m, scale m."""
    g = p / scale
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    f = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * hash01(i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz)
    return out * 2 - 1


class Section:
    """처음 모양에서 x 축(긴 축)을 따라 단면 중심·반지름 표. 원기둥꼴 식재료의 단면 좌표(rho, theta)용."""

    def __init__(self, X, p, step=0.002):
        lo, hi = X[:, 0].min(), X[:, 0].max()
        self.xs = np.arange(lo, hi + step, step)
        tab = []
        for x in self.xs:
            m = np.abs(X[:, 0] - x) < max(step, p)
            P = X[m] if m.sum() >= 8 else X[np.argsort(np.abs(X[:, 0] - x))[:8]]
            a, b = P[:, 1:].min(0) - p / 2, P[:, 1:].max(0) + p / 2
            tab.append([*(a + b) / 2, *np.maximum((b - a) / 2, p)])
        self.tab = gaussian_filter1d(np.array(tab), 1.5, axis=0)

    def coords(self, R):
        c = np.stack([np.interp(R[:, 0], self.xs, self.tab[:, k]) for k in range(4)], 1)
        u, v = (R[:, 1] - c[:, 0]) / c[:, 2], (R[:, 2] - c[:, 1]) / c[:, 3]
        return np.hypot(u, v), np.arctan2(v, u)


def srgb(c):
    return np.clip(c, 0, 1)


def cucumber_colors(R, depth, skin, sec, base_skin, base_flesh):
    """R: 처음 위치(m), depth: 처음 겉면에서 깊이(m), skin: 껍질 입자 비율(0~1) → RGB(0~1)."""
    rho, th = sec.coords(R)
    n1 = vnoise(R, 0.004)
    n2 = vnoise(R + 0.37, 0.0012)
    flesh = mix([0.80, 0.86, 0.68], np.asarray(base_flesh) * [0.93, 0.98, 0.86], smoothstep(0.45, 0.9, rho))
    flesh = mix(flesh, [0.56, 0.74, 0.40], smoothstep(0.80, 0.97, rho) * 0.85)  # 껍질 바로 안쪽 연두 띠
    flesh *= (1 + 0.025 * n1)[:, None]
    # 씨 자리(가운데 세 갈래): 반투명해 조금 어둡고 노르스름
    core = smoothstep(0.58, 0.48, rho + 0.04 * n1)
    col = mix(flesh, [0.72, 0.78, 0.55], core * 0.9)
    for k in range(3):  # 세 갈래 태좌(하얀 선)
        a = np.pi / 2 + k * 2 * np.pi / 3
        d_ang = np.angle(np.exp(1j * (th - a)))
        line = smoothstep(0.10, 0.03, np.abs(d_ang) * np.maximum(rho, 0.05)) * smoothstep(0.52, 0.42, rho)
        col = mix(col, [0.93, 0.94, 0.84], line * 0.7)
        for side in (-1, 1):  # 갈래마다 씨 두 줄, x 방향 4mm 간격으로 엇갈리게
            ac = a + side * 0.42
            pitch = 0.004
            xi = np.floor(R[:, 0] / pitch + (0.5 if side > 0 else 0.0))
            xc = (xi + (0.0 if side < 0 else -0.5) + 0.5) * pitch + (hash01(xi.astype(np.int64), np.full_like(xi, k, np.int64).astype(np.int64)) - 0.5) * 0.0012
            rc = 0.36 + 0.05 * (hash01(xi.astype(np.int64) + 7, np.full(len(xi), k * 2 + side, np.int64)) - 0.5)
            dth = np.angle(np.exp(1j * (th - ac)))
            e = ((R[:, 0] - xc) / 0.0018) ** 2 + ((rho - rc) / 0.11) ** 2 + (dth * rho / 0.06) ** 2
            seed = smoothstep(1.0, 0.45, e)
            col = mix(col, [0.97, 0.96, 0.88], seed)
    # 껍질: 짙은 녹색 + 세로 줄무늬 + 작은 돌기 점
    stripe = smoothstep(0.6, 0.98, np.cos(9 * th + 1.3 * np.sin(2 * th) + 1.2 * n1))
    sk = mix(base_skin, [0.30, 0.44, 0.24], stripe * 0.45)
    sk *= (1 + 0.06 * n2)[:, None]
    cell_x, cell_t = np.floor(R[:, 0] / 0.005), np.floor((th + np.pi) / (2 * np.pi / 28))
    h = hash01(cell_x.astype(np.int64), cell_t.astype(np.int64), np.full(len(th), 5, np.int64))
    cx = (cell_x + 0.5) * 0.005
    ct = (cell_t + 0.5) * (2 * np.pi / 28) - np.pi
    dd = np.hypot((R[:, 0] - cx) / 0.0009, np.angle(np.exp(1j * (th - ct))) * 0.021 / 0.0009)
    sk = mix(sk, [0.40, 0.50, 0.30], smoothstep(1.0, 0.3, dd) * (h < 0.25) * 0.8)
    w_skin = np.maximum(smoothstep(0.30, 0.65, skin), smoothstep(0.0024, 0.0015, depth))
    return srgb(mix(col, sk, w_skin))


def generic_colors(R, depth, skin, base_skin, base_flesh):
    n1 = vnoise(R, 0.004)
    flesh = np.asarray(base_flesh, float)[None] * (1 + 0.03 * n1)[:, None]
    sk = np.asarray(base_skin, float)[None] * (1 + 0.08 * vnoise(R + 0.3, 0.0015))[:, None]
    w = np.maximum(smoothstep(0.30, 0.65, skin), smoothstep(0.0024, 0.0015, depth))
    return srgb(mix(flesh, sk, w))


def rest_depth(x_rest, meta, p):
    """처음 위치에서 식재료 겉면까지 깊이. 메쉬가 있으면 정확한 거리, 없으면 입자 밀도로 만든 부피의 거리 변환."""
    mp = meta.get("food_mesh")
    if mp and Path(mp).exists():
        m = trimesh.load(mp, force="mesh")
        m.apply_translation([0, 0, meta.get("z_lift", 0.0)])
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(np.asarray(m.vertices, np.float32)),
                         o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
        return np.maximum(-sc.compute_signed_distance(o3d.core.Tensor(x_rest.astype(np.float32))).numpy(), 0.0)
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    h = p / 4
    lo = x_rest.min(0) - 3 * p
    idx = np.round((x_rest - lo) / h).astype(int)
    g = np.zeros(idx.max(0) + 4 * int(p / h))
    np.add.at(g, tuple(idx.T), 1.0)
    g = gaussian_filter(g, p / h * 0.6)
    inside = g > 0.5 * np.percentile(g[tuple(idx.T)], 50)
    return distance_transform_edt(inside)[tuple(idx.T)] * h


def reconstruct(P, attrs, r):
    rec = pysplashsurf.reconstruction_pipeline(
        P.astype(np.float64), attributes_to_interpolate=attrs, particle_radius=r, rest_density=1000.0,
        smoothing_length=2.0, cube_size=0.45, iso_surface_threshold=0.6, subdomain_grid=False,
        mesh_smoothing_iters=45, mesh_smoothing_weights=True, compute_normals=True, normals_smoothing_iters=15,
        mesh_cleanup=True)
    return rec[0] if isinstance(rec, tuple) else rec


def mesh_arrays(mw):
    """pysplashsurf MeshWithData → (V, F, {이름: 값})"""
    V = np.asarray(mw.mesh.vertices, float)
    F = np.asarray(mw.mesh.triangles, np.int64)
    return V, F, {k: np.asarray(v, float) for k, v in mw.point_attributes.items()}


def to_o3d(V, F, C=None, N=None):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    if N is not None and len(N) == len(V):
        m.vertex_normals = o3d.utility.Vector3dVector(N)
    else:
        m.compute_vertex_normals()
    if C is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(C)
    return m


def wood_texture(w=1024, h=768, seed=3):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(float)
    warp = sum(a * np.sin(x / s + rng.uniform(0, 6.28)) for a, s in ((5, 420), (2, 130), (0.6, 33)))
    ring = np.sin((y + warp) / 9.0 + 1.5 * np.sin(y / 170.0)) * 0.5 + 0.5
    fine = gaussian_filter1d(rng.normal(size=(h, w)), 0.7, axis=0)
    fine = gaussian_filter1d(fine, 40, axis=1)
    t = np.clip(0.30 * ring ** 4 + 0.6 * fine * 2.5 + 0.35, 0, 1)
    c0, c1 = np.array([0.80, 0.64, 0.45]), np.array([0.70, 0.53, 0.35])
    img = c0[None, None] * (1 - t[..., None]) + c1[None, None] * t[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def mat(color=(1, 1, 1), rough=0.6, metal=0.0, shader="defaultLit", reflect=0.4):
    m = rendering.MaterialRecord()
    m.shader = shader
    m.base_color = [*color, 1.0]
    m.base_roughness = rough
    m.base_metallic = metal
    m.base_reflectance = reflect
    return m


def knife_meshes(kn):
    """칼날(시뮬레이션과 같은 모양) + 손잡이(보기용, 칼날 -y 끝 칼등 쪽). 칼 좌표계: 원점 = 날 끝 한가운데."""
    b = blade_mesh(kn["thickness"], kn["length"], kn["height"], kn["bevel_height"], kn["edge_width"])
    L, H = kn["length"], kn["height"]
    hl, hh, ht = 0.09, 0.021, 0.015
    handle = trimesh.creation.box((ht, hl, hh))
    handle = handle.subdivide().subdivide()
    handle.vertices[:, [0, 2]] *= 1.0  # 모서리를 둥글게: 단면을 타원에 가깝게
    v = handle.vertices
    r = np.hypot(v[:, 0] / (ht / 2), v[:, 2] / (hh / 2))
    v[:, 0] *= np.where(r > 0, np.minimum(1.0, 1.0 / np.maximum(r, 1e-9) ** 0.35), 1)
    v[:, 2] *= np.where(r > 0, np.minimum(1.0, 1.0 / np.maximum(r, 1e-9) ** 0.35), 1)
    handle.vertices = v
    handle.apply_translation([0, L / 2 + hl / 2 - 0.003, H - hh / 2 - 0.002])
    return b, handle


def segment(a, b, r):
    """a→b 를 잇는 캡슐(반지름 r)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = trimesh.creation.capsule(height=max(np.linalg.norm(b - a), 1e-4), radius=r, count=[14, 14])
    z = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    c.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], z))
    c.apply_translation((a + b) / 2)
    return c


def hand_meshes(hold, surf_z):
    """다른 손(왼손) 표시: 고정 상자 자리에서 손가락 끝 넷이 오이 윗면을 짚고(마디가 칼 쪽), 손등과 팔은 뒤쪽으로.
    surf_z(x, y): 그 자리 식재료 윗면 높이. 모양만 손처럼 보이게 한 것이고 물리에는 쓰지 않는다."""
    cx, cy, cz, sx, sy, sz = hold
    parts = []
    x_tip = cx + sx / 2 - 0.003
    knuckles = []
    for k, dy in enumerate((-0.0168, -0.0056, 0.0056, 0.0168)):
        r = 0.0064 if k in (1, 2) else 0.0058
        y = cy + dy
        tip = np.array([x_tip - r - 0.002 * abs(k - 1.5), y, surf_z(x_tip - r, y) + r * 0.8])
        mid = tip + [-0.009, 0.0, 0.010]
        knu = mid + [-0.017, 0.0, 0.003]
        parts += [segment(tip, mid, r), segment(mid, knu, r * 1.08)]
        knuckles.append(knu)
    kn = np.array(knuckles)
    back = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    back.apply_scale([0.030, 0.027, 0.0105])
    back.apply_transform(trimesh.transformations.rotation_matrix(-0.18, [0, 1, 0]))
    bc = kn.mean(0) + [-0.022, 0.0, 0.001]
    back.apply_translation(bc)
    parts.append(back)
    parts.append(segment(bc + [-0.012, 0.0, 0.0], bc + [-0.085, -0.028, 0.030], 0.021))  # 손목·팔
    th0 = bc + [-0.004, -0.024, -0.004]
    thumb_tip = np.array([x_tip - 0.016, cy - 0.027, surf_z(x_tip - 0.016, cy - 0.020) - 0.006])
    parts.append(segment(th0, thumb_tip, 0.0068))
    return trimesh.util.concatenate(parts)


# ------------------------------------------------------------------ 메쉬 방식
def kabsch(A, B):
    """B ≈ A @ R.T + t 인 회전 R, 이동 t."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - ca @ R.T


def split_pieces(mesh, marks, x_rest, X, LAB, steps):
    """처음 메쉬를 칼질마다 나눈다. 칼질 k 에서 라벨이 뒤집힌 입자와 안 뒤집힌 입자가 함께 있는 조각만, 그 조각이 칼질
    직전에 놓인 자세를 되돌린 칼질 평면으로 나눈다. returns [{M: manifold, idx: 입자 번호, planes: [(n, d, 단면 바깥 부호)]}]"""
    import manifold3d as mf

    M0 = mf.Manifold(mesh=mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                                  tri_verts=np.asarray(mesh.faces, np.uint32)))
    pieces = [dict(M=M0, idx=np.arange(len(x_rest)), planes=[])]
    for mk in marks:
        fb = int(np.searchsorted(steps, mk["step"], side="left")) - 1  # 칼질 직전 프레임
        fa = min(fb + 1, len(steps) - 1)
        flipped = LAB[fa] != LAB[fb]
        yaw = np.radians(mk["yaw_deg"])
        n_w = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        p_w = np.array([mk["x_mm"], mk["y_mm"], 0.0]) / 1000.0
        out = []
        for pc in pieces:
            fl = flipped[pc["idx"]]
            if fl.sum() < 20 or (~fl).sum() < 20:
                out.append(pc)
                continue
            R, t = kabsch(x_rest[pc["idx"]], X[fb][pc["idx"]])
            n_r = R.T @ n_w
            d = float(n_r @ (R.T @ (p_w - t)))
            A, B = pc["M"].split_by_plane(tuple(float(v) for v in n_r), d)  # A: n.x >= d = 뒤집힌 쪽
            out.append(dict(M=A, idx=pc["idx"][fl], planes=pc["planes"] + [(n_r, d, -1.0)]))
            out.append(dict(M=B, idx=pc["idx"][~fl], planes=pc["planes"] + [(n_r, d, 1.0)]))
        pieces = out
    return pieces


def boundary_loops(V, F, n, d, tol):
    """껍질 메쉬의 경계선 중 평면(n.x = d) 위에 있는 것을 순서 있는 고리들로."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(E, axis=0, return_counts=True)
    be = uniq[cnt == 1]
    on = np.abs(V @ n - d) < tol
    be = be[on[be].all(1)]
    nb = {}
    for a, b in be:
        nb.setdefault(int(a), []).append(int(b))
        nb.setdefault(int(b), []).append(int(a))
    loops, seen = [], set()
    for s0 in nb:
        if s0 in seen:
            continue
        loop, prev, cur = [s0], None, s0
        seen.add(s0)
        while True:
            nxt = [v for v in nb[cur] if v != prev and (v not in seen or v == s0)]
            if not nxt or nxt[0] == s0:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur)
            seen.add(cur)
        if len(loop) >= 3:
            loops.append(np.array(loop))
    return loops


def cap_mesh(V, loops, n, sgn, h):
    """평면 위 고리들 안을 경계점 + 간격 h 격자점으로 들로네 삼각분할한 단면. 경계점은 껍질 꼭짓점과 같은 자리.
    returns (꼭짓점, 면, 평면 2D 좌표, (원점, e1, e2), 경계 꼭짓점 번호)"""
    from matplotlib.path import Path as MPath
    from scipy.spatial import Delaunay, cKDTree

    e1 = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    bidx = np.concatenate(loops)
    o = V[bidx].mean(0)
    to2 = lambda P: np.stack([(P - o) @ e1, (P - o) @ e2], 1)
    B2 = to2(V[bidx])
    paths = [MPath(to2(V[l])) for l in loops]
    lo, hi = B2.min(0), B2.max(0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], h), np.arange(lo[1], hi[1], h * 0.866))
    gx = gx + (np.arange(gx.shape[0])[:, None] % 2) * h / 2  # 삼각 격자
    g = np.stack([gx.ravel(), gy.ravel()], 1)
    inside = np.zeros(len(g), bool)
    for pth in paths:
        inside ^= pth.contains_points(g)
    dist, _ = cKDTree(B2).query(g)
    P2 = np.concatenate([B2, g[inside & (dist > 0.55 * h)]])
    tri = Delaunay(P2).simplices
    keep = np.zeros(len(tri), bool)
    cen = P2[tri].mean(1)
    for pth in paths:
        keep ^= pth.contains_points(cen)
    tri = tri[keep]
    # 넓이가 거의 0인 삼각형(경계점이 일직선)은 뺀다. 렌더러가 질감 방향을 계산하다 NaN 을 내 화면 전체가 검게 됐다
    a2 = np.cross(P2[tri[:, 1]] - P2[tri[:, 0]], P2[tri[:, 2]] - P2[tri[:, 0]])
    tri = tri[np.abs(a2) > 1e-3 * h * h]
    P3 = o + P2[:, :1] * e1 + P2[:, 1:] * e2
    P3[: len(bidx)] = V[bidx]
    fn = np.cross(P3[tri[:, 1]] - P3[tri[:, 0]], P3[tri[:, 2]] - P3[tri[:, 0]])
    flip = fn @ (sgn * n) < 0
    tri[flip] = tri[flip][:, ::-1]
    return P3, tri, P2, (o, e1, e2), bidx


def cucumber_bumps(R, sec):
    """껍질 돌기 높이(m): 무늬의 옅은 점 자리에 작은 혹 + 잔물결."""
    rho, th = sec.coords(R)
    cell_x, cell_t = np.floor(R[:, 0] / 0.005), np.floor((th + np.pi) / (2 * np.pi / 28))
    h = hash01(cell_x.astype(np.int64), cell_t.astype(np.int64), np.full(len(th), 5, np.int64))
    cx = (cell_x + 0.5) * 0.005
    ct = (cell_t + 0.5) * (2 * np.pi / 28) - np.pi
    dd = np.hypot((R[:, 0] - cx) / 0.0009, np.angle(np.exp(1j * (th - ct))) * 0.021 / 0.0009)
    return 0.00030 * np.exp(-dd ** 2) * (h < 0.25) + 0.00004 * vnoise(R, 0.0016)


class MeshPiece:
    """조각 하나: 껍질(꼭짓점 색) + 단면들(질감). 매 프레임 입자 움직임으로 꼭짓점을 옮긴다."""

    def __init__(self, idx, parts, x_rest, sigma, k=8):
        from scipy.spatial import cKDTree

        self.idx, self.parts = idx, parts  # parts: [(이름, V0, F, 색 또는 None, uv 또는 None, 질감 또는 None)]
        self.Xr = x_rest[idx]
        V0 = np.concatenate([pt[1] for pt in parts])
        dist, self.nb = cKDTree(self.Xr).query(V0, k=k)
        self.w = np.exp(-(dist / sigma) ** 2)
        self.V0 = V0
        self.cuts = np.cumsum([0] + [len(pt[1]) for pt in parts])

    def deform(self, Xf, r_max=0.004):
        P = Xf[self.idx]
        R, t = kabsch(self.Xr, P)
        res = P - (self.Xr @ R.T + t)
        ok = np.linalg.norm(res, axis=1) < r_max  # 떨어져 나간 입자 몇 개가 메쉬를 잡아끌지 않게
        w = self.w * ok[self.nb]
        V = self.V0 @ R.T + t + (w[..., None] * res[self.nb]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9)
        return [V[a:b] for a, b in zip(self.cuts[:-1], self.cuts[1:])]


def build_mesh_pieces(meta, x_rest, X, LAB, steps, sec, base_skin, base_flesh, p, tex_px=1024):
    mesh = trimesh.load(meta["food_mesh"], force="mesh")
    mesh.apply_translation([0, 0, meta.get("z_lift", 0.0)])
    if mesh.edges_unique_length.mean() > 0.0006:
        mesh = mesh.subdivide()
    # 사진에서 만든 형상은 복셀을 메쉬로 바꾼 것이라 0.5mm 계단이 있다. 모양은 두고 계단만 지운다
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=40)
    sdf = o3d.t.geometry.RaycastingScene()
    sdf.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                      o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    depth_of = lambda P: np.maximum(-sdf.compute_signed_distance(o3d.core.Tensor(P.astype(np.float32))).numpy(), 0.0)
    color = (lambda P, dep, sk: cucumber_colors(P, dep, sk, sec, base_skin, base_flesh)) if sec is not None else (
        lambda P, dep, sk: generic_colors(P, dep, sk, base_skin, base_flesh))
    tol = 2e-6
    out = []
    for pi, pc in enumerate(split_pieces(mesh, meta.get("marks", []), x_rest, X, LAB, steps)):
        m = pc["M"].to_mesh()
        V = np.asarray(m.vert_properties, float)[:, :3]
        F = np.asarray(m.tri_verts, np.int64)
        on_cap = np.zeros(len(F), bool)
        for n, d, _ in pc["planes"]:
            on_cap |= (np.abs(V @ n - d) < tol)[F].all(1)
        Fs = F[~on_cap]
        used = np.unique(Fs)
        remap = -np.ones(len(V), np.int64)
        remap[used] = np.arange(len(used))
        Vs, Fs = V[used], remap[Fs]
        # 껍질: 처음 겉면 위 점이라 깊이 0. 단면 경계와 먼 곳만 돌기만큼 바깥으로 민다
        Cs = color(Vs, np.zeros(len(Vs)), np.ones(len(Vs)))
        if sec is not None:
            nrm = trimesh.Trimesh(Vs, Fs, process=False).vertex_normals
            near_cut = np.zeros(len(Vs), bool)
            for n, d, _ in pc["planes"]:
                near_cut |= np.abs(Vs @ n - d) < 0.0006
            Vs = Vs + nrm * (cucumber_bumps(Vs, sec) * ~near_cut)[:, None]
        parts = [(f"p{pi}_skin", Vs, Fs, Cs, None, None)]
        for j, (n, d, sgn) in enumerate(pc["planes"]):
            loops = boundary_loops(Vs, Fs, n, d, 1e-5)
            if not loops:
                continue
            P3, tri, P2, (o, e1, e2), _ = cap_mesh(Vs, loops, n, sgn, 0.0008)
            lo = P2.min(0)
            S = float((P2.max(0) - lo).max()) + 1e-4
            uv = (P2 - lo) / S
            # 질감: 화소 중심의 처음 위치에서 과육·씨·껍질 테두리 색
            g = (np.arange(tex_px) + 0.5) / tex_px
            U, Vv = np.meshgrid(g, g)
            Pt = o + (lo[0] + U.ravel()[:, None] * S) * e1 + (lo[1] + Vv.ravel()[:, None] * S) * e2
            img = color(Pt, depth_of(Pt), np.zeros(len(Pt))).reshape(tex_px, tex_px, 3)
            img = (np.clip(img, 0, 1) ** (1 / 1.0) * 255).astype(np.uint8)[::-1].copy()  # 질감 v 는 아래가 0
            parts.append((f"p{pi}_cap{j}", P3, tri, None, uv[tri].reshape(-1, 2), img))
        out.append(MeshPiece(pc["idx"], parts, x_rest, sigma=p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cams", default="persp,side,top,face")
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--first_only", action="store_true", help="첫·중간·마지막 프레임 그림만(조정용)")
    ap.add_argument("--times", default=None, help="이 시각들(s, 쉼표)의 그림만 stills.jpg 로")
    ap.add_argument("--mode", choices=["mesh", "recon"], default="mesh")
    ap.add_argument("--ibl", default="default", help="주변광 환경맵(Open3D 내장: default, hall, konzerthaus, ...)")
    args = ap.parse_args()
    run = Path(args.run)
    d = np.load(run / "frames.npz")
    meta = json.loads(str(d["meta"]))
    X, LAB, T, Q = d["x"], d["lab"], d["t"], d["q"]
    x_rest, matid = d["x_rest"].astype(float), d["mat"]
    p = meta["particle_size"]
    r = p / 2
    stride = max(1, int(round(meta["fps"] / args.fps)))
    idx = np.arange(0, len(X), stride)
    if args.max_frames:
        idx = idx[: args.max_frames]
    if args.times:
        args.first_only = True
        idx = np.array([int(np.argmin(np.abs(T - float(v)))) for v in args.times.split(",")])
    elif args.first_only:
        idx = np.unique([0, len(X) // 3, 2 * len(X) // 3, len(X) - 1])

    names = meta["names"]
    is_skin = np.isin(matid, [names.index("skin")] if "skin" in names else []).astype(float)
    depth = rest_depth(x_rest, meta, p)
    cols = meta.get("colors", {})
    base_skin = np.array(cols.get("skin", [0.18, 0.37, 0.24]))
    base_flesh = np.array(cols.get("flesh", [0.78, 0.84, 0.72]))
    cat = meta.get("category", "default")
    sec = Section(x_rest, p) if cat == "cucumber" else None
    tree_xy = None

    def surf_z(xq, yq):
        """처음 모양에서 (x, y) 근처 입자의 가장 높은 곳 + 입자 반지름."""
        m = (np.abs(x_rest[:, 0] - xq) < 2 * p) & (np.abs(x_rest[:, 1] - yq) < 2 * p)
        return float(x_rest[m, 2].max() + r) if m.any() else float(x_rest[:, 2].max() + r)

    # 장면
    cams = {c: meta["cams"][c] for c in args.cams.split(",")}
    res = next(iter(cams.values()))["res"]
    W, H = res[0] * args.ss, res[1] * args.ss
    rr = rendering.OffscreenRenderer(W, H)
    sc = rr.scene
    sc.set_background([0.80, 0.80, 0.78, 1.0])
    sc.scene.set_sun_light([0.35, 0.45, -0.82], [1.0, 0.97, 0.92], 70000)
    sc.scene.enable_sun_light(True)
    import os

    ibl = os.path.join(os.path.dirname(o3d.__file__), "resources", args.ibl)
    if args.ibl != "default" and os.path.exists(ibl + "_ibl.ktx"):
        sc.scene.set_indirect_light(ibl)
    sc.scene.enable_indirect_light(True)
    sc.scene.set_indirect_light_intensity(32000)
    vc = np.mean([np.asarray(c["lookat"])[:2] for c in cams.values()], 0)
    board = o3d.geometry.TriangleMesh.create_box(0.34, 0.25, 0.018, create_uv_map=True, map_texture_to_each_face=True)
    board.translate([vc[0] - 0.17, vc[1] - 0.125, -0.018])
    board.compute_vertex_normals()
    mb = mat(rough=0.7, reflect=0.3)
    mb.albedo_img = o3d.geometry.Image(wood_texture())
    sc.add_geometry("board", board, mb)
    counter = o3d.geometry.TriangleMesh.create_box(1.6, 1.2, 0.02)
    counter.translate([vc[0] - 0.8, vc[1] - 0.45, -0.038])
    counter.compute_vertex_normals()
    sc.add_geometry("counter", counter, mat((0.58, 0.58, 0.60), rough=0.5))
    blade, handle = knife_meshes(meta["knife"])
    sc.add_geometry("blade", to_o3d(blade.vertices, blade.faces), mat((0.80, 0.82, 0.85), rough=0.22, metal=1.0))
    sc.add_geometry("handle", to_o3d(handle.vertices, handle.faces), mat((0.10, 0.08, 0.07), rough=0.45))
    for i, h in enumerate(meta.get("holds") or []):
        hm = hand_meshes(h, surf_z)
        sc.add_geometry(f"hand{i}", to_o3d(hm.vertices, hm.faces), mat((0.87, 0.66, 0.55), rough=0.55, reflect=0.35))
    m_food = mat(rough=0.42, reflect=0.45)
    if args.mode == "mesh":
        t_b = time.time()
        pieces = build_mesh_pieces(meta, x_rest, X, LAB, d["step"], sec, base_skin, base_flesh, p)
        m_skin = mat(rough=0.40, reflect=0.45)
        m_skin.base_clearcoat, m_skin.base_clearcoat_roughness = 0.5, 0.25  # 껍질의 왁스 광택
        cap_mats = {}
        for pc in pieces:
            for nm, *_, img in pc.parts:
                if img is not None:
                    mc = mat(rough=0.32, reflect=0.5)
                    mc.base_clearcoat, mc.base_clearcoat_roughness = 0.6, 0.15  # 젖은 단면(너무 세면 하얗게 날아감)
                    mc.albedo_img = o3d.geometry.Image(img)
                    cap_mats[nm] = mc
        print(f"  메쉬 조각 {len(pieces)}개, 꼭짓점 {sum(len(pc.V0) for pc in pieces)}개 ({time.time() - t_b:.0f}s)",
              flush=True)

    out = run / "render"
    out.mkdir(exist_ok=True)
    writers = {}
    if not args.first_only:
        for c in cams:
            cont = av.open(str(out / f"{c}.mp4"), "w")
            st = cont.add_stream("libx264", rate=int(args.fps))
            st.width, st.height, st.pix_fmt = res[0], res[1], "yuv420p"
            st.options = {"crf": "20", "preset": "medium"}
            writers[c] = (cont, st)
    food_names = []
    t0 = time.time()
    stills = []
    for n, f in enumerate(idx):
        for nm in food_names:
            sc.remove_geometry(nm)
        food_names = []
        P, lab = X[f].astype(float), LAB[f]
        if args.mode == "mesh":
            for pc in pieces:
                for (nm, V0, Fp, Cp, uvp, img), Vf in zip(pc.parts, pc.deform(P)):
                    gm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vf),
                                                   o3d.utility.Vector3iVector(Fp.astype(np.int32)))
                    gm.compute_vertex_normals()
                    if Cp is not None:
                        gm.vertex_colors = o3d.utility.Vector3dVector(np.clip(Cp, 0, 1) ** 2.2)  # 꼭짓점 색은 선형
                        sc.add_geometry(nm, gm, m_skin)
                    else:
                        gm.triangle_uvs = o3d.utility.Vector2dVector(uvp)
                        sc.add_geometry(nm, gm, cap_mats[nm])
                    food_names.append(nm)
        for g in (np.unique(lab) if args.mode == "recon" else []):
            m = lab == g
            attrs = {k: np.ascontiguousarray(v, np.float64) for k, v in (
                ("rest", x_rest[m]), ("depth", depth[m]), ("skin", is_skin[m]), ("one", np.ones(int(m.sum()))))}
            mw = reconstruct(P[m], attrs, r)
            V, F, A = mesh_arrays(mw)
            if len(F) == 0:
                continue
            one = np.maximum(A["one"], 1e-6)
            R = A["rest"] / one[:, None]
            dep, sk = A["depth"] / one, A["skin"] / one
            C = (cucumber_colors(R, dep, sk, sec, base_skin, base_flesh) if sec is not None
                 else generic_colors(R, dep, sk, base_skin, base_flesh))
            Nrm = A.get("normals")
            nm = f"food{g}"
            sc.add_geometry(nm, to_o3d(V, F, C, Nrm), m_food)
            food_names.append(nm)
        q = Q[f]
        Tk = np.eye(4)
        cz, sz = np.cos(q[3]), np.sin(q[3])
        Tk[:3, :3] = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
        Tk[:3, 3] = q[:3]
        for nm in ("blade", "handle"):
            sc.set_geometry_transform(nm, Tk)
        for c, spec in cams.items():
            eye, look = np.asarray(spec["pos"]), np.asarray(spec["lookat"])
            dirv = (look - eye) / np.linalg.norm(look - eye)
            up = [0, 1, 0] if abs(dirv[2]) > 0.97 else [0, 0, 1]
            # 근접면을 직접 준다(자동이면 장면 크기로 정해져 카메라 앞 9cm 의 물체가 잘려 속이 보였다)
            rr.scene.camera.set_projection(spec["fov"], W / H, 0.005, 5.0, rendering.Camera.FovType.Vertical)
            rr.scene.camera.look_at(look, eye, up)
            img = np.asarray(rr.render_to_image())
            for _ in range(3):  # 드물게 검은 화면이 나오면(렌더러 일시 오류) 다시 그린다
                if img.mean() > 5:
                    break
                img = np.asarray(rr.render_to_image())
            if args.ss > 1:
                img = np.asarray(Image.fromarray(img).resize((res[0], res[1]), Image.LANCZOS))
            if args.first_only:
                stills.append(img)
            else:
                cont, st = writers[c]
                for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")):
                    cont.mux(pk)
            if n == 0 and not args.first_only:
                Image.fromarray(img).save(out / f"first_{c}.png")
        if n % 20 == 0:
            print(f"  frame {n}/{len(idx)} t={T[f]:.2f}s {(time.time() - t0) / (n + 1):.2f}s/frame", flush=True)
    for cont, st in writers.values():
        for pk in st.encode():
            cont.mux(pk)
        cont.close()
    if args.first_only:
        k = len(cams)
        rows = [np.concatenate(stills[i * k:(i + 1) * k], 1) for i in range(len(idx))]
        Image.fromarray(np.concatenate(rows, 0)).save(out / "stills.jpg", quality=88)
        print(out / "stills.jpg")
    print(f"done {len(idx)} frames in {time.time() - t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
