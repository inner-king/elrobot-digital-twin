"""DiSECt(LS-DYNA) 참고 힘 곡선 읽기와 시뮬레이션 곡선 비교.

CSV: 2줄 머리말 뒤 [t, Fx, t, Fy, t, Fz] (y-up, 칼은 -y 로 하강, 단위 N). 칼에 걸리는 위쪽 힘 = +Fy.
깊이 축은 "힘이 처음 문턱을 넘은 시점"을 0 으로 맞춘다(참고 데이터는 칼 메쉬 위치로 계산한 접촉 시점과
힘이 오르기 시작하는 시점이 0.1s 어긋나 있어, 양쪽을 같은 규칙으로 맞추는 쪽이 공정하다).
"""
from pathlib import Path

import numpy as np

DISECT_DIR = Path(__file__).resolve().parents[2] / "data" / "disect"
# DiSECt README 의 LS-DYNA 물성(재질 이름 → E, nu, rho)
DISECT_MATERIALS = {"cucumber": (2.5e6, 0.37, 950.0), "apple": (3.0e6, 0.17, 787.0), "potato": (2.0e6, 0.45, 630.0)}
# 힘 CSV 이름 → (형상, 재질, 하강 속도 m/s)
DISECT_RUNS = {
    "cylinder_fine": ("cylinder", "cucumber", 0.05),
    "cylinder_fine_vel35": ("cylinder", "cucumber", 0.035),
    "cylinder_fine_vel45": ("cylinder", "cucumber", 0.045),
    "cylinder_fine_vel55": ("cylinder", "cucumber", 0.055),
    "cylinder_fine_vel65": ("cylinder", "cucumber", 0.065),
    "cylinder_fine_prismprops": ("cylinder", "potato", 0.05),
    "cylinder_fine_sphereprops": ("cylinder", "apple", 0.05),
    "prism_fine": ("prism", "potato", 0.05),
    "sphere_fine": ("sphere", "apple", 0.05),
}
MESH_FILES = {"cylinder": "ansys_cyl_5mm.stl", "prism": "ansys_prism_5mm.stl", "sphere": "ansys_sphere_5mm.stl"}
# DiSECt 칼: 날 끝 0.08mm → 등 2mm 쐐기, 높이 40mm, 길이 125mm
DISECT_KNIFE = {"thickness": 0.002, "length": 0.125, "height": 0.04, "bevel_height": 0.04, "edge_width": 0.00008}


def load_force_csv(path):
    d = np.genfromtxt(path, delimiter=",", skip_header=2)
    return d[:, 0], np.stack([d[:, 1], d[:, 3], d[:, 5]], 1)


def smooth_depth(depth, force, window):
    """깊이 축 기준 이동 평균(window m)."""
    out = np.empty_like(force)
    for i, d in enumerate(depth):
        m = np.abs(depth - d) <= window / 2
        out[i] = force[m].mean()
    return out


def onset_index(force, thresh):
    idx = np.flatnonzero(force > thresh)
    return int(idx[0]) if len(idx) else None


def reference_curve(name, onset_N=1.0, smooth_m=0.001):
    """returns depth(m), force(N, 위쪽+), info. 깊이 0 = 힘이 onset_N 을 처음 넘은 시점."""
    shape, material, v = DISECT_RUNS[name]
    t, F = load_force_csv(DISECT_DIR / "forces" / f"{name}_resultant_force_xyz.csv")
    fy = F[:, 1]
    i0 = onset_index(fy, onset_N)
    depth = v * (t - t[i0])
    keep = depth >= 0
    depth, fy = depth[keep], fy[keep]
    return depth, smooth_depth(depth, fy, smooth_m), {"shape": shape, "material": material, "v": v,
                                                      "t_onset": float(t[i0]), "depth_end": float(depth[-1])}


def load_disect_mesh(shape):
    """DiSECt 메쉬(y-up, m) → z-up, 바닥 z=0, 칼날 면 법선 = x (원기둥 축 = x)."""
    import trimesh

    m = trimesh.load(DISECT_DIR / "meshes" / MESH_FILES[shape], force="mesh")
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # +y → +z
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return m


def compare(sim_depth, sim_force, ref_depth, ref_force, d_max, step=0.0005):
    """공통 깊이 구간 [0, d_max] 에서 RMSE 와 요약 지표."""
    d_max = min(d_max, sim_depth.max(), ref_depth.max())
    grid = np.arange(0.0, d_max + 1e-12, step)
    s = np.interp(grid, sim_depth, sim_force)
    r = np.interp(grid, ref_depth, ref_force)
    plateau = (grid >= 0.010) & (grid <= 0.040)
    return {
        "rmse_N": float(np.sqrt(np.mean((s - r) ** 2))),
        "rel_rmse": float(np.sqrt(np.mean((s - r) ** 2)) / max(np.abs(r).mean(), 1e-9)),
        "sim_plateau_N": float(s[plateau].mean()) if plateau.any() else np.nan,
        "ref_plateau_N": float(r[plateau].mean()) if plateau.any() else np.nan,
        "sim_peak_N": float(s.max()),
        "ref_peak_N": float(r.max()),
        "depth_compared_m": float(d_max),
    }, grid, s, r
