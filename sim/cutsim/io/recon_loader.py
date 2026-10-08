"""팀원 복원 결과 폴더(data/recon/<object_id>/)를 읽어 검증하고 단위·좌표를 맞춘다.

규약: mesh.{obj,ply,glb} + meta.yaml (필수), pointcloud.ply / traj_knife.npz (선택).
정규화 결과는 m 단위, z-up, 물체 바닥 중심이 원점.
"""
from pathlib import Path

import numpy as np
import trimesh
import yaml

UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001}
REQUIRED = {"object_id", "source", "category", "units", "up_axis", "frame"}
MESH_SUFFIXES = (".obj", ".ply", ".glb")


def read_meta(folder):
    folder = Path(folder)
    from cutsim.yamlio import load

    meta = load(folder / "meta.yaml")
    if missing := REQUIRED - meta.keys():
        raise ValueError(f"meta.yaml 누락: {sorted(missing)}")
    if meta["units"] not in UNIT:
        raise ValueError(f"units 는 {list(UNIT)} 중 하나여야 함: {meta['units']}")
    if meta["up_axis"] not in ("z", "y"):
        raise ValueError(f"up_axis 는 z|y: {meta['up_axis']}")
    if meta["frame"] == "world" and meta.get("T_world_object") is None:
        raise ValueError("frame=world 인데 T_world_object 가 없음")
    return meta


def find_mesh(folder):
    folder = Path(folder)
    for suffix in MESH_SUFFIXES:
        if (p := folder / f"mesh{suffix}").exists():
            return p
    raise FileNotFoundError(f"{folder} 에 mesh{{{','.join(MESH_SUFFIXES)}}} 없음")


def normalize_transform(units="m", up_axis="z", T_world_object=None):
    """원본 좌표 → (물체 좌표계, m, z-up) 아핀 변환(균일 스케일 포함). 바닥 중심 이동은 따로 붙인다."""
    A = np.eye(4)
    if T_world_object is not None:  # T_world_object 는 원본 단위 기준이라 가정
        A = np.linalg.inv(np.asarray(T_world_object, dtype=float)) @ A
    A = np.diag([UNIT[units]] * 3 + [1.0]) @ A
    if up_axis == "y":  # +90° about x: +y -> +z
        A = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]) @ A
    return A


def bottom_center(mesh):
    lo, hi = mesh.bounds
    return trimesh.transformations.translation_matrix([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])


def clean_mesh(mesh):
    """가장 큰 연결 성분만 남겨 손·테이블 조각을 떼고 작은 구멍을 메운다."""
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: m.area)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    return mesh, len(parts)


def load_recon(folder, size_range=(0.02, 0.30)):
    """returns (meta, mesh[m, z-up, 바닥중심], warnings, T_norm).

    T_norm: 원본 좌표 → 정규화 좌표 아핀 변환(스케일 포함). 칼 궤적도 같은 변환으로 옮겨야 한다.
    """
    folder = Path(folder)
    meta = read_meta(folder)
    raw = trimesh.load(find_mesh(folder), force="mesh")

    T = meta.get("T_world_object") if meta["frame"] == "world" else None
    A = normalize_transform(meta["units"], meta["up_axis"], T)
    mesh = raw.copy()
    mesh.apply_transform(A)
    mesh, n_parts = clean_mesh(mesh)
    C = bottom_center(mesh)  # 조각 제거 뒤의 바닥 중심
    mesh.apply_transform(C)
    T_norm = C @ A

    warnings = []
    if n_parts > 1:
        warnings.append(f"연결 성분 {n_parts}개 중 가장 큰 것만 사용")
    if not mesh.is_watertight:
        warnings.append("수밀 아님: 복셀 재구성 필요 (assets.volumize.voxel_remesh)")
    if not size_range[0] < mesh.extents.max() < size_range[1]:
        warnings.append(f"크기 의심 {mesh.extents.round(3)} m: 단위 확인")
    if not meta.get("scale_checked", False):
        warnings.append("scale_checked=false: 실물 치수 비교 전")
    return meta, mesh, warnings, T_norm


def load_knife_traj(folder, T_norm):
    """traj_knife.npz(t, T_world_knife, fps) → 정규화 좌표계 칼 자세 (N,4,4). 없으면 None.

    위치는 T_norm(스케일 포함)으로, 회전은 T_norm 의 회전 부분으로만 옮긴다.
    """
    p = Path(folder) / "traj_knife.npz"
    if not p.exists():
        return None
    d = np.load(p)
    P = d["T_world_knife"]
    s = np.cbrt(np.linalg.det(T_norm[:3, :3]))
    out = np.tile(np.eye(4), (len(P), 1, 1))
    out[:, :3, :3] = (T_norm[:3, :3] / s) @ P[:, :3, :3]
    out[:, :3, 3] = P[:, :3, 3] @ T_norm[:3, :3].T + T_norm[:3, 3]
    return {"t": d["t"], "T_norm_knife": out, "fps": float(d["fps"])}


def chamfer_and_scale(mesh, ref, n=20000, seed=0):
    """복원 품질 채점: 대칭 Chamfer(평균, m)와 크기 비율(복원/정답, bbox 대각선).

    두 메쉬는 같은 좌표계에 정렬되어 있어야 한다. 크기 비율이 0.1·0.01 근처면 단위 오류.
    """
    from scipy.spatial import cKDTree

    a = trimesh.sample.sample_surface(mesh, n, seed=seed)[0]
    b = trimesh.sample.sample_surface(ref, n, seed=seed)[0]
    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    scale = np.linalg.norm(mesh.extents) / np.linalg.norm(ref.extents)
    return {"chamfer_m": float((d_ab.mean() + d_ba.mean()) / 2), "scale_ratio": float(scale)}
