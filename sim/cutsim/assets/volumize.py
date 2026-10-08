"""표면 메쉬를 MPM 에 넣을 수 있는 수밀 부피로 만들고 껍질·과육으로 나눈다.

과육 = 수밀 메쉬를 복셀화해 skin_thickness 만큼 침식한 뒤 marching cubes.
껍질 = 외부 메쉬 - 과육 (manifold3d 차집합). 두 물체가 겹치지 않아야 입자가 이중으로 생기지 않는다.
"""
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage


MAX_VOXELS = 3e7  # 이보다 크면 단위 오류일 가능성이 높다(8cm 사과를 0.5mm 로 나눠도 5e6)


def _filled_voxels(mesh, pitch, pad=2):
    """복셀 중심마다 일반화 winding number 로 안팎을 판정한다.

    표면 복셀화 + 내부 채우기(flood fill)는 메쉬에 구멍이 있으면 안쪽까지 새어 나가 속이 빈 껍데기가 된다.
    winding number 는 구멍·겹침이 있어도 안쪽을 안정적으로 판정한다.
    """
    import igl

    lo = mesh.bounds[0] - pad * pitch
    n = np.ceil((mesh.bounds[1] + pad * pitch - lo) / pitch).astype(int) + 1
    if np.prod(n.astype(float)) > MAX_VOXELS:
        raise ValueError(f"복셀 {n.tolist()}개(pitch {pitch * 1000:.2f}mm, 물체 {mesh.extents.round(3)} m)가 너무 많음: "
                         "단위(units) 확인")
    axes = [lo[i] + pitch * np.arange(n[i]) for i in range(3)]
    q = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    w = igl.fast_winding_number(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64), q)
    return (w > 0.5).reshape(n), lo  # lo = 복셀 (0,0,0) 중심의 월드 좌표


def _grid_to_mesh(grid, origin, pitch):
    from skimage import measure

    verts, faces, _, _ = measure.marching_cubes(grid.astype(np.float32), level=0.5)
    mesh = trimesh.Trimesh(verts * pitch + origin, faces, process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def voxel_remesh(mesh, pitch=0.001):
    """수밀이 아닌 복원 메쉬를 복셀 재구성으로 수밀하게 만든다(형상은 pitch 만큼 뭉개짐)."""
    grid, origin = _filled_voxels(mesh, pitch)
    return _grid_to_mesh(grid, origin, pitch)


def split_skin_flesh(mesh, skin_thickness, pitch=None):
    """returns (skin, flesh). 껍질이 너무 얇아 침식 후 과육이 사라지면 ValueError."""
    if not mesh.is_watertight:
        raise ValueError("split_skin_flesh 는 수밀 메쉬가 필요: voxel_remesh 먼저")
    pitch = pitch or skin_thickness / 2
    grid, origin = _filled_voxels(mesh, pitch)
    iters = max(1, int(round(skin_thickness / pitch)))
    inner = ndimage.binary_erosion(grid, iterations=iters)
    if inner.sum() == 0:
        raise ValueError(f"skin_thickness={skin_thickness} 가 물체보다 두꺼움")
    flesh = _grid_to_mesh(inner, origin, pitch)
    skin = trimesh.boolean.difference([mesh, flesh], engine="manifold")
    return skin, flesh


def export_parts(mesh, skin_thickness, out_dir, pitch=None):
    """껍질·과육 메쉬를 out_dir/{skin,flesh}.obj 로 저장하고 경로를 돌려준다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skin, flesh = split_skin_flesh(mesh, skin_thickness, pitch)
    paths = {"skin": out_dir / "skin.obj", "flesh": out_dir / "flesh.obj"}
    skin.export(paths["skin"])
    flesh.export(paths["flesh"])
    return paths, {"skin_volume": skin.volume, "flesh_volume": flesh.volume, "outer_volume": mesh.volume}
