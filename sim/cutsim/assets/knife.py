"""칼날 메쉬(쐐기 단면)와 칼 지그 URDF 생성.

지그: world ─(prismatic x)→ slide_x ─(prismatic y)→ slide_y ─(prismatic z)→ slide_z ─(revolute z: yaw)→ blade
blade 링크 원점 = 칼날 끝(edge) 한가운데. yaw=0 일 때 칼날 길이 방향 = y, 칼날 면의 법선 = x.
slide_* 링크에는 충돌 형상이 없어 MPM 결합에 참여하지 않는다(결합 비용은 형상 수에 비례).
"""
import os
from pathlib import Path

import numpy as np
import trimesh

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "urdf"


def blade_mesh(thickness=0.0015, length=0.08, height=0.035, bevel_height=0.006, edge_width=0.0002):
    """x=두께, y=길이, z=높이. z=0 이 날 끝(폭 edge_width), z=bevel_height 부터 두께 thickness."""
    t, e, L, H, b = thickness / 2, edge_width / 2, length / 2, height, bevel_height
    # 단면(x-z) 다각형: 날 끝 → 베벨 → 등. 베벨이 높이 전체면(DiSECt 칼) 사다리꼴 하나
    if b >= H:
        prof = np.array([[-e, 0.0], [e, 0.0], [t, H], [-t, H]])
    else:
        prof = np.array([[-e, 0.0], [e, 0.0], [t, b], [t, H], [-t, H], [-t, b]])
    verts = [[x, y, z] for y in (-L, L) for x, z in prof]
    n = len(prof)
    faces = []
    for i in range(n):  # 옆면
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]
    for k in range(1, n - 1):  # 양 끝 마구리
        faces += [[0, k + 1, k], [n, n + k, n + k + 1]]
    mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    assert mesh.is_watertight and mesh.is_convex
    return mesh


def _inertial(mass, size=0.01):
    i = mass * size**2 / 6
    return (f'<inertial><mass value="{mass}"/><origin xyz="0 0 0"/>'
            f'<inertia ixx="{i}" iyy="{i}" izz="{i}" ixy="0" ixz="0" iyz="0"/></inertial>')


def write_knife_jig(name=None, thickness=0.0015, length=0.08, height=0.035,
                    bevel_height=0.006, blade_mass=0.1, edge_width=0.0002, travel=0.3, out_dir=ASSET_DIR):
    """칼 치수마다 다른 파일 이름을 써서, 칼이 다른 실행이 동시에 돌아도 서로 덮어쓰지 않게 한다."""
    if name is None:
        dims = (thickness, length, height, bevel_height, edge_width)
        name = "knife_jig_" + "_".join(f"{d * 1e4:.0f}" for d in dims)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / f"{name}_blade.obj"
    # 같은 칼로 동시에 도는 실행이 반쯤 쓴 파일을 읽지 않게 임시 파일에 쓰고 바꿔 끼운다
    tmp = out_dir / f"{name}_blade.{os.getpid()}.obj"
    blade_mesh(thickness, length, height, bevel_height, edge_width).export(tmp)
    os.replace(tmp, mesh_path)

    def prismatic(nm, parent, child, axis):
        return (f'<joint name="{nm}" type="prismatic"><parent link="{parent}"/><child link="{child}"/>'
                f'<origin xyz="0 0 0"/><axis xyz="{axis}"/>'
                f'<limit lower="{-travel}" upper="{travel}" effort="1000" velocity="2"/></joint>')

    urdf = f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="world_anchor">{_inertial(1.0)}</link>
  <link name="slide_x">{_inertial(0.05)}</link>
  <link name="slide_y">{_inertial(0.05)}</link>
  <link name="slide_z">{_inertial(0.05)}</link>
  <link name="blade">
    {_inertial(blade_mass, size=length)}
    <visual><geometry><mesh filename="{mesh_path.name}"/></geometry></visual>
    <collision><geometry><mesh filename="{mesh_path.name}"/></geometry></collision>
  </link>
  {prismatic("x", "world_anchor", "slide_x", "1 0 0")}
  {prismatic("y", "slide_x", "slide_y", "0 1 0")}
  {prismatic("z", "slide_y", "slide_z", "0 0 1")}
  <joint name="yaw" type="revolute"><parent link="slide_z"/><child link="blade"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.2" upper="3.2" effort="100" velocity="10"/></joint>
</robot>
"""
    urdf_path = out_dir / f"{name}.urdf"
    tmp = out_dir / f"{name}.{os.getpid()}.urdf"
    tmp.write_text(urdf)
    os.replace(tmp, urdf_path)
    return urdf_path


if __name__ == "__main__":
    print(write_knife_jig())
