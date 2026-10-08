"""도마(바닥 평면)·칼 지그·식재료(MPM) 씬 구성.

gs.init 은 호출하는 쪽에서 한 번 한다. 식재료는 FoodSpec 목록으로 받는다(껍질·과육처럼 여러 물체 가능).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yaml

import genesis as gs

from cutsim.assets.knife import write_knife_jig
from cutsim.assets.volumize import export_parts

ROOT = Path(__file__).resolve().parents[2]
KNIFE_DOFS = ("x", "y", "z", "yaw")


def load_yaml(path):
    from cutsim.yamlio import load

    return load(path)


def particle_size(grid_density):
    return 0.01 * 64.0 / grid_density  # Genesis MPMOptions 기본 규칙


def wave_speed(E, nu, rho):
    """탄성 압력파 속도 sqrt((lambda + 2 mu) / rho)."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return float(np.sqrt((lam + 2 * mu) / rho))


def cfl_substeps(mats, grid_density, step_dt, cfl=0.2, min_substeps=20):
    """가장 단단한 재료 기준으로 c*dt/dx <= cfl 이 되게 스텝당 분할 수를 정한다."""
    c = max(wave_speed(m["E"], m["nu"], m["rho"]) for m in mats)
    return max(min_substeps, int(np.ceil(step_dt / (cfl / grid_density / c))))


@dataclass
class FoodSpec:
    name: str
    morph: object  # gs.morphs.*
    mat: dict      # materials.yaml 의 flesh/skin 항목
    vis_mode: str = "particle"


@dataclass
class CutScene:
    scene: object
    knife: object
    foods: dict
    cams: dict
    cfg: dict
    dofs_idx: list = field(default_factory=list)
    parity: dict = field(default_factory=dict)  # 입자별 조각 짝/홀 라벨(two-field 패치)

    @property
    def dt(self):
        return self.cfg["sim"]["step_dt"]

    def knife_q(self):
        return self.knife.get_dofs_position(self.dofs_idx).cpu().numpy()

    def knife_ctrl_force(self):
        return self.knife.get_dofs_control_force(self.dofs_idx).cpu().numpy()

    def command(self, q, qd):
        self.knife.control_dofs_position_velocity(
            np.asarray(q, dtype=np.float32), np.asarray(qd, dtype=np.float32), self.dofs_idx)

    def set_knife(self, q):
        self.knife.set_dofs_position(np.asarray(q, dtype=np.float32), self.dofs_idx)
        self.knife.set_dofs_velocity(np.zeros(len(q), dtype=np.float32), self.dofs_idx)

    def particles(self):
        """returns {name: pos (N,3) np}"""
        return {k: e.get_particles_pos().cpu().numpy().reshape(-1, 3) for k, e in self.foods.items()}

    def mark_cut(self, point, normal):
        """절단 하나가 끝났을 때 호출: 절단면 + 쪽 입자의 짝/홀 라벨을 뒤집는다.

        평행한 절단이 여러 번이어도 이웃한 조각은 항상 라벨이 달라진다. 라벨이 다른 입자는 서로 다른
        격자를 쓰고 파고들 때만 접촉하므로, 칼을 뺀 뒤 떨어진 조각이 격자 공유로 다시 붙지 않는다.
        (Genesis 에 patches/make_genesis_multifield.py 가 적용되어 있어야 한다.)
        """
        point, normal = np.asarray(point, float), np.asarray(normal, float)
        solver = next(iter(self.foods.values())).solver
        flips = {}
        for k, e in self.foods.items():
            pos = e.get_particles_pos().cpu().numpy().reshape(-1, 3)
            par = self.parity.setdefault(k, np.zeros(len(pos), np.int32))
            flips[k] = (pos - point) @ normal > 0
            par ^= flips[k].astype(np.int32)
            solver.set_particles_field(e, par)
        return flips  # 이번 절단으로 라벨이 뒤집힌 입자(물체별)

    def grip_food(self, omega=3000.0):
        """손 상자 바닥 면적(x·y) 안의 입자를 그 상자 링크에 붙인다(Genesis 입자 구속: 서브스텝마다 임계 감쇠
        스프링). omega: 입자 하나의 고유 진동수(rad/s), 클수록 단단히 쥔다. returns 붙인 입자 수."""
        p = particle_size(self.cfg["mpm"]["grid_density"])
        n = 0
        for ent, c, s in self.holds:
            lo, hi = c - s / 2, c + s / 2
            for e in self.foods.values():
                mask = e.get_particles_in_bbox((lo[0], lo[1], -1.0), (hi[0], hi[1], 1.0))
                k = e.material.rho * p ** 3 * omega ** 2  # 입자 질량 x omega^2
                e.set_particle_constraints(mask, int(ent.base_link_idx), float(k))
                n += int(mask.sum())
        return n

    def set_contact_friction(self, mu):
        next(iter(self.foods.values())).solver.set_multifield_friction(mu)

    def set_multifield_fill(self, fill):
        """조각 사이 접촉을 채움 비율 fill 이상인 격자점에서만(patches 4단계). 0 이면 원래 동작."""
        e = next(iter(self.foods.values()))
        e.solver.set_multifield_fill(fill, max(e.material.rho for e in self.foods.values()))

    def set_damping(self, c):
        """입자별 감쇠율(1/s, 물체 순서대로 이어 붙인 배열). Genesis 가 서브스텝마다 속도·회전에 건다(patches 5단계)."""
        start = 0
        for e in self.foods.values():
            e.solver.set_particles_damping(e, np.asarray(c[start:start + e.n_particles], np.float32))
            start += e.n_particles

    def set_inside_sep(self, flag=True):
        """칼 안쪽 격자점을 양쪽 재료 모두와 분리(patches 2단계). 칼-격자 정렬에 따른 가짜 힘·비대칭을 없앤다."""
        next(iter(self.foods.values())).solver.set_cpic_inside_sep(flag)

    def set_knife_force_limit(self, dof, limit):
        i = KNIFE_DOFS.index(dof)
        self.knife.set_dofs_force_range(np.array([-limit], np.float32), np.array([limit], np.float32),
                                        [self.dofs_idx[i]])

    def particle_J(self):
        """returns {name: det(F) (N,)} — 입자 부피 비율(현재/초기)."""
        out = {}
        for k, e in self.foods.items():
            F = e.get_state().F
            F = F.cpu().numpy() if hasattr(F, "cpu") else np.asarray(F)
            out[k] = np.linalg.det(F.reshape(-1, 3, 3))
        return out


def primitive_mesh(food_cfg):
    r = food_cfg["radius"]
    if food_cfg["shape"] == "cylinder":
        m = trimesh.creation.cylinder(radius=r, height=food_cfg["length"], sections=96)
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))  # z축 → x축
    elif food_cfg["shape"] == "sphere":
        m = trimesh.creation.icosphere(subdivisions=4, radius=r)
    else:
        raise ValueError(food_cfg["shape"])
    m.apply_translation([0, 0, -m.bounds[0, 2]])  # 바닥 = z 0
    return m


def food_specs_from_mesh(mesh, category, materials, grid_density, two_layer, work_dir, z_lift=0.0005,
                         vis_mode="particle"):
    """수밀 메쉬(m, 바닥 z=0) → FoodSpec 목록. two_layer 면 껍질·과육으로 나눈다."""
    mat = materials.get(category, materials["default"])
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pos = (0.0, 0.0, z_lift)
    if not two_layer:
        path = work_dir / "whole.obj"
        mesh.export(path)
        return [FoodSpec("flesh", gs.morphs.Mesh(file=str(path), pos=pos), mat["flesh"], vis_mode)], {}
    # 입자보다 얇은 껍질은 표현 못 하므로 최소 1입자 두께로 올린다.
    p = particle_size(grid_density)
    skin_t = max(mat["skin_thickness_m"], p)
    paths, vols = export_parts(mesh, skin_t, work_dir, pitch=min(skin_t / 2, 0.0005))
    vols["skin_thickness_used_m"] = skin_t
    specs = [FoodSpec(k, gs.morphs.Mesh(file=str(paths[k]), pos=pos), mat[k], vis_mode) for k in ("flesh", "skin")]
    return specs, vols


def build_cut_scene(cfg, food_specs, record_dir=None, fps=25, with_food=True, view_scale=1.0, face_x=0.0,
                    view_center=(0.0, 0.0), hold_boxes=(), grip=False):
    """view_scale: 카메라 거리 배율(물체 크기 / 7cm). face_x: 'face' 카메라가 바라볼 절단면 x 위치.
    view_center: 카메라들을 이 (x, y) 를 중심으로 옮긴다(긴 물체의 한쪽 끝을 썰 때).
    hold_boxes: [(중심 xyz, 크기 xyz)] 식재료를 위에서 눌러 붙잡는 고정 상자(사람의 다른 손 대용).
    grip: 상자가 재료와 부딪히지 않는 손 모양 표시만 하고, 붙잡기는 grip_food() 의 입자 구속으로 한다."""
    mcfg, kcfg, scfg = cfg["mpm"], cfg["knife"], cfg["sim"]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=scfg["step_dt"], substeps=scfg["substeps"]),
        mpm_options=gs.options.MPMOptions(
            lower_bound=tuple(mcfg["lower_bound"]), upper_bound=tuple(mcfg["upper_bound"]),
            grid_density=mcfg["grid_density"], enable_CPIC=mcfg["enable_CPIC"]),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        vis_options=gs.options.VisOptions(visualize_mpm_boundary=False, show_world_frame=False),
        show_viewer=False,
    )
    bcfg = cfg["board"]
    board_color = (0.75, 0.62, 0.45)
    # 격자점에서만 접촉을 계산하므로, 떨어진 물체는 도마 면 격자점의 영향권(약 1.3칸) 위에서 멈춘다. 충돌 면을
    # collision_offset_dx 칸 내려 두면 입자가 보이는 도마(z=0)에 닿아 쉰다. 보이는 도마는 충돌 없는 판으로 따로 둔다.
    off = float(bcfg.get("collision_offset_dx", 0.0)) / mcfg["grid_density"]
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -off), visualization=off == 0.0),
                     material=gs.materials.Rigid(coup_friction=bcfg["coup_friction"],
                                                 coup_softness=bcfg["coup_softness"]),
                     surface=gs.surfaces.Default(color=board_color))
    if off > 0.0:
        scene.add_entity(gs.morphs.Box(pos=(0.0, 0.0, -0.005), size=(0.6, 0.6, 0.01), fixed=True, collision=False),
                         material=gs.materials.Rigid(needs_coup=False), surface=gs.surfaces.Default(color=board_color))
    urdf = write_knife_jig(thickness=kcfg["thickness"], length=kcfg["length"], height=kcfg["height"],
                           bevel_height=kcfg["bevel_height"], blade_mass=kcfg["blade_mass"],
                           edge_width=kcfg.get("edge_width", 0.0002))
    knife = scene.add_entity(
        gs.morphs.URDF(file=str(urdf), fixed=True, merge_fixed_links=False, convexify=False),
        material=gs.materials.Rigid(coup_friction=kcfg["coup_friction"], coup_softness=kcfg["coup_softness"],
                                    sdf_cell_size=kcfg["sdf_cell_size"], sdf_max_res=kcfg["sdf_max_res"],
                                    gravity_compensation=1.0),
        surface=gs.surfaces.Default(color=(0.75, 0.78, 0.82)),
    )
    holds = []
    for center, size in hold_boxes:
        # 칼질마다 MPM 이 남은 몸통을 격자 한 칸쯤 밀어내는데, 사람은 다른 손으로 누르고 썬다. 그 손 대용.
        if grip:
            mat = gs.materials.Rigid(needs_coup=False)
            morph = gs.morphs.Box(pos=tuple(center), size=tuple(size), fixed=True, collision=False)
        else:
            mat = gs.materials.Rigid(coup_friction=1.0, coup_softness=kcfg["coup_softness"], sdf_cell_size=0.0005)
            morph = gs.morphs.Box(pos=tuple(center), size=tuple(size), fixed=True)
        holds.append((scene.add_entity(morph, material=mat, surface=gs.surfaces.Default(color=(0.86, 0.68, 0.58))),
                      np.asarray(center, float), np.asarray(size, float)))
    foods = {}
    if with_food:
        for spec in food_specs:
            m = spec.mat
            foods[spec.name] = scene.add_entity(
                spec.morph,
                material=gs.materials.MPM.ElastoPlastic(E=m["E"], nu=m["nu"], rho=m["rho"],
                                                       von_mises_yield_stress=m["yield_stress"]),
                surface=gs.surfaces.Default(color=tuple(m["color"]), vis_mode=spec.vis_mode),
            )
    c = cfg["camera"]
    k = view_scale
    cx, cy = view_center
    o = np.array([cx, cy, 0.0])
    cam_specs = {
        "persp": (np.array(c["pos"]) * k + o, np.array(c["lookat"]) * k + o),
        # 칼날 면을 정면으로 보는 옆 카메라(x 축 방향): 칼이 지나간 틈이 세로선으로 보인다
        "side": ((cx, cy - 0.17 * k, 0.02 * k), (cx, cy, 0.015 * k)),
        # 위에서 내려다보는 카메라: 조각이 벌어지는지(틈) 확인
        "top": ((cx, cy - 0.001, 0.2 * k), (cx, cy, 0.0)),
        # 절단면을 +x 쪽 비스듬히 위에서 보는 카메라: 오른쪽 조각을 치운 뒤 왼쪽 조각의 단면(과육·껍질)이 보인다
        "face": ((face_x + 0.09 * k, cy - 0.04 * k, 0.07 * k), (face_x, cy, 0.015 * k)),
    }
    cam_specs = {n: {"pos": [float(v) for v in p], "lookat": [float(v) for v in l], "fov": float(c["fov"]),
                     "res": [int(v) for v in c["res"]]} for n, (p, l) in cam_specs.items()}
    cams = {}
    if record_dir is not None:
        for n, sp in cam_specs.items():
            cams[n] = scene.add_camera(res=tuple(sp["res"]), pos=tuple(sp["pos"]), lookat=tuple(sp["lookat"]),
                                       fov=sp["fov"], GUI=False)
    scene.build()

    dofs_idx = [knife.get_joint(n).dofs_idx_local[0] for n in KNIFE_DOFS]
    knife.set_dofs_kp(np.array(kcfg["kp"], dtype=np.float32), dofs_idx)
    knife.set_dofs_kv(np.array(kcfg["kv"], dtype=np.float32), dofs_idx)
    cs = CutScene(scene, knife, foods, cams, cfg, dofs_idx)
    cs.cam_specs, cs.holds, cs.board_offset = cam_specs, holds, off
    if record_dir is not None:
        # Genesis 는 NVIDIA 하드웨어 인코더(NVENC)를 먼저 고르는데, 노트북 GPU 는 동시 세션 수가 제한돼 실행 여러 개가
        # 함께 영상을 쓰면 저장이 실패한다. CPU 인코더(libx264)를 먼저 쓰게 한다(CUTSIM_VIDEO_CODEC 로 바꿀 수 있음).
        import genesis.utils.video_encoder as venc

        pref = os.environ.get("CUTSIM_VIDEO_CODEC", "libx264")
        venc.H264_CODEC_CANDIDATES = (pref,) + tuple(c for c in venc.H264_CODEC_CANDIDATES if c != pref)
        Path(record_dir).mkdir(parents=True, exist_ok=True)
        for k, cam in cams.items():
            cam.start_recording(save_to_filename=str(Path(record_dir) / f"{k}.mp4"), fps=fps)
    return cs


def stop_recording(cs):
    for cam in cs.cams.values():
        cam.stop_recording()
