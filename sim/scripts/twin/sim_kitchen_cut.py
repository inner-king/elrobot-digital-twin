"""복원 주방 전체를 Genesis 한 장면에 실제 물체로 넣고, ElRobot 이 쥔 칼로 당근을 썬다(물체를 그림에만 넣던 것 → 물리로).

01_cut_primitive.py 의 TR 재생(칼 지그·당근 MPM·절단 라벨·감쇠·힘 예산)을 그대로 돌리되 장면 만드는 함수와 스텝 함수만 바꿔 끼운다.
- 좌표: 시뮬레이션은 01 과 같은 당근 정규화 좌표. 주방 물체·로봇·바닥을 T_cw(세계 → 정규화)로 옮겨 넣는다.
- 바닥: 무한 평면(세계 z=0). 도마·딸기·사과·바나나·머그·냄비: 복원 메쉬(볼록 껍질 하나로 충돌), 중력·마찰·서로 충돌하는
  자유 강체. 질량은 실물 무게 추정값(MASS_KG)으로 맞춘다(볼록 껍질은 속이 찬 덩어리라 밀도를 그대로 쓰면 머그·냄비가 너무 무겁다).
  복원된 칼(두께 18 mm 덩어리)은 자를 수 없어 빼고 로봇이 쥔 칼 모델로, 복원된 당근은 MPM 당근으로 대신한다.
  복원 도마는 윗면에 붙어 남은 물체 밑면 턱(최대 +5 mm)과 바닥 아래 −1.5 mm 를 잘라 평평하게(두께 19.9 mm, 정답 20 mm).
- 잠재우기(sleeping, 물리 엔진의 흔한 방식): MPM 때문에 한 스텝(1 ms)을 199 번 나눠(5 µs) 계산하는데, float32 에서는 멈춰 있는
  물체의 한 번 이동량이 위치 최소 단위보다 작아 반올림으로 한 방향으로 1 단위씩 미끄러진다(머그 약 1.5 mm/s, 마찰 모델과 무관).
  그래서 다 내려앉아 멈춘(속도 < 3 mm/s, 3°/s 가 30 스텝) 물체는 그 자세로 붙잡아 두고, 무언가에 부딪혀 빨라지면(> 20 mm/s,
  20°/s) 깨워서 다시 물리로 움직이게 한다. 깨어난 기록은 kitchen.json 에 남긴다.
- 당근(MPM)은 01 처럼 도마 윗면 높이의 받침 평면에 놓인다. 받침 평면은 입자와만 닿고 강체와는 충돌하지 않는다(충돌 그룹 비트를
  아무와도 안 맞게). 격자 접촉 때문에 받침 면을 한 칸 내려 두는 01 의 방식(collision_offset_dx)을 강체 도마로는 할 수 없어서,
  도마 강체는 입자와 직접 닿지 않는다.
- 로봇: URDF(팔 7 + 집게), 받침 고정. 매 스텝 실제 칼 자세 → 손잡이를 쥐는 집게 목표 → IK → 관절 위치·속도 제어(PD, 모터 힘
  상한 2.94 N·m, 중력 보상 없음). 바닥·물체와 충돌한다. 칼은 01 처럼 지그가 움직인다(로봇 모터 힘으로 써는 것은 아님). 그래서
  로봇과 칼은 서로 충돌하지 않게 했다(손잡이는 그림에만 있다).
python scripts/twin/sim_kitchen_cut.py --layout_only        # 배치·사전 충돌 검사만(GPU 안 씀)
python scripts/twin/sim_kitchen_cut.py [--smoke 1.5]         # --smoke: 궤적 앞 1.5 s 만(점검용, scratch 로)
결과: reports/09_elrobot_twin/robot_cut_full/TR_kitchen_full/{metrics.json, frames.npz, kitchen_frames.npz, kitchen.json, *.mp4}
"""
import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(Path(__file__).parent)]
from cutsim.io.recon_loader import load_recon
from make_robot_cut import knife_grip, knife_T
from robot_kin import ARM, GRIP, TCP_L, URDF, Robot

SCEN = ROOT / "data/recon/kitchen_robot_cut_carrot"
RECON = ROOT / "data/kitchen_recon/full"
IMPORT = ROOT / "reports/09_elrobot_twin/robot_cut/import/kitchen_robot_cut_carrot"
OUT_ROOT = ROOT / "reports/09_elrobot_twin/robot_cut_full"
MASS_KG = {"도마": 0.8, "딸기": 0.02, "사과": 0.2, "바나나": 0.15, "머그": 0.35, "냄비": 1.2}   # 실물 무게 추정
EXCLUDE = ("당근", "칼")
CLEAR = 0.0005                                   # 처음 놓을 때 받침 위 여유(m): 껍질끼리 겹친 채 시작하지 않게
FLOOR, OBJ, ROB, KNF = 1, 2, 4, 8                # 충돌 그룹 비트(contype/conaffinity)
SUP_T, SUP_A = 1 << 10, 1 << 11                  # 받침 평면: 누구와도 짝이 안 맞는 비트(충돌 형상은 남아 입자와만 닿음)
JAWS = ("rev_motor_08_1", "rev_motor_08_2")
GRIP_GAP_MM = 15.0                               # 손잡이 두께
PHYS_CUT_ARGS = ["--multifield", "--category", "potato", "--set", "food.two_layer=true", "--set", "mpm.grid_density=256",
                 "--test", "TR", "--skin_color", "0.545,0.29,0.129", "--flesh_color", "0.97,0.55,0.15",
                 "--hold_box=-0.0522,0,0.0258,0.047,0.05,0.015", "--cut_force", "--no_baseline", "--grip",
                 "--set", "board.collision_offset_dx=1.0", "--mf_fill", "0.75", "--sep_damp", "300", "--sep_damp_r", "0.015",
                 "--save_frames", "25", "--z_force_budget", "12", "--domain_pad", "0.02"]   # 이전 실행(TR_robot_carrot)과 같은 값


def wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return (w, x, y, z)


def top_on(hull, xy):
    """볼록 껍질 윗면 높이(위에서 아래로 쏜 광선). 안 맞으면 nan"""
    o = np.c_[xy, np.full(len(xy), 2.0)]
    loc, ray, _ = hull.ray.intersects_location(o, np.tile([0, 0, -1.0], (len(xy), 1)), multiple_hits=False)
    z = np.full(len(xy), np.nan)
    z[ray] = loc[:, 2]
    return z


def halfspaces(h):
    n = h.face_normals
    return n, (n * h.triangles[:, 0]).sum(1)


def depth_inside(P, hs):
    """볼록 껍질 안으로 들어간 깊이(m, 밖이면 ≤0)"""
    n, d = hs
    return -(P @ n.T - d).max(1)


def layout(verbose=True):
    """주방 물체를 세계 좌표(바닥 z=0)에 놓고, 당근 정규화 좌표와의 변환을 정한다."""
    _, carrot_n, _, T_norm = load_recon(SCEN)
    assert np.allclose(T_norm[2, :3], [0, 0, 1]) and np.allclose(T_norm[:2, 2], 0), "정규화가 수평 회전이 아니다"
    info = json.load(open(RECON / "info.json"))
    objs = {}
    for o in info["objects"]:
        if o["match"] in EXCLUDE:
            continue
        m = trimesh.load(RECON / "objects" / f"obj_{o['id']}.obj", force="mesh", process=False)
        objs[o["match"]] = dict(id=o["id"], mesh=m, on_board=(o.get("support_mm") or 0) >= 5)
    board = objs["도마"]["mesh"]
    # 복원 도마 윗면에는 딸기·당근·칼 밑면이 붙어 남은 턱(최대 +5 mm)이, 아래에는 바닥과 붙은 −1.5 mm 가 있다. 볼록 껍질로
    # 충돌하면 그 턱 높이가 도마 윗면이 되어 물체가 떠 보인다 → 윗면 대부분의 높이(위를 보는 꼭짓점 90%)와 바닥 0 사이로 자른다.
    V = board.vertices.copy()
    z_top = float(np.percentile(V[board.vertex_normals[:, 2] > 0.7, 2], 90))
    n_clip = int(((V[:, 2] > z_top) | (V[:, 2] < 0)).sum())
    V[:, 2] = np.clip(V[:, 2], 0.0, z_top)
    board.vertices = V
    board.apply_translation([0, 0, CLEAR])
    bh = board.convex_hull
    objs["도마"]["lift_mm"] = CLEAR * 1000
    objs["도마"]["flatten"] = {"top_mm": round(z_top * 1000, 1), "clipped_vertices": n_clip}
    for name, o in objs.items():
        if name == "도마":
            continue
        m = o["mesh"]
        V = m.convex_hull.vertices
        if o["on_board"]:                       # 도마 껍질(Genesis 가 충돌에 쓰는 모양) 위에
            zt = top_on(bh, V[:, :2])
            ok = np.isfinite(zt)
            lift = float(np.max(zt[ok] - V[ok, 2])) + CLEAR
        else:
            lift = CLEAR - float(V[:, 2].min())
        m.apply_translation([0, 0, lift])
        o["lift_mm"] = lift * 1000
    # 당근: 정규화 좌표 바닥(z=0) = 그 자리 도마 껍질 윗면(가장 높은 곳)
    T_wc = np.linalg.inv(T_norm)
    car_w = carrot_n.copy().apply_transform(T_wc)
    lo, hi = car_w.bounds
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 15), np.linspace(lo[1], hi[1], 5))
    zt = top_on(bh, np.c_[gx.ravel(), gy.ravel()])
    dz = float(np.nanmax(zt) - T_wc[2, 3])
    T_wc[2, 3] += dz
    T_cw = np.linalg.inv(T_wc)
    for name, o in objs.items():
        m = o["mesh"]
        h = m.convex_hull
        c = h.centroid
        o["center_w"] = c
        o["pos_c"] = (T_cw @ np.r_[c, 1.0])[:3]
        o["quat_c"] = wxyz(T_cw[:3, :3])
        o["hull_vol"] = float(h.volume)
        o["rho"] = MASS_KG[name] / h.volume
    if verbose:
        print(f"당근 바닥을 도마 껍질 윗면에 맞춤: {dz * 1000:+.1f} mm (당근 밑 껍질 윗면 높이 범위 "
              f"{np.nanmin(zt) * 1000:.1f}~{np.nanmax(zt) * 1000:.1f} mm)")
        for name, o in objs.items():
            print(f"  {name}: {'도마 위' if o['on_board'] else '바닥'} 올림 {o['lift_mm']:+.1f} mm, 껍질 부피 "
                  f"{o['hull_vol'] * 1e6:.0f} cm³ → 밀도 {o['rho']:.0f} kg/m³ ({MASS_KG[name] * 1000:.0f} g)")
    meta = json.loads(json.dumps(__import__("yaml").safe_load((SCEN / "meta.yaml").read_text())["robot_grip"]))
    G = knife_grip(np.radians(meta["alpha_deg"]), meta["s_m"])
    return dict(T_wc=T_wc, T_cw=T_cw, objs=objs, board_hull=bh, carrot_w=carrot_n.copy().apply_transform(T_wc), dz=dz,
                T_base_w=np.asarray(meta["T_base"], float), G=G, z_floor_c=float((T_cw @ [0, 0, 0, 1.0])[2]))


def planned_knife(traj, every=40, dt=0.001):
    """01 의 TR 변환과 같게: 날 끝 위치 + 수평 방위 → (x, y, z, yaw), 1 ms 간격 중 every 마다"""
    d = np.load(traj)
    t, pos, rot = (np.asarray(d[k], float) for k in ("t", "pos", "rot"))
    yaw = np.unwrap(np.arctan2(-rot[:, 0, 1], rot[:, 1, 1]))
    tn = np.arange(t[0], t[-1] + 0.5 * dt, dt)[::every]
    return np.column_stack([np.interp(tn, t, v) for v in (*pos.T, yaw)])


def solve_ik_path(rb, targets, n_start=16):
    best, rng = None, np.random.default_rng(1)
    for k in range(n_start):
        r = rb.ik(targets[0], np.zeros(7) if k == 0 else rng.uniform(rb.lo, rb.hi))
        if best is None or r[1] + 0.05 * r[2] < best[1] + 0.05 * best[2]:
            best = r
    q, qs, errs = best[0], [], []
    for T in targets:
        q, e1, e2 = rb.ik(T, q)
        qs.append(q); errs.append((e1, e2))
    return np.array(qs), np.array(errs)


def precheck(L, traj):
    """계획 궤적에서 로봇(링크 볼록 껍질)·칼 손잡이가 물체 껍질에 들어가는지, 물체끼리 처음부터 겹치는지, MPM 영역과 겹치는지."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import render_surface as RS
    hulls = {n: o["mesh"].convex_hull for n, o in L["objs"].items()}
    hs = {n: halfspaces(h) for n, h in hulls.items()}
    print("물체끼리 처음 겹침(껍질 표면 점이 다른 껍질 안으로 0.2 mm 넘게):")
    for a in hulls:
        P = np.r_[hulls[a].vertices, hulls[a].sample(3000, seed=0) if hasattr(hulls[a], "sample") else hulls[a].vertices]
        for b in hulls:
            if a < b:
                dd = depth_inside(P, hs[b])
                if (dd > 0.0002).any():
                    print(f"  {a}-{b}: {int((dd > 0.0002).sum())}점, 최대 {dd.max() * 1000:.1f} mm")
    # MPM 영역(01 의 --food_mesh 규칙, domain_pad 0.02)
    lo, hi = trimesh.load(IMPORT / "normalized.obj", force="mesh").bounds
    pad = 3.5 / 256
    dlo = np.array([lo[0] - 0.032 - pad, lo[1] - 0.045 - pad, -0.03])
    dhi = np.array([hi[0] + 0.032 + pad, hi[1] + 0.045 + pad, hi[2] + 0.02 + pad])
    for n, h in hulls.items():
        Vc = h.vertices @ L["T_cw"][:3, :3].T + L["T_cw"][:3, 3]
        a, b = Vc.min(0), Vc.max(0)
        if (a < dhi).all() and (b > dlo).all():
            print(f"  주의: {n} 껍질이 MPM 영역과 겹침(입자와 닿을 수 있음)")
    # 로봇·손잡이 궤적
    rb = Robot(T_base=L["T_base_w"])
    q4 = planned_knife(traj)
    Tk = [L["T_wc"] @ knife_T(q) for q in q4]
    t0 = time.time()
    qs, errs = solve_ik_path(rb, [T @ L["G"] for T in Tk])
    print(f"IK {len(qs)}점: 위치 오차 최대 {errs[:, 0].max():.2f} mm, 방향 {errs[:, 1].max():.2f}° "
          f"({(time.time() - t0) / len(qs) * 1000:.0f} ms/점, 40 ms 간격이라 1 ms 간격보다 느림)")
    links = {n: m.convex_hull for n, m, _ in rb.meshes()}
    pts = {n: np.r_[h.vertices, h.sample(400, seed=0)] for n, h in links.items()}
    _, handle = RS.knife_meshes({"thickness": 0.0012, "length": 0.08, "height": 0.035, "bevel_height": 0.006,
                                 "edge_width": 0.0002})
    hp = np.r_[handle.vertices, handle.sample(500, seed=0)]
    hits = {}
    for q, T in zip(qs, Tk):
        Tl = rb.fk(q, Robot.grip_angle(GRIP_GAP_MM))
        for ln, P in pts.items():
            Pw = P @ Tl[ln][:3, :3].T + Tl[ln][:3, 3]
            if Pw[:, 2].min() < 0:
                hits[(ln, "바닥")] = max(hits.get((ln, "바닥"), 0), -Pw[:, 2].min())
            for on, h in hs.items():
                dd = depth_inside(Pw, h)
                if dd.max() > 0:
                    hits[(ln, on)] = max(hits.get((ln, on), 0), dd.max())
        Hw = hp @ T[:3, :3].T + T[:3, 3]
        for on, h in hs.items():
            dd = depth_inside(Hw, h)
            if dd.max() > 0:
                hits[("손잡이", on)] = max(hits.get(("손잡이", on), 0), dd.max())
    print("계획 궤적에서 로봇 링크 껍질·손잡이가 물체 껍질에 들어간 최대 깊이:", {f"{a}→{b}": f"{v * 1000:.1f} mm" for (a, b), v in hits.items()} or "없음")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout_only", action="store_true")
    ap.add_argument("--smoke", type=float, default=0.0, help="궤적 앞 이 시간(s)만 돌려 점검(결과는 scratch)")
    ap.add_argument("--scratch", default=os.environ.get("CUTSIM_SCRATCH", "/tmp"))
    ap.add_argument("--tag", default="kitchen_full")
    ap.add_argument("--kp_hz", type=float, default=20.0, help="로봇 관절 PD 고유 진동수(Hz, 관절 관성 기준)")
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args()
    L = layout()
    traj = IMPORT / "traj_sim.npz"
    if a.layout_only:
        precheck(L, traj)
        return
    out_root = OUT_ROOT
    if a.smoke > 0:
        d = dict(np.load(traj))
        k = d["t"] <= d["t"][0] + a.smoke
        out_root = Path(a.scratch) / "smoke"
        out_root.mkdir(parents=True, exist_ok=True)
        traj = out_root / "traj_smoke.npz"
        np.savez(traj, **{n: (v[k] if getattr(v, "ndim", 0) and len(v) == len(k) else v) for n, v in d.items()})
    out = out_root / f"TR_{a.tag}"
    mdir = out / "kitchen_meshes"
    mdir.mkdir(parents=True, exist_ok=True)
    for name, o in L["objs"].items():           # 물체 좌표 원점 = 껍질 중심(세계 축 방향 그대로)
        m = o["mesh"].copy()
        m.apply_translation(-o["center_w"])
        o["file"] = str(mdir / f"obj_{o['id']}.obj")
        m.export(o["file"])

    import cutsim.scene.build as B
    rec = {}
    B.build_cut_scene = make_build(B, L, a, rec)
    sys.argv = ["01_cut_primitive.py", "--food_mesh", str(IMPORT / "normalized.obj"), "--knife_traj", str(traj),
                *PHYS_CUT_ARGS, "--out_root", str(out_root), "--tag", a.tag] + (["--no_video"] if a.no_video else [])
    t0 = time.time()
    runpy.run_path(str(ROOT / "scripts/01_cut_primitive.py"), run_name="__main__")
    save_kitchen(out, L, rec, time.time() - t0)


def make_build(B, L, a, rec):
    import genesis as gs
    from cutsim.assets.knife import write_knife_jig

    def set_mask(ent, ct, ca):
        for g in ent.geoms:
            g.desc.contype, g.desc.conaffinity = ct, ca

    def build(cfg, food_specs, record_dir=None, fps=25, with_food=True, view_scale=1.0, face_x=0.0,
              view_center=(0.0, 0.0), hold_boxes=(), grip=False):
        """cutsim.scene.build.build_cut_scene + 주방 물체·로봇·바닥(받침 평면은 입자 전용)"""
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
        off = float(bcfg.get("collision_offset_dx", 0.0)) / mcfg["grid_density"]
        sup = scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -off), visualization=False),
                               material=gs.materials.Rigid(coup_friction=bcfg["coup_friction"],
                                                           coup_softness=bcfg["coup_softness"]))
        set_mask(sup, SUP_T, SUP_A)
        floor = scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, L["z_floor_c"])), material=gs.materials.Rigid(needs_coup=False),
                                 surface=gs.surfaces.Default(color=(0.62, 0.62, 0.64)))
        set_mask(floor, FLOOR, OBJ | ROB | KNF)
        objs = {}
        for name, o in L["objs"].items():
            e = scene.add_entity(gs.morphs.Mesh(file=o["file"], pos=tuple(float(v) for v in o["pos_c"]), quat=o["quat_c"],
                                                convexify=True, decompose_object_error_threshold=float("inf")),
                                 material=gs.materials.Rigid(needs_coup=False, rho=float(o["rho"])),
                                 surface=gs.surfaces.Default(vis_mode="visual"))
            set_mask(e, OBJ, FLOOR | OBJ | ROB | KNF)
            objs[name] = e
        Tb = L["T_cw"] @ L["T_base_w"]
        robot = scene.add_entity(gs.morphs.URDF(file=str(URDF), fixed=True, merge_fixed_links=False,
                                                pos=tuple(float(v) for v in Tb[:3, 3]), quat=wxyz(Tb[:3, :3])),
                                 material=gs.materials.Rigid(needs_coup=False),
                                 surface=gs.surfaces.Default(color=(0.88, 0.88, 0.86)))
        set_mask(robot, ROB, FLOOR | OBJ)
        urdf = write_knife_jig(thickness=kcfg["thickness"], length=kcfg["length"], height=kcfg["height"],
                               bevel_height=kcfg["bevel_height"], blade_mass=kcfg["blade_mass"],
                               edge_width=kcfg.get("edge_width", 0.0002))
        # Genesis 화면에서도 집게가 쥔 손잡이가 보이게: 손잡이(render_surface.knife_meshes 와 같은 자리·크기)를 보이기만 하는 형상으로
        Lk, Hk = kcfg["length"], kcfg["height"]
        hv = (f'<visual><origin xyz="0 {Lk / 2 + 0.042:.4f} {Hk - 0.0125:.4f}"/><geometry><box size="0.015 0.09 0.021"/>'
              f'</geometry></visual>\n    <collision>')
        uh = Path(urdf).with_name(Path(urdf).stem + "_handle.urdf")
        tmp = uh.with_name(f"{uh.stem}.{os.getpid()}.urdf")
        tmp.write_text(Path(urdf).read_text().replace("<collision>", hv, 1))
        os.replace(tmp, uh)
        urdf = uh
        knife = scene.add_entity(
            gs.morphs.URDF(file=str(urdf), fixed=True, merge_fixed_links=False, convexify=False),
            material=gs.materials.Rigid(coup_friction=kcfg["coup_friction"], coup_softness=kcfg["coup_softness"],
                                        sdf_cell_size=kcfg["sdf_cell_size"], sdf_max_res=kcfg["sdf_max_res"],
                                        gravity_compensation=1.0),
            surface=gs.surfaces.Default(color=(0.75, 0.78, 0.82)),
        )
        set_mask(knife, KNF, FLOOR | OBJ)
        holds = []
        for center, size in hold_boxes:          # --grip: 표시만(충돌·결합 없음), 붙잡기는 입자 구속
            holds.append((scene.add_entity(gs.morphs.Box(pos=tuple(center), size=tuple(size), fixed=True, collision=False),
                                           material=gs.materials.Rigid(needs_coup=False),
                                           surface=gs.surfaces.Default(color=(0.86, 0.68, 0.58))),
                          np.asarray(center, float), np.asarray(size, float)))
        foods = {}
        if with_food:
            for spec in food_specs:
                m = spec.mat
                foods[spec.name] = scene.add_entity(
                    spec.morph,
                    material=gs.materials.MPM.ElastoPlastic(E=m["E"], nu=m["nu"], rho=m["rho"],
                                                           von_mises_yield_stress=m["yield_stress"]),
                    surface=gs.surfaces.Default(color=tuple(m["color"]), vis_mode=spec.vis_mode))
        c, k = cfg["camera"], view_scale
        cx, cy = view_center
        o_ = np.array([cx, cy, 0.0])
        Tc = L["T_cw"]
        cam_specs = {
            "persp": (np.array(c["pos"]) * k + o_, np.array(c["lookat"]) * k + o_),
            "side": ((cx, cy - 0.17 * k, 0.02 * k), (cx, cy, 0.015 * k)),
            "top": ((cx, cy - 0.001, 0.2 * k), (cx, cy, 0.0)),
            # Genesis 화면 그대로(입자 = 구)로 주방 전체: 모든 물체가 같은 시뮬레이션 안에 있다는 확인용
            "scene": ((Tc @ [0.62, -0.50, 0.40, 1])[:3], (Tc @ [0.17, 0.02, 0.05, 1])[:3]),
        }
        cam_specs = {n: {"pos": [float(v) for v in p], "lookat": [float(v) for v in l], "fov": float(c["fov"]) if n != "scene" else 42.0,
                         "res": [int(v) for v in c["res"]]} for n, (p, l) in cam_specs.items()}
        cams = {}
        if record_dir is not None:
            for n, sp in cam_specs.items():
                cams[n] = scene.add_camera(res=tuple(sp["res"]), pos=tuple(sp["pos"]), lookat=tuple(sp["lookat"]),
                                           fov=sp["fov"], GUI=False)
        t_b = time.time()
        scene.build()
        print(f"장면 만들기 {time.time() - t_b:.0f}s", flush=True)
        dofs_idx = [knife.get_joint(n).dofs_idx_local[0] for n in B.KNIFE_DOFS]
        knife.set_dofs_kp(np.array(kcfg["kp"], dtype=np.float32), dofs_idx)
        knife.set_dofs_kv(np.array(kcfg["kv"], dtype=np.float32), dofs_idx)
        cs = B.CutScene(scene, knife, foods, cams, cfg, dofs_idx)
        cs.cam_specs, cs.holds, cs.board_offset = cam_specs, holds, off
        if record_dir is not None:
            import genesis.utils.video_encoder as venc
            pref = os.environ.get("CUTSIM_VIDEO_CODEC", "libx264")
            venc.H264_CODEC_CANDIDATES = (pref,) + tuple(x for x in venc.H264_CODEC_CANDIDATES if x != pref)
            Path(record_dir).mkdir(parents=True, exist_ok=True)
            for n, cam in cams.items():
                cam.start_recording(save_to_filename=str(Path(record_dir) / f"{n}.mp4"), fps=fps)
        if with_food:
            attach_robot(cs, robot, objs, L, a, rec, scfg["step_dt"])
        return cs

    return build


def attach_robot(cs, robot, objs, L, a, rec, dt):
    """매 스텝 전에 실제 칼 자세로 IK → 로봇 관절 목표, 25 fps 마다 물체·로봇 자세 기록(01 의 frames.npz 와 같은 스텝)."""
    rb = Robot(T_base=L["T_base_w"])
    arm = [robot.get_joint(n).dofs_idx_local[0] for n in ARM]
    grp = [robot.get_joint(n).dofs_idx_local[0] for n in (GRIP, *JAWS)]
    q8 = Robot.grip_angle(GRIP_GAP_MM)
    mim = [rb.joints[n]["mimic"][1] * q8 for n in JAWS]
    q_g = np.array([q8, *mim])
    idx = arm + grp
    every = max(1, int(round(1.0 / (25.0 * dt))))
    # 관절 PD: 관절 공간 관성(대각)으로 고유 진동수 kp_hz, 임계 감쇠. 힘 상한 = 모터 정지 토크 2.94 N·m(URDF effort)
    M = robot.get_mass_mat().cpu().numpy()
    Md = np.diag(M)[idx]
    w = 2 * np.pi * a.kp_hz
    robot.set_dofs_kp((Md * w * w).astype(np.float32), idx)
    robot.set_dofs_kv((2 * Md * w).astype(np.float32), idx)
    robot.set_dofs_force_range(np.full(len(idx), -2.94, np.float32), np.full(len(idx), 2.94, np.float32), idx)
    links = [l.name for l in robot.links]
    gb = links.index("Gripper_Base_v1_1")
    T_cw = L["T_cw"]
    st = dict(n=0, q=None, t_ik=0.0)
    rec.update(link_names=links, obj_names=list(objs), dof_names=list(ARM) + [GRIP, *JAWS], every=every, step=[],
               links_pos=[], links_quat=[], q=[], q_cmd=[], obj_pos=[], obj_quat=[], tau=[], kp=(Md * w * w).tolist(),
               track_mm=[], step10=[], ik_mm=[], ik_deg=[], sleep_at={}, wake=[])
    ents = list(objs.items())
    zz = dict(asleep=[False] * len(ents), calm=[0] * len(ents), qpos=[None] * len(ents))
    orig = cs.scene.step

    def step(*args, **kw):
        qk = cs.knife_q()
        Tg_w = L["T_wc"] @ knife_T(qk) @ L["G"]
        t1 = time.time()
        if st["q"] is None:                       # 첫 스텝: 여러 출발점 IK 로 시작 자세를 잡고 그 자리에 놓는다
            qs, _ = solve_ik_path(rb, [Tg_w])
            st["q"] = qs[0]
            robot.set_dofs_position(np.r_[st["q"], q_g].astype(np.float32), idx)
            robot.set_dofs_velocity(np.zeros(len(idx), np.float32), idx)
        q, e_mm, e_deg = rb.ik(Tg_w, st["q"])
        st["t_ik"] += time.time() - t1
        qd = (q - st["q"]) / dt
        st["q"] = q
        robot.control_dofs_position_velocity(np.r_[q, q_g].astype(np.float32),
                                             np.r_[qd, np.zeros(3)].astype(np.float32), idx)
        rec["ik_mm"].append(e_mm); rec["ik_deg"].append(e_deg)
        out = orig(*args, **kw)
        i = st["n"]
        st["n"] += 1
        for j, (nm, e) in enumerate(ents):        # 잠재우기 / 깨우기
            v = e.get_dofs_velocity().cpu().numpy()
            lin, ang = float(np.linalg.norm(v[:3])), float(np.degrees(np.linalg.norm(v[3:])))
            if zz["asleep"][j]:
                if lin > 0.020 or ang > 20.0:
                    zz["asleep"][j], zz["calm"][j] = False, 0
                    rec["wake"].append({"step": i, "object": nm, "v_mm_s": round(lin * 1000, 1), "w_deg_s": round(ang, 1)})
                    print(f"    {nm} 깨어남(step {i}, {lin * 1000:.0f} mm/s, {ang:.0f}°/s)", flush=True)
                else:
                    e.set_qpos(zz["qpos"][j], zero_velocity=True)
            else:
                zz["calm"][j] = zz["calm"][j] + 1 if (i >= 50 and lin < 0.003 and ang < 3.0) else 0
                if zz["calm"][j] >= 30:
                    zz["asleep"][j], zz["qpos"][j] = True, e.get_qpos().clone()
                    rec["sleep_at"].setdefault(nm, []).append(i)
        if i % 10 == 0:                           # 집게 TCP 가 목표(손잡이)에서 얼마나 떨어졌나(10 스텝마다)
            p = robot.get_links_pos()[gb].cpu().numpy()
            qq = robot.get_links_quat()[gb].cpu().numpy()
            R = Rotation.from_quat([qq[1], qq[2], qq[3], qq[0]]).as_matrix()
            tcp_c = R @ TCP_L + p
            tgt_c = (T_cw @ Tg_w)[:3, 3]
            rec["track_mm"].append(float(np.linalg.norm(tcp_c - tgt_c) * 1000))
            rec["tau"].append(robot.get_dofs_control_force(idx).cpu().numpy())
            rec["step10"].append(i)
        if i % every == 0:
            rec["step"].append(i)
            rec["links_pos"].append(robot.get_links_pos().cpu().numpy())
            rec["links_quat"].append(robot.get_links_quat().cpu().numpy())
            rec["q"].append(robot.get_dofs_position(idx).cpu().numpy())
            rec["q_cmd"].append(np.r_[q, q_g])
            rec["obj_pos"].append(np.stack([e.get_links_pos()[0].cpu().numpy() for e in objs.values()]))   # 메쉬 원점(놓은 자리)
            rec["obj_quat"].append(np.stack([e.get_links_quat()[0].cpu().numpy() for e in objs.values()]))
        if i % 500 == 0:
            print(f"    로봇: IK {st['t_ik'] / (i + 1) * 1000:.1f} ms/스텝, 집게 추종 오차 {rec['track_mm'][-1]:.2f} mm", flush=True)
        return out

    cs.scene.step = step
    rec["state"] = st


def save_kitchen(out, L, rec, wall):
    names = rec["obj_names"]
    P, Qt = np.stack(rec["obj_pos"]), np.stack(rec["obj_quat"])
    step = np.array(rec["step"])
    dt = 0.001
    ref = int(np.searchsorted(step * dt, 0.15))   # settle(0.15 s) 끝 = 기준
    summ = {"wall_s_total": round(wall, 1), "ik_ms_per_step": round(rec["state"]["t_ik"] / max(1, rec["state"]["n"]) * 1000, 2),
            "T_wc": L["T_wc"].tolist(), "carrot_dz_mm": L["dz"] * 1000, "objects": {}}
    for j, n in enumerate(names):
        d0 = np.linalg.norm(P[ref, j] - P[0, j]) * 1000
        dmax = np.linalg.norm(P[ref:, j] - P[ref, j], axis=1).max() * 1000
        r = Rotation.from_quat(Qt[ref:, j][:, [1, 2, 3, 0]])
        r0 = Rotation.from_quat(Qt[ref, j][[1, 2, 3, 0]])
        amax = float(np.degrees((r0.inv() * r).magnitude()).max())
        summ["objects"][n] = {"mass_kg": MASS_KG[n], "settle_move_mm": round(float(d0), 2),
                              "max_move_after_settle_mm": round(float(dmax), 2), "max_rot_after_settle_deg": round(amax, 2)}
    tau = np.abs(np.stack(rec["tau"]))
    summ["sleeping"] = {"slept_at_step": rec["sleep_at"], "woke": rec["wake"]}
    tr = np.array(rec["track_mm"]); s10 = np.array(rec["step10"])
    summ["robot"] = {"track_err_mm": {"max": round(float(tr.max()), 2), "mean": round(float(tr.mean()), 2),
                                      "max_after_0.2s": round(float(tr[s10 * dt >= 0.2].max()), 2) if (s10 * dt >= 0.2).any() else None},
                     "ik_err_max": {"mm": round(max(rec["ik_mm"]), 2), "deg": round(max(rec["ik_deg"]), 2)},
                     "max_torque_ratio": {n: round(float(v) / 2.94, 2) for n, v in zip(rec["dof_names"], tau.max(0))},
                     "max_torque_ratio_after_0.2s": {n: round(float(v) / 2.94, 2) for n, v in
                                                     zip(rec["dof_names"], tau[s10 * dt >= 0.2].max(0))} if (s10 * dt >= 0.2).any() else None,
                     "kp": dict(zip(rec["dof_names"], [round(v, 2) for v in rec["kp"]]))}
    np.savez_compressed(out / "kitchen_frames.npz", step=step, t=step * dt, links_pos=np.stack(rec["links_pos"]),
                        links_quat=np.stack(rec["links_quat"]), q=np.stack(rec["q"]), q_cmd=np.stack(rec["q_cmd"]),
                        obj_pos=P, obj_quat=Qt, T_wc=L["T_wc"], step10=s10, tau10=np.stack(rec["tau"]), track10=tr, meta=json.dumps(
                            {"link_names": rec["link_names"], "obj_names": names, "dof_names": rec["dof_names"],
                             "obj_files": {n: L["objs"][n]["file"] for n in names},
                             "obj_center_w": {n: L["objs"][n]["center_w"].tolist() for n in names}}, ensure_ascii=False))
    (out / "kitchen.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summ.items() if k != "T_wc"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
