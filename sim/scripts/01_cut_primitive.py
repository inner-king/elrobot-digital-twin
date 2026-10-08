"""절단 테스트 T1~T4 와 칼 궤적 재생 TR.

T1 수직 누르기 / T2 톱질 / T3 분리 유지(자른 뒤 칼을 빼고, 90° 돌린 칼 옆면으로 한쪽 조각만 옆으로 밀기)
/ T4 연속 썰기 / TR 주어진 칼 궤적(사람 시연 등) 재생. 재료 없는 빈 동작을 먼저 돌려 기준선 힘을 빼고 절단력을 구한다.

python scripts/01_cut_primitive.py --test T1 [--speed 0.03] [--set mpm.grid_density=384] [--tag x]
python scripts/01_cut_primitive.py --test TR --knife_traj reports/03_import/<id>/traj_sim.npz --food_mesh ... (보통
       03_import_recon.py --cut 이 불러 준다)
결과: reports/01_cut/<test>[_tag]/{metrics.json, log.npz, force_depth.png, persp/side/top/face.mp4}
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cutsim.yamlio import loads

ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True, choices=["T1", "T2", "T3", "T4", "T3g", "TR"])
ap.add_argument("--knife_traj", default=None,
                help="TR: 칼 궤적 npz(t, pos (N,3) 날 끝 중심, rot (N,3,3)). 03_import_recon.py 의 traj_sim.npz")
ap.add_argument("--view_scale", type=float, default=None, help="카메라 거리 배율(기본: 물체 크기로 자동)")
ap.add_argument("--hold_box", action="append", default=[],
                help="cx,cy,cz,sx,sy,sz (m): 식재료를 위에서 눌러 붙잡는 고정 상자(다른 손 대용). 여러 번 가능")
ap.add_argument("--grip", action="store_true",
                help="--hold_box 를 누르는 상자 대신 쥐는 손으로: 상자 바닥 면적 안의 재료 입자를 구속으로 붙잡는다(상자는 표시만)")
ap.add_argument("--grip_omega", type=float, default=3000.0, help="쥐는 구속의 고유 진동수 rad/s(클수록 단단히)")
ap.add_argument("--sep_damp", type=float, default=0.0,
                help="분리 감쇠율 1/s: 조각 라벨을 붙인 순간부터 날 끝이 식재료 윗면 위로 빠져나간 뒤 --sep_damp_s 까지 "
                     "재료 속도를 감쇠한다. 격자 때문에 칼이 실제보다 넓게(약 한 칸) 밀어 쌓인 탄성이, 라벨로 연결부가 "
                     "끊길 때와 칼이 빠질 때 한꺼번에 풀려 조각을 튕겨 내는 것을 없앤다")
ap.add_argument("--sep_damp_s", type=float, default=0.06)
ap.add_argument("--sep_damp_r", type=float, default=0.0,
                help="감쇠를 칼날 면에서 이 거리(m) 안에 걸친 조각과 거기 맞닿은 조각에만, 조각 전체에 똑같이(0 이면 전부). "
                     "조각은 칼질마다 라벨이 뒤집힌 이력으로 구분한다. 떨어져서 넘어지는 조각은 그대로 둔다")
ap.add_argument("--save_frames", type=float, default=0.0,
                help="이 fps 로 입자 위치·조각 라벨·칼 자세를 frames.npz 에 저장(표면 렌더링용, 0 이면 안 함)")
ap.add_argument("--domain_pad", type=float, default=0.0,
                help="--food_mesh 때 MPM 영역 옆(x·y) 여유를 이만큼(m) 더 둔다(조각이 넘어지며 멀리 갈 때)")
ap.add_argument("--config", default="configs/scene_default.yaml")
ap.add_argument("--materials", default="configs/materials.yaml")
ap.add_argument("--set", action="append", default=[], help="설정 덮어쓰기 a.b=값")
ap.add_argument("--speed", type=float, default=0.03, help="하강 속도 m/s")
ap.add_argument("--saw_amp", type=float, default=0.01, help="T2 톱질 진폭 m")
ap.add_argument("--saw_freq", type=float, default=2.0, help="T2 톱질 주파수 Hz")
ap.add_argument("--pre_gap", type=float, default=0.006, help="T3g 두 토막 사이 틈 m")
ap.add_argument("--multifield", action="store_true", help="절단 완료 시 조각 라벨(two-field 패치)")
ap.add_argument("--mf_mu", type=float, default=0.0, help="조각 사이 접촉 마찰")
ap.add_argument("--mf_fill", type=float, default=0.0,
                help="조각 사이 접촉을 이 채움 비율 이상인 격자점에서만(패치 4단계). 0 이면 원래 동작(11.7mm 안이면 접촉)")
ap.add_argument("--food_mesh", default=None, help="정규화된 메쉬(m, z-up, 바닥 중심 원점). 주면 기본 도형 대신 사용")
ap.add_argument("--category", default=None, help="--food_mesh 의 materials.yaml 키")
ap.add_argument("--cut_x", type=float, default=0.0, help="T1~T3 절단면 x 위치(m)")
ap.add_argument("--inside_sep", action="store_true", help="칼 안쪽 격자점 분리(Genesis 패치 2단계)")
ap.add_argument("--cut_force", action="store_true", help="절단 저항 모델(G_c·L + τ·A)을 칼에 건다")
ap.add_argument("--force_every", type=int, default=2, help="절단 저항을 몇 스텝마다 다시 계산할지")
ap.add_argument("--z_force_limit", type=float, default=None, help="칼 z 관절 힘 상한 N(누르는 힘 한계)")
ap.add_argument("--skin_color", default=None, help="r,g,b (0~1) 껍질 색 덮어쓰기")
ap.add_argument("--flesh_color", default=None, help="r,g,b (0~1) 과육 색 덮어쓰기")
ap.add_argument("--t4_cuts", default="0.024,0.012,0.0,-0.012", help="T4 절단 위치들(m, 쉼표)")
ap.add_argument("--z_force_budget", type=float, default=None,
                help="누르는 힘 예산 N: 명령 동작의 수직 저항이 이보다 크면 칼을 더 내리지 않는다(톱질은 계속)")
ap.add_argument("--press_tail", type=float, default=0.0, help="하강 뒤 같은 동작을 더 이어 가는 시간 s")
ap.add_argument("--tag", default="")
ap.add_argument("--no_video", action="store_true")
ap.add_argument("--no_baseline", action="store_true")
ap.add_argument("--out_root", default="reports/01_cut")
args = ap.parse_args()

from cutsim.scene.build import (build_cut_scene, cfl_substeps, food_specs_from_mesh, load_yaml, particle_size,
                                primitive_mesh, stop_recording)
from cutsim.control.knife import KnifeTraj
from cutsim.metrics import cut as M

cfg = load_yaml(args.config)
for kv in args.set:
    key, val = kv.split("=", 1)
    d = cfg
    *path, last = key.split(".")
    for p in path:
        d = d[p]
    d[last] = loads(val)
materials = load_yaml(args.materials)
for _part, _arg in (("skin", args.skin_color), ("flesh", args.flesh_color)):
    if _arg:
        _c = [float(v) for v in _arg.split(",")]
        for _cat in materials.values():
            if isinstance(_cat, dict) and _part in _cat:
                _cat[_part]["color"] = _c
name = args.test + (f"_{args.tag}" if args.tag else "")
out = Path(args.out_root) / name
out.mkdir(parents=True, exist_ok=True)

import genesis as gs

gs.init(backend=gs.cuda, precision="32", logging_level="warning")

fcfg, kcfg = cfg["food"], cfg["knife"]
gd = cfg["mpm"]["grid_density"]
mesh = primitive_mesh(fcfg)
if args.food_mesh:
    import trimesh

    mesh = trimesh.load(args.food_mesh, force="mesh")
    fcfg["shape"], fcfg["category"] = "mesh", args.category or fcfg["category"]
    lo, hi = mesh.bounds
    # 물체에 맞춰 MPM 영역(옆·위 여유 + 아래 안전 여백)과 칼 길이를 잡는다
    pad = 3.5 / gd
    mx, my = 0.012 + args.domain_pad, 0.025 + args.domain_pad
    cfg["mpm"]["lower_bound"] = [float(lo[0]) - mx - pad, float(lo[1]) - my - pad, -0.03]
    cfg["mpm"]["upper_bound"] = [float(hi[0]) + mx + pad, float(hi[1]) + my + pad, float(hi[2]) + 0.02 + pad]
    kcfg_ = cfg["knife"]
    kcfg_["length"] = max(kcfg_["length"], float(hi[1] - lo[1]) + 0.04)
    fcfg["radius"] = float(max(-lo[1], hi[1]))  # T3 에서 칼을 놓을 y 위치 계산용
z_lift = 0.0005
if args.test == "T3g":  # 길이 절반짜리 토막 둘을 x=0 양쪽에 틈 pre_gap 으로 놓는다
    import trimesh
    half = dict(fcfg, length=(fcfg["length"] - args.pre_gap) / 2)
    hm = primitive_mesh(half)
    off = (half["length"] + args.pre_gap) / 2
    mesh = trimesh.util.concatenate([hm.copy().apply_translation([-off, 0, 0]),
                                     hm.copy().apply_translation([off, 0, 0])])
specs, layer_info = food_specs_from_mesh(mesh, fcfg["category"], materials, gd, fcfg["two_layer"],
                                         out / "food_mesh", z_lift=z_lift, vis_mode=fcfg.get("vis_mode", "particle"))
z_top = float(mesh.bounds[1, 2]) + z_lift
# 재료 강성에 맞춘 시간 간격(설정값보다 작아야 할 때만 늘림)과 식재료별 유효 칼 마찰
set_keys = {kv.split("=", 1)[0] for kv in args.set}
need = cfl_substeps([sp.mat for sp in specs], gd, cfg["sim"]["step_dt"])
if need > cfg["sim"]["substeps"] and "sim.substeps" not in set_keys:
    print(f"substeps {cfg['sim']['substeps']} → {need} (CFL)", flush=True)
    cfg["sim"]["substeps"] = need
cat_mat = materials.get(fcfg["category"], materials["default"])
if "knife_friction" in cat_mat and "knife.coup_friction" not in set_keys:
    kcfg["coup_friction"] = cat_mat["knife_friction"]
r = fcfg["radius"]
dt = cfg["sim"]["step_dt"]
z_clear = z_top + 0.008
z_min = kcfg["edge_min_z"]

# ---------------- 궤적 ----------------
cut_xs = [args.cut_x]
tr = KnifeTraj([args.cut_x, 0.0, z_clear, 0.0], dt)
tr.hold(0.15, "settle")
if args.test in ("T1", "T2", "T3"):
    tr.move_to(None, speed=0.05, phase="approach", z=z_top + 0.003)
    amp = args.saw_amp if args.test == "T2" else 0.0
    tr.press(z_min, args.speed, saw_amp=amp, saw_freq=args.saw_freq, phase="cut", tail_s=args.press_tail)
    tr.hold(0.1, "dwell")
    tr.move_to(None, speed=0.05, phase="withdraw", z=z_clear)
    tr.hold(0.3, "rest")
if args.test in ("T3", "T3g"):
    x_push = 0.008 + kcfg["length"] / 2  # 칼날이 x>=8mm 만 덮게: 오른쪽 조각만 민다
    y_start = -(r + 0.006)
    tr.move_to(None, speed=0.08, phase="reposition", x=x_push)
    tr.move_to(None, speed=0.08, phase="reposition", yaw=np.pi / 2)
    tr.move_to(None, speed=0.08, phase="reposition", y=y_start)
    tr.move_to(None, speed=0.05, phase="lower", z=0.002)
    tr.hold(0.05, "pre_push")
    tr.move_to(None, speed=0.02, phase="push", y=y_start + 0.02)
    tr.hold(0.15, "post_push")
if args.test == "T4":
    cut_xs = [float(v) for v in args.t4_cuts.split(",")]
    for i, cx in enumerate(cut_xs):
        tr.move_to(None, speed=0.08, phase=f"move{i}", x=cx)
        tr.move_to(None, speed=0.05, phase=f"approach{i}", z=z_top + 0.003)
        tr.press(z_min, args.speed, phase=f"cut{i}")
        tr.hold(0.05, f"dwell{i}")
        tr.move_to(None, speed=0.05, phase=f"withdraw{i}", z=z_clear)
    tr.hold(0.3, "rest")
view_c, face_x, tr_info = (0.0, 0.0), (cut_xs[-1] if args.test == "T4" else cut_xs[0]), None
if args.test == "TR":
    # 사람 칼 궤적 재생. 지그가 4자유도라 날 끝 중심 위치와 칼날 길이 방향(링크 y 축)의 수평 방위(yaw)만
    # 따라가고, 칼 기울기(앞뒤·옆)는 버린 뒤 그 크기를 기록한다.
    if not args.knife_traj:
        ap.error("--test TR 에는 --knife_traj 가 필요하다")
    d = np.load(args.knife_traj)
    t_src, pos_src, rot_src = (np.asarray(d[k], float) for k in ("t", "pos", "rot"))
    yaw_src = np.unwrap(np.arctan2(-rot_src[:, 0, 1], rot_src[:, 1, 1]))
    tilt_src = np.degrees(np.arccos(np.clip(rot_src[:, 2, 2], -1.0, 1.0)))
    t_new = np.arange(t_src[0], t_src[-1] + 0.5 * dt, dt)
    qr = np.column_stack([np.interp(t_new, t_src, v) for v in (*pos_src.T, yaw_src)])
    tilt = np.interp(t_new, t_src, tilt_src)
    below = qr[:, 2] < z_min
    qr[:, 2] = np.maximum(qr[:, 2], z_min)  # 도마 아래로는 명령하지 않는다
    if qr[0, 2] < z_top:
        print(f"경고: 궤적 시작에서 날 끝({qr[0, 2] * 1000:.1f}mm)이 식재료 윗면({z_top * 1000:.1f}mm)보다 낮다", flush=True)
    inside = qr[:, 2] < z_top  # 날 끝이 식재료 윗면보다 아래 = 자르는 중일 수 있는 구간(획)
    stroke = np.cumsum(np.diff(inside.astype(int), prepend=0) == 1)
    ph = [f"cut{s}" if a else f"air{s}" for a, s in zip(inside, stroke)]
    tr = KnifeTraj(qr[0], dt)
    tr.hold(0.15, "settle")
    tr.extend(qr, np.gradient(qr, dt, axis=0), ph)
    tr.hold(0.3, "rest")
    tr_info = {"traj": args.knife_traj, "duration_s": float(t_new[-1] - t_new[0]), "n_strokes": int(stroke.max()),
               "tilt_dropped_deg_max": float(tilt.max()), "tilt_dropped_deg_mean": float(tilt.mean()),
               "below_board_clamped_steps": int(below.sum())}
    if inside.any():
        cut_xs = [float(qr[inside, 0][0])]
        view_c = (float(qr[inside, 0].mean()), float(qr[inside, 1].mean()))
        face_x = float(qr[inside, 0][-1])
Q, QD, PH = tr.arrays()
print(f"[{name}] steps={len(Q)} sim_time={len(Q) * dt:.2f}s grid_density={gd} dx={1 / gd * 1000:.2f}mm "
      f"particle={particle_size(gd) * 1000:.2f}mm", flush=True)


def blade_contact_count(cs, q):
    """칼날 면 바로 옆(두께/2 + 입자 2개 안, 칼날 길이·높이 안)에 있는 입자 수: 칼이 재료를 지나왔는지 판정."""
    pos = np.concatenate(list(cs.particles().values()))
    rel = pos - q[:3]
    nrm = np.array([np.cos(q[3]), np.sin(q[3]), 0.0])
    tan = np.array([-np.sin(q[3]), np.cos(q[3]), 0.0])
    near = ((np.abs(rel @ nrm) < kcfg["thickness"] / 2 + 2 * particle_size(gd)) & (np.abs(rel @ tan) < kcfg["length"] / 2)
            & (rel[:, 2] < kcfg["height"]))
    return int(near.sum())


def damp_cluster(pos, piece_id, seeds, touch=None):
    """seeds 조각에서 시작해 맞닿은(입자 사이 touch 이내) 조각을 줄줄이 모은다. 감쇠를 1스텝마다 거는 사이에도
    접촉으로 운동량이 넘어가므로, 칼 옆 조각만 감쇠하면 거기 기댄 조각이 대신 튕겨 나간다."""
    from scipy.spatial import cKDTree

    touch = touch or 1.5 * particle_size(gd)
    ids = np.unique(piece_id)
    trees = {k: cKDTree(pos[piece_id == k]) for k in ids}
    out, todo = set(seeds), list(seeds)
    while todo:
        a = todo.pop()
        for b in ids:
            if b not in out and trees[a].query(pos[piece_id == b], distance_upper_bound=touch)[0].min() < touch:
                out.add(b)
                todo.append(b)
    return list(out)


def run(with_food, record_dir):
    view_scale = args.view_scale or max(1.0, float(mesh.extents.max()) / 0.07) ** 0.5
    holds = [(v[:3], v[3:]) for v in ([float(s) for s in h.split(",")] for h in args.hold_box)]
    cs = build_cut_scene(cfg, specs, record_dir=record_dir, with_food=with_food, view_scale=view_scale,
                         face_x=face_x, view_center=view_c, hold_boxes=holds, grip=args.grip)
    cs.set_knife(Q[0])
    frames = None
    if with_food and args.save_frames > 0:
        every = max(1, int(round(1.0 / (args.save_frames * dt))))
        frames = {"x": [], "lab": [], "step": [], "x_rest": np.concatenate(list(cs.particles().values())),
                  "mat": np.concatenate([np.full(e.n_particles, j, np.int8) for j, e in enumerate(cs.foods.values())]),
                  "names": list(cs.foods), "cams": cs.cam_specs, "board_offset": cs.board_offset,
                  "holds": [np.concatenate([c, s]).tolist() for _, c, s in cs.holds]}
    armed, marks, touch = True, [], 0
    damp_until, damping, damp_on = -1, False, False
    piece_id = np.zeros(sum(e.n_particles for e in cs.foods.values()) if with_food else 0, np.int64)
    n = len(Q)
    q_log, f_log = np.zeros((n, 4)), np.zeros((n, 4))
    snaps, snapsJ = {}, {}
    t0 = time.time()
    nan = False
    if with_food and args.multifield:
        cs.set_contact_friction(args.mf_mu)
        cs.set_multifield_fill(args.mf_fill)
    if with_food and args.inside_sep:
        cs.set_inside_sep(True)
    if args.z_force_limit:
        cs.set_knife_force_limit("z", args.z_force_limit)
    applier = None
    cf_log = np.zeros((len(Q), 7))  # L, A, F_cut, F_fric, Fx, Fy, Fz (모델이 칼에 건 힘)
    if with_food and args.cut_force:
        from cutsim.control.cut_force import BladeFrame, CutForceApplier, CutForceModel

        model = CutForceModel(G_c=cat_mat["cut_toughness_Npm"], tau=cat_mat["face_friction_Pa"])
        blade = BladeFrame(length=kcfg["length"], height=kcfg["height"], thickness=kcfg["thickness"])
        applier = CutForceApplier(cs, model, blade, particle_size(gd), every=args.force_every)
    z_eff = Q[0, 2]
    stall = np.zeros(n, bool)
    for i in range(n):
        q_cmd, qd_cmd = Q[i].copy(), QD[i].copy()
        if args.z_force_budget is not None:
            # 힘 예산 판정(준정적): 지금 칼이 가르는 길이·잠긴 면적과 명령 속도로 필요한 수직력을 계산해
            # 예산보다 크면 z 를 더 내리지 않는다. 뒤처졌으면 허용될 때 원래 하강 속도로 따라잡는다.
            # 위로 올리는 명령은 항상 따른다.
            if Q[i, 2] < z_eff - 1e-9:
                # 명령 궤적의 하강 속도로 내리되, 뒤처졌으면 최소 speed 로 따라잡는다
                dz = -min(z_eff - Q[i, 2], max(Q[i - 1, 2] - Q[i, 2] if i else 0.0, args.speed * dt))
                if applier is not None:
                    inf0 = applier.last[1]
                    yaw = Q[i, 3]
                    v_t = QD[i, 0] * -np.sin(yaw) + QD[i, 1] * np.cos(yaw)
                    need = applier.model.vertical_resistance(inf0["L"], inf0["A"], -dz / dt, v_t)
                    if need > args.z_force_budget:
                        dz, stall[i] = 0.0, True
            else:
                dz = Q[i, 2] - z_eff
            z_eff = z_eff + dz
            q_cmd[2] = z_eff
            qd_cmd[2] = dz / dt
        cs.command(q_cmd, qd_cmd)
        if applier is not None:
            F_m, inf = applier.update()
            cf_log[i] = [inf["L"], inf["A"], inf["F_cut"], inf["F_fric"], *F_m]
        cs.scene.step()
        if with_food and args.multifield and args.test == "TR":
            # 재생 궤적은 획이 어디서 끝나는지 모르므로, 날 끝이 도마 근처에 처음 닿는 순간 한 번 라벨을 붙이고
            # 날 끝이 1cm 넘게 다시 올라가야 다음 획을 받는다. 이번 획 동안(날 끝이 식재료 윗면 아래일 때 20스텝마다)
            # 칼날 옆에 재료가 한 번도 없었으면(허공·도마만 내리친 경우) 붙이지 않는다. 닿는 순간만 보면 칼 안쪽
            # 격자점 분리로 두 조각이 칼에서 5mm 넘게 밀려나 있을 때 놓친다.
            q = cs.knife_q()
            if armed and q[2] < z_top and i % 20 == 0:
                touch = max(touch, blade_contact_count(cs, q))
            if armed and q[2] <= z_min + 0.0015:
                armed = False
                n_touch = max(touch, blade_contact_count(cs, q))
                if n_touch >= 5:
                    flips = cs.mark_cut(point=(q[0], q[1], 0.0), normal=(np.cos(q[3]), np.sin(q[3]), 0.0))
                    piece_id |= np.concatenate(list(flips.values())).astype(np.int64) << len(marks)
                    damping = True
                    marks.append({"step": i, "phase": str(PH[i]), "x_mm": float(q[0] * 1000),
                                  "y_mm": float(q[1] * 1000), "yaw_deg": float(np.degrees(q[3]))})
                    print(f"  mark_cut at ({q[0] * 1000:.1f}, {q[1] * 1000:.1f})mm yaw {np.degrees(q[3]):.0f}° "
                          f"(step {i}, {PH[i]})", flush=True)
                else:
                    print(f"  날 끝이 도마에 닿았지만 칼날 옆 재료 입자 {n_touch}개: 라벨 안 붙임 (step {i})", flush=True)
            elif not armed and q[2] > z_min + 0.01:
                armed, touch = True, 0
        elif with_food and args.multifield and PH[i].startswith("cut") and (i == n - 1 or PH[i + 1] != PH[i]):
            q = cs.knife_q()
            if q[2] <= z_min + 0.0015:  # 날 끝이 도마 근처까지 내려간 경우만 끝까지 자른 것으로 본다
                yaw = q[3]
                cs.mark_cut(point=(q[0], q[1], 0.0), normal=(np.cos(yaw), np.sin(yaw), 0.0))
                damping = True
                print(f"  mark_cut at x={q[0] * 1000:.1f}mm (step {i}, {PH[i]})", flush=True)
            else:
                print(f"  절단 미완료(날 끝 {q[2] * 1000:.1f}mm): 조각 라벨 안 붙임 (step {i}, {PH[i]})", flush=True)
        if args.sep_damp > 0 and (damping or i <= damp_until):
            # 감쇠는 Genesis 안에서 서브스텝마다 속도와 회전(APIC C)에 함께 건다(패치 5단계)
            qk = cs.knife_q()
            sel = np.ones(len(piece_id), bool)
            if args.sep_damp_r > 0:  # 칼날 면 가까이에 걸친 조각과, 그 조각에 (줄줄이) 맞닿은 조각 전체
                pos = np.concatenate(list(cs.particles().values()))
                near = np.abs((pos - qk[:3]) @ np.array([np.cos(qk[3]), np.sin(qk[3]), 0.0])) < args.sep_damp_r
                sel = np.isin(piece_id, damp_cluster(pos, piece_id, set(np.unique(piece_id[near]).tolist())))
            cs.set_damping(np.where(sel, args.sep_damp, 0.0))
            damp_on = True
            if damping and qk[2] > z_top:  # 칼이 빠져나갔으면 조금 더 감쇠하고 끝
                damping, damp_until = False, i + int(round(args.sep_damp_s / dt))
        elif damp_on:
            cs.set_damping(np.zeros(len(piece_id)))
            damp_on = False
        if with_food and args.grip and PH[i] == "settle" and i + 1 < n and PH[i + 1] != "settle":
            # 다 내려앉은 뒤에 쥔다(처음 놓인 높이로 붙잡지 않게)
            print(f"  grip: 입자 {cs.grip_food(args.grip_omega)}개를 손 상자에 붙임 (step {i})", flush=True)
        if frames is not None and i % every == 0:
            frames["x"].append(np.concatenate(list(cs.particles().values())).astype(np.float32))
            frames["lab"].append(np.concatenate([cs.parity.get(k, np.zeros(e.n_particles, np.int32))
                                                 for k, e in cs.foods.items()]).astype(np.int8))
            frames["step"].append(i)
        q_log[i] = cs.knife_q()
        f_log[i] = cs.knife_ctrl_force()
        last_of_phase = i == n - 1 or PH[i + 1] != PH[i]
        if with_food and last_of_phase:
            pos = np.concatenate(list(cs.particles().values()))
            snaps[PH[i]] = pos
            snapsJ[PH[i]] = np.concatenate(list(cs.particle_J().values()))
            if not np.isfinite(pos).all():
                nan = True
                print(f"  NaN at step {i} ({PH[i]})", flush=True)
                break
        if i % 500 == 0:
            print(f"  step {i}/{n} {PH[i]} z={q_log[i, 2] * 1000:.1f}mm Fz={f_log[i, 2]:.2f}N "
                  f"{(time.time() - t0) / (i + 1) * 1000:.1f}ms/step", flush=True)
    wall = time.time() - t0
    if record_dir is not None:
        stop_recording(cs)
    sizes = {k: int(e.n_particles) for k, e in cs.foods.items()}
    return dict(q=q_log, f=f_log, snaps=snaps, snapsJ=snapsJ, wall=wall, nan=nan, sizes=sizes, cf=cf_log, stall=stall,
                marks=marks, parity={k: v.copy() for k, v in cs.parity.items()}, frames=frames)


base = None
if not args.no_baseline:
    # 빈 동작 기준선은 재료와 무관하므로 궤적·칼·제어·시간 간격·힘 상한이 같으면 재사용한다.
    # 빈 동작에는 닿는 것이 없어 칼을 평행이동해도 같으므로 궤적은 시작점 기준 상대값으로 비교한다.
    import hashlib

    key = hashlib.md5((Q - Q[0]).tobytes() + json.dumps([cfg["knife"], cfg["sim"], args.z_force_limit],
                                                         sort_keys=True, default=str).encode()).hexdigest()[:12]
    base_path = Path(args.out_root) / "_baseline_cache" / f"{key}.npz"
    if base_path.exists():
        base = {"f": np.load(base_path)["f"]}
        print(f"baseline 재사용 {base_path.name}", flush=True)
    else:
        print("baseline (빈 동작)...", flush=True)
        base = run(False, None)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = base_path.with_name(f"{key}.{os.getpid()}.npz")  # 동시에 도는 실행끼리 반쯤 쓴 파일을 읽지 않게
        np.savez(tmp, f=base["f"])
        os.replace(tmp, base_path)
print("main run...", flush=True)
main = run(True, None if args.no_video else out)

# ---------------- 지표 ----------------
import torch

free, total = torch.cuda.mem_get_info()
res = {"test": args.test, "tag": args.tag, "multifield": args.multifield, "mf_mu": args.mf_mu, "mf_fill": args.mf_fill, "grid_density": gd, "dx_mm": 1000 / gd,
       "particle_mm": particle_size(gd) * 1000, "speed_mps": args.speed, "n_steps": len(Q),
       "sim_time_s": len(Q) * dt, "wall_s": round(main["wall"], 1),
       "wall_ms_per_step": round(main["wall"] / len(Q) * 1000, 2), "substeps": cfg["sim"]["substeps"],
       "vram_used_gb_device": round((total - free) / 1e9, 2), "particles": main["sizes"], "nan": main["nan"],
       "knife": {k: kcfg.get(k, 0.0002 if k == "edge_width" else None)
                 for k in ("thickness", "length", "bevel_height", "edge_width", "coup_friction", "coup_softness")},
       "material": cat_mat["flesh"], "two_layer": fcfg["two_layer"], "layer_info": layer_info}
if args.test == "T2":
    res["saw"] = {"amp_m": args.saw_amp, "freq_hz": args.saw_freq}

S = main["snaps"]
pos0 = S["settle"]
labels = M.piece_labels(pos0, cut_xs)
spacing = M.nn_spacing(pos0)
res["nn_spacing_mm"] = spacing * 1000
layer = 1.5 * spacing
fz = main["f"][:, 2] - (base["f"][:, 2] if base is not None else 0.0)
resist = -fz  # 위(+) 방향 저항력
z_edge = main["q"][:, 2]
res["min_edge_z_mm"] = float(z_edge.min() * 1000)
res["track_err_max_mm"] = float(np.abs(main["q"][:, :3] - Q[:, :3]).max() * 1000)
res["pass_penetrate"] = bool(not main["nan"] and z_edge.min() <= z_min + 0.001)

near = np.abs(pos0[:, 0] - cut_xs[0]) < 2 * spacing
if not near.any():
    near[:] = True
top0, bot0 = float(pos0[near, 2].max()), float(pos0[near, 2].min())
full_depth = top0 - bot0
depth = top0 - z_edge
cut_mask = np.char.startswith(PH, "cut")
res["food_top_mm"], res["full_depth_mm"] = top0 * 1000, full_depth * 1000
if args.test == "T3g":
    res["pre_gap_mm"] = args.pre_gap * 1000
    res["pre_gap_dx"] = args.pre_gap * gd
    res["pass_penetrate"] = None
if args.test in ("T1", "T2", "T3"):
    k = max(1, int(0.01 / dt))  # 10ms 이동 평균
    fs = M.smooth(resist, k)
    res["force"] = M.force_curve_checks(depth[cut_mask], fs[cut_mask], full_depth)
    in_mat = cut_mask & (depth > 0) & (depth < full_depth)
    res["mean_vertical_force_N"] = float(resist[in_mat].mean())
    res["work_J"] = float(np.sum(resist[cut_mask] * np.maximum(-np.diff(z_edge, prepend=z_edge[0])[cut_mask], 0)))
    # 칼을 뺀 직후(rest 끝)의 틈과 조각
    for ph in ("dwell", "rest"):
        res[f"gap_mm_{ph}"] = M.gap_width(S[ph], pos0, labels, cut_xs[0], 0, 1, layer) * 1000
    res["pass_gap"] = bool(res["gap_mm_rest"] <= 2 * kcfg["thickness"] * 1000)
    res["gap_rule"] = "틈(자르기 전 대비 맞닿은 입자층 거리 증가분) <= 칼날 두께 x2. 재결합은 T3 로 따로 잰다"
if args.test in ("T3", "T3g"):
    ratio, d_p, d_o, signed = M.follow_ratio(S["pre_push"], S["post_push"], labels, pushed=1, other=0)
    res["T3"] = {"follow_ratio": ratio, "follow_along_push": signed, "pushed_disp_mm": d_p * 1000,
                 "other_disp_mm": d_o * 1000,
                 "gap_mm_after_push": M.gap_width(S["post_push"], pos0, labels, cut_xs[0], 0, 1, layer) * 1000}
    res["pass_separation"] = bool(ratio <= 0.10)
if args.test == "T4":
    res["T4"] = {"gaps_mm_rest": [M.gap_width(S["rest"], pos0, labels, cx, j, j + 1, layer) * 1000
                                  for j, cx in enumerate(sorted(cut_xs))]}
    # 얇은 조각이 쓰러졌는지: 조각의 가장 얇은 주축이 x 축에서 얼마나 기울었나
    tilts = []
    for l in np.unique(labels):
        p = S["rest"][labels == l]
        w, v = np.linalg.eigh(np.cov((p - p.mean(0)).T))
        tilts.append(float(np.degrees(np.arccos(min(1.0, abs(v[0, 0]))))))
    res["T4"]["slice_tilt_deg"] = tilts
if args.test == "TR":
    # 획마다: 날 끝이 가장 낮게 내려간 높이, 도마까지 갔는지, 그동안의 저항력(모델 + 남는 MPM 힘)
    res["TR"] = dict(tr_info, marks=main["marks"], strokes=[])
    fs = M.smooth(resist, max(1, int(0.01 / dt)))
    for s in range(1, tr_info["n_strokes"] + 1):
        mk = PH == f"cut{s}"
        st = {"stroke": s, "t_s": [float(np.argmax(mk) * dt), float((len(mk) - np.argmax(mk[::-1])) * dt)],
              "min_edge_z_mm": float(z_edge[mk].min() * 1000), "reached_board": bool(z_edge[mk].min() <= z_min + 0.0015),
              "mean_resist_N": float(resist[mk].mean()), "max_resist_N": float(fs[mk].max())}
        if args.cut_force:
            st["max_model_force_N"] = float((main["cf"][mk, 2] + main["cf"][mk, 3]).max())
        if args.z_force_budget is not None:
            st["stall_steps"] = int(main["stall"][mk].sum())
        res["TR"]["strokes"].append(st)
res["expected_pieces"] = None if args.test == "TR" else len(cut_xs) + 1
final = S[PH[-1]]
for rad_mult in (1.25, 1.6):
    nc, sizes = M.n_components(final, rad_mult * spacing)
    res[f"components_r{rad_mult}"] = {"n": nc, "sizes": sizes}
res["pass_pieces"] = None if args.test == "TR" else res["components_r1.25"]["n"] == res["expected_pieces"]
J0, J1 = main["snapsJ"]["settle"], main["snapsJ"][PH[-1]]
res["volume_change_pct"] = (M.volume_change(J1) - M.volume_change(J0)) * 100
res["pass_volume"] = bool(abs(res["volume_change_pct"]) <= 2.0)

if args.cut_force:
    cf = main["cf"]
    res["cut_force_model"] = {"G_c_Npm": cat_mat["cut_toughness_Npm"], "tau_Pa": cat_mat["face_friction_Pa"],
                              "max_L_mm": float(cf[:, 0].max() * 1000), "max_A_cm2": float(cf[:, 1].max() * 1e4),
                              "max_F_cut_N": float(cf[:, 2].max()), "max_F_fric_N": float(cf[:, 3].max())}
res["flags"] = {"inside_sep": args.inside_sep, "cut_force": args.cut_force, "z_force_limit": args.z_force_limit,
                "hold_box": args.hold_box, "grip": args.grip and bool(args.hold_box),
                "sep_damp": [args.sep_damp, args.sep_damp_s, args.sep_damp_r] if args.sep_damp > 0 else None,
                "board_collision_offset_dx": cfg["board"].get("collision_offset_dx", 0.0),
                "z_force_budget": args.z_force_budget, "cut_x": args.cut_x}
if args.z_force_budget is not None:
    res["stall_steps"] = int(main["stall"].sum())
    res["stall_first_depth_mm"] = (float((res["food_top_mm"] / 1000 - main["q"][np.argmax(main["stall"]), 2]) * 1000)
                                   if main["stall"].any() else None)
np.savez_compressed(out / "log.npz", stall=main["stall"], cf=main["cf"], q=main["q"], f=main["f"], f_base=base["f"] if base else 0, Q=Q, QD=QD,
                    phase=PH, labels=labels, **{f"pos_{k}": v for k, v in S.items()},
                    **{f"parity_{k}": v for k, v in main["parity"].items()})
(out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
if main["frames"] is not None:
    fr = main["frames"]
    st = np.array(fr["step"])
    meta = {"names": fr["names"], "cams": fr["cams"], "holds": fr["holds"], "board_offset": fr["board_offset"],
            "fps": args.save_frames, "dt": dt, "particle_size": particle_size(gd), "dx": 1.0 / gd,
            "knife": {k: kcfg[k] for k in ("thickness", "length", "height", "bevel_height")} | {
                "edge_width": kcfg.get("edge_width", 0.0002)}, "category": fcfg["category"],
            "colors": {k: materials.get(fcfg["category"], materials["default"])[k]["color"] for k in fr["names"]},
            "food_mesh": args.food_mesh, "z_lift": z_lift, "marks": main["marks"]}
    np.savez_compressed(out / "frames.npz", x=np.stack(fr["x"]), lab=np.stack(fr["lab"]), step=st, t=st * dt,
                        q=main["q"][st], phase=PH[st], x_rest=fr["x_rest"].astype(np.float32), mat=fr["mat"],
                        meta=json.dumps(meta, default=float))

# ---------------- 그림 ----------------
from cutsim.plotting import plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
t = np.arange(len(Q)) * dt
ax[0].plot(t, resist, lw=0.5, alpha=0.4, label="raw")
ax[0].plot(t, M.smooth(resist, max(1, int(0.01 / dt))), lw=1.2, label="10ms avg")
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("resistance Fz (N)"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax0b = ax[0].twinx(); ax0b.plot(t, z_edge * 1000, "k--", lw=0.8); ax0b.set_ylabel("edge z (mm)")
if args.test == "TR":  # 획이 여러 번이라 깊이 대신 시간 축: 모델이 칼에 건 힘과 힘 예산에 막힌 순간
    cf = main["cf"]
    ax[1].plot(t, cf[:, 2], lw=0.8, label="model F_cut (G_c·L)")
    ax[1].plot(t, cf[:, 3], lw=0.8, label="model F_fric (τ·A)")
    if main["stall"].any():
        ax[1].plot(t[main["stall"]], np.zeros(int(main["stall"].sum())), "r|", ms=12, label="stalled (budget)")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("force (N)"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
else:
    if cut_mask.any():
        ax[1].plot(depth[cut_mask] * 1000, M.smooth(resist, max(1, int(0.01 / dt)))[cut_mask])
        ax[1].axvline(full_depth * 1000, color="r", ls=":", label="board")
    ax[1].set_xlabel("depth (mm)"); ax[1].set_ylabel("resistance (N)"); ax[1].grid(alpha=0.3); ax[1].legend()
fig.suptitle(f"{name}  gd={gd}  v={args.speed}m/s")
fig.tight_layout(); fig.savefig(out / "force_depth.png", dpi=110)
print(json.dumps({k: v for k, v in res.items() if k not in ("layer_info", "material")}, indent=1,
                 ensure_ascii=False, default=float))
