"""복원 주방의 당근을 ElRobot 이 칼을 쥐고 두 번 써는 시나리오를 만든다(칼 궤적 + 몸통 고정 상자) — 로봇이 닿는지(IK)도 확인.

식재료 = data/recon/kitchen_recon_carrot(가상 카메라 360° 복원 당근, 원래 메쉬 안 씀). 칼 궤적은 사람 손 대신 로봇 손:
위에서 내려와 칼날 길이 방향으로 앞뒤 톱질하며 도마까지(로봇이 누를 수 있는 힘이 약 12 N 이라 누르기만으로는 안 잘린다),
로봇 쪽 끝에서 2.0 cm·3.4 cm 자리를 썬다. 칼날 면 법선 = 당근 긴 축(가로로 썰기).
로봇 손: 손잡이 가운데를 위에서 쥔다(집게 접근 = 아래, 닫힘 = 칼날 면 법선). 집게가 옆으로 누우면 집게 몸통(닫힘 방향
±4 cm)이 도마를 뚫어서 위에서 쥔다. 손잡이 모양은 render_surface.knife_meshes 와 같다.
python scripts/twin/make_robot_cut.py
결과: data/recon/kitchen_robot_cut_carrot/{mesh.obj, meta.yaml, traj_knife.npz}
"""
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from cutsim.io.recon_loader import load_recon
from meshfix import smooth_keep_volume
from robot_kin import Robot

spec = importlib.util.spec_from_file_location("demos", ROOT / "scripts/07_make_demos.py")
demos = importlib.util.module_from_spec(spec); spec.loader.exec_module(demos)
Hand, top_near, FPS = demos.Hand, demos.top_near, demos.FPS

SRC, DST = ROOT / "data/recon/kitchen_recon_carrot", ROOT / "data/recon/kitchen_robot_cut_carrot"
KN = dict(length=0.08, height=0.035)                # configs/scene_default.yaml 의 칼(당근 폭 + 4cm < 8cm 라 그대로)
T_BASE = np.eye(4); T_BASE[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]   # 로봇 원점, 팔 앞(+y_base) = 세계 +x


def knife_grip(alpha=0.0, s=0.0, L=KN["length"], H=KN["height"]):
    """칼 좌표(원점 날 끝 가운데, x 법선, y 길이, z 칼등) → 집게 좌표(원점 TCP, x 닫힘, y 접근).
    기본은 손잡이 가운데를 위에서 쥔 것. 막대 모양 손잡이라 닫힘 축(칼 x) 둘레로 alpha 만큼 기울여 쥐거나 손잡이를 따라
    s 만큼 옮겨 쥐어도 같은 쥐기다(손잡이 길이 9 cm)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    T = np.eye(4)
    T[:3, :3] = Rx @ np.c_[[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    T[:3, 3] = [0.0, L / 2 + 0.042 + s, H - 0.0125]
    return T


def knife_T(q):
    """(x, y, z, yaw) → 4×4"""
    T = np.eye(4)
    c, s = np.cos(q[3]), np.sin(q[3])
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = q[:3]
    return T


def plan():
    DST.mkdir(parents=True, exist_ok=True)
    import trimesh
    m0 = trimesh.load(SRC / "mesh.obj", force="mesh", process=False)
    m1, shrink = smooth_keep_volume(m0, SMOOTH)        # 겉면 돌기 다듬기(입자 덩어리 떨어짐 방지), 부피는 되돌림
    m1.export(DST / "mesh.obj")
    print(f"메쉬 다듬기 Taubin {SMOOTH}번(다듬을 때 부피 {shrink * 100:+.1f}% → 되돌림)")
    meta = yaml.safe_load((SRC / "meta.yaml").read_text())
    meta["object_id"] = DST.name
    meta["mesh_fix"] = f"Taubin {SMOOTH}번 다듬고 부피 되돌림(복원 겉면 돌기 때문에 입자 덩어리가 떨어지는 것 방지)"
    (DST / "meta.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
    meta, mesh, warns, T_norm = load_recon(DST)       # 정규화 좌표: 당근 긴 축 = x(+x 가 로봇 쪽 끝), 바닥 가운데 원점
    lo, hi = mesh.bounds
    x1, x2 = hi[0] - 0.020, hi[0] - 0.034
    h = Hand([x1, 0.0, hi[2] + 0.05, 0.0, 0.0, 0.0])
    h.hold(0.3, "ready")
    for i, xc in enumerate((x1, x2)):
        zt = top_near(mesh, xc)
        h.move(0.6, f"approach{i}", x=xc, y=0.0, z=zt + 0.004)
        h.saw(v_down=0.02, amp=0.012, freq=1.4, phase=f"cut{i}", extra_T=1.0)
        h.hold(0.1, f"dwell{i}").move(0.4, f"lift{i}", z=zt + 0.015)
    h.move(0.6, "retreat", x=x1, z=hi[2] + 0.05).hold(0.3, "ready")
    t, q, ph = h.arrays(np.random.default_rng(0), {})
    q4 = q[:, :4]
    # 다른 손 대신 썰지 않는 쪽 끝(정규화 −x 쪽 4.7 cm)을 붙잡는 상자(07_make_demos 와 같은 방식, 그림에는 안 그림)
    a, b = lo[0] + 0.003, lo[0] + 0.050
    top = top_near(mesh, (a + b) / 2, half=(b - a) / 2) + 0.0005
    hold = [round(float(v), 4) for v in ((a + b) / 2, 0.0, top - 0.0013 + 0.0075, b - a, 0.05, 0.015)]
    T_w = np.linalg.inv(T_norm)
    Tk = np.stack([T_w @ knife_T(x) for x in q4])
    np.savez(DST / "traj_knife.npz", t=t, T_world_knife=Tk, fps=float(FPS), phase=ph)
    meta.update(hold_box=[hold], hold_box_note="썰지 않는 쪽 끝을 붙잡는 상자(정규화 좌표, m). 로봇 장면에서는 그리지 않음(고정구 대용)",
                task="ElRobot 이 칼 손잡이를 위에서 쥐고, 복원된 당근을 로봇 쪽 끝에서 2.0·3.4 cm 자리를 톱질로 두 번 썰기",
                knife_grip="집게 TCP = 손잡이 가운데, 접근 = 칼 −z(아래), 닫힘 = 칼 x(칼날 면 법선)")
    (DST / "meta.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
    print("궤적", f"{t[-1]:.2f}s", len(t), "점 | 썰 자리(정규화 x mm)", round(x1 * 1000, 1), round(x2 * 1000, 1), "| hold", hold)
    return T_w, q4, ph


def check_ik(T_w, q4, ph, dz=0.005, G=None, every=FPS // 25, quiet=False):
    """칼 자세마다 집게 목표 → IK(every 점마다). dz: 그림에서 도마 윗면에 맞추는 높이 차(대략). returns (오차, 관절들, 칼 자세들)"""
    rb = Robot(T_base=T_BASE)
    G = knife_grip() if G is None else G
    sel = np.arange(0, len(q4), every)
    D = np.eye(4); D[2, 3] = dz
    Tks = [D @ T_w @ knife_T(q4[i]) for i in sel]
    targets = [T @ G for T in Tks]
    best = None
    rng = np.random.default_rng(1)
    for k in range(12):                                # 첫 자세는 여러 출발점에서
        q0 = np.zeros(7) if k == 0 else rng.uniform(rb.lo, rb.hi)
        r = rb.ik(targets[0], q0)
        if best is None or r[1] + 0.05 * r[2] < best[1] + 0.05 * best[2]:
            best = r
    q, errs, jumps, qs = best[0], [], [], []
    for T in targets:
        qn, e_mm, e_deg = rb.ik(T, q)
        jumps.append(float(np.abs(qn - q).max())); errs.append((e_mm, e_deg)); q = qn; qs.append(qn)
    errs = np.array(errs)
    if not quiet:
        print(f"IK: 위치 오차 최대 {errs[:, 0].max():.2f} mm, 방향 오차 최대 {errs[:, 1].max():.2f}°, "
              f"한 프레임 관절 변화 최대 {np.degrees(max(jumps[1:])):.1f}° ({FPS // every} fps)")
    return errs, qs, Tks


def collisions(qs, Tks, carrot, board_top, dz=0.005):
    """집게·손목 메쉬 점이 도마 윗면 아래(도마 위), 당근(5 mm 여유), 칼날에 들어간 수. 손잡이를 쥔 손가락 끝은 뺀다."""
    import trimesh
    rb = Robot(T_base=T_BASE)
    parts = {n: m for n, m, _ in rb.meshes() if n in ("Gripper_Base_v1_1", "ST3215_8_v1_1", "Gripper_Gear_v1_1",
                                                     "Gripper_Jaw_01_v1_1", "Gripper_Jaw_02_v1_1", "ST3215_7_v1_1",
                                                     "Joint_06_v1_1", "ST3215_6_v1_1", "Joint_05_v1_1")}
    pts = {n: m.sample(300, seed=0) if hasattr(m, "sample") else m.vertices[::max(1, len(m.vertices) // 300)] for n, m in parts.items()}
    car = carrot.copy(); car.apply_translation([0, 0, dz])
    blo, bhi = np.array([0.148, -0.152]), np.array([0.352, 0.152])
    n_board = n_car = n_blade = 0
    for q, Tk in zip(qs, Tks):
        T = rb.fk(q, Robot.grip_angle(15.0))
        P = np.concatenate([(pts[n] @ T[n][:3, :3].T + T[n][:3, 3]) for n in pts])
        Pk = (P - Tk[:3, 3]) @ Tk[:3, :3]               # 칼 좌표
        handle = (np.abs(Pk[:, 0]) < 0.03) & (Pk[:, 1] > KN["length"] / 2 - 0.005) & (Pk[:, 2] > KN["height"] - 0.04)
        on_board = ((P[:, :2] >= blo) & (P[:, :2] <= bhi)).all(1)
        n_board += int((on_board & (P[:, 2] < board_top + dz + 0.003)).sum() + (P[:, 2] < 0.003).sum())
        sd = trimesh.proximity.signed_distance(car, P)    # + = 안
        n_car += int((sd > -0.005).sum())
        n_blade += int((~handle & (np.abs(Pk[:, 0]) < 0.004) & (np.abs(Pk[:, 1]) < KN["length"] / 2) & (Pk[:, 2] > -0.002)
                        & (Pk[:, 2] < KN["height"] + 0.002)).sum())
    return n_board, n_car, n_blade


def choose_grip(T_w, q4, ph):
    """고정 쥐기(기울기 alpha, 손잡이 위 자리 s, 손잡이 방향 side): IK 오차 + 충돌이 가장 작은 것."""
    import trimesh
    carrot = trimesh.load(DST / "mesh.obj", force="mesh")
    board_top = 0.0213                                  # 복원 도마 가운데 윗면(바닥 위 2.5 mm 올린 뒤)
    res = []
    for side in (0.0, np.pi):
        qs4 = q4.copy(); qs4[:, 3] += side
        for a in np.radians(np.arange(-90, 91, 15)):
            for s in (-0.03, 0.0, 0.03):
                e, qs, Tks = check_ik(T_w, qs4, ph, G=knife_grip(a, s), every=FPS // 4, quiet=True)
                if e[:, 0].max() > 3 or e[:, 1].max() > 3:
                    res.append((1e9, side, a, s, e[:, 0].max(), e[:, 1].max(), None))
                    continue
                c = collisions(qs, Tks, carrot, board_top)
                res.append((sum(c) + e[:, 0].max() + e[:, 1].max(), side, a, s, e[:, 0].max(), e[:, 1].max(), c))
    res.sort(key=lambda r: r[0])
    for r in res[:5]:
        print(f"  쥐기 후보 side={np.degrees(r[1]):.0f}° alpha={np.degrees(r[2]):.0f}° s={r[3] * 100:.0f}cm → "
              f"위치 {r[4]:.2f} mm, 방향 {r[5]:.2f}°, 충돌 점(도마, 당근, 칼날) {r[6]}")
    return res[0][1:4]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--grip", default=None, help="alpha_deg,s_m,side_deg (주면 쥐기 탐색을 건너뜀; 탐색 결과 45,0,0)")
    a_ = ap.parse_args()
    SMOOTH = a_.smooth
    T_w, q4, ph = plan()
    if a_.grip:
        a, s, side = (float(v) for v in a_.grip.split(","))
        a, side = np.radians(a), np.radians(side)
    else:
        side, a, s = choose_grip(T_w, q4, ph)
    q4[:, 3] += side
    np.savez(DST / "traj_knife.npz", t=np.arange(len(q4)) / FPS, T_world_knife=np.stack([T_w @ knife_T(x) for x in q4]),
             fps=float(FPS), phase=ph)
    meta = yaml.safe_load((DST / "meta.yaml").read_text())
    meta["robot_grip"] = {"alpha_deg": round(float(np.degrees(a)), 1), "s_m": float(s), "side_deg": round(float(np.degrees(side)), 1),
                          "T_base": T_BASE.tolist()}
    (DST / "meta.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
    check_ik(T_w, q4, ph, G=knife_grip(a, s))
