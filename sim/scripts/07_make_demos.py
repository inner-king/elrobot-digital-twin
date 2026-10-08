"""KIT 대신 쓸 합성 시연 데이터: 식재료 메쉬 + 사람이 써는 것처럼 움직이는 칼 궤적.

KIT 계정이 열리기 전에 "복원 폴더 → 임포트 → 칼 궤적 재생 → 영상" 파이프라인을 끝까지 돌려 보려고 만든다.
식재료는 Chop & Learn 사진으로 만든 형상(data/recon/cnl_*, 05_food_from_photos.py), 칼 궤적은 사람 동작을 흉내 낸
것이다(최소 저크 이동, 앞뒤 톱질, 손떨림, 획마다 조금씩 다른 속도·두께). 실측이 아니므로 학습용이 아니라 시험용이다.

폴더는 복원 데이터 규약(README)을 따르고, 좌표·단위를 일부러 다르게 저장해 임포트 변환도 함께 시험한다.
  cucumber_slices : 오이 끝쪽 절반을 끝에서부터 네 번 썰기.  world 좌표·mm (KIT 처럼, 탁자 위 비스듬히 놓임)
  apple_quarters  : 사과를 반으로 가른 뒤 칼을 90° 돌려 다시 갈라 4등분.  object 좌표·m
  potato_press_saw: 감자를 누르기만 하다가(힘이 모자라 멈춤) 톱질로 바꿔 반 가르기.  object 좌표·cm·y-up

칼 자세(traj_knife.npz 의 T_world_knife): 원점 = 날 끝 한가운데, x = 칼날 면 법선, y = 칼날 길이 방향,
z = 날 끝 → 칼등. 실제 데이터의 칼 좌표가 이와 다르면 고정 변환을 곱해 맞춰야 한다.

python scripts/07_make_demos.py
결과: data/demos/<id>/{mesh.obj, meta.yaml, traj_knife.npz, gt_traj_object.npz}, reports/07_demos/trajectories.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from cutsim.io.recon_loader import normalize_transform

OUT = Path("data/demos")
KEYS = ("x", "y", "z", "yaw", "pitch", "roll")
FPS = 100
Z_END = 0.0  # 사람은 칼이 도마에 닿을 때까지 내린다(시뮬레이션은 날 끝 최저 높이로 자른다)


class Hand:
    """사람 손 칼 동작(물체 좌표, m·rad). 한 행 = (x, y, z, yaw, pitch, roll), FPS Hz."""

    def __init__(self, q0):
        self.q, self.ph = [np.asarray(q0, float)], ["start"]

    @property
    def last(self):
        return self.q[-1].copy()

    def _add(self, q, ph):
        self.q.append(np.asarray(q, float))
        self.ph.append(ph)

    def move(self, T, phase="move", **tgt):
        """최소 저크 이동(사람 팔처럼 부드럽게 출발·정지)."""
        q0 = self.last
        q1 = q0.copy()
        for k, v in tgt.items():
            q1[KEYS.index(k)] = v
        n = max(1, int(round(T * FPS)))
        for i in range(1, n + 1):
            s = i / n
            self._add(q0 + (q1 - q0) * (10 * s**3 - 15 * s**4 + 6 * s**5), phase)
        return self

    def hold(self, T, phase="hold"):
        for _ in range(int(round(T * FPS))):
            self._add(self.last, phase)
        return self

    def saw(self, v_down, amp, freq, phase="cut", press_T=0.0, v_press=0.02, extra_T=0.0):
        """칼날 길이 방향으로 앞뒤로 썰며 도마까지 내려간다.

        press_T: 처음 이 시간 동안은 톱질 없이 v_press 로 누르기만 한다. extra_T: 도마에 닿은 뒤에도 이만큼
        더 썬다(사람은 다 잘릴 때까지 계속 썬다). 끝나면 톱질 위치를 가운데로 되돌린다.
        """
        q0 = self.last
        tdir = np.array([-np.sin(q0[3]), np.cos(q0[3])])
        z = q0[2]
        for _ in range(int(round(press_T * FPS))):
            z = max(Z_END, z - v_press / FPS)
            q = q0.copy()
            q[2] = z
            self._add(q, phase)
        n = int(np.ceil(max(z - Z_END, 0.0) / v_down * FPS + extra_T * FPS))
        for i in range(1, n + 1):
            t = i / FPS
            q = q0.copy()
            q[2] = max(Z_END, z - v_down * t)
            q[0:2] = q0[0:2] + tdir * amp * np.sin(2 * np.pi * freq * t)
            self._add(q, phase)
        return self.move(0.15, phase, x=q0[0], y=q0[1])

    def arrays(self, rng, noise):
        """손떨림: 0.2초 정도로 천천히 변하는 잡음을 x·y·yaw·pitch·roll 에 더한다(z 는 도마를 뚫지 않게 뺀다)."""
        q = np.stack(self.q)
        for k, std in noise.items():
            w = gaussian_filter1d(rng.standard_normal(len(q)), sigma=0.2 * FPS)
            q[:, KEYS.index(k)] += w / (w.std() + 1e-12) * std
        return np.arange(len(q)) / FPS, q, np.array(self.ph)


def poses(q):
    """(N,6) → (N,4,4) 칼 자세. 회전 = Rz(yaw)·Rx(pitch)·Ry(roll): pitch 는 칼끝 오르내림, roll 은 칼날 옆 기울기."""
    T = np.tile(np.eye(4), (len(q), 1, 1))
    T[:, :3, :3] = Rotation.from_euler("ZXY", q[:, 3:6]).as_matrix()
    T[:, :3, 3] = q[:, :3]
    return T


def top_near(mesh, x, half=0.004):
    v = mesh.vertices
    m = np.abs(v[:, 0] - x) < half
    return float(v[m, 2].max())


def bottom_center(mesh):
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return mesh


def cucumber_slices(rng):
    full = trimesh.load("data/recon/cnl_cucumber/mesh.obj", force="mesh")
    # 끝쪽 절반(10cm)만: 사람도 긴 오이는 반 토막 내서 썬다. 계산 영역이 반으로 줄어 2배쯤 빨라진다.
    box = trimesh.creation.box(extents=[0.3, 0.3, 0.3])
    box.apply_translation([0.15, 0.0, 0.0])  # x >= 0 쪽
    mesh = bottom_center(trimesh.boolean.intersection([full, box], engine="manifold"))
    x_tip = mesh.bounds[1, 0]
    cuts = x_tip - np.array([0.020, 0.032, 0.044, 0.056]) + rng.normal(0, 0.0008, 4)
    h = Hand([cuts[0] + 0.03, -0.01, mesh.bounds[1, 2] + 0.03, 0.0, 0.0, 0.0])
    for i, cx in enumerate(cuts):
        zt = top_near(mesh, cx)
        h.move(0.35 if i == 0 else 0.25, "approach", x=cx, y=-0.006, z=zt + 0.004, yaw=rng.normal(0, 0.03))
        h.saw(v_down=0.055 * rng.uniform(0.9, 1.1), amp=0.012, freq=1.4 * rng.uniform(0.9, 1.1))
        h.hold(0.08).move(0.3, "lift", z=zt + 0.012)
    h.move(0.3, "retreat", x=cuts[-1] - 0.02, z=mesh.bounds[1, 2] + 0.03).hold(0.2)
    task = "반 토막 오이를 끝에서부터 약 1.2cm 두께로 네 번 썰기(앞뒤로 밀며 썰기)"
    # 다른 손 대용: 썰지 않는 쪽 끝(2.6cm)을 위에서 1.3mm 눌러 붙잡는 상자. 정규화 좌표(바닥 중심 원점, m),
    # 시뮬레이션은 식재료를 0.5mm 띄워 놓으므로 윗면 + 0.5mm 기준.
    lo = mesh.bounds[0, 0]
    x0, x1 = lo + 0.0015, lo + 0.0275
    top = top_near(mesh, (x0 + x1) / 2, half=(x1 - x0) / 2) + 0.0005
    hold = [round(float(v), 4) for v in ((x0 + x1) / 2, -0.0015, top - 0.0013 + 0.0075, x1 - x0, 0.05, 0.015)]
    return mesh, h, task, dict(units="mm", up_axis="z", frame="world",
                               T_world_object=world_pose(25.0, [520.0, -140.0, 760.0]), hold_box=[hold]), \
        "data/recon/cnl_cucumber 의 끝쪽 절반"


def apple_quarters(rng):
    mesh = trimesh.load("data/recon/cnl_apple/mesh.obj", force="mesh")
    top = mesh.bounds[1, 2]
    h = Hand([0.0, 0.0, top + 0.03, 0.0, 0.0, 0.0])
    h.move(0.3, "approach", z=top + 0.003)
    h.saw(v_down=0.045, amp=0.015, freq=1.2, extra_T=0.2)
    h.hold(0.1).move(0.35, "lift", z=top + 0.015)
    h.move(0.45, "turn", yaw=np.pi / 2)  # 칼을 90° 돌려 첫 절단면과 직각으로
    h.move(0.2, "approach", z=top + 0.003)
    h.saw(v_down=0.045, amp=0.015, freq=1.2, extra_T=0.2)
    h.hold(0.1).move(0.35, "lift", z=top + 0.03).hold(0.2)
    task = "사과를 반으로 가른 뒤 칼을 90° 돌려 다시 갈라 4등분하기"
    return mesh, h, task, dict(units="m", up_axis="z", frame="object", T_world_object=None), "data/recon/cnl_apple"


def potato_press_saw(rng):
    mesh = trimesh.load("data/recon/cnl_potato/mesh.obj", force="mesh")
    top = mesh.bounds[1, 2]
    h = Hand([0.0, 0.0, top + 0.03, 0.0, 0.0, 0.0])
    h.move(0.3, "approach", z=top + 0.002)
    # 1.2초 동안 2cm/s 로 누르기만 하다가 톱질로 바꾼다. 도마에 닿은 뒤에도 1초 더 썬다.
    h.saw(v_down=0.025, amp=0.016, freq=1.8, press_T=1.2, v_press=0.02, extra_T=1.0)
    h.hold(0.1).move(0.35, "lift", z=top + 0.03).hold(0.2)
    task = "감자를 누르기만 하다가 안 들어가자 톱질로 바꿔 반으로 가르기"
    return mesh, h, task, dict(units="cm", up_axis="y", frame="object", T_world_object=None), "data/recon/cnl_potato"


def world_pose(yaw_deg, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    T[:3, 3] = t
    return T


DEMOS = {
    "cucumber_slices": (cucumber_slices, "cucumber"),
    "apple_quarters": (apple_quarters, "apple"),
    "potato_press_saw": (potato_press_saw, "potato"),
}
NOISE = {"x": 0.0003, "y": 0.0003, "yaw": np.radians(1.0), "pitch": np.radians(2.0), "roll": np.radians(1.5)}


def export(eid, category, mesh, t, q, ph, task, frame, food_src, seed):
    d = OUT / eid
    d.mkdir(parents=True, exist_ok=True)
    frame = dict(frame)
    hold = frame.pop("hold_box", None)
    M = normalize_transform(frame["units"], frame["up_axis"], frame["T_world_object"])  # 파일 좌표 → 물체(m, z-up)
    Minv = np.linalg.inv(M)
    s = np.cbrt(np.linalg.det(M[:3, :3]))
    m_file = mesh.copy()
    m_file.apply_transform(Minv)
    m_file.export(d / "mesh.obj")
    T_file = np.einsum("ij,njk->nik", Minv, poses(q))
    T_file[:, :3, :3] *= s  # 회전 부분의 배율을 걷어 정규 직교로
    np.savez(d / "traj_knife.npz", t=t, T_world_knife=T_file, fps=float(FPS), phase=ph)
    np.savez(d / "gt_traj_object.npz", t=t, q=q, phase=ph)
    src_meta = yaml.safe_load(open(Path(food_src.split()[0]) / "meta.yaml"))
    tilt = np.degrees(np.arccos(np.clip(poses(q)[:, 2, 2], -1, 1)))
    meta = {"object_id": f"demo_{eid}", "source": "synthetic_demo", "category": category, **{
        k: (np.asarray(v).round(6).tolist() if isinstance(v, np.ndarray) else v) for k, v in frame.items()},
        "color_skin": src_meta.get("color_skin"), "color_flesh": src_meta.get("color_flesh"),
        "task": task, "food_source": food_src,
        "demo": {"fps": FPS, "duration_s": round(float(t[-1]), 2), "seed": seed,
                 "knife_tilt_deg_max": round(float(tilt.max()), 2),
                 "note": "사람 동작을 흉내 낸 합성 궤적(실측 아님). 시뮬레이션 칼 지그는 4자유도라 기울기는 버린다"},
        "knife_frame": "원점 = 날 끝 한가운데, x = 칼날 면 법선, y = 칼날 길이 방향, z = 날 끝→칼등"}
    if hold:
        meta["hold_box"] = hold
        meta["hold_box_note"] = "다른 손 대용 고정 상자 [cx, cy, cz, sx, sy, sz], 정규화 좌표(m, 바닥 중심 원점)"
    (d / "meta.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
    return meta


if __name__ == "__main__":
    from cutsim.plotting import plt

    fig, axs = plt.subplots(len(DEMOS), 2, figsize=(13, 3.2 * len(DEMOS)))
    for r, (eid, (fn, cat)) in enumerate(DEMOS.items()):
        seed = 100 + r
        rng = np.random.default_rng(seed)
        mesh, hand, task, frame, food_src = fn(rng)
        t, q, ph = hand.arrays(rng, NOISE)
        meta = export(eid, cat, mesh, t, q, ph, task, frame, food_src, seed)
        print(f"{eid}: {t[-1]:.2f}s, food {np.round(mesh.extents * 100, 1)} cm, {frame['units']}/{frame['frame']}/"
              f"{frame['up_axis']}-up, tilt max {meta['demo']['knife_tilt_deg_max']}°")
        ax = axs[r, 0]
        ax.plot(t, q[:, 2] * 1000, "k", lw=1, label="edge z (mm)")
        ax.plot(t, q[:, 0] * 1000, lw=0.8, label="x (mm)")
        ax.plot(t, q[:, 1] * 1000, lw=0.8, label="y (mm)")
        ax.plot(t, np.degrees(q[:, 3]), lw=0.8, label="yaw (deg)")
        ax.axhline(mesh.bounds[1, 2] * 1000, color="g", ls=":", lw=0.8, label="food top")
        ax.set_title(f"{eid}: {task}", fontsize=9); ax.set_xlabel("time (s)"); ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3)
        ax = axs[r, 1]  # 위에서 본 식재료 윤곽과 칼날이 도마에 닿은 자리
        sec = mesh.section(plane_origin=[0, 0, mesh.bounds[1, 2] * 0.4], plane_normal=[0, 0, 1])
        for ent in (sec.discrete if sec is not None else []):
            ax.plot(ent[:, 0] * 100, ent[:, 1] * 100, color="tab:green")
        low = q[:, 2] < 0.002
        for i in np.flatnonzero(low)[:: 15]:
            c, tv = q[i, :2], np.array([-np.sin(q[i, 3]), np.cos(q[i, 3])])
            p = np.stack([c - tv * 0.04, c + tv * 0.04]) * 100
            ax.plot(p[:, 0], p[:, 1], color="gray", lw=0.5, alpha=0.5)
        ax.set_aspect("equal"); ax.set_title("top view (cm): food outline, blade at board", fontsize=9)
    fig.tight_layout()
    Path("reports/07_demos").mkdir(parents=True, exist_ok=True)
    fig.savefig("reports/07_demos/trajectories.png", dpi=100)
