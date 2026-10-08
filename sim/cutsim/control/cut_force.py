"""칼에 거는 절단 저항(현상론 모델). MPM 은 식재료 모양·분리만 맡고, 저항은 여기서 계산해 칼에 건다.

왜 따로 두나: 칼날(1.5~2mm)이 MPM 격자 칸보다 얇아, MPM 이 주는 칼 힘은 칼과 격자의 상대 위치에 따라
0.1N~10N 으로 바뀐다(README.md 4장). 여기서는 칼날이 재료를 실제로 가르는 길이로 힘을 정한다.

  절단(파괴) 항: 칼날 끝 중 바로 앞(칼이 들어가는 쪽)에 재료가 있는 길이 L.
                 일률 F·v = G_c · L · v_n 이 되도록 힘을 칼의 면내 속도 반대 방향으로 건다(Atkins slice-push).
                 수직 성분 = G_c·L / (1+ξ²), ξ = 칼날 방향 속도 / 들어가는 방향 속도 → 톱질하면 줄어든다.
  옆면 마찰 항:  칼 옆면이 재료에 묻힌 면적 A(양면)에 비례하는 마찰 τ·A, 면내 속도 반대 방향.

G_c(N/m, 단위 길이당 절단 저항 = 파괴 인성)와 τ(Pa, 옆면 마찰 응력)는 DiSECt 힘 곡선에 맞춘다
(scripts/04_fit_cut_force.py). 힘은 칼에만 걸리고 재료에는 걸리지 않는다(작용-반작용 불일치, 알려진 한계).
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class BladeFrame:
    """칼날 기하(칼 링크 좌표계 기준): 원점 = 날 끝 한가운데, 길이 방향 t, 면 법선 n, 등 쪽 u(날 끝 → 등)."""
    length: float
    height: float
    thickness: float


def blade_axes(R):
    """칼 링크 회전행렬 → (t 길이 방향, n 면 법선, u 등 쪽) 월드 벡터. 지그 규약: 링크 y=길이, x=법선, z=등 쪽."""
    return R[:, 1], R[:, 0], R[:, 2]


def engaged_geometry(pos, tree, origin, R, blade, p_size, n_samples=64):
    """칼날이 재료를 가르는 길이 L(m)과 옆면이 재료에 묻힌 면적 A(m², 양면 합)를 입자 위치로 잰다.

    L: 날 끝을 n_samples 점으로 나눠, 각 점 바로 앞(-u 방향 0~1.5 입자, 칼날 면 ±1 입자 폭)에 입자가 있으면 그 구간이
       재료를 가르는 중으로 본다. 이미 자르고 지나간 틈(재료가 칼 옆에만 있음)은 세지 않는다.
    A: 칼 옆면(면 법선 방향 ±(두께/2 + 1.5 입자)) 안쪽, 날 끝~등 높이, 칼 길이 안의 입자 수 × 입자 하나의 면적 / 2.
    """
    t, n, u = blade_axes(R)
    s = np.linspace(-blade.length / 2, blade.length / 2, n_samples)
    seg = blade.length / n_samples
    pts = origin[None] + s[:, None] * t[None]
    r_query = max(1.5 * p_size, seg)
    idx_lists = tree.query_ball_point(pts, r=r_query + 1.5 * p_size)
    engaged = np.zeros(n_samples, bool)
    for i, idx in enumerate(idx_lists):
        if not idx:
            continue
        d = pos[idx] - pts[i]
        ahead = -(d @ u)  # 날 끝에서 칼이 들어가는 방향(-u)으로 얼마나 앞인가
        side = np.abs(d @ n)
        along = np.abs(d @ t)
        engaged[i] = np.any((ahead > -0.25 * p_size) & (ahead < 1.5 * p_size) & (side < p_size) & (along < seg))
    L = engaged.mean() * blade.length

    rel = pos - origin[None]
    a_n = np.abs(rel @ n)
    a_t = rel @ t
    a_u = rel @ u
    face = (a_n < blade.thickness / 2 + 1.5 * p_size) & (np.abs(a_t) < blade.length / 2) & (a_u > 0) & (a_u < blade.height)
    A = face.sum() * p_size ** 2  # 면마다 입자 한 겹이 닿는다고 보고 개수 × 입자 단면적(양면 합). 틈이 있는 격자 배치로 검증
    return L, A, engaged


@dataclass
class CutForceModel:
    G_c: float          # N/m
    tau: float          # Pa (옆면 마찰 응력)
    v0: float = 0.002   # m/s, 속도 0 근처에서 힘을 tanh(속도/v0) 로 부드럽게 줄인다(켜졌다 꺼지는 떨림 방지)

    def force(self, v, R, L, A):
        """칼 속도 v(월드)와 기하 L, A 로 칼에 걸 힘(월드, N)과 분해 정보를 돌려준다."""
        t, n, u = blade_axes(R)
        v_n = -(v @ u)       # 칼이 재료 쪽(-u)으로 들어가는 속도
        v_t = v @ t          # 칼날 방향(톱질) 속도
        v_p = v_n * (-u) + v_t * t  # 면내 속도
        speed = np.linalg.norm(v_p)
        F = np.zeros(3)
        info = {"L": L, "A": A, "v_n": v_n, "v_t": v_t, "F_cut": 0.0, "F_fric": 0.0}
        if speed < 1e-9:
            return F, info
        dirn = v_p / speed
        reg = np.tanh(speed / self.v0)
        if v_n > 0 and L > 0:
            f_cut = self.G_c * L * v_n / speed * reg  # F·v = G_c L v_n (빠를 때)
            F -= f_cut * dirn
            info["F_cut"] = f_cut
        if A > 0:
            f_fric = self.tau * A * reg
            F -= f_fric * dirn
            info["F_fric"] = f_fric
        return F, info

    def vertical_resistance(self, L, A, v_n, v_t):
        """명령 속도(들어가는 v_n, 톱질 v_t)로 칼을 밀 때 필요한 들어가는 방향 힘. Atkins: G_c L/(1+ξ²) + τA·v_n/|v|."""
        if v_n <= 0:
            return 0.0
        speed = np.hypot(v_n, v_t)
        return self.G_c * L * (v_n / speed) ** 2 + self.tau * A * v_n / speed


class CutForceApplier:
    """씬 스텝마다: 입자 위치 → L, A → 힘 → 칼 링크에 외력으로 건다."""

    def __init__(self, cs, model, blade, p_size, every=1):
        self.cs, self.model, self.blade, self.p_size, self.every = cs, model, blade, p_size, every
        self.link = cs.knife.get_link("blade")
        self.last = (np.zeros(3), {"L": 0.0, "A": 0.0, "F_cut": 0.0, "F_fric": 0.0, "v_n": 0.0, "v_t": 0.0})
        self.k = 0

    def update(self):
        if self.k % self.every == 0:
            pos = np.concatenate(list(self.cs.particles().values()))
            tree = cKDTree(pos)
            origin = self.link.get_pos().cpu().numpy().reshape(3)
            quat = self.link.get_quat().cpu().numpy().reshape(4)  # w, x, y, z
            R = quat_to_R(quat)
            v = self.link.get_vel().cpu().numpy().reshape(3)
            L, A, _ = engaged_geometry(pos, tree, origin, R, self.blade, self.p_size)
            self.last = self.model.force(v, R, L, A)
        self.k += 1
        F, info = self.last
        if np.any(F):
            self.link.apply_external_force(F.astype(np.float32))
        return F, info


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ---------------- DiSECt 반원기둥에 맞추기 위한 해석 기하 ----------------
def half_cylinder_chord(d, r):
    """반지름 r 반원 단면을 위에서 깊이 d 만큼 자를 때 날 끝이 재료 안에 있는 길이(현)."""
    d = np.clip(d, 0, r)
    return 2 * np.sqrt(np.maximum(r ** 2 - (r - d) ** 2, 0))


def half_cylinder_segment_area(d, r):
    """위에서 깊이 d 까지 잠긴 반원 단면의 활꼴 넓이(한쪽 면)."""
    d = np.clip(d, 0, r)
    h = r - d
    return r ** 2 * np.arccos(h / r) - h * np.sqrt(np.maximum(r ** 2 - h ** 2, 0))
