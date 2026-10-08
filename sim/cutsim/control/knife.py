"""칼 궤적: 관절 (x, y, z, yaw) 목표 위치·속도를 스텝마다 미리 만들어 둔다.

z 는 날 끝 높이(m), yaw=0 일 때 칼날 길이 방향 = y, 칼날 면 법선 = x.
"""
import numpy as np


class KnifeTraj:
    def __init__(self, q0, dt):
        self.dt = dt
        self.q = [np.asarray(q0, dtype=float)]
        self.qd = [np.zeros(4)]
        self.phase = ["start"]

    @property
    def last(self):
        return self.q[-1].copy()

    def _append(self, q, qd, phase):
        self.q.append(np.asarray(q, dtype=float))
        self.qd.append(np.asarray(qd, dtype=float))
        self.phase.append(phase)

    def move_to(self, target, speed=0.03, yaw_speed=2.0, phase="move", **named):
        """직선 등속 이동. target 은 길이 4 배열, 또는 None 이고 x=,y=,z=,yaw= 키워드로 일부만."""
        q0 = self.last
        q1 = q0.copy() if target is None else np.asarray(target, dtype=float)
        for i, k in enumerate(("x", "y", "z", "yaw")):
            if k in named:
                q1[i] = named[k]
        dq = q1 - q0
        T = max(np.abs(dq[:3]).max() / speed, abs(dq[3]) / yaw_speed, self.dt)
        n = int(np.ceil(T / self.dt))
        v = dq / (n * self.dt)
        for i in range(1, n + 1):
            self._append(q0 + dq * i / n, v, phase)
        return self

    def hold(self, duration, phase="hold"):
        q = self.last
        for _ in range(int(round(duration / self.dt))):
            self._append(q, np.zeros(4), phase)
        return self

    def press(self, z_end, vz, saw_amp=0.0, saw_freq=0.0, phase="cut", tail_s=0.0):
        """z_end 까지 등속 vz 로 하강. saw_amp>0 이면 y 로 사인 톱질을 겹친다.

        tail_s: z_end 에 닿은 뒤에도 같은 동작(톱질 또는 누르기)을 이만큼 더 이어 간다. 힘 예산 때문에 칼이 뒤처져도
        끝까지 썰 시간을 준다(사람이 다 썰릴 때까지 계속 써는 것과 같다).
        """
        q0 = self.last
        n_down = int(np.ceil((q0[2] - z_end) / vz / self.dt))
        n = n_down + int(round(tail_s / self.dt))
        for i in range(1, n + 1):
            t = i * self.dt
            q, qd = q0.copy(), np.zeros(4)
            if i <= n_down:
                q[2], qd[2] = q0[2] - vz * t, -vz
            else:
                q[2], qd[2] = z_end, 0.0
            if saw_amp > 0:
                w = 2 * np.pi * saw_freq
                q[1] = q0[1] + saw_amp * np.sin(w * t)
                qd[1] = saw_amp * w * np.cos(w * t)
            self._append(q, qd, phase)
        if saw_amp > 0:  # 톱질 위치를 원래 y 로 되돌림
            self.move_to(None, speed=0.05, phase=phase, y=q0[1])
        return self

    def extend(self, q, qd, phases):
        """미리 만든 궤적(사람 칼 궤적 재생 등)을 그대로 붙인다."""
        for a, b, p in zip(q, qd, phases):
            self._append(a, b, str(p))
        return self

    def arrays(self):
        return np.stack(self.q), np.stack(self.qd), np.array(self.phase)

    def __len__(self):
        return len(self.q)
