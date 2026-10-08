"""ElRobot(norma-core URDF) 정·역기구학과 표시용 메쉬. Genesis 없이 numpy 로(그림·궤적 계획용).

URDF 의 관절 원점 회전(rpy)은 모두 0 이라 링크 자세 = 부모 자세 · 이동(origin) · 관절 회전/이동.
집게 끝(TCP) = 두 집게 끝 10 mm 의 가운데(Genesis 점검과 같은 정의), 집게 몸통 링크 좌표 (0.0017, 0.0995, 0.0001).
집게 좌표계: y = 접근 방향(집게 몸통 → 집게 끝), x = 집게가 닫히는 방향, z = x × y.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "third_party/norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"
ARM = [f"rev_motor_0{i}" for i in range(1, 8)]
GRIP = "rev_motor_08"
TCP_L = np.array([0.0017, 0.0995, 0.0001])
JAW_GAP_OPEN_MM, JAW_MM_PER_RAD = 53.1, 2 * 11.5          # 집게 끝 사이 간격 = 53.1 − 23·q8 (mm)


def vec(s):
    return np.array([float(x) for x in s.split()])


class Robot:
    def __init__(self, urdf=URDF, T_base=np.eye(4)):
        self.urdf = Path(urdf)
        root = ET.parse(self.urdf).getroot()
        self.T_base = np.asarray(T_base, float)
        self.links, self.joints = {}, {}
        for l in root.iter("link"):
            v = l.find("visual")
            self.links[l.get("name")] = None if v is None else (vec(v.find("origin").get("xyz")), v.find("geometry/mesh").get("filename"))
        for j in root.iter("joint"):
            o, a, lim, mm = j.find("origin"), j.find("axis"), j.find("limit"), j.find("mimic")
            assert o is None or np.allclose(vec(o.get("rpy", "0 0 0")), 0), "rpy 가 0 이 아닌 관절"
            self.joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                                              xyz=vec(o.get("xyz")) if o is not None else np.zeros(3),
                                              axis=vec(a.get("xyz")) / np.linalg.norm(vec(a.get("xyz"))) if a is not None else None,
                                              lim=(float(lim.get("lower")), float(lim.get("upper"))) if lim is not None and j.get("type") != "fixed" else None,
                                              mimic=(mm.get("joint"), float(mm.get("multiplier")), float(mm.get("offset", 0))) if mm is not None else None)
        kids = {j["child"] for j in self.joints.values()}
        self.base = next(n for n in self.links if n not in kids)
        self.order = []                               # 부모 → 자식 순서의 관절
        frontier = [self.base]
        while frontier:
            p = frontier.pop(0)
            for n, j in self.joints.items():
                if j["parent"] == p:
                    self.order.append(n); frontier.append(j["child"])
        self.lo = np.array([self.joints[n]["lim"][0] for n in ARM])
        self.hi = np.array([self.joints[n]["lim"][1] for n in ARM])
        a = TCP_L / np.linalg.norm(TCP_L)
        c = -self.joints["rev_motor_08_2"]["axis"]     # 집게 손가락 이동 축(몸통 좌표) ≈ +x
        c = c - a * (c @ a); c /= np.linalg.norm(c)
        self.R_gb_grip = np.c_[c, a, np.cross(c, a)]   # 집게 몸통 링크 → 집게 좌표계
        self._mesh = {}

    def fk(self, q_arm, q_grip=0.0):
        """→ {링크 이름: 4×4 세계 자세}"""
        val = dict(zip(ARM, q_arm)) | {GRIP: q_grip}
        T = {self.base: self.T_base.copy()}
        for n in self.order:
            j = self.joints[n]
            M = np.eye(4); M[:3, 3] = j["xyz"]
            v = val.get(n, 0.0)
            if j["mimic"]:
                src, mul, off = j["mimic"]
                v = val.get(src, 0.0) * mul + off
            if j["type"] == "revolute":
                R = np.eye(4); R[:3, :3] = Rotation.from_rotvec(j["axis"] * v).as_matrix()
                M = M @ R
            elif j["type"] == "prismatic":
                P = np.eye(4); P[:3, 3] = j["axis"] * v
                M = M @ P
            T[j["child"]] = T[j["parent"]] @ M
        return T

    def grip_pose(self, q_arm):
        """집게 좌표계(원점 TCP)의 세계 자세"""
        Tg = self.fk(q_arm)["Gripper_Base_v1_1"]
        out = np.eye(4)
        out[:3, :3] = Tg[:3, :3] @ self.R_gb_grip
        out[:3, 3] = Tg[:3, :3] @ TCP_L + Tg[:3, 3]
        return out

    def ik(self, T_target, q0, w_rot=0.05, reg=1e-3):
        """집게 좌표계 목표 자세(4×4) → 팔 관절 7개. 위치(m)와 회전(rad × w_rot m)을 함께 맞추고, q0 에 가깝게."""
        q0 = np.clip(np.asarray(q0, float), self.lo + 1e-6, self.hi - 1e-6)

        def res(q):
            T = self.grip_pose(q)
            dp = T[:3, 3] - T_target[:3, 3]
            dr = Rotation.from_matrix(T_target[:3, :3].T @ T[:3, :3]).as_rotvec()
            return np.r_[dp, dr * w_rot, reg * (q - q0)]

        r = least_squares(res, q0, bounds=(self.lo, self.hi), xtol=1e-10, ftol=1e-10, max_nfev=400)
        T = self.grip_pose(r.x)
        err_mm = float(np.linalg.norm(T[:3, 3] - T_target[:3, 3]) * 1000)
        err_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(T_target[:3, :3].T @ T[:3, :3]).as_rotvec())))
        return r.x, err_mm, err_deg

    def meshes(self):
        """표시용: [(링크 이름, trimesh(링크 좌표), 서보인지)]"""
        if not self._mesh:
            for n, v in self.links.items():
                if v is None:
                    continue
                off, fn = v
                m = trimesh.load(self.urdf.parent / fn, force="mesh")
                m.apply_scale(0.001); m.apply_translation(off)
                self._mesh[n] = (m, n.startswith("ST3215"))
        return [(n, m, servo) for n, (m, servo) in self._mesh.items()]

    @staticmethod
    def grip_angle(gap_mm):
        """집게 끝 간격(mm) → rev_motor_08 각도(rad)"""
        return float(np.clip((JAW_GAP_OPEN_MM - gap_mm) / JAW_MM_PER_RAD, 0.0, 2.2028))


if __name__ == "__main__":
    rb = Robot()
    T = rb.grip_pose(np.zeros(7))
    print("q=0 TCP(m)", T[:3, 3].round(4), "(Genesis 점검값 0.0047, 0.3591, 0.219)")
    print("접근 방향", T[:3, 1].round(3), "닫힘 방향", T[:3, 0].round(3))
