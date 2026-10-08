"""Grasp mode for the arm console: registration, object selection, planning, execution.

Commands (from the browser):
  reg_set      {robot: 16, world: 16}         user-adjusted robot / iPhone-world transforms (row-major 4×4), saved
  grasp_select {p_world, M_scene_world, M_scene_base}   segment the object under the click
  grasp_plan   {}                              plan a grasp for the selected object (planner.py)
  grasp_go     {}                              plan, and when a grasp is found execute it right away
  grasp_pick   {p_world, M_scene_world, M_scene_base}   right-click on an object: select + grasp_go
  grasp_place  {p_world, M_scene_world, M_scene_base, D_world: 16}   object moved in the console (gizmo): pick it
               and place it at D_world · its pose (ARKit world, row-major 4×4) — plan, predict, execute
  grasp_exec   {}                              drive the real arm through the plan
  grasp_cancel {}                              stop / clear

Joint ↔ servo mapping is the console's: p = (raw − min)/(max − min) (1 − p if inverted),
joints 1–7: angle = lower + p·(upper − lower); gripper rev_motor_08: angle = upper − (lower + p·(upper − lower)).
"""
import json
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REG_FILE = HERE.parent / "registration.json"
JOINTS = [f"rev_motor_0{i}" for i in range(1, 9)]
CTRL_HZ = 20


def _urdf_limits(urdf):
    lim = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        l = j.find("limit")
        if l is not None and j.get("name") in JOINTS:
            lim[j.get("name")] = (float(l.get("lower")), float(l.get("upper")))
    return lim


SHOULDER_BASE = np.array([-0.02, 0.042, 0.091])   # joint 2 in the base frame: TCP reach ≤ 43.4 cm from it (sampled)


def _yaw_only(m16):
    """[ours] registration (row-major 4×4, scene y up) with its tilt removed: only the turn about the vertical stays,
    the position is kept. A tilted robot / world (left by a free-rotation gizmo) sat askew in the scene."""
    M = np.array(m16, float).reshape(4, 4)
    x = M[:3, 0]
    a = np.arctan2(-x[2], x[0])                     # heading of the x axis in the horizontal plane
    c, s_ = np.cos(a), np.sin(a)
    M[:3, :3] = [[c, 0, s_], [0, 1, 0], [-s_, 0, c]]
    return M.ravel().tolist()


def _quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class GraspManager:
    def __init__(self, arm, cam, urdf_path):
        self.arm, self.cam = arm, cam
        self.lim = _urdf_limits(urdf_path)
        try:
            reg = json.loads(REG_FILE.read_text())
        except Exception:
            reg = {"robot": np.eye(4).ravel().tolist(), "world": np.eye(4).ravel().tolist()}
        reg = {**reg, **{k: _yaw_only(reg[k]) for k in ("robot", "world")}}
        self.state = {"registration": reg, "stage": "idle", "error": None, "plan": None, "object": None,
                      "obj_version": 0, "exec": None, "virtual_moved": {}}
        self._vmoved = {}         # [ours] recon object id → 4×4 ARKit-world move done by the virtual robot (real camera)
        self._sel_oid = None
        self._sel = None          # (points_base, tris, click_base, up_base, T_base_world)
        self._seg = None          # (obj_idx, support, env_idx) when the object came from the reconstruction
        self._obj_mesh = None     # per-object TSDF mesh blob (ARKit world)
        self._obj_world = None
        self._abort = threading.Event()
        self._thread = None
        threading.Thread(target=self._warm_up, daemon=True).start()

    @staticmethod
    def _warm_up():
        """First grasp after start took ~15 s (imports, IK model, gripper hulls, MuJoCo compile); do it at start."""
        try:
            import mujoco
            import planner
            import sim
            planner._gt().gripper_hulls()
            planner._model()
            mujoco.MjSpec.from_string(sim.arm_xml()).compile()
        except Exception as e:
            print("grasp warm-up failed:", e)

    # ---- joint mapping
    def angle_of(self, mid, raw):
        a = self.arm.calib["motors"].get(str(mid))
        if not a:
            return None
        lo, hi = self.lim[JOINTS[mid - 1]]
        p = (raw - a["min"]) / (a["max"] - a["min"])
        p = 1 - p if a.get("invert") else p
        v = lo + p * (hi - lo)
        return hi - v + lo if mid == 8 else v     # gripper: upper − (lower + p·range) with lower = 0

    def raw_of(self, mid, angle):
        a = self.arm.calib["motors"].get(str(mid))
        lo, hi = self.lim[JOINTS[mid - 1]]
        v = hi - angle + lo if mid == 8 else angle
        p = (v - lo) / (hi - lo)
        p = 1 - p if a.get("invert") else p
        return int(round(a["min"] + np.clip(p, 0, 1) * (a["max"] - a["min"])))

    def q_now(self):
        q = []
        for mid in range(1, 9):
            m = self.arm.state["motors"].get(mid)
            ang = self.angle_of(mid, m["pos"]) if m else None
            if ang is None:
                raise RuntimeError(f"모터 {mid} 상태 또는 캘리브레이션이 없습니다")
            q.append(ang)
        return np.array(q)

    # ---- scene data from the camera
    def _scene_points(self):
        """Raw LiDAR points of the newest keyframes (registered by ARKit, 4 mm grid) — sharper than the 1 cm TSDF
        mesh, which rounds edges and skirts objects (a 40×50 mm box came out 95×83 mm). Mesh / live cloud = fallback."""
        rc = self.cam.recon if self.cam else None
        if rc is not None and getattr(rc, "_kf", None):
            with rc._lock:
                kfs = list(rc._kf[-12:])
            pts = []
            for k in kfs:
                d = k["depth"][::2, ::2]
                v, u = np.nonzero(d > 0.05)
                z = d[v, u]
                K = k["K"]
                pc = np.c_[(u * 2 - K[0, 2]) / K[0, 0] * z, (v * 2 - K[1, 2]) / K[1, 1] * z, z]
                pts.append(pc @ k["T"][:3, :3].T + k["T"][:3, 3])
            P = np.vstack(pts)
            # 4 mm grid, one point per cell = the cell's mean (averages the ~2 mm LiDAR noise instead of keeping it)
            _, inv, cnt = np.unique(np.floor(P / 0.004).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
            inv = inv.ravel()
            M = np.zeros((len(cnt), 3))
            np.add.at(M, inv, P)
            return M / cnt[:, None], None, f"원본 깊이 점 (키프레임 {len(kfs)}장, 4 mm 평균)"
        blob = rc.mesh_bytes() if rc else None
        if blob and len(blob) > 8:
            nv, nt = np.frombuffer(blob[:8], np.uint32)
            v = np.frombuffer(blob, np.float32, nv * 3, 8).reshape(-1, 3).astype(np.float64)
            tris = np.frombuffer(blob, np.uint32, nt * 3, 8 + nv * 24).reshape(-1, 3).astype(np.int64)
            return v, tris, "복원 메시"
        pts = getattr(self.cam, "_last_world_pts", None) if self.cam else None
        if pts is None or not len(pts):
            raise RuntimeError("장면 데이터가 없습니다 (iPhone 스트림 / 복원 확인)")
        return pts.astype(np.float64), None, "현재 점군"

    # ---- commands
    def handle(self, c):
        t = c["type"]
        if t == "reg_set":
            reg = {"robot": _yaw_only([float(x) for x in c["robot"]]), "world": _yaw_only([float(x) for x in c["world"]])}
            cam = c.get("camera") or self.state["registration"].get("camera")
            if cam:                                   # the robot's camera (scene frame, any orientation)
                reg["camera"] = [float(x) for x in cam]
            self.state["registration"] = reg
            REG_FILE.write_text(json.dumps(reg))
        elif t == "grasp_select":
            self._select(c)
        elif t == "grasp_plan":
            self._plan()
        elif t == "grasp_exec":
            self._execute()
        elif t in ("grasp_go", "grasp_pick", "grasp_place"):
            if t in ("grasp_pick", "grasp_place"):  # right-click / gizmo move on an object: select + plan + execute
                if self._thread and self._thread.is_alive():
                    raise RuntimeError("이미 실행 중입니다")
                self._select(c)
            D = None
            if t == "grasp_place":
                Dw = np.array(c["D_world"], float).reshape(4, 4)
                T_bw = self._sel[4]
                D = T_bw @ Dw @ np.linalg.inv(T_bw)   # the same rigid move, in the base frame
            self._plan(D)
            if self.state["stage"] != "planned":
                raise RuntimeError(self.state.get("error") or "파지 계획 실패")
            sim = self.state["plan"].get("sim") or {}
            if sim.get("verdict") == "안 들림":       # [ours] the prediction says this would only knock the object
                self.state["error"] = "물리 예측: 안 들림 → 자동 실행 안 함 (파지 탭 '실행'으로 강제 실행 가능)"
                raise RuntimeError(self.state["error"])
            pl = sim.get("place")
            if D is not None and pl is not None and not pl.get("ok"):
                self.state["error"] = (f"물리 예측: 목표에서 {pl['err_mm']} mm / {pl['err_deg']}° 어긋남 → 자동 실행 안 함 "
                                       "(파지 탭 '실행'으로 강제 실행 가능)")
                raise RuntimeError(self.state["error"])
            self._execute()
        elif t == "grasp_cancel":
            self._abort.set()
            self._vmoved.clear()
            self.state["virtual_moved"] = {}
            vc = getattr(self.cam, "_virtual", None) if self.cam else None
            if vc is not None and (vc.offset or getattr(vc, "yaw", None) or getattr(vc, "rot", None)):   # back to the start
                vc.offset.clear()
                getattr(vc, "yaw", {}).clear()
                getattr(vc, "rot", {}).clear()
                self._refresh_changed()
            if hasattr(getattr(self.arm, "bus", None), "stall"):
                self.arm.bus.stall.clear()
            self.state.update(stage="idle", plan=None, object=None, error=None, exec=None)
            self._sel = self._obj_world = self._obj_mesh = None
            self.state["obj_version"] += 1

    def _select(self, c):
        from planner import segment_object, SEG_ABOVE_M, SEG_ABOVE_RAW_M
        M_sw = np.array(c["M_scene_world"], float).reshape(4, 4)
        M_sb = np.array(c["M_scene_base"], float).reshape(4, 4)
        T_bw = np.linalg.inv(M_sb) @ M_sw
        up_base = np.linalg.inv(M_sb)[:3, :3] @ np.array([0, 1.0, 0])    # scene y = up
        v, tris, src = self._scene_points()
        pb = v @ T_bw[:3, :3].T + T_bw[:3, 3]
        click = T_bw[:3, :3] @ np.array(c["p_world"], float) + T_bw[:3, 3]
        self._seg = None
        rc = getattr(self.cam, "recon", None)
        self._sel_oid = int(c["obj_id"]) if c.get("obj_id") is not None else None
        geo = rc.object_geometry(self._sel_oid) if self._sel_oid is not None and rc is not None else None
        if geo is not None and self._sel_oid in self._vmoved:
            # moved earlier by the virtual robot (the real one is still where the camera sees it): plan on the moved
            # copy, and the real one's depth points are not an obstacle
            from scipy.spatial import cKDTree
            Mv = self._vmoved[self._sel_oid]
            Vw0, Tw0, raw0, sup0 = geo
            dd, _ = cKDTree(np.vstack([Vw0, raw0])).query(v, distance_upper_bound=0.008)
            v = v[~np.isfinite(dd)]
            pb = v @ T_bw[:3, :3].T + T_bw[:3, 3]
            geo = (Vw0 @ Mv[:3, :3].T + Mv[:3, 3], Tw0, raw0 @ Mv[:3, :3].T + Mv[:3, 3], sup0 + Mv[1, 3])
        if geo is not None:
            # [ours] the reconstruction's own split object (completed back side, separate from the board it stands on):
            # its completed surface is what the gripper closes on; scene points on it leave the collision set
            from planner import segment_given
            Vw, Tw, raw, sup_y = geo
            ow = np.vstack([Vw, raw])
            ow = ow[np.random.default_rng(0).choice(len(ow), min(len(ow), 4000), replace=False)]
            ob_ = ow @ T_bw[:3, :3].T + T_bw[:3, 3]
            cw = ow.mean(axis=0)
            spb = T_bw[:3, :3] @ np.array([cw[0], sup_y, cw[2]]) + T_bw[:3, 3]
            pb, obj_idx, support, env_idx = segment_given(pb, ob_, spb, up_base)
            click = ob_.mean(axis=0)
            v = (pb - T_bw[:3, 3]) @ T_bw[:3, :3]                          # world copy of the merged points
            self._seg = (obj_idx, support, env_idx, (Vw @ T_bw[:3, :3].T + T_bw[:3, 3], Tw))   # + completed mesh, base
            tris, src = None, f"복원 객체 #{c['obj_id']} (완성 메시)"
        try:
            if self._seg is not None:
                obj_idx, support = self._seg[0], self._seg[1]
            else:
                obj_idx, support, _ = segment_object(pb, click, up_base, SEG_ABOVE_M if tris is not None else SEG_ABOVE_RAW_M)
        except Exception as e:
            self.state.update(stage="idle", error=str(e), object=None)
            raise
        self._sel = (pb, tris, click, up_base, T_bw)
        self._obj_world = v[obj_idx].astype(np.float32)
        ob = pb[obj_idx]
        obj = {"points": int(len(obj_idx)), "source": src, "size_mm": (np.ptp(ob, axis=0) * 1000).round(0).tolist(),
               "centre_base": ob.mean(axis=0).round(3).tolist(), "support_h": round(support, 3), "obj_id": self._sel_oid}
        self._obj_mesh = None
        rc = getattr(self.cam, "recon", None)
        if rc is not None and getattr(rc, "_kf", None):
            # [ours] per-object TSDF in the object's box: ARKit world y is gravity-up, the box bottom sits just above
            # the support (segmentation starts SEG_ABOVE above it) so the floor stays out
            from recon import OBJ_MARGIN_M
            w = self._obj_world.astype(float)
            lo, hi = w.min(axis=0) - OBJ_MARGIN_M, w.max(axis=0) + OBJ_MARGIN_M
            lo[1] = w[:, 1].min() - (SEG_ABOVE_RAW_M if tris is None else SEG_ABOVE_M) + 0.002
            try:
                blob, st, vw = rc.object_tsdf(lo, hi)
                vb = vw @ T_bw[:3, :3].T + T_bw[:3, 3] if len(vw) else vw
                st["size_mm"] = (np.ptp(vb, axis=0) * 1000).round(0).tolist() if len(vb) else None
                self._obj_mesh = blob
                obj["tsdf"] = st
            except Exception as e:
                obj["tsdf"] = {"error": str(e)}
        self.state.update(stage="selected", error=None, plan=None, object=obj)
        self.state["obj_version"] += 1

    def object_mesh_bytes(self):
        return self._obj_mesh or b""

    def object_bytes(self):
        o = self._obj_world
        return b"" if o is None else np.array([len(o)], np.uint32).tobytes() + o.tobytes()

    SIM_TRIES = 5          # [ours] feasible grasps checked by the physics prediction before giving up

    def _plan(self, D=None):
        """[TACO candidates → ours: IK, straight paths] then the physics prediction picks among the first SIM_TRIES
        feasible grasps: the first one predicted to be held, else the one that lifted the object most.
        D (4×4, base frame): also place the object at D · its pose — a grasp counts only if its place path exists
        (planner.plan_place), and the prediction runs the whole pick-and-place."""
        from planner import plans, plan_place, _drop_outliers
        from sim import simulate
        if self._sel is None:
            raise RuntimeError("먼저 물체를 선택하세요")
        pb, tris, click, up, T_bw = self._sel
        self.state.update(stage="planning", error=None)
        q0 = self.q_now()
        t0 = time.time()
        best, tried, place_fail, failed = None, [], [], []
        for r in plans(pb, tris, click, up, q0, seg=self._seg):
            obj_idx = r.pop("obj_idx", None)
            if "error" in r:
                if best is None:
                    self.state.update(stage="selected", error=r["error"], plan=r)
                    return
                break
            r["q_now"] = q0.round(4).tolist()
            if D is not None:
                r["place"] = plan_place(r, D, pb, obj_idx, up)
                if "error" in r["place"]:
                    place_fail.append(r["place"]["tried"][-1:] if r["place"]["tried"] else [])
                    failed.append((r, obj_idx))
                    if len(place_fail) >= 12:
                        break
                    continue
            self._simulate(r, obj_idx, pb, q0)
            tried.append({"width_mm": r["width_mm"], "roll_deg": r["roll_deg"], "verdict": r["sim"].get("verdict", "오류"),
                          "lift_mm": r["sim"].get("lift_mm")})
            good = r["sim"].get("held") and (D is None or (r["sim"].get("place") or {}).get("ok"))
            if best is None or r["sim"].get("lift_mm", -1e9) > best["sim"].get("lift_mm", -1e9):
                best = r
            if good or len(tried) >= self.SIM_TRIES:
                if good:
                    best = r
                break
        if best is None and D is not None and failed:
            # [ours] the target is out of reach (or blocked): the closest pose that works instead — the move cut back
            # along the way from where the object is to the target (centre on the straight line, turn slerped),
            # largest fraction that still plans, per grasp; the best grasp's goes to the prediction
            best = self._closest_place(failed, D, pb, up, q0)
            if best is not None:
                r, obj_idx = best
                self._simulate(r, obj_idx, pb, q0)
                tried.append({"width_mm": r["width_mm"], "roll_deg": r["roll_deg"], "verdict": r["sim"].get("verdict", "오류"),
                              "lift_mm": r["sim"].get("lift_mm")})
                best = r
        if best is None:                              # grasps exist, but none can place the object there
            from collections import Counter
            why = Counter()
            for f in place_fail:
                for t in f:
                    w = t["fail"]
                    why["목표가 다른 물체와 겹침" if w.startswith("목표 자리 충돌: 물체") else "놓을 때 집게가 닿음" if "집게" in w and "목표" in w
                        else "옮기는 길에 충돌" if w.startswith("옮기는 길") else "팔이 닿지 않음 (IK)" if "IK" in w else "수직 하강 불가"] += 1
            top = " · ".join(f"{k} {n}" for k, n in why.most_common(3))
            self.state.update(stage="selected", error=f"목표 위치에 놓을 수 있는 파지가 없습니다 — 파지 후보 {len(place_fail)}개: {top}",
                              plan={"place_fail": place_fail})
            return
        best["sim_tried"] = tried
        best["total_ms"] = round((time.time() - t0) * 1000)
        self.state.update(stage="planned", plan=best)

    def _simulate(self, r, obj_idx, pb, q0):
        from planner import _drop_outliers
        from sim import simulate
        try:
            t1 = time.time()
            sz = r.get("support_z")
            sz = float(pb[obj_idx][:, 2].min() - 0.008) if sz is None else sz
            r["sim"] = simulate(r, _drop_outliers(pb[obj_idx]), sz, q0, self._torque_frac(),
                                obj_mesh=self._seg[3] if self._seg is not None and len(self._seg) > 3 else None)
            r["sim"]["ms"] = round((time.time() - t1) * 1000)
        except Exception as e:
            r["sim"] = {"error": str(e), "lift_mm": -1e9}

    APPROX_GRASPS, APPROX_STEPS = 6, 7

    def _closest_place(self, failed, D, pb, up, q0):
        """Largest fraction a ∈ (0, 1) of the requested move that plans (bisection, APPROX_STEPS), over the first
        APPROX_GRASPS feasible grasps. → (plan with place + place["approx"], obj_idx) or None."""
        from planner import plan_place
        from scipy.spatial.transform import Rotation, Slerp
        best = None
        for r, obj_idx in failed[:self.APPROX_GRASPS]:
            c = pb[obj_idx].mean(axis=0)
            c_t = D[:3, :3] @ c + D[:3, 3]
            sl = Slerp([0, 1], Rotation.from_matrix([np.eye(3), D[:3, :3]]))

            def D_at(a):
                Ra = sl([a]).as_matrix()[0]
                M = np.eye(4)
                M[:3, :3] = Ra
                M[:3, 3] = c + a * (c_t - c) - Ra @ c
                return M
            def search(make):
                lo, hi, got = 0.0, 1.0, None
                for _ in range(self.APPROX_STEPS):
                    a = (lo + hi) / 2
                    Da = make(a)
                    pl = plan_place(r, Da, pb, obj_idx, up)
                    if "error" in pl:
                        hi = a
                    else:
                        lo, got = a, (a, pl, Da)
                return got
            # (1) part of the way along the move; (2) the target pulled toward the shoulder (the arm's reach is a ball
            # about joint 2): for a target beyond the robot, (2) gets much closer than (1)
            sh = SHOULDER_BASE
            far = c_t - sh

            def D_pull(a):                   # centre on the shoulder→target line, 50 … 100 % of the way; turn as asked
                M = D.copy()
                M[:3, 3] = sh + far * (0.5 + 0.5 * a) - D[:3, :3] @ c
                return M
            for got in (search(D_at), search(D_pull)):
                if not got:
                    continue
                Da = got[2]
                miss = float(np.linalg.norm(c_t - (Da[:3, :3] @ c + Da[:3, 3])))
                if best is None or miss < best[0]:
                    best = (miss, got, r, obj_idx)
            if best and best[0] < 0.01:
                break
        if best is None:
            return None
        miss, (a, pl, _), r, obj_idx = best
        pl["approx"] = {"fraction": round(a, 3), "miss_mm": round(miss * 1000), "requested_D": np.round(D, 5).tolist()}
        r = dict(r)
        r["place"] = pl
        return r, obj_idx

    def _torque_frac(self):
        """Fraction of stall torque each real servo may use: RAM torque_limit capped by the console (server.TORQUE_CAP)."""
        st, cap = self.arm.state.get("settings", {}), self.arm.state.get("torque_cap", {})
        out = []
        for mid in range(1, 9):
            tl = (self.arm.state.get("ram", {}).get(mid) or {}).get("torque_limit") or st.get("torque_limit", 400)
            out.append(min(tl, cap.get(mid, 1000)) / 1000)
        return out

    def _execute(self):
        p = self.state["plan"]
        if not p or "q_grasp" not in p:
            raise RuntimeError("실행할 계획이 없습니다")
        if self._thread and self._thread.is_alive():
            raise RuntimeError("이미 실행 중입니다")
        off = [m for m in range(1, 9) if not self.arm.state["ram"].get(m, {}).get("torque")]
        if off:
            raise RuntimeError(f"토크가 꺼진 모터가 있습니다: {off} — 전체 ON 후 실행하세요")
        from motion import segments
        seq = segments(self.q_now(), p)          # same timeline the physics prediction used
        self._abort.clear()
        self._thread = threading.Thread(target=self._run, args=(seq,), daemon=True)
        self._thread.start()

    def _refresh_changed(self):
        rc = getattr(self.cam, "recon", None) if self.cam else None
        if rc is not None and self._changed_box is not None:
            rc.mark_changed(*self._changed_box)

    def abort(self):
        self._abort.set()

    def _virtual_world(self):
        """Virtual robot + virtual camera: (bus, camera, scene object index, base→ARKit-world transform) so the
        execution can follow the physics prediction (the virtual scene has no physics of its own)."""
        from virtual import VirtualBus
        bus, vc = getattr(self.arm, "bus", None), getattr(self.cam, "_virtual", None) if self.cam else None
        if not isinstance(bus, VirtualBus) or vc is None or self._sel is None or self._obj_world is None:
            return None
        centres = vc.object_centres()
        if not centres:
            return None
        c = self._obj_world.mean(axis=0)
        idx = min(centres, key=lambda k: np.linalg.norm((centres[k] - c)[[0, 2]]))
        return bus, vc, idx, np.linalg.inv(self._sel[4])

    def _run(self, seq):
        from motion import at, PAUSE_S
        sim = (self.state.get("plan") or {}).get("sim") or {}
        frames = np.array(sim.get("frames") or [])
        vw = self._virtual_world() if len(frames) else None
        if vw:
            bus, vc, idx, T_wb = vw
            bus.stall.pop(8, None)

            def release():                      # jaws opened: the object drops back onto the support where it is
                off = vc.offset.get(idx)
                if off is not None:
                    vc.offset[idx] = np.array([off[0], 0.0, off[2]])
                    self._refresh_changed()
            bus.on_release = release
            p0 = T_wb[:3, :3] @ frames[0, 1:4] + T_wb[:3, 3]
            base_off = vc.offset.get(idx, np.zeros(3)).copy()
            base_rot = np.asarray(vc.rot.get(idx, np.eye(3))) if hasattr(vc, "rot") else None
            R0 = _quat_R(frames[0, 4:8])
            # jaws blocked by the object in the simulation: the gripper angle it ends at (2.2 = closed on nothing)
            stall_q = float(frames[-1, 15]) if frames[-1, 15] < 2.1 else None
        t_seg = 0.0                     # nominal time on the motion.py timeline (= the simulation's clock)
        self._changed_box = None
        if self._obj_world is not None:  # the grasp will move this object: region to refresh in the reconstruction
            w = self._obj_world.astype(float)
            lo, hi = w.min(axis=0) - 0.03, w.max(axis=0) + 0.03
            lo[1] = w[:, 1].min() - 0.02
            hi[1] += 0.12                  # it rises up to the lift height (+ the gripper above it)
            pl = (self.state.get("plan") or {}).get("place")
            if pl and "D" in pl and self._sel is not None:   # placed elsewhere: the target region changes too
                T_wb = np.linalg.inv(self._sel[4])
                Dw = T_wb @ np.array(pl["D"]) @ self._sel[4]
                wt = w @ Dw[:3, :3].T + Dw[:3, 3]
                lo = np.minimum(lo, wt.min(axis=0) - 0.03)
                hi = np.maximum(hi, wt.max(axis=0) + 0.03)
            self._changed_box = (lo, hi)
        try:
            for i, (W, dur, label) in enumerate(seq):
                self.state["exec"] = {"step": i + 1, "of": len(seq), "label": label, "duration_s": round(dur, 1)}
                self.state["stage"] = "executing"
                n = max(2, int(dur * CTRL_HZ))
                for k in range(1, n + 1):
                    if self._abort.is_set():
                        raise RuntimeError("중단됨")
                    q = at(W, k / n)                                  # smoothstep along the waypoint path
                    self.arm.cmds.put({"type": "goal_many", "raws": {m: self.raw_of(m, q[m - 1]) for m in range(1, 9)}})
                    t_nom = t_seg + dur * k / n
                    self.state["exec"]["t"] = round(t_nom, 2)
                    if vw:                                            # virtual scene follows the prediction
                        f = frames[min(len(frames) - 1, int(t_nom * 20))]
                        vc.offset[idx] = base_off + (T_wb[:3, :3] @ f[1:4] + T_wb[:3, 3]) - p0
                        if base_rot is not None:                      # any turn (6-DoF target), world frame
                            Rw = T_wb[:3, :3] @ (_quat_R(f[4:8]) @ R0.T) @ T_wb[:3, :3].T
                            vc.rot[idx] = Rw @ base_rot
                        if stall_q is not None and label == "집게 닫기" and 8 not in bus.stall:
                            lim = self.raw_of(8, stall_q)            # jaws stop on the object, as in the simulation
                            bus.stall[8] = (lim, int(np.sign(self.raw_of(8, 2.2) - self.raw_of(8, 0.0))))
                    time.sleep(1.0 / CTRL_HZ)
                time.sleep(PAUSE_S)
                t_seg += dur + PAUSE_S
            if vw and (self.state.get("plan") or {}).get("place") and not self.state["plan"]["place"].get("hold"):
                off = vc.offset.get(idx)                     # placed: resting on the same support as before
                if off is not None:
                    vc.offset[idx] = np.array([off[0], base_off[1], off[2]])
            if not vw and len(frames) and self._sel_oid is not None and self._seg is not None:
                # [ours] real camera, virtual robot: the real object did not move — remember where the prediction left
                # it (world frame), so the console shows it there and the next plan starts from there
                def pose(f):
                    P = np.eye(4)
                    P[:3, :3], P[:3, 3] = _quat_R(f[4:8]), f[1:4]
                    return P
                T_bw = self._sel[4]
                Db = pose(frames[-1]) @ np.linalg.inv(pose(frames[0]))
                Dw = np.linalg.inv(T_bw) @ Db @ T_bw
                self._vmoved[self._sel_oid] = Dw @ self._vmoved.get(self._sel_oid, np.eye(4))
                self.state["virtual_moved"] = {str(k): np.round(M, 5).ravel().tolist() for k, M in self._vmoved.items()}
            self.state.update(stage="done", exec={"label": "완료"})
            self._refresh_changed()
        except Exception as e:
            self.state.update(stage="planned", error=str(e), exec=None)
