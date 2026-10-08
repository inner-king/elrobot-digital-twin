"""[ours] Live physics of the virtual world: the solid (coloured) robot and objects, as opposed to the semi-transparent
prediction (sim.simulate) that a plan replays.

Input : the reconstruction's split objects (completed meshes, ARKit world), the robot ↔ world registration (T_base_world),
        the support height, the virtual servo bus (present joint positions, 20 Hz)
Output: every object's pose, published ≈20 Hz:
          virtual camera (kitchen / boxes scene)  → its objects' offset / rot, so the reconstruction sees them move
          real camera + virtual robot             → manager._vmoved (the console shows the moved copy)
        and the jaws held back on an object (VirtualBus.stall), like the real servo stalling.

Model : elrobot.xml (force-limited position servos, decomposed jaw parts — sim.arm_xml), a support plane colliding with
        the objects only (see sim.py), one free body per object = convex pieces of its completed mesh (sim.decompose),
        mass from its volume. Physics settings as sim.py (TACO's): elliptic cone, noslip, impratio 10.
"""
import threading
import time

import numpy as np

DT = 0.002
ROLL_MU = 0.002            # rolling friction (condim 6): a round scanned object does not roll away on its facets
PUBLISH_HZ = 20


def _quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class VirtualWorld:
    def __init__(self, gm):
        self.gm = gm                     # GraspManager: arm, cam, joint mapping, registration, virtual moves
        self.status = {"running": False}
        self._thread = None
        self._stop = threading.Event()

    # ---- build
    def build(self, T_bw):
        """T_bw: ARKit world → robot base (4×4)."""
        import mujoco
        from sim import arm_xml, decompose, DENSITY, MASS_RANGE, OBJ_MU, JAW_MU, STALL_NM
        rc = getattr(self.gm.cam, "recon", None) if self.gm.cam else None
        objs = (rc.status.get("objects") or []) if rc is not None else []
        if not objs:
            raise RuntimeError("복원된 물체가 없습니다 — 복원을 먼저 하세요")
        t0 = time.time()
        spec = mujoco.MjSpec.from_string(arm_xml())
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.noslip_iterations = 10
        spec.option.impratio = 10.0
        spec.memory = 256 * 1024 * 1024      # many contacts (≈250 convex pieces): the default 16 MB arena overflowed
        R, t = T_bw[:3, :3], T_bw[:3, 3]
        fy = rc.floor_y if rc.floor_y is not None else min(o["centre"][1] for o in objs) - 0.05
        sup_z = float((R @ np.array([0, fy, 0]) + t)[2])
        spec.worldbody.add_geom(name="support", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.01], pos=[0, 0, sup_z],
                                friction=[OBJ_MU, 0.005, 0.0001], contype=2, conaffinity=2)
        self._bodies = []                # (recon id, body name, centre (base) at start)
        for o in objs:
            geo = rc.object_geometry(o["id"])
            if geo is None:
                continue
            Vw, Tw, _, _ = geo
            Vw = np.asarray(Vw, float)
            Mv = self.gm._vmoved.get(o["id"])       # already moved by the virtual robot: start from there
            if Mv is not None:
                Vw = Vw @ Mv[:3, :3].T + Mv[:3, 3]
            Vb = Vw @ R.T + t
            try:
                parts, counts, vol = decompose(Vb, np.asarray(Tw))
            except Exception:
                continue
            mass = float(np.clip(vol * DENSITY, *MASS_RANGE))
            c = np.vstack([p.vertices for p in parts]).mean(axis=0)
            name = f"obj{o['id']}"
            body = spec.worldbody.add_body(name=name, pos=c.tolist())
            body.add_freejoint()
            for j, (p_, n_) in enumerate(zip(parts, counts)):
                spec.add_mesh(name=f"{name}_m{j}", uservert=(p_.vertices - c).ravel().tolist(), userface=p_.faces.ravel().tolist())
                body.add_geom(name=f"{name}_g{j}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_m{j}",
                              mass=mass * n_ / sum(counts), friction=[OBJ_MU, 0.005, ROLL_MU], condim=6, contype=3, conaffinity=3)
            self._bodies.append((o["id"], name, c))
        if not self._bodies:
            raise RuntimeError("물리 모델로 만들 수 있는 물체가 없습니다")
        m = spec.compile()
        m.opt.timestep = DT
        jaw = {m.body(n).id for n in ("Gripper_Jaw_01_v1_1", "Gripper_Jaw_02_v1_1")}
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in jaw:
                m.geom_friction[g, 0] = JAW_MU
        acts = [m.actuator(f"motor_0{i}").id for i in range(1, 9)]
        tf = np.clip(np.asarray(self.gm._torque_frac(), float), 0.02, 1.0)
        for a, f in zip(acts, tf):
            m.actuator_forcerange[a] = [-STALL_NM * f, STALL_NM * f]
        d = mujoco.MjData(m)
        q = self.gm.q_now()
        self._jq = [m.jnt_qposadr[m.joint(f"rev_motor_0{i}").id] for i in range(1, 9)]
        for i, a in enumerate(self._jq):
            d.qpos[a] = q[i]
        d.qpos[m.jnt_qposadr[m.joint("rev_motor_08_1").id]] = -0.0115 * q[7]
        d.qpos[m.jnt_qposadr[m.joint("rev_motor_08_2").id]] = 0.0115 * q[7]
        d.ctrl[acts] = q
        mujoco.mj_forward(m, d)
        self._depenetrate(m, d)
        self.m, self.d, self.acts, self.jaw = m, d, acts, jaw
        self.T_bw = T_bw.copy()
        self._oid = {m.body(n).id: oid for oid, n, _ in self._bodies}
        self._pose0 = {oid: (d.xpos[m.body(n).id].copy(), d.xmat[m.body(n).id].reshape(3, 3).copy()) for oid, n, _ in self._bodies}
        self._vm0 = {oid: self.gm._vmoved.get(oid, np.eye(4)).copy() for oid, _, _ in self._bodies}
        self._map_virtual()
        self.status.update(objects=len(self._bodies), geoms=int(m.ngeom), build_ms=round((time.time() - t0) * 1000),
                           support_z=round(sup_z, 4), error=None)

    @staticmethod
    def _depenetrate(m, d, rounds=6):
        """[ours] Objects completed one by one overlap where they touch (items 3 mm into the board they stand on):
        the upper one of each overlapping pair goes up by the overlap, so they start touching instead of being shot
        apart (a strawberry rolled 70 mm off the board in 3 s)."""
        import mujoco
        obj = {b for b in range(m.nbody) if m.body(b).name.startswith("obj")}
        for _ in range(rounds):
            lift = {}
            for i in range(d.ncon):
                c = d.contact[i]
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if c.dist >= -0.0003 or not ({b1, b2} <= obj | {0}):
                    continue
                if b1 == 0 or b2 == 0:                 # into the support plane: up
                    b = b2 if b1 == 0 else b1
                else:
                    b = b1 if d.xpos[b1][2] > d.xpos[b2][2] else b2
                lift[b] = max(lift.get(b, 0.0), -c.dist + 0.0003)
            if not lift:
                break
            for b, h in lift.items():
                d.qpos[m.jnt_qposadr[m.body_jntadr[b]] + 2] += h
            mujoco.mj_forward(m, d)

    def _map_virtual(self):
        """recon object → the virtual camera's own object (nearest centre, top view), with its pose at the start"""
        vc = getattr(self.gm.cam, "_virtual", None) if self.gm.cam else None
        self._vmap = {}
        if vc is None:
            return
        rc = self.gm.cam.recon
        cen = vc.object_centres()
        for o in rc.status.get("objects") or []:
            if not cen:
                break
            c = np.array(o["centre"])
            k = min(cen, key=lambda i: np.linalg.norm((cen[i] - c)[[0, 2]]))
            if np.linalg.norm((cen[k] - c)[[0, 2]]) < 0.06:
                self._vmap[o["id"]] = (k, cen[k].copy(), np.asarray(vc.offset.get(k, np.zeros(3)), float).copy(),
                                       np.asarray(vc.rot.get(k, np.eye(3)) if hasattr(vc, "rot") else np.eye(3)).copy())
        self._vc = vc

    # ---- run
    def start(self, T_bw):
        self.stop()
        self.build(T_bw)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.status["running"] = True

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        self.status["running"] = False

    def _loop(self):
        import mujoco
        m, d = self.m, self.d
        n_pub = int(round(1 / (PUBLISH_HZ * DT)))
        t_wall, steps, k = time.time(), 0, 0
        while not self._stop.is_set():
            try:
                q = self.gm.q_now()                    # the virtual servos' present positions
            except Exception:
                time.sleep(0.05)
                continue
            d.ctrl[self.acts] = q
            for _ in range(n_pub):
                mujoco.mj_step(m, d)
            steps += n_pub
            k += 1
            self._publish()
            # real time: the simulated 1/PUBLISH_HZ s should not run ahead of the wall clock
            ahead = steps * DT - (time.time() - t_wall)
            if ahead > 0:
                time.sleep(ahead)
            if k % PUBLISH_HZ == 0:
                self.status["rt_factor"] = round(steps * DT / max(time.time() - t_wall, 1e-6), 2)

    def _publish(self):
        m, d = self.m, self.d
        T_wb = np.linalg.inv(self.T_bw)
        moved = {}
        for oid, name, _ in self._bodies:
            b = m.body(name).id
            p0, R0 = self._pose0[oid]
            Rb = d.xmat[b].reshape(3, 3) @ R0.T                       # base-frame move since the start
            Db = np.eye(4)
            Db[:3, :3], Db[:3, 3] = Rb, d.xpos[b] - Rb @ p0
            Dw = T_wb @ Db @ self.T_bw                                # same move, ARKit world
            moved[oid] = Dw @ self._vm0[oid]
            v = self._vmap.get(oid)
            if v is not None:                                         # virtual camera: its object follows
                kidx, c0, off0, rot0 = v
                self._vc.offset[kidx] = off0 + (Dw[:3, :3] @ c0 + Dw[:3, 3]) - c0
                if hasattr(self._vc, "rot"):
                    self._vc.rot[kidx] = Dw[:3, :3] @ rot0
        if not self._vmap:                                            # real camera: the console shows the moved copies
            self.gm._vmoved.update(moved)
            self.gm.state["virtual_moved"] = {str(k): np.round(M, 5).ravel().tolist() for k, M in self.gm._vmoved.items()}
        self._jaw_stall()

    def _jaw_stall(self):
        """jaws closing on an object stop where the physics stops them (the virtual servo would close through it)"""
        from virtual import VirtualBus
        bus = getattr(self.gm.arm, "bus", None)
        if not isinstance(bus, VirtualBus):
            return
        m, d = self.m, self.d
        touching = False
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if (b1 in self.jaw and b2 in self._oid) or (b2 in self.jaw and b1 in self._oid):
                touching = True
                break
        q8 = float(d.qpos[self._jq[7]])
        goal = self.gm.q_now()[7]
        if touching and goal > q8 + 0.05 and 8 not in bus.stall:      # commanded further closed than the jaws can go
            lim = self.gm.raw_of(8, q8)
            bus.stall[8] = (lim, int(np.sign(self.gm.raw_of(8, 2.2) - self.gm.raw_of(8, 0.0))))
