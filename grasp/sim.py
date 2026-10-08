"""Predict what the object does during a planned grasp (MuJoCo, simple physics).

Input : plan from planner.py (q_now, q_pre, q_grasp, q_lift), object points (base frame, m), support height (m)
Output: object pose track at 20 Hz on the executor's own timeline + a verdict:
        {"frames": [[t, x, y, z, qw, qx, qy, qz, q1..q8], ...], "lift_mm", "held", "verdict", "jaw_contacts", "hull": {...}}
        (object pose in the base frame; q = the commanded joints at that instant, so arm and object replay in sync)

Joints are force-limited position servos (elrobot.xml kp 18.22, forcerange = stall 3.43 N·m × the console's torque
limit), so the jaws stall on the object instead of being forced shut (the full 3.43 N·m through the 11.5 mm rack
squeezed ≈ 300 N per jaw and shot boxes out).
Physics settings are TACO's (track_rl/hold_check.py, settled 2026-09-26): elliptic friction cone, 10 noslip
iterations, impratio 10, jaw friction 2 — the default pyramidal cone ejects a squeezed box at 16 m/s.
[ours] object = convex hull of the segmented points, 0.1 kg, μ = 1 (unknown in a live scene); support = a plane;
       arm joints follow the plan kinematically (the real servos track position), the gripper closes through its
       actuator so the squeeze force stays capped (as in TACO's sim: close past contact, force capped).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARM_DIR = ROOT / "elrobot_mujoco"
MJCF = ARM_DIR / "elrobot.xml"


def arm_xml():
    """elrobot.xml with the jaw fingers' single convex collision hull replaced by their decomposed parts.

    The hull fills the L-shaped finger's inner corner (contacts up to 9 mm outside the pad, TACO 2026-09-27);
    same substitution as simtoolreal/env.py _xml, using the cached elrobot_mujoco/assets/*_part*.stl."""
    import json
    import re
    src = MJCF.read_text().replace('meshdir="assets"', f'meshdir="{ARM_DIR / "assets"}"')
    spec = json.loads((ROOT / "taco_viewer" / "robot" / "gripper.json").read_text())
    meshes = ""
    for part in spec["parts"]:
        if "slide" not in part:
            continue
        stem = part["mesh"][:-4]
        paths = sorted((ARM_DIR / "assets").glob(f"{stem}_part*.stl"))
        if not paths:
            continue                                   # no cache: keep the hull
        meshes += "".join(f'<mesh name="{stem}_c{i}" file="{p}" scale="0.001 0.001 0.001"/>' for i, p in enumerate(paths))
        m = re.search(rf'<geom class="collision" mesh="{stem}_collision" pos="([^"]+)" />', src)
        src = src.replace(m.group(0), "".join(
            f'<geom class="collision" mesh="{stem}_c{i}" pos="{m.group(1)}" friction="1 0.02 0.001" condim="4" '
            f'solref="0.004 1" solimp="0.95 0.99 0.001"/>' for i in range(len(paths))))
    return src.replace("</asset>", meshes + "</asset>", 1)
OBJ_MASS, OBJ_MU, JAW_MU = 0.10, 1.0, 2.0
DT, OUT_HZ = 0.002, 20


def timeline(q0, plan):
    """The executor's own segments (motion.py), as q(t) over the whole run."""
    from motion import segments, at, PAUSE_S
    segs = segments(q0, plan)
    def q_at(t):
        for W, dur, _ in segs:
            if t < dur:
                return at(W, t / dur)
            t -= dur
            if t < PAUSE_S:
                return W[-1]
            t -= PAUSE_S
        return segs[-1][0][-1]
    total = sum(d + PAUSE_S for _, d, _ in segs)
    return q_at, total, [(W[0], W[-1], d, label) for W, d, label in segs]


STALL_NM = 3.43                 # ST3215 stall torque in elrobot.xml (actuator forcerange)


def simulate(plan, obj_points, support_h, q0, torque_frac=None):
    """torque_frac: 8 fractions of the stall torque the real servos are allowed (torque_limit register ∧ console cap).
    Every joint is a force-limited position servo (elrobot.xml kp), so the arm sags/stalls and the jaws stop on the
    object like the real ones; frames record the *simulated* joints, the jaw contact force and the gripper torque."""
    import mujoco
    import trimesh
    # the segmentation cuts the bottom few mm off; close the solid down to the support (the object rests on it),
    # otherwise the slanted hull bottom rocks and the body rolls away before the gripper arrives
    base = np.asarray(obj_points, float).copy()
    base[:, 2] = support_h
    hull = trimesh.convex.convex_hull(np.vstack([obj_points, base]))
    c = hull.centroid
    spec = mujoco.MjSpec.from_string(arm_xml())
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 10
    spec.option.impratio = 10.0
    spec.add_mesh(name="obj_hull", uservert=(hull.vertices - c).ravel().tolist(), userface=hull.faces.ravel().tolist())
    # [ours] the support collides with the object only (contype/conaffinity bit 2): it is an infinite plane, and at
    # cutting-board height (12 mm) the robot base standing on the floor sat inside it — the friction held joint 1 back
    # 25° and the jaws closed 10 cm beside the object. The arm vs. the support is the planner's scene-point check.
    spec.worldbody.add_geom(name="support", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], pos=[0, 0, support_h],
                            friction=[OBJ_MU, 0.005, 0.0001], contype=2, conaffinity=2)
    body = spec.worldbody.add_body(name="object", pos=c.tolist())
    body.add_freejoint()
    body.add_geom(name="object", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="obj_hull", mass=OBJ_MASS,
                  friction=[OBJ_MU, 0.005, 0.0001], condim=4, contype=3, conaffinity=3)
    m = spec.compile()
    m.opt.timestep = DT
    jaw_bodies = {m.body(n).id for n in ("Gripper_Jaw_01_v1_1", "Gripper_Jaw_02_v1_1")}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in jaw_bodies:
            m.geom_friction[g, 0] = JAW_MU
    d = mujoco.MjData(m)
    q_at, total, segs = timeline(q0, plan)
    acts = [m.actuator(f"motor_0{i}").id for i in range(1, 9)]
    tf = np.clip(np.asarray(torque_frac if torque_frac is not None else [0.4] * 7 + [0.25], float), 0.02, 1.0)
    for a, f in zip(acts, tf):
        m.actuator_forcerange[a] = [-STALL_NM * f, STALL_NM * f]
    jq = [m.jnt_qposadr[m.joint(f"rev_motor_0{i}").id] for i in range(1, 9)]
    g_act = acts[7]
    oid = m.body("object").id
    q_init = q_at(0)
    for i, a in enumerate(jq):
        d.qpos[a] = q_init[i]
    d.qpos[m.jnt_qposadr[m.joint("rev_motor_08_1").id]] = -0.0115 * q_init[7]
    d.qpos[m.jnt_qposadr[m.joint("rev_motor_08_2").id]] = 0.0115 * q_init[7]
    d.ctrl[acts] = q_init
    mujoco.mj_forward(m, d)
    for _ in range(int(0.3 / DT)):                     # settle: hull on the support, arm sagging into its servos
        mujoco.mj_step(m, d)
    z0 = float(d.xpos[oid][2])
    frames, t, nxt = [], 0.0, 0.0
    seg_ends = np.cumsum([dur + 0.4 for _, _, dur, _ in segs])
    seg_log, si, touched, first, at_lift = [], 0, set(), None, None
    p_start, R_start = d.xpos[oid].copy(), d.xmat[oid].reshape(3, 3).copy()
    tcp_sid = m.site("tcp").id
    f6 = np.zeros(6)
    peak_force, track_err = 0.0, 0.0
    while t <= total + 1e-9:
        q = q_at(t)
        d.ctrl[acts] = q
        mujoco.mj_step(m, d)
        jaw_n = 0.0
        for k in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[k].geom1], m.geom_bodyid[d.contact[k].geom2]
            if oid in (b1, b2):
                other = b2 if b1 == oid else b1
                touched.add(m.body(other).name)
                if other in jaw_bodies:
                    mujoco.mj_contactForce(m, d, k, f6)
                    jaw_n += abs(f6[0])                    # normal component
                if first is None and other != 0:           # first arm contact in this segment, in the TCP frame
                    gk = d.contact[k].geom2 if b1 == oid else d.contact[k].geom1
                    loc = (d.contact[k].pos - d.site_xpos[tcp_sid]) @ d.site_xmat[tcp_sid].reshape(3, 3)
                    first = {"t": round(t, 2), "body": m.body(other).name, "geom": m.geom(gk).name or f"#{gk}",
                             "tcp_mm": np.round(loc * 1000, 1).tolist()}
        peak_force = max(peak_force, jaw_n)
        if si < len(segs) and t >= seg_ends[si]:
            seg_log.append({"step": segs[si][3], "obj_mm": (np.round(d.xpos[oid], 4) * 1000).tolist(),
                            "touched": sorted(touched), "first_hit": first})
            if segs[si][3] == "들어 올리기":                 # held? judged at the top of the lift (a place follows)
                at_lift = (float(d.xpos[oid][2]), sum(1 for k in range(d.ncon) if oid in (
                    m.geom_bodyid[d.contact[k].geom1], m.geom_bodyid[d.contact[k].geom2]) and ({
                    m.geom_bodyid[d.contact[k].geom1], m.geom_bodyid[d.contact[k].geom2]} & jaw_bodies)))
            touched, first, si = set(), None, si + 1
        if t >= nxt - 1e-9:
            p, qq = d.xpos[oid], d.xquat[oid]
            qs = d.qpos[jq]
            track_err = max(track_err, float(np.abs(qs[:7] - q[:7]).max()))
            frames.append([round(t, 3), *np.round(p, 4).tolist(), *np.round(qq, 4).tolist(), *np.round(qs, 4).tolist(),
                           round(jaw_n, 1), round(float(d.actuator_force[g_act]), 3)])
            nxt += 1.0 / OUT_HZ
        t += DT
    jaw_contacts = 0
    for k in range(d.ncon):
        b1, b2 = m.geom_bodyid[d.contact[k].geom1], m.geom_bodyid[d.contact[k].geom2]
        if oid in (b1, b2) and ({b1, b2} & jaw_bodies):
            jaw_contacts += 1
    lift = (float(d.xpos[oid][2]) - z0) * 1000
    if at_lift is not None:
        lift, jaw_contacts = (at_lift[0] - z0) * 1000, at_lift[1]
    planned = (plan.get("lift_mm") or 50)
    held = lift > 0.6 * planned and jaw_contacts >= 2
    verdict = "잡혀서 들림" if held else ("미끄러짐/일부만 들림" if lift > 5 else "안 들림")
    place = None
    pl = plan.get("place")
    if pl and "D" in pl:                               # [ours] where the object ended vs the target pose D · start
        D = np.array(pl["D"], float)
        want = D[:3, :3] @ p_start + D[:3, 3]
        Rw = D[:3, :3] @ R_start
        Re = d.xmat[oid].reshape(3, 3)
        err = float(np.linalg.norm((d.xpos[oid] - want)[:2]) * 1000)
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Rw.T @ Re) - 1) / 2, -1, 1))))
        if pl.get("flip_deg"):                          # round object placed with the 180°-turned grasp
            ang = abs(ang - 180.0)
        place = {"err_mm": round(err, 1), "err_deg": round(ang, 1), "dz_mm": round(float(d.xpos[oid][2] - want[2]) * 1000, 1),
                 "ok": bool(held and err < 15 and ang < 15)}
        verdict = verdict if not held else ("옮겨 놓음" if place["ok"] else f"옮겼지만 {err:.0f} mm / {ang:.0f}° 어긋남")
    return {"frames": frames, "lift_mm": round(lift, 1), "planned_lift_mm": planned, "held": bool(held),
            "verdict": verdict, "jaw_contacts": jaw_contacts, "place": place, "duration_s": round(total, 2), "segments": seg_log,
            "peak_jaw_force_n": round(peak_force, 1), "max_track_err_deg": round(float(np.degrees(track_err)), 2),
            "torque_frac": np.round(tf, 3).tolist(),
            "frame_cols": "t, 물체 xyz, 물체 quat(wxyz), 관절 q1..q8 (시뮬 실제값), 집게 접촉 수직력 N, 집게 토크 N·m",
            "hull": {"v": np.round(hull.vertices - c, 4).tolist(), "f": hull.faces.tolist()},
            "assumptions": f"볼록 껍질, 질량 {OBJ_MASS} kg, 마찰 {OBJ_MU}, 집게 마찰 {JAW_MU} (TACO 물리 설정), "
                           f"관절 = 위치 서보 kp 18.22 · 토크 상한 {', '.join(f'{x:.0%}' for x in tf)}"}
