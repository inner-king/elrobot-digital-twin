"""Pick an object from the live reconstruction and plan an ElRobot grasp.

Two layers, kept apart on purpose:

  [TACO]  taco_tools/grasp_transfer.py, called unchanged:
          gripper_hulls()                         gripper convex parts in the tcp frame
          sample_antipodal(mesh, roi, normals)    ray-cast antipodal pairs (2 mm ≤ width ≤ 51 − 6 mm, normals ≤ 35°)
          place_gripper(candidate, roll, approach) tcp frame: x closes, y approaches, z = x × y
          hull_hits(points, hulls, gear)          object points inside the gripper → reject
          CANDIDATES, ROLLS, GRIPPER_SPAN, CLEARANCE_MM, GEAR_PER_METRE
  [ours]  everything a TACO demo provided and a live scene does not:
          - object = region grown from the clicked point above the local support surface
            (replaces the human-contact region; the whole object surface is the ROI)
          - closed surface for ray casting = the observed mesh if it yields candidates, else its convex hull
            (a one-sided TSDF scan has no exit surface for the rays)
          - approach = straight down along gravity; ranking centre = the closed hull's volume centroid (replaces the
            human's; the mean of LiDAR points sits on the top face and pulled grasps onto the top edge, where the
            simulated squeeze ejected the box)
          - collision points = object + surroundings (support surface) within 15 cm, not the object alone
          - candidate order as TACO's best_grasp, but a candidate also has to pass IK for the grasp and
            the 10 cm pre-grasp (elrobot_mujoco/pick_demo.py solve_ik, unchanged)
          - approach sweep: TACO's approach_clear rule (gripper swept 10 cm back along the approach must not hit
            anything) re-implemented with point-in-hull tests on the scene points instead of CoACD parts

Frames: base = ElRobot URDF base_link (z up), world = ARKit world (y up), all metres.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]          # 8_DoF_arm/
sys.path.insert(0, str(ROOT / "taco_tools"))
sys.path.insert(0, str(ROOT / "elrobot_mujoco"))

SEG_RADIUS_M = 0.25        # object search radius around the click
SEG_STEP_M = 0.012         # region-growing neighbour distance
# object points must sit this far above the support surface. The 1 cm TSDF mesh skirts every object where it meets
# the floor; at 8 mm those skirts bridged neighbouring objects 10 cm apart into one 203×125 mm "object".
SEG_ABOVE_M = 0.018         # for the TSDF mesh fallback
SEG_ABOVE_RAW_M = 0.008     # raw depth points have no skirts
COLLISION_RADIUS_M = 0.15
# back-off along −approach before descending: TACO's approach_clear uses 10 cm, but ElRobot's top-down workspace
# shrinks with height (pick_demo.py: "top-down grasps … reach only ~0.22 m in front of the base at z = 0.09 m"),
# so the longest reachable back-off of these is used, and the approach sweep covers exactly that distance
PREGRASP_OPTIONS_M = (0.10, 0.07, 0.05, 0.03)
PREGRASP_M = PREGRASP_OPTIONS_M[0]
LIFT_OPTIONS_M = (0.05, 0.03)
# [ours] extra straight rise above the pre-grasp: the free joint-space move from the current pose then ends high
# above the object (a 3 cm pre-grasp let that move sweep the jaws through a 40 mm box in the physics check)
RISE_OPTIONS_M = (0.08, 0.05, 0.03)
IK_POS_TOL, IK_ROT_TOL = 2e-3, 0.03
# [ours] where along the pads the object sits: TCP moved back along the approach by this much (+ = contact nearer the
# fingertips, which end 13 mm past the TCP — a 17–24 mm-high knife on a board had every candidate's tips in the board;
# − = deeper in the jaws, more pad on the object)
GRASP_DEPTH_M = (0.0, -0.010, 0.006, 0.010)


def _gt():
    import grasp_transfer as gt   # noqa: imported lazily (trimesh, taco_replay)
    return gt


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(ROOT / "elrobot_mujoco" / "elrobot.xml"))
        _MODEL = (m, mujoco.MjData(m))
    return _MODEL


# ---------------------------------------------------------------- [ours] object from a click
def segment_object(points_base, click_base, up, above=SEG_ABOVE_M):
    """Points (N×3, base frame) → (object points, support height along `up`, surroundings)."""
    up = up / np.linalg.norm(up)
    rel = points_base - click_base
    h = rel @ up
    horiz = np.linalg.norm(rel - np.outer(h, up), axis=1)
    near = horiz < SEG_RADIUS_M
    if near.sum() < 50:
        raise RuntimeError("클릭 주변에 점이 너무 적습니다")
    # support = the strongest horizontal layer below the click inside the search radius
    hb = h[near & (h < 0.005)]
    if len(hb) < 30:
        raise RuntimeError("물체 아래 받침면(바닥/책상)을 찾지 못했습니다")
    hist, edges = np.histogram(hb, bins=np.arange(hb.min() - 0.01, 0.015, 0.005))
    k = int(np.argmax(hist))
    sel = (hb >= edges[k] - 0.005) & (hb < edges[k + 1] + 0.005)
    support = float(np.median(hb[sel]))
    cand = np.nonzero(near & (h > support + above))[0]
    if not len(cand):
        raise RuntimeError("받침면 위에 물체 점이 없습니다")
    tree = cKDTree(points_base[cand])
    seed = int(np.argmin(np.linalg.norm(points_base[cand] - click_base, axis=1)))
    if np.linalg.norm(points_base[cand[seed]] - click_base) > 0.03:
        raise RuntimeError("클릭한 곳이 받침면 위 물체가 아닙니다")
    seen = np.zeros(len(cand), bool)
    seen[seed] = True
    frontier = [seed]
    while frontier:                                     # region growing over the points above the support
        nb = tree.query_ball_point(points_base[cand[frontier]], SEG_STEP_M)
        nxt = {j for lst in nb for j in lst if not seen[j]}
        for j in nxt:
            seen[j] = True
        frontier = list(nxt)
    obj_idx = cand[seen]
    dist = np.linalg.norm(points_base - points_base[obj_idx].mean(axis=0), axis=1)
    env_idx = np.nonzero((dist < COLLISION_RADIUS_M + 0.1) & ~np.isin(np.arange(len(points_base)), obj_idx))[0]
    return obj_idx, support, env_idx


# ---------------------------------------------------------------- [ours] closed surface for TACO's ray casting
def _drop_outliers(P, k=8, n_sigma=2.0):
    """Statistical outlier removal: a convex hull takes the extreme points, so stray noise widens the object."""
    if len(P) <= k + 1:
        return P
    d, _ = cKDTree(P).query(P, k + 1)
    m = d[:, 1:].mean(axis=1)
    return P[m < m.mean() + n_sigma * m.std()]


def object_meshes(v_base, tris, obj_idx, support_z=None):
    import trimesh
    keep = np.zeros(len(v_base), bool)
    keep[obj_idx] = True
    faces = tris[keep[tris].all(axis=1)] if tris is not None and len(tris) else np.zeros((0, 3), int)
    meshes = []
    if len(faces) >= 20:
        m = trimesh.Trimesh(v_base, faces, process=True)
        m.remove_unreferenced_vertices()
        meshes.append(("관측 메시", m))
    pts = _drop_outliers(v_base[obj_idx])
    if support_z is not None:                          # close the solid down to the support it rests on
        foot = pts.copy()
        foot[:, 2] = support_z
        pts = np.vstack([pts, foot])
    meshes.append(("볼록 껍질", trimesh.convex.convex_hull(pts)))
    return meshes


# ---------------------------------------------------------------- [ours] IK wrapper around pick_demo.solve_ik
def ik(target_pos, target_rot, q_seed, restarts=6):
    from pick_demo import solve_ik   # unchanged implementation (damped least squares on the tcp site)
    m, d = _model()
    d.qpos[:] = 0
    d.qpos[:7] = q_seed
    q, pe, re = solve_ik(m, d, np.asarray(target_pos, float), np.asarray(target_rot, float), np.asarray(q_seed, float),
                         iters=200, restarts=restarts)
    return q, pe, re


LINE_STEP_M, LINE_DEV_TOL_M, LINE_MAX_DQ = 0.005, 0.003, 0.8


def _tcp_at(q):
    import mujoco
    m, d = _model()
    d.qpos[:] = 0
    d.qpos[:7] = q
    mujoco.mj_kinematics(m, d)
    return d.site("tcp").xpos.copy()


def line_path(q_from, p_from, R, direction, dist):
    """[ours] TCP straight line p_from → p_from + dist·direction (orientation R kept): one IK every LINE_STEP_M,
    each seeded by the previous solution with no random restarts, so the arm stays on one branch. The executor
    interpolates joints between waypoints, so the TCP at each midpoint must also stay within LINE_DEV_TOL_M of the line.
    Returns ([q_from, …, q_end], worst (pos err m, rot err rad)) or None."""
    n = max(1, int(np.ceil(dist / LINE_STEP_M)))
    qs, worst = [np.asarray(q_from, float)], (0.0, 0.0)
    for k in range(1, n + 1):
        q, pe, re = ik(p_from + direction * dist * k / n, R, qs[-1], restarts=0)
        if pe > IK_POS_TOL or re > IK_ROT_TOL or np.abs(q - qs[-1]).max() > LINE_MAX_DQ:
            return None
        v = _tcp_at((q + qs[-1]) / 2) - p_from
        if np.linalg.norm(v - (v @ direction) * direction) > LINE_DEV_TOL_M:
            return None
        qs.append(q)
        worst = (max(worst[0], pe), max(worst[1], re))
    return qs, worst


def approach_clear(points, frame, hulls, gt, steps=10, backoff=PREGRASP_M):
    """[ours, TACO rule] open gripper swept back `backoff` along −approach hits no scene point."""
    return sweep_clear(points, frame, -frame[:3, 1], backoff, hulls, gt, steps)


def sweep_clear(points, frame, direction, dist, hulls, gt, steps=10):
    """[ours] open gripper (orientation kept) translated up to `dist` along `direction` hits no scene point."""
    R = frame[:3, :3]
    near = points[np.linalg.norm(points - frame[:3, 3], axis=1) < COLLISION_RADIUS_M + dist]
    for s in np.linspace(0, dist, steps + 1)[1:]:
        o = frame[:3, 3] + s * direction
        if gt.hull_hits((near - o) @ R, hulls, 0.0):
            return False
    return True


def plan(points_base, tris, click_base, up, q_now, rng_seed=0):
    """Returns a dict with the chosen grasp, joint waypoints and every number needed to judge it."""
    return next(plans(points_base, tris, click_base, up, q_now, rng_seed))


def segment_given(points_base, obj_base, support_base, up, gap=0.006):
    """[ours] The object is already known (the reconstruction's split object, completed): scene points within `gap` of
    it are its own and leave the collision set; its points are appended. → (points, obj_idx, support along up from its
    centre, env_idx) — the same contract as segment_object."""
    up = up / np.linalg.norm(up)
    d, _ = cKDTree(obj_base).query(points_base, distance_upper_bound=gap)
    # and whatever stands inside its footprint (+1 cm) above the support: its own residue the completion missed
    # (left behind as an obstacle at the old place, it blocked moving the object a few cm)
    e1 = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    H = np.c_[points_base @ e1, points_base @ e2]
    Ho = np.c_[obj_base @ e1, obj_base @ e2]
    inside = ((H >= Ho.min(0) - 0.01) & (H <= Ho.max(0) + 0.01)).all(axis=1) & (points_base @ up > support_base @ up + 0.004)
    rest = points_base[~np.isfinite(d) & ~inside]
    P = np.vstack([rest, obj_base])
    obj_idx = np.arange(len(rest), len(P))
    c = obj_base.mean(axis=0)
    support = float((support_base - c) @ up)
    dist = np.linalg.norm(rest - c, axis=1)
    env_idx = np.nonzero(dist < COLLISION_RADIUS_M + 0.1)[0]
    return P, obj_idx, support, env_idx


def plans(points_base, tris, click_base, up, q_now, rng_seed=0, seg=None):
    """Feasible grasps in ranking order (a generator); the last item is an error dict when nothing (more) fits.
    The caller can test each one (physics prediction) and keep the first that works."""
    gt = _gt()
    t0 = time.time()
    up = up / np.linalg.norm(up)
    given_mesh = None
    if seg is not None:                                # the object was given (segment_given)
        obj_idx, support, env_idx = seg[:3]
        given_mesh = seg[3] if len(seg) > 3 else None  # its completed mesh (V, T) in the base frame
    else:
        obj_idx, support, env_idx = segment_object(points_base, click_base, up,
                                                   SEG_ABOVE_M if tris is not None else SEG_ABOVE_RAW_M)
    obj = points_base[obj_idx]
    centre = obj.mean(axis=0)
    approach = -up                                     # [ours] top-down, the TACO slot for the human approach
    hulls, _ = gt.gripper_hulls()
    rng = np.random.default_rng(rng_seed)
    out = {"object_points": int(len(obj)), "support_height": round(support, 4),
           "object_size_m": np.round(np.ptp(obj, axis=0), 3).tolist(), "tried": []}
    collide = np.vstack([obj, points_base[env_idx]]) if len(env_idx) else obj
    support_z = float(click_base @ up + support) if abs(up[2]) > 0.95 else None   # base z up: absolute support height
    out["support_z"] = support_z
    surfaces = object_meshes(points_base, tris, obj_idx, support_z)
    if given_mesh is not None:
        # [ours] the completed (non-convex) surface first: a mug's or pot's rim, a banana's inner curve fit the 51 mm
        # jaws; their convex hull (94 / 266 / 75 mm across) gave no candidate at all
        import trimesh
        cm = trimesh.Trimesh(given_mesh[0], given_mesh[1], process=True)
        surfaces = [("완성 메시", cm)] + [x for x in surfaces if x[0] != "관측 메시"]
    for label, mesh in surfaces:
        # [TACO] sample on this surface; normals = mesh vertex normals (outward)
        roi = np.asarray(mesh.vertices)
        nrm = np.asarray(mesh.vertex_normals)
        if len(roi) > 3000:
            pick = rng.choice(len(roi), 3000, replace=False)
            roi, nrm = roi[pick], nrm[pick]
        mesh_cm = mesh.copy()
        mesh_cm.apply_scale(100.0)                     # TACO convention: mesh in cm, ROI points in m
        cands = gt.sample_antipodal(mesh_cm, roi, nrm, rng)
        ref = mesh.center_mass if mesh.is_volume else centre
        out["tried"].append({"surface": label, "candidates": len(cands)})
        if not cands:
            continue
        # [TACO] best_grasp ranking (distance to the reference centre − 0.02·quality), top 60, ROLLS each
        ranked = sorted(cands, key=lambda c: np.linalg.norm(c["centre"] - ref) - 0.02 * c["quality"])[:60]
        n_free = n_ik = 0
        for c in ranked:
            gear = (gt.GRIPPER_SPAN - c["width"] - 2 * gt.CLEARANCE_MM / 1000) * gt.GEAR_PER_METRE
            near = collide[np.linalg.norm(collide - c["centre"], axis=1) < COLLISION_RADIUS_M]
            for roll, depth in ((r_, d_) for r_ in gt.ROLLS for d_ in GRASP_DEPTH_M):
                frame = gt.place_gripper(c, roll, approach)
                frame[:3, 3] -= depth * frame[:3, 1]           # [ours] pad position on the object (GRASP_DEPTH_M)
                local = (near - frame[:3, 3]) @ frame[:3, :3]
                if gt.hull_hits(local, hulls, 0.0) or gt.hull_hits(local, hulls, gear):
                    continue                           # open jaws and closed-to-contact jaws must both be free
                R, p = frame[:3, :3], frame[:3, 3]
                if not approach_clear(collide, frame, hulls, gt, backoff=PREGRASP_OPTIONS_M[-1]):
                    continue
                n_free += 1
                ok = lambda e: e[0] <= IK_POS_TOL and e[1] <= IK_ROT_TOL
                q_g, pe, re = ik(p, R, np.asarray(q_now[:7], float))      # the grasp itself first
                if not ok((pe, re)):
                    continue
                pre_sol = None
                for b in PREGRASP_OPTIONS_M:                                # longest reachable, clear back-off
                    if not approach_clear(collide, frame, hulls, gt, backoff=b):
                        continue
                    path = line_path(q_g, p, R, -R[:, 1], b)                # [ours] straight descent, one branch
                    if not path:
                        continue
                    desc, worst, rise = path[0][::-1], path[1], 0.0
                    pre_frame = frame.copy(); pre_frame[:3, 3] = p - b * R[:, 1]
                    for h in RISE_OPTIONS_M:                                 # then straight up from there
                        if not sweep_clear(collide, pre_frame, up, h, hulls, gt):
                            continue
                        r_path = line_path(desc[0], pre_frame[:3, 3], R, up, h)
                        if r_path:
                            desc = r_path[0][::-1] + desc[1:]
                            worst = tuple(max(a, c) for a, c in zip(worst, r_path[1]))
                            rise = h
                            break
                    pre_sol = (b, desc, worst, rise)
                    break
                lift_sol = None
                for lh in LIFT_OPTIONS_M:
                    path = line_path(q_g, p, R, up, lh)
                    if path:
                        lift_sol = (lh, path[0], path[1])
                        break
                if pre_sol is None or lift_sol is None:
                    continue
                qs = [pre_sol[1][0], q_g, lift_sol[1][-1]]
                errs = [pre_sol[2], (pe, re), lift_sol[2]]
                n_ik += 1
                out.update(surface=label, candidates=len(cands), collision_free_tried=n_free, ik_ok=n_ik,
                           width_mm=round(c["width"] * 1000, 1), quality=round(c["quality"], 3),
                           roll_deg=round(float(np.degrees(roll)), 0), gear=round(float(gear), 3), depth_mm=round(depth * 1000),
                           frame=np.round(frame, 5).tolist(),
                           ik_err_mm=[round(float(e[0]) * 1000, 2) for e in errs], ik_err_deg=[round(float(np.degrees(e[1])), 2) for e in errs],
                           q_pre=qs[0].round(4).tolist(), q_grasp=qs[1].round(4).tolist(), q_lift=qs[2].round(4).tolist(),
                           approach_mm=round(pre_sol[0] * 1000), rise_mm=round(pre_sol[3] * 1000), lift_mm=round(lift_sol[0] * 1000),
                           q_descend=[q.round(4).tolist() for q in pre_sol[1]], q_rise=[q.round(4).tolist() for q in lift_sol[1]],
                           obj_idx=obj_idx, ms=round((time.time() - t0) * 1000))
                yield dict(out)
        out["tried"][-1].update(collision_free=n_free, ik_ok=n_ik)
    out = {k: out[k] for k in ("object_points", "support_height", "object_size_m", "tried", "support_z")}
    out.update(obj_idx=obj_idx, ms=round((time.time() - t0) * 1000), error="충돌 없고 IK가 닿는 파지를 찾지 못했습니다")
    yield out


# ---------------------------------------------------------------- [ours] place: carry the grasped object to a target
PLACE_GAP_M = 0.006            # release this far above the target (the object drops the rest)
CARRY_EXTRA_M = (0.0, 0.04, 0.08)   # extra straight rise after the lift when the carry would hit something
CARRY_CLEAR_M = 0.006          # carried object ↔ scene points
HOLD_ABOVE_M = 0.015          # target this far above the support: nothing to set it on — hold it there instead
PLACE_BELOW_M = 0.010          # at the target, scene points this low above the support are the support itself


def _tcp_frame(q):
    import mujoco
    m, d = _model()
    d.qpos[:] = 0
    d.qpos[:7] = q
    mujoco.mj_kinematics(m, d)
    F = np.eye(4)
    F[:3, :3] = d.site("tcp").xmat.reshape(3, 3)
    F[:3, 3] = d.site("tcp").xpos
    return F


def _path_clear(qs, env, obj_local, hulls, gear, gt, n=8, obj_env=None):
    """Gripper (closed on the object, `gear`) and the carried object (its points in the tcp frame) stay clear of
    the scene points `env` at n samples between every pair of waypoints (the executor interpolates joints)."""
    obj_env = env if obj_env is None else obj_env
    tree = cKDTree(obj_env) if len(obj_env) else None
    for a, b in zip(qs[:-1], qs[1:]):
        for s in np.linspace(0, 1, n + 1)[1:]:
            F = _tcp_frame(a + (b - a) * s)
            R, o = F[:3, :3], F[:3, 3]
            near = env[np.linalg.norm(env - o, axis=1) < COLLISION_RADIUS_M] if len(env) else env
            if len(near) and gt.hull_hits((near - o) @ R, hulls, gear):
                return "집게"
            if tree is not None:
                dd, ii = tree.query(obj_local @ R.T + o, distance_upper_bound=CARRY_CLEAR_M)
                if dd.min() < CARRY_CLEAR_M:
                    hit = obj_env[ii[np.argmin(dd)]]
                    return f"물체 (장면 점 {np.round(hit * 1000).astype(int).tolist()} mm)"
    return None


def plan_place(gplan, D, points_base, obj_idx, up):
    """[ours] Pick-and-place from a grasp plan. The object's target pose = D · current pose (D: 4×4 in the base frame,
    any rigid move from the console's gizmo — 6 DoF: the object is held rigidly, so the grasp frame simply moves with
    it, G' = D·G, tilted if the object is). Released PLACE_GAP_M above the target, then straight back up; a target
    more than HOLD_ABOVE_M above the support has nothing under it and is held there instead (no release).
      carry   : lift end → (extra straight rise if needed) → above the target, joint-space, checked against the scene
      descend : straight line down to the release pose (planner.line_path)
      retreat : jaws open, same line back up
    G' turned 180° about the approach axis is the same grasp for the gripper but turns the object 180° too: tried
    second, only for a round footprint (principal extents within 15 %), where that turn does not show.
    Returns the place dict (joint waypoints, errors) or {"error": …}."""
    gt = _gt()
    hulls, _ = gt.gripper_hulls()
    up = up / np.linalg.norm(up)
    G = np.array(gplan["frame"], float)
    gear = float(gplan["gear"])
    lh = gplan["lift_mm"] / 1000
    obj = points_base[obj_idx]
    obj_local = (obj - G[:3, 3]) @ G[:3, :3]               # object in the tcp frame: carried rigidly
    tgt = obj @ D[:3, :3].T + D[:3, 3]
    keep = np.ones(len(points_base), bool)
    keep[obj_idx] = False
    env = points_base[keep]
    # the support right under the object at its start is not an obstacle while it is lifted out (same rule as the
    # pick: the object sits SEG_ABOVE on it); everything else is
    q_lift = np.array(gplan["q_rise"][-1] if gplan.get("q_rise") else gplan["q_lift"], float)
    p_lift = G[:3, 3] + up * lh
    tried = []
    Hc = obj - obj.mean(axis=0)
    Hc = Hc - np.outer(Hc @ up, up)
    ev = np.sort(np.linalg.eigvalsh(Hc.T @ Hc / len(Hc)))[-2:]
    round_ = bool(np.sqrt(ev[0] / max(ev[1], 1e-12)) > 0.85)
    sz = gplan.get("support_z")
    hold = bool(sz is not None and float((tgt @ up).min()) > sz + SEG_ABOVE_RAW_M + HOLD_ABOVE_M)
    vertical = abs((D @ G)[:3, 1] @ up) > 0.95             # the 180° twin is only the same for a top-down grasp
    for flip in ((0.0, np.pi) if round_ and vertical else (0.0,)):
        Gp = D @ G
        if flip:
            Rf = np.eye(4)
            a = Gp[:3, 1]                                   # approach axis (tcp y)
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            Rf[:3, :3] = np.eye(3) + 2 * K @ K              # Rodrigues, θ = π
            Gp[:3, :3] = Rf[:3, :3] @ Gp[:3, :3]
        R, p_rel = Gp[:3, :3], Gp[:3, 3] + (0.0 if hold else PLACE_GAP_M) * up
        for extra in CARRY_EXTRA_M:
            h = lh + extra - PLACE_GAP_M                    # carry height above the release pose
            carry = [q_lift]
            if extra:
                path = line_path(q_lift, p_lift, G[:3, :3], up, extra)
                if not path:
                    tried.append({"flip": bool(flip), "extra_mm": extra * 1000, "fail": "추가 상승 IK"})
                    continue
                carry = path[0]
            q_rel, pe, re = ik(p_rel, R, carry[-1])
            if pe > IK_POS_TOL or re > IK_ROT_TOL:
                tried.append({"flip": bool(flip), "extra_mm": extra * 1000, "fail": f"놓기 IK {pe * 1000:.1f} mm"})
                break                                       # higher carry does not change the release pose
            down = line_path(q_rel, p_rel, R, up, h)
            if not down:
                tried.append({"flip": bool(flip), "extra_mm": extra * 1000, "fail": "수직 하강 경로"})
                continue
            descend = down[0][::-1]                         # above the target → release
            transit = carry + [descend[0]]
            why = _path_clear(transit, env, obj_local, hulls, gear, gt)
            if why:
                tried.append({"flip": bool(flip), "extra_mm": extra * 1000, "fail": f"옮기는 길 충돌: {why}"})
                continue
            # descent: the carried object must not hit what is at the target (the support stays CARRY_CLEAR below)
            # (the support it is set down on is not an obstacle for the object: it is released PLACE_GAP above it)
            # (support-surface noise and edges reach a few mm up: the lowest PLACE_BELOW_M above it are not obstacles)
            above = env[env @ up > sz + PLACE_BELOW_M] if sz is not None else env
            why = _path_clear(descend, env, obj_local, hulls, gear, gt, n=2, obj_env=above)
            if why:
                tried.append({"flip": bool(flip), "extra_mm": extra * 1000, "fail": f"목표 자리 충돌: {why}"})
                break
            return {"q_carry": [q.round(4).tolist() for q in carry], "q_place_descend": [q.round(4).tolist() for q in descend],
                    "q_release": q_rel.round(4).tolist(), "carry_extra_mm": round(extra * 1000), "flip_deg": round(np.degrees(flip)),
                    "release_gap_mm": 0 if hold else PLACE_GAP_M * 1000, "round": round_, "hold": hold, "ik_err_mm": round(pe * 1000, 2), "ik_err_deg": round(float(np.degrees(re)), 2),
                    "target_frame": np.round(Gp, 5).tolist(), "D": np.round(D, 5).tolist(),
                    "target_centre": tgt.mean(axis=0).round(4).tolist(), "tried": tried}
    return {"error": "목표 위치에 놓는 경로를 찾지 못했습니다", "tried": tried}
