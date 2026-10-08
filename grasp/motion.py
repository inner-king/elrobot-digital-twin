"""[ours] Motion timeline shared by the executor (manager._run) and the physics prediction (sim.simulate).

Input : q0 (8 joint angles, rad: 7 arm + gripper rev_motor_08), plan from planner.py
Output: segments [(waypoints K×8 rad, duration s, label)], executed in order with PAUSE_S between them.
        pick: 접근 전 위치 → 파지 위치로 하강 → 집게 닫기 → 들어 올리기
        + place (plan["place"]): 옮기기 → 내려놓기 → 집게 열기 → 물러나기   (held in the air: 옮기기 → 목표 자세로 → 잡고 유지)

Within a segment the arm moves through the waypoints with one smoothstep over the whole path (piecewise linear
in joint space between waypoints). The descent and lift waypoints are IK solutions on the straight TCP
line (planner.line_path, 5 mm apart, midpoints ≤ 3 mm off the line), so the real TCP follows the line approach_clear checked; with only
the end points, joint interpolation between two IK solutions swung the jaws 18 mm off the line into the object.
"""
import numpy as np

GRIP_OPEN, GRIP_CLOSE = 0.0, 2.2        # rev_motor_08 [rad]: fully open / fully closed
MAX_JOINT_SPEED = 0.4                   # rad/s, arm joints
GRIP_SPEED = 1.2                        # rad/s, the gripper (its 0 → 2.2 rad at the arm's speed took 5.5 s each way)
MIN_SEGMENT_S = 0.8
PAUSE_S = 0.2
GRIP_CONTACT_EXTRA = 0.26               # plan["gear"] keeps 3 mm per side of clearance: contact is ≈ 0.26 rad further
GRIP_SQUEEZE = 0.35                     # close this far past contact; the servo's torque cap sets the force
GRIP_OPEN_MARGIN = 0.20                 # open only this far beyond the width needed (plus the plan's clearance)


def grip_targets(plan):
    """[ours] (open, close) gripper angles for this grasp: open just enough for the object (the planner checked the
    jaws fully open and closed to contact; any opening in between lies between those), close just past contact."""
    gear = plan.get("gear")
    if gear is None:
        return GRIP_OPEN, GRIP_CLOSE
    return max(GRIP_OPEN, gear - GRIP_OPEN_MARGIN), min(GRIP_CLOSE, gear + GRIP_CONTACT_EXTRA + GRIP_SQUEEZE)


def _duration(W):
    d = np.abs(np.diff(W, axis=0))
    return max(MIN_SEGMENT_S, float(d[:, :7].max(axis=1).sum()) / MAX_JOINT_SPEED, float(d[:, 7].sum()) / GRIP_SPEED)


def segments(q0, plan):
    cur = np.r_[np.asarray(q0[:7], float), q0[7]]
    arm = lambda qs, g: [np.r_[np.asarray(q, float), g] for q in qs]
    desc = plan.get("q_descend") or [plan["q_pre"], plan["q_grasp"]]
    rise = plan.get("q_rise") or [plan["q_grasp"], plan["q_lift"]]
    go, gc = grip_targets(plan)
    out = []
    for wps, label in [([cur, *arm([plan["q_pre"]], go)], "접근 전 위치"),
                       (arm(desc, go), "파지 위치로 하강"),
                       (arm([plan["q_grasp"]], go) + arm([plan["q_grasp"]], gc), "집게 닫기"),
                       (arm(rise, gc), "들어 올리기")] + _place(plan, arm, go, gc):
        W = np.array(wps)
        W[0] = cur                                       # start exactly where the previous segment ended
        dur = _duration(W)
        out.append((W, dur, label))
        cur = W[-1]
    return out


def _place(plan, arm, go=GRIP_OPEN, gc=GRIP_CLOSE):
    """[ours] pick-and-place tail (planner.plan_place): carry, straight descent, open, back up the same line."""
    pl = plan.get("place")
    if not pl or "q_release" not in pl:
        return []
    if pl.get("hold"):                     # target in the air: brought there and held (no release)
        return [(arm(pl["q_carry"] + pl["q_place_descend"][:1], gc), "옮기기"),
                (arm(pl["q_place_descend"], gc), "목표 자세로"),
                (arm([pl["q_release"]] * 2, gc), "잡고 유지")]
    return [(arm(pl["q_carry"] + pl["q_place_descend"][:1], gc), "옮기기"),
            (arm(pl["q_place_descend"], gc), "내려놓기"),
            (arm([pl["q_release"]], gc) + arm([pl["q_release"]], go), "집게 열기"),
            (arm(pl["q_place_descend"][::-1], go), "물러나기")]


def at(W, s):
    """Point at path fraction s ∈ [0, 1] after smoothstep, by joint-space arc length between waypoints."""
    s = s * s * (3 - 2 * s)
    if len(W) == 1:
        return W[0]
    seg = np.abs(np.diff(W, axis=0)).max(axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    if cum[-1] <= 0:
        return W[-1]
    x = s * cum[-1]
    i = min(int(np.searchsorted(cum, x, side="right")) - 1, len(W) - 2)
    f = (x - cum[i]) / seg[i] if seg[i] > 0 else 1.0
    return W[i] + (W[i + 1] - W[i]) * f
