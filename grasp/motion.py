"""[ours] Motion timeline shared by the executor (manager._run) and the physics prediction (sim.simulate).

Input : q0 (8 joint angles, rad: 7 arm + gripper rev_motor_08), plan from planner.py
Output: segments [(waypoints K×8 rad, duration s, label)], executed in order with PAUSE_S between them.
        pick: 접근 전 위치 → 파지 위치로 하강 → 집게 닫기 → 들어 올리기
        + place (plan["place"]): 옮기기 → 내려놓기 → 집게 열기 → 물러나기

Within a segment the arm moves through the waypoints with one smoothstep over the whole path (piecewise linear
in joint space between waypoints). The descent and lift waypoints are IK solutions on the straight TCP
line (planner.line_path, 5 mm apart, midpoints ≤ 3 mm off the line), so the real TCP follows the line approach_clear checked; with only
the end points, joint interpolation between two IK solutions swung the jaws 18 mm off the line into the object.
"""
import numpy as np

GRIP_OPEN, GRIP_CLOSE = 0.0, 2.2        # rev_motor_08 [rad]; close past contact, the servo torque cap limits force
MAX_JOINT_SPEED = 0.4                   # rad/s
MIN_SEGMENT_S = 1.2
PAUSE_S = 0.4


def segments(q0, plan):
    cur = np.r_[np.asarray(q0[:7], float), q0[7]]
    arm = lambda qs, g: [np.r_[np.asarray(q, float), g] for q in qs]
    desc = plan.get("q_descend") or [plan["q_pre"], plan["q_grasp"]]
    rise = plan.get("q_rise") or [plan["q_grasp"], plan["q_lift"]]
    out = []
    for wps, label in [([cur, *arm([plan["q_pre"]], GRIP_OPEN)], "접근 전 위치"),
                       (arm(desc, GRIP_OPEN), "파지 위치로 하강"),
                       (arm([plan["q_grasp"]], GRIP_OPEN) + arm([plan["q_grasp"]], GRIP_CLOSE), "집게 닫기"),
                       (arm(rise, GRIP_CLOSE), "들어 올리기")] + _place(plan, arm):
        W = np.array(wps)
        W[0] = cur                                       # start exactly where the previous segment ended
        dur = max(MIN_SEGMENT_S, float(np.abs(np.diff(W, axis=0)).max(axis=1).sum()) / MAX_JOINT_SPEED)
        out.append((W, dur, label))
        cur = W[-1]
    return out


def _place(plan, arm):
    """[ours] pick-and-place tail (planner.plan_place): carry, straight descent, open, back up the same line."""
    pl = plan.get("place")
    if not pl or "q_release" not in pl:
        return []
    return [(arm(pl["q_carry"] + pl["q_place_descend"][:1], GRIP_CLOSE), "옮기기"),
            (arm(pl["q_place_descend"], GRIP_CLOSE), "내려놓기"),
            (arm([pl["q_release"]], GRIP_CLOSE) + arm([pl["q_release"]], GRIP_OPEN), "집게 열기"),
            (arm(pl["q_place_descend"][::-1], GRIP_OPEN), "물러나기")]


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
