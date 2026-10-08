"""Offline check of the full grasp chain on the virtual scene (no server, no hardware).

virtual camera → reconstructor keyframes → GraspManager._scene_points (raw depth points)
→ segment_object → plan → reports object size vs truth and the plan outcome per object.
Run: uv run --python 3.12 --with numpy --with scipy --with trimesh --with mujoco --with rtree --with open3d \
         --with opencv-python python tests/virtual_grasp.py
"""
import json
import queue
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "camera"), str(HERE / "grasp")]
import recon as R              # noqa: E402
from virtual import VirtualCamera, FLOOR_Y   # noqa: E402
from manager import GraspManager   # noqa: E402

R.REBUILD_EVERY_S = 0.3
cam = VirtualCamera(); cam.start()
rc = R.Reconstructor(); rc.handle({"type": "recon_start"})
CV = np.diag([1.0, -1, -1, 1])
for _ in range(int(25 / 0.25)):            # 25 s of the sweep
    cam._t0 -= 0.25
    f = cam.wait_for_frame(); K = f.intrinsics; s = 256 / K.width
    small = cv2.cvtColor(cv2.resize(f.color, (256, 192), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    rc._last_submit = 0
    rc.submit(np.where(f.depth > .05, f.depth, 0).astype(np.float32), small, K.fx * s, K.fy * s, K.ppx * s, K.ppy * s,
              f.transform.astype(np.float64) @ CV)
time.sleep(1.0)

calib = json.load(open(HERE / "calibrations" / "leader_v1.json"))
arm = types.SimpleNamespace(calib=calib, cmds=queue.Queue(), state={
    "motors": {m: {"pos": (calib["motors"][str(m)]["min"] + calib["motors"][str(m)]["max"]) // 2} for m in range(1, 9)},
    "ram": {m: {"torque": 1} for m in range(1, 9)}})
gm = GraspManager(arm, types.SimpleNamespace(recon=rc, _last_world_pts=None),
                  HERE.parent / "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf")
# browser matrices: auto floor puts the floor at scene y = 0, robot base = viewer's fixed rotation
M_sw = np.eye(4); M_sw[1, 3] = -FLOOR_Y
c, s_ = np.cos(-np.pi / 2), np.sin(-np.pi / 2)
M_sb = np.eye(4); M_sb[:3, :3] = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]) @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
truth = {"빨간 상자 40×50×70": (0.22, 0.07, 0.0), "파란 상자 30×30×40": (0.18, 0.04, -0.11), "초록 원통 Ø50×90": (0.25, 0.09, 0.10)}
for name, (x, h, z) in truth.items():
    gm.handle({"type": "grasp_cancel"})
    gm.handle({"type": "grasp_select", "p_world": [x, FLOOR_Y + h - 0.002, z], "M_scene_world": M_sw.ravel().tolist(),
               "M_scene_base": M_sb.ravel().tolist()})
    o = gm.state["object"]
    t0 = time.time(); gm._plan(); p = gm.state["plan"]
    res = (f"폭 {p['width_mm']} mm 롤 {p['roll_deg']}° IK {p['ik_err_mm']} mm" if "q_grasp" in p
           else f"실패 {p.get('tried')}")
    print(f"{name:<18} 분할 {o['size_mm']} mm ({o['points']}점) → {res}  {time.time() - t0:.1f}s")
