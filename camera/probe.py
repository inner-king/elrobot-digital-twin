"""Record3D stream probe: measure what the iPhone actually sends before building on it.

Run (iPhone connected by USB, Record3D in USB streaming mode, record button pressed):
  uv run --with record3d --with numpy python camera/probe.py
Prints fps, frame shapes/dtypes, depth stats, intrinsics, pose drift; saves one frame to camera/probe_frame.npz
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np
from record3d import Record3DStream

N_FRAMES = 90
OUT = Path(__file__).parent / "probe_frame.npz"

devs = Record3DStream.get_connected_devices()
print(f"devices: {len(devs)}")
if not devs:
    sys.exit("iPhone이 안 보입니다: USB 연결, Record3D 'USB Streaming' 모드, 녹화 버튼을 확인하세요.")
for d in devs:
    print(f"  udid={d.udid} product_id={d.product_id}")

evt = threading.Event()
s = Record3DStream()
s.on_new_frame = lambda: evt.set()
s.on_stream_stopped = lambda: print("stream stopped")
s.connect(devs[0])

stamps, poses = [], []
for i in range(N_FRAMES):
    if not evt.wait(5.0):
        sys.exit(f"{i}프레임 후 5초간 새 프레임 없음")
    evt.clear()
    stamps.append(time.time())
    depth, rgb, conf = s.get_depth_frame(), s.get_rgb_frame(), s.get_confidence_frame()
    K, p = s.get_intrinsic_mat(), s.get_camera_pose()
    poses.append([p.tx, p.ty, p.tz, p.qx, p.qy, p.qz, p.qw])
    if i == 0:
        print(f"device type: {s.get_device_type()}  (0=TrueDepth, 1=LiDAR)")
        print(f"rgb   {rgb.shape} {rgb.dtype}")
        print(f"depth {depth.shape} {depth.dtype}")
        print(f"conf  {conf.shape} {conf.dtype}  values {np.unique(conf)[:5] if conf.size else 'empty'}")
        print(f"intrinsics fx={K.fx:.1f} fy={K.fy:.1f} cx={K.tx:.1f} cy={K.ty:.1f}")
    if i == N_FRAMES - 1:
        valid = depth[np.isfinite(depth) & (depth > 0)]
        print(f"depth valid {valid.size}/{depth.size}  min {valid.min():.3f} m  median {np.median(valid):.3f} m  max {valid.max():.3f} m")
        np.savez_compressed(OUT, depth=depth, rgb=rgb, conf=conf, K=[K.fx, K.fy, K.tx, K.ty], pose=poses[-1])

dt = np.diff(stamps)
P = np.array(poses)
print(f"fps {1 / dt.mean():.1f}  (frame gap p50 {np.median(dt) * 1e3:.1f} ms, max {dt.max() * 1e3:.1f} ms)")
print(f"pose t first {P[0, :3].round(4)}  last {P[-1, :3].round(4)}  travel {np.linalg.norm(P[-1, :3] - P[0, :3]) * 1e3:.1f} mm")
print(f"pose q first {P[0, 3:].round(4)}  (qx qy qz qw)")
print(f"saved {OUT}")
s.disconnect()
