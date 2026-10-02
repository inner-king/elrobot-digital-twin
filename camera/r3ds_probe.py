"""Record3DStream (PathOn) probe over USB: measure fps, resolutions, depth range, pose.

Run: uv run --with pymobiledevice3 --with numpy --with opencv-python python camera/r3ds_probe.py [seconds]
USB tunnel localhost:8888 -> iPhone:8888 via pymobiledevice3 (no Homebrew iproxy needed).
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "r3ds_sdk"))
from sdk import IPhoneSensorClient  # noqa: E402

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 10
fwd = subprocess.Popen([sys.executable, "-m", "pymobiledevice3", "usbmux", "forward", "8888", "8888"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
try:
    client = IPhoneSensorClient("localhost", port=8888)
    for _ in range(20):  # the forwarder takes a few seconds to start listening
        time.sleep(0.5)
        if client.start():
            break
    else:
        sys.exit("연결 실패: 앱에서 스트리밍이 켜져 있는지, 포트가 8888인지 확인")
    frames, arrivals, t_end = [], [], time.time() + SECS
    while time.time() < t_end:
        f = client.wait_for_frame()
        if f is None:
            continue
        arrivals.append(time.time()); frames.append(f)
        if len(frames) == 1:
            print(f"first frame id={f.frame_id} t={f.timestamp:.3f}")
            print(f"  color {f.color.shape} {f.color.dtype}   depth {f.depth.shape} {f.depth.dtype}")
            conf = getattr(f, "confidence", None)
            print(f"  confidence {None if conf is None else (conf.shape, np.unique(conf)[:4])}   imu {getattr(f, 'imu', None)}")
            print(f"  intrinsics {f.intrinsics}")
    client.stop()
finally:
    fwd.terminate()
if len(frames) < 2:
    sys.exit(f"프레임 {len(frames)}개만 수신")
dt = np.diff(arrivals)
ids = [f.frame_id for f in frames]
print(f"frames {len(frames)} in {arrivals[-1] - arrivals[0]:.1f}s -> fps {1 / dt.mean():.1f} "
      f"(gap p50 {np.median(dt) * 1e3:.0f} ms, max {dt.max() * 1e3:.0f} ms), id gaps (dropped) {sum(np.diff(ids) - 1)}")
f = frames[-1]
d = f.depth; v = d[np.isfinite(d) & (d > 0)]
print(f"depth valid {v.size}/{d.size}  min {v.min():.3f}  p50 {np.median(v):.3f}  max {v.max():.3f} m")
P = np.array([x.transform[:3, 3] for x in frames])
print("pose (last):\n", np.round(f.transform, 4))
print(f"camera travel {np.linalg.norm(P[-1] - P[0]) * 1e3:.1f} mm, path {np.linalg.norm(np.diff(P, axis=0), axis=1).sum() * 1e3:.1f} mm")
np.savez_compressed(HERE / "r3ds_frame.npz", color=f.color, depth=d, T=f.transform,
                    conf=getattr(f, "confidence", None) if getattr(f, "confidence", None) is not None else np.zeros(0),
                    K=np.array([f.intrinsics.fx, f.intrinsics.fy, f.intrinsics.ppx, f.intrinsics.ppy, f.intrinsics.width, f.intrinsics.height]))
print("saved camera/r3ds_frame.npz")
