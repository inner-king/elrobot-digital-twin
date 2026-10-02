"""NeRFCapture (iOS, Online mode) DDS probe: measure what actually arrives over WiFi.

Run: CYCLONEDDS_HOME=camera/cyclonedds-0.10.5 uv run --with "cyclonedds==0.10.5" --with numpy \
       python camera/nerfcapture_probe.py [seconds] [iphone_ip]
0.10.x on purpose: the app crashes (SIGSEGV in dq.builtins) on discovery data from cyclonedds 11.x.
Message layout follows NVlabs/instant-ngp scripts/nerfcapture2nerf.py.
"""
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# macOS caps socket buffers at 8 MB (kern.ipc.maxsockbuf), so the 10 MB from instant-ngp fails here
# Peers: unicast discovery for routers that drop multicast (pass the iPhone IP as argv[2])
PEER = sys.argv[2] if len(sys.argv) > 2 else None
os.environ.setdefault("CYCLONEDDS_URI", f"""<CycloneDDS><Domain id="any">
  <Internal><SocketReceiveBufferSize min="8MB"/>
    <DefragUnreliableMaxSamples>64</DefragUnreliableMaxSamples>
    <DefragReliableMaxSamples>64</DefragReliableMaxSamples></Internal>
  <Discovery>{f'<Peers><Peer address="{PEER}"/></Peers>' if PEER else ''}
    <ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>20</MaxAutoParticipantIndex></Discovery>
</Domain></CycloneDDS>""")
from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsParticipant, BuiltinTopicDcpsPublication
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct, annotations as annotate, types
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration


@dataclass
@annotate.final
@annotate.autoid("sequential")
class NeRFCaptureFrame(IdlStruct, typename="NeRFCaptureData.NeRFCaptureFrame"):
    id: types.uint32
    annotate.key("id")
    timestamp: types.float64
    fl_x: types.float32
    fl_y: types.float32
    cx: types.float32
    cy: types.float32
    transform_matrix: types.array[types.float32, 16]
    width: types.uint32
    height: types.uint32
    image: types.sequence[types.uint8]
    has_depth: bool
    depth_width: types.uint32
    depth_height: types.uint32
    depth_scale: types.float32
    depth_image: types.sequence[types.uint8]


SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 15
OUT = Path(__file__).parent / "nerfcapture_frame.npz"
dp = DomainParticipant()
# Reliable (as instant-ngp): a multi-MB frame is thousands of UDP fragments; best-effort drops the whole
# frame on any lost fragment over WiFi. Set NC_BEST_EFFORT=1 to compare.
rel = Policy.Reliability.BestEffort if os.environ.get("NC_BEST_EFFORT") else Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1))
reader = DataReader(dp, Topic(dp, "Frames", NeRFCaptureFrame), qos=Qos(rel))
print(f"listening {SECS:.0f}s for NeRFCapture frames (iPhone: Online mode, same WiFi)...")
t_end, arrivals, frames = time.time() + SECS, [], []
matched, peers = False, set()
parts, pubs = BuiltinDataReader(dp, BuiltinTopicDcpsParticipant), BuiltinDataReader(dp, BuiltinTopicDcpsPublication)
while time.time() < t_end:
    for p in parts.take(N=20):
        if str(p.key) not in peers:
            peers.add(str(p.key)); print(f"participant #{len(peers)} seen at {SECS - (t_end - time.time()):.1f}s")
    for p in pubs.take(N=20):
        if p.topic_name == "Frames" and not matched:
            matched = True; print(f"'Frames' publisher ({p.type_name}) seen at {SECS - (t_end - time.time()):.1f}s")
    for f in reader.take(N=10):
        arrivals.append(time.time())
        frames.append(f)
        if len(frames) == 1:
            print(f"first frame id={f.id} t={f.timestamp:.3f} rgb {f.width}x{f.height} ({len(f.image)} B) "
                  f"depth={f.has_depth} {f.depth_width}x{f.depth_height} scale={f.depth_scale} ({len(f.depth_image)} B)")
    time.sleep(0.005)
if not frames:
    print(f"participants seen {len(peers)} (1 = only this Mac), Frames publisher seen: {matched}")
    sys.exit("프레임 없음: 같은 WiFi인지, 앱이 Online 모드인지, macOS 방화벽이 막는지 확인 필요")
f = frames[-1]
rgb = np.asarray(f.image, np.uint8).reshape(f.height, f.width, 3)
T = np.asarray(f.transform_matrix, np.float32).reshape(4, 4).T  # instant-ngp: column-major -> camera-to-world
msg = [f"frames {len(frames)} in {arrivals[-1] - arrivals[0]:.1f}s"]
if len(arrivals) > 1:
    dt = np.diff(arrivals)
    msg.append(f"fps {1 / dt.mean():.1f}  gap p50 {np.median(dt) * 1e3:.0f} ms max {dt.max() * 1e3:.0f} ms")
    ts = np.array([x.timestamp for x in frames]); msg.append(f"device-timestamp fps {1 / np.diff(ts).mean():.1f}")
print("\n".join(msg))
print(f"intrinsics fx={f.fl_x:.1f} fy={f.fl_y:.1f} cx={f.cx:.1f} cy={f.cy:.1f}")
print("camera-to-world (last frame):\n", T.round(4))
P = np.array([np.asarray(x.transform_matrix, np.float32).reshape(4, 4).T[:3, 3] for x in frames])
print(f"camera travel over capture {np.linalg.norm(P[-1] - P[0]) * 1e3:.1f} mm, path {np.linalg.norm(np.diff(P, axis=0), axis=1).sum() * 1e3:.1f} mm")
save = dict(rgb=rgb, T=T, K=[f.fl_x, f.fl_y, f.cx, f.cy])
if f.has_depth:
    d = np.frombuffer(bytes(f.depth_image), np.float32).reshape(f.depth_height, f.depth_width)
    v = d[np.isfinite(d) & (d > 0)]
    print(f"depth valid {v.size}/{d.size}  min {v.min():.3f}  median {np.median(v):.3f}  max {v.max():.3f}  (raw float32 units)")
    save["depth"] = d
np.savez_compressed(OUT, **save)
print(f"saved {OUT}")
