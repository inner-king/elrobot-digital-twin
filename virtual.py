"""Virtual devices so the whole console runs without hardware.

VirtualBus     drop-in for server.Bus: 8 ST3215 servos as register files. With torque on, the present position moves
               toward the goal at the goal speed (steps/s, acceleration ignored); load follows the remaining error.
VirtualCamera  drop-in for the Record3DStream client: a floor with three objects in front of the robot, seen by a
               camera that sweeps slowly around them. 30 fps frames with the same fields as the SDK frame
               (depth 256×192 float32 m, BGR colour 640×480, intrinsics at colour resolution, ARKit camera-to-world).
"""
import threading
import time
import types

import numpy as np


# ---------------------------------------------------------------- robot
RELEASE_STEPS = 40            # opening this far past the held object releases it
class VirtualBus:
    def __init__(self, calib):
        self.stall = {}               # mid → (raw the servo cannot pass, closing sign): an object between the jaws
        self.on_release = None        # called when the jaws are opened past the object (it is let go)
        self.reg = {}
        self._t = {}
        for mid in range(1, 9):
            r = bytearray(80)
            r[3:5] = (777).to_bytes(2, "little")          # model
            r[5] = mid
            r[9:11] = (0).to_bytes(2, "little"); r[11:13] = (4095).to_bytes(2, "little")
            r[13], r[14], r[15] = 70, 140, 40
            r[16:18] = (1000).to_bytes(2, "little")
            r[18], r[19], r[21], r[22] = 12, 44, 32, 32
            a = calib.get("motors", {}).get(str(mid))
            pos = (a["min"] + a["max"]) // 2 if a else 2048
            if a:
                off = int(a["offset"])
                r[31:33] = ((abs(off) & 0x7FF) | (0x800 if off < 0 else 0)).to_bytes(2, "little")
            r[42:44] = pos.to_bytes(2, "little")
            r[46:48] = (400).to_bytes(2, "little")
            r[48:50] = (1000).to_bytes(2, "little")
            r[55] = 1
            r[56:58] = pos.to_bytes(2, "little")
            r[62], r[63] = 120, 30
            self.reg[mid] = r
            self._t[mid] = time.time()

    def _step(self, mid):
        r, now = self.reg[mid], time.time()
        dt, self._t[mid] = now - self._t[mid], now
        pos = int.from_bytes(r[56:58], "little")
        if r[40]:
            goal = int.from_bytes(r[42:44], "little")
            speed = max(1, int.from_bytes(r[46:48], "little"))
            step = int(np.clip(goal - pos, -speed * dt - 1, speed * dt + 1))
            pos = int(np.clip(pos + step, 0, 4095))
            st = self.stall.get(mid)
            if st is not None:                         # (limit raw, closing direction ±1): blocked by the object
                lim, sgn = st
                if (goal - lim) * sgn < -RELEASE_STEPS:  # commanded open past the object: let it go
                    self.stall.pop(mid)
                    if self.on_release:
                        self.on_release()
                elif (pos - lim) * sgn > 0:
                    pos, step = lim, 0
            err = goal - pos
            r[58:60] = (min(abs(step) * 10, 0x7FFF) | (0x8000 if step < 0 else 0)).to_bytes(2, "little")
            load = min(abs(err), 1000)
            r[60:62] = (load | (0x400 if err < 0 else 0)).to_bytes(2, "little")
            r[66] = 1 if err else 0
        else:
            r[58:62] = bytes(4)
            r[66] = 0
        r[56:58] = pos.to_bytes(2, "little")

    def read(self, sid, addr, n, retries=2):
        if sid not in self.reg:
            return None
        if addr <= 56 < addr + n:
            self._step(sid)
        return bytes(self.reg[sid][addr:addr + n])

    def write(self, sid, addr, data, retries=2):
        if sid not in self.reg:
            return False
        r = self.reg[sid]
        if addr == 40 and data[0] and not r[40]:
            self._t[sid] = time.time()
        r[addr:addr + len(data)] = bytes(data)
        if addr == 5 and data[0] != sid:                  # ID change
            self.reg[data[0]] = self.reg.pop(sid)
        return True

    def write_eeprom(self, sid, addr, data):
        return self.write(sid, addr, data)

    class _Ser:
        def close(self):
            pass

    ser = _Ser()


# ---------------------------------------------------------------- camera
W_D, H_D = 256, 192            # depth
W_C, H_C = 640, 480            # colour
FX_C = 500.0                   # colour focal length [px]
FLOOR_Y = -0.70                # floor height in the virtual ARKit world (camera starts ~0.7 m above it)
# objects on the floor in front of the robot (robot base at world x=z=0, facing +x in the viewer)
BOXES = [  # centre x, z, half-sizes (x, y, z), BGR colour
    (0.22, 0.00, (0.020, 0.035, 0.025), (60, 70, 210)),
    (0.18, -0.11, (0.015, 0.020, 0.015), (200, 140, 40)),
]
CYLINDERS = [(0.25, 0.10, 0.025, 0.09, (60, 170, 60))]   # x, z, radius, height, BGR


class VirtualCamera:
    """Object with the SDK client's interface: start(), stop(), is_connected, wait_for_frame(timeout)."""

    def __init__(self):
        self.is_connected = False
        self._t0 = time.time()
        self._n = 0
        self.hidden = set()          # object indices removed from the scene (dynamic-scene tests)
        self.offset = {}             # object index → ARKit-world translation (moved by a grasp, from the physics prediction)
        self.yaw = {}                # object index → turn about the vertical through its centre, rad (kitchen scene only)

    def start(self):
        self.is_connected = True
        return True

    def stop(self):
        self.is_connected = False

    def _pose(self, t):
        a = np.radians(60) * np.sin(0.25 * t)                 # sweep ±60° around the robot's front
        look = np.array([0.22, FLOOR_Y + 0.03, 0.0])
        pos = look + np.array([0.55 * np.cos(a), 0.45, 0.55 * np.sin(a)])
        f = look - pos; f /= np.linalg.norm(f)
        r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, pos      # ARKit: looks along −z, y up
        return T

    def _render(self, T, w, h, fx):
        cx, cy = w / 2, h / 2
        vv, uu = np.mgrid[0:h, 0:w]
        dirs = np.stack([(uu - cx) / fx, -(vv - cy) / fx, -np.ones_like(uu, float)], -1)   # ARKit camera axes
        d = dirs @ T[:3, :3].T
        o = T[:3, 3]
        t = np.full((h, w), np.inf)
        col = np.zeros((h, w, 3), np.uint8)
        tf = (FLOOR_Y - o[1]) / d[..., 1]
        hit = o + d * tf[..., None]
        floor = tf > 0
        t = np.where(floor, tf, t)
        checker = ((np.floor(hit[..., 0] / 0.1) + np.floor(hit[..., 2] / 0.1)) % 2).astype(bool)
        col[floor] = np.where(checker[floor, None], [190, 190, 190], [150, 150, 150])
        for i, (bx, bz, hs, c) in enumerate(BOXES):
            if i in self.hidden:
                continue
            off = self.offset.get(i, np.zeros(3))
            lo = np.array([bx - hs[0], FLOOR_Y, bz - hs[2]]) + off; hi = np.array([bx + hs[0], FLOOR_Y + 2 * hs[1], bz + hs[2]]) + off
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - o) / d, (hi - o) / d
            tn, tx = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
            m = (tn <= tx) & (tn > 0) & (tn < t)
            t = np.where(m, tn, t); col[m] = c
        for j, (qx, qz, rad, hgt, c) in enumerate(CYLINDERS):
            if len(BOXES) + j in self.hidden:
                continue
            off = self.offset.get(len(BOXES) + j, np.zeros(3))
            qx, qz, base_y = qx + off[0], qz + off[2], FLOOR_Y + off[1]
            ox, oz = o[0] - qx, o[2] - qz
            a = d[..., 0] ** 2 + d[..., 2] ** 2
            b = 2 * (ox * d[..., 0] + oz * d[..., 2])
            cc = ox ** 2 + oz ** 2 - rad ** 2
            disc = b * b - 4 * a * cc
            with np.errstate(invalid="ignore"):
                ts = (-b - np.sqrt(disc)) / (2 * a)
            y = o[1] + d[..., 1] * ts
            side = (disc > 0) & (ts > 0) & (y > base_y) & (y < base_y + hgt) & (ts < t)
            tt = (base_y + hgt - o[1]) / d[..., 1]
            hp = o + d * tt[..., None]
            top = (tt > 0) & ((hp[..., 0] - qx) ** 2 + (hp[..., 2] - qz) ** 2 < rad ** 2) & (tt < np.where(side, ts, t))
            t = np.where(side, ts, t); col[side] = c
            t = np.where(top, tt, t); col[top] = np.minimum(np.array(c) + 40, 255)
        depth = np.where(np.isfinite(t), t, 0).astype(np.float32)        # dirs have z = −1 → t is the z-depth
        return depth, col

    def object_centres(self):
        """index → ARKit-world centre of every visible object (with its current offset)."""
        out = {}
        for i, (bx, bz, hs, _) in enumerate(BOXES):
            if i not in self.hidden:
                out[i] = np.array([bx, FLOOR_Y + hs[1], bz]) + self.offset.get(i, 0)
        for j, (qx, qz, _, hgt, _) in enumerate(CYLINDERS):
            k = len(BOXES) + j
            if k not in self.hidden:
                out[k] = np.array([qx, FLOOR_Y + hgt / 2, qz]) + self.offset.get(k, 0)
        return out

    def wait_for_frame(self, timeout=1.0):
        if not self.is_connected:
            return None
        time.sleep(1 / 30)
        t = time.time() - self._t0
        T = self._pose(t)
        depth, _ = self._render(T, W_D, H_D, FX_C * W_D / W_C)
        depth += np.random.default_rng(self._n).normal(0, 0.002, depth.shape).astype(np.float32) * (depth > 0)
        _, colour = self._render(T, W_C // 2, H_C // 2, FX_C / 2)
        import cv2
        colour = cv2.resize(colour, (W_C, H_C), interpolation=cv2.INTER_NEAREST)
        self._n += 1
        K = types.SimpleNamespace(fx=FX_C, fy=FX_C, ppx=W_C / 2, ppy=H_C / 2, width=W_C, height=H_C)
        return types.SimpleNamespace(frame_id=self._n, timestamp=t, depth=depth, color=colour, confidence=None,
                                     intrinsics=K, transform=T.astype(np.float32), imu=None)
