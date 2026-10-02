"""iPhone (Record3DStream app) → live point cloud for the arm viewer.

Input  (USB, ~60 fps): RGB 1920×1440 JPEG, depth 256×192 float32 [m], confidence 256×192 uint8 {0,1,2},
                       intrinsics at RGB resolution, camera-to-world 4×4 (ARKit: x right, y up, z backward)
Output (≈10 Hz):       points N×3 float32 [m] in ARKit world (y up, gravity aligned), colors N×3 uint8,
                       camera pose 4×4, AprilTag detections (T_world_tag 4×4) for robot registration
"""
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "r3ds_sdk"))
from sdk import IPhoneSensorClient  # noqa: E402  (personal-use SDK, see r3ds_sdk/SOURCE.txt)

PORT = 8888
STRIDE = 2            # depth subsampling: 256×192 → 128×96 ≈ 12k points
MAX_DEPTH_M = 2.5
MIN_CONF = 1          # drop ARKit "low" confidence depth
TAG_SIZE_M = 0.06     # printed AprilTag 36h11 black-square edge length
# OpenCV camera (x right, y down, z forward) → ARKit camera (x right, y up, z backward)
CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])


class CameraStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = {"connected": False, "error": None, "fps": 0.0, "points": 0, "tags": {}}
        self.packet = None          # latest binary point-cloud message for the browser
        self.seq = 0
        self.T_world_cam = None
        self._fwd = None
        self._client = None
        self._det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                            cv2.aruco.DetectorParameters())
        threading.Thread(target=self._run, daemon=True).start()

    # ---- connection
    def _connect(self):
        if self._fwd is None or self._fwd.poll() is not None:
            # TCP over USB: localhost:PORT → iPhone:PORT
            self._fwd = subprocess.Popen([sys.executable, "-m", "pymobiledevice3", "usbmux", "forward", str(PORT), str(PORT)],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        client = IPhoneSensorClient("localhost", port=PORT)
        for _ in range(20):
            time.sleep(0.5)
            if client.start():
                return client
        raise RuntimeError("iPhone 스트림에 연결하지 못함 (USB 연결·앱 스트리밍 확인)")

    def _run(self):
        last_emit, n, t0 = 0.0, 0, time.time()
        while True:
            try:
                if self._client is None or not self._client.is_connected:
                    self.status.update(connected=False)
                    self._client = self._connect()
                    self.status.update(connected=True, error=None)
                f = self._client.wait_for_frame(timeout=1.0)
                if f is None:
                    continue
                n += 1
                now = time.time()
                if now - t0 >= 1.0:
                    self.status["fps"] = round(n / (now - t0), 1)
                    n, t0 = 0, now
                if now - last_emit >= 0.1:
                    last_emit = now
                    self._process(f)
            except Exception as e:
                self.status.update(connected=False, error=str(e))
                try:
                    self._client and self._client.stop()
                except Exception:
                    pass
                self._client = None
                time.sleep(1.0)

    # ---- per-frame processing
    def _process(self, f):
        T = f.transform.astype(np.float64)
        K = f.intrinsics
        depth, color = f.depth, f.color
        conf = getattr(f, "confidence", None)
        dh, dw = depth.shape
        sx, sy = dw / K.width, dh / K.height   # intrinsics are given at RGB resolution
        fx, fy, cx, cy = K.fx * sx, K.fy * sy, K.ppx * sx, K.ppy * sy

        v, u = np.mgrid[0:dh:STRIDE, 0:dw:STRIDE]
        d = depth[v, u]
        ok = np.isfinite(d) & (d > 0.05) & (d < MAX_DEPTH_M)
        if conf is not None and conf.shape == depth.shape:
            ok &= conf[v, u] >= MIN_CONF
        u, v, d = u[ok], v[ok], d[ok]
        # back-project in ARKit camera axes (image v grows downward, camera looks along -z)
        pc = np.stack([(u - cx) / fx * d, -(v - cy) / fy * d, -d, np.ones_like(d)], axis=0)
        pw = (T @ pc)[:3].T.astype(np.float32)
        ch, cw = color.shape[:2]
        rgb = color[(v * ch // dh).clip(0, ch - 1), (u * cw // dw).clip(0, cw - 1)][:, ::-1]  # BGR→RGB
        tags = self._detect_tags(color, K, T)

        header = np.array([len(pw)], np.uint32).tobytes() + T.astype(np.float32).T.tobytes()  # column-major for three.js
        with self.lock:
            self.packet = header + pw.tobytes() + np.ascontiguousarray(rgb, np.uint8).tobytes()
            self.seq += 1
            self.T_world_cam = T
            self.status.update(points=int(len(pw)), tags=tags)

    def _detect_tags(self, color, K, T_world_cam):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._det.detectMarkers(gray)
        out = {}
        if ids is None:
            return out
        Kc = np.array([[K.fx, 0, K.ppx], [0, K.fy, K.ppy], [0, 0, 1]], np.float64)
        h = TAG_SIZE_M / 2  # corner order: TL, TR, BR, BL in the tag plane (z out of the tag)
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
        for c, i in zip(corners, ids.ravel()):
            ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), Kc, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            T_cv_tag = np.eye(4)
            T_cv_tag[:3, :3], T_cv_tag[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
            T_world_tag = T_world_cam @ CV_TO_ARKIT @ T_cv_tag
            out[int(i)] = {"T": T_world_tag.round(5).tolist(), "dist_m": round(float(np.linalg.norm(tvec)), 3)}
        return out

    def latest_packet(self):
        with self.lock:
            return self.packet

    def latest(self):
        with self.lock:
            return self.seq, self.packet
