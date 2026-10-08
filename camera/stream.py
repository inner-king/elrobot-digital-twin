"""iPhone (Record3DStream app) → live point cloud for the arm viewer.

Input  (USB, ~60 fps): RGB 1920×1440 JPEG, depth 256×192 float32 [m], confidence 256×192 uint8 {0,1,2},
                       intrinsics at RGB resolution, camera-to-world 4×4 (ARKit: x right, y up, z backward)
Output (≈10 Hz):       points N×3 float32 [m] in ARKit world (y up, gravity aligned), colors N×3 uint8,
                       camera pose 4×4, AprilTag detections (T_world_tag 4×4) for robot registration,
                       ChArUco floor board (camera/board.json) → floor frame T_world_floor 4×4
                       (origin: board centre, y: up out of the board, x: board left→right)
"""
import json
import subprocess
from collections import deque
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "r3ds_sdk"))
from sdk import IPhoneSensorClient  # noqa: E402  (personal-use SDK, see r3ds_sdk/SOURCE.txt)
from recon import Reconstructor  # noqa: E402

PORT = 8888
CONFIG_FILE = Path(__file__).parent / "camera.json"   # {"link": "usb" | "wifi", "host": "<iPhone IP>"}
FRAME_TIMEOUT_S = 2.0   # "connected" = a frame arrived within this window
# automatic floor: lowest strong horizontal surface in the gravity-aligned (ARKit y-up) height histogram
AF_BIN_M, AF_RANGE_M, AF_DECAY = 0.01, 4.0, 0.97
AF_MIN_SHARE = 0.05     # a floor peak must hold ≥5 % of the strongest peak (floor is often only partly visible)
AF_MIN_POINTS = 300
STRIDE = 2            # depth subsampling: 256×192 → 128×96 ≈ 12k points
MAX_DEPTH_M = 2.5
MIN_CONF = 1          # drop ARKit "low" confidence depth
TAG_SIZE_M = 0.06     # printed AprilTag 36h11 black-square edge length
# OpenCV camera (x right, y down, z forward) → ARKit camera (x right, y up, z backward)
CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])
BOARD_CFG = json.loads((Path(__file__).parent / "board.json").read_text())   # design (what make_board.py prints)
PLANE_INLIER_M = 0.006   # LiDAR points within 6 mm of the board plane count as inliers
PLANE_MIN_POINTS = 40
BOARD_MIN_CORNERS = 6
BOARD_EVERY = 3       # run the board detector on every 3rd processed frame (≈3 Hz)
# board axes (x right, y down the rows, z into the paper) → floor axes (x right, y up, z = x × y)
R_BOARD_FLOOR = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)
# floor marker: one AprilTag 36h11 (TAG_SIZE_M). Tag axes (x right, y up, z out of the paper) → floor axes
FLOOR_TAG_ID = 0
R_TAG_FLOOR = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)
# floor refinement at lock time: LiDAR points within this band of the marker plane
FLOOR_BAND_M, FLOOR_RADIUS_M, FLOOR_MIN_POINTS = 0.02, 1.5, 300
# a 6 cm tag covers only a few depth pixels: fit the plane on the floor around it (the paper lies on the floor)
TAG_PLANE_GROW, TAG_PLANE_MIN_RADIUS_PX = 4.0, 14


def _umeyama(src, dst):
    """Similarity dst ≈ s·R·src + t (least squares)."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (A ** 2).sum(axis=1).mean())
    return s, R, md - s * R @ ms


def _mean_pose(Ts):
    """Average of rigid transforms: mean translation, rotation projected back onto SO(3)."""
    Ts = np.asarray(Ts)
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].sum(axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, Ts[:, :3, 3].mean(axis=0)
    return T


class CameraStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = {"connected": False, "error": None, "fps": 0.0, "points": 0, "tags": {}}
        self.packet = None          # latest binary point-cloud message for the browser
        self.seq = 0
        self.T_world_cam = None
        self._fwd = None
        self._client = None
        self._last_frame_t = 0.0
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
        except Exception:
            cfg = {}
        self.status.update(link=cfg.get("link", "usb"), host=cfg.get("host", ""), stage="시작", usb_devices=[],
                           virtual_hidden=[], virtual_scene=cfg.get("virtual_scene", "kitchen"), virtual_objects=[])
        self._virtual = None
        threading.Thread(target=self._watch_usb, daemon=True).start()
        self._det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                            cv2.aruco.DetectorParameters())
        self.board_cfg = dict(BOARD_CFG)
        self._build_board()
        self._board_hist = deque(maxlen=15)   # recent T_world_floor detections
        self._nproc = 0
        self.status["board"] = None           # live detection
        self.status["floor"] = None           # locked floor frame (valid for this ARKit session only)
        self._af_hist = np.zeros(int(2 * AF_RANGE_M / AF_BIN_M))
        self.status["auto_floor"] = None
        self.recon = Reconstructor()
        self.status["recon"] = self.recon.status
        from detect import Detector
        self.detect = Detector(self.recon)
        self.recon.labeler = self.detect.label
        self.recon.instances = self.detect.instances
        self.status["detect"] = self.detect.status
        threading.Thread(target=self._run, daemon=True).start()

    # ---- connection
    def _connect(self):
        if self.status["link"] == "virtual":
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from virtual import VirtualCamera
            import virtual_kitchen
            self._stop_forwarder()
            scene = self.status.get("virtual_scene", "kitchen")
            client = virtual_kitchen.KitchenCamera() if scene == "kitchen" and virtual_kitchen.available() else VirtualCamera()
            self.status["virtual_objects"] = getattr(client, "names", ["빨간 상자", "파란 상자", "초록 원통"])
            self.status["virtual_hidden"] = []
            client.start()
            self._virtual = client
            self.detect.virtual = client
            return client
        self._virtual = None
        self.detect.virtual = None
        if self.status["link"] == "wifi":
            host = self.status.get("host") or ""
            if not host:
                raise RuntimeError("WiFi 모드: iPhone IP 주소를 입력하세요 (앱 화면에 표시)")
            self._stop_forwarder()
            self.status["stage"] = f"WiFi {host}:{PORT} 연결 중"
            client = IPhoneSensorClient(host, port=PORT)
            if not client.start():
                raise RuntimeError(f"{host}:{PORT}에 연결하지 못함 (같은 WiFi인지, 앱 스트리밍 중인지 확인)")
            return client
        # USB: only try when the iPhone is actually on the USB mux, otherwise the local forwarder accepts
        # the TCP connection and drops it immediately (that made the status flap connected/disconnected)
        import asyncio
        import inspect
        from pymobiledevice3.usbmux import list_devices
        devs = list_devices()
        if inspect.isawaitable(devs):   # async in recent pymobiledevice3
            devs = asyncio.run(devs)
        if not any(d.connection_type == "USB" for d in devs):
            self._stop_forwarder()
            raise RuntimeError("iPhone이 USB로 연결되지 않음")
        if self._fwd is None or self._fwd.poll() is not None:
            # TCP over USB: localhost:PORT → iPhone:PORT
            self._fwd = subprocess.Popen([sys.executable, "-m", "pymobiledevice3", "usbmux", "forward", str(PORT), str(PORT)],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.status["stage"] = "USB 연결 중"
        client = IPhoneSensorClient("localhost", port=PORT)
        for _ in range(20):
            time.sleep(0.5)
            if client.start():
                return client
        raise RuntimeError("USB 포워딩은 됐지만 앱 응답 없음 (앱 스트리밍 확인)")

    def _watch_usb(self):
        """iPhones on the USB mux, refreshed every 3 s for the connection list."""
        import asyncio
        import inspect
        while True:
            try:
                from pymobiledevice3.usbmux import list_devices
                devs = list_devices()
                if inspect.isawaitable(devs):
                    devs = asyncio.run(devs)
                self.status["usb_devices"] = [{"serial": d.serial} for d in devs if d.connection_type == "USB"]
            except Exception:
                self.status["usb_devices"] = []
            time.sleep(3)

    def _stop_forwarder(self):
        if self._fwd is not None and self._fwd.poll() is None:
            self._fwd.terminate()
        self._fwd = None

    def configure(self, link, host=""):
        if link not in ("usb", "wifi", "virtual"):
            raise RuntimeError(f"알 수 없는 연결 방식 {link}")
        self.status.update(link=link, host=host.strip())
        CONFIG_FILE.write_text(json.dumps({"link": link, "host": host.strip(), "virtual_scene": self.status.get("virtual_scene", "kitchen")}))
        try:
            self._client and self._client.stop()   # the worker reconnects with the new settings
        except Exception:
            pass

    def _run(self):
        last_emit, n, t0 = 0.0, 0, time.time()
        while True:
            try:
                if self._client is None or not self._client.is_connected:
                    self.status.update(connected=False, fps=0.0)
                    self._client = self._connect()
                    self.status.update(error=None, stage="연결됨, 프레임 대기")
                f = self._client.wait_for_frame(timeout=1.0)
                streaming = time.time() - self._last_frame_t < FRAME_TIMEOUT_S
                self.status["connected"] = streaming
                if f is None:
                    if not streaming:
                        self.status.update(fps=0.0, stage="연결됨, 앱에서 프레임이 안 옴")
                    continue
                self._last_frame_t = time.time()
                self.status.update(connected=True, stage="스트리밍")
                n += 1
                now = time.time()
                if now - t0 >= 1.0:
                    self.status["fps"] = round(n / (now - t0), 1)
                    n, t0 = 0, now
                if now - last_emit >= 0.1:
                    last_emit = now
                    self._process(f)
            except Exception as e:
                self.status.update(connected=False, fps=0.0, error=str(e), stage="재시도 대기")
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
        # an all-zero confidence map carries no information (seen over WiFi): ignore it instead of dropping every point
        if conf is not None and (conf.shape != depth.shape or not conf.any()):
            conf = None
        self.status["confidence"] = conf is not None
        dh, dw = depth.shape
        sx, sy = dw / K.width, dh / K.height   # intrinsics are given at RGB resolution
        fx, fy, cx, cy = K.fx * sx, K.fy * sy, K.ppx * sx, K.ppy * sy

        v, u = np.mgrid[0:dh:STRIDE, 0:dw:STRIDE]
        d = depth[v, u]
        ok = np.isfinite(d) & (d > 0.05) & (d < MAX_DEPTH_M)
        if conf is not None:
            ok &= conf[v, u] >= MIN_CONF
        u, v, d = u[ok], v[ok], d[ok]
        # back-project in ARKit camera axes (image v grows downward, camera looks along -z)
        pc = np.stack([(u - cx) / fx * d, -(v - cy) / fy * d, -d, np.ones_like(d)], axis=0)
        pw = (T @ pc)[:3].T.astype(np.float32)
        ch, cw = color.shape[:2]
        rgb = color[(v * ch // dh).clip(0, ch - 1), (u * cw // dw).clip(0, cw - 1)][:, ::-1]  # BGR→RGB
        tags = self._detect_tags(color, K, T)
        self._nproc += 1
        self._last_world_pts = pw
        self._update_auto_floor(pw)
        fl, af = self.status.get("floor"), self.status.get("auto_floor")
        self.recon.floor_y = fl["T"][1][3] if fl else af["y"] if af else None     # ARKit y (gravity-up) of the floor
        if self.recon.status["running"]:
            dclean = np.where(np.isfinite(depth) & (depth > 0.05) & (depth < MAX_DEPTH_M), depth, 0).astype(np.float32)
            if conf is not None:
                dclean[conf < MIN_CONF] = 0
            small = cv2.cvtColor(cv2.resize(color, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            self.recon.submit(dclean, small, fx, fy, cx, cy, T @ CV_TO_ARKIT)
            self.detect.submit(color, dclean, [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], T @ CV_TO_ARKIT)
        if self._nproc % BOARD_EVERY == 0:
            fl = self._detect_board(color, K, T, depth, conf)
            if "T" not in fl and self._floor_tag_px is not None:   # no board in view: use the floor tag
                fl = self._detect_floor_tag(K, T, depth, conf)
            self.status["board"] = fl

        header = np.array([len(pw)], np.uint32).tobytes() + T.astype(np.float32).T.tobytes()  # column-major for three.js
        with self.lock:
            self.packet = header + pw.tobytes() + np.ascontiguousarray(rgb, np.uint8).tobytes()
            self.seq += 1
            self.T_world_cam = T
            self.status.update(points=int(len(pw)), tags=tags)

    def _update_auto_floor(self, pw):
        """Floor height in ARKit world: gravity is y, a horizontal surface is a sharp peak in the y histogram."""
        if len(pw) < AF_MIN_POINTS:
            return
        y = pw[:, 1].astype(np.float64)
        h, _ = np.histogram(y, bins=len(self._af_hist), range=(-AF_RANGE_M, AF_RANGE_M))
        self._af_hist = self._af_hist * AF_DECAY + h
        sm = np.convolve(self._af_hist, [1, 2, 1], mode="same") / 4
        # steady-state of the decayed sum is n/(1-decay): require ≈100 points per frame in the bin as well
        thr = max(sm.max() * AF_MIN_SHARE, 100 / (1 - AF_DECAY))
        peaks = np.nonzero((sm >= thr) & (sm >= np.roll(sm, 1)) & (sm >= np.roll(sm, -1)))[0]
        if not len(peaks):
            return
        k = peaks.min()                                              # lowest qualifying peak
        y0 = -AF_RANGE_M + (k + 0.5) * AF_BIN_M
        near = y[np.abs(y - y0) < 0.015]
        if len(near) >= AF_MIN_POINTS // 3:
            y0 = float(np.median(near))
        prev = self.status.get("auto_floor")
        if prev and abs(prev["y"] - y0) < 0.03:                      # smooth small changes, jump on big ones
            y0 = 0.8 * prev["y"] + 0.2 * y0
        self.status["auto_floor"] = {"y": round(y0, 4), "pts": int(len(near)),
                                     "share": round(float(sm[k] / sm.max()), 2),
                                     "is_lowest_peak_strongest": bool(sm[k] == sm.max())}

    def _detect_tags(self, color, K, T_world_cam):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._det.detectMarkers(gray)
        out = {}
        self._floor_tag_px = None
        if ids is None:
            return out
        Kc = np.array([[K.fx, 0, K.ppx], [0, K.fy, K.ppy], [0, 0, 1]], np.float64)
        h = TAG_SIZE_M / 2  # corner order: TL, TR, BR, BL in the tag plane (z out of the tag)
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
        for c, i in zip(corners, ids.ravel()):
            if int(i) == FLOOR_TAG_ID:
                self._floor_tag_px = c.reshape(4, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), Kc, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            T_cv_tag = np.eye(4)
            T_cv_tag[:3, :3], T_cv_tag[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
            T_world_tag = T_world_cam @ CV_TO_ARKIT @ T_cv_tag
            out[int(i)] = {"T": T_world_tag.round(5).tolist(), "dist_m": round(float(np.linalg.norm(tvec)), 3)}
        return out

    def _build_board(self):
        b = self.board_cfg
        self._board = cv2.aruco.CharucoBoard((b["squares_x"], b["squares_y"]), b["square_m"], b["marker_m"],
                                             cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b["dictionary"])))
        self._board_det = cv2.aruco.CharucoDetector(self._board)


    def _detect_board(self, color, K, T_world_cam, depth, conf):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        cc, ci, _, _ = self._board_det.detectBoard(gray)
        n = 0 if ci is None else len(ci)
        if n < BOARD_MIN_CORNERS:
            return {"source": "보드", "corners": n}
        obj, img = self._board.matchImagePoints(cc, ci)
        b = self.board_cfg
        centre = [b["squares_x"] * b["square_m"] / 2, b["squares_y"] * b["square_m"] / 2, 0]
        return self._floor_pose("보드", obj.reshape(-1, 3), img.reshape(-1, 2), R_BOARD_FLOOR, centre,
                                K, T_world_cam, depth, conf, design_square_m=b["square_m"])

    def _detect_floor_tag(self, K, T_world_cam, depth, conf):
        h = TAG_SIZE_M / 2  # TL, TR, BR, BL, as returned by the ArUco detector
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
        return self._floor_pose(f"태그 ID {FLOOR_TAG_ID}", obj, self._floor_tag_px, R_TAG_FLOOR, [0, 0, 0],
                                K, T_world_cam, depth, conf, design_square_m=TAG_SIZE_M,
                                grow=TAG_PLANE_GROW, min_radius_px=TAG_PLANE_MIN_RADIUS_PX, allow_pnp=False)

    def _floor_pose(self, source, obj, img, R_obj_floor, origin_obj, K, T_world_cam, depth, conf, design_square_m,
                    grow=1.0, min_radius_px=0, allow_pnp=True):
        """Pattern gives orientation + corner identity; LiDAR gives metric distance and size.

        corners (px) → rays; LiDAR points inside the corners' hull → plane (OpenCV camera frame);
        ray ∩ plane → metric 3D corners; similarity fit design→3D (Umeyama) → R, t and the printed scale.
        Falls back to plain PnP (design size) when the depth around the pattern is too sparse.
        """
        obj, img = np.asarray(obj, np.float64), np.asarray(img, np.float64)
        Kc = np.array([[K.fx, 0, K.ppx], [0, K.fy, K.ppy], [0, 0, 1]], np.float64)
        out = {"source": source, "corners": len(img)}
        R = t = None
        scale = 1.0
        plane = self._board_plane(img, K, depth, conf, grow, min_radius_px)
        if plane is not None:
            nrm, c0, inl, rms = plane
            rays = np.c_[(img[:, 0] - K.ppx) / K.fx, (img[:, 1] - K.ppy) / K.fy, np.ones(len(img))]
            X = rays * ((nrm @ c0) / (rays @ nrm))[:, None]            # metric corners, OpenCV camera frame
            scale, R, t = _umeyama(obj, X)
            resid = np.linalg.norm((scale * obj) @ R.T + t - X, axis=1)
            out.update(method="LiDAR", plane_pts=inl, plane_rms_mm=round(rms * 1000, 2),
                       fit_mm=round(float(np.sqrt((resid ** 2).mean()) * 1000), 2),
                       size_mm=round(design_square_m * scale * 1000, 2), print_scale=round(scale * 100, 1))
        if R is None and not allow_pnp:
            # 4-corner PnP on a small tag is ambiguous/noisy (measured: 71° tilt at 1.5 m) — better no floor than a wrong one
            out["note"] = "LiDAR 평면을 못 맞춤 (너무 멀거나 깊이 부족) — 더 가까이"
            return out
        if R is None:
            flags = cv2.SOLVEPNP_IPPE_SQUARE if len(obj) == 4 else cv2.SOLVEPNP_ITERATIVE
            ok, rvec, tvec = cv2.solvePnP(obj, img, Kc, None, flags=flags)
            if not ok:
                return out
            R, t = cv2.Rodrigues(rvec)[0], tvec.ravel()
            out.update(method="PnP (설계 크기)")
        proj = (Kc @ ((scale * obj) @ R.T + t).T).T
        proj = proj[:, :2] / proj[:, 2:]
        out["reproj_px"] = round(float(np.sqrt(((proj - img) ** 2).sum(axis=1).mean())), 2)
        T_cv_obj = np.eye(4)
        T_cv_obj[:3, :3], T_cv_obj[:3, 3] = R, t
        T_obj_floor = np.eye(4)
        T_obj_floor[:3, :3], T_obj_floor[:3, 3] = R_obj_floor, scale * np.asarray(origin_obj, np.float64)
        T_world_floor = T_world_cam @ CV_TO_ARKIT @ T_cv_obj @ T_obj_floor
        self._board_hist.append(T_world_floor)
        up = T_world_floor[:3, 1]
        out.update(dist_m=round(float(np.linalg.norm(t)), 3),
                   # angle between the pattern normal and ARKit gravity-up: how level the floor is
                   tilt_deg=round(float(np.degrees(np.arccos(np.clip(up[1], -1, 1)))), 2),
                   T=T_world_floor.round(5).tolist())
        return out

    def _refine_floor(self, T_world_floor):
        """Re-fit floor tilt + height on all LiDAR points near the marker plane; keep origin and yaw."""
        P = getattr(self, "_last_world_pts", None)
        if P is None or len(P) == 0:
            return T_world_floor, None
        Tfw = np.linalg.inv(T_world_floor)
        q = P.astype(np.float64) @ Tfw[:3, :3].T + Tfw[:3, 3]                 # points in the floor frame
        sel = (np.abs(q[:, 1]) < FLOOR_BAND_M) & (np.hypot(q[:, 0], q[:, 2]) < FLOOR_RADIUS_M)
        if sel.sum() < FLOOR_MIN_POINTS:
            return T_world_floor, {"floor_pts": int(sel.sum())}
        Q = q[sel]
        for _ in range(3):
            c0 = Q.mean(axis=0)
            nrm = np.linalg.svd(Q - c0)[2][-1]
            nrm = nrm if nrm[1] > 0 else -nrm
            d = (Q - c0) @ nrm
            Q = Q[np.abs(d) < 0.006]
        # new floor frame: up = fitted normal, x = old x projected onto the plane, origin = old origin dropped onto the plane
        y = nrm
        x = np.array([1.0, 0, 0]) - y[0] * y
        x /= np.linalg.norm(x)
        z = np.cross(x, y)
        o = -((0 - c0) @ y) * y                                                # old origin projected onto the plane
        T_old_new = np.eye(4)
        T_old_new[:3, :3] = np.c_[x, y, z]
        T_old_new[:3, 3] = o
        T = T_world_floor @ T_old_new
        info = {"floor_pts": int(len(Q)), "floor_rms_mm": round(float(np.sqrt(((Q - c0) @ y) .var()) * 1000), 2),
                "tilt_fix_deg": round(float(np.degrees(np.arccos(np.clip(y[1], -1, 1)))), 3),
                "height_fix_mm": round(float(o[1] * 1000), 2)}
        return T, info

    def _board_plane(self, img_px, K, depth, conf, grow=1.0, min_radius_px=0):
        """Plane through the LiDAR points inside the detected corners' hull (OpenCV camera frame).

        grow / min_radius_px enlarge the hull about its centre (depth pixels) to include the floor around a small tag."""
        if depth is None:
            return None
        dh, dw = depth.shape
        sx, sy = dw / K.width, dh / K.height
        pts = img_px * [sx, sy]
        c = pts.mean(axis=0)
        r = np.linalg.norm(pts - c, axis=1).max()
        k = max(grow, min_radius_px / r if r > 0 else grow)
        hull = cv2.convexHull((c + (pts - c) * k).astype(np.float32))
        mask = np.zeros((dh, dw), np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
        if grow == 1.0 and min_radius_px == 0:
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))      # keep away from the board edge
        v, u = np.nonzero(mask)
        d = depth[v, u]
        ok = np.isfinite(d) & (d > 0.05)
        if conf is not None:
            ok &= conf[v, u] >= MIN_CONF
        u, v, d = u[ok], v[ok], d[ok]
        if len(d) < PLANE_MIN_POINTS:
            return None
        fx, fy, cx, cy = K.fx * sx, K.fy * sy, K.ppx * sx, K.ppy * sy
        P = np.c_[(u - cx) / fx * d, (v - cy) / fy * d, d]
        keep = np.ones(len(P), bool)
        for _ in range(3):                                          # fit, drop outliers, refit
            c0 = P[keep].mean(axis=0)
            nrm = np.linalg.svd(P[keep] - c0)[2][-1]
            dist = np.abs((P - c0) @ nrm)
            new = dist < PLANE_INLIER_M
            if new.sum() < PLANE_MIN_POINTS:
                return None
            keep = new
        if nrm @ c0 > 0:                                            # normal towards the camera
            nrm = -nrm
        rms = float(np.sqrt((dist[keep] ** 2).mean()))
        return nrm, c0, int(keep.sum()), rms

    def handle(self, c):
        if c["type"] == "cam_config":
            return self.configure(c.get("link", "usb"), c.get("host", ""))
        if c["type"] == "cam_virtual_scene":                   # "kitchen" (meshes) | "boxes"
            self.status["virtual_scene"] = c["scene"]
            self.recon.handle({"type": "recon_reset"})
            return self.configure(self.status["link"], self.status.get("host", ""))
        if c["type"] == "cam_virtual_toggle":                  # remove / put back a virtual object (dynamic-scene test)
            if self._virtual is None:
                raise RuntimeError("가상 카메라가 아닙니다")
            i = int(c["index"])
            self._virtual.hidden ^= {i}
            self.status["virtual_hidden"] = sorted(self._virtual.hidden)
            return
        if c["type"].startswith("recon_"):
            return self.recon.handle(c)
        if c["type"].startswith("detect_"):
            return self.detect.handle(c)
        if c["type"] == "floor_lock":
            if len(self._board_hist) < 5:
                raise RuntimeError("보드 검출이 부족합니다 (5회 이상 필요)")
            Ts = np.array(self._board_hist)
            T = _mean_pose(Ts)
            spread_mm = float(np.linalg.norm(Ts[:, :3, 3] - T[:3, 3], axis=1).max() * 1000)
            T, refine = self._refine_floor(T)
            self.status["floor"] = {"T": T.round(5).tolist(), "n": len(Ts), "spread_mm": round(spread_mm, 2),
                                    "refine": refine, "time": time.strftime("%H:%M:%S")}
        elif c["type"] == "floor_clear":
            self.status["floor"] = None


    def latest_packet(self):
        with self.lock:
            return self.packet

    def latest(self):
        with self.lock:
            return self.seq, self.packet
