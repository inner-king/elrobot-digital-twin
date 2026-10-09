"""[ours] Raw capture of EVERY received camera frame (before any throttling), for offline reconstruction / GS tests.

Input : SDK Frame at the stream rate (≈30 fps): JPEG colour as received (1920×1440), depth 256×192 float32 [m],
        confidence 256×192 uint8 (0/1/2), T_world_cam 4×4 (ARKit, camera-to-world), K 3×3 at colour resolution
Output: captures/<YYYYmmdd_HHMMSS>/
          rgb/<frame_id:06d>.jpg     colour bytes exactly as sent by the phone (no re-encode)
          depth/<frame_id:06d>.png   uint16 depth in mm (0 = none; LiDAR resolution is ≈2–3 cm, mm loses nothing)
          conf/<frame_id:06d>.png    uint8 confidence (missing when the frame had none)
          frames.jsonl               one line per frame: id, t [s], T (16, row-major), K (9), rgb_wh, depth_wh, conf
A writer thread does the disk I/O; when it falls behind, frames are dropped and counted (never block the camera).
"""
import json
import shutil
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "captures"
QUEUE_MAX = 90          # ≈3 s at 30 fps
MIN_FREE_GB = 3.0       # stop recording below this free disk space (≈0.6 GB per minute at 30 fps)


class RawRecorder:
    def __init__(self):
        self._q = queue.Queue(maxsize=QUEUE_MAX)
        self._dir = None
        self._log = None
        self.status = {"on": True, "dir": None, "frames": 0, "dropped": 0, "bytes": 0, "queued": 0}
        threading.Thread(target=self._loop, daemon=True).start()

    def new_session(self):
        """next frame starts a new capture folder (camera reconnect / another camera)"""
        self._q.put(None)

    def set_on(self, on):
        if on:
            self.status["error"] = None
        self.status["on"] = bool(on)
        if not on:
            self.new_session()

    def put(self, f):
        if not self.status["on"]:
            return
        try:
            self._q.put_nowait(f)
        except queue.Full:
            self.status["dropped"] += 1
        self.status["queued"] = self._q.qsize()

    def _open(self):
        d = ROOT / time.strftime("%Y%m%d_%H%M%S")
        for sub in ("rgb", "depth", "conf"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        self._dir, self._log = d, open(d / "frames.jsonl", "a", buffering=1)
        self.status.update(dir=str(d.relative_to(ROOT.parent)), frames=0, dropped=0, bytes=0)

    def _close(self):
        if self._log:
            self._log.close()
        self._dir, self._log = None, None

    def _loop(self):
        while True:
            f = self._q.get()
            self.status["queued"] = self._q.qsize()
            if f is None:
                self._close()
                continue
            try:
                if self._dir is None:
                    self._open()
                if self.status["frames"] % 30 == 0:
                    free = shutil.disk_usage(ROOT).free / 1e9
                    self.status["free_gb"] = round(free, 1)
                    if free < MIN_FREE_GB:
                        self.status["error"] = f"디스크 여유 {free:.1f} GB < {MIN_FREE_GB} GB: 원본 저장 중지"
                        self.set_on(False)
                        continue
                self._write(f)
            except Exception as e:
                self.status["error"] = str(e)

    def _write(self, f):
        name = f"{int(f.frame_id):06d}"
        n = 0
        jpg = getattr(f, "rgb_jpeg", None)
        if jpg is None:                                   # SDK without raw bytes: lossless fallback is too big, q95
            jpg = cv2.imencode(".jpg", f.color, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        (self._dir / "rgb" / f"{name}.jpg").write_bytes(jpg)
        n += len(jpg)
        dmm = np.clip(np.rint(np.nan_to_num(f.depth, nan=0.0) * 1000.0), 0, 65535).astype(np.uint16)
        p = self._dir / "depth" / f"{name}.png"
        cv2.imwrite(str(p), dmm)
        n += p.stat().st_size
        conf = getattr(f, "confidence", None)
        if conf is not None:
            p = self._dir / "conf" / f"{name}.png"
            cv2.imwrite(str(p), conf)
            n += p.stat().st_size
        K = f.intrinsics
        Km = [K.fx, 0, K.ppx, 0, K.fy, K.ppy, 0, 0, 1]
        w, h = K.width, K.height
        self._log.write(json.dumps({"id": int(f.frame_id), "t": float(f.timestamp),
                                    "T": np.asarray(f.transform, np.float64).ravel().round(7).tolist(),
                                    "K": [round(float(v), 4) for v in Km],
                                    "rgb_wh": [int(w), int(h)], "depth_wh": [int(f.depth.shape[1]), int(f.depth.shape[0])],
                                    "conf": conf is not None}) + "\n")
        self.status["frames"] += 1
        self.status["bytes"] += n
