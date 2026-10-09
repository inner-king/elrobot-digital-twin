"""[ours] Object segmentation BEFORE fusion: one keyframe (colour + LiDAR depth + pose) → object label image.

Input : colour BGR (≈640×480), depth 256×192 float32 [m] (0 = none), depth intrinsics K 3×3, T_world_cam 4×4
        (OpenCV camera axes, ARKit world: y up), optional prior height of the work surface (ARKit y)
Output: labels 256×192 int16 (0 = support / background, 1..n = objects in this frame), plane height, plane σ [m]

Method (thresholds come from the frame itself, not from a scene):
  1. support plane: gravity-aligned, 1-D RANSAC over point heights; noise σ = 1.4826·MAD of its inliers
  2. "above the support" = height > 3σ; SAM 2.1-tiny point prompts on a grid inside that region
     (Meta SAM 2.1 hiera-tiny via ultralytics, class-agnostic)
  3. each pixel goes to the smallest mask containing it (objects on a board keep their own mask, the board the rest);
     a mask is an object when its median height above the plane exceeds 3σ (the support itself does not)
Measured on the virtual kitchen against ground-truth ids (4 views × 8 objects): mean IoU 0.879, 24/32 ≥ 0.8, no mask
covering two objects, 370–640 ms per frame on Apple MPS; weakest: the knife blade (IoU ≈ 0.5).
"""
import threading
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "sam2.1_t.pt"
PROMPT_STRIDE = 24          # px on the colour image between point prompts
MIN_MASK_PX = 200           # colour pixels
PLANE_WINDOW_M = 0.01       # RANSAC inlier band for the support height
PRIOR_WINDOW_M = 0.05       # with a known work-surface height, search the plane this close to it
SIGMA_K = 3.0


def back_project(depth, K, T):
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    P = np.stack([(uu - K[0, 2]) / K[0, 0] * depth, (vv - K[1, 2]) / K[1, 1] * depth, depth], -1)
    return P @ T[:3, :3].T + T[:3, 3]


def support_plane(Y, ok, prior=None, rng=None):
    """height c of the gravity-aligned support plane and its noise σ (robust)"""
    rng = rng or np.random.default_rng(0)
    ys = Y[ok]
    if prior is not None:
        near = np.abs(ys - prior) < PRIOR_WINDOW_M
        if near.sum() > 200:
            ys = ys[near]
    if len(ys) < 200:
        return None, None
    best = None
    for _ in range(200):
        c = ys[rng.integers(len(ys))]
        n = int((np.abs(ys - c) < PLANE_WINDOW_M).sum())
        if best is None or n > best[1]:
            best = (c, n)
    inl = ys[np.abs(ys - best[0]) < PLANE_WINDOW_M]
    c = float(np.median(inl))
    sig = float(max(1.4826 * np.median(np.abs(inl - c)), 5e-4))
    return c, sig


class Segmenter:
    def __init__(self, device=None):
        self._model = None
        self._lock = threading.Lock()
        self.device = device
        self.status = {"available": WEIGHTS.exists(), "ms": 0, "frames": 0, "error": None}

    def _load(self):
        import torch
        from ultralytics import SAM
        self.device = self.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self._model = SAM(str(WEIGHTS))

    def segment(self, bgr, depth, K, T, prior=None):
        import time
        t0 = time.time()
        with self._lock:
            if self._model is None:
                self._load()
            ok = depth > 0.05
            Y = back_project(depth, K, T)[..., 1]
            c, sig = support_plane(Y, ok, prior)
            labels = np.zeros(depth.shape, np.int16)
            if c is None:
                return labels, None, None
            H, W = bgr.shape[:2]
            above = cv2.resize((ok & (Y > c + SIGMA_K * sig)).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            above = cv2.morphologyEx(above, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
            vv, uu = np.mgrid[PROMPT_STRIDE // 2:H:PROMPT_STRIDE, PROMPT_STRIDE // 2:W:PROMPT_STRIDE]
            pts = [[[int(u), int(v)]] for u, v in zip(uu.ravel(), vv.ravel()) if above[v, u]]
            if not pts:
                return labels, c, sig
            res = self._model(bgr, points=pts, labels=[[1]] * len(pts), device=self.device, verbose=False)[0]
            if res.masks is None:
                return labels, c, sig
            ms = res.masks.data.cpu().numpy().astype(bool)
            hgt = cv2.resize((Y - c).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            val = cv2.resize(ok.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            taken = np.zeros((H, W), bool)
            lab_c = np.zeros((H, W), np.int16)
            n = 0
            for i in np.argsort([m.sum() for m in ms]):          # smallest mask wins each pixel
                m = ms[i] & ~taken
                if m.sum() < MIN_MASK_PX:
                    continue
                hv = hgt[m & val]
                if len(hv) < 50 or np.median(hv) <= SIGMA_K * sig:
                    continue                                   # the support itself
                n += 1
                lab_c[m] = n
                taken |= m
            labels = cv2.resize(lab_c, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int16)
        self.status["ms"] = round((time.time() - t0) * 1000)
        self.status["frames"] += 1
        return labels, c, sig
