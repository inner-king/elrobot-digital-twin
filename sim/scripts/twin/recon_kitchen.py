"""가상 주방(elrobot-digital-twin 의 virtual_kitchen.py LAYOUT, 메쉬는 data/kitchen)을 그 저장소 가상 카메라로 찍고,
그 저장소 복원 코드(camera/recon.py, 손대지 않음)로 다시 만든 장면만 꺼낸다. 원래 메쉬는 채점·이름 붙이기에만 쓴다.

python scripts/twin/recon_kitchen.py [--orbit full|sweep]
입력: third_party/elrobot-digital-twin(그 저장소), data/kitchen/{ycb,objaverse}
결과: data/kitchen_recon/<orbit>/objects/obj_<id>.obj(z-up, 바닥 z=0, 꼭짓점 색), background.obj, gt/<이름>.obj, info.json
좌표: 그쪽 ARKit 세계 (x, y, z) → 우리 (x, −z, y − FLOOR_Y). 로봇 원점·+x 방향은 virtual.py 와 같다.
배치 변경 두 가지: 대체 도마는 두께 20 mm(cutting_board_20mm.glb)·긴 변을 칼·당근 방향으로(yaw 90),
칼은 당근 끝과 겹치지 않게 x 0.205 → 0.17(맞닿은 물체는 그쪽 설계상 YOLOE 인식이 나누는데, 인식 없이 시험하므로).
"""
import argparse
import json
import sys
import threading
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=str(ROOT / "third_party/elrobot-digital-twin"))
ap.add_argument("--out", default=str(ROOT / "data/kitchen_recon"))
ap.add_argument("--orbit", default="full", choices=["full", "sweep"])
ap.add_argument("--secs", type=float, default=25.0)
args = ap.parse_args()
repo = Path(args.repo)
out = Path(args.out) / args.orbit
(out / "objects").mkdir(parents=True, exist_ok=True)
(out / "gt").mkdir(exist_ok=True)
sys.path[:0] = [str(repo), str(repo / "camera")]
import cv2
import trimesh

import recon as R


class _NoThread:
    """배경 재구성 루프(_run)는 띄우지 않고 직접 _rebuild 를 부른다. 그 밖의 작업(바닥 정사영)은 그 자리에서 실행."""

    def __init__(self, target=None, args=(), kwargs=None, **k):
        self.t, self.a, self.k = target, args, kwargs or {}

    def start(self):
        if self.t is not None and getattr(self.t, "__name__", "") != "_run":
            self.t(*self.a, **self.k)

    def is_alive(self):
        return False


R.threading = types.SimpleNamespace(Thread=_NoThread, Lock=threading.Lock)
import virtual as VV
import virtual_kitchen as VK

K = ROOT / "data/kitchen"
VK.LAYOUT = [(n, K / Path(p).relative_to(VK.A), zu, t, y, x, z, on) for n, p, zu, t, y, x, z, on in VK.LAYOUT]
VK.LAYOUT = [(n, K / "objaverse/cutting_board_20mm.glb", zu, t, 90, x, z, on) if n == "도마" else
             (n, p, zu, t, y, 0.17, z, on) if n == "칼" else (n, p, zu, t, y, x, z, on)
             for n, p, zu, t, y, x, z, on in VK.LAYOUT]
VK.CACHE = Path(args.out) / "kitchen_cache.npz"
cam = VK.KitchenCamera()


def pose_full(t):
    """VirtualCamera._pose 와 같은 높이·거리로 360° 한 바퀴"""
    a = 2 * np.pi * t / args.secs
    look = np.array([0.22, VV.FLOOR_Y + 0.03, 0.0])
    pos = look + np.array([0.55 * np.cos(a), 0.45, 0.55 * np.sin(a)])
    f = look - pos; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, np.cross(r, f), -f, pos
    return T


rec = R.Reconstructor()
rec.status["running"] = True
rec.floor_y = VV.FLOOR_Y
CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])
W_D, H_D, W_C, H_C, FX_C = VV.W_D, VV.H_D, VV.W_C, VV.H_C, VV.FX_C
s = W_D / W_C
next_reb = R.REBUILD_EVERY_S
for k, t in enumerate(np.arange(0, args.secs, 1 / R.MAX_SUBMIT_HZ)):     # stream.py 가 recon 에 넘기는 것과 같게
    T = cam._pose(t) if args.orbit == "sweep" else pose_full(t)
    depth, _ = cam._render(T, W_D, H_D, FX_C * s)
    depth += np.random.default_rng(k).normal(0, 0.002, depth.shape).astype(np.float32) * (depth > 0)
    _, colour = cam._render(T, W_C // 2, H_C // 2, FX_C / 2)
    colour = cv2.resize(colour, (W_C, H_C), interpolation=cv2.INTER_NEAREST)
    dclean = np.where(np.isfinite(depth) & (depth > 0.05) & (depth < R.DEPTH_MAX_M), depth, 0).astype(np.float32)
    small = cv2.cvtColor(cv2.resize(colour, (W_D, H_D), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    rec._last_submit = 0.0
    rec.submit(dclean, small, FX_C * s, FX_C * s, W_C / 2 * s, H_C / 2 * s, T @ CV_TO_ARKIT)
    if t >= next_reb:
        rec._rebuild(list(rec._kf)); next_reb += R.REBUILD_EVERY_S
for _ in range(R.REG_UNLABELLED):                    # 라벨 없는 물체가 확정될 때까지
    rec._rebuild(list(rec._kf))

Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
to_sim = lambda V: np.asarray(V, float) @ Y2Z.T - [0, 0, VV.FLOOR_Y]      # ARKit → 우리 z-up, 바닥 z=0


def mesh(V, T, C=None):
    m = trimesh.Trimesh(to_sim(V), np.asarray(T, np.int64), process=False)
    if C is not None and len(C):
        m.visual.vertex_colors = np.c_[np.asarray(C, np.uint8), np.full(len(C), 255, np.uint8)]
    return m


objs = []
b = rec._objects
if b and len(b) >= 4:
    off = 4
    for _ in range(int(np.frombuffer(b[:4], np.uint32)[0])):
        oid = int(np.frombuffer(b[off:off + 4], np.uint32)[0])
        Vr, Tr, Cr, off = R._blob_arrays(b, off + 4)
        Vc, Tc, Cc, off = R._blob_arrays(b, off)
        objs.append((oid, mesh(Vc, Tc, Cc)))
bg = rec._mesh
if bg:
    nv, nt = (int(x) for x in np.frombuffer(bg[:8], np.uint32))
    Vb = np.frombuffer(bg[8:8 + nv * 12], np.float32).reshape(-1, 3)
    Tb = np.frombuffer(bg[8 + nv * 24:8 + nv * 24 + nt * 12], np.uint32).reshape(-1, 3)
    Cb = np.frombuffer(bg[8 + nv * 24 + nt * 12:8 + nv * 24 + nt * 12 + nv * 3], np.uint8).reshape(-1, 3)
    mesh(Vb, Tb, Cb).export(out / "background.obj")
info = {o["id"]: o for o in rec.status.get("objects") or []}

gts = {}
for i, (name, *_r) in enumerate(VK.LAYOUT):        # 정답(채점·이름 붙이기에만)
    g = trimesh.Trimesh(to_sim(cam._place(i)), cam._obj[i]["F"], process=False)
    g.export(out / "gt" / f"{name}.obj")
    gts[name] = g
from scipy.spatial import cKDTree

samp = lambda m, n=8000: trimesh.sample.sample_surface(m, n, seed=0)[0]
gs_pts = {n: samp(g) for n, g in gts.items()}
gs_kd = {n: cKDTree(p) for n, p in gs_pts.items()}
rep = {"orbit": args.orbit, "keyframes": len(rec._kf), "n_objects": len(objs), "error": rec.status.get("error"), "objects": []}
all_rec = []
for oid, m in objs:
    m.export(out / "objects" / f"obj_{oid}.obj")
    p = samp(m)
    all_rec.append(p)
    d = {n: float(np.mean(kd.query(p)[0])) for n, kd in gs_kd.items()}
    best = min(d, key=d.get)
    g = gts[best]
    dg = cKDTree(p).query(gs_pts[best])[0]
    it = info.get(oid, {})
    rep["objects"].append({"id": oid, "match": best, "mean_dist_mm": round(d[best] * 1000, 1),
                           "rec_extents_mm": (m.extents * 1000).round(0).tolist(), "gt_extents_mm": (g.extents * 1000).round(0).tolist(),
                           "gt_covered_5mm": round(float(np.mean(dg < 0.005)), 2), "shape": it.get("shape"),
                           "support_mm": it.get("support_mm"), "coverage_deg": it.get("coverage_deg")})
kd_all = cKDTree(np.concatenate(all_rec)) if all_rec else None
rep["gt_coverage_5mm"] = {n: round(float(np.mean(kd_all.query(p)[0] < 0.005)), 2) for n, p in gs_pts.items()} if kd_all else {}
(out / "info.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
print(json.dumps({k: v for k, v in rep.items() if k != "objects"}, ensure_ascii=False, default=str))
for o in rep["objects"]:
    print(o["id"], o["match"], o["mean_dist_mm"], "mm", o["shape"], "rec", o["rec_extents_mm"], "gt", o["gt_extents_mm"],
          "cov", o["gt_covered_5mm"], "support_mm", o["support_mm"])
