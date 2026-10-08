"""Object models: generated once per object, then tracked in 6-DoF (plan A structure, Mac version).

  ① generate  [external]  best keyframe RGB + instance mask → TripoSR (MIT) → full mesh, arbitrary scale / axes
                           (GPU later: TRELLIS)
  ② align     [ours, Any6D-style, simplified]  mesh + the LiDAR points seen of the object + the support height →
                           similarity transform to the world: up axis (6 candidates) · scale from the height
                           (observed top − support) · bottom on the support · 24 yaw seeds × ICP (observed → model
                           surface, partial-to-complete) · final ICP with scaling (bounded)
  ③ track     [ours]      rigid ICP of the model to the current observed points (GPU later: FoundationPose)
Inputs / outputs are numpy arrays in the ARKit world frame (m, y up).
"""
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_TSR = None


def _tsr(device):
    global _TSR
    if _TSR is None:
        import torch
        from skimage.measure import marching_cubes as _mc

        def marching_cubes(level, thr):                   # torchmcubes stand-in (no CUDA build on the Mac)
            v, f, _, _ = _mc(level.detach().cpu().numpy(), thr)
            return torch.from_numpy(v.astype(np.float32)), torch.from_numpy(f.astype(np.int64))
        sys.modules.setdefault("torchmcubes", types.SimpleNamespace(marching_cubes=marching_cubes))
        sys.modules.setdefault("rembg", types.SimpleNamespace(remove=None, new_session=None))   # masks come from YOLOE
        sys.path.insert(0, str(ROOT / "third_party" / "TripoSR"))
        from tsr.system import TSR
        m = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
        m.renderer.set_chunk_size(8192)
        _TSR = m.to(device)
    return _TSR


def generate(rgb, mask, device="mps", resolution=192):
    """rgb H×W×3 uint8 (RGB), mask H×W bool → trimesh with vertex colours (TripoSR frame)."""
    import torch
    from PIL import Image
    m = _tsr(device)                                       # (also installs the stand-in modules tsr imports)
    from tsr.utils import resize_foreground
    rgba = np.dstack([rgb, (mask * 255).astype(np.uint8)])
    img = resize_foreground(Image.fromarray(rgba), 0.85)
    a = np.asarray(img).astype(np.float32) / 255
    img = Image.fromarray(((a[..., :3] * a[..., 3:4] + (1 - a[..., 3:4]) * 0.5) * 255).astype(np.uint8))
    with torch.no_grad():
        codes = m([img], device=device)
        return m.extract_mesh(codes, True, resolution=resolution)[0]


def _icp(src, tgt_pcd, init, dist, scaling=False, iters=40):
    import open3d as o3d
    s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=scaling)
    r = o3d.pipelines.registration.registration_icp(s, tgt_pcd, dist, init, est,
                                                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
    return r.transformation, r.fitness, r.inlier_rmse


def align(model_v, obs, sup, n_yaw=24, top_k=1):
    """Similarity M (4×4) taking model vertices to the world so that the observed points lie on the model surface."""
    import open3d as o3d
    t0 = time.time()
    obs = np.asarray(obs, float)
    top = float(np.percentile(obs[:, 1], 99))
    h_obs = top - sup
    best, hyps = None, []
    axes = np.eye(3)
    for ax in range(3):
        for sg in (1, -1):
            up = axes[ax] * sg
            # rotation taking `up` to +y
            v = np.cross(up, [0, 1, 0]); c = float(up @ [0, 1, 0])
            if np.linalg.norm(v) < 1e-9:
                R0 = np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R0 = np.eye(3) + vx + vx @ vx / (1 + c)
            V = model_v @ R0.T
            s = h_obs / max(np.ptp(V[:, 1]), 1e-6)
            V = V * s
            V = V - [(V[:, 0].min() + V[:, 0].max()) / 2, V[:, 1].min() - sup, (V[:, 2].min() + V[:, 2].max()) / 2]
            cxz = obs[:, [0, 2]].mean(axis=0)
            for k in range(n_yaw):
                a = 2 * np.pi * k / n_yaw
                Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
                Vk = (V - [0, sup, 0]) @ Ry.T + [cxz[0], sup, cxz[1]]
                tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Vk))
                Tk, fit, rmse = _icp(obs, tgt, np.eye(4), 0.02, iters=25)
                score = rmse + (1 - fit) * 0.02
                hyps.append((score, ax, sg, a, R0, s, Tk, fit, rmse, Vk))
                if best is None or score < best[0]:
                    best = hyps[-1]
    if top_k > 1:                                          # several hypotheses for free-space scoring (align_best)
        out = []
        for h in sorted(hyps, key=lambda h: h[0])[:top_k]:
            Vw = (np.c_[h[9], np.ones(len(h[9]))] @ np.linalg.inv(h[6]).T)[:, :3]
            out.append(_similarity(model_v, Vw))
        return out
    score, ax, sg, a, R0, s, Tk, fit, rmse, Vk = best
    # final: scaled ICP (observed → model), then express everything as model → world
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Vk))
    T2, fit2, rmse2 = _icp(obs, tgt, Tk, 0.01, scaling=True, iters=60)
    sc = np.cbrt(abs(np.linalg.det(T2[:3, :3])))
    if not 0.85 < sc < 1.15:                               # partial views cannot fix the scale: keep the height's
        T2, fit2, rmse2 = _icp(obs, tgt, Tk, 0.01, scaling=False, iters=60)
    Vw = (np.c_[Vk, np.ones(len(Vk))] @ np.linalg.inv(T2).T)[:, :3]
    # rebuild the full model → world similarity from the vertex correspondence (model_v ↔ Vw)
    from numpy.linalg import svd
    mu_a, mu_b = model_v.mean(0), Vw.mean(0)
    Aa, Bb = model_v - mu_a, Vw - mu_b
    U, S, Wt = svd(Bb.T @ Aa)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Wt))])
    R = U @ D @ Wt
    scale = (S * np.diag(D)).sum() / (Aa ** 2).sum()
    M = np.eye(4); M[:3, :3] = scale * R; M[:3, 3] = mu_b - scale * R @ mu_a
    return M, {"up_axis": f"{'+' if sg > 0 else '-'}{'xyz'[ax]}", "yaw_deg": round(np.degrees(a)), "scale": round(float(scale), 4),
               "fitness": round(fit2, 3), "rmse_mm": round(rmse2 * 1000, 2), "ms": round((time.time() - t0) * 1000)}


def _similarity(A, B):
    """Umeyama similarity (4×4) mapping point set A onto the corresponding point set B."""
    mu_a, mu_b = A.mean(0), B.mean(0)
    Aa, Bb = A - mu_a, B - mu_b
    U, S, Wt = np.linalg.svd(Bb.T @ Aa)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Wt))])
    R = U @ D @ Wt
    sc = (S * np.diag(D)).sum() / (Aa ** 2).sum()
    M = np.eye(4); M[:3, :3] = sc * R; M[:3, 3] = mu_b - sc * R @ mu_a
    return M


def free_violation(P, kfs, tau=0.01):
    """Share of the model's surface samples that lie in space a keyframe saw as empty (in front of its measured depth
    by more than tau): a too large or wrongly turned model cuts into the observed free space."""
    viol = np.zeros(len(P), bool)
    seen = np.zeros(len(P), bool)
    for k in kfs:
        Tcw = np.linalg.inv(k["T"])
        pc = P @ Tcw[:3, :3].T + Tcw[:3, 3]
        z = pc[:, 2]
        K = k["K"]
        ok = z > 0.05
        u = np.round(K[0, 0] * pc[:, 0] / np.where(ok, z, 1) + K[0, 2]).astype(np.int64)
        w = np.round(K[1, 1] * pc[:, 1] / np.where(ok, z, 1) + K[1, 2]).astype(np.int64)
        dep = k["depth"]
        h, wd = dep.shape
        ok &= (u >= 0) & (u < wd) & (w >= 0) & (w < h)
        # 3×3 minimum depth: at 256×192 an object is ~25 px wide and its silhouette samples round onto background
        # pixels (the true mesh scored 50 % "violation" against the single pixel — measured)
        import cv2
        dmin = cv2.erode(np.where(dep > 0, dep, 1e3).astype(np.float32), np.ones((3, 3), np.uint8))
        d = np.zeros(len(P))
        d[ok] = dmin[w[ok], u[ok]]
        ok &= (d > 0.05) & (d < 1e2)
        seen |= ok
        viol |= ok & (z < d - tau)
    return float(viol.sum() / max(seen.sum(), 1))


def align_best(model_v, obs, sup, T_wc, kfs, scales=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4), lam=0.03):
    """[ours, render-and-compare hypothesis scoring as in Any6D / FoundationPose, simplified] hypotheses = the view-
    convention alignment + the best axis/yaw-search alignments, each at several scales about its own centre, refined by
    rigid ICP; score = ICP rmse + lam · free-space violation share. Returns (M, stats)."""
    import open3d as o3d
    import trimesh
    t0 = time.time()
    hyps = [align_view(model_v, obs, T_wc)[0]] + align(model_v, obs, sup, top_k=3)
    best = None
    for H in hyps:
        for f in scales:
            Vw = (np.c_[model_v, np.ones(len(model_v))] @ H.T)[:, :3]
            c = Vw.mean(0)
            Vw = c + (Vw - c) * f
            Vw[:, 1] += sup - Vw[:, 1].min() if Vw[:, 1].min() < sup - 0.01 else 0.0
            tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Vw))
            T1, fit, rmse = _icp(obs, tgt, np.eye(4), 0.01, iters=30)
            Vf = (np.c_[Vw, np.ones(len(Vw))] @ np.linalg.inv(T1).T)[:, :3]
            sample = Vf[np.random.default_rng(0).choice(len(Vf), min(len(Vf), 3000), replace=False)]
            fv = free_violation(sample, kfs)
            score = rmse + (1 - fit) * 0.02 + lam * fv
            if best is None or score < best[0]:
                best = (score, Vf, fit, rmse, fv, f)
    score, Vf, fit, rmse, fv, f = best
    M = _similarity(model_v, Vf)
    return M, {"scale": round(float(np.cbrt(abs(np.linalg.det(M[:3, :3])))), 4), "fitness": round(fit, 3),
               "rmse_mm": round(rmse * 1000, 2), "free_violation": round(fv, 3), "scale_factor": f,
               "hypotheses": len(hyps) * len(scales), "ms": round((time.time() - t0) * 1000)}


def align_view(model_v, obs, T_wc):
    """[ours] Alignment from TripoSR's view convention instead of a search: it reconstructs the input image as seen from
    +x with z up (render camera azimuth 0, elevation 0), so model +x → towards the camera, +y → image right, +z → image
    up of the keyframe that was fed in. Orientation then comes from that keyframe's pose (ARKit camera: looks along −z,
    y up); scale from the image-plane extents of the observed points; then ICP with scaling (bounded) refines.
    T_wc: camera → world (ARKit axes)."""
    import open3d as o3d
    t0 = time.time()
    obs = np.asarray(obs, float)
    R_cm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)    # columns: model x→cam +z, y→cam +x, z→cam +y
    R = T_wc[:3, :3] @ R_cm
    Vr = model_v @ R.T
    xc, yc, zc = T_wc[:3, 0], T_wc[:3, 1], T_wc[:3, 2]
    s = np.mean([np.ptp(obs @ xc) / max(np.ptp(Vr @ xc), 1e-6), np.ptp(obs @ yc) / max(np.ptp(Vr @ yc), 1e-6)])
    Vs = Vr * s
    # translation: image-plane centres coincide; along the view, the model's nearest layer meets the observed surface
    d_obs, d_m = obs @ zc, Vs @ zc                       # +z_cam points back towards the camera
    t = (obs @ xc).mean() * xc + (obs @ yc).mean() * yc - ((Vs @ xc).mean() * xc + (Vs @ yc).mean() * yc)
    t += (np.percentile(d_obs, 90) - np.percentile(d_m, 98)) * zc
    Vw = Vs + t
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Vw))
    T1, f1, r1 = _icp(obs, tgt, np.eye(4), 0.02, iters=40)
    T2, f2, r2 = _icp(obs, tgt, T1, 0.01, scaling=True, iters=60)
    sc = np.cbrt(abs(np.linalg.det(T2[:3, :3])))
    if not 0.8 < sc < 1.25:
        T2, f2, r2 = T1, f1, r1
    Ti = np.linalg.inv(T2)
    M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = t
    M = Ti @ M
    return M, {"scale": round(float(np.cbrt(abs(np.linalg.det(M[:3, :3])))), 4), "fitness": round(f2, 3),
               "rmse_mm": round(r2 * 1000, 2), "ms": round((time.time() - t0) * 1000)}


def track(model_world_v, obs, dist=0.015):
    """Rigid ICP of the placed model to new observations → 4×4 world correction (apply to the model)."""
    import open3d as o3d
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_world_v))
    T, fit, rmse = _icp(np.asarray(obs, float), tgt, np.eye(4), dist, iters=30)
    return np.linalg.inv(T), fit, rmse
