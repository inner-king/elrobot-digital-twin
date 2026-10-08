"""Chop & Learn 사진에서 식재료 3D 형상과 껍질·과육 색을 만들어 복원 폴더 규약(data/recon/<id>/)으로 저장한다.

Chop & Learn 에는 3D·깊이·카메라 보정이 없다. 녹색 도마 위에서 위(cam1)에서 찍은 통째 사진의 실루엣으로
형상을 만들고, 실제 크기는 일반적인 치수로 가정한다(meta.yaml 에 가정을 적고 scale_checked=false).
  오이: 실루엣 중심선을 따라 폭을 지름으로 하는 관(휜 모양 유지). 바닥이 도마에 닿게 축 높이 = 반지름.
  사과: 실루엣에 맞춘 타원(위에서 본 지름) × 높이/지름 비(가정)의 회전 타원체.
  감자: 실루엣 윤곽을 그대로 쓰고(울퉁불퉁한 외곽 유지) 윤곽 안쪽 거리로 위아래를 부풀린다. 높이/폭 비는 가정.
색: 통째 사진 실루엣 안쪽의 중앙값 = 껍질, 썬 사진(오이·감자 rs, 사과 hrs)의 조각 안쪽(테두리 깎음) 중앙값 = 과육.

python scripts/05_food_from_photos.py --object cucumber --size_m 0.20
python scripts/05_food_from_photos.py --object apple --size_m 0.075 --height_ratio 0.85
python scripts/05_food_from_photos.py --object potato --size_m 0.10 --height_ratio 0.75
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh
import yaml
from PIL import Image
from scipy import ndimage

from cutsim.assets.volumize import voxel_remesh
from cutsim.plotting import plt

ap = argparse.ArgumentParser()
ap.add_argument("--object", required=True, choices=["cucumber", "apple", "potato"])
ap.add_argument("--size_m", type=float, required=True, help="오이·감자: 끝-끝 길이, 사과: 위에서 본 평균 지름")
ap.add_argument("--height_ratio", type=float, default=0.85, help="사과: 높이/지름, 감자: 높이/폭(가정)")
ap.add_argument("--participant", default="p1")
ap.add_argument("--images", default="data/chopnlearn/images")
ap.add_argument("--pitch", type=float, default=0.0007)
args = ap.parse_args()
IMG = Path(args.images)
oid = f"cnl_{args.object}"
rep_dir = Path("reports/05_photos") / args.object
rep_dir.mkdir(parents=True, exist_ok=True)
out_dir = Path("data/recon") / oid
out_dir.mkdir(parents=True, exist_ok=True)
CUT_STATE = {"cucumber": "rs", "apple": "hrs", "potato": "rs"}[args.object]


def rgb2lab(rgb):
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def food_mask(img, thresh=28.0, min_frac=0.002):
    """가장자리 띠의 중앙값 색을 도마 색으로 보고, Lab 색차가 큰 픽셀을 식재료로 본다(조각 여러 개 허용)."""
    lab = rgb2lab(img)
    border = np.concatenate([lab[:12].reshape(-1, 3), lab[-12:].reshape(-1, 3), lab[:, :12].reshape(-1, 3),
                             lab[:, -12:].reshape(-1, 3)])
    board = np.median(border, 0)
    m = np.linalg.norm(lab - board, axis=-1) > thresh
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.binary_closing(m, iterations=3)
    m = ndimage.binary_fill_holes(m)
    lab_cc, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab_cc, range(1, n + 1))
    keep = np.isin(lab_cc, 1 + np.flatnonzero(sizes >= min_frac * m.size))
    return keep


def median_color(img, mask, erode=0):
    if erode:
        mask = ndimage.binary_erosion(mask, iterations=erode)
    return (np.median(img[mask], 0) / 255.0).round(3).tolist()


def load(name):
    return np.asarray(Image.open(IMG / f"{name}.png").convert("RGB"))


# ---------------- 색 ----------------
skins, fleshes = [], []
for p in ("p1", "p2", "p3"):
    for cam in ("cam1",):
        n_w, n_c = f"{args.object}_w_{cam}_{p}", f"{args.object}_{CUT_STATE}_{cam}_{p}"
        if (IMG / f"{n_w}.png").exists():
            im = load(n_w)
            mk = food_mask(im)
            lab_cc, n = ndimage.label(mk)
            if n:  # 통째 사진은 가장 큰 덩어리 하나
                big = lab_cc == (1 + np.argmax(ndimage.sum(mk, lab_cc, range(1, n + 1))))
                skins.append(median_color(im, big, erode=3))
        if (IMG / f"{n_c}.png").exists():
            im = load(n_c)
            fleshes.append(median_color(im, food_mask(im), erode=5))  # 조각 테두리(껍질) 깎고 과육만
skin = np.median(np.array(skins), 0).round(3).tolist()
flesh = np.median(np.array(fleshes), 0).round(3).tolist()

# ---------------- 형상 ----------------
img = load(f"{args.object}_w_cam1_{args.participant}")
mask = food_mask(img)
lab_cc, n = ndimage.label(mask)
mask = lab_cc == (1 + np.argmax(ndimage.sum(mask, lab_cc, range(1, n + 1))))
ys, xs = np.nonzero(mask)
P = np.stack([xs, -ys], 1).astype(float)  # 이미지 y 는 아래로 → 위로
c0 = P.mean(0)
U, S, Vt = np.linalg.svd(P - c0, full_matrices=False)
ax_main, ax_perp = Vt[0], Vt[1]
s = (P - c0) @ ax_main
w = (P - c0) @ ax_perp

if args.object == "cucumber":
    nb = 120
    edges = np.linspace(s.min(), s.max(), nb + 1)
    centers, radii, ss = [], [], []
    for i in range(nb):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if m.sum() < 3:
            continue
        lo, hi = w[m].min(), w[m].max()
        ss.append((edges[i] + edges[i + 1]) / 2)
        centers.append((lo + hi) / 2)
        radii.append((hi - lo) / 2)
    ss, centers, radii = map(np.array, (ss, centers, radii))
    k = 5
    ker = np.ones(k) / k
    centers = np.convolve(np.pad(centers, k // 2, mode="edge"), ker, "valid")
    radii = np.convolve(np.pad(radii, k // 2, mode="edge"), ker, "valid")
    scale = args.size_m / (ss.max() - ss.min())  # m / pixel
    X = (ss - ss.mean()) * scale
    Y = (centers - centers.mean()) * scale
    Rr = np.maximum(radii * scale, 0.002)
    # 관: 중심선 (X, Y) 를 따라 링, 축 높이 = 반지름(바닥이 도마에 닿음)
    nseg = 40
    th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    T = np.gradient(np.stack([X, Y], 1), axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    B1 = np.stack([-T[:, 1], T[:, 0], np.zeros(len(T))], 1)
    verts = []
    for i in range(len(X)):
        cen = np.array([X[i], Y[i], Rr[i]])
        ring = cen[None] + Rr[i] * (np.cos(th)[:, None] * B1[i][None] + np.sin(th)[:, None] * np.array([0, 0, 1.0])[None])
        verts.append(ring)
    verts = np.concatenate(verts)
    faces = []
    nr = len(X)
    for i in range(nr - 1):
        for j in range(nseg):
            a, b = i * nseg + j, i * nseg + (j + 1) % nseg
            c, d = a + nseg, b + nseg
            faces += [[a, b, d], [a, d, c]]
    tip0 = len(verts)
    tip1 = tip0 + 1
    verts = np.vstack([verts, [X[0] - 0.5 * Rr[0] * T[0, 0], Y[0] - 0.5 * Rr[0] * T[0, 1], Rr[0]],
                       [X[-1] + 0.5 * Rr[-1] * T[-1, 0], Y[-1] + 0.5 * Rr[-1] * T[-1, 1], Rr[-1]]])
    for j in range(nseg):
        faces += [[tip0, (j + 1) % nseg, j], [tip1, (nr - 1) * nseg + j, (nr - 1) * nseg + (j + 1) % nseg]]
    mesh = trimesh.Trimesh(verts, np.array(faces), process=True)
    profile = {"length_m": float(X.max() - X.min()), "max_diameter_m": float(2 * Rr.max()),
               "mean_diameter_m": float(2 * Rr.mean()), "lateral_bend_m": float(Y.max() - Y.min())}
    shape_note = f"위 시점 실루엣 중심선을 따라 만든 관, 끝-끝 길이 {args.size_m} m 로 가정"
elif args.object == "potato":
    # 윤곽 안쪽 거리 d(가장자리 0)로 두께를 정한다: 반높이 = H/2·sqrt(u(2-u)), u = d/d_max.
    # 가장자리에서는 수직(타원 단면), 가운데로 갈수록 두꺼워지고 끝으로 갈수록 얇아진다. 긴 축 = x.
    from cutsim.assets.volumize import _grid_to_mesh

    scale = args.size_m / (s.max() - s.min())  # m / pixel
    dist = ndimage.gaussian_filter(ndimage.distance_transform_edt(mask), 2.0)
    width = (w.max() - w.min()) * scale
    H = args.height_ratio * width
    pitch = args.pitch
    pad = 3
    xs_m = np.arange(s.min() * scale - pad * pitch, s.max() * scale + pad * pitch, pitch)
    ys_m = np.arange(w.min() * scale - pad * pitch, w.max() * scale + pad * pitch, pitch)
    zs_m = np.arange(-pad * pitch, H + pad * pitch, pitch)
    XX, YY = np.meshgrid(xs_m, ys_m, indexing="ij")
    Pp = c0 + (XX[..., None] / scale) * ax_main + (YY[..., None] / scale) * ax_perp  # (x, -y) 픽셀 좌표
    dpx = ndimage.map_coordinates(dist, [-Pp[..., 1], Pp[..., 0]], order=1, cval=0.0)
    u = np.clip(dpx / dist.max(), 0.0, 1.0)
    half = 0.5 * H * np.sqrt(u * (2.0 - u))
    grid = (np.abs(zs_m[None, None, :] - H / 2) <= half[..., None]) & (dpx[..., None] > 0.5)
    mesh = _grid_to_mesh(grid, np.array([xs_m[0], ys_m[0], zs_m[0]]), pitch)
    mesh = max(mesh.split(only_watertight=False), key=lambda m: m.volume)
    profile = {"length_m": float(np.ptp(s) * scale), "width_m": float(width), "height_m": float(H)}
    shape_note = (f"위 시점 실루엣 윤곽 + 안쪽 거리로 부풀린 두께(높이/폭 {args.height_ratio} 가정), "
                  f"끝-끝 길이 {args.size_m} m 로 가정")
else:
    # 위에서 본 윤곽에 타원(2차 모멘트) 맞추기 → 회전 타원체
    a_px, b_px = 2 * np.sqrt(S[0] ** 2 / len(P)), 2 * np.sqrt(S[1] ** 2 / len(P))  # 균일 타원의 반축 = 2·표준편차
    scale = args.size_m / (a_px + b_px)  # 평균 지름 = a+b (픽셀) → size_m
    a, b = a_px * scale, b_px * scale
    c = args.height_ratio * (a + b) / 2
    mesh = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    mesh.apply_scale([a, b, c])
    profile = {"diameter_major_m": float(2 * a), "diameter_minor_m": float(2 * b), "height_m": float(2 * c)}
    shape_note = (f"위 시점 실루엣에 맞춘 타원 × 높이/지름 {args.height_ratio}(가정), 평균 지름 {args.size_m} m 로 가정")

mesh = voxel_remesh(mesh, pitch=args.pitch)
lo, hi = mesh.bounds
mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
mesh.export(out_dir / "mesh.obj")
meta = {"object_id": oid, "source": "chopnlearn", "category": args.object, "units": "m", "up_axis": "z",
        "frame": "object", "T_world_object": None, "recon_method": "silhouette (top view)", "scale_checked": False,
        "shape_note": shape_note, "color_skin": skin, "color_flesh": flesh,
        "photos": {"shape": f"{args.object}_w_cam1_{args.participant}.png",
                   "flesh": f"{args.object}_{CUT_STATE}_cam1_p*.png"}}
(out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
info = {"profile": profile, "extents_m": mesh.extents.round(4).tolist(), "volume_cm3": round(mesh.volume * 1e6, 1),
        "watertight": bool(mesh.is_watertight), "skin_color": skin, "flesh_color": flesh,
        "n_skin_photos": len(skins), "n_flesh_photos": len(fleshes)}
(rep_dir / "shape.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
print(json.dumps(info, ensure_ascii=False))

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
axs[0].imshow(img); axs[0].contour(mask, levels=[0.5], colors="r", linewidths=1)
axs[0].set_title(f"실루엣 ({args.object}_w_cam1_{args.participant})"); axs[0].axis("off")
cut_img = load(f"{args.object}_{CUT_STATE}_cam1_p1")
axs[1].imshow(cut_img); axs[1].contour(food_mask(cut_img), levels=[0.5], colors="y", linewidths=0.8)
axs[1].set_title(f"과육 색 기준 ({CUT_STATE})"); axs[1].axis("off")
ax3 = fig.add_subplot(1, 3, 3, projection="3d"); axs[2].axis("off")
v = mesh.vertices
ax3.plot_trisurf(v[:, 0], v[:, 1], mesh.faces, v[:, 2], color=skin, lw=0, alpha=0.9)
ax3.set_box_aspect(np.array(mesh.extents, float)); ax3.set_title(f"메쉬 {np.round(mesh.extents * 100, 1)} cm")
for i, (nm, col) in enumerate((("껍질", skin), ("과육", flesh))):
    axs[1].add_patch(plt.Rectangle((10 + 70 * i, 10), 60, 40, color=col, transform=axs[1].transData))
    axs[1].text(12 + 70 * i, 62, nm, color="w", fontsize=8)
fig.tight_layout(); fig.savefig(rep_dir / "shape.png", dpi=100)
