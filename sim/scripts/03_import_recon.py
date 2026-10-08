"""팀원 복원 결과 폴더를 읽어 검증·정규화하고, 껍질·과육 MPM 물체용 메쉬와 리포트를 만든다.

python scripts/03_import_recon.py data/recon/<object_id> [--ref_mesh <TACO 스캔 모델 _cm.obj>] [--cut]
결과: reports/03_import/<object_id>/{normalized.obj, parts/{skin,flesh}.obj, traj_sim.npz, report.json, report.png}
--ref_mesh : TACO 스캔 모델(cm)을 정답으로 삼아 Chamfer 거리·크기 비율 채점(물체 좌표계에서 비교)
--cut      : 정규화된 메쉬를 두 층(껍질·과육)으로 잘라 본다(GPU). 칼 궤적이 있으면 그 궤적을 재생(TR)하고,
             없으면 가운데를 수직으로 누른다(T1). meta.yaml 에 color_skin/color_flesh 가 있으면 그 색을 쓴다.
--cut_args : 01_cut_primitive.py 에 덧붙일 인자(따옴표로 묶어서), 예: "--cut_force --z_force_budget 12"
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh
import yaml

from cutsim.assets.volumize import export_parts, voxel_remesh
from cutsim.io.recon_loader import UNIT, bottom_center, chamfer_and_scale, find_mesh, load_knife_traj, load_recon

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--materials", default="configs/materials.yaml")
ap.add_argument("--ref_mesh", default=None)
ap.add_argument("--remesh_pitch", type=float, default=0.001)
ap.add_argument("--sim_dt", type=float, default=1e-3)
ap.add_argument("--cut", action="store_true")
ap.add_argument("--cut_args", default="")
ap.add_argument("--grid_density", type=float, default=256)
ap.add_argument("--out_root", default="reports/03_import")
args = ap.parse_args()

folder = Path(args.folder)
meta, mesh, warnings, T_norm = load_recon(folder)
out = Path(args.out_root) / meta["object_id"]
out.mkdir(parents=True, exist_ok=True)
raw = trimesh.load(find_mesh(folder), force="mesh")
rep = {"object_id": meta["object_id"], "meta": meta, "raw": {"n_vertices": len(raw.vertices),
       "n_components": len(raw.split(only_watertight=False)), "extents_raw_units": raw.extents.round(5).tolist(),
       "watertight": bool(raw.is_watertight)}}

# 1) 수밀화: 구멍 메우기로 안 되면 복셀 재구성
remeshed = False
if not mesh.is_watertight:
    mesh = voxel_remesh(mesh, pitch=args.remesh_pitch)
    C = bottom_center(mesh)  # 재구성으로 바닥이 조금 바뀌므로 다시 맞추고 궤적 변환에도 반영
    mesh.apply_transform(C)
    T_norm = C @ T_norm
    remeshed = True
    warnings.append(f"복셀 재구성(pitch {args.remesh_pitch * 1000:.1f}mm) 후 수밀={mesh.is_watertight}")
mesh.export(out / "normalized.obj")
rep["normalized"] = {"extents_m": mesh.extents.round(4).tolist(), "volume_cm3": round(float(mesh.volume) * 1e6, 2),
                     "watertight": bool(mesh.is_watertight), "remeshed": remeshed}

# 2) 껍질·과육 분리
from cutsim.yamlio import load as load_yaml

materials = load_yaml(args.materials)
cat = meta["category"]
if cat not in materials:
    warnings.append(f"category '{cat}' 가 materials.yaml 에 없음 → default 사용")
mat = materials.get(cat, materials["default"])
p_size = 0.01 * 64.0 / args.grid_density
skin_t = max(mat["skin_thickness_m"], p_size)
if skin_t > mat["skin_thickness_m"]:
    warnings.append(f"껍질 {mat['skin_thickness_m'] * 1000:.1f}mm 는 입자({p_size * 1000:.1f}mm)보다 얇아 "
                    f"{skin_t * 1000:.1f}mm 로 올림")
try:
    paths, vols = export_parts(mesh, skin_t, out / "parts", pitch=min(skin_t / 2, 0.0005))
    rep["parts"] = {k: str(v) for k, v in paths.items()}
    rep["parts_volume_cm3"] = {k: round(v * 1e6, 2) for k, v in vols.items()}
    rep["parts_volume_check"] = round((vols["skin_volume"] + vols["flesh_volume"]) / vols["outer_volume"], 4)
except Exception as e:  # 너무 작거나 얇은 물체
    warnings.append(f"껍질·과육 분리 실패: {e!r}")

# 3) 칼 궤적(선택): 물체 좌표계 + 시뮬레이션 시간 간격으로 보간
T = np.asarray(meta["T_world_object"]) if meta["frame"] == "world" and meta.get("T_world_object") else None
traj = load_knife_traj(folder, T_norm)
if traj is not None:
    from scipy.spatial.transform import Rotation, Slerp

    Tk = traj["T_norm_knife"]
    t_sim = np.arange(traj["t"][0], traj["t"][-1], args.sim_dt)
    pos = np.stack([np.interp(t_sim, traj["t"], Tk[:, i, 3]) for i in range(3)], 1)
    rot = Slerp(traj["t"], Rotation.from_matrix(Tk[:, :3, :3]))(t_sim).as_matrix()
    np.savez(out / "traj_sim.npz", t=t_sim, pos=pos, rot=rot)
    rep["traj"] = {"n_src": len(traj["t"]), "fps": traj["fps"], "n_sim": len(t_sim), "duration_s": float(t_sim[-1]),
                   "start_pos_m": pos[0].round(4).tolist(), "end_pos_m": pos[-1].round(4).tolist()}

# 4) TACO 스캔 모델로 채점(물체 좌표계: 단위·world→object 만 적용, 바닥 중심 이동 전)
if args.ref_mesh:
    ref = trimesh.load(args.ref_mesh, force="mesh")
    ref.apply_scale(0.01 if "_cm" in Path(args.ref_mesh).stem else 1.0)
    rec = raw.copy()
    if T is not None:
        rec.apply_transform(np.linalg.inv(T))
    rec.apply_scale(UNIT[meta["units"]])
    rec = max(rec.split(only_watertight=False), key=lambda m: m.area)
    score = chamfer_and_scale(rec, ref)
    rep["taco_score"] = score
    if not 0.8 < score["scale_ratio"] < 1.25:
        warnings.append(f"크기 비율 {score['scale_ratio']:.3f}: 단위 오류(0.1·0.01·10배) 의심")

rep["warnings"] = warnings

# 5) 그림: 원본(정점) vs 정규화 메쉬 vs 껍질·과육 단면
from cutsim.plotting import plt

fig = plt.figure(figsize=(13, 4))
ax = fig.add_subplot(1, 3, 1, projection="3d")
v = raw.vertices[:: max(1, len(raw.vertices) // 4000)]
ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=0.5)
ax.set_title(f"raw ({meta['units']}, {meta['up_axis']}-up)", fontsize=9)
ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces, mesh.vertices[:, 2], color="tab:green",
                alpha=0.6, lw=0)
ax.set_title(f"normalized (m, z-up) {np.round(mesh.extents * 100, 1)} cm", fontsize=9)
ax.set_box_aspect(np.array(mesh.extents, dtype=float))
ax = fig.add_subplot(1, 3, 3)
for name, color in (("skin", "tab:red"), ("flesh", "gold")):
    p = out / "parts" / f"{name}.obj"
    if p.exists():
        part = trimesh.load(p, force="mesh")
        sec = part.section(plane_origin=part.bounds.mean(0), plane_normal=[1, 0, 0])
        if sec is not None:
            for ent in sec.discrete:
                ax.plot(ent[:, 1] * 100, ent[:, 2] * 100, color=color, lw=1, label=name)
ax.set_aspect("equal"); ax.set_title("x=center section (cm)", fontsize=9)
h, l = ax.get_legend_handles_labels()
ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8)
fig.suptitle(" / ".join(warnings)[:180] or "no warnings", fontsize=8)
fig.tight_layout(); fig.savefig(out / "report.png", dpi=100)
(out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
print(json.dumps(rep, indent=1, ensure_ascii=False, default=float))

# 6) 절단까지(선택)
if args.cut:
    cmd = [sys.executable, "scripts/01_cut_primitive.py", "--multifield", "--food_mesh",
           str(out / "normalized.obj"), "--category", cat if cat in materials else "default",
           "--set", "food.two_layer=true", "--set", f"mpm.grid_density={args.grid_density:g}",
           "--tag", f"import_{meta['object_id']}"]
    cmd += ["--test", "TR", "--knife_traj", str(out / "traj_sim.npz")] if traj is not None else ["--test", "T1"]
    for part in ("skin", "flesh"):
        if meta.get(f"color_{part}"):
            cmd += [f"--{part}_color", ",".join(f"{c:g}" for c in meta[f"color_{part}"])]
    for hb in meta.get("hold_box") or []:  # 정규화 좌표(m) [cx, cy, cz, sx, sy, sz]: 다른 손 대용 고정 상자
        cmd += ["--hold_box=" + ",".join(f"{v:g}" for v in hb)]  # 값이 - 로 시작해도 옵션으로 오인하지 않게
    cmd += shlex.split(args.cut_args)  # 뒤에 준 값이 앞의 값을 덮는다(--tag 등)
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))
