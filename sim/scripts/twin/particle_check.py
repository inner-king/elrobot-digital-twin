"""자르기 전(입자를 채운 직후) 식재료 입자 중 몸통과 이어지지 않은 입자 수를 센다 — 복원 메쉬와 원래 메쉬 비교.

01_cut_primitive.py 와 같은 방법(껍질·과육 두 층, 격자 256)으로 Genesis 입자를 채우고, 한 스텝도 돌리지 않은 위치에서
반지름 1.25 입자(metrics 의 연결 성분 기준) 안에 이웃이 없는 입자와, 몸통이 아닌 작은 덩어리(20 입자 미만) 입자를 센다.
--smooth N: 메쉬 겉면을 Taubin 으로 N 번 다듬은 뒤 같은 것을 센다(부피는 거의 그대로).
python scripts/twin/particle_check.py data/recon/kitchen_recon_carrot data/recon/kitchen_carrot [--smooth 10]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cutsim.io.recon_loader import load_recon
sys.path.insert(0, str(Path(__file__).parent))
from meshfix import smooth_keep_volume

ap = argparse.ArgumentParser()
ap.add_argument("folders", nargs="+")
ap.add_argument("--smooth", type=int, default=0)
ap.add_argument("--gd", type=float, default=256)
args = ap.parse_args()

import genesis as gs

gs.init(backend=gs.cuda, precision="32", logging_level="warning")
from cutsim.scene.build import build_cut_scene, food_specs_from_mesh, load_yaml, particle_size

cfg0 = load_yaml(ROOT / "configs/scene_default.yaml")
materials = load_yaml(ROOT / "configs/materials.yaml")
p = particle_size(args.gd)
out = {}
for fd in args.folders:
    for sm in sorted({0, args.smooth}):
        meta, mesh, _, _ = load_recon(fd)
        if sm:
            mesh, dv = smooth_keep_volume(mesh, sm)          # 다듬고 부피는 되돌림(dv = 되돌리기 전 줄어든 비율)
        cfg = json.loads(json.dumps(cfg0))
        cfg["mpm"]["grid_density"] = args.gd
        lo, hi = mesh.bounds
        pad = 3.5 / args.gd
        cfg["mpm"]["lower_bound"] = [float(lo[0]) - 0.012 - pad, float(lo[1]) - 0.025 - pad, -0.03]
        cfg["mpm"]["upper_bound"] = [float(hi[0]) + 0.012 + pad, float(hi[1]) + 0.025 + pad, float(hi[2]) + 0.02 + pad]
        cat = meta["category"] if meta["category"] in materials else "default"
        specs, _ = food_specs_from_mesh(mesh, cat, materials, args.gd, True, Path(tempfile.mkdtemp()))
        cs = build_cut_scene(cfg, specs, with_food=True)
        P = np.concatenate(list(cs.particles().values()))
        pairs = cKDTree(P).query_pairs(1.25 * p, output_type="ndarray")
        n = len(P)
        deg = np.bincount(pairs.ravel(), minlength=n)
        A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        k, lab = connected_components(A, directed=False)
        sizes = np.bincount(lab)
        small = sizes[lab] < 20
        key = f"{Path(fd).name}" + (f" (Taubin {sm}번, 다듬을 때 부피 {dv * 100:+.1f}% → 배율로 되돌림)" if sm else "")
        out[key] = {"입자": n, "이웃 없는 입자": int((deg == 0).sum()), "이웃 1~2개": int(((deg > 0) & (deg < 3)).sum()),
                    "작은 덩어리(<20) 입자": int(small.sum()), "덩어리 수": int(k)}
        print(key, out[key], flush=True)
print(json.dumps(out, ensure_ascii=False, indent=1))
