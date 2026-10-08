"""절단 저항 모델 F = G_c·L(d) + τ·A(d) 의 두 계수를 DiSECt 힘 곡선에 맞춘다(시뮬레이션 없이 해석 기하로).

반원기둥(반지름 5cm)을 칼날 면이 축에 수직으로 가르면, 깊이 d 에서 날 끝이 재료 안에 있는 길이 L 은 반원의 현,
칼 옆면이 잠긴 면적 A 는 활꼴 넓이의 2배(양면)다. 재질별로 cylinder_* 곡선에 맞춘다.

python scripts/04_fit_cut_force.py [--d_max 0.043]
결과: reports/04_cut_force/{fit.json, fit.png}, configs/materials.yaml 의 cut_toughness_Npm / face_friction_Pa
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import nnls

from cutsim.calib.disect import reference_curve
from cutsim.control.cut_force import half_cylinder_chord, half_cylinder_segment_area
from cutsim.plotting import plt

ap = argparse.ArgumentParser()
ap.add_argument("--d_max", type=float, default=0.043)
ap.add_argument("--no_write", action="store_true")
args = ap.parse_args()
R = 0.05
REFS = {"cucumber": "cylinder_fine", "apple": "cylinder_fine_sphereprops", "potato": "cylinder_fine_prismprops"}
out = Path("reports/04_cut_force")
out.mkdir(parents=True, exist_ok=True)

fits = {}
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for k, (mat, ref) in enumerate(REFS.items()):
    d, F, info = reference_curve(ref)
    m = d <= args.d_max
    d, F = d[m], F[m]
    L = half_cylinder_chord(d, R)
    A = 2 * half_cylinder_segment_area(d, R)
    for name, cols in (("cut+fric", [L, A]), ("cut only", [L])):
        X = np.stack(cols, 1)
        coef, _ = nnls(X, F)
        pred = X @ coef
        rmse = float(np.sqrt(np.mean((pred - F) ** 2)))
        fits.setdefault(mat, {})[name] = {"G_c_Npm": float(coef[0]), "tau_Pa": float(coef[1]) if len(coef) > 1 else 0.0,
                                         "rmse_N": rmse, "rel_rmse": rmse / float(np.mean(F)), "ref": ref}
        ax[k].plot(d * 1000, pred, lw=1.2, label=f"{name}: G_c={coef[0]:.0f}N/m" +
                   (f", τ={coef[1]:.0f}Pa" if len(coef) > 1 else "") + f" (RMSE {rmse:.1f}N)")
    ax[k].plot(d * 1000, F, color="0.3", lw=2, label=f"DiSECt {ref}")
    ax[k].set_title(mat); ax[k].set_xlabel("깊이 (mm)"); ax[k].set_ylabel("수직력 (N)"); ax[k].grid(alpha=0.3)
    ax[k].legend(fontsize=7)
fig.tight_layout(); fig.savefig(out / "fit.png", dpi=110)
(out / "fit.json").write_text(json.dumps(fits, indent=2, ensure_ascii=False))
print(json.dumps(fits, indent=1, ensure_ascii=False))

if not args.no_write:  # materials.yaml 에 기록(주석 보존을 위해 해당 줄만 고친다)
    p = Path("configs/materials.yaml")
    lines = p.read_text().splitlines()
    outl, cur = [], None
    for line in lines:
        if line and not line.startswith(" ") and not line.startswith("#") and line.endswith(":"):
            cur = line[:-1]
        if cur in fits and (line.strip().startswith("cut_toughness_Npm") or line.strip().startswith("face_friction_Pa")):
            continue
        outl.append(line)
        if cur in fits and line.strip().startswith("skin_thickness_m"):
            f = fits[cur]["cut+fric"]
            outl.append(f"  cut_toughness_Npm: {f['G_c_Npm']:.1f}   # 절단 저항 모델(04_fit_cut_force.py, {f['ref']})")
            outl.append(f"  face_friction_Pa: {f['tau_Pa']:.1f}")
    p.write_text("\n".join(outl) + "\n")
    print("materials.yaml 갱신")
