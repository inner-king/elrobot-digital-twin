"""스모크 테스트: (1) torch CUDA (2) Genesis GPU 백엔드 (3) 헤드리스 렌더가 NVIDIA 인지 (4) MPM 예제 완주·속도·VRAM.

python scripts/00_smoke_test.py [--grid_density 128] [--steps 200]
결과: reports/00_smoke/{smoke.json, smoke.mp4}
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # 헤드리스 렌더는 EGL 로 강제

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--grid_density", type=float, default=128)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--out", default="reports/00_smoke")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
result = {}

# (1) torch
result["torch"] = torch.__version__
result["cuda_capability"] = list(torch.cuda.get_device_capability())
result["gpu"] = torch.cuda.get_device_name()
assert torch.cuda.is_available() and (torch.ones(3, device="cuda") * 2).sum().item() == 6

# (2) Genesis 백엔드
import genesis as gs

gs.init(backend=gs.cuda, precision="32", logging_level="warning")
result["genesis"] = gs.__version__
result["gs_backend"] = str(gs.backend)
result["gs_device"] = str(gs.device)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=2e-3, substeps=20),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-0.15, -0.15, -0.05), upper_bound=(0.15, 0.15, 0.25), grid_density=args.grid_density
    ),
    vis_options=gs.options.VisOptions(visualize_mpm_boundary=True),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(
    material=gs.materials.MPM.ElastoPlastic(E=3e5, nu=0.3, rho=1000, von_mises_yield_stress=3e4),
    morph=gs.morphs.Box(pos=(0.0, 0.0, 0.08), size=(0.06, 0.06, 0.06)),
    surface=gs.surfaces.Default(color=(0.9, 0.6, 0.3), vis_mode="particle"),
)
cam = scene.add_camera(res=(640, 480), pos=(0.35, -0.35, 0.25), lookat=(0, 0, 0.04), fov=40, GUI=False)
t0 = time.time()
scene.build()
result["build_s"] = round(time.time() - t0, 1)
result["n_particles"] = int(cube.n_particles)

# (3) 렌더러 확인
# GL 컨텍스트가 렌더 스레드에 있어 glGetString 을 못 쓰므로, 실제로 올라온 GL 라이브러리로 판정한다.
t0 = time.time()
for _ in range(10):
    cam.render()
result["render_ms"] = round((time.time() - t0) / 10 * 1000, 1)
maps = Path("/proc/self/maps").read_text()
libs = sorted({line.split()[-1].rsplit("/", 1)[-1] for line in maps.splitlines()
               if any(k in line for k in ("EGL", "GLX", "nvidia-gl", "nvidia-egl", "swrast", "llvmpipe", "mesa"))})
result["gl_libs"] = libs
result["render_on_nvidia"] = any("nvidia" in l for l in libs) and not any(k in " ".join(libs) for k in ("swrast", "llvmpipe"))

# (4) MPM 완주, 스텝당 시간, VRAM
cam.start_recording(save_to_filename=str(out / "smoke.mp4"), fps=25)
torch.cuda.synchronize()
t0 = time.time()
for i in range(args.steps):
    scene.step()
torch.cuda.synchronize()
elapsed = time.time() - t0
pos = cube.get_particles_pos()
pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
cam.stop_recording()

result["steps"] = args.steps
result["sim_time_s"] = args.steps * 2e-3
result["wall_s_per_step"] = round(elapsed / args.steps, 4)
result["nan"] = bool(np.isnan(pos).any())
result["final_z_min_max"] = [round(float(pos[..., 2].min()), 4), round(float(pos[..., 2].max()), 4)]
free, total = torch.cuda.mem_get_info()
result["vram_used_gb_device"] = round((total - free) / 1e9, 2)
(out / "smoke.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps(result, indent=2, ensure_ascii=False))
