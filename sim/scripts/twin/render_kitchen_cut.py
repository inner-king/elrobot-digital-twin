"""복원 주방 전체 + 칼을 쥔 ElRobot + 썰리는 당근을 실제 물체처럼(입자 대신 겉면) 여러 각도에서 그린다.

당근: render_surface.py 메쉬 방식(처음 메쉬를 칼질 평면으로 나누고 조각마다 입자 움직임으로 옮김). 조각 강체 맞춤은
  떨어져 나간 입자를 빼고 다시 맞춘다(그대로는 흩어진 입자가 조각을 끌었다). 단면은 겉살(주황)·심(연한 주황) 색.
칼: 시뮬레이션과 같은 칼날 + 손잡이. 로봇: URDF 메쉬, 칼 자세에서 IK(쥐기는 시나리오 meta 의 robot_grip).
나머지: 가상 카메라 360° 영상으로 복원한 메쉬(data/kitchen_recon/full, 꼭짓점 색). 복원된 칼은 두께 18 mm 덩어리라
  자를 수 없어 빼고, 로봇이 쥔 칼 모델로 대신한다. 당근 몸통을 붙잡은 고정 상자(--grip)는 그리지 않는다.
물리 모드: <run>/kitchen_frames.npz(sim_kitchen_cut.py 결과)가 있으면 주방 물체·로봇을 시뮬레이션이 계산한 자세 그대로 그린다
  (물체 = 그 실행에 넣은 메쉬, 로봇 = Genesis 링크 자세, 좌표 변환 T_wc 도 그 실행 값). 없으면 예전처럼 정적 배치 + IK.
python scripts/twin/render_kitchen_cut.py reports/09_elrobot_twin/robot_cut/TR_robot_carrot [--first_only] [--cam_set far]
결과: <run>/kitchen_render/{wide,top,back,near}.mp4, mosaic.mp4, stills.jpg
"""
import argparse
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import open3d as o3d
import trimesh
from open3d.visualization import rendering
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(Path(__file__).parent)]
import render_surface as RS
from cutsim.io.recon_loader import load_recon
from make_robot_cut import knife_T, knife_grip
from robot_kin import Robot

CAMS = {  # 이름: (눈, 보는 곳, 위, fov, 설명) — 세계 좌표(로봇 원점, 앞 +x, 바닥 z=0)
    "wide": ((0.98, -0.80, 0.70), (0.12, 0.05, 0.03), (0, 0, 1), 44, "전체: 비스듬히 앞"),
    "top": ((0.11, 0.07, 1.30), (0.11, 0.07, 0.0), (1, 0, 0), 52, "전체: 위에서"),
    "back": ((-0.55, -0.45, 0.62), (0.22, 0.02, 0.03), (0, 0, 1), 46, "전체: 로봇 뒤에서"),
    "near": ((0.56, -0.40, 0.30), (0.24, -0.03, 0.03), (0, 0, 1), 40, "도마 둘레: 썰기와 주변 물체"),
}
CAMS_FAR = CAMS
CAMS_CLOSE = {  # 조금 더 가까이(주방 전체가 아니라 로봇·도마·둘레 물체가 화면을 채우게)
    "wide": ((0.70, -0.52, 0.46), (0.17, 0.0, 0.05), (0, 0, 1), 44, "비스듬히 앞"),
    "top": ((0.16, 0.0, 0.92), (0.16, 0.0, 0.0), (1, 0, 0), 52, "위에서"),
    "back": ((-0.30, -0.33, 0.42), (0.22, -0.01, 0.04), (0, 0, 1), 46, "로봇 뒤에서"),
    "near": ((0.38, 0.21, 0.19), (0.235, -0.035, 0.035), (0, 0, 1), 40, "도마 둘레: 썰기(손잡이 반대쪽에서)"),
}
PHASE = {"start": "준비", "ready": "준비", "approach0": "첫 번째 자리로", "cut0": "첫 번째 썰기(앞뒤 톱질, 누르는 힘 12 N 이하)",
         "dwell0": "첫 번째 썰기 끝", "lift0": "칼 들어 올림", "approach1": "두 번째 자리로",
         "cut1": "두 번째 썰기(앞뒤 톱질, 누르는 힘 12 N 이하)", "dwell1": "두 번째 썰기 끝", "lift1": "칼 들어 올림",
         "retreat": "물러남"}


def font(size):
    try:
        from matplotlib import font_manager
        import cutsim.plotting  # noqa: F401
        from cutsim.plotting import plt
        return ImageFont.truetype(font_manager.findfont(plt.rcParams["font.family"][0], fallback_to_default=True), size)
    except Exception:
        return ImageFont.load_default()


def robust_deform(self, Xf, r_max=0.004):
    """MeshPiece.deform + 떨어져 나간 입자를 강체 맞춤에서 빼고 다시 맞춘다."""
    P = Xf[self.idx]
    keep = np.ones(len(P), bool)
    for _ in range(4):
        R, t = RS.kabsch(self.Xr[keep], P[keep])
        e = np.linalg.norm(P - (self.Xr @ R.T + t), axis=1)
        new = e < max(r_max, 3 * float(np.median(e)))
        if new.sum() < 10 or (new == keep).all():
            break
        keep = new
    res = P - (self.Xr @ R.T + t)
    ok = np.linalg.norm(res, axis=1) < r_max
    w = self.w * ok[self.nb]
    V = self.V0 @ R.T + t + (w[..., None] * res[self.nb]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9)
    return [V[a:b] for a, b in zip(self.cuts[:-1], self.cuts[1:])]


def carrot_colors(R, depth, skin, base_skin, base_flesh):
    """당근: 단면은 겉살(진한 주황)과 심(연한 주황) + 경계의 옅은 고리, 껍질은 가로 주름."""
    n1 = RS.vnoise(R, 0.004)
    cortex, core = np.array([0.93, 0.45, 0.09]), np.array([0.98, 0.64, 0.24])
    flesh = RS.mix(cortex, core, RS.smoothstep(0.0045, 0.0080, depth)) * (1 + 0.04 * n1)[:, None]
    flesh = flesh * (1 - 0.10 * np.exp(-((depth - 0.0060) / 0.0007) ** 2))[:, None]
    sk = np.asarray(base_skin, float)[None] * (1 + 0.10 * RS.vnoise(R + 0.3, 0.0015))[:, None]
    ridge = 0.5 + 0.5 * np.sin(R[:, 0] / 0.0024 * 2 * np.pi + 2.5 * RS.vnoise(R, 0.012))
    sk = sk * (1 - 0.10 * ridge ** 8)[:, None]
    w = np.maximum(RS.smoothstep(0.30, 0.65, skin), RS.smoothstep(0.0012, 0.0006, depth))
    return np.clip(RS.mix(flesh, sk, w), 0, 1)


def o3d_mesh(m, color=None):
    g = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(m.vertices, float)),
                                  o3d.utility.Vector3iVector(np.asarray(m.faces, np.int32)))
    g.compute_vertex_normals()
    if color is not None:
        g.vertex_colors = o3d.utility.Vector3dVector(np.clip(color, 0, 1) ** 2.2)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--scenario", default=str(ROOT / "data/recon/kitchen_robot_cut_carrot"))
    ap.add_argument("--recon", default=str(ROOT / "data/kitchen_recon/full"))
    ap.add_argument("--res", default="800,600")
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--first_only", action="store_true")
    ap.add_argument("--recon_floor", action="store_true", help="복원된 바닥 메쉬도 그린다")
    ap.add_argument("--cam_set", default="close", choices=["close", "far"])
    ap.add_argument("--no_physics", action="store_true", help="kitchen_frames.npz 가 있어도 정적 배치로")
    args = ap.parse_args()
    global CAMS
    CAMS = CAMS_CLOSE if args.cam_set == "close" else CAMS_FAR
    run, scen, rdir = Path(args.run), Path(args.scenario), Path(args.recon)
    W0, H0 = (int(v) for v in args.res.split(","))
    W, H = W0 * args.ss, H0 * args.ss

    d = np.load(run / "frames.npz")
    meta = json.loads(str(d["meta"]))
    X, LAB, T, Q = d["x"], d["lab"], d["t"], d["q"]
    x_rest = d["x_rest"].astype(float)
    p = meta["particle_size"]
    stride = max(1, int(round(meta["fps"] / args.fps)))
    idx = np.arange(0, len(X), stride)
    smeta, _, _, T_norm = load_recon(scen)
    import yaml
    smeta = yaml.safe_load((scen / "meta.yaml").read_text())
    grip = smeta["robot_grip"]
    G = knife_grip(np.radians(grip["alpha_deg"]), grip["s_m"], meta["knife"]["length"], meta["knife"]["height"])
    tr = np.load(scen / "traj_knife.npz")
    t_tr, ph_tr = tr["t"], tr["phase"]

    kfp = run / "kitchen_frames.npz"
    phys = kfp.exists() and not args.no_physics
    # ---- 복원 장면(정적): 바닥 물체는 바닥 1 mm 위, 도마 위 물체는 도마 윗면(아래로 쏜 광선) 위에
    info = json.load(open(rdir / "info.json"))
    objs = {o["match"]: dict(o, mesh=trimesh.load(rdir / "objects" / f"obj_{o['id']}.obj", force="mesh", process=False))
            for o in info["objects"]}
    board = objs["도마"]["mesh"]
    board.apply_translation([0, 0, 0.001 - board.bounds[0, 2]])

    def board_top(xy):
        o = np.c_[xy, np.full(len(xy), 1.0)]
        loc, ray, _ = board.ray.intersects_location(o, np.tile([0, 0, -1.0], (len(xy), 1)), multiple_hits=False)
        return float(np.median(loc[:, 2])) if len(loc) else float(board.bounds[1, 2])

    def footprint(m, n=7):
        lo, hi = m.bounds
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
        return np.c_[gx.ravel(), gy.ravel()]

    static = {}
    for name, o in objs.items():
        if name in ("당근", "칼"):
            continue
        m = o["mesh"]
        if name != "도마":
            surf = 0.0 if (o.get("support_mm") or 0) < 5 else board_top(footprint(m))
            m.apply_translation([0, 0, surf + 0.0008 - m.bounds[0, 2]])
        static[name] = m
    # 시뮬레이션(정규화) 좌표 → 세계: 당근 바닥이 그 자리 도마 윗면에 오게
    T_wc = np.linalg.inv(T_norm)
    car_w = trimesh.load(meta["food_mesh"], force="mesh").apply_transform(T_wc)
    dz = board_top(footprint(car_w)) - (T_wc @ [0, 0, 0, 1])[2]
    T_wc[2, 3] += dz
    print(f"당근 바닥 높이를 도마 윗면에 맞춤: {dz * 1000:+.1f} mm", flush=True)
    if phys:                                             # 물리 모드: 그 실행이 쓴 메쉬·변환·자세 그대로
        K = np.load(kfp)
        km = json.loads(str(K["meta"]))
        T_wc = K["T_wc"]
        static = {n: trimesh.load(km["obj_files"][n], force="mesh", process=False) for n in km["obj_names"]}
        kidx = np.searchsorted(K["step"], d["step"])
        assert (K["step"][np.minimum(kidx, len(K["step"]) - 1)] == d["step"]).all(), "frames.npz 와 kitchen_frames.npz 스텝이 다르다"
        lname = {n: i for i, n in enumerate(km["link_names"])}

        def pose(p, q):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            T[:3, 3] = p
            return T
        print(f"물리 모드: 물체 {len(static)}개·로봇 링크 {len(lname)}개 자세를 시뮬레이션 결과로", flush=True)

    # ---- 로봇 관절(칼 자세 → 집게 목표 → IK)
    rb = Robot(T_base=np.asarray(grip["T_base"]))
    q_grip = Robot.grip_angle(15.0)                      # 손잡이 두께 15 mm
    Tk_w = [T_wc @ knife_T(Q[f]) for f in idx]
    best, rng = None, np.random.default_rng(1)
    for k in range(0 if phys else 16):
        r = rb.ik(Tk_w[0] @ G, np.zeros(7) if k == 0 else rng.uniform(rb.lo, rb.hi))
        if best is None or r[1] + 0.05 * r[2] < best[1] + 0.05 * best[2]:
            best = r
    if not phys:
        q, Qa, errs = best[0], [], []
        for Tk in Tk_w:
            q, e1, e2 = rb.ik(Tk @ G, q)
            Qa.append(q); errs.append((e1, e2))
        errs = np.array(errs)
        print(f"IK {len(Qa)}프레임: 위치 오차 최대 {errs[:, 0].max():.2f} mm, 방향 {errs[:, 1].max():.2f}°", flush=True)

    # ---- 장면
    rr = rendering.OffscreenRenderer(W, H)
    sc = rr.scene
    sc.set_background([0.86, 0.86, 0.84, 1.0])
    sc.scene.set_sun_light([0.35, 0.45, -0.82], [1.0, 0.97, 0.92], 70000)
    sc.scene.enable_sun_light(True)
    sc.scene.enable_indirect_light(True)
    sc.scene.set_indirect_light_intensity(30000)
    floor = o3d.geometry.TriangleMesh.create_box(3.0, 3.0, 0.02)
    floor.translate([-1.4, -1.4, -0.02 - (0.004 if args.recon_floor else 0.0)])
    floor.compute_vertex_normals()
    sc.add_geometry("floor", floor, RS.mat((0.62, 0.62, 0.64), rough=0.6))
    if args.recon_floor and (rdir / "background.obj").exists():
        bgm = trimesh.load(rdir / "background.obj", force="mesh", process=False)
        sc.add_geometry("recon_floor", o3d_mesh(bgm, np.asarray(bgm.visual.vertex_colors)[:, :3] / 255.0), RS.mat(rough=0.7))
    for name, m in static.items():
        col = np.asarray(m.visual.vertex_colors)[:, :3] / 255.0
        mt = RS.mat(rough=0.55 if name != "냄비" else 0.35, metal=0.0, reflect=0.4)
        sc.add_geometry(f"s_{name}", o3d_mesh(m, col), mt)
    m_pla, m_servo = RS.mat((0.90, 0.90, 0.88), rough=0.55), RS.mat((0.10, 0.10, 0.11), rough=0.35)
    rlinks = rb.meshes()
    for n, m, servo in rlinks:
        sc.add_geometry(f"r_{n}", o3d_mesh(m), m_servo if servo else m_pla)
    blade, handle = RS.knife_meshes(meta["knife"])
    sc.add_geometry("blade", o3d_mesh(blade), RS.mat((0.78, 0.80, 0.83), rough=0.38, metal=0.65, reflect=0.6))
    sc.add_geometry("handle", o3d_mesh(handle), RS.mat((0.10, 0.08, 0.07), rough=0.45))

    RS.MeshPiece.deform = robust_deform
    RS.generic_colors = carrot_colors
    cols = meta.get("colors", {})
    base_skin = np.clip(np.array(cols.get("skin", smeta["color_skin"])) / 0.8, 0, 1)   # 가상 카메라 음영(평균 0.8)을 되돌림
    base_flesh = np.array(cols.get("flesh", smeta["color_flesh"]))
    t_b = time.time()
    pieces = RS.build_mesh_pieces(meta, x_rest, X, LAB, d["step"], None, base_skin, base_flesh, p)
    m_skin = RS.mat(rough=0.45, reflect=0.4)
    m_skin.base_clearcoat, m_skin.base_clearcoat_roughness = 0.3, 0.3
    cap_mats = {}
    for pc in pieces:
        for nm, *_, img in pc.parts:
            if img is not None:
                mc = RS.mat(rough=0.32, reflect=0.5)
                mc.base_clearcoat, mc.base_clearcoat_roughness = 0.6, 0.15
                mc.albedo_img = o3d.geometry.Image(img)
                cap_mats[nm] = mc
    print(f"당근 메쉬 조각 {len(pieces)}개 ({time.time() - t_b:.0f}s)", flush=True)

    # ---- 프레임: 앞 0.6 s·뒤 0.8 s 는 멈춘 장면
    seq = [0] * int(0.6 * args.fps) + list(range(len(idx))) + [len(idx) - 1] * int(0.8 * args.fps)
    if args.first_only:
        ks = [int(np.argmin(np.abs(T[idx] - t))) for t in (0.2, float(t_tr[np.argmax(ph_tr == "cut0")]) + 1.2,
                                                           float(t_tr[np.argmax(ph_tr == "cut1")]) + 1.2, T[idx][-1])]
        seq = ks
    out = run / "kitchen_render"
    out.mkdir(exist_ok=True)
    F_t, F_s = font(22), font(17)
    writers = {}
    if not args.first_only:
        for c in list(CAMS) + ["mosaic"]:
            cont = av.open(str(out / f"{c}.mp4"), "w")
            st = cont.add_stream("libx264", rate=int(args.fps))
            st.width, st.height = (W0 * 2, H0 * 2 + 70) if c == "mosaic" else (W0, H0)
            st.pix_fmt, st.options = "yuv420p", {"crf": "20", "preset": "medium"}
            writers[c] = (cont, st)
    for c, (eye, look, up, fov, _) in CAMS.items():     # 미리 몇 장 그려 렌더러를 데운다
        rr.scene.camera.set_projection(fov, W / H, 0.01, 6.0, rendering.Camera.FovType.Vertical)
        rr.scene.camera.look_at(look, eye, up)
        for _ in range(2):
            rr.render_to_image()
    food_names, stills, t0 = [], [], time.time()
    for n, k in enumerate(seq):
        f = idx[k]
        for nm in food_names:
            sc.remove_geometry(nm)
        food_names = []
        P = X[f].astype(float)
        R_, t_ = T_wc[:3, :3], T_wc[:3, 3]
        for pc in pieces:
            for (nm, V0, Fp, Cp, uvp, img), Vf in zip(pc.parts, pc.deform(P)):
                gm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vf @ R_.T + t_),
                                               o3d.utility.Vector3iVector(Fp.astype(np.int32)))
                gm.compute_vertex_normals()
                if Cp is not None:
                    gm.vertex_colors = o3d.utility.Vector3dVector(np.clip(Cp, 0, 1) ** 2.2)
                    sc.add_geometry(nm, gm, m_skin)
                else:
                    gm.triangle_uvs = o3d.utility.Vector2dVector(uvp)
                    sc.add_geometry(nm, gm, cap_mats[nm])
                food_names.append(nm)
        if phys:
            j = kidx[f]
            for nl, _, _ in rlinks:
                sc.set_geometry_transform(f"r_{nl}", T_wc @ pose(K["links_pos"][j, lname[nl]], K["links_quat"][j, lname[nl]]))
            for i_o, nm in enumerate(km["obj_names"]):
                sc.set_geometry_transform(f"s_{nm}", T_wc @ pose(K["obj_pos"][j, i_o], K["obj_quat"][j, i_o]))
        else:
            Tl = rb.fk(Qa[k], q_grip)
            for nl, _, _ in rlinks:
                sc.set_geometry_transform(f"r_{nl}", Tl[nl])
        for nm in ("blade", "handle"):
            sc.set_geometry_transform(nm, Tk_w[k])
        imgs = {}
        for c, (eye, look, up, fov, _) in CAMS.items():
            rr.scene.camera.set_projection(fov, W / H, 0.01, 6.0, rendering.Camera.FovType.Vertical)
            rr.scene.camera.look_at(look, eye, up)
            img = np.asarray(rr.render_to_image())
            for _ in range(5):                            # 렌더러가 가끔 화면 일부를 검게 낸다(첫 프레임에서 봄) → 다시 그림
                if (img.max(axis=2) < 4).mean() < 0.02:
                    break
                img = np.asarray(rr.render_to_image())
            imgs[c] = np.asarray(Image.fromarray(img).resize((W0, H0), Image.LANCZOS)) if args.ss > 1 else img
        ph = str(ph_tr[min(int(np.searchsorted(t_tr, T[f])), len(ph_tr) - 1)])
        mos = Image.new("RGB", (W0 * 2, H0 * 2 + 70), (250, 250, 250))
        dr = ImageDraw.Draw(mos)
        dr.text((12, 8), ("복원한 가상 주방 전체를 한 물리 장면에(물체·로봇·당근) — ElRobot 이 칼을 쥐고 당근 썰기, Genesis, 겉면 렌더" if phys else
                          "복원한 가상 주방에서 ElRobot 이 칼을 쥐고 복원된 당근을 썰기 — Genesis MPM, 겉면 렌더"), fill=(20, 20, 20), font=F_t)
        dr.text((12, 40), f"{PHASE.get(ph, ph)} · 시뮬레이션 {T[f]:.2f} s", fill=(70, 70, 70), font=F_s)
        for i, c in enumerate(CAMS):
            x0, y0 = (i % 2) * W0, 70 + (i // 2) * H0
            mos.paste(Image.fromarray(imgs[c]), (x0, y0))
            dr.rectangle([x0 + 6, y0 + 6, x0 + 18 + 12 * len(CAMS[c][4]), y0 + 32], fill=(255, 255, 255))
            dr.text((x0 + 10, y0 + 8), CAMS[c][4], fill=(20, 20, 20), font=F_s)
        if args.first_only:
            stills.append(np.asarray(mos))
            for c in CAMS:
                Image.fromarray(imgs[c]).save(out / f"still{n}_{c}.png")
        else:
            for c in CAMS:
                cont, st = writers[c]
                for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(imgs[c]), format="rgb24")):
                    cont.mux(pk)
            cont, st = writers["mosaic"]
            for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(np.asarray(mos)), format="rgb24")):
                cont.mux(pk)
        if n % 25 == 0:
            print(f"  {n}/{len(seq)} t={T[f]:.2f}s {(time.time() - t0) / (n + 1):.2f}s/frame", flush=True)
    for cont, st in writers.values():
        for pk in st.encode():
            cont.mux(pk)
        cont.close()
    if args.first_only:
        Image.fromarray(np.concatenate(stills, 0)).resize((W0, (H0 * 2 + 70) * len(stills) // 2), Image.LANCZOS).save(
            out / "stills.jpg", quality=88)
        print(out / "stills.jpg")
    print(f"done {len(seq)} frames in {time.time() - t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
