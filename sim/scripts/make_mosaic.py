"""실행 폴더의 카메라 영상 4개를 2×2 로 붙이고, 아래에 힘·날 끝 높이 그래프와 지금 시점 표시를 단 영상 하나로 만든다.

python scripts/make_mosaic.py reports/07_demos/TR_demo_cucumber_slices [--title "..."] [--thumbs_only] [--src render]
결과: <run>/mosaic.mp4, <run>/thumbs.jpg(진행 중 장면 4개 + 절단면 카메라 마지막 장면)
--src render: render_surface.py 가 만든 <run>/render/*.mp4 로 <run>/mosaic_render.mp4, thumbs_render.jpg
여러 폴더를 한 번에 줄 수 있다. 영상이 없는 폴더는 건너뛴다.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMS = ("persp", "side", "top", "face")
CAM_LABEL = {"persp": "persp: 비스듬히 위", "side": "side: 칼날 면 정면", "top": "top: 위에서",
             "face": "face: 절단면 쪽"}
FPS = 25


def font(size):
    from matplotlib import font_manager

    import cutsim.plotting  # noqa: F401  (한글 폰트 이름 설정)
    from cutsim.plotting import plt

    try:
        path = font_manager.findfont(plt.rcParams["font.family"][0], fallback_to_default=True)
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def read(path):
    return np.stack(list(iio.imiter(path, plugin="pyav")))


def series(run):
    """시간축 그래프에 쓸 값: 수직 저항(빈 동작 뺀 값, 위 +), 모델 힘, 날 끝 높이, 단계, 막힘."""
    d = np.load(run / "log.npz")
    m = json.loads((run / "metrics.json").read_text())
    dt = m["sim_time_s"] / m["n_steps"]
    f_base = d["f_base"] if np.ndim(d["f_base"]) else 0.0
    resist = -(d["f"][:, 2] - (f_base[:, 2] if np.ndim(f_base) else 0.0))
    k = max(1, int(0.02 / dt))
    resist = np.convolve(resist, np.ones(k) / k, mode="same")
    model = d["cf"][:, 2] + d["cf"][:, 3] if "cf" in d and np.abs(d["cf"]).sum() > 0 else None
    return dict(t=np.arange(len(resist)) * dt, resist=resist, model=model, z=d["q"][:, 2] * 1000,
                phase=d["phase"], stall=d["stall"] if "stall" in d else None, metrics=m)


def plot_strip(s, width, height):
    """그래프 이미지와 시간 → 가로 픽셀 변환 함수."""
    from cutsim.plotting import plt

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.plot(s["t"], s["resist"], lw=0.9, color="tab:red", label="수직 저항 N(재료가 칼을 밀어 올리는 힘)")
    if s["model"] is not None:
        ax.plot(s["t"], s["model"], lw=0.8, color="tab:orange", alpha=0.8, label="절단 저항 모델 크기 N")
    if s["stall"] is not None and s["stall"].any():
        ax.plot(s["t"][s["stall"]], np.zeros(int(s["stall"].sum())), "|", color="purple", ms=8,
                label="힘 예산 초과로 멈춤")
    ax.set_xlim(0, s["t"][-1]); ax.set_ylabel("N"); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(s["t"], s["z"], "k--", lw=0.8, label="날 끝 높이 mm")
    ax2.set_ylabel("mm")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", ncol=4)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    x0, x1 = ax.get_xlim()
    bb = ax.get_window_extent()
    plt.close(fig)
    return img, lambda t: int(bb.x0 + (t - x0) / (x1 - x0) * bb.width)


def write_mp4(path, frames, fps=FPS):
    c = av.open(str(path), "w")
    st = c.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
    st.options = {"crf": "23", "preset": "veryfast"}
    for fr in frames:
        for p in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
            c.mux(p)
    for p in st.encode():
        c.mux(p)
    c.close()


def thumbs(run, vids, name="thumbs.jpg"):
    tiles = []
    v = vids["persp"]
    for frac in (0.2, 0.45, 0.7, 1.0):
        tiles.append(v[min(len(v) - 1, int(frac * (len(v) - 1)))])
    if "face" in vids:
        tiles.append(vids["face"][-1])
    tiles = [np.asarray(Image.fromarray(t).resize((256, 192))) for t in tiles]
    Image.fromarray(np.concatenate(tiles, 1)).save(run / name, quality=82)


def mosaic(run, title, src="."):
    vdir = run / src
    suffix = "" if src == "." else f"_{Path(src).name}"
    vids = {c: read(vdir / f"{c}.mp4") for c in CAMS if (vdir / f"{c}.mp4").exists()}
    if "persp" not in vids:
        print(f"{run}: 영상 없음, 건너뜀")
        return
    thumbs(run, vids, f"thumbs{suffix}.jpg")
    n = min(len(v) for v in vids.values())
    H, W = next(iter(vids.values())).shape[1:3]
    s = series(run)
    strip, t2x = plot_strip(s, 2 * W, 230)
    f_big, f_small = font(20), font(15)
    out = []
    for k in range(n):
        tiles = []
        for c in CAMS:
            im = Image.fromarray(vids[c][k] if c in vids else np.zeros((H, W, 3), np.uint8))
            ImageDraw.Draw(im).text((8, H - 24), CAM_LABEL[c], fill=(255, 255, 255), font=f_small,
                                    stroke_width=2, stroke_fill=(0, 0, 0))
            tiles.append(np.asarray(im))
        grid = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
        t = k / FPS
        i = min(len(s["t"]) - 1, int(round(t / (s["t"][1] - s["t"][0]))))
        bar = Image.fromarray(strip.copy())
        x = t2x(t)
        ImageDraw.Draw(bar).line([(x, 5), (x, bar.height - 25)], fill=(30, 30, 200), width=2)
        frame = Image.fromarray(np.concatenate([grid, np.asarray(bar)], 0))
        dr = ImageDraw.Draw(frame)
        txt = (f"{title}\n t={t:4.2f}s  단계 {s['phase'][i]}  날 끝 {s['z'][i]:5.1f}mm  수직 저항 {s['resist'][i]:5.1f}N")
        dr.multiline_text((10, 8), txt, fill=(255, 255, 255), font=f_big, stroke_width=3, stroke_fill=(0, 0, 0))
        out.append(np.asarray(frame))
    write_mp4(run / f"mosaic{suffix}.mp4", out)
    print(f"{run / f'mosaic{suffix}.mp4'}: {n} frames, {out[0].shape[1]}x{out[0].shape[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--title", default=None)
    ap.add_argument("--thumbs_only", action="store_true")
    ap.add_argument("--src", default=".", help="카메라 영상이 있는 하위 폴더(render 면 표면 렌더 영상)")
    args = ap.parse_args()
    for r in map(Path, args.runs):
        if args.thumbs_only:
            vids = {c: read(r / f"{c}.mp4") for c in ("persp", "face") if (r / f"{c}.mp4").exists()}
            if "persp" in vids:
                thumbs(r, vids)
            continue
        mosaic(r, args.title or r.name, args.src)
