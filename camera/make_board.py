"""Print-ready ChArUco floor board.

Input : camera/board.json (squares, square/marker size in m, ArUco dictionary)
Output: camera/boards/charuco_a4.pdf (+ .png), A4 landscape at 300 dpi, true scale, with a 50 mm check ruler.
Run   : uv run --with opencv-python --with numpy --with pillow python camera/make_board.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
cfg = json.loads((HERE / "board.json").read_text())
DPI = 300
MM = DPI / 25.4                              # px per mm
PAGE_W, PAGE_H = round(297 * MM), round(210 * MM)  # A4 landscape

board = cv2.aruco.CharucoBoard((cfg["squares_x"], cfg["squares_y"]), cfg["square_m"], cfg["marker_m"],
                               cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"])))
bw_mm, bh_mm = cfg["squares_x"] * cfg["square_m"] * 1000, cfg["squares_y"] * cfg["square_m"] * 1000
bw, bh = round(bw_mm * MM), round(bh_mm * MM)
img = board.generateImage((bw, bh), marginSize=0, borderBits=1)

page = Image.new("L", (PAGE_W, PAGE_H), 255)
x0, y0 = (PAGE_W - bw) // 2, round(14 * MM)
page.paste(Image.fromarray(img), (x0, y0))
d = ImageDraw.Draw(page)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", round(3.2 * MM))
except OSError:
    font = ImageFont.load_default()
# 50 mm scale check below the board
ry = y0 + bh + round(10 * MM)
d.line([(x0, ry), (x0 + round(50 * MM), ry)], fill=0, width=round(0.4 * MM))
for t in (0, 50):
    xx = x0 + round(t * MM)
    d.line([(xx, ry - round(2 * MM)), (xx, ry + round(2 * MM))], fill=0, width=round(0.4 * MM))
d.text((x0 + round(53 * MM), ry - round(2 * MM)), "50 mm  (measure to check 100% print scale)", fill=0, font=font)
d.text((x0, ry + round(5 * MM)),
       f"ChArUco {cfg['squares_x']}x{cfg['squares_y']}  square {cfg['square_m']*1000:.0f} mm  marker {cfg['marker_m']*1000:.0f} mm  {cfg['dictionary']}"
       f"   |   x axis: left -> right,  origin: board centre,  up: out of the paper", fill=0, font=font)

out = HERE / "boards" / "charuco_a4"
page.save(out.with_suffix(".png"), dpi=(DPI, DPI))
page.convert("RGB").save(out.with_suffix(".pdf"), resolution=DPI)
print(f"board {bw_mm:.0f}x{bh_mm:.0f} mm on A4 landscape -> {out}.pdf / .png")
