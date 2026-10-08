"""Print-ready floor marker: AprilTag 36h11 ID 0, black square 60 mm, centred on A4 portrait (300 dpi).

Print scale does not matter: the floor pose takes metric size from LiDAR (stream.py: _floor_pose).
Run: uv run --with opencv-python --with numpy --with pillow python camera/make_tag.py
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
DPI, TAG_MM, TAG_ID = 300, 60, 0
MM = DPI / 25.4
W, H = round(210 * MM), round(297 * MM)
side = round(TAG_MM * MM)
# 36h11 = 6x6 data bits + 1-bit black border → 8 cells across the black square
tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), TAG_ID, side, borderBits=1)
page = Image.new("L", (W, H), 255)
x0, y0 = (W - side) // 2, (H - side) // 2
page.paste(Image.fromarray(tag), (x0, y0))
d = ImageDraw.Draw(page)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", round(3.5 * MM))
except OSError:
    font = ImageFont.load_default()
# arrow = floor +x, so the user knows which way the frame points
ay = y0 + side + round(12 * MM)
d.line([(x0, ay), (x0 + side, ay)], fill=0, width=round(0.5 * MM))
d.polygon([(x0 + side, ay), (x0 + side - round(3 * MM), ay - round(2 * MM)), (x0 + side - round(3 * MM), ay + round(2 * MM))], fill=0)
d.text((x0 + side + round(3 * MM), ay - round(2.2 * MM)), "x", fill=0, font=font)
d.text((x0 - round(20 * MM), ay + round(6 * MM)), f"AprilTag 36h11  ID {TAG_ID}  ({TAG_MM} mm)   origin = tag centre, up = out of paper",
       fill=0, font=font)
out = HERE / "boards" / "floor_tag_id0"
page.save(out.with_suffix(".png"), dpi=(DPI, DPI))
page.convert("RGB").save(out.with_suffix(".pdf"), resolution=DPI)
print(f"{out}.png / .pdf  tag {side}px = {side / MM:.2f} mm")
