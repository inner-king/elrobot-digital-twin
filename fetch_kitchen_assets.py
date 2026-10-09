"""Download the virtual kitchen meshes from their original sources into assets_kitchen/ (not redistributed here).

  YCB google_16k scans (CC BY 4.0): 025_mug, 032_knife, 012_strawberry, 011_banana, 013_apple
      Calli et al., "The YCB object and model set", ICAR 2015 — ycb-benchmarks S3
  Objaverse / Sketchfab (CC BY): cutting board (klessgyzen), metal pot (warkarma), carrot (xiezhong)
      fetched by uid from the Objaverse 1.0 release on Hugging Face (allenai/objaverse)

Usage: uv run --with objaverse python fetch_kitchen_assets.py      (≈60 MB; then the virtual camera shows the kitchen)
"""
import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

A = Path(__file__).resolve().parent / "assets_kitchen"
YCB_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/{}_google_16k.tgz"
YCB = ["025_mug", "032_knife", "012_strawberry", "011_banana", "013_apple"]
OBJAVERSE = {
    "cutting_board": {"uid": "9d45f15fb21f4fc89118b2631eec5fda", "name": "Cutting Board Asset", "author": "klessgyzen"},
    "pot": {"uid": "f77a7095794e460a8ba0b2a55f20a27c", "name": "Old Metal Pot", "author": "warkarma"},
    "carrot": {"uid": "583b980bb3c4432abe5422d8a54ffe99", "name": "carrot", "author": "xiezhong"},
}


def fetch_ycb():
    for n in YCB:
        if (A / "ycb" / n / "google_16k" / "textured.obj").exists():
            continue
        print("YCB", n)
        data = urllib.request.urlopen(YCB_URL.format(n), timeout=120).read()
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(A / "ycb", filter="data")
    (A / "ycb" / "CREDITS.txt").write_text(
        "YCB Object and Model Set (google_16k meshes) — CC BY 4.0\n"
        "Calli, Singh, Walsman, Srinivasa, Abbeel, Dollar, \"The YCB object and model set\", ICAR 2015.\n"
        f"Source: {YCB_URL.format('<object>')}\nObjects: {', '.join(YCB)}\n")


def fetch_objaverse():
    import objaverse
    out = A / "objaverse"
    out.mkdir(parents=True, exist_ok=True)
    todo = {k: v for k, v in OBJAVERSE.items() if not (out / f"{k}.glb").exists()}
    if todo:
        print("Objaverse", ", ".join(todo))
        paths = objaverse.load_objects(uids=[v["uid"] for v in todo.values()], download_processes=1)
        for k, v in todo.items():
            shutil.copy(paths[v["uid"]], out / f"{k}.glb")
    (out / "CREDITS.json").write_text(json.dumps(
        {k: {**v, "license": "by", "url": f"https://sketchfab.com/3d-models/{v['uid']}"} for k, v in OBJAVERSE.items()},
        indent=1, ensure_ascii=False))


if __name__ == "__main__":
    fetch_ycb()
    fetch_objaverse()
    print("done:", A)
