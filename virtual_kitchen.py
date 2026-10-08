"""Virtual kitchen scene for the console's virtual camera: real meshes instead of boxes.

Meshes (fetched locally into assets_kitchen/, not redistributed — see the CREDITS files there):
  YCB google_16k scans (CC BY 4.0, real scale): 025_mug, 032_knife, 012_strawberry, 011_banana, 013_apple
  Objaverse / Sketchfab (CC BY): cutting board, metal pot, carrot — rescaled to real sizes below

KitchenCamera has VirtualCamera's interface (start, stop, is_connected, wait_for_frame, hidden, offset,
object_centres). Frames: depth 256×192 float32 m + BGR colour 640×480, rendered by Open3D ray casting with the
meshes' textures and a simple Lambert light; the floor stays the analytic checker plane of VirtualCamera.
"""
from pathlib import Path

import numpy as np

from virtual import FLOOR_Y, VirtualCamera

HERE = Path(__file__).resolve().parent
A = HERE / "assets_kitchen"
YCB = lambda n: A / "ycb" / n / "google_16k" / "textured.obj"
OBJ = lambda n: A / "objaverse" / f"{n}.glb"

# name, file, z_up (YCB) , target longest size (m, None = keep), yaw (deg about up), x, z, standing on ("floor" | name)
LAYOUT = [
    ("도마", OBJ("cutting_board"), False, 0.30, 0, 0.25, 0.00, "floor"),
    ("칼", YCB("032_knife"), True, None, 90, 0.205, 0.03, "도마"),
    ("당근", OBJ("carrot"), False, 0.16, 90, 0.29, 0.03, "도마"),
    ("딸기", YCB("012_strawberry"), True, None, 0, 0.27, -0.10, "도마"),
    ("머그", YCB("025_mug"), True, None, 200, 0.16, -0.21, "floor"),
    ("바나나", YCB("011_banana"), True, None, 30, 0.12, 0.29, "floor"),
    ("사과", YCB("013_apple"), True, None, 0, 0.31, 0.22, "floor"),
    ("냄비", OBJ("pot"), False, 0.27, 0, -0.04, -0.36, "floor"),       # beside the robot, clear of the mug
]


TEX_MAX = 1024
CACHE = A / "kitchen_cache.npz"


def available():
    return all(Path(f).exists() for _, f, *_ in LAYOUT)


def _load(path, z_up, target, yaw):
    """→ (V n×3 in a y-up frame, base on y = 0, centred in x/z), F, UV n×2, texture H×W×3 uint8."""
    import trimesh
    sc = trimesh.load(str(path), force="scene")
    meshes = list(sc.dump())
    m = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    V = np.asarray(m.vertices, float)
    if z_up:                                          # YCB: z up → y up
        V = V @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float).T     # (x, y, z) → (x, z, −y)
    if target:
        V *= target / np.ptp(V, axis=0).max()
    a = np.radians(yaw)
    V = V @ np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]).T
    c = (V.min(0) + V.max(0)) / 2
    V -= [c[0], V[:, 1].min(), c[2]]
    vis = m.visual
    uv = np.asarray(getattr(vis, "uv", None) if getattr(vis, "uv", None) is not None else np.zeros((len(V), 2)), float)
    mat = getattr(vis, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is not None:
        img = img.convert("RGB")
        if max(img.size) > TEX_MAX:                    # 4096² scans → 1024²: same look at the camera's resolution
            img = img.resize((TEX_MAX, TEX_MAX))
        tex = np.asarray(img)
    else:
        col = np.asarray(getattr(mat, "baseColorFactor", [180, 180, 180, 255]))[:3]
        tex = np.full((2, 2, 3), col, np.uint8)
    return V.astype(np.float32), np.asarray(m.faces, np.int64), uv, tex


class KitchenCamera(VirtualCamera):
    def __init__(self):
        super().__init__()
        self.names = [n for n, *_ in LAYOUT]
        self._obj = []
        tops = {"floor": FLOOR_Y}
        key = repr([(n, str(p), zu, t, y) for n, p, zu, t, y, *_ in LAYOUT])
        cache = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}
        loaded = cache["data"].item() if cache.get("key") == key else None
        if loaded is None:                             # first start: parse meshes + textures once (≈17 s), then cached
            loaded = {n: _load(p, zu, t, y) for n, p, zu, t, y, *_ in LAYOUT}
            np.savez(CACHE, key=key, data=np.array(loaded, dtype=object))
        for name, path, z_up, target, yaw, x, z, on in LAYOUT:
            V, F, uv, tex = loaded[name]
            base = np.array([x, tops[on], z], np.float32)
            self._obj.append({"V": V, "F": F, "uv": uv, "tex": tex, "base": base})
            tops[name] = float(tops[on] + V[:, 1].max())
        self._scene_key = None

    # ---- geometry
    def _place(self, i):
        o = self._obj[i]
        V = o["V"]
        a = self.yaw.get(i, 0.0)
        if a:                                          # mesh is centred in x/z: turn about its own vertical axis
            c, s_ = np.cos(a), np.sin(a)
            V = V @ np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]], np.float32).T
        return V + o["base"] + self.offset.get(i, np.zeros(3, np.float32)).astype(np.float32)

    def object_centres(self):
        out = {}
        for i in range(len(self._obj)):
            if i not in self.hidden:
                V = self._place(i)
                out[i] = (V.min(0) + V.max(0)) / 2
        return out

    def _raycaster(self):
        import open3d as o3d
        key = (tuple(sorted(self.hidden)), tuple((k, tuple(np.round(v, 4))) for k, v in sorted(self.offset.items())),
               tuple((k, round(float(v), 3)) for k, v in sorted(self.yaw.items())))
        if key != self._scene_key:
            sc = o3d.t.geometry.RaycastingScene()
            self._gid = {}
            for i, o in enumerate(self._obj):
                if i in self.hidden:
                    continue
                g = sc.add_triangles(o3d.core.Tensor(self._place(i)), o3d.core.Tensor(o["F"].astype(np.uint32)))
                self._gid[g] = i
            self._rc, self._scene_key = sc, key
        return self._rc

    # ---- rendering (overrides the analytic boxes)
    def _render(self, T, w, h, fx):
        import open3d as o3d
        depth, col = self._floor(T, w, h, fx)
        cx, cy = w / 2, h / 2
        vv, uu = np.mgrid[0:h, 0:w]
        dirs = np.stack([(uu - cx) / fx, -(vv - cy) / fx, -np.ones_like(uu, float)], -1) @ T[:3, :3].T
        rays = np.concatenate([np.broadcast_to(T[:3, 3], dirs.shape), dirs], -1).reshape(-1, 6).astype(np.float32)
        r = self._raycaster().cast_rays(o3d.core.Tensor(rays))
        t = r["t_hit"].numpy().reshape(h, w)
        gid = r["geometry_ids"].numpy().reshape(h, w)
        hit = np.isfinite(t) & ((depth <= 0) | (t < depth))
        if hit.any():
            prim = r["primitive_ids"].numpy().reshape(h, w)[hit]
            bary = r["primitive_uvs"].numpy().reshape(h, w, 2)[hit]
            nrm = r["primitive_normals"].numpy().reshape(h, w, 3)[hit]
            g = gid[hit]
            rgb = np.zeros((hit.sum(), 3))
            for gg, i in self._gid.items():
                sel = g == gg
                if not sel.any():
                    continue
                o = self._obj[i]
                f = o["F"][prim[sel]]
                b = bary[sel]
                uv = (1 - b[:, :1] - b[:, 1:]) * o["uv"][f[:, 0]] + b[:, :1] * o["uv"][f[:, 1]] + b[:, 1:] * o["uv"][f[:, 2]]
                th, tw = o["tex"].shape[:2]
                px = np.clip((uv[:, 0] % 1.0) * (tw - 1), 0, tw - 1).astype(int)
                py = np.clip((1 - uv[:, 1] % 1.0) * (th - 1), 0, th - 1).astype(int)
                rgb[sel] = o["tex"][py, px]
            light = np.array([0.35, 0.85, 0.4]); light /= np.linalg.norm(light)
            shade = 0.55 + 0.45 * np.abs(nrm @ light)
            col[hit] = np.clip(rgb * shade[:, None], 0, 255).astype(np.uint8)[:, ::-1]     # RGB → BGR
            depth = np.where(hit, t, depth)
        return depth.astype(np.float32), col

    def _floor(self, T, w, h, fx):
        cx, cy = w / 2, h / 2
        vv, uu = np.mgrid[0:h, 0:w]
        d = np.stack([(uu - cx) / fx, -(vv - cy) / fx, -np.ones_like(uu, float)], -1) @ T[:3, :3].T
        o = T[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = (FLOOR_Y - o[1]) / d[..., 1]
        hitp = o + d * tf[..., None]
        floor = tf > 0
        col = np.zeros((h, w, 3), np.uint8)
        checker = ((np.floor(hitp[..., 0] / 0.1) + np.floor(hitp[..., 2] / 0.1)) % 2).astype(bool)
        col[floor] = np.where(checker[floor, None], [190, 190, 190], [150, 150, 150])
        return np.where(floor, tf, 0.0), col
