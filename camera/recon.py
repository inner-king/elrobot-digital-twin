"""Local, dynamic TSDF reconstruction of the room from the iPhone stream (CPU, Open3D VoxelBlockGrid).

Input  (≤5 Hz from stream.py): depth 256×192 float32 [m] (low-confidence and far pixels zeroed),
                               RGB resized to the depth resolution, depth intrinsics, camera-to-world (OpenCV axes, ARKit world)
Output (every ~1.5 s):         triangle mesh in ARKit world [m]: vertices N×3 float32, normals N×3 float32,
                               colors N×3 uint8, triangles M×3 uint32 — served as one binary blob (/recon_mesh)

Per-object TSDF (object_tsdf): for a selected object, the keyframe depth is masked to the object's box and fused
at 3 mm voxels into its own small grid — a sharper, closed object surface than the 1 cm room mesh.

Completion (camera/complete.py, every rebuild): each object's unseen sides are closed as a box / cylinder / height map
down to the support; floor holes (enclosed by observed floor, or under an object) are filled flat (/recon_fill).

Object split (every rebuild, needs the floor height): room-mesh triangles more than SPLIT_ABOVE_M above the floor
are split into connected pieces; each object-sized piece standing on the floor gets its own per-object TSDF and is
cut out of the room mesh, which then holds only the background (/recon_objects, /recon_mesh).

Dynamic scene handling (objects that move or disappear):
  1. Keyframes: a frame is kept when the camera moved ≥ KF_MOVE_M or turned ≥ KF_TURN_DEG since the last
     keyframe; below that it refreshes the newest keyframe. An older keyframe from (almost) the same viewpoint
     (≤ KF_REPLACE_M / KF_REPLACE_DEG, a revisit) is dropped — the newest view of a place wins.
  2. The TSDF is rebuilt from the surviving keyframes every REBUILD_EVERY_S (no stale accumulation).
  3. Newest observation wins: each mesh vertex is checked against the newest (≤ 2) keyframes that observe it;
     when they all see *through* it (3×3 minimum depth deeper than the vertex by more than CARVE_TAU_M) the vertex
     is removed — the ghost of an object that moved or disappeared.
"""
import threading
import time

import numpy as np

from complete import CONTAINERS, carve_object, complete_container, complete_object, fill_floor, fuse_volume, hybrid

try:
    import open3d as o3d
    import open3d.core as o3c
except Exception as e:  # optional dependency
    o3d = None
    _IMPORT_ERROR = str(e)

VOXEL_M = 0.01
# truncation in voxels (Open3D default 8 = 8 cm at 1 cm voxels): a wide band smears every object into a skirt on the
# floor around it; LiDAR noise is ~2 mm, so a few voxels are enough
TRUNC_VOXELS = 4.0          # measured: floor skirt within 6 cm of objects 45 → 12 triangles (2 voxels opens holes)
BLOCK_RES = 8          # 8³ voxels per block (8 cm blocks at 1 cm voxels)
BLOCK_COUNT = 30000    # preallocated capacity ≈ 300 MB
DEPTH_MAX_M = 2.5
MAX_SUBMIT_HZ = 5
KF_MOVE_M, KF_TURN_DEG = 0.06, 6.0        # new keyframe after this much motion since the newest one
# an older keyframe is replaced only by a revisit of (almost) the same viewpoint; this must stay below the
# keyframe spacing above, otherwise every new keyframe deletes its predecessor and only the newest view survives
KF_REPLACE_M, KF_REPLACE_DEG = 0.04, 4.0
EXTRACT_MIN_WEIGHT = 1.0                    # keyframes are mostly unique per place: one observation is enough
KF_MAX = 200
REBUILD_EVERY_S = 1.5
CARVE_FRAMES, CARVE_TAU_M, CARVE_MIN = 15, 0.03, 2
OBJ_VOXEL_M, OBJ_KEYFRAMES, OBJ_MARGIN_M = 0.003, 40, 0.012
COMPLETE_MODE = "auto"         # auto: carving when seen all around, else symmetry priors (+ observed faces kept)
CARVE_MIN_COVER_DEG = 240
SKIN_M = 0.005                 # object-TSDF triangles entirely below support + this are the support, not the object   # per-object TSDF (finer voxels, object box only)
# automatic object split of the room mesh: pieces standing on the floor, object-sized
SPLIT_ABOVE_M, SPLIT_MIN_TRIS, SPLIT_MAX_W_M, SPLIT_MAX_H_M, SPLIT_TOUCH_M = 0.008, 40, 0.45, 0.45, 0.04
FLOOR_PX_M = 0.004            # floor orthophoto cell
FLOOR_BAND_M = 0.015          # room-mesh triangles this close to the floor give way to the orthophoto plane (far floor
                              # at grazing view is ±1 cm noisy: at 6 mm a jagged ridge stayed on the horizon)
FLOOR_SNAP_M = 0.006
# a flat, upward surface inside a floor object (cutting board, tray, plate) is a support of its own: pieces more than
# SUPPORT_ABOVE_M above it become separate objects standing on it
SUPPORT_MIN_AREA_M2, SUPPORT_SHARE, SUPPORT_ABOVE_M, SUPPORT_MIN_RISE_M = 0.008, 0.35, 0.006, 0.012
SUPPORT_ENCLOSE = 0.6
INST_ONLY_MIN_OBS = 5
PRIOR_CONF = 0.5               # label confidence needed before a category shape prior is used
REG_STABLE, REG_UNLABELLED, REG_TTL_S = 3, 8, 60.0   # committing objects (see _commit)
MOVE_MATCH_M = 0.5             # a committed object not where it was: same-shape piece this far away (top view) = it, moved
REFINE_COVER_DEG = 60          # a committed object is completed again once the views around it grew this much
TRACK_MIN_FIT = 0.5
INST_OWN = 0.3                 # …and at least this share of the instance's own points lie in that object
INST_COVER = 0.5               # a geometric object this covered by a recognised instance is replaced by it          # a piece above whose top-view box covers this share of the support's: a container wall         # floor-band vertices snapped onto the floor plane
FOOT_MARGIN_M = 0.015          # footprint margin cut out of the background under each object


def _mesh_blob(V, T, C, N=None):
    """u32 nv | u32 nt | f32 v | f32 n | u32 tris | u8 rgb — the /recon_mesh layout."""
    V = np.asarray(V, np.float32)
    T = np.asarray(T, np.uint32).reshape(-1, 3)
    if N is None:                                      # area-weighted vertex normals
        N = np.zeros_like(V)
        if len(T):
            fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
            for k in range(3):
                np.add.at(N, T[:, k], fn)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    C = np.asarray(C, float)
    C = np.clip(C * (255.0 if len(C) and C.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    return (np.array([len(V), len(T)], np.uint32).tobytes() + V.tobytes() + np.asarray(N, np.float32).tobytes()
            + T.tobytes() + C.tobytes())


def _blob_arrays(b, off):
    nv, nt = [int(x) for x in np.frombuffer(b[off:off + 8], np.uint32)]
    off += 8
    V = np.frombuffer(b[off:off + nv * 12], np.float32).reshape(-1, 3).astype(float); off += nv * 24
    T = np.frombuffer(b[off:off + nt * 12], np.uint32).reshape(-1, 3).astype(np.int64); off += nt * 12
    C = np.frombuffer(b[off:off + nv * 3], np.uint8).reshape(-1, 3).astype(float); off += nv * 3
    return V, T, C, off


def _rigid(A, B):
    """rigid transform (4×4) taking corresponding points A → B (Kabsch)"""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Wt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Wt.T @ U.T))])
    R = Wt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cb - R @ ca
    return T


def _consistency(V, T, obs, kfs):
    """How well a candidate surface explains the views (lower = better, metres): RMS distance of the observed points to
    it + 2 cm × share of the observed points it leaves uncovered (> 1 cm) + 3 cm × share of it inside observed free space."""
    import trimesh
    from scipy.spatial import cKDTree
    from objmodel import free_violation
    S = trimesh.Trimesh(V, T, process=False).sample(4000)
    d = cKDTree(S).query(obs)[0]
    return float(np.sqrt(np.mean(np.minimum(d, 0.01) ** 2)) + 0.02 * (d > 0.01).mean() + 0.03 * free_violation(S, kfs))


def _azimuth_coverage(pts, kfs):
    """Degrees of the horizon around the object (30° sectors) from which some keyframe looked at it."""
    if not kfs:
        return 0
    c = pts.mean(axis=0)
    d = np.array([k["T"][:3, 3] - c for k in kfs])
    ang = np.degrees(np.arctan2(d[:, 2], d[:, 0])) % 360
    return int(len(np.unique((ang // 30).astype(int))) * 30)


def _angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


class Reconstructor:
    def __init__(self):
        self.status = {"available": o3d is not None, "running": False, "frames": 0, "keyframes": 0, "blocks": 0,
                       "verts": 0, "tris": 0, "carved": 0, "integrate_ms": 0.0, "extract_ms": 0.0, "version": 0,
                       "voxel_mm": VOXEL_M * 1000, "error": None if o3d is not None else f"open3d 없음: {_IMPORT_ERROR}"}
        self._lock = threading.Lock()
        self._kf = []                # list of dicts: depth, rgb, K, T (cam→world, OpenCV axes), t
        self._mesh = None
        self._objects = b""          # u32 count | per object: u32 id | raw TSDF mesh blob | completed mesh blob
        self._fill = b""             # inferred floor (mesh blob)
        self._floor_img = None       # floor orthophoto (bounds, png)
        self._floor_thread = None
        self.floor_y = None          # set by stream.py (ARKit world y of the floor)
        self.object_points = {}      # object id → TSDF vertices (≤ 2000, ARKit world) for detect.py's association
        self.labeler = None          # detect.Detector.label(id) → {"name", "conf", ...} or None
        self.instances = None        # detect.Detector.instances() → recognised per-instance TSDFs
        self.registry = {}           # object id → committed mesh / label / pose (_commit, _track_registry)
        self._prev_obj = []          # (id, centre) for stable ids between rebuilds
        self._next_id = 1
        self.status["objects"] = []
        self._last_submit = 0.0
        self._dirty = False
        if o3d is not None:
            threading.Thread(target=self._run, daemon=True).start()

    # ---- called from the camera thread
    def submit(self, depth, rgb, fx, fy, cx, cy, T_world_cvcam):
        if not self.status["running"] or o3d is None:
            return
        now = time.time()
        if now - self._last_submit < 1.0 / MAX_SUBMIT_HZ:
            return
        self._last_submit = now
        self.status["frames"] += 1
        T = np.asarray(T_world_cvcam, np.float64)
        with self._lock:
            if self._kf:
                last = self._kf[-1]["T"]
                dist, turn = np.linalg.norm(T[:3, 3] - last[:3, 3]), _angle_deg(T[:3, :3], last[:3, :3])
                if dist < KF_MOVE_M and turn < KF_TURN_DEG:
                    if dist < 0.01 and turn < 1.0:
                        # standing still at the newest keyframe: refresh its images (pose kept, so they stay registered)
                        self._kf[-1].update(depth=depth.copy(), rgb=rgb.copy(), t=now)
                        self._dirty = True
                    return              # between keyframes: skip (the keyframe must not follow the camera)
            # drop older keyframes taken from (almost) the same place: the new view supersedes them
            self._kf = [k for k in self._kf
                        if not (np.linalg.norm(T[:3, 3] - k["T"][:3, 3]) < KF_REPLACE_M
                                and _angle_deg(T[:3, :3], k["T"][:3, :3]) < KF_REPLACE_DEG)]
            self._kf.append({"depth": depth.copy(), "rgb": rgb.copy(), "T": T, "t": now,
                             "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)})
            if len(self._kf) > KF_MAX:
                self._kf = self._kf[-KF_MAX:]
            self.status["keyframes"] = len(self._kf)
            self._dirty = True

    def handle(self, c):
        t = c["type"]
        if o3d is None:
            raise RuntimeError(self.status["error"])
        if t == "recon_start":
            self.status["running"] = True
        elif t == "recon_stop":
            self.status["running"] = False
        elif t == "recon_reset":
            with self._lock:
                self._kf = []
                self._mesh = None
                self._objects = b""
                self._fill = b""
                self._floor_img = None
                self.registry = {}
            self.status["objects"] = []
            self.status["fill"] = None
            self.status["floor_photo"] = None
            self.status.update(frames=0, keyframes=0, blocks=0, verts=0, tris=0, carved=0,
                               version=self.status["version"] + 1)

    def objects_bytes(self):
        with self._lock:
            return self._objects

    def object_geometry(self, oid):
        """[ours] One split object for the grasp planner: (completed mesh V, T, observed TSDF vertices, support y) in
        the ARKit world, or None. Objects standing on a board are separate here; a region grown from a click is not."""
        with self._lock:
            b = self._objects
        info = next((o for o in self.status.get("objects") or [] if o["id"] == oid), None)
        if not b or len(b) < 4 or info is None:
            return None
        off = 4
        for _ in range(int(np.frombuffer(b[:4], np.uint32)[0])):
            i = int(np.frombuffer(b[off:off + 4], np.uint32)[0])
            raw, _, _, off = _blob_arrays(b, off + 4)
            V, T, _, off = _blob_arrays(b, off)
            if i == oid:
                return V, T, raw, (self.floor_y or 0.0) + info["support_mm"] / 1000
        return None

    def floor_png(self):
        """(json bounds, PNG bytes) of the floor orthophoto, or None."""
        with self._lock:
            return self._floor_img

    def fill_bytes(self):
        with self._lock:
            return self._fill

    def mark_changed(self, lo, hi):
        """[ours] The scene inside the ARKit-world box lo..hi changed (our own grasp moved something): drop what the
        existing keyframes saw there, so the TSDF rebuilds that region only from views taken after the change.
        Dropped: every depth pixel whose ray (camera → measured point) passes through the box, not only the points in
        it — an old ray through the box that hit the floor behind it is "empty here" evidence, and it carved away an
        object put down inside the box (measured: a placed strawberry vanished in 1 of 3 runs, and for good live)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        n = 0
        with self._lock:
            for k in self._kf:
                dep, K, T = k["depth"], k["K"], k["T"]
                h, w = dep.shape
                vv, uu = np.mgrid[0:h, 0:w]
                P = np.stack([(uu - K[0, 2]) / K[0, 0] * dep, (vv - K[1, 2]) / K[1, 1] * dep, dep], -1) @ T[:3, :3].T + T[:3, 3]
                o = T[:3, 3]
                d = P - o                                    # segment o → P, t ∈ [0, 1] (slab test)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t1, t2 = (lo - o) / d, (hi - o) / d
                tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
                tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
                inside = (dep > 0) & (tmax >= np.maximum(tmin, 0)) & (tmin <= 1)
                if inside.any():
                    k["depth"] = np.where(inside, 0, dep).astype(np.float32)
                    n += int(inside.sum())
            self._dirty = True
        self.status["cleared_px"] = n
        return n

    def mesh_bytes(self):
        with self._lock:
            return self._mesh

    # ---- worker
    def _run(self):
        while True:
            time.sleep(REBUILD_EVERY_S)
            if not self._dirty:
                continue
            self._dirty = False
            try:
                with self._lock:
                    kfs = list(self._kf)
                if kfs:
                    self._rebuild(kfs)
                self.status["error"] = None
            except Exception as e:
                self.status["error"] = str(e)

    def _rebuild(self, kfs):
        t0 = time.time()
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)), voxel_size=VOXEL_M, block_resolution=BLOCK_RES,
            block_count=BLOCK_COUNT, device=o3c.Device("CPU:0"))
        for k in kfs:
            d = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(k["depth"], np.float32)))
            c = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(k["rgb"], np.float32)))
            Kt, Et = o3c.Tensor(k["K"]), o3c.Tensor(np.linalg.inv(k["T"]))
            blk = vbg.compute_unique_block_coordinates(d, Kt, Et, 1.0, DEPTH_MAX_M, trunc_voxel_multiplier=TRUNC_VOXELS)
            vbg.integrate(blk, d, c, Kt, Et, 1.0, DEPTH_MAX_M, trunc_voxel_multiplier=TRUNC_VOXELS)
        self.status["integrate_ms"] = round((time.time() - t0) * 1000, 1)
        t1 = time.time()
        m = vbg.extract_triangle_mesh(weight_threshold=EXTRACT_MIN_WEIGHT).to_legacy()
        v = np.asarray(m.vertices, np.float64)
        tri = np.asarray(m.triangles, np.int64)
        keep, newest_rgb = self._visibility_keep(v, kfs[-CARVE_FRAMES:])
        colv = np.asarray(m.vertex_colors).copy()
        have = newest_rgb[:, 0] >= 0
        if len(colv) == len(v) and have.any():           # colour = newest keyframe that sees the vertex (no stale blends)
            colv[have] = newest_rgb[have] * (255.0 if colv.max(initial=0) > 1.5 else 1.0)
            m.vertex_colors = o3d.utility.Vector3dVector(colv)
        carved = int((~keep).sum())
        if carved:
            m.remove_triangles_by_mask(~(keep[tri].all(axis=1)))
            m.remove_unreferenced_vertices()
        t2 = time.time()
        self._carve_kfs = kfs
        objs, fill = self._split_objects(m) if self.floor_y is not None else (b"", b"")
        self.status["split_ms"] = round((time.time() - t2) * 1000, 1)
        m.compute_vertex_normals()
        v = np.asarray(m.vertices, np.float32)
        n = np.asarray(m.vertex_normals, np.float32)
        colv = np.asarray(m.vertex_colors)
        col = np.clip(colv * (255.0 if colv.max(initial=0) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        tri = np.asarray(m.triangles, np.uint32)
        blob = (np.array([len(v), len(tri)], np.uint32).tobytes() + v.tobytes() + n.tobytes()
                + tri.tobytes() + col.tobytes())
        with self._lock:
            self._mesh = blob
            self._objects = objs
            self._fill = fill
        self.status.update(verts=int(len(v)), tris=int(len(tri)), blocks=int(vbg.hashmap().size()), carved=carved,
                           extract_ms=round((time.time() - t1) * 1000, 1), version=self.status["version"] + 1)

    def _split_objects(self, m):
        """[ours] Cut object-sized pieces standing on the floor out of the room mesh `m` (in place) and fuse each
        with its own per-object TSDF. Returns the /recon_objects blob; status["objects"] lists them."""
        fy = self.floor_y
        v = np.asarray(m.vertices)
        tri = np.asarray(m.triangles)
        if not len(tri):
            self.status["objects"] = []
            return b"", b""
        above = (v[:, 1] > fy + SPLIT_ABOVE_M) & (v[:, 1] < fy + SPLIT_MAX_H_M + 0.1)
        cand = above[tri].all(axis=1)
        sub = o3d.geometry.TriangleMesh(m)
        sub.remove_triangles_by_mask(~cand)
        ids, cnt, _ = sub.cluster_connected_triangles()
        ids, cnt = np.asarray(ids), np.asarray(cnt)
        st = np.asarray(sub.triangles)
        sv = np.asarray(sub.vertices)
        cut = np.zeros(len(tri), bool)
        cand_idx = np.nonzero(cand)[0]                  # sub's triangle k == room triangle cand_idx[k]
        out, info, prev = [], [], self._prev_obj
        obj_pts, geo_pts, geo_mesh = {}, [], []
        pieces = []
        for c in np.nonzero(cnt >= SPLIT_MIN_TRIS)[0]:
            pts = sv[np.unique(st[ids == c])]
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            w = max(hi[0] - lo[0], hi[2] - lo[2])
            if w > SPLIT_MAX_W_M or hi[1] - fy > SPLIT_MAX_H_M or lo[1] - fy > SPLIT_ABOVE_M + SPLIT_TOUCH_M:
                continue                                 # wall / furniture / floating piece: background
            cut[cand_idx[ids == c]] = True
            box_lo, box_hi = lo - OBJ_MARGIN_M, hi + OBJ_MARGIN_M
            box_lo[1] = fy + 0.002
            try:
                _, stt, (ov, otri, ocol) = self.object_tsdf(box_lo, box_hi, arrays=True, core=(lo, hi))
            except Exception:
                continue
            if not len(otri):
                continue
            # [ours] a board / tray inside this piece is a support of its own: split what stands on it (3 mm mesh;
            # the 1 cm room mesh smears a 13 mm board into the floor)
            for sel, sup, cap in self._split_on_support(ov, otri, np.arange(len(otri)), fy):
                T = otri[sel]
                if cap is not None:                      # the support keeps only itself, not what stands on it
                    T = T[(ov[T][:, :, 1] <= cap).all(axis=1)]
                # the support's own surface inside the object box (a thin floor/board skin) is not the object
                T = T[~(ov[T][:, :, 1] < sup + SKIN_M).all(axis=1)] if cap is None else T
                if len(T) < SPLIT_MIN_TRIS:
                    continue
                keep, inv = np.unique(T, return_inverse=True)
                pv, pc, pt = ov[keep], ocol[keep], inv.reshape(-1, 3)
                plo, phi = pv.min(axis=0), pv.max(axis=0)
                centre = (plo + phi) / 2
                pieces.append((pv, pc, pt, sup, plo, phi, centre))
        oids = self._assign_ids(pieces, prev)
        for (pv, pc, pt, sup, plo, phi, centre), oid in zip(pieces, oids):
            cov = _azimuth_coverage(pv, self._carve_kfs)
            reg = self.registry.get(oid)
            if reg is not None and reg.get("committed") and cov < reg["cov"] + REFINE_COVER_DEG:
                # [ours] committed object: no re-completion, only its 6-DoF pose follows the observation (ICP)
                t1 = time.time()
                V, Tm, C = self._track_registry(reg, pv)
                obj_pts[oid] = pv[np.random.default_rng(0).choice(len(pv), min(len(pv), 2000), replace=False)]
                geo_pts.append(obj_pts[oid])
                geo_mesh.append((pv, pt, pc, sup, (V, Tm)))
                out.append(np.array([oid], np.uint32).tobytes() + _mesh_blob(pv, pt, pc) + _mesh_blob(V, Tm, C))
                info.append({**reg["info"], "_lo": plo.copy(), "_hi": phi.copy(), "id": int(oid), "verts": int(len(pv)),
                             "centre": ((V.min(0) + V.max(0)) / 2).round(3).tolist(), "committed": True,
                             "track_ms": round((time.time() - t1) * 1000, 1), "track_fit": reg.get("fit"),
                             "_on_floor": sup == fy, "coverage_deg": cov, "_obs_c": centre.copy()})
                continue
            comp = complete_object(pv, pc, sup)              # [ours] fill the unseen sides
            # views all around → space carving (measured best: 1.3–2.8 mm, 94–100 %); a one-sided sweep leaves the
            # back unknown and carving fills it to the box edge (a 47 mm carrot came out 69 mm) → symmetry priors
            if COMPLETE_MODE == "carve" or (COMPLETE_MODE == "auto" and cov >= CARVE_MIN_COVER_DEG):
                try:
                    cv_, ct_, cc_ = carve_object(pv, pc, sup, self._carve_kfs)
                    comp.update(V=cv_, T=ct_, C=cc_, shape="carved")
                except Exception:
                    pass
            comp["coverage_deg"] = cov
            if COMPLETE_MODE in ("hybrid", "carve", "auto"):  # observed faces kept, completion only where unseen
                comp["V"], comp["T"], comp["C"] = self._fuse(pv, pt, pc, comp)   # one closed, smooth surface
            obj_pts[oid] = pv[np.random.default_rng(0).choice(len(pv), min(len(pv), 2000), replace=False)]
            geo_pts.append(obj_pts[oid])
            geo_mesh.append((pv, pt, pc, sup, (comp["V"], comp["T"])))
            out.append(np.array([oid], np.uint32).tobytes() + _mesh_blob(pv, pt, pc) + _mesh_blob(comp["V"], comp["T"], comp["C"]))
            info.append({"_lo": plo.copy(), "_hi": phi.copy(), "id": int(oid), "size_mm": (np.ptp(pv, axis=0) * 1000).round(0).tolist(),
                         "centre": centre.round(3).tolist(), "verts": int(len(pv)), "ms": stt["ms"],
                         "shape": comp["shape"], "dims_mm": comp["dims_mm"], "rms_mm": comp["rms_mm"], "fits_mm": comp["fits_mm"],
                         "complete_ms": comp["ms"], "support_mm": round((sup - fy) * 1000), "_on_floor": sup == fy,
                     "coverage_deg": comp.get("coverage_deg"), "_obs_c": centre.copy()})
        if self.instances is not None:
            out, info = self._merge_instances(out, info, geo_pts, geo_mesh, obj_pts, fy)
        self._commit(out, info)
        # the object's bottom (below SPLIT_ABOVE_M) belongs to the object: cut its footprint out of the background;
        # the floor it hides is filled with the rest of the floor holes below
        must = []
        for o in [o for o in info if o["_on_floor"]]:
            lo, hi = o["_lo"] - FOOT_MARGIN_M, o["_hi"] + FOOT_MARGIN_M
            inxz = (v[:, 0] > lo[0]) & (v[:, 0] < hi[0]) & (v[:, 2] > lo[2]) & (v[:, 2] < hi[2])
            # any raised corner inside the footprint: a triangle bridging floor and object edge stands up as a "tooth"
            cut |= (inxz & (v[:, 1] > fy + 0.003))[tri].any(axis=1) & inxz[tri].all(axis=1)
            must.append((lo, hi))
        if cut.any():                                    # room mesh keeps only the background
            m.remove_triangles_by_mask(cut)
            m.remove_unreferenced_vertices()
        # [ours] planar prior: floor-band vertices onto the floor plane (1 cm marching cubes leaves ±2 mm ripples
        # that the scene light turns into a bumpy, low-poly look)
        V = np.asarray(m.vertices).copy()
        band = np.abs(V[:, 1] - fy) < FLOOR_SNAP_M
        V[band, 1] = fy
        m.vertices = o3d.utility.Vector3dVector(V)
        FV, FT, FC, fst = fill_floor(np.asarray(m.vertices), np.asarray(m.vertex_colors), fy, must)   # [ours]
        # the textured floor plane (orthophoto, background thread: ≈230 ms) replaces the floor band of the room mesh —
        # its 1 cm marching-cubes ripples and the colour bleed around objects looked broken
        if self._floor_thread is None or not self._floor_thread.is_alive():
            vf, kfs = np.asarray(m.vertices).copy(), list(self._carve_kfs)
            self._floor_thread = threading.Thread(target=self._floor_job, args=(vf, fy, must, kfs), daemon=True)
            self._floor_thread.start()
        fl = (np.abs(np.asarray(m.vertices)[:, 1] - fy) < FLOOR_BAND_M)[np.asarray(m.triangles)].all(axis=1)
        m.remove_triangles_by_mask(fl)
        m.remove_unreferenced_vertices()
        self.status["fill"] = fst
        fill = _mesh_blob(FV, FT, FC, np.tile([0, 1, 0], (len(FV), 1))) if len(FV) else b""
        self._prev_obj = [(o["id"], np.array(o.get("_obs_c", o["centre"]))) for o in info]
        for o in info:
            for k in [k for k in o if k.startswith("_")]:
                o.pop(k)
        if self.labeler:
            for o in info:
                if "label" not in o:
                    o["label"] = self.labeler(o["id"])
        self.status["objects"] = info
        self.object_points = obj_pts
        return np.array([len(out)], np.uint32).tobytes() + b"".join(out), fill

    def _floor_job(self, v, fy, must, kfs):
        try:
            t0 = time.time()
            img = self._orthophoto(v, fy, must, kfs)
            img[0]["ms"] = round((time.time() - t0) * 1000)
            with self._lock:
                self._floor_img = img
            self.status["floor_photo"] = img[0]
        except Exception as e:
            self.status["floor_photo"] = {"error": str(e)}

    def _orthophoto(self, v, fy, must, kfs, cell=FLOOR_PX_M):
        """[ours] Floor texture (RGBA PNG): every 4 mm cell of the floor plane takes its colour from the keyframe that
        sees it closest, among those whose depth there and in the 5×5 pixels around agrees with the floor (an object
        in front or at the silhouette edge — where the downsampled colour is a blend — does not paint the floor).
        Object footprints and small unseen gaps are inpainted (OpenCV Telea); large unseen areas stay transparent."""
        import cv2
        fl = v[np.abs(v[:, 1] - fy) < 0.01]
        if len(fl) < 100:
            raise RuntimeError("바닥 정점 부족")
        lo = np.percentile(fl[:, [0, 2]], 1, axis=0) - 0.05
        hi = np.percentile(fl[:, [0, 2]], 99, axis=0) + 0.05
        nx, nz = np.ceil((hi - lo) / cell).astype(int)
        X, Z = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * cell, lo[1] + (np.arange(nz) + 0.5) * cell, indexing="xy")
        P = np.stack([X.ravel(), np.full(X.size, fy), Z.ravel()], 1)      # image rows = +z, columns = +x
        img = np.zeros((len(P), 3), np.float32)
        zbest = np.full(len(P), np.inf)
        ker = np.ones((5, 5), np.uint8)
        for k in kfs:
            dep, rgb, K = k["depth"], k["rgb"], k["K"]
            if rgb.shape[:2] != dep.shape:
                continue
            Tcw = np.linalg.inv(k["T"])
            pc = P @ Tcw[:3, :3].T + Tcw[:3, 3]
            z = pc[:, 2]
            ok = (z > 0.05) & (z < zbest)
            zs = np.where(ok, z, 1)
            uf = K[0, 0] * pc[:, 0] / zs + K[0, 2]
            wf = K[1, 1] * pc[:, 1] / zs + K[1, 2]
            h, wd = dep.shape
            ok &= (uf >= 0) & (uf < wd - 1) & (wf >= 0) & (wf < h - 1)
            idx = np.nonzero(ok)[0]
            u, w = np.round(uf[idx]).astype(np.int64), np.round(wf[idx]).astype(np.int64)
            # per pixel: does the depth lie on the floor plane? (ray–plane depth; LiDAR noise grows with range)
            vv, uu = np.mgrid[0:h, 0:wd]
            dy = (k["T"][1, :3] * np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)).sum(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                zf = (fy - k["T"][1, 3]) / dy
            onf = (dep > 0.05) & (np.abs(dep - zf) < np.maximum(0.006, 0.012 * dep))
            onf = cv2.erode(onf.astype(np.uint8), ker) > 0          # and so are its neighbours: no silhouette edge
            idx = idx[onf[w, u]]
            # bilinear colour: the 256×192 keyframe is coarse next to the 4 mm cells
            x0, y0 = np.floor(uf[idx]).astype(np.int64), np.floor(wf[idx]).astype(np.int64)
            ax, ay = (uf[idx] - x0)[:, None], (wf[idx] - y0)[:, None]
            c = ((1 - ax) * (1 - ay) * rgb[y0, x0] + ax * (1 - ay) * rgb[y0, x0 + 1]
                 + (1 - ax) * ay * rgb[y0 + 1, x0] + ax * ay * rgb[y0 + 1, x0 + 1])
            img[idx] = c
            zbest[idx] = z[idx]
        have = np.isfinite(zbest).reshape(nz, nx)
        img = np.clip(img, 0, 255).reshape(nz, nx, 3).astype(np.uint8)
        foot = np.zeros((nz, nx), np.uint8)
        for blo, bhi in must:                                  # object footprints: repainted from around them
            a = np.floor((np.asarray(blo)[[0, 2]] - lo) / cell).astype(int)
            b = np.ceil((np.asarray(bhi)[[0, 2]] - lo) / cell).astype(int)
            foot[max(a[1], 0):max(b[1] + 1, 0), max(a[0], 0):max(b[0] + 1, 0)] = 1
        near = cv2.morphologyEx(have.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # gaps ≤ 6 cm
        alpha = (near > 0) | (foot > 0)
        img = cv2.inpaint(img, (~have | (foot > 0)).astype(np.uint8), 5, cv2.INPAINT_TELEA)   # sources: seen floor only
        rgba = np.dstack([img[..., ::-1], (alpha * 255).astype(np.uint8)])          # RGB → BGR(A) for OpenCV
        _, png = cv2.imencode(".png", rgba)
        meta = {"lo": [float(lo[0]), float(lo[1])], "hi": [float(lo[0] + nx * cell), float(lo[1] + nz * cell)],
                "y": float(fy), "px": [int(nx), int(nz)], "seen": round(float(have.mean()), 3),
                "version": int(self.status.get("version", 0)) + 1}
        return meta, png.tobytes()

    def _fuse(self, pv, pt, pc, comp):
        """observed mesh + completion → one closed smooth surface (complete.fuse_volume); stitched as fallback"""
        try:
            return fuse_volume(pv, pt, pc, comp["V"], comp["T"], comp["C"], kfs=self._carve_kfs,
                               shell=comp.get("shape") == "container" or "plausible" in comp)
        except Exception as e:
            self.status["fuse_error"] = str(e)
            return hybrid(pv, pt, pc, comp["V"], comp["T"], comp["C"])

    def _assign_ids(self, pieces, prev):
        """[ours] Stable ids. 1) a piece within 3 cm of last rebuild's piece keeps its id (closest first);
        2) a committed object not found that way moved (picked up and placed): the leftover piece of the same height
        and footprint within MOVE_MATCH_M takes it (cheapest first); 3) the rest are new objects."""
        oids = [None] * len(pieces)
        C = [p[6] for p in pieces]
        pairs = sorted((np.linalg.norm(q - C[k]), k, i) for k in range(len(pieces)) for i, q in prev
                       if np.linalg.norm(q - C[k]) < 0.03)
        used = set()
        for _, k, i in pairs:
            if oids[k] is None and i not in used:
                oids[k] = i
                used.add(i)
        pairs = []
        for k, p in enumerate(pieces):
            if oids[k] is not None:
                continue
            size = np.ptp(p[0], axis=0)
            for rid, r in self.registry.items():
                if not r.get("committed") or rid in used:
                    continue
                rs = np.array(r["info"]["size_mm"]) / 1000
                dh = size[1] / max(rs[1], 1e-3)                      # height survives any yaw
                fp = np.sort(size[[0, 2]]) / np.maximum(np.sort(rs[[0, 2]]), 1e-3)   # footprint, yaw-free
                d = np.linalg.norm((C[k] - r["obs_c"])[[0, 2]])
                if d < MOVE_MATCH_M and 0.7 < dh < 1.4 and (0.5 < fp).all() and (fp < 2.0).all():
                    pairs.append((d / MOVE_MATCH_M + abs(np.log(dh)) + np.abs(np.log(fp)).sum(), k, rid))
        for _, k, rid in sorted(pairs):
            if oids[k] is None and rid not in used:
                oids[k] = rid
                used.add(rid)
                self.registry[rid]["moved"] = True
        for k in range(len(pieces)):
            if oids[k] is None:
                oids[k] = self._next_id
                self._next_id += 1
        return oids

    def _track_registry(self, reg, obs):
        """ICP of the committed mesh to what is observed now (observed → model); the correction moves the model."""
        from objmodel import track
        V, T, C = reg["mesh"]
        # coarse: the observed piece's top-view centre (it moved by more than the ICP reach); fine: ICP 2 cm → 1 cm
        # (observed centre now − observed centre last time: both one-sided alike, the completed model's centre is not)
        sh = np.zeros(3)
        oc, mc = (obs.min(0) + obs.max(0)) / 2, (V.min(0) + V.max(0)) / 2
        d = (oc - reg.get("obs_c", oc))[[0, 2]]
        if np.linalg.norm(d) > 0.008:
            sh[[0, 2]] = d
        S0 = V[np.random.default_rng(0).choice(len(V), min(len(V), 4000), replace=False)]
        # a moved (picked and placed) object may also have turned: yaw hypotheses about its centre, best ICP kept
        yaws = (0, 45, 90, 135, 180, 225, 270, 315) if reg.pop("moved", False) else (0,)
        best = None
        for yd in yaws:
            a = np.radians(yd)
            R0 = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            T0 = np.eye(4)
            T0[:3, :3] = R0
            T0[:3, 3] = mc + sh - R0 @ mc                       # turn about the model centre, then shift
            S = S0 @ T0[:3, :3].T + T0[:3, 3]
            T1, f1, _ = track(S, obs, dist=0.02)
            S1 = S @ T1[:3, :3].T + T1[:3, 3]
            T2, fit, rmse = track(S1, obs, dist=0.01)
            if best is None or (fit - 20 * rmse) > best[0]:
                best = (fit - 20 * rmse, T2 @ T1 @ T0, fit, rmse)
        _, Tt, fit, rmse = best
        reg["fit"] = round(float(fit), 3)
        if fit > TRACK_MIN_FIT and np.linalg.norm(Tt[:3, 3]) > 0.001:
            V = V @ Tt[:3, :3].T + Tt[:3, 3]
            reg["mesh"] = (V, T, C)
            reg["pose"] = Tt @ reg["pose"]
        return reg["mesh"]

    def _commit(self, out, info):
        """[ours] An object seen REG_STABLE rebuilds in a row with the same label (or unlabelled for longer) is committed:
        its completed mesh and label are frozen; from then on only its pose is tracked (fast path above)."""
        now = time.time()
        for blob, o in zip(out, info):
            if "_obs_c" in o and o["id"] in self.registry:
                self.registry[o["id"]]["obs_c"] = o["_obs_c"]
            if o.get("committed"):
                self.registry[o["id"]]["last"] = now
                continue
            reg = self.registry.setdefault(o["id"], {"seen": 0, "labels": []})
            reg["obs_c"] = o.get("_obs_c", np.array(o["centre"]))
            reg["seen"] += 1
            reg["last"] = now
            lab = o.get("label")
            reg["labels"] = (reg["labels"] + [lab["en"] if lab else None])[-REG_STABLE:]
            same = len(reg["labels"]) == REG_STABLE and len(set(reg["labels"])) == 1
            if (same and reg["labels"][0] is not None and lab["conf"] >= PRIOR_CONF) or reg["seen"] >= REG_UNLABELLED:
                off = 4
                _, _, _, off = _blob_arrays(blob, off)          # raw TSDF mesh
                V, T, C, _ = _blob_arrays(blob, off)            # completed mesh
                o2 = {k: v for k, v in o.items() if not k.startswith("_")}
                reg.update(committed=True, mesh=(V, T, C), pose=np.eye(4), cov=o.get("coverage_deg") or 0,
                           info={**o2, "committed": True}, committed_at=time.strftime("%H:%M:%S"))
                o["committed"] = True
        for oid in [k for k, r in self.registry.items() if now - r["last"] > REG_TTL_S]:
            self.registry.pop(oid)                          # gone from the scene for a while

    def _merge_instances(self, out, info, geo_pts, geo_mesh, obj_pts, fy):
        """[ours] Geometry from the multi-view TSDF, identity and splits from the masks (detect.py instances):
          one instance covers a geometric object      → keep the geometric mesh, take the instance's label
          two or more instances cover one             → cut the geometric mesh between them: every triangle goes to the
                                                        nearest instance (instance meshes alone came out incomplete —
                                                        carrot 42 %, knife 12 %: an instance only sees its detections)
          an instance with no geometric object        → add it (thinner than the depth noise, e.g. a knife blade)
        (measured on the virtual kitchen: instance meshes alone lost 10–60 points of completeness — an instance only
        sees the frames it was detected in)."""
        from scipy.spatial import cKDTree
        insts = self.instances()
        if not insts:
            return out, info
        trees = [cKDTree(i["pts"]) for i in insts]
        cover = {}                                         # geometric index → instances covering it
        used = set()
        for k, G in enumerate(geo_pts):
            for j, tr in enumerate(trees):
                b = float((tr.query(G, distance_upper_bound=0.012)[0] < np.inf).mean())        # share of the object
                a = float((cKDTree(G).query(insts[j]["pts"], distance_upper_bound=0.012)[0] < np.inf).mean())
                if b >= INST_COVER or a >= 0.6:
                    cover.setdefault(k, []).append((b, a, j))
                    used.add(j)
        keep, add, parts_out, parts_info = [], [], [], []
        # one instance owning several geometric pieces (a pot at the edge of the view split in two): merge them first
        owner_of = {}
        for k in range(len(info)):
            own = [x for x in cover.get(k, []) if x[1] >= INST_OWN]
            if len(own) == 1:
                owner_of.setdefault(own[0][2], []).append(k)
        for j, ks in owner_of.items():
            if len(ks) < 2:
                continue
            Vs, Ts, Cs, n0 = [], [], [], 0
            for k in ks:
                pv, pt, pc, sup, _ = geo_mesh[k]
                Vs.append(pv); Cs.append(pc); Ts.append(pt + n0); n0 += len(pv)
            sup = min(geo_mesh[k][3] for k in ks)
            pv, pt, pc = np.vstack(Vs), np.vstack(Ts), np.vstack(Cs)
            comp = complete_object(pv, pc, sup)
            comp["V"], comp["T"], comp["C"] = self._fuse(pv, pt, pc, comp)
            k0 = ks[0]
            geo_mesh[k0] = (pv, pt, pc, sup, (comp["V"], comp["T"]))
            out[k0] = out[k0][:4] + _mesh_blob(pv, pt, pc) + _mesh_blob(comp["V"], comp["T"], comp["C"])
            lo, hi = pv.min(axis=0), pv.max(axis=0)
            info[k0].update(_lo=lo, _hi=hi, size_mm=(np.ptp(pv, axis=0) * 1000).round(0).tolist(),
                            centre=((lo + hi) / 2).round(3).tolist(), shape=comp["shape"], dims_mm=comp["dims_mm"],
                            rms_mm=comp["rms_mm"], verts=int(len(pv)))
            for k in ks[1:]:
                cover[k] = []                              # absorbed into ks[0]
                info[k]["_absorbed"] = True
        for k in range(len(info)):
            if info[k].get("_absorbed"):
                continue
            if info[k].get("committed"):
                reg = self.registry.get(info[k]["id"])
                own = [x for x in cover.get(k, []) if x[1] >= INST_OWN]
                mdl = insts[own[0][2]].get("model") if own else None
                if reg is not None and mdl is not None and insts[own[0][2]].get("use_model", True) and not reg.get("ai"):
                    V, F, C, st = mdl                       # the AI model asked for replaces the frozen mesh
                    pv, pt, pc, sup, _ = geo_mesh[k]
                    V, F, C = self._fuse(pv, pt, pc, {"V": V, "T": F, "C": C})   # as the completion (see below)
                    reg.update(mesh=(V, F, C), ai=True)
                    reg["info"].update(shape="ai", dims_mm=(np.ptp(V, axis=0) * 1000).round(0).astype(int).tolist(), model=st)
                    out[k] = out[k][:4] + _mesh_blob(pv, pt, pc) + _mesh_blob(V, F, C)
                    info[k].update(shape="ai", dims_mm=reg["info"]["dims_mm"], model=st)
                keep.append(k)
                continue
            js = cover.get(k, [])
            # owners: instances most of whose points lie in this object (a board's instance also covers the carrot on
            # it, but most of the board's points are on the board → not the carrot's owner)
            own = [x for x in js if x[1] >= INST_OWN]
            if len(own) >= 2:                              # two objects merged into one geometric piece: cut it
                po, pi_ = self._cut_by_instances(geo_mesh[k], [insts[j] for _, _, j in sorted(own, reverse=True)], fy, obj_pts)
                parts_out += po
                parts_info += pi_
                continue
            if own:
                j = own[0][2]
                info[k]["label"] = lab = self.labeler(1000 + insts[j]["id"]) if self.labeler else None
                info[k]["inst"] = 1000 + insts[j]["id"]
                mdl = insts[j].get("model")
                if mdl is not None:                          # AI model (TripoSR, aligned) vs the geometric completion
                    from objmodel import track
                    V, F, C, st = mdl
                    Tt, fit, rmse = track(V, geo_mesh[k][0])   # [ours] 6-DoF follow-up: ICP to what is seen now
                    if fit > 0.6 and np.linalg.norm(Tt[:3, 3]) > 0.002:
                        V = V @ Tt[:3, :3].T + Tt[:3, 3]
                        insts[j]["model"] = (V, F, C, st)
                    pv, pt, pc, sup, (gV, gT) = geo_mesh[k]
                    s_ai, s_geo = _consistency(V, F, pv, self._carve_kfs), _consistency(gV, gT, pv, self._carve_kfs)
                    info[k]["model_vs_geo"] = [round(s_ai * 1000, 2), round(s_geo * 1000, 2)]
                    if not insts[j].get("use_model", True):    # shown only when asked for (button toggles it)
                        mdl = None
                if mdl is not None:
                    # [ours] the AI model fills only what was not seen: fused with the observed surface, cut where the
                    # views saw through it (measured, virtual pot / mug: AI mesh alone 10.2 / 4.7 mm mean error — one
                    # occluded oblique view gives wrong proportions; as the completion 2.6 / 3.1 mm). Fused once per
                    # model, then moved with it.
                    fz = insts[j].get("fused")
                    if fz is None or fz[0] is not mdl[3]:
                        fz = (mdl[3], V.copy(), *self._fuse(pv, pt, pc, {"V": V, "T": F, "C": C}))
                        insts[j]["fused"] = fz
                    elif np.abs(V - fz[1]).max() > 1e-6:      # moved by the tracking since it was fused
                        Tm = _rigid(fz[1], V)
                        fz = (fz[0], V.copy(), fz[2] @ Tm[:3, :3].T + Tm[:3, 3], fz[3], fz[4])
                        insts[j]["fused"] = fz
                    V, F, C = fz[2], fz[3], fz[4]
                    out[k] = out[k][:4] + _mesh_blob(pv, pt, pc) + _mesh_blob(V, F, C)
                    info[k].update(shape="ai", dims_mm=(np.ptp(V, axis=0) * 1000).round(0).astype(int).tolist(),
                                   rms_mm=st.get("rmse_mm"), model=st)
                    keep.append(k)
                    continue
                if lab and lab["en"] in CONTAINERS and lab["conf"] >= PRIOR_CONF:   # [ours] category prior: open shell
                    pv, pt, pc, sup, _ = geo_mesh[k]
                    comp = complete_container(pv, pc, sup)
                    if comp["plausible"]:                    # a wrong label must not force a wrong shape
                        Vc, Tc, Cc = self._fuse(pv, pt, pc, comp)
                        out[k] = out[k][:4] + _mesh_blob(pv, pt, pc) + _mesh_blob(Vc, Tc, Cc)
                        info[k].update(shape="container", dims_mm=comp["dims_mm"], rms_mm=comp["rms_mm"], fits_mm={})
            keep.append(k)
        add += [j for j in range(len(insts)) if j not in used and len(insts[j]["obs"]) >= INST_ONLY_MIN_OBS]
        new_out, new_info = [], []
        for j in dict.fromkeys(add):
            inst = insts[j]
            V, Tr, C = inst["mesh"]
            Vc, Tc, Cc, comp = inst["comp"]
            oid = 1000 + inst["id"]
            lo, hi = V.min(axis=0), V.max(axis=0)
            sup = inst["support"]
            new_out.append(np.array([oid], np.uint32).tobytes() + _mesh_blob(V, Tr, C) + _mesh_blob(Vc, Tc, Cc))
            new_info.append({"_lo": lo.copy(), "_hi": hi.copy(), "id": oid, "size_mm": (np.ptp(V, axis=0) * 1000).round(0).tolist(),
                             "centre": ((lo + hi) / 2).round(3).tolist(), "verts": int(len(V)), "ms": 0,
                             "shape": comp["shape"], "dims_mm": comp["dims_mm"], "rms_mm": comp["rms_mm"],
                             "fits_mm": comp["fits_mm"], "complete_ms": comp["ms"], "support_mm": round((sup - fy) * 1000),
                             "_on_floor": abs(sup - fy) < 0.003, "source": "인식 인스턴스",
                             "label": self.labeler(oid) if self.labeler else None})
            obj_pts[oid] = V[np.random.default_rng(0).choice(len(V), min(len(V), 2000), replace=False)]
        for k in range(len(info)):
            if k not in keep:
                obj_pts.pop(info[k]["id"], None)
        return [out[k] for k in keep] + parts_out + new_out, [info[k] for k in keep] + parts_info + new_info

    def _cut_by_instances(self, gm, insts, fy, obj_pts):
        """Split one geometric object between the instances covering it; insts[0] covers most of it (the support)."""
        from scipy.spatial import cKDTree
        pv, pt, pc, sup0, _ = gm
        cen = pv[pt].mean(axis=1)
        D = np.stack([cKDTree(i["pts"]).query(cen)[0] for i in insts], axis=1)
        owner = np.argmin(D, axis=1)
        owner[D.min(axis=1) > 0.02] = 0                    # far from every mask: the main (supporting) instance
        out, info = [], []
        for j, inst in enumerate(insts):
            T = pt[owner == j]
            if len(T) < SPLIT_MIN_TRIS:
                continue
            used, inv = np.unique(T, return_inverse=True)
            V, C, Tj = pv[used], pc[used], inv.reshape(-1, 3)
            bottom = float(V[:, 1].min())
            sup = sup0 if j == 0 or bottom - sup0 < 0.004 else bottom   # what stands on the support starts on it
            comp = complete_object(V, C, sup)
            if COMPLETE_MODE in ("hybrid", "carve", "auto"):
                comp["V"], comp["T"], comp["C"] = self._fuse(V, Tj, C, comp)
            oid = 1000 + inst["id"]
            lo, hi = V.min(axis=0), V.max(axis=0)
            out.append(np.array([oid], np.uint32).tobytes() + _mesh_blob(V, Tj, C) + _mesh_blob(comp["V"], comp["T"], comp["C"]))
            info.append({"_lo": lo.copy(), "_hi": hi.copy(), "id": oid, "size_mm": (np.ptp(V, axis=0) * 1000).round(0).tolist(),
                         "centre": ((lo + hi) / 2).round(3).tolist(), "verts": int(len(V)), "ms": 0,
                         "shape": comp["shape"], "dims_mm": comp["dims_mm"], "rms_mm": comp["rms_mm"],
                         "fits_mm": comp["fits_mm"], "complete_ms": comp["ms"], "support_mm": round((sup - fy) * 1000),
                         "_on_floor": abs(sup - fy) < 0.003, "source": "기하 메시 + 마스크 분할",
                         "label": self.labeler(oid) if self.labeler else None})
            obj_pts[oid] = V[np.random.default_rng(0).choice(len(V), min(len(V), 2000), replace=False)]
        return out, info

    @staticmethod
    def _split_on_support(v, tri, tri_c, fy):
        """[ours] One floor piece → [(triangles, support height, cap)]. When it holds a large flat upward surface below its
        top (a cutting board under a knife and a carrot), what rises more than SUPPORT_ABOVE_M above that surface is
        split into its own objects standing on it; the rest is the support object itself, capped at its surface."""
        P = v[tri[tri_c]]
        nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        area = np.linalg.norm(nrm, axis=1) / 2
        ny = nrm[:, 1] / np.maximum(2 * area, 1e-12)
        cy = P[:, :, 1].mean(axis=1)
        up = ny > 0.9
        top = float(P[:, :, 1].max())
        if up.sum() < 20:
            return [(tri_c, fy, None)]
        bins = np.arange(fy, top + 0.005, 0.005)
        if len(bins) < 3:
            return [(tri_c, fy, None)]
        hist, _ = np.histogram(cy[up], bins, weights=area[up])
        k = int(np.argmax(hist))
        hp = float(np.median(cy[up & (np.abs(cy - (bins[k] + bins[k + 1]) / 2) < 0.005)]))
        if hist[k] < SUPPORT_MIN_AREA_M2 or hist[k] < SUPPORT_SHARE * area[up].sum() or top - hp < SUPPORT_MIN_RISE_M:
            return [(tri_c, fy, None)]                   # no shelf in it: one object
        above = (P[:, :, 1] > hp + SUPPORT_ABOVE_M).all(axis=1)
        if above.sum() < SPLIT_MIN_TRIS:
            return [(tri_c, fy, None)]
        sub = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(tri[tri_c[above]].astype(np.int32)))
        ids, cnt, _ = sub.cluster_connected_triangles()
        ids, cnt = np.asarray(ids), np.asarray(cnt)
        out, used = [], np.zeros(len(tri_c), bool)
        idx_above = np.nonzero(above)[0]
        plane = up & (np.abs(cy - hp) < 0.005)
        pxz = P[plane][:, :, [0, 2]].reshape(-1, 2)
        p_area = float(np.prod(np.ptp(pxz, axis=0)))
        for c in np.nonzero(cnt >= SPLIT_MIN_TRIS // 2)[0]:
            sel = idx_above[ids == c]
            cxz = P[sel][:, :, [0, 2]].reshape(-1, 2)
            if np.prod(np.ptp(cxz, axis=0)) > SUPPORT_ENCLOSE * p_area:
                return [(tri_c, fy, None)]               # it surrounds the "support": a pot's wall around its bottom
            out.append((tri_c[sel], hp, None))
            used[sel] = True
        if not out:
            return [(tri_c, fy, None)]
        return [(tri_c[~used], fy, hp + 0.004)] + out

    def object_tsdf(self, lo, hi, voxel=OBJ_VOXEL_M, n_kf=OBJ_KEYFRAMES, arrays=False, core=None):
        """[ours] Per-object TSDF: only depth pixels whose 3-D point falls inside the box lo..hi (ARKit world, m)
        are integrated, at a finer voxel than the room. Returns the same blob layout as mesh_bytes() and stats.
        core (lo, hi): the piece the box was drawn around — every connected part centred inside it (top view) is kept,
        not just the largest (a strawberry half over the board's edge touched the board only at its bottom: the rest,
        a separate part, was dropped). Parts centred in the margin are a neighbour's edge and go."""
        if o3d is None:
            raise RuntimeError(self.status["error"])
        t0 = time.time()
        with self._lock:
            kfs = list(self._kf[-n_kf:])
        if not kfs:
            raise RuntimeError("키프레임이 없습니다")
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=8, block_count=4000,
            device=o3c.Device("CPU:0"))
        used = 0
        for k in kfs:
            dep, K, T = k["depth"], k["K"], k["T"]
            h, w = dep.shape
            vv, uu = np.mgrid[0:h, 0:w]
            z = dep
            P = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], -1) @ T[:3, :3].T + T[:3, 3]
            inside = (z > 0.05) & np.all((P >= lo) & (P <= hi), axis=-1)
            if inside.sum() < 30:
                continue
            used += 1
            d = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(np.where(inside, dep, 0), np.float32)))
            c = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(k["rgb"], np.float32)))
            Kt, Et = o3c.Tensor(K), o3c.Tensor(np.linalg.inv(T))
            blk = vbg.compute_unique_block_coordinates(d, Kt, Et, 1.0, DEPTH_MAX_M, trunc_voxel_multiplier=4.0)
            vbg.integrate(blk, d, c, Kt, Et, 1.0, DEPTH_MAX_M, trunc_voxel_multiplier=4.0)
        if not used:
            raise RuntimeError("물체를 본 키프레임이 없습니다")
        m = vbg.extract_triangle_mesh(weight_threshold=2.0).to_legacy()
        v = np.asarray(m.vertices)
        if len(v):                                        # the box edges cut the truncation band: keep inside only
            out = np.any((v < lo) | (v > hi), axis=1)
            m.remove_vertices_by_mask(out)
        if len(m.triangles):                              # largest connected piece = the object (+ parts in the core)
            ids, cnt, _ = m.cluster_connected_triangles()
            ids, cnt = np.asarray(ids), np.asarray(cnt)
            keep = np.zeros(len(cnt), bool)
            keep[int(np.argmax(cnt))] = True
            if core is not None:
                cv, tv = np.asarray(m.vertices), np.asarray(m.triangles)
                clo, chi = np.asarray(core[0], float), np.asarray(core[1], float)
                for c in np.nonzero(cnt >= 20)[0]:
                    q = cv[tv[ids == c]].reshape(-1, 3)
                    mc = (q.min(0) + q.max(0)) / 2
                    keep[c] |= bool((mc[[0, 2]] >= clo[[0, 2]]).all() and (mc[[0, 2]] <= chi[[0, 2]]).all())
            m.remove_triangles_by_mask(~keep[ids])
            m.remove_unreferenced_vertices()
        m.compute_vertex_normals()
        v = np.asarray(m.vertices, np.float32)
        n = np.asarray(m.vertex_normals, np.float32)
        colv = np.asarray(m.vertex_colors)
        col = np.clip(colv * (255.0 if colv.max(initial=0) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        tri = np.asarray(m.triangles, np.uint32)
        blob = (np.array([len(v), len(tri)], np.uint32).tobytes() + v.tobytes() + n.tobytes() + tri.tobytes()
                + col.tobytes())
        stats = {"verts": int(len(v)), "tris": int(len(tri)), "keyframes": used, "voxel_mm": voxel * 1000,
                 "ms": round((time.time() - t0) * 1000)}
        return blob, stats, ((v.astype(np.float64), tri.astype(np.int64), col.astype(float)) if arrays else v)

    @staticmethod
    def _visibility_keep(v, recent):
        """Returns (keep mask, newest rgb 0..1 or −1). Newest observation wins: a vertex is removed when the newest keyframes that actually observe it
        (≤ 2, newest first) all see *through* it. "Through" uses the 3×3 minimum depth, so a vertex next to a depth
        edge is not carved by the background pixel beside it; a vertex hidden behind something nearer is not an
        observation at all."""
        if len(v) == 0 or not recent:
            return np.ones(len(v), bool), np.full((len(v), 3), -1.0)
        import cv2
        seen = np.zeros(len(v), np.int32)
        contra = np.zeros(len(v), np.int32)
        rgb = np.full((len(v), 3), -1.0)
        for k in reversed(recent):                       # newest first
            todo = seen < 2
            if not todo.any():
                break
            Tcw = np.linalg.inv(k["T"])
            pc = v @ Tcw[:3, :3].T + Tcw[:3, 3]
            z = pc[:, 2]
            front = z > 0.05
            K = k["K"]
            zs = np.where(front, z, 1)
            u = np.round(K[0, 0] * pc[:, 0] / zs + K[0, 2]).astype(np.int64)
            w = np.round(K[1, 1] * pc[:, 1] / zs + K[1, 2]).astype(np.int64)
            dep = k["depth"]
            h, wd = dep.shape
            inside = front & (u >= 0) & (u < wd) & (w >= 0) & (w < h)
            dmin = cv2.erode(np.where(dep > 0, dep, 1e3).astype(np.float32), np.ones((3, 3), np.uint8))
            d = np.zeros(len(v))
            dm = np.zeros(len(v))
            d[inside] = dep[w[inside], u[inside]]
            dm[inside] = dmin[w[inside], u[inside]]
            valid = inside & (d > 0)
            through = valid & (dm > z + CARVE_TAU_M) & (dm < 1e2)
            on = valid & (np.abs(d - z) <= CARVE_TAU_M)
            obs = todo & (through | on)                  # occluded (d < z − τ) is not an observation
            contra += obs & through
            seen += obs
            paint = valid & on & (rgb[:, 0] < 0)
            if paint.any() and k["rgb"].shape[:2] == dep.shape:
                rgb[paint] = k["rgb"][w[paint], u[paint]] / 255.0
        return ~((seen >= 1) & (contra == seen)), rgb
