"""Object recognition → per-instance TSDF maps (panoptic multi-TSDF style), lightweight, ~1 Hz.

Pipeline (Panoptic Mapping / ConceptGraphs style, simplified for one CPU + MPS):
  ① recognise  [external]  keyframe RGB → YOLOE instance masks + class + score (text prompts from vocab.json, editable)
  ② depth      [external, optional]  Prompt Depth Anything (LiDAR as prompt) → depth at the image resolution, then an
                           affine fit to the LiDAR (scale bias measured at 1.5–1.8 %); off = the LiDAR depth itself
  ③ lift       [ours]      mask (eroded) ∩ depth → 3-D points per detection (ARKit world)
  ④ associate  [ours]      detection ↔ instance by F1 of the two coverage ratios (≤ ASSOC_DIST_M); none ≥ NEW_MIN → new
                           instance. Class = score-weighted vote with decay
  ⑤ map        [ours]      every instance keeps its last OBS_MAX masked depth images and fuses only those pixels into
                           its own 3 mm TSDF (open3d VoxelBlockGrid) → mesh → hybrid completion (camera/complete.py).
                           A knife blade or two touching objects are separate because the masks are, not the geometry.
Output: instances() for recon.py (which replaces the geometric objects they cover), status for the UI.
"""
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "models"
VOCAB_FILE = HERE / "vocab.json"
PERIOD_S = 1.0
CONF_MIN = 0.15
ASSOC_DIST_M = 0.012
MERGE_F1 = 0.5               # instances overlapping this much are the same object
NEW_MIN = 0.2                 # F1 below this with every instance → a new instance
DECAY = 0.95
OBS_MAX = 30                  # 12 left a carrot at 43 % completeness (measured): an instance sees only its detections
INST_VOXEL_M = 0.003
REBUILD_MIN_S = 2.0
INST_TTL_S = 120.0
AUTO_MODEL = False            # automatic choice AI vs geometry picked the wrong one for apple/strawberry (measured): on request only
MODEL_MIN_OBS, MODEL_MIN_CONF = 10, 0.5
DEFAULT_VOCAB = [  # en (prompt) → ko (display)
    ["wooden cutting board", "도마"], ["knife", "칼"], ["carrot", "당근"], ["strawberry", "딸기"], ["coffee mug", "머그"],
    ["cup", "컵"], ["banana", "바나나"], ["apple", "사과"], ["cooking pot", "냄비"], ["saucepan", "냄비"], ["bowl", "그릇"],
    ["plate", "접시"], ["spoon", "숟가락"], ["fork", "포크"], ["chopsticks", "젓가락"],
    ["tomato", "토마토"], ["onion", "양파"], ["potato", "감자"], ["cucumber", "오이"], ["frying pan", "프라이팬"],
    ["lemon", "레몬"], ["orange", "오렌지"], ["egg", "달걀"], ["sponge", "스펀지"],
    # generic words ("box", "bottle", "can") pulled the pot / mug / apple to themselves in the kitchen test: left out
]


class Detector:
    def __init__(self, recon):
        self.recon = recon
        self.status = {"available": False, "running": False, "mode": None, "model": None, "ms": 0, "depth_ms": 0,
                       "frames": 0, "last": [], "instances": [], "depth_mode": "off", "error": None}
        self._latest = None
        self._lock = threading.Lock()
        self._inst = []
        self._next_id = 1
        self._model = None
        self._pda = None
        try:
            self.vocab = dict(json.loads(VOCAB_FILE.read_text()))
        except Exception:
            self.vocab = dict(DEFAULT_VOCAB)
        self.status["vocab_list"] = list(self.vocab.items())
        self._model_queue = []
        self.virtual = None                                # set by stream.py: re-render a sharp still in the virtual scene
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self._models, daemon=True).start()

    # ---- inputs (camera thread)
    def submit(self, bgr, depth, K, T_world_cvcam):
        """bgr: colour image, depth: LiDAR H×W m at the resolution of K, T: camera→world (OpenCV axes)."""
        if self.status["running"]:
            with self._lock:
                self._latest = (bgr, depth.copy(), np.asarray(K, float), np.asarray(T_world_cvcam, float))

    def handle(self, c):
        t = c["type"]
        if t == "detect_start":
            self.status["running"] = True
        elif t == "detect_stop":
            self.status["running"] = False
        elif t == "detect_reset":
            with self._lock:
                self._inst = []
            self.status.update(instances=[], last=[])
        elif t == "detect_vocab":                          # [[en, ko], ...] — any words, re-embedded on next frame
            self.vocab = {str(e).strip(): str(k).strip() or str(e).strip() for e, k in c["vocab"] if str(e).strip()}
            VOCAB_FILE.write_text(json.dumps(list(self.vocab.items()), ensure_ascii=False, indent=1))
            self.status["vocab_list"] = list(self.vocab.items())
            self._model = None
        elif t == "detect_model":                          # build (or toggle) the AI model of one instance
            iid = int(c["id"]) - 1000
            for i in self._inst:
                if i["id"] == iid and "model" in i:
                    i["use_model"] = not i.get("use_model", True)
                    return
            self._model_queue.append(iid)
        elif t == "detect_depth":                          # "off" | "small" | "large"
            self.status["depth_mode"] = c["mode"]
            self._pda = None

    # ---- outputs
    def label_of(self, inst):
        v = inst["votes"]
        cls, s = max(v.items(), key=lambda kv: kv[1])
        return {"name": self.vocab.get(cls, cls), "en": cls, "conf": round(s / sum(v.values()), 2), "score": round(sum(v.values()), 2)}

    def instances(self):
        """For recon.py: instances that have a mesh."""
        with self._lock:
            return [dict(i) for i in self._inst if i.get("mesh") is not None]

    def label(self, oid):                                  # recon's labeler hook: instance ids are 1000 + id
        for i in self._inst:
            if 1000 + i["id"] == oid:
                return self.label_of(i)
        return None

    # ---- models
    def _load(self):
        """Text prompts (vocab.json, any words). The prompt-free model's built-in 4585-word vocabulary named the virtual
        kitchen 'cake', 'studio shot', 'scale model' (measured), so text prompts are the default."""
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # never pip-install behind the server's back
        import torch
        from ultralytics import YOLOE
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        tp = MODELS / "yoloe-11s-seg.pt"
        m = YOLOE(str(tp))
        names = list(self.vocab)
        cache = MODELS / "vocab_pe.pt"                    # the text encoder takes ~1 min on first use: cache the embeddings
        pe = None
        if cache.exists():
            d = torch.load(cache)
            pe = d["pe"] if d.get("names") == names else None
        if pe is None:
            cwd = os.getcwd()
            os.chdir(MODELS)                              # the text encoder (mobileclip_blt.ts) lives in models/
            try:
                pe = m.get_text_pe(names)
            finally:
                os.chdir(cwd)
            torch.save({"names": names, "pe": pe}, cache)
        m.set_classes(names, pe)
        self.status.update(mode="text", model=tp.name, vocab_size=len(names))
        self._model = m
        self.status["available"] = True

    def _load_pda(self):
        import sys
        sys.path.insert(0, str(ROOT / "third_party" / "PromptDA"))
        from promptda.promptda import PromptDA
        enc = {"small": "vits", "large": "vitl"}[self.status["depth_mode"]]
        self._pda = PromptDA.from_pretrained(f"depth-anything/prompt-depth-anything-{enc}",
                                             model_kwargs={"encoder": enc}).to(self._device).eval()

    def _depth(self, bgr, lidar, K):
        """→ (depth, K) used for the masks: PromptDA at the image resolution aligned to the LiDAR, or the LiDAR."""
        if self.status["depth_mode"] == "off":
            return lidar, K
        import cv2
        import torch
        if self._pda is None:
            self._load_pda()
        h = min(bgr.shape[0], 756) // 14 * 14
        w = int(round(bgr.shape[1] * h / bgr.shape[0])) // 14 * 14
        img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
        t0 = time.time()
        with torch.no_grad():
            out = self._pda.predict(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].to(self._device),
                                    torch.from_numpy(lidar.astype(np.float32))[None, None].to(self._device))[0, 0].float().cpu().numpy()
        small = cv2.resize(out, (lidar.shape[1], lidar.shape[0]), interpolation=cv2.INTER_AREA)
        m = lidar > 0.05
        if m.sum() > 100:                                 # affine alignment to the LiDAR (robust: refit on the best 80 %)
            a, b = np.polyfit(small[m], lidar[m], 1)
            r = np.abs(a * small[m] + b - lidar[m])
            k = r < np.percentile(r, 80)
            a, b = np.polyfit(small[m][k], lidar[m][k], 1)
            out = a * out + b
        self.status["depth_ms"] = round((time.time() - t0) * 1000)
        S = np.diag([w / lidar.shape[1], h / lidar.shape[0], 1.0])
        return out.astype(np.float32), S @ K

    # ---- worker
    def _run(self):
        while True:
            time.sleep(PERIOD_S)
            if not self.status["running"]:
                continue
            with self._lock:
                item, self._latest = self._latest, None
            if item is None:
                continue
            try:
                if self._model is None:
                    self._load()
                self._process(*item)
                self._rebuild_due()
                self.status["error"] = None
            except Exception as e:
                self.status["error"] = f"{type(e).__name__}: {e}"

    def _process(self, bgr, lidar, K, T):
        import cv2
        K, T = np.asarray(K, float), np.asarray(T, float)
        t0 = time.time()
        r = self._model.predict(bgr, conf=CONF_MIN, verbose=False, device=self._device)[0]
        self.status["ms"] = round((time.time() - t0) * 1000)
        self.status["frames"] += 1
        depth, Kd = self._depth(bgr, lidar, K)
        h, w = depth.shape
        sx, sy = w / bgr.shape[1], h / bgr.shape[0]
        rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        names = self._model.names
        dets = []
        if r.masks is not None:
            er = np.ones((3, 3), np.uint8) if w <= 320 else np.ones((5, 5), np.uint8)
            raw = []
            for poly, c, s in zip(r.masks.xy, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if len(poly) < 3:
                    continue
                mask = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mask, [np.round(poly * [sx, sy]).astype(np.int32)], 1)
                raw.append((int(mask.sum()), mask, c, s))
            # panoptic rule: a pixel belongs to the smallest mask containing it, so a cutting board's mask loses the
            # knife and carrot lying on it (measured: the board instance otherwise came out 53 mm tall)
            taken = np.zeros((h, w), bool)
            items = []
            for _, mask, c, s in sorted(raw, key=lambda x: x[0]):
                m = mask.astype(bool) & ~taken
                taken |= mask.astype(bool)
                items.append((m, c, s))
            for m, c, s in items:
                mask = cv2.erode(m.astype(np.uint8), er).astype(bool) & (depth > 0.05)   # edges straddle depth edges
                v, u = np.nonzero(mask)
                if len(v) < 15:
                    continue
                z = depth[v, u]
                pc = np.c_[(u - Kd[0, 2]) / Kd[0, 0] * z, (v - Kd[1, 2]) / Kd[1, 1] * z, z]
                dets.append({"cls": names[int(c)], "score": float(s), "pts": pc @ T[:3, :3].T + T[:3, 3],
                             "obs": (np.where(mask, depth, 0).astype(np.float32), rgb, Kd, T),
                             "view": (int(m.sum()), bgr, np.diag([bgr.shape[1] / lidar.shape[1], bgr.shape[0] / lidar.shape[0], 1.0]) @ K, T)})
        self._associate(dets)
        self.status["last"] = [{"name": self.vocab.get(d["cls"], d["cls"]), "en": d["cls"], "score": round(d["score"], 2),
                                "points": int(len(d["pts"])), "matched": d.get("inst")} for d in dets]

    # ---- association
    def _associate(self, dets):
        from scipy.spatial import cKDTree
        now = time.time()
        with self._lock:
            insts = list(self._inst)
        trees = [cKDTree(i["pts"]) for i in insts]
        for i in insts:
            for k in i["votes"]:
                i["votes"][k] *= DECAY
        taken = set()
        for d in sorted(dets, key=lambda d: -d["score"]):
            P = d["pts"]
            dt_ = cKDTree(P)
            best, best_f = None, 0.0
            for j, (inst, tr) in enumerate(zip(insts, trees)):
                if j in taken:
                    continue
                a = float((tr.query(P, distance_upper_bound=ASSOC_DIST_M)[0] < np.inf).mean())
                b = float((dt_.query(inst["pts"], distance_upper_bound=ASSOC_DIST_M)[0] < np.inf).mean())
                f = 2 * a * b / (a + b) if a + b else 0.0
                f *= 1.0 if d["cls"] in inst["votes"] else 0.8          # same class: a little preferred
                if f > best_f:
                    best, best_f = j, f
            if best is None or best_f < NEW_MIN:
                inst = {"id": self._next_id, "votes": {}, "obs": deque(maxlen=OBS_MAX), "pts": _grid(P), "mesh": None,
                        "last": now, "built": 0.0, "dirty": True}
                self._next_id += 1
                insts.append(inst)
                trees.append(cKDTree(inst["pts"]))
                best = len(insts) - 1
            taken.add(best)
            inst = insts[best]
            inst["votes"][d["cls"]] = inst["votes"].get(d["cls"], 0.0) + d["score"]
            inst["obs"].append(d["obs"])
            if d["view"][0] > inst.get("view", (0,))[0]:          # the frame that shows it largest: input for ③
                inst["view"] = d["view"]
            inst["pts"] = _grid(np.vstack([inst["pts"], P]))[-6000:]
            inst["last"], inst["dirty"] = now, True
            d["inst"] = 1000 + inst["id"]
        insts = self._merge_overlapping([i for i in insts if now - i["last"] < INST_TTL_S])
        with self._lock:
            self._inst = insts
        self.status["instances"] = [{"id": 1000 + i["id"], **self.label_of(i), "obs": len(i["obs"]),
                                     "verts": int(len(i["mesh"][0])) if i.get("mesh") else 0} for i in insts]

    @staticmethod
    def _merge_overlapping(insts):
        """[ours, ConceptGraphs' merge step] two instances occupying the same space are one object seen under two names
        (the cutting board was tracked as 'cutting board' and 'sponge' and got cut in half): merge, add up the votes."""
        from scipy.spatial import cKDTree
        out = []
        for inst in sorted(insts, key=lambda i: -len(i["pts"])):
            for big in out:
                a = float((cKDTree(big["pts"]).query(inst["pts"], distance_upper_bound=ASSOC_DIST_M)[0] < np.inf).mean())
                b = float((cKDTree(inst["pts"]).query(big["pts"], distance_upper_bound=ASSOC_DIST_M)[0] < np.inf).mean())
                if a + b and 2 * a * b / (a + b) >= MERGE_F1:
                    for k, v in inst["votes"].items():
                        big["votes"][k] = big["votes"].get(k, 0.0) + v
                    big["obs"].extend(inst["obs"])
                    big["pts"] = _grid(np.vstack([big["pts"], inst["pts"]]))[-6000:]
                    big["last"], big["dirty"] = max(big["last"], inst["last"]), True
                    break
            else:
                out.append(inst)
        return out

    # ---- AI object models (generate once, align, then track)
    def _models(self):
        while True:
            time.sleep(1.0)
            if not self.status["running"]:
                continue
            with self._lock:
                insts = list(self._inst)
            want = [i for i in insts if i["id"] in self._model_queue] or \
                   [i for i in insts if AUTO_MODEL and "model" not in i and not i.get("model_failed") and len(i["obs"]) >= MODEL_MIN_OBS
                    and i.get("view") and self.label_of(i)["conf"] >= MODEL_MIN_CONF]
            if not want:
                continue
            inst = want[0]
            if inst["id"] in self._model_queue:
                self._model_queue.remove(inst["id"])
            try:
                self._build_model(inst)
            except Exception as e:
                inst["model_failed"] = f"{type(e).__name__}: {e}"
                self.status["model_error"] = inst["model_failed"]

    def _build_model(self, inst):
        """③ [external TripoSR] + [ours] alignment: best view → mesh → metric pose from the instance's multi-view
        points and the keyframes' free space."""
        import cv2
        import objmodel
        t0 = time.time()
        area, bgr, Kc, T = inst["view"]
        if self.virtual is not None:                       # the virtual colour stream is 640×480 from a 320×240 render:
            k = 1920 / bgr.shape[1]                        # re-render that pose sharp (an iPhone frame is 1920×1440)
            _, bgr = self.virtual._render(T @ np.diag([1.0, -1, -1, 1]), 1920, 1440, Kc[0, 0] * k)
            Kc = np.diag([k, k, 1.0]) @ Kc
        r = self._model.predict(bgr, conf=CONF_MIN, verbose=False, device=self._device)[0]
        if r.masks is None:
            raise RuntimeError("대표 시점에서 마스크 없음")
        # the mask that covers the projection of this instance's points best
        Tcw = np.linalg.inv(T)
        pc = inst["pts"] @ Tcw[:3, :3].T + Tcw[:3, 3]
        pc = pc[pc[:, 2] > 0.05]
        uv = np.c_[Kc[0, 0] * pc[:, 0] / pc[:, 2] + Kc[0, 2], Kc[1, 1] * pc[:, 1] / pc[:, 2] + Kc[1, 2]].astype(int)
        h, w = bgr.shape[:2]
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
        best, best_hit = None, 0.0
        for poly in r.masks.xy:
            mk = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mk, [np.round(poly).astype(np.int32)], 1)
            hit = float(mk[uv[:, 1], uv[:, 0]].mean()) if len(uv) else 0.0
            if hit > best_hit:
                best, best_hit = mk.astype(bool), hit
        if best is None or best_hit < 0.3:
            raise RuntimeError("대표 시점 마스크가 인스턴스와 맞지 않음")
        mesh = objmodel.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), best, device=self._device)
        t1 = time.time()
        kfs = list(getattr(self.recon, "_kf", [])[-16:])
        sup = inst.get("support") or float(inst["pts"][:, 1].min())
        M, st = objmodel.align_best(np.asarray(mesh.vertices, float), inst["pts"], sup, T @ np.diag([1.0, -1, -1, 1]), kfs)
        V = (np.c_[mesh.vertices, np.ones(len(mesh.vertices))] @ M.T)[:, :3]
        C = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(float)
        inst["use_model"] = True
        inst["model"] = (V, np.asarray(mesh.faces, np.int64), C,
                         {**st, "generate_s": round(t1 - t0, 1), "total_s": round(time.time() - t0, 1), "view_px": int(area)})
        self.status["models"] = {1000 + i["id"]: i["model"][3] for i in self._inst if "model" in i}

    # ---- per-instance TSDF
    def _rebuild_due(self):
        now = time.time()
        for inst in list(self._inst):
            if inst["dirty"] and now - inst["built"] >= REBUILD_MIN_S:
                self._build(inst)
                inst["built"], inst["dirty"] = now, False

    def _build(self, inst):
        import open3d as o3d
        import open3d.core as o3c
        from complete import complete_object, hybrid
        obs = list(inst["obs"])
        lo, hi = inst["pts"].min(axis=0) - 0.02, inst["pts"].max(axis=0) + 0.02
        vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
                                           attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                                           attr_channels=((1), (1), (3)), voxel_size=INST_VOXEL_M, block_resolution=8,
                                           block_count=6000, device=o3c.Device("CPU:0"))
        for depth, rgb, K, T in obs:
            d = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth)))
            c = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb, np.float32)))
            Kt, Et = o3c.Tensor(np.asarray(K, np.float64)), o3c.Tensor(np.linalg.inv(T))
            blk = vbg.compute_unique_block_coordinates(d, Kt, Et, 1.0, 2.5, trunc_voxel_multiplier=4.0)
            vbg.integrate(blk, d, c, Kt, Et, 1.0, 2.5, trunc_voxel_multiplier=4.0)
        m = vbg.extract_triangle_mesh(weight_threshold=1.0).to_legacy()
        if len(m.triangles) < 20:
            return
        V = np.asarray(m.vertices)
        m.remove_vertices_by_mask(np.any((V < lo) | (V > hi), axis=1))
        ids, cnt, _ = m.cluster_connected_triangles()
        if len(cnt):
            m.remove_triangles_by_mask(np.asarray(ids) != int(np.argmax(cnt)))
            m.remove_unreferenced_vertices()
        V, Tr = np.asarray(m.vertices, float), np.asarray(m.triangles, np.int64)
        if len(Tr) < 20:
            return
        C = np.asarray(m.vertex_colors, float)
        C = C * (255.0 if C.max(initial=0) <= 1.5 else 1.0)
        fy = getattr(self.recon, "floor_y", None)
        bottom = float(V[:, 1].min())
        sup = fy if fy is not None and bottom - fy < 0.015 else bottom   # resting on the floor, or on what is below it
        comp = complete_object(V, C, sup)
        Vc, Tc, Cc = hybrid(V, Tr, C, comp["V"], comp["T"], comp["C"])
        inst["mesh"] = (V, Tr, C)
        inst["comp"] = (Vc, Tc, Cc, comp)
        inst["support"] = sup


def _grid(P, cell=0.004):
    _, i = np.unique(np.floor(P / cell).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(i)]
