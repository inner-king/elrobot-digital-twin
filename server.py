"""ElRobot arm viewer backend.

Owns the ST3215 serial bus, polls all motors, streams state over a WebSocket,
and executes commands (torque, goal position, calibration) sent by the web UI.

Run:  uv run --with pyserial --with fastapi --with "uvicorn[standard]" python server.py
"""
import asyncio
import glob
import json
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path

import serial
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).parent
ROBOT_DIR = HERE.parent / "norma-core" / "hardware" / "elrobot" / "simulation"
CALIB_DIR = HERE / "calibrations"
ACTIVE_FILE = CALIB_DIR / ".active"
PORT_PATTERNS = ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*", "/dev/cu.SLAB_USBtoUART*")
ROBOT_FILE = HERE / "robot.json"   # {"source": "auto" | "/dev/cu..." | "virtual"}


def list_ports():
    return sorted({p for pat in PORT_PATTERNS for p in glob.glob(pat)})


def find_port():
    """Re-scanned on every (re)connect attempt, so plugging the board in later just works."""
    return os.environ.get("ARM_PORT") or next(iter(list_ports()), None)


class _Reconnect(BaseException):
    """Raised inside the bus loop to switch the robot source (BaseException: skips the per-command error handler)."""


PORT = find_port()
BAUD = 1_000_000
MOTOR_IDS = list(range(1, 9))
HTTP_PORT = int(os.environ.get("ARM_HTTP_PORT", "8765"))
# 0.0.0.0 = reachable from other devices on the LAN (no auth: anyone on the network can drive the arm)
HTTP_HOST = os.environ.get("ARM_HTTP_HOST", "0.0.0.0")

# Per-motor safety caps on Torque_Limit (0..1000 = 0..100%). Motor 8 is a 7.4V servo on a 12V bus.
TORQUE_CAP = {i: 600 for i in MOTOR_IDS} | {8: 250}
CALIB_HOLD_S = 0.1      # a range end only counts once the joint has been still this long...
CALIB_STILL_STEPS = 10  # ...i.e. moved less than this (≈0.9°) within the window
DEFAULT_SETTINGS = {"speed": 400, "accel": 20, "torque_limit": 400, "motor_tl": {}}  # motor_tl: per-motor override

# ---------------------------------------------------------------- protocol
INST_PING, INST_READ, INST_WRITE = 0x01, 0x02, 0x03

# EEPROM / RAM register map (Feetech STS3215)
EEPROM_FIELDS = [  # name, addr, size
    ("fw_major", 0, 1), ("fw_minor", 1, 1), ("model", 3, 2), ("id", 5, 1), ("baud", 6, 1),
    ("return_delay", 7, 1), ("min_angle", 9, 2), ("max_angle", 11, 2), ("max_temp", 13, 1),
    ("max_voltage", 14, 1), ("min_voltage", 15, 1), ("max_torque", 16, 2), ("phase", 18, 1), ("unload", 19, 1),
    ("p", 21, 1), ("d", 22, 1), ("i", 23, 1), ("min_startup", 24, 2), ("cw_dead", 26, 1),
    ("ccw_dead", 27, 1), ("protect_current", 28, 2), ("resolution", 30, 1), ("offset", 31, 2),
    ("mode", 33, 1), ("protect_torque", 34, 1), ("overload_torque", 36, 1),
]
RAM_FIELDS = [("torque", 40, 1), ("accel", 41, 1), ("goal", 42, 2), ("goal_speed", 46, 2),
              ("torque_limit", 48, 2), ("lock", 55, 1)]


def _ck(b):
    return (~sum(b)) & 0xFF


def _sm_decode(v, bit):  # sign-magnitude
    return -(v & ((1 << bit) - 1)) if v & (1 << bit) else v


def _sm_encode(v, bit):
    return (abs(v) & ((1 << bit) - 1)) | ((1 << bit) if v < 0 else 0)


class Bus:
    def __init__(self, port):
        self.ser = serial.Serial(port, BAUD, timeout=0.012)

    def _tx(self, sid, inst, params=b""):
        body = bytes([sid, len(params) + 2, inst]) + bytes(params)
        self.ser.reset_input_buffer()
        self.ser.write(b"\xff\xff" + body + bytes([_ck(body)]))

    def _rx(self):
        d = self.ser.read(6)
        if len(d) < 6 or d[:2] != b"\xff\xff":
            return None
        n = d[3]
        pkt = d + (self.ser.read(n - 2) if n > 2 else b"")
        if len(pkt) != n + 4 or _ck(pkt[2:-1]) != pkt[-1]:
            return None
        return pkt[5:-1]

    def read(self, sid, addr, n, retries=2):
        for _ in range(retries + 1):
            self._tx(sid, INST_READ, [addr, n])
            r = self._rx()
            if r is not None and len(r) == n:
                return r
        return None

    def write(self, sid, addr, data, retries=2):
        for _ in range(retries + 1):
            self._tx(sid, INST_WRITE, [addr] + list(data))
            if self._rx() is not None:
                return True
        return False

    def write_eeprom(self, sid, addr, data):
        ok = self.write(sid, 55, [0])
        ok &= self.write(sid, addr, data)
        time.sleep(0.01)
        ok &= self.write(sid, 55, [1])
        time.sleep(0.01)
        return ok


def u16(b, i=0):
    return b[i] | (b[i + 1] << 8)


# ---------------------------------------------------------------- calibration store
def _calib_path(name):
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_.") or "calibration"
    return CALIB_DIR / f"{safe}.json"


def list_calibs():
    CALIB_DIR.mkdir(exist_ok=True)
    out = []
    for f in sorted(CALIB_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            n = len(json.loads(f.read_text()).get("motors", {}))
        except Exception:
            n = None
        out.append({"name": f.stem, "motors": n, "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))})
    return out


def load_calib(name):
    if not name:
        return {"name": None, "motors": {}}
    try:
        c = json.loads(_calib_path(name).read_text())
        c["name"] = name
        return c
    except Exception:
        return {"name": None, "motors": {}}


def save_calib(c):
    CALIB_DIR.mkdir(exist_ok=True)
    _calib_path(c["name"]).write_text(json.dumps(c, indent=2, ensure_ascii=False))
    ACTIVE_FILE.write_text(c["name"])


def active_name():
    try:
        return ACTIVE_FILE.read_text().strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------- bus worker
class Arm:
    def __init__(self):
        self.cmds = queue.Queue()
        self.lock = threading.Lock()
        self.calib = load_calib(active_name())
        self.new_name = None
        self.settings = json.loads(json.dumps(DEFAULT_SETTINGS))
        try:
            self.source = json.loads(ROBOT_FILE.read_text()).get("source", "auto")
        except Exception:
            self.source = "auto"
        self.state = {"port": PORT, "connected": False, "error": None, "motors": {}, "robot_source": self.source,
                      "ports": list_ports(), "clients": 0,
                      "eeprom": {}, "ram": {}, "calibrating": False, "rec": {},
                      "calib": self.calib, "calib_files": list_calibs(), "calib_raw": "", "settings": self.settings, "log": [],
                      "torque_cap": TORQUE_CAP}
        self.bus = None
        self.rec = {}
        self.publish_calib()  # calibration recording: id -> {last, acc, lo, hi}

    def set_source(self, src):
        """Called straight from the socket handler: works while disconnected (the command queue only runs connected)."""
        if src not in ("auto", "virtual") and src not in list_ports():
            raise RuntimeError(f"포트 {src}가 없습니다")
        self.source = src
        self.state["robot_source"] = src
        ROBOT_FILE.write_text(json.dumps({"source": src}))
        self._switch = True

    def log(self, msg):
        with self.lock:
            self.state["log"] = (self.state["log"] + [f"{time.strftime('%H:%M:%S')} {msg}"])[-60:]

    # ---- reads
    def read_eeprom(self, mid):
        out = {}
        blk = self.bus.read(mid, 0, 40)
        if blk is None:
            return None
        for name, addr, size in EEPROM_FIELDS:
            out[name] = blk[addr] if size == 1 else u16(blk, addr)
        out["offset"] = _sm_decode(out["offset"], 11)
        return out

    def read_ram(self, mid):
        blk = self.bus.read(mid, 40, 16)
        if blk is None:
            return None
        return {name: (blk[a - 40] if s == 1 else u16(blk, a - 40)) for name, a, s in RAM_FIELDS}

    def read_live(self, mid):
        b = self.bus.read(mid, 56, 15)
        if b is None:
            return None
        return {"pos": u16(b, 0), "speed": _sm_decode(u16(b, 2), 15), "load": _sm_decode(u16(b, 4), 10),
                "voltage": b[6] / 10, "temp": b[7], "status": b[9], "moving": b[10],
                "current_ma": round(_sm_decode(u16(b, 13), 15) * 6.5, 1), "t": time.time()}

    # ---- commands
    def arc(self, mid):
        return self.calib["motors"].get(str(mid))

    def cmd_torque(self, mid, on):
        if on:
            if self.state["calibrating"]:
                raise RuntimeError("캘리브레이션 중에는 토크를 켤 수 없습니다")
            live = self.read_live(mid)
            if live is None:
                raise RuntimeError(f"모터 {mid} 응답 없음")
            cap = TORQUE_CAP[mid]
            tl = self.eff_tl(mid)
            # goal := present position first, otherwise the servo snaps to the stale goal register
            self.bus.write(mid, 42, list(live["pos"].to_bytes(2, "little")))
            self.bus.write(mid, 41, [self.settings["accel"]])
            self.bus.write(mid, 46, list(self.settings["speed"].to_bytes(2, "little")))
            self.bus.write(mid, 48, list(tl.to_bytes(2, "little")))
            self.bus.write(mid, 40, [1])
        else:
            self.bus.write(mid, 40, [0])

    def cmd_goal(self, mid, raw):
        a = self.arc(mid)
        if not a:
            raise RuntimeError(f"모터 {mid}는 캘리브레이션 전이라 위치 명령을 막았습니다")
        raw = int(max(a["min"], min(a["max"], raw)))
        self.bus.write(mid, 46, list(self.settings["speed"].to_bytes(2, "little")))
        self.bus.write(mid, 42, list(raw.to_bytes(2, "little")))

    def eff_tl(self, mid):
        v = self.settings["motor_tl"].get(str(mid), self.settings["torque_limit"])
        return int(max(0, min(v, TORQUE_CAP[mid])))

    def apply_settings(self):
        for mid in MOTOR_IDS:
            tl = self.eff_tl(mid)
            self.bus.write(mid, 41, [self.settings["accel"]])
            self.bus.write(mid, 46, list(self.settings["speed"].to_bytes(2, "little")))
            self.bus.write(mid, 48, list(tl.to_bytes(2, "little")))

    def calib_start(self):
        for mid in MOTOR_IDS:
            self.bus.write(mid, 40, [0])
        time.sleep(0.05)
        for mid in MOTOR_IDS:  # raw encoder frame: offset 0, full angle range
            self.bus.write_eeprom(mid, 31, [0, 0])
            self.bus.write_eeprom(mid, 9, list((0).to_bytes(2, "little")) + list((4095).to_bytes(2, "little")))
        time.sleep(0.05)
        self.rec = {}
        for mid in MOTOR_IDS:
            live = self.read_live(mid)
            if live:
                p = live["pos"]
                self.rec[mid] = {"last": p, "acc": p, "lo": p, "hi": p, "plo": p, "phi": p,
                                 "win": deque([(time.time(), p)])}
        self.state["calibrating"] = True
        self.log("캘리브레이션 시작: 토크 OFF, offset=0, 각도 제한 해제")

    def calib_track(self, mid, pos):
        r = self.rec.get(mid)
        if r is None:
            return
        d = ((pos - r["last"] + 2048) % 4096) - 2048
        r["last"] = pos
        r["acc"] += d
        r["plo"] = min(r["plo"], r["acc"])  # instantaneous peaks, shown but not saved
        r["phi"] = max(r["phi"], r["acc"])
        now, win = time.time(), r["win"]
        win.append((now, r["acc"]))
        while len(win) >= 2 and win[1][0] <= now - CALIB_HOLD_S:
            win.popleft()
        if win[0][0] > now - CALIB_HOLD_S:
            return  # window doesn't span the hold time yet
        vals = sorted(v for _, v in win)
        if vals[-1] - vals[0] > CALIB_STILL_STEPS:
            return  # still moving: spikes and pass-through values never count
        rest = vals[len(vals) // 2]
        r["hi"] = max(r["hi"], rest)
        r["lo"] = min(r["lo"], rest)

    def calib_finish(self, only=None):
        old = self.calib.get("motors", {})
        motors = {}  # calib_start zeroed every offset, so nothing from the old file is still valid
        for mid, r in self.rec.items():
            if only and mid not in only:
                continue
            span = r["hi"] - r["lo"]
            if span < 100:
                self.log(f"모터 {mid}: 움직임 {span} step — 너무 작아 건너뜀")
                continue
            if span >= 4090:
                self.log(f"모터 {mid}: 움직임이 한 바퀴 이상 — 건너뜀")
                continue
            mid_raw = (r["lo"] + span // 2) % 4096
            offset = mid_raw - 2048
            if offset > 2047:
                offset -= 4096
            offset = max(-2047, min(2047, offset))  # 11-bit sign-magnitude
            self.bus.write_eeprom(mid, 31, list(_sm_encode(offset, 11).to_bytes(2, "little")))
            lo = (r["lo"] - offset) % 4096  # present = actual - offset
            hi = lo + span
            time.sleep(0.03)
            live = self.read_live(mid)
            got = live["pos"] if live else None
            exp = (r["acc"] - offset) % 4096
            prev = old.get(str(mid), {})
            motors[str(mid)] = {"min": lo, "max": hi, "offset": offset, "invert": prev.get("invert", False),
                                "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            self.log(f"모터 {mid}: 범위 {span} step ({span * 360 / 4096:.0f}°), offset {offset}, "
                     f"검증 pos {got} (예상 {exp})")
        name = self.new_name or time.strftime("calib_%Y%m%d_%H%M%S")
        self.calib = {"name": name, "robot": "elrobot_follower", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "port": self.state.get("port"), "format": "raw arc in offset frame; angle = lower + p*(upper-lower)",
                      "motors": motors}
        save_calib(self.calib)
        self.publish_calib()
        self.log(f"캘리브레이션 저장: calibrations/{name}.json")
        self.state["calibrating"] = False
        self.rec = {}
        self.refresh_eeprom()

    def publish_calib(self):
        self.state["calib"] = self.calib
        self.state["calib_files"] = list_calibs()
        p = _calib_path(self.calib["name"]) if self.calib.get("name") else None
        self.state["calib_raw"] = p.read_text() if p and p.exists() else ""

    def calib_select(self, name):
        c = load_calib(name)
        if not c.get("name"):
            raise RuntimeError(f"{name} 파일을 읽지 못했습니다")
        # the stored arcs live in the frame of the stored offsets, so the EEPROM must match the file
        for mid in MOTOR_IDS:
            self.bus.write(mid, 40, [0])
        for mid_s, m in c["motors"].items():
            self.bus.write_eeprom(int(mid_s), 31, list(_sm_encode(int(m["offset"]), 11).to_bytes(2, "little")))
        self.calib = c
        ACTIVE_FILE.write_text(name)
        self.publish_calib()
        self.refresh_eeprom()
        self.log(f"캘리브레이션 적용: {name} (토크 OFF, offset {len(c['motors'])}개 기록)")

    def refresh_eeprom(self):
        for mid in MOTOR_IDS:
            e = self.read_eeprom(mid)
            if e:
                self.state["eeprom"][mid] = e

    def handle(self, c):
        t = c.get("type")
        if t == "torque":
            ids = MOTOR_IDS if c["id"] == "all" else [int(c["id"])]
            for mid in ids:
                self.cmd_torque(mid, bool(c["on"]))
        elif t == "goal":
            self.cmd_goal(int(c["id"]), int(c["raw"]))
        elif t == "goal_many":                       # grasp executor: one interpolation step for all joints
            for mid, raw in c["raws"].items():
                self.cmd_goal(int(mid), int(raw))
        elif t == "settings":
            for k in ("speed", "accel", "torque_limit"):
                if k in c:
                    self.settings[k] = int(c[k])
            self.settings["torque_limit"] = max(0, min(1000, self.settings["torque_limit"]))
            self.apply_settings()
            self.log(f"설정 변경: 속도 {self.settings['speed']}, 가속 {self.settings['accel']}, 토크 {self.settings['torque_limit']}‰")
        elif t == "motor_tl":
            mid = int(c["id"])
            if c.get("value") is None:
                self.settings["motor_tl"].pop(str(mid), None)
            else:
                self.settings["motor_tl"][str(mid)] = int(c["value"])
            self.bus.write(mid, 48, list(self.eff_tl(mid).to_bytes(2, "little")))
        elif t == "estop":
            for mid in MOTOR_IDS:
                self.bus.write(mid, 40, [0])
            self.log("비상정지: 전체 토크 OFF")
        elif t == "calib_start":
            self.calib_start()
        elif t == "calib_finish":
            self.calib_finish()
        elif t == "calib_cancel":
            self.state["calibrating"] = False
            self.rec = {}
            if self.calib.get("name"):
                self.calib_select(self.calib["name"])  # restore the offsets the active file expects
                self.log("캘리브레이션 취소 — 기존 파일의 offset 복원")
            else:
                self.log("캘리브레이션 취소 (offset은 0인 상태로 남음)")
                self.refresh_eeprom()
        elif t == "invert":
            m = self.calib["motors"].get(str(c["id"]))
            if m and self.calib.get("name"):
                m["invert"] = bool(c["on"])
                save_calib(self.calib)
                self.publish_calib()
        elif t == "calib_select":
            self.calib_select(c["name"])
        elif t == "calib_new":
            self.new_name = _calib_path(c.get("name") or time.strftime("calib_%Y%m%d_%H%M%S")).stem
            self.state["new_name"] = self.new_name
            self.calib_start()
        elif t == "refresh":
            self.refresh_eeprom()


    def run(self):
        while True:
            try:
                self.state["ports"] = list_ports()
                if self.bus is None:
                    self.state["motors"] = {}
                    if self.source == "virtual":
                        from virtual import VirtualBus
                        port = "가상 로봇"
                        self.bus = VirtualBus(self.calib)
                    else:
                        port = find_port() if self.source == "auto" else self.source
                        if not port or port not in list_ports():
                            self.state["port"] = None
                            raise RuntimeError("로봇 USB 장치 없음 — 드라이버 보드 USB 연결 확인 (또는 가상 로봇 선택)")
                        self.bus = Bus(port)
                    self.state["port"] = port
                    self.refresh_eeprom()
                    for mid in MOTOR_IDS:
                        r = self.read_ram(mid)
                        if r:
                            self.state["ram"][mid] = r
                    self.state["connected"] = True
                    self.state["error"] = None
                    self.log(f"연결됨 {port}")
                tick = 0
                self._switch = False
                while True:
                    if self._switch:
                        raise _Reconnect()
                    while not self.cmds.empty():
                        c = self.cmds.get_nowait()
                        try:
                            self.handle(c)
                        except Exception as e:
                            self.log(f"명령 실패 {c.get('type')}: {e}")
                    missing = 0
                    for mid in MOTOR_IDS:
                        live = self.read_live(mid)
                        if live is None:
                            missing += 1
                            prev = self.state["motors"].get(mid)
                            if prev:
                                prev["online"] = False
                            continue
                        live["online"] = True
                        self.state["motors"][mid] = live
                        if self.state["calibrating"]:
                            self.calib_track(mid, live["pos"])
                    if tick % 200 == 0:
                        self.state["ports"] = list_ports()
                    if tick % 10 == 0:
                        for mid in MOTOR_IDS:
                            r = self.read_ram(mid)
                            if r:
                                self.state["ram"][mid] = r
                    if self.state["calibrating"]:
                        self.state["rec"] = {m: {"lo": r["lo"], "hi": r["hi"], "acc": r["acc"], "plo": r["plo"], "phi": r["phi"]}
                                             for m, r in self.rec.items()}
                    self.state["connected"] = missing < len(MOTOR_IDS)
                    tick += 1
                    time.sleep(0.005)
            except _Reconnect:
                self.state.update(connected=False, error=None)
                self.log(f"로봇 연결 전환: {self.source}")
                try:
                    self.bus and self.bus.ser.close()
                except Exception:
                    pass
                self.bus = None
                continue
            except Exception as e:
                self.state["connected"] = False
                self.state["error"] = str(e)
                if getattr(self, "_switch", False):          # source changed while disconnected: retry now
                    self._switch = False
                    self.log(f"로봇 연결 전환: {self.source}")
                    self.bus = None
                    continue
                try:
                    self.bus and self.bus.ser.close()
                except Exception:
                    pass
                self.bus = None
                time.sleep(1.0)


arm = Arm()
threading.Thread(target=arm.run, daemon=True).start()

try:
    import sys as _sys
    _sys.path.insert(0, str(HERE / "camera"))
    from stream import CameraStream
    cam = CameraStream()
    arm.state["camera"] = cam.status  # live dict, updated by the camera thread
except Exception as e:  # camera deps missing: the arm console still works
    cam = None
    arm.state["camera"] = {"connected": False, "error": f"camera disabled: {e}"}

try:
    _sys.path.insert(0, str(HERE / "grasp"))
    from manager import GraspManager
    grasp = GraspManager(arm, cam, ROBOT_DIR / "elrobot_follower.urdf")
    arm.state["grasp"] = grasp.state
except Exception as e:
    grasp = None
    arm.state["grasp"] = {"stage": "disabled", "error": f"grasp disabled: {e}"}

app = FastAPI()
app.mount("/robot", StaticFiles(directory=ROBOT_DIR), name="robot")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    arm.state["clients"] += 1

    async def reader():
        while True:
            c = json.loads(await sock.receive_text())
            if c.get("type") == "robot_source":
                try:
                    arm.set_source(c.get("source", "auto"))
                except Exception as e:
                    arm.log(f"명령 실패 robot_source: {e}")
                continue
            if c.get("type") == "estop" and grasp is not None:
                grasp.abort()                         # stop a running grasp before the torque goes off
            if c.get("type", "").startswith(("reg_", "grasp_")):
                try:
                    if grasp is None:
                        raise RuntimeError(arm.state["grasp"].get("error", "파지 모듈 없음"))
                    await asyncio.to_thread(grasp.handle, c)   # planning takes ~0.1–2 s; keep the socket responsive
                    if c["type"] != "reg_set":
                        arm.log({"grasp_select": "물체 선택", "grasp_plan": "파지 계획", "grasp_exec": "파지 실행 시작",
                                 "grasp_go": "집기: 계획 후 실행 시작", "grasp_pick": "집기", "grasp_place": "옮기기",
                                 "grasp_world_start": "물리 세계 시작", "grasp_world_reset": "물리 세계 다시 구성",
                                 "grasp_world_stop": "물리 세계 정지",
                                 "grasp_cancel": "파지 취소"}.get(c["type"], c["type"]) + (f": {grasp.state['error']}" if grasp.state.get("error") else ""))
                except Exception as e:
                    arm.log(f"명령 실패 {c['type']}: {e}")
                continue
            if c.get("type", "").startswith(("floor_", "recon_", "cam_", "detect_")):
                try:
                    if cam is None:
                        raise RuntimeError("카메라 모듈이 꺼져 있음")
                    cam.handle(c)
                    if c["type"] == "cam_virtual_toggle":
                        arm.log(f"가상 물체 {c['index'] + 1} {'치움' if c['index'] in cam.status['virtual_hidden'] else '다시 놓음'}")
                    elif c["type"] == "cam_config":
                        arm.log(f"카메라 연결 방식: {cam.status['link']}" + (f" {cam.status.get('host')}" if cam.status["link"] == "wifi" else ""))
                    elif c["type"].startswith("recon_"):
                        arm.log({"recon_start": "복원 시작", "recon_stop": "복원 정지", "recon_reset": "복원 초기화"}[c["type"]])
                    elif c["type"].startswith("detect_"):
                        arm.log({"detect_start": "물체 인식 시작", "detect_stop": "물체 인식 정지", "detect_reset": "인식 기록 초기화",
                                 "detect_vocab": "인식 어휘 변경"}.get(c["type"], c["type"]))
                    elif c["type"].startswith("cam_"):
                        arm.log(c["type"])
                    else:
                        arm.log(f"바닥 {'고정' if c['type'] == 'floor_lock' else '해제'}: {(cam.status.get('floor') or {}).get('spread_mm', '-')} mm 편차")
                except Exception as e:
                    arm.log(f"명령 실패 {c['type']}: {e}")
            else:
                arm.cmds.put(c)

    task = asyncio.create_task(reader())
    try:
        while True:
            try:
                txt = json.dumps(arm.state, default=str)
            except RuntimeError:  # state mutated by the bus thread mid-dump; retry next tick
                await asyncio.sleep(0.005)
                continue
            await sock.send_text(txt)
            await asyncio.sleep(0.05)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        arm.state["clients"] -= 1
        task.cancel()


@app.get("/grasp_object")
def grasp_object():
    """u32 N | f32[N*3] selected object points (ARKit world, m)"""
    return Response(content=grasp.object_bytes() if grasp else b"", media_type="application/octet-stream")


@app.get("/grasp_object_mesh")
def grasp_object_mesh():
    """per-object TSDF of the selection, same layout as /recon_mesh (ARKit world, m)"""
    return Response(content=grasp.object_mesh_bytes() if grasp else b"", media_type="application/octet-stream")


@app.get("/recon_objects")
def recon_objects():
    """u32 count | per object: u32 id | raw TSDF mesh | completed mesh (each: u32 nv, nt | f32 v | f32 n | u32 tris | u8 rgb)"""
    return Response(content=cam.recon.objects_bytes() if cam else b"", media_type="application/octet-stream")


@app.get("/detect_tracks")
def detect_tracks():
    """mask-only objects: u32 count | per track: u32 id | u32 n | f32 n×3 (ARKit world)"""
    fn = getattr(cam.detect, "track_bytes", None) if cam else None
    return Response(content=fn() if fn else b"", media_type="application/octet-stream")


@app.get("/camera.jpg")
def camera_jpg():
    """the camera's own newest colour image (iPhone or virtual camera), for the console's inset"""
    j = cam.color_jpeg() if cam else None
    return Response(content=j or b"", media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/recon_floor.png")
def recon_floor_png():
    """floor orthophoto (rows = +z, columns = +x of the ARKit world); bounds in /recon_floor.json"""
    f = cam.recon.floor_png() if cam else None
    return Response(content=f[1] if f else b"", media_type="image/png")


@app.get("/recon_floor.json")
def recon_floor_json():
    f = cam.recon.floor_png() if cam else None
    return f[0] if f else {}


@app.get("/recon_fill")
def recon_fill():
    """inferred floor (holes + under objects), /recon_mesh layout"""
    return Response(content=cam.recon.fill_bytes() if cam else b"", media_type="application/octet-stream")


@app.get("/recon_mesh")
def recon_mesh():
    """u32 nv | u32 nt | f32[nv*3] xyz | f32[nv*3] normals | u32[nt*3] tris | u8[nv*3] rgb  (ARKit world, m)"""
    blob = cam.recon.mesh_bytes() if cam is not None else None
    return Response(content=blob or b"", media_type="application/octet-stream")


@app.websocket("/ws_cam")
async def ws_cam(sock: WebSocket):
    """Binary point-cloud stream: u32 N | f32[16] T_world_cam (column-major) | f32[N*3] xyz | u8[N*3] rgb"""
    await sock.accept()
    last = -1
    try:
        while True:
            if cam is not None:
                seq, pkt = cam.latest()
                if pkt is not None and seq != last:
                    last = seq
                    await sock.send_bytes(pkt)
            await asyncio.sleep(0.03)
    except (WebSocketDisconnect, RuntimeError):
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HTTP_HOST, port=HTTP_PORT, log_level="warning")
