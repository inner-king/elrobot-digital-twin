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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).parent
ROBOT_DIR = HERE.parent / "norma-core" / "hardware" / "elrobot" / "simulation"
CALIB_DIR = HERE / "calibrations"
ACTIVE_FILE = CALIB_DIR / ".active"
PORT = os.environ.get("ARM_PORT") or next(iter(sorted(glob.glob("/dev/cu.usbmodem*"))), None)
BAUD = 1_000_000
MOTOR_IDS = list(range(1, 9))
HTTP_PORT = int(os.environ.get("ARM_HTTP_PORT", "8765"))

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
        self.state = {"port": PORT, "connected": False, "error": None, "motors": {},
                      "eeprom": {}, "ram": {}, "calibrating": False, "rec": {},
                      "calib": self.calib, "calib_files": list_calibs(), "calib_raw": "", "settings": self.settings, "log": [],
                      "torque_cap": TORQUE_CAP}
        self.bus = None
        self.rec = {}
        self.publish_calib()  # calibration recording: id -> {last, acc, lo, hi}

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
                      "port": PORT, "format": "raw arc in offset frame; angle = lower + p*(upper-lower)",
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
                if self.bus is None:
                    if not PORT:
                        raise RuntimeError("시리얼 포트를 찾지 못했습니다 (ARM_PORT 지정)")
                    self.bus = Bus(PORT)
                    self.refresh_eeprom()
                    for mid in MOTOR_IDS:
                        r = self.read_ram(mid)
                        if r:
                            self.state["ram"][mid] = r
                    self.state["connected"] = True
                    self.state["error"] = None
                    self.log(f"연결됨 {PORT}")
                tick = 0
                while True:
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
            except Exception as e:
                self.state["connected"] = False
                self.state["error"] = str(e)
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

app = FastAPI()
app.mount("/robot", StaticFiles(directory=ROBOT_DIR), name="robot")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()

    async def reader():
        while True:
            msg = await sock.receive_text()
            arm.cmds.put(json.loads(msg))

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
        task.cancel()


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
    uvicorn.run(app, host="127.0.0.1", port=HTTP_PORT, log_level="warning")
