"""任意 log 訊息 / 欄位索引:欄位繪圖(欄位樹 / 預設組 / 運算式)、參數、訊息面板共用。

快路徑(.bin):整個檔案分塊讀進 numpy(可回報進度),借用 pymavlink DFReader_binary
建好的「每種訊息的檔案位移」索引(dfindexer,Cython;略過 init_clock),隨即關檔
(Windows 不關會鎖檔),再依 FMT 格式字元組出 packed little-endian 結構化 dtype,一次取出
某型別的全部 payload —— 不必逐筆建立 DFMessage。數值轉換(乘數、字串解碼、
陣列欄位)逐條對照 DFReader 的 FORMAT_TO_STRUCT 與 DFMessage.__getattr__,
結果與 recv_match 逐筆讀出的值相同(tests/test_dflog.py 逐型別逐欄驗證)。

延遲載入:開檔只建位移索引;某型別第一次被要求時才解碼並快取。
快取的陣列設為唯讀 —— 需要就地修改請先 .copy()。

時間一律是 log 時基秒(TimeUS / 1e6);沒有 TimeUS 的舊 log 用 TimeMS / 1e3,
兩者都沒有的訊息取檔案中前一筆有時間戳訊息的時間。

慢路徑:非二進位 log(文字 .log)或 pymavlink 內部結構不符時,改用 recv_msg
逐筆讀(結果相同,只是慢)。
"""
from __future__ import annotations

import math
import os
import re
import threading
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MsgInfo", "Param", "LogMessage", "LogIndex", "parse_key",
    "EV_NAMES", "ERR_SUBSYS", "ARM_METHODS", "err_text", "ev_text",
    "VEHICLE_MAV_TYPE", "MK_MSG", "MK_EV", "MK_ERR", "MK_MODE", "MK_ARM", "MK_DISARM",
]


# ---------------- 公開資料結構 ----------------

@dataclass
class MsgInfo:
    """一種訊息的概要(欄位樹用)。"""
    name: str
    fields: list[str]                    # 全部欄位(不含 TimeUS),依 log 中的順序
    count: int                           # 筆數(所有 instance 合計)
    instances: list[int] | None = None  # 有 instance 欄(FMTU '#')時的 instance 編號
    units: dict[str, str] = field(default_factory=dict)   # 欄位 → 單位(只列有單位者)


@dataclass
class Param:
    name: str
    value: float                 # 最後的值
    default: float = math.nan    # PARM.Default(沒有則 nan)
    t: float = math.nan          # 被設成最後這個值的時間(log 秒)


# LogMessage.kind
MK_MSG, MK_EV, MK_ERR, MK_MODE, MK_ARM, MK_DISARM = "msg", "ev", "err", "mode", "arm", "disarm"


@dataclass
class LogMessage:
    t: float
    kind: str      # "msg" | "ev" | "err" | "mode" | "arm" | "disarm"
    text: str


# ---------------- ArduPilot 列舉 ----------------

# LogEvent(AP_Logger.h)。80/81 舊版名為 FENCE_FLOOR_*,4.5 起改稱 FENCE_ALT_MIN_*(同義)
EV_NAMES: dict[int, str] = {
    10: "ARMED", 11: "DISARMED", 15: "AUTO_ARMED", 17: "LAND_COMPLETE_MAYBE",
    18: "LAND_COMPLETE", 19: "LOST_GPS", 21: "FLIP_START", 22: "FLIP_END",
    25: "SET_HOME", 26: "SET_SIMPLE_ON", 27: "SET_SIMPLE_OFF", 28: "NOT_LANDED",
    29: "SET_SUPERSIMPLE_ON", 30: "AUTOTUNE_INITIALISED", 31: "AUTOTUNE_OFF",
    32: "AUTOTUNE_RESTART", 33: "AUTOTUNE_SUCCESS", 34: "AUTOTUNE_FAILED",
    35: "AUTOTUNE_REACHED_LIMIT", 36: "AUTOTUNE_PILOT_TESTING", 37: "AUTOTUNE_SAVEDGAINS",
    38: "SAVE_TRIM", 39: "SAVEWP_ADD_WP", 41: "FENCE_ENABLE", 42: "FENCE_DISABLE",
    43: "ACRO_TRAINER_OFF", 44: "ACRO_TRAINER_LEVELING", 45: "ACRO_TRAINER_LIMITED",
    46: "GRIPPER_GRAB", 47: "GRIPPER_RELEASE", 49: "PARACHUTE_DISABLED",
    50: "PARACHUTE_ENABLED", 51: "PARACHUTE_RELEASED", 52: "LANDING_GEAR_DEPLOYED",
    53: "LANDING_GEAR_RETRACTED", 54: "MOTORS_EMERGENCY_STOPPED",
    55: "MOTORS_EMERGENCY_STOP_CLEARED", 56: "MOTORS_INTERLOCK_DISABLED",
    57: "MOTORS_INTERLOCK_ENABLED", 58: "ROTOR_RUNUP_COMPLETE",
    59: "ROTOR_SPEED_BELOW_CRITICAL", 60: "EKF_ALT_RESET", 61: "LAND_CANCELLED_BY_PILOT",
    62: "EKF_YAW_RESET", 63: "AVOIDANCE_ADSB_ENABLE", 64: "AVOIDANCE_ADSB_DISABLE",
    65: "AVOIDANCE_PROXIMITY_ENABLE", 66: "AVOIDANCE_PROXIMITY_DISABLE",
    67: "GPS_PRIMARY_CHANGED", 71: "ZIGZAG_STORE_A", 72: "ZIGZAG_STORE_B",
    73: "LAND_REPO_ACTIVE", 74: "STANDBY_ENABLE", 75: "STANDBY_DISABLE",
    76: "FENCE_ALT_MAX_ENABLE", 77: "FENCE_ALT_MAX_DISABLE", 78: "FENCE_CIRCLE_ENABLE",
    79: "FENCE_CIRCLE_DISABLE", 80: "FENCE_FLOOR_ENABLE", 81: "FENCE_FLOOR_DISABLE",
    82: "FENCE_POLYGON_ENABLE", 83: "FENCE_POLYGON_DISABLE",
    85: "EK3_SOURCES_SET_TO_PRIMARY", 86: "EK3_SOURCES_SET_TO_SECONDARY",
    87: "EK3_SOURCES_SET_TO_TERTIARY", 90: "AIRSPEED_PRIMARY_CHANGED",
    163: "SURFACED", 164: "NOT_SURFACED", 165: "BOTTOMED", 166: "NOT_BOTTOMED",
}
EV_ARMED, EV_DISARMED = 10, 11

# LogErrorSubsystem(AP_Logger.h)
ERR_SUBSYS: dict[int, str] = {
    1: "MAIN", 2: "RADIO", 3: "COMPASS", 4: "OPTFLOW", 5: "FAILSAFE_RADIO",
    6: "FAILSAFE_BATT", 7: "FAILSAFE_GPS", 8: "FAILSAFE_GCS", 9: "FAILSAFE_FENCE",
    10: "FLIGHT_MODE", 11: "GPS", 12: "CRASH_CHECK", 13: "FLIP", 14: "AUTOTUNE",
    15: "PARACHUTES", 16: "EKFCHECK", 17: "FAILSAFE_EKFINAV", 18: "BARO", 19: "CPU",
    20: "FAILSAFE_ADSB", 21: "TERRAIN", 22: "NAVIGATION", 23: "FAILSAFE_TERRAIN",
    24: "EKF_PRIMARY", 25: "THRUST_LOSS_CHECK", 26: "FAILSAFE_SENSORS", 27: "FAILSAFE_LEAK",
    28: "PILOT_INPUT", 29: "FAILSAFE_VIBE", 30: "INTERNAL_ERROR", 31: "FAILSAFE_DEADRECKON",
}

# LogErrorCode:各子系統常見代碼(0 = 已恢復;FAILSAFE_* 一律 0 解除 / 非 0 觸發)
_ERR_CODES: dict[str, dict[int, str]] = {
    "MAIN": {1: "INS 延遲"},
    "RADIO": {2: "遙控訊號延遲"},
    "COMPASS": {1: "初始化失敗", 4: "異常"},
    "CRASH_CHECK": {1: "偵測到墜機", 2: "失去控制"},
    "FLIP": {2: "翻滾中止"},
    "PARACHUTES": {2: "高度過低未開傘", 3: "已著地未開傘"},
    "EKFCHECK": {0: "變異數恢復", 2: "變異數過大"},
    "BARO": {1: "初始化失敗", 2: "Glitch", 3: "深度異常", 4: "異常"},
    "GPS": {2: "Glitch"},
    "TERRAIN": {2: "缺地形資料"},
    "NAVIGATION": {2: "無法設定目的地", 3: "RTL 重新開始", 4: "Circle 初始化失敗",
                   5: "目的地在圍欄外", 6: "RTL 缺測距儀"},
    "INTERNAL_ERROR": {1: "偵測到內部錯誤"},
    "THRUST_LOSS_CHECK": {1: "推力不足"},
}

# AP_Arming::Method(ARM.Method;只列確定的幾個)
ARM_METHODS: dict[int, str] = {0: "方向舵", 1: "MAVLink", 2: "AUX 開關", 3: "馬達測試",
                               4: "Lua 腳本", 12: "墜機偵測(CRASH_CHECK)", 13: "落地後自動", 17: "閒置自動"}


def err_text(subsys: int, ecode: int) -> str:
    """ERR → 「子系統:說明 (代碼)」,例:FAILSAFE_BATT:觸發 (1)、GPS:Glitch (2)。"""
    subsys, ecode = int(subsys), int(ecode)
    name = ERR_SUBSYS.get(subsys)
    if name is None:
        return f"ERR {subsys} 代碼 {ecode}"
    if name.startswith("FAILSAFE_"):
        desc = "解除" if ecode == 0 else "觸發"
    elif name == "FLIGHT_MODE":
        desc = "模式切換失敗"          # ECode = 想切換過去的模式編號
    elif name == "EKF_PRIMARY":
        desc = "主核心切換"            # ECode = 新的主核心編號
    else:
        desc = _ERR_CODES.get(name, {}).get(ecode)
        if desc is None:
            if ecode != 0:
                return f"{name} 代碼 {ecode}"
            desc = "已恢復"
    return f"{name}:{desc} ({ecode})"


def ev_text(ev_id: int) -> str:
    """EV Id → 事件名稱(未知編號回傳「EV n」)。"""
    ev_id = int(ev_id)
    return EV_NAMES.get(ev_id, f"EV {ev_id}")


def arm_text(armed: bool, method: int | None = None, forced: bool = False) -> str:
    s = "解鎖 ARMED" if armed else "上鎖 DISARMED"
    extra = []
    if method is not None and int(method) in ARM_METHODS:
        extra.append(ARM_METHODS[int(method)])
    if forced:
        extra.append("強制")
    return s + (f"({'、'.join(extra)})" if extra else "")


# 機型字串 → MAV_TYPE(MODE 編號 → 名稱的對照依機型而異)
VEHICLE_MAV_TYPE = {"copter": 2, "heli": 4, "plane": 1, "rover": 10, "sub": 12,
                    "blimp": 7, "tracker": 5}
_MAV_TYPE_VEHICLE = {2: "copter", 3: "copter", 13: "copter", 14: "copter", 15: "copter",
                     29: "copter", 35: "copter", 4: "heli", 1: "plane", 10: "rover",
                     11: "rover", 12: "sub", 7: "blimp", 5: "tracker"}
_VER_BUILD = {1: 10, 2: 2, 3: 1, 4: 5, 7: 12, 12: 7, 13: 4}   # VER.BU → MAV_TYPE(同 DFReader)
_MAV_TYPE_DEFAULT = 1    # DFReader 預設 FIXED_WING(log 判斷不出機型時的 MODE 名稱對照)


def _mav_type_from_text(text: str) -> int | None:
    """MSG 文字 → MAV_TYPE(規則與 DFReader._add_msg 相同)。"""
    if "Rover" in text:
        return 10
    if "Plane" in text:
        return 1
    if "Copter" in text:
        return 2
    if text.startswith("Antenna"):
        return 5
    if "ArduSub" in text:
        return 12
    if "Blimp" in text:
        return 7
    return None


def mode_mapping(mav_type: int) -> dict[int, str] | None:
    try:
        from pymavlink import mavutil
    except ImportError:   # pragma: no cover
        return None
    return mavutil.mode_mapping_bynumber(mav_type)


# ---------------- key 解析 ----------------

_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_]+)(?:\[(\d+)\])?(?:\.([A-Za-z0-9_]+))?\s*$")


def parse_key(key: str) -> tuple[str, int | None, str | None]:
    """「GPS[1].Spd」→ ("GPS", 1, "Spd");「ATT」→ ("ATT", None, None)。格式不符拋 ValueError。"""
    m = _KEY_RE.match(key or "")
    if m is None:
        raise ValueError(f"無效的欄位名稱:{key!r}")
    name, inst, fld = m.groups()
    return name, (int(inst) if inst is not None else None), fld


def make_key(name: str, instance: int | None = None, fld: str | None = None) -> str:
    s = name if instance is None else f"{name}[{instance}]"
    return s if fld is None else f"{s}.{fld}"


# ---------------- 格式字元(對照 DFReader.FORMAT_TO_STRUCT) ----------------

_FMT_DTYPE = {
    "a": ("<i2", (32,)), "b": "i1", "B": "u1", "g": "<f2", "h": "<i2", "H": "<u2",
    "i": "<i4", "I": "<u4", "f": "<f4", "n": "S4", "N": "S16", "Z": "S64",
    "c": "<i2", "C": "<u2", "e": "<i4", "E": "<u4", "L": "<i4", "d": "<f8",
    "M": "i1", "q": "<i8", "Q": "<u8",
}
_FMT_MULT = {"c": 0.01, "C": 0.01, "e": 0.01, "E": 0.01, "L": 1.0e-7}
_STR_CHARS = frozenset("nNZ")
_META = frozenset({"FMT", "FMTU", "UNIT", "MULT"})     # 描述格式用,不列在 messages()
_NO_PLOT = _META | {"PARM", "FILE"}                     # 不列進 field_keys()
_INST_CANDIDATES = ("I", "Instance", "Inst", "C", "IMU", "Core")


def _char_size(ch: str) -> int:
    return np.dtype(_FMT_DTYPE[ch]).itemsize


def _decode_strings(raw: np.ndarray) -> np.ndarray:
    """bytes 陣列 → str 陣列:先試 UTF-8、失敗用 ISO-8859-1,再截到第一個 \\0(同 DFReader)。"""
    if raw.size == 0:
        return np.zeros(0, dtype="<U1")
    uniq, inv = np.unique(raw, return_inverse=True)
    out = []
    for b in uniq.tolist():
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            s = b.decode("ISO-8859-1")
        i = s.find("\0")
        out.append(s if i < 0 else s[:i])
    return np.asarray(out, dtype=str)[inv.reshape(-1)]


def _convert(raw: np.ndarray, ch: str) -> np.ndarray:
    """結構化欄位(little-endian)→ 原生 dtype,套用格式乘數。"""
    if ch in _STR_CHARS:
        return _decode_strings(raw)
    if ch == "a":
        return raw.astype(np.int16)
    mult = _FMT_MULT.get(ch)
    if mult is not None:
        v = raw.astype(np.float64)
        # 同 DFReader:0 < mult < 1 時用除法(÷1e7 比 ×1e-7 準)
        return v / (1.0 / mult) if 0.0 < mult < 1.0 else v * mult
    if ch in "fgd":
        return raw.astype(np.float64)
    return raw.astype(raw.dtype.newbyteorder("="))


def _readonly(a: np.ndarray) -> np.ndarray:
    try:
        a.flags.writeable = False
    except ValueError:   # 別人的 view,改不動就算了
        pass
    return a


def _is_sorted(t: np.ndarray) -> bool:
    return t.size < 2 or bool(np.all(t[1:] >= t[:-1]))


# ---------------- 儲存層:二進位(快) / 表格(from_tables、慢路徑) ----------------

@dataclass
class _Fmt:
    name: str
    columns: list[str]             # 欄位(含 TimeUS)
    chars: str = ""                # 格式字元(與 columns 對齊);表格來源為空字串
    length: int = 0                # 整筆位元組數(含 3 位元組 header)
    inst_field: str | None = None
    units: dict[str, str] = field(default_factory=dict)
    ok: bool = True                # 格式總長與 FMT.Length 相符(否則 DFReader 也解不出來)
    offs: list[int] = field(default_factory=list)   # 各欄在 payload 中的位移
    implicit_time: bool = False    # 記憶體表格:"t" 即 TimeUS / 1e6

    def char(self, col: str) -> str:
        if not self.chars:
            return ""
        i = len(self.columns) - 1 - self.columns[::-1].index(col)   # 重名欄位取最後一個(同 colhash)
        return self.chars[i]

    def time_field(self) -> str | None:
        if self.implicit_time or "TimeUS" in self.columns:
            return "TimeUS"
        if "TimeMS" in self.columns:
            return "TimeMS"
        return None


def _units_list_to_dict(columns: list[str], units) -> dict[str, str]:
    if not units:
        return {}
    return {c: u for c, u in zip(columns, units) if u and c != "TimeUS"}


class _BinStore:
    """整個 .bin 的位元組 + 每型別位移;按需向量化解碼。"""

    def __init__(self, buf: np.ndarray, fmts: dict[str, _Fmt], offsets: dict[str, np.ndarray],
                 all_names: list[str]):
        self.buf = buf
        self.fmts = fmts
        self.offsets = offsets
        self.all_names = all_names
        self._dtypes: dict[str, np.dtype] = {}
        self._timeline = None   # (位移, 時間):給沒有時間欄的訊息找「前一筆」時間

    @classmethod
    def from_reader(cls, mlog, buf: np.ndarray | None = None) -> "_BinStore":
        """buf:已讀進記憶體的整個檔案(省一次複製);None 則從 reader 的 mmap 複製。"""
        formats = mlog.formats
        name_to_id = mlog.name_to_id
        offsets_all = mlog.offsets
        if not (isinstance(offsets_all, list) and len(offsets_all) == 256
                and all(isinstance(k, int) for k in formats)):
            raise TypeError("不是 DFReader_binary 的位移索引")       # 例如文字 log → 走慢路徑
        if buf is None or buf.size != mlog.data_len:
            buf = np.frombuffer(mlog.data_map, dtype=np.uint8).copy()   # mmap → 自有記憶體,之後可關檔
        size = buf.size
        fmts: dict[str, _Fmt] = {}
        offs: dict[str, np.ndarray] = {}
        for name, tid in name_to_id.items():
            f = formats.get(tid)
            if f is None:
                continue
            chars = "".join(f.format) if not isinstance(f.format, str) else f.format
            cols = list(f.columns)
            n = min(len(chars), len(cols))
            fm = _Fmt(name, cols[:n], chars[:n], int(f.len), getattr(f, "instance_field", None),
                      _units_list_to_dict(cols, getattr(f, "units", None)))
            try:
                sizes = [_char_size(c) for c in chars]
            except KeyError:
                fm.ok = False
                sizes = []
            if fm.ok:
                fm.ok = sum(sizes) == fm.length - 3
                pos = np.concatenate([[0], np.cumsum(sizes)]).astype(int).tolist()
                fm.offs = pos[:n]
            o = np.asarray(offsets_all[tid], dtype=np.int64)
            if o.size:
                o = o[o + fm.length <= size]    # 檔尾被截斷的最後一筆
            fmts[name] = fm
            offs[name] = o if fm.ok else np.zeros(0, np.int64)
        all_names = sorted({f.name for f in formats.values()})
        return cls(buf, fmts, offs, all_names)

    # --- 資訊 ---
    def names(self) -> list[str]:
        return [n for n, o in self.offsets.items() if o.size and self.fmts[n].ok]

    def count(self, name: str) -> int:
        return int(self.offsets[name].size)

    def kind(self, name: str, col: str) -> str:
        ch = self.fmts[name].char(col)
        return "str" if ch in _STR_CHARS else ("arr" if ch == "a" else "num")

    def positions(self, name: str) -> np.ndarray:
        return self.offsets[name]

    # --- 解碼 ---
    def _dtype(self, fm: _Fmt) -> np.dtype:
        dt = self._dtypes.get(fm.name)
        if dt is None:
            dt = np.dtype({"names": [f"f{i}" for i in range(len(fm.columns))],
                           "formats": [_FMT_DTYPE[c] for c in fm.chars],
                           "offsets": fm.offs, "itemsize": fm.length - 3})
            self._dtypes[fm.name] = dt
        return dt

    def _gather(self, offs: np.ndarray, start: int, width: int) -> np.ndarray:
        """每筆取 [ofs+3+start, +width) 位元組 → (n, width) uint8(只複製需要的位元組)。"""
        if offs.size == 0 or width <= 0:
            return np.zeros((offs.size, max(width, 0)), np.uint8)
        win = np.lib.stride_tricks.as_strided(
            self.buf, shape=(self.buf.size - width + 1, width), strides=(1, 1), writeable=False)
        return win[offs + (3 + start)]

    def load(self, name: str) -> dict[str, np.ndarray]:
        """檔案順序的全部欄位(含 TimeUS 原始欄)。"""
        fm = self.fmts[name]
        offs = self.offsets[name]
        dt = self._dtype(fm)
        payload = self._gather(offs, 0, fm.length - 3)
        rec = payload.view(dt).reshape(offs.size) if offs.size else np.zeros(0, dt)
        return {col: (rec[f"f{i}"].copy() if name == "FILE" and ch == "Z"   # 同 DFReader:FILE 內容保留 bytes
                      else _convert(rec[f"f{i}"], ch))
                for i, (col, ch) in enumerate(zip(fm.columns, fm.chars))}

    def column(self, name: str, col: str) -> np.ndarray:
        """只解一欄(時間、instance)— 不必解整筆。"""
        fm = self.fmts[name]
        i = len(fm.columns) - 1 - fm.columns[::-1].index(col)
        ch = fm.chars[i]
        spec = np.dtype(_FMT_DTYPE[ch])
        raw = self._gather(self.offsets[name], fm.offs[i], spec.itemsize)
        raw = raw.view(spec).reshape(raw.shape[0], *spec.shape) if raw.size else np.zeros(0, spec)
        return _convert(raw, ch)

    def times(self, name: str) -> np.ndarray:
        fm = self.fmts[name]
        tf = fm.time_field()
        if tf == "TimeUS":
            return self.column(name, "TimeUS").astype(np.float64) * 1e-6
        if tf == "TimeMS":
            return self.column(name, "TimeMS").astype(np.float64) * 1e-3
        # 沒有時間欄:取檔案中前一筆有時間戳訊息的時間
        if self._timeline is None:
            pos, ts = [], []
            for n in self.names():
                if self.fmts[n].time_field() is not None:
                    pos.append(self.offsets[n])
                    ts.append(self.times(n))
            pos = np.concatenate(pos) if pos else np.zeros(0, np.int64)
            ts = np.concatenate(ts) if ts else np.zeros(0)
            o = np.argsort(pos, kind="stable")
            self._timeline = (pos[o], ts[o])
        pos, ts = self._timeline
        k = np.searchsorted(pos, self.offsets[name], side="right") - 1
        out = np.full(k.size, np.nan)
        ok = k >= 0
        out[ok] = ts[k[ok]]
        return out


class _TableStore:
    """記憶體表格(from_tables / 慢路徑):tables[name] = 檔案(時間)順序的全部列,含 "t"。"""

    def __init__(self, tables: dict[str, dict[str, np.ndarray]], fmts: dict[str, _Fmt],
                 all_names: list[str]):
        self.tables = tables
        self.fmts = fmts
        self.all_names = all_names
        self._pos: dict[str, np.ndarray] = {}
        base = 0
        for name, tbl in tables.items():
            n = int(tbl["t"].size)
            self._pos[name] = np.arange(base, base + n, dtype=np.int64)
            base += n

    def names(self) -> list[str]:
        return [n for n, tbl in self.tables.items() if tbl["t"].size]

    def count(self, name: str) -> int:
        return int(self.tables[name]["t"].size)

    def kind(self, name: str, col: str) -> str:
        a = self.tables[name][col]
        if a.ndim != 1:
            return "arr"
        return "num" if a.dtype.kind in "biuf" else "str"

    def positions(self, name: str) -> np.ndarray:
        return self._pos[name]

    def load(self, name: str) -> dict[str, np.ndarray]:
        return {k: v for k, v in self.tables[name].items() if k != "t"}

    def column(self, name: str, col: str) -> np.ndarray:
        return self.tables[name][col]

    def times(self, name: str) -> np.ndarray:
        return self.tables[name]["t"]

    # --- 慢路徑:逐筆 recv_msg ---
    @classmethod
    def from_recv(cls, mlog, progress_cb=None) -> "_TableStore":
        try:
            mlog.rewind()
        except AttributeError:
            pass
        rows: dict[str, list[list]] = {}
        times: dict[str, list[float]] = {}
        fmts: dict[str, _Fmt] = {}
        seen_cols: dict[str, list[str]] = {}
        last_t = math.nan
        k = 0
        while True:
            m = mlog.recv_msg()
            if m is None:
                break
            fmt = getattr(m, "fmt", None)
            if fmt is None:
                raise ValueError("不是 DataFlash log(找不到 FMT 格式定義)")
            name = m.get_type()
            cols = list(fmt.columns)
            fm = fmts.get(name)
            if fm is None or seen_cols[name] != cols:     # 新型別(或 log 中途重新定義格式)
                seen_cols[name] = cols
                chars = fmt.format if isinstance(fmt.format, str) else "".join(fmt.format)
                n = min(len(cols), len(chars))
                fmts[name] = fm = _Fmt(name, cols[:n], chars[:n], int(getattr(fmt, "len", 0)),
                                       getattr(fmt, "instance_field", None),
                                       _units_list_to_dict(cols, getattr(fmt, "units", None)))
                rows[name] = []
                times[name] = []
            vals = []
            for c in fm.columns:
                try:
                    vals.append(getattr(m, c))
                except Exception:  # noqa: BLE001
                    vals.append(None)
            tf = fm.time_field()
            if tf is not None:
                tv = vals[fm.columns.index(tf)]
                last_t = float(tv) * (1e-6 if tf == "TimeUS" else 1e-3)
            rows[name].append(vals)
            times[name].append(last_t)
            k += 1
            if progress_cb is not None and k % 20000 == 0:
                pct = getattr(mlog, "percent", None)
                if pct is not None:
                    progress_cb(int(pct))
        tables: dict[str, dict[str, np.ndarray]] = {}
        for name, fm in fmts.items():
            cols = list(zip(*rows[name])) if rows[name] else [() for _ in fm.columns]
            tbl = {"t": np.asarray(times[name], dtype=np.float64)}
            for c, ch, vals in zip(fm.columns, fm.chars, cols):
                tbl[c] = _column_from_values(list(vals), ch)
            tables[name] = tbl
        all_names = sorted({f.name for f in getattr(mlog, "formats", {}).values()}
                           if isinstance(getattr(mlog, "formats", None), dict) else fmts)
        return cls(tables, fmts, all_names)


def _column_from_values(vals: list, ch: str) -> np.ndarray:
    """慢路徑:Python 值 → 與快路徑相同 dtype 的陣列。"""
    if ch in _STR_CHARS:
        if vals and isinstance(vals[0], bytes):          # FILE.Data(同 DFReader 保留 bytes)
            return np.asarray(vals, dtype="S64")
        return np.asarray(["" if v is None else str(v) for v in vals], dtype=str) \
            if vals else np.zeros(0, dtype="<U1")
    if ch == "a":
        return np.asarray([list(v) for v in vals], dtype=np.int16).reshape(len(vals), 32)
    if ch in _FMT_MULT or ch in "fgd":
        return np.asarray([math.nan if v is None else v for v in vals], dtype=np.float64)
    dt = np.dtype(_FMT_DTYPE.get(ch, "<i8")).newbyteorder("=")
    return np.asarray([0 if v is None else v for v in vals], dtype=dt)


def _make_index_reader(path: str, progress_cb=None):
    """DFReader_binary,但略過 init_clock:我們用 TimeUS,不需要 GPS 時基;
    而沒有 GPS 的 log(室內 VIO 飛行常見)會讓 init_clock 逐筆解析整個檔案,非常慢。"""
    from pymavlink import DFReader

    class _IndexReader(DFReader.DFReader_binary):
        def init_clock(self):
            self._rewind(keep_messages=True)

    return _IndexReader(path, progress_callback=progress_cb)


def _read_file(path: str, size: int, progress_cb, lo: int, hi: int,
               chunk: int = 8 << 20) -> np.ndarray:
    buf = np.empty(size, dtype=np.uint8)
    mv = memoryview(buf)
    off = 0
    with open(path, "rb") as f:
        while off < size:
            n = f.readinto(mv[off:off + chunk])
            if not n:
                break
            off += n
            if progress_cb is not None:
                progress_cb(lo + int((hi - lo) * off / size))
    return buf[:off]


def _is_binary_log(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\xa3\x95"


# ---------------- LogIndex ----------------

class LogIndex:
    """一份 log 的全部訊息 / 欄位索引(延遲解碼 + 快取)。

    key 寫法:「ATT.Roll」「GPS[1].Spd」;有 instance 的訊息不寫 instance 時取最小的那個。
    """

    path: str
    vehicle: str    # "copter" | "plane" | "rover" | "sub" | "heli" | "blimp" | "tracker" | "unknown"

    def __init__(self, store, path: str = "", vehicle: str | None = None):
        self._store = store
        self._names = frozenset(store.names())     # 有資料、可解碼的型別
        self.path = path
        self._lock = threading.RLock()
        self._raw: dict[str, dict[str, np.ndarray]] = {}
        self._tables: dict[tuple[str, int | None], dict[str, np.ndarray]] = {}
        self._series: dict[tuple[str, int | None, str], np.ndarray] = {}
        self._inst_vals: dict[str, np.ndarray | None] = {}
        self._times: dict[str, np.ndarray] = {}
        self._infos: dict[str, MsgInfo] | None = None
        self._field_keys: list[str] | None = None
        self._params: list[Param] | None = None
        self._log_msgs: list[LogMessage] | None = None
        self._t_range: tuple[float, float] | None = None
        if vehicle is None:
            self.mav_type = self._detect_mav_type()
            self.vehicle = _MAV_TYPE_VEHICLE.get(self.mav_type, "unknown") \
                if self.mav_type is not None else "unknown"
        else:
            self.vehicle = vehicle
            self.mav_type = VEHICLE_MAV_TYPE.get(vehicle)

    def __repr__(self) -> str:
        return (f"LogIndex({os.path.basename(self.path) or '(記憶體)'}, {self.vehicle}, "
                f"{len(self.messages())} 種訊息)")

    # --- 建構 ---
    @classmethod
    def open(cls, path: str, progress_cb=None) -> "LogIndex":
        """開 .bin(快路徑);文字 log 或 pymavlink 內部結構不符時改走逐筆慢路徑。

        progress_cb(int 0..100):建立位移索引的進度(大檔最慢的部分)。
        """
        size = os.path.getsize(path)
        if size == 0:
            raise ValueError(f"檔案是空的(0 位元組):{os.path.basename(path)}")
        if _is_binary_log(path):
            # 先分塊讀進記憶體(冷快取時讀檔才是最慢的,這段可回報進度),
            # pymavlink 建位移索引時檔案已在系統快取中
            buf = _read_file(path, size, progress_cb, 0, 70)
            idx_cb = (lambda p: progress_cb(70 + int(0.3 * p))) if progress_cb else None
            try:
                mlog = _make_index_reader(path, idx_cb)
            except Exception:  # noqa: BLE001 — 自訂 reader 失敗就退回一般連線
                mlog = None
            if mlog is not None:
                try:
                    try:
                        store = _BinStore.from_reader(mlog, buf)
                    except (AttributeError, TypeError, KeyError, ValueError, IndexError):
                        store = _TableStore.from_recv(mlog)
                    return cls(store, path)
                finally:
                    try:
                        mlog.close()   # Windows 上不關會鎖住檔案
                    except Exception:  # noqa: BLE001
                        pass
        from pymavlink import mavutil
        mlog = mavutil.mavlink_connection(path)
        try:
            return cls(_TableStore.from_recv(mlog, progress_cb), path)
        finally:
            try:
                mlog.close()
            except Exception:  # noqa: BLE001
                pass

    @classmethod
    def from_reader(cls, mlog, path: str = "") -> "LogIndex":
        """由已開好的 DFReader 建索引(不會關閉 mlog,由呼叫端負責)。"""
        try:
            store = _BinStore.from_reader(mlog)
        except (AttributeError, TypeError, KeyError, ValueError, IndexError):
            store = _TableStore.from_recv(mlog)
        return cls(store, path or getattr(mlog, "filename", "") or "")

    @classmethod
    def from_tables(cls, tables: dict[str, dict[str, np.ndarray]], vehicle: str = "copter",
                    path: str = "", units: dict[str, dict[str, str]] | None = None) -> "LogIndex":
        """記憶體表格 → 索引(模擬資料用)。

        tables 的 key 為「ATT」或「GPS[1]」;每張表含 "t"(秒)+ 各欄。帶 instance 的表若
        沒有 instance 欄(I / Instance / Inst / C / IMU / Core),自動補一欄 "I"。
        units:{訊息: {欄位: 單位}}(選填)。
        """
        groups: dict[str, dict[int | None, dict[str, np.ndarray]]] = {}
        for key, tbl in tables.items():
            name, inst, fld = parse_key(key)
            if fld is not None:
                raise ValueError(f"表格 key 不可含欄位:{key!r}")
            if "t" not in tbl:
                raise ValueError(f"{key} 缺少 \"t\" 欄")
            # 一律複製:快取會設唯讀旗標,不能動到呼叫端(例如 TrajSource)的陣列
            cols: dict[str, np.ndarray] = {"t": np.array(tbl["t"], dtype=np.float64).reshape(-1)}
            n = cols["t"].size
            for c, v in tbl.items():
                if c in ("t", "TimeUS"):
                    continue
                a = np.array(v)
                if a.dtype.kind in "OS":
                    a = np.asarray([x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)
                                    for x in a.tolist()], dtype=str) if a.size else np.zeros(0, "<U1")
                elif a.dtype.kind == "b":
                    a = a.astype(np.uint8)
                if a.shape[:1] != (n,):
                    raise ValueError(f"{key}.{c} 長度 {a.shape[:1]} 與 t ({n}) 不符")
                cols[c] = a
            groups.setdefault(name, {})[inst] = cols

        merged: dict[str, dict[str, np.ndarray]] = {}
        fmts: dict[str, _Fmt] = {}
        units = units or {}
        for name, parts in groups.items():
            u = dict(units.get(name, {}))
            if None in parts:
                if len(parts) > 1:
                    raise ValueError(f"{name} 不可同時有無 instance 與帶 instance 的表")
                tbl = parts[None]
                fmts[name] = _Fmt(name, [c for c in tbl if c != "t"], units=u, implicit_time=True)
                merged[name] = tbl
                continue
            insts = sorted(parts)
            first = parts[insts[0]]
            ifield = next((c for c in _INST_CANDIDATES if c in first), None)
            cols = [c for c in first if c != "t"]
            if ifield is None:
                ifield = "I"
                cols = ["I"] + cols
                for i in insts:
                    p = parts[i]
                    parts[i] = {"t": p["t"], "I": np.full(p["t"].size, i, np.uint8),
                                **{c: v for c, v in p.items() if c != "t"}}
            u.setdefault(ifield, "instance")
            for i in insts:
                if [c for c in parts[i] if c != "t"] != cols:
                    raise ValueError(f"{name}[{i}] 欄位與 {name}[{insts[0]}] 不一致")
                if not np.all(parts[i][ifield] == i):
                    raise ValueError(f"{name}[{i}] 的 instance 欄 {ifield} 值應全為 {i}")
            cat = {c: np.concatenate([parts[i][c] for i in insts]) for c in ["t"] + cols}
            o = np.argsort(cat["t"], kind="stable")
            merged[name] = {c: v[o] for c, v in cat.items()}
            fmts[name] = _Fmt(name, cols, inst_field=ifield, units=u, implicit_time=True)
        for name, tbl in merged.items():   # 單張表也依時間排序(穩定)
            if not _is_sorted(tbl["t"]):
                o = np.argsort(tbl["t"], kind="stable")
                merged[name] = {c: v[o] for c, v in tbl.items()}
        store = _TableStore(merged, fmts, sorted(merged))
        return cls(store, path, vehicle=vehicle)

    # --- 內部 ---
    def _fmt(self, name: str) -> _Fmt | None:
        return self._store.fmts.get(name)

    def _detect_mav_type(self) -> int | None:
        """VER.BU 優先;否則第一則能判斷機型的 MSG(規則同 DFReader)。"""
        names = self._names
        if "VER" in names and "BU" in self._store.fmts["VER"].columns:
            for bu in self._store.column("VER", "BU").tolist():
                mt = _VER_BUILD.get(int(bu)) if not isinstance(bu, str) else None
                if mt is not None:
                    return mt
        if "MSG" in names and "Message" in self._store.fmts["MSG"].columns:
            for text in self._store.column("MSG", "Message").tolist():
                mt = _mav_type_from_text(str(text))
                if mt is not None:
                    return mt
        return None

    def _instance_values(self, name: str) -> np.ndarray | None:
        """每列的 instance 編號(檔案順序);沒有數值 instance 欄則 None。"""
        if name in self._inst_vals:
            return self._inst_vals[name]
        fm = self._fmt(name)
        vals = None
        if fm is not None and fm.inst_field and fm.inst_field in fm.columns \
                and self._store.kind(name, fm.inst_field) == "num":
            raw = self._store.column(name, fm.inst_field)
            vals = np.asarray(raw).astype(np.int64) if raw.dtype.kind in "biu" else \
                np.where(np.isfinite(raw), raw, -1).astype(np.int64)
        self._inst_vals[name] = vals
        return vals

    def _instances(self, name: str) -> list[int] | None:
        v = self._instance_values(name)
        return None if v is None else sorted(int(x) for x in np.unique(v))

    def _time_of(self, name: str) -> np.ndarray:
        t = self._times.get(name)
        if t is None:
            t = self._times[name] = _readonly(np.asarray(self._store.times(name), dtype=np.float64))
        return t

    def field_format(self, name: str, fld: str) -> str:
        """欄位的 DataFlash 格式字元(來自 .bin 才有;記憶體表格回傳空字串)。"""
        fm = self._fmt(name)
        return fm.char(fld) if fm is not None and fld in fm.columns else ""

    # --- 欄位樹 ---
    def messages(self) -> dict[str, MsgInfo]:
        """有資料的訊息(依名稱排序);不含 FMT / FMTU / UNIT / MULT。"""
        if self._infos is None:
            with self._lock:
                infos = {}
                for name in sorted(self._store.names()):
                    if name in _META:
                        continue
                    fm = self._fmt(name)
                    infos[name] = MsgInfo(name, [c for c in fm.columns if c != "TimeUS"],
                                          self._store.count(name), self._instances(name),
                                          {k: v for k, v in fm.units.items() if v})
                self._infos = infos
        return dict(self._infos)

    def format_names(self) -> list[str]:
        """log 中定義過的全部訊息名稱(含沒有資料的)— 找不到東西時給使用者的線索。"""
        return list(self._store.all_names)

    def instance_field(self, name: str) -> str | None:
        fm = self._fmt(name)
        return fm.inst_field if fm is not None and self._instance_values(name) is not None else None

    def time_field(self, name: str) -> str | None:
        """該訊息的時間欄(TimeUS / TimeMS / None)。"""
        fm = self._fmt(name)
        return fm.time_field() if fm is not None else None

    def field_keys(self) -> list[str]:
        """可繪圖的數值欄位 key(「ATT.Roll」「GPS[1].Spd」),排序後回傳。

        有 instance 的訊息只列 instance 形式;不含時間欄、instance 欄、字串 / 陣列欄,
        也不含 PARM / FILE。
        """
        if self._field_keys is None:
            keys = []
            for name, info in self.messages().items():
                if name in _NO_PLOT:
                    continue
                fm = self._fmt(name)
                skip = {"TimeUS", "TimeMS", fm.inst_field if info.instances else None}
                flds = [f for f in info.fields
                        if f not in skip and self._store.kind(name, f) == "num"]
                if info.instances:
                    keys += [f"{name}[{i}].{f}" for i in info.instances for f in flds]
                else:
                    keys += [f"{name}.{f}" for f in flds]
            self._field_keys = sorted(set(keys))
        return list(self._field_keys)

    def unit(self, key: str) -> str:
        """欄位單位(沒有則空字串),例:unit("ATT.Roll") → "deg"。"""
        try:
            name, _, fld = parse_key(key)
        except ValueError:
            return ""
        fm = self._fmt(name)
        return fm.units.get(fld, "") if fm is not None and fld else ""

    # --- 資料 ---
    def raw_table(self, name: str) -> dict[str, np.ndarray] | None:
        """全部列(所有 instance 合併)、檔案順序;"t" + 欄位(不含 TimeUS)。快取、唯讀。"""
        if name not in self._names:
            return None
        hit = self._raw.get(name)
        if hit is not None:
            return hit
        with self._lock:
            hit = self._raw.get(name)
            if hit is None:
                cols = self._store.load(name)
                tbl = {"t": self._time_of(name)}
                tbl.update((c, _readonly(v)) for c, v in cols.items() if c != "TimeUS")
                self._raw[name] = hit = tbl
        return hit

    def raw_positions(self, name: str) -> np.ndarray | None:
        """raw_table 各列在檔案中的位置(二進位 log 為位元組位移);用於跨型別同時刻排序。"""
        if name not in self._names:
            return None
        return self._store.positions(name)

    def table(self, name: str, instance: int | None = None) -> dict[str, np.ndarray] | None:
        """依時間排序(穩定)的表:"t" + 全部欄位(不含 TimeUS)。快取、唯讀。

        有 instance 的訊息:instance=None 取最小的 instance;沒有 instance 的訊息
        指定 instance 回傳 None。字串欄為 np.ndarray[str]。
        """
        if name not in self._names:
            return None
        insts = self._instances(name)
        if insts:
            if instance is None:
                instance = insts[0]
            elif instance not in insts:
                return None
        elif instance is not None:
            return None
        key = (name, instance)
        hit = self._tables.get(key)
        if hit is not None:
            return hit
        with self._lock:
            hit = self._tables.get(key)
            if hit is not None:
                return hit
            raw = self._raw.get(name)
            if raw is None:
                cols = self._store.load(name)
                raw = {"t": self._time_of(name)}
                raw.update((c, v) for c, v in cols.items() if c != "TimeUS")
            if insts:
                iv = self._instance_values(name)
                for i in insts:   # 一次把所有 instance 都拆好
                    sel = np.flatnonzero(iv == i)
                    self._tables[(name, i)] = self._sorted({c: v[sel] for c, v in raw.items()})
            else:
                self._tables[key] = self._sorted(raw)
            return self._tables[key]

    @staticmethod
    def _sorted(tbl: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        t = tbl["t"]
        if not _is_sorted(t):
            o = np.argsort(t, kind="stable")
            tbl = {c: v[o] for c, v in tbl.items()}
        return {c: _readonly(v) for c, v in tbl.items()}

    def series(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        """「ATT.Roll」→ (t, v) float64;欄位不存在或非數值回傳 None。"""
        try:
            name, inst, fld = parse_key(key)
        except ValueError:
            return None
        if fld is None:
            return None
        tbl = self.table(name, inst)
        if tbl is None or fld not in tbl or fld == "t":
            return None
        col = tbl[fld]
        if col.ndim != 1 or col.dtype.kind not in "biuf":
            return None
        if col.dtype == np.float64:
            return tbl["t"], col
        insts = self._instances(name)
        ck = (name, inst if inst is not None else (insts[0] if insts else None), fld)
        v = self._series.get(ck)
        if v is None:
            v = self._series[ck] = _readonly(col.astype(np.float64))
        return tbl["t"], v

    def has(self, key: str) -> bool:
        """「ATT」「ATT.Roll」「GPS[1].Spd」是否存在且有資料。"""
        try:
            name, inst, fld = parse_key(key)
        except ValueError:
            return False
        info = self.messages().get(name)
        if info is None:
            return False
        if inst is not None and (not info.instances or inst not in info.instances):
            return False
        return fld is None or fld in info.fields

    def t_range(self) -> tuple[float, float]:
        """全部訊息(不含 FMT / FMTU / UNIT / MULT)的時間範圍。"""
        if self._t_range is None:
            lo, hi = math.inf, -math.inf
            for name in self.messages():
                t = self._time_of(name)
                if t.size and np.isfinite(t).any():
                    lo = min(lo, float(np.nanmin(t)))
                    hi = max(hi, float(np.nanmax(t)))
            self._t_range = (lo, hi) if lo <= hi else (0.0, 0.0)
        return self._t_range

    # --- 模式名稱 ---
    def mode_labels(self, tbl: dict[str, np.ndarray] | None = None) -> list[str]:
        """MODE 各列的模式名稱:字串 Mode 轉大寫;否則 ModeNum 依機型查表;
        查不到、或認不出機型時為「MODE n」。tbl 預設為 raw_table("MODE")。"""
        if tbl is None:
            tbl = self.raw_table("MODE")
        if tbl is None:
            return []
        n = tbl["t"].size
        mode = tbl.get("Mode")
        num = tbl.get("ModeNum")
        # 認不出機型(log 缺 MSG / VER)時不套 DFReader 預設的固定翼表:Copter 的
        # 0/5/3 會被叫成 MANUAL/FBWA/TRAINING —— 寧可只顯示編號
        mapping = mode_mapping(self.mav_type) if self.mav_type is not None else None
        out = []
        for i in range(n):
            label = None
            if mode is not None and mode.dtype.kind == "U":
                label = str(mode[i]).upper()
            elif num is not None:
                k = int(num[i])
                label = mapping[k] if mapping is not None and k in mapping else "UNKNOWN"
            elif mode is not None:
                k = int(mode[i])
                from pymavlink import mavutil
                label = mavutil.mode_string_acm(k)
            if not label or label == "UNKNOWN":
                v = num[i] if num is not None else (mode[i] if mode is not None else "?")
                label = f"MODE {v.item() if hasattr(v, 'item') else v}"
            out.append(label)
        return out

    # --- 參數 ---
    def params(self) -> list[Param]:
        """參數(依名稱排序):最後的值、預設值、被設成該值的時間。"""
        if self._params is None:
            tbl = self.raw_table("PARM")
            out: dict[str, Param] = {}
            if tbl is not None and "Name" in tbl and "Value" in tbl:
                names = tbl["Name"].tolist()
                vals = np.asarray(tbl["Value"], dtype=np.float64).tolist()
                dflt = tbl.get("Default")
                dflt = np.asarray(dflt, dtype=np.float64).tolist() if dflt is not None else None
                ts = tbl["t"].tolist()
                for i, (nm, v) in enumerate(zip(names, vals)):
                    p = out.get(nm)
                    if p is None:
                        p = out[nm] = Param(nm, v, math.nan, ts[i])
                    elif not (p.value == v or (math.isnan(p.value) and math.isnan(v))):
                        p.value, p.t = v, ts[i]
                    if dflt is not None and not math.isnan(dflt[i]):
                        p.default = dflt[i]
            self._params = [out[k] for k in sorted(out)]
        return list(self._params)

    def param_values(self) -> dict[str, float]:
        return {p.name: p.value for p in self.params()}

    def param_history(self, name: str) -> list[tuple[float, float]]:
        """某參數在 log 中的 (時間, 值) 變化(只列值有改變的時刻)。"""
        tbl = self.raw_table("PARM")
        if tbl is None or "Name" not in tbl:
            return []
        sel = np.flatnonzero(tbl["Name"] == name)
        out: list[tuple[float, float]] = []
        for i in sel.tolist():
            v = float(tbl["Value"][i])
            if not out or not (out[-1][1] == v or (math.isnan(out[-1][1]) and math.isnan(v))):
                out.append((float(tbl["t"][i]), v))
        return out

    # --- 訊息面板 ---
    def log_messages(self) -> list[LogMessage]:
        """MSG 文字、EV 事件、ERR 錯誤、MODE 模式、ARM 解鎖 / 上鎖,依時間排序(同時刻依檔案順序)。

        EV 10/11(ARMED / DISARMED)只在沒有 ARM 訊息時轉成 arm / disarm,否則略過(避免重複)。
        """
        if self._log_msgs is not None:
            return list(self._log_msgs)
        items: list[tuple[float, int, str, str]] = []

        def add(name, kinds_texts):
            tbl = self.raw_table(name)
            pos = self.raw_positions(name)
            ts = tbl["t"].tolist()
            ps = pos.tolist()
            for i, kt in enumerate(kinds_texts):
                if kt is not None:
                    items.append((ts[i], ps[i], kt[0], kt[1]))

        names = set(self.messages())
        if "MSG" in names and "Message" in self.raw_table("MSG"):
            add("MSG", [(MK_MSG, s) for s in self.raw_table("MSG")["Message"].tolist()])
        has_arm = "ARM" in names and "ArmState" in self.raw_table("ARM")
        if has_arm:
            tbl = self.raw_table("ARM")
            meth = tbl.get("Method")
            forced = tbl.get("Forced")
            kt = []
            for i, st in enumerate(tbl["ArmState"].tolist()):
                armed = bool(int(st)) if st == st else None
                if armed is None:
                    kt.append(None)
                    continue
                kt.append((MK_ARM if armed else MK_DISARM,
                           arm_text(armed, int(meth[i]) if meth is not None else None,
                                    bool(forced[i]) if forced is not None else False)))
            add("ARM", kt)
        if "EV" in names and "Id" in self.raw_table("EV"):
            kt = []
            for eid in self.raw_table("EV")["Id"].tolist():
                eid = int(eid)
                if eid in (EV_ARMED, EV_DISARMED):
                    kt.append(None if has_arm else
                              (MK_ARM if eid == EV_ARMED else MK_DISARM, arm_text(eid == EV_ARMED)))
                else:
                    kt.append((MK_EV, ev_text(eid)))
            add("EV", kt)
        if "ERR" in names:
            tbl = self.raw_table("ERR")
            if "Subsys" in tbl and "ECode" in tbl:
                add("ERR", [(MK_ERR, err_text(s, c))
                            for s, c in zip(tbl["Subsys"].tolist(), tbl["ECode"].tolist())])
        if "MODE" in names:
            add("MODE", [(MK_MODE, s) for s in self.mode_labels()])
        items.sort(key=lambda x: (x[0], x[1]))
        self._log_msgs = [LogMessage(t, k, s) for t, _, k, s in items]
        return list(self._log_msgs)
