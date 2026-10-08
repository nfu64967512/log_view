"""產生 ArduPilot DataFlash .bin(供自我測試 pymavlink 解析路徑)。

格式:每筆訊息 = 0xA3 0x95 <msgid> <payload>;msgid 128 (FMT) 定義其他訊息的
名稱 / 格式字元 / 欄位;FMTU 給每欄單位 / 乘數代碼(UNIT / MULT 定義代碼表),
單位字串中的 '#' 標出 instance 欄(GPS 的 I、BAT 的 Inst…)—— pymavlink 依此拆 instance。
格式字元沿用 ArduPilot 慣例(與 DFReader.FORMAT_TO_STRUCT 相同):
  b/B/h/H/i/I/q/Q 整數、f/d/g 浮點、M=int8(飛行模式編號)、n/N/Z 4/16/64 字元字串、
  a=int16[32]、c/C=int16/uint16 ×0.01、e/E=int32/uint32 ×0.01、L=int32 度×1e-7

兩種寫法:
- LogData 帶 index(core.dflog.LogIndex,模擬資料都有)→ write_index_to_bin:把索引裡
  每張表原樣寫出(向量化),load_bin 讀回的索引與記憶體中的相同(只差格式量化)。
- 沒有 index 的 LogData → 舊寫法:MSG + ORGN + GPS + POS + VISP + 事件。
"""
from __future__ import annotations

import struct

import numpy as np

from .types import EV_ARM, EV_DISARM, EV_MODE, EV_RESET, KIND_GLOBAL, KIND_NED, LogData

_HEAD = b"\xa3\x95"
_FMT_ID = 0x80

# 格式字元 → (struct 碼, 位元組數, 寫入前乘數)
_PACK = {
    "b": ("b", 1, None),
    "B": ("B", 1, None),
    "M": ("b", 1, None),
    "h": ("h", 2, None),
    "H": ("H", 2, None),
    "i": ("i", 4, None),
    "I": ("I", 4, None),
    "q": ("q", 8, None),
    "Q": ("Q", 8, None),
    "f": ("f", 4, None),
    "d": ("d", 8, None),
    "g": ("e", 2, None),
    "c": ("h", 2, 100.0),
    "C": ("H", 2, 100.0),
    "e": ("i", 4, 100.0),
    "E": ("I", 4, 100.0),
    "L": ("i", 4, 1e7),
    "n": ("4s", 4, None),
    "N": ("16s", 16, None),
    "Z": ("64s", 64, None),
    "a": ("64s", 64, None),
}


def _payload_size(fmt: str) -> int:
    return sum(_PACK[ch][1] for ch in fmt)


class DFBinWriter:
    """逐筆寫出(舊寫法 / 測試手工造 log 用)。"""

    def __init__(self):
        self.buf = bytearray()
        self.defs: dict[str, tuple[int, str]] = {}
        self.next_id = 129
        # FMT 自我定義要放最前面;Length 欄位 = 3 位元組 header + payload
        self._emit(_FMT_ID, "BB4s16s64s",
                   [_FMT_ID, 3 + _payload_size("BBnNZ"), b"FMT", b"BBnNZ",
                    b"Type,Length,Name,Format,Columns"])

    def _emit(self, msg_id: int, struct_fmt: str, values: list):
        self.buf += _HEAD + bytes([msg_id]) + struct.pack("<" + struct_fmt, *values)

    def add_message_type(self, name: str, fmt: str, columns: str) -> None:
        msg_id = self.next_id
        self.next_id += 1
        self.defs[name] = (msg_id, fmt)
        self._emit(_FMT_ID, "BB4s16s64s",
                   [msg_id, 3 + _payload_size(fmt),
                    name.encode()[:4], fmt.encode()[:16], columns.encode()[:64]])

    def write(self, name: str, values: list) -> None:
        msg_id, fmt = self.defs[name]
        packed, codes = [], []
        for ch, v in zip(fmt, values):
            code, _, mult = _PACK[ch]
            codes.append(code)
            if isinstance(v, str):
                v = v.encode()
            packed.append(int(round(v * mult)) if mult is not None else v)
        self._emit(msg_id, "".join(codes), packed)

    def raw(self, data: bytes) -> None:
        """直接附加位元組(測試造壞資料用)。"""
        self.buf += data

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(bytes(self.buf))


# ArduCopter 模式名稱 → 編號(MODE.ModeNum;DFReader 依 MSG 判斷機種後反查名稱)
COPTER_MODES = {
    "STABILIZE": 0, "ACRO": 1, "ALT_HOLD": 2, "AUTO": 3, "GUIDED": 4, "LOITER": 5,
    "RTL": 6, "CIRCLE": 7, "LAND": 9, "DRIFT": 11, "SPORT": 13, "FLIP": 14,
    "AUTOTUNE": 15, "POSHOLD": 16, "BRAKE": 17, "THROW": 18, "AVOID_ADSB": 19,
    "GUIDED_NOGPS": 20, "SMART_RTL": 21,
}
_FW_MSG = b"ArduCopter V4.5.7 (fakebin)"
_EV_ARMED, _EV_DISARMED = 10, 11   # ArduPilot LogEvent::ARMED / DISARMED


def write_logdata_to_bin(logdata: LogData, path: str, arm_msg: bool = True) -> None:
    """把(模擬)LogData 寫成 dataflash .bin。

    有 logdata.index → 寫出索引中的全部表格(write_index_to_bin);arm_msg=False 時
    略過 ARM 表(模擬只有 EV 10/11 的舊韌體)。

    沒有 index → 舊寫法(XKF* 來源寫成 XKF1,其餘 NED 來源寫成 VISP):
    ARM/DISARM → 一律寫 EV(Id 10/11),arm_msg=True 時另寫 ARM 訊息
    (新韌體兩者並存,用來測去重);模式切換 → MODE(ArduCopter 編號,未知名稱略過);
    VIO 重置 → 該 NED 來源的 VISP.ResetCnt 累加。
    """
    if getattr(logdata, "index", None) is not None:
        write_index_to_bin(logdata.index, path, skip=() if arm_msg else ("ARM",))
        return
    _write_legacy(logdata, path, arm_msg)


def _write_legacy(logdata: LogData, path: str, arm_msg: bool = True) -> None:
    w = DFBinWriter()
    w.add_message_type("MSG", "QZ", "TimeUS,Message")
    w.add_message_type("ORGN", "QBLLe", "TimeUS,Type,Lat,Lng,Alt")
    # GPS 需帶 GWk/GMS:DFReader 靠這兩欄建立 log 時基,缺了會逐筆解析失敗
    w.add_message_type("GPS", "QBBHILLe", "TimeUS,I,Status,GWk,GMS,Lat,Lng,Alt")
    w.add_message_type("POS", "QLLf", "TimeUS,Lat,Lng,Alt")
    w.add_message_type("VISP", "QfffB", "TimeUS,PX,PY,PZ,ResetCnt")
    w.add_message_type("EV", "QB", "TimeUS,Id")
    w.add_message_type("MODE", "QMBB", "TimeUS,Mode,ModeNum,Rsn")
    if arm_msg:
        w.add_message_type("ARM", "QBIBB", "TimeUS,ArmState,ArmChecks,Forced,Method")
    if any(s.kind == KIND_NED and s.name.upper().startswith("XKF") for s in logdata.sources.values()):
        w.add_message_type("XKF1", "QBfff", "TimeUS,C,PN,PE,PD")

    records: list[tuple[int, str, list]] = []
    t0 = logdata.t_base
    gps_week, gps_ms0 = 2320, 300_000  # 任意有效 GPS 週/週內毫秒
    if logdata.origin is not None:
        lat0, lng0, alt0 = logdata.origin
        records.append((int(t0 * 1e6), "ORGN", [int(t0 * 1e6), 0, lat0, lng0, alt0]))

    for e in logdata.events:
        us = int(e.t * 1e6)
        if e.kind in (EV_ARM, EV_DISARM):
            armed = e.kind == EV_ARM
            if arm_msg:
                records.append((us, "ARM", [us, 1 if armed else 0, 0, 0, 0]))
            records.append((us, "EV", [us, _EV_ARMED if armed else _EV_DISARMED]))
        elif e.kind == EV_MODE and e.label.upper() in COPTER_MODES:
            num = COPTER_MODES[e.label.upper()]
            records.append((us, "MODE", [us, num, num, 0]))

    for src in logdata.sources.values():
        if src.kind == KIND_GLOBAL:
            name = "GPS" if src.name.upper().startswith("GPS") else "POS"
            for k in range(src.n):
                us = int(src.t[k] * 1e6)
                if name == "GPS":
                    gms = gps_ms0 + int(round((src.t[k] - t0) * 1000))
                    records.append((us, "GPS",
                                    [us, 0, 3, gps_week, gms,
                                     src.lat[k], src.lng[k], src.alt[k]]))
                else:
                    records.append((us, "POS", [us, src.lat[k], src.lng[k], src.alt[k]]))
        elif src.kind == KIND_NED and src.name.upper().startswith("XKF"):
            for k in range(src.n):
                us = int(src.t[k] * 1e6)
                records.append((us, "XKF1", [us, 0, src.ned[k, 0], src.ned[k, 1], src.ned[k, 2]]))
        elif src.kind == KIND_NED:
            # ResetCnt = 此來源在該樣本時刻(含)之前發生過幾次重置
            rst_t = np.sort([e.t for e in logdata.events
                             if e.kind == EV_RESET and e.label == src.name])
            rst = np.searchsorted(rst_t, src.t, side="right")
            for k in range(src.n):
                us = int(src.t[k] * 1e6)
                records.append((us, "VISP",
                                [us, src.ned[k, 0], src.ned[k, 1], src.ned[k, 2],
                                 int(rst[k]) % 256]))

    records.sort(key=lambda r: r[0])   # 穩定排序:同時刻維持寫入順序
    # 韌體版本 MSG 放最前面:DFReader 靠它判斷機種,才能把 MODE 編號反查成名稱
    t_msg = min(records[0][0], int(t0 * 1e6)) if records else int(t0 * 1e6)
    w.write("MSG", [t_msg, _FW_MSG])
    for _, name, vals in records:
        w.write(name, vals)
    w.save(path)


# ======================================================================
# 索引表格 → .bin(向量化)
# ======================================================================

_FMT_DTYPE = {
    "a": ("<i2", (32,)), "b": "i1", "B": "u1", "g": "<f2", "h": "<i2", "H": "<u2",
    "i": "<i4", "I": "<u4", "f": "<f4", "n": "S4", "N": "S16", "Z": "S64",
    "c": "<i2", "C": "<u2", "e": "<i4", "E": "<u4", "L": "<i4", "d": "<f8",
    "M": "i1", "q": "<i8", "Q": "<u8",
}
_FMT_MULT = {"c": 0.01, "C": 0.01, "e": 0.01, "E": 0.01, "L": 1.0e-7}

# ArduPilot UNIT 代碼表(AP_Logger/LogStructure.h)
UNIT_IDS = {
    "-": "", "?": "UNKNOWN", "A": "A", "a": "Ah", "d": "deg", "b": "B", "B": "B/s",
    "k": "deg/s", "D": "deglatitude", "e": "deg/s/s", "E": "rad/s", "G": "Gauss",
    "h": "degheading", "i": "A.s", "J": "W.s", "l": "l", "L": "rad/s/s", "m": "m",
    "n": "m/s", "o": "m/s/s", "O": "degC", "%": "%", "S": "satellites", "s": "s",
    "q": "rpm", "r": "rad", "U": "deglongitude", "u": "ppm", "v": "V", "P": "Pa",
    "w": "Ohm", "W": "Watt", "X": "W.h", "y": "l/s", "Y": "us", "z": "Hz", "#": "instance",
}
# MULT 代碼表
MULT_IDS = {"-": 0.0, "?": 1.0, "2": 1e2, "1": 1e1, "0": 1e0, "A": 1e-1, "B": 1e-2,
            "C": 1e-3, "D": 1e-4, "E": 1e-5, "F": 1e-6, "G": 1e-7, "I": 1e-9,
            "!": 3.6, "/": 3600.0}
_PREFIX_MULT = {"d": "A", "c": "B", "m": "C", "µ": "F", "n": "I"}   # DFReader MULT_TO_PREFIX 的反查
_LABEL_TO_UNIT = {v: k for k, v in UNIT_IDS.items() if k not in "-?"}

# 已知欄位的格式字元(仿真實 ArduPilot 格式;其餘依 dtype 推斷)
FMT_OVERRIDE: dict[tuple[str, str], str] = {
    ("GPS", "HDop"): "c", ("GPS", "Alt"): "e", ("ORGN", "Alt"): "e",
    ("XKF1", "Roll"): "c", ("XKF1", "Pitch"): "c", ("XKF1", "Yaw"): "C",
    ("XKF1", "GX"): "c", ("XKF1", "GY"): "c", ("XKF1", "GZ"): "c", ("XKF1", "OH"): "e",
    ("BAT", "Temp"): "c", ("MODE", "Mode"): "M",
    ("MSG", "Message"): "Z", ("PARM", "Name"): "N",
}
_INT_CHARS = {np.dtype(k): v for k, v in (("i1", "b"), ("u1", "B"), ("i2", "h"), ("u2", "H"),
                                           ("i4", "i"), ("u4", "I"), ("i8", "q"), ("u8", "Q"))}


def _infer_char(msg: str, col: str, a: np.ndarray, known: str = "") -> str:
    if known:
        return known
    ch = FMT_OVERRIDE.get((msg, col))
    if ch:
        return ch
    if a.ndim == 2:
        return "a"
    k = a.dtype.kind
    if k in "US":
        lens = np.char.str_len(np.char.encode(a.astype(str), "utf-8")) if a.size else np.zeros(1)
        return "Z" if int(np.max(lens, initial=0)) > 16 else "N"
    if k == "f":
        return "L" if col in ("Lat", "Lng", "Lon") else "f"
    if k == "b":
        return "B"
    return _INT_CHARS.get(a.dtype.newbyteorder("="), "q" if k == "i" else "Q")


def _to_raw(a: np.ndarray, ch: str, n: int) -> np.ndarray:
    """欄位值 → 該格式字元的原始儲存值(乘數反算、四捨五入、夾到型別範圍)。"""
    spec = np.dtype(_FMT_DTYPE[ch])
    if ch in "nNZ":
        if a.dtype.kind == "S":
            return a
        return np.char.encode(np.asarray(a, dtype=str), "utf-8") if n else np.zeros(0, "S1")
    if ch == "a":
        return np.asarray(a, dtype=np.int16).reshape(n, 32)
    mult = _FMT_MULT.get(ch)
    if mult is not None:
        v = np.asarray(a, dtype=np.float64) / mult
    elif ch in "fgd":
        return np.asarray(a, dtype=np.float64).astype(spec)
    else:
        v = np.asarray(a)
        if v.dtype.kind not in "f":
            info = np.iinfo(spec)
            return np.clip(v.astype(np.int64) if v.dtype != np.uint64 else v,
                           info.min, info.max).astype(spec)
        v = v.astype(np.float64)
    info = np.iinfo(spec)
    v = np.where(np.isfinite(v), np.rint(v), 0.0)
    return np.clip(v, info.min, info.max).astype(spec)


def _record_dtype(type_id: int, chars: str) -> np.dtype:
    names = ["h0", "h1", "id"] + [f"f{i}" for i in range(len(chars))]
    formats = ["u1", "u1", "u1"] + [_FMT_DTYPE[c] for c in chars]
    offs, o = [], 0
    for f in formats:
        offs.append(o)
        o += np.dtype(f).itemsize
    return np.dtype({"names": names, "formats": formats, "offsets": offs, "itemsize": o})


def _encode(type_id: int, chars: str, cols: list[np.ndarray], n: int) -> np.ndarray:
    """一型別全部列 → (n, 長度) uint8。"""
    dt = _record_dtype(type_id, chars)
    rec = np.zeros(n, dt)
    rec["h0"], rec["h1"], rec["id"] = 0xA3, 0x95, type_id
    for i, (ch, a) in enumerate(zip(chars, cols)):
        rec[f"f{i}"] = _to_raw(a, ch, n)
    return rec.view(np.uint8).reshape(n, dt.itemsize)


class _UnitTable:
    """單位字串 → (UNIT 代碼, MULT 代碼);未知字串動態配一個未用的代碼。"""

    def __init__(self):
        self.units = dict(UNIT_IDS)
        self.lookup = dict(_LABEL_TO_UNIT)
        self._free = [c for c in "jgKHFxRTVQ@&$*+=" if c not in self.units]

    def ids(self, unit: str, has_fmt_mult: bool) -> tuple[str, str]:
        if not unit:
            return "-", "-"
        if unit in self.lookup:
            return self.lookup[unit], "-"
        if not has_fmt_mult and unit[0] in _PREFIX_MULT and unit[1:] in self.lookup:
            return self.lookup[unit[1:]], _PREFIX_MULT[unit[0]]
        if not self._free:
            return "?", "-"
        c = self._free.pop(0)
        self.units[c] = unit
        self.lookup[unit] = c
        return c, "-"


def write_index_to_bin(index, path: str, *, skip=(), fmt_override: dict | None = None) -> None:
    """把 LogIndex 的全部表格寫成 .bin:FMT → UNIT / MULT / FMTU → 依時間交錯的資料列。

    有 instance 的訊息(GPS[0]、GPS[1]…)合併成一個型別,FMTU 單位字串在 instance 欄
    標 '#'。格式字元:fmt_override {(訊息, 欄位): 字元} 優先;索引若來自 .bin 沿用原字元;
    否則 FMT_OVERRIDE / 依 dtype 推斷(浮點預設 f = float32)。
    同時刻的列依表格順序、再依列序寫出。skip:不寫的訊息名稱。
    """
    user = dict(fmt_override or {})
    known = getattr(index, "field_format", lambda *_: "")
    units = _UnitTable()

    types = []   # (name, chars, columns, unit_ids, mult_ids, t_us, cols)
    for name, info in index.messages().items():
        if name in skip:
            continue
        if info.instances:
            parts = [index.table(name, i) for i in info.instances]
            t = np.concatenate([p["t"] for p in parts])
            o = np.argsort(t, kind="stable")
            tbl = {c: np.concatenate([p[c] for p in parts])[o] for c in ["t"] + info.fields}
        else:
            tbl = index.table(name)
        ifield = index.instance_field(name) if info.instances else None
        cols = [tbl[c] for c in info.fields]
        chars = "".join(user.get((name, c)) or _infer_char(name, c, a, known(name, c))
                        for c, a in zip(info.fields, cols))
        uids, mids = ["s"], ["F"]                          # TimeUS:µs
        for c, ch in zip(info.fields, chars):
            if c == ifield:
                u, m = "#", "-"
            else:
                u, m = units.ids(info.units.get(c, ""), ch in _FMT_MULT)
            uids.append(u)
            mids.append(m)
        fmt = "Q" + chars
        colstr = ",".join(["TimeUS"] + info.fields)
        if len(name) > 4 or len(fmt) > 16 or len(colstr) > 64:
            raise ValueError(f"{name}:名稱 ≤ 4 字、格式 ≤ 16 字元、欄位字串 ≤ 64 字元"
                             f"(得 {len(name)} / {len(fmt)} / {len(colstr)})")
        t_us = np.rint(np.asarray(tbl["t"], dtype=np.float64) * 1e6)
        t_us = np.clip(np.where(np.isfinite(t_us), t_us, 0), 0, None).astype(np.uint64)
        types.append((name, chars, colstr, "".join(uids), "".join(mids), t_us, cols))

    # 同時刻的列:依各型別在原 log(或 from_tables 的表格順序)中首次出現的位置排先後,
    # 例如開機時韌體版本 MSG 要排在同一時刻的 MODE 之前
    def first_pos(name: str) -> int:
        pos = index.raw_positions(name) if hasattr(index, "raw_positions") else None
        return int(np.min(pos)) if pos is not None and pos.size else 1 << 62
    types.sort(key=lambda ty: first_pos(ty[0]))

    if len(types) + 132 > 255:
        raise ValueError(f"訊息型別太多({len(types)}),超過 DataFlash 的 255 個 id")
    t_first = min((int(ty[5].min()) for ty in types if ty[5].size), default=0)

    w = DFBinWriter()
    w.add_message_type("UNIT", "QbZ", "TimeUS,Id,Label")
    w.add_message_type("MULT", "Qbd", "TimeUS,Id,Mult")
    w.add_message_type("FMTU", "QBNN", "TimeUS,FmtType,UnitIds,MultIds")
    w.next_id = 132
    type_ids = []
    for name, chars, colstr, *_ in types:
        type_ids.append(w.next_id)
        w.add_message_type(name, "Q" + chars, colstr)
    for c, label in units.units.items():
        w.write("UNIT", [t_first, ord(c), label])
    for c, m in MULT_IDS.items():
        w.write("MULT", [t_first, ord(c), m])
    for tid, (_, _, _, uids, mids, _, _) in zip(type_ids, types):
        w.write("FMTU", [t_first, tid, uids, mids])

    # 資料列:每型別一次編碼,再依 (時間, 型別順序, 列序) 交錯
    blobs, keys_t, keys_k, keys_j, lens = [], [], [], [], []
    for k, (tid, (name, chars, _, _, _, t_us, cols)) in enumerate(zip(type_ids, types)):
        n = t_us.size
        blob = _encode(tid, "Q" + chars, [t_us] + cols, n)
        blobs.append(memoryview(blob.tobytes()))
        lens.append(blob.shape[1])
        keys_t.append(t_us)
        keys_k.append(np.full(n, k, np.int64))
        keys_j.append(np.arange(n, dtype=np.int64))
    if blobs:
        tt, kk, jj = (np.concatenate(x) for x in (keys_t, keys_k, keys_j))
        order = np.lexsort((jj, kk, tt))
        ks, js = kk[order].tolist(), jj[order].tolist()
        parts = [blobs[k][j * lens[k]:(j + 1) * lens[k]] for k, j in zip(ks, js)]
        data = b"".join(parts)
    else:
        data = b""
    with open(path, "wb") as f:
        f.write(bytes(w.buf))
        f.write(data)
