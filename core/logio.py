"""ArduPilot DataFlash .bin 解析:位置來源 + 時間軸事件 + 任意欄位索引。

建在 core.dflog.LogIndex 之上(向量化:整型別一次解碼,不逐筆建 DFMessage),
數百 MB 的開機到關機 log 也只要幾秒;最慢的是 pymavlink 建位移索引那一步。

位置來源:掃描候選訊息型別,凡欄位含 Lat/Lng(經緯度)或 PX/PY/PZ、PN/PE/PD
(局部 NED)者皆收為可選軌跡來源;依 I / C / Instance 欄拆 instance(GPS[1] 常見為
VIO 注入的第二顆 GPS);ORGN 取為 ENU 原點。

時間軸事件:ARM/DISARM(ARM 訊息,沒有才退回 EV 10/11)、飛行模式(MODE)、
VIO 重置(位置訊息的重置計數欄位改變)、其他 EV 事件(EV_EVENT)、ERR 錯誤(EV_ERROR)。

舊版逐筆 recv_match 掃描保留為 _legacy_load_bin(只給測試比對新舊結果一致)。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .dflog import EV_ARMED, EV_DISARMED, LogIndex, err_text, ev_text
from .types import (EV_ARM, EV_DISARM, EV_ERROR, EV_EVENT, EV_MODE, EV_RESET, KIND_GLOBAL,
                    KIND_NED, LogData, LogEvent, TrajSource)

# 想掃的訊息(涵蓋常見位置來源;VISO 記的是增量位移,不是位置,故不含)
#   GPS/GPS2/GPSB  衛星定位(VIO 若以 GPS_INPUT 注入會出現在 instance 1)
#   POS            EKF 融合後的位置
#   VISP           VISION_POSITION_ESTIMATE(ROS2 餵進來的 vision position)
#   XKF1/NKF1      EKF3/EKF2 的內部位置估計
#   AHR2           次要 AHRS 解
CANDIDATE_TYPES = ["GPS", "GPS2", "GPSB", "POS", "VISP", "XKF1", "NKF1", "AHR2", "ORGN"]

# 時間軸事件
#   ARM   解鎖狀態 ArmState(新韌體;與 EV 重複記錄時以 ARM 為準)
#   EV    事件 Id 10 = ARMED、11 = DISARMED(舊韌體只有這個);其他 Id → EV_EVENT
#   MODE  飛行模式;名稱依 MSG / VER 判斷出的機種查表
#   ERR   子系統錯誤 / 失效保護 → EV_ERROR
EVENT_TYPES = ["ARM", "EV", "MODE"]
_EVENT_SET = frozenset(EVENT_TYPES)
_SCAN_TYPES = CANDIDATE_TYPES + EVENT_TYPES

# 位置訊息中的重置計數欄位(VISP:舊韌體 ResetCnt、新韌體 Rst)
_RESET_FIELDS = ("ResetCnt", "Rst")
_INSTANCE_FIELDS = ("I", "C", "Instance")
_EV_ARMED, _EV_DISARMED = EV_ARMED, EV_DISARMED

_MIN_SAMPLES = 10


def _fix_latlng(v: np.ndarray) -> np.ndarray:
    """保險:某些工具鏈輸出未除 1e7 的原始整數經緯度。"""
    if v.size and np.nanmax(np.abs(v)) > 400.0:
        return v * 1e-7
    return v


def load_bin(path: str, progress_cb=None) -> LogData:
    """解析 .bin,回傳 LogData(LogData.index = LogIndex)。

    progress_cb(percent:int 0..100) 供 UI 進度條:0~90 為建位移索引(大檔最慢的部分)。
    """
    last = [-1]

    def report(p: float) -> None:
        p = int(max(0, min(100, p)))
        if progress_cb is not None and p != last[0]:
            last[0] = p
            progress_cb(p)

    index = LogIndex.open(path, progress_cb=(lambda p: report(0.9 * p)) if progress_cb else None)
    report(90)
    logdata = logdata_from_index(index, path)
    report(100)
    return logdata


# ---------------- 由索引組 LogData(向量化,與舊版逐筆掃描結果相同) ----------------

def _ints(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐元素 int()(往零截斷);回傳 (整數陣列, 是否有效)。NaN / inf / 非數字字串視為無效。"""
    a = np.asarray(a)
    if a.dtype.kind in "biu":
        return a.astype(np.int64), np.ones(a.shape, bool)
    if a.dtype.kind == "f":
        ok = np.isfinite(a)
        return np.where(ok, np.trunc(np.where(ok, a, 0.0)), 0).astype(np.int64), ok
    out = np.zeros(a.shape, np.int64)
    ok = np.zeros(a.shape, bool)
    for i, s in enumerate(a.tolist()):
        try:
            out[i] = int(s)
            ok[i] = True
        except (TypeError, ValueError, OverflowError):
            pass
    return out, ok


def _timed(index: LogIndex, name: str, present: set[str]) -> dict[str, np.ndarray] | None:
    """有 TimeUS 的訊息才收(與舊版一致:沒有 TimeUS 的不計)。"""
    if name not in present or index.time_field(name) != "TimeUS":
        return None
    return index.raw_table(name)


def index_events(index: LogIndex) -> list[LogEvent]:
    """EV(10/11 以外)→ EV_EVENT、ERR → EV_ERROR 事件(依時間排序)。模擬資料也用這個,
    確保模擬的 events 與 load_bin(fakebin(模擬)) 一致。"""
    present = set(index.messages())
    out: list[LogEvent] = []
    tbl = _timed(index, "EV", present)
    if tbl is not None and "Id" in tbl:
        ids, ok = _ints(tbl["Id"])
        for t, i, g in zip(tbl["t"].tolist(), ids.tolist(), ok.tolist()):
            if g and i not in (_EV_ARMED, _EV_DISARMED):
                out.append(LogEvent(t, EV_EVENT, ev_text(i)))
    tbl = _timed(index, "ERR", present)
    if tbl is not None and "Subsys" in tbl and "ECode" in tbl:
        ss, ok1 = _ints(tbl["Subsys"])
        ec, ok2 = _ints(tbl["ECode"])
        for t, s, c, g in zip(tbl["t"].tolist(), ss.tolist(), ec.tolist(), (ok1 & ok2).tolist()):
            if g:
                out.append(LogEvent(t, EV_ERROR, err_text(s, c)))
    out.sort(key=lambda e: e.t)
    return out


def _origin(index: LogIndex, present: set[str]):
    """ORGN:Type 0(EKF 原點)優先,否則最後一筆。回傳 (origin_ekf, origin_any)。"""
    tbl = index.raw_table("ORGN") if "ORGN" in present else None
    if tbl is None or not all(c in tbl for c in ("Lat", "Lng", "Alt")) or not tbl["t"].size:
        return None, None

    def at(i: int):
        o = (float(tbl["Lat"][i]), float(tbl["Lng"][i]), float(tbl["Alt"][i]))
        return (o[0] * 1e-7 if abs(o[0]) > 400 else o[0],
                o[1] * 1e-7 if abs(o[1]) > 400 else o[1], o[2])

    typ, ok = _ints(tbl["Type"]) if "Type" in tbl else (np.zeros(tbl["t"].size, np.int64),
                                                         np.ones(tbl["t"].size, bool))
    rows = np.flatnonzero(ok)
    if not rows.size:
        return None, None
    ekf = rows[typ[rows] == 0]
    return (at(int(ekf[-1])) if ekf.size else None), at(int(rows[-1]))


def logdata_from_index(index: LogIndex, path: str = "") -> LogData:
    """LogIndex → LogData(位置來源、原點、事件、註記)。與舊版逐筆掃描結果一致。"""
    present = set(index.messages())
    origin_ekf, origin_any = _origin(index, present)

    # --- 候選位置訊息:依 instance 分桶(檔案順序),順便偵測重置計數改變 ---
    buckets: dict[tuple[str, int | None], dict[str, np.ndarray]] = {}
    reset_hits: list[tuple[int, float, tuple[str, int | None]]] = []   # (檔案位置, t, key)
    for name in CANDIDATE_TYPES:
        if name == "ORGN":
            continue
        tbl = _timed(index, name, present)
        if tbl is None:
            continue
        t = tbl["t"]
        pos = index.raw_positions(name)
        inst_col = next((f for f in _INSTANCE_FIELDS if f in tbl), None)
        if inst_col is None:
            groups = [(None, np.arange(t.size))]
        else:
            iv, iok = _ints(tbl[inst_col])
            codes = np.where(iok, iv, np.iinfo(np.int64).min)   # 無效 → None 桶
            _, first, inv = np.unique(codes, return_index=True, return_inverse=True)
            groups = []
            for u in np.argsort(first, kind="stable"):            # 依首次出現順序
                rows = np.flatnonzero(inv.reshape(-1) == u)
                k = int(codes[rows[0]])
                groups.append((None if k == np.iinfo(np.int64).min else k, rows))
        rf = next((f for f in _RESET_FIELDS if f in tbl), None)
        if "Lat" in tbl and ("Lng" in tbl or "Lon" in tbl):
            kind = KIND_GLOBAL
        elif "PX" in tbl:
            kind, cols = KIND_NED, ("PX", "PY", "PZ")
        elif "PN" in tbl:
            kind, cols = KIND_NED, ("PN", "PE", "PD")
        else:
            kind = None
        for inst, rows in groups:
            key = (name, inst)
            if rf is not None:
                rc, ok = _ints(tbl[rf][rows])
                r = rows[ok]
                v = rc[ok]
                for j in (np.flatnonzero(v[1:] != v[:-1]) + 1).tolist():   # 第一筆不算
                    reset_hits.append((int(pos[r[j]]), float(t[r[j]]), key))
            b: dict[str, np.ndarray] = {"t": t[rows].astype(float)}
            if kind == KIND_GLOBAL:
                b["lat"] = np.asarray(tbl["Lat"][rows], dtype=float)
                b["lng"] = np.asarray(tbl["Lng" if "Lng" in tbl else "Lon"][rows], dtype=float)
                b["alt"] = (np.asarray(tbl["Alt"][rows], dtype=float) if "Alt" in tbl
                            else np.zeros(rows.size))
                if "Status" in tbl:
                    b["status"] = _ints(tbl["Status"][rows])[0]
            elif kind == KIND_NED:
                b["ned"] = np.column_stack([np.asarray(tbl[c][rows], dtype=float) for c in cols])
            else:
                b["t"] = np.zeros(0)     # 不是位置訊息
            buckets[key] = b

    # 每個訊息名稱有哪些 instance(決定顯示名稱)
    inst_map: dict[str, list] = defaultdict(list)
    for (name, inst) in buckets:
        inst_map[name].append(inst)

    sources: dict[str, TrajSource] = {}
    key_label: dict[tuple[str, int | None], str] = {}
    notes: list[str] = []
    for (name, inst), b in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        t = b["t"]
        if t.size == 0:
            continue
        label = name
        if len(inst_map[name]) > 1 or (inst not in (None, 0)):
            label = f"{name}[{inst}]"
            if name == "GPS" and inst == 1:
                label = "GPS[1] (GPS2)"

        if "lat" in b:
            kind = KIND_GLOBAL
            cols = [_fix_latlng(b["lat"]), _fix_latlng(b["lng"]), b["alt"]]
            if "status" in b:
                ok = b["status"] >= 3
                dropped = int((~ok).sum())
                if dropped and ok.sum() >= _MIN_SAMPLES:
                    t, cols = t[ok], [c[ok] for c in cols]
                    notes.append(f"{label}:濾除 {dropped} 筆非 3D-fix 樣本")
            lat, lng, alt = cols
            good = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lng) & np.isfinite(alt)
            ned = None
        else:
            kind = KIND_NED
            ned = b["ned"]
            lat = lng = alt = None
            good = np.isfinite(t) & np.all(np.isfinite(ned), axis=1)

        # 有效值 → 時間排序(穩定)→ 去除重複時間戳(內插要求嚴格遞增,保留檔案中先出現者)
        sel = np.flatnonzero(good)
        sel = sel[np.argsort(t[sel], kind="stable")]
        if sel.size:
            sel = sel[np.concatenate([[True], np.diff(t[sel]) > 1e-9])]
        t = t[sel]
        if kind == KIND_GLOBAL:
            lat, lng, alt = lat[sel], lng[sel], alt[sel]
        else:
            ned = ned[sel]

        if t.size < _MIN_SAMPLES:
            notes.append(f"{label}:樣本不足({t.size} 筆),略過")
            continue
        sources[label] = TrajSource(label, kind, t, lat=lat, lng=lng, alt=alt, ned=ned)
        key_label[(name, inst)] = label

    origin = origin_ekf or origin_any
    origin_desc = "ORGN(EKF 原點)" if origin_ekf else ("ORGN" if origin_any else "")
    if not sources:
        raise ValueError(
            "log 中找不到任何可用的位置訊息(找過 "
            + "、".join(CANDIDATE_TYPES) + ")。\n\n"
            "這份 log 實際包含的訊息型別:\n" + _format_names(index.format_names()) + "\n\n"
            "若你的 VIO 位置記在別的訊息名稱下,請把該名稱告知或加進 "
            "core/logio.py 的 CANDIDATE_TYPES。")

    # --- 事件 ---
    arm_evs: list[LogEvent] = []
    tbl = _timed(index, "ARM", present)
    if tbl is not None and "ArmState" in tbl:
        st, ok = _ints(tbl["ArmState"])
        arm_evs = [LogEvent(t, EV_ARM if s else EV_DISARM)
                   for t, s, g in zip(tbl["t"].tolist(), st.tolist(), ok.tolist()) if g]
    ev_evs: list[LogEvent] = []
    tbl = _timed(index, "EV", present)
    if tbl is not None and "Id" in tbl:
        ids, ok = _ints(tbl["Id"])
        ev_evs = [LogEvent(t, EV_ARM if i == _EV_ARMED else EV_DISARM)
                  for t, i, g in zip(tbl["t"].tolist(), ids.tolist(), ok.tolist())
                  if g and i in (_EV_ARMED, _EV_DISARMED)]
    mode_evs: list[LogEvent] = []
    tbl = _timed(index, "MODE", present)
    if tbl is not None:
        mode_evs = [LogEvent(t, EV_MODE, lb) for t, lb in zip(tbl["t"].tolist(), index.mode_labels(tbl))]
    reset_hits.sort(key=lambda h: h[0])     # 檔案順序(同時刻的重置依出現先後)
    # ARM 與 EV 常同時記錄同一次解鎖:有 ARM 訊息就只用 ARM,避免重複計算
    events = (arm_evs or ev_evs) + mode_evs
    events += [LogEvent(t, EV_RESET, key_label[key]) for _, t, key in reset_hits if key in key_label]
    events += index_events(index)
    events.sort(key=lambda e: e.t)

    logdata = LogData(sources=sources, origin=origin, origin_desc=origin_desc,
                      path=path, notes=notes, events=events, index=index)
    n_seg = len(logdata.segments)
    if n_seg >= 2:
        notes.append(f"這份 log 有 {n_seg} 個 ARM 飛行段(開機到關機一份 log)— 可在時間軸選段分析")
    if mode_evs and index.mav_type is None:
        notes.append("無法判斷機型(log 缺 MSG / VER),飛行模式只顯示編號")
    return logdata


def _format_names(names: list[str], per_line: int = 12) -> str:
    names = sorted(set(names))
    if not names:
        return "(無)"
    return "\n".join("  " + "  ".join(names[i:i + per_line])
                     for i in range(0, len(names), per_line))


# ======================================================================
# 舊版:逐筆 recv_match 掃描(原樣保留,只給 tests/test_dflog.py 比對新舊結果)
# ======================================================================

def _instance_of(m) -> int | None:
    for f in ("I", "C", "Instance"):
        if hasattr(m, f):
            try:
                return int(getattr(m, f))
            except (TypeError, ValueError):
                return None
    return None


def _legacy_load_bin(path: str, progress_cb=None) -> LogData:
    """舊版 load_bin(逐筆 recv_match)。只給測試用。"""
    from pymavlink import mavutil  # 延遲載入,加快程式啟動

    mlog = mavutil.mavlink_connection(path)
    try:
        return _legacy_scan(mlog, path, progress_cb)
    finally:
        try:
            mlog.close()   # Windows 上不關會鎖住檔案
        except Exception:  # noqa: BLE001
            pass


def _legacy_scan(mlog, path: str, progress_cb) -> LogData:
    buckets: dict[tuple[str, int | None], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    origin_ekf = None      # ORGN Type==0(EKF 原點,優先)
    origin_any = None
    count = 0
    arm_evs: list[LogEvent] = []     # 來自 ARM 訊息
    ev_evs: list[LogEvent] = []      # 來自 EV 10/11(沒有 ARM 訊息時才用)
    mode_evs: list[LogEvent] = []
    reset_field: dict[str, str | None] = {}            # 訊息名稱 → 重置計數欄位
    reset_last: dict[tuple[str, int | None], int] = {}
    reset_hits: list[tuple[tuple[str, int | None], float]] = []

    while True:
        m = mlog.recv_match(type=_SCAN_TYPES)
        if m is None:
            break
        name = m.get_type()
        count += 1
        if progress_cb is not None and count % 5000 == 0:
            pct = getattr(mlog, "percent", None)
            if pct is not None:
                progress_cb(int(pct))

        if name in _EVENT_SET:
            ev = _legacy_parse_event(m, name, mlog)
            if ev is not None:
                (arm_evs if name == "ARM" else ev_evs if name == "EV" else mode_evs).append(ev)
            continue

        if name == "ORGN":
            try:
                o = (float(m.Lat), float(m.Lng), float(m.Alt))
                o = (o[0] * 1e-7 if abs(o[0]) > 400 else o[0],
                     o[1] * 1e-7 if abs(o[1]) > 400 else o[1], o[2])
                if int(getattr(m, "Type", 0)) == 0:
                    origin_ekf = o
                origin_any = o
            except (AttributeError, TypeError, ValueError):
                pass
            continue

        t_us = getattr(m, "TimeUS", None)
        if t_us is None:
            continue
        key = (name, _instance_of(m))
        b = buckets[key]
        b["t"].append(float(t_us) * 1e-6)

        rf = reset_field.get(name, "")
        if rf == "":
            fields = m.get_fieldnames()
            rf = reset_field[name] = next((f for f in _RESET_FIELDS if f in fields), None)
        if rf is not None:
            try:
                rc = int(getattr(m, rf))
            except (AttributeError, TypeError, ValueError):
                rc = None
            if rc is not None:
                prev = reset_last.get(key)
                if prev is not None and rc != prev:      # 第一筆不算
                    reset_hits.append((key, float(t_us) * 1e-6))
                reset_last[key] = rc

        if hasattr(m, "Lat") and (hasattr(m, "Lng") or hasattr(m, "Lon")):
            b["lat"].append(float(m.Lat))
            b["lng"].append(float(getattr(m, "Lng", getattr(m, "Lon", 0.0))))
            b["alt"].append(float(getattr(m, "Alt", 0.0)))
            if hasattr(m, "Status"):
                b["status"].append(int(m.Status))
        elif hasattr(m, "PX"):
            b["px"].append(float(m.PX)); b["py"].append(float(m.PY)); b["pz"].append(float(m.PZ))
        elif hasattr(m, "PN"):
            b["px"].append(float(m.PN)); b["py"].append(float(m.PE)); b["pz"].append(float(m.PD))
        else:
            b["t"].pop()  # 不是位置訊息,收回

    if progress_cb is not None:
        progress_cb(100)

    # 每個訊息名稱有哪些 instance(決定顯示名稱)
    inst_map: dict[str, list] = defaultdict(list)
    for (name, inst), _ in buckets.items():
        inst_map[name].append(inst)

    sources: dict[str, TrajSource] = {}
    key_label: dict[tuple[str, int | None], str] = {}   # 實際收下的來源 → 顯示名稱
    notes: list[str] = []
    for (name, inst), b in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        t = np.asarray(b["t"], dtype=float)
        if t.size == 0:
            continue

        label = name
        if len(inst_map[name]) > 1 or (inst not in (None, 0)):
            label = f"{name}[{inst}]"
            if name == "GPS" and inst == 1:
                label = "GPS[1] (GPS2)"

        if "lat" in b:
            kind = KIND_GLOBAL
            lat = _fix_latlng(np.asarray(b["lat"], dtype=float))
            lng = _fix_latlng(np.asarray(b["lng"], dtype=float))
            alt = np.asarray(b["alt"], dtype=float)
            cols = [lat, lng, alt]
            if "status" in b and len(b["status"]) == t.size:
                ok = np.asarray(b["status"]) >= 3
                dropped = int((~ok).sum())
                if dropped and ok.sum() >= _MIN_SAMPLES:
                    t, cols = t[ok], [c[ok] for c in cols]
                    notes.append(f"{label}:濾除 {dropped} 筆非 3D-fix 樣本")
            lat, lng, alt = cols
            ned = None
        else:
            kind = KIND_NED
            ned = np.column_stack([b["px"], b["py"], b["pz"]]).astype(float)
            lat = lng = alt = None

        # 有效值 → 時間排序 → 去除重複時間戳(內插要求嚴格遞增)
        if kind == KIND_GLOBAL:
            good = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lng) & np.isfinite(alt)
        else:
            good = np.isfinite(t) & np.all(np.isfinite(ned), axis=1)
        sel = np.flatnonzero(good)
        sel = sel[np.argsort(t[sel], kind="stable")]
        if sel.size:
            ts = t[sel]
            sel = sel[np.concatenate([[True], np.diff(ts) > 1e-9])]
        t = t[sel]
        if kind == KIND_GLOBAL:
            lat, lng, alt = lat[sel], lng[sel], alt[sel]
        else:
            ned = ned[sel]

        if t.size < _MIN_SAMPLES:
            notes.append(f"{label}:樣本不足({t.size} 筆),略過")
            continue

        sources[label] = TrajSource(label, kind, t, lat=lat, lng=lng, alt=alt, ned=ned)
        key_label[(name, inst)] = label

    origin = origin_ekf or origin_any
    origin_desc = "ORGN(EKF 原點)" if origin_ekf else ("ORGN" if origin_any else "")
    if not sources:
        raise ValueError(
            "log 中找不到任何可用的位置訊息(找過 "
            + "、".join(CANDIDATE_TYPES) + ")。\n\n"
            "這份 log 實際包含的訊息型別:\n" + _legacy_available_types(mlog) + "\n\n"
            "若你的 VIO 位置記在別的訊息名稱下,請把該名稱告知或加進 "
            "core/logio.py 的 CANDIDATE_TYPES。")
    # ARM 與 EV 常同時記錄同一次解鎖:有 ARM 訊息就只用 ARM,避免重複計算
    events = (arm_evs or ev_evs) + mode_evs
    events += [LogEvent(t, EV_RESET, key_label[key]) for key, t in reset_hits if key in key_label]
    events.sort(key=lambda e: e.t)
    logdata = LogData(sources=sources, origin=origin, origin_desc=origin_desc,
                      path=path, notes=notes, events=events)
    n_seg = len(logdata.segments)
    if n_seg >= 2:
        notes.append(f"這份 log 有 {n_seg} 個 ARM 飛行段(開機到關機一份 log)— 可在時間軸選段分析")
    return logdata


def _legacy_parse_event(m, name: str, mlog) -> LogEvent | None:
    """ARM / EV / MODE 訊息 → LogEvent(不相關的 EV Id 回傳 None)。"""
    t_us = getattr(m, "TimeUS", None)
    if t_us is None:
        return None
    t = float(t_us) * 1e-6
    try:
        if name == "ARM":
            return LogEvent(t, EV_ARM if int(m.ArmState) else EV_DISARM)
        if name == "EV":
            eid = int(m.Id)
            if eid == _EV_ARMED:
                return LogEvent(t, EV_ARM)
            if eid == _EV_DISARMED:
                return LogEvent(t, EV_DISARM)
            return None
    except (AttributeError, TypeError, ValueError):
        return None
    # MODE:DFReader 在 recv_match 回傳前已依機種把編號解成名稱
    label = getattr(mlog, "flightmode", None)
    if not label or label == "UNKNOWN":
        num = getattr(m, "ModeNum", None) if "ModeNum" in m.get_fieldnames() else getattr(m, "Mode", "?")
        label = f"MODE {num}"
    return LogEvent(t, EV_MODE, str(label))


def _legacy_available_types(mlog, per_line: int = 12) -> str:
    """列出 log 中實際定義過的訊息名稱 — 找不到位置訊息時給使用者的線索。"""
    try:
        names = sorted({f.name for f in mlog.formats.values()})
    except AttributeError:
        return "(無法列出)"
    if not names:
        return "(無)"
    return "\n".join("  " + "  ".join(names[i:i + per_line])
                     for i in range(0, len(names), per_line))
