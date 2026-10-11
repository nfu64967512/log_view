"""預定任務航點:從 log 的 CMD 訊息讀出,轉成局部 ENU,並推估「飛到第幾個航點」。

ArduPilot 會在任務啟動時把整份任務寫進 log(CMD:CTot 總數、CNum 序號、CId MAVLink 指令、
Lat / Lng / Alt / Frame);有些版本還會在每個指令開始執行時再寫一筆。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import frames

# 帶位置的導航指令(MAV_CMD_NAV_*):CId → 顯示名稱
NAV_NAMES = {
    16: "WAYPOINT", 17: "LOITER_UNLIM", 18: "LOITER_TURNS", 19: "LOITER_TIME", 20: "RTL",
    21: "LAND", 22: "TAKEOFF", 31: "LOITER_TO_ALT", 82: "SPLINE_WAYPOINT",
    84: "VTOL_TAKEOFF", 85: "VTOL_LAND",
}
_ABS_FRAMES = (0, 5)     # MAV_FRAME_GLOBAL(海拔)/ GLOBAL_INT:高度為海拔;其餘(3 / 6 相對 home、10 地形)為相對


@dataclass
class Waypoint:
    n: int                 # 任務序號(CNum;0 為 home,不畫)
    cmd: int               # MAVLink 指令碼
    name: str
    lat: float
    lng: float
    alt: float             # 原始高度(依 frame)
    frame: int
    t: float               # 該筆 CMD 在 log 的時間(秒,log 時基)

    @property
    def has_position(self) -> bool:
        return abs(self.lat) > 1e-7 or abs(self.lng) > 1e-7


def load_mission(logdata) -> list[Waypoint]:
    """CMD → 帶位置的導航航點(依序號);沒有 CMD 或沒有帶位置的航點回傳 []。

    同一個 CNum 出現多次時取最後一筆(任務重新上傳、或指令開始執行時的再次紀錄);
    只保留 CNum < 最後一筆 CMD 的 CTot 者,排除舊任務殘留的尾巴。
    """
    idx = getattr(logdata, "index", None)
    if idx is None:
        return []
    try:                                  # 預定航線只是輔助資訊:索引壞掉 / 欄位不全都當作沒有任務
        if not idx.has("CMD.CNum"):
            return []
        tbl = idx.table("CMD", None)
    except Exception:  # noqa: BLE001
        return []
    if tbl is None or len(tbl["t"]) == 0 or not {"CNum", "CId", "Lat", "Lng", "Alt"} <= set(tbl):
        return []
    last: dict[int, int] = {}
    for i, n in enumerate(np.asarray(tbl["CNum"], dtype=int)):
        last[int(n)] = i
    total = int(tbl["CTot"][-1]) if "CTot" in tbl else max(last) + 1
    out = []
    for n in sorted(last):
        if n == 0 or n >= total:
            continue
        i = last[n]
        cid = int(tbl["CId"][i])
        lat, lng = float(tbl["Lat"][i]), float(tbl["Lng"][i])
        if abs(lat) > 90 or abs(lng) > 180:        # 少數格式把經緯度存成 1e7 倍整數(度 × 1e7)
            lat, lng = lat * 1e-7, lng * 1e-7
        w = Waypoint(n=n, cmd=cid, name=NAV_NAMES.get(cid, f"CMD {cid}"),
                     lat=lat, lng=lng, alt=float(tbl["Alt"][i]),
                     frame=int(tbl["Frame"][i]) if "Frame" in tbl else 3, t=float(tbl["t"][i]))
        if cid in NAV_NAMES and w.has_position:
            out.append(w)
    return out


def waypoints_enu(wps: list[Waypoint], origin: tuple[float, float, float]) -> np.ndarray:
    """航點 → Nx3 ENU(公尺,相對 origin)。相對高度以 origin 高度為 home 高度近似。"""
    if not wps:
        return np.zeros((0, 3))
    lat = np.array([w.lat for w in wps])
    lng = np.array([w.lng for w in wps])
    alt = np.array([w.alt if w.frame in _ABS_FRAMES else origin[2] + w.alt for w in wps])
    return frames.geodetic_to_enu(lat, lng, alt, origin)


def reach_radius(logdata) -> float:
    """航點到達半徑(公尺):Copter WPNAV_RADIUS(cm)/ Plane WP_RADIUS(m);至少 3 m(GPS 雜訊與取樣間隔)。"""
    r = 2.0
    idx = getattr(logdata, "index", None)
    try:
        pv = idx.param_values() if idx is not None else {}
    except Exception:  # noqa: BLE001 — 參數讀不到就用預設
        pv = {}
    if "WPNAV_RADIUS" in pv:
        r = float(pv["WPNAV_RADIUS"]) / 100.0
    elif "WP_RADIUS" in pv:
        r = float(pv["WP_RADIUS"])
    return max(3.0, r)


def auto_mask(modes, t: np.ndarray, name: str = "AUTO") -> np.ndarray:
    """t 每個時刻是否處於 name 模式(modes 為 timeline.mode_changes() 的輸出)。"""
    t = np.asarray(t, dtype=float)
    if not modes:
        return np.zeros(t.size, dtype=bool)
    mt = np.array([m.t for m in modes], dtype=float)
    ml = np.array([m.label for m in modes])
    i = np.searchsorted(mt, t, side="right") - 1
    return (i >= 0) & (ml[np.clip(i, 0, None)] == name)


def estimate_progress(t: np.ndarray, enu: np.ndarray, wps_enu: np.ndarray, auto_mask: np.ndarray,
                      radius: float, look_ahead: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """推估「目前飛往第幾個航點」(log 沒有逐航點的執行紀錄時用)。

    只在 auto_mask(AUTO 模式)為真的樣本推進:航跡進入「目前目標」(或其後 look_ahead 個)航點的
    radius 內(3D 距離)就視為到達,目標移到下一個;離開 AUTO 後進度保留(再進 AUTO 從原處續飛,
    和飛控的任務續飛一致)。回傳「變化點」:(時間, 目標航點索引)—— 索引為 wps 清單的 0 起算位置,
    len(wps) = 全部到達,-1 = 不在 AUTO(沒有目標)。
    """
    n = len(wps_enu)
    t = np.asarray(t, dtype=float)
    enu = np.asarray(enu, dtype=float)
    cur, prev = 0, None
    ch_t: list[float] = []
    ch_k: list[int] = []
    for i in range(t.size):
        if n == 0 or not auto_mask[i]:
            k = -1
        else:
            while cur < n:
                hi = min(n, cur + 1 + look_ahead)
                d = np.linalg.norm(wps_enu[cur:hi] - enu[i], axis=1)
                hit = np.nonzero(d < radius)[0]
                if hit.size == 0:
                    break
                cur += int(hit[-1]) + 1
            k = cur
        if k != prev:
            ch_t.append(float(t[i]))
            ch_k.append(k)
            prev = k
    return np.array(ch_t), np.array(ch_k, dtype=int)
