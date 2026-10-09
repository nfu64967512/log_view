"""模擬資料:仿照戶外方形任務(如 Mission Planner 截圖場景)產生
GPS(5 Hz)/ VISP(30 Hz,含 VIO 漂移・偏航偏差・尺度誤差)/ POS(10 Hz)。

用途:在還沒有真實 .bin 之前驗證整條 ATE 流程與 UI;也供自我測試比對
已知注入誤差(偏航 / 尺度)是否能被對齊演算法還原。

- make_sim_with_truth():單次 ARM 的任務(一般「每次 ARM 一份 log」)。
- make_multi_flight_sim():開機到關機一份 log,含地面待機、兩次起降、
  上鎖時手提移動、第二次 ARM 時 VIO 重置 — 用來示範為何要逐段分析。

兩者都附 LogData.index(core.dflog.LogIndex):仿真實 log 的輔助訊息 ATT / GPS / POS /
VISP / XKF1 / CTUN / BAT / VIBE / RCIN / RCOU / PARM / MSG / MODE / ARM / EV / ERR / ORGN,
由真值運動推得(姿態 = 加速度的傾角、航向 = 對地航跡),用「另一組」亂數產生 ——
既有軌跡(GPS / VISP / POS)與舊版逐位元相同。XKF1(EKF 局部位置)也列為軌跡來源。
"""
from __future__ import annotations

import numpy as np

from . import frames
from .types import (EV_ARM, EV_DISARM, EV_MODE, EV_RESET, KIND_GLOBAL, KIND_NED,
                    LogData, LogEvent, TrajSource)

# 模擬場景參數(虎尾附近經緯度,對應截圖場地尺度)
ORIGIN = (23.70200, 120.43100, 38.0)
YAW_MIS_DEG = 1.5        # 注入的 VIO 偏航偏差
SCALE_ERR = 1.004        # 注入的 VIO 尺度誤差
T_BOOT = 100.0           # 模擬開機時基(秒)
_DT = 1.0 / 60.0         # 真值取樣週期

# 多航段模擬:VIO 重置後的新座標框偏航、隨機漫步每步標準差(30 Hz)
YAW_RESET_DEG = -6.0
_RW_STEP = 0.012         # ≈ 單航段模擬的漂移速率(190 s 約 0.9 m)


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    pad = (win // 2, win - 1 - win // 2)
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(win) / win, mode="valid")


def _gauss_markov(n: int, dt: float, tau: float, sigma: float, rng) -> np.ndarray:
    a = float(np.exp(-dt / tau))
    q = sigma * np.sqrt(1.0 - a * a)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    w = rng.normal(0.0, q, n)
    for k in range(1, n):
        x[k] = a * x[k - 1] + w[k]
    return x


# ---------------- 真值軌跡積木 ----------------
# 階段 = (名稱, 時長 s, fn(tau) -> (E, N, U)),tau 為階段內時間

def _hold(e0: float = 0.0, n0: float = 0.0, u0: float = 0.0):
    """原地不動(地面待機 / 懸停)。"""
    return lambda tau: (np.full_like(tau, e0), np.full_like(tau, n0), np.full_like(tau, u0))


def _flight_phases(laps: int, side: float, speed: float, alt: float,
                   e0: float = 0.0, n0: float = 0.0) -> list:
    """一次飛行:爬升 → 懸停 → 方形繞圈(含高度正弦調變)→ 懸停 → 降落。

    (e0, n0) 為起降點;方形以起降點為一角。
    """
    corners = [(0.0, 0.0)]
    for _ in range(laps):
        corners += [(side, 0.0), (side, side), (0.0, side), (0.0, 0.0)]
    pts = np.asarray(corners)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    t_cruise = total / speed

    def en_at(s):
        return np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])

    def climb(tau):
        s = np.clip(tau / 8.0, 0, 1)
        return np.full_like(tau, e0), np.full_like(tau, n0), alt * (3 * s ** 2 - 2 * s ** 3)

    def cruise(tau):
        e, n = en_at(np.minimum(tau * speed, total))
        if e0 or n0:
            e, n = e + e0, n + n0
        return e, n, alt + 1.2 * np.sin(2 * np.pi * tau / 45.0)

    def descend(tau):
        s = np.clip(tau / 8.0, 0, 1)
        return np.full_like(tau, e0), np.full_like(tau, n0), alt * (1 - (3 * s ** 2 - 2 * s ** 3))

    return [
        ("climb", 8.0, climb),
        ("hover", 3.0, _hold(e0, n0, alt)),
        ("cruise", t_cruise, cruise),
        ("hover", 3.0, _hold(e0, n0, alt)),
        ("descend", 8.0, descend),
    ]


def _render(phases: list, t_start: float = T_BOOT):
    """取樣並串接各階段,平滑後回傳 (t, enu Nx3, [(階段名稱, 起始時間), ...])。"""
    dt = _DT
    es, ns, us, starts = [], [], [], []
    k = 0
    for name, dur, fn in phases:
        tau = np.arange(0.0, dur, dt)
        e, n, u = fn(tau)
        es.append(e); ns.append(n); us.append(u)
        starts.append((name, k))
        k += tau.size
    e = np.concatenate(es); n = np.concatenate(ns); u = np.concatenate(us)
    w_en = int(round(1.0 / dt)) | 1
    w_u = int(round(0.5 / dt)) | 1
    e = _smooth(e, w_en); n = _smooth(n, w_en); u = _smooth(u, w_u)
    t = np.arange(e.size) * dt + t_start
    return t, np.column_stack([e, n, u]), [(name, float(t[i])) for name, i in starts]


def _truth(laps: int, side: float, speed: float, alt: float, with_phases: bool = False):
    """組出:地面 → 爬升 → 懸停 → 方形繞圈(含高度正弦調變)→ 懸停 → 降落 → 地面。

    with_phases=True 時另回傳各階段起始時間 [(名稱, t), ...]。
    """
    phases = ([("ground", 6.0, _hold())]
              + _flight_phases(laps, side, speed, alt)
              + [("ground", 4.0, _hold())])
    t, enu, starts = _render(phases)
    return (t, enu, starts) if with_phases else (t, enu)


def _flight_events(starts: list, i_climb: int, stabilize_after: bool) -> list[LogEvent]:
    """依 _flight_phases 的階段時間產生一次飛行的模式 / ARM 事件。

    starts[i_climb] 為該次爬升;其後依序是懸停、巡航、懸停、降落、(著地後)下一階段。
    """
    t_climb = starts[i_climb][1]
    t_cruise = starts[i_climb + 2][1]
    t_descend = starts[i_climb + 4][1]
    t_touch = starts[i_climb + 5][1]
    t_arm = t_climb - 1.0
    t_disarm = t_touch + 2.0
    evs = [
        LogEvent(t_arm - 1.0, EV_MODE, "LOITER"),
        LogEvent(t_arm, EV_ARM),
        LogEvent(t_cruise, EV_MODE, "AUTO"),
        LogEvent(t_descend, EV_MODE, "LAND"),
        LogEvent(t_disarm, EV_DISARM),
    ]
    if stabilize_after:
        evs.append(LogEvent(t_disarm + 1.0, EV_MODE, "STABILIZE"))
    return evs


# ---------------- 感測器模型 ----------------

def _gps_from_truth(t, enu, rng) -> TrajSource:
    """GPS 5 Hz:Gauss-Markov 慢漂 + 白雜訊。"""
    dt = _DT
    idx = np.arange(0, t.size, 12)
    tg = t[idx]
    ng = idx.size
    noise = np.column_stack([
        _gauss_markov(ng, dt * 12, 20.0, 0.35, rng) + rng.normal(0, 0.03, ng),
        _gauss_markov(ng, dt * 12, 20.0, 0.35, rng) + rng.normal(0, 0.03, ng),
        _gauss_markov(ng, dt * 12, 25.0, 0.70, rng) + rng.normal(0, 0.05, ng),
    ])
    lat, lng, galt = frames.enu_to_geodetic(enu[idx] + noise, ORIGIN)
    return TrajSource("GPS", KIND_GLOBAL, tg, lat=lat, lng=lng, alt=galt,
                      meta="模擬 GPS 5 Hz(水平 σ≈0.35 m、垂直 σ≈0.7 m 慢漂)")


def _pos_from_truth(t, enu, rng) -> TrajSource:
    """POS 10 Hz:EKF 融合輸出(小誤差)。"""
    dt = _DT
    idx = np.arange(0, t.size, 6)
    tp = t[idx]
    npn = idx.size
    noise = np.column_stack([
        _gauss_markov(npn, dt * 6, 30.0, 0.10, rng) + rng.normal(0, 0.02, npn),
        _gauss_markov(npn, dt * 6, 30.0, 0.10, rng) + rng.normal(0, 0.02, npn),
        _gauss_markov(npn, dt * 6, 30.0, 0.15, rng) + rng.normal(0, 0.03, npn),
    ])
    lat, lng, palt = frames.enu_to_geodetic(enu[idx] + noise, ORIGIN)
    return TrajSource("POS", KIND_GLOBAL, tp, lat=lat, lng=lng, alt=palt,
                      meta="模擬 EKF 位置 10 Hz(σ≈0.1 m)")


def _yaw_scale_ned(ned: np.ndarray, yaw_deg: float, scale: float) -> np.ndarray:
    """NED 內繞 D 軸轉 yaw_deg 並乘尺度(VIO 座標框偏差)。"""
    mis = np.radians(yaw_deg)
    c, s = np.cos(mis), np.sin(mis)
    n_rot = c * ned[:, 0] - s * ned[:, 1]
    e_rot = s * ned[:, 0] + c * ned[:, 1]
    return np.column_stack([n_rot, e_rot, ned[:, 2]]) * scale


# ---------------- 單航段 ----------------

def make_sim_with_truth(seed: int = 7, laps: int = 4, side: float = 30.0,
                        speed: float = 3.0, alt: float = 10.0):
    """回傳 (LogData, truth_info)。truth_info 供自我測試核對注入誤差。"""
    rng = np.random.default_rng(seed)
    t, enu, starts = _truth(laps, side, speed, alt, with_phases=True)

    # --- GPS 5 Hz ---
    gps = _gps_from_truth(t, enu, rng)

    # --- VISP 30 Hz:偏航偏差 + 尺度 + 隨機漫步漂移 + 白雜訊(NED,相對 EKF 原點) ---
    idx = np.arange(0, t.size, 2)
    tv = t[idx]
    nv = idx.size
    ned_v = _yaw_scale_ned(frames.enu_to_ned(enu[idx]), YAW_MIS_DEG, SCALE_ERR)
    step = 0.9 / np.sqrt(nv)
    drift = np.cumsum(rng.normal(0, step, (nv, 3)) * np.array([1.0, 1.0, 0.5]), axis=0)
    ned_v += drift + rng.normal(0, 0.02, (nv, 3))
    visp = TrajSource("VISP", KIND_NED, tv, ned=ned_v,
                      meta=f"模擬 VIO 30 Hz(偏航 {YAW_MIS_DEG}°、尺度 {SCALE_ERR}、漂移≈0.9 m RW)")

    # --- POS 10 Hz ---
    pos = _pos_from_truth(t, enu, rng)

    # --- 事件:開機 STABILIZE → 起飛前 LOITER、ARM → 巡航 AUTO → 降落 LAND → 著地後 DISARM ---
    events = [LogEvent(float(t[0]), EV_MODE, "STABILIZE")]
    events += _flight_events(starts, 1, stabilize_after=False)
    events.sort(key=lambda e: e.t)

    logdata = LogData(
        sources={"GPS": gps, "VISP": visp, "POS": pos},
        origin=ORIGIN, origin_desc="ORGN(模擬)",
        path="(模擬資料)",
        notes=[
            f"模擬資料:{side:.0f} m 方形任務 × {laps} 圈,巡航 {speed} m/s,高度 {alt} m ± 1.2 m",
            f"VIO 注入誤差:偏航 {YAW_MIS_DEG}°、尺度 {SCALE_ERR}、隨機漫步漂移 ≈ 0.9 m(seed={seed})",
        ],
        events=events,
    )
    # --- 輔助訊息(另一組亂數):巡航中段一次短暫 GPS glitch(ERR GPS 2 → 0) ---
    f1 = _flight_story(starts, 1, side, laps, speed)
    story = {
        "flights": [f1],
        "ev": [],
        "err": [(f1["t_cruise"] + 25.0, 11, 2), (f1["t_cruise"] + 27.0, 11, 0)],
        "msg": [(f1["t_cruise"] + 25.0, "GPS Glitch or Compass error"),
                (f1["t_cruise"] + 27.0, "Glitch cleared")],
        "rc7_t": None,
        "visp_yaw": lambda tt: np.full(np.shape(tt), YAW_MIS_DEG),
    }
    _attach_index(logdata, t, enu, story, seed)
    truth_info = {"t": t, "enu": enu, "yaw_mis_deg": YAW_MIS_DEG, "scale": SCALE_ERR, "origin": ORIGIN,
                  "phases": starts}
    return logdata, truth_info


def make_sim_logdata(seed: int = 7) -> LogData:
    return make_sim_with_truth(seed)[0]


# ---------------- 多航段(開機到關機一份 log) ----------------

def _carry(e0: float, n0: float, de: float, walk_s: float, lift: float = 0.8):
    """上鎖後手提搬移:抬起 1 s → 等速步行 walk_s 秒東移 de 公尺 → 放下 1 s。"""
    total = walk_s + 2.0

    def fn(tau):
        e = e0 + de * np.clip((tau - 1.0) / walk_s, 0, 1)
        u = lift * np.clip(np.minimum(tau, total - tau), 0, 1)
        return e, np.full_like(tau, n0), u
    return total, fn


def make_multi_flight_sim(seed: int = 11) -> tuple[LogData, dict]:
    """開機到關機一份 log:待機 → 第 1 航段 → 上鎖手提東移 → VIO 重置 + 第 2 航段 → 待機。

    VIO 在第二次 ARM 前重置(VISP ResetCnt +1):新座標框原點 = 重置當下位置、
    偏航改為 YAW_RESET_DEG。整份 log 一起對齊時兩個座標框無法用同一個剛體變換
    擬合,誤差被灌大;逐段(每次 ARM)對齊才是有意義的 ATE。

    回傳 (LogData, info),info = {"flights": [(t_arm, t_disarm), ...], "reset_t",
    "origin", 以及 "t" / "enu" 真值與注入參數}。
    """
    rng = np.random.default_rng(seed)
    speed = 3.0
    shift_e, walk_speed = 15.0, 1.2
    walk_s = shift_e / walk_speed
    carry_s, carry_fn = _carry(0.0, 0.0, shift_e, walk_s)

    phases = [("ground", 20.0, _hold())]
    i_f1 = len(phases)
    phases += _flight_phases(2, 30.0, speed, 10.0)
    phases += [("ground", 5.0, _hold()),
               ("carry", carry_s, carry_fn),
               ("ground", 30.0 - 5.0 - carry_s, _hold(shift_e, 0.0))]
    i_f2 = len(phases)
    phases += _flight_phases(2, 20.0, speed, 8.0, e0=shift_e)
    phases += [("ground", 10.0, _hold(shift_e, 0.0))]
    t, enu, starts = _render(phases)

    events = [LogEvent(float(t[0]), EV_MODE, "STABILIZE")]
    ev1 = _flight_events(starts, i_f1, stabilize_after=True)
    ev2 = _flight_events(starts, i_f2, stabilize_after=True)
    events += ev1 + ev2
    flights = []
    for evs in (ev1, ev2):
        t_arm = next(e.t for e in evs if e.kind == EV_ARM)
        t_disarm = next(e.t for e in evs if e.kind == EV_DISARM)
        flights.append((t_arm, t_disarm))
    reset_t = flights[1][0] - 0.5          # 第二次 ARM 前 0.5 s VIO 重新初始化
    events.append(LogEvent(reset_t, EV_RESET, "VISP"))
    events.sort(key=lambda e: e.t)

    # --- GPS 5 Hz ---
    gps = _gps_from_truth(t, enu, rng)

    # --- VISP 30 Hz:重置前 = 框 A(EKF 原點、偏航 1.5°);重置後 = 框 B(原點 = 重置位置、偏航 −6°) ---
    idx = np.arange(0, t.size, 2)
    tv = t[idx]
    ned = frames.enu_to_ned(enu[idx])
    after = tv >= reset_t
    ned_reset = frames.enu_to_ned(enu[[int(np.searchsorted(t, reset_t))]])[0]
    ned_v = np.empty_like(ned)
    ned_v[~after] = _yaw_scale_ned(ned[~after], YAW_MIS_DEG, SCALE_ERR)
    ned_v[after] = _yaw_scale_ned(ned[after] - ned_reset, YAW_RESET_DEG, SCALE_ERR)
    w = np.array([1.0, 1.0, 0.5])
    for m in (~after, after):          # 每個座標框各自從零開始漂移
        k = int(m.sum())
        ned_v[m] += np.cumsum(rng.normal(0, _RW_STEP, (k, 3)) * w, axis=0)
    ned_v += rng.normal(0, 0.02, ned_v.shape)
    visp = TrajSource("VISP", KIND_NED, tv, ned=ned_v,
                      meta=f"模擬 VIO 30 Hz(偏航 {YAW_MIS_DEG}°→重置後 {YAW_RESET_DEG}°、"
                           f"尺度 {SCALE_ERR}、隨機漫步漂移)")

    # --- POS 10 Hz ---
    pos = _pos_from_truth(t, enu, rng)

    tb = float(t[0])
    (a1, d1), (a2, d2) = flights
    logdata = LogData(
        sources={"GPS": gps, "VISP": visp, "POS": pos},
        origin=ORIGIN, origin_desc="ORGN(模擬)",
        path="(模擬資料:多航段)",
        notes=[
            "模擬資料(多航段):開機到關機一份 log — 地面待機 20 s → 第 1 航段 30 m 方形 × 2 圈"
            f"(高 10 m)→ 上鎖後手提東移 {shift_e:.0f} m → 第 2 航段 20 m 方形 × 2 圈(高 8 m)→ 待機 10 s",
            f"ARM 1:{a1 - tb:.1f} ~ {d1 - tb:.1f} s;ARM 2:{a2 - tb:.1f} ~ {d2 - tb:.1f} s(相對 log 起點)",
            f"VIO 在 {reset_t - tb:.1f} s 重置(ResetCnt +1):原點改為當下位置、偏航 {YAW_RESET_DEG}°;"
            "整份一起對齊會被兩個座標框拖垮,請在時間軸逐段分析",
            f"VIO 注入誤差:偏航 {YAW_MIS_DEG}°(重置後 {YAW_RESET_DEG}°)、尺度 {SCALE_ERR}、"
            f"隨機漫步漂移(seed={seed})",
        ],
        events=events,
    )
    # --- 輔助訊息(另一組亂數)---
    # 第 1 航段上鎖後切 EKF 來源到 ExternalNav(RC7 開關 → EV 86);VIO 重置後 EKF 改用
    # 外部導航;第 2 航段爬升時 EKF 變異數短暫偏大(ERR EKFCHECK 2 → 0)
    f1 = _flight_story(starts, i_f1, 30.0, 2, speed)
    f2 = _flight_story(starts, i_f2, 20.0, 2, speed)
    t_src = d1 + 4.0
    story = {
        "flights": [f1, f2],
        "ev": [(t_src, 86)],
        "err": [(f2["t_climb"] + 2.0, 16, 2), (f2["t_climb"] + 3.5, 16, 0)],
        "msg": [(t_src + 0.01, "EKF3 IMU0 switching to ExternalNav"),
                (t_src + 0.02, "EKF3 IMU1 switching to ExternalNav"),
                (reset_t + 0.25, "EKF3 IMU0 is using external nav data"),
                (reset_t + 0.26, "EKF3 IMU1 is using external nav data")],
        "rc7_t": t_src,
        "visp_yaw": lambda tt: np.where(np.asarray(tt) >= reset_t, YAW_RESET_DEG, YAW_MIS_DEG),
    }
    _attach_index(logdata, t, enu, story, seed)
    info = {"flights": flights, "reset_t": float(reset_t), "origin": ORIGIN,
            "t": t, "enu": enu, "phases": starts,
            "yaw_mis_deg": YAW_MIS_DEG, "yaw_reset_deg": YAW_RESET_DEG, "scale": SCALE_ERR}
    return logdata, info


# ---------------- 輔助訊息表格(LogIndex) ----------------
# 全部由真值運動推得,亂數用 default_rng([seed, 0xA5]) —— 與軌跡的亂數各自獨立,
# 既有 GPS / VISP / POS 不受影響。

_G = 9.80665
_GPS_WEEK, _GPS_MS0 = 2320, 300_000
# 飛行模式開關(RC5)各段的 PWM
_MODE_PWM = {"STABILIZE": 1165, "ALT_HOLD": 1295, "LOITER": 1425, "POSHOLD": 1555,
             "LAND": 1685, "AUTO": 1815}
# (名稱, 值, 預設值) —— 典型的 VIO 室外 / 室內切換設定
SIM_PARAMS = [
    ("ARMING_CHECK", 1, 1), ("BATT_CAPACITY", 5200, 3300), ("BATT_MONITOR", 4, 0),
    ("EK3_SRC1_POSXY", 3, 3), ("EK3_SRC1_POSZ", 1, 1), ("EK3_SRC1_VELXY", 3, 3),
    ("EK3_SRC1_YAW", 1, 1), ("EK3_SRC2_POSXY", 6, 0), ("EK3_SRC2_POSZ", 1, 1),
    ("EK3_SRC2_VELXY", 6, 0), ("EK3_SRC2_YAW", 6, 0), ("FRAME_CLASS", 1, 0),
    ("FRAME_TYPE", 1, 1), ("GPS_TYPE", 1, 1), ("LOG_BITMASK", 180222, 180222),
    ("LOG_DISARMED", 1, 0), ("RC7_OPTION", 90, 0), ("VISO_DELAY_MS", 30, 10),
    ("VISO_POS_M_NSE", 0.2, 0.2), ("VISO_TYPE", 1, 0), ("WPNAV_SPEED", 300, 1000),
]
_BOOT_MSGS = [
    (0.000, "ArduCopter V4.5.7 (2a3dc4b7)"),
    (0.001, "ChibiOS: 6e2c5d1a"),
    (0.002, "CubeOrange 0037002F 3239510B 33353834"),
    (0.003, "RCOut: PWM:1-14"),
    (0.004, "IMU0: fast sampling enabled 8.0kHz/2.0kHz"),
    (0.005, "Frame: QUAD/X"),
    (0.350, "GPS 1: detected as u-blox at 230400 baud"),
    (0.620, "EKF3 IMU0 initialised"), (0.630, "EKF3 IMU1 initialised"),
    (0.640, "AHRS: EKF3 active"),
    (1.100, "EKF3 IMU0 tilt alignment complete"), (1.110, "EKF3 IMU1 tilt alignment complete"),
    (1.500, "EKF3 IMU0 MAG0 initial yaw alignment complete"),
    (1.510, "EKF3 IMU1 MAG0 initial yaw alignment complete"),
    (2.000, "EKF3 IMU0 origin set"), (2.010, "EKF3 IMU1 origin set"),
    (2.400, "EKF3 IMU0 is using GPS"), (2.410, "EKF3 IMU1 is using GPS"),
]
SIM_UNITS = {
    "ATT": {"DesRoll": "deg", "Roll": "deg", "DesPitch": "deg", "Pitch": "deg",
            "DesYaw": "degheading", "Yaw": "degheading"},
    "GPS": {"GMS": "ms", "NSats": "satellites", "Lat": "deglatitude", "Lng": "deglongitude",
            "Alt": "m", "Spd": "m/s", "GCrs": "degheading", "VZ": "m/s", "Yaw": "degheading"},
    "POS": {"Lat": "deglatitude", "Lng": "deglongitude", "Alt": "m", "RelHomeAlt": "m",
            "RelOriginAlt": "m"},
    "VISP": {"RTimeUS": "µs", "PX": "m", "PY": "m", "PZ": "m", "Roll": "deg",
             "Pitch": "deg", "Yaw": "degheading", "PErr": "m", "AErr": "deg"},
    "XKF1": {"Roll": "deg", "Pitch": "deg", "Yaw": "degheading", "VN": "m/s", "VE": "m/s",
             "VD": "m/s", "dPD": "m/s", "PN": "m", "PE": "m", "PD": "m", "GX": "deg/s",
             "GY": "deg/s", "GZ": "deg/s", "OH": "m"},
    "CTUN": {"DAlt": "m", "Alt": "m", "BAlt": "m", "DCRt": "cm/s", "CRt": "cm/s"},
    "BAT": {"Volt": "V", "VoltR": "V", "Curr": "A", "CurrTot": "mAh", "EnrgTot": "W.h",
            "Temp": "degC", "Res": "Ohm", "RemPct": "%"},
    "VIBE": {"VibeX": "m/s/s", "VibeY": "m/s/s", "VibeZ": "m/s/s"},
    "RCIN": {f"C{i}": "us" for i in range(1, 9)},
    "RCOU": {f"C{i}": "us" for i in range(1, 5)},
    "ORGN": {"Lat": "deglatitude", "Lng": "deglongitude", "Alt": "m"},
}


def _flight_story(starts: list, i_climb: int, side: float, laps: int, speed: float) -> dict:
    """一次飛行的關鍵時刻(與 _flight_events 同一套階段索引)。"""
    t_climb = starts[i_climb][1]
    t_touch = starts[i_climb + 5][1]
    return {"t_climb": t_climb, "t_cruise": starts[i_climb + 2][1],
            "t_descend": starts[i_climb + 4][1], "t_touch": t_touch,
            "t_arm": t_climb - 1.0, "t_disarm": t_touch + 2.0,
            "legs": 4 * laps, "leg_s": side / speed}


def _grid(t0: float, t1: float, hz: float) -> np.ndarray:
    """[t0, t1] 內的等間隔取樣(不超過 t1:輔助來源不能改變 LogData.t_end)。"""
    g = t0 + np.arange(int(np.floor((t1 - t0) * hz + 1e-6)) + 1) / hz
    return g[g <= t1]


def _f32(x) -> np.ndarray:
    """先量化成 float32 再轉回 float64:.bin(格式 f)讀回來逐位元相同。"""
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def _wrap32(deg) -> np.ndarray:
    """航向角:先包到 [0, 360) 再量化成 float32(避免 359.99999 量化成 360)。"""
    return _f32(np.asarray(deg) % 360.0) % 360.0


def _hold_course(vel: np.ndarray, min_speed: float = 0.4) -> np.ndarray:
    """對地航向(度,北起順時針);速度太小(懸停、爬升、地面)時沿用上一個有效值。"""
    vh = np.hypot(vel[:, 0], vel[:, 1])
    course = np.degrees(np.arctan2(vel[:, 0], vel[:, 1])) % 360.0
    moving = vh > min_speed
    if not moving.any():
        return np.zeros(vel.shape[0])
    idx = np.maximum.accumulate(np.where(moving, np.arange(vel.shape[0]), -1))
    idx[idx < 0] = int(np.argmax(moving))    # 第一次移動之前:沿用第一個航向
    return course[idx]


def _attach_index(logdata: LogData, t: np.ndarray, enu: np.ndarray, story: dict, seed: int) -> None:
    """產生輔助訊息表格 → logdata.index;並把 EV / ERR 事件、XKF1 來源加進 logdata。"""
    from .dflog import LogIndex
    from .fakebin import COPTER_MODES
    from .logio import index_events
    from .timeline import mode_at, mode_changes

    rng = np.random.default_rng([seed, 0xA5])
    tb, te = logdata.t_base, logdata.t_end       # 輔助表格都落在既有來源的時間範圍內
    segs = [(s.t0, s.t1) for s in logdata.segments]
    modes = mode_changes(logdata.events)

    def armed(tt):
        tt = np.asarray(tt)
        m = np.zeros(tt.shape, bool)
        for a, d in segs:
            m |= (tt >= a) & (tt < d)
        return m

    def at(x, tt):
        return np.interp(tt, t, x)

    def gm(n, dt, tau, sigma):
        return _gauss_markov(n, dt, tau, sigma, rng) if n else np.zeros(0)

    # --- 真值運動學(60 Hz):速度、加速度、航向、由加速度推得的傾角 ---
    vel = np.gradient(enu, t, axis=0)
    acc = np.gradient(vel, t, axis=0)
    acc = np.column_stack([_smooth(acc[:, i], 15) for i in range(3)])
    hdg = np.degrees(np.unwrap(np.radians(_hold_course(vel))))
    hdg = _smooth(hdg, 61)                       # 機頭轉向約 1 s 完成
    psi = np.radians(hdg)
    a_fwd = acc[:, 0] * np.sin(psi) + acc[:, 1] * np.cos(psi)
    a_right = acc[:, 0] * np.cos(psi) - acc[:, 1] * np.sin(psi)
    roll_c = np.degrees(np.arctan(a_right / _G))      # 右轉(向心加速度朝右)→ 正 roll
    pitch_c = -np.degrees(np.arctan(a_fwd / _G))      # 向前加速 → 低頭(負 pitch)
    up, vup = enu[:, 2], vel[:, 2]
    on_ground = (up < 0.03) & (np.hypot(vel[:, 0], vel[:, 1]) < 0.05)
    roll_c = np.where(on_ground, 0.7, roll_c)          # 地面:停機坪微傾
    pitch_c = np.where(on_ground, -1.1, pitch_c)

    tables: dict[str, dict[str, np.ndarray]] = {}

    # --- MSG:開機訊息、任務航點、劇情訊息 ---
    msgs = [(tb + dt, s) for dt, s in _BOOT_MSGS]
    for f in story["flights"]:
        msgs += [(f["t_cruise"] + k * f["leg_s"] + 0.05, f"Mission: {k + 1} WP")
                 for k in range(f["legs"])]
    msgs += list(story["msg"])
    msgs.sort(key=lambda x: x[0])
    tables["MSG"] = {"t": np.array([m[0] for m in msgs]),
                     "Message": np.array([m[1] for m in msgs], dtype=str)}

    # --- CMD:預定任務(第一次飛行的每個轉角一個航點 + RTL),任務啟動時一次傾印 ---
    f0 = story["flights"][0]
    tw = np.array([f0["t_cruise"] + (k + 1) * f0["leg_s"] for k in range(f0["legs"])])
    w_enu = np.column_stack([at(enu[:, i], tw) for i in range(3)])
    w_lat, w_lng, _ = frames.enu_to_geodetic(w_enu, ORIGIN)
    nw = tw.size
    n_cmd = nw + 2
    tables["CMD"] = {
        "t": tb + 3.0 + 0.001 * np.arange(n_cmd),
        "CTot": np.full(n_cmd, n_cmd, np.uint16), "CNum": np.arange(n_cmd, dtype=np.uint16),
        "CId": np.array([16] + [16] * nw + [20], dtype=np.uint16),
        "Prm1": np.zeros(n_cmd), "Prm2": np.zeros(n_cmd), "Prm3": np.zeros(n_cmd),
        "Prm4": np.zeros(n_cmd),
        "Lat": np.concatenate([[0.0], w_lat, [0.0]]), "Lng": np.concatenate([[0.0], w_lng, [0.0]]),
        "Alt": _f32(np.concatenate([[0.0], np.full(nw, np.round(np.median(w_enu[:, 2]))), [0.0]])),
        "Frame": np.array([0] + [3] * nw + [0], dtype=np.uint8),
    }

    # --- PARM:開機時一次寫出 ---
    k = len(SIM_PARAMS)
    tables["PARM"] = {"t": tb + 0.010 + 0.001 * np.arange(k),
                      "Name": np.array([p[0] for p in SIM_PARAMS], dtype=str),
                      "Value": _f32([p[1] for p in SIM_PARAMS]),
                      "Default": _f32([p[2] for p in SIM_PARAMS])}

    # --- MODE / ARM / EV / ERR ---
    mode_ev = [e for e in logdata.events if e.kind == EV_MODE and e.label.upper() in COPTER_MODES]
    nums = np.array([COPTER_MODES[e.label.upper()] for e in mode_ev], dtype=np.int64)
    tables["MODE"] = {"t": np.array([e.t for e in mode_ev]), "Mode": nums.astype(np.int8),
                      "ModeNum": nums.astype(np.uint8),
                      "Rsn": np.array([0] + [1] * (len(mode_ev) - 1), dtype=np.uint8)[:len(mode_ev)]}
    arm_ev = [e for e in logdata.events if e.kind in (EV_ARM, EV_DISARM)]
    is_arm = np.array([e.kind == EV_ARM for e in arm_ev])
    tables["ARM"] = {"t": np.array([e.t for e in arm_ev]), "ArmState": is_arm.astype(np.uint8),
                     "ArmChecks": np.ones(len(arm_ev), np.uint32),
                     "Forced": np.zeros(len(arm_ev), np.uint8),
                     "Method": np.where(is_arm, 1, 13).astype(np.uint8)}   # MAVLink 解鎖 / 落地自動上鎖
    evs = [(e.t, 10 if e.kind == EV_ARM else 11) for e in arm_ev]
    for f in story["flights"]:
        evs += [(f["t_climb"] + 0.4, 28),              # NOT_LANDED
                (f["t_touch"] + 0.3, 17),              # LAND_COMPLETE_MAYBE
                (f["t_touch"] + 1.0, 18)]              # LAND_COMPLETE
    evs += list(story["ev"])
    evs.sort(key=lambda x: x[0])
    tables["EV"] = {"t": np.array([e[0] for e in evs]),
                    "Id": np.array([e[1] for e in evs], dtype=np.uint8)}
    errs = sorted(story["err"])
    tables["ERR"] = {"t": np.array([e[0] for e in errs]),
                     "Subsys": np.array([e[1] for e in errs], dtype=np.uint8),
                     "ECode": np.array([e[2] for e in errs], dtype=np.uint8)}

    # --- ORGN:EKF 原點(Type 0)+ 每次解鎖的 home(Type 1) ---
    homes = [f["t_arm"] for f in story["flights"]]
    h_enu = np.array([[at(enu[:, i], th) for i in range(3)] for th in homes])
    h_lat, h_lng, h_alt = frames.enu_to_geodetic(h_enu, ORIGIN)
    tables["ORGN"] = {"t": np.concatenate([[tb + 2.0], homes]),
                      "Type": np.array([0] + [1] * len(homes), dtype=np.uint8),
                      "Lat": np.concatenate([[ORIGIN[0]], h_lat]),
                      "Lng": np.concatenate([[ORIGIN[1]], h_lng]),
                      "Alt": np.round(np.concatenate([[ORIGIN[2]], h_alt]), 2)}

    # --- ATT 25 Hz:姿態 = 真值傾角 + 擾動(空中)/ 手持晃動(上鎖搬移時) ---
    ta = _grid(tb, te, 25.0)
    na = ta.size
    arm_a = armed(ta)
    up_a = at(up, ta)
    air_a = arm_a & (up_a > 0.05)
    carry_a = ~arm_a & ((up_a > 0.03) | (np.hypot(at(vel[:, 0], ta), at(vel[:, 1], ta)) > 0.05))
    shake = np.where(air_a, 1.0, 0.0)[:, None] * np.column_stack(
        [gm(na, 0.04, 0.3, 0.45), gm(na, 0.04, 0.3, 0.45), gm(na, 0.04, 2.0, 0.6)])
    shake += np.where(carry_a, 1.0, 0.0)[:, None] * np.column_stack(
        [gm(na, 0.04, 0.6, 2.5), gm(na, 0.04, 0.6, 2.5), gm(na, 0.04, 1.5, 4.0)])
    shake += np.column_stack([rng.normal(0, 0.05, na) for _ in range(3)])
    roll_a = at(roll_c, ta) + shake[:, 0]
    pitch_a = at(pitch_c, ta) + shake[:, 1]
    yaw_u = at(hdg, ta) + shake[:, 2]
    lead = ta + 0.1                                   # 期望姿態領先實際約 0.1 s
    tables["ATT"] = {
        "t": ta,
        "DesRoll": _f32(np.where(arm_a, at(roll_c, lead), 0.0)),
        "Roll": _f32(roll_a),
        "DesPitch": _f32(np.where(arm_a, at(pitch_c, lead), 0.0)),
        "Pitch": _f32(pitch_a),
        "DesYaw": _wrap32(np.where(arm_a, at(hdg, lead), yaw_u)),
        "Yaw": _wrap32(yaw_u),
        "AEKF": np.full(na, 3, np.uint8),
    }

    # --- GPS[0]:沿用 GPS 來源的時間 / 經緯高,補上衛星數、HDop、速度、航跡 ---
    gsrc = logdata.sources["GPS"]
    tg = gsrc.t
    ng = tg.size
    steps = np.where(rng.random(ng) < 0.02, rng.choice([-1, 1], ng), 0)
    nsats = np.clip(15 + np.cumsum(steps), 12, 19).astype(np.uint8)
    hdop = np.clip(np.round(1.6 - 0.055 * nsats + rng.normal(0, 0.015, ng), 2), 0.55, 1.4)
    vg = np.column_stack([at(vel[:, i], tg) for i in range(3)])
    tables["GPS[0]"] = {
        "t": tg, "I": np.zeros(ng, np.uint8), "Status": np.full(ng, 3, np.uint8),
        "GMS": (_GPS_MS0 + np.rint((tg - tb) * 1000)).astype(np.uint32),
        "GWk": np.full(ng, _GPS_WEEK, np.uint16), "NSats": nsats, "HDop": hdop,
        "Lat": gsrc.lat, "Lng": gsrc.lng, "Alt": gsrc.alt,
        "Spd": _f32(np.abs(np.hypot(vg[:, 0], vg[:, 1]) + rng.normal(0, 0.03, ng))),
        "GCrs": _wrap32(at(_hold_course(vel), tg) + rng.normal(0, 0.4, ng)),
        "VZ": _f32(-vg[:, 2] + rng.normal(0, 0.04, ng)),
        "Yaw": np.zeros(ng), "U": np.ones(ng, np.uint8),
    }

    # --- POS:沿用 POS 來源 ---
    psrc = logdata.sources["POS"]
    rel = psrc.alt - ORIGIN[2]
    tables["POS"] = {"t": psrc.t, "Lat": psrc.lat, "Lng": psrc.lng, "Alt": psrc.alt,
                     "RelHomeAlt": _f32(rel), "RelOriginAlt": _f32(rel)}

    # --- VISP:沿用 VISP 來源的位置;姿態在 VIO 自己的座標框(航向差 = 該框偏航) ---
    vsrc = logdata.sources["VISP"]
    tv = vsrc.t
    nv = tv.size
    rst_t = np.sort([e.t for e in logdata.events if e.kind == EV_RESET and e.label == "VISP"])
    tables["VISP"] = {
        "t": tv,
        "RTimeUS": np.rint((tv - tb + 37.512) * 1e6).astype(np.uint64),   # 機載電腦時鐘
        "PX": vsrc.ned[:, 0], "PY": vsrc.ned[:, 1], "PZ": vsrc.ned[:, 2],
        "Roll": _f32(at(roll_c, tv) + rng.normal(0, 0.3, nv)),
        "Pitch": _f32(at(pitch_c, tv) + rng.normal(0, 0.3, nv)),
        "Yaw": _wrap32(at(hdg, tv) + story["visp_yaw"](tv) + rng.normal(0, 0.4, nv)),
        "PErr": _f32(np.full(nv, 0.2)), "AErr": _f32(np.full(nv, 0.2)),
        "ResetCnt": (np.searchsorted(rst_t, tv, side="right") % 256).astype(np.uint8),
    }

    # --- XKF1[0] 10 Hz:EKF3 核心 0 的姿態 / 速度 / 局部位置(NED,相對 EKF 原點) ---
    tx = _grid(tb, te, 10.0)
    nx = tx.size
    ned_x = frames.enu_to_ned(np.column_stack([at(enu[:, i], tx) for i in range(3)]))
    ned_x = _f32(ned_x + np.column_stack([gm(nx, 0.1, 15.0, 0.06), gm(nx, 0.1, 15.0, 0.06),
                                          gm(nx, 0.1, 20.0, 0.08)]))
    vx = np.column_stack([at(vel[:, 1], tx), at(vel[:, 0], tx), -at(vel[:, 2], tx)])
    vx += rng.normal(0, 0.02, vx.shape)
    tables["XKF1[0]"] = {
        "t": tx, "C": np.zeros(nx, np.uint8),
        "Roll": np.round(at(roll_c, tx) + rng.normal(0, 0.08, nx), 2),
        "Pitch": np.round(at(pitch_c, tx) + rng.normal(0, 0.08, nx), 2),
        "Yaw": np.round((at(hdg, tx) + rng.normal(0, 0.15, nx)) % 360.0, 2) % 360.0,
        "VN": _f32(vx[:, 0]), "VE": _f32(vx[:, 1]), "VD": _f32(vx[:, 2]),
        "dPD": _f32(vx[:, 2] + rng.normal(0, 0.01, nx)),
        "PN": ned_x[:, 0], "PE": ned_x[:, 1], "PD": ned_x[:, 2],
        "GX": np.full(nx, 0.03), "GY": np.full(nx, -0.02), "GZ": np.full(nx, 0.01),
        "OH": np.full(nx, ORIGIN[2]),
    }

    # --- CTUN / BAT / VIBE / RCIN / RCOU 10 Hz ---
    tc = tx
    nc = nx
    arm_c = armed(tc)
    up_c, vup_c = at(up, tc), at(vup, tc)
    air_c = arm_c & (up_c > 0.05)
    r_c, p_c = np.radians(at(roll_c, tc)), np.radians(at(pitch_c, tc))
    thh = np.full(nc, 0.375)
    tho = thh * (1.0 + at(acc[:, 2], tc) / _G) / (np.cos(r_c) * np.cos(p_c))
    tho = np.where(air_c, np.clip(tho + gm(nc, 0.1, 0.5, 0.01), 0.0, 1.0), 0.0)
    thi = np.where(arm_c, np.clip(0.5 + 0.35 * np.clip(vup_c / 1.5, -1, 1)
                                  + rng.normal(0, 0.005, nc), 0, 1), 0.0)
    tables["CTUN"] = {
        "t": tc, "ThI": _f32(thi), "ABst": np.zeros(nc), "ThO": _f32(tho), "ThH": _f32(thh),
        "DAlt": _f32(np.where(arm_c, at(up, tc + 0.5), up_c)),
        "Alt": _f32(up_c + gm(nc, 0.1, 2.0, 0.05)),
        "BAlt": _f32(up_c + gm(nc, 0.1, 40.0, 0.25)),
        "DCRt": np.rint(np.where(arm_c, at(vup, tc + 0.3), 0.0) * 100).astype(np.int16),
        "CRt": np.rint((vup_c + rng.normal(0, 0.03, nc)) * 100).astype(np.int16),
    }

    curr = np.where(arm_c, np.where(air_c, 1.2 + 32.0 * tho ** 1.5, 2.5), 0.6) + rng.normal(0, 0.08, nc)
    curr = np.clip(curr, 0.0, None)
    cap = 5200.0
    used = np.cumsum(curr * 0.1) / 3.6                # mAh
    ocv = 16.75 - 1.15 * used / cap
    volt = ocv - 0.016 * curr + rng.normal(0, 0.01, nc)
    tables["BAT[0]"] = {
        "t": tc, "Inst": np.zeros(nc, np.uint8), "Volt": _f32(volt),
        "VoltR": _f32(ocv + rng.normal(0, 0.004, nc)), "Curr": _f32(curr), "CurrTot": _f32(used),
        "EnrgTot": _f32(np.cumsum(volt * curr * 0.1) / 3600.0),
        "Temp": np.round(29.0 + 6.0 * used / cap + rng.normal(0, 0.05, nc), 2),
        "Res": _f32(np.full(nc, 0.016)),
        "RemPct": np.rint(100.0 * (1.0 - used / cap)).astype(np.uint8),
    }

    base_v = np.where(air_c, 3.0 + 4.0 * tho, np.where(arm_c, 1.2, 0.02))
    tables["VIBE[0]"] = {
        "t": tc, "IMU": np.zeros(nc, np.uint8),
        "VibeX": _f32(np.abs(base_v + base_v * gm(nc, 0.1, 1.0, 0.15))),
        "VibeY": _f32(np.abs(base_v + base_v * gm(nc, 0.1, 1.0, 0.15))),
        "VibeZ": _f32(np.abs(1.7 * base_v + base_v * gm(nc, 0.1, 1.0, 0.2))),
        "Clip": np.zeros(nc, np.uint32),
    }

    mode_c = [mode_at(modes, x) or "STABILIZE" for x in tc.tolist()]
    loiter = np.array([m == "LOITER" for m in mode_c])
    stick = lambda s: np.rint(1500 + rng.normal(0, s, nc))          # noqa: E731
    thr = np.where(arm_c & (up_c > 0.03), 1500 + 400 * np.clip(vup_c / 1.5, -1, 1), 1000)
    thr = np.where(arm_c & ~loiter & (up_c > 0.03), 1500, thr)
    c7 = np.full(nc, 1100) if story["rc7_t"] is None else np.where(tc >= story["rc7_t"], 1900, 1100)
    rcin = {"t": tc, "C1": stick(2.0), "C2": stick(2.0), "C3": np.rint(thr + rng.normal(0, 2, nc)),
            "C4": stick(1.5), "C5": np.array([_MODE_PWM.get(m, 1165) for m in mode_c], float),
            "C6": np.full(nc, 1000.0), "C7": c7.astype(float), "C8": np.full(nc, 1000.0)}
    tables["RCIN"] = {k: (v if k == "t" else np.clip(v, 900, 2100).astype(np.uint16))
                      for k, v in rcin.items()}

    roll_x, pitch_x = at(roll_c, tc), at(pitch_c, tc)
    dr = 4.0 * (at(roll_c, tc + 0.1) - roll_x) + rng.normal(0, 4, nc)
    dp = 4.0 * (at(pitch_c, tc + 0.1) - pitch_x) + rng.normal(0, 4, nc)
    dy = rng.normal(0, 3, nc)
    base = np.where(air_c, 1000.0 + 1000.0 * tho, np.where(arm_c, 1100.0, 1000.0))
    on = air_c.astype(float)
    mot = [base + on * (-dr + dp + dy), base + on * (dr - dp + dy),
           base + on * (dr + dp - dy), base + on * (-dr - dp - dy)]   # Quad X:右前、左後、左前、右後
    tables["RCOU"] = {"t": tc, **{f"C{i + 1}": np.clip(np.rint(m), 1000, 2000).astype(np.uint16)
                                  for i, m in enumerate(mot)}}

    index = LogIndex.from_tables(tables, vehicle="copter", path=logdata.path, units=SIM_UNITS)
    logdata.index = index
    logdata.events = sorted(logdata.events + index_events(index), key=lambda e: e.t)
    logdata.sources["XKF1"] = TrajSource("XKF1", KIND_NED, tx.copy(), ned=ned_x.copy(),
                                         meta="模擬 EKF3 局部位置 10 Hz(σ≈0.06 m 慢漂)")
