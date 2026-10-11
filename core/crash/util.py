"""失事分析用的小工具(純 numpy)。"""
from __future__ import annotations

import numpy as np


def wrap180(deg):
    """角度折到 [-180, 180)。"""
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0


def in_window(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """a <= t <= b 的布林遮罩。"""
    t = np.asarray(t, dtype=float)
    return (t >= a) & (t <= b)


def runs(mask: np.ndarray, t: np.ndarray, min_dur: float = 0.0, max_gap: float | None = None):
    """連續為真的區段 [(t_start, t_end, i0, i1)]。

    t_start / t_end 是區段第一 / 最後一個為真樣本的時間;i0..i1 為索引(含)。
    相鄰樣本時間差 > max_gap(資料空洞)視為斷開;持續時間 (t_end - t_start) < min_dur 的區段丟掉。
    """
    mask = np.asarray(mask, dtype=bool)
    t = np.asarray(t, dtype=float)
    out = []
    n = mask.size
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1] and (max_gap is None or t[j + 1] - t[j] <= max_gap):
            j += 1
        if t[j] - t[i] >= min_dur:
            out.append((float(t[i]), float(t[j]), i, j))
        i = j + 1
    return out


def median_in(t: np.ndarray, v: np.ndarray, a: float, b: float, default=np.nan) -> float:
    """[a, b] 內 v 的中位數(沒有樣本回傳 default)。"""
    m = in_window(t, a, b)
    return float(np.nanmedian(v[m])) if m.any() and np.isfinite(v[m]).any() else default


def interp_at(t_src: np.ndarray, v: np.ndarray, t_dst) -> np.ndarray:
    """線性內插(超出範圍取端點值)。"""
    return np.interp(np.asarray(t_dst, dtype=float), t_src, v)


def moving_average(t: np.ndarray, v: np.ndarray, win_s: float) -> np.ndarray:
    """時間窗(以樣本時間計,約 win_s 秒、置中)的移動平均;資料空洞照常平均(用樣本數窗)。"""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.size < 3:
        return v.copy()
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(win_s / dt)))
    if k == 1:
        return v.copy()
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = np.empty_like(v)
    h = k // 2
    for i in range(v.size):  # 向量化版本在端點會偏;資料量不大(≤數十萬)時這個迴圈夠快
        lo, hi = max(0, i - h), min(v.size, i + h + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def body_z_ned(roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
    """機體 z 軸(朝下)在 NED 座標的單位向量,shape (N, 3)。

    R = Rz(yaw)·Ry(pitch)·Rx(roll) 的第三欄:
      x = cosφ·sinθ·cosψ + sinφ·sinψ
      y = cosφ·sinθ·sinψ − sinφ·cosψ
      z = cosφ·cosθ
    """
    r = np.radians(np.asarray(roll_deg, dtype=float))
    p = np.radians(np.asarray(pitch_deg, dtype=float))
    y = np.radians(np.asarray(yaw_deg, dtype=float))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.stack([cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp], axis=-1)


def thrust_vector_error_deg(roll, pitch, yaw, droll, dpitch, dyaw) -> np.ndarray:
    """實際與目標機體 z 軸的夾角(度)—— 和韌體 crash_check 的 AngErr 同定義(ArduCopter 150.8 vs 150)。

    不用歐拉角直接相減:翻轉過 ±180° 時會跳;也不用含 yaw 的完整旋轉角(正常飛行就有 ~45°)。
    """
    a = body_z_ned(roll, pitch, yaw)
    b = body_z_ned(droll, dpitch, dyaw)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def tilt_deg(roll, pitch) -> np.ndarray:
    """機體 z 軸與垂直向下的夾角(度):0 = 水平正放,90 = 側立,180 = 倒扣。"""
    return np.degrees(np.arccos(np.clip(np.cos(np.radians(roll)) * np.cos(np.radians(pitch)), -1.0, 1.0)))
