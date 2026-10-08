"""座標框架轉換:WGS84 經緯度 ↔ 局部 ENU、NED ↔ ENU。

小範圍(< 數 km)飛行用局部切平面近似,以原點緯度的子午圈/卯酉圈曲率半徑換算,
公尺級軌跡誤差分析綽綽有餘(百公尺尺度誤差 < mm)。
"""
from __future__ import annotations

import numpy as np

_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def _local_radii(lat0_deg: float, alt0: float) -> tuple[float, float]:
    """回傳 (每弧度北向公尺, 每弧度東向公尺) @ 原點。"""
    lat0 = np.radians(lat0_deg)
    s2 = np.sin(lat0) ** 2
    rn = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * s2)                      # 卯酉圈
    rm = _WGS84_A * (1.0 - _WGS84_E2) / (1.0 - _WGS84_E2 * s2) ** 1.5  # 子午圈
    return rm + alt0, (rn + alt0) * np.cos(lat0)


def geodetic_to_enu(lat, lng, alt, origin: tuple[float, float, float]) -> np.ndarray:
    """經緯高(度/公尺) → 局部 ENU (Nx3, 公尺)。origin = (lat0, lng0, alt0)。"""
    lat0, lng0, alt0 = origin
    per_rad_n, per_rad_e = _local_radii(lat0, alt0)
    dlng = (np.asarray(lng, dtype=float) - lng0 + 180.0) % 360.0 - 180.0  # 跨 ±180 保護
    e = np.radians(dlng) * per_rad_e
    n = np.radians(np.asarray(lat, dtype=float) - lat0) * per_rad_n
    u = np.asarray(alt, dtype=float) - alt0
    return np.column_stack([e, n, u])


def enu_to_geodetic(enu: np.ndarray, origin: tuple[float, float, float]):
    """局部 ENU (Nx3) → (lat, lng, alt) 度/公尺。與 geodetic_to_enu 精確互逆。"""
    lat0, lng0, alt0 = origin
    per_rad_n, per_rad_e = _local_radii(lat0, alt0)
    enu = np.asarray(enu, dtype=float)
    lat = lat0 + np.degrees(enu[:, 1] / per_rad_n)
    lng = lng0 + np.degrees(enu[:, 0] / per_rad_e)
    alt = alt0 + enu[:, 2]
    return lat, lng, alt


def ned_to_enu(ned: np.ndarray) -> np.ndarray:
    """NED (N,E,D) → ENU (E,N,U)。"""
    ned = np.asarray(ned, dtype=float)
    return np.column_stack([ned[:, 1], ned[:, 0], -ned[:, 2]])


def enu_to_ned(enu: np.ndarray) -> np.ndarray:
    enu = np.asarray(enu, dtype=float)
    return np.column_stack([enu[:, 1], enu[:, 0], -enu[:, 2]])
