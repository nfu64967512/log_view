"""ATE 統計指標。

ATE(Absolute Trajectory Error):兩軌跡經時間同步(與可選空間對齊)後,
成對位置差向量的統計。3D 為 ‖Δ‖、水平為 ‖Δ_EN‖,各軸為帶號分量。
"""
from __future__ import annotations

import numpy as np


def _mag_stats(v: np.ndarray) -> dict:
    """非負誤差量值統計。"""
    v = np.asarray(v, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(v ** 2))),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "max": float(np.max(v)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
    }


def _axis_stats(v: np.ndarray) -> dict:
    """帶號軸向誤差統計(mean/median 保留正負 = 偏差;max/P90/P95 取絕對值)。"""
    v = np.asarray(v, dtype=float)
    a = np.abs(v)
    return {
        "rmse": float(np.sqrt(np.mean(v ** 2))),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "max": float(np.max(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
    }


def ate_stats(err: np.ndarray) -> dict:
    """err: Nx3 ENU 誤差向量(估測 − 參考)。"""
    err = np.asarray(err, dtype=float)
    e3 = np.linalg.norm(err, axis=1)
    e2 = np.linalg.norm(err[:, :2], axis=1)
    return {
        "n": int(err.shape[0]),
        "3d": _mag_stats(e3),
        "2d": _mag_stats(e2),
        "axes": {
            "E": _axis_stats(err[:, 0]),
            "N": _axis_stats(err[:, 1]),
            "U": _axis_stats(err[:, 2]),
        },
    }


def path_length(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    if p.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def path_cumulative(p: np.ndarray) -> np.ndarray:
    """逐點累積移動距離(第一點 = 0,最後一點 = path_length)。"""
    p = np.asarray(p, dtype=float)
    out = np.zeros(p.shape[0])
    if p.shape[0] >= 2:
        out[1:] = np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    return out
