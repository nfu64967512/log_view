"""軌跡空間對齊:平移 / 4DOF(偏航+平移)/ SE(3) / Sim(3) (Umeyama)。"""
from __future__ import annotations

import numpy as np

# (key, 中文標籤) — UI 下拉選單順序
ALIGN_METHODS = [
    ("se3", "剛體對齊 SE(3)(Umeyama,建議)"),
    ("sim3", "相似對齊 Sim(3)(含尺度)"),
    ("yaw", "偏航 + 平移(4DOF)"),
    ("translation", "僅平移(質心對齊)"),
    ("none", "不對齊(同座標框直接比較)"),
]
ALIGN_LABEL = dict(ALIGN_METHODS)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """最小平方求 dst ≈ c·R·src + t (Umeyama 1991)。回傳 (c, R, t)。"""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape[0] < 3:
        raise ValueError("對齊需要至少 3 個成對樣本")
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = dc.T @ sc / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1.0
    r = u @ s @ vt
    if with_scale:
        var_s = float((sc ** 2).sum()) / n
        c = float(np.trace(np.diag(d) @ s) / var_s) if var_s > 1e-12 else 1.0
    else:
        c = 1.0
    t = mu_d - c * (r @ mu_s)
    return c, r, t


def _yaw_translation(src: np.ndarray, dst: np.ndarray):
    """只允許繞 Z(天軸)旋轉 + 3D 平移的最小平方對齊。"""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    a = float(np.sum(sc[:, 0] * dc[:, 0] + sc[:, 1] * dc[:, 1]))
    b = float(np.sum(sc[:, 0] * dc[:, 1] - sc[:, 1] * dc[:, 0]))
    th = np.arctan2(b, a)
    cth, sth = np.cos(th), np.sin(th)
    r = np.array([[cth, -sth, 0.0], [sth, cth, 0.0], [0.0, 0.0, 1.0]])
    t = mu_d - r @ mu_s
    return 1.0, r, t


def align(est: np.ndarray, ref: np.ndarray, method: str):
    """把估測軌跡 est 對齊到參考軌跡 ref(兩者為同步後的 Nx3 ENU 成對點)。

    回傳 (est_aligned, info dict)。info 含 method/label/yaw_deg/translation/scale。
    """
    est = np.asarray(est, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if method == "none":
        c, r, t = 1.0, np.eye(3), np.zeros(3)
    elif method == "translation":
        c, r = 1.0, np.eye(3)
        t = ref.mean(axis=0) - est.mean(axis=0)
    elif method == "yaw":
        c, r, t = _yaw_translation(est, ref)
    elif method == "se3":
        c, r, t = umeyama(est, ref, with_scale=False)
    elif method == "sim3":
        c, r, t = umeyama(est, ref, with_scale=True)
    else:
        raise ValueError(f"未知對齊方式: {method}")

    est_aligned = c * (est @ r.T) + t
    yaw_deg = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
    info = {
        "method": method,
        "label": ALIGN_LABEL.get(method, method),
        "yaw_deg": yaw_deg,
        "translation": [float(v) for v in t],
        "scale": float(c),
        "rotation": r.tolist(),
    }
    return est_aligned, info


def apply_alignment(p: np.ndarray, info: dict) -> np.ndarray:
    """把 align() 求得的變換套到任意 Nx3 點(例如整份 log 的估測軌跡,供播放顯示)。"""
    p = np.asarray(p, dtype=float)
    r = np.asarray(info.get("rotation", np.eye(3)), dtype=float)
    t = np.asarray(info.get("translation", [0.0, 0.0, 0.0]), dtype=float)
    return float(info.get("scale", 1.0)) * (p @ r.T) + t
