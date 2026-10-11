"""3D 地圖(CesiumJS):衛星底圖上的飛行軌跡、依飛行模式著色、帶姿態的載具模型,
與時間軸播放同步 —— 仿 UAV Log Viewer(plot.ardupilot.org)的 3D 檢視。

- build_scene():純 Python,把 LogData(+ ATE 結果)整理成可 JSON 序列化的場景。
- Map3DWidget:QWebEngineView 載入 ui/web/map3d.html;Cesium 由 CDN 下載。
  JS → Python 走 QWebChannel(ready / seek / error),Python → JS 用 runJavaScript。

座標:各來源先轉成以 EKF 原點(ORGN)為原點的局部 ENU,再換回經緯度;
高度 h = 原點地面高 + U(ENU 天向),所以地面段貼在影像上(橢球地形時地面高 = 0,
設了 Cesium ion token 開地形時取原點處地形高)。
時間:場景內一律存「相對 t_base 的秒」;set_time() / seeked 用 log 絕對秒。
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback

import numpy as np
from PyQt6.QtCore import (QEvent, QLockFile, QObject, QSettings, Qt, QTimer, QUrl, pyqtSignal,
                          pyqtSlot)
from PyQt6.QtGui import QColor, QPixmap, QStandardItemModel
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QLabel,
                             QLineEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from core import align as align_mod
from core import clock, frames, pipeline
from core import mission as mission_mod
from core.timeline import horizontal_speed, mode_changes
from core.types import EV_RESET

from . import models, theme

if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
    # 無頭測試:沒有 GPU 可用,關掉 Chromium 的 GPU 程序(否則終端被 GL 錯誤洗版);
    # 網頁會照常載入並回報「此環境不支援 WebGL」
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --log-level=3")

try:  # 必須在建立 QApplication 之前 import(PyQt6 的限制)
    from PyQt6.QtWebChannel import QWebChannel
    from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAVE_WEBENGINE = True
    WEBENGINE_ERROR = ""
except Exception as _e:  # noqa: BLE001 — 缺套件或 import 時機太晚都改顯示說明
    HAVE_WEBENGINE = False
    WEBENGINE_ERROR = f"{type(_e).__name__}: {_e}"

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
HTML_PATH = os.path.join(WEB_DIR, "map3d.html")
CESIUM_VERSION = "1.146.0"
CESIUM_CDN = f"https://cdn.jsdelivr.net/npm/cesium@{CESIUM_VERSION}/Build/Cesium/"
SETTINGS = ("vio_ate_studio", "vio_ate_studio")
USER_AGENT = (f"vio_ate_studio/1.0 (ArduPilot log ATE viewer; PyQt6 QtWebEngine; "
              f"Cesium {CESIUM_VERSION})")
KEY_TOKEN = "cesium/ion_token"
KEY_BASE = "cesium/base_layer"

MAX_LINE = 20000       # 每條軌跡折線的最多頂點
MAX_DENSE = 60000      # 每條軌跡供逐時內插的最多樣本(播放頭位置、點擊跳轉)
MAX_ATT = 60000        # 姿態序列最多樣本
MAX_ERR = 60000        # 誤差序列最多樣本
_DEBUG = bool(os.environ.get("VIO_ATE_DEBUG"))

BASE_LAYERS = (("esri", "衛星(Esri)"), ("osm", "街道(OpenStreetMap)"), ("ion", "Cesium ion(需 token)"))
COLOR_MODES = (("mode", "飛行模式"), ("error", "ATE 誤差"), ("role", "參考 · 估測"))


# =====================================================================
# 場景(純 Python)
# =====================================================================

def _stride_idx(n: int, max_n: int) -> np.ndarray:
    """均勻抽稀的索引(含頭尾);n <= max_n 時全取。"""
    if n <= max_n:
        return np.arange(n)
    step = int(math.ceil(n / max_n))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _gap_of(t: np.ndarray) -> float:
    """內插間隙上限:取樣週期 3 倍、至少 1 秒(與 2D 播放圖一致)。"""
    if t.size < 2:
        return 1.0
    return max(1.0, 3.0 * float(np.median(np.diff(t))))


def _moving_average(t: np.ndarray, v: np.ndarray, window_s: float) -> np.ndarray:
    if v.size < 3:
        return v
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    n = max(3, int(round(window_s / max(dt, 1e-6)))) | 1
    if n >= v.size:
        return np.full_like(v, float(v.mean()))
    pad = n // 2
    vp = np.pad(v, (pad, pad), mode="edge")
    return np.convolve(vp, np.ones(n) / n, mode="valid")[:v.size]


def _pieces(t: np.ndarray, gap: float, cuts=(), keep=(), max_line: int = MAX_LINE) -> list[list[int]]:
    """折線分段:資料空洞(> gap 秒)與 cuts 時刻(VIO 重置)斷開;每段均勻抽稀,
    保留段頭尾與 keep 內的索引(模式切換、分析區間邊界),讓顏色交界落在正確位置。"""
    n = t.size
    if n < 2:
        return []
    starts = set((np.flatnonzero(np.diff(t) > gap) + 1).tolist())
    for c in cuts:
        j = int(np.searchsorted(t, c))
        if 0 < j < n:
            starts.add(j)
    bounds = [0] + sorted(starts) + [n]
    step = max(1, int(math.ceil(n / max_line)))
    keep = np.asarray(sorted(set(int(k) for k in keep if 0 <= k < n)), dtype=int)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        idx = np.arange(a, b, step)
        extra = keep[(keep >= a) & (keep < b)]
        idx = np.union1d(np.union1d(idx, extra), [b - 1])
        out.append(idx.astype(int).tolist())
    return out


def _r(a, nd: int) -> list:
    """四捨五入成 list(NaN / inf → None,JSON 才合法)。"""
    a = np.round(np.asarray(a, dtype=float), nd)
    if np.isfinite(a).all():
        return a.tolist()
    return [None if not math.isfinite(x) else x for x in a.tolist()]


def _track(t, enu, origin, t_base: float, cuts=(), keep_t=(), spd: bool = False) -> dict:
    """一條軌跡的場景資料:t(相對秒)、lat / lng / u(ENU 天向)、[spd]、折線分段。"""
    t = np.asarray(t, dtype=float)
    enu = np.asarray(enu, dtype=float)
    ok = np.isfinite(t) & np.isfinite(enu).all(axis=1)
    t, enu = t[ok], enu[ok]
    out: dict = {"n_src": int(t.size)}
    if spd and t.size >= 2:
        ts, vs = horizontal_speed(t, enu)
        vs = _moving_average(ts, vs, 1.0)          # 位置差分的速度很毛,平滑約 1 秒
        v_at = np.interp(t, ts, vs) if ts.size else np.zeros(t.size)
    else:
        v_at = None
    idx = _stride_idx(t.size, MAX_DENSE)
    t, enu = t[idx], enu[idx]
    lat, lng, _ = frames.enu_to_geodetic(enu, origin)
    gap = _gap_of(t)
    keep = [int(np.searchsorted(t, k)) for k in keep_t]
    out.update({
        "n": int(t.size),
        "t": _r(t - t_base, 3),
        "lat": _r(lat, 7), "lng": _r(lng, 7), "u": _r(enu[:, 2], 2),
        "gap": round(gap, 3),
        "pieces": _pieces(t, gap, cuts, keep),
    })
    if v_at is not None:
        out["spd"] = _r(v_at[idx], 2)
    out["_t_abs"] = t                      # 供 build_scene 後續計算(輸出前移除)
    out["_enu"] = enu
    return out


def _derived_yaw(t: np.ndarray, enu: np.ndarray, min_speed: float = 0.8) -> np.ndarray:
    """由航跡方向推估航向(度,0 = 北、順時針);低速時保持上一個航向。"""
    if t.size < 3:
        return np.zeros(t.size)
    e = _moving_average(t, enu[:, 0], 1.0)
    n = _moving_average(t, enu[:, 1], 1.0)
    ve, vn = np.gradient(e, t), np.gradient(n, t)
    sp = np.hypot(ve, vn)
    yaw = np.degrees(np.arctan2(ve, vn))
    good = sp >= min_speed
    if not good.any():
        return np.zeros(t.size)
    # 前向填補(第一個有效值之前用第一個有效值)
    idx = np.where(good, np.arange(t.size), 0)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(good))
    idx[:first] = first
    return yaw[idx]


def _series(index, key: str):
    try:
        s = index.series(key)
    except Exception:  # noqa: BLE001 — 索引實作不完整時退回推估
        return None
    if s is None:
        return None
    t, v = (np.asarray(x, dtype=float) for x in s)
    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < 2:
        return None
    return t[ok], v[ok]


def _attitude(logdata, ref_t: np.ndarray, ref_enu: np.ndarray, t_base: float) -> dict:
    """姿態序列:ATT(或 AHR2)的 Roll / Pitch / Yaw(度);沒有就由航跡推估航向。

    Yaw 先 unwrap 再抽稀,JS 端直接線性內插(不會在 359°→0° 處轉一圈)。
    """
    index = getattr(logdata, "index", None)
    if index is not None:
        for msg in ("ATT", "AHR2"):
            r = _series(index, f"{msg}.Roll")
            p = _series(index, f"{msg}.Pitch")
            y = _series(index, f"{msg}.Yaw")
            if r is None or p is None or y is None:
                continue
            t = r[0]
            order = np.argsort(t, kind="stable")
            t, roll = t[order], r[1][order]
            pitch = p[1][np.argsort(p[0], kind="stable")] if p[0].size == t.size else np.interp(t, *p)
            yt, yv = y
            yo = np.argsort(yt, kind="stable")
            yt, yv = yt[yo], np.degrees(np.unwrap(np.radians(yv[yo])))
            yaw = yv if yt.size == t.size else np.interp(t, yt, yv)
            idx = _stride_idx(t.size, MAX_ATT)
            return {"source": msg, "t": _r(t[idx] - t_base, 3), "roll": _r(roll[idx], 1),
                    "pitch": _r(pitch[idx], 1), "yaw": _r(yaw[idx], 1),
                    "gap": round(_gap_of(t[idx]), 3)}
    yaw = np.degrees(np.unwrap(np.radians(_derived_yaw(ref_t, ref_enu))))
    idx = _stride_idx(ref_t.size, MAX_ATT)
    z = np.zeros(idx.size)
    return {"source": "derived", "t": _r(ref_t[idx] - t_base, 3), "roll": _r(z, 1),
            "pitch": _r(z, 1), "yaw": _r(yaw[idx], 1), "gap": round(_gap_of(ref_t[idx]), 3)}


def _err_at(res, t: np.ndarray) -> np.ndarray:
    """每個參考樣本時刻的 ATE 3D 誤差;分析區間外 / 結果空洞內為 NaN。"""
    out = np.full(t.size, np.nan)
    if res is None or res.n < 2:
        return out
    rt, e3 = np.asarray(res.t, dtype=float), np.asarray(res.e3, dtype=float)
    m = (t >= rt[0] - 1e-6) & (t <= rt[-1] + 1e-6)
    if not m.any():
        return out
    j = np.clip(np.searchsorted(rt, t[m]), 1, rt.size - 1)
    near = np.minimum(np.abs(rt[j] - t[m]), np.abs(t[m] - rt[j - 1]))
    gap = max(1.0, 2.0 * float(res.cfg.get("max_gap_s", 0.5)))
    v = np.interp(t[m], rt, e3)
    v[near > gap] = np.nan
    out[m] = v
    return out


def _short_align(info: dict) -> str:
    label = str(info.get("label") or info.get("method", ""))
    head = label.rsplit("(", 1)[0].strip() if label.endswith(")") else label
    return head or label


def _wall0(logdata, tb: float) -> float | None:
    wc = clock.wall_clock(logdata)
    return None if wc is None else round(wc.unix_at(tb), 3)


def _mission(logdata, origin, ref_t, ref_enu, modes, tb) -> dict | None:
    """預定任務航點(log 的 CMD)→ 場景資料;沒有帶位置的航點回傳 None。"""
    wps = mission_mod.load_mission(logdata)
    if not wps:
        return None
    enu = mission_mod.waypoints_enu(wps, origin)
    lat = np.array([w.lat for w in wps])
    lng = np.array([w.lng for w in wps])
    # 目前飛往第幾個航點:log 沒有逐航點紀錄,由 AUTO 期間的航跡推估(只存「變化點」)
    radius = mission_mod.reach_radius(logdata)
    pt, pk = mission_mod.estimate_progress(
        ref_t, ref_enu, enu, mission_mod.auto_mask(modes, ref_t), radius)
    return {"source": "CMD", "radius": radius,
            "prog": {"t": _r(pt - tb, 3), "k": [int(k) for k in pk]},
            "wps": [{"n": w.n, "cmd": w.cmd, "name": w.name, "lat": round(float(la), 7),
                     "lng": round(float(lo), 7), "u": round(float(u), 2), "alt": round(w.alt, 2),
                     "frame": w.frame}
                    for w, la, lo, u in zip(wps, lat, lng, enu[:, 2])]}


def build_scene(logdata, ref_name: str, est_name: str, res=None) -> dict | None:
    """LogData(+ ATE 結果)→ 3D 地圖場景(可 JSON 序列化)。log 沒有地理原點時回傳 None。

    res 與 (ref_name, est_name) 相符時,估測軌跡套用其對齊與時間偏移、並附上誤差序列;
    否則估測畫原始座標。
    """
    if logdata is None or ref_name not in logdata.sources:
        return None
    ref = logdata.sources[ref_name]
    est = logdata.sources.get(est_name) if est_name and est_name != ref_name else None
    if ref.n < 2:
        return None
    origin = pipeline.resolve_origin(logdata, ref, est if est is not None else ref)[0]
    if origin is None:
        origin = logdata.origin            # 兩者皆局部 NED:NED 本來就相對 EKF 原點
    if origin is None:
        return None
    origin = tuple(float(x) for x in origin)
    if res is not None and (res.ref_name, res.est_name) != (ref_name, est_name):
        res = None
    tb = float(logdata.t_base)

    modes = mode_changes(logdata.events)
    resets = [(float(e.t), e.label) for e in logdata.events if e.kind == EV_RESET]
    keep_t = [m.t for m in modes]
    if res is not None and res.n:
        keep_t += [float(res.t[0]), float(res.t[-1])]

    rp = ref.enu(origin)
    r_tr = _track(ref.t, rp, origin, tb, cuts=[t for t, lab in resets if lab == ref_name],
                  keep_t=keep_t, spd=True)
    r_tr["name"] = ref_name
    rt_abs, renu = r_tr.pop("_t_abs"), r_tr.pop("_enu")
    e3 = _err_at(res, rt_abs)
    if res is not None:
        r_tr["e3"] = _r(e3, 3)

    est_d = None
    if est is not None and est.n >= 2:
        ep = est.enu(origin)
        et = np.asarray(est.t, dtype=float)
        off = 0.0
        if res is not None:
            ep = align_mod.apply_alignment(ep, res.align_info)
            off = float(res.cfg.get("offset_s", 0.0) or 0.0)
            et = et + off
        est_d = _track(et, ep, origin, tb, cuts=[t + off for t, lab in resets if lab == est_name])
        est_d.pop("_t_abs")
        est_d.pop("_enu")
        est_d["name"] = est_name
        est_d["aligned"] = res is not None and res.align_info.get("method") != "none"
        est_d["offset_s"] = round(off, 4)
        if res is None:
            est_d["note"] = "估測未對齊(原始座標)— 計算 ATE 後套用對齊"
        elif res.align_info.get("method") == "none":
            est_d["note"] = "ATE 設定為不對齊,估測照原始座標"
        else:
            est_d["note"] = f"估測已套用 {_short_align(res.align_info)}" + (
                f"、時間偏移 {off * 1000:+.0f} ms" if abs(off) >= 5e-4 else "")

    err = None
    if res is not None and res.n >= 2:
        idx = _stride_idx(res.n, MAX_ERR)
        e = np.asarray(res.e3, dtype=float)
        err = {"t": _r(np.asarray(res.t, dtype=float)[idx] - tb, 3), "e3": _r(e[idx], 3),
               "t0": round(float(res.t[0]) - tb, 3), "t1": round(float(res.t[-1]) - tb, 3),
               "gap": round(max(1.0, 2.0 * float(res.cfg.get("max_gap_s", 0.5))), 3),
               "vmin": round(float(np.nanmin(e)), 3), "vmax": round(float(np.nanmax(e)), 3),
               "rmse": round(float(res.stats["3d"]["rmse"]), 3)}

    att = _attitude(logdata, rt_abs, renu, tb)
    mission = _mission(logdata, origin, ref.t, rp, modes, tb)
    index = getattr(logdata, "index", None)
    vehicle = str(getattr(index, "vehicle", "") or "copter").lower() if index is not None else "copter"
    kind = models.model_kind(vehicle)

    present, seen = [], set()
    for m in modes:
        if m.label not in seen:
            seen.add(m.label)
            present.append([m.label, theme.mode_color(m.label)])

    return {
        "version": 1,
        "origin": [round(origin[0], 9), round(origin[1], 9), round(origin[2], 3)],
        "t_base": tb,
        "wall0": _wall0(logdata, tb),     # rel = 0 的 UTC unix 秒(HUD 顯示真實時刻);沒有 GPS 時間為 None
        "t_span": round(float(getattr(logdata, "t_last", logdata.t_end)) - tb, 3),   # 時間軸長度
        "ground": 0.0,
        "vehicle": vehicle,
        "model": {"kind": kind, "uri": models.model_data_uri(kind),
                  "lift": models.MODEL_LIFT.get(kind, 0.0)},
        "ref": r_tr,
        "est": est_d,
        "modes": [[round(m.t - tb, 3), m.label, theme.mode_color(m.label)] for m in modes],
        "mode_legend": present,
        "arm": [[round(s.t0 - tb, 3), round(s.t1 - tb, 3), s.label] for s in logdata.segments],
        "resets": [[round(t - tb, 3), lab] for t, lab in resets],
        "err": err,
        "att": att,
        "attitude": att["source"],
        "mission": mission,
        "colors": {"ref": theme.C_REF, "est": theme.C_EST, "err": theme.C_ERR, "mission": theme.C_MISSION,
                   "ink": theme.INK,
                   "ink2": theme.INK_2, "muted": theme.MUTED, "surface": theme.SURFACE,
                   "panel": theme.PANEL, "border": theme.BORDER, "ramp": list(theme.SEQ_RAMP)},
    }


def cesium_base_url() -> str:
    """Cesium 來源目錄(Build/Cesium/)。環境變數 VIO_ATE_CESIUM_BASE 可改成本機副本
    (例如 file:///C:/cesium/Build/Cesium/,離線使用)。"""
    return os.environ.get("VIO_ATE_CESIUM_BASE", "").strip() or CESIUM_CDN


def page_url():
    """map3d.html 的網址(非預設 Cesium 來源時以 ?cesium= 傳給網頁)。"""
    url = QUrl.fromLocalFile(HTML_PATH)
    base = cesium_base_url()
    if base != CESIUM_CDN:
        from PyQt6.QtCore import QUrlQuery
        q = QUrlQuery()
        q.addQueryItem("cesium", base)
        url.setQuery(q)
    return url


def scene_json(scene: dict | None) -> str:
    """緊湊 JSON(ensure_ascii:可直接嵌進 JS 原始碼)。"""
    return json.dumps(scene, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


# =====================================================================
# Qt 元件
# =====================================================================

if HAVE_WEBENGINE:
    class _Bridge(QObject):
        """QWebChannel 物件(JS 端名稱 bridge)。槽內例外一律吞掉,不讓它逃到 Qt。"""
        readyChanged = pyqtSignal(bool, str)
        seekRequested = pyqtSignal(float)
        errorRaised = pyqtSignal(str)
        statusChanged = pyqtSignal(str)

        @pyqtSlot(bool, str)
        def ready(self, ok, reason):
            try:
                self.readyChanged.emit(bool(ok), str(reason or ""))
            except Exception:  # noqa: BLE001
                traceback.print_exc()

        @pyqtSlot(float)
        def seek(self, t):
            try:
                if math.isfinite(float(t)):
                    self.seekRequested.emit(float(t))
            except Exception:  # noqa: BLE001
                traceback.print_exc()

        @pyqtSlot(str)
        def status(self, msg):
            try:
                self.statusChanged.emit(str(msg))
            except Exception:  # noqa: BLE001
                traceback.print_exc()

        @pyqtSlot(str)
        def error(self, msg):
            try:
                self.errorRaised.emit(str(msg))
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    class _Page(QWebEnginePage):
        """收集 JS console 訊息(測試用;設 VIO_ATE_DEBUG 時也印到終端)。"""

        def __init__(self, profile, parent=None):
            super().__init__(profile, parent)
            self.console: list[tuple[str, str, int, str]] = []

        def javaScriptConsoleMessage(self, level, message, line, source):
            try:
                lv = getattr(level, "name", str(level)).replace("MessageLevel", "").lower()
                self.console.append((lv, str(message), int(line), str(source)))
                if len(self.console) > 500:
                    del self.console[:100]
                if _DEBUG:
                    print(f"[map3d js {lv}] {message} ({os.path.basename(str(source))}:{line})")
            except Exception:  # noqa: BLE001
                pass

_PROFILE = None          # 共用的持久 profile(磁碟快取 Cesium 與底圖圖磚,離線時也看得到看過的區域)
_PROFILE_LOCK = None     # QLockFile:佔住 profile 目錄直到程式結束
_PROFILE_ROOT = None     # 這個程序用的 profile 目錄
_PROFILE_NOTE = ""       # 沒拿到可用的快取目錄時,狀態列給使用者的說明
PROFILE_SLOTS = 6        # webengine、webengine-2 … webengine-6
NOTE_SHARED_CACHE = "地圖快取正被其他視窗使用:這個視窗不保留圖磚快取(離線時看不到看過的區域)"


def profile_base_root() -> str:
    from PyQt6.QtCore import QStandardPaths
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericCacheLocation)
    return os.path.join(base or os.path.expanduser("~"), "vio_ate_studio", "webengine")


def lock_profile_root(base_root: str, slots: int = PROFILE_SLOTS):
    """找第一個沒被其他執行中程式佔用的 profile 目錄(base_root、base_root-2、…)。

    Chromium 的磁碟快取不能兩個程序共用(後開的會「Unable to create cache」而完全不快取),
    所以用 QLockFile 標記使用中。回傳 (目錄, 鎖);全被佔用時回傳 (None, None)。
    """
    os.makedirs(os.path.dirname(base_root), exist_ok=True)
    for k in range(1, slots + 1):
        root = base_root if k == 1 else f"{base_root}-{k}"
        lock = QLockFile(root + ".lock")
        lock.setStaleLockTime(0)       # 只以 PID 判斷殘留的鎖(程式當掉後可回收),不看存在多久
        if lock.tryLock(0):
            return root, lock
    return None, None


def _profile():
    global _PROFILE, _PROFILE_LOCK, _PROFILE_ROOT, _PROFILE_NOTE
    if _PROFILE is not None:
        from PyQt6 import sip
        if sip.isdeleted(_PROFILE):        # 前一個 QApplication 結束時一起刪掉了
            _PROFILE = None
    if _PROFILE is None:
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if _PROFILE_LOCK is None:
            try:
                _PROFILE_ROOT, _PROFILE_LOCK = lock_profile_root(profile_base_root())
            except OSError:
                traceback.print_exc()
                _PROFILE_ROOT, _PROFILE_LOCK = None, None
            if _PROFILE_LOCK is not None:
                import atexit
                atexit.register(_PROFILE_LOCK.unlock)      # 正常結束時刪掉鎖檔(當掉留下的靠 PID 回收)
        root = _PROFILE_ROOT
        if root is None:
            prof = QWebEngineProfile(app)          # off-the-record:只快取在記憶體
            _PROFILE_NOTE = NOTE_SHARED_CACHE
        else:
            prof = QWebEngineProfile("vio_ate_studio", app)
            prof.setPersistentStoragePath(os.path.join(root, "storage"))
            prof.setCachePath(os.path.join(root, "cache"))
            prof.setHttpCacheType(QWebEngineProfile.HttpCacheType.DiskHttpCache)
            prof.setHttpCacheMaximumSize(512 * 1024 * 1024)
            _PROFILE_NOTE = ""
        # 圖磚伺服器(特別是 OpenStreetMap)要求可辨識的應用程式 User-Agent,
        # 通用的 QtWebEngine 字串會被擋(回傳「Access blocked」圖磚)
        prof.setHttpUserAgent(USER_AGENT)
        _PROFILE = prof
    return _PROFILE


_OPTS_QSS = (f"QWidget#Map3DOpts{{background:{theme.PANEL};border-bottom:1px solid {theme.BORDER};}}"
             f"QLabel{{background:transparent;color:{theme.MUTED};font-size:9pt;}}"
             f"QLabel#MapStatus{{color:{theme.MUTED};}}"
             f"QCheckBox{{background:transparent;color:{theme.INK_2};font-size:9pt;}}"
             f"QComboBox{{padding:2px 8px;min-width:56px;font-size:9pt;}}"
             f"QPushButton{{padding:3px 10px;font-size:9pt;}}")


class Map3DWidget(QWidget):
    """Cesium 3D 地圖分頁。

    set_scene(build_scene(...)) 換資料;set_time(t)(log 絕對秒)移動播放頭 —— 只在元件
    可見時才送進網頁。場景連序列化都延到可見、網頁 ready 後的下一輪事件迴圈,期間連續換場景
    (隱藏時重算、載入 log 時清空又算完)只處理最後一份;前一個 setTime 還沒回來就只記住
    最新時刻(合併),播放再快也不會塞車。
    使用者點擊軌跡時發出 seeked(t)。grab_map() 截地圖本身(匯出 PNG 用)。
    """

    seeked = pyqtSignal(float)
    availabilityChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene_timer = QTimer(self)        # 換場景合併到下一輪事件迴圈才送
        self._scene_timer.setSingleShot(True)
        self._scene_timer.setInterval(0)
        self._scene_timer.timeout.connect(self._send_scene)
        self._scene: dict | None = None         # 待送的場景(送出時才序列化)
        self._scene_js: str | None = None       # 已序列化的場景(網頁重載時重送;None = 清空)
        self._scene_reason = ""                 # 沒有場景的原因('' / 'noorigin')
        self._scene_dirty = False
        self._has_scene = False
        self._color_pref = "mode"               # 使用者選的著色(暫時沒有 ATE 結果時也記著)
        self._auto_color = False
        self._status_full = ""
        self._note = ""                         # 常駐提示(例如快取被其他視窗佔用)
        self._t: float | None = None
        self._time_dirty = False
        self._pending_since: float | None = None   # setTime 呼叫送出時刻(perf_counter)
        self._ready = False
        self._available = False
        self._reason = ""
        self.timings: list[float] = []            # setTime 往返時間(秒),供測試 / 除錯
        self.errors: list[str] = []
        self.view = None
        self.page = None
        self._settings = QSettings(*SETTINGS)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._build_options())

        if not HAVE_WEBENGINE:
            self._set_controls_enabled(False)
            lay.addWidget(self._placeholder(
                "3D 地圖需要 PyQt6-WebEngine\n\n請安裝:pip install PyQt6-WebEngine\n"
                "(必須在建立 QApplication 之前 import ui.map3d)\n\n" + WEBENGINE_ERROR), 1)
            self._reason = "nowebengine"
            return

        self.view = QWebEngineView(self)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        try:
            prof = _profile()
            self._note = _PROFILE_NOTE
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            prof = QWebEngineProfile.defaultProfile()
        self._set_status("")
        self.page = _Page(prof, self.view)
        self.page.setBackgroundColor(QColor(theme.SURFACE))
        st = self.page.settings()
        st.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        st.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        st.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        st.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        st.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, False)
        self._bridge = _Bridge(self)
        self._bridge.readyChanged.connect(self._on_ready)
        self._bridge.seekRequested.connect(self._on_seek)
        self._bridge.errorRaised.connect(self._on_js_error)
        self._bridge.statusChanged.connect(self._set_status)
        self._channel = QWebChannel(self.page)
        self._channel.registerObject("bridge", self._bridge)
        self.page.setWebChannel(self._channel)
        self.page.loadStarted.connect(self._on_load_started)
        self.page.loadFinished.connect(self._on_load_finished)
        self.page.renderProcessTerminated.connect(self._on_render_terminated)
        self.view.setPage(self.page)
        lay.addWidget(self.view, 1)
        self.view.load(page_url())

        # 送出後太久沒回應(網頁重載、卡住)就放行下一個 setTime
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(1000)
        self._watchdog.timeout.connect(self._check_pending)
        self._watchdog.start()

    # ---------------- 版面 ----------------

    def _build_options(self) -> QWidget:
        bar = QWidget(self)
        bar.setObjectName("Map3DOpts")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setStyleSheet(_OPTS_QSS)
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 4, 10, 4)
        h.setSpacing(4)
        h.addWidget(QLabel("底圖"))
        self.cmb_base = QComboBox()
        for key, txt in BASE_LAYERS:
            self.cmb_base.addItem(txt, key)
        self.cmb_base.setToolTip("衛星影像:Esri World Imagery;街道:OpenStreetMap;"
                                 "Cesium ion 需要先設定 token")
        h.addWidget(self.cmb_base)
        h.addSpacing(6)
        h.addWidget(QLabel("著色"))
        self.cmb_color = QComboBox()
        for key, txt in COLOR_MODES:
            self.cmb_color.addItem(txt, key)
        self.cmb_color.setToolTip("參考軌跡的著色方式:飛行模式 / ATE 3D 誤差(只在分析區間內)/ "
                                  "序列身分(參考藍、估測橘)")
        h.addWidget(self.cmb_color)
        h.addSpacing(6)
        self.chk_est = QCheckBox("估測軌跡")
        self.chk_est.setChecked(True)
        self.chk_est.setToolTip("顯示估測軌跡(有 ATE 結果時已套用對齊與時間偏移)")
        h.addWidget(self.chk_est)
        self.chk_mission = QCheckBox("預定航線")
        self.chk_mission.setChecked(True)
        self.chk_mission.setToolTip("顯示 log 內的預定任務航點(CMD):虛線航線 + 編號;沒有任務的 log 不畫")
        h.addWidget(self.chk_mission)
        self.chk_follow = QCheckBox("跟隨飛機")
        self.chk_follow.setToolTip("視角跟著飛機移動,保留目前的旋轉與縮放(可用滑鼠環繞飛機)")
        h.addWidget(self.chk_follow)
        self.btn_all = QPushButton("全覽")
        self.btn_all.setToolTip("視角移到看得見整條參考軌跡(斜視 45°)")
        h.addWidget(self.btn_all)
        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("MapStatus")
        self.lbl_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        # 靠左、放不下時截掉句尾(加 …):開頭才是「什麼失敗了」;完整文字在 tooltip
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_status.installEventFilter(self)
        h.addWidget(self.lbl_status, 1)
        self.btn_token = QPushButton("Token…")
        self.btn_token.setToolTip("設定 Cesium ion access token:啟用全球地形(軌跡貼地形高度)"
                                  "與 ion 衛星影像")
        h.addWidget(self.btn_token)

        base = str(self._settings.value(KEY_BASE, "esri") or "esri")
        i = self.cmb_base.findData(base)
        self.cmb_base.setCurrentIndex(max(0, i))
        self._update_ion_item()
        self.cmb_base.currentIndexChanged.connect(self._on_base_changed)
        self.cmb_color.currentIndexChanged.connect(self._on_color_changed)
        self.cmb_color.activated.connect(self._on_color_activated)
        self.chk_est.toggled.connect(lambda _on: self._push_options())
        self.chk_mission.toggled.connect(lambda _on: self._push_options())
        self.chk_follow.toggled.connect(lambda _on: self._push_options())
        self.btn_all.clicked.connect(self.view_all)
        self.btn_token.clicked.connect(self._ask_token)
        self._set_error_item(False)
        return bar

    def _placeholder(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lab.setWordWrap(True)
        lab.setStyleSheet(f"background:{theme.SURFACE};color:{theme.MUTED};font-size:11pt;padding:24px;")
        return lab

    def _set_controls_enabled(self, on: bool):
        for w in (self.cmb_base, self.cmb_color, self.chk_est, self.chk_mission, self.chk_follow, self.btn_all,
                  self.btn_token):
            w.setEnabled(on)

    def _set_item_enabled(self, cmb: QComboBox, key: str, on: bool):
        model = cmb.model()
        i = cmb.findData(key)
        if i >= 0 and isinstance(model, QStandardItemModel):
            model.item(i).setEnabled(on)

    def _token(self) -> str:
        return str(self._settings.value(KEY_TOKEN, "") or "").strip()

    def _update_ion_item(self):
        has = bool(self._token())
        self._set_item_enabled(self.cmb_base, "ion", has)
        if not has and self.cmb_base.currentData() == "ion":
            self.cmb_base.setCurrentIndex(0)

    def _set_error_item(self, has_err: bool):
        """「ATE 誤差」著色只在有結果時可選。沒有結果時暫時改顯示飛行模式,但不忘記使用者的
        選擇:結果回來(例如打完新的分析終點、載入新 log 後重算)就換回 ATE 誤差。"""
        self._set_item_enabled(self.cmb_color, "error", has_err)
        i = self.cmb_color.findData("error")
        self.cmb_color.setItemData(i, None if has_err else "尚未計算 ATE",
                                   Qt.ItemDataRole.ToolTipRole)
        cur = self.cmb_color.currentData()
        want = None
        if not has_err and cur == "error":
            want = "mode"
        elif has_err and self._color_pref == "error" and cur != "error":
            want = "error"
        if want is not None:
            self._auto_color = True
            try:
                self.cmb_color.setCurrentIndex(self.cmb_color.findData(want))   # 會送選項給網頁
            finally:
                self._auto_color = False

    def _on_color_changed(self, _i: int):
        if not self._auto_color:
            self._color_pref = self.cmb_color.currentData() or "mode"
        self._push_options()

    def _on_color_activated(self, _i: int):
        # 使用者在下拉選單選了一項(包含重選目前這項):一律當成新的偏好
        self._color_pref = self.cmb_color.currentData() or "mode"

    # ---------------- 公開介面 ----------------

    @property
    def available(self) -> bool:
        """Cesium 與 WebGL 都正常(網頁已回報 ready)。"""
        return self._available

    @property
    def ready(self) -> bool:
        """網頁已初始化完畢(不論 Cesium 是否可用)。"""
        return self._ready

    @property
    def reason(self) -> str:
        """不可用的原因:''、'offline'、'nowebgl'、'nowebengine'、'error' …"""
        return self._reason

    @property
    def console_messages(self) -> list:
        return list(self.page.console) if self.page is not None else []

    @property
    def has_scene(self) -> bool:
        """目前設了場景(set_scene 收到非 None;不論網頁是否可用、是否已送出)。"""
        return self._has_scene

    @property
    def status_text(self) -> str:
        """狀態列的完整文字(畫面上放不下時會截短)。"""
        return self._status_full

    def set_scene(self, scene: dict | None, reason: str = "") -> None:
        """換場景(build_scene 的輸出;None = 清空)。

        reason 只在 scene 為 None 時有意義:'noorigin' = log 已載入但沒有地理原點 / 經緯度來源,
        網頁改顯示說明(而不是「載入 log」)。序列化與傳送都延到元件可見、網頁 ready 後的下一輪
        事件迴圈,期間連續換場景只會處理最後一份。has_scene 立即反映新狀態。
        """
        self._scene = scene
        self._scene_js = None
        self._scene_reason = "" if scene is not None else str(reason or "")
        self._has_scene = scene is not None
        self._set_error_item(bool(scene and scene.get("err")))
        self._scene_dirty = True
        self._flush_scene()

    def grab_map(self) -> QPixmap:
        """只截地圖本身(不含上方選項列),供匯出 PNG;沒有網頁元件時截整個元件。"""
        if self.view is not None:
            return self.view.grab()
        return self.grab()

    def set_time(self, t: float | None) -> None:
        """移動播放頭(log 絕對秒;None = 隱藏載具)。"""
        self._t = None if t is None or not math.isfinite(float(t)) else float(t)
        self._time_dirty = True
        self._flush_time()

    def view_all(self) -> None:
        """全覽整條參考軌跡(會取消跟隨)。"""
        if self.chk_follow.isChecked():
            self.chk_follow.blockSignals(True)
            self.chk_follow.setChecked(False)
            self.chk_follow.blockSignals(False)
            self._push_options()
        self._js("app.viewAll(0.8)")

    def run_js(self, code: str, callback=None) -> bool:
        """在網頁執行任意 JS(測試 / 除錯用);網頁尚未 ready 時回傳 False。"""
        if self.page is None or not self._ready:
            return False
        if callback is None:
            self.page.runJavaScript(code)
        else:
            self.page.runJavaScript(code, 0, callback)
        return True

    # ---------------- Python → JS ----------------

    def _js(self, code: str, callback=None) -> bool:
        if self.page is None or not (self._ready and self._available):
            return False
        try:
            if callback is None:
                self.page.runJavaScript(code)
            else:
                self.page.runJavaScript(code, 0, callback)
            return True
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return False

    def _options(self) -> dict:
        return {"base": self.cmb_base.currentData() or "esri",
                "color": self.cmb_color.currentData() or "mode",
                "showEst": self.chk_est.isChecked(),
                "showMission": self.chk_mission.isChecked(),
                "follow": self.chk_follow.isChecked(),
                "token": self._token()}

    def _push_options(self):
        self._js(f"app.setOptions({json.dumps(self._options(), ensure_ascii=True)})")

    def _flush_scene(self):
        """排程送出待送的場景(下一輪事件迴圈);不可見或網頁未 ready 時等 showEvent / ready。"""
        if self._scene_dirty and self._ready and self._available and self.isVisible():
            self._scene_timer.start()

    def _send_scene(self):
        if not self._scene_dirty or not (self._ready and self._available) or not self.isVisible():
            return
        self._scene_dirty = False
        if self._scene is not None:
            try:
                self._scene_js = scene_json(self._scene)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._scene_js = None
                self._has_scene = False
                self.errors.append(f"場景序列化失敗:{e}")
            self._scene = None             # 之後只留 JSON(重載時重送;dict 佔的記憶體大得多)
        js = self._scene_js if self._scene_js is not None else "null"
        self._js(f"app.setScene({js},{json.dumps(self._scene_reason)})")
        self._time_dirty = True
        self._flush_time()

    def _flush_time(self):
        if not self._time_dirty or not (self._ready and self._available) or not self.isVisible():
            return
        if self._scene_dirty:
            self._flush_scene()            # 先送場景(送完會再送最新時刻)
            return
        if self._pending_since is not None:
            return                         # 上一個還沒回來:回來時再送最新時刻
        self._time_dirty = False
        t = "null" if self._t is None else repr(self._t)
        self._pending_since = time.perf_counter()
        if not self._js(f"app.setTime({t})", self._on_time_done):
            self._pending_since = None

    def _on_time_done(self, _result):
        try:
            if self._pending_since is not None:
                self.timings.append(time.perf_counter() - self._pending_since)
                if len(self.timings) > 2000:
                    del self.timings[:1000]
            self._pending_since = None
            if self._time_dirty:
                self._flush_time()
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    def _check_pending(self):
        if self._pending_since is not None and time.perf_counter() - self._pending_since > 2.0:
            self._pending_since = None
            self._flush_time()

    # ---------------- JS → Python ----------------

    def _on_load_started(self):
        # 重新載入(使用者按「重新載入」、程序重啟)後要等新的 ready,再重送場景與選項
        self._ready = False
        self._available = False
        self._pending_since = None
        self._scene_dirty = True
        self._time_dirty = True

    def _on_load_finished(self, ok: bool):
        if not ok and not self._ready:
            self._reason = "error"
            self._set_status("3D 地圖網頁載入失敗")

    def _on_ready(self, ok: bool, reason: str):
        self._ready = True
        was = self._available
        self._available = bool(ok)
        self._reason = "" if ok else (reason or "error")
        self._set_controls_enabled(bool(ok))
        if ok:
            self._set_status("")
            self._push_options()
            self._scene_dirty = True
            self._flush_scene()
        else:
            self._set_status({"offline": "無法下載 Cesium(需要網路)",
                              "nowebgl": "此環境不支援 WebGL"}.get(reason, "3D 地圖無法使用"))
        if was != self._available:
            self.availabilityChanged.emit(self._available)

    def _on_seek(self, t: float):
        self.seeked.emit(float(t))

    def _on_js_error(self, msg: str):
        if msg.startswith("Cesium ion 影像") and self.cmb_base.currentData() == "ion":
            # 網頁已退回 Esri 衛星圖;不寫進設定(下次啟動仍試 ion —— token 修好就會用)
            self.cmb_base.blockSignals(True)
            self.cmb_base.setCurrentIndex(self.cmb_base.findData("esri"))
            self.cmb_base.blockSignals(False)
        self.errors.append(msg)
        if len(self.errors) > 200:
            del self.errors[:100]
        if _DEBUG:
            print(f"[map3d] {msg}")
        self._set_status(msg)

    def _on_render_terminated(self, status, code):
        self._ready = False
        self._available = False
        self._reason = "error"
        self._pending_since = None
        self._set_status(f"3D 網頁程序結束({code}),重新載入中…")
        self.errors.append(f"render process terminated: {status} {code}")
        QTimer.singleShot(500, lambda: self.view is not None and self.view.reload())

    def _set_status(self, text: str):
        """狀態列文字;空字串時顯示常駐提示(若有)。"""
        self._status_full = str(text or "") or self._note
        self.lbl_status.setToolTip(self._status_full)
        self._elide_status()

    def _elide_status(self):
        lbl = self.lbl_status
        w = lbl.contentsRect().width()
        full = self._status_full
        lbl.setText(lbl.fontMetrics().elidedText(full, Qt.TextElideMode.ElideRight, w) if w > 0 else full)

    # ---------------- 選項 ----------------

    def _on_base_changed(self, _i: int):
        key = self.cmb_base.currentData() or "esri"
        self._settings.setValue(KEY_BASE, key)
        self._push_options()

    def _ask_token(self):
        cur = self._token()
        text, ok = QInputDialog.getText(
            self, "Cesium ion token",
            "Cesium ion access token(到 ion.cesium.com 免費申請)\n"
            "設定後可用全球地形(軌跡高度改為原點地形高 + 相對高度)與 ion 衛星影像。\n"
            "清空 = 不使用 ion。",
            QLineEdit.EchoMode.Normal, cur)
        if not ok:
            return
        text = text.strip()
        if text:
            self._settings.setValue(KEY_TOKEN, text)
        else:
            self._settings.remove(KEY_TOKEN)
        self._update_ion_item()
        self._push_options()

    # ---------------- Qt 事件 ----------------

    def eventFilter(self, obj, ev):
        if obj is getattr(self, "lbl_status", None) and ev.type() in (QEvent.Type.Resize,
                                                                       QEvent.Type.FontChange,
                                                                       QEvent.Type.StyleChange):
            self._elide_status()
        return super().eventFilter(obj, ev)

    def showEvent(self, ev):
        super().showEvent(ev)
        self._time_dirty = True            # 隱藏期間沒送的播放頭,顯示時補送(場景若待送會先送)
        self._flush_scene()
        self._flush_time()
