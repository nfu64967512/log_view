"""時間軸播放:播放頭、ARM 飛行段 / 模式色帶 / 事件標記、迷你資料列、分析區間選取。

一份 log 可能涵蓋整次開機(前後長段地面待機、中間好幾次起降、VIO 可能在段間重置),
這裡讓使用者在整份 log 上拖曳 / 播放,看出飛行段在哪,並挑一段當 ATE 分析區間。

- TimelineWidget:QPainter 自繪(不用 matplotlib)。由上而下:
    時間尺(下緣是分析區間的把手)
    飛行段列:ARM 綠帶;◆ EV 事件、▲ ERR 錯誤掛在綠帶下緣;▼ VIO 重置在上緣
    模式列:各飛行模式的色帶(同 UAV Log Viewer;顏色 = theme.mode_color,全 app 一致)
    迷你資料列 + 誤差列
  靜態層畫進快取 QPixmap;播放頭、區間把手與滑鼠游標每次只疊在上面,播放時很省。
  模式列永遠保留(沒有 MODE 訊息時只顯示淡色提示),換 log 時面板高度不會跳動。
- PlaybackPanel:播放控制列 + TimelineWidget + PlaybackClock(core.timeline)。
- guard_popup_keys():下拉清單開著時,方向鍵 / Home / End / 空白鍵交給清單,不觸發播放快捷鍵。

所有時間都是「log 絕對秒」(TimeUS/1e6);顯示一律相對 log 起點(t_base)。
時間軸範圍是 [t_base, t_last](log_time_range):涵蓋位置資料停了之後才記的訊息與錯誤。
Qt 事件處理與 slot 一律經過 _guarded:PyQt6 遇到未處理的例外會直接 abort 整個程式。
"""
from __future__ import annotations

import functools
import html
import math
import sys
import time
import traceback

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import (QElapsedTimer, QEvent, QLineF, QObject, QPoint, QPointF, QRect, QRectF,
                          QSize, Qt, QTimer, pyqtSignal)
from PyQt6.QtGui import (QAction, QColor, QFont, QFontMetricsF, QIcon, QKeySequence, QPainter, QPen,
                         QPixmap, QPolygonF, QTextCursor, QTextDocument)
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QMenu,
                             QPushButton, QSizePolicy, QStyle, QStyleOptionComboBox, QStylePainter,
                             QToolButton, QToolTip, QVBoxLayout, QWidget)

from core.timeline import (PlaybackClock, boundaries, format_clock, mode_at, mode_changes,
                           segment_index_at, segment_matching)
from core.types import EV_ARM, EV_DISARM, EV_ERROR, EV_EVENT, EV_RESET

from . import theme

# ---------------- 版面常數(邏輯像素) ----------------
GUTTER = 62           # 左側列名欄寬
PAD_R = 10            # 右側留白
AXIS_H = 26           # 時間尺(下緣放分析區間把手)
EVENT_H = 22          # 飛行段 + 事件標記(◆ EV / ▲ ERR 在綠帶下緣、▼ 重置在上緣)
MODE_H = 16           # 飛行模式色帶
ROW_H = 18            # 每條資料列
PAD_B = 4
HANDLE_W, HANDLE_H = 9, 10   # 分析區間把手(時間尺下緣,往區間內側伸出)
EDGE_GRAB_PX = 6      # 分析區間邊界 / 播放頭可抓取的距離
SNAP_PX = 6           # 拖曳邊界時吸附到飛行段邊界 / 事件 / 播放頭的距離
MARK_PX = 5           # 滑鼠提示列出這個距離內的事件
MIN_VIEW_S = 0.05     # 最大縮放:可見範圍至少這麼多秒(和欄位圖的最小跨度一致,兩邊縮放才對得上)
MIN_WINDOW_S = 0.5    # 分析區間最短長度
MODE_ALPHA = 190      # 模式色帶不透明度(約 0.75)
_EMPTY_HINT = "載入 log 後可在此播放與選段"
_ERR_HINT = "計算 ATE 後顯示"
_NO_MODE_HINT = "此 log 沒有 MODE 訊息"
_TICK_STEPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800,
               3600, 7200, 10800, 21600, 43200)
_DIM = QColor(0, 0, 0, 125)          # 分析區間外變暗
_SNAP_KINDS = (EV_ARM, EV_DISARM, EV_RESET, EV_EVENT, EV_ERROR)
_JUMP_KINDS = (EV_ARM, EV_DISARM, EV_RESET, EV_EVENT, EV_ERROR)


# ---------------- 例外保護 ----------------

handler_errors: list[str] = []       # 被攔下的例外(最近 50 筆;測試用來確認沒有錯誤被吞掉)
_seen_errors: set[str] = set()


def _report_exception() -> None:
    txt = traceback.format_exc()
    handler_errors.append(txt)
    del handler_errors[:-50]
    if txt not in _seen_errors:          # 同一個錯誤只印一次(繪圖錯誤會每格重複)
        _seen_errors.add(txt)
        sys.stderr.write("[ui.timeline] 事件處理發生例外(已攔下):\n" + txt)


def _guarded(fn):
    """Qt 事件處理 / slot 的保護:例外只記錄不外拋(PyQt6 遇到未處理例外會 abort)。

    連到訊號時被包的函式要能接受訊號的全部參數(lambda 用 *_)。
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            _report_exception()
            return None
    return wrapper


def _qc(hex_color: str, alpha: int = 255) -> QColor:
    c = QColor(hex_color)
    c.setAlpha(alpha)
    return c


def _luminance(rgb) -> float:
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _ink_on(hex_color: str, alpha: int = MODE_ALPHA) -> QColor:
    """半透明色帶(疊在 SURFACE 上)上的文字色:深 / 淺取對比較高者。"""
    c, s = QColor(hex_color), QColor(theme.SURFACE)
    a = alpha / 255.0
    rgb = [a * x + (1.0 - a) * y for x, y in zip((c.redF(), c.greenF(), c.blueF()),
                                                 (s.redF(), s.greenF(), s.blueF()))]
    lum = _luminance(rgb)
    dark = QColor(theme.PLANE)
    lum_d = _luminance((dark.redF(), dark.greenF(), dark.blueF()))
    return dark if (lum + 0.05) / (lum_d + 0.05) >= 1.05 / (lum + 0.05) else QColor(theme.INK)


def _tick_decimals(step: float) -> int:
    """刻度間距 → 標籤小數位數(0.01~0.05 秒兩位、0.1~0.5 秒一位、整秒不帶)。"""
    return 2 if step < 0.1 - 1e-9 else (1 if step < 1 else 0)


def log_time_range(logdata) -> tuple[float, float]:
    """時間軸 / 播放範圍 (t_base, t_last):涵蓋位置來源、事件與文字訊息(MSG / ERR…)。

    LOG_DISARMED=0 時位置資料在 DISARM 後就停,訊息與錯誤還會繼續記;t_end 只看位置來源
    (ATE 用),時間軸要到 t_last 才看得到、跳得到那些事件。沒有 t_last 的舊物件退回 t_end。
    """
    lo, end = float(logdata.t_base), float(logdata.t_end)
    try:
        last = float(getattr(logdata, "t_last", end))
    except Exception:  # noqa: BLE001 — 索引壞掉就只用位置來源的範圍
        last = end
    hi = max(end, last) if np.isfinite(last) else end
    return lo, max(hi, lo)


def _fmt_val(v: float | None, unit: str) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    a = abs(v)
    txt = f"{v:.0f}" if a >= 100 else (f"{v:.1f}" if a >= 10 else f"{v:.2f}")
    return f"{txt} {unit}".rstrip()


class _Row:
    """一條迷你資料列:已排序、去 NaN 的 (t, v),外加正規化範圍與分箱快取。"""

    def __init__(self, label: str, t, v, color: str, unit: str = ""):
        t = np.asarray(t, dtype=float).ravel()
        v = np.asarray(v, dtype=float)
        if v.ndim == 2:                       # 給了 Nx3 向量 → 取長度
            v = np.linalg.norm(v, axis=1)
        v = v.ravel()
        n = min(t.size, v.size)
        t, v = t[:n], v[:n]
        ok = np.isfinite(t) & np.isfinite(v)
        t, v = t[ok], v[ok]
        if t.size > 1 and np.any(np.diff(t) < 0):
            o = np.argsort(t, kind="stable")
            t, v = t[o], v[o]
        self.label, self.color, self.unit = str(label), str(color), str(unit or "")
        self.t, self.v = t, v
        # 資料空洞門檻:超過就不連線(例如地面待機時 VISP 沒資料)
        med = float(np.median(np.diff(t))) if t.size > 2 else 0.2
        self.gap = max(1.0, 5.0 * med)
        # 正規化:1–99 百分位;非負序列(速度、誤差)錨定在 0
        if v.size:
            lo, hi = (float(x) for x in np.percentile(v, [1.0, 99.0]))
        else:
            lo, hi = 0.0, 1.0
        if lo >= 0.0 and lo <= 0.5 * hi:
            lo = 0.0
        min_span = max(0.5, 0.05 * max(abs(lo), abs(hi)))
        if hi - lo < min_span:
            if lo == 0.0:
                hi = min_span
            else:
                c = 0.5 * (lo + hi)
                lo, hi = c - min_span / 2, c + min_span / 2
        self.lo, self.hi = lo, hi
        self._cache_key = None
        self._cache = None
        self._shape_key = None
        self._shapes: list = []

    def envelope(self, v0: float, v1: float, ncols: int):
        key = (ncols, v0, v1)
        if self._cache_key != key:
            self._cache = _envelope(self.t, self.v, v0, v1, ncols, self.gap)
            self._cache_key = key
        return self._cache

    def shapes(self, v0: float, v1: float, ncols: int, x0: float, ya: float, yb: float):
        """畫面用的 (每欄一個包絡矩形, 平均線線段),依視窗 / 尺寸快取。

        包絡用 drawRects(比同形狀的多邊形填色快約 30 倍);平均線拆成獨立線段
        用 drawLines —— 高 DPI 下 2 像素寬的鋸齒折線走 QStroker 會慢到數十 ms。
        拖曳區間邊界時靜態層每次滑鼠移動都重畫,要夠快才跟得上。
        """
        key = (ncols, v0, v1, x0, ya, yb)
        if self._shape_key == key:
            return self._shapes
        mn, mx, mean = self.envelope(v0, v1, ncols)
        span = self.hi - self.lo

        def ymap(vals):
            return yb - np.clip((vals - self.lo) / span, 0.0, 1.0) * (yb - ya)

        ok = np.isfinite(mn)
        cols = np.flatnonzero(ok)
        ytop = ymap(mx[cols])
        ybot = np.maximum(ymap(mn[cols]), ytop + 1.0)
        rects = [QRectF(x, a, 1.0, b - a)
                 for x, a, b in zip((x0 + cols).tolist(), ytop.tolist(), ybot.tolist())]
        lines = []
        for run in _runs(ok):
            if run.size > 1:
                xs = (x0 + run + 0.5).tolist()
                ym = ymap(mean[run]).tolist()
                lines += [QLineF(xs[i], ym[i], xs[i + 1], ym[i + 1]) for i in range(len(xs) - 1)]
        self._shape_key, self._shapes = key, (rects, lines)
        return self._shapes

    def value_at(self, t: float, tol: float = 0.0) -> float | None:
        tt = self.t
        n = tt.size
        if n == 0:
            return None
        j = int(np.searchsorted(tt, t))
        if j < n and abs(tt[j] - t) <= max(tol, 1e-9):
            return float(self.v[j])
        if j == 0:
            return float(self.v[0]) if tt[0] - t <= tol else None
        if j >= n:
            return float(self.v[-1]) if t - tt[-1] <= tol else None
        a, b = j - 1, j
        if tt[b] - tt[a] > self.gap:
            return None
        w = (t - tt[a]) / (tt[b] - tt[a])
        return float(self.v[a] * (1.0 - w) + self.v[b] * w)


def _envelope(t: np.ndarray, v: np.ndarray, lo: float, hi: float, n: int, gap: float):
    """把 (t, v) 壓成 n 個像素欄:每欄 (min, max, mean),無資料為 NaN。

    視為分段線性訊號:欄邊界的內插值也算進該欄(放大到每欄不到一個樣本時,
    線仍連續);兩樣本間距 > gap 的空洞不內插,該處欄位留空。
    """
    n = max(1, int(n))
    nan = np.full(n, np.nan)
    if t.size == 0 or hi <= lo:
        return nan, nan.copy(), nan.copy()
    edges = np.linspace(lo, hi, n + 1)

    def interp(x):
        j = np.searchsorted(t, x, side="right")
        jj = np.clip(j, 1, t.size - 1) if t.size > 1 else np.zeros_like(j)
        ok = (j > 0) & (j < t.size)
        if t.size > 1:
            ok &= (t[jj] - t[jj - 1]) <= gap
        ok |= np.isclose(x, t[-1], rtol=0.0, atol=1e-9)
        out = np.interp(x, t, v)
        out[~ok] = np.nan
        return out

    ev = interp(edges)
    cv = interp(0.5 * (edges[:-1] + edges[1:]))
    el, er = ev[:-1], ev[1:]

    cnt = np.zeros(n)
    s = np.zeros(n)
    smn = np.full(n, np.inf)
    smx = np.full(n, -np.inf)
    a = int(np.searchsorted(t, lo, side="left"))
    b = int(np.searchsorted(t, hi, side="right"))
    if b > a:
        ts, vs = t[a:b], v[a:b]
        idx = np.clip(((ts - lo) / (hi - lo) * n).astype(int), 0, n - 1)
        cnt = np.bincount(idx, minlength=n).astype(float)
        s = np.bincount(idx, weights=vs, minlength=n)
        starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
        cols = idx[starts]
        smn[cols] = np.minimum.reduceat(vs, starts)
        smx[cols] = np.maximum.reduceat(vs, starts)

    with np.errstate(invalid="ignore"):
        mn = np.fmin(smn, np.fmin(el, er))
        mx = np.fmax(smx, np.fmax(el, er))
        mean = np.where(cnt >= 2, s / np.maximum(cnt, 1), cv)
        mean = np.where(np.isfinite(mean), mean, np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan))
    valid = (cnt > 0) | (np.isfinite(el) & np.isfinite(er))
    mn[~valid] = np.nan
    mx[~valid] = np.nan
    mean[~valid] = np.nan
    mean = np.clip(mean, mn, mx)
    return mn, mx, mean


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    return np.split(idx, cuts)


# =====================================================================
#                              時間軸元件
# =====================================================================

class TimelineWidget(QWidget):
    """可縮放的時間軸:時間尺 / 飛行段 / 模式 / 迷你資料列 / 誤差列 + 播放頭與分析區間。"""

    seeked = pyqtSignal(float)              # 點擊或拖曳播放頭(log 絕對秒),拖曳中連續發出
    windowEdited = pyqtSignal(float, float) # 拖完分析區間邊界(放開滑鼠時才發)
    segmentActivated = pyqtSignal(int)      # 雙擊 ARM 段(segments 索引)
    viewChanged = pyqtSignal(float, float)  # 使用者改了可見範圍(滾輪、平移、全覽、縮放到…、播放翻頁);
                                            # set_view() / set_log() 等程式設定不發,可安心雙向連動

    MIN_VIEW_S = MIN_VIEW_S                 # 可見範圍最短跨度(秒);set_view 只把更短的撐到這個長度

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._has_log = False
        self._lo = self._hi = 0.0
        self._v0 = self._v1 = 0.0
        self._segments: list = []
        self._events: list = []
        self._modes: list = []
        self._resets: list = []
        self._marks: list = []            # EV 事件 + ERR 錯誤
        self._rows: list[_Row] = []
        self._err: _Row | None = None
        self._win: tuple[float, float] | None = None
        self._t: float | None = None
        self._hover: QPointF | None = None
        self._hover_handle: int | None = None
        self._drag = None                 # None | "seek" | 0(起點) | 1(終點)
        self._drag_win: list[float] | None = None
        self._drag_orig: tuple[float, float] | None = None
        self._static: QPixmap | None = None
        self._update_height()

    # ---------------- 公開介面 ----------------

    def set_log(self, lo: float, hi: float, segments: list, events: list) -> None:
        """設定整份 log 的範圍與事件;縮放回全覽。(不清除資料列,請另呼叫 set_rows / set_error)"""
        lo, hi = float(lo), float(hi)
        self._has_log = True
        self._lo, self._hi = lo, max(hi, lo + 1e-3)
        self._v0, self._v1 = self._lo, self._hi
        self._segments = list(segments or [])
        self._events = sorted(events or [], key=lambda e: e.t)
        self._modes = mode_changes(self._events)
        self._resets = [e for e in self._events if e.kind == EV_RESET]
        self._marks = [e for e in self._events if e.kind in (EV_EVENT, EV_ERROR)]
        self._win = None
        self._t = self._lo
        self._drag = None
        self._drag_win = None
        self._hover_handle = None
        self._update_height()
        self._invalidate()

    def set_rows(self, rows: list) -> None:
        """rows:[(列名, t 絕對秒, 值, 色碼, 單位), …]"""
        n_old = len(self._rows)
        out = []
        for r in rows or []:
            label, t, v, color = r[0], r[1], r[2], r[3]
            unit = r[4] if len(r) > 4 else ""
            out.append(_Row(label, t, v, color, unit))
        self._rows = out
        if len(out) != n_old:
            self._update_height()
        self._invalidate()

    def set_error(self, t, e) -> None:
        """誤差列(只涵蓋 ATE 結果的時間範圍);t 或 e 為 None 時清除。"""
        if t is None or e is None or len(t) == 0:
            self._err = None
        else:
            self._err = _Row("誤差", t, e, theme.C_ERR, "m")
        self._invalidate()

    def set_window(self, t0: float, t1: float) -> None:
        t0, t1 = sorted((float(t0), float(t1)))
        self._win = (t0, t1)
        if self._drag in (0, 1):            # 拖曳中被外部更新:以拖曳預覽為準
            return
        self._invalidate()

    def set_playhead(self, t: float) -> None:
        """播放時每秒呼叫約 30 次:只記下時間並重畫(靜態層用快取)。"""
        self._t = float(t)
        self.update()

    def clear(self) -> None:
        self._has_log = False
        self._lo = self._hi = self._v0 = self._v1 = 0.0
        self._segments, self._events, self._modes, self._resets = [], [], [], []
        self._marks = []
        self._rows, self._err, self._win, self._t = [], None, None, None
        self._drag = None
        self._drag_win = None
        self._hover = None
        self._hover_handle = None
        self.unsetCursor()
        self._update_height()
        self._invalidate()

    def view_range(self) -> tuple[float, float]:
        return self._v0, self._v1

    def reset_view(self) -> None:
        """回到全覽(程式呼叫,不發 viewChanged)。"""
        self.set_view(self._lo, self._hi)

    # --- 額外輔助(PlaybackPanel 使用) ---

    def set_view(self, v0: float, v1: float) -> tuple[float, float]:
        """設定可見範圍;不發 viewChanged。回傳實際套用的範圍(連動的圖可拿來對齊)。

        跨度 ≥ MIN_VIEW_S 且在 log 範圍內時原樣套用;太短就以中心撐到 MIN_VIEW_S,
        超出 log 範圍就平移回來(跨度不變),比整份 log 長就是全覽。
        """
        if not self._has_log:
            return self.view_range()
        a, b = sorted((float(v0), float(v1)))
        if not (math.isfinite(a) and math.isfinite(b)):
            return self.view_range()
        full = self._hi - self._lo
        min_span = min(self.MIN_VIEW_S, full)
        if b - a < min_span:
            c = 0.5 * (a + b)
            a, b = c - 0.5 * min_span, c + 0.5 * min_span
        if b - a >= full:
            a, b = self._lo, self._hi
        elif a < self._lo:
            a, b = self._lo, self._lo + (b - a)
        elif b > self._hi:
            a, b = self._hi - (b - a), self._hi
        if (a, b) != (self._v0, self._v1):
            self._v0, self._v1 = a, b
            self._invalidate()
        return self._v0, self._v1

    def _user_view(self, v0: float, v1: float) -> None:
        """使用者操作改變可見範圍:有變才發 viewChanged。"""
        old = (self._v0, self._v1)
        self.set_view(v0, v1)
        if (self._v0, self._v1) != old:
            self.viewChanged.emit(self._v0, self._v1)

    def is_zoomed(self) -> bool:
        return self._has_log and (self._v1 - self._v0) < (self._hi - self._lo) * (1 - 1e-9)

    def ensure_visible(self, t: float) -> None:
        """播放頭跑出可見範圍時翻頁(縮放中才有作用;會發 viewChanged 讓連動的圖跟著翻)。"""
        if not self.is_zoomed() or self._v0 <= t <= self._v1:
            return
        span = self._v1 - self._v0
        v0 = t - 0.1 * span if t > self._v1 else t - 0.9 * span
        self._user_view(v0, v0 + span)

    def is_scrubbing(self) -> bool:
        return self._drag == "seek"

    # ---------------- 座標 ----------------

    def _x0(self) -> float:
        return float(GUTTER)

    def _x1(self) -> float:
        return float(max(GUTTER + 20, self.width() - PAD_R))

    def _t2x(self, t: float) -> float:
        span = max(self._v1 - self._v0, 1e-9)
        return self._x0() + (t - self._v0) / span * (self._x1() - self._x0())

    def _x2t(self, x: float, clamp: bool = True) -> float:
        x0, x1 = self._x0(), self._x1()
        if clamp:
            x = min(max(x, x0), x1)
        return self._v0 + (x - x0) / (x1 - x0) * (self._v1 - self._v0)

    def _row_count(self) -> int:
        return (len(self._rows) if self._has_log else max(len(self._rows), 2)) + 1

    def _mode_top(self) -> int:
        return AXIS_H + EVENT_H

    def _rows_top(self) -> int:
        return AXIS_H + EVENT_H + MODE_H

    def _height(self) -> int:
        return self._rows_top() + self._row_count() * ROW_H + PAD_B

    def _update_height(self) -> None:
        """高度由列數決定(固定高度:放進任何版面都不會被拉伸或壓扁)。"""
        h = self._height()
        if self.minimumHeight() != h or self.maximumHeight() != h:
            self.setFixedHeight(h)

    def sizeHint(self) -> QSize:
        return QSize(800, self._height())

    def minimumSizeHint(self) -> QSize:
        return QSize(320, self._height())

    def _all_rows(self) -> list:
        """畫面上的列:資料列 + 誤差列(誤差列在有 log 時一定保留位置)。"""
        return list(self._rows) + [self._err]

    def _eff_window(self):
        return tuple(self._drag_win) if self._drag in (0, 1) and self._drag_win else self._win

    def _invalidate(self) -> None:
        self._static = None
        self.update()

    def _font(self, pt: float, bold: bool = False) -> QFont:
        f = QFont(self.font())
        f.setPointSizeF(pt)
        f.setBold(bold)
        return f

    # --- 版面查詢(繪圖、滑鼠與測試共用) ---

    def _event_band_y(self) -> tuple[float, float]:
        """飛行段綠帶的上下緣。"""
        return AXIS_H + 3.0, AXIS_H + EVENT_H - 5.0

    def _mark_y(self) -> float:
        """◆ / ▲ 事件標記的中心高度:跨在綠帶下緣,和置中的 ARM 標籤文字錯開。"""
        return AXIS_H + EVENT_H - 4.5

    def _mode_spans(self) -> list[tuple[float, float, str]]:
        """[(起, 迄, 模式)]:每次切換到下一次切換(最後一段到 log 結尾),裁在 log 範圍內。"""
        out = []
        n = len(self._modes)
        for i, m in enumerate(self._modes):
            a = max(m.t, self._lo)
            b = min(self._modes[i + 1].t if i + 1 < n else self._hi, self._hi)
            if b > a:
                out.append((a, b, m.label))
        return out

    def _mode_bands(self) -> list[tuple[float, float, str, str]]:
        """可見範圍內的模式色帶 [(x 起, x 迄, 模式, 色碼)],已裁到繪圖區。"""
        if not self._has_log:
            return []
        x0, x1 = self._x0(), self._x1()
        return [(max(self._t2x(a), x0), min(self._t2x(b), x1), name, theme.mode_color(name))
                for a, b, name in self._mode_spans() if b > self._v0 and a < self._v1]

    def _mode_span_at(self, t: float):
        for a, b, name in self._mode_spans():
            if a <= t < b or (t == b == self._hi):
                return a, b, name
        return None

    def _event_marks(self) -> list[tuple[float, str, str]]:
        """可見範圍內的事件標記 [(x, 種類, 標籤)](EV 事件、ERR 錯誤)。"""
        x0, x1 = self._x0(), self._x1()
        out = []
        for e in self._marks:
            x = self._t2x(e.t)
            if x0 - 4 <= x <= x1 + 4:
                out.append((x, e.kind, e.label))
        return out

    def _handle_rect(self, side: int, win) -> QRectF | None:
        """分析區間把手:時間尺下緣、往區間內側伸出的小旗(邊界不在可見範圍時為 None)。"""
        x = self._t2x(win[side])
        if x < self._x0() - 1 or x > self._x1() + 1:
            return None
        hx = x - 1.0 if side == 0 else x - HANDLE_W + 1.0
        return QRectF(hx, AXIS_H - HANDLE_H, HANDLE_W, HANDLE_H)

    # ---------------- 繪圖 ----------------

    @_guarded
    def resizeEvent(self, ev):
        self._static = None
        super().resizeEvent(ev)

    @_guarded
    def paintEvent(self, ev):
        dpr = self.devicePixelRatioF()
        if (self._static is None or self._static.devicePixelRatio() != dpr
                or self._static.width() != max(1, round(self.width() * dpr))
                or self._static.height() != max(1, round(self.height() * dpr))):
            self._static = self._render_static(dpr)
        p = QPainter(self)
        try:
            p.drawPixmap(0, 0, self._static)
            if self._has_log:
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                self._paint_dynamic(p)
        finally:
            p.end()

    def _render_static(self, dpr: float) -> QPixmap:
        w, h = self.width(), self.height()
        pm = QPixmap(max(1, round(w * dpr)), max(1, round(h * dpr)))
        pm.setDevicePixelRatio(dpr)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            p.setPen(QPen(QColor(theme.BORDER), 1.0))
            p.setBrush(QColor(theme.SURFACE))
            p.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 6, 6)
            if not self._has_log:
                p.setPen(QColor(theme.MUTED))
                p.setFont(self._font(9.0))
                p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, _EMPTY_HINT)
                return pm
            self._paint_gutter(p)
            self._paint_axis(p)
            self._paint_events(p)
            self._paint_modes(p)
            self._paint_rows(p)
            self._paint_window(p)
        finally:
            p.end()
        return pm

    def _paint_gutter(self, p: QPainter):
        f = self._font(8.5)
        p.setFont(f)
        fm = QFontMetricsF(f)
        p.setPen(QColor(theme.MUTED))
        p.drawText(QRectF(10, AXIS_H, GUTTER - 14, EVENT_H - 2),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "飛行段")
        p.drawText(QRectF(10, self._mode_top(), GUTTER - 14, MODE_H),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "模式")
        y = self._rows_top()
        for k, row in enumerate(self._all_rows()):
            yt = y + k * ROW_H
            # 列分隔線
            p.setPen(QPen(QColor(theme.GRID), 1.0))
            p.drawLine(QPointF(6, yt + 0.5), QPointF(self.width() - 6, yt + 0.5))
            label = row.label if row is not None else "誤差"
            color = row.color if row is not None else theme.C_ERR
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(color))
            p.drawRoundedRect(QRectF(10, yt + ROW_H / 2 - 3, 6, 6), 1.5, 1.5)
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(21, yt, GUTTER - 23, ROW_H),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       fm.elidedText(label, Qt.TextElideMode.ElideRight, GUTTER - 23))

    def _ticks(self):
        x0, x1 = self._x0(), self._x1()
        span = max(self._v1 - self._v0, 1e-9)
        px_per_s = (x1 - x0) / span
        fm = QFontMetricsF(self._font(8.0))
        total = max(self._hi - self._lo, 1.0)
        long_log = self._v1 - self._lo >= 3600      # 超過 1 小時 format_clock 不顯示小數
        min_px: dict[int, float] = {}               # 小數位數 → 標籤最小間距(帶小數的標籤較寬)
        step, dec = _TICK_STEPS[-1], 0
        for s in _TICK_STEPS:
            d = _tick_decimals(s)
            if d and long_log:
                continue
            if d not in min_px:
                min_px[d] = fm.horizontalAdvance(format_clock(total, d)) + 30
            if s * px_per_s >= min_px[d]:
                step, dec = s, d
                break
        k0 = math.ceil((self._v0 - self._lo) / step - 1e-9)
        k1 = math.floor((self._v1 - self._lo) / step + 1e-9)
        return [(self._lo + k * step, round(k * step, 6), dec) for k in range(k0, k1 + 1)]

    def _paint_axis(self, p: QPainter):
        x0, x1 = self._x0(), self._x1()
        h = self.height()
        f = self._font(8.0)
        fm = QFontMetricsF(f)
        p.setFont(f)
        # 縮放中:左上角(列名欄)畫一條迷你捲軸,示意目前看的是整份 log 的哪一段
        if self.is_zoomed():
            full = self._hi - self._lo
            ga, gb = 10.0, GUTTER - 12.0
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.BASELINE))
            p.drawRoundedRect(QRectF(ga, 11, gb - ga, 4), 2, 2)
            a = ga + (self._v0 - self._lo) / full * (gb - ga)
            b = ga + (self._v1 - self._lo) / full * (gb - ga)
            p.setBrush(QColor(theme.INK_2))
            p.drawRoundedRect(QRectF(min(a, gb - 3), 11, max(b - a, 3.0), 4), 2, 2)
        for t, rel, dec in self._ticks():
            x = round(self._t2x(t)) + 0.5
            p.setPen(QPen(QColor(theme.GRID), 1.0))
            p.drawLine(QPointF(x, AXIS_H), QPointF(x, h - PAD_B))
            p.setPen(QPen(QColor(theme.BASELINE), 1.0))
            p.drawLine(QPointF(x, AXIS_H - 5), QPointF(x, AXIS_H))
            txt = format_clock(rel, dec)
            tw = fm.horizontalAdvance(txt)
            lx = min(max(x - tw / 2, x0 - 8), self.width() - 4 - tw)
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(lx, 2, tw + 2, 14),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)
        p.setPen(QPen(QColor(theme.BASELINE), 1.0))
        p.drawLine(QPointF(x0, AXIS_H - 0.5), QPointF(x1, AXIS_H - 0.5))

    def _paint_events(self, p: QPainter):
        x0, x1 = self._x0(), self._x1()
        ya, yb = self._event_band_y()
        occupied: list[tuple[float, float]] = []

        def free(a: float, b: float) -> bool:
            return a >= x0 - 1 and b <= x1 + 1 and all(b <= oa or a >= ob for oa, ob in occupied)

        f = self._font(8.0)
        fm = QFontMetricsF(f)
        p.setFont(f)
        good = QColor(theme.S_GOOD)
        # --- ARM 飛行段 ---
        for s in self._segments:
            xa, xb = self._t2x(s.t0), self._t2x(s.t1)
            if xb < x0 or xa > x1:
                continue
            ca, cb = max(xa, x0), min(xb, x1)
            if cb - ca < 2:
                cb = ca + 2
            jag_l = s.open_start and xa >= x0 - 0.5
            jag_r = s.open_end and xb <= x1 + 0.5
            left = self._edge_pts(ca, ya, yb, +1) if jag_l else [QPointF(ca, yb), QPointF(ca, ya)]
            right = self._edge_pts(cb, ya, yb, -1) if jag_r else [QPointF(cb, ya), QPointF(cb, yb)]
            if jag_l:
                left = left[::-1]           # 由下往上
            poly = QPolygonF([QPointF(ca, ya), *right, QPointF(ca, yb), *left])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_qc(theme.S_GOOD, 90))
            p.drawPolygon(poly)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(good, 1.0))
            p.drawLine(QPointF(ca, ya + 0.5), QPointF(cb, ya + 0.5))
            p.drawLine(QPointF(ca, yb - 0.5), QPointF(cb, yb - 0.5))
            if xa >= x0 - 0.5:
                p.drawPolyline(QPolygonF(left))
            if xb <= x1 + 0.5:
                p.drawPolyline(QPolygonF(right))
            # 標籤:寬夠就附時長,不夠就省略
            room = cb - ca - 10
            short = s.label.split()[-1] if s.label.split() else ""   # 「ARM 3」→「3」
            txt = next((c for c in (f"{s.label} · {format_clock(s.duration, 0)}", s.label, short)
                        if c and fm.horizontalAdvance(c) <= room), "")
            if txt:
                tw = fm.horizontalAdvance(txt)
                p.setPen(QColor(theme.INK))
                p.drawText(QRectF(ca + 5, ya, tw + 2, yb - ya),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, txt)
                occupied.append((ca + 3, ca + 7 + tw))
        # --- EV 事件(◆)/ ERR 錯誤(▲):跨在綠帶下緣,和 ARM 標籤錯開;錯誤畫在上層 ---
        ym = self._mark_y()
        p.setPen(QPen(QColor(theme.SURFACE), 1.0))
        marks = self._event_marks()
        p.setBrush(QColor(theme.INK_2))
        for x, kind, _ in marks:
            if kind == EV_EVENT:
                p.drawPolygon(QPolygonF([QPointF(x, ym - 4.0), QPointF(x + 3.6, ym),
                                         QPointF(x, ym + 4.0), QPointF(x - 3.6, ym)]))
        p.setBrush(QColor(theme.S_CRIT))
        for x, kind, _ in marks:
            if kind == EV_ERROR:
                p.drawPolygon(QPolygonF([QPointF(x, ym - 4.5), QPointF(x + 4.3, ym + 3.5),
                                         QPointF(x - 4.3, ym + 3.5)]))
        # --- VIO 重置(▼ + 標籤) ---
        warn = QColor(theme.S_WARN)
        fs = self._font(8.0)
        fms = QFontMetricsF(fs)
        for e in self._resets:
            x = self._t2x(e.t)
            if x < x0 - 4 or x > x1 + 4:
                continue
            pen = QPen(_qc(theme.S_WARN, 110), 1.0)
            pen.setStyle(Qt.PenStyle.DotLine)
            p.setPen(pen)
            p.drawLine(QPointF(x, AXIS_H + 7), QPointF(x, self.height() - PAD_B))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(warn)
            p.drawPolygon(QPolygonF([QPointF(x - 4.5, AXIS_H + 1), QPointF(x + 4.5, AXIS_H + 1),
                                     QPointF(x, AXIS_H + 7.5)]))
            occupied.append((x - 5, x + 5))
        for e in self._resets:
            x = self._t2x(e.t)
            tw = fms.horizontalAdvance("重置")
            if free(x + 5, x + 8 + tw):
                p.setFont(fs)
                p.setPen(warn)
                p.drawText(QRectF(x + 6, ya - 2, tw + 2, 12),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "重置")
                occupied.append((x + 5, x + 8 + tw))

    @staticmethod
    def _edge_pts(x: float, ya: float, yb: float, inward: int) -> list[QPointF]:
        """鋸齒邊(表示 log 開頭/結尾時仍在 ARM,實際起訖不在 log 內),由上往下。"""
        n = max(2, int(round((yb - ya) / 3.0)))
        return [QPointF(x + (3.0 * inward if k % 2 else 0.0), ya + (yb - ya) * k / n)
                for k in range(n + 1)]

    def _mode_layout(self) -> list[tuple[QRectF, str, str, str, QColor]]:
        """模式列版面 [(色帶矩形, 模式, 色碼, 帶內文字(放不下為 ""), 文字色)](繪圖與測試共用)。"""
        y = self._mode_top()
        ya, yb = y + 2.0, y + MODE_H - 2.0
        fm = QFontMetricsF(self._font(7.5))
        min_w = fm.horizontalAdvance("MO…")
        bands = self._mode_bands()
        out = []
        for k, (xa, xb, name, col) in enumerate(bands):
            ra = int(round(xa))
            rb = int(round(xb)) - (1 if k + 1 < len(bands) else 0)   # 相鄰色帶間留 1 px 縫
            w = max(1, rb - ra)
            room = w - 8
            txt = ""
            if room >= min_w:
                txt = name if fm.horizontalAdvance(name) <= room else                     fm.elidedText(name, Qt.TextElideMode.ElideRight, room)
                if txt == "…":
                    txt = ""
            out.append((QRectF(ra, ya, w, yb - ya), name, col, txt, _ink_on(col)))
        return out

    def _paint_modes(self, p: QPainter):
        """模式列:每個模式一條色帶(切換點到下一次切換),放得下就把名稱寫在帶內(深 / 淺字取對比高者)。"""
        y = self._mode_top()
        p.setFont(self._font(7.5))
        if not self._modes:
            p.setPen(_qc(theme.MUTED, 150))
            p.drawText(QRectF(self._x0() + 6, y, 260, MODE_H),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, _NO_MODE_HINT)
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for r, _name, col, txt, ink in self._mode_layout():
            p.fillRect(r, _qc(col, MODE_ALPHA))
            if txt:
                p.setPen(ink)
                p.drawText(QRectF(r.left() + 4, r.top() - 1, r.width() - 6, r.height() + 2),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, txt)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    def _paint_rows(self, p: QPainter):
        x0, x1 = self._x0(), self._x1()
        ncols = max(1, int(round(x1 - x0)))
        y = self._rows_top()
        for k, row in enumerate(self._all_rows()):
            yt = y + k * ROW_H
            ya, yb = yt + 3.0, yt + ROW_H - 2.0
            if row is None:
                # 提示放在分析區間起點(區間外會被變暗)
                win = self._eff_window()
                hx = min(max(x0, self._t2x(win[0]) if win else x0), x1 - 120) + 6
                p.setFont(self._font(8.0))
                p.setPen(_qc(theme.MUTED, 200))
                p.drawText(QRectF(hx, yt, 200, ROW_H),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, _ERR_HINT)
                continue
            rects, lines = row.shapes(self._v0, self._v1, ncols, x0, ya, yb)
            # 包絡對齊像素欄,不需反鋸齒;平均線才開反鋸齒
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_qc(row.color, 140))
            if rects:
                p.drawRects(rects)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(QColor(row.color), 1.0)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if lines:
                p.drawLines(lines)

    def _window_edge_color(self, win, side: int | None = None) -> QColor:
        # 區間 = 整份 log 時邊界淡一點(仍可抓),免得像在框住整張圖;滑過 / 拖曳中的把手最亮
        if side is not None and (self._drag == side or (self._drag is None and self._hover_handle == side)):
            return QColor(theme.INK)
        whole = win[0] <= self._lo + 1e-6 and win[1] >= self._hi - 1e-6
        return _qc(theme.INK_2, 120 if whole else 255)

    def _paint_window(self, p: QPainter):
        """區間外變暗 + 邊界線(把手畫在動態層,永遠在播放頭之上、抓得到)。"""
        win = self._eff_window()
        if not win:
            return
        x0, x1 = self._x0(), self._x1()
        h = self.height()
        xa, xb = self._t2x(win[0]), self._t2x(win[1])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_DIM)
        top = float(AXIS_H)
        if xa > x0:
            p.drawRect(QRectF(x0, top, min(xa, x1) - x0, h - PAD_B - top))
        if xb < x1:
            p.drawRect(QRectF(max(xb, x0), top, x1 - max(xb, x0), h - PAD_B - top))
        p.setPen(QPen(self._window_edge_color(win), 2.0))
        for x in (xa, xb):
            if x0 - 1 <= x <= x1 + 1:
                p.drawLine(QPointF(x, AXIS_H), QPointF(x, h - PAD_B))

    def _paint_handles(self, p: QPainter):
        win = self._eff_window()
        if not win:
            return
        for side in (0, 1):
            r = self._handle_rect(side, win)
            if r is None:
                continue
            col = self._window_edge_color(win, side)
            x = self._t2x(win[side])
            p.setPen(QPen(col, 2.0))
            p.drawLine(QPointF(x, r.top() + 1), QPointF(x, AXIS_H))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(r, 2, 2)
            p.setPen(QPen(QColor(theme.SURFACE), 1.0))
            cx = r.center().x()
            for gx in (cx - 1.5, cx + 1.5):        # 抓握紋
                p.drawLine(QPointF(gx, r.top() + 2.5), QPointF(gx, r.bottom() - 2.5))

    def _paint_dynamic(self, p: QPainter):
        x0, x1 = self._x0(), self._x1()
        h = self.height()
        # 滑鼠游標線
        if self._hover is not None and self._drag is None:
            hx = self._hover.x()
            if x0 <= hx <= x1 and self._hover.y() >= AXIS_H:
                p.setPen(QPen(_qc(theme.MUTED, 200), 1.0))
                xr = round(hx) + 0.5
                p.drawLine(QPointF(xr, AXIS_H), QPointF(xr, h - PAD_B))
        # 播放頭
        if self._t is not None:
            x = self._t2x(self._t)
            f = self._font(8.0, bold=True)
            fm = QFontMetricsF(f)
            # 放大到 1 秒以內時多一位小數,播放頭標籤才分得出刻度之間的位置
            txt = format_clock(self._t - self._lo, 2 if self._v1 - self._v0 < 1.0 else 1)
            if x < x0 - 0.5:
                txt, x, inside = "‹ " + txt, x0, False
            elif x > x1 + 0.5:
                txt, x, inside = txt + " ›", x1, False
            else:
                inside = True
            if inside:
                p.setPen(QPen(QColor(theme.INK), 1.75))
                p.drawLine(QPointF(x, AXIS_H - 6), QPointF(x, h - PAD_B))
            tw = fm.horizontalAdvance(txt) + 12
            if inside:
                px = min(max(x - tw / 2, 3.0), self.width() - 3.0 - tw)
            else:                              # 播放頭在可見範圍外:貼齊該側,箭頭指向它
                px = x0 if x <= x0 else x1 - tw
            rect = QRectF(px, 2, tw, 14)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.INK) if inside else _qc(theme.INK_2, 220))
            p.drawRoundedRect(rect, 7, 7)
            if inside:
                p.drawPolygon(QPolygonF([QPointF(x - 4, 15.0), QPointF(x + 4, 15.0),
                                         QPointF(x, AXIS_H - 6.0)]))
            p.setFont(f)
            p.setPen(QColor(theme.PLANE))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)
        # 分析區間把手最後畫:和播放頭重疊時(例如選段後)仍看得到、抓得到
        self._paint_handles(p)

    # ---------------- 滑鼠 ----------------

    def _edge_hit(self, x: float):
        win = self._win
        if not win:
            return None
        xa, xb = self._t2x(win[0]), self._t2x(win[1])
        da, db = abs(x - xa), abs(x - xb)
        if min(da, db) > EDGE_GRAB_PX:
            return None
        if da <= EDGE_GRAB_PX and db <= EDGE_GRAB_PX:
            if x < xa:
                return 0
            if x > xb:
                return 1
        return 0 if da <= db else 1

    def _handle_hit(self, pos: QPointF):
        """時間尺下緣的分析區間把手(0 = 起點、1 = 終點、None)。"""
        win = self._win
        if not win or not (AXIS_H - HANDLE_H - 2 <= pos.y() <= AXIS_H + 2):
            return None
        best, bd = None, None
        for side in (0, 1):
            r = self._handle_rect(side, win)
            if r is None or not (r.left() - 2 <= pos.x() <= r.right() + 2):
                continue
            d = abs(pos.x() - r.center().x())
            if bd is None or d < bd:
                best, bd = side, d
        return best

    def _near_playhead(self, x: float) -> bool:
        if self._t is None:
            return False
        xp = self._t2x(self._t)
        return self._x0() - 1 <= xp <= self._x1() + 1 and abs(x - xp) <= EDGE_GRAB_PX

    def _press_target(self, pos: QPointF):
        """按下會抓到什麼:0 / 1 = 分析區間起 / 終點,None = 播放頭(點擊跳轉 / 拖曳)。

        優先序:時間尺下緣的把手 > 播放頭 > 區間邊線。播放頭與區間邊界重疊時(例如選段後
        播放頭就在段首),在其他列按住拖曳的是播放頭;要調整區間請抓把手。
        """
        h = self._handle_hit(pos)
        if h is not None:
            return h
        if self._near_playhead(pos.x()):
            return None
        return self._edge_hit(pos.x())

    def _snap(self, t: float, x: float) -> float:
        cands = [self._lo, self._hi]
        cands += [v for s in self._segments for v in (s.t0, s.t1)]
        cands += [e.t for e in self._events if e.kind in _SNAP_KINDS]
        if self._t is not None:
            cands.append(self._t)
        best, bd = t, SNAP_PX + 1e-9
        for c in cands:
            d = abs(self._t2x(c) - x)
            if d < bd:
                best, bd = c, d
        return best

    @_guarded
    def mousePressEvent(self, ev):
        if not self._has_log or ev.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(ev)
        pos = ev.position()
        if pos.x() < self._x0() - 4:
            return
        edge = self._press_target(pos)
        if edge is not None:
            self._drag = edge
            self._drag_orig = self._win
            self._drag_win = list(self._win)
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            QToolTip.hideText()
            self.update()                       # 把手改成拖曳中的亮色(動態層)
            return
        self._drag = "seek"
        self._seek_to_x(pos.x())

    def _seek_to_x(self, x: float):
        t = min(max(self._x2t(x), self._lo), self._hi)
        if t != self._t:
            self._t = t
            self.update()
        self.seeked.emit(t)

    @_guarded
    def mouseMoveEvent(self, ev):
        pos = ev.position()
        self._hover = pos
        if not self._has_log:
            return
        if self._drag == "seek":
            t = min(max(self._x2t(pos.x()), self._lo), self._hi)
            if t != self._t:
                self._t = t
                self.seeked.emit(t)
            self.update()
            return
        if self._drag in (0, 1):
            self._drag_edge(pos.x(), bool(ev.modifiers() & Qt.KeyboardModifier.AltModifier))
            t0, t1 = self._drag_win
            QToolTip.showText(ev.globalPosition().toPoint() + QPoint(12, 16),
                              f"{'分析起點' if self._drag == 0 else '分析終點'} "
                              f"{format_clock((t0 if self._drag == 0 else t1) - self._lo, 1)}"
                              f"(區間長 {format_clock(t1 - t0, 1)})", self)
            return
        self._hover_handle = self._handle_hit(pos)
        if pos.x() >= self._x0() - 4 and self._press_target(pos) is not None:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.unsetCursor()
        tip = self._tooltip_at(pos)
        if tip:
            QToolTip.showText(ev.globalPosition().toPoint() + QPoint(14, 18), tip, self)
        else:
            QToolTip.hideText()
        self.update()

    def _drag_edge(self, x: float, no_snap: bool):
        t = min(max(self._x2t(x, clamp=False), self._lo), self._hi)
        if not no_snap:
            t = self._snap(t, x)
        t0, t1 = self._drag_win
        mw = min(MIN_WINDOW_S, (self._hi - self._lo) / 2)
        if self._drag == 0:
            t0 = min(max(t, self._lo), t1 - mw)
        else:
            t1 = max(min(t, self._hi), t0 + mw)
        self._drag_win = [t0, t1]
        self._invalidate()

    @_guarded
    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(ev)
        if self._drag in (0, 1):
            new = (float(self._drag_win[0]), float(self._drag_win[1]))
            orig = self._drag_orig
            self._drag = None
            self._drag_win = None
            self._win = new
            self._invalidate()
            QToolTip.hideText()
            if orig is None or abs(new[0] - orig[0]) > 1e-9 or abs(new[1] - orig[1]) > 1e-9:
                self.windowEdited.emit(*new)
        elif self._drag == "seek":
            self._drag = None
            self.update()

    @_guarded
    def mouseDoubleClickEvent(self, ev):
        if not self._has_log or ev.button() != Qt.MouseButton.LeftButton:
            return super().mouseDoubleClickEvent(ev)
        pos = ev.position()
        if pos.y() < AXIS_H and self._handle_hit(pos) is None:
            self._user_view(self._lo, self._hi)
            return
        if AXIS_H <= pos.y() < AXIS_H + EVENT_H:
            i = self._band_at(pos.x())
            if i is not None:
                self.segmentActivated.emit(i)
                return
        self.mousePressEvent(ev)

    def _band_at(self, x: float):
        for i, s in enumerate(self._segments):
            if self._t2x(s.t0) - 2 <= x <= self._t2x(s.t1) + 2:
                return i
        return None

    @_guarded
    def wheelEvent(self, ev):
        if not self._has_log:
            return super().wheelEvent(ev)
        d = ev.angleDelta()
        dx, dy = d.x(), d.y()
        span = self._v1 - self._v0
        if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier or abs(dx) > abs(dy):
            delta = dx if abs(dx) > abs(dy) else dy
            shift = -(delta / 120.0) * 0.15 * span
            self._user_view(self._v0 + shift, self._v1 + shift)
        elif dy:
            full = self._hi - self._lo
            new_span = min(max(span * (0.8 ** (dy / 120.0)), min(self.MIN_VIEW_S, full)), full)
            tc = self._x2t(ev.position().x())
            v0 = tc - (tc - self._v0) * new_span / span
            self._user_view(v0, v0 + new_span)
        ev.accept()

    @_guarded
    def leaveEvent(self, ev):
        self._hover = None
        self._hover_handle = None
        QToolTip.hideText()
        if self._drag is None:
            self.unsetCursor()
        self.update()
        super().leaveEvent(ev)

    @_guarded
    def contextMenuEvent(self, ev):
        if not self._has_log:
            return
        # menu.exec() 會跑巢狀事件迴圈:期間背景載入可能完成並換掉 log,
        # 所以先抓住目前的飛行段清單,選完後確認沒換過才套用(索引才不會錯位)
        segs = self._segments
        x = ev.pos().x()
        t = self._x2t(x)
        menu = QMenu(self)
        try:
            a_all = menu.addAction("全覽")
            a_all.setEnabled(self.is_zoomed())
            a_win = menu.addAction("縮放到分析區間")
            a_win.setEnabled(self._win is not None)
            i = segment_index_at(segs, t) if x >= self._x0() else None
            a_zseg = a_set = None
            if i is not None:
                menu.addSeparator()
                a_zseg = menu.addAction(f"縮放到 {segs[i].label}")
                a_set = menu.addAction("把分析區間設成此 ARM 段")
            chosen = menu.exec(ev.globalPos())
        finally:
            menu.deleteLater()               # 每次右鍵都新建一個選單:用完要釋放
        if chosen is None or self._segments is not segs or not self._has_log:
            return
        if chosen is a_all:
            self._user_view(self._lo, self._hi)
        elif chosen is a_win and self._win:
            self._zoom_to(*self._win)
        elif chosen is a_zseg and i is not None:
            s = segs[i]
            self._zoom_to(s.t0, s.t1)
        elif chosen is a_set and i is not None:
            self.segmentActivated.emit(i)

    def _zoom_to(self, t0: float, t1: float):
        pad = 0.04 * max(t1 - t0, self.MIN_VIEW_S)
        self._user_view(t0 - pad, t1 + pad)

    # ---------------- 提示 ----------------

    def _event_line(self, e) -> str:
        rel = format_clock(e.t - self._lo, 1)
        lab = html.escape(e.label or "")
        if e.kind == EV_RESET:
            return (f"<span style='color:{theme.S_WARN}'>▼ VIO 重置</span>"
                    f"{'(' + lab + ')' if lab else ''} @ {rel}")
        if e.kind == EV_ERROR:
            return f"<span style='color:{theme.S_CRIT}'>▲ 錯誤</span> {lab or 'ERR'} @ {rel}"
        return f"<span style='color:{theme.INK_2}'>◆</span> {lab or 'EV 事件'} @ {rel}"

    def _events_near(self, x: float) -> list:
        return sorted((e for e in (*self._resets, *self._marks)
                       if abs(self._t2x(e.t) - x) <= MARK_PX), key=lambda e: e.t)

    def _tooltip_at(self, pos: QPointF) -> str:
        x, y = pos.x(), pos.y()
        mut = theme.MUTED
        rows = self._all_rows()
        in_events = AXIS_H <= y < self._mode_top()
        in_modes = self._mode_top() <= y < self._rows_top()
        if x < self._x0() - 4:
            k = int((y - self._rows_top()) // ROW_H)
            if 0 <= k < len(rows) and y >= self._rows_top():
                r = rows[k]
                if r is None:
                    return "誤差:計算 ATE 後顯示(只涵蓋分析區間)"
                return (f"<b>{html.escape(r.label)}</b>:列高範圍 {_fmt_val(r.lo, r.unit)} – "
                        f"{_fmt_val(r.hi, r.unit)}<br><span style='color:{mut}'>"
                        f"以全部資料的 1–99 百分位正規化;深色帶 = 每像素的最小~最大,線 = 平均</span>")
            if in_events:
                return ("飛行段:綠色帶 = 一次 ARM→DISARM(鋸齒邊 = log 開頭/結尾時仍在 ARM)<br>"
                        f"<span style='color:{theme.INK_2}'>◆</span> = EV 事件 · "
                        f"<span style='color:{theme.S_CRIT}'>▲</span> = 錯誤(ERR)· "
                        f"<span style='color:{theme.S_WARN}'>▼</span> = VIO 重置")
            if in_modes:
                return ("模式:每段色帶 = 一個飛行模式持續的時間<br>"
                        f"<span style='color:{mut}'>顏色與其他圖表的模式色帶一致</span>"
                        if self._modes else f"模式:{_NO_MODE_HINT}")
            return ""
        edge = self._press_target(pos)
        if edge is not None:
            te = self._win[edge]
            return (f"<b>分析{'起點' if edge == 0 else '終點'}</b> {format_clock(te - self._lo, 1)}<br>"
                    f"<span style='color:{mut}'>拖曳調整;會吸附到飛行段邊界、事件與播放頭"
                    f"(按住 Alt 不吸附)</span>")
        if y < AXIS_H:
            return ("時間尺(相對 log 起點)<br><span style='color:%s'>"
                    "滾輪:縮放 · Shift+滾輪:平移 · 雙擊:回到全覽 · "
                    "下緣把手:拖曳分析區間起訖</span>" % mut)
        t = self._x2t(x)
        tol = (self._v1 - self._v0) / max(self._x1() - self._x0(), 1.0)
        lines = [f"<b>{format_clock(t - self._lo, 1)}</b> "
                 f"<span style='color:{mut}'>(相對 log 起點)</span>"]
        i = segment_index_at(self._segments, t)
        state = (f"<span style='color:{theme.S_GOOD}'>■</span> "
                 f"{html.escape(self._segments[i].label)}(ARM 中)"
                 if i is not None else f"<span style='color:{mut}'>□</span> 地面 / 未 ARM")
        span = self._mode_span_at(t)
        if in_modes and span is not None:
            a, b, name = span
            lines.append(state)
            lines.append(f"<span style='color:{theme.mode_color(name)}'>■</span> 模式 "
                         f"<b>{html.escape(name)}</b> {format_clock(a - self._lo, 1)} → "
                         f"{format_clock(b - self._lo, 1)}(時長 {format_clock(b - a, 1)})")
        else:
            mode = mode_at(self._modes, t)
            lines.append(state + (f" · 模式 {html.escape(mode)}" if mode else ""))
        near = self._events_near(x)
        for e in near[:6]:
            lines.append(self._event_line(e))
        if len(near) > 6:
            lines.append(f"<span style='color:{mut}'>…另 {len(near) - 6} 個事件</span>")
        for r in rows:
            if r is None:
                continue
            v = r.value_at(t, tol)
            lines.append(f"<span style='color:{r.color}'>■</span> {html.escape(r.label)} "
                         f"<b>{_fmt_val(v, r.unit)}</b>")
        if self._win:
            inside = self._win[0] <= t <= self._win[1]
            lines.append(f"<span style='color:{mut}'>{'在' if inside else '不在'}分析區間內</span>")
        if in_events and self._band_at(x) is not None:
            lines.append(f"<span style='color:{mut}'>雙擊:把分析區間設成此段</span>")
        elif self._near_playhead(x) and self._edge_hit(x) is not None:
            lines.append(f"<span style='color:{mut}'>拖曳 = 移動播放頭;"
                         f"調整分析區間請拖時間尺下緣的把手</span>")
        else:
            lines.append(f"<span style='color:{mut}'>點擊 / 拖曳:移動播放頭 · 滾輪:縮放</span>")
        return "<br>".join(lines)


# =====================================================================
#                       下拉清單按鍵保護 / 讀數標籤
# =====================================================================

class _PopupKeyGuard(QObject):
    """下拉清單開著時:沒有 Ctrl / Alt / Meta 的按鍵一律交給清單,不觸發快捷鍵。

    播放快捷鍵是 WidgetWithChildrenShortcut,而 Qt 判斷範圍時會穿過 popup 視窗往上找,
    所以清單開著時 Home / End / PgUp / PgDn / 空白鍵 / ← → 會被播放動作吃掉。
    接受 ShortcutOverride = 「這個鍵我要自己處理」;回傳 False 讓清單照常收到按鍵。
    """

    _MODS = (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
             | Qt.KeyboardModifier.MetaModifier)

    def eventFilter(self, obj, ev):
        try:
            if ev.type() == QEvent.Type.ShortcutOverride and not (ev.modifiers() & self._MODS):
                ev.accept()
        except Exception:  # noqa: BLE001
            _report_exception()
        return False


_popup_guard: _PopupKeyGuard | None = None
_GUARD_PROP = "_popupKeyGuard"


def guard_popup_keys(root: QWidget) -> int:
    """在 root(含)底下每個 QComboBox 的下拉清單裝上按鍵保護;可重複呼叫(已裝過的略過)。

    之後新增的下拉選單要再呼叫一次。回傳這次新裝的數量。
    """
    global _popup_guard
    if _popup_guard is None or sip.isdeleted(_popup_guard):
        _popup_guard = _PopupKeyGuard(QApplication.instance())
    combos = list(root.findChildren(QComboBox))
    if isinstance(root, QComboBox):
        combos.append(root)
    n = 0
    for c in combos:
        v = c.view()
        if v is None or v.property(_GUARD_PROP):
            continue
        v.installEventFilter(_popup_guard)
        v.setProperty(_GUARD_PROP, True)
        n += 1
    return n


class _ElideCombo(QComboBox):
    """被版面壓窄時,目前項目文字以「…」截斷(QComboBox 預設是硬裁切,會切在半個字上)。"""

    @_guarded
    def paintEvent(self, ev):
        p = QStylePainter(self)
        try:
            opt = QStyleOptionComboBox()
            self.initStyleOption(opt)
            p.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
            r = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt,
                                            QStyle.SubControl.SC_ComboBoxEditField, self)
            fm = self.fontMetrics()
            if fm.horizontalAdvance(opt.currentText) > r.width():
                opt.currentText = fm.elidedText(opt.currentText, Qt.TextElideMode.ElideRight,
                                                max(0, r.width() - 2))
            p.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)
        finally:
            p.end()


class _SoftLabel(QLabel):
    """最小寬度 0 的標籤:不會撐大面板的最小寬度(放不下時由 PlaybackPanel 藏起來)。"""

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())


class _ReadoutLabel(QLabel):
    """播放列右側讀數(rich text):寬度不夠時從尾端截斷補「…」,保留粗體 / 顏色。

    整合端把最重要的「3D 誤差 x.xx m」放最前面;最小寬度保證它(連同截斷號)一定完整可見。
    text() 回傳完整內容;實際顯示的(可能截斷)用 shown_text();截斷時滑鼠提示顯示全文。
    sizeHint 用固定的典型內容(不隨文字變):播放中每格更新讀數時版面不會跟著抖動,
    空間不夠時和飛行段選單平分縮減。
    """

    MIN_SAMPLE = "3D 誤差 <b>00.00 m</b> …"
    HINT_SAMPLE = "3D 誤差 <b>0.00 m</b> · ARM 0 · LOITER"     # 典型內容

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""
        self._fit_key = None
        self._font_key = None
        self._hint_w = 0
        self.setTextFormat(Qt.TextFormat.RichText)

    def sizeHint(self) -> QSize:
        return QSize(max(self._hint_w, self.minimumWidth()), super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(self.minimumWidth(), super().minimumSizeHint().height())

    def text(self) -> str:
        return self._full

    def setText(self, text) -> None:
        self._full = str(text or "")
        self._refit()

    def shown_text(self) -> str:
        return QLabel.text(self)

    def is_elided(self) -> bool:
        return self.shown_text() != self._full

    def min_text_width(self) -> int:
        return math.ceil(self._doc(self.MIN_SAMPLE).idealWidth()) + 6

    def _doc(self, rich: str) -> QTextDocument:
        d = QTextDocument()
        d.setDocumentMargin(0)
        d.setDefaultFont(self.font())
        d.setHtml(rich)
        return d

    @_guarded
    def changeEvent(self, ev):
        super().changeEvent(ev)
        if ev.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._refit()

    @_guarded
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._refit()

    def _refit(self) -> None:
        fk = self.font().key()
        if fk != self._font_key:                 # 字型(樣式表)變了 → 重算最小 / 建議寬度
            self._font_key = fk
            m = self.contentsMargins().left() + self.contentsMargins().right()
            mw = self.min_text_width() + m
            self._hint_w = math.ceil(self._doc(self.HINT_SAMPLE).idealWidth()) + 6 + m
            if self.minimumWidth() != mw:
                self.setMinimumWidth(mw)
            self.updateGeometry()
        avail = self.contentsRect().width() - 2
        key = (self._full, avail, fk)
        if key == self._fit_key:
            return
        self._fit_key = key
        doc = self._doc(self._full)
        if not self._full or doc.idealWidth() <= avail:
            QLabel.setText(self, self._full)
            self.setToolTip("")
            return
        plain = doc.toPlainText()
        lo, hi, best = 0, len(plain), None      # 二分搜尋:最多保留幾個字
        while lo < hi:
            mid = (lo + hi + 1) // 2
            d = self._truncated(doc, plain, mid)
            if d.idealWidth() <= avail:
                lo, best = mid, d
            else:
                hi = mid - 1
        QLabel.setText(self, best.toHtml() if best is not None else "…")
        self.setToolTip(plain)

    def _truncated(self, doc: QTextDocument, plain: str, k: int) -> QTextDocument:
        while k > 0 and plain[k - 1] in " ·":     # 不留孤立的分隔符號
            k -= 1
        d = doc.clone()
        d.setDocumentMargin(0)
        d.setDefaultFont(self.font())
        cur = QTextCursor(d)
        cur.setPosition(k)
        cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cur.removeSelectedText()
        cur.insertText("…")
        return d


# =====================================================================
#                              播放面板
# =====================================================================

def _make_icon(kind: str) -> QIcon:
    """自繪播放鍵圖示(不依賴 emoji / 符號字型);1×、2×、3× 解析度各一張。"""
    icon = QIcon()
    size = 16
    for dpr in (1.0, 2.0, 3.0):
        for mode, col in ((QIcon.Mode.Normal, theme.INK), (QIcon.Mode.Disabled, "#6a6963")):
            pm = QPixmap(round(size * dpr), round(size * dpr))
            pm.setDevicePixelRatio(dpr)
            pm.fill(QColor(0, 0, 0, 0))
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(col))
            if kind == "play":
                p.drawPolygon(QPolygonF([QPointF(4.5, 2.5), QPointF(13.5, 8), QPointF(4.5, 13.5)]))
            elif kind == "pause":
                p.drawRoundedRect(QRectF(3.5, 3, 3.2, 10), 0.8, 0.8)
                p.drawRoundedRect(QRectF(9.3, 3, 3.2, 10), 0.8, 0.8)
            elif kind == "prev":
                p.drawRoundedRect(QRectF(3, 3.5, 2.2, 9), 0.6, 0.6)
                p.drawPolygon(QPolygonF([QPointF(13, 3.5), QPointF(13, 12.5), QPointF(5.8, 8)]))
            elif kind == "next":
                p.drawPolygon(QPolygonF([QPointF(3, 3.5), QPointF(3, 12.5), QPointF(10.2, 8)]))
                p.drawRoundedRect(QRectF(10.8, 3.5, 2.2, 9), 0.6, 0.6)
            p.end()
            icon.addPixmap(pm, mode)
    return icon


_SEG_TIP = ("把一次 ARM→DISARM 飛行段設為 ATE 分析區間。\n"
            "一份 log 若涵蓋整次開機,可能有好幾段起降,中間 VIO 也可能重置;\n"
            "地面待機會稀釋統計,建議逐段分析。也可在時間軸上雙擊綠色飛行段")
_JUMP_WHAT = "ARM / DISARM / EV 事件 / 錯誤 / VIO 重置 / 分析區間邊界"


class PlaybackPanel(QWidget):
    """播放控制列 + 時間軸。時間一律為 log 絕對秒。"""

    timeChanged = pyqtSignal(float)            # 播放頭移動(播放、拖曳、步進、跳轉)
    windowRequested = pyqtSignal(float, float) # 使用者要求設定 ATE 分析區間
    viewChanged = pyqtSignal(float, float)     # 使用者改了時間軸可見範圍(set_view 不發,可雙向連動)

    SPEEDS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    TICK_MS = 33                               # 約 30 Hz
    _NEAR = 0.05                               # jump() 忽略離目前時刻這麼近的點
    MIN_VIEW_S = MIN_VIEW_S                    # 時間軸可見範圍最短跨度(秒;連動的圖用同一個值)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Playback")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.clock = PlaybackClock()
        self._has_log = False
        self._lo = self._hi = 0.0
        self._segments: list = []
        self._events: list = []
        self._win: tuple[float, float] | None = None
        self._actions: list[QAction] = []
        self._icons = {k: _make_icon(k) for k in ("play", "pause", "prev", "next")}

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setSingleShot(True)        # 每次 tick 結束才排下一次,不會堆積
        self.timer.setInterval(self.TICK_MS)
        self.timer.timeout.connect(self._tick)
        self._wall = QElapsedTimer()

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 10)
        v.setSpacing(7)
        row = self._row = QHBoxLayout()
        row.setSpacing(6)

        self.btn_prev =self._tool("prev", f"跳到上一個事件:{_JUMP_WHAT}\n快捷鍵:PgUp")
        self.btn_play = self._tool("play", "")
        self.btn_play.setObjectName("PlayButton")
        self.btn_play.setFixedSize(38, 28)
        self.btn_next = self._tool("next", f"跳到下一個事件:{_JUMP_WHAT}\n快捷鍵:PgDown")
        self.btn_prev.clicked.connect(_guarded(lambda *_: self.jump(-1)))
        self.btn_play.clicked.connect(_guarded(lambda *_: self.toggle()))
        self.btn_next.clicked.connect(_guarded(lambda *_: self.jump(+1)))
        for b in (self.btn_prev, self.btn_play, self.btn_next):
            row.addWidget(b)

        self.cmb_speed = QComboBox()
        for x in self.SPEEDS:
            self.cmb_speed.addItem(f"{x:g}×", float(x))
        self.cmb_speed.setCurrentIndex(self.SPEEDS.index(1.0))
        self.cmb_speed.setToolTip("播放速度(相對真實時間)。長時間地面待機可用 20× / 50× 快轉")
        self.cmb_speed.currentIndexChanged.connect(
            _guarded(lambda i, *_: self.set_speed(self.cmb_speed.itemData(i)) if i >= 0 else None))
        row.addWidget(self.cmb_speed)

        self.lbl_time = QLabel("—")
        self.lbl_time.setObjectName("PlayTime")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mono = QFont("Consolas")
        mono.setPointSizeF(10.0)
        fmw = QFontMetricsF(mono)
        self.lbl_time.setMinimumWidth(int(max(fmw.horizontalAdvance("88:88.8 / 88:88.8"),
                                              fmw.horizontalAdvance("8:88:88 / 8:88:88")) + 14))
        self.lbl_time.setToolTip("播放頭 / log 總長(相對 log 起點)\n"
                                 "← / →:前後 1 秒 · Shift+← / →:前後 10 秒 · Home / End:播放範圍起訖")
        row.addWidget(self.lbl_time)

        sep = QFrame()
        sep.setObjectName("PlaySep")
        sep.setFixedSize(1, 18)
        row.addSpacing(2)
        row.addWidget(sep)
        row.addSpacing(2)

        self._lbl_segment = _SoftLabel("飛行段")
        row.addWidget(self._lbl_segment)
        self.cmb_segment = _ElideCombo()
        # 寬度夠時完整顯示「ARM 1 — 120.0–360.0 s(4:00)」;窄視窗可縮到約 6 字(以「…」截斷,下拉清單仍完整)
        self.cmb_segment.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cmb_segment.setMinimumContentsLength(6)
        self.cmb_segment.setToolTip(_SEG_TIP)
        self.cmb_segment.activated.connect(_guarded(self._on_segment_activated))
        row.addWidget(self.cmb_segment)

        self.btn_set_in = QPushButton("[ 設起點")
        self.btn_set_in.setToolTip("把播放頭位置設為分析起點(快捷鍵 [ )\n也可拖曳時間尺下緣的區間把手")
        self.btn_set_in.clicked.connect(_guarded(lambda *_: self._on_set_in()))
        self.btn_set_out = QPushButton("設終點 ]")
        self.btn_set_out.setToolTip("把播放頭位置設為分析終點(快捷鍵 ] )\n也可拖曳時間尺下緣的區間把手")
        self.btn_set_out.clicked.connect(_guarded(lambda *_: self._on_set_out()))
        row.addWidget(self.btn_set_in)
        row.addWidget(self.btn_set_out)

        self.chk_window_only = QCheckBox("只播分析區間")
        self.chk_window_only.setChecked(True)
        self.chk_window_only.setToolTip("勾選:播放只走分析區間,走到終點(或被拖到區間外)自動停\n"
                                        "不勾:播放整份 log")
        self.chk_window_only.toggled.connect(_guarded(lambda *_: self._apply_play_range()))
        row.addWidget(self.chk_window_only)

        self.lbl_readout = _ReadoutLabel()
        self.lbl_readout.setObjectName("PlayReadout")
        self.lbl_readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Expanding、不設 stretch:不足時和飛行段選單平分縮減(設了 stretch 的項目 Qt 會先壓到最小),
        # 有餘裕時多出的寬度全給讀數
        self.lbl_readout.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row.addWidget(self.lbl_readout)
        v.addLayout(row)

        self.timeline = TimelineWidget(self)
        v.addWidget(self.timeline)
        self.timeline.seeked.connect(_guarded(self.seek))
        self.timeline.windowEdited.connect(_guarded(self._on_window_edited))
        self.timeline.segmentActivated.connect(_guarded(self.select_segment))
        self.timeline.viewChanged.connect(self.viewChanged)

        # 按鈕不搶鍵盤焦點,讓空白鍵 / 方向鍵快捷鍵持續有效
        for w in (self.btn_prev, self.btn_play, self.btn_next, self.btn_set_in,
                  self.btn_set_out, self.chk_window_only):
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        guard_popup_keys(self)                 # 速度 / 飛行段清單開著時按鍵交給清單
        self._sync_play_button()
        self.clear()

    @_guarded
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit_controls()

    def _fit_controls(self):
        """控制列放不下全部建議寬度時,先藏「飛行段」標籤(最不重要;選單內容與提示已說明用途),
        剩下的不足再由飛行段選單與讀數平分縮減(讀數最少保留「3D 誤差 00.00 m …」)。

        標籤的最小寬度是 0,不會撐大面板的最小寬度(主視窗 1280 px 時面板只有 ~880 px)。
        """
        lbl = self._lbl_segment
        lay = self.layout()
        if lay is None:
            return
        vis = lbl.isVisibleTo(self)
        need = lay.sizeHint().width() + (0 if vis else lbl.sizeHint().width() + self._row.spacing())
        show = self.width() >= need
        if vis != show:
            lbl.setVisible(show)

    def _tool(self, kind: str, tip: str) -> QToolButton:
        b = QToolButton()
        b.setIcon(self._icons[kind])
        b.setIconSize(QSize(16, 16))
        b.setFixedSize(30, 28)
        b.setToolTip(tip)
        b.setAutoRaise(False)
        return b

    # ---------------- 載入 / 清除 ----------------

    def set_log(self, logdata) -> None:
        """換 log:播放頭回到起點、分析區間 = 整份、清掉舊的誤差列(資料列保留,請重設)。

        範圍是 [t_base, t_last](見 log_time_range):位置資料停了之後才記的訊息 / 錯誤
        也看得到、跳得到;「整份 log」與預設分析區間也是這個範圍。
        """
        self.pause()
        self._lo, self._hi = log_time_range(logdata)
        self._segments = list(logdata.segments)
        self._events = list(logdata.events)
        self._has_log = True
        self.clock.set_bounds(self._lo, self._hi)
        self.clock.set_play_range(self._lo, self._hi)
        self.timeline.set_log(self._lo, self._hi, self._segments, self._events)
        self.timeline.set_error(None, None)       # 舊 log 的 ATE 結果已不適用
        self._rebuild_segment_combo()
        self._set_controls_enabled(True)
        self.set_window(self._lo, self._hi)
        self.seek(self._lo)

    def clear(self) -> None:
        self.pause()
        self._has_log = False
        self._lo = self._hi = 0.0
        self._segments, self._events, self._win = [], [], None
        self.clock.set_bounds(0.0, 0.0)
        self.clock.set_play_range(0.0, 0.0)
        self.clock.seek(0.0)
        self.timeline.clear()
        self.cmb_segment.blockSignals(True)
        self.cmb_segment.clear()
        self.cmb_segment.blockSignals(False)
        self.lbl_time.setText("—:— / —:—")
        self.lbl_readout.setText("")
        self._set_controls_enabled(False)

    def _set_controls_enabled(self, on: bool):
        for w in (self.btn_prev, self.btn_play, self.btn_next, self.cmb_speed, self.cmb_segment,
                  self._lbl_segment, self.btn_set_in, self.btn_set_out, self.chk_window_only):
            w.setEnabled(on)
        for a in self._actions:
            a.setEnabled(on)

    def _rebuild_segment_combo(self):
        c = self.cmb_segment
        c.blockSignals(True)
        c.clear()
        dur = self._hi - self._lo
        c.addItem(f"整份 log({format_clock(dur, 0)})", -1)
        c.setItemData(0, "分析整份 log(含地面待機)", Qt.ItemDataRole.ToolTipRole)
        for i, s in enumerate(self._segments):
            a, b = s.t0 - self._lo, s.t1 - self._lo
            c.addItem(f"{s.label} — {a:.1f}–{b:.1f} s({format_clock(s.duration, 0)})", i)
            notes = []
            if s.open_start:
                notes.append("log 開頭時已是 ARM(沒記到 ARM 事件)")
            if s.open_end:
                notes.append("到 log 結尾都沒有 DISARM")
            tip = f"{s.label}:{format_clock(a, 1)} → {format_clock(b, 1)},時長 {format_clock(s.duration, 1)}"
            c.setItemData(c.count() - 1, "\n".join([tip, *notes]), Qt.ItemDataRole.ToolTipRole)
        c.addItem("自訂區間", -2)
        idx = c.count() - 1
        c.setItemData(idx, "分析區間不等於任何飛行段(拖曳時間軸邊界或用「設起點/設終點」設定)",
                      Qt.ItemDataRole.ToolTipRole)
        item = c.model().item(idx) if hasattr(c.model(), "item") else None
        if item is not None:                    # 只能由程式選取
            item.setEnabled(False)
        c.setCurrentIndex(0)
        # 選單本身可能被版面壓窄(文字截斷),下拉清單仍要完整顯示每一項
        fm = c.view().fontMetrics()
        c.view().setMinimumWidth(max(fm.horizontalAdvance(c.itemText(k)) for k in range(c.count())) + 36)
        c.blockSignals(False)
        c.setToolTip(_SEG_TIP if self._segments else
                     "這份 log 沒有 ARM / DISARM 事件,只能分析整份 log 或自訂區間")
        QTimer.singleShot(0, self._fit_controls_safe)   # 選單建議寬度變了:等版面更新後重排

    @_guarded
    def _fit_controls_safe(self, *_):
        self._fit_controls()

    # ---------------- 資料列轉送 ----------------

    def set_rows(self, rows) -> None:
        self.timeline.set_rows(rows)

    def set_error(self, t, e) -> None:
        self.timeline.set_error(t, e)

    def set_readout(self, text: str) -> None:
        """右側讀數(rich text);寬度不夠時從尾端截斷,所以最重要的(誤差)請放最前面。"""
        self.lbl_readout.setText(text)

    # ---------------- 可見範圍(與其他圖的 x 軸連動) ----------------

    def set_view(self, t0: float, t1: float) -> tuple[float, float]:
        """設定時間軸可見範圍(log 絕對秒);不會發 viewChanged(不回送)。

        回傳實際套用的範圍:限制在 log 範圍內、最短 MIN_VIEW_S 秒;和要求的不同時,
        連動的圖應改用回傳值,兩邊才一致。
        """
        return self.timeline.set_view(t0, t1)

    def view_range(self) -> tuple[float, float]:
        return self.timeline.view_range()

    # ---------------- 分析區間 ----------------

    def set_window(self, t0: float, t1: float) -> None:
        """反映分析區間(不會再發 windowRequested)。"""
        t0, t1 = sorted((float(t0), float(t1)))
        self._win = (t0, t1)
        self.timeline.set_window(t0, t1)
        c = self.cmb_segment
        if self._has_log and c.count():
            i = segment_matching(self._segments, t0, t1, tol=0.05)
            if i is not None:
                idx = i + 1
            elif abs(t0 - self._lo) <= 0.05 and abs(t1 - self._hi) <= 0.05:
                idx = 0
            else:
                idx = c.count() - 1
            c.blockSignals(True)
            c.setCurrentIndex(idx)
            c.blockSignals(False)
        self._apply_play_range()

    def _apply_play_range(self):
        if self.chk_window_only.isChecked() and self._win is not None:
            self.clock.set_play_range(*self._win)
        else:
            self.clock.set_play_range(self._lo, self._hi)

    def _request_window(self, t0: float, t1: float):
        self.set_window(t0, t1)
        self.windowRequested.emit(float(t0), float(t1))

    def _on_window_edited(self, t0: float, t1: float):
        self._request_window(t0, t1)

    def _on_segment_activated(self, idx: int):
        data = self.cmb_segment.itemData(idx)
        if data is None or data == -2:
            if self._win:
                self.set_window(*self._win)     # 還原選取
            return
        self.select_segment(int(data))

    def select_segment(self, index: int) -> None:
        """-1 = 整份 log,否則為飛行段索引:要求設定分析區間並把播放頭移到段首(暫停)。"""
        if not self._has_log:
            return
        if index == -1:
            t0, t1 = self._lo, self._hi
        elif 0 <= index < len(self._segments):
            s = self._segments[index]
            t0, t1 = s.t0, s.t1
        else:
            return
        self.pause()
        self._request_window(t0, t1)
        self.seek(t0)

    def _window_or_full(self) -> tuple[float, float]:
        return self._win if self._win is not None else (self._lo, self._hi)

    def _on_set_in(self):
        if not self._has_log:
            return
        t, (_, t1) = self.clock.t, self._window_or_full()
        if t < t1 - 0.1:
            self._request_window(t, t1)
        else:
            self._hint(self.btn_set_in, f"起點必須早於分析終點({format_clock(t1 - self._lo, 1)})")

    def _on_set_out(self):
        if not self._has_log:
            return
        t, (t0, _) = self.clock.t, self._window_or_full()
        if t > t0 + 0.1:
            self._request_window(t0, t)
        else:
            self._hint(self.btn_set_out, f"終點必須晚於分析起點({format_clock(t0 - self._lo, 1)})")

    def _hint(self, w: QWidget, text: str):
        QToolTip.showText(w.mapToGlobal(QPoint(0, w.height() + 2)), text, w, QRect(), 2500)

    # ---------------- 播放頭 / 播放 ----------------

    def time(self) -> float:
        return self.clock.t

    def seek(self, t: float) -> None:
        """移動播放頭(限制在 log 範圍內);播放中則從新位置繼續播。"""
        if not self._has_log:
            return
        self._apply_time(self.clock.seek(t))

    def _apply_time(self, t: float):
        self.timeline.set_playhead(t)
        if self.timeline.is_zoomed() and not self.timeline.is_scrubbing():
            self.timeline.ensure_visible(t)
        self.lbl_time.setText(f"{format_clock(t - self._lo)} / {format_clock(self._hi - self._lo)}")
        self.timeChanged.emit(t)

    def play(self) -> None:
        if not self._has_log:
            return
        t_before = self.clock.t
        self.clock.play()
        if self.clock.playing:
            if self.clock.t != t_before:          # 在範圍外 / 已到終點 → 從範圍起點開始
                self._apply_time(self.clock.t)
            self._wall.start()
            self.timer.start(self.TICK_MS)
        self._sync_play_button()

    def pause(self) -> None:
        self.clock.pause()
        self.timer.stop()
        self._sync_play_button()

    def toggle(self) -> None:
        self.pause() if self.is_playing() else self.play()

    def is_playing(self) -> bool:
        return bool(self.clock.playing)

    def step(self, dt: float) -> None:
        self.seek(self.clock.t + float(dt))

    def set_speed(self, x: float) -> None:
        x = float(x)
        self.clock.speed = x
        i = next((k for k in range(self.cmb_speed.count())
                  if abs(self.cmb_speed.itemData(k) - x) < 1e-9), -1)
        if i >= 0 and i != self.cmb_speed.currentIndex():
            self.cmb_speed.blockSignals(True)
            self.cmb_speed.setCurrentIndex(i)
            self.cmb_speed.blockSignals(False)

    def jump(self, direction: int) -> None:
        """跳到上(-1) / 下(+1)一個:飛行段起訖、ARM / DISARM / EV 事件 / 錯誤 / VIO 重置、分析區間邊界。"""
        if not self._has_log:
            return
        evs = [e for e in self._events if e.kind in _JUMP_KINDS]
        pts = boundaries(self._segments, evs)
        if self._win is not None:
            pts += list(self._win)
        pts += [self._lo, self._hi]
        t = self.clock.t
        if direction > 0:
            cand = [p for p in pts if p > t + self._NEAR]
            target = min(cand) if cand else None
        else:
            cand = [p for p in pts if p < t - self._NEAR]
            target = max(cand) if cand else None
        if target is not None:
            self.seek(target)

    def _tick(self):
        try:
            self._advance()
        except Exception:  # noqa: BLE001
            _report_exception()
            self.pause()                          # 出錯就停,不要留下「播放中但不動」的狀態

    def _advance(self):
        if not self.clock.playing:
            self.timer.stop()
            self._sync_play_button()
            return
        start = time.perf_counter()
        dt = min(self._wall.nsecsElapsed() * 1e-9, 0.25)
        self._wall.restart()
        if not self.timeline.is_scrubbing():      # 使用者拖曳播放頭時暫不前進
            c = self.clock
            if c.t >= c.play_hi - 1e-9 or c.t < c.play_lo - 1e-9:
                self.pause()                      # 被拖 / 跳到播放範圍外:就地停(不穿越區間外)
                return
            t_old = c.t
            t = c.advance(dt)
            if t != t_old:
                self._apply_time(t)
        if self.clock.playing:
            spent_ms = (time.perf_counter() - start) * 1000.0
            self.timer.start(max(1, int(self.TICK_MS - spent_ms)))
        else:
            self.timer.stop()
            self._sync_play_button()

    def _sync_play_button(self):
        playing = self.is_playing()
        self.btn_play.setIcon(self._icons["pause" if playing else "play"])
        self.btn_play.setToolTip(("暫停" if playing else "播放") + "(空白鍵)\n"
                                 "播放頭在播放範圍外或已到終點時,從範圍起點開始播")

    # ---------------- 快捷鍵 ----------------

    def install_shortcuts(self, scope: QWidget) -> list:
        """建立播放用 QAction(WidgetWithChildrenShortcut)加到 scope,並回傳(可再放進選單)。

        也替 scope 底下現有的下拉選單裝上 guard_popup_keys(清單開著時按鍵交給清單)。
        """
        specs = (
            ("播放/暫停", "Space", self.toggle),
            ("後退 1 秒", "Left", lambda: self.step(-1.0)),
            ("前進 1 秒", "Right", lambda: self.step(1.0)),
            ("後退 10 秒", "Shift+Left", lambda: self.step(-10.0)),
            ("前進 10 秒", "Shift+Right", lambda: self.step(10.0)),
            ("上一個事件", "PgUp", lambda: self.jump(-1)),
            ("下一個事件", "PgDown", lambda: self.jump(+1)),
            ("跳到播放範圍起點", "Home", lambda: self.seek(self.clock.play_lo)),
            ("跳到播放範圍終點", "End", lambda: self.seek(self.clock.play_hi)),
            ("設為分析起點", "[", self._on_set_in),
            ("設為分析終點", "]", self._on_set_out),
        )
        acts = []
        for text, key, fn in specs:
            a = QAction(text, self)
            a.setShortcut(QKeySequence(key))
            a.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            a.triggered.connect(_guarded(lambda *_, f=fn: f()))
            a.setEnabled(self._has_log)
            scope.addAction(a)
            acts.append(a)
        self._actions += acts
        guard_popup_keys(scope)
        return acts
