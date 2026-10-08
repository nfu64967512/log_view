"""任意 log 欄位的時間序列圖(仿 UAV Log Viewer 的繪圖區)。

- 序列 = 欄位運算式(core.expr):「ATT.Roll」、「GPS[1].Spd*3.6」、
  「degrees(atan2(VISP.PY, VISP.PX))」…;最多 4 支 y 軸(1 左、2 右、3 右外、4 左外)。
- 背景依飛行模式上低透明度色帶,上方細色條 + 模式名稱(重疊的名稱略過);
  ARM 飛行段在底緣畫綠色細條 +「ARM n」。
- 橫軸 = 相對 log 起點(logdata.t_base)的 m:ss;所有軸共用。
- 大 log:每條序列只取目前視野、每像素約 2 點的 min/max 抽稀(換視野時延遲重算),
  單次重繪每條序列最多約 8k 點。
- 操作:滾輪以游標為中心縮放時間、Shift+滾輪平移、左鍵拖曳平移、雙擊回到全範圍、
  單擊 → seeked(t);使用者改變視野 → viewChanged(t0, t1)(延遲合併送出)。
- 播放游標沿用 PlotWidget 的 blit 機制:垂直線 + 各序列此刻的點 + 數值框。
  數值框只顯示 log 裡可能出現的值:整數 / 列舉 / 計數欄位取「最後一筆 ≤ t」的樣本
  (線也畫成階梯);航向類(degheading、Yaw、wrap_180(…)…)走最短角度內插再折回原範圍;
  其餘連續欄位線性內插。
- 外部 set_view(時間軸縮放)不同步重算:併入重新抽稀的計時器(連續事件只重畫一次,
  長手勢最多每 _REFRESH_MAX_S 秒更新一次);view_range() 仍立即反映。

時間一律為 log 絕對秒(TimeUS / 1e6);畫布資料座標 = t - t_base。
log 範圍為 [t_base, t_last]:位置資料停了之後的 MSG / ERR 也點得到、游標也走得到。
"""
from __future__ import annotations

import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, Locator, ScalarFormatter
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QToolTip

from core import expr as cexpr
from core.timeline import Track, mode_at, mode_changes

from . import theme
from .plots import PlotWidget

EMPTY_HINT = "從左側「欄位」選欄位或預設組合來繪圖"

# 欄位序列的類別色:深色底上的 6 個色相(OKLCH L 0.65~0.67、對比 ≥ 4.9:1),
# 相鄰兩色 CVD ΔE ≥ 15、一般視覺 ΔE ≥ 19;刻意避開參考藍 / 估測橘 / 誤差洋紅,
# 免得被讀成軌跡角色。序列身分永遠同時有圖例與數值框文字,不單靠顏色。
SERIES_COLORS = ["#1ca886", "#a874e6", "#a89600", "#c565c0", "#4fa83a", "#14a0c4"]
MAX_AXES = 4
AXIS_NAMES = {1: "軸 1(左)", 2: "軸 2(右)", 3: "軸 3(右外)", 4: "軸 4(左外)"}

_REFRESH_MS = 30          # 視野改變 → 重新抽稀
_REFRESH_MAX_S = 0.2      # 連續改視野(長手勢)時,最多延後這麼久就先重畫一次
_VIEW_MS = 140            # 視野改變 → 送出 viewChanged(合併連續滾輪 / 拖曳)
_MAX_BINS = 4000          # 抽稀格數上限(每格 min/max 兩點)
MIN_SPAN_S = 0.05         # 最小可縮放到的時間跨度(秒);整合端可據此對齊時間軸的最小視野
_DRAG_PX = 4              # 移動超過這麼多像素才算拖曳(否則是單擊 = 跳轉)
_MARKER_MAX = 160         # 視野內原始樣本少於此數時加畫點標記(稀疏訊息才看得見)
_KEY_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)(?:\[(\d+)\])?\.([A-Za-z_][A-Za-z0-9_]*)\s*$")
_CALL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(")
_CLOCK_STEPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900,
                1800, 3600, 7200)
# 單位不是 degheading、但名稱是航向的欄位(VISP.Yaw、DCM.Yaw… 單位只標 deg,範圍常為 ±180)
_HEADING_FIELDS = frozenset({"Yaw", "DesYaw", "GCrs", "Hdg", "Heading", "Crs", "YC"})

# 序列種類(決定游標數值怎麼取)
K_LINEAR = "linear"       # 連續量:線性內插
K_STEP = "step"           # 整數 / 列舉 / 計數:最後一筆 ≤ t 的樣本(線畫成階梯)
K_HEADING = "heading"     # 角度(會折返):最短角度內插後折回原範圍


# ---------------- 抽稀 ----------------

def gap_indices(t: np.ndarray, gap: float) -> np.ndarray:
    """資料空洞的位置:t[g] - t[g-1] > gap 的 g(遞增)。"""
    if t.size < 2:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.diff(t) > gap) + 1


def decimate(t: np.ndarray, v: np.ndarray, lo: float, hi: float, nbins: int,
             gaps: np.ndarray | None = None, hold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """取 [lo, hi](兩側各多一點)內的樣本;多於 2×nbins 時每格只留最小 / 最大值。

    保留真實樣本(時間不偏移),依原順序輸出;跨過資料空洞(gaps,見 gap_indices)
    或整格 NaN 處插入 NaN 斷線,不在沒資料的地方連出假直線。斷線點預設在空洞中點;
    hold 給定時放在空洞前一筆之後 hold 秒(不超過中點)—— 階梯線只延伸到游標還有值的地方。
    """
    n = t.size
    if n == 0 or hi < lo:
        return np.empty(0), np.empty(0)
    a = max(int(np.searchsorted(t, lo, side="left")) - 1, 0)
    b = min(int(np.searchsorted(t, hi, side="right")) + 1, n)
    if b <= a:
        return np.empty(0), np.empty(0)
    m = b - a
    nb = max(1, int(nbins))
    if m <= 2 * nb:
        idx = np.arange(a, b)
    else:
        k = -(-m // nb)                       # 每格樣本數(向上取整)
        full = m // k
        cut = a + full * k
        seg = v[a:cut].reshape(full, k)
        nan = np.isnan(seg)
        has_nan = bool(nan.any())
        s_lo = np.where(nan, np.inf, seg) if has_nan else seg
        s_hi = np.where(nan, -np.inf, seg) if has_nan else seg
        base = a + np.arange(full, dtype=np.int64) * k
        parts = [base + np.argmin(s_lo, axis=1), base + np.argmax(s_hi, axis=1),
                 np.array([a, b - 1])]
        if has_nan:                           # 含 NaN 的格子留一個 NaN 點 → 斷線
            rows = np.flatnonzero(nan.any(axis=1))
            parts.append(base[rows] + np.argmax(nan[rows], axis=1))
        if cut < b:
            r = v[cut:b]
            if np.isnan(r).all():
                parts.append(np.array([cut]))
            else:
                parts.append(np.array([cut + int(np.nanargmin(r)), cut + int(np.nanargmax(r))]))
        idx = np.unique(np.concatenate(parts))
    ts, vs = t[idx], v[idx]
    if gaps is not None and gaps.size and idx.size > 1:
        c = np.searchsorted(gaps, idx, side="right")
        brk = np.flatnonzero(np.diff(c) > 0) + 1
        if brk.size:
            mid = 0.5 * (ts[brk - 1] + ts[brk])
            at = mid if hold is None else np.minimum(ts[brk - 1] + max(float(hold), 0.0), mid)
            ts = np.insert(ts, brk, at)
            vs = np.insert(vs, brk, np.nan)
    return ts, vs


def _series_gap(t: np.ndarray) -> float:
    """斷線門檻:中位取樣間隔的 10 倍,至少 2 秒(低頻 / 事件型訊息不會被切碎)。"""
    if t.size < 2:
        return 2.0
    d = np.diff(t)
    d = d[d > 0]
    return max(2.0, 10.0 * float(np.median(d))) if d.size else 2.0


def fmt_value(v: float | None) -> str:
    """數值框 / 提示用的精簡數字(約 4~5 位有效數字,不用科學記號直到很大)。"""
    if v is None or not np.isfinite(v):
        return "—"
    a = abs(v)
    if a < 1e-9:                      # 浮點雜訊(例如 -2e-26)當 0
        return "0"
    if a >= 1e7 or a < 1e-4:
        return f"{v:.3e}"
    if a >= 1000:
        return f"{v:.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _fit_text(text: str, units: float) -> str:
    """截短到約 units 個半形字寬(全形字算 2),超過時結尾加「…」。"""
    w = 0.0
    for i, ch in enumerate(text):
        w += 2.0 if ord(ch) >= 0x2E80 else 1.0
        if w > units:
            return text[:max(i - 1, 1)] + "…"
    return text


# ---------------- 時間刻度 ----------------

def fmt_clock(sec: float, decimals: int = 1) -> str:
    """秒 → 「m:ss.s」,一小時以上「h:mm:ss.s」(小數位數照 decimals,進位正確)。

    core.timeline.format_clock 超過一小時就不帶小數,放大到秒以下時刻度會全部一樣。
    """
    if sec is None or not np.isfinite(sec):
        return "—"
    d = max(int(decimals), 0)
    q = 10 ** d
    n = int(round(abs(float(sec)) * q))         # 先四捨五入到顯示精度,再拆時分秒
    sign = "-" if sec < 0 and n else ""
    whole, frac = divmod(n, q)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    tail = f".{frac:0{d}d}" if d else ""
    if h:
        return f"{sign}{h}:{m:02d}:{s:02d}{tail}"
    return f"{sign}{m}:{s:02d}{tail}"


def _step_decimals(step: float) -> int:
    return 0 if step >= 1 - 1e-9 else 1 if step >= 0.1 - 1e-9 else 2


class _ClockLocator(Locator):
    """以秒為單位的「好看」刻度間距(0.01 s … 2 h),密度依軸寬自動調整。"""

    def __init__(self, px_per_tick: float = 84.0):
        self.px = px_per_tick
        self.step = 1.0

    def __call__(self):
        lo, hi = self.axis.get_view_interval()
        return self.tick_values(lo, hi)

    def tick_values(self, vmin, vmax):
        lo, hi = sorted((float(vmin), float(vmax)))
        span = max(hi - lo, 1e-9)
        try:
            ax = self.axis.axes
            width = ax.bbox.width / max(getattr(ax.figure.canvas, "device_pixel_ratio", 1.0), 1.0)
        except Exception:  # noqa: BLE001
            width = 800.0
        nmax = max(2, int(width / self.px))
        step = next((s for s in _CLOCK_STEPS if span / s <= nmax), None)
        if step is None:
            step = _CLOCK_STEPS[-1] * math.ceil(span / (nmax * _CLOCK_STEPS[-1]))
        self.step = step
        start = math.ceil(lo / step - 1e-9) * step
        return np.arange(start, hi + step * 1e-6, step)


def _clock_formatter(locator: _ClockLocator) -> FuncFormatter:
    def fmt(x, _pos=None):
        return fmt_clock(round(float(x), 6), _step_decimals(locator.step))
    return FuncFormatter(fmt)


def _top_call(expr: str) -> tuple[str, str] | None:
    """整個運算式就是一個函式呼叫 f(…) 時回傳 (f, 括號內文字),否則 None。"""
    m = _CALL_RE.match(expr or "")
    if not m:
        return None
    s = expr.rstrip()
    depth = 0
    for i in range(m.end() - 1, len(s)):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return (m.group(1), s[m.end():i]) if i == len(s) - 1 else None
    return None


def series_kind(expr: str, unit: str, v: np.ndarray) -> tuple[str, float]:
    """序列種類 (K_LINEAR / K_STEP / K_HEADING, 角度折返範圍 180 或 360)。

    航向:單位 degheading、名稱是航向的 deg 欄位,或整條運算式是 wrap_180 / wrap_360 /
    degrees(atan2(…)),且數值落在角度範圍內。整數:所有有限值都是整數(列舉、計數、狀態、
    PWM、比較式的 0/1…)。折返範圍:有負值 → ±180,否則 0~360。
    """
    v = np.asarray(v, dtype=np.float64)
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return K_LINEAR, 360.0
    lo, hi = float(fin.min()), float(fin.max())
    wrap = 180.0 if lo < 0 else 360.0
    heading = False
    m = _KEY_RE.match(expr or "")
    if m:
        heading = unit == "degheading" or (m.group(3) in _HEADING_FIELDS and unit in ("", "deg"))
    else:
        call = _top_call(expr)
        if call is not None:
            name, inner = call
            if name in ("wrap_180", "wrap_360"):
                heading, wrap = True, 180.0 if name == "wrap_180" else 360.0
            elif name == "degrees":
                c2 = _top_call(inner)
                heading = c2 is not None and c2[0] == "atan2"
                wrap = 180.0
    if heading and lo >= -180.5 and hi <= 360.5:
        return K_HEADING, wrap
    if np.array_equal(fin, np.round(fin)):
        return K_STEP, wrap
    return K_LINEAR, wrap


def _wrap_deg(x: float, wrap: float) -> float:
    """折回 [0, 360) 或 (-180, 180](與 ArduPilot / core.expr 的 wrap_360 / wrap_180 相同)。"""
    r = x % 360.0
    return r - 360.0 if wrap == 180.0 and r > 180.0 else r


# ---------------- 序列 ----------------

@dataclass
class _Series:
    expr: str
    axis: int
    color: str
    label: str
    t: np.ndarray                    # log 絕對秒(遞增)
    v: np.ndarray
    unit: str = ""
    visible: bool = True
    extra: bool = False              # set_extra() 注入的衍生序列(例如 ATE 誤差)
    gap: float = 2.0
    gaps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    track: Track | None = None
    kind: str = K_LINEAR             # 見 series_kind
    wrap: float = 360.0              # K_HEADING 的折返範圍(180 = ±180)
    line: Line2D | None = None
    dot: Line2D | None = None

    def value_at(self, t: float) -> float | None:
        """游標此刻的值;超出範圍或落在資料空洞內為 None。

        連續量線性內插;整數 / 列舉取最後一筆 ≤ t 的樣本;航向走最短角度內插再折回。
        空洞邊緣(Track 判定還貼近某一端)取較近那一筆 —— 永遠是 log 裡可能出現的值。
        """
        tr = self.track
        if tr is None:
            return None
        p = tr.at(t)
        if p is None:
            return None
        if self.kind == K_LINEAR:
            val = float(p[0])
        else:
            ts, vs = self.t, self.v
            j = int(np.searchsorted(ts, t, side="right"))
            if j <= 0:
                val = float(vs[0])
            elif j >= ts.size:
                val = float(vs[-1])
            else:
                ta, tb = float(ts[j - 1]), float(ts[j])
                a, b = float(vs[j - 1]), float(vs[j])
                if tb - ta > tr.max_gap:
                    val = a if t - ta <= tb - t else b
                elif self.kind == K_STEP or tb <= ta or t <= ta:
                    val = a
                else:
                    w = (t - ta) / (tb - ta)
                    d = (b - a + 180.0) % 360.0 - 180.0          # 最短角度差
                    val = _wrap_deg(a + w * d, self.wrap)
        return val if np.isfinite(val) else None

    def fmt(self, v: float | None) -> str:
        """數值框文字:整數序列不帶小數;航向四捨五入後不出現 360.0 / -180.0。"""
        if v is None or not np.isfinite(v):
            return fmt_value(None)
        if self.kind == K_STEP and abs(v) < 1e15:
            return str(int(v))
        if self.kind == K_HEADING:
            r = round(v, 1)
            if self.wrap == 360.0 and r >= 360.0:
                v = r - 360.0
            elif self.wrap == 180.0 and r <= -180.0:
                v = r + 360.0
        return fmt_value(v)


def _prep(t, v) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if t.shape != v.shape:
        raise ValueError("時間與數值長度不一致")
    ok = np.isfinite(t)
    if not ok.all():
        t, v = t[ok], v[ok]
    if t.size > 1 and np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
    return t, v


# ---------------- 圖表 ----------------

class FieldPlot(PlotWidget):
    """任意欄位 / 運算式的時間序列圖。見模組說明。"""

    seeked = pyqtSignal(float)              # 單擊圖面:跳到該時刻(log 絕對秒)
    viewChanged = pyqtSignal(float, float)  # 使用者改變時間視野(log 絕對秒;已合併)
    seriesChanged = pyqtSignal()            # 序列增刪 / 屬性改變(含 set_extra)
    error = pyqtSignal(str)                 # 繁中錯誤訊息(運算式失敗等)

    min_span = MIN_SPAN_S                   # 最小時間跨度(秒;滾輪縮放的下限)

    def __init__(self, parent=None, toolbar: bool = True):
        self._logdata = None
        self._index = None
        self._series: list[_Series] = []
        self._extra: _Series | None = None
        self._unplottable: set[str] = set()   # 這份 log 畫不出來的運算式(欄位全 NaN 等)
        self._modes = []
        self._segs = []
        self._log_rng = (0.0, 0.0)          # logdata 的 (t_base, t_last)
        self._view: tuple[float, float] | None = None   # 相對秒;None = 全範圍
        self._refresh_since = 0.0           # 這一輪延後的重算從何時開始等(monotonic 秒)
        self._y_auto = True
        self._quiet = 0                     # >0:程式自己改視野,不算使用者操作
        self._no_echo = 0                   # >0:set_view 中,不送 viewChanged
        self._press = None                  # 左鍵按下:(x_px, y_px, xlim)
        self._dragging = False
        self._clear_refs()
        super().__init__(parent, toolbar)
        self.setObjectName("FieldPlot")
        self.setMinimumHeight(140)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(_REFRESH_MS)
        self._refresh_timer.timeout.connect(self._refresh)
        self._view_timer = QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(_VIEW_MS)
        self._view_timer.timeout.connect(self._emit_view)

        for name, fn in (("button_press_event", self._on_press),
                         ("button_release_event", self._on_release),
                         ("motion_notify_event", self._on_motion),
                         ("scroll_event", self._on_scroll),
                         ("resize_event", self._on_resize),
                         ("figure_leave_event", lambda _e: QToolTip.hideText())):
            self.canvas.mpl_connect(name, fn)
        self._patch_toolbar()

    def _clear_refs(self):
        self._main = None                   # 軸 1(色帶、橫軸、游標線都在這支)
        self._axes: dict[int, object] = {}
        self._rib = None                    # 模式色條軸(在主圖上方)
        self._legend = None
        self._legend_ncols = None
        self._mode_texts: list = []
        self._arm_texts: list = []
        self._margins = None                # (left, right, top, bottom) 像素
        self._cursor_line = None
        self._box = None
        self._box_head = None
        self._box_rows: list = []
        self._locator = None
        self._legend_items = ([], [])
        self._bg_xlim = None                # blit 底圖畫的是哪個視野(與目前不同 = 底圖過期)

    def _patch_toolbar(self):
        """工具列:「首頁」改成回到全範圍;拿掉會破壞版面的「子圖設定 / 圖表選項」。"""
        tb = self.toolbar
        if tb is None:
            return
        acts = getattr(tb, "_actions", {}) or {}
        home = acts.get("home")
        if home is not None:
            try:
                home.triggered.disconnect()
            except TypeError:
                pass
            home.triggered.connect(lambda *_: self.reset_view())
            home.setToolTip("回到全範圍(也可雙擊圖面)")
        for key in ("configure_subplots", "edit_parameters"):
            a = acts.get(key)
            if a is not None:
                a.setVisible(False)
        for a in tb.actions():
            if a.text() in ("Subplots", "Customize"):
                a.setVisible(False)

    # ---------------- 公開介面 ----------------

    def set_log(self, logdata) -> None:
        """換 log:已繪製的序列以新 log 重算(新 log 沒有的欄位移除);清掉衍生序列。"""
        self._logdata = logdata
        self._index = getattr(logdata, "index", None) if logdata is not None else None
        self.t_base = float(logdata.t_base) if logdata is not None else 0.0
        self._log_rng = (self.t_base, self._t_last(logdata)) if logdata is not None else (0.0, 0.0)
        self._unplottable = set()
        self._modes = mode_changes(logdata.events) if logdata is not None else []
        try:
            self._segs = list(logdata.segments) if logdata is not None else []
        except Exception:  # noqa: BLE001
            self._segs = []
        self._extra = None
        self._view = None
        self._y_auto = True
        old = self._series
        self._series = []
        lost = []
        if self._index is not None:
            for s in old:
                ns = self._make(s.expr, s.axis, s.color, quiet=True)
                if ns is None:
                    lost.append(s.expr)
                else:
                    ns.visible = s.visible
                    self._series.append(ns)
        elif old:
            lost = [s.expr for s in old]
        if lost and logdata is not None:
            self.error.emit("新 log 沒有這些欄位,已從圖中移除:" + "、".join(lost))
        self._rebuild()
        self.seriesChanged.emit()

    @staticmethod
    def _t_last(logdata) -> float:
        """log 範圍終點:t_last(含位置資料停了之後的訊息 / 事件);舊物件退回 t_end。"""
        hi = float(logdata.t_end)
        try:
            tl = getattr(logdata, "t_last", None)
            if tl is not None and np.isfinite(float(tl)):
                hi = max(hi, float(tl))
        except Exception:  # noqa: BLE001 — 索引壞掉就只用 t_end
            pass
        return hi

    def is_unplottable(self, expr: str) -> bool:
        """這個運算式在目前的 log 試畫過且畫不出來(欄位全是 NaN、運算錯誤…)。"""
        return (expr or "").strip() in self._unplottable

    def add_series(self, expr: str, axis: int = 1, color: str | None = None) -> bool:
        """加一條序列;已在圖上則不重複加(回傳 True)。失敗時送出 error 並回傳 False。"""
        expr = (expr or "").strip()
        if any(s.expr == expr for s in self._series):
            return True
        s = self._make(expr, axis, color)
        if s is None:
            return False
        self._series.append(s)
        self._rebuild()
        self.seriesChanged.emit()
        return True

    def add_preset(self, preset) -> int:
        """加入預設組合的所有序列(已在圖上的略過);回傳新加的數量。"""
        added = 0
        failed = []
        have = {s.expr for s in self._series}
        for spec in getattr(preset, "series", []):
            e = (spec.expr or "").strip()
            if e in have:
                continue
            s = self._make(e, getattr(spec, "axis", 1), getattr(spec, "color", None), quiet=True)
            if s is None:
                failed.append(e)
                continue
            self._series.append(s)
            have.add(e)
            added += 1
        if failed:
            self.error.emit(f"預設組合「{getattr(preset, 'name', '')}」有欄位無法繪製:"
                            + "、".join(failed))
        if added:
            self._rebuild()
            self.seriesChanged.emit()
        return added

    def remove_series(self, i: int) -> None:
        if 0 <= i < len(self._series):
            del self._series[i]
            self._rebuild()
            self.seriesChanged.emit()

    def clear_series(self) -> None:
        if not self._series:
            return
        self._series = []
        self._rebuild()
        self.seriesChanged.emit()

    def series(self) -> list[dict]:
        return [dict(expr=s.expr, axis=s.axis, color=s.color, visible=s.visible, label=s.label)
                for s in self._series]

    def set_series_visible(self, i: int, on: bool) -> None:
        if 0 <= i < len(self._series) and self._series[i].visible != bool(on):
            self._series[i].visible = bool(on)
            self._rebuild()
            self.seriesChanged.emit()

    def set_series_axis(self, i: int, axis: int) -> None:
        axis = min(max(int(axis), 1), MAX_AXES)
        if 0 <= i < len(self._series) and self._series[i].axis != axis:
            self._series[i].axis = axis
            self._y_auto = True
            self._rebuild()
            self.seriesChanged.emit()

    def set_series_color(self, i: int, color: str) -> None:
        if 0 <= i < len(self._series) and color and self._series[i].color != color:
            self._series[i].color = str(color)
            self._rebuild()
            self.seriesChanged.emit()

    def set_extra(self, name: str | None, t=None, v=None, color: str | None = None) -> None:
        """注入 / 更新 / 清除一條衍生序列(例如「ATE 3D 誤差」);放在自己的 y 軸。

        name 為 None 時清除。會送出 seriesChanged(整合端可據此決定是否展開圖區)。
        """
        if name is None or t is None or v is None:
            if self._extra is None:
                return
            self._extra = None
        else:
            try:
                tt, vv = _prep(t, v)
            except ValueError as e:
                self.error.emit(f"無法加入「{name}」:{e}")
                return
            if tt.size == 0:
                self._extra = None
            else:
                gap = _series_gap(tt)
                self._extra = _Series(expr=str(name), axis=1, color=color or theme.C_ERR,
                                      label=str(name), t=tt, v=vv, extra=True, gap=gap, gaps=gap_indices(tt, gap),
                                      track=Track(tt, vv[:, None], gap))
        self._rebuild()
        self.seriesChanged.emit()

    def has_content(self) -> bool:
        """圖上是否有東西(使用者序列或衍生序列)。"""
        return bool(self._series) or self._extra is not None

    def set_view(self, t0: float, t1: float) -> None:
        """套用時間視野(log 絕對秒);不會回送 viewChanged。

        只改 xlim(view_range() 立即反映);重新抽稀 + 重畫併入計時器,
        時間軸連續滾輪 / 觸控板縮放時不會每個事件都整張重畫。
        """
        if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
            return
        self._view_timer.stop()               # 尚未送出的使用者視野已被取代
        self._view = (t0 - self.t_base, t1 - self.t_base)
        if self._main is None:
            return
        with self._no_view_echo():
            self._main.set_xlim(*self._view)  # → _on_xlim_changed 排程重算
        self._schedule_refresh()

    def view_range(self) -> tuple[float, float] | None:
        """目前視野(log 絕對秒);沒有圖時 None。"""
        if self._main is None:
            return None
        x0, x1 = self._main.get_xlim()
        return x0 + self.t_base, x1 + self.t_base

    def full_range(self) -> tuple[float, float]:
        """全範圍(log 絕對秒):log 範圍與所有序列範圍的聯集。"""
        lo, hi = self._log_rng
        has_log = hi > lo
        for s in self._all_series():
            if s.t.size:
                lo = min(lo, float(s.t[0])) if has_log else float(s.t[0])
                hi = max(hi, float(s.t[-1])) if has_log else float(s.t[-1])
                has_log = True
        if hi - lo < 1e-3:
            hi = lo + 1.0
        return lo, hi

    def reset_view(self) -> None:
        """回到全範圍、y 軸自動縮放(雙擊 / 工具列首頁);視為使用者操作。"""
        self._view = None
        self._y_auto = True
        if self._main is None:
            return
        lo, hi = self.full_range()
        self._main.set_xlim(lo - self.t_base, hi - self.t_base)
        self._view = None                     # 全範圍:之後加了更長的序列也跟著涵蓋
        self._refresh()

    def render(self, res, logdata):
        """FieldPlot 不畫 ATE 結果(用 set_log / add_series / set_extra);保留介面避免誤呼叫時出錯。"""

    def show_placeholder(self, text: str | None = None):
        super().show_placeholder(EMPTY_HINT if text is None else text)

    # ---------------- 建立序列 ----------------

    def _make(self, expr: str, axis: int, color: str | None, quiet: bool = False) -> _Series | None:
        if not expr:
            if not quiet:
                self.error.emit("運算式是空的")
            return None
        if self._index is None:
            if not quiet:
                self.error.emit("這份資料沒有 log 欄位索引(模擬資料或舊格式),無法繪製欄位")
            return None
        try:
            t, v = cexpr.evaluate(expr, self._index)
            t, v = _prep(t, v)
        except Exception as e:  # noqa: BLE001 — ExprError 或 index 內部錯誤都要回報
            self._unplottable.add(expr)
            if not quiet:
                self.error.emit(f"無法繪製「{expr}」:{e}")
            return None
        if t.size == 0 or not np.isfinite(v).any():
            self._unplottable.add(expr)
            if not quiet:
                self.error.emit(f"「{expr}」沒有可繪製的資料")
            return None
        self._unplottable.discard(expr)
        gap = _series_gap(t)
        unit = self._unit_of(expr)
        kind, wrap = series_kind(expr, unit, v)
        return _Series(expr=expr, axis=min(max(int(axis or 1), 1), MAX_AXES),
                       color=color or self._next_color(), label=expr, t=t, v=v,
                       unit=unit, gap=gap, gaps=gap_indices(t, gap),
                       track=Track(t, v[:, None], gap), kind=kind, wrap=wrap)

    def _next_color(self) -> str:
        used = [s.color for s in self._series]
        return min(SERIES_COLORS, key=lambda c: (used.count(c), SERIES_COLORS.index(c)))

    def _unit_of(self, expr: str) -> str:
        m = _KEY_RE.match(expr)
        if not m or self._index is None:
            return ""
        try:
            info = self._index.messages().get(m.group(1))
            return str((getattr(info, "units", None) or {}).get(m.group(3), "") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _all_series(self) -> list[_Series]:
        return self._series + ([self._extra] if self._extra is not None else [])

    def _shown(self) -> list[_Series]:
        return [s for s in self._all_series() if s.visible]

    def _extra_axis(self) -> int:
        used = {s.axis for s in self._series if s.visible}
        return next((k for k in range(1, MAX_AXES + 1) if k not in used), MAX_AXES)

    # ---------------- 繪製 ----------------

    @contextmanager
    def _silent(self):
        self._quiet += 1
        try:
            yield
        finally:
            self._quiet -= 1

    @contextmanager
    def _no_view_echo(self):
        self._no_echo += 1
        try:
            yield
        finally:
            self._no_echo -= 1

    def _clear(self):
        super()._clear()
        self._clear_refs()
        for s in self._all_series():
            s.line = s.dot = None

    def _rebuild(self):
        # 保留目前視野(重建前若已有圖);y 軸若被使用者手動縮放過,重建後恢復自動
        if self._main is not None and self._view is not None:
            x0, x1 = self._main.get_xlim()
            self._view = (x0, x1)
        if self._extra is not None:
            self._extra.axis = self._extra_axis()
        if not self._shown():
            self._clear()
            self.show_placeholder(EMPTY_HINT if not self._series else
                                  "所有序列都已隱藏 — 在左側「已繪製」勾選要顯示的序列")
            return
        self._redraw(self._draw)

    def _draw(self):
        fig = self.fig
        shown = self._shown()
        used = sorted({s.axis for s in shown})
        with self._silent():
            main = fig.add_axes([0.08, 0.1, 0.84, 0.8])
            main.set_facecolor(theme.SURFACE)
            self._main = main
            self._axes = {1: main}
            for k in used:
                if k == 1:
                    continue
                ax = main.twinx()
                ax.set_facecolor("none")
                ax.patch.set_visible(False)
                ax.grid(False)
                if k == 4:
                    ax.yaxis.set_label_position("left")
                    ax.yaxis.set_ticks_position("left")
                    ax.yaxis.set_offset_position("left")
                for side in ("top", "bottom", "left", "right"):
                    ax.spines[side].set_visible(side == ("left" if k == 4 else "right"))
                self._axes[k] = ax
            self._style_axes(shown)
            self._draw_bands(main)
            for s in shown:
                ax = self._axes[s.axis]
                # 整數 / 列舉畫成階梯:和游標「最後一筆 ≤ t」的值一致,不畫出不存在的中間值
                s.line = ax.plot([], [], color=s.color, lw=1.4, zorder=3,
                                 solid_joinstyle="round", markersize=3.2,
                                 drawstyle="steps-post" if s.kind == K_STEP else "default")[0]
            self._build_legend(shown)
            self._build_cursor(shown)

            lo, hi = self.full_range()
            x0, x1 = self._view if self._view is not None else (lo - self.t_base, hi - self.t_base)
            main.set_xlim(x0, x1)
            for ax in self._axes.values():
                ax.callbacks.connect("ylim_changed", self._on_ylim_changed)
            main.callbacks.connect("xlim_changed", self._on_xlim_changed)
            self._y_auto = True
            self._update_data()
            self._layout(force=True)
            self._update_labels()
        if self.toolbar is not None:
            self.toolbar.update()            # 清空導覽堆疊(上一張圖的視野不適用)

    def _style_axes(self, shown):
        main = self._main
        loc = _ClockLocator()
        self._locator = loc
        main.xaxis.set_major_locator(loc)
        main.xaxis.set_major_formatter(_clock_formatter(loc))
        main.tick_params(axis="x", colors=theme.MUTED, labelsize=8.5)
        for side in ("top", "right"):
            main.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            main.spines[side].set_color(theme.BASELINE)
        main.grid(True, axis="x", color=theme.GRID, linewidth=0.8)
        if 1 in {s.axis for s in shown}:
            main.grid(True, axis="y", color=theme.GRID, linewidth=0.8)
        else:
            main.grid(False, axis="y")
        for k, ax in self._axes.items():
            on_k = [s for s in shown if s.axis == k]
            side = "left" if k in (1, 4) else "right"
            fmt = ScalarFormatter(useOffset=False)
            fmt.set_scientific(False)        # 不要軸頂的偏移 / 指數文字(會壓到模式色條)
            ax.yaxis.set_major_formatter(fmt)
            if not on_k:                     # 軸 1 沒有序列:收起左側刻度
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.spines["left"].set_visible(False)
                ax.set_ylabel("")
                continue
            col = on_k[0].color if len(on_k) == 1 else theme.MUTED
            lab_col = on_k[0].color if len(on_k) == 1 else theme.INK_2
            ax.tick_params(axis="y", colors=col, labelsize=8, length=3)
            ax.spines[side].set_color(on_k[0].color if len(on_k) == 1 else theme.BASELINE)
            ax.spines[side].set_linewidth(1.0)
            units = {s.unit for s in on_k}
            if len(on_k) > 1 and len(units) == 1 and on_k[0].unit:   # 同單位只寫一次
                name = " / ".join(s.label for s in on_k) + f"({on_k[0].unit})"
            else:
                name = " / ".join(self._label_of(s) for s in on_k)
            ax.set_ylabel(name, color=lab_col, fontsize=8.5, labelpad=4)
            ax._full_label = name               # _layout 依軸高截短
            ax.format_coord = self._coord_fmt(k)
        main.format_coord = self._coord_fmt(1 if any(s.axis == 1 for s in shown) else None)

    def _coord_fmt(self, k):
        def fmt(x, y):
            head = f"t = {fmt_clock(x, self._time_decimals())}"
            return head if k is None else f"{head} · 軸 {k} = {fmt_value(y)}"
        return fmt

    def _time_decimals(self) -> int:
        """時刻顯示的小數位數:放大到幾秒內時多給位數(一般 1 位)。"""
        if self._main is None:
            return 1
        x0, x1 = self._main.get_xlim()
        span = abs(x1 - x0)
        return 1 if span >= 10 else 2 if span >= 1 else 3

    @staticmethod
    def _label_of(s: _Series) -> str:
        return f"{s.label}({s.unit})" if s.unit else s.label

    def _mode_spans(self):
        """[(t0, t1, 模式名)](絕對秒);最後一段延伸到全範圍終點。"""
        if not self._modes:
            return []
        end = self.full_range()[1]
        out = []
        for i, m in enumerate(self._modes):
            b = self._modes[i + 1].t if i + 1 < len(self._modes) else max(end, m.t)
            if b > m.t:
                out.append((m.t, b, m.label))
        return out

    def _draw_bands(self, main):
        tb = self.t_base
        spans = self._mode_spans()
        for a, b, name in spans:
            main.axvspan(a - tb, b - tb, color=theme.mode_color(name), alpha=0.10,
                         lw=0, zorder=0.2)
        if spans:
            rib = self.fig.add_axes([0.08, 0.9, 0.84, 0.05], sharex=main)
            rib.set_axis_off()
            rib.set_ylim(0, 1)
            rib.set_navigate(False)
            for a, b, name in spans:
                rib.axvspan(a - tb, b - tb, ymin=0.0, ymax=0.26,
                            fc=theme.mode_color(name), ec=theme.SURFACE, lw=1.0, alpha=0.95)
            self._rib = rib
        # ARM 飛行段:底緣綠色細條(狀態色,永遠配「ARM n」文字)
        if self._segs:
            off = transforms.offset_copy(main.get_xaxis_transform(), fig=self.fig,
                                         y=1.6, units="points")
            for s in self._segs:
                main.plot([s.t0 - tb, s.t1 - tb], [0, 0], transform=off, color=theme.S_GOOD,
                          lw=3.0, solid_capstyle="butt", zorder=2.5, alpha=0.9)

    def _build_legend(self, shown):
        handles = [Line2D([], [], color=s.color, lw=2.2) for s in shown]
        multi = len({s.axis for s in shown}) > 1
        labels = [self._label_of(s) + (f" ·軸{s.axis}" if multi else "") for s in shown]
        self._legend_items = (handles, labels)

    def _make_legend(self, ncols: int, y_top_px: float):
        if self._legend is not None:
            self._legend.remove()
        H = self.fig.bbox.height
        W = self.fig.bbox.width
        handles, labels = self._legend_items
        self._legend = self.fig.legend(
            handles, labels, loc="upper left", ncols=ncols,
            bbox_to_anchor=(self._px(8) / W, 1 - y_top_px / H), bbox_transform=self.fig.transFigure,
            frameon=False, fontsize=8.5, handlelength=1.5, handletextpad=0.5,
            columnspacing=1.3, borderaxespad=0.0, borderpad=0.0, labelspacing=0.3,
            labelcolor=theme.INK_2)
        self._legend.set_in_layout(False)
        self._legend_ncols = ncols
        return self._legend

    def _build_cursor(self, shown):
        main = self._main
        ident = transforms.IdentityTransform()
        self._cursor_line = self._animated(main.plot([], [], color=theme.INK, lw=1.1, zorder=20,
                                                     transform=main.get_xaxis_transform())[0])
        for s in shown:
            s.dot = self._animated(self._axes[s.axis].plot(
                [], [], "o", ms=5.5, mfc=s.color, mec=theme.SURFACE, mew=1.2, zorder=21)[0])
        self._box = self._animated(FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=5", transform=ident,
            fc=theme.SURFACE, ec=theme.BORDER, lw=1.0, alpha=0.92, zorder=30))
        self.fig.add_artist(self._box)
        self._box_head = self._animated(self.fig.text(0, 0, "", transform=ident, color=theme.INK,
                                                      fontsize=8.5, fontweight="bold",
                                                      ha="left", va="top", zorder=31))
        self._box_rows = []
        for s in shown:
            mark = Line2D([0, 1], [0, 0], transform=ident, color=s.color, lw=2.6,
                          solid_capstyle="round", zorder=31)
            self.fig.add_artist(mark)
            lab = self.fig.text(0, 0, s.label, transform=ident, color=theme.INK_2, fontsize=8.5,
                                ha="left", va="top", zorder=31)
            val = self.fig.text(0, 0, "", transform=ident, color=theme.INK, fontsize=8.5,
                                ha="right", va="top", zorder=31)
            self._box_rows.append((s, self._animated(mark), self._animated(lab), self._animated(val)))

    # ---------------- 版面(像素邊界) ----------------

    def _px(self, v: float) -> float:
        return v * max(getattr(self.canvas, "device_pixel_ratio", 1.0) or 1.0, 1.0)

    def _yaxis_width(self, ax, k: int, rend) -> float:
        """y 軸刻度 + 標籤佔的寬度(像素,不含 spine 外移量)。"""
        try:
            bb = ax.yaxis.get_tightbbox(rend)
        except Exception:  # noqa: BLE001
            bb = None
        if bb is None or bb.width <= 0:
            return self._px(6)
        spine = ax.spines["left" if k in (1, 4) else "right"]
        sx = spine.get_window_extent(rend)
        if k in (1, 4):
            return max(sx.x0 - bb.x0, self._px(6))
        return max(bb.x1 - sx.x1, self._px(6))

    def _layout(self, force: bool = False) -> bool:
        """依實際刻度寬度算出四邊像素邊界,擺放主圖、模式色條與圖例。改了才回傳 True。"""
        main = self._main
        if main is None:
            return False
        fig = self.fig
        W, H = fig.bbox.width, fig.bbox.height
        if W < self._px(120) or H < self._px(90):
            return False
        rend = self.canvas.get_renderer()
        px = self._px

        # 圖例:能一行就一行,太寬就減欄數
        top = px(6)
        handles, labels = self._legend_items
        if handles:
            n = len(handles)
            ncols = n
            while True:
                leg = self._make_legend(ncols, top)
                w = leg.get_window_extent(rend).width
                if w <= W - px(16) or ncols == 1:
                    break
                ncols = max(1, min(ncols - 1, int(ncols * (W - px(16)) / max(w, 1.0))))
            top += leg.get_window_extent(rend).height + px(6)
        rib_h = px(20) if self._rib is not None else 0.0
        bottom = px(24)

        # y 軸標籤依軸高截短(太長會超出軸高互相重疊)
        ax_h = max(H - top - rib_h - bottom, px(40))
        for ax in self._axes.values():
            full = getattr(ax, "_full_label", "")
            if full:
                ax.set_ylabel(_fit_text(full, ax_h / px(5.6)))

        w = {k: self._yaxis_width(ax, k, rend) for k, ax in self._axes.items()}
        gap = px(8)
        off2 = w.get(2, 0.0) + gap if 2 in self._axes else 0.0
        off1 = w[1] + gap
        left = px(6) + w[1] + (w[4] + gap if 4 in self._axes else 0.0)
        right = px(10) + (w[2] + gap if 2 in self._axes else 0.0) + (w[3] + gap if 3 in self._axes else 0.0)
        new = (left, right, top, bottom)
        old = self._margins
        if not force and old is not None and all(abs(a - b) < px(3) for a, b in zip(old, new)):
            return False
        self._margins = new
        dpi = fig.dpi
        if 3 in self._axes:
            self._axes[3].spines["right"].set_position(("outward", off2 * 72.0 / dpi))
        if 4 in self._axes:
            self._axes[4].spines["left"].set_position(("outward", off1 * 72.0 / dpi))
        aw = max(W - left - right, px(40))
        main.set_position([left / W, bottom / H, aw / W, ax_h / H])
        for k, ax in self._axes.items():
            if k != 1:
                ax.set_position(main.get_position())
        if self._rib is not None:
            self._rib.set_position([left / W, (bottom + ax_h) / H, aw / W, rib_h / H])
        return True

    # ---------------- 資料 / 標籤更新(視野改變時) ----------------

    def _update_data(self):
        main = self._main
        if main is None:
            return
        x0, x1 = main.get_xlim()
        tb = self.t_base
        lo, hi = x0 + tb, x1 + tb
        nb = int(min(max(main.bbox.width, 100), _MAX_BINS))
        ranges: dict[int, list[float]] = {}
        for s in self._shown():
            if s.line is None:
                continue
            ts, vs = decimate(s.t, s.v, lo, hi, nb, s.gaps,
                              hold=s.gap / 2 if s.kind == K_STEP else None)
            s.line.set_data(ts - tb, vs)
            s.line.set_marker("o" if 0 < ts.size <= _MARKER_MAX else "None")
            inside = (ts >= lo) & (ts <= hi) & np.isfinite(vs)
            if not inside.any() and ts.size:    # 視野內沒有樣本:用跨過視野的兩端點
                inside = np.isfinite(vs)
            if inside.any():
                r = ranges.setdefault(s.axis, [np.inf, -np.inf])
                r[0] = min(r[0], float(np.min(vs[inside])))
                r[1] = max(r[1], float(np.max(vs[inside])))
        if self._y_auto:
            with self._silent():
                for k, ax in self._axes.items():
                    r = ranges.get(k)
                    if r is None:
                        continue
                    a, b = r
                    span = b - a
                    pad = span * 0.07 if span > 0 else max(abs(a) * 0.05, 0.5)
                    ax.set_ylim(a - pad, b + pad)

    def _update_labels(self):
        """模式名稱 / ARM 標籤:只畫放得下的(段內寬度夠才標,否則試縮寫,再不行就略過)。"""
        for t in self._mode_texts + self._arm_texts:
            t.remove()
        self._mode_texts, self._arm_texts = [], []
        main = self._main
        if main is None:
            return
        rend = self.canvas.get_renderer()
        x0, x1 = main.get_xlim()
        tb = self.t_base
        px = self._px

        def place(ax, spans, y, color_fn, fs, out, bold):
            last = -np.inf
            for a, b, name in spans:
                a, b = a - tb, b - tb
                if b <= x0 or a >= x1:
                    continue
                xa = max(a, x0)
                pa = main.transData.transform((xa, 0))[0]
                pb = main.transData.transform((min(b, x1), 0))[0]
                if pa < last + px(6):
                    continue
                room = pb - pa - px(5)
                for txt in (name, name[:3]):
                    if len(txt) * fs * self.fig.dpi / 72.0 * 0.5 > room:
                        continue                 # 估計就放不下,不必建立文字量測
                    t = ax.text(xa, y, txt, color=color_fn(name), fontsize=fs,
                                fontweight="bold" if bold else "normal", ha="left", va="bottom",
                                transform=transforms.offset_copy(
                                    ax.get_xaxis_transform() if ax is main else ax.transData,
                                    fig=self.fig, x=2.5, units="points"),
                                clip_on=True, zorder=6)
                    wpx = t.get_window_extent(rend).width
                    if wpx <= room:
                        out.append(t)
                        last = pa + wpx
                        break
                    t.remove()

        if self._rib is not None:
            place(self._rib, self._mode_spans(), 0.34, theme.mode_color, 7.5,
                  self._mode_texts, True)
        if self._segs:
            spans = [(s.t0, s.t1, s.label) for s in self._segs]
            place(main, spans, 0.0, lambda _n: theme.S_GOOD, 7, self._arm_texts, True)
            off = px(4.5) / max(main.bbox.height, 1.0)
            for t in self._arm_texts:          # 放在綠條上方一點
                t.set_y(off)

    def _schedule_refresh(self):
        """延後重算(合併連續的視野變更);已經等超過 _REFRESH_MAX_S 就不再往後延。"""
        now = time.monotonic()
        if not self._refresh_timer.isActive():
            self._refresh_since = now
            self._refresh_timer.start()
        elif now - self._refresh_since < _REFRESH_MAX_S:
            self._refresh_timer.start()       # 重新計時
        # 否則讓計時器照原定時間觸發:長手勢中途也會更新

    def _refresh(self):
        """視野改變後:重新抽稀、y 自動縮放、重排版面與標籤,排程重繪。"""
        self._refresh_timer.stop()
        if self._main is None:
            return
        try:
            with self._silent():
                self._update_data()
                self._layout()
                self._update_labels()
                if self._anim:
                    self.update_cursor(self._t)
        except Exception:  # noqa: BLE001 — 計時器槽內的例外會讓 PyQt6 直接中止程式
            import traceback
            traceback.print_exc()
        self.canvas.draw_idle()

    def _emit_view(self):
        r = self.view_range()
        if r is not None:
            self.viewChanged.emit(float(r[0]), float(r[1]))

    # ---------------- matplotlib 回呼 ----------------

    def _on_xlim_changed(self, ax):
        if self._quiet:
            return
        x0, x1 = ax.get_xlim()
        self._view = (x0, x1)
        self._schedule_refresh()
        if not self._no_echo:
            self._view_timer.start()

    def _on_ylim_changed(self, _ax):
        if not self._quiet:
            self._y_auto = False              # 使用者(工具列)手動縮放過 y,不再自動

    def _on_resize(self, _event):
        if self._main is None:
            return
        try:
            with self._silent():
                self._layout(force=True)
                self._update_data()
                self._update_labels()
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()

    # ---------------- 滑鼠 ----------------

    def _toolbar_busy(self) -> bool:
        tb = self.toolbar
        if tb is None:
            return False
        mode = getattr(tb, "mode", "")
        return bool(getattr(mode, "value", mode))

    def _in_plot_x(self, ev) -> bool:
        m = self._main
        if m is None or ev.x is None:
            return False
        bb = m.bbox
        top = self._rib.bbox.y1 if self._rib is not None else bb.y1
        return bb.x0 <= ev.x <= bb.x1 and bb.y0 <= ev.y <= top

    def _x_at(self, x_px: float) -> float:
        return float(self._main.transData.inverted().transform((x_px, 0))[0])

    def _on_press(self, ev):
        if self._main is None or getattr(ev.button, "value", ev.button) != 1 or self._toolbar_busy():
            return
        if ev.dblclick:
            self._press = None
            if self._in_plot_x(ev):
                self.reset_view()
            return
        if not self._in_plot_x(ev):
            return
        self._press = (ev.x, ev.y, self._main.get_xlim())
        self._dragging = False

    def _on_motion(self, ev):
        if self._press is None:
            self._hover(ev)
            return
        if ev.x is None:
            return
        x_px, y_px, (x0, x1) = self._press
        dx = ev.x - x_px
        if not self._dragging and abs(dx) < self._px(_DRAG_PX) and abs(ev.y - y_px) < self._px(_DRAG_PX):
            return
        self._dragging = True
        QToolTip.hideText()
        scale = (x1 - x0) / max(self._main.bbox.width, 1.0)
        shift = -dx * scale
        lo, hi = self._clamp_span(x0 + shift, x1 + shift)
        self._main.set_xlim(lo, hi)
        self.canvas.draw_idle()

    def _on_release(self, ev):
        press, dragging = self._press, self._dragging
        self._press = None
        self._dragging = False
        if press is None or dragging or self._main is None:
            return
        t = self._x_at(press[0]) + self.t_base
        lo, hi = self.full_range()
        self.seeked.emit(float(min(max(t, lo), hi)))

    def _clamp_span(self, a: float, b: float) -> tuple[float, float]:
        """視野限制在全範圍內(平移到邊界就停,不改變跨度)。"""
        lo, hi = self.full_range()
        lo, hi = lo - self.t_base, hi - self.t_base
        span = b - a
        if span >= hi - lo:
            return lo, hi
        if a < lo:
            a, b = lo, lo + span
        if b > hi:
            a, b = hi - span, hi
        return a, b

    @staticmethod
    def _shift_held(ev) -> bool:
        if "shift" in (getattr(ev, "modifiers", None) or ()) or ev.key == "shift":
            return True
        gui = getattr(ev, "guiEvent", None)
        try:
            return bool(gui is not None and gui.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        except (AttributeError, RuntimeError, TypeError):
            return False

    def _on_scroll(self, ev):
        if self._main is None or not self._in_plot_x(ev) or self._toolbar_busy():
            return
        x0, x1 = self._main.get_xlim()
        if self._shift_held(ev):                         # Shift+滾輪:平移
            d = -(x1 - x0) * 0.12 * ev.step
            a, b = self._clamp_span(x0 + d, x1 + d)
        else:
            xc = self._x_at(ev.x)
            f = 0.8 ** float(ev.step)
            lo, hi = self.full_range()
            span = min(max((x1 - x0) * f, float(self.min_span)), hi - lo)
            r = (xc - x0) / max(x1 - x0, 1e-12)
            a, b = self._clamp_span(xc - r * span, xc - r * span + span)
        if (a, b) != (x0, x1):
            self._main.set_xlim(a, b)
            self.canvas.draw_idle()

    def _hover(self, ev):
        """滑到模式色條上:提示模式名稱與起訖時間。"""
        rib = self._rib
        if rib is None or ev.x is None or ev.inaxes is not rib and not (
                rib.bbox.x0 <= ev.x <= rib.bbox.x1 and rib.bbox.y0 <= ev.y <= rib.bbox.y1):
            QToolTip.hideText()
            return
        t = self._x_at(ev.x) + self.t_base
        for a, b, name in self._mode_spans():
            if a <= t < b:
                tb = self.t_base
                QToolTip.showText(QCursor.pos(),
                                  f"{name}\n{fmt_clock(a - tb)} – {fmt_clock(b - tb)}"
                                  f"({b - a:.1f} 秒)", self)
                return
        QToolTip.hideText()

    # ---------------- 播放游標 ----------------

    def _bg_stale(self) -> bool:
        """視野已改、底圖還是舊視野(重算 / 重畫還沒跑):這時 blit 會把游標疊在錯的圖上。"""
        return (self._main is not None and self._bg is not None and self._bg_xlim is not None
                and tuple(self._main.get_xlim()) != self._bg_xlim)

    def set_time(self, t) -> None:
        if not self._bg_stale():
            super().set_time(t)
            return
        # 只更新游標位置,不 blit;排定的重畫(_refresh → draw_idle)完成時 _on_draw 會疊上游標
        self._t = None if t is None else float(t)
        if self._anim:
            try:
                self.update_cursor(self._t)
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
        if not self._refresh_timer.isActive():
            self.canvas.draw_idle()

    def _on_draw(self, event):
        super()._on_draw(event)
        if self._main is not None and not self.canvas.is_saving() and self.fig.canvas is self.canvas:
            self._bg_xlim = tuple(self._main.get_xlim())

    def update_cursor(self, t):
        if self._main is None or self._cursor_line is None:
            return False
        main = self._main
        x0, x1 = main.get_xlim()
        x = None if t is None else t - self.t_base
        if x is None or not (x0 <= x <= x1):
            for a in self._anim:
                a.set_visible(False)
            return False
        self._cursor_line.set_data([x, x], [0.0, 1.0])
        self._cursor_line.set_visible(True)
        for s in self._shown():
            if s.dot is None:
                continue
            val = s.value_at(t)
            if val is None:
                s.dot.set_visible(False)
            else:
                s.dot.set_data([x], [val])
                s.dot.set_visible(True)
        self._layout_box(t, x)
        return False

    def _layout_box(self, t: float, x: float):
        """數值框:播放頭旁(貼近右緣時翻到左側),列出時刻、模式與各序列此刻的值。"""
        main = self._main
        rend = self.canvas.get_renderer()
        px = self._px
        head = fmt_clock(t - self.t_base, self._time_decimals())
        if self._modes:
            m = mode_at(self._modes, t)
            if m:
                head += f" · {m}"
        self._box_head.set_text(head)
        self._box_head.set_visible(True)
        lh = 8.5 / 72.0 * self.fig.dpi * 1.5
        pad = px(7)
        mark_w = px(12)
        max_rows = max(0, int((main.bbox.height - 2 * pad - lh) // lh))
        rows = self._box_rows[:max_rows]
        for _s, mark, lab, val in self._box_rows[max_rows:]:
            for a in (mark, lab, val):
                a.set_visible(False)
        lab_w = val_w = 0.0
        for s, mark, lab, val in rows:
            v = s.value_at(t)
            val.set_text(s.fmt(v) + (f" {s.unit}" if s.unit and v is not None else ""))
            for a in (mark, lab, val):
                a.set_visible(True)
            lab_w = max(lab_w, lab.get_window_extent(rend).width)
            val_w = max(val_w, val.get_window_extent(rend).width)
        head_w = self._box_head.get_window_extent(rend).width
        body_w = mark_w + px(6) + lab_w + px(14) + val_w if rows else 0.0
        w = max(head_w, body_w) + 2 * pad
        h = lh * (1 + len(rows)) + 2 * pad - lh * 0.25
        xp = main.transData.transform((x, 0))[0]
        bb = main.bbox
        left = xp + px(10)
        if left + w > bb.x1 - px(4):
            left = xp - px(10) - w
        left = max(left, bb.x0 + px(4))
        top = bb.y1 - px(6)
        self._box.set_bounds(left, top - h, w, h)
        self._box.set_visible(True)
        self._box_head.set_position((left + pad, top - pad))
        y = top - pad - lh
        for s, mark, lab, val in rows:
            yc = y - lh * 0.36
            mark.set_data([left + pad, left + pad + mark_w], [yc, yc])
            lab.set_position((left + pad + mark_w + px(6), y))
            val.set_position((left + w - pad, y))
            y -= lh
