"""matplotlib 圖表元件(全部畫在深色表面上)。

每張圖的序列角色固定:參考=藍、估測=橘、誤差=洋紅、誤差水平=紫。
軸向誤差用單序列 small multiples(標題命名身分),不另外配類別色。

播放游標:各圖「目前時刻」的標記是 animated artist,不參與一般重繪;
set_time() 只把它們疊在快取的底圖上再 blit,播放時每秒數十次也不必整張重畫。

延後繪製:視窗開著、但圖藏在別的分頁後時,重繪只先清空,真正的繪製留到分頁顯示
(showEvent)才做,而且只做最後一次;從沒顯示過的獨立元件(離線截圖、測試)照常立即畫。
"""
from __future__ import annotations

import traceback

import matplotlib
import numpy as np

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,  # noqa: E402
                             QSizePolicy, QVBoxLayout, QWidget)

from core import align as align_mod  # noqa: E402
from core import pipeline  # noqa: E402
from core.timeline import (Track, format_clock, horizontal_speed, mode_at,  # noqa: E402
                           mode_changes, segment_index_at)
from core.types import EV_RESET  # noqa: E402

from . import theme  # noqa: E402

theme.apply_mpl(matplotlib)

_HINT = "載入 log 或模擬資料後,按「計算 ATE」"
# U+2212 在多數中文字型缺字形,一律用 ASCII 連字號
MINUS = "-"
CMAP_ERR = LinearSegmentedColormap.from_list("ate_seq", theme.SEQ_RAMP)
_ANNOT_BOX = dict(boxstyle="round,pad=0.25", fc=theme.SURFACE, ec="none", alpha=0.85)


class _Toolbar(NavigationToolbar2QT):
    """matplotlib 工具列;「存檔」鈕和 PlotWidget.save_png 一樣遵守 EXPORT_CURSOR。"""

    def save_figure(self, *args):
        owner = self.parent()
        flush = getattr(owner, "flush_draw", None)
        if callable(flush):
            flush()                        # 延後的繪製先做完,存的才是目前內容
        shown = ([] if getattr(owner, "EXPORT_CURSOR", True)
                 else [(a, a.get_visible()) for a in getattr(owner, "_anim", [])])
        try:
            for a, _ in shown:
                a.set_visible(False)
            return super().save_figure(*args)
        finally:
            for a, vis in shown:
                a.set_visible(vis)
            self.canvas.draw_idle()


class PlotWidget(QWidget):
    """帶工具列的 matplotlib 畫布;子類實作 draw_result()。

    播放游標:子類以 _animated() 登記游標 artist 並實作 update_cursor(t),
    外部用 set_time(t) 驅動。每次完整重繪(draw_event)時快取不含游標的底圖並
    疊上游標(縮放、平移、改視窗大小後游標都還在);之後 set_time() 只做
    「還原底圖 → 重畫游標 → blit」。

    延後繪製:視窗看得到、本圖卻藏著(在別的分頁)時,_redraw() 只清空圖面並記下
    最後一次的 (draw_fn, args),showEvent 時才真的畫;flush_draw() 可提早做完
    (save_png 會先呼叫)。從沒顯示過的獨立元件照常立即畫。
    """

    EXPORT_CURSOR = False   # 存圖是否保留播放游標(一般圖表不留;播放畫面要留才有意義)

    def __init__(self, parent=None, toolbar: bool = True):
        super().__init__(parent)
        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor=theme.SURFACE)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._anim: list = []              # 游標 artist(animated)
        self._bg = None                    # blit 底圖(不含游標)
        self._bg_size = None               # 底圖對應的畫布像素大小
        self._t: float | None = None       # 目前播放頭(log 絕對秒)
        self.t_base = 0.0                  # 繪圖時的 logdata.t_base(橫軸 = t - t_base)
        self._pending = None               # 延後的繪製 (draw_fn, args):藏著時只留最後一次
        self._draw_stale = False           # 藏著時欠一次畫布重繪(顯示時補)
        self.canvas.mpl_connect("draw_event", self._on_draw)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.toolbar = None
        if toolbar:
            tb = _Toolbar(self.canvas, self)
            tb.setStyleSheet(
                f"QToolBar{{background:{theme.PANEL};border:none;padding:2px;}}"
                f"QToolButton{{color:{theme.INK_2};padding:3px;}}"
                f"QToolButton:hover{{background:#3a3a37;border-radius:4px;}}"
                f"QLabel{{color:{theme.MUTED};}}")
            lay.addWidget(tb)
            self.toolbar = tb
        lay.addWidget(self.canvas)
        self.show_placeholder()

    def show_placeholder(self, text: str = _HINT):
        self._pending = None               # 佔位訊息是最新內容:取代尚未畫的延後繪製
        self._clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(theme.SURFACE)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center",
                color=theme.MUTED, fontsize=11, transform=ax.transAxes)
        self._request_draw()

    def render(self, res, logdata):
        self._redraw(self.draw_result, res, logdata)

    def draw_result(self, res, logdata):
        raise NotImplementedError

    def _clear(self):
        self.fig.clear()
        self._anim = []
        self._bg = None

    def _redraw(self, draw_fn, *args):
        """清空 → 畫靜態內容 → 游標移到目前播放頭 → 排程重繪;失敗時顯示佔位訊息。

        藏在別的分頁後時只清空(舊游標不再被驅動),繪製延到顯示時;只保留最後一次。
        """
        try:
            self._clear()
        except Exception as e:  # noqa: BLE001
            self.show_placeholder(f"繪圖失敗:{type(e).__name__}: {e}")
            return
        if self._defer_ok():
            self._pending = (draw_fn, args)
            self._draw_stale = True
            return
        self._pending = None
        self._draw_now(draw_fn, args)

    def _draw_now(self, draw_fn, args) -> None:
        try:
            draw_fn(*args)
            if self._t is not None and self._anim:
                self.update_cursor(self._t)
        except Exception as e:  # noqa: BLE001
            self.show_placeholder(f"繪圖失敗:{type(e).__name__}: {e}")
            return
        self._request_draw()

    # ---------------- 延後繪製 ----------------

    def _defer_ok(self) -> bool:
        """視窗看得到、本圖卻藏著(別的分頁):繪製可以延到顯示時。

        從沒顯示過的獨立元件(視窗本身也看不到)不延後 —— 離線截圖直接 savefig 要有內容。
        """
        return not self.isVisible() and self.window().isVisible()

    def _request_draw(self) -> None:
        """排程畫布重繪;藏著時只記下,顯示時才畫(藏著的畫布重繪沒人看得到)。"""
        if self._defer_ok():
            self._draw_stale = True
            return
        self._draw_stale = False
        self.canvas.draw_idle()

    def has_pending_draw(self) -> bool:
        """有延後(尚未執行)的繪製。"""
        return self._pending is not None

    def flush_draw(self) -> None:
        """把延後的繪製做完(顯示時自動呼叫;存圖前、或要讀圖面內容前可手動呼叫)。"""
        pending, self._pending = self._pending, None
        if pending is not None:
            self._draw_now(*pending)
        if self._draw_stale and not self._defer_ok():
            self._draw_stale = False
            self.canvas.draw_idle()

    def showEvent(self, ev):
        super().showEvent(ev)
        try:
            self.flush_draw()
        except Exception:  # noqa: BLE001 — Qt 事件處理內未捕捉的例外會讓 PyQt6 直接中止程式
            traceback.print_exc()

    def _equal_xy(self, ax, x, y, pad_frac: float = 0.08):
        """等比例座標:避免形狀被拉伸而誤判軌跡偏差。

        用 adjustable="box"(縮 axes 外框)而非 "datalim"(改資料範圍),
        後者會與明確設定的 xlim/ylim 衝突並被 matplotlib 默默忽略其一。
        """
        ax.set_aspect("equal", adjustable="box")
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        half = max(x.max() - x.min(), y.max() - y.min(), 1.0) / 2 * (1 + pad_frac)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    # ---------------- 播放游標 ----------------

    def set_time(self, t: float | None) -> None:
        """移動播放頭(log 絕對秒;None = 不顯示游標)。只需對可見的圖呼叫。"""
        self._t = None if t is None else float(t)
        if not self._anim:                 # 沒有游標(或繪製延後中:畫完時會套用 self._t)
            return
        try:
            full = self.update_cursor(self._t)
        except Exception:  # noqa: BLE001 — 計時器槽內未捕捉的例外會讓 PyQt6 直接中止程式
            traceback.print_exc()
            self._drop_cursors()
            self._request_draw()
            return
        if (full or self._bg is None or self._draw_stale or self._defer_ok()
                or self._bg_size != self._canvas_size()):
            self._request_draw()           # 重繪完成時 _on_draw 會重抓底圖並疊上游標
            return
        self.canvas.restore_region(self._bg)
        self._draw_cursors()
        self.canvas.blit(self.fig.bbox)

    def update_cursor(self, t: float | None) -> bool:
        """子類覆寫:依 t 更新游標 artist 的資料與可見性。

        t 為 None 或超出本圖資料範圍時隱藏游標。只有視野因此改變(例如跟隨模式)
        才回傳 True,要求整張重繪。
        """
        return False

    def save_png(self, path: str, dpi: int = 160) -> None:
        """存圖。savefig 會把 animated artist 一起畫進去:EXPORT_CURSOR 為假時先藏起游標。"""
        self.flush_draw()                  # 延後的繪製先做完,存的才是目前內容
        shown = [] if self.EXPORT_CURSOR else [(a, a.get_visible()) for a in self._anim]
        try:
            for a, _ in shown:
                a.set_visible(False)
            self.fig.savefig(path, dpi=dpi, facecolor=theme.SURFACE)
        finally:
            for a, vis in shown:
                a.set_visible(vis)
            self._request_draw()           # savefig 換過 renderer,畫面重畫一次

    def _animated(self, artist):
        """登記游標 artist:animated(一般重繪略過)、預設隱藏、不參與版面計算。"""
        artist.set_animated(True)
        artist.set_visible(False)
        artist.set_in_layout(False)
        self._anim.append(artist)
        return artist

    def _pair_cursor(self, ax, z: float = 8.0):
        """參考(圓)/ 估測(菱形)目前位置 + 兩者之間的誤差連線。"""
        link = self._animated(ax.plot([], [], color=theme.INK, lw=1.4,
                                      solid_capstyle="round", zorder=z)[0])
        mref = self._animated(ax.plot([], [], "o", ms=10, mfc=theme.C_REF,
                                      mec=theme.INK, mew=1.6, zorder=z + 1)[0])
        mest = self._animated(ax.plot([], [], "D", ms=8.5, mfc=theme.C_EST,
                                      mec=theme.INK, mew=1.6, zorder=z + 2)[0])
        return link, mref, mest

    def _drop_cursors(self):
        for a in self._anim:
            a.set_visible(False)
        self._anim = []

    def _canvas_size(self) -> tuple[int, int]:
        w, h = self.fig.bbox.size
        return int(round(w)), int(round(h))

    def _on_draw(self, _event):
        # savefig 也會觸發 draw_event;那時 animated artist 已由 matplotlib 畫進檔案。
        # 存 PDF / SVG 時 fig.canvas 會暫時換成向量畫布,Qt 畫布的 is_saving() 為假
        if self.canvas.is_saving() or self.fig.canvas is not self.canvas:
            return
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._bg_size = self._canvas_size()
        self._draw_cursors()

    def _draw_cursors(self):
        for a in sorted(self._anim, key=lambda a: a.get_zorder()):
            if a.get_visible():
                self.fig.draw_artist(a)


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, color=theme.GRID, linewidth=0.8)


def _endpoints(ax, x, y, color, label_prefix=""):
    """起點空心圓、終點實心方塊 — 形狀差異讓方向不靠顏色。"""
    ax.plot(x[0], y[0], "o", ms=9, mfc="none", mec=color, mew=2.0, zorder=5,
            label=f"{label_prefix}起點" if label_prefix else None)
    ax.plot(x[-1], y[-1], "s", ms=8, color=color, zorder=5,
            mec=theme.SURFACE, mew=1.5,
            label=f"{label_prefix}終點" if label_prefix else None)


# ---------------- 游標小工具 ----------------

def _set_point(line, p) -> None:
    """單點標記移到 p 的 (x, y);p 為 None 時隱藏。"""
    if p is None:
        line.set_visible(False)
        return
    line.set_data([p[0]], [p[1]])
    line.set_visible(True)


def _set_vline(line, x) -> None:
    """垂直播放線(transform 為 axes 的 xaxis transform);x 為 None 時隱藏。"""
    if x is None:
        line.set_visible(False)
        return
    line.set_data([x, x], [0.0, 1.0])
    line.set_visible(True)


def _show_pair(cur, r, e) -> None:
    link, mref, mest = cur
    _set_point(mref, r)
    _set_point(mest, e)
    if r is None or e is None:
        link.set_visible(False)
    else:
        link.set_data([r[0], e[0]], [r[1], e[1]])
        link.set_visible(True)


_TRAIL_FULL = 1500      # 尾巴最近這麼多點保持原解析度
_TRAIL_OLD = 2500       # 更舊的部分最多畫這麼多點(「全部」尾巴在長 log 上才不會拖慢播放)


def _set_trail(line, brk, t_from: float, t_to: float, cur) -> None:
    """尾巴 = [t_from, t_to] 內的樣本 + 播放頭內插點;此刻無資料時隱藏。

    很長的尾巴把較舊的部分抽稀:取絕對索引為 2 的冪次倍數的樣本(播放時取樣點固定,
    不會閃動),並保留空洞 / 重置處的 NaN 斷點。
    """
    if cur is None:
        line.set_visible(False)
        return
    tt, pp, nan_idx = brk
    a = int(np.searchsorted(tt, t_from, side="left"))
    b = int(np.searchsorted(tt, t_to, side="right"))
    if b - a > _TRAIL_FULL + _TRAIL_OLD:
        m = b - _TRAIL_FULL
        step = 1 << int(np.ceil(np.log2((m - a) / _TRAIL_OLD)))
        idx = np.arange(-(-a // step) * step, m, step)
        brk_in = nan_idx[np.searchsorted(nan_idx, a):np.searchsorted(nan_idx, m)]
        if brk_in.size:
            idx = np.union1d(idx, brk_in)
        xy = np.concatenate([pp[idx, :2], pp[m:b, :2]])
    else:
        xy = pp[a:b, :2]
    line.set_data(np.append(xy[:, 0], cur[0]), np.append(xy[:, 1], cur[1]))
    line.set_visible(True)


_FAINT_MAX = 8000       # 淡色「整份 log / 分析區間」線最多畫這麼多點(長 log 每次重繪才不會拖慢)


def _decimate_path(p, n_max: int = _FAINT_MAX) -> np.ndarray:
    """平面軌跡 (N×≥2,可含 NaN 斷點列) 抽稀成約 n_max 點,回傳 (M×2);見 _decimate_keep。"""
    xy = np.asarray(p, dtype=float)[:, :2]
    return xy if xy.shape[0] <= n_max else xy[_decimate_keep(xy, n_max)]


def _decimate_keep(xy: np.ndarray, n_max: int) -> np.ndarray:
    """抽稀要留下的列(布林遮罩)。

    依累積路徑長每 eps(= 總長 / n_max)留一點:飛行段點距均勻、地面停留的大量重疊點
    整段省掉;每個原始點離前一個留下的點的路徑長都小於 eps(形狀誤差 < eps)。
    NaN 斷點列與其前後端點、首末點一律保留,空洞 / 重置處仍然斷開。
    """
    n = xy.shape[0]
    bad = ~np.isfinite(xy).all(axis=1)
    step = np.hypot(*np.diff(xy, axis=0).T)
    step[~np.isfinite(step)] = 0.0
    s = np.concatenate(([0.0], np.cumsum(step)))
    eps = s[-1] / n_max
    if eps > 0:
        b = np.floor(s / eps)
        keep = np.r_[True, b[1:] != b[:-1]]
    else:                                    # 完全沒移動:只留端點與斷點
        keep = np.zeros(n, dtype=bool)
        keep[0] = True
    keep[-1] = True
    keep |= bad
    keep[:-1] |= bad[1:]                     # 斷點前一點
    keep[1:] |= bad[:-1]                     # 斷點後一點
    return keep


def _in_range(t, lo: float, hi: float) -> bool:
    return t is not None and lo - 1e-6 <= t <= hi + 1e-6


def _res_gap(res) -> float:
    """結果軌跡的游標內插間隙上限(同步時已剔除估測空洞,這裡只防參考端掉資料)。"""
    return max(1.0, 2.0 * float(res.cfg.get("max_gap_s", 0.5)))


def _src_gap(t) -> float:
    """單一來源的內插間隙上限:取樣週期的 3 倍,至少 1 秒(低頻來源不會格格被當空洞)。"""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return 1.0
    return max(1.0, 3.0 * float(np.median(np.diff(t))))


def _gap_broken(t, p, max_gap: float, cuts=()):
    """在資料空洞(> max_gap 秒)與指定時刻(如 VIO 重置)插入 NaN 列,
    畫線時才不會跨過去連成一條假直線。

    回傳 (t', p', NaN 列索引);t' 仍遞增,可直接 searchsorted。
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    k = set()
    if t.size >= 2:
        k = set((np.flatnonzero(np.diff(t) > max_gap) + 1).tolist())
        for c in cuts:
            j = int(np.searchsorted(t, c))
            if 0 < j < t.size:
                k.add(j)
    if not k:
        return t, p, np.empty(0, dtype=int)
    k = np.asarray(sorted(k))
    return (np.insert(t, k, 0.5 * (t[k - 1] + t[k])),
            np.insert(p, k, np.nan, axis=0),
            k + np.arange(k.size))


class TopViewPlot(PlotWidget):
    """平面軌跡(東-北)+ 誤差連線。"""

    def draw_result(self, res, logdata):
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(theme.SURFACE)
        r, e = res.ref_enu, res.est_enu

        # 誤差連線:每隔幾點畫一條,直接讓「差多少」看得見。
        # 軌跡尺度通常遠大於誤差,這裡只給尺度感 — 要看誤差落在哪請用「誤差熱點」分頁。
        stride = max(1, res.n // 140)
        k = np.arange(0, res.n, stride)
        ax.add_collection(LineCollection(
            np.stack([np.column_stack([r[k, 0], r[k, 1]]),
                      np.column_stack([e[k, 0], e[k, 1]])], axis=1),
            colors=theme.C_ERR, linewidths=1.1, alpha=0.65, zorder=1))

        ax.plot(r[:, 0], r[:, 1], color=theme.C_REF, lw=2.0,
                label=f"參考:{res.ref_name}", zorder=3)
        ax.plot(e[:, 0], e[:, 1], color=theme.C_EST, lw=2.0,
                label=f"估測:{res.est_name}", zorder=4)
        _endpoints(ax, r[:, 0], r[:, 1], theme.C_REF)
        _endpoints(ax, e[:, 0], e[:, 1], theme.C_EST)

        # 最大誤差點:註解出來(狀態色 + 文字標籤,不靠顏色單獨表意)
        k = int(np.argmax(res.e3))
        ax.plot([r[k, 0], e[k, 0]], [r[k, 1], e[k, 1]],
                color=theme.S_WARN, lw=2.0, zorder=6)
        ax.annotate(f"最大誤差 {res.e3[k]:.2f} m",
                    xy=(e[k, 0], e[k, 1]), xytext=(12, 12), textcoords="offset points",
                    color=theme.S_WARN, fontsize=9, fontweight="bold", zorder=7,
                    arrowprops=dict(arrowstyle="-", color=theme.S_WARN, lw=1.0))

        self._equal_xy(ax, np.concatenate([r[:, 0], e[:, 0]]),
                       np.concatenate([r[:, 1], e[:, 1]]))
        ax.set_xlabel("東 E(公尺)")
        ax.set_ylabel("北 N(公尺)")
        ax.set_title(f"平面軌跡對照 — 3D RMSE {res.stats['3d']['rmse']:.3f} m"
                     f"(軌跡尺度 {res.path_len_ref:.0f} m,誤差相對很小是正常的)")
        _style_axes(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99),
                  labelspacing=0.35, handlelength=1.6)

        # 播放游標:目前時刻的參考 / 估測位置與兩者之間的誤差
        gap = _res_gap(res)
        self._trk = (Track(res.t, r, gap, edge=gap / 2), Track(res.t, e, gap, edge=gap / 2))
        self._cur = self._pair_cursor(ax)
        self.fig.tight_layout()

    def update_cursor(self, t):
        if t is None:
            _show_pair(self._cur, None, None)
        else:
            _show_pair(self._cur, self._trk[0].at(t), self._trk[1].at(t))
        return False


class HeatmapPlot(PlotWidget):
    """誤差熱點:參考軌跡依該時刻的 3D 誤差著色。

    軌跡尺度(數十~數百公尺)通常遠大於誤差(公尺級),兩條線疊圖看不出差別;
    把誤差映成顏色畫在軌跡上,才看得出「在場地的哪個位置」誤差變大。
    """

    def draw_result(self, res, logdata):
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(theme.SURFACE)
        p = res.ref_enu
        segs = np.stack([p[:-1, :2], p[1:, :2]], axis=1)
        mag = 0.5 * (res.e3[:-1] + res.e3[1:])          # 線段兩端誤差的平均

        norm = Normalize(vmin=float(res.e3.min()), vmax=float(res.e3.max()))
        lc = LineCollection(segs, cmap=CMAP_ERR, norm=norm, linewidths=3.2,
                            capstyle="round", zorder=3)
        lc.set_array(mag)
        ax.add_collection(lc)

        cb = self.fig.colorbar(lc, ax=ax, pad=0.02, fraction=0.045)
        cb.set_label("3D 誤差(公尺)", color=theme.INK_2, fontsize=9)
        cb.ax.tick_params(colors=theme.MUTED, labelsize=8.5)
        cb.outline.set_edgecolor(theme.BASELINE)
        for v, lbl in ((res.stats["3d"]["p95"], "P95"), (res.stats["3d"]["rmse"], "RMSE")):
            cb.ax.axhline(v, color=theme.SURFACE, lw=2.4)
            cb.ax.axhline(v, color=theme.S_WARN if lbl == "P95" else theme.INK_2, lw=1.2)

        k = int(np.argmax(res.e3))
        ax.plot(p[k, 0], p[k, 1], "o", ms=13, mfc="none",
                mec=theme.S_WARN, mew=2.2, zorder=5)
        ax.annotate(f"最大 {res.e3[k]:.2f} m", xy=(p[k, 0], p[k, 1]),
                    xytext=(14, 12), textcoords="offset points",
                    color=theme.S_WARN, fontsize=9.5, fontweight="bold", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=theme.S_WARN, lw=1.0))
        _endpoints(ax, p[:, 0], p[:, 1], theme.MUTED)
        ax.annotate("○ 起點  ■ 終點", xy=(0.01, 0.985), xycoords="axes fraction",
                    va="top", color=theme.MUTED, fontsize=8.5)

        self._equal_xy(ax, p[:, 0], p[:, 1])
        ax.set_xlabel("東 E(公尺)")
        ax.set_ylabel("北 N(公尺)")
        ax.set_title(f"誤差熱點 — 顏色越亮誤差越大(畫在 {res.ref_name} 軌跡上)")
        _style_axes(ax)

        # 播放游標:參考軌跡上的目前位置(深色外暈 + 白圈,壓在亮色段上也看得到)
        self._trk = Track(res.t, p, _res_gap(res), edge=_res_gap(res) / 2)
        self._cur = [self._animated(ax.plot([], [], "o", ms=15, mfc="none", mec=theme.SURFACE,
                                            mew=4.2, zorder=8)[0]),
                     self._animated(ax.plot([], [], "o", ms=15, mfc="none", mec=theme.INK,
                                            mew=2.0, zorder=9)[0])]
        self.fig.tight_layout()

    def update_cursor(self, t):
        p = None if t is None else self._trk.at(t)
        for ln in self._cur:
            _set_point(ln, p)
        return False


class Plot3D(PlotWidget):
    """3D 軌跡。"""

    def draw_result(self, res, logdata):
        ax = self.fig.add_subplot(111, projection="3d")
        ax.set_facecolor(theme.SURFACE)
        r, e = res.ref_enu, res.est_enu
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color=theme.C_REF, lw=1.8,
                label=f"參考:{res.ref_name}")
        ax.plot(e[:, 0], e[:, 1], e[:, 2], color=theme.C_EST, lw=1.8,
                label=f"估測:{res.est_name}")
        ax.scatter(r[0, 0], r[0, 1], r[0, 2], s=55, facecolors="none",
                   edgecolors=theme.C_REF, linewidths=2)
        ax.scatter(e[0, 0], e[0, 1], e[0, 2], s=55, facecolors="none",
                   edgecolors=theme.C_EST, linewidths=2)

        # 播放游標:3D 中的參考 / 估測目前位置(空資料不影響自動範圍)
        gap = _res_gap(res)
        self._trk = (Track(res.t, r, gap, edge=gap / 2), Track(res.t, e, gap, edge=gap / 2))
        self._cur = (
            self._animated(ax.plot([], [], [], "o", ms=9, mfc=theme.C_REF,
                                   mec=theme.INK, mew=1.5, zorder=9)[0]),
            self._animated(ax.plot([], [], [], "D", ms=8, mfc=theme.C_EST,
                                   mec=theme.INK, mew=1.5, zorder=10)[0]))

        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0.102, 0.102, 0.098, 1.0))
            pane._axinfo["grid"]["color"] = theme.GRID
            pane._axinfo["grid"]["linewidth"] = 0.7
        ax.set_xlabel("東 E(m)")
        ax.set_ylabel("北 N(m)")
        ax.set_zlabel("天 U(m)")
        ax.set_title("3D 軌跡對照")
        # 等比例:XY 同尺度,Z 依實際範圍(高度變化通常遠小於水平)
        allp = np.vstack([r, e])
        rng = allp.max(axis=0) - allp.min(axis=0)
        xy = max(rng[0], rng[1], 1.0)
        ax.set_box_aspect((1, 1, max(0.25, min(1.0, rng[2] / xy * 3))))
        ax.legend(loc="upper left")
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04)

    def update_cursor(self, t):
        for trk, ln in zip(self._trk, self._cur):
            p = None if t is None else trk.at(t)
            if p is None:
                ln.set_visible(False)
            else:
                ln.set_data_3d([p[0]], [p[1]], [p[2]])
                ln.set_visible(True)
        return False


class ErrorTimePlot(PlotWidget):
    """誤差-時間序列(3D / 水平)+ P95 參考線。"""

    def draw_result(self, res, logdata):
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(theme.SURFACE)
        t = res.t - logdata.t_base
        ax.fill_between(t, 0, res.e3, color=theme.C_ERR, alpha=0.14, lw=0)
        ax.plot(t, res.e3, color=theme.C_ERR, lw=1.8, label="3D 誤差 ‖Δ‖")
        ax.plot(t, res.e2, color=theme.C_ERR2, lw=1.5, label="水平誤差 ‖Δ_EN‖")

        # 兩條參考線的標籤貼在同一側、用軸座標定位,才不會被軸邊界切掉
        for val, lbl, col, ls in ((res.stats["3d"]["p95"], "3D P95", theme.S_WARN, "--"),
                                  (res.stats["3d"]["rmse"], "3D RMSE", theme.MUTED, ":")):
            ax.axhline(val, color=col, lw=1.3, ls=ls, zorder=2)
            ax.annotate(f"{lbl} = {val:.2f} m", xy=(0.995, val),
                        xycoords=("axes fraction", "data"),
                        xytext=(0, 5), textcoords="offset points",
                        ha="right", color=col, fontsize=9, fontweight="bold", zorder=6,
                        bbox=_ANNOT_BOX)

        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, max(res.e3.max() * 1.12, 1e-3))
        ax.set_xlabel("時間(秒,相對 log 起點)")
        ax.set_ylabel("誤差(公尺)")
        ax.set_title(f"誤差隨時間變化 — {res.est_name} 相對 {res.ref_name}")
        _style_axes(ax)
        ax.legend(loc="upper left", ncols=2)

        # 播放游標:垂直播放線 + 3D 誤差曲線上的點 + 數值標籤
        self.t_base = logdata.t_base
        self._rng = (float(res.t[0]), float(res.t[-1]))
        self._trk = Track(res.t, res.e3[:, None], _res_gap(res), edge=_res_gap(res) / 2)
        self._cur_line = self._animated(ax.plot([], [], color=theme.INK, lw=1.2, zorder=8,
                                                transform=ax.get_xaxis_transform())[0])
        self._cur_dot = self._animated(ax.plot([], [], "o", ms=7, mfc=theme.C_ERR,
                                               mec=theme.INK, mew=1.4, zorder=9)[0])
        self._cur_lbl = self._animated(ax.annotate(
            "", xy=(t[0], 0.0), xytext=(8, 8), textcoords="offset points",
            color=theme.INK, fontsize=9, fontweight="bold", zorder=10, bbox=_ANNOT_BOX))
        self.fig.tight_layout()

    def update_cursor(self, t):
        inside = _in_range(t, *self._rng)
        x = t - self.t_base if inside else None
        _set_vline(self._cur_line, x)
        v = self._trk.at(t) if inside else None
        if v is None:                       # 範圍外或資料空洞:只留播放線
            self._cur_dot.set_visible(False)
            self._cur_lbl.set_visible(False)
            return False
        y = float(v[0])
        _set_point(self._cur_dot, (x, y))
        # 貼近右緣 / 上緣時標籤翻到另一側,不被軸邊界切掉
        ax = self._cur_dot.axes
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        right = x1 != x0 and (x - x0) / (x1 - x0) > 0.85
        top = y1 != y0 and (y - y0) / (y1 - y0) > 0.8
        lbl = self._cur_lbl
        lbl.xy = (x, y)
        lbl.xyann = (-8 if right else 8, -8 if top else 8)
        lbl.set_ha("right" if right else "left")
        lbl.set_va("top" if top else "bottom")
        lbl.set_text(f"{y:.2f} m")
        lbl.set_visible(True)
        return False


def _moving_average(t: np.ndarray, v: np.ndarray, window_s: float) -> np.ndarray:
    """等時距移動平均(以樣本數近似,取樣率固定時等價)。"""
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    n = max(3, int(round(window_s / max(dt, 1e-6)))) | 1
    if n >= v.size:
        return np.full_like(v, float(v.mean()))
    pad = n // 2
    vp = np.pad(v, (pad, pad), mode="edge")
    return np.convolve(vp, np.ones(n) / n, mode="valid")[:v.size]


class AxisErrorPlot(PlotWidget):
    """軸向誤差 small multiples(東/北/天),共用時間軸與 y 範圍。

    疊一條移動平均趨勢線:對齊後各軸平均誤差恆為 0(質心已對齊),
    畫平均值沒有資訊量;低頻趨勢才看得出 VIO 往哪個方向慢慢漂。
    """

    def draw_result(self, res, logdata):
        t = res.t - logdata.t_base
        win = max(5.0, res.duration / 20.0)
        lim = float(np.abs(res.err).max()) * 1.12 or 1.0
        axes = self.fig.subplots(3, 1, sharex=True, sharey=True)

        for ax, (zh, k) in zip(axes, [("東 E", 0), ("北 N", 1), ("天 U", 2)]):
            ax.set_facecolor(theme.SURFACE)
            v = res.err[:, k]
            ax.fill_between(t, 0, v, color=theme.C_ERR, alpha=0.16, lw=0)
            ax.plot(t, v, color=theme.C_ERR, lw=1.1, alpha=0.75)
            ax.plot(t, _moving_average(t, v, win), color=theme.S_WARN, lw=2.0)
            ax.axhline(0, color=theme.BASELINE, lw=1.2)
            st = res.stats["axes"]["ENU"[k]]
            ax.set_title(f"{zh} 軸 — RMSE {st['rmse']:.3f} m,P95 {st['p95']:.3f} m,"
                         f"最大 {st['max']:.3f} m", loc="left", fontsize=9.5)
            ax.set_ylim(-lim, lim)
            _style_axes(ax)

        axes[0].annotate(f"粗線 = {win:.0f} 秒移動平均(低頻漂移趨勢)",
                         xy=(0.995, 0.88), xycoords="axes fraction",
                         ha="right", color=theme.S_WARN, fontsize=8.5)
        axes[1].set_ylabel("誤差(公尺)")
        axes[-1].set_xlabel("時間(秒,相對 log 起點)")
        axes[-1].set_xlim(t[0], t[-1])
        self.fig.suptitle(f"各軸誤差(估測 {MINUS} 參考)— 看得出是哪個方向在偏",
                          color=theme.INK, fontsize=11, fontweight="bold")
        self.fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.09, hspace=0.42)

        # 播放游標:三軸同一條垂直播放線
        self.t_base = logdata.t_base
        self._rng = (float(res.t[0]), float(res.t[-1]))
        self._cur = [self._animated(ax.plot([], [], color=theme.INK, lw=1.2, zorder=8,
                                            transform=ax.get_xaxis_transform())[0])
                     for ax in axes]

    def update_cursor(self, t):
        x = t - self.t_base if _in_range(t, *self._rng) else None
        for ln in self._cur:
            _set_vline(ln, x)
        return False


class DistributionPlot(PlotWidget):
    """誤差分佈:直方圖 + 累積分佈(CDF)。沒有時間軸,不畫播放游標。"""

    def draw_result(self, res, logdata):
        ax1, ax2 = self.fig.subplots(1, 2)
        for ax in (ax1, ax2):
            ax.set_facecolor(theme.SURFACE)

        e = res.e3
        st = res.stats["3d"]
        ax1.hist(e, bins=min(60, max(12, res.n // 25)), color=theme.C_ERR,
                 alpha=0.85, edgecolor=theme.SURFACE, linewidth=0.8)
        # 三條標記線的標籤錯開高度,避免中位數與 RMSE 靠近時字疊在一起
        # 三條標記線靠「線型 + 文字標籤」區分身分,不靠顏色
        marks = ((st["median"], "中位數", theme.MUTED, ":", 0.97),
                 (st["rmse"], "RMSE", theme.MUTED, "-", 0.88),
                 (st["p95"], "P95", theme.S_WARN, "--", 0.79))
        for val, lbl, col, ls, y in marks:
            ax1.axvline(val, color=col, lw=1.2, ls=ls)
            ax1.annotate(f"{lbl} {val:.2f} m", xy=(val, y), xycoords=("data", "axes fraction"),
                         xytext=(5, 0), textcoords="offset points",
                         va="center", ha="left", color=col, fontsize=8.5,
                         bbox=dict(boxstyle="round,pad=0.22", fc=theme.SURFACE, ec="none", alpha=0.85))
        ax1.set_xlabel("3D 誤差(公尺)")
        ax1.set_ylabel("樣本數")
        ax1.set_title("誤差分佈", loc="left")
        _style_axes(ax1)

        s = np.sort(e)
        cdf = np.arange(1, s.size + 1) / s.size * 100
        ax2.plot(s, cdf, color=theme.C_ERR, lw=2.0)
        ax2.fill_between(s, 0, cdf, color=theme.C_ERR, alpha=0.12, lw=0)
        for pct, col in ((50, theme.MUTED), (90, theme.INK_2), (95, theme.S_WARN)):
            val = float(np.percentile(e, pct))
            ax2.plot([s[0], val], [pct, pct], color=col, lw=1.1, ls="--")
            ax2.plot([val, val], [0, pct], color=col, lw=1.1, ls="--")
            ax2.annotate(f"P{pct} = {val:.2f} m", xy=(val, pct), xytext=(6, -11),
                         textcoords="offset points", color=col, fontsize=8.5)
        ax2.set_xlim(s[0], s[-1])
        ax2.set_ylim(0, 100)
        ax2.set_xlabel("3D 誤差(公尺)")
        ax2.set_ylabel("累積百分比(%)")
        ax2.set_title("累積分佈 — 有幾成樣本落在誤差之內", loc="left")
        _style_axes(ax2)
        self.fig.tight_layout()


class OverviewPlot(PlotWidget):
    """載入後、計算前的總覽:每個來源一張小平面圖(small multiples)+ 時間涵蓋。

    用小倍數而非疊圖:真實 log 常有 5~6 個位置來源,再怎麼配色都無法讓那麼多條
    線彼此可辨(尤其對色覺缺陷讀者);每張小圖單一序列、標題命名身分,
    數量再多也不會失效,而且各來源的形狀看得更清楚。
    """

    def draw_logdata(self, logdata, origin):
        self._redraw(self._draw, logdata, origin)

    def _draw(self, logdata, origin):
        names = sorted(logdata.sources)
        drawable = []
        for name in names:
            try:
                drawable.append((name, logdata.sources[name], logdata.sources[name].enu(origin)))
            except ValueError:
                pass
        if not drawable:
            raise ValueError("沒有可繪製的來源(缺少 ENU 原點)")

        ncols = min(3, len(drawable))
        nrows = int(np.ceil(len(drawable) / ncols))
        gs = self.fig.add_gridspec(nrows + 1, ncols,
                                   height_ratios=[3] * nrows + [1.5],
                                   hspace=0.55, wspace=0.3)

        self.t_base = tb = logdata.t_base
        self._marks = []                    # (Track, 標記):各來源在播放頭時刻的位置
        for i, (name, src, p) in enumerate(drawable):
            ax = self.fig.add_subplot(gs[i // ncols, i % ncols])
            ax.set_facecolor(theme.SURFACE)
            ax.plot(p[:, 0], p[:, 1], color=theme.C_REF, lw=1.3)
            _endpoints(ax, p[:, 0], p[:, 1], theme.C_REF)
            self._equal_xy(ax, p[:, 0], p[:, 1])
            ax.set_title(f"{name} — {src.n} 筆 · {src.rate_hz:.0f} Hz",
                         loc="left", fontsize=9.5)
            ax.tick_params(labelsize=7.5)
            _style_axes(ax)
            if i // ncols == nrows - 1:
                ax.set_xlabel("東 E(m)", fontsize=8.5)
            if i % ncols == 0:
                ax.set_ylabel("北 N(m)", fontsize=8.5)
            self._marks.append((Track(src.t, p, _src_gap(src.t)),
                                self._animated(ax.plot([], [], "o", ms=8, mfc=theme.INK,
                                                       mec=theme.C_REF, mew=2.0, zorder=9)[0])))

        axt = self.fig.add_subplot(gs[nrows, :])
        axt.set_facecolor(theme.SURFACE)
        for i, (name, src, _) in enumerate(drawable):
            axt.barh(i, src.t[-1] - src.t[0], left=src.t[0] - tb,
                     height=0.55, color=theme.C_REF)
        axt.set_yticks(range(len(drawable)))
        axt.set_yticklabels([n for n, _, _ in drawable], fontsize=8.5)
        axt.invert_yaxis()
        span = max(logdata.duration, 1.0)
        title = "各來源的時間涵蓋 — 錯開或缺漏的區段不能拿來比較"

        # ARM 飛行段:綠底(狀態色)+「ARM n」文字標籤;標籤放在長條上方另留的空間
        segs = logdata.segments
        if segs:
            for s in segs:
                axt.axvspan(s.t0 - tb, s.t1 - tb, color=theme.S_GOOD, alpha=0.15,
                            lw=0, zorder=0.6)
            axt.set_ylim(len(drawable) - 0.5, -1.25)
            self._arm_labels(axt, segs, tb, span)
            title += f" · 綠底 = ARM 飛行段(共 {len(segs)} 段)"

        axt.set_xlabel("時間涵蓋(秒,相對 log 起點)")
        axt.set_xlim(0, span)
        axt.set_title(title, loc="left", fontsize=9.5)
        _style_axes(axt)
        axt.grid(axis="y", visible=False)
        self._span = span
        self._cur_line = self._animated(axt.plot([], [], color=theme.INK, lw=1.2, zorder=8,
                                                 transform=axt.get_xaxis_transform())[0])

        self.fig.suptitle(f"資料總覽 — {len(drawable)} 個位置來源",
                          color=theme.INK, fontsize=11, fontweight="bold")
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.08)

    def _arm_labels(self, axt, segs, tb, span):
        """在每段起點標「ARM n」;依目前畫布寬度估字寬,會跟前一個重疊的就略過。"""
        fs = 7.5
        ax_px = self.fig.get_figwidth() * self.fig.dpi * (0.98 - 0.06)
        last = -np.inf
        for s in segs:
            w = (len(s.label) * fs * 0.62 + 10) * self.fig.dpi / 72 / max(ax_px, 1.0) * span
            x = min(s.t0 - tb, span - w)
            if x < last + w:
                continue
            axt.annotate(s.label, xy=(max(x, 0.0), -0.82), xytext=(3, 0),
                         textcoords="offset points", va="center", ha="left",
                         color=theme.S_GOOD, fontsize=fs, fontweight="bold", zorder=3)
            last = x

    def update_cursor(self, t):
        for trk, mk in self._marks:
            _set_point(mk, None if t is None else trk.at(t))
        x = None if t is None else t - self.t_base
        _set_vline(self._cur_line, x if x is not None and -1e-6 <= x <= self._span + 1e-6 else None)
        return False


def _short_align(info: dict) -> str:
    """對齊方式短名:去掉結尾的說明括號,「剛體對齊 SE(3)(Umeyama,建議)」→「剛體對齊 SE(3)」。"""
    label = str(info.get("label") or info.get("method", ""))
    head = label.rsplit("(", 1)[0].strip() if label.endswith(")") else label
    return head or label


class PlaybackPlot(PlotWidget):
    """軌跡播放:整份 log 的平面軌跡 + 播放頭時刻的位置、尾巴與即時數值。

    一份 log 若涵蓋整次開機(長時間地面待機、多次起降、VIO 中途重置),
    拖曳時間軸就看得出每個時刻飛機在哪、VIO 與 GPS 差多少,再挑要分析的飛行段。
    沒有 ATE 結果也能播(估測畫原始座標);有結果時估測套用同一組對齊與時間偏移。
    """

    EXPORT_CURSOR = True    # 存下「此刻」的畫面:尾巴、位置與讀數都要在

    TAILS = (("5 s", 5.0), ("15 s", 15.0), ("30 s", 30.0), ("60 s", 60.0), ("全部", None))
    # 跟隨:飛機離開視野中央這個比例的方框(寬高各 50%)才重新置中、整張重畫;其餘只 blit 游標
    FOLLOW_BOX = 0.5

    def __init__(self, parent=None, toolbar: bool = True):
        self._ax = None
        self._home = None
        self._follow_keep = None
        self._center_next = False           # 下一次更新游標時把視野中心直接移到飛機
        self.ref_name = self.est_name = ""
        self.aligned = False
        super().__init__(parent, toolbar)

        bar = QWidget(self)
        bar.setObjectName("PlaybackOpts")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setStyleSheet(
            f"QWidget#PlaybackOpts{{background:{theme.PANEL};border-top:1px solid {theme.BORDER};}}"
            f"QLabel{{background:transparent;color:{theme.MUTED};font-size:9pt;}}"
            f"QCheckBox{{background:transparent;color:{theme.INK_2};font-size:9pt;}}"
            f"QComboBox{{padding:2px 8px;min-width:56px;font-size:9pt;}}")
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 3, 10, 4)
        h.setSpacing(8)
        h.addWidget(QLabel("軌跡尾巴"))
        self.cmb_tail = QComboBox()
        for txt, sec in self.TAILS:
            self.cmb_tail.addItem(txt, sec)
        self.cmb_tail.setCurrentIndex(1)
        self.cmb_tail.setToolTip("播放頭之前要畫多長的軌跡")
        h.addWidget(self.cmb_tail)
        h.addSpacing(12)
        self.chk_follow = QCheckBox("跟隨飛機")
        self.chk_follow.setToolTip("視野跟著目前位置移動(保持目前縮放;取消勾選回到全圖)")
        h.addWidget(self.chk_follow)
        h.addStretch(1)
        lay = self.layout()
        lay.insertWidget(lay.indexOf(self.canvas), bar)
        self.options_bar = bar

        self.cmb_tail.currentIndexChanged.connect(self._on_tail_changed)
        self.chk_follow.toggled.connect(self._on_follow_toggled)

    # ---------------- 繪製 ----------------

    def render(self, res, logdata):
        self.draw_playback(logdata, res.ref_name, res.est_name, res)

    def draw_result(self, res, logdata):
        self._draw_playback(logdata, res.ref_name, res.est_name, res)

    def draw_playback(self, logdata, ref_name: str, est_name: str, res=None) -> None:
        """畫整份 log 的兩條軌跡;res 給了且是同一組來源時,估測套用其對齊與時間偏移。"""
        # 身分先記下:頁面藏著時繪製會延後,這些狀態仍要反映新內容
        same = res is not None and (res.ref_name, res.est_name) == (ref_name, est_name)
        self.ref_name, self.est_name, self.aligned = ref_name, est_name, same
        self._redraw(self._draw_playback, logdata, ref_name, est_name, res)

    def _clear(self):
        # 跟隨中重畫(重算 ATE、換飛行段)要保留使用者的縮放,不能退回全圖;
        # fig.clear() 會把舊 axes 的範圍清掉,所以要在清之前記下。
        # 已經清過(佔位訊息、延後繪製中)時沿用先前記下的,連續清空不會把它弄丟
        if self._ax is not None:
            self._follow_keep = None
            if self.chk_follow.isChecked():
                self._follow_keep = (np.ptp(self._ax.get_xlim()) / 2,
                                     np.ptp(self._ax.get_ylim()) / 2)
        super()._clear()
        self._ax = self._home = None

    def _draw_playback(self, logdata, ref_name, est_name, res):
        keep, self._follow_keep = self._follow_keep, None
        self._ax = self._home = None
        ref = logdata.sources[ref_name]
        est = logdata.sources[est_name]
        if res is not None and (res.ref_name, res.est_name) != (ref_name, est_name):
            res = None                      # 結果屬於別組來源,不能套用
        origin = pipeline.resolve_origin(logdata, ref, est)[0]
        rp = ref.enu(origin)
        ep = est.enu(origin)
        rt = np.asarray(ref.t, dtype=float)
        et = np.asarray(est.t, dtype=float)
        off = 0.0
        if res is not None:
            ep = align_mod.apply_alignment(ep, res.align_info)
            off = float(res.cfg.get("offset_s", 0.0))
            et = et + off
        self.ref_name, self.est_name, self.aligned = ref_name, est_name, res is not None

        # 播放頭查詢用的資料(全部預先算好,update_cursor 只做查表)
        self.t_base = logdata.t_base
        self._segs = logdata.segments
        self._modes = mode_changes(logdata.events)
        gr, ge = _src_gap(rt), _src_gap(et)
        self._trk_ref = Track(rt, rp, gr)
        self._trk_est = Track(et, ep, ge)
        self._brk_ref = _gap_broken(rt, rp, gr, self._resets(logdata, ref_name, 0.0))
        self._brk_est = _gap_broken(et, ep, ge, self._resets(logdata, est_name, off))
        ts, vs = horizontal_speed(rt, rp)
        if vs.size >= 3:
            vs = _moving_average(ts, vs, 1.0)          # 位置差分的速度很毛,平滑約 1 秒
        self._trk_spd = Track(ts, vs[:, None], gr) if vs.size else None
        self._trk_err = None
        if res is not None:
            gap = _res_gap(res)
            self._trk_err = Track(res.t, res.err, gap, edge=gap / 2)
        # 分析區間(log 絕對秒):區分「播放頭在區間外」與「區間內但此刻沒有配對樣本」
        self._res_win = None
        if res is not None:
            lo = res.cfg.get("t0")
            hi = res.cfg.get("t1")
            self._res_win = (float(res.t[0] if lo is None else lo),
                             float(res.t[-1] if hi is None else hi))

        ax = self.fig.add_subplot(111)
        ax.set_facecolor(theme.SURFACE)
        self._ax = ax
        (tr, pr, _), (_, pe, _) = self._brk_ref, self._brk_est
        # 淡線只是背景脈絡:抽稀到幾千點(長 log 原解析度數十萬點,跟隨時每次重畫要上百 ms)
        for p, col in ((pr, theme.C_REF), (pe, theme.C_EST)):
            d = _decimate_path(p)
            ax.plot(d[:, 0], d[:, 1], color=col, lw=1.0, alpha=0.22, zorder=2)
        handles = [
            Line2D([], [], color=theme.C_REF, lw=2.4, marker="o", ms=7, mfc=theme.C_REF,
                   mec=theme.INK, mew=1.2, label=f"參考:{ref_name}"),
            Line2D([], [], color=theme.C_EST, lw=2.4, marker="D", ms=6, mfc=theme.C_EST,
                   mec=theme.INK, mew=1.2, label=f"估測:{est_name}")]
        if res is not None:
            # 參考軌跡在 ATE 分析區間內的部分畫亮一點:看得出分析的是哪一段
            a = int(np.searchsorted(tr, res.t[0], side="left"))
            b = int(np.searchsorted(tr, res.t[-1], side="right"))
            d = _decimate_path(pr[a:b])
            ax.plot(d[:, 0], d[:, 1], color=theme.C_REF, lw=1.3, alpha=0.45, zorder=3)
            handles.append(Line2D([], [], color=theme.C_REF, lw=1.3, alpha=0.45,
                                  label="ATE 分析區間(參考)"))

        self._equal_xy(ax, np.concatenate([rp[:, 0], ep[:, 0]]),
                       np.concatenate([rp[:, 1], ep[:, 1]]))
        # 上方多留一條空間給左上的數值框與右上的圖例,全圖時不壓到軌跡
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.2 * (y1 - y0))
        ax.set_xlabel("東 E(公尺)")
        ax.set_ylabel("北 N(公尺)")
        if res is None:
            note = "(估測未對齊 — 計算 ATE 後套用對齊)"
        elif res.align_info.get("method") == "none":
            note = "(ATE 設定為不對齊,估測照原始座標)"
        else:
            note = f"(估測已套用 {_short_align(res.align_info)} 對齊"
            note += f"、時間偏移 {off * 1000:+.0f} ms)" if abs(off) >= 5e-4 else ")"
        ax.set_title(f"軌跡播放 — {ref_name} vs {est_name}{note}")
        _style_axes(ax)
        # 圖例放右下:左上是數值框,圖窄時放右上會被壓到
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.06),
                  labelspacing=0.35, handlelength=1.6)
        ax.annotate("淡線 = 整份 log · 粗線 = 播放頭前的軌跡尾巴 · 白線 = 此刻兩者的差距",
                    xy=(0.01, 0.012), xycoords="axes fraction", va="bottom",
                    color=theme.MUTED, fontsize=8.5)

        # 播放游標:尾巴、目前位置、誤差連線、即時數值框
        self._trail_ref = self._animated(ax.plot([], [], color=theme.C_REF, lw=2.4, zorder=6,
                                                 solid_capstyle="round",
                                                 solid_joinstyle="round")[0])
        self._trail_est = self._animated(ax.plot([], [], color=theme.C_EST, lw=2.4, zorder=6,
                                                 solid_capstyle="round",
                                                 solid_joinstyle="round")[0])
        self._cur = self._pair_cursor(ax, z=7.0)
        self._hud = self._animated(ax.text(
            0.015, 0.985, "", transform=ax.transAxes, ha="left", va="top",
            color=theme.INK_2, fontsize=9, linespacing=1.55, zorder=12,
            bbox=dict(boxstyle="round,pad=0.5", fc=theme.SURFACE, alpha=0.85, ec=theme.BORDER)))
        self.fig.tight_layout()

        self._home = (ax.get_xlim(), ax.get_ylim())
        if self.toolbar is not None:        # 工具列「首頁」= 全圖(不是跟隨中的視野)
            self.toolbar.update()
            self.toolbar.push_current()
        if self.chk_follow.isChecked():
            self._apply_follow_zoom(keep)
            self._center_next = True        # 重畫後視野中心直接對準飛機

    @staticmethod
    def _resets(logdata, name: str, off: float) -> list[float]:
        return [e.t + off for e in logdata.events if e.kind == EV_RESET and e.label == name]

    # ---------------- 游標 ----------------

    def update_cursor(self, t):
        if t is None:
            for a in self._anim:
                a.set_visible(False)
            return False
        tail = self.cmb_tail.currentData()
        t_from = -np.inf if tail is None else t - float(tail)
        r = self._trk_ref.at(t)
        e = self._trk_est.at(t)
        _set_trail(self._trail_ref, self._brk_ref, t_from, t, r)
        _set_trail(self._trail_est, self._brk_est, t_from, t, e)
        _show_pair(self._cur, r, e)
        self._hud.set_text(self._hud_text(t, r, e))
        self._hud.set_visible(True)
        if self.chk_follow.isChecked():
            c = r if r is not None else e
            if c is not None:
                force, self._center_next = self._center_next, False
                return self._recenter(float(c[0]), float(c[1]), force)
        return False

    def _hud_text(self, t: float, r, e) -> str:
        rel = t - self.t_base
        head = f"t = {format_clock(rel)}({rel:.1f} s)"
        if self._segs:
            i = segment_index_at(self._segs, t)
            head += f" · {self._segs[i].label}" if i is not None else " · 地面(未 ARM)"
        if self._modes:
            m = mode_at(self._modes, t)
            if m:
                head += f" · {m}"

        if r is None:
            ref_line = f"參考 {self.ref_name} —(此刻無資料)"
        else:
            ref_line = f"參考 高度 {r[2]:.1f} m"
            v = self._trk_spd.at(t) if self._trk_spd is not None else None
            if v is not None:
                ref_line += f" · 水平速度 {float(v[0]):.1f} m/s"
        if e is None:
            ref_line += f" · 估測 {self.est_name} 此刻無資料"

        if self._trk_err is None:
            err_line = "誤差 —(尚未計算 ATE)"
        else:
            d = self._trk_err.at(t)
            if d is None:
                w = self._res_win
                inside = w is not None and w[0] <= t <= w[1]
                err_line = "誤差 —(此刻無配對樣本)" if inside else "誤差 —(不在分析區間)"
            else:
                err_line = (f"誤差 3D {float(np.linalg.norm(d)):.2f} m · "
                            f"水平 {float(np.linalg.norm(d[:2])):.2f} m")
        return "\n".join((head, ref_line, err_line))

    def _recenter(self, x: float, y: float, force: bool = False) -> bool:
        """跟隨:(x, y) 離開視野中央 FOLLOW_BOX 方框時把視野中心移過去(保持範圍);
        force 為真時一律置中。有移動才回傳 True(要整張重畫),否則游標只 blit。"""
        ax = self._ax
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
        dx, dy = abs(x0 + hx - x), abs(y0 + hy - y)
        if not force and dx <= self.FOLLOW_BOX * abs(hx) and dy <= self.FOLLOW_BOX * abs(hy):
            return False
        if dx <= 1e-4 * abs(hx) and dy <= 1e-4 * abs(hy):
            return False
        ax.set_xlim(x - hx, x + hx)
        ax.set_ylim(y - hy, y + hy)
        return True

    def _apply_follow_zoom(self, keep):
        """跟隨用的視野:keep = 先前的半寬 / 半高;沒有就在全圖時拉近(否則跟隨看不出差別)。"""
        ax = self._ax
        x0, x1 = ax.get_xlim()
        cx, cy = (x0 + x1) / 2, sum(ax.get_ylim()) / 2
        if keep is not None:
            hx, hy = keep
        else:
            full = self._home[0][1] - self._home[0][0]
            if x1 - x0 < 0.9 * full:        # 使用者自己縮放過就保留
                return
            hx = hy = min((x1 - x0) / 2, max(10.0, full / 6))
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)

    def _on_tail_changed(self, _index: int):
        if self._t is not None:
            self.set_time(self._t)

    def _on_follow_toggled(self, on: bool):
        if self._ax is None or self._home is None or not self._anim:
            return
        ax = self._ax
        (hx0, hx1), (hy0, hy1) = self._home
        if on:
            self._apply_follow_zoom(None)
            self._center_next = True        # 一勾就對準飛機,不等它離開中央方框
        else:
            ax.set_xlim(hx0, hx1)
            ax.set_ylim(hy0, hy1)
        try:
            if self._t is not None:
                self.update_cursor(self._t)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._drop_cursors()
        self._request_draw()
