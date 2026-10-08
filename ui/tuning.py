"""PID 調參建議面板(ArduCopter / QuadPlane 的姿態控制器)。

只依賴 core.tuning_model 的資料模型(TuningReport 由 core.tuning 產生)。版面:

  上方列   標題、分析範圍、資料品質標籤(取樣率 / PID 記錄 / FFT 批次)、整體結論、「重新分析」
  安全須知 可收合(一次只改一個參數、每次 10–25 %、先低空試飛、建議 AutoTune)
  三軸卡片 Roll / Pitch / Yaw:結論 + RMS 誤差、延遲、過衝、震盪、頻寬、最大誤差;點卡片選軸
  左:診斷  依嚴重度分組的發現,每項附調整方向;雙擊或「跳到」→ seekRequested(t_focus)
  右:分頁  建議調整表(複製 TSV / 匯出 .param / 複製報告)、追蹤、頻譜、頻率響應、步階響應

圖表都繼承 ui.plots.PlotWidget:追蹤圖有 blit 播放游標(set_time 很便宜),
藏在別的分頁時延後繪製。期望 = C_REF 藍、實際 = C_EST 橘、誤差 = C_ERR 洋紅;
狀態色(良好 / 注意 / 嚴重)一律配文字。
"""
from __future__ import annotations

import html
import math
import warnings

import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, Locator, MaxNLocator, NullFormatter
from PyQt6.QtCore import QPointF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QAbstractTextDocumentLayout, QColor, QFont, QGuiApplication, QPalette,
                         QTextDocument, QTextOption)
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QFileDialog, QFrame,
                             QGridLayout, QHBoxLayout, QHeaderView, QLabel, QProgressBar, QPushButton,
                             QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QStyle,
                             QStyledItemDelegate, QStyleOptionViewItem, QTableWidget,
                             QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from core.timeline import Track, format_clock
from core.tuning_model import (AXES, AXIS_LABEL, DIR_CHECK, DIR_DOWN, DIR_SET, DIR_UP,
                               SEV_BAD, SEV_INFO, SEV_LABEL, SEV_OK, SEV_ORDER, SEV_WARN,
                               AxisAnalysis, Finding, Suggestion, TuningReport)

from . import theme
from .plots import PlotWidget

# ---------------- 色彩與文字對照(狀態色永遠配文字) ----------------

C_INFO = "#86b6ef"     # 「可改善」:淡藍(不是狀態保留色,也不和圖表序列同色)
SEV_COLOR = {SEV_OK: theme.S_GOOD, SEV_INFO: C_INFO, SEV_WARN: theme.S_WARN, SEV_BAD: theme.S_CRIT}
SEV_GLYPH = {SEV_OK: "✓", SEV_INFO: "●", SEV_WARN: "▲", SEV_BAD: "■"}   # 形狀也不同(不只靠顏色)

DIR_TEXT = {DIR_UP: "↑ 調高", DIR_DOWN: "↓ 調低", DIR_SET: "→ 設定", DIR_CHECK: "◎ 檢查"}
DIR_COLOR = {DIR_UP: "#7bd389", DIR_DOWN: "#ff8fb1", DIR_SET: "#86b6ef", DIR_CHECK: "#d4d46a"}

AXIS_SHORT = {"roll": "Roll", "pitch": "Pitch", "yaw": "Yaw"}
LEVEL_TEXT = {"rate": "角速度", "angle": "角度"}

LOW_RATE_NYQ = 25.0    # Nyquist 低於此(取樣 < 50 Hz)= 看不到常見的 5–20 Hz PID 震盪
OSC_BAND = (5.0, 20.0)

EMPTY_MSG = "載入 log 後分析"
NO_AXIS_MSG = "這一軸沒有可用的資料"

SAFETY_HEAD = ("建議僅供參考:一次只改一個參數、每次 10–25 %,先在低空以 STABILIZE / ALT_HOLD 試飛;"
               "能跑 AutoTune 就先跑 AutoTune。")
SAFETY_MORE = [
    "改參數前先備份目前的 .param;改完變差就立刻改回。",
    "先處理振動、馬達飽和、機構鬆動與重心,再調 PID —— 馬達飽和或振動過大時不要調高增益。",
    "高頻(≥ 15 Hz)抖動先降 D;4–15 Hz 的擺動先降 P(I 跟著,D 相對 P 偏高時才先降 D);"
    "約 50 Hz 以上的窄峰是馬達噪音,用陷波處理。反應遲鈍才小幅調高 P(Roll / Pitch 的 P 與 I 通常保持相同)。",
    "每改一次就重新記錄、重新分析,比較前後差異;遠離人群,隨時準備切回手動並降落。",
    "低取樣率 log 只看得出慢的問題;要判斷震盪,請在 LOG_BITMASK 開 ATTITUDE_FAST(bit 0)"
    "與 PID(bit 12)(QuadPlane:ATTITUDE_FULLRATE bit 20),FFT 用 INS_LOG_BAT_MASK = 1。",
]

PAGE_TABLE, PAGE_TRACK, PAGE_SPEC, PAGE_FREQ, PAGE_STEP = range(5)

_ANNOT_BOX = dict(boxstyle="round,pad=0.28", fc=theme.SURFACE, ec="none", alpha=0.88)
MINUS = "-"            # U+2212 在多數中文字型缺字形


def _rgba(color: str, a: float) -> str:
    c = QColor(color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{a:.2f})"


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fmt_param(v) -> str:
    """參數值顯示:整數不帶小數(LOG_BITMASK 180223),其餘最多 4 位有效數字。"""
    v = _num(v)
    if v is None:
        return "—"
    if float(v).is_integer() and abs(v) < 1e9:
        return str(int(v))
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.4g}"


def fmt_pct(v) -> str:
    v = _num(v)
    if v is None:
        return "—"
    return f"{'+' if v > 0 else MINUS if v < 0 else ''}{abs(v):.0f} %"


def _unit(level: str) -> str:
    return "°/s" if level == "rate" else "°"


def _chip(text: str, color: str | None = None, tip: str = "", strong: bool = False) -> QLabel:
    """圓角小標籤;color = None 為中性灰。狀態色只會和文字一起出現。"""
    lab = QLabel(text)
    lab.setTextFormat(Qt.TextFormat.PlainText)
    if color is None:
        css = (f"color:{theme.INK_2};background:#2a2a28;border:1px solid {theme.BORDER};")
    else:
        css = (f"color:{theme.INK if strong else color};background:{_rgba(color, 0.30 if strong else 0.13)};"
               f"border:1px solid {_rgba(color, 0.75 if strong else 0.5)};")
    lab.setStyleSheet(f"QLabel{{{css}border-radius:9px;padding:1px 8px;font-size:9pt;"
                      f"{'font-weight:600;' if strong else ''}}}")
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lab.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    if tip:
        lab.setToolTip(tip)
    return lab


def sev_chip(sev: str) -> QLabel:
    return _chip(f"{SEV_GLYPH.get(sev, '●')} {SEV_LABEL.get(sev, sev)}", SEV_COLOR.get(sev, theme.MUTED))


def _hz(v, sep: str = " ") -> str:
    """取樣率文字(「10 Hz」);沒有值時為空字串。"""
    v = _num(v)
    return f"{sep}{v:g} Hz" if v else ""


def _clear_layout(lay) -> None:
    while lay.count():
        it = lay.takeAt(0)
        w = it.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        elif it.layout() is not None:
            _clear_layout(it.layout())


# ---------------- 報告 → 顯示用資料 ----------------

def axis_metrics(aa: AxisAnalysis | None) -> list[tuple[str, str, str, str]]:
    """卡片上的 6 個指標:(標籤, 數值, 單位, 說明)。缺資料時數值為「—」。"""
    if aa is None:
        return [(lab, "—", "", "") for lab in ("RMS 誤差", "延遲", "過衝", "震盪", "頻寬", "最大誤差")]
    m = aa.metrics or {}
    u = _unit(aa.level)

    def pick(*keys):
        for k in keys:
            v = _num(m.get(k))
            if v is not None:
                return v
        return None

    rms = pick("rms_err", "rms")
    lag = pick("lag_ms")
    over = pick("overshoot_pct")
    if over is None and aa.step is not None:
        over = _num(aa.step.overshoot_pct)
    bw = pick("bandwidth_hz")
    if bw is None and aa.freq is not None:
        bw = _num(aa.freq.bandwidth_hz)
    mx = pick("max_err")
    osc = pick("osc_hz", "osc_freq_hz", "peak_hz")
    # 分析引擎明確寫了 osc_hz = None(峰可能是噪音混疊 / 高於控制頻寬)時不要拿頻譜峰補,那是過度宣稱
    if osc is None and "osc_hz" not in m and aa.spectrum is not None and aa.spectrum.peaks:
        osc = _num(max(aa.spectrum.peaks, key=lambda p: p[1])[0])
    nyq = _nyquist(aa)
    low = nyq is not None and nyq < LOW_RATE_NYQ
    out = [
        ("RMS 誤差", "—" if rms is None else f"{rms:.1f}", u if rms is not None else "",
         "期望與實際的均方根差;越小代表跟得越緊"),
        ("延遲", "—" if lag is None else f"{lag:.0f}", "ms" if lag is not None else "",
         "實際值落後期望值的時間(互相關估計)"),
        ("過衝", "—" if over is None else f"{over:.0f}", "%" if over is not None else "",
         "步階響應超過目標的比例;> 15–20 % 通常代表 P 或 D 比例不當"),
    ]
    noise = pick("noise_hz")
    if osc is not None:
        out.append(("震盪", f"{osc:.1f}", "Hz", "誤差頻譜中最明顯的峰值頻率"))
    elif noise is not None:
        out.append(("震盪", "—", "", f"沒有控制震盪;{noise:.0f} Hz 的峰高於控制頻寬,是馬達 / 機架噪音(用陷波處理)"))
    elif low:
        out.append(("震盪", "—", f"(只見 ≤{nyq:g} Hz)",
                    f"取樣 {2 * nyq:g} Hz 只看得到 {nyq:g} Hz 以下;PID 震盪常見 5–20 Hz,這份 log 無法判斷"))
    else:
        out.append(("震盪", "—", "", "誤差頻譜中沒有明顯的峰值"))
    out += [
        ("頻寬", "—" if bw is None else f"{bw:.1f}", "Hz" if bw is not None else "",
         "|實際 / 期望| 掉到 -3 dB 的頻率;越高代表反應越快"),
        ("最大誤差", "—" if mx is None else f"{mx:.0f}", u if mx is not None else "",
         "分析範圍內最大的追蹤誤差"),
    ]
    return out


def _nyquist(aa: AxisAnalysis | None) -> float | None:
    if aa is None:
        return None
    if aa.spectrum is not None and _num(aa.spectrum.nyquist):
        return float(aa.spectrum.nyquist)
    hz = _num(aa.sample_hz)
    return hz / 2 if hz else None


def data_chips(report: TuningReport) -> list[tuple[str, str | None, str]]:
    """資料品質標籤 (文字, 顏色或 None, 說明);會限制判斷能力的用 S_WARN。"""
    d = report.data or {}
    out: list[tuple[str, str | None, str]] = []
    pid = d.get("PID")
    pid_hz = _num(d.get("PID_hz"))
    rate_hz = _num(d.get("RATE_hz"))
    if rate_hz is None:
        hz = [aa.sample_hz for aa in report.axes.values() if aa.level == "rate" and _num(aa.sample_hz)]
        rate_hz = max(hz) if hz else None
    bm = _num(report.params.get("LOG_BITMASK")) if report.params else None
    bm_tip = ""
    if report.param_prefix == "Q_A_":
        # Plane:bit 0 只有 25 Hz、bit 12 是 CAMERA;迴圈頻率的 ATT / PIQx 要 ATTITUDE_FULLRATE(bit 20)
        full = 1 << 20
        if bm is not None and not (int(bm) & full):
            bm_tip = f"\n建議 LOG_BITMASK = {int(bm) | full}(目前 {int(bm)},加上 ATTITUDE_FULLRATE bit 20)。"
        elif bm is None:
            bm_tip = "\n建議在 LOG_BITMASK 加上 ATTITUDE_FULLRATE(bit 20)。"
    elif bm is not None and (int(bm) & 4097) != 4097:
        bm_tip = f"\n建議 LOG_BITMASK = {int(bm) | 1 | 4096}(目前 {int(bm)},加上 ATTITUDE_FAST bit 0 與 PID bit 12)。"
    elif bm is None:
        bm_tip = "\n建議在 LOG_BITMASK 加上 ATTITUDE_FAST(bit 0)與 PID(bit 12)。"

    best_hz = pid_hz if (pid and pid_hz) else rate_hz
    name = "PID 記錄" if (pid and pid_hz) else "RATE"
    pid_tip = "\n有 PIDR / PIDP / PIDY:可看到 P / I / D / FF 各項輸出。" if pid else ""
    if best_hz is not None:
        low = best_hz / 2 < LOW_RATE_NYQ
        tip = (f"角速度資料取樣 {best_hz:g} Hz,Nyquist {best_hz / 2:g} Hz。"
               + (f"\nPID 震盪常見 5–20 Hz,這個取樣率看不到(可能混疊成假的低頻)。{bm_tip}" if low
                  else "\n足以看出常見的 PID 震盪。") + pid_tip)
        out.append((f"{name} {best_hz:g} Hz", theme.S_WARN if low else None, tip))
    else:
        att_hz = _num(d.get("ATT_hz"))
        if att_hz is not None:
            out.append((f"ATT {att_hz:g} Hz", theme.S_WARN if att_hz / 2 < LOW_RATE_NYQ else None,
                        f"沒有角速度資料,只用姿態角(ATT {att_hz:g} Hz)分析。{bm_tip}"))
    if pid is True and not pid_hz:
        out.append(("PID 記錄 ✓", None, "有 PIDR / PIDP / PIDY:可看到 P / I / D / FF 各項輸出。"))
    elif pid is False:
        why = ("沒有 PIQR / PIQP / PIQY(Plane 隨姿態記錄,要開 ATTITUDE_FULLRATE)" if report.param_prefix == "Q_A_"
               else "LOG_BITMASK 沒開 PID(bit 12)")
        out.append(("無 PID 記錄", theme.S_WARN,
                    why + ":看不到 P / I / D 各項輸出,無法分辨震盪來自 P 還是 D。" + bm_tip))
    batch = next((d[k] for k in ("batch", "FFT", "fft", "ISBH", "batch_fft") if k in d), None)
    if batch is True:
        out.append(("FFT 批次資料 ✓", None, "有 ISBH / ISBD 批次陀螺儀取樣,可看振動與陷波濾波頻率。"))
    elif batch is False:
        out.append(("無 FFT 批次資料", theme.S_WARN,
                    "沒有批次陀螺儀取樣(ISBH / ISBD):無法判斷振動頻率與陷波濾波器設定。"
                    "\n建議 INS_LOG_BAT_MASK = 1。"))
    frame = (report.frame or "").strip()
    veh = {"copter": "Copter", "plane": "Plane"}.get((report.vehicle or "").lower(), report.vehicle or "")
    if report.param_prefix == "Q_A_":
        veh = "QuadPlane"
    if veh or frame:
        out.append((" ".join(x for x in (veh, frame) if x), None,
                    f"機型 {veh} {frame};參數前綴 {report.param_prefix}"))
    return out


def chart_page_for(finding: Finding) -> int:
    """依診斷代碼挑最能說明問題的圖。"""
    c = (finding.code or "").lower()
    if any(k in c for k in ("osc", "noise", "vib", "notch", "filter", "fft", "resonan", "alias")):
        return PAGE_SPEC
    if any(k in c for k in ("overshoot", "step", "settle", "ring")):
        return PAGE_STEP
    if any(k in c for k in ("bandwidth", "freq", "bw")):
        return PAGE_FREQ
    return PAGE_TRACK


def build_report_text(report: TuningReport, t_base: float = 0.0, scope_text: str = "") -> str:
    """純文字報告(core.tuning.format_report_text 不在時的備用版)。"""
    L: list[str] = ["PID 調參建議報告", "=" * 24]
    L.append(scope_text or default_scope(report, t_base))
    L.append(f"整體評估:{SEV_LABEL.get(report.verdict, report.verdict)}")
    chips = data_chips(report)
    if chips:
        L.append("資料:" + "、".join(c[0] for c in chips))
    L.append("")
    for ax in AXES:
        aa = report.axes.get(ax)
        if aa is None:
            continue
        L.append(f"[{AXIS_LABEL[ax]}] {SEV_LABEL.get(aa.verdict, aa.verdict)}"
                 f"({LEVEL_TEXT.get(aa.level, aa.level)}{_hz(aa.sample_hz)})")
        if aa.summary:
            L.append(f"  {aa.summary}")
        L.append("  " + ",".join(f"{lab} {val}{(' ' + u) if u and not u.startswith('(') else u}"
                                 for lab, val, u, _ in axis_metrics(aa)))
    L.append("")
    L.append("診斷")
    if not report.findings:
        L.append("  (沒有發現問題)")
    for f in report.findings:
        ax = f"[{AXIS_SHORT.get(f.axis, f.axis)}] " if f.axis else ""
        when = f" @ {format_clock(f.t_focus - t_base, 1)}" if f.t_focus is not None else ""
        L.append(f"  [{SEV_LABEL.get(f.severity, f.severity)}] {ax}{f.title}{when}")
        if f.detail:
            L.append(f"      {f.detail}")
        for s in f.suggestions:
            L.append(f"      → {suggestion_line(s)}")
    sugg = report.suggestions()
    if sugg:
        L.append("")
        L.append("建議調整(依優先順序)")
        for i, s in enumerate(sugg, 1):
            L.append(f"  {i}. [P{s.priority}] {suggestion_line(s)} —— {s.reason}")
    if report.notes:
        L.append("")
        L.append("備註")
        L += [f"  - {n}" for n in report.notes]
    L.append("")
    L.append("安全須知:" + SAFETY_HEAD)
    L += [f"  - {x}" for x in SAFETY_MORE]
    return "\n".join(L) + "\n"


def suggestion_line(s: Suggestion) -> str:
    d = DIR_TEXT.get(s.direction, s.direction)
    if s.direction == DIR_CHECK:
        return f"{d} {s.param}"
    cur, new = fmt_param(s.current), fmt_param(s.proposed)
    pct = f"({fmt_pct(s.change_pct)})" if _num(s.change_pct) is not None else ""
    if s.proposed is None:
        return f"{d} {s.param}(目前 {cur})"
    return f"{d} {s.param} {cur} → {new}{pct}"


def default_scope(report: TuningReport, t_base: float = 0.0) -> str:
    return (f"分析範圍:{format_clock(report.t0 - t_base, 0)}–{format_clock(report.t1 - t_base, 0)}"
            f" · 飛行 {report.flight_s:.0f} s")


# =====================================================================
# 圖表
# =====================================================================

_MAX_PTS = 5000        # 追蹤圖每條線最多畫這麼多點(放大時重新取樣)


def _ffill_x(x: np.ndarray) -> np.ndarray:
    """NaN(斷點)以前一個有效值填補,供 searchsorted 使用(仍為非遞減)。"""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.all():
        return x
    idx = np.where(ok, np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    if not ok[0]:
        first = int(np.argmax(ok)) if ok.any() else x.size
        out[:first] = -np.inf
    return out


def _envelope(x: np.ndarray, y: np.ndarray, max_pts: int = _MAX_PTS):
    """點數超過 max_pts 時改畫每段的最小 / 最大值包絡(震盪不會被抽稀抽掉);斷點保留。"""
    n = x.size
    if n <= max_pts:
        return x, y
    k = int(math.ceil(n / (max_pts / 2)))
    m = n // k * k
    xb = x[:m].reshape(-1, k)
    yb = y[:m].reshape(-1, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanmin(yb, axis=1)
        hi = np.nanmax(yb, axis=1)
        xc = np.nanmean(xb, axis=1)
    brk = ~np.isfinite(xb).all(axis=1) | ~np.isfinite(yb).any(axis=1)
    lo[brk] = np.nan
    hi[brk] = np.nan
    xs = np.repeat(xc, 2)
    xs[np.repeat(brk, 2)] = np.nan
    ys = np.column_stack([lo, hi]).ravel()
    if m < n:
        xs = np.concatenate([xs, x[m:]])
        ys = np.concatenate([ys, y[m:]])
    return xs, ys


def _style(ax) -> None:
    ax.set_facecolor(theme.SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.BASELINE)
    ax.grid(True, color=theme.GRID, linewidth=0.8)


def _note(ax, text: str, color: str = theme.MUTED, loc: str = "lower right") -> None:
    x, ha = (0.99, "right") if "right" in loc else (0.01, "left")
    y, va = (0.03, "bottom") if "lower" in loc else (0.97, "top")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=color, fontsize=8.5,
            bbox=_ANNOT_BOX, zorder=7)


def _legend_top(ax, handles=None, labels=None) -> None:
    """圖例放在軸框上方右側(和靠左的標題同一列),不壓到資料。"""
    kw = dict(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=4, fontsize=8.5,
              borderaxespad=0.15, handlelength=1.5, columnspacing=1.1, frameon=False)
    if handles is not None:
        ax.legend(handles, labels, **kw)
    else:
        ax.legend(**kw)


_CLOCK_STEPS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)


class _ClockLocator(Locator):
    """時間刻度落在「時鐘上好讀」的間距(… 10 s、15 s、30 s、1 分、2 分、5 分 …),密度依軸寬。"""

    def __init__(self, px_per_tick: float = 90.0):
        self.px = px_per_tick

    def __call__(self):
        lo, hi = self.axis.get_view_interval()
        return self.tick_values(lo, hi)

    def tick_values(self, vmin, vmax):
        lo, hi = sorted((float(vmin), float(vmax)))
        try:
            width = float(self.axis.axes.bbox.width)
        except Exception:  # noqa: BLE001
            width = 700.0
        nmax = max(2, int(width / self.px))
        span = max(hi - lo, 1e-9)
        step = next((s for s in _CLOCK_STEPS if span / s <= nmax), None)
        if step is None:
            step = 3600.0 * math.ceil(span / (nmax * 3600.0))
        start = math.ceil(lo / step - 1e-9) * step
        return np.arange(start, hi + step * 1e-6, step)


def _clock_formatter(ax) -> FuncFormatter:
    def fmt(x, _pos=None):
        locs = ax.xaxis.get_majorticklocs()
        step = float(np.min(np.diff(locs))) if len(locs) > 1 else 1.0
        dec = 0 if step >= 1 - 1e-9 else 1 if step >= 0.1 - 1e-9 else 2
        return format_clock(round(float(x), 6), dec)
    return FuncFormatter(fmt)


def _set_vline(line, x) -> None:
    if x is None:
        line.set_visible(False)
        return
    line.set_data([x, x], [0.0, 1.0])
    line.set_visible(True)


def _set_point(line, p) -> None:
    if p is None or not np.isfinite(p[1]):
        line.set_visible(False)
        return
    line.set_data([p[0]], [p[1]])
    line.set_visible(True)


class _TuningPlot(PlotWidget):
    """調參圖表共用:沒有 ATE 提示文字、佔位訊息、選軸資訊。"""

    def __init__(self, parent=None, toolbar: bool = False):
        self._px = None                    # 版面邊界(像素),改變大小時重新換算成比例
        super().__init__(parent, toolbar=toolbar)
        self.axis_name: str | None = None
        self.canvas.mpl_connect("resize_event", self._apply_layout)
        self.show_placeholder(EMPTY_MSG)

    def _layout(self, left: int = 66, right: int = 14, top: int = 30, bottom: int = 46,
                gap: int = 0, nrows: int = 1) -> None:
        """以像素指定邊界(矮 / 窄的圖也不會把標題、刻度擠出去);gap = 子圖間距像素。"""
        self._px = (left, right, top, bottom, gap, nrows)
        self._apply_layout()

    def _apply_layout(self, *_):
        if self._px is None:
            return
        w, h = self.fig.get_size_inches() * self.fig.dpi
        if w < 120 or h < 120:
            return
        l, r, t, b, gap, n = self._px
        kw = dict(left=l / w, right=1 - r / w, top=1 - t / h, bottom=b / h)
        if n > 1:
            hh = h - t - b
            kw["hspace"] = max(gap * n / max(hh - (n - 1) * gap, 1.0), 0.05)
        try:
            self.fig.subplots_adjust(**kw)
        except ValueError:
            pass

    def _clear(self):
        self._px = None
        # twinx 共用的對數 x 軸在 fig.clear() 時會被重設成 (0, 1) 而發出無害的警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Attempt to set non-positive xlim")
            super()._clear()

    def show_axis(self, aa: AxisAnalysis | None, t_base: float = 0.0, **kw) -> None:
        self.t_base = float(t_base)
        self.axis_name = aa.axis if aa is not None else None
        if aa is None:
            self.show_placeholder(NO_AXIS_MSG)
            return
        self._redraw(self.draw_axis, aa, kw)

    def draw_axis(self, aa: AxisAnalysis, kw: dict) -> None:
        raise NotImplementedError

    def _empty(self, text: str, sub: str = "") -> None:
        """畫一個說明文字(資料不足時),仍走 _redraw 的流程。"""
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(theme.SURFACE)
        ax.axis("off")
        ax.text(0.5, 0.54, text, ha="center", va="center", color=theme.INK_2, fontsize=11,
                transform=ax.transAxes)
        if sub:
            ax.text(0.5, 0.44, sub, ha="center", va="center", color=theme.MUTED, fontsize=9,
                    transform=ax.transAxes, wrap=True)


class TrackingPlot(_TuningPlot):
    """追蹤:期望 vs 實際(角速度層級)、誤差、角度層級(有的話)。橫軸 m:ss(相對 t_base)。

    點一下圖面(非縮放 / 平移模式)→ seeked(log 絕對秒)。
    """

    seeked = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent, toolbar=True)
        self._lines: list = []             # (line, x 相對秒含 NaN, x 填補版, y)
        self._trk: Track | None = None
        self._trk_ang: Track | None = None
        self._rng: tuple[float, float] | None = None
        self._press = None
        self._vlines: list = []
        self._dots: list = []
        self._dots_ang: list = []
        self._lbl = None
        self._unit = ""
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _clear(self):
        super()._clear()
        self._lines = []
        self._trk = self._trk_ang = None
        self._rng = None
        self._vlines, self._dots, self._dots_ang = [], [], []
        self._lbl = None

    # ---------- 繪製 ----------

    def _add_line(self, ax, x, y, **kw):
        xf = _ffill_x(x)
        xs, ys = _envelope(x, y)
        ln, = ax.plot(xs, ys, **kw)
        self._lines.append((ln, x, xf, y))
        return ln

    def draw_axis(self, aa: AxisAnalysis, kw: dict) -> None:
        tb = self.t_base
        t = np.asarray(aa.t, dtype=float)
        des = np.asarray(aa.des, dtype=float)
        act = np.asarray(aa.act, dtype=float)
        if t.size < 2 or not np.isfinite(t).any():
            self._empty("這一軸的追蹤資料不足", "需要 RATE(或 PIDR / PIDP / PIDY)或 ATT 的期望與實際值")
            return
        has_ang = (aa.level == "rate" and aa.t_ang is not None and aa.des_ang is not None
                   and aa.act_ang is not None and np.size(aa.t_ang) >= 2)
        u = _unit(aa.level)
        self._unit = u
        lvl = LEVEL_TEXT.get(aa.level, aa.level)
        if has_ang:
            axs = self.fig.subplots(3, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.35, 2.1]))
        else:
            axs = self.fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.35]))
        ax_r, ax_e = axs[0], axs[1]
        x = t - tb
        self._add_line(ax_r, x, des, color=theme.C_REF, lw=1.5, label="期望", zorder=3)
        self._add_line(ax_r, x, act, color=theme.C_EST, lw=1.2, label="實際", zorder=4, alpha=0.95)
        ax_r.set_title(f"{lvl}追蹤 — {AXIS_LABEL.get(aa.axis, aa.axis)}{_hz(aa.sample_hz, ' · ')}",
                       loc="left", fontsize=10)
        ax_r.set_ylabel(f"{lvl}({u})")
        _legend_top(ax_r)

        err = act - des
        self._add_line(ax_e, x, err, color=theme.C_ERR, lw=1.1, zorder=3)
        ax_e.axhline(0, color=theme.BASELINE, lw=1.0, zorder=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rms = float(np.sqrt(np.nanmean(err ** 2))) if np.isfinite(err).any() else float("nan")
            lim = float(np.nanpercentile(np.abs(err), 99.5)) if np.isfinite(err).any() else 1.0
        lim = max(lim * 1.25, 1e-3)
        ax_e.set_ylim(-lim, lim)
        ax_e.set_title(f"誤差(實際 {MINUS} 期望)", loc="left", fontsize=9.5)
        ax_e.set_ylabel(u)
        if np.isfinite(rms):
            _note(ax_e, f"RMS {rms:.1f} {u}", theme.C_ERR, "upper right")

        ax_a = None
        if has_ang:
            ax_a = axs[2]
            ta = np.asarray(aa.t_ang, dtype=float) - tb
            self._add_line(ax_a, ta, np.asarray(aa.des_ang, dtype=float), color=theme.C_REF, lw=1.5,
                           zorder=3)
            self._add_line(ax_a, ta, np.asarray(aa.act_ang, dtype=float), color=theme.C_EST, lw=1.2,
                           zorder=4)
            ax_a.set_title("角度追蹤(ATT)— 外環", loc="left", fontsize=9.5)
            ax_a.set_ylabel("角度(°)")

        # 所選診斷的時間範圍
        hl = kw.get("highlight")
        if hl is not None:
            w = hl.window
            if w is None and hl.t_focus is not None:
                w = (hl.t_focus - 1.0, hl.t_focus + 1.0)
            if w is not None:
                for a in axs:
                    a.axvspan(w[0] - tb, w[1] - tb, color=theme.INK, alpha=0.07, lw=0, zorder=1)
                ax_r.annotate("所選診斷", xy=(0.5 * (w[0] + w[1]) - tb, 1.0),
                              xycoords=("data", "axes fraction"), xytext=(0, -2),
                              textcoords="offset points", ha="center", va="top",
                              color=theme.INK_2, fontsize=8.5, bbox=_ANNOT_BOX, zorder=7)
        for a in axs:
            _style(a)
            a.tick_params(labelsize=8)
        bottom = axs[-1]
        bottom.xaxis.set_major_locator(_ClockLocator())
        bottom.xaxis.set_major_formatter(_clock_formatter(bottom))
        bottom.set_xlabel("時間(m:ss,相對 log 起點)— 點圖面可跳到該時刻")
        xf = x[np.isfinite(x)]
        x0, x1 = float(xf.min()), float(xf.max())
        if x1 <= x0:
            x1 = x0 + 1.0
        bottom.set_xlim(x0, x1)
        ax_r.callbacks.connect("xlim_changed", self._on_xlim)
        self._on_xlim(ax_r)
        lo, hi = self._ylim(des, act)
        ax_r.set_ylim(lo, hi)
        if ax_a is not None:
            lo, hi = self._ylim(np.asarray(aa.des_ang, float), np.asarray(aa.act_ang, float))
            ax_a.set_ylim(lo, hi)
        self._layout(left=66, right=14, top=30, bottom=46, gap=28, nrows=len(axs))

        # 播放游標
        ok = np.isfinite(t)
        dt = float(np.median(np.diff(t[ok]))) if ok.sum() > 1 else 0.1
        self._trk = Track(t[ok], np.column_stack([des[ok], act[ok]]), max(3 * dt, 0.05), edge=dt)
        self._rng = (float(t[ok][0]), float(t[ok][-1]))
        if has_ang:
            ta_abs = np.asarray(aa.t_ang, dtype=float)
            oka = np.isfinite(ta_abs)
            dta = float(np.median(np.diff(ta_abs[oka]))) if oka.sum() > 1 else 0.1
            self._trk_ang = Track(ta_abs[oka], np.column_stack([np.asarray(aa.des_ang, float)[oka],
                                                                np.asarray(aa.act_ang, float)[oka]]),
                                  max(3 * dta, 0.05), edge=dta)
        for a in axs:
            self._vlines.append(self._animated(a.plot([], [], color=theme.INK, lw=1.1, zorder=8,
                                                      transform=a.get_xaxis_transform())[0]))
        self._dots = [self._animated(ax_r.plot([], [], "o", ms=6.5, mfc=c, mec=theme.INK, mew=1.2,
                                               zorder=9)[0]) for c in (theme.C_REF, theme.C_EST)]
        if ax_a is not None:
            self._dots_ang = [self._animated(ax_a.plot([], [], "o", ms=6, mfc=c, mec=theme.INK,
                                                       mew=1.2, zorder=9)[0])
                              for c in (theme.C_REF, theme.C_EST)]
        self._lbl = self._animated(ax_r.text(0.01, 0.97, "", transform=ax_r.transAxes, ha="left",
                                             va="top", color=theme.INK, fontsize=8.5,
                                             bbox=_ANNOT_BOX, zorder=10))

    @staticmethod
    def _ylim(*ys) -> tuple[float, float]:
        v = np.concatenate([np.asarray(y, float).ravel() for y in ys])
        v = v[np.isfinite(v)]
        if v.size == 0:
            return -1.0, 1.0
        lo, hi = np.percentile(v, [0.2, 99.8])
        pad = max((hi - lo) * 0.12, 0.5)
        return float(lo - pad), float(hi + pad * 1.6)      # 上方留給圖例 / 讀數

    def _on_xlim(self, ax) -> None:
        """視野改變:依可見範圍重新取樣(放大時看得到原始波形)。"""
        x0, x1 = ax.get_xlim()
        for ln, x, xf, y in self._lines:
            a = max(int(np.searchsorted(xf, x0, side="left")) - 1, 0)
            b = min(int(np.searchsorted(xf, x1, side="right")) + 1, x.size)
            xs, ys = _envelope(x[a:b], y[a:b])
            ln.set_data(xs, ys)

    # ---------- 游標 ----------

    def update_cursor(self, t):
        inside = t is not None and self._rng is not None and \
            self._rng[0] - 1e-6 <= t <= self._rng[1] + 1e-6
        x = t - self.t_base if inside else None
        for vl in self._vlines:
            _set_vline(vl, x)
        v = self._trk.at(t) if (inside and self._trk is not None) else None
        for i, d in enumerate(self._dots):
            _set_point(d, None if v is None else (x, v[i]))
        if self._dots_ang:
            va = self._trk_ang.at(t) if (inside and self._trk_ang is not None) else None
            for i, d in enumerate(self._dots_ang):
                _set_point(d, None if va is None else (x, va[i]))
        if self._lbl is not None:
            if v is None or not np.all(np.isfinite(v)):
                self._lbl.set_visible(False)
            else:
                u = self._unit
                self._lbl.set_text(f"{format_clock(x, 1)}  期望 {v[0]:.1f} · 實際 {v[1]:.1f} {u}"
                                   f" · 誤差 {v[1] - v[0]:+.1f}")
                self._lbl.set_visible(True)
        return False

    # ---------- 點擊 → 跳到該時刻 ----------

    def _on_press(self, ev):
        self._press = (ev.x, ev.y) if (ev.button == 1 and ev.inaxes is not None) else None

    def _on_release(self, ev):
        p, self._press = self._press, None
        if p is None or ev.button != 1 or ev.inaxes is None or ev.xdata is None or not self._lines:
            return
        if self.toolbar is not None and self.toolbar.mode != "":
            return                         # 縮放 / 平移模式
        if abs(ev.x - p[0]) > 4 or abs(ev.y - p[1]) > 4:
            return
        self.seeked.emit(float(ev.xdata) + self.t_base)


_HZ_TICKS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)


def _hz_log_axis(ax, lo: float, hi: float) -> None:
    """對數頻率軸;刻度 1-2-5 序列、ASCII 文字(中文字型缺 U+2212,不用 mathtext)。"""
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator([v for v in _HZ_TICKS if lo <= v <= hi * 1.0001]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _log_y_fmt(v, _pos=None) -> str:
    if not (v > 0):
        return ""
    e = math.log10(v)
    if abs(e - round(e)) > 1e-6:
        return ""
    e = int(round(e))
    return f"{v:g}" if -3 <= e <= 3 else f"1e{e}"


def _log_y_axis(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_log_y_fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())


class SpectrumPlot(_TuningPlot):
    """頻譜:誤差與實際值的 PSD(對數 y)、偵測到的峰值、Nyquist。

    高取樣率用對數頻率軸並淡淡標出常見 PID 震盪頻段;低取樣率(Nyquist < 25 Hz)用線性軸延伸到
    25 Hz,把 0.8·Nyquist 以上標成「無法判斷」—— 讓人一眼看出 5–20 Hz 落在盲區。
    """

    def __init__(self, parent=None):
        super().__init__(parent, toolbar=True)

    def draw_axis(self, aa: AxisAnalysis, kw: dict) -> None:
        sp = aa.spectrum
        if sp is None or np.size(sp.f) < 3:
            self._empty("這一軸資料不足,無法計算頻譜",
                        "需要至少數秒連續的飛行資料" + (f"(取樣{_hz(aa.sample_hz)})" if _hz(aa.sample_hz) else ""))
            return
        f = np.asarray(sp.f, dtype=float)
        nyq = float(sp.nyquist) if _num(sp.nyquist) else float(f.max())
        low = nyq < LOW_RATE_NYQ
        ax = self.fig.add_subplot(111)
        _style(ax)
        u2 = "(°/s)²/Hz" if aa.level == "rate" else "°²/Hz"

        def pos(y):
            y = np.asarray(y, dtype=float).copy()
            y[~(y > 0)] = np.nan
            return y

        m = f > 0
        f = f[m]
        pe, pa = pos(sp.psd_err)[m], pos(sp.psd_act)[m]
        ax.plot(f, pa, color=theme.C_EST, lw=1.4, label="實際", zorder=3)
        ax.plot(f, pe, color=theme.C_ERR, lw=1.6, label=f"誤差(實際 {MINUS} 期望)", zorder=4)
        _log_y_axis(ax)
        finite = np.concatenate([pe[np.isfinite(pe)], pa[np.isfinite(pa)]])
        if finite.size:
            ax.set_ylim(float(np.percentile(finite, 1)) / 3, float(finite.max()) * 40)
        if low:
            xmax = max(nyq, OSC_BAND[1] + 5.0)
            ax.set_xlim(0, xmax)
            x0 = 0.8 * nyq
            ax.axvspan(x0, xmax, color=theme.MUTED, alpha=0.13, lw=0, zorder=1, hatch="//")
            ax.text(0.5 * (nyq + xmax), 0.5,
                    f"無法判斷\n取樣 {2 * nyq:g} Hz 看不到\n{x0:g} Hz 以上的震盪\n\n"
                    f"PID 震盪常見 {OSC_BAND[0]:g}–{OSC_BAND[1]:g} Hz",
                    transform=ax.get_xaxis_transform(), ha="center", va="center",
                    color=theme.INK_2, fontsize=9.5, bbox=_ANNOT_BOX, zorder=6)
        else:
            lo = max(float(f[0]), 0.3) if f.size else 0.3
            _hz_log_axis(ax, lo, nyq)
            ax.axvspan(*OSC_BAND, color=theme.INK, alpha=0.04, lw=0, zorder=1)
            ax.text(math.sqrt(OSC_BAND[0] * OSC_BAND[1]), 0.985,
                    f"常見 PID 震盪 {OSC_BAND[0]:g}–{OSC_BAND[1]:g} Hz",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    color=theme.MUTED, fontsize=8, zorder=2)
        ax.axvline(nyq, color=theme.INK_2, lw=1.1, ls="--", zorder=5)
        ax.annotate(f"Nyquist {nyq:g} Hz", xy=(nyq, 0.97), xycoords=("data", "axes fraction"),
                    xytext=(-4, 0), textcoords="offset points", rotation=90, ha="right", va="top",
                    color=theme.INK_2, fontsize=8.5, zorder=6)
        # 峰值
        peaks = [p for p in sorted(sp.peaks or [], key=lambda p: -p[1])
                 if np.isfinite(p[0]) and p[0] > 0][:4]
        u = _unit(aa.level)
        fill = float(np.nanmin(pe)) if np.isfinite(pe).any() else 1e-6
        for i, (fp, amp) in enumerate(peaks):
            y = float(np.interp(fp, f, np.nan_to_num(pe, nan=fill))) * 1.8
            alias = low and fp >= 0.8 * nyq
            ax.plot([fp], [y], "v", ms=8, color=theme.INK, mec=theme.SURFACE, zorder=7)
            try:                                 # 靠右邊的峰把標籤放左邊(不要壓到 Nyquist 標示、被裁掉)
                xf = float(ax.transAxes.inverted().transform(ax.transData.transform((fp, y)))[0])
            except Exception:  # noqa: BLE001
                xf = 0.0
            right = xf > 0.7
            ax.annotate(f"{fp:.1f} Hz · {amp:.1f} {u} RMS" + ("\n(可能是混疊)" if alias else ""),
                        xy=(fp, y), xytext=(-10 if right else 6, 10 + 13 * (i % 2)), textcoords="offset points",
                        ha="right" if right else "left",
                        color=theme.INK, fontsize=8.5, bbox=_ANNOT_BOX, zorder=8)
        ax.set_xlabel("頻率(Hz)" + ("" if low else ",對數刻度"))
        ax.set_ylabel(f"功率譜密度 {u2}")
        title = f"頻譜 — {AXIS_LABEL.get(aa.axis, aa.axis)}"
        title += f" · 峰值 {len(peaks)} 個" if peaks else " · 沒有明顯峰值"
        ax.set_title(title, loc="left", fontsize=10)
        _legend_top(ax)
        if not peaks and not low:
            _note(ax, "誤差頻譜平滑、沒有尖峰 = 沒有持續的震盪", loc="lower left")
        self._layout(left=70, right=14, top=30, bottom=46)


class FreqRespPlot(_TuningPlot):
    """頻率響應 |H| = 實際 / 期望(dB)與相位;-3 dB 頻寬、共振峰;相干性淡線(低處估計不可靠)。"""

    def __init__(self, parent=None):
        super().__init__(parent, toolbar=True)

    def draw_axis(self, aa: AxisAnalysis, kw: dict) -> None:
        fr = aa.freq
        if fr is None or np.size(fr.f) < 3:
            self._empty("這一軸資料不足,無法估計頻率響應",
                        "需要足夠的操控輸入(期望值要有變化)才能估出 實際 / 期望")
            return
        f = np.asarray(fr.f, dtype=float)
        mag = np.asarray(fr.mag, dtype=float)
        ph = np.asarray(fr.phase_deg, dtype=float)
        coh = np.asarray(fr.coherence, dtype=float) if fr.coherence is not None else None
        m = f > 0
        f, mag, ph = f[m], mag[m], ph[m]
        coh = coh[m] if coh is not None and coh.size == m.size else None
        db = 20 * np.log10(np.maximum(mag, 1e-6))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2.2, 1])
        ax = self.fig.add_subplot(gs[0])
        axp = self.fig.add_subplot(gs[1])
        logx = f.size > 2 and f[-1] / max(f[0], 1e-9) > 8
        for a in (ax, axp):
            _style(a)
            if logx:
                _hz_log_axis(a, float(f[0]), float(f[-1]))
            else:
                a.set_xlim(float(f[0]), float(f[-1]))
        ax.tick_params(labelbottom=False)
        good = np.ones_like(f, dtype=bool) if coh is None else coh >= 0.5
        ax.plot(f, db, color=theme.C_EST, lw=1.0, alpha=0.25, zorder=3)
        ax.plot(f, np.where(good, db, np.nan), color=theme.C_EST, lw=1.9, zorder=4,
                label="|實際 / 期望|")
        ax.axhline(0, color=theme.BASELINE, lw=1.1, zorder=2)
        ax.axhline(-3, color=theme.INK_2, lw=1.0, ls="--", zorder=2)
        ax.annotate(f"{MINUS}3 dB", xy=(0.005, -3), xycoords=("axes fraction", "data"),
                    xytext=(2, -11), textcoords="offset points", color=theme.INK_2, fontsize=8.5)
        ok = good & np.isfinite(db)
        lo = float(np.min(db[ok])) if ok.any() else -12
        hi = float(np.max(db[ok])) if ok.any() else 3
        ax.set_ylim(max(min(lo, -6) - 3, -40), min(max(hi, 2), 30) + 5)
        bw = _num(fr.bandwidth_hz)
        if bw is not None:
            for a in (ax, axp):
                a.axvline(bw, color=theme.INK, lw=1.1, ls=":", zorder=5)
            ax.plot([bw], [-3], "o", ms=6, mfc=theme.INK, mec=theme.SURFACE, mew=1.2, zorder=7)
            ax.annotate(f"頻寬 {bw:.1f} Hz", xy=(bw, -3), xytext=(7, -7), textcoords="offset points",
                        va="top", color=theme.INK, fontsize=9, fontweight="bold", bbox=_ANNOT_BOX,
                        zorder=8)
        pk, pf = _num(fr.peak_mag), _num(fr.peak_hz)
        if pk is not None and pf is not None and pk > 1.05 and f.size:
            pdb = 20 * math.log10(pk)
            c_at = float(np.interp(pf, f, coh)) if coh is not None else 1.0
            trust = c_at >= 0.5
            col = theme.S_WARN if (pdb > 3 and trust) else theme.INK_2
            ax.plot([pf], [pdb], "o", ms=8, mfc="none", mec=col, mew=2, zorder=7)
            txt = f"共振峰 +{pdb:.1f} dB @ {pf:.1f} Hz"
            txt += ("(會過衝)" if pdb > 3 else "") if trust else "(相干性低,不可靠)"
            span = (math.log(f[-1] / f[0]) if logx else f[-1] - f[0]) or 1.0
            frac = (math.log(pf / f[0]) if logx else pf - f[0]) / span
            right = frac > 0.6
            ax.annotate(txt, xy=(pf, pdb), xytext=(-8 if right else 8, 8), textcoords="offset points",
                        ha="right" if right else "left", color=col, fontsize=9, fontweight="bold",
                        bbox=_ANNOT_BOX, zorder=8)
        if coh is not None:
            axc = ax.twinx()
            axc.plot(f, coh, color=theme.MUTED, lw=1.0, ls=":", alpha=0.9, zorder=2, label="相干性")
            axc.set_ylim(0, 1.6)
            axc.set_yticks([0, 0.5, 1.0])
            axc.tick_params(colors=theme.MUTED, labelsize=7.5)
            axc.set_ylabel("相干性", color=theme.MUTED, fontsize=8)
            axc.grid(False)
            for s in ("top", "left", "bottom"):
                axc.spines[s].set_visible(False)
            axc.spines["right"].set_color(theme.BASELINE)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = axc.get_legend_handles_labels()
            _legend_top(ax, h1 + h2, l1 + l2)
            if (~good).any():
                _note(ax, "淡色 = 相干性 < 0.5,該頻段估計不可靠", loc="lower left")
        else:
            _legend_top(ax)
        axp.plot(f, ph, color=theme.C_EST, lw=1.0, alpha=0.25, zorder=3)
        axp.plot(f, np.where(good, ph, np.nan), color=theme.C_EST, lw=1.5, zorder=4)
        for v in (-90, -180):
            axp.axhline(v, color=theme.BASELINE, lw=1.0, ls=":")
        okp = good & np.isfinite(ph)
        pmin = float(np.min(ph[okp])) if okp.any() else -180
        pmax = float(np.max(ph[okp])) if okp.any() else 0
        axp.set_ylim(max(min(pmin, -100), -400) - 15, min(max(pmax, 0), 90) + 20)
        axp.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 4.5, 9, 10]))
        axp.set_ylabel("相位(°)")
        axp.set_xlabel("頻率(Hz)" + (",對數刻度" if logx else ""))
        ax.set_ylabel("增益(dB)")
        t = f"頻率響應 — {AXIS_LABEL.get(aa.axis, aa.axis)}"
        if bw is not None:
            t += f" · 頻寬 {bw:.1f} Hz"
        ax.set_title(t, loc="left", fontsize=10)
        nyq = _nyquist(aa)
        if nyq is not None and nyq < LOW_RATE_NYQ:
            _note(axp, f"取樣 {2 * nyq:g} Hz:只估得到 {nyq:g} Hz 以下", theme.S_WARN, "lower left")
        self._layout(left=66, right=50, top=30, bottom=46, gap=10, nrows=2)


class StepPlot(_TuningPlot):
    """步階響應:y(t)、理想 1.0、±5 % 帶、過衝 / 上升時間 / 穩定時間標註。"""

    def draw_axis(self, aa: AxisAnalysis, kw: dict) -> None:
        st = aa.step
        if st is None or np.size(st.t) < 3:
            self._empty("這一軸資料不足,無法估計步階響應",
                        "需要多段有明顯操控輸入的時間窗")
            return
        t = np.asarray(st.t, dtype=float)
        y = np.asarray(st.y, dtype=float)
        ms = t[-1] < 3.0
        tx = t * 1000 if ms else t
        unit = "ms" if ms else "s"
        ax = self.fig.add_subplot(111)
        _style(ax)
        ax.axhspan(0.95, 1.05, color=theme.INK, alpha=0.05, lw=0, zorder=1)
        ax.axhline(1.0, color=theme.C_REF, lw=1.4, ls="--", zorder=2, label="期望(理想步階)")
        ax.axhline(0.0, color=theme.BASELINE, lw=1.0, zorder=2)
        ax.plot(tx, y, color=theme.C_EST, lw=2.2, zorder=4, label="實際響應(估計)")
        ymax = float(np.nanmax(y)) if np.isfinite(y).any() else 1.2
        ax.set_ylim(min(-0.1, float(np.nanmin(y)) - 0.05) if np.isfinite(y).any() else -0.1,
                    max(1.35, ymax + 0.18))
        ax.set_xlim(tx[0], tx[-1])

        def k(v):  # 秒 → 顯示單位
            return v * 1000 if ms else v

        def fmt_t(v):
            return f"{v * 1000:.0f} ms" if v < 2 else f"{v:.2f} s"

        os_pct = _num(st.overshoot_pct)
        if os_pct is not None and os_pct > 0.5 and np.isfinite(y).any():
            i = int(np.nanargmax(y))
            col = theme.S_SERIOUS if os_pct > 25 else theme.S_WARN if os_pct > 12 else theme.INK_2
            ax.plot([tx[i]], [y[i]], "o", ms=7, mfc=col, mec=theme.INK, mew=1.2, zorder=6)
            ax.annotate(f"過衝 {os_pct:.0f} %", xy=(tx[i], y[i]), xytext=(8, 8),
                        textcoords="offset points", color=col, fontsize=9.5, fontweight="bold",
                        bbox=_ANNOT_BOX, zorder=7)
        elif os_pct is not None:
            _note(ax, f"過衝 {os_pct:.0f} %(幾乎沒有過衝)", loc="upper right")
        rise = _num(st.rise_s)
        t10 = self._cross(t, y, 0.1)
        t90 = self._cross(t, y, 0.9)
        if rise is not None and t10 is not None:
            t90 = t10 + rise
        if t10 is not None and t90 is not None and t90 > t10:
            yb = 0.5
            ax.annotate("", xy=(k(t90), yb), xytext=(k(t10), yb),
                        arrowprops=dict(arrowstyle="<->", color=theme.INK_2, lw=1.2), zorder=6)
            for tv, yv in ((t10, 0.1), (t90, 0.9)):
                ax.plot([k(tv), k(tv)], [yv, yb], color=theme.INK_2, lw=0.8, ls=":", zorder=5)
            ax.annotate(f"上升 {fmt_t(t90 - t10)}", xy=(k(t90), yb), xytext=(6, -4),
                        textcoords="offset points", va="top", color=theme.INK, fontsize=9,
                        bbox=_ANNOT_BOX, zorder=7)
        settle = _num(st.settle_s)
        if settle is not None:
            ax.axvline(k(settle), color=theme.INK_2, lw=1.0, ls="--", zorder=3)
            ax.annotate(f"穩定 {fmt_t(settle)}(±5 %)", xy=(k(settle), 0.08),
                        xycoords=("data", "axes fraction"), xytext=(5, 0), textcoords="offset points",
                        color=theme.INK_2, fontsize=8.5, bbox=_ANNOT_BOX, zorder=7)
        ax.set_xlabel(f"時間({unit})")
        ax.set_ylabel("正規化響應")
        title = f"步階響應 — {AXIS_LABEL.get(aa.axis, aa.axis)}"
        if st.n_windows:
            title += f"({st.n_windows} 個時間窗平均)"
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(loc="lower right", fontsize=8.5)
        dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
        if dt >= 0.02:
            _note(ax, f"時間解析度 {dt * 1000:.0f} ms(取樣率低),僅供參考", theme.S_WARN, "upper left")
        self._layout(left=62, right=14, top=30, bottom=46)

    @staticmethod
    def _cross(t, y, level) -> float | None:
        ok = np.isfinite(y)
        idx = np.flatnonzero(ok[:-1] & ok[1:] & (y[:-1] < level) & (y[1:] >= level))
        if idx.size == 0:
            return None
        i = int(idx[0])
        return float(t[i] + (level - y[i]) / (y[i + 1] - y[i]) * (t[i + 1] - t[i]))


# =====================================================================
# 元件
# =====================================================================

_QSS = f"""
QWidget#TuningPanel {{ background: {theme.PLANE}; }}
QLabel#TpTitle {{ color: {theme.INK}; font-size: 12.5pt; font-weight: 700; }}
QLabel#TpScope {{ color: {theme.INK_2}; font-size: 9.5pt; }}
QLabel#TpMuted {{ color: {theme.MUTED}; font-size: 9pt; }}
QLabel#TpMsg {{ color: {theme.INK_2}; font-size: 13pt; }}
QLabel#TpMsgSub {{ color: {theme.MUTED}; font-size: 9.5pt; }}
QFrame#AxisCard {{ background: {theme.PANEL}; border: 2px solid {theme.BORDER}; border-radius: 8px; }}
QFrame#AxisCard:hover {{ border-color: #55554f; }}
QFrame#AxisCard[selected="true"] {{ border-color: {theme.C_REF}; }}
QFrame#AxisCard QWidget {{ background: transparent; }}
QLabel#AxTitle {{ color: {theme.INK}; font-size: 10.5pt; font-weight: 700; }}
QLabel#AxLevel {{ color: {theme.MUTED}; font-size: 8.5pt; }}
QLabel#MLabel {{ color: {theme.MUTED}; font-size: 8.5pt; }}
QLabel#MValue {{ color: {theme.INK}; font-size: 11pt; font-weight: 600; }}
QLabel#MUnit {{ color: {theme.MUTED}; font-size: 8.5pt; }}
QLabel#AxSummary {{ color: {theme.INK_2}; font-size: 9pt; }}
QFrame#Banner {{ background: {_rgba(theme.S_WARN, 0.08)}; border: 1px solid {_rgba(theme.S_WARN, 0.45)};
                 border-radius: 8px; }}
QFrame#Banner QWidget {{ background: transparent; }}
QLabel#BannerHead {{ color: {theme.INK}; font-size: 9.5pt; }}
QLabel#BannerItem {{ color: {theme.INK_2}; font-size: 9pt; }}
QToolButton#LinkBtn {{ background: transparent; border: none; color: {theme.S_WARN}; font-size: 9pt;
                       padding: 0 4px; }}
QToolButton#LinkBtn:hover {{ color: {theme.INK}; }}
QScrollArea#FindScroll {{ background: {theme.SURFACE}; border: none; }}
QWidget#FindHost {{ background: {theme.SURFACE}; }}
QFrame#FindingCard QWidget {{ background: transparent; }}
QLabel#FTitle {{ color: {theme.INK}; font-size: 9.5pt; font-weight: 600; }}
QLabel#FDetail {{ color: {theme.INK_2}; font-size: 9pt; }}
QLabel#FSugg {{ color: {theme.INK_2}; font-size: 9pt; }}
QLabel#GroupHead {{ font-size: 9.5pt; font-weight: 700; padding: 6px 2px 0 2px; }}
QToolButton#SmallBtn {{ background: #2f2f2d; border: 1px solid {theme.BORDER}; border-radius: 5px;
                        color: {theme.INK}; font-size: 8.5pt; padding: 1px 7px; }}
QToolButton#SmallBtn:hover {{ background: #3a3a37; }}
QToolButton#Fold {{ background: transparent; border: none; color: {theme.MUTED}; font-size: 9pt; }}
QToolButton#Fold:hover {{ color: {theme.INK}; }}
QToolButton#AxisSeg {{ background: #2a2a28; border: 1px solid {theme.BORDER}; color: {theme.INK_2};
                       padding: 3px 10px; font-size: 9pt; }}
QToolButton#AxisSeg:checked {{ background: {theme.C_REF}; border-color: {theme.C_REF}; color: #ffffff;
                               font-weight: 600; }}
QToolButton#AxisSeg:hover:!checked {{ background: #353532; }}
QTableWidget#SuggTable {{ border: none; }}
QTabWidget#ChartTabs QTabBar::tab {{ padding: 6px 11px; }}
QPushButton {{ padding: 5px 11px; }}
QProgressBar#Busy {{ max-height: 6px; min-height: 6px; border: none; background: #2a2a28;
                     border-radius: 3px; }}
"""


class _ElideLabel(QLabel):
    """單行、超出寬度時以「…」截斷(完整文字放在 tooltip)。"""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(10)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 — Qt 命名
        self._full = text or ""
        self.setToolTip(self._full if self._full else "")
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, ev):  # noqa: N802
        super().resizeEvent(ev)
        self._elide()

    def _elide(self) -> None:
        w = max(self.width() - 2, 10)
        super().setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, w))


class AxisCard(QFrame):
    """單軸摘要卡:軸名 + 結論標籤、6 個指標、一句話結論;點一下選這一軸。"""

    clicked = pyqtSignal(str)

    def __init__(self, axis: str, parent=None):
        super().__init__(parent)
        self.axis = axis
        self.setObjectName("AxisCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("selected", False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        v = QVBoxLayout(self)
        v.setContentsMargins(11, 8, 11, 8)
        v.setSpacing(5)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = QLabel(AXIS_LABEL.get(axis, axis))
        self.title.setObjectName("AxTitle")
        self.level = QLabel("")
        self.level.setObjectName("AxLevel")
        head.addWidget(self.title)
        head.addWidget(self.level)
        head.addStretch(1)
        self.chip_box = QHBoxLayout()
        self.chip_box.setContentsMargins(0, 0, 0, 0)
        head.addLayout(self.chip_box)
        v.addLayout(head)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        self._cells = []
        for i in range(6):
            r, c = divmod(i, 3)
            lab = QLabel("")
            lab.setObjectName("MLabel")
            val = QLabel("—")
            val.setObjectName("MValue")
            unit = QLabel("")
            unit.setObjectName("MUnit")
            row = QHBoxLayout()
            row.setSpacing(3)
            row.addWidget(val)
            row.addWidget(unit, 0, Qt.AlignmentFlag.AlignBottom)
            row.addStretch(1)
            cell = QVBoxLayout()
            cell.setSpacing(0)
            cell.addWidget(lab)
            cell.addLayout(row)
            grid.addLayout(cell, r, c)
            self._cells.append((lab, val, unit))
        for c in range(3):
            grid.setColumnStretch(c, 1)
        v.addLayout(grid)
        self.summary = _ElideLabel("")
        self.summary.setObjectName("AxSummary")
        v.addWidget(self.summary)
        self.set_axis(None)

    def set_axis(self, aa: AxisAnalysis | None) -> None:
        _clear_layout(self.chip_box)
        if aa is None:
            self.chip_box.addWidget(_chip("無資料", None))
            self.level.setText("")
            self.summary.setText("沒有這一軸的資料")
        else:
            self.chip_box.addWidget(sev_chip(aa.verdict))
            self.level.setText(f"{LEVEL_TEXT.get(aa.level, aa.level)}{_hz(aa.sample_hz, ' · ')}")
            self.summary.setText(aa.summary or "")
        for (lab, val, unit), (l_, v_, u_, tip) in zip(self._cells, axis_metrics(aa)):
            lab.setText(l_)
            val.setText(v_)
            unit.setText(u_)
            for w in (lab, val, unit):
                w.setToolTip(tip)
        self.verdict = aa.verdict if aa is not None else None

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", bool(on))
        self.style().unpolish(self)
        self.style().polish(self)

    def is_selected(self) -> bool:
        return bool(self.property("selected"))

    def mousePressEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.axis)
        super().mousePressEvent(ev)

    def summary_text(self) -> str:
        return self.summary.full_text()

    def metric_texts(self) -> list[str]:
        return [f"{lab.text()} {val.text()}{unit.text()}" for lab, val, unit in self._cells]


class FindingCard(QFrame):
    """一項診斷:圖示 + 標題 + 軸標籤 +「跳到」+ 展開;下方每條建議一行(展開時附原因與說明)。"""

    clicked = pyqtSignal(object)
    seek = pyqtSignal(float)

    def __init__(self, f: Finding, t_base: float, expanded: bool = False, parent=None):
        super().__init__(parent)
        self.finding = f
        self.setObjectName("FindingCard")
        self._sel = False
        self._expanded = expanded
        col = SEV_COLOR.get(f.severity, theme.MUTED)
        self._col = col
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 7, 8, 8)
        v.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(6)
        icon = QLabel(SEV_GLYPH.get(f.severity, "●"))
        icon.setStyleSheet(f"color:{col};font-size:10pt;")
        icon.setToolTip(SEV_LABEL.get(f.severity, f.severity))
        icon.setAlignment(Qt.AlignmentFlag.AlignTop)
        row.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        self.title = QLabel(f.title)
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.title.setObjectName("FTitle")
        self.title.setWordWrap(True)
        self.title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row.addWidget(self.title, 1)
        ax_name = AXIS_SHORT.get(f.axis, f.axis or "")
        if ax_name and ax_name.lower() not in (f.title or "").lower():
            row.addWidget(_chip(ax_name, None), 0, Qt.AlignmentFlag.AlignTop)
        self.btn_seek = None
        if f.t_focus is not None:
            b = QToolButton()
            b.setObjectName("SmallBtn")
            b.setText(f"跳到 {format_clock(f.t_focus - t_base, 0)}")
            b.setToolTip(f"把播放頭移到 {format_clock(f.t_focus - t_base, 1)}(雙擊卡片也可以)")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda: self.seek.emit(float(self.finding.t_focus)))
            row.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
            self.btn_seek = b
        self.fold = QToolButton()
        self.fold.setObjectName("Fold")
        self.fold.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fold.clicked.connect(self.toggle)
        row.addWidget(self.fold, 0, Qt.AlignmentFlag.AlignTop)
        v.addLayout(row)

        self.detail = QLabel(f.detail or "")
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.detail.setObjectName("FDetail")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.detail)
        self.sugg_labels = []
        for s in f.suggestions:
            lab = QLabel()
            lab.setObjectName("FSugg")
            lab.setWordWrap(True)
            lab.setTextFormat(Qt.TextFormat.RichText)
            v.addWidget(lab)
            self.sugg_labels.append((lab, s))
        self._apply()

    @staticmethod
    def _sugg_html(s: Suggestion, with_reason: bool) -> str:
        c = DIR_COLOR.get(s.direction, theme.INK_2)
        d = DIR_TEXT.get(s.direction, s.direction)
        head = (f"<span style='color:{c};font-weight:600'>{d}</span>&nbsp; "
                f"<b style='color:{theme.INK}'>{html.escape(s.param or '')}</b>")
        if s.direction != DIR_CHECK:
            if s.proposed is not None:
                pct = fmt_pct(s.change_pct) if _num(s.change_pct) is not None else ""
                head += (f"&nbsp; {fmt_param(s.current)} → <b style='color:{theme.INK}'>"
                         f"{fmt_param(s.proposed)}</b>" + (f"({pct})" if pct else ""))
            elif s.current is not None:
                head += f"&nbsp;(目前 {fmt_param(s.current)})"
        if with_reason and s.reason:
            head += f"<br><span style='color:{theme.MUTED}'>{html.escape(s.reason)}</span>"
        return head

    def _apply(self) -> None:
        exp = self._expanded
        has_more = bool(self.finding.detail) or any(s.reason for s in self.finding.suggestions)
        self.fold.setVisible(has_more)
        self.fold.setText("▼" if exp else "▶")
        self.fold.setToolTip("收合" if exp else "展開說明與原因")
        self.detail.setVisible(exp and bool(self.finding.detail))
        for lab, s in self.sugg_labels:
            lab.setText(self._sugg_html(s, exp))
        self._restyle()

    def _restyle(self) -> None:
        edge = theme.C_REF if self._sel else theme.BORDER
        bg = "#262a2f" if self._sel else theme.PANEL
        self.setStyleSheet(
            f"QFrame#FindingCard{{background:{bg};border:1px solid {edge};"
            f"border-left:3px solid {self._col};border-radius:6px;}}")

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._apply()

    def is_expanded(self) -> bool:
        return self._expanded

    def set_selected(self, on: bool) -> None:
        self._sel = bool(on)
        self._restyle()

    def mousePressEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.finding)
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton and self.finding.t_focus is not None:
            self.seek.emit(float(self.finding.t_focus))
        super().mouseDoubleClickEvent(ev)


class FindingsView(QScrollArea):
    """依嚴重度分組的診斷清單(重 → 輕),最後是備註。"""

    selected = pyqtSignal(object)
    seekRequested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FindScroll")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        host.setObjectName("FindHost")
        self._v = QVBoxLayout(host)
        self._v.setContentsMargins(8, 4, 8, 10)
        self._v.setSpacing(6)
        self.setWidget(host)
        self.cards: list[FindingCard] = []
        self._current: Finding | None = None

    def set_findings(self, findings: list[Finding], t_base: float, notes: list[str] | None = None) -> None:
        _clear_layout(self._v)
        self.cards = []
        self._current = None
        groups: dict[str, list[Finding]] = {}
        for f in findings:
            groups.setdefault(f.severity, []).append(f)
        order = sorted(groups, key=lambda s: -SEV_ORDER.get(s, 0))
        first = True
        for sev in order:
            fs = groups[sev]
            col = SEV_COLOR.get(sev, theme.MUTED)
            head = QLabel(f"<span style='color:{col}'>{SEV_GLYPH.get(sev, '●')} "
                          f"{SEV_LABEL.get(sev, sev)}</span>"
                          f"<span style='color:{theme.MUTED};font-weight:400'> · {len(fs)} 項</span>")
            head.setObjectName("GroupHead")
            self._v.addWidget(head)
            for f in fs:
                # 最嚴重的第一項展開;「需要處理」全部展開
                exp = sev == SEV_BAD or (first and sev != SEV_OK)
                first = False
                card = FindingCard(f, t_base, expanded=exp)
                card.clicked.connect(self._on_click)
                card.seek.connect(self.seekRequested.emit)
                self._v.addWidget(card)
                self.cards.append(card)
        if not findings:
            lab = QLabel("沒有發現需要調整的地方。")
            lab.setObjectName("TpMsgSub")
            lab.setContentsMargins(4, 10, 4, 4)
            self._v.addWidget(lab)
        if notes:
            head = QLabel(f"<span style='color:{theme.INK_2}'>備註</span>")
            head.setObjectName("GroupHead")
            self._v.addWidget(head)
            for n in notes:
                lab = QLabel(f"• {n}")
                lab.setTextFormat(Qt.TextFormat.PlainText)
                lab.setObjectName("FDetail")
                lab.setWordWrap(True)
                lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                lab.setContentsMargins(4, 0, 2, 0)
                self._v.addWidget(lab)
        self._v.addStretch(1)
        self.verticalScrollBar().setValue(0)

    def _on_click(self, f: Finding) -> None:
        self.select(f)
        self.selected.emit(f)

    def select(self, f: Finding | None) -> None:
        self._current = f
        for c in self.cards:
            c.set_selected(c.finding is f)

    def current(self) -> Finding | None:
        return self._current


_COLS = ["優先", "參數", "目前值", "方向", "建議值", "變動", "原因"]


class _WrapDelegate(QStyledItemDelegate):
    """可換行的儲存格(長英文參數名也能在任何字元斷行,不會被「…」截掉),列高跟著內容。

    content(index) 回傳 (文字, 是否 HTML);回傳 None 時照一般方式繪製。
    """

    PAD_X, PAD_Y = 6, 5

    def __init__(self, table, content=None):
        super().__init__(table)
        self._content = content or (lambda index: (index.data() or "", False))

    def _doc(self, text: str, html: bool, font, width: float) -> QTextDocument:
        doc = QTextDocument()
        if html:                             # HTML 自己用 <b> 標粗體;底字型不沿用儲存格的粗體
            font = QFont(font)
            font.setBold(False)
        doc.setDefaultFont(font)
        doc.setDocumentMargin(0)
        opt = QTextOption()
        opt.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        doc.setDefaultTextOption(opt)
        if html:
            doc.setHtml(text)
        else:
            doc.setPlainText(text)
        doc.setTextWidth(max(width, 16.0))
        return doc

    def paint(self, painter, option, index):
        c = self._content(index)
        if c is None:
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        r = opt.rect.adjusted(self.PAD_X, self.PAD_Y, -self.PAD_X, -self.PAD_Y)
        doc = self._doc(c[0], c[1], opt.font, r.width())
        fg = index.data(Qt.ItemDataRole.ForegroundRole)
        color = fg.color() if fg is not None else opt.palette.color(QPalette.ColorRole.Text)
        if opt.state & QStyle.StateFlag.State_Selected:
            color = QColor(theme.INK)
        ctx = QAbstractTextDocumentLayout.PaintContext()
        ctx.palette.setColor(QPalette.ColorRole.Text, color)
        painter.save()
        painter.setClipRect(opt.rect)
        dy = max((r.height() - doc.size().height()) / 2, 0)
        painter.translate(QPointF(r.left(), r.top() + dy))
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        c = self._content(index)
        if c is None:
            return super().sizeHint(option, index)
        table = self.parent()
        w = (table.columnWidth(index.column()) if table is not None else 200) - 2 * self.PAD_X
        doc = self._doc(c[0], c[1], option.font, w)
        return QSize(int(w), int(doc.size().height()) + 2 * self.PAD_Y + 2)


class SuggestionTable(QTableWidget):
    """建議調整表(report.suggestions() 去重、依優先排序)。

    窄的時候(< COMPACT_W)改成精簡版:「原因」欄收起,原因改成參數名下方的灰色第二行。
    """

    COMPACT_W = 600
    C_PARAM, C_REASON = 1, len(_COLS) - 1

    def __init__(self, parent=None):
        super().__init__(0, len(_COLS), parent)
        self.setObjectName("SuggTable")
        self.setHorizontalHeaderLabels(_COLS)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setWordWrap(True)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.TextElideMode.ElideNone)
        h = self.horizontalHeader()
        h.setMinimumSectionSize(36)
        h.sectionResized.connect(self._on_section_resized)
        self.setItemDelegateForColumn(self.C_REASON, _WrapDelegate(self))
        self.setItemDelegateForColumn(self.C_PARAM, _WrapDelegate(self, self._param_content))
        self.compact = False
        self._fit_pending = False              # 要在 _apply_modes 之前:設欄寬會觸發 sectionResized
        self.rows: list[Suggestion] = []
        self._apply_modes()

    def _param_content(self, index):
        """精簡版:參數名(粗體)+ 原因(灰色、可換行)。一般版照常繪製。"""
        if not self.compact:
            return None
        r = index.row()
        if not (0 <= r < len(self.rows)):
            return None
        s = self.rows[r]
        esc = (lambda t: (t or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        html = f"<b style='color:{theme.INK}'>{esc(s.param)}</b>"
        if s.reason:
            html += f"<br><span style='color:{theme.MUTED};font-size:8.5pt'>{esc(s.reason)}</span>"
        return html, True

    PARAM_MAX_FRAC = 0.34          # 一般版:參數欄最多佔表寬 34 %(長的檢查項目換行,不要把「原因」擠成一條)

    def _apply_modes(self) -> None:
        h = self.horizontalHeader()
        for c in range(len(_COLS)):
            if c == self.C_REASON:
                mode = QHeaderView.ResizeMode.Stretch
            elif c == self.C_PARAM:
                mode = QHeaderView.ResizeMode.Stretch if self.compact else QHeaderView.ResizeMode.Interactive
            else:
                mode = QHeaderView.ResizeMode.ResizeToContents
            h.setSectionResizeMode(c, mode)
        self.setColumnHidden(self.C_REASON, self.compact)
        self._fit_param_width()

    def _fit_param_width(self) -> None:
        """一般版的參數欄:內容寬度,但不超過表寬的 PARAM_MAX_FRAC。"""
        if self.compact:
            return
        want = max(self.sizeHintForColumn(self.C_PARAM), self.horizontalHeader().sectionSizeHint(self.C_PARAM))
        cap = int(self.PARAM_MAX_FRAC * max(self.viewport().width(), 300))
        self.setColumnWidth(self.C_PARAM, max(80, min(want, cap)))

    def set_compact(self, on: bool) -> None:
        on = bool(on)
        if on == self.compact:
            return
        self.compact = on
        self._apply_modes()
        self._fit_rows_later()

    def set_suggestions(self, sugg: list[Suggestion]) -> None:
        self.rows = list(sugg)
        self.setRowCount(len(self.rows))
        right = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        left = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        center = Qt.AlignmentFlag.AlignCenter
        for r, s in enumerate(self.rows):
            cells = [
                (str(s.priority), center, theme.INK if s.priority <= 1 else theme.INK_2, s.priority <= 1),
                (s.param, left, theme.INK, True),
                (fmt_param(s.current) if s.direction != DIR_CHECK else "—", right, theme.INK_2, False),
                (DIR_TEXT.get(s.direction, s.direction), left, DIR_COLOR.get(s.direction, theme.INK_2), True),
                (fmt_param(s.proposed) if s.direction != DIR_CHECK else "—", right, theme.INK, True),
                (fmt_pct(s.change_pct), right, theme.INK_2, False),
                (s.reason, left, theme.INK_2, False),
            ]
            for c, (text, al, color, bold) in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setTextAlignment(al)
                it.setForeground(QColor(color))
                if bold:
                    fnt = it.font()
                    fnt.setBold(True)
                    it.setFont(fnt)
                if c in (self.C_PARAM, self.C_REASON) and s.reason:
                    it.setToolTip(s.reason)
                if c == 0:
                    it.setToolTip("1 = 先改這個;一次只改一項,試飛確認後再改下一項")
                self.setItem(r, c, it)
        self._fit_param_width()
        self.resizeRowsToContents()
        self._fit_rows_later()

    def tsv(self) -> str:
        lines = ["\t".join(_COLS)]
        for s in self.rows:
            lines.append("\t".join([
                str(s.priority), s.param,
                fmt_param(s.current) if s.direction != DIR_CHECK else "",
                DIR_TEXT.get(s.direction, s.direction),
                fmt_param(s.proposed) if s.direction != DIR_CHECK else "",
                fmt_pct(s.change_pct) if _num(s.change_pct) is not None else "",
                (s.reason or "").replace("\t", " ").replace("\n", " "),
            ]))
        return "\n".join(lines) + "\n"

    def _on_section_resized(self, idx: int, _old: int, _new: int) -> None:
        if idx in (self.C_PARAM, self.C_REASON):
            self._fit_rows_later()

    def _fit_rows_later(self) -> None:
        """換行欄位的列高跟著欄寬變;排版穩定後(下一輪事件)再算,只算一次。"""
        if self._fit_pending:
            return
        self._fit_pending = True

        def run():
            self._fit_pending = False
            self.resizeRowsToContents()
        QTimer.singleShot(0, run)

    def resizeEvent(self, ev):  # noqa: N802
        super().resizeEvent(ev)
        w = self.viewport().width()          # 遲滯:捲軸出現 / 消失時不會來回切換
        self.set_compact(w < self.COMPACT_W if not self.compact else w < self.COMPACT_W + 30)
        self._fit_param_width()
        self._fit_rows_later()


class SafetyBanner(QFrame):
    """安全須知(S_WARN):標題一行永遠看得到,細節可收合。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Banner")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 5, 8, 5)
        v.setSpacing(3)
        row = QHBoxLayout()
        row.setSpacing(6)
        icon = QLabel("⚠")
        icon.setStyleSheet(f"color:{theme.S_WARN};font-size:11pt;")
        row.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        self.head = QLabel(f"<b style='color:{theme.S_WARN}'>安全須知</b>&nbsp; {SAFETY_HEAD}")
        self.head.setObjectName("BannerHead")
        self.head.setWordWrap(True)
        row.addWidget(self.head, 1)
        self.btn = QToolButton()
        self.btn.setObjectName("LinkBtn")
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.clicked.connect(lambda: self.set_expanded(not self._exp))
        row.addWidget(self.btn, 0, Qt.AlignmentFlag.AlignTop)
        v.addLayout(row)
        self.more = QLabel("<br>".join(f"• {x}" for x in SAFETY_MORE))
        self.more.setObjectName("BannerItem")
        self.more.setWordWrap(True)
        self.more.setContentsMargins(22, 0, 0, 2)
        v.addWidget(self.more)
        self._exp = True
        self.set_expanded(False)

    def set_expanded(self, on: bool) -> None:
        self._exp = bool(on)
        self.more.setVisible(self._exp)
        self.btn.setText("收合 ▲" if self._exp else "詳細 ▼")

    def is_expanded(self) -> bool:
        return self._exp


# =====================================================================
# 面板
# =====================================================================

class TuningPanel(QWidget):
    """PID 調參建議分頁。外部只用 set_report / set_busy / set_time 與兩個訊號。"""

    seekRequested = pyqtSignal(float)    # 診斷 / 圖面點擊 → 播放頭(log 絕對秒)
    analyzeRequested = pyqtSignal()      # 「重新分析」

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TuningPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_QSS)
        self._report: TuningReport | None = None
        self._t_base = 0.0
        self._scope = ""
        self._message = EMPTY_MSG
        self._busy = False
        self._axis = "roll"
        self._t: float | None = None
        self._hl: Finding | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)
        root.addLayout(self._build_top())
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_message_page())
        self.stack.addWidget(self._build_main_page())
        root.addWidget(self.stack, 1)
        self.set_report(None)

    # ---------------- 版面 ----------------

    def _build_top(self):
        v = QVBoxLayout()
        v.setSpacing(4)
        r1 = QHBoxLayout()
        r1.setSpacing(10)
        t = QLabel("PID 調參建議")
        t.setObjectName("TpTitle")
        r1.addWidget(t)
        self.lbl_scope = _ElideLabel("")
        self.lbl_scope.setObjectName("TpScope")
        r1.addWidget(self.lbl_scope, 1)
        self.busy_bar = QProgressBar()
        self.busy_bar.setObjectName("Busy")
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setFixedWidth(90)
        self.busy_bar.hide()
        self.lbl_busy = QLabel("分析中…")
        self.lbl_busy.setObjectName("TpMuted")
        self.lbl_busy.hide()
        r1.addWidget(self.lbl_busy)
        r1.addWidget(self.busy_bar)
        self.verdict_box = QHBoxLayout()
        self.verdict_box.setContentsMargins(0, 0, 0, 0)
        r1.addLayout(self.verdict_box)
        self.btn_analyze = QPushButton("重新分析")
        self.btn_analyze.setObjectName("Primary")
        self.btn_analyze.setToolTip("依目前的分析區間重新分析")
        self.btn_analyze.clicked.connect(self.analyzeRequested.emit)
        r1.addWidget(self.btn_analyze)
        v.addLayout(r1)
        self.row_chips = QWidget()
        r2 = QHBoxLayout(self.row_chips)
        r2.setContentsMargins(0, 0, 0, 0)
        r2.setSpacing(6)
        lab = QLabel("資料:")
        lab.setObjectName("TpMuted")
        r2.addWidget(lab)
        self.chip_box = QHBoxLayout()
        self.chip_box.setSpacing(6)
        r2.addLayout(self.chip_box)
        r2.addStretch(1)
        v.addWidget(self.row_chips)
        return v

    def _build_message_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addStretch(2)
        self.lbl_msg = QLabel(EMPTY_MSG)
        self.lbl_msg.setObjectName("TpMsg")
        self.lbl_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_msg.setWordWrap(True)
        v.addWidget(self.lbl_msg)
        sub = QLabel("分析 ArduCopter(或 QuadPlane 的 Q_A_)姿態控制器:期望 vs 實際的追蹤、延遲、過衝與震盪,"
                     "並給出 Roll / Pitch / Yaw 各參數的調整方向。\n"
                     "需要 RATE / ATT;LOG_BITMASK 開 ATTITUDE_FAST(bit 0)與 PID(bit 12)"
                     "(QuadPlane:ATTITUDE_FULLRATE bit 20)可判斷震盪。")
        sub.setObjectName("TpMsgSub")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        sub.setMaximumWidth(620)
        row = QHBoxLayout()                  # 用左右彈簧置中:換行標籤才拿得到正確的高度
        row.addStretch(1)
        row.addWidget(sub, 4)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(3)
        return w

    def _build_main_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        self.banner = SafetyBanner()
        v.addWidget(self.banner)
        cards = QHBoxLayout()
        cards.setSpacing(8)
        self.cards: dict[str, AxisCard] = {}
        for ax in AXES:
            c = AxisCard(ax)
            c.clicked.connect(self.select_axis)
            cards.addWidget(c, 1)
            self.cards[ax] = c
        v.addLayout(cards)

        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.split.setChildrenCollapsible(False)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        lt = QFrame()
        lt.setObjectName("Card")
        ltv = QVBoxLayout(lt)
        ltv.setContentsMargins(4, 6, 2, 2)
        ltv.setSpacing(0)
        self.lbl_find = QLabel("診斷")
        self.lbl_find.setObjectName("SectionTitle")
        self.lbl_find.setContentsMargins(8, 0, 0, 0)
        ltv.addWidget(self.lbl_find)
        self.findings = FindingsView()
        self.findings.selected.connect(self._on_finding)
        self.findings.seekRequested.connect(self.seekRequested.emit)
        ltv.addWidget(self.findings, 1)
        lv.addWidget(lt)
        left.setMinimumWidth(260)
        self.split.addWidget(left)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("ChartTabs")
        tbl_page = QWidget()
        tv = QVBoxLayout(tbl_page)
        tv.setContentsMargins(8, 8, 8, 8)
        tv.setSpacing(6)
        self.table = SuggestionTable()
        tv.addWidget(self.table, 1)
        brow = QHBoxLayout()
        brow.setSpacing(6)
        self.btn_copy = QPushButton("複製表格")
        self.btn_copy.setToolTip("以 Tab 分隔複製,可直接貼到試算表")
        self.btn_copy.clicked.connect(self.copy_tsv)
        self.btn_export = QPushButton("匯出 .param…")
        self.btn_export.setToolTip("只含有建議值的參數(Mission Planner 相容);載入前請逐項確認")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_report = QPushButton("複製報告文字")
        self.btn_report.clicked.connect(self.copy_report)
        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("TpMuted")
        brow.addWidget(self.btn_copy)
        brow.addWidget(self.btn_export)
        brow.addWidget(self.btn_report)
        brow.addWidget(self.lbl_status, 1)
        tv.addLayout(brow)
        self.p_track = TrackingPlot()
        self.p_track.seeked.connect(self.seekRequested.emit)
        self.p_spec = SpectrumPlot()
        self.p_freq = FreqRespPlot()
        self.p_step = StepPlot()
        self.tabs.addTab(tbl_page, "建議調整")
        self.tabs.addTab(self.p_track, "追蹤")
        self.tabs.addTab(self.p_spec, "頻譜")
        self.tabs.addTab(self.p_freq, "頻率響應")
        self.tabs.addTab(self.p_step, "步階響應")
        self.tabs.currentChanged.connect(self._on_tab)

        seg = QWidget()
        sh = QHBoxLayout(seg)
        sh.setContentsMargins(0, 0, 4, 2)
        sh.setSpacing(0)
        self.seg_group = QButtonGroup(self)
        self.seg_group.setExclusive(True)
        self.seg_btns: dict[str, QToolButton] = {}
        for i, ax in enumerate(AXES):
            b = QToolButton()
            b.setObjectName("AxisSeg")
            b.setText(AXIS_SHORT[ax])
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            r = "border-top-left-radius:6px;border-bottom-left-radius:6px;" if i == 0 else \
                "border-top-right-radius:6px;border-bottom-right-radius:6px;" if i == len(AXES) - 1 else ""
            if r:
                b.setStyleSheet(f"QToolButton#AxisSeg{{{r}}}")
            b.clicked.connect(lambda _=False, a=ax: self.select_axis(a))
            self.seg_group.addButton(b)
            sh.addWidget(b)
            self.seg_btns[ax] = b
        self.tabs.setCornerWidget(seg, Qt.Corner.TopRightCorner)
        self.split.addWidget(self.tabs)
        self.split.setStretchFactor(0, 38)
        self.split.setStretchFactor(1, 62)
        self.split.setSizes([380, 620])       # 第一次排版時依比例縮放到實際寬度
        v.addWidget(self.split, 1)
        return w

    # ---------------- 對外 API ----------------

    def set_report(self, report: TuningReport | None, t_base: float = 0.0,
                   scope_text: str = "", message: str = "") -> None:
        """顯示分析結果;report 為 None 時顯示 message(例:「載入 log 後分析」或錯誤訊息)。"""
        self._report = report
        self._t_base = float(t_base or 0.0)
        self._message = message or EMPTY_MSG
        self._hl = None
        _clear_layout(self.verdict_box)
        _clear_layout(self.chip_box)
        if report is None:
            self._scope = scope_text or ""
            self.lbl_scope.setText(self._scope)
            self.row_chips.hide()
            self.lbl_msg.setText("分析中…" if self._busy else self._message)
            self.stack.setCurrentIndex(0)
            for c in self.cards.values():
                c.set_axis(None)
            self.findings.set_findings([], self._t_base)
            self.table.set_suggestions([])
            for p in self._plots():
                p.show_placeholder(self._message)
            self._update_buttons()
            return
        self._scope = scope_text or default_scope(report, self._t_base)
        self.lbl_scope.setText(self._scope)
        v = report.verdict
        self.verdict_box.addWidget(_chip(f"{SEV_GLYPH.get(v, '●')} 整體:{SEV_LABEL.get(v, v)}",
                                         SEV_COLOR.get(v, theme.MUTED), strong=True,
                                         tip="所有診斷中最嚴重的一項"))
        chips = data_chips(report)
        for text, color, tip in chips:
            self.chip_box.addWidget(_chip(text, color, tip))
        self.row_chips.setVisible(bool(chips))
        self.stack.setCurrentIndex(1)
        for ax, c in self.cards.items():
            c.set_axis(report.axes.get(ax))
        self.findings.set_findings(report.findings, self._t_base, report.notes)
        n_bad = sum(1 for f in report.findings if f.severity != SEV_OK)
        self.lbl_find.setText(f"診斷 · {n_bad} 項待處理" if n_bad else "診斷")
        sugg = report.suggestions()
        self.table.set_suggestions(sugg)
        self.tabs.setTabText(PAGE_TABLE, f"建議調整 ({len(sugg)})" if sugg else "建議調整")
        self.lbl_status.setText("")
        self._update_buttons()
        # 預設選最嚴重的一軸
        axes = [a for a in AXES if a in report.axes]
        if axes:
            worst = max(axes, key=lambda a: (SEV_ORDER.get(report.axes[a].verdict, 0), -AXES.index(a)))
            self._axis = worst
        self.tabs.setCurrentIndex(PAGE_TABLE if sugg else PAGE_TRACK)
        self._on_tab(self.tabs.currentIndex())
        self._render_axis()

    def set_busy(self, on: bool) -> None:
        self._busy = bool(on)
        self.busy_bar.setVisible(self._busy)
        self.lbl_busy.setVisible(self._busy)
        self.btn_analyze.setEnabled(not self._busy)
        if self._report is None:
            self.lbl_msg.setText("分析中…" if self._busy else self._message)

    def set_time(self, t: float | None) -> None:
        """播放頭(log 絕對秒;None = 隱藏)。只移動追蹤圖的 blit 游標,很便宜。"""
        self._t = None if t is None else float(t)
        self.p_track.set_time(self._t)

    # ---------------- 其他公開方法(測試 / 整合用) ----------------

    def report(self) -> TuningReport | None:
        return self._report

    def selected_axis(self) -> str:
        return self._axis

    def is_busy(self) -> bool:
        return self._busy

    def select_axis(self, axis: str) -> None:
        if axis not in AXES:
            return
        changed = axis != self._axis
        self._axis = axis
        if changed and self._hl is not None and self._hl.axis not in (None, axis):
            self._hl = None
            self.findings.select(None)
        self._render_axis()

    def suggestion_tsv(self) -> str:
        return self.table.tsv()

    def copy_tsv(self) -> str:
        text = self.table.tsv()
        QGuiApplication.clipboard().setText(text)
        self.lbl_status.setText(f"已複製 {len(self.table.rows)} 列")
        return text

    def report_text(self) -> str:
        r = self._report
        if r is None:
            return ""
        try:
            from core.tuning import format_report_text  # T2 提供時優先使用
        except Exception:  # noqa: BLE001 — 模組不存在或匯入失敗都退回內建版本
            format_report_text = None
        if format_report_text is not None:
            try:
                text = format_report_text(r)
                if isinstance(text, str) and text.strip():
                    return text
            except Exception:  # noqa: BLE001
                pass
        return build_report_text(r, self._t_base, self._scope)

    def copy_report(self) -> str:
        text = self.report_text()
        if text:
            QGuiApplication.clipboard().setText(text)
            self.lbl_status.setText("已複製報告文字")
        return text

    def export_param(self, path: str) -> int:
        """寫出 report.param_file();回傳參數筆數。"""
        if self._report is None:
            return 0
        text = self._report.param_file()
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return sum(1 for ln in text.splitlines() if ln.strip() and not ln.startswith("#"))

    def _on_export(self) -> None:
        if self._report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "匯出建議參數", "pid_suggestions.param",
                                              "參數檔 (*.param *.parm);;所有檔案 (*)")
        if path:
            n = self.export_param(path)
            self.lbl_status.setText(f"已匯出 {n} 個參數(載入前請逐項確認)")

    # ---------------- 內部 ----------------

    def _plots(self):
        return (self.p_track, self.p_spec, self.p_freq, self.p_step)

    def _update_buttons(self) -> None:
        r = self._report
        sugg = self.table.rows
        self.btn_copy.setEnabled(bool(sugg))
        self.btn_report.setEnabled(r is not None)
        self.btn_export.setEnabled(any(s.direction in (DIR_UP, DIR_DOWN, DIR_SET) and s.proposed is not None
                                       for s in sugg))

    def _render_axis(self) -> None:
        for ax, c in self.cards.items():
            c.set_selected(ax == self._axis)
        for ax, b in self.seg_btns.items():
            b.setChecked(ax == self._axis)
        r = self._report
        if r is None:
            return
        aa = r.axes.get(self._axis)
        hl = self._hl if (self._hl is not None and self._hl.axis in (None, self._axis)) else None
        self.p_track.show_axis(aa, self._t_base, highlight=hl)
        for p in (self.p_spec, self.p_freq, self.p_step):
            p.show_axis(aa, self._t_base)
        if self._t is not None:
            self.p_track.set_time(self._t)

    def _on_finding(self, f: Finding) -> None:
        self._hl = f
        if f.axis in AXES and f.axis != self._axis:
            self._axis = f.axis
            self._render_axis()
        else:
            self.p_track.show_axis(self._report.axes.get(self._axis) if self._report else None,
                                   self._t_base, highlight=f)
        if self._report is not None and self._report.axes.get(self._axis) is not None:
            self.tabs.setCurrentIndex(chart_page_for(f))

    def _on_tab(self, i: int) -> None:
        self.tabs.cornerWidget(Qt.Corner.TopRightCorner).setVisible(i != PAGE_TABLE)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(1100, 700)
