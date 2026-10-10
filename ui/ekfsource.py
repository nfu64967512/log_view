"""「EKF 來源」分頁:EKF 有沒有偷參考 GPS(純 VIO 飛行時要看的)。

上:來源組設定表(★ = 該軸用 GPS)與每一段使用中來源組的判讀;
下:洩漏係數 α 隨時間(0 = EKF 只跟 VIO、1 = 被 GPS 帶著走)。
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QHeaderView, QLabel, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from core import ekfsrc
from core.timeline import format_clock

from . import theme
from .plots import PlotWidget, _in_range, _set_vline, _style_axes

_SET_COLORS = {1: "#3987e5", 2: "#d95926", 3: "#9085e9"}   # 來源組 1 / 2 / 3 的色帶(旁邊一定有文字標籤)


class LeakPlot(PlotWidget):
    """洩漏係數 α 隨時間;背景色帶 = 使用中的來源組。"""

    def __init__(self, parent=None):
        self._segs: list = []
        super().__init__(parent, toolbar=False)

    def render_leak(self, res, segs, logdata):
        self._segs = segs
        self._redraw(self.draw_result, res, logdata)

    def draw_result(self, res, logdata):
        tb = logdata.t_base
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(theme.SURFACE)
        for sg in self._segs:
            ax.axvspan(sg.t0 - tb, sg.t1 - tb, color=_SET_COLORS.get(sg.set_n, "#888"), alpha=0.10, lw=0)
            ax.annotate(f"來源組 {sg.set_n}", xy=((sg.t0 + sg.t1) / 2 - tb, 1.0),
                        xycoords=("data", "axes fraction"), xytext=(0, -12),
                        textcoords="offset points", ha="center", fontsize=8, color=theme.MUTED)
        ax.axhline(0.0, color=theme.C_REF, lw=1.2, ls="--")
        ax.axhline(1.0, color=theme.C_EST, lw=1.2, ls="--")
        ax.text(0.005, 0.0, " 0 = 只跟 VIO", transform=ax.get_yaxis_transform(), va="bottom",
                fontsize=9, color=theme.C_REF)
        ax.text(0.005, 1.0, " 1 = 被 GPS 帶著走", transform=ax.get_yaxis_transform(), va="top",
                fontsize=9, color=theme.C_EST)
        t = res.t - tb
        ax.plot(t, np.clip(res.alpha, -0.6, 1.6), "-o", color=theme.INK, lw=1.8, ms=4,
                label=f"視窗 {res.win_s:g} s 的 α")
        bad = ~np.isfinite(res.alpha)
        if bad.any():
            ax.plot(t[bad], np.full(int(bad.sum()), 0.5), "x", color=theme.MUTED, ms=6,
                    label=f"不可判定(GPS 與 VIO 差 < {res.min_std:g} m)")
        ax.set_ylim(-0.6, 1.6)
        lo = min([float(res.t_all[0])] + [s.t0 for s in self._segs])
        hi = max([float(res.t_all[-1])] + [s.t1 for s in self._segs])
        ax.set_xlim(lo - tb, hi - tb)
        ax.set_xlabel("時間(秒,相對 log 起點)")
        ax.set_ylabel("洩漏係數 α")
        ax.set_title(f"EKF({res.ekf_name})相對 GPS({res.gps_name})/ VIO({res.vio_name})")
        _style_axes(ax)
        ax.legend(loc="lower right", fontsize=8)
        self.t_base = tb
        self._rng = (lo, hi)
        self._cur_line = self._animated(ax.plot([], [], color=theme.INK, lw=1.2, zorder=8,
                                                transform=ax.get_xaxis_transform())[0])
        self.fig.tight_layout()

    def update_cursor(self, t):
        _set_vline(self._cur_line, t - self.t_base if _in_range(t, *self._rng) else None)
        return False


class EkfSourcePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        lbl = QLabel("EKF 來源 — EKF 有沒有偷參考 GPS")
        lbl.setObjectName("SectionTitle")
        v.addWidget(lbl)

        self.tbl_sets = self._table(["", "來源組 1", "來源組 2", "來源組 3"], len(ekfsrc.AXES) + 1,
                                    (len(ekfsrc.AXES) + 2) * 28 + 8)
        v.addWidget(self.tbl_sets)
        self.tbl_seg = self._table(["時段(相對 log 起點)", "來源組", "GPS 定位", "α", "判讀"], 0, 100)
        self.tbl_seg.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.tbl_seg)
        self.lbl_note = QLabel("")
        self.lbl_note.setObjectName("Hint")
        self.lbl_note.setWordWrap(True)
        v.addWidget(self.lbl_note)
        self.plot = LeakPlot()
        self.plot.setMinimumHeight(220)
        v.addWidget(self.plot, 1)
        self._ld = None
        self._info = None
        self.set_logdata(None)

    @staticmethod
    def _table(headers, rows, min_h) -> QTableWidget:
        t = QTableWidget(rows, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.setMinimumHeight(min_h)
        return t

    # --- 資料 ---
    def set_logdata(self, ld):
        self._ld = ld
        self._info = None
        self.tbl_sets.clearContents()
        self.tbl_seg.setRowCount(0)
        self.lbl_note.setText("")
        if ld is None:
            self.plot.show_placeholder("載入 log 後在這裡檢查 EKF 來源")
            return
        try:
            self._info = ekfsrc.inspect(ld)
        except Exception as e:  # noqa: BLE001
            self.plot.show_placeholder(f"無法讀取 EKF 來源設定:{type(e).__name__}: {e}")
            return
        self._fill_sets()
        self._fill_segments()
        self._set_note("")
        self.plot.show_placeholder("計算 ATE 後(需要 VIO、GPS 與 EKF 三個來源)顯示 GPS 洩漏係數")

    def set_leak(self, res, message: str = ""):
        """res = ekfsrc.LeakResult 或 None(message 說明為什麼沒有)。"""
        if self._info is None or self._ld is None:
            return
        ekfsrc.apply_leak(self._info, res)
        self._fill_segments()
        if res is None:
            self.plot.show_placeholder(message or "沒有比對資料")
        else:
            self.plot.render_leak(res, self._info.segments, self._ld)
        self._set_note(message if res is None else "")

    def set_time(self, t):
        self.plot.set_time(t)

    # --- 填表 ---
    def _fill_sets(self):
        info = self._info
        for r, (axis, zh) in enumerate(ekfsrc.AXES):
            self._put(self.tbl_sets, r, 0, zh, bold=True)
            for n in (1, 2, 3):
                gps = info.sets[n].codes.get(axis, 0) in ekfsrc._GPS[axis]
                self._put(self.tbl_sets, r, n, ("★ " if gps else "") + info.sets[n].name(axis),
                          color=theme.S_WARN if gps else None)
        r = len(ekfsrc.AXES)
        self._put(self.tbl_sets, r, 0, "判定", bold=True)
        for n in (1, 2, 3):
            s = info.sets[n]
            col = theme.S_GOOD if (s.vio_axes and not s.gps_axes) else (theme.S_WARN if s.gps_axes else None)
            self._put(self.tbl_sets, r, n, s.verdict, color=col)

    def _fill_segments(self):
        info, tb = self._info, self._ld.t_base
        self.tbl_seg.setRowCount(len(info.segments))
        for r, sg in enumerate(info.segments):
            self._put(self.tbl_seg, r, 0, f"{format_clock(sg.t0 - tb, 1)} ~ {format_clock(sg.t1 - tb, 1)}")
            self._put(self.tbl_seg, r, 1, str(sg.set_n))
            self._put(self.tbl_seg, r, 2, "—" if sg.gps_fix is None else f"{sg.gps_fix * 100:.0f}%")
            self._put(self.tbl_seg, r, 3, "—" if sg.alpha is None else f"{sg.alpha:.2f}")
            text = ekfsrc.judge(info, sg)
            self._put(self.tbl_seg, r, 4, text, color=theme.S_CRIT if text.startswith("⚠") else None)
        self.tbl_seg.setMinimumHeight(min(8, max(2, len(info.segments))) * 28 + 36)

    def _set_note(self, text: str):
        base = ("★ = 該軸使用 GPS(設定層:設定允許,不等於實際有融合)。"
                + "".join(f" 註:{n}。" for n in (self._info.notes if self._info else [])))
        self.lbl_note.setText((text + "\n" if text else "") + base)

    @staticmethod
    def _put(tbl, r, c, text, bold=False, color=None):
        it = QTableWidgetItem(text)
        it.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        if bold:
            f = it.font()
            f.setBold(True)
            it.setFont(f)
        if color:
            it.setForeground(QColor(color))
        tbl.setItem(r, c, it)
