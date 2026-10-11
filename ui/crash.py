"""「失事分析」分頁:結論、假說排名、關鍵時間軸、證據圖、完整報告。

分析在背景執行緒做(main_window 負責);這裡只負責顯示。
  seekRequested(t_abs)    時間軸雙擊 → 播放頭跳到該時刻(log 絕對秒)
  analyzeRequested()      「重新分析」
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                             QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget)

from core.crash import util
from core.timeline import format_clock

from . import theme
from .plots import PlotWidget, _in_range, _set_vline, _style_axes

_TAG_ZH = {"precursor": "先兆", "development": "發展", "consequence": "後果", "post_crash": "事後",
           "ground_info": "地面", "standing": "持續狀態", "": ""}
_TAG_COLOR = {"precursor": theme.S_WARN, "development": theme.S_SERIOUS, "consequence": theme.S_CRIT,
              "post_crash": theme.MUTED, "ground_info": theme.MUTED}
_STATUS_ZH = {"ranked": "第一名", "contributing": "貢獻因素", "minor": "次要", "ruled_out": "已排除",
              "not_applicable": "不適用", "no_evidence": "無證據", "candidate": "候選(無法確定)"}
_ANCHOR_LINES = (("t_touch", "觸地", theme.C_REF), ("t_warn", "誤差>10°", theme.S_WARN),
                 ("t_loss", "誤差>30°", theme.S_SERIOUS), ("t_first_hit", "硬撞擊", theme.S_CRIT),
                 ("t_detect", "判定失事", theme.C_ERR))


def _clk(t) -> str:
    """相對秒(和偵測器說明文字一致)。"""
    return "—" if t is None else f"{float(t):.1f} s"


class EvidencePlot(PlotWidget):
    """共用時間軸的證據圖:姿態、高度 / 油門、馬達、撞擊加速度、EKF 測試比;陰影 = 先兆 / 發展 / 後果。"""

    def draw_result(self, rep, logdata):
        self.t_base = tb = float(logdata.t_base)
        ix = logdata.index
        a = rep.anchors

        def ser(key):
            s = ix.series(key) if ix is not None else None
            return None if s is None else (np.asarray(s[0], dtype=float) - tb, np.asarray(s[1], dtype=float))

        lo = (a.t_first_anomaly if a.t_first_anomaly is not None else (a.t_detect or 0.0)) - 30.0
        hi = (a.t_impact_peak if a.t_impact_peak is not None else (a.t_disarm or lo + 60.0)) + 5.0
        if rep.interval:
            lo = max(lo, rep.interval["t0"])
        self._rng = (lo, hi)
        axes = self.fig.subplots(5, 1, sharex=True)
        for ax in axes:
            ax.set_facecolor(theme.SURFACE)
            _style_axes(ax)

        def clip(s):
            if s is None:
                return None
            m = (s[0] >= lo - 1) & (s[0] <= hi + 1)
            return s[0][m], s[1][m]

        # P1 姿態
        ax = axes[0]
        R, P, DR, DP = (clip(ser(k)) for k in ("ANG.Roll", "ANG.Pitch", "ANG.DesRoll", "ANG.DesPitch"))
        if R is None:
            R, P, DR, DP = (clip(ser(k)) for k in ("ATT.Roll", "ATT.Pitch", "ATT.DesRoll", "ATT.DesPitch"))
        if R is not None:
            ax.plot(R[0], R[1], color=theme.C_REF, lw=1.2, label="Roll")
            if DR is not None:
                ax.plot(DR[0], DR[1], color=theme.C_REF, lw=1.0, ls="--", alpha=0.7, label="目標 Roll")
            if P is not None:
                ax.plot(P[0], P[1], color=theme.C_EST, lw=1.2, label="Pitch")
            if DP is not None:
                ax.plot(DP[0], DP[1], color=theme.C_EST, lw=1.0, ls="--", alpha=0.7, label="目標 Pitch")
            ax.legend(loc="upper left", ncols=4, fontsize=7)
        ax.set_ylabel("姿態(°)")

        # P2 高度 / 油門
        ax = axes[1]
        alt, crt, tho = clip(ser("CTUN.Alt")), clip(ser("CTUN.CRt")), clip(ser("CTUN.ThO"))
        if alt is not None:
            base = np.nanmedian(alt[1][alt[0] <= (rep.interval["t0"] + 3.0 if rep.interval else alt[0][0] + 3.0)]) \
                if rep.interval else 0.0
            ax.plot(alt[0], alt[1] - (base if np.isfinite(base) else 0.0), color=theme.C_REF, lw=1.3, label="高度(−解鎖時)")
        ax.set_ylabel("高度(m)")
        if tho is not None:
            ax2 = ax.twinx()
            ax2.plot(tho[0], tho[1], color=theme.C_ERR2, lw=1.1, label="油門 ThO")
            ax2.set_ylabel("ThO", color=theme.C_ERR2)
            ax2.set_ylim(0, 1)
            ax2.tick_params(colors=theme.MUTED)
        ax.legend(loc="upper left", fontsize=7)

        # P3 馬達
        ax = axes[2]
        tb_rc = ix.table("RCOU", None) if ix is not None else None
        n_drawn = 0
        if tb_rc is not None:
            t = np.asarray(tb_rc["t"], dtype=float) - tb
            m = (t >= lo - 1) & (t <= hi + 1)
            for k in range(1, 9):
                c = tb_rc.get(f"C{k}")
                if c is None:
                    continue
                v = np.asarray(c, dtype=float)
                if np.nanmax(v[m]) > 1150 if m.any() else False:
                    ax.plot(t[m], v[m], lw=1.0, label=f"M{k}")
                    n_drawn += 1
        if n_drawn:
            ax.legend(loc="upper left", ncols=6, fontsize=7)
        ax.set_ylabel("馬達輸出(µs)")

        # P4 撞擊加速度
        ax = axes[3]
        ax_, ay_, az_ = (clip(ser(f"IMU.Acc{c}")) for c in "XYZ")
        if ax_ is not None and ay_ is not None and az_ is not None and ax_[0].size == ay_[0].size == az_[0].size:
            mag = np.sqrt(ax_[1] ** 2 + ay_[1] ** 2 + az_[1] ** 2)
            ax.plot(ax_[0], mag, color=theme.C_ERR, lw=0.9, label="|加速度|")
            ax.axhline(20, color=theme.S_WARN, lw=0.8, ls=":")
            ax.axhline(40, color=theme.S_CRIT, lw=0.8, ls=":")
            ax.legend(loc="upper left", fontsize=7)
        ax.set_ylabel("|a| (m/s²)")

        # P5 EKF 測試比 + 來源組
        ax = axes[4]
        for key, col, lbl in (("XKF4.SV", theme.C_REF, "速度"), ("XKF4.SP", theme.C_EST, "位置"), ("XKF4.SH", theme.C_ERR2, "高度")):
            s = clip(ser(key))
            if s is not None:
                ax.plot(s[0], s[1], color=col, lw=1.0, label=lbl)
        ax.axhline(1.0, color=theme.S_CRIT, lw=0.8, ls=":")
        ss = clip(ser("XKFS.SS"))
        if ss is not None:
            ax3 = ax.twinx()
            ax3.step(ss[0], ss[1] + 1, where="post", color=theme.MUTED, lw=1.0)
            ax3.set_ylim(0.5, 3.5)
            ax3.set_yticks([1, 2, 3])
            ax3.set_ylabel("來源組", color=theme.MUTED)
            ax3.tick_params(colors=theme.MUTED)
        ax.set_ylabel("EKF 測試比")
        ax.legend(loc="upper left", ncols=3, fontsize=7)
        ax.set_xlabel("時間(秒,相對 log 起點)")

        # 因果陰影 + 錨點線
        for ax in axes:
            if a.t_warn is not None:
                ax.axvspan(lo, a.t_warn, color=theme.S_WARN, alpha=0.05, lw=0)
                if a.t_loss is not None:
                    ax.axvspan(a.t_warn, a.t_loss, color=theme.S_SERIOUS, alpha=0.08, lw=0)
                    ax.axvspan(a.t_loss, hi, color=theme.S_CRIT, alpha=0.07, lw=0)
            for name, lbl, col in _ANCHOR_LINES:
                v = getattr(a, name)
                if v is not None and lo <= v <= hi:
                    ax.axvline(v, color=col, lw=0.9, ls="--", alpha=0.9)
        for name, lbl, col in _ANCHOR_LINES:
            v = getattr(a, name)
            if v is not None and lo <= v <= hi:
                axes[0].annotate(lbl, xy=(v, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -10),
                                 textcoords="offset points", fontsize=7, color=col, rotation=90, va="top")
        axes[0].set_xlim(lo, hi)
        axes[0].set_title("失事前後證據(陰影:先兆 → 發展 → 後果)")
        self._cur = [self._animated(ax.plot([], [], color=theme.INK, lw=1.1, zorder=9,
                                            transform=ax.get_xaxis_transform())[0]) for ax in axes]
        self.fig.tight_layout()

    def update_cursor(self, t):
        x = t - self.t_base if _in_range(t - self.t_base, *self._rng) else None
        for ln in self._cur:
            _set_vline(ln, x)
        return False


class CrashPanel(QWidget):
    seekRequested = pyqtSignal(float)
    analyzeRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        head = QHBoxLayout()
        lab = QLabel("失事原因分析")
        lab.setObjectName("SectionTitle")
        head.addWidget(lab)
        head.addStretch(1)
        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("Hint")
        head.addWidget(self.lbl_status)
        self.btn_analyze = QPushButton("重新分析")
        self.btn_analyze.clicked.connect(self.analyzeRequested.emit)
        self.btn_copy = QPushButton("複製報告")
        self.btn_copy.clicked.connect(self._copy)
        self.btn_save = QPushButton("匯出…")
        self.btn_save.clicked.connect(self._save)
        for b in (self.btn_analyze, self.btn_copy, self.btn_save):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            head.addWidget(b)
        v.addLayout(head)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_summary.setStyleSheet(f"background:{theme.PANEL};border:1px solid {theme.BORDER};"
                                       f"border-radius:8px;padding:10px;color:{theme.INK};")
        v.addWidget(self.lbl_summary)

        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)

        # --- 總覽:假說 + 時間軸 ---
        over = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self._title("假說排名(選一列看證據)"))
        self.tbl_h = self._table(["#", "假說", "分數", "信心", "狀態"])
        self.tbl_h.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tbl_h.itemSelectionChanged.connect(self._on_hyp_selected)
        lv.addWidget(self.tbl_h, 2)
        self.txt_detail = QTextEdit()
        self.txt_detail.setReadOnly(True)
        lv.addWidget(self.txt_detail, 2)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self._title("關鍵時間軸(雙擊跳到播放頭;後果不是原因)"))
        self.tbl_t = self._table(["時間", "分類", "事件"])
        self.tbl_t.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tbl_t.cellDoubleClicked.connect(self._on_timeline_dbl)
        rv.addWidget(self.tbl_t, 1)
        over.addWidget(left)
        over.addWidget(right)
        over.setStretchFactor(0, 1)
        over.setStretchFactor(1, 1)
        self.tabs.addTab(over, "總覽")

        self.plot = EvidencePlot()
        self.tabs.addTab(self.plot, "證據圖")
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        mono = self.txt_report.font()
        mono.setFamily("Consolas")
        self.txt_report.setFont(mono)
        self.tabs.addTab(self.txt_report, "完整報告")

        self._rep = None
        self._ld = None
        self._timeline_abs: list[float] = []
        self.set_report(None, message="載入 log 後,在這裡分析失事 / 失效原因")

    @staticmethod
    def _title(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("Hint")
        return lab

    @staticmethod
    def _table(headers) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        return t

    # ---------------- 對外介面 ----------------
    def set_logdata(self, ld):
        self._ld = ld

    def set_busy(self, on: bool):
        self.btn_analyze.setEnabled(not on)
        if on:
            self.lbl_status.setText("分析中…")

    def set_time(self, t_abs):
        self.plot.set_time(t_abs)

    def set_report(self, rep, message: str = ""):
        self._rep = rep
        self.tbl_h.setRowCount(0)
        self.tbl_t.setRowCount(0)
        self.txt_detail.clear()
        self._timeline_abs = []
        has = rep is not None
        for b in (self.btn_copy, self.btn_save):
            b.setEnabled(has)
        if not has:
            self.lbl_summary.setText(f"<span style='color:{theme.MUTED}'>{message}</span>")
            self.txt_report.setPlainText(message)
            self.lbl_status.setText("")
            self.plot.show_placeholder(message or "沒有資料")
            return
        self.lbl_status.setText("")
        color = {"ranked": theme.S_SERIOUS, "insufficient": theme.S_WARN, "no_failure": theme.S_GOOD}.get(rep.verdict, theme.INK)
        self.lbl_summary.setText(f"<b style='color:{color}'>{_VERDICT_ZH.get(rep.verdict, '')}</b><br>{rep.summary_zh}")
        self._fill_hyp(rep)
        self._fill_timeline(rep)
        self.txt_report.setPlainText(rep.format_text())
        if rep.end_type == "not_armed" or self._ld is None:
            self.plot.show_placeholder("這份 log 沒有解鎖飛行,沒有證據圖" if rep.end_type == "not_armed" else "")
        else:
            self.plot.render(rep, self._ld)

    # ---------------- 填表 ----------------
    def _fill_hyp(self, rep):
        rows = [h for h in rep.hypotheses if h.score > 0 or h.status in ("ruled_out", "not_applicable")]
        self.tbl_h.setRowCount(len(rows))
        self._hyp_rows = rows
        for r, h in enumerate(rows):
            vals = [str(r + 1), h.title_zh, f"{h.score:.1f}" if h.score > 0 else "—", h.confidence or "—",
                    _STATUS_ZH.get(h.status, h.status)]
            for c, text in enumerate(vals):
                it = QTableWidgetItem(text)
                if h.status == "ruled_out" or h.status == "not_applicable":
                    it.setForeground(QColor(theme.MUTED))
                if h.status == "ranked" and h.score > 0:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                self.tbl_h.setItem(r, c, it)
        if rows:
            self.tbl_h.selectRow(0)

    def _fill_timeline(self, rep):
        tb = rep.t_base
        self.tbl_t.setRowCount(len(rep.timeline))
        self._timeline_abs = [e.t + tb for e in rep.timeline]
        for r, e in enumerate(rep.timeline):
            for c, text in enumerate((_clk(e.t), _TAG_ZH.get(e.tag, e.tag), e.label)):
                it = QTableWidgetItem(text)
                col = _TAG_COLOR.get(e.tag)
                if col:
                    it.setForeground(QColor(col))
                self.tbl_t.setItem(r, c, it)

    def _on_hyp_selected(self):
        rows = self.tbl_h.selectionModel().selectedRows() if self.tbl_h.selectionModel() else []
        if not rows or not getattr(self, "_hyp_rows", None):
            return
        h = self._hyp_rows[rows[0].row()]
        L = [f"<b>{h.title_zh}</b>  分數 {h.score:.1f}  {_STATUS_ZH.get(h.status, h.status)}"]
        if not h.gate_ok:
            L.append(f"<span style='color:{theme.S_WARN}'>⚠ 缺必要證據:{h.gate_note}</span>")
        for e in sorted(h.evidence, key=lambda e: -e["contrib"]):
            if e["contrib"] > 0:
                L.append(f"· <b>{e['det']}</b>({e['status']},{_TAG_ZH.get(e['tag'], e['tag'])},{_clk(e['t_onset'])};"
                         f"+{e['contrib']:.1f}):{e.get('text_zh', '')}")
        for b in h.ruled_out_basis:
            L.append(f"· 排除依據:{b}")
        if h.advice_zh and h.status in ("ranked", "contributing"):
            L.append(f"<br><b>建議:</b>{h.advice_zh}")
        self.txt_detail.setHtml("<br>".join(L))

    def _on_timeline_dbl(self, row: int, _col: int):
        if 0 <= row < len(self._timeline_abs):
            self.seekRequested.emit(self._timeline_abs[row])

    # ---------------- 匯出 ----------------
    def _copy(self):
        if self._rep is not None:
            QApplication.clipboard().setText(self._rep.format_text())
            self.lbl_status.setText("已複製報告")

    def _save(self):
        if self._rep is None:
            return
        path, flt = QFileDialog.getSaveFileName(self, "匯出失事分析報告", "crash_report.txt",
                                                "文字報告 (*.txt);;JSON (*.json)")
        if not path:
            return
        import json
        with open(path, "w", encoding="utf-8") as f:
            if path.lower().endswith(".json"):
                json.dump(self._rep.to_dict(), f, ensure_ascii=False, indent=1, default=float)
            else:
                f.write(self._rep.format_text())
        self.lbl_status.setText(f"已寫出 {path}")


_VERDICT_ZH = {"ranked": "已找出最可能原因", "insufficient": "證據不足 / 無法確定", "no_failure": "沒有失事需要解釋"}
