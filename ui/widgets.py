"""小型可重用元件:卡片、統計磚、統計表格。"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QHeaderView, QLabel,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import theme


class Card(QFrame):
    def __init__(self, title: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(12, 10, 12, 12)
        self.v.setSpacing(7)
        if title:
            lab = QLabel(title)
            lab.setObjectName("SectionTitle")
            self.v.addWidget(lab)

    def add(self, w):
        self.v.addWidget(w)
        return w

    def add_layout(self, lay):
        self.v.addLayout(lay)
        return lay


class StatTile(QFrame):
    """單一數字 + 標籤;數字本身就是重點,不畫圖。"""

    def __init__(self, label: str, unit: str = "m", decimals: int = 3, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._decimals = decimals
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 9, 12, 10)
        v.setSpacing(1)
        cap = self.cap = QLabel(label)
        cap.setObjectName("StatLabel")
        row = QHBoxLayout()
        row.setSpacing(4)
        self.value = QLabel("—")
        self.value.setObjectName("StatBig")
        u = QLabel(unit)
        u.setObjectName("StatUnit")
        u.setAlignment(Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self.value)
        row.addWidget(u)
        row.addStretch(1)
        self.sub = QLabel("")
        self.sub.setObjectName("Hint")
        v.addWidget(cap)
        v.addLayout(row)
        v.addWidget(self.sub)

    def set_label(self, label: str):
        self.cap.setText(label)

    def set(self, value: float | None, sub: str = "", accent: str | None = None):
        self.value.setText("—" if value is None else f"{value:.{self._decimals}f}")
        self.value.setStyleSheet(f"color:{accent};" if accent else "")
        self.sub.setText(sub)


_ROWS = [("rmse", "RMSE"), ("mean", "平均(帶號)"), ("median", "中位數(帶號)"),
         ("std", "標準差"), ("p90", "P90"), ("p95", "P95"), ("max", "最大")]
_COLS = ["統計量", "3D", "水平", "東 E", "北 N", "天 U"]


class StatsTable(QTableWidget):
    """ATE 統計表 — 圖表的表格版(無障礙:數值不只靠圖形傳達)。"""

    def __init__(self, parent=None):
        super().__init__(len(_ROWS), len(_COLS), parent)
        self.setHorizontalHeaderLabels(_COLS)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for c in range(1, len(_COLS)):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        self.setMinimumHeight(len(_ROWS) * 32 + 42)
        self.clear_values()

    def clear_values(self):
        for r, (_, zh) in enumerate(_ROWS):
            self._put(r, 0, zh, bold=True, align_left=True)
            for c in range(1, len(_COLS)):
                self._put(r, c, "—")

    def _put(self, r, c, text, bold=False, align_left=False, color=None):
        it = QTableWidgetItem(text)
        it.setTextAlignment(Qt.AlignmentFlag.AlignVCenter |
                            (Qt.AlignmentFlag.AlignLeft if align_left
                             else Qt.AlignmentFlag.AlignRight))
        if bold:
            f = it.font(); f.setBold(True); it.setFont(f)
        if color:
            from PyQt6.QtGui import QColor
            it.setForeground(QColor(color))
        self.setItem(r, c, it)

    def set_stats(self, stats: dict):
        s3, s2, ax = stats["3d"], stats["2d"], stats["axes"]
        for r, (key, zh) in enumerate(_ROWS):
            self._put(r, 0, zh, bold=True, align_left=True)
            vals = [s3[key], s2[key], ax["E"][key], ax["N"][key], ax["U"][key]]
            for c, v in enumerate(vals, start=1):
                col = theme.INK if key == "rmse" else None
                self._put(r, c, f"{v:+.3f}" if key in ("mean", "median") and c >= 3
                          else f"{v:.3f}", bold=(key == "rmse"), color=col)


class LegendChip(QWidget):
    """一個色塊 + 文字 — 讓身分不只靠顏色(顏色旁一定有名字)。"""

    def __init__(self, color: str, text: str, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        dot = QLabel()
        dot.setFixedSize(11, 11)
        dot.setStyleSheet(f"background:{color};border-radius:3px;")
        self.lab = QLabel(text)
        self.lab.setObjectName("Hint")
        h.addWidget(dot)
        h.addWidget(self.lab)
        h.addStretch(1)

    def set_text(self, text: str):
        self.lab.setText(text)
