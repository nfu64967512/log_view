"""側欄面板(仿 UAV Log Viewer 左側):欄位瀏覽、參數、訊息。

- FieldBrowser 「欄位」頁:搜尋、預設組合、訊息 → instance → 欄位樹(勾選 / 雙擊繪圖)、
  新欄位要放的 y 軸、「已繪製」清單(顏色 / 軸 / 顯示 / 移除)、新增運算式對話框。
- ParamsPanel 「參數」頁:篩選、與預設值比較(已修改)、記錄中變更過的參數(次數 + 歷程)、
  匯出 Mission Planner 相容的 .param、複製選取。
- MessagesPanel 「訊息」頁:MSG / EV / ERR / MODE / ARM 時間表,依種類與文字篩選,
  雙擊或 Enter 跳到該時刻,播放時標出目前時刻的那一列(可跟隨捲動)。

時間一律為 log 絕對秒;畫面上顯示相對 logdata.t_base 的 m:ss.s。
沒有 log 索引(模擬資料 / 舊格式)時各面板顯示提示,不當機。
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
from PyQt6.QtCore import (QAbstractTableModel, QEvent, QModelIndex, QSortFilterProxyModel,
                          QStringListModel, Qt, QTimer, pyqtSignal)
from PyQt6.QtGui import (QAction, QColor, QFont, QFontMetrics, QGuiApplication, QKeySequence,
                         QShortcut)
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog, QComboBox,
                             QCompleter, QDialog, QDialogButtonBox, QFileDialog,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QMenu, QPushButton, QSizePolicy, QSplitter,
                             QTableView, QToolButton, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from core import expr as cexpr
from core import presets as cpresets
from core.timeline import format_clock

from . import theme

NO_INDEX_HINT = "這份資料沒有 log 欄位索引(模擬資料或舊格式)。\n開啟 .bin 記錄檔後才能使用此頁。"
AXIS_ITEMS = [(1, "軸 1(左)"), (2, "軸 2(右)"), (3, "軸 3(右外)"), (4, "軸 4(左外)")]
_SEL_BG = "#2c4a6e"
_CUR_BG = "#22364c"          # 訊息表:播放頭所在列

# 面板共用樣式(只套在各面板自身,不動 theme.QSS)
_PANEL_QSS = f"""
QTreeWidget, QTableView, QListWidget {{
    background: {theme.SURFACE}; alternate-background-color: #1f1f1e;
    border: 1px solid {theme.BORDER}; border-radius: 6px;
    color: {theme.INK_2}; outline: none;
    selection-background-color: {_SEL_BG}; selection-color: {theme.INK};
}}
QTreeWidget::item {{ padding: 2px 0; border: none; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: #262624; }}
QTreeWidget::item:selected, QListWidget::item:selected, QTableView::item:selected {{
    background: {_SEL_BG}; color: {theme.INK};
}}
QTreeWidget::indicator, QTableView::indicator {{
    width: 13px; height: 13px; border-radius: 3px;
    border: 1px solid #55554f; background: #2a2a28;
}}
QTreeWidget::indicator:checked {{ background: {theme.C_REF}; border-color: {theme.C_REF}; }}
QTreeWidget::indicator:indeterminate {{ background: #2c4a6e; border-color: {theme.C_REF}; }}
QTableView {{ gridline-color: {theme.GRID}; }}
QTableView::item {{ padding: 0 4px; }}
QListWidget::item {{ border: none; padding: 0; }}
QLabel#PanelTitle {{ color: {theme.INK}; font-weight: 600; font-size: 10pt; background: transparent; }}
QLabel#PanelCount {{ color: {theme.MUTED}; font-size: 9pt; background: transparent; }}
QLabel#PanelHint {{ color: {theme.MUTED}; font-size: 9.5pt; padding: 18px 8px; }}
QLabel#PanelStatus {{ font-size: 9pt; }}
QPushButton {{ padding: 4px 10px; }}
QLineEdit {{ padding: 4px 8px; }}
QComboBox {{ padding: 3px 8px; }}
QToolButton#RowRemove {{
    background: transparent; border: none; color: {theme.MUTED}; font-size: 12pt; padding: 0 4px;
}}
QToolButton#RowRemove:hover {{ color: {theme.INK}; }}
QWidget#SeriesRow {{ background: transparent; }}
QWidget#SeriesRow QLabel {{ background: transparent; color: {theme.INK_2}; }}
QWidget#SeriesRow QCheckBox {{ background: transparent; }}
QWidget#SeriesRow QComboBox {{ padding: 1px 4px; min-width: 30px; font-size: 9pt; }}
"""


def _title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("PanelTitle")
    return lbl


def _count_label(shrink: bool = True) -> QLabel:
    """灰色計數 / 摘要文字;shrink 時不撐寬面板(窄側欄裡寧可截掉也不擠壓其他控制項)。"""
    lbl = QLabel("")
    lbl.setObjectName("PanelCount")
    if shrink:
        lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lbl.setMinimumWidth(0)
    return lbl


def _hint_label() -> QLabel:
    lbl = QLabel(NO_INDEX_HINT)
    lbl.setObjectName("PanelHint")
    lbl.setWordWrap(True)
    lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
    return lbl


def _roomy(w: QWidget, extra: int = 8) -> QWidget:
    """勾選框 / 按鈕多留幾像素,字型替代時文字不會被切掉。"""
    w.setMinimumWidth(w.sizeHint().width() + extra)
    return w


def _rel_clock(t: float, t_base: float, decimals: int = 1) -> str:
    return format_clock(t - t_base, decimals) if np.isfinite(t) else "—"


class _ElidedLabel(QLabel):
    """寬度不夠時中間省略的標籤(完整文字放在提示)。"""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(24)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._elide()

    def _elide(self):
        w = max(self.width() - 2, 12)
        super().setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, w))


# =====================================================================
# 欄位瀏覽
# =====================================================================

_ROLE_KEY = Qt.ItemDataRole.UserRole + 1      # 欄位 key("ATT.Roll" / "GPS[1].Spd")
_ROLE_PRESET = Qt.ItemDataRole.UserRole + 2   # 預設組合在 _presets 的索引
_ROLE_TEXT = Qt.ItemDataRole.UserRole + 3     # 篩選用的小寫文字
_ROLE_BAD = Qt.ItemDataRole.UserRole + 4      # 預設組合的提示目前標了「畫不出來」的序列
_KEY_RE = re.compile(r"^([A-Z][A-Z0-9_]*)(?:\[(\d+)\])?\.([A-Za-z_][A-Za-z0-9_]*)$")


def _split_key(key: str):
    m = _KEY_RE.match(key)
    if not m:
        return None
    return m.group(1), (int(m.group(2)) if m.group(2) is not None else None), m.group(3)


class _SeriesRow(QWidget):
    """「已繪製」清單的一列:色塊、顯示勾選、名稱、軸、移除。"""

    colorClicked = pyqtSignal(int)
    visibleToggled = pyqtSignal(int, bool)
    axisChanged = pyqtSignal(int, int)
    removeClicked = pyqtSignal(int)

    def __init__(self, i: int, s: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("SeriesRow")
        self.i = i
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 2, 2, 2)
        h.setSpacing(6)
        self.swatch = QToolButton()
        self.swatch.setFixedSize(14, 14)
        self.swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.swatch.setToolTip("變更顏色")
        self.swatch.setStyleSheet(
            f"QToolButton{{background:{s['color']};border:1px solid #00000066;border-radius:7px;}}"
            f"QToolButton:hover{{border:2px solid {theme.INK};}}")
        self.chk = QCheckBox()
        self.chk.setChecked(bool(s["visible"]))
        self.chk.setToolTip("顯示 / 隱藏")
        self.label = _ElidedLabel(s["label"])
        self.label.setToolTip(f"{s['expr']}\n軸 {s['axis']}")
        if not s["visible"]:
            self.label.setStyleSheet(f"color:{theme.MUTED};")
        self.axis = QComboBox()
        for k, name in AXIS_ITEMS:
            self.axis.addItem(f"軸{k}", k)
            self.axis.setItemData(self.axis.count() - 1, name, Qt.ItemDataRole.ToolTipRole)
        self.axis.setCurrentIndex(max(0, min(3, int(s["axis"]) - 1)))
        self.axis.setToolTip("y 軸:1 左、2 右、3 右外、4 左外")
        self.axis.setFixedWidth(58)
        self.btn_rm = QToolButton()
        self.btn_rm.setObjectName("RowRemove")
        self.btn_rm.setText("×")
        self.btn_rm.setToolTip("從圖中移除")
        self.btn_rm.setCursor(Qt.CursorShape.PointingHandCursor)
        h.addWidget(self.chk)
        h.addWidget(self.swatch)
        h.addSpacing(2)
        h.addWidget(self.label, 1)
        h.addWidget(self.axis)
        h.addWidget(self.btn_rm)
        self.swatch.clicked.connect(lambda: self.colorClicked.emit(self.i))
        self.chk.toggled.connect(lambda on: self.visibleToggled.emit(self.i, on))
        self.axis.activated.connect(lambda _x: self.axisChanged.emit(self.i, self.axis.currentData()))
        self.btn_rm.clicked.connect(lambda: self.removeClicked.emit(self.i))


class ExpressionDialog(QDialog):
    """新增運算式:欄位自動完成 + 即時檢查(core.expr.validate)。"""

    EXAMPLES = ["degrees(atan2(VISP.PY, VISP.PX))", "GPS[1].Spd*3.6",
                "mag(VIBE.VibeX, VIBE.VibeY, VIBE.VibeZ)"]

    def __init__(self, index, axis: int = 1, parent=None, text: str = ""):
        super().__init__(parent)
        self.setWindowTitle("新增運算式")
        self.setObjectName("ExpressionDialog")
        self.setStyleSheet(_PANEL_QSS)
        self.setMinimumWidth(520)
        self._index = index
        self._valid = False
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        v.addWidget(_title("運算式"))
        self.edit = QLineEdit(text)
        self.edit.setPlaceholderText("例:GPS[1].Spd*3.6")
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSizeF(10.5)
        self.edit.setFont(f)
        v.addWidget(self.edit)
        self.status = QLabel(" ")
        self.status.setObjectName("PanelStatus")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        row = QHBoxLayout()
        row.addWidget(QLabel("放到"))
        self.cmb_axis = QComboBox()
        for k, name in AXIS_ITEMS:
            self.cmb_axis.addItem(name, k)
        self.cmb_axis.setCurrentIndex(max(0, min(3, int(axis) - 1)))
        row.addWidget(self.cmb_axis)
        row.addStretch(1)
        v.addLayout(row)

        funcs = [sig for sig, _doc in getattr(cexpr, "FUNCTION_HELP", [])]
        help_txt = ("欄位寫成 訊息.欄位 或 訊息[instance].欄位;可用 + − × ÷ ** 與比較。\n"
                    "範例:" + " · ".join(self.EXAMPLES))
        if funcs:
            help_txt += "\n函式 / 常數:" + "、".join(funcs)
        hint = QLabel(help_txt)
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.btn_ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.btn_ok.setText("加入圖表")
        self.btn_ok.setObjectName("Primary")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self._try_accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)

        # 自動完成:補「游標前那個欄位 / 函式名稱」,不是整行
        words = []
        try:
            words = list(index.field_keys()) if index is not None else []
        except Exception:  # noqa: BLE001
            words = []
        words += [sig.split("(")[0] + "(" for sig in funcs if "(" in sig]
        self.completer = QCompleter(QStringListModel(words, self), self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.completer.setMaxVisibleItems(12)
        self.completer.setWidget(self.edit)
        self.completer.activated.connect(self._insert_completion)
        self.completer.popup().setStyleSheet(
            f"QListView{{background:#2a2a28;color:{theme.INK};border:1px solid {theme.BORDER};"
            f"selection-background-color:{theme.C_REF};font-family:Consolas;}}")

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(220)
        self._timer.timeout.connect(self.validate_now)
        self.edit.textChanged.connect(self._on_text_changed)
        self.edit.textEdited.connect(self._complete)
        self._on_text_changed(text)

    def expression(self) -> str:
        return self.edit.text().strip()

    def axis(self) -> int:
        return int(self.cmb_axis.currentData())

    def is_valid(self) -> bool:
        return self._valid

    def _on_text_changed(self, _text):
        self._valid = False
        self.btn_ok.setEnabled(False)
        if not self.expression():
            self._set_status("", None)
            return
        self._set_status("檢查中…", theme.MUTED)
        self._timer.start()

    def validate_now(self) -> str | None:
        """立即檢查;回傳錯誤文字(None = 正確)。"""
        self._timer.stop()
        e = self.expression()
        if not e:
            self._valid = False
            self.btn_ok.setEnabled(False)
            self._set_status("", None)
            return "運算式是空的"
        try:
            err = cexpr.validate(e, self._index)
        except Exception as ex:  # noqa: BLE001
            err = f"{type(ex).__name__}: {ex}"
        self._valid = err is None
        self.btn_ok.setEnabled(self._valid)
        if err is None:
            self._set_status("✓ 運算式有效", theme.S_GOOD)
        else:
            self._set_status(f"✗ 錯誤:{err}", theme.S_CRIT)
        return err

    def _set_status(self, text: str, color: str | None):
        self.status.setText(text or " ")
        self.status.setStyleSheet(f"color:{color};" if color else "")

    def _try_accept(self):
        if self.validate_now() is None:
            self.accept()

    def _token(self) -> tuple[int, str]:
        text = self.edit.text()
        pos = self.edit.cursorPosition()
        m = re.search(r"[A-Za-z_][A-Za-z0-9_\[\]\.]*$", text[:pos])
        return (m.start(), m.group()) if m else (pos, "")

    def _complete(self, _text):
        _start, tok = self._token()
        if len(tok) < 1:
            self.completer.popup().hide()
            return
        self.completer.setCompletionPrefix(tok)
        if self.completer.completionCount() == 0 or (
                self.completer.completionCount() == 1 and self.completer.currentCompletion() == tok):
            self.completer.popup().hide()
            return
        rect = self.edit.cursorRect()
        rect.setWidth(max(260, self.completer.popup().sizeHintForColumn(0) + 24))
        self.completer.complete(rect)

    def _insert_completion(self, word: str):
        start, _tok = self._token()
        text = self.edit.text()
        pos = self.edit.cursorPosition()
        self.edit.setText(text[:start] + word + text[pos:])
        self.edit.setCursorPosition(start + len(word))


class FieldBrowser(QWidget):
    """側欄「欄位」頁。bind(plot) 後與 FieldPlot 雙向同步。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FieldBrowser")
        self.setStyleSheet(_PANEL_QSS)
        self._index = None
        self._plot = None
        self._presets: list = []
        self._field_items: dict[str, QTreeWidgetItem] = {}
        self._pick_color = lambda initial: QColorDialog.getColor(
            QColor(initial), self, "選擇序列顏色")      # 測試可替換

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜尋訊息或欄位(例:ATT、Roll、GPS[1])")
        self.search.setClearButtonEnabled(True)
        v.addWidget(self.search)

        row = QHBoxLayout()
        row.setSpacing(6)
        lbl = QLabel("新欄位放到")
        lbl.setObjectName("Hint")
        row.addWidget(lbl)
        self.cmb_axis = QComboBox()
        for k, name in AXIS_ITEMS:
            self.cmb_axis.addItem(name, k)
        self.cmb_axis.setToolTip("勾選欄位時要畫在哪一支 y 軸(之後可在「已繪製」清單改)")
        row.addWidget(self.cmb_axis)
        row.addStretch(1)
        self.btn_expr = QPushButton("新增運算式…")
        self.btn_expr.setToolTip("自訂運算式,例如 GPS[1].Spd*3.6、degrees(atan2(VISP.PY, VISP.PX))")
        row.addWidget(self.btn_expr)
        v.addLayout(row)

        self.hint = _hint_label()
        v.addWidget(self.hint)

        # --- 三個區塊放在可拖曳的分隔器裡 ---
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.setHandleWidth(7)
        self.split.setStyleSheet("QSplitter::handle{background:transparent;}")

        box, self.lbl_presets_n = self._section("預設組合")
        self.tree_presets = QTreeWidget()
        self.tree_presets.setHeaderHidden(True)
        self.tree_presets.setColumnCount(1)
        self.tree_presets.setRootIsDecorated(True)
        self.tree_presets.setIndentation(14)
        self.tree_presets.setToolTip("點一下加入 / 移除整組(勾 = 整組已繪製,方塊 = 部分)")
        box.layout().addWidget(self.tree_presets)
        self.split.addWidget(box)

        box, self.lbl_fields_n = self._section("訊息欄位")
        self.tree_fields = QTreeWidget()
        self.tree_fields.setHeaderHidden(True)
        self.tree_fields.setColumnCount(2)
        self.tree_fields.setIndentation(14)
        self.tree_fields.setUniformRowHeights(True)
        hdr = self.tree_fields.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_fields.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        box.layout().addWidget(self.tree_fields)
        self.split.addWidget(box)

        box, self.lbl_plotted_n = self._section("已繪製")
        self.btn_clear = QPushButton("清除全部")
        self.btn_clear.setToolTip("移除圖上所有欄位序列")
        box.layout().itemAt(0).layout().addWidget(self.btn_clear)
        self.list_plotted = QListWidget()
        self.list_plotted.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list_plotted.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        box.layout().addWidget(self.list_plotted)
        self.lbl_empty = QLabel("尚未繪製任何欄位")
        self.lbl_empty.setObjectName("Hint")
        self.lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.layout().addWidget(self.lbl_empty)
        self.split.addWidget(box)
        self.split.setStretchFactor(0, 2)
        self.split.setStretchFactor(1, 5)
        self.split.setStretchFactor(2, 2)
        self.split.setSizes([170, 360, 150])
        v.addWidget(self.split, 1)

        self.status = QLabel("")
        self.status.setObjectName("PanelStatus")
        self.status.setWordWrap(True)
        self.status.hide()
        v.addWidget(self.status)
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(self.status.hide)

        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(90)
        self._filter_timer.timeout.connect(self.apply_filter)
        self.search.textChanged.connect(lambda _t: self._filter_timer.start())
        self.search.returnPressed.connect(self.apply_filter)
        self.btn_expr.clicked.connect(self.open_expression_dialog)
        self.btn_clear.clicked.connect(self._on_clear)
        self.tree_fields.itemChanged.connect(self._on_field_changed)
        self.tree_fields.itemDoubleClicked.connect(self._on_field_double)
        self.tree_fields.customContextMenuRequested.connect(self._field_menu)
        self.tree_presets.itemClicked.connect(self._on_preset_clicked)
        self.set_index(None)

    @staticmethod
    def _section(title: str):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(_title(title))
        n = _count_label(shrink=False)
        head.addWidget(n)
        head.addStretch(1)
        lay.addLayout(head)
        return box, n

    # ---------------- 公開介面 ----------------

    def bind(self, plot) -> None:
        """與 FieldPlot 連動:勾選 / 預設組合 / 已繪製清單都直接操作這張圖。"""
        if self._plot is not None:
            for sig, slot in ((self._plot.seriesChanged, self._sync_plot),
                              (self._plot.error, self.show_error)):
                try:
                    sig.disconnect(slot)
                except TypeError:
                    pass
        self._plot = plot
        if plot is not None:
            plot.seriesChanged.connect(self._sync_plot)
            plot.error.connect(self.show_error)
        self._sync_plot()

    def set_index(self, index) -> None:
        self._index = index
        has = index is not None
        self.hint.setVisible(not has)
        self.split.setVisible(has)
        for w in (self.search, self.cmb_axis, self.btn_expr):
            w.setEnabled(has)
        self._build_presets()
        self._build_fields()
        self._sync_plot()
        self.apply_filter()

    def show_error(self, text: str) -> None:
        self.status.setText(f"錯誤:{text}")
        self.status.setStyleSheet(f"color:{theme.S_CRIT};")
        self.status.show()
        self._status_timer.start(9000)

    def open_expression_dialog(self, text: str = "") -> bool:
        """開啟「新增運算式」對話框;按「加入圖表」且成功繪製才回傳 True。"""
        if self._index is None or self._plot is None:
            return False
        dlg = ExpressionDialog(self._index, self.target_axis(), self, text=text or "")
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        return bool(self._plot.add_series(dlg.expression(), dlg.axis()))

    def target_axis(self) -> int:
        return int(self.cmb_axis.currentData())

    # ---------------- 建樹 ----------------

    def _build_presets(self):
        t = self.tree_presets
        t.blockSignals(True)
        t.clear()
        self._presets = []
        if self._index is not None:
            try:
                avail = cpresets.available_presets(self._index)
                groups = cpresets.group_presets(avail)
            except Exception as e:  # noqa: BLE001
                groups = {}
                self.show_error(f"預設組合載入失敗:{e}")
            bold = QFont(t.font())
            bold.setWeight(QFont.Weight.DemiBold)
            for gname, plist in groups.items():
                g = QTreeWidgetItem([f"{gname}"])
                g.setFont(0, bold)
                g.setForeground(0, QColor(theme.INK))
                g.setData(0, _ROLE_TEXT, gname.lower())
                g.setFlags(Qt.ItemFlag.ItemIsEnabled)
                for p in plist:
                    exprs = [s.expr for s in p.series]
                    it = QTreeWidgetItem([p.name])
                    it.setData(0, _ROLE_PRESET, len(self._presets))
                    it.setData(0, _ROLE_TEXT, " ".join([gname, p.name] + exprs).lower())
                    it.setCheckState(0, Qt.CheckState.Unchecked)
                    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    it.setToolTip(0, self._preset_tip(p, set()))
                    self._presets.append(p)
                    g.addChild(it)
                t.addTopLevelItem(g)
            t.expandAll()
        t.blockSignals(False)

    @staticmethod
    def _preset_tip(p, bad: set[str]) -> str:
        axes = {getattr(s, "axis", 1) for s in p.series}
        tip = "\n".join(f"{s.expr}" + (f"(軸 {s.axis})" if len(axes) > 1 else "")
                        + ("  ← 這份 log 沒有可畫的資料" if s.expr in bad else "")
                        for s in p.series)
        return f"{p.name}\n{tip}\n\n點一下加入圖表;整組已在圖上時點一下移除"

    def _build_fields(self):
        t = self.tree_fields
        t.blockSignals(True)
        t.clear()
        self._field_items = {}
        n_fields = 0
        n_msgs = 0
        if self._index is not None:
            try:
                infos = self._index.messages()
                keys = list(self._index.field_keys())
            except Exception as e:  # noqa: BLE001
                infos, keys = {}, []
                self.show_error(f"讀取欄位清單失敗:{e}")
            tree: dict[str, dict] = {}
            for key in keys:
                sp = _split_key(key)
                if sp is None:
                    continue
                msg, inst, fld = sp
                tree.setdefault(msg, {}).setdefault(inst, []).append((fld, key))
            bold = QFont(t.font())
            bold.setWeight(QFont.Weight.DemiBold)
            muted = QColor(theme.MUTED)
            leaf_flags = (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                          | Qt.ItemFlag.ItemIsUserCheckable)
            for msg in sorted(tree):
                info = infos.get(msg)
                count = int(getattr(info, "count", 0) or 0)
                units = dict(getattr(info, "units", None) or {})
                insts = tree[msg]
                nf = sum(len(v) for v in insts.values())
                top = QTreeWidgetItem([msg, f"{count:,}" if count else ""])
                top.setFont(0, bold)
                top.setForeground(0, QColor(theme.INK))
                top.setForeground(1, muted)
                top.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                top.setData(0, _ROLE_TEXT, msg.lower())
                top.setFlags(Qt.ItemFlag.ItemIsEnabled)
                inst_txt = ""
                if any(k is not None for k in insts):
                    inst_txt = " · instance " + "、".join(str(k) for k in sorted(i for i in insts if i is not None))
                top.setToolTip(0, f"{msg} — {count:,} 筆 · {nf} 個數值欄位{inst_txt}")
                top.setToolTip(1, f"{count:,} 筆")
                n_msgs += 1
                for inst in sorted(insts, key=lambda k: -1 if k is None else k):
                    parent = top
                    if inst is not None:
                        parent = QTreeWidgetItem([f"{msg}[{inst}]", ""])
                        parent.setData(0, _ROLE_TEXT, f"{msg}[{inst}]".lower())
                        parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
                        parent.setToolTip(0, f"{msg} instance {inst}")
                        top.addChild(parent)
                    for fld, key in insts[inst]:
                        unit = str(units.get(fld, "") or "")
                        it = QTreeWidgetItem([fld, unit])
                        it.setFlags(leaf_flags)
                        it.setCheckState(0, Qt.CheckState.Unchecked)
                        it.setData(0, _ROLE_KEY, key)
                        it.setData(0, _ROLE_TEXT, fld.lower())
                        it.setForeground(1, muted)
                        it.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                        it.setToolTip(0, f"{key}\n單位:{unit or '—'}\n{msg} 共 {count:,} 筆\n"
                                         "勾選或雙擊繪圖;右鍵選擇放到哪一支軸")
                        parent.addChild(it)
                        self._field_items[key] = it
                        n_fields += 1
                t.addTopLevelItem(top)
        t.blockSignals(False)

    # ---------------- 篩選 ----------------

    def apply_filter(self) -> None:
        """依搜尋字串篩選兩棵樹(訊息名或欄位名,不分大小寫)。"""
        self._filter_timer.stop()
        q = self.search.text().strip().lower()
        self._filter_tree(self.tree_presets, q, expand_all=True)
        self._filter_tree(self.tree_fields, q, expand_all=False)
        self._update_counts(q)

    def _update_counts(self, q: str = ""):
        n_p = len(self._presets)
        n_f = len(self._field_items)
        n_m = self.tree_fields.topLevelItemCount()
        if not q:
            self.lbl_presets_n.setText(f"{n_p} 組" if n_p else "")
            self.lbl_fields_n.setText(f"{n_m} 個訊息 · {n_f} 個欄位" if n_m else "")
            return

        def shown(it):
            while it is not None:
                if it.isHidden():
                    return False
                it = it.parent()
            return True

        sp = 0
        for g in range(self.tree_presets.topLevelItemCount()):
            gi = self.tree_presets.topLevelItem(g)
            sp += sum(1 for k in range(gi.childCount()) if shown(gi.child(k)))
        sf = sum(1 for it in self._field_items.values() if shown(it))
        self.lbl_presets_n.setText(f"符合 {sp} / {n_p} 組")
        self.lbl_fields_n.setText(f"符合 {sf} / {n_f} 個欄位")

    @staticmethod
    def _filter_tree(tree: QTreeWidget, q: str, expand_all: bool):
        """訊息名稱符合 → 整個訊息留著(收合);欄位名稱符合 → 展開到該欄位。
        含「.」的查詢(例:gps[1].spd)比對完整 key。"""
        tree.setUpdatesEnabled(False)
        dotted = "." in q

        def walk(item: QTreeWidgetItem, parent_match: bool) -> tuple[bool, bool]:
            own = bool(q) and q in str(item.data(0, _ROLE_TEXT) or "")
            key = item.data(0, _ROLE_KEY)
            if q and dotted and key and q in str(key).lower():
                own = True
            n = item.childCount()
            if n == 0:
                show = not q or own or parent_match
                item.setHidden(not show)
                return show, own
            any_show = any_hit = False
            for k in range(n):
                s, h = walk(item.child(k), parent_match or own)
                any_show |= s
                any_hit |= h
            show = not q or own or parent_match or any_show
            item.setHidden(not show)
            item.setExpanded(expand_all or (bool(q) and any_hit))
            return show, own or any_hit

        for i in range(tree.topLevelItemCount()):
            walk(tree.topLevelItem(i), False)
        tree.setUpdatesEnabled(True)

    # ---------------- 與圖同步 ----------------

    def _plotted(self) -> list[dict]:
        return self._plot.series() if self._plot is not None else []

    def _sync_plot(self):
        series = self._plotted()
        exprs = {s["expr"] for s in series}
        t = self.tree_fields
        t.blockSignals(True)
        for key, it in self._field_items.items():
            want = Qt.CheckState.Checked if key in exprs else Qt.CheckState.Unchecked
            if it.checkState(0) != want:
                it.setCheckState(0, want)
        t.blockSignals(False)

        self._sync_presets(exprs)

        lst = self.list_plotted
        lst.clear()
        for i, s in enumerate(series):
            row = _SeriesRow(i, s)
            row.colorClicked.connect(self._on_row_color)
            row.visibleToggled.connect(lambda i, on: self._plot.set_series_visible(i, on))
            row.axisChanged.connect(lambda i, a: self._plot.set_series_axis(i, a))
            row.removeClicked.connect(lambda i: self._plot.remove_series(i))
            item = QListWidgetItem()
            item.setSizeHint(row.sizeHint())
            lst.addItem(item)
            lst.setItemWidget(item, row)
        lst.setVisible(bool(series))
        self.lbl_empty.setVisible(not series)
        self.btn_clear.setEnabled(bool(series))
        self.lbl_plotted_n.setText(f"{len(series)} 條" if series else "")

    def _cannot_plot(self, expr: str) -> bool:
        fn = getattr(self._plot, "is_unplottable", None)
        try:
            return bool(fn(expr)) if callable(fn) else False
        except Exception:  # noqa: BLE001
            return False

    def _preset_state(self, p, exprs) -> tuple[int, set[str]]:
        """(圖上已有的序列數, 這份 log 畫不出來的序列)。"""
        n_in = sum(1 for s in p.series if s.expr in exprs)
        bad = {s.expr for s in p.series if s.expr not in exprs and self._cannot_plot(s.expr)}
        return n_in, bad

    def _preset_complete(self, p, exprs) -> bool:
        """整組「能畫的都在圖上」:畫不出來的欄位(例如全是 NaN)不算缺。"""
        n_in, bad = self._preset_state(p, exprs)
        return bool(n_in) and n_in + len(bad) == len(p.series)

    def _sync_presets(self, exprs=None):
        """預設組合的勾選狀態:能畫的都在圖上 = 勾、部分 = 方塊、沒有 = 空。"""
        if exprs is None:
            exprs = {s["expr"] for s in self._plotted()}
        tp = self.tree_presets
        tp.blockSignals(True)
        for g in range(tp.topLevelItemCount()):
            gi = tp.topLevelItem(g)
            for k in range(gi.childCount()):
                it = gi.child(k)
                p = self._presets[int(it.data(0, _ROLE_PRESET))]
                n_in, bad = self._preset_state(p, exprs)
                it.setCheckState(0, Qt.CheckState.Checked if n_in and n_in + len(bad) == len(p.series)
                                 else Qt.CheckState.PartiallyChecked if n_in else Qt.CheckState.Unchecked)
                if bad or it.data(0, _ROLE_BAD):
                    it.setToolTip(0, self._preset_tip(p, bad))
                    it.setData(0, _ROLE_BAD, bool(bad))
        tp.blockSignals(False)

    def _on_row_color(self, i: int):
        series = self._plotted()
        if not (0 <= i < len(series)):
            return
        c = self._pick_color(series[i]["color"])
        if c is not None and QColor(c).isValid():
            self._plot.set_series_color(i, QColor(c).name())

    def _on_clear(self):
        if self._plot is not None:
            self._plot.clear_series()

    def _on_field_changed(self, item: QTreeWidgetItem, col: int):
        key = item.data(0, _ROLE_KEY)
        if col != 0 or not key or self._plot is None:
            return
        if item.checkState(0) == Qt.CheckState.Checked:
            self.plot_field(key, self.target_axis())
        else:
            self.unplot_field(key)

    def plot_field(self, key: str, axis: int | None = None) -> bool:
        if self._plot is None:
            return False
        ok = self._plot.add_series(key, axis or self.target_axis())
        if not ok:
            self._sync_plot()                 # 勾選狀態退回
        return ok

    def unplot_field(self, key: str) -> None:
        if self._plot is None:
            return
        for i in reversed([i for i, s in enumerate(self._plotted()) if s["expr"] == key]):
            self._plot.remove_series(i)

    def _on_field_double(self, item: QTreeWidgetItem, _col: int):
        if item.data(0, _ROLE_KEY):
            on = item.checkState(0) == Qt.CheckState.Checked
            item.setCheckState(0, Qt.CheckState.Unchecked if on else Qt.CheckState.Checked)

    def _on_preset_clicked(self, item: QTreeWidgetItem, _col: int):
        idx = item.data(0, _ROLE_PRESET)
        if idx is None:
            item.setExpanded(not item.isExpanded())
            return
        if self._plot is None:
            return
        p = self._presets[int(idx)]
        exprs = [s["expr"] for s in self._plotted()]
        if p.series and self._preset_complete(p, set(exprs)):
            drop = {s.expr for s in p.series}
            for i in reversed([i for i, e in enumerate(exprs) if e in drop]):
                self._plot.remove_series(i)
        elif not self._plot.add_preset(p):
            # 沒加到新序列(剩下的都畫不出來)→ 圖沒變、不會送 seriesChanged,這裡自己更新勾選狀態
            self._sync_presets()

    def _field_menu(self, pos):
        item = self.tree_fields.itemAt(pos)
        key = item.data(0, _ROLE_KEY) if item is not None else None
        if not key:
            return
        m = QMenu(self)
        plotted = key in {s["expr"] for s in self._plotted()}
        for k, name in AXIS_ITEMS:
            a = QAction(f"繪製到{name}", m)
            a.triggered.connect(lambda _c=False, k=k: self._plot_on_axis(key, k))
            m.addAction(a)
        if plotted:
            m.addSeparator()
            a = QAction("從圖中移除", m)
            a.triggered.connect(lambda: self.unplot_field(key))
            m.addAction(a)
        m.addSeparator()
        a = QAction("用這個欄位寫運算式…", m)
        a.triggered.connect(lambda: self.open_expression_dialog(key))
        m.addAction(a)
        a = QAction("複製欄位名稱", m)
        a.triggered.connect(lambda: QGuiApplication.clipboard().setText(key))
        m.addAction(a)
        m.exec(self.tree_fields.viewport().mapToGlobal(pos))

    def _plot_on_axis(self, key: str, axis: int):
        if self._plot is None:
            return
        for i, s in enumerate(self._plotted()):
            if s["expr"] == key:
                self._plot.set_series_axis(i, axis)
                return
        self.plot_field(key, axis)


# =====================================================================
# 參數
# =====================================================================

def _same(a: float, b: float) -> bool:
    """參數值比較:容許 float32 儲存誤差。"""
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)


def fmt_param(v: float) -> str:
    """參數值:整數不帶小數;其餘最多 7 位有效數字(float32 精度)。"""
    if not np.isfinite(v):
        return "nan" if np.isnan(v) else ("inf" if v > 0 else "-inf")
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    s = f"{v:.7g}"
    if "e" in s:
        return s
    return s.rstrip("0").rstrip(".") if "." in s else s


class _ParamRow:
    __slots__ = ("name", "value", "default", "modified", "history", "changes")

    def __init__(self, name, value, default, history):
        self.name = name
        self.value = float(value)
        self.default = float(default)
        self.modified = None if not np.isfinite(self.default) else not _same(self.value, self.default)
        self.history = history                # [(t, v)]:值有改變的記錄(含第一筆)
        self.changes = max(0, len(history) - 1)


class _ParamModel(QAbstractTableModel):
    HEAD = ("名稱", "數值", "預設值", "狀態")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[_ParamRow] = []
        self.t_base = 0.0
        self._mono = QFont("Consolas")
        self._mono.setStyleHint(QFont.StyleHint.Monospace)

    def set_rows(self, rows, t_base):
        self.beginResetModel()
        self.rows = rows
        self.t_base = t_base
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 4

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.HEAD[section]
        return None

    @staticmethod
    def status_text(r: _ParamRow) -> str:
        parts = []
        if r.modified:
            parts.append("已修改")
        if r.changes:
            parts.append(f"變更 {r.changes} 次")
        return " · ".join(parts)

    def history_tip(self, r: _ParamRow) -> str:
        if not r.changes:
            return ""
        lines = [f"{_rel_clock(t, self.t_base)} → {fmt_param(v)}" for t, v in r.history]
        return f"{r.name} 在記錄中變更過 {r.changes} 次:\n" + "\n".join(lines)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r = self.rows[index.row()]
        c = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if c == 0:
                return r.name
            if c == 1:
                return fmt_param(r.value)
            if c == 2:
                return fmt_param(r.default) if np.isfinite(r.default) else "—"
            return self.status_text(r)
        if role == Qt.ItemDataRole.UserRole:              # 排序鍵
            return (r.name, r.value, r.default if np.isfinite(r.default) else -np.inf,
                    (2 if r.changes else 0) + (1 if r.modified else 0))[c]
        if role == Qt.ItemDataRole.ForegroundRole:
            if c == 3:
                return QColor(theme.S_SERIOUS if r.changes else theme.S_WARN)
            if c == 2 and not np.isfinite(r.default):
                return QColor(theme.MUTED)
            if c == 1 and r.modified:
                return QColor(theme.INK)
            return None
        if role == Qt.ItemDataRole.FontRole and c in (1, 2):
            return self._mono
        if role == Qt.ItemDataRole.TextAlignmentRole and c in (1, 2):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            tip = self.history_tip(r)
            if c == 0:
                return r.name + ("\n\n" + tip if tip else "")
            if c == 2 and not np.isfinite(r.default):
                return "log 沒有記錄預設值"
            if c == 3 and r.modified and not tip:
                return f"與預設值 {fmt_param(r.default)} 不同"
            return tip or None
        return None


class _ParamFilter(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = ""
        self.only_modified = False
        self.setSortRole(Qt.ItemDataRole.UserRole)

    def set_filter(self, text: str, only_modified: bool):
        self.text = text.strip().upper()
        self.only_modified = only_modified
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        r = self.sourceModel().rows[row]
        if self.only_modified and not (r.modified or r.changes):
            return False
        return not self.text or self.text in r.name.upper()


class ParamsPanel(QWidget):
    """側欄「參數」頁。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ParamsPanel")
        self.setStyleSheet(_PANEL_QSS)
        self._index = None
        self._path = ""
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        row = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("篩選參數名稱(例:ATC_RAT、EK3)")
        self.filter.setClearButtonEnabled(True)
        row.addWidget(self.filter, 1)
        self.chk_modified = QCheckBox("只看已修改")
        self.chk_modified.setToolTip("只列出與預設值不同、或在記錄中變更過的參數")
        row.addWidget(_roomy(self.chk_modified))
        v.addLayout(row)
        self.summary = _count_label()
        v.addWidget(self.summary)
        self.hint = _hint_label()
        v.addWidget(self.hint)

        self.model = _ParamModel(self)
        self.proxy = _ParamFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(22)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setHighlightSections(False)
        v.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.btn_copy = QPushButton("複製選取")
        self.btn_copy.setToolTip("把選取的參數以「名稱,數值」複製到剪貼簿(Ctrl+C;沒選取時複製目前清單)")
        self.btn_export = QPushButton("匯出 .param…")
        self.btn_export.setToolTip("存成 Mission Planner / MAVProxy 可載入的參數檔(全部參數)")
        row.addWidget(self.btn_copy)
        row.addStretch(1)
        row.addWidget(_roomy(self.btn_export))
        v.addLayout(row)

        self.filter.textChanged.connect(self._apply)
        self.chk_modified.toggled.connect(self._apply)
        self.btn_copy.clicked.connect(self.copy_selection)
        self.btn_export.clicked.connect(self._on_export)
        sc = QShortcut(QKeySequence.StandardKey.Copy, self.table)
        sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc.activated.connect(self.copy_selection)
        self.set_index(None)

    def set_index(self, index, t_base: float | None = None) -> None:
        """t_base:歷程時間的零點(建議傳 logdata.t_base;省略時用 log 最早時間)。"""
        self._index = index
        has = index is not None
        self.hint.setVisible(not has)
        for w in (self.table, self.filter, self.chk_modified, self.btn_copy, self.btn_export):
            w.setVisible(has)
        rows = []
        if has:
            self._path = str(getattr(index, "path", "") or "")
            if t_base is None:
                try:
                    t_base = float(index.t_range()[0])
                except Exception:  # noqa: BLE001
                    t_base = 0.0
            try:
                rows = self._collect(index)
            except Exception as e:  # noqa: BLE001
                rows = []
                self.summary.setText(f"讀取參數失敗:{e}")
        self.model.set_rows(rows, float(t_base or 0.0))
        self._apply()

    @staticmethod
    def _collect(index) -> list[_ParamRow]:
        groups: dict[str, list] = {}
        for p in index.params():
            groups.setdefault(p.name, []).append(p)
        try:
            final = dict(index.param_values())
        except Exception:  # noqa: BLE001
            final = {}
        # LogIndex.params() 每個名稱只有一筆(最後的值);變更歷程要從 param_history 拿
        ph = getattr(index, "param_history", None)
        rows = []
        for name, plist in groups.items():
            plist.sort(key=lambda p: (-np.inf if not np.isfinite(p.t) else p.t))
            raw = None
            if callable(ph):
                try:
                    raw = [(float(t), float(v)) for t, v in ph(name)] or None
                except Exception:  # noqa: BLE001 — 退回用 params() 分組
                    raw = None
            if raw is None:
                raw = [(float(p.t), float(p.value)) for p in plist]
            hist = []
            for t, v in raw:
                if not hist or not (_same(v, hist[-1][1]) or (math.isnan(v) and math.isnan(hist[-1][1]))):
                    hist.append((t, v))
            value = float(final.get(name, plist[-1].value))
            default = next((float(p.default) for p in reversed(plist) if np.isfinite(p.default)),
                           float("nan"))
            rows.append(_ParamRow(name, value, default, hist))
        for name, value in final.items():               # param_values 有、params 沒有的
            if name not in groups:
                rows.append(_ParamRow(name, value, float("nan"), [(float("nan"), float(value))]))
        rows.sort(key=lambda r: r.name)
        return rows

    def _apply(self, *_):
        self.proxy.set_filter(self.filter.text(), self.chk_modified.isChecked())
        rows = self.model.rows
        if self._index is None:
            self.summary.setText("")
            return
        if not rows:
            self.summary.setText("這份 log 沒有記錄參數(PARM)")
            return
        n_mod = sum(1 for r in rows if r.modified)
        n_chg = sum(1 for r in rows if r.changes)
        txt = f"共 {len(rows):,} 個參數 · 已修改 {n_mod} · 記錄中變更 {n_chg}"
        shown = self.proxy.rowCount()
        if shown != len(rows):
            txt += f" · 顯示 {shown}"
        self.summary.setText(txt)

    def visible_names(self) -> list[str]:
        return [self.proxy.data(self.proxy.index(r, 0)) for r in range(self.proxy.rowCount())]

    def _selected_rows(self) -> list[_ParamRow]:
        sel = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if sel:
            src = sorted(self.proxy.mapToSource(i).row() for i in sel)
        else:
            src = [self.proxy.mapToSource(self.proxy.index(r, 0)).row()
                   for r in range(self.proxy.rowCount())]
        return [self.model.rows[i] for i in src]

    def copy_selection(self) -> str:
        """選取的參數(沒選取 = 目前清單)以「名稱,數值」複製到剪貼簿;回傳複製的文字。"""
        text = "\n".join(f"{r.name},{fmt_param(r.value)}" for r in self._selected_rows())
        if text:
            QGuiApplication.clipboard().setText(text)
        return text

    def export_param(self, path: str) -> int:
        """全部參數寫成 .param(每行「名稱,數值」,依名稱排序);回傳筆數。"""
        rows = sorted(self.model.rows, key=lambda r: r.name)
        with open(path, "w", encoding="utf-8", newline="\r\n") as f:
            src = os.path.basename(self._path) if self._path else ""
            f.write("# Parameters exported by VIO ATE Studio"
                    + (f" from {src}" if src.isascii() and src else "") + "\n")
            for r in rows:
                f.write(f"{r.name},{fmt_param(r.value)}\n")
        return len(rows)

    def _on_export(self):
        if not self.model.rows:
            return
        base = os.path.splitext(os.path.basename(self._path))[0] if self._path else "params"
        path, _ = QFileDialog.getSaveFileName(self, "匯出參數檔", f"{base}.param",
                                              "參數檔 (*.param *.parm);;所有檔案 (*)")
        if path:
            n = self.export_param(path)
            self.summary.setText(f"已匯出 {n} 個參數 → {os.path.basename(path)}")


# =====================================================================
# 訊息
# =====================================================================

# (種類, 顯示名稱, 顏色):狀態色一律配文字;模式用模式色
KINDS = {
    "msg": ("訊息", theme.INK_2),
    "ev": ("事件", theme.S_WARN),
    "err": ("錯誤", theme.S_CRIT),
    "mode": ("模式", None),
    "arm": ("ARM", theme.S_GOOD),
    "disarm": ("DISARM", theme.S_SERIOUS),
}
FILTERS = [("msg", "訊息", ("msg",)), ("ev", "事件", ("ev",)), ("err", "錯誤", ("err",)),
           ("mode", "模式", ("mode",)), ("arm", "ARM", ("arm", "disarm"))]


class _MsgModel(QAbstractTableModel):
    HEAD = ("時間", "類型", "內容")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.t = np.empty(0)
        self.kind: list[str] = []
        self.text: list[str] = []
        self.low: list[str] = []
        self.vis = np.empty(0, dtype=np.int64)
        self.vt = np.empty(0)
        self.cur = -1
        self.t_base = 0.0
        self._mono = QFont("Consolas")
        self._mono.setStyleHint(QFont.StyleHint.Monospace)
        self._colors = {k: QColor(c) for k, (_n, c) in KINDS.items() if c}
        self._cur_bg = QColor(_CUR_BG)

    def set_messages(self, msgs, t_base: float):
        self.beginResetModel()
        msgs = sorted(msgs, key=lambda m: m.t)
        self.t = np.array([float(m.t) for m in msgs], dtype=float)
        self.kind = [str(m.kind) for m in msgs]
        self.text = [str(m.text) for m in msgs]
        self.low = [f"{s.lower()} {KINDS.get(k, (k,))[0].lower()}" for s, k in zip(self.text, self.kind)]
        self.vis = np.arange(len(msgs), dtype=np.int64)
        self.vt = self.t.copy()
        self.cur = -1
        self.t_base = float(t_base)
        self.endResetModel()

    def apply_filter(self, kinds: set[str], text: str):
        self.beginResetModel()
        q = text.strip().lower()
        idx = [i for i, k in enumerate(self.kind) if k in kinds and (not q or q in self.low[i])]
        self.vis = np.asarray(idx, dtype=np.int64)
        self.vt = self.t[self.vis] if self.vis.size else np.empty(0)
        self.cur = -1
        self.endResetModel()

    def row_at(self, t: float) -> int:
        """最後一筆時間 ≤ t 的(篩選後)列;沒有則 -1。"""
        if t is None or not self.vt.size:
            return -1
        return int(np.searchsorted(self.vt, t, side="right")) - 1

    def set_current(self, row: int) -> bool:
        if row == self.cur:
            return False
        old, self.cur = self.cur, row
        for r in (old, row):
            if 0 <= r < self.vis.size:
                self.dataChanged.emit(self.index(r, 0), self.index(r, 2))
        return True

    def time_of(self, row: int) -> float:
        return float(self.vt[row])

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else int(self.vis.size)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 3

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.HEAD[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r = index.row()
        i = int(self.vis[r])
        c = index.column()
        k = self.kind[i]
        if role == Qt.ItemDataRole.DisplayRole:
            if c == 0:
                return ("▶ " if r == self.cur else "") + _rel_clock(self.t[i], self.t_base)
            if c == 1:
                return KINDS.get(k, (k, None))[0]
            return self.text[i]
        if role == Qt.ItemDataRole.ForegroundRole:
            if c == 1:
                col = self._colors.get(k)
                return col if col is not None else QColor(theme.mode_color(self.text[i]))
            if c == 2 and k == "mode":
                return QColor(theme.mode_color(self.text[i]))
            if c == 2 and k == "err":
                return QColor(theme.INK)
            if c == 0:
                return QColor(theme.INK if r == self.cur else theme.MUTED)
            return None
        if role == Qt.ItemDataRole.BackgroundRole and r == self.cur:
            return self._cur_bg
        if role == Qt.ItemDataRole.FontRole and c == 0:
            return self._mono
        if role == Qt.ItemDataRole.TextAlignmentRole and c == 0:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole and c == 2:
            return f"{_rel_clock(self.t[i], self.t_base)}  {self.text[i]}\n雙擊跳到這個時刻"
        return None


class MessagesPanel(QWidget):
    """側欄「訊息」頁。"""

    seekRequested = pyqtSignal(float)      # log 絕對秒

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MessagesPanel")
        self.setStyleSheet(_PANEL_QSS)
        self._t: float | None = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.kind_checks: dict[str, QCheckBox] = {}
        for key, name, _kinds in FILTERS:
            cb = QCheckBox(name)
            cb.setChecked(True)
            col = KINDS[key][1] or theme.INK_2
            cb.setStyleSheet(f"QCheckBox{{color:{col};}}")
            cb.setToolTip(f"顯示「{name}」類型")
            row.addWidget(cb)
            self.kind_checks[key] = cb
        row.addStretch(1)
        v.addLayout(row)

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("篩選內容(例:EKF、PreArm)")
        self.filter.setClearButtonEnabled(True)
        v.addWidget(self.filter)
        self.hint = _hint_label()
        v.addWidget(self.hint)

        self.model = _MsgModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(22)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.ensurePolished()
        fm_time = QFontMetrics(self.model._mono)
        fm_kind = self.table.fontMetrics()
        hh.resizeSection(0, fm_time.horizontalAdvance("▶ 00:00.0") + 18)
        hh.resizeSection(1, max(fm_kind.horizontalAdvance(n) for n, _c in KINDS.values()) + 26)
        hh.setHighlightSections(False)
        self.table.installEventFilter(self)
        self.table.setToolTip("")
        v.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.summary = _count_label()
        self.summary.setToolTip("雙擊或按 Enter 跳到該時刻")
        row.addWidget(self.summary, 1)
        self.chk_follow = QCheckBox("跟隨播放頭")
        self.chk_follow.setChecked(True)
        self.chk_follow.setToolTip("播放 / 拖曳時間軸時,捲動到目前時刻的那一列")
        row.addWidget(_roomy(self.chk_follow))
        v.addLayout(row)

        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(120)
        self._filter_timer.timeout.connect(self.apply_filter)
        self.filter.textChanged.connect(lambda _t: self._filter_timer.start())
        for cb in self.kind_checks.values():
            cb.toggled.connect(self.apply_filter)
        self.table.doubleClicked.connect(self._on_activate)
        self.chk_follow.toggled.connect(lambda on: on and self._scroll_current())
        self.set_log(None)

    # ---------------- 公開介面 ----------------

    def set_log(self, logdata) -> None:
        idx = getattr(logdata, "index", None) if logdata is not None else None
        has = idx is not None
        self.hint.setVisible(not has)
        for w in (self.table, self.filter, self.chk_follow, *self.kind_checks.values()):
            w.setVisible(has)
        msgs = []
        if has:
            try:
                msgs = list(idx.log_messages())
            except Exception as e:  # noqa: BLE001
                msgs = []
                self.summary.setText(f"讀取訊息失敗:{e}")
        self.model.set_messages(msgs, float(logdata.t_base) if logdata is not None else 0.0)
        self.apply_filter()

    def set_time(self, t: float | None) -> None:
        """標出最後一筆時間 ≤ t 的列(bisect,不重設模型);勾「跟隨播放頭」時捲過去。"""
        self._t = None if t is None else float(t)
        row = self.model.row_at(self._t) if self._t is not None else -1
        if self.model.set_current(row) and self.chk_follow.isChecked():
            self._scroll_current()

    def current_row(self) -> int:
        return self.model.cur

    def apply_filter(self, *_):
        self._filter_timer.stop()
        kinds = set()
        for key, _name, ks in FILTERS:
            if self.kind_checks[key].isChecked():
                kinds.update(ks)
        self.model.apply_filter(kinds, self.filter.text())
        total = len(self.model.kind)
        shown = self.model.rowCount()
        if self.hint.isVisible():
            self.summary.setText("")
        else:
            txt = f"共 {total:,} 筆" + (f" · 顯示 {shown:,}" if shown != total else "")
            self.summary.setText(txt + " · 雙擊跳轉")
        if self._t is not None:
            row = self.model.row_at(self._t)
            self.model.set_current(row)
            if self.chk_follow.isChecked():
                self._scroll_current()

    # ---------------- 內部 ----------------

    def _scroll_current(self):
        r = self.model.cur
        if 0 <= r < self.model.rowCount():
            self.table.scrollTo(self.model.index(r, 0), QAbstractItemView.ScrollHint.EnsureVisible)

    def _on_activate(self, index: QModelIndex):
        if index.isValid():
            self.seekRequested.emit(self.model.time_of(index.row()))

    def eventFilter(self, obj, ev):
        if obj is self.table and ev.type() == QEvent.Type.KeyPress and ev.key() in (
                Qt.Key.Key_Return, Qt.Key.Key_Enter):
            idx = self.table.currentIndex()
            if idx.isValid():
                self._on_activate(idx)
                return True
        return super().eventFilter(obj, ev)
