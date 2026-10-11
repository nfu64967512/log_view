"""主視窗:仿 plot.ardupilot.org(UAV Log Viewer)的版面。

左側分頁(ATE 分析 / 欄位 / 參數 / 訊息)、右側上方欄位圖 + 下方 3D 地圖與各分析分頁、
底部時間軸播放列 —— 地圖、欄位圖、各分析圖與訊息表都跟著同一個播放頭。
"""
from __future__ import annotations

import os
import re
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                             QDockWidget, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                             QScrollArea, QSplitter, QTabWidget, QTextEdit, QVBoxLayout,
                             QWidget)

from core import align as align_mod
from core import clock, ekfsrc, logio, pipeline, sim, sim_tuning, sync
from core import timeline as tl
from core.types import KIND_GLOBAL, KIND_LABEL, KIND_NED

from . import plots, theme
from .attitude import AttitudeIndicator
from .crash import CrashPanel
from .ekfsource import EkfSourcePanel
from .fieldplot import FieldPlot
# QtWebEngine 必須在 QApplication 建立之前載入:run() 與測試都先 import 本模組
from .map3d import Map3DWidget, build_scene
from .sidebar import FieldBrowser, MessagesPanel, ParamsPanel
from .timeline import PlaybackPanel, guard_popup_keys
from .tuning import TuningPanel
from .widgets import Card, LegendChip, StatsTable, StatTile


class _Cancelled(BaseException):
    """背景載入被取消(換檔、關閉視窗)。繼承 BaseException:解析程式碼裡的
    `except Exception` 退路(例如改走慢路徑)不會把取消吞掉。"""


class _LoadSignals(QObject):
    """背景載入執行緒 → GUI。從 Python 執行緒 emit 會自動排進主執行緒的事件迴圈。"""
    progress = pyqtSignal(int, int)      # (載入世代, 百分比)
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class _CrashSignals(QObject):
    """失事原因分析(背景執行緒)→ GUI。"""
    done = pyqtSignal(int, object)        # (世代, CrashReport)
    failed = pyqtSignal(int, str)


class _TuneSignals(QObject):
    """PID 調參分析(背景執行緒)→ GUI。"""
    done = pyqtSignal(int, object, str)   # (世代, TuningReport, 範圍說明)
    failed = pyqtSignal(int, str)


class _WheelGuard(QObject):
    """數值欄位沒有焦點時不吃滾輪:捲動左側面板時不會誤改分析區間 / 偏移(還觸發重算)。"""

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Wheel and not obj.hasFocus():
            ev.ignore()
            return True              # 不交給數值欄位;忽略的事件會往上傳給捲動區
        return False


class MainWindow(QMainWindow):
    _DIST_LABEL_REF = "參考 移動距離"
    _DIST_LABEL_EST = "估測 移動距離"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VIO / GPS ATE Studio — 飛行 log 檢視與絕對軌跡誤差分析")
        self.resize(1560, 1000)
        self.logdata = None
        self.result = None
        self._play_res = None        # 與目前設定一致的結果(計算失敗時為 None,播放不沿用舊結果)
        self._err_track = None
        self._segments = []
        self._modes = []
        # 背景載入:世代編號讓較晚選的檔案永遠勝出;舊的載入可取消
        self._load_gen = 0
        self._load_cancel: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._prev_file_label = ""
        self._loading_log = False     # set_logdata 進行中(播放列還是舊 log 的範圍)
        self._map_pending = None      # 地圖分頁沒顯示時延後建場景:(res,)
        # PID 調參分析:分頁顯示時才算、在背景執行緒算;區間一變就標成過期
        self._tune_gen = 0
        self._tune_key = None         # 已分析(或分析中)的 (log, t0, t1)
        self._tune_sig = _TuneSignals(self)
        self._tune_sig.done.connect(self._on_tune_done)
        self._tune_sig.failed.connect(self._on_tune_failed)
        self._tune_timer = QTimer(self)
        self._tune_timer.setSingleShot(True)
        self._tune_timer.setInterval(400)
        self._tune_timer.timeout.connect(self._run_tuning)
        # 失事原因分析:同樣分頁顯示時才算、背景執行緒;同一份 log 不重算
        self._crash_gen = 0
        self._crash_key = None
        self._crash_sig = _CrashSignals(self)
        self._crash_sig.done.connect(self._on_crash_done)
        self._crash_sig.failed.connect(self._on_crash_failed)
        # 數值欄位改動後稍等再重算:打字 / 按住方向鍵不會每一下都整套重算
        self._recalc = QTimer(self)
        self._recalc.setSingleShot(True)
        self._recalc.setInterval(300)
        self._recalc.timeout.connect(lambda: self.compute(quiet=True))
        self._load_sig = _LoadSignals(self)
        self._load_sig.progress.connect(self._on_load_progress)
        self._load_sig.done.connect(self._on_loaded)
        self._load_sig.failed.connect(self._on_load_failed)

        self._build_menu()
        self._build_attitude_dock()
        self._build_ui()
        self._build_play_menu()
        self._build_view_menu()
        self.statusBar().showMessage("就緒 — 請載入 .bin 記錄檔,或先用模擬資料試跑")

    # ---------------- 介面組裝 ----------------

    def _build_menu(self):
        m = self.menuBar().addMenu("檔案(&F)")
        for text, slot, key in (
            ("開啟 .bin 記錄檔…", self.on_open, QKeySequence.StandardKey.Open),
            ("載入模擬資料", self.on_load_sim, "Ctrl+D"),
            ("載入模擬資料(開機到關機一份 log、兩次 ARM)", self.on_load_sim_multi, "Ctrl+Shift+D"),
            (None, None, None),
            ("匯出 CSV…", self.on_export_csv, "Ctrl+E"),
            ("匯出報告…", self.on_export_report, "Ctrl+R"),
            ("匯出目前圖表 PNG…", self.on_export_png, "Ctrl+P"),
            (None, None, None),
            ("結束", self.close, "Ctrl+Q"),
        ):
            if text is None:
                m.addSeparator()
                continue
            a = QAction(text, self)
            a.triggered.connect(slot)
            if key:
                a.setShortcut(key)
            m.addAction(a)

        sm = m.addMenu("載入 PID 調參模擬情境")
        self.act_tune_slow = QAction("模擬一般 log 的記錄方式(10 Hz、無 PID 記錄)", self, checkable=True)
        self.act_tune_slow.setToolTip("勾選:ATT / RATE 只記 10 Hz、沒有 PIDR/PIDP/PIDY —— 看看低速 log 能判斷到哪裡")
        sm.addAction(self.act_tune_slow)
        sm.addSeparator()
        for key, desc in sim_tuning.SCENARIOS.items():
            a = QAction(f"{key} — {desc}", self)
            a.triggered.connect(lambda _=False, k=key: self.on_load_tuning_sim(k))
            sm.addAction(a)
        # 「結束」移到最後
        quit_act = next(a for a in m.actions() if a.text() == "結束")
        m.removeAction(quit_act)
        m.addSeparator()
        m.addAction(quit_act)

        self._help_menu = self.menuBar().addMenu("說明(&H)")
        a = QAction("使用說明", self)
        a.triggered.connect(self.on_help)
        self._help_menu.addAction(a)

    def _build_play_menu(self):
        """播放選單:放的是播放列的同一批 QAction(快捷鍵只在右側圖表區有焦點時生效)。"""
        m = self.menuBar().addMenu("播放(&P)")
        self.menuBar().insertMenu(self._help_menu.menuAction(), m)
        for a in self._play_actions:
            m.addAction(a)

    def _build_attitude_dock(self):
        """姿態儀:浮動小視窗(可拖到別處、可停靠回主視窗),切換分頁時仍看得到目前姿態。"""
        self.p_att = AttitudeIndicator()
        self.dock_att = QDockWidget("姿態儀", self)
        self.dock_att.setObjectName("dock_attitude")
        self.dock_att.setWidget(self.p_att)
        self.dock_att.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_att)
        self.dock_att.setFloating(True)
        self.dock_att.resize(460, 340)
        self.dock_att.hide()              # 3D 地圖 HUD 已有小姿態儀;要跨分頁常駐再從「檢視」打開
        self.dock_att.visibilityChanged.connect(self._on_att_visible)
        self._att_placed = False

    def _on_att_visible(self, vis: bool):
        if vis and self.logdata is not None:
            self.p_att.set_time(self.playback.time())

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._att_placed:          # 第一次顯示:浮動視窗放在主視窗右上角
            self._att_placed = True
            g = self.geometry()
            self.dock_att.move(g.right() - self.dock_att.width() - 24, g.top() + 90)

    def _build_view_menu(self):
        m = self.menuBar().addMenu("檢視(&V)")
        self.menuBar().insertMenu(self._help_menu.menuAction(), m)
        self.act_err_extra = QAction("欄位圖加上 ATE 3D 誤差", self, checkable=True)
        self.act_err_extra.setToolTip("把目前 ATE 結果的 3D 誤差畫進上方欄位圖,和任意 log 欄位對照")
        self.act_err_extra.toggled.connect(self._update_field_extra)
        m.addAction(self.act_err_extra)
        a = QAction("清除欄位圖", self)
        a.triggered.connect(self.fplot.clear_series)
        m.addAction(a)
        act = self.dock_att.toggleViewAction()
        act.setText("姿態儀(浮動視窗)")
        m.addAction(act)
        m.addSeparator()
        a = QAction("3D 地圖:全覽", self)
        a.triggered.connect(self.p_map.view_all)
        m.addAction(a)
        for text, page in (("切到「欄位」", self.browser), ("切到「參數」", self.params),
                           ("切到「訊息」", self.messages), ("切到「ATE 分析」", self._ate_page)):
            a = QAction(text, self)
            a.triggered.connect(lambda _=False, p=page: self.side.setCurrentWidget(p))
            m.addAction(a)

    def _build_ui(self):
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 1160])
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(10, 10, 10, 6)
        lay.addWidget(split)
        self.setCentralWidget(wrap)

    def _build_left(self) -> QWidget:
        """左側分頁:ATE 分析(原控制面板)/ 欄位 / 參數 / 訊息。"""
        self.side = QTabWidget()
        self.side.setMinimumWidth(360)
        self._ate_page = self._build_ate_panel()
        self.browser = FieldBrowser()
        self.params = ParamsPanel()
        self.messages = MessagesPanel()
        self.side.addTab(self._ate_page, "ATE 分析")
        self.side.addTab(self.browser, "欄位")
        self.side.addTab(self.params, "參數")
        self.side.addTab(self.messages, "訊息")
        self.side.setTabToolTip(1, "繪製任意 log 欄位、預設組合與運算式(上方欄位圖)")
        self.side.setTabToolTip(2, "PARM 參數:數值、預設值、飛行中的變更")
        self.side.setTabToolTip(3, "MSG 文字訊息 / 事件 / 錯誤 / 模式 / ARM;雙擊跳到該時刻")
        self.messages.seekRequested.connect(self._seek)
        return self.side

    def _build_ate_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 6, 6, 0)
        v.setSpacing(9)

        # --- 資料 ---
        c = Card("1. 資料來源")
        row = QHBoxLayout()
        b1 = QPushButton("開啟 .bin…")
        b1.clicked.connect(self.on_open)
        b2 = QPushButton("模擬資料")
        b2.setToolTip("產生一段含已知 VIO 誤差(偏航 1.5°、尺度 1.004、漂移)的方形任務,用來試跑流程。\n"
                      "「檔案」選單另有「開機到關機一份 log、兩次 ARM」的多航段模擬資料")
        b2.clicked.connect(self.on_load_sim)
        for b in (b1, b2):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # 不搶焦點:空白鍵是播放鍵,不該重新載入
        row.addWidget(b1); row.addWidget(b2)
        c.add_layout(row)
        self.lbl_file = QLabel("尚未載入")
        self.lbl_file.setObjectName("Hint")
        self.lbl_file.setWordWrap(True)
        c.add(self.lbl_file)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        c.add(self.progress)
        v.addWidget(c)

        # --- 軌跡選擇 ---
        c = Card("2. 選擇比較的兩條軌跡")
        g = QGridLayout()
        g.setSpacing(6)
        g.addWidget(QLabel("參考(基準)"), 0, 0)
        self.cmb_ref = QComboBox()
        self.cmb_ref.setToolTip("當成「真值」的軌跡,通常是 GPS")
        g.addWidget(self.cmb_ref, 0, 1)
        g.addWidget(QLabel("估測(受測)"), 1, 0)
        self.cmb_est = QComboBox()
        self.cmb_est.setToolTip("要評估的軌跡,例如 ROS2 餵進去的 VISP(vision position)")
        g.addWidget(self.cmb_est, 1, 1)
        g.setColumnStretch(1, 1)
        c.add_layout(g)
        self.chip_ref = LegendChip(theme.C_REF, "參考 —")
        self.chip_est = LegendChip(theme.C_EST, "估測 —")
        c.add(self.chip_ref)
        c.add(self.chip_est)
        self.lbl_origin = QLabel("")
        self.lbl_origin.setObjectName("Hint")
        self.lbl_origin.setWordWrap(True)
        c.add(self.lbl_origin)
        v.addWidget(c)

        # --- 對齊與同步 ---
        c = Card("3. 對齊與同步")
        g = QGridLayout()
        g.setSpacing(6)
        g.addWidget(QLabel("空間對齊"), 0, 0)
        self.cmb_align = QComboBox()
        for k, zh in align_mod.ALIGN_METHODS:
            self.cmb_align.addItem(zh, k)
        self.cmb_align.setCurrentIndex(0)
        self.cmb_align.setToolTip(
            "VIO 起始位置/朝向與 GPS 不同框時要對齊。\n"
            "SE(3):平移+旋轉;Sim(3):再加尺度(可看出 VIO 尺度誤差);\n"
            "4DOF:只繞垂直軸轉(適合已知重力對齊的情形)")
        g.addWidget(self.cmb_align, 0, 1, 1, 2)

        g.addWidget(QLabel("時間偏移"), 1, 0)
        self.spn_offset = QDoubleSpinBox()
        self.spn_offset.setRange(-30.0, 30.0)
        self.spn_offset.setDecimals(3)
        self.spn_offset.setSingleStep(0.01)
        self.spn_offset.setSuffix(" s")
        self.spn_offset.setToolTip("加在估測軌跡時間戳上的偏移")
        g.addWidget(self.spn_offset, 1, 1)
        self.btn_auto_off = QPushButton("自動估")
        self.btn_auto_off.setToolTip("在分析區間內用兩條軌跡的速度剖面互相關,自動找最佳時間偏移")
        self.btn_auto_off.clicked.connect(self.on_auto_offset)
        g.addWidget(self.btn_auto_off, 1, 2)

        # 兩位小數:飛行段起訖要能原樣放進來,時間軸才認得出「目前區間 = 哪個 ARM 段」
        g.addWidget(QLabel("分析起點"), 2, 0)
        self.spn_t0 = QDoubleSpinBox()
        self.spn_t0.setRange(0, 1e6); self.spn_t0.setDecimals(2); self.spn_t0.setSuffix(" s")
        g.addWidget(self.spn_t0, 2, 1, 1, 2)
        g.addWidget(QLabel("分析終點"), 3, 0)
        self.spn_t1 = QDoubleSpinBox()
        self.spn_t1.setRange(0, 1e6); self.spn_t1.setDecimals(2); self.spn_t1.setSuffix(" s")
        g.addWidget(self.spn_t1, 3, 1, 1, 2)
        for spn in (self.spn_t0, self.spn_t1):
            spn.setToolTip("秒,相對 log 起點。也可在下方時間軸拖曳區間邊界、"
                           "或從「飛行段」選單直接選某次 ARM")

        g.addWidget(QLabel("內插間隙上限"), 4, 0)
        self.spn_gap = QDoubleSpinBox()
        self.spn_gap.setRange(0.02, 10.0); self.spn_gap.setDecimals(2)
        self.spn_gap.setValue(0.5); self.spn_gap.setSuffix(" s")
        self.spn_gap.setToolTip("估測軌跡若有資料空洞,超過此長度的區段不參與比較")
        g.addWidget(self.spn_gap, 4, 1, 1, 2)
        g.setColumnStretch(1, 1)
        c.add_layout(g)

        self.chk_auto = QCheckBox("改動設定時自動重算")
        self.chk_auto.setChecked(True)
        c.add(self.chk_auto)
        v.addWidget(c)

        self.btn_run = QPushButton("計算 ATE")
        self.btn_run.setObjectName("Primary")
        self.btn_run.setMinimumHeight(38)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.compute)
        v.addWidget(self.btn_run)

        # --- 結果磚 ---
        c = Card("結果")
        g = QGridLayout()
        g.setSpacing(8)
        self.tile_3d = StatTile("ATE RMSE(3D)")
        self.tile_2d = StatTile("ATE RMSE(水平)")
        self.tile_p95 = StatTile("3D P95")
        self.tile_max = StatTile("3D 最大")
        g.addWidget(self.tile_3d, 0, 0); g.addWidget(self.tile_2d, 0, 1)
        g.addWidget(self.tile_p95, 1, 0); g.addWidget(self.tile_max, 1, 1)
        # 移動距離:標籤隨所選來源改名(GPS / VISP …),見 _show_result
        self.tile_dist_ref = StatTile(self._DIST_LABEL_REF, decimals=1)
        self.tile_dist_est = StatTile(self._DIST_LABEL_EST, decimals=1)
        g.addWidget(self.tile_dist_ref, 2, 0); g.addWidget(self.tile_dist_est, 2, 1)
        c.add_layout(g)
        self.lbl_align_info = QLabel("")
        self.lbl_align_info.setObjectName("Hint")
        self.lbl_align_info.setWordWrap(True)
        c.add(self.lbl_align_info)
        v.addWidget(c)

        row = QHBoxLayout()
        for text, slot in (("匯出 CSV", self.on_export_csv),
                           ("匯出報告", self.on_export_report),
                           ("存圖 PNG", self.on_export_png)):
            b = QPushButton(text)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)
        v.addStretch(1)

        for w in (self.cmb_ref, self.cmb_est, self.cmb_align):
            w.currentIndexChanged.connect(self._on_setting_changed)
        self._wheel_guard = _WheelGuard(self)
        for w in (self.spn_offset, self.spn_t0, self.spn_t1, self.spn_gap):
            w.setKeyboardTracking(False)          # 打完(Enter / 離開欄位)才算一次
            w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            w.installEventFilter(self._wheel_guard)
            w.valueChanged.connect(self._on_spin_changed)
        self.cmb_ref.currentIndexChanged.connect(self._update_timeline_rows)
        self.cmb_est.currentIndexChanged.connect(self._update_timeline_rows)

        sc = QScrollArea()
        sc.setWidget(panel)
        sc.setWidgetResizable(True)
        sc.setFrameShape(QScrollArea.Shape.NoFrame)
        return sc

    def _build_right(self) -> QWidget:
        self.tabs = QTabWidget()
        self.p_map = Map3DWidget()
        self.p_overview = plots.OverviewPlot()
        self.p_play = plots.PlaybackPlot()
        self.p_top = plots.TopViewPlot()
        self.p_heat = plots.HeatmapPlot()
        self.p_3d = plots.Plot3D()
        self.p_err = plots.ErrorTimePlot()
        self.p_axis = plots.AxisErrorPlot()
        self.p_dist = plots.DistributionPlot()
        self.tabs.addTab(self.p_map, "3D 地圖")
        self.p_tune = TuningPanel()
        self.p_tune.seekRequested.connect(self._seek)
        self.p_tune.analyzeRequested.connect(lambda: self._run_tuning(force=True))
        self.p_tune.set_report(None, message="載入 log 後,在這裡分析 Roll / Pitch / Yaw 的追蹤與震盪,給 PID 調整方向")
        self.tabs.addTab(self.p_tune, "PID 調參")
        self.p_crash = CrashPanel()
        self.p_crash.seekRequested.connect(self._seek)
        self.p_crash.analyzeRequested.connect(lambda: self._run_crash(force=True))
        self.p_crash.set_report(None, message="載入 log 後,切到這個分頁分析失事 / 失效原因")
        self.tabs.addTab(self.p_crash, "失事分析")
        self.tabs.addTab(self.p_overview, "資料總覽")
        self.tabs.addTab(self.p_play, "軌跡播放")
        self.tabs.addTab(self.p_top, "平面軌跡")
        self.tabs.addTab(self.p_heat, "誤差熱點")
        self.tabs.addTab(self.p_3d, "3D 軌跡")
        self.tabs.addTab(self.p_err, "誤差時序")
        self.tabs.addTab(self.p_axis, "軸向誤差")
        self.tabs.addTab(self.p_dist, "誤差分佈")
        self.p_map.seeked.connect(self._seek)

        tbl = QWidget()
        tv = QVBoxLayout(tbl)
        tv.setContentsMargins(12, 12, 12, 12)
        lab = QLabel("ATE 統計(公尺)— 圖表的表格版")
        lab.setObjectName("SectionTitle")
        tv.addWidget(lab)
        self.table = StatsTable()
        tv.addWidget(self.table)
        hint = QLabel("「平均 / 中位數」在軸向欄位保留正負號(代表系統性偏差);"
                      "「P90 / P95 / 最大」取絕對值。")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        tv.addWidget(hint)
        tv.addStretch(1)
        self.tabs.addTab(tbl, "統計表")
        self.p_ekf = EkfSourcePanel()
        self.tabs.addTab(self.p_ekf, "EKF 來源")

        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setPlainText("計算後在此顯示完整文字報告。")
        self.tabs.addTab(self.txt_report, "報告")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # 上方欄位圖(UAV Log Viewer 式):沒畫任何欄位時收起來
        self.fplot = FieldPlot()
        self.fplot.seeked.connect(self._seek)
        self.fplot.error.connect(lambda msg: self.statusBar().showMessage(f"⚠ {msg}", 8000))
        self.fplot.seriesChanged.connect(self._on_series_changed)
        self.browser.bind(self.fplot)
        self.vsplit = QSplitter(Qt.Orientation.Vertical)
        self.vsplit.addWidget(self.fplot)
        self.vsplit.addWidget(self.tabs)
        self.vsplit.setCollapsible(0, True)
        self.vsplit.setCollapsible(1, False)
        self.vsplit.setStretchFactor(0, 2)
        self.vsplit.setStretchFactor(1, 3)
        self.vsplit.setSizes([0, 1000])

        # 時間軸播放列:一份 log 常涵蓋整次開機(多次 ARM、長段地面待機),
        # 在這裡拖曳 / 播放找出飛行段,並直接把它設成分析區間
        self.playback = PlaybackPanel()
        self.playback.timeChanged.connect(self._on_time)
        self.playback.windowRequested.connect(self._on_window_requested)
        self.playback.viewChanged.connect(self.fplot.set_view)
        self.fplot.viewChanged.connect(self._on_fplot_view)
        self.vsplit.splitterMoved.connect(self._on_vsplit_moved)
        self.playback.clear()

        right = QWidget()
        # 點右側任何地方(圖、分頁列、播放鈕)焦點都落到右側,播放快捷鍵才會生效
        right.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)
        rv.addWidget(self.vsplit, 1)
        rv.addWidget(self.playback, 0)
        self._play_actions = self.playback.install_shortcuts(right)
        guard_popup_keys(right)
        return right

    def _result_plots(self):
        return (self.p_top, self.p_heat, self.p_3d, self.p_err, self.p_axis, self.p_dist)

    # ---------------- 資料載入 ----------------

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇 ArduPilot 記錄檔", "", "DataFlash 記錄檔 (*.bin *.BIN);;所有檔案 (*.*)")
        if path:
            self.load_bin(path)

    def _cancel_load(self):
        if self._load_cancel is not None:
            self._load_cancel.set()
            self._load_cancel = None
        self.progress.setVisible(False)

    def load_bin(self, path: str):
        """在背景執行緒解析 .bin;期間舊 log 照常可用,較晚選的檔案永遠勝出。"""
        if self._load_cancel is None:
            self._prev_file_label = self.lbl_file.text()
        self._cancel_load()
        self._load_gen += 1
        gen = self._load_gen
        cancel = threading.Event()
        self._load_cancel = cancel
        self.playback.pause()
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.lbl_file.setText(f"解析中:{os.path.basename(path)}")
        self.statusBar().showMessage(f"解析 {os.path.basename(path)} …")
        sig = self._load_sig

        def progress_cb(pct):
            if cancel.is_set():
                raise _Cancelled
            sig.progress.emit(gen, int(pct))

        def run():
            try:
                ld = logio.load_bin(path, progress_cb=progress_cb)
            except _Cancelled:
                return
            except Exception as e:  # noqa: BLE001
                if not cancel.is_set():
                    sig.failed.emit(gen, f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
                return
            if not cancel.is_set():
                sig.done.emit(gen, ld)

        # daemon:關閉視窗時不必等解析結束(QThread 在執行中被銷毀會讓程式異常終止)
        self._thread = threading.Thread(target=run, name="load_bin", daemon=True)
        self._thread.start()

    def _on_load_progress(self, gen: int, pct: int):
        if gen == self._load_gen and self._load_cancel is not None:
            self.progress.setValue(pct)

    def _on_loaded(self, gen: int, logdata):
        if gen != self._load_gen:
            return                              # 較舊的載入:使用者後來又選了別的檔
        self._load_cancel = None
        self._thread = None
        self.progress.setVisible(False)
        self.set_logdata(logdata)

    def _on_load_failed(self, gen: int, msg: str):
        if gen != self._load_gen:
            return
        self._load_cancel = None
        self._thread = None
        self.progress.setVisible(False)
        if self.logdata is not None:
            self.lbl_file.setText(self._prev_file_label
                                  + "<br><span style='color:%s'>⚠ 新檔載入失敗,仍顯示先前的 log</span>"
                                  % theme.S_WARN)
        else:
            self.lbl_file.setText("載入失敗")
        self.btn_run.setEnabled(self.logdata is not None)
        self.statusBar().showMessage("載入失敗")
        QMessageBox.critical(self, "無法解析記錄檔", msg[:2000])

    def on_load_sim(self):
        self._load_direct(sim.make_sim_logdata())

    def on_load_sim_multi(self):
        self._load_direct(sim.make_multi_flight_sim()[0])

    def on_load_tuning_sim(self, scenario: str):
        slow = self.act_tune_slow.isChecked()
        ld, _info = sim_tuning.make_tuning_sim(scenario, fast_log=not slow, pid_log=not slow)
        self._load_direct(ld)
        self.tabs.setCurrentWidget(self.p_tune)

    def _load_direct(self, logdata):
        self._cancel_load()
        self._load_gen += 1                     # 進行中的背景載入作廢
        self._thread = None
        self.set_logdata(logdata)

    def _display_origin(self):
        """總覽用的 ENU 原點:ORGN,否則第一個經緯度來源的第一點。"""
        if self.logdata.origin is not None:
            return self.logdata.origin
        for s in self.logdata.sources.values():
            if s.kind == KIND_GLOBAL:
                return (float(s.lat[0]), float(s.lng[0]), float(s.alt[0]))
        return None

    def set_logdata(self, logdata):
        self._loading_log = True
        try:
            self._set_logdata(logdata)
        finally:
            self._loading_log = False
        # 播放列換成新 log 之後才同步欄位圖(換 log 途中播放列還是舊 log 的範圍)
        if self._field_plot_visible():
            lo, hi = self.playback.view_range()
            if hi > lo:
                self.fplot.set_view(lo, hi)
            self.fplot.set_time(self.playback.time())
        self.compute()

    def _set_logdata(self, logdata):
        self.playback.pause()
        self._recalc.stop()
        self._map_pending = None
        self._tune_gen += 1                    # 舊 log 的分析結果作廢
        self._tune_key = None
        self.p_tune.set_busy(False)
        self.p_tune.set_report(None, message="切到這個分頁時開始分析")
        self._crash_gen += 1                   # 舊 log 的失事分析作廢
        self._crash_key = None
        self.p_crash.set_busy(False)
        self.p_crash.set_logdata(logdata)
        self.p_crash.set_report(None, message="切到這個分頁時開始分析失事 / 失效原因")
        self.logdata = logdata
        self.result = None
        self._play_res = None
        self._err_track = None
        self._segments = logdata.segments
        self._modes = tl.mode_changes(logdata.events)
        names = list(logdata.sources)

        for cmb in (self.cmb_ref, self.cmb_est):
            cmb.blockSignals(True)
            cmb.clear()
            for n in names:
                src = logdata.sources[n]
                cmb.addItem(f"{n} — {KIND_LABEL[src.kind]}, {src.n} 筆 @ {src.rate_hz:.0f} Hz", n)
        # 預設:GPS 當參考、VISP(vision position)當估測
        ref_i = next((i for i, n in enumerate(names) if n.upper().startswith("GPS")), 0)
        est_i = next((i for i, n in enumerate(names) if "VIS" in n.upper()),
                     1 if len(names) > 1 else 0)
        if est_i == ref_i and len(names) > 1:
            est_i = (ref_i + 1) % len(names)
        self.cmb_ref.setCurrentIndex(ref_i)
        self.cmb_est.setCurrentIndex(est_i)
        for cmb in (self.cmb_ref, self.cmb_est):
            cmb.blockSignals(False)

        # 時間軸涵蓋到最後一筆訊息(t_last);ATE 只會用到兩條軌跡重疊的部分
        dur = logdata.duration
        span = max(0.0, logdata.t_last - logdata.t_base)
        for spn, val in ((self.spn_t0, 0.0), (self.spn_t1, span)):
            spn.blockSignals(True)
            spn.setRange(0.0, max(span, 1.0))
            spn.setValue(val)
            spn.blockSignals(False)
        self.spn_offset.blockSignals(True)
        self.spn_offset.setValue(0.0)
        self.spn_offset.blockSignals(False)

        name = os.path.basename(logdata.path) if logdata.path else "模擬資料"
        if logdata.path.startswith("("):
            name = logdata.path.strip("()")
        seg_txt = ""
        if self._segments:
            seg_txt = (f"<br>{len(self._segments)} 個 ARM 飛行段"
                       + (" — 可在下方時間軸選段" if len(self._segments) > 1 else ""))
        veh = getattr(logdata.index, "vehicle", "") if logdata.index is not None else ""
        wc = clock.wall_clock(logdata)
        when = ""
        if wc is not None:
            a, b = wc.local_at(logdata.t_base), wc.local_at(logdata.t_last)
            when = (f"<br>記錄時間:{clock.format_local(a)} ~ "
                    f"{clock.format_local(b, a.date() != b.date())}(本機時區)")
        self.lbl_file.setText(
            f"<b style='color:{theme.INK}'>{name}</b><br>"
            f"{len(names)} 個位置來源,時長 {dur:.0f} 秒"
            + (f",機型 {veh}" if veh and veh != "unknown" else "")
            + when
            + (f"<br>原點:{logdata.origin_desc}" if logdata.origin_desc else "")
            + seg_txt)
        self._prev_file_label = self.lbl_file.text()
        self.lbl_origin.setText("")
        self.btn_run.setEnabled(True)

        self.p_overview.draw_logdata(logdata, self._display_origin())
        self.p_ekf.set_logdata(logdata)
        self.p_att.set_logdata(logdata)
        for p in self._result_plots():
            p.show_placeholder()
        self.table.clear_values()
        self._clear_result_tiles()
        self.lbl_align_info.setText("")
        self.txt_report.setPlainText("計算後在此顯示完整文字報告。")

        # 播放 / 地圖先清掉(舊圖不能被新 log 的播放頭驅動);最後的 compute()
        # 會再畫一次 —— 不在這裡先畫,大 log 才不會整套畫兩遍
        self.p_play.show_placeholder("")
        self.p_map.set_scene(None)

        # UAV Log Viewer 式的欄位圖與側欄面板
        self.fplot.set_log(logdata)
        self.browser.set_index(logdata.index)
        self.params.set_index(logdata.index, logdata.t_base)
        self.messages.set_log(logdata)

        self.playback.set_log(logdata)          # 播放頭回到 log 起點(會發 timeChanged)
        self._update_timeline_rows()
        self.playback.set_error(None, None)
        self._sync_window()
        ref, est = self._names()
        on_map = self.p_map.reason != "nowebengine" and self._placeable(ref, est)
        self.tabs.setCurrentWidget(self.p_map if on_map else self.p_overview)

        msgs = " | ".join(logdata.notes[:3])
        self.statusBar().showMessage(
            f"已載入 {name}:{', '.join(names)}" + (f"  ⚠ {msgs}" if msgs else ""))
        self._update_chips()

    def _draw_play_raw(self, ref, est):
        if ref and est and ref != est:
            self.p_play.draw_playback(self.logdata, ref, est, None)
        else:
            self.p_play.show_placeholder("需要兩個不同的位置來源才能播放比較")

    def _placeable(self, ref, est) -> bool:
        """這份 log 能不能放上地球(有地理原點或經緯度來源)。"""
        ld = self.logdata
        if ld is None or ref not in ld.sources:
            return False
        if ld.origin is not None:
            return True
        r = ld.sources[ref]
        e = ld.sources.get(est, r)
        return pipeline.resolve_origin(ld, r, e)[0] is not None

    def _refresh_map(self, res):
        """更新 3D 地圖場景;地圖分頁沒顯示時先記下,切過去才建(大 log 建場景要零點幾秒)。"""
        if self.logdata is None:
            self._map_pending = None
            self.p_map.set_scene(None)
            return
        if self.tabs.currentWidget() is not self.p_map:
            self._map_pending = (res,)
            return
        self._map_pending = None
        ref, est = self._names()
        try:
            scene = build_scene(self.logdata, ref, est, res) if ref else None
        except Exception:  # noqa: BLE001 — 地圖準備失敗不能拖垮其他分頁
            traceback.print_exc()
            scene = None
        if scene is None and ref and not self._placeable(ref, est):
            self.p_map.set_scene(None, reason="noorigin")
        else:
            self.p_map.set_scene(scene)

    # ---------------- 時間軸 / 播放 ----------------

    def _update_timeline_rows(self):
        """時間軸迷你圖:參考軌跡的高度與水平速度 — 一眼看出哪裡在飛、哪裡停在地上。"""
        if self.logdata is None:
            return
        ld = self.logdata
        src = ld.sources.get(self.cmb_ref.currentData())
        rows = []
        if src is not None:
            # 原點與 ATE / 軌跡播放相同(resolve_origin),高度讀數才會一致
            est = ld.sources.get(self.cmb_est.currentData(), src)
            origin = pipeline.resolve_origin(ld, src, est)[0]
            if origin is None and src.kind == KIND_GLOBAL:
                origin = self._display_origin()
            try:
                p = src.enu(origin)
            except ValueError:
                p = None
            if p is not None:
                tm, spd = tl.horizontal_speed(src.t, p)
                rows = [("高度", src.t, p[:, 2], theme.C_REF, "m"),
                        ("速度", tm, spd, theme.C_REF, "m/s")]
        self.playback.set_rows(rows)

    def _window_abs(self) -> tuple[float, float]:
        tb = self.logdata.t_base if self.logdata is not None else 0.0
        return tb + self.spn_t0.value(), tb + self.spn_t1.value()

    def _sync_window(self):
        if self.logdata is None:
            return
        t0, t1 = self._window_abs()
        if t1 > t0:                             # 起訖顛倒時不把它當成有效區間顯示
            self.playback.set_window(t0, t1)
            if self.tabs.currentWidget() is self.p_tune:
                self._tune_timer.start()        # 分頁開著:區間停手後重新分析

    def _on_window_requested(self, t0: float, t1: float):
        """時間軸 / 飛行段選單要求改分析區間(log 絕對秒)→ 寫回左側起訖欄位。"""
        if self.logdata is None:
            return
        tb = self.logdata.t_base
        a, b = sorted((t0 - tb, t1 - tb))
        for spn, val in ((self.spn_t0, a), (self.spn_t1, b)):
            spn.blockSignals(True)
            spn.setValue(min(max(val, spn.minimum()), spn.maximum()))
            spn.blockSignals(False)
        self._on_setting_changed()
        if not self.chk_auto.isChecked():
            self.statusBar().showMessage(
                f"分析區間改為 {a:.1f} ~ {b:.1f} s — 按「計算 ATE」套用")

    def _seek(self, t: float):
        self.playback.seek(t)

    def _on_tab_changed(self, _i: int = -1):
        w = self.tabs.currentWidget()
        if w is self.p_map and self._map_pending is not None:
            self._refresh_map(self._map_pending[0])
        if w is self.p_tune:
            self._run_tuning()
        if w is self.p_crash:
            self._run_crash()
        if isinstance(w, (plots.PlotWidget, Map3DWidget, TuningPanel, EkfSourcePanel, CrashPanel)):
            w.set_time(self.playback.time())

    def _field_plot_visible(self) -> bool:
        return self.vsplit.sizes()[0] > 0

    def _on_time(self, t: float):
        w = self.tabs.currentWidget()
        if isinstance(w, (plots.PlotWidget, Map3DWidget, TuningPanel, EkfSourcePanel, CrashPanel)):
            w.set_time(t)
        if self._field_plot_visible():
            self.fplot.set_time(t)
        self.messages.set_time(t)
        if self.dock_att.isVisible():
            self.p_att.set_time(t)
        self.playback.set_readout(self._readout(t))

    # ---------------- PID 調參 ----------------

    def _tune_scope(self, t0: float, t1: float) -> str:
        ld = self.logdata
        tb = ld.t_base
        clock = f"{tl.format_clock(t0 - tb, 0)}–{tl.format_clock(t1 - tb, 0)}"
        i = tl.segment_matching(self._segments, t0, t1)
        if i is not None:
            return f"{self._segments[i].label} · {clock}"
        if abs(t0 - tb) < 0.05 and abs(t1 - ld.t_last) < 0.05:
            return f"整份 log · {clock}"
        return f"自訂區間 · {clock}"

    def _run_tuning(self, force: bool = False):
        """分析目前分析區間的 PID 表現(背景執行緒;同一個 log + 區間不重算)。"""
        self._tune_timer.stop()
        if self.logdata is None:
            self.p_tune.set_report(None, message="載入 log 後,在這裡分析 PID 調整方向")
            return
        t0, t1 = self._window_abs()
        if t1 <= t0:
            self.p_tune.set_report(None, message="分析區間無效(終點必須大於起點)")
            return
        key = (id(self.logdata), round(t0, 3), round(t1, 3))
        if key == self._tune_key and not force:
            return
        self._tune_key = key
        self._tune_gen += 1
        gen, ld, sig = self._tune_gen, self.logdata, self._tune_sig
        scope = self._tune_scope(t0, t1)
        self.p_tune.set_busy(True)

        def run():
            try:
                from core import tuning
                rep = tuning.analyze(ld, t0, t1)
            except Exception as e:  # noqa: BLE001
                sig.failed.emit(gen, f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
                return
            sig.done.emit(gen, rep, scope)

        threading.Thread(target=run, name="tuning", daemon=True).start()

    def _on_tune_done(self, gen: int, report, scope: str):
        if gen != self._tune_gen or self.logdata is None:
            return                              # 已過期(換了 log 或區間)
        self.p_tune.set_busy(False)
        self.p_tune.set_report(report, self.logdata.t_base, scope)
        self.p_tune.set_time(self.playback.time())
        counts = [(lbl, sum(1 for f in report.findings if f.severity == sev))
                  for sev, lbl in (("bad", "需要處理"), ("warn", "建議調整"), ("info", "可改善"))]
        parts = [f"{lbl} {n} 項" for lbl, n in counts if n]
        self.statusBar().showMessage(
            f"PID 調參分析({scope}):" + ("、".join(parts) if parts else "沒有明顯問題"), 8000)

    def _run_crash(self, force: bool = False):
        """失事原因分析(背景執行緒;同一份 log 不重算,「重新分析」強制重算)。"""
        if self.logdata is None:
            self.p_crash.set_report(None, message="載入 log 後,在這裡分析失事 / 失效原因")
            return
        key = id(self.logdata)
        if key == self._crash_key and not force:
            return
        self._crash_key = key
        self._crash_gen += 1
        gen, ld, sig = self._crash_gen, self.logdata, self._crash_sig
        self.p_crash.set_busy(True)

        def run():
            try:
                from core.crash import engine
                rep = engine.analyze(ld)
            except Exception as e:  # noqa: BLE001
                sig.failed.emit(gen, f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
                return
            sig.done.emit(gen, rep)

        threading.Thread(target=run, name="crash", daemon=True).start()

    def _on_crash_done(self, gen: int, rep):
        if gen != self._crash_gen or self.logdata is None:
            return
        self.p_crash.set_busy(False)
        self.p_crash.set_report(rep)
        self.p_crash.set_time(self.playback.time())
        self.statusBar().showMessage("失事分析:" + (rep.verdict_note or rep.summary_zh)[:80], 8000)

    def _on_crash_failed(self, gen: int, msg: str):
        if gen != self._crash_gen:
            return
        self.p_crash.set_busy(False)
        self._crash_key = None
        self.p_crash.set_report(None, message=f"分析失敗:{msg.splitlines()[0]}")
        print(msg)

    def _on_tune_failed(self, gen: int, msg: str):
        if gen != self._tune_gen:
            return
        self.p_tune.set_busy(False)
        self._tune_key = None
        self.p_tune.set_report(None, message=f"分析失敗:{msg.splitlines()[0]}")
        print(msg)                              # 完整 traceback 留在主控台,方便回報

    def _readout(self, t: float) -> str:
        # 誤差放最前面:播放列變窄時被截掉的是後面的飛行段 / 模式
        parts = []
        if self._err_track is not None:
            v = self._err_track.at(t)
            if v is not None:
                parts.append(f"3D 誤差 <b style='color:{theme.INK}'>{v[0]:.2f} m</b>")
            else:
                t0, t1 = self._window_abs()
                parts.append("誤差 —(此刻無配對樣本)" if t0 <= t <= t1 else "誤差 —(區間外)")
        if self._segments:
            i = tl.segment_index_at(self._segments, t)
            parts.append(self._segments[i].label if i is not None else "地面(未 ARM)")
        mode = tl.mode_at(self._modes, t)
        if mode:
            parts.append(mode)
        return " · ".join(parts)

    def _on_series_changed(self):
        """欄位圖有東西就展開、清空就收起(UAV Log Viewer 的行為)。"""
        has = self.fplot.has_content()
        sizes = self.vsplit.sizes()
        total = max(1, sum(sizes))
        if has and sizes[0] == 0:
            top = int(total * 0.42)
            self.vsplit.setSizes([top, total - top])
            if self._loading_log:
                return                  # 播放列還是舊 log;set_logdata 結束時再同步
            self.fplot.set_time(self.playback.time())
            lo, hi = self.playback.view_range()
            if hi > lo:
                self.fplot.set_view(lo, hi)
        elif not has and sizes[0] > 0:
            self.vsplit.setSizes([0, total])

    def _on_vsplit_moved(self, *_):
        # 欄位圖收起時不更新游標;重新拉開時補上目前的播放頭
        if self._field_plot_visible():
            self.fplot.set_time(self.playback.time())

    def _on_fplot_view(self, t0: float, t1: float):
        """欄位圖縮放 → 時間軸;時間軸有最小範圍 / log 邊界的限制,夾過的範圍回送欄位圖,兩邊才一致。"""
        self.playback.set_view(t0, t1)
        v = self.playback.view_range()
        if abs(v[0] - t0) > 1e-6 or abs(v[1] - t1) > 1e-6:
            self.fplot.set_view(*v)

    def _update_field_extra(self, *_):
        res = self._play_res
        if self.act_err_extra.isChecked() and res is not None:
            self.fplot.set_extra("ATE 3D 誤差(m)", res.t, res.e3, theme.C_ERR)
        else:
            self.fplot.set_extra(None, None, None)

    # ---------------- 計算 ----------------

    def _names(self):
        return self.cmb_ref.currentData(), self.cmb_est.currentData()

    def _update_chips(self):
        ref, est = self._names()
        self.chip_ref.set_text(f"參考 — {ref or '—'}")
        self.chip_est.set_text(f"估測 — {est or '—'}")

    def _on_setting_changed(self):
        self._update_chips()
        self._sync_window()
        if self.chk_auto.isChecked() and self.logdata is not None:
            self.compute(quiet=True)

    def _on_spin_changed(self):
        """數值欄位:時間軸區間立刻跟上,重算等使用者停手 0.3 秒(compute() 會取消排程)。"""
        self._update_chips()
        self._sync_window()
        if self.chk_auto.isChecked() and self.logdata is not None:
            self._recalc.start()

    def compute(self, quiet: bool = False):
        self._recalc.stop()
        if self.logdata is None:
            return
        ref, est = self._names()
        if not ref or not est:
            return
        if ref == est:
            self._invalidate_result("參考與估測是同一個來源,請換一個")
            return

        tb = self.logdata.t_base
        t0, t1 = self.spn_t0.value(), self.spn_t1.value()
        if t1 <= t0:
            self._invalidate_result("分析終點必須大於起點")
            return

        try:
            res = pipeline.compute(
                self.logdata, ref, est,
                t0=tb + t0, t1=tb + t1,
                offset_s=self.spn_offset.value(),
                method=self.cmb_align.currentData(),
                max_gap_s=self.spn_gap.value())
        except Exception as e:  # noqa: BLE001
            self._invalidate_result(str(e))
            if not quiet:
                QMessageBox.warning(self, "無法計算 ATE", str(e))
            return

        self.result = res
        self._show_result(res)

    def _clear_result_tiles(self):
        for t in (self.tile_3d, self.tile_2d, self.tile_p95, self.tile_max,
                  self.tile_dist_ref, self.tile_dist_est):
            t.set(None)
        self.tile_dist_ref.set_label(self._DIST_LABEL_REF)
        self.tile_dist_est.set_label(self._DIST_LABEL_EST)

    def _update_ekf_source(self, res):
        """EKF 來源分頁的洩漏係數:參考 = GPS(經緯度)、估測 = VIO(局部 NED)才算得出來。"""
        ld = self.logdata

        def is_ekf(n: str) -> bool:
            return n in ("POS", "AHR2") or n.startswith(("XKF", "XKQ"))

        ref, est = ld.sources[res.ref_name], ld.sources[res.est_name]
        if (est.kind != KIND_NED or is_ekf(res.est_name)
                or ref.kind != KIND_GLOBAL or is_ekf(res.ref_name)):
            self.p_ekf.set_leak(None, "洩漏係數需要:參考 = GPS(經緯度)、估測 = VIO(局部 NED,例如 VISP)。"
                                      "請在左側「選擇比較的兩條軌跡」選好。")
            return
        try:
            lr = ekfsrc.leak_analysis(ld, res.est_name, res.ref_name, ekfsrc.pick_ekf_source(ld),
                                      t0=res.cfg.get("t0"), t1=res.cfg.get("t1"),
                                      max_gap_s=res.cfg.get("max_gap_s", 0.5))
        except ValueError as e:
            self.p_ekf.set_leak(None, f"無法計算洩漏係數:{e}")
            return
        self.p_ekf.set_leak(lr)

    def _invalidate_result(self, msg: str):
        """目前設定算不出結果:清掉所有舊結果(不能讓舊區間的數字看起來像新區間的)。"""
        self.result = None
        self.p_ekf.set_leak(None, f"無法計算 ATE:{msg}")
        self._clear_result_tiles()
        self.table.clear_values()
        self.lbl_align_info.setText("")
        self.lbl_origin.setText("")
        self.txt_report.setPlainText(f"無法計算 ATE:{msg}")
        for p in self._result_plots():
            p.show_placeholder(f"無法計算 ATE:{msg}")
        self._set_play_result(None)
        ref, est = self._names()
        self._draw_play_raw(ref, est)
        self._refresh_map(None)
        self.statusBar().showMessage(f"⚠ {msg}")
        self._on_tab_changed()

    def _set_play_result(self, res):
        self._play_res = res
        if res is None:
            self._err_track = None
            self.playback.set_error(None, None)
        else:
            gap = max(1.0, 2 * res.cfg.get("max_gap_s", 0.5))
            self._err_track = tl.Track(res.t, np.column_stack([res.e3, res.e2]),
                                       max_gap=gap, edge=gap / 2)
            self.playback.set_error(res.t, res.e3)
        self._update_field_extra()
        self.playback.set_readout(self._readout(self.playback.time()))

    def _show_result(self, res):
        s = res.stats
        self.tile_3d.set(s["3d"]["rmse"], f"{res.n} 對樣本 · {res.duration:.0f} s", theme.INK)
        self.tile_2d.set(s["2d"]["rmse"], "東北平面")
        self.tile_p95.set(s["3d"]["p95"], "95% 樣本在此之內")
        self.tile_max.set(s["3d"]["max"], "單點最差")
        self.tile_dist_ref.set_label(f"{res.ref_name} 移動距離")
        self.tile_dist_ref.set(res.path_len_ref, "分析區間內累積 3D 路徑")
        self.tile_dist_est.set_label(f"{res.est_name} 移動距離")
        r = res.dist_ratio
        self.tile_dist_est.set(res.path_len_est,
                               f"為 {res.ref_name} 的 {r * 100:.1f}%(對齊前)" if r is not None else "")

        a = res.align_info
        scale_txt = (f",尺度 {a['scale']:.4f}(≈{(a['scale'] - 1) * 100:+.2f}%)"
                     if a["method"] == "sim3" else "")
        self.lbl_align_info.setText(
            f"對齊:{a['label']}<br>"
            f"偏航修正 {a['yaw_deg']:+.2f}°,平移 "
            f"[{a['translation'][0]:+.2f}, {a['translation'][1]:+.2f}, "
            f"{a['translation'][2]:+.2f}] m{scale_txt}")
        self.lbl_origin.setText(f"ENU 原點:{res.origin_desc}" if res.origin_desc else "")

        self.table.set_stats(s)
        self._update_ekf_source(res)
        self.txt_report.setPlainText(pipeline.format_report(res, self.logdata))
        for p in self._result_plots():
            p.render(res, self.logdata)
        self.p_play.render(res, self.logdata)
        self._refresh_map(res)
        self._set_play_result(res)
        if self.tabs.currentWidget() is self.p_overview:
            self.tabs.setCurrentWidget(self.p_top)
        self._on_tab_changed()
        self.statusBar().showMessage(
            f"{res.est_name} 相對 {res.ref_name}:3D RMSE {s['3d']['rmse']:.3f} m,"
            f"水平 {s['2d']['rmse']:.3f} m,P95 {s['3d']['p95']:.3f} m({res.n} 對樣本)")

    def on_auto_offset(self):
        if self.logdata is None:
            return
        ref, est = self._names()
        if ref == est:
            QMessageBox.warning(self, "無法估測", "參考與估測是同一個來源")
            return
        # 只在分析區間內搜尋:多航段 log 的 VIO 重置跳點會讓整份 log 的速度剖面失真
        t0, t1 = self._window_abs()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            rt, rp, et, ep = pipeline.get_enu_pair(self.logdata, ref, est, t0=t0, t1=t1)
            off, corr = sync.find_time_offset(rt, rp, et, ep, search_s=5.0)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "無法自動估測時間偏移", str(e))
            return
        QApplication.restoreOverrideCursor()
        self.spn_offset.setValue(off)
        self.statusBar().showMessage(
            f"自動時間偏移:{off * 1000:+.0f} ms(速度剖面相關係數 {corr:.3f})"
            + ("" if corr > 0.5 else " — 相關性偏低,結果僅供參考"))

    # ---------------- 匯出 ----------------
    # 存檔對話框會跑巢狀事件迴圈:期間背景載入可能完成、結果可能換掉 ——
    # 一律先抓住按下當時的結果與 log,匯出那一份

    def _need_result(self) -> bool:
        if self.result is None:
            QMessageBox.information(self, "尚無結果", "請先計算 ATE。")
            return False
        return True

    @staticmethod
    def _default_name(res, logdata, ext: str) -> str:
        base = os.path.splitext(os.path.basename(logdata.path or "sim"))[0] or "sim"
        name = f"{base}_ATE_{res.est_name}_vs_{res.ref_name}".replace("[", "").replace("]", "")
        # Windows 檔名不能有 <>:"/\|?*(模擬資料的路徑標籤含冒號)
        return re.sub(r'[<>:"/\\|?*]', "_", name).strip() + ext

    def on_export_csv(self):
        if not self._need_result():
            return
        res, ld = self.result, self.logdata
        path, _ = QFileDialog.getSaveFileName(self, "匯出逐點誤差 CSV",
                                              self._default_name(res, ld, ".csv"), "CSV (*.csv)")
        if not path:
            return
        try:
            pipeline.save_csv(res, ld, path)
        except OSError as e:
            QMessageBox.critical(self, "寫入失敗", str(e))
            return
        self.statusBar().showMessage(f"已匯出 {path}")

    def on_export_report(self):
        if not self._need_result():
            return
        res, ld = self.result, self.logdata
        path, _ = QFileDialog.getSaveFileName(self, "匯出文字報告",
                                              self._default_name(res, ld, ".txt"), "文字檔 (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(pipeline.format_report(res, ld))
        except OSError as e:
            QMessageBox.critical(self, "寫入失敗", str(e))
            return
        self.statusBar().showMessage(f"已匯出 {path}")

    def on_export_png(self):
        w = self.tabs.currentWidget()
        if not isinstance(w, (plots.PlotWidget, Map3DWidget)):
            QMessageBox.information(self, "無法存圖", "目前分頁不是圖表,請切到任一張圖再存。")
            return
        res, ld = self.result, self.logdata
        tab = re.sub(r"\s+", "", self.tabs.tabText(self.tabs.currentIndex()))
        default = self._default_name(res, ld, f"_{tab}.png") if res is not None else f"{tab}.png"
        path, _ = QFileDialog.getSaveFileName(self, "儲存圖表", default, "PNG 影像 (*.png)")
        if not path:
            return
        if self.logdata is not ld:
            self.statusBar().showMessage("⚠ 選檔期間載入了新的 log,未存圖 — 請再存一次")
            return
        try:
            if isinstance(w, Map3DWidget):
                if not w.grab_map().save(path, "PNG"):
                    raise OSError("無法寫入 PNG")
            else:
                w.save_png(path, dpi=160)
        except OSError as e:
            QMessageBox.critical(self, "寫入失敗", str(e))
            return
        self.statusBar().showMessage(f"已儲存 {path}")

    def on_help(self):
        QMessageBox.information(self, "使用說明", HELP_TEXT)

    def closeEvent(self, ev):
        self.playback.pause()
        self._cancel_load()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)      # 取消後通常立刻結束;沒結束也是 daemon,不擋關閉
        ev.accept()


HELP_TEXT = """VIO / GPS ATE Studio — 操作方式仿 plot.ardupilot.org(UAV Log Viewer)

版面
• 左側分頁:ATE 分析(兩條軌跡比較)/ 欄位(任意 log 欄位繪圖)/ 參數 / 訊息。
• 右側上方:欄位圖(畫了欄位才展開);下方:3D 地圖與各分析分頁。
• 底部:時間軸。所有圖、地圖、訊息表都跟著同一個播放頭。

3D 地圖
• 衛星底圖上的參考軌跡依飛行模式上色(也可改成依 ATE 誤差、或參考藍 / 估測橘),
  3D 機體依 ATT 的 roll / pitch / yaw 轉動;估測軌跡為細線。
• 點航跡 = 跳到那一刻;「跟隨飛機」、「全覽」在地圖上方。需要網路(Cesium 與衛星圖)。

欄位 / 參數 / 訊息
• 欄位:搜尋或展開訊息樹勾選欄位,或點「預設組合」(姿態、振動、電力、VIO / 外部導航…);
  「新增運算式」可寫 mag(VIBE.VibeX, VIBE.VibeY, VIBE.VibeZ)、degrees(...) 等。
  欄位圖背景依飛行模式上色;滾輪縮放時間、拖曳平移、點一下跳到該時刻,縮放會和時間軸同步。
• 參數:數值、預設值、哪些改過、飛行中的變更;可匯出 .param。
• 訊息:MSG / 事件 / 錯誤 / 模式 / ARM,雙擊跳到該時刻;可勾「跟隨播放頭」。

PID 調參建議(「PID 調參」分頁)
• 分析範圍 = 目前的分析區間(在時間軸選 ARM 段最準);只看離地飛行的部分。
• 每一軸比較「期望 vs 實際」:追蹤誤差、延遲、過衝、頻寬、穩態偏差,並在頻譜上找震盪;
  依症狀給出參數方向,例如 ATC_RAT_RLL_P 0.135 → 0.11(−20%)。
• 振動太大、馬達飽和時會先叫你處理那些,不會建議加大增益。
• log 只記 10 Hz 時看不到 5 Hz 以上的震盪 —— 分頁會告訴你要開哪些記錄位元(LOG_BITMASK)。
• 建議僅供參考:一次只改一個參數、每次 10–25%、先在安全環境試飛;也可以用 AutoTune。

時間軸與播放
一份 log 常是「開機到關機」而不是「一次 ARM」:前後有長段地面待機,中間可能
起降好幾次,VIO 也可能在兩次 ARM 之間重置。時間軸上:
• 綠色帶 = ARM 飛行段;「模式」列 = 依飛行模式上色;黃色三角 = VIO 重置;
  ◆ = 事件(EKF 來源切換、降落完成…);紅色 ▲ = 錯誤 / 失效保護。
• 高度 / 速度迷你圖看得出哪裡在飛;計算後多一列誤差。
• 「飛行段」選單或雙擊綠色帶 → 把分析區間設成那次 ARM(每段各自對齊才準)。
• 拖曳刻度列上的區間把手可微調;「設起點 / 設終點」把播放頭位置設為區間邊界。
• 點擊或拖曳 = 跳到該時間;滾輪縮放、Shift+滾輪平移、雙擊刻度列 = 全覽。
快捷鍵(焦點在右側時,也列在「播放」選單):空白鍵 播放/暫停、← → 前後 1 秒(加 Shift 為 10 秒)、
PgUp / PgDn 上/下一個事件、Home / End 播放範圍起訖、[ ] 設分析起點/終點。

ATE(Absolute Trajectory Error,絕對軌跡誤差)
把兩條軌跡做時間同步、(可選)空間對齊後,逐點相減得到誤差向量,再取統計量。
1. 參考選 GPS,估測選 VISP(從 ROS2 餵進去的 vision position)。
2. 空間對齊:SE(3) 平移+旋轉(最常用);Sim(3) 再加尺度;4DOF 只繞垂直軸;不對齊 = 同座標框。
3. 若兩邊時間戳有系統性延遲,按「自動估」(只在分析區間內搜尋)。
4. 起降前後的地面段會稀釋統計,用分析區間裁掉,或直接選 ARM 段。

注意
GPS 本身有 0.3~2 m 誤差,所以拿 GPS 當「真值」時,算出來的 ATE 是
「VIO 與 GPS 的差異」,不是 VIO 的絕對精度。要真值請用 RTK 或動作捕捉。
"""


def run():
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        w.load_bin(sys.argv[1])
    sys.exit(app.exec())
