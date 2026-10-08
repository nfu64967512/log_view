"""深色主題:色彩角色、matplotlib rcParams、Qt 樣式表。

色彩角色(全 app 一致,顏色跟著「實體」走而非排序):
  參考軌跡 = 藍、估測軌跡 = 橘、誤差 = 洋紅、誤差(水平)= 紫。

色板取自已驗證的深色類別色階(OKLab CVD ΔE / 對比度皆通過檢查):
  參考↔估測      CVD ΔE 26.8、normal 31.8、對比 ≥ 3:1
  誤差 3D↔水平   CVD ΔE 16.0、normal 19.7、對比 ≥ 3:1
軸向誤差改用單序列 small multiples(標題命名身分),不靠額外類別色。
"""
from __future__ import annotations

# --- 表面與文字 ---
SURFACE = "#1a1a19"      # 圖表底
PLANE = "#0d0d0d"        # 頁面底
PANEL = "#232322"        # 控制面板底
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
BORDER = "#3a3a37"

# --- 序列 ---
C_REF = "#3987e5"        # 參考軌跡
C_EST = "#d95926"        # 估測軌跡
C_ERR = "#d55181"        # 誤差(3D / 軸向 / 分佈)
C_ERR2 = "#9085e9"       # 誤差(水平)

# 連續量值色階(誤差大小):單一藍色相 600→100 步階,深色底上「暗=小、亮=大」,
# 最暗端仍有 2.15:1 對比,亮度單調遞增,色相跨度 3.7°。
SEQ_RAMP = ["#184f95", "#256abf", "#2a78d6", "#3987e5",
            "#5598e7", "#86b6ef", "#b7d3f6", "#cde2fb"]

# --- 狀態(保留色,永遠配文字標籤,不當序列色) ---
S_GOOD = "#0ca30c"
S_WARN = "#fab219"
S_SERIOUS = "#ec835a"
S_CRIT = "#d03b3b"

# --- 飛行模式(類別色,地圖航跡 / 圖表背景色帶 / 時間軸共用) ---
# 避開參考藍、估測橘、誤差洋紅,讓「模式」與「序列身分」不會混淆;
# 模式色永遠伴隨模式名稱(圖例、色帶標籤),不單靠顏色辨識。
MODE_COLORS = {
    "STABILIZE": "#a3a3a0", "MANUAL": "#a3a3a0", "QSTABILIZE": "#a3a3a0",
    "ACRO": "#ff8fb1", "QACRO": "#ff8fb1",
    "ALT_HOLD": "#5fc6e8", "QHOVER": "#5fc6e8", "FBWA": "#5fc6e8",
    "LOITER": "#2fb5a3", "QLOITER": "#2fb5a3", "POSHOLD": "#9ccc65", "FBWB": "#9ccc65",
    "AUTO": "#f2c94c", "GUIDED": "#7bd389", "CIRCLE": "#c9a66b", "CRUISE": "#c9a66b",
    "RTL": "#b28dff", "QRTL": "#b28dff", "SMART_RTL": "#8e7cc3",
    "LAND": "#e05d5d", "QLAND": "#e05d5d", "BRAKE": "#d4d46a", "TAKEOFF": "#d4d46a",
}
_MODE_FALLBACK = ["#6fd3c1", "#e6a3ff", "#c2e07a", "#ffd18c", "#8fb8ff",
                  "#ff9e9e", "#b5d6a7", "#d9c3ff"]


def mode_color(name: str | None) -> str:
    """飛行模式 → 固定顏色(未列名的模式依名稱雜湊取備用色,跨執行一致)。"""
    if not name:
        return MUTED
    key = name.upper()
    if key in MODE_COLORS:
        return MODE_COLORS[key]
    import zlib
    return _MODE_FALLBACK[zlib.crc32(key.encode()) % len(_MODE_FALLBACK)]


FONT_STACK = ["Microsoft JhengHei UI", "Microsoft JhengHei",
              "Segoe UI", "DejaVu Sans", "sans-serif"]


def apply_mpl(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "axes.unicode_minus": False,
        "figure.autolayout": False,
    })


QSS = f"""
QWidget {{
    background: {PLANE};
    color: {INK_2};
    font-family: "Microsoft JhengHei UI", "Segoe UI", sans-serif;
    font-size: 10pt;
}}
QMainWindow, QDialog {{ background: {PLANE}; }}
QFrame#Card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#Card QLabel, QFrame#Card QCheckBox {{ background: transparent; }}
QLabel#SectionTitle {{
    color: {INK};
    font-weight: 600;
    font-size: 10.5pt;
    padding: 2px 0 4px 0;
}}
QLabel#Hint {{ color: {MUTED}; font-size: 9pt; }}
QLabel#StatBig {{ color: {INK}; font-size: 21pt; font-weight: 600; }}
QLabel#StatUnit {{ color: {MUTED}; font-size: 9pt; }}
QLabel#StatLabel {{ color: {INK_2}; font-size: 9pt; }}

QPushButton {{
    background: #2f2f2d;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 12px;
    color: {INK};
}}
QPushButton:hover:!disabled {{ background: #3a3a37; border-color: #4d4d49; }}
QPushButton:pressed:!disabled {{ background: #262624; }}
QPushButton:disabled {{ color: #6a6963; background: #232322; border-color: #2c2c2a; }}
QPushButton#Primary {{
    background: {C_REF}; border-color: {C_REF}; color: #ffffff; font-weight: 600;
}}
QPushButton#Primary:hover:!disabled {{ background: #4d95ef; border-color: #4d95ef; }}
QPushButton#Primary:disabled {{ background: #2a3a4d; border-color: #2c2c2a; color: #6a6963; }}

QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox {{
    background: #2a2a28;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    color: {INK};
    selection-background-color: {C_REF};
}}
QComboBox:hover, QDoubleSpinBox:hover {{ border-color: #4d4d49; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: #2a2a28; border: 1px solid {BORDER};
    selection-background-color: {C_REF}; color: {INK}; outline: none;
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; background: #353532; border: none; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; background: {SURFACE}; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {MUTED};
    padding: 7px 15px; margin-right: 2px;
    border: 1px solid transparent; border-bottom: none;
    border-top-left-radius: 7px; border-top-right-radius: 7px;
}}
QTabBar::tab:selected {{ background: {SURFACE}; color: {INK};
                         border-color: {BORDER}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ color: {INK_2}; }}

QTableWidget {{
    background: {SURFACE}; alternate-background-color: #1f1f1e;
    gridline-color: {GRID}; border: none; color: {INK_2};
    selection-background-color: #2c4a6e; selection-color: {INK};
}}
QHeaderView::section {{
    background: #262624; color: {INK}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 6px; font-weight: 600;
}}
QTableWidget::item {{ padding: 4px 6px; }}

QTextEdit {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
    color: {INK_2}; font-family: Consolas, "Courier New", monospace; font-size: 9.5pt;
}}
QSlider::groove:horizontal {{ height: 4px; background: {BASELINE}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {C_REF}; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: #2c5f9e; border-radius: 2px; }}
QProgressBar {{
    background: #2a2a28; border: 1px solid {BORDER}; border-radius: 6px;
    text-align: center; color: {INK}; height: 18px;
}}
QProgressBar::chunk {{ background: {C_REF}; border-radius: 5px; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QToolTip {{
    background: #2a2a28; color: {INK};
    border: 1px solid {BORDER}; padding: 5px; border-radius: 4px;
}}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #3d3d39; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #4d4d49; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: #3d3d39; border-radius: 5px; min-width: 24px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid {BORDER}; background: #2a2a28;
}}
QCheckBox::indicator:checked {{ background: {C_REF}; border-color: {C_REF}; }}
QSplitter::handle {{ background: {BORDER}; }}
QMenuBar {{ background: {PANEL}; color: {INK_2}; }}
QMenuBar::item:selected {{ background: #353532; }}
QMenu {{ background: #2a2a28; border: 1px solid {BORDER}; color: {INK_2}; }}
QMenu::item:selected {{ background: {C_REF}; color: #ffffff; }}
"""

# --- 播放面板(ui/timeline.py);全部限定在 #Playback 內,不影響 matplotlib 工具列 ---
QSS += f"""
QWidget#Playback {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QWidget#Playback QLabel, QWidget#Playback QCheckBox {{ background: transparent; }}
QWidget#Playback QToolButton {{
    background: #2f2f2d; border: 1px solid {BORDER}; border-radius: 6px; padding: 0;
}}
QWidget#Playback QToolButton:hover:!disabled {{ background: #3a3a37; border-color: #4d4d49; }}
QWidget#Playback QToolButton:pressed:!disabled {{ background: #262624; }}
QWidget#Playback QToolButton:disabled {{ background: #232322; border-color: #2c2c2a; }}
QWidget#Playback QToolButton#PlayButton {{ background: {C_REF}; border-color: {C_REF}; }}
QWidget#Playback QToolButton#PlayButton:hover:!disabled {{ background: #4d95ef; border-color: #4d95ef; }}
QWidget#Playback QToolButton#PlayButton:disabled {{ background: #2a3a4d; border-color: #2c2c2a; }}
QWidget#Playback QPushButton {{ padding: 4px 10px; }}
QWidget#Playback QComboBox {{ padding: 3px 8px; }}
QWidget#Playback QComboBox:disabled {{ color: #6a6963; background: #232322; border-color: #2c2c2a; }}
QWidget#Playback QCheckBox:disabled {{ color: #6a6963; }}
QWidget#Playback QCheckBox::indicator:checked:disabled {{ background: #2a3a4d; border-color: #2c2c2a; }}
QWidget#Playback QLabel:disabled {{ color: #6a6963; }}
QWidget#Playback QFrame#PlaySep {{ background: {BORDER}; border: none; }}
QLabel#PlayTime {{
    color: {INK}; font-family: Consolas, "Cascadia Mono", monospace; font-size: 10pt;
}}
QLabel#PlayReadout {{ color: {INK_2}; }}
"""
