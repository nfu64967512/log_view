"""時間軸 / 播放面板煙霧測試(離線、不需人工點擊)。

用自建的「一份 log 涵蓋整次開機」合成資料:地面待機 → ARM 1 → DISARM → VIO 重置
→ ARM 2(含 EV 事件、ERR 錯誤)→ DISARM → 地面待機;檢查飛行段選單、選段、播放 / 自動停、
跳轉、拖曳分析區間邊界、快捷鍵、模式色帶、事件標記、可見範圍連動,以及幾個回歸案例
(右鍵選單 exec 期間換 log、播放頭與區間邊界重疊、只播區間時跳到區間前、
下拉清單開著時的按鍵、窄視窗的讀數寬度、最短可見跨度 MIN_VIEW_S 與欄位圖一致、
位置資料停了之後的 ERR / EV / 訊息(範圍 = [t_base, t_last]);有真實 log 時也驗一次),
並把面板存成 PNG。

  python tests\\test_timeline_ui.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32" and os.path.isdir(r"C:\Windows\Fonts"):
    # offscreen 平台預設沒有字型資料庫(中文會變方塊),指向系統字型目錄
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PyQt6.QtCore import QEvent, QEventLoop, QPoint, QPointF, Qt, QTimer  # noqa: E402
from PyQt6.QtGui import (QColor, QContextMenuEvent, QFontMetricsF, QKeySequence, QMouseEvent,  # noqa: E402
                         QTextDocument, QWheelEvent)
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMenu, QWidget, QVBoxLayout  # noqa: E402

from core import frames  # noqa: E402
from core.timeline import format_clock, horizontal_speed  # noqa: E402
from core.types import (EV_ARM, EV_DISARM, EV_ERROR, EV_EVENT, EV_MODE, EV_RESET,  # noqa: E402
                        KIND_GLOBAL, KIND_NED, LogData, LogEvent, TrajSource)
from ui import theme  # noqa: E402
import ui.timeline as uitl  # noqa: E402
from ui.timeline import (AXIS_H, EVENT_H, MODE_ALPHA, PlaybackPanel, TimelineWidget,  # noqa: E402
                         guard_popup_keys)

T0 = 500.0                     # log 時基(秒)
ORIGIN = (23.702, 120.431, 38.0)
# 相對 log 起點的事件時刻
ARM1, DISARM1, RESET, ARM2, DISARM2, DUR = 120.0, 360.0, 420.0, 480.0, 730.0, 900.0
EV_T, ERR_T, LAND_T = 550.0, 610.0, 727.0     # ARM 2 內的 EV 事件 / ERR 錯誤 / 降落完成


def _alt_profile(tr: np.ndarray) -> np.ndarray:
    alt = np.zeros_like(tr)
    for a, b, h in ((ARM1, DISARM1, 20.0), (ARM2, DISARM2, 35.0)):
        up = np.clip((tr - (a + 8)) / 10.0, 0, 1)
        down = np.clip(((b - 8) - tr) / 12.0, 0, 1)
        alt += h * np.minimum(up, down) * ((tr > a) & (tr < b))
    return alt


def _enu(tr: np.ndarray) -> np.ndarray:
    e = np.zeros_like(tr)
    n = np.zeros_like(tr)
    for a, b, r in ((ARM1, DISARM1, 25.0), (ARM2, DISARM2, 40.0)):
        m = (tr > a + 20) & (tr < b - 25)
        ph = (tr[m] - (a + 20)) / 40.0 * 2 * np.pi
        e[m] = r * np.sin(ph)
        n[m] = r * (1 - np.cos(ph))
    return np.column_stack([e, n, _alt_profile(tr)])


def make_boot_logdata() -> LogData:
    """合成「開機到關機」一份 log:兩次起降、中間一次 VIO 重置、VISP 地面時沒資料。"""
    rng = np.random.default_rng(3)
    tg = np.arange(0.0, DUR + 1e-6, 0.2)              # GPS 5 Hz(含終點)
    enu = _enu(tg) + rng.normal(0, 0.15, (tg.size, 3))
    lat, lng, alt = frames.enu_to_geodetic(enu, ORIGIN)
    gps = TrajSource("GPS", KIND_GLOBAL, T0 + tg, lat=lat, lng=lng, alt=alt)
    tv = np.arange(0.0, DUR, 1 / 30)                  # VISP 30 Hz,只在飛行前後有
    keep = ((tv > ARM1 - 20) & (tv < DISARM1 + 20)) | ((tv > ARM2 - 20) & (tv < DISARM2 + 20))
    tv = tv[keep]
    ev = _enu(tv) + rng.normal(0, 0.05, (tv.size, 3))
    visp = TrajSource("VISP", KIND_NED, T0 + tv, ned=frames.enu_to_ned(ev))
    events = [
        LogEvent(T0 + 5.0, EV_MODE, "STABILIZE"),
        LogEvent(T0 + ARM1 - 2, EV_MODE, "LOITER"),
        LogEvent(T0 + ARM1, EV_ARM),
        LogEvent(T0 + 150.0, EV_MODE, "AUTO"),
        LogEvent(T0 + 300.0, EV_MODE, "RTL"),
        LogEvent(T0 + 335.0, EV_MODE, "LAND"),
        LogEvent(T0 + DISARM1, EV_DISARM),
        LogEvent(T0 + RESET, EV_RESET, "VISP"),
        LogEvent(T0 + ARM2, EV_ARM),
        LogEvent(T0 + 500.0, EV_MODE, "AUTO"),
        LogEvent(T0 + EV_T, EV_EVENT, "EK3_SOURCES_SET_TO_SECONDARY"),
        LogEvent(T0 + ERR_T, EV_ERROR, "FAILSAFE_GPS 1"),
        LogEvent(T0 + 700.0, EV_MODE, "LAND"),
        LogEvent(T0 + LAND_T, EV_EVENT, "LAND_COMPLETE"),
        LogEvent(T0 + DISARM2, EV_DISARM),
    ]
    return LogData(sources={"GPS": gps, "VISP": visp}, origin=ORIGIN,
                   origin_desc="合成", path="boot_synth.bin", events=events)


LATE_ERR1, LATE_ERR2, LATE_EV, LATE_MSG = 60.0, 70.0, 80.0, 120.0   # 相對位置資料終點(t_end)


class _MsgIndex:
    """只提供 log_messages() 的假索引(LogData.t_last 只看文字訊息 / 事件的時間)。"""

    def __init__(self, times):
        self._msgs = [SimpleNamespace(t=float(t), kind="msg", text="") for t in times]

    def log_messages(self):
        return list(self._msgs)


def make_late_logdata() -> LogData:
    """位置資料在 t_end 就停(LOG_DISARMED=0),之後還有 ERR / EV 事件與文字訊息
    (同真實 log「203 2025-3-7」:位置停在 5:01.7、MSG / ERR 記到 8:10.8)。"""
    ld = make_boot_logdata()
    te = ld.t_end
    late = [LogEvent(te + LATE_ERR1, EV_ERROR, "GPS:Glitch (2)"),
            LogEvent(te + LATE_ERR2, EV_ERROR, "GPS:已恢復 (0)"),
            LogEvent(te + LATE_EV, EV_EVENT, "EKF_YAW_RESET")]
    out = with_events(ld, sorted(ld.events + late, key=lambda e: e.t))
    out.index = _MsgIndex([ld.t_base + 30.0, te + 5.0, te + LATE_MSG])
    return out


def with_events(ld: LogData, events: list) -> LogData:
    """同樣的軌跡、換一組事件(模擬背景載入完成後換成另一份 log)。"""
    return LogData(sources=ld.sources, origin=ld.origin, origin_desc=ld.origin_desc,
                   path=ld.path, events=list(events))


def make_rows(ld: LogData):
    src = ld.sources["GPS"]
    p = src.enu(ld.origin)
    ts, spd = horizontal_speed(src.t, p)
    return [("高度", src.t, p[:, 2], theme.C_REF, "m"),
            ("速度", ts, spd, theme.C_EST, "m/s")]


def make_error(t0: float, t1: float):
    t = np.arange(t0, t1, 0.2)
    rng = np.random.default_rng(5)
    e = 0.25 + 0.6 * (t - t0) / max(t1 - t0, 1) + np.abs(rng.normal(0, 0.08, t.size))
    return t, e


def _pump(ms: int = 120):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _mouse(w: QWidget, kind: QEvent.Type, x: float, y: float,
           button=Qt.MouseButton.LeftButton, buttons=Qt.MouseButton.LeftButton):
    pos = QPointF(x, y)
    ev = QMouseEvent(kind, pos, QPointF(w.mapToGlobal(pos.toPoint())), button, buttons,
                     Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(w, ev)


def _blend(hex_color: str, alpha: int, under: str = theme.SURFACE) -> tuple[int, int, int]:
    c, u, a = QColor(hex_color), QColor(under), alpha / 255.0
    return tuple(round(a * x + (1 - a) * y) for x, y in
                 ((c.red(), u.red()), (c.green(), u.green()), (c.blue(), u.blue())))


def _pixel(img, x: float, y: float) -> tuple[int, int, int]:
    d = img.devicePixelRatio()
    c = img.pixelColor(int(x * d), int(y * d))
    return c.red(), c.green(), c.blue()


def _close(a, b, tol: int = 8) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _plain(rich: str) -> str:
    d = QTextDocument()
    d.setHtml(rich)
    return d.toPlainText()


def _context_menu(w: QWidget, x: float, y: float):
    p = QPoint(int(x), int(y))
    QApplication.sendEvent(w, QContextMenuEvent(QContextMenuEvent.Reason.Mouse, p, w.mapToGlobal(p)))


def _flush_deletes():
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


class _ScriptedMenu(QMenu):
    """取代 ui.timeline 裡的 QMenu:exec() 不開視窗,直接選文字開頭相符的項目;
    選之前可先執行 during()(模擬 exec 巢狀事件迴圈期間背景載入完成、換掉 log)。"""
    pick = ""
    during = None
    created = 0

    def __init__(self, *a):
        super().__init__(*a)
        _ScriptedMenu.created += 1

    def exec(self, *a):
        act = next((x for x in self.actions() if self.pick and x.text().startswith(self.pick)), None)
        if _ScriptedMenu.during is not None:
            _ScriptedMenu.during()
        return act


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)
    failed = 0

    def check(label, cond, extra=""):
        nonlocal failed
        if cond:
            print(f"  [OK]   {label} {extra}")
        else:
            failed += 1
            print(f"  [FAIL] {label} {extra}")

    host = QWidget()
    lay = QVBoxLayout(host)
    lay.setContentsMargins(8, 8, 8, 8)
    panel = PlaybackPanel()
    lay.addWidget(panel)
    host.resize(1116, 200)
    host.show()
    _pump(150)

    times: list[float] = []
    wins: list[tuple[float, float]] = []
    panel.timeChanged.connect(times.append)
    panel.windowRequested.connect(lambda a, b: wins.append((a, b)))

    print("[無 log 狀態]")
    check("控制項停用", not panel.btn_play.isEnabled() and not panel.cmb_segment.isEnabled())
    panel.play()
    check("無 log 時按播放不動作", not panel.is_playing())
    pm = panel.grab()
    check("空狀態可繪製", not pm.isNull())

    ld = make_boot_logdata()
    lo, hi = ld.t_base, ld.t_end
    segs = ld.segments

    print("\n[載入 log]")
    panel.set_log(ld)
    panel.set_rows(make_rows(ld))
    _pump(50)
    check("飛行段 = 2", len(segs) == 2, f"({[(s.t0 - lo, s.t1 - lo) for s in segs]})")
    check("控制項啟用", panel.btn_play.isEnabled() and panel.cmb_segment.isEnabled())
    check("播放頭在 log 起點", abs(panel.time() - lo) < 1e-9 and times and times[-1] == lo)
    n = panel.cmb_segment.count()
    check("選單項數 = 整份 + 2 段 + 自訂", n == 4, f"({n})")
    txt = [panel.cmb_segment.itemText(i) for i in range(n)]
    check("第 0 項為整份 log", txt[0].startswith("整份 log") and "15:00" in txt[0], f"({txt[0]})")
    check("ARM 1 標籤含相對秒數", txt[1].startswith("ARM 1") and "120.0–360.0 s" in txt[1],
          f"({txt[1]})")
    check("最後一項為自訂區間", txt[-1] == "自訂區間")
    check("預設選整份 log", panel.cmb_segment.currentIndex() == 0)
    check("時間標籤格式", panel.lbl_time.text() == "0:00.0 / 15:00.0", f"({panel.lbl_time.text()})")
    check("時間軸可見範圍 = 整份", panel.timeline.view_range() == (lo, hi))

    print("\n[選段]")
    wins.clear()
    panel.select_segment(1)
    check("select_segment 發出 windowRequested(段起訖)",
          wins == [(segs[1].t0, segs[1].t1)], f"({[(a - lo, b - lo) for a, b in wins]})")
    check("播放頭移到段首", abs(panel.time() - segs[1].t0) < 1e-9)
    check("選單跟著選到 ARM 2", panel.cmb_segment.currentIndex() == 2)
    check("選段後暫停", not panel.is_playing())
    wins.clear()
    panel.cmb_segment.activated.emit(1)         # 使用者在選單點 ARM 1
    check("選單 activated → windowRequested", wins == [(segs[0].t0, segs[0].t1)])
    wins.clear()
    panel.select_segment(-1)
    check("select_segment(-1) = 整份 log", wins == [(lo, hi)])

    print("\n[set_window 反映到選單,不回送]")
    wins.clear()
    panel.set_window(segs[0].t0 + 0.04, segs[0].t1 - 0.03)   # 0.1 s 數值框四捨五入的誤差
    check("容差內 → 選到 ARM 1", panel.cmb_segment.currentIndex() == 1)
    panel.set_window(lo, hi)
    check("整份 → 第 0 項", panel.cmb_segment.currentIndex() == 0)
    panel.set_window(lo + 30, lo + 200)
    check("其他 → 自訂區間", panel.cmb_segment.currentIndex() == n - 1)
    check("set_window 不發 windowRequested", wins == [])
    check("只播分析區間 → 播放範圍 = 區間",
          (panel.clock.play_lo, panel.clock.play_hi) == (lo + 30, lo + 200))
    panel.chk_window_only.setChecked(False)
    check("取消勾選 → 播放範圍 = 整份",
          (panel.clock.play_lo, panel.clock.play_hi) == (lo, hi))
    panel.chk_window_only.setChecked(True)

    print("\n[播放]")
    panel.set_window(lo, hi)
    panel.seek(lo + 100)
    panel.set_speed(10)
    check("速度選單同步", panel.cmb_speed.currentText() == "10×", f"({panel.cmb_speed.currentText()})")
    times.clear()
    t_start = panel.time()
    panel.play()
    check("開始播放", panel.is_playing() and panel.timer.isActive())
    _pump(600)
    adv = panel.time() - t_start
    check("10× 播放 0.6 s 約前進 6 s", 3.0 < adv < 8.5, f"(前進 {adv:.2f} s, {len(times)} 次 tick)")
    check("timeChanged 持續發出且遞增",
          len(times) >= 5 and all(b > a for a, b in zip(times, times[1:])))
    panel.pause()
    t_p = panel.time()
    _pump(200)
    check("暫停後不再前進", panel.time() == t_p and not panel.timer.isActive())
    panel.seek(lo + 50)
    panel.play()
    panel.seek(lo + 200)                         # 播放中拖曳:從新位置繼續
    _pump(200)
    check("播放中 seek 後繼續播", panel.is_playing() and panel.time() > lo + 200,
          f"({panel.time() - lo:.2f})")
    panel.pause()

    print("\n[自動停在播放範圍終點]")
    panel.set_window(lo + 10, lo + 11)
    panel.seek(lo + 10)
    panel.set_speed(50)
    panel.play()
    _pump(400)
    check("走到區間終點自動停", not panel.is_playing() and not panel.timer.isActive())
    check("停在區間終點", abs(panel.time() - (lo + 11)) < 1e-9, f"({panel.time() - lo:.3f})")
    check("播放鍵恢復為「播放」", panel.btn_play.toolTip().startswith("播放"))
    panel.play()                                  # 在終點再按播放 → 從起點重播
    check("在終點按播放 → 從區間起點重播", abs(panel.time() - (lo + 10)) < 0.2 and panel.is_playing())
    panel.pause()
    panel.set_speed(1)
    panel.set_window(lo, hi)

    print("\n[seek 限制範圍 / step]")
    panel.seek(lo - 100)
    check("seek 下限", panel.time() == lo)
    panel.seek(hi + 100)
    check("seek 上限", panel.time() == hi)
    panel.seek(lo + 300)
    panel.step(-10)
    check("step(-10)", abs(panel.time() - (lo + 290)) < 1e-9)

    print("\n[跳到上 / 下一個事件]")
    panel.seek(lo + 200)
    panel.jump(+1)
    check("下一個 = DISARM 1", abs(panel.time() - (lo + DISARM1)) < 1e-9, f"({panel.time() - lo:.2f})")
    panel.jump(+1)
    check("再下一個 = VIO 重置", abs(panel.time() - (lo + RESET)) < 1e-9, f"({panel.time() - lo:.2f})")
    panel.jump(-1)
    check("上一個 = DISARM 1", abs(panel.time() - (lo + DISARM1)) < 1e-9)
    panel.seek(lo + ARM1 + 0.03)                  # 0.05 s 內的點忽略
    panel.jump(-1)
    check("忽略 0.05 s 內的點", abs(panel.time() - lo) < 1e-9, f"({panel.time() - lo:.2f})")
    panel.set_window(lo + 50, lo + 600)
    panel.seek(lo + 10)
    panel.jump(+1)
    check("分析區間邊界也是跳點", abs(panel.time() - (lo + 50)) < 1e-9)

    print("\n[設起點 / 設終點]")
    panel.set_window(lo + 50, lo + 600)
    wins.clear()
    panel.seek(lo + 100)
    panel.btn_set_in.click()
    check("設起點 → windowRequested(t, 原終點)", wins == [(lo + 100, lo + 600)])
    panel.seek(lo + 700)
    wins.clear()
    panel.btn_set_in.click()
    check("起點晚於終點時不動作", wins == [])
    panel.seek(lo + 400)
    panel.btn_set_out.click()
    check("設終點 → windowRequested(原起點, t)", wins == [(lo + 100, lo + 400)])

    print("\n[時間軸:拖曳分析區間邊界 → windowRequested]")
    tl = panel.timeline
    panel.set_window(lo + 100, lo + 400)
    _pump(50)
    wins.clear()
    xa = tl._t2x(lo + 100)
    y = tl._rows_top() + 9                       # 第一條資料列
    seeks: list[float] = []
    tl.seeked.connect(seeks.append)
    _mouse(tl, QEvent.Type.MouseButtonPress, xa + 2, y)
    _mouse(tl, QEvent.Type.MouseMove, xa + 40, y, Qt.MouseButton.NoButton)
    check("拖曳中不發 windowEdited", wins == [])
    _mouse(tl, QEvent.Type.MouseButtonRelease, xa + 40, y, buttons=Qt.MouseButton.NoButton)
    check("放開 → windowRequested 一次", len(wins) == 1, f"({wins})")
    if wins:
        exp = tl._x2t(xa + 40)
        check("新起點 ≈ 滑鼠位置、終點不變",
              abs(wins[0][0] - exp) < 1.0 and wins[0][1] == lo + 400,
              f"({wins[0][0] - lo:.2f} / {wins[0][1] - lo:.2f})")
        check("拖邊界時不移動播放頭", seeks == [])
    wins.clear()
    tl.windowEdited.emit(lo + 20, lo + 80)
    check("windowEdited 訊號 → windowRequested", wins == [(lo + 20, lo + 80)])

    print("\n[時間軸:點擊 / 拖曳播放頭]")
    panel.set_window(lo, hi)
    _pump(30)
    seeks.clear()
    x_mid = (tl._x0() + tl._x1()) / 2
    _mouse(tl, QEvent.Type.MouseButtonPress, x_mid, y)
    _mouse(tl, QEvent.Type.MouseMove, x_mid + 30, y, Qt.MouseButton.NoButton)
    _mouse(tl, QEvent.Type.MouseMove, x_mid + 60, y, Qt.MouseButton.NoButton)
    _mouse(tl, QEvent.Type.MouseButtonRelease, x_mid + 60, y, buttons=Qt.MouseButton.NoButton)
    check("拖曳中連續發 seeked", len(seeks) == 3, f"({len(seeks)})")
    check("面板播放頭跟著走", abs(panel.time() - tl._x2t(x_mid + 60)) < 1e-6)

    print("\n[時間軸:雙擊 ARM 段 → 選段]")
    wins.clear()
    xb = tl._t2x((segs[0].t0 + segs[0].t1) / 2)
    _mouse(tl, QEvent.Type.MouseButtonDblClick, xb, AXIS_H + EVENT_H / 2)
    check("雙擊 ARM 1 → windowRequested(ARM 1)", wins == [(segs[0].t0, segs[0].t1)])
    check("選單選到 ARM 1", panel.cmb_segment.currentIndex() == 1)

    print("\n[時間軸:縮放 / 平移]")
    tl.set_view(lo + 100, lo + 200)
    check("縮放範圍", tl.view_range() == (lo + 100, lo + 200))
    m = uitl.MIN_VIEW_S
    check("最短可見跨度是公開常數(TimelineWidget / PlaybackPanel 同值)",
          TimelineWidget.MIN_VIEW_S == PlaybackPanel.MIN_VIEW_S == tl.MIN_VIEW_S == m and m <= 0.1, f"({m})")
    import ui.fieldplot as uifp
    fp_min = getattr(uifp, "MIN_SPAN_S", getattr(uifp, "_MIN_SPAN", None))
    if fp_min is not None:
        check("和欄位圖的最小跨度一致(兩邊縮放才對得上)", abs(fp_min - m) < 1e-12, f"({fp_min} vs {m})")
    got = tl.set_view(lo + 100, lo + 100.5)
    check("跨度 ≥ 最短值時原樣套用(0.5 s,不再撐到 2 s)",
          tl.view_range() == (lo + 100, lo + 100.5) and got == tl.view_range(), f"({got})")
    got = panel.set_view(lo + 100, lo + 100 + m / 4)
    v0, v1 = tl.view_range()
    check("太短 → 以中心撐到最短跨度,回傳實際套用的範圍",
          abs((v1 - v0) - m) < 1e-9 and abs((v0 + v1) / 2 - (lo + 100 + m / 8)) < 1e-9 and got == (v0, v1),
          f"({v0 - lo:.4f}–{v1 - lo:.4f})")
    tks = tl._ticks()
    lbls = [format_clock(rel, dec) for _, rel, dec in tks]
    fm8 = QFontMetricsF(tl._font(8.0))
    xs = [tl._t2x(t) for t, _, _ in tks]
    check("放大到最短跨度:有刻度、兩位小數、標籤互不相同也不重疊",
          len(tks) >= 2 and all(d == 2 for _, _, d in tks) and len(set(lbls)) == len(lbls)
          and all(b - a >= fm8.horizontalAdvance(s) + 4 for a, b, s in zip(xs, xs[1:], lbls)), f"({lbls})")
    tl.set_view(lo + 100, lo + 100.3)
    lbls = [format_clock(rel, dec) for _, rel, dec in tl._ticks()]
    check("0.3 s 可見範圍的刻度", len(lbls) >= 2 and len(set(lbls)) == len(lbls), f"({lbls})")
    tl.set_view(hi - 50, hi + 50)
    v0, v1 = tl.view_range()
    check("不超出 log 範圍", v1 <= hi + 1e-9 and abs((v1 - v0) - 100) < 1e-9)
    panel.seek(lo + 10)
    v0, v1 = tl.view_range()
    check("播放頭跳出可見範圍時自動翻頁", v0 <= lo + 10 <= v1, f"({v0 - lo:.1f}–{v1 - lo:.1f})")
    _mouse(tl, QEvent.Type.MouseButtonDblClick, x_mid, AXIS_H / 2)
    check("雙擊時間尺 → 全覽", tl.view_range() == (lo, hi))

    def wheel(dy, mods=Qt.KeyboardModifier.NoModifier):
        pos = QPointF(x_mid, y)
        QApplication.sendEvent(tl, QWheelEvent(
            pos, QPointF(tl.mapToGlobal(pos.toPoint())), QPoint(0, 0), QPoint(0, dy),
            Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False))

    t_anchor = tl._x2t(x_mid)
    wheel(120)
    v0, v1 = tl.view_range()
    check("滾輪放大(以游標為中心)", (v1 - v0) < (hi - lo) and abs(tl._x2t(x_mid) - t_anchor) < 1e-6,
          f"(可見 {v1 - v0:.0f} s)")
    wheel(-120, Qt.KeyboardModifier.ShiftModifier)
    w0, w1 = tl.view_range()
    check("Shift+滾輪平移(長度不變)", w0 > v0 and abs((w1 - w0) - (v1 - v0)) < 1e-6)
    tl.reset_view()

    print("\n[快捷鍵]")
    acts = panel.install_shortcuts(host)
    keys = {a.text(): a.shortcut().toString() for a in acts}
    check("11 個 QAction", len(acts) == 11, f"({len(acts)})")
    check("加到 scope", all(a in host.actions() for a in acts))
    check("WidgetWithChildrenShortcut",
          all(a.shortcutContext() == Qt.ShortcutContext.WidgetWithChildrenShortcut for a in acts))
    check("按鍵對應", keys.get("播放/暫停") == "Space" and keys.get("後退 10 秒") == "Shift+Left"
          and keys.get("下一個事件") == QKeySequence("PgDown").toString()
          and keys.get("設為分析起點") == "[", f"({keys})")
    act = {a.text(): a for a in acts}
    panel.set_window(lo + 100, lo + 400)
    panel.seek(lo + 200)
    act["前進 1 秒"].trigger()
    check("→ 前進 1 秒", abs(panel.time() - (lo + 201)) < 1e-9)
    act["後退 10 秒"].trigger()
    check("Shift+← 後退 10 秒", abs(panel.time() - (lo + 191)) < 1e-9)
    act["跳到播放範圍終點"].trigger()
    check("End → 播放範圍終點", abs(panel.time() - (lo + 400)) < 1e-9)
    act["跳到播放範圍起點"].trigger()
    check("Home → 播放範圍起點", abs(panel.time() - (lo + 100)) < 1e-9)
    act["下一個事件"].trigger()
    check("PgDown → 下一個事件(ARM 1 起點)", abs(panel.time() - (lo + ARM1)) < 1e-9,
          f"({panel.time() - lo:.2f})")
    act["上一個事件"].trigger()
    check("PgUp → 上一個事件", abs(panel.time() - (lo + 100)) < 1e-9)
    act["播放/暫停"].trigger()
    check("Space → 播放", panel.is_playing())
    act["播放/暫停"].trigger()
    check("Space → 暫停", not panel.is_playing())
    wins.clear()
    panel.seek(lo + 150)
    act["設為分析起點"].trigger()
    panel.seek(lo + 300)
    act["設為分析終點"].trigger()
    check("[ / ] 設定分析區間", wins == [(lo + 150, lo + 400), (lo + 150, lo + 300)], f"({wins})")

    n_err0 = len(uitl.handler_errors)

    print("\n[跳轉:EV 事件 / ERR 錯誤也是跳點]")
    panel.set_window(lo, hi)
    panel.seek(lo + 500)
    panel.jump(+1)
    check("下一個 = EV 事件", abs(panel.time() - (lo + EV_T)) < 1e-9, f"({panel.time() - lo:.2f})")
    panel.jump(+1)
    check("再下一個 = ERR 錯誤", abs(panel.time() - (lo + ERR_T)) < 1e-9, f"({panel.time() - lo:.2f})")
    panel.jump(+1)
    check("再下一個 = LAND_COMPLETE", abs(panel.time() - (lo + LAND_T)) < 1e-9, f"({panel.time() - lo:.2f})")
    panel.jump(-1)
    check("上一個 = ERR 錯誤", abs(panel.time() - (lo + ERR_T)) < 1e-9)

    print("\n[模式列:色帶 / 顏色 / 帶內名稱]")
    tl.reset_view()
    panel.seek(lo)
    _pump(30)
    lay_m = tl._mode_layout()
    names = [m[1] for m in lay_m]
    check("模式色帶依序", names == ["STABILIZE", "LOITER", "AUTO", "RTL", "LAND", "AUTO", "LAND"], f"({names})")
    check("色碼 = theme.mode_color", all(m[2] == theme.mode_color(m[1]) for m in lay_m))
    r_first, r_last = lay_m[0][0], lay_m[-1][0]
    check("第一段從模式切換點開始、最後一段延伸到 log 結尾",
          abs(r_first.left() - round(tl._t2x(lo + 5.0))) <= 1 and abs(r_last.right() - tl._x1()) <= 1.5,
          f"({r_first.left():.0f} / {r_last.right():.1f} vs {tl._x1():.1f})")
    check("相鄰色帶不重疊", all(a[0].right() <= b[0].left() for a, b in zip(lay_m, lay_m[1:])))
    auto = lay_m[2]
    check("寬色帶內寫模式名稱", auto[3] == "AUTO", f"({auto[3]!r})")
    check("黃色帶用深色字、紅色帶用白字",
          auto[4].name() == QColor(theme.PLANE).name() and lay_m[4][4].name() == QColor(theme.INK).name(),
          f"({auto[4].name()} / {lay_m[4][4].name()})")
    fm75 = QFontMetricsF(tl._font(7.5))
    check("帶內文字不超出色帶", all(fm75.horizontalAdvance(m[3]) <= m[0].width() - 6 for m in lay_m if m[3]))
    loiter = lay_m[1]
    check("窄色帶(LOITER 32 s)名稱以「…」截斷或省略", loiter[3] == "" or loiter[3].endswith("…"),
          f"({loiter[3]!r}, 寬 {loiter[0].width():.0f} px)")
    tl.set_view(lo + 100, lo + 200)
    lay_z = tl._mode_layout()
    check("縮放後:色帶裁到可見範圍、名稱放不下就截斷或省略",
          [m[1] for m in lay_z] == ["STABILIZE", "LOITER", "AUTO"]
          and all(not m[3] or m[3] == m[1] or m[3].endswith("…") for m in lay_z),
          f"({[(m[1], m[3]) for m in lay_z]})")
    tl.reset_view()
    _pump(30)
    img = tl.grab().toImage()
    ok_px = []
    for r, name, col, txt, _ in lay_m:
        if r.width() < 12:
            continue
        got = _pixel(img, r.right() - 3, r.top() + 0.5)    # 色帶右上角(避開文字)
        ok_px.append((name, _close(got, _blend(col, MODE_ALPHA)), got, _blend(col, MODE_ALPHA)))
    check("像素探針:色帶顏色 = 模式色(α≈0.75)", ok_px and all(o[1] for o in ok_px),
          f"({[(o[0], o[2], o[3]) for o in ok_px if not o[1]]})")
    x_auto = (auto[0].left() + auto[0].right()) / 2
    tip = tl._tooltip_at(QPointF(x_auto, tl._mode_top() + 8))
    check("模式列提示顯示模式名稱與起訖", "模式" in tip and "<b>AUTO</b>" in tip and "→" in tip, f"({tip[:90]}…)")
    check("事件列不再畫模式刻度標籤(提示不提刻度)",
          "刻度" not in tl._tooltip_at(QPointF(tl._x0() - 20, AXIS_H + 8)))
    h_before = tl.height()
    no_mode = with_events(ld, [e for e in ld.events if e.kind != EV_MODE])
    solo2 = TimelineWidget()
    solo2.resize(800, solo2.sizeHint().height())
    solo2.set_log(no_mode.t_base, no_mode.t_end, no_mode.segments, no_mode.events)
    solo2.set_rows(make_rows(no_mode))
    check("沒有 MODE 訊息:不畫色帶、列高不變", solo2._mode_layout() == [] and solo2.height() == h_before,
          f"({solo2.height()} vs {h_before})")
    check("沒有 MODE 訊息仍可繪製", not solo2.grab().isNull())

    print("\n[事件標記:◆ EV / ▲ ERR + 提示]")
    marks = tl._event_marks()
    kinds = sorted(k for _, k, _ in marks)
    check("標記 = 2 個 EV + 1 個 ERR", kinds == sorted([EV_EVENT, EV_EVENT, EV_ERROR]), f"({kinds})")
    x_err = tl._t2x(lo + ERR_T)
    img = tl.grab().toImage()
    got = _pixel(img, x_err, tl._mark_y() + 1)
    check("像素探針:ERR 三角形為 S_CRIT 色", _close(got, _blend(theme.S_CRIT, 255), 12),
          f"({got} vs {QColor(theme.S_CRIT).getRgb()[:3]})")
    x_ev = tl._t2x(lo + EV_T)
    got = _pixel(img, x_ev, tl._mark_y())
    check("像素探針:EV 菱形為 INK_2 色", _close(got, _blend(theme.INK_2, 255), 12), f"({got})")
    ya, yb = tl._event_band_y()
    check("標記掛在綠帶下緣(不壓置中的 ARM 標籤)", tl._mark_y() - 4.5 >= (ya + yb) / 2 + 3,
          f"(標記頂 {tl._mark_y() - 4.5:.1f}, 帶中線 {(ya + yb) / 2:.1f})")
    tip = tl._tooltip_at(QPointF(x_err + 2, AXIS_H + 12))
    check("ERR 提示:標籤 + 相對時間 + 錯誤字樣(狀態色)",
          "FAILSAFE_GPS 1" in tip and "10:10.0" in tip and "錯誤" in tip and theme.S_CRIT in tip,
          f"({tip[:120]}…)")
    tip = tl._tooltip_at(QPointF(x_ev - 3, tl._rows_top() + 5))
    check("EV 提示(資料列上也列出附近事件)", "EK3_SOURCES_SET_TO_SECONDARY" in tip and "9:10.0" in tip)
    tip = tl._tooltip_at(QPointF((x_ev + x_err) / 2, AXIS_H + 12))
    check("離標記遠時不列事件", "EK3_" not in tip and "FAILSAFE" not in tip)

    print("\n[可見範圍連動:viewChanged]")
    views, pviews = [], []
    tl.viewChanged.connect(lambda a, b: views.append((a, b)))
    panel.viewChanged.connect(lambda a, b: pviews.append((a, b)))
    tl.set_view(lo + 100, lo + 300)
    panel.set_view(lo + 50, lo + 250)
    tl.reset_view()
    check("程式 set_view / PlaybackPanel.set_view / reset_view 不發", views == [] and pviews == [])
    panel.set_view(lo + 50, lo + 250)
    check("PlaybackPanel.set_view 轉給時間軸", tl.view_range() == (lo + 50, lo + 250)
          and panel.view_range() == (lo + 50, lo + 250))
    tl.reset_view()
    wheel(120)
    check("滾輪縮放 → viewChanged(絕對秒)", len(views) == 1 and views[-1] == tl.view_range(),
          f"({len(views)})")
    check("PlaybackPanel 轉發 viewChanged", pviews == views)
    wheel(-120, Qt.KeyboardModifier.ShiftModifier)
    check("Shift+滾輪平移 → viewChanged", len(views) == 2 and views[-1] == tl.view_range())
    _mouse(tl, QEvent.Type.MouseButtonDblClick, x_mid, AXIS_H / 2)
    check("雙擊時間尺(全覽)→ viewChanged", len(views) == 3 and views[-1] == (lo, hi))
    _mouse(tl, QEvent.Type.MouseButtonDblClick, x_mid, AXIS_H / 2)
    check("已是全覽:不重複發", len(views) == 3)
    panel.set_log(ld)
    panel.set_rows(make_rows(ld))
    check("set_log 不發", len(views) == 3)
    tl.set_view(lo + 100, lo + 150)
    panel.seek(lo + 400)                           # 播放頭跑出可見範圍 → 翻頁
    check("播放翻頁 → viewChanged", len(views) == 4 and views[-1][0] <= lo + 400 <= views[-1][1])

    print("\n[右鍵選單:exec 期間換 log、選單釋放]")
    real_menu = uitl.QMenu
    uitl.QMenu = _ScriptedMenu
    try:
        two = ld                                      # 兩段
        one = with_events(ld, [e for e in ld.events if e.t < T0 + RESET])   # 只剩 ARM 1
        panel.set_log(two)
        panel.set_rows(make_rows(two))
        _pump(20)
        views.clear()
        acts_seg: list[int] = []
        tl.segmentActivated.connect(acts_seg.append)
        x_arm2 = tl._t2x((segs[1].t0 + segs[1].t1) / 2)
        _ScriptedMenu.pick, _ScriptedMenu.during = "縮放到 ARM 2", lambda: panel.set_log(one)
        n_err = len(uitl.handler_errors)
        _context_menu(tl, x_arm2, AXIS_H + 10)
        check("exec 期間換 log 後選「縮放到 ARM 2」:沒有例外", len(uitl.handler_errors) == n_err,
              (uitl.handler_errors[-1].strip().splitlines()[-1] if len(uitl.handler_errors) > n_err else ""))
        check("換 log 後不套用舊選擇(仍是全覽、不發 viewChanged)", not tl.is_zoomed() and views == [])
        panel.set_log(two)
        wins.clear()
        x_arm1 = tl._t2x((segs[0].t0 + segs[0].t1) / 2)
        _ScriptedMenu.pick, _ScriptedMenu.during = "把分析區間設成", lambda: panel.set_log(two)
        _context_menu(tl, x_arm1, AXIS_H + 10)
        check("exec 期間換 log 後選「設成此 ARM 段」:不發過期索引", acts_seg == [] and wins == [],
              f"({acts_seg} / {wins})")
        _ScriptedMenu.pick, _ScriptedMenu.during = "縮放到 ARM 1", None
        _context_menu(tl, x_arm1, AXIS_H + 10)
        v0, v1 = tl.view_range()
        check("正常選「縮放到 ARM 1」→ 縮放 + viewChanged",
              v0 < segs[0].t0 < segs[0].t1 < v1 and (v1 - v0) < 1.2 * segs[0].duration
              and len(views) == 1, f"({v0 - lo:.1f}–{v1 - lo:.1f})")
        _ScriptedMenu.pick = "全覽"
        _context_menu(tl, x_mid, tl._rows_top() + 5)
        check("選「全覽」→ 回到全覽 + viewChanged", not tl.is_zoomed() and len(views) == 2)
        _ScriptedMenu.pick = "把分析區間設成"
        _context_menu(tl, x_arm1, AXIS_H + 10)
        check("正常選「設成此 ARM 段」→ 選段", acts_seg == [0] and wins[-1:] == [(segs[0].t0, segs[0].t1)])
        _ScriptedMenu.pick = ""
        for _ in range(5):
            _context_menu(tl, x_mid, tl._rows_top() + 5)
        _flush_deletes()
        check("右鍵選單用完即釋放(不累積)", tl.findChildren(QMenu) == [],
              f"({len(tl.findChildren(QMenu))} 個殘留 / 共建立 {_ScriptedMenu.created})")
        tl.segmentActivated.disconnect(acts_seg.append)
    finally:
        uitl.QMenu = real_menu

    print("\n[播放頭與區間邊界重疊:拖曳的是播放頭]")
    panel.set_log(ld)
    panel.set_rows(make_rows(ld))
    panel.select_segment(0)                      # 播放頭 = 區間起點
    _pump(30)
    xe = tl._t2x(segs[0].t0)
    yr = tl._rows_top() + 9
    seeks.clear()
    wins.clear()
    _mouse(tl, QEvent.Type.MouseMove, xe, yr, Qt.MouseButton.NoButton, Qt.MouseButton.NoButton)
    check("滑過重疊處:游標不是左右調整", tl.cursor().shape() != Qt.CursorShape.SizeHorCursor)
    tip = tl._tooltip_at(QPointF(xe, yr))
    check("提示說明要拖把手才調整區間", "把手" in tip, f"({tip[-60:]})")
    QTest.mousePress(tl, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(round(xe), yr))
    check("按下 = 抓播放頭(scrub)", tl.is_scrubbing())
    QTest.mouseMove(tl, QPoint(round(xe) + 40, yr))
    QTest.mouseRelease(tl, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       QPoint(round(xe) + 40, yr))
    check("拖曳移動播放頭", seeks and abs(panel.time() - tl._x2t(round(xe) + 40)) < 1e-6,
          f"({len(seeks)} 次 seeked)")
    check("分析區間沒被改(不會觸發重算)", wins == [] and panel._win == (segs[0].t0, segs[0].t1))
    panel.select_segment(0)
    _pump(20)
    wins.clear()
    hr = tl._handle_rect(0, tl._win)
    hx, hy = hr.center().x(), hr.center().y()
    _mouse(tl, QEvent.Type.MouseMove, hx, hy, Qt.MouseButton.NoButton, Qt.MouseButton.NoButton)
    check("滑到時間尺下緣的把手:左右調整游標", tl.cursor().shape() == Qt.CursorShape.SizeHorCursor)
    check("把手在時間尺下緣", hr.bottom() <= AXIS_H + 0.5 and hr.top() >= 14,
          f"({hr.top():.0f}–{hr.bottom():.0f})")
    QTest.mousePress(tl, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(round(hx), round(hy)))
    check("按把手 = 拖區間起點(即使播放頭在同一處)", tl._drag == 0)
    QTest.mouseMove(tl, QPoint(round(xe) + 70, round(hy)))
    QTest.mouseRelease(tl, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       QPoint(round(xe) + 70, round(hy)))
    check("拖把手 → windowRequested(新起點)",
          len(wins) == 1 and abs(wins[0][0] - tl._x2t(round(xe) + 70)) < 1.0 and wins[0][1] == segs[0].t1,
          f"({[(a - lo, b - lo) for a, b in wins]})")
    check("拖把手不移動播放頭", abs(panel.time() - segs[0].t0) < 1e-9)
    panel.seek(lo + 600)
    panel.set_window(lo + 100, lo + 400)
    _pump(20)
    xa = tl._t2x(lo + 100)
    _mouse(tl, QEvent.Type.MouseMove, xa + 2, yr, Qt.MouseButton.NoButton, Qt.MouseButton.NoButton)
    check("播放頭不在邊界上時:資料列上的邊界仍可直接拖",
          tl.cursor().shape() == Qt.CursorShape.SizeHorCursor and tl._press_target(QPointF(xa + 2, yr)) == 0)

    print("\n[只播分析區間:播放中跳到區間前 → 就地暫停]")
    panel.chk_window_only.setChecked(True)
    panel.set_window(lo + 300, lo + 400)
    panel.set_speed(1)
    panel.seek(lo + 310)
    panel.play()
    _pump(100)
    check("區間內播放中", panel.is_playing())
    panel.seek(lo + 250)                          # 播放中拖到區間起點之前
    _pump(150)
    check("跳到區間前 → 暫停", not panel.is_playing() and not panel.timer.isActive())
    check("停在原地(不穿越區間外、不跳回區間)", abs(panel.time() - (lo + 250)) < 1e-9,
          f"({panel.time() - lo:.3f})")
    panel.play()
    check("再按播放 → 從區間起點開始", abs(panel.time() - (lo + 300)) < 0.2 and panel.is_playing())
    panel.seek(lo + 450)                          # 拖到區間終點之後(既有規則)
    _pump(120)
    check("跳到區間後 → 暫停", not panel.is_playing() and abs(panel.time() - (lo + 450)) < 1e-9)
    panel.chk_window_only.setChecked(False)
    panel.seek(lo + 250)
    panel.play()
    _pump(120)
    check("不勾「只播分析區間」:區間外照常播", panel.is_playing() and panel.time() > lo + 250)
    panel.pause()
    panel.chk_window_only.setChecked(True)

    print("\n[下拉清單開著時:按鍵交給清單,不觸發播放快捷鍵]")
    panel.set_window(lo, hi)
    host.activateWindow()
    check("測試視窗為作用中", QTest.qWaitForWindowActive(host))
    check("guard_popup_keys 可重複呼叫(已裝過的略過)",
          guard_popup_keys(panel) == 0 and guard_popup_keys(host) == 0)
    panel.seek(lo + 200)
    cmb = panel.cmb_speed
    cmb.showPopup()
    _pump(50)
    view = cmb.view()
    check("清單已開啟", QApplication.activePopupWidget() is not None)
    t_b = panel.time()
    QTest.keyClick(view, Qt.Key.Key_End)
    check("End:清單選到最後一項", view.currentIndex().row() == cmb.count() - 1,
          f"({view.currentIndex().row()})")
    check("End:播放頭不動", panel.time() == t_b, f"({panel.time() - lo:.1f})")
    for k in (Qt.Key.Key_Home, Qt.Key.Key_PageDown, Qt.Key.Key_Right, Qt.Key.Key_Left, Qt.Key.Key_PageUp):
        QTest.keyClick(view, k)
    check("Home / PgDn / → / ← / PgUp:播放頭都不動", panel.time() == t_b)
    cmb.hidePopup()
    _pump(30)
    check("關閉清單後速度沒被改", panel.cmb_speed.currentText() == "1×", f"({panel.cmb_speed.currentText()})")
    cmb2 = panel.cmb_segment
    cmb2.showPopup()
    _pump(50)
    QTest.keyClick(cmb2.view(), Qt.Key.Key_Down)
    check("飛行段清單:↓ 移動選取", cmb2.view().currentIndex().row() == 1)
    QTest.keyClick(cmb2.view(), Qt.Key.Key_End)
    QTest.keyClick(cmb2.view(), Qt.Key.Key_PageDown)
    check("飛行段清單:End / PgDn 不動播放頭", panel.time() == t_b)
    cmb2.hidePopup()
    _pump(30)
    tl.setFocus()
    _pump(20)
    QTest.keyClick(tl, Qt.Key.Key_End)
    check("時間軸上 End 仍跳到播放範圍終點", abs(panel.time() - panel.clock.play_hi) < 1e-9,
          f"({panel.time() - lo:.1f})")
    QTest.keyClick(tl, Qt.Key.Key_Home)
    check("時間軸上 Home 仍跳到播放範圍起點", abs(panel.time() - panel.clock.play_lo) < 1e-9)

    print("\n[窄視窗:讀數至少保留誤差值、控制列不溢出]")
    full_readout = (f"3D 誤差 <b style='color:{theme.INK}'>0.43 m</b> · ARM 2 · AUTO · "
                    f"水平 <b>0.31 m</b> · 其他很長的說明文字")
    panel.set_readout(full_readout)
    rd = panel.lbl_readout
    for pw in (880, 1000, 1100):
        host.resize(pw + 16, host.height())        # host 左右邊距 8 → 面板寬 = pw
        _pump(80)
        shown = _plain(rd.shown_text())
        ws = [w for w in (panel.btn_prev, panel.btn_play, panel.btn_next, panel.cmb_speed, panel.lbl_time,
                          panel._lbl_segment, panel.cmb_segment, panel.btn_set_in, panel.btn_set_out,
                          panel.chk_window_only, rd) if w.isVisible()]
        geo = sorted((w.geometry() for w in ws), key=lambda g: g.x())
        no_overlap = all(a.right() < b.left() for a, b in zip(geo, geo[1:]))
        check(f"{pw} px:控制列不重疊、不溢出",
              panel.width() == pw and no_overlap and geo[-1].right() <= pw - 10,
              f"(面板 {panel.width()}、最右 {geo[-1].right()}、讀數寬 {rd.width()})")
        check(f"{pw} px:讀數寬 ≥「3D 誤差 00.00 m …」", rd.width() >= rd.min_text_width(),
              f"({rd.width()} ≥ {rd.min_text_width()})")
        check(f"{pw} px:誤差值完整可見", shown.startswith("3D 誤差 0.43 m"), f"({shown!r})")
    check("寬度不夠時以「…」截斷、text() 仍回傳完整內容",
          rd.is_elided() and _plain(rd.shown_text()).endswith("…") and rd.text() == full_readout)
    check("截斷時滑鼠提示顯示全文", "其他很長的說明文字" in rd.toolTip())
    check("面板最小寬度 ≤ 870 px(主視窗 1280 px 時約 880)", panel.minimumSizeHint().width() <= 870,
          f"({panel.minimumSizeHint().width()})")
    panel.set_readout(f"3D 誤差 <b style='color:{theme.INK}'>0.43 m</b> · ARM 2 · AUTO")
    _pump(30)
    check("1100 px:典型讀數完整顯示", not rd.is_elided(), f"({_plain(rd.shown_text())!r})")
    host.resize(1116, host.height())
    _pump(50)

    print("\n[位置資料停了之後的事件 / 訊息:範圍 = [t_base, t_last](E2E-1 / F3)]")
    late = make_late_logdata()
    lb, le, ll = late.t_base, late.t_end, late.t_last
    check("測資:t_last 涵蓋 t_end 之後的訊息", abs(ll - (le + LATE_MSG)) < 1e-9, f"({ll - lb:.1f} vs {le - lb:.1f})")
    check("log_time_range = (t_base, t_last)", uitl.log_time_range(late) == (lb, ll))
    check("log_time_range:沒有 t_last 的舊物件退回 t_end",
          uitl.log_time_range(SimpleNamespace(t_base=1.0, t_end=5.0)) == (1.0, 5.0)
          and uitl.log_time_range(SimpleNamespace(t_base=1.0, t_end=5.0, t_last=float("nan"))) == (1.0, 5.0))
    wins.clear()
    panel.chk_window_only.setChecked(True)
    panel.set_log(late)
    panel.set_rows(make_rows(late))
    _pump(30)
    check("時鐘範圍 = [t_base, t_last]", (panel.clock.lo, panel.clock.hi) == (lb, ll),
          f"({panel.clock.hi - lb:.1f})")
    check("時間軸範圍與預設可見範圍 = [t_base, t_last]",
          (tl._lo, tl._hi) == (lb, ll) and tl.view_range() == (lb, ll))
    check("預設分析區間 / 播放範圍 = [t_base, t_last]",
          panel._win == (lb, ll) and (panel.clock.play_lo, panel.clock.play_hi) == (lb, ll))
    panel.chk_window_only.setChecked(False)
    check("不勾「只播分析區間」:播放範圍 = [t_base, t_last]",
          (panel.clock.play_lo, panel.clock.play_hi) == (lb, ll))
    panel.chk_window_only.setChecked(True)
    c0 = panel.cmb_segment.itemText(0)
    check("「整份 log」= t_last 的長度", format_clock(ll - lb, 0) in c0 and panel.cmb_segment.currentIndex() == 0,
          f"({c0})")
    check("時間標籤的總長 = t_last", panel.lbl_time.text().endswith(format_clock(ll - lb)),
          f"({panel.lbl_time.text()})")
    panel.select_segment(-1)
    check("選「整份 log」→ windowRequested(t_base, t_last)", wins == [(lb, ll)], f"({wins})")
    panel.set_window(lb, ll)
    check("set_window(t_base, t_last) → 選單第 0 項", panel.cmb_segment.currentIndex() == 0)
    _pump(30)
    late_marks = [(x, k, s) for x, k, s in tl._event_marks() if tl._x2t(x) > le + 1]
    check("t_end 之後的 ERR / EV 標記有畫(2 個 ▲ + 1 個 ◆)",
          sorted(k for _, k, _ in late_marks) == sorted([EV_ERROR, EV_ERROR, EV_EVENT]), f"({late_marks})")
    img = tl.grab().toImage()
    got = _pixel(img, tl._t2x(le + LATE_ERR1), tl._mark_y() + 1)
    check("像素探針:t_end 之後的 ERR 三角形為 S_CRIT 色", _close(got, _blend(theme.S_CRIT, 255), 12), f"({got})")
    tip = tl._tooltip_at(QPointF(tl._t2x(le + LATE_ERR1) + 1, AXIS_H + 12))
    check("t_end 之後的 ERR 提示", "GPS:Glitch (2)" in tip, f"({tip[:80]}…)")
    panel.seek(le - 5.0)
    hops = []
    for _ in range(4):
        panel.jump(+1)
        hops.append(round(panel.time() - le, 3))
    check("PgDn 跳得到 t_end 之後的 ERR / EV,最後到 t_last",
          hops == [LATE_ERR1, LATE_ERR2, LATE_EV, LATE_MSG], f"({hops})")
    panel.jump(-1)
    check("PgUp 從 t_last 回到 t_end 之後的 EV", abs(panel.time() - (le + LATE_EV)) < 1e-9)
    panel.seek(le + 5.0)
    check("seek 到 t_end 之後的訊息時刻不被夾回 t_end", abs(panel.time() - (le + 5.0)) < 1e-9,
          f"({panel.time() - le:+.2f})")
    act["跳到播放範圍終點"].trigger()
    check("End → t_last", abs(panel.time() - ll) < 1e-9)
    panel.seek(le + LATE_ERR1 - 3.0)
    panel.set_speed(2)
    panel.play()
    _pump(300)
    check("t_end 之後照常播放", panel.is_playing() and panel.time() > le + LATE_ERR1 - 3.0 + 0.2,
          f"({panel.time() - le:+.2f})")
    panel.pause()
    panel.set_speed(1)
    tl.set_view(le + 50, le + 100)
    check("可見範圍可以整段落在 t_end 之後", tl.view_range() == (le + 50, le + 100))
    tl.set_view(le + 50, le + 500)
    v0, v1 = tl.view_range()
    check("超出 t_last 時平移回來(跨度不變)", abs(v1 - ll) < 1e-9 and abs((v1 - v0) - 450) < 1e-9,
          f"({v0 - le:+.1f}–{v1 - le:+.1f})")
    tl.reset_view()

    real = os.environ.get("VIO_REAL_LOG") or r"C:\Users\admin\Downloads\203 2025-3-7 上午 10-53-08.bin"
    if os.path.isfile(real):
        from core import logio
        rl = logio.load_bin(real)
        panel.set_log(rl)
        _pump(30)
        errs_after = [e for e in rl.events if e.kind == EV_ERROR and e.t > rl.t_end]
        marks_after = [m for m in tl._event_marks() if m[1] == EV_ERROR and tl._x2t(m[0]) > rl.t_end]
        check("真實 log:範圍到最後一筆訊息(8:10.8)、ERR 標記全部畫出",
              panel.clock.hi == rl.t_last and len(errs_after) >= 1 and len(marks_after) == len(errs_after),
              f"({format_clock(rl.t_last - rl.t_base)}、t_end 之後 ERR {len(errs_after)} 個 / 標記 {len(marks_after)})")
        if errs_after:
            panel.seek(rl.t_end)
            panel.jump(+1)
            check("真實 log:從位置資料終點 PgDn 到第一個 GPS Glitch",
                  abs(panel.time() - errs_after[0].t) < 1e-9, f"({format_clock(panel.time() - rl.t_base)})")
    else:
        print(f"  (略過真實 log 檢查:找不到 {real})")
    panel.set_log(ld)
    panel.set_rows(make_rows(ld))
    _pump(30)

    check("以上操作沒有任何被攔下的例外", len(uitl.handler_errors) == n_err0,
          "".join(uitl.handler_errors[n_err0:])[-300:])

    print("\n[繪製 + 截圖]")
    panel.select_segment(0)
    t_err, e_err = make_error(segs[0].t0, segs[0].t1)
    panel.set_error(t_err, e_err)
    panel.set_window(segs[0].t0, segs[0].t1)
    panel.seek(segs[0].t0 + 95.0)
    panel.set_readout(f"<b style='color:{theme.INK}'>ARM 1</b> · AUTO · 3D 誤差 "
                      f"<b style='color:{theme.C_ERR}'>0.42 m</b>")
    _pump(100)
    for _ in range(3):                            # 重畫數次:靜態層要用快取
        panel.timeline.set_playhead(panel.time() + 0.5)
        panel.timeline.repaint()
    shot = panel.grab()
    out_dir = os.environ.get("TIMELINE_SHOT_DIR") or tempfile.gettempdir()
    out = os.path.join(out_dir, "timeline_ui.png")
    ok = shot.save(out)
    check("截圖存檔", ok and os.path.getsize(out) > 5000, f"({out}, {shot.width()}x{shot.height()})")
    check("面板高度合理", 120 <= panel.height() <= 200, f"({panel.height()} px)")
    panel.set_error(None, None)
    check("清除誤差列後仍可繪製", not panel.timeline.grab().isNull())

    print("\n[清除]")
    panel.clear()
    check("清除後控制項停用", not panel.btn_play.isEnabled() and not acts[0].isEnabled())
    check("清除後選單為空", panel.cmb_segment.count() == 0)
    check("清除後時間軸可繪製", not panel.timeline.grab().isNull())

    # 單獨的 TimelineWidget 也要能用
    solo = TimelineWidget()
    solo.resize(600, solo.sizeHint().height())
    solo.set_log(lo, hi, segs, ld.events)
    solo.set_rows(make_rows(ld))
    solo.set_window(lo, hi)
    solo.set_playhead(lo + 30)
    check("TimelineWidget 單獨繪製", not solo.grab().isNull())

    host.close()
    _pump(50)
    print(f"\n{'=' * 60}\n{'全部通過' if not failed else f'{failed} 項失敗'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
