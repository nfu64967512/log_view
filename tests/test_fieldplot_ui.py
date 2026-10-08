"""欄位圖(FieldPlot)與側欄面板(FieldBrowser / ParamsPanel / MessagesPanel)測試。

離線執行、不需人工點擊;用 duck-typed 假 LogIndex(不依賴 core/dflog.py)。
審查缺陷的迴歸測試(F2 參數歷程)另用真的 LogIndex.from_tables;使用者的真實 log 存在時多跑一段。
core/expr.py、core/presets.py 若尚未就緒,只在本測試內放一個極簡替身。

  python tests\\test_fieldplot_ui.py              # 測試
  python tests\\test_fieldplot_ui.py --shots DIR  # 另把圖與各面板截圖存到 DIR
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32" and os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402

from core import frames  # noqa: E402
from core.types import (EV_ARM, EV_DISARM, EV_MODE, KIND_GLOBAL, LogData,  # noqa: E402
                        LogEvent, TrajSource)

T0 = 500.0                      # log 時基(秒)
DUR = 600.0
ORIGIN = (23.702, 120.431, 38.0)
ARM1, DISARM1 = 60.0, 540.0
MODES = [(5.0, "STABILIZE"), (55.0, "LOITER"), (90.0, "AUTO"), (200.0, "GUIDED"),
         (215.0, "AUTO"), (330.0, "LOITER"), (420.0, "RTL"), (505.0, "LAND"), (545.0, "STABILIZE")]


# ---------------- 替身:core.expr / core.presets(只有在真的模組無法匯入時) ----------------

def _install_fallbacks() -> list[str]:
    used = []
    try:
        import core.expr  # noqa: F401
    except Exception:  # noqa: BLE001
        mod = types.ModuleType("core.expr")
        ref = re.compile(r"([A-Z][A-Z0-9_]*(?:\[\d+\])?\.[A-Za-z_]\w*)")

        class ExprError(ValueError):
            pass

        def referenced_keys(expr):
            out = []
            for k in ref.findall(expr or ""):
                if k not in out:
                    out.append(k)
            return out

        def evaluate(expr, index):
            keys = referenced_keys(expr)
            if not keys:
                raise ExprError("運算式至少要引用一個欄位")
            env = {}
            tb = None
            src = expr
            for i, k in enumerate(keys):
                r = index.series(k)
                if r is None:
                    raise ExprError(f"找不到欄位「{k}」")
                t, v = r
                if tb is None:
                    tb = t
                env[f"_v{i}"] = np.interp(tb, t, v, left=np.nan, right=np.nan)
                src = src.replace(k, f"_v{i}")
            env.update(np=np, degrees=np.degrees, atan2=np.arctan2, sqrt=np.sqrt, abs=np.abs,
                       mag=lambda *a: np.sqrt(sum(np.square(x) for x in a)))
            try:
                v = eval(src, {"__builtins__": {}}, env)  # noqa: S307 — 測試替身
            except Exception as e:  # noqa: BLE001
                raise ExprError(f"語法錯誤:{e}") from None
            return np.asarray(tb, float), np.broadcast_to(np.asarray(v, float), tb.shape).copy()

        def validate(expr, index, **_kw):
            try:
                evaluate(expr, index)
            except ExprError as e:
                return str(e)
            return None

        mod.ExprError, mod.evaluate, mod.referenced_keys, mod.validate = (
            ExprError, evaluate, referenced_keys, validate)
        sys.modules["core.expr"] = mod
        import core
        core.expr = mod
        used.append("core.expr")
    try:
        import core.presets  # noqa: F401
    except Exception:  # noqa: BLE001
        mod = types.ModuleType("core.presets")

        @dataclass
        class SeriesSpec:
            expr: str
            axis: int = 1
            color: str | None = None

        @dataclass
        class Preset:
            group: str
            name: str
            series: list = field(default_factory=list)

        presets = [
            Preset("姿態", "Roll / Pitch", [SeriesSpec("ATT.Roll"), SeriesSpec("ATT.DesRoll"),
                                            SeriesSpec("ATT.Pitch"), SeriesSpec("ATT.DesPitch")]),
            Preset("姿態", "Yaw", [SeriesSpec("ATT.Yaw"), SeriesSpec("ATT.DesYaw")]),
            Preset("振動", "VIBE", [SeriesSpec("VIBE.VibeX"), SeriesSpec("VIBE.VibeY"),
                                    SeriesSpec("VIBE.VibeZ")]),
            Preset("電源", "電池", [SeriesSpec("BAT.Volt"), SeriesSpec("BAT.Curr", axis=2)]),
            Preset("VIO", "VISP 位置", [SeriesSpec("VISP.PX"), SeriesSpec("VISP.PY")]),
            Preset("缺欄位", "全部缺", [SeriesSpec("XYZ.Q")]),
        ]

        def available_presets(index):
            out = []
            for p in presets:
                ss = [s for s in p.series if all(index.has(k) for k in
                                                 re.findall(r"[A-Z][A-Z0-9_]*(?:\[\d+\])?\.\w+", s.expr))]
                if ss:
                    out.append(Preset(p.group, p.name, ss))
            return out

        def group_presets(ps):
            g = {}
            for p in ps:
                g.setdefault(p.group, []).append(p)
            return g

        mod.SeriesSpec, mod.Preset, mod.PRESETS = SeriesSpec, Preset, presets
        mod.available_presets, mod.group_presets = available_presets, group_presets
        sys.modules["core.presets"] = mod
        import core
        core.presets = mod
        used.append("core.presets")
    return used


FALLBACKS = _install_fallbacks()

from PyQt6.QtCore import QEventLoop, QPoint, QPointF, Qt, QTimer  # noqa: E402
from PyQt6.QtGui import QColor, QWheelEvent  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from core import presets as cpresets  # noqa: E402
from ui import theme  # noqa: E402
from ui.fieldplot import EMPTY_HINT, FieldPlot, decimate, gap_indices  # noqa: E402


# ---------------- 假 LogIndex ----------------

@dataclass
class FMsgInfo:
    name: str
    fields: list
    count: int
    instances: list | None
    units: dict


@dataclass
class FParam:
    name: str
    value: float
    default: float = float("nan")
    t: float = float("nan")


@dataclass
class FLogMessage:
    t: float
    kind: str
    text: str


def _parse_key(key: str):
    m = re.match(r"^([A-Z][A-Z0-9_]*)(?:\[(\d+)\])?(?:\.(\w+))?$", key)
    if not m:
        return key, None, None
    return m.group(1), (int(m.group(2)) if m.group(2) else None), m.group(3)


class FakeIndex:
    """與 core.dflog.LogIndex 相同介面的最小替身(tables 以 "ATT" / "GPS[1]" 為鍵)。"""

    def __init__(self, tables: dict, units: dict | None = None, params=None, messages=None,
                 vehicle: str = "copter", path: str = "fake.bin"):
        self.tables = tables
        self.units = units or {}
        self._params = params or []
        self._msgs = sorted(messages or [], key=lambda m: m.t)
        self.vehicle = vehicle
        self.path = path

    def messages(self):
        out = {}
        for key, tab in self.tables.items():
            name, inst, _ = _parse_key(key)
            cols = [c for c in tab if c != "t"]
            n = len(tab["t"])
            info = out.get(name)
            if info is None:
                out[name] = FMsgInfo(name, cols, n, [inst] if inst is not None else None,
                                     dict(self.units.get(name, {})))
            else:
                info.count += n
                if inst is not None:
                    info.instances = sorted((info.instances or []) + [inst])
        return out

    def field_keys(self):
        keys = []
        for key, tab in self.tables.items():
            for c, v in tab.items():
                if c != "t" and np.asarray(v).dtype.kind in "fiub":
                    keys.append(f"{key}.{c}")
        return sorted(keys)

    def table(self, name, instance=None):
        key = f"{name}[{instance}]" if instance is not None else name
        return self.tables.get(key)

    def series(self, key):
        name, inst, f = _parse_key(key)
        tab = self.table(name, inst)
        if tab is None or f not in tab:
            return None
        v = np.asarray(tab[f])
        if v.dtype.kind not in "fiub":
            return None
        return np.asarray(tab["t"], float), v.astype(float)

    def has(self, key):
        name, inst, f = _parse_key(key)
        tab = self.table(name, inst)
        return tab is not None and (f is None or f in tab)

    def params(self):
        return list(self._params)

    def param_values(self):
        out = {}
        for p in sorted(self._params, key=lambda p: (np.nan_to_num(p.t, nan=-1e18))):
            out[p.name] = p.value
        return out

    def log_messages(self):
        return list(self._msgs)

    def t_range(self):
        ts = [tab["t"] for tab in self.tables.values() if len(tab["t"])]
        return (min(float(t[0]) for t in ts), max(float(t[-1]) for t in ts)) if ts else (0.0, 0.0)


def _att_table(rate: float, rng) -> dict:
    t = T0 + np.arange(0.0, DUR, 1.0 / rate)
    tr = t - T0
    fly = (tr > ARM1 + 10) & (tr < DISARM1 - 10)
    roll = 12 * np.sin(tr / 9.0) * fly + rng.normal(0, 0.6, t.size)
    pitch = 8 * np.sin(tr / 13.0 + 1.0) * fly + rng.normal(0, 0.5, t.size)
    yaw = np.mod(40 + tr * 1.3, 360.0)
    return {"t": t, "Roll": roll, "DesRoll": 12 * np.sin(tr / 9.0) * fly,
            "Pitch": pitch, "DesPitch": 8 * np.sin(tr / 13.0 + 1.0) * fly,
            "Yaw": yaw, "DesYaw": np.mod(yaw + rng.normal(0, 1, t.size), 360),
            "ErrRP": np.abs(rng.normal(0.02, 0.01, t.size)), "AEKF": np.full(t.size, 3)}


def make_fake_log(n_msgs: int = 400, att_rate: float = 25.0, seed: int = 5) -> LogData:
    rng = np.random.default_rng(seed)
    tables = {"ATT": _att_table(att_rate, rng)}
    tg = T0 + np.arange(0.0, DUR, 0.2)
    trg = tg - T0
    fly = (trg > ARM1) & (trg < DISARM1)
    spd = np.where(fly, 4 + 2 * np.sin(trg / 20), 0) + rng.normal(0, 0.1, tg.size)
    for inst, off in ((0, 0.0), (1, 0.3)):
        tables[f"GPS[{inst}]"] = {"t": tg, "Spd": np.abs(spd + off), "Alt": 38 + 20 * fly + off,
                                  "NSats": np.where(trg > 20, 17 - inst, 6).astype(float),
                                  "Status": np.where(trg > 20, 3, 1).astype(float)}
    tv = T0 + np.arange(0.0, DUR, 0.1)
    trv = tv - T0
    tables["VIBE"] = {"t": tv, "VibeX": 6 + 3 * np.abs(np.sin(trv / 7)) + rng.normal(0, .5, tv.size),
                      "VibeY": 7 + 2 * np.abs(np.sin(trv / 5)) + rng.normal(0, .5, tv.size),
                      "VibeZ": 15 + 5 * np.abs(np.sin(trv / 11)) + rng.normal(0, .8, tv.size),
                      "Clip": np.floor(trv / 200)}
    tables["BAT"] = {"t": tv, "Volt": 16.8 - trv * 0.003 - 0.6 * ((trv > ARM1) & (trv < DISARM1)),
                     "Curr": np.where((trv > ARM1) & (trv < DISARM1), 18 + rng.normal(0, 1.5, tv.size), 0.4)}
    # VISP 只在飛行前後有資料(中間地面段沒有)→ 斷線
    tp = T0 + np.arange(ARM1 - 15, DISARM1 + 15, 1 / 30)
    trp = tp - T0
    tables["VISP"] = {"t": tp, "PX": 25 * np.sin(trp / 40), "PY": 25 * (1 - np.cos(trp / 40)),
                      "PZ": -np.clip((trp - ARM1) / 5, 0, 20)}
    tables["MSG"] = {"t": np.array([T0 + 1.0, T0 + 2.0]), "Message": np.array(["ArduCopter V4.5.7", "Frame: QUAD"])}

    units = {"ATT": {"Roll": "deg", "Pitch": "deg", "Yaw": "deg", "DesRoll": "deg",
                     "DesPitch": "deg", "DesYaw": "deg"},
             "GPS": {"Spd": "m/s", "Alt": "m"}, "BAT": {"Volt": "V", "Curr": "A"},
             "VIBE": {"VibeX": "m/s/s", "VibeY": "m/s/s", "VibeZ": "m/s/s"},
             "VISP": {"PX": "m", "PY": "m", "PZ": "m"}}

    params = []
    for i in range(60):
        name = f"P{i:02d}_" + ("GAIN", "RATE", "MAX", "ENABLE")[i % 4]
        d = float(i % 7)
        params.append(FParam(name, d + (0.5 if i % 9 == 0 else 0.0), d, T0 + 0.5))
    params += [FParam("ANGLE_MAX", 4500.0, 3000.0, T0 + 0.5),
               FParam("ATC_RAT_RLL_P", 0.135, 0.135, T0 + 0.5),
               FParam("ATC_RAT_RLL_P", 0.15000000596, 0.135, T0 + 200.0),
               FParam("ATC_RAT_RLL_P", 0.16, 0.135, T0 + 320.0),
               FParam("WPNAV_SPEED", 500.0, 1000.0, T0 + 0.5),
               FParam("WPNAV_SPEED", 800.0, 1000.0, T0 + 100.0),
               FParam("EK3_SRC1_POSXY", 6.0, 3.0, T0 + 0.5),
               FParam("FLTMODE1", 5.0, float("nan"), T0 + 0.5),
               FParam("INS_ACCEL_FILTER", 20.0000001, 20.0, T0 + 0.5)]

    msgs = [FLogMessage(T0 + 1.0, "msg", "ArduCopter V4.5.7 (2a3dc4b7)"),
            FLogMessage(T0 + 2.0, "msg", "Frame: QUAD/X"),
            FLogMessage(T0 + 3.5, "msg", "EKF3 IMU0 initialised"),
            FLogMessage(T0 + 4.0, "msg", "EKF3 IMU0 origin set"),
            FLogMessage(T0 + 30.0, "err", "EKF 檢查:變異數過大(錯誤碼 2)"),
            FLogMessage(T0 + 31.0, "err", "EKF 檢查:已恢復(錯誤碼 0)"),
            FLogMessage(T0 + 507.0, "ev", "LAND_COMPLETE_MAYBE"),
            FLogMessage(T0 + 538.0, "ev", "LAND_COMPLETE")]
    for tt, name in MODES:
        msgs.append(FLogMessage(T0 + tt, "mode", name))
    msgs += [FLogMessage(T0 + ARM1, "arm", "ARMED"), FLogMessage(T0 + DISARM1, "disarm", "DISARMED")]
    for i in range(max(0, n_msgs - len(msgs))):
        msgs.append(FLogMessage(T0 + 5 + (DUR - 10) * i / max(n_msgs, 1), "msg",
                                f"Mission: {i} WP #{i % 17} reached · dist {i % 13}.{i % 10} m"))

    lat, lng, alt = frames.enu_to_geodetic(np.column_stack([np.zeros_like(trg), np.zeros_like(trg),
                                                            20.0 * fly]), ORIGIN)
    gps = TrajSource("GPS", KIND_GLOBAL, tg, lat=lat, lng=lng, alt=alt)
    events = [LogEvent(T0 + tt, EV_MODE, name) for tt, name in MODES]
    events += [LogEvent(T0 + ARM1, EV_ARM), LogEvent(T0 + DISARM1, EV_DISARM)]
    events.sort(key=lambda e: e.t)
    idx = FakeIndex(tables, units, params, msgs)
    return LogData(sources={"GPS": gps}, origin=ORIGIN, path="fake.bin", events=events, index=idx)


# ---------------- 工具 ----------------

def _pump(ms: int = 60):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


class Checker:
    def __init__(self):
        self.failed = 0

    def __call__(self, label, cond, extra=""):
        if cond:
            print(f"  [OK]   {label} {extra}")
        else:
            self.failed += 1
            print(f"  [FAIL] {label} {extra}")


def _canvas_point(plot: FieldPlot, t_abs: float) -> QPoint:
    """log 絕對秒 → 畫布上的 Qt 邏輯座標(主圖垂直中央)。"""
    main = plot._main
    xd, yd = main.transData.transform((t_abs - plot.t_base, 0))
    bb = main.bbox
    dpr = plot.canvas.device_pixel_ratio or 1.0
    h = plot.canvas.height()
    return QPoint(int(round(xd / dpr)), int(round(h - (bb.y0 + bb.y1) / 2 / dpr)))


def _wheel(widget, pos: QPoint, delta: int, mods=Qt.KeyboardModifier.NoModifier):
    ev = QWheelEvent(QPointF(pos), QPointF(widget.mapToGlobal(pos)), QPoint(0, 0), QPoint(0, delta),
                     Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, ev)


def _grab(widget, path: str):
    widget.grab().save(path)


# ---------------- FieldPlot ----------------

def test_decimate(check):
    print("[抽稀]")
    n = 1_000_000
    t = np.linspace(0.0, 1000.0, n)
    v = np.sin(t * 3.0) + 0.001 * t
    v[400_000] = 50.0                 # 單一尖峰必須保留
    v[600_000:600_050] = np.nan       # NaN 段 → 斷線
    t0 = time.perf_counter()
    ts, vs = decimate(t, v, 0.0, 1000.0, 1500)
    dt = (time.perf_counter() - t0) * 1000
    check("1M 點抽稀到 ≤ 2×格數 + 少量", ts.size <= 2 * 1500 + 64, f"({ts.size} 點,{dt:.1f} ms)")
    check("抽稀保留尖峰最大值", np.nanmax(vs) == 50.0)
    check("抽稀保留最小值", abs(np.nanmin(vs) - np.nanmin(v)) < 1e-12)
    check("NaN 段造成斷線", np.isnan(vs).any())
    check("輸出時間遞增", bool(np.all(np.diff(ts[~np.isnan(vs)]) >= 0)))
    ts2, _ = decimate(t, v, 100.0, 100.5, 1500)
    check("小範圍時回傳原始樣本(含兩側各一點)", 500 <= ts2.size <= 504 and ts2[0] < 100.0 < ts2[1],
          f"({ts2.size} 點)")
    tt = np.concatenate([np.arange(0, 10, 0.1), np.arange(50, 60, 0.1)])
    gs = gap_indices(tt, 2.0)
    ts3, vs3 = decimate(tt, np.ones_like(tt), 0, 60, 1000, gs)
    check("時間空洞插入 NaN 斷線", np.isnan(vs3).sum() == 1 and gs.tolist() == [100])
    check("1M 點抽稀 < 60 ms", dt < 60, f"({dt:.1f} ms)")


def test_fieldplot(check, app, shots: str | None):
    print("[FieldPlot]")
    log = make_fake_log()
    p = FieldPlot()
    p.resize(1100, 520)
    p.show()
    errs = []
    p.error.connect(errs.append)
    changes = []
    p.seriesChanged.connect(lambda: changes.append(1))
    views = []
    p.viewChanged.connect(lambda a, b: views.append((a, b)))
    seeks = []
    p.seeked.connect(seeks.append)
    _pump()
    check("沒載入時顯示佔位文字", p._main is None)

    p.set_log(log)
    _pump()
    check("空圖顯示提示", p._main is None and not p.has_content()
          and any(t.get_text() == EMPTY_HINT for ax in p.fig.axes for t in ax.texts))

    ok = p.add_series("ATT.Roll")
    _pump()
    check("add_series ATT.Roll", ok and len(p.series()) == 1 and p._main is not None)
    s0 = p.series()[0]
    check("預設序列色不是軌跡角色色", s0["color"] not in (theme.C_REF, theme.C_EST, theme.C_ERR),
          s0["color"])
    check("重複加同一欄位不會多一條", p.add_series("ATT.Roll") and len(p.series()) == 1)
    ok = p.add_series("GPS[1].Spd*3.6", axis=2)
    check("運算式序列放到軸 2", ok and p.series()[1]["axis"] == 2 and 2 in p._axes)
    check("兩條序列顏色不同", p.series()[0]["color"] != p.series()[1]["color"])
    n_err = len(errs)
    ok = p.add_series("NOPE.Field")
    check("不存在的欄位 → False + error", not ok and len(errs) == n_err + 1, errs[-1] if errs else "")
    ok = p.add_series("ATT.Roll +")
    check("語法錯誤 → False + error", not ok and len(errs) == n_err + 2)

    p.set_series_axis(1, 3)
    check("set_series_axis → 軸 3(右外)", p.series()[1]["axis"] == 3 and 3 in p._axes and 2 not in p._axes)
    p.set_series_axis(0, 4)
    check("set_series_axis → 軸 4(左外)", 4 in p._axes and p._axes[4].spines["left"].get_visible())
    p.set_series_axis(0, 1)
    p.set_series_color(0, "#ff00aa")
    check("set_series_color", p.series()[0]["color"] == "#ff00aa"
          and p._series[0].line.get_color() == "#ff00aa")
    p.set_series_visible(1, False)
    check("set_series_visible(False) → 線不畫", p._series[1].line is None and not p.series()[1]["visible"])
    p.set_series_visible(1, True)
    check("set_series_visible(True) → 線回來", p._series[1].line is not None)
    n = len(changes)
    p.remove_series(1)
    check("remove_series", len(p.series()) == 1 and len(changes) == n + 1)
    p.clear_series()
    check("clear_series → 空圖", not p.series() and p._main is None)

    pre = next(pp for pp in cpresets.available_presets(log.index)
               if all(s.expr.startswith("ATT.") for s in pp.series))
    added = p.add_preset(pre)
    _pump()
    check("add_preset 加入整組", added == len(pre.series) and len(p.series()) == len(pre.series),
          f"({pre.group}/{pre.name}: {added} 條)")
    check("模式色帶已畫出", p._rib is not None and len(p._mode_spans()) == len(MODES))
    bands = [a for a in p._main.patches]
    check("主圖有低透明度模式背景帶", len(bands) >= len(MODES) and all(b.get_alpha() <= 0.15 for b in bands))
    check("模式名稱標籤(至少數個,不重疊)", len(p._mode_texts) >= 3, f"({len(p._mode_texts)} 個)")
    rend = p.canvas.get_renderer()
    bbs = sorted((t.get_window_extent(rend) for t in p._mode_texts), key=lambda b: b.x0)
    check("模式名稱彼此不重疊", all(a.x1 <= b.x0 + 0.5 for a, b in zip(bbs, bbs[1:])))
    check("ARM 段標籤", len(p._arm_texts) == 1 and p._arm_texts[0].get_text() == "ARM 1")
    p.canvas.draw()
    xt = [lbl.get_text() for lbl in p._main.get_xticklabels() if lbl.get_text()]
    check("橫軸為 m:ss", bool(xt) and all(re.match(r"^-?\d+:\d\d", s) for s in xt), str(xt[:4]))

    # 播放游標
    t_c = T0 + 150.0
    p.set_time(t_c)
    _pump()
    vis = [a for a in p._anim if a.get_visible()]
    check("游標顯示(線 + 數值框)", p._cursor_line.get_visible() and p._box.get_visible(), f"({len(vis)} 個)")
    roll = log.index.series("ATT.Roll")
    exp = float(np.interp(t_c, *roll))
    row = next(r for r in p._box_rows if r[0].expr == "ATT.Roll")
    check("數值框顯示內插值", row[3].get_text().startswith(f"{exp:.2f}"), f"{row[3].get_text()} vs {exp:.2f}")
    check("數值框開頭是時刻與模式", p._box_head.get_text().startswith("2:30.0") and "AUTO" in p._box_head.get_text(),
          p._box_head.get_text())
    p.set_time(T0 + DUR + 100)
    check("超出視野 → 游標隱藏", not any(a.get_visible() for a in p._anim))
    p.set_time(t_c)
    if shots:
        p.resize(1100, 520)
        _pump(120)
        p.canvas.draw()
        _pump()
        _grab(p, os.path.join(shots, "fieldplot_attitude.png"))

    # 單擊 → seeked
    _pump(50)
    pt = _canvas_point(p, T0 + 300.0)
    QTest.mouseClick(p.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pt)
    _pump()
    check("單擊 → seeked(t)", len(seeks) == 1 and abs(seeks[0] - (T0 + 300.0)) < 2.0,
          f"{seeks[0] - T0:.2f}" if seeks else "無")
    check("單擊不改變視野", not views)

    # 滾輪 → 縮放 + viewChanged
    full = p.view_range()
    _wheel(p.canvas, _canvas_point(p, T0 + 300.0), 240)
    _pump(300)
    v = p.view_range()
    check("滾輪放大(以游標為中心)", v[1] - v[0] < 0.8 * (full[1] - full[0]) and v[0] < T0 + 300 < v[1],
          f"{v[0] - T0:.1f}~{v[1] - T0:.1f}")
    check("viewChanged 送出一次(合併)", len(views) == 1 and abs(views[0][0] - v[0]) < 1e-6)
    _wheel(p.canvas, _canvas_point(p, T0 + 300.0), 120, Qt.KeyboardModifier.ShiftModifier)
    _pump(300)
    v2 = p.view_range()
    check("Shift+滾輪平移(跨度不變)", abs((v2[1] - v2[0]) - (v[1] - v[0])) < 1e-6 and v2[0] != v[0])

    # set_view 不回送
    nviews = len(views)
    p.set_view(T0 + 100.0, T0 + 160.0)
    _pump(300)
    v3 = p.view_range()
    check("set_view 套用", abs(v3[0] - (T0 + 100)) < 1e-6 and abs(v3[1] - (T0 + 160)) < 1e-6)
    check("set_view 不回送 viewChanged", len(views) == nviews)
    pts = max(s.line.get_xdata().size for s in p._series)
    check("縮放後重新抽稀(1 分鐘 25 Hz → 原始點)", 1400 <= pts <= 1600, f"({pts} 點)")
    ylo, yhi = p._main.get_ylim()
    seg = roll[1][(roll[0] >= T0 + 100) & (roll[0] <= T0 + 160)]
    check("y 軸跟著視野自動縮放", ylo <= seg.min() and yhi >= seg.max()
          and (yhi - ylo) < 1.4 * (seg.max() - seg.min()) + 2)

    # 拖曳平移
    a = _canvas_point(p, T0 + 140.0)
    b = QPoint(a.x() - 120, a.y())
    QTest.mousePress(p.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    for k in range(1, 7):
        QTest.mouseMove(p.canvas, QPoint(a.x() - 20 * k, a.y()))
    QTest.mouseRelease(p.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
    _pump(300)
    v4 = p.view_range()
    check("拖曳平移(往右看)", v4[0] > v3[0] + 1 and abs((v4[1] - v4[0]) - 60) < 1e-6,
          f"{v4[0] - T0:.1f}")
    check("拖曳不觸發 seeked", len(seeks) == 1)
    check("拖曳送出 viewChanged", len(views) == nviews + 1)

    # 雙擊 → 全範圍
    QTest.mouseDClick(p.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      _canvas_point(p, T0 + 150.0))
    _pump(300)
    v5 = p.view_range()
    lo, hi = p.full_range()
    check("雙擊回到全範圍", abs(v5[0] - lo) < 1e-6 and abs(v5[1] - hi) < 1e-6)

    # 衍生序列
    te = np.arange(T0 + 100, T0 + 400, 0.2)
    p.set_extra("ATE 3D 誤差", te, 0.5 + 0.1 * np.sin(te), theme.C_ERR)
    used = {s.axis for s in p._series}
    check("set_extra 放在自己的軸", p._extra is not None and p._extra.axis not in used
          and p._extra.axis in p._axes, f"(軸 {p._extra.axis if p._extra else '-'})")
    check("set_extra 不出現在 series()", all(s["expr"] != "ATE 3D 誤差" for s in p.series()))
    p.set_time(T0 + 150.0)
    rowe = next((r for r in p._box_rows if r[0].extra), None)
    check("數值框含衍生序列", rowe is not None and rowe[3].get_text() != "—")
    p.set_extra(None)
    check("set_extra(None) 清除", p._extra is None)
    p.close()
    return p


def test_bigdata(check):
    print("[大資料]")
    log = make_fake_log(att_rate=1000.0)          # 600 秒 × 1 kHz ≈ 60 萬點 / 欄位
    p = FieldPlot()
    p.resize(1200, 500)
    p.show()
    p.set_log(log)
    t0 = time.perf_counter()
    p.add_series("ATT.Roll")
    p.add_series("ATT.Pitch")
    p.canvas.draw()
    dt = (time.perf_counter() - t0) * 1000
    n = max(s.line.get_xdata().size for s in p._series)
    check("60 萬點序列每次繪製 ≤ 10k 點", n <= 10_000, f"({n} 點)")
    t1 = time.perf_counter()
    p.set_view(T0 + 100, T0 + 400)
    p._refresh()                                  # set_view 本身延後重算;這裡量完整成本
    p.canvas.draw()
    dt2 = (time.perf_counter() - t1) * 1000
    check("加入 + 首次繪製 < 1.5 s", dt < 1500, f"({dt:.0f} ms)")
    check("換視野重繪 < 400 ms", dt2 < 400, f"({dt2:.0f} ms)")
    t2 = time.perf_counter()
    for k in range(30):
        p.set_time(T0 + 100 + k)
    dt3 = (time.perf_counter() - t2) * 1000 / 30
    check("游標 blit 每次 < 25 ms", dt3 < 25, f"({dt3:.1f} ms)")
    p.close()


def test_legacy(check):
    print("[舊資料(無 index)]")
    log = make_fake_log()
    log.index = None
    p = FieldPlot()
    errs = []
    p.error.connect(errs.append)
    p.set_log(log)
    ok = p.add_series("ATT.Roll")
    check("index None → add_series False + 提示", not ok and errs and "索引" in errs[-1], errs[-1] if errs else "")
    te = np.arange(T0, T0 + 100, 0.5)
    p.set_extra("ATE 3D 誤差", te, np.ones_like(te))
    check("index None 仍可顯示衍生序列", p._main is not None and p.has_content())
    p.close()


def test_collapsed(check):
    """整合端把欄位圖放在垂直分隔器上方、沒東西時收起(高度 0);收起時加序列再展開要正常。"""
    from PyQt6.QtWidgets import QSplitter, QWidget
    print("[收起的圖區]")
    log = make_fake_log()
    split = QSplitter(Qt.Orientation.Vertical)
    p = FieldPlot()
    split.addWidget(p)
    split.addWidget(QWidget())
    split.setCollapsible(0, True)
    split.resize(1000, 700)
    split.show()
    split.setSizes([0, 700])
    _pump()
    p.set_log(log)
    ok = p.add_series("ATT.Roll")
    p.set_time(T0 + 100.0)
    _pump()
    check("收起時加序列不出錯", ok and p.has_content())
    split.setSizes([360, 340])
    _pump(200)
    p.canvas.draw()
    _pump()
    check("展開後版面就位(圖例 + 邊界)", p._margins is not None and p._legend is not None
          and p._main.bbox.height > 200, f"(主圖高 {p._main.bbox.height:.0f} px)")
    check("展開後線有資料", p._series[0].line.get_xdata().size > 100)
    p.set_time(T0 + 100.0)
    check("展開後游標正常", p._cursor_line.get_visible() and p._box.get_visible())
    split.setSizes([0, 700])
    _pump()
    p.set_time(T0 + 120.0)
    p.set_view(T0 + 50, T0 + 150)
    _pump(100)
    check("再收起時 set_time / set_view 不出錯", abs(p.view_range()[0] - (T0 + 50)) < 1e-6)
    split.close()


# ---------------- 側欄 ----------------

def _visible_leaves(tree, role):
    out = []

    def walk(it, hidden):
        hidden = hidden or it.isHidden()
        if it.childCount() == 0:
            if not hidden and it.data(0, role):
                out.append(it.data(0, role))
            return
        for k in range(it.childCount()):
            walk(it.child(k), hidden)

    for i in range(tree.topLevelItemCount()):
        walk(tree.topLevelItem(i), False)
    return out


def _find_item(tree, text):
    def walk(it):
        if it.text(0) == text:
            return it
        for k in range(it.childCount()):
            r = walk(it.child(k))
            if r is not None:
                return r
        return None

    for i in range(tree.topLevelItemCount()):
        r = walk(tree.topLevelItem(i))
        if r is not None:
            return r
    return None


def test_field_browser(check, shots: str | None):
    from ui.sidebar import _ROLE_KEY, _ROLE_PRESET, ExpressionDialog, FieldBrowser
    print("[FieldBrowser]")
    log = make_fake_log()
    plot = FieldPlot()
    plot.resize(900, 420)
    fb = FieldBrowser()
    fb.resize(380, 820)
    fb.show()
    fb.bind(plot)
    fb.set_index(None)
    check("index None → 顯示提示、內容隱藏", fb.hint.isVisible() and not fb.split.isVisible())
    plot.set_log(log)
    fb.set_index(log.index)
    _pump()
    keys_all = _visible_leaves(fb.tree_fields, _ROLE_KEY)
    check("欄位樹列出所有數值欄位(字串欄位不列)", sorted(keys_all) == log.index.field_keys()
          and not any(k.startswith("MSG.") for k in keys_all), f"({len(keys_all)} 個)")
    gps = _find_item(fb.tree_fields, "GPS")
    check("有 instance 的訊息 → instance 子節點", gps is not None and gps.childCount() == 2
          and gps.child(1).text(0) == "GPS[1]")

    fb.search.setText("roll")
    fb.apply_filter()
    vis = _visible_leaves(fb.tree_fields, _ROLE_KEY)
    check("搜尋欄位名(不分大小寫)", set(vis) == {"ATT.Roll", "ATT.DesRoll"}, str(vis))
    att = _find_item(fb.tree_fields, "ATT")
    check("欄位符合時展開訊息", att.isExpanded())
    fb.search.setText("VIBE")
    fb.apply_filter()
    vis = _visible_leaves(fb.tree_fields, _ROLE_KEY)
    check("搜尋訊息名 → 該訊息全部欄位", set(vis) == {k for k in keys_all if k.startswith("VIBE.")}, str(vis))
    fb.search.setText("gps[1].sp")
    fb.apply_filter()
    vis = _visible_leaves(fb.tree_fields, _ROLE_KEY)
    check("搜尋完整 key(含 instance)", vis == ["GPS[1].Spd"], str(vis))
    fb.search.setText("")
    fb.apply_filter()
    check("清除搜尋 → 全部回來", len(_visible_leaves(fb.tree_fields, _ROLE_KEY)) == len(keys_all))

    # 勾選欄位 → 繪圖
    fb.cmb_axis.setCurrentIndex(1)               # 軸 2
    it = fb._field_items["GPS[1].Spd"]
    it.setCheckState(0, Qt.CheckState.Checked)
    _pump()
    s = plot.series()
    check("勾選欄位 → 畫到選定的軸", len(s) == 1 and s[0]["expr"] == "GPS[1].Spd" and s[0]["axis"] == 2)
    check("已繪製清單同步", fb.list_plotted.count() == 1 and fb.lbl_plotted_n.text() == "1 條")
    it.setCheckState(0, Qt.CheckState.Unchecked)
    check("取消勾選 → 從圖移除", not plot.series())
    fb.cmb_axis.setCurrentIndex(0)
    roll = fb._field_items["ATT.Roll"]
    fb._on_field_double(roll, 0)
    check("雙擊欄位 → 繪圖並勾選", [x["expr"] for x in plot.series()] == ["ATT.Roll"]
          and roll.checkState(0) == Qt.CheckState.Checked)
    plot.remove_series(0)
    check("圖上移除 → 樹的勾選同步取消", roll.checkState(0) == Qt.CheckState.Unchecked)

    # 預設組合
    first = None
    for g in range(fb.tree_presets.topLevelItemCount()):
        gi = fb.tree_presets.topLevelItem(g)
        if gi.childCount():
            first = gi.child(0)
            break
    check("預設組合樹有內容", first is not None and len(fb._presets) >= 3, f"({len(fb._presets)} 組)")
    p = fb._presets[int(first.data(0, _ROLE_PRESET))]
    fb._on_preset_clicked(first, 0)
    check("點預設組合 → 整組畫上", {x["expr"] for x in plot.series()} == {s.expr for s in p.series}
          and first.checkState(0) == Qt.CheckState.Checked, f"({p.name})")
    plot.remove_series(0)
    check("部分在圖上 → 方塊狀態", first.checkState(0) == Qt.CheckState.PartiallyChecked)
    fb._on_preset_clicked(first, 0)
    check("再點 → 補齊", first.checkState(0) == Qt.CheckState.Checked)
    fb._on_preset_clicked(first, 0)
    check("整組已在圖上時再點 → 移除", not plot.series())
    fb.search.setText("vibe")
    fb.apply_filter()
    vis_p = [fb._presets[i].name for i in _visible_leaves(fb.tree_presets, _ROLE_PRESET)]
    check("搜尋也篩選預設組合", bool(vis_p) and all(
        any("VIBE" in s.expr for s in fb._presets[i].series) or "振動" in fb._presets[i].group
        for i in _visible_leaves(fb.tree_presets, _ROLE_PRESET)), str(vis_p[:3]))
    fb.search.setText("")
    fb.apply_filter()

    # 已繪製清單:顏色、軸、顯示、移除
    plot.add_series("ATT.Roll")
    plot.add_series("BAT.Volt", 2)
    row = fb.list_plotted.itemWidget(fb.list_plotted.item(1))
    fb._pick_color = lambda initial: QColor("#123456")
    row.swatch.click()
    check("色塊 → 換顏色", plot.series()[1]["color"] == "#123456")
    row = fb.list_plotted.itemWidget(fb.list_plotted.item(1))
    row.axis.setCurrentIndex(2)
    row.axis.activated.emit(2)
    check("清單改軸", plot.series()[1]["axis"] == 3)
    row = fb.list_plotted.itemWidget(fb.list_plotted.item(0))
    row.chk.setChecked(False)
    check("清單取消顯示", not plot.series()[0]["visible"])
    row = fb.list_plotted.itemWidget(fb.list_plotted.item(0))
    row.chk.setChecked(True)
    row = fb.list_plotted.itemWidget(fb.list_plotted.item(0))
    row.btn_rm.click()
    check("清單移除", [x["expr"] for x in plot.series()] == ["BAT.Volt"])
    fb.btn_clear.click()
    check("清除全部", not plot.series() and fb.list_plotted.count() == 0)

    # 錯誤顯示
    plot.add_series("NOPE.X")
    check("圖的 error → 面板顯示錯誤", fb.status.isVisible() and fb.status.text().startswith("錯誤"),
          fb.status.text())

    # 運算式對話框
    dlg = ExpressionDialog(log.index, 2, fb)
    dlg.edit.setText("GPS[1].Spd*3.6")
    err = dlg.validate_now()
    check("運算式有效 → 可按加入", err is None and dlg.btn_ok.isEnabled()
          and "有效" in dlg.status.text(), dlg.status.text())
    dlg.edit.setText("GPS[1].Spd*")
    err = dlg.validate_now()
    check("語法錯誤 → 顯示錯誤、不能按", bool(err) and not dlg.btn_ok.isEnabled()
          and "錯誤" in dlg.status.text(), dlg.status.text())
    dlg.edit.setText("ATT.Rol")
    err = dlg.validate_now()
    check("欄位不存在 → 錯誤", bool(err), err or "")
    dlg.edit.setText("")
    dlg.edit.setFocus()
    QTest.keyClicks(dlg.edit, "VIBE.Vi")
    _pump()
    model = dlg.completer.completionModel()
    comps = [model.data(model.index(r, 0)) for r in range(model.rowCount())]
    check("自動完成列出符合的欄位", {"VIBE.VibeX", "VIBE.VibeY", "VIBE.VibeZ"} <= set(comps), str(comps[:4]))
    dlg.completer.popup().hide()
    dlg._insert_completion("VIBE.VibeZ")
    check("自動完成只替換游標前的欄位", dlg.edit.text() == "VIBE.VibeZ")
    dlg.edit.setText("mag(VIBE.VibeX, VIBE.Vi")
    dlg.edit.setCursorPosition(len(dlg.edit.text()))
    dlg._insert_completion("VIBE.VibeY")
    check("自動完成在運算式中間", dlg.edit.text() == "mag(VIBE.VibeX, VIBE.VibeY")
    dlg.edit.setText("mag(VIBE.VibeX, VIBE.VibeY, VIBE.VibeZ)")
    _pump(400)
    check("輸入後延遲自動檢查", dlg.is_valid() and dlg.btn_ok.isEnabled())
    check("對話框結果(運算式 / 軸)", dlg.expression() == "mag(VIBE.VibeX, VIBE.VibeY, VIBE.VibeZ)"
          and dlg.axis() == 2)
    if shots:
        dlg.edit.setText("degrees(atan2(VISP.PY, VISP.PX))")
        dlg.validate_now()
        dlg.show()
        _pump(80)
        _grab(dlg, os.path.join(shots, "expr_dialog.png"))
    dlg.close()

    if shots:
        fb.status.hide()
        fb.cmb_axis.setCurrentIndex(0)
        plot.add_preset(next(pp for pp in fb._presets if pp.series[0].expr.startswith("ATT.")))
        plot.add_series("GPS[1].Spd", 2)
        fb.search.setText("")
        fb.apply_filter()
        _find_item(fb.tree_fields, "ATT").setExpanded(True)
        _find_item(fb.tree_fields, "GPS").setExpanded(True)
        _find_item(fb.tree_fields, "GPS[1]").setExpanded(True)
        _pump(80)
        _grab(fb, os.path.join(shots, "sidebar_fields.png"))
        fb.search.setText("spd")
        fb.apply_filter()
        _pump(80)
        _grab(fb, os.path.join(shots, "sidebar_fields_search.png"))
    fb.close()
    plot.close()


def test_params(check, shots: str | None, tmp: str):
    from ui.sidebar import ParamsPanel, fmt_param
    print("[ParamsPanel]")
    log = make_fake_log()
    pp = ParamsPanel()
    pp.resize(380, 640)
    pp.show()
    check("index None → 提示", pp.hint.isVisible() and not pp.table.isVisible())
    pp.set_index(log.index, log.t_base)
    _pump()
    rows = {r.name: r for r in pp.model.rows}
    check("每個參數一列(重複記錄合併)", len(rows) == len(pp.model.rows) and "ATC_RAT_RLL_P" in rows,
          f"({len(rows)} 個)")
    r = rows["ATC_RAT_RLL_P"]
    check("記錄中變更 → 次數與最後的值", r.changes == 2 and abs(r.value - 0.16) < 1e-9, f"{r.changes} 次")
    tip = pp.model.history_tip(r)
    check("變更歷程提示(相對時間 → 值)", "0:00.5 → 0.135" in tip and "3:20.0 → 0.15" in tip,
          tip.replace("\n", " | "))
    check("float32 誤差不算已修改", rows["INS_ACCEL_FILTER"].modified is False)
    check("與預設不同 → 已修改", rows["ANGLE_MAX"].modified is True
          and pp.model.status_text(rows["ANGLE_MAX"]) == "已修改")
    check("沒有預設值 → 狀態未知", rows["FLTMODE1"].modified is None)
    check("數值格式", fmt_param(4500.0) == "4500" and fmt_param(0.15000000596) == "0.15"
          and fmt_param(-0.0001) == "-0.0001")
    n_all = pp.proxy.rowCount()
    pp.chk_modified.setChecked(True)
    mod = pp.visible_names()
    expect = sorted(n for n, r in rows.items() if r.modified or r.changes)
    check("只看已修改", sorted(mod) == expect and len(mod) < n_all, f"({len(mod)} / {n_all})")
    pp.chk_modified.setChecked(False)
    pp.filter.setText("wpnav")
    check("名稱篩選(不分大小寫)", pp.visible_names() == ["WPNAV_SPEED"])
    if shots:
        pp.filter.setText("")
        pp.table.selectRow(2)
        _pump(60)
        _grab(pp, os.path.join(shots, "sidebar_params.png"))
    pp.filter.setText("ATC")
    pp.table.selectAll()
    txt = pp.copy_selection()
    check("複製選取 → 名稱,數值", txt == "ATC_RAT_RLL_P,0.16"
          and QApplication.clipboard().text() == txt, txt)
    pp.filter.setText("")
    path = os.path.join(tmp, "out.param")
    n = pp.export_param(path)
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    body = [ln for ln in lines if not ln.startswith("#")]
    check(".param 匯出:每行 名稱,數值、依名稱排序", n == len(rows) and len(body) == n
          and body == sorted(body) and "ANGLE_MAX,4500" in body and "ATC_RAT_RLL_P,0.16" in body,
          f"({n} 行)")
    check(".param 每行剛好兩欄", all(len(ln.split(",")) == 2 for ln in body))
    pp.set_index(None)
    check("set_index(None) → 提示", pp.hint.isVisible() and pp.model.rowCount() == 0)
    pp.close()


def test_messages(check, shots: str | None):
    from ui.sidebar import MessagesPanel
    print("[MessagesPanel]")
    log = make_fake_log(n_msgs=20000)
    mp = MessagesPanel()
    mp.resize(380, 640)
    mp.show()
    check("logdata None → 提示", mp.hint.isVisible() and not mp.table.isVisible())
    seeks = []
    mp.seekRequested.connect(seeks.append)
    t0 = time.perf_counter()
    mp.set_log(log)
    _pump()
    dt = (time.perf_counter() - t0) * 1000
    total = len(log.index.log_messages())
    check("2 萬筆載入", mp.model.rowCount() == total, f"({total} 筆,{dt:.0f} ms)")
    check("時間欄為相對 m:ss.s", mp.model.data(mp.model.index(0, 0)) == "0:01.0")
    for key in ("msg", "ev", "mode"):
        mp.kind_checks[key].setChecked(False)
    kinds = {mp.model.kind[i] for i in mp.model.vis}
    check("種類篩選(ARM 含 DISARM)", kinds == {"err", "arm", "disarm"}, str(sorted(kinds)))
    for key in ("msg", "ev", "mode"):
        mp.kind_checks[key].setChecked(True)
    mp.filter.setText("ekf")
    mp.apply_filter()
    texts = [mp.model.text[i] for i in mp.model.vis]
    check("文字篩選(不分大小寫)", len(texts) == 4 and all("EKF" in t for t in texts), str(len(texts)))
    mp.filter.setText("")
    mp.apply_filter()

    t_q = T0 + 150.0
    t1 = time.perf_counter()
    mp.set_time(t_q)
    dt2 = (time.perf_counter() - t1) * 1000
    r = mp.current_row()
    check("set_time 標出最後一筆 ≤ t", r >= 0 and mp.model.time_of(r) <= t_q
          and (r + 1 >= mp.model.rowCount() or mp.model.time_of(r + 1) > t_q), f"(列 {r},{dt2:.2f} ms)")
    bg = mp.model.data(mp.model.index(r, 0), Qt.ItemDataRole.BackgroundRole)
    check("目前列有底色 + ▶", bg is not None and mp.model.data(mp.model.index(r, 0)).startswith("▶"))
    _pump()
    vr = mp.table.viewport().rect()
    rect = mp.table.visualRect(mp.model.index(r, 0))
    check("跟隨播放頭 → 捲到目前列", vr.intersects(rect))
    t2 = time.perf_counter()
    for k in range(200):
        mp.set_time(T0 + 10 + k * 2.5)
    dt3 = (time.perf_counter() - t2) * 1000 / 200
    check("set_time 夠快(播放時每格呼叫)", dt3 < 5, f"({dt3:.2f} ms/次)")
    mp.chk_follow.setChecked(False)
    mp.table.scrollToTop()
    mp.set_time(T0 + 590)
    check("不跟隨 → 不捲動", mp.table.rowAt(0) == 0)
    mp.chk_follow.setChecked(True)

    mp.set_time(t_q)
    _pump()
    r = mp.current_row()
    idx = mp.model.index(r, 2)
    rect = mp.table.visualRect(idx)
    # Qt6 的 QTest.mouseDClick 不含前一次按下,item view 需先單擊才認得雙擊
    QTest.mouseClick(mp.table.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     rect.center())
    QTest.mouseDClick(mp.table.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      rect.center())
    _pump()
    check("雙擊 → seekRequested(t)", bool(seeks) and abs(seeks[-1] - mp.model.time_of(r)) < 1e-9)
    mp.table.setCurrentIndex(mp.model.index(3, 0))
    QTest.keyClick(mp.table, Qt.Key.Key_Return)
    check("Enter → seekRequested", len(seeks) >= 2 and abs(seeks[-1] - mp.model.time_of(3)) < 1e-9)
    if shots:
        small = make_fake_log(n_msgs=120)
        mp.set_log(small)
        mp.set_time(T0 + 92)
        _pump(80)
        _grab(mp, os.path.join(shots, "sidebar_messages.png"))
    mp.set_log(None)
    check("set_log(None) → 提示", mp.hint.isVisible() and mp.model.rowCount() == 0)
    mp.close()


# ---------------- 迴歸測試(審查發現的缺陷) ----------------

def _small_log(tables: dict, units: dict | None = None, msgs=None, params=None,
               t_end: float | None = None) -> LogData:
    """最小的 LogData:一條位置來源(GPS,到 t_end)+ 給定的欄位表。"""
    t_end = T0 + 60.0 if t_end is None else t_end
    tg = np.arange(T0, t_end + 1e-9, 0.2)
    lat, lng, alt = frames.enu_to_geodetic(np.zeros((tg.size, 3)), ORIGIN)
    gps = TrajSource("GPS", KIND_GLOBAL, tg, lat=lat, lng=lng, alt=alt)
    idx = FakeIndex(tables, units or {}, params or [], msgs or [])
    return LogData(sources={"GPS": gps}, origin=ORIGIN, path="small.bin", events=[], index=idx)


def _ang_dist(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_cursor_values(check):
    """F4:數值框只顯示 log 裡可能出現的值(航向不在北方交界讀成 180、計數不出現 0.5)。"""
    from ui.fieldplot import K_HEADING, K_LINEAR, K_STEP, series_kind
    print("[游標數值(F4)]")
    ta = T0 + np.arange(0.0, 60.0, 1 / 25)
    tra = ta - T0
    tv = T0 + 0.013 + np.arange(0.0, 60.0, 1 / 30)
    trv = tv - T0
    tgs = T0 + np.arange(0.0, 60.0, 0.2)
    tables = {
        "ATT": {"t": ta, "Yaw": np.mod(350.0 + 2.0 * tra, 360.0), "Roll": 10 * np.sin(tra / 3)},
        "VISP": {"t": tv, "Yaw": (170.0 + 2.0 * trv + 180.0) % 360.0 - 180.0,   # ±180,5 s 時跨過 180
                 "ResetCnt": (trv >= 20.0).astype(float), "PX": 5 * np.cos(trv / 4),
                 "PY": 5 * np.sin(trv / 4)},
        "GPS": {"t": tgs, "Status": np.where(tgs - T0 >= 10.0, 3.0, 1.0)},
        "MODE": {"t": np.array([T0 + 1.0, T0 + 30.0]), "ModeNum": np.array([0.0, 3.0])},
    }
    units = {"ATT": {"Yaw": "degheading", "Roll": "deg"}, "VISP": {"Yaw": "deg"}}
    log = _small_log(tables, units)

    kinds = {e: series_kind(e, u, cexpr_eval(e, log.index))
             for e, u in (("ATT.Yaw", "degheading"), ("VISP.Yaw", "deg"), ("ATT.Roll", "deg"),
                          ("VISP.ResetCnt", ""), ("GPS.Status", ""), ("wrap_180(ATT.Yaw)", ""),
                          ("degrees(atan2(VISP.PY, VISP.PX))", ""), ("VISP.ResetCnt * 10", ""),
                          ("wrap_180(ATT.Yaw) + 1", ""), ("ATT.Roll * 2", ""))}
    check("分類:degheading → 航向(0~360)", kinds["ATT.Yaw"] == (K_HEADING, 360.0), str(kinds["ATT.Yaw"]))
    check("分類:名為 Yaw 的 deg 欄位(±180)→ 航向(±180)", kinds["VISP.Yaw"] == (K_HEADING, 180.0),
          str(kinds["VISP.Yaw"]))
    check("分類:連續量 → 線性", kinds["ATT.Roll"][0] == K_LINEAR and kinds["ATT.Roll * 2"][0] == K_LINEAR)
    check("分類:計數 / 狀態 / 整數運算式 → 階梯", all(kinds[k][0] == K_STEP for k in
                                               ("VISP.ResetCnt", "GPS.Status", "VISP.ResetCnt * 10")))
    check("分類:整條是 wrap_180(…) / degrees(atan2(…)) → 航向(±180)",
          kinds["wrap_180(ATT.Yaw)"] == (K_HEADING, 180.0)
          and kinds["degrees(atan2(VISP.PY, VISP.PX))"] == (K_HEADING, 180.0))
    check("分類:wrap_180(…) + 1 不是整條包角 → 不當航向", kinds["wrap_180(ATT.Yaw) + 1"][0] != K_HEADING)

    p = FieldPlot()
    p.resize(1000, 480)
    p.show()
    p.set_log(log)
    for e in ("ATT.Yaw", "ATT.Roll", "VISP.Yaw", "VISP.ResetCnt", "GPS.Status", "MODE.ModeNum",
              "wrap_180(ATT.Yaw)"):
        p.add_series(e)
    _pump()
    S = {s.expr: s for s in p._series}

    def row_text(expr):
        return next(r[3].get_text() for r in p._box_rows if r[0].expr == expr)

    # ATT.Yaw:359.x → 0.x 中間
    ya = tables["ATT"]["Yaw"]
    j = int(np.flatnonzero(np.diff(ya) < -180)[0]) + 1
    tm = 0.5 * (ta[j - 1] + ta[j])
    p.set_time(tm)
    va = S["ATT.Yaw"].value_at(tm)
    check("航向跨北方:讀值在 0/360 附近(不是 180)", va is not None and _ang_dist(va, 0.0) < 1.0
          and 0.0 <= va < 360.0, f"({ya[j - 1]:.2f} → {ya[j]:.2f}:{va})")
    txt = row_text("ATT.Yaw")
    check("數值框文字也不是 180(四捨五入也不出現 360.0)", _ang_dist(float(txt.split()[0]), 0.0) < 1.0
          and not txt.startswith("360"), txt)
    dot = S["ATT.Yaw"].dot
    check("游標點與讀值一致", dot.get_visible() and abs(float(dot.get_ydata()[0]) - va) < 1e-9)
    w = S["wrap_180(ATT.Yaw)"].value_at(tm)
    check("wrap_180(ATT.Yaw) 跨 0 正常", w is not None and abs(w) < 1.0, f"({w})")
    # VISP.Yaw:179.x → -179.x 中間
    yv = tables["VISP"]["Yaw"]
    k = int(np.flatnonzero(np.abs(np.diff(yv)) > 180)[0]) + 1
    tm2 = tv[k - 1] + 0.5 * (tv[k] - tv[k - 1])
    vv = S["VISP.Yaw"].value_at(tm2)
    check("±180 航向跨 180:讀值在 ±180 附近(不是 0)、範圍 (-180, 180]",
          vv is not None and abs(abs(vv) - 180.0) < 1.0 and -180.0 < vv <= 180.0,
          f"({yv[k - 1]:.2f} → {yv[k]:.2f}:{vv})")
    # 一般時刻的航向 = 正常內插
    tq = T0 + 30.013
    exp = float(np.interp(tq, ta, np.degrees(np.unwrap(np.radians(ya))))) % 360.0
    check("航向一般時刻 = 內插值", _ang_dist(S["ATT.Yaw"].value_at(tq), exp) < 1e-6)
    # 計數 0 → 1:兩筆之間取前一筆(0),不是 0.5
    rc = tables["VISP"]["ResetCnt"]
    k1 = int(np.flatnonzero(np.diff(rc) > 0)[0]) + 1
    t_rc = tv[k1 - 1] + 0.9 * (tv[k1] - tv[k1 - 1])
    p.set_time(t_rc)
    check("計數:兩筆之間 = 前一筆(0),不是內插", S["VISP.ResetCnt"].value_at(t_rc) == 0.0
          and row_text("VISP.ResetCnt") == "0", row_text("VISP.ResetCnt"))
    check("計數:恰在新樣本時刻 = 新值", S["VISP.ResetCnt"].value_at(float(tv[k1])) == 1.0)
    rng = np.random.default_rng(3)
    ok = True
    for e in ("VISP.ResetCnt", "GPS.Status", "MODE.ModeNum"):
        vals = set(np.asarray(S[e].v).tolist())
        for tt in T0 + rng.uniform(0, 60, 400):
            v = S[e].value_at(float(tt))
            ok &= v is None or v in vals
    check("整數 / 狀態 / 模式:隨機 1200 個時刻的讀值都是 log 裡的值", ok)
    p.set_time(T0 + 20.0)
    check("稀疏模式編號:兩次切換之間 = 目前模式(0,不是 1.5)",
          S["MODE.ModeNum"].value_at(T0 + 15.5) == 0.0 and row_text("MODE.ModeNum") == "0",
          row_text("MODE.ModeNum"))
    check("整數序列畫成階梯、連續量照常", S["GPS.Status"].line.get_drawstyle() == "steps-post"
          and S["ATT.Roll"].line.get_drawstyle() == "default")
    tq = T0 + 12.345
    roll = log.index.series("ATT.Roll")
    check("連續量仍是線性內插", abs(S["ATT.Roll"].value_at(tq) - float(np.interp(tq, *roll))) < 1e-9)
    check("階梯線在資料空洞前斷開(不延伸到空洞中點)",
          _hold_ok(decimate(np.array([0.0, 1.0, 2.0, 50.0, 51.0]), np.array([1.0, 1, 2, 3, 3]),
                            0, 51, 100, np.array([3]), hold=1.0)))
    p.close()


def cexpr_eval(expr, index):
    from core import expr as cexpr
    return cexpr.evaluate(expr, index)[1]


def _hold_ok(r) -> bool:
    ts, vs = r
    i = int(np.flatnonzero(np.isnan(vs))[0])
    return abs(ts[i] - 3.0) < 1e-9


def test_unplottable_preset(check):
    """E2E-4:預設組合含全 NaN 欄位(CTUN.DSAlt)→ 能畫的都畫上算「整組」,再點一下可移除。"""
    from ui.sidebar import _ROLE_PRESET, FieldBrowser
    print("[預設組合含畫不出來的欄位(E2E-4)]")
    if "core.presets" in FALLBACKS:
        check("(略過:core.presets 無法匯入)", True)
        return
    tc = T0 + np.arange(0.0, 60.0, 0.1)
    trc = tc - T0
    tables = {"CTUN": {"t": tc, "SAlt": 0.2 + trc * 0.1, "DSAlt": np.full(tc.size, np.nan),
                       "TAlt": 5 + 0 * trc, "Alt": trc * 0.1}}
    log = _small_log(tables)
    plot = FieldPlot()
    plot.resize(800, 400)
    errs = []
    plot.error.connect(errs.append)
    fb = FieldBrowser()
    fb.bind(plot)
    plot.set_log(log)
    fb.set_index(log.index)
    item = None
    for g in range(fb.tree_presets.topLevelItemCount()):
        gi = fb.tree_presets.topLevelItem(g)
        for k in range(gi.childCount()):
            if fb._presets[int(gi.child(k).data(0, _ROLE_PRESET))].name == "測距 / 地形高度":
                item = gi.child(k)
    check("預設「測距 / 地形高度」有列出(欄位存在)", item is not None)
    if item is None:
        return
    p = fb._presets[int(item.data(0, _ROLE_PRESET))]
    check("組內含全 NaN 的 CTUN.DSAlt", "CTUN.DSAlt" in {s.expr for s in p.series})
    fb._on_preset_clicked(item, 0)
    got = {s["expr"] for s in plot.series()}
    check("點一下 → 能畫的畫上、DSAlt 報錯", got == {"CTUN.SAlt", "CTUN.TAlt"} and errs
          and "CTUN.DSAlt" in errs[-1], str(sorted(got)))
    check("FieldPlot.is_unplottable(DSAlt)", plot.is_unplottable("CTUN.DSAlt")
          and not plot.is_unplottable("CTUN.SAlt"))
    check("勾選狀態 = 勾(不是方塊)", item.checkState(0) == Qt.CheckState.Checked)
    check("提示標出畫不出來的欄位", "沒有可畫的資料" in item.toolTip(0))
    n_err = len(errs)
    fb._on_preset_clicked(item, 0)
    check("再點一下 → 整組移除(不再重報錯)", not plot.series() and len(errs) == n_err
          and item.checkState(0) == Qt.CheckState.Unchecked)
    # 換 log → 清掉「畫不出來」的記錄;先個別畫上能畫的,再點組合:沒新增也要更新狀態
    plot.set_log(log)
    fb.set_index(log.index)
    check("set_log 重設 is_unplottable", not plot.is_unplottable("CTUN.DSAlt"))
    item = next(gi.child(k) for g in range(fb.tree_presets.topLevelItemCount())
                for gi in [fb.tree_presets.topLevelItem(g)] for k in range(gi.childCount())
                if fb._presets[int(gi.child(k).data(0, _ROLE_PRESET))].name == "測距 / 地形高度")
    fb.plot_field("CTUN.SAlt")
    fb.plot_field("CTUN.TAlt")
    check("個別畫上能畫的 → 方塊(還不知道 DSAlt 畫不出來)",
          item.checkState(0) == Qt.CheckState.PartiallyChecked)
    fb._on_preset_clicked(item, 0)
    check("點組合(只剩畫不出來的)→ 狀態更新成勾", item.checkState(0) == Qt.CheckState.Checked
          and len(plot.series()) == 2)
    fb._on_preset_clicked(item, 0)
    check("再點 → 移除", not plot.series())
    fb.close()
    plot.close()


def test_clock_past_hour(check):
    """E2E-5:超過 1 小時放大到秒以下,刻度不可全部一樣;數值框時刻保留小數。"""
    from ui.fieldplot import fmt_clock
    print("[超過 1 小時的時間刻度(E2E-5)]")
    check("fmt_clock 一小時以上帶小數", fmt_clock(3700.25, 2) == "1:01:40.25"
          and fmt_clock(3700.0, 0) == "1:01:40" and fmt_clock(150.0) == "2:30.0", fmt_clock(3700.25, 2))
    check("fmt_clock 進位正確", fmt_clock(3599.96, 1) == "1:00:00.0" and fmt_clock(59.96, 1) == "1:00.0"
          and fmt_clock(-0.04, 1) == "0:00.0" and fmt_clock(-61.5, 1) == "-1:01.5",
          f"{fmt_clock(3599.96, 1)} {fmt_clock(59.96, 1)}")
    ta = T0 + np.arange(0.0, 4000.0, 0.05)
    log = _small_log({"ATT": {"t": ta, "Roll": np.sin((ta - T0) / 3)}}, t_end=T0 + 3999.0)
    p = FieldPlot()
    p.resize(1000, 420)
    p.show()
    p.set_log(log)
    p.add_series("ATT.Roll")
    for a, b, pat in ((3700.0, 3703.0, r"^1:01:4\d\.\d$"), (3700.0, 3700.3, r"^1:01:40\.\d\d$"),
                      (3000.0, 4000.0, r"^(\d+:\d\d|1:\d\d:\d\d)$")):
        p.set_view(T0 + a, T0 + b)
        p._refresh()
        p.canvas.draw()
        labels = [lb.get_text() for lb in p._main.get_xticklabels() if lb.get_text()]
        check(f"視野 {b - a:g} s @ {a:g} s:刻度彼此不同", len(labels) >= 2 and len(set(labels)) == len(labels)
              and all(re.match(pat, s) for s in labels), str(labels[:6]))
    p.set_view(T0 + 3700.0, T0 + 3706.0)
    p._refresh()
    p.set_time(T0 + 3701.37)
    head = p._box_head.get_text()
    check("數值框時刻(放大時)有小數", head.startswith("1:01:41.37"), head)
    p.close()


def test_set_view_coalesced(check):
    """PERF-6:時間軸連續縮放 → set_view 不同步重算;合併成少數幾次重畫,view_range 立即反映。"""
    print("[set_view 合併重算(PERF-6)]")
    log = make_fake_log(att_rate=1000.0)
    p = FieldPlot()
    p.resize(1200, 500)
    p.show()
    p.set_log(log)
    for e in ("ATT.Roll", "ATT.Pitch", "ATT.DesRoll", "VIBE.VibeZ"):
        p.add_series(e)
    p.canvas.draw()
    _pump(100)
    calls = []
    orig = p._update_data
    p._update_data = lambda: (calls.append(1), orig())[1]
    views = [(T0 + 100 + k, T0 + 400 - 3 * k) for k in range(40)]
    t0 = time.perf_counter()
    for a, b in views:
        p.set_view(a, b)
    dt = (time.perf_counter() - t0) * 1000 / len(views)
    vr = p.view_range()
    check("連續 40 次 set_view:每次只改 xlim(< 5 ms)", dt < 5, f"({dt:.2f} ms/次)")
    check("view_range() 立即是最後的視野", abs(vr[0] - views[-1][0]) < 1e-6 and abs(vr[1] - views[-1][1]) < 1e-6)
    check("呼叫當下不重新抽稀", not calls)
    _pump(200)
    check("事件迴圈後只重算一次", len(calls) == 1, f"({len(calls)} 次)")
    n_pts = max(s.line.get_xdata().size for s in p._series)
    check("重算後依新視野抽稀", n_pts > 1000, f"({n_pts} 點)")

    # 手勢:8 ms 一個事件、持續約 0.5 s → 中途最多每 _REFRESH_MAX_S 更新一次
    calls.clear()
    gaps = []
    for k in range(60):
        t1 = time.perf_counter()
        p.set_view(T0 + 50 + k, T0 + 500 - 2 * k)
        _pump(8)
        gaps.append((time.perf_counter() - t1) * 1000)
    _pump(200)
    check("長手勢:中途有更新、但遠少於事件數", 2 <= len(calls) <= 8, f"({len(calls)} 次 / 60 事件)")
    check("最後的視野有套用", abs(p.view_range()[0] - (T0 + 109)) < 1e-6)

    # 底圖過期時不 blit(避免游標疊在舊視野的圖上);重畫後恢復 blit
    p.set_time(T0 + 250.0)
    _pump(200)
    blits = []
    orig_blit = p.canvas.blit
    p.canvas.blit = lambda *a, **kw: (blits.append(1), orig_blit(*a, **kw))[1]
    p.set_view(T0 + 200, T0 + 300)
    p.set_time(T0 + 260.0)
    x = float(p._cursor_line.get_xdata()[0])
    check("視野剛換:不在舊底圖上 blit,但游標位置已更新", not blits and abs(x - (260.0 + T0 - p.t_base)) < 1e-9)
    _pump(250)
    p.set_time(T0 + 261.0)
    check("重畫後恢復 blit", len(blits) == 1 and p._bg_xlim == tuple(p._main.get_xlim()))
    p.canvas.blit = orig_blit
    p.close()


def test_params_real_index(check):
    """F2:真正的 LogIndex.params() 每個名稱只有一筆 → 歷程要從 param_history 拿。"""
    from core.dflog import LogIndex
    from ui.sidebar import ParamsPanel
    print("[參數變更歷程(F2,真 LogIndex)]")
    names = ["ATC_RAT_RLL_P", "WPNAV_SPEED", "ATC_RAT_RLL_P", "ANGLE_MAX", "ATC_RAT_RLL_P",
             "WPNAV_SPEED", "ATC_RAT_RLL_P", "ATC_RAT_RLL_P", "ANGLE_MAX"]
    vals = [0.135, 500.0, 0.135, 4500.0, 0.15, 800.0, 0.15, 0.16, 4500.0]
    ts = [T0 + 0.5, T0 + 0.5, T0 + 50.0, T0 + 0.5, T0 + 200.0, T0 + 100.0, T0 + 250.0, T0 + 320.0, T0 + 400.0]
    order = np.argsort(ts, kind="stable")
    tables = {"PARM": {"t": np.array(ts)[order], "Name": np.array(names)[order],
                       "Value": np.array(vals, dtype=np.float32).astype(float)[order],
                       "Default": np.array([0.135, 1000.0, 0.135, 3000.0, 0.135, 1000.0, 0.135, 0.135,
                                            3000.0])[order]},
              "ATT": {"t": T0 + np.arange(0.0, 500.0, 1.0), "Roll": np.zeros(500)}}
    idx = LogIndex.from_tables(tables)
    check("(前提)LogIndex.params() 每個名稱一筆", len(idx.params()) == 3)
    pp = ParamsPanel()
    pp.set_index(idx, T0)
    rows = {r.name: r for r in pp.model.rows}
    r = rows.get("ATC_RAT_RLL_P")
    check("變更次數來自 param_history", r is not None and r.changes == 2 and abs(r.value - 0.16) < 1e-6,
          f"({r.changes if r else '-'} 次)")
    tip = pp.model.history_tip(r) if r else ""
    check("歷程提示列出每次變更", "0:00.5 → 0.135" in tip and "3:20.0 → 0.15" in tip
          and "5:20.0 → 0.16" in tip, tip.replace("\n", " | "))
    check("重複記錄同值不算變更", rows["ANGLE_MAX"].changes == 0 and rows["WPNAV_SPEED"].changes == 1)
    check("摘要的「記錄中變更」不再是 0", "記錄中變更 2" in pp.summary.text(), pp.summary.text())
    pp.chk_modified.setChecked(True)
    check("只看已修改含記錄中變更的參數", set(pp.visible_names()) == {"ATC_RAT_RLL_P", "WPNAV_SPEED",
                                                                    "ANGLE_MAX"}, str(pp.visible_names()))
    pp.close()


def test_t_last(check):
    """E2E-1 / F3:位置資料停了之後的 MSG / ERR 仍在欄位圖範圍內(點得到、游標走得到)。"""
    print("[log 範圍到 t_last(E2E-1)]")
    msgs = [FLogMessage(T0 + 1.0, "msg", "ArduCopter"), FLogMessage(T0 + 200.0, "err", "GPS:Glitch (2)")]
    ta = T0 + np.arange(0.0, 60.0, 0.04)
    te = T0 + np.array([150.0, 200.0])
    log = _small_log({"ATT": {"t": ta, "Roll": np.sin(ta - T0)}, "ERR": {"t": te, "ECode": np.array([2.0, 0.0])}},
                     msgs=msgs, t_end=T0 + 60.0)
    log.events = [LogEvent(T0 + 2.0, EV_MODE, "LOITER")]
    check("(前提)t_end 只到位置資料、t_last 到最後一筆訊息",
          abs(log.t_end - (T0 + 60.0)) < 0.3 and abs(log.t_last - (T0 + 200.0)) < 1e-9)
    p = FieldPlot()
    p.resize(1000, 420)
    p.show()
    seeks = []
    p.seeked.connect(seeks.append)
    p.set_log(log)
    p.add_series("ATT.Roll")
    _pump()
    check("全範圍涵蓋到 t_last", abs(p.full_range()[1] - (T0 + 200.0)) < 1e-9, f"{p.full_range()[1] - T0:.1f}")
    check("模式色帶延伸到 t_last", abs(p._mode_spans()[-1][1] - (T0 + 200.0)) < 1e-9)
    pt = _canvas_point(p, T0 + 180.0)
    QTest.mouseClick(p.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pt)
    _pump()
    check("點位置資料之後的時刻 → seeked 不被夾到 t_end", bool(seeks) and abs(seeks[-1] - (T0 + 180.0)) < 1.0,
          f"{seeks[-1] - T0:.1f}" if seeks else "無")
    p.set_time(T0 + 180.0)
    row = next(r for r in p._box_rows if r[0].expr == "ATT.Roll")
    check("游標走得到(線顯示),沒資料的序列讀 —", p._cursor_line.get_visible() and row[3].get_text() == "—")
    p.add_series("ERR.ECode * 10")
    p.set_time(T0 + 200.0)
    row = next(r for r in p._box_rows if r[0].expr == "ERR.ECode * 10")
    check("ERR.ECode * 10 在 t_last 讀得到", row[3].get_text() == "0", row[3].get_text())
    p.close()


REAL_LOG = os.path.join(os.path.expanduser("~"), "Downloads", "203 2025-3-7 上午 10-53-08.bin")


def test_real_log(check):
    """使用者的真實 log(有才跑):DSAlt 全 NaN 的預設、PARM 變更、t_last。"""
    if not os.path.isfile(REAL_LOG):
        print("[真實 log] 略過(找不到檔案)")
        return
    from core.logio import load_bin
    from ui.sidebar import _ROLE_PRESET, FieldBrowser, ParamsPanel
    print("[真實 log]")
    ld = load_bin(REAL_LOG)
    plot = FieldPlot()
    plot.resize(1000, 420)
    fb = FieldBrowser()
    fb.bind(plot)
    plot.set_log(ld)
    fb.set_index(ld.index)
    check("欄位圖範圍到 t_last(8:10.8)", abs(plot.full_range()[1] - ld.t_last) < 1e-6
          and ld.t_last - ld.t_base > 490, f"({ld.t_last - ld.t_base:.1f} s)")
    item = None
    for g in range(fb.tree_presets.topLevelItemCount()):
        gi = fb.tree_presets.topLevelItem(g)
        for k in range(gi.childCount()):
            if fb._presets[int(gi.child(k).data(0, _ROLE_PRESET))].name == "測距 / 地形高度":
                item = gi.child(k)
    if item is not None:
        fb._on_preset_clicked(item, 0)
        n = len(plot.series())
        st = item.checkState(0)
        fb._on_preset_clicked(item, 0)
        check("測距 / 地形高度:加上 → 勾,再點 → 移除", n >= 1 and st == Qt.CheckState.Checked
              and not plot.series(), f"({n} 條)")
    plot.add_series("GPS[0].GCrs")
    s = plot._series[-1]
    y = np.asarray(s.v)
    jj = np.flatnonzero(np.abs(np.diff(y)) > 180) + 1
    bad = n = 0
    for j in jj:
        a, b = float(y[j - 1]), float(y[j])
        if _ang_dist(a, b) > 179:
            continue
        v = s.value_at(0.5 * (float(s.t[j - 1]) + float(s.t[j])))
        n += 1
        bad += v is None or _ang_dist(v, a) + _ang_dist(v, b) > _ang_dist(a, b) + 1e-6
    check("GPS.GCrs 跨北方:讀值都在兩筆之間的短弧上", s.kind == "heading" and n > 50 and bad == 0,
          f"({n} 次跨越,{bad} 個錯)")
    pp = ParamsPanel()
    t0 = time.perf_counter()
    pp.set_index(ld.index, ld.t_base)
    dt = (time.perf_counter() - t0) * 1000
    n_chg = sum(1 for r in pp.model.rows if r.changes)
    check("參數面板:記錄中變更 > 0", n_chg > 0 and f"記錄中變更 {n_chg}" in pp.summary.text(),
          f"({pp.summary.text()},{dt:.0f} ms)")
    pp.close()
    fb.close()
    plot.close()


def main() -> int:
    shots = None
    if "--shots" in sys.argv:
        shots = sys.argv[sys.argv.index("--shots") + 1]
        os.makedirs(shots, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)
    if FALLBACKS:
        print(f"(注意:{', '.join(FALLBACKS)} 無法匯入,使用測試內替身)")
    check = Checker()
    test_decimate(check)
    test_fieldplot(check, app, shots)
    test_bigdata(check)
    test_legacy(check)
    test_collapsed(check)
    with tempfile.TemporaryDirectory() as tmp:
        test_field_browser(check, shots)
        test_params(check, shots, tmp)
        test_messages(check, shots)
    test_cursor_values(check)
    test_unplottable_preset(check)
    test_clock_past_hour(check)
    test_set_view_coalesced(check)
    test_params_real_index(check)
    test_t_last(check)
    test_real_log(check)
    print("全部通過" if not check.failed else f"失敗 {check.failed} 項")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
