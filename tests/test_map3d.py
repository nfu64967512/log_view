"""3D 地圖測試:build_scene(純 Python)、glTF 模型、Map3DWidget。

  python tests\\test_map3d.py                     # 場景 / 模型單元測試 + 元件煙霧測試(offscreen)
  python tests\\test_map3d.py --visual [輸出目錄]  # 真實視窗(需要網路與 GPU):等 Cesium 與圖磚
                                                   # 載入後截圖、點擊跳轉、跟隨、姿態方向、往返時間、
                                                   # 誤差連線、noorigin、地形重新全覽、假 ion token

offscreen 沒有 WebGL:元件只需要回報「不可用」而不當機;視覺檢查請用 --visual。
"""
from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

VISUAL = "--visual" in sys.argv
if not VISUAL:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "win32" and os.path.isdir(r"C:\Windows\Fonts"):
        os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from ui import map3d, models  # noqa: E402  (QtWebEngine 必須在 QApplication 之前 import)
from PyQt6.QtCore import QEventLoop, QPoint, QSettings, Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from core import frames, pipeline, sim  # noqa: E402
from core.types import KIND_GLOBAL, KIND_NED, LogData, TrajSource  # noqa: E402
from ui import theme  # noqa: E402

failed = 0


def check(label, cond, extra=""):
    global failed
    if cond:
        print(f"  [OK]   {label} {extra}")
    else:
        failed += 1
        print(f"  [FAIL] {label} {extra}")


def _pump(ms: int = 100):
    loop = QEventLoop()
    QTimer.singleShot(int(ms), loop.quit)
    loop.exec()


def _wait_until(pred, timeout: float = 30.0, step: int = 100) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            if pred():
                return True
        except Exception as e:  # noqa: BLE001
            print("    (輪詢例外)", e)
        _pump(step)
    return False


class _AttIndex:
    """最小的 LogIndex 替身:只提供 series() / vehicle(core.dflog 不在時用)。"""

    def __init__(self, d: dict, vehicle: str = "copter"):
        self.d = d
        self.vehicle = vehicle

    def series(self, key):
        return self.d.get(key)


def _index_from_tables(tables: dict, vehicle: str = "copter"):
    try:
        from core.dflog import LogIndex
        return LogIndex.from_tables(tables, vehicle=vehicle)
    except ImportError:
        d = {f"{m}.{c}": (tb["t"], tb[c]) for m, tb in tables.items() for c in tb if c != "t"}
        return _AttIndex(d, vehicle)


def _big_logdata(n: int = 400_000) -> LogData:
    """很長的 log(30 Hz × n 筆):GPS 圓周 + 偏移 0.3 m 的 VISP。"""
    tl = 1000 + np.arange(n) * (1 / 30)
    ang = tl / 60.0
    enu = np.column_stack([200 * np.cos(ang), 200 * np.sin(ang), 30 + 5 * np.sin(tl / 7)])
    lat, lng, alt = frames.enu_to_geodetic(enu, sim.ORIGIN)
    g = TrajSource("GPS", KIND_GLOBAL, tl, lat=lat, lng=lng, alt=alt)
    v = TrajSource("VISP", KIND_NED, tl, ned=frames.enu_to_ned(enu + 0.3))
    return LogData(sources={"GPS": g, "VISP": v}, origin=sim.ORIGIN)


def _enu_of(track: dict, origin) -> np.ndarray:
    lat, lng, u = (np.asarray(track[k], dtype=float) for k in ("lat", "lng", "u"))
    e = frames.geodetic_to_enu(lat, lng, np.zeros_like(lat) + origin[2], origin)
    e[:, 2] = u
    return e


# =====================================================================
# 場景(純 Python)
# =====================================================================

def test_scene():
    print("\n[build_scene:多航段模擬,無 ATE 結果]")
    ld, info = sim.make_multi_flight_sim()
    tb = ld.t_base
    sc = map3d.build_scene(ld, "GPS", "VISP", None)
    check("回傳場景", isinstance(sc, dict))
    check("原點 = ORGN", np.allclose(sc["origin"], info["origin"], atol=1e-7), f"({sc['origin']})")
    check("t_base", abs(sc["t_base"] - tb) < 1e-9)
    gps, visp = ld.sources["GPS"], ld.sources["VISP"]
    ref, est = sc["ref"], sc["est"]
    check("參考樣本數 = GPS", ref["n"] == min(gps.n, map3d.MAX_DENSE), f"({ref['n']} / {gps.n})")
    check("估測樣本數 = VISP", est["n"] == min(visp.n, map3d.MAX_DENSE), f"({est['n']} / {visp.n})")
    check("各欄長度一致", all(len(ref[k]) == ref["n"] for k in ("t", "lat", "lng", "u", "spd"))
          and all(len(est[k]) == est["n"] for k in ("t", "lat", "lng", "u")))
    check("時間為相對 t_base", abs(ref["t"][0] - (gps.t[0] - tb)) < 1e-3 and ref["t"][0] >= -1e-6)
    enu = _enu_of(ref, ld.origin)
    d = np.abs(enu - gps.enu(ld.origin)).max()
    check("經緯度往返 ENU 誤差 < 2 cm", d < 0.02, f"({d * 100:.2f} cm)")
    check("參考折線 1 段(無空洞)", len(ref["pieces"]) == 1, f"({[len(p) for p in ref['pieces']]})")
    check("估測折線在 VIO 重置處斷開", len(est["pieces"]) == 2, f"({[len(p) for p in est['pieces']]})")
    pts = sum(len(p) for p in ref["pieces"])
    check("折線頂點 <= MAX_LINE + 保留點", pts <= map3d.MAX_LINE + 64, f"({pts})")
    names = [m[1] for m in sc["modes"]]
    check("飛行模式切換", names[:4] == ["STABILIZE", "LOITER", "AUTO", "LAND"], f"({names})")
    check("模式顏色取自 theme.mode_color", all(m[2] == theme.mode_color(m[1]) for m in sc["modes"]))
    check("模式圖例不重複", [m[0] for m in sc["mode_legend"]] == ["STABILIZE", "LOITER", "AUTO", "LAND"])
    check("ARM 段", [a[2] for a in sc["arm"]] == ["ARM 1", "ARM 2"]
          and abs(sc["arm"][0][0] - (info["flights"][0][0] - tb)) < 1e-3)
    check("VIO 重置時刻", len(sc["resets"]) == 1 and abs(sc["resets"][0][0] - (info["reset_t"] - tb)) < 1e-3)
    check("沒有 ATE 結果 → 無誤差序列", sc["err"] is None and "e3" not in ref)
    check("估測註記未對齊", "未對齊" in est["note"] and not est["aligned"])
    att = sc["att"]
    check("姿態來源 = ATT(模擬附 LogIndex)" if ld.index is not None else "姿態來源 = 推估",
          sc["attitude"] == ("ATT" if ld.index is not None else "derived"), f"({sc['attitude']})")
    check("姿態序列長度一致", len(att["t"]) == len(att["roll"]) == len(att["pitch"]) == len(att["yaw"]))
    check("載具 copter → 四旋翼模型", sc["vehicle"] == "copter" and sc["model"]["kind"] == "quad")
    check("模型為 data URI", sc["model"]["uri"].startswith("data:model/gltf-binary;base64,"))
    check("顏色取自 theme", sc["colors"]["ref"] == theme.C_REF and sc["colors"]["ramp"] == theme.SEQ_RAMP)
    js = map3d.scene_json(sc)
    check("可 JSON 序列化(無 NaN)", json.loads(js)["ref"]["n"] == ref["n"], f"({len(js) / 1e6:.2f} MB)")
    check("酬載 < 2 MB", len(js) < 2e6)

    # 航向:第一段巡航往東(yaw ≈ 90°)、第二段往北(≈ 0°)
    t_att = np.asarray(att["t"])
    yaw = np.asarray(att["yaw"]) % 360.0

    def yaw_at(rel):
        return float(yaw[np.searchsorted(t_att, rel)])
    t_cruise = next(m[0] for m in sc["modes"] if m[1] == "AUTO")
    y_e, y_n = yaw_at(t_cruise + 5.0), yaw_at(t_cruise + 15.0)
    check("巡航第一段航向 ≈ 東", abs(((y_e - 90 + 180) % 360) - 180) < 15, f"({y_e:.1f}°)")
    check("巡航第二段航向 ≈ 北", abs(((y_n + 180) % 360) - 180) < 15, f"({y_n:.1f}°)")
    yw = np.asarray(att["yaw"])
    check("yaw 已 unwrap(無 > 180° 跳變)", np.abs(np.diff(yw)).max() < 180)

    print("\n[build_scene:含 ATE 結果(第 1 航段)]")
    f1 = info["flights"][0]
    res = pipeline.compute(ld, "GPS", "VISP", t0=f1[0], t1=f1[1])
    sc2 = map3d.build_scene(ld, "GPS", "VISP", res)
    err = sc2["err"]
    check("誤差序列", err is not None and len(err["t"]) == len(err["e3"]) == min(res.n, map3d.MAX_ERR))
    check("誤差範圍 = res.e3", abs(err["vmin"] - res.e3.min()) < 1e-3 and abs(err["vmax"] - res.e3.max()) < 1e-3)
    check("分析區間", abs(err["t0"] - (res.t[0] - tb)) < 1e-3 and abs(err["t1"] - (res.t[-1] - tb)) < 1e-3)
    e3 = np.array([np.nan if v is None else v for v in sc2["ref"]["e3"]])
    rt = np.asarray(sc2["ref"]["t"]) + tb
    inside = (rt >= res.t[0] - 1e-6) & (rt <= res.t[-1] + 1e-6)
    check("參考逐點誤差只在分析區間內", np.isfinite(e3[inside]).mean() > 0.95 and not np.isfinite(e3[~inside]).any(),
          f"({np.isfinite(e3).sum()} / {inside.sum()})")
    # 對齊後的估測軌跡在 res.t 上應與參考相差 res.e3(差別只來自內插)
    est2 = sc2["est"]
    ee = _enu_of(est2, ld.origin)
    et = np.asarray(est2["t"]) + tb
    est_at = np.column_stack([np.interp(res.t, et, ee[:, i]) for i in range(3)])
    dist = np.linalg.norm(est_at - res.ref_enu, axis=1)
    dev = np.abs(dist - res.e3)
    check("對齊後估測 − 參考 ≈ ATE 誤差", np.percentile(dev, 95) < 0.05,
          f"(P95 偏差 {np.percentile(dev, 95) * 100:.1f} cm,中位誤差 {np.median(res.e3):.2f} m)")
    check("估測註記已對齊", est2["aligned"] and "SE(3)" in est2["note"], f"({est2['note']})")
    sc3 = map3d.build_scene(ld, "VISP", "GPS", res)
    check("結果屬於別組來源 → 不套用", sc3 is not None and sc3["err"] is None and "未對齊" in sc3["est"]["note"])
    check("估測 = 參考 → 只有參考", map3d.build_scene(ld, "GPS", "GPS", None)["est"] is None)
    check("不存在的參考 → None", map3d.build_scene(ld, "NOPE", "GPS", None) is None)


def test_attitude_and_origin():
    print("\n[姿態來源 / 原點]")
    ld, _ = sim.make_multi_flight_sim()
    tb = ld.t_base
    t = tb + np.arange(0.0, 60.0, 0.1)
    yaw = (350.0 + 2.0 * (t - tb)) % 360.0                 # 跨 360° → 0°
    att = {"t": t, "Roll": 10 * np.sin(t), "Pitch": -5 + 0 * t, "Yaw": yaw}
    ld.index = _index_from_tables({"ATT": att, "AHR2": dict(att, Roll=0 * t)}, vehicle="plane")
    sc = map3d.build_scene(ld, "GPS", "VISP")
    a = sc["att"]
    check("ATT 優先", sc["attitude"] == "ATT", f"({sc['attitude']})")
    yw = np.asarray(a["yaw"])
    check("yaw 跨 360° 已 unwrap", np.abs(np.diff(yw)).max() < 1.0 and yw[-1] > 360, f"({yw[0]:.0f}→{yw[-1]:.0f})")
    check("roll / pitch 照抄", abs(a["pitch"][0] + 5) < 0.05 and abs(a["roll"][10] - 10 * math.sin(t[10])) < 0.06)
    check("plane → 定翼機模型", sc["vehicle"] == "plane" and sc["model"]["kind"] == "plane")

    ld.index = _index_from_tables({"AHR2": att}, vehicle="rover")
    sc = map3d.build_scene(ld, "GPS", "VISP")
    check("沒有 ATT → 用 AHR2", sc["attitude"] == "AHR2")
    check("rover → 車輛模型", sc["model"]["kind"] == "rover")

    class Broken:
        vehicle = "copter"

        def series(self, key):
            raise RuntimeError("壞掉的索引")
    ld.index = Broken()
    sc = map3d.build_scene(ld, "GPS", "VISP")
    check("索引出錯 → 由航跡推估", sc["attitude"] == "derived" and len(sc["att"]["t"]) == sc["ref"]["n"])
    ld.index = None
    sc = map3d.build_scene(ld, "GPS", "VISP")
    check("沒有索引 → 推估 + copter", sc["attitude"] == "derived" and sc["vehicle"] == "copter")
    check("推估姿態 roll / pitch = 0", not np.any(sc["att"]["roll"]) and not np.any(sc["att"]["pitch"]))

    # 只有局部 NED 來源、log 沒有原點 → 無法放上地圖
    rng = np.random.default_rng(1)
    tt = 100 + np.arange(0, 30, 0.1)
    ned = np.column_stack([np.sin(tt), np.cos(tt), -5 + 0 * tt]) + rng.normal(0, 0.01, (tt.size, 3))
    v1 = TrajSource("VISP", KIND_NED, tt, ned=ned)
    v2 = TrajSource("XKF1", KIND_NED, tt, ned=ned + 0.1)
    ned_only = LogData(sources={"VISP": v1, "XKF1": v2}, origin=None)
    check("只有 NED 且無原點 → None", map3d.build_scene(ned_only, "VISP", "XKF1") is None)
    ned_only.origin = sim.ORIGIN
    sc = map3d.build_scene(ned_only, "VISP", "XKF1")
    check("只有 NED 但有 ORGN → 以 ORGN 放置", sc is not None and np.allclose(sc["origin"], sim.ORIGIN))
    check("只有 NED → 高度 = -D", sc is not None and abs(sc["ref"]["u"][0] - 5.0) < 0.05)

    # 很長的 log:抽稀後的酬載要有上限
    big = _big_logdata()
    t0 = time.perf_counter()
    sc = map3d.build_scene(big, "GPS", "VISP")
    js = map3d.scene_json(sc)
    dt = time.perf_counter() - t0
    pts = sum(len(p) for p in sc["ref"]["pieces"])
    check("長 log 抽稀:內插樣本 <= MAX_DENSE", sc["ref"]["n"] <= map3d.MAX_DENSE and sc["est"]["n"] <= map3d.MAX_DENSE)
    check("長 log 抽稀:折線頂點 <= MAX_LINE", pts <= map3d.MAX_LINE + 2, f"({pts})")
    check("長 log 酬載 < 10 MB", len(js) < 10e6, f"({len(js) / 1e6:.1f} MB,{dt:.2f} s)")


def test_models():
    print("\n[glTF 模型]")
    for kind in models.MODEL_KINDS:
        data = models.model_glb(kind)
        gltf, binc = models.parse_glb(data)
        prims = gltf["meshes"][0]["primitives"]
        ok = len(binc) == gltf["buffers"][0]["byteLength"] and len(data) % 4 == 0
        for p in prims:
            acc = gltf["accessors"]
            pos, nrm, idx = acc[p["attributes"]["POSITION"]], acc[p["attributes"]["NORMAL"]], acc[p["indices"]]
            ok &= pos["count"] == nrm["count"] and idx["count"] % 3 == 0 and "min" in pos and "max" in pos
            view = gltf["bufferViews"][idx["bufferView"]]
            dt = np.uint16 if idx["componentType"] == 5123 else np.uint32
            ii = np.frombuffer(binc, dt, idx["count"], view["byteOffset"])
            ok &= int(ii.max()) < pos["count"]
        lo = np.min([gltf["accessors"][p["attributes"]["POSITION"]]["min"] for p in prims], axis=0)
        hi = np.max([gltf["accessors"][p["attributes"]["POSITION"]]["max"] for p in prims], axis=0)
        # glTF 軸:+X 左、+Y 上、+Z 前;機頭(紅色)在 +Z
        red = [p for p in prims if gltf["materials"][p["material"]]["pbrMetallicRoughness"]["baseColorFactor"][:3]
               == [0.91, 0.24, 0.2]]
        red_front = all(gltf["accessors"][p["attributes"]["POSITION"]]["max"][2] > 0.2 for p in red)
        check(f"{kind}:結構正確", ok, f"({len(data) / 1024:.0f} KB,{len(prims)} 個材質,"
              f"範圍 {np.round(lo, 2).tolist()} ~ {np.round(hi, 2).tolist()})")
        check(f"{kind}:紅色記號在前方(glTF +Z)", bool(red) and red_front)
    check("model_kind 對應", [models.model_kind(v) for v in ("copter", "heli", "plane", "rover", None)]
          == ["quad", "quad", "plane", "rover", "quad"])


# =====================================================================
# 元件
# =====================================================================

class _StubPage:
    """替身頁面:記錄 runJavaScript 呼叫,callback 由測試手動觸發。"""

    def __init__(self):
        self.calls: list[str] = []
        self.cbs: list = []

    def runJavaScript(self, code, *args):
        self.calls.append(code)
        cb = args[-1] if args and callable(args[-1]) else None
        if cb is not None:
            self.cbs.append(cb)


def test_widget():
    print("\n[Map3DWidget 煙霧測試(offscreen)]")
    ld, info = sim.make_multi_flight_sim()
    tb = ld.t_base
    sc = map3d.build_scene(ld, "GPS", "VISP")
    w = map3d.Map3DWidget()
    w.resize(900, 600)
    w.show()
    seeks: list[float] = []
    w.seeked.connect(seeks.append)
    ok = _wait_until(lambda: w.ready, 40)
    check("網頁回報 ready", ok, f"(available={w.available}, reason={w.reason!r})")
    check("offscreen 沒有 WebGL → 不可用並說明原因", w.available or w.reason in ("nowebgl", "offline", "error"))
    try:
        w.set_scene(sc)
        for k in range(50):
            w.set_time(tb + k * 0.5)
        w.set_time(None)
        w.view_all()
        w.cmb_base.setCurrentIndex(1)
        w.cmb_color.setCurrentIndex(2)
        w.chk_est.setChecked(False)
        w.chk_follow.setChecked(True)
        w.set_scene(None)
        w.set_time(tb + 3.0)
        w.set_scene(sc)
        _pump(300)
        raised = None
    except Exception as e:  # noqa: BLE001
        raised = e
    check("不可用時呼叫各介面不丟例外", raised is None, f"({raised!r})" if raised else "")
    check("沒有 ATE 結果 →「ATE 誤差」著色停用",
          not w.cmb_color.model().item(w.cmb_color.findData("error")).isEnabled())
    if not w.available:
        check("網頁有說明訊息", any("WebGL" in m[1] or "Cesium" in m[1] for m in w.console_messages)
              or w.reason in ("nowebgl", "offline"), f"({w.console_messages[-2:]})")

    print("\n[set_time 合併 / 只在可見時送出(替身頁面)]")
    real_page, stub = w.page, _StubPage()
    w.page = stub
    w._ready = w._available = True
    w._scene_dirty = False
    w._pending_since = None
    w.hide()
    w.set_time(tb + 1.0)
    check("隱藏時不送 setTime", not stub.calls)
    w.show()
    _pump(50)
    check("顯示時補送目前時刻", len(stub.calls) == 1 and stub.calls[0] == f"app.setTime({tb + 1.0!r})",
          f"({stub.calls})")
    for k in range(20):
        w.set_time(tb + 2.0 + k)
    check("前一個未回來時不重複送", len(stub.calls) == 1)
    stub.cbs.pop(0)(1)
    check("回來後只送最新時刻", len(stub.calls) == 2 and stub.calls[1] == f"app.setTime({tb + 21.0!r})",
          f"({stub.calls[-1:]})")
    stub.cbs.pop(0)(1)
    check("沒有新時刻就不再送", len(stub.calls) == 2)
    check("記錄往返時間", len(w.timings) >= 2)
    w.set_scene(sc)
    _pump(20)                            # 場景合併到下一輪事件迴圈才送
    check("換場景先送 setScene 再送時間", len(stub.calls) == 4 and stub.calls[2].startswith("app.setScene(")
          and stub.calls[2].endswith(',"")') and stub.calls[3] == f"app.setTime({tb + 21.0!r})",
          f"({[c[:20] for c in stub.calls[2:]]})")
    stub.cbs.clear()
    w._pending_since = None
    w._bridge.seek(tb + 42.0)
    check("JS seek → seeked 訊號", seeks and abs(seeks[-1] - (tb + 42.0)) < 1e-9)
    w._bridge.seek(float("nan"))
    check("NaN seek 被忽略", len(seeks) == 1)
    w.page = real_page
    w._on_load_started()                 # 回到真實狀態

    print("\n[沒有 PyQt6-WebEngine 時的佔位]")
    saved = map3d.HAVE_WEBENGINE
    map3d.HAVE_WEBENGINE = False
    try:
        p = map3d.Map3DWidget()
        labels = [lab.text() for lab in p.findChildren(QLabel)]
        check("顯示「需要 PyQt6-WebEngine」", any("PyQt6-WebEngine" in t for t in labels))
        p.set_scene(sc)
        p.set_time(tb + 5)
        p.view_all()
        check("佔位元件 available = False", not p.available and p.reason == "nowebengine")
        p.deleteLater()
    finally:
        map3d.HAVE_WEBENGINE = saved

    w.close()
    w.deleteLater()
    _pump(100)


def _scene_calls(stub) -> list[str]:
    return [c for c in stub.calls if c.startswith("app.setScene(")]


def test_scene_deferred():
    """PERF-1(3)/ MAP-NOORIGIN-MSG:場景延到可見 + ready 才序列化與傳送、只留最新一份;reason 傳給網頁。"""
    print("\n[set_scene 延後序列化 / 合併 / noorigin(替身頁面)]")
    ld, info = sim.make_multi_flight_sim()
    tb = ld.t_base
    f1 = info["flights"][0]
    res = pipeline.compute(ld, "GPS", "VISP", t0=f1[0], t1=f1[1])
    sc_a = map3d.build_scene(ld, "GPS", "VISP")
    sc_b = map3d.build_scene(ld, "GPS", "VISP", res)
    w = map3d.Map3DWidget()
    w.resize(900, 600)
    w.show()
    _wait_until(lambda: w.ready, 40)
    real_page, stub = w.page, _StubPage()
    w.page = stub
    w._ready = w._available = True
    w._scene_dirty = False
    w._pending_since = None
    ser: list = []
    orig = map3d.scene_json

    def counting(s):
        ser.append(s)
        return orig(s)
    map3d.scene_json = counting
    try:
        w.hide()
        w.set_time(tb + 5.0)
        for s in (sc_a, sc_b, sc_a, sc_b):
            w.set_scene(s)
        _pump(30)
        check("隱藏時不序列化、不傳送", not ser and not stub.calls, f"({len(ser)} 次序列化,{len(stub.calls)} 個呼叫)")
        check("has_scene 立即反映", w.has_scene)
        w.show()
        _pump(50)
        scenes = _scene_calls(stub)
        check("顯示後只序列化最後一份", len(ser) == 1 and ser[0] is sc_b, f"({len(ser)})")
        check("只送一次 setScene(最後一份)", len(scenes) == 1 and scenes[0] == f'app.setScene({orig(sc_b)},"")')
        check("場景之後補送目前時刻", stub.calls[-1] == f"app.setTime({tb + 5.0!r})", f"({stub.calls[-1][:30]})")
        check("送出後不再保留場景 dict(只留 JSON)", w._scene is None and w._scene_js is not None)

        # 可見時,同一輪事件迴圈內連續換場景(載入 log:清空 → 算完 ATE)只送最後一份
        stub.calls.clear()
        stub.cbs.clear()
        w._pending_since = None
        ser.clear()
        w.set_scene(None)
        w.set_scene(sc_a)
        check("同一輪內尚未送出", not stub.calls)
        _pump(30)
        scenes = _scene_calls(stub)
        check("可見時連續換場景只送最後一份", len(scenes) == 1 and len(ser) == 1 and ser[0] is sc_a,
              f"({len(scenes)} 次 setScene)")

        # 場景待送時播放頭先等:順序一定是場景 → 時間
        stub.calls.clear()
        stub.cbs.clear()
        w._pending_since = None
        w.set_scene(sc_b)
        w.set_time(tb + 7.0)
        check("場景待送時 setTime 先等", not stub.calls)
        _pump(30)
        check("先送場景再送時間", [c.split("(")[0] for c in stub.calls] == ["app.setScene", "app.setTime"]
              and stub.calls[-1] == f"app.setTime({tb + 7.0!r})", f"({[c[:16] for c in stub.calls]})")

        # 網頁重載:用留著的 JSON 重送,不必重新序列化
        stub.calls.clear()
        stub.cbs.clear()
        w._pending_since = None
        ser.clear()
        w._on_load_started()
        w._on_ready(True, "")
        _pump(30)
        scenes = _scene_calls(stub)
        check("網頁重載後重送場景(不重新序列化)", len(scenes) == 1 and not ser
              and scenes[0].startswith('app.setScene({"version"'))

        # 沒有地理原點:reason 傳給網頁(頁面顯示說明而不是「載入 log」)
        stub.calls.clear()
        w.set_scene(None, reason="noorigin")
        check("noorigin → has_scene = False", not w.has_scene)
        _pump(30)
        check("noorigin 原因傳給網頁", _scene_calls(stub) == ['app.setScene(null,"noorigin")'], f"({_scene_calls(stub)})")
        stub.calls.clear()
        w.set_scene(None)
        _pump(30)
        check("一般清空不帶原因", _scene_calls(stub) == ['app.setScene(null,"")'])
        stub.calls.clear()
        w.set_scene(sc_a, reason="noorigin")
        _pump(30)
        check("有場景時忽略 reason", len(_scene_calls(stub)) == 1 and _scene_calls(stub)[0].endswith(',"")')
              and w.has_scene)

        # 網頁還沒 ready:不序列化、不送
        stub.calls.clear()
        ser.clear()
        w._ready = False
        w.set_scene(sc_b)
        _pump(30)
        check("網頁未 ready → 不序列化", not ser and not stub.calls)
        w._ready = True
    finally:
        map3d.scene_json = orig

    # 長 log 的大場景:隱藏時 set_scene 不再先序列化(以前每次重算都要付這筆)
    big = map3d.build_scene(_big_logdata(), "GPS", "VISP")
    w.hide()
    t0 = time.perf_counter()
    w.set_scene(big)
    dt_set = time.perf_counter() - t0
    t0 = time.perf_counter()
    js = map3d.scene_json(big)
    dt_json = time.perf_counter() - t0
    check("隱藏時 set_scene(大場景)幾乎不花時間", dt_set < max(0.005, 0.1 * dt_json),
          f"({dt_set * 1000:.2f} ms;序列化要 {dt_json * 1000:.0f} ms,{len(js) / 1e6:.1f} MB)")
    w.page = real_page
    w._on_load_started()
    w.close()
    w.deleteLater()
    _pump(100)


def test_color_pref():
    """MAP-COLOR-RESET:暫時沒有 ATE 結果(打字中途、載入新 log)不會把「ATE 誤差」著色永久改掉。"""
    print("\n[著色選擇:暫時沒有 ATE 結果時記住使用者的選擇]")
    ld, info = sim.make_multi_flight_sim()
    f1 = info["flights"][0]
    res = pipeline.compute(ld, "GPS", "VISP", t0=f1[0], t1=f1[1])
    sc0 = map3d.build_scene(ld, "GPS", "VISP")
    sc1 = map3d.build_scene(ld, "GPS", "VISP", res)
    w = map3d.Map3DWidget()
    real_page, stub = w.page, _StubPage()
    w.page = stub
    w._ready = w._available = True
    cmb = w.cmb_color

    def last_color():
        opts = [c for c in stub.calls if c.startswith("app.setOptions(")]
        return json.loads(opts[-1][len("app.setOptions("):-1])["color"] if opts else None

    def err_enabled():
        return cmb.model().item(cmb.findData("error")).isEnabled()

    w.set_scene(sc1)
    cmb.setCurrentIndex(cmb.findData("error"))
    check("選 ATE 誤差", cmb.currentData() == "error" and last_color() == "error")
    w.set_scene(sc0)                     # 例如分析終點打到一半(t1 < t0)→ 結果失效
    check("暫時沒有結果 → 顯示飛行模式、誤差項停用", cmb.currentData() == "mode" and not err_enabled()
          and last_color() == "mode")
    w.set_scene(sc1)
    check("結果回來 → 換回 ATE 誤差", cmb.currentData() == "error" and err_enabled() and last_color() == "error")
    for s in (None, sc0, sc1):           # 載入新 log:清空 → 未計算 → 算完
        w.set_scene(s)
    check("載入新 log 重算後仍是 ATE 誤差", cmb.currentData() == "error")
    w.set_scene(sc0)
    cmb.setCurrentIndex(cmb.findData("role"))
    w.set_scene(sc1)
    check("沒結果時改選別的 → 以新選擇為準", cmb.currentData() == "role")
    cmb.setCurrentIndex(cmb.findData("error"))
    w.set_scene(sc0)
    cmb.activated.emit(cmb.currentIndex())   # 使用者在選單重選目前顯示的「飛行模式」
    w.set_scene(sc1)
    check("重選目前顯示的飛行模式也算數", cmb.currentData() == "mode")
    w.page = real_page
    w._on_load_started()
    w.deleteLater()
    _pump(50)


def test_status_and_grab():
    """MAP-STATUS-CLIP / E2E-8 / MAP-ION-TOKEN(Python 端):狀態列截句尾、grab_map 不含選項列、
    ion 影像失敗退回 Esri 但不寫進設定。"""
    print("\n[狀態列 / grab_map / ion 影像失敗退回]")
    w = map3d.Map3DWidget()
    w.resize(900, 600)
    w.show()
    _wait_until(lambda: w.ready, 40)
    _pump(100)
    lbl = w.lbl_status
    msg = ("Cesium ion 影像載入失敗(token 無效或無網路):Request has failed. Status Code: 401 — "
           + "後面還有很長的說明文字" * 12)
    w._on_js_error(msg)
    _pump(50)
    txt = lbl.text()
    check("狀態靠左", bool(lbl.alignment() & Qt.AlignmentFlag.AlignLeft))
    check("放不下時保留開頭、句尾截成 …", txt.startswith("Cesium ion 影像載入失敗") and txt.endswith("…")
          and len(txt) < len(msg), f"({txt!r})")
    check("完整文字在 tooltip / status_text / errors", lbl.toolTip() == msg and w.status_text == msg
          and w.errors[-1] == msg)
    n0 = len(txt)
    w.resize(1700, 600)
    _pump(100)
    check("變寬 → 顯示更多字", len(lbl.text()) > n0, f"({n0} → {len(lbl.text())} 字)")
    w._set_status("短訊息")
    check("短訊息完整顯示", lbl.text() == "短訊息")

    pm, full = w.grab_map(), w.grab()
    dpr = pm.devicePixelRatio() or 1.0
    check("grab_map 只截地圖網頁(不含選項列)", w.view is not None
          and abs(pm.height() / dpr - w.view.height()) <= 1 and pm.height() < full.height(),
          f"({pm.width()}×{pm.height()};整個元件 {full.width()}×{full.height()})")

    s = QSettings(*map3d.SETTINGS)
    s.setValue(map3d.KEY_TOKEN, "test-token")
    w._update_ion_item()
    w.cmb_base.setCurrentIndex(w.cmb_base.findData("ion"))
    check("選 ion 底圖並記住", w.cmb_base.currentData() == "ion" and s.value(map3d.KEY_BASE) == "ion")
    w._on_js_error("Cesium ion 影像載入失敗(token 無效或無網路):Request has failed. Status Code: 401")
    check("ion 影像失敗 → 選單退回 Esri", w.cmb_base.currentData() == "esri")
    check("自動退回不寫進設定(下次啟動仍試 ion)", QSettings(*map3d.SETTINGS).value(map3d.KEY_BASE) == "ion")
    s.remove(map3d.KEY_TOKEN)
    s.remove(map3d.KEY_BASE)
    w.close()
    w.deleteLater()
    _pump(100)

    saved = map3d.HAVE_WEBENGINE
    map3d.HAVE_WEBENGINE = False
    try:
        p = map3d.Map3DWidget()
        p.resize(400, 300)
        check("沒有網頁元件 → grab_map 退回整個元件", p.grab_map().size() == p.grab().size())
        p.set_scene({"err": None}, reason="")
        check("佔位元件也記錄 has_scene", p.has_scene)
        p.deleteLater()
    finally:
        map3d.HAVE_WEBENGINE = saved


_HOLD_LOCK_SRC = (
    "import sys\n"
    "from PyQt6.QtCore import QLockFile\n"
    "lk = QLockFile(sys.argv[1]); lk.setStaleLockTime(0)\n"
    "print('locked' if lk.tryLock(0) else 'busy', flush=True)\n"
    "sys.stdin.readline()\n")


def test_profile_lock():
    """MAP-PROFILE-LOCK:第二個執行中的程式改用 webengine-2(Chromium 磁碟快取不能共用)。"""
    print("\n[WebEngine profile 目錄鎖(同時開兩個視窗)]")
    tmp = tempfile.mkdtemp(prefix="vio_ate_lock_")
    base = os.path.join(tmp, "vio_ate_studio", "webengine")
    os.makedirs(os.path.dirname(base))
    holder = subprocess.Popen([sys.executable, "-c", _HOLD_LOCK_SRC, base + ".lock"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        state = holder.stdout.readline().strip()
        check("另一個程序佔住預設目錄", state == "locked", f"({state})")
        root, lock = map3d.lock_profile_root(base)
        check("被佔用 → 改用 webengine-2", root == base + "-2" and lock is not None and lock.isLocked(),
              f"({root})")
        r2, l2 = map3d.lock_profile_root(base, slots=2)
        check("全部被佔用 → (None, None)", r2 is None and l2 is None)
        lock.unlock()
        holder.kill()                    # 模擬當掉:鎖檔留著,但 PID 已不在
        holder.wait(10)
        left = os.path.exists(base + ".lock")
        r3, l3 = map3d.lock_profile_root(base)
        check("回收當掉程序留下的鎖", r3 == base and l3 is not None, f"({r3};鎖檔殘留={left})")
        if l3 is not None:
            l3.unlock()
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(10)
        shutil.rmtree(tmp, ignore_errors=True)

    # 一個目錄都拿不到 → off-the-record profile + 狀態列說明
    keys = ("_PROFILE", "_PROFILE_LOCK", "_PROFILE_ROOT", "_PROFILE_NOTE", "lock_profile_root")
    saved = {k: getattr(map3d, k) for k in keys}
    map3d._PROFILE = map3d._PROFILE_LOCK = map3d._PROFILE_ROOT = None
    map3d.lock_profile_root = lambda base_root, slots=map3d.PROFILE_SLOTS: (None, None)
    try:
        p = map3d.Map3DWidget()
        check("全被佔用 → off-the-record profile", p.page.profile().isOffTheRecord())
        check("狀態列說明快取被佔用", p.status_text == map3d.NOTE_SHARED_CACHE, f"({p.status_text!r})")
        p._set_status("其他訊息")
        other = p.status_text
        p._set_status("")
        check("訊息清掉後回到常駐說明", other == "其他訊息" and p.status_text == map3d.NOTE_SHARED_CACHE)
        p.deleteLater()
        _pump(50)
    finally:
        for k, v in saved.items():
            setattr(map3d, k, v)


def test_js_source():
    """MAP-ION-TOKEN(靜態):ion 影像與地形都明確帶使用者的 token,不會退回 Cesium 內建示範 token。
    實際請求在 --visual 用假 token 驗證。"""
    print("\n[map3d.js 靜態檢查]")
    src = open(os.path.join(map3d.WEB_DIR, "map3d.js"), encoding="utf-8").read()
    check("不用 fromWorldImagery / createWorldTerrainAsync(只吃 Ion.defaultAccessToken)",
          "fromWorldImagery(" not in src and "createWorldTerrainAsync(" not in src)
    check("影像與地形都帶 accessToken", src.count("{ accessToken: token }") >= 2)
    check("清掉 Cesium 內建示範 token", 'Ion.defaultAccessToken = ""' in src)
    check("誤差連線只在分析區間內", 'errAt(rel).state === "ok"' in src and "&& inWin" in src)
    check("地面高到了再全覽(fitKey)", "S.fitKey === key" in src and "S.fitKey = null" in src)


# =====================================================================
# 真實視窗視覺檢查(--visual)
# =====================================================================

def _js(w, code: str, timeout: float = 10.0):
    box = {}
    if not w.run_js(code, lambda r: box.setdefault("r", r)):
        return None
    t0 = time.perf_counter()
    while "r" not in box and time.perf_counter() - t0 < timeout:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
    return box.get("r")


def _status(w) -> dict:
    r = _js(w, "JSON.stringify(app.status())")
    return json.loads(r) if r else {}


def _settle(w, extra: int = 500, timeout: float = 40.0) -> bool:
    ok = _wait_until(lambda: _status(w).get("tilesLoaded"), timeout, 150)
    _pump(extra)
    return ok


def _shot(w, out_dir: str, name: str, full: bool = False) -> str:
    """截圖:預設只截地圖(grab_map);full=True 連選項列 / 狀態列一起截。"""
    p = os.path.join(out_dir, f"map3d_{name}.png")
    (w.grab() if full else w.grab_map()).save(p)
    print(f"    截圖 {p}")
    return p


def _fake_ion_token() -> str:
    """格式正確(JWT:header.payload.signature)但無效的 Cesium ion token(測試用,隨機產生)。"""
    def enc(d):
        return base64.urlsafe_b64encode(json.dumps(d, separators=(",", ":")).encode()).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
    return (f"{enc({'alg': 'HS256', 'typ': 'JWT'})}."
            f"{enc({'jti': '00000000-0000-4000-8000-000000000000', 'id': 1, 'iat': 1700000000})}.{sig}")


def _shifted_scene(sc: dict, dlat: float) -> dict:
    """把場景整個往北挪 dlat 度(模擬另一份在別處飛的 log)。"""
    s = json.loads(map3d.scene_json(sc))
    s["origin"][0] += dlat
    for k in ("ref", "est"):
        if s.get(k):
            s[k]["lat"] = [None if v is None else v + dlat for v in s[k]["lat"]]
    return s


_CAM_JS = """JSON.stringify((() => {
  const C = Cesium, cam = S.viewer.camera, cv = S.viewer.scene.canvas, o = S.scene.origin;
  const h = C.Cartographic.fromCartesian(cam.positionWC).height;
  const p = C.SceneTransforms.worldToWindowCoordinates(S.viewer.scene, C.Cartesian3.fromDegrees(o[1], o[0], S.ground + 5));
  return { h, ground: S.ground, fitKey: S.fitKey, px: p ? [p.x, p.y] : null, w: cv.clientWidth, hgt: cv.clientHeight };
})())"""

# 假的 ion 地形:整片高 H m;原點地面高的取樣延遲 MS 毫秒才回來(模擬 sampleTerrainMostDetailed)
_FAKE_TERRAIN_JS = """(() => {
  window.__gd = 0;
  groundAt = async () => { await new Promise((r) => setTimeout(r, %(ms)d)); window.__gd++; return %(h)f; };
  S.viewer.terrainProvider = new Cesium.CustomHeightmapTerrainProvider({ width: 32, height: 32,
      callback: () => new Float32Array(32 * 32).fill(%(h)f) });
  S.opts.token = "fake"; S.applied.token = "fake"; S.ion.ter = "ok";
  return 1;
})()"""


def _cam(w) -> dict:
    r = _js(w, _CAM_JS)
    return json.loads(r) if r else {}


def _in_view(c: dict) -> bool:
    px = c.get("px")
    return bool(px) and 0 <= px[0] <= c["w"] and 0 <= px[1] <= c["hgt"]


_MODEL_AXES_JS = """(() => {
  const C = Cesium, m = S.model.modelMatrix;
  const pos = C.Matrix4.getTranslation(m, new C.Cartesian3());
  const enu = C.Transforms.eastNorthUpToFixedFrame(pos);
  const inv = C.Matrix4.inverseTransformation(enu, new C.Matrix4());
  const ax = (x, y, z) => C.Matrix4.multiplyByPointAsVector(inv,
      C.Matrix4.multiplyByPointAsVector(m, new C.Cartesian3(x, y, z), new C.Cartesian3()), new C.Cartesian3());
  const f = ax(1, 0, 0), l = ax(0, 1, 0);
  return JSON.stringify({fwd: [f.x, f.y, f.z], left: [l.x, l.y, l.z]});
})()"""


def visual(out_dir: str):
    print(f"\n[真實視窗視覺檢查] 輸出:{out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    ld, info = sim.make_multi_flight_sim()
    tb = ld.t_base
    f1 = info["flights"][0]
    res = pipeline.compute(ld, "GPS", "VISP", t0=f1[0], t1=f1[1])
    sc = map3d.build_scene(ld, "GPS", "VISP", res)
    w = map3d.Map3DWidget()
    w.setWindowTitle("Map3DWidget 視覺檢查")
    w.resize(1280, 800)
    w.show()
    seeks: list[float] = []
    w.seeked.connect(seeks.append)
    t0 = time.perf_counter()
    check("Cesium 載入完成", _wait_until(lambda: w.ready, 60) and w.available,
          f"({time.perf_counter() - t0:.1f} s,reason={w.reason!r})")
    if not w.available:
        print("   ", w.console_messages[-5:])
        w.close()
        return
    w.set_scene(sc)
    t_view = tb + 36.0
    w.set_time(t_view)
    check("模型載入", _wait_until(lambda: _status(w).get("modelReady"), 30))
    check("底圖圖磚載入", _settle(w, 800))
    st = _status(w)
    check("場景已顯示", st.get("hasScene") and st.get("linePts", 0) > 100, f"({st})")
    _shot(w, out_dir, "01_overview_mode")

    # MAP-ERRLINK-OUTSIDE:分析區間(ARM 1)外不畫估測點與白色誤差連線
    f2 = info["flights"][1]
    link_js = "JSON.stringify([S.dyn.est.show, S.dyn.link.show])"
    w.set_time(f1[0] + 25.0)
    _pump(200)
    inside = json.loads(_js(w, link_js) or "[false, false]")
    w.set_time(f2[0] + 25.0)
    _pump(200)
    outside = json.loads(_js(w, link_js) or "[true, true]")
    hud = _js(w, "document.getElementById('hud').innerText") or ""
    check("分析區間內:估測點 + 誤差連線", inside == [True, True], f"({inside})")
    check("分析區間外:不畫估測點與誤差連線(HUD 區間外)", outside == [False, False] and "區間外" in hud,
          f"({outside})")
    # MAP-HUD-ATT:HUD 內的小姿態儀(有 ATT 才畫;畫布不是空白)
    w.set_time(f1[0] + 25.0)
    _pump(200)
    att_js = ("(()=>{const c=document.getElementById('hud-att');"
              "const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;"
              "let n=0;for(let i=3;i<d.length;i+=4)if(d[i]>0)n++;"
              "return JSON.stringify({vis:!c.classList.contains('hidden'),px:n,tot:d.length/4})})()")
    cv = json.loads(_js(w, att_js) or "{}")
    check("HUD 姿態儀:有 ATT 時顯示且有畫內容", cv.get("vis") and cv.get("px", 0) > 0.5 * cv.get("tot", 1),
          f"({cv})")
    _settle(w, 300)
    _shot(w, out_dir, "01b_outside_window")

    # E2E-8:grab_map 只截地圖、而且截得到 WebGL 內容
    pm = w.grab_map()
    img = pm.toImage()
    cols = {img.pixel(x, y) for x in range(0, img.width(), 37) for y in range(0, img.height(), 29)}
    check("grab_map:只含地圖(不含選項列)且有 WebGL 內容",
          abs(pm.height() / (pm.devicePixelRatio() or 1) - w.view.height()) <= 1 and len(cols) > 50,
          f"({pm.width()}×{pm.height()},取樣 {len(cols)} 種顏色)")
    pm.save(os.path.join(out_dir, "map3d_01c_grab_map.png"))

    # 點擊軌跡跳轉:取參考軌跡在 1:00 的點,投影到螢幕後用滑鼠點
    rel = 60.0
    ref = sc["ref"]
    i = int(np.searchsorted(ref["t"], rel))
    w.set_time(tb + rel)
    _pump(200)
    xy = _js(w, f"JSON.stringify(app.project({ref['lat'][i]}, {ref['lng'][i]}, {ref['u'][i]}))")
    xy = json.loads(xy) if xy else None
    seeks.clear()
    if xy:
        from PyQt6.QtTest import QTest
        target = w.view.focusProxy() or w.view
        QTest.mouseClick(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(int(round(xy[0])), int(round(xy[1]))))
        _wait_until(lambda: bool(seeks), 3)
    via = "滑鼠"
    if not seeks and xy:
        via = "app.click"
        _js(w, f"app.click({xy[0]}, {xy[1]})")
        _wait_until(lambda: bool(seeks), 3)
    check(f"點擊軌跡 → seeked({via})", bool(seeks) and abs(seeks[-1] - (tb + ref["t"][i])) < 1.0,
          f"(點 {ref['t'][i]:.2f} s → {seeks[-1] - tb:.2f} s)" if seeks else f"(xy={xy})")

    # 跟隨 + 姿態方向:機頭(模型 +X)應指向航跡方向,轉彎時往內側傾
    w.chk_follow.setChecked(True)
    t_auto = next(m[0] for m in sc["modes"] if m[1] == "AUTO")
    legs = [("巡航往東", t_auto + 5.0, 90.0), ("巡航往北", t_auto + 15.0, 0.0),
            ("巡航往西", t_auto + 25.0, 270.0)]
    for name, rel_t, expect in legs:
        w.set_time(tb + rel_t)
        _pump(150)
        ax = _js(w, _MODEL_AXES_JS)
        fwd = json.loads(ax)["fwd"] if ax else [0, 0, 0]
        hdg = math.degrees(math.atan2(fwd[0], fwd[1])) % 360
        check(f"{name}:機頭方位 ≈ {expect:.0f}°", abs(((hdg - expect + 180) % 360) - 180) < 12, f"({hdg:.1f}°)")
    check("跟隨中", _status(w).get("following") is True)
    w.set_time(tb + t_auto + 5.0)
    _settle(w, 400)
    _shot(w, out_dir, "02_follow")

    # 轉彎(第一個角:往東 → 往北,左轉)— 從後上方看
    att_t = np.asarray(sc["att"]["t"])
    roll = np.asarray(sc["att"]["roll"])
    win = (att_t > t_auto + 8) & (att_t < t_auto + 12)
    k = int(np.argmax(np.abs(roll) * win))
    t_turn = float(att_t[k])
    w.set_time(tb + t_turn)
    _pump(150)
    ax = json.loads(_js(w, _MODEL_AXES_JS) or '{"left":[0,0,0]}')
    check("左轉時左側往下(roll < 0 ⇒ 左軸朝下)", roll[k] < -3 and ax["left"][2] < -0.05,
          f"(roll {roll[k]:.1f}°,左軸天向分量 {ax['left'][2]:+.2f})")
    chase = ("(() => { const C = Cesium, cam = S.viewer.camera; const a = attAt(S.t - S.scene.t_base);"
             " cam.lookAtTransform(cam.transform, new C.HeadingPitchRange(C.Math.toRadians(a.yaw),"
             " C.Math.toRadians(-%s), %s)); S.viewer.scene.requestRender(); return 1; })()")
    _js(w, chase % (25, 4.0))
    _settle(w, 400)
    _shot(w, out_dir, "03_turn_chase")
    _js(w, chase % (12, 8.0))
    w.set_time(tb + 10.0)
    _settle(w, 400)
    _shot(w, out_dir, "04_ground_chase")
    w.chk_follow.setChecked(False)

    # 誤差著色 / 身分著色 / 街道底圖
    w.view_all()
    _pump(1200)
    w.cmb_color.setCurrentIndex(w.cmb_color.findData("error"))
    w.set_time(tb + 70.0)
    _settle(w, 400)
    _shot(w, out_dir, "05_error")
    w.cmb_color.setCurrentIndex(w.cmb_color.findData("role"))
    _settle(w, 300)
    _shot(w, out_dir, "06_role")
    w.cmb_color.setCurrentIndex(w.cmb_color.findData("mode"))
    w.cmb_base.setCurrentIndex(w.cmb_base.findData("osm"))
    check("街道底圖載入", _settle(w, 800))
    _shot(w, out_dir, "07_osm")
    w.cmb_base.setCurrentIndex(w.cmb_base.findData("esri"))

    # setTime 往返時間(播放時每秒 30~60 次)
    w.timings.clear()
    for k in range(300):
        w.set_time(tb + 20.0 + k * 0.1)
        QApplication.processEvents()
        _pump(16)
    _pump(300)
    tm = np.asarray(w.timings) * 1000
    if tm.size:
        print(f"    setTime 往返:n={tm.size},中位數 {np.median(tm):.2f} ms,P95 {np.percentile(tm, 95):.2f} ms,"
              f"最大 {tm.max():.2f} ms")
    check("setTime 往返 P95 < 20 ms", tm.size > 50 and np.percentile(tm, 95) < 20)

    # MAP-NOORIGIN-MSG:log 沒有地理原點 → 頁面說明,而不是「載入 log」
    w.set_scene(None, reason="noorigin")
    _pump(500)
    st = _status(w)
    check("noorigin → 頁面顯示「無法放上地圖」", st.get("overlay") == "無法放上地圖" and not w.has_scene
          and st.get("hasScene") is False, f"({st.get('overlay')!r})")
    _settle(w, 300)
    _shot(w, out_dir, "09_noorigin", full=True)
    w.set_scene(sc)
    _pump(500)
    check("再給場景 → 說明消失", _status(w).get("overlay") == "" and _status(w).get("hasScene"))

    visual_terrain_refit(w, sc, out_dir)

    w.set_scene(None)
    _pump(300)
    check("set_scene(None) 清空", _status(w).get("hasScene") is False)
    errs = [m for m in w.console_messages if m[0] == "error"]
    check("JS console 沒有錯誤", not errs and not w.errors, f"({errs[-3:]} {w.errors[-3:]})")
    w.close()
    w.deleteLater()
    _pump(200)

    visual_fake_ion_token(out_dir, sc, tb + 36.0)

    # 離線:Cesium 來源連不上 → 網頁說明、available = False
    os.environ["VIO_ATE_CESIUM_BASE"] = "https://cesium.invalid/Build/Cesium/"
    try:
        off = map3d.Map3DWidget()
        off.resize(800, 500)
        off.show()
        ok = _wait_until(lambda: off.ready, 45)
        check("離線 → 不可用(offline)", ok and not off.available and off.reason == "offline",
              f"(reason={off.reason!r})")
        _shot(off, out_dir, "08_offline")
        off.close()
        off.deleteLater()
    finally:
        os.environ.pop("VIO_ATE_CESIUM_BASE", None)
    _pump(200)


def visual_terrain_refit(w, sc: dict, out_dir: str):
    """MAP-TERRAIN-FIT:ion 地形的地面高非同步到達後重新全覽;使用者先動了相機就不動。
    用假的地形(整片 1500 m)與延遲回傳的地面高取樣模擬(不需要真的 token)。"""
    print("\n  [地形:地面高到了再全覽]")
    _js(w, _FAKE_TERRAIN_JS % {"h": 1500.0, "ms": 1500})
    _pump(300)
    w.set_scene(_shifted_scene(sc, 0.02))            # 另一份 log(不同場地)
    _pump(400)
    before = _cam(w)
    # 地形圖磚到了以後 Cesium 的碰撞偵測會把相機推到地表(貼地),還沒到就在地形底下
    check("第一次全覽用的是舊地面高(相機在 1500 m 地形下或貼地)", before.get("h", 1e9) < 1500 + 20
          and before.get("fitKey"), f"(相機高 {before.get('h', float('nan')):.0f} m)")
    _shot(w, out_dir, "10a_terrain_before")
    _wait_until(lambda: (_js(w, "window.__gd") or 0) >= 1, 10)
    _pump(300)
    _settle(w, 600)
    after = _cam(w)
    check("地面高到了 → 重新全覽(相機在地形上、原點在畫面內)",
          after.get("ground") == 1500 and after.get("fitKey") is None and after.get("h", 0) > 1500 + 20
          and _in_view(after), f"(相機高 {after.get('h', float('nan')):.0f} m,原點 {after.get('px')})")
    _shot(w, out_dir, "10b_terrain_after")

    # 使用者在地面高回來前動了相機:不再自動全覽
    _js(w, _FAKE_TERRAIN_JS % {"h": 300.0, "ms": 1500})
    w.set_scene(_shifted_scene(sc, -0.02))
    _pump(300)
    from PyQt6.QtTest import QTest
    target = w.view.focusProxy() or w.view
    pt = QPoint(w.view.width() - 40, w.view.height() // 2)
    QTest.mousePress(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pt)
    QTest.mouseRelease(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pt)
    _pump(100)
    mid = _cam(w)
    _wait_until(lambda: (_js(w, "window.__gd") or 0) >= 1, 10)
    _pump(800)
    end = _cam(w)
    check("使用者動過相機 → 地面高到了也不改視角", mid.get("fitKey") is None and end.get("ground") == 300
          and abs(end.get("h", 0) - mid.get("h", 1e9)) < 1.0,
          f"(相機高 {mid.get('h', float('nan')):.0f} → {end.get('h', float('nan')):.0f} m)")

    # 還原:沒有 token → 橢球地形
    _js(w, "(() => { S.opts.token = ''; return 1; })()")
    w._push_options()
    w.set_scene(sc)
    _pump(500)
    w.view_all()
    _pump(1000)
    check("還原橢球地形", _js(w, "S.viewer.terrainProvider instanceof Cesium.EllipsoidTerrainProvider") is True
          and _status(w).get("ground") == 0)


def visual_fake_ion_token(out_dir: str, sc: dict, t_view: float):
    """MAP-ION-TOKEN:格式正確但無效的 token → ion 請求只用這個 token(不偷用 Cesium 示範 token),
    影像與地形都失敗後退回 Esri + 橢球地形,狀態列說清楚。"""
    print("\n[假的 Cesium ion token(格式正確、但無效)]")
    from urllib.parse import parse_qs, urlparse

    from PyQt6.QtWebEngineCore import QWebEngineUrlRequestInterceptor

    class Spy(QWebEngineUrlRequestInterceptor):
        def __init__(self):
            super().__init__()
            self.urls: list[str] = []

        def interceptRequest(self, info):
            u = info.requestUrl().toString()
            if "cesium.com" in u:
                self.urls.append(u)

    tok = _fake_ion_token()
    s = QSettings(*map3d.SETTINGS)
    s.setValue(map3d.KEY_TOKEN, tok)
    s.setValue(map3d.KEY_BASE, "ion")
    s.sync()
    spy = Spy()
    prof = map3d._profile()
    prof.setUrlRequestInterceptor(spy)
    w = map3d.Map3DWidget()
    w.setWindowTitle("Map3DWidget 假 ion token")
    w.resize(1280, 800)
    w.show()
    try:
        ok = _wait_until(lambda: w.ready, 60) and w.available
        check("載入完成、底圖選 ion", ok and s.value(map3d.KEY_BASE) == "ion", f"(reason={w.reason!r})")
        if not ok:
            return
        w.set_scene(sc)
        w.set_time(t_view)
        failed = _wait_until(lambda: _status(w).get("ion") == {"img": "fail", "ter": "fail"}, 40)
        check("ion 影像與地形都失敗", failed, f"({_status(w).get('ion')},{w.errors[-2:]})")
        _settle(w, 1200)
        st = _status(w)
        check("退回 Esri 衛星圖", st.get("base") == "esri" and w.cmb_base.currentData() == "esri", f"({st.get('base')})")
        ell = _js(w, "S.viewer.terrainProvider instanceof Cesium.EllipsoidTerrainProvider")
        check("退回橢球地形、地面高 0", ell is True and st.get("ground") == 0)
        check("狀態列說明 token 無效並退回 Esri + 橢球地形",
              w.status_text.startswith("Cesium ion 無法使用(token 無效或無網路)") and "Esri" in w.status_text
              and "橢球" in w.status_text, f"({w.status_text!r})")
        check("狀態列看得到開頭(放不下時截句尾)", w.lbl_status.text().startswith("Cesium ion 無法使用"),
              f"({w.lbl_status.text()!r})")
        toks: set = set()
        for u in spy.urls:
            toks.update(parse_qs(urlparse(u).query).get("access_token", []))
        assets = sorted({urlparse(u).path for u in spy.urls})
        check("ion 請求只帶使用者的 token(不偷用 Cesium 示範 token)", bool(spy.urls) and toks == {tok},
              f"({len(spy.urls)} 個請求 {assets},token {len(toks)} 種)")
        cred = _js(w, "(document.querySelector('.cesium-widget-credits') || {}).innerText || ''") or ""
        check("沒有「default ion access token」橫幅", "default ion access token" not in cred.lower(),
              f"({cred[:60]!r})")
        check("設定仍是 ion(下次啟動會再試)", QSettings(*map3d.SETTINGS).value(map3d.KEY_BASE) == "ion")
        _shot(w, out_dir, "11_fake_ion_token", full=True)
        w.errors.clear()
    finally:
        prof.setUrlRequestInterceptor(None)
        s.remove(map3d.KEY_TOKEN)
        s.remove(map3d.KEY_BASE)
        w.close()
        w.deleteLater()
        _pump(200)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)
    # 測試會切換底圖等選項:改用獨立的設定區,不動到使用者的 QSettings
    from PyQt6.QtCore import QSettings
    map3d.SETTINGS = ("vio_ate_studio_test", "test_map3d")
    QSettings(*map3d.SETTINGS).clear()
    try:
        if VISUAL:
            rest = [a for a in sys.argv[1:] if a != "--visual"]
            out = rest[0] if rest else os.path.join(tempfile.gettempdir(), "map3d_visual")
            visual(out)
        else:
            test_scene()
            test_attitude_and_origin()
            test_models()
            test_js_source()
            test_widget()
            test_scene_deferred()
            test_color_pref()
            test_status_and_grab()
            test_profile_lock()
    finally:
        QSettings(*map3d.SETTINGS).clear()
    print(f"\n{'=' * 60}\n{'全部通過' if not failed else f'{failed} 項失敗'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
