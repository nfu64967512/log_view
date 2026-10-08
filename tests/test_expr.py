"""欄位運算式與圖表預設組驗證(不需 PyQt6、不依賴 core/dflog):python tests/test_expr.py"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import expr, presets  # noqa: E402
from core.expr import ExprError, evaluate, referenced_keys, validate  # noqa: E402

_KEY = re.compile(r"^([A-Z][A-Z0-9_]*)(?:\[(\d+)\])?(?:\.(\w+))?$")


# --- 假的 LogIndex(duck type:series / has / field_keys / messages) -------------
class _Info:
    def __init__(self, name, fields, count, instances):
        self.name, self.fields, self.count, self.instances = name, fields, count, instances
        self.units = {}


class FakeIndex:
    """tables 以 "ATT" 或 "GPS[1]" 為鍵;每張表有 "t" 與各欄位。
    有 instance 的訊息只接受 "GPS[1].Spd" 寫法(與 LogIndex.field_keys 相同的嚴格語意),
    用來驗證 expr 的 instance 退路。"""

    def __init__(self, tables):
        self.tables = {}
        for k, tab in tables.items():
            m = _KEY.match(k)
            name, inst = m.group(1), (int(m.group(2)) if m.group(2) else None)
            self.tables[(name, inst)] = {c: np.asarray(v) for c, v in tab.items()}
        self.series_calls = 0

    def messages(self):
        out = {}
        for (name, inst), tab in self.tables.items():
            fields = [c for c in tab if c != "t"]
            info = out.get(name)
            if info is None:
                info = out[name] = _Info(name, fields, 0, None)
            info.count += len(tab["t"])
            if inst is not None:
                info.instances = sorted((info.instances or []) + [inst])
        return out

    def field_keys(self):
        keys = []
        for (name, inst), tab in self.tables.items():
            for c, v in tab.items():
                if c != "t" and v.dtype.kind in "biuf":
                    keys.append(f"{name}[{inst}].{c}" if inst is not None else f"{name}.{c}")
        return sorted(keys)

    def _lookup(self, key):
        m = _KEY.match(key)
        if not m:
            return None, None
        name, inst, field = m.group(1), (int(m.group(2)) if m.group(2) else None), m.group(3)
        return self.tables.get((name, inst)), field

    def has(self, key):
        tab, field = self._lookup(key)
        return tab is not None and (field is None or field in tab)

    def series(self, key):
        self.series_calls += 1
        tab, field = self._lookup(key)
        if tab is None or field is None or field not in tab:
            return None
        v = tab[field]
        if v.dtype.kind not in "biuf":
            return None
        return tab["t"], v.astype(np.float64)


def make_index():
    t_att = np.arange(0.0, 10.0 + 1e-9, 0.1)               # 10 Hz, 0..10 s
    t_baro = np.arange(2.0, 8.0 + 1e-9, 1 / 3)            # 3 Hz, 2..8 s
    t_gps0 = np.arange(0.0, 10.0 + 1e-9, 0.2)
    t_gps1 = np.arange(1.0, 9.0 + 1e-9, 0.25)
    t_gap = np.r_[np.arange(0.0, 3.0 + 1e-9, 0.1), np.arange(6.0, 10.0 + 1e-9, 0.1)]
    return FakeIndex({
        "ATT": {"t": t_att, "Roll": np.sin(t_att), "Pitch": 2.0 * t_att,
                "Yaw": np.mod(30.0 * t_att, 360.0), "DesRoll": np.sin(t_att) + 0.1},
        "BARO": {"t": t_baro, "Alt": 100.0 + 3.0 * t_baro, "Temp": 25.0 + 0 * t_baro},
        "GPS[0]": {"t": t_gps0, "Spd": 5.0 + 0 * t_gps0, "NSats": 12.0 + 0 * t_gps0,
                   "Alt": 50.0 + t_gps0},
        "GPS[1]": {"t": t_gps1, "Spd": 7.0 + 0 * t_gps1, "NSats": 9.0 + 0 * t_gps1},
        "VISP": {"t": t_gap, "PX": 1.0 * t_gap, "PY": -2.0 * t_gap, "Rst": 0 * t_gap},
        "MSG": {"t": np.array([0.5, 1.5]), "Message": np.array(["a", "b"])},
        "EMPTY": {"t": np.array([], float), "X": np.array([], float)},
    })


def raises(e, index, *subs):
    try:
        if index is None:
            expr._compile(e)
        else:
            evaluate(e, index)
    except ExprError as err:
        msg = str(err)
        for s in subs:
            assert s in msg, f"{e!r} 的錯誤訊息應含「{s}」,實際:{msg}"
        return msg
    raise AssertionError(f"{e!r} 應該拋 ExprError")


# --- 解析與基本運算 -----------------------------------------------------------
def test_basic_arith():
    idx = make_index()
    t, v = evaluate("ATT.Roll", idx)
    tab = idx.tables[("ATT", None)]
    assert np.array_equal(t, tab["t"]) and np.allclose(v, tab["Roll"])
    _, v = evaluate("ATT.Pitch * 2 + 1 - -ATT.Pitch", idx)
    assert np.allclose(v, 2 * tab["Pitch"] + 1 + tab["Pitch"])
    _, v = evaluate("ATT.Pitch ** 2 / 4 % 7 // 1", idx)
    assert np.allclose(v, np.floor_divide(np.mod(tab["Pitch"] ** 2 / 4, 7), 1))
    _, v = evaluate("+ATT.Pitch - 1e1 + 1_0", idx)
    assert np.allclose(v, tab["Pitch"])
    _, v = evaluate("  ATT.Pitch\n *\t3 ", idx)
    assert np.allclose(v, 3 * tab["Pitch"])
    _, v = evaluate("1 / ATT.Pitch", idx)                 # 除以零 → inf,不拋例外
    assert np.isinf(v[0]) and np.isclose(v[1], 1 / tab["Pitch"][1])
    _, v = evaluate("ATT.Pitch / ATT.Pitch", idx)         # 0/0 → NaN
    assert np.isnan(v[0]) and np.allclose(v[1:], 1.0)
    print("  四則 / 次方 / 取餘 / 整除 / 正負號 / 空白換行 / 除以零 正確")


def test_referenced_keys():
    assert referenced_keys("degrees(atan2(VISP.PY, VISP.PX))") == ["VISP.PY", "VISP.PX"]
    assert referenced_keys("GPS[ 1 ].Spd*3.6 + GPS[1].Spd + GPS.Spd") == ["GPS[1].Spd", "GPS.Spd"]
    assert referenced_keys("mag(VIBE.VibeX, VIBE.VibeY, VIBE.VibeZ)") == \
        ["VIBE.VibeX", "VIBE.VibeY", "VIBE.VibeZ"]
    assert referenced_keys("XKF1.PN - XKF1.PN") == ["XKF1.PN"]
    assert referenced_keys("RCOU.C1 + RCIN_2.X_y") == ["RCOU.C1", "RCIN_2.X_y"]
    assert referenced_keys("ATT.Roll +") == ["ATT.Roll"], "語法錯也應以字面掃描回傳"
    assert referenced_keys("1e5.x + foo.ATT.Roll") == [], "不應吃到 1E5.x / foo.ATT.Roll"
    assert referenced_keys("") == []
    print("  依出現順序、不重複、instance 正規化、語法錯仍可掃描")


def test_whitelist_rejects():
    cases = [
        ("ATT.Roll.__class__", "底線"),
        ("ATT.Roll.real", "屬性"),
        ("__import__('os').system('echo hi')", "底線"),
        ("_F0_ + ATT.Roll", "底線"),
        ("(lambda: 1)()", "lambda"),
        ("[x for x in ATT.Roll]", "推導式"),
        ("abs(x for x in ATT.Roll)", "推導式"),
        ("sum(ATT.Roll)", "不支援的函式"),
        ("{x: 1 for x in ATT.Roll}", "推導式"),
        ("ATT.Roll[0]", "索引"),
        ("ATT.Roll[0:3]", "索引"),
        ("ATT[1]", "索引"),
        ("import os", "語法錯誤"),
        ("ATT.Roll if ATT.Pitch else 2", "條件"),
        ("ATT.Roll and ATT.Pitch", "and / or"),
        ("not ATT.Roll", "not"),
        ("~ATT.Roll", "~"),
        ("ATT.Roll << 1", "<<"),
        ("ATT.Roll @ ATT.Pitch", "@"),
        ("ATT.Roll is ATT.Pitch", "is"),
        ("ATT.Roll in ATT.Pitch", "in"),
        ("ATT.Roll + 'a'", "字串"),
        ("ATT.Roll + True", "常數"),
        ("ATT.Roll + None", "常數"),
        ("ATT.Roll * 1j", "複數"),
        ("(ATT.Roll, 1)", "元組"),
        ("[ATT.Roll]", "串列"),
        ("{ATT.Roll: 1}", "字典"),
        ("(x := ATT.Roll)", ":="),
        ("f'{ATT.Roll}'", "f 字串"),
        ("sqrt(x=ATT.Roll)", "具名參數"),
        ("max(*ATT.Roll)", "展開"),
        ("ATT.Roll(1)", "當函式"),
        ("pi(2)", "常數"),
        ("sqrt + ATT.Roll", "加括號"),
        ("open('x')", "不支援的函式"),
        ("eval('1') + ATT.Roll", "不支援的函式"),
        ("getattr(ATT, 'Roll')", "不支援的函式"),
        ("np.sin(ATT.Roll)", "訊息名稱要全大寫"),
        ("ATT.Roll; ATT.Pitch", "語法錯誤"),
        ("ATT.Roll\x00", "無效字元"),
        (" + ".join(["ATT.Roll"] * 200), "太長"),
        ("(" * 250 + "ATT.Roll" + ")" * 250, "巢狀太深"),
        ("abs(" * 190 + "ATT.Roll" + ")" * 190, "太複雜"),
        ("-" * 500 + "ATT.Roll", "太"),
        (" + ".join(["ATT.Roll"] * 3 + ["1"] * 200), "太複雜"),
    ]
    for e, sub in cases:
        raises(e, None, sub)
    raises(12345, None, "文字")
    print(f"  {len(cases) + 1} 種不允許的寫法皆被拒絕,且錯誤訊息具體")


def test_syntax_errors():
    raises("", None, "空的")
    raises("   ", None, "空的")
    raises("ATT.Roll +", None, "語法錯誤")
    msg = raises("(ATT.Roll", None, "括號沒有閉合", "第 1 個字元")
    raises("ATT.Roll)", None, "括號不成對")
    msg = raises("GPS[1].Spd ATT.Roll", None, "語法錯誤")
    assert "第 12 個字元" in msg, f"位置應對回原始字串:{msg}"
    raises("ATT.Roll = 1", None, "語法錯誤")
    raises("1 +* ATT.Roll", None, "語法錯誤")
    print("  語法錯誤訊息為繁中,位置對回使用者輸入的原字串")


def test_name_hints():
    raises("att.Roll", None, "全大寫", "ATT.Roll")
    raises("ATT .Roll", None, "空白", "ATT.Roll")
    raises("ATT", None, "少了欄位名稱")
    raises("Roll + 1", None, "不認得的名稱", "訊息.欄位")
    print("  小寫訊息 / 中間空白 / 只寫訊息名 / 只寫欄位名 都有對應提示")


def test_unknown_field_messages():
    idx = make_index()
    raises("ATT.Rol", idx, "沒有欄位", "ATT.Roll")
    raises("ATT.roll", idx, "ATT.Roll")                      # 大小寫錯
    raises("ATX.Roll", idx, "沒有訊息", "ATT.Roll")
    raises("ATT.Zzzzzz", idx, "沒有欄位", "可用欄位")
    raises("GPS[5].Spd", idx, "沒有 instance 5", "GPS[0]", "GPS[1]")
    raises("MSG.Message", idx, "不是數值")
    raises("ATT.Roll + BARO.Altt", idx, "BARO.Alt")
    assert validate("ATT.Roll", idx) is None
    assert "ATT.Roll" in validate("ATT.Rol", idx)
    assert validate("ATT.Rol", idx, deep=False) is not None
    assert validate("ATT.Roll * 2", idx, deep=False) is None
    assert "尚未載入" in validate("ATT.Roll", None)
    assert "尚未載入" in validate("ATT.Roll", None, deep=False)
    print("  未知欄位以 difflib 建議相近欄位;instance 不存在列出可用者;非數值欄位明確說明")


def test_unknown_function_and_arity():
    raises("sqr(ATT.Roll)", None, "不支援的函式", "sqrt")
    raises("atan(ATT.Roll, ATT.Pitch)", None, "需要 1 個", "收到 2 個")
    raises("atan2(ATT.Roll)", None, "需要 2 個")
    raises("clip(ATT.Roll, 1)", None, "需要 3 個")
    raises("min(ATT.Roll)", None, "至少需要 2 個")
    raises("mag()", None, "至少需要 1 個")
    raises("lowpass(ATT.Roll)", None, "需要 2 個")
    print("  未知函式建議相近名稱;參數個數錯誤說明需要幾個")


def test_constant_only():
    for e in ("1 + 2", "pi", "sqrt(g) * e", "10 ** 10000"):
        raises(e, None, "至少要引用一個欄位")
    print("  只有常數的運算式被拒絕")


# --- 時間基準與內插 -----------------------------------------------------------
def test_time_base_and_interp():
    idx = make_index()
    att, baro = idx.tables[("ATT", None)], idx.tables[("BARO", None)]
    t, v = evaluate("ATT.Pitch + BARO.Alt", idx)
    assert np.array_equal(t, att["t"]), "時間基準應為第一個引用欄位"
    inside = (t >= 2.0 - 1e-9) & (t <= 8.0 + 1e-9)
    assert np.all(np.isnan(v[~inside])), "超出 BARO 時間範圍應為 NaN"
    assert np.allclose(v[inside], 2.0 * t[inside] + 100.0 + 3.0 * t[inside]), "線性內插不正確"
    t2, v2 = evaluate("BARO.Alt + ATT.Pitch", idx)
    assert np.array_equal(t2, baro["t"]) and np.allclose(v2, baro["Alt"] + 2.0 * baro["t"])
    # 端點恰好相等不算超出
    t3, v3 = evaluate("BARO.Alt - BARO.Alt + ATT.Pitch", idx)
    assert np.allclose(v3[[0, -1]], 2.0 * baro["t"][[0, -1]])
    print("  時間基準 = 第一個引用欄位;其他欄位線性內插、超出自身範圍為 NaN")


def test_same_message_direct():
    """同一訊息直接用原始樣本 —— 連 NaN 都不會被內插抹平。"""
    t = np.array([0.0, 0.1, 0.25, 0.3, 0.7])
    idx = FakeIndex({"A": {"t": t, "X": np.array([1.0, np.nan, 3.0, 4.0, 5.0]),
                           "Y": np.array([10.0, 20.0, 30.0, 40.0, 50.0])}})
    tt, v = evaluate("A.Y - A.X", idx)
    assert np.array_equal(tt, t)
    assert np.isnan(v[1]) and np.allclose(v[[0, 2, 3, 4]], [9.0, 27.0, 36.0, 45.0])
    print("  同訊息不經內插,逐點對應")


def test_gap_nan():
    """被內插的欄位有資料空洞(VISP 3~6 s 斷訊)時,空洞裡為 NaN,不畫假直線。"""
    idx = make_index()
    t, v = evaluate("ATT.Pitch * 0 + VISP.PX", idx)
    in_gap = (t > 3.0 + 1e-9) & (t < 6.0 - 1e-9)
    assert in_gap.any() and np.all(np.isnan(v[in_gap])), "空洞內應為 NaN"
    visp_t = idx.tables[("VISP", None)]["t"]
    ok = ~in_gap & (t <= visp_t[-1])
    assert np.allclose(v[ok], t[ok]), "空洞外應正常內插"
    # 小間隔(< 1 s 且 < 10×中位間隔)照常內插
    tt = np.r_[np.arange(0.0, 2.0, 0.1), np.arange(2.5, 4.0, 0.1)]
    idx2 = FakeIndex({"B": {"t": np.arange(0.0, 4.0, 0.05), "Z": np.zeros(80)},
                      "C": {"t": tt, "W": tt * 2.0}})
    t2, v2 = evaluate("B.Z + C.W", idx2)
    m = (t2 >= tt[0]) & (t2 <= tt[-1])
    assert np.all(np.isfinite(v2[m])) and np.allclose(v2[m], 2.0 * t2[m])
    print("  資料空洞(> max(10×中位間隔, 1 s))內為 NaN;短間隔照常內插")


# --- 航向內插(跨 0/360) ------------------------------------------------------
class UnitIndex(FakeIndex):
    """有 unit()(同 LogIndex)的假 index;units 以欄位 key 為鍵。"""

    def __init__(self, tables, units):
        super().__init__(tables)
        self.units = units

    def unit(self, key):
        return self.units.get(key, "")


def _hdg(t):
    return 300.0 + 36.0 * t                                  # 20 s 轉兩圈:過北兩次


def test_heading_interp_wrap():
    """ATT 25 Hz 偏航穿過 360→0、VISP 30 Hz(±180):wrap_180(VISP.Yaw − ATT.Yaw) 應一直
    ≈ 2°,不在每次過北時跳到 ±180(以前 359.9 與 0.1 之間會內插出 ≈180)。"""
    ta = np.arange(0.0, 20.0, 1 / 25)
    tv = np.arange(0.013, 20.0, 1 / 30)
    idx = FakeIndex({"ATT": {"t": ta, "Yaw": np.mod(_hdg(ta), 360.0), "Roll": 0 * ta},
                     "VISP": {"t": tv, "Yaw": expr._wrap_180(_hdg(tv) + 2.0), "PX": 0 * tv}})
    plain = np.interp(tv, ta, np.mod(_hdg(ta), 360.0))
    assert np.max(np.abs(expr._wrap_180(_hdg(tv) + 2.0 - plain))) > 30, "測試資料應會跨北"
    for e, want in (("wrap_180(VISP.Yaw - ATT.Yaw)", 2.0), ("wrap_180(ATT.Yaw - VISP.Yaw)", -2.0)):
        t, v = evaluate(e, idx)
        ok = np.isfinite(v)
        assert ok.sum() >= t.size - 2, e                     # 只有超出對方時間範圍的端點是 NaN
        err = float(np.max(np.abs(v[ok] - want)))
        assert err < 1e-6, f"{e}:最大誤差 {err:.3g}°(跨北被線性內插)"
    # 維持欄位自己的範圍:ATT.Yaw 0~360、VISP.Yaw ±180,且內插值正確
    t, v = evaluate("VISP.PX + ATT.Yaw", idx)
    ok = np.isfinite(v)
    assert v[ok].min() >= 0.0 and v[ok].max() < 360.0
    assert np.max(np.abs(expr._wrap_180(v[ok] - _hdg(t[ok])))) < 1e-6
    t, v = evaluate("ATT.Roll + VISP.Yaw", idx)
    ok = np.isfinite(v)
    assert v[ok].min() > -180.0 and v[ok].max() <= 180.0
    assert np.max(np.abs(expr._wrap_180(v[ok] - _hdg(t[ok]) - 2.0))) < 1e-6
    print("  跨 0/360(±180)走最短弧:VISP−ATT 全程 2°、沒有 ±180 尖峰;範圍維持 0~360 / ±180")


def test_heading_detection():
    """航向判斷:單位 degheading;沒有單位或單位 deg 時看欄位名(Yaw / GCrs…);
    單位 deg/s 的 Yaw、名稱不像航向的欄位、已展開的累積角、不跨北的航向都與一般內插逐位元相同。"""
    ta = np.arange(0.0, 4.0, 0.04)
    tb = np.arange(0.01, 4.0, 1 / 30)
    wrapped = np.mod(350.0 + 10.0 * ta, 360.0)              # t = 1 s 過北
    truth = 350.0 + 10.0 * tb

    def plain(x):
        return np.interp(tb, ta, x, left=np.nan, right=np.nan)

    def ev(e, index):
        return evaluate(e, index)[1]

    tabs = {"A": {"t": ta, "Hd": wrapped, "Yaw": wrapped, "Rate": wrapped},
            "C": {"t": ta, "Yaw": wrapped}, "B": {"t": tb, "Z": 0 * tb}}
    idx = UnitIndex(tabs, {"A.Hd": "degheading", "A.Yaw": "deg", "C.Yaw": "deg/s"})
    for e in ("B.Z + A.Hd", "B.Z + A.Yaw"):                  # degheading;deg + 名稱 Yaw
        assert np.nanmax(np.abs(expr._wrap_180(ev(e, idx) - truth))) < 1e-9, e
    for e in ("B.Z + C.Yaw", "B.Z + A.Rate"):                # 角速度;沒單位且名稱不像航向
        assert np.array_equal(ev(e, idx), plain(wrapped), equal_nan=True), e
    assert np.nanmax(np.abs(expr._wrap_180(ev("B.Z + A.Yaw", FakeIndex(tabs)) - truth))) < 1e-9, \
        "沒有 unit() 時依欄位名 Yaw 判斷"
    cum = 200.0 * ta                                          # 已展開的累積角 0~800°
    got = ev("B.Z + A.Yaw", FakeIndex({"A": {"t": ta, "Yaw": cum}, "B": tabs["B"]}))
    assert np.array_equal(got, plain(cum), equal_nan=True), "超出 0~360 / ±180 的角度不該被包角"
    # 跨北兩側的樣本是 NaN:那一段維持 NaN,不會被最短弧補出值
    w = wrapped.copy()
    i = int(np.flatnonzero(np.abs(np.diff(w)) > 180)[0])
    w[i] = np.nan
    got = ev("B.Z + A.Yaw", FakeIndex({"A": {"t": ta, "Yaw": w}, "B": tabs["B"]}))
    near = (tb > ta[i - 1]) & (tb < ta[i + 1])
    assert np.all(np.isnan(got[near])) and np.isfinite(got[~near & (tb <= ta[-1])]).all()
    # 不跨北的航向(make_index 的 ATT.Yaw 0~300°):逐位元等於一般線性內插
    mi = make_index()
    t, v = evaluate("BARO.Alt * 0 + ATT.Yaw", mi)
    att = mi.tables[("ATT", None)]
    assert np.array_equal(v, np.interp(t, att["t"], att["Yaw"]))
    print("  degheading / deg+Yaw / 無單位+Yaw 視為航向;deg/s、累積角、不跨北者與一般內插相同")


def test_instance_fallback():
    idx = make_index()
    _, v = evaluate("GPS.Spd", idx)
    assert np.allclose(v, 5.0), "GPS.Spd 應退到最小 instance GPS[0]"
    t, v = evaluate("GPS[1].Spd * 3.6", idx)
    assert np.allclose(v, 7.0 * 3.6) and np.isclose(t[0], 1.0)
    t, v = evaluate("GPS[1].Spd - GPS.Spd", idx)
    assert np.allclose(v, 2.0)
    idx2 = FakeIndex({"ATT": {"t": np.arange(3.0), "Roll": np.arange(3.0)}})
    _, v = evaluate("ATT[0].Roll", idx2)
    assert np.allclose(v, [0, 1, 2]), "[0] 對沒有 instance 的訊息應退到無 instance 寫法"
    assert expr.resolve_key(idx, "GPS.Spd") == "GPS[0].Spd"
    assert expr.resolve_key(idx, "GPS[1].Spd") == "GPS[1].Spd"
    assert expr.resolve_key(idx, "GPS[2].Spd") is None
    assert expr.resolve_key(idx, "ATT.Nope") is None
    assert expr.resolve_key(None, "ATT.Roll") is None
    print("  未指定 instance 退到最小 instance;[0] 對無 instance 訊息也可用")


# --- 函式 ---------------------------------------------------------------------
def _one(e, xs, ys=None):
    """在單一訊息 A(X、Y 欄)上求值。"""
    xs = np.asarray(xs, float)
    cols = {"t": np.arange(xs.size, dtype=float), "X": xs}
    if ys is not None:
        cols["Y"] = np.asarray(ys, float)
    return evaluate(e, FakeIndex({"A": cols}))[1]


def test_functions():
    x = np.array([-2.5, -0.5, 0.0, 0.5, 2.5])
    y = np.array([1.0, -1.0, 2.0, -3.0, 0.5])
    u = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
    eq = np.testing.assert_allclose
    eq(_one("abs(A.X)", x), np.abs(x))
    v = _one("sqrt(A.X)", x)
    assert np.all(np.isnan(v[:2])) and np.allclose(v[2:], np.sqrt(x[2:])), "負數開根號應為 NaN"
    for f, ref in [("sin", np.sin), ("cos", np.cos), ("tan", np.tan), ("atan", np.arctan),
                   ("exp", np.exp), ("floor", np.floor), ("ceil", np.ceil),
                   ("sign", np.sign), ("degrees", np.degrees), ("radians", np.radians)]:
        eq(_one(f"{f}(A.X)", x), ref(x), err_msg=f)
    eq(_one("asin(A.X)", u), np.arcsin(u))
    eq(_one("acos(A.X)", u), np.arccos(u))
    eq(_one("atan2(A.Y, A.X)", x, y), np.arctan2(y, x))
    eq(_one("degrees(atan2(A.Y, A.X))", [-1, -1, 1, 1, 0], [1, -1, -1, 1, 1]),
       [135, -135, -45, 45, 90])
    eq(_one("hypot(A.X, A.Y)", x, y), np.hypot(x, y))
    eq(_one("mag(A.X, A.Y, 2)", x, y), np.sqrt(x ** 2 + y ** 2 + 4))
    eq(_one("mag(A.X)", x), np.abs(x))
    eq(_one("min(A.X, A.Y)", x, y), np.minimum(x, y))
    eq(_one("max(A.X, A.Y, 1)", x, y), np.maximum(np.maximum(x, y), 1))
    eq(_one("min(A.X, A.Y, 0, -1)", x, y), np.minimum(np.minimum(x, y), -1))
    pos = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
    eq(_one("log(A.X)", pos), np.log(pos))
    eq(_one("log10(A.X)", pos), [-1, 0, 1, 2, 3])
    assert np.isnan(_one("log(A.X)", [-1.0, 1.0])[0]), "負數取 log 應為 NaN"
    eq(_one("clip(A.X, -1, 1)", x), np.clip(x, -1, 1))
    eq(_one("clip(A.X, A.Y, 3)", x, y), np.clip(x, y, 3))
    eq(_one("A.X * 0 + pi + e + g", x), np.full(5, np.pi + np.e + 9.80665))
    eq(_one("rel(A.X)", [np.nan, 3.0, 4.0, 1.0]), [np.nan, 0.0, 1.0, -2.0])
    print("  全部白名單函式結果與 numpy 參考一致(含定義域外為 NaN、多參數 min/max/mag)")


def test_wrap():
    a = [180.0, -180.0, 540.0, 190.0, -190.0, 0.0, 359.9, -1e-20, 720.5, -360.0]
    np.testing.assert_allclose(_one("wrap_180(A.X)", a),
                               [180, 180, 180, -170, 170, 0, -0.1, -1e-20, 0.5, 0], atol=1e-9)
    w = _one("wrap_360(A.X)", a)
    np.testing.assert_allclose(w, [180, 180, 180, 190, 170, 0, 359.9, 0.0, 0.5, 0], atol=1e-9)
    assert np.all((w >= 0) & (w < 360)), "wrap_360 應落在 [0, 360)"
    print("  wrap_180 → (−180, 180]、wrap_360 → [0, 360)(與 ArduPilot 相同)")


def test_compare():
    x = [1.0, 2.0, np.nan, 4.0]
    np.testing.assert_array_equal(_one("A.X > 1.5", x), [0, 1, np.nan, 1])
    np.testing.assert_array_equal(_one("A.X == 2", x), [0, 1, np.nan, 0])
    np.testing.assert_array_equal(_one("A.X != 2", x), [1, 0, np.nan, 1])
    np.testing.assert_array_equal(_one("1 < A.X <= 3", x), [0, 1, np.nan, 0])
    np.testing.assert_array_equal(_one("(A.X >= 2) * A.X", x), [0, 2, np.nan, 4])
    print("  比較結果為 0/1(float),NaN 傳遞;連續比較 a < x <= b 正確")


def test_ddt():
    rng = np.random.default_rng(3)
    t = np.cumsum(rng.uniform(0.005, 0.03, 400))           # 不等距
    idx = FakeIndex({"A": {"t": t, "Q": t ** 2, "S": np.sin(t), "K": 3.0 + 0 * t}})
    _, d = evaluate("ddt(A.Q)", idx)
    assert np.allclose(d[1:-1], 2 * t[1:-1], atol=1e-9), "二次函數內部點應精確"
    assert abs(d[0] - 2 * t[0]) < 0.1 and abs(d[-1] - 2 * t[-1]) < 0.1
    _, d = evaluate("ddt(A.S)", idx)
    assert np.max(np.abs(d[1:-1] - np.cos(t[1:-1]))) < 1e-3
    _, d = evaluate("ddt(A.K)", idx)
    assert np.allclose(d, 0.0)
    _, d2 = evaluate("ddt(ddt(A.Q))", idx)
    assert np.allclose(d2[2:-2], 2.0, atol=1e-6), "二階導數應為常數 2"
    # NaN 只影響自己;鄰點退回單側差分
    x = t * 5.0
    x[100] = np.nan
    _, d = evaluate("ddt(A.X)", FakeIndex({"A": {"t": t, "X": x}}))
    assert np.isnan(d[100]) and np.sum(np.isnan(d)) == 1 and np.allclose(d[~np.isnan(d)], 5.0)
    # 重複時間戳:不產生 inf
    tr = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    _, d = evaluate("ddt(A.X)", FakeIndex({"A": {"t": tr, "X": 4.0 * tr}}))
    assert np.all(np.isfinite(d)) and np.allclose(d, 4.0), d
    # 常數參數 / 單一樣本
    _, d = evaluate("A.X * 0 + ddt(3)", FakeIndex({"A": {"t": tr, "X": tr}}))
    assert np.allclose(d, 0.0)
    try:
        evaluate("ddt(A.X)", FakeIndex({"A": {"t": [1.0], "X": [2.0]}}))
        raise AssertionError("單一樣本的微分應沒有有效值")
    except ExprError as e:
        assert "有效數值" in str(e)
    print("  ddt:不等距二階中央差分(t² 精確)、NaN 只影響自己、重複時間戳安全")


def _lp_ref(t, x, tau):
    """逐點參考實作:NaN 保持狀態、輸出 NaN。"""
    out = np.full(len(x), np.nan)
    y = None
    tp = None
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        if y is None:
            y = x[i]
        else:
            a = np.exp(-(t[i] - tp) / tau)
            y = a * y + (1 - a) * x[i]
        tp = t[i]
        out[i] = y
    return out


def test_lowpass_step_exact():
    """步階響應在不等距取樣下應精確等於 1 − e^(−t/τ)。"""
    rng = np.random.default_rng(4)
    t = np.r_[0.0, np.cumsum(rng.uniform(0.001, 0.05, 2000))]
    x = np.ones_like(t)
    x[0] = 0.0
    idx = FakeIndex({"A": {"t": t, "X": x}})
    for tau in (0.05, 0.5, 3.0):
        _, y = evaluate(f"lowpass(A.X, {tau})", idx)
        assert np.max(np.abs(y - (1 - np.exp(-t / tau)))) < 1e-9, f"τ={tau} 步階響應不符"
    _, y = evaluate("lowpass(A.X, 1/2)", idx)                    # τ 可以是常數運算式
    assert np.max(np.abs(y - (1 - np.exp(-t / 0.5)))) < 1e-9
    print("  lowpass 步階響應 = 1 − e^(−t/τ)(不等距取樣,誤差 < 1e-9)")


def test_lowpass_vs_reference():
    rng = np.random.default_rng(5)
    n = 6000
    t = np.cumsum(rng.uniform(0.001, 0.2, n))                    # 跨 ~600 s
    dup = rng.choice(np.arange(1, n), 50, replace=False)
    t[dup] = t[dup - 1]                                          # 重複時間戳
    t = np.maximum.accumulate(t)
    x = rng.normal(0, 3, n) + 50.0 * np.sin(t / 20)
    x[rng.choice(n, 300, replace=False)] = np.nan
    x[:5] = np.nan                                               # 開頭缺值
    x[3000:3400] = np.nan                                        # 長空洞
    for tau in (0.01, 0.3, 2.0, 1e4):                            # τ=0.01 → 多個向量化區塊
        y = expr.lowpass(t, x, tau)
        ref = _lp_ref(t, x, tau)
        assert np.array_equal(np.isnan(y), np.isnan(ref)), f"τ={tau} NaN 位置不同"
        m = ~np.isnan(ref)
        err = np.max(np.abs(y[m] - ref[m]))
        assert err < 1e-8, f"τ={tau} 與逐點參考差 {err:.3e}"
    # 極小 τ(區塊過碎)走逐點路徑,結果仍正確;τ→0 為直通
    y = expr.lowpass(t, x, 0.0005)
    ref = _lp_ref(t, x, 0.0005)
    m = ~np.isnan(ref)
    assert np.max(np.abs(y[m] - ref[m])) < 1e-8
    y = expr.lowpass(t, x, 1e-12)
    assert np.array_equal(np.isnan(y), np.isnan(x)) and np.allclose(y[m], x[m])
    print("  lowpass 與逐點參考一致(< 1e-8):NaN 保持狀態、長空洞、重複時間戳、多區塊、極小 τ")


def test_lowpass_errors():
    idx = make_index()
    raises("lowpass(ATT.Roll, 0)", idx, "τ", "大於 0")
    raises("lowpass(ATT.Roll, -1)", idx, "大於 0")
    raises("lowpass(ATT.Roll, ATT.Pitch)", idx, "常數")
    raises("lowpass(ATT.Roll, 1/0)", idx, "大於 0")
    _, v = evaluate("ATT.Roll * 0 + lowpass(5, 1)", idx)
    assert np.allclose(v, 5.0), "常數經低通仍是常數"
    print("  lowpass τ ≤ 0 / 非常數 / inf 皆被拒絕")


# --- 結果與錯誤 ---------------------------------------------------------------
def test_empty_result():
    idx = make_index()
    raises("EMPTY.X", idx, "沒有資料")
    raises("ATT.Roll + EMPTY.X", idx, "有效數值")
    idx2 = FakeIndex({"A": {"t": np.arange(0.0, 5.0), "X": np.ones(5)},
                      "B": {"t": np.arange(10.0, 15.0), "Y": np.ones(5)}})
    raises("A.X + B.Y", idx2, "時間範圍")
    raises("sqrt(-1 - abs(ATT.Roll))", idx, "有效數值")
    try:
        evaluate("ATT.Roll", None)
        raise AssertionError("未載入 log 應拋 ExprError")
    except ExprError as e:
        assert "尚未載入" in str(e)
    print("  空結果 / 全為 NaN / 時間不重疊 / 未載入 log 皆有說明")


def test_result_is_copy():
    idx = make_index()
    t, v = evaluate("ATT.Roll", idx)
    t[:] = -1.0
    v[:] = -1.0
    t2, v2 = evaluate("ATT.Roll", idx)
    assert t2[1] > 0 and not np.allclose(v2, -1.0), "回傳陣列應為複本"
    assert t.dtype == np.float64 and v.dtype == np.float64
    print("  回傳 float64 新陣列,修改不影響 index 快取")


def test_performance_1m():
    n = 1_000_000
    t = np.arange(n) * 0.0025                                    # 400 Hz, 2500 s
    rng = np.random.default_rng(6)
    tb = np.arange(n // 4) * 0.01
    idx = FakeIndex({"IMU": {"t": t, "AccX": rng.normal(0, 1, n), "AccY": rng.normal(0, 1, n),
                             "AccZ": rng.normal(-9.8, 1, n)},
                     "BARO": {"t": tb, "Alt": np.sin(tb)}})
    t0 = time.perf_counter()
    _, v = evaluate("lowpass(mag(IMU.AccX, IMU.AccY, IMU.AccZ), 0.2) - g + ddt(BARO.Alt)", idx)
    dt = time.perf_counter() - t0
    assert v.size == n and np.isfinite(v[1000:-1000]).all()
    assert dt < 3.0, f"1M 樣本求值 {dt:.2f} s 太慢"
    t0 = time.perf_counter()
    expr.lowpass(t, idx.tables[("IMU", None)]["AccX"], 0.0001)  # 逐點路徑
    dt2 = time.perf_counter() - t0
    print(f"  1M 樣本(mag + lowpass + ddt + 內插){dt * 1000:.0f} ms;"
          f"lowpass 逐點路徑 {dt2 * 1000:.0f} ms")


def test_compile_cache():
    a = expr._compile("ATT.Roll + 1")
    b = expr._compile("ATT.Roll + 1")
    assert a is b, "相同運算式應重用解析結果"
    print("  解析結果有快取")


# --- 預設組 -------------------------------------------------------------------
def test_presets_wellformed():
    names = set()
    for p in presets.PRESETS:
        assert p.group in presets.GROUPS, p.group
        assert (p.group, p.name) not in names, f"重複的預設:{p.group} / {p.name}"
        names.add((p.group, p.name))
        assert p.series and p.min_series >= 1
        for s in p.series:
            expr._compile(s.expr)                                # 每條運算式都必須合法
            assert s.axis in (1, 2), (p.name, s.axis)
            assert s.color is None or re.fullmatch(r"#[0-9a-fA-F]{6}", s.color), s.color
        axes = {s.axis for s in p.series}
        assert 1 in axes, f"{p.name} 只用右軸"
    groups = [p.group for p in presets.PRESETS]
    for g in ("姿態", "高度", "速度", "振動", "電力", "RC 輸入", "馬達輸出", "GPS", "EKF",
              "VIO / 外部導航", "控制器", "板子"):
        assert g in groups, f"缺少群組 {g}"
    print(f"  {len(presets.PRESETS)} 個預設、{len(set(groups))} 個群組:運算式皆合法、"
          "名稱不重複、軸只有 1/2")


def test_presets_colors_match_theme():
    from ui import theme
    assert presets.C_REF.lower() == theme.C_REF.lower()
    assert presets.C_EST.lower() == theme.C_EST.lower()
    vio = [p for p in presets.PRESETS if p.name.startswith("北向位置")][0]
    assert vio.series[0].color == theme.C_EST and vio.series[1].color == theme.C_REF
    print("  VIO 比較圖的角色色與 ui/theme 一致(VIO=估測橘、EKF=參考藍)")


def test_presets_filtering():
    t = np.arange(0.0, 5.0, 0.1)
    z = np.zeros_like(t)
    idx = FakeIndex({
        "ATT": {"t": t, "Roll": z, "Pitch": z, "DesRoll": z},            # 沒有 DesPitch / Yaw
        "VISP": {"t": t, "PX": z, "PY": z, "PZ": z, "Rst": z},
        "XKF1[0]": {"t": t, "PN": z, "PE": z, "PD": z, "VN": z, "VE": z, "VD": z},
        "XKF1[1]": {"t": t, "PN": z, "PE": z, "PD": z, "VN": z, "VE": z, "VD": z},
        "GPS[0]": {"t": t, "Spd": z, "NSats": z, "Alt": z},
        "BAT": {"t": t, "Volt": z, "Curr": z},
    })
    idx.series_calls = 0
    avail = presets.available_presets(idx)
    assert idx.series_calls == 0, "過濾預設不應載入資料"
    by = {(p.group, p.name): p for p in avail}
    roll = by[("姿態", "Roll:目標 / 實際")]
    assert [s.expr for s in roll.series] == ["ATT.DesRoll", "ATT.Roll"]
    pitch = by[("姿態", "Pitch:目標 / 實際")]
    assert [s.expr for s in pitch.series] == ["ATT.Pitch"], "缺 DesPitch 應只剩實際值"
    assert ("姿態", "Yaw:目標 / 實際") not in by, "完全沒有欄位的預設應被丟掉"
    north = by[("VIO / 外部導航", "北向位置:VIO vs EKF")]
    assert [s.expr for s in north.series] == ["VISP.PX", "XKF1.PN"], "XKF1 應退到 instance 0"
    diff = by[("VIO / 外部導航", "位置差 VIO − EKF(N / E / D)")]
    assert len(diff.series) == 3
    assert ("VIO / 外部導航", "地速:GPS[0] vs GPS[1](GPS_INPUT)") not in by, \
        "只有一個 GPS instance 時雙 GPS 比較應隱藏(min_series=2)"
    bat = by[("電力", "電池電壓 / 電流")]
    assert [(s.expr, s.axis) for s in bat.series] == [("BAT.Volt", 1), ("BAT.Curr", 2)]
    assert by[("電力", "功率 W")].series[0].expr == "BAT.Volt * BAT.Curr"
    reset = by[("VIO / 外部導航", "VIO 品質 / 重置 / 忽略")]
    assert [s.expr for s in reset.series] == ["VISP.Rst"], "舊韌體欄位名不存在時應隱藏"
    assert not any(p.group == "振動" for p in avail)
    # 不改動原始 PRESETS
    orig = [p for p in presets.PRESETS if p.name == "Pitch:目標 / 實際"][0]
    assert len(orig.series) == 2
    pitch.series.clear()
    assert len(orig.series) == 2
    # 每個留下來的序列都真的算得出來
    for p in avail:
        for s in p.series:
            assert validate(s.expr, idx) is None, (p.name, s.expr, validate(s.expr, idx))
    assert presets.available_presets(None) == []
    print(f"  依 log 內容過濾:{len(avail)} 個預設可用;缺欄位的序列隱藏、空預設丟掉、"
          "min_series 生效、不載入資料")


def test_group_presets():
    avail = presets.available_presets(make_index())
    groups = presets.group_presets(avail)
    order = list(groups)
    first_seen = []
    for p in avail:
        if p.group not in first_seen:
            first_seen.append(p.group)
    assert order == first_seen
    flat = [p for g in groups.values() for p in g]
    assert flat == sorted(avail, key=lambda p: order.index(p.group)), "組內順序應保留"
    assert presets.group_presets([]) == {}
    print(f"  分組依第一次出現的順序:{'、'.join(order)}")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  !! 失敗:{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 60}\n{len(tests) - failed}/{len(tests)} 通過")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
