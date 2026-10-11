"""失效保護偵測模組測試(core/crash/det_failsafe.py):python tests/test_crash_failsafe.py 或 pytest tests/test_crash_failsafe.py

涵蓋:
  1. 00000124 飛行 2(墜機)+ 真相錨點 / 只有 basic_anchors:CLEAR(1258.94 之前沒有任何失效保護 / 模式切換),
     GPS glitch 1258.94 列為資訊(後果)、1266.24 的 EKFCHECK / FAILSAFE_EKFINAV 在上鎖後 4.1 s(後果);
     解鎖後第一筆 ERR = 1258.94(驗收 A7「1258.94 之前沒有 ERR / 模式切換」);
  2. 00000124 飛行 1(正常降落):CLEAR;飛行 2 截到 1255.0(觸地前):CLEAR(驗收 B 的 MUST NOT);
  3. 6 個 baseline(從未解鎖):全部 NA 'not_armed'(地面事件只放 values['ground_events'],不是 HIT);
  4. 合成資料(真實 log 只有一次失事、飛行中完全沒有失效保護,所以「有事件」的分支只能用合成資料證明):
     EKF / 遙控 / 電池失效保護 + 模式切換、落地期間、上鎖後、地面、EV54、被動模式切換、MSG-only、落地狀態未知、uint8 欄位、
     代表觸發的挑選(precursor 優先)、_phase / _is_precursor 直接取自 engine.tag_time(含邊界 t == 參考時間、換掉 engine
     規則偵測器跟著變)、上鎖列之後的模式列 = post_disarm、上鎖前但晚於 t_loss = 後果 0 分、ERR 10 目標模式名稱、
     ModeReason 分類(9 / 32 / 53 指令、25 / 50 失效保護、0 / 表外 = 原因不明)、模式觸發帶 rsn、沒有 Rsn 欄 = SOFT、
     MODE 表缺失 = CLEAR 但不說「無切換」、bytes MSG、上萬筆 ERR + MSG 的耗時;
  5. 經 engine 的公開入口 analyze_ctx(via_engine,不碰私有函式):124 的 H_FAILSAFE_ACTION 0 分 / 已排除(依據帶數字);
     117 / 123 的地面失效保護不進排名;失控後的 ERR 6 / ERR 29 只放 late_triggers,H_POWER / H_VIBRATION_IMU 0 分
     (驗收 A3 / A7 / C / D4);
  6. 每個 id 在每種情境都至少回一筆;values 可 JSON 序列化;例外被 guarded 包成 NA;決定性;耗時 < 1.5 s。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import crashkit  # noqa: E402
from core.crash import det_failsafe as df  # noqa: E402
from core.crash.anchors import basic_anchors  # noqa: E402
from core.crash.ctx import Ctx  # noqa: E402
from core.crash.types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval  # noqa: E402

OWNED = ("inflight_failsafe_or_mode_change",)
T_BASE = 100.0                    # 合成 log 的 t_base(證明相對 / 絕對時間轉換有做)
MODE_NAMES = {0: "STABILIZE", 2: "ALT_HOLD", 3: "AUTO", 5: "LOITER", 6: "RTL", 9: "LAND", 14: "FLIP"}


# ======================================================================================
# 合成 log:假的 LogData / LogIndex,讓真正的 Ctx 去做 uint8 轉型、相對時間、解鎖區間、落地階梯
# ======================================================================================

class _Ix:
    def __init__(self, tables, params, series=None):
        self.tables, self.params, self._series = tables, params, series or {}

    def param_values(self):
        return dict(self.params)

    def log_messages(self):
        return []

    def table(self, name, instance=None):
        tb = self.tables.get(name)
        return None if tb is None else {k: np.asarray(v) for k, v in tb.items()}

    def raw_table(self, name):
        return self.tables[name]

    def mode_labels(self, raw):
        return [MODE_NAMES.get(int(n), f"MODE {int(n)}") for n in raw["ModeNum"]]

    def has(self, key):
        return False

    def series(self, key):
        return self._series.get(key)

    def messages(self):
        return {}


class _LD:
    def __init__(self, tables, params, t_last, series=None):
        self.index = _Ix(tables, params, series)
        self.t_base = T_BASE
        self.t_last = t_last + T_BASE
        self.events = []


def synth(arm=((10.0, 200.0, 0),), ev=(), err=(), mode=None, msg=(), params=None, t_end=300.0, takeoff=15.0, land=None, tho=None):
    """合成 Ctx。時間都是相對秒;ev / err / mode / msg 格式:ev=(t, id)、err=(t, subsys, ecode)、mode=(t, num, rsn)、msg=(t, text)。

    takeoff:每次解鎖後 EV28(離地)的時間(None = 不寫 NOT_LANDED,模擬落地狀態未知);land:EV18(落地)時間或 None;
    tho:CTUN.ThO 序列 (t_rel 陣列, 值陣列) 或 None(沒有 CTUN)。
    """
    arm_t, arm_s, arm_m = [], [], []
    for t0, t1, meth in arm:
        arm_t += [t0 + T_BASE, t1 + T_BASE]
        arm_s += [1, 0]
        arm_m += [0, meth]
    evs = list(ev)
    if takeoff is not None:
        evs.append((takeoff, 28))
    if land is not None:
        evs.append((land, 18))
    evs.sort()
    tables = {"ARM": {"t": np.array(arm_t), "ArmState": np.array(arm_s, dtype=np.uint8),
                      "Method": np.array(arm_m, dtype=np.uint8), "Forced": np.zeros(len(arm_t), dtype=np.uint8)}}
    if evs:
        tables["EV"] = {"t": np.array([e[0] + T_BASE for e in evs]), "Id": np.array([e[1] for e in evs], dtype=np.uint8)}
    if err:
        e = sorted(err)
        tables["ERR"] = {"t": np.array([x[0] + T_BASE for x in e]), "Subsys": np.array([x[1] for x in e], dtype=np.uint8),
                         "ECode": np.array([x[2] for x in e], dtype=np.uint8)}
    mrows = sorted(mode) if mode is not None else [(0.0, 5, 26)]
    tables["MODE"] = {"t": np.array([m[0] + T_BASE for m in mrows]), "Mode": np.array([m[1] for m in mrows], dtype=np.uint8),
                      "ModeNum": np.array([m[1] for m in mrows], dtype=np.uint8),
                      "Rsn": np.array([m[2] for m in mrows], dtype=np.uint8)}
    if msg:
        m = sorted(msg)
        tables["MSG"] = {"t": np.array([x[0] + T_BASE for x in m]), "Message": np.array([x[1] for x in m])}
    series = {}
    if tho is not None:
        series["CTUN.ThO"] = (np.asarray(tho[0], dtype=float) + T_BASE, np.asarray(tho[1], dtype=float))
    return Ctx(_LD(tables, params or {}, t_end, series))


def anchors(ctx, **kw):
    a = basic_anchors(ctx)
    a.end_type = kw.pop("end_type", "unknown")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class _PinAnchors:
    """假的「前置模組」(取代 end_state / attitude 模組):在 engine 的 Anchors 上寫入指定錨點,不回 Finding。"""
    __name__ = "pin_anchors"

    def __init__(self, **kw):
        self.kw = kw

    def run(self, ctx, a):
        for k, v in self.kw.items():
            setattr(a, k, v)
        return []


def via_engine(ctx, **kw):
    """經 engine 的公開入口 analyze_ctx(貼標 + 假說評分),不碰 engine 的私有函式。

    回傳 (report, 本模組的 Finding, {假說 id: HypScore})。錨點用 _PinAnchors 在本模組之前寫入。"""
    from core.crash import engine
    rep = engine.analyze_ctx(ctx, modules=[_PinAnchors(**kw), df])
    return rep, rep.finding(OWNED[0]), {h.id: h for h in rep.hypotheses}


def fs_contrib(hs, hid):
    """某假說中來自本偵測器的計分總和(> 0 = 本偵測器的證據被算進去)。"""
    return sum(e["contrib"] for e in hs[hid].evidence if e["det"] == OWNED[0])


# ======================================================================================
# 共用檢查
# ======================================================================================

def pick(fs, det=OWNED[0]):
    c = [f for f in fs if f.id == det]
    assert c, f"{det} 沒有輸出:{[(f.id, f.status, f.context) for f in fs]}"
    return c[0]


def check_contract(fs, label):
    ids = {f.id for f in fs}
    assert set(OWNED) <= ids, f"{label}:缺少 {set(OWNED) - ids}"
    for f in fs:
        assert f.domain == df.DOMAIN == "failsafe" and f.id in OWNED, (label, f.id, f.domain)
        assert f.status in (HIT, SOFT, CLEAR, NA), f.status
        json.dumps(f.values)                                  # values 必須可 JSON 序列化
        assert f.text_zh, (label, f.id, "text_zh 空")
        if f.status in (HIT, SOFT):
            assert f.t_onset is not None and f.t_alarm is not None and f.t_onset <= f.t_alarm + 1e-9, (label, f.t_onset, f.t_alarm)
            assert f.values.get("triggers"), (label, "HIT/SOFT 沒有 triggers 清單")
            for tr in f.values["triggers"]:
                assert {"t", "kind", "label"} <= set(tr), tr
        else:
            assert f.t_onset is None


def statuses(fs):
    return [(f.id, f.status, f.context) for f in fs]


def near(x, ref, tol):
    return x is not None and abs(float(x) - ref) <= tol


# ======================================================================================
# 1. 00000124 飛行 2(墜機)
# ======================================================================================

def test_124_crash_flight_clear_with_truth_anchors():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.truth_anchors_124(ctx)
    t0 = time.perf_counter()
    fs = df.run(ctx, a)
    dt = time.perf_counter() - t0
    check_contract(fs, "124 飛行2")
    f = pick(fs)
    assert f.status == CLEAR and f.context == "air", statuses(fs)
    v = f.values
    # 解鎖 873.14–1262.13 s 之間沒有任何失效保護 / 模式切換(驗收 A7:no ERR/mode change before 1258.94)
    assert v["triggers"] == [] and v["n_triggers"] == 0
    assert near(v["interval"][0], 873.14, 0.01) and near(v["interval"][1], 1262.125, 0.01), v["interval"]
    assert v["modes"] == ["LOITER"] and v["mode_changes"] == [], (v["modes"], v["mode_changes"])
    assert v["landed_state_known"] is True and 380.0 < v["airborne_s"] < 386.0, v["airborne_s"]    # EV28 879.01 → 上鎖
    assert v["landed_events"] == []                     # 上鎖當下補寫的 EV17 / 18 不算降落
    # GPS glitch 1258.94–1259.74(ERR 11)只是資訊,而且晚於 t_warn(後果)
    g = v["info_events"]
    assert len(g) == 1 and near(g[0]["t"], 1258.94, 0.05) and near(g[0]["t_end"], 1259.74, 0.05), g
    assert g[0]["phase"] == "consequence" and 2.5 < -g[0]["lead_to_warn_s"] < 3.5, g
    # 1266.24 EKFCHECK / FAILSAFE_EKFINAV:上鎖後 4.1 s,是後果,不計入飛行中
    p = v["post_disarm"]
    assert len(p) == 1 and near(p[0]["t"], 1266.24, 0.05) and p[0]["subsys"] == [16, 17], p
    assert near(p[0]["dt_after_disarm"], 4.115, 0.05), p
    # 753–815 s 的 GPS glitch / EKF 失效保護在飛行 1 與飛行 2 之間的地面(未解鎖):只列為地面資訊
    gt = [(round(x["t"], 1), x["label"].split("(")[0].split(" ")[0]) for x in v["ground_events"]]
    assert len(v["ground_events"]) == 3 and near(v["ground_events"][0]["t"], 753.24, 0.05) \
        and near(v["ground_events"][1]["t"], 754.34, 0.05) and near(v["ground_events"][1]["t_end"], 811.54, 0.05) \
        and near(v["ground_events"][2]["t"], 810.64, 0.05), gt
    assert v["fs_params"]["FS_EKF_ACTION"] == 0.0 and v["caveats"]
    # 驗收 A7 的數字:解鎖後第一筆 ERR(任何子系統)= 1258.94 GPS glitch → 之前沒有任何 ERR / 模式切換
    fe = v["first_err_in_interval"]
    assert near(fe["t"], 1258.94, 0.05) and fe["subsys"] == 11 and fe["ecode"] == 2, fe
    assert "LOITER" in f.text_zh and "後果" in f.text_zh and "1266.24" in f.text_zh
    assert "第一筆 ERR 在 1258.94" in f.text_zh, f.text_zh
    assert dt < 1.5, f"耗時 {dt:.2f}s"
    print(f"  CLEAR(air);glitch {g[0]['t']:.2f}–{g[0]['t_end']:.2f}(後果)、EKF failsafe {p[0]['t']:.2f}"
          f"(上鎖後 {p[0]['dt_after_disarm']:.2f}s)、地面事件 {len(v['ground_events'])} 筆;{dt * 1000:.0f} ms")


def test_124_crash_flight_clear_with_bare_anchors():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = basic_anchors(ctx)
    assert a.interval is not None and a.t_warn is None and a.t_loss is None
    fs = df.run(ctx, a)
    check_contract(fs, "124 bare")
    f = pick(fs)
    assert f.status == CLEAR and f.values["triggers"] == []
    assert len(f.values["post_disarm"]) == 1
    # 沒有 t_warn:參考時間退到 t_detect → 沿用 engine 的規則(不丟例外、不亂標 consequence)
    assert f.values["info_events"][0].get("phase") in ("precursor", "consequence", None)
    print("  basic_anchors:CLEAR,不依賴其他模組的錨點")


def test_124_flight1_normal_landing_clear():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.flight1_anchors_124(ctx)
    fs = df.run(ctx, a)
    check_contract(fs, "124 飛行1")
    f = pick(fs)
    assert f.status == CLEAR and f.context == "air", statuses(fs)
    v = f.values
    assert v["triggers"] == [] and v["modes"] == ["LOITER"] and v["mode_changes"] == []
    assert near(v["interval"][0], 335.24, 0.01) and near(v["interval"][1], 619.14, 0.01)
    assert 270.0 < v["airborne_s"] < 282.0, v["airborne_s"]                 # EV28 340.62 → EV18 617.19
    assert v["info_events"] == [] and v["post_disarm"] == []
    # 飛行 2 的 1258.94 / 1266.24 不屬於這次分析;753–815 與 1266.24 都是「地面」(不在任何解鎖區間內)
    gts = [round(x["t"], 1) for x in v["ground_events"]]
    assert 753.2 in gts and 754.3 in gts and 810.6 in gts and 1266.2 in gts and not any(abs(t - 1258.9) < 0.2 for t in gts), gts
    assert v["first_err_in_interval"] is None and "沒有任何 ERR" in f.text_zh       # 飛行 1 整段沒有 ERR 列
    assert "無切換" in f.text_zh
    print(f"  飛行 1:CLEAR;離地 {v['airborne_s']:.0f}s;地面事件 {len(gts)} 筆")


def test_124_flight2_before_1255_no_hit():
    """驗收 B 的 MUST NOT:飛行 2 在 1255.0 之前(觸地前)不得有任何 HIT / SOFT。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    full = ctx.armed[1]
    a = Anchors(intervals=list(ctx.armed))
    a.interval = ArmedInterval(full.index, full.t0, 1255.0, method=0)       # 截到觸地之前
    a.t_disarm = 1255.0
    a.end_type = "unknown"
    fs = df.run(ctx, a)
    check_contract(fs, "124 飛行2 <1255")
    f = pick(fs)
    assert f.status == CLEAR and f.values["triggers"] == [] and f.values["mode_changes"] == [], statuses(fs)
    assert f.values["info_events"] == [] and f.values["landed_events"] == [] and f.values["first_err_in_interval"] is None
    assert f.values["modes"] == ["LOITER"] and 370.0 < f.values["airborne_s"] < 380.0, f.values["airborne_s"]
    print(f"  飛行 2 截至 1255.0:CLEAR,離地 {f.values['airborne_s']:.0f}s 內沒有任何 ERR / 模式切換")


# ======================================================================================
# 3. baseline(從未解鎖)
# ======================================================================================

def test_baselines_never_armed_are_na_only():
    expect_ground = {"00000117": 1, "00000118": 1, "00000119": 0, "00000120": 0, "00000122": 2, "00000123": 9}
    for name in crashkit.BASELINES:
        ctx = crashkit.load_ctx(name)
        a = crashkit.not_armed_anchors(ctx)
        assert a.interval is None and a.end_type == "not_armed"
        fs = df.run(ctx, a)
        check_contract(fs, name)
        assert [f.status for f in fs] == [NA] and fs[0].context == "not_armed", (name, statuses(fs))
        assert fs[0].values["n_ground_events"] == expect_ground[name], (name, fs[0].values["n_ground_events"])
        assert "未解鎖" in fs[0].text_zh
    # 地面事件的時間(驗收 C):117 FAILSAFE_GCS 611.09;118 EKF 94.36–101.76;122 兩段 GPS glitch;123 的 7 次 EKFCHECK
    g117 = df.run(crashkit.load_ctx("00000117"), Anchors())[0].values["ground_events"]
    assert near(g117[0]["t"], 611.09, 0.05) and g117[0]["subsys"] == [8] and g117[0]["msg"] == "GCS Failsafe", g117
    g118 = df.run(crashkit.load_ctx("00000118"), Anchors())[0].values["ground_events"]
    assert near(g118[0]["t"], 94.36, 0.05) and near(g118[0]["t_end"], 101.76, 0.05) and g118[0]["subsys"] == [16, 17], g118
    g122 = df.run(crashkit.load_ctx("00000122"), Anchors())[0].values["ground_events"]
    assert [(round(x["t"], 1), round(x["t_end"], 1)) for x in g122] == [(398.6, 449.6), (519.8, 602.3)], g122
    g123 = df.run(crashkit.load_ctx("00000123"), Anchors())[0].values["ground_events"]
    ekf = [x for x in g123 if x.get("subsys") == [16, 17]]
    assert [round(x["t"], 1) for x in ekf] == [155.2, 179.6, 203.4, 213.0, 241.6, 314.2, 347.3], [x["t"] for x in ekf]
    print("  6 個 baseline 全部 NA not_armed;地面事件(117 GCS / 118 EKF / 122 glitch / 123 ×7 EKFCHECK)只列資訊")


# ======================================================================================
# 4. 合成資料:有事件的分支
# ======================================================================================

def test_synth_ekf_failsafe_with_land_mode_is_hit():
    ctx = synth(err=[(100.0, 16, 2), (100.0, 17, 1), (110.0, 16, 0), (110.0, 17, 0)],
                mode=[(0.0, 5, 26), (10.0, 5, 1), (100.05, 9, 6)], msg=[(100.0, "EKF variance")])
    a = anchors(ctx)
    fs = df.run(ctx, a)
    check_contract(fs, "ekf")
    f = pick(fs)
    assert f.status == HIT and f.severity == "high" and f.context == "air", statuses(fs)
    assert near(f.t_onset, 100.0, 1e-6) and near(f.t_alarm, 100.0, 1e-6)       # 絕對 → 相對時間有轉
    tr = f.values["triggers"]
    assert len(tr) == 1 and tr[0]["kind"] == "err" and tr[0]["subsys"] == [16, 17], tr    # EKFCHECK + INAV 合成一段
    assert tr[0]["mode_to"] == "LAND" and tr[0]["mode_reason"] == "EKF_FAILSAFE" and near(tr[0]["t_mode"], 100.05, 1e-6), tr
    assert near(tr[0]["t_end"], 110.0, 1e-6) and near(tr[0]["duration"], 10.0, 1e-6) and tr[0]["state"] == "flying"
    assert tr[0]["msg"] == "EKF variance"
    assert f.values["n_hit"] == 1 and f.values["t_first_err"] == 100.0 and f.values["confidence"] == "low"
    assert f.values["mode_changes"][0]["cls"] == "failsafe" and f.values["mode_changes"][0]["reason"] == "EKF_FAILSAFE"
    assert f.values["modes"] == ["LOITER", "LAND"]
    first = f.text_zh.split("。")[0]                                            # engine 用第一句當時間軸標籤
    assert "100.00 s" in first and "EKF" in first and "LAND" in first and "持續 10.0 s" in first, first
    print(f"  EKF failsafe + LAND:{first}")


def test_synth_radio_failsafe_mode_change_only_merges_with_err():
    ctx = synth(err=[(60.0, 5, 1), (75.0, 5, 0)], mode=[(0.0, 5, 26), (60.02, 6, 3), (75.0, 5, 1)])
    a = anchors(ctx, t_warn=150.0, t_loss=152.0)
    f = pick(df.run(ctx, a))
    assert f.status == HIT
    tr = f.values["triggers"]
    # ERR 5 + 隨後 MODE(Rsn 3)= 同一個觸發;之後 Rsn 1(飛手切回 LOITER)不算觸發
    assert len(tr) == 1 and tr[0]["mode_to"] == "RTL" and tr[0]["mode_reason"] == "RADIO_FAILSAFE", tr
    assert tr[0]["phase"] == "precursor" and near(tr[0]["lead_to_warn_s"], 90.0, 1e-6) and near(tr[0]["lead_to_loss_s"], 92.0, 1e-6)
    assert f.values["phase"] == "precursor" and "可能是原因之一" in f.text_zh
    assert [m["cls"] for m in f.values["mode_changes"]] == ["failsafe", "pilot"]
    print("  遙控失效保護:ERR 5 與 MODE(Rsn 3)合併為一筆,先於 t_warn 90 s")


def test_synth_mode_change_failsafe_reason_without_err_is_hit():
    ctx = synth(mode=[(0.0, 5, 26), (50.0, 6, 4)])               # 電池失效保護造成 RTL,沒有 ERR 6(例如 log 掉了)
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == HIT and near(f.t_onset, 50.0, 1e-6)
    tr = f.values["triggers"]
    assert len(tr) == 1 and tr[0]["kind"] == "mode" and tr[0]["mode_reason"] == "BATTERY_FAILSAFE", tr
    assert "BATTERY_FAILSAFE" in tr[0]["label"]
    print("  只有 MODE(Rsn 4)也能 HIT")


def test_synth_failsafe_after_loss_is_consequence_phase():
    """失效保護晚於姿態失控:仍是 HIT(事實),但 phase = consequence、文字寫「後果」,engine 會依 t_onset 貼 consequence。"""
    ctx = synth(err=[(180.0, 8, 1)])
    a = anchors(ctx, t_warn=170.0, t_loss=175.0)
    f = pick(df.run(ctx, a))
    assert f.status == HIT and near(f.t_onset, 180.0, 1e-6)
    assert f.values["phase"] == "consequence" and f.values["triggers"][0]["phase"] == "consequence"
    assert f.values["triggers"][0]["subsys"] == [8] and "地面站" in f.values["triggers"][0]["label"]
    assert "後果" in f.text_zh
    # development
    f2 = pick(df.run(ctx, anchors(ctx, t_warn=170.0, t_loss=190.0)))
    assert f2.values["phase"] == "development"
    print("  t_onset 晚於 t_loss → consequence;介於 t_warn / t_loss → development")


def test_synth_precursor_preferred_over_later_hit():
    """代表觸發的挑選:早於 t_warn 的證據優先(觸發型證據只有 precursor 才算原因),失控後的 HIT 不能蓋掉它。"""
    # 50 s 被動切換(SOFT,precursor);160 s EKF 失效保護(HIT,晚於 t_loss 155 → 後果)
    ctx = synth(err=[(160.0, 16, 2), (160.0, 17, 1)], mode=[(0.0, 5, 26), (50.0, 6, 14)])
    a = anchors(ctx, t_warn=150.0, t_loss=155.0)
    f = pick(df.run(ctx, a))
    assert f.status == SOFT and near(f.t_onset, 50.0, 1e-6) and f.values["phase"] == "precursor", statuses([f])
    assert f.values["n_hit"] == 1 and f.values["n_precursor"] == 1 and near(f.values["t_selected"], 50.0, 1e-6)
    assert "之後另有 1 項" in f.text_zh and "160.00 s" in f.text_zh and "後果" in f.text_zh, f.text_zh
    # triggers 只列 precursor(與 Finding 的標籤同一層);160 s 的後果移到 late_triggers
    assert [x["t"] for x in f.values["triggers"]] == [50.0], f.values["triggers"]
    assert [(x["t"], x["phase"]) for x in f.values["late_triggers"]] == [(160.0, "consequence")]
    assert f.values["n_triggers"] == 2 and f.values["n_late"] == 1
    # 早的 SOFT + precursor HIT → HIT(precursor HIT 優先於更早的 SOFT)
    ctx2 = synth(err=[(100.0, 5, 1), (101.0, 5, 0)], mode=[(0.0, 5, 26), (50.0, 6, 14)])
    f2 = pick(df.run(ctx2, anchors(ctx2, t_warn=150.0, t_loss=155.0)))
    assert f2.status == HIT and near(f2.t_onset, 100.0, 1e-6) and near(f2.values["t_first_trigger"], 50.0, 1e-6)
    # 都不是 precursor:development 的 SOFT(152)+ consequence 的 HIT(160)→ HIT 160(後果)
    ctx3 = synth(err=[(160.0, 8, 1)], mode=[(0.0, 5, 26), (152.0, 6, 14)])
    f3 = pick(df.run(ctx3, anchors(ctx3, t_warn=150.0, t_loss=155.0)))
    assert f3.status == HIT and near(f3.t_onset, 160.0, 1e-6) and f3.values["phase"] == "consequence"
    assert f3.values["n_precursor"] == 0 and "late_triggers" not in f3.values and len(f3.values["triggers"]) == 2
    assert "另有 1 項也發生在姿態異常之後" in f3.text_zh, f3.text_zh
    # development 的觸發:文字要說是反應 / 不算原因(engine 會把觸發型 development 改標 consequence)
    ctx4 = synth(err=[(152.0, 8, 1)])
    f4 = pick(df.run(ctx4, anchors(ctx4, t_warn=150.0, t_loss=155.0)))
    assert f4.values["phase"] == "development" and "不算原因" in f4.text_zh, f4.text_zh
    print("  precursor SOFT 優先於失控後的 HIT;precursor HIT 優先於更早的 SOFT;都晚於 t_warn → 最早的 HIT")


def test_late_triggers_never_score_other_hypotheses():
    """Finding 標 precursor 時,失控後才出現的電池(ERR 6)/ 振動(ERR 29)失效保護只放 late_triggers,
    不得讓 H_POWER / H_VIBRATION_IMU 計分(後果永不計分;審查 P3c / P3d / P3f 的重現)。"""
    kw = dict(t_warn=150.0, t_loss=155.0, end_type="hard_impact")
    cases = {"P3c 被動切換 + ERR 6": dict(err=[(160.0, 6, 1)], mode=[(0.0, 5, 26), (50.0, 6, 14)]),
             "P3d 被動切換 + ERR 29": dict(err=[(160.0, 29, 1)], mode=[(0.0, 5, 26), (50.0, 6, 14)]),
             "P3f 遙控 + ERR 6": dict(err=[(100.0, 5, 1), (101.0, 5, 0), (160.0, 6, 1)]),
             "遙控 + ERR 29": dict(err=[(100.0, 5, 1), (101.0, 5, 0), (160.0, 29, 1)])}
    for label, c in cases.items():
        rep, f, hs = via_engine(synth(**c), **kw)
        assert f.tag == "precursor" and f.values["phase"] == "precursor", (label, f.tag)
        assert all(x.get("phase") == "precursor" for x in f.values["triggers"]), (label, f.values["triggers"])
        late = f.values["late_triggers"]
        assert len(late) == 1 and late[0]["phase"] == "consequence" and near(late[0]["t"], 160.0, 1e-9), (label, late)
        assert not any(s in (6, 29) for x in f.values["triggers"] for s in x.get("subsys", [])), (label, f.values["triggers"])
        for hid in ("H_POWER", "H_VIBRATION_IMU"):
            assert fs_contrib(hs, hid) == 0.0 and hs[hid].raw == 0.0 and hs[hid].score == 0.0 \
                and hs[hid].status != "ranked", (label, hid, hs[hid].raw, hs[hid].status)
        assert fs_contrib(hs, "H_FAILSAFE_ACTION") > 0, label         # 真正的前兆照常計分
    # 對照(證明上面不是空集合):ERR 29 早於 t_warn 時 H_VIBRATION_IMU 確實由本偵測器計分
    rep, f, hs = via_engine(synth(err=[(140.0, 29, 1)]), **kw)
    assert f.tag == "precursor" and fs_contrib(hs, "H_VIBRATION_IMU") > 0 and "late_triggers" not in f.values, \
        (f.tag, fs_contrib(hs, "H_VIBRATION_IMU"))
    print("  失控後的 ERR 6 / ERR 29 只在 late_triggers:H_POWER / H_VIBRATION_IMU 0 分;早於 t_warn 的 ERR 29 照常計分")


def test_mode_triggers_carry_rsn():
    """模式切換觸發與「ERR + 模式列」合併的觸發都帶數字 rsn(假說用 rsn 找電池失效保護等原因)。"""
    from core.crash import hypotheses as hyp
    ctx = synth(mode=[(0.0, 5, 26), (140.0, 6, 4)])                       # 只有 MODE Rsn 4,沒有 ERR 6
    f = pick(df.run(ctx, anchors(ctx, t_warn=150.0, t_loss=155.0)))
    tr = f.values["triggers"]
    assert f.status == HIT and len(tr) == 1 and tr[0]["kind"] == "mode" and tr[0]["rsn"] == 4, tr
    ctx2 = synth(err=[(140.0, 6, 1)], mode=[(0.0, 5, 26), (140.05, 6, 4)])   # ERR 6 + MODE Rsn 4 合併
    f2 = pick(df.run(ctx2, anchors(ctx2, t_warn=150.0, t_loss=155.0)))
    tr2 = f2.values["triggers"]
    assert len(tr2) == 1 and tr2[0]["kind"] == "err" and tr2[0]["rsn"] == 4 and tr2[0]["subsys"] == [6], tr2
    fs_trigger = getattr(hyp, "_fs_trigger", None)                       # 假說端的消費者(若仍存在)要認得出
    if fs_trigger is not None:
        assert fs_trigger(f, rsn=frozenset({4})) and fs_trigger(f2, rsn=frozenset({4}))
    json.dumps(f.values)
    print("  模式觸發 / 合併觸發都帶 rsn = 4")


def test_mode_reason_classes():
    """ModeReason 分類(ArduPilot ModeReason.h):9 推油門脫離 LAND / 32 腳本 / 53 AUX = 指令(CLEAR);
    25 一般失效保護 / 50 航位推算 = HIT;0 UNKNOWN / 表外編號 = SOFT「原因不明」;14 避障 = SOFT「被動切換」。"""
    expect = {9: (CLEAR, None), 32: (CLEAR, None), 53: (CLEAR, None), 52: (CLEAR, None), 41: (CLEAR, None),
              25: (HIT, "失效保護切換模式"), 50: (HIT, "失效保護切換模式"), 29: (HIT, "失效保護切換模式"),
              0: (SOFT, "原因不明"), 99: (SOFT, "原因不明"), 14: (SOFT, "被動切換模式"), 17: (SOFT, "被動切換模式")}
    for rsn, (st, word) in expect.items():
        ctx = synth(mode=[(0.0, 5, 26), (50.0, 2, rsn)])
        f = pick(df.run(ctx, anchors(ctx)))
        assert f.status == st, (rsn, f.status, f.text_zh)
        if word:
            assert word in f.values["triggers"][0]["label"], (rsn, f.values["triggers"][0]["label"])
        else:
            assert "皆為飛手指令" in f.text_zh, (rsn, f.text_zh)
    # 25 FAILSAFE(一般失效保護)可與任何失效保護 ERR 合併;50 DEADRECKON 與 ERR 31 合併
    ctx = synth(err=[(50.0, 26, 1)], mode=[(0.0, 5, 26), (50.05, 6, 25)])
    tr = pick(df.run(ctx, anchors(ctx))).values["triggers"]
    assert len(tr) == 1 and tr[0]["subsys"] == [26] and tr[0]["mode_reason"] == "FAILSAFE", tr
    ctx = synth(err=[(50.0, 31, 1)], mode=[(0.0, 5, 26), (50.05, 6, 50)])
    tr = pick(df.run(ctx, anchors(ctx))).values["triggers"]
    assert len(tr) == 1 and tr[0]["subsys"] == [31] and tr[0]["mode_reason"] == "DEADRECKON_FAILSAFE", tr
    print("  Rsn 9/32/41/52/53 = 指令;25/29/50 = HIT(可與 ERR 合併);0 / 99 = 原因不明 SOFT;14 / 17 = 被動 SOFT")


def test_phase_matches_engine_tag_time():
    """_phase / _is_precursor 直接取自 engine.tag_time(不另抄規則),含邊界 t == 參考時間、正常結束、區間外;
    engine 的規則改了,偵測器跟著改(用 monkeypatch 換掉 tag_time 證明沒有私藏的副本)。"""
    from core.crash import engine
    from core.crash.engine import tag_time
    from core.crash.types import TAG_CONSEQUENCE, TAG_PRECURSOR
    ctx = synth()
    rng = np.random.default_rng(7)
    combos = [dict(), dict(t_warn=150.0, t_loss=155.0), dict(t_warn=150.0), dict(t_first_hit=170.0),
              dict(t_detect=190.0), dict(t_loss=120.0, t_detect=190.0), dict(t_warn=60.0, t_loss=61.0, t_first_hit=62.0),
              dict(t_first_hit=150.0, t_detect=152.0, end_type="hard_impact"),
              dict(t_warn=150.0, t_loss=155.0, end_type="crash_check"), dict(end_type="normal_landing"),
              dict(t_warn=150.0, end_type="normal_landing")]
    fixed = [5.0, 10.0, 60.0, 61.0, 62.0, 120.0, 150.0, 152.0, 155.0, 170.0, 190.0, 200.0, 200.005, 230.0]
    n = 0
    for kw in combos:
        a = anchors(ctx, **kw)
        for t in np.concatenate([rng.uniform(10.0, 200.0, 200), fixed]):
            t = float(t)
            tag = tag_time(t, a)
            ph, lead = df._phase(t, a)
            assert df._is_precursor(t, a) == (tag == TAG_PRECURSOR), (kw, t, tag, ph)
            assert ph == (tag or None), (kw, t, tag, ph)
            n += 1
    # 換掉 engine 的規則(把「與參考時間同時」改成 precursor)→ 偵測器的階段與文字跟著變,和計分一致
    ctx2 = synth(arm=((10.0, 160.0, 12),), err=[(150.0, 25, 1)])
    a2 = anchors(ctx2, t_first_hit=150.0, t_detect=152.0, end_type="hard_impact")
    orig = engine.tag_time

    def tag_le(t, a, standing=False):
        r = orig(t, a, standing)
        return TAG_PRECURSOR if (r == TAG_CONSEQUENCE and a.t_warn is None and a.t_first_hit is not None
                                 and t <= a.t_first_hit + 1e-6) else r
    engine.tag_time = tag_le
    try:
        f = pick(df.run(ctx2, a2))
        assert f.values["phase"] == "precursor" and "可能是原因之一" in f.text_zh and "-0.0" not in f.text_zh, f.text_zh
    finally:
        engine.tag_time = orig
    f = pick(df.run(ctx2, a2))
    assert f.values["phase"] == tag_time(150.0, a2), (f.values["phase"], tag_time(150.0, a2))
    print(f"  {n} 個 (錨點, 時間) 組合:_phase / _is_precursor 與 engine.tag_time 一致;換掉 engine 規則偵測器跟著變")


def test_boundary_at_reference_time_text_matches_engine_score():
    """觸發剛好在參考時間(沒有 t_warn、t == t_first_hit):偵測器的階段 / 文字和 engine 的標籤 / 計分一致
    (審查重現:舊版文字寫「後果 -0.0 s」而 engine 標 precursor 並計 4 分)。"""
    ctx = synth(arm=((10.0, 160.0, 12),), err=[(150.0, 25, 1)])
    rep, f, hs = via_engine(ctx, t_first_hit=150.0, t_detect=152.0, end_type="hard_impact")
    assert f.status == HIT and near(f.t_onset, 150.0, 1e-9)
    assert "-0.0" not in f.text_zh and "第一次撞擊" in f.text_zh and "姿態異常" not in f.text_zh, f.text_zh
    assert f.values["lead_ref"] == "t_first_hit" and f.values["lead_to_warn_s"] == 0.0
    raw = hs["H_FAILSAFE_ACTION"].raw
    if f.tag == "precursor":                 # engine 把「同時」算起點 → 偵測器也說可能是原因,而且計分
        assert f.values["phase"] == "precursor" and "可能是原因之一" in f.text_zh and raw > 0, (f.tag, raw, f.text_zh)
    else:                                    # engine 把「同時」算之後 → 偵測器也說後果,而且 0 分
        assert f.tag == "consequence" and f.values["phase"] == "consequence" and "後果" in f.text_zh and raw == 0.0, \
            (f.tag, raw, f.text_zh)
    # 參考時間是 t_detect 時文字寫「飛控判定失事」
    ctx2 = synth(arm=((10.0, 200.0, 12),), err=[(170.0, 8, 1)])
    f2 = pick(df.run(ctx2, anchors(ctx2, t_detect=190.0, end_type="crash_check")))
    assert "早於飛控判定失事 20.0 s" in f2.text_zh and f2.values["lead_ref"] == "t_detect", f2.text_zh
    print(f"  t == t_first_hit:engine 標 {f.tag}、偵測器 phase {f.values['phase']}、H_FAILSAFE raw {raw:.1f}(一致)")


def test_synth_crash_failsafe_mode_at_disarm_scores_zero():
    """飛行中 = 上鎖列之前(含):上鎖列之後才寫的 CRASH_FAILSAFE 模式列 → post_disarm(CLEAR);
    上鎖前、晚於 t_loss 的同一列 → HIT 但後果,經 engine 後 H_FAILSAFE_ACTION = 0 且已排除(驗收 D4)。"""
    kw = dict(t_warn=190.0, t_loss=195.0, t_detect=200.0, end_type="crash_check")
    ctx = synth(arm=((10.0, 200.0, 12),), mode=[(0.0, 5, 26), (200.004, 9, 19)])
    rep, f, hs = via_engine(ctx, **kw)
    assert f.status == CLEAR and f.values["triggers"] == [], statuses([f])
    assert [(x["t"], x["dt_after_disarm"]) for x in f.values["post_disarm"]] == [(200.004, 0.004)], f.values["post_disarm"]
    assert "上鎖後 4 ms" in f.text_zh and "飛行中觸發" not in f.text_zh, f.text_zh
    h = hs["H_FAILSAFE_ACTION"]
    assert h.raw == 0.0 and h.status == "ruled_out" and h.ruled_out_basis, (h.raw, h.status)
    # 上鎖前 1 ms(仍在飛行中)、晚於 t_loss → HIT,但 engine 標後果,不計分
    ctx1 = synth(arm=((10.0, 200.0, 12),), mode=[(0.0, 5, 26), (199.999, 9, 19)])
    rep1, f1, hs1 = via_engine(ctx1, **kw)
    assert f1.status == HIT and near(f1.t_onset, 199.999, 1e-6) and f1.values["phase"] == "consequence"
    assert f1.values["triggers"][0]["mode_reason"] == "CRASH_FAILSAFE" and f1.tag == "consequence", f1.tag
    h1 = hs1["H_FAILSAFE_ACTION"]
    assert h1.raw == 0.0 and h1.score == 0.0 and h1.status == "ruled_out" and fs_contrib(hs1, "H_FAILSAFE_ACTION") == 0.0
    assert any("後果" in b for b in h1.ruled_out_basis), h1.ruled_out_basis
    # 對照:同一個 Rsn 19 若早於 t_warn(precursor)就會計分
    ctx2 = synth(mode=[(0.0, 5, 26), (150.0, 9, 19)])
    rep2, f2, hs2 = via_engine(ctx2, t_warn=170.0, t_loss=175.0, end_type="crash_check")
    h2 = hs2["H_FAILSAFE_ACTION"]
    assert f2.tag == "precursor" and near(h2.raw, 4.0, 1e-9) and h2.gate_ok, (f2.tag, h2.raw)
    print("  上鎖後 4 ms 的模式列 → post_disarm;上鎖前、晚於 t_loss → 後果 0 分已排除;早於 t_warn → 計 4 分")


def test_inflight_window_ends_at_disarm_row():
    """飛行中的右邊界就是上鎖列:剛好在上鎖列 → 飛行中;晚 5 ms → 上鎖後(文字不得寫「飛行中」)。"""
    ctx = synth(err=[(200.0, 5, 1)])
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == HIT and near(f.t_onset, 200.0, 1e-9) and f.values["post_disarm"] == []
    ctx2 = synth(err=[(200.005, 5, 1)])
    f2 = pick(df.run(ctx2, anchors(ctx2)))
    assert f2.status == CLEAR and f2.values["triggers"] == [], statuses([f2])
    assert [x["t"] for x in f2.values["post_disarm"]] == [200.005] and "上鎖後 5 ms" in f2.text_zh, f2.text_zh
    assert f2.values["first_err_in_interval"] is None                 # 上鎖後的 ERR 也不算「解鎖期間第一筆」
    print("  上鎖列當下 = 飛行中;上鎖後 5 ms = post_disarm")


def test_synth_mode_refused_target_name_from_vehicle_table():
    # ERR 10 的 ECode = 想切換的模式編號;目標模式從沒出現在 MODE 表 → 用機型模式表取名字(16 = POSHOLD)
    ctx = synth(err=[(80.0, 10, 16)])
    ctx.ix.mav_type = 2                                             # MAV_TYPE_QUADROTOR
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == SOFT and "POSHOLD" in f.values["triggers"][0]["label"], f.values["triggers"]
    ctx2 = synth(err=[(80.0, 10, 16)])                              # 認不出機型 → 只寫編號
    f2 = pick(df.run(ctx2, anchors(ctx2)))
    assert "模式 16" in f2.values["triggers"][0]["label"]
    print("  ERR 10 目標模式名稱:機型模式表(POSHOLD);認不出機型時寫編號")


def test_synth_events_while_landed_are_not_evidence():
    # 起飛(EV28)在 15 s;12 s 的 GCS 失效保護在地面(解鎖未離地);190 s 降落(EV18)後的 EKF 失效保護也不算
    ctx = synth(err=[(12.0, 8, 1), (13.0, 8, 0), (192.0, 16, 2), (192.0, 17, 1)], land=190.0)
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == CLEAR, statuses([f])
    le = f.values["landed_events"]
    assert [round(x["t"]) for x in le] == [12, 192] and all(x["state"] == "landed" for x in le), le
    assert f.values["triggers"] == [] and "落地期間有 2 筆" in f.text_zh
    print("  解鎖未離地 / 降落後的事件不算飛行中證據(放 landed_events)")


def test_synth_event_after_disarm_and_in_gap_are_not_evidence():
    ctx = synth(arm=((10.0, 100.0, 0), (200.0, 260.0, 0)), takeoff=15.0,
                err=[(104.0, 8, 1),            # 第一次上鎖後 4 s → post_disarm
                     (180.0, 16, 2), (180.0, 17, 1),     # 兩次飛行之間(上鎖 60 s 之後:地面)
                     (230.0, 5, 1)])           # 第二次飛行中:分析第一次時不能算進去
    ctx_ = ctx
    a = Anchors(intervals=list(ctx_.armed))
    a.interval = ctx_.armed[0]
    a.t_disarm = a.interval.t1
    a.end_type = "crash_check"
    f = pick(df.run(ctx_, a))
    assert f.status == CLEAR
    assert [round(x["t"]) for x in f.values["post_disarm"]] == [104] and f.values["post_disarm"][0]["subsys"] == [8]
    assert [round(x["t"]) for x in f.values["ground_events"]] == [180]
    assert "是後果" in f.text_zh
    # 分析第二次飛行:230 s 的遙控失效保護是證據(takeoff=15 s 只寫了一次 EV28,第二次飛行沒有離地紀錄 → 狀態未知 → SOFT)
    a2 = Anchors(intervals=list(ctx_.armed))
    a2.interval = ctx_.armed[1]
    f2 = pick(df.run(ctx_, a2))
    assert f2.status == SOFT and f2.context == "landed_state_unknown", statuses([f2])
    print("  上鎖後 → post_disarm、兩次飛行之間 → ground、其他飛行的事件不算")


def test_synth_emergency_stop_ev54():
    ctx = synth(ev=[(120.0, 54), (121.5, 55), (13.0, 54), (14.0, 55)])          # 13 s 在地面(未離地)→ 不算
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == HIT and near(f.t_onset, 120.0, 1e-6)
    tr = f.values["triggers"]
    assert len(tr) == 1 and tr[0]["kind"] == "ev" and near(tr[0]["duration"], 1.5, 1e-6) and "EV54" in tr[0]["label"], tr
    assert len(f.values["landed_events"]) == 1
    print("  EV54 馬達緊急停止(飛行中)HIT;地面的不算")


def test_synth_passive_mode_change_and_refused_are_soft():
    # Rsn 14 AVOIDANCE:被動切換 → SOFT;Rsn 8 MISSION_END / 12 BRAKE_TIMEOUT 是正常自動切換;Rsn 1 飛手指令
    ctx = synth(mode=[(0.0, 5, 26), (30.0, 2, 1), (40.0, 5, 12), (50.0, 3, 8), (70.0, 6, 14)])
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == SOFT and f.severity == "medium", statuses([f])
    assert len(f.values["triggers"]) == 1 and f.values["triggers"][0]["mode_reason"] == "AVOIDANCE" and f.values["n_hit"] == 0
    assert near(f.t_onset, 70.0, 1e-6)
    assert [m["cls"] for m in f.values["mode_changes"]] == ["pilot", "benign", "benign", "passive"]
    assert "證據偏弱" in f.text_zh
    # ERR 10(想切到 ALT_HOLD 被拒)
    ctx2 = synth(err=[(80.0, 10, 2)], mode=[(0.0, 5, 26), (30.0, 2, 1), (31.0, 5, 1)])
    f2 = pick(df.run(ctx2, anchors(ctx2)))
    assert f2.status == SOFT and "被拒" in f2.values["triggers"][0]["label"] and "ALT_HOLD" in f2.values["triggers"][0]["label"]
    # 只有飛手 / 一般自動切換 → CLEAR
    ctx3 = synth(mode=[(0.0, 5, 26), (30.0, 2, 1), (40.0, 5, 12), (50.0, 3, 8)])
    f3 = pick(df.run(ctx3, anchors(ctx3)))
    assert f3.status == CLEAR and len(f3.values["mode_changes"]) == 3 and "3 次切換" in f3.text_zh
    print("  被動切換模式 / 切換被拒 = SOFT;飛手指令與一般自動切換 = CLEAR")


def test_synth_msg_only_failsafe_is_soft_and_msg_merges_otherwise():
    ctx = synth(msg=[(90.0, "Radio Failsafe"), (5.0, "PreArm: Radio failsafe on"), (95.0, "Radio Failsafe Cleared"),
                     (20.0, "LUA: Script1 Delayed Failsafe initialized")])
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == SOFT and len(f.values["triggers"]) == 1 and f.values["triggers"][0]["kind"] == "msg", f.values["triggers"]
    assert near(f.t_onset, 90.0, 1e-6)
    # 有 ERR 時 MSG 只併進去,不另算一筆
    ctx2 = synth(err=[(90.0, 5, 1)], msg=[(90.0, "Radio Failsafe")])
    f2 = pick(df.run(ctx2, anchors(ctx2)))
    assert f2.status == HIT and len(f2.values["triggers"]) == 1 and f2.values["triggers"][0]["msg"] == "Radio Failsafe"
    print("  MSG-only failsafe = SOFT(PreArm / Cleared / initialized 不算);有 ERR 時併入同一筆")


def test_msg_merge_rules_and_bytes_messages():
    """MSG 配對規則(searchsorted 版與原本語意相同):±2 s 內有 ERR / 模式觸發 → 併入最近的一筆;連續的 failsafe
    訊息併成一筆;ERR ±0.3 s 內的伴隨訊息附原文;MSG 欄是 bytes 也要先解碼(PreArm 不得變成 SOFT)。"""
    ctx = synth(err=[(100.0, 16, 2), (100.0, 17, 1)],
                msg=[(100.2, "EKF variance"), (101.5, "EKF Failsafe"), (130.0, "Radio Failsafe"), (131.0, "Radio Failsafe"),
                     (133.5, "Radio Failsafe")])
    f = pick(df.run(ctx, anchors(ctx)))
    tr = f.values["triggers"]
    assert [(x["t"], x["kind"]) for x in tr] == [(100.0, "err"), (130.0, "msg"), (133.5, "msg")], tr
    assert tr[0]["msg"] == "EKF variance"                     # 伴隨訊息(±0.3 s);101.5 的 failsafe 訊息已併入,不另算
    # bytes 欄位(S 型別):PreArm → 不算;一般 failsafe 訊息解碼後才寫進文字(不能出現 b'…')
    ctx2 = synth()
    ctx2._tables[("MSG", None)] = {"t": np.array([90.0, 150.0]),
                                   "Message": np.array([b"PreArm: Radio failsafe on", b"Radio Failsafe"])}
    f2 = pick(df.run(ctx2, anchors(ctx2)))
    assert f2.status == SOFT and len(f2.values["triggers"]) == 1 and near(f2.t_onset, 150.0, 1e-9), f2.values["triggers"]
    assert f2.values["triggers"][0]["msg"] == "Radio Failsafe" and "b'" not in f2.text_zh, f2.text_zh
    ctx3 = synth()
    ctx3._tables[("MSG", None)] = {"t": np.array([190.0]), "Message": np.array([b"PreArm: Radio failsafe on"])}
    assert pick(df.run(ctx3, anchors(ctx3))).status == CLEAR
    print("  MSG 併入最近的觸發、連續訊息併一筆、伴隨訊息附原文;bytes 訊息先解碼(PreArm 不誤報)")


def test_many_rows_stay_fast():
    """上萬筆 ERR + MSG(病態 log)仍 < 1.5 s(審查重現:舊版巢狀迴圈 20,000 + 20,000 筆要 14 s);未解鎖 log 也一樣。"""
    n = 20000
    tt = np.linspace(20.0, 190.0, n)
    err = [(float(x), 8, 1 if i % 2 == 0 else 0) for i, x in enumerate(tt)]
    msg = [(float(x) + 0.05, "GCS Failsafe" if i % 2 == 0 else "Some failsafe text") for i, x in enumerate(tt)]
    for armed in (True, False):
        ctx = synth(arm=((10.0, 200.0, 0),) if armed else (), err=err, msg=msg, takeoff=15.0 if armed else None)
        a = anchors(ctx) if armed else basic_anchors(ctx)
        t0 = time.perf_counter()
        fs = df.run(ctx, a)
        dt = time.perf_counter() - t0
        check_contract(fs, f"perf armed={armed}")
        f = pick(fs)
        assert f.status == (HIT if armed else NA) and dt < 1.5, (armed, f.status, dt)
        if armed:
            assert f.values["n_triggers"] == n // 2 and len(f.values["triggers"]) == df.MAX_LISTED
            assert all(x["kind"] == "err" for x in f.values["triggers"])     # 每則 MSG 都在 ERR ±2 s 內 → 不另算
        print(f"  ERR {n} + MSG {n} 筆、armed={armed}:{f.status} {dt:.2f} s")


def test_synth_landed_state_unknown_downgrades_to_soft():
    ctx = synth(err=[(100.0, 6, 1)], takeoff=None)               # log 沒有 EV28
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == SOFT and f.context == "landed_state_unknown", statuses([f])
    assert f.values["landed_state_known"] is False and f.values["triggers"][0]["state"] == "unknown"
    assert "降為 SOFT" in f.text_zh
    # 沒有事件、狀態未知:整段解鎖時間仍可判定 CLEAR
    ctx2 = synth(takeoff=None)
    f2 = pick(df.run(ctx2, anchors(ctx2)))
    assert f2.status == CLEAR and f2.values["landed_state_known"] is False and near(f2.values["airborne_s"], 190.0, 0.1) \
        and f2.values["airborne_basis"] == "armed_time"
    assert "解鎖約 190 s,無離地紀錄" in f2.text_zh and "離地約" not in f2.text_zh, f2.text_zh   # 不能把解鎖時間說成離地時間
    print("  沒有 NOT_LANDED 事件 → 事件降為 SOFT(context landed_state_unknown)")


def test_synth_no_flight_is_na_not_clear():
    ctx = synth(arm=((10.0, 100.0, 0),), takeoff=98.0)           # 解鎖 90 s 但只離地 2 s
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == NA and f.context == "no_flight" and f.values["reason"] == "no_flight", statuses([f])
    assert f.values["airborne_basis"] == "ev28" and near(f.values["airborne_s"], 2.0, 0.01)
    print("  解鎖但只離地 2 s → NA no_flight(不是 CLEAR)")


def test_synth_no_ev28_uses_throttle_output_to_decide_flight():
    tt = np.arange(0.0, 300.0, 0.1)
    # 沒有 EV28、油門輸出一直是 0(桌上解鎖空轉)→ NA no_flight;期間的失效保護事件是「落地期間」資訊
    ctx = synth(arm=((10.0, 100.0, 0),), takeoff=None, tho=(tt, np.zeros(tt.size)), err=[(50.0, 8, 1)])
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == NA and f.context == "no_flight" and f.values["airborne_basis"] == "tho", statuses([f])
    assert len(f.values["landed_events"]) == 1 and f.values["triggers"] == []
    # 沒有 EV28、油門輸出顯示有飛(ThO 0.15 持續 80 s)→ 離地秒數 ≈ 80;沒有事件 → CLEAR,有事件 → SOFT(狀態未知)
    th = np.where((tt >= 20.0) & (tt < 100.0), 0.15, 0.0)
    ctx2 = synth(arm=((10.0, 100.0, 0),), takeoff=None, tho=(tt, th))
    f2 = pick(df.run(ctx2, anchors(ctx2)))
    assert f2.status == CLEAR and f2.values["airborne_basis"] == "tho" and near(f2.values["airborne_s"], 80.0, 0.5), f2.values["airborne_s"]
    ctx3 = synth(arm=((10.0, 100.0, 0),), takeoff=None, tho=(tt, th), err=[(60.0, 8, 1)])
    f3 = pick(df.run(ctx3, anchors(ctx3)))
    assert f3.status == SOFT and f3.context == "landed_state_unknown"
    # 連 CTUN 都沒有:用整段解鎖時間(armed_time)
    ctx4 = synth(arm=((10.0, 100.0, 0),), takeoff=None)
    f4 = pick(df.run(ctx4, anchors(ctx4)))
    assert f4.status == CLEAR and f4.values["airborne_basis"] == "armed_time" and near(f4.values["airborne_s"], 90.0, 0.1)
    print("  沒有 EV28:用 CTUN.ThO 判斷有沒有離地(沒離地 → NA;有 → 事件降為 SOFT);沒有 CTUN → 整段解鎖時間")


def test_synth_no_event_tables_is_na():
    ctx = synth()
    ctx._tables[("MODE", None)] = None
    ctx._tables[("EV", None)] = None
    ctx._tables[("ERR", None)] = None
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == NA and f.context == "no_data"
    # ERR 表不存在但 MODE / EV 存在 = 沒有錯誤事件(正常 log 常見)→ CLEAR,不是 NA
    ctx2 = synth()
    assert ctx2.table("ERR") is None
    assert pick(df.run(ctx2, anchors(ctx2))).status == CLEAR
    print("  MODE / EV / ERR 全缺 → NA;只缺 ERR(沒有任何錯誤)→ CLEAR")


def test_synth_not_armed_ground_events_listed_as_info():
    ctx = synth(arm=(), err=[(50.0, 8, 1), (60.0, 8, 0), (70.0, 11, 2)], takeoff=None)
    a = basic_anchors(ctx)
    assert a.interval is None
    f = pick(df.run(ctx, a))
    assert f.status == NA and f.context == "not_armed" and f.values["n_ground_events"] == 2
    assert f.t_onset is None and "未解鎖" in f.text_zh
    print("  未解鎖:NA not_armed;地面事件放 ground_events")


def test_uint8_codes_and_time_base():
    """ERR / EV / MODE 都是 uint8:ECode 0 與 255、Subsys 31 不能回繞;絕對時間 → 相對時間。"""
    ctx = synth(err=[(40.0, 31, 255), (41.0, 31, 0)])
    assert ctx.table("ERR")["ECode"].dtype == np.float64                      # Ctx 已轉型
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == HIT and f.values["triggers"][0]["subsys"] == [31] and f.values["triggers"][0]["ecode"] == {"31": 255}
    assert near(f.t_onset, 40.0, 1e-6) and near(f.values["triggers"][0]["duration"], 1.0, 1e-6)
    print("  uint8 欄位、相對時間轉換正確")


def test_pure_helpers():
    # err_episodes:EKF 家族合併、觸發 / 解除成對、孤立的解除列忽略、重複觸發不拆段
    t = [1, 2, 3, 4, 5, 6, 7, 8]
    sub = [16, 17, 16, 17, 11, 25, 25, 25]
    code = [2, 1, 0, 0, 0, 1, 1, 0]
    eps = df.err_episodes(t, sub, code, subsys=set(df.FAILSAFE_SUBSYS), family=df.ERR_FAMILY)
    assert len(eps) == 2, eps
    assert eps[0]["t"] == 1.0 and eps[0]["t_end"] == 4.0 and eps[0]["subsys"] == [16, 17]
    assert eps[1]["t"] == 6.0 and eps[1]["t_end"] == 8.0 and eps[1]["n"] == 2 and eps[1]["subsys"] == [25]
    eps2 = df.err_episodes([1, 2], [8, 8], [1, 1])
    assert len(eps2) == 1 and eps2[0]["t_end"] is None
    # mode_changes:第一列(開機)與同模式重記不算
    mc = df.mode_changes([0, 1, 2, 3], [0, 5, 5, 6], [26, 1, 1, 3], ["STABILIZE", "LOITER", "LOITER", "RTL"])
    assert [(m["t"], m["from"], m["to"], m["rsn"]) for m in mc] == [(1.0, "STABILIZE", "LOITER", 1), (3.0, "LOITER", "RTL", 3)]
    assert df.mode_changes([0, 1], [0, 5], None, ["A", "B"])[0]["rsn"] is None
    # reason_class
    assert [df.reason_class(r) for r in (3, 4, 5, 6, 7, 10, 11, 19, 25, 29, 50, 51)] == ["failsafe"] * 12
    assert [df.reason_class(r) for r in (1, 2, 9, 32, 42, 52, 53)] == ["pilot"] * 7
    assert [df.reason_class(r) for r in (8, 12, 13, 16, 18, 26, 31, 41, 43, 45, 48)] == ["benign"] * 11
    assert [df.reason_class(r) for r in (14, 15, 17, 33, 34)] == ["passive"] * 5
    assert [df.reason_class(r) for r in (0, 56, 99, 255)] == ["unexplained"] * 4
    assert df.reason_class(None) == "unrecorded"
    assert df.MODE_REASON_NAMES[9] == "THROTTLE_LAND_ESCAPE" and df.MODE_REASON_NAMES[25] == "FAILSAFE" \
        and df.MODE_REASON_NAMES[32] == "SCRIPTING" and df.MODE_REASON_NAMES[50] == "DEADRECKON_FAILSAFE" \
        and df.MODE_REASON_NAMES[53] == "AUX_FUNCTION" and len(df.MODE_REASON_NAMES) == 56
    # EV54 / 55
    st = df.ev_stop_episodes([1, 2, 3, 4], [54, 55, 54, 10])
    assert st == [{"t": 1.0, "t_end": 2.0}, {"t": 3.0, "t_end": None}]
    # msg_failsafes
    m = df.msg_failsafes([1, 2, 3, 4, 5], ["GCS Failsafe", "PreArm: GCS failsafe on", "Failsafe cleared", "Radio Failsafe", "EKF variance"])
    assert [x["text"] for x in m] == ["GCS Failsafe", "Radio Failsafe"]
    # bytes(S 型別欄)先解碼:b'PreArm…' 不能因為前面多了 b' 而躲過 ^PreArm 排除
    mb = df.msg_failsafes([1, 2, 3], [b"PreArm: Radio failsafe on", np.bytes_(b"Radio Failsafe\x00\x00"), b"GCS Failsafe"])
    assert [x["text"] for x in mb] == ["Radio Failsafe", "GCS Failsafe"], mb
    print("  err_episodes / mode_changes / reason_class / ev_stop_episodes / msg_failsafes 邏輯正確")


def _drop_rsn(ctx, rows):
    """把 MODE 表換成沒有 Rsn 欄的舊格式(rows = [(t_rel, ModeNum)];Ctx 的表已是相對秒)。"""
    ctx._tables[("MODE", None)] = {"t": np.array([float(r[0]) for r in rows]),
                                   "ModeNum": np.array([float(r[1]) for r in rows])}


def test_rsn_column_missing_mode_change_is_soft_not_clear():
    """MODE 表沒有 Rsn 欄:飛行中的模式切換原因無法判斷 → SOFT「原因未記錄」(不能 CLEAR、不能寫「皆為飛手指令」);
    ERR 失效保護 1 s 內的切換併入該 ERR;沒有切換才 CLEAR(「無切換」仍是真的)。"""
    ctx = synth(mode=[(0.0, 5, 26), (50.0, 6, 3)])
    _drop_rsn(ctx, [(0.0, 5), (50.0, 6)])
    f = pick(df.run(ctx, anchors(ctx, t_warn=90.0, t_loss=92.0)))
    assert f.status == SOFT and f.values["rsn_available"] is False and f.values["mode_check"] == "no_rsn", statuses([f])
    assert f.values["mode_changes"][0]["cls"] == "unrecorded" and f.values["triggers"][0]["mode_cls"] == "unrecorded"
    assert "原因未記錄的模式切換 LOITER→RTL" in f.text_zh and "皆為飛手" not in f.text_zh, f.text_zh
    assert any("Rsn" in c for c in f.values["caveats"])
    # ERR 5 + 0.05 s 後的模式列:併成一筆 HIT(依時間配對)
    ctx2 = synth(err=[(50.0, 5, 1)], mode=[(0.0, 5, 26), (50.05, 6, 3)])
    _drop_rsn(ctx2, [(0.0, 5), (50.05, 6)])
    f2 = pick(df.run(ctx2, anchors(ctx2)))
    tr = f2.values["triggers"]
    assert f2.status == HIT and len(tr) == 1 and tr[0]["kind"] == "err" and tr[0]["mode_to"] == "RTL" \
        and "mode_reason_note" in tr[0] and "rsn" not in tr[0], tr
    # 沒有切換:CLEAR,「無切換」屬實
    ctx3 = synth()
    _drop_rsn(ctx3, [(0.0, 5)])
    f3 = pick(df.run(ctx3, anchors(ctx3)))
    assert f3.status == CLEAR and "模式全程 LOITER 無切換" in f3.text_zh and f3.values["mode_check"] == "no_rsn"
    print("  沒有 Rsn 欄:飛行中切換 = SOFT(原因未記錄);ERR 旁的切換併入 ERR;沒有切換才 CLEAR")


def test_mode_table_missing_clear_does_not_claim_no_change():
    """MODE 表缺失但 ERR / EV 在:ERR / EV 的檢查仍有效(CLEAR),但文字不得說「無切換」,要寫無法檢查並降信心。"""
    ctx = synth()
    ctx._tables[("MODE", None)] = None
    f = pick(df.run(ctx, anchors(ctx)))
    assert f.status == CLEAR and f.values["mode_check"] == "no_mode_table" and f.values["confidence"] == "low"
    assert "MODE 表缺失,無法檢查模式切換" in f.text_zh and "無切換" not in f.text_zh and "模式切換或" not in f.text_zh, f.text_zh
    assert any("MODE 表缺失" in c for c in f.values["caveats"])
    # 有 MODE 表時 CLEAR 的信心是 medium
    assert pick(df.run(synth(), anchors(synth()))).values["confidence"] == "medium"
    print("  MODE 表缺失:CLEAR(只檢查 ERR / EV / MSG)、文字寫「無法檢查模式切換」、信心 low")


# ======================================================================================
# 5. 經 engine 的整合結果(驗收 A3 / A7 / C:H_FAILSAFE_ACTION 已排除、地面事件不進排名)
# ======================================================================================

def test_engine_124_failsafe_ruled_out_with_numbers():
    from core.crash import engine
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    rep = engine.analyze(ctx.ld)
    f = rep.finding(OWNED[0])
    assert f is not None and f.status == CLEAR and f.values["triggers"] == [], (f and f.status)
    hs = {h.id: h for h in rep.hypotheses}["H_FAILSAFE_ACTION"]
    assert hs.score == 0.0 and hs.status == "ruled_out", (hs.score, hs.status)
    basis = " ".join(hs.ruled_out_basis)
    assert "LOITER" in basis and "1258.94" in basis and "1266.24" in basis, basis      # 已排除列帶數字
    # 失效保護類事件在時間軸上都不是 precursor(1258.94 glitch = 後果、1266.24 EKF = 上鎖後)
    for e in rep.timeline:
        if e.category == "err" and 1250.0 <= e.t <= 1270.0:
            assert e.tag != "precursor", (e.t, e.label, e.tag)
    print(f"  engine:H_FAILSAFE_ACTION 0 分、已排除;依據:{basis[:60]}…")


def test_engine_baselines_ground_failsafes_not_ranked():
    from core.crash import engine
    n_fs = {}
    first_t = {}
    for name in ("00000117", "00000123"):            # 117 FAILSAFE_GCS 611.09;123 七次 EKFCHECK
        ctx = crashkit.load_ctx(name)
        rep = engine.analyze(ctx.ld)
        assert rep.end_type == "not_armed" and rep.verdict == "no_failure", (name, rep.end_type, rep.verdict)
        assert all(h.score == 0.0 for h in rep.hypotheses), (name, [(h.id, h.score) for h in rep.hypotheses])
        assert rep.top() is None
        f = rep.finding(OWNED[0])
        assert f is None or f.status == NA, (name, f.status)
        fs_ev = [e for e in rep.ground_events
                 if e.category == "err" and ("FAILSAFE" in e.label or "EKFCHECK" in e.label)]
        assert fs_ev and all(e.tag == "ground_info" for e in fs_ev), (name, [(e.t, e.label, e.tag) for e in fs_ev])
        n_fs[name] = len(fs_ev)
        first_t[name] = fs_ev[0].t
    assert near(first_t["00000117"], 611.09, 0.05) and near(first_t["00000123"], 155.2, 0.1), first_t
    print(f"  117 / 123 經 engine:not_armed、no_failure、沒有排名;失效保護 ERR 列只在 ground_events(ground_info){n_fs}")


# ======================================================================================
# 6. 契約:每個 id 至少一筆 / 例外 → NA / 決定性
# ======================================================================================

def test_every_id_in_every_scenario_and_guard():
    ctx124 = crashkit.load_ctx(crashkit.CRASH_LOG)
    scenarios = {
        "124 truth": df.run(ctx124, crashkit.truth_anchors_124(ctx124)),
        "124 bare": df.run(ctx124, basic_anchors(ctx124)),
        "124 f1": df.run(ctx124, crashkit.flight1_anchors_124(ctx124)),
        "empty anchors": df.run(ctx124, Anchors()),
    }
    for n in crashkit.BASELINES:
        c = crashkit.load_ctx(n)
        scenarios[n] = df.run(c, crashkit.not_armed_anchors(c))
    # 合成情境:HIT / SOFT / CLEAR / NA(no_flight、not_armed、no_data)各分支都要恰一筆且守契約
    synth_cases = {
        "synth hit": synth(err=[(100.0, 6, 1)]),
        "synth soft": synth(mode=[(0.0, 5, 26), (70.0, 6, 14)]),
        "synth clear": synth(),
        "synth no_flight": synth(arm=((10.0, 100.0, 0),), takeoff=98.0),
        "synth not_armed": synth(arm=(), err=[(50.0, 8, 1)], takeoff=None),
        "synth unknown": synth(err=[(100.0, 6, 1)], takeoff=None),
    }
    for label, c in synth_cases.items():
        scenarios[label] = df.run(c, anchors(c, t_warn=150.0, t_loss=155.0) if c.armed else basic_anchors(c))
    got = {label: fs[0].status for label, fs in scenarios.items() if label.startswith("synth")}
    assert got == {"synth hit": HIT, "synth soft": SOFT, "synth clear": CLEAR, "synth no_flight": NA,
                   "synth not_armed": NA, "synth unknown": SOFT}, got
    for label, fs in scenarios.items():
        check_contract(fs, label)
        assert len([f for f in fs if f.id == OWNED[0]]) == 1                  # 一個 id 只回一筆(評分時多筆會重複計權重)
    # 例外 → guarded 轉 NA(context 'error')
    bad = df.run(None, Anchors())
    assert len(bad) == 1 and bad[0].status == NA and bad[0].context == "error" and "error" in bad[0].values
    print("  每個情境都有 inflight_failsafe_or_mode_change 恰一筆;例外被包成 NA error")


def test_deterministic_and_json():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.truth_anchors_124(ctx)
    r1 = [(f.id, f.status, f.context, f.t_onset, json.dumps(f.values, sort_keys=True), f.text_zh) for f in df.run(ctx, a)]
    r2 = [(f.id, f.status, f.context, f.t_onset, json.dumps(f.values, sort_keys=True), f.text_zh) for f in df.run(ctx, a)]
    assert r1 == r2
    print("  決定性:同輸入同輸出;values 可 JSON 序列化")


def test_performance_under_1_5s():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.truth_anchors_124(ctx)
    df.run(ctx, a)                      # 暖機(表快取)
    t0 = time.perf_counter()
    for _ in range(5):
        df.run(ctx, a)
    dt = (time.perf_counter() - t0) / 5
    assert dt < 1.5, dt
    print(f"  每次 run {dt * 1000:.1f} ms(< 1500 ms)")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
