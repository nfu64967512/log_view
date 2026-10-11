"""power 偵測模組驗證(core/crash/det_power.py):python tests/test_crash_power.py 或 pytest。

涵蓋:
  1. 00000124 飛行 2(失事)× truth_anchors_124 / 裸 basic_anchors:低電壓 / 崩落 / 電源軌 CLEAR(附實測數字),
     電流計無效 / 保護參數 HIT(一般健檢);
  2. 00000124 飛行 1(正常降落對照)與飛行 2 的 1255 s 之前:電源三項 CLEAR,沒有誤報(驗收 B);一般健檢兩項是真的
     持續性問題 → HIT 但 context='standing'、不在任何假說權重裡(驗收 B「power_* 全 CLEAR」的字面偏離,見 test 說明);
     engine 層:H_POWER 0 分已排除、排除依據帶數字與「電流計無效」保留、報告不談內阻 / 剩餘電量(驗收 A7 / A8);
  3. 六個基準 log(從未解鎖)→ 五個 id 全部 NA('not_armed');117/122/123 註明 USB 供電、電池分析不適用,
     118/120 註明尾端是拔電池不是電壓崩落(驗收 C);
  4. 合成資料(真實資料只有一次失事,合成才能證明偵測器沒有寫死 124 的數字):12S / 6S、低電壓、BATT_CRT_VOLT 計時、
     ERR 6、飛行中崩落、log 尾端拔電池(飛行中 / 落地後 / 解鎖結束後)、資料空洞、USB 供電、電源軌各條件、
     電流計品質、保護參數稽核、串數備援、偵測器例外保護;
     審查回歸:串數估計不多估(LiHV 4S/6S 健康飛行 CLEAR、估計門檻最多 SOFT)、崩落要持續(壞樣本 / 上跳回落不是 HIT)、
     t_loss 之後的條件寫成後果、解鎖中地面崩落但 log 繼續記錄不是「拔電池」CLEAR、解鎖前就累計的過流旗標只給 SOFT、
     文字不印 nan;
  5. 真實 124 資料「注入故障」:改 BAT / POWR 表(低電壓 / 崩落 / 壞樣本 / 翻滾後掉電 / 解鎖前過流 / 參數歸零)後
     偵測器的結論與 engine 的因果標籤要對 → 偵測邏輯確實走真實資料結構;
  6. 每個 id 在每種情境都有輸出、values 可 JSON 序列化、文字為繁體中文且不含 nan、每個偵測器 < 1.5 s。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import crashkit  # noqa: E402
from core.crash import det_power  # noqa: E402
from core.crash.anchors import basic_anchors  # noqa: E402
from core.crash.types import CLEAR, HIT, NA, SEVERITIES, SOFT, STATUSES, Anchors, ArmedInterval  # noqa: E402

IDS = ["power_low_voltage", "power_voltage_collapse", "power_rail_fault", "power_data_quality", "power_failsafe_config_audit"]
CRASH_IDS = IDS[:3]
HEALTH_IDS = IDS[3:]


def check(cond, msg):
    assert cond, msg


def by_id(fs):
    return {f.id: f for f in fs}


def near(x, ref, tol, what=""):
    check(x is not None and abs(x - ref) <= tol, f"{what}: {x} 與 {ref} 差超過 ±{tol}")


# ---------------------------------------------------------------------------
# 合成 Ctx:只提供偵測器用到的介面(table / param / err_rows / ev_times / landed_at / armed / t_end)
# ---------------------------------------------------------------------------
class FakeCtx:
    def __init__(self, tables, params, armed, t_end, err=(), ev=()):
        self._tables = tables
        self.params = dict(params)
        self.armed = armed
        self.t_end = t_end
        self.t_base = 0.0
        self._err = list(err)           # (t, subsys, ecode)
        self._ev = sorted(ev)           # (t, id)

    def table(self, name, instance=None):
        return self._tables.get(name)

    def param(self, name, default=None):
        return self.params.get(name, default)

    def err_rows(self):
        if not self._err:
            return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        e = sorted(self._err)
        return (np.array([x[0] for x in e], float), np.array([x[1] for x in e], int), np.array([x[2] for x in e], int))

    def ev_times(self, ev_id, a=None, b=None):
        return [t for t, i in self._ev if i == ev_id and (a is None or t >= a) and (b is None or t <= b)]

    def landed_at(self, t, iv=None):
        state = True
        for tt, i in self._ev:
            if tt > t:
                break
            if i == 28:
                state = False
            elif i == 18:
                state = True
        return state

    def armed_at(self, t):
        return next((iv for iv in self.armed if iv.t0 <= t <= iv.t1), None)


class BoomCtx(FakeCtx):
    """table() 一律丟例外:驗證 @guarded 把例外降級為 NA。"""
    def table(self, name, instance=None):
        raise RuntimeError("boom")

    def param(self, name, default=None):
        raise RuntimeError("boom")


def build(volt_fn=None, *, cells=12, t_total=160.0, arm=(20.0, 140.0), rate=10.0, gaps=(), params=None,
          curr_fn=None, thr_fn=None, tho_fn=None, vcc_fn=None, vservo_fn=None, flags_fn=None, acc_fn=None,
          mcu_fn=None, err=(), ev=(), open_end=False, usb=False, no_ctun=False, no_params=False):
    """合成一次飛行:rest 4.15 V/cell → 起飛下陷 ~3 V(12S)→ 緩降;可替換各通道。回傳 (ctx, anchors)。"""
    t = np.arange(0.0, t_total, 1.0 / rate)
    for g0, g1 in gaps:
        t = t[~((t >= g0) & (t < g1))]
    t0, t1 = arm
    if open_end:
        t1 = float(t[-1])
    scale = cells / 12.0
    rest = 4.15 * cells

    def default_volt(tt):
        up = np.clip((tt - t0) / 1.5, 0.0, 1.0)
        v = rest - 3.0 * scale * up - 0.004 * np.clip(tt - t0, 0.0, None)
        return np.where(tt > t1, rest - 0.5 * scale, v)

    volt = (volt_fn or default_volt)(t).astype(float)
    if usb:
        volt = np.full(t.size, 0.05)
    in_arm = (t >= t0) & (t <= t1)
    thr = (thr_fn(t) if thr_fn else np.where(in_arm, 0.15, 0.0)).astype(float)
    tho = (tho_fn(t) if tho_fn else np.where(in_arm, 0.15, 0.0)).astype(float)
    curr = (curr_fn(t, thr) if curr_fn else np.where(in_arm, 2.0 + 150.0 * thr, 0.5)).astype(float)
    n = t.size
    flags = (flags_fn(t) if flags_fn else np.full(n, 4.0 if usb else 1.0)).astype(float)
    tables = {
        "BAT": {"t": t, "Volt": volt, "Curr": curr},
        "POWR": {"t": t, "Vcc": (vcc_fn(t) if vcc_fn else np.full(n, 4.8 if usb else 5.22)).astype(float),
                 "VServo": (vservo_fn(t) if vservo_fn else np.full(n, 5.0)).astype(float),
                 "Flags": flags, "AccFlags": (acc_fn(t) if acc_fn else np.full(n, 1.0)).astype(float),
                 "Safety": np.where(in_arm, 6.0, 1.0)},
        "MCU": {"t": t, "MVolt": (mcu_fn(t) if mcu_fn else np.full(n, 3.29)).astype(float)},
        "MOTB": {"t": t, "ThrOut": thr, "BatVolt": volt / (cells * 4.2), "LiftMax": np.ones(n)},
    }
    if not no_ctun:
        tables["CTUN"] = {"t": t, "ThO": tho, "ThH": np.full(n, 0.15)}
    p = {} if no_params else {
        "MOT_BAT_VOLT_MAX": cells * 4.2, "MOT_BAT_VOLT_MIN": cells * 3.3, "MOT_THST_HOVER": 0.15,
        "BATT_LOW_VOLT": cells * 3.6, "BATT_CRT_VOLT": cells * 3.4, "BATT_LOW_TIMER": 10.0,
        "BATT_FS_LOW_ACT": 2.0, "BATT_FS_CRT_ACT": 1.0, "BATT_LOW_MAH": 2000.0, "BATT_CRT_MAH": 1000.0,
        "BATT_ARM_VOLT": cells * 3.8, "ARMING_CHECK": 1.0, "FS_CRASH_CHECK": 1.0, "FS_EKF_ACTION": 1.0,
    }
    p.update(params or {})
    iv = ArmedInterval(0, t0, t1, None, False, open_end)
    ctx = FakeCtx(tables, p, [iv], float(t[-1]), err=err, ev=ev)
    a = Anchors(interval=iv, intervals=[iv], end_type="normal_landing", t_disarm=t1)
    return ctx, a


def run_all(ctx, a):
    fs = det_power.run(ctx, a)
    return by_id(fs)


def ramp(t, t_a, t_b, v_a, v_b):
    """t_a..t_b 線性由 v_a 變到 v_b(之外 0 權重):回傳 0..1 權重。"""
    return np.clip((t - t_a) / (t_b - t_a), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 1–3 真實 log
# ---------------------------------------------------------------------------
def test_124_crash_flight_truth_anchors():
    ctx = crashkit.load_ctx("00000124")
    a = crashkit.truth_anchors_124(ctx)
    d = by_id(det_power.run(ctx, a))
    check(list(d) == IDS, f"id 與順序: {list(d)}")

    f = d["power_low_voltage"]
    check(f.status == CLEAR and f.context == "air", f"low_voltage: {f.status}/{f.context}")
    v = f.values
    near(v["v1_min"], 44.76, 0.15, "1 秒中位數最低")
    near(v["v1_min_per_cell"], 3.73, 0.02, "V/cell")
    near(v["vinst_min"], 42.38, 0.05, "瞬間最低")
    near(v["vinst_min_t"], 1259.44, 0.15, "瞬間最低時間")
    near(v["margin_to_crit_v"], 5.16, 0.2, "對 MOT_BAT_VOLT_MIN 餘裕")
    near(v["margin_to_crt_v"], 2.76, 0.2, "對 BATT_CRT_VOLT 餘裕")
    check(v["cells"] == 12 and v["cells_src"] == "param" and v["config_inverted"] is True, v)
    check(v["min_after_loss"] is True, "瞬間最低 1259.44 在 t_loss 1258.10 之後 → 後果")
    near(v["motb_batvolt_min_v_30s"], 44.62, 0.3, "MOTB.BatVolt × 50.4")
    check(not v["crit"] and not v["high"] and v["err6_count"] == 0, v)
    check("44.8" in f.text_zh and "3.73" in f.text_zh and "後果" in f.text_zh, f.text_zh)
    # 驗收 A7「caveat BAT.Curr invalid」:run() 在電流計無效時於電池排除文字補上保留
    check(v["curr_meter_invalid"] is True and f.text_zh.endswith(det_power.CURR_CAVEAT_ZH), f.text_zh)
    check("curr_meter_invalid" not in det_power.power_low_voltage(ctx, a)[0].values, "單獨呼叫偵測器時不做跨偵測器補註")

    f = d["power_voltage_collapse"]
    check(f.status == CLEAR and f.context == "armed", f"collapse: {f.status}/{f.context}")
    v = f.values
    near(v["max_drop_1s_v"], 3.65, 0.1, "解鎖中最大 1 秒壓降(單點對 1 秒前中位數;規格單點對單點 3.65 V)")
    near(v["max_drop_persist_v"], 3.12, 0.1, "持續 ≥ 0.3 s 的最大壓降")
    check(v["transient_samples"] == 0, "真實資料沒有單點跨線的壞樣本")
    near(v["threshold_v"], 10.08, 0.05, "崩落門檻")
    check(v["margin_x"] >= 2.5, f"餘裕倍數 {v['margin_x']}")
    tail = v["log_tail"]
    check(tail is not None and tail["armed_at_tail"] is False, tail)
    near(tail["t_start"], 1289.24, 0.2, "拔電池尾端起點")
    near(tail["secs_after_disarm"], 27.1, 0.4, "尾端距上鎖")
    check(tail["v_before"] > 45 and tail["v_end"] < 25, tail)
    check("拔電池" in f.text_zh, f.text_zh)

    f = d["power_rail_fault"]
    check(f.status == CLEAR, f"rail: {f.status}")
    v = f.values
    near(v["vcc_min"], 5.182, 0.01, "Vcc 最低")
    near(v["vcc_min_t"], 1259.4, 0.15, "Vcc 最低時間")
    near(v["vservo_min"], 4.855, 0.01, "VServo 最低")
    check(3.28 < v["mcu_volt_min"] <= v["mcu_volt_max"] < 3.30, v)
    check(v["brick_fraction"] == 1.0 and v["acc_flags_max"] == 1 and v["conditions"] == [], v)
    near(v["vcc_dip_v"], -0.044, 0.02, "Vcc 相對 10 秒中位數下陷")
    check("5.18 V" in f.text_zh and f"{v['mcu_volt_min']:.3f}–{v['mcu_volt_max']:.3f}" in f.text_zh
          and "電源模組(磚)一直有效" in f.text_zh, f.text_zh)

    f = d["power_data_quality"]
    check(f.status == HIT and f.context == "standing", f"data_quality: {f.status}/{f.context}")
    v = f.values
    check(v["i_med_a"] < 0.01 and v["sag_v"] >= 1.5 and v["corr_curr_thr"] < -0.5 and v["curr_invalid"] is True, v)
    check({"BAT.Curr", "BAT.CurrTot", "BAT.RemPct", "BAT.Res"} <= set(v["do_not_use"]), v["do_not_use"])
    check(f.t_onset is not None and 873.14 <= f.t_onset <= 1262.2, f.t_onset)
    check(f.t_alarm >= f.t_onset + 19.0, (f.t_onset, f.t_alarm))

    f = d["power_failsafe_config_audit"]
    check(f.status == HIT and f.context == "standing", f"config_audit: {f.status}/{f.context}")
    v = f.values
    check(set(v["issues"]) == {"crt_ge_low", "no_failsafe_action", "capacity_failsafe_off", "batt_arm_volt_not_enforced"}, v["issues"])
    near(v["low_per_cell"], 3.08, 0.01, "LOW/cell")
    near(v["crt_per_cell"], 3.5, 0.01, "CRT/cell")
    check(v["err_subsys6_count"] == 0 and v["fs_crash_check"] == 1 and v["fs_ekf_action"] == 0, v)
    check("順序顛倒" in f.text_zh and "與失事原因無直接關係" in f.text_zh, f.text_zh)


def test_124_bare_basic_anchors_same_verdicts():
    """沒有其他模組的錨點(basic_anchors:只有解鎖區間 + CRASH_CHECK)也要得到相同結論。"""
    ctx = crashkit.load_ctx("00000124")
    a = basic_anchors(ctx)
    check(a.interval is not None and abs(a.interval.t1 - 1262.125) < 0.01, "basic_anchors 應挑到飛行 2")
    d = by_id(det_power.run(ctx, a))
    check([d[i].status for i in IDS] == [CLEAR, CLEAR, CLEAR, HIT, HIT], [d[i].status for i in IDS])
    check(d["power_low_voltage"].values["min_after_loss"] is None, "沒有 t_loss → 不下 min_after_loss 判斷")
    near(d["power_low_voltage"].values["v1_min"], 44.76, 0.15, "V1 最低")


def test_124_flight1_control():
    """飛行 1(正常降落對照):電源三項 CLEAR,沒有誤報;一般健檢兩項是資料 / 設定的真實問題,照樣 HIT。"""
    ctx = crashkit.load_ctx("00000124")
    a = crashkit.flight1_anchors_124(ctx)
    d = by_id(det_power.run(ctx, a))
    check(list(d) == IDS, list(d))
    for i in CRASH_IDS:
        check(d[i].status == CLEAR, f"{i} 在正常飛行應 CLEAR,得 {d[i].status}: {d[i].text_zh}")
    near(d["power_low_voltage"].values["v1_min"], 46.87, 0.15, "飛行 1 V1 最低")
    near(d["power_voltage_collapse"].values["max_drop_1s_v"], 3.52, 0.1, "飛行 1 最大 1 秒壓降")
    near(d["power_voltage_collapse"].values["max_drop_persist_v"], 2.97, 0.1, "飛行 1 持續壓降")
    near(d["power_rail_fault"].values["vcc_min"], 5.196, 0.02, "飛行 1 Vcc 最低")
    check(d["power_low_voltage"].values["min_after_loss"] is None, "正常飛行沒有 t_loss")
    # 尾端屬於之後的另一次飛行:值裡標註,文字不談
    tail = d["power_voltage_collapse"].values["log_tail"]
    check(tail is not None and tail["later_flight_between"] is True, tail)
    check("拔電池" not in d["power_voltage_collapse"].text_zh, d["power_voltage_collapse"].text_zh)
    # 一般健檢:BAT.Curr 在飛行 1 一樣無效、設定一樣有問題 → 是真的 HIT(不是誤報),但不屬於失事原因
    check(d["power_data_quality"].status == HIT and d["power_data_quality"].values["i_med_a"] < 0.01, d["power_data_quality"].values)
    check(d["power_failsafe_config_audit"].status == HIT, d["power_failsafe_config_audit"].status)


def test_124_flight2_before_touchdown():
    """驗收 B 最後一句:飛行 2 在 1255.0 s 之前(382 s 健康懸停 / 下降)電源三項沒有 HIT / SOFT(全 CLEAR、沒有 t_onset)。"""
    ctx = crashkit.load_ctx("00000124")
    iv = ctx.armed[1]
    sub = ArmedInterval(1, iv.t0, 1255.0, 0)
    a = Anchors(interval=sub, intervals=list(ctx.armed))
    d = by_id(det_power.run(ctx, a))
    check(list(d) == IDS, list(d))
    for i in CRASH_IDS:
        check(d[i].status == CLEAR and d[i].t_onset is None, f"{i} 在 1255 s 前應 CLEAR,得 {d[i].status}: {d[i].text_zh}")
    v = d["power_low_voltage"].values
    near(v["v1_min"], 45.15, 0.15, "1255 s 前 V1 最低(研究階段:平穩懸停 45.2 V)")
    check(v["v1_min_t"] <= 1255.0 and v["vinst_min"] > 45.0, v)
    check(v["min_after_loss"] is None and v["curr_meter_invalid"] is True, v)
    c = d["power_voltage_collapse"].values
    near(c["max_drop_1s_v"], 3.65, 0.1, "1255 s 前最大 1 秒壓降(起飛)")
    check(c["max_drop_t"] < 890.0, f"最大壓降在起飛階段: {c['max_drop_t']}")
    r = d["power_rail_fault"].values
    check(5.18 <= r["vcc_min"] < 5.25 and r["conditions"] == [], r)
    # 一般健檢:是持續性狀態(電流計真的無效、參數真的有問題),不是這段時間的「事件」
    for i in HEALTH_IDS:
        check(d[i].status == HIT and d[i].context == "standing" and sub.t0 <= d[i].t_onset <= sub.t1, f"{i}: {d[i].status}/{d[i].context}")


def test_health_items_are_standing_and_never_scored():
    """驗收 B「power_* 全 CLEAR」與 A8 的取捨:power_data_quality / power_failsafe_config_audit 是一般健檢,
    在飛行 1 照樣 HIT(電流計真的無效、參數 7 份 log 都一樣不安全 → 改成 CLEAR 會是假話)。它們帶 context='standing'、
    被 engine 列為 STANDING_IDS,且不在任何假說的權重 / 必要證據裡 → 結構上不可能被當成失事原因;三個電源偵測器是 H_POWER 的
    計分項與主要證據。H_POWER 的其他權重(規格 hypotheses.json 的 abnormal_disarm_classifier log_ended_armed、
    inflight_failsafe_or_mode_change 電池失效保護)屬於別的模組,這裡不鎖死整張權重表。"""
    from core.crash import engine
    from core.crash import hypotheses as hyp
    check(set(HEALTH_IDS) <= engine.STANDING_IDS, "一般健檢要被 engine 視為持續狀態")
    used = {w.det for h in hyp.HYPOTHESES for w in h.weights} | {p for h in hyp.HYPOTHESES for p in h.primary}
    check(not (set(HEALTH_IDS) & used), f"一般健檢不得計分: {set(HEALTH_IDS) & used}")
    hp = next(h for h in hyp.HYPOTHESES if h.id == "H_POWER")
    weighted = {w.det for w in hp.weights}
    check(set(CRASH_IDS) <= weighted, f"三個電源偵測器都要計入 H_POWER: {weighted}")
    check(set(hp.primary) == set(CRASH_IDS), f"H_POWER 的主要證據是三個電源偵測器: {hp.primary}")
    check(not (set(HEALTH_IDS) & (weighted | set(hp.primary))), "一般健檢不得進 H_POWER")
    ctx = crashkit.load_ctx("00000124")
    for a in (crashkit.flight1_anchors_124(ctx), crashkit.truth_anchors_124(ctx)):
        d = by_id(det_power.run(ctx, a))
        for i in HEALTH_IDS:
            check(d[i].context == "standing" and d[i].severity == "medium", f"{i}: {d[i].context}/{d[i].severity}")


def test_engine_report_power_rows_124():
    """驗收 A7 / A8(報告層):H_POWER 分數 0、已排除;排除依據附數字(1 秒中位數 44.8 V = 3.73 V/cell、餘裕 5.2 V、
    Vcc 5.18 V)與「電流計無效」保留;電源不在排名;報告全文不談內阻 / 剩餘電量 / 電池老化。"""
    from core.crash import engine
    ctx = crashkit.load_ctx("00000124")
    rep = engine.analyze(ctx.ld)
    h = next(h for h in rep.hypotheses if h.id == "H_POWER")
    check(h.score == 0.0 and h.status == "ruled_out", f"H_POWER: {h.score} {h.status}")
    check("H_POWER" not in [x.id for x in rep.hypotheses if x.score > 0], "電源不得進排名")
    basis = " ".join(h.ruled_out_basis)
    for s in ("44.8 V", "3.73 V/cell", "達 5.2 V", "5.18 V", "電流計無效", "拔電池"):
        check(s in basis, f"排除依據缺「{s}」: {basis}")
    for f in rep.findings:
        if f.domain != "power":
            continue
        if f.id in CRASH_IDS:
            check(f.status == CLEAR, f"{f.id}: {f.status}")
        else:
            check(f.status == HIT and f.tag == "standing", f"{f.id}: {f.status}/{f.tag}")
    txt = rep.format_text()
    check(not re.search(r"內阻|剩餘電量|電池老化|電量剩餘", txt), "報告不得描述 BAT.Curr 衍生量")
    check("BAT.Curr 讀值無效" in txt, "限制區塊要說明電流計無效")


def test_baselines_acceptance_c_power_notes():
    """驗收 C:117/122/123 是 USB 供電 → 文字「電池分析不適用」;118/120 尾端 50 → 17..21 V 是拔電池、不是電壓崩落;
    119 有電池、沒有尾端。狀態一律 NA 'not_armed'(未解鎖不做判定,只附說明)。"""
    for name in ("00000117", "00000122", "00000123"):
        ctx = crashkit.load_ctx(name)
        d = by_id(det_power.run(ctx, crashkit.not_armed_anchors(ctx)))
        for i in IDS[:4]:
            f = d[i]
            check(f.status == NA and f.context == "not_armed", f"{name} {i}: {f.status}/{f.context}")
            check(f.values.get("power_source") == "usb_or_no_battery" and f.values["bat_volt_median"] < 0.2, f"{name} {i}: {f.values}")
            check("USB" in f.text_zh and "電池分析不適用" in f.text_zh, f"{name} {i}: {f.text_zh}")
        check("log_tail" not in d["power_voltage_collapse"].values, f"{name}: USB log 不談尾端")
    for name in ("00000118", "00000120"):
        ctx = crashkit.load_ctx(name)
        f = by_id(det_power.run(ctx, crashkit.not_armed_anchors(ctx)))["power_voltage_collapse"]
        tail = f.values["log_tail"]
        check(f.status == NA and f.values["power_source"] == "battery", f"{name}: {f.status} {f.values}")
        check(tail is not None and tail["v_before"] > 49.0 and 16.5 <= tail["v_end"] <= 21.5
              and tail["tail_len_s"] <= 1.0 and tail["armed_at_tail"] is False, f"{name}: {tail}")
        check("拔電池" in f.text_zh and "不是電壓崩落" in f.text_zh, f"{name}: {f.text_zh}")
    ctx = crashkit.load_ctx("00000119")
    f = by_id(det_power.run(ctx, crashkit.not_armed_anchors(ctx)))["power_voltage_collapse"]
    check(f.values["power_source"] == "battery" and f.values["log_tail"] is None and "拔電池" not in f.text_zh, f.values)


def test_baselines_are_all_na():
    """從未解鎖的 6 個 log(桌上 / 手持 / USB / 拔電池尾端):五個 id 全 NA,context='not_armed',不得有 HIT/SOFT/CLEAR。"""
    for name in crashkit.BASELINES:
        ctx = crashkit.load_ctx(name)
        a = crashkit.not_armed_anchors(ctx)
        check(a.interval is None, f"{name} 不應有解鎖區間")
        fs = det_power.run(ctx, a)
        check([f.id for f in fs] == IDS, f"{name}: {[f.id for f in fs]}")
        for f in fs:
            check(f.status == NA and f.context == "not_armed", f"{name} {f.id}: {f.status}/{f.context}")
            check(f.t_onset is None, f"{name} {f.id} NA 不應有 t_onset")
            check("不適用" in f.text_zh, f"{name} {f.id}: {f.text_zh}")
        # 靜態稽核結果仍附在 values(給報告的衛生區塊用),但狀態是 NA
        aud = by_id(fs)["power_failsafe_config_audit"]
        check("crt_ge_low" in aud.values.get("static_issues", []), f"{name}: {aud.values}")


def test_baseline_tails_not_collapse():
    """118/120 的 log 尾端拔電池(50 → 17..21 V)在未解鎖時絕不能變成 HIT。擋住它的有兩層:
    (1) 解鎖閘門(從未解鎖 → NA);(2) 拔電池當下油門在最低(CTUN.ThO ≤ 0.03)且 log 就結束在掉壓上 → CLEAR 'unplug_on_ground'。
    實測「拿掉閘門」(整份 log 當成一次開放式解鎖):118/120 得到 CLEAR 'unplug_on_ground'(第 2 層擋住,不是 HIT);
    再拿掉落地證據(刪掉 CTUN,也沒有 EV)才會變成 HIT 'log_end_in_flight' → 兩層缺一不可。124 的尾端在上鎖後 27 秒,也不算。"""
    for name in ("00000118", "00000120"):
        ctx = crashkit.load_ctx(name)
        fs = det_power.run(ctx, crashkit.not_armed_anchors(ctx))
        check(all(f.status == NA for f in fs), name)
        tb = ctx.table("BAT")
        tail = det_power.unplug_tail(tb["t"], tb["Volt"])
        check(tail is not None and tail["v_end"] < 25 and tail["ref"] > 49, f"{name} 應辨識出拔電池尾端: {tail}")
        check(tail["tail_len_s"] <= 1.0, tail)
        # 拿掉閘門:整份 log 當成一次開放式解鎖
        iv = ArmedInterval(0, float(tb["t"][0]), ctx.t_end, None, False, open_end=True)
        a = Anchors(interval=iv, intervals=[iv])
        f = det_power.power_voltage_collapse(ctx, a)[0]
        check(f.status == CLEAR and f.context == "unplug_on_ground" and f.values["kind"] == "log_end_in_flight"
              and f.values["on_ground"] is True, f"{name} 無閘門: {f.status}/{f.context} {f.values.get('kind')}")
        near(f.t_onset, tail["t_start"], 0.3, f"{name} 無閘門時的掉壓起點 = 尾端起點")
        # 再拿掉落地證據(沒有 CTUN、沒有 EV 28)→ 無法證明在地面 → 保守判為飛行中斷電
        no_ctun = PatchedCtx(ctx, CTUN=None)
        no_ctun.ev_times = lambda *args, **kw: []
        f = det_power.power_voltage_collapse(no_ctun, a)[0]
        check(f.status == HIT and f.context == "log_end_in_flight", f"{name} 無閘門、無落地證據: {f.status}/{f.context}")
    for name in ("00000119", "00000117", "00000122", "00000123"):
        tb = crashkit.load_ctx(name).table("BAT")
        check(det_power.unplug_tail(tb["t"], tb["Volt"]) is None, f"{name} 不該有拔電池尾端")


# ---------------------------------------------------------------------------
# 4 合成資料
# ---------------------------------------------------------------------------
def test_synth_healthy_12s():
    ctx, a = build()
    d = run_all(ctx, a)
    for i in CRASH_IDS:
        check(d[i].status == CLEAR, f"{i}: {d[i].status} {d[i].text_zh}")
    check(d["power_data_quality"].status == CLEAR and d["power_data_quality"].values["curr_invalid"] is False, d["power_data_quality"].values)
    check(d["power_failsafe_config_audit"].status == CLEAR and d["power_failsafe_config_audit"].values["issues"] == [],
          d["power_failsafe_config_audit"].values)
    check(d["power_low_voltage"].values["cells"] == 12, d["power_low_voltage"].values)
    near(d["power_low_voltage"].values["v1_min"], 4.15 * 12 - 3.0 - 0.004 * 120, 0.2, "合成 V1 最低")
    # 電流計有效 → 不加「電流類故障無法排除」的保留
    check(d["power_low_voltage"].values["curr_meter_invalid"] is False
          and det_power.CURR_CAVEAT_ZH not in d["power_low_voltage"].text_zh, d["power_low_voltage"].text_zh)


def test_synth_low_voltage_critical_and_onset():
    """電壓線性掉到 38 V(< MOT_BAT_VOLT_MIN 39.6 V):HIT,t_onset 約等於通過 39.6 V 的時間。"""
    def volt(t):
        base = 46.0
        return base - 8.0 * np.clip((t - 80.0) / 10.0, 0, 1) + 8.0 * np.clip((t - 95.0) / 5.0, 0, 1)
    ctx, a = build(volt)
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.status == HIT and f.context == "air" and f.values["crit"] is True, f"{f.status} {f.values}")
    t_cross = 80.0 + (46.0 - 39.6) / 0.8                       # 88.0
    near(f.t_onset, t_cross, 0.7, "t_onset")
    check(f.t_alarm == f.t_onset, "crit 立即成立:t_alarm = t_onset")
    check(f.severity == "critical" and "MOT_BAT_VOLT_MIN" in f.text_zh, f.text_zh)
    near(f.values["v1_min"], 38.0, 0.3, "V1 最低")
    # 崩落偵測不該被「緩降 8 V / 10 s」觸發
    d = run_all(ctx, a)
    check(d["power_voltage_collapse"].status == CLEAR, d["power_voltage_collapse"].values)


def test_synth_scale_invariant_6s():
    """6S(25.2 V):健康 → CLEAR;掉到 18.5 V(< 19.8 V)→ HIT;1 秒掉 4.0 V 不算崩落、掉 5.6 V 算(門檻 = 20% × 25.2 = 5.04 V)。"""
    ctx, a = build(cells=6)
    d = run_all(ctx, a)
    check([d[i].status for i in CRASH_IDS] == [CLEAR] * 3, [d[i].status for i in CRASH_IDS])
    check(d["power_low_voltage"].values["cells"] == 6, d["power_low_voltage"].values)
    near(d["power_voltage_collapse"].values["threshold_v"], 5.04, 0.01, "6S 崩落門檻")

    def low(t):
        return 22.0 - 3.5 * np.clip((t - 80.0) / 3.0, 0, 1) + 3.5 * np.clip((t - 90.0) / 3.0, 0, 1)
    ctx, a = build(low, cells=6)
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.status == HIT and 78.0 < f.t_onset < 84.0, f"{f.status} {f.t_onset}")

    def step(drop):
        return lambda t: 22.0 - np.where((t > 100.0) & (t < 103.0), drop, 0.0) * np.clip((t - 100.0) / 1.0, 0, 1)
    ctx, a = build(step(4.0), cells=6)
    check(run_all(ctx, a)["power_voltage_collapse"].status == CLEAR, "4.0 V / 1 s on 6S 不算崩落")
    ctx, a = build(step(5.6), cells=6)
    f = run_all(ctx, a)["power_voltage_collapse"]
    check(f.status == HIT and 99.0 < f.t_onset < 101.2, f"{f.status} {f.t_onset}")


def test_synth_batt_crt_timer():
    """V1 落在 MOT_BAT_VOLT_MIN 與 BATT_CRT_VOLT 之間:持續 ≥ BATT_LOW_TIMER(10 s)才算;參數顛倒(CRT ≥ LOW)只給 SOFT。"""
    def pulse(dur, level=40.2):
        return lambda t: 46.0 - (46.0 - level) * ((t >= 80.0) & (t < 80.0 + dur)) * 1.0
    ctx, a = build(pulse(12.0))                                  # 理智的設定:LOW 43.2 > CRT 40.8
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.status == HIT and f.values["high"] and not f.values["crit"] and not f.values["config_inverted"], f"{f.status} {f.values}")
    near(f.t_onset, 80.0, 0.7, "計時起點")
    near(f.t_alarm - f.t_onset, 10.0, 0.01, "t_alarm = 起點 + BATT_LOW_TIMER")
    ctx, a = build(pulse(6.0))                                   # 不到 10 s
    check(run_all(ctx, a)["power_low_voltage"].status == CLEAR, "6 s < BATT_LOW_TIMER 不算")
    ctx, a = build(pulse(12.0, 41.5), params={"BATT_CRT_VOLT": 42.0, "BATT_LOW_VOLT": 37.0})  # 顛倒設定
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.status == SOFT and f.values["config_inverted"] is True and "顛倒" in f.text_zh, f"{f.status} {f.values}")


def test_synth_err6_failsafe():
    ctx, a = build(err=[(100.0, 6, 1)])
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.status == HIT and f.values["conditions"] == ["err6"] and abs(f.t_onset - 100.0) < 1e-6, f"{f.status} {f.values}")
    for err, why in (([(100.0, 6, 0)], "ECode 0 = 已解除"), ([(5.0, 6, 1)], "未解鎖時的 ERR"), ([(100.0, 5, 1)], "別的子系統")):
        ctx, a = build(err=err)
        check(run_all(ctx, a)["power_low_voltage"].status == CLEAR, why)


def test_synth_midflight_collapse():
    """飛行中 0.5 秒內 46 → 25 V 後維持:崩落 HIT(context in_flight),t_onset 約在掉壓起點;低電壓也 HIT(25 V < 39.6 V)。"""
    def volt(t):
        return 46.0 - 21.0 * np.clip((t - 90.0) / 0.5, 0, 1)
    ctx, a = build(volt)
    d = run_all(ctx, a)
    f = d["power_voltage_collapse"]
    check(f.status == HIT and f.context == "in_flight" and f.values["on_ground"] is False, f"{f.status} {f.context} {f.values}")
    near(f.t_onset, 90.0, 0.35, "掉壓起點")
    near(f.values["drop_v"], 21.0, 3.0, "掉壓幅度")
    check(d["power_low_voltage"].status == HIT, "壓在 25 V 也是低電壓")
    # 8 V / 1 s(12S)低於門檻 10.08 V:CLEAR
    ctx, a = build(lambda t: 46.0 - 8.0 * np.clip((t - 90.0) / 1.0, 0, 1))
    check(run_all(ctx, a)["power_voltage_collapse"].status == CLEAR, "8 V / 1 s 在 12S 不算崩落")


def _tail_volt(t_tail, hover=46.0, end=17.0, tail_len=0.5):
    return lambda t: np.where(t < t_tail, hover - 0.004 * t, hover - 0.004 * t_tail - (hover - end) * np.clip((t - t_tail) / tail_len, 0, 1))


def test_synth_unplug_tail_in_flight_vs_on_ground():
    """解鎖中 log 就結束在拔電池曲線上:油門非最低 → 飛行中斷電 HIT;油門已在最低(落地後拔)→ CLEAR 'unplug_on_ground'。"""
    t_total = 160.0
    tail_t = t_total - 0.6
    ctx, a = build(_tail_volt(tail_t), t_total=t_total, arm=(20.0, 139.0), open_end=True)
    d = run_all(ctx, a)
    f = d["power_voltage_collapse"]
    check(a.interval.open_end and a.interval.t1 > tail_t, "測試前提:區間含尾端")
    check(f.status == HIT and f.context == "log_end_in_flight", f"{f.status} {f.context} {f.values}")
    near(f.t_onset, tail_t, 0.35, "尾端起點")
    # 尾端樣本從低電壓統計裁掉,避免把拔電池當低電壓
    lv = d["power_low_voltage"]
    check(lv.status == CLEAR and lv.values["tail_trimmed"] is True and lv.values["v1_min"] > 44.0, f"{lv.status} {lv.values}")

    ctx, a = build(_tail_volt(tail_t), t_total=t_total, arm=(20.0, 139.0), open_end=True,
                   tho_fn=lambda t: np.where(t < 120.0, 0.15, 0.0))     # 120 s 起油門到底(已落地)
    f = run_all(ctx, a)["power_voltage_collapse"]
    check(f.status == CLEAR and f.context == "unplug_on_ground" and f.values["on_ground"] is True, f"{f.status} {f.context}")
    check("落地後" in f.text_zh, f.text_zh)

    # 沒有 CTUN 時改用 EV 落地狀態(EV 28 離地 → EV 18 落地)
    ctx, a = build(_tail_volt(tail_t), t_total=t_total, arm=(20.0, 139.0), open_end=True, no_ctun=True,
                   ev=[(25.0, 28), (125.0, 18)])
    f = run_all(ctx, a)["power_voltage_collapse"]
    check(f.status == CLEAR and f.context == "unplug_on_ground", f"EV 備援: {f.status} {f.context}")
    ctx, a = build(_tail_volt(tail_t), t_total=t_total, arm=(20.0, 139.0), open_end=True, no_ctun=True, ev=[(25.0, 28)])
    f = run_all(ctx, a)["power_voltage_collapse"]
    check(f.status == HIT, "EV 顯示一直在空中 → 保守判為飛行中")
    ctx, a = build(_tail_volt(tail_t), t_total=t_total, arm=(20.0, 139.0), open_end=True, no_ctun=True)
    f = run_all(ctx, a)["power_voltage_collapse"]
    check(f.status == HIT and f.values["on_ground"] is None, "沒有 CTUN 也沒有 EV → 無法證明在地面 → 保守 HIT")


def test_synth_tail_after_disarm_is_not_collapse():
    """上鎖後 25 秒才拔電池(124 的情況):CLEAR,尾端只記在 values['log_tail'] 且 armed_at_tail=False。"""
    t_total = 170.0
    ctx, a = build(_tail_volt(t_total - 0.6), t_total=t_total, arm=(20.0, 140.0))
    d = run_all(ctx, a)
    f = d["power_voltage_collapse"]
    check(f.status == CLEAR, f"{f.status} {f.text_zh}")
    tail = f.values["log_tail"]
    check(tail is not None and tail["armed_at_tail"] is False and 28.0 < tail["secs_after_disarm"] < 30.0, tail)
    check("拔電池" in f.text_zh, f.text_zh)
    check(d["power_low_voltage"].status == CLEAR and d["power_low_voltage"].values["v1_min"] > 40, d["power_low_voltage"].values)


def test_synth_collapse_needs_persistence():
    """審查回歸:舊版單點對單點比較會把一個壞樣本(0 V)或向上跳動後的回落當成 critical 崩落 HIT。
    現在:單點跨線 → SOFT 'transient';0.2 s(2 筆)就恢復 → SOFT;持續 0.5 s → HIT;上跳 15 V 1 秒再回落 → CLEAR;
    log 在一個低樣本上結束(來不及驗證持續)→ 視為斷電導致 log 結束 → HIT。"""
    def collapse(volt_fn, **kw):
        ctx, a = build(volt_fn, **kw)
        return run_all(ctx, a)

    d = collapse(lambda t: np.where(np.isclose(t, 100.0), 0.0, 46.0))
    f = d["power_voltage_collapse"]
    check(f.status == SOFT and f.context == "transient" and f.values["transient_samples"] == 1, f"單一壞樣本: {f.status}/{f.context} {f.values}")
    near(f.t_onset, 99.9, 0.15, "壞樣本起點")
    check(f.values["max_drop_persist_v"] < 1.0 and f.values["max_drop_1s_v"] > 40.0, f.values)
    check("證據力弱" in f.text_zh and "崩落" not in f.text_zh, f.text_zh)
    check(d["power_low_voltage"].status == CLEAR, "1 秒中位數不受單一壞樣本影響")

    f = collapse(lambda t: np.where((t > 99.95) & (t < 100.15), 20.0, 46.0))["power_voltage_collapse"]
    check(f.status == SOFT and f.values["transient_samples"] == 2, f"0.2 s 就恢復: {f.status} {f.values['transient_samples']}")

    f = collapse(lambda t: np.where((t > 99.95) & (t < 100.45), 20.0, 46.0))["power_voltage_collapse"]
    check(f.status == HIT and f.context == "in_flight", f"持續 0.5 s 的掉壓: {f.status}/{f.context}")
    near(f.t_onset, 99.9, 0.15, "持續掉壓起點")
    near(f.values["drop_v"], 26.0, 0.5, "持續掉壓幅度")

    f = collapse(lambda t: np.where((t > 100.0) & (t < 101.0), 61.0, 46.0))["power_voltage_collapse"]
    check(f.status == CLEAR and f.values["max_drop_1s_v"] < 1.0, f"上跳後回落不是崩落: {f.status} {f.values['max_drop_1s_v']}")

    t_total = 160.0
    ctx, a = build(lambda t: np.where(t > t_total - 0.15, 20.0, 46.0), t_total=t_total, open_end=True)
    f = run_all(ctx, a)["power_voltage_collapse"]
    check(a.interval.open_end and f.status == HIT and f.context == "log_end_in_flight", f"log 在低樣本上結束: {f.status}/{f.context}")


def test_synth_midlog_collapse_on_ground_is_not_unplug():
    """審查回歸:解鎖中、油門在最低時電壓崩落,但 log 之後還繼續記錄(例如翻滾中電池脫落)→ 不能說成「落地後拔電池」的 CLEAR;
    回 HIT context='on_ground',由 engine 依時間貼先兆 / 後果。只有 log 在掉壓上結束才回 CLEAR 'unplug_on_ground'
    (見 test_synth_unplug_tail_in_flight_vs_on_ground)。"""
    ctx, a = build(lambda t: np.where(t < 110.0, 46.0, 15.0), tho_fn=lambda t: np.where(t < 100.0, 0.15, 0.0))
    f = run_all(ctx, a)["power_voltage_collapse"]
    check(f.status == HIT and f.context == "on_ground" and f.values["on_ground"] is True and f.values["kind"] == "on_ground",
          f"{f.status}/{f.context} {f.values.get('kind')}")
    near(f.t_onset, 109.9, 0.15, "掉壓起點")
    check("不能解釋成人員拔電池" in f.text_zh and "不是飛行中掉電" not in f.text_zh, f.text_zh)
    check(f.values["log_tail"] is None, "log 繼續記錄 → 不是尾端")


def test_synth_consequence_wording_after_t_loss():
    """審查回歸:HIT 的文字會照原樣顯示在時間軸。條件在 t_loss 之後才開始 → 寫「屬後果」、不寫「可能導致失控 / 通常是接頭鬆脫」;
    在 t_loss 之前(或沒有 t_loss)→ 原本的原因說法。values['after_loss'] 同步。"""
    volt = lambda t: np.where((t >= 100.0) & (t < 130.0), 30.0, 46.0)      # 100 s 掉到 30 V:崩落 + 低電壓
    kw = dict(vcc_fn=lambda t: np.where((t >= 100.0) & (t < 101.0), 4.2, 5.22), err=[(100.5, 6, 1)])
    cause = {"power_low_voltage": "可能導致失控", "power_voltage_collapse": "通常是電池接頭鬆脫", "power_rail_fault": "可能造成飛控重啟"}
    for t_warn, t_loss, late in ((95.0, 98.0, True), (105.0, 110.0, False), (None, None, None)):
        ctx, a = build(volt, **kw)
        a.t_warn, a.t_loss = t_warn, t_loss
        d = run_all(ctx, a)
        for i in CRASH_IDS:
            f = d[i]
            check(f.status == HIT and f.values["after_loss"] is late, f"{i} t_loss={t_loss}: {f.status} after_loss={f.values.get('after_loss')}")
            if late:
                check("屬後果" in f.text_zh and "不是失控原因" in f.text_zh and cause[i] not in f.text_zh, f"{i} 應寫成後果: {f.text_zh}")
            else:
                check("屬後果" not in f.text_zh and cause[i] in f.text_zh, f"{i} 失控前應維持原因說法: {f.text_zh}")
        if late:
            check(d["power_low_voltage"].text_zh.count("屬後果") == 2, f"低電壓與 ERR 6 兩段都要寫成後果: {d['power_low_voltage'].text_zh}")


def test_synth_rail_acc_flags_pre_arm():
    """審查回歸:POWR.AccFlags 是開機以來的累計值。解鎖前就成立的過流位元不是飛行中事件:
    SOFT context='pre_arm'、t_onset = 真正第一次出現的時間(engine 貼 ground_info,不計分);飛行中才新出現的位元 / 即時旗標 → HIT。"""
    from core.crash import engine
    from core.crash.types import TAG_GROUND
    ctx, a = build(acc_fn=lambda t: np.where(t >= 5.0, 9.0, 1.0))
    f = run_all(ctx, a)["power_rail_fault"]
    check(f.status == SOFT and f.context == "pre_arm" and f.values["conditions"] == [], f"{f.status}/{f.context} {f.values['conditions']}")
    near(f.t_onset, 5.0, 0.05, "累計旗標真正第一次出現")
    check(f.values["acc_overcurrent_pre_arm"] == [{"bit": 8, "t_first": 5.0}], f.values["acc_overcurrent_pre_arm"])
    check(engine.tag_time(f.t_onset, a) == TAG_GROUND, "解鎖前的事件只能是地面資訊")
    check("解鎖前" in f.text_zh and "不是這次飛行中發生的" in f.text_zh, f.text_zh)
    # 解鎖前有 bit3,飛行中 100 s 又新出現 bit4 → HIT,t_onset = 100 s;文字另註 bit3 是解鎖前的
    ctx, a = build(acc_fn=lambda t: np.where(t >= 100.0, 25.0, np.where(t >= 5.0, 9.0, 1.0)))
    f = run_all(ctx, a)["power_rail_fault"]
    check(f.status == HIT and f.values["conditions"] == ["periph_overcurrent"], f"{f.status} {f.values['conditions']}")
    near(f.t_onset, 100.0, 0.15, "飛行中新出現的位元")
    check("解鎖前的 5.0 s" in f.text_zh, f.text_zh)
    # 即時旗標(Flags)在飛行中過流 → HIT(累計旗標解鎖前就有也一樣)
    ctx, a = build(acc_fn=lambda t: np.where(t >= 5.0, 9.0, 1.0), flags_fn=lambda t: np.where((t >= 100.0) & (t < 101.0), 9.0, 1.0))
    f = run_all(ctx, a)["power_rail_fault"]
    check(f.status == HIT and "periph_overcurrent" in f.values["conditions"], f"{f.status} {f.values['conditions']}")
    near(f.t_onset, 100.0, 0.15, "即時旗標起點")


def test_synth_texts_never_print_nan():
    """審查回歸:算不出來的數字不能以「nan」出現在文字裡,也不能宣稱沒檢查過的事。"""
    ctx, a = build()
    ctx._tables["POWR"]["Vcc"][500:510] = np.nan
    d = run_all(ctx, a)
    f = d["power_rail_fault"]
    check(f.status == CLEAR and abs(f.values["vcc_min"] - 5.22) < 1e-6 and "nan" not in f.text_zh.lower(), f.text_zh)
    # 油門是常數 → 電流與油門的相關係數算不出來:不寫「r = nan」也不宣稱「不呈負相關」
    f = d["power_data_quality"]
    check(f.status == CLEAR and f.values["corr_curr_thr"] is None, f"{f.status} {f.values}")
    check("nan" not in f.text_zh.lower() and "r =" not in f.text_zh and "不呈負相關" not in f.text_zh, f.text_zh)
    # NaN 夾雜在真正的掉電裡:照樣抓到,文字是真實數值
    ctx, a = build(vcc_fn=lambda t: np.where((t >= 100.0) & (t < 101.0), 4.2, 5.22))
    ctx._tables["POWR"]["Vcc"][990:995] = np.nan
    f = run_all(ctx, a)["power_rail_fault"]
    check(f.status == HIT and "4.20 V" in f.text_zh and "nan" not in f.text_zh.lower(), f.text_zh)
    # Vcc 全是 NaN → NA(不是 CLEAR)
    ctx, a = build()
    ctx._tables["POWR"]["Vcc"][:] = np.nan
    f = run_all(ctx, a)["power_rail_fault"]
    check(f.status == NA and f.context == "too_short", f"{f.status}/{f.context}")


def test_synth_gaps_do_not_fool_time_windows():
    """BAT 有 0.8 s 空洞:1 秒中位數 / 1 秒壓降一律用時間窗,緩慢掉壓 + 空洞不能變成崩落或低電壓。"""
    gaps = [(g, g + 0.8) for g in np.arange(30.0, 135.0, 7.0)]
    def volt(t):
        return 46.0 - 0.03 * (t - 20.0)                          # 緩降 0.03 V/s,全程 3.6 V(46 → 42.4 V,高於 BATT_CRT_VOLT 40.8 V)
    ctx, a = build(volt, gaps=gaps)
    d = run_all(ctx, a)
    check(d["power_voltage_collapse"].status == CLEAR, d["power_voltage_collapse"].values)
    check(d["power_low_voltage"].status == CLEAR, d["power_low_voltage"].values)
    check(d["power_voltage_collapse"].values["max_drop_1s_v"] < 2.0, d["power_voltage_collapse"].values)
    # 掉壓發生在空洞裡(空洞前 46 V → 空洞後 30 V):前後樣本相隔 0.9 s ≤ 1.5 s,照樣抓得到
    def hole(t):
        return np.where(t < 80.4, 46.0, 30.0)
    ctx, a = build(hole, gaps=[(80.0, 80.8)])
    f = run_all(ctx, a)["power_voltage_collapse"]
    check(f.status == HIT, f"空洞中掉壓應抓到: {f.status}")


def test_window_median_time_based():
    t = np.concatenate([np.arange(0.0, 1.05, 0.1), np.arange(3.0, 4.05, 0.1)])      # 0..1.0 與 3.0..4.0,中間空 2 秒
    v = np.concatenate([np.full(11, 10.0), np.full(11, 20.0)])
    m = det_power._window_median(t, v, 1.0)
    check(np.allclose(m[:11][2:9], 10.0) and np.allclose(m[11:][2:9], 20.0), m)       # 空洞兩側互不混合
    check(not np.isnan(m[0]), "邊緣窗仍有 ≥3 個樣本")
    sparse = det_power._window_median(np.array([0.0, 5.0, 10.0]), np.array([1.0, 2.0, 3.0]), 1.0)
    check(np.isnan(sparse).all(), "窗內樣本 < 3 → NaN(空洞太大不下判斷)")
    check(det_power._window_median(np.zeros(0), np.zeros(0), 1.0).size == 0, "空輸入")
    big = np.arange(0.0, 3000.0, 0.01)                                              # 30 萬點分塊不爆
    mm = det_power._window_median(big, np.sin(big), 0.5)
    check(mm.shape == big.shape and np.isfinite(mm[50:-50]).all(), "大陣列")


def test_unplug_tail_unit():
    t = np.arange(0.0, 60.0, 0.1)
    v = np.full(t.size, 50.0)
    v[t >= 59.0] = 50.0 - (np.arange((t >= 59.0).sum()) + 1) * 6.0                   # 59.0 起每 0.1 s 掉 6 V → 末端約 20 V
    tail = det_power.unplug_tail(t, v)
    check(tail is not None and 58.8 <= tail["t_start"] <= 59.1 and tail["v_end"] < 30 and tail["slope_v_s"] < -30, tail)
    check(det_power.unplug_tail(t, np.full(t.size, 50.0)) is None, "平的不算")
    v2 = np.full(t.size, 50.0)
    v2[t >= 50.0] = 40.0                                                            # 10 秒前就掉(不是尾端)
    check(det_power.unplug_tail(t, v2) is None, "掉壓離結尾太遠 / 未低於 60%")
    v3 = np.full(t.size, 50.0)
    v3[-30:] = np.linspace(50.0, 10.0, 30)                                          # 3 秒慢掉 → 超過 1.5 s
    check(det_power.unplug_tail(t, v3) is None, "尾端超過 1.5 s 不算拔電池")
    check(det_power.unplug_tail(t[:10], v[:10]) is None, "樣本太少")
    check(det_power.unplug_tail(t, np.full(t.size, 0.05)) is None, "USB(<5 V)不談尾端")
    rng = np.random.default_rng(3)
    vn = 46.0 + rng.normal(0, 0.6, t.size)
    vn[-6:] = np.linspace(40, 17, 6)
    tn = det_power.unplug_tail(t, vn)
    check(tn is not None and 58.8 <= tn["t_start"] <= 59.6, f"有雜訊的尾端: {tn}")


def test_synth_usb_and_no_battery_na():
    ctx, a = build(usb=True)
    d = run_all(ctx, a)
    for i in CRASH_IDS + ["power_data_quality"]:
        check(d[i].status == NA and d[i].context == "usb_or_no_battery", f"{i}: {d[i].status}/{d[i].context}")
        check("USB" in d[i].text_zh, d[i].text_zh)
    # 沒接電池但旗標也不是 USB(BAT.Volt 趨近 0):同樣 NA
    ctx, a = build(volt_fn=lambda t: np.full(t.size, 0.03))
    d = run_all(ctx, a)
    check(all(d[i].status == NA for i in CRASH_IDS), [d[i].status for i in CRASH_IDS])


def test_synth_too_short_and_no_data():
    ctx, a = build(arm=(20.0, 21.5))
    d = run_all(ctx, a)
    check(d["power_low_voltage"].status == NA and d["power_low_voltage"].context == "too_short", d["power_low_voltage"].context)
    check(d["power_voltage_collapse"].status == NA, d["power_voltage_collapse"].status)
    ctx, a = build()
    del ctx._tables["BAT"]
    d = run_all(ctx, a)
    check(d["power_low_voltage"].status == NA and d["power_low_voltage"].context == "no_data", d["power_low_voltage"].context)
    check(d["power_data_quality"].status == NA and d["power_data_quality"].context == "no_data", d["power_data_quality"].context)
    ctx, a = build()
    del ctx._tables["POWR"]
    check(run_all(ctx, a)["power_rail_fault"].context == "no_data", "沒有 POWR → NA")


def test_synth_rail_faults():
    ctx, a = build()
    check(run_all(ctx, a)["power_rail_fault"].status == CLEAR, "健康")

    def dip(level, t_a=100.0, t_b=101.0):
        return lambda t: np.where((t >= t_a) & (t < t_b), level, 5.22)

    cases = {
        "vcc_low": dict(vcc_fn=dip(4.2)),
        "mcu_low": dict(mcu_fn=lambda t: np.where((t >= 100.0) & (t < 101.0), 3.0, 3.29)),
        "servo_rail_low": dict(vservo_fn=lambda t: np.where((t >= 100.0) & (t < 101.0), 4.0, 5.0)),
        "periph_overcurrent": dict(acc_fn=lambda t: np.where(t >= 100.0, 9.0, 1.0)),
        "brick_lost": dict(flags_fn=lambda t: np.where((t >= 100.0) & (t < 101.0), 0.0, 1.0)),
    }
    for name, kw in cases.items():
        ctx, a = build(**kw)
        f = run_all(ctx, a)["power_rail_fault"]
        check(f.status == HIT and name in f.values["conditions"], f"{name}: {f.status} {f.values['conditions']}")
        near(f.t_onset, 100.0, 0.15, f"{name} t_onset")
        check(f.severity == "high" and f.context == "armed", f.severity)
    # 未解鎖時的低值(解鎖前伺服軌 0.1 V、解鎖前 Vcc 掉)不算:故障只發生在區間外
    ctx, a = build(vservo_fn=lambda t: np.where(t < 20.0, 0.1, 5.0), vcc_fn=lambda t: np.where(t < 15.0, 4.0, 5.22))
    check(run_all(ctx, a)["power_rail_fault"].status == CLEAR, "解鎖區間外的低值不算")
    # 伺服軌沒接(整段 0.1 V):不評估 VServo,不誤報
    ctx, a = build(vservo_fn=lambda t: np.full(t.size, 0.1))
    f = run_all(ctx, a)["power_rail_fault"]
    check(f.status == CLEAR and f.values["vservo_evaluated"] is False, f.values)
    # 這機型沒接電源模組(Flags 恆為 2 = 只有伺服軌有效):沒有磚旗標就不談「磚失效」
    ctx, a = build(flags_fn=lambda t: np.full(t.size, 2.0))
    f = run_all(ctx, a)["power_rail_fault"]
    check(f.status == CLEAR and f.values["brick_fraction"] == 0.0, f.values)
    check("一直有效" not in f.text_zh and "沒有電源模組有效旗標" in f.text_zh, f"文字不得宣稱沒檢查的項目: {f.text_zh}")
    # 連 Flags / AccFlags 欄位都沒有:不提磚、不提過流旗標
    ctx, a = build()
    for k in ("Flags", "AccFlags"):
        del ctx._tables["POWR"][k]
    f = run_all(ctx, a)["power_rail_fault"]
    check(f.status == CLEAR and "電源模組" not in f.text_zh and "過流" not in f.text_zh, f.text_zh)
    # 同時多項:列出全部,t_onset 取最早
    ctx, a = build(vcc_fn=dip(4.2, 110.0, 111.0), mcu_fn=lambda t: np.where((t >= 100.0) & (t < 101.0), 3.0, 3.29))
    f = run_all(ctx, a)["power_rail_fault"]
    check(set(f.values["conditions"]) == {"vcc_low", "mcu_low"} and abs(f.t_onset - 100.0) < 0.15, f.values)


def test_synth_data_quality():
    # 懸停 12S 六軸電流 20..30 A、與油門正相關 → CLEAR
    ctx, a = build(thr_fn=lambda t: np.where((t >= 20) & (t <= 140), 0.15 + 0.03 * np.sin(t), 0.0),
                   curr_fn=lambda t, thr: 5.0 + 150.0 * thr)
    f = run_all(ctx, a)["power_data_quality"]
    check(f.status == CLEAR and f.values["curr_invalid"] is False and f.values["corr_curr_thr"] > 0.9, f"{f.status} {f.values}")
    # 電流 ≈ 0(無效)+ 下陷 3 V → HIT
    ctx, a = build(curr_fn=lambda t, thr: np.full(t.size, 0.004))
    f = run_all(ctx, a)["power_data_quality"]
    check(f.status == HIT and f.values["i_med_a"] < 0.01 and f.values["sag_v"] >= 1.5 and f.severity == "medium", f"{f.status} {f.values}")
    check("內阻" not in f.text_zh and "老化" not in f.text_zh and "剩餘電量" not in f.text_zh, f.text_zh)
    # 電流與油門反相關但電壓下陷不明顯、電流 > 1 A → SOFT
    ctx, a = build(thr_fn=lambda t: np.where((t >= 20) & (t <= 140), 0.15 + 0.03 * np.sin(t), 0.0),
                   curr_fn=lambda t, thr: 20.0 - 60.0 * thr)
    f = run_all(ctx, a)["power_data_quality"]
    check(f.status == SOFT and f.values["corr_curr_thr"] < 0, f"{f.status} {f.values}")
    # 懸停不到 20 秒 → NA
    ctx, a = build(arm=(20.0, 35.0))
    f = run_all(ctx, a)["power_data_quality"]
    check(f.status == NA and f.context == "no_hover" and "20 秒" in f.text_zh, f"{f.status} {f.context} {f.text_zh}")
    # 解鎖前沒有靜止電壓(一開機就解鎖)→ NA,而不是亂猜
    ctx, a = build(arm=(0.5, 140.0))
    f = run_all(ctx, a)["power_data_quality"]
    check(f.status == NA and f.context == "no_rest_voltage", f"{f.status} {f.context}")
    check("靜止電壓" in f.text_zh, f.text_zh)
    # 懸停油門來源:MOT_THST_HOVER → CTUN.ThH → 0.35 預設(0.15 懸停的機型用 0.35 會把整段懸停排除)
    ctx, a = build(curr_fn=lambda t, thr: np.full(t.size, 0.004))
    check(run_all(ctx, a)["power_data_quality"].values["hover_src"] == "param", "有參數用參數")
    ctx, a = build(curr_fn=lambda t, thr: np.full(t.size, 0.004), params={"MOT_THST_HOVER": None})
    f = run_all(ctx, a)["power_data_quality"]
    check(f.status == HIT and f.values["hover_src"] == "CTUN.ThH" and abs(f.values["mot_thst_hover"] - 0.15) < 1e-6, f"{f.status} {f.values}")
    ctx, a = build(curr_fn=lambda t, thr: np.full(t.size, 0.004), params={"MOT_THST_HOVER": None}, no_ctun=True)
    f = run_all(ctx, a)["power_data_quality"]
    check(f.status == NA and f.context == "no_hover", f"沒有參數也沒有 CTUN.ThH → 0.35 預設 → 懸停不足: {f.status} {f.context}")
    # 輕載機:電流小、下陷也小、沒有負相關 → NA 'indeterminate'(不硬判)
    ctx, a = build(volt_fn=lambda t: np.full(t.size, 46.0), curr_fn=lambda t, thr: np.full(t.size, 0.5) + 0.2 * thr)
    f = run_all(ctx, a)["power_data_quality"]
    check(f.status == NA and f.context == "indeterminate", f"{f.status} {f.context}")


def test_synth_config_audit():
    def audit(**p):
        ctx, a = build(params=p)
        return run_all(ctx, a)["power_failsafe_config_audit"]

    f = audit()
    check(f.status == CLEAR and f.values["issues"] == [], f.values)
    f = audit(BATT_CRT_VOLT=44.0, BATT_LOW_VOLT=43.0)
    check(f.status == HIT and f.values["issues"] == ["crt_ge_low"], f.values)
    f = audit(BATT_CRT_VOLT=43.0, BATT_LOW_VOLT=43.0)
    check(f.values["issues"] == ["crt_ge_low"], "相等也算顛倒")
    f = audit(BATT_CRT_VOLT=0.0)
    check(f.status == CLEAR, "CRT 未啟用(0)不算顛倒")
    f = audit(BATT_FS_LOW_ACT=0.0, BATT_FS_CRT_ACT=0.0)
    check(f.values["issues"] == ["no_failsafe_action"], f.values)
    f = audit(BATT_FS_LOW_ACT=0.0)
    check(f.status == CLEAR, "只有一個動作為 0 不算")
    f = audit(BATT_LOW_MAH=0.0, BATT_CRT_MAH=0.0)
    check(f.values["issues"] == ["capacity_failsafe_off"], f.values)
    f = audit(ARMING_CHECK=10304.0)
    check(f.values["issues"] == ["batt_arm_volt_not_enforced"], f.values)
    f = audit(ARMING_CHECK=256.0 + 64.0)
    check(f.status == CLEAR, "ARMING_CHECK 有 bit8(電池電量)就生效")
    f = audit(ARMING_CHECK=10304.0, BATT_ARM_VOLT=0.0)
    check(f.status == CLEAR, "BATT_ARM_VOLT=0(沒啟用)就沒有「不生效」的問題")
    f = audit(FS_CRASH_CHECK=0.0)
    check(f.status == CLEAR and "crash_check_disabled" in f.values["notes"], f.values)
    f = audit(BATT_CRT_VOLT=44.0, BATT_LOW_VOLT=43.0, BATT_FS_LOW_ACT=0.0, BATT_FS_CRT_ACT=0.0)
    check(set(f.values["issues"]) == {"crt_ge_low", "no_failsafe_action"} and "順序顛倒" in f.text_zh and "動作皆為 0" in f.text_zh
          and "容量門檻" not in f.text_zh, f.text_zh)
    ctx, a = build(no_params=True)
    f = run_all(ctx, a)["power_failsafe_config_audit"]
    check(f.status == NA and f.context == "no_params", f"{f.status} {f.context}")
    # 沒有 MOT_BAT_VOLT_* 但有 BATT_*:仍可稽核(每顆電壓換算略過)
    ctx, a = build(params={"MOT_BAT_VOLT_MAX": 0.0, "BATT_CRT_VOLT": 44.0, "BATT_LOW_VOLT": 43.0})
    f = run_all(ctx, a)["power_failsafe_config_audit"]
    check(f.status == HIT and f.values["low_per_cell"] is None, f.values)


def test_synth_cells_fallback_when_params_missing():
    """MOT_BAT_VOLT_MAX / MIN 都沒設(ArduPilot 預設 0):串數 = ⌈到上鎖為止 1 秒中位數最高電壓 / 4.35⌉(49.8 V → ⌈11.45⌉ = 12),
    門檻改用 3.3 V/cell。門檻是估的 → 跌破時只給 SOFT,文字寫「估計 N 串 × 3.3 V/cell」而不是 MOT_BAT_VOLT_MIN。"""
    unset = {"MOT_BAT_VOLT_MAX": 0.0, "MOT_BAT_VOLT_MIN": 0.0}

    def with_rest(fn, rest=4.15 * 12, t0=20.0):            # 解鎖前是 12S 的靜止電壓(串數才估得到 12)
        return lambda t: np.where(t < t0, rest, fn(t))

    ctx, a = build(params=unset)
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.status == CLEAR and f.values["cells"] == 12 and f.values["cells_src"] == "estimated", f.values)
    check(f.values["v_crit_cfg_src"] == "3.3V*cells" and abs(f.values["v_crit_cfg"] - 39.6) < 0.01 and f.values["v_crit_estimated"], f.values)
    near(f.values["cells_top_v"], 4.15 * 12, 0.05, "估串數用的最高 1 秒中位數電壓")
    check("MOT_BAT_VOLT_MIN 39.6" not in f.text_zh and "估計 12 串" in f.text_zh, f"沒設參數就不能說是 MOT_BAT_VOLT_MIN: {f.text_zh}")
    sag = with_rest(lambda t: 46.0 - 8.0 * np.clip((t - 80.0) / 10.0, 0, 1))
    ctx, a = build(sag, params=dict(unset, BATT_CRT_VOLT=0.0))
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.status == SOFT and f.values["crit"] is True and f.values["conditions"] == ["critical_low"], f"估計門檻只給 SOFT: {f.status} {f.values}")
    check("估計 12 串 × 3.3 V/cell" in f.text_zh and "MOT_BAT_VOLT_MIN 39.6" not in f.text_zh and "證據力較弱" in f.text_zh, f.text_zh)
    near(f.t_onset, 88.0, 0.7, "估計門檻的跨線時間")
    # 有設 BATT_CRT_VOLT(使用者設定值 40.8 V)且持續 ≥ 10 s 低於它 → 那一項照樣 HIT(估計門檻只是不能單獨撐起 HIT)
    ctx, a = build(with_rest(lambda t: 46.0 - 8.0 * np.clip((t - 80.0) / 2.0, 0, 1) * ((t >= 80.0) & (t < 100.0))), params=unset)
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.status == HIT and set(f.values["conditions"]) == {"critical_low", "critical_timer"}, f"{f.status} {f.values['conditions']}")
    # MOT_BAT_VOLT_MIN 有設、MOT_BAT_VOLT_MAX 沒設:門檻是設定值 → HIT,只有 V/cell 換算註明串數是估的
    ctx, a = build(sag, params={"MOT_BAT_VOLT_MAX": 0.0})
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.status == HIT and not f.values["v_crit_estimated"] and "MOT_BAT_VOLT_MIN 39.6" in f.text_zh and "估計 12 串" in f.text_zh, f.text_zh)


def _lihv_volt(cells, v_rest_cell, v_end_cell, t0=20.0, t1=140.0):
    """靜止 v_rest_cell V/cell;解鎖後下陷 0.25 V/cell 再線性降到 v_end_cell V/cell(健康放電)。"""
    def f(t):
        frac = np.clip((t - t0) / (t1 - t0), 0, 1)
        loaded = cells * (v_rest_cell - 0.25) - cells * (v_rest_cell - 0.25 - v_end_cell) * frac
        return np.where((t >= t0) & (t <= t1), loaded, cells * v_rest_cell)
    return f


def test_synth_cells_estimate_never_too_high_lihv():
    """審查回歸:舊的 ⌈Vmax / 4.25⌉ 把 LiHV(4.35 V/cell)4S 估成 5 串 → 門檻 16.5 V,健康飛行 critical HIT。
    現在 ⌈Vmax / 4.35⌉:LiHV 4S / 6S 與一般 LiPo 健康飛行都要 CLEAR、串數正確;讀值 ≤ 4.35 V/cell 時估計值不會多於實際。"""
    unset = {"MOT_BAT_VOLT_MAX": 0.0, "MOT_BAT_VOLT_MIN": 0.0}
    for cells, rest, end in ((4, 4.35, 3.75), (6, 4.35, 3.80), (12, 4.35, 3.75), (6, 4.20, 3.70), (12, 4.27, 3.80), (4, 4.05, 3.60)):
        ctx, a = build(_lihv_volt(cells, rest, end), cells=cells,
                       params=dict(unset, BATT_LOW_VOLT=cells * 3.5, BATT_CRT_VOLT=cells * 3.3))
        f = run_all(ctx, a)["power_low_voltage"]
        check(f.status == CLEAR, f"{cells}S rest {rest} V/cell 健康飛行應 CLEAR,得 {f.status}: {f.text_zh}")
        check(f.values["cells"] == cells and f.values["cells_src"] == "estimated", f"{cells}S rest {rest}: 估成 {f.values['cells']} 串")
    # 部分放電的 12S(靜止 3.95 V/cell = 47.4 V)會少估成 11 串:門檻變寬鬆(36.3 V),不會變成誤報
    ctx, a = build(_lihv_volt(12, 3.95, 3.60), params=unset)
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.values["cells"] == 11 and f.status == CLEAR and "串數若低估則門檻偏寬鬆" in f.text_zh, f"{f.values['cells']} {f.status} {f.text_zh}")
    # 一個向上跳動的壞樣本(解鎖前 61 V)不能讓串數多一(用 1 秒中位數,不用單點最大值)
    base = _lihv_volt(12, 4.15, 3.80)
    ctx, a = build(lambda t: np.where(np.isclose(t, 10.0), 61.0, base(t)), params=unset)
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.values["cells"] == 12 and f.status == CLEAR, f"上跳壞樣本: {f.values['cells']} 串 {f.status}")
    # 讀值偏高超過 4.35 V/cell(LiHV + 校正偏高 3%)仍可能多估一串:健康放電(實際 3.6 V/cell)跌破被抬高的門檻也只會是 SOFT,不會 HIT
    ctx, a = build(_lihv_volt(4, 4.35 * 1.03, 3.6 * 1.03), cells=4, params=unset)
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.values["cells"] == 5 and f.status == SOFT and f.values["v_crit_estimated"], f"多估一串時最多 SOFT: {f.values['cells']} {f.status}")
    # 同一份資料再設 BATT_CRT_VOLT 15.5 V(使用者設定,約 101.5 s 起持續跌破)→ HIT,但起點是 BATT_CRT_VOLT 那一項的起點,
    # 不能被估計門檻的 SOFT(約 44 s 就跨線)提前
    ctx, a = build(_lihv_volt(4, 4.35 * 1.03, 3.6 * 1.03), cells=4, params=dict(unset, BATT_CRT_VOLT=15.5, BATT_LOW_VOLT=16.0))
    f = run_all(ctx, a)["power_low_voltage"]
    check(f.status == HIT and set(f.values["conditions"]) == {"critical_low", "critical_timer"}, f"{f.status} {f.values['conditions']}")
    near(f.t_onset, 101.5, 1.0, "HIT 的起點取自 BATT_CRT_VOLT 條件")


def test_not_armed_all_na_synth():
    ctx, a = build()
    a2 = Anchors()                                    # interval=None
    d = run_all(ctx, a2)
    check(list(d) == IDS and all(f.status == NA and f.context == "not_armed" for f in d.values()), {k: v.status for k, v in d.items()})


def test_guard_converts_exceptions_to_na():
    ctx, a = build()
    boom = BoomCtx(ctx._tables, ctx.params, ctx.armed, ctx.t_end)
    fs = det_power.run(boom, a)
    check([f.id for f in fs] == IDS, [f.id for f in fs])
    for f in fs:
        check(f.status == NA and f.context == "error" and "RuntimeError" in f.values["error"], f"{f.id}: {f.status}/{f.context}")


# ---------------------------------------------------------------------------
# 5 真實 124 資料注入故障
# ---------------------------------------------------------------------------
class PatchedCtx:
    """包住真實 Ctx,只換掉某些表。"""
    def __init__(self, real, **tables):
        self._real = real
        self._over = tables

    def table(self, name, instance=None):
        return self._over[name] if name in self._over else self._real.table(name, instance)

    def __getattr__(self, k):
        return getattr(self._real, k)


def test_real_data_fault_injection():
    real = crashkit.load_ctx("00000124")
    a = crashkit.truth_anchors_124(real)
    bat = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in real.table("BAT").items()}
    t = bat["t"]

    # (a) 1200..1230 s 把電壓整體壓低 8 V → 低電壓 HIT(V1 最低約 37 V < 39.6 V),起點約 1200 s
    b1 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in bat.items()}
    b1["Volt"][(t >= 1200.0) & (t < 1230.0)] -= 8.0
    d = by_id(det_power.run(PatchedCtx(real, BAT=b1), a))
    f = d["power_low_voltage"]
    check(f.status == HIT and f.values["crit"] is True, f"{f.status} {f.values}")
    near(f.t_onset, 1200.0, 3.0, "注入低電壓起點")
    check(f.values["v1_min"] < 39.6, f.values["v1_min"])
    check(f.t_onset < a.t_warn, "起點早於 t_warn → engine 會貼 precursor")

    # (b) 1230.0 s 起 5 秒電壓掉到 20 V(0.4 秒掉)→ 崩落 HIT
    b2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in bat.items()}
    w = (t >= 1230.0) & (t < 1235.0)
    b2["Volt"][w] = 20.0
    d = by_id(det_power.run(PatchedCtx(real, BAT=b2), a))
    f = d["power_voltage_collapse"]
    check(f.status == HIT and f.context == "in_flight", f"{f.status} {f.context}")
    near(f.t_onset, 1230.0, 0.5, "注入崩落起點")
    check(f.values["drop_v"] > 20.0, f.values)

    # (c) 同一份注入但發生在飛行 1(解鎖區間之外)對飛行 2 無影響;飛行 1 的錨點則會抓到
    a1 = crashkit.flight1_anchors_124(real)
    b3 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in bat.items()}
    b3["Volt"][(t >= 400.0) & (t < 430.0)] -= 9.0
    check(by_id(det_power.run(PatchedCtx(real, BAT=b3), a))["power_low_voltage"].status == CLEAR, "飛行 2 不受飛行 1 區段影響")
    check(by_id(det_power.run(PatchedCtx(real, BAT=b3), a1))["power_low_voltage"].status == HIT, "飛行 1 區段內注入應抓到")

    # (d) 電源軌:Vcc 在 1200 s 掉到 4.2 V
    pw = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in real.table("POWR").items()}
    pw["Vcc"][(pw["t"] >= 1200.0) & (pw["t"] < 1201.0)] = 4.2
    f = by_id(det_power.run(PatchedCtx(real, POWR=pw), a))["power_rail_fault"]
    check(f.status == HIT and f.values["conditions"] == ["vcc_low"], f"{f.status} {f.values['conditions']}")
    near(f.t_onset, 1200.0, 0.15, "注入 Vcc 掉電起點")


class ParamCtx(PatchedCtx):
    """PatchedCtx + 換掉參數表。"""
    def __init__(self, real, params, **tables):
        super().__init__(real, **tables)
        self._params = params

    def param(self, name, default=None):
        return self._params.get(name, default)


def test_real_data_review_injections():
    """審查回歸(真實 124 飛行 2 + truth 錨點,t_loss = 1258.10):
    (a) 1200 s 單一 0 V 壞樣本 → SOFT 'transient'(舊版:critical 'in_flight' HIT);
    (b) 1258.6..1261.5 s 下陷 7 V(失控後)→ 低電壓 HIT,engine 標 consequence,文字寫「屬後果」不寫「可能導致失控」;
    (c) 1258.45 s 起電池掉到 15 V 且 log 繼續(翻滾中電池脫落)→ 崩落 HIT 'on_ground'(舊版:CLEAR「落地後拔電池」),標 consequence;
    (d) AccFlags bit3 從 100 s(未解鎖)起累計 → 電源軌 SOFT 'pre_arm',t_onset 100.1 s,標 ground_info(舊版:873.2 s HIT);
    (e) MOT_BAT_VOLT_* 歸零(ArduPilot 預設)且 BAT.Volt 校正偏高 2% → 仍估成 12 串、CLEAR(舊版:13 串、門檻 42.9 V)。"""
    from core.crash import engine
    from core.crash.types import TAG_CONSEQUENCE, TAG_GROUND
    real = crashkit.load_ctx("00000124")
    a = crashkit.truth_anchors_124(real)
    bat = real.table("BAT")
    t = bat["t"]

    def bat_copy():
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in bat.items()}

    b = bat_copy()
    b["Volt"][int(np.searchsorted(t, 1200.0))] = 0.0
    d = by_id(det_power.run(PatchedCtx(real, BAT=b), a))
    f = d["power_voltage_collapse"]
    check(f.status == SOFT and f.context == "transient" and f.values["transient_samples"] == 1, f"(a) {f.status}/{f.context}")
    near(f.t_onset, 1199.94, 0.15, "(a) 壞樣本起點")
    check(d["power_low_voltage"].status == CLEAR, "(a) 低電壓不受影響")

    b = bat_copy()
    b["Volt"][(t >= 1258.6) & (t < 1261.5)] -= 7.0
    f = by_id(det_power.run(PatchedCtx(real, BAT=b), a))["power_low_voltage"]
    check(f.status == HIT and f.t_onset >= a.t_loss and f.values["after_loss"] is True, f"(b) {f.status} {f.t_onset}")
    check(engine.tag_time(f.t_onset, a) == TAG_CONSEQUENCE, "(b) 失控後 → 後果")
    check("屬後果" in f.text_zh and "可能導致失控" not in f.text_zh, f"(b) {f.text_zh}")

    b = bat_copy()
    b["Volt"][(t >= 1258.45) & (t <= 1262.2)] = 15.0
    f = by_id(det_power.run(PatchedCtx(real, BAT=b), a))["power_voltage_collapse"]
    check(f.status == HIT and f.context == "on_ground" and f.values["on_ground"] is True, f"(c) {f.status}/{f.context}")
    near(f.t_onset, 1258.44, 0.15, "(c) 掉壓起點")
    check(engine.tag_time(f.t_onset, a) == TAG_CONSEQUENCE and "屬後果" in f.text_zh and "不是飛行中掉電" not in f.text_zh, f"(c) {f.text_zh}")

    pw = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in real.table("POWR").items()}
    pw["AccFlags"][pw["t"] >= 100.0] = 9.0
    f = by_id(det_power.run(PatchedCtx(real, POWR=pw), a))["power_rail_fault"]
    check(f.status == SOFT and f.context == "pre_arm" and f.values["conditions"] == [], f"(d) {f.status}/{f.context}")
    near(f.t_onset, 100.1, 0.15, "(d) 累計旗標真正第一次出現")
    check(engine.tag_time(f.t_onset, a) == TAG_GROUND, "(d) 解鎖前 → 地面資訊")

    p = dict(real.params, MOT_BAT_VOLT_MAX=0.0, MOT_BAT_VOLT_MIN=0.0)
    for gain in (1.0, 1.02):
        b = bat_copy()
        b["Volt"] = b["Volt"] * gain
        f = det_power.power_low_voltage(ParamCtx(real, p, BAT=b), a)[0]
        check(f.status == CLEAR and f.values["cells"] == 12 and f.values["cells_src"] == "estimated", f"(e) ×{gain}: {f.status} {f.values['cells']}")
        near(f.values["v_crit_cfg"], 39.6, 0.01, f"(e) ×{gain} 估計門檻")
        check("估計 12 串" in f.text_zh and "MOT_BAT_VOLT_MIN 39.6" not in f.text_zh, f.text_zh)


# ---------------------------------------------------------------------------
# 6 通則:id / JSON / 中文 / 效能
# ---------------------------------------------------------------------------
def _scenarios():
    real = crashkit.load_ctx("00000124")
    yield "124 truth", det_power.run(real, crashkit.truth_anchors_124(real))
    yield "124 flight1", det_power.run(real, crashkit.flight1_anchors_124(real))
    yield "124 bare", det_power.run(real, basic_anchors(real))
    iv = real.armed[1]
    yield "124 flight2 <1255", det_power.run(real, Anchors(interval=ArmedInterval(1, iv.t0, 1255.0, 0), intervals=list(real.armed)))
    for n in crashkit.BASELINES:
        c = crashkit.load_ctx(n)
        yield n, det_power.run(c, crashkit.not_armed_anchors(c))
    ctx, a = build()
    yield "synth healthy", det_power.run(ctx, a)
    ctx, a = build(usb=True)
    yield "synth usb", det_power.run(ctx, a)
    ctx, a = build(lambda t: 46.0 - 21.0 * np.clip((t - 90.0) / 0.5, 0, 1), err=[(100.0, 6, 1)])
    yield "synth collapse+err6", det_power.run(ctx, a)
    ctx, a = build(curr_fn=lambda t, thr: np.full(t.size, 0.004), params={"MOT_THST_HOVER": None}, flags_fn=lambda t: np.full(t.size, 2.0))
    yield "synth bad-curr no-hover-param no-brick", det_power.run(ctx, a)
    ctx, a = build(arm=(0.5, 140.0))
    yield "synth no-rest-voltage", det_power.run(ctx, a)
    ctx, a = build(lambda t: np.where(np.isclose(t, 100.0), 0.0, 46.0), acc_fn=lambda t: np.where(t >= 5.0, 9.0, 1.0))
    ctx._tables["POWR"]["Vcc"][500:510] = np.nan
    yield "synth glitch + pre-arm AccFlags + Vcc NaN", det_power.run(ctx, a)
    ctx, a = build(_lihv_volt(4, 4.35, 3.75), cells=4, params={"MOT_BAT_VOLT_MAX": 0.0, "MOT_BAT_VOLT_MIN": 0.0})
    yield "synth LiHV 4S params unset", det_power.run(ctx, a)
    ctx, a = build(lambda t: np.where(t < 110.0, 46.0, 15.0), tho_fn=lambda t: np.where(t < 100.0, 0.15, 0.0))
    a.t_warn, a.t_loss = 100.0, 105.0
    yield "synth on-ground collapse after t_loss", det_power.run(ctx, a)


def test_every_id_in_every_scenario_json_and_zh():
    cjk = re.compile(r"[一-鿿]")
    n = 0
    for label, fs in _scenarios():
        check([f.id for f in fs] == IDS, f"{label}: {[f.id for f in fs]}")
        for f in fs:
            n += 1
            check(f.domain == "power" and f.status in STATUSES and f.severity in SEVERITIES, f"{label} {f.id}")
            json.dumps(f.values, allow_nan=False)                                   # 可序列化、無 NaN
            check(cjk.search(f.text_zh) and len(f.text_zh) > 8, f"{label} {f.id} 缺中文說明")
            check("nan" not in f.text_zh.lower(), f"{label} {f.id} 文字有 nan: {f.text_zh}")
            check(f.tag == "", f"{label} {f.id}: tag 由 engine 貼,偵測器不貼")
            if f.status in (HIT, SOFT):
                check(f.t_onset is not None and f.t_alarm is not None and f.t_alarm >= f.t_onset - 1e-9, f"{label} {f.id} 時間")
                check(f.severity != "info", f"{label} {f.id}")
            if f.status == NA:
                check(f.context != "", f"{label} {f.id} NA 要有原因 context")
            for v in f.values.values():
                check(not isinstance(v, (np.floating, np.integer, np.ndarray)), f"{label} {f.id}: numpy 型別外洩")
    print(f"  {n} 筆 Finding:id / JSON / 中文 / 時間欄位都符合")


def test_texts_never_report_current_derived_claims():
    """A8:不得描述 BAT.Curr 衍生量(內阻 / 剩餘電量 / 老化);不得把電源說成失事原因。"""
    bad = re.compile(r"內阻|老化|剩餘電量|電量剩餘")
    for label, fs in _scenarios():
        for f in fs:
            check(not bad.search(f.text_zh), f"{label} {f.id}: {f.text_zh}")
    ctx = crashkit.load_ctx("00000124")
    d = by_id(det_power.run(ctx, crashkit.truth_anchors_124(ctx)))
    check(d["power_low_voltage"].status == CLEAR and "可排除電池耗盡" in d["power_low_voltage"].text_zh, d["power_low_voltage"].text_zh)
    for i in CRASH_IDS:
        check(d[i].status == CLEAR, f"{i} 應排除")
    for i in HEALTH_IDS:
        check("與失事原因無直接關係" in d[i].text_zh or "不是失事原因" in d[i].text_zh or "而非失事原因" in d[i].text_zh, f"{i}: 一般健檢要註明非失事原因: {d[i].text_zh}")


def test_performance_per_detector():
    ctx = crashkit.load_ctx("00000124")
    for label, a in (("truth", crashkit.truth_anchors_124(ctx)), ("flight1", crashkit.flight1_anchors_124(ctx))):
        for fn in (det_power.power_low_voltage, det_power.power_voltage_collapse, det_power.power_rail_fault,
                   det_power.power_data_quality, det_power.power_failsafe_config_audit):
            t0 = time.perf_counter()
            fn(ctx, a)
            dt = time.perf_counter() - t0
            check(dt < 1.5, f"{label} {fn.__name__} 太慢: {dt:.2f}s")
    t0 = time.perf_counter()
    det_power.run(ctx, crashkit.truth_anchors_124(ctx))
    print(f"  run() 全部 5 個偵測器(124 飛行 2):{(time.perf_counter() - t0) * 1000:.0f} ms")


def test_deterministic():
    ctx = crashkit.load_ctx("00000124")
    a = crashkit.truth_anchors_124(ctx)
    x = [(f.id, f.status, f.t_onset, json.dumps(f.values, sort_keys=True), f.text_zh) for f in det_power.run(ctx, a)]
    y = [(f.id, f.status, f.t_onset, json.dumps(f.values, sort_keys=True), f.text_zh) for f in det_power.run(ctx, a)]
    check(x == y, "同樣輸入應得同樣輸出")


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
