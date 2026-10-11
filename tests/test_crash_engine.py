"""失事分析 · 引擎端到端驗證:python -m tests.test_crash_engine 或 pytest tests/test_crash_engine.py

(1) 00000124 飛行 2(失事):終點 / 型態 / 子型態 / 關鍵時間 / 假說排名與信心 / 機制文字 / 後果標籤 /
    已排除項目的數字 / 不得宣稱的原因(acceptance A1–A8)
(2) 00000124 飛行 1(正常降落,interval_index=0):不排名、沒有失事字眼、相關偵測器 CLEAR(acceptance B)
(3) 飛行 2 在 1255 s 之前沒有任何非持續性的異常(acceptance B 末句)
(4) 6 份從未解鎖的 log:未解鎖 / 不做失事分析 / 不適用 / 地面事件只當資訊(acceptance C)
(5) 跨切面:標籤與計分規則、確定性、JSON、效能 < 10 s(acceptance D4/D5/D7)
(6) 合成情境(假偵測模組 + 合成 Ctx):證明評分 / 閘門 / 型態 / 判定邏輯不是寫死 124 的數字
(7) 審查修正的回歸:後果邊界(撞擊早於 t_loss / 沒有 t_loss)、自動選最後一次異常解鎖(含 124 去掉 CRASH_CHECK +
    地面重新解鎖的變形)、沒有錨點時型態不明(含 124 截斷在 1250 s)、證據不足只列候選、模組載入 / 執行失敗、
    閘門缺資料不預設通過、結束型態不明不排名、報告用字 / 飛手與 RC9 時間軸 / t_abs / Markdown
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crashkit  # noqa: E402
from core.crash import engine as E  # noqa: E402
from core.crash import hypotheses as H  # noqa: E402
from core.crash import report as R  # noqa: E402
from core.crash.ctx import Ctx  # noqa: E402
from core.crash.types import (CLEAR, HIT, NA, SOFT, TAG_CONSEQUENCE, TAG_DEVELOPMENT, TAG_GROUND,  # noqa: E402
                              TAG_POST, TAG_PRECURSOR, TAG_STANDING, Anchors, ArmedInterval, Finding)

CRASH = crashkit.CRASH_LOG
HYP_IDS = [h.id for h in H.HYPOTHESES]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def near(x, ref, tol, what):
    check(x is not None and abs(float(x) - ref) <= tol, f"{what} = {x}(預期 {ref} ± {tol})")


_REP: dict = {}


def rep124(idx=None):
    """00000124 的報告(同一行程只算一次)。idx=None 自動選(飛行 2),0 = 飛行 1。"""
    if idx not in _REP:
        _REP[idx] = E.analyze_ctx(crashkit.load_ctx(CRASH), path=crashkit.real_path(CRASH), interval_index=idx)
    return _REP[idx]


def hyp(rep, hid):
    h = rep.hypothesis(hid)
    check(h is not None, f"沒有假說 {hid}")
    return h


def fnd(rep, det_id, status=None):
    fs = [f for f in rep.findings if f.id == det_id and (status is None or f.status == status)]
    check(fs, f"沒有 {det_id}({status or '任意狀態'})的 Finding")
    return fs


def tl(rep, pred):
    return [e for e in rep.timeline if pred(e)]


# =============================================================== (1) 00000124 飛行 2

def test_124_end_type_and_key_times():
    """A1 / A4:CRASH_CHECK 終點、重算觸發、撞擊與判定延遲、各關鍵時間。"""
    r = rep124()
    a = r.anchors
    check(r.end_type == "crash_check", f"end_type={r.end_type}")
    check(r.interval["index"] == 1 and r.interval["method"] == 12, f"分析區間 {r.interval}")
    near(a.t_detect, 1262.125, 0.05, "t_detect(ERR 12/1)")
    rp = fnd(r, "crash_check_replay", HIT)[0]
    near(rp.t_alarm, a.t_detect, 0.05, "crash_check_replay 觸發時間")
    near(rp.values.get("ae_at_fire_deg"), 150.0, 2.0, "重算 AngErr")
    near(a.t_impact_peak, 1259.50, 0.1, "t_impact_peak")
    lag = a.t_detect - a.t_impact_peak
    check(2.4 <= lag <= 2.8, f"飛控判定延遲 {lag:.2f} s(預期 2.4–2.8)")
    near(a.t_touch, 1255.225, 0.1, "t_touch")
    near(a.t_warn, 1255.9, 0.15, "t_warn")
    near(a.t_loss, 1258.10, 0.1, "t_loss")
    near(a.t_first_hit, 1258.40, 0.1, "t_first_hit")
    check(a.t_first_anomaly == a.t_touch, f"t_first_anomaly {a.t_first_anomaly} ≠ t_touch {a.t_touch}")
    # A1:最後一次撞擊(倒扣、約 88.7 m/s² 濾波值、dV 6.9、591°/s);A1 訊息在時間軸
    last = [f for f in fnd(r, "impact_shock_clusters", HIT) if f.context == "hard_impact"][-1]
    near(last.values["peak"], 88.7, 2.0, "最後撞擊峰值")
    near(last.values["dV"], 6.9, 0.3, "最後撞擊 dV")
    near(last.values["gyro_peak_dps"], 591.0, 15.0, "最後撞擊角速度")
    check(last.values["dir"] == "top", f"最後撞擊方向 {last.values['dir']}(應為倒扣 top)")
    check(tl(r, lambda e: "AngErr=150>30, Accel=0.1<3.0" in e.label and abs(e.t - 1262.125) < 0.05), "時間軸缺 Crash 訊息")
    # A4:油門收到怠速 1255.25–1255.54、置中搖桿傾斜 1255.83 起峰值 24–26°、傾角 ≥120° 於 1259.2–1259.3
    tn = fnd(r, "touchdown_not_recognised", HIT)[0]
    near(tn.values["t_tho_drop"], 1255.25, 0.1, "ThO 開始下降")
    near(tn.values["t_tho_zero"], 1255.54, 0.1, "ThO 到怠速")
    lean = fnd(r, "lean_without_pilot", HIT)[0]
    near(lean.t_onset, 1255.83, 0.15, "置中搖桿傾斜起點")
    check(24.0 <= lean.values["lean_peak"] <= 26.0, f"傾斜峰值 {lean.values['lean_peak']}")
    att = fnd(r, "attitude_loss_of_control", HIT)[0]
    check(1259.2 <= att.values["t_tilt120"] <= 1259.3, f"傾角 ≥120° 時間 {att.values['t_tilt120']}")
    near(r.t_base, 8.1593, 0.001, "t_base")
    check(f"log 絕對 {a.t_detect + r.t_base:.3f} s" in r.format_text(), "關鍵時間沒有附上 log 絕對秒")
    print(f"  終點 crash_check;觸地 {a.t_touch:.3f}、預警 {a.t_warn:.3f}、失控 {a.t_loss:.3f}、"
          f"撞擊 {a.t_first_hit:.2f}/{a.t_impact_peak:.2f}、判定 {a.t_detect:.3f}(延遲 {lag:.2f} s)")


def test_124_disarm_events_not_landing():
    """A1:上鎖那一刻補寫的 EV17/18/11 不算落地(時間軸不列、touchdown 仍判定未被辨識)。"""
    r = rep124()
    td = r.anchors.t_disarm
    bad = tl(r, lambda e: "LAND_COMPLETE" in e.label and td - 0.001 <= e.t <= td + 0.01)
    check(not bad, f"上鎖時補寫的落地事件被列進時間軸:{[(e.t, e.label) for e in bad]}")
    tn = fnd(r, "touchdown_not_recognised", HIT)[0]
    near(tn.values.get("no_ev_s"), 6.9, 0.2, "觸地後無 EV17/18 的時間")
    check(r.scenario_features.get("landed") is False, "預警時飛控應仍認為在空中")


def test_124_scenario():
    """A2:GROUND_TIPOVER(LOITER 手動下降),子型態 autopilot_lean + land_not_declared;翻覆前沒有自由落體。"""
    r = rep124()
    check(r.scenario == "GROUND_TIPOVER", f"scenario={r.scenario}")
    check("autopilot_lean" in r.scenario_sub and "land_not_declared" in r.scenario_sub, f"子型態 {r.scenario_sub}")
    check("pilot_stick" not in r.scenario_sub, "預警前沒有飛手大動作,不應有 pilot_stick")
    f = r.scenario_features
    near(f.get("agl_m"), -0.13, 0.2, "觸地 agl")
    near(f.get("vz_pre_cms"), -30.0, 20.0, "觸地前 5 s 垂直速度(cm/s)")
    check(f.get("mode") == "LOITER", f"模式 {f.get('mode')}")
    for ff in fnd(r, "free_fall_run"):
        check(ff.status not in (HIT, SOFT) or ff.t_onset > r.anchors.t_warn,
              f"自由落體 {ff.t_onset} 應在翻覆途中(> t_warn)")
    print(f"  型態 {r.scenario} {r.scenario_sub};agl {f['agl_m']} m、vz_pre {f['vz_pre_cms']} cm/s")


def test_124_ranking():
    """A3:第一名 H_LAND_TIPOVER(高)、第二名 H_NAV_FAILURE 貢獻因素(中)、其餘 0 分且已排除 / 不適用。"""
    r = rep124()
    check(r.verdict == "ranked", f"verdict={r.verdict} {r.verdict_note}")
    top = r.top()
    check(top is not None and top.id == "H_LAND_TIPOVER", f"第一名 {top and top.id}")
    check(top.confidence == "高", f"第一名信心 {top.confidence}")
    nav = hyp(r, "H_NAV_FAILURE")
    check(nav.status == "contributing" and nav.confidence == "中", f"H_NAV {nav.status} {nav.confidence}")
    check(r.hypotheses[0].id == "H_LAND_TIPOVER" and r.hypotheses[1].id == "H_NAV_FAILURE", "排名順序")
    land = hyp(r, "H_LAND_TIPOVER")
    check(land.score >= 12.0, f"S(H_LAND)={land.score:.1f} < 12")
    ratio = nav.score / land.score
    check(0.35 <= ratio <= 0.7, f"S(NAV)/S(LAND)={ratio:.2f}(預期 0.35–0.7)")
    for h in r.hypotheses:
        if h.id in ("H_LAND_TIPOVER", "H_NAV_FAILURE"):
            continue
        check(h.score == 0.0, f"{h.id} 不應有分數:{h.score}")
        check(h.status in ("ruled_out", "not_applicable"), f"{h.id} 狀態 {h.status}")
        if h.status == "ruled_out":
            check(h.ruled_out_basis and any(re.search(r"\d", b) for b in h.ruled_out_basis), f"{h.id} 已排除但沒有數字依據")
    print(f"  H_LAND {land.score:.1f}(高)、H_NAV {nav.score:.1f}(中,比值 {ratio:.2f});其餘 0 分")


def test_124_mechanism_text():
    """A5:機制文字要提到未判定落地、搖桿置中的 Loiter 傾斜與 EKF 漂移、VIO 中斷、怠速馬達差、重力翻倒、
    VIO-only 來源組與 GPS 無定位(貢獻因素,中),以及「吻合但未證實」的上游觸發候選。"""
    r = rep124()
    text = r.format_text()
    for kw in ("LAND_COMPLETE_MAYBE", "搖桿置中", "Loiter", "1.16", "220 µs", "重力", "VELXY", "934", "吻合", "未證實"):
        check(kw in text, f"報告沒有提到「{kw}」")
    check("信心:中" in r.summary_zh and "貢獻因素" in r.summary_zh, "摘要沒有把導航列為貢獻因素(信心中)")
    ph = fnd(r, "ekf_phantom_motion_on_ground", HIT)[0]
    near(ph.values.get("ekf_drift_since_touch_m"), 1.7, 0.3, "觸地後 EKF 漂移(m)")
    check(ph.values.get("vio_drift_since_touch_m", 9) < 0.2, "VIO 位移應遠小於 EKF 漂移")
    lat = fnd(r, "touchdown_lateral_speed", HIT)[0]
    check(0.5 <= lat.values["v_ekf"] <= 1.2 and 0.5 <= lat.values["v_vio"] <= 1.2, f"觸地水平速度 {lat.values}")
    idle = fnd(r, "idle_motor_asymmetry", HIT)[0]
    near(idle.values.get("spread_med_us"), 220.0, 15.0, "怠速馬達 PWM 全距")
    # 事件鏈依時間排序,第一名的證據沒有後果
    timed = [c for c in r.chain if c["tag"] != TAG_STANDING]
    check(timed == sorted(timed, key=lambda c: c["t"]), "事件鏈沒有依時間排序")
    check(all(c["tag"] in (TAG_PRECURSOR, TAG_DEVELOPMENT, TAG_STANDING) for c in r.chain), "事件鏈混入後果")
    check(any(c["role"] == "contributing" for c in r.chain), "事件鏈沒有標出貢獻因素的項目")


def test_124_consequences_labelled():
    """A6:失控之後 / 上鎖之後的事件都標成後果 / 事後;753–815 s 的地面事件只當資訊。"""
    r = rep124()

    def tag_near(t, pred, tags, tol=0.15, what=""):
        hits = tl(r, lambda e: abs(e.t - t) <= tol and pred(e))
        check(hits, f"時間軸找不到 {what}@{t}")
        check(all(e.tag in tags for e in hits), f"{what}@{t} 標籤 {[e.tag for e in hits]}(應為 {tags})")

    cons = (TAG_CONSEQUENCE,)
    post = (TAG_POST,)
    pil = fnd(r, "pilot_input_context", HIT)[0]
    near(pil.t_onset, 1257.84, 0.15, "飛手大動作")
    check(pil.tag == TAG_CONSEQUENCE, f"飛手大動作標籤 {pil.tag}")
    tag_near(1258.84, lambda e: e.category == "ev" and "SECONDARY" in e.label, cons, what="EV86")
    tag_near(1259.64, lambda e: e.category == "ev" and "TERTIARY" in e.label, cons, what="EV87")
    tag_near(1258.94, lambda e: e.category == "err" and "GPS" in e.label, cons, what="GPS glitch")
    tag_near(1259.06, lambda e: "stale" in e.label, cons, what="VIO stale")
    last_visp = [f for f in fnd(r, "vio_dropout", HIT) if abs(f.t_onset - 1258.146) < 0.05]
    check(last_visp and last_visp[0].tag == TAG_CONSEQUENCE, "最後一筆 VISP 1258.146 應為後果")
    sm = fnd(r, "ekf_mag_reject", HIT)[0]
    near(sm.t_onset, 1259.70, 0.15, "SM>1")
    check(sm.tag == TAG_CONSEQUENCE, f"SM>1 標籤 {sm.tag}")
    to = [f for f in fnd(r, "ekf_aiding_timeout", HIT)]
    check(to and all(f.tag == TAG_POST for f in to) and 1264.8 <= to[0].t_onset <= 1265.5, f"EKF 輔助逾時 {[(f.t_onset, f.tag) for f in to]}")
    tag_near(1266.24, lambda e: e.category == "err" and "EKFCHECK" in e.label, post, what="EKFCHECK")
    tag_near(1266.24, lambda e: e.category == "err" and "FAILSAFE_EKFINAV" in e.label, post, what="FAILSAFE_EKFINAV")
    tag_near(1273.24, lambda e: "stopped aiding" in e.label, post, tol=0.05, what="stopped aiding")
    tag_near(1266.89, lambda e: "Arm: Need Position" in e.label, post, what="GCS 解鎖拒絕")
    tag_near(1273.48, lambda e: "Arm: Need Position" in e.label, post, what="GCS 解鎖拒絕")
    gnd = [e for e in r.ground_events if 753.0 <= e.t <= 815.5]
    check(any("Glitch" in e.label for e in gnd) and any("EKFCHECK" in e.label for e in gnd), "753–815 s 地面事件沒列出")
    check(all(e.tag == TAG_GROUND for e in gnd), "753–815 s 事件應標 ground_info")
    check(not tl(r, lambda e: 740.0 <= e.t <= 820.0), "753–815 s(上鎖中)的事件不應進入失事時間軸")


def test_124_ruled_out_numbers():
    """A7:已排除項目附數字;資料品質限制把電源 / 馬達的排除信心封頂為中;EKF 健康數字列在導航假說的已檢查正常。"""
    r = rep124()
    text = r.format_text()
    mot = hyp(r, "H_MOTOR_THRUST")
    b = " ".join(mot.ruled_out_basis)
    check(mot.status == "ruled_out" and "1566" in b and "0.44" in b, f"馬達排除依據 {b[:120]}")
    check(mot.rule_out_conf == "中" and "遙測" in mot.rule_out_note, "沒有 ESC 遙測 → 馬達排除信心應為中")
    pw = hyp(r, "H_POWER")
    b = " ".join(pw.ruled_out_basis)
    check(pw.status == "ruled_out" and "44.8" in b and "3.73" in b and "5.18" in b, f"電源排除依據 {b[:160]}")
    check(pw.rule_out_conf == "中" and "BAT.Curr" in pw.rule_out_note, "BAT.Curr 無效 → 電源排除信心應為中")
    vib = hyp(r, "H_VIBRATION_IMU")
    check(vib.status == "ruled_out" and vib.rule_out_conf == "高", f"振動 {vib.status} {vib.rule_out_conf}")
    vf = fnd(r, "vibration_flight_level", CLEAR)[0]
    check(vf.values["p95"] <= 10.1 and vf.values.get("clip_delta", 0) == 0, f"事件前振動 p95 {vf.values['p95']}")
    check(any("IMU" in x for x in vib.ruled_out_basis), "振動排除依據缺 IMU 一致性")
    cmp_ = hyp(r, "H_COMPASS_YAW")
    check(cmp_.status == "ruled_out" and "882.5" in " ".join(cmp_.ruled_out_basis), "羅盤排除依據缺最後一次航向重置")
    raw = fnd(r, "compass_raw_consistency", HIT)[0]
    near(raw.t_onset, 1258.04, 0.15, "第一個羅盤警報")
    check(raw.t_onset > r.anchors.t_warn and raw.tag == TAG_CONSEQUENCE, "羅盤警報應晚於 t_warn 且為後果")
    nav = hyp(r, "H_NAV_FAILURE")
    ok = " ".join(nav.checked_ok)
    check("0.21" in ok and "3.2°" in ok, f"導航假說的已檢查正常缺 EKF 健康數字:{ok[:160]}")
    pil = hyp(r, "H_CONTROL_PILOT")
    check(pil.status == "ruled_out" and any("29%" in x for x in pil.ruled_out_basis), "飛手排除依據缺事件前搖桿數字")
    fs = hyp(r, "H_FAILSAFE_ACTION")
    check(fs.status == "ruled_out" and "1258.94" in " ".join(fs.ruled_out_basis), "失效保護排除依據缺 1258.94")
    check("排除信心:中" in text and "排除信心:高" in text, "報告沒有列排除信心")


def test_124_must_not_claim():
    """A8:電池 / 馬達 / 振動 / 羅盤 / GPS / 失效保護 / 飛手 / IMU 都不得成為原因;不得描述 BAT.Curr 衍生量;
    保養項目只出現在「非本次失事原因」段落。"""
    r = rep124()
    text = r.format_text()
    live = {h.id for h in r.hypotheses if h.score > 0}
    check(live == {"H_LAND_TIPOVER", "H_NAV_FAILURE"}, f"有分數的假說 {live}")
    for bad in ("內阻", "剩餘電量", "電池老化"):
        check(bad not in text, f"報告不得出現「{bad}」")
    for h in r.hypotheses:
        for e in h.evidence:
            check(not (e["det"] == "source_switch_audit" and e["contrib"] > 0), "來源組切換不得計分")
    hdr = text.index("非本次失事原因")
    cfg = fnd(r, "power_failsafe_config_audit", HIT)[0]
    pos = text.find(cfg.text_zh)
    check(pos > hdr, "電池保護參數稽核只能列在「非本次失事原因」段落")
    mag = fnd(r, "compass_motor_interference")[0]
    check(text.find(mag.text_zh) > hdr, "MAG[1] 干擾只能列在「非本次失事原因」段落")
    check(not any(o["id"] in ("touchdown_not_recognised", "idle_motor_asymmetry") for o in r.observations),
          "計分的證據不應出現在其他觀察")


def test_124_tags_and_scoring_rules():
    """D4:每筆 Finding 都有標籤;有證據的非持續性 Finding 都有 t_onset;後果 / 事後 / 地面事件不計分。"""
    for idx in (None, 0):
        r = rep124(idx)
        for f in r.findings:
            check(f.tag is not None, f"{f.id} 沒有標籤")
            if f.status in (HIT, SOFT) and f.tag != TAG_STANDING:
                check(f.t_onset is not None, f"{f.id} 有證據卻沒有 t_onset")
            check(f.status != HIT or f.tag != "" or r.verdict == "no_failure", f"{f.id} HIT 卻沒有因果標籤")
        for h in r.hypotheses:
            for e in h.evidence:
                if e["tag"] in (TAG_CONSEQUENCE, TAG_POST, TAG_GROUND):
                    check(e["contrib"] == 0.0, f"{h.id} 的 {e['det']}({e['tag']})計了分")
            check(abs(h.raw - sum(e["contrib"] for e in h.evidence)) < 1e-9, f"{h.id} 分數 ≠ 證據加總")


def test_124_flight2_quiet_before_1255():
    """B 末句:飛行 2 在 1255.0 s 之前,相關偵測器沒有 HIT;其他非持續性證據最早只到觸地前 1 s 的視窗。"""
    r = rep124()
    ids = ("touchdown_not_recognised", "lean_without_pilot", "idle_motor_asymmetry", "motor_thrust_loss_check",
           "motor_thrust_imbalance", "yaw_imbalance_iterm", "power_low_voltage", "power_voltage_collapse",
           "power_rail_fault", "rate_demand_not_achieved", "rate_output_saturation")
    for f in r.findings:
        if f.id in ids and f.status in (HIT, SOFT):
            check(f.t_onset is not None and f.t_onset >= 1255.0, f"{f.id} 在 {f.t_onset} 就有證據")
        if f.status in (HIT, SOFT) and f.tag != TAG_STANDING:
            check(f.t_onset >= 1254.0, f"{f.id}({f.status})在 {f.t_onset} 就有非持續性證據")
    lean = fnd(r, "lean_without_pilot", HIT)[0]
    check(lean.values["air_lean_sustained_max"] < 12.0, f"空中置中搖桿傾斜 {lean.values['air_lean_sustained_max']}")
    rd = fnd(r, "rate_demand_not_achieved", HIT)[0]
    check(max(rd.values["max_rmsd_normal_roll"], rd.values["max_rmsd_normal_pitch"]) <= 22.5 + 0.05, "正常飛行 rms 指令")


# =============================================================== (2) 00000124 飛行 1

def test_124_flight1_normal_landing():
    """B:飛行 1 = normal_landing、不排名、沒有失事字眼;相關偵測器 CLEAR;時間軸不貼先兆 / 後果。"""
    r = rep124(0)
    check(r.interval["index"] == 0, "應分析第 1 次解鎖")
    check(r.end_type == "normal_landing" and r.verdict == "no_failure", f"{r.end_type} / {r.verdict}")
    check(r.hypotheses == [] and r.chain == [] and r.scenario == "", "正常降落不應有原因排名 / 事件鏈 / 型態")
    check(r.anchors.t_warn is None and r.anchors.t_loss is None and r.anchors.t_first_anomaly is None, "不應有失控錨點")
    for det in ("touchdown_not_recognised", "lean_without_pilot", "idle_motor_asymmetry", "motor_thrust_loss_check",
                "motor_thrust_imbalance", "yaw_imbalance_iterm", "power_low_voltage", "power_voltage_collapse",
                "power_rail_fault", "rate_demand_not_achieved", "rate_output_saturation", "attitude_loss_of_control",
                "crash_check_replay", "crash_check_anchor"):
        st = {f.status for f in fnd(r, det)}
        check(st == {CLEAR}, f"飛行 1 {det} 狀態 {st}")
    td = fnd(r, "touchdown_not_recognised")[0]
    near(td.values.get("dt_ev17_s"), 0.82, 0.05, "觸地 → EV17")
    idle = fnd(r, "idle_motor_asymmetry")[0]
    check(idle.values["spread_p95_us"] <= 57.0, f"怠速全距 p95 {idle.values['spread_p95_us']}")
    check(fnd(r, "lean_without_pilot")[0].values["ground_lean_sustained_max"] <= 2.2, "怠速傾斜")
    check(fnd(r, "attitude_loss_of_control")[0].values["ae_max"] <= 3.6, "ae max")
    body = "\n".join(ln for ln in r.format_text().splitlines() if not ln.startswith("  第 "))   # 區間清單列出飛行 2 是事實
    for bad in ("墜機", "翻覆", "CRASH_CHECK", "失控", "假說排名", "先兆", "後果"):
        check(bad not in body and bad not in r.summary_zh, f"飛行 1 報告出現「{bad}」")
    check(all(e.tag in ("", TAG_GROUND) for e in r.timeline), "正常飛行的時間軸不應有因果標籤")
    check(any("LAND_COMPLETE" == e.label for e in r.timeline), "時間軸應列出正常的落地事件")
    print(f"  飛行 1:{r.end_type},健康觀察 {len(r.observations)} 項")


# =============================================================== (4) 未解鎖的 log

_GROUND_EXPECT = {
    "00000117": [("FAILSAFE_GCS", 611.09)],
    "00000118": [("EKFCHECK", 94.36), ("EKFCHECK", 101.76)],
    "00000122": [("GPS", 398.56), ("GPS", 449.56), ("GPS", 519.76), ("GPS", 602.26)],
    "00000123": [("GPS", 155.06), ("GPS", 346.46), ("EKFCHECK", 354.56)],
}


def test_baselines_not_armed():
    """C:未解鎖 / 非飛行;不排名、沒有 CRASH_CHECK / 翻覆 / 自由落體 / 推力損失;馬達 / 振動 / 電池不適用;
    地面事件只當資訊;USB 供電 / 拔電池尾端的說明。"""
    for name in crashkit.BASELINES:
        ctx = crashkit.load_ctx(name)
        r = E.analyze_ctx(ctx, path=crashkit.real_path(name))
        text = r.format_text()
        check(r.end_type == "not_armed" and r.verdict == "no_failure", f"{name}: {r.end_type}/{r.verdict}")
        check(r.hypotheses == [] and r.timeline == [] and r.chain == [], f"{name}: 不應有排名 / 時間軸")
        check(r.findings and all(f.status == NA for f in r.findings), f"{name}: 偵測器應全部 NA")
        check({f.id for f in r.findings} >= {"motor_thrust_loss_check", "vibration_flight_level", "power_low_voltage"},
              f"{name}: 缺少 NA 的偵測器")
        check("未解鎖" in r.summary_zh and "不做失事原因分析" in r.summary_zh, f"{name}: 摘要 {r.summary_zh}")
        for dom in ("馬達:不適用", "振動 / IMU:不適用", "電池 / 電源:不適用"):
            check(dom in text, f"{name}: 沒有「{dom}」")
        for bad in ("CRASH_CHECK", "翻覆", "自由落體", "推力損失", "假說排名", "墜機", "正常範圍"):
            check(bad not in text, f"{name}: 不得出現「{bad}」")
        check(all(e.tag == TAG_GROUND for e in r.ground_events), f"{name}: 地面事件標籤")
        for kw, t in _GROUND_EXPECT.get(name, []):
            check(any(kw in e.label and abs(e.t - t) <= 0.15 for e in r.ground_events), f"{name}: 地面事件 {kw}@{t} 沒列出")
        if name == "00000123":
            n = sum(1 for e in r.ground_events if e.label.startswith("EKFCHECK") and "過大" in e.label)
            check(n == 7, f"123 EKFCHECK 次數 {n}(預期 7)")
        if name in ("00000119", "00000120"):
            check(any(e.label.startswith("EK3_SOURCES_SET") for e in r.ground_events), f"{name}: 來源切換沒列出")
        if name in ("00000117", "00000122", "00000123"):
            check("USB" in text, f"{name}: 應說明 USB 供電、電池分析不適用")
        if name in ("00000118", "00000120"):
            m = re.search(r"([\d.]+) ?→ ?([\d.]+) V 是拔電池", text)
            check(m is not None, f"{name}: 應說明 log 尾端是拔電池")
            check(float(m.group(1)) >= 45.0 and 15.0 <= float(m.group(2)) <= 22.0, f"{name}: 拔電池 {m.groups()}")
        json.dumps(r.to_dict(), ensure_ascii=False, default=float)
        print(f"  {name}:未解鎖,地面事件 {len(r.ground_events)} 筆")


# =============================================================== (5) 跨切面

def test_determinism_and_performance():
    """D5 / D7:同一輸入 → 完全相同的報告;從 LogData 重新分析(新 Ctx)< 10 s 且結果一致;JSON 可序列化。"""
    ctx = crashkit.load_ctx(CRASH)
    d1 = json.dumps(rep124().to_dict(), ensure_ascii=False, default=float, sort_keys=True)
    d2 = json.dumps(E.analyze_ctx(ctx, path=crashkit.real_path(CRASH)).to_dict(), ensure_ascii=False, default=float,
                    sort_keys=True)
    check(d1 == d2, "同一 Ctx 兩次分析結果不同")
    t = time.time()
    r3 = E.analyze(ctx.ld)                       # 新 Ctx(含所有表格轉換)
    dt = time.time() - t
    check(dt < 10.0, f"分析 124 花 {dt:.1f} s(> 10 s)")
    s3 = [(h.id, round(h.score, 9), h.status, h.confidence) for h in r3.hypotheses]
    s1 = [(h.id, round(h.score, 9), h.status, h.confidence) for h in rep124().hypotheses]
    check(s1 == s3, f"新 Ctx 的排名不同:{s3}")
    check(r3.format_text() == rep124().format_text(), "新 Ctx 的報告文字不同")
    print(f"  analyze(124) 新 Ctx {dt:.2f} s;重跑結果完全相同")


# =============================================================== (6) 合成情境

class SynthCtx(Ctx):
    """不讀 log 檔、直接塞表的 Ctx。tables 的 key 為表名(instance=None);series 為 'TBL.Field' → (t, v)。"""

    def __init__(self, armed, tables=None, series=None, params=None, messages=None, t_end=1000.0):
        self.ld = None
        self.ix = None
        self.t_base = 0.0
        self.t_end = t_end
        self._tables = {(k, None): v for k, v in (tables or {}).items()}
        self._series = dict(series or {})
        for name, tb in (tables or {}).items():
            for col, v in tb.items():
                if col != "t" and np.asarray(v).dtype.kind in "biuf":
                    self._series.setdefault(f"{name}.{col}", (np.asarray(tb["t"], float), np.asarray(v, float)))
        self.params = dict(params or {})
        self.messages = list(messages or [])
        self.events = []
        self.armed = list(armed)


class FakeModule:
    """假偵測模組:fn(ctx, a) 設錨點並回傳 Finding。"""

    def __init__(self, fn, name="fake"):
        self.fn = fn
        self.__name__ = name

    def run(self, ctx, a):
        return self.fn(ctx, a)


def F(det, dom, status, t=None, context="", **values):
    return Finding(det, dom, status, "high" if status == HIT else "info", t_onset=t, t_alarm=t, context=context,
                   values=values, text_zh=f"合成 {det} {status}" + (f" {t:.2f} s" if t is not None else "") + "。數字 1.0。")


def flight_ctx(t0=100.0, t1=206.0, method=None, alt_air=20.0, t_air=(110.0, 1e9), tho_idle_after=None,
               ev=(), err=(), mode_num=5, vz=0.0, t_vz=None):
    """一次合成飛行:CTUN(Alt / CRt / DCRt / ThO)、核心表、EV / ERR / MODE。"""
    t = np.arange(t0 - 5.0, t1 + 5.0, 0.1)
    alt = np.where((t >= t_air[0]) & (t < t_air[1]), alt_air, 0.0)
    crt = np.zeros_like(t)
    if t_vz is not None:
        crt[(t >= t_vz[0]) & (t <= t_vz[1])] = vz
    tho = np.full_like(t, 0.4)
    if tho_idle_after is not None:
        tho[t >= tho_idle_after] = 0.0
    core = {"t": t.copy(), "X": np.zeros_like(t)}
    tables = {"CTUN": {"t": t, "Alt": alt, "CRt": crt, "DCRt": np.zeros_like(t), "ThO": tho},
              "ANG": dict(core), "IMU": dict(core), "RCOU": dict(core), "XKF4": dict(core),
              "MODE": {"t": np.array([t0 - 1.0]), "ModeNum": np.array([float(mode_num)])}}
    if ev:
        e = sorted(ev)
        tables["EV"] = {"t": np.array([x[0] for x in e], float), "Id": np.array([x[1] for x in e], float)}
    if err:
        e = sorted(err)
        tables["ERR"] = {"t": np.array([x[0] for x in e], float), "Subsys": np.array([x[1] for x in e], float),
                         "ECode": np.array([x[2] for x in e], float)}
    return SynthCtx([ArmedInterval(0, t0, t1, method)], tables=tables, t_end=t1 + 5.0)


def anchors_setter(**kw):
    def set_(a):
        for k, v in kw.items():
            setattr(a, k, v)
    return set_


def run_fake(ctx, setter, findings, interval_index=None):
    def fn(c, a):
        setter(a)
        return [Finding(f.id, f.domain, f.status, f.severity, f.t_onset, f.t_alarm, f.context, dict(f.values), f.text_zh)
                for f in findings]
    return E.analyze_ctx(ctx, path="synthetic", interval_index=interval_index, modules=[FakeModule(fn)])


_CLEAR_REST = [F("touchdown_not_recognised", "ground", CLEAR), F("ekf_testratio_gate", "nav", CLEAR),
               F("ekf_aiding_timeout", "nav", CLEAR), F("ekf_phantom_motion_on_ground", "ground", CLEAR),
               F("power_low_voltage", "power", CLEAR), F("power_voltage_collapse", "power", CLEAR),
               F("power_rail_fault", "power", CLEAR), F("vibration_flight_level", "vibration", CLEAR),
               F("imu_consistency_health", "vibration", CLEAR), F("ekf_mag_reject", "compass", CLEAR),
               F("compass_sensor_health", "compass", CLEAR), F("ekf_yaw_reset_lane_switch", "compass", CLEAR),
               F("pilot_input_context", "attitude", CLEAR), F("inflight_failsafe_or_mode_change", "failsafe", CLEAR),
               F("impact_shock_clusters", "end_state", CLEAR), F("idle_motor_asymmetry", "ground", CLEAR),
               F("lean_without_pilot", "attitude", CLEAR)]


def _motor_case(shift=0.0):
    """空中推力損失:t_warn 200、t_loss 201;馬達證據在 shift 秒之後(shift>2 → 全部變成後果)。"""
    ctx = flight_ctx()
    st = anchors_setter(end_type="crash_check_replay", t_detect=205.0, t_warn=200.0, t_loss=201.0,
                        t_first_hit=202.5, t_impact_peak=202.6)
    fs = [F("motor_thrust_loss_check", "motor", HIT, 198.5 + shift, "air"),
          F("rate_output_saturation", "attitude", HIT, 199.0 + shift, "air"),
          F("motor_thrust_imbalance", "motor", HIT, 197.0 + shift * 2, "air"),
          F("yaw_imbalance_iterm", "motor", CLEAR, None, "air")] + _CLEAR_REST
    return run_fake(ctx, st, fs)


def test_synth_motor_failure_ranked():
    """合成:空中推力損失(與 124 完全不同的時間 / 數字)→ H_MOTOR_THRUST 第一名(高)、AIR_LOSS_OF_CONTROL / motor_saturation。"""
    r = _motor_case()
    check(r.verdict == "ranked", f"{r.verdict} {r.verdict_note}")
    top = r.top()
    check(top.id == "H_MOTOR_THRUST" and top.confidence == "高", f"第一名 {top.id} {top.confidence}")
    near(top.score, 9.0, 1e-9, "H_MOTOR 分數(4+2+3)")
    check(r.scenario == "AIR_LOSS_OF_CONTROL" and r.scenario_sub == ["motor_saturation"], f"{r.scenario} {r.scenario_sub}")
    check(hyp(r, "H_LAND_TIPOVER").status == "ruled_out", "觸地翻覆應排除")
    check(hyp(r, "H_POWER").status == "ruled_out" and hyp(r, "H_POWER").rule_out_conf == "高", "電源排除(沒有資料品質旗標 → 高)")
    check(hyp(r, "H_MOTOR_THRUST").rule_out_conf == "", "第一名不應有排除信心")
    check("馬達" in r.summary_zh and "推力" in r.summary_zh, "摘要沒有寫出馬達原因")
    check(r.recommendations and r.recommendations[0] == H.HYP_BY_ID["H_MOTOR_THRUST"].advice_zh, "建議應來自第一名")


def test_synth_consequence_scores_zero():
    """合成:同樣的馬達證據全部晚於 t_loss → 0 分、標為後果、已排除;沒有其他證據 → 證據不足(不硬給原因)。"""
    r = _motor_case(shift=3.0)
    mot = hyp(r, "H_MOTOR_THRUST")
    check(mot.score == 0.0 and mot.status == "ruled_out", f"H_MOTOR {mot.score} {mot.status}")
    check(any("後果" in b for b in mot.ruled_out_basis), f"排除依據沒標後果:{mot.ruled_out_basis}")
    check(r.verdict == "insufficient" and "沒有任何假說" in r.verdict_note, f"{r.verdict} {r.verdict_note}")
    check(all(f.tag == TAG_CONSEQUENCE for f in r.findings if f.status == HIT), "晚於 t_loss 的 HIT 應全是後果")
    check("無法確定" in r.summary_zh, "證據不足時摘要要說無法確定")


def test_synth_ambiguous():
    """合成:兩個假說分數相同 → 無法區分(insufficient),兩者都只是候選(candidate,不是第一名)、信心低。"""
    ctx = flight_ctx()
    st = anchors_setter(end_type="hard_impact", t_detect=205.0, t_warn=200.0, t_loss=201.0, t_first_hit=202.0)
    fs = [F("motor_thrust_loss_check", "motor", HIT, 199.0, "air"), F("power_low_voltage", "power", HIT, 199.2, "air")]
    r = run_fake(ctx, st, fs)
    check(r.verdict == "insufficient" and "無法區分" in r.verdict_note, f"{r.verdict} {r.verdict_note}")
    a, b = r.hypotheses[0], r.hypotheses[1]
    check({a.id, b.id} == {"H_MOTOR_THRUST", "H_POWER"} and a.status == b.status == "candidate", "兩個並列候選")
    check(a.confidence == b.confidence == "低", "無法區分時信心為低")
    check(r.top() is None, "無法區分時不得有第一名")


def test_synth_standing_only_nav_gated():
    """合成:導航只有持續狀態證據 → 必要證據閘門 ×0.3 → 低於 S_MIN → 證據不足。"""
    ctx = flight_ctx()
    st = anchors_setter(end_type="crash_check", t_detect=205.0, t_warn=200.0, t_loss=201.0)
    fs = [F("ekf_vio_disagreement", "nav", HIT, 150.0, "standing"),
          F("nav_velocity_gps_fallback_risk", "nav", HIT, 120.0, "standing"),
          F("touchdown_lateral_speed", "ground", HIT, 199.0, "touchdown")]
    r = run_fake(ctx, st, fs)
    nav = hyp(r, "H_NAV_FAILURE")
    check(not nav.gate_ok, "只有持續狀態不應通過導航閘門")
    near(nav.raw, 2 * 0.6 + 1 * 0.6 + 1.0, 1e-9, "H_NAV raw")
    near(nav.score, nav.raw * H.GATE_PENALTY, 1e-9, "閘門懲罰")
    check(r.verdict == "insufficient", f"verdict {r.verdict}")


def test_synth_land_tipover_shifted():
    """合成:124 的事件結構搬到完全不同的時間 / 數字 → 同樣排出 H_LAND(高)+ H_NAV 貢獻因素(中),
    型態 GROUND_TIPOVER(autopilot_lean + land_not_declared)。"""
    t0, t1 = 500.0, 640.0
    ctx = flight_ctx(t0, t1, method=12, t_air=(510.0, 599.5), tho_idle_after=600.3, ev=((510.0, 28),),
                     err=((605.0, 12, 1),), vz=-40.0, t_vz=(594.0, 599.9))
    st = anchors_setter(t_warn=600.8, t_loss=602.5, t_first_hit=603.0, t_impact_peak=603.5, t_touch=600.0)
    fs = [F("touchdown_not_recognised", "ground", HIT, 600.0, "near_ground", agl_m=0.2, vz_pre_cms=-40.0, ae_deg=2.0),
          F("lean_without_pilot", "attitude", HIT, 600.6, "ground_idle", lean_peak=18.0, lean_from_pos=True),
          F("impact_shock_clusters", "end_state", HIT, 600.0, "ground_contact", cls="ground_contact"),
          F("idle_motor_asymmetry", "ground", HIT, 601.0, "ground_idle"),
          F("ekf_phantom_motion_on_ground", "ground", HIT, 601.2, "static_idle"),
          F("vio_dropout", "nav", HIT, 599.9, "near_touchdown"),
          F("nav_velocity_gps_fallback_risk", "nav", HIT, 520.0, "standing"),
          F("pilot_input_context", "attitude", HIT, 602.0, "reaction"),
          F("source_switch_audit", "nav", HIT, 602.8, "switch"),
          F("motor_thrust_loss_check", "motor", CLEAR, None, "air"),
          F("power_low_voltage", "power", CLEAR, None, "air")]
    r = run_fake(ctx, st, fs)
    check(r.end_type == "crash_check", f"end_type {r.end_type}(ERR 12 → crash_check)")
    land, nav = hyp(r, "H_LAND_TIPOVER"), hyp(r, "H_NAV_FAILURE")
    near(land.score, 4 + 3 + 2 + 3 * 0.8 + 1 * 0.8 + 1, 1e-9, "H_LAND 分數")
    near(nav.score, 3 * 0.8 + 2 + 2 + 0.6, 1e-9, "H_NAV 分數")
    check(r.top().id == "H_LAND_TIPOVER" and land.confidence == "高", f"第一名 {r.top().id} {land.confidence}")
    check(nav.status == "contributing" and nav.confidence == "中", f"H_NAV {nav.status} {nav.confidence}")
    check(r.scenario == "GROUND_TIPOVER", f"scenario {r.scenario}")
    check(set(r.scenario_sub) == {"autopilot_lean", "land_not_declared"}, f"子型態 {r.scenario_sub}")
    pil = [f for f in r.findings if f.id == "pilot_input_context"][0]
    check(pil.tag == TAG_CONSEQUENCE, f"t_warn 後的飛手動作應為後果,得到 {pil.tag}")
    sw = [f for f in r.findings if f.id == "source_switch_audit"][0]
    check(sw.tag == TAG_CONSEQUENCE, f"發展期才出現的來源切換(觸發型)應改標後果,得到 {sw.tag}")
    check(hyp(r, "H_CONTROL_PILOT").score == 0.0, "事後的飛手動作不得計分")


def test_synth_scenarios():
    """合成:型態判定優先序 COMMANDED_FLIP > FREE_FALL > GROUND_TIPOVER > HARD_LANDING > AIR_LOSS_OF_CONTROL。"""
    def scen(**kw):
        ctx = flight_ctx(**{k: v for k, v in kw.items() if k in ("alt_air", "t_air", "mode_num", "vz", "t_vz", "ev")})
        a = Anchors(intervals=list(ctx.armed), interval=ctx.armed[0], end_type="crash_check", t_disarm=206.0,
                    t_warn=200.0, t_loss=201.0, t_touch=kw.get("t_touch"))
        fs = kw.get("findings", [])
        return E._scenario(ctx, a, fs)

    check(scen()[0] == "AIR_LOSS_OF_CONTROL" and scen()[1] == ["unknown"], f"高空失控 {scen()}")
    check(scen(mode_num=1)[0] == "COMMANDED_FLIP", "ACRO(ModeNum 1)→ 刻意動作")
    check(scen(ev=((199.0, 21),))[0] == "COMMANDED_FLIP", "FLIP 事件 → 刻意動作")
    check(scen(vz=-500.0, t_vz=(195.0, 200.0))[0] == "FREE_FALL", "vz_pre ≤ −400 → 墜落")
    ff = [F("free_fall_run", "end_state", HIT, 199.0, "free_fall")]
    check(scen(findings=ff)[0] == "FREE_FALL", "預警前的自由落體 → 墜落")
    ff_late = [F("free_fall_run", "end_state", HIT, 203.0, "in_tumble")]
    check(scen(findings=ff_late)[0] == "AIR_LOSS_OF_CONTROL", "預警後的自由落體不改變型態")
    gnd = dict(t_air=(110.0, 199.0), t_touch=199.5)
    check(scen(**gnd)[0] == "GROUND_TIPOVER", f"貼地 + 慢速 → 觸地翻覆 {scen(**gnd)}")
    hard = dict(t_air=(110.0, 199.0), t_touch=199.5, vz=-250.0, t_vz=(195.0, 199.4))
    check(scen(**hard)[0] == "HARD_LANDING", f"貼地 + 快速下降 → 重落地 {scen(**hard)}")
    lab, sub, feats = scen(alt_air=15.0, findings=[F("lean_without_pilot", "attitude", HIT, 199.5, "air", lean_peak=20.0)])
    check(lab == "AIR_LOSS_OF_CONTROL" and "attitude_cmd_runaway" in sub, f"空中置中搖桿大傾角 {lab} {sub}")
    check(feats["agl_m"] == 15.0, f"agl 特徵 {feats}")


def test_synth_power_log_end():
    """合成:log 在解鎖中結束 —— 最後掉壓(power_voltage_collapse HIT)→ H_POWER 第一名;電壓正常 → 不硬給原因(證據不足,
    可能只是記錄中斷)。log 結束本身在沒有姿態預警時標為起點(precursor),不是後果。"""
    ctx = flight_ctx(t1=300.0)
    ctx.armed = [ArmedInterval(0, 100.0, 300.0, None, open_end=True)]
    for collapsed, expect in ((True, "ranked"), (False, "insufficient")):
        fs = [F("abnormal_disarm_classifier", "end_state", HIT, 300.0, "log_ended_armed", **{"class": "log_ended_armed",
                                                                                           "volt_collapsed": collapsed}),
              F("power_voltage_collapse", "power", HIT if collapsed else CLEAR, 299.4 if collapsed else None, "air"),
              F("power_low_voltage", "power", CLEAR), F("motor_thrust_loss_check", "motor", CLEAR)]
        r = run_fake(ctx, anchors_setter(end_type="log_ended_armed", t_detect=300.0), fs)
        p = hyp(r, "H_POWER")
        check(p.gate_ok is collapsed, f"volt_collapsed={collapsed}: 閘門 {p.gate_ok}")
        check(r.verdict == expect, f"volt_collapsed={collapsed}: verdict {r.verdict} {r.verdict_note}")
        ab = [f for f in r.findings if f.id == "abnormal_disarm_classifier"][0]
        check(ab.tag == TAG_PRECURSOR, f"log 結束事件應為起點,得到 {ab.tag}")
        if collapsed:
            check(r.top().id == "H_POWER" and p.score == 4.0, f"電壓崩落 → H_POWER 第一名 {p.score}")
        else:
            check(p.score == 0.0 and p.status == "ruled_out", f"電壓正常 → H_POWER 已排除 {p.status}")


def test_synth_flip_and_manual_disarm():
    """合成:ACRO 模式下的翻滾 → H_CONTROL_PILOT(刻意動作);空中手動上鎖(無姿態預警)→ 上鎖本身是起點。"""
    ctx = flight_ctx(mode_num=1)
    r = run_fake(ctx, anchors_setter(end_type="tumble", t_detect=203.0, t_warn=200.0, t_loss=201.0), [])
    check(r.scenario == "COMMANDED_FLIP", f"scenario {r.scenario}")
    p = hyp(r, "H_CONTROL_PILOT")
    check(p.gate_ok and p.score == 4.0 and r.top().id == "H_CONTROL_PILOT", f"刻意動作 {p.score} {p.gate_ok}")
    ctx = flight_ctx()
    fs = [F("abnormal_disarm_classifier", "end_state", HIT, 206.0, "manual_disarm_in_air", **{"class": "manual_disarm_in_air"})]
    r = run_fake(ctx, anchors_setter(end_type="abnormal_disarm", t_detect=206.0, t_first_hit=207.0), fs)
    p = hyp(r, "H_CONTROL_PILOT")
    check(p.gate_ok and p.score == 4.0 and r.top().id == "H_CONTROL_PILOT", f"空中上鎖 {p.score} {p.status}")
    # 空中上鎖分類沒有實測樣本(規格 catalogue)→ 只靠它的假說信心封頂為低,並說明原因
    check(p.confidence == "低" and "沒有實測樣本" in p.confidence_note, f"空中上鎖信心 {p.confidence} {p.confidence_note}")
    check("沒有實測樣本" in r.format_text(), "報告應說明信心封頂原因")
    check(r.scenario == "UNKNOWN", f"沒有姿態預警、撞擊在上鎖之後 → 型態不明,得到 {r.scenario}")


def test_synth_no_failure_and_not_armed():
    """合成:正常降落 → 不排名、只列健康觀察、時間軸不貼因果標籤;未解鎖 → 地面事件 + USB 說明。"""
    ctx = flight_ctx(ev=((110.0, 28), (203.0, 17), (204.0, 18)))
    fs = [F("vibe_clip_transient", "end_state", SOFT, 150.0, "short_burst"), F("motor_thrust_loss_check", "motor", CLEAR)]
    r = run_fake(ctx, anchors_setter(end_type="normal_landing"), fs)
    check(r.verdict == "no_failure" and r.hypotheses == [], f"{r.verdict} {len(r.hypotheses)}")
    check([o["id"] for o in r.observations] == ["vibe_clip_transient"], f"健康觀察 {r.observations}")
    check(all(e.tag in ("", TAG_GROUND) for e in r.timeline), "正常飛行時間軸不應貼先兆 / 後果")
    check("假說排名" not in r.format_text(), "正常飛行不應有假說排名")
    # 未解鎖
    t = np.arange(0.0, 50.0, 0.1)
    ctx = SynthCtx([], tables={"ERR": {"t": np.array([12.0]), "Subsys": np.array([10.0]), "ECode": np.array([1.0])},
                               "BAT": {"t": t, "Volt": np.full_like(t, 0.05)}})
    called = []
    r = E.analyze_ctx(ctx, modules=[FakeModule(lambda c, a: called.append(a.interval) or [F("x", "motor", NA, None, "not_armed")])])
    check(called == [None], "未解鎖時模組仍要被呼叫(回 NA)")
    check(r.end_type == "not_armed" and r.verdict == "no_failure" and r.hypotheses == [], "未解鎖")
    check(len(r.ground_events) == 1 and r.ground_events[0].tag == TAG_GROUND, "地面 ERR")
    check(any("USB" in n for n in r.notes), f"USB 說明 {r.notes}")
    check(r.na_domains and r.na_domains[0]["domain"] == "motor", "不適用的領域")


def test_synth_disarm_written_events_and_fallback():
    """合成:沒有 end_state 模組時用 ARM / EV 推終點;上鎖 1 ms 內補寫的 EV17/18 不算落地、不列時間軸;缺模組有記錄。"""
    ev = ((110.0, 28), (206.0004, 18), (206.0006, 17))
    ctx = flight_ctx(method=12, ev=ev)
    r = E.analyze_ctx(ctx, modules=["det_does_not_exist"])
    check(r.missing_modules == ["det_does_not_exist"], f"缺模組 {r.missing_modules}")
    check(r.end_type == "crash_check", f"Method 12 → crash_check,得到 {r.end_type}")
    check(not [e for e in r.timeline if "LAND_COMPLETE" in e.label], "上鎖補寫的落地事件不應列出")
    check("下列偵測模組不存在" in r.format_text(), "限制段應說明缺模組")
    ctx = flight_ctx(method=0, ev=((110.0, 28), (203.0, 17), (204.0, 18)))
    check(E.analyze_ctx(ctx, modules=[]).end_type == "normal_landing", "Method 0 + 先 EV18 → normal_landing")
    ctx = flight_ctx(method=0, ev=((110.0, 28), (206.0004, 18)))
    check(E.analyze_ctx(ctx, modules=[]).end_type == "unknown", "只有上鎖補寫的 EV18 → 不能算正常降落")


def test_synth_module_failure_and_short_data():
    """合成:模組丟例外 → 記在 notes、其餘照跑;解鎖不到 20 s → 資料不足。"""
    def boom(c, a):
        raise RuntimeError("壞掉")
    ctx = flight_ctx()
    st = FakeModule(lambda c, a: (anchors_setter(end_type="crash_check", t_detect=205.0, t_warn=200.0, t_loss=201.0)(a),
                                  [F("motor_thrust_loss_check", "motor", HIT, 199.0, "air")])[1])
    r = E.analyze_ctx(ctx, modules=[FakeModule(boom, "boom"), st])
    check(any("boom" in n and "壞掉" in n for n in r.notes), f"notes {r.notes}")
    check(r.top() is not None and r.top().id == "H_MOTOR_THRUST", "其他模組照跑")
    ctx = flight_ctx(t0=100.0, t1=110.0)
    r = run_fake(ctx, anchors_setter(end_type="crash_check", t_detect=109.0, t_warn=105.0, t_loss=106.0),
                 [F("motor_thrust_loss_check", "motor", HIT, 104.0, "air")])
    check(r.verdict == "insufficient" and "資料不足" in r.verdict_note, f"{r.verdict} {r.verdict_note}")


def test_tag_time_rules():
    """因果標籤規則(與 124 無關的時間)。"""
    iv = ArmedInterval(0, 10.0, 110.0, 12)
    a = Anchors(interval=iv, end_type="crash_check", t_warn=100.0, t_loss=103.0, t_disarm=110.0)
    cases = [(95.0, TAG_PRECURSOR), (101.0, TAG_DEVELOPMENT), (104.0, TAG_CONSEQUENCE), (112.0, TAG_POST),
             (200.0, TAG_GROUND), (5.0, TAG_GROUND), (None, TAG_STANDING)]
    for t, want in cases:
        check(E.tag_time(t, a) == want, f"t={t}: {E.tag_time(t, a)} ≠ {want}")
    check(E.tag_time(50.0, a, standing=True) == TAG_STANDING, "standing")
    b = Anchors(interval=iv, end_type="abnormal_disarm", t_first_hit=105.0, t_disarm=110.0)
    check(E.tag_time(104.9, b) == TAG_PRECURSOR and E.tag_time(105.0, b) == TAG_CONSEQUENCE, "沒有預警:撞擊之前是先兆")
    # 結束事件本身(空中上鎖 / log 中斷)在沒有預警時是起點:t_onset == t_detect 也要標 precursor
    d = Anchors(interval=iv, end_type="abnormal_disarm", t_detect=110.0, t_disarm=110.0)
    fs = [Finding("abnormal_disarm_classifier", "end_state", HIT, t_onset=110.0, context="manual_disarm_in_air",
                  values={"class": "manual_disarm_in_air"}),
          Finding("abnormal_disarm_classifier", "end_state", HIT, t_onset=110.0, context="crash_check",
                  values={"class": "crash_check"}),
          Finding("pilot_input_context", "attitude", HIT, t_onset=110.0, context="reaction")]
    E._tag_findings(None, d, fs)
    check([f.tag for f in fs] == [TAG_PRECURSOR, TAG_CONSEQUENCE, TAG_CONSEQUENCE], f"結束事件標籤 {[f.tag for f in fs]}")
    c = Anchors(interval=iv, end_type="normal_landing", t_disarm=110.0)
    check(E.tag_time(50.0, c) == "" and E.tag_time(112.0, c) == TAG_GROUND, "正常飛行不貼先兆 / 事後")
    check(H.prox(Finding("x", "d", HIT, t_onset=95.0, tag=TAG_PRECURSOR), a) == 1.0, "lead 5 s → 1.0")
    check(H.prox(Finding("x", "d", HIT, t_onset=-100.0, tag=TAG_PRECURSOR), a) == 0.6, "lead 200 s → 0.6")
    check(H.prox(Finding("x", "d", HIT, t_onset=-300.0, tag=TAG_PRECURSOR), a) == 0.3, "lead 400 s → 0.3")
    check(H.prox(Finding("x", "d", HIT, t_onset=104.0, tag=TAG_CONSEQUENCE), a) == 0.0, "後果 → 0")


# =============================================================== (7) 審查修正的回歸測試

def test_tag_consequence_boundary():
    """後果邊界 = t_loss 與第一次硬撞擊中不早於 t_warn 的最早者:低空失控在 ae 到 30° 前就撞地時,撞擊後的
    EKF 垃圾 / 電壓擺盪 / 振動都是後果(0 分),不能以「發展」0.8 計分;撞擊早於 t_loss 時同理。"""
    iv = ArmedInterval(0, 10.0, 110.0, None)
    a = Anchors(interval=iv, end_type="hard_impact", t_warn=100.0, t_loss=None, t_first_hit=101.0, t_disarm=110.0)
    check(E.consequence_boundary(a) == 101.0, f"邊界 {E.consequence_boundary(a)}")
    for t, want in ((99.0, TAG_PRECURSOR), (100.5, TAG_DEVELOPMENT), (101.0, TAG_CONSEQUENCE), (104.0, TAG_CONSEQUENCE)):
        check(E.tag_time(t, a) == want, f"t_loss=None, 撞擊 101:t={t} → {E.tag_time(t, a)}(應為 {want})")
    b = Anchors(interval=iv, end_type="hard_impact", t_warn=100.0, t_loss=103.0, t_first_hit=101.0, t_disarm=110.0)
    check(E.tag_time(102.0, b) == TAG_CONSEQUENCE, "撞擊(101)早於 t_loss(103):102 應為後果")
    c = Anchors(interval=iv, end_type="tumble", t_warn=100.0, t_disarm=110.0)
    check(E.consequence_boundary(c) is None and E.tag_time(105.0, c) == TAG_DEVELOPMENT, "沒有失控 / 撞擊:預警後到上鎖都是發展")
    d = Anchors(interval=iv, end_type="hard_impact", t_warn=100.0, t_loss=103.0, t_first_hit=99.0, t_disarm=110.0)
    check(E.consequence_boundary(d) == 103.0 and E.tag_time(101.0, d) == TAG_DEVELOPMENT, "早於 t_warn 的撞擊不當後果邊界")

    # 審查 probe 1:t_warn 200、沒有 t_loss、201 硬撞擊 → 203 EKF、204 電壓、202 振動(持續狀態型)全部是後果、0 分
    ctx = flight_ctx()
    st = anchors_setter(end_type="hard_impact", t_detect=206.0, t_warn=200.0, t_loss=None, t_first_hit=201.0,
                        t_impact_peak=201.1)
    fs = [F("ekf_testratio_gate", "nav", HIT, 203.0, "air"), F("power_low_voltage", "power", HIT, 204.0, "air"),
          F("vibration_flight_level", "vibration", HIT, 202.0, "air"),
          F("imu_consistency_health", "vibration", HIT, 150.0, "standing"),
          F("impact_shock_clusters", "end_state", HIT, 201.0, "hard_impact", cls="hard_impact", ae_at_impact=12.0)]
    r = run_fake(ctx, st, fs)
    tags = {f.id: f.tag for f in r.findings}
    check(tags["ekf_testratio_gate"] == tags["power_low_voltage"] == TAG_CONSEQUENCE, f"撞擊後的 HIT 標籤 {tags}")
    check(tags["vibration_flight_level"] == TAG_CONSEQUENCE, f"撞擊後才開始的持續狀態型 HIT 應改標後果:{tags}")
    check(tags["imu_consistency_health"] == TAG_STANDING, "撞擊前就存在的持續狀態維持 standing")
    for hid in ("H_NAV_FAILURE", "H_POWER"):
        check(hyp(r, hid).score == 0.0, f"{hid} 只有撞擊後的證據,不得計分:{hyp(r, hid).score}")
    vib = hyp(r, "H_VIBRATION_IMU")
    near(vib.raw, 3.0 * 0.6, 1e-9, "H_VIB 只計撞擊前的 IMU 持續狀態")
    check(all(e["contrib"] == 0 for e in vib.evidence if e["det"] == "vibration_flight_level"), "撞擊後的振動不得計分")
    late_vib = [f for f in r.findings if f.id == "vibration_flight_level"]
    check(not H._gate_vib(H.Ctx2(late_vib, r.anchors))[0], "撞擊後才出現的振動不能滿足振動假說的必要證據")
    check("無法區分" not in r.verdict_note, f"不應出現由撞擊後資料造成的無法區分:{r.verdict_note}")
    late = [e for e in r.timeline if e.source == "vibration_flight_level"]
    check(late and late[0].tag == TAG_CONSEQUENCE, "改標後果的持續狀態型 HIT 要列在時間軸")
    # 審查 probe 1b:t_warn 200 < 撞擊 201 < t_loss 203,EKF HIT 在 202 → 後果、H_NAV 0 分
    st = anchors_setter(end_type="hard_impact", t_detect=206.0, t_warn=200.0, t_loss=203.0, t_first_hit=201.0)
    r = run_fake(ctx, st, [F("ekf_testratio_gate", "nav", HIT, 202.0, "air"),
                           F("power_low_voltage", "power", HIT, 199.0, "air")])
    check(hyp(r, "H_NAV_FAILURE").score == 0.0, "撞擊後、t_loss 前的 EKF 異常不得計分")
    check(hyp(r, "H_POWER").score == 4.0, "預警前的電壓證據照常計分(不是全部歸零)")


def test_synth_interval_auto_pick():
    """dataflow STEP 2:自動分析「最後一次異常結束」的解鎖 —— 失效保護上鎖、上鎖時仍未判定落地(空中上鎖 / 沒有
    CRASH_CHECK 的撞擊)、CRASH_CHECK ERR、log 在解鎖中結束;之後的地面解鎖 / 上鎖不能把它蓋掉。"""
    t = np.arange(0.0, 400.0, 0.1)

    def mk(armed, ev=(), err=()):
        tables = {"CTUN": {"t": t, "Alt": np.zeros_like(t)}}
        if ev:
            tables["EV"] = {"t": np.array([x[0] for x in ev], float), "Id": np.array([x[1] for x in ev], float)}
        if err:
            tables["ERR"] = {"t": np.array([x[0] for x in err], float), "Subsys": np.array([x[1] for x in err], float),
                             "ECode": np.array([x[2] for x in err], float)}
        return SynthCtx(armed, tables=tables, t_end=400.0)

    def pick(ctx):
        return E.select_anchors(ctx).interval.index

    # 審查 probe 2:第 1 次失效保護上鎖(Method 5)、第 2 次地面解鎖 → 選第 1 次
    ctx = mk([ArmedInterval(0, 10.0, 200.0, 5), ArmedInterval(1, 300.0, 330.0, 0)])
    check(pick(ctx) == 0, f"失效保護上鎖應被選中,得到 {pick(ctx)}")
    check([i["abnormal"] for i in E._interval_summaries(ctx)] == [True, False], "區間清單的異常旗標")
    # 起飛後沒有 LAND_COMPLETE 就被手動上鎖(Method 0;FS_CRASH_CHECK=0 的撞擊也是這樣),之後地面重新解鎖
    ev = ((20.0, 28), (200.0004, 18))           # 上鎖 1 ms 內補寫的 EV18 不算落地
    ctx = mk([ArmedInterval(0, 10.0, 200.0, 0), ArmedInterval(1, 300.0, 330.0, 0)], ev=ev)
    check(pick(ctx) == 0, f"上鎖時仍在空中的那次應被選中,得到 {pick(ctx)}")
    s = E._interval_summaries(ctx)
    check(s[0]["abnormal"] and "仍認為在空中" in s[0]["end"] and not s[1]["abnormal"], f"區間清單 {s}")
    # 兩次都正常(第 1 次有 EV18、第 2 次沒起飛)→ 最後一次
    ctx = mk([ArmedInterval(0, 10.0, 200.0, 0), ArmedInterval(1, 300.0, 330.0, 0)], ev=((20.0, 28), (190.0, 18)))
    check(pick(ctx) == 1 and not any(i["abnormal"] for i in E._interval_summaries(ctx)), "都正常 → 最後一次")
    # CRASH_CHECK ERR(上鎖原因沒記到)在第 1 次 → 選第 1 次並帶出 t_detect
    ctx = mk([ArmedInterval(0, 10.0, 200.0, None), ArmedInterval(1, 300.0, 330.0, 0)], err=((200.2, 12, 1),))
    a = E.select_anchors(ctx)
    check(a.interval.index == 0 and a.end_type == "crash_check" and a.t_detect == 200.2, f"ERR 12 → {a.interval.index} {a.end_type}")
    # log 在解鎖中結束的最後一次 → 選它;明確指定 interval_index 仍照指定
    ctx = mk([ArmedInterval(0, 10.0, 200.0, 0), ArmedInterval(1, 300.0, 399.0, None, open_end=True)])
    check(pick(ctx) == 1, "log 在解鎖中結束 → 異常")
    check(E.select_anchors(ctx, 0).interval.index == 0, "指定 interval_index=0")


def _variant_ctx(base, rearm=(1280.0, 1285.0)):
    """00000124 的變形:拿掉 CRASH_CHECK(ERR 12、ARM.Method 12 → 0、'Crash:' 訊息;等同 FS_CRASH_CHECK=0 + 手動上鎖),
    並在 rearm 秒加一次地面解鎖 / 上鎖(審查 p4)。只換掉 ERR / ARM / MSG 三張表,其餘資料共用。"""
    import copy
    c = copy.copy(base)
    for k in [k for k in c.__dict__ if k.startswith("_") and k not in ("_series", "_tables")]:
        del c.__dict__[k]                     # 偵測模組掛在 ctx 上的快取不能跟原本的 Ctx 共用
    c._tables = dict(base._tables)
    c._series = dict(base._series)
    err, arm, msg = base.table("ERR"), base.table("ARM"), base.table("MSG")
    keep = err["Subsys"] != 12
    c._tables[("ERR", None)] = {k: np.asarray(v)[keep] for k, v in err.items()}
    arm = {k: np.asarray(v).copy() for k, v in arm.items()}
    arm["Method"][arm["Method"] == 12] = 0
    extra = {"t": np.array(rearm, float), "ArmState": np.array([1.0, 0.0])}
    arm = {k: np.concatenate([v, extra.get(k, np.zeros(2))]) for k, v in arm.items()}
    c._tables[("ARM", None)] = arm
    keep = np.array([not str(m).startswith("Crash") for m in msg["Message"]])
    c._tables[("MSG", None)] = {k: np.asarray(v)[keep] for k, v in msg.items()}
    c.messages = [(t, s) for t, s in base.messages if not s.startswith("Crash")]
    c.armed = c._armed_intervals()
    return c


def test_124_no_crash_check_then_ground_rearm():
    """審查 p4(真實資料):124 飛行 2 沒有 CRASH_CHECK(Method 0 手動上鎖)、之後在 1280 s 地面重新解鎖一次 →
    仍要自動分析飛行 2(上鎖時飛控仍認為在空中),而不是最後那次 5 s 的地面解鎖。"""
    ctx = _variant_ctx(crashkit.load_ctx(CRASH))
    check(len(ctx.armed) == 3 and not E._crash_err_times(ctx), f"變形資料 {len(ctx.armed)} 次解鎖")
    r = E.analyze_ctx(ctx, path="124-no-crash-check")
    check(r.interval["index"] == 1, f"自動選到第 {r.interval['index'] + 1} 次(應為第 2 次)")
    iv = r.intervals
    check(iv[1]["abnormal"] and "仍認為在空中" in iv[1]["end"] and not iv[2]["abnormal"] and not iv[0]["abnormal"],
          f"區間異常旗標 {[(i['abnormal'], i['end']) for i in iv]}")
    check(r.end_type == "crash_check_replay", f"沒有 ERR 12 → 重算判定,得到 {r.end_type}")
    check(r.scenario == "GROUND_TIPOVER" and r.top() is not None and r.top().id == "H_LAND_TIPOVER",
          f"{r.scenario} {r.top() and r.top().id}")
    print(f"  自動選第 {r.interval['index'] + 1} 次;{r.end_type};第一名 {r.top().id} {r.top().score:.1f}")


class _TruncIx:
    """LogIndex 包裝:把所有表截到 t_cut(log 絕對秒)為止,模擬「log 在解鎖中、空中正常飛行時中斷」。"""

    def __init__(self, ix, t_cut):
        self._ix, self._cut, self._c = ix, t_cut, {}

    def __getattr__(self, k):
        return getattr(self._ix, k)

    def _cut_tb(self, tb):
        if tb is None:
            return None
        m = np.asarray(tb["t"]) <= self._cut
        return {c: np.asarray(v)[m] for c, v in tb.items()} if m.any() else None

    def table(self, name, instance=None):
        k = ("T", name, instance)
        if k not in self._c:
            self._c[k] = self._cut_tb(self._ix.table(name, instance))
        return self._c[k]

    def raw_table(self, name):
        k = ("R", name)
        if k not in self._c:
            self._c[k] = self._cut_tb(self._ix.raw_table(name))
        return self._c[k]

    def series(self, key):
        s = self._ix.series(key)
        if s is None:
            return None
        m = np.asarray(s[0]) <= self._cut
        return (np.asarray(s[0])[m], np.asarray(s[1])[m]) if m.any() else None

    def log_messages(self):
        return [m for m in self._ix.log_messages() if m.t <= self._cut]


def test_124_log_cut_mid_air_scenario_unknown():
    """審查 p3 D(真實資料):124 在 1250 s(空中、電壓正常、還沒觸地)被截斷 → log_ended_armed、證據不足,
    型態「不明」—— 沒有姿態預警也沒有撞擊時不得寫成「空中失去控制」;也沒有任何假說被排成第一名。"""
    from types import SimpleNamespace
    ld = crashkit.load_ctx(CRASH).ld
    cut = 1250.0 + float(ld.t_base)
    fake = SimpleNamespace(index=_TruncIx(ld.index, cut), t_base=ld.t_base, t_last=cut,
                           events=[e for e in ld.events if e.t <= cut], path="124-cut-1250")
    r = E.analyze_ctx(Ctx(fake), path="124-cut-1250")
    check(r.interval["index"] == 1 and r.interval["open_end"], f"分析區間 {r.interval}")
    check(r.end_type == "log_ended_armed" and r.verdict == "insufficient", f"{r.end_type} / {r.verdict}")
    check(r.scenario == "UNKNOWN" and r.scenario_sub == [], f"型態 {r.scenario} {r.scenario_sub}")
    check("空中失去控制" not in r.format_text() and "型態不明" in r.summary_zh, "不得宣稱空中失去控制")
    check(r.top() is None and all(h.status != "ranked" for h in r.hypotheses), "證據不足時不得有第一名")


def test_synth_scenario_without_anchors():
    """沒有 t_warn、也沒有不晚於判定的硬撞擊:只有刻意動作 / 墜落證據才給型態,否則 UNKNOWN。"""
    def scen(findings=(), **kw):
        ctx = flight_ctx(**{k: v for k, v in kw.items() if k in ("mode_num", "vz", "t_vz")})
        a = Anchors(intervals=list(ctx.armed), interval=ctx.armed[0], end_type="log_ended_armed", t_disarm=206.0,
                    t_detect=206.0, t_first_hit=kw.get("t_first_hit"))
        return E._scenario(ctx, a, list(findings))

    check(scen() == ("UNKNOWN", [], scen()[2]), f"沒有錨點 → UNKNOWN:{scen()[:2]}")
    check(scen(mode_num=1)[0] == "COMMANDED_FLIP", "ACRO 仍是刻意動作")
    check(scen([F("free_fall_run", "end_state", HIT, 205.0, "free_fall")])[0] == "FREE_FALL", "自由落體證據 → 墜落")
    check(scen(vz=-500.0, t_vz=(200.0, 206.0))[0] == "FREE_FALL", "vz_pre ≤ −400 → 墜落")
    check(scen(t_first_hit=207.0)[0] == "UNKNOWN", "撞擊在判定(上鎖)之後 → 不拿來當型態參考")
    check(scen(t_first_hit=204.0)[0] == "AIR_LOSS_OF_CONTROL", "判定前的硬撞擊仍是事件參考點")


def test_synth_insufficient_presentation():
    """審查 probe 8:證據不足時沒有第一名 —— 有分數的假說都是 candidate、事件鏈角色 candidate、報告標「候選」、
    建議先寫補資料,候選的建議加上「若要驗證」的條件式。"""
    ctx = flight_ctx()
    st = anchors_setter(end_type="crash_check", t_detect=205.0, t_warn=200.0, t_loss=201.0)
    fs = [F("motor_thrust_loss_check", "motor", SOFT, 199.0, "air"), F("power_rail_fault", "power", SOFT, 199.0, "air")]
    r = run_fake(ctx, st, fs)
    check(r.verdict == "insufficient", r.verdict)
    live = [h for h in r.hypotheses if h.score > 0]
    check(live and all(h.status == "candidate" for h in live), f"狀態 {[(h.id, h.status) for h in live]}")
    check(r.top() is None, "不得有第一名")
    check(r.chain and all(c["role"] == "candidate" for c in r.chain), f"事件鏈角色 {[c['role'] for c in r.chain]}")
    text = r.format_text()
    check("第一名" not in text and "【候選假說】" in text and "候選(無法確定)" in text, "報告不得寫第一名")
    check("【事件鏈(機制)】" not in text and "【已知證據(依時間)】" in text, "證據不足不寫成機制")
    check(r.recommendations[0].startswith("補齊缺少的資料"), f"建議第一條 {r.recommendations[0][:20]}")
    check(all(x.startswith("(候選,未證實)") for x in r.recommendations[1:]) and len(r.recommendations) >= 2,
          f"候選建議 {[x[:12] for x in r.recommendations]}")
    check(r.recommendations[0] != H.HYP_BY_ID["H_MOTOR_THRUST"].advice_zh, "不得直接拿最高分假說的建議當結論")
    # 資料不足的路徑也一樣
    r = run_fake(flight_ctx(t0=100.0, t1=110.0), anchors_setter(end_type="crash_check", t_detect=109.0, t_warn=105.0,
                                                                 t_loss=106.0),
                 [F("motor_thrust_loss_check", "motor", HIT, 104.0, "air")])
    check(r.verdict == "insufficient" and hyp(r, "H_MOTOR_THRUST").status == "candidate", "資料不足 → 候選")


def test_synth_robustness():
    """偵測模組載入失敗(SyntaxError / 缺相依套件)、執行失敗、回傳 values=None / NaN t_onset / 單一 Finding、
    閘門或篩選函式出錯:分析都要完成;失敗的模組留下該領域的 NA 佔位(不是 CLEAR)與限制說明。"""
    import importlib.abc
    import importlib.machinery

    class Boom(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name in ("core.crash.det_broken_syntax", "core.crash.det_needs_dep"):
                return importlib.machinery.ModuleSpec(name, self)
            return None

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            if module.__name__.endswith("det_needs_dep"):
                raise ModuleNotFoundError("No module named 'scipy'", name="scipy")
            raise SyntaxError("invalid syntax (det_broken_syntax.py, line 12)")

    class Raiser:
        __name__ = "det_raiser"
        DOMAIN = "power"

        def run(self, ctx, a):
            raise RuntimeError("壞掉")

    def fn(c, a):
        anchors_setter(end_type="crash_check", t_detect=205.0, t_warn=200.0, t_loss=201.0)(a)
        return Finding("power_low_voltage", "power", HIT, t_onset=199.0, context="air", values=None, text_zh="電壓低。")

    def fn2(c, a):
        return [Finding("motor_thrust_loss_check", "motor", HIT, t_onset=float("nan"), context="air"), "garbage", None]

    boom = Boom()
    sys.meta_path.insert(0, boom)
    try:
        r = E.analyze_ctx(flight_ctx(), modules=["det_broken_syntax", "det_needs_dep", "det_does_not_exist", Raiser(),
                                                 FakeModule(fn, "single"), FakeModule(fn2, "odd")])
    finally:
        sys.meta_path.remove(boom)
        for m in ("core.crash.det_broken_syntax", "core.crash.det_needs_dep"):
            sys.modules.pop(m, None)
    check(r.missing_modules == ["det_does_not_exist"], f"不存在的模組 {r.missing_modules}")
    check(r.failed_modules == ["det_broken_syntax", "det_needs_dep", "det_raiser"], f"失敗的模組 {r.failed_modules}")
    ph = {f.id: f for f in r.findings if f.id.startswith(E.MODULE_ERROR_PREFIX)}
    check(set(ph) == {E.MODULE_ERROR_PREFIX + n for n in ("det_broken_syntax", "det_needs_dep", "det_raiser")}, f"佔位 {set(ph)}")
    check(all(f.status == NA for f in ph.values()), "佔位必須是 NA")
    check(ph[E.MODULE_ERROR_PREFIX + "det_raiser"].domain == "power", "執行失敗的佔位用模組的 DOMAIN")
    check(any("SyntaxError" in n for n in r.notes) and any("壞掉" in n for n in r.notes), f"notes {r.notes}")
    check(any("載入或執行失敗" in x for x in r.limitations), f"限制段 {r.limitations}")
    p = [f for f in r.findings if f.id == "power_low_voltage"][0]
    check(p.values == {} and hyp(r, "H_POWER").score == 4.0, "values=None 的 Finding 照常計分")
    m = [f for f in r.findings if f.id == "motor_thrust_loss_check"][0]
    check(m.t_onset is None, "NaN t_onset → None")
    json.dumps(r.to_dict(), ensure_ascii=False, default=float, allow_nan=False)
    r.format_text()
    # 閘門 / 篩選函式出錯:當成「未驗證 / 不符」,不中斷分析
    h = H.Hypothesis("H_X", "測試", [H.Weight("x", 1.0, lambda f, c: 1 / 0)], ["x"], "", lambda c: 1 / 0)
    ok, note = H._safe_gate(h, H.Ctx2([], Anchors()))
    check(ok is False and "檢查失敗" in note, f"閘門出錯 {ok} {note}")
    check(H._safe_when(h.weights[0], Finding("x", "d", HIT), H.Ctx2([], Anchors())) is False, "篩選出錯 → 不符")


def test_gates_require_evidence():
    """必要證據缺資料時不得預設通過:觸地翻覆缺 agl / vz / 姿態 → 未驗證;外力撞擊必須是「姿態正常時的硬撞擊」,
    一般觸地(ground_contact)不算。"""
    iv = ArmedInterval(0, 10.0, 110.0, 12)
    a = Anchors(interval=iv, end_type="crash_check", t_warn=100.0, t_loss=103.0, t_disarm=110.0)

    def gate(fn, fs):
        for f in fs:
            f.tag = E.tag_time(f.t_onset, a)
        return fn(H.Ctx2(fs, a))

    td = Finding("touchdown_not_recognised", "ground", HIT, t_onset=99.0, context="near_ground")
    ok, note = gate(H._gate_land, [td])
    check(not ok and "未驗證" in note, f"觸地沒有數字 → 未驗證:{ok} {note}")
    a.extra["scenario"] = {"t_feat": 99.0, "agl_m": 0.1, "vz_pre_cms": -20.0}
    ok, note = gate(H._gate_land, [Finding("touchdown_not_recognised", "ground", HIT, t_onset=99.0)])
    check(not ok and "姿態" in note, f"有高度 / 速度但沒有姿態資料 → 未驗證:{note}")
    att = Finding("attitude_loss_of_control", "attitude", HIT, t_onset=100.0)
    ok, note = gate(H._gate_land, [Finding("touchdown_not_recognised", "ground", HIT, t_onset=99.0), att])
    check(ok, f"高度 / 速度取型態特徵、姿態由 t_warn 證實 → 通過:{note}")
    a.extra = {}
    ok, _ = gate(H._gate_land, [Finding("touchdown_not_recognised", "ground", HIT, t_onset=99.0,
                                        values={"agl_m": 0.2, "vz_pre_cms": -30.0, "ae_deg": 2.0})])
    check(ok, "偵測器自帶數字 → 通過")
    ok, note = gate(H._gate_land, [Finding("touchdown_not_recognised", "ground", HIT, t_onset=99.0,
                                           values={"agl_m": 0.2, "vz_pre_cms": -300.0, "ae_deg": 2.0})])
    check(not ok and "重落地" in note, note)
    # 外力撞擊
    gc = Finding("impact_shock_clusters", "end_state", HIT, t_onset=98.0, context="ground_contact",
                 values={"cls": "ground_contact"})
    ok, _ = gate(H._gate_impact, [gc, att])
    check(not ok, "一般觸地不能滿足外力撞擊的必要證據")
    hard = Finding("impact_shock_clusters", "end_state", HIT, t_onset=98.0, context="hard_impact", values={"cls": "hard_impact"})
    ok, note = gate(H._gate_impact, [hard])
    check(not ok and "未驗證" in note, f"沒有姿態資料 → 未驗證:{note}")
    hard = Finding("impact_shock_clusters", "end_state", HIT, t_onset=98.0, context="hard_impact", values={"cls": "hard_impact"})
    ok, _ = gate(H._gate_impact, [hard, att])
    check(ok, "姿態模組證實撞擊時姿態正常 → 通過")
    hard = Finding("impact_shock_clusters", "end_state", HIT, t_onset=98.0, context="hard_impact",
                   values={"cls": "hard_impact", "ae_at_impact": 25.0})
    ok, _ = gate(H._gate_impact, [hard, att])
    check(not ok, "撞擊當下姿態誤差 25° → 不通過")


def test_synth_unknown_end_not_ranked():
    """審查 probe 6:結束型態不明(沒有上鎖原因、沒有 EV18)且沒有姿態預警 / 撞擊 / 判定 → 不排名原因、只列觀察,
    時間軸不貼先兆 / 後果。"""
    ctx = flight_ctx(method=None, ev=((110.0, 28),))
    r = run_fake(ctx, anchors_setter(), [F("ekf_testratio_gate", "nav", HIT, 150.0, "air"),
                                         F("motor_thrust_loss_check", "motor", HIT, 160.0, "air")])
    check(r.end_type == "unknown" and r.verdict == "insufficient", f"{r.end_type} {r.verdict}")
    check(r.hypotheses == [] and r.chain == [] and r.top() is None, "不得排名")
    check("無法判定結束型態" in r.verdict_note and "無法判定" in r.summary_zh, r.summary_zh)
    check(sorted(o["id"] for o in r.observations) == ["ekf_testratio_gate", "motor_thrust_loss_check"], "觀察")
    check(all(f.tag in ("", TAG_STANDING) for f in r.findings), f"不貼因果標籤 {[f.tag for f in r.findings]}")
    check(all(e.tag in ("", TAG_GROUND) for e in r.timeline), "時間軸不貼因果標籤")
    text = r.format_text()
    check("假說排名" not in text and "第一名" not in text and "【觀察到的異常】" in text, "報告不得有排名")


def test_synth_timeline_vio_rows():
    """時間軸的 VIO 衍生事件:≥ 1 s 的 VISP 中斷、上鎖前就停止的最後一筆;已由 vio_dropout 列出的不重複。"""
    ctx = flight_ctx()
    tv = np.concatenate([np.arange(100.0, 185.0, 0.04), np.arange(186.5, 190.0, 0.04), np.arange(191.2, 198.0, 0.04)])
    ctx._tables[("VISP", None)] = {"t": tv, "PX": np.zeros_like(tv)}
    st = anchors_setter(end_type="crash_check", t_detect=205.0, t_warn=200.0, t_loss=201.0)
    r = run_fake(ctx, st, [F("vio_dropout", "nav", HIT, 185.0, "in_vio_mode")])
    vio = [e for e in r.timeline if e.category == "vio"]
    check(len(vio) == 2, f"VIO 列 {[(round(e.t, 2), e.label) for e in vio]}")
    check(abs(vio[0].t - 190.0) < 0.05 and "中斷 1.2" in vio[0].label and vio[0].tag == TAG_PRECURSOR, vio[0])
    check(abs(vio[1].t - 197.96) < 0.05 and "最後一筆" in vio[1].label, vio[1])
    check(not [e for e in vio if abs(e.t - 185.0) < 0.3], "已由 vio_dropout 列出的中斷不重複")


def test_124_report_wording_and_timeline():
    """審查 wording:正常飛行(飛行 1)不談失事;失事報告(飛行 2)的搖桿偏移有解釋、RC9 開關與飛手動作逐筆列在
    時間軸並標後果(A6)、時間軸有 log 絕對秒、表頭有 t_base 與 FS_CRASH_CHECK、錨點備註、Markdown 匯出。"""
    r1 = rep124(0)
    text = r1.format_text()
    check(text.startswith("飛行結束分析"), f"正常飛行的標題 {text.splitlines()[0]}")
    body = [ln for ln in text.splitlines() if not ln.startswith("  第 ") and "校準" not in ln]
    bad = [ln for ln in body if "失事" in ln]
    check(not bad, f"正常飛行報告仍有失事字眼:{bad[:2]}")
    check("FS_CRASH_CHECK" not in text, "正常飛行不列 FS_CRASH_CHECK")
    r = rep124()
    text = r.format_text()
    check(f"t_base({r.t_base:.4f} s)" in text and "FS_CRASH_CHECK = 1" in text, "表頭缺 t_base / FS_CRASH_CHECK")
    f = r.scenario_features
    near(f.get("stick_dev_t"), 1255.44, 0.15, "預警前搖桿最大偏移的時間")
    check(f.get("stick_dev_us") < R.STICK_PILOT_US and "未達飛手動作" in text, f"搖桿偏移 {f.get('stick_dev_us')} 沒有解釋")
    check("置中 ±30 µs" in text, "應附 lean 偵測器的搖桿置中容許值")
    rc9 = [e for e in r.timeline if e.category == "pilot" and e.source == "RCIN.C9"]
    check(len(rc9) == 2 and abs(rc9[0].t - 1258.64) <= 0.15 and abs(rc9[1].t - 1259.44) <= 0.15,
          f"RC9 切換 {[(e.t, e.label) for e in rc9]}")
    check(all(e.tag == TAG_CONSEQUENCE for e in rc9), f"RC9 切換應為後果 {[e.tag for e in rc9]}")
    pil = [e for e in r.timeline if e.category == "pilot" and e.source == "pilot_input_context"]
    check(pil and all(e.tag == TAG_CONSEQUENCE for e in pil), f"飛手動作 {[(e.t, e.tag) for e in pil]}")
    thr = [e for e in r.timeline if e.source == "RCIN.C3"]
    check(thr and abs(thr[0].t - 1256.54) <= 0.15, f"油門收到最低 {[(e.t, e.tag) for e in thr]}")
    check(all(e.t_abs is not None and abs(e.t_abs - (e.t + r.t_base)) < 1e-6 for e in r.timeline + r.ground_events),
          "時間軸缺 log 絕對秒")
    check(all("t_abs" in d for d in r.to_dict()["timeline"]), "JSON 時間軸缺 t_abs")
    check("【分析備註】" in text and "結束型態 crash_check" in text, "錨點備註沒印出")
    md = r.format_markdown()
    check(md.startswith("# 失事原因分析") and "\n## 結論\n" in md and "\n## 關鍵時間軸\n" in md, "Markdown 匯出格式")
    check("RC9 來源組開關" in md, "Markdown 應包含時間軸內容")



def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        t = time.time()
        try:
            fn()
            print(f"  ok({time.time() - t:.1f} s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  !! 失敗:{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 60}\n{len(tests) - failed}/{len(tests)} 通過")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
