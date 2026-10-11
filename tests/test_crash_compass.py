"""羅盤 / 航向偵測模組(core/crash/det_compass.py)驗證:python -m tests.test_crash_compass

分四層:
  1. 真實 log 00000124:飛行 2(truth 錨點 / 只有 CRASH_CHECK 的 basic 錨點)、飛行 1(正常降落對照)、
     飛行 2 在 1255 s 之前(健康懸停對照)、完整 engine 跑一次(羅盤不得被列為原因,驗收 A3 / A7 / A8);
  2. 六個 baseline(從未解鎖的桌上 / 手持 log)→ 只能 NA;拿掉解鎖閘門也不得 HIT(119 的羅盤校正窗要被排除);
  3. 合成資料(假 ctx,一架「健康」的飛行再逐項注入故障):證明偵測器沒有寫死 124 的數字 ——
     124 只有一次失事而且羅盤全是「後果」,所有『原因』分支只能靠合成資料驗證;
  4. 純函式(投票 / 併接 / 滾動參考 / 傾角補償航向 / 因果判定)與介面規則(6 個 id 必出現、
     values 可 JSON 序列化、例外降級為 NA、效能、確定性、沒有寫死 124 的時間)。
"""
from __future__ import annotations

import inspect
import io
import json
import os
import sys
import time
import tokenize

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import crashkit as ck  # noqa: E402
from core.crash import det_compass as dc  # noqa: E402
from core.crash.anchors import basic_anchors  # noqa: E402
from core.crash.types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval  # noqa: E402

IDS = ("ekf_mag_reject", "compass_raw_consistency", "compass_motor_interference",
       "ekf_yaw_reset_lane_switch", "compass_sensor_health", "ekf_attitude_consistency")
T_WARN_124 = 1255.9          # 姿態誤差 > 10°(truth 錨點)
TOL_10HZ = 0.15              # 10 Hz 序列的時間容許(驗收文件)


def check(cond, msg):
    assert cond, msg


def by_id(fs):
    out = {}
    for f in fs:
        out.setdefault(f.id, []).append(f)
    return out


def statuses(fs):
    return [(f.id, f.status, f.context) for f in fs]


def run_checked(ctx, a):
    """跑模組並檢查介面規則:6 個 id 都有且各一筆、沒有 error、values 可 JSON 序列化、text_zh 非空、
    HIT / SOFT 一定有 t_onset、CLEAR / NA 的 severity 是 info。"""
    fs = dc.run(ctx, a)
    d = by_id(fs)
    for i in IDS:
        check(i in d and len(d[i]) == 1, f"缺少偵測器 {i} 的 Finding:{statuses(fs)}")
    for f in fs:
        check(f.domain == "compass", f"{f.id} domain={f.domain}")
        check(f.context != "error", f"{f.id} 執行失敗:{f.values}")
        check(f.status in (HIT, SOFT, CLEAR, NA), f.status)
        check(bool(f.text_zh), f"{f.id} 沒有 text_zh")
        json.dumps(f.values, allow_nan=False)
        if f.status in (HIT, SOFT):
            check(f.t_onset is not None, f"{f.id} {f.status} 沒有 t_onset")
        else:
            check(f.severity == "info", f"{f.id} {f.status} severity 應為 info:{f.severity}")
    return {i: d[i][0] for i in IDS}


def real(name):
    return ck.load_ctx(name)


def near(x, ref, tol):
    return x is not None and abs(x - ref) <= tol


# ===================================================================== 1. 真實 log:00000124 飛行 2

def test_real_124_crash_truth_anchors():
    """驗收 A6 / A7:羅盤警報全部是翻機後果(SM>1 自 1259.70;|B0| 突變 1258.04 晚於 t_warn);事前 SM ≤ 0.15。"""
    ctx = real("00000124")
    a = ck.truth_anchors_124(ctx)
    r = run_checked(ctx, a)

    mr = r["ekf_mag_reject"]
    v = mr.values
    check(mr.status == HIT and mr.context == "symptom", f"{mr.status} {mr.context}")
    check(near(mr.t_onset, 1259.70, TOL_10HZ), f"SM>1 應自 1259.70 起(驗收 A6):{mr.t_onset}")
    check(v["verdict"] == "effect_not_cause" and v["lead_s"] < -3.0, f"應晚於 t_warn 3.8 s:{v['lead_s']}")
    check(v["lanes_alarmed"] == 3 and v["n_lanes"] == 3 and v["mode_at_alarm"] == "mag3d", v)
    check(3.3 <= v["peak_sm"] <= 3.6, f"SM 峰值約 3.40-3.49:{v['peak_sm']}")
    check(500 <= v["peak_im_mG"] <= 560 and v["peak_im_axis"] == "Y", f"|IMY| 峰值約 530 mG:{v['peak_im_mG']}")
    check(v["pre_sm_max"] is not None and v["pre_sm_max"] <= 0.16, f"事前 SM 最大 0.15(驗收 A7):{v['pre_sm_max']}")
    check(v["pre_im_max_mG"] is not None and v["pre_im_max_mG"] <= 43, f"事前 |IM| ≤ 正常飛行最大 43 mG:{v['pre_im_max_mG']}")
    check(near(v["pre_end_t"], 1255.225, 0.01), f"事前統計止於觸地:{v['pre_end_t']}")
    check(v["pre_iyaw_max_deg"] is not None and v["pre_iyaw_max_deg"] < 3.0, v["pre_iyaw_max_deg"])
    check(v["t_end"] <= a.interval.t1 + 1e-6, "只看解鎖區間(上鎖後 EKF 掉回模式 1、IYAW 暴衝不算)")
    check("不是原因" in mr.text_zh and "連帶現象" in mr.text_zh, mr.text_zh)

    rc = r["compass_raw_consistency"]
    v = rc.values
    check(rc.status == HIT and rc.context == "symptom", f"{rc.status} {rc.context}")
    check(near(rc.t_onset, 1258.04, TOL_10HZ) and v["first_alarm"] == "magnitude",
          f"第一個羅盤警報應是 |B0| 突變 1258.04(驗收 A7):{rc.t_onset} {v.get('first_alarm')}")
    check(near(v["alarm_times"]["heading"], 1259.64, TOL_10HZ), v["alarm_times"])
    check(near(v["alarm_times"]["pair"], 1259.84, TOL_10HZ), v["alarm_times"])
    check(v["verdict"] == "effect_not_cause" and -2.3 <= v["lead_s"] <= -1.9, f"晚於 t_warn 約 2.1 s:{v['lead_s']}")
    pm = v["checks"]["magnitude"]["pre_mag_range_mG"]
    check(425 <= pm[0] and pm[1] <= 460, f"事前 |B0| 約 430.7-450.4 mG:{pm}")
    check(v["checks"]["pair"]["pre_max_deg"] <= 17.6, f"事前雙羅盤夾角 ≤ 17.5°:{v['checks']['pair']['pre_max_deg']}")
    # 航向殘差改在扣掉馬達磁場 x·c 之後算:未補償 5.5°、補償後 6.0°(MAG[0] 干擾小,向量補償只多一點雜訊)
    check(v["checks"]["heading"]["pre_max_dev_deg"] <= 6.1, f"事前航向殘差 ≤ 6.0°:{v['checks']['heading']['pre_max_dev_deg']}")
    check(v["used_compass"] == 0 and v["gated_frac"] == 0.0, v)

    mi = r["compass_motor_interference"]
    v = mi.values
    check(mi.status == CLEAR and mi.context == "hygiene" and v["hygiene"], f"{mi.status} {mi.context}")
    check(v["used_compass"] == 0 and abs(v["used_ratio"] - 0.074) < 0.005, f"MAG[0] 飛行 2 約 7.4%:{v['used_ratio']}")
    check(v["backup_compass"] == 1 and abs(v["backup_ratio"] - 0.342) < 0.005, f"MAG[1] 飛行 2 約 34.2%:{v['backup_ratio']}")
    check(abs(v["backup_dB_scalar_mG"] - 87.6) < 3.0, v["backup_dB_scalar_mG"])
    seg = v["segments"][0]
    check(seg["index"] == 0 and abs(seg["ratios"]["0"] - 0.128) < 0.005 and abs(seg["ratios"]["1"] - 0.404) < 0.005,
          f"飛行 1:MAG[0] 12.8% / MAG[1] 40.4%:{seg}")
    check(v["flight_window"][1] < 1255.4, f"飛行集要在 t_warn − 0.5 前結束:{v['flight_window']}")
    check("不是失事原因" in mi.text_zh, mi.text_zh)

    yr = r["ekf_yaw_reset_lane_switch"]
    v = yr.values
    check(yr.status == CLEAR, f"{yr.status} {yr.text_zh}")
    check(near(v["last_reset_t"], 882.49, 0.05) and v["last_reset_kind"] == "inflight_alignment",
          f"最近一次航向重置 882.5 s(驗收 A7):{v['last_reset_t']} {v['last_reset_kind']}")
    r882 = [x for x in v["resets"] if x["kind"] == "inflight_alignment"][0]
    j0 = r882["jump_by_lane"]["0"]
    check(near(j0, -0.14, 0.1), f"882.49 lane 0 的實際階躍 87.60 → 87.46(之後 0.1 s 才開始轉向):{j0}")
    # 分類用三條 lane 中 |跳變| 最大者(lane 2 −0.31°),仍遠小於 10°
    check(r882["jump_lane"] == "2" and near(r882["jump_deg"], -0.31, 0.05), r882)
    check(v["lane_switches"] == [] and v["evidence"] == [], v)
    sw = [(round(x["t"], 1), x["to_set"], x["yaw_changed"]) for x in v["source_switches"]]
    check(sw == [(919.0, 3, False), (1258.8, 2, False), (1259.6, 3, False)], f"來源組切換不改航向來源:{sw}")
    check(v["lane_yaw_spread_max_deg"] <= 1.83, f"lane 航向差(時間對齊)≤ 1.83°:{v['lane_yaw_spread_max_deg']}")

    sh = r["compass_sensor_health"]
    v = sh.values
    check(sh.status == CLEAR and v["alarms"] == [], f"{sh.status} {v['alarms']}")
    check(v["n_compasses"] == 2 and (v["ts_max"] & dc.SH_TS_MAG_BIT) == 0 and v["fs_max"] == 0, v)
    check(abs(v["max_gap_s"]["0"] - 0.80) < 0.02, f"MAG 最大間隔 0.8 s(956 s):{v['max_gap_s']}")

    ac = r["ekf_attitude_consistency"]
    v = ac.values
    check(ac.status == CLEAR and ac.context == "clear", f"{ac.status} {ac.context}")
    check(v["event_max_droll_deg"] <= 3.25 and v["event_max_dpitch_deg"] <= 1.95 and v["event_max_dyaw_deg"] <= 4.55,
          f"EKF vs AHR2 ≤ 3.2 / 1.9 / 4.5°(驗收 A7):{v['event_max_droll_deg']} {v['event_max_dpitch_deg']} {v['event_max_dyaw_deg']}")
    check(v["max_dpitch_deg"] <= 7.8 and v["max_dcm_yaw_deg"] <= 6.4, v)
    check(2.8 <= v["median_dcm_yaw_deg"] <= 3.2, f"ATT − DCM 偏航固定約 3.0°:{v['median_dcm_yaw_deg']}")
    check(v["window"][1] <= a.interval.t1, v["window"])
    print(f"  SM>1 {mr.t_onset:.2f}(lead {mr.values['lead_s']:.1f} s)、|B0| 突變 {rc.t_onset:.2f}、"
          f"MAG[0]/[1] 干擾 {mi.values['used_ratio']:.3f}/{mi.values['backup_ratio']:.3f}、"
          f"AHR2 差 {v['event_max_droll_deg']:.1f}/{v['event_max_dpitch_deg']:.1f}/{v['event_max_dyaw_deg']:.1f}°")


def test_real_124_basic_anchors_only():
    """只有 CRASH_CHECK(沒有 t_warn / t_touch)時,以「傾角 ≥45°」為因果參考:兩個 HIT 仍是後果,事前統計不含翻滾。"""
    ctx = real("00000124")
    a = basic_anchors(ctx)
    check(a.t_warn is None and a.t_touch is None, "basic 錨點不該有 t_warn / t_touch")
    r = run_checked(ctx, a)
    mr, rc = r["ekf_mag_reject"], r["compass_raw_consistency"]
    check(mr.status == HIT and mr.values["verdict"] == "effect_not_cause" and mr.values["after_tilt45"], mr.values)
    check(near(mr.values["t_tilt45"], 1258.14, TOL_10HZ), mr.values["t_tilt45"])
    check(near(mr.values["pre_end_t"], mr.values["t_tilt45"], 1e-6), f"事前統計止於傾角 45°:{mr.values['pre_end_t']}")
    check(mr.values["pre_sm_max"] <= 0.30, f"觸地 → 翻滾期間 SM 最大 0.29:{mr.values['pre_sm_max']}")
    check(rc.status == HIT and rc.values["verdict"] == "effect_not_cause" and rc.context == "symptom", rc.values)
    check(0.0 < rc.values["lead_tilt45_s"] < dc.LEAD_MIN_S, f"|B0| 突變只比 45° 早 0.1 s:{rc.values['lead_tilt45_s']}")
    check("翻機的連帶現象" in rc.text_zh, rc.text_zh)
    for k in ("compass_motor_interference", "ekf_yaw_reset_lane_switch", "compass_sensor_health", "ekf_attitude_consistency"):
        check(r[k].status == CLEAR, f"{k}: {r[k].status}")


def test_real_124_flight1_control():
    """驗收 B:飛行 1(正常降落)是陰性對照 —— 6 個偵測器全部 CLEAR(不是 NA),沒有 HIT / SOFT。"""
    ctx = real("00000124")
    a = ck.flight1_anchors_124(ctx)
    r = run_checked(ctx, a)
    check(all(f.status == CLEAR for f in r.values()), statuses(r.values()))
    check(all("失事" not in f.text_zh for f in r.values()), [f.text_zh for f in r.values() if "失事" in f.text_zh])
    mr = r["ekf_mag_reject"].values
    check(mr["pre_sm_max"] <= 0.16 and mr["pre_im_max_mG"] < 50 and mr["pre_iyaw_max_deg"] < 3.0, mr)
    check(mr["coverage_3d_s"] > 250, mr["coverage_3d_s"])
    ch = r["compass_raw_consistency"].values["checks"]
    check(ch["magnitude"]["max_dev_mG"] <= 20 and ch["pair"]["max_deg"] <= 17.5 and ch["heading"]["max_dev_deg"] <= 5.5,
          f"飛行 1 實測 16 mG / 17.0° / 3.9°:{ch['magnitude']['max_dev_mG']} {ch['pair']['max_deg']} {ch['heading']['max_dev_deg']}")
    mi = r["compass_motor_interference"].values
    check(abs(mi["used_ratio"] - 0.128) < 0.005 and abs(mi["backup_ratio"] - 0.404) < 0.005, mi)
    yr = r["ekf_yaw_reset_lane_switch"].values
    check(near(yr["last_reset_t"], 344.89, 0.05) and yr["last_reset_kind"] == "inflight_alignment", yr)
    r345 = [x for x in yr["resets"] if x["kind"] == "inflight_alignment"][0]
    check(near(r345["jump_by_lane"]["0"], 1.5, 0.1), f"344.89 空中對準階躍 89.69 → 91.19:{r345['jump_by_lane']}")
    check(near(r345["jump_deg"], 1.59, 0.05) and r345["jump_lane"] == "1", f"三條 lane 1.51 / 1.59 / 1.43°:{r345}")
    check(len(yr["source_switches"]) == 2 and not any(x["yaw_changed"] for x in yr["source_switches"]), yr["source_switches"])
    ac = r["ekf_attitude_consistency"].values
    check(ac["event_ref_t"] is None and "event_max_droll_deg" not in ac, "正常降落沒有事件窗")
    check(ac["max_droll_deg"] < 15 and ac["max_dpitch_deg"] < 15 and ac["max_dyaw_deg"] < 20, ac)
    print(f"  飛行 1 全 CLEAR:SM {mr['pre_sm_max']}、夾角 {ch['pair']['max_deg']:.1f}°、MAG[1] {mi['backup_ratio']:.3f}")


def test_real_124_flight2_before_1255():
    """驗收 B 最後一句:飛行 2 在 1255.0 之前(健康懸停 376 s)沒有 HIT / SOFT。"""
    ctx = real("00000124")
    iv = ctx.armed[1]
    sub = ArmedInterval(1, iv.t0, 1255.0, 0)
    a = Anchors(interval=sub, intervals=list(ctx.armed))
    a.end_type = "unknown"
    r = run_checked(ctx, a)
    check(all(f.status == CLEAR for f in r.values()), statuses(r.values()))
    check(r["ekf_mag_reject"].values["pre_sm_max"] <= 0.16, r["ekf_mag_reject"].values)
    ch = r["compass_raw_consistency"].values["checks"]
    check(ch["pair"]["max_deg"] <= 17.6 and ch["heading"]["max_dev_deg"] <= 6.1 and ch["magnitude"]["max_dev_mG"] <= 25, ch)


def test_real_124_engine_compass_not_cause():
    """驗收 A3 / A7 / A8:完整 engine —— 羅盤 HIT 都標為後果、H_COMPASS_YAW 0 分且已排除,不得列為原因。"""
    from core import logio
    from core.crash import engine
    rep = engine.analyze(logio.load_bin(ck.real_path("00000124")))
    comp = [f for f in rep.findings if f.domain == "compass"]
    check(sorted(f.id for f in comp) == sorted(IDS), [f.id for f in comp])
    for f in comp:
        if f.status in (HIT, SOFT):
            check(f.tag == "consequence", f"{f.id} {f.status} 應標為後果:{f.tag}")
            check(f.values.get("verdict") == "effect_not_cause", f"{f.id} verdict={f.values.get('verdict')}")
    check(rep.finding("compass_motor_interference").tag == "standing", "馬達干擾是持續性健康指標(standing)")
    h = [x for x in rep.hypotheses if x.id == "H_COMPASS_YAW"]
    check(h and h[0].score == 0 and h[0].status == "ruled_out", f"H_COMPASS_YAW 應 0 分且已排除:{h and (h[0].score, h[0].status)}")
    check(not h[0].gate_ok, "羅盤警報都晚於姿態失控 → 必要證據不成立")
    top = rep.top()
    check(top is not None and top.id != "H_COMPASS_YAW", f"第一名不得是羅盤:{top and top.id}")
    print(f"  engine:H_COMPASS_YAW={h[0].score} {h[0].status};第一名 {top.id} {top.score:.1f}")


class UseMag1:
    """00000124 的「假設情境」:XKFS.MI 改成 1(EKF 用干擾 34-40% 的 IST8310 —— 業餘機架常見的擺法)。
    其他資料不動,所以 EKF 航向其實仍來自 MAG[0];用來驗證馬達電流不會變成羅盤警報。"""

    def __init__(self, c):
        self.c, self.armed, self.messages, self.params = c, c.armed, c.messages, c.params

    def table(self, name, instance=None):
        tb = self.c.table(name, instance)
        if name == "XKFS" and tb is not None:
            tb = dict(tb)
            tb["MI"] = np.ones(tb["t"].size)
        return tb

    def __getattr__(self, k):
        return getattr(self.c, k)


def test_real_124_used_compass_motor_current_not_alarm():
    """審查缺陷(major):EKF 用的羅盤有一般的馬達電流干擾(124 MAG[1] 34-40%)時,起飛 / 降落的油門階躍、
    懸停中的快速轉向不得變成 compass_raw_consistency HIT(修正前:飛行 1 340.34 s HIT 223 mG、飛行 2 878.64 s HIT,
    帶 truth 錨點時被判 possible_cause、早 377 s)。馬達干擾偵測器同時把這顆評為「邊際」(SOFT),兩者一致。"""
    base = real("00000124")
    ctx = UseMag1(base)
    a1 = ck.flight1_anchors_124(base)
    f1 = dc.compass_raw_consistency(ctx, a1)[0]
    ch = f1.values["checks"]
    check(f1.status == CLEAR and f1.values["used_compass"] == 1, f"飛行 1(正常降落)不得 HIT:{f1.status} {f1.text_zh}")
    check(ch["magnitude"]["max_sustained_dev_mG"] < dc.RC_MAG_JUMP_MG and ch["heading"]["max_sustained_dev_deg"] < dc.RC_HEAD_DEG, ch)
    exp1 = ch["magnitude"]["motor_explained"]
    check(any(near(e["t"], 340.4, 0.3) for e in exp1), f"起飛的 |B| 變化要歸因馬達電流:{exp1}")
    mi1 = dc.compass_motor_interference(ctx, a1)[0]
    check(mi1.status == SOFT and mi1.values["used_compass"] == 1 and abs(mi1.values["used_ratio"] - 0.404) < 0.01, mi1.values)
    iv = base.armed[1]
    sub = ArmedInterval(1, iv.t0, 1255.0, 0)
    a2 = Anchors(interval=sub, intervals=list(base.armed))
    a2.end_type = "unknown"
    f2 = dc.compass_raw_consistency(ctx, a2)[0]
    ch2 = f2.values["checks"]
    check(f2.status == CLEAR, f"飛行 2 在 1255 之前不得 HIT(修正前 878.64 起飛 HIT、1053.5 轉向 50° 航向 20°):{f2.text_zh}")
    check(ch2["heading"]["motor_compensated"] and ch2["heading"]["max_dev_deg"] < 13.0, ch2["heading"])
    check(any(near(e["t"], 878.7, 0.3) for e in ch2["magnitude"]["motor_explained"]), ch2["magnitude"]["motor_explained"])
    check(f2.values["motor_model"]["1"]["mag_mode"] == "vector", f2.values["motor_model"])
    # 帶 truth 錨點:不得出現早於 t_warn 的羅盤警報(修正前 possible_cause、lead 377 s)
    ft = dc.compass_raw_consistency(ctx, ck.truth_anchors_124(base))[0]
    check(ft.values.get("verdict") != "possible_cause" and (ft.t_onset is None or ft.t_onset > T_WARN_124),
          f"{ft.status} {ft.t_onset} {ft.values.get('verdict')}")
    # 同一份資料、EKF 用 MAG[0](實際情形):A7 的 |B0| 1258.04 照常警報
    fr = dc.compass_raw_consistency(base, ck.truth_anchors_124(base))[0]
    check(fr.status == HIT and near(fr.t_onset, 1258.04, TOL_10HZ) and fr.values["first_alarm"] == "magnitude"
          and fr.values["checks"]["magnitude"]["motor_explained"] == [], fr.values.get("alarm_times"))
    print(f"  MI=1:飛行 1 / 飛行 2 前段 CLEAR(起飛 {exp1[0]['t']:.2f} 歸因馬達 {exp1[0]['motor_mG']:.0f} mG);"
          f"MI=0:|B0| 仍 {fr.t_onset:.2f} 警報")


# ===================================================================== 2. baseline(從未解鎖)

def test_real_baselines_not_armed_only_na():
    """驗收 C:6 個從未解鎖的 log → 6 個偵測器都只能 NA 'not_armed'(不能是 CLEAR)。"""
    for name in ck.BASELINES:
        ctx = real(name)
        a = ck.not_armed_anchors(ctx)
        check(a.interval is None, f"{name} 應該沒有解鎖區間")
        r = run_checked(ctx, a)
        check(all(f.status == NA and f.context == "not_armed" for f in r.values()), (name, statuses(r.values())))
        check(all("未解鎖" in f.text_zh for f in r.values()), name)
    print(f"  {len(ck.BASELINES)} 個 baseline 全部 NA/not_armed")


def test_real_baselines_gate_removed_no_hit():
    """把整段 baseline 當成解鎖區間(閘門被拿掉):羅盤偵測器不得 HIT。

    - 數值要落在研究階段的實測範圍(IYAW ≤ 9.9°、|B| 偏離 ≤ 29.4 mG、夾角 ≤ 21.3°、航向殘差 ≤ 4.0°、
      EKF vs AHR2 ≤ 3.3 / 3.8 / 10.0°);
    - 119 的羅盤校正(MAG.Ofs 在 506.0 變動)要被找到並排除(驗收 C:'compass calibration window gated');
    - 開機 3-4 s 的 EKF 初始航向對準(ATT.Yaw 停在 0/360)不得造成 EKF vs AHR2 假警報;
    - 只容許 118 / 123 在 EKF 失效保護時真實發生的 'EKF3 lane switch'(地面事件)成為 SOFT。
    """
    soft_ok = {"00000118": [94.26], "00000123": [212.86]}
    for name in ck.BASELINES:
        ctx = real(name)
        iv = ArmedInterval(0, 0.0, ctx.t_end)
        a = Anchors(interval=iv, intervals=[iv])
        a.t_disarm = ctx.t_end
        r = run_checked(ctx, a)
        check(not any(f.status == HIT for f in r.values()), (name, statuses(r.values())))
        for f in r.values():
            if f.status == SOFT:
                check(f.id == "ekf_yaw_reset_lane_switch" and name in soft_ok, (name, f.id, f.text_zh))
                check(near(f.t_onset, soft_ok[name][0], 0.05) and f.values["first_kind"] == "lane_switch", f.values)
        mr = r["ekf_mag_reject"].values
        check(mr["pre_iyaw_max_deg"] is not None and mr["pre_iyaw_max_deg"] <= 10.0, (name, mr["pre_iyaw_max_deg"]))
        check(mr["coverage_3d_s"] == 0.0, f"{name} 從未進入 3D 磁力計融合:{mr['coverage_3d_s']}")
        ch = r["compass_raw_consistency"].values["checks"]
        check(ch["magnitude"]["max_dev_mG"] <= 29.5 and ch["pair"]["max_deg"] <= 21.4 and ch["heading"]["max_dev_deg"] <= 4.1,
              (name, ch["magnitude"]["max_dev_mG"], ch["pair"]["max_deg"], ch["heading"]["max_dev_deg"]))
        ac = r["ekf_attitude_consistency"]
        check(ac.status == CLEAR and ac.values["window"][0] >= dc.AC_BOOT_S, (name, ac.status, ac.values["window"]))
        check(ac.values["max_droll_deg"] <= 3.4 and ac.values["max_dpitch_deg"] <= 3.9 and ac.values["max_dyaw_deg"] <= 10.1,
              (name, ac.values["max_droll_deg"], ac.values["max_dpitch_deg"], ac.values["max_dyaw_deg"]))
        check(r["compass_motor_interference"].status == NA, f"{name}:馬達從未轉 → 干擾量 NA")
        if name == "00000119":
            wins = dc._cal_windows(dc._mag_tables(ctx))
            # Ofs 在 455.0(Mag(1) good orientation)與 506.0(Mag(0))兩次變動 → 合併窗 [335.0, 511.0]
            check(len(wins) == 1 and near(wins[0][0], 335.0, 0.2) and near(wins[0][1], 511.0, 0.2), f"119 校正窗:{wins}")
            check(0.30 < r["compass_raw_consistency"].values["gated_frac"] < 0.37, r["compass_raw_consistency"].values["gated_frac"])
    print("  閘門拿掉:baseline 沒有任何羅盤 HIT;119 校正窗已排除")


# ===================================================================== 3. 合成資料

DEC_DEG = -4.7                                   # 磁偏角(EKF 航向 − 羅盤航向的常數偏差)
B_NED = np.array([250.0 * np.cos(np.radians(DEC_DEG)), 250.0 * np.sin(np.radians(DEC_DEG)), 370.0])   # |B| ≈ 446 mG
HOVER_THR = 0.15


def truth_att(s, yaw0=90.0, turns=()):
    """真實姿態(度):小幅 roll / pitch 擺動、航向緩慢左右轉 ±30°;turns = [(開始, 角度, 秒)] 快速轉向(等速)。"""
    roll = 3.0 * np.sin(2 * np.pi * s / 7.0)
    pitch = 2.0 * np.sin(2 * np.pi * s / 5.0 + 1.0)
    yaw = yaw0 + 30.0 * np.sin(2 * np.pi * s / 120.0)
    for ts, deg, dur in turns:
        yaw = yaw + deg * np.clip((s - ts) / dur, 0.0, 1.0)
    return roll, pitch, yaw


def body_field(roll, pitch, yaw):
    """地磁(NED)→ 機體系:b = Rᵀ·B_ned。"""
    R = dc._rot_body_to_ned(roll, pitch, yaw)
    return np.stack([R[0][k] * B_NED[0] + R[1][k] * B_NED[1] + R[2][k] * B_NED[2] for k in range(3)], axis=1)


class World:
    """一架健康的合成飛行:開機 0 s、解鎖 t0、上鎖 t1、上鎖後再記錄 tail 秒;所有時間 + off(平移測試)。

    MAG 10 Hz 兩顆(MAG[1] 有固定偏移,干擾比例 ratios 沿機體方向 interf_dir 與油門成正比;預設 z 軸)、
    ATT / AHR2 / DCM / MOTB 10 Hz、XKF1/3/4/S 三條 lane 25 Hz(時間戳錯開)、EV 62 在 3.8 s(初始對準)與 t0+9
    (空中對準)、ERR COMPASS 0 開機雜訊。油門:解鎖後 idle_thr 5 s → 懸停 0.15 → 上鎖前 3 s 歸零;
    turns = [(開始, 角度, 秒)] 快速轉向(所有姿態表一致)。測試直接修改 self.tb 裡的陣列來注入故障(遮罩用 win())。
    """

    def __init__(self, t0=100.0, t1=400.0, tail=30.0, off=0.0, seed=0, n_lanes=3, n_mags=2,
                 ratios=(0.04, 0.40), yaw0=90.0, interf_dir=(0.0, 0.0, 1.0), turns=(), idle_thr=0.05):
        rng = np.random.default_rng(seed)
        self.off = off
        T = t1 + tail
        self.armed = [ArmedInterval(0, t0 + off, t1 + off, 0)]
        self.params = {"COMPASS_USE": 1.0, "COMPASS_USE2": 1.0, "COMPASS_ORIENT": 0.0, "COMPASS_ORIENT2": 0.0,
                       "EK3_SRC1_YAW": 1.0, "EK3_SRC2_YAW": 1.0, "EK3_SRC3_YAW": 1.0}
        self.ev = [(3.8, 62), (t0 + 9.0, 62)]
        self.err = [(0.05, 3, 0)]
        self.msgs: list[tuple[float, str]] = []
        self.tb: dict[str, dict[int, dict]] = {}
        s = np.arange(0.0, T, 0.1)
        roll, pitch, yaw = truth_att(s, yaw0, turns)
        thr = np.zeros(s.size)
        thr[(s >= t0) & (s < t0 + 5.0)] = idle_thr
        thr[(s >= t0 + 5.0) & (s <= t1 - 3.0)] = HOVER_THR
        self.tb["ATT"] = {0: {"t": s + off, "Roll": roll, "Pitch": pitch, "Yaw": np.mod(yaw, 360.0)}}
        self.tb["AHR2"] = {0: {"t": s + 0.02 + off, "Roll": roll + rng.normal(0, 0.3, s.size),
                               "Pitch": pitch + rng.normal(0, 0.3, s.size), "Yaw": np.mod(yaw + rng.normal(0, 0.3, s.size), 360.0)}}
        self.tb["DCM"] = {0: {"t": s + 0.04 + off, "Yaw": np.mod(yaw + 3.0 + rng.normal(0, 0.3, s.size), 360.0)}}
        self.tb["MOTB"] = {0: {"t": s + off, "ThrOut": thr}}
        sm_ = np.arange(0.05, T, 0.1)
        b = body_field(*truth_att(sm_, yaw0, turns))
        thr_m = np.interp(sm_, s, thr)
        offsets = [np.zeros(3), np.array([15.0, -10.0, 20.0])]
        udir = np.asarray(interf_dir, dtype=float) / np.linalg.norm(interf_dir)
        mags = {}
        for i in range(n_mags):
            clean = b + offsets[i]
            bi = clean + rng.normal(0, 1.0, b.shape)
            gm = float(np.linalg.norm(clean, axis=1).mean())
            bi += (ratios[i] * gm / HOVER_THR * thr_m)[:, None] * udir[None, :]   # 馬達電流干擾(機體系固定方向,與油門成正比)
            mags[i] = {"t": sm_ + off, "MagX": bi[:, 0], "MagY": bi[:, 1], "MagZ": bi[:, 2], "Health": np.ones(sm_.size),
                       "OfsX": np.full(sm_.size, 10.0), "OfsY": np.full(sm_.size, -5.0), "OfsZ": np.full(sm_.size, 3.0)}
        self.tb["MAG"] = mags
        for name in ("XKF1", "XKF3", "XKF4", "XKFS"):
            self.tb[name] = {}
        for ln in range(n_lanes):
            sl = np.arange(0.013 * ln, T, 0.04)
            mf = np.where(sl < t0 + 9.0, 1.0, np.where(sl <= t1, 2.0, 1.0))
            mf[sl < 1.0] = 0.0
            r_, p_, y_ = truth_att(sl, yaw0, turns)
            n = sl.size
            self.tb["XKF1"][ln] = {"t": sl + off, "Roll": r_ + 0.1 * ln, "Pitch": p_, "Yaw": np.mod(y_ + 0.2 * ln, 360.0)}
            self.tb["XKF3"][ln] = {"t": sl + off, "IMX": np.where(mf == 2, np.round(rng.normal(0, 4, n)), 0.0),
                                   "IMY": np.where(mf == 2, np.round(rng.normal(0, 4, n)), 0.0),
                                   "IMZ": np.where(mf == 2, np.round(rng.normal(0, 4, n)), 0.0),
                                   "IYAW": np.where(mf == 1, rng.normal(0, 1.0, n), 0.0)}
            self.tb["XKF4"][ln] = {"t": sl + off, "SM": np.where(mf == 2, np.abs(rng.normal(0.03, 0.02, n)), np.abs(rng.normal(0.01, 0.01, n))),
                                   "TS": np.full(n, 48.0), "FS": np.where(sl < 1.0, 128.0, 0.0)}
            self.tb["XKFS"][ln] = {"t": sl + off, "MAG_FUSION": mf, "MI": np.zeros(n), "SS": np.zeros(n)}

    def win(self, name, inst, ta, tb):
        """表 name[inst] 在開機相對時間 [ta, tb) 的遮罩。"""
        t = self.tb[name][inst]["t"] - self.off
        return (t >= ta) & (t < tb)

    def drop(self, name, inst, ta, tb):
        """刪掉 [ta, tb) 的樣本(資料空洞)。"""
        keep = ~self.win(name, inst, ta, tb)
        self.tb[name][inst] = {k: v[keep] for k, v in self.tb[name][inst].items()}

    def ctx(self):
        return FakeCtx(self)

    def anchors(self, **kw):
        a = Anchors(intervals=list(self.armed))
        a.interval = self.armed[-1]
        a.t_disarm = a.interval.t1
        a.end_type = kw.pop("end_type", "normal_landing")
        for k, v in kw.items():
            setattr(a, k, (v + self.off) if (k.startswith("t_") and v is not None) else v)
        return a


class FakeCtx:
    """只實作 det_compass 用到的 ctx 介面:armed / messages / table() / param() / ev_rows() / err_rows()。"""

    def __init__(self, w: World | None = None, armed=None):
        self.armed = list(w.armed) if w is not None else list(armed or [])
        self._tb = w.tb if w is not None else {}
        self.params = dict(w.params) if w is not None else {}
        off = w.off if w is not None else 0.0
        self.messages = sorted((t + off, s) for t, s in (w.msgs if w is not None else []))
        self._ev = sorted((t + off, i) for t, i in (w.ev if w is not None else []))
        self._err = sorted((t + off, s, c) for t, s, c in (w.err if w is not None else []))

    def table(self, name, instance=None):
        d = self._tb.get(name)
        if not d:
            return None
        if instance is None:
            instance = min(d)
        return d.get(instance)

    def param(self, name, default=None):
        return self.params.get(name, default)

    def ev_rows(self):
        if not self._ev:
            return np.zeros(0), np.zeros(0, dtype=int)
        e = np.array(self._ev, dtype=float)
        return e[:, 0], e[:, 1].astype(int)

    def err_rows(self):
        if not self._err:
            return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        e = np.array(self._err, dtype=float)
        return e[:, 0], e[:, 1].astype(int), e[:, 2].astype(int)


def one(fn, w, **anchor_kw):
    return fn(w.ctx(), w.anchors(**anchor_kw))[0]


def test_synth_healthy_flight_all_clear():
    """健康的合成飛行:6 個偵測器全部 CLEAR,數值合理(偏航對準重置是良性的、干擾比例被估回來)。"""
    w = World()
    r = run_checked(w.ctx(), w.anchors())
    check(all(f.status == CLEAR for f in r.values()), statuses(r.values()))
    check(r["ekf_mag_reject"].values["pre_sm_max"] < 0.2 and r["ekf_mag_reject"].values["coverage_3d_s"] > 250,
          r["ekf_mag_reject"].values)
    ch = r["compass_raw_consistency"].values["checks"]
    check(all(c["state"] == "ok" for c in ch.values()), {k: c["state"] for k, c in ch.items()})
    check(ch["heading"]["max_dev_deg"] < 3.0 and ch["magnitude"]["max_dev_mG"] < 25, ch)
    mi = r["compass_motor_interference"].values
    check(abs(mi["used_ratio"] - 0.04) < 0.01 and abs(mi["backup_ratio"] - 0.40) < 0.02, f"{mi['used_ratio']} {mi['backup_ratio']}")
    check(mi["hygiene"] and r["compass_motor_interference"].context == "hygiene", "備援 40% → 衛生項目")
    yr = r["ekf_yaw_reset_lane_switch"].values
    check([x["kind"] for x in yr["resets"]] == ["initial_alignment", "inflight_alignment"], yr["resets"])
    check(r["ekf_attitude_consistency"].values["median_dcm_yaw_deg"] > 2.5, "DCM 固定 3° 偏差是正常的")


# ---------- ekf_mag_reject

def _sm(w, lanes, ta, tb, val=2.0, col="SM", name="XKF4"):
    for ln in lanes:
        m = w.win(name, ln, ta, tb)
        w.tb[name][ln][col][m] = val


def test_synth_mag_reject_two_lanes_precursor():
    w = World()
    _sm(w, (0, 1), 250.0, 252.0)
    f = one(dc.ekf_mag_reject, w, t_warn=260.0, end_type="crash_check")
    check(f.status == HIT and 249.95 <= f.t_onset <= 250.1, f"{f.status} {f.t_onset}")
    v = f.values
    check(v["lanes_alarmed"] == 2 and v["verdict"] == "possible_cause" and 9.8 <= v["lead_s"] <= 10.1, v)
    check(f.context == "precursor_candidate" and "可能是原因" in f.text_zh, f"{f.context} {f.text_zh}")
    check(v["pre_sm_max"] < 0.2 and v["pre_end_t"] <= 250.1, "事前統計必須在警報之前結束")
    check(abs(v["peak_sm"] - 2.0) < 1e-9 and v["mode_at_alarm"] == "mag3d", v)
    # 同樣的警報出現在 t_warn 之後 0.5 s 內:未達 1 s 前導 → 後果
    g = one(dc.ekf_mag_reject, w, t_warn=250.5, end_type="crash_check")
    check(g.status == HIT and g.values["verdict"] == "effect_not_cause" and g.context == "symptom", g.values)


def test_synth_mag_reject_single_lane_soft_and_short_clear():
    w = World()
    _sm(w, (2,), 250.0, 252.0)
    f = one(dc.ekf_mag_reject, w)
    check(f.status == SOFT and f.context == "single_lane" and f.values["lanes_alarmed"] == 1, f"{f.status} {f.context}")
    check("僅 1 組" in f.text_zh, f.text_zh)
    w2 = World()
    _sm(w2, (0, 1, 2), 250.0, 250.3)                       # 0.3 s < 0.5 s
    check(one(dc.ekf_mag_reject, w2).status == CLEAR, "短於 0.5 s 不算")
    w3 = World(n_lanes=1)                                  # 只有一條 lane:需要的 lane 數 = 1
    _sm(w3, (0,), 250.0, 252.0)
    g = one(dc.ekf_mag_reject, w3)
    check(g.status == HIT and g.values["lanes_needed"] == 1, f"{g.status} {g.values['lanes_needed']}")


def test_synth_mag_reject_bridge_single_dip():
    """SM 約 8 Hz 更新:單次更新跌破 1(0.72)不能把一次事件切成兩段,t_onset 是第一段的起點。"""
    w = World()
    _sm(w, (0, 1), 250.0, 250.12, 2.0)
    _sm(w, (0, 1), 250.12, 250.2, 0.72)
    _sm(w, (0, 1), 250.2, 251.0, 2.0)
    f = one(dc.ekf_mag_reject, w)
    check(f.status == HIT and 249.95 <= f.t_onset <= 250.05, f"併接後 onset 應是 250.0:{f.status} {f.t_onset}")
    # 跌破的時間 > 0.15 s:兩段分開,第一段太短 → onset 在第二段
    w2 = World()
    _sm(w2, (0, 1), 250.0, 250.12, 2.0)
    _sm(w2, (0, 1), 250.12, 250.4, 0.72)
    _sm(w2, (0, 1), 250.4, 251.2, 2.0)
    g = one(dc.ekf_mag_reject, w2)
    check(g.status == HIT and 250.35 <= g.t_onset <= 250.45, f"{g.status} {g.t_onset}")


def test_synth_mag_reject_innovation_gate():
    w = World()
    _sm(w, (0, 1), 250.0, 251.0, 150.0, col="IMY", name="XKF3")
    f = one(dc.ekf_mag_reject, w)
    check(f.status == HIT and f.values["peak_im_axis"] == "Y" and f.values["peak_im_mG"] == 150.0, f.values)
    w2 = World()
    _sm(w2, (0, 1), 250.0, 251.0, -90.0, col="IMZ", name="XKF3")      # |IM| 90 < 100
    check(one(dc.ekf_mag_reject, w2).status == CLEAR, "|IM| < 100 mG 不算")


def test_synth_mag_reject_mode_selection():
    """依 MAG_FUSION 選訊號:模式 1 看 IYAW;模式 2 的 IYAW、模式 1 的 SM 不算;上鎖後的不算。"""
    w = World()
    for ln in range(3):
        w.tb["XKFS"][ln]["MAG_FUSION"][w.win("XKFS", ln, 300.0, 310.0)] = 1.0
    _sm(w, (0, 1, 2), 302.0, 303.5, 20.0, col="IYAW", name="XKF3")
    f = one(dc.ekf_mag_reject, w)
    check(f.status == HIT and f.values["mode_at_alarm"] == "yaw_only" and abs(f.values["peak_iyaw_deg"] - 20.0) < 1e-9, f.values)
    check(301.95 <= f.t_onset <= 302.1 and "地面偏航創新量" in f.text_zh, f"{f.t_onset} {f.text_zh}")
    w2 = World()
    _sm(w2, (0, 1, 2), 302.0, 304.0, 30.0, col="IYAW", name="XKF3")   # 模式 2 時 IYAW 無意義
    _sm(w2, (0, 1, 2), 50.0, 52.0, 3.0)                                  # 解鎖前(模式 1)的 SM
    _sm(w2, (0, 1, 2), 405.0, 410.0, 3.0)                                # 上鎖後
    _sm(w2, (0, 1, 2), 405.0, 410.0, 60.0, col="IYAW", name="XKF3")    # 上鎖後掉回模式 1 的 IYAW 暴衝
    g = one(dc.ekf_mag_reject, w2)
    check(g.status == CLEAR, f"{g.status} {g.text_zh}")


# ---------- compass_raw_consistency

def _scale_mag(w, i, ta, tb, k):
    m = w.win("MAG", i, ta, tb)
    for c in ("MagX", "MagY", "MagZ"):
        w.tb["MAG"][i][c][m] *= k


def _rotate_mag(w, i, ta, tb, deg):
    m = w.win("MAG", i, ta, tb)
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    x, y = w.tb["MAG"][i]["MagX"][m].copy(), w.tb["MAG"][i]["MagY"][m].copy()
    w.tb["MAG"][i]["MagX"][m] = c * x - s * y
    w.tb["MAG"][i]["MagY"][m] = s * x + c * y


def test_synth_raw_magnitude_jump():
    w = World()
    _scale_mag(w, 0, 250.0, 252.0, 1.15)                   # |B| +67 mG,方向不變
    f = one(dc.compass_raw_consistency, w, t_warn=260.0)
    v = f.values
    check(f.status == HIT and v["first_alarm"] == "magnitude" and 249.9 <= f.t_onset <= 250.2, f"{f.status} {f.t_onset} {v.get('states')}")
    check(v["states"] == {"magnitude": "alarm", "pair": "ok", "heading": "ok"}, v["states"])
    check(60 <= v["checks"]["magnitude"]["peak_dev_mG"] <= 75, v["checks"]["magnitude"])
    check(v["verdict"] == "possible_cause" and f.context == "precursor_candidate", v["verdict"])
    w2 = World()
    _scale_mag(w2, 0, 250.0, 252.0, 1.05)                  # +22 mG < 35
    check(one(dc.compass_raw_consistency, w2).status == CLEAR, "22 mG 不算")
    w3 = World()
    _scale_mag(w3, 1, 250.0, 252.0, 1.15)                  # 未被 EKF 使用的羅盤:強度檢查只看 EKF 用的那顆
    check(one(dc.compass_raw_consistency, w3).status == CLEAR, "備援羅盤的強度變化不算")
    w4 = World()
    for ln in range(3):
        w4.tb["XKFS"][ln]["MI"][:] = 1.0                   # EKF 改用 MAG[1]
    _scale_mag(w4, 1, 250.0, 252.0, 1.15)
    g = one(dc.compass_raw_consistency, w4)
    check(g.status == HIT and g.values["used_compass"] == 1 and g.values["checks"]["magnitude"]["compass"] == 1, g.values["states"])


def test_synth_raw_pair_angle():
    w = World()
    _rotate_mag(w, 1, 250.0, 253.0, 75.0)                  # 繞機體 z 轉 75°:兩顆羅盤夾角約 40°
    f = one(dc.compass_raw_consistency, w)
    check(f.status == HIT and f.values["first_alarm"] == "pair" and f.values["checks"]["pair"]["peak_deg"] > 30, f.values["states"])
    check("雙羅盤夾角" in f.text_zh, f.text_zh)
    w.params["COMPASS_USE2"] = 0.0                         # 備援停用 → 夾角不可判,其餘照常
    g = one(dc.compass_raw_consistency, w)
    check(g.status == CLEAR and g.values["checks"]["pair"] == {"state": "na", "reason": "backup_disabled"}, g.values["checks"]["pair"])
    w.params["COMPASS_USE2"] = 1.0
    w.params["COMPASS_ORIENT2"] = 4.0                      # 朝向不同 → 不比
    h = one(dc.compass_raw_consistency, w)
    check(h.status == CLEAR and h.values["checks"]["pair"]["reason"] == "orient_differs", h.values["checks"]["pair"])
    w1 = World(n_mags=1)
    k = one(dc.compass_raw_consistency, w1)
    check(k.status == CLEAR and k.values["checks"]["pair"]["reason"] == "single_compass", k.values["checks"]["pair"])


def test_synth_raw_heading_pulled_away():
    """EKF 航向被拉走 25°(羅盤沒變):航向殘差 > 15°。常數磁偏角(−4.7°)不會觸發。"""
    w = World()
    m = w.win("ATT", 0, 300.0, 303.0)
    w.tb["ATT"][0]["Yaw"][m] = np.mod(w.tb["ATT"][0]["Yaw"][m] + 25.0, 360.0)
    f = one(dc.compass_raw_consistency, w)
    check(f.status == HIT and f.values["first_alarm"] == "heading" and 299.9 <= f.t_onset <= 300.3, f"{f.t_onset} {f.values['states']}")
    check(20 <= f.values["checks"]["heading"]["peak_dev_deg"] <= 30, f.values["checks"]["heading"])
    # 航向跨過 0/360(yaw0 = 0):不能因為折返產生假警報
    w2 = World(yaw0=0.0)
    check(one(dc.compass_raw_consistency, w2).status == CLEAR, "航向在 0/360 附近來回不該觸發")


def test_synth_raw_calibration_window_gated():
    """羅盤校正(MAG.Ofs 變動)前 120 s 到後 5 s 不判定;窗外照常;窗占解鎖區間一半以上 → NA。"""
    w = World()
    for i in (0, 1):
        w.tb["MAG"][i]["OfsX"][w.win("MAG", i, 240.0, 1e9)] = 25.0
    _scale_mag(w, 0, 200.0, 202.0, 1.15)                   # 校正窗 [120, 245] 內
    f = one(dc.compass_raw_consistency, w)
    check(f.status == CLEAR and 0.35 < f.values["gated_frac"] < 0.5, f"{f.status} {f.values.get('gated_frac')}")
    _scale_mag(w, 0, 300.0, 302.0, 1.15)                   # 窗外
    g = one(dc.compass_raw_consistency, w)
    check(g.status == HIT and 299.9 <= g.t_onset <= 300.2, f"{g.status} {g.t_onset}")
    w2 = World()
    for i in (0, 1):
        w2.tb["MAG"][i]["OfsY"][w2.win("MAG", i, 200.0, 1e9)] = 7.0
        w2.tb["MAG"][i]["OfsY"][w2.win("MAG", i, 330.0, 1e9)] = 9.0
    h = one(dc.compass_raw_consistency, w2)
    check(h.status == NA and h.context == "calibration_window", f"{h.status} {h.context}")
    # 開機 5 s 內的偏移變動是參數載入,不是校正
    w3 = World()
    for i in (0, 1):
        w3.tb["MAG"][i]["OfsZ"][w3.win("MAG", i, 0.0, 2.0)] = 0.0
    check(dc._cal_windows(dc._mag_tables(w3.ctx())) == [], "開機 5 s 內的偏移變動不算校正")


def _thr_burst(w, ta, tb, val, ratios, gain=1.0):
    """油門在 [ta, tb) 改成 val,兩顆羅盤的干擾照比例(× gain:電流比油門的線性模型更大)跟著變。"""
    s = w.tb["MOTB"][0]["t"] - w.off
    old = w.tb["MOTB"][0]["ThrOut"].copy()
    w.tb["MOTB"][0]["ThrOut"][(s >= ta) & (s < tb)] = val
    for i, r in enumerate(ratios):
        mt = w.tb["MAG"][i]["t"] - w.off
        d = np.interp(mt, s, w.tb["MOTB"][0]["ThrOut"]) - np.interp(mt, s, old)
        w.tb["MAG"][i]["MagZ"] = w.tb["MAG"][i]["MagZ"] + gain * r * 446.0 / HOVER_THR * d


def test_synth_raw_motor_current_is_not_an_alarm():
    """審查缺陷(major):EKF 用的羅盤有 10-80% 馬達干擾的「健康」飛行 —— 起飛 / 降落 / 爬升油門階躍都不得 HIT
    (修正前 0.10 / 0.15 / 0.20 在 105 s 起飛時 HIT、偏離 41 / 61 / 81 mG,而馬達干擾偵測器同時說「良好」)。"""
    for r in ((0.10, 0.10), (0.15, 0.40), (0.20, 0.10), (0.40, 0.04), (0.80, 0.10)):
        w = World(ratios=r)
        f = one(dc.compass_raw_consistency, w)
        check(f.status == CLEAR, f"{r}:{f.status} {f.text_zh}")
        m = f.values["checks"]["magnitude"]
        check(m["motor_model"] is not None and m["max_sustained_dev_mG"] < 10.0, (r, m))
        check(abs(f.values["motor_model"]["0"]["slope_mG"]) > 100.0, f.values["motor_model"])
    # 爬升:懸停 0.15 → 0.30 持續 8 s(修正前 0.10 的羅盤 HIT,帶 t_warn 300 時 possible_cause、lead 195 s)
    for r in ((0.10, 0.10), (0.40, 0.04)):
        w = World(ratios=r)
        _thr_burst(w, 250.0, 258.0, 0.30, r)
        f = one(dc.compass_raw_consistency, w, t_warn=300.0, end_type="crash_check", t_detect=305.0)
        check(f.status == CLEAR and f.values.get("verdict") is None, f"爬升 {r}:{f.status} {f.text_zh}")
    # 同一個羅盤、馬達干擾偵測器的評等(健康度)與一致性檢查不矛盾:0.80 → 差(HIT),一致性 CLEAR
    check(one(dc.compass_motor_interference, World(ratios=(0.80, 0.10))).status == HIT, "0.80 → 馬達干擾差")


def test_synth_raw_motor_transient_explained_and_arming_step():
    """補償後仍超標、但油門變化足以解釋的 run(電流比線性模型大 60% 的 1 s 油門尖峰)→ 歸因馬達,不是警報;
    把 RC_MAG_MOTOR_FRAC 設成無限大(關掉歸因)同一筆資料就 HIT —— 證明是這條規則在起作用。
    解鎖瞬間 idle 電流讓 |B| 跳 45 mG 而 ThrOut 仍為 0:參考值只用解鎖後的樣本,不得 HIT(有 / 沒有油門資料都一樣)。"""
    w = World(ratios=(0.40, 0.04))
    _thr_burst(w, 250.0, 251.0, 0.35, (0.40, 0.04), gain=1.6)
    f = one(dc.compass_raw_consistency, w, t_warn=300.0, end_type="crash_check", t_detect=305.0)
    ex = f.values["checks"]["magnitude"]["motor_explained"]
    check(f.status == CLEAR and len(ex) == 1 and near(ex[0]["t"], 250.0, 0.15), f"{f.status} {ex}")
    check(ex[0]["motor_mG"] >= dc.RC_MAG_MOTOR_FRAC * ex[0]["peak_dev_mG"], ex)
    check("馬達電流模型相符" in f.text_zh, f.text_zh)
    frac = dc.RC_MAG_MOTOR_FRAC
    try:
        dc.RC_MAG_MOTOR_FRAC = 1e9
        g = one(dc.compass_raw_consistency, w, t_warn=300.0, end_type="crash_check", t_detect=305.0)
    finally:
        dc.RC_MAG_MOTOR_FRAC = frac
    check(g.status == HIT and g.values["first_alarm"] == "magnitude", f"關掉歸因應 HIT:{g.status}")
    for no_thr in (False, True):
        w2 = World(ratios=(0.04, 0.40), idle_thr=0.0)
        w2.tb["MAG"][0]["MagZ"][w2.win("MAG", 0, 100.0, 1e9)] += 45.0
        if no_thr:
            del w2.tb["MOTB"]
        h = one(dc.compass_raw_consistency, w2)
        check(h.status == CLEAR, f"解鎖 idle 電流階躍(no_thr={no_thr}):{h.status} {h.text_zh}")


def test_synth_raw_motor_field_heading_compensated():
    """懸停時固定在機體上的干擾有水平分量:快速轉向 70°(油門不變)時原始羅盤航向偏 30° 以上。
    扣掉向量模型 x·c 之後不得 HIT;拿掉向量補償(只留純量)同一筆資料就航向 HIT —— 證明補償是必要的。"""
    kw = dict(ratios=(0.40, 0.04), interf_dir=(0.6, -0.6, 0.53), turns=[(250.0, 70.0, 1.5)])
    f = one(dc.compass_raw_consistency, World(**kw))
    check(f.status == CLEAR and f.values["checks"]["heading"]["motor_compensated"], f"{f.status} {f.text_zh}")
    check(f.values["motor_model"]["0"]["mag_mode"] == "vector", f.values["motor_model"])
    orig = dc._motor_models

    def scalar_only(*a, **k):
        out = orig(*a, **k)
        for v in out.values():
            v["c"], v["mag_mode"] = None, "scalar"
        return out
    try:
        dc._motor_models = scalar_only
        g = one(dc.compass_raw_consistency, World(**kw))
    finally:
        dc._motor_models = orig
    gh = g.values["checks"]["heading"]
    check(g.status == HIT and gh["state"] == "alarm" and 250.0 <= gh["t_first"] <= 252.0
          and gh["max_sustained_dev_deg"] > 25.0, f"沒有向量補償應在轉向時航向 HIT:{gh}")


def test_synth_raw_real_fault_still_detected_with_motor_model():
    """補償不能把真的故障吃掉:(a) 40% 干擾的羅盤在懸停(油門不變)時 |B| +15% → HIT;(b) 同上但干擾有水平分量
    (向量模式);(c) 4% 干擾的羅盤 |B| 跳變剛好和油門 0.15 → 0.20 同時 → 油門只能解釋幾 mG → 仍 HIT。"""
    w = World(ratios=(0.40, 0.04))
    _scale_mag(w, 0, 250.0, 252.0, 1.15)
    f = one(dc.compass_raw_consistency, w, t_warn=260.0)
    check(f.status == HIT and f.values["first_alarm"] == "magnitude" and 249.95 <= f.t_onset <= 250.15
          and f.values["verdict"] == "possible_cause", f"{f.status} {f.t_onset}")
    check(f.values["checks"]["magnitude"]["motor_explained"] == [], f.values["checks"]["magnitude"]["motor_explained"])
    w2 = World(ratios=(0.40, 0.04), interf_dir=(0.6, -0.6, 0.53), turns=[(200.0, 60.0, 1.5)])
    _scale_mag(w2, 0, 250.0, 252.0, 1.15)
    g = one(dc.compass_raw_consistency, w2)
    check(g.status == HIT and g.values["first_alarm"] == "magnitude" and g.values["motor_model"]["0"]["mag_mode"] == "vector",
          f"{g.status} {g.values.get('states')}")
    w3 = World(ratios=(0.04, 0.40))
    _thr_burst(w3, 250.0, 260.0, 0.20, (0.04, 0.40))
    _scale_mag(w3, 0, 250.0, 252.0, 1.15)
    h = one(dc.compass_raw_consistency, w3)
    check(h.status == HIT and 249.95 <= h.t_onset <= 250.15, f"{h.status} {h.t_onset}")


# ---------- compass_motor_interference

def test_synth_motor_interference_levels():
    f = one(dc.compass_motor_interference, World(ratios=(0.80, 0.10)))
    check(f.status == HIT and f.values["level"] == "差" and abs(f.values["used_ratio"] - 0.80) < 0.03, f"{f.status} {f.values['used_ratio']}")
    check(f.context == "health" and not f.values["hygiene"], f.context)
    g = one(dc.compass_motor_interference, World(ratios=(0.45, 0.10)))
    check(g.status == SOFT and g.values["level"] == "邊際", f"{g.status} {g.values['used_ratio']}")
    h = one(dc.compass_motor_interference, World(ratios=(0.10, 0.10)))
    check(h.status == CLEAR and h.context == "health" and h.values["backup_level"] == "良好", f"{h.status} {h.context}")
    # status 只看 EKF 實際使用的羅盤:EKF 改用 40% 的 MAG[1] → SOFT
    w = World(ratios=(0.04, 0.40))
    for ln in range(3):
        w.tb["XKFS"][ln]["MI"][:] = 1.0
    k = one(dc.compass_motor_interference, w)
    check(k.status == SOFT and k.values["used_compass"] == 1 and k.values["backup_compass"] == 0, f"{k.status} {k.values['used_compass']}")
    check(abs(k.values["used_ratio"] - 0.40) < 0.02, k.values["used_ratio"])


def test_synth_motor_interference_na_cases():
    f = one(dc.compass_motor_interference, World(t0=20.0))
    check(f.status == NA and f.context == "no_ground_reference", f"開機 20 s 就解鎖:沒有 20 s 地面參考:{f.context}")
    w = World()
    w.tb["MOTB"][0]["ThrOut"][:] = 0.0
    g = one(dc.compass_motor_interference, w)
    check(g.status == NA and g.context == "no_motor_flight", g.context)
    w2 = World()
    del w2.tb["MOTB"]
    h = one(dc.compass_motor_interference, w2)
    check(h.status == NA and h.context == "no_throttle", h.context)
    # 有 t_warn:飛行集在 t_warn − 0.5 s 截止(翻機後的資料不納入)
    k = one(dc.compass_motor_interference, World(), t_warn=300.0)
    check(k.values["flight_window"][1] <= 299.5 + 0.11, k.values["flight_window"])


# ---------- ekf_yaw_reset_lane_switch

def _jump_xkf1(w, t_from, deg, lanes=(0, 1, 2)):
    for ln in lanes:
        m = w.win("XKF1", ln, t_from, 1e9)
        w.tb["XKF1"][ln]["Yaw"][m] = np.mod(w.tb["XKF1"][ln]["Yaw"][m] + deg, 360.0)


def test_synth_yaw_reset_large_jump_hit():
    w = World()
    w.ev.append((250.0, 62))
    _jump_xkf1(w, 250.0, 25.0)
    f = one(dc.ekf_yaw_reset_lane_switch, w, t_warn=262.0)
    check(f.status == HIT and abs(f.t_onset - 250.0) < 1e-6 and f.values["first_kind"] == "yaw_reset", f"{f.status} {f.t_onset}")
    j = [r for r in f.values["resets"] if abs(r["t"] - 250.0) < 1e-6][0]
    check(j["kind"] == "inflight_large_jump" and 24.0 <= j["jump_deg"] <= 26.0, j)
    check(f.values["verdict"] == "possible_cause" and "跳變 +2" in f.text_zh, f"{f.values['verdict']} {f.text_zh}")
    check(f.values["events_in_pre30s"] == ["yaw_reset"], f.values["events_in_pre30s"])
    # 小跳變(2°):良性
    w2 = World()
    w2.ev.append((250.0, 62))
    _jump_xkf1(w2, 250.0, 2.0)
    g = one(dc.ekf_yaw_reset_lane_switch, w2)
    check(g.status == CLEAR and g.values["last_reset_kind"] == "inflight_small_jump", f"{g.status} {g.values['last_reset_kind']}")
    # 解鎖後 15 s 內的大跳變:空中對準期間 → 只給 SOFT;開機 10 s 內 / 上鎖後:不算
    w3 = World()
    w3.ev.append((105.0, 62))
    _jump_xkf1(w3, 105.0, 25.0)
    h = one(dc.ekf_yaw_reset_lane_switch, w3)
    j105 = [r for r in h.values["resets"] if abs(r["t"] - 105.0) < 1e-6][0]
    check(h.status == SOFT and j105["kind"] == "inflight_large_jump_after_arm", h.values["resets"])
    w4 = World()
    w4.ev += [(5.0, 62), (410.0, 62)]
    k = one(dc.ekf_yaw_reset_lane_switch, w4)
    check(k.status == CLEAR and {r["kind"] for r in k.values["resets"]} == {"initial_alignment", "inflight_alignment", "outside_interval"},
          k.values["resets"])
    # 沒有 XKF1 無法量跳變:SOFT
    w5 = World()
    w5.ev.append((250.0, 62))
    del w5.tb["XKF1"]
    m = one(dc.ekf_yaw_reset_lane_switch, w5)
    check(m.status == SOFT and m.values["resets"][-1]["kind"] == "inflight_unknown_jump", m.values["resets"])


def test_synth_lane_switch_and_source_switch():
    w = World()
    w.msgs.append((260.0, "EKF3 lane switch 1"))
    f = one(dc.ekf_yaw_reset_lane_switch, w)
    check(f.status == SOFT and f.values["lane_switches"] == [{"t": 260.0, "lane": 1}], f.values["lane_switches"])
    w2 = World()
    w2.err.append((261.0, dc.ERR_SUBSYS_EKF_PRIMARY, 2))
    g = one(dc.ekf_yaw_reset_lane_switch, w2)
    check(g.status == SOFT and g.values["lane_switches"][0]["lane"] == 2, g.values["lane_switches"])
    w2.msgs.append((260.8, "EKF3 lane switch 2"))                       # 同一次切換的 MSG + ERR 只算一次
    check(len(one(dc.ekf_yaw_reset_lane_switch, w2).values["lane_switches"]) == 1, "MSG 與 ERR 要去重")
    # 來源組切換:航向來源相同 → 只列出;不同(羅盤 → GPS 航向)→ HIT
    w3 = World()
    w3.ev.append((260.0, 86))
    h = one(dc.ekf_yaw_reset_lane_switch, w3)
    check(h.status == CLEAR and h.values["source_switches"] == [{"t": 260.0, "from_set": 1, "to_set": 2, "yaw_changed": False}],
          h.values["source_switches"])
    check("來源組切換" in h.text_zh, h.text_zh)
    w3.params["EK3_SRC2_YAW"] = 2.0
    k = one(dc.ekf_yaw_reset_lane_switch, w3, t_warn=262.0)
    check(k.status == HIT and k.values["first_kind"] == "yaw_source_changed" and k.values["verdict"] == "possible_cause", k.values)
    # 解鎖前的切換只更新「目前來源組」,不列入
    w4 = World()
    w4.ev.append((50.0, 87))
    w4.params["EK3_SRC3_YAW"] = 2.0
    m = one(dc.ekf_yaw_reset_lane_switch, w4)
    check(m.status == CLEAR and m.values["source_switches"] == [], m.values["source_switches"])


def test_synth_lane_yaw_spread():
    w = World()
    _jump_xkf1(w, 280.0, 15.0, lanes=(1,))
    _jump_xkf1(w, 283.0, -15.0, lanes=(1,))
    f = one(dc.ekf_yaw_reset_lane_switch, w)
    check(f.status == SOFT and f.values["first_kind"] == "lane_yaw_spread" and 279.9 <= f.t_onset <= 280.1, f"{f.status} {f.t_onset}")
    check(14.0 <= f.values["lane_yaw_spread_max_deg"] <= 16.0, f.values["lane_yaw_spread_max_deg"])
    w2 = World()                                         # 各 lane 時間戳不同、航向快速變化也不能有假差異
    for ln in range(3):
        s = w2.tb["XKF1"][ln]["t"]
        w2.tb["XKF1"][ln]["Yaw"] = np.mod(90.0 + 40.0 * s, 360.0)      # 40°/s 持續旋轉
    g = one(dc.ekf_yaw_reset_lane_switch, w2)
    check(g.status == CLEAR and g.values["lane_yaw_spread_max_deg"] < 2.0, f"時間對齊後不該有差:{g.values['lane_yaw_spread_max_deg']}")


# ---------- compass_sensor_health

def test_synth_sensor_health_alarms():
    w = World()
    w.tb["MAG"][0]["Health"][w.win("MAG", 0, 200.0, 201.5)] = 0.0
    f = one(dc.compass_sensor_health, w, t_warn=210.0)
    check(f.status == HIT and f.values["alarms"][0]["kind"] == "health_zero" and 199.9 <= f.t_onset <= 200.1, f.values["alarms"])
    check(f.values["verdict"] == "possible_cause" and "MAG[0]" in f.text_zh, f.text_zh)
    w2 = World()
    w2.tb["MAG"][1]["Health"][w2.win("MAG", 1, 200.0, 201.5)] = 0.0
    check(one(dc.compass_sensor_health, w2).status == SOFT, "備援羅盤不健康 → SOFT")
    w3 = World()
    w3.tb["MAG"][0]["Health"][w3.win("MAG", 0, 200.0, 200.3)] = 0.0
    check(one(dc.compass_sensor_health, w3).status == CLEAR, "0.3 s < 0.5 s")
    w4 = World()
    w4.err += [(50.0, 3, 4), (250.0, 3, 0)]                # 地面的 ECode 4、飛行中的 ECode 0(清除)
    g = one(dc.compass_sensor_health, w4)
    check(g.status == CLEAR and g.values["ground_events"] == [{"t": 50.0, "ecode": 4}], g.values["ground_events"])
    w4.err.append((250.0, 3, 1))
    h = one(dc.compass_sensor_health, w4)
    check(h.status == HIT and h.values["alarms"][0]["kind"] == "err_compass" and h.values["alarms"][0]["ecode"] == 1, h.values["alarms"])


def test_synth_sensor_health_data_gaps():
    w = World()
    for i in (0, 1):
        w.drop("MAG", i, 200.0, 203.5)
    f = one(dc.compass_sensor_health, w)
    check(f.status == HIT and f.values["alarms"][0]["kind"] == "data_gap" and 199.9 <= f.t_onset <= 200.0, f.values["alarms"])
    check(3.5 <= f.values["alarms"][0]["dur_s"] <= 3.7, f.values["alarms"][0])
    w2 = World()
    for i in (0, 1):
        w2.drop("MAG", i, 200.0, 201.5)
    check(one(dc.compass_sensor_health, w2).status == CLEAR, "1.6 s 空洞 < 2 s")
    # 羅盤在飛行中停止記錄(之後再也沒有樣本):掉線最常見的樣子
    w3 = World()
    for i in (0, 1):
        w3.drop("MAG", i, 300.0, 1e9)
    g = one(dc.compass_sensor_health, w3)
    check(g.status == HIT and g.values["alarms"][0]["kind"] == "data_gap" and 299.9 <= g.t_onset <= 300.0, g.values["alarms"])
    check(99.0 <= g.values["alarms"][0]["dur_s"] <= 100.1 and g.values["max_gap_s"]["0"] > 99.0, g.values)
    # 上鎖之後才停:不算
    w4 = World()
    for i in (0, 1):
        w4.drop("MAG", i, 401.0, 1e9)
    check(one(dc.compass_sensor_health, w4).status == CLEAR, "上鎖後停止記錄不算")


def test_synth_sensor_health_ekf_bits():
    w = World()
    w.tb["XKF4"][0]["TS"][w.win("XKF4", 0, 250.0, 251.0)] = 48.0 + 8.0
    f = one(dc.compass_sensor_health, w)
    check(f.status == HIT and f.values["alarms"][0]["kind"] == "ekf_mag_timeout", f.values["alarms"])
    w2 = World()
    w2.tb["XKF4"][0]["TS"][w2.win("XKF4", 0, 250.0, 251.0)] = 48.0 + 3.0      # 位置 / 速度逾時位元不是羅盤
    check(one(dc.compass_sensor_health, w2).status == CLEAR, "TS bit0/1 不是磁力計逾時")
    w3 = World()
    w3.tb["XKF4"][1]["FS"][w3.win("XKF4", 1, 250.0, 250.5)] = 4.0
    g = one(dc.compass_sensor_health, w3)
    check(g.status == HIT and g.values["alarms"][0]["kind"] == "ekf_fault" and g.values["alarms"][0]["fs"] == 4, g.values["alarms"])
    # 開機 1 s 內的 FS=128(未初始化)不算,即使解鎖區間從 0.5 s 開始
    w4 = World(t0=0.5)
    check(one(dc.compass_sensor_health, w4).status == CLEAR, "開機 FS=128 是正常的")
    # 沒有 MAG 但有 XKF4:只能從 EKF 判斷(context ekf_only)
    w5 = World()
    del w5.tb["MAG"]
    h = one(dc.compass_sensor_health, w5)
    check(h.status == CLEAR and h.context == "ekf_only" and "0 顆" not in h.text_zh, f"{h.status} {h.context} {h.text_zh}")


# ---------- ekf_attitude_consistency

def test_synth_attitude_consistency():
    w = World()
    w.tb["AHR2"][0]["Roll"][w.win("AHR2", 0, 200.0, 201.0)] += 20.0
    f = one(dc.ekf_attitude_consistency, w)
    check(f.status == HIT and f.values["axis"] == "roll" and 199.9 <= f.t_onset <= 200.2, f"{f.status} {f.values.get('axis')} {f.t_onset}")
    check("姿態估計器可能出錯" in f.text_zh, f.text_zh)
    w2 = World()
    m = w2.win("AHR2", 0, 200.0, 201.0)
    w2.tb["AHR2"][0]["Yaw"][m] = np.mod(w2.tb["AHR2"][0]["Yaw"][m] + 25.0, 360.0)
    g = one(dc.ekf_attitude_consistency, w2)
    check(g.status == HIT and g.values["axis"] == "yaw", g.values.get("axis"))
    w3 = World()
    m = w3.win("DCM", 0, 200.0, 202.0)
    w3.tb["DCM"][0]["Yaw"][m] = np.mod(w3.tb["DCM"][0]["Yaw"][m] + 20.0, 360.0)
    h = one(dc.ekf_attitude_consistency, w3)
    check(h.status == SOFT and h.values["axis"] == "dcm_yaw" and "低信心" in h.text_zh, f"{h.status} {h.values.get('axis')}")
    w4 = World()
    w4.tb["AHR2"][0]["Pitch"][w4.win("AHR2", 0, 200.0, 200.2)] += 30.0          # 0.2 s 的單點 < 0.3 s
    check(one(dc.ekf_attitude_consistency, w4).status == CLEAR, "短於 0.3 s 不算")


def test_synth_attitude_consistency_crash_event_window():
    """失事(end_type crash_check):除了整段統計,另外給 [t_c − 15, t_c] 的事件窗數字;翻滾本身(ATT 與 AHR2 一起轉)不算不一致。"""
    w = World()
    for name in ("ATT", "AHR2"):
        m = w.win(name, 0, 395.0, 400.1)
        w.tb[name][0]["Roll"][m] = np.linspace(0.0, 179.0, int(m.sum())) + (0.5 if name == "AHR2" else 0.0)
    f = one(dc.ekf_attitude_consistency, w, end_type="crash_check", t_detect=400.0)
    check(f.status == CLEAR and f.values["event_window"] == [385.0, 400.0], f"{f.status} {f.values.get('event_window')}")
    check(f.values["event_max_droll_deg"] < 2.0 and "失事前 15 s" in f.text_zh, f.values["event_max_droll_deg"])


def test_synth_attitude_consistency_boot_and_na():
    """開機初始航向對準前 ATT.Yaw 停在 0(EKF 還沒對準),AHR2 已有航向:不得報 EKF 與獨立姿態不一致。"""
    w = World(t0=0.5)
    w.tb["ATT"][0]["Yaw"][w.win("ATT", 0, 0.0, 4.5)] = 0.0
    f = one(dc.ekf_attitude_consistency, w)
    check(f.status == CLEAR and f.values["window"][0] >= dc.AC_BOOT_S, f"{f.status} {f.values.get('window')}")
    w2 = World()
    del w2.tb["AHR2"]
    g = one(dc.ekf_attitude_consistency, w2)
    check(g.status == CLEAR and g.context == "dcm_yaw_only" and not g.values["roll_pitch_checked"], f"{g.status} {g.context}")
    del w2.tb["DCM"]
    h = one(dc.ekf_attitude_consistency, w2)
    check(h.status == NA and h.context == "no_data", h.context)
    w3 = World()                                          # AHR2 只在解鎖前有紀錄、沒有 DCM → 不可比對(不是 CLEAR)
    w3.drop("AHR2", 0, 90.0, 1e9)
    del w3.tb["DCM"]
    k = one(dc.ekf_attitude_consistency, w3)
    check(k.status == NA and k.context == "no_data", f"{k.status} {k.context}")


# ---------- 共通:NA、因果、平移、假說整合

def test_synth_not_armed_and_no_data():
    w = World()
    fs = dc.run(w.ctx(), Anchors())
    check(sorted(f.id for f in fs) == sorted(IDS) and all(f.status == NA and f.context == "not_armed" for f in fs), statuses(fs))
    armed = [ArmedInterval(0, 10.0, 300.0, 0)]
    a = Anchors(intervals=armed, interval=armed[0])
    r = run_checked(FakeCtx(armed=armed), a)
    check(all(f.status == NA for f in r.values()), statuses(r.values()))
    # 解鎖區間太短(< 10 s 有效資料)→ NA,不是 CLEAR
    w2 = World(t0=100.0, t1=106.0)
    r2 = run_checked(w2.ctx(), w2.anchors())
    for k in ("ekf_mag_reject", "compass_raw_consistency", "ekf_yaw_reset_lane_switch", "compass_sensor_health",
              "ekf_attitude_consistency", "compass_motor_interference"):
        check(r2[k].status == NA, f"{k}:6 s 的解鎖不足以判定:{r2[k].status} {r2[k].context}")


def test_synth_time_shift_invariance():
    """整份資料(含解鎖區間與錨點)平移 5000 s:結論、t_onset(扣掉平移)與數值不變 → 沒有寫死絕對時間。"""
    res = []
    for off in (0.0, 5000.0):
        w = World(off=off)
        _sm(w, (0, 1), 250.0, 252.0)
        _scale_mag(w, 0, 300.0, 302.0, 1.15)
        r = run_checked(w.ctx(), w.anchors(t_warn=320.0, end_type="crash_check", t_detect=330.0))
        res.append({k: (f.status, None if f.t_onset is None else round(f.t_onset - off, 6), f.values.get("verdict")) for k, f in r.items()})
    check(res[0] == res[1], f"{res[0]} != {res[1]}")
    check(res[0]["ekf_mag_reject"][0] == HIT and res[0]["compass_raw_consistency"][0] == HIT, res[0])


def test_synth_hypothesis_scores_precursor_only():
    """與假說評分整合:同樣的羅盤 HIT,早於 t_warn ≥1 s → H_COMPASS_YAW 有分且必要證據成立;晚於姿態失控 → 0 分。"""
    from core.crash import engine
    from core.crash import hypotheses as hyp

    def score(t_warn, t_loss):
        w = World()
        _sm(w, (0, 1), 250.0, 252.0)
        a = w.anchors(t_warn=t_warn, t_loss=t_loss, end_type="crash_check", t_detect=t_loss + 4.0)
        fs = dc.run(w.ctx(), a)
        if len(inspect.signature(engine._tag_findings).parameters) == 3:     # engine 介面在演進:(ctx, a, fs) 或 (a, fs)
            engine._tag_findings(w.ctx(), a, fs)
        else:
            engine._tag_findings(a, fs)
        h = [x for x in hyp.score_all(fs, a) if x.id == "H_COMPASS_YAW"][0]
        return h, {f.id: f.tag for f in fs}

    h, tags = score(260.0, 262.0)
    check(tags["ekf_mag_reject"] == "precursor" and h.score > 0 and h.gate_ok, f"{tags} {h.score} {h.gate_ok}")
    h2, tags2 = score(245.0, 248.0)
    check(tags2["ekf_mag_reject"] == "consequence" and h2.score == 0 and not h2.gate_ok, f"{tags2} {h2.score}")


# ---------- 審查缺陷 2-6 的回歸測試

def test_synth_mag_reject_missing_signals_not_clear():
    """審查缺陷:覆蓋只能算「該模式的訊號真的有值」的樣本。XKF4 沒有 SM、也沒有 XKF3 → NA
    (修正前 CLEAR、內文空白『EKF 沒有拒絕磁力計:。』);只有 SM → CLEAR 但只列 SM;SM 全 NaN → NA。"""
    w = World()
    for ln in range(3):
        del w.tb["XKF4"][ln]["SM"]
    del w.tb["XKF3"]
    f = one(dc.ekf_mag_reject, w)
    check(f.status == NA and f.values["coverage_3d_s"] == 0.0 and f.values["coverage_yaw_only_s"] == 0.0,
          f"{f.status} {f.text_zh}")
    w2 = World()
    del w2.tb["XKF3"]
    g = one(dc.ekf_mag_reject, w2)
    check(g.status == CLEAR and "SM 最大" in g.text_zh and "創新量" not in g.text_zh and "IYAW" not in g.text_zh, g.text_zh)
    check(g.values["coverage_3d_s"] > 250 and g.values["coverage_yaw_only_s"] == 0.0, g.values)
    w3 = World()
    for ln in range(3):
        w3.tb["XKF4"][ln]["SM"][:] = np.nan
    del w3.tb["XKF3"]
    check(one(dc.ekf_mag_reject, w3).status == NA, "SM 全 NaN、沒有 XKF3 → NA")
    w4 = World()
    del w4.tb["XKF3"]
    _sm(w4, (0, 1), 250.0, 252.0)
    h = one(dc.ekf_mag_reject, w4)
    check(h.status == HIT and "SM 峰值 2.0" in h.text_zh and "|IM" not in h.text_zh and "nan" not in h.text_zh, h.text_zh)


def test_synth_sensor_health_gaps_clipped_to_interval():
    """審查缺陷:取樣空洞只算解鎖區間內的部分(修正前跨上鎖的 5.6 s 暫停 HIT、跨解鎖的 HIT 起點在解鎖前 94.95 s);
    整個 log 一起暫停(SD 卡卡住)不是羅盤掉線;LOG_DISARMED=0 的 log(解鎖才開始記錄)MAG 晚幾秒出現不是中斷。"""
    w = World()
    for i in (0, 1):
        w.drop("MAG", i, 399.5, 405.0)                     # 區間內只有 0.55 s
    f = one(dc.compass_sensor_health, w)
    check(f.status == CLEAR and f.values["max_gap_s"]["0"] < 1.0, f"{f.status} {f.values['alarms']}")
    w2 = World()
    for i in (0, 1):
        w2.drop("MAG", i, 95.0, 100.6)                     # 區間內只有 0.65 s
    g = one(dc.compass_sensor_health, w2)
    check(g.status == CLEAR, f"{g.status} {g.values['alarms']}")
    w3 = World()
    for i in (0, 1):
        w3.drop("MAG", i, 97.0, 103.0)                     # 區間內 3.05 s
    h = one(dc.compass_sensor_health, w3)
    check(h.status == HIT and h.t_onset >= 100.0 - 1e-9 and 2.9 <= h.values["alarms"][0]["dur_s"] <= 3.2,
          f"{h.status} {h.t_onset} {h.values['alarms']}")
    w4 = World()
    for name in list(w4.tb):
        for inst in list(w4.tb[name]):
            w4.drop(name, inst, 200.0, 203.5)              # 整個 log 暫停 3.5 s
    k = one(dc.compass_sensor_health, w4)
    check(k.status == CLEAR and k.values["log_pauses"] and near(k.values["log_pauses"][0]["t"], 199.95, 0.06),
          f"{k.status} {k.values['alarms']} {k.values['log_pauses']}")
    w5 = World()
    for name in list(w5.tb):
        for inst in list(w5.tb[name]):
            w5.drop(name, inst, 0.0, 100.0)                # log 從解鎖才開始
    for i in (0, 1):
        w5.drop("MAG", i, 100.0, 104.0)                    # MAG 晚 4 s 才出現
    m = one(dc.compass_sensor_health, w5)
    check(m.status == CLEAR and not m.values["log_before_arm"], f"{m.status} {m.values['alarms']}")
    w6 = World()
    for i in (0, 1):
        w6.drop("MAG", i, 0.0, 104.0)                      # log 解鎖前就在記錄,但 MAG 解鎖後 4 s 才出現 → 中斷
    n = one(dc.compass_sensor_health, w6)
    check(n.status == HIT and n.values["alarms"][0]["kind"] == "data_gap" and near(n.t_onset, 100.0, 1e-6), n.values["alarms"])


def test_cause_text_lead_below_one_second():
    """審查缺陷:早於 t_warn 但不到 1 s 的警報不能寫成「翻機後的連帶現象」(它在姿態誤差之前),只能說無法視為原因。"""
    a = Anchors(t_warn=100.0)
    c = dc._causality(a, 99.5, None)
    txt = dc._cause_text(c, a)
    check(c["verdict"] == "effect_not_cause" and "無法視為原因" in txt and "翻機後" not in txt and "只早 0.5 s" in txt, txt)
    c2 = dc._causality(a, 102.0, None)
    check("晚於姿態誤差" in dc._cause_text(c2, a) and "翻機後的連帶現象" in dc._cause_text(c2, a), dc._cause_text(c2, a))
    b = Anchors()
    t3 = dc._cause_text(dc._causality(b, 79.9, 80.0), b)
    check("無法視為原因" in t3 and "翻機的連帶現象" in t3, t3)
    w = World()
    _sm(w, (0, 1), 250.0, 252.0)
    g = one(dc.ekf_mag_reject, w, t_warn=250.5, end_type="crash_check")
    check(g.values["verdict"] == "effect_not_cause" and "無法視為原因" in g.text_zh and "翻機後" not in g.text_zh, g.text_zh)


def test_synth_yaw_reset_unmeasured_no_ev_and_other_lane():
    """審查缺陷:(a) 缺 XKF1 時起飛後 9 s 的空中對準不得每次都變 SOFT(修正前 'inflight_unknown_jump');
    (b) 沒有 EV 表看不到航向重置 → 不能說「整段沒有航向重置」CLEAR,回 NA(其他證據照常);
    (c) 跳變取所有 XKF1 lane 的最大者(只有 lane 2 跳 25° 也要 HIT)。"""
    w = World()
    del w.tb["XKF1"]
    f = one(dc.ekf_yaw_reset_lane_switch, w)
    check(f.status == CLEAR and f.values["last_reset_kind"] == "inflight_alignment_unmeasured", f"{f.status} {f.values['resets']}")
    w2 = World()
    w2.ev = []
    g = one(dc.ekf_yaw_reset_lane_switch, w2)
    check(g.status == NA and g.context == "no_ev" and not g.values["has_ev"], f"{g.status} {g.context} {g.text_zh}")
    w2.msgs.append((260.0, "EKF3 lane switch 1"))
    check(one(dc.ekf_yaw_reset_lane_switch, w2).status == SOFT, "沒有 EV 但有 lane 切換:證據照常回報")
    w3 = World()
    w3.ev.append((250.0, 62))
    _jump_xkf1(w3, 250.0, 25.0, lanes=(2,))
    h = one(dc.ekf_yaw_reset_lane_switch, w3, t_warn=262.0)
    r = [x for x in h.values["resets"] if abs(x["t"] - 250.0) < 1e-6][0]
    check(h.status == HIT and r["jump_lane"] == "2" and 24.0 <= r["jump_deg"] <= 26.0 and abs(r["jump_by_lane"]["0"]) < 1.0, r)


def test_text_conditions_and_unverified_claims():
    """審查缺陷:(a) 正常降落的健康觀察不得有失事字眼(驗收 B),失事時才說「不是失事原因」;(b) MAG 沒有 Health 欄位
    不得說「全程回報健康」,EKF 沒在融合磁力計不得說「羅盤資料持續送進 EKF」;(c) AHR2 / DCM 全 NaN 不得 CLEAR。"""
    w = World()
    f = one(dc.compass_motor_interference, w)
    check(f.status == CLEAR and "失事" not in f.text_zh, f.text_zh)
    g = one(dc.compass_motor_interference, w, end_type="crash_check", t_detect=380.0)
    check("不是失事原因" in g.text_zh, g.text_zh)
    w2 = World()
    for i in (0, 1):
        del w2.tb["MAG"][i]["Health"]
    h = one(dc.compass_sensor_health, w2)
    check(h.status == CLEAR and "全程回報健康" not in h.text_zh and "未檢查" in h.text_zh and not h.values["health_checked"], h.text_zh)
    w3 = World()
    del w3.tb["MAG"]
    for ln in range(3):
        w3.tb["XKFS"][ln]["MAG_FUSION"][:] = 0.0
    k = one(dc.compass_sensor_health, w3)
    check(k.status == CLEAR and k.context == "ekf_only" and "持續送進 EKF" not in k.text_zh and not k.values["ekf_fusing_mag"], k.text_zh)
    w4 = World()
    del w4.tb["MAG"]
    m = one(dc.compass_sensor_health, w4)
    check("持續送進 EKF" in m.text_zh and m.values["ekf_fusing_mag"], m.text_zh)
    w5 = World()
    for c in ("Roll", "Pitch", "Yaw"):
        w5.tb["AHR2"][0][c][:] = np.nan
    n = one(dc.ekf_attitude_consistency, w5)
    check(n.status == CLEAR and n.context == "dcm_yaw_only" and not n.values["roll_pitch_checked"], f"{n.status} {n.context}")
    w5.tb["DCM"][0]["Yaw"][:] = np.nan
    o = one(dc.ekf_attitude_consistency, w5)
    check(o.status == NA and o.context == "no_data" and "nan" not in o.text_zh, f"{o.status} {o.text_zh}")


def test_synth_all_values_nan_never_clear():
    """所有欄位都是 NaN(時間戳正常):6 個偵測器都只能 NA,不得 CLEAR,也不得執行失敗(修正前 2 個 error、1 個 CLEAR 'nan°')。"""
    w = World()
    for d in w.tb.values():
        for tb in d.values():
            for k in tb:
                if k != "t":
                    tb[k] = np.full(tb[k].size, np.nan)
    r = run_checked(w.ctx(), w.anchors())
    check(all(f.status == NA for f in r.values()), statuses(r.values()))


# ===================================================================== 4. 純函式與介面規則

def test_vote_and_bridge_helpers():
    check(dc._vote([[(0.0, 2.0)], [(1.0, 3.0)], [(5.0, 6.0)]], 2, 0.2) == [(1.0, 2.0)], "兩組重疊 1.0-2.0")
    check(dc._vote([[(0.0, 1.0)], [(0.9, 2.0)]], 2, 0.2) == [], "重疊 0.1 s < 0.2 s")
    check(dc._vote([[(0.0, 1.0)], [(0.5, 2.0)], [(0.7, 1.5)]], 2, 0.2) == [(0.5, 1.5)], dc._vote([[(0.0, 1.0)], [(0.5, 2.0)], [(0.7, 1.5)]], 2, 0.2))
    check(dc._vote([[(0.0, 1.0)]], 1, 0.2) == [(0.0, 1.0)], "單 lane")
    t = np.arange(0.0, 3.0, 0.04)
    m = ((t >= 1.0) & (t < 1.12)) | ((t >= 1.2) & (t < 2.0))
    rb = dc._runs_bridged(m, t, 0.5, 0.15, 0.25)
    check(len(rb) == 1 and abs(rb[0][0] - 1.0) < 1e-9, rb)
    check(dc._runs_bridged(m, t, 0.5, 0.0, 0.25)[0][0] > 1.19, "不併接時第一段太短被丟掉")
    # 資料空洞(> max_dt)斷開 run
    t2 = np.concatenate([np.arange(0.0, 1.0, 0.04), np.arange(2.0, 3.0, 0.04)])
    check(len(dc._runs_bridged(np.ones(t2.size, bool), t2, 0.5, 0.15, 0.25)) == 2, "空洞要斷開")


def test_rolling_ref_time_based():
    t = np.concatenate([np.arange(0.0, 40.0, 0.1), np.arange(60.0, 100.0, 0.1)])      # 40-60 s 空洞
    x = np.where(t < 50.0, 100.0, 200.0)
    idx = np.arange(t.size)
    ref = dc._rolling_ref(t, x, idx, (30.0, 0.5), 20)
    i = int(np.searchsorted(t, 61.0))                    # 61 s:窗 (31, 60.5] 內只有 31-40 s 的舊值
    check(ref[i] == 100.0, ref[i])
    j = int(np.searchsorted(t, 95.0))
    check(ref[j] == 200.0, ref[j])
    check(np.isnan(ref[0]) and np.isnan(ref[int(np.searchsorted(t, 2.0))]), "樣本不足 20 → NaN")
    # floor_t:參考值不可用校正窗結束前的樣本
    fl = np.where(t >= 70.0, 70.0, -np.inf)
    ref2 = dc._rolling_ref(t, x, idx, (30.0, 0.5), 20, fl)
    k = int(np.searchsorted(t, 71.0))
    check(np.isnan(ref2[k]), "校正窗後 1 s 內樣本不足 → NaN")
    # 圓中位數:跨 ±180 的角度
    ang = np.where(np.arange(t.size) % 2 == 0, 179.0, -179.0)
    rc = dc._rolling_ref(t, ang, idx, (20.0, 1.0), 20, circular=True)
    check(abs(abs(rc[j]) - 180.0) < 1.01, f"±179 的圓中位數應約 ±180:{rc[j]}")


def test_tilt_compensated_heading():
    s = np.arange(0.0, 60.0, 0.5)
    roll, pitch, yaw = truth_att(s)
    roll, pitch = roll * 6.0, pitch * 8.0                 # 加大到 ±18° / ±16°
    b = body_field(roll, pitch, yaw)
    h = dc._tilt_comp_heading_deg(b, roll, pitch)
    res = dc.wrap180(yaw - h)
    check(np.max(np.abs(res - DEC_DEG)) < 1e-6, f"傾角補償後的殘差應等於磁偏角:{np.max(np.abs(res - DEC_DEG))}")


def test_yaw_jump_estimator():
    """航向重置跳變:扣掉正常轉動(40°/s 打舵不是跳變)、重置後才開始的轉向不污染、跨 0/360 正確。"""
    t = np.arange(0.0, 20.0, 0.04) + 0.013
    tev = 10.0

    def tb(y):
        return {"t": t, "Yaw": np.mod(y, 360.0)}
    check(abs(dc._yaw_jump(tb(90.0 + 40.0 * t), tev)) < 0.05, "等速旋轉沒有跳變")
    y = 90.0 + 40.0 * t + np.where(t >= tev, 25.0, 0.0)
    check(abs(dc._yaw_jump(tb(y), tev) - 25.0) < 0.05, dc._yaw_jump(tb(y), tev))
    y = 87.6 + np.where(t >= tev, -0.14, 0.0) + np.where(t >= tev + 0.12, 7.0 * (t - tev - 0.12), 0.0)
    check(abs(dc._yaw_jump(tb(y), tev) + 0.14) < 0.02, f"重置後 0.12 s 才轉向:{dc._yaw_jump(tb(y), tev)}")
    y = 355.0 + 5.0 * t + np.where(t >= tev, -20.0, 0.0)            # 跨 0/360
    check(abs(dc._yaw_jump(tb(y), tev) + 20.0) < 0.05, dc._yaw_jump(tb(y), tev))
    check(dc._yaw_jump({"t": t[:3], "Yaw": np.zeros(3)}, tev) is None, "資料不足 → None")


def test_zoh_and_causality():
    t = np.array([0.0, 1.0, 2.0])
    out = dc._zoh(t, np.array([10.0, 20.0, 30.0]), np.array([-0.5, 0.0, 0.5, 1.99, 5.0]))
    check(np.isnan(out[0]) and list(out[1:]) == [10.0, 10.0, 20.0, 30.0], out)
    a = Anchors(t_warn=100.0)
    check(dc._causality(a, 98.9, None)["verdict"] == "possible_cause", "早 1.1 s")
    check(dc._causality(a, 99.5, None)["verdict"] == "effect_not_cause", "早 0.5 s < 1 s")
    check(dc._causality(a, 101.0, 102.0)["verdict"] == "effect_not_cause", "晚於 t_warn")
    b = Anchors()
    check(dc._causality(b, 50.0, None)["verdict"] == "unknown", "沒有任何錨點")
    check(dc._causality(b, 50.0, 80.0)["verdict"] == "unknown", "早於翻覆 30 s:無法判定")
    c = dc._causality(b, 79.5, 80.0)
    check(c["verdict"] == "effect_not_cause" and not c["after_tilt45"], "與翻覆同時(早 0.5 s)")
    check(dc._causality(b, 81.0, 80.0)["after_tilt45"], "翻覆後")


def test_every_scenario_returns_all_ids():
    scen = []
    ctx = real("00000124")
    scen += [(ctx, ck.truth_anchors_124(ctx)), (ctx, ck.flight1_anchors_124(ctx)), (ctx, basic_anchors(ctx))]
    for name in ck.BASELINES:
        c = real(name)
        scen.append((c, ck.not_armed_anchors(c)))
    w = World()
    scen += [(w.ctx(), w.anchors()), (FakeCtx(armed=w.armed), w.anchors()), (FakeCtx(), Anchors())]
    for c, a in scen:
        fs = dc.run(c, a)
        check(sorted(f.id for f in fs) == sorted(IDS), [f.id for f in fs])
        check(not any(f.context == "error" for f in fs), [(f.id, f.values) for f in fs if f.context == "error"])
        for f in fs:
            json.dumps(f.values, allow_nan=False)


def test_guard_converts_errors_to_na():
    class Boom(FakeCtx):
        def table(self, *a, **k):
            raise RuntimeError("boom")
    w = World()
    fs = dc.run(Boom(w), w.anchors())
    check(len(fs) == 6 and all(f.status == NA and f.context == "error" for f in fs), statuses(fs))


def test_no_hardcoded_124_numbers():
    """程式碼(不含註解與字串)裡不可有 124 的時間 / t_base:數值常數只能是門檻與窗長。"""
    src = open(dc.__file__, encoding="utf-8").read()
    bad = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NUMBER:
            try:
                v = float(tok.string)
            except ValueError:
                continue
            if (300.0 <= v <= 1300.0 and v != int(v)) or abs(v - 8.1593) < 1e-3:
                bad.append(tok.string)
        if tok.type == tokenize.NAME and tok.string in ("CRASH_LOG",):
            bad.append(tok.string)
    check(not bad, f"det_compass.py 寫死了 124 的數字:{bad}")
    check("00000124" not in "".join(t.string for t in tokenize.generate_tokens(io.StringIO(src).readline)
                                    if t.type not in (tokenize.COMMENT, tokenize.STRING)), "不可引用 log 名稱")


def _timed(fn, ctx, a):
    t0 = time.perf_counter()
    fn(ctx, a)
    return time.perf_counter() - t0


def test_performance_on_124():
    ctx = real("00000124")
    a = ck.truth_anchors_124(ctx)
    dc.run(ctx, a)                            # 暖機:ctx 的 table 轉換 / 快取
    worst = 0.0
    for fn in dc.DETECTORS:
        best = min(_timed(fn, ctx, a) for _ in range(2))
        worst = max(worst, best)
        print(f"  {fn.__name__}: {best:.3f} s")
    check(worst < 1.5, f"偵測器應 < 1.5 s:{worst:.2f}")


def test_deterministic():
    ctx = real("00000124")
    a = ck.truth_anchors_124(ctx)
    r1 = [(f.id, f.status, f.t_onset, json.dumps(f.values, sort_keys=True), f.text_zh) for f in dc.run(ctx, a)]
    r2 = [(f.id, f.status, f.t_onset, json.dumps(f.values, sort_keys=True), f.text_zh) for f in dc.run(ctx, a)]
    check(r1 == r2, "同樣輸入應有同樣輸出")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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
