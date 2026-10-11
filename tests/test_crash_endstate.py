"""失事分析 · end_state 模組驗證:python -m tests.test_crash_endstate 或 pytest tests/test_crash_endstate.py

(1) 00000124 飛行 2(失事)用 truth / bare 錨點 → 預期狀態與關鍵時間(容許誤差見 acceptance.txt A1/A2/A4/D3/D6)
(2) 00000124 飛行 1(正常降落對照)→ CLEAR,沒有誤報;飛行 2 在 1255.0 之前 → 偵測器全部 CLEAR(acceptance B)
(3) 基準 log 117/118/119/120/122/123(從未解鎖)→ 全部 NA;整份 log 掃描 0 個 >20 m/s² 衝擊;
    閘門拿掉也零觸發(acceptance C / D2,記錄 119 手持揮動離門檻多近)
(4) 合成資料:證明偵測邏輯不是寫死 124 的數字(重算 / 衝擊群 / 撞擊錨點分串 / 自由落體 / 翻覆 / VIBE / 上鎖分類 /
    結束型態優先序),並驗證 hypotheses 依賴的欄位(impact ae_at_impact、vibe type)
(4b) 審查缺陷的回歸測試:無撞擊的多顆 IMU 削波 = 振動削波;飛行中推油門 / 急拉不是撞擊;未宣告落地的觸地只有在之後
    翻覆 / 硬撞擊時才是『觸地後翻覆的起點』;空中上鎖不倒推 2.6 s 撞擊時間;資料覆蓋不足不得 CLEAR;
    ERR 遺失時 ARM.Method=12 仍判墜機;沒起飛的解鎖不是『正常降落』;run(ctx, None) 不丟例外
(5) 每個 id 在每個情境都至少回傳一筆 Finding;確定性;效能 < 1.5 s / 偵測器
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crashkit  # noqa: E402
from core.crash import det_endstate as D  # noqa: E402
from core.crash.anchors import basic_anchors  # noqa: E402
from core.crash.ctx import Ctx  # noqa: E402
from core.crash.types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval  # noqa: E402

IDS = ["crash_check_anchor", "crash_check_replay", "abnormal_disarm_classifier", "impact_shock_clusters",
       "free_fall_run", "tumble_inversion", "vibe_clip_transient"]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def by_id(fs, det_id):
    return [f for f in fs if f.id == det_id]


def one(fs, det_id):
    r = by_id(fs, det_id)
    check(len(r) >= 1, f"{det_id} 沒有 Finding")
    return r[0]


def statuses(fs):
    return {i: sorted({f.status for f in by_id(fs, i)}) for i in IDS}


def check_all_ids(fs, where):
    for i in IDS:
        check(by_id(fs, i), f"[{where}] {i} 沒有回傳 Finding")
    for f in fs:
        check(f.domain == "end_state", f"[{where}] {f.id} domain={f.domain}")
        check(f.status in (HIT, SOFT, CLEAR, NA), f"{f.id} 狀態不明 {f.status}")
        check(f.text_zh, f"[{where}] {f.id} 沒有 text_zh")
        json.dumps(f.values)    # 必須 JSON 可序列化


# =============================================================== 合成 Ctx

G = 9.80665


class SynthCtx(Ctx):
    """不讀 log 檔、直接塞表的 Ctx(測試用)。"""

    def __init__(self, tables, armed, params=None, series=None, instances=None, messages=None, t_end=100.0):
        self.ld = None
        self.ix = None
        self.t_base = 0.0
        self.t_end = t_end
        self._series = dict(series or {})
        self._tables = dict(tables)
        self.params = dict(params or {})
        self.messages = list(messages or [])
        self.events = []
        self.armed = list(armed)
        self._inst = dict(instances or {})

    def instances(self, name):
        return list(self._inst.get(name, []))


def _ramp(t, t0, t1, v0, v1):
    return np.interp(t, [t0, t1], [v0, v1])


def build_synth(dur=60.0, arm=(5.0, 50.0), method=0, open_end=False, roll=None, pitch=None, droll=None,
                spikes=(), ev=(), err=(), modes=(), params=None, series=None, vibe=None, no_imu=False, no_att=False,
                arm_table=True, hz=400.0, freefall=()):
    """合成一次飛行:IMU(2 顆)+ ANG + ARM/EV/ERR/MODE(+ VIBE)。比力由姿態算出(沒有真實加速度),
    spikes = [(t, 寬度 s, 峰值 m/s², 方向 +1/-1 沿 body z)];freefall = [(t, 寬度 s)] 把比力壓到 ~0.5。"""
    t = np.arange(0.0, dur, 1.0 / hz)
    r = np.zeros_like(t) if roll is None else roll(t)
    p = np.zeros_like(t) if pitch is None else pitch(t)
    dr = np.zeros_like(t) if droll is None else droll(t)
    phi, th = np.radians(r), np.radians(p)
    f = np.stack([G * np.sin(th), -G * np.sin(phi) * np.cos(th), -G * np.cos(phi) * np.cos(th)], axis=1)
    for ts, w, peak, sgn in spikes:
        m = (t >= ts) & (t < ts + w)
        f[m, 2] = sgn * peak
        f[m, 0] = 0.0
        f[m, 1] = 0.0
    for ts, w in freefall:
        m = (t >= ts) & (t < ts + w)
        f[m] = 0.0
        f[m, 2] = -0.5
    tables, inst = {}, {}
    if not no_imu:
        for i in (0, 1):
            tables[("IMU", i)] = {"t": t.copy(), "AccX": f[:, 0].copy(), "AccY": f[:, 1].copy(), "AccZ": f[:, 2].copy(),
                                  "GyrX": np.zeros_like(t), "GyrY": np.zeros_like(t), "GyrZ": np.zeros_like(t)}
        inst["IMU"] = [0, 1]
    if not no_att:
        tables[("ANG", None)] = {"t": t.copy(), "Roll": ((r + 180.0) % 360.0) - 180.0, "Pitch": p, "Yaw": np.zeros_like(t),
                                 "DesRoll": dr, "DesPitch": np.zeros_like(t), "DesYaw": np.zeros_like(t)}
    if ev:
        e = sorted(ev)
        tables[("EV", None)] = {"t": np.array([x[0] for x in e], float), "Id": np.array([x[1] for x in e], float)}
    if err:
        e = sorted(err)
        tables[("ERR", None)] = {"t": np.array([x[0] for x in e], float), "Subsys": np.array([x[1] for x in e], float),
                                 "ECode": np.array([x[2] for x in e], float)}
    if modes:
        tables[("MODE", None)] = {"t": np.array([x[0] for x in modes], float), "ModeNum": np.array([x[1] for x in modes], float),
                                  "Mode": np.array([x[1] for x in modes], float)}
    t1 = dur - 0.5 if open_end else arm[1]
    iv = ArmedInterval(0, arm[0], t1, None if open_end else method, False, open_end)
    if arm_table:
        rows_t, rows_s, rows_m = [arm[0]], [1.0], [0.0]
        if not open_end:
            rows_t.append(arm[1]); rows_s.append(0.0); rows_m.append(float(method))
        tables[("ARM", None)] = {"t": np.array(rows_t), "ArmState": np.array(rows_s), "Method": np.array(rows_m),
                                 "Forced": np.zeros(len(rows_t))}
    if vibe is not None:
        tv, vx, vy, vz, clip = vibe
        for i in range(clip.shape[0]):
            tables[("VIBE", i)] = {"t": tv, "VibeX": vx[i], "VibeY": vy[i], "VibeZ": vz[i], "Clip": clip[i]}
        inst["VIBE"] = list(range(clip.shape[0]))
    ctx = SynthCtx(tables, [iv], params=params, series=series, instances=inst, t_end=dur)
    return ctx, iv


def anchors_for(ctx):
    return basic_anchors(ctx)


def crash_roll(t, t0=30.0, dur=2.0):
    return _ramp(t, t0, t0 + dur, 0.0, 180.0)


def make_vibe(dur=60.0, level=5.0, bursts=(), clips=()):
    """3 顆 IMU 的 VIBE(10 Hz):level 為底;bursts=[(t0, t1, 值)];clips=[(t, [各 instance 增量])]。"""
    tv = np.arange(0.0, dur, 0.1)
    v = np.full((3, tv.size), level)
    for a_, b_, val in bursts:
        v[:, (tv >= a_) & (tv < b_)] = val
    clip = np.zeros((3, tv.size))
    for tc, incs in clips:
        k = int(np.searchsorted(tv, tc))
        for i, d in enumerate(incs):
            clip[i, k:] += d
    return tv, v.copy(), v.copy(), v.copy(), clip


# =============================================================== (1) 124 失事飛行

_CACHE = {}


def crash_run(which="truth"):
    if which not in _CACHE:
        ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
        a = crashkit.truth_anchors_124(ctx) if which == "truth" else basic_anchors(ctx)
        t = time.time()
        fs = D.run(ctx, a)
        _CACHE[which] = (ctx, a, fs, time.time() - t)
    return _CACHE[which]


def test_124_statuses_truth():
    ctx, a, fs, dt = crash_run("truth")
    check_all_ids(fs, "124/truth")
    st = statuses(fs)
    print(f"  狀態:{st}  run() {dt:.2f}s")
    for i in IDS:
        check(st[i] == [HIT], f"{i} 應為 HIT,實際 {st[i]}")
    check(a.end_type == "crash_check", a.end_type)


def test_124_crash_check_anchor():
    ctx, a, fs, _ = crash_run("truth")
    f = one(fs, "crash_check_anchor")
    v = f.values
    check(abs(f.t_alarm - 1262.125) <= 0.05, f"ERR 時間 {f.t_alarm}")
    check(v["arm_method"] == 12 and "ARM.Method=12" in v["confirmed_by"], v)
    check("MSG" in v["confirmed_by"] and "EV11" in v["confirmed_by"], v["confirmed_by"])
    check(v["angerr_deg"] == 150.0 and v["angerr_limit"] == 30.0 and v["accel_ms2"] == 0.1 and v["accel_limit"] == 3.0, v)
    check(v["ecode"] == 1 and abs(v["t_err"] - 1262.125) <= 0.05, v)
    check(2.4 <= v["err_minus_impact_s"] <= 2.8, f"ERR 比撞擊晚 {v['err_minus_impact_s']}")
    check(abs(v["t_rest"] - 1260.04) <= 0.05, f"t_rest {v['t_rest']}")
    check(v["inverted"] is True, "靜止時應是倒扣")
    check(f.t_onset <= f.t_alarm and f.context == "crash_check", (f.t_onset, f.context))
    print(f"  ERR {f.t_alarm:.3f}  t_rest {v['t_rest']}  ERR−撞擊 {v['err_minus_impact_s']} s  倒扣 {v['inverted']}")


def test_124_crash_check_replay():
    ctx, a, fs, _ = crash_run("truth")
    f = one(fs, "crash_check_replay")
    v = f.values
    check(abs(f.t_alarm - 1262.125) <= 0.05, f"重算 t_fire {f.t_alarm}")
    check(v["validated"] is True and abs(v["dt_vs_err_s"]) < 0.05, v)
    check(abs(f.t_onset - 1260.130) <= 0.05, f"條件起點 {f.t_onset}")
    check(149.0 <= v["ae_at_fire_deg"] <= 151.0, f"重算 AngErr {v['ae_at_fire_deg']}(acceptance 149~151,MSG 150)")
    check(v["accf_at_fire"] < 0.5, v["accf_at_fire"])          # MSG Accel=0.1
    check(f.context == "replay" and f.status == HIT and f.severity == "high", (f.context, f.severity))
    print(f"  fire {f.t_alarm:.3f}(ERR 差 {v['dt_vs_err_s'] * 1000:.0f} ms)起點 {f.t_onset:.3f} AngErr {v['ae_at_fire_deg']}° accf {v['accf_at_fire']}")


def test_124_impact_clusters():
    ctx, a, fs, _ = crash_run("truth")
    fl = by_id(fs, "impact_shock_clusters")
    check(len(fl) == 3, f"應有 3 筆(觸地 + 2 次硬撞擊),實際 {len(fl)}")
    gc = [f for f in fl if f.context == "ground_contact"]
    hard = [f for f in fl if f.context == "hard_impact"]
    check(len(gc) == 1 and len(hard) == 2, [f.context for f in fl])
    check(gc[0].status == HIT and abs(gc[0].t_onset - 1255.225) <= 0.05 and 26 <= gc[0].values["peak"] <= 30, gc[0].values)
    check(gc[0].values["recognised_dt"] is None and gc[0].values["dV"] < D.HARD_DV, "觸地後 2 s 內沒有 EV17/EV18")
    # 『觸地後翻覆的起點』要有後續:2.84 s 後翻覆(1258.06),2.96 s 後首撞
    check(gc[0].values["followed_by"] == "tumble" and 2.7 <= gc[0].values["follow_dt_s"] <= 3.0 and "翻覆』的起點" in gc[0].text_zh,
          (gc[0].values["followed_by"], gc[0].values["follow_dt_s"]))
    # 脈衝形狀:124 的每段連續 >20 m/s² 都 <0.1 s(SUSTAINED_RUN_S=0.2 的依據),沒有平台
    for f in fl:
        check(f.values["pulse_max_s"] < 0.1 and f.values["sustained_s"] == 0.0 and f.values["peak_impulse"] == f.values["peak"], f.values)
        check(f.values["coverage"]["ok"] is True and f.values["coverage"]["max_gap_s"] < 0.5, f.values["coverage"])
    check(a.extra["end_state"]["sustained_accel"] == [], a.extra["end_state"]["sustained_accel"])
    h1, h2 = hard
    # acceptance D3:1258.183–1258.808 hard (62.1)、1259.463–1259.958 hard (90.1 raw / 88.7 濾波)
    check(abs(h1.t_alarm - 1258.40) <= 0.1 and abs(h1.t_onset - 1258.183) <= 0.05, (h1.t_onset, h1.t_alarm))
    check(abs(h1.values["t_end"] - 1258.808) <= 0.05 and abs(h2.values["t_start"] - 1259.463) <= 0.05
          and abs(h2.values["t_end"] - 1259.958) <= 0.05, (h1.values["t_end"], h2.values["t_start"], h2.values["t_end"]))
    check(56 <= h1.values["peak"] <= 64 and h1.values["dir"] == "lateral" and 4.0 <= h1.values["dV"] <= 5.2, h1.values)
    check(abs(h2.values["t_peak"] - 1259.50) <= 0.05 and 86 <= h2.values["peak"] <= 92, h2.values)
    check(6.4 <= h2.values["dV"] <= 7.4 and h2.values["dir"] == "top", h2.values)
    # 撞擊前的姿態:觸地時姿態正常(<10°),兩次硬撞擊前都已失控(>30°)→ 撞擊是結果,外力撞擊假說不成立
    check(gc[0].values["ae_at_impact"] < 10.0, gc[0].values["ae_at_impact"])
    check(h1.values["ae_at_impact"] > 30.0 and h2.values["ae_at_impact"] > 150.0, (h1.values["ae_at_impact"], h2.values["ae_at_impact"]))
    check(h2.values["tilt_at_impact"] > 170.0, h2.values["tilt_at_impact"])
    check(570 <= h2.values["gyro_peak_dps"] <= 610, h2.values["gyro_peak_dps"])
    check(0.04 <= h2.values["width40_s"] <= 0.07, h2.values["width40_s"])
    allc = a.extra["end_state"]["impacts_all"]
    got = [(round(c["t_start"], 1), c["cls"]) for c in allc]
    check(got == [(615.6, "ground_contact"), (1255.2, "ground_contact"), (1258.2, "hard_impact"), (1259.5, "hard_impact")], got)
    # 錨點
    check(abs(a.t_impact_peak - 1259.50) <= 0.1, a.t_impact_peak)
    check(abs(a.t_first_hit - 1258.40) <= 0.1, a.t_first_hit)
    check(abs(a.t_rest - 1260.04) <= 0.05, a.t_rest)
    check(len(a.impacts) == 3 and all(c["interval"] == 1 for c in a.impacts), a.impacts)
    check(2.4 <= a.t_detect - a.t_impact_peak <= 2.8, a.t_detect - a.t_impact_peak)
    print(f"  群:{got}  t_impact {a.t_impact_peak}  t_first_hit {a.t_first_hit}  t_rest {a.t_rest}")


def test_124_free_fall():
    ctx, a, fs, _ = crash_run("truth")
    f = one(fs, "free_fall_run")
    v = f.values
    check(abs(f.t_onset - 1259.266) <= 0.05 and abs(v["dur_s"] - 0.157) <= 0.03, (f.t_onset, v["dur_s"]))
    check(f.context == "in_tumble" and v["then_impact"] is True and v["drop_m"] >= 0.1, (f.context, v))
    check(v["n_runs"] == 1, v["n_runs"])
    print(f"  自由落體 {f.t_onset:.3f} s,{v['dur_s']} s,落差 ≥{v['drop_m']} m,{f.context}")


def test_124_tumble():
    ctx, a, fs, _ = crash_run("truth")
    f = one(fs, "tumble_inversion")
    v = f.values
    check(abs(f.t_onset - 1258.06) <= 0.1, f.t_onset)
    check(1259.1 <= v["t_inverted"] <= 1259.35, v["t_inverted"])
    check(v["axis"] == "roll" and v["accel_confirmed"] is True and f.context == "inverted", (v["axis"], v["accel_confirmed"], f.context))
    check(1.5 <= v["time_to_invert_s"] <= 2.5, v["time_to_invert_s"])
    check(v["t_inverted"] <= 1259.4 and v["tilt_max_deg"] > 170, v)
    check(a.t_warn is not None and a.t_inverted == 1259.2, "不應改動別的模組的錨點")
    print(f"  翻覆 {f.t_onset:.3f}  倒扣 {v['t_inverted']}  到倒扣 {v['time_to_invert_s']} s  加速度證實 {v['accel_confirmed']}")


def test_124_vibe():
    ctx, a, fs, _ = crash_run("truth")
    f = one(fs, "vibe_clip_transient")
    v = f.values
    check(f.context == "impact_transient" and f.status == HIT, f.context)
    check(abs(v["t_first_over"] - 1258.44) <= 0.15, v["t_first_over"])
    check(abs(v["clip_events"][0]["t"] - 1259.74) <= 0.15 and v["clip_events"][0]["n_inst"] == 3, v["clip_events"])
    check(v["pre_impact_p95"] <= 10.1 and v["pre_impact_max"] <= 11.5, (v["pre_impact_p95"], v["pre_impact_max"]))
    check(v["clip_before_first_hit"] == 0 and v["peak_vibe"] > 70, v)
    check(v["type"] == "impact clipping", v["type"])        # hypotheses 以 values['type'] 篩選(不是振動問題)
    check(v["impact_cls"] == "hard_impact" and abs(v["impact_t_peak"] - 1258.41) <= 0.05 and v["stats_window"] == "pre_impact", v)
    check(all(x.context == "impact_transient" for x in by_id(fs, "vibe_clip_transient")), "撞擊暫態以外不該有別的 VIBE 結論")
    print(f"  VIBE 暴增 {v['t_first_over']}  Clip {v['clip_events'][0]['t']}  撞擊前 p95 {v['pre_impact_p95']} max {v['pre_impact_max']}")


def test_124_abnormal_disarm():
    ctx, a, fs, _ = crash_run("truth")
    f = one(fs, "abnormal_disarm_classifier")
    v = f.values
    check(f.status == HIT and f.context == "crash_check" and v["class"] == "crash_check" and v["method"] == 12, (f.context, v))
    check(v["landed_before"] is False and v["ev17_18_written_by_disarm"] == 2, v)
    check(abs(v["agl_5s_m"] - (-0.13)) <= 0.1, f"agl {v['agl_5s_m']}")
    check([i["class"] for i in v["intervals"]] == ["normal_landing", "crash_check"], v["intervals"])
    check(v["dt17_s"] is None and v["dt18_s"] is None, "上鎖當下補寫的 EV17/EV18 不得算成落地")
    print(f"  分類 {v['class']}  各區間 {[i['class'] for i in v['intervals']]}  agl {v['agl_5s_m']}")


def test_124_end_state_anchors():
    ctx, a, fs, _ = crash_run("truth")
    check(a.end_type == "crash_check", a.end_type)
    check(abs(a.t_detect - 1262.125) <= 0.05 and abs(a.t_disarm - 1262.125) <= 0.05, (a.t_detect, a.t_disarm))
    check(a.t_warn == 1255.9 and a.t_loss == 1258.10 and a.t_touch == 1255.225, "別的模組的錨點不得被改動")
    print(f"  end_type {a.end_type}  t_detect {a.t_detect}  notes {a.notes}")


def test_124_bare_anchors():
    """只有 basic_anchors(沒有 t_warn / t_loss ...)也要得到同樣的結論。"""
    ctx, a, fs, _ = crash_run("bare")
    check_all_ids(fs, "124/bare")
    st = statuses(fs)
    for i in IDS:
        check(st[i] == [HIT], f"{i}: {st[i]}")
    check(a.end_type == "crash_check" and abs(a.t_impact_peak - 1259.50) <= 0.1 and abs(a.t_first_hit - 1258.40) <= 0.1,
          (a.end_type, a.t_impact_peak, a.t_first_hit))
    check(a.t_warn is None and a.t_loss is None and a.t_inverted is None and a.t_touch is None, "不得設別人的錨點")
    print(f"  bare:end {a.end_type}  impact {a.t_impact_peak}  first {a.t_first_hit}  rest {a.t_rest}")


def test_124_pre_event_no_hit():
    """飛行 2 在 1255.0 之前(健康下降 / 懸停):沒有衝擊、自由落體、翻覆、VIBE 超標。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    check(D.impact_clusters_for_window(ctx, 873.14, 1255.0) == [], "起飛 / 懸停不該有 >20 m/s² 衝擊")
    w = D.build_imu_window(D._imu_instances(ctx), 873.14, 1255.0)
    check(float(w.acons.min()) > 5.0 and float(w.acons.max()) < 20.0, (w.acons.min(), w.acons.max()))
    check(not D._runs(w.acons < D.FF_ACC, w.t, D.FF_MIN_S, D.FF_GAP_S), "不該有自由落體")
    a = basic_anchors(ctx)
    sh = D._Sh(ctx, a)
    att = sh.att(873.14, 1255.0)
    m = att["t"] <= 1255.0
    from core.crash.util import tilt_deg
    tmax = float(tilt_deg(att["roll"][m], att["pitch"][m]).max())
    check(tmax < 15.0, f"失事前最大傾角 {tmax}")
    print(f"  失事前:acons {w.acons.min():.1f}~{w.acons.max():.1f}  最大傾角 {tmax:.1f}°")


def test_124_pre_event_detectors_clear():
    """acceptance B(MUST NOT):同一批偵測器跑在飛行 2 的 1255.0 之前 → 全部 CLEAR(不只是原始訊號乾淨)。
    解鎖區間截到 1254.0(上鎖後 1 s 的衝擊視窗止於 1255.0,不含 1255.225 的觸地);
    abnormal_disarm_classifier 不適用(截斷的區間沒有真正的上鎖)。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    f2 = ctx.armed[1]
    iv = ArmedInterval(f2.index, f2.t0, 1254.0, 0)
    a = Anchors(interval=iv, intervals=[ctx.armed[0], iv])
    sh = D._Sh(ctx, a)
    got = {}
    for fn in (D.crash_check_anchor, D.crash_check_replay, D.impact_shock_clusters, D.free_fall_run,
               D.tumble_inversion, D.vibe_clip_transient):
        fs = fn(ctx, a, sh)
        check(fs and all(f.status == CLEAR for f in fs), f"{fn.__name__} 在 1255 前應 CLEAR:{[(f.status, f.context) for f in fs]}")
        got[fn.__name__] = fs[0].values
    check(got["crash_check_replay"]["ae_max_deg"] < 10.0, got["crash_check_replay"]["ae_max_deg"])
    check(got["tumble_inversion"]["tilt_max_deg"] < 15.0, got["tumble_inversion"]["tilt_max_deg"])
    check(got["impact_shock_clusters"]["acons_max_in_flight"] < D.IMPACT_ONSET, got["impact_shock_clusters"])
    check(got["vibe_clip_transient"]["vibe_max"] < 15.0 and got["vibe_clip_transient"]["n_clip_events"] == 0, got["vibe_clip_transient"])
    print(f"  873.14–1254.0:6 個偵測器 CLEAR;ae max {got['crash_check_replay']['ae_max_deg']}°、傾角 {got['tumble_inversion']['tilt_max_deg']}°、"
          f"|a| max {got['impact_shock_clusters']['acons_max_in_flight']}、VIBE max {got['vibe_clip_transient']['vibe_max']}")


# =============================================================== (2) 飛行 1 對照

def test_124_flight1_clear():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.flight1_anchors_124(ctx)
    fs = D.run(ctx, a)
    check_all_ids(fs, "124/f1")
    st = statuses(fs)
    print(f"  狀態:{st}")
    for i in IDS:
        check(st[i] == [CLEAR], f"{i} 應為 CLEAR,實際 {st[i]}")
    check(a.end_type == "normal_landing" and a.t_detect is None and a.t_impact_peak is None and a.t_first_hit is None and a.t_rest is None,
          (a.end_type, a.t_detect, a.t_impact_peak))
    check(abs(a.t_disarm - 619.14) <= 0.05, a.t_disarm)
    f = one(fs, "impact_shock_clusters")
    check(f.context == "normal_touchdown" and 24 <= f.values["clusters"][0]["peak"] <= 27, f.values)
    check(f.values["coverage"]["ok"] is True and f.values["n_sustained"] == 0, f.values)   # 資料完整 → 可以 CLEAR
    check(len(a.impacts) == 1 and a.impacts[0]["cls"] == "ground_contact" and abs(a.impacts[0]["t_start"] - 615.57) <= 0.05, a.impacts)
    d = one(fs, "abnormal_disarm_classifier")
    check(d.context == "normal_landing" and d.values["method"] == 0, d.values)
    check(abs(d.values["dt17_s"] - 2.75) <= 0.1 and abs(d.values["dt18_s"] - 1.95) <= 0.1, d.values)
    t = one(fs, "tumble_inversion").values
    check(t["tilt_max_deg"] <= 12.0, t["tilt_max_deg"])
    r = one(fs, "crash_check_replay").values
    check(r["ae_max_deg"] <= 3.6, r["ae_max_deg"])          # acceptance B:ae max ≤3.6°
    v = one(fs, "vibe_clip_transient").values
    check(v["vibe_max"] <= 12.0 and v["n_clip_events"] == 0, v)
    check(one(fs, "free_fall_run").values["acons_min"] > 5.0, "飛行 1 不該有自由落體")
    print(f"  end_type {a.end_type}  觸地 {a.impacts[0]['t_start']} ({a.impacts[0]['peak']} m/s²)  dt17 {d.values['dt17_s']} dt18 {d.values['dt18_s']}")


# =============================================================== (3) 基準 log:從未解鎖

def test_baselines_all_na():
    for name in crashkit.BASELINES:
        ctx = crashkit.load_ctx(name)
        a = crashkit.not_armed_anchors(ctx)
        fs = D.run(ctx, a)
        check_all_ids(fs, name)
        st = statuses(fs)
        for i in IDS:
            check(st[i] == [NA], f"{name} {i} 應為 NA,實際 {st[i]}")
        check(all(f.context == "not_armed" for f in fs), [f.context for f in fs])
        check(a.end_type == "not_armed" and a.t_impact_peak is None and a.impacts == [], a.end_type)
    print(f"  {len(crashkit.BASELINES)} 個基準 log:7 個偵測器全部 NA(not_armed),end_type=not_armed")


def test_baselines_no_impact_clusters():
    """整份 log 掃描(不看解鎖):沒有任何 >20 m/s² 衝擊,合加速度最大 10~17 m/s²(= 手持 / 桌上的上限)。"""
    mx = {}
    for name in crashkit.BASELINES:
        ctx = crashkit.load_ctx(name)
        cl = D.impact_clusters_for_window(ctx, -10.0, ctx.t_end + 1.0)
        w = D.build_imu_window(D._imu_instances(ctx), -10.0, ctx.t_end + 1.0)
        check(cl == [], f"{name} 不該有衝擊群:{cl}")
        mx[name] = float(w.acons.max())
        check(9.0 < mx[name] < 17.5, f"{name} 最大合加速度 {mx[name]}")
        check(not D._runs(w.acons < D.FF_ACC, w.t, D.FF_MIN_S, D.FF_GAP_S), f"{name} 不該有自由落體(手持最低也 >3)")
    print("  整份 log 最大合加速度:" + "  ".join(f"{k[-3:]}={v:.1f}" for k, v in mx.items()))


def test_baselines_ungated_no_fire():
    """acceptance D(2) / C:把解鎖閘門拿掉(整份 log 當成一次解鎖)— 重算 CRASH_CHECK、翻覆、自由落體、衝擊、VIBE 仍全部 CLEAR。
    記下最接近門檻的量測(119 手持揮動):ae 49°(條件最長只連續 ~0.5 s,門檻 2 s)、傾角 49° 但 ≥45° 不到 0.3 s、
    合加速度最低 3.6(門檻 <3);120 搖桿 ae 30.4°。實際報告靠解鎖閘門給 NA,這裡證明門檻本身也有餘裕。"""
    near = {}
    for name in crashkit.BASELINES:
        ctx = crashkit.load_ctx(name)
        # 「整份 log」= 資料實際涵蓋的範圍(各串流第一筆之後);從 −10 s 起算會有 ~10 s 沒資料的開頭,
        # 那樣偵測器正確地回 NA(partial_coverage),不能說整段乾淨 —— 一併驗證
        t_first = max(float(ctx.table(n, i)["t"][0]) for n, i in (("IMU", 0), ("ANG", None), ("VIBE", 0)))
        iv_bad = ArmedInterval(0, -10.0, ctx.t_end, 0)
        a_bad = Anchors(interval=iv_bad, intervals=[iv_bad])
        for fn in (D.crash_check_replay, D.tumble_inversion, D.free_fall_run, D.impact_shock_clusters, D.vibe_clip_transient):
            fs = fn(ctx, a_bad, D._Sh(ctx, a_bad))
            check(all(f.status == NA and f.context == "partial_coverage" for f in fs),
                  f"{name} {fn.__name__}:開頭 ~10 s 沒資料不得 CLEAR:{[(f.status, f.context) for f in fs]}")
        iv = ArmedInterval(0, t_first, ctx.t_end, 0)
        a = Anchors(interval=iv, intervals=[iv])
        sh = D._Sh(ctx, a)
        vals = {}
        for fn in (D.crash_check_replay, D.tumble_inversion, D.free_fall_run, D.impact_shock_clusters, D.vibe_clip_transient):
            fs = fn(ctx, a, sh)
            check(fs and all(f.status == CLEAR for f in fs), f"{name} {fn.__name__} 不閘門也該 CLEAR:{[(f.status, f.context) for f in fs]}")
            vals.update(fs[0].values)
        near[name] = vals
    v119 = near["00000119"]
    check(v119["ae_max_deg"] > 45.0 and v119["longest_cond_s"] < 1.0, (v119["ae_max_deg"], v119["longest_cond_s"]))
    check(v119["tilt_max_deg"] > 45.0, v119["tilt_max_deg"])          # 傾角超過 45°,但持續不到 0.3 s
    check(3.0 < v119["acons_min"] < 4.0, v119["acons_min"])
    check(near["00000120"]["ae_max_deg"] >= 30.0 and near["00000120"]["longest_cond_s"] < 0.5, near["00000120"])
    print("  不閘門:" + "  ".join(f"{k[-3:]} ae {v['ae_max_deg']}/{v['longest_cond_s']}s 傾角 {v['tilt_max_deg']} |a|min {v['acons_min']}"
                                  for k, v in near.items()))


# =============================================================== (4) 合成資料

def test_impact_clusters_pure():
    hz = 400.0
    t = np.arange(0, 20, 1 / hz)
    ac = np.full(t.size, 9.81)

    def bump(t0, w, peak):
        m = (t >= t0) & (t < t0 + w)
        ac[m] = 9.81 + (peak - 9.81) * np.sin(np.pi * (t[m] - t0) / w)

    bump(3.0, 0.04, 26.0)        # 正常觸地
    # 硬撞擊:仿 124 首撞(0.6 s 內一串 30~60 ms 的短脈衝;124 每段連續 >20 m/s² 最長 0.09 s),同一群(間隔 <0.3 s)
    for t0_, w_, pk_ in ((8.00, 0.04, 30.0), (8.10, 0.05, 45.0), (8.22, 0.06, 62.0), (8.40, 0.05, 45.0), (8.55, 0.03, 28.0)):
        bump(t0_, w_, pk_)
    bump(14.0, 0.01, 55.0)       # 尖峰:peak≥40 但 dV<3 → hard_landing
    acc_xyz = np.zeros((t.size, 3))
    acc_xyz[:, 2] = -ac          # 力朝 −z(機底)
    cl = D.impact_clusters(t, ac, acc_xyz, np.full(t.size, 30.0))
    got = [(round(c["t_start"], 1), c["cls"]) for c in cl]
    check(got == [(3.0, "ground_contact"), (8.0, "hard_impact"), (14.0, "hard_landing")], got)
    check(cl[0]["dir"] == "bottom" and cl[0]["gyro_peak_dps"] == 30.0, cl[0])
    check(cl[1]["t_hard"] is not None and 8.0 < cl[1]["t_hard"] < 8.3 and cl[1]["dV"] >= 3.0, cl[1])
    check(cl[1]["sustained_s"] == 0.0 and cl[1]["pulse_max_s"] < 0.1 and cl[1]["peak_impulse"] == cl[1]["peak"], cl[1])
    check(cl[2]["t_hard"] is not None and cl[2]["dV"] < 3.0, cl[2])
    # 比力方向 → 側向 / 倒扣
    acc_xyz[:] = 0
    acc_xyz[:, 1] = -ac
    check(D.impact_clusters(t, ac, acc_xyz)[0]["dir"] == "lateral", "方向應為 lateral")
    acc_xyz[:] = 0
    acc_xyz[:, 2] = ac
    check(D.impact_clusters(t, ac, acc_xyz)[0]["dir"] == "top", "方向應為 top")
    # dV 的資料空洞上限:空洞(0.25 s,仍 ≤0.3 s 合併成同一群)不得被當成長時間的高加速度
    t2 = np.array([0.0, 0.1, 0.2, 0.45, 0.5])
    ac2 = np.array([9.8, 30.0, 30.0, 30.0, 9.8])
    c2 = D.impact_clusters(t2, ac2)
    check(len(c2) == 1 and c2[0]["dV"] < 3.0, c2)          # 不設上限會積出 (30−9.81)×0.4 = 8.1 m/s → 誤判硬撞擊
    check(c2[0]["cls"] == "ground_contact", c2)
    # 間隔 >0.3 s → 兩群
    t3 = np.array([0.0, 0.1, 0.2, 1.2, 1.3])
    c3 = D.impact_clusters(t3, ac2)
    check([round(c["t_start"], 1) for c in c3] == [0.1, 1.2] and all(c["dV"] < 3.0 for c in c3), c3)
    check(D.impact_clusters(t, np.full(t.size, 9.81)) == [], "沒有衝擊要回傳空清單")
    # 飛行動作的平台(油門推滿 / 急拉):連續 >0.2 s 的 >20 m/s² → sustained_accel,dV 不累積(舊算法 25×0.8 s 會積出 12 m/s)
    sus = {}
    for lvl, w_ in ((25.0, 0.8), (22.0, 0.25), (45.0, 0.4), (30.0, 2.0)):
        acp = np.full(t.size, 9.81)
        m = (t >= 5.0) & (t < 5.0 + w_)
        acp[m] = lvl + 1.2 * np.sin(2 * np.pi * 37.0 * t[m])                       # 平台上的振動雜訊
        up = (t >= 4.95) & (t < 5.0)
        acp[up] = 9.81 + (lvl - 9.81) * (t[up] - 4.95) / 0.05                         # 50 ms 爬升(馬達加速)
        cp = D.impact_clusters(t, acp)
        check(len(cp) == 1 and cp[0]["cls"] == "sustained_accel" and cp[0]["dV"] < D.HARD_DV
              and cp[0]["peak_impulse"] < D.IMPACT_ONSET and cp[0]["t_hard"] is None, (lvl, w_, cp))
        sus[(lvl, w_)] = cp[0]["dV"]
    # 平台上疊一個真的撞擊脈衝 → 仍是硬撞擊等級(只扣平台,不扣脈衝)
    acp = np.full(t.size, 9.81)
    acp[(t >= 5.0) & (t < 5.8)] = 25.0
    m = (t >= 5.3) & (t < 5.38)
    acp[m] = 25.0 + 65.0 * np.sin(np.pi * (t[m] - 5.3) / 0.08)
    cp = D.impact_clusters(t, acp)
    check(len(cp) == 1 and cp[0]["cls"] == "hard_impact" and cp[0]["peak_impulse"] >= D.HARD_PEAK and 5.3 < cp[0]["t_hard"] < 5.38, cp)
    # 已知限制(刻意的設計):單一平滑、連續 >0.2 s 的寬鼓包(0.3 s 半正弦到 62 m/s²)與平台分不開 → 不算撞擊;
    # 124 的撞擊每段 ≤0.09 s,真實撞地多半 <0.1 s,只有極軟的地面(雪堆 / 灌木)才可能這麼寬
    acp = np.full(t.size, 9.81)
    m = (t >= 5.0) & (t < 5.3)
    acp[m] = 9.81 + 52.19 * np.sin(np.pi * (t[m] - 5.0) / 0.3)
    check(D.impact_clusters(t, acp)[0]["cls"] == "sustained_accel", "寬鼓包 = 平台(已知限制)")
    print(f"  群:{got};空洞上限 dV={c2[0]['dV']};間隔 1 s 分成 {len(c3)} 群;平台 dV {sus}")


def test_crash_dwell_pure():
    t = np.arange(0, 10, 0.02)
    cond = (t >= 3.0) & (t < 4.9)
    check(D.crash_dwell(t, cond)[0] is None, "1.9 s 不得觸發")
    cond = (t >= 3.0) & (t < 5.1)
    tf, ts, lg = D.crash_dwell(t, cond)
    check(tf is not None and abs(tf - 4.98) < 0.03 and abs(ts - 2.98) < 0.03 and lg >= 2.0, (tf, ts, lg))
    cond = ((t >= 3.0) & (t < 4.0)) | ((t >= 4.1) & (t < 5.5))     # 中斷一個樣本以上 → 重新計時
    cond[np.searchsorted(t, 4.0):np.searchsorted(t, 4.1)] = False
    check(D.crash_dwell(t, cond)[0] is None, "中斷後重新計時,不得合併成 2.4 s")
    tt = np.r_[np.arange(0, 3, 0.02), np.arange(4.0, 8, 0.02)]      # 資料空洞 1 s:不得當成連續
    cond = (tt >= 2.0) & (tt < 4.9)
    check(D.crash_dwell(tt, cond)[0] is None, "空洞 >0.5 s 要重新計時")
    check(D.crash_dwell(np.array([0.0]), np.array([True]))[0] is None, "樣本太少")
    print("  dwell:1.9 s 不觸發、2.1 s 觸發、中斷與空洞重新計時")


def test_replay_synthetic_crash():
    """翻轉 180° 後倒扣不動:重算在『衝擊後加速度濾波值回到 <3』再 2 s 後觸發;不是寫死 124 的時間。"""
    for t_flip in (30.0, 18.0):
        ctx, iv = build_synth(dur=60, arm=(5, 55), method=0, roll=lambda t, f=t_flip: crash_roll(t, f),
                              spikes=[(t_flip + 2.0, 0.05, -70.0, +1)], ev=[(6.0, 28)], params={"FS_CRASH_CHECK": 1.0})
        a = anchors_for(ctx)
        f = D.crash_check_replay(ctx, a)[0]
        check(f.status == HIT, f"{f.status} {f.text_zh}")
        exp = t_flip + 2.0 + 0.3 + D.CRASH_DWELL_S           # 撞擊後 1 Hz 濾波殘留 ~0.3 s + 2 s
        check(abs(f.t_alarm - exp) <= 0.35, f"t_fire {f.t_alarm} 預期 ≈{exp}")
        check(abs((f.t_alarm - f.t_onset) - D.CRASH_DWELL_S) < 1e-6, "t_onset = t_fire − 2 s")
        check(f.values["ae_at_fire_deg"] > 170 and "t_err" not in f.values, f.values)
        check(f.context == "replay_no_err", f.context)
        print(f"  翻轉 {t_flip}:重算觸發 {f.t_alarm:.2f}(預期 ≈{exp:.2f})")


def test_replay_gates():
    base = dict(dur=60, arm=(5, 55), method=0, roll=lambda t: crash_roll(t, 30.0), params={"FS_CRASH_CHECK": 1.0})
    # 已落地(EV18)→ 不觸發
    ctx, _ = build_synth(**base, ev=[(6.0, 28), (29.0, 18)])
    f = D.crash_check_replay(ctx, anchors_for(ctx))[0]
    check(f.status == CLEAR, f"已落地狀態不該觸發:{f.status}")
    # ACRO(1)/ FLIP(14) 模式 → 不觸發
    for mode in (1, 14):
        ctx, _ = build_synth(**base, ev=[(6.0, 28)], modes=[(0.0, 5), (20.0, mode)])
        f = D.crash_check_replay(ctx, anchors_for(ctx))[0]
        check(f.status == CLEAR, f"模式 {mode} 不該觸發:{f.status}")
    # 有 DesRoll 跟著翻(姿態誤差 <30°)→ 不觸發(刻意翻轉 / 控制得住)
    ctx, _ = build_synth(**base, ev=[(6.0, 28)], droll=lambda t: crash_roll(t, 30.0))
    f = D.crash_check_replay(ctx, anchors_for(ctx))[0]
    check(f.status == CLEAR and f.values["ae_max_deg"] < 5, f.values)
    # 沒有 EV28 的舊韌體:不用落地遮罩,照樣能重算
    ctx, _ = build_synth(**base, ev=[(29.0, 18)])
    f = D.crash_check_replay(ctx, anchors_for(ctx))[0]
    check(f.status == HIT, "沒有 EV28 時不應被 EV18 遮蔽")
    # 翻轉後一直有加速度(例如被搬動):濾波值 >3 不觸發。震動要從翻轉開始就存在 —— 合成的翻轉本身沒有
    # 平移加速度(比力 = 純重力),若震動晚 3 s 才開始,條件在翻轉途中就已連續成立 2 s(那是正確的觸發)。
    ctx, _ = build_synth(**base, ev=[(6.0, 28)], spikes=[(30.0 + 0.5 * k, 0.1, 40.0, +1) for k in range(50)])
    f = D.crash_check_replay(ctx, anchors_for(ctx))[0]
    check(f.status == CLEAR and f.values["ae_max_deg"] > 170, f"持續劇烈震動不該觸發:{f.status} {f.values}")
    check(f.values["longest_cond_s"] < D.CRASH_DWELL_S, f.values)
    # 資料缺失 → NA(不是 CLEAR)
    ctx, _ = build_synth(**base, ev=[(6.0, 28)], no_att=True)
    check(D.crash_check_replay(ctx, anchors_for(ctx))[0].status == NA, "沒姿態資料要回 NA")
    ctx, _ = build_synth(**base, ev=[(6.0, 28)], no_imu=True)
    check(D.crash_check_replay(ctx, anchors_for(ctx))[0].status == NA, "沒 IMU 要回 NA")
    # 未解鎖
    ctx, _ = build_synth(**base)
    ctx.armed = []
    f = D.crash_check_replay(ctx, anchors_for(ctx))[0]
    check(f.status == NA and f.context == "not_armed", f.context)
    print("  已落地 / ACRO / FLIP / 控制得住 / 持續震動 → 不觸發;舊韌體照樣重算;缺資料 → NA")


def test_anchor_err_variants():
    base = dict(dur=60, arm=(5, 40), roll=lambda t: crash_roll(t, 30.0), ev=[(6.0, 28)])
    msg = [(35.0, "Crash: Disarming: AngErr=151>30, Accel=0.2<3.0")]
    ctx, iv = build_synth(**base, method=12, err=[(35.0, 12, 1)], params={"FS_CRASH_CHECK": 1.0})
    ctx.messages = msg
    ctx._tables[("ARM", None)]["t"][1] = 35.0
    iv.t1 = 35.0
    a = anchors_for(ctx)
    f = D.crash_check_anchor(ctx, a)[0]
    check(f.status == HIT and f.context == "crash_check" and f.values["angerr_deg"] == 151.0, f.values)
    # ECode 2 = 失去控制
    ctx2, iv2 = build_synth(**base, method=12, err=[(35.0, 12, 2)])
    iv2.t1 = 35.0
    f = D.crash_check_anchor(ctx2, anchors_for(ctx2))[0]
    check(f.status == HIT and f.context == "loss_of_control", f.context)
    # 沒 ERR:CLEAR;FS_CRASH_CHECK=0:NA
    ctx3, _ = build_synth(**base, method=0, params={"FS_CRASH_CHECK": 1.0})
    check(D.crash_check_anchor(ctx3, anchors_for(ctx3))[0].status == CLEAR, "沒有 ERR 應 CLEAR")
    ctx4, _ = build_synth(**base, method=0, params={"FS_CRASH_CHECK": 0.0})
    f = D.crash_check_anchor(ctx4, anchors_for(ctx4))[0]
    check(f.status == NA and f.context == "fs_crash_check_off", f.context)
    # 別的子系統的 ERR 不算
    ctx5, _ = build_synth(**base, method=0, err=[(20.0, 11, 2), (21.0, 16, 2)])
    check(D.crash_check_anchor(ctx5, anchors_for(ctx5))[0].status == CLEAR, "GPS / EKFCHECK 不是 CRASH_CHECK")
    # 解鎖區間之外的 CRASH_CHECK 不算這次
    ctx6, _ = build_synth(**base, method=0, err=[(3.0, 12, 1)])
    check(D.crash_check_anchor(ctx6, anchors_for(ctx6))[0].status == CLEAR, "區間外的 ERR 不算")
    print("  ECode 1 / 2、無 ERR、FS_CRASH_CHECK=0、其他子系統、區間外 → 各自正確")


def test_free_fall_synthetic():
    # 空中先失去重力(姿態還正常)0.12 s → HIT before_attitude_error
    ctx, _ = build_synth(freefall=[(20.0, 0.12)], ev=[(6.0, 28)])
    f = D.free_fall_run(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.context == "before_attitude_error" and f.severity == "high", (f.status, f.context))
    check(abs(f.values["dur_s"] - 0.12) < 0.03 and f.values["drop_m"] > 0.05, f.values)
    # 0.08 s 太短 → CLEAR
    ctx, _ = build_synth(freefall=[(20.0, 0.07)], ev=[(6.0, 28)])
    check(D.free_fall_run(ctx, anchors_for(ctx))[0].status == CLEAR, "0.07 s 不夠")
    # 翻覆途中的自由落體 → in_tumble(後果)
    ctx, _ = build_synth(roll=lambda t: crash_roll(t, 30.0, 1.0), freefall=[(31.2, 0.15)], ev=[(6.0, 28)])
    f = D.free_fall_run(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.context == "in_tumble" and f.severity == "medium", (f.context, f.severity))
    # 空洞切斷:兩段各 0.06 s,中間有 0.1 s 空洞 → 不能併成一段
    ctx, _ = build_synth(freefall=[(20.0, 0.06), (20.16, 0.06)], ev=[(6.0, 28)])
    for k in ("IMU",):
        for i in (0, 1):
            tb = ctx._tables[(k, i)]
            keep = ~((tb["t"] > 20.06) & (tb["t"] < 20.16))
            ctx._tables[(k, i)] = {c: v[keep] for c, v in tb.items()}
    check(D.free_fall_run(ctx, anchors_for(ctx))[0].status == CLEAR, "時間戳空洞要斷開")
    # 自由落體後接硬撞擊
    ctx, _ = build_synth(freefall=[(20.0, 0.2)], spikes=[(20.25, 0.1, 70.0, -1)], ev=[(6.0, 28)])
    f = D.free_fall_run(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.values["then_impact"] is True, f.values)
    # 缺 IMU → NA
    ctx, _ = build_synth(no_imu=True)
    check(D.free_fall_run(ctx, anchors_for(ctx))[0].status == NA, "沒 IMU 要 NA")
    print("  before_attitude_error / in_tumble / 太短 / 空洞 / 接撞擊 / NA")


def test_tumble_synthetic():
    ev = [(6.0, 28)]
    # 滾轉翻到倒扣(含跨 ±180° 的包繞)
    ctx, _ = build_synth(roll=lambda t: _ramp(t, 30, 32, 0, 200), ev=ev)
    f = D.tumble_inversion(ctx, anchors_for(ctx))[0]
    v = f.values
    check(f.status == HIT and f.context == "inverted" and f.severity == "critical", (f.status, f.context))
    check(abs(f.t_onset - (30 + 2 * 45 / 200)) < 0.05 and v["axis"] == "roll", (f.t_onset, v["axis"]))
    check(v["tilt_max_deg"] > 170 and v["accel_confirmed"] is True, v)
    # 俯仰翻
    ctx, _ = build_synth(pitch=lambda t: _ramp(t, 30, 32, 0, 100), ev=ev)
    f = D.tumble_inversion(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.values["axis"] == "pitch", (f.status, f.values.get("axis")))
    # 傾斜 60° 不倒扣 → tumble 不是 inverted
    ctx, _ = build_synth(roll=lambda t: _ramp(t, 30, 31, 0, 60), ev=ev)
    f = D.tumble_inversion(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.context == "tumble" and f.severity == "high", (f.context, f.severity))
    # 30° 側傾(激烈機動)→ CLEAR
    ctx, _ = build_synth(roll=lambda t: _ramp(t, 30, 31, 0, 35), ev=ev)
    check(D.tumble_inversion(ctx, anchors_for(ctx))[0].status == CLEAR, "35° 不是翻覆")
    # 尖峰 <0.3 s 不算
    ctx, _ = build_synth(roll=lambda t: np.where((t > 30) & (t < 30.15), 90.0, 0.0), ev=ev)
    check(D.tumble_inversion(ctx, anchors_for(ctx))[0].status == CLEAR, "0.15 s 尖峰不算翻覆")
    # 特技 / 翻滾模式 → NA(刻意)
    for mode in (1, 14, 13):
        ctx, _ = build_synth(roll=lambda t: _ramp(t, 30, 32, 0, 200), ev=ev, modes=[(0.0, 5), (20.0, mode)])
        f = D.tumble_inversion(ctx, anchors_for(ctx))[0]
        check(f.status == NA and f.context == "commanded_flip", (mode, f.status, f.context))
    # EV21 FLIP_START … EV22 + 3 s 視窗內的完整 360° 翻滾(翻完回正)不算
    ctx, _ = build_synth(roll=lambda t: _ramp(t, 30, 31, 0, 360), ev=ev + [(29.5, 21), (31.0, 22)])
    f = D.tumble_inversion(ctx, anchors_for(ctx))[0]
    check(f.status == NA and f.context == "commanded_flip", (f.status, f.context))
    # FLIP 失敗、翻完仍倒扣:視窗(EV22 + 3 s = 34 s)之後還倒扣 → 失控翻覆,起點 = 視窗結束
    ctx, _ = build_synth(roll=lambda t: _ramp(t, 30, 31, 0, 200), ev=ev + [(29.5, 21), (31.0, 22)])
    f = D.tumble_inversion(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.context == "inverted" and 34.0 <= f.t_onset <= 34.1, (f.status, f.context, f.t_onset))
    # 姿態說翻覆、加速度計說水平(姿態估計壞掉)→ SOFT
    ctx, _ = build_synth(roll=lambda t: _ramp(t, 30, 32, 0, 180), ev=ev)
    for i in (0, 1):
        tb = ctx._tables[("IMU", i)]
        tb["AccX"][:] = 0.0
        tb["AccY"][:] = 0.0
        tb["AccZ"][:] = -G
    f = D.tumble_inversion(ctx, anchors_for(ctx))[0]
    check(f.status == SOFT and f.values["accel_confirmed"] is False and f.context.endswith("attitude_only"), (f.status, f.context))
    # 沒有 IMU:仍可由姿態判定,accel_confirmed=None
    ctx, _ = build_synth(roll=lambda t: _ramp(t, 30, 32, 0, 180), ev=ev, no_imu=True)
    f = D.tumble_inversion(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.values["accel_confirmed"] is None, f.values)
    # 沒有 ANG 但有 ATT:用 ATT
    ctx, _ = build_synth(roll=lambda t: _ramp(t, 30, 32, 0, 180), ev=ev, no_att=True)
    check(D.tumble_inversion(ctx, anchors_for(ctx))[0].status == NA, "沒有姿態 → NA")
    ctx, _ = build_synth(roll=lambda t: _ramp(t, 30, 32, 0, 180), ev=ev, no_att=True)
    ang = ctx.table("IMU", 0)
    ctx._tables[("ATT", None)] = {"t": ang["t"][::40], "Roll": _ramp(ang["t"][::40], 30, 32, 0, 180), "Pitch": np.zeros(ang["t"][::40].size),
                                  "Yaw": np.zeros(ang["t"][::40].size), "DesRoll": np.zeros(ang["t"][::40].size),
                                  "DesPitch": np.zeros(ang["t"][::40].size), "DesYaw": np.zeros(ang["t"][::40].size)}
    f = D.tumble_inversion(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.values["attitude_src"] == "ATT", (f.status, f.values.get("attitude_src")))
    # 上鎖後的畫面不算:翻覆發生在上鎖之後 → CLEAR
    ctx, _ = build_synth(roll=lambda t: _ramp(t, 52, 54, 0, 180), arm=(5, 50), ev=ev)
    check(D.tumble_inversion(ctx, anchors_for(ctx))[0].status == CLEAR, "上鎖後的姿態不得納入")
    print("  倒扣(含 ±180° 包繞)/ 俯仰 / 傾斜 60° / 35° / 尖峰 / 特技與 FLIP / SOFT / 無 IMU / ATT 備援 / 上鎖後不算")


def test_vibe_synthetic():
    kw = dict(dur=60, arm=(5, 55), ev=[(6.0, 28)])
    # 乾淨
    ctx, _ = build_synth(**kw, vibe=make_vibe(60, 6.0))
    f = D.vibe_clip_transient(ctx, anchors_for(ctx))[0]
    check(f.status == CLEAR and f.values["vibe_max"] == 6.0 and f.values["type"] == "none", f.values)
    # Clip 一開始就是 2 且不增加(119 型)→ CLEAR
    ctx, _ = build_synth(**kw, vibe=make_vibe(60, 6.0))
    ctx._tables[("VIBE", 0)]["Clip"][:] = 2.0
    check(D.vibe_clip_transient(ctx, anchors_for(ctx))[0].status == CLEAR, "累計值不是增量")
    # 持續振動 10 s(>30),沒有撞擊 → sustained
    ctx, _ = build_synth(**kw, vibe=make_vibe(60, 6.0, bursts=[(20, 30, 45.0)]))
    f = D.vibe_clip_transient(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.context == "sustained_vibration" and f.severity == "high", (f.status, f.context))
    check(f.values["type"] == "sustained vibration" and abs(f.t_onset - 20.0) <= 0.15, (f.values["type"], f.t_onset))
    ctx, _ = build_synth(**kw, vibe=make_vibe(60, 6.0, bursts=[(20, 30, 70.0)]))
    check(D.vibe_clip_transient(ctx, anchors_for(ctx))[0].severity == "critical", ">=60 要 critical")
    # 30 s 內累計 ≥20%(每次 1.5 s 的短暴衝 × 5 次,各自 <5 s)→ sustained
    bursts = [(10 + 5 * k, 11.5 + 5 * k, 40.0) for k in range(6)]
    ctx, _ = build_synth(**kw, vibe=make_vibe(60, 6.0, bursts=bursts))
    f = D.vibe_clip_transient(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.context == "sustained_vibration", (f.status, f.context))
    # 沒撞擊的反覆削波(每次 2 顆 IMU 同時 —— 振動削波本來就常是多顆一起)→ vibration_clipping;1~2 次 → SOFT
    ctx, _ = build_synth(**kw, vibe=make_vibe(60, 8.0, clips=[(15, [3, 3, 0]), (25, [2, 2, 0]), (35, [1, 1, 0])]))
    fs = D.vibe_clip_transient(ctx, anchors_for(ctx))
    f = fs[0]
    check(len(fs) == 1 and f.status == HIT and f.context == "vibration_clipping", [(x.status, x.context) for x in fs])
    check(f.values["n_other_clips"] == 3 and f.values["n_multi_imu_clips"] == 3 and f.values["type"] == "vibration clipping", f.values)
    check("撞擊" not in f.text_zh.replace("沒有撞擊", ""), f"沒有撞擊時不得出現撞擊敘述:{f.text_zh}")
    check(f.values["stats_window"] == "armed", f.values["stats_window"])
    # 4 s VIBE 45 暴衝 + 每秒一次三顆 IMU 同時削波,沒有撞擊 → vibration_clipping(不是撞擊暫態,也不寫『撞擊前 p95』)
    ctx, _ = build_synth(**kw, vibe=make_vibe(60, 25.0, bursts=[(20, 24, 45.0)],
                                              clips=[(20.5, [1, 1, 1]), (21.5, [1, 1, 1]), (22.5, [1, 1, 1]), (23.5, [1, 1, 1])]))
    fs = D.vibe_clip_transient(ctx, anchors_for(ctx))
    check([x.context for x in fs] == ["vibration_clipping"] and fs[0].status == HIT, [(x.status, x.context) for x in fs])
    check("撞擊前" not in fs[0].text_zh and "撞擊造成" not in fs[0].text_zh, fs[0].text_zh)
    # 同上但接著翻覆墜機(CRASH_CHECK):削波在失事前 → 仍是 vibration_clipping(H_VIBRATION_IMU 的證據),不是撞擊暫態
    ctx, _ = build_synth(dur=60, arm=(5, 36.6), method=12, roll=lambda t: crash_roll(t, 30.0), spikes=[(32.0, 0.05, 70.0, +1)],
                         ev=[(6.0, 28)], err=[(36.6, 12, 1)], params={"FS_CRASH_CHECK": 1.0},
                         vibe=make_vibe(60, 25.0, bursts=[(20, 24, 45.0)],
                                        clips=[(20.5, [1, 1, 1]), (21.5, [1, 1, 1]), (22.5, [1, 1, 1]), (23.5, [1, 1, 1])]))
    a = anchors_for(ctx)
    fs = by_id(D.run(ctx, a), "vibe_clip_transient")
    check([x.context for x in fs] == ["vibration_clipping"] and fs[0].t_onset == 20.5, [(x.status, x.context, x.t_onset) for x in fs])
    check(a.end_type == "crash_check" and abs(a.t_first_hit - 32.0) < 0.1, (a.end_type, a.t_first_hit))
    ctx, _ = build_synth(**kw, vibe=make_vibe(60, 8.0, clips=[(15, [1, 0, 0]), (25, [1, 0, 0])]))
    f = D.vibe_clip_transient(ctx, anchors_for(ctx))[0]
    check(f.status == SOFT and f.context == "clip_increment", (f.status, f.context))
    ctx, _ = build_synth(**kw, vibe=make_vibe(60, 8.0, clips=[(15, [1, 0, 0]), (25, [1, 0, 0]), (35, [1, 0, 0]), (45, [1, 0, 0])]))
    f = D.vibe_clip_transient(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.context == "vibration_clipping" and f.values["n_other_clips"] == 4, (f.status, f.context))
    check(f.values["type"] == "vibration clipping", f.values["type"])     # H_VIBRATION_IMU 以此計分
    # 撞擊暫態:硬撞擊 + 隨後 1 s 的 VIBE 暴衝 + 三顆 IMU 同時 Clip → impact_transient;撞擊前乾淨
    ctx, _ = build_synth(**kw, spikes=[(30.0, 0.1, 80.0, -1)],
                         vibe=make_vibe(60, 6.0, bursts=[(30.1, 31.0, 55.0)], clips=[(30.3, [12, 12, 1])]))
    f = D.vibe_clip_transient(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.context == "impact_transient", (f.status, f.context))
    check(f.values["clip_before_first_hit"] == 0 and f.values["pre_impact_p95"] <= 6.5 and abs(f.t_onset - 30.1) < 0.3, f.values)
    check(f.values["type"] == "impact clipping", f.values["type"])
    check(f.values["impact_cls"] == "hard_impact" and abs(f.values["impact_t_peak"] - 30.0) < 0.01 and "30.00 s 的硬撞擊" in f.text_zh,
          (f.values.get("impact_t_peak"), f.text_zh))
    check(f.values["stats_window"] == "pre_impact", f.values["stats_window"])
    # 沒有任何 IMU 衝擊群,三顆 IMU 同一筆大量 Clip → 不能說是撞擊(沒有撞擊可對應)→ SOFT clip_increment,如實描述
    ctx, _ = build_synth(**kw, vibe=make_vibe(60, 6.0, clips=[(30.3, [12, 12, 1])]))
    f = D.vibe_clip_transient(ctx, anchors_for(ctx))[0]
    check(f.status == SOFT and f.context == "clip_increment" and "3 顆 IMU 同時" in f.text_zh and "附近沒有 IMU 衝擊群" in f.text_zh,
          (f.status, f.context, f.text_zh))
    # 多顆同時 Clip 在觸地衝擊(ground_contact,26 m/s²)±1 s 內 → 撞擊暫態(對應到觸地)
    ctx, _ = build_synth(**kw, spikes=[(30.0, 0.04, 26.0, -1)], vibe=make_vibe(60, 6.0, clips=[(30.2, [2, 2, 0])]))
    f = D.vibe_clip_transient(ctx, anchors_for(ctx))[0]
    check(f.context == "impact_transient" and f.values["impact_cls"] == "ground_contact" and "觸地衝擊" in f.text_zh, (f.context, f.text_zh))
    # 單顆 dc≥5 且在硬撞擊 ±1 s 內 → 撞擊暫態;遠離撞擊不算
    ctx, _ = build_synth(**kw, spikes=[(30.0, 0.1, 80.0, -1)], vibe=make_vibe(60, 6.0, clips=[(30.5, [6, 0, 0])]))
    check(D.vibe_clip_transient(ctx, anchors_for(ctx))[0].context == "impact_transient", "dc>=5 近撞擊")
    # 油門推滿的平台(sustained_accel)不是撞擊:同時的多顆 Clip 仍算削波
    ctx, _ = build_synth(**kw, spikes=[(30.0, 0.8, 25.0, -1)], vibe=make_vibe(60, 6.0, clips=[(30.3, [2, 2, 0])]))
    f = D.vibe_clip_transient(ctx, anchors_for(ctx))[0]
    check(f.status == SOFT and f.context == "clip_increment", (f.status, f.context))
    # 短暫超標、無撞擊 → SOFT short_burst
    ctx, _ = build_synth(**kw, vibe=make_vibe(60, 6.0, bursts=[(30, 31, 40.0)]))
    f = D.vibe_clip_transient(ctx, anchors_for(ctx))[0]
    check(f.status == SOFT and f.context == "short_burst", (f.status, f.context))
    # 缺 VIBE → NA
    ctx, _ = build_synth(**kw)
    check(D.vibe_clip_transient(ctx, anchors_for(ctx))[0].status == NA, "沒 VIBE 要 NA")
    print("  clean / 累計 Clip / 持續 / critical / 20% 視窗 / 削波 / 撞擊暫態 / 多顆同時 / 短暫 / NA")


def test_impact_findings_synthetic():
    kw = dict(dur=60, arm=(5, 55))
    # 觸地後 0.8 s 宣告 EV17 → CLEAR normal_touchdown
    ctx, _ = build_synth(**kw, spikes=[(40.0, 0.04, 26.0, -1)], ev=[(6.0, 28), (40.8, 17), (41.6, 18)])
    fs = D.impact_shock_clusters(ctx, anchors_for(ctx))
    check(len(fs) == 1 and fs[0].status == CLEAR and fs[0].context == "normal_touchdown", [(f.status, f.context) for f in fs])
    # 沒有宣告、且 5 s 內翻覆 → HIT ground_contact(『觸地後翻覆』的起點)
    ctx, _ = build_synth(**kw, spikes=[(40.0, 0.04, 26.0, -1)], ev=[(6.0, 28)], roll=lambda t: _ramp(t, 42.0, 43.0, 0.0, 180.0))
    fs = D.impact_shock_clusters(ctx, anchors_for(ctx))
    check(len(fs) == 1 and fs[0].status == HIT and fs[0].context == "ground_contact", [(f.status, f.context) for f in fs])
    check(fs[0].values["followed_by"] == "tumble" and 2.0 < fs[0].values["follow_dt_s"] < 3.0 and "翻覆』的起點" in fs[0].text_zh,
          (fs[0].values["followed_by"], fs[0].values["follow_dt_s"], fs[0].text_zh))
    # 上鎖當下才補寫的 EV17/EV18 不算宣告落地
    ctx, iv = build_synth(**kw, spikes=[(54.0, 0.04, 26.0, -1)], ev=[(6.0, 28), (55.0, 17), (55.0, 18)])
    fs = D.impact_shock_clusters(ctx, anchors_for(ctx))
    check(fs[0].status == SOFT and fs[0].context == "ground_contact_at_disarm", [(f.status, f.context) for f in fs])
    # 硬撞擊 → HIT critical;沒有 IMU 沒有 VIBE → NA
    ctx, _ = build_synth(**kw, spikes=[(40.0, 0.1, 70.0, -1)], ev=[(6.0, 28)])
    fs = D.impact_shock_clusters(ctx, anchors_for(ctx))
    check(len(fs) == 1 and fs[0].status == HIT and fs[0].severity == "critical" and fs[0].context == "hard_impact", [(f.status, f.context) for f in fs])
    ctx, _ = build_synth(**kw, no_imu=True)
    check(D.impact_shock_clusters(ctx, anchors_for(ctx))[0].status == NA, "沒資料要 NA")
    # 沒有 IMU 但 VIBE 三顆同時大量 Clip → SOFT vibe_fallback
    ctx, _ = build_synth(**kw, no_imu=True, vibe=make_vibe(60, 6.0, clips=[(40.0, [12, 12, 1])]))
    f = D.impact_shock_clusters(ctx, anchors_for(ctx))[0]
    check(f.status == SOFT and f.context == "vibe_fallback", (f.status, f.context))
    # 完全平靜 → CLEAR no_impact
    ctx, _ = build_synth(**kw)
    f = D.impact_shock_clusters(ctx, anchors_for(ctx))[0]
    check(f.status == CLEAR and f.context == "no_impact", f.context)
    print("  正常觸地 / 未宣告+翻覆 / 上鎖補寫不算 / 硬撞擊 / 缺資料 / VIBE 備援")


def test_ground_contact_wording():
    """審查缺陷:每個 2 s 內沒宣告落地的觸地都被說成『觸地後翻覆的起點』。RECOG_WIN_S=2.0 只有 n=1 的正常降落(0.82 s)當依據,
    之後沒有翻覆 / 硬撞擊就只能 SOFT、中性描述;較晚才宣告 → 『較晚宣告(x s)』。"""
    kw = dict(dur=60, arm=(5, 50), method=0)
    # 觸地重飛(touch-and-go):20 s 觸地、沒翻覆,45 s 正常降落 → SOFT 中性
    ctx, _ = build_synth(**kw, spikes=[(20.0, 0.04, 26.0, -1), (45.0, 0.04, 26.0, -1)], ev=[(6.0, 28), (45.8, 17), (46.6, 18)])
    a = anchors_for(ctx)
    fs = D.run(ctx, a)
    imp = by_id(fs, "impact_shock_clusters")
    check([(f.status, f.context) for f in imp] == [(SOFT, "ground_contact_unrecognised")], [(f.status, f.context) for f in imp])
    check("翻覆』的起點" not in imp[0].text_zh and "沒有翻覆或硬撞擊" in imp[0].text_zh, imp[0].text_zh)
    check(imp[0].values["recognised_late_dt"] is None, "45.8 s 的 EV17 屬於下一次觸地,不是這次的『較晚宣告』")
    check(a.end_type == "normal_landing" and a.t_first_hit is None and a.t_impact_peak is None, (a.end_type, a.t_impact_peak))
    # 慢的落地偵測(手動降落收油門慢):45.0 s 觸地,EV17 47.5 s、EV18 48.3 s,50 s 正常上鎖 → SOFT 『較晚宣告 2.5 s』
    ctx, _ = build_synth(**kw, spikes=[(45.0, 0.04, 26.0, -1)], ev=[(6.0, 28), (47.5, 17), (48.3, 18)])
    a = anchors_for(ctx)
    fs = D.run(ctx, a)
    f = one(fs, "impact_shock_clusters")
    check(f.status == SOFT and f.context == "ground_contact_late" and abs(f.values["recognised_late_dt"] - 2.5) < 0.01, (f.status, f.context, f.values))
    check("2.5 s 後才宣告落地" in f.text_zh and "翻覆』的起點" not in f.text_zh, f.text_zh)
    check(a.end_type == "normal_landing", a.end_type)
    # 未宣告且 3 s 後硬撞擊 → HIT(followed_by hard_impact)
    ctx, _ = build_synth(**kw, spikes=[(30.0, 0.04, 26.0, -1), (33.0, 0.08, 70.0, -1)], ev=[(6.0, 28)])
    fs = D.impact_shock_clusters(ctx, anchors_for(ctx))
    gc = [f for f in fs if f.values.get("cls") == "ground_contact"]
    check(len(gc) == 1 and gc[0].status == HIT and gc[0].values["followed_by"] == "hard_impact"
          and abs(gc[0].values["follow_dt_s"] - 3.0) < 0.05, [(f.status, f.context, f.values.get("followed_by")) for f in fs])
    # 翻覆在觸地 6 s 後(超過 TIPOVER_FOLLOW_S=5 s)→ 不算這次觸地造成的 → SOFT
    ctx, _ = build_synth(**kw, spikes=[(30.0, 0.04, 26.0, -1)], ev=[(6.0, 28)], roll=lambda t: _ramp(t, 36.0, 37.0, 0.0, 180.0))
    f = one(D.impact_shock_clusters(ctx, anchors_for(ctx)), "impact_shock_clusters")
    check(f.status == SOFT and f.context == "ground_contact_unrecognised", (f.status, f.context))
    print("  觸地重飛 → SOFT 中性;慢宣告 2.5 s → SOFT late;未宣告+硬撞擊 / 翻覆(≤5 s)→ HIT;翻覆在 6 s 後 → SOFT")


def _punch_case(lvl, w, ramp=0.0):
    """飛行中 20 s 起油門推滿 / 急拉(比力 lvl 持續 w 秒),45 s 正常降落(EV17/EV18)。"""
    ctx, _ = build_synth(dur=60, arm=(5, 50), method=0, spikes=[(20.0, w, lvl, -1), (45.0, 0.04, 26.0, -1)],
                         ev=[(6.0, 28), (45.8, 17), (46.6, 18)])
    if ramp:
        for i in (0, 1):
            tb = ctx._tables[("IMU", i)]
            m = (tb["t"] >= 20.0 - ramp) & (tb["t"] < 20.0)
            tb["AccZ"][m] = -(G + (lvl - G) * (tb["t"][m] - (20.0 - ramp)) / ramp)
    return ctx


def test_punch_out_not_impact():
    """審查缺陷:飛行中 >20 m/s² 的持續加速(Stabilize / AltHold 推滿油門、2 g 急拉、競速機 4.5 g)被當成重落地 / 撞擊,
    正常飛行的報告還印出『第一次硬撞擊』。修正:連續 >0.2 s 的平台不是撞擊脈衝(cls sustained_accel),只列在 values。"""
    for lvl, w, ramp in ((25.0, 0.8, 0.0), (22.0, 0.25, 0.0), (45.0, 0.4, 0.0), (30.0, 1.5, 0.08)):
        ctx = _punch_case(lvl, w, ramp)
        a = anchors_for(ctx)
        fs = D.run(ctx, a)
        check_all_ids(fs, f"punch {lvl}/{w}")
        imp = by_id(fs, "impact_shock_clusters")
        check(all(f.status == CLEAR for f in imp) and imp[0].context == "normal_touchdown", [(f.status, f.context) for f in imp])
        check(imp[0].values["n_sustained"] == 1 and "不是撞擊的短脈衝" in imp[0].text_zh, imp[0].text_zh)
        check([c["cls"] for c in imp[0].values["clusters"]] == ["sustained_accel", "ground_contact"], imp[0].values["clusters"])
        check(a.end_type == "normal_landing" and a.t_first_hit is None and a.t_impact_peak is None, (lvl, w, a.end_type, a.t_first_hit))
        check([c["cls"] for c in a.impacts] == ["ground_contact"], f"a.impacts 只放撞擊 / 觸地:{a.impacts}")
        check(len(a.extra["end_state"]["sustained_accel"]) == 1, a.extra["end_state"])
        check(all(f.status != HIT for f in fs if f.id != "abnormal_disarm_classifier"), [(f.id, f.status, f.context) for f in fs if f.status == HIT])
    # 正常降落的飛行即使有真的重落地(短脈衝),也不填撞擊錨點(報告不得出現『第一次硬撞擊』)
    ctx, _ = build_synth(dur=60, arm=(5, 50), method=0, spikes=[(30.0, 0.08, 70.0, -1), (45.0, 0.04, 26.0, -1)],
                         ev=[(6.0, 28), (45.8, 17), (46.6, 18)])
    a = anchors_for(ctx)
    fs = D.run(ctx, a)
    check(any(f.context == "hard_impact" and f.status == HIT for f in fs), "短脈衝 70 m/s² 仍是硬撞擊")
    check(a.end_type == "normal_landing" and a.t_first_hit is None and a.t_impact_peak is None, (a.end_type, a.t_first_hit, a.t_impact_peak))
    print("  25 g×0.8 s / 22×0.25 s / 45×0.4 s / 30×1.5 s(含爬升)→ sustained_accel,不出撞擊;正常降落不填撞擊錨點")


def test_impact_episode_anchors():
    """t_first_hit / t_impact_peak 只取最後一串硬撞擊(相鄰 ≤10 s):起飛時的重落地不能變成幾十秒後失事的『第一次撞擊』
    (engine 沒有 t_warn 時用 t_first_hit 當因果分界)。同時驗證 ae_at_impact(外力撞擊閘門用)。"""
    common = dict(dur=60, arm=(5, 46.6), method=12, roll=lambda t: crash_roll(t, 40.0), ev=[(6.0, 28)],
                  err=[(46.6, 12, 1)], params={"FS_CRASH_CHECK": 1.0})
    # (a) 10 s 的重落地 + 42 s 翻覆後的撞擊 → 錨點是 42 s 那一串
    ctx, _ = build_synth(**common, spikes=[(10.0, 0.1, 70.0, -1), (42.0, 0.05, 70.0, +1)])
    a = anchors_for(ctx)
    fs = D.run(ctx, a)
    check_all_ids(fs, "synth/episode")
    check(a.end_type == "crash_check", a.end_type)
    check(abs(a.t_first_hit - 42.0) < 0.05 and abs(a.t_impact_peak - 42.0) < 0.06, (a.t_first_hit, a.t_impact_peak))
    check([round(c["t_start"]) for c in a.impacts if c["cls"] == "hard_impact"] == [10, 42], a.impacts)   # 清單仍完整
    hard = [f for f in by_id(fs, "impact_shock_clusters") if f.context == "hard_impact"]
    check(len(hard) == 2 and hard[0].values["ae_at_impact"] < 10.0 and hard[1].values["ae_at_impact"] > 30.0,
          [(f.t_onset, f.values["ae_at_impact"]) for f in hard])
    check("撞擊在先" in hard[0].text_zh and "撞擊是結果" in hard[1].text_zh, [f.text_zh for f in hard])
    first_a = a.t_first_hit
    # (b) 38 s 姿態正常時被撞(外力)、42 s 翻覆後再撞:同一串 → t_first_hit = 38 s
    ctx, _ = build_synth(**common, spikes=[(38.0, 0.1, 70.0, -1), (42.0, 0.05, 70.0, +1)])
    a = anchors_for(ctx)
    D.run(ctx, a)
    check(abs(a.t_first_hit - 38.0) < 0.05 and abs(a.t_impact_peak - 42.0) < 0.06, (a.t_first_hit, a.t_impact_peak))
    # (c) 純函式:相隔正好超過 10 s 才切開
    mk = lambda t0: {"t_start": t0, "t_end": t0 + 0.2, "t_peak": t0 + 0.1, "t_hard": t0 + 0.05, "cls": "hard_impact"}   # noqa: E731
    check([c["t_start"] for c in D._last_episode([mk(1.0), mk(11.0), mk(30.0)])] == [30.0], "間隔 18.8 s 要切開")
    check([c["t_start"] for c in D._last_episode([mk(1.0), mk(11.0), mk(21.0)])] == [1.0, 11.0, 21.0], "間隔 9.8 s 要相連")
    check(D._last_episode([]) == [], "空清單")
    print(f"  10 s 重落地不搶錨點:first_hit {first_a};38/42 s 同一串:first_hit {a.t_first_hit};外力撞擊 ae<10 / 翻覆後撞擊 ae>30")


def test_abnormal_disarm_synthetic():
    def cls_of(**kw):
        ctx, iv = build_synth(**kw)
        fs = D.abnormal_disarm_classifier(ctx, anchors_for(ctx))
        return fs, ctx

    # 正常:EV28 → EV17 → EV18 → 上鎖(方向舵)
    fs, _ = cls_of(ev=[(6.0, 28), (47.0, 17), (47.8, 18)], method=0)
    check(fs[0].status == CLEAR and fs[0].values["class"] == "normal_landing", fs[0].values)
    # 空中手動上鎖:EV28、沒有 EV18、Method 0;高度 / 爬升率大 → critical
    alt = (np.arange(0, 60, 0.1), np.where(np.arange(0, 60, 0.1) < 40, 0.0, 6.0))
    crt = (np.arange(0, 60, 0.1), np.full(600, 120.0))
    fs, _ = cls_of(ev=[(6.0, 28)], method=0, series={"CTUN.Alt": alt, "CTUN.CRt": crt})
    check(fs[0].status == HIT and fs[0].values["class"] == "manual_disarm_in_air" and fs[0].severity == "critical", (fs[0].values, fs[0].severity))
    # 同上但低高度、低速度 → high
    alt2 = (np.arange(0, 60, 0.1), np.zeros(600))
    crt2 = (np.arange(0, 60, 0.1), np.zeros(600))
    fs, _ = cls_of(ev=[(6.0, 28)], method=0, series={"CTUN.Alt": alt2, "CTUN.CRt": crt2})
    check(fs[0].values["class"] == "manual_disarm_in_air" and fs[0].severity == "high", fs[0].severity)
    # 失效保護類 Method(5~11、14+)→ failsafe_disarm
    for m in (5, 7, 9, 14, 18):
        fs, _ = cls_of(ev=[(6.0, 28)], method=m)
        check(fs[0].values["class"] == "failsafe_disarm" and fs[0].status == HIT, (m, fs[0].values["class"]))
    # 墜機上鎖:Method 12;上鎖當下才補寫的 EV18/EV17(1 ms 內)不算落地
    t_d = 50.0
    fs, _ = cls_of(ev=[(6.0, 28), (t_d + 0.00006, 11), (t_d + 0.00007, 18), (t_d + 0.00008, 17)], method=12)
    check(fs[0].values["class"] == "crash_check" and fs[0].values["landed_before"] is False, fs[0].values)
    # Method 0 且 EV18 僅在上鎖後 6 µs 才補寫 → 仍是『空中上鎖』,不是正常降落
    fs, _ = cls_of(ev=[(6.0, 28), (t_d + 0.000006, 18), (t_d + 0.000015, 17)], method=0)
    check(fs[0].values["class"] == "manual_disarm_in_air", fs[0].values)
    # EV18 在上鎖前 1.9 s → 正常
    fs, _ = cls_of(ev=[(6.0, 28), (t_d - 1.9, 18)], method=0)
    check(fs[0].values["class"] == "normal_landing", fs[0].values)
    # Method 是正常類別但沒落地(馬達測試 3)→ SOFT normal_end,不當成異常
    fs, _ = cls_of(ev=[(6.0, 28)], method=3)
    check(fs[0].status == SOFT and fs[0].values["class"] == "normal_end", (fs[0].status, fs[0].values["class"]))
    # 完全沒有落地 / 離地事件 → NA(不能亂猜)
    fs, _ = cls_of(ev=[], method=0)
    check(fs[0].status == NA and fs[0].context == "no_landing_info", fs[0].context)
    # 接近 CRASH_CHECK 的 ERR(Method 看似正常)也判 crash_check
    fs, _ = cls_of(ev=[(6.0, 28)], method=0, err=[(50.0, 12, 1)])
    check(fs[0].values["class"] == "crash_check", fs[0].values["class"])
    # log 在解鎖中結束:電壓崩落 → critical;電壓正常 → high
    tv = np.arange(0, 60, 0.1)
    vcollapse = np.where(tv > 57.0, 8.0, 44.0)
    fs, _ = cls_of(ev=[(6.0, 28)], open_end=True, series={"BAT.Volt": (tv, vcollapse)})
    check(fs[0].values["class"] == "log_ended_armed" and fs[0].severity == "critical" and fs[0].values["volt_collapsed"] is True, (fs[0].values, fs[0].severity))
    fs, _ = cls_of(ev=[(6.0, 28)], open_end=True, series={"BAT.Volt": (tv, np.full(600, 44.0))})
    check(fs[0].severity == "high" and fs[0].values["volt_collapsed"] is False, fs[0].severity)
    # 結束時已落地:SOFT info
    fs, _ = cls_of(ev=[(6.0, 28), (30.0, 18)], open_end=True, series={"BAT.Volt": (tv, np.full(600, 44.0))})
    check(fs[0].status == SOFT and fs[0].severity == "info", (fs[0].status, fs[0].severity))
    # 空中緊急停止 EV54
    fs, _ = cls_of(ev=[(6.0, 28), (30.0, 54)], method=0)
    check(any(f.context == "emergency_stop_in_air" and f.status == HIT for f in fs), [f.context for f in fs])
    fs, _ = cls_of(ev=[(6.0, 28), (30.0, 18), (31.0, 54)], method=0)
    check(not any(f.context == "emergency_stop_in_air" for f in fs), "已落地的緊急停止不算空中")
    print("  正常 / 空中手動(critical, high)/ 失效保護 / 墜機 / EV 補寫順序 / normal_end / 無事件 NA / log 中斷 / EV54")


def test_end_type_priorities():
    ev_air = [(6.0, 28)]

    def run_case(**kw):
        ctx, iv = build_synth(**kw)
        a = anchors_for(ctx)
        fs = D.run(ctx, a)
        check_all_ids(fs, "synth")
        return a, fs

    # (1) CRASH_CHECK ERR → crash_check
    ctx, iv = build_synth(dur=60, arm=(5, 36.6), method=12, roll=lambda t: crash_roll(t, 30.0), spikes=[(32.0, 0.05, 70.0, +1)],
                          ev=ev_air, err=[(36.6, 12, 1)], params={"FS_CRASH_CHECK": 1.0})
    a = anchors_for(ctx)
    fs = D.run(ctx, a)
    check(a.end_type == "crash_check" and abs(a.t_detect - 36.6) < 1e-6, (a.end_type, a.t_detect))
    check(abs(a.t_impact_peak - 32.0) < 0.1 and abs(a.t_first_hit - 32.0) < 0.1 and a.t_rest is not None, (a.t_impact_peak, a.t_first_hit, a.t_rest))
    check(abs(a.t_rest - 32.05) < 0.2, a.t_rest)
    # (2) 沒有 ERR 但重算觸發(FS_CRASH_CHECK=0,飛手事後手動上鎖)→ crash_check_replay,優先於 abnormal_disarm
    a, fs = run_case(dur=60, arm=(5, 55), method=0, roll=lambda t: crash_roll(t, 30.0), spikes=[(32.0, 0.05, 70.0, +1)],
                     ev=ev_air, params={"FS_CRASH_CHECK": 0.0})
    check(a.end_type == "crash_check_replay" and a.t_detect is not None and 34.0 < a.t_detect < 34.8, (a.end_type, a.t_detect))
    check(one(fs, "abnormal_disarm_classifier").values["class"] == "manual_disarm_in_air", "事後手動上鎖")
    check(one(fs, "crash_check_anchor").status == NA, "FS_CRASH_CHECK=0 → 錨點 NA")
    check(a.t_rest is not None and abs(a.t_rest - 32.05) < 0.3, a.t_rest)
    # (1b) CRASH_CHECK 但沒有任何衝擊群(軟翻倒)→ t_impact_peak = t_detect − 2.6(CRASH_CHECK 2 s 靜止造成的延遲)
    ctx, iv = build_synth(dur=60, arm=(5, 36.6), method=12, roll=lambda t: crash_roll(t, 30.0), ev=ev_air, err=[(36.6, 12, 1)],
                          params={"FS_CRASH_CHECK": 1.0})
    a = anchors_for(ctx)
    D.run(ctx, a)
    check(a.end_type == "crash_check" and abs(a.t_impact_peak - (36.6 - D.IMPACT_LAG_TYPICAL_S)) < 0.01 and a.t_first_hit is None
          and a.extra["end_state"]["impact_source"] == "lag_estimate", (a.end_type, a.t_impact_peak, a.extra["end_state"]))
    # (3) 空中手動上鎖(姿態正常)→ abnormal_disarm;不得用 CRASH_CHECK 的 2.6 s 延遲倒推撞擊時間(審查缺陷)
    a, fs = run_case(dur=60, arm=(5, 50), method=0, ev=ev_air)
    check(a.end_type == "abnormal_disarm" and a.t_detect == a.t_disarm == 50.0, (a.end_type, a.t_detect))
    check(a.t_impact_peak is None and a.t_first_hit is None and a.extra["end_state"]["impact_source"] == "",
          f"空中上鎖沒有量到撞擊就不填:{a.t_impact_peak}")
    # 10 m 高空上鎖,1.5 s 後才撞地(在上鎖後 1 s 視窗之外)→ 仍不填;衝擊偵測如實說明上鎖後的墜落不在視窗內
    alt = (np.arange(0, 60, 0.1), np.where(np.arange(0, 60, 0.1) < 10, 0.0, 10.0))
    a, fs = run_case(dur=60, arm=(5, 40), method=0, spikes=[(41.5, 0.1, 80.0, -1)], ev=ev_air,
                     series={"CTUN.Alt": alt, "CTUN.CRt": (np.arange(0, 60, 0.1), np.zeros(600))})
    check(a.end_type == "abnormal_disarm" and a.t_impact_peak is None and a.t_first_hit is None, (a.end_type, a.t_impact_peak))
    f = one(fs, "impact_shock_clusters")
    check(f.status == CLEAR and "上鎖後的墜落與落地撞擊在分析視窗" in f.text_zh, f.text_zh)
    # 上鎖後 0.5 s 撞地(在 +1 s 視窗內)→ 用量到的撞擊
    a, fs = run_case(dur=60, arm=(5, 40), method=0, spikes=[(40.5, 0.1, 80.0, -1)], ev=ev_air)
    check(a.end_type == "abnormal_disarm" and abs(a.t_impact_peak - 40.5) < 0.05 and abs(a.t_first_hit - 40.5) < 0.05, (a.t_impact_peak, a.t_first_hit))
    a, fs = run_case(dur=60, arm=(5, 50), method=8, ev=ev_air)
    check(a.end_type == "abnormal_disarm" and a.t_impact_peak is None, (a.end_type, a.t_impact_peak))
    # (4) 沒有 ERR、重算沒觸發、已宣告落地,但上鎖前 3 s 有硬撞擊 → hard_impact
    a, fs = run_case(dur=60, arm=(5, 50), method=0, spikes=[(47.0, 0.1, 75.0, -1)], ev=ev_air + [(48.0, 18)])
    check(a.end_type == "hard_impact" and abs(a.t_impact_peak - 47.05) < 0.1 and a.t_first_hit is not None, (a.end_type, a.t_impact_peak))
    # 同樣的硬撞擊但在上鎖前 20 s(視窗外)→ 不算 hard_impact,正常降落不填撞擊錨點
    a, fs = run_case(dur=60, arm=(5, 50), method=0, spikes=[(30.0, 0.1, 75.0, -1)], ev=ev_air + [(48.0, 18)])
    check(a.end_type == "normal_landing" and a.t_impact_peak is None and a.t_first_hit is None, (a.end_type, a.t_impact_peak))
    # (5) log 在解鎖中結束:沒有衝擊 → 不填撞擊時間(不倒推 2.6 s)
    a, fs = run_case(dur=60, arm=(5, 60), open_end=True, ev=ev_air)
    check(a.end_type == "log_ended_armed" and a.t_detect == a.t_disarm and a.t_impact_peak is None, (a.end_type, a.t_detect, a.t_impact_peak))
    # (6) 倒扣 ≥1 s、沒有 ERR / 重算沒觸發(落地旗標擋掉)、正常類上鎖且已落地 → tumble
    a, fs = run_case(dur=60, arm=(5, 31.8), method=0, roll=lambda t: _ramp(t, 30, 30.5, 0, 180), ev=ev_air + [(31.5, 18)])
    check(a.end_type == "tumble" and 31.0 < a.t_detect < 32.0 and a.t_impact_peak is None, (a.end_type, a.t_detect, a.t_impact_peak))
    # 倒扣不到 1 s → 不算 tumble
    a, fs = run_case(dur=60, arm=(5, 30.9), method=0, roll=lambda t: _ramp(t, 30, 30.2, 0, 180), ev=ev_air + [(30.5, 18)])
    check(a.end_type == "normal_landing", a.end_type)
    # 正常降落
    a, fs = run_case(dur=60, arm=(5, 50), method=0, spikes=[(45.0, 0.04, 26.0, -1)], ev=ev_air + [(45.8, 17), (46.6, 18)])
    check(a.end_type == "normal_landing" and a.t_impact_peak is None and a.t_detect is None, (a.end_type, a.t_impact_peak))
    # 沒有任何落地事件 → normal_end(不亂猜)
    a, fs = run_case(dur=60, arm=(5, 50), method=0, ev=[])
    check(a.end_type == "normal_end", a.end_type)
    # 沒有解鎖
    ctx, iv = build_synth(dur=60)
    ctx.armed = []
    a = anchors_for(ctx)
    fs = D.run(ctx, a)
    check_all_ids(fs, "synth/not_armed")
    check(all(f.status == NA and f.context == "not_armed" for f in fs) and a.end_type == "not_armed", [f.status for f in fs])
    print("  優先序 crash_check > replay > abnormal_disarm > hard_impact > log_ended_armed > tumble > normal;未解鎖 not_armed")


def test_missing_data_never_raises():
    """空的 Ctx(有解鎖但幾乎沒資料)→ 每個 id 至少一筆、不丟例外、不回 CLEAR 假象(資料缺失的偵測器是 NA)。"""
    ctx, iv = build_synth(dur=60, arm=(5, 50), no_imu=True, no_att=True, arm_table=False)
    a = anchors_for(ctx)
    fs = D.run(ctx, a)
    check_all_ids(fs, "empty")
    for i in ("crash_check_replay", "impact_shock_clusters", "free_fall_run", "tumble_inversion", "vibe_clip_transient"):
        check(all(f.status == NA for f in by_id(fs, i)), f"{i} 缺資料應為 NA:{[f.status for f in by_id(fs, i)]}")
    # 偵測器內部丟例外 → 轉成 NA(guard)
    class Boom(SynthCtx):
        def ev_rows(self):
            raise RuntimeError("boom")
    c2 = Boom({}, [ArmedInterval(0, 1.0, 20.0, 0)], t_end=30.0)
    a2 = Anchors(interval=c2.armed[0], intervals=list(c2.armed))
    fs = D.run(c2, a2)
    check_all_ids(fs, "boom")
    print("  缺資料全部 NA;偵測器例外被 guard 轉成 NA")


def _truncate(ctx, key, t_stop, insts=(None,)):
    for i in insts:
        tb = ctx._tables[(key, i)]
        keep = tb["t"] < t_stop
        ctx._tables[(key, i)] = {c: v[keep] for c, v in tb.items()}


def test_coverage_pure():
    t = np.arange(0.0, 60.0, 0.0025)
    c = D.coverage(t, 5.0, 50.0)
    check(c["ok"] and c["frac"] > 0.999 and c["max_gap_s"] < 0.01, c)
    # 10 Hz 串流有 0.8 s 掉資料(124 ATT / VIBE 實測最大)→ 仍 ok
    tv = np.arange(0.0, 60.0, 0.1)
    tv = tv[~((tv > 30.0) & (tv < 30.8))]
    check(D.coverage(tv, 5.0, 50.0)["ok"], D.coverage(tv, 5.0, 50.0))
    # 2 s 空洞 → 不 ok;最後 10 s 缺 1.5 s(最大空洞剛好 1.5 s)→ 尾端覆蓋 <90% → 不 ok
    check(not D.coverage(t[(t < 20.0) | (t > 22.0)], 5.0, 50.0)["ok"], "2 s 空洞")
    c = D.coverage(t[t < 48.6], 5.0, 50.0)
    check(not c["ok"] and c["tail_frac"] < 0.9 and c["max_gap_s"] <= D.COVER_MAX_GAP_S and c["frac"] > 0.95, c)
    # 開頭沒資料(t0 早於第一筆 10 s)
    c = D.coverage(t, -10.0, 50.0)
    check(not c["ok"] and abs(c["max_gap_s"] - 10.0) < 0.01 and c["gap_at"] == -10.0, c)
    check(not D.coverage(None, 0.0, 10.0)["ok"] and not D.coverage(np.zeros(0), 0.0, 10.0)["ok"], "沒資料")
    # interp_valid:範圍外、長空洞內的點不可內插(np.interp 會沿用端點 / 畫直線)
    ts = np.r_[np.arange(0.0, 10.0, 0.1), np.arange(12.0, 20.0, 0.1)]
    tq = np.array([-0.5, 0.0, 5.05, 10.5, 11.95, 12.0, 19.85, 20.5])
    check(D.interp_valid(tq, ts).tolist() == [False, True, True, False, False, True, True, False], D.interp_valid(tq, ts).tolist())
    print("  全覆蓋 / 0.8 s 掉資料 ok;2 s 空洞、尾端缺 1.5 s、開頭缺 10 s → 不 ok;interp_valid 範圍外 / 長空洞 → False")


def test_partial_coverage_never_clear():
    """審查缺陷:IMU / ANG 只記到一半時,偵測器用有資料的那段回 CLEAR(還說『解鎖期間最大…』),engine 拿 CLEAR 去排除假說。
    修正:覆蓋不足 → NA 'partial_coverage'(說明涵蓋多少、空洞在哪);有資料段裡找到的證據照樣 HIT。"""
    crash = dict(dur=60, arm=(5, 50), method=0, roll=lambda t: crash_roll(t, 30.0), spikes=[(32.0, 0.1, 70.0, +1)],
                 freefall=[(31.5, 0.3)], ev=[(6.0, 28)])
    # IMU 在 20 s 停止記錄(失事在 30~32 s)
    ctx, _ = build_synth(**crash)
    _truncate(ctx, "IMU", 20.0, (0, 1))
    fs = D.run(ctx, anchors_for(ctx))
    check_all_ids(fs, "imu_partial")
    for i in ("impact_shock_clusters", "free_fall_run", "crash_check_replay"):
        r = by_id(fs, i)
        check(all(f.status == NA and f.context == "partial_coverage" for f in r), f"{i}:{[(f.status, f.context) for f in r]}")
        check("33%" in r[0].text_zh and "20.0 s" in r[0].text_zh and r[0].values["coverage"]["ok"] is False, r[0].text_zh)
    f = one(fs, "tumble_inversion")
    check(f.status == HIT and f.values["accel_confirmed"] is None, f"姿態完整 → 翻覆照樣 HIT;翻覆時沒有 IMU = 無法驗證(不是 SOFT):{f.status} {f.values.get('accel_confirmed')}")
    # ANG 在 20 s 停止(沒有 ATT)→ 翻覆 / 重算 NA
    ctx, _ = build_synth(dur=60, arm=(5, 50), method=0, roll=lambda t: crash_roll(t, 30.0), ev=[(6.0, 28)])
    _truncate(ctx, "ANG", 20.0)
    fs = D.run(ctx, anchors_for(ctx))
    for i in ("tumble_inversion", "crash_check_replay"):
        r = by_id(fs, i)
        check(all(f.status == NA and f.context == "partial_coverage" for f in r), f"{i}:{[(f.status, f.context) for f in r]}")
    check("姿態(ANG)" in one(fs, "tumble_inversion").text_zh, one(fs, "tumble_inversion").text_zh)
    # IMU 只缺最後 3 s(失事在 30~32 s,資料涵蓋)→ 證據照樣 HIT,文字附註覆蓋不足
    ctx, _ = build_synth(**crash)
    _truncate(ctx, "IMU", 47.0, (0, 1))
    fs = D.run(ctx, anchors_for(ctx))
    hard = [f for f in by_id(fs, "impact_shock_clusters") if f.status == HIT]
    check(hard and "注意" in hard[0].text_zh and hard[0].values["coverage"]["ok"] is False, [(f.status, f.context) for f in by_id(fs, "impact_shock_clusters")])
    check(one(fs, "free_fall_run").status == HIT, "有資料段的自由落體照樣 HIT")
    # 重算:姿態只到 31 s(翻到一半),IMU 繼續 → 不得沿用最後一筆姿態去算 31 s 之後的誤差
    ctx, _ = build_synth(dur=60, arm=(5, 50), method=0, roll=lambda t: crash_roll(t, 30.0), ev=[(6.0, 28)])
    _truncate(ctx, "ANG", 31.0)
    rp = D._replay(D._Sh(ctx, anchors_for(ctx)), ctx, ctx.armed[0])
    late = rp["t"] > 31.0 + D.ATT_INTERP_MAX_GAP_S
    check(np.all(np.isnan(rp["ae"][late])) and rp["t_fire"] is None and rp["att_valid_frac"] < 0.65, (rp["t_fire"], rp["att_valid_frac"]))
    # VIBE 只到 20 s → vibe NA(不是 CLEAR)
    ctx, _ = build_synth(dur=60, arm=(5, 50), ev=[(6.0, 28)], vibe=make_vibe(60, 6.0))
    _truncate(ctx, "VIBE", 20.0, (0, 1, 2))
    f = D.vibe_clip_transient(ctx, anchors_for(ctx))[0]
    check(f.status == NA and f.context == "partial_coverage" and "VIBE" in f.text_zh, (f.status, f.context))
    print("  IMU / ANG / VIBE 只記到一半 → NA partial_coverage;有資料段內的證據照樣 HIT;重算不沿用最後一筆姿態")


def test_crash_check_anchor_without_err():
    """審查缺陷:ERR 列遺失(log 掉資料)但 ARM.Method=12 / 'Crash: Disarming' 都在 → 錨點回 CLEAR,和上鎖分類矛盾。"""
    base = dict(dur=60, arm=(5, 36.6), roll=lambda t: crash_roll(t, 30.0), spikes=[(32.0, 0.05, 70.0, +1)], ev=[(6.0, 28)],
                params={"FS_CRASH_CHECK": 1.0})
    ctx, _ = build_synth(**base, method=12)
    a = anchors_for(ctx)
    fs = D.run(ctx, a)
    f = one(fs, "crash_check_anchor")
    check(f.status == HIT and f.context == "crash_check" and f.values["err_missing"] is True and f.values["ecode"] is None, (f.status, f.values))
    check("ARM.Method=12" in f.values["confirmed_by"] and abs(f.t_alarm - 36.6) < 1e-6 and "ERR" in f.text_zh, (f.values, f.text_zh))
    check(one(fs, "abnormal_disarm_classifier").values["class"] == "crash_check" and a.end_type == "crash_check"
          and abs(a.t_detect - 36.6) < 1e-6, (a.end_type, a.t_detect))
    # 沒有 ARM 表、也沒有 ERR,只有 'Crash: Disarming' 訊息 → 仍 HIT(來源 MSG)
    ctx, iv = build_synth(**base, method=0, arm_table=False)
    ctx.messages = [(36.6, "Crash: Disarming: AngErr=151>30, Accel=0.2<3.0")]
    f = D.crash_check_anchor(ctx, anchors_for(ctx))[0]
    check(f.status == HIT and f.values["confirmed_by"] == ["MSG"] and f.values["angerr_deg"] == 151.0, (f.status, f.values))
    print("  缺 ERR:ARM.Method=12 → HIT(err_missing);只有訊息 → HIT(MSG)")


def test_ground_only_and_landing_text():
    """審查缺陷:解鎖後沒起飛就上鎖(這次解鎖期間沒有 EV28)被寫成『正常降落(順序:即將落地 → 落地 → 上鎖)』。"""
    ctx, _ = build_synth(dur=60, arm=(5, 20), method=0, ev=[(40.0, 28)])
    a = anchors_for(ctx)
    fs = D.run(ctx, a)
    f = one(fs, "abnormal_disarm_classifier")
    check(f.status == CLEAR and f.context == "ground_only" and f.values["class"] == "ground_only", (f.status, f.context))
    check("沒有起飛" in f.text_zh and "即將落地" not in f.text_zh, f.text_zh)
    check(a.end_type == "normal_end" and a.t_impact_peak is None, a.end_type)
    # 正常降落但只有 EV18(沒有 EV17)→ 不印「即將落地 → 落地 → 上鎖」的順序句
    ctx, _ = build_synth(dur=60, arm=(5, 50), method=0, ev=[(6.0, 28), (48.1, 18)])
    f = D.abnormal_disarm_classifier(ctx, anchors_for(ctx))[0]
    check(f.context == "normal_landing" and "即將落地" not in f.text_zh and "1.9 s" in f.text_zh, f.text_zh)
    ctx, _ = build_synth(dur=60, arm=(5, 50), method=0, ev=[(6.0, 28), (47.2, 17), (48.1, 18)])
    f = D.abnormal_disarm_classifier(ctx, anchors_for(ctx))[0]
    check(f.context == "normal_landing" and "即將落地 → 落地 → 上鎖" in f.text_zh, f.text_zh)
    print("  沒起飛 → ground_only / normal_end;只有 EV18 → 不印順序句")


def test_run_without_anchors():
    """審查缺陷:run(ctx, None) 丟 AttributeError(模組合約:不丟例外)。"""
    ctx, _ = build_synth(dur=60, arm=(5, 50), ev=[(6.0, 28)])
    fs = D.run(ctx, None)
    check_all_ids(fs, "None anchors")
    check(all(f.status == NA and f.context == "not_armed" for f in fs), [(f.id, f.status, f.context) for f in fs])
    print("  run(ctx, None) → 7 個 id 全部 NA(not_armed)")


def test_json_and_determinism():
    ctx, a, fs, _ = crash_run("truth")
    ctx2 = crashkit.load_ctx(crashkit.CRASH_LOG)
    a2 = crashkit.truth_anchors_124(ctx2)
    fs2 = D.run(ctx2, a2)
    s1 = json.dumps([(f.id, f.status, f.severity, f.t_onset, f.t_alarm, f.context, f.values, f.text_zh) for f in fs], sort_keys=True, ensure_ascii=False)
    s2 = json.dumps([(f.id, f.status, f.severity, f.t_onset, f.t_alarm, f.context, f.values, f.text_zh) for f in fs2], sort_keys=True, ensure_ascii=False)
    check(s1 == s2, "相同輸入必須得到相同輸出(確定性)")
    check((a.end_type, a.t_detect, a.t_impact_peak, a.t_first_hit, a.t_rest) == (a2.end_type, a2.t_detect, a2.t_impact_peak, a2.t_first_hit, a2.t_rest),
          "錨點確定性")
    for f in fs:
        check(f.t_onset is not None or f.status in (CLEAR, NA), f"{f.id} HIT/SOFT 必須有 t_onset")
        if f.status in (HIT, SOFT) and f.t_onset is not None:
            check(f.t_onset == round(f.t_onset, 3), "時間應收斂到 ms")
    print(f"  {len(fs)} 筆 Finding:JSON 可序列化、重跑一致")


def test_performance():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    for i in (0, 1, 2):         # 表轉換由所有模組共用,不算在偵測器頭上
        ctx.table("IMU", i)
    ctx.table("ANG")
    ctx.table("VIBE", 0)
    worst = ("", 0.0)
    for mk in (crashkit.truth_anchors_124, crashkit.flight1_anchors_124):
        a = mk(ctx)
        for fn in D._DETECTOR_ORDER:
            sh = D._Sh(ctx, a)          # 每個偵測器自己付全部的 IMU 前處理(最壞情況)
            t = time.time()
            fn(ctx, a, sh)
            dt = time.time() - t
            if dt > worst[1]:
                worst = (fn.__name__, dt)
            check(dt < 1.5, f"{fn.__name__} 花了 {dt:.2f} s(>1.5 s)")
    print(f"  最慢:{worst[0]} {worst[1]:.2f} s(< 1.5 s)")


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
