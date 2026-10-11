"""馬達偵測模組測試(core/crash/det_motor.py):python tests/test_crash_motor.py 或 pytest tests/test_crash_motor.py

涵蓋:
  1. 00000124 飛行 2(墜機)+ 真相錨點 / 只有 basic_anchors:空中 CLEAR、地面怠速推力失衡 HIT(症狀),數字和研究階段實測一致;
  2. 00000124 飛行 1(正常降落)與飛行 2 的 1255 s 之前:沒有 HIT / SOFT;
  3. 6 個 baseline(從未解鎖):全部 NA 'not_armed'(強制套整段區間也只會 NA 'no_spin',不是「正常」);
  4. 合成資料:偵測器的核心邏輯(證明不是把 124 的數字寫死);
  5. 每個偵測器 id 在每種情境都至少回一筆;values 可 JSON 序列化;耗時 < 1.5 s。
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
from core.crash import det_motor as dm  # noqa: E402
from core.crash.anchors import basic_anchors  # noqa: E402
from core.crash.types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval  # noqa: E402

OWNED = ("motor_thrust_loss_check", "motor_thrust_imbalance", "yaw_imbalance_iterm")


# ======================================================================================
# 共用
# ======================================================================================

def by_id(fs):
    out = {}
    for f in fs:
        out.setdefault(f.id, []).append(f)
    return out


def pick(fs, det, context=None):
    c = [f for f in fs if f.id == det and (context is None or f.context == context)]
    assert c, f"{det}(context={context})沒有輸出:{[(f.id, f.status, f.context) for f in fs]}"
    return c[0]


def check_contract(fs, label):
    """每個 id 至少一筆、domain 正確、values 是嚴格 JSON(不得有 NaN / inf)、HIT/SOFT 的 t_onset ≤ t_alarm。"""
    ids = {f.id for f in fs}
    assert set(OWNED) <= ids, f"{label}:缺少 {set(OWNED) - ids}"
    for f in fs:
        assert f.domain == "motor" and f.id in OWNED, (label, f.id, f.domain)
        assert f.status in (HIT, SOFT, CLEAR, NA), f.status
        json.dumps(f.values, allow_nan=False)
        assert f.text_zh, (label, f.id, "text_zh 空")
        if f.status in (HIT, SOFT):
            assert f.t_onset is not None and f.t_alarm is not None and f.t_onset <= f.t_alarm + 1e-9, (label, f)
        if f.status in (CLEAR, NA):
            assert f.t_onset is None, (label, f.id, "CLEAR/NA 不該有 t_onset")
        if f.status == CLEAR:
            check_clear_margins(f, label)


def check_clear_margins(f, label):
    """CLEAR 不得和自己印出的餘裕矛盾:每個判準的量都要在門檻的「安全側」(評審意見:不能 CLEAR 卻印出 ≤ 0 的餘裕)。"""
    v, txt = f.values, f.text_zh
    if f.id == "motor_thrust_loss_check":
        assert "還有 -" not in txt and "還有 0 µs" not in txt, (label, txt)
        assert v["flag_bit0_or_msg"] == [], (label, v["flag_bit0_or_msg"])
        for k in ("a_longest_s", "full_thr_ctrl_desc_longest_s", "pre_event_sat_longest_s"):
            assert v[k] < dm.TL_SOFT_HOLD_S - 1e-9, (label, k, v[k])
        if v["max_thr_in_ctrl_desc"] is not None and v["max_thr_in_ctrl_desc"] >= dm.TL_THR_MIN:
            assert "受控下降中最高油門需求" not in txt, (label, txt)
        if v["max_throut"] >= dm.TL_THR_MIN:
            assert f"全程最高油門輸出 {v['max_throut']:.2f}(門檻" not in txt, (label, txt)
        for r in v["b_runs"]:
            assert r["dur"] < (dm.TL_HOLD_S if r["after"] else dm.TL_SOFT_HOLD_S) - 1e-9, (label, r)
    elif f.id == "yaw_imbalance_iterm":
        y = v["axes"].get("PIDY") or {}
        assert v["msg_hits"] == 0 and (y.get("lpf_max_ratio") or 0.0) <= dm.YI_RATIO, (label, y)
        assert (y.get("raw_max_ratio") or 0.0) < dm.YI_IMAX_EQ and y.get("over_s", 0.0) == 0.0, (label, y)


def statuses(fs):
    return [(f.id, f.context, f.status) for f in fs]


# ======================================================================================
# 合成資料(假 Ctx):證明邏輯不是寫死 124 的數字
# ======================================================================================

class FakeCtx:
    """最小 Ctx 替身:只實作 det_motor 用到的介面。"""

    def __init__(self, tables, params=None, messages=(), evs=(), errs=(), modes=None, steps=None, boom=False):
        self.tables = tables
        self.params = params or {}
        self.messages = list(messages)
        self.evs = list(evs)             # [(t, id)]
        self.errs = list(errs)           # [(t, subsys, ecode)]
        self.modes = modes or [(0.0, 5, "LOITER")]
        self.steps = steps               # landed_steps 階梯 [(t, landed)]
        self.boom = boom

    def table(self, name, instance=None):
        if self.boom:
            raise RuntimeError("boom")
        return self.tables.get(name)

    def param(self, name, default=None):
        return self.params.get(name, default)

    def has(self, key):
        return False

    def messages_between(self, a, b, pattern=None):
        import re
        rx = re.compile(pattern, re.I) if pattern else None
        return [(t, s) for t, s in self.messages if a <= t <= b and (rx is None or rx.search(s))]

    def ev_rows(self):
        return (np.array([e[0] for e in self.evs], dtype=float), np.array([e[1] for e in self.evs], dtype=int))

    def err_rows(self):
        return (np.array([e[0] for e in self.errs], dtype=float), np.array([e[1] for e in self.errs], dtype=int),
                np.array([e[2] for e in self.errs], dtype=int))

    def mode_rows(self):
        return (np.array([m[0] for m in self.modes], dtype=float), np.array([m[1] for m in self.modes], dtype=int),
                [m[2] for m in self.modes])

    def landed_steps(self, iv):
        return list(self.steps) if self.steps is not None else [(iv.t0, True)]

    def ground_alt(self, iv, win=3.0):
        ct = self.tables.get("CTUN")
        if ct is None or "Alt" not in ct:
            return None
        m = (ct["t"] >= iv.t0) & (ct["t"] <= iv.t0 + win)
        return float(np.nanmedian(ct["Alt"][m])) if m.any() else None


class Scene:
    """合成飛行:10 Hz 的 RCOU / MOTB / CTUN / ATT、400 Hz 的 ANG / PIDx。起飛 takeoff 秒、落地 land 秒(落地後 landed)。

    CTUN.Alt:地面 0 m、空中 10 m;CTUN.ThI 預設不寫(模組退回 MOTB.ThrOut),設 sc.thi 才寫。
    """

    def __init__(self, n_motors=6, dur=60.0, takeoff=2.0, land=57.0, dt=0.1):
        self.n = n_motors
        self.dur = dur
        self.dt = dt
        self.t = np.arange(0.0, dur, dt)
        self.t4 = np.arange(0.0, dur, 1 / 400.0)
        n = self.t.size
        air = (self.t >= takeoff) & (self.t < land)
        base = np.where(air, 1480.0, 1235.0)
        self.pwm = np.tile(base[:, None], (1, n_motors))
        self.throut = np.where(air, 0.15, 0.0)
        self.thi = None                                # CTUN.ThI(韌體 get_throttle_in());None = 不寫這欄
        self.tho = self.throut.copy()
        self.thh = np.full(n, 0.15)
        self.crt = np.zeros(n)
        self.alt = np.where(air, 10.0, 0.0)
        self.flags = np.full(n, 2.0)
        self.batv = np.full(n, 0.9)
        self.des_roll = np.zeros(n)
        self.des_pitch = np.zeros(n)
        self.roll4 = np.zeros(self.t4.size)           # ANG.Roll(DesRoll = 0 → ae = |roll|)
        self.pid = {k: np.zeros(self.t4.size) for k in ("PIDY", "PIDR", "PIDP")}
        self.steps = [(0.0, True), (takeoff, False), (land, True)]
        self.takeoff, self.land = takeoff, land
        self.messages, self.evs, self.errs, self.modes = [], [], [], None
        self.drop_tables: set = set()
        self.drop_cols: dict = {}

    def w(self, a, b):                                # 10 Hz 取樣遮罩 a ≤ t < b
        return (self.t >= a) & (self.t < b)

    def ang(self, a, b, deg):                         # 400 Hz 滾轉誤差
        self.roll4[(self.t4 >= a) & (self.t4 < b)] = deg

    def ctx(self, **kw):
        n_ch = self.n
        rc = {"t": self.t}
        for k in range(n_ch):
            rc[f"C{k + 1}"] = self.pwm[:, k]
        z4 = np.zeros(self.t4.size)
        # ATT 10 Hz 的 DesRoll / Roll 與 ANG 一致(Roll 取樣自 400 Hz)
        roll10 = np.interp(self.t, self.t4, self.roll4)
        ctun = {"t": self.t, "ThO": self.tho, "ThH": self.thh, "CRt": self.crt, "Alt": self.alt}
        if self.thi is not None:
            ctun["ThI"] = self.thi
        tabs = {
            "RCOU": rc,
            "MOTB": {"t": self.t, "ThrOut": self.throut, "BatVolt": self.batv, "FailFlags": self.flags},
            "CTUN": ctun,
            "ATT": {"t": self.t, "DesRoll": self.des_roll, "Roll": roll10, "DesPitch": self.des_pitch,
                    "Pitch": np.zeros(self.t.size), "DesYaw": np.zeros(self.t.size), "Yaw": np.zeros(self.t.size)},
            "ANG": {"t": self.t4, "DesRoll": z4, "Roll": self.roll4, "DesPitch": z4, "Pitch": z4, "DesYaw": z4, "Yaw": z4},
        }
        for k, v in self.pid.items():
            tabs[k] = {"t": self.t4, "I": v}
        for name in self.drop_tables:
            tabs.pop(name, None)
        for name, cols in self.drop_cols.items():
            for c in cols:
                tabs[name].pop(c, None)
        params = {"MOT_PWM_MIN": 1100.0, "MOT_PWM_MAX": 2000.0, "MOT_SPIN_MIN": 0.15, "MOT_SPIN_MAX": 0.95,
                  "MOT_THST_EXPO": 0.65, "ATC_RAT_YAW_IMAX": 0.5, "ATC_RAT_RLL_IMAX": 0.5, "ATC_RAT_PIT_IMAX": 0.5}
        for k in range(n_ch):
            params[f"SERVO{k + 1}_FUNCTION"] = 33.0 + k
        return FakeCtx(tabs, params, self.messages, self.evs, self.errs, self.modes, self.steps, **kw)

    def anchors(self, **kw):
        """kw:t_touch / t_warn / t_loss 等錨點(預設都沒有,模擬只有 basic_anchors 的情形)。"""
        iv = ArmedInterval(0, 0.0, self.dur - 0.1, 0)
        a = Anchors(interval=iv, intervals=[iv])
        a.end_type = "unknown"
        for k, v in kw.items():
            setattr(a, k, v)
        return a


def run_scene(sc, anchors=None, **kw):
    """跑合成情境;每個情境都檢查模組契約(三個 id 都有輸出、嚴格 JSON、時間順序)。"""
    fs = dm.run(sc.ctx(**kw), anchors if anchors is not None else sc.anchors())
    check_contract(fs, "synthetic")
    return fs


# ======================================================================================
# 1~3 真實 log
# ======================================================================================

def test_real_crash_truth_anchors():
    """124 飛行 2:空中 CLEAR(馬達健康);怠速推力失衡 HIT 但 context=ground_idle(翻覆症狀)。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    fs = dm.run(ctx, crashkit.truth_anchors_124(ctx))
    check_contract(fs, "124 truth")
    tl = pick(fs, "motor_thrust_loss_check")
    assert tl.status == CLEAR and tl.context == "air", tl
    v = tl.values
    assert abs(v["max_throut"] - 0.444) < 0.01, v["max_throut"]
    assert abs(v["max_throut_t"] - 1258.5) < 0.15, v["max_throut_t"]
    assert v["pre_event_max_pwm"] == 1566, v["pre_event_max_pwm"]
    assert v["pwm_headroom"] >= 385, v["pwm_headroom"]
    assert v["a_longest_s"] == 0.0 and v["b_longest_s"] <= 0.1 + 1e-9, (v["a_longest_s"], v["b_longest_s"])
    assert v["b_sat_samples"] == 1 and v["flag_bit0_or_msg"] == []
    assert v["esc_telemetry"] is False and "ESC" in tl.text_zh
    # 韌體 (a) 用 CTUN.ThI:受控下降(ae < 30°)中最高油門需求只有 ~0.24(門檻 0.9);唯一的滿載取樣 1259.44 在失控(1258.14)之後
    assert v["a_throttle_src"] == "CTUN.ThI" and v["max_thr_in_ctrl_desc"] < 0.3, v["max_thr_in_ctrl_desc"]
    assert v["b_runs"] == [{"dur": 0.1, "t": 1259.44, "sat_on": 1259.44, "ae_on": 1258.14, "after": True}], v["b_runs"]
    for s in ("1566", "0.44", "0.9"):                                    # 驗收 A7 的排除數字
        assert s in tl.text_zh, (s, tl.text_zh)
    imb_air = pick(fs, "motor_thrust_imbalance", "air")
    assert imb_air.status == CLEAR, imb_air
    assert imb_air.values["source"] == "MOTB.FailFlags" and imb_air.values["bad_air_s"] == 0.0
    assert 1.25 < imb_air.values["air_ratio_p99"] < 1.45 and imb_air.values["air_ratio_max"] < 1.5
    assert imb_air.values["flag_vs_recomputed_pct"] > 99.5, imb_air.values["flag_vs_recomputed_pct"]
    gi = pick(fs, "motor_thrust_imbalance", "ground_idle")
    assert gi.status == HIT and gi.values["symptom_of"] == "ground_tipover", gi
    assert abs(gi.t_onset - 1256.34) < 0.15, gi.t_onset                  # FailFlags 2→0 在 1256.3
    assert abs(gi.t_alarm - 1258.34) < 0.15, gi.t_alarm                  # 研究階段:8 s 窗累計 2.0 s 於 1258.34
    assert gi.t_onset > crashkit.truth_anchors_124(ctx).t_touch          # 症狀晚於觸地(不是起點)
    assert 5.0 < gi.values["bad_idle_s"] < 6.5, gi.values["bad_idle_s"]
    assert gi.values["source"] == "MOTB.FailFlags" and "FailFlags" in gi.text_zh
    assert gi.values["near_ground_src"] == "agl+t_touch" and gi.values["peak_ratio_src"] == "recomputed"
    assert -0.5 < gi.values["agl_at_onset_m"] < 0.5 and "重算峰值比 2.0" in gi.text_zh, (gi.values["agl_at_onset_m"], gi.text_zh)
    assert gi.values["peak_ratio"] >= 1.9 and 4.0 <= gi.values["severe_s"] <= 5.0, gi.values
    assert gi.values["onset_high_channels"] == [2, 3, 6], gi.values["onset_high_channels"]   # 左側 C2/C3/C6 先衝高
    assert not any(f.id == "motor_thrust_imbalance" and f.context == "air" and f.status in (HIT, SOFT) for f in fs)
    ya = pick(fs, "yaw_imbalance_iterm")
    assert ya.status == CLEAR and ya.values["yaw_raw_max_ratio"] <= 0.07 + 1e-4, ya.values["yaw_raw_max_ratio"]
    assert ya.values["axes"]["PIDR"]["raw_max_ratio"] < 0.2 and ya.values["axes"]["PIDP"]["raw_max_ratio"] < 0.2
    print(f"  124 墜機: 推力損失 CLEAR(ThrOut 最大 {v['max_throut']}、事件前 PWM {v['pre_event_max_pwm']}、餘裕 {v['pwm_headroom']})")
    print(f"  推力失衡: air CLEAR(p99 {imb_air.values['air_ratio_p99']})+ ground_idle HIT 起點 {gi.t_onset:.2f} s 共 {gi.values['bad_idle_s']} s")
    print(f"  偏航積分: PIDY {ya.values['yaw_raw_max_ratio'] * 100:.1f}% IMAX;PIDR {ya.values['axes']['PIDR']['raw_max_ratio'] * 100:.0f}% PIDP {ya.values['axes']['PIDP']['raw_max_ratio'] * 100:.0f}%")


def test_real_crash_basic_anchors():
    """只有 basic_anchors(沒有 t_touch / t_warn):退回本地 ae>10° 算事件前窗,結論不變。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = basic_anchors(ctx)
    fs = dm.run(ctx, a)
    check_contract(fs, "124 basic")
    tl = pick(fs, "motor_thrust_loss_check")
    assert tl.status == CLEAR and tl.values["pre_event_end_src"] == "local_t_warn", tl.values
    assert abs(tl.values["pre_event_end"] - 1255.9) < 0.2, tl.values["pre_event_end"]
    assert tl.values["pre_event_max_pwm"] == 1566
    assert pick(fs, "motor_thrust_imbalance", "air").status == CLEAR
    gi = pick(fs, "motor_thrust_imbalance", "ground_idle")         # 沒有觸地錨點:靠 CTUN.Alt 判貼地(翻覆全程離地 ≤ 0.2 m)
    assert gi.status == HIT and gi.values["near_ground_src"] == "agl", gi.values["near_ground_src"]
    assert not any(f.context == "idle_air" for f in fs), statuses(fs)
    assert pick(fs, "yaw_imbalance_iterm").status == CLEAR
    print(f"  basic_anchors: 本地 t_warn {tl.values['pre_event_end']:.2f} s,結論同真相錨點(貼地由 CTUN.Alt 判定)")


def test_real_flight1_normal_landing():
    """飛行 1(正常降落對照):全部 CLEAR(空中),沒有 ground_idle 失衡,事件前最大 PWM 1525。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    fs = dm.run(ctx, crashkit.flight1_anchors_124(ctx))
    check_contract(fs, "124 flight1")
    assert statuses(fs) == [("motor_thrust_loss_check", "air", CLEAR), ("motor_thrust_imbalance", "air", CLEAR),
                            ("yaw_imbalance_iterm", "air", CLEAR)], statuses(fs)
    tl = pick(fs, "motor_thrust_loss_check")
    assert tl.values["pre_event_max_pwm"] == 1525 and abs(tl.values["max_throut"] - 0.238) < 0.01, tl.values
    imb = pick(fs, "motor_thrust_imbalance")
    assert imb.values["bad_air_s"] == 0.0 and imb.values["max_cum_8s"] == 0.0
    # 原始 FailFlags 在降落時確實掉到 0 約 0.9 s(616.9–617.8):是「已落地 / 怠速閘門」把它濾掉,不是資料裡沒有
    mb = ctx.table("MOTB")
    iv = ctx.armed[0]
    raw0 = (mb["t"] >= iv.t0) & (mb["t"] <= iv.t1) & ((mb["FailFlags"].astype(np.int64) & 2) == 0)
    assert 0.5 <= raw0.sum() * 0.1 <= 1.5 and 616.5 < mb["t"][raw0][0] < 617.5, (raw0.sum(), mb["t"][raw0][:3])
    ya = pick(fs, "yaw_imbalance_iterm")
    assert abs(ya.values["yaw_raw_max_ratio"] - 0.0704) < 0.005, ya.values["yaw_raw_max_ratio"]       # 研究階段:7%
    assert ya.values["axes"]["PIDR"]["raw_max_ratio"] < 0.2 and ya.values["axes"]["PIDP"]["raw_max_ratio"] < 0.2
    print(f"  飛行 1: 全 CLEAR;PWM 最大 {tl.values['max_pwm']}、重算比值 p99 {imb.values['air_ratio_p99']}、PIDY {ya.values['yaw_raw_max_ratio'] * 100:.1f}%"
          f";降落時原始 FailFlags=0 {raw0.sum() * 0.1:.1f} s 被閘門濾掉")


def test_real_flight2_before_touchdown():
    """飛行 2 在 1255 s 之前(墜機前 376 s):同樣的偵測器沒有 HIT / SOFT(驗收 B 最後一句)。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    iv = ctx.armed[1]
    sub = ArmedInterval(1, iv.t0, 1255.0, 0)
    a = Anchors(interval=sub, intervals=list(ctx.armed))
    a.end_type = "unknown"
    fs = dm.run(ctx, a)
    check_contract(fs, "124 flight2 pre-1255")
    assert all(f.status == CLEAR and f.context == "air" for f in fs), statuses(fs)
    tl = pick(fs, "motor_thrust_loss_check")
    assert tl.values["pre_event_max_pwm"] == 1566 and abs(tl.values["max_throut"] - 0.253) < 0.005, tl.values
    print(f"  飛行 2(<1255 s): 全 CLEAR;max PWM {tl.values['max_pwm']}、ThrOut {tl.values['max_throut']}")


def test_real_baselines_not_armed():
    """6 個從未解鎖的 baseline:只有 NA 'not_armed'(不能是 CLEAR —— 馬達從未轉動不是「正常」)。"""
    for nm in crashkit.BASELINES:
        ctx = crashkit.load_ctx(nm)
        fs = dm.run(ctx, crashkit.not_armed_anchors(ctx))
        check_contract(fs, nm)
        assert len(fs) == 3 and all(f.status == NA and f.context == "not_armed" for f in fs), (nm, statuses(fs))
        # 驗收 C:印「不適用 / 未解鎖」,絕不能是馬達「正常 / OK」
        assert all("未解鎖" in f.text_zh and "不適用" in f.text_zh for f in fs)
        assert not any("正常" in f.text_zh or "OK" in f.text_zh for f in fs)
    print(f"  {len(crashkit.BASELINES)} 個 baseline 全部 NA not_armed")


def test_real_baselines_forced_interval_is_na_no_spin():
    """硬把整段 log 當成解鎖區間(模擬閘門被拿掉):馬達從未轉動 → NA 'no_spin',仍不會誤報也不會說「正常」。"""
    for nm in crashkit.BASELINES:
        ctx = crashkit.load_ctx(nm)
        iv = ArmedInterval(0, 0.0, ctx.t_end)
        a = Anchors(interval=iv, intervals=[iv])
        fs = dm.run(ctx, a)
        check_contract(fs, nm + " forced")
        assert all(f.status == NA and f.context == "no_spin" for f in fs), (nm, statuses(fs))
    print("  強制區間: baseline 全部 NA no_spin(RCOU 全 1100)")


def test_real_timing():
    """每個偵測器 < 1.5 s(冷快取:清掉 det_motor 前處理快取,資料表快取保留 = 引擎實際行為)。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.truth_anchors_124(ctx)
    for fn in (dm.motor_thrust_loss_check, dm.motor_thrust_imbalance, dm.yaw_imbalance_iterm):
        ctx._motor_prep = {}
        t0 = time.perf_counter()
        fn(ctx, a)
        dt = time.perf_counter() - t0
        assert dt < 1.5, f"{fn.__name__} {dt:.2f}s"
        print(f"  {fn.__name__}: {dt * 1000:.0f} ms")
    # 30 分鐘合成飛行(400 Hz PIDx 各 72 萬點):偏航低通是逐點迴圈,仍要 < 1.5 s
    sc = Scene(dur=1800.0, land=1797.0)
    ctx = sc.ctx()
    for fn in (dm.motor_thrust_loss_check, dm.motor_thrust_imbalance, dm.yaw_imbalance_iterm):
        ctx._motor_prep = {}
        t0 = time.perf_counter()
        fn(ctx, sc.anchors())
        dt = time.perf_counter() - t0
        assert dt < 1.5, f"30-min {fn.__name__} {dt:.2f}s"
        print(f"  30 分鐘合成 {fn.__name__}: {dt * 1000:.0f} ms")


# ======================================================================================
# 4 合成:數值工具
# ======================================================================================

def test_lpf1_matches_loop():
    rng = np.random.default_rng(3)
    for a, n in ((0.1815, 1000), (7.85e-4, 20000), (0.5, 17)):
        x = rng.normal(size=n)
        ref = np.empty(n)
        prev = 0.0
        for i in range(n):
            prev = prev + a * (x[i] - prev)
            ref[i] = prev
        got = dm._lpf1(x, a)
        assert np.max(np.abs(got - ref)) < 1e-9, (a, np.max(np.abs(got - ref)))
    x2 = rng.normal(size=(500, 6)) + 3.0
    ref = np.empty_like(x2)
    prev = x2[0].copy()
    for i in range(500):
        prev = prev + 0.1815 * (x2[i] - prev)
        ref[i] = prev
    assert np.max(np.abs(dm._lpf1(x2, 0.1815, y0=x2[0]) - ref)) < 1e-9
    print("  單極低通(分塊向量化)和逐點迴圈一致(1-D / 2-D,a=0.18 / 7.85e-4)")


def test_lpf_yaw_fw_matches_reference():
    """韌體 yaw_I_filt 重現 vs 逐點參考:gate 斷開歸零、|y| ≤ |x| 上限(重設為帶號的 x)、NaN 保持前值。"""
    rng = np.random.default_rng(7)
    n = 5000
    x = np.cumsum(rng.normal(0, 0.01, n)) + 0.2 * np.sin(np.arange(n) / 300.0)
    x[1000:1010] = np.nan
    gate = np.ones(n, bool)
    gate[2000:2100] = False
    gate[4000] = False
    alpha = 7.85e-4 * 20
    ref = np.zeros(n)
    prev = 0.0
    for k in range(n):
        if not gate[k]:
            prev = 0.0
            ref[k] = 0.0
            continue
        v = x[k]
        if not np.isnan(v):
            prev = prev + alpha * (v - prev)
            if abs(prev) > abs(v):
                prev = v
        ref[k] = prev
    got = dm._lpf_yaw_fw(x, gate, alpha)
    assert np.max(np.abs(got - ref)) < 1e-12, np.max(np.abs(got - ref))
    ok = gate & np.isfinite(x)
    assert np.all(np.abs(got[ok]) <= np.abs(x[ok]) + 1e-15)
    print("  韌體偏航低通(歸零 / 上限 / NaN)和逐點參考一致")


def test_latch_runs_cumwindow():
    ratio = np.array([1.0, 1.4, 1.6, 1.4, 1.3, 1.2, 1.4, 1.5, 1.3, 1.26, 1.25, 1.4])
    ref, s = [], False
    for r in ratio:
        if r >= 1.5:
            s = True
        elif r <= 1.25:
            s = False
        ref.append(s)
    assert list(dm._latch(ratio)) == ref
    t = np.arange(0, 3.0, 0.1)
    m = np.zeros(t.size, bool)
    m[3:6] = True
    m[10:12] = True
    m[20] = True
    assert dm._runs_v(m, t) == [(3, 5), (10, 11), (20, 20)]
    t2 = t.copy()
    t2[6:] += 1.0                                   # 資料空洞:第 5→6 之間斷 1 s
    m2 = np.zeros(t.size, bool)
    m2[4:8] = True
    assert dm._runs_v(m2, t2, max_gap=0.25) == [(4, 5), (6, 7)] and dm._runs_v(m2, t2) == [(4, 7)]
    dur, t0, i0 = dm._longest_run(m, t, 0.1)
    assert abs(dur - 0.3) < 1e-9 and abs(t0 - 0.3) < 1e-9 and i0 == 3
    tt = np.arange(0, 30, 0.1)
    mk = np.random.default_rng(1).random(tt.size) < 0.2
    got = dm._cum_window(mk, tt, 0.1, 8.0)
    for k in (0, 50, 79, 80, 150, 299):
        lo = tt[k] - 8.0
        exp = np.sum(mk[(tt > lo) & (tt <= tt[k])]) * 0.1
        assert abs(got[k] - exp) < 1e-9, (k, got[k], exp)
    print("  遲滯閂鎖 / 連續區段 / 空洞斷開 / 8 s 窗累計 都和暴力算法一致")


def test_thrust_ratio_math():
    n = 400
    act = np.full((n, 6), 0.4)
    ratio, f = dm._thrust_ratio(act, np.full(n, 0.9), 0.65, 0.1)
    assert np.allclose(ratio, 1.0)
    act2 = act.copy()
    act2[:, [1, 2, 5]] = 0.8
    act2[:, [0, 3, 4]] = 0.0
    ratio2, f2 = dm._thrust_ratio(act2, np.full(n, 0.9), 0.65, 0.1)
    assert abs(ratio2[-1] - 2.0) < 1e-3, ratio2[-1]                      # 三顆全推、三顆零 → max·N/Σ = 2
    assert np.argmax(f2[-1]) in (1, 2, 5)
    a_ = 1 - (1 - 1 / (1 + 0.5 * 400)) ** 40
    assert abs(a_ - 0.1815) < 1e-3                                       # 規格的每 0.1 s alpha
    print(f"  推力比值:全等 → 1.00;三顆全推 → {ratio2[-1]:.3f}(理論 2.0)")


# ======================================================================================
# 4 合成:推力損失
# ======================================================================================

def _fw_scene(n_motors=6, thi=0.95, des=(5.0, 0.0), ae=5.0, crt=-300.0, pwm=1700.0, a=20.0, b=23.0, **kw):
    """韌體 (a) 的教科書情境:[a, b) 期望傾角小、姿態仍受控(ae 小)、油門需求全開、卻仍在下降。"""
    sc = Scene(n_motors=n_motors, **kw)
    s = sc.w(a, b)
    sc.thi = sc.throut.copy()
    sc.thi[s] = thi
    sc.throut[s] = max(thi, 0.15)
    sc.pwm[s] = pwm                                  # 1700 → act ≈ 0.65:沒有馬達飽和 → 只靠 (a)
    sc.crt[s] = crt
    sc.des_roll[s], sc.des_pitch[s] = des
    sc.ang(a - 0.05, b + 0.05, ae)
    return sc


def test_loss_firmware_style_synthetic():
    """(a) 照 ArduCopter thrust_loss_check():姿態仍受控(ae < 30°)、期望傾角 ≤ 15°、油門需求滿、仍下降 1 s → HIT。

    評審重現:原本規則要求 ae > 30°(和韌體相反),這種「教科書式推力損失」會被判 CLEAR、還印出 −5 µs 的餘裕。
    """
    f = pick(run_scene(_fw_scene()), "motor_thrust_loss_check")
    assert f.status == HIT and f.context == "air" and f.values["sub_a"] and not f.values["sub_b"], f
    assert abs(f.t_onset - 20.0) < 0.11 and abs(f.t_alarm - f.t_onset - dm.TL_HOLD_S) < 1e-6, f
    assert f.values["a_longest_s"] >= 2.9 and f.values["a_throttle_src"] == "CTUN.ThI" and f.values["after_attitude_loss"] is False
    # 姿態已失控(ae 40° ≥ 30°):韌體不算推力損失(那是姿態失控,由 attitude 模組處理)→ 不是 HIT
    f = pick(run_scene(_fw_scene(ae=40.0)), "motor_thrust_loss_check")
    assert f.status != HIT and f.values["a_longest_s"] == 0.0, f
    # 期望傾角用 hypot:(12°, 12°) → 17° > 15° → (a) 不成立(舊的 max(|DesRoll|,|DesPitch|) = 12° 會誤放行);
    # 但「受控下降中油門已滿」3 s → SOFT(接近),不是 CLEAR
    f = pick(run_scene(_fw_scene(des=(12.0, 12.0))), "motor_thrust_loss_check")
    assert f.status == SOFT and f.values["a_longest_s"] == 0.0 and f.values["full_thr_ctrl_desc_longest_s"] >= 2.9, f
    assert pick(run_scene(_fw_scene(des=(12.0, 0.0))), "motor_thrust_loss_check").status == HIT
    # 沒有下降(CRt ≥ 0)→ CLEAR(全油門爬升是正常的)
    f = pick(run_scene(_fw_scene(crt=50.0)), "motor_thrust_loss_check")
    assert f.status == CLEAR and "全程最高油門輸出 0.95" in f.text_zh, f
    # 油門需求 0.7 < 0.9 且沒有馬達到上限 → CLEAR,且印出的油門餘裕為正
    f = pick(run_scene(_fw_scene(thi=0.7)), "motor_thrust_loss_check")
    assert f.status == CLEAR and f.values["max_thr_in_ctrl_desc"] == 0.7 and "受控下降中最高油門需求 0.70(門檻 0.9)" in f.text_zh, f
    # 「已達上限」分支:油門需求 0.5 但有一顆馬達已頂到最大(act ≥ 0.99)→ HIT
    sc = _fw_scene(thi=0.5)
    sc.pwm[sc.w(20.0, 23.0), 2] = 1960.0
    assert pick(run_scene(sc), "motor_thrust_loss_check").status == HIT
    # 油門需求 < 0.25(韌體:低油門的推力限制不算)→ (a) 不成立;馬達滿載 3 s 仍會被「餘裕已用盡」列為 SOFT,不是 CLEAR
    sc = _fw_scene(thi=0.2)
    sc.pwm[sc.w(20.0, 23.0), 2] = 1960.0
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == SOFT and f.values["a_longest_s"] == 0.0 and f.values["pre_event_sat_longest_s"] >= 2.9, f
    # 已落地(land_complete)→ 韌體歸零 → (a) 不成立
    f = pick(run_scene(_fw_scene(land=19.0)), "motor_thrust_loss_check")
    assert f.status != HIT and f.values["a_longest_s"] == 0.0, f
    # 沒有 CTUN.ThI → 退回 MOTB.ThrOut(註記)
    sc = _fw_scene()
    sc.thi = None
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == HIT and f.values["a_throttle_src"] == "MOTB.ThrOut" and any("ThI" in n for n in f.values["notes"]), f
    # 四軸同樣成立(不是寫死六軸)
    f = pick(run_scene(_fw_scene(n_motors=4, pwm=1850.0, ae=8.0, crt=-200.0)), "motor_thrust_loss_check")
    assert f.status == HIT and f.values["n_motors"] == 4, f
    print("  (a) 韌體式:受控下降 + 油門滿 → HIT;ae≥30 / hypot 傾角>15 / 不下降 / 油門<0.9 / ThI<0.25 / 落地 各自擋掉;上限分支、ThrOut 備援、四軸皆可")


def test_loss_reviewer_p3_controlled_thrust_loss():
    """評審 P3:六軸、DesRoll 2°、姿態 5°、ThrOut 0.95、CRt −200 cm/s 6 s、一顆馬達 1960、沒有旗標 / 訊息 → 必須 HIT(原本 CLEAR)。"""
    sc = Scene(n_motors=6)
    s = sc.w(20.0, 26.0)
    sc.throut[s] = 0.95
    sc.tho[s] = 0.95
    sc.pwm[s] = 1850.0
    sc.pwm[np.ix_(s, [0])] = 1960.0
    sc.crt[s] = -200.0
    sc.des_roll[s] = 2.0
    sc.ang(20.0, 26.0, 5.0)
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == HIT and f.values["sub_a"] and abs(f.t_onset - 20.0) < 0.11 and f.values["a_longest_s"] >= 5.9, f
    assert "還有 -" not in f.text_zh
    print(f"  評審 P3(受控下沉、全油門):HIT 起點 {f.t_onset:.2f} s、(a) 連續 {f.values['a_longest_s']} s")


def _sat_scene(dur_s, lead=0.3, ae=50.0, n_motors=4, ch=2, pwm=1960.0, mode=None):
    """(b):馬達在 20.1 − lead 起滿載,姿態誤差 15°(< 30°)從 20.05 − lead 起、ae 從 20.05 起才超過 30°,維持 dur_s。

    lead > 0 = 馬達先滿載、姿態後失控(推力損失的型態);lead < 0 = 姿態先失控、馬達才滿載(翻覆的後果)。
    """
    sc = Scene(n_motors=n_motors)
    sc.pwm[sc.w(20.1 - max(lead, 0.0), 20.1 + dur_s - 0.05), ch] = pwm
    if lead > 0:
        sc.ang(20.05 - lead, 20.05, 15.0)
    sc.ang(20.05 + min(lead, 0.0), 20.05 + dur_s, ae)
    if mode:
        sc.modes = mode
    return sc


def test_loss_saturation_synthetic():
    f = pick(run_scene(_sat_scene(1.5)), "motor_thrust_loss_check")
    assert f.status == HIT and f.values["sub_b"] and abs(f.values["b_longest_s"] - 1.5) < 1e-6, f
    assert abs(f.t_onset - 19.8) < 0.11 and abs(f.t_alarm - 21.1) < 0.11, f      # t_onset = 滿載起點(早於失控)
    assert f.values["after_attitude_loss"] is False and f.context == "air"
    f = pick(run_scene(_sat_scene(0.7)), "motor_thrust_loss_check")
    assert f.status == SOFT and f.severity == "high" and f.context == "air", f
    f = pick(run_scene(_sat_scene(0.3)), "motor_thrust_loss_check")
    assert f.status == CLEAR, f                      # 墜機 124 的情形:單個 0.1 s 取樣
    # 刻意動作(ACRO)中的姿態誤差沒有意義 → 不算
    f = pick(run_scene(_sat_scene(1.5, mode=[(0.0, 5, "LOITER"), (19.0, 1, "ACRO"), (30.0, 5, "LOITER")])), "motor_thrust_loss_check")
    assert f.status == CLEAR, f
    # 同樣的飽和但姿態誤差 < 30° → 不算
    assert pick(run_scene(_sat_scene(1.5, ae=20.0)), "motor_thrust_loss_check").status == CLEAR
    print("  (b) 先滿載後失控:1.5 s HIT(起點 = 滿載起點 19.8)/ 0.7 s SOFT / 0.3 s CLEAR;ACRO 與小角度不算")


def test_loss_saturation_after_attitude_loss_is_consequence():
    """(b) 只在姿態失控之後才出現(翻覆中控制器把一側推滿):SOFT、context='loss_of_control'、after_attitude_loss,
    文字說明這是後果、不證明推力損失,也不叫人去查槳(評審 P4)。"""
    # 同一格開始(10 Hz 分不出先後)→ 不能當成「先滿載」
    f = pick(run_scene(_sat_scene(1.5, lead=0.0)), "motor_thrust_loss_check")
    assert f.status == SOFT and f.context == "loss_of_control" and f.values["after_attitude_loss"] is True, f
    # 姿態先失控 0.5 s、馬達才滿載
    f = pick(run_scene(_sat_scene(1.5, lead=-0.5)), "motor_thrust_loss_check")
    assert f.status == SOFT and f.context == "loss_of_control" and f.severity == "medium", f
    assert "後果" in f.text_zh and "不能證明推力損失" in f.text_zh and "請檢查槳葉" not in f.text_zh, f.text_zh
    assert f.values["b_runs"][0]["ae_on"] < f.values["b_runs"][0]["sat_on"]
    # 滿載起點比本地 ae>30° 早,但已晚於錨點 a.t_loss(姿態模組算出的失控起點)→ 仍是後果
    sc = _sat_scene(1.5, lead=0.3)
    f = pick(run_scene(sc, anchors=sc.anchors(t_warn=19.0, t_loss=19.5)), "motor_thrust_loss_check")
    assert f.status == SOFT and f.context == "loss_of_control", f
    # 失控後的短暫滿載(< 1 s)不升級:CLEAR,且印出的「最長 0.7 s」仍低於 1.0 s 門檻
    f = pick(run_scene(_sat_scene(0.7, lead=-0.5)), "motor_thrust_loss_check")
    assert f.status == CLEAR and "滿載且姿態誤差>30° 最長 0.7 s" in f.text_zh, f
    print("  (b) 失控後才滿載:SOFT loss_of_control(後果,不證明推力損失);錨點 t_loss 之後也算後果")


def _p4_scene():
    """評審 P4:地面翻覆(怠速、未判定落地),32 s 起 ae 60°,33.0–34.6 s 一顆馬達 1955(翻覆中控制器把一側推滿)。"""
    sc = Scene(n_motors=6, dur=40.0, takeoff=2.0, land=39.5)
    sc.steps = [(0.0, True), (2.0, False)]
    idle = sc.w(30.0, 39.0)
    sc.tho[idle] = 0.0
    sc.throut[idle] = 0.077
    sc.pwm[idle] = 1235.0
    sc.alt[sc.w(29.9, 40.0)] = 0.0
    sc.ang(32.0, 39.0, 60.0)
    sc.pwm[np.ix_(sc.w(33.0, 34.6), [3])] = 1955.0
    return sc


def test_loss_reviewer_p4_ground_tipover_gates():
    """評審 P4 + 假說閘門:失控後的滿載不能讓 H_MOTOR 閘門打開,也不能讓 H_LAND 被「同時有推力損失證據」扣分。"""
    from core.crash import engine
    from core.crash import hypotheses as hyp
    sc = _p4_scene()
    a = sc.anchors(t_touch=29.9, t_warn=31.9, t_loss=32.0)
    fs = run_scene(sc, anchors=a)
    f = pick(fs, "motor_thrust_loss_check")
    assert f.status == SOFT and f.context == "loss_of_control" and f.values["after_attitude_loss"] is True, f
    assert f.t_onset >= a.t_loss, f
    for g in fs:
        g.tag = engine.tag_time(g.t_onset, a)
    assert f.tag == "consequence", f.tag
    c2 = hyp.Ctx2(fs, a)
    assert c2.hit("motor_thrust_loss_check") is False
    assert hyp._gate_motor(c2)[0] is False
    print(f"  評審 P4:SOFT loss_of_control [{f.tag}];H_MOTOR 閘門關閉、c.hit(thrust_loss) = False")


def test_loss_flag_message_err_synthetic():
    """(c) 韌體旗標 / 訊息 / ERR 25:韌體條件持續 1 s 才發出 → t_alarm = 發出時間、t_onset = t_alarm − 1 s(不早於解鎖)。"""
    sc = Scene()
    sc.flags[sc.w(30.0, 30.2)] = 3.0                # bit0 thrust boost(bit1 仍為平衡)
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == HIT and "FailFlags.bit0" in f.values["flag_bit0_or_msg"], f
    assert abs(f.t_alarm - 30.0) < 0.11 and abs(f.t_onset - 29.0) < 0.11 and f.values["sub_c"], f
    sc = Scene()
    sc.messages = [(25.0, "Potential Thrust Loss (3)")]
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == HIT and f.t_alarm == 25.0 and f.t_onset == 24.0 and f.values["flag_bit0_or_msg"] == ["MSG"], f
    sc.messages = [(0.4, "Potential Thrust Loss (1)")]             # 起點推估不早於解鎖
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == HIT and f.t_onset == 0.0 and f.t_alarm == 0.4, f
    sc.messages = [(75.0, "Potential Thrust Loss (3)")]            # 區間外(解鎖前 / 上鎖後)不算
    assert pick(run_scene(sc), "motor_thrust_loss_check").status == CLEAR
    sc = Scene()
    sc.errs = [(40.0, 25, 1)]
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == HIT and f.t_alarm == 40.0 and f.t_onset == 39.0 and f.values["flag_bit0_or_msg"] == ["ERR25"], f
    sc.errs = [(40.0, 25, 0), (41.0, 12, 1)]                       # 恢復碼 / 別的子系統不算
    assert pick(run_scene(sc), "motor_thrust_loss_check").status == CLEAR
    print("  (c) FailFlags bit0 / 'Potential Thrust Loss' / ERR 25 各自 HIT,起點 = 發出時間 − 1 s;區間外與恢復碼不算")


def test_loss_na_paths_synthetic():
    # 沒有姿態資料:旗標檢查無異常也不能下 CLEAR
    sc = Scene()
    sc.drop_tables = {"ANG", "ATT"}
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == NA and f.context == "no_attitude", f
    # 沒有姿態資料但有韌體訊息 → 仍 HIT
    sc.messages = [(10.0, "Potential Thrust Loss (2)")]
    assert pick(run_scene(sc), "motor_thrust_loss_check").status == HIT
    # 馬達有轉但幾乎沒飛(只有地面怠速):不能說「沒有推力問題」
    sc = Scene(dur=20.0, takeoff=19.0, land=19.5)
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == NA and f.context == "no_air_data", f
    # 沒有 RCOU
    sc = Scene()
    sc.drop_tables = {"RCOU"}
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == NA and f.context == "no_rcou", f
    # 馬達從未轉
    sc = Scene()
    sc.pwm[:] = 1100.0
    for d in run_scene(sc):
        assert d.status == NA and d.context == "no_spin", d
    # 缺 CTUN(沒有 CRt):韌體條件 (a) 無法評估 → 不能 CLEAR(其餘檢查的數字仍附上)
    sc = Scene()
    sc.drop_tables = {"CTUN"}
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == NA and f.context == "partial_data" and f.values["a_evaluated"] is False and "1480" in f.text_zh, f
    # 錨點本身是 None → NA not_armed(不是例外)
    fs = dm.run(Scene().ctx(), None)
    assert [f.context for f in fs] == ["not_armed"] * 3, statuses(fs)
    print("  NA 路徑:無姿態 / 沒飛過 / 無 RCOU / 馬達從未轉 / 缺 CTUN(部分資料)/ 錨點 None")


def test_loss_takeoff_flip_context():
    """起飛瞬間(尚未判定離地)馬達先滿載、機身才翻倒 ≥ 1 s:HIT 且 context='ground'(這正是 (b) 不看 ~L 的理由)。"""
    sc = Scene(takeoff=10.0)
    sc.pwm[sc.w(5.0, 7.0), 0] = 1960.0
    sc.ang(4.95, 5.25, 15.0)                         # 滿載時姿態才開始偏(< 30°)
    sc.ang(5.25, 7.05, 60.0)
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == HIT and f.context == "ground" and f.values["onset_landed"] is True and abs(f.t_onset - 5.0) < 0.11, f
    print("  起飛翻覆(L=True,先滿載後翻):HIT,context=ground")


def test_loss_clear_never_prints_nonpositive_margin():
    """事件前已有馬達頂到滿載指令:短暫(0.1 s)→ CLEAR 但不印負的 PWM 餘裕;持續 ≥ 0.5 s → SOFT(餘裕已用盡)。"""
    sc = Scene()
    sc.pwm[sc.w(30.0, 30.1), 1] = 1960.0
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == CLEAR and f.values["pwm_headroom"] < 0 and "還有 -" not in f.text_zh, f
    assert "事件前曾有馬達到滿載指令 PWM 1960" in f.text_zh, f.text_zh
    sc = Scene()
    sc.pwm[sc.w(30.0, 31.0), 1] = 1960.0
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == SOFT and f.severity == "medium" and abs(f.t_onset - 30.0) < 0.11, f
    assert f.values["pre_event_sat_longest_s"] >= 0.9, f.values
    print("  CLEAR 不印 ≤ 0 的餘裕;事件前滿載 ≥ 0.5 s → SOFT")


def test_degenerate_inputs_are_explicit_na():
    """評審 P6:重複時間戳 / MOT_PWM_MAX == MIN / MOT_SPIN_MIN = None 等 → 明確的 NA('bad_params' / 'insufficient_data'),不是例外或 CLEAR。"""
    sc = Scene()
    ctx = sc.ctx()
    ctx.tables["RCOU"]["t"] = np.repeat(sc.t[::2], 2)[: sc.t.size]           # 兩兩重複的時間戳:取正的間隔(0.2 s)
    fs = dm.run(ctx, sc.anchors())
    check_contract(fs, "dup timestamps")
    assert not any(f.context == "error" for f in fs), statuses(fs)
    ctx = sc.ctx()
    ctx.tables["RCOU"]["t"] = np.full(sc.t.size, 5.0)                       # 全部同一時間
    fs = dm.run(ctx, sc.anchors())
    check_contract(fs, "all same t")
    assert all(f.status == NA and f.context == "insufficient_data" for f in fs), statuses(fs)
    for patch in ({"MOT_PWM_MAX": 1100.0}, {"MOT_SPIN_MIN": None}, {"MOT_SPIN_MIN": 0.95}, {"MOT_THST_EXPO": float("nan")},
                  {"MOT_PWM_MIN": "abc"}):
        ctx = sc.ctx()
        ctx.params.update(patch)
        with np.errstate(all="raise"):                                       # 不得有除以零之類的警告
            fs = dm.run(ctx, sc.anchors())
        check_contract(fs, f"bad params {patch}")
        assert all(f.status == NA and f.context == "bad_params" for f in fs), (patch, statuses(fs))
        assert all("參數無效" in f.text_zh for f in fs)
    print("  重複時間戳照常判定;全部同一時間 → NA insufficient_data;PWM / SPIN / EXPO 參數無效 → NA bad_params")


# ======================================================================================
# 4 合成:推力失衡
# ======================================================================================

def test_imbalance_air_synthetic():
    def run_w(a, b):
        sc = Scene()
        sc.flags[sc.w(a, b)] = 0.0
        fs = run_scene(sc)
        return fs, pick(fs, "motor_thrust_imbalance", "air")
    fs, f = run_w(20.0, 23.0)
    assert f.status == HIT and abs(f.t_onset - 20.0) < 0.11 and abs(f.t_alarm - 21.9) < 0.15 and f.severity == "high", f
    assert len(by_id(fs)["motor_thrust_imbalance"]) == 1, "空中(ThO=0.15)的失衡不該再多一筆 ground_idle"
    fs, f = run_w(20.0, 21.5)
    assert f.status == SOFT and f.severity == "medium", f
    fs, f = run_w(20.0, 20.6)
    assert f.status == CLEAR, f
    # 三個 0.8 s 突發落在同一個 8 s 窗 → 累計 2.4 s → HIT;各隔 10 s → 每窗只有 0.8 s → CLEAR
    sc = Scene()
    for a in (20.0, 22.0, 24.0):
        sc.flags[sc.w(a, a + 0.8)] = 0.0
    assert pick(run_scene(sc), "motor_thrust_imbalance", "air").status == HIT
    sc = Scene()
    for a in (10.0, 20.0, 30.0):
        sc.flags[sc.w(a, a + 0.8)] = 0.0
    assert pick(run_scene(sc), "motor_thrust_imbalance", "air").status == CLEAR
    print("  空中失衡: 3.0 s HIT(起點 20.0)/ 1.5 s SOFT / 0.6 s CLEAR;累計規則(8 s 窗)正確")


def test_imbalance_ground_idle_never_satisfies_air():
    """CTUN.ThO ≤ 0.03(怠速)且貼地時的失衡:只回 context='ground_idle' 的 HIT,空中那筆仍是 CLEAR。"""
    sc = Scene()
    sc.tho[sc.w(15.0, 35.0)] = 0.0
    sc.alt[sc.w(15.0, 35.0)] = 0.2                  # 貼地(離地 0.2 m)
    sc.flags[sc.w(20.0, 26.0)] = 0.0
    fs = run_scene(sc)
    check_contract(fs, "ground_idle synthetic")
    air = pick(fs, "motor_thrust_imbalance", "air")
    gi = pick(fs, "motor_thrust_imbalance", "ground_idle")
    assert air.status == CLEAR and air.values["bad_air_s"] == 0.0, air
    assert gi.status == HIT and abs(gi.t_onset - 20.0) < 0.11 and abs(gi.values["bad_idle_s"] - 6.0) < 0.11, gi
    # 翻覆時 ThO 零星冒出幾個 > 0.03 的突波(124 的實況)不會把症狀改判成空中
    sc.tho[sc.w(22.0, 22.1)] = 0.4
    sc.tho[sc.w(24.0, 24.1)] = 0.2
    fs = run_scene(sc)
    assert pick(fs, "motor_thrust_imbalance", "air").status == CLEAR
    assert pick(fs, "motor_thrust_imbalance", "ground_idle").status == HIT
    print("  地面怠速的失衡不會滿足空中偵測(含零星 ThO 突波)")


def test_imbalance_idle_in_air_is_not_ground_tipover():
    """評審 P5:空中(離地 10 m)收油門 + 失衡 → context='idle_air'、中性文字,不說是地面翻覆;
    旗標判失衡但重算比值 1.0 時,文字不印「峰值比 1.0」當成失衡的證據。"""
    sc = Scene()
    sc.tho[sc.w(30.0, 36.0)] = 0.0
    sc.flags[sc.w(30.5, 35.0)] = 0.0
    fs = run_scene(sc)
    assert [f.context for f in fs if f.id == "motor_thrust_imbalance"] == ["air", "idle_air"], statuses(fs)
    f = pick(fs, "motor_thrust_imbalance", "idle_air")
    assert f.status == HIT and abs(f.t_onset - 30.5) < 0.11 and f.values["agl_at_onset_m"] == 10.0, f
    assert "翻覆的症狀" not in f.text_zh and "在地面硬要修正" not in f.text_zh, f.text_zh
    assert "離地約 10.0 m" in f.text_zh and "symptom_of" not in f.values, f.text_zh
    assert f.values["peak_ratio_src"] == "recomputed" and f.values["peak_ratio"] < dm.IMB_RATIO_LATCH
    assert "與旗標不一致" in f.text_zh and "嚴重" not in f.text_zh, f.text_zh
    # 同樣的資料但有觸地錨點(t_touch 在失衡之前)→ 地面翻覆症狀
    fs = run_scene(sc, anchors=sc.anchors(t_touch=29.0))
    assert pick(fs, "motor_thrust_imbalance", "ground_idle").values["near_ground_src"] == "agl+t_touch"
    # 沒有高度也沒有觸地錨點 → 無法確認貼地 → idle_air(不歸給地面翻覆)
    sc.drop_cols = {"CTUN": ["Alt"]}
    f = pick(run_scene(sc), "motor_thrust_imbalance", "idle_air")
    assert f.values["near_ground_src"] == "unknown" and "沒有可用的高度" in f.text_zh, f
    print("  空中收油門的失衡 → idle_air(中性);有觸地錨點 → ground_idle;沒高度 → idle_air;不印與旗標矛盾的峰值比")


def test_imbalance_landed_gate():
    sc = Scene(land=50.0)
    sc.flags[sc.w(52.0, 56.0)] = 0.0                # 落地後(L=True)旗標掉 0:上鎖前後的瞬態,不算
    fs = run_scene(sc)
    assert [f.context for f in fs if f.id == "motor_thrust_imbalance"] == ["air"]
    assert pick(fs, "motor_thrust_imbalance", "air").status == CLEAR
    sc = Scene()
    sc.flags[sc.w(0.0, 1.5)] = 0.0                  # 起飛前(L=True)也不算
    assert pick(run_scene(sc), "motor_thrust_imbalance", "air").status == CLEAR
    print("  已落地(L=True)的旗標瞬態被閘掉")


def test_imbalance_quad_and_missing_inputs():
    sc = Scene(n_motors=4)
    sc.flags[sc.w(20.0, 25.0)] = 0.0
    f = pick(run_scene(sc), "motor_thrust_imbalance")
    assert f.status == NA and f.context == "n_motors<6", f
    sc = Scene()
    sc.drop_tables = {"CTUN"}
    f = pick(run_scene(sc), "motor_thrust_imbalance")
    assert f.status == NA and f.context == "no_tho", f
    print("  四軸 → NA(n_motors<6);缺 CTUN → NA(無法分空中 / 地面)")


def test_imbalance_recomputed_fallback():
    # (1) MOTB 沒有 FailFlags:由 RCOU 重算(左三顆高、右三顆怠速 = 比值 2.0)
    sc = Scene()
    s = sc.w(20.0, 30.0)
    sc.pwm[np.ix_(s, [1, 2, 5])] = 1700.0
    sc.pwm[np.ix_(s, [0, 3, 4])] = 1235.0
    sc.drop_cols = {"MOTB": ["FailFlags"]}
    f = pick(run_scene(sc), "motor_thrust_imbalance", "air")
    assert f.status == HIT and f.values["source"] == "recomputed" and 20.0 <= f.t_onset < 21.5, f
    assert f.values["peak_ratio"] > 1.9 and f.values["onset_high_channels"] == [2, 3, 6], f.values
    # (2) 平衡的飛行 → CLEAR
    sc = Scene()
    sc.drop_cols = {"MOTB": ["FailFlags"]}
    f = pick(run_scene(sc), "motor_thrust_imbalance", "air")
    assert f.status == CLEAR and f.values["source"] == "recomputed" and f.values["air_ratio_max"] < 1.1, f
    # (3) FailFlags 整欄都是 0(別的韌體語意)→ 不信旗標,改用重算,不會整段誤報
    sc = Scene()
    sc.flags[:] = 0.0
    f = pick(run_scene(sc), "motor_thrust_imbalance", "air")
    assert f.status == CLEAR and f.values["source"] == "recomputed", f
    print("  FailFlags 缺 / 語意不明 → 改用 RCOU 重算(HIT/CLEAR 都正確,不會整段誤報)")


# ======================================================================================
# 4 合成:偏航 / 姿態積分項
# ======================================================================================

def test_yaw_imbalance_synthetic():
    def run_i(axis, a, b, level):
        sc = Scene()
        sc.pid[axis][(sc.t4 >= a) & (sc.t4 < b)] = level
        return pick(run_scene(sc), "yaw_imbalance_iterm")
    # 0.45 = 90% IMAX 持續:LPF(τ=3.18 s)約在 5.7 s 後越過 0.75
    f = run_i("PIDY", 20.0, 50.0, 0.45)
    assert f.status == HIT and f.values["fired_axis"] == "PIDY" and 24.5 < f.t_onset < 27.0, f
    assert abs(f.t_alarm - f.t_onset - dm.YI_HOLD_S) < 1e-6 and f.severity == "high"
    f = run_i("PIDY", 20.0, 50.0, 0.30)                                  # 60% IMAX:低於 75%
    assert f.status == CLEAR and 0.55 < f.values["yaw_raw_max_ratio"] < 0.65, f
    # 8 s:低通後越過 75% 約 2.3 s。韌體在越過的當下就會警告(沒有持續時間)→ 至少 SOFT,不能說「可排除」
    f = run_i("PIDY", 20.0, 28.0, 0.45)
    assert f.status == SOFT and 2.0 < f.values["axes"]["PIDY"]["over_s"] < 2.6 and "韌體這時已會發出" in f.text_zh, f
    f = run_i("PIDY", 20.0, 34.0, 0.45)                                  # 未達 10 s → SOFT
    assert f.status == SOFT, f
    f = run_i("PIDR", 20.0, 50.0, 0.45)                                  # 滾轉 I-term wind-up 也算
    assert f.status == HIT and f.values["fired_axis"] == "PIDR", f
    f = run_i("PIDR", 20.0, 28.0, 0.45)                                  # 滾轉沒有韌體檢查:維持 5 s 的 SOFT 門檻
    assert f.status == CLEAR, f
    print("  積分項: 90% IMAX 持續 → HIT(起點 ≈ 25.7 s);60% → CLEAR;偏航短暫越過 → SOFT(韌體已會警告);滾轉 5 s 規則")


def test_yaw_firmware_details():
    """韌體 yaw_imbalance_check() 細節:I == IMAX 分支、落地時濾波器歸零、訊息時間 = 條件起點(10 s 只是訊息間隔)。"""
    def scene_with(fn):
        sc = Scene()
        fn(sc)
        return pick(run_scene(sc), "yaw_imbalance_iterm")
    # I 頂到 IMAX(0.5):不必等低通越過 75%(約 4.4 s),起點就是頂到的時刻
    f = scene_with(lambda sc: sc.pid["PIDY"].__setitem__((sc.t4 >= 20.0) & (sc.t4 < 35.0), 0.5))
    assert f.status == HIT and abs(f.t_onset - 20.0) < 0.01, f
    f = scene_with(lambda sc: sc.pid["PIDY"].__setitem__((sc.t4 >= 20.0) & (sc.t4 < 35.0), -0.5))   # 負向頂到也算
    assert f.status == HIT and abs(f.t_onset - 20.0) < 0.01, f
    # 落地時濾波器歸零:起飛前(L=True)I 在 0.45、起飛後 4 s 才降到 0.30 —— 沒歸零的話低通一起飛就 > 75%
    sc = Scene(takeoff=20.0)
    sc.pid["PIDY"][sc.t4 < 24.0] = 0.45
    sc.pid["PIDY"][sc.t4 >= 24.0] = 0.30
    f = pick(run_scene(sc), "yaw_imbalance_iterm")
    assert f.status == CLEAR and f.values["axes"]["PIDY"]["lpf_max_ratio"] < 0.70, f
    # 濾波值不得大於 |I|(韌體 "never allow to be larger than I"):I 從 0.45 掉到 0.1 時濾波值立刻跟著掉
    sc = Scene()
    sc.pid["PIDY"][(sc.t4 >= 10.0) & (sc.t4 < 30.0)] = 0.45
    sc.pid["PIDY"][(sc.t4 >= 30.0) & (sc.t4 < 50.0)] = 0.1
    f = pick(run_scene(sc), "yaw_imbalance_iterm")
    assert f.status == HIT and abs(f.values["axes"]["PIDY"]["over_s"] - (30.0 - 15.7)) < 0.3, f   # 越過 75% 到 30.0 s 立刻結束
    # 訊息:t_onset = 訊息時間(韌體條件一成立就送)
    sc = Scene()
    sc.messages = [(33.0, "Yaw Imbalance 82%")]
    f = pick(run_scene(sc), "yaw_imbalance_iterm")
    assert f.status == HIT and f.t_onset == 33.0 == f.t_alarm, f
    print("  I == IMAX 分支(正負)、落地歸零、濾波值不超過 |I|、訊息時間 = 起點")


def test_yaw_imbalance_message_and_gate():
    sc = Scene()
    sc.messages = [(33.0, "Yaw Imbalance 82%")]
    f = pick(run_scene(sc), "yaw_imbalance_iterm")
    assert f.status == HIT and f.t_onset == 33.0 and f.values["fired_axis"] == "MSG", f
    # 地面(已落地)的積分項堆積不算:30 s 起落地、I 在落地後飽和
    sc = Scene(land=30.0)
    sc.pid["PIDY"][(sc.t4 >= 30.0)] = 0.45
    f = pick(run_scene(sc), "yaw_imbalance_iterm")
    assert f.status == CLEAR, f
    # 沒有 PIDY → NA
    sc = Scene()
    sc.drop_tables = {"PIDY"}
    f = pick(run_scene(sc), "yaw_imbalance_iterm")
    assert f.status == NA and f.context == "no_pidy", f
    # IMAX 參數改變時比值跟著變(0.45 / 1.0 = 45% → CLEAR)
    sc = Scene()
    sc.pid["PIDY"][(sc.t4 >= 20.0)] = 0.45
    ctx = sc.ctx()
    ctx.params["ATC_RAT_YAW_IMAX"] = 1.0
    f = pick(dm.run(ctx, sc.anchors()), "yaw_imbalance_iterm")
    assert f.status == CLEAR and abs(f.values["yaw_raw_max_ratio"] - 0.45) < 1e-3, f
    print("  Yaw Imbalance 訊息 HIT;落地後堆積不算;缺 PIDY → NA;IMAX 參數有被讀")


# ======================================================================================
# 4 合成:強健性
# ======================================================================================

def test_channel_mapping_and_dtypes():
    ctx = Scene().ctx()
    assert dm._motor_channels(ctx) == [1, 2, 3, 4, 5, 6]
    ctx.params = {"SERVO3_FUNCTION": 34.0, "SERVO1_FUNCTION": 33.0, "SERVO2_FUNCTION": 36.0, "SERVO5_FUNCTION": 35.0,
                  "SERVO9_FUNCTION": 82.0, "SERVO10_FUNCTION": -1.0, "SERVO4_FUNCTION": 0.0}
    assert dm._motor_channels(ctx) == [1, 3, 5, 2]                          # 依馬達編號 33→1、34→3、35→5、36→2
    # uint16 輸入(模擬沒轉 float 的 RCOU / MOTB)不得回繞:1100 - 1235 不能變成 65401
    sc = Scene()
    ctx = sc.ctx()
    for k in range(6):
        ctx.tables["RCOU"][f"C{k + 1}"] = ctx.tables["RCOU"][f"C{k + 1}"].astype(np.uint16)
    ctx.tables["MOTB"]["FailFlags"] = ctx.tables["MOTB"]["FailFlags"].astype(np.uint8)
    ctx.tables["RCOU"]["C1"][:50] = 1100
    fs = dm.run(ctx, sc.anchors())
    check_contract(fs, "uint16")
    assert [f.status for f in fs] == [CLEAR, CLEAR, CLEAR], statuses(fs)
    print("  通道依 SERVOn_FUNCTION 排序;uint16 / uint8 輸入不回繞")


def test_nan_and_exceptions_are_contained():
    sc = Scene()
    ctx = sc.ctx()
    ctx.tables["MOTB"]["ThrOut"][100:120] = np.nan
    ctx.tables["ANG"]["Roll"][1000:1100] = np.nan
    ctx.tables["PIDY"]["I"][5000:5100] = np.nan
    fs = dm.run(ctx, sc.anchors())
    check_contract(fs, "nan")                       # 嚴格 JSON:NaN 不得流進 values(曾經 max_throut = NaN)
    assert not any(f.status in (HIT, SOFT) for f in fs), statuses(fs)
    tl = pick(fs, "motor_thrust_loss_check")
    assert tl.status == CLEAR and abs(tl.values["max_throut"] - 0.15) < 1e-6 and 2.0 <= tl.values["max_throut_t"] < 57.0, tl.values
    # 姿態表整段 NaN:不能下 CLEAR(資料無效 → NA no_attitude)
    sc = Scene()
    ctx = sc.ctx()
    ctx.tables["ANG"]["Roll"][:] = np.nan
    ctx.tables["ATT"]["Roll"][:] = np.nan
    fs = dm.run(ctx, sc.anchors())
    check_contract(fs, "nan attitude")
    f = pick(fs, "motor_thrust_loss_check")
    assert f.status == NA and f.context == "no_attitude", f
    # 只有 ANG 全 NaN → 退回 ATT(10 Hz)照常判定
    sc = Scene()
    ctx = sc.ctx()
    ctx.tables["ANG"]["Roll"][:] = np.nan
    assert pick(dm.run(ctx, sc.anchors()), "motor_thrust_loss_check").status == CLEAR
    # PIDY.I 整段 NaN → 偏航 NA(不是 CLEAR、也不是 NaN 比值)
    sc = Scene()
    ctx = sc.ctx()
    ctx.tables["PIDY"]["I"][:] = np.nan
    fs = dm.run(ctx, sc.anchors())
    check_contract(fs, "nan pidy")
    f = pick(fs, "yaw_imbalance_iterm")
    assert f.status == NA and f.values["yaw_raw_max_ratio"] is None, f
    # 例外 → guarded 轉 NA(context='error'),三個 id 都還在
    fs = dm.run(sc.ctx(boom=True), sc.anchors())
    assert {f.id for f in fs} == set(OWNED) and all(f.status == NA and f.context == "error" for f in fs), statuses(fs)
    # 沒有解鎖區間
    a = Anchors()
    fs = dm.run(sc.ctx(), a)
    assert len(fs) == 3 and all(f.status == NA and f.context == "not_armed" for f in fs)
    print("  NaN 不致誤報;例外被 guarded 包成 NA;沒有區間 → NA not_armed")


def test_deterministic_and_cache_isolated():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.truth_anchors_124(ctx)
    r1 = [(f.id, f.status, f.context, f.t_onset, f.values) for f in dm.run(ctx, a)]
    r2 = [(f.id, f.status, f.context, f.t_onset, f.values) for f in dm.run(ctx, a)]
    assert r1 == r2
    # 不同區間不會共用快取
    a1 = crashkit.flight1_anchors_124(ctx)
    f1 = dm.run(ctx, a1)
    assert pick(f1, "motor_thrust_loss_check").values["max_pwm"] == 1525
    assert pick(dm.run(ctx, a), "motor_thrust_loss_check").values["max_pwm"] == 1954
    print("  決定性:同輸入同輸出;不同區間快取互不污染")


# ======================================================================================
# 4 合成:證明門檻 / 機型不是寫死 124 的設定
# ======================================================================================

def test_yaw_soft_run_does_not_hide_later_hit():
    """同一軸先有一段 5~10 s(SOFT)再有一段 ≥ 10 s:要回 HIT(曾經在第一段 SOFT 就停止掃描)。"""
    sc = Scene(dur=120.0, land=117.0)
    sc.pid["PIDY"][(sc.t4 >= 10.0) & (sc.t4 < 24.0)] = 0.45
    sc.pid["PIDY"][(sc.t4 >= 50.0) & (sc.t4 < 100.0)] = 0.45
    f = pick(run_scene(sc), "yaw_imbalance_iterm")
    assert f.status == HIT and f.values["fired_axis"] == "PIDY" and 54.0 < f.t_onset < 58.0, f
    # 不同軸:滾轉只有 SOFT、俯仰有 HIT → HIT(俯仰)
    sc = Scene(dur=120.0, land=117.0)
    sc.pid["PIDR"][(sc.t4 >= 10.0) & (sc.t4 < 24.0)] = 0.45
    sc.pid["PIDP"][(sc.t4 >= 50.0) & (sc.t4 < 100.0)] = -0.45            # 負向積分同樣算(取絕對值)
    f = pick(run_scene(sc), "yaw_imbalance_iterm")
    assert f.status == HIT and f.values["fired_axis"] == "PIDP", f
    print(f"  先 SOFT 後 HIT:回 HIT(起點 {f.t_onset:.1f} s);負向積分 / 跨軸同樣正確")


def test_pwm_range_params_are_read():
    """同一條 PWM 1880 的軌跡:MOT_PWM 1000–2000 / SPIN 0.10–0.90 → act 0.975 飽和 HIT;本機 1100–2000 / 0.15–0.95 → act 0.90 CLEAR。"""
    sc = _sat_scene(1.5, lead=0.3, ae=45.0, ch=1, pwm=1880.0)
    f = pick(run_scene(sc), "motor_thrust_loss_check")
    assert f.status == CLEAR and f.values["pwm_sat"] == 1919, f.values          # 124 的設定:滿載門檻 PWM 1919
    ctx = sc.ctx()
    ctx.params.update({"MOT_PWM_MIN": 1000.0, "MOT_PWM_MAX": 2000.0, "MOT_SPIN_MIN": 0.10, "MOT_SPIN_MAX": 0.90})
    fs = dm.run(ctx, sc.anchors())
    check_contract(fs, "pwm range")
    f = pick(fs, "motor_thrust_loss_check")
    assert f.status == HIT and f.values["pwm_sat"] == 1860 and f.values["pwm_cmd_max"] == 1900, f.values
    assert abs(f.values["b_longest_s"] - 1.5) < 1e-6
    # MOT_PWM_MIN = 1200、全段輸出 1200(= 未轉動的最低輸出)→ NA no_spin;若把 1100 寫死會誤判成「在轉」
    sc = Scene()
    sc.pwm[:] = 1200.0
    ctx = sc.ctx()
    ctx.params["MOT_PWM_MIN"] = 1200.0
    fs = dm.run(ctx, sc.anchors())
    check_contract(fs, "pwm min 1200")
    assert all(f.status == NA and f.context == "no_spin" for f in fs), statuses(fs)
    print("  PWM 範圍 / SPIN 參數有被讀:同一軌跡 1100–2000 CLEAR、1000–2000 HIT(滿載門檻 1919 → 1860);MOT_PWM_MIN 1200 的轉動判定跟著變")


def test_octo_imbalance():
    """八軸(N=8,SERVO1..8_FUNCTION 33..40):旗標與重算兩條路都成立;不是寫死六軸。"""
    sc = Scene(n_motors=8)
    sc.flags[sc.w(20.0, 23.0)] = 0.0
    fs = run_scene(sc)
    f = pick(fs, "motor_thrust_imbalance", "air")
    assert f.status == HIT and f.values["n_motors"] == 8 and f.values["rcou_channels"] == list(range(1, 9)), f
    sc = Scene(n_motors=8)
    s = sc.w(20.0, 30.0)
    sc.pwm[np.ix_(s, [0, 2, 4, 6])] = 1700.0
    sc.pwm[np.ix_(s, [1, 3, 5, 7])] = 1235.0
    sc.drop_cols = {"MOTB": ["FailFlags"]}
    f = pick(run_scene(sc), "motor_thrust_imbalance", "air")
    assert f.status == HIT and f.values["source"] == "recomputed" and f.values["peak_ratio"] > 1.9, f.values
    assert f.values["onset_high_channels"] == [1, 3, 5, 7], f.values
    print("  八軸:旗標 HIT;重算(4 顆高 / 4 顆怠速)比值 2.0 HIT,先衝高的是 C1/C3/C5/C7")


def test_no_motb_and_no_servo_params():
    # 沒有 MOTB:推力損失用各馬達推力平均代替 ThrOut(註記),推力失衡改用重算
    sc = Scene()
    sc.drop_tables = {"MOTB"}
    fs = run_scene(sc)
    tl = pick(fs, "motor_thrust_loss_check")
    assert tl.status == CLEAR and tl.values["has_motb"] is False and any("MOTB" in n for n in tl.values["notes"]), tl.values
    assert pick(fs, "motor_thrust_imbalance", "air").values["source"] == "recomputed"
    sc = _sat_scene(1.5, lead=0.3, n_motors=6)
    sc.drop_tables = {"MOTB"}
    assert pick(run_scene(sc), "motor_thrust_loss_check").status == HIT                 # (b) 不靠 MOTB
    # 沒有 SERVOn_FUNCTION:改由「曾轉動的 RCOU 通道」推定(註記),結論照常
    sc = Scene()
    ctx = sc.ctx()
    for k in range(1, 17):
        ctx.params.pop(f"SERVO{k}_FUNCTION", None)
    ctx.tables["RCOU"]["C7"] = np.full(sc.t.size, 1100.0)                                 # 沒在轉的通道不算馬達
    fs = dm.run(ctx, sc.anchors())
    check_contract(fs, "no servo params")
    tl = pick(fs, "motor_thrust_loss_check")
    assert tl.status == CLEAR and tl.values["rcou_channels"] == [1, 2, 3, 4, 5, 6], tl.values
    assert any("SERVOn_FUNCTION" in n for n in tl.values["notes"])
    print("  缺 MOTB → 以推力平均代替 + 重算失衡;缺 SERVOn_FUNCTION → 由轉動通道推定(皆有註記)")


# ======================================================================================
# 5 引擎整合:驗收 A3 / A7 / A8 的馬達部分
# ======================================================================================

_REP124: dict = {}


def _rep124():
    """124 的完整引擎報告(每個 process 只跑一次)。"""
    if "rep" not in _REP124:
        from core.crash import engine
        _REP124["rep"] = engine.analyze(crashkit.load_ctx(crashkit.CRASH_LOG).ld)
    return _REP124["rep"]


def test_engine_consequence_saturation_keeps_ranking_124():
    """評審 P4 的引擎層回歸:本模組對「翻覆中才滿載」算出的推力損失結果(SOFT loss_of_control),
    移到 124 的 1258.5 s(晚於 t_loss 1258.10)取代原本那筆,排名完全不變:
    H_LAND_TIPOVER 第 1(高、分數相同、閘門通過)、H_NAV 不變、H_MOTOR_THRUST 0 分且已排除(依據標「後果,不計分」)。"""
    import copy
    from core.crash import engine
    from core.crash import hypotheses as hyp
    rep = _rep124()
    a = rep.anchors
    assert abs(a.t_loss - 1258.10) < 0.1, a.t_loss
    sc = _p4_scene()
    syn = pick(run_scene(sc, anchors=sc.anchors(t_touch=29.9, t_warn=31.9, t_loss=32.0)), "motor_thrust_loss_check")
    assert syn.status == SOFT and syn.context == "loss_of_control", syn
    inj = copy.deepcopy(syn)
    shift = 1258.5 - syn.t_onset
    inj.t_onset += shift
    inj.t_alarm += shift
    inj.tag = engine.tag_time(inj.t_onset, a)
    assert inj.tag == "consequence", inj.tag

    def ranked(findings):
        s = hyp.score_all([copy.deepcopy(f) for f in findings], a)
        s, _, _ = hyp.rank(s, a, data_ok=True)
        return sorted(s, key=lambda h: -h.score)
    base = ranked(rep.findings)
    new = ranked([f for f in rep.findings if f.id != "motor_thrust_loss_check"] + [inj])
    bh, nh = {h.id: h for h in base}, {h.id: h for h in new}
    assert base[0].id == new[0].id == "H_LAND_TIPOVER" and new[0].confidence == "高", (new[0].id, new[0].confidence)
    assert nh["H_LAND_TIPOVER"].gate_ok and "推力損失" not in nh["H_LAND_TIPOVER"].gate_note, nh["H_LAND_TIPOVER"].gate_note
    for hid in ("H_LAND_TIPOVER", "H_NAV_FAILURE"):
        assert abs(nh[hid].score - bh[hid].score) < 1e-9, (hid, nh[hid].score, bh[hid].score)
    hm = nh["H_MOTOR_THRUST"]
    assert hm.score == 0.0 and hm.status == "ruled_out", (hm.score, hm.status)
    assert any("後果,不計分" in b for b in hm.ruled_out_basis), hm.ruled_out_basis
    print(f"  注入失控後滿載(SOFT loss_of_control @1258.5 [{inj.tag}]):H_LAND {nh['H_LAND_TIPOVER'].score:.1f} 高 第 1 不變、"
          f"H_MOTOR 0 分已排除")

def test_engine_motor_rows_124():
    """完整引擎跑 124:H_MOTOR_THRUST 分數 0 且『已排除』,排除依據帶數字(1566 / 0.44 / ESC 註記);
    任何馬達證據都不是 precursor;地面怠速的推力失衡只標 development,不計入馬達假說。"""
    from core.crash import engine
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    t0 = time.perf_counter()
    rep = engine.analyze(ctx.ld)
    dt = time.perf_counter() - t0
    _REP124["rep"] = rep
    mot = [f for f in rep.findings if f.domain == "motor"]
    check_contract(mot, "engine 124")
    hm = {h.id: h for h in rep.hypotheses}["H_MOTOR_THRUST"]
    assert hm.status == "ruled_out" and hm.score == 0.0, (hm.status, hm.score)
    basis = " ".join(hm.ruled_out_basis)
    for s in ("1566", "0.44", "0.9", "ESC", "5%"):
        assert s in basis, (s, basis)
    assert rep.top() is not None and rep.top().id != "H_MOTOR_THRUST"
    ev = [f for f in mot if f.status in (HIT, SOFT)]
    assert [(f.id, f.context) for f in ev] == [("motor_thrust_imbalance", "ground_idle")], statuses(ev)
    assert ev[0].tag == "development", ev[0].tag
    assert all(f.tag != "precursor" for f in ev)
    print(f"  引擎: H_MOTOR_THRUST 已排除(0 分);唯一馬達證據 = 怠速失衡 [{ev[0].tag}];analyze {dt:.2f} s")


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
