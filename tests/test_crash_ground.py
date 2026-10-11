"""地面階段偵測模組測試(core/crash/det_ground.py):python tests/test_crash_ground.py 或 pytest tests/test_crash_ground.py

涵蓋:
  1. 00000124 飛行 2(真相錨點 / 只有 basic_anchors):5 個偵測器都 HIT,關鍵時間與數字對照驗收 A2/A4/A5
     (觸地 1255.225、6.9 s 沒有 EV17/18、油門 1255.25–1255.54 收掉、怠速全距 ~218 µs 左高右低、1257 起重力翻倒、
     觸地前 VIO 1.14 / EKF 0.54 m/s、EKF 漂移 ~1.6 m 而 VIO 幾乎沒動),並設定 a.t_touch;
  2. 00000124 飛行 1(正常降落,驗收 B)全部 CLEAR / NA;飛行 2 在 1255 s 之前沒有 HIT / SOFT;
  3. 6 個從未解鎖的 baseline:全部 NA not_armed;強制把整段當解鎖區間也只有 NA(不會說『正常』,也不誤報);
  4. 合成資料(數字刻意和 124 不同):每個偵測器的核心邏輯、降級(SOFT)與不適用(NA)路徑;
  5. 契約:每個 id 在每種情境都恰好一筆、values 可 JSON 序列化、HIT/SOFT 有 t_onset ≤ t_alarm、例外變 NA、耗時 < 1.5 s;
  6. 引擎整合:地面證據的因果標籤不是 consequence、t_first_anomaly = t_touch。
  7. 審查缺陷回歸(review_ground):
     a. a.t_touch 只給和事件相關的觸地(落地後再起飛 / 彈跳後空中失控 → None,engine 情境 = AIR_LOSS_OF_CONTROL);
     b. 空中急轉 / FLIP 的大角度不會截掉之後的地面翻覆證據,也不會被當成翻倒視窗;只在空中傾斜 → NA airborne_tilt;
     c. 空中收油門(未落地、離地 5 m)的不對稱不算 idle_motor_asymmetry(只有空中怠速 → NA air_idle_only);
     d. NaN 樣本不會讓結果變 CLEAR / 降級,text_zh 不出現 nan;
     e. 怠速不對稱的文字指出實際觸發的條件(全距 / 力矩 / 油門補償),馬達說明附『沒有 ESC 遙測』;
     f. open_end 區間:文字說『直到 log 結束』、區間末端的 EV17/18 不當成上鎖寫的;
     g. EKF 漂移對照 VIO 時回報 VIO 中斷與實際涵蓋時段(124:1255.128–1256.287 中斷 1.16 s)。
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
from core.crash import det_ground as dg  # noqa: E402
from core.crash.anchors import basic_anchors  # noqa: E402
from core.crash.ctx import Ctx  # noqa: E402
from core.crash.types import (CLEAR, HIT, NA, SOFT, TAG_CONSEQUENCE, TAG_DEVELOPMENT, TAG_PRECURSOR,  # noqa: E402
                              Anchors, ArmedInterval)

OWNED = ("touchdown_not_recognised", "idle_motor_asymmetry", "gravity_only_tipover",
         "touchdown_lateral_speed", "ekf_phantom_motion_on_ground")
G = 9.80665


# ======================================================================================
# 共用
# ======================================================================================

def pick(fs, det):
    c = [f for f in fs if f.id == det]
    assert c, f"{det} 沒有輸出:{statuses(fs)}"
    return c[0]


def statuses(fs):
    return [(f.id, f.status, f.context) for f in fs]


def check_contract(fs, label, a: Anchors | None = None, allow_error=False):
    """每個 id 恰好一筆(依序)、domain 正確、values 可 JSON 序列化、text_zh 非空、
    HIT/SOFT 有 t_onset ≤ t_alarm 且在解鎖區間內、NA 不帶 t_onset。"""
    assert [f.id for f in fs] == list(OWNED), (label, [f.id for f in fs])
    for f in fs:
        assert f.domain == "ground", (label, f.id, f.domain)
        assert f.status in (HIT, SOFT, CLEAR, NA), (label, f.status)
        json.dumps(f.values)
        assert f.text_zh, (label, f.id, "text_zh 空")
        if not allow_error:
            assert f.context != "error", (label, f.id, f.values)
        if f.status in (HIT, SOFT):
            assert f.t_onset is not None and f.t_alarm is not None and f.t_onset <= f.t_alarm + 1e-9, (label, f.id)
            if a is not None and a.interval is not None:
                assert a.interval.t0 <= f.t_onset <= a.interval.t1, (label, f.id, f.t_onset)
        if f.status == NA:
            assert f.t_onset is None, (label, f.id, "NA 不該有 t_onset")


def no_evidence(fs):
    return not any(f.status in (HIT, SOFT) for f in fs)


# ======================================================================================
# 合成資料(假 Ctx):證明邏輯不是寫死 124 的數字
# ======================================================================================

class FakeCtx:
    """最小 Ctx 替身。落地狀態 / 地面高度 / EV 讀取直接借用真 Ctx 的方法(語意一致)。"""
    landed_steps = Ctx.landed_steps
    landed_at = Ctx.landed_at
    ground_alt = Ctx.ground_alt
    ev_rows = Ctx.ev_rows
    ev_times = Ctx.ev_times

    # engine._scenario 用得到的最小介面(合成情境一律 LOITER、搖桿置中)
    def mode_rows(self):
        return np.array([0.0]), np.array([5]), ["LOITER"]

    def mode_at(self, t):
        return "LOITER"

    def rc_trim(self, ch, default=1500.0):
        return default

    def __init__(self, tables, params, boom=False):
        self.tables = tables
        self.params = params
        self.boom = boom
        self.armed = []

    def table(self, name, instance=None):
        if self.boom:
            raise RuntimeError("boom")
        return self.tables.get(name)

    def series(self, key):
        if self.boom:
            raise RuntimeError("boom")
        name, col = key.split(".", 1)
        tb = self.tables.get(name)
        if tb is None or col not in tb:
            return None
        return tb["t"], tb[col]

    def param(self, name, default=None):
        return self.params.get(name, default)

    def armed_at(self, t):
        return None


LEFT, RIGHT = (2, 3, 6), (1, 4, 5)          # hexa-X:左 C2/C3/C6、右 C1/C4/C5


class Scene:
    """合成 hexa-X 飛行(數字刻意和 124 不同):
    0 s 解鎖(地面怠速 1190 µs)、3 s 起飛(EV28)、3–8 s 爬到 5 m、8–30 s 懸停(PWM 1480、ThO=ThH=0.15)、
    30 s → td 以 0.5 m/s 下降、td 觸地衝擊、td+0.3 s 起油門 0 / 馬達 1235 µs。

    kind='tipover':td+1.0–3.5 s 左側 1450 / 右側 1235 µs、EKF 在地面『走』0.4 m/s(VIO 不動)、td+3.0 s 起只靠重力翻倒
                   (滾轉二次曲線 2°→90°,翻到 60° 後 0.35 s 一次硬撞擊 60 m/s²)、觸地前 1 s VIO 0.8 / EKF 0.3 m/s、
                   EV17 / 18 只在 CRASH_CHECK 上鎖那一刻(+60 µs)寫出。
    kind='normal' :EV17 td+0.6、EV18 td+1.4,怠速對稱、沒有翻倒,觸地前 VIO 0.1 / EKF 0.08 / GPS 0.12 m/s。"""

    def __init__(self, kind="tipover", td=40.0, t1=None, dur=60.0, n_motors=6, peak=33.0, seed=7):
        self.kind, self.td, self.dur, self.n = kind, td, dur, n_motors
        self.t1 = t1 if t1 is not None else (td + 8.0 if kind == "tipover" else td + 5.0)
        rng = np.random.default_rng(seed)
        self.t4 = np.arange(int(dur * 400)) / 400.0
        self.t10 = np.arange(int(dur * 10)) / 10.0
        self.t25 = np.arange(int(dur * 25)) / 25.0
        self.t27 = np.arange(int(dur * 27)) / 27.0
        self.t5 = np.arange(int(dur * 5)) / 5.0
        n4 = self.t4.size
        # IMU(FRD;AccZ 正放 −g)
        self.acc = np.stack([rng.normal(0, 0.05, n4), rng.normal(0, 0.05, n4), -G + rng.normal(0, 0.05, n4)], axis=1)
        self.gyr = rng.normal(0, 0.005, (n4, 3))
        self.roll4 = np.zeros(n4)
        self.desroll4 = np.zeros(n4)
        # CTUN
        t = self.t10
        self.alt = np.interp(t, [0, 3, 8, 30, td, dur], [0.5, 0.5, 5.5, 5.5, 0.5, 0.5])
        self.tho = np.where((t >= 3) & (t <= td), 0.15, 0.0)
        self.tho[(t > td) & (t < td + 0.3)] = 0.06
        self.thh = np.full(t.size, 0.15)
        self.pwm = np.tile(np.where(t < 3, 1190.0, np.where(t < td, 1480.0, 1235.0))[:, None], (1, n_motors))
        self.pwm[(t >= td) & (t < td + 0.3)] = 1300.0
        self.pwm += rng.integers(-8, 9, self.pwm.shape)
        self.throut = self.tho.copy()
        self.tan = np.zeros(t.size)
        # 導航
        self.vn_ekf = np.zeros(self.t25.size)
        self.vx_vio = np.zeros(self.t27.size)
        self.gps = {"Status": np.full(self.t5.size, 1.0), "NSats": np.zeros(self.t5.size),
                    "Spd": np.full(self.t5.size, 14.76)}               # 沒有定位:Spd 是舊值,不能用
        self.evs = [(0.0, 10), (3.0, 28)]
        self.drop_tables: set = set()
        self.params = {"FRAME_CLASS": 2.0, "FRAME_TYPE": 1.0, "MOT_PWM_MIN": 1100.0, "MOT_PWM_MAX": 2000.0,
                       "MOT_SPIN_MIN": 0.15}
        for k in range(n_motors):
            self.params[f"SERVO{k + 1}_FUNCTION"] = 33.0 + k
        self.shock(td + 0.02, peak)
        if kind == "tipover":
            self.pre_touch_speed(vio=0.8, ekf=0.3)
            self.asym(td + 1.0, td + 3.5)
            self.ekf_walk(td + 0.5, td + 3.0, 0.4)
            self.tan[(t >= td + 0.5) & (t < td + 2.5)] = 3.0              # 位置控制器在地面要 3 m/s² 加速度(纏繞)
            self.t_tip = td + 3.0
            self.tip(self.t_tip)
            self.evs += [(self.t1, 11), (self.t1 + 0.00006, 18), (self.t1 + 0.00007, 17), (self.t1 + 0.0019, 56)]
        else:
            self.pre_touch_speed(vio=0.1, ekf=0.08)
            m5 = (self.t5 >= 8) & (self.t5 < self.t1)
            self.gps["Status"][m5], self.gps["NSats"][m5], self.gps["Spd"][m5] = 3.0, 12.0, 0.12
            self.evs += [(td + 0.6, 17), (td + 1.4, 18), (self.t1, 11), (self.t1 + 0.0005, 56)]

    # ---------------- 編輯工具 ----------------
    def w10(self, a, b):
        return (self.t10 >= a) & (self.t10 < b)

    def w4(self, a, b):
        return (self.t4 >= a) & (self.t4 < b)

    def shock(self, tc, peak, half=0.03):
        """三角形衝擊:|a| 從 g 升到 peak 再回 g(加在 AccZ 的負方向)。"""
        tri = np.clip(1.0 - np.abs(self.t4 - tc) / half, 0.0, None)
        self.acc[:, 2] -= (peak - G) * tri

    def pre_touch_speed(self, vio, ekf, gps=None):
        td = self.td
        self.vx_vio[(self.t27 >= td - 1.0) & (self.t27 < td)] = vio
        self.vn_ekf[(self.t25 >= td - 1.0) & (self.t25 < td)] = ekf
        if gps is not None:
            m = (self.t5 >= td - 1.0) & (self.t5 < td)
            self.gps["Status"][m], self.gps["NSats"][m], self.gps["Spd"][m] = 3.0, 10.0, gps

    def asym(self, a, b, hi=1450.0, lo=1235.0, boost=0.075):
        m = self.w10(a, b)
        for k in range(self.n):
            self.pwm[m, k] = hi if (k + 1) in LEFT else lo
        self.throut[m] = boost

    def ekf_walk(self, a, b, v):
        self.vn_ekf[(self.t25 >= a) & (self.t25 < b)] = v

    def tip(self, t0, dur=1.0, extra_accz=0.0):
        """只靠重力的翻倒:roll = 2 + 53·u²(u=(t−t0)/dur)到 90° 停住;IMU 只看到重力(extra_accz ≠ 0 模擬推力)。
        翻到 90° 後 0.11 s 一次硬撞擊(60 m/s²、dV≈7.5)。"""
        u = (self.t4 - t0) / dur
        roll = np.where(self.t4 >= t0, np.minimum(2.0 + 53.0 * u * u, 90.0), self.roll4)
        m = self.t4 >= t0
        self.roll4[m] = roll[m]
        r = np.radians(roll[m])
        noise = np.random.default_rng(11).normal(0, 0.01, (int(m.sum()), 2))
        self.acc[m, 1] = G * np.sin(r) + noise[:, 0]
        self.acc[m, 2] = -G * np.cos(r) + noise[:, 1] - extra_accz * (self.roll4[m] < 90.0)
        rate = np.where(roll[m] < 90.0, np.radians(106.0 * u[m] / dur), 0.0)
        self.gyr[m, 0] = rate
        t90 = t0 + dur * np.sqrt(88.0 / 53.0)
        self.shock(t90 + 0.11, 60.0, half=0.15)

    # ---------------- 組成 Ctx ----------------
    def ctx(self, boom=False):
        t = self.t10
        crt = np.gradient(self.alt, t) * 100.0
        att10 = np.interp(t, self.t4, self.roll4)
        z4, z10 = np.zeros(self.t4.size), np.zeros(t.size)
        pn = np.cumsum(self.vn_ekf) / 25.0
        px = np.cumsum(self.vx_vio) / 27.0
        rc = {"t": t.copy()}
        for k in range(self.n):
            rc[f"C{k + 1}"] = self.pwm[:, k].copy()
        tabs = {
            "IMU": {"t": self.t4.copy(), "AccX": self.acc[:, 0].copy(), "AccY": self.acc[:, 1].copy(),
                    "AccZ": self.acc[:, 2].copy(), "GyrX": self.gyr[:, 0].copy(), "GyrY": self.gyr[:, 1].copy(),
                    "GyrZ": self.gyr[:, 2].copy()},
            "ANG": {"t": self.t4.copy(), "Roll": self.roll4.copy(), "Pitch": z4, "Yaw": z4,
                    "DesRoll": self.desroll4.copy(), "DesPitch": z4, "DesYaw": z4},
            "ATT": {"t": t.copy(), "Roll": att10, "Pitch": z10, "Yaw": z10,
                    "DesRoll": np.interp(t, self.t4, self.desroll4), "DesPitch": z10, "DesYaw": z10},
            "CTUN": {"t": t.copy(), "ThO": self.tho.copy(), "ThH": self.thh.copy(), "Alt": self.alt.copy(), "CRt": crt},
            "RCOU": rc,
            "MOTB": {"t": t.copy(), "ThrOut": self.throut.copy()},
            "PSCN": {"t": t.copy(), "TAN": self.tan.copy()},
            "PSCE": {"t": t.copy(), "TAE": z10.copy()},
            "XKF1": {"t": self.t25.copy(), "VN": self.vn_ekf.copy(), "VE": np.zeros(self.t25.size), "PN": pn,
                     "PE": np.zeros(self.t25.size)},
            "VISV": {"t": self.t27.copy(), "VX": self.vx_vio.copy(), "VY": np.zeros(self.t27.size)},
            "VISP": {"t": self.t27.copy(), "PX": px, "PY": np.zeros(self.t27.size)},
            "GPS": {"t": self.t5.copy(), **{k: v.copy() for k, v in self.gps.items()}},
            "EV": {"t": np.array([e[0] for e in self.evs], dtype=float), "Id": np.array([e[1] for e in self.evs], dtype=float)},
        }
        for name in self.drop_tables:
            tabs.pop(name, None)
        return FakeCtx(tabs, dict(self.params), boom=boom)

    def anchors(self, **kw):
        iv = ArmedInterval(0, 0.0, self.t1, 12 if self.kind == "tipover" else 0)
        a = Anchors(interval=iv, intervals=[iv])
        for k, v in kw.items():
            setattr(a, k, v)
        return a


def run_scene(sc, a=None, **kw):
    a = a or sc.anchors()
    fs = dg.run(sc.ctx(**kw), a)
    return fs, a


# ======================================================================================
# 1~3 真實 log
# ======================================================================================

def test_real_crash_truth_anchors():
    """124 飛行 2(真相錨點):5 個偵測器全部 HIT,數字對照驗收 A2/A4/A5 與研究階段實測。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.truth_anchors_124(ctx)
    a.t_touch = None                                   # 讓模組自己設
    fs = dg.run(ctx, a)
    check_contract(fs, "124 truth", a)
    assert all(f.status == HIT for f in fs), statuses(fs)

    td = pick(fs, "touchdown_not_recognised")
    v = td.values
    assert td.context == "near_ground" and abs(td.t_onset - 1255.225) <= 0.05, (td.context, td.t_onset)     # A4 ±0.05(400 Hz)
    assert a.t_touch is not None and abs(a.t_touch - 1255.225) <= 0.1, a.t_touch                         # A4 t_touch
    assert abs(td.t_alarm - td.t_onset - dg.RECOGNISE_WIN) < 1e-9
    assert abs(v["peak_ms2"] - 28.9) <= 0.5 and abs(v["dv_ms"] - 1.20) <= 0.1, v                         # ground_contact 28.9、dV 1.20
    assert abs(v["no_ev_s"] - 6.9) <= 0.05 and v["late_recognition"] is False, v["no_ev_s"]               # 6.9 s 沒有 EV17/18
    assert v["dt_ev17_s"] is None and v["dt_ev18_s"] is None
    assert v["ev_ignored_at_disarm"] == [17, 18], v["ev_ignored_at_disarm"]                               # 上鎖寫的不算
    assert abs(v["agl_m"] - (-0.13)) <= 0.1 and -45.0 <= v["vz_pre_cms"] <= -15.0, (v["agl_m"], v["vz_pre_cms"])   # A2
    assert v["landing_class"] == "GROUND_TIPOVER_like"
    assert 1255.10 <= v["t_tho_drop"] <= 1255.40 and abs(v["t_tho_zero"] - 1255.54) <= 0.15, v           # A4 ThO 1255.25–1255.54
    assert v["ldet_count_max"] == 22 and abs(v["ldet_bit8_t"] - 1256.193) <= 0.05, v                     # LDET Count 22/400、bit8
    assert v["windup"] is True and 24.0 <= v["lean_max_deg"] <= 26.5, (v["windup"], v["lean_max_deg"])   # A4 lean 24–26°
    assert 1.2 <= v["ekf_drift_m"] <= 2.0, v["ekf_drift_m"]                                               # A5 EKF 漂移 ~1.7 m
    assert "LAND_COMPLETE_MAYBE" in td.text_zh and "6.9" in td.text_zh

    im = pick(fs, "idle_motor_asymmetry")
    v = im.values
    assert im.context == "ground_idle" and abs(im.t_onset - 1256.34) <= 0.15, im.t_onset                 # 第一次 ≥0.5 s 1256.34
    assert abs(im.t_alarm - im.t_onset - dg.IDLE_MIN_S) < 1e-9
    assert abs(v["t_run_end"] - 1258.04) <= 0.15, v["t_run_end"]                                          # t_loss 截止
    assert abs(v["spread_med_us"] - 218) <= 10 and v["spread_max_us"] <= 230, v                           # ~220 µs
    assert v["left_pwm_range"][0] > v["right_pwm_range"][1], (v["left_pwm_range"], v["right_pwm_range"])   # 左 C2/C3/C6 > 右 C1/C4/C5
    assert v["roll_mom_med"] >= dg.IDLE_ROLLMOM_US and abs(v["boost_med"] - 0.077) <= 0.005, v
    assert v["n_floor"] >= 1 and v["pwm_med"]["C5"] <= 1240, v["pwm_med"]

    gt = pick(fs, "gravity_only_tipover")
    v = gt.values
    assert gt.context == "ground_tipover" and 1256.9 <= gt.t_onset <= 1257.3, gt.t_onset                  # A5『~1257 起重力翻倒』
    assert abs(v["t_end"] - 1258.05) <= 0.1 and abs(v["rms_ms2"] - 0.28) <= 0.05, v                      # 殘差 RMS 0.28
    assert v["tho_p90"] <= dg.TIP_THO_MAX and v["max_pwm"] <= 1460 and v["max_pwm"] < v["hover_pwm"], v   # RCOU ≤1460 < 懸停
    assert abs(v["hover_pwm"] - 1487) <= 12 and all(v["checks"].values()) and v["failed"] == [], v

    ls = pick(fs, "touchdown_lateral_speed")
    v = ls.values
    assert ls.context == "touchdown" and abs(v["t_touch"] - 1255.225) <= 0.05 and ls.t_alarm == td.t_onset
    assert abs(v["v_vio"] - 1.14) <= 0.05 and abs(v["v_ekf"] - 0.54) <= 0.05, v                           # VISV 1.14 / EKF 0.54
    assert abs(v["ekf_vio_diff"] - 0.61) <= 0.05 and v["v_gps"] is None, v                               # GPS 無定位 → 不採用
    assert abs(v["lateral_impact_ms2"] - 12.5) <= 1.0, v["lateral_impact_ms2"]
    assert 1254.2 <= ls.t_onset <= 1255.225, ls.t_onset

    ph = pick(fs, "ekf_phantom_motion_on_ground")
    v = ph.values
    assert ph.context == "static_idle" and 1255.95 <= ph.t_onset <= 1256.45, ph.t_onset                  # 1256.0–1256.75
    assert v["ekf_speed_med"] >= 0.2 and v["ekf_disp_m"] >= dg.PH_DISP and v["ekf_wrong"] is True, v
    assert v["vio_disp_m"] is not None and v["vio_disp_m"] < 0.05, v["vio_disp_m"]
    assert 1.4 <= v["ekf_drift_since_touch_m"] <= 1.9 and v["vio_drift_since_touch_m"] <= 0.2, v        # A5 EKF ~1.7 m / VIO ~0.07 m
    # VIO 在觸地時中斷 1255.128–1256.287(1.16 s):VIO 位移只涵蓋 1256.29 之後,要說清楚並給同段 EKF 對照
    assert abs(v["vio_gap_t0"] - 1255.128) <= 0.01 and abs(v["vio_gap_t1"] - 1256.287) <= 0.01, v
    assert abs(v["vio_gap_s"] - 1.16) <= 0.02 and abs(v["vio_span_t0"] - 1256.287) <= 0.01, v
    assert v["ekf_drift_vio_span_m"] > 3 * v["vio_drift_since_touch_m"], v                              # 同段比較 EKF 仍遠大於 VIO
    assert "中斷" in ph.text_zh and "1255.13–1256.29" in ph.text_zh, ph.text_zh
    print(f"  觸地 {td.t_onset:.3f} s({td.values['peak_ms2']} m/s²)無 EV17/18 {td.values['no_ev_s']} s;"
          f"油門 {td.values['t_tho_drop']}→{td.values['t_tho_zero']};傾斜指令最大 {td.values['lean_max_deg']}°")
    print(f"  怠速不對稱 {im.t_onset:.2f} s 全距 {im.values['spread_med_us']} µs;重力翻倒 {gt.t_onset:.2f}–{gt.values['t_end']:.2f} s "
          f"RMS {gt.values['rms_ms2']};觸地速度 VIO {ls.values['v_vio']} / EKF {ls.values['v_ekf']};"
          f"幻影運動 {ph.t_onset:.2f} s 漂移 {ph.values['ekf_drift_since_touch_m']} m")


def test_real_crash_basic_anchors():
    """只有 basic_anchors(沒有 t_warn / t_loss / t_touch):本地截止(硬撞擊 / 傾斜 60°)下結論不變,a.t_touch 由本模組填。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = basic_anchors(ctx)
    assert a.t_touch is None and a.t_warn is None
    fs = dg.run(ctx, a)
    check_contract(fs, "124 basic", a)
    assert all(f.status == HIT for f in fs), statuses(fs)
    assert abs(a.t_touch - 1255.225) <= 0.1, a.t_touch
    assert abs(pick(fs, "idle_motor_asymmetry").t_onset - 1256.34) <= 0.15
    assert 24.0 <= pick(fs, "touchdown_not_recognised").values["lean_max_deg"] <= 26.5
    print(f"  basic_anchors: 5 HIT,t_touch {a.t_touch:.3f}")


def test_real_flight1_normal_landing():
    """飛行 1(正常降落對照,驗收 B):觸地 615.57 後 EV17 +0.82 / EV18 +1.62 → CLEAR;怠速全距 ≤57 µs;沒有任何 HIT / SOFT。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.flight1_anchors_124(ctx)
    fs = dg.run(ctx, a)
    check_contract(fs, "124 flight1", a)
    assert no_evidence(fs), statuses(fs)
    assert statuses(fs) == [("touchdown_not_recognised", CLEAR, "recognised"), ("idle_motor_asymmetry", CLEAR, "ground_idle"),
                            ("gravity_only_tipover", NA, "no_tipover"), ("touchdown_lateral_speed", CLEAR, "touchdown"),
                            ("ekf_phantom_motion_on_ground", CLEAR, "static_idle")], statuses(fs)
    td = pick(fs, "touchdown_not_recognised")
    assert abs(td.t_onset - 615.57) <= 0.05 and abs(td.values["peak_ms2"] - 25.5) <= 0.5, td.values
    assert abs(td.values["dt_ev17_s"] - 0.82) <= 0.05 and abs(td.values["dt_ev18_s"] - 1.62) <= 0.05, td.values
    assert a.t_touch is None                                  # 沒有 t_warn:正常降落不設觸地錨點
    im = pick(fs, "idle_motor_asymmetry").values
    assert im["spread_p95_us"] <= 57 and im["boost_max"] <= 0.023 and im["flag_longest_s"] < 0.2, im
    assert pick(fs, "gravity_only_tipover").values["max_tilt_deg"] < 12.0
    ls = pick(fs, "touchdown_lateral_speed").values
    assert ls["v_max"] <= 0.2 and ls["ekf_vio_diff"] <= 0.05 and ls["v_gps"] is not None, ls          # 0.12–0.17 m/s
    assert abs(ls["lateral_impact_ms2"] - 7.4) <= 1.0, ls["lateral_impact_ms2"]
    ph = pick(fs, "ekf_phantom_motion_on_ground").values
    assert ph["speed_med_max"] <= 0.05 and ph["disp_max"] <= 0.035, ph                                  # 健康怠速 ≤0.05 m/s、0.03 m
    print(f"  飛行 1: 觸地 {td.t_onset:.2f} s EV17 +{td.values['dt_ev17_s']} s;怠速全距 p95 {im['spread_p95_us']} µs;"
          f"觸地速度 {ls['v_max']} m/s;EKF 地面速度 ≤{ph['speed_med_max']} m/s")


def test_real_flight2_before_touchdown():
    """飛行 2 在 1255 s 之前(含 376 s 空中):沒有 HIT / SOFT(驗收 B 最後一句);整段飛行的地面證據也都在觸地附近之後。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    iv = ctx.armed[1]
    for t_cut in (1255.0, 1255.2):
        sub = ArmedInterval(1, iv.t0, t_cut, 0)
        a = Anchors(interval=sub, intervals=list(ctx.armed))
        fs = dg.run(ctx, a)
        check_contract(fs, f"124 flight2 <{t_cut}", a)
        assert no_evidence(fs), (t_cut, statuses(fs))
        assert pick(fs, "touchdown_lateral_speed").context == "no_touchdown"
        assert pick(fs, "idle_motor_asymmetry").values["spread_p95_us"] <= 57
        assert a.t_touch is None
    # 觸地後 2 s 內就截斷:觀察窗不足 → NA(不能 HIT 也不能 CLEAR)
    a = Anchors(interval=ArmedInterval(1, iv.t0, 1256.5, 0), intervals=list(ctx.armed))
    assert pick(dg.run(ctx, a), "touchdown_not_recognised").context == "window_truncated"
    # 完整飛行 2:所有證據 t_onset ≥ 觸地前 1 s(觸地速度看觸地前 1 s 是定義上的),其餘都在觸地之後
    a = crashkit.truth_anchors_124(ctx)
    fs = dg.run(ctx, a)
    for f in fs:
        lo = 1254.2 if f.id == "touchdown_lateral_speed" else 1255.2
        assert f.t_onset >= lo, (f.id, f.t_onset)
    print("  飛行 2 <1255 s:沒有 HIT / SOFT;觸地後 1.3 s 截斷 → NA window_truncated")


def test_real_baselines_not_armed():
    """6 個從未解鎖的 baseline:5 筆全部 NA 'not_armed'(不能是 CLEAR)。"""
    for nm in crashkit.BASELINES:
        ctx = crashkit.load_ctx(nm)
        a = crashkit.not_armed_anchors(ctx)
        fs = dg.run(ctx, a)
        check_contract(fs, nm, a)
        assert all(f.status == NA and f.context == "not_armed" for f in fs), (nm, statuses(fs))
        assert all("未解鎖" in f.text_zh for f in fs) and a.t_touch is None
    print(f"  {len(crashkit.BASELINES)} 個 baseline 全部 NA not_armed")


def test_real_baselines_forced_interval_no_false_alarm():
    """硬把整段 log 當解鎖區間(模擬閘門被拿掉):仍然沒有 HIT / SOFT,也沒有 CLEAR(沒有起飛 / 馬達沒轉 → NA)。"""
    for nm in crashkit.BASELINES:
        ctx = crashkit.load_ctx(nm)
        iv = ArmedInterval(0, 0.0, ctx.t_end)
        a = Anchors(interval=iv, intervals=[iv])
        fs = dg.run(ctx, a)
        check_contract(fs, nm + " forced", a)
        assert all(f.status == NA for f in fs), (nm, statuses(fs))
        assert pick(fs, "touchdown_not_recognised").context == "no_takeoff"
        assert pick(fs, "idle_motor_asymmetry").context == "no_idle_window"
    print("  強制區間: baseline 全部 NA(no_takeoff / no_idle_window / no_tipover / no_touchdown / no_static_idle)")


def test_real_shock_clusters_match_acceptance():
    """本模組內部的衝擊簇和驗收 D(3) 一致:124 = {615.57 觸地, 1255.225 觸地, 1258.183–1258.808 硬 62.1, 1259.463–1259.958 硬 90.1};
    baseline 強制區間 0 個 >20 m/s² 的簇(最大 10.1–16.5)。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    sh = dg._shocks(ctx, ctx.armed[0]) + dg._shocks(ctx, ctx.armed[1])
    exp = [(615.57, None, 25.5, "ground_contact"), (1255.225, None, 28.9, "ground_contact"),
           (1258.183, 1258.808, 62.1, "hard_impact"), (1259.463, 1259.958, 90.1, "hard_impact")]
    assert len(sh) == len(exp), [(s.t_start, s.peak) for s in sh]
    for s, (t0, t1, pk, cls) in zip(sh, exp):
        assert abs(s.t_start - t0) <= 0.05 and abs(s.peak - pk) <= 0.5 and s.cls == cls, (s, t0, pk, cls)
        if t1 is not None:
            assert abs(s.t_end - t1) <= 0.05, (s.t_end, t1)
    assert abs(sh[0].dv - 0.64) <= 0.1 and abs(sh[2].dv - 4.6) <= 0.2 and abs(sh[3].dv - 6.9) <= 0.2, [s.dv for s in sh]
    for nm in crashkit.BASELINES:
        c = crashkit.load_ctx(nm)
        assert dg._shocks(c, ArmedInterval(0, 0.0, c.t_end)) == [], nm
    print("  " + "; ".join(f"{s.t_start:.3f} {s.peak:.1f} m/s² dV {s.dv:.2f} {s.cls}" for s in sh) + ";baseline 0 簇")


def test_real_engine_integration():
    """引擎整合:地面證據不是 consequence(會計分)、t_first_anomaly = t_touch(驗收 A4)、情境 GROUND_TIPOVER(A2)。
    假說分數由多個模組決定,這裡只驗地面模組負責的部分。"""
    from core.crash import engine
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    rep = engine.analyze(ctx.ld)
    a = rep.anchors
    assert abs(a.t_touch - 1255.225) <= 0.1 and abs(a.t_first_anomaly - a.t_touch) <= 1e-6, (a.t_touch, a.t_first_anomaly)
    assert rep.scenario == "GROUND_TIPOVER", rep.scenario
    g = {f.id: f for f in rep.findings if f.domain == "ground"}
    assert set(g) == set(OWNED) and all(f.status == HIT for f in g.values()), {k: f.status for k, f in g.items()}
    assert g["touchdown_not_recognised"].tag == TAG_PRECURSOR and g["touchdown_lateral_speed"].tag == TAG_PRECURSOR
    for k in ("idle_motor_asymmetry", "gravity_only_tipover", "ekf_phantom_motion_on_ground"):
        assert g[k].tag in (TAG_PRECURSOR, TAG_DEVELOPMENT) and g[k].tag != TAG_CONSEQUENCE, (k, g[k].tag)
    top = rep.top()
    assert top is not None and top.id == "H_LAND_TIPOVER", top
    print(f"  引擎:t_touch = t_first_anomaly = {a.t_touch:.3f};{', '.join(f'{k}={f.tag}' for k, f in g.items())};"
          f"排名第一 {top.id} {top.score:.1f}")


def test_real_timing():
    """每個偵測器 < 1.5 s(冷快取:清掉衝擊簇快取;資料表快取保留 = 引擎實際行為)。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    for label, a in (("truth", crashkit.truth_anchors_124(ctx)), ("flight1", crashkit.flight1_anchors_124(ctx))):
        for fn in (dg.touchdown_not_recognised, dg.idle_motor_asymmetry, dg.gravity_only_tipover,
                   dg.touchdown_lateral_speed, dg.ekf_phantom_motion_on_ground):
            dg._SHOCK_CACHE.clear()
            t0 = time.perf_counter()
            fn(ctx, a)
            dt = time.perf_counter() - t0
            assert dt < 1.5, f"{label} {fn.__name__} {dt:.2f}s"
            print(f"  {label} {fn.__name__}: {dt * 1000:.0f} ms")


def test_real_deterministic_and_cache_isolated():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    r1 = [(f.id, f.status, f.context, f.t_onset, f.values) for f in dg.run(ctx, crashkit.truth_anchors_124(ctx))]
    r2 = [(f.id, f.status, f.context, f.t_onset, f.values) for f in dg.run(ctx, crashkit.truth_anchors_124(ctx))]
    assert r1 == r2
    f1 = dg.run(ctx, crashkit.flight1_anchors_124(ctx))               # 不同區間不共用衝擊簇快取
    assert pick(f1, "touchdown_not_recognised").status == CLEAR
    assert pick(dg.run(ctx, crashkit.truth_anchors_124(ctx)), "touchdown_not_recognised").status == HIT
    print("  決定性:同輸入同輸出;不同區間快取互不污染")


# ======================================================================================
# 4 合成:各偵測器
# ======================================================================================

def test_syn_tipover_all_hit():
    """合成翻覆(td=40、33 m/s²、上鎖 48 s):5 個都 HIT,時間點落在設計值;a.t_touch = 觸地。"""
    sc = Scene("tipover")
    fs, a = run_scene(sc)
    check_contract(fs, "syn tipover", a)
    assert all(f.status == HIT for f in fs), statuses(fs)
    td = pick(fs, "touchdown_not_recognised")
    assert abs(td.t_onset - 40.0) <= 0.02 and abs(a.t_touch - td.t_onset) < 1e-9, td.t_onset
    v = td.values
    assert abs(v["peak_ms2"] - 33.0) <= 0.5 and abs(v["no_ev_s"] - 8.0) <= 0.05 and v["ev_ignored_at_disarm"] == [17, 18], v
    assert abs(v["agl_m"]) <= 0.05 and abs(v["vz_pre_cms"] + 50.0) <= 5.0, v
    assert abs(v["t_tho_drop"] - 40.1) < 1e-6 and abs(v["t_tho_zero"] - 40.3) < 1e-6 and abs(v["t_tho_sustained"] - 40.3) < 1e-6, v
    assert v["windup"] is True and abs(v["windup_t"] - 40.5) <= 0.11 and v["ldet_count_max"] is None, v
    im = pick(fs, "idle_motor_asymmetry")
    assert abs(im.t_onset - 41.0) < 1e-6 and abs(im.values["run_s"] - 2.4) <= 0.11, (im.t_onset, im.values["run_s"])
    assert abs(im.values["spread_med_us"] - 215) <= 16 and im.values["roll_mom_med"] >= 600, im.values
    gt = pick(fs, "gravity_only_tipover")
    t5, t50 = 43.0 + np.sqrt(3 / 53), 43.0 + np.sqrt(48 / 53)
    assert abs(gt.t_onset - t5) <= 0.01 and abs(gt.values["t_end"] - t50) <= 0.01, (gt.t_onset, gt.values["t_end"])
    assert gt.values["rms_ms2"] < 0.1 and gt.values["gyro_rho"] > 0.95 and gt.values["max_pwm"] < gt.values["hover_pwm"]
    ls = pick(fs, "touchdown_lateral_speed")
    assert abs(ls.values["v_vio"] - 0.8) < 1e-6 and abs(ls.values["v_ekf"] - 0.3) < 1e-6 and ls.values["v_gps"] is None
    assert 39.0 <= ls.t_onset <= 39.1, ls.t_onset
    ph = pick(fs, "ekf_phantom_motion_on_ground")
    assert 40.2 <= ph.t_onset <= 40.5 and ph.values["ekf_wrong"] and ph.values["vio_disp_m"] < 0.01, (ph.t_onset, ph.values)
    assert abs(ph.values["ekf_drift_since_touch_m"] - 1.0) <= 0.05, ph.values
    print(f"  合成翻覆: 觸地 {td.t_onset:.3f}、怠速不對稱 {im.t_onset:.1f}、翻倒 {gt.t_onset:.2f}–{gt.values['t_end']:.2f}、"
          f"幻影 {ph.t_onset:.2f}(漂移 {ph.values['ekf_drift_since_touch_m']} m)")


def test_syn_normal_landing_all_clear():
    """合成正常降落:EV17 +0.6 / EV18 +1.4 → CLEAR;沒有翻倒 → NA;沒有任何 HIT / SOFT。有 t_warn 時 t_touch = 觸地。"""
    sc = Scene("normal", peak=24.0)
    fs, a = run_scene(sc)
    check_contract(fs, "syn normal", a)
    assert statuses(fs) == [("touchdown_not_recognised", CLEAR, "recognised"), ("idle_motor_asymmetry", CLEAR, "ground_idle"),
                            ("gravity_only_tipover", NA, "no_tipover"), ("touchdown_lateral_speed", CLEAR, "touchdown"),
                            ("ekf_phantom_motion_on_ground", CLEAR, "static_idle")], statuses(fs)
    td = pick(fs, "touchdown_not_recognised")
    assert abs(td.values["dt_ev17_s"] - (40.6 - td.t_onset)) <= 1e-3 and abs(td.values["dt_ev18_s"] - (41.4 - td.t_onset)) <= 1e-3
    assert a.t_touch is None
    assert abs(pick(fs, "touchdown_lateral_speed").values["v_gps"] - 0.12) < 1e-6        # 有定位的 GPS 才採用
    _, a2 = run_scene(Scene("normal", peak=24.0), Scene("normal").anchors(t_warn=42.0))
    assert a2.t_touch is not None and abs(a2.t_touch - td.t_onset) < 1e-9
    print(f"  合成正常降落: 全 CLEAR / NA;EV17 +{td.values['dt_ev17_s']:.2f} s")


def test_syn_touchdown_variants():
    """觸地未辨識:延遲辨識、觀察窗不足、空中衝擊、沒有起飛事件、缺姿態 / 油門未收 → SOFT、缺 EV / IMU → NA。"""
    # 延遲辨識(EV17 +3.0 s):仍 HIT,values 記延遲,文字不說『直到上鎖都沒有』
    sc = Scene("tipover")
    sc.evs += [(43.0, 17)]
    fs, _ = run_scene(sc)
    f = pick(fs, "touchdown_not_recognised")
    assert f.status == HIT and f.values["late_recognition"] is True and abs(f.values["dt_ev17_s"] - (43.0 - f.t_onset)) <= 1e-3
    assert abs(f.values["no_ev_s"] - 3.0) <= 0.02 and "直到上鎖" not in f.text_zh and "才出現" in f.text_zh, f.text_zh
    # EV17 在門檻內(+1.5 s)→ CLEAR
    sc = Scene("tipover")
    sc.evs += [(41.5, 17)]
    assert pick(run_scene(sc)[0], "touchdown_not_recognised").status == CLEAR
    # 觸地後 1.5 s 就上鎖:觀察窗不足 → NA
    sc = Scene("normal", t1=41.5)
    sc.evs = [(0.0, 10), (3.0, 28), (41.5, 11)]
    f = pick(run_scene(sc)[0], "touchdown_not_recognised")
    assert f.status == NA and f.context == "window_truncated", f
    # 整段沒有 EV28(飛控從未宣告離地):不能 CLEAR,也不能 HIT
    sc = Scene("tipover")
    sc.evs = [e for e in sc.evs if e[1] != 28]
    fs, a = run_scene(sc)
    f = pick(fs, "touchdown_not_recognised")
    assert f.status == NA and f.context == "no_takeoff" and a.t_touch is None, f
    assert pick(fs, "touchdown_lateral_speed").context == "no_touchdown"
    # 空中衝擊(離地 5 m,沒有降落)→ NA airborne_shock
    sc = Scene("normal", t1=30.0)
    sc.evs = [(0.0, 10), (3.0, 28), (30.0, 11)]
    sc.shock(20.0, 26.0)
    f = pick(run_scene(sc)[0], "touchdown_not_recognised")
    assert f.status == NA and f.context == "airborne_shock" and abs(f.values["agl_m"] - 5.0) <= 0.1, f
    # 缺姿態(ANG / ATT)→ SOFT(無法確認觸地時姿態正常)
    sc = Scene("tipover")
    sc.drop_tables |= {"ANG", "ATT"}
    fs, _ = run_scene(sc)
    f = pick(fs, "touchdown_not_recognised")
    assert f.status == SOFT and f.values["downgrade"] == ["ae_unavailable"], f.values
    assert pick(fs, "gravity_only_tipover").context == "no_attitude"
    # 觸地後油門沒收(彈跳 / 繼續飛)→ SOFT
    sc = Scene("tipover")
    sc.tho[sc.t10 >= 40.0] = 0.15
    sc.throut[:] = sc.tho
    f = pick(run_scene(sc)[0], "touchdown_not_recognised")
    assert f.status == SOFT and "no_throttle_collapse" in f.values["downgrade"], f.values
    # 缺 EV 表 / 缺 IMU
    sc = Scene("tipover")
    sc.drop_tables.add("EV")
    assert pick(run_scene(sc)[0], "touchdown_not_recognised").context == "no_ev"
    sc = Scene("tipover")
    sc.drop_tables.add("IMU")
    fs, _ = run_scene(sc)
    assert [pick(fs, i).context for i in ("touchdown_not_recognised", "touchdown_lateral_speed", "ekf_phantom_motion_on_ground")] \
        == ["no_imu", "no_imu", "no_imu"], statuses(fs)
    print("  延遲辨識 HIT(記延遲)/ 1.5 s 內 CLEAR / 截斷 NA / 無 EV28 NA / 空中衝擊 NA / 缺姿態、油門未收 SOFT / 缺 EV、IMU NA")


def test_syn_touchdown_fallback_paths():
    """沒有觸地衝擊(<20 m/s² 的輕觸地)時的備援:油門怠速卻未落地 → SOFT;很快辨識 → CLEAR;
    空中油門崩潰(spec:不是本偵測器)→ NA airborne_throttle_collapse;崩潰不足 3 s → CLEAR。"""
    sc = Scene("normal", peak=15.0)
    sc.evs = [(0.0, 10), (3.0, 28), (45.0, 11)]
    fs, a = run_scene(sc)
    f = pick(fs, "touchdown_not_recognised")
    assert f.status == SOFT and f.context == "idle_no_shock" and abs(f.t_onset - 40.3) < 1e-6, f
    assert a.t_touch is None, a.t_touch                  # 沒有事件參考點時 SOFT(證據弱)不設觸地錨點
    assert pick(fs, "touchdown_lateral_speed").context == "no_touchdown"
    _, a = run_scene(sc, sc.anchors(t_warn=42.0))        # 有 t_warn 且在相關窗內 → 設定
    assert a.t_touch is not None and abs(a.t_touch - 40.3) < 1e-6, a.t_touch
    sc = Scene("normal", peak=15.0)
    f = pick(run_scene(sc)[0], "touchdown_not_recognised")
    assert f.status == CLEAR and f.context == "recognised_no_shock", f
    # 空中(5 m)油門 0 持續 4 s,接著硬撞擊(dV≥3,不是觸地型)
    sc = Scene("normal", t1=40.0)
    sc.evs = [(0.0, 10), (3.0, 28), (40.0, 11)]
    sc.tho[sc.w10(25.0, 29.0)] = 0.0
    sc.shock(32.0, 70.0, half=0.15)
    f = pick(run_scene(sc)[0], "touchdown_not_recognised")
    assert f.status == NA and f.context == "airborne_throttle_collapse", f
    assert abs(f.values["t_start"] - 25.0) < 1e-6 and f.values["idle_s"] >= 3.0 and f.values["agl_m"] > 4.0, f.values
    sc = Scene("normal", t1=40.0)
    sc.evs = [(0.0, 10), (3.0, 28), (40.0, 11)]
    sc.tho[sc.w10(25.0, 27.0)] = 0.0
    f = pick(run_scene(sc)[0], "touchdown_not_recognised")
    assert f.status == CLEAR and f.context == "no_ground_contact", f
    print("  備援:怠速未落地 SOFT / 辨識 CLEAR / 空中油門崩潰 4 s → NA airborne_throttle_collapse / 2 s → CLEAR")


def test_syn_idle_variants():
    """怠速不對稱:0.3 s → SOFT、0.1 s → CLEAR、四軸只用全距、只有油門補償也算、t_loss 之後不算、馬達沒轉 → NA、uint16 不回繞。"""
    def idle_only(**mods):
        sc = Scene("normal", peak=24.0)
        for k, v in mods.items():
            setattr(sc, k, v)
        return sc

    sc = idle_only()
    sc.asym(42.0, 42.35)
    f = pick(run_scene(sc)[0], "idle_motor_asymmetry")
    assert f.status == SOFT and abs(f.values["run_s"] - 0.3) <= 0.01 and abs(f.t_alarm - (f.t_onset + 0.3)) <= 0.011, f.values
    sc = idle_only()
    sc.asym(42.0, 42.15)
    f = pick(run_scene(sc)[0], "idle_motor_asymmetry")
    assert f.status == CLEAR and f.values["n_flagged_samples"] == 2 and "0.1 s" in f.text_zh, (f.values, f.text_zh)
    # 四軸(FRAME_CLASS 1):沒有左右力矩,只用全距
    sc = Scene("normal", peak=24.0, n_motors=4)
    sc.params["FRAME_CLASS"] = 1.0
    sc.pwm[sc.w10(42.0, 43.0), 0] = 1400.0
    f = pick(run_scene(sc)[0], "idle_motor_asymmetry")
    assert f.status == HIT and f.values["roll_mom_med"] is None and f.values["spread_med_us"] >= 150, f.values
    # PWM 對稱,只有 MOTB 油門補償 0.06(≥0.04)持續 1 s
    sc = idle_only()
    sc.throut[sc.w10(42.0, 43.0)] = 0.06
    f = pick(run_scene(sc)[0], "idle_motor_asymmetry")
    assert f.status == HIT and f.values["spread_med_us"] < 30 and f.values["boost_med"] >= 0.04, f.values
    # 不對稱只發生在 t_loss 之後(撞擊 / 失控後的倒扣怠速)→ 不算
    sc = idle_only()
    sc.asym(42.5, 44.0)
    f = pick(run_scene(sc, sc.anchors(t_loss=42.0))[0], "idle_motor_asymmetry")
    assert f.status == CLEAR, f
    # 馬達從沒轉起來(怠速 1100)→ NA
    sc = idle_only()
    sc.pwm[sc.t10 < 3.0] = 1100.0
    sc.pwm[sc.t10 >= 40.0] = 1100.0
    f = pick(run_scene(sc)[0], "idle_motor_asymmetry")
    assert f.status == NA and f.context == "no_idle_window", f
    # uint16 RCOU(正常是 Ctx 轉好;直接餵 uint16 也不會回繞)
    sc = Scene("tipover")
    ctx = sc.ctx()
    for k in range(6):
        ctx.tables["RCOU"][f"C{k + 1}"] = ctx.tables["RCOU"][f"C{k + 1}"].astype(np.uint16)
    f = pick(dg.run(ctx, sc.anchors()), "idle_motor_asymmetry")
    assert f.status == HIT and 600 <= f.values["roll_mom_med"] <= 700, f.values
    print("  0.3 s SOFT / 0.1 s CLEAR / 四軸全距 HIT / 只有補償 HIT / t_loss 後不算 / 1100 µs NA / uint16 OK")


def test_syn_gravity_variants():
    """重力翻倒:推力參與(殘差大)→ CLEAR;油門高 / PWM 高於懸停 → SOFT;太快(<0.3 s)→ NA;只有 ATT 也能算。"""
    sc = Scene("tipover")
    sc.acc[:, 2] = -G + np.random.default_rng(1).normal(0, 0.05, sc.t4.size)
    sc.roll4[:] = 0.0
    sc.tip(43.0, extra_accz=3.0)
    f = pick(run_scene(sc)[0], "gravity_only_tipover")
    assert f.status == CLEAR and f.context == "not_gravity_only" and f.values["rms_ms2"] > 2.0, f.values
    sc = Scene("tipover")
    sc.tho[sc.w10(42.5, 45.0)] = 0.3
    f = pick(run_scene(sc)[0], "gravity_only_tipover")
    assert f.status == SOFT and f.values["failed"] == ["throttle"], f.values
    sc = Scene("tipover")
    sc.pwm[sc.w10(43.2, 43.9), 1] = 1600.0
    f = pick(run_scene(sc)[0], "gravity_only_tipover")
    assert f.status == SOFT and "pwm_below_hover" in f.values["failed"], f.values
    sc = Scene("tipover")
    sc.roll4[:] = 0.0
    sc.tip(43.0, dur=0.2)
    f = pick(run_scene(sc)[0], "gravity_only_tipover")
    assert f.status == NA and f.context == "window_too_short", f
    sc = Scene("tipover")
    sc.drop_tables.add("ANG")
    f = pick(run_scene(sc)[0], "gravity_only_tipover")
    assert f.status == HIT and abs(f.t_onset - (43.0 + np.sqrt(3 / 53))) <= 0.1, (f.status, f.t_onset)
    print("  推力參與 CLEAR / 油門高 SOFT / PWM 高 SOFT / 0.2 s 翻轉 NA / 只有 ATT(10 Hz)仍 HIT")


def test_syn_lateral_variants():
    """觸地水平速度:EKF/VIO 不一致單獨成立、有定位 GPS、舊值 GPS 不用、VISP 差分備援、SOFT 區、t_warn 選窗內第一個觸地。"""
    def lat(sc, a=None):
        return pick(run_scene(sc, a)[0], "touchdown_lateral_speed")

    sc = Scene("normal", peak=24.0)
    sc.pre_touch_speed(vio=0.33, ekf=0.0)
    f = lat(sc)
    assert f.status == HIT and f.values["v_max"] < dg.LAT_SPEED and f.values["ekf_vio_diff"] >= dg.LAT_MISMATCH, f.values
    sc = Scene("normal", peak=24.0)
    sc.pre_touch_speed(vio=0.1, ekf=0.1, gps=0.5)
    f = lat(sc)
    assert f.status == HIT and abs(f.values["v_gps"] - 0.5) < 1e-6, f.values
    sc = Scene("normal", peak=24.0)
    sc.gps = {"Status": np.full(sc.t5.size, 1.0), "NSats": np.zeros(sc.t5.size), "Spd": np.full(sc.t5.size, 14.76)}
    f = lat(sc)
    assert f.status == CLEAR and f.values["v_gps"] is None, f.values                       # 舊值 14.76 不能用
    sc = Scene("tipover")
    sc.drop_tables.add("VISV")
    f = lat(sc)
    assert f.status == HIT and abs(f.values["v_vio"] - 0.8) <= 0.05, f.values               # VISP 0.5 s 差分
    sc = Scene("normal", peak=24.0)
    sc.pre_touch_speed(vio=0.28, ekf=0.2)
    f = lat(sc)
    assert f.status == SOFT and "SOFT" in f.text_zh, f.values
    sc = Scene("normal", peak=24.0)
    f = lat(sc, sc.anchors(t_warn=20.0))
    assert f.status == NA and f.context == "no_touchdown_near_warn", f
    # 第一個觸地(40.0,觸地前 0.1 m/s)vs 0.6 s 後的彈跳(之前在滑 0.6 m/s):選第一個
    sc = Scene("tipover")
    sc.pre_touch_speed(vio=0.1, ekf=0.1)
    sc.vx_vio[(sc.t27 >= 40.05) & (sc.t27 < 40.6)] = 0.6
    sc.shock(40.62, 22.0)
    fs, a = run_scene(sc, sc.anchors(t_warn=41.0))
    f = pick(fs, "touchdown_lateral_speed")
    assert f.status == CLEAR and abs(f.values["t_touch"] - 40.0) <= 0.02, f.values
    assert abs(a.t_touch - 40.0) <= 0.02
    print("  不一致 HIT / GPS 0.5 HIT / 舊值 GPS 忽略 / VISP 差分 / 0.28 SOFT / 遠離 t_warn NA / 選第一個觸地")


def test_syn_phantom_variants():
    """EKF 地面幻影運動:VIO 也動了 → SOFT;VIO 中斷 → 仍 HIT(不交叉驗證);機身不靜止 → NA;t_loss 之後的漂移不算。"""
    sc = Scene("tipover")
    sc.vx_vio[(sc.t27 >= 40.5) & (sc.t27 < 41.5)] = 0.3                     # VIO 也移動 0.3 m:機身真的被推
    f = pick(run_scene(sc)[0], "ekf_phantom_motion_on_ground")
    assert f.status == SOFT and f.values["ekf_wrong"] is False and f.values["vio_disp_m"] >= dg.PH_VIO_DISP, f.values
    sc = Scene("tipover")
    ctx = sc.ctx()
    vp = ctx.tables["VISP"]
    keep = (vp["t"] < 40.0) | (vp["t"] > 44.0)
    ctx.tables["VISP"] = {k: v[keep] for k, v in vp.items()}
    f = pick(dg.run(ctx, sc.anchors()), "ekf_phantom_motion_on_ground")
    assert f.status == HIT and f.values["vio_disp_m"] is None, f.values
    sc = Scene("tipover")
    sc.gyr[:, 2] += np.radians(20.0)                                       # 一直在轉 20°/s:沒有靜止段
    f = pick(run_scene(sc)[0], "ekf_phantom_motion_on_ground")
    assert f.status == NA and f.context == "no_static_idle", f
    sc = Scene("normal", peak=24.0)
    sc.ekf_walk(43.0, 44.5, 0.4)
    f = pick(run_scene(sc, sc.anchors(t_loss=42.5))[0], "ekf_phantom_motion_on_ground")
    assert f.status == CLEAR, f
    sc = Scene("normal", peak=24.0)
    sc.ekf_walk(43.0, 44.5, 0.4)
    f = pick(run_scene(sc)[0], "ekf_phantom_motion_on_ground")
    assert f.status == HIT and abs(f.values["ekf_disp_m"] - 0.6) <= 0.05, f.values         # 同一段沒 t_loss 截止 → HIT
    print("  VIO 也動 SOFT / VIO 中斷 HIT / 不靜止 NA / t_loss 後不算(對照:不截止 → HIT)")


def test_syn_contract_guard_nan():
    """沒有區間 → 5 筆 NA not_armed;資料存取丟例外 → 5 筆 NA error;NaN 不會造成誤報。"""
    sc = Scene("normal", peak=24.0)
    fs = dg.run(sc.ctx(), Anchors())
    check_contract(fs, "no interval")
    assert all(f.status == NA and f.context == "not_armed" for f in fs)
    fs = dg.run(sc.ctx(boom=True), sc.anchors())
    check_contract(fs, "boom", allow_error=True)
    assert all(f.status == NA and f.context == "error" and "RuntimeError" in f.values["error"] for f in fs), statuses(fs)
    ctx = sc.ctx()
    ctx.tables["CTUN"]["ThO"][420:430] = np.nan
    ctx.tables["XKF1"]["VN"][1050:1070] = np.nan
    ctx.tables["IMU"]["AccZ"][17000:17010] = np.nan
    ctx.tables["RCOU"]["C2"][440:445] = np.nan
    fs = dg.run(ctx, sc.anchors())
    check_contract(fs, "nan")
    assert no_evidence(fs), statuses(fs)
    print("  沒區間 NA not_armed;例外 NA error;NaN 不誤報")


# ======================================================================================
# 7 審查缺陷回歸(review_ground)
# ======================================================================================

def _retakeoff_scene(mode, up=50.0):
    """td=40 降落後於 up 秒重新起飛到 5.5 m 懸停(up+5 s 到),95 s 上鎖。
    mode='landed'     :40 s 觸地被辨識(EV17 +0.6 / EV18 +1.4)、地面怠速到 up、EV28 起飛;
    mode='bounce'     :40 s 觸地沒收油門(touch-and-go,沒有 EV17/18、沒有 EV28);
    mode='unrec_quick':40 s 觸地油門收到 0 卻沒被辨識(像 124),up 秒直接爬升(飛控一直認為在空中,所以沒有 EV28)。"""
    sc = Scene("normal", td=40.0, dur=100.0, t1=95.0, peak=24.0)
    t = sc.t10
    sc.alt = np.interp(t, [0, 3, 8, 30, 40, up, up + 5, 100], [0.5, 0.5, 5.5, 5.5, 0.5, 0.5, 5.5, 5.5])
    if mode == "bounce":
        sc.tho = np.where(t >= 3, 0.15, 0.0)
    else:
        sc.tho = np.where((t >= 3) & (t <= 40), 0.15, 0.0)
        sc.tho[(t > 40) & (t < 40.3)] = 0.06
        sc.tho[t >= up] = 0.15
    sc.pwm[t >= up] = 1480.0
    sc.throut = sc.tho.copy()
    sc.evs = [(0.0, 10), (3.0, 28), (95.0, 11), (95.0005, 56)]
    if mode == "landed":
        sc.evs += [(40.6, 17), (41.4, 18), (up, 28)]
    sc.evs.sort()
    return sc


def test_review_t_touch_only_for_related_touchdown():
    """缺陷 1:較早的觸地(之後又起飛)不能當 a.t_touch。engine 拿 t_touch 當 t_first_anomaly 與情境判定時間點,
    用錯會把 80 s 在 5 m 的空中失控標成 GROUND_TIPOVER(之前 t_touch = 40.01、情境 GROUND_TIPOVER)。"""
    from core.crash import engine
    air = dict(t_warn=80.0, t_loss=80.5, t_first_hit=82.0, t_detect=84.0, end_type="crash_check")
    # (a) 40 s 正常降落(已辨識)→ 50 s EV28 再起飛 → 80 s 在 5 m 空中失控
    sc = _retakeoff_scene("landed")
    ctx = sc.ctx()
    a = sc.anchors(**air)
    fs = dg.run(ctx, a)
    check_contract(fs, "landed+retakeoff", a)
    f = pick(fs, "touchdown_not_recognised")
    assert a.t_touch is None, a.t_touch
    assert f.status == CLEAR and f.values["related_to_event"] is False and f.values["why"] == "far_from_event", f.values
    assert pick(fs, "touchdown_lateral_speed").context == "no_touchdown_near_warn"
    label, _, feats = engine._scenario(ctx, a, fs)
    assert label == "AIR_LOSS_OF_CONTROL" and abs(feats["agl_m"] - 5.0) <= 0.1, (label, feats)
    # (b) 40 s touch-and-go(油門沒收、沒有 EV17/18)→ 80 s 空中失控:未辨識的觸地和事件無關 → NA(不能 SOFT,也不能當錨點)
    sc = _retakeoff_scene("bounce")
    ctx = sc.ctx()
    a = sc.anchors(**air)
    fs = dg.run(ctx, a)
    check_contract(fs, "bounce+air", a)
    f = pick(fs, "touchdown_not_recognised")
    assert f.status == NA and f.context == "touchdown_unrelated" and a.t_touch is None, (f, a.t_touch)
    assert engine._scenario(ctx, a, fs)[0] == "AIR_LOSS_OF_CONTROL"
    _, a0 = run_scene(_retakeoff_scene("bounce"))                  # 沒有任何事件參考點:SOFT 的彈跳也不設
    assert a0.t_touch is None, a0.t_touch
    # (c) 觸地沒被辨識(油門收到 0)但 43 s 直接爬升(沒有 EV28),48 s 空中失控:在 10 s 窗內,仍因 agl>1.5 m 判定已重新起飛
    sc = _retakeoff_scene("unrec_quick", up=43.0)
    a = sc.anchors(t_warn=48.0, t_loss=48.5)
    fs = dg.run(sc.ctx(), a)
    f = pick(fs, "touchdown_not_recognised")
    assert f.status == NA and f.context == "touchdown_unrelated" and f.values["why"] == "took_off_again", f.values
    assert f.values["how"] == "agl" and 44.0 <= f.values["t_refly"] <= 45.5 and a.t_touch is None, f.values
    #     對照:同一個觸地、事件在再起飛之前(t_warn 42)→ HIT,t_touch = 觸地
    a = sc.anchors(t_warn=42.0)
    f = pick(dg.run(sc.ctx(), a), "touchdown_not_recognised")
    assert f.status == HIT and abs(f.t_onset - 40.0) <= 0.05 and a.t_touch is not None and abs(a.t_touch - f.t_onset) < 1e-9
    # (d) 已辨識降落、43 s EV28 再起飛、44.5 s 事件(在 10 s 窗內)→ EV28 判定無關;觸地速度也不拿這次觸地
    sc = _retakeoff_scene("landed", up=43.0)
    a = sc.anchors(t_warn=44.5)
    fs = dg.run(sc.ctx(), a)
    f = pick(fs, "touchdown_not_recognised")
    assert f.status == CLEAR and f.values["how"] == "EV28" and abs(f.values["t_refly"] - 43.0) < 1e-6 and a.t_touch is None, f.values
    assert "重新起飛" in f.text_zh, f.text_zh
    assert pick(fs, "touchdown_lateral_speed").context == "no_touchdown_near_warn"
    print("  降落→再起飛→空中失控:t_touch None、情境 AIR_LOSS_OF_CONTROL;彈跳 NA unrelated;agl / EV28 判定再起飛;對照 HIT")


def _bank(sc, t0, dur=3.0, peak=62.0):
    """空中協調急轉(指令內):roll 0 → peak → 0(dur 秒),推力平衡重力(AccZ = −g/cosφ、AccY = 0)。"""
    m = (sc.t4 >= t0) & (sc.t4 < t0 + dur)
    sc.roll4[m] = peak * np.sin(np.pi * (sc.t4[m] - t0) / dur)
    sc.desroll4[m] = sc.roll4[m]
    sc.acc[m, 2] = -G / np.cos(np.radians(sc.roll4[m]))
    sc.acc[m, 1] = 0.0


def _flip(sc, t0, dur=2.0):
    """空中 360° 翻滾(FLIP,有推力):傾斜一路經過 180°。"""
    m = (sc.t4 >= t0) & (sc.t4 < t0 + dur)
    roll = 360.0 * (sc.t4[m] - t0) / dur
    sc.roll4[m] = (roll + 180.0) % 360.0 - 180.0
    sc.desroll4[m] = sc.roll4[m]
    sc.acc[m, 2] = -G * np.cos(np.radians(roll)) - 3.0
    sc.acc[m, 1] = G * np.sin(np.radians(roll))


def test_review_air_tilt_does_not_hide_ground_tipover():
    """缺陷 2:15 s 空中急轉 62° / FLIP 360°,40 s 觸地後在地面翻覆:地面證據要和沒有空中大角度時完全一樣
    (之前:60° 截止落在 15 s → idle / phantom CLEAR 假排除;翻倒視窗選到空中那次 → CLEAR not_gravity_only / NA)。"""
    eng = dict(t_warn=41.0, t_loss=43.95)
    ref = Scene("tipover")
    fs0, _ = run_scene(ref, ref.anchors(**eng))
    assert all(f.status == HIT for f in fs0), statuses(fs0)
    base = {f.id: (f.status, f.context, f.t_onset) for f in fs0}
    for label, edit in (("bank", lambda s: _bank(s, 15.0)), ("flip", lambda s: _flip(s, 15.0))):
        sc = Scene("tipover")
        edit(sc)
        fs, a = run_scene(sc, sc.anchors(**eng))
        check_contract(fs, label, a)
        got = {f.id: (f.status, f.context, f.t_onset) for f in fs}
        assert got == base, (label, got, base)
        g = pick(fs, "gravity_only_tipover")
        assert g.values["n_air_tilt50_before"] > 0 and abs(g.values["t_start"] - (43.0 + np.sqrt(3 / 53))) <= 0.01, g.values
        fs, a = run_scene(sc)                                    # 只有區間(沒有 t_loss):本地 60° 截止也不能落在空中
        assert all(f.status == HIT for f in fs), (label, statuses(fs))
        assert abs(a.t_touch - 40.0) <= 0.05
    # 只在空中傾斜 ≥50°、之後正常降落 → NA airborne_tilt(之前 CLEAR『有推力參與,不是重力翻倒』)
    sc = Scene("normal", peak=24.0)
    _bank(sc, 15.0)
    f = pick(run_scene(sc)[0], "gravity_only_tipover")
    assert f.status == NA and f.context == "airborne_tilt" and f.values["agl_m"] > 4.0, f
    # 從 5 m 空中開始翻滾、落到地面時已 ≥50°:視窗起點在空中 → NA airborne_tilt
    sc = Scene("normal", t1=30.0, peak=24.0)
    sc.evs = [(0.0, 10), (3.0, 28), (30.0, 11), (30.0005, 56)]
    sc.alt = np.interp(sc.t10, [0, 3, 8, 20, 22, 30], [0.5, 0.5, 5.5, 5.5, 0.5, 0.5])
    m = sc.t4 >= 20.0
    sc.roll4[m] = np.minimum(45.0 * (sc.t4[m] - 20.0), 90.0)
    f = pick(run_scene(sc)[0], "gravity_only_tipover")
    assert f.status == NA and f.context == "airborne_tilt" and f.values["agl_start_m"] > 4.0, f
    print("  空中急轉 / FLIP 後地面翻覆:5 筆和對照相同(HIT);只在空中 ≥50° → NA airborne_tilt;空中翻滾落地 → NA")


def test_review_idle_air_throttle_cut_not_counted():
    """缺陷 3:空中(5 m、未落地)收油門時的不對稱不是 idle_motor_asymmetry(之前 HIT 'ground_idle'、文字說在地面撬翻)。"""
    sc = Scene("normal", peak=24.0)
    sc.tho[sc.w10(20.0, 21.5)] = 0.0
    sc.asym(20.0, 21.5)                                          # 空中左 1450 / 右 1235、油門補償 0.075
    fs, a = run_scene(sc)
    f = pick(fs, "idle_motor_asymmetry")
    assert f.status == CLEAR and f.context == "ground_idle", f
    assert abs(f.values["air_idle_excluded_s"] - 1.5) <= 0.15 and f.values["air_idle_flag_longest_s"] >= 1.3, f.values
    assert "空中怠速不納入" in f.text_zh, f.text_zh
    fs, a = run_scene(sc, sc.anchors(t_warn=21.6, t_loss=22.0))  # 之後空中失控:也不能有地面怠速證據
    assert pick(fs, "idle_motor_asymmetry").status not in (HIT, SOFT), statuses(fs)
    # 起飛前馬達沒轉起來(1100 µs)、整段只有空中怠速 → NA air_idle_only(不是 CLEAR,也不是 HIT)
    sc = Scene("normal", t1=30.0, peak=24.0)
    sc.evs = [(0.0, 10), (3.0, 28), (30.0, 11), (30.0005, 56)]
    sc.pwm[sc.t10 < 3.0] = 1100.0
    sc.tho[sc.w10(20.0, 21.5)] = 0.0
    sc.asym(20.0, 21.5)
    f = pick(run_scene(sc)[0], "idle_motor_asymmetry")
    assert f.status == NA and f.context == "air_idle_only" and f.values["air_idle_agl_max_m"] > 4.0, f
    # 四軸空中收油門、只有油門補償條件成立(未落地時 ThrOut 被夾在 0.5×hover)→ 仍不算
    sc = Scene("normal", peak=24.0, n_motors=4)
    sc.params["FRAME_CLASS"] = 1.0
    sc.tho[sc.w10(20.0, 21.5)] = 0.0
    sc.throut[sc.w10(20.0, 21.5)] = 0.075
    assert pick(run_scene(sc)[0], "idle_motor_asymmetry").status == CLEAR
    # 對照:同樣的不對稱發生在地面(落地後 42 s)→ HIT
    sc = Scene("normal", peak=24.0)
    sc.asym(42.0, 43.0)
    assert pick(run_scene(sc)[0], "idle_motor_asymmetry").status == HIT
    print("  空中收油門不對稱 → 不計(CLEAR / NA air_idle_only);地面同樣不對稱 → HIT")


def test_review_nan_never_clears_or_downgrades():
    """缺陷 4:單一 NaN 不能把 HIT 變成 CLEAR / SOFT,text_zh 不能出現 nan;IMU 全 NaN → NA no_imu(不能說沒有衝擊)。"""
    def run_mod(fn):
        sc = Scene("tipover")
        ctx = sc.ctx()
        fn(ctx.tables)
        a = sc.anchors()
        fs = dg.run(ctx, a)
        check_contract(fs, "nan", a)
        # format(nan) 會印出小寫 'nan';NA 說明裡刻意寫的『NaN』不算
        assert all("nan" not in f.text_zh for f in fs), [f.text_zh for f in fs if "nan" in f.text_zh]
        return fs

    def at(tb, t):
        return int(np.searchsorted(tb["t"], t))

    fs = run_mod(lambda T: T["IMU"]["AccZ"].__setitem__(at(T["IMU"], 43.5), np.nan))   # 翻倒視窗 43.24–43.95 內
    g = pick(fs, "gravity_only_tipover")
    assert g.status == HIT and g.values["n_invalid_imu"] == 1 and g.values["rms_ms2"] < 0.1, g.values
    fs = run_mod(lambda T: T["VISV"]["VX"].__setitem__(at(T["VISV"], 39.5), np.nan))   # 觸地前 1 s
    f = pick(fs, "touchdown_lateral_speed")
    assert f.status == HIT and abs(f.values["v_vio"] - 0.8) < 1e-6 and abs(f.values["ekf_vio_diff"] - 0.5) < 1e-6, f.values

    def ekf_nan(T):
        T["XKF1"]["VN"][at(T["XKF1"], 41.0)] = np.nan
        T["XKF1"]["PN"][at(T["XKF1"], 43.1)] = np.nan
    f = pick(run_mod(ekf_nan), "ekf_phantom_motion_on_ground")
    assert f.status == HIT and abs(f.values["ekf_speed_med"] - 0.4) < 1e-6 and f.values["ekf_disp_m"] > 0.9, f.values
    # 靜止段內一個 IMU NaN:cumsum 不能把之後所有視窗都變成『不靜止』
    f = pick(run_mod(lambda T: T["IMU"]["GyrX"].__setitem__(at(T["IMU"], 40.6), np.nan)), "ekf_phantom_motion_on_ground")
    assert f.status == HIT and abs(f.t_onset - 40.2) <= 0.1, (f.status, f.context, f.t_onset)

    def imu_all_nan(T):
        for c in ("AccX", "AccY", "AccZ"):
            T["IMU"][c][:] = np.nan
    fs = run_mod(imu_all_nan)
    assert [pick(fs, i).context for i in ("touchdown_not_recognised", "gravity_only_tipover", "touchdown_lateral_speed",
                                          "ekf_phantom_motion_on_ground")] == ["no_imu"] * 4, statuses(fs)
    print("  AccZ / VISV / XKF1 / GyrX 單點 NaN 仍 HIT、文字無 nan;IMU 全 NaN → NA no_imu")


def test_review_idle_text_names_trigger():
    """缺陷 5:文字說出實際觸發的條件;只有油門補償時不能說『全距達 12 µs』或『撬起翻倒』;馬達說明附沒有 ESC 遙測。"""
    sc = Scene("normal", peak=24.0)                              # 已辨識落地後,只有油門補償 0.06
    sc.throut[sc.w10(42.0, 43.0)] = 0.06
    f = pick(run_scene(sc)[0], "idle_motor_asymmetry")
    first = f.text_zh.split("。")[0]
    assert f.status == HIT and f.values["criteria"] == ["boost"], f.values
    assert "MOTB 油門輸出比 ThO 高 0.060" in first and "全距" not in first and "撬起" not in f.text_zh, f.text_zh
    assert "已判定落地" in f.text_zh and "沒有 ESC" in f.text_zh and "不是馬達故障" not in f.text_zh, f.text_zh
    sc = Scene("tipover")                                        # 未辨識落地、PWM 對稱、只有油門補償 → 說明『仍認為在空中』
    sc.pwm[sc.w10(41.0, 43.5)] = 1235.0
    f = pick(run_scene(sc)[0], "idle_motor_asymmetry")
    assert f.status == HIT and f.values["criteria"] == ["boost"] and "仍認為在空中" in f.text_zh, (f.values, f.text_zh)
    f = pick(run_scene(Scene("tipover"))[0], "idle_motor_asymmetry")   # 全距 + 力矩都超標 → 說明側傾撬翻
    assert {"spread", "roll_mom"} <= set(f.values["criteria"]) and "撬起" in f.text_zh and "左側 1450" in f.text_zh, f.text_zh
    assert f.values["esc_telemetry"] is False
    print("  只有油門補償 → 文字只講補償(落地 / 未落地各有說明);全距 + 力矩 → 側傾撬翻;附無 ESC 遙測")


def test_review_open_end_interval():
    """缺陷 6:log 在解鎖中結束(open_end):文字說『直到 log 結束』不說『直到上鎖』;區間末端 1 ms 內的 EV17 照常算。"""
    sc = Scene("tipover")
    sc.evs = [e for e in sc.evs if e[1] not in (11, 17, 18, 56)]
    iv = ArmedInterval(0, 0.0, 48.0, None, False, open_end=True)
    a = Anchors(interval=iv, intervals=[iv])
    fs = dg.run(sc.ctx(), a)
    check_contract(fs, "open_end", a)
    f = pick(fs, "touchdown_not_recognised")
    assert f.status == HIT and "直到 log 結束" in f.text_zh and "直到上鎖" not in f.text_zh and f.values["open_end"] is True, f.text_zh
    sc.evs += [(47.9995, 17)]
    f = pick(dg.run(sc.ctx(), Anchors(interval=iv, intervals=[iv])), "touchdown_not_recognised")
    assert f.status == HIT and f.values["late_recognition"] is True and f.values["ev_ignored_at_disarm"] == [], f.values
    iv2 = ArmedInterval(0, 0.0, 48.0, 12)                         # 對照:真的上鎖 → 同一時刻的 EV17 是上鎖寫的
    f = pick(dg.run(sc.ctx(), Anchors(interval=iv2, intervals=[iv2])), "touchdown_not_recognised")
    assert f.values["late_recognition"] is False and f.values["ev_ignored_at_disarm"] == [17], f.values
    print("  open_end:『直到 log 結束』、末端 EV17 算延遲辨識;一般上鎖 → 忽略")


def test_review_phantom_vio_gap_reported():
    """缺陷 7:觸地時 VIO 中斷時,『觸地後漂移』要說 VIO 只涵蓋中斷之後,並給同段 EKF 對照(124 數字見 truth 測試)。"""
    sc = Scene("tipover")
    ctx = sc.ctx()
    vp = ctx.tables["VISP"]
    keep = (vp["t"] < 39.9) | (vp["t"] > 41.0)
    ctx.tables["VISP"] = {k: v[keep] for k, v in vp.items()}
    f = pick(dg.run(ctx, sc.anchors()), "ekf_phantom_motion_on_ground")
    v = f.values
    assert f.status == HIT and 39.85 <= v["vio_gap_t0"] < 39.9 and 41.0 < v["vio_gap_t1"] <= 41.05, v
    assert v["vio_span_t0"] == v["vio_gap_t1"] and v["ekf_drift_vio_span_m"] < v["ekf_drift_since_touch_m"], v
    assert "中斷" in f.text_zh and "同段 EKF" in f.text_zh, f.text_zh
    f = pick(run_scene(Scene("tipover"))[0], "ekf_phantom_motion_on_ground")      # 沒有中斷:不提中斷
    assert f.values["vio_gap_s"] is None and "中斷" not in f.text_zh.split("這是")[0], f.text_zh
    print(f"  VIO 中斷 {v['vio_gap_t0']:.2f}–{v['vio_gap_t1']:.2f} s:VIO 只涵蓋之後,同段 EKF {v['ekf_drift_vio_span_m']} m "
          f"/ 全段 {v['ekf_drift_since_touch_m']} m")


# ======================================================================================
# 5 小工具
# ======================================================================================

def test_find_shocks_merge_and_dv():
    """衝擊簇:間隔 ≤0.3 s 合併、>0.3 s 分開;dV 只積 |a|>g 的部分,IMU 空洞不灌成衝量;分類界線。"""
    t = np.arange(0, 5, 1 / 400.0)
    a = np.full(t.size, G)
    for c, pk in ((1.0, 30.0), (1.2, 25.0), (2.0, 22.0)):
        a += (pk - G) * np.clip(1.0 - np.abs(t - c) / 0.03, 0, None)
    sh = dg.find_shocks(t, a)
    assert len(sh) == 2 and abs(sh[0].peak - 30.0) < 0.1 and sh[0].t_end > 1.15 and sh[0].cls == "ground_contact", sh
    exp = 0.5 * (30 - G) * 0.06 + 0.5 * (25 - G) * 0.06
    assert abs(sh[0].dv - exp) < 0.02, (sh[0].dv, exp)
    # 空洞:兩個 >20 的樣本中間缺 0.2 s 的資料,不能把 0.2 s 積成 dV
    t2 = np.array([0.0, 0.0025, 0.2025, 0.205])
    sh2 = dg.find_shocks(t2, np.array([25.0, 25.0, 25.0, 25.0]))
    assert len(sh2) == 1 and sh2[0].dv <= (25 - G) * (0.0025 * 2 + dg.IMU_DV_DT_MAX) + 1e-9, sh2[0].dv
    assert dg.Shock(0, 0.3, 0.1, 62.0, 4.6).cls == "hard_impact" and dg.Shock(0, 0.3, 0.1, 35.0, 3.5).cls == "long_shock"
    assert dg.find_shocks(t, np.full(t.size, G)) == []
    print(f"  合併 / 分開 OK;dV {sh[0].dv:.2f}(理論 {exp:.2f});空洞不灌水;hard / long / ground 分類")


def test_static_windows_and_spearman():
    """滑動靜止視窗不受分箱對齊影響(0.6 s 靜止段 ≥0.5 s);Spearman 對單調序列 = 1。"""
    t = np.arange(0, 3, 1 / 400.0)
    g = np.where((t >= 1.03) & (t < 1.63), 1.0, 30.0)                      # 0.6 s 靜止,刻意不對齊 0.2 s 邊界
    an = np.full(t.size, G)
    ts, te, nn, gm, am, asd = dg._static_windows(t, an, g, 0.0)
    ok = gm < dg.PH_GYRO_DEG
    k = np.flatnonzero(ok)
    assert k.size and te[k[-1]] - ts[k[0]] >= dg.PH_MIN_S - dg.DUR_TOL, (ts[k[0]], te[k[-1]])
    assert abs(dg._spearman(np.array([1.0, 2.0, 5.0, 9.0, 20.0])) - 1.0) < 1e-12
    assert dg._spearman(np.array([1.0, 2.0])) != dg._spearman(np.array([1.0, 2.0]))          # NaN(樣本太少)
    print(f"  靜止段 {ts[k[0]]:.2f}–{te[k[-1]]:.2f} s(≥0.5 s);Spearman OK")


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
