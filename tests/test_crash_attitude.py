"""姿態 / 控制偵測模組測試(core/crash/det_attitude.py):python tests/test_crash_attitude.py 或 pytest tests/test_crash_attitude.py

涵蓋:
  1. 00000124 飛行 2(墜機)+ 真相錨點 / 只有 basic_anchors:六個偵測器的狀態、關鍵時間(驗收 A4 / A6 / A7)、
     錨點 t_warn / t_loss / t_inverted 由本模組寫入;engine 端 lean_without_pilot 有算進 H_LAND_TIPOVER(context 契約);
  2. 00000124 飛行 1(正常降落)與飛行 2 的 1255 s 之前:沒有 HIT / SOFT(驗收 B,MUST NOT);
  3. 6 個 baseline(從未解鎖):全部 NA 'not_armed';把閘門拿掉(強制整段當解鎖)則 117/119/120 會誤報(驗收 D1);
  4. 真實資料單元:推力向量誤差重現 AngErr 150、Roll ±180° 折返不跳、沒有 ANG 時退回 ATT(10 Hz 容許 ±0.15 s);
  5. 合成資料(假 Ctx,不同 trim / 半程 / 懸停油門 / 時間基準):證明邏輯不是寫死 124 的數字;
  6. 每個偵測器 id 在每種情境都至少回一筆;values 可 JSON 序列化;例外 → NA;決定性;耗時 < 1.5 s;
  7. 審查缺陷回歸:駕駛前兆窗 60 s(較早動作只列參考)、多段事件挑「這次失事的那一段」(較早的空中事件不能蓋掉
     失事當下的地面證據,engine 分數不變)、地面怠速需離地 ≤ 1.5 m、RCIN NaN 不能 CLEAR、油門背景窗裁在上鎖前、
     Loiter 煞車期排除、已恢復的陣風不當 t_warn、文字不寫死 124 的數字。
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
from core.crash import det_attitude as da  # noqa: E402
from core.crash import util  # noqa: E402
from core.crash.anchors import basic_anchors  # noqa: E402
from core.crash.ctx import Ctx  # noqa: E402
from core.crash import hypotheses  # noqa: E402
from core.crash.types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval  # noqa: E402

OWNED = ("attitude_loss_of_control", "rate_demand_not_achieved", "rate_output_saturation",
         "lean_without_pilot", "pilot_input_context", "uncommanded_yaw_or_sink")
DETECTORS = (da.attitude_loss_of_control, da.rate_demand_not_achieved, da.rate_output_saturation,
             da.lean_without_pilot, da.pilot_input_context, da.uncommanded_yaw_or_sink)


# ======================================================================================
# 共用
# ======================================================================================

def pick(fs, det):
    c = [f for f in fs if f.id == det]
    assert c, f"{det} 沒有輸出:{[(f.id, f.status, f.context) for f in fs]}"
    return c[0]


def statuses(fs):
    return [(f.id, f.status, f.context) for f in fs]


def near(x, ref, tol, what=""):
    assert x is not None and abs(float(x) - ref) <= tol, f"{what}:{x} 不在 {ref} ± {tol}"


def check_contract(fs, label):
    """每個 id 恰好一筆、domain 正確、沒有 error、values 可 JSON 序列化、text_zh 非空、HIT/SOFT 的 t_onset ≤ t_alarm。"""
    ids = [f.id for f in fs]
    assert sorted(ids) == sorted(OWNED), f"{label}:id 不齊 {ids}"
    for f in fs:
        assert f.domain == "attitude", (label, f.id, f.domain)
        assert f.status in (HIT, SOFT, CLEAR, NA), (label, f.status)
        assert f.severity in ("critical", "high", "medium", "info"), (label, f.id, f.severity)
        assert f.context != "error", f"{label}:{f.id} 執行失敗 {f.values}"
        json.dumps(f.values, allow_nan=False)
        assert f.text_zh, (label, f.id, "text_zh 空")
        assert "None" not in f.text_zh and "nan" not in f.text_zh, (label, f.id, f.text_zh)
        if f.status in (HIT, SOFT):
            assert f.t_onset is not None and f.t_alarm is not None and f.t_onset <= f.t_alarm + 1e-9, (label, f)
        else:
            assert f.t_onset is None, (label, f.id, "CLEAR/NA 不該有 t_onset")
    return {f.id: f for f in fs}


def run_checked(ctx, a, label):
    return check_contract(da.run(ctx, a), label)


# ======================================================================================
# 合成資料:假 Ctx(沿用真正的 Ctx 方法:landed_steps / mode_rows / ev_times / armed_mask / ground_alt)
# ======================================================================================

FAST, SLOW = 400.0, 10.0
AIR_LEAN = da.AIR_LEAN_DEG
HALF = 400.0                      # 合成用 RC 半程(RCn_MIN/MAX 1100/1900);124 是 420


class FakeCtx(Ctx):
    """不讀檔的 Ctx:表格直接給(保留原始 dtype,用來驗證 uint16 不回繞),series 由表格切出。"""

    def __init__(self, tables, params=None, armed=(), boom=False):  # noqa: D107 — 刻意不呼叫 Ctx.__init__
        self.ld = None
        self.ix = None
        self.t_base = 0.0
        self._tables = {(k, None): v for k, v in tables.items()}
        self._series = {}
        self.params = dict(params or {})
        self.messages = []
        self.events = []
        self.armed = list(armed)
        self.t_end = max((float(v["t"][-1]) for v in tables.values() if len(v["t"])), default=0.0)
        self.boom = boom

    def table(self, name, instance=None):
        if self.boom:
            raise RuntimeError("boom")
        return self._tables.get((name, None))

    def series(self, key):
        name, col = key.split(".", 1)
        tb = self.table(name)
        if tb is None or col not in tb:
            return None
        return np.asarray(tb["t"], dtype=float), np.asarray(tb[col], dtype=float)


class Scene:
    """合成一次 LOITER 飛行(相對秒):解鎖 t_arm → 起飛(EV28)→ 空中 → 觸地 t_land(EV17 / EV18)→ 上鎖。

    故意和 124 不同:RC trim 1490 / 1530 / 1505、半程 400 µs、懸停油門 0.35、四軸、時間基準可平移。
    400 Hz:ANG / RATE / IMU;10 Hz:ATT / CTUN / RCIN / RCOU / PSCN / PSCE。
    """

    def __init__(self, t_arm=20.0, takeoff=6.0, flight=60.0, after_land=5.0, hover=0.35, alt_air=10.0,
                 amp=1.0, trims=(1490.0, 1530.0, 1100.0, 1505.0), seed=7):
        rng = self.rng = np.random.default_rng(seed)
        self.t_arm = float(t_arm)
        self.t_to = self.t_arm + takeoff
        self.t_land = self.t_to + flight
        self.t_disarm = self.t_land + after_land
        self.trims = trims
        self.hover = hover
        t0 = self.t_arm - 5.0
        span = self.t_disarm + 5.0 - t0
        tf = self.tf = t0 + np.arange(int(round(span * FAST))) / FAST
        ts = self.ts = t0 + np.arange(int(round(span * SLOW))) / SLOW
        # ---- 姿態 / 角速度(空中平緩操作,機身跟著指令)----
        env = np.clip((tf - self.t_to - 2) / 2, 0, 1) * np.clip((self.t_land - 2 - tf) / 2, 0, 1) * amp
        self.droll = 5 * np.sin(2 * np.pi * 0.1 * tf) * env
        self.dpitch = 4 * np.sin(2 * np.pi * 0.07 * tf + 1) * env
        self.roll = self.droll + rng.normal(0, 0.3, tf.size)
        self.pitch = self.dpitch + rng.normal(0, 0.3, tf.size)
        self.dyaw = (40 + 20 * np.sin(2 * np.pi * 0.01 * tf)) % 360
        self.yaw = (self.dyaw + rng.normal(0, 0.3, tf.size)) % 360
        self.rdes = np.gradient(self.droll, tf)
        self.r = 0.9 * self.rdes + rng.normal(0, 0.5, tf.size)
        self.pdes = np.gradient(self.dpitch, tf)
        self.p = 0.9 * self.pdes + rng.normal(0, 0.5, tf.size)
        self.ydes = np.degrees(np.gradient(np.unwrap(np.radians(self.dyaw)), tf))
        self.y = self.ydes + rng.normal(0, 0.5, tf.size)
        self.rout = 0.003 * self.rdes + rng.normal(0, 0.01, tf.size)
        self.pout = 0.003 * self.pdes + rng.normal(0, 0.01, tf.size)
        self.yout = 0.003 * self.ydes + rng.normal(0, 0.01, tf.size)
        self.acc = np.stack([rng.normal(0, 0.2, tf.size), rng.normal(0, 0.2, tf.size), -9.8 + rng.normal(0, 0.3, tf.size)])
        # ---- 10 Hz ----
        air_s = (ts >= self.t_to) & (ts < self.t_land)
        self.tho = np.where(air_s, hover, 0.0)
        ramp = np.clip((ts - self.t_to) / 10.0, 0, 1) * np.clip((self.t_land - ts) / 10.0, 0, 1)
        self.alt = 0.5 + alt_air * ramp
        self.crt = np.gradient(self.alt, ts) * 100.0
        self.dcrt = self.crt + rng.normal(0, 2.0, ts.size)
        self.c1 = trims[0] + rng.normal(0, 4.0, ts.size)
        self.c2 = trims[1] + rng.normal(0, 4.0, ts.size)
        self.c3 = np.where(air_s, 1500.0, 1100.0)
        self.c4 = trims[3] + rng.normal(0, 3.0, ts.size)
        self.c4[(ts >= self.t_arm - 2) & (ts <= self.t_arm + 1)] = 1900.0          # 解鎖手勢(偏航滿舵,地面)
        self.c9 = np.full(ts.size, 1095.0)
        armed = (ts >= self.t_arm) & (ts <= self.t_disarm)
        self.pwm = np.where(armed, np.where(air_s, 1500.0, 1190.0), 1100.0)
        self.evs = [(self.t_arm, 10), (self.t_to, 28), (self.t_land + 0.8, 17), (self.t_land + 1.5, 18),
                    (self.t_disarm, 11)]
        self.modes = [(t0, 5)]
        self.params = {"RC1_TRIM": trims[0], "RC2_TRIM": trims[1], "RC3_TRIM": trims[2], "RC4_TRIM": trims[3],
                       "RC1_MIN": 1100.0, "RC1_MAX": 1900.0, "RC2_MIN": 1100.0, "RC2_MAX": 1900.0,
                       "RC3_MIN": 1100.0, "RC3_MAX": 1900.0, "RC9_OPTION": 90.0, "MOT_THST_HOVER": hover,
                       "PILOT_SPEED_DN": 250.0, **{f"SERVO{n}_FUNCTION": 32.0 + n for n in range(1, 5)}}

    # ---------------- 輸出 ----------------

    def tables(self):
        tf, ts = self.tf, self.ts
        i_att = np.clip(np.searchsorted(tf, ts - 1e-9), 0, tf.size - 1)
        dr_s, dp_s = np.interp(ts, tf, self.droll), np.interp(ts, tf, self.dpitch)
        z = np.zeros(ts.size)
        ang = {"t": tf, "Roll": self.roll, "Pitch": self.pitch, "Yaw": self.yaw,
               "DesRoll": self.droll, "DesPitch": self.dpitch, "DesYaw": self.dyaw}
        ev = sorted(self.evs)
        md = sorted(self.modes)
        return {
            "ANG": {k: v.copy() for k, v in ang.items()},
            "ATT": {k: v[i_att].copy() for k, v in ang.items()},
            "RATE": {"t": tf, "RDes": self.rdes, "R": self.r, "PDes": self.pdes, "P": self.p, "YDes": self.ydes,
                     "Y": self.y, "ROut": self.rout, "POut": self.pout, "YOut": self.yout},
            "CTUN": {"t": ts, "ThO": self.tho, "ThH": np.full(ts.size, self.hover), "Alt": self.alt,
                     "CRt": self.crt, "DCRt": self.dcrt},
            "RCIN": {"t": ts, "C1": self.c1, "C2": self.c2, "C3": self.c3, "C4": self.c4, "C9": self.c9},
            "RCOU": {"t": ts, **{f"C{n}": self.pwm.copy() for n in range(1, 5)}},
            "PSCN": {"t": ts, "TAN": -9.81 * np.tan(np.radians(dp_s)), "PN": z, "TPN": z},
            "PSCE": {"t": ts + 1e-4, "TAE": 9.81 * np.tan(np.radians(dr_s)), "PE": z, "TPE": z},
            "MODE": {"t": np.array([m[0] for m in md]), "ModeNum": np.array([float(m[1]) for m in md])},
            "EV": {"t": np.array([e[0] for e in ev]), "Id": np.array([float(e[1]) for e in ev])},
            "IMU": {"t": tf, "AccX": self.acc[0], "AccY": self.acc[1], "AccZ": self.acc[2]},
        }

    def interval(self):
        return ArmedInterval(0, self.t_arm, self.t_disarm, 0)

    def ctx(self, drop=(), boom=False, u16=False):
        tb = {k: v for k, v in self.tables().items() if k not in drop}
        if u16:
            for name in ("RCIN", "RCOU"):
                if name in tb:
                    for c in list(tb[name]):
                        if c != "t":
                            tb[name][c] = np.round(tb[name][c]).astype(np.uint16)
            tb["EV"]["Id"] = tb["EV"]["Id"].astype(np.uint8)
            tb["MODE"]["ModeNum"] = tb["MODE"]["ModeNum"].astype(np.uint8)
        return FakeCtx(tb, self.params, [self.interval()], boom=boom)

    def anchors(self, end_type="normal_landing", t_first_hit=None):
        iv = self.interval()
        a = Anchors(interval=iv, intervals=[iv])
        a.t_disarm = iv.t1
        a.end_type = end_type
        a.t_first_hit = t_first_hit
        return a

    # ---------------- 情境 ----------------

    def tipover(self):
        """觸地 T 後未判定落地(沒有 EV17/18,上鎖那一刻才有):搖桿置中,位置控制器把傾角指令拉到 18°,
        機身躺在地上不動(rate 不跟、ROut 偏大),T+2.5 起翻覆,T+6 CRASH 上鎖。"""
        T = self.t_land
        self.t_disarm = T + 6.0
        self.evs = [(self.t_arm, 10), (self.t_to, 28), (self.t_disarm, 11), (self.t_disarm + 6e-5, 18),
                    (self.t_disarm + 7e-5, 17)]
        tf, rng = self.tf, self.rng
        x = tf - T
        g = x >= 0
        n = rng.normal(0, 0.2, tf.size)
        self.droll = np.where(g, np.clip(12.0 * (x - 0.5), 0.0, 18.0), self.droll)
        self.dpitch = np.where(g, 0.0, self.dpitch)
        self.roll = np.where(g, np.where(x < 2.5, n, np.minimum(178.0, 60.0 * (x - 2.5) ** 2)), self.roll)
        self.pitch = np.where(g, 0.5 * n, self.pitch)
        self.rdes = np.where(g, np.where((x >= 0.8) & (x < 2.5), 50.0, np.where(x >= 2.5, -80.0, 0.0)), self.rdes)
        self.r = np.where(g, np.where(x < 2.5, rng.normal(0, 1.0, tf.size), np.gradient(self.roll, tf)), self.r)
        self.pdes = np.where(g, 0.0, self.pdes)
        self.p = np.where(g, rng.normal(0, 0.5, tf.size), self.p)
        self.rout = np.where(g, np.where((x >= 1.0) & (x < 2.5), 0.25, np.where(x >= 2.5, -1.1, 0.0))
                             + rng.normal(0, 0.005, tf.size), self.rout)
        post = tf > self.t_disarm                     # 上鎖後:倒扣的垃圾姿態,不能被分析
        self.roll[post], self.droll[post] = 179.0, 0.0
        return self

    def air_loss(self, T_f):
        """空中 T_f 起機身自己往一側翻(指令反向修正但追不上)、輸出飽和、掉高度;T_f+3 上鎖。"""
        self.t_disarm = T_f + 3.0
        self.evs = [(self.t_arm, 10), (self.t_to, 28), (self.t_disarm, 11)]
        tf, ts, rng = self.tf, self.ts, self.rng
        x = tf - T_f
        g = x >= 0
        self.droll = np.where(g, -np.minimum(10.0, 20.0 * x), self.droll)
        self.roll = np.where(g, np.minimum(178.0, 40.0 * x ** 2), self.roll)
        self.dpitch = np.where(g, 0.0, self.dpitch)
        self.pitch = np.where(g, rng.normal(0, 0.2, tf.size), self.pitch)
        self.rdes = np.where(g, -80.0, self.rdes)
        self.r = np.where(g, np.gradient(self.roll, tf), self.r)
        self.rout = np.where(x >= 0.1, -1.2, self.rout)
        xs = ts - T_f
        self.dcrt = np.where(xs >= 0, 0.0, self.dcrt)
        self.crt = np.where(xs >= 0.5, -400.0, np.where(xs >= 0, 0.0, self.crt))
        self.alt = np.where(xs >= 0.5, np.maximum(0.5, self.alt - 4.0 * (xs - 0.5)), self.alt)
        ff = (x >= 0.05) & (x <= 0.45)
        self.acc[:, ff] = np.array([[0.1], [0.1], [-1.0]])          # 起點前的失重(自由落體佐證)
        return self

    def flip(self, T_f, how="none"):
        """T_f 起 1 s 的滾轉 360°(指令領先 40°、rate 不跟、輸出 1.0)。how:'acro' 模式 / 'ev' EV21-22 / 'none'。"""
        x = self.tf - T_f
        w = (x >= 0) & (x < 1.0)
        ph = 360.0 * x
        self.roll = np.where(w, util.wrap180(ph), self.roll)
        self.droll = np.where(w, util.wrap180(ph + 40.0), self.droll)
        self.pitch = np.where(w, 0.0, self.pitch)
        self.dpitch = np.where(w, 0.0, self.dpitch)
        self.rdes = np.where(w, 400.0, self.rdes)
        self.r = np.where(w, 60.0, self.r)
        self.rout = np.where(w, 1.0, self.rout)
        if how == "acro":
            self.modes = [(self.tf[0], 5), (T_f - 1.0, 1), (T_f + 2.0, 5)]
        elif how == "ev":
            self.evs = self.evs + [(T_f - 0.05, 21), (T_f + 1.0, 22)]
        return self

    def lean_air(self, T_l, deg=15.0, dur=2.0, mode=None, stick=0.0):
        """空中搖桿置中(或偏 stick µs),傾角指令 0.3 s 內爬到 deg 並維持 dur 秒,機身照做(ae 小)。"""
        tf, ts, rng = self.tf, self.ts, self.rng
        x = tf - T_l
        w = (x >= 0) & (x <= dur)
        prof = np.where(w, np.minimum(np.clip(x / 0.3, 0, 1), np.clip((dur - x) / 0.3, 0, 1)) * deg, 0.0)
        self.droll = np.where(w, prof, self.droll)
        self.roll = np.where(w, prof + rng.normal(0, 0.3, tf.size), self.roll)
        w2 = (x >= -0.1) & (x <= dur + 0.1)
        rd = np.gradient(self.droll, tf)
        self.rdes = np.where(w2, rd, self.rdes)
        self.r = np.where(w2, 0.9 * rd + rng.normal(0, 0.5, tf.size), self.r)
        self.rout = np.where(w2, 0.003 * rd + rng.normal(0, 0.01, tf.size), self.rout)
        if mode is not None:
            self.modes = [(self.tf[0], 5), (T_l - 2.0, mode), (T_l + dur + 2.0, 5)]
        if stick:
            m = (ts >= T_l - 0.5) & (ts <= T_l + dur + 0.5)
            self.c1[m] = self.trims[0] + stick
        return self

    def stick(self, t, dur, ch=1, dev=0.9):
        m = (self.ts >= t - 1e-6) & (self.ts < t + dur - 1e-6)
        (self.c1 if ch == 1 else self.c2)[m] = self.trims[ch - 1] + dev * HALF
        return self

    def rc9(self, seq):
        for t, v in seq:
            self.c9[self.ts >= t - 1e-6] = v
        return self

    def yaw_spin(self, T_y, rate=45.0, dur=2.0, stick=0.0):
        w = (self.tf >= T_y) & (self.tf < T_y + dur)
        self.y = np.where(w, rate + self.rng.normal(0, 0.5, self.tf.size), self.y)
        ws = (self.ts >= T_y - 1.5) & (self.ts < T_y + dur)
        self.c4[ws] = self.trims[3] + stick
        return self

    def sink(self, T_s, err=-100.0, dur=1.0):
        m = (self.ts >= T_s - 1e-6) & (self.ts < T_s + dur - 1e-6)
        self.crt[m] = self.dcrt[m] + err
        return self


# ======================================================================================
# 1. 真實 log:00000124 飛行 2(墜機)
# ======================================================================================

def _check_124_crash(d, a, label):
    """飛行 2 的共同驗收(真相錨點與 basic 錨點都要成立)。"""
    al = d["attitude_loss_of_control"]
    assert al.status == HIT and al.severity == "critical", (label, al.status, al.severity)
    v = al.values
    assert v["source"] == "ANG", v["source"]
    near(al.t_onset, 1255.9, 0.15, "t_warn(驗收 A4)")
    near(al.t_onset, 1255.915, 0.05, "t_warn(400 Hz 實測)")
    near(v["t_loss"], 1258.10, 0.10, "t_loss(驗收 A4)")
    assert 1259.2 - 0.05 <= v["t_tilt120"] <= 1259.3, ("tilt ≥ 120°(驗收 A4)", v["t_tilt120"])
    assert v["t_loss"] < v["t_inverted"] < v["t_tilt120"], v                  # 倒扣(> 90°)在失控之後
    assert v["ae_max_normal"] <= 3.65, ("正常飛行 ae 最大 3.6°", v["ae_max_normal"])
    assert v["driver"] == "demand" and v["d_des_deg"] > 3 * v["d_act_deg"], v
    near(v["lead_time_s"], 6.2, 0.1, "預警比上鎖早")
    assert v["anchor_mode"] == "attitude_first", v["anchor_mode"]
    # 錨點由本模組寫入
    near(a.t_warn, 1255.915, 0.05, "a.t_warn")
    near(a.t_loss, 1258.10, 0.05, "a.t_loss")
    assert a.t_inverted is not None and a.t_loss < a.t_inverted < 1259.3, a.t_inverted

    rd = d["rate_demand_not_achieved"]
    assert rd.status == HIT and rd.context == "ground_idle", (rd.status, rd.context)
    near(rd.t_onset, 1255.62, 0.05, "rate_demand 第一個成立視窗結束")
    assert rd.values["axis"] == "roll" and rd.values["gain_at_peak"] < 0.3 and rd.values["rmsd_peak"] >= 75, rd.values
    assert rd.values["max_rmsd_normal_pitch"] <= 22.6 and rd.values["max_rmsd_normal_roll"] <= 17.6, rd.values

    sat = d["rate_output_saturation"]
    assert sat.status == HIT and sat.severity == "critical" and sat.context == "ground_idle", (sat.status, sat.context)
    near(sat.values["t_soft"], 1256.05, 0.05, "ROut 0.25 s 平均 ≥ 0.2")
    near(sat.values["t_hit"], 1258.03, 0.05, "|x| > 0.5")
    near(sat.values["t_crit"], 1258.39, 0.05, "|x| ≥ 0.9")
    near(sat.t_onset, 1256.05, 0.05, "飽和事件起點")
    near(sat.t_alarm, 1258.39, 0.05, "飽和最高層級確認")
    assert sat.values["raw_max"] > 2.0 and sat.values["raw_max_normal"] <= 0.255, sat.values
    assert "1258.39" in sat.text_zh and "1256.05" in sat.text_zh, sat.text_zh   # 文字分層列時間,不會說 1256 就碰上限

    ln = d["lean_without_pilot"]
    assert ln.status == HIT and ln.severity == "critical", (ln.status, ln.severity)
    assert ln.context == "ground_idle" and ln.values["kind"] == "ground_idle", (ln.context, ln.values["kind"])
    assert ln.values["near_ground"] is True and ln.values["agl_at_onset_m"] <= 0.0, ln.values
    near(ln.t_onset, 1255.83, 0.15, "lean > 8°(驗收 A4)")
    assert 24.0 <= ln.values["lean_peak"] <= 26.0, ln.values["lean_peak"]
    near(ln.values["t_crit"], 1256.175, 0.05, "lean > 15°(擋住降落偵測)")
    assert ln.values["lean_from_pos"] is True and ln.values["mode_at_onset"] == "LOITER", ln.values
    assert 1.5 <= ln.values["pos_err_max_m"] <= 2.5, ln.values["pos_err_max_m"]
    assert ln.values["stick_trim"] == [1513.0, 1515.0], ln.values["stick_trim"]

    pi = d["pilot_input_context"]
    assert pi.status == HIT and pi.context == "reaction" and pi.severity == "info", (pi.status, pi.context)
    near(pi.t_onset, 1257.84, 0.15, "駕駛 pitch 滿舵(驗收 A6:反應)")
    assert pi.values["delay_s"] > 1.5 and pi.values["n_precursor_events"] == 0, pi.values
    assert pi.values["precursor_max_dev"] <= 0.30, ("失控前 60 s 平穩", pi.values["precursor_max_dev"])
    assert "(平穩)" in pi.text_zh and pi.values["n_earlier_events"] == 0, pi.text_zh
    near(pi.values["precursor_window"][0], a.t_warn - 60.0, 0.01, "前兆窗起點 = t_warn − 60")
    c = [e for e in pi.values["events"] if e["kind"] == "C_source_switch_flipback"]
    assert c and c[0]["channel"] == 9, pi.values["events"]
    near(c[0]["t"], 1258.64, 0.15, "RC9 切換(驗收 A6)")
    near(c[0]["t_end"], 1259.44, 0.15, "RC9 切回(驗收 A6)")
    assert all(e["t"] >= a.t_warn for e in pi.values["events"]), pi.values["events"]

    ys = d["uncommanded_yaw_or_sink"]
    assert ys.status == CLEAR, (ys.status, ys.context, ys.text_zh)
    assert ys.values["yaw_abs_mean_max_dps"] < 30.0 and ys.values["sink_err_sustained_min_cms"] > -60.0, ys.values

    # MUST NOT:觸地(1255.0)之前沒有任何姿態 / 控制證據
    for f in d.values():
        if f.status in (HIT, SOFT):
            assert f.t_onset >= 1255.0, (label, f.id, f.t_onset)


def test_real_124_crash_truth_anchors():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.truth_anchors_124(ctx)
    d = run_checked(ctx, a, "124 truth")
    _check_124_crash(d, a, "124 truth")
    ys = d["uncommanded_yaw_or_sink"]
    # 有 t_first_hit:下沉判讀只看撞擊前(撞擊後 1259.44 那一點 −126 cm/s 不能進來)
    assert ys.values["sink_clipped_at_impact"] is True and ys.values["sink_err_min_cms"] > -60.0, ys.values
    print(f"  t_warn {a.t_warn:.3f}  t_loss {a.t_loss:.3f}  t_inverted {a.t_inverted:.2f};"
          f"lean {d['lean_without_pilot'].t_onset:.2f}(峰值 {d['lean_without_pilot'].values['lean_peak']}°)")


def test_real_124_crash_basic_anchors():
    """只有 CRASH_CHECK 的 basic_anchors(沒有 end_state 的 t_first_hit):本模組自己把 t_warn / t_loss / t_inverted 填好。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = basic_anchors(ctx)
    assert a.t_warn is None and a.t_loss is None and a.t_inverted is None
    d = run_checked(ctx, a, "124 basic")
    _check_124_crash(d, a, "124 basic")
    ys = d["uncommanded_yaw_or_sink"]
    # 沒有撞擊錨點時整段都看:翻滾中的單點 −126 cm/s 不持續 → 仍 CLEAR,文字要說明是未持續的尖峰
    assert ys.values["sink_clipped_at_impact"] is False and ys.values["sink_err_min_cms"] < -100.0, ys.values
    assert "未持續" in ys.text_zh, ys.text_zh
    print(f"  basic 錨點:同樣的狀態;下沉單點 {ys.values['sink_err_min_cms']} cm/s 未持續 → CLEAR")


def test_real_124_impact_first_anchor_rule():
    """規格:撞擊早於姿態預警(或 ae 從未 > 10°)→ t_warn = t_loss = t_first_hit;正常降落不套用。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.truth_anchors_124(ctx)
    a.t_warn = a.t_loss = a.t_inverted = None
    a.t_first_hit = 1255.5                     # 假設撞擊在姿態預警(1255.915)之前
    f = pick(da.attitude_loss_of_control(ctx, a), "attitude_loss_of_control")
    assert f.values["anchor_mode"] == "impact_first", f.values["anchor_mode"]
    assert a.t_warn == a.t_loss == 1255.5, (a.t_warn, a.t_loss)
    near(a.t_inverted, 1258.97, 0.05, "倒扣仍取姿態算出的時間")
    near(f.t_onset, 1255.915, 0.05, "Finding 仍報 ae 的起點")
    a2 = crashkit.truth_anchors_124(ctx)
    a2.end_type = "normal_landing"
    a2.t_first_hit = 1255.5
    da.attitude_loss_of_control(ctx, a2)
    near(a2.t_warn, 1255.915, 0.05, "正常降落不套用撞擊優先")
    print("  撞擊優先規則:t_warn = t_loss = t_first_hit;end_type 正常時不套用")


def test_real_124_engine_counts_lean_for_land_tipover():
    """跨模組契約:engine 要把 lean_without_pilot(地面怠速)算進 H_LAND_TIPOVER(權重 3);驗收 A3 排名;A6 駕駛是反應。"""
    from core.crash import engine
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    rep = engine.analyze(ctx.ld)
    hs = {h.id: h for h in rep.hypotheses}
    land, nav = hs["H_LAND_TIPOVER"], hs["H_NAV_FAILURE"]
    ev = [e for e in land.evidence if e.get("det") == "lean_without_pilot"]
    assert ev and abs(ev[0]["contrib"] - 3.0) < 1e-6 and ev[0]["tag"] == "precursor", ev
    assert rep.hypotheses[0].id == "H_LAND_TIPOVER" and land.confidence == "高", (rep.hypotheses[0].id, land.confidence)
    assert land.score >= 12.0 and 0.35 <= nav.score / land.score <= 0.7, (land.score, nav.score)
    assert nav.status == "contributing", nav.status
    for hid in ("H_MOTOR_THRUST", "H_CONTROL_PILOT"):
        assert hs[hid].score == 0.0 and hs[hid].status != "ranked", (hid, hs[hid].score, hs[hid].status)
    tags = {f.id: f.tag for f in rep.findings if f.domain == "attitude"}
    assert tags["pilot_input_context"] == "consequence", tags                 # 驗收 A6:駕駛動作是事後反應
    assert tags["lean_without_pilot"] == "precursor" and tags["rate_demand_not_achieved"] == "precursor", tags
    assert tags["attitude_loss_of_control"] == "development", tags
    print(f"  H_LAND {land.score:.1f}(lean +3.0)、H_NAV {nav.score:.1f}(比 {nav.score / land.score:.2f});標籤 {tags}")


# ======================================================================================
# 2. 真實 log:正常飛行(MUST NOT)
# ======================================================================================

def test_real_124_flight1_control():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.flight1_anchors_124(ctx)
    d = run_checked(ctx, a, "124 flight1")
    assert all(f.status == CLEAR for f in d.values()), statuses(d.values())
    assert a.t_warn is None and a.t_loss is None and a.t_inverted is None, "正常飛行不能寫入失控錨點"
    assert d["attitude_loss_of_control"].values["ae_max"] <= 3.65, d["attitude_loss_of_control"].values["ae_max"]
    ln = d["lean_without_pilot"].values
    assert ln["ground_lean_sustained_max"] <= 2.2 and ln["air_lean_sustained_max"] < 9.0, ln
    # 搖桿置中的較大空中傾角都在搖桿回中後的 Loiter 煞車期內(4.6 s = 0.1 + 600/150 + 150/300);煞車期後最大 5.7°
    assert ln["air_lean_sustained_max"] < 6.0 and 7.0 <= ln["air_brake_lean_max"] < AIR_LEAN, ln
    assert ln["brake_window_s"]["loiter"] == 4.6 and ln["idle_above_ground_s"] == 0.0, ln
    assert "未判讀" in d["lean_without_pilot"].text_zh, d["lean_without_pilot"].text_zh
    rd = d["rate_demand_not_achieved"].values
    assert max(rd["max_rmsd_normal_roll"], rd["max_rmsd_normal_pitch"]) <= 17.6, rd
    assert d["rate_output_saturation"].values["raw_max_normal"] <= 0.255, d["rate_output_saturation"].values
    assert d["pilot_input_context"].context == "calm" and d["pilot_input_context"].values["precursor_max_dev"] <= 0.40
    assert d["uncommanded_yaw_or_sink"].values["sink_err_sustained_min_cms"] > -13.0
    print(f"  飛行 1 全 CLEAR:ae {d['attitude_loss_of_control'].values['ae_max']}°、地面怠速 lean "
          f"{ln['ground_lean_sustained_max']}°、空中 {ln['air_lean_sustained_max']}°")


def test_real_124_flight2_before_touchdown():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    iv = ctx.armed[1]
    sub = ArmedInterval(1, iv.t0, 1255.0, 0)
    a = Anchors(interval=sub, intervals=list(ctx.armed))
    a.end_type = "unknown"
    d = run_checked(ctx, a, "124 flight2 <1255")
    assert all(f.status == CLEAR for f in d.values()), statuses(d.values())
    assert a.t_warn is None
    assert d["lean_without_pilot"].values["air_lean_sustained_max"] <= 9.7, d["lean_without_pilot"].values
    # 規格的「正常最大 9.7°」是搖桿回中後的煞車(持續 0.5 s 8.6°);煞車期後只有 3.0°
    lv = d["lean_without_pilot"].values
    assert lv["air_lean_sustained_max"] <= 3.5 and 8.0 <= lv["air_brake_lean_max"] < AIR_LEAN, lv
    rd = d["rate_demand_not_achieved"].values
    assert rd["max_rmsd_normal_pitch"] <= 22.6, rd
    assert d["attitude_loss_of_control"].values["ae_max"] <= 3.65
    print(f"  飛行 2(<1255 s)全 CLEAR:空中置中 lean {d['lean_without_pilot'].values['air_lean_sustained_max']}°、"
          f"rmsD pitch {rd['max_rmsd_normal_pitch']}")


# ======================================================================================
# 3. 真實 log:從未解鎖的 baseline
# ======================================================================================

def test_real_baselines_not_armed():
    for nm in crashkit.BASELINES:
        ctx = crashkit.load_ctx(nm)
        a = crashkit.not_armed_anchors(ctx)
        d = run_checked(ctx, a, nm)
        assert all(f.status == NA and f.context == "not_armed" for f in d.values()), (nm, statuses(d.values()))
        assert all("未解鎖" in f.text_zh for f in d.values()), nm
        assert a.t_warn is None and a.t_loss is None and a.t_inverted is None, nm
    print(f"  {len(crashkit.BASELINES)} 個 baseline 全部 NA not_armed,錨點不動")


def test_real_baselines_ungated_gate_is_essential():
    """驗收 D1:閘門拿掉(整段當解鎖)時,手持 / 桌上的 117 / 119 / 120 會被姿態 / 角速度偵測器誤報 → 閘門必要。"""
    for nm, ae_min in (("00000117", 10.0), ("00000119", 45.0), ("00000120", 29.0)):
        ctx = crashkit.load_ctx(nm)
        tb = ctx.table("ANG")
        iv = ArmedInterval(0, float(tb["t"][0]), float(tb["t"][-1]), None)
        a = Anchors(interval=iv, intervals=[iv])
        a.end_type = "unknown"
        d = run_checked(ctx, a, f"{nm} ungated")
        assert d["attitude_loss_of_control"].status == HIT, (nm, statuses(d.values()))
        assert d["rate_demand_not_achieved"].status == HIT, (nm, statuses(d.values()))
        assert d["attitude_loss_of_control"].values["ae_max"] >= ae_min, (nm, d["attitude_loss_of_control"].values["ae_max"])
        # 馬達從沒轉、油門 0:lean / 駕駛 / 偏航下沉 不會說「正常」
        for k in ("lean_without_pilot", "pilot_input_context", "uncommanded_yaw_or_sink"):
            assert d[k].status == NA, (nm, k, d[k].status)
    print("  117 / 119 / 120 不加閘門 → attitude / rate HIT(ae 16.5 / 49 / 30.4°),所以解鎖閘門不可少")


# ======================================================================================
# 4. 真實資料單元
# ======================================================================================

def test_real_angerr_and_roll_wrap():
    """推力向量誤差重現韌體 AngErr=150(±2,上鎖前最後的 ATT 樣本);Roll +172.4 → −178.8 折返時傾角 / 誤差連續。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    att = ctx.table("ATT")
    ae = util.thrust_vector_error_deg(att["Roll"], att["Pitch"], att["Yaw"], att["DesRoll"], att["DesPitch"], att["DesYaw"])
    tilt = util.tilt_deg(att["Roll"], att["Pitch"])
    i = int(np.searchsorted(att["t"], 1262.125)) - 1
    near(ae[i], 150.0, 2.0, "AngErr 重現")
    m = (att["t"] > 1259.40) & (att["t"] < 1259.60)
    r = att["Roll"][m]
    assert r.size == 2 and r[0] > 170 and r[1] < -170, r                    # Euler 折返
    assert abs(tilt[m][1] - tilt[m][0]) < 5.0 and abs(ae[m][1] - ae[m][0]) < 15.0, (tilt[m], ae[m])
    # RCIN 經 Ctx 已轉 float:減 trim 不會回繞成 65535
    rc = ctx.table("RCIN")
    assert rc["C1"].dtype == np.float64 and float(np.min(rc["C1"] - 1513.0)) > -1000.0
    print(f"  AngErr {ae[i]:.1f}°(韌體 150);Roll {r[0]:.1f} → {r[1]:.1f} 時 tilt {tilt[m][0]:.1f} → {tilt[m][1]:.1f}")


class _HideANG:
    """包住真的 Ctx,假裝沒有 ANG(測 ATT 10 Hz 備援)。"""

    def __init__(self, c):
        self._c = c

    def __getattr__(self, k):
        return getattr(self._c, k)

    def table(self, name, instance=None):
        return None if name == "ANG" else self._c.table(name, instance)


def test_real_att_fallback_without_ang():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.truth_anchors_124(ctx)
    d = run_checked(_HideANG(ctx), a, "124 ATT only")
    al = d["attitude_loss_of_control"]
    assert al.status == HIT and al.values["source"] == "ATT", (al.status, al.values["source"])
    near(al.t_onset, 1255.9, 0.15, "ATT t_warn")
    near(al.values["t_loss"], 1258.10, 0.15, "ATT t_loss")
    assert 1259.15 <= al.values["t_tilt120"] <= 1259.35, al.values["t_tilt120"]
    ln = d["lean_without_pilot"]
    assert ln.status == HIT and ln.context == "ground_idle" and ln.values["source"] == "ATT", (ln.status, ln.context)
    near(ln.t_onset, 1255.83, 0.15, "ATT lean > 8°")
    f1 = run_checked(_HideANG(ctx), crashkit.flight1_anchors_124(ctx), "124 flight1 ATT only")
    assert all(f.status == CLEAR for f in f1.values()), statuses(f1.values())
    print(f"  只有 ATT:t_warn {al.t_onset:.2f}、t_loss {al.values['t_loss']:.2f}、lean {ln.t_onset:.2f};飛行 1 全 CLEAR")


def test_real_timing_and_determinism():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    worst = 0.0
    for mk in (crashkit.truth_anchors_124, crashkit.flight1_anchors_124):
        for fn in DETECTORS:
            a = mk(ctx)
            t0 = time.perf_counter()
            fn(ctx, a)
            dt = time.perf_counter() - t0
            worst = max(worst, dt)
            assert dt < 1.5, (fn.__name__, dt)
    r1 = [(f.id, f.status, f.context, f.t_onset, json.dumps(f.values, sort_keys=True), f.text_zh)
          for f in da.run(ctx, crashkit.truth_anchors_124(ctx))]
    r2 = [(f.id, f.status, f.context, f.t_onset, json.dumps(f.values, sort_keys=True), f.text_zh)
          for f in da.run(ctx, crashkit.truth_anchors_124(ctx))]
    assert r1 == r2
    print(f"  每個偵測器 < 1.5 s(最慢 {worst:.3f} s);同輸入同輸出")


# ======================================================================================
# 5. 合成資料
# ======================================================================================

def test_synth_healthy_flight_all_clear():
    sc = Scene()
    # 解鎖前手持亂晃、上鎖後倒扣的垃圾:都在區間外,不能被分析
    pre = sc.tf < sc.t_arm
    sc.roll[pre], sc.pitch[pre] = 40.0, -30.0
    post = sc.tf > sc.t_disarm
    sc.roll[post], sc.droll[post] = 175.0, 0.0
    sc.rdes[pre], sc.r[pre], sc.rout[pre] = 200.0, 0.0, 0.8
    a = sc.anchors()
    d = run_checked(sc.ctx(), a, "healthy")
    assert all(f.status == CLEAR for f in d.values()), statuses(d.values())
    assert a.t_warn is None and a.t_loss is None and a.t_inverted is None
    assert d["attitude_loss_of_control"].values["ae_max"] < 3.0
    assert d["lean_without_pilot"].values["stick_trim"] == [1490.0, 1530.0]       # 讀 RCn_TRIM,不是 1500 / 1513
    assert d["pilot_input_context"].values["trims"] == [1490.0, 1530.0]
    assert d["pilot_input_context"].context == "calm"                              # 解鎖手勢 C4=1900 不算
    print("  健康飛行(含解鎖前 / 上鎖後垃圾、解鎖手勢):六個全 CLEAR,錨點不動")


def _tipover_expect(sc, d, a, label):
    T = sc.t_land
    al = d["attitude_loss_of_control"]
    assert al.status == HIT and al.severity == "critical" and al.context == "ground_idle", (label, al.status, al.context)
    # 機身姿態帶 0.2° 雜訊、指令斜率 12°/s → 越線時間有 ~±0.05 s 的抖動
    near(al.t_onset, T + 0.5 + 10 / 12, 0.05, f"{label} t_warn(指令 10°)")
    near(al.values["t_loss"], T + 2.5 + np.sqrt(48 / 60), 0.01, f"{label} t_loss(roll − 18 > 30)")
    near(al.values["t_inverted"], T + 2.5 + np.sqrt(90 / 60), 0.01, f"{label} t_inverted(roll > 90)")
    assert al.values["driver"] == "demand", al.values["driver"]
    near(a.t_warn, al.t_onset, 1e-9, "錨點 = Finding")
    ln = d["lean_without_pilot"]
    assert ln.status == HIT and ln.context == "ground_idle" and ln.severity == "critical", (label, ln.status, ln.context)
    assert ln.values["near_ground"] is True, ln.values
    near(ln.t_onset, T + 0.5 + 8 / 12, 0.01, f"{label} lean ≥ 8°")
    near(ln.values["t_crit"], T + 0.5 + 15 / 12, 0.01, f"{label} lean ≥ 15°")
    assert abs(ln.values["lean_peak"] - 18.0) < 0.1 and ln.values["lean_from_pos"] is True, ln.values
    rd = d["rate_demand_not_achieved"]
    assert rd.status == HIT and rd.context == "ground_idle" and rd.values["axis"] == "roll", (label, rd.status, rd.context)
    near(rd.t_onset, T + 0.8 + 71 / 400, 0.01, f"{label} rate_demand(rmsD 50·√(k/200) ≥ 30)")
    sat = d["rate_output_saturation"]
    assert sat.status == HIT and sat.severity == "critical" and sat.context == "ground_idle", (label, sat.status, sat.context)
    near(sat.values["t_soft"], T + 1.0 + 79 / 400, 0.02, f"{label} 0.25 s 平均 ≥ 0.2")
    near(sat.values["t_crit"], T + 2.5, 0.01, f"{label} ≥ 0.9")
    near(sat.t_onset, sat.values["t_soft"], 1e-3, "事件起點 = 最早層級")
    assert d["pilot_input_context"].status == CLEAR, statuses(d.values())
    assert d["uncommanded_yaw_or_sink"].status == CLEAR, statuses(d.values())


def test_synth_ground_tipover():
    """124 型態但數字不同(18° 指令、trim 1490/1530、懸停 0.35、四軸、上鎖時才寫 EV17/18):時間要對到解析值。"""
    sc = Scene().tipover()
    a = sc.anchors(end_type="crash_check")
    d = run_checked(sc.ctx(), a, "tipover")
    _tipover_expect(sc, d, a, "tipover")
    print(f"  地面翻覆:rate {d['rate_demand_not_achieved'].t_onset - sc.t_land:.3f} → lean "
          f"{d['lean_without_pilot'].t_onset - sc.t_land:.3f} → warn {d['attitude_loss_of_control'].t_onset - sc.t_land:.3f} s(相對觸地)")


def test_synth_time_shift_and_uint16_invariance():
    """時間基準平移 1000 s、RCIN/RCOU 改 uint16、EV/MODE 改 uint8:結果只平移,不回繞。"""
    base = Scene().tipover()
    d0 = run_checked(base.ctx(), base.anchors(end_type="crash_check"), "tipover base")
    sh = Scene(t_arm=1020.0).tipover()
    a1 = sh.anchors(end_type="crash_check")
    d1 = run_checked(sh.ctx(u16=True), a1, "tipover shifted u16")
    _tipover_expect(sh, d1, a1, "shifted")
    for k in OWNED:
        assert d0[k].status == d1[k].status and d0[k].context == d1[k].context, (k, d0[k].status, d1[k].status)
        if d0[k].t_onset is not None:
            near(d1[k].t_onset - d0[k].t_onset, 1000.0, 0.003, k)
    print("  平移 1000 s + uint16 / uint8 輸入:狀態相同、時間只平移")


def test_synth_air_loss_vehicle_driven():
    """空中機身自己翻(指令反向修正):context 'air'、driver 'vehicle';輸出飽和 / 角速度追不上 / 下沉都要報。"""
    sc = Scene()
    T_f = sc.t_to + 30.0
    sc.air_loss(T_f)
    a = sc.anchors(end_type="crash_check")
    d = run_checked(sc.ctx(), a, "air loss")
    al = d["attitude_loss_of_control"]
    assert al.status == HIT and al.context == "air" and al.values["driver"] == "vehicle", (al.status, al.context, al.values)
    assert T_f < al.t_onset < T_f + 0.5 and al.values["t_loss"] < T_f + 1.0, al.values
    rd, sat = d["rate_demand_not_achieved"], d["rate_output_saturation"]
    assert rd.status == HIT and rd.context == "air" and rd.values["gain_min"] < 0, (rd.status, rd.context, rd.values)
    assert sat.status == HIT and sat.context == "air" and sat.severity == "critical", (sat.status, sat.context)
    near(sat.values["t_crit"], T_f + 0.1, 0.01, "ROut −1.2")
    assert "後段" in sat.text_zh and "電調遙測" in sat.text_zh, sat.text_zh      # 空中飽和通常是後果
    ys = d["uncommanded_yaw_or_sink"]
    assert ys.status == HIT and ys.context == "sink", (ys.status, ys.context)
    near(ys.t_onset, T_f + 0.5, 0.11, "下沉起點")
    ev = ys.values["events"][0]
    assert ev["kind"] == "sink" and ev["err_min_cms"] <= -390 and ev["free_fall_before"] is True, ev
    assert d["lean_without_pilot"].status == CLEAR and d["pilot_input_context"].status == CLEAR, statuses(d.values())
    print(f"  空中失控:warn +{al.t_onset - T_f:.2f} s、driver vehicle、ROut 飽和 air、下沉 HIT(含失重佐證)")


def test_synth_commanded_flip_is_exempt():
    """同樣的 360° 滾轉:LOITER 沒有特技標記 → HIT;ACRO 模式或 EV21/22 FLIP 窗內 → 不判讀(不是失控)。"""
    out = {}
    for how in ("none", "acro", "ev"):
        sc = Scene()
        sc.flip(sc.t_to + 25.0, how)
        a = sc.anchors()
        out[how] = (run_checked(sc.ctx(), a, f"flip {how}"), a)
    d, a = out["none"]
    assert d["attitude_loss_of_control"].status == HIT and a.t_inverted is not None, statuses(d.values())
    assert d["rate_demand_not_achieved"].status == HIT and d["rate_output_saturation"].status == HIT, statuses(d.values())
    for how in ("acro", "ev"):
        d, a = out[how]
        for k in ("attitude_loss_of_control", "rate_demand_not_achieved", "rate_output_saturation"):
            assert d[k].status == CLEAR, (how, k, d[k].status, d[k].text_zh)
        assert a.t_warn is None and a.t_inverted is None, (how, a.t_warn, a.t_inverted)
    print("  同一個翻滾:LOITER → HIT;ACRO / EV21-22 → CLEAR(特技豁免)")


def test_synth_lean_in_air_variants():
    T = None
    res = {}
    for key, kw, alt in (("air", {}, 10.0), ("air_low", {}, 1.0), ("althold", {"mode": 2}, 10.0),
                         ("auto", {"mode": 3}, 10.0), ("stick", {"stick": 100.0}, 10.0), ("small", {"deg": 11.0}, 10.0)):
        sc = Scene(amp=0.0, alt_air=alt)
        T = sc.t_to + 30.0
        sc.lean_air(T, **kw)
        res[key] = run_checked(sc.ctx(), sc.anchors(), f"lean {key}")
    f = res["air"]["lean_without_pilot"]
    assert f.status == HIT and f.context == "air" and f.values["kind"] == "air" and f.severity == "high", (f.status, f.severity)
    near(f.t_onset, T + 0.3 * 12 / 15, 0.01, "lean ≥ 12°")
    assert f.values["lean_from_pos"] is True and abs(f.values["lean_peak"] - 15.0) < 0.1, f.values
    assert f.values["near_ground"] is False and f.values["agl_at_onset_m"] > 5.0, f.values
    assert res["air"]["attitude_loss_of_control"].status == CLEAR      # 機身有照做:不是姿態失控
    g = res["air_low"]["lean_without_pilot"]
    assert g.status == HIT and g.context == "air" and g.values["near_ground"] is True, (g.status, g.context, g.values)
    near(g.values["agl_at_onset_m"], 1.0, 0.05, "離地 1 m(≤ 1.5 m 才算近地翻覆證據)")
    assert "離地僅約" in g.text_zh, g.text_zh
    for key in ("althold", "auto", "stick", "small"):
        h = res[key]["lean_without_pilot"]
        assert h.status == CLEAR, (key, h.status, h.text_zh)
    print("  空中 lean:LOITER 10 m → air、1 m → air + near_ground;ALT_HOLD / AUTO(自主路徑)/ 搖桿有推 / 11° → CLEAR")


def test_synth_pilot_inputs():
    # (a) 失控前有滿舵(兩次:很早一次、失控前 5 s 一次)→ precursor,t_onset 取最接近 t_warn 的那次
    sc = Scene()
    T_f = sc.t_to + 40.0
    sc.stick(sc.t_to + 10.0, 0.5, ch=2, dev=-0.9).stick(T_f - 5.0, 0.5, ch=1, dev=0.9).air_loss(T_f)
    a = sc.anchors(end_type="crash_check")
    d = run_checked(sc.ctx(), a, "pilot precursor")
    p = d["pilot_input_context"]
    assert p.status == HIT and p.context == "precursor" and p.severity == "medium", (p.status, p.context)
    near(p.t_onset, T_f - 5.0, 0.01, "最接近 t_warn 的前兆")
    assert p.values["n_precursor_events"] >= 2 and p.values["delay_s"] < -4.0, p.values
    near(p.values["first_input_t"], sc.t_to + 10.0, 0.01, "最早的動作仍保留")
    # (b) 只有失控後的反應(滿舵 + RC9 折返)→ reaction
    sc = Scene()
    T_f = sc.t_to + 40.0
    sc.air_loss(T_f).stick(T_f + 1.0, 0.5, ch=2, dev=1.0).rc9([(T_f + 1.5, 1515.0), (T_f + 2.3, 1095.0)])
    a = sc.anchors(end_type="crash_check")
    p = run_checked(sc.ctx(), a, "pilot reaction")["pilot_input_context"]
    assert p.status == HIT and p.context == "reaction" and p.values["n_precursor_events"] == 0, (p.status, p.context)
    assert p.t_onset >= a.t_warn and any(e["kind"] == "C_source_switch_flipback" for e in p.values["events"]), p.values
    # (c) 沒有失控:B 快速甩桿(0.7 行程,不到 A 的 0.8)與單向切換(1095→1515→1935)→ 只有 B,no_reference
    sc = Scene()
    m = (sc.ts >= sc.t_to + 20.0 - 1e-6) & (sc.ts < sc.t_to + 20.3 - 1e-6)
    sc.c2[m] = sc.trims[1] + 0.7 * HALF
    sc.rc9([(sc.t_to + 30.0, 1515.0), (sc.t_to + 30.1, 1935.0)])
    p = run_checked(sc.ctx(), sc.anchors(), "pilot no ref")["pilot_input_context"]
    assert p.status == HIT and p.context == "no_reference", (p.status, p.context)
    kinds = {e["kind"] for e in p.values["events"]}
    assert kinds == {"B_fast_move"}, kinds
    # (d) 地面(起飛前)的大動作與解鎖手勢不算
    sc = Scene()
    sc.stick(sc.t_arm + 1.0, 2.0, ch=1, dev=1.0)
    p = run_checked(sc.ctx(), sc.anchors(), "pilot ground")["pilot_input_context"]
    assert p.status == CLEAR, (p.status, p.values["events"])
    # 純函式:折返 vs 單向
    t = np.arange(10) * 0.1
    assert len(da.level_flipbacks(t, np.array([1095, 1095, 1515, 1515, 1095, 1095, 1095, 1095, 1095, 1095.0]))) == 1
    assert da.level_flipbacks(t, np.array([1095, 1095, 1515, 1935, 1935, 1935, 1935, 1935, 1935, 1935.0])) == []
    t2 = np.arange(40) * 0.1
    v2 = np.where((t2 >= 0.5) & (t2 < 2.6), 1515.0, 1095.0)              # 2.1 s 才切回:超過 1.5 s 不算折返
    assert da.level_flipbacks(t2, v2) == []
    print("  駕駛:前兆取最接近 t_warn 的(medium)、反應(info)、無參考;地面動作 / 單向切換 / 慢切回不算")


def test_synth_uncommanded_yaw_and_sink():
    sc = Scene()
    T_y = sc.t_to + 25.0
    sc.yaw_spin(T_y)
    d = run_checked(sc.ctx(), sc.anchors(), "yaw spin")
    ys = d["uncommanded_yaw_or_sink"]
    assert ys.status == HIT and ys.context == "yaw", (ys.status, ys.context)
    near(ys.t_onset, T_y + 133 / 400, 0.02, "0.5 s 平均 ≥ 30°/s 的第一個視窗結束")
    assert "信心低" in ys.text_zh and "電調遙測" in ys.text_zh, ys.text_zh
    # 偏航搖桿有打 → 是駕駛指令,不報
    sc = Scene()
    sc.yaw_spin(T_y, stick=150.0)
    assert pick(da.run(sc.ctx(), sc.anchors()), "uncommanded_yaw_or_sink").status == CLEAR
    # 下沉:−100 cm/s 持續 1 s → HIT;只有 0.3 s → CLEAR
    sc = Scene()
    sc.sink(sc.t_to + 25.0, -100.0, 1.0)
    f = pick(da.run(sc.ctx(), sc.anchors()), "uncommanded_yaw_or_sink")
    assert f.status == HIT and f.context == "sink", (f.status, f.context)
    near(f.t_onset, sc.t_to + 25.0, 0.01, "下沉起點")
    assert f.values["events"][0]["free_fall_before"] is False
    sc = Scene()
    sc.sink(sc.t_to + 25.0, -100.0, 0.3)
    f = pick(da.run(sc.ctx(), sc.anchors()), "uncommanded_yaw_or_sink")
    assert f.status == CLEAR and f.values["sink_err_min_cms"] < -90 and "未持續" in f.text_zh, (f.status, f.values)
    # 撞擊之後的下沉不看(t_first_hit 之後的高度是垃圾)
    sc = Scene()
    sc.sink(sc.t_to + 25.0, -100.0, 1.0)
    f = pick(da.run(sc.ctx(), sc.anchors(end_type="crash_check", t_first_hit=sc.t_to + 24.0)), "uncommanded_yaw_or_sink")
    assert f.status == CLEAR and f.values["sink_clipped_at_impact"] is True, (f.status, f.values)
    print("  偏航自轉 HIT / 有打舵 CLEAR;下沉 1 s HIT / 0.3 s CLEAR / 撞擊後不看")


def test_synth_impact_first_anchors():
    sc = Scene().tipover()
    T = sc.t_land
    a = sc.anchors(end_type="crash_check", t_first_hit=T + 1.0)          # 撞擊早於 ae 預警(T+1.333)
    f = pick(da.run(sc.ctx(), a), "attitude_loss_of_control")
    assert f.values["anchor_mode"] == "impact_first" and a.t_warn == a.t_loss == T + 1.0, (f.values, a.t_warn, a.t_loss)
    near(a.t_inverted, T + 2.5 + np.sqrt(1.5), 0.01, "倒扣時間")
    a = sc.anchors(end_type="crash_check", t_first_hit=T + 2.0)          # 撞擊晚於預警 → 姿態優先
    pick(da.run(sc.ctx(), a), "attitude_loss_of_control")
    near(a.t_warn, T + 0.5 + 10 / 12, 0.05, "姿態優先")
    # 撞擊型:ae 從未 > 10°(健康飛行 + 撞擊錨點)→ Finding CLEAR,但錨點 = 撞擊
    sc = Scene()
    a = sc.anchors(end_type="hard_impact", t_first_hit=sc.t_to + 30.0)
    f = pick(da.run(sc.ctx(), a), "attitude_loss_of_control")
    assert f.status == CLEAR and a.t_warn == a.t_loss == sc.t_to + 30.0, (f.status, a.t_warn)
    print("  撞擊優先:t_first_hit 早於預警 / 無預警 → t_warn = t_loss = t_first_hit;晚於預警 → 不改")


def test_synth_gaps_break_runs():
    # 400 Hz:ae > 10° 兩段各 0.2 s,中間 0.1 s 資料空洞 → 不能湊成 0.3 s
    t = np.concatenate([np.arange(0, 0.2, 1 / FAST), np.arange(0.3, 0.5, 1 / FAST)]) + 100.0
    ae = np.full(t.size, 12.0)
    ev = da.attitude_events(t, ae, np.zeros(t.size), da.GAP_FAST_S)
    assert ev["warn"] is None and ev["info"] is None, ev
    t2 = np.arange(0, 0.5, 1 / FAST) + 100.0                             # 沒有空洞 → 成立,起點 = run 起點
    ev = da.attitude_events(t2, np.full(t2.size, 12.0), np.zeros(t2.size), da.GAP_FAST_S)
    assert ev["warn"] is not None and abs(ev["warn"][0] - 100.0) < 1e-9 and ev["loss"] is None, ev
    # 第一次持續越線(不是最後一次):中途掉回 8° 不影響 t_warn
    t3 = np.arange(0, 1.0, 1 / FAST) + 100.0
    ae3 = np.where(t3 < 100.35, 12.0, np.where(t3 < 100.4, 8.0, 12.0))
    ev = da.attitude_events(t3, ae3, np.zeros(t3.size), da.GAP_FAST_S)
    assert abs(ev["warn"][0] - 100.0) < 1e-9, ev
    # 合成飛行裡 RATE 跨 0.2 s 空洞的視窗要略過:角速度指令 50°/s 只出現在空洞兩側 → 不湊窗
    sc = Scene()
    T = sc.t_to + 25.0
    w = (sc.tf >= T) & (sc.tf < T + 0.6)
    sc.rdes[w], sc.r[w] = 50.0, 0.0
    f = pick(da.run(sc.ctx(), sc.anchors()), "rate_demand_not_achieved")
    assert f.status == HIT, (f.status, f.text_zh)                          # 對照組:沒有空洞時會成立
    tb = sc.tables()
    keep = ~((sc.tf >= T + 0.15) & (sc.tf < T + 0.45))
    tb["RATE"] = {k: v[keep] for k, v in tb["RATE"].items()}
    ctx = FakeCtx(tb, sc.params, [sc.interval()])
    f = pick(da.run(ctx, sc.anchors()), "rate_demand_not_achieved")
    assert f.status in (CLEAR, SOFT), (f.status, f.text_zh)
    print("  空洞斷開 run、取第一次持續越線、跨空洞的滾動窗略過")


def test_synth_na_paths_never_clear():
    sc = Scene()
    # 沒有解鎖區間
    a = Anchors()
    fs = da.run(sc.ctx(), a)
    check_contract(fs, "no interval")
    assert all(f.status == NA and f.context == "not_armed" for f in fs), statuses(fs)
    # 例外 → guarded 轉 NA(context='error'),六個 id 都還在
    fs = da.run(sc.ctx(boom=True), sc.anchors())
    assert sorted(f.id for f in fs) == sorted(OWNED) and all(f.status == NA and f.context == "error" for f in fs), statuses(fs)
    # 沒有任何表
    fs = da.run(FakeCtx({}, sc.params, [sc.interval()]), sc.anchors())
    check_contract(fs, "no tables")
    assert all(f.status == NA for f in fs), statuses(fs)
    # 解鎖只有 3 s(地面):資料不足 → 全 NA,不能說「正常」
    short = ArmedInterval(0, sc.t_arm, sc.t_arm + 3.0, 0)
    a = Anchors(interval=short, intervals=[short])
    fs = da.run(FakeCtx(sc.tables(), sc.params, [short]), a)
    check_contract(fs, "short")
    assert all(f.status == NA for f in fs), statuses(fs)
    # 缺 RCIN:lean / pilot / yaw → NA,姿態類照常
    d = run_checked(sc.ctx(drop=("RCIN",)), sc.anchors(), "no RCIN")
    assert d["lean_without_pilot"].status == NA and d["pilot_input_context"].status == NA, statuses(d.values())
    assert d["attitude_loss_of_control"].status == CLEAR
    assert d["uncommanded_yaw_or_sink"].status == NA and d["uncommanded_yaw_or_sink"].context == "partial_data"
    # 缺 RATE:rate 兩個 NA;沒有 ANG 退 ATT
    d = run_checked(sc.ctx(drop=("RATE", "ANG")), sc.anchors(), "no RATE/ANG")
    assert d["rate_demand_not_achieved"].status == NA and d["rate_output_saturation"].status == NA, statuses(d.values())
    assert d["attitude_loss_of_control"].status == CLEAR and d["attitude_loss_of_control"].values["source"] == "ATT"
    # 資料是 NaN:不能 CLEAR
    sc2 = Scene()
    sc2.roll[:] = np.nan
    d = run_checked(sc2.ctx(drop=("ATT",)), sc2.anchors(), "NaN ANG")
    assert d["attitude_loss_of_control"].status == NA, d["attitude_loss_of_control"].status
    # 解鎖後幾秒就失控(預警前沒有 5 s 正常基準):不能因格式化 None 而失敗
    sc3 = Scene(takeoff=1.0)
    w = sc3.tf >= sc3.t_arm + 1.0
    sc3.droll = np.where(w, 25.0, sc3.droll)
    al = pick(da.run(sc3.ctx(), sc3.anchors(end_type="crash_check")), "attitude_loss_of_control")
    assert al.status == HIT and al.values["ae_max_normal"] is None and "基準" in al.text_zh, (al.status, al.values)
    print("  無區間 / 例外 / 無表 / 3 s / 缺 RCIN / 缺 RATE / NaN → NA;早期失控沒有基準也不出錯")


def test_pure_helpers():
    rng = np.random.default_rng(3)
    # _runs 與 util.runs 一致(含空洞與最短持續)
    for _ in range(30):
        n = int(rng.integers(1, 300))
        t = np.cumsum(rng.choice([0.0025, 0.0025, 0.0025, 0.08], n))
        m = rng.random(n) > 0.4
        for md in (0.0, 0.01):
            assert da._runs(m, t, md, 0.05) == util.runs(m, t, md, 0.05)
    assert da._runs(np.zeros(0, bool), np.zeros(0)) == []
    # rolling_window_stats 對照暴力迴圈
    N, n = 500, 50
    t = np.arange(N) / FAST
    dd, xx = rng.normal(0, 30, N), rng.normal(0, 30, N)
    xx[100] = np.nan
    ts, te, rms, gain, nbad = da.rolling_window_stats(t, dd, xx, n)
    for i in (0, 77, 120, N - n):
        dw, xw = dd[i:i + n], xx[i:i + n]
        assert te[i] == t[i + n - 1] and ts[i] == t[i]
        if np.isfinite(xw).all():
            assert abs(rms[i] - np.sqrt(np.mean(dw ** 2))) < 1e-9 and nbad[i] == 0
            assert abs(gain[i] - np.sum(dw * xw) / np.sum(dw * dw)) < 1e-9
        else:
            assert nbad[i] == 1                      # 含 NaN 的窗被標成無效(呼叫端略過)
    # moving_mean 對照
    _, te, mean, nb = da.moving_mean(t, np.abs(dd), 20)
    assert abs(mean[33] - np.mean(np.abs(dd[33:53]))) < 1e-9 and nb.sum() == 0
    # _sustained_max:連續 n 個都 ≥ L 的最大 L
    assert da._sustained_max(np.array([1, 5, 6, 7, 2, 9, 9.0]), 3) == 5.0
    assert da._sustained_max(np.array([1.0, 2.0]), 3) is None
    # _count_in_trailing:窗 (t − span, t] 的計數與窗內第一個為真的索引
    t = np.arange(10) * 0.05
    flag = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0], bool)
    cnt, first = da._count_in_trailing(t, flag, 0.1)
    assert list(cnt) == [0, 1, 2, 1, 0, 1, 1, 0, 0, 0] and first[2] == 1 and first[3] == 2 and first[6] == 5, (cnt, first)
    # 推力向量誤差:純 yaw 差不算誤差;純 roll 差 = 角度差;跨 ±180 不跳
    assert float(util.thrust_vector_error_deg(0, 0, 10, 0, 0, 100)) < 1e-6
    assert abs(float(util.thrust_vector_error_deg(20, 0, 0, 0, 0, 0)) - 20.0) < 1e-6
    assert abs(float(util.thrust_vector_error_deg(179, 0, 0, -179, 0, 0)) - 2.0) < 1e-6
    # stick_events:A 需持續 0.2 s;B 比較 0.5 s 前的樣本
    t = np.arange(50) * 0.1
    p1 = np.where((t >= 1.0) & (t < 1.35), 380.0, 0.0)
    ra, rb, dev, mv = da.stick_events(t, p1, np.zeros(50), 400.0, 400.0, np.ones(50, bool))
    assert len(ra) == 1 and abs(ra[0][0] - 1.0) < 1e-9 and rb and abs(rb[0][0] - 1.0) < 1e-9, (ra, rb)
    p1 = np.where((t >= 1.0) & (t < 1.15), 380.0, 0.0)                   # 只有 0.1 s:不算 A
    ra, _, _, _ = da.stick_events(t, p1, np.zeros(50), 400.0, 400.0, np.ones(50, bool))
    assert ra == []
    print("  _runs = util.runs;滾動窗 / 平均 / 持續最大 / 尾窗計數 / 推力向量 / 搖桿事件 對照正確")


# ======================================================================================
# 6. 審查缺陷的回歸測試
# ======================================================================================

def _engine_attitude_only(sc):
    """只跑本模組的 engine(CRASH_CHECK 上鎖),回傳 {假說 id: 分數} 與 findings。"""
    from core.crash import engine
    iv = ArmedInterval(0, sc.t_arm, sc.t_disarm, 12)
    rep = engine.analyze_ctx(FakeCtx(sc.tables(), sc.params, [iv]), modules=["det_attitude"])
    return {h.id: h.score for h in rep.hypotheses}, rep.findings


def test_synth_pilot_precursor_window():
    """缺陷 1:前兆只算 [max(t_arm+3, t_warn−60), t_warn) 內的大動作;更早的只列 earlier_inputs,不能變成「可能肇因」。"""
    res = {}
    for gap in (30.0, 70.0, 280.0):
        sc = Scene(flight=400.0)
        T_f = sc.t_to + 320.0
        sc.stick(T_f - gap, 0.5, ch=2, dev=-0.9).air_loss(T_f)
        a = sc.anchors(end_type="crash_check")
        res[gap] = (run_checked(sc.ctx(), a, f"pilot gap {gap}")["pilot_input_context"], a, sc)
    p, a, _ = res[30.0]                                   # 窗內 → precursor(規格)
    assert p.status == HIT and p.context == "precursor" and p.values["n_earlier_events"] == 0, (p.status, p.context)
    for gap in (70.0, 280.0):                             # 窗外 → 前兆窗平穩 → CLEAR,動作只列參考
        p, a, sc = res[gap]
        assert p.status == CLEAR and p.context == "calm", (gap, p.status, p.context, p.text_zh)
        assert p.values["n_earlier_events"] >= 1 and p.values["precursor_max_dev"] <= 0.1, p.values
        near(p.values["earlier_inputs"][0]["t"], sc.t_to + 320.0 - gap, 0.01, "較早的動作時間(T_f − gap)")
        assert "前兆窗之外" in p.text_zh, p.text_zh
    # engine:280 s 前的動作不會被標成前兆、飛手假說的必要證據不成立
    scores, fs = _engine_attitude_only(res[280.0][2])
    assert not any(f.id == "pilot_input_context" and f.tag == "precursor" for f in fs), [(f.id, f.tag) for f in fs]
    assert scores["H_CONTROL_PILOT"] < scores["H_MOTOR_THRUST"], scores
    # 很早的動作 + 失控後的反應 → reaction(原寫法:早期動作勝出成 precursor,反應被吃掉)
    sc = Scene(flight=400.0)
    T_f = sc.t_to + 320.0
    sc.stick(T_f - 200.0, 0.5, ch=2, dev=-0.9).air_loss(T_f).stick(T_f + 1.0, 0.5, ch=2, dev=1.0)
    a = sc.anchors(end_type="crash_check")
    p = run_checked(sc.ctx(), a, "pilot early + reaction")["pilot_input_context"]
    assert p.status == HIT and p.context == "reaction" and p.t_onset >= a.t_warn, (p.status, p.context, p.t_onset)
    assert p.values["n_earlier_events"] >= 1 and p.values["n_precursor_events"] == 0 and "(平穩)" in p.text_zh, p.values
    # 「(平穩)」只在前兆窗最大偏移 < 0.4 行程時寫:慢慢推到 0.6 行程(不到 A / B 門檻)→ 不能說平穩
    sc = Scene(flight=100.0)
    T_f = sc.t_to + 60.0
    x = sc.ts - (T_f - 20.0)
    prof = np.clip(np.minimum(x / 2.0, (6.0 - x) / 2.0), 0.0, 1.0) * 0.6 * HALF
    sc.c2 = sc.c2 + prof
    sc.air_loss(T_f).stick(T_f + 1.0, 0.5, ch=2, dev=1.0)
    a = sc.anchors(end_type="crash_check")
    p = run_checked(sc.ctx(), a, "pilot moderate pre")["pilot_input_context"]
    assert p.context == "reaction" and 0.5 <= p.values["precursor_max_dev"] < 0.8, (p.context, p.values["precursor_max_dev"])
    assert "(平穩)" not in p.text_zh and "未達" in p.text_zh, p.text_zh
    print("  前兆窗 60 s:30 s 前 → precursor;70 / 280 s 前 → CLEAR + earlier_inputs;早期動作不蓋掉反應;偏移 0.6 不寫平穩")


def test_synth_crash_episode_selection():
    """缺陷 2:較早、無關的空中事件(搖桿置中 13° 傾角 / ROut 0.25 陣風 / 0.4 s 角速度未達成)不能蓋掉失事當下的證據。"""
    clean = Scene(flight=60.0, amp=0.0).tipover()
    a0 = clean.anchors(end_type="crash_check")
    d0 = run_checked(clean.ctx(), a0, "tipover clean")
    sc = Scene(flight=60.0, amp=0.0)
    E = sc.t_to + 20.0
    sc.lean_air(E, deg=13.0, dur=1.5)                                    # 搖桿一直置中:真的是 lean_without_pilot 事件
    w = (sc.tf >= E + 5.0) & (sc.tf < E + 5.3)
    sc.rout[w] = 1.2                                                     # 空中短暫碰到上限(CRIT 層,和失事那段同層級)
    w = (sc.tf >= E + 10.0) & (sc.tf < E + 10.6)
    sc.rout[w] = 0.25                                                    # 空中短暫輸出偏大(SOFT 層),已恢復
    w = (sc.tf >= E + 20.0) & (sc.tf < E + 20.4)
    sc.rdes[w], sc.r[w] = 60.0, 0.0                                      # 空中短暫角速度未達成(持續成立)
    sc.tipover()
    a = sc.anchors(end_type="crash_check")
    d = run_checked(sc.ctx(), a, "tipover + earlier air events")
    _tipover_expect(sc, d, a, "earlier events")                         # 失事當下的數字與乾淨的翻覆完全一樣
    for k in OWNED:
        assert d[k].status == d0[k].status and d[k].context == d0[k].context, (k, d[k].status, d0[k].status)
        if d0[k].t_onset is not None:                                   # 雜訊序列不同(lean_air 先抽了亂數)→ 容許 10 ms
            near(d[k].t_onset, d0[k].t_onset, 0.01, k)
    ln, rd, sat = d["lean_without_pilot"], d["rate_demand_not_achieved"], d["rate_output_saturation"]
    assert ln.values["n_episodes"] == 2 and ln.values["other_episodes"][0]["kind"] == "air", ln.values
    near(ln.values["other_episodes"][0]["t"], E + 0.3 * 12 / 13, 0.02, "較早的空中 lean 仍列在 values")
    assert hypotheses._lean_near_ground(ln, None), ln.values                    # H_LAND 的 lean 證據還在
    assert rd.values["n_episodes"] == 2 and rd.values["other_episodes"][0]["t"] < E + 21.0, rd.values
    assert sat.values["n_episodes"] == 3 and [e["tier"] for e in sat.values["other_episodes"]] == ["CRIT", "SOFT"], sat.values
    near(sat.values["t_soft"], sat.t_onset, 1e-3, "t_soft 屬於失事那一段,不是 E+10 的陣風")
    assert sat.values["m25_max_normal_roll"] < 0.2 and sat.values["raw_max_normal"] < 0.5, sat.values   # 基準不含其他事件
    assert "另有 2 段" in sat.text_zh, sat.text_zh
    # engine(只跑本模組):多了較早的事件,各假說分數不變(原寫法 H_LAND 1.74 → 0.84,H_MOTOR 多拿 rate 'air')
    s0, _ = _engine_attitude_only(clean)
    s1, _ = _engine_attitude_only(sc)
    for hid in ("H_LAND_TIPOVER", "H_NAV_FAILURE", "H_MOTOR_THRUST"):
        assert abs(s0[hid] - s1[hid]) < 1e-9, (hid, s0[hid], s1[hid])
    # 正常結束(沒有失事參考):取最強、最早的一段(只有那一次 13° 空中傾角)
    sc = Scene(flight=60.0, amp=0.0)
    sc.lean_air(E, deg=13.0, dur=1.5)
    f = pick(da.run(sc.ctx(), sc.anchors()), "lean_without_pilot")
    assert f.status == HIT and f.context == "air" and f.values["n_episodes"] == 1, (f.status, f.context)
    # 純函式:併段與挑選
    eps = [{"t0": 1.0, "t1": 2.0, "k": 0}, {"t0": 4.5, "t1": 5.0, "k": 0}, {"t0": 20.0, "t1": 21.0, "k": 1}]
    g = da.group_episodes(eps, 3.0)
    assert [len(x) for x in g] == [2, 1], g
    E2 = [{"t0": 1.0, "t1": 5.0, "k": 1}, {"t0": 50.0, "t1": 52.0, "k": 0}]
    assert da.choose_episode(E2, 51.0, lambda e, dd: (e["k"], -dd))["t0"] == 50.0      # 參考時間附近的優先(即使較弱)
    assert da.choose_episode(E2, None, lambda e, dd: (e["k"], -dd))["t0"] == 1.0       # 沒有參考:最強
    assert da.choose_episode(E2, 100.0, lambda e, dd: (-dd,))["t0"] == 50.0           # 都很早:最接近的
    print("  較早的空中 lean / ROut 陣風 / 角速度未達成只列 other_episodes;失事當下的 ground_idle 證據與分數不變")


def test_synth_lean_ground_requires_low_agl():
    """缺陷 3:油門在底但離地 10 m(空中油門崩潰)不能被當成「地面怠速」(原寫法 → ground_idle + near_ground → H_LAND +3)。"""
    for mode in (3, 5):
        sc = Scene(amp=0.0, alt_air=10.0)
        TL = sc.t_to + 30.0
        sc.lean_air(TL, deg=10.0, dur=1.5, mode=mode)
        sc.tho[(sc.ts >= TL - 0.1) & (sc.ts <= TL + 0.9)] = 0.0
        f = pick(run_checked(sc.ctx(), sc.anchors(), f"air idle mode {mode}").values(), "lean_without_pilot")
        assert f.status == CLEAR, (mode, f.status, f.context, f.text_zh)
        assert f.values["idle_above_ground_s"] >= 0.5 and f.values["agl_available"] is True, f.values
    # 沒有高度資料時才退回只看油門(文件化的備援),文字要說明
    sc = Scene(amp=0.0, alt_air=10.0)
    TL = sc.t_to + 30.0
    sc.lean_air(TL, deg=10.0, dur=1.5)
    sc.tho[(sc.ts >= TL - 0.1) & (sc.ts <= TL + 0.9)] = 0.0
    tb = sc.tables()
    del tb["CTUN"]["Alt"]
    f = pick(da.run(FakeCtx(tb, sc.params, [sc.interval()]), sc.anchors()), "lean_without_pilot")
    assert f.status == HIT and f.context == "ground_idle" and f.values["agl_at_onset_m"] is None, (f.status, f.context)
    assert "沒有高度資料" in f.text_zh and f.values["near_ground"] is True, f.text_zh
    # 翻覆的地面判讀照常(離地 0 m)
    sc = Scene().tipover()
    f = pick(da.run(sc.ctx(), sc.anchors(end_type="crash_check")), "lean_without_pilot")
    assert f.context == "ground_idle" and abs(f.values["agl_at_onset_m"]) < 0.05 and f.values["near_ground"] is True
    print("  離地 10 m 的油門崩潰 → 不判 GROUND(CLEAR、記 idle_above_ground_s);沒有高度資料才退回只看油門")


def test_synth_nan_rc_never_clear():
    """缺陷 4:RCIN 是 NaN 時不能說「駕駛平穩」或「沒有未下指令的偏航」。"""
    sc = Scene()
    tb = sc.tables()
    for c in list(tb["RCIN"]):
        if c != "t":
            tb["RCIN"][c] = np.full(tb["RCIN"]["t"].size, np.nan)
    d = check_contract(da.run(FakeCtx(tb, sc.params, [sc.interval()]), sc.anchors()), "NaN RCIN")
    assert d["pilot_input_context"].status == NA and d["pilot_input_context"].context == "no_data", d["pilot_input_context"]
    assert d["uncommanded_yaw_or_sink"].status == NA and d["lean_without_pilot"].status == NA, statuses(d.values())
    # 部分 NaN(空中 5 s):其他時段仍可判讀,NaN 樣本被計數、文字沒有 nan
    sc = Scene()
    sc.c1[(sc.ts >= sc.t_to + 20.0) & (sc.ts < sc.t_to + 25.0)] = np.nan
    p = run_checked(sc.ctx(), sc.anchors(), "partial NaN RCIN")["pilot_input_context"]
    assert p.status == CLEAR and 45 <= p.values["n_invalid_airborne"] <= 55, (p.status, p.values.get("n_invalid_airborne"))
    assert "nan" not in p.text_zh and p.values["precursor_max_dev"] is not None, p.text_zh
    # 偏航自轉期間 C4 是 NaN:不知道搖桿位置 → 不能說「沒打舵」(原寫法把 NaN 當置中)
    sc = Scene()
    T_y = sc.t_to + 25.0
    sc.yaw_spin(T_y)
    sc.c4[(sc.ts >= T_y - 1.5) & (sc.ts < T_y + 2.0)] = np.nan
    f = pick(run_checked(sc.ctx(), sc.anchors(), "yaw spin NaN C4").values(), "uncommanded_yaw_or_sink")
    assert not (f.status == HIT and "yaw" in f.context), (f.status, f.context)
    print("  RCIN 全 NaN → 駕駛 / lean / 偏航 NA;部分 NaN 只略過那段;C4 NaN 不算置中")


def test_throttle_context_clipped_to_interval():
    """缺陷 5:油門背景的窗口裁在解鎖區間內(上鎖後的 CTUN 不能決定 air / ground_idle)。"""
    ts = np.arange(0, 201) / SLOW
    tho = np.where(ts <= 10.0 + 1e-9, 0.0, 0.9)                      # 上鎖(10 s)後 ThO 變成垃圾 0.9
    ctx = FakeCtx({"CTUN": {"t": ts, "ThO": tho, "ThH": np.full(ts.size, 0.35)}}, {}, [])
    iv = ArmedInterval(0, 0.0, 10.0, 0)
    assert da._throttle_context(ctx, 9.6, 10.5)[0] == "air"             # 沒裁:一半是上鎖後的資料
    c, med, _ = da._throttle_context(ctx, 9.6, 10.5, iv)
    assert c == "ground_idle" and med == 0.0, (c, med)
    print("  油門背景窗口裁在 iv.t1:上鎖後的 ThO 不會把 ground_idle 變成 air")


def test_synth_brake_phase_excluded():
    """缺陷 7:搖桿回中 / 切入 LOITER 後的煞車傾角(預設 LOIT_BRK_ACCEL 250 → 14°)不是「搖桿置中卻被推翻」。"""
    def run_lean(release_dt, how="stick"):
        sc = Scene(amp=0.0)
        L = sc.t_to + 30.0
        if how == "stick":
            m = (sc.ts >= L - 3.0) & (sc.ts < L - 0.05)
            sc.c2[m] = sc.trims[1] + 200.0                              # 駕駛往前飛 3 s,在 L 放開
        else:
            sc.modes = [(sc.tf[0], 2), (L, 5)]                          # ALT_HOLD → LOITER(帶速度切入也會煞車)
        sc.lean_air(L + release_dt, deg=13.0, dur=1.5)
        return pick(run_checked(sc.ctx(), sc.anchors(), f"brake {how} {release_dt}").values(), "lean_without_pilot")
    f = run_lean(0.3)
    assert f.status == CLEAR and f.values["air_brake_lean_max"] >= 12.5, (f.status, f.values["air_brake_lean_max"])
    assert f.values["brake_window_s"] == {"loiter": 6.5, "poshold": 9.0}, f.values["brake_window_s"]
    f = run_lean(0.3, how="mode")
    assert f.status == CLEAR and f.values["air_brake_lean_max"] >= 12.5, (f.status, f.values)
    f = run_lean(7.0)                                                   # 煞車期(6.5 s)之後才傾斜 → 照報
    assert f.status == HIT and f.context == "air" and f.severity == "high", (f.status, f.context, f.severity)
    # 煞車期由參數決定(124:0.1 + 600/150 + 150/300 = 4.6 s),並裁在 [2, 10] s
    ctx = FakeCtx({}, {"LOIT_SPEED": 600.0, "LOIT_BRK_ACCEL": 150.0, "LOIT_BRK_DELAY": 0.1, "LOIT_BRK_JERK": 300.0,
                       "PHLD_BRAKE_ANGLE": 2000.0, "PHLD_BRAKE_RATE": 10.0}, [])
    lo, ph = da.brake_seconds(ctx)
    assert abs(lo - 4.6) < 1e-9 and abs(ph - (2 * 20 / 10 + 1.5)) < 1e-9, (lo, ph)
    assert da.brake_seconds(FakeCtx({}, {"LOIT_SPEED": 5000.0}, []))[0] == da.AIR_BRAKE_MAX_S
    print("  回中 / 切入 LOITER 後 6.5 s 內的 13° 煞車傾角 → CLEAR(記在 air_brake_lean_max);煞車期後才傾斜 → HIT high")


def test_synth_ae_crash_episode():
    """缺陷 7(ae):失事前已恢復的陣風不能當 t_warn;正常結束仍照規格取第一次;撞擊在先時錨點 = 撞擊。"""
    def gust_scene():
        sc = Scene(flight=60.0)
        G = sc.t_to + 20.0
        w = (sc.tf >= G) & (sc.tf < G + 0.6)
        sc.roll[w] += 14.0                                               # 外力讓機身偏 14° 0.6 s,之後恢復
        return sc, G
    sc, G = gust_scene()
    sc.tipover()
    a = sc.anchors(end_type="crash_check")
    d = run_checked(sc.ctx(), a, "gust + tipover")
    _tipover_expect(sc, d, a, "gust + tipover")                         # t_warn / lean / rate / sat 都是失事那一段
    al = d["attitude_loss_of_control"]
    assert len(al.values["earlier_warn_runs"]) == 1 and abs(al.values["earlier_warn_runs"][0][0] - G) < 0.01, al.values
    assert "已恢復" in al.text_zh and al.values["episode_rule"] == "crash_episode", al.text_zh
    # 正常結束:規格的「第一次持續越線」
    sc, G = gust_scene()
    a = sc.anchors()
    f = pick(da.run(sc.ctx(), a), "attitude_loss_of_control")
    assert f.status == HIT and abs(f.t_onset - G) < 0.01 and f.values["episode_rule"] == "first_crossing", (f.status, f.t_onset)
    # 撞擊在先、之前只有已恢復的陣風:SOFT 'recovered',錨點 = 撞擊(原寫法:t_warn = 30 s 前的陣風,擋掉撞擊優先規則)
    sc, G = gust_scene()
    hit = sc.t_to + 50.0
    a = sc.anchors(end_type="hard_impact", t_first_hit=hit)
    f = pick(run_checked(sc.ctx(), a, "gust + impact").values(), "attitude_loss_of_control")
    assert f.status == SOFT and f.context == "recovered" and abs(f.t_onset - G) < 0.01, (f.status, f.context, f.t_onset)
    assert a.t_warn == a.t_loss == hit and f.values["anchor_mode"] == "impact_first", (a.t_warn, a.t_loss)
    # 純函式:段內短暫回落(124:4.49° 0.42 s)不算恢復;回落 < 5° 持續 2.5 s 才算
    t = 100.0 + np.arange(0, 4.0, 1 / FAST)
    for dip_s, warn0 in ((0.4, 100.0), (2.5, 101.0 + 2.5)):
        ae = np.where(t < 101.0, 12.0, np.where(t < 101.0 + dip_s, 4.0, 35.0))
        ev = da.attitude_events(t, ae, np.zeros(t.size), da.GAP_FAST_S, t_crash=float(t[-1]))
        near(ev["warn"][0], warn0, 1e-6, f"dip {dip_s} s → t_warn")
        assert ev["loss"] is not None and len(ev["earlier_warn"]) == (1 if dip_s > 2 else 0), ev
    print("  已恢復的陣風 → earlier_warn_runs,t_warn = 失事那段;正常結束照規格;撞擊在先 → SOFT recovered + 撞擊錨點")


def test_synth_text_caveats():
    """缺陷 6:文字不寫死 124 的數字當成任意機體的「正常」、不在不平穩時寫「平穩」、標出未判讀的範圍。"""
    sc = Scene()
    T5 = sc.t_to + 25.0
    w = (sc.tf >= T5) & (sc.tf < T5 + 0.5)
    sc.roll[w] += 7.0                                                    # ae 7°(資訊級)0.5 s
    f = pick(run_checked(sc.ctx(), sc.anchors(), "ae info").values(), "attitude_loss_of_control")
    assert f.status == SOFT and f.context == "minor", (f.status, f.context)
    assert "00000124" in f.text_zh and "此前追蹤誤差最大" in f.text_zh and "正常飛行最大約" not in f.text_zh, f.text_zh
    sc = Scene(amp=0.0)
    sc.lean_air(sc.t_to + 30.0)
    f = pick(da.run(sc.ctx(), sc.anchors()), "lean_without_pilot")
    assert "00000124" in f.text_zh and "9.7" not in f.text_zh and "煞車期" in f.text_zh, f.text_zh
    f = pick(da.run(Scene().ctx(), Scene().anchors()), "lean_without_pilot")
    assert f.status == CLEAR and "AUTO / GUIDED / RTL" in f.text_zh and "未判讀" in f.text_zh, f.text_zh
    print("  SOFT / 空中 lean 文字註明 124 校準;lean CLEAR 註明自主模式與煞車期未判讀")


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
