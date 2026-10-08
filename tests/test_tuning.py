r"""PID 調參分析引擎(core/tuning.py)測試:python tests\test_tuning.py

只用合成訊號(已知延遲、已知二階系統、注入正弦、固定偏差、飽和、低取樣率)驗證
訊號處理工具與規則;再跑內建模擬 log、core.sim_tuning 情境矩陣(10 情境 × 3 種記錄設定,約 35 s;
TUNING_MATRIX=0 略過、TUNING_MATRIX_SEED 換種子)與使用者的實機 log(唯讀,不存在就略過)。
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import sim, tuning as T  # noqa: E402
from core.dflog import LogIndex  # noqa: E402
from core.logio import index_events  # noqa: E402
from core.tuning_model import (DIR_CHECK, DIR_DOWN, DIR_SET, DIR_UP, SEV_BAD, SEV_INFO,  # noqa: E402
                               SEV_OK, SEV_ORDER, SEV_WARN, Finding, Suggestion, TuningReport)
from core.types import EV_ARM, EV_DISARM, EV_MODE, LogData, LogEvent  # noqa: E402

REAL_LOG = os.environ.get("TUNING_REAL_LOG",
                          r"C:\Users\admin\Downloads\203 2025-3-7 上午 10-53-08.bin")

DEFAULT_PARAMS = {
    "ATC_RAT_RLL_P": 0.135, "ATC_RAT_RLL_I": 0.135, "ATC_RAT_RLL_D": 0.0036,
    "ATC_RAT_PIT_P": 0.135, "ATC_RAT_PIT_I": 0.135, "ATC_RAT_PIT_D": 0.0036,
    "ATC_RAT_YAW_P": 0.18, "ATC_RAT_YAW_I": 0.018, "ATC_RAT_YAW_D": 0.0,
    "ATC_RAT_RLL_IMAX": 0.5, "ATC_RAT_PIT_IMAX": 0.5, "ATC_RAT_YAW_IMAX": 0.5,
    "ATC_RAT_RLL_FLTD": 20.0, "ATC_RAT_PIT_FLTD": 20.0,
    "ATC_ANG_RLL_P": 4.5, "ATC_ANG_PIT_P": 4.5, "ATC_ANG_YAW_P": 4.5, "ATC_INPUT_TC": 0.15,
    "ATC_ACCEL_R_MAX": 110000.0, "ATC_ACCEL_P_MAX": 110000.0, "ATC_ACCEL_Y_MAX": 27000.0,
    "LOG_BITMASK": 176126.0, "INS_GYRO_FILTER": 20.0, "INS_HNTCH_ENABLE": 0.0, "INS_LOG_BAT_MASK": 0.0,
    "MOT_THST_HOVER": 0.35, "MOT_HOVER_LEARN": 2.0, "MOT_PWM_MIN": 1000.0, "MOT_PWM_MAX": 2000.0,
    "MOT_SPIN_MIN": 0.15, "MOT_SPIN_MAX": 0.95, "FRAME_CLASS": 1.0, "FRAME_TYPE": 1.0,
    "SERVO1_FUNCTION": 33.0, "SERVO2_FUNCTION": 34.0, "SERVO3_FUNCTION": 35.0, "SERVO4_FUNCTION": 36.0,
}


# =====================================================================
# 合成訊號工具
# =====================================================================

def sim2(u: np.ndarray, fs: float, fn: float, zeta: float, delay: float = 0.0) -> np.ndarray:
    """二階系統 wn²/(s² + 2ζwn s + wn²) + 純延遲,精確 ZOH 離散化。"""
    wn = 2 * np.pi * fn
    A = np.array([[0.0, 1.0], [-wn * wn, -2 * zeta * wn]])
    B = np.array([0.0, wn * wn])
    w, V = np.linalg.eig(A)
    Ad = (V @ np.diag(np.exp(w / fs)) @ np.linalg.inv(V)).real
    Bd = np.linalg.solve(A, (Ad - np.eye(2)) @ B)
    nd = int(round(delay * fs))
    if nd:
        u = np.concatenate([np.full(nd, u[0]), u[:-nd]])
    a11, a12, a21, a22 = Ad.ravel()
    b1, b2 = Bd
    x1 = x2 = 0.0
    y = np.empty(u.size)
    for k, uk in enumerate(u.tolist()):
        y[k] = x1
        x1, x2 = a11 * x1 + a12 * x2 + b1 * uk, a21 * x1 + a22 * x2 + b2 * uk
    return y


def stick(n: int, fs: float, rng, amp: float = 80.0, tc: float = 0.03) -> np.ndarray:
    """像搖桿的隨機階梯(0.2–1 s 一段)經一階平滑(時間常數 tc)。"""
    u = np.zeros(n)
    i = 0
    while i < n:
        L = int(rng.uniform(0.2, 1.0) * fs)
        u[i:i + L] = rng.choice([-1, 0, 1]) * rng.uniform(0.3, 1.0) * amp
        i += L
    a = 1 - math.exp(-1 / (fs * tc))
    y = np.empty(n)
    acc = 0.0
    for k, v in enumerate(u.tolist()):
        acc += a * (v - acc)
        y[k] = acc
    return y


def bandlimited(n: int, fs: float, f_hi: float, rng) -> np.ndarray:
    X = np.fft.rfft(rng.normal(0, 1, n))
    X[np.fft.rfftfreq(n, 1 / fs) > f_hi] = 0
    x = np.fft.irfft(X, n)
    return x / x.std()


def frac_delay(x: np.ndarray, fs: float, tau: float) -> np.ndarray:
    """以頻域相移做精確的分數延遲(循環)。"""
    n = x.size
    f = np.fft.rfftfreq(n, 1 / fs)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * tau), n)


# =====================================================================
# 合成 log
# =====================================================================

def lpf1(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """一階低通(同 AC_PID 的 FLTT:alpha = dt / (dt + 1/(2π fc)))。"""
    a = (1 / fs) / (1 / fs + 1 / (2 * np.pi * fc))
    y = np.empty(x.size)
    acc = float(x[0])
    for k, v in enumerate(x.tolist()):
        acc += a * (v - acc)
        y[k] = acc
    return y


def make_log(*, fs=400.0, dur=80.0, att_fs=None, fn=8.0, zeta=0.7, delay=0.004, osc=None,
             noise=0.5, bias=None, sat_frac=0.0, vibe=5.0, clip_inc=0, pid=False, pid_dmod=1.0,
             batch_peak=None, imu_fs=None, ctrl=False, params=None, mode="STABILIZE",
             ev=True, seed=0, amp=80.0, out_offset=None, quadplane=False, modes=None,
             out_sat=None, ctrl_pd=(0.02, 0.03), with_rate=True, pid_fltt=None, qtun=True,
             gaps=None) -> LogData:
    """合成 ArduCopter log(四軸 X):角速度環 = 二階系統,可注入震盪、偏差、飽和……

    with_rate=False:不寫 RATE(只有 PIDx);pid_fltt:PIDx.Tar 經 FLTT 一階低通(同 ArduPilot);
    quadplane:Plane 的 CTUN(ThO = 推進油門 0)+ QTUN(VTOL 升力油門;qtun=False 不寫)。
    gaps:[(t0, t1)…] 這些時段的 ATT / RATE / PIDx 列全部刪掉(記錄空洞)。
    """
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    t_arm, t_to, t_land, t_dis = 1.0, 3.0, dur - 4.0, dur - 2.0
    air = (t >= t_to) & (t <= t_land)
    osc = osc or {}
    bias = bias or {}
    tables: dict[str, dict] = {}
    rate = {"t": t}
    att = {"t": t}
    pidt = {}
    for ax, (fd, fa, fo), (ad, aa) in zip(("roll", "pitch", "yaw"), T._RATE_FIELDS.values(),
                                         T._ATT_FIELDS.values()):
        des = stick(n, fs, rng, amp=amp * (0.6 if ax == "yaw" else 1.0)) * air
        act = sim2(des, fs, fn if ax != "yaw" else fn / 2, zeta, delay)
        act = act + rng.normal(0, noise, n)
        if ax in osc:
            f0, a0 = osc[ax]
            act = act + a0 * np.sqrt(2) * np.sin(2 * np.pi * f0 * t + 0.3) * air
        if ax in bias:
            act = act - bias[ax] * air * (np.abs(des) < 5)
        out = 0.004 * des + 0.02 * rng.normal(0, 1, n)
        if out_offset and ax in out_offset:
            out = out + out_offset[ax]
        if out_sat and ax in out_sat:                 # 一部分時間輸出頂到 ±0.97
            hit = rng.random(n) < out_sat[ax]
            out = np.where(hit, 0.97 * np.sign(des + 1e-9), out)
        rate[fd], rate[fa], rate[fo] = des, act, out * air
        att[ad] = np.cumsum(des) / fs
        att[aa] = np.cumsum(act) / fs
        if pid:
            r = np.radians
            dm = np.where(rng.random(n) < (1 - pid_dmod), 0.5, 1.0)
            tar = lpf1(des, fs, pid_fltt) if pid_fltt else des
            pidt[ax] = {"t": t, "Tar": r(tar), "Act": r(act), "Err": r(des - act),
                        "P": 0.135 * r(des - act), "I": np.full(n, 0.02), "D": 0.0036 * np.gradient(r(des - act)) * fs,
                        "FF": np.zeros(n), "DFF": np.zeros(n), "Dmod": dm, "SRate": np.zeros(n),
                        "Flags": np.zeros(n, np.uint8)}
    tables["RATE"] = rate
    if att_fs:
        k = int(round(fs / att_fs))
        tables["ATT"] = {c: v[::k] for c, v in att.items()}
        if att_fs < fs:            # 低速記錄:RATE 也同樣降採樣
            tables["RATE"] = {c: v[::k] for c, v in rate.items()}
    else:
        tables["ATT"] = att
    if pid:
        for ax, msg in (T._PIQ_MSG if quadplane else T._PID_MSG).items():
            tables[msg] = pidt[ax]
    if not with_rate:
        del tables["RATE"]
    for g0, g1 in (gaps or []):
        for name in [k_ for k_ in tables if k_ in ("ATT", "RATE") or k_ in T._PID_MSG.values()
                     or k_ in T._PIQ_MSG.values()]:
            keep = ~((tables[name]["t"] >= g0) & (tables[name]["t"] < g1))
            tables[name] = {c: v[keep] for c, v in tables[name].items()}
    t10 = np.arange(0, dur, 0.1)
    a10 = (t10 >= t_to) & (t10 <= t_land)
    if quadplane:
        # ArduPlane:CTUN.ThO 是固定翼推進油門(VTOL 懸停時 0)、沒有 Alt / CRt;升力油門在 QTUN(CRt 為 m/s)
        tables["CTUN"] = {"t": t10, "NavRoll": np.zeros(t10.size), "Roll": np.zeros(t10.size),
                          "ThO": np.zeros(t10.size), "ThD": np.zeros(t10.size)}
        if qtun:
            tables["QTUN"] = {"t": t10, "ThO": np.where(a10, 0.35, 0.0), "Alt": np.where(a10, 5.0, 0.0),
                              "CRt": np.zeros(t10.size)}
    else:
        tables["CTUN"] = {"t": t10, "ThO": np.where(a10, 0.35, 0.0), "Alt": np.where(a10, 5.0, 0.0),
                          "CRt": np.zeros(t10.size)}
    tables["VIBE[0]"] = {"t": t10, "IMU": np.zeros(t10.size, np.uint8),
                         "VibeX": np.full(t10.size, vibe), "VibeY": np.full(t10.size, vibe),
                         "VibeZ": np.full(t10.size, vibe * 1.5),
                         "Clip": np.round(np.linspace(0, clip_inc, t10.size)).astype(np.uint32)}
    mot = 1500 + 20 * rng.normal(0, 1, (t10.size, 4))
    if sat_frac > 0:
        hit = rng.random(t10.size) < sat_frac
        mot[hit & a10, 0] = 1990
    mot[~a10] = 1000
    tables["RCOU"] = {"t": t10, **{f"C{i + 1}": np.clip(mot[:, i], 1000, 2000).astype(np.uint16)
                                   for i in range(4)}}
    if imu_fs:
        k = int(round(fs / imu_fs))
        tables["IMU[0]"] = {"t": t[::k], "I": np.zeros(t[::k].size, np.uint8),
                            "GyrX": np.radians(rate["R"][::k]), "GyrY": np.radians(rate["P"][::k]),
                            "GyrZ": np.radians(rate["Y"][::k])}
    if ctrl:
        tables["CTRL"] = {"t": t10, "RMSRollP": np.full(t10.size, ctrl_pd[0]),
                          "RMSRollD": np.full(t10.size, ctrl_pd[1]),
                          "RMSPitchP": np.full(t10.size, 0.02), "RMSPitchD": np.full(t10.size, 0.01),
                          "RMSYaw": np.full(t10.size, 0.01)}
    if batch_peak:
        bfs, bn = 1000.0, 512
        tb = np.arange(t_to + 1, t_land - 1, 1.0)
        hdr = {"t": [], "N": [], "type": [], "instance": [], "mul": [], "smp_cnt": [], "SampleUS": [],
               "smp_rate": []}
        dat = {"t": [], "N": [], "seqno": [], "x": [], "y": [], "z": []}
        mul = 938.0
        for j, tj in enumerate(tb):
            for typ in (0, 1):
                hdr["t"].append(tj + 0.001 * typ)
                for k_, v_ in (("N", 2 * j + typ), ("type", typ), ("instance", 0), ("mul", mul),
                               ("smp_cnt", bn), ("SampleUS", int(tj * 1e6)), ("smp_rate", bfs)):
                    hdr[k_].append(v_)
                ts = np.arange(bn) / bfs
                xyz = [0.02 * rng.normal(0, 1, bn) + (0.08 * np.sin(2 * np.pi * batch_peak * ts) if typ == 1 else 0)
                       for _ in range(3)]
                raw = [np.clip(np.round(v * mul), -32767, 32767).astype(np.int16).reshape(-1, 32) for v in xyz]
                for s_ in range(bn // 32):
                    dat["t"].append(tj + 0.001 * typ + 0.0001 * (s_ + 1))
                    dat["N"].append(2 * j + typ)
                    dat["seqno"].append(s_)
                    dat["x"].append(raw[0][s_])
                    dat["y"].append(raw[1][s_])
                    dat["z"].append(raw[2][s_])
        tables["ISBH"] = {k_: np.asarray(v_) for k_, v_ in hdr.items()}
        tables["ISBD"] = {"t": np.asarray(dat["t"]), "N": np.asarray(dat["N"], np.uint16),
                          "seqno": np.asarray(dat["seqno"], np.uint16),
                          **{c: np.vstack(dat[c]).astype(np.int16) for c in ("x", "y", "z")}}
    prm = dict(DEFAULT_PARAMS)
    if quadplane:                    # QuadPlane:ATC_ → Q_A_、MOT_ → Q_M_、FRAME_ → Q_FRAME_
        prm = {("Q_A_" + k[4:] if k.startswith("ATC_") else "Q_M_" + k[4:] if k.startswith("MOT_")
                else "Q_" + k if k.startswith("FRAME_") else k): v for k, v in prm.items()}
        prm["Q_ENABLE"] = 1.0
    prm.update(params or {})
    names = sorted(prm)
    tables["PARM"] = {"t": np.full(len(names), 0.01), "Name": np.array(names),
                      "Value": np.array([prm[k_] for k_ in names], float)}
    tables["MSG"] = {"t": np.array([0.0, 0.001]), "Message": np.array(["ArduCopter V4.5.7 (test)", "Frame: QUAD/X"])}
    tables["MODE"] = {"t": np.array([0.5]), "Mode": np.array([0], np.uint8), "ModeNum": np.array([0], np.uint8)}
    evs = [(t_arm, 10), (t_dis, 11)]
    if ev:
        evs += [(t_to, 28), (t_land, 17), (t_land + 0.5, 18)]
    evs.sort()
    tables["EV"] = {"t": np.array([e[0] for e in evs]), "Id": np.array([e[1] for e in evs], np.uint8)}
    if quadplane:
        tables["MSG"]["Message"] = np.array(["ArduPlane V4.5.7 (test)", "QuadPlane initialised"])
    idx = LogIndex.from_tables(tables, vehicle="plane" if quadplane else "copter",
                               units={"RATE": {"RDes": "deg/s", "R": "deg/s", "PDes": "deg/s", "P": "deg/s",
                                               "YDes": "deg/s", "Y": "deg/s"},
                                      "ATT": {"DesRoll": "deg", "Roll": "deg"},
                                      "IMU": {"GyrX": "rad/s", "GyrY": "rad/s", "GyrZ": "rad/s"}})
    events = [LogEvent(t_arm, EV_ARM), LogEvent(t_dis, EV_DISARM)]
    events += [LogEvent(tm, EV_MODE, name) for tm, name in (modes or [(0.5, mode)])]
    events += index_events(idx)
    events.sort(key=lambda e: e.t)
    return LogData(sources={}, events=events, index=idx)


def codes(rep: TuningReport, axis=None) -> list[str]:
    return [f.code for f in rep.findings if axis is None or f.axis == axis]


def finding(rep: TuningReport, code: str, axis=None):
    for f in rep.findings:
        if f.code == code and (axis is None or f.axis == axis):
            return f
    return None


def no_gain_increase(rep: TuningReport) -> bool:
    return not any(s.direction == DIR_UP and T._is_gain(s.param) for f in rep.findings for s in f.suggestions)


# =====================================================================
# 訊號處理
# =====================================================================

def test_welch_sine_and_noise():
    rng = np.random.default_rng(1)
    fs, n, A, sig = 200.0, 200 * 120, 10.0, 1.0
    t = np.arange(n) / fs
    x = A * np.sin(2 * np.pi * 7.0 * t) + rng.normal(0, sig, n)
    f, p = T.welch_psd(x, fs, 512)
    rms = T.band_rms(f, p, 6.0, 8.0)
    assert abs(rms - A / np.sqrt(2)) / (A / np.sqrt(2)) < 0.03, rms
    floor = np.median(p[(f > 20) & (f < 90)])
    assert abs(floor - 2 * sig ** 2 / fs) / (2 * sig ** 2 / fs) < 0.1, floor
    # 白雜訊總功率 = 變異數
    f2, p2 = T.welch_psd(rng.normal(0, 3.0, n), fs)
    tot = float(np.sum(p2) * (f2[1] - f2[0]))
    assert abs(tot - 9.0) / 9.0 < 0.05, tot
    # 多段輸入 = 各段的窗一起平均
    f3, p3 = T.welch_psd([x[: n // 2], x[n // 2:]], fs, 512)
    assert abs(T.band_rms(f3, p3, 6, 8) - rms) / rms < 0.02
    print(f"  正弦 RMS {rms:.3f}(理論 {A / np.sqrt(2):.3f});白雜訊位準 {floor:.4f}(理論 {2 * sig ** 2 / fs:.4f});總功率 {tot:.2f}")


def test_csd_coherence():
    rng = np.random.default_rng(2)
    fs, n = 400.0, 400 * 60
    x = rng.normal(0, 1, n)
    y = 2.0 * frac_delay(x, fs, 0.01) + rng.normal(0, 0.1, n)
    f, pxy = T.csd(x, y, fs, 256)
    _, pxx = T.welch_psd(x, fs, 256)
    m = (f > 1) & (f < 150)
    g = np.abs(pxy[m]) / pxx[m]
    assert np.all(np.abs(g - 2.0) < 0.15), (g.min(), g.max())
    ph = np.unwrap(np.angle(pxy[m]))            # y 落後 → 相位 = −2πfτ
    slope = np.polyfit(f[m], ph, 1)[0]
    assert abs(-slope / (2 * np.pi) - 0.01) < 5e-4, slope
    _, c = T.coherence(x, y, fs, 256)
    assert np.mean(c[m]) > 0.98, np.mean(c[m])
    _, c2 = T.coherence(x, rng.normal(0, 1, n), fs, 256)
    assert np.mean(c2[m]) < 0.05, np.mean(c2[m])
    print(f"  |Pxy|/Pxx ≈ 2(範圍 {g.min():.2f}–{g.max():.2f});相位斜率延遲 {-slope / (2 * np.pi) * 1000:.2f} ms;"
          f"同調 {np.mean(c[m]):.3f} / 無關訊號 {np.mean(c2[m]):.3f}")


def test_find_peaks():
    rng = np.random.default_rng(3)
    fs, n = 400.0, 400 * 90
    t = np.arange(n) / fs
    x = 5 * np.sin(2 * np.pi * 7.3 * t) + 2 * np.sin(2 * np.pi * 23.0 * t) + rng.normal(0, 2, n)
    f, p = T.welch_psd(x, fs, 512)
    pk = T.find_peaks(f, p, fmin=1, fmax=160, min_rms=0.5)
    assert len(pk) >= 2, pk
    assert abs(pk[0].f - 7.3) < 0.3 and abs(pk[0].rms - 5 / np.sqrt(2)) / (5 / np.sqrt(2)) < 0.1, pk[0]
    assert abs(pk[1].f - 23.0) < 0.3 and abs(pk[1].rms - 2 / np.sqrt(2)) / (2 / np.sqrt(2)) < 0.15, pk[1]
    # 純雜訊、1/f² 斜坡:不該有顯著峰
    f2, p2 = T.welch_psd(rng.normal(0, 2, n), fs, 512)
    assert not T.find_peaks(f2, p2, fmin=1, min_rms=0.5), T.find_peaks(f2, p2, fmin=1, min_rms=0.5)
    walk = np.cumsum(rng.normal(0, 1, n))
    f3, p3 = T.welch_psd(walk, fs, 512)
    assert not T.find_peaks(f3, p3, fmin=0.5, fmax=100, min_rms=0.05)
    # 靠近 Nyquist 的寬峰(像 25 Hz 取樣看到的 9–10 Hz 震盪)
    fs4, n4 = 25.0, 25 * 300
    t4 = np.arange(n4) / fs4
    hump = sum(np.sin(2 * np.pi * fr * t4 + rng.uniform(0, 6)) for fr in np.linspace(8.5, 10.2, 7))
    x4 = 1.2 * hump + bandlimited(n4, fs4, 2.0, rng) * 6 + rng.normal(0, 1.0, n4)
    f4, p4 = T.welch_psd(x4, fs4)
    pk4 = T.find_peaks(f4, p4, fmin=4, fmax=10.0, min_rms=1.0)
    assert pk4 and 8.0 < pk4[0].f < 10.5, pk4
    print(f"  峰 {pk[0].f:.2f} Hz RMS {pk[0].rms:.2f}、{pk[1].f:.2f} Hz RMS {pk[1].rms:.2f};雜訊 / 斜坡無峰;"
          f"近 Nyquist 寬峰 {pk4[0].f:.1f} Hz")


def test_freq_response_second_order():
    rng = np.random.default_rng(4)
    fs, n = 400.0, 400 * 120
    u = bandlimited(n, fs, 60.0, rng) * 50
    for zeta, fn in ((0.3, 8.0), (0.5, 12.0)):
        y = sim2(u, fs, fn, zeta) + rng.normal(0, 0.5, n)
        fr = T.freq_response(u, y, fs, 512)
        hp = 1 / (2 * zeta * np.sqrt(1 - zeta ** 2))
        fp = fn * np.sqrt(1 - 2 * zeta ** 2)
        wb = np.sqrt(1 - 2 * zeta ** 2 + np.sqrt(4 * zeta ** 4 - 4 * zeta ** 2 + 2))
        assert abs(fr.peak_mag - hp) / hp < 0.08, (zeta, fr.peak_mag, hp)
        assert abs(fr.peak_hz - fp) < 1.0, (fr.peak_hz, fp)
        assert fr.bandwidth_hz is not None and abs(fr.bandwidth_hz - wb * fn) / (wb * fn) < 0.08, \
            (fr.bandwidth_hz, wb * fn)
        print(f"  ζ={zeta} fn={fn}:|H|峰 {fr.peak_mag:.2f}(理論 {hp:.2f})@ {fr.peak_hz:.1f} Hz(理論 {fp:.1f}),"
              f"頻寬 {fr.bandwidth_hz:.1f} Hz(理論 {wb * fn:.1f})")
    # 窗數太少:同調不可信 → 不判斷
    fr2 = T.freq_response(u[:2048], u[:2048], fs, 512)
    assert fr2.peak_mag is None and fr2.bandwidth_hz is None


def test_step_response_wiener():
    rng = np.random.default_rng(5)
    fs, n = 400.0, 400 * 120
    u = stick(n, fs, rng)
    for zeta in (0.3, 0.4, 0.7):
        y = sim2(u, fs, 8.0, zeta) + rng.normal(0, 1.0, n)
        st = T.step_response_wiener(u, y, fs)
        ov = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2))
        assert st.n_windows >= 50 and st.overshoot_pct is not None
        assert abs(st.overshoot_pct - ov) < 7.0, (zeta, st.overshoot_pct, ov)
        assert abs(st.y[-1] - 1.0) < 0.1
        print(f"  ζ={zeta}:過衝 {st.overshoot_pct:.1f} %(理論 {ov:.1f} %),上升 {st.rise_s * 1000:.0f} ms,"
              f"{st.n_windows} 窗")
    # 純延遲 30 ms:上升前要有約 30 ms 的平台
    y = frac_delay(u, fs, 0.03)
    st = T.step_response_wiener(u, y, fs)
    i10 = int(np.argmax(st.y >= 0.5))
    assert abs(st.t[i10] - 0.03) < 0.01, st.t[i10]
    # 輸入太平滑 → 不給步階(避免假過衝)
    d = bandlimited(int(25 * 180), 25.0, 0.6, rng) * 20
    st2 = T.step_response_wiener(d, frac_delay(d, 25.0, 0.1), 25.0, 4.0, 1.5, f_cut=10,
                                 min_input_bw=T.STEP_MIN_INPUT_BW_ANG)
    assert st2.t.size == 0 and st2.n_windows > 0
    # 10 Hz 的角速度資料:0.5 s 響應只有 5 點 → 不算
    assert T.step_response_wiener(u[::40], u[::40], 10.0).t.size == 0
    m = T.step_metrics(np.arange(5) * 0.1, np.array([0, 0.5, 1.2, 1.0, 1.0]))
    assert abs(m[0] - 20.0) < 1e-9 and abs(m[2] - 0.3) < 1e-9, m
    print("  純延遲 30 ms 正確、平滑輸入與 10 Hz 資料不給步階、step_metrics 正確")


def test_lag_xcorr():
    rng = np.random.default_rng(6)
    fs, n = 400.0, 400 * 60
    x = bandlimited(n, fs, 15.0, rng)
    lag, rho = T.lag_xcorr(x, frac_delay(x, fs, 0.0375), fs, 0.3)
    assert abs(lag - 0.0375) < 0.002 and rho > 0.95, (lag, rho)
    lag2, _ = T.lag_xcorr(frac_delay(x, fs, 0.02), x, fs, 0.3)       # 負延遲 = x 落後
    assert abs(lag2 + 0.02) < 0.002, lag2
    x10 = bandlimited(10 * 300, 10.0, 2.0, rng)
    lag3, _ = T.lag_xcorr(x10, frac_delay(x10, 10.0, 0.06), 10.0, 0.5)
    assert abs(lag3 - 0.06) < 0.02, lag3
    pieces = [x[:8000], x[10000:20000]]
    lag4, _ = T.lag_xcorr(pieces, [frac_delay(p, fs, 0.05) for p in pieces], fs, 0.3)
    assert abs(lag4 - 0.05) < 0.003, lag4
    lag5, rho5 = T.lag_xcorr(x, frac_delay(x, fs, 0.5), fs, 0.2)         # 超出搜尋範圍:不可信
    assert lag5 is None or rho5 < T.LAG_MIN_RHO, (lag5, rho5)
    print(f"  400 Hz:{lag * 1000:.2f} ms(真 37.5);反向 {lag2 * 1000:.2f} ms;"
          f"10 Hz:{lag3 * 1000:.0f} ms(真 60);多段 {lag4 * 1000:.1f} ms")


def test_level_stats_bias_lag_peaks():
    rng = np.random.default_rng(7)
    fs, n = 400.0, 400 * 60
    t = 100.0 + np.arange(n) / fs
    des = stick(n, fs, rng, amp=60)
    act = sim2(des, fs, 8.0, 0.7, 0.01) + rng.normal(0, 0.5, n)
    act = act - 3.0 * (np.abs(des) < 5)                                  # 準穩態偏差 +3 °/s
    act = act + 6 * np.sqrt(2) * np.sin(2 * np.pi * 22.0 * t)            # 22 Hz、RMS 6 的自激震盪
    ls = T.level_stats([(t, [des, act])], fs, "rate", "roll")
    assert abs(ls.bias - 3.0) < 0.3, ls.bias
    assert 20.0 < ls.lag_ms < 60.0, ls.lag_ms
    pk = max(ls.peaks, key=lambda p: p.rms)
    assert abs(pk.f - 22.0) < 0.5 and abs(pk.rms - 6.0) / 6.0 < 0.15 and pk.sustained, pk
    assert pk.t_focus is not None and t[0] <= pk.t_focus <= t[-1]
    # 追蹤指令的延遲誤差和期望同調:不能被當成震盪
    ls2 = T.level_stats([(t, [des, sim2(des, fs, 3.0, 0.9, 0.1)])], fs, "rate", "roll")
    assert not [p for p in ls2.peaks if p.rms >= T.OSC_RATE_WARN_RMS], ls2.peaks
    print(f"  偏差 {ls.bias:+.2f} °/s、延遲 {ls.lag_ms:.0f} ms、峰 {pk.f:.1f} Hz RMS {pk.rms:.1f}(持續 {pk.sustain:.0%});"
          f"純追蹤延遲不算震盪")


def test_make_pieces_no_interp_lowpass():
    """規則取樣直接用原始樣本:接近 Nyquist 的震盪不能被內插壓掉。"""
    fs = 25.0
    t = 10 + np.arange(int(fs * 120)) / fs + 0.0004 * np.sin(np.arange(int(fs * 120)))   # 微小抖動
    x = np.sin(2 * np.pi * 10.0 * t)
    pcs = T._make_pieces(t, [x], [(t[0], t[-1])], fs)
    f, p = T.welch_psd([q[1][0] for q in pcs], fs)
    assert abs(T.band_rms(f, p, 9, 11) - np.sqrt(0.5)) < 0.05
    # 遇資料空洞切段
    t2 = np.concatenate([t[:1000], t[1500:]])
    pcs2 = T._make_pieces(t2, [np.sin(t2)], [(t2[0], t2[-1])], fs)
    assert len(pcs2) == 2
    print("  10 Hz @ 25 Hz 振幅保持;空洞切成 2 段")


def test_make_pieces_dropped_samples():
    """SP-2:0.1–1 % 的掉訊息(SD 卡忙)不能把資料切碎丟光;原始樣本放回格點、只內插缺點。"""
    fs = 400.0
    rng = np.random.default_rng(3)
    n = int(fs * 90)
    t = 50.0 + np.arange(n) / fs + rng.normal(0, 2e-5, n)            # 排程抖動 ±20 µs
    x = 5 * np.sin(2 * np.pi * 7.7 * t) + rng.normal(0, 0.3, n)
    for frac in (0.001, 0.003, 0.01, 0.03):
        keep = rng.random(n) >= frac
        pcs = T._make_pieces(t[keep], [x[keep]], [(t[0], t[-1])], fs)
        cov = T._coverage(pcs, [(t[0], t[-1])])
        assert cov > 0.9, (frac, cov)                                   # 以前:0.3 % → 42 %、1 % → 0 %
        f, p = T.welch_psd([q[1][0] for q in pcs], fs)
        amp = T.band_rms(f, p, 6.5, 9.0)
        assert abs(amp - 5 / np.sqrt(2)) < 0.25, (frac, amp)
        # 保留的原始樣本完全不動(只內插缺的格點)
        tt, (xx,) = pcs[0]
        hit = np.isin(np.round(tt, 9), np.round(t[keep], 9))
        assert hit.mean() > 0.9
    print("  掉 0.1–3 % 的樣本:涵蓋率 > 90 %、7.7 Hz 振幅不變")


def test_low_coverage():
    """SP-2:記錄有大量空洞、能分析的片段不到一半 → low_coverage 提醒;涵蓋率寫進 data / metrics。"""
    gaps = [(t0, t0 + 1.6) for t0 in np.arange(4.0, 66.0, 2.4)]        # 每 2.4 s 只剩 0.8 s(< MIN_PIECE_S)
    rep = T.analyze(make_log(fs=400.0, dur=70.0, gaps=gaps))
    f = finding(rep, "low_coverage")
    assert f is not None and f.severity == SEV_INFO and rep.data["coverage"] < 0.5, (codes(rep), rep.data.get("coverage"))
    rep2 = T.analyze(make_log(fs=400.0, dur=70.0))
    assert finding(rep2, "low_coverage") is None and rep2.data["coverage"] > 0.98
    assert rep2.axes["roll"].metrics["rate_coverage"] > 0.98
    print(f"  有空洞:涵蓋率 {rep.data['coverage'] * 100:.0f} % → low_coverage;正常 {rep2.data['coverage'] * 100:.0f} %")


def test_ctrl_fest_discrete():
    """SP-8:CTRL D/P → 等效頻率用離散 PID 模型(迴圈頻率的後向差分 + FLTD);接近上限只給下限。"""
    kp, kd, fltd, loop = 0.135, 0.0036, 11.0, 400.0
    r = T.dp_ratio_model(kp, kd, fltd, loop, np.linspace(0.1, 199.9, 4000))
    assert abs(r.max() - 1.697) < 0.01 and np.all(np.diff(r) > 0)     # 連續模型的漸近值是 1.84
    for ratio, f_true in ((1.045, 7.91), (1.2, 10.11)):
        f, gt = T._invert_dp(ratio, kp, kd, fltd, loop)
        assert gt is None and abs(f - f_true) < 0.15, (ratio, f)
    f, gt = T._invert_dp(1.468, kp, kd, fltd, loop)                    # 實機 roll:≥ 0.8 × 上限 → 病態
    assert f is None and gt is not None and gt > 10
    tbl = {"CTRL": {"t": np.arange(0, 60, 0.1), "RMSRollP": np.full(600, 0.02), "RMSRollD": np.full(600, 0.0294),
                    "RMSPitchP": np.full(600, 0.02), "RMSPitchD": np.full(600, 0.01), "RMSYaw": np.full(600, 0.01)}}
    idx = LogIndex.from_tables(tbl, vehicle="copter")
    cs = T._ctrl_stats(idx, "roll", [(0, 60)], {"ATC_RAT_RLL_P": kp, "ATC_RAT_RLL_D": kd, "ATC_RAT_RLL_FLTD": fltd},
                       T.gain_names("ATC_", "roll"))
    assert cs.f_est is None and cs.f_est_gt is not None
    ctx = mk_ctx(ctrl=cs)
    assert "等效頻率 >" in T._ctrl_text(ctx)
    print(f"  D/P 1.045 → {T._invert_dp(1.045, kp, kd, fltd, loop)[0]:.2f} Hz、1.2 → "
          f"{T._invert_dp(1.2, kp, kd, fltd, loop)[0]:.2f} Hz;1.468 → > {gt:.1f} Hz(只給下限)")


def _wobble_detect(f0: float, fs: float, seed: int) -> bool:
    rng = np.random.default_rng(seed)
    n = int(200 * fs)
    t = np.arange(n) / fs
    des = lpf1(lpf1(rng.normal(0, 1, n), fs, 0.5), fs, 0.5)            # 0.5 Hz 二階低通的慢速搖桿
    des *= 5 / des.std()
    act = des + 0.6 * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6)) + rng.normal(0, 0.05, n)
    ls = T.level_stats([(60 + t, [des, act])], fs, "angle", "roll")
    return any(abs(p.f - f0) < 0.3 and p.rms >= T.OSC_ANG_WARN_RMS for p in ls.peaks)


def test_angle_window_high_rate():
    """SP-6:400 Hz 的角度層級窗長也要約 5 s(以前被 1024 點卡成 2.56 s,0.6–0.8 Hz 的擺動 0/6 偵測)。"""
    assert T.choose_nperseg(400.0, 10 ** 6, T.WELCH_WIN_S_ANG, T.WELCH_MAX_N_ANG) == 2048
    assert T.choose_nperseg(400.0, 10 ** 6) == 512                     # 角速度層級維持已校正的 1.28 s
    for f0 in (0.6, 0.8):
        hits = sum(_wobble_detect(f0, 400.0, s) for s in range(3))
        assert hits == 3, (f0, hits)
    print("  400 Hz 角度窗 2048 點(5.12 s);0.6 / 0.8 Hz 擺動 3/3 偵測")


def test_imu_fast_motor_noise():
    """F4:ATT / RATE 10 Hz、IMU 400 Hz(IMU_FAST)、陀螺儀只有 123 Hz 馬達噪音 → gyro_noise,不降 P / D。"""
    ld = make_log(fs=400.0, dur=70.0, att_fs=10.0, imu_fs=400.0, osc={"roll": (123.4, 5.0)}, ctrl=True,
                  ctrl_pd=(0.02, 0.02))
    rep = T.analyze(ld)
    g = finding(rep, "gyro_noise", "roll")
    assert g is not None and abs(g.metrics["noise_hz"] - 123.4) < 1.0, codes(rep, "roll")
    assert not any(c in ("rate_osc_high", "rate_osc_mid") for c in codes(rep, "roll"))
    assert not any(s.direction == DIR_DOWN and s.param in ("ATC_RAT_RLL_P", "ATC_RAT_RLL_D")
                   for f in rep.findings for s in f.suggestions)
    assert sug(g, "INS_LOG_BAT_MASK").proposed == 1.0
    assert rep.axes["roll"].metrics.get("osc_hz") is None and rep.axes["roll"].metrics["noise_hz"] > 120
    print(f"  IMU 400 Hz 的 123 Hz 峰 → {g.title}")


def test_yaw_out_of_range_end_to_end():
    """F1 端到端:Yaw 8 Hz 震盪 + Yaw P 已低於文件範圍 + 振動過大 → 匯出的 .param 不能提高 Yaw P。"""
    ld = make_log(fs=400.0, dur=70.0, osc={"yaw": (8.0, 6.0)}, vibe=70.0, clip_inc=500,
                  params={"ATC_RAT_YAW_P": 0.08, "ATC_RAT_YAW_I": 0.008})
    rep = T.analyze(ld)
    f = finding(rep, "rate_osc_mid", "yaw")
    s = sug(f, "ATC_RAT_YAW_P")
    assert s.proposed is None and "文件範圍" in s.reason
    assert "ATC_RAT_YAW_P" not in rep.param_file() and no_gain_increase(rep)
    assert not any(x.proposed is not None and x.current is not None and x.proposed > x.current
                   and T._is_gain(x.param) for x in rep.suggestions())
    print("  Yaw P 0.08(< 0.1)震盪 → 不給值、.param 不含 ATC_RAT_YAW_P")


def test_default_gains_and_pilot_modes():
    """F10:預設增益的說明方向(小 / 動力強的機常震盪);F12:FLOWHOLD / DRIFT 不算搖桿直接給姿態。"""
    f = T._default_gain_finding(DEFAULT_PARAMS, "ATC_")
    assert "動力強" in f.detail and "常會震盪" in f.detail and "大機(大槳)常偏高" not in f.detail
    assert "FLOWHOLD" not in T.PILOT_MODES and "DRIFT" not in T.PILOT_MODES and "STABILIZE" in T.PILOT_MODES
    ld = make_log(fs=50.0, dur=60.0, att_fs=50.0, mode="FLOWHOLD")
    iv, _ = T.flight_intervals(ld, 0.0, 60.0)
    assert T._pilot_share(ld, iv) == 0.0
    print("  預設增益說明已更正;FLOWHOLD / DRIFT → 位置回授模式")


def test_motor_saturation_split():
    """F2:馬達飽和分上 / 下限;只壓到下限 + 雜訊證據 → 先處理雜訊,不說推力不足、不降 ACCEL。"""
    base = {"metrics": {"motor_sat_pct": 39.0}, "t_focus": 100.0}
    hi = T._motor_sat_findings(dict(base, share=0.13, hi=0.13, lo=0.0), DEFAULT_PARAMS, "ATC_", 0.4, [])
    assert [f.code for f in hi] == ["motor_saturation"] and sug(hi[0], T.THRUST_CHECK) is not None
    assert sug(hi[0], "ATC_ACCEL_R_MAX").proposed == 88000.0
    lo_n = T._motor_sat_findings(dict(base, share=0.39, hi=0.0, lo=0.39), DEFAULT_PARAMS, "ATC_", 0.35,
                                 ["Roll D 項 90 % 的功率在 ≥ 15 Hz"])
    f = lo_n[0]
    assert f.code == "motor_sat_low" and f.severity == SEV_BAD and "不是推力不足" in f.detail
    assert sug(f, T.THRUST_CHECK) is None and not any("ACCEL" in s.param for s in f.suggestions)
    assert f.suggestions[0].priority == 1 and "陷波" in f.suggestions[0].param
    lo_q = T._motor_sat_findings(dict(base, share=0.05, hi=0.0, lo=0.05), DEFAULT_PARAMS, "ATC_", 0.35, [])
    assert lo_q[0].code == "motor_sat_low" and lo_q[0].severity == SEV_INFO and not lo_q[0].suggestions
    # 懸停油門高 → 就算碰下限較多,也當推力問題
    hh = T._motor_sat_findings(dict(base, share=0.05, hi=0.01, lo=0.04), DEFAULT_PARAMS, "ATC_", 0.66, [])
    assert hh[0].code == "motor_saturation"
    # 軸層級:壓到下限為主時,PIDx limit 旗標只和「頂到上限」比例一起採用、不說推重比不足
    c = mk_ctx(rate={}, pid=T.PidStats(fs=400, p_rms=0.02, d_rms=0.01, limit_share=0.39))
    c.veh.motor_sat, c.veh.motor_hi, c.veh.motor_lo, c.veh.hover = 0.39, 0.0, 0.39, 0.35
    assert not any(x.code == "saturation" for x in T.axis_rules(c))
    c2 = mk_ctx(rate={"sat_share": 0.05}, pid=T.PidStats(fs=400, p_rms=0.02, d_rms=0.01, d_hf_frac=0.9))
    c2.veh.motor_sat, c2.veh.motor_hi, c2.veh.motor_lo, c2.veh.hover = 0.3, 0.0, 0.3, 0.35
    s2 = next(x for x in T.axis_rules(c2) if x.code == "saturation")
    assert "推重比" not in " ".join(s.reason + s.param for s in s2.suggestions)
    assert not any("ACCEL" in s.param for s in s2.suggestions)
    print("  頂到上限 → 推重比 + 降 ACCEL;壓到下限 + 雜訊 → 先處理雜訊;壓到下限、沒雜訊 → info")


def test_motor_pwm_range_fallback():
    """SP-4:MOT_PWM_MIN / MAX = 0(ArduCopter 4.0 / 4.1 預設)→ 用 RC3_MIN / MAX;都沒有 → 不猜 1000 / 2000。"""
    assert T._pwm_range({"MOT_PWM_MIN": 1100.0, "MOT_PWM_MAX": 1900.0}, "MOT_")[:2] == (1100.0, 1900.0)
    rg = T._pwm_range({"MOT_PWM_MIN": 0.0, "MOT_PWM_MAX": 0.0, "RC3_MIN": 1100.0, "RC3_MAX": 1900.0}, "MOT_")
    assert rg[:2] == (1100.0, 1900.0) and "RC3" in rg[2]
    rg2 = T._pwm_range({"MOT_PWM_MIN": 0.0, "MOT_PWM_MAX": 0.0, "RCMAP_THROTTLE": 4.0,
                        "RC4_MIN": 1000.0, "RC4_MAX": 2000.0}, "MOT_")
    assert rg2[:2] == (1000.0, 2000.0)
    assert T._pwm_range({"MOT_PWM_MIN": 0.0, "MOT_PWM_MAX": 0.0}, "MOT_") is None
    # 範圍不明 + PID limit 旗標常設 → 仍擋提高增益
    veh = T.VehicleContext(params={}, motor_sat=None, pid_limit=0.1)
    assert veh.block_increase is not None
    print("  MOT_PWM = 0 → RC3 範圍;都沒有 → 不判斷、改看 PID limit 旗標")


# =====================================================================
# 規則(手工組的統計)
# =====================================================================

def mk_ctx(axis="roll", fs=400.0, params=None, vibe="ok", motor_sat=0.0, rate=None, ang=None, **kw):
    veh = T.VehicleContext(prefix="ATC_", params=dict(DEFAULT_PARAMS, **(params or {})), vibe=vibe,
                           motor_sat=motor_sat, pilot_share=1.0)
    r = None
    if rate is not None:
        r = T.LevelStats("rate", fs, dur_s=60, rms_err=2.0, max_err=10, rms_des=30, rms_act=30, lag_ms=20,
                         lag_rho=0.9, sat_share=0.0)
        for k, v in rate.items():
            setattr(r, k, v)
    a = None
    if ang is not None:
        a = T.LevelStats("angle", fs, dur_s=60, rms_err=0.3, max_err=2, rms_des=5, rms_act=5, lag_ms=60,
                         lag_rho=0.9)
        for k, v in ang.items():
            setattr(a, k, v)
    return T.AxisContext(axis, veh, r, a, **kw)


def sug(f, param):
    return next((s for s in f.suggestions if s.param == param), None)


def test_rules_oscillation():
    pk = T.OscPeak(25.0, 10.0, 8.0, 0.9, 5.0, (4.0, 6.0), "rate", 400.0)
    fl = T.axis_rules(mk_ctx(rate={"peaks": [pk]}))
    f = fl[0]
    assert f.code == "rate_osc_high" and f.severity == SEV_BAD, (f.code, f.severity)
    d, p = sug(f, "ATC_RAT_RLL_D"), sug(f, "ATC_RAT_RLL_P")
    assert d.direction == DIR_DOWN and d.priority == 1 and abs(d.change_pct + 25) < 1 and d.proposed == 0.0027
    assert p.direction == DIR_DOWN and p.priority == 2 and abs(p.change_pct + 15) < 1
    assert any(s.direction == DIR_CHECK for s in f.suggestions) and f.t_focus == 5.0
    # 中頻 12 Hz:降 P 20 %,I 跟著 P;D 項主導時 D 也小降
    pk2 = T.OscPeak(12.0, 5.0, 6.0, 0.8, 7.0, None, "rate", 400.0)
    f2 = T.axis_rules(mk_ctx(axis="pitch", rate={"peaks": [pk2]}))[0]
    assert f2.code == "rate_osc_mid" and f2.severity == SEV_WARN
    p2, i2 = sug(f2, "ATC_RAT_PIT_P"), sug(f2, "ATC_RAT_PIT_I")
    assert p2.priority == 1 and p2.proposed == 0.108 and i2.proposed == 0.108
    assert sug(f2, "ATC_RAT_PIT_D") is None
    # F6:同一個 12 Hz 震盪、CTRL D/P 1.5 —— 預設增益在 12 Hz 的 D/P 本來就約 1.7(只由增益比與頻率決定),
    # 不代表 D 太高:仍先降 P(I 跟著),D 只在順位 3 一起小降 10 %,文字不說「由 D 驅動」
    exp12 = float(T.dp_ratio_model(0.135, 0.0036, 20.0, 400.0, 12.0))
    assert 1.5 < exp12 < 2.0, exp12
    ctx = mk_ctx(axis="pitch", rate={"peaks": [pk2]},
                 ctrl=T.CtrlStats(p_rms=0.02, d_rms=0.03, ratio=1.5, f_est=13.0))
    f2b = T.axis_rules(ctx)[0]
    assert sug(f2b, "ATC_RAT_PIT_P").priority == 1 and sug(f2b, "ATC_RAT_PIT_I").priority == 2
    d2b = sug(f2b, "ATC_RAT_PIT_D")
    assert d2b.priority == 3 and abs(d2b.change_pct + 10) < 1 and "本來就較大" in f2b.detail
    assert "由 D 驅動" not in f2b.detail
    # D 相對 P 偏高(增益比 0.0144 / 0.135 = 0.107 > 0.05)→ 先降 D 25 %,再降 P 15 %
    f2d = T.axis_rules(mk_ctx(axis="pitch", params={"ATC_RAT_PIT_D": 0.0144}, rate={"peaks": [pk2]}))[0]
    assert sug(f2d, "ATC_RAT_PIT_D").priority == 1 and sug(f2d, "ATC_RAT_PIT_P").priority == 2
    assert "增益比" in f2d.detail
    # 預設增益、但 PIDx 量到的 D/P 遠超過該頻率應有的(≥ max(3, 1.5 × 預期))→ D 吃到雜訊,先降 D
    f2e = T.axis_rules(mk_ctx(axis="pitch", rate={"peaks": [pk2]},
                              pid=T.PidStats(fs=400, p_rms=0.01, d_rms=0.035)))[0]
    assert f2e.code == "rate_osc_mid" and sug(f2e, "ATC_RAT_PIT_D").priority == 1, f2e.detail
    # PIDx D 項輸出 ≥ 3 × P 項(沒有明顯峰)→ d_dominant:降 D
    f2c = T.axis_rules(mk_ctx(rate={}, pid=T.PidStats(fs=400, p_rms=0.01, d_rms=0.06)))[0]
    assert f2c.code == "d_dominant" and sug(f2c, "ATC_RAT_RLL_D").direction == DIR_DOWN
    # 偶發(只出現 20 % 時間)→ 只是 info;沒有增益提高
    pk3 = T.OscPeak(12.0, 5.0, 6.0, 0.2, 7.0, None, "rate", 400.0)
    f3 = T.axis_rules(mk_ctx(rate={"peaks": [pk3]}))[0]
    assert f3.severity == SEV_INFO
    # 偏航震盪:降 Yaw P 20 %(Yaw 沒有 D)
    f4 = T.axis_rules(mk_ctx(axis="yaw", rate={"peaks": [T.OscPeak(9.0, 6.0, 5.0, 0.9, None, None, "rate", 400.0)]}))[0]
    assert f4.code == "rate_osc_mid" and sug(f4, "ATC_RAT_YAW_P").change_pct == -20.0
    assert sug(f4, "ATC_RAT_YAW_I").proposed == round(0.144 / 10, 4)
    # F4 / SP-5:高於控制頻寬(INS_GYRO_FILTER 20 → 約 50 Hz)的峰是馬達噪音:gyro_noise、不降 P、不算震盪
    for f_hz in (60.0, 90.0, 120.0):
        npk = T.OscPeak(f_hz, 4.5, 8.0, 0.9, 5.0, None, "rate", 400.0)
        fl_n = T.axis_rules(mk_ctx(rate={"peaks": [npk]}))
        assert [x.code for x in fl_n] == ["gyro_noise"], [x.code for x in fl_n]
        assert fl_n[0].severity == SEV_WARN and fl_n[0].metrics["noise_hz"] == f_hz
        assert not any(s.direction == DIR_DOWN and T._is_gain(s.param) for s in fl_n[0].suggestions)
        assert sug(fl_n[0], "INS_LOG_BAT_MASK").proposed == 1.0 and "osc_hz" not in fl_n[0].metrics
    # 小型機 INS_GYRO_FILTER 80 Hz:上限 = min(2.5 × 80, 0.2 × 400) = 80 Hz → 60 Hz 仍可能是 D 震盪
    assert T.ctrl_fmax({"INS_GYRO_FILTER": 80.0}) == 80.0 and T.ctrl_fmax({"INS_GYRO_FILTER": 20.0}) == 50.0
    f60 = T.axis_rules(mk_ctx(params={"INS_GYRO_FILTER": 80.0},
                              rate={"peaks": [T.OscPeak(60.0, 4.5, 8.0, 0.9, 5.0, None, "rate", 400.0)]}))[0]
    assert f60.code == "rate_osc_high"
    print("  高頻 → 先降 D 25 % 再降 P 15 %;中頻 → 降 P 20 %、I 跟 P(D 增益比偏高 / 量到的 D/P 遠超預期才先降 D);"
          "D/P ≥ 3 → 降 D;偶發 → info;Yaw → P −20 %、I = P/10;> 控制頻寬 → gyro_noise(陷波)")


def test_rules_low_rate_suspect():
    imu = [T.OscPeak(9.1, 4.5, 5.0, 0.9, 3.0, None, "imu", 25.0)]
    rate10 = {"peaks": [], "rms_err": 8.0, "rms_des": 4.0}
    # CTRL D/P 1.47(推估 14.5 Hz)落在「馬達噪音進 D 項也會造成」的區間(noisy_notch 模擬 1.5–1.6、實機 1.47)
    # → 無法確定是震盪:只提醒(info)、不給增益建議,文字附「若確實看到抖動」的條件方向
    ctx = mk_ctx(fs=10.0, rate=rate10, imu_peaks=imu, imu_fs=25.0,
                 ctrl=T.CtrlStats(p_rms=0.018, d_rms=0.026, ratio=1.47, f_est=14.5))
    fl = T.axis_rules(ctx)
    f = fl[0]
    assert f.code == "rate_osc_suspect" and f.severity == SEV_INFO, [(x.code, x.severity) for x in fl]
    assert not f.suggestions and f.metrics["support"] is None
    assert "混疊" in f.detail and "噪音" in f.detail and "ATC_RAT_RLL_D 約 20 %" in f.detail, f.detail
    assert not any(x.code == "rate_sluggish" for x in fl)     # 疑似震盪時不說「偏軟」
    assert all(s.direction != DIR_UP for x in fl for s in x.suggestions)
    # CTRL P 項主導(D/P 0.8)且推估 15 Hz ≈ 混疊候選 25 − 9.1 = 15.9 Hz → 互相印證:warn、小降 P(順位 2)、I 跟 P
    ctx_p = mk_ctx(fs=10.0, params={"ATC_RAT_RLL_P": 0.3, "ATC_RAT_RLL_I": 0.3}, rate=rate10, imu_peaks=imu,
                   imu_fs=25.0, ctrl=T.CtrlStats(p_rms=0.05, d_rms=0.04, ratio=0.8, f_est=15.0))
    fp = T.axis_rules(ctx_p)[0]
    assert fp.severity == SEV_WARN and fp.metrics["support"] == "p" and "15.9" in fp.detail
    assert sug(fp, "ATC_RAT_RLL_P").priority == 2 and sug(fp, "ATC_RAT_RLL_P").proposed == 0.255
    assert sug(fp, "ATC_RAT_RLL_I").proposed == 0.255 and sug(fp, "ATC_RAT_RLL_D") is None
    # CTRL D/P ≥ 3 → D 項主導:warn、先小降 D(順位 2)再降 P(順位 3)
    ctx_d = mk_ctx(fs=10.0, rate=rate10, imu_peaks=imu, imu_fs=25.0,
                   ctrl=T.CtrlStats(p_rms=0.014, d_rms=0.06, ratio=4.3, f_est=None))
    fd = T.axis_rules(ctx_d)[0]
    assert fd.severity == SEV_WARN and fd.metrics["support"] == "d"
    assert sug(fd, "ATC_RAT_RLL_D").priority == 2 and sug(fd, "ATC_RAT_RLL_P").priority == 3
    # 偏航的低速 IMU 峰(偏航環在 5–20 Hz 自激很少見)→ 只提醒
    fy = T.axis_rules(mk_ctx(axis="yaw", fs=10.0, rate=rate10, imu_peaks=imu, imu_fs=25.0))[0]
    assert fy.code == "rate_osc_suspect" and fy.severity == SEV_INFO and not fy.suggestions
    # 低速 + 追蹤落差大,但沒有 CTRL 排除看不到的高頻震盪 → info、不建議提高增益
    lag10 = {"rms_err": 16.0, "rms_des": 19.0, "lag_ms": 113.0, "lag_rho": 0.93}
    fl2 = T.axis_rules(mk_ctx(fs=10.0, rate=lag10))
    f2 = next(x for x in fl2 if x.code == "rate_sluggish")
    assert f2.severity == SEV_INFO and not f2.suggestions and "CTRL" in f2.detail
    # 同樣落差 + CTRL 顯示誤差能量在低頻(推估 2.3 Hz、P 項主導)→ warn、保守地 P +10 %(順位 2)
    lowf = T.CtrlStats(p_rms=0.0063, d_rms=0.0025, ratio=0.39, f_est=2.3)
    f3 = T.axis_rules(mk_ctx(fs=10.0, rate=lag10, ctrl=lowf))[0]
    p3 = sug(f3, "ATC_RAT_RLL_P")
    assert f3.code == "rate_sluggish" and f3.severity == SEV_WARN and p3.direction == DIR_UP
    assert p3.priority == 2 and abs(p3.change_pct - T.STEP_P_UP_LOW) < 1 and "10 Hz" in f3.title
    # …但期望姿態來自 LOITER(位置回授)或相關係數不夠 → 只提醒
    c4 = mk_ctx(fs=10.0, rate=lag10, ctrl=lowf)
    c4.veh.pilot_share = 0.0
    f4 = T.axis_rules(c4)[0]
    assert f4.severity == SEV_INFO and not f4.suggestions and "LOITER" in f4.detail
    f5 = T.axis_rules(mk_ctx(fs=10.0, rate=dict(lag10, lag_rho=0.6), ctrl=lowf))[0]
    assert f5.severity == SEV_INFO and not f5.suggestions
    # 10 Hz 的延遲估計偏高:roll/pitch 低於 90 ms 不算落差(高速時的門檻是 75 ms)
    assert not T.axis_rules(mk_ctx(fs=10.0, rate=dict(lag10, lag_ms=80.0), ctrl=lowf))
    # 偏航(10 Hz 已看得到偏航環動態)延遲 380 ms → warn、Yaw P +10 %
    f6 = T.axis_rules(mk_ctx(axis="yaw", fs=10.0, rate=dict(lag10, lag_ms=380.0)))[0]
    assert f6.severity == SEV_WARN and sug(f6, "ATC_RAT_YAW_P").direction == DIR_UP
    print("  25 Hz IMU 峰:CTRL 不明確 → info + 條件方向;P 主導且頻率吻合 → 降 P;D/P ≥ 3 → 降 D;"
          "10 Hz 偏軟:有 CTRL 低頻佐證、搖桿模式才 P +10 %,否則只提醒")


def test_rules_damping_and_sluggish():
    # 只有 |H| 峰:弱證據 → info、P −10 %(順位 3)、不加 D
    f = T.axis_rules(mk_ctx(rate={"h_peak": 1.6, "h_peak_hz": 9.0}))[0]
    assert f.code == "rate_underdamped" and f.severity == SEV_INFO
    assert abs(sug(f, "ATC_RAT_RLL_P").change_pct + 10) < 1 and sug(f, "ATC_RAT_RLL_P").priority == 3
    assert sug(f, "ATC_RAT_RLL_D") is None
    # 3 Hz 以下的 |H| 峰(角度環回授)不採用
    assert not T.axis_rules(mk_ctx(rate={"h_peak": 1.6, "h_peak_hz": 2.3}))
    # 可信的步階過衝 45 %(> 35 %)→ warn:先降 P 10 %,或加 D 15 %
    strong = {"h_peak": 1.6, "h_peak_hz": 9.0, "overshoot": 45.0, "step_n": 50, "input_bw": 20.0}
    f = T.axis_rules(mk_ctx(rate=strong))[0]
    assert f.severity == SEV_WARN and sug(f, "ATC_RAT_RLL_P").priority == 1
    # 過衝 28 %:在反卷積估計的散布內(預設增益實測 4–27 %),沒有 |H| 峰 → 不判阻尼不足
    assert not T.axis_rules(mk_ctx(rate={"h_peak": 1.1, "h_peak_hz": 6.0, "overshoot": 28.0, "step_n": 50,
                                         "input_bw": 20.0}))
    assert sug(f, "ATC_RAT_RLL_D").direction == DIR_UP and sug(f, "ATC_RAT_RLL_D").priority == 2
    # D 已偏高(D/P 增益比 0.074 > 0.05)→ 共振多半是 D 造成:先小降 D 15 %,再降 P(不再建議加 D)
    f_ = T.axis_rules(mk_ctx(params={"ATC_RAT_RLL_D": 0.01}, rate=strong))[0]
    d_ = sug(f_, "ATC_RAT_RLL_D")
    assert d_.direction == DIR_DOWN and d_.priority == 1 and abs(d_.change_pct + 15) < 1
    assert sug(f_, "ATC_RAT_RLL_P").priority == 2
    # 追蹤比大但沒有延遲 / 頻寬 / 上升時間證據 → 不判偏軟
    assert not T.axis_rules(mk_ctx(rate={"rms_err": 21.0, "rms_des": 30.0}))
    # 偏軟:延遲 90 ms、追蹤比 0.7、沒有過衝
    ctx = mk_ctx(rate={"lag_ms": 90.0, "rms_err": 21.0, "rms_des": 30.0, "h_peak": 1.0})
    f2 = T.axis_rules(ctx)[0]
    assert f2.code == "rate_sluggish" and f2.severity == SEV_WARN
    p, i = sug(f2, "ATC_RAT_RLL_P"), sug(f2, "ATC_RAT_RLL_I")
    assert p.direction == DIR_UP and p.proposed == 0.162 and i.proposed == 0.162
    # 振動過大 / 馬達飽和 → 安全後處理拿掉所有「提高增益」
    for veh_kw in ({"vibe": "bad"}, {"motor_sat": 0.05}):
        c = mk_ctx(rate={"lag_ms": 90.0, "rms_err": 21.0, "rms_des": 30.0}, **veh_kw)
        fl = T.axis_rules(c)
        T.apply_safety(fl, c.veh)
        assert all(s.direction != DIR_UP for x in fl for s in x.suggestions), veh_kw
        assert "暫不建議提高" in fl[0].detail
    # 振動偏高:不加 D,但 P 可以
    c = mk_ctx(rate=strong, vibe="warn")
    fl = T.axis_rules(c)
    assert sug(fl[0], "ATC_RAT_RLL_D") is None and sug(fl[0], "ATC_RAT_RLL_P") is not None
    # 輸出飽和:不說偏軟,改成 saturation + 降 ACCEL
    c = mk_ctx(rate={"lag_ms": 90.0, "rms_err": 21.0, "rms_des": 30.0, "sat_share": 0.05})
    fl = T.axis_rules(c)
    assert "rate_sluggish" not in [x.code for x in fl]
    fs_ = next(x for x in fl if x.code == "saturation")
    assert sug(fs_, "ATC_ACCEL_R_MAX").proposed == 88000.0
    # 偏航偏軟:Yaw P +20 %,I = P/10
    f3 = T.axis_rules(mk_ctx(axis="yaw", rate={"lag_ms": 400.0, "rms_err": 20.0, "rms_des": 25.0}))[0]
    assert not T.axis_rules(mk_ctx(axis="yaw", rate={"lag_ms": 200.0}))   # 偏航 200 ms 屬正常
    assert sug(f3, "ATC_RAT_YAW_P").proposed == 0.216 and sug(f3, "ATC_RAT_YAW_I").proposed == 0.0216
    print("  |H| 峰 → info;步階過衝 → P −10 %(雜訊小才加 D);偏軟 → P/I +20 %;振動 / 飽和時不加增益;Yaw I = P/10")


def test_rules_angle_bias_trim():
    wob = T.OscPeak(1.5, 0.8, 5.0, 0.9, 12.0, None, "angle", 50.0)
    f = T.axis_rules(mk_ctx(rate={}, ang={"peaks": [wob]}))[0]
    assert f.code == "angle_wobble" and f.severity == SEV_WARN
    # F7:ATT.Des* 是 INPUT_TC 整形後的目標 → INPUT_TC 改不了量到的響應,不列為修正項(只在說明裡交代)
    assert sug(f, "ATC_ANG_RLL_P").proposed == 3.82 and sug(f, "ATC_INPUT_TC") is None
    assert "ATC_INPUT_TC 只改變搖桿" in f.detail
    # 角度落後、角速度正常 → 提高 ANG_P(不再附「或減小 INPUT_TC」)
    f2 = T.axis_rules(mk_ctx(rate={}, ang={"lag_ms": 250.0}))[0]
    assert f2.code == "angle_lag" and sug(f2, "ATC_ANG_RLL_P").direction == DIR_UP
    assert sug(f2, "ATC_INPUT_TC") is None and "或減小" not in f2.detail
    # …振動過大時 ANG_P 增加被安全規則拿掉 → 只剩提醒,降為 info(不會留下讓反應更積極的 TC 減小)
    c_v = mk_ctx(vibe="bad", rate={"h_peak": 1.05, "h_peak_hz": 6.0}, ang={"lag_ms": 260.0})
    fl_v = T.axis_rules(c_v)
    T.apply_safety(fl_v, c_v.veh)
    fv = next(x for x in fl_v if x.code == "angle_lag")
    assert fv.severity == SEV_INFO and not fv.suggestions and "暫不建議提高 ATC_ANG_RLL_P" in fv.detail
    # 角速度偏差 → I +25 % + 機構檢查
    f3 = T.axis_rules(mk_ctx(rate={"bias": 4.0}))[0]
    assert f3.code == "bias" and abs(sug(f3, "ATC_RAT_RLL_I").change_pct - 25) < 1
    assert any(s.direction == DIR_CHECK for s in f3.suggestions)
    # 角度 |H| 共振峰(Roll、搖桿模式)→ warn、ANG_P −15 %;偏航或 LOITER(位置回授)的 |H| 不採用
    fh = T.axis_rules(mk_ctx(rate={}, ang={"h_peak": 1.8, "h_peak_hz": 4.5}))[0]
    assert fh.code == "angle_freq_peak" and fh.severity == SEV_WARN and sug(fh, "ATC_ANG_RLL_P").priority == 1
    assert not T.axis_rules(mk_ctx(axis="yaw", rate={}, ang={"h_peak": 3.5, "h_peak_hz": 1.1}))
    c_l = mk_ctx(rate={}, ang={"h_peak": 2.3, "h_peak_hz": 1.1})
    c_l.veh.pilot_share = 0.0
    assert not T.axis_rules(c_l)
    # 馬達飽和時,角度擺動 / 落後先歸因於沒有控制裕度,不怪角度環
    assert not any(x.code.startswith("angle") for x in
                   T.axis_rules(mk_ctx(motor_sat=0.1, rate={}, ang={"peaks": [wob], "lag_ms": 250.0})))
    # 低速:角度誤差峰正好是疑似震盪(IMU 7.6 Hz)在 10 Hz 的混疊(2.4 Hz)→ 不怪角度環
    imu = [T.OscPeak(7.6, 11.9, 9.0, 0.9, 3.0, None, "imu", 25.0)]
    al = T.OscPeak(2.4, 0.5, 5.0, 0.9, 3.0, None, "angle", 10.0)
    fl_al = T.axis_rules(mk_ctx(fs=10.0, rate={"rms_err": 14.0, "rms_des": 18.0}, ang={"peaks": [al]},
                                imu_peaks=imu, imu_fs=25.0,
                                ctrl=T.CtrlStats(p_rms=0.06, d_rms=0.024, ratio=0.4, f_est=7.6)))
    assert [x.code for x in fl_al] == ["rate_osc_suspect"], [x.code for x in fl_al]
    # PIDx 積分常頂到 IMAX → 機構檢查,不再單純加 I
    pid = T.PidStats(fs=400, p_rms=0.02, i_rms=0.45, d_rms=0.01, i_sat_share=0.3, imax=0.5)
    f4 = T.axis_rules(mk_ctx(rate={"bias": 4.0}, pid=pid))[0]
    assert f4.code == "i_saturated" and all(s.direction == DIR_CHECK for s in f4.suggestions)
    # 懸停輸出偏一邊
    f5 = T.axis_rules(mk_ctx(rate={}, out_mean=0.08))[0]
    assert f5.code == "trim_offset" and f5.severity == SEV_INFO and "偏左" in f5.detail
    # slew 限制器
    f6 = T.axis_rules(mk_ctx(rate={}, pid=T.PidStats(fs=400, p_rms=0.02, d_rms=0.01, dmod_share=0.2)))[0]
    assert f6.code == "slew_limit" and sug(f6, "ATC_RAT_RLL_D").priority == 1
    print("  角度擺動 → ANG_P −15 %;角度落後 → ANG_P +15 %;偏差 → I +25 %;IMAX / 配平 / slew 限制正確")


def test_gain_suggestion_clamp_and_prefix():
    s = T.gain_suggestion("ATC_RAT_RLL_P", {"ATC_RAT_RLL_P": 0.45}, 20, "x")
    assert s.proposed == 0.5 and abs(s.change_pct - 11.1) < 0.1
    s2 = T.gain_suggestion("ATC_RAT_RLL_P", {"ATC_RAT_RLL_P": 0.5}, 20, "x")
    assert s2.proposed is None and "邊界" in s2.reason
    s3 = T.gain_suggestion("ATC_RAT_RLL_D", {}, -25, "x")
    assert s3.current is None and s3.proposed is None and s3.change_pct == -25
    s4 = T.gain_suggestion("Q_A_ANG_PIT_P", {"Q_A_ANG_PIT_P": 3.2}, -15, "x")
    assert s4.proposed == 3.0 and T.param_range("Q_A_RAT_YAW_P") == T.PARAM_RANGES["ATC_RAT_YAW_P"]
    s5 = T.gain_suggestion("ATC_RAT_PIT_D", {"ATC_RAT_PIT_D": 0.0036}, -25, "x")
    assert s5.proposed == 0.0027 and s5.change_pct == -25.0
    print("  夾到範圍、邊界不再調、缺目前值、Q_A_ 對應 ATC_ 範圍、3 位有效數字")


def test_gain_suggestion_out_of_range():
    """F1 / F8 / F9:目前值在文件範圍外時絕不反向、不放大步幅;ACCEL = 0(不限制)不說「調低」。"""
    # F1:往範圍外調 → 不給值(以前會夾到邊界而變成反方向的大改)
    for prm, pct in (({"ATC_RAT_YAW_P": 0.08}, -25), ({"ATC_ANG_RLL_P": 18.0}, 15), ({"ATC_INPUT_TC": 0.7}, 20)):
        (p, cur), = prm.items()
        s = T.gain_suggestion(p, prm, pct, "x")
        assert s.proposed is None and s.change_pct is None and s.current == cur, (p, s)
        assert s.direction == (DIR_UP if pct > 0 else DIR_DOWN) and "文件範圍" in s.reason and "AutoTune" in s.reason
    # F8:往範圍內調 → 照原本的步幅(不直接跳到邊界),註明「請確認是否為 AutoTune 結果」
    s = T.gain_suggestion("ATC_RAT_YAW_P", {"ATC_RAT_YAW_P": 0.054}, T.STEP_P_UP_LOW, "x")
    assert s.proposed == 0.0594 and abs(s.change_pct - 10) < 0.1 and "AutoTune" in s.reason
    s = T.gain_suggestion("ATC_RAT_RLL_I", {"ATC_RAT_RLL_I": 0.002}, T.STEP_I_UP, "x")
    assert s.proposed == 0.0025 and abs(s.change_pct - 25) < 0.1
    s = T.gain_suggestion("ATC_ANG_RLL_P", {"ATC_ANG_RLL_P": 18.0}, T.STEP_ANG_DOWN, "x")
    assert s.proposed == 15.3 and s.direction == DIR_DOWN
    # 步幅上限 25 %
    assert T.gain_suggestion("ATC_RAT_RLL_P", {"ATC_RAT_RLL_P": 0.2}, -40, "x").change_pct == -25.0
    # I 跟 P:同樣受 25 % 與範圍規則約束(以前 Yaw I 0.0054 → 0.01 是 +85 %)
    ctx = mk_ctx(axis="yaw", params={"ATC_RAT_YAW_P": 0.054, "ATC_RAT_YAW_I": 0.0054})
    ps = T.gain_suggestion("ATC_RAT_YAW_P", ctx.veh.params, T.STEP_P_UP_LOW, "x")
    fi = T._follow_i(ctx, ps, 3)
    assert fi.proposed == 0.00594 and fi.direction == DIR_UP and abs(fi.change_pct - 10) < 0.1
    ctx2 = mk_ctx(params={"ATC_RAT_RLL_P": 0.5, "ATC_RAT_RLL_I": 0.5})
    fi2 = T._follow_i(ctx2, T.gain_suggestion("ATC_RAT_RLL_P", ctx2.veh.params, -20, "x"), 2)
    assert fi2.proposed == 0.4 and fi2.direction == DIR_DOWN
    # F9:ATC_ACCEL_*_MAX = 0 代表不限制 → 不給值、不說「從小值開始」;規則改成檢查項
    s = T.gain_suggestion("ATC_ACCEL_R_MAX", {"ATC_ACCEL_R_MAX": 0.0}, T.STEP_ACCEL_DOWN, "x")
    assert s.proposed is None and "從小值開始" not in s.reason and "停用" in s.reason
    assert "從小值開始" in T.gain_suggestion("ATC_RAT_RLL_D", {"ATC_RAT_RLL_D": 0.0}, 20, "x").reason
    fl = T.axis_rules(mk_ctx(params={"ATC_ACCEL_R_MAX": 0.0}, rate={"sat_share": 0.08}))
    sat = next(x for x in fl if x.code == "saturation")
    acc = sug(sat, "ATC_ACCEL_R_MAX")
    assert acc.direction == DIR_CHECK and "不限制" in acc.reason
    print("  範圍外:往外 → 不給值、往內 → 照原步幅;單步 ≤ 25 %;I 跟 P 同規則;ACCEL = 0 → 檢查項")


def test_apply_safety_value_based():
    """F1 第二道防線 + F7:是否「讓控制更積極」看建議值 vs 目前值,INPUT_TC 減小也算。"""
    veh = T.VehicleContext(params={}, vibe="bad")
    rev = Suggestion("ATC_RAT_YAW_P", DIR_DOWN, "x", 0.08, 0.1, 25.0, 1)      # 標成 decrease、其實是提高
    tc = Suggestion("ATC_INPUT_TC", DIR_DOWN, "x", 0.15, 0.12, -20.0, 2)
    acc = Suggestion("ATC_ACCEL_R_MAX", DIR_DOWN, "x", 110000.0, 88000.0, -20.0, 2)
    p_dn = Suggestion("ATC_RAT_YAW_P", DIR_DOWN, "x", 0.18, 0.144, -20.0, 1)
    f = Finding("rate_osc_mid", SEV_WARN, "yaw", "t", "d", [rev, tc, acc, p_dn])
    T.apply_safety([f], veh, {"yaw": True})
    assert [(s.param, s.proposed) for s in f.suggestions] == [("ATC_ACCEL_R_MAX", 88000.0), ("ATC_RAT_YAW_P", 0.144)]
    assert "暫不建議提高 ATC_RAT_YAW_P" in f.detail and "暫不建議減小 ATC_INPUT_TC" in f.detail
    # 沒有整機問題、但該軸有震盪跡象:TC 減小同樣拿掉
    f2 = Finding("angle_lag", SEV_WARN, "roll", "t", "d", [Suggestion("ATC_INPUT_TC", DIR_DOWN, "x", 0.15, 0.12)])
    T.apply_safety([f2], T.VehicleContext(params={}), {"roll": True})
    assert not f2.suggestions and f2.severity == SEV_INFO
    print("  標成 decrease 的提高、INPUT_TC 減小都被擋;降增益 / 降 ACCEL 保留")


# =====================================================================
# 端到端
# =====================================================================

def test_flight_intervals():
    ld = make_log(fs=50.0, dur=60.0, att_fs=50.0)
    iv, _ = T.flight_intervals(ld, 0.0, 60.0)
    assert len(iv) == 1 and abs(iv[0][0] - 5.0) < 1e-6 and abs(iv[0][1] - 54.0) < 1e-6, iv
    ld2 = make_log(fs=50.0, dur=60.0, att_fs=50.0, ev=False)       # 沒有起降事件 → 油門 + 高度
    iv2, notes = T.flight_intervals(ld2, 0.0, 60.0)
    assert len(iv2) == 1 and 4.5 < iv2[0][0] < 5.5 and 53.5 < iv2[0][1] < 54.5 and notes, (iv2, notes)
    iv3, _ = T.flight_intervals(ld, 20.0, 30.0)
    assert iv3 == [(20.0, 30.0)], iv3          # 分析範圍的邊界不裁
    print(f"  事件:{iv[0][0]:.1f}–{iv[0][1]:.1f} s;油門判斷:{iv2[0][0]:.1f}–{iv2[0][1]:.1f} s;子範圍正確")


def test_analyze_fast_oscillation():
    ld = make_log(fs=400.0, dur=70.0, osc={"roll": (22.0, 9.0)}, batch_peak=None)
    t0 = time.perf_counter()
    rep = T.analyze(ld)
    dt = time.perf_counter() - t0
    f = finding(rep, "rate_osc_high", "roll")
    assert f is not None and f.severity == SEV_BAD, codes(rep, "roll")
    assert abs(rep.axes["roll"].metrics["osc_hz"] - 22.0) < 0.5
    assert sug(f, "ATC_RAT_RLL_D").priority == 1
    assert rep.axes["pitch"].verdict in (SEV_OK, SEV_INFO), codes(rep, "pitch")
    assert "low_rate" not in codes(rep) and rep.data["RATE_hz"] == 400.0
    aa = rep.axes["pitch"]
    assert aa.level == "rate" and aa.freq is not None and aa.freq.bandwidth_hz is not None
    assert aa.step is not None and aa.step.overshoot_pct is not None and aa.step.overshoot_pct < 15
    assert np.isnan(aa.des[0]) and np.isfinite(aa.des[aa.des.size // 2])          # 地面為 NaN
    assert rep.flight_s > 55 and dt < 10, (rep.flight_s, dt)
    txt = T.format_report_text(rep)
    assert "Roll 角速度高頻震盪" in txt and "ATC_RAT_RLL_D" in txt
    print(f"  22 Hz 震盪 → {f.title}({f.severity});Pitch {aa.verdict},頻寬 {aa.freq.bandwidth_hz:.1f} Hz、"
          f"過衝 {aa.step.overshoot_pct:.0f} %;分析 {dt:.2f} s")


def test_analyze_underdamped_and_saturation():
    ld = make_log(fs=400.0, dur=70.0, zeta=0.25, fn=7.0, seed=3)
    rep = T.analyze(ld)
    f = finding(rep, "rate_underdamped", "roll")
    assert f is not None and sug(f, "ATC_RAT_RLL_P").direction == DIR_DOWN, codes(rep, "roll")
    # 馬達 5 % 時間頂到上限 + 偏軟系統 → 只會建議檢查 / 降 ACCEL,不會加增益
    ld2 = make_log(fs=400.0, dur=70.0, fn=3.0, zeta=0.9, delay=0.04, sat_frac=0.05, seed=4)
    rep2 = T.analyze(ld2)
    assert finding(rep2, "motor_saturation") is not None, codes(rep2)
    assert no_gain_increase(rep2)
    print(f"  ζ=0.25 → {f.title};馬達飽和 {finding(rep2, 'motor_saturation').metrics['motor_sat_pct']:.1f} % → 不加增益")


def test_analyze_vibration_and_pid():
    ld = make_log(fs=400.0, dur=60.0, vibe=70.0, clip_inc=500, fn=3.0, zeta=0.9, delay=0.04)
    rep = T.analyze(ld)
    f = finding(rep, "vibration")
    assert f is not None and f.severity == SEV_BAD and "先處理振動" in f.title
    assert no_gain_increase(rep)
    # PIDx(Tar/Act 為 rad/s、沒有單位):Dmod 常 < 1 → slew 限制器;追蹤 / 延遲用 RATE(未濾波目標)
    ld2 = make_log(fs=400.0, dur=60.0, pid=True, pid_dmod=0.7)
    rep2 = T.analyze(ld2)
    assert rep2.data["PID"] and rep2.data["rate_source"] == "RATE"
    assert rep2.axes["roll"].metrics["rate_target"] == "RATE.RDes"
    assert abs(rep2.axes["roll"].metrics["rms_err"] - make_log_rms(ld2)) / make_log_rms(ld2) < 0.05
    assert finding(rep2, "slew_limit", "roll") is not None, codes(rep2, "roll")
    assert "no_pid_log" not in codes(rep2)
    # 只有 PIDx(沒有 RATE):rad/s 自動換算
    ld3 = make_log(fs=400.0, dur=60.0, pid=True, with_rate=False)
    rep3 = T.analyze(ld3)
    assert rep3.data["rate_source"] in ("PIDR", "PIDP", "PIDY") and rep3.axes["roll"].level == "rate"
    print(f"  振動 70 m/s² → {f.title};slew 限制器 → {finding(rep2, 'slew_limit', 'roll').title};"
          f"有 RATE 時追蹤用 RATE.RDes,只有 PIDx 時 {rep3.data['rate_source']}(rad/s 自動換算)")


def test_rate_target_fltt():
    """SP-7:PIDx.Tar 已經過 FLTT 低通,RATE.RDes 沒有;延遲 / |H| 不能因為有沒有開 PID 記錄而跳門檻。"""
    kw = dict(fs=400.0, dur=70.0, fn=3.0, zeta=0.9, delay=0.03, seed=5, pid=True, pid_fltt=11.0,
              params={"ATC_RAT_RLL_FLTT": 11.0, "ATC_RAT_PIT_FLTT": 11.0, "ATC_RAT_YAW_FLTT": 11.0})
    ra = T.analyze(make_log(**kw))                                   # RATE + PIDx
    rb = T.analyze(make_log(with_rate=False, **kw))                       # 只有 PIDx(Tar 經 FLTT 11 Hz)
    la, lb = ra.axes["roll"].metrics["lag_ms"], rb.axes["roll"].metrics["lag_ms"]
    assert ra.axes["roll"].metrics["source"] == "RATE" and rb.axes["roll"].metrics["source"] == "PIDR"
    assert "FLTT 11" in rb.axes["roll"].metrics["rate_target"]
    assert abs(la - lb) < 4.0, (la, lb)                                 # 未補償時差 1000/(2π·11) ≈ 14.5 ms
    ba, bb = ra.axes["roll"].metrics["bandwidth_hz"], rb.axes["roll"].metrics["bandwidth_hz"]
    assert ba and bb and abs(ba - bb) / ba < 0.15, (ba, bb)
    print(f"  延遲 RATE {la:.1f} ms vs PIDx(+FLTT 補償){lb:.1f} ms;頻寬 {ba:.2f} / {bb:.2f} Hz")


def make_log_rms(ld: LogData) -> float:
    r = ld.index.table("RATE")
    m = (r["t"] >= 5) & (r["t"] <= ld.index.t_range()[1] - 6)
    return float(np.sqrt(np.mean((r["RDes"][m] - r["R"][m]) ** 2)))


def test_analyze_axis_saturation_and_trim():
    ld = make_log(fs=200.0, dur=60.0, fn=3.0, zeta=0.9, delay=0.04, out_sat={"roll": 0.05},
                  out_offset={"pitch": 0.08}, seed=8)
    rep = T.analyze(ld)
    f = finding(rep, "saturation", "roll")
    assert f is not None and abs(f.metrics["sat_pct"] - 5) < 1.5, codes(rep, "roll")
    assert sug(f, "ATC_ACCEL_R_MAX").direction == DIR_DOWN
    assert not any(s.direction == DIR_UP for x in rep.findings if x.axis == "roll" for s in x.suggestions)
    tr = finding(rep, "trim_offset", "pitch")
    assert tr is not None and "低頭" in tr.detail and abs(tr.metrics["out_mean"] - 0.08) < 0.01
    assert rep.data["RCOU_hz"] == 10.0
    print(f"  Roll 輸出飽和 {f.metrics['sat_pct']:.1f} % → 降 ACCEL、不加增益;Pitch 配平 {tr.title}")


def test_analyze_batch_notch():
    ld = make_log(fs=200.0, dur=50.0, batch_peak=120.0)
    rep = T.analyze(ld)
    f = finding(rep, "notch_setup")
    assert f is not None and rep.data["batch"], codes(rep)
    fr = sug(f, "INS_HNTCH_FREQ").proposed
    assert abs(fr - 120) < 3, fr
    assert abs(sug(f, "INS_HNTCH_BW").proposed - fr / 2) < 2
    assert sug(f, "INS_HNTCH_REF").proposed == 0.35 and sug(f, "INS_HNTCH_ENABLE").proposed == 1
    bs = T.batch_gyro_spectrum(ld.index, [(0, 100)])
    assert bs["n_batches"] >= 30 and abs(bs["peak_hz"] - 120) < 3
    print(f"  批次 FFT {bs['n_batches']} 批 → 噪音峰 {bs['peak_hz']:.1f} Hz → INS_HNTCH_FREQ {fr:g}、BW {sug(f, 'INS_HNTCH_BW').proposed:g}")


def test_analyze_low_rate():
    ld = make_log(fs=400.0, dur=90.0, att_fs=10.0, osc={"roll": (9.5, 8.0)}, imu_fs=25.0, ctrl=True)
    rep = T.analyze(ld)
    lr = finding(rep, "low_rate")
    assert lr is not None and "10 Hz" in lr.detail and "Nyquist 5" in lr.detail
    bm = sug(lr, "LOG_BITMASK")
    assert bm.direction == DIR_SET and bm.proposed == 180223.0 and bm.priority == 1
    assert rep.axes["roll"].sample_hz < 11 and abs(rep.axes["roll"].spectrum.nyquist - 5.0) < 0.01
    assert no_gain_increase(rep), [(f.code, s.param) for f in rep.findings for s in f.suggestions
                                   if s.direction == DIR_UP]
    # CTRL D/P 1.5:馬達噪音也會造成 → 無法確定是震盪,只提醒、不給增益,卡片也不顯示震盪頻率
    sus = finding(rep, "rate_osc_suspect", "roll")
    assert sus is not None and sus.severity == SEV_INFO and "混疊" in sus.detail and "15.5" in sus.detail, \
        (codes(rep, "roll"), sus and sus.detail)
    assert not sus.suggestions and "ATC_RAT_RLL_D 約 20 %" in sus.detail
    assert rep.axes["roll"].metrics.get("osc_hz") is None and rep.axes["roll"].metrics["imu_peak_hz"] > 9
    # 同樣的 9.5 Hz 震盪,但 P 較高(D/P 增益比小)、CTRL P 項主導且推估約 9.5 Hz → 互相印證:warn、小降 P
    ldp = make_log(fs=400.0, dur=90.0, att_fs=10.0, osc={"roll": (9.5, 8.0)}, imu_fs=25.0, ctrl=True,
                   ctrl_pd=(0.05, 0.0325), params={"ATC_RAT_RLL_P": 0.3, "ATC_RAT_RLL_I": 0.3})
    repp = T.analyze(ldp)
    susp = finding(repp, "rate_osc_suspect", "roll")
    assert susp is not None and susp.severity == SEV_WARN and susp.metrics["support"] == "p", \
        (codes(repp, "roll"), susp and susp.detail)
    assert sug(susp, "ATC_RAT_RLL_P").direction == DIR_DOWN and sug(susp, "ATC_RAT_RLL_P").priority == 2
    assert abs(repp.axes["roll"].metrics["osc_hz"] - 9.5) < 0.5 and no_gain_increase(repp)
    # 沒有 IMU、只有 CTRL 線索 → 只提醒(info),不給增益建議
    ld2 = make_log(fs=400.0, dur=90.0, att_fs=10.0, osc={"roll": (12.0, 8.0)}, ctrl=True)
    rep2 = T.analyze(ld2)
    s2 = finding(rep2, "rate_osc_suspect", "roll")
    assert s2 is not None and s2.severity == SEV_INFO and not s2.suggestions, codes(rep2, "roll")
    print(f"  10 Hz:{lr.title};LOG_BITMASK → {bm.proposed:.0f};Roll {sus.title}({sus.severity});"
          f"CTRL P 主導 → {susp.severity} 降 P;只有 CTRL → {s2.title}")


def test_analyze_quadplane():
    ld = make_log(fs=400.0, dur=70.0, osc={"roll": (22.0, 9.0)}, quadplane=True, pid=True,
                  modes=[(0.5, "QHOVER"), (30.0, "FBWA"), (40.0, "QLOITER")])
    rep = T.analyze(ld)
    assert rep.param_prefix == "Q_A_" and rep.vehicle == "plane" and rep.frame == "quad X",         (rep.param_prefix, rep.vehicle, rep.frame)
    assert rep.data["PID"] and rep.axes["roll"].metrics.get("pid_d_over_p") is not None   # PIQx 細項
    f = finding(rep, "rate_osc_high", "roll")
    assert f is not None and sug(f, "Q_A_RAT_RLL_D") is not None, codes(rep, "roll")
    assert abs(rep.flight_s - (64.0 - 5.0 - 10.0)) < 0.5, rep.flight_s  # 5–64 s,FBWA 的 10 s 不算
    assert any("VTOL" in n for n in rep.notes)
    # 只有 PIQx(沒有 RATE)時用 PIQR 的 Tar / Act
    rep_p = T.analyze(make_log(fs=400.0, dur=70.0, osc={"roll": (22.0, 9.0)}, quadplane=True, pid=True, with_rate=False,
                               modes=[(0.5, "QHOVER")]))
    assert rep_p.data["rate_source"] in ("PIQR", "PIQP", "PIQY")
    print(f"  QuadPlane:前綴 {rep.param_prefix};排除固定翼模式後 {rep.flight_s:.0f} s;"
          f"建議 {sug(f, 'Q_A_RAT_RLL_D').param} → {sug(f, 'Q_A_RAT_RLL_D').proposed};只有 PIQx → {rep_p.data['rate_source']}")


def test_quadplane_throttle_and_logging():
    """F3 / F5:QuadPlane 的升力油門在 QTUN(Plane 的 CTUN.ThO 是推進油門);LOG_BITMASK 用 bit 20。"""
    kw = dict(fs=400.0, dur=70.0, quadplane=True, pid=True, mode="QHOVER", params={"Q_M_HOVER_LEARN": 0.0})
    for ev in (True, False):
        rep = T.analyze(make_log(ev=ev, **kw))
        assert abs(rep.flight_s - 59.0) < 1.0, (ev, rep.flight_s)          # 沒有 EV 時用 QTUN 判斷在空中
        assert "hover_low" not in codes(rep) and "no_flight" not in codes(rep)
        assert not any(s.param.endswith("THST_HOVER") for f in rep.findings for s in f.suggestions)
    # 懸停與 Q_M_THST_HOVER 不符時,實測值取自 QTUN
    rep2 = T.analyze(make_log(params={"Q_M_HOVER_LEARN": 0.0, "Q_M_THST_HOVER": 0.5},
                              **{k: v for k, v in kw.items() if k != "params"}))
    hp = finding(rep2, "hover_param")
    assert hp is not None and sug(hp, "Q_M_THST_HOVER").proposed == 0.35 and "QTUN" in hp.detail
    # 沒有 QTUN:不讀 Plane 的 CTUN,跳過懸停檢查並註明
    rep3 = T.analyze(make_log(qtun=False, **kw))
    assert not any(c.startswith("hover") for c in codes(rep3)) and any("QTUN" in n for n in rep3.notes)
    # F5:Plane 的 LOG_BITMASK:加 ATTITUDE_FULLRATE(bit 20),不提 PID bit 12
    rep4 = T.analyze(make_log(fs=400.0, dur=70.0, att_fs=10.0, quadplane=True, params={"LOG_BITMASK": 65470.0}))
    bm = sug(finding(rep4, "low_rate"), "LOG_BITMASK")
    assert bm.proposed == float(65470 | (1 << 20)) and "bit 20" in bm.reason and "bit 12" not in bm.reason
    assert rep4.data["fast_attitude"] is False
    assert T._fast_attitude(65470 | (1 << 20), True, 10.0) and not T._fast_attitude(65470 | 1 | 4096, True, 10.0)
    print(f"  QuadPlane:QTUN 判斷在空中 / 懸停;LOG_BITMASK 65470 → {bm.proposed:.0f}(bit 20)")


def test_analyze_degenerate():
    # 固定翼(沒有 Q_ENABLE)→ 不支援
    tables = {"ATT": {"t": np.arange(100) / 10, "DesRoll": np.zeros(100), "Roll": np.zeros(100)},
              "PARM": {"t": np.zeros(1), "Name": np.array(["RLL2SRV_P"]), "Value": np.array([1.0])}}
    ld = LogData(sources={}, events=[], index=LogIndex.from_tables(tables, vehicle="plane"))
    rep = T.analyze(ld)
    assert [f.code for f in rep.findings] == ["unsupported"] and not rep.axes and rep.vehicle == "plane"
    # 飛行太短
    rep2 = T.analyze(make_log(fs=100.0, dur=14.0, att_fs=100.0))
    assert "no_flight" in codes(rep2) and not rep2.axes
    # 沒有 index
    rep3 = T.analyze(LogData(sources={}))
    assert codes(rep3) == ["no_index"]
    # 子範圍
    ld4 = make_log(fs=100.0, dur=60.0, att_fs=100.0)
    rep4 = T.analyze(ld4, 20.0, 40.0)
    assert rep4.t0 == 20.0 and rep4.t1 == 40.0 and abs(rep4.flight_s - 20.0) < 0.1, rep4.flight_s
    for r in (rep, rep2, rep3, rep4):
        assert T.format_report_text(r).startswith("PID 調參建議")
    print("  固定翼 → unsupported;飛行 < 10 s → no_flight;沒有 index → no_index;子範圍 20–40 s → 20 s")


def test_analyze_sim_logdata():
    ld = sim.make_sim_logdata()
    rep = T.analyze(ld)
    assert set(rep.axes) == {"roll", "pitch", "yaw"}
    assert all(aa.level == "angle" and aa.t_ang is not None for aa in rep.axes.values())
    assert "no_rate_data" in codes(rep) and "low_rate" in codes(rep)
    assert all(SEV_ORDER[aa.verdict] <= SEV_ORDER[SEV_INFO] for aa in rep.axes.values()), \
        {a: aa.verdict for a, aa in rep.axes.items()}
    lag = rep.axes["roll"].metrics["lag_ms"]
    assert abs(lag - 100) < 10, lag                    # 模擬的期望姿態領先 0.1 s
    assert rep.frame == "quad X" and rep.vehicle == "copter"
    assert no_gain_increase(rep)
    print(f"  內建模擬:只有 ATT 25 Hz → 角度層級;延遲 {lag:.0f} ms(真 100);各軸 "
          + "、".join(f"{a} {aa.verdict}" for a, aa in rep.axes.items()))


# =====================================================================
# 校正回歸:core.sim_tuning 情境 × 記錄設定
# =====================================================================

# 記錄設定:高速 + PIDx、高速但沒有 PID、10 Hz(仿使用者的 LOG_BITMASK 176126)
MATRIX_MODES = {"fast+PID": dict(fast_log=True, pid_log=True), "fast": dict(fast_log=True, pid_log=False),
                "10Hz": dict(fast_log=False)}
# sim 的 DIR_CHECK 標籤 → (應出現的 finding code, 檢查項目名稱的關鍵字)
MATRIX_CHECKS = {"motor_headroom": ("motor_saturation", "推重比"), "vibration": ("vibration", "減振"),
                 "cg_balance": ("bias", "重心")}
# 10 Hz 時仍應給出的方向(看不到的就不要求;ang_p_high 只有接近 Nyquist 的 |H| 弱證據,方向對即可)
MATRIX_SLOW_EXPECT = {
    "p_high": [("ATC_RAT_RLL_P", DIR_DOWN), ("ATC_RAT_PIT_P", DIR_DOWN)],     # IMU 25 Hz 峰 + CTRL P 主導
    "d_high": [("ATC_RAT_RLL_D", DIR_DOWN), ("ATC_RAT_PIT_D", DIR_DOWN)],     # CTRL D/P ≈ 7.5
    "p_low": [("ATC_RAT_RLL_P", DIR_UP), ("ATC_RAT_PIT_P", DIR_UP)],          # 延遲 + CTRL 低頻佐證
    "i_low_bias": [("ATC_RAT_RLL_I", DIR_UP), ("ATC_RAT_PIT_I", DIR_UP)],
    "ang_p_high": [("ATC_ANG_RLL_P", DIR_DOWN), ("ATC_ANG_PIT_P", DIR_DOWN)],
    "yaw_p_low": [("ATC_RAT_YAW_P", DIR_UP)],
}
MATRIX_SLOW_CODES = {"saturation": "motor_saturation", "vibration": "vibration", "noisy_notch": "notch_setup"}
_DIR_MARK = {DIR_UP: "↑", DIR_DOWN: "↓", DIR_SET: "=", DIR_CHECK: "?"}


def _gain_order(rep: TuningReport) -> list[tuple[str, str]]:
    """去重排序後的增益建議(report.suggestions() 的順序)。"""
    return [(s.param, s.direction) for s in rep.suggestions()
            if s.direction in (DIR_UP, DIR_DOWN) and T._is_gain(s.param)]


def matrix_errors(sc: str, mode: str, rep: TuningReport, info: dict) -> list[str]:
    """一格(情境 × 記錄設定)的驗收條件;回傳不符合的項目。"""
    errs = []
    allsug = [(s.param, s.direction) for f in rep.findings for s in f.suggestions]
    allcodes = codes(rep)
    for x in info["forbid"]:
        if x in allsug:
            errs.append(f"出現禁止的建議 {x}")
    warn = [f.code for f in rep.findings if SEV_ORDER[f.severity] >= SEV_ORDER[SEV_WARN]]
    go = _gain_order(rep)
    if info["expect_ok"]:
        if warn:
            errs.append(f"good 有 warn/bad:{warn}")
        if go:
            errs.append(f"good 有增益建議:{go}")
    if sc in ("saturation", "vibration") and not no_gain_increase(rep):
        errs.append("飽和 / 振動情境建議了提高增益")
    if sc in ("saturation", "vibration") and "ok" in allcodes:          # F11:刻意不判斷的軸不能說「追蹤良好」
        errs.append("飽和 / 振動情境卻有軸「追蹤良好」")
    if mode != "10Hz":
        for p, d in info["expected"]:
            if d == DIR_CHECK:
                code, key = MATRIX_CHECKS[p]
                f = finding(rep, code)
                if f is None or not any(s.direction == DIR_CHECK and key in s.param for s in f.suggestions):
                    errs.append(f"缺少檢查 {p}({code})")
            elif (p, d) not in allsug:
                errs.append(f"缺少 {p} {d}")
        gexp = [e for e in info["expected"] if e[1] in (DIR_UP, DIR_DOWN)]
        if gexp and (not go or go[0] not in gexp):
            errs.append(f"第一個增益建議 {go[:1]} 不是預期的 {gexp}")
        if any(c in ("low_rate", "rate_osc_suspect") for c in allcodes):
            errs.append("高速記錄卻判成低速 / 疑似")
        if mode == "fast" and "no_pid_log" not in allcodes:
            errs.append("沒有 PID 記錄卻沒提醒")
    else:
        if any(c in ("rate_osc_high", "rate_osc_mid", "rate_underdamped") for c in allcodes):
            errs.append("10 Hz 卻直接宣稱高頻震盪 / 阻尼")
        lr = finding(rep, "low_rate")
        if lr is None or not any(s.param == "LOG_BITMASK" and s.proposed == 180223.0 for s in lr.suggestions):
            errs.append("10 Hz 沒有說明限制 / LOG_BITMASK 180223")
        for p, d in MATRIX_SLOW_EXPECT.get(sc, []):
            if (p, d) not in allsug:
                errs.append(f"10 Hz 缺少 {p} {d}")
        if sc in MATRIX_SLOW_EXPECT and go and go[0] not in MATRIX_SLOW_EXPECT[sc]:
            errs.append(f"10 Hz 第一個增益建議 {go[0]} 不是預期的")
        if sc in MATRIX_SLOW_CODES and MATRIX_SLOW_CODES[sc] not in allcodes:
            errs.append(f"10 Hz 缺少 {MATRIX_SLOW_CODES[sc]}")
    for p, _, v in info["expected_logging"]:        # 缺 ATTITUDE_FAST / PID → LOG_BITMASK;沒 batch → INS_LOG_BAT_MASK
        if not any(s.param == p and s.proposed == v for f in rep.findings for s in f.suggestions):
            errs.append(f"缺少記錄建議 {p} = {v:g}")
    return errs


def test_sim_scenario_matrix():
    """10 個模擬情境 × 3 種記錄設定:主要修正方向、排序、安全規則(TUNING_MATRIX=0 可略過)。"""
    if os.environ.get("TUNING_MATRIX", "1") == "0":
        print("  (略過:TUNING_MATRIX=0)")
        return
    from core import sim_tuning as ST
    seed = int(os.environ.get("TUNING_MATRIX_SEED", "1"))
    fails = []
    t0 = time.perf_counter()
    t_an = 0.0
    for sc in ST.SCENARIOS:
        for mode, kw in MATRIX_MODES.items():
            ld, info = ST.make_tuning_sim(sc, seed=seed, **kw)
            ta = time.perf_counter()
            rep = T.analyze(ld)
            t_an += time.perf_counter() - ta
            errs = matrix_errors(sc, mode, rep, info)
            top = [f"{f.axis[0].upper() if f.axis else '-'}:{f.code}" for f in rep.findings
                   if SEV_ORDER[f.severity] >= SEV_ORDER[SEV_WARN]][:3]
            gs = " ".join(f"{p.replace('ATC_', '')}{_DIR_MARK[d]}" for p, d in _gain_order(rep)[:2]) or "—"
            print(f"  {'OK ' if not errs else '!! '}{sc:11s} {mode:8s} {rep.verdict:4s} 增益:{gs:30s} "
                  f"{', '.join(top) or '(只有資料 / 提醒)'}" + (f"  ← {errs}" if errs else ""))
            if errs:
                fails.append((sc, mode, errs))
    print(f"  {len(ST.SCENARIOS) * len(MATRIX_MODES)} 格,模擬 + 分析 {time.perf_counter() - t0:.1f} s"
          f"(分析 {t_an:.2f} s),seed {seed}")
    assert not fails, fails


def _gain_cuts(rep: TuningReport, max_priority: int = 9) -> list[tuple[str, int]]:
    return [(s.param, s.priority) for f in rep.findings for s in f.suggestions
            if s.direction == DIR_DOWN and s.priority <= max_priority
            and re.search(r"RAT_(RLL|PIT|YAW)_(P|D)$", s.param)]


def test_sim_stress_cells():
    """SP-1 / SP-2 / SP-3 / SP-4 / SP-5 / SP-9 / F2:模擬的壓力測試格(TUNING_MATRIX=0 可略過)。

    只懸停的 P / D 過高(期望與震盪高度同調)、濾波後陀螺儀的 108 Hz 窄頻噪音(100 / 50 Hz 記錄混疊到 8 Hz、
    400 Hz 時高於控制頻寬)、掉 1 % 訊息、MOT_PWM = 0 的舊韌體、INS_GYRO_FILTER 100 + 馬達噪音(壓到下限)。
    """
    if os.environ.get("TUNING_MATRIX", "1") == "0":
        print("  (略過:TUNING_MATRIX=0)")
        return
    from core import sim_tuning as ST
    seed = int(os.environ.get("TUNING_MATRIX_SEED", "1"))
    fails = []
    t0 = time.perf_counter()

    def run(sc, **kw):
        ld, info = ST.make_tuning_sim(sc, seed=seed, **kw)
        return T.analyze(ld), info

    def chk(name, cond, msg=""):
        print(f"  {'OK ' if cond else '!! '}{name}" + (f"  ← {msg}" if (msg and not cond) else ""))
        if not cond:
            fails.append((name, msg))

    pid400 = dict(log_hz=400.0, fast_log=True, pid_log=True)
    # SP-3:108 Hz 噪音在 100 / 50 Hz 記錄混疊到 8 Hz → 不降 P / D,提示「可能是噪音混疊」+ 迴圈頻率記錄
    for hz in (100.0, 50.0):
        rep, _ = run("good", log_hz=hz, fast_log=True, pid_log=True, tone=(108.0, 4.5))
        f = finding(rep, "rate_osc_suspect", "roll") or finding(rep, "gyro_noise", "roll")
        chk(f"good + 108 Hz 噪音 @ {hz:g} Hz:不降 P / D", not _gain_cuts(rep), _gain_cuts(rep))
        chk(f"good + 108 Hz 噪音 @ {hz:g} Hz:疑似噪音混疊 + LOG_FILE_RATEMAX = 0",
            f is not None and sug(f, "LOG_FILE_RATEMAX") is not None and f.severity == SEV_INFO,
            codes(rep, "roll"))
    # SP-5:400 Hz 時 108 Hz 高於控制頻寬 → gyro_noise(陷波);不降 P,D 最多只當順位 3 的次要項
    rep, _ = run("good", tone=(108.0, 4.5), **pid400)
    chk("good + 108 Hz 噪音 @ 400 Hz:gyro_noise", finding(rep, "gyro_noise", "roll") is not None, codes(rep, "roll"))
    chk("good + 108 Hz 噪音 @ 400 Hz:不降 P、D 只在順位 3", not _gain_cuts(rep, 2)
        and not any(p.endswith("_P") for p, _ in _gain_cuts(rep)), _gain_cuts(rep))
    # SP-1:只懸停時 P / D 過高的震盪(期望值同調)仍要抓到,RMS 與真值相差 < 30 %
    for sc, first in (("p_high", "ATC_RAT_RLL_P"), ("d_high", "ATC_RAT_RLL_D")):
        rep, info = run(sc, pilot="hover", **pid400)
        f = finding(rep, "rate_osc_mid", "roll") or finding(rep, "rate_osc_high", "roll")
        truth = info["osc"]["roll"]["band_rms"]
        ok = (f is not None and f.metrics["sustain"] >= T.OSC_SUSTAIN_FRAC
              and abs(f.metrics["osc_rms"] - truth) / truth < 0.3
              and min(f.suggestions, key=lambda s: s.priority).param == first)
        chk(f"{sc} 只懸停 @ 400 Hz:持續震盪、{first} 先降、RMS ≈ 真值 {truth:.1f}", ok,
            (codes(rep, "roll"), f and (f.metrics, [(s.param, s.priority) for s in f.suggestions])))
        if sc == "p_high" and f is not None:
            p = sug(f, "ATC_RAT_RLL_P")
            chk("p_high 只懸停:P −20 %", p is not None and abs(p.change_pct + 20) < 1, p)
    # SP-2:掉 1 % 訊息 → 涵蓋率仍 > 80 %、仍在角速度層級判斷出震盪
    rep, _ = run("p_high", drop_frac=0.01, **pid400)
    chk("p_high 掉 1 % 訊息:涵蓋率 > 80 %、rate_osc_mid",
        rep.data.get("coverage", 0) > 0.8 and finding(rep, "rate_osc_mid", "roll") is not None
        and rep.axes["roll"].level == "rate", (rep.data.get("coverage"), codes(rep, "roll")))
    # SP-4:MOT_PWM_MIN / MAX = 0(4.0 / 4.1)→ 用 RC3 範圍,仍判斷出飽和、不提高增益
    for kw in (dict(fast_log=False), dict(log_hz=100.0, fast_log=True, pid_log=True)):
        rep, _ = run("saturation", legacy_pwm=True, **kw)
        chk(f"saturation + MOT_PWM = 0({'10 Hz' if not kw['fast_log'] else '100 Hz'}):馬達飽和、不加增益",
            finding(rep, "motor_saturation") is not None and no_gain_increase(rep), codes(rep))
    # F2:INS_GYRO_FILTER 100、沒有陷波 → 雜訊推動輸出、壓到下限;先處理雜訊,不說推力不足、不降 ACCEL
    rep, info = run("noisy_notch", overrides={"INS_GYRO_FILTER": 100.0}, **pid400)
    allsug = [s for f in rep.findings for s in f.suggestions]
    ml = finding(rep, "motor_sat_low")
    chk("noisy_notch GF100:motor_sat_low(雜訊推動)", ml is not None and "雜訊" in ml.title
        and SEV_ORDER[ml.severity] >= SEV_ORDER[SEV_WARN] and finding(rep, "motor_saturation") is None, codes(rep))
    chk("noisy_notch GF100:沒有推重比檢查 / ACCEL 調降",
        not any(s.param == T.THRUST_CHECK or "ACCEL" in s.param for s in allsug),
        [(s.param, s.direction) for s in allsug if s.param == T.THRUST_CHECK or "ACCEL" in s.param])
    chk("noisy_notch GF100:陷波建議", finding(rep, "notch_setup") is not None and no_gain_increase(rep))
    print(f"  壓力測試 {time.perf_counter() - t0:.1f} s,seed {seed}")
    assert not fails, fails


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_real_log():
    if not os.path.exists(REAL_LOG):
        print(f"  (略過:找不到 {REAL_LOG})")
        return
    from core.logio import load_bin
    before = (_sha(REAL_LOG), os.path.getsize(REAL_LOG), os.path.getmtime(REAL_LOG))
    ld = load_bin(REAL_LOG)
    t0 = time.perf_counter()
    rep = T.analyze(ld)
    dt = time.perf_counter() - t0
    after = (_sha(REAL_LOG), os.path.getsize(REAL_LOG), os.path.getmtime(REAL_LOG))
    assert before == after, "實機 log 被改動了!"
    assert dt < 10, dt
    assert rep.vehicle == "copter" and rep.frame == "hexa X" and rep.param_prefix == "ATC_"
    assert abs(rep.data["RATE_hz"] - 10) < 0.5 and rep.data["PID"] is False and rep.data["batch"] is False
    assert 200 < rep.flight_s < 300, rep.flight_s
    lr = finding(rep, "low_rate")
    assert lr is not None and "10 Hz" in lr.detail
    assert sug(lr, "LOG_BITMASK").proposed == 180223.0 and sug(lr, "LOG_BITMASK").current == 176126.0
    assert finding(rep, "default_gains") is not None
    nf = finding(rep, "notch_off")
    assert nf is not None and sug(nf, "INS_LOG_BAT_MASK").proposed == 1.0
    assert no_gain_increase(rep)
    for ax, aa in rep.axes.items():
        assert aa.level == "rate" and abs(aa.sample_hz - 10) < 0.5 and aa.step is None
    # 校正後的判讀(10 Hz、全程 LOITER):不做資料撐不起的宣稱
    # — 不改任何增益(25 Hz IMU 峰 + CTRL D/P 1.47 無法和馬達噪音區分 → 只提醒 + 條件方向)
    # — LOITER 的角度 |H| 有位置回授偏差 → 不採用,改寫在 notes
    assert SEV_ORDER[rep.verdict] <= SEV_ORDER[SEV_INFO], [(f.code, f.severity) for f in rep.findings]
    sugs = rep.suggestions()
    assert sugs[0].param == "LOG_BITMASK" and not any(T._is_gain(s.param) for s in sugs), \
        [(s.param, s.direction) for s in sugs]
    assert not any(c.startswith("angle_") for c in codes(rep))
    assert any("LOITER" in n for n in rep.notes) and rep.data["pilot_share"] == 0.0
    sus = finding(rep, "rate_osc_suspect", "roll")
    assert sus is not None and sus.severity == SEV_INFO and "噪音" in sus.detail
    assert finding(rep, "trim_offset", "roll") is not None
    assert all(aa.metrics.get("osc_hz") is None for aa in rep.axes.values())
    assert finding(rep, "vibration_ok") is not None and finding(rep, "motor_saturation") is None
    txt = T.format_report_text(rep, ld.t_base)
    for token in ("180223", "10 Hz", "INS_LOG_BAT_MASK", "AutoTune", "陷波"):
        assert token in txt, token
    pf = rep.param_file()
    assert "LOG_BITMASK,180223" in pf
    print(f"  實機 log:{dt:.2f} s,飛行 {rep.flight_s:.0f} s,整體 {rep.verdict};"
          + ";".join(f"{a}:{aa.summary[:28]}…" for a, aa in rep.axes.items()))
    print("  建議前三:" + " / ".join(f"{s.param}→{s.proposed if s.proposed is not None else '檢查'}"
                                    for s in rep.suggestions()[:3]))


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    t_all = time.perf_counter()
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  !! 失敗:{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 60}\n{len(tests) - failed}/{len(tests)} 通過({time.perf_counter() - t_all:.1f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
