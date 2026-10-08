"""PID 調參閉迴路模擬:ArduCopter 4.5 姿態控制器 + 多旋翼機體 → 仿真 log。

用途:讓 PID 調參建議(core.tuning)有「已知答案」可驗證 —— 每個情境用特定參數「飛」一次,
症狀(震盪、遲滯、偏差、飽和、振動、馬達雜訊)由閉迴路動態自然產生,不是事後畫上去的。

模型(400 Hz,與 SCHED_LOOP_RATE 相同):
- 姿態控制(AC_AttitudeControl):駕駛員角度經 ATC_INPUT_TC 輸入整形(sqrt_controller +
  ATC_ACCEL_*_MAX 角加速度限制)、ATC_ANG_*_P(sqrt_controller)、ATC_RATE_FF_ENAB 角速度前饋。
- 角速度 PID(AC_PID::update_all):FLTT 目標濾波、FLTE 誤差濾波、對濾波後誤差微分再經 FLTD、
  P / I(IMAX、馬達飽和時只准往回積)/ D / FF / D_FF;輸出為正規化 −1..1 交給混控器。
- 混控器(AP_MotorsMatrix::output_armed_stabilizing):Quad X / Hexa X 係數正規化到 ±0.5、
  MOT_YAW_HEADROOM、姿態優先於油門、超出範圍等比縮放並設 limit 旗標;推力→PWM 走
  MOT_THST_EXPO / MOT_SPIN_MIN / MOT_SPIN_MAX / MOT_PWM_MIN / MOT_PWM_MAX。
- 機體:馬達一階延遲 τ、控制權 K(rad/s² / 單位輸出)、氣動阻尼、亂流力矩、重心偏移力矩;
  陀螺儀 = 真值經 INS_GYRO_FILTER 二階低通(LowPassFilter2p)+ 1 個迴圈延遲 + 感測雜訊
  (寬頻、馬達頻率諧波;雜訊在 2 kHz 生成、經同一低通後抽到 400 Hz —— 線性濾波可拆開算)。
- 垂直:高度 PD(ALT_HOLD 類)+ 傾角補償(ANGLE_BOOST);水平:傾角加速 + 阻力,駕駛員以
  小增益把飛機拉回原點(位置在 GPS / POS 裡)。

log(LogIndex.from_tables,格式 / 單位仿 ArduPilot 4.5):
  ATT / RATE(fast_log 時為 log_hz,否則 10 Hz,以「取樣」抽取 —— 混疊照實發生)、
  PIDR / PIDP / PIDY(Tar / Act / Err 為 rad/s,與 RATE 的 deg/s 不同)、CTRL、RCOU、RCIN、
  CTUN、MOTB、BAT、VIBE、IMU(25 Hz)、ISBH / ISBD(batch)、GPS / POS / ORGN、
  PARM(含 Default)、MODE / ARM / EV / MSG。

公開:
  make_tuning_sim(scenario, ...) -> (LogData, info)
                                  另有 pilot="hover"(只懸停)、tone=(Hz, °/s)(濾波後陀螺儀窄頻噪音)、
                                  drop_frac(掉訊息)、legacy_pwm(MOT_PWM_MIN/MAX = 0)等壓力測試選項
  SCENARIOS                       情境名稱 → 說明
  write_tuning_bin(logdata, path) 以仿真格式字元寫成 .bin(fakebin)
  rate_step_response / angle_step_response   無雜訊的步階實驗(校準與測試用)
  scenario_params / default_params           情境「飛」的參數 / ArduPilot 預設值
"""
from __future__ import annotations

import math
import time
import zlib

import numpy as np

from . import frames
from .tuning_model import DIR_CHECK, DIR_DOWN, DIR_SET, DIR_UP
from .types import LogData

__all__ = [
    "SCENARIOS", "PLANT", "LOOP_HZ", "SENSOR_HZ", "ORIGIN", "SIM_UNITS", "SIM_FORMATS",
    "LOG_BITMASK_USER", "make_tuning_sim", "write_tuning_bin", "default_params",
    "scenario_params", "rate_step_response", "angle_step_response", "log_bitmask",
]

# ======================================================================
# 常數與校準
# ======================================================================

LOOP_HZ = 400.0          # 主迴圈(SCHED_LOOP_RATE)
SENSOR_HZ = 2000.0       # IMU 取樣(雜訊生成、batch sampler)
T_BOOT = 60.0            # log 時基起點(秒)
ORIGIN = (23.70200, 120.43100, 38.0)
G = 9.80665
_D2R = math.pi / 180.0
_R2D = 180.0 / math.pi

# 機體校準(預設增益 → 角速度步階過衝 Roll 10.8 % / Pitch 9.2 % / Yaw 11.9 %、上升約 45 ms;
#   連續時間模型估計相位裕度約 55°、穿越頻率約 3.9 Hz;P ×3 → 約 7.5 Hz、D ×5 → 約 12 Hz 震盪)
#   K:每單位混控輸出的角加速度 rad/s²(Roll 200 ≈ 11 500 deg/s²);damp:氣動阻尼 1/s
#   tau_motor:馬達推力一階延遲;gyro_delay:陀螺儀到控制器的迴圈延遲(+ ZOH 半格 ≈ 3.75 ms)
PLANT = {
    "K": {"roll": 200.0, "pitch": 190.0, "yaw": 40.0},
    "damp": {"roll": 1.0, "pitch": 1.0, "yaw": 0.8},
    "tau_motor": 0.040,
    "gyro_delay": 1,
    "turb_sigma": {"roll": 1.2, "pitch": 1.2, "yaw": 0.4},     # 亂流角加速度 rad/s²(GM τ=0.3 s)
    "torque_noise": {"roll": 3.0, "pitch": 3.0, "yaw": 0.8},   # 槳流 / 推力漣波 rad/s²(5~60 Hz)
    "cg_offset": {"roll": 0.5, "pitch": -0.35},                # 重心偏移力矩 rad/s²
    "motor_hz_hover": 105.0,                                   # 懸停時馬達基頻
}

# 使用者真實 log 的 LOG_BITMASK(無 ATTITUDE_FAST bit0、無 PID bit12)
LOG_BITMASK_USER = 176126
_MASK_ATT_FAST = 1
_MASK_PID = 4096


def log_bitmask(fast_log: bool, pid_log: bool) -> int:
    """依 fast_log / pid_log 組出 LOG_BITMASK(以使用者的 176126 為底)。"""
    m = LOG_BITMASK_USER
    if fast_log:
        m |= _MASK_ATT_FAST
    if pid_log:
        m |= _MASK_PID
    return m


SCENARIOS: dict[str, str] = {
    "good": "調得好:預設增益,響應俐落、無持續震盪(對照組)",
    "p_high": "Roll / Pitch 角速度 P 過高(×3):中頻(約 8 Hz)持續震盪、步階大幅過衝",
    "d_high": "Roll / Pitch 角速度 D 過高(×5):約 12 Hz 震盪 + D 項放大高頻雜訊(馬達發燙)",
    "p_low": "Roll / Pitch 角速度 P、D 過低(×0.35):反應遲鈍、追蹤誤差與延遲大、無震盪",
    "i_low_bias": "Roll / Pitch 角速度 I 極小 + 重心偏移:實際角度長時間偏離期望角度",
    "ang_p_high": "Roll / Pitch 角度 P 過高(ATC_ANG_*_P = 12):角度層過衝 / 約 5 Hz 共振(角速度層正常)",
    "yaw_p_low": "Yaw 角速度 P / I 過低(×0.3):偏航跟不上、轉向遲滯",
    "saturation": "重機(懸停油門 0.65)+ 激烈操作:馬達頂到上限、操作時追不上",
    "vibration": "機架振動大(VIBE 35~60 m/s²、加速度計 clipping):陀螺儀雜訊大",
    "noisy_notch": "馬達雜訊峰(約 150 Hz)未開諧波陷波(INS_HNTCH_ENABLE=0);預設附 batch 取樣",
}

# 預設增益(ArduCopter 4.5)
_P0, _I0, _D0 = 0.135, 0.135, 0.0036

# 各情境:參數覆寫、機體 / 雜訊 / 操作風格修改、預期的主要修正方向
_SPEC: dict[str, dict] = {
    "good": dict(
        params={}, expected=[], codes=[], forbid=[]),
    "p_high": dict(
        params={"ATC_RAT_RLL_P": round(_P0 * 3.0, 4), "ATC_RAT_PIT_P": round(_P0 * 3.0, 4)},
        expected=[("ATC_RAT_RLL_P", DIR_DOWN), ("ATC_RAT_PIT_P", DIR_DOWN)],
        codes=["rate_osc_mid"], forbid=[("ATC_RAT_RLL_P", DIR_UP), ("ATC_RAT_PIT_P", DIR_UP)],
        osc_axes=("roll", "pitch")),
    "d_high": dict(
        params={"ATC_RAT_RLL_D": round(_D0 * 5.0, 5), "ATC_RAT_PIT_D": round(_D0 * 5.0, 5)},
        expected=[("ATC_RAT_RLL_D", DIR_DOWN), ("ATC_RAT_PIT_D", DIR_DOWN)],
        codes=["rate_osc_high"], forbid=[("ATC_RAT_RLL_D", DIR_UP), ("ATC_RAT_PIT_D", DIR_UP)],
        osc_axes=("roll", "pitch")),
    "p_low": dict(
        params={"ATC_RAT_RLL_P": round(_P0 * 0.35, 4), "ATC_RAT_PIT_P": round(_P0 * 0.35, 4),
                "ATC_RAT_RLL_D": round(_D0 * 0.35, 5), "ATC_RAT_PIT_D": round(_D0 * 0.35, 5)},
        expected=[("ATC_RAT_RLL_P", DIR_UP), ("ATC_RAT_PIT_P", DIR_UP)],
        secondary=[("ATC_RAT_RLL_D", DIR_UP), ("ATC_RAT_PIT_D", DIR_UP)],
        codes=["rate_lag"], forbid=[("ATC_RAT_RLL_P", DIR_DOWN), ("ATC_RAT_PIT_P", DIR_DOWN)]),
    "i_low_bias": dict(
        params={"ATC_RAT_RLL_I": 0.002, "ATC_RAT_PIT_I": 0.002},
        cg_offset={"roll": 6.5, "pitch": -5.8},
        expected=[("ATC_RAT_RLL_I", DIR_UP), ("ATC_RAT_PIT_I", DIR_UP)],
        secondary=[("cg_balance", DIR_CHECK)],
        codes=["bias"], forbid=[("ATC_RAT_RLL_I", DIR_DOWN), ("ATC_RAT_PIT_I", DIR_DOWN)]),
    "ang_p_high": dict(
        params={"ATC_ANG_RLL_P": 12.0, "ATC_ANG_PIT_P": 12.0},
        expected=[("ATC_ANG_RLL_P", DIR_DOWN), ("ATC_ANG_PIT_P", DIR_DOWN)],
        codes=["angle_overshoot"], forbid=[("ATC_ANG_RLL_P", DIR_UP), ("ATC_ANG_PIT_P", DIR_UP)]),
    "yaw_p_low": dict(
        params={"ATC_RAT_YAW_P": round(0.18 * 0.3, 4), "ATC_RAT_YAW_I": round(0.018 * 0.3, 5)},
        expected=[("ATC_RAT_YAW_P", DIR_UP), ("ATC_RAT_YAW_I", DIR_UP)],
        codes=["yaw_lag"], forbid=[("ATC_RAT_YAW_P", DIR_DOWN)], pilot="yaw"),
    "saturation": dict(
        params={"MOT_THST_HOVER": 0.65},
        k_scale=0.55, pilot="aggressive",     # 懸停 0.65 vs 0.35 → 質量 ×1.86、慣量大 → K ×0.55
        expected=[("motor_headroom", DIR_CHECK)],
        secondary=[("ATC_ACCEL_R_MAX", DIR_DOWN), ("ATC_ACCEL_P_MAX", DIR_DOWN)],
        codes=["saturation"],
        forbid=[(p, DIR_UP) for p in ("ATC_RAT_RLL_P", "ATC_RAT_PIT_P", "ATC_RAT_YAW_P",
                                      "ATC_RAT_RLL_D", "ATC_RAT_PIT_D", "ATC_ANG_RLL_P",
                                      "ATC_ANG_PIT_P")]),
    "vibration": dict(
        params={}, noise="vibration",
        expected=[("vibration", DIR_CHECK)],
        codes=["vibration"],
        forbid=[(p, DIR_UP) for p in ("ATC_RAT_RLL_P", "ATC_RAT_PIT_P", "ATC_RAT_RLL_D",
                                      "ATC_RAT_PIT_D", "INS_GYRO_FILTER")]),
    "noisy_notch": dict(
        params={}, noise="motor_peak", motor_hz=150.0, batch=True,
        expected=[("INS_HNTCH_ENABLE", DIR_SET)],
        codes=["noise_peak"],
        forbid=[("ATC_RAT_RLL_D", DIR_UP), ("ATC_RAT_PIT_D", DIR_UP), ("INS_GYRO_FILTER", DIR_UP)]),
}

# 只有 10 Hz log(使用者的情況)時,這個症狀還看不看得出來
_SLOW_LOG = {
    "good": ("yes", "10 Hz 看得出沒有明顯問題,但 5 Hz 以上的震盪無法排除"),
    "p_high": ("partial", "約 8 Hz 震盪在 10 Hz log 混疊到約 2 Hz;CTRL.RMSRollP(迴圈頻率計算)仍明顯偏高"),
    "d_high": ("partial", "約 12 Hz 震盪混疊到約 2 Hz;CTRL.RMSRollD 仍明顯偏高"),
    "p_low": ("yes", "延遲與追蹤誤差在 10 Hz 仍看得出"),
    "i_low_bias": ("yes", "穩態偏差與取樣率無關"),
    "ang_p_high": ("no", "角度層共振約 5 Hz,等於 10 Hz 的 Nyquist,無法判斷"),
    "yaw_p_low": ("yes", "偏航遲滯在 10 Hz 仍看得出"),
    "saturation": ("yes", "RCOU 10 Hz 即可看出頂到上限、CTUN.ThO 與 MOTB.ThrOut 的差"),
    "vibration": ("yes", "VIBE / Clip 本來就是 10 Hz 摘要"),
    "noisy_notch": ("no", "要有 batch 取樣(INS_LOG_BAT_MASK=1)才看得到馬達頻率的雜訊峰"),
}

# 雜訊等級(deg/s 為陀螺儀原始值、m/s² 為加速度計原始值;皆為「馬達轉動中」的值)
_NOISE = {
    "normal": dict(gyro_white=0.6, gyro_broad=4.0, gyro_motor=8.0, gyro_motor2=3.0,
                   acc_broad=5.0, acc_motor=6.0, acc_spike=0.0, acc_z_gain=1.6),
    "vibration": dict(gyro_white=1.5, gyro_broad=40.0, gyro_motor=30.0, gyro_motor2=15.0,
                      acc_broad=22.0, acc_motor=18.0, acc_spike=27.0, acc_z_gain=1.3),
    "motor_peak": dict(gyro_white=0.6, gyro_broad=5.0, gyro_motor=160.0, gyro_motor2=55.0,
                       acc_broad=7.0, acc_motor=12.0, acc_spike=0.0, acc_z_gain=1.5),
}

# ArduCopter 4.5 參數預設值(只列模擬 / 分析會用到的)
_DEFAULTS: dict[str, float] = {
    "ANGLE_MAX": 3000.0,
    "ATC_ACCEL_P_MAX": 110000.0, "ATC_ACCEL_R_MAX": 110000.0, "ATC_ACCEL_Y_MAX": 27000.0,
    "ATC_ANGLE_BOOST": 1.0, "ATC_ANG_LIM_TC": 1.0,
    "ATC_ANG_PIT_P": 4.5, "ATC_ANG_RLL_P": 4.5, "ATC_ANG_YAW_P": 4.5,
    "ATC_INPUT_TC": 0.15, "ATC_RATE_FF_ENAB": 1.0,
    "ATC_RATE_P_MAX": 0.0, "ATC_RATE_R_MAX": 0.0, "ATC_RATE_Y_MAX": 0.0,
    "ATC_SLEW_YAW": 6000.0, "ATC_THR_MIX_MAN": 0.1, "ATC_THR_MIX_MAX": 0.5, "ATC_THR_MIX_MIN": 0.1,
    "AUTOTUNE_AGGR": 0.1, "AUTOTUNE_AXES": 7.0, "AUTOTUNE_MIN_D": 0.001,
    "FRAME_CLASS": 0.0, "FRAME_TYPE": 1.0,
    "INS_ACCEL_FILTER": 20.0, "INS_FAST_SAMPLE": 1.0, "INS_GYRO_FILTER": 20.0,
    "INS_GYRO_RATE": 0.0, "INS_HNTCH_ENABLE": 0.0, "INS_HNTC2_ENABLE": 0.0,
    "INS_LOG_BAT_CNT": 1024.0, "INS_LOG_BAT_LGCT": 32.0, "INS_LOG_BAT_LGIN": 20.0,
    "INS_LOG_BAT_MASK": 0.0, "INS_LOG_BAT_OPT": 0.0,
    "LOG_BITMASK": 180222.0, "LOG_DISARMED": 0.0, "LOG_FILE_RATEMAX": 0.0,
    "MOT_BAT_VOLT_MAX": 0.0, "MOT_BAT_VOLT_MIN": 0.0, "MOT_HOVER_LEARN": 2.0,
    "MOT_PWM_MAX": 2000.0, "MOT_PWM_MIN": 1000.0, "MOT_PWM_TYPE": 0.0,
    "MOT_SPIN_ARM": 0.1, "MOT_SPIN_MAX": 0.95, "MOT_SPIN_MIN": 0.15, "MOT_SPOOL_TIME": 0.5,
    "MOT_THST_EXPO": 0.65, "MOT_THST_HOVER": 0.35, "MOT_YAW_HEADROOM": 200.0,
    "PILOT_ACCEL_Z": 250.0, "PILOT_SPEED_UP": 250.0, "PILOT_Y_RATE": 202.5, "PILOT_Y_RATE_TC": 0.0,
    "PSC_ACCZ_I": 1.0, "PSC_ACCZ_P": 0.5, "SCHED_LOOP_RATE": 400.0,
}
for _ax, _p, _i, _d, _flte, _fltd in (("RLL", _P0, _I0, _D0, 0.0, 20.0),
                                      ("PIT", _P0, _I0, _D0, 0.0, 20.0),
                                      ("YAW", 0.18, 0.018, 0.0, 2.5, 0.0)):
    _DEFAULTS.update({
        f"ATC_RAT_{_ax}_P": _p, f"ATC_RAT_{_ax}_I": _i, f"ATC_RAT_{_ax}_D": _d,
        f"ATC_RAT_{_ax}_FF": 0.0, f"ATC_RAT_{_ax}_D_FF": 0.0, f"ATC_RAT_{_ax}_IMAX": 0.5,
        f"ATC_RAT_{_ax}_FLTT": 20.0, f"ATC_RAT_{_ax}_FLTE": _flte, f"ATC_RAT_{_ax}_FLTD": _fltd,
        f"ATC_RAT_{_ax}_SMAX": 0.0, f"ATC_RAT_{_ax}_PDMX": 0.0,
        f"ATC_RAT_{_ax}_NTF": 0.0, f"ATC_RAT_{_ax}_NEF": 0.0,
    })
# 諧波陷波子參數:ENABLE=0 時 ArduPilot 不顯示 / 不記錄這些參數
_HNTCH_SUB = {"INS_HNTCH_MODE": 1.0, "INS_HNTCH_REF": 0.0, "INS_HNTCH_FREQ": 80.0,
              "INS_HNTCH_BW": 40.0, "INS_HNTCH_ATT": 40.0, "INS_HNTCH_HMNCS": 3.0,
              "INS_HNTCH_OPTS": 0.0}

# 使用者機上與 ArduPilot 預設不同、所有情境共用的值(仿使用者:yaw FLTE=2;學到的懸停油門)
_COMMON = {"ATC_RAT_YAW_FLTE": 2.0, "MOT_THST_HOVER": 0.35, "LOG_DISARMED": 1.0,
           "INS_FAST_SAMPLE": 1.0, "INS_GYRO_RATE": 1.0}

_FRAMES = {
    # 名稱 → (FRAME_CLASS, FRAME_TYPE, [(角度 deg, 偏航係數)...]、開機訊息)
    "quad": (1, 1, [(45, 1), (-135, 1), (-45, -1), (135, -1)], "QUAD/X"),
    "hexa": (2, 1, [(90, -1), (-90, 1), (-30, -1), (150, 1), (30, 1), (-150, -1)], "HEXA/X"),
}


def default_params(frame: str = "quad") -> dict[str, float]:
    """ArduPilot 4.5 預設值(PARM.Default 用)。"""
    d = dict(_DEFAULTS)
    d.update(_HNTCH_SUB)
    return d


def scenario_params(scenario: str = "good", frame: str = "quad",
                    overrides: dict[str, float] | None = None) -> dict[str, float]:
    """情境「飛」的參數值(未含 LOG_BITMASK / INS_LOG_BAT_MASK 的記錄設定)。"""
    if scenario not in _SPEC:
        raise ValueError(f"未知情境 {scenario!r};可用:{', '.join(SCENARIOS)}")
    if frame not in _FRAMES:
        raise ValueError(f"未知機架 {frame!r};可用:quad、hexa")
    p = dict(_DEFAULTS)
    p.update(_COMMON)
    fc, ft, _, _ = _FRAMES[frame]
    p["FRAME_CLASS"], p["FRAME_TYPE"] = float(fc), float(ft)
    p.update(_SPEC[scenario]["params"])
    if overrides:
        p.update({k: float(v) for k, v in overrides.items()})
    if p.get("INS_HNTCH_ENABLE", 0.0) > 0:
        for k, v in _HNTCH_SUB.items():
            p.setdefault(k, v)
    return p


# ======================================================================
# 小工具(numpy only)
# ======================================================================

def _lpf2_coef(fs: float, fc: float) -> tuple[float, float, float, float, float]:
    """ArduPilot LowPassFilter2p(雙線性 Butterworth)係數 (b0, b1, b2, a1, a2);fc ≤ 0 → 直通。"""
    if fc <= 0 or fc >= fs / 2:
        return 1.0, 0.0, 0.0, 0.0, 0.0
    ohm = math.tan(math.pi * fc / fs)
    c = 1.0 + 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm
    b0 = ohm * ohm / c
    return (b0, 2.0 * b0, b0, 2.0 * (ohm * ohm - 1.0) / c,
            (1.0 - 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm) / c)


def _notch_coef(fs: float, f0: float, bw: float, att_db: float) -> tuple[float, float, float, float, float]:
    """ArduPilot NotchFilter(中心 f0、頻寬 bw、衰減 att_db)係數 (b0, b1, b2, a1, a2)。"""
    f0 = min(max(f0, 1.0), 0.48 * fs)
    octaves = math.log2(max(f0 + bw / 2, f0 * 1.01) / max(f0 - bw / 2, 1.0))
    q = math.sqrt(2.0 ** octaves) / (2.0 ** octaves - 1.0)
    a_lin = 10.0 ** (-att_db / 40.0)
    omega = 2.0 * math.pi * f0 / fs
    alpha = math.sin(omega) / (2.0 * q)
    a0 = 1.0 + alpha
    return ((1.0 + alpha * a_lin * a_lin) / a0, -2.0 * math.cos(omega) / a0,
            (1.0 - alpha * a_lin * a_lin) / a0, -2.0 * math.cos(omega) / a0, (1.0 - alpha) / a0)


def _iir(x: np.ndarray, b, a, pad: int | None = None) -> np.ndarray:
    """因果 IIR(零初始狀態),以 FFT 計算:y = filter(b, a, x)。pad 需涵蓋脈衝響應衰減長度。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x.copy()
    pad = int(pad if pad is not None else min(max(4096, n // 4), 200_000))
    nfft = 1 << int(math.ceil(math.log2(n + pad)))
    w = np.exp(-2j * np.pi * np.fft.rfftfreq(nfft))
    num = np.polyval(np.asarray(b, dtype=np.float64)[::-1], w)
    den = np.polyval(np.asarray(a, dtype=np.float64)[::-1], w)
    y = np.fft.irfft(np.fft.rfft(x, nfft) * (num / den), nfft)[:n]
    return y


def _biquad(x: np.ndarray, coef) -> np.ndarray:
    b0, b1, b2, a1, a2 = coef
    return _iir(x, (b0, b1, b2), (1.0, a1, a2))


def _lpf1(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """一階低通(ArduPilot LowPassFilter / calc_lowpass_alpha_dt)。"""
    dt = 1.0 / fs
    alpha = dt / (dt + 1.0 / (2.0 * math.pi * fc))
    return _iir(x, (alpha,), (1.0, alpha - 1.0), pad=int(20 * fs / fc) + 64)


def _gm(rng, n: int, fs: float, tau: float, sigma: float) -> np.ndarray:
    """Gauss-Markov(一階 AR)雜訊,穩態標準差 sigma。"""
    if n == 0:
        return np.zeros(0)
    a = math.exp(-1.0 / (fs * tau))
    w = rng.normal(0.0, sigma * math.sqrt(1.0 - a * a), n)
    w[0] = rng.normal(0.0, sigma)
    return _iir(w, (1.0,), (1.0, -a), pad=int(15 * tau * fs) + 64)


def _bandnoise(rng, n: int, fs: float, f_lo: float, f_hi: float, sigma: float) -> np.ndarray:
    """頻帶限制的高斯雜訊(FFT 遮罩),標準差約 sigma。"""
    if n == 0 or sigma <= 0:
        return np.zeros(n)
    spec = np.fft.rfft(rng.normal(0.0, 1.0, n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spec[(f < f_lo) | (f > f_hi)] = 0.0
    y = np.fft.irfft(spec, n)
    s = float(np.std(y))
    return y * (sigma / s) if s > 0 else y


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win) / win
    xp = np.pad(x, (win // 2, win - 1 - win // 2), mode="edge")
    return np.convolve(xp, k, mode="valid")


def _f32(x) -> np.ndarray:
    """量化成 float32 再轉回 float64:寫成 .bin(格式 f)讀回逐位元相同。"""
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def _cd(x) -> np.ndarray:
    """量化到 0.01(格式 c / C / e:百分之一單位整數)。"""
    return np.round(np.asarray(x, dtype=np.float64) * 100.0) / 100.0


def _motor_factors(frame: str) -> list[tuple[float, float, float]]:
    """AP_MotorsMatrix:add_motor(角度, 偏航係數) → 正規化到 ±0.5 的 (roll, pitch, yaw) 係數。"""
    defs = _FRAMES[frame][2]
    r = [math.cos(math.radians(a + 90.0)) for a, _ in defs]
    p = [math.cos(math.radians(a)) for a, _ in defs]
    y = [float(yf) for _, yf in defs]
    r = [0.0 if abs(v) < 1e-9 else v for v in r]
    p = [0.0 if abs(v) < 1e-9 else v for v in p]
    rm, pm, ym = max(map(abs, r)), max(map(abs, p)), max(map(abs, y))
    return [(0.5 * a / rm, 0.5 * b / pm, 0.5 * c / ym) for a, b, c in zip(r, p, y)]


def _thrust_to_pwm(c: np.ndarray, prm: dict) -> np.ndarray:
    """推力(0..1)→ PWM(MOT_THST_EXPO、MOT_SPIN_MIN/MAX、MOT_PWM_MIN/MAX;lift_max = 1)。"""
    e = float(np.clip(prm["MOT_THST_EXPO"], -1.0, 1.0))
    c = np.clip(c, 0.0, 1.0)
    if abs(e) < 1e-6:
        ratio = c
    else:
        ratio = ((e - 1.0) + np.sqrt((1.0 - e) ** 2 + 4.0 * e * c)) / (2.0 * e)
    act = prm["MOT_SPIN_MIN"] + (prm["MOT_SPIN_MAX"] - prm["MOT_SPIN_MIN"]) * np.clip(ratio, 0, 1)
    return prm["MOT_PWM_MIN"] + (prm["MOT_PWM_MAX"] - prm["MOT_PWM_MIN"]) * act


# ======================================================================
# 控制器設定
# ======================================================================

def _alpha(dt: float, hz: float) -> float:
    """calc_lowpass_alpha_dt:截止 ≤ 0 → 1(不濾)。"""
    if hz <= 0:
        return 1.0
    return dt / (dt + 1.0 / (2.0 * math.pi * hz))


def _ctrl_cfg(prm: dict, frame: str, plant: dict, hover: float) -> dict:
    dt = 1.0 / LOOP_HZ
    cfg = {"dt": dt, "hover": hover, "motors": _motor_factors(frame)}
    for ax, key in (("roll", "RLL"), ("pitch", "PIT"), ("yaw", "YAW")):
        g = lambda s: float(prm[f"ATC_RAT_{key}_{s}"])      # noqa: E731
        cfg[ax] = dict(
            kp=g("P"), ki=g("I"), kd=g("D"), kff=g("FF"), kdff=g("D_FF"), imax=g("IMAX"),
            aT=_alpha(dt, g("FLTT")), aE=_alpha(dt, g("FLTE")), aD=_alpha(dt, g("FLTD")),
            K=plant["K"][ax], damp=plant["damp"][ax])
    deg_ss = lambda name: math.radians(float(prm[name]) / 100.0)   # noqa: E731  cdeg/s² → rad/s²
    acc = {"roll": deg_ss("ATC_ACCEL_R_MAX"), "pitch": deg_ss("ATC_ACCEL_P_MAX"),
           "yaw": deg_ss("ATC_ACCEL_Y_MAX")}
    cfg["acc"] = acc
    # 角度控制器 sqrt_controller 的二階限制:accel/2,夾在 ArduPilot 的上下限
    lim_rp = (math.radians(40.0), math.radians(720.0))
    lim_y = (math.radians(10.0), math.radians(120.0))
    cfg["acc_ctrl"] = {
        "roll": min(max(acc["roll"] / 2, lim_rp[0]), lim_rp[1]) if acc["roll"] > 0 else 0.0,
        "pitch": min(max(acc["pitch"] / 2, lim_rp[0]), lim_rp[1]) if acc["pitch"] > 0 else 0.0,
        "yaw": min(max(acc["yaw"] / 2, lim_y[0]), lim_y[1]) if acc["yaw"] > 0 else 0.0,
    }
    cfg["angp"] = {"roll": float(prm["ATC_ANG_RLL_P"]), "pitch": float(prm["ATC_ANG_PIT_P"]),
                   "yaw": float(prm["ATC_ANG_YAW_P"])}
    cfg["rate_max"] = {"roll": math.radians(float(prm["ATC_RATE_R_MAX"])),
                       "pitch": math.radians(float(prm["ATC_RATE_P_MAX"])),
                       "yaw": math.radians(float(prm["ATC_RATE_Y_MAX"]))}
    cfg["ff_enab"] = bool(prm["ATC_RATE_FF_ENAB"])
    cfg["input_tc"] = max(float(prm["ATC_INPUT_TC"]), 0.01)
    cfg["yaw_headroom"] = float(prm["MOT_YAW_HEADROOM"]) * 0.001
    cfg["thr_mix"] = float(prm["ATC_THR_MIX_MAX"])
    cfg["a_motor"] = 1.0 - math.exp(-dt / plant["tau_motor"])
    cfg["lpf"] = _lpf2_coef(LOOP_HZ, float(prm["INS_GYRO_FILTER"]))
    cfg["notches"] = _notch_list(prm, LOOP_HZ, hover)
    return cfg


def _notch_list(prm: dict, fs: float, hover: float) -> list[tuple]:
    """INS_HNTCH_ENABLE=1 時的諧波陷波係數(以懸停油門下的頻率近似成固定陷波)。"""
    if prm.get("INS_HNTCH_ENABLE", 0.0) <= 0:
        return []
    f = float(prm.get("INS_HNTCH_FREQ", 80.0))
    ref = float(prm.get("INS_HNTCH_REF", 0.0))
    if prm.get("INS_HNTCH_MODE", 1.0) == 1.0 and ref > 0:
        f *= math.sqrt(max(hover, 1e-3) / ref)
    bw = float(prm.get("INS_HNTCH_BW", f / 2))
    att = float(prm.get("INS_HNTCH_ATT", 40.0))
    hm = int(prm.get("INS_HNTCH_HMNCS", 3.0))
    out = []
    for h in range(1, 9):
        if hm & (1 << (h - 1)) and f * h < 0.48 * fs:
            out.append(_notch_coef(fs, f * h, bw * h, att))
    return out


# ======================================================================
# 主迴圈(400 Hz,純 Python 純量運算)
# ======================================================================

# 每步紀錄欄位(依序)
_REC = ("atr", "atp", "aty", "ar", "ap", "ay", "rtr", "rtp", "rty", "gmr", "gmp", "gmy",
        "wr", "wp", "wy",
        "tfr", "efr", "pr", "ir", "dr", "ffr",
        "tfp", "efp", "pp", "ip", "dp", "ffp",
        "tfy", "efy", "py", "iy", "dy", "ffy",
        "R", "P", "Y", "thr_out", "flags", "thr_in", "thr", "thr_avg_max",
        "z", "vz", "zt", "vzt", "pn", "pe", "vn", "ve", "des_r", "des_p", "az", "umt", "a_cmd")
_RI = {k: i for i, k in enumerate(_REC)}

# flags 位元
_F_LIM_R, _F_LIM_P, _F_LIM_Y, _F_THR_UP, _F_THR_LO, _F_LANDED = 1, 2, 4, 8, 16, 32


def _sqrt_ctrl(error: float, p: float, lim: float, dt: float) -> float:
    """AP_Math sqrt_controller。"""
    if lim <= 0.0:
        rate = error * p
    elif p == 0.0:
        rate = math.copysign(math.sqrt(2.0 * lim * abs(error)), error) if error else 0.0
    else:
        lin = lim / (p * p)
        if error > lin:
            rate = math.sqrt(2.0 * lim * (error - lin / 2.0))
        elif error < -lin:
            rate = -math.sqrt(2.0 * lim * (-error - lin / 2.0))
        else:
            rate = error * p
    m = abs(error) / dt
    return m if rate > m else (-m if rate < -m else rate)


def _simulate(cfg: dict, prog: dict, nz: dict, dist: dict, cg: tuple[float, float],
              n: int, init: dict | None = None) -> np.ndarray:
    """跑 n 個 400 Hz 步;回傳 (n, len(_REC)) 紀錄。

    prog:des_r / des_p(rad)、des_yr(rad/s)、zt / vzt(m, m/s)、stage(0 = 地面怠速,1 = 飛行)、
         acro(可選:(rate_r, rate_p, rate_y) rad/s,直接當角速度目標)
    nz:陀螺儀雜訊(已濾波,rad/s);dist:擾動角加速度(rad/s²);cg:重心力矩(空中才有)
    """
    dt = cfg["dt"]
    hover = cfg["hover"]
    R_, P_, Y_ = cfg["roll"], cfg["pitch"], cfg["yaw"]
    kp_r, ki_r, kd_r, kff_r, kdff_r, im_r = R_["kp"], R_["ki"], R_["kd"], R_["kff"], R_["kdff"], R_["imax"]
    kp_p, ki_p, kd_p, kff_p, kdff_p, im_p = P_["kp"], P_["ki"], P_["kd"], P_["kff"], P_["kdff"], P_["imax"]
    kp_y, ki_y, kd_y, kff_y, kdff_y, im_y = Y_["kp"], Y_["ki"], Y_["kd"], Y_["kff"], Y_["kdff"], Y_["imax"]
    aT_r, aE_r, aD_r = R_["aT"], R_["aE"], R_["aD"]
    aT_p, aE_p, aD_p = P_["aT"], P_["aE"], P_["aD"]
    aT_y, aE_y, aD_y = Y_["aT"], Y_["aE"], Y_["aD"]
    K_r, K_p, K_y = R_["K"], P_["K"], Y_["K"]
    c_r, c_p, c_y = R_["damp"], P_["damp"], Y_["damp"]
    acc_r, acc_p, acc_y = cfg["acc"]["roll"], cfg["acc"]["pitch"], cfg["acc"]["yaw"]
    al_r, al_p, al_y = cfg["acc_ctrl"]["roll"], cfg["acc_ctrl"]["pitch"], cfg["acc_ctrl"]["yaw"]
    ap_r, ap_p, ap_y = cfg["angp"]["roll"], cfg["angp"]["pitch"], cfg["angp"]["yaw"]
    rm_r, rm_p, rm_y = cfg["rate_max"]["roll"], cfg["rate_max"]["pitch"], cfg["rate_max"]["yaw"]
    ff_enab = cfg["ff_enab"]
    inv_tc = 1.0 / cfg["input_tc"]
    yaw_hr = cfg["yaw_headroom"]
    mix = cfg["thr_mix"]
    am = cfg["a_motor"]
    b0, b1, b2, a1, a2 = cfg["lpf"]
    notches = cfg["notches"]
    motors = cfg["motors"]
    sq = _sqrt_ctrl
    cos, tan, sin = math.cos, math.tan, math.sin

    des_r, des_p, des_yr = prog["des_r"], prog["des_p"], prog["des_yr"]
    zt_a, vzt_a, stage_a = prog["zt"], prog["vzt"], prog["stage"]
    acro = prog.get("acro")
    if acro is not None:
        acro_r, acro_p, acro_y = acro
    nzr, nzp, nzy = nz["r"], nz["p"], nz["y"]
    dsr, dsp, dsy = dist["r"], dist["p"], dist["y"]
    cg_r, cg_p = cg

    s = dict(ar=0.0, ap=0.0, ay=0.0, wr=0.0, wp=0.0, wy=0.0, z=0.0, vz=0.0, airborne=False,
             ground_r=0.0, ground_p=0.0, pn=0.0, pe=0.0, vn=0.0, ve=0.0, umt=0.0)
    if init:
        s.update(init)
    ar, ap, ay = s["ar"], s["ap"], s["ay"]
    wr, wp, wy = s["wr"], s["wp"], s["wy"]
    z, vz = s["z"], s["vz"]
    pn, pe, vn, ve = s["pn"], s["pe"], s["vn"], s["ve"]
    g_r, g_p = s["ground_r"], s["ground_p"]
    airborne = s["airborne"]
    umt = s["umt"]
    umr = ump = umy = 0.0
    # 陀螺儀低通狀態(以目前角速度初始化成穩態)
    gfr, gfp, gfy = wr, wp, wy
    dc = 1.0 + a1 + a2
    x1r = x2r = wr / (b0 + b1 + b2) * dc if dc else 0.0
    x1p = x2p = wp / (b0 + b1 + b2) * dc if dc else 0.0
    x1y = x2y = wy / (b0 + b1 + b2) * dc if dc else 0.0
    nst = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in notches]   # 每個陷波:r/p/y 各兩個延遲元素
    # 姿態目標 / 前饋
    atr, atp, aty = ar, ap, ay
    ffr = ffp = ffy = 0.0
    # PID 狀態
    tfr = tfp = tfy = 0.0
    efr = efp = efy = 0.0
    dfr = dfp = dfy = 0.0
    ir = ip = iy = 0.0
    lim_r = lim_p = lim_y = False
    izz = 0.0
    corr_r = corr_p = 0.0
    landed = not airborne
    rec = []
    push = rec.append

    for k in range(n):
        # --- 1. 陀螺儀:上一步的濾波值(1 個迴圈延遲)+ 雜訊;再把目前真值送進低通 ---
        gmr = gfr + nzr[k]
        gmp = gfp + nzp[k]
        gmy = gfy + nzy[k]
        x0 = wr - x1r * a1 - x2r * a2
        gfr = x0 * b0 + x1r * b1 + x2r * b2
        x2r = x1r
        x1r = x0
        x0 = wp - x1p * a1 - x2p * a2
        gfp = x0 * b0 + x1p * b1 + x2p * b2
        x2p = x1p
        x1p = x0
        x0 = wy - x1y * a1 - x2y * a2
        gfy = x0 * b0 + x1y * b1 + x2y * b2
        x2y = x1y
        x1y = x0
        for (nb0, nb1, nb2, na1, na2), st in zip(notches, nst):
            x0 = gfr - st[0] * na1 - st[1] * na2
            v = x0 * nb0 + st[0] * nb1 + st[1] * nb2
            st[1] = st[0]
            st[0] = x0
            gfr = v
            x0 = gfp - st[2] * na1 - st[3] * na2
            v = x0 * nb0 + st[2] * nb1 + st[3] * nb2
            st[3] = st[2]
            st[2] = x0
            gfp = v
            x0 = gfy - st[4] * na1 - st[5] * na2
            v = x0 * nb0 + st[4] * nb1 + st[5] * nb2
            st[5] = st[4]
            st[4] = x0
            gfy = v

        stage = stage_a[k]
        # --- 2. 駕駛員輸入 + 拉回原點的小修正(10 Hz 更新) ---
        if k % 40 == 0:
            if airborne:
                cy, sy = cos(ay), sin(ay)
                fwd = pn * cy + pe * sy
                rgt = -pn * sy + pe * cy
                vf = vn * cy + ve * sy
                vr = -vn * sy + ve * cy
                corr_p = 0.014 * fwd + 0.045 * vf
                corr_r = -(0.014 * rgt + 0.045 * vr)
                corr_p = 0.105 if corr_p > 0.105 else (-0.105 if corr_p < -0.105 else corr_p)
                corr_r = 0.105 if corr_r > 0.105 else (-0.105 if corr_r < -0.105 else corr_r)
            else:
                corr_r = corr_p = 0.0
        dr_ = des_r[k] + corr_r
        dp_ = des_p[k] + corr_p
        dyr = des_yr[k]

        # --- 3. 姿態目標:輸入整形(input_shaping_angle)/ 直接設定 ---
        if acro is not None:
            ffr, ffp, ffy = acro_r[k], acro_p[k], acro_y[k]
            atr += ffr * dt
            atp += ffp * dt
            aty += ffy * dt
        elif ff_enab:
            v = sq(dr_ - atr, inv_tc, acc_r, dt)
            lo = ffr - acc_r * dt
            hi = ffr + acc_r * dt
            ffr = lo if v < lo else (hi if v > hi else v)
            v = sq(dp_ - atp, inv_tc, acc_p, dt)
            lo = ffp - acc_p * dt
            hi = ffp + acc_p * dt
            ffp = lo if v < lo else (hi if v > hi else v)
            lo = ffy - acc_y * dt
            hi = ffy + acc_y * dt
            ffy = lo if dyr < lo else (hi if dyr > hi else dyr)
            atr += ffr * dt
            atp += ffp * dt
            aty += ffy * dt
        else:
            atr, atp = dr_, dp_
            aty += dyr * dt
            ffr = ffp = ffy = 0.0
        if landed:
            aty = ay           # 地面:偏航目標跟著實際(reset_yaw_target_and_rate)
            ffy = 0.0

        # --- 4. 角度誤差 → 角速度目標(+ 前饋) ---
        if acro is not None and not ff_enab:
            rtr, rtp, rty = ffr, ffp, ffy
        else:
            rtr = sq(atr - ar, ap_r, al_r, dt) + (ffr if (ff_enab or acro is not None) else 0.0)
            rtp = sq(atp - ap, ap_p, al_p, dt) + (ffp if (ff_enab or acro is not None) else 0.0)
            rty = sq(aty - ay, ap_y, al_y, dt) + (ffy if (ff_enab or acro is not None) else 0.0)
        if rm_r > 0:
            rtr = rm_r if rtr > rm_r else (-rm_r if rtr < -rm_r else rtr)
        if rm_p > 0:
            rtp = rm_p if rtp > rm_p else (-rm_p if rtp < -rm_p else rtp)
        if rm_y > 0:
            rty = rm_y if rty > rm_y else (-rm_y if rty < -rm_y else rty)

        # --- 5. 角速度 PID(AC_PID::update_all) ---
        tl = tfr
        el = efr
        tfr += aT_r * (rtr - tfr)
        efr += aE_r * ((tfr - gmr) - efr)
        dfr += aD_r * ((efr - el) / dt - dfr)
        if landed:
            ir = 0.0
        elif ki_r and (not lim_r or (ir > 0 > efr) or (ir < 0 < efr)):
            ir += efr * ki_r * dt
            ir = im_r if ir > im_r else (-im_r if ir < -im_r else ir)
        pr_ = efr * kp_r
        dr_o = dfr * kd_r
        ffr_o = tfr * kff_r + (tfr - tl) / dt * kdff_r
        out_r = pr_ + dr_o + ir + ffr_o

        tl = tfp
        el = efp
        tfp += aT_p * (rtp - tfp)
        efp += aE_p * ((tfp - gmp) - efp)
        dfp += aD_p * ((efp - el) / dt - dfp)
        if landed:
            ip = 0.0
        elif ki_p and (not lim_p or (ip > 0 > efp) or (ip < 0 < efp)):
            ip += efp * ki_p * dt
            ip = im_p if ip > im_p else (-im_p if ip < -im_p else ip)
        pp_ = efp * kp_p
        dp_o = dfp * kd_p
        ffp_o = tfp * kff_p + (tfp - tl) / dt * kdff_p
        out_p = pp_ + dp_o + ip + ffp_o

        tl = tfy
        el = efy
        tfy += aT_y * (rty - tfy)
        efy += aE_y * ((tfy - gmy) - efy)
        dfy += aD_y * ((efy - el) / dt - dfy)
        if landed:
            iy = 0.0
        elif ki_y and (not lim_y or (iy > 0 > efy) or (iy < 0 < efy)):
            iy += efy * ki_y * dt
            iy = im_y if iy > im_y else (-im_y if iy < -im_y else iy)
        py_ = efy * kp_y
        dy_o = dfy * kd_y
        ffy_o = tfy * kff_y + (tfy - tl) / dt * kdff_y
        out_y = py_ + dy_o + iy + ffy_o

        # --- 6. 油門:高度 PD + 積分 → 傾角補償(ANGLE_BOOST) ---
        ct = cos(ar) * cos(ap)
        zt = zt_a[k]
        vzt = vzt_a[k]
        a_cmd = 2.2 * (zt - z) + 3.0 * (vzt - vz) + izz
        thr_in = hover * (1.0 + a_cmd / G)
        thr_in = 0.0 if thr_in < 0.0 else (1.0 if thr_in > 1.0 else thr_in)
        cb = 0.5 if ct < 0.5 else ct
        thr = thr_in / cb
        thr = 1.0 if thr > 1.0 else thr
        if airborne:
            izz += 0.4 * (zt - z) * dt
            izz = 2.0 if izz > 2.0 else (-2.0 if izz < -2.0 else izz)
        else:
            izz = 0.0

        # --- 7. 混控器(AP_MotorsMatrix::output_armed_stabilizing) ---
        flags = _F_LANDED if landed else 0
        if stage == 0:
            Re = Pe = Ye = 0.0
            thr_out = 0.0
            tam = 0.0
            lim_r = lim_p = lim_y = False
        else:
            lim_r = lim_p = lim_y = False
            if thr <= 0.0:
                thr = 0.0
                flags |= _F_THR_LO
            tam = thr * (1.0 - mix) + hover * mix
            if tam < thr:
                tam = thr
            if tam > 1.0:
                tam = 1.0
            best = 0.5 if tam > 0.5 else tam
            rp_lo = 1.0
            rp_hi = -1.0
            yaw_allowed = 1.0
            for rf, pf, yf in motors:
                v = out_r * rf + out_p * pf
                if v < rp_lo:
                    rp_lo = v
                if v > rp_hi:
                    rp_hi = v
                if out_y * yf > 0:
                    room = (1.0 - (best + v)) / (yf if yf > 0 else -yf)
                else:
                    room = (best + v) / (yf if yf > 0 else -yf)
                if room < yaw_allowed:
                    yaw_allowed = room if room > 0 else 0.0
            if rp_hi - rp_lo > 1.0 or tam < -rp_lo:
                lim_r = lim_p = True
            if yaw_allowed < yaw_hr:
                yaw_allowed = yaw_hr
            yv = out_y
            if yv > yaw_allowed:
                yv = yaw_allowed
                lim_y = True
            elif yv < -yaw_allowed:
                yv = -yaw_allowed
                lim_y = True
            lo = 1.0
            hi = -1.0
            for rf, pf, yf in motors:
                v = out_r * rf + out_p * pf + yv * yf
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
            scale = 1.0
            if hi - lo > 1.0:
                scale = 1.0 / (hi - lo)
            if tam + lo < 0.0:
                sc2 = -tam / lo
                if sc2 < scale:
                    scale = sc2
            hi *= scale
            lo *= scale
            best = -lo
            adj = thr - best
            if scale < 1.0:
                lim_r = lim_p = lim_y = True
                if adj > 0.0:
                    flags |= _F_THR_UP
                adj = 0.0
            elif adj < 0.0:
                adj = 0.0
            elif adj > 1.0 - (best + hi):
                adj = 1.0 - (best + hi)
                flags |= _F_THR_UP
            thr_out = best + adj
            Re = out_r * scale
            Pe = out_p * scale
            Ye = yv * scale
        if lim_r:
            flags |= _F_LIM_R
        if lim_p:
            flags |= _F_LIM_P
        if lim_y:
            flags |= _F_LIM_Y

        # --- 8. 馬達延遲 ---
        umr += am * (Re - umr)
        ump += am * (Pe - ump)
        umy += am * (Ye - umy)
        umt += am * (thr_out - umt)

        # --- 9. 機體動態 ---
        az = G * (umt * ct / hover - 1.0) - 0.3 * vz
        if airborne:
            wr += dt * (K_r * umr - c_r * wr + dsr[k] + cg_r)
            wp += dt * (K_p * ump - c_p * wp + dsp[k] + cg_p)
            wy += dt * (K_y * umy - c_y * wy + dsy[k])
            ar += dt * wr
            ap += dt * wp
            ay += dt * wy
            vz += dt * az
            z += dt * vz
            cy, sy = cos(ay), sin(ay)
            afw = -G * tan(ap)
            arg = G * tan(ar)
            vn += dt * (afw * cy - arg * sy - 0.35 * vn)
            ve += dt * (afw * sy + arg * cy - 0.35 * ve)
            pn += dt * vn
            pe += dt * ve
            if z <= 0.0 and vz < 0.0:          # 著地
                z = 0.0
                vz = 0.0
                airborne = False
                landed = True
        else:
            # 地面:角速度歸零、姿態回到停機坪傾角(落地瞬間約 0.1 s 收斂)
            wr = wp = 0.0
            wy *= 0.9
            ar += (g_r - ar) * 0.06
            ap += (g_p - ap) * 0.06
            ay += dt * wy
            vn = ve = 0.0
            if stage == 1 and az > 0.05:
                airborne = True
                landed = False
                vz = dt * az
                z = dt * vz
            else:
                az = 0.0 if az < 0.0 else az
                z = 0.0
                vz = 0.0

        push((atr, atp, aty, ar, ap, ay, rtr, rtp, rty, gmr, gmp, gmy, wr, wp, wy,
              tfr, efr, pr_, ir, dr_o, ffr_o,
              tfp, efp, pp_, ip, dp_o, ffp_o,
              tfy, efy, py_, iy, dy_o, ffy_o,
              Re, Pe, Ye, thr_out, flags, thr_in, thr, tam,
              z, vz, zt, vzt, pn, pe, vn, ve, dr_, dp_, az, umt, a_cmd))

    out = np.array(rec, dtype=np.float64) if rec else np.zeros((0, len(_REC)))
    return out


# ======================================================================
# 駕駛員操作與時間軸
# ======================================================================

def _timeline(duration_s: float) -> dict:
    """相對 T_BOOT 的關鍵時刻(秒)。"""
    d = float(duration_s)
    t_arm = 6.0
    t_spool = t_arm + 2.0
    t_climb = t_spool + 0.6
    alt = 5.0
    climb_s = 4.0
    t_althold = t_climb + climb_s + 0.5
    t_disarm = d - 5.0
    t_touch_plan = t_disarm - 3.5
    descend_s = 6.0
    t_land = t_touch_plan - descend_s
    t_man0 = t_althold + 2.0
    t_man1 = t_land - 1.5
    return dict(t_arm=t_arm, t_spool=t_spool, t_climb=t_climb, alt=alt, climb_s=climb_s,
                t_althold=t_althold, t_land=t_land, descend_s=descend_s,
                t_disarm=t_disarm, t_man0=t_man0, t_man1=t_man1, duration=d)


def _pilot_program(rng, tl: dict, style: str, n: int, fs: float) -> tuple[dict, list, list]:
    """駕駛員操作:roll / pitch 對偶脈衝(doublet)、步階、偏航角速度步階,中間穿插安靜懸停。

    回傳 (陣列 dict, 操作清單 [(t0, t1, 種類, 軸, 幅度)], 懸停窗 [(t0, t1)]);時間為相對 T_BOOT 秒。
    """
    t = np.arange(n) / fs
    roll = np.zeros(n)
    pitch = np.zeros(n)
    yawr = np.zeros(n)
    events: list[tuple] = []
    aggressive = style == "aggressive"
    yaw_heavy = style == "yaw"
    hover_only = style == "hover"          # 只懸停:搖桿只有小幅手抖(期望與自激震盪高度同調的最難情況)
    tt = tl["t_man0"]
    t_end = tl["t_man1"]
    hover_windows: list[tuple[float, float]] = []
    n_man = 0

    def put(arr, t0, t1, val):
        i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
        arr[max(i0, 0):max(min(i1, n), 0)] += val

    # 開頭留一段長懸停(PSD 用)
    hover_windows.append((tl["t_man0"] - 1.5, tl["t_man0"] + 4.0))
    tt += 4.0
    if hover_only:
        tt = t_end                          # 不做任何操作,整段都是懸停
    while tt < t_end - 2.5:
        if aggressive:
            kinds, w = ["combo", "doublet", "yaw"], [0.5, 0.35, 0.15]
        elif yaw_heavy:
            kinds, w = ["doublet", "step", "yaw"], [0.25, 0.2, 0.55]
        else:
            kinds, w = ["doublet", "step", "yaw", "combo"], [0.38, 0.32, 0.2, 0.1]
        kind = str(rng.choice(kinds, p=w))
        axis = "roll" if rng.random() < 0.5 else "pitch"
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if kind == "doublet":
            amp = rng.uniform(22, 30) if aggressive else rng.uniform(10, 25)
            h = rng.uniform(0.35, 0.55) if aggressive else rng.uniform(0.5, 0.9)
            arr = roll if axis == "roll" else pitch
            put(arr, tt, tt + h, sign * amp)
            put(arr, tt + h, tt + 2 * h, -sign * amp)
            dur = 2 * h
        elif kind == "step":
            amp = rng.uniform(10, 25)
            h = rng.uniform(1.0, 1.8)
            arr = roll if axis == "roll" else pitch
            put(arr, tt, tt + h, sign * amp)
            put(arr, tt + h, tt + 1.6 * h, -0.6 * sign * amp)   # 反向煞停,速度回到約 0
            dur = 1.6 * h
        elif kind == "yaw":
            rate = rng.uniform(120, 180) if aggressive else rng.uniform(40, 90)
            h = rng.uniform(1.0, 2.2)
            put(yawr, tt, tt + h, sign * rate)
            dur = h
            axis = "yaw"
        else:   # combo:roll + pitch 同時(激烈時再加偏航)
            amp = rng.uniform(24, 30) if aggressive else rng.uniform(10, 18)
            h = rng.uniform(0.35, 0.55) if aggressive else rng.uniform(0.6, 0.9)
            s2 = 1.0 if rng.random() < 0.5 else -1.0
            put(roll, tt, tt + h, sign * amp)
            put(roll, tt + h, tt + 2 * h, -sign * amp)
            put(pitch, tt, tt + h, s2 * amp)
            put(pitch, tt + h, tt + 2 * h, -s2 * amp)
            if aggressive:
                put(yawr, tt, tt + 2 * h, (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(120, 180))
            dur = 2 * h
            axis = "roll+pitch"
            amp = float(amp)
        events.append((tt, tt + dur, kind, axis, float(sign * (amp if kind != "yaw" else rate))))
        n_man += 1
        quiet = rng.uniform(0.8, 1.8) if aggressive else rng.uniform(1.6, 3.2)
        if n_man % 5 == 0:
            quiet = rng.uniform(4.5, 5.5)          # 定期的長懸停(頻譜 / 偏差分析用)
        if quiet >= 3.0:
            hover_windows.append((tt + dur + 1.0, tt + dur + quiet))
        tt += dur + quiet
    # 降落前也是懸停
    if hover_only:
        hover_windows = [(tl["t_man0"] - 1.5, tl["t_land"])]
    else:
        hover_windows.append((tt + 1.0, tl["t_land"]))
    hover_windows = [(a, b) for a, b in hover_windows if b - a >= 1.5]

    # 搖桿有約 60 ms 的推桿時間 + 小幅手抖
    w = int(0.06 * fs) | 1
    roll = _smooth(roll, w)
    pitch = _smooth(pitch, w)
    yawr = _smooth(yawr, int(0.1 * fs) | 1)
    flying = (t >= tl["t_climb"]) & (t <= tl["t_land"] + tl["descend_s"])
    roll += _gm(rng, n, fs, 0.6, 0.4) * flying
    pitch += _gm(rng, n, fs, 0.6, 0.4) * flying
    yawr += _gm(rng, n, fs, 0.8, 1.5) * flying

    # 高度:爬升 → 懸停(緩慢起伏)→ 下降;著地後駕駛員持續拉低油門
    alt = tl["alt"]
    zt = np.full(n, -0.6)
    s = np.clip((t - tl["t_climb"]) / tl["climb_s"], 0.0, 1.0)
    up = (t >= tl["t_climb"])
    zt = np.where(up, -0.6 + (alt + 0.6) * (3 * s ** 2 - 2 * s ** 3), zt)
    cruise = (t >= tl["t_climb"] + tl["climb_s"]) & (t < tl["t_land"])
    zt = np.where(cruise, alt + 0.4 * np.sin(2 * np.pi * (t - tl["t_climb"]) / 37.0), zt)
    sd = np.clip((t - tl["t_land"]) / tl["descend_s"], 0.0, 1.0)
    z_land0 = alt + 0.4 * np.sin(2 * np.pi * (tl["t_land"] - tl["t_climb"]) / 37.0)
    down = t >= tl["t_land"]
    zt = np.where(down, z_land0 - (z_land0 + 0.8) * sd, zt)
    after = t >= tl["t_land"] + tl["descend_s"]
    zt = np.where(after, -0.8 - 0.5 * (t - tl["t_land"] - tl["descend_s"]), zt)
    zt = np.maximum(zt, -2.0)
    vzt = np.gradient(zt, 1.0 / fs)
    vzt = _smooth(vzt, int(0.2 * fs) | 1)
    stage = (t >= tl["t_spool"]).astype(np.int8)

    prog = {"des_r": roll * _D2R, "des_p": pitch * _D2R, "des_yr": yawr * _D2R,
            "zt": zt, "vzt": vzt, "stage": stage}
    return prog, events, hover_windows


# ======================================================================
# 感測器雜訊(2 kHz 生成)
# ======================================================================

def _motor_freq(t_s: np.ndarray, tl: dict, f_hover: float, rng) -> np.ndarray:
    """各時刻的馬達基頻(Hz;未轉 = 0、怠速約 35 %、飛行 ≈ 懸停值 ± 小幅變動)。"""
    f = np.zeros(t_s.size)
    idle = (t_s >= tl["t_arm"]) & (t_s < tl["t_spool"])
    fly = (t_s >= tl["t_spool"]) & (t_s < tl["t_disarm"])
    f[idle] = 0.35 * f_hover
    ramp = np.clip((t_s - tl["t_spool"]) / 1.2, 0.0, 1.0)
    mod = 1.0 + 0.04 * _gm(rng, t_s.size, SENSOR_HZ, 1.5, 1.0)
    f[fly] = (0.45 + 0.55 * ramp[fly]) * f_hover * mod[fly]
    return f


def _sensor_noise(rng, tl: dict, n_s: int, level: dict, f_hover: float, frame: str):
    """原始陀螺儀雜訊(rad/s,3 軸)與加速度計振動(m/s²,3 軸),2 kHz。"""
    fs = SENSOR_HZ
    t_s = np.arange(n_s) / fs
    f_m = _motor_freq(t_s, tl, f_hover, rng)
    spin = (f_m > 0).astype(float)
    # 振幅隨轉速平方
    amp_scale = np.where(f_m > 0, (f_m / f_hover) ** 2, 0.0)
    n_mot = len(_FRAMES[frame][2])
    gyro = np.zeros((n_s, 3))
    acc = np.zeros((n_s, 3))
    harm = np.zeros(n_s)
    harm2 = np.zeros(n_s)
    for _ in range(n_mot):
        detune = 1.0 + rng.uniform(-0.015, 0.015)
        ph = 2.0 * np.pi * np.cumsum(f_m * detune) / fs + rng.uniform(0, 2 * np.pi)
        harm += np.sin(ph) / math.sqrt(n_mot)
        harm2 += np.sin(2.0 * ph + rng.uniform(0, 2 * np.pi)) / math.sqrt(n_mot)
    for j in range(3):
        mix = rng.uniform(0.6, 1.0)
        g = (level["gyro_motor"] * harm * mix + level["gyro_motor2"] * harm2 * mix) * amp_scale
        g += _bandnoise(rng, n_s, fs, 30.0, 0.45 * fs, level["gyro_broad"]) * np.sqrt(amp_scale)
        g += rng.normal(0.0, level["gyro_white"], n_s) * (0.3 + 0.7 * spin)
        gyro[:, j] = g * _D2R
        zg = level["acc_z_gain"] if j == 2 else 1.0
        a = (level["acc_motor"] * harm * rng.uniform(0.7, 1.0)
             + 0.4 * level["acc_motor"] * harm2) * amp_scale * zg
        a += _bandnoise(rng, n_s, fs, 25.0, 0.45 * fs, level["acc_broad"] * zg) * np.sqrt(amp_scale)
        if level["acc_spike"] > 0:
            # 重尾(Laplace)衝擊:螺旋槳不平衡 / 機架共振造成的尖峰 → clipping
            a += rng.laplace(0.0, level["acc_spike"] * zg / math.sqrt(2), n_s) * amp_scale
        a += rng.normal(0.0, 0.05, n_s)
        acc[:, j] = a
    return gyro, acc, f_m


def _gyro_tone(rng, tl: dict, n: int, f0: float, rms_dps: float) -> np.ndarray:
    """濾波後陀螺儀上的窄頻噪音(rad/s,n × 3,LOOP_HZ):中心 f0、±1 % 頻率漂移,只在馬達轉動時。

    Roll / Pitch 為 rms_dps、Yaw 一半;控制器看到的就是這個(所以也會進到 D 項)。
    f0 不要是記錄率的整數倍(例 108 Hz:100 Hz 記錄混疊到 8 Hz、50 Hz 也是 8 Hz)。
    """
    if not 0 < f0 < LOOP_HZ / 2:
        raise ValueError(f"tone 頻率需在 0–{LOOP_HZ / 2:g} Hz")
    t = np.arange(n) / LOOP_HZ
    spin = ((t >= tl["t_spool"]) & (t < tl["t_disarm"])).astype(float)
    out = np.zeros((n, 3))
    for j, k in enumerate((1.0, 1.0, 0.5)):
        fm = f0 * (1.0 + 0.01 * _gm(rng, n, LOOP_HZ, 1.5, 1.0))
        ph = 2.0 * np.pi * np.cumsum(fm) / LOOP_HZ + rng.uniform(0, 2 * np.pi)
        out[:, j] = k * rms_dps * math.sqrt(2.0) * np.sin(ph) * spin * _D2R
    return out


def _drop_rows(tables: dict, frac: float, rng) -> None:
    """就地隨機丟掉 ATT / RATE / PIDx 的 frac 比例列(SD 卡忙時的掉訊息)。"""
    for name in ("ATT", "RATE", "PIDR", "PIDP", "PIDY"):
        tb = tables.get(name)
        if tb is None:
            continue
        keep = rng.random(tb["t"].size) >= frac
        tables[name] = {c: v[keep] for c, v in tb.items()}


def _legacy_pwm_parm(tables: dict, prm: dict) -> None:
    """PARM 改成 ArduCopter 4.0 / 4.1 的預設:MOT_PWM_MIN / MAX = 0,輸出範圍由 RC3_MIN / MAX 決定。"""
    p = tables["PARM"]
    names = list(p["Name"])
    vals = list(np.asarray(p["Value"], float))
    dflt = list(np.asarray(p["Default"], float))
    lo, hi = float(prm["MOT_PWM_MIN"]), float(prm["MOT_PWM_MAX"])
    for k in ("MOT_PWM_MIN", "MOT_PWM_MAX"):
        i = names.index(k)
        vals[i] = dflt[i] = 0.0
    for k, v in (("RC3_MAX", hi), ("RC3_MIN", lo)):
        if k not in names:
            names.append(k)
            vals.append(v)
            dflt.append(1900.0 if k == "RC3_MAX" else 1100.0)
    order = np.argsort(np.array(names))
    t0 = float(p["t"][0])
    tables["PARM"] = {"t": t0 + 0.0002 * np.arange(len(names)), "Name": np.array(names, dtype=str)[order],
                      "Value": _f32(np.array(vals)[order]), "Default": _f32(np.array(dflt)[order])}


# ======================================================================
# 主函式
# ======================================================================

def _ground_tilt(rng) -> tuple[float, float]:
    return rng.uniform(0.3, 0.9) * _D2R, -rng.uniform(0.5, 1.1) * _D2R


def make_tuning_sim(scenario: str = "good", *, seed: int = 1, duration_s: float = 120.0,
                    log_hz: float = 100.0, fast_log: bool = True, pid_log: bool | None = None,
                    batch: bool | None = None, frame: str = "quad",
                    overrides: dict[str, float] | None = None, pilot: str | None = None,
                    tone: tuple[float, float] | None = None, drop_frac: float = 0.0,
                    legacy_pwm: bool = False) -> tuple[LogData, dict]:
    """模擬一次 ArduCopter 飛行(地面 → ARM → 起飛 → 操作 → 降落 → DISARM → 地面)。

    scenario:SCENARIOS 之一。fast_log:True → ATT / RATE(與 PIDx)以 log_hz 記錄(LOG_BITMASK
    含 ATTITUDE_FAST;log_hz < 400 時同 LOG_FILE_RATEMAX 以「取樣」抽取,混疊照實發生);
    False → 仿使用者的 log(10 Hz、LOG_BITMASK=176126)。
    pid_log:None = 同 fast_log(fast_log=False 時沒有 PIDx,與使用者的 log 相同);True 且
    fast_log=False → PIDx 以 10 Hz 記錄(ArduPilot 預設 LOG_BITMASK=180222 的行為)。
    batch:None = 情境預設(noisy_notch 開,其他關);True → ISBH / ISBD(INS_LOG_BAT_MASK=1)。
    overrides:覆寫參數(例如套用調參建議後重飛,驗證是否改善)。
    pilot:操作風格覆寫("normal" / "aggressive" / "yaw" / "hover";hover = 只懸停、搖桿只有手抖)。
    tone:(Hz, deg/s RMS)—— 濾波「後」陀螺儀上的窄頻馬達噪音(±1 % 頻率漂移;控制器也看得到),
          例 (108, 4.5):100 / 50 Hz 記錄時混疊到 8 Hz(馬達噪音假冒控制震盪的測試)。
    drop_frac:隨機丟掉這個比例的 ATT / RATE / PIDx 列(SD 卡忙時掉訊息)。
    legacy_pwm:PARM 寫成 ArduCopter 4.0 / 4.1 的 MOT_PWM_MIN / MAX = 0(輸出範圍改用 RC3_MIN / MAX)。

    回傳 (LogData, info)。LogData 由 logio.logdata_from_index 組成(與 load_bin 相同路徑)。
    info(時間皆為 log 絕對秒):
      expected / secondary  [(參數或檢查標籤, DIR_*)]:主要 / 次要修正方向;DIR_CHECK 的標籤為
                            "motor_headroom"、"vibration"、"cg_balance"
      forbid                不該出現的建議(例:馬達飽和時調高增益)
      expected_codes        建議的 Finding.code;expect_ok:只有 good 為 True
      expected_logging      [(參數, DIR_SET, 值)]:應建議的記錄設定(LOG_BITMASK 確切值、INS_LOG_BAT_MASK)
      slow_log              {"detectable": "yes"|"partial"|"no", "note"}:只有 10 Hz log 時看不看得出
      osc_hz                震盪情境的真實頻率(懸停時真實角速度頻譜峰,roll / pitch 平均);其他 None
      osc[axis]             {"hz", "peak_ratio", "band_rms"}(所有情境都有,供比較)
      noise_hz              noisy_notch 的馬達雜訊頻率;motor_hz_hover / motor_hz_range
      step[axis] / angle_step[axis]   無雜訊步階實驗:overshoot_pct、rise_s、settle_s、peak_s
      metrics[axis]         懸停角速度 RMS、操作時角速度 / 角度誤差 RMS、懸停角度偏差、延遲 ms、
                            控制輸出高頻 RMS、P / D 項 RMS
      saturation            at_max_pct / at_min_pct(空中、400 Hz)、pwm_top / pwm_bottom
      vibe                  空中 VIBE 平均 / 最大、clip_total
      params / changed      飛行參數 / 本情境改過的參數 (值, 預設值)
      times / windows / manoeuvres    ARM、起飛、著地、DISARM;懸停窗、操作窗、飛行段
      plant / noise_level / log / runtime_s   校準常數、雜訊等級、記錄設定與實際速率、耗時
      truth                 400 Hz 真值:t、rate / rate_tgt / gyro / ang / ang_tgt(N×3,deg)、airborne
    """
    t_run0 = time.perf_counter()
    if scenario not in _SPEC:
        raise ValueError(f"未知情境 {scenario!r};可用:{', '.join(SCENARIOS)}")
    if duration_s < 40.0:
        raise ValueError("duration_s 至少 40 秒(地面 + 起降 + 操作)")
    spec = _SPEC[scenario]
    if pid_log is None:
        pid_log = bool(fast_log)
    if batch is None:
        batch = bool(spec.get("batch", False))
    stride = max(1, int(round(LOOP_HZ / max(float(log_hz), 1.0))))
    fast_hz = LOOP_HZ / stride

    prm = scenario_params(scenario, frame, overrides)
    prm["LOG_BITMASK"] = float(log_bitmask(fast_log, pid_log))
    prm["INS_LOG_BAT_MASK"] = 1.0 if batch else 0.0
    # 快速記錄低於迴圈頻率 = 以 LOG_FILE_RATEMAX 限制串流訊息(ATT / RATE / PIDx)的速率
    prm["LOG_FILE_RATEMAX"] = float(fast_hz) if fast_log and stride > 1 else 0.0
    hover = float(prm["MOT_THST_HOVER"])

    plant = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PLANT.items()}
    ks = float(spec.get("k_scale", 1.0))
    plant["K"] = {ax: v * ks for ax, v in PLANT["K"].items()}
    cg_spec = spec.get("cg_offset", PLANT["cg_offset"])
    f_hover = float(spec.get("motor_hz", PLANT["motor_hz_hover"]))
    level = _NOISE[spec.get("noise", "normal")]

    crc = zlib.crc32(scenario.encode())
    rng_prog = np.random.default_rng([seed, crc, 1])
    rng_dist = np.random.default_rng([seed, crc, 2])
    rng_sens = np.random.default_rng([seed, crc, 3])
    rng_log = np.random.default_rng([seed, crc, 4])

    tl = _timeline(duration_s)
    dt = 1.0 / LOOP_HZ
    n_all = int(round(duration_s * LOOP_HZ))
    k_arm = int(round(tl["t_arm"] * LOOP_HZ))
    k_dis = int(round(tl["t_disarm"] * LOOP_HZ))
    n_loop = k_dis - k_arm

    style = pilot if pilot is not None else spec.get("pilot", "normal")
    if style not in ("normal", "aggressive", "yaw", "hover"):
        raise ValueError(f"未知操作風格 {style!r};可用:normal、aggressive、yaw、hover")
    prog_all, man_events, hover_windows = _pilot_program(rng_prog, tl, style, n_all, LOOP_HZ)

    # --- 感測器雜訊(2 kHz)→ 經陀螺儀低通(+ 陷波)→ 抽到 400 Hz ---
    dec = int(round(SENSOR_HZ / LOOP_HZ))
    n_s = n_all * dec
    gyro_raw_n, acc_raw_n, f_motor = _sensor_noise(rng_sens, tl, n_s, level, f_hover, frame)
    lpf_s = _lpf2_coef(SENSOR_HZ, float(prm["INS_GYRO_FILTER"]))
    nz_full = np.zeros((n_all, 3))
    for j in range(3):
        y = _biquad(gyro_raw_n[:, j], lpf_s)
        for nc in _notch_list(prm, SENSOR_HZ, hover):
            y = _biquad(y, nc)
        nz_full[:, j] = y[::dec]
    if tone is not None:
        nz_full += _gyro_tone(np.random.default_rng([seed, crc, 5]), tl, n_all, float(tone[0]), float(tone[1]))

    # --- 擾動力矩:亂流(GM)+ 槳流 / 推力漣波(5~60 Hz)---
    dist_full = np.zeros((n_all, 3))
    for j, ax in enumerate(("roll", "pitch", "yaw")):
        dist_full[:, j] = (_gm(rng_dist, n_all, LOOP_HZ, 0.3, plant["turb_sigma"][ax])
                           + _bandnoise(rng_dist, n_all, LOOP_HZ, 5.0, 60.0, plant["torque_noise"][ax]))
    if spec.get("noise") == "vibration":
        dist_full *= 1.3

    g_r, g_p = _ground_tilt(rng_dist)
    yaw0 = math.radians(rng_dist.uniform(20.0, 340.0))

    cfg = _ctrl_cfg(prm, frame, plant, hover)
    sl = slice(k_arm, k_dis)
    prog = {k: (v[sl].tolist() if isinstance(v, np.ndarray) else v) for k, v in prog_all.items()}
    nz = {"r": nz_full[sl, 0].tolist(), "p": nz_full[sl, 1].tolist(), "y": nz_full[sl, 2].tolist()}
    dist = {"r": dist_full[sl, 0].tolist(), "p": dist_full[sl, 1].tolist(),
            "y": dist_full[sl, 2].tolist()}
    init = dict(ar=g_r, ap=g_p, ay=yaw0, ground_r=g_r, ground_p=g_p, airborne=False)
    t_loop0 = time.perf_counter()
    rec = _simulate(cfg, prog, nz, dist, (cg_spec.get("roll", 0.0), cg_spec.get("pitch", 0.0)),
                    n_loop, init)
    t_loop = time.perf_counter() - t_loop0

    # --- 組回整段時間(地面 / 上鎖期間用靜止值) ---
    full = np.zeros((n_all, len(_REC)))
    I = _RI
    full[:, I["ar"]] = g_r
    full[:, I["ap"]] = g_p
    full[:, I["ay"]] = yaw0
    full[:, I["aty"]] = yaw0
    full[:, I["flags"]] = _F_LANDED
    full[sl] = rec
    full[k_dis:, I["ar"]] = rec[-1, I["ar"]]
    full[k_dis:, I["ap"]] = rec[-1, I["ap"]]
    full[k_dis:, I["ay"]] = rec[-1, I["ay"]]
    full[k_dis:, I["aty"]] = rec[-1, I["ay"]]
    full[k_dis:, I["pn"]] = rec[-1, I["pn"]]
    full[k_dis:, I["pe"]] = rec[-1, I["pe"]]
    # 上鎖時的陀螺儀 = 雜訊(未轉時很小)
    for j, key in enumerate(("gmr", "gmp", "gmy")):
        m = np.ones(n_all, bool)
        m[sl] = False
        full[m, I[key]] = nz_full[m, j]
        full[m, I["ef" + key[-1]]] = -nz_full[m, j]
        full[m, I["p" + key[-1]]] = -nz_full[m, j] * cfg[("roll", "pitch", "yaw")[j]]["kp"]

    t_rel = np.arange(n_all) * dt
    t_abs = T_BOOT + t_rel
    flags = full[:, I["flags"]].astype(np.int64)
    landed = (flags & _F_LANDED) != 0
    armed = np.zeros(n_all, bool)
    armed[sl] = True
    airborne = armed & ~landed
    if not airborne.any():
        raise RuntimeError("模擬的飛機沒有起飛(檢查 MOT_THST_HOVER / 油門模型)")
    k_lift = int(np.argmax(airborne))
    k_touch = int(n_all - 1 - np.argmax(airborne[::-1])) + 1
    ev_t = dict(arm=tl["t_arm"], auto_armed=tl["t_spool"] + 0.25, liftoff=k_lift * dt,
                touch=k_touch * dt, disarm=tl["t_disarm"])

    # --- 每個馬達的推力命令 / PWM(400 Hz) ---
    motors = cfg["motors"]
    R, P, Y, thr_out = (full[:, I[k]] for k in ("R", "P", "Y", "thr_out"))
    cmd = np.column_stack([thr_out + rf * R + pf * P + yf * Y for rf, pf, yf in motors])
    pwm = _thrust_to_pwm(cmd, prm)
    spin_arm_pwm = prm["MOT_PWM_MIN"] + (prm["MOT_PWM_MAX"] - prm["MOT_PWM_MIN"]) * prm["MOT_SPIN_ARM"]
    idle = armed & (prog_all["stage"] == 0)
    pwm[idle] = spin_arm_pwm
    spool = armed & (t_rel >= tl["t_spool"]) & (t_rel < tl["t_spool"] + prm["MOT_SPOOL_TIME"])
    if spool.any():
        frac = (t_rel[spool] - tl["t_spool"]) / prm["MOT_SPOOL_TIME"]
        lo_pwm = _thrust_to_pwm(np.zeros(1), prm)[0]
        ramp = spin_arm_pwm + (lo_pwm - spin_arm_pwm) * frac
        pwm[spool] = np.maximum(pwm[spool] * frac[:, None] + ramp[:, None] * (1 - frac[:, None]),
                                ramp[:, None])
    pwm[~armed] = prm["MOT_PWM_MIN"]
    pwm = np.rint(pwm)
    pwm_top = np.rint(_thrust_to_pwm(np.ones(1), prm)[0])
    pwm_bot = np.rint(_thrust_to_pwm(np.zeros(1), prm)[0])
    at_max = (pwm >= pwm_top - 1).any(axis=1) & airborne
    at_min = (pwm <= pwm_bot + 1).any(axis=1) & airborne

    # --- 原始加速度(比力 + 振動)、VIBE、clipping ---
    acc_body = np.zeros((n_all, 3))
    acc_body[:, 2] = -G
    acc_body[:, 0] = G * np.sin(full[:, I["ap"]])
    acc_body[:, 1] = -G * np.sin(full[:, I["ar"]]) * np.cos(full[:, I["ap"]])
    fly_m = airborne
    acc_body[fly_m, 0] = 0.0
    acc_body[fly_m, 1] = 0.0
    acc_body[fly_m, 2] = -(full[fly_m, I["az"]] + G) / np.maximum(
        np.cos(full[fly_m, I["ar"]]) * np.cos(full[fly_m, I["ap"]]), 0.5)
    acc_raw = np.repeat(acc_body, dec, axis=0) + acc_raw_n
    clip_lim = 15.5 * G
    clipped = (np.abs(acc_raw) > clip_lim).any(axis=1)
    acc_raw = np.clip(acc_raw, -16.0 * G, 16.0 * G)
    vibe = np.zeros((n_s, 3))
    for j in range(3):
        floor = _lpf1(acc_raw[:, j], SENSOR_HZ, 5.0)
        vibe[:, j] = np.sqrt(np.maximum(_lpf1((acc_raw[:, j] - floor) ** 2, SENSOR_HZ, 2.0), 0.0))
    clip_cum = np.cumsum(clipped)
    acc_lp = np.column_stack([_biquad(acc_raw[:, j], _lpf2_coef(SENSOR_HZ, float(prm["INS_ACCEL_FILTER"])))
                              for j in range(3)])
    true_rate_s = np.column_stack([np.interp(np.arange(n_s) / dec, np.arange(n_all), full[:, I[k]])
                                   for k in ("wr", "wp", "wy")])
    gyro_raw = true_rate_s + gyro_raw_n

    # --- log 表格 ---
    tables, units = _build_tables(
        full=full, t_abs=t_abs, prm=prm, pwm=pwm, rng=rng_log, stride=stride,
        fast_log=fast_log, pid_log=pid_log, batch=batch, frame=frame, tl=tl, ev_t=ev_t,
        vibe=vibe, clip_cum=clip_cum, acc_lp=acc_lp, gyro_raw=gyro_raw, acc_raw=acc_raw,
        dec=dec, armed=armed, hover=hover)
    if drop_frac > 0:
        _drop_rows(tables, float(drop_frac), np.random.default_rng([seed, crc, 6]))
    if legacy_pwm:
        _legacy_pwm_parm(tables, prm)

    from .dflog import LogIndex
    from .logio import logdata_from_index
    path = f"(模擬資料:PID 調參 {scenario})"
    index = LogIndex.from_tables(tables, vehicle="copter", path=path, units=units)
    logdata = logdata_from_index(index, path)
    logdata.notes += [
        f"PID 調參模擬:{scenario} — {SCENARIOS[scenario]}",
        (f"記錄:ATT / RATE {fast_hz:g} Hz" if fast_log else "記錄:ATT / RATE 10 Hz(同使用者的 log)")
        + (f"、PIDR/P/Y {fast_hz if fast_log else 10:g} Hz" if pid_log else "、無 PIDx")
        + (f"、batch {SENSOR_HZ:g} Hz" if batch else "")
        + f";LOG_BITMASK={int(prm['LOG_BITMASK'])}(seed={seed})",
    ]
    extra = [x for x in (f"操作 {style}" if pilot is not None else "",
                         f"陀螺儀噪音 {tone[0]:g} Hz {tone[1]:g} °/s" if tone is not None else "",
                         f"掉訊息 {drop_frac * 100:g} %" if drop_frac > 0 else "",
                         "MOT_PWM_MIN/MAX = 0(4.0 / 4.1)" if legacy_pwm else "") if x]
    if extra:
        logdata.notes.append("模擬選項:" + "、".join(extra))

    info = _make_info(scenario, spec, prm, plant, full, t_abs, airborne, armed, at_max, at_min,
                      man_events, hover_windows, ev_t, vibe, clip_cum, f_motor, f_hover, level, seed,
                      frame, fast_log, pid_log, batch, dec, tables)
    info["options"] = {"pilot": style, "tone": tuple(tone) if tone is not None else None,
                       "drop_frac": float(drop_frac), "legacy_pwm": bool(legacy_pwm)}
    info["runtime_s"] = {"total": time.perf_counter() - t_run0, "loop": t_loop,
                         "loop_steps": int(n_loop)}
    return logdata, info


# ======================================================================
# log 表格
# ======================================================================

SIM_UNITS: dict[str, dict[str, str]] = {
    "ATT": {"DesRoll": "deg", "Roll": "deg", "DesPitch": "deg", "Pitch": "deg",
            "DesYaw": "degheading", "Yaw": "degheading", "ErrRP": "deg", "ErrYaw": "degheading"},
    "RATE": {"RDes": "deg/s", "R": "deg/s", "PDes": "deg/s", "P": "deg/s", "YDes": "deg/s",
             "Y": "deg/s", "ADes": "cm/s/s", "A": "cm/s/s"},
    # ArduPilot 的 PIDx FMTU 全是 '-'(PIDA 等共用格式);這裡註明實際單位方便顯示。
    # 分析時一律當 rad/s,不要依賴單位字串(真實 log 沒有)。
    "PIDR": {"Tar": "rad/s", "Act": "rad/s", "Err": "rad/s"},
    "PIDP": {"Tar": "rad/s", "Act": "rad/s", "Err": "rad/s"},
    "PIDY": {"Tar": "rad/s", "Act": "rad/s", "Err": "rad/s"},
    "CTUN": {"DAlt": "m", "Alt": "m", "BAlt": "m", "DSAlt": "m", "SAlt": "m", "TAlt": "m",
             "DCRt": "cm/s", "CRt": "cm/s"},
    "RCOU": {f"C{i}": "us" for i in range(1, 15)},
    "RCIN": {f"C{i}": "us" for i in range(1, 15)},
    "VIBE": {"VibeX": "m/s/s", "VibeY": "m/s/s", "VibeZ": "m/s/s"},
    "IMU": {"GyrX": "rad/s", "GyrY": "rad/s", "GyrZ": "rad/s", "AccX": "m/s/s", "AccY": "m/s/s",
            "AccZ": "m/s/s", "T": "degC", "GHz": "Hz", "AHz": "Hz"},
    "ISBH": {"SampleUS": "µs", "smp_rate": "Hz"},
    "ISBD": {"x": "m/s/s", "y": "m/s/s", "z": "m/s/s"},
    "BAT": {"Volt": "V", "VoltR": "V", "Curr": "A", "CurrTot": "mAh", "EnrgTot": "W.h",
            "Temp": "degC", "Res": "Ohm", "RemPct": "%"},
    "GPS": {"GMS": "ms", "NSats": "satellites", "Lat": "deglatitude", "Lng": "deglongitude",
            "Alt": "m", "Spd": "m/s", "GCrs": "degheading", "VZ": "m/s", "Yaw": "degheading"},
    "POS": {"Lat": "deglatitude", "Lng": "deglongitude", "Alt": "m", "RelHomeAlt": "m",
            "RelOriginAlt": "m"},
    "ORGN": {"Lat": "deglatitude", "Lng": "deglongitude", "Alt": "m"},
}

# ArduPilot 4.5 的格式字元(write_tuning_bin 用;未列者 float → f、整數依 dtype)
SIM_FORMATS: dict[tuple[str, str], str] = {
    **{("ATT", c): "c" for c in ("DesRoll", "Roll", "DesPitch", "Pitch")},
    **{("ATT", c): "C" for c in ("DesYaw", "Yaw", "ErrRP", "ErrYaw")},
    ("CTUN", "BAlt"): "e", ("BAT", "Temp"): "c", ("GPS", "HDop"): "c", ("GPS", "Alt"): "e",
    ("ORGN", "Alt"): "e", ("MODE", "Mode"): "M", ("MSG", "Message"): "Z", ("PARM", "Name"): "N",
}

_PWM_MODE = {"STABILIZE": 1165, "ALT_HOLD": 1295}


def _log_idx(n_all: int, hz: float, rng, jitter: bool = True) -> np.ndarray:
    """以 hz 從 400 Hz 迴圈取樣的索引(取樣,不平均);低速訊息有 ±1 迴圈的排程抖動。"""
    step = LOOP_HZ / hz
    k = np.arange(0.0, n_all - 1, step)
    k = np.rint(k).astype(np.int64)
    if jitter and step >= 8:
        k = k + rng.choice([-1, 0, 0, 0, 1], k.size)
    k = np.clip(k, 0, n_all - 1)
    return np.unique(k)


def _build_tables(*, full, t_abs, prm, pwm, rng, stride, fast_log, pid_log, batch, frame,
                  tl, ev_t, vibe, clip_cum, acc_lp, gyro_raw, acc_raw, dec, armed, hover):
    from .fakebin import COPTER_MODES
    I = _RI
    n_all = full.shape[0]
    tb = T_BOOT
    dt = 1.0 / LOOP_HZ
    tables: dict[str, dict[str, np.ndarray]] = {}

    # --- MSG(版本放第一筆:DFReader / LogIndex 靠它判斷機型)---
    frame_txt = _FRAMES[frame][3]
    msgs = [(tb + 0.000, "ArduCopter V4.5.7 (sim)"), (tb + 0.001, "ChibiOS: sim"),
            (tb + 0.002, "SimBoard 00000000 00000000 00000000"),
            (tb + 0.003, "RCOut: PWM:1-14"),
            (tb + 0.004, f"IMU0: fast sampling enabled {SENSOR_HZ / 1000:.1f}kHz"),
            (tb + 0.005, f"Frame: {frame_txt}"),
            (tb + 0.62, "EKF3 IMU0 initialised"), (tb + 1.10, "EKF3 IMU0 tilt alignment complete"),
            (tb + 1.50, "EKF3 IMU0 MAG0 initial yaw alignment complete"),
            (tb + 2.00, "EKF3 IMU0 origin set"), (tb + 2.40, "EKF3 IMU0 is using GPS")]
    tables["MSG"] = {"t": np.array([m[0] for m in msgs]),
                     "Message": np.array([m[1] for m in msgs], dtype=str)}

    # --- PARM(開機時一次寫出,含 Default)---
    dflt = default_params(frame)
    names = sorted(k for k in prm if not (k.startswith("INS_HNTCH_") and k != "INS_HNTCH_ENABLE"
                                          and prm.get("INS_HNTCH_ENABLE", 0.0) <= 0))
    tables["PARM"] = {"t": tb + 0.010 + 0.0002 * np.arange(len(names)),
                      "Name": np.array(names, dtype=str),
                      "Value": _f32([prm[k] for k in names]),
                      "Default": _f32([dflt.get(k, prm[k]) for k in names])}

    # --- MODE / ARM / EV ---
    t_alt = tb + tl["t_althold"]
    tables["MODE"] = {"t": np.array([tb + 0.006, t_alt]),
                      "Mode": np.array([COPTER_MODES["STABILIZE"], COPTER_MODES["ALT_HOLD"]], np.int8),
                      "ModeNum": np.array([COPTER_MODES["STABILIZE"], COPTER_MODES["ALT_HOLD"]], np.uint8),
                      "Rsn": np.array([0, 1], np.uint8)}
    t_arm, t_dis = tb + ev_t["arm"], tb + ev_t["disarm"]
    tables["ARM"] = {"t": np.array([t_arm, t_dis]), "ArmState": np.array([1, 0], np.uint8),
                     "ArmChecks": np.array([1, 1], np.uint32), "Forced": np.zeros(2, np.uint8),
                     "Method": np.array([0, 0], np.uint8)}          # 方向舵解鎖 / 上鎖
    evs = [(t_arm, 10), (tb + ev_t["auto_armed"], 15), (tb + ev_t["liftoff"] + 0.05, 28),
           (tb + ev_t["touch"] + 0.3, 17), (tb + ev_t["touch"] + 1.0, 18), (t_dis, 11)]
    tables["EV"] = {"t": np.array([e[0] for e in evs]), "Id": np.array([e[1] for e in evs], np.uint8)}

    # --- 10 Hz 共用索引(同一排程工作:ATT/RATE(慢)、CTRL、RCOU、CTUN、MOTB、VIBE…)---
    k10 = _log_idx(n_all, 10.0, rng)
    k25 = _log_idx(n_all, 25.0, rng)
    kf = np.arange(0, n_all, stride) if fast_log else k10
    kpid = kf if fast_log else k10

    def att_table(k):
        return {
            "t": t_abs[k] + 0.00012,
            "DesRoll": _cd(np.degrees(full[k, I["atr"]])),
            "Roll": _cd(np.degrees(full[k, I["ar"]])),
            "DesPitch": _cd(np.degrees(full[k, I["atp"]])),
            "Pitch": _cd(np.degrees(full[k, I["ap"]])),
            "DesYaw": _cd(np.degrees(full[k, I["aty"]]) % 360.0) % 360.0,
            "Yaw": _cd(np.degrees(full[k, I["ay"]]) % 360.0) % 360.0,
            "ErrRP": _cd(np.where(armed[k], 0.02, 0.01) + 0.01 * (rng.random(k.size) < 0.3)),
            "ErrYaw": _cd(np.where(armed[k], 0.01, 0.0) + 0.01 * (rng.random(k.size) < 0.2)),
            "AEKF": np.full(k.size, 3, np.uint8),
        }

    tables["ATT"] = att_table(kf)

    out_r = full[:, I["pr"]] + full[:, I["ir"]] + full[:, I["dr"]] + full[:, I["ffr"]]
    out_p = full[:, I["pp"]] + full[:, I["ip"]] + full[:, I["dp"]] + full[:, I["ffp"]]
    out_y = full[:, I["py"]] + full[:, I["iy"]] + full[:, I["dy"]] + full[:, I["ffy"]]
    thr = full[:, I["thr"]]
    thr_slew = _lpf1(np.abs(np.gradient(thr, dt)), LOOP_HZ, 5.0)
    k = kf
    a_up = full[:, I["az"]]
    tables["RATE"] = {
        "t": t_abs[k],
        "RDes": _f32(np.degrees(full[k, I["rtr"]])), "R": _f32(np.degrees(full[k, I["gmr"]])),
        "ROut": _f32(out_r[k]),
        "PDes": _f32(np.degrees(full[k, I["rtp"]])), "P": _f32(np.degrees(full[k, I["gmp"]])),
        "POut": _f32(out_p[k]),
        "YDes": _f32(np.degrees(full[k, I["rty"]])), "Y": _f32(np.degrees(full[k, I["gmy"]])),
        "YOut": _f32(out_y[k]),
        "ADes": _f32(full[k, I["a_cmd"]] * 100.0 * armed[k]),
        "A": _f32(a_up[k] * 100.0 + rng.normal(0, 3.0, k.size)),
        "AOut": _f32(thr[k] * armed[k]), "AOutSlew": _f32(thr_slew[k] * armed[k]),
    }

    if pid_log:
        flags = full[:, I["flags"]].astype(np.int64)
        for name, ax, lim_bit in (("PIDR", "r", _F_LIM_R), ("PIDP", "p", _F_LIM_P),
                                  ("PIDY", "y", _F_LIM_Y)):
            pd = full[:, I["p" + ax]] + full[:, I["d" + ax]]
            srate = _lpf1(_lpf1(np.abs(np.diff(pd, prepend=pd[0])) / dt, LOOP_HZ, 10.0), LOOP_HZ, 0.5)
            fl = (np.where((flags & lim_bit) != 0, 1, 0)
                  | np.where((flags & _F_LANDED) != 0, 8, 0)).astype(np.uint8)
            k = kpid
            tables[name] = {
                "t": t_abs[k] + 0.00025,
                "Tar": _f32(full[k, I["tf" + ax]]), "Act": _f32(full[k, I["gm" + ax]]),
                "Err": _f32(full[k, I["ef" + ax]]), "P": _f32(full[k, I["p" + ax]]),
                "I": _f32(full[k, I["i" + ax]]), "D": _f32(full[k, I["d" + ax]]),
                "FF": _f32(full[k, I["ff" + ax]]), "DFF": np.zeros(k.size),
                "Dmod": np.ones(k.size), "SRate": _f32(srate[k]), "Flags": fl[k],
            }

    # --- CTRL:控制器 RMS 監視(每迴圈 rms = 0.99 rms + 0.01 x²,記錄 sqrt)---
    def rms(x):
        return np.sqrt(np.maximum(_iir(x * x, (0.01,), (1.0, -0.99), pad=4000), 0.0))
    k = k10
    tables["CTRL"] = {
        "t": t_abs[k] + 0.0004,
        "RMSRollP": _f32(rms(full[:, I["pr"]] + full[:, I["ffr"]])[k]),
        "RMSRollD": _f32(rms(full[:, I["dr"]])[k]),
        "RMSPitchP": _f32(rms(full[:, I["pp"]] + full[:, I["ffp"]])[k]),
        "RMSPitchD": _f32(rms(full[:, I["dp"]])[k]),
        "RMSYaw": _f32(rms(full[:, I["py"]] + full[:, I["dy"]] + full[:, I["ffy"]])[k]),
    }

    # --- RCOU(14 通道,馬達在前)---
    n_mot = pwm.shape[1]
    rcou = {"t": t_abs[k10] + 0.0005}
    for i in range(14):
        rcou[f"C{i + 1}"] = (pwm[k10, i] if i < n_mot else np.zeros(k10.size)).astype(np.uint16)
    tables["RCOU"] = rcou

    # --- RCIN:搖桿(roll / pitch / throttle / yaw / 模式開關)---
    amax = prm["ANGLE_MAX"] / 100.0
    stick_r = np.degrees(full[k10, I["des_r"]]) / amax
    stick_p = np.degrees(full[k10, I["des_p"]]) / amax
    yr = np.gradient(np.unwrap(full[:, I["aty"]]), dt)
    stick_y = np.degrees(yr[k10]) / prm["PILOT_Y_RATE"]
    thr_st = np.where(armed[k10], 1500.0 + 400.0 * np.clip(full[k10, I["vzt"]] / 2.5, -1, 1), 1000.0)
    mode_pwm = np.where(t_abs[k10] >= t_alt, _PWM_MODE["ALT_HOLD"], _PWM_MODE["STABILIZE"])
    chans = [1500 + 500 * np.clip(stick_r, -1, 1), 1500 - 500 * np.clip(stick_p, -1, 1), thr_st,
             1500 + 500 * np.clip(stick_y, -1, 1), mode_pwm] + [np.full(k10.size, 1000.0)] * 9
    rcin = {"t": t_abs[k10] + 0.0002}
    for i, c in enumerate(chans):
        rcin[f"C{i + 1}"] = np.clip(np.rint(c + rng.normal(0, 0.6, k10.size)), 900, 2100).astype(np.uint16)
    tables["RCIN"] = rcin

    # --- CTUN ---
    z = full[:, I["z"]]
    k = k10
    alt_mode = t_abs[k] >= t_alt
    baro = z[k] + _gm(rng, k.size, 10.0, 20.0, 0.15) + rng.normal(0, 0.05, k.size)
    tables["CTUN"] = {
        "t": t_abs[k] + 0.0006,
        "ThI": _f32(full[k, I["thr_in"]] * armed[k]),
        "ABst": _f32((full[k, I["thr"]] - full[k, I["thr_in"]]) * armed[k]),
        "ThO": _f32(full[k, I["thr"]] * armed[k]),
        "ThH": _f32(np.full(k.size, hover)),
        "DAlt": _f32(np.where(alt_mode & armed[k], full[k, I["zt"]], 0.0)),
        "Alt": _f32(z[k] + rng.normal(0, 0.01, k.size)),
        "BAlt": _cd(baro),
        "DSAlt": np.full(k.size, np.nan),
        "SAlt": np.zeros(k.size),
        "TAlt": _f32(z[k]),
        "DCRt": np.rint(np.where(alt_mode & armed[k], full[k, I["vzt"]], 0.0) * 100).astype(np.int16),
        "CRt": np.rint(full[k, I["vz"]] * 100 + rng.normal(0, 1.0, k.size)).astype(np.int16),
    }

    # --- MOTB ---
    tables["MOTB"] = {
        "t": t_abs[k] + 0.0007,
        "LiftMax": np.ones(k.size), "BatVolt": np.ones(k.size), "ThLimit": np.ones(k.size),
        "ThrAvMx": _f32(full[k, I["thr_avg_max"]]), "ThrOut": _f32(full[k, I["thr_out"]]),
        "FailFlags": np.zeros(k.size, np.uint8),
    }

    # --- BAT[0] ---
    cells = 4 if frame == "quad" else 6
    mot_th = np.mean(np.clip((pwm - prm["MOT_PWM_MIN"]) / (prm["MOT_PWM_MAX"] - prm["MOT_PWM_MIN"]),
                             0, 1), axis=1)
    curr_full = np.where(armed, 0.8 + 70.0 * mot_th ** 2.2 * (n_mot / 4.0), 0.4)
    curr = curr_full[k] + rng.normal(0, 0.05, k.size)
    used = np.cumsum(np.clip(curr, 0, None) * 0.1) / 3.6
    cap = 5200.0
    ocv = cells * (4.18 - 0.35 * used / cap)
    volt = ocv - 0.012 * cells / 4 * curr + rng.normal(0, 0.005, k.size)
    tables["BAT[0]"] = {
        "t": t_abs[k] + 0.0008, "Inst": np.zeros(k.size, np.uint8), "Volt": _f32(volt),
        "VoltR": _f32(ocv), "Curr": _f32(curr), "CurrTot": _f32(used),
        "EnrgTot": _f32(np.cumsum(volt * np.clip(curr, 0, None) * 0.1) / 3600.0),
        "Temp": _cd(np.full(k.size, 0.0)), "Res": _f32(np.full(k.size, 0.012 * cells / 4)),
        "RemPct": np.clip(np.rint(100.0 * (1.0 - used / cap)), 0, 100).astype(np.uint8),
        "H": np.ones(k.size, np.uint8), "SH": np.zeros(k.size, np.uint8),
    }

    # --- VIBE[0](10 Hz)---
    ks = np.clip(k10 * dec, 0, vibe.shape[0] - 1)
    tables["VIBE[0]"] = {
        "t": t_abs[k10] + 0.0009, "IMU": np.zeros(k10.size, np.uint8),
        "VibeX": _f32(vibe[ks, 0]), "VibeY": _f32(vibe[ks, 1]), "VibeZ": _f32(vibe[ks, 2]),
        "Clip": clip_cum[ks].astype(np.uint32),
    }

    # --- IMU[0](25 Hz:濾波後的陀螺儀 / 加速度計,取樣)---
    ks = np.clip(k25 * dec, 0, acc_lp.shape[0] - 1)
    tables["IMU[0]"] = {
        "t": t_abs[k25] + 0.0011, "I": np.zeros(k25.size, np.uint8),
        "GyrX": _f32(full[k25, I["gmr"]]), "GyrY": _f32(full[k25, I["gmp"]]),
        "GyrZ": _f32(full[k25, I["gmy"]]),
        "AccX": _f32(acc_lp[ks, 0]), "AccY": _f32(acc_lp[ks, 1]), "AccZ": _f32(acc_lp[ks, 2]),
        "EG": np.zeros(k25.size, np.uint32), "EA": np.zeros(k25.size, np.uint32),
        "T": _f32(np.full(k25.size, 45.0)), "GH": np.ones(k25.size, np.uint8),
        "AH": np.ones(k25.size, np.uint8),
        "GHz": np.full(k25.size, int(SENSOR_HZ), np.uint16),
        "AHz": np.full(k25.size, int(SENSOR_HZ), np.uint16),
    }

    # --- ISBH / ISBD(batch sampler:加速度 / 陀螺儀交替,每批 1024 筆、每則 32 筆)---
    if batch:
        tables.update(_batch_tables(gyro_raw, acc_raw, prm))

    # --- GPS 5 Hz / POS 10 Hz / ORGN ---
    pn, pe = full[:, I["pn"]], full[:, I["pe"]]
    enu = np.column_stack([pe, pn, z])
    k5 = _log_idx(n_all, 5.0, rng, jitter=False)
    ng = k5.size
    gnoise = np.column_stack([_gm(rng, ng, 5.0, 15.0, 0.25) + rng.normal(0, 0.02, ng),
                              _gm(rng, ng, 5.0, 15.0, 0.25) + rng.normal(0, 0.02, ng),
                              _gm(rng, ng, 5.0, 20.0, 0.5) + rng.normal(0, 0.04, ng)])
    lat, lng, galt = frames.enu_to_geodetic(enu[k5] + gnoise, ORIGIN)
    vel = np.column_stack([np.gradient(pn, dt), np.gradient(pe, dt), np.gradient(z, dt)])
    nsat = np.clip(16 + np.cumsum(np.where(rng.random(ng) < 0.02, rng.choice([-1, 1], ng), 0)),
                   12, 20).astype(np.uint8)
    tables["GPS[0]"] = {
        "t": t_abs[k5] + 0.0013, "I": np.zeros(ng, np.uint8), "Status": np.full(ng, 3, np.uint8),
        "GMS": (300_000 + np.rint((t_abs[k5] - tb) * 1000)).astype(np.uint32),
        "GWk": np.full(ng, 2356, np.uint16), "NSats": nsat,
        "HDop": _cd(np.clip(0.7 + rng.normal(0, 0.02, ng), 0.5, 1.2)),
        "Lat": lat, "Lng": lng, "Alt": _cd(galt),
        "Spd": _f32(np.hypot(vel[k5, 0], vel[k5, 1]) + np.abs(rng.normal(0, 0.03, ng))),
        "GCrs": _f32(np.degrees(np.arctan2(vel[k5, 1], vel[k5, 0])) % 360.0),
        "VZ": _f32(-vel[k5, 2] + rng.normal(0, 0.03, ng)), "Yaw": np.zeros(ng),
        "U": np.ones(ng, np.uint8),
    }
    k = k10
    pnoise = np.column_stack([_gm(rng, k.size, 10.0, 30.0, 0.05) for _ in range(3)])
    plat, plng, palt = frames.enu_to_geodetic(enu[k] + pnoise, ORIGIN)
    tables["POS"] = {"t": t_abs[k] + 0.0014, "Lat": plat, "Lng": plng, "Alt": _f32(palt),
                     "RelHomeAlt": _f32(palt - ORIGIN[2]), "RelOriginAlt": _f32(palt - ORIGIN[2])}
    tables["ORGN"] = {"t": np.array([tb + 2.0, t_arm]), "Type": np.array([0, 1], np.uint8),
                      "Lat": np.array([ORIGIN[0], ORIGIN[0]]), "Lng": np.array([ORIGIN[1], ORIGIN[1]]),
                      "Alt": np.array([ORIGIN[2], ORIGIN[2]])}

    units = {k_: dict(v) for k_, v in SIM_UNITS.items()}
    return tables, units


def _batch_tables(gyro_raw: np.ndarray, acc_raw: np.ndarray, prm: dict) -> dict:
    """AP_InertialSensor::BatchSampler:ISBH(批次標頭)+ ISBD(int16[32] × 3 軸)。

    原始(濾波前)取樣、SENSOR_HZ;加速度(type 0)與陀螺儀(type 1)交替。
    乘數:加速度 INT16_MAX / (16 g)、陀螺儀 INT16_MAX / radians(2000)。值 = 原始 / mul。
    """
    fs = SENSOR_HZ
    cnt = int(prm.get("INS_LOG_BAT_CNT", 1024))
    per = 32
    lgin = float(prm.get("INS_LOG_BAT_LGIN", 20)) / 1000.0
    mul_acc = int(32767 / (16 * G))
    mul_gyr = int(32767 / math.radians(2000.0))
    n_s = gyro_raw.shape[0]
    hdr = {k: [] for k in ("t", "N", "type", "instance", "mul", "smp_cnt", "SampleUS", "smp_rate")}
    dat = {k: [] for k in ("t", "N", "seqno", "x", "y", "z")}
    i0 = int(0.5 * fs)
    seq = 0
    typ = 0
    while i0 + cnt < n_s:
        src, mul = (acc_raw, mul_acc) if typ == 0 else (gyro_raw, mul_gyr)
        blk = np.clip(np.rint(src[i0:i0 + cnt] * mul), -32768, 32767).astype(np.int16)
        t_first = T_BOOT + i0 / fs
        t_hdr = T_BOOT + (i0 + cnt) / fs + 0.001
        hdr["t"].append(t_hdr)
        hdr["N"].append(seq)
        hdr["type"].append(typ)
        hdr["instance"].append(0)
        hdr["mul"].append(mul)
        hdr["smp_cnt"].append(cnt)
        hdr["SampleUS"].append(int(round(t_first * 1e6)))
        hdr["smp_rate"].append(fs)
        n_msg = cnt // per
        for j in range(n_msg):
            dat["t"].append(t_hdr + (j + 1) * lgin)
            dat["N"].append(seq)
            dat["seqno"].append(j)
            dat["x"].append(blk[j * per:(j + 1) * per, 0])
            dat["y"].append(blk[j * per:(j + 1) * per, 1])
            dat["z"].append(blk[j * per:(j + 1) * per, 2])
        i0 = int(round((t_hdr + (n_msg + 1) * lgin - T_BOOT) * fs))
        seq += 1
        typ ^= 1
    isbh = {"t": np.array(hdr["t"]), "N": np.array(hdr["N"], np.uint16),
            "type": np.array(hdr["type"], np.uint8), "instance": np.array(hdr["instance"], np.uint8),
            "mul": np.array(hdr["mul"], np.uint16), "smp_cnt": np.array(hdr["smp_cnt"], np.uint16),
            "SampleUS": np.array(hdr["SampleUS"], np.uint64), "smp_rate": _f32(hdr["smp_rate"])}
    isbd = {"t": np.array(dat["t"]), "N": np.array(dat["N"], np.uint16),
            "seqno": np.array(dat["seqno"], np.uint16),
            "x": np.array(dat["x"], np.int16).reshape(-1, per),
            "y": np.array(dat["y"], np.int16).reshape(-1, per),
            "z": np.array(dat["z"], np.int16).reshape(-1, per)}
    return {"ISBH": isbh, "ISBD": isbd}


def write_tuning_bin(logdata: LogData, path: str) -> None:
    """以 ArduPilot 格式字元(ATT 用 c / C 等)把模擬資料寫成 .bin。"""
    from .fakebin import write_index_to_bin
    write_index_to_bin(logdata.index, path, fmt_override=SIM_FORMATS)


# ======================================================================
# 真值(info)
# ======================================================================

def _psd(x: np.ndarray, fs: float, nper: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD(Hann、50 % 重疊、去平均)。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size < nper:
        nper = max(16, 1 << int(math.floor(math.log2(max(x.size, 16)))))
        if x.size < nper:
            return np.zeros(1), np.zeros(1)
    w = np.hanning(nper)
    step = nper // 2
    segs = [x[i:i + nper] for i in range(0, x.size - nper + 1, step)]
    acc = np.zeros(nper // 2 + 1)
    for s in segs:
        s = s - s.mean()
        acc += np.abs(np.fft.rfft(s * w)) ** 2
    acc /= max(len(segs), 1) * fs * np.sum(w ** 2)
    acc[1:-1] *= 2
    return np.fft.rfftfreq(nper, 1.0 / fs), acc


def _window_mask(t_rel: np.ndarray, wins: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(t_rel.size, bool)
    for a, b in wins:
        m |= (t_rel >= a) & (t_rel < b)
    return m


def _hover_psd(x: np.ndarray, wins, fs: float, nper: int = 512):
    """把每個懸停窗(相對 T_BOOT 秒;x 從 T_BOOT 起以 fs 取樣)各自做 Welch,再加權平均。"""
    acc = None
    cnt = 0
    f = None
    for a, b in wins:
        i0, i1 = int(a * fs), int(b * fs)
        seg = x[i0:i1]
        if seg.size < nper:
            continue
        f, p = _psd(seg, fs, nper)
        acc = p * seg.size if acc is None else acc + p * seg.size
        cnt += seg.size
    if acc is None:
        return np.zeros(1), np.zeros(1)
    return f, acc / cnt


def _osc_peak(f: np.ndarray, p: np.ndarray, lo: float = 3.0, hi: float = 40.0) -> dict:
    """頻段內最高峰:頻率(對數 PSD 拋物線內插)、峰值 / 頻段中位數、峰值 ±1.5 Hz 的 RMS。"""
    m = (f >= lo) & (f <= hi)
    if not m.any() or f.size < 3:
        return {"hz": float("nan"), "peak_ratio": 0.0, "band_rms": 0.0}
    i = int(np.argmax(np.where(m, p, -1.0)))
    hz = float(f[i])
    if 0 < i < f.size - 1 and p[i - 1] > 0 and p[i + 1] > 0 and p[i] > 0:
        la, lb, lc = np.log(p[i - 1]), np.log(p[i]), np.log(p[i + 1])
        den = la - 2 * lb + lc
        if den < 0:
            hz += float(0.5 * (la - lc) / den) * float(f[1] - f[0])
    med = float(np.median(p[m])) or 1e-30
    band = (f >= hz - 1.5) & (f <= hz + 1.5)
    return {"hz": hz, "peak_ratio": float(p[i] / med),
            "band_rms": float(np.sqrt(np.trapz(p[band], f[band]))) if band.sum() > 1 else 0.0}


def _make_info(scenario, spec, prm, plant, full, t_abs, airborne, armed, at_max, at_min,
               man_events, hover_windows, ev_t, vibe, clip_cum, f_motor, f_hover, level, seed,
               frame, fast_log, pid_log, batch, dec, tables) -> dict:
    I = _RI
    fs = LOOP_HZ
    t_rel = t_abs - T_BOOT
    deg = np.degrees
    rate = deg(full[:, [I["wr"], I["wp"], I["wy"]]])
    rate_tgt = deg(full[:, [I["rtr"], I["rtp"], I["rty"]]])
    gyro = deg(full[:, [I["gmr"], I["gmp"], I["gmy"]]])
    ang = deg(full[:, [I["ar"], I["ap"], I["ay"]]])
    ang_tgt = deg(full[:, [I["atr"], I["atp"], I["aty"]]])
    hov = _window_mask(t_rel, hover_windows) & airborne
    man = _window_mask(t_rel, [(a, b + 1.0) for a, b, *_ in man_events]) & airborne

    axes = ("roll", "pitch", "yaw")
    osc = {}
    metrics: dict[str, dict] = {}
    hw = list(hover_windows)
    for j, ax in enumerate(axes):
        a = ax[0]
        out = full[:, I["p" + a]] + full[:, I["i" + a]] + full[:, I["d" + a]] + full[:, I["ff" + a]]
        fo, po = _hover_psd(out, hw, fs)
        hf_o = (fo >= 20.0) & (fo <= 150.0)
        dterm = full[:, I["d" + a]]
        f, p = _hover_psd(rate[:, j] - np.mean(rate[hov, j]) if hov.any() else rate[:, j], hw, fs)
        osc[ax] = _osc_peak(f, p)          # 懸停時真實角速度的頻譜峰(deg/s)
        hf = (f >= 20.0) & (f <= 80.0)
        err_ang = ang_tgt[:, j] - ang[:, j]
        if ax == "yaw":
            err_ang = (err_ang + 180.0) % 360.0 - 180.0
        metrics[ax] = {
            "rate_rms_hover_dps": float(np.sqrt(np.mean((rate[hov, j] - rate[hov, j].mean()) ** 2)))
            if hov.any() else float("nan"),
            "rate_hf_power": float(np.trapz(p[hf], f[hf])) if hf.any() else 0.0,
            "rate_err_rms_dps": float(np.sqrt(np.mean((rate_tgt[man, j] - gyro[man, j]) ** 2)))
            if man.any() else float("nan"),
            "angle_err_rms_deg": float(np.sqrt(np.mean(err_ang[man] ** 2))) if man.any() else float("nan"),
            "angle_bias_deg": float(np.mean(err_ang[hov])) if hov.any() else float("nan"),
            "angle_lag_ms": _lag_ms(ang_tgt[:, j], ang[:, j], man, fs) if ax != "yaw"
            else _lag_ms(np.gradient(np.unwrap(np.radians(ang_tgt[:, j]))),
                         np.gradient(np.unwrap(np.radians(ang[:, j]))), man, fs),
            "out_hf_rms": float(np.sqrt(np.trapz(po[hf_o], fo[hf_o]))) if hf_o.any() else 0.0,
            "dterm_rms_hover": float(np.sqrt(np.mean(dterm[hov] ** 2))) if hov.any() else float("nan"),
            "pterm_rms_hover": float(np.sqrt(np.mean(full[hov, I["p" + a]] ** 2))) if hov.any()
            else float("nan"),
        }
    osc_axes = spec.get("osc_axes", ())
    osc_hz = float(np.mean([osc[a]["hz"] for a in osc_axes])) if osc_axes else None

    step = {ax: rate_step_response(prm=prm, axis=ax, frame=frame, plant=plant,
                                   hover=float(prm["MOT_THST_HOVER"]))["metrics"]
            for ax in axes}
    ang_step = {ax: angle_step_response(prm=prm, axis=ax, frame=frame, plant=plant,
                                        hover=float(prm["MOT_THST_HOVER"]))["metrics"]
                for ax in ("roll", "pitch")}
    n_air = int(airborne.sum())
    vk = np.clip(np.arange(0, full.shape[0], 40) * dec, 0, vibe.shape[0] - 1)
    air10 = airborne[::40][:vk.size]
    vib_air = vibe[vk][air10] if air10.any() else vibe[vk]
    info = {
        "scenario": scenario,
        "description": SCENARIOS[scenario],
        "seed": seed,
        "frame": frame,
        "expected": list(spec.get("expected", [])),
        "secondary": list(spec.get("secondary", [])),
        "forbid": list(spec.get("forbid", [])),
        "expected_codes": list(spec.get("codes", [])),
        "expect_ok": scenario == "good",      # 只有 good 應該「沒有需要調整的地方」
        # 記錄設定建議:缺 ATTITUDE_FAST / PID 時應建議的確切 LOG_BITMASK;沒 batch 時建議開 batch
        "expected_logging": ([("LOG_BITMASK", DIR_SET, float(int(prm["LOG_BITMASK"]) | _MASK_ATT_FAST
                                                             | _MASK_PID))]
                             if not (fast_log and pid_log) else [])
                            + ([("INS_LOG_BAT_MASK", DIR_SET, 1.0)] if not batch else []),
        "slow_log": {"detectable": _SLOW_LOG[scenario][0], "note": _SLOW_LOG[scenario][1]},
        "params": dict(prm),
        "changed": {k: (float(prm[k]), float(default_params(frame).get(k, prm[k])))
                    for k in spec.get("params", {})},
        "osc_hz": osc_hz,
        "osc": osc,
        "noise_hz": f_hover if spec.get("noise") == "motor_peak" else None,
        "motor_hz_hover": f_hover,
        "motor_hz_range": (float(np.percentile(f_motor[f_motor > 0.8 * f_hover], 2))
                           if (f_motor > 0.8 * f_hover).any() else None,
                           float(np.percentile(f_motor[f_motor > 0.8 * f_hover], 98))
                           if (f_motor > 0.8 * f_hover).any() else None),
        "step": step,
        "angle_step": ang_step,
        "metrics": metrics,
        "saturation": {"at_max_pct": 100.0 * float(at_max.sum()) / max(n_air, 1),
                       "at_min_pct": 100.0 * float(at_min.sum()) / max(n_air, 1),
                       "pwm_top": float(np.rint(_thrust_to_pwm(np.ones(1), prm)[0])),
                       "pwm_bottom": float(np.rint(_thrust_to_pwm(np.zeros(1), prm)[0]))},
        "vibe": {"mean": [float(v) for v in vib_air.mean(axis=0)],
                 "max": [float(v) for v in vib_air.max(axis=0)],
                 "clip_total": int(clip_cum[-1])},
        "times": {k: T_BOOT + v for k, v in ev_t.items()},
        "windows": {"hover": [(T_BOOT + a, T_BOOT + b) for a, b in hover_windows],
                    "manoeuvre": [(T_BOOT + a, T_BOOT + b) for a, b, *_ in man_events],
                    "flight": (T_BOOT + ev_t["liftoff"], T_BOOT + ev_t["touch"])},
        "manoeuvres": [(T_BOOT + a, T_BOOT + b, kind, axis, amp)
                       for a, b, kind, axis, amp in man_events],
        "plant": {"K_rad_s2": dict(plant["K"]),
                  "K_deg_s2": {k: v * _R2D for k, v in plant["K"].items()},
                  "damp": dict(plant["damp"]), "tau_motor_s": plant["tau_motor"],
                  "gyro_delay_s": plant["gyro_delay"] / LOOP_HZ,
                  "total_delay_s": (plant["gyro_delay"] + 0.5) / LOOP_HZ,
                  "gyro_filter_hz": float(prm["INS_GYRO_FILTER"]),
                  "hover": float(prm["MOT_THST_HOVER"]),
                  "cg_offset_rad_s2": dict(spec.get("cg_offset", PLANT["cg_offset"]))},
        "noise_level": dict(level),
        "log": {"fast_log": fast_log, "pid_log": pid_log, "batch": batch,
                "LOG_BITMASK": int(prm["LOG_BITMASK"]),
                "INS_LOG_BAT_MASK": int(prm["INS_LOG_BAT_MASK"]),
                "rates_hz": {name: _rate_of(tables.get(name)) for name in
                             ("ATT", "RATE", "PIDR", "CTRL", "RCOU", "CTUN", "MOTB", "VIBE[0]",
                              "IMU[0]", "GPS[0]", "POS")}},
        "truth": {"t": t_abs, "rate": rate, "rate_tgt": rate_tgt, "gyro": gyro, "ang": ang,
                  "ang_tgt": ang_tgt, "airborne": airborne, "armed": armed, "fs": fs},
    }
    return info


def _rate_of(tbl) -> float | None:
    if tbl is None or tbl["t"].size < 3:
        return None
    return float(1.0 / np.median(np.diff(tbl["t"])))


def _lag_ms(des: np.ndarray, act: np.ndarray, mask: np.ndarray, fs: float, max_ms: float = 400.0) -> float:
    """期望 → 實際的延遲(互相關峰值,僅操作期間)。"""
    if mask.sum() < 50:
        return float("nan")
    d = np.where(mask, des - np.mean(des[mask]), 0.0)
    a = np.where(mask, act - np.mean(act[mask]), 0.0)
    m = int(max_ms / 1000.0 * fs)
    nfft = 1 << int(math.ceil(math.log2(d.size + m + 1)))
    cc = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(d, nfft)), nfft)
    lags = np.concatenate([cc[:m + 1]])
    return float(np.argmax(lags) / fs * 1000.0)


# ======================================================================
# 步階實驗(無雜訊、空中、懸停油門)
# ======================================================================

def _step_metrics(t: np.ndarray, y: np.ndarray, target: float) -> dict:
    """正規化步階:過衝 %、10→90 % 上升時間、±5 % 安定時間、峰值時間。"""
    yn = y / target
    peak = float(np.max(yn))
    i10 = np.flatnonzero(yn >= 0.1)
    i90 = np.flatnonzero(yn >= 0.9)
    rise = float(t[i90[0]] - t[i10[0]]) if i10.size and i90.size else float("nan")
    out = np.flatnonzero(np.abs(yn - 1.0) > 0.05)
    settle = float(t[out[-1] + 1]) if out.size and out[-1] + 1 < t.size else (0.0 if not out.size else float("nan"))
    return {"overshoot_pct": max(0.0, (peak - 1.0) * 100.0), "rise_s": rise, "settle_s": settle,
            "peak_s": float(t[int(np.argmax(yn))])}


def _step_setup(prm, frame, plant, hover, n):
    prm = dict(scenario_params("good", frame)) if prm is None else dict(prm)
    plant = PLANT if plant is None else plant
    hover = float(prm["MOT_THST_HOVER"]) if hover is None else hover
    cfg = _ctrl_cfg(prm, frame, plant, hover)
    zeros = [0.0] * n
    prog = {"des_r": list(zeros), "des_p": list(zeros), "des_yr": list(zeros),
            "zt": [10.0] * n, "vzt": zeros, "stage": [1] * n}
    nz = {"r": zeros, "p": zeros, "y": zeros}
    init = dict(z=10.0, airborne=True, umt=hover)
    return prm, cfg, prog, nz, init


def rate_step_response(*, prm: dict | None = None, axis: str = "roll", amp_dps: float = 60.0,
                       duration_s: float = 0.8, frame: str = "quad", plant: dict | None = None,
                       hover: float | None = None) -> dict:
    """角速度步階(ACRO 式直接給角速度目標,FF 照 ATC_RATE_FF_ENAB):回傳 t、目標、真實角速度、指標。"""
    n = int(duration_s * LOOP_HZ) + int(0.1 * LOOP_HZ)
    prm, cfg, prog, nz, init = _step_setup(prm, frame, plant, hover, n)
    cfg = dict(cfg)
    cfg["ff_enab"] = False                    # 只看角速度迴圈:目標 = 步階本身
    j = ("roll", "pitch", "yaw").index(axis)
    k0 = int(0.1 * LOOP_HZ)
    step = [0.0] * k0 + [math.radians(amp_dps)] * (n - k0)
    acro = [[0.0] * n, [0.0] * n, [0.0] * n]
    acro[j] = step
    prog["acro"] = tuple(acro)
    rec = _simulate(cfg, prog, nz, {"r": [0.0] * n, "p": [0.0] * n, "y": [0.0] * n}, (0.0, 0.0), n, init)
    t = np.arange(n - k0) / LOOP_HZ
    y = np.degrees(rec[k0:, _RI[("wr", "wp", "wy")[j]]])
    return {"t": t, "target": np.full(t.size, amp_dps), "rate": y,
            "metrics": _step_metrics(t, y, amp_dps)}


def angle_step_response(*, prm: dict | None = None, axis: str = "roll", amp_deg: float = 10.0,
                        duration_s: float = 1.5, frame: str = "quad", plant: dict | None = None,
                        hover: float | None = None) -> dict:
    """駕駛員角度步階(STABILIZE):回傳 t、期望角(整形後目標)、實際角、指標(相對駕駛員輸入)。"""
    n = int(duration_s * LOOP_HZ) + int(0.1 * LOOP_HZ)
    prm, cfg, prog, nz, init = _step_setup(prm, frame, plant, hover, n)
    k0 = int(0.1 * LOOP_HZ)
    key = {"roll": "des_r", "pitch": "des_p"}[axis]
    prog[key] = [0.0] * k0 + [math.radians(amp_deg)] * (n - k0)
    rec = _simulate(cfg, prog, nz, {"r": [0.0] * n, "p": [0.0] * n, "y": [0.0] * n}, (0.0, 0.0), n, init)
    t = np.arange(n - k0) / LOOP_HZ
    j = {"roll": 0, "pitch": 1}[axis]
    y = np.degrees(rec[k0:, _RI[("ar", "ap")[j]]])
    tgt = np.degrees(rec[k0:, _RI[("atr", "atp")[j]]])
    m = _step_metrics(t, y, amp_deg)
    # 相對整形後目標的過衝(目標本身不過衝,超出目標的部分才是控制器造成)
    m["overshoot_vs_target_pct"] = max(0.0, float(np.max(y - tgt)) / amp_deg * 100.0)
    return {"t": t, "target": tgt, "angle": y, "metrics": m}
