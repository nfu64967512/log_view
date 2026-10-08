"""PID 調參建議引擎(ArduCopter 多旋翼;QuadPlane 以 Q_A_ 參數初步支援)。

只用 numpy:Welch PSD / CSD / 同調、峰值偵測、頻率響應、Wiener 反卷積步階響應、互相關延遲都在這裡實作。

流程(analyze):
1. 資料:機型 / 機架、參數前綴(ATC_ / Q_A_)、各訊息實際取樣率(中位 dt)。
2. 飛行遮罩:ARM 段 ∩ EV NOT_LANDED…LAND_COMPLETE(沒有就用油門 + 高度),起降前後各裁 2 s,
   排除 AUTOTUNE / FLIP。
3. 每軸:角速度層級(PIDx 或 RATE)與角度層級(ATT)的追蹤誤差、偏差、延遲、輸出飽和、
   誤差頻譜峰值、頻率響應、步階響應;另外用 IMU 陀螺儀(取樣較快時)、CTRL、PIDx 細項補證據。
4. 整機:振動、馬達飽和、懸停油門、陷波濾波 / 批次 FFT、預設增益、記錄設定。
5. 規則 → Finding + Suggestion:一次一個參數、每次 10–25 %、夾在合理範圍(絕不因夾範圍而反向或加大步幅);
   振動過大 / 馬達飽和 / 有震盪跡象時不建議提高增益(也不減小 INPUT_TC)。
6. 低速記錄(≤ 25 Hz):只有在迴圈頻率計算的 CTRL 互相印證時才給方向——IMU 峰 + CTRL P 項主導且頻率吻合
   → 小降 P;CTRL D/P ≥ 3 → 降 D;延遲大 + CTRL 顯示誤差在低頻 + 搖桿模式 → P +10 %。
   CTRL D/P 介於 1.2–3(馬達噪音進 D 項也會如此)或位置控制模式(LOITER / AUTO)→ 只提醒、不改增益。
7. 記錄率低於迴圈頻率(LOG_FILE_RATEMAX 50 / 100 Hz)的角速度峰要姿態角交叉檢查(真的擺動在姿態角上
   有 A/(2πf)),否則當成可能的噪音混疊;高於控制頻寬(約 50 Hz)的峰是馬達噪音 → 陷波,不降 P / D。

校正:tests/test_tuning.py 的 test_sim_scenario_matrix(core.sim_tuning 10 情境 × 高速+PID / 高速 / 10 Hz)。

所有門檻都是模組層級具名常數(附 ArduPilot 依據),方便之後用模擬情境與實機 log 校正。
時間一律是 log 絕對秒;角度 deg、角速度 deg/s。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .tuning_model import (AXES, AXIS_LABEL, AXIS_PARAM, DIR_CHECK, DIR_DOWN, DIR_SET, DIR_UP,
                           SEV_BAD, SEV_INFO, SEV_LABEL, SEV_OK, SEV_ORDER, SEV_WARN,
                           AxisAnalysis, Finding, FreqResponse, Spectrum, StepResponse,
                           Suggestion, TuningReport)

__all__ = [
    "analyze", "format_report_text", "PARAM_RANGES",
    "hann", "choose_nperseg", "spectral_frames", "welch_psd", "csd", "coherence", "band_rms",
    "find_peaks", "Peak", "freq_response", "lag_xcorr", "step_response_wiener", "step_metrics",
    "flight_intervals", "gain_suggestion", "gain_names", "param_range", "level_stats",
    "batch_gyro_spectrum", "Frames", "ctrl_fmax", "dp_ratio_model",
    "OscPeak", "LevelStats", "CtrlStats", "PidStats", "VehicleContext", "AxisContext",
    "axis_rules", "apply_safety",
]

# =====================================================================
# 門檻常數(校正時只改這裡)
# =====================================================================

# ---- 資料品質 ----
LOW_RATE_HZ = 25.0          # 取樣率 ≤ 此值:5–20 Hz 的 PID 震盪看不到(或混疊成假的低頻),只判讀慢速行為
INCREASE_MIN_HZ = 50.0      # 角速度資料率低於此值時不建議「提高角速度增益」:低頻落差可能是混疊假象
LOG_BIT_ATT_FAST = 1        # Copter LOG_BITMASK bit 0:ATTITUDE_FAST(ATT / RATE / PID 以迴圈頻率記錄)
LOG_BIT_PID = 4096          # Copter LOG_BITMASK bit 12:PID(PIDR / PIDP / PIDY)
LOG_BIT_PLANE_FULLRATE = 1 << 20   # Plane LOG_BITMASK bit 20:ATTITUDE_FULLRATE(VTOL 的 ATT / PIQx 以迴圈頻率;
                                   #   Plane 的 bit 0 只有 25 Hz、bit 12 是 CAMERA,沒有獨立的 PID bit)
LOOP_RATE_DEFAULT = 400.0   # SCHED_LOOP_RATE 沒記錄時的迴圈頻率
FULL_RATE_FRAC = 0.9        # 記錄率 ≥ 0.9 × 迴圈頻率:log 看到的就是控制器看到的(沒有記錄造成的混疊)
COVERAGE_WARN = 0.5         # 實際納入分析的飛行樣本 < 50 %(資料空洞 / 掉訊息)→ 提醒

# ---- 飛行遮罩 ----
MIN_FLIGHT_S = 10.0         # 納入分析的飛行少於 10 s → 資料不足
TRIM_TAKEOFF_S = 2.0        # 起飛後裁掉 2 s(離地瞬間的地效 / 積分器重置)
TRIM_LAND_S = 2.0           # 落地前裁掉 2 s
MIN_PIECE_S = 2.0           # 短於此的連續片段不納入(頻譜 / 互相關需要長度)
GAP_FACTOR = 3.0            # 相鄰樣本間隔 > 3 × 中位 dt(且 > GAP_MIN_S)視為資料空洞,分段
GAP_MIN_S = 0.02            # 400 Hz 時連掉 2–7 筆(< 20 ms)只內插、不切段
FLIGHT_THR_FRAC = 0.5       # 沒有起降事件時:CTUN.ThO > 0.5 × MOT_THST_HOVER 視為在空中
FLIGHT_THR_MIN = 0.08       # …且至少 0.08
FLIGHT_ALT_MIN_M = 0.5      # …且 CTUN.Alt 比該段最低點高 0.5 m
EXCLUDE_MODES = ("AUTOTUNE", "FLIP", "QAUTOTUNE")   # 增益在變 / 刻意的特技動作:不納入

# ---- 頻譜 ----
WELCH_WIN_S = 2.0           # 角速度層級:fs × 2 s 向下取 2 的次方、上限 1024 → 50 Hz 以上約 1.28 s(已校正,勿改)
WELCH_WIN_S_ANG = 6.0       # 角度層級:約 5–6 s(要分辨 0.5–3 Hz 的擺動;各取樣率解析度一致)
WELCH_MIN_N = 64
WELCH_MAX_N = 1024          # 角速度層級窗長上限
WELCH_MAX_N_ANG = 4096      # 角度層級窗長上限(400 Hz 時 2048 點 = 5.12 s;1024 只有 2.56 s,< 1 Hz 的擺動會漏)
WELCH_OVERLAP = 0.5
WELCH_MAX_FRAMES = 4000     # 窗數上限(超長、高速 log 時加大步距,控制記憶體;統計上已綽綽有餘)
STEP_MAX_WINDOWS = 3000     # 步階反卷積的時間窗上限(同上)
PEAK_FMIN_HZ = 0.5          # 更慢的是飛行動作本身,不算震盪
PEAK_FMAX_FRAC = 0.8        # 只看 0.8 × Nyquist 以下(更高處受抗混疊濾波 / 窗洩漏影響)
IMU_PEAK_FMAX_FRAC = 0.95   # IMU 記錄是瞬時取樣(沒有抗混疊衰減),低速時接近 Nyquist 的峰也要看
PEAK_PROM_RATIO = 3.0       # 峰值 / 左右谷底(較高者)≥ 3(≈ 5 dB)才算峰
PEAK_MED_RATIO = 1.5        # 且峰值 ≥ 1.5 × 滑動中位數(排除短資料的雜訊起伏)
PEAK_BASE_HZ = 4.0          # 找谷底 / 中位數的視窗半寬(Hz;比一般震盪峰寬)
PEAK_EDGE_RATIO = 2.0       # 峰的範圍:連續高於谷底 2 倍的頻帶(用來積分峰的 RMS)
PEAK_MIN_RMS_RATE = 1.0     # deg/s:更小的峰不列出
PEAK_MIN_RMS_ANG = 0.15     # deg
COH_MIN = 0.6               # 頻率響應只採用同調 ≥ 0.6 的頻點
COH_MIN_FRAMES = 8          # Welch 窗數太少時同調度偏高、不可信

# ---- 震盪分類(ArduPilot 調參指南:高頻抖動先降 D;中頻擺動降 P;慢速擺動看角度環)----
OSC_MID_HZ = 4.0            # 4–15 Hz:角速度環 P 偏高的典型頻率;< 4 Hz 的角度誤差峰歸角度環(慢速擺動)
OSC_HIGH_HZ = 15.0          # ≥ 15 Hz:D 項 / 雜訊 / 濾波不足的典型頻率
OSC_RATE_WARN_RMS = 3.0     # deg/s:持續峰 RMS ≥ 此值 → 建議調整
OSC_RATE_BAD_RMS = 8.0      # deg/s:→ 需要處理
OSC_ANG_WARN_RMS = 0.4      # deg
OSC_ANG_BAD_RMS = 1.0       # deg
OSC_YAW_ANG_WARN_RMS = 1.0  # deg:偏航角(轉向時 1–2° 的落後很常見,門檻較寬)
OSC_YAW_ANG_BAD_RMS = 2.5
OSC_SUSTAIN_FRAC = 0.5      # 峰出現在 ≥ 50 % 的時間窗 → 持續震盪(否則為偶發)
ALIAS_SAFE_HZ = 40.0        # 取樣率 ≥ 40 Hz(0.8·Nyquist ≥ 16 Hz)的角速度峰才「可能」是真實頻率;
                            #   低於迴圈頻率時還要姿態角交叉檢查(PHYS_MIN_RATIO),因為記錄是瞬時取樣、沒有抗混疊
PHYS_MIN_RATIO = 0.5        # 真的 f Hz 角速度震盪(RMS A)在姿態角上有 A/(2πf) 的擺動;量到的比例 < 0.5
                            #   → 多半是更高頻的馬達噪音混疊下來(108 Hz → 8 Hz 時比例只有約 0.07)
OSC_CTRL_FMAX_MIN_HZ = 40.0     # 控制迴路可能自激的頻率上限 = min(max(40, 2.5 × INS_GYRO_FILTER), 0.2 × 迴圈頻率);
OSC_CTRL_FMAX_GF = 2.5          #   陀螺儀低通 ~20 Hz、1 個迴圈延遲、馬達 ~40 ms 延遲的多旋翼在 60–120 Hz 沒有迴路增益,
OSC_CTRL_FMAX_LOOP = 0.2        #   那裡的峰是馬達 / 機架噪音 → 交給陷波(gyro_noise),不降 P / D
OSC_SUSPECT_MIN_RMS = 2.0   # deg/s:低速記錄的「疑似震盪」至少這麼大才提出
SUSPECT_FMATCH_FRAC = 0.2   # 低速 IMU 峰的混疊候選頻率與 CTRL 推估頻率相差 ≤ 20 % 才算「互相印證」
ALIAS_MATCH_HZ = 0.3        # 角度誤差峰與疑似震盪的混疊頻率相差 ≤ 0.3 Hz → 視為同一個震盪的混疊,不怪角度環
PILOT_SHARE_MIN = 0.5       # 搖桿直接給姿態的模式 < 50 % 時,角度層級 |H| / 步階只當弱證據
# 「搖桿直接給姿態」的模式。不含 FLOWHOLD(搖桿回中時以光流回授定點,同 LOITER)與
# DRIFT(橫滾目標由機體側向速度算出、俯仰依速度煞停):兩者的期望姿態含速度 / 位置回授
PILOT_MODES = ("STABILIZE", "ACRO", "ALT_HOLD", "SPORT", "QSTABILIZE", "QACRO", "QHOVER")

# ---- 頻率響應 / 步階響應 ----
HPEAK_RATE_WARN = 1.3       # 角速度環 |H| 共振峰 > 1.3(≈ +2.3 dB)→ 阻尼不足
HPEAK_ANG_WARN = 1.6        # Roll / Pitch 角度環(期望角含角度環回授:正常機約 1.2–1.4);偏航與 LOITER / AUTO 不採用 |H|
HPEAK_SLUGGISH_MAX = 1.2    # 判「偏軟」時 |H| 峰需小於此(沒有共振;正常機的估計值約 1.0–1.18)
HPEAK_RATE_FMIN_HZ = 3.0    # 角速度 |H| 峰在 3 Hz 以下多半是角度環回授造成(期望角速度 = ANG_P × 角度誤差),不採用
STEP_MIN_WINDOWS = 3        # 反卷積至少要 3 個有輸入的時間窗
STEP_MIN_SAMPLES = 10       # 步階響應至少 10 個樣本(10 Hz 的角速度資料算不出來)
STEP_RATE_WIN_S = 1.0       # 角速度層級:1 s 窗、0.5 s 響應(PID-Analyzer 慣例)
STEP_RATE_RESP_S = 0.5
STEP_ANG_WIN_S = 4.0        # 角度層級:4 s 窗、1.5 s 響應
STEP_ANG_RESP_S = 1.5
STEP_FCUT_RATE_HZ = 30.0    # 反卷積低通截止
STEP_FCUT_ANG_HZ = 10.0
STEP_RULE_MIN_HZ_RATE = 100.0   # 角速度層級:≥ 100 Hz 才用步階過衝 / 上升時間下判斷
STEP_RULE_MIN_HZ_ANG = 20.0     # 角度層級:≥ 20 Hz
WIENER_REG = 0.02           # Wiener 正則化:λ = 0.02 × 頻帶內平均 |X|²
STEP_INPUT_DR_DB = 30.0     # 輸入有效頻寬:平均輸入頻譜在峰值 30 dB 以內的最高頻率
STEP_MIN_INPUT_BW_RATE = 8.0    # 角速度步階:輸入頻寬至少 8 Hz(太平滑的輸入反卷積只得到被低通的假響應)
STEP_MIN_INPUT_BW_ANG = 4.0     # 角度步階:至少 4 Hz(ATC_INPUT_TC 平滑後的期望角只有約 3 Hz,
                                #   反卷積出的角度過衝在 2–68 % 亂跳 → 一般搖桿飛行不採用)
STEP_BW_MARGIN = 1.2        # 且 ≥ 1.2 × 共振頻率、≥ −3 dB 頻寬(否則過衝會被低估)
OVERSHOOT_RATE_WARN = 35.0  # %:角速度步階過衝 > 35 % → 阻尼不足(反卷積估計對 10 % 真實過衝會散布在 4–27 %)
OVERSHOOT_ANG_WARN = 20.0   # %:角度步階過衝
OVERSHOOT_SLUGGISH_MAX = 15.0   # 判「偏軟」時過衝需 < 15 %(同上,估計誤差約 ±10 %)
RISE_RATE_SLOW_S = 0.12     # 角速度 10–90 % 上升時間 > 120 ms 偏軟
RISE_ANG_SLOW_S = 0.40      # 角度上升時間 > 400 ms 偏軟

# ---- 偏軟 / 延遲 ----
LAG_RATE_WARN_MS = 75.0     # roll/pitch 角速度:期望 → 實際延遲 > 75 ms 偏軟(調好的中小型機約 40–55 ms)
LAG_RATE_WARN_LOW_MS = 90.0     # 低速記錄(≤ 25 Hz)的 roll/pitch 門檻:10 Hz 取樣的延遲估計偏高約 10 ms、較粗
LAG_YAW_RATE_WARN_MS = 300.0    # 偏航角速度環本來就慢(FLTE 濾波),> 300 ms 才算偏軟
LAG_MIN_RHO_LOW = 0.8       # 低速記錄要建議調高增益時,互相關係數至少 0.8(期望與實際高度相關)
LAG_ANG_WARN_MS = 180.0     # roll/pitch 角度
LAG_YAW_ANG_WARN_MS = 400.0
LAG_MIN_RHO = 0.5           # 互相關係數 < 0.5:延遲估計不可信
LAG_MAX_RATE_S = 0.3        # 互相關搜尋範圍
LAG_MAX_YAW_RATE_S = 1.0    # 偏航角速度:延遲可達數百 ms
LAG_MAX_ANG_S = 0.8
BW_RATE_LOW_HZ = 3.0        # roll/pitch 角速度環頻寬 < 3 Hz 偏軟
BW_YAW_RATE_LOW_HZ = 1.5    # 偏航角速度環頻寬 < 1.5 Hz 偏軟
BW_ANG_LOW_HZ = 1.0         # roll/pitch 角度環頻寬 < 1 Hz 偏軟
TRACK_RATIO_WARN = 0.5      # RMS 誤差 / 期望活動量 > 0.5 → 追蹤差
MIN_ACTIVITY_RATE = 3.0     # deg/s:期望角速度活動量太小時不評「追蹤比」
MIN_ACTIVITY_ANG = 0.5      # deg

# ---- 偏差 / 配平 ----
BIAS_RATE_DES_MAX = 5.0     # deg/s:|期望角速度| 小於此 = 準穩態
BIAS_ANG_RATE_MAX = 5.0     # deg/s:期望角度變化率小於此 = 準穩態
BIAS_MIN_SHARE = 0.2        # 準穩態樣本至少佔 20 % 才計算偏差
BIAS_RATE_WARN = 2.0        # deg/s:準穩態平均角速度誤差 → I 不足 / 機構問題
BIAS_ANG_WARN = 1.0         # deg(roll / pitch)
BIAS_YAW_ANG_WARN = 3.0     # deg
TRIM_OUT_INFO = 0.05        # 懸停平均輸出(−1…1)偏一邊 ≥ 5 % → 提醒檢查重心 / 馬達
TRIM_OUT_WARN = 0.15
I_SAT_FRAC = 0.9            # |I| ≥ 0.9 × IMAX 視為積分飽和
I_SAT_WARN_SHARE = 0.05     # 飛行中 ≥ 5 % 的時間積分飽和

# ---- 飽和 ----
OUT_SAT_LEVEL = 0.9         # RATE.xOut 絕對值 ≥ 0.9 視為該軸輸出飽和
MOTOR_SAT_MARGIN = 0.02     # 馬達 PWM 在上 / 下限(SPIN_MAX / SPIN_MIN)2 % 範圍內視為飽和
SAT_SHARE_WARN = 0.02       # 飛行中 ≥ 2 % 的時間飽和 → 注意(不再建議提高增益)
SAT_SHARE_BAD = 0.10        # ≥ 10 % → 需要處理

# ---- 振動(ArduPilot:< 30 m/s² 一般可接受、> 60 幾乎一定有問題;任何 clipping 都不好)----
VIBE_WARN = 30.0
VIBE_BAD = 60.0
CLIP_WARN = 1
CLIP_BAD = 100

# ---- 懸停油門 ----
HOVER_HIGH = 0.6            # 懸停油門 > 0.6:推力不足、控制裕度小
HOVER_LOW = 0.15            # < 0.15:推力過剩、低油門解析度差
HOVER_MISMATCH = 0.05       # 與 MOT_THST_HOVER 差 > 0.05 且未開 MOT_HOVER_LEARN
HOVER_MAX_CLIMB_CMS = 50.0  # |CTUN.CRt| < 50 cm/s 視為懸停

# ---- CTRL / PIDx ----
CTRL_D_OVER_P_WARN = 1.2    # CTRL D 項 RMS / P 項 RMS > 1.2:誤差能量偏高頻
CTRL_FEST_UPPER_FRAC = 0.8  # D/P 比 ≥ 0.8 × 離散模型最大值:頻率反推病態(差 5 % 就差好幾 Hz)→ 只給下限
DMOD_ACTIVE = 0.9           # Dmod < 0.9:slew 限制器正在壓 P/D(代表有震盪傾向)
DMOD_WARN_SHARE = 0.05
D_HF_FRAC_WARN = 0.5        # D 項功率 ≥ 50 % 在 ≥ 15 Hz:D 項雜訊大,不建議再加 D
D_OVER_P_HIGH = 0.05        # D / P 增益比 > 0.05 視為 D 已偏高(不再建議加 D;震盪時先降 D)
D_DOMINANT_RATIO = 3.0      # PIDx D 項輸出 RMS ≥ 3 × P 項 → D 偏高(調得好的機約 0.5–1.5)
D_DOM_EXPECT_MULT = 1.5     # 同一頻率的誤差下 D/P 輸出比 = (Kd/Kp)·|微分|·|FLTD 低通|(只由增益比與頻率決定);
                            #   量到的 D/P ≥ 1.5 × 這個預期值才代表 D 項多吃了高頻雜訊

# ---- 陷波 / 批次 FFT ----
NOTCH_FMIN_HZ = 20.0        # 馬達噪音峰只在 ≥ 20 Hz 找
NOTCH_PROM_RATIO = 4.0

# ---- 建議幅度(%)----
STEP_D_OSC_HIGH = -25.0     # 高頻震盪:先降 D 25 %
STEP_P_OSC_HIGH = -15.0     # …仍震盪再降 P 15 %
STEP_P_OSC_MID = -20.0      # 中頻震盪:降 P 20 %
STEP_D_OSC_MID = -10.0      # …D 項活動主導時 D 也降 10 %
STEP_SUSPECT_D = -20.0      # 低速記錄的疑似震盪(待確認)
STEP_SUSPECT_P = -15.0
STEP_D_WITH_P = -10.0       # 中頻震盪、D/P 只是頻率造成的偏大:P、I 先降,D 之後一起小降(ArduPilot:P、I、D 一起降)
STEP_ANG_DOWN = -15.0       # 角度環擺動 / 過衝:降 ANG_P 15 %
STEP_P_DAMP = -10.0         # 阻尼不足:降 P 10 %
STEP_D_DAMP = 15.0          # …或加 D 15 %
STEP_D_DAMP_DOWN = -15.0    # 阻尼不足但 D 相對 P 已偏高(D/P > D_OVER_P_HIGH 或 D 項輸出主導):先降 D 15 %
STEP_P_UP = 20.0            # 偏軟:P +20 %(D 按比例)
STEP_P_UP_LOW = 10.0        # 低速記錄判斷的偏軟:只建議 +10 %(證據較弱,先開高速記錄確認)
STEP_ANG_UP = 15.0          # 角度環偏慢:ANG_P +15 %
# ATC_INPUT_TC 不列為修正項:ATT.DesRoll / DesPitch 是 INPUT_TC 整形「之後」的目標,
# 量到的期望 → 實際響應與 INPUT_TC 無關(它只改變搖桿手感)
STEP_I_UP = 25.0            # 持續偏差:I +25 %
STEP_YAW_P_UP = 20.0
STEP_YAW_P_DOWN = -20.0
STEP_ACCEL_DOWN = -20.0     # 輸出飽和:降 ATC_ACCEL_*_MAX 20 %
STEP_SLEW_D = -20.0         # slew 限制器常啟動:降 D 20 %

PLOT_MAX_POINTS = 200_000   # 給 UI 的時間序列最多這麼多點(只影響繪圖,分析用完整資料)

# 參數合理範圍(ArduPilot 文件的 Range;INPUT_TC 取文件標示的 Very Crisp…Very Soft 實用範圍)。
# Q_A_ 開頭的 QuadPlane 參數對應同名 ATC_。
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "ATC_RAT_RLL_P": (0.01, 0.5), "ATC_RAT_PIT_P": (0.01, 0.5), "ATC_RAT_YAW_P": (0.10, 2.5),
    "ATC_RAT_RLL_I": (0.01, 2.0), "ATC_RAT_PIT_I": (0.01, 2.0), "ATC_RAT_YAW_I": (0.01, 1.0),
    "ATC_RAT_RLL_D": (0.0, 0.05), "ATC_RAT_PIT_D": (0.0, 0.05), "ATC_RAT_YAW_D": (0.0, 0.02),
    "ATC_RAT_RLL_IMAX": (0.0, 1.0), "ATC_RAT_PIT_IMAX": (0.0, 1.0), "ATC_RAT_YAW_IMAX": (0.0, 1.0),
    "ATC_RAT_RLL_FLTD": (5.0, 100.0), "ATC_RAT_PIT_FLTD": (5.0, 100.0), "ATC_RAT_YAW_FLTD": (0.0, 100.0),
    "ATC_RAT_RLL_FLTT": (5.0, 100.0), "ATC_RAT_PIT_FLTT": (5.0, 100.0), "ATC_RAT_YAW_FLTT": (5.0, 100.0),
    "ATC_RAT_RLL_FLTE": (0.0, 100.0), "ATC_RAT_PIT_FLTE": (0.0, 100.0), "ATC_RAT_YAW_FLTE": (0.0, 100.0),
    "ATC_ANG_RLL_P": (3.0, 12.0), "ATC_ANG_PIT_P": (3.0, 12.0), "ATC_ANG_YAW_P": (3.0, 12.0),
    "ATC_INPUT_TC": (0.05, 0.5),
    "ATC_ACCEL_R_MAX": (0.0, 180000.0), "ATC_ACCEL_P_MAX": (0.0, 180000.0),
    "ATC_ACCEL_Y_MAX": (0.0, 72000.0),
    "INS_GYRO_FILTER": (0.0, 256.0),
    "INS_HNTCH_FREQ": (10.0, 495.0), "INS_HNTCH_BW": (5.0, 250.0), "INS_HNTCH_REF": (0.0, 1.0),
    "INS_HNTCH_ATT": (5.0, 50.0),
    "MOT_THST_HOVER": (0.125, 0.6875), "MOT_YAW_HEADROOM": (0.0, 500.0),
}

# 調參安全須知(放進 report.notes)
SAFETY_NOTES = [
    "一次只改一個參數,每次 10–25 %;改完先在空曠安全處以 STABILIZE / ALT_HOLD 低空試飛,再看新的 log。",
    "先處理振動與陷波濾波,再調 PID;不確定時優先使用 AutoTune(ArduPilot 調參流程:初始參數 → 陷波 → AutoTune / 手動微調)。",
    "建議值只是方向,不是保證;若改了之後更糟,立刻改回原值。",
]

# 訊息欄位對照
_RATE_FIELDS = {"roll": ("RDes", "R", "ROut"), "pitch": ("PDes", "P", "POut"),
                "yaw": ("YDes", "Y", "YOut")}
_ATT_FIELDS = {"roll": ("DesRoll", "Roll"), "pitch": ("DesPitch", "Pitch"), "yaw": ("DesYaw", "Yaw")}
_IMU_FIELDS = {"roll": "GyrX", "pitch": "GyrY", "yaw": "GyrZ"}
_CTRL_FIELDS = {"roll": ("RMSRollP", "RMSRollD"), "pitch": ("RMSPitchP", "RMSPitchD"),
                "yaw": ("RMSYaw", None)}
_PID_MSG = {"roll": "PIDR", "pitch": "PIDP", "yaw": "PIDY"}
_PIQ_MSG = {"roll": "PIQR", "pitch": "PIQP", "yaw": "PIQY"}      # QuadPlane 的 VTOL 角速度 PID
_ACC_AX = {"roll": "R", "pitch": "P", "yaw": "Y"}
_AXIS_SHORT = {"roll": "Roll", "pitch": "Pitch", "yaw": "Yaw"}

FRAME_CLASS_NAMES = {1: "quad", 2: "hexa", 3: "octa", 4: "octaquad", 5: "y6", 6: "heli", 7: "tri",
                     8: "single", 9: "coax", 10: "bicopter", 11: "heli dual", 12: "dodecahexa",
                     13: "heliquad", 14: "deca", 15: "scripting", 16: "6dof", 17: "dynamic"}
FRAME_TYPE_NAMES = {0: "+", 1: "X", 2: "V", 3: "H", 4: "V-tail", 5: "A-tail", 10: "Y6B", 11: "Y6F",
                    12: "BF-X", 13: "DJI-X", 14: "CW-X", 15: "I", 18: "BF-X rev", 19: "Y4"}
_MOTOR_COUNT = {1: 4, 2: 6, 3: 8, 4: 8, 5: 6, 7: 3, 9: 2, 10: 2, 12: 12, 14: 10}
_HELI_CLASSES = (6, 11, 13)
_DEFAULT_RP = (0.135, 0.135, 0.0036)     # ArduCopter 4.x roll/pitch P, I, D 預設
_DEFAULT_YAW = (0.18, 0.018)


# =====================================================================
# 訊號處理(純 numpy)
# =====================================================================

def hann(n: int) -> np.ndarray:
    """週期型 Hann 窗(頻譜分析用,同 scipy.signal.get_window('hann', n))。"""
    n = int(n)
    if n <= 1:
        return np.ones(max(n, 0))
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def choose_nperseg(fs: float, max_len: int, win_s: float = WELCH_WIN_S,
                   max_n: int = WELCH_MAX_N) -> int:
    """Welch 窗長(2 的次方):fs × win_s 向下取 2 的次方,夾在 [WELCH_MIN_N, max_n],且不超過最長片段。"""
    n = 2 ** int(np.floor(np.log2(max(fs * win_s, 2.0))))
    n = int(min(max(n, WELCH_MIN_N), max_n))
    while n > max(int(max_len), 0) and n > 8:
        n //= 2
    return n


def _pieces(x) -> list[np.ndarray]:
    """單一陣列或「多段陣列的 list」→ list[1D float 陣列]。"""
    if isinstance(x, (list, tuple)):
        return [np.asarray(p, dtype=float).ravel() for p in x]
    return [np.asarray(x, dtype=float).ravel()]


def _detrend_rows(seg: np.ndarray, kind: str) -> np.ndarray:
    if kind == "linear":
        n = seg.shape[-1]
        tt = np.arange(n) - (n - 1) / 2.0
        slope = seg @ tt / max(float(tt @ tt), 1e-12)
        return seg - seg.mean(axis=-1, keepdims=True) - slope[:, None] * tt[None, :]
    if kind in ("mean", "constant"):
        return seg - seg.mean(axis=-1, keepdims=True)
    return seg


@dataclass
class Frames:
    """Welch 分窗結果(多段訊號共用)。|X|² × scale = 該窗的單邊 PSD。"""
    f: np.ndarray
    X: np.ndarray                 # (窗數, 頻點) 已乘窗的 rFFT
    Y: np.ndarray | None          # 第二個訊號(CSD / 同調用)
    piece: np.ndarray             # 每窗來自第幾段
    center: np.ndarray            # 窗中心在該段內的樣本位置
    scale: np.ndarray             # (頻點,)
    nperseg: int
    fs: float

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def df(self) -> float:
        return float(self.fs / self.nperseg)


def spectral_frames(x, fs: float, nperseg: int, y=None, overlap: float = WELCH_OVERLAP,
                    detrend: str = "mean") -> Frames:
    """把(多段)訊號切成重疊的 Hann 窗,回傳每窗的 rFFT(與 y 的)。

    x / y:1D 陣列或多段陣列的 list(各段長度需對應);短於 nperseg 的段略過。
    """
    xs = _pieces(x)
    ys = _pieces(y) if y is not None else None
    nperseg = int(nperseg)
    w = hann(nperseg)
    step = max(1, int(round(nperseg * (1.0 - overlap))))
    lens = [xp.size if ys is None else min(xp.size, ys[k].size) for k, xp in enumerate(xs)]
    total = sum((n - nperseg) // step + 1 for n in lens if n >= nperseg)
    if total > WELCH_MAX_FRAMES:
        step *= int(math.ceil(total / WELCH_MAX_FRAMES))
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    xl, yl, pl, cl = [], [], [], []
    for k, xp in enumerate(xs):
        n = xp.size if ys is None else min(xp.size, ys[k].size)
        if n < nperseg:
            continue
        starts = np.arange(0, n - nperseg + 1, step)
        for c0 in range(0, starts.size, 2048):          # 分批,長 log 也不會一次吃太多記憶體
            st = starts[c0:c0 + 2048]
            idx = st[:, None] + np.arange(nperseg)[None, :]
            xl.append(np.fft.rfft(_detrend_rows(xp[idx], detrend) * w, axis=1))
            if ys is not None:
                yl.append(np.fft.rfft(_detrend_rows(ys[k][idx], detrend) * w, axis=1))
            pl.append(np.full(st.size, k))
            cl.append(st + nperseg / 2.0)
    X = np.vstack(xl) if xl else np.zeros((0, f.size), complex)
    Y = (np.vstack(yl) if yl else np.zeros((0, f.size), complex)) if ys is not None else None
    scale = np.full(f.size, 2.0 / (fs * float(np.sum(w ** 2))))
    scale[0] /= 2.0
    if nperseg % 2 == 0:
        scale[-1] /= 2.0
    return Frames(f, X, Y, np.concatenate(pl) if pl else np.zeros(0, int),
                  np.concatenate(cl) if cl else np.zeros(0), scale, nperseg, float(fs))


def _auto_nperseg(xs: list[np.ndarray], fs: float, nperseg: int | None) -> int:
    if nperseg is not None:
        return int(nperseg)
    return choose_nperseg(fs, max((p.size for p in xs), default=0))


def welch_psd(x, fs: float, nperseg: int | None = None, overlap: float = WELCH_OVERLAP,
              detrend: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Welch 單邊功率譜密度(單位²/Hz);x 可為多段 list。資料不足回傳全 NaN。

    正弦 A·sin → 峰下面積 = A²/2;白雜訊變異數 σ² → 平坦位準 2σ²/fs。
    """
    xs = _pieces(x)
    fr = spectral_frames(xs, fs, _auto_nperseg(xs, fs, nperseg), overlap=overlap, detrend=detrend)
    if fr.n == 0:
        return fr.f, np.full(fr.f.size, np.nan)
    return fr.f, np.mean(np.abs(fr.X) ** 2, axis=0) * fr.scale


def csd(x, y, fs: float, nperseg: int | None = None, overlap: float = WELCH_OVERLAP,
        detrend: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Welch 交叉譜 Pxy = E[conj(X)·Y](同 scipy.signal.csd 的慣例)。"""
    xs = _pieces(x)
    fr = spectral_frames(xs, fs, _auto_nperseg(xs, fs, nperseg), y=_pieces(y), overlap=overlap,
                         detrend=detrend)
    if fr.n == 0:
        return fr.f, np.full(fr.f.size, np.nan, dtype=complex)
    return fr.f, np.mean(np.conj(fr.X) * fr.Y, axis=0) * fr.scale


def _spectra(fr: Frames) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pxx = np.mean(np.abs(fr.X) ** 2, axis=0) * fr.scale
    pyy = np.mean(np.abs(fr.Y) ** 2, axis=0) * fr.scale
    pxy = np.mean(np.conj(fr.X) * fr.Y, axis=0) * fr.scale
    return pxx, pyy, pxy


def _coh(pxx, pyy, pxy) -> np.ndarray:
    den = pxx * pyy
    out = np.zeros(pxx.shape)
    ok = den > 0
    out[ok] = np.abs(pxy[ok]) ** 2 / den[ok]
    return np.clip(out, 0.0, 1.0)


def coherence(x, y, fs: float, nperseg: int | None = None, overlap: float = WELCH_OVERLAP,
              detrend: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """幅度平方同調 |Pxy|² / (Pxx·Pyy),0…1。"""
    xs = _pieces(x)
    fr = spectral_frames(xs, fs, _auto_nperseg(xs, fs, nperseg), y=_pieces(y), overlap=overlap,
                         detrend=detrend)
    if fr.n == 0:
        return fr.f, np.full(fr.f.size, np.nan)
    return fr.f, _coh(*_spectra(fr))


def band_rms(f: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    """PSD 在 [f_lo, f_hi] 的積分開根號 = 該頻帶的 RMS 振幅。"""
    f = np.asarray(f, float)
    p = np.asarray(psd, float)
    if f.size < 2:
        return 0.0
    m = (f >= f_lo) & (f <= f_hi) & np.isfinite(p)
    return float(np.sqrt(max(np.sum(p[m]) * (f[1] - f[0]), 0.0)))


def _running_median(p: np.ndarray, half: int) -> np.ndarray:
    """±half 點的滑動中位數;靠近兩端時視窗往內移(不鏡射,避免把邊緣的峰複製進底噪)。"""
    n = p.size
    w = int(min(2 * max(1, int(half)) + 1, n))
    if w < 1:
        return p.copy()
    med = np.median(np.lib.stride_tricks.sliding_window_view(p, w), axis=1)
    start = np.clip(np.arange(n) - w // 2, 0, n - w)
    return med[start]


class Peak(NamedTuple):
    """頻譜峰:頻率、峰的 RMS 振幅(扣掉底噪)、突出比、峰範圍、計算 RMS 用的底噪位準。"""
    f: float
    rms: float
    prominence: float
    f_lo: float
    f_hi: float
    floor: float = 0.0


def find_peaks(f, psd, fmin: float = 0.0, fmax: float | None = None,
               prom_ratio: float = PEAK_PROM_RATIO, min_rms: float = 0.0,
               base_hz: float = PEAK_BASE_HZ, edge_ratio: float = PEAK_EDGE_RATIO,
               max_peaks: int = 5, med_ratio: float = PEAK_MED_RATIO) -> list[Peak]:
    """PSD 峰值偵測(依 RMS 由大到小)。

    突出比 = 峰值 / 左右 ±base_hz 內「兩個谷底中較高者」(同 scipy 的 prominence 概念:
    寬峰不會墊高自己的底噪,斜坡上的小起伏不算峰);另需 ≥ med_ratio × 滑動中位數(排除雜訊起伏)。
    峰的範圍 = 連續高於 edge_ratio × 谷底的頻帶;RMS = √(Σ(PSD − 谷底)·df)。
    頻率以 log-PSD 拋物線內插細化。
    """
    f = np.asarray(f, float)
    p = np.asarray(psd, float)
    n = p.size
    if n < 5 or f.size != n or not np.isfinite(p).any():
        return []
    p = np.where(np.isfinite(p), np.maximum(p, 0.0), 0.0)
    df = float(f[1] - f[0])
    h = max(3, int(round(base_hz / df)))
    med = _running_median(p, h)
    tiny = 1e-30
    fmax = float(f[-1]) if fmax is None else float(fmax)
    ok = (f >= fmin) & (f <= fmax)
    k_all = np.arange(1, n - 1)
    is_max = (p[1:-1] >= p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] > 0)
    cand = k_all[is_max & ok[1:-1] & (p[1:-1] >= med_ratio * np.maximum(med[1:-1], tiny))]
    used = np.zeros(n, bool)
    out: list[Peak] = []
    for k in sorted(cand.tolist(), key=lambda i: -p[i]):
        if used[k]:
            continue
        left = p[max(1, k - h):k]                     # 不含 DC(已去平均,恆小,會讓紅雜訊的第一格像峰)
        vl = float(left.min()) if left.size else float(p[k])
        vr = float(p[k + 1:k + h + 1].min()) if k + 1 < n else float(p[k])
        ref = max(vl, vr, tiny)
        if p[k] < prom_ratio * ref:
            continue
        lo = k
        while lo - 1 >= 1 and p[lo - 1] > edge_ratio * ref:
            lo -= 1
        hi = k
        while hi + 1 <= n - 1 and p[hi + 1] > edge_ratio * ref:
            hi += 1
        lo, hi = min(lo, k - 1), max(hi, min(k + 1, n - 1))
        if used[lo:hi + 1].any():
            continue
        used[lo:hi + 1] = True
        rms = float(np.sqrt(np.sum(np.maximum(p[lo:hi + 1] - ref, 0.0)) * df))
        fk = float(f[k])
        if p[k - 1] > 0 and p[k + 1] > 0:
            a, b, c = np.log(p[k - 1]), np.log(p[k]), np.log(p[k + 1])
            den = a - 2.0 * b + c
            if den < 0:
                fk += float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) * df
        if rms >= min_rms:
            out.append(Peak(fk, rms, float(p[k] / ref), float(f[lo]), float(f[hi]), ref))
    out.sort(key=lambda q: -q.rms)
    return out[:max_peaks]


def freq_response(x, y, fs: float, nperseg: int | None = None, coh_min: float = COH_MIN,
                  fmax: float | None = None, frames: Frames | None = None,
                  detrend: str = "mean", h_mult: np.ndarray | None = None) -> FreqResponse:
    """期望 x → 實際 y 的頻率響應 H = Pxy / Pxx(H1 估計)與同調。

    只用同調 ≥ coh_min 的頻點找共振峰 |H|max 與 −3 dB 頻寬(|H| 首次掉到 0.707);
    頻寬在可信範圍內都沒掉下去 → bandwidth_hz = None。
    h_mult:每個頻點再乘上的複數響應(例:PIDx.Tar 已經過 FLTT 低通,乘回 LPF 換算成「未濾波目標 → 實際」)。
    """
    if frames is None:
        xs = _pieces(x)
        frames = spectral_frames(xs, fs, _auto_nperseg(xs, fs, nperseg), y=_pieces(y),
                                 detrend=detrend)
    f = frames.f
    if frames.n == 0 or frames.Y is None:
        z = np.full(f.size, np.nan)
        return FreqResponse(f, z, z.copy(), z.copy())
    pxx, pyy, pxy = _spectra(frames)
    h = np.full(f.size, np.nan, dtype=complex)
    okx = pxx > 0
    h[okx] = pxy[okx] / pxx[okx]
    if h_mult is not None:
        h = h * np.asarray(h_mult, dtype=complex)
    coh = _coh(pxx, pyy, pxy)
    mag = np.abs(h)
    phase = np.degrees(np.angle(h))
    nyq = fs / 2.0
    fmax = PEAK_FMAX_FRAC * nyq if fmax is None else min(fmax, nyq)
    valid = (f > 0) & (f <= fmax) & (coh >= coh_min) & np.isfinite(mag)
    if frames.n < COH_MIN_FRAMES:
        valid[:] = False
    fr = FreqResponse(f, mag, phase, coh)
    vi = np.flatnonzero(valid)
    if vi.size:
        pk = vi[f[vi] >= max(frames.df, PEAK_FMIN_HZ)]
        if pk.size:
            j = pk[int(np.argmax(mag[pk]))]
            fr.peak_mag, fr.peak_hz = float(mag[j]), float(f[j])
        below = np.flatnonzero(mag[vi] < 1.0 / np.sqrt(2.0))
        if below.size and below[0] > 0:
            i1 = vi[below[0]]
            i0 = vi[below[0] - 1]
            if i1 - i0 == 1 and mag[i0] > mag[i1]:
                w = (mag[i0] - 1 / np.sqrt(2.0)) / (mag[i0] - mag[i1])
                fr.bandwidth_hz = float(f[i0] + w * (f[i1] - f[i0]))
            else:
                fr.bandwidth_hz = float(f[i1])
    return fr


def _bw_lower_bound(fr: FreqResponse, coh_min: float = COH_MIN) -> float | None:
    """頻寬沒算出來、但可信頻點的 |H| 都 ≥ 0.707 → 頻寬至少是最高可信頻率。"""
    if fr.bandwidth_hz is not None or fr.coherence is None:
        return None
    m = (fr.f > 0) & (fr.coherence >= coh_min) & np.isfinite(fr.mag)
    if not m.any() or (fr.mag[m] < 1 / np.sqrt(2.0)).any():
        return None
    return float(fr.f[m].max())


def _bw_upper_bound(fr: FreqResponse, n_frames: int, coh_min: float = COH_MIN) -> float | None:
    """最低的可信頻點 |H| 就已 < 0.707 → 頻寬低於該頻率(回傳該頻率);否則 None。"""
    if fr.bandwidth_hz is not None or fr.coherence is None or n_frames < COH_MIN_FRAMES:
        return None
    m = np.flatnonzero((fr.f > 0) & (fr.coherence >= coh_min) & np.isfinite(fr.mag))
    if not m.size or fr.mag[m[0]] >= 1 / np.sqrt(2.0):
        return None
    return float(fr.f[m[0]])


def lag_xcorr(x, y, fs: float, max_lag_s: float = 0.5) -> tuple[float | None, float]:
    """互相關估計 y 落後 x 的時間(秒,正 = y 較晚)與該處的相關係數。

    x / y 可為多段 list(各段相關函數相加);峰值以拋物線內插到次樣本。
    峰落在搜尋邊界 → (None, 相關係數)。
    """
    xs, ys = _pieces(x), _pieces(y)
    K = max(1, int(round(max_lag_s * fs)))
    acc = np.zeros(2 * K + 1)
    sxx = syy = 0.0
    for xp, yp in zip(xs, ys):
        n = min(xp.size, yp.size)
        if n < 8:
            continue
        a = xp[:n] - xp[:n].mean()
        b = yp[:n] - yp[:n].mean()
        nfft = 1 << int(np.ceil(np.log2(n + K + 1)))
        c = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(b, nfft), nfft)
        kk = min(K, n - 1)
        seg = np.zeros(2 * K + 1)
        seg[K - kk:K] = c[nfft - kk:]
        seg[K:K + kk + 1] = c[:kk + 1]
        acc += seg
        sxx += float(a @ a)
        syy += float(b @ b)
    if sxx <= 0 or syy <= 0:
        return None, 0.0
    r = acc / np.sqrt(sxx * syy)
    k = int(np.argmax(r))
    rho = float(r[k])
    if k == 0 or k == 2 * K:
        return None, rho
    a, b, c = r[k - 1], r[k], r[k + 1]
    den = a - 2.0 * b + c
    delta = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0
    return (k - K + delta) / fs, rho


def step_metrics(t: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """正規化步階響應(穩態 1)→ (過衝 %, 10–90 % 上升時間 s, 進入 ±5 % 不再出來的時間 s)。"""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t.size < 3 or not np.isfinite(y).all():
        return None, None, None
    over = float(max(0.0, (np.max(y) - 1.0) * 100.0))

    def cross(level):
        i = np.flatnonzero(y >= level)
        if not i.size:
            return None
        i = int(i[0])
        if i == 0:
            return float(t[0])
        w = (level - y[i - 1]) / (y[i] - y[i - 1]) if y[i] != y[i - 1] else 0.0
        return float(t[i - 1] + w * (t[i] - t[i - 1]))

    t10, t90 = cross(0.1), cross(0.9)
    rise = (t90 - t10) if (t10 is not None and t90 is not None) else None
    out = np.flatnonzero(np.abs(y - 1.0) > 0.05)
    if not out.size:
        settle = float(t[0])
    elif out[-1] + 1 < t.size:
        settle = float(t[out[-1] + 1])
    else:
        settle = None
    return over, rise, settle


def step_response_wiener(x, y, fs: float, win_s: float = STEP_RATE_WIN_S,
                         resp_s: float = STEP_RATE_RESP_S, min_input: float | None = None,
                         f_cut: float | None = None, reg: float = WIENER_REG,
                         hop_frac: float = 0.25, min_input_bw: float | None = None) -> StepResponse:
    """由期望 x / 實際 y 以加窗 Wiener 反卷積估步階響應(PID-Analyzer 做法)。

    每個時間窗(Hann)求 H = Y·X* / (|X|² + λ),λ 在截止頻率以上急遽變大(低通);
    脈衝響應累加 = 步階響應;只用輸入有活動的窗(標準差 ≥ min_input,預設 = 各窗中位數),
    平均後以尾段(後 40 %)平均值正規化成穩態 1。資料不足回傳空的 StepResponse。

    輸入太平滑時(平均輸入頻譜在峰值 STEP_INPUT_DR_DB 以內的最高頻率 < min_input_bw)
    反卷積只會得到被低通的假響應(上升變慢、甚至假過衝),此時回傳空的 StepResponse
    (n_windows 仍填入)。
    """
    st, _ = _step_core(x, y, fs, win_s, resp_s, min_input, f_cut, reg, hop_frac, min_input_bw)
    return st


def _step_core(x, y, fs, win_s, resp_s, min_input, f_cut, reg, hop_frac, min_input_bw):
    """step_response_wiener 本體;另回傳輸入有效頻寬(Hz)。"""
    xs, ys = _pieces(x), _pieces(y)
    R = int(round(resp_s * fs))
    L = max(int(round(win_s * fs)), 2 * R)
    empty = StepResponse(np.zeros(0), np.zeros(0))
    if R < STEP_MIN_SAMPLES:
        return empty, None
    hop = max(1, int(L * hop_frac))
    total = sum((min(xp.size, yp.size) - L) // hop + 1 for xp, yp in zip(xs, ys)
                if min(xp.size, yp.size) >= L)
    if total > STEP_MAX_WINDOWS:
        hop *= int(math.ceil(total / STEP_MAX_WINDOWS))
    xw, yw = [], []
    for xp, yp in zip(xs, ys):
        n = min(xp.size, yp.size)
        if n < L:
            continue
        st = np.arange(0, n - L + 1, hop)
        idx = st[:, None] + np.arange(L)[None, :]
        xw.append(xp[idx])
        yw.append(yp[idx])
    if not xw:
        return empty, None
    Xw = np.vstack(xw)
    Yw = np.vstack(yw)
    act = Xw.std(axis=1)
    thr = float(np.median(act)) if min_input is None else float(min_input)
    sel = (act >= thr) & (act > 0)
    if int(sel.sum()) < STEP_MIN_WINDOWS:
        empty.n_windows = int(sel.sum())
        return empty, None
    Xw, Yw = Xw[sel], Yw[sel]
    w = np.hanning(L)
    # 去掉 Hann 加權平均:實際值的固定偏差(重心偏移、I 不足)只會落在 DC,不能讓它扭曲步階;
    # DC 因此沒有資訊,之後以最低頻點外插 H(0)
    Xw = Xw - ((Xw * w).sum(axis=1) / w.sum())[:, None]
    Yw = Yw - ((Yw * w).sum(axis=1) / w.sum())[:, None]
    nfft = 1 << int(np.ceil(np.log2(2 * L)))
    X = np.fft.rfft(Xw * w, nfft, axis=1)
    Y = np.fft.rfft(Yw * w, nfft, axis=1)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    nyq = fs / 2.0
    fc = min(f_cut if f_cut else nyq, PEAK_FMAX_FRAC * nyq)
    taper = np.clip((fc - f) / (0.3 * fc), 0.0, 1.0)            # 0.7 fc 以下 = 1,fc 以上 = 0
    taper = 0.5 - 0.5 * np.cos(np.pi * taper)
    sxx = np.abs(X) ** 2
    inband = (f <= fc) & (f > 0)
    lam = reg * np.mean(sxx[:, inband], axis=1, keepdims=True)
    # 輸入有效頻寬:平均輸入頻譜仍在峰值 STEP_INPUT_DR_DB 以內的最高頻率
    avg = np.mean(sxx, axis=0)
    okf = np.flatnonzero(inband & (avg >= np.max(avg[inband]) * 10 ** (-STEP_INPUT_DR_DB / 10)))
    f_eff = float(f[okf[-1]]) if okf.size else 0.0
    if min_input_bw is not None and f_eff < min_input_bw:
        empty.n_windows = int(sel.sum())
        return empty, f_eff
    lam = lam * (1.0 + 1e4 * (1.0 - taper))[None, :]
    H = Y * np.conj(X) / (sxx + np.maximum(lam, 1e-30))
    H[:, 0] = H[:, 1].real
    h = np.fft.irfft(H, nfft, axis=1)[:, :R]
    steps = np.cumsum(h, axis=1)
    yavg = steps.mean(axis=0)
    t = np.arange(R) / fs
    yss = float(np.mean(yavg[int(0.6 * R):]))
    yn = yavg / yss if 0.2 < yss < 5.0 else yavg
    over, rise, settle = step_metrics(t, yn) if 0.2 < yss < 5.0 else (None, None, None)
    return StepResponse(t, yn, over, rise, settle, int(sel.sum())), f_eff


# =====================================================================
# 時間區間工具
# =====================================================================

Interval = tuple[float, float]


def _merge(iv: list[Interval], gap: float = 0.0) -> list[Interval]:
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in iv if b > a):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    out = []
    for x0, x1 in a:
        for y0, y1 in b:
            lo, hi = max(x0, y0), min(x1, y1)
            if hi > lo:
                out.append((lo, hi))
    return _merge(out)


def _subtract(a: list[Interval], b: list[Interval]) -> list[Interval]:
    out = list(a)
    for y0, y1 in b:
        nxt = []
        for x0, x1 in out:
            if y1 <= x0 or y0 >= x1:
                nxt.append((x0, x1))
                continue
            if y0 > x0:
                nxt.append((x0, y0))
            if y1 < x1:
                nxt.append((y1, x1))
        out = nxt
    return _merge(out)


def _bool_intervals(t: np.ndarray, m: np.ndarray, max_gap: float) -> list[Interval]:
    """布林遮罩 → 連續為真的時間區間(相鄰樣本間隔 > max_gap 也切開)。"""
    t = np.asarray(t, float)
    m = np.asarray(m, bool)
    if not t.size or not m.any():
        return []
    idx = np.flatnonzero(m)
    brk = np.flatnonzero((np.diff(idx) > 1) | (np.diff(t[idx]) > max_gap)) + 1
    return [(float(t[idx[s]]), float(t[idx[e - 1]]))
            for s, e in zip(np.r_[0, brk], np.r_[brk, idx.size]) if e - 1 >= s]


def _in_intervals(t: np.ndarray, iv: list[Interval]) -> np.ndarray:
    t = np.asarray(t, float)
    m = np.zeros(t.shape, bool)
    for a, b in iv:
        m |= (t >= a) & (t <= b)
    return m


def _total(iv: list[Interval]) -> float:
    return float(sum(b - a for a, b in iv))


# =====================================================================
# 飛行遮罩
# =====================================================================

_TAKEOFF_EV = ("NOT_LANDED",)
_LAND_EV = ("LAND_COMPLETE_MAYBE", "LAND_COMPLETE")


def _segments(logdata, lo: float, hi: float) -> list[Interval]:
    from .timeline import arm_segments
    segs = []
    try:
        if logdata.sources:
            segs = [(s.t0, s.t1) for s in logdata.segments]
        else:
            segs = [(s.t0, s.t1) for s in arm_segments(logdata.events, lo, hi)]
    except Exception:  # noqa: BLE001 — 事件壞掉就當作沒有 ARM 資訊
        segs = []
    return segs


def _thr_table(index, quadplane: bool = False):
    """油門(0…1)/ 高度 / 爬升率的來源 → (表格或 None, 訊息名)。

    Copter:CTUN。QuadPlane:QTUN(VTOL 升力油門 0…1、Alt、CRt)—— Plane 的 CTUN.ThO 是固定翼推進油門
    (−100…100 %,VTOL 懸停時 ≈ 0)而且沒有 Alt / CRt,不能拿來判斷懸停或是否在空中。
    """
    name = "QTUN" if quadplane else "CTUN"
    tbl = index.table(name) if index is not None else None
    if tbl is None or "ThO" not in tbl:
        return None, name
    return tbl, name


def _throttle_airborne(index, a: float, b: float, hover: float | None,
                       quadplane: bool = False) -> list[Interval] | None:
    """沒有起降事件時:油門 + 高度判斷在空中。沒有油門表(CTUN / QuadPlane 的 QTUN)→ None(無法判斷)。"""
    tbl, _ = _thr_table(index, quadplane)
    if tbl is None:
        return None
    t = tbl["t"]
    m = (t >= a) & (t <= b)
    if m.sum() < 3:
        return []
    tt = t[m]
    tho = np.asarray(tbl["ThO"], float)[m]
    thr = max(FLIGHT_THR_MIN, FLIGHT_THR_FRAC * hover) if hover else FLIGHT_THR_MIN
    air = tho > thr
    if "Alt" in tbl:
        alt = np.asarray(tbl["Alt"], float)[m]
        if np.isfinite(alt).any():
            air &= (alt - np.nanmin(alt)) > FLIGHT_ALT_MIN_M
    iv = _bool_intervals(tt, air, max_gap=1.0)
    return [(x, y) for x, y in _merge(iv, gap=1.0) if y - x >= 1.0]


def flight_intervals(logdata, t0: float, t1: float, quadplane: bool = False,
                     hover: float | None = None) -> tuple[list[Interval], list[str]]:
    """[t0, t1] 內「在空中、可分析」的時間區間 + 說明。

    先在整份 log 上找:ARM 段 ∩(EV NOT_LANDED … LAND_COMPLETE(_MAYBE));該段沒有起降事件時改用
    CTUN 油門 + 高度。起飛後 / 落地前各裁 TRIM_*_S 秒(只裁真正的起降,不裁分析範圍的邊界),
    排除 AUTOTUNE / FLIP(QuadPlane 只留 Q 開頭的 VTOL 模式),最後才切到 [t0, t1],
    丟掉短於 MIN_PIECE_S 的片段。
    """
    from .timeline import mode_changes
    from .types import EV_EVENT
    notes: list[str] = []
    index = getattr(logdata, "index", None)
    lo, hi = float(t0), float(t1)
    if index is not None:
        r0, r1 = index.t_range()
        lo, hi = min(lo, r0), max(hi, r1)
    segs = _segments(logdata, lo, hi)
    if not segs:
        segs = [(lo, hi)]
        notes.append("log 沒有 ARM / DISARM 事件:以整個範圍搭配油門 / 高度判斷是否在空中。")
    evs = sorted((e for e in logdata.events if e.kind == EV_EVENT
                  and e.label in _TAKEOFF_EV + _LAND_EV), key=lambda e: e.t)
    air: list[Interval] = []
    used_thr = False
    no_thr = False
    for a, b in segs:
        inside = [e for e in evs if a <= e.t <= b]
        if inside:
            start = None
            first = True
            for e in inside:
                if e.label in _TAKEOFF_EV:
                    if start is None:
                        start = e.t
                elif e.label in _LAND_EV:
                    if start is not None:
                        air.append((start, e.t))
                        start = None
                    elif first:                    # 第一個事件就是落地:之前已在空中(log 從空中開始)
                        air.append((a, e.t))
                first = False
            if start is not None:
                air.append((start, b))
            continue
        thr = _throttle_airborne(index, a, b, hover, quadplane)
        if thr is None:
            no_thr = True
            air.append((a, b))
        else:
            used_thr = True
            air.extend(thr)
    if used_thr:
        notes.append("部分飛行段沒有起降事件(EV NOT_LANDED / LAND_COMPLETE),改用"
                     + ("QTUN 升力油門" if quadplane else "油門") + "與高度判斷。")
    if no_thr and quadplane:
        notes.append("QuadPlane 沒有起降事件也沒有 QTUN(VTOL 升力油門):以整個解鎖段 + VTOL 模式當作飛行"
                     "(Plane 的 CTUN.ThO 是推進油門,不能用來判斷)。")
    air = [(a + TRIM_TAKEOFF_S, b - TRIM_LAND_S) for a, b in _merge(air)
           if b - a > TRIM_TAKEOFF_S + TRIM_LAND_S]
    # 模式:排除 AUTOTUNE / FLIP;QuadPlane 只留 VTOL(Q*)模式
    modes = mode_changes(logdata.events)
    if modes:
        bad = []
        for i, m in enumerate(modes):
            end = modes[i + 1].t if i + 1 < len(modes) else math.inf
            name = (m.label or "").upper()
            if name in EXCLUDE_MODES or (quadplane and not name.startswith("Q")):
                bad.append((m.t, end))
        if bad:
            before = _total(_intersect(air, [(t0, t1)]))
            air = _subtract(air, bad)
            cut = before - _total(_intersect(air, [(t0, t1)]))
            if cut > 0.5:
                why = "非 VTOL 模式 / AUTOTUNE" if quadplane else "AUTOTUNE / FLIP 模式"
                notes.append(f"排除 {why} 的 {cut:.0f} s(增益在變或是刻意動作)。")
    out = [(a, b) for a, b in _intersect(air, [(float(t0), float(t1))]) if b - a >= MIN_PIECE_S]
    return out, notes


# =====================================================================
# 資料存取
# =====================================================================

def _fs_of(t) -> float:
    t = np.asarray(t, float)
    if t.size < 3:
        return 0.0
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    if not d.size:
        return 0.0
    return float(1.0 / np.median(d))


def _low_fs(fs: float | None) -> bool:
    """取樣率屬於「低速記錄」(≤ LOW_RATE_HZ;容許 10 % 計時誤差,25.05 Hz 也算 25 Hz)。"""
    return fs is None or fs < LOW_RATE_HZ * 1.1


def _fs_ok(fs: float | None, need: float) -> bool:
    """取樣率至少約 need(容許 5 % 計時誤差)。"""
    return fs is not None and fs >= need * 0.95


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _wrap180(x):
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0


def _round_sig(x: float, sig: int = 3) -> float:
    if x == 0 or not math.isfinite(x):
        return float(x)
    return float(f"{x:.{sig}g}")


def _fmt(x, nd: int = 1, unit: str = "") -> str:
    v = _num(x)
    if v is None:
        return "—"
    return f"{v:.{nd}f}{unit}"


def _fmt_param(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1000 or float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:g}"


IRREGULAR_MAX_SHARE = 0.02   # 間隔偏離整數個取樣週期 > 0.25 格的比例 < 2 %:原始樣本放回格點(不整段內插)
SNAP_TOL = 0.25              # |dt·fs − round(dt·fs)| ≤ 0.25 視為「規則」(只是掉了幾筆)


def _make_pieces(t: np.ndarray, cols: list[np.ndarray], intervals: list[Interval], fs: float,
                 min_len_s: float = MIN_PIECE_S) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """飛行區間內的連續、等間隔片段。

    只在真正的資料空洞(間隔 > GAP_FACTOR / fs)切段。片段內:
    - 間隔都接近整數個取樣週期(ArduPilot 排程器 + 偶爾掉一兩筆訊息,SD 卡忙時很常見)→
      原始樣本放在 round(dt·fs) 累加的格點上、不移動,只線性內插「缺的」格點
      (單一缺點的內插在 0.25 fs 以下影響很小;不要因為 0.1–1 % 的掉訊息把資料切碎丟掉);
    - 取樣時間真的不規則(偏離格點的比例 ≥ IRREGULAR_MAX_SHARE)→ 整段線性內插到 1/fs 的時間格。
      內插等於低通(半格偏移時 0.4·fs 的振幅只剩 31 %),會把接近 Nyquist 的震盪壓掉,所以能不用就不用。
    """
    out = []
    t = np.asarray(t, float)
    if t.size < 2 or fs <= 0:
        return out
    ok = np.isfinite(t)
    for c in cols:
        ok &= np.isfinite(c)
    t = t[ok]
    cols = [np.asarray(c, float)[ok] for c in cols]
    gap = max(GAP_FACTOR / fs, GAP_MIN_S)
    for a, b in intervals:
        i0 = int(np.searchsorted(t, a, "left"))
        i1 = int(np.searchsorted(t, b, "right"))
        if i1 - i0 < 2:
            continue
        ts = t[i0:i1]
        cs = [c[i0:i1] for c in cols]
        inc = np.r_[True, np.diff(ts) > 0]              # 重複 / 倒退的時間戳只留第一筆
        if not inc.all():
            ts, cs = ts[inc], [c[inc] for c in cs]
        brk = np.flatnonzero(np.diff(ts) > gap) + 1
        for s, e in zip(np.r_[0, brk], np.r_[brk, ts.size]):
            tt = ts[s:e]
            if tt.size < 2 or tt[-1] - tt[0] < min_len_s:
                continue
            seg = [c[s:e] for c in cs]
            steps = np.diff(tt) * fs
            r = np.rint(steps)
            irr = (np.abs(steps - r) > SNAP_TOL) | (r < 1)
            if irr.mean() < IRREGULAR_MAX_SHARE:
                if not irr.any() and np.all(r == 1):
                    out.append((tt, seg))
                    continue
                idx = np.r_[0.0, np.cumsum(np.maximum(r, 0.0))]
                idx, first = np.unique(idx, return_index=True)       # 同一格(間隔 < 半格)只留第一筆
                tt_u = tt[first]
                k = np.arange(int(idx[-1]) + 1, dtype=float)
                out.append((np.interp(k, idx, tt_u), [np.interp(k, idx, c[first]) for c in seg]))
                continue
            n = int(np.floor((tt[-1] - tt[0]) * fs + 1e-9)) + 1
            grid = tt[0] + np.arange(n) / fs
            out.append((grid, [np.interp(grid, tt, c) for c in seg]))
    return out


def _coverage(pieces, intervals: list[Interval]) -> float | None:
    """片段涵蓋的時間 / 飛行區間總長(0–1;實際納入分析的比例)。"""
    tot = _total(intervals)
    if tot <= 0:
        return None
    cov = sum(float(p[0][-1] - p[0][0]) for p in pieces if p[0].size > 1)
    return float(min(1.0, cov / tot))


def _highpass(x: np.ndarray, n: int) -> np.ndarray:
    """減去 n 點滑動平均(去掉慢速漂移,延遲估計用)。"""
    n = int(max(3, min(n, x.size)))
    k = np.ones(n) / n
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return x - np.convolve(pad, k, mode="valid")


def _unwrap_yaw(des: np.ndarray, act: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """航向角:實際值展開成連續角度,期望 = 實際 + 包到 ±180 的誤差(兩者都連續)。"""
    a = np.degrees(np.unwrap(np.radians(np.asarray(act, float))))
    d = a + _wrap180(np.asarray(des, float) - np.asarray(act, float))
    return d, a


# =====================================================================
# 統計資料結構(規則的輸入;測試可直接組出來)
# =====================================================================

@dataclass
class OscPeak:
    """誤差 / 陀螺儀頻譜中的一個峰。"""
    f: float
    rms: float                         # deg/s(角度層級為 deg)
    prominence: float = 0.0
    sustain: float = 1.0               # 出現在多少比例的時間窗(0–1)
    t_focus: float | None = None
    window: tuple[float, float] | None = None
    source: str = "rate"               # "rate" / "angle" / "imu"
    fs: float = 0.0                    # 該訊號取樣率(判斷混疊)
    f_lo: float | None = None          # 峰的頻帶(Hz)
    f_hi: float | None = None
    phys: float | None = None          # 姿態角交叉檢查:實際擺動 / 該頻率應有的擺動(None = 沒檢查)

    @property
    def sustained(self) -> bool:
        return self.sustain >= OSC_SUSTAIN_FRAC

    @property
    def phys_failed(self) -> bool:
        """姿態角在 f 沒有對應的擺動 → 不是真的 f Hz 角速度震盪(多半是更高頻的噪音混疊)。"""
        return self.phys is not None and self.phys < PHYS_MIN_RATIO

    def alias_cands(self) -> list[float]:
        """瞬時取樣時真實頻率可能是 f、fs−f、fs+f、2fs−f…(姿態角交叉檢查不過時不含 f 本身)。"""
        if self.fs <= 0:
            return [self.f]
        c = [self.f, self.fs - self.f, self.fs + self.f, 2 * self.fs - self.f]
        if self.fs > LOW_RATE_HZ * 2:
            c.append(2 * self.fs + self.f)
        else:
            c = [x for x in c if x <= 60]
        if self.phys_failed:
            c = c[1:]
        return sorted({round(x, 1) for x in c if x > 0})

    def alias_text(self) -> str:
        """低取樣率 / 未達迴圈頻率時,真實頻率的候選(低速只列 ≤ 60 Hz)。"""
        if self.fs <= 0:
            return ""
        return "、".join(f"{c:g}" for c in self.alias_cands())


@dataclass
class LevelStats:
    """單一層級(角速度或角度)的追蹤統計。"""
    level: str                          # "rate" / "angle"
    fs: float
    dur_s: float = 0.0
    rms_err: float | None = None
    max_err: float | None = None
    t_max_err: float | None = None
    rms_des: float | None = None
    rms_act: float | None = None
    bias: float | None = None           # 準穩態平均誤差(期望 − 實際)
    lag_ms: float | None = None
    lag_rho: float | None = None
    sat_share: float | None = None      # 輸出飽和比例(0–1)
    peaks: list[OscPeak] = field(default_factory=list)
    h_peak: float | None = None
    h_peak_hz: float | None = None
    bw_hz: float | None = None
    bw_gt: float | None = None          # 頻寬至少這麼高(可信範圍內都沒掉到 −3 dB)
    bw_lt: float | None = None          # 頻寬低於這個頻率(最低的可信頻點就已在 −3 dB 以下)
    overshoot: float | None = None
    rise_s: float | None = None
    settle_s: float | None = None
    step_n: int = 0
    incoherent: float | None = None     # 實際值功率中與期望無關的比例(0–1)
    spectrum: Spectrum | None = None
    freq: FreqResponse | None = None
    step: StepResponse | None = None
    span: tuple[float, float] | None = None
    input_bw: float | None = None       # 步階反卷積的輸入有效頻寬(Hz)

    @property
    def track_ratio(self) -> float | None:
        if self.rms_err is None or self.rms_des is None:
            return None
        floor = MIN_ACTIVITY_RATE if self.level == "rate" else MIN_ACTIVITY_ANG
        if self.rms_des < floor:
            return None
        return self.rms_err / self.rms_des

    def step_usable(self) -> bool:
        """步階響應可信:取樣率夠、窗數夠,且輸入頻寬 ≥ 系統共振 / 頻寬頻率的 1.2 倍。"""
        rate = self.level == "rate"
        need = STEP_RULE_MIN_HZ_RATE if rate else STEP_RULE_MIN_HZ_ANG
        if self.step_n < STEP_MIN_WINDOWS or not _fs_ok(self.fs, need) or self.overshoot is None:
            return False
        ref = STEP_MIN_INPUT_BW_RATE if rate else STEP_MIN_INPUT_BW_ANG
        if (self.h_peak or 0) > 1.05 and self.h_peak_hz:      # 有共振:輸入要涵蓋共振頻率之上
            ref = max(ref, STEP_BW_MARGIN * self.h_peak_hz)
        if self.bw_hz:
            ref = max(ref, self.bw_hz)
        return (self.input_bw or 0.0) >= ref


@dataclass
class CtrlStats:
    """CTRL 訊息(控制器 P / D 項輸出的 RMS)。"""
    p_rms: float
    d_rms: float | None = None
    ratio: float | None = None          # D / P
    f_est: float | None = None          # 由 D/P 比與增益(離散 PID 模型)換算的等效頻率(Hz)
    f_est_gt: float | None = None       # D/P 接近模型上限、反推不可靠:只知道等效頻率高於此值
    spike: float | None = None          # 最大值 / 中位數
    t_spike: float | None = None


@dataclass
class PidStats:
    """PIDR / PIDP / PIDY 細項。"""
    fs: float
    p_rms: float = 0.0
    i_rms: float = 0.0
    d_rms: float = 0.0
    ff_rms: float = 0.0
    i_mean: float | None = None
    imax: float | None = None
    i_sat_share: float | None = None
    dmod_share: float | None = None
    t_dmod: float | None = None
    limit_share: float | None = None
    d_hf_frac: float | None = None
    srate_p95: float | None = None      # 輸出變化率(SRate)第 95 百分位
    smax: float | None = None           # ATC_RAT_xxx_SMAX(0 = 未啟用 slew 限制)


@dataclass
class VehicleContext:
    """整機層級的狀態(規則共用)。"""
    prefix: str = "ATC_"
    params: dict = field(default_factory=dict)
    vibe: str = "ok"                    # ok / warn / bad / unknown
    motor_sat: float | None = None      # 馬達飽和比例(上 + 下限;None = 無法判斷)
    motor_hi: float | None = None       # 頂到上限(SPIN_MAX)的比例
    motor_lo: float | None = None       # 壓到下限(SPIN_MIN)的比例
    hover: float | None = None          # 實測懸停油門
    thrust_loss: bool = False
    rate_fs: float | None = None        # 角速度層級最佳取樣率
    pilot_share: float | None = None    # 飛行中「直接由搖桿給姿態」模式(STABILIZE / ALT_HOLD / ACRO…)的比例
    batch: bool = False                 # 有批次陀螺儀取樣(ISBH / ISBD)
    pid_limit: float | None = None      # PIDx Flags 的 limit 旗標比例(各軸最大;馬達範圍不明時當飽和證據)

    @property
    def closed_loop_des(self) -> bool:
        """期望姿態主要來自位置控制(LOITER / AUTO…):角度層級的 |H|、步階會被回授扭曲。"""
        return self.pilot_share is not None and self.pilot_share < PILOT_SHARE_MIN

    @property
    def sat_low_dom(self) -> bool:
        """馬達飽和主要是壓到下限(SPIN_MIN)、上限幾乎沒碰、懸停油門也不高:不是推力不足。"""
        return bool(self.motor_sat is not None and self.motor_sat >= SAT_SHARE_WARN
                    and (self.motor_hi or 0.0) < SAT_SHARE_WARN
                    and not (self.hover is not None and self.hover > HOVER_HIGH))

    @property
    def block_increase(self) -> str | None:
        """不應提高任何增益的原因(None = 可以)。"""
        if self.vibe == "bad":
            return "振動過大"
        if self.thrust_loss:
            return "偵測到推力不足 / 馬達失效"
        if self.motor_sat is not None and self.motor_sat >= SAT_SHARE_WARN:
            return "馬達輸出飽和"
        if self.motor_sat is None and self.pid_limit is not None and self.pid_limit >= SAT_SHARE_WARN:
            return "PID 記錄顯示輸出常受限(馬達飽和)"
        return None


@dataclass
class AxisContext:
    """單軸規則的輸入。"""
    axis: str
    veh: VehicleContext
    rate: LevelStats | None = None
    ang: LevelStats | None = None
    imu_peaks: list[OscPeak] = field(default_factory=list)
    imu_fs: float | None = None
    ctrl: CtrlStats | None = None
    pid: PidStats | None = None
    out_mean: float | None = None       # 懸停時平均輸出(RATE.xOut;-1…1)

    @property
    def names(self) -> dict[str, str]:
        return gain_names(self.veh.prefix, self.axis)


# =====================================================================
# 建議
# =====================================================================

def gain_names(prefix: str, axis: str) -> dict[str, str]:
    a = AXIS_PARAM[axis]
    return {"P": f"{prefix}RAT_{a}_P", "I": f"{prefix}RAT_{a}_I", "D": f"{prefix}RAT_{a}_D",
            "IMAX": f"{prefix}RAT_{a}_IMAX", "FLTD": f"{prefix}RAT_{a}_FLTD",
            "FLTE": f"{prefix}RAT_{a}_FLTE", "FLTT": f"{prefix}RAT_{a}_FLTT", "SMAX": f"{prefix}RAT_{a}_SMAX",
            "ANG": f"{prefix}ANG_{a}_P", "ACC": f"{prefix}ACCEL_{_ACC_AX[axis]}_MAX",
            "TC": f"{prefix}INPUT_TC"}


def param_range(name: str) -> tuple[float, float] | None:
    """參數合理範圍(QuadPlane 的 Q_A_ 對應 ATC_)。"""
    key = "ATC_" + name[4:] if name.startswith("Q_A_") else name
    return PARAM_RANGES.get(key)


STEP_MAX_PCT = 25.0         # 單次建議的最大幅度(SAFETY_NOTES:每次 10–25 %)


def _bounded_step(param: str, cur: float, prop: float) -> tuple[float | None, str]:
    """把 cur → prop 夾進 ArduPilot 合理範圍,但絕不反向、也不為了「進到範圍內」而放大步幅。

    回傳 (建議值或 None, 附註)。目前值在範圍內:夾在範圍(只會縮小步幅)。目前值在範圍外:
    - 往範圍內調 → 照原本的步幅(不直接跳到邊界),附註「目前值在文件範圍外,請確認是否為 AutoTune 結果」;
    - 往範圍外調 → 不給值(AutoTune 的結果常超出文件範圍,例如 ANG_P 15–20、Yaw P < 0.1),請人工確認。
    """
    rg = param_range(param)
    if rg is None:
        return prop, ""
    lo, hi = rg
    up = prop > cur
    if lo <= cur <= hi:
        return min(max(prop, lo), hi), ""
    side = "低於" if cur < lo else "高於"
    rng = f"[{_fmt_param(lo)}, {_fmt_param(hi)}]"
    if (cur < lo and up) or (cur > hi and not up):
        return (min(prop, hi) if up else max(prop, lo),
                f"(目前值{side} ArduPilot 文件範圍 {rng},請確認是否為 AutoTune 結果;建議用 AutoTune 而非手動大改)")
    return None, (f"(目前值 {_fmt_param(cur)} 已{side} ArduPilot 文件範圍 {rng},再往這個方向調會更超出範圍,"
                  "無法自動給值;請人工確認或用 AutoTune)")


def gain_suggestion(param: str, params: dict, pct: float, reason: str,
                    priority: int = 2) -> Suggestion:
    """以目前值 × (1 + pct/100) 產生建議(|pct| ≤ 25 %),夾在 PARAM_RANGES、四捨五入到 3 位有效數字。

    log 沒記目前值 → proposed None;已在範圍邊界 → proposed None 並在理由註明;
    目前值在範圍外 → 見 _bounded_step(絕不因夾範圍而反向或加大步幅);
    目前為 0:增益 → 請從小值開始;其他參數(例:ATC_ACCEL_*_MAX = 0 代表不限制)→ 不給值、請人工確認。
    """
    pct = float(max(-STEP_MAX_PCT, min(STEP_MAX_PCT, pct)))
    direction = DIR_UP if pct > 0 else DIR_DOWN
    cur = _num(params.get(param))
    if cur is None:
        return Suggestion(param, direction, reason + "(log 沒有記錄目前值)", None, None,
                          float(pct), priority)
    if cur == 0:
        if _is_gain(param) and pct > 0:
            why = "(目前為 0,請依調參指南從小值開始)"
        elif _is_gain(param):
            why = "(目前已是 0,無法再調低)"
        else:
            why = "(目前為 0(停用 / 不限制),請人工確認)"
        return Suggestion(param, direction, reason + why, cur, None, None, priority)
    prop, note = _bounded_step(param, cur, cur * (1.0 + pct / 100.0))
    if prop is None:
        return Suggestion(param, direction, reason + note, cur, None, None, priority)
    prop = _round_sig(prop, 3)
    if (pct < 0 and prop > cur) or (pct > 0 and prop < cur):          # 保險:絕不給反方向的值
        return Suggestion(param, direction, reason + "(夾到合理範圍後方向相反,不自動給值;請人工確認)",
                          cur, None, None, priority)
    if prop == _round_sig(cur, 3):
        rg = param_range(param)
        lim = f"[{_fmt_param(rg[0])}, {_fmt_param(rg[1])}]" if rg else ""
        return Suggestion(param, direction, reason + f"(已達合理範圍 {lim} 邊界,不再調整)",
                          cur, None, None, priority)
    return Suggestion(param, direction, reason + note, cur, prop, round((prop / cur - 1.0) * 100.0, 1),
                      priority)


def _follow_i(ctx: AxisContext, p_sug: Suggestion, priority: int) -> Suggestion | None:
    """P 改了之後 I 要不要跟:Roll/Pitch 慣例 I = P;Yaw 慣例 I ≈ P / 10(步幅同樣 ≤ 25 %、不反向)。"""
    nm = ctx.names
    prm = ctx.veh.params
    cur_p, cur_i = _num(prm.get(nm["P"])), _num(prm.get(nm["I"]))
    if p_sug.proposed is None or not cur_p or not cur_i:
        return None
    if ctx.axis == "yaw":
        if abs(cur_i / (cur_p / 10.0) - 1.0) > 0.25:
            return None
        target, why = p_sug.proposed / 10.0, "Yaw 的 I 慣例約為 P 的 1/10,隨 P 一起調"
    else:
        if abs(cur_i / cur_p - 1.0) > 0.1:
            return None
        target, why = p_sug.proposed, "Roll / Pitch 的 I 慣例與 P 相同,隨 P 一起調"
    up = p_sug.proposed > cur_p
    lim = cur_i * (1.0 + STEP_MAX_PCT / 100.0) if up else cur_i * (1.0 - STEP_MAX_PCT / 100.0)
    target = min(target, lim) if up else max(target, lim)
    target, note = _bounded_step(nm["I"], cur_i, target)
    if target is None:
        return None
    target = _round_sig(target, 3)
    if target == _round_sig(cur_i, 3) or (target > cur_i) != up:
        return None
    return Suggestion(nm["I"], DIR_UP if up else DIR_DOWN, why + note, cur_i, target,
                      round((target / cur_i - 1.0) * 100.0, 1), priority)


THRUST_CHECK = "推重比 / 螺旋槳 / 重量 / 電池"     # 馬達飽和、懸停油門偏高共用(建議清單只列一次)


def _check(what: str, reason: str, priority: int = 2) -> Suggestion:
    return Suggestion(what, DIR_CHECK, reason, priority=priority)


def _is_gain(param: str) -> bool:
    return bool(re.search(r"(RAT_(RLL|PIT|YAW)_(P|I|D|FF)|ANG_(RLL|PIT|YAW)_P)$", param))


def _accel_suggestion(param: str, params: dict, reason: str, priority: int) -> Suggestion:
    """降低 ATC_ACCEL_*_MAX;目前為 0(= 不限制角加速度)時「調低」沒有意義 → 改成檢查項。"""
    cur = _num(params.get(param))
    if cur is not None and cur == 0:
        return _check(param, "目前為 0 = 不限制角加速度;可依 MP「Initial Tune Parameters」依槳徑設定,"
                             "或讓 AutoTune 設定", priority)
    return gain_suggestion(param, params, STEP_ACCEL_DOWN, reason, priority)


def _raises_aggressiveness(s: Suggestion) -> bool:
    """這條建議會不會讓控制更積極:提高增益、減小 INPUT_TC、放寬角加速度上限。

    方向以「建議值 vs 目前值」為準(就算方向標籤寫 decrease,建議值比目前大也算提高)。
    """
    if s.direction == DIR_CHECK:
        return False
    vals = s.proposed is not None and s.current is not None
    up = s.direction == DIR_UP or (vals and s.proposed > s.current)
    down = s.direction == DIR_DOWN or (vals and s.proposed < s.current)
    if _is_gain(s.param) or re.search(r"ACCEL_[RPY]_MAX$", s.param):
        return up
    if s.param.endswith("INPUT_TC"):
        return down
    return False


# =====================================================================
# 各層級統計
# =====================================================================

def _peak_frames(fr: Frames, pk: Peak, t0s: list[float], min_rms: float, source: str) -> OscPeak:
    """把 Welch 峰對回各時間窗:持續比例、最嚴重的時刻。"""
    f = fr.f
    band = (f >= pk.f_lo) & (f <= pk.f_hi)
    if fr.n == 0 or not band.any():
        return OscPeak(pk.f, pk.rms, pk.prominence, 1.0, None, None, source, fr.fs, pk.f_lo, pk.f_hi)
    pw = (np.abs(fr.X[:, band]) ** 2) * fr.scale[band][None, :]
    ex = np.maximum(pw - pk.floor, 0.0).sum(axis=1) * fr.df
    rms_i = np.sqrt(ex)
    thr = max(0.5 * pk.rms, 0.5 * min_rms)
    sustain = float(np.mean(rms_i >= thr))
    k = int(np.argmax(rms_i))
    tc = t0s[int(fr.piece[k])] + float(fr.center[k]) / fr.fs
    half = fr.nperseg / fr.fs / 2.0
    return OscPeak(pk.f, pk.rms, pk.prominence, sustain, tc, (tc - half, tc + half), source, fr.fs,
                   pk.f_lo, pk.f_hi)


def _spectrum_peaks(fr: Frames, psd: np.ndarray, fmin: float, fmax: float, min_rms: float,
                    t0s: list[float], source: str, prom: float = PEAK_PROM_RATIO) -> list[OscPeak]:
    if fr.n == 0 or not np.isfinite(psd).any():
        return []
    pks = find_peaks(fr.f, psd, fmin=fmin, fmax=fmax, prom_ratio=prom, min_rms=min_rms)
    return [_peak_frames(fr, pk, t0s, min_rms, source) for pk in pks]


def _merge_peaks(peaks: list[OscPeak]) -> list[OscPeak]:
    """不同頻譜找到的峰合併:頻帶重疊(或相距 < 1 Hz)的只留 RMS 大的。"""
    out: list[OscPeak] = []
    for p in sorted(peaks, key=lambda q: -q.rms):
        lo_p, hi_p = (p.f_lo if p.f_lo is not None else p.f), (p.f_hi if p.f_hi is not None else p.f)
        if any(abs(p.f - q.f) < 1.0 or (lo_p <= (q.f_hi if q.f_hi is not None else q.f)
                                         and hi_p >= (q.f_lo if q.f_lo is not None else q.f)) for q in out):
            continue
        out.append(p)
    return out


def level_stats(pieces: list[tuple[np.ndarray, list[np.ndarray]]], fs: float, level: str,
                axis: str, tar_fltt: float | None = None) -> LevelStats | None:
    """均勻取樣的 (時間, [期望, 實際]) 片段 → LevelStats。

    tar_fltt:期望值是 PIDx.Tar(已經過 ATC_RAT_xxx_FLTT 一階低通)時填 FLTT(Hz):
    延遲加回 1/(2π·FLTT)、頻率響應乘回 LPF,換算成與 RATE.RDes(未濾波目標)相同的基準,門檻才一致。
    """
    if not pieces or fs <= 0:
        return None
    ts = [p[0] for p in pieces]
    d = [p[1][0] for p in pieces]
    a = [p[1][1] for p in pieces]
    e = [dd - aa for dd, aa in zip(d, a)]
    T, D, E = np.concatenate(ts), np.concatenate(d), np.concatenate(e)
    if E.size < 16:
        return None
    ls = LevelStats(level, float(fs), dur_s=float(E.size / fs))
    ls.span = (float(ts[0][0]), float(ts[-1][-1]))
    ls.rms_err = float(np.sqrt(np.mean(E ** 2)))
    k = int(np.argmax(np.abs(E)))
    ls.max_err, ls.t_max_err = float(abs(E[k])), float(T[k])
    ls.rms_des = float(np.sqrt(np.mean(np.concatenate([x - x.mean() for x in d]) ** 2)))
    ls.rms_act = float(np.sqrt(np.mean(np.concatenate([x - x.mean() for x in a]) ** 2)))
    # 偏差:準穩態(期望角速度小 / 期望角度變化慢)時的平均誤差
    if level == "rate":
        sel = np.abs(D) < BIAS_RATE_DES_MAX
    else:
        sel = np.concatenate([np.abs(np.gradient(x) * fs) < BIAS_ANG_RATE_MAX if x.size > 1
                              else np.zeros(x.size, bool) for x in d])
    if sel.mean() >= BIAS_MIN_SHARE and sel.sum() >= 5 * fs:
        ls.bias = float(np.mean(E[sel]))
    # 延遲:角度層級先去掉慢速漂移
    if level == "rate":
        lag, rho = lag_xcorr(d, a, fs, LAG_MAX_YAW_RATE_S if axis == "yaw" else LAG_MAX_RATE_S)
    else:
        n = int(round(fs * 1.0))
        lag, rho = lag_xcorr([_highpass(x, n) for x in d], [_highpass(x, n) for x in a], fs,
                             LAG_MAX_ANG_S)
    ls.lag_ms = None if lag is None else float(lag * 1000.0)
    if ls.lag_ms is not None and tar_fltt:
        ls.lag_ms += 1000.0 / (2.0 * math.pi * tar_fltt)        # FLTT 一階低通的低頻群延遲
    ls.lag_rho = float(rho)
    # 頻譜:誤差 E = D − A。
    # - < OSC_MID_HZ:峰在「期望值解釋不了」的殘差 R = E − H_de·D 上找(搖桿造成的追蹤誤差和期望同調,不算震盪);
    # - ≥ OSC_MID_HZ(角速度層級):誤差頻譜與殘差頻譜都找、合併(重疊的取 RMS 大者)。期望角速度 =
    #   ANG_P × 角度誤差 + 前饋,角速度環的自激震盪會經角度環回授進期望值、和期望同調(懸停時同調度 0.9 以上),
    #   只看殘差會把震盪本身扣掉(RMS 剩 25–50 %);而與期望無關的窄峰(馬達噪音混疊)在殘差上最清楚
    #   (誤差頻譜在操作頻帶附近有追蹤誤差墊高底噪)。經 INPUT_TC 整形的搖桿輸入在 4 Hz 以上能量很小,不會造成假峰。
    rate = level == "rate"
    nper = choose_nperseg(fs, max(x.size for x in e), WELCH_WIN_S if rate else WELCH_WIN_S_ANG,
                          WELCH_MAX_N if rate else WELCH_MAX_N_ANG)
    detr = "mean" if rate else "linear"
    fr_da = spectral_frames(d, fs, nper, y=a, detrend=detr)
    nyq = fs / 2.0
    t0s = [float(x[0]) for x in ts]
    if fr_da.n:
        X, A = fr_da.X, fr_da.Y
        Ef = X - A
        pdd, paa, pda = _spectra(fr_da)
        pee = np.mean(np.abs(Ef) ** 2, axis=0) * fr_da.scale
        pde = np.mean(np.conj(X) * Ef, axis=0) * fr_da.scale
        hde = np.where(pdd > 0, pde / np.where(pdd > 0, pdd, 1.0), 0.0)
        Rf = Ef - hde[None, :] * X
        fr_r = Frames(fr_da.f, Rf, None, fr_da.piece, fr_da.center, fr_da.scale, nper, fr_da.fs)
        pres = np.mean(np.abs(Rf) ** 2, axis=0) * fr_da.scale
        min_rms = PEAK_MIN_RMS_RATE if rate else PEAK_MIN_RMS_ANG
        fmax = PEAK_FMAX_FRAC * nyq
        if rate:
            fr_e = Frames(fr_da.f, Ef, None, fr_da.piece, fr_da.center, fr_da.scale, nper, fr_da.fs)
            hi = _merge_peaks(_spectrum_peaks(fr_e, pee, OSC_MID_HZ, fmax, min_rms, t0s, level)
                              + _spectrum_peaks(fr_r, pres, OSC_MID_HZ, fmax, min_rms, t0s, level))
            lo = _spectrum_peaks(fr_r, pres, PEAK_FMIN_HZ, min(fmax, OSC_MID_HZ - 1e-9), min_rms, t0s, level)
            lo = [p for p in lo if not any(q.f_lo is not None and p.f_hi is not None and p.f_hi >= q.f_lo
                                           for q in hi)]
            ls.peaks = sorted(hi + lo, key=lambda p: -p.rms)
        else:
            ls.peaks = _spectrum_peaks(fr_r, pres, PEAK_FMIN_HZ, fmax, min_rms, t0s, level)
        ls.spectrum = Spectrum(fr_da.f, pee, paa, [(p.f, p.rms) for p in ls.peaks], nyq)
        coh = _coh(pdd, paa, pda)
        m = (fr_da.f > 0) & (fr_da.f <= PEAK_FMAX_FRAC * nyq)
        if paa[m].sum() > 0:
            ls.incoherent = float(np.sum(paa[m] * (1.0 - coh[m])) / np.sum(paa[m]))
        h_mult = (1.0 / (1.0 + 1j * fr_da.f / tar_fltt)) if tar_fltt else None
        ls.freq = freq_response(None, None, fs, frames=fr_da, h_mult=h_mult)
        ls.h_peak, ls.h_peak_hz, ls.bw_hz = ls.freq.peak_mag, ls.freq.peak_hz, ls.freq.bandwidth_hz
        ls.bw_gt = _bw_lower_bound(ls.freq)
        ls.bw_lt = _bw_upper_bound(ls.freq, fr_da.n)
    # 步階響應(輸入頻寬不足時不採用)
    if level == "rate":
        st, ls.input_bw = _step_core(d, a, fs, STEP_RATE_WIN_S, STEP_RATE_RESP_S, None,
                                     STEP_FCUT_RATE_HZ, WIENER_REG, 0.25, None)
    else:
        st, ls.input_bw = _step_core(d, a, fs, STEP_ANG_WIN_S, STEP_ANG_RESP_S, None,
                                     STEP_FCUT_ANG_HZ, WIENER_REG, 0.25, None)
    if st.t.size:
        ls.step = st
        ls.overshoot, ls.rise_s, ls.settle_s = st.overshoot_pct, st.rise_s, st.settle_s
    ls.step_n = st.n_windows
    return ls


def _gyro_peaks(t: np.ndarray, g: np.ndarray, intervals: list[Interval]) -> tuple[float, list[OscPeak]]:
    """陀螺儀(deg/s)在飛行中的頻譜峰(只列 ≥ OSC_MID_HZ:更慢的是飛行動作)。"""
    fs = _fs_of(t)
    if fs <= 0:
        return 0.0, []
    pcs = _make_pieces(t, [g], intervals, fs)
    if not pcs:
        return fs, []
    x = [p[1][0] for p in pcs]
    nper = choose_nperseg(fs, max(v.size for v in x))
    fr = spectral_frames(x, fs, nper)
    if fr.n == 0:
        return fs, []
    psd = np.mean(np.abs(fr.X) ** 2, axis=0) * fr.scale
    pk = _spectrum_peaks(fr, psd, OSC_MID_HZ, IMU_PEAK_FMAX_FRAC * fs / 2.0, PEAK_MIN_RMS_RATE,
                         [float(p[0][0]) for p in pcs], "imu")
    return fs, pk


def _loop_rate(params: dict) -> float:
    v = _num(params.get("SCHED_LOOP_RATE"))
    return float(v) if v and v > 0 else LOOP_RATE_DEFAULT


def _full_rate(fs: float | None, params: dict) -> bool:
    """記錄率約等於迴圈頻率:log 看到的就是控制器看到的(記錄本身不會造成混疊)。"""
    return fs is not None and fs >= FULL_RATE_FRAC * _loop_rate(params)


def ctrl_fmax(params: dict) -> float:
    """角速度環可能自激的最高頻率(Hz):min(max(40, 2.5 × INS_GYRO_FILTER), 0.2 × 迴圈頻率)。"""
    gf = _num(params.get("INS_GYRO_FILTER")) or 0.0
    return float(min(max(OSC_CTRL_FMAX_MIN_HZ, OSC_CTRL_FMAX_GF * gf), OSC_CTRL_FMAX_LOOP * _loop_rate(params)))


def dp_ratio_model(kp: float, kd: float, fltd: float, loop_hz: float, f) -> np.ndarray:
    """單一頻率 f 的誤差正弦下,AC_PID 的 D 項 / P 項輸出比(離散:迴圈頻率的後向差分 + 一階 FLTD 低通)。

    r(f) = (Kd/Kp)·|(1 − z⁻¹)/T|·|α / (1 − (1 − α)z⁻¹)|,z = e^{j2πfT},α = T / (T + 1/(2π·FLTD))(FLTD = 0 → α = 1)。
    FLTE 同時作用在 P 與 D 上,比值中消掉;slew 限制器的 Dmod 也同時縮放 P 與 D。
    """
    f = np.asarray(f, float)
    T = 1.0 / loop_hz
    z = np.exp(1j * 2.0 * np.pi * f * T)
    al = T / (T + 1.0 / (2.0 * np.pi * fltd)) if fltd and fltd > 0 else 1.0
    return (kd / kp) * np.abs((1.0 - 1.0 / z) / T) * np.abs(al / (1.0 - (1.0 - al) / z))


def _invert_dp(ratio: float, kp: float, kd: float, fltd: float, loop_hz: float) -> tuple[float | None, float | None]:
    """D/P 輸出比 → 等效頻率 (f_est, None);接近模型上限(病態)→ (None, 下限頻率)。"""
    grid = np.linspace(0.05, loop_hz / 2.0, 8000)
    r = dp_ratio_model(kp, kd, fltd, loop_hz, grid)
    rmax = float(r.max())
    if ratio >= CTRL_FEST_UPPER_FRAC * rmax:
        k = int(np.argmax(r >= CTRL_FEST_UPPER_FRAC * rmax))
        return None, float(grid[k])
    k = int(np.argmax(r >= ratio))
    if k == 0:
        return float(grid[0]), None
    w = (ratio - r[k - 1]) / (r[k] - r[k - 1]) if r[k] != r[k - 1] else 0.0
    return float(grid[k - 1] + w * (grid[k] - grid[k - 1])), None


def _ctrl_stats(index, axis: str, intervals: list[Interval], params: dict,
                names: dict[str, str]) -> CtrlStats | None:
    tbl = index.table("CTRL")
    pf, dfld = _CTRL_FIELDS[axis]
    if tbl is None or pf not in tbl:
        return None
    m = _in_intervals(tbl["t"], intervals)
    if m.sum() < 10:
        return None
    p = np.asarray(tbl[pf], float)[m]
    tt = tbl["t"][m]
    pm = float(np.median(p))
    cs = CtrlStats(p_rms=pm)
    if pm > 0:
        k = int(np.argmax(p))
        cs.spike, cs.t_spike = float(p[k] / pm), float(tt[k])
    if dfld and dfld in tbl:
        d = np.asarray(tbl[dfld], float)[m]
        dm = float(np.median(d))
        cs.d_rms = dm
        if pm > 0:
            cs.ratio = dm / pm
            kp, kd = _num(params.get(names["P"])), _num(params.get(names["D"]))
            fc = _num(params.get(names["FLTD"])) or 0.0
            if kp and kd and kp > 0 and kd > 0:
                # D/P = (Kd/Kp)·|離散微分|·|FLTD 低通| → 數值反推等效頻率(CTRL 是寬頻 RMS 比,不是單一頻率)
                cs.f_est, cs.f_est_gt = _invert_dp(cs.ratio, kp, kd, fc, _loop_rate(params))
    return cs


def _pid_stats(index, msg: str, intervals: list[Interval], params: dict,
               names: dict[str, str]) -> PidStats | None:
    tbl = index.table(msg)
    if tbl is None or "P" not in tbl:
        return None
    m = _in_intervals(tbl["t"], intervals)
    if m.sum() < 20:
        return None
    t = tbl["t"][m]
    fs = _fs_of(t)
    col = {k: np.asarray(tbl[k], float)[m] for k in ("P", "I", "D", "FF", "Dmod", "SRate", "Flags")
           if k in tbl and tbl[k].ndim == 1}
    ps = PidStats(fs=fs)

    def rms(k):
        return float(np.sqrt(np.mean(col[k] ** 2))) if k in col else 0.0

    ps.p_rms, ps.i_rms, ps.d_rms, ps.ff_rms = rms("P"), rms("I"), rms("D"), rms("FF")
    if "I" in col:
        ps.i_mean = float(np.mean(col["I"]))
        imax = _num(params.get(names["IMAX"]))
        ps.imax = imax
        if imax and imax > 0:
            ps.i_sat_share = float(np.mean(np.abs(col["I"]) >= I_SAT_FRAC * imax))
    if "Dmod" in col:
        dm = col["Dmod"]
        ps.dmod_share = float(np.mean(dm < DMOD_ACTIVE))
        ps.t_dmod = float(t[int(np.argmin(dm))])
    if "SRate" in col:
        ps.srate_p95 = float(np.percentile(np.abs(col["SRate"]), 95))
    ps.smax = _num(params.get(names["SMAX"]))
    if "Flags" in col:
        ps.limit_share = float(np.mean((col["Flags"].astype(np.int64) & 1) > 0))
    if "D" in col and fs * 0.4 > OSC_HIGH_HZ * 1.5:
        pcs = _make_pieces(t, [col["D"]], intervals, fs)
        if pcs:
            f, psd = welch_psd([p[1][0] for p in pcs], fs)
            if np.isfinite(psd).any() and np.nansum(psd) > 0:
                ps.d_hf_frac = float(np.nansum(psd[f >= OSC_HIGH_HZ]) / np.nansum(psd))
    return ps


# =====================================================================
# 單軸規則
# =====================================================================

def _phys_ratio(ctx: AxisContext, pk: OscPeak) -> float | None:
    """姿態角交叉檢查:真的 f Hz 角速度擺動(帶內 RMS A)在姿態角上應有 A/(2πf) 的擺動。

    回傳 (姿態角帶內 RMS × 2πf) / 角速度帶內 RMS(真實擺動 ≈ 1;F Hz 的噪音混疊到 f 只有 ≈ f/F);
    沒有同速率的 ATT 或頻譜 → None(無法檢查)。
    """
    r, a = ctx.rate, ctx.ang
    if r is None or a is None or r.spectrum is None or a.spectrum is None:
        return None
    if a.fs < 0.9 * r.fs or pk.f >= PEAK_FMAX_FRAC * a.fs / 2.0:
        return None
    fr_, fa_ = np.asarray(r.spectrum.f, float), np.asarray(a.spectrum.f, float)
    if fr_.size < 3 or fa_.size < 3:
        return None
    df_r = float(fr_[1] - fr_[0])
    half = max(2.0 * df_r, 0.75)            # 峰的主瓣(Hann ±2 格)
    flank = max(2.0 * df_r, 1.0)
    # 姿態角 PSD × (2πf)² = 「角速度等效」PSD:真實運動在兩者相同;只有陀螺儀看到的(噪音混疊)只在角速度上
    p_ang = np.asarray(a.spectrum.psd_act, float) * (2.0 * math.pi * fa_) ** 2
    rx = _band_excess(fr_, np.asarray(r.spectrum.psd_act, float), pk.f, half, flank)
    ax_ = _band_excess(fa_, p_ang, pk.f, half, flank)
    if not (rx > 0) or not np.isfinite(ax_):
        return None
    return float(math.sqrt(max(ax_, 0.0) / rx))


def _band_excess(f: np.ndarray, p: np.ndarray, f0: float, half: float, flank: float) -> float:
    """f0 ± half 頻帶內高出局部底噪(兩側 flank 寬頻帶中位數的較小者)的功率。"""
    df = float(f[1] - f[0])
    band = (f >= f0 - half) & (f <= f0 + half)
    sides = [(f >= f0 - half - flank) & (f < f0 - half), (f > f0 + half) & (f <= f0 + half + flank)]
    fl = [float(np.median(p[m])) for m in sides if m.any() and np.isfinite(p[m]).all()]
    floor = min(fl) if fl else 0.0
    if not band.any():
        return 0.0
    return float(np.sum(np.maximum(p[band] - floor, 0.0)) * df)


def _osc_evidence(ctx: AxisContext) -> tuple[OscPeak | None, str]:
    """最可信的控制震盪峰與它的可信度。

    "direct":角速度以迴圈頻率記錄,或低於迴圈頻率但姿態角有對應的擺動(_phys_ratio ≥ PHYS_MIN_RATIO);
    "suspect":可能是噪音混疊 / 只有陀螺儀看到 → 要有迴圈頻率的 CTRL 佐證才給方向;"":沒有。
    只看 OSC_MID_HZ…ctrl_fmax 的峰(更高的是馬達 / 機架噪音,交給 _noise_peak)。
    記錄是瞬時取樣、沒有抗混疊:LOG_FILE_RATEMAX 抽成 50 / 100 Hz 時,92 / 108 Hz 的馬達噪音會落在 8 Hz。
    """
    prm = ctx.veh.params
    fmax = ctrl_fmax(prm)
    r = ctx.rate
    if r is not None and _fs_ok(r.fs, ALIAS_SAFE_HZ):
        pk = [p for p in r.peaks if OSC_MID_HZ <= p.f <= fmax and p.rms >= OSC_RATE_WARN_RMS]
        if pk:
            best = max(pk, key=lambda p: p.rms)
            if _full_rate(r.fs, prm):
                return best, "direct"
            best.phys = _phys_ratio(ctx, best)
            if best.phys is not None and best.phys >= PHYS_MIN_RATIO:
                return best, "direct"
            return best, "suspect"
    # 只有陀螺儀(IMU)看到的峰:不論 IMU 記錄多快,都要 CTRL(迴圈頻率)佐證才算控制震盪
    imu = [p for p in ctx.imu_peaks if OSC_MID_HZ <= p.f <= fmax]
    if imu and ctx.imu_fs and ctx.imu_fs > (r.fs if r is not None else 0.0):
        return max(imu, key=lambda p: p.rms), "suspect"
    return None, ""


def _noise_peak(ctx: AxisContext) -> OscPeak | None:
    """高於控制頻寬上限(ctrl_fmax)的陀螺儀 / 角速度峰:馬達 / 螺旋槳 / 機架噪音,不是 PID 震盪。"""
    fmax = ctrl_fmax(ctx.veh.params)
    cand = [p for p in (ctx.rate.peaks if ctx.rate is not None else []) + list(ctx.imu_peaks)
            if p.f > fmax and p.rms >= OSC_RATE_WARN_RMS]
    return max(cand, key=lambda p: p.rms) if cand else None


def _measured_dp(ctx: AxisContext) -> tuple[float | None, str]:
    if ctx.pid is not None and ctx.pid.p_rms > 0:
        return ctx.pid.d_rms / ctx.pid.p_rms, "PID"
    if ctx.ctrl is not None and ctx.ctrl.ratio is not None:
        return ctx.ctrl.ratio, "CTRL"
    return None, ""


def _expected_dp(ctx: AxisContext, f: float | None) -> float | None:
    """增益決定的 D/P 輸出比預期值(頻率 f 的單一正弦誤差)。"""
    prm, nm = ctx.veh.params, ctx.names
    kp, kd = _num(prm.get(nm["P"])), _num(prm.get(nm["D"]))
    if not f or not kp or not kd or kp <= 0 or kd <= 0:
        return None
    fltd = _num(prm.get(nm["FLTD"])) or 0.0
    return float(dp_ratio_model(kp, kd, fltd, _loop_rate(prm), min(f, _loop_rate(prm) / 2.0 - 1e-6)))


def _d_dominant(ctx: AxisContext, f: float | None) -> bool:
    """震盪 / 共振是不是 D 造成的(要先降 D)。

    同一頻率的誤差下,D/P 輸出比 ≈ (Kd/Kp)·2πf·|FLTD 低通|,只由增益比與頻率決定:預設增益在 7.7 Hz 以上
    就超過 1.2,所有增益一起放大也不變 —— 所以「D/P 輸出比大」本身不代表 D 太高。判定 D 主導需要:
    增益比 Kd/Kp > D_OVER_P_HIGH(D 相對 P 偏高),或量到的 D/P ≥ max(D_DOMINANT_RATIO, 1.5 × 預期值)
    (多出來的是 D 項吃到的高頻雜訊),或 PIDx 的 D 項功率大多在 ≥ 15 Hz(震盪時 D 項功率通常集中在震盪頻率)。
    """
    prm, nm = ctx.veh.params, ctx.names
    kp, kd = _num(prm.get(nm["P"])), _num(prm.get(nm["D"]))
    if not kp or not kd or kp <= 0 or kd <= 0:
        return False
    if kd / kp > D_OVER_P_HIGH:
        return True
    if ctx.pid is not None and ctx.pid.d_hf_frac is not None and ctx.pid.d_hf_frac >= D_HF_FRAC_WARN:
        return True
    meas, _ = _measured_dp(ctx)
    if meas is None:
        return False
    exp = _expected_dp(ctx, f)
    return meas >= max(D_DOMINANT_RATIO, D_DOM_EXPECT_MULT * exp if exp else 0.0)


def _ctrl_text(ctx: AxisContext) -> str:
    c = ctx.ctrl
    if c is None or c.ratio is None:
        return ""
    s = f"CTRL 的 D 項 / P 項 RMS 比 {c.ratio:.2f}"
    if c.f_est is not None:
        s += f"(依增益換算的等效頻率約 {c.f_est:.0f} Hz;CTRL 是寬頻 RMS 比,不是單一頻率)"
    elif c.f_est_gt is not None:
        s += f"(接近增益決定的上限,等效頻率 > {c.f_est_gt:.0f} Hz:誤差能量偏高頻)"
    elif c.ratio >= CTRL_D_OVER_P_WARN:
        s += "(誤差能量偏高頻)"
    return s + "。"


def _hf_noise(ctx: AxisContext) -> bool:
    """有高頻雜訊 / 震盪跡象(不建議加 D)。"""
    if ctx.pid is not None and ctx.pid.d_hf_frac is not None and ctx.pid.d_hf_frac >= D_HF_FRAC_WARN:
        return True
    for pk in (ctx.rate.peaks if ctx.rate else []) + ctx.imu_peaks:
        if pk.f >= OSC_HIGH_HZ and pk.rms >= OSC_RATE_WARN_RMS:
            return True
    return ctx.veh.vibe in ("warn", "bad")


def _fold(f: float, fs: float) -> float:
    """頻率 f 以 fs 取樣後看到的(混疊)頻率,0…fs/2。"""
    if fs <= 0:
        return f
    r = f % fs
    return fs - r if r > fs / 2.0 else r


def _alias_cands(osc: OscPeak) -> list[float]:
    """瞬時取樣峰 osc.f 的可能真實頻率(f、fs−f、fs+f…;姿態角交叉檢查不過時不含 f)。"""
    return osc.alias_cands()


def _suspect_support(ctx: AxisContext, osc: OscPeak) -> tuple[str, float | None]:
    """疑似震盪(低速 / 只有陀螺儀 / 交叉檢查不過)有沒有迴圈頻率的 CTRL 佐證。

    "p":CTRL 的 D/P 比低(P 項主導)且等效頻率與某個(控制頻寬內的)候選頻率相符 → 很可能是 P 偏高的震盪;
    "d":CTRL D/P ≥ D_DOMINANT_RATIO → D 項主導;"":沒有佐證(也可能只是馬達 / 機架噪音混疊)。
    回傳 (類型, 相符的頻率)。
    """
    c = ctx.ctrl
    if c is None or c.ratio is None:
        return "", None
    if ctx.axis != "yaw" and c.ratio >= D_DOMINANT_RATIO:
        return "d", None
    fmax = ctrl_fmax(ctx.veh.params)
    cands = [f for f in _alias_cands(osc) if OSC_MID_HZ <= f <= fmax]
    if c.ratio < CTRL_D_OVER_P_WARN and c.f_est is not None and cands:
        near = min(cands, key=lambda f: abs(f - c.f_est))
        if abs(near - c.f_est) <= SUSPECT_FMATCH_FRAC * near:
            return "p", near
    return "", None


def _no_hidden_hf(ctx: AxisContext) -> bool:
    """低速記錄時,有沒有證據顯示「看不到的高頻震盪」不存在(才敢把低頻落差當成反應太慢)。

    Roll / Pitch:CTRL(迴圈頻率計算)的 D/P 比低、推估的誤差頻率 < OSC_MID_HZ;
    Yaw:偏航環本身很慢(FLTE),10 Hz 已看得到其動態,只要求沒有疑似震盪(呼叫端已確認)。
    """
    if ctx.axis == "yaw":
        return True
    c = ctx.ctrl
    return bool(c is not None and c.ratio is not None and c.ratio < CTRL_D_OVER_P_WARN
                and c.f_est is not None and c.f_est < OSC_MID_HZ)


def _low_rate_alias_peak(ctx: AxisContext) -> OscPeak | None:
    """低速記錄的角速度誤差中,可能是混疊的持續窄峰(RMS ≥ OSC_RATE_WARN_RMS)。

    只在 CTRL 推估誤差能量 ≥ OSC_MID_HZ(不在低頻)時採用:誤差能量在低頻時,這種峰多半是真的慢速擺動
    (例如角速度環偏軟造成的 1 Hz 擺動),交給其他規則。D 項主導(D/P ≥ D_DOMINANT_RATIO)另有規則。
    """
    r, c = ctx.rate, ctx.ctrl
    if r is None or not _low_fs(r.fs) or c is None or c.f_est is None or c.ratio is None:
        return None
    if c.f_est < OSC_MID_HZ or c.ratio >= D_DOMINANT_RATIO:
        return None
    pk = [p for p in r.peaks if p.rms >= OSC_RATE_WARN_RMS and p.sustained]
    return max(pk, key=lambda p: p.rms) if pk else None


def _motor_blocked(ctx: AxisContext) -> bool:
    """馬達飽和 / 推力不足:姿態誤差先歸因於沒有控制裕度,不怪角度環。"""
    v = ctx.veh
    return v.thrust_loss or (v.motor_sat is not None and v.motor_sat >= SAT_SHARE_WARN)


def _fast_log_hint(veh: VehicleContext) -> str:
    """「開高速記錄」的具體做法(Copter 與 Plane 的 LOG_BITMASK 位元不同)。"""
    if veh.prefix == "Q_A_":
        return "LOG_BITMASK 加 ATTITUDE_FULLRATE bit 20"
    return "LOG_BITMASK 加 ATTITUDE_FAST bit 0 / PID bit 12"


def _bat_mask_suggestion(params: dict, priority: int = 2) -> Suggestion:
    return Suggestion("INS_LOG_BAT_MASK", DIR_SET,
                      "開啟第一顆 IMU 的批次取樣(FFT),才量得到馬達噪音頻率;"
                      "4.2 以後可再設 INS_LOG_BAT_OPT = 4 同時記錄濾波前後",
                      _num(params.get("INS_LOG_BAT_MASK")), 1.0, None, priority)


def _axis_noise_evidence(ctx: AxisContext) -> list[str]:
    """這一軸「雜訊進到控制迴路」的證據(文字;空 = 沒有)。"""
    ev = []
    p = ctx.pid
    if p is not None and p.d_hf_frac is not None and p.d_hf_frac >= D_HF_FRAC_WARN:
        ev.append(f"{_AXIS_SHORT[ctx.axis]} D 項 {p.d_hf_frac * 100:.0f} % 的功率在 ≥ {OSC_HIGH_HZ:g} Hz")
    meas, src = _measured_dp(ctx)
    if ctx.axis != "yaw" and meas is not None and meas >= D_DOMINANT_RATIO:
        ev.append(f"{_AXIS_SHORT[ctx.axis]} {src} D/P 輸出比 {meas:.1f}")
    npk = _noise_peak(ctx)
    if npk is not None:
        ev.append(f"{_AXIS_SHORT[ctx.axis]} 陀螺儀 {npk.f:.0f} Hz 噪音峰")
    return ev


def _noise_finding(ctx: AxisContext, pk: OscPeak) -> Finding:
    """高於控制頻寬上限的峰 → 「陀螺儀噪音」:先陷波 / 量噪音頻率,不降 P(必要時才小降 D)。"""
    prm, nm = ctx.veh.params, ctx.names
    lab = _AXIS_SHORT[ctx.axis]
    fmax = ctrl_fmax(prm)
    gf = _num(prm.get("INS_GYRO_FILTER"))
    en = _num(prm.get("INS_HNTCH_ENABLE"))
    full = _full_rate(pk.fs, prm)
    src = "陀螺儀(IMU)" if pk.source == "imu" else "角速度誤差"
    sev = SEV_WARN if pk.sustained else SEV_INFO
    detail = (f"{src}頻譜在 {pk.f:.0f} Hz 有窄峰(RMS {pk.rms:.1f} °/s,{'持續' if pk.sustained else '偶發'})。"
              f"這個頻率高於姿態控制可能自激的範圍(約 {fmax:.0f} Hz 以下:INS_GYRO_FILTER "
              f"{_fmt_param(gf)} Hz、迴圈 {_loop_rate(prm):.0f} Hz,再加上馬達延遲,迴路在這裡沒有增益),"
              "多半是馬達 / 螺旋槳 / 機架的振動噪音:應該用諧波陷波濾波處理,而不是降 P / D。")
    if not full:
        alias = pk.alias_text()
        if alias:
            detail += f"({pk.fs:.0f} Hz 記錄低於迴圈頻率,真實頻率也可能是 {alias} Hz。)"
    sug: list[Suggestion] = []
    if en is not None and en >= 1:
        sug.append(_check("INS_HNTCH_FREQ / BW / MODE / REF",
                          f"陷波已啟用但仍有 {pk.f:.0f} Hz 噪音:確認陷波有追蹤到這個頻率(或加第二組 INS_HNTC2)", 1))
    elif ctx.veh.batch:
        sug.append(_check("INS_HNTCH_ENABLE(見陷波建議)", "依批次 FFT 量到的噪音峰設定諧波陷波", 1))
    else:
        sug.append(_bat_mask_suggestion(prm, 1))
        sug.append(_check(f"INS_HNTCH_ENABLE / INS_HNTCH_FREQ ≈ {pk.f:.0f} Hz",
                          "用批次取樣確認噪音頻率後啟用諧波陷波:FREQ ≈ 懸停時的噪音峰、BW ≈ FREQ 的一半、"
                          "REF = 懸停油門(頻率先不要只靠這份記錄盲設)", 2))
    sug.append(_check("INS_GYRO_FILTER", f"目前 {_fmt_param(gf)} Hz;陷波設好之前不要調高", 3))
    kd = _num(prm.get(nm["D"]))
    p = ctx.pid
    if ctx.axis != "yaw" and kd and p is not None and p.d_hf_frac is not None and p.d_hf_frac >= D_HF_FRAC_WARN:
        detail += f"D 項有 {p.d_hf_frac * 100:.0f} % 的功率在 ≥ {OSC_HIGH_HZ:g} Hz(噪音進到 D 項,馬達會發燙)。"
        sug.append(gain_suggestion(nm["D"], prm, STEP_D_DAMP_DOWN,
                                   "噪音進到 D 項、馬達發燙時,陷波設好前可先小降 D", 3))
    return Finding("gyro_noise", sev, ctx.axis, f"{lab} 陀螺儀 {pk.f:.0f} Hz 噪音峰(高於控制頻寬,不是 PID 震盪)",
                   detail, [s for s in sug if s], pk.t_focus, pk.window,
                   {"noise_hz": pk.f, "noise_rms": pk.rms, "sustain": pk.sustain})


def axis_rules(ctx: AxisContext) -> list[Finding]:
    """單軸診斷規則(多旋翼)。每條規則註明 ArduPilot 的調參依據。"""
    out: list[Finding] = []
    ax = ctx.axis
    yaw = ax == "yaw"
    lab = _AXIS_SHORT[ax]
    nm = ctx.names
    prm = ctx.veh.params
    rate, ang = ctx.rate, ctx.ang
    kp, kd = _num(prm.get(nm["P"])), _num(prm.get(nm["D"]))
    osc, conf = _osc_evidence(ctx)
    osc_flag = False        # 直接看到的震盪 / 阻尼問題(擋掉其他規則)
    suspect = False         # 低速記錄的疑似震盪(擋掉「提高增益」類規則)
    sus_osc: OscPeak | None = None      # 疑似震盪的 IMU 峰(角度層級排除它的混疊)
    motor_blocked = _motor_blocked(ctx)

    # ---- A0. 高於控制頻寬的陀螺儀噪音峰(不是 PID 震盪)→ 陷波,不降 P / D ----
    npk = _noise_peak(ctx)
    if npk is not None:
        out.append(_noise_finding(ctx, npk))

    # ---- A. 角速度環震盪 ----
    # ArduPilot 調參指南:高頻(≥ 15–20 Hz)抖動多半是 D 太高(或濾波不足)→ 先降 D 再降 P;
    # 4–15 Hz 的擺動多半是 P 太高 → 降 P(Roll/Pitch 的 I 跟著 P);D 相對 P 偏高時先降 D。
    if osc is not None and conf == "direct" and osc.rms >= OSC_RATE_WARN_RMS:
        osc_flag = True
        sev = SEV_BAD if (osc.sustained and osc.rms >= OSC_RATE_BAD_RMS) else \
            (SEV_WARN if osc.sustained else SEV_INFO)
        src = "陀螺儀(IMU)頻譜" if osc.source == "imu" else "角速度誤差頻譜"
        when = "幾乎整段飛行都存在" if osc.sustained else f"只出現在約 {osc.sustain * 100:.0f} % 的時間"
        base = (f"{src}在 {osc.f:.1f} Hz 有明顯峰值(RMS {osc.rms:.1f} °/s,"
                f"高出附近底噪 {10 * math.log10(max(osc.prominence, 1.0)):.0f} dB),{when}。")
        if osc.phys is not None:
            base += (f"({osc.fs:.0f} Hz 記錄低於迴圈頻率;姿態角在同頻率有對應的擺動"
                     f"(預期的 {osc.phys * 100:.0f} %),確認是真的 {osc.f:.1f} Hz 擺動、不是噪音混疊。)")
        sug: list[Suggestion] = []
        meas_dp, dp_src = _measured_dp(ctx)
        if osc.f >= OSC_HIGH_HZ:
            code, title = "rate_osc_high", f"{lab} 角速度高頻震盪 約 {osc.f:.0f} Hz"
            detail = base + "這個頻段的震盪多半是 D 太高或濾波不足(馬達噪音進到 D 項)。"
            if not yaw and kd:
                sug.append(gain_suggestion(nm["D"], prm, STEP_D_OSC_HIGH, "高頻震盪先降 D", 1))
                sug.append(gain_suggestion(nm["P"], prm, STEP_P_OSC_HIGH, "降 D 後仍震盪再降 P", 2))
            else:
                sug.append(gain_suggestion(nm["P"], prm, STEP_YAW_P_DOWN if yaw else STEP_P_OSC_HIGH,
                                           "高頻震盪:降低角速度 P", 1))
            sug.append(_check("INS_GYRO_FILTER / 陷波濾波 / " + nm["FLTD"],
                              "確認陀螺儀低通、諧波陷波與 D 項濾波設定,馬達噪音不要進到控制迴路", 3))
        elif not yaw and kd and _d_dominant(ctx, osc.f):
            # D 相對 P 偏高(增益比),或 D 項輸出遠超過這個頻率應有的比例(吃到雜訊)→ 先降 D
            code, title = "rate_osc_mid", f"{lab} 角速度環震盪 約 {osc.f:.1f} Hz"
            exp = _expected_dp(ctx, osc.f)
            if kp and kd / kp > D_OVER_P_HIGH:
                detail = base + (f"D / P 增益比 {kd / kp:.3f}(預設約 0.027)偏高:D 相對 P 太大,"
                                 "這種中頻擺動先降 D。")
            elif ctx.pid is not None and (ctx.pid.d_hf_frac or 0.0) >= D_HF_FRAC_WARN:
                detail = base + (f"D 項有 {ctx.pid.d_hf_frac * 100:.0f} % 的功率在 ≥ {OSC_HIGH_HZ:g} Hz:"
                                 "D 項吃到大量高頻雜訊,先降 D(並檢查濾波)。")
            else:
                detail = base + (f"{dp_src} 的 D 項 / P 項輸出比 {meas_dp:.2f},遠大於這個頻率依增益應有的約 "
                                 f"{exp:.2f}:D 項吃到額外的高頻雜訊,先降 D。")
            sug.append(gain_suggestion(nm["D"], prm, STEP_D_OSC_HIGH, "D 偏高的震盪:先降 D", 1))
            sug.append(gain_suggestion(nm["P"], prm, STEP_P_OSC_HIGH, "降 D 後仍震盪再降 P", 2))
        else:
            code, title = "rate_osc_mid", f"{lab} 角速度環震盪 約 {osc.f:.1f} Hz"
            detail = base + "這個頻段的持續擺動多半是角速度 P 太高(ArduPilot:震盪時先降 P、I,必要時 D 也一起降)。"
            ps = gain_suggestion(nm["P"], prm, STEP_YAW_P_DOWN if yaw else STEP_P_OSC_MID,
                                 "中頻震盪:降低角速度 P", 1)
            sug.append(ps)
            fi = _follow_i(ctx, ps, 2)
            if fi:
                sug.append(fi)
            if not yaw and kd and meas_dp is not None and meas_dp >= CTRL_D_OVER_P_WARN:
                exp = _expected_dp(ctx, osc.f)
                detail += (f"{dp_src} 的 D 項 / P 項輸出比 {meas_dp:.2f}"
                           + (f"(這個頻率依增益預期約 {exp:.2f})" if exp else "")
                           + ":在此頻率 D 項輸出本來就較大,不代表 D 太高。")
                sug.append(gain_suggestion(nm["D"], prm, STEP_D_WITH_P, "降 P、I 後仍震盪,D 也一起小降", 3))
        detail += _ctrl_text(ctx)
        out.append(Finding(code, sev, ax, title, detail, [s for s in sug if s],
                           osc.t_focus, osc.window,
                           {"osc_hz": osc.f, "osc_rms": osc.rms, "sustain": osc.sustain}))

    # ---- A2. 疑似震盪(低速記錄 / 只有陀螺儀 / 低於迴圈頻率且姿態角對不上:可能是噪音混疊)----
    elif osc is not None and conf == "suspect" and osc.rms >= OSC_SUSPECT_MIN_RMS:
        # 記錄是瞬時取樣(沒有抗混疊):峰可能是 5–20 Hz 的控制震盪,也可能是馬達 / 機架噪音
        # 混疊下來。只有迴圈頻率的 CTRL 也指向同一件事時,才給出調整方向(WARN);否則只提醒。
        suspect = True
        sus_osc = osc
        support, f_match = _suspect_support(ctx, osc)
        strong = osc.sustained and osc.rms >= OSC_RATE_WARN_RMS and bool(support) and not yaw
        sev = SEV_WARN if strong else SEV_INFO
        full = _full_rate(osc.fs, prm)
        alias = "" if full else osc.alias_text()
        is_imu = osc.source == "imu"
        what = "陀螺儀" if is_imu else "角速度"
        low = _low_fs(osc.fs)
        if support:
            title = f"{lab} 疑似震盪({what} {osc.f:.1f} Hz,{'低速記錄待確認' if low else '待迴圈頻率記錄確認'})"
        elif low:
            title = f"{lab} 陀螺儀 {osc.f:.1f} Hz 有峰(低速記錄:震盪或噪音混疊,無法判斷)"
        elif osc.phys_failed:
            title = f"{lab} 角速度 {osc.f:.1f} Hz 有峰,但姿態角沒有對應擺動(多半是噪音混疊)"
        else:
            title = f"{lab} {what} {osc.f:.1f} Hz 有峰(無法確認是控制震盪)"
        if is_imu and full:
            detail = (f"IMU 陀螺儀以 {osc.fs:.0f} Hz 記錄(約迴圈頻率,沒有記錄造成的混疊),頻譜在 {osc.f:.1f} Hz"
                      f" 有峰值(RMS {osc.rms:.1f} °/s,{'持續' if osc.sustained else '偶發'});"
                      "但沒有迴圈頻率的角速度 / PID 記錄能確認它是控制迴路震盪(也可能是機架共振或亂流)。")
        elif is_imu:
            detail = (f"IMU 陀螺儀只有 {osc.fs:.0f} Hz 取樣(瞬時取樣、沒有抗混疊),頻譜在 {osc.f:.1f} Hz 有峰值"
                      f"(RMS {osc.rms:.1f} °/s,{'持續' if osc.sustained else '偶發'})。")
        else:
            detail = (f"角速度只以 {osc.fs:.0f} Hz 記錄(低於迴圈頻率 {_loop_rate(prm):.0f} Hz;"
                      f"記錄是瞬時取樣、沒有抗混疊),誤差頻譜在 {osc.f:.1f} Hz 有峰值"
                      f"(RMS {osc.rms:.1f} °/s,{'持續' if osc.sustained else '偶發'})。")
            if osc.phys is not None:
                detail += (f"但姿態角(ATT)在同頻率的擺動只有真的 {osc.f:.1f} Hz 角速度擺動應有的 "
                           f"{osc.phys * 100:.0f} %:多半是更高頻的馬達 / 機架噪音混疊下來,不是控制震盪。")
            else:
                detail += "沒有同速率的姿態角可以交叉檢查。"
        if alias and osc.phys_failed:
            detail += f"在 {osc.fs:.0f} Hz 記錄下會混疊到 {osc.f:.1f} Hz 的真實頻率有 {alias} Hz。"
        elif alias:
            detail += f"受混疊影響,真實頻率可能是 {alias} Hz 其中之一,也可能是更高頻的馬達 / 機架噪音混疊下來。"
        detail += _ctrl_text(ctx)
        if support == "p":
            detail += (f"CTRL(迴圈頻率計算)換算的等效頻率與可能頻率 {f_match:.1f} Hz 吻合,且 P 項主導:"
                       "很可能是角速度 P 偏高造成的震盪。")
        elif support == "d":
            detail += "CTRL 顯示 D 項輸出遠大於 P 項:很可能是 D 偏高造成的高頻抖動。"
        if rate is not None and rate.incoherent is not None and rate.incoherent > 0.5:
            detail += f"角速度實際值有 {rate.incoherent * 100:.0f} % 的功率與期望值無關。"
        sug = []
        if strong:
            detail += "建議先開啟迴圈頻率記錄確認頻率;若目視 / 聽得到抖動,可先保守地降低增益。"
            if support == "d" and kd:
                sug.append(gain_suggestion(nm["D"], prm, STEP_SUSPECT_D,
                                           "D 項主導的疑似高頻震盪:先小降 D(確認後再調)", 2))
                sug.append(gain_suggestion(nm["P"], prm, STEP_SUSPECT_P, "降 D 後仍抖再降 P", 3))
            else:
                ps = gain_suggestion(nm["P"], prm, STEP_SUSPECT_P,
                                     "疑似角速度 P 偏高的震盪:小降 P(確認後再調)", 2)
                sug.append(ps)
                fi = _follow_i(ctx, ps, 3)
                if fi:
                    sug.append(fi)
        else:
            how = _fast_log_hint(ctx.veh) if low else "LOG_FILE_RATEMAX = 0,串流訊息不限速"
            if osc.phys_failed:
                detail += (f"姿態角沒有對應的擺動,代表機體本身沒有在 {osc.f:.1f} Hz 擺動:先不要改增益;"
                           f"請開迴圈頻率記錄({how})與批次取樣(INS_LOG_BAT_MASK)確認噪音頻率,再設陷波。")
            else:
                detail += (f"CTRL 沒有足夠佐證,無法確定是控制迴路震盪,先不建議改增益:請開迴圈頻率記錄({how})"
                           "與批次取樣(INS_LOG_BAT_MASK)再判斷。")
                if not yaw:
                    detail += (f"若飛行中確實看得到 / 聽得到高頻抖動(馬達聲尖、發燙),再先降 {nm['D']} 約 20 %,"
                               f"仍抖才降 {nm['P']} 約 15 %。")
        if not low and not is_imu:
            rmx = _num(prm.get("LOG_FILE_RATEMAX"))
            if rmx is not None and rmx > 0 and not _full_rate(osc.fs, prm):
                sug.append(Suggestion("LOG_FILE_RATEMAX", DIR_SET,
                                      "取消串流訊息限速,ATT / RATE / PID 以迴圈頻率記錄才看得到真實頻率"
                                      "(記錄檔會變大,調參飛行時再開)", rmx, 0.0, None, 2))
            if not ctx.veh.batch:
                sug.append(_bat_mask_suggestion(prm, 2))
        out.append(Finding("rate_osc_suspect", sev, ax, title, detail, sug, osc.t_focus,
                           osc.window, {"osc_hz": osc.f, "osc_rms": osc.rms, "imu_hz": osc.fs,
                                        "support": support or None, "phys": osc.phys}))
    elif (lpk := _low_rate_alias_peak(ctx)) is not None:
        # 低速角速度誤差的窄峰:可能是真的慢速擺動,也可能是 5–20 Hz 震盪混疊;CTRL 顯示誤差能量不在低頻,
        # 但無法分辨來源(例:角度 P 過高的 4–5 Hz 共振也長這樣)→ 只提醒,不改增益、不擋其他規則
        cands = "、".join(f"{c:.1f}" for c in sorted({round(c, 1) for c in
                                                      (lpk.f, rate.fs - lpk.f, rate.fs + lpk.f) if c > 0}))
        out.append(Finding(
            "rate_osc_suspect", SEV_INFO, ax, f"{lab} 角速度誤差在 {lpk.f:.1f} Hz 有窄峰(低速記錄,無法判斷來源)",
            f"{rate.fs:.0f} Hz 的角速度誤差在 {lpk.f:.1f} Hz 有明顯的持續峰(RMS {lpk.rms:.1f} °/s)。"
            f"取樣太慢,這可能是真的 {lpk.f:.1f} Hz 擺動,也可能是 {cands} Hz 的震盪混疊下來;"
            f"CTRL(迴圈頻率)換算的等效頻率約 {ctx.ctrl.f_est:.0f} Hz,不在低頻。"
            f"請開高速記錄({_fast_log_hint(ctx.veh)})再判斷,先不改增益。",
            [], lpk.t_focus, lpk.window, {"osc_hz": lpk.f, "osc_rms": lpk.rms, "support": None}))
    elif (rate is not None and _low_fs(rate.fs) and ctx.ctrl is not None
          and ctx.ctrl.ratio is not None and ctx.ctrl.ratio >= CTRL_D_OVER_P_WARN
          and (ctx.pid is not None or yaw or ctx.ctrl.ratio < D_DOMINANT_RATIO)
          and (rate.rms_err or 0) >= OSC_RATE_WARN_RMS):
        suspect = True
        out.append(Finding(
            "rate_osc_suspect", SEV_INFO, ax, f"{lab} 誤差偏高頻(低速記錄看不到)",
            f"角速度誤差 RMS {rate.rms_err:.1f} °/s;" + _ctrl_text(ctx)
            + f"{rate.fs:.0f} Hz 的記錄看不到 5–20 Hz,請開高速記錄確認是否有震盪。",
            [], ctx.ctrl.t_spike, None, {"ctrl_ratio": ctx.ctrl.ratio}))

    # ---- A3. PIDx:slew 限制器常啟動(Dmod < 1)→ 有震盪傾向 ----
    if ctx.pid is not None and ctx.pid.dmod_share is not None and ctx.pid.dmod_share >= DMOD_WARN_SHARE:
        osc_flag = True
        sug = [gain_suggestion(nm["D"], prm, STEP_SLEW_D, "slew 限制器常啟動:先降 D", 1)] \
            if (not yaw and kd) else []
        sug.append(gain_suggestion(nm["P"], prm, STEP_P_DAMP, "仍常啟動再降 P", 2))
        sr = ""
        if ctx.pid.srate_p95 is not None and ctx.pid.smax:
            sr = f"輸出變化率 SRate 第 95 百分位 {ctx.pid.srate_p95:.1f}(SMAX = {ctx.pid.smax:g})。"
        out.append(Finding(
            "slew_limit", SEV_WARN, ax, f"{lab} slew 限制器常啟動(Dmod < 1)",
            f"PID 記錄中 {ctx.pid.dmod_share * 100:.0f} % 的時間 Dmod < {DMOD_ACTIVE:g}:"
            f"ArduPilot 偵測到輸出變化過快而自動壓低 P/D,代表迴路有震盪傾向。" + sr,
            [s for s in sug if s], ctx.pid.t_dmod, None, {"dmod_pct": ctx.pid.dmod_share * 100}))

    # ---- A3b. PIDx:D 項輸出遠大於 P 項 → D 偏高(調得好的機 D/P 約 0.5–1.5)----
    # 沒有 PIDx 時改用 CTRL(RMSRollD / RMSRollP 由飛控以迴圈頻率計算,不受記錄頻率影響)
    has_d_down = any(s.param == nm["D"] and s.direction == DIR_DOWN for f in out for s in f.suggestions)
    r_dp, src_dp = None, ""
    if ctx.pid is not None and ctx.pid.p_rms > 0:
        r_dp, src_dp = ctx.pid.d_rms / ctx.pid.p_rms, "PID 記錄中"
    elif ctx.pid is None and ctx.ctrl is not None and ctx.ctrl.ratio is not None:
        r_dp, src_dp = ctx.ctrl.ratio, "CTRL(飛控以迴圈頻率計算,不受記錄頻率影響)顯示"
    if r_dp is not None and not yaw and kd and not has_d_down and r_dp >= D_DOMINANT_RATIO:
        osc_flag = True
        extra = ""
        if ctx.pid is None and ctx.ctrl is not None and ctx.ctrl.f_est is not None:
            extra = f"依增益推估誤差能量集中在約 {ctx.ctrl.f_est:.0f} Hz。"
        out.append(Finding(
            "d_dominant", SEV_WARN, ax, f"{lab} D 項輸出過大(D/P = {r_dp:.1f})",
            f"{src_dp} D 項輸出 RMS 是 P 項的 {r_dp:.1f} 倍(調得好的機通常約 0.5–1.5 倍):"
            "D 增益偏高或 D 項吃到太多雜訊,會讓馬達發熱、容易出現 10–20 Hz 抖動。" + extra,
            [gain_suggestion(nm["D"], prm, STEP_D_OSC_HIGH, "D 項輸出過大:降 D", 1),
             _check("INS_GYRO_FILTER / 陷波濾波 / " + nm["FLTD"], "確認雜訊沒有進到 D 項", 3)],
            ctx.ctrl.t_spike if (ctx.pid is None and ctx.ctrl is not None) else None, None,
            {"pid_d_over_p" if ctx.pid is not None else "ctrl_d_over_p": r_dp}))

    # ---- A4. PIDx:D 項高頻雜訊 ----
    if ctx.pid is not None and ctx.pid.d_hf_frac is not None and ctx.pid.d_hf_frac >= D_HF_FRAC_WARN \
            and ctx.pid.d_rms >= 0.5 * max(ctx.pid.p_rms, 1e-9) and not yaw:
        out.append(Finding(
            "d_noise", SEV_INFO, ax, f"{lab} D 項高頻雜訊大",
            f"D 項輸出有 {ctx.pid.d_hf_frac * 100:.0f} % 的功率在 ≥ {OSC_HIGH_HZ:g} Hz:"
            "馬達 / 機架噪音進到 D 項,會讓馬達發熱並限制可用的 D。",
            [_check("INS_GYRO_FILTER / 陷波濾波 / " + nm["FLTD"],
                    "檢查陀螺儀低通、諧波陷波與 D 項濾波;不要再加 D", 2)],
            None, None, {"d_hf_pct": ctx.pid.d_hf_frac * 100}))

    sat_axis = rate.sat_share if rate is not None else None
    if ctx.pid is not None and ctx.pid.limit_share is not None:
        # PIDx limit 旗標在馬達頂到上限或壓到下限時都會設;只和「頂到上限」的比例一起採用
        # (壓到下限多半是急降或雜訊推動輸出,另由整機的 motor_sat_low 處理)
        lim = ctx.pid.limit_share
        if ctx.veh.motor_hi is not None:
            lim = min(lim, ctx.veh.motor_hi)
        sat_axis = max(sat_axis or 0.0, lim)

    # ---- B. 角速度環阻尼不足(沒有持續震盪)----
    # |H| 共振峰 > 1.3 或步階過衝 > 25 %:P 相對 D 太高 → 降 P 10 %,或(雜訊小、D 不高時)加 D 15 %。
    # 只有 |H| 峰(沒有可信的步階過衝)時是弱證據:期望角速度含角度環回授,|H| 會被墊高 → info、順位延後。
    if rate is not None and not osc_flag and not suspect and _fs_ok(rate.fs, INCREASE_MIN_HZ):
        hp = (rate.h_peak is not None and rate.h_peak > HPEAK_RATE_WARN
              and (rate.h_peak_hz or 0.0) >= HPEAK_RATE_FMIN_HZ)
        ov = rate.step_usable() and rate.overshoot > OVERSHOOT_RATE_WARN
        if hp or ov:
            ev = []
            if ov:
                ev.append(f"步階響應過衝約 {rate.overshoot:.0f} %")
            if hp:
                ev.append(f"頻率響應共振峰 |H| = {rate.h_peak:.2f}(@ {rate.h_peak_hz:.1f} Hz)")
            pr0 = 1 if ov else 3
            if not yaw and kd and _d_dominant(ctx, rate.h_peak_hz if hp else None):
                # D 相對 P 已偏高(增益比,或 D 項輸出遠超過該頻率應有的比例):共振多半由 D 造成 → 先小降 D
                sug = [gain_suggestion(nm["D"], prm, STEP_D_DAMP_DOWN, "D 相對 P 偏高的共振:先小降 D", pr0),
                       gain_suggestion(nm["P"], prm, STEP_P_DAMP, "仍過衝再小降 P", pr0 + 1)]
            else:
                sug = [gain_suggestion(nm["P"], prm, STEP_YAW_P_DOWN if yaw else STEP_P_DAMP,
                                       "阻尼不足:小降 P", pr0)]
                if ov and not yaw and kd and not _hf_noise(ctx):
                    sug.append(gain_suggestion(nm["D"], prm, STEP_D_DAMP, "或提高 D 增加阻尼(雜訊小時)", 2))
            tail = "。沒有持續震盪,但反應會過衝、回擺。"
            if not ov:
                tail += "(只有頻率響應的證據;期望角速度含角度環回授,|H| 可能偏高,建議做幾次快速搖桿動作再確認。)"
            out.append(Finding("rate_underdamped", SEV_WARN if ov else SEV_INFO, ax,
                               f"{lab} 角速度環阻尼不足", ";".join(ev) + tail, sug,
                               rate.t_max_err, None,
                               {"overshoot_pct": rate.overshoot, "h_peak": rate.h_peak}))
            osc_flag = True

    # ---- C. 角速度環偏軟 ----
    # 延遲大 / 頻寬低 / 上升慢,且過衝小、沒飽和 → 提高 P 15–25 %(D 按比例;Yaw:I = P/10)。
    # 追蹤誤差大只是旁證。先判這條:角速度環太軟也會讓角度環過衝,那時不該怪角度 P。
    sluggish = False
    if rate is not None and not osc_flag and not suspect:
        ind = []
        fast = _fs_ok(rate.fs, INCREASE_MIN_HZ)
        if yaw:
            lag_lim = LAG_YAW_RATE_WARN_MS
        else:
            lag_lim = LAG_RATE_WARN_MS if fast else LAG_RATE_WARN_LOW_MS
        if rate.lag_ms is not None and (rate.lag_rho or 0) >= LAG_MIN_RHO and rate.lag_ms > lag_lim:
            ind.append(f"期望 → 實際延遲約 {rate.lag_ms:.0f} ms")
        bw_lim = BW_YAW_RATE_LOW_HZ if yaw else BW_RATE_LOW_HZ
        bw = rate.bw_hz if rate.bw_hz is not None else (rate.bw_lt if fast else None)
        if bw is not None and bw < bw_lim:
            ind.append(f"頻寬約 {bw:.1f} Hz" if rate.bw_hz is not None else f"頻寬 < {bw:.1f} Hz")
        if rate.step_usable() and rate.rise_s is not None and rate.rise_s > RISE_RATE_SLOW_S:
            ind.append(f"上升時間約 {rate.rise_s * 1000:.0f} ms")
        tr = rate.track_ratio
        dyn = bool(ind)
        if tr is not None and tr > TRACK_RATIO_WARN:
            ind.append(f"RMS 誤差 {rate.rms_err:.1f} °/s(為期望活動量的 {tr * 100:.0f} %)")
        ov_ok = not rate.step_usable() or rate.overshoot < OVERSHOOT_SLUGGISH_MAX
        hp_ok = (rate.h_peak is None or rate.h_peak < HPEAK_SLUGGISH_MAX
                 or (rate.h_peak_hz or 0.0) < HPEAK_RATE_FMIN_HZ
                 or not fast)
        sat_ok = sat_axis is None or sat_axis < SAT_SHARE_WARN
        if dyn and ov_ok and hp_ok and sat_ok:
            # 角速度環偏軟時,角度層級的擺動 / 落後也可能是它造成的 → 不再怪角度環
            sluggish = True
            if not fast:
                # 低速記錄:延遲夠大、期望與實際高度相關、搖桿直接給姿態(非位置控制回授),
                # 且迴圈頻率的 CTRL 顯示誤差能量在低頻(沒有看不到的高頻震盪)→ 才保守地建議 +10 %
                strong = ((rate.lag_rho or 0) >= LAG_MIN_RHO_LOW and not ctx.veh.closed_loop_des
                          and _no_hidden_hf(ctx))
                why_low = (f"只有 {rate.fs:.0f} Hz 取樣(Nyquist {rate.fs / 2:.0f} Hz):"
                           "步階響應 / 過衝量不到、5 Hz 以上的震盪也看不到。")
                if strong:
                    if yaw:
                        ps = gain_suggestion(nm["P"], prm, STEP_P_UP_LOW,
                                             "偏航反應慢:小幅提高 Yaw P(低速資料,先開高速記錄確認更好)", 2)
                    else:
                        ps = gain_suggestion(nm["P"], prm, STEP_P_UP_LOW,
                                             "反應偏軟:小幅提高角速度 P(低速資料,先開高速記錄確認更好)", 2)
                    sug = [ps]
                    fi = _follow_i(ctx, ps, 3)
                    if fi:
                        sug.append(fi)
                    ctrl_txt = "" if yaw else (f"CTRL(迴圈頻率)推估誤差能量集中在約 {ctx.ctrl.f_est:.1f} Hz,"
                                               "沒有高頻震盪的跡象;")
                    out.append(Finding(
                        "rate_sluggish", SEV_WARN, ax, f"{lab} 角速度環偏軟({rate.fs:.0f} Hz 資料)",
                        ";".join(ind) + f"(相關係數 {rate.lag_rho:.2f})。" + ctrl_txt + why_low
                        + "這個落差在低頻、看得到,可以小幅提高增益;每次改完都要確認沒有開始抖動。",
                        sug, rate.t_max_err, None, {"lag_ms": rate.lag_ms, "track_ratio": tr}))
                else:
                    why = []
                    if ctx.veh.closed_loop_des:
                        why.append("期望姿態主要由位置控制(LOITER / AUTO…)產生,期望角速度會隨實際運動回授")
                    if (rate.lag_rho or 0) < LAG_MIN_RHO_LOW:
                        why.append(f"期望與實際的相關係數只有 {rate.lag_rho:.2f}")
                    if not yaw and not _no_hidden_hf(ctx):
                        why.append("沒有 CTRL 佐證排除看不到的高頻震盪(混疊也會造成這種落差)")
                    det = ";".join(ind) + "。" + why_low
                    if why:
                        det += "、".join(why) + "。"
                    det += "無法分辨是反應太慢還是量測限制,先不建議提高增益;請開高速記錄再判斷。"
                    out.append(Finding(
                        "rate_sluggish", SEV_INFO, ax, f"{lab} 角速度追蹤落差大(低速資料,僅供參考)",
                        det, [], rate.t_max_err, None, {"lag_ms": rate.lag_ms, "track_ratio": tr}))
            else:
                if yaw:
                    ps = gain_suggestion(nm["P"], prm, STEP_YAW_P_UP, "偏航反應慢:提高 Yaw P", 1)
                else:
                    ps = gain_suggestion(nm["P"], prm, STEP_P_UP, "反應偏軟:提高角速度 P", 1)
                sug = [ps]
                fi = _follow_i(ctx, ps, 2)
                if fi:
                    sug.append(fi)
                if not yaw and kd and not _hf_noise(ctx):
                    sug.append(gain_suggestion(nm["D"], prm, STEP_P_UP, "D 按比例跟著 P", 3))
                out.append(Finding("rate_sluggish", SEV_WARN, ax, f"{lab} 角速度環偏軟",
                                   ";".join(ind) + "。過衝小,可以考慮小幅提高增益。", sug,
                                   rate.t_max_err, None,
                                   {"lag_ms": rate.lag_ms, "bandwidth_hz": rate.bw_hz,
                                    "track_ratio": tr}))

    # ---- D. 角度環慢速擺動 / 過衝(角速度環正常時才歸給角度環)----
    # ArduPilot:角度 P 太高會在 1–3 Hz 慢速過衝 / 擺動 → 降 ATC_ANG_xxx_P。
    # (ATC_INPUT_TC 不列:ATT.Des* 是 INPUT_TC 整形後的目標,量到的期望 → 實際響應與它無關,它只改搖桿手感。)
    # 馬達飽和時姿態誤差先歸因於沒有控制裕度(不怪角度 P)。
    if ang is not None and not osc_flag and not sluggish and not motor_blocked:
        a_warn, a_bad = (OSC_YAW_ANG_WARN_RMS, OSC_YAW_ANG_BAD_RMS) if yaw else (OSC_ANG_WARN_RMS,
                                                                                OSC_ANG_BAD_RMS)
        wob = [p for p in ang.peaks if p.f < OSC_MID_HZ and p.rms >= a_warn]
        if sus_osc is not None:
            # 角度誤差峰正好落在疑似高頻震盪的混疊頻率 → 是同一個震盪,不是角度環的問題
            al = [_fold(c, ang.fs) for c in _alias_cands(sus_osc)]
            wob = [p for p in wob if all(abs(p.f - a) > ALIAS_MATCH_HZ for a in al)]
        over = sus_osc is None and ang.step_usable() and ang.overshoot > OVERSHOOT_ANG_WARN
        # |H| 共振峰只當 Roll / Pitch 的證據,且期望姿態要直接來自搖桿:
        # 偏航的期望航向會被拉向實際航向、LOITER / AUTO 的期望姿態含位置回授,|H| 都會被墊高、無法判讀
        hpk = (not yaw and sus_osc is None and not ctx.veh.closed_loop_des
               and ang.h_peak is not None and ang.h_peak > HPEAK_ANG_WARN)
        if wob or over or hpk:
            ev = []
            pk = max(wob, key=lambda p: p.rms) if wob else None
            if pk is not None:
                ev.append(f"角度誤差在 {pk.f:.1f} Hz 有擺動(RMS {pk.rms:.2f}°,"
                          f"{'持續' if pk.sustained else '偶發'})")
            if over:
                ev.append(f"角度步階過衝約 {ang.overshoot:.0f} %")
            if hpk:
                ev.append(f"角度頻率響應共振峰 |H| = {ang.h_peak:.2f}(@ {ang.h_peak_hz:.1f} Hz)")
            # 證據強弱:誤差頻譜的持續擺動峰最可靠;步階在定點 / 自動模式下會被位置控制回授扭曲;
            # 低速記錄的 |H| 峰接近 Nyquist、只當弱證據
            weak = pk is None and (_low_fs(ang.fs) or ctx.veh.closed_loop_des)
            if pk is not None:
                sev = SEV_BAD if (pk.sustained and pk.rms >= a_bad and not _low_fs(ang.fs)) \
                    else (SEV_WARN if pk.sustained else SEV_INFO)
            else:
                sev = SEV_INFO if weak else SEV_WARN
            pr0 = 3 if weak else (2 if _low_fs(ang.fs) else 1)
            sug = [gain_suggestion(nm["ANG"], prm, STEP_ANG_DOWN, "角度環過衝 / 擺動:降低角度 P", pr0)]
            if rate is not None and _fs_ok(rate.fs, INCREASE_MIN_HZ):
                why = "。角速度環沒有明顯問題,問題在外環(角度 P)。"
            else:
                why = ("。最可能是外環(角度 P)太積極;但這份資料無法確認角速度環"
                       "(角速度環本身偏慢也會讓角度環擺動),改完請用高速記錄再確認。")
            if weak and ctx.veh.closed_loop_des:
                why += ("期望姿態主要由位置控制(LOITER / AUTO…)產生,會隨機體實際運動回授,"
                        "|H| 與步階估計可能偏高;要確認請在 STABILIZE / ALT_HOLD 下做幾次快速搖桿動作再分析。")
            if not yaw:
                why += (f"({nm['TC']} 只改變搖桿 → 期望姿態的平滑程度(手感),不會改變這裡量到的期望 → 實際響應,"
                        "所以不列為修正項。)")
            code = "angle_wobble" if pk is not None else ("angle_overshoot" if over else "angle_freq_peak")
            out.append(Finding(
                code, sev, ax,
                f"{lab} 角度環{'擺動' if pk is not None else ('過衝' if over else '共振(|H| 峰)')}",
                ";".join(ev) + why,
                sug, pk.t_focus if pk else ang.t_max_err, pk.window if pk else None,
                {"osc_hz": pk.f if pk else None, "overshoot_pct": ang.overshoot}))
            osc_flag = True

    # ---- E. 角度環偏慢(角速度環正常)----
    # 角速度追得上但角度落後 → ATC_ANG_xxx_P +10–20 %(INPUT_TC 只影響手感,不列)。
    if ang is not None and not osc_flag and not suspect and not sluggish and not motor_blocked:
        ind = []
        lag_lim = LAG_YAW_ANG_WARN_MS if yaw else LAG_ANG_WARN_MS
        if ang.lag_ms is not None and (ang.lag_rho or 0) >= LAG_MIN_RHO and ang.lag_ms > lag_lim:
            ind.append(f"期望角 → 實際角延遲約 {ang.lag_ms:.0f} ms")
        if not yaw and ang.bw_hz is not None and ang.bw_hz < BW_ANG_LOW_HZ:
            ind.append(f"角度環頻寬約 {ang.bw_hz:.2f} Hz")
        if ang.step_usable() and ang.rise_s is not None and ang.rise_s > RISE_ANG_SLOW_S:
            ind.append(f"角度上升時間約 {ang.rise_s * 1000:.0f} ms")
        ov_ok = not ang.step_usable() or ang.overshoot < OVERSHOOT_SLUGGISH_MAX
        sat_ok = sat_axis is None or sat_axis < SAT_SHARE_WARN
        if ind and ov_ok and sat_ok:
            low = _low_fs(ang.fs)
            sug = [gain_suggestion(nm["ANG"], prm, STEP_ANG_UP, "角度環偏慢:提高角度 P", 1)]
            det = ";".join(ind) + "。" + (f"({ang.fs:.0f} Hz 資料,僅供參考)" if low else "")
            if not yaw:
                det += f"({nm['TC']} 只影響搖桿手感,不會改變這裡量到的落後。)"
            out.append(Finding("angle_lag", SEV_INFO if low else SEV_WARN, ax, f"{lab} 角度環偏慢",
                               det, sug, ang.t_max_err, None,
                               {"lag_ms": ang.lag_ms, "bandwidth_hz": ang.bw_hz}))

    # ---- F. 持續偏差 / 積分飽和 ----
    # 懸停時平均誤差不為 0 → I 不足(Roll/Pitch I 慣例 = P、Yaw I ≈ P/10)或機構問題(重心、馬達歪斜、機架扭曲)。
    if ctx.pid is not None and ctx.pid.i_sat_share is not None and ctx.pid.i_sat_share >= I_SAT_WARN_SHARE:
        out.append(Finding(
            "i_saturated", SEV_WARN, ax, f"{lab} 積分項常頂到上限",
            f"{ctx.pid.i_sat_share * 100:.0f} % 的時間 |I| ≥ {I_SAT_FRAC:g} × IMAX({_fmt_param(ctx.pid.imax)}):"
            "需要很大的固定力矩才能維持姿態,通常是機構問題。",
            [_check("重心 / 馬達安裝角 / 機架扭曲" if not yaw else "馬達傾角 / 機架扭曲",
                    "先排除機構造成的固定力矩", 1),
             _check(nm["IMAX"], "機構沒問題才考慮放寬積分上限", 3)],
            None, None, {"i_sat_pct": ctx.pid.i_sat_share * 100}))
    else:
        b_rate = rate.bias if (rate is not None and rate.bias is not None
                               and abs(rate.bias) >= BIAS_RATE_WARN) else None
        lim = BIAS_YAW_ANG_WARN if yaw else BIAS_ANG_WARN
        b_ang = ang.bias if (ang is not None and ang.bias is not None and abs(ang.bias) >= lim) else None
        if b_rate is not None or b_ang is not None:
            ev = []
            if b_rate is not None:
                ev.append(f"準穩態平均角速度誤差 {b_rate:+.1f} °/s")
            if b_ang is not None:
                ev.append(f"準穩態平均角度誤差 {b_ang:+.1f}°")
            sug = [gain_suggestion(nm["I"], prm, STEP_I_UP, "持續偏差:提高 I 讓穩態誤差收斂", 1),
                   _check("馬達傾角 / 機架扭曲" if yaw else "重心 / 馬達安裝角 / 機架扭曲",
                          "固定偏差常來自機構,調 I 只是補償", 2)]
            det = ";".join(ev) + "。"
            ki = _num(prm.get(nm["I"]))
            conv = (kp / 10.0 if yaw else kp) if kp else None
            if ki is not None and conv and ki < 0.5 * conv:
                det += (f"{nm['I']} 目前 {_fmt_param(ki)},只有慣例值({'P / 10' if yaw else 'I = P'} ≈ {_fmt_param(conv)})"
                        f"的 {ki / conv * 100:.0f} %:每次 +25 % 要很多輪,建議確認參數來源或直接跑 AutoTune。")
            out.append(Finding("bias", SEV_WARN, ax, f"{lab} 懸停時有持續偏差", det,
                               sug, None, rate.span if (rate and b_rate is not None) else
                               (ang.span if ang else None),
                               {"bias": b_rate if b_rate is not None else b_ang}))

    # ---- F2. 懸停輸出偏一邊(配平)----
    if ctx.out_mean is not None and abs(ctx.out_mean) >= TRIM_OUT_INFO:
        m = ctx.out_mean
        if ax == "roll":
            why = "機體傾向往左滾(重心偏左或左側推力較弱)" if m > 0 else "機體傾向往右滾(重心偏右或右側推力較弱)"
        elif ax == "pitch":
            why = "機體傾向低頭(重心偏前或前方推力較弱)" if m > 0 else "機體傾向抬頭(重心偏後或後方推力較弱)"
        else:
            why = "機體傾向逆時針偏航(馬達歪斜 / 槳不一致)" if m > 0 else "機體傾向順時針偏航(馬達歪斜 / 槳不一致)"
        out.append(Finding(
            "trim_offset", SEV_WARN if abs(m) >= TRIM_OUT_WARN else SEV_INFO, ax,
            f"{lab} 懸停輸出長期偏一邊({m * 100:+.0f} %)",
            f"懸停時 {lab} 平均輸出 {m:+.3f}(−1…1):{why}。積分項在補償,但會吃掉控制裕度。",
            [_check("重心 / 馬達 / 螺旋槳" if not yaw else "馬達傾角 / 螺旋槳",
                    "調整電池 / 酬載位置,檢查各馬達與槳是否一致", 3)],
            None, None, {"out_mean": m}))

    # ---- G. 該軸輸出飽和 ----
    # 輸出常頂到極限:不能再加增益;Roll/Pitch 降 ATC_ACCEL_x_MAX、檢查推力;Yaw 檢查馬達傾角 / MOT_YAW_HEADROOM。
    # 整機的馬達飽和主要是「壓到下限」時不是推力不足:有雜訊證據 → 先處理雜訊(ACCEL 只限制姿態目標,救不了);
    # 沒有 → 多半是急降 / 低油門時的大動作,降 ACCEL 仍有幫助,但不說「推重比不足」。
    if sat_axis is not None and sat_axis >= SAT_SHARE_WARN:
        sev = SEV_BAD if sat_axis >= SAT_SHARE_BAD else SEV_WARN
        noise_ev = _axis_noise_evidence(ctx)
        detail = (f"{sat_axis * 100:.1f} % 的飛行時間 {lab} 輸出頂到極限:控制器要求的力矩給不出來,"
                  "此時不應再提高增益。")
        if ctx.veh.sat_low_dom and noise_ev and not yaw:
            detail += ("整機馬達飽和主要是壓到下限(SPIN_MIN)、不是推力不足;同時有雜訊證據("
                       + "、".join(noise_ev) + "):雜訊讓輸出劇烈跳動。先處理雜訊,降 ATC_ACCEL 救不了。")
            sug = [_check("陷波濾波 / INS_GYRO_FILTER / " + nm["FLTD"],
                          "雜訊推動的輸出飽和:先設諧波陷波、確認陀螺儀低通與 D 項濾波", 1)]
        elif yaw:
            sug = [_check("馬達傾角 / MOT_YAW_HEADROOM", "偏航力矩不足:馬達稍微傾斜可大幅增加偏航能力", 1),
                   _accel_suggestion(nm["ACC"], prm, "降低偏航角加速度上限,少要一點偏航力矩", 2)]
        elif ctx.veh.sat_low_dom:
            detail += "整機馬達飽和主要是壓到下限(SPIN_MIN),不是推力不足;常見於急降或低油門時的大動作。"
            sug = [_accel_suggestion(nm["ACC"], prm, "降低角加速度上限,低油門時少要一點力矩", 1)]
        else:
            sug = [_accel_suggestion(nm["ACC"], prm, "降低角加速度上限,避免要求超過推力能給的力矩", 1),
                   _check("推力 / 螺旋槳 / 重量", "推重比不足時任何增益都救不回來", 2)]
        out.append(Finding("saturation", sev, ax, f"{lab} 輸出飽和 {sat_axis * 100:.0f} %",
                           detail, [s for s in sug if s], None, None,
                           {"sat_pct": sat_axis * 100}))
    return out


# =====================================================================
# 整機檢查
# =====================================================================

def _vibe_findings(index, intervals: list[Interval]) -> tuple[list[Finding], str, dict]:
    info = index.messages().get("VIBE")
    if info is None:
        return [], "unknown", {}
    worst = None
    rows = []
    for inst in (info.instances or [None]):
        tbl = index.table("VIBE", inst)
        if tbl is None or not all(k in tbl for k in ("VibeX", "VibeY", "VibeZ")):
            continue
        m = _in_intervals(tbl["t"], intervals)
        if m.sum() < 5:
            continue
        v = np.column_stack([np.asarray(tbl[k], float)[m] for k in ("VibeX", "VibeY", "VibeZ")])
        mean = v.mean(axis=0)
        vmax = v.max(axis=1)
        k = int(np.argmax(vmax))
        clip = 0
        if "Clip" in tbl:
            c = np.asarray(tbl["Clip"], float)[m]
            clip = int(max(0.0, np.nanmax(c) - np.nanmin(c)))
        row = {"inst": 0 if inst is None else int(inst), "mean": float(mean.max()),
               "axis_mean": mean.tolist(), "p99": float(np.percentile(vmax, 99)),
               "t_max": float(tbl["t"][m][k]), "clip": clip}
        rows.append(row)
        if worst is None or (row["mean"], row["clip"]) > (worst["mean"], worst["clip"]):
            worst = row
    if not rows:
        return [], "unknown", {}
    mean_max = max(r["mean"] for r in rows)
    p99_max = max(r["p99"] for r in rows)
    clip_max = max(r["clip"] for r in rows)
    level = "ok"
    if mean_max >= VIBE_BAD or clip_max >= CLIP_BAD:
        level = "bad"
    elif mean_max >= VIBE_WARN or p99_max >= VIBE_BAD or clip_max >= CLIP_WARN:
        level = "warn"
    txt = ";".join(f"IMU{r['inst']} 平均 X/Y/Z = " + "/".join(f"{x:.1f}" for x in r["axis_mean"])
                   + f" m/s²、P99 {r['p99']:.0f}" + (f"、clipping +{r['clip']}" if r["clip"] else "")
                   for r in rows)
    metrics = {"vibe_mean": mean_max, "vibe_p99": p99_max, "clip": clip_max}
    t_focus = max(rows, key=lambda r: r["p99"])["t_max"]
    if level == "ok":
        return [Finding("vibration_ok", SEV_OK, None, "振動正常",
                        txt + f"(ArduPilot:< {VIBE_WARN:g} m/s² 一般可接受)。", [], None, None,
                        metrics)], level, metrics
    sev = SEV_BAD if level == "bad" else SEV_WARN
    title = "振動過大:先處理振動,再調 PID" if level == "bad" else "振動偏高"
    detail = (txt + f"。ArduPilot 建議 < {VIBE_WARN:g} m/s²,> {VIBE_BAD:g} m/s² 幾乎一定影響定高 / 定點;"
              "加速度計 clipping 代表量測已飽和。振動會讓 D 項放大雜訊、讓所有調參結果失真。")
    sug = [_check("減振座 / 飛控固定", "飛控與機架之間要有適當減振,線材不要拉扯飛控", 1),
           _check("螺旋槳平衡 / 馬達軸承 / 機架剛性", "找出振動源", 1)]
    return [Finding("vibration", sev, None, title, detail, sug, t_focus, None, metrics)], level, metrics


def _motor_channels(index, params: dict, frame_class: int | None) -> list[str]:
    tbl = index.table("RCOU")
    if tbl is None:
        return []
    chans = []
    for i in range(1, 33):
        fn = _num(params.get(f"SERVO{i}_FUNCTION"))
        if fn is not None and (33 <= fn <= 40 or 82 <= fn <= 85) and f"C{i}" in tbl:
            chans.append(f"C{i}")
    if not chans and frame_class in _MOTOR_COUNT:
        chans = [f"C{i}" for i in range(1, _MOTOR_COUNT[frame_class] + 1) if f"C{i}" in tbl]
    return chans


def _pwm_range(params: dict, mp: str) -> tuple[float, float, str] | None:
    """馬達輸出的 PWM 範圍 (最小, 最大, 來源說明);無法得知 → None。

    同 AP_MotorsMulticopter::get_pwm_output_min/max:MOT_PWM_MIN / MAX 未設定(≤ 0 或 MAX ≤ MIN)時
    改用油門輸入通道 RC{RCMAP_THROTTLE}_MIN / MAX(ArduCopter 4.0 / 4.1 的預設值就是 0 / 0;
    4.2 起預設 1000 / 2000、而且 0 會解鎖失敗)。QuadPlane 的 Q_M_PWM_* 也一樣。
    兩者都沒有記錄時不猜 1000 / 2000(猜錯會讓上下限永遠碰不到 → 誤判沒飽和、還建議提高增益)。
    """
    lo, hi = _num(params.get(mp + "PWM_MIN")), _num(params.get(mp + "PWM_MAX"))
    if lo is not None and hi is not None and lo > 0 and hi > 0 and hi > lo:
        return lo, hi, f"{mp}PWM_MIN / MAX"
    ch = int(_num(params.get("RCMAP_THROTTLE")) or 3)
    rlo, rhi = _num(params.get(f"RC{ch}_MIN")), _num(params.get(f"RC{ch}_MAX"))
    if rlo is not None and rhi is not None and rlo > 0 and rhi > rlo:
        return rlo, rhi, f"RC{ch}_MIN / MAX({mp}PWM_MIN / MAX 未設定,同 ArduPilot 改用油門通道範圍)"
    return None


def _motor_stats(index, params: dict, intervals: list[Interval], frame_class: int | None,
                 prefix: str) -> dict:
    """RCOU 馬達 PWM 頂到上限 / 壓到下限的比例(飛行中)與 MOTB 推力失效 / 限流。

    回傳 {"share", "hi", "lo"(0–1 或 None = 無法判斷), "metrics", "t_focus", "findings", "thrust_loss", "note"}。
    """
    st: dict = {"share": None, "hi": None, "lo": None, "metrics": {}, "t_focus": None, "findings": [],
                "thrust_loss": False, "note": None}
    chans = _motor_channels(index, params, frame_class)
    tbl = index.table("RCOU")
    mp = "Q_M_" if prefix == "Q_A_" else "MOT_"
    if chans and tbl is not None:
        m = _in_intervals(tbl["t"], intervals)
        rg = _pwm_range(params, mp)
        if m.sum() >= 10 and rg is None:
            st["note"] = (f"沒有 {mp}PWM_MIN / MAX 也沒有油門通道的 RC_MIN / MAX:無法判斷馬達是否頂到上下限"
                          "(改用 PID 記錄的 limit 旗標;沒有就無法判斷)。")
        elif m.sum() >= 10:
            M = np.column_stack([np.asarray(tbl[c], float)[m] for c in chans])
            lo_pwm, hi_pwm, src = rg
            rng = hi_pwm - lo_pwm
            smin = _num(params.get(mp + "SPIN_MIN"))
            smax = _num(params.get(mp + "SPIN_MAX"))
            smin = 0.15 if smin is None else smin
            smax = 0.95 if smax is None else smax
            hi_lvl = lo_pwm + smax * rng - MOTOR_SAT_MARGIN * rng
            lo_lvl = lo_pwm + smin * rng + MOTOR_SAT_MARGIN * rng
            hi = (M >= hi_lvl).any(axis=1)
            lo = (M <= lo_lvl).any(axis=1)
            sat = hi | lo
            share = float(np.mean(sat))
            st.update(share=share, hi=float(hi.mean()), lo=float(lo.mean()))
            st["metrics"] = {"motor_sat_pct": share * 100, "motor_hi_pct": float(hi.mean() * 100),
                             "motor_lo_pct": float(lo.mean() * 100),
                             "motor_mean": [float(x) for x in M.mean(axis=0)], "pwm_range": (lo_pwm, hi_pwm),
                             "pwm_range_src": src}
            if share >= SAT_SHARE_WARN:
                tt = tbl["t"][m]
                bins = np.floor((tt - tt[0]) / 2.0).astype(int)
                k = int(np.argmax(np.bincount(bins, weights=sat.astype(float))))
                st["t_focus"] = float(tt[0] + 2.0 * k + 1.0)
    mb = index.table("MOTB")
    out: list[Finding] = st["findings"]
    if mb is not None:
        m = _in_intervals(mb["t"], intervals)
        if m.sum() >= 5:
            if "FailFlags" in mb:
                ff = np.asarray(mb["FailFlags"], float)[m].astype(np.int64)
                if (ff & 1).any():
                    st["thrust_loss"] = True
                    k = int(np.argmax((ff & 1) > 0))
                    out.append(Finding("thrust_loss", SEV_BAD, None, "偵測到馬達失效 / 推力不足",
                                       "MOTB.FailFlags 顯示有馬達被判定失效(推力補償啟動):先檢查馬達、電變與螺旋槳。",
                                       [_check("馬達 / 電變 / 螺旋槳", "推力不足時調 PID 沒有意義", 1)],
                                       float(mb["t"][m][k]), None, {}))
            if "ThLimit" in mb:
                tl = np.asarray(mb["ThLimit"], float)[m]
                lim = float(np.mean(tl < 0.99))
                st["metrics"]["thr_limit_pct"] = lim * 100
                if lim >= SAT_SHARE_WARN:
                    out.append(Finding("thrust_limit", SEV_INFO, None, f"油門被限流 {lim * 100:.0f} %",
                                       "MOTB.ThLimit < 1:電池電流限制在壓油門,控制裕度變小。",
                                       [_check(mp + "BAT_CURR_MAX / 電池", "確認電流限制與電池能力", 3)],
                                       None, None, {"thr_limit_pct": lim * 100}))
    return st


def _motor_sat_findings(st: dict, params: dict, prefix: str, hover: float | None,
                        noise_ev: list[str]) -> list[Finding]:
    """馬達飽和的整機診斷;上限與下限分開判斷。

    - 頂到上限(或懸停油門 > HOVER_HIGH)→ 推力不足:檢查推重比 + 降 ATC_ACCEL_*_MAX(motor_saturation);
    - 主要是壓到下限 → 不是推力不足(motor_sat_low):
      有雜訊證據(D 項高頻功率、D/P 過大、陀螺儀噪音峰、批次 FFT 噪音峰)→ 雜訊推動輸出,先處理陷波 /
      INS_GYRO_FILTER / FLTD(ATC_ACCEL 只限制姿態目標,救不了);沒有 → 多半是急降或大幅偏航,只提醒。
    """
    share, hi, lo = st.get("share"), st.get("hi"), st.get("lo")
    if share is None or share < SAT_SHARE_WARN:
        return []
    metrics, t_focus = st["metrics"], st["t_focus"]
    win = (t_focus - 1.0, t_focus + 1.0) if t_focus is not None else None
    parts = []
    if hi:
        parts.append(f"頂到上限(SPIN_MAX){hi * 100:.1f} %")
    if lo:
        parts.append(f"壓到下限(SPIN_MIN){lo * 100:.1f} %")
    sev = SEV_BAD if share >= SAT_SHARE_BAD else SEV_WARN
    high_hover = hover is not None and hover > HOVER_HIGH
    if (hi or 0.0) >= SAT_SHARE_WARN or high_hover:
        sug = [_check(THRUST_CHECK, "馬達頂到上限代表推力不夠(減重 / 換動力組 / 確認電池)", 1)]
        for a in ("roll", "pitch"):
            g = _accel_suggestion(gain_names(prefix, a)["ACC"], params, "降低角加速度上限,少要一點力矩", 2)
            if g.proposed is not None:
                sug.append(g)
        return [Finding("motor_saturation", sev, None, f"馬達輸出飽和 {share * 100:.0f} %",
                        f"飛行中 {share * 100:.1f} % 的時間有馬達" + "、".join(parts)
                        + ":此時姿態控制沒有裕度,不建議提高任何增益。", sug, t_focus, win, metrics)]
    hov_txt = f"、懸停油門 {hover:.2f}" if hover is not None else ""
    base = (f"飛行中 {share * 100:.1f} % 的時間有馬達" + "、".join(parts)
            + f"(頂到上限只有 {(hi or 0.0) * 100:.1f} %{hov_txt}):不是推力不足。")
    if noise_ev:
        gf = _num(params.get("INS_GYRO_FILTER"))
        detail = (base + "同時有雜訊證據(" + "、".join(noise_ev) + "):陀螺儀 / D 項雜訊讓馬達輸出劇烈跳動、"
                  "碰到下限。先處理雜訊(諧波陷波、陀螺儀低通、D 項濾波),不要調 ATC_ACCEL_*_MAX"
                  "(它只限制姿態目標,救不了雜訊造成的輸出飽和);此時不建議提高任何增益。")
        sug = [_check("陷波濾波 INS_HNTCH_* / INS_GYRO_FILTER / " + prefix + "RAT_*_FLTD",
                      f"雜訊推動的輸出飽和:先設諧波陷波;INS_GYRO_FILTER 目前 {_fmt_param(gf)} Hz,"
                      "偏高時馬達噪音直接進到控制迴路", 1)]
        return [Finding("motor_sat_low", sev, None, f"馬達常壓到下限 {(lo or 0.0) * 100:.0f} %(雜訊推動輸出)",
                        detail, sug, t_focus, win, metrics)]
    detail = (base + "壓到下限常見於急降、低油門時的大動作或大幅偏航;懸停時經常碰到下限才需要注意"
              "(推力過剩 / 雜訊)。此時不建議提高增益。")
    return [Finding("motor_sat_low", SEV_INFO, None, f"馬達偶爾壓到下限 {(lo or 0.0) * 100:.0f} %",
                    detail, [], t_focus, win, metrics)]


def _hover_findings(index, params: dict, intervals: list[Interval],
                    quadplane: bool = False) -> tuple[list[Finding], dict]:
    """懸停油門(Copter:CTUN.ThO;QuadPlane:QTUN.ThO 升力油門)與 MOT_ / Q_M_THST_HOVER。"""
    tbl, name = _thr_table(index, quadplane)
    if tbl is None:
        return [], {}
    m = _in_intervals(tbl["t"], intervals)
    if "CRt" in tbl:
        crt = np.asarray(tbl["CRt"], float)
        u = index.unit(f"{name}.CRt")
        # Copter CTUN.CRt 是 cm/s;QTUN.CRt 以 'c' 格式記錄(讀出來已是 m/s)
        is_ms = u in ("m/s",) or (quadplane and u not in ("cm/s",))
        crt_cms = crt * 100.0 if is_ms else crt
        m &= np.abs(crt_cms) < HOVER_MAX_CLIMB_CMS
    if m.sum() < 20:
        return [], {}
    tho = np.asarray(tbl["ThO"], float)[m]
    hov = float(np.median(tho))
    mp = "Q_M_" if quadplane else "MOT_"
    cur = _num(params.get(mp + "THST_HOVER"))
    learn = _num(params.get(mp + "HOVER_LEARN"))
    met = {"hover_thr": hov, "thr_p99": float(np.percentile(tho, 99)), "hover_src": f"{name}.ThO"}
    out = []
    if hov > HOVER_HIGH:
        out.append(Finding("hover_high", SEV_WARN, None, f"懸停油門偏高({hov:.2f})",
                           f"懸停油門約 {hov:.2f}(> {HOVER_HIGH:g}):推力裕度小,姿態控制容易飽和,"
                           "PID 再怎麼調也會受限。", [_check(THRUST_CHECK, "減重或換更大推力的動力組", 1)],
                           None, None, met))
    elif hov < HOVER_LOW:
        out.append(Finding("hover_low", SEV_INFO, None, f"懸停油門偏低({hov:.2f})",
                           f"懸停油門約 {hov:.2f}(< {HOVER_LOW:g}):推力過剩,低油門解析度差,"
                           "增益在這種機上會顯得偏高。", [], None, None, met))
    # 實測值 < 0.05 多半是資料來源不對(例:讀到推進油門),不據此改 THST_HOVER
    if cur is not None and hov >= 0.05 and abs(hov - cur) > HOVER_MISMATCH and (learn is None or learn == 0):
        rg = param_range(mp + "THST_HOVER") or PARAM_RANGES["MOT_THST_HOVER"]
        prop = _round_sig(min(max(hov, rg[0]), rg[1]), 3)
        out.append(Finding("hover_param", SEV_INFO, None, f"{mp}THST_HOVER 與實際懸停油門不符",
                           f"實際懸停油門約 {hov:.3f}({name}.ThO),參數為 {cur:.3f},且未開啟懸停學習。",
                           [Suggestion(mp + "THST_HOVER", DIR_SET, "設成實測懸停油門(或設 HOVER_LEARN = 2 自動學習)",
                                       cur, prop, None, 2)], None, None, met))
    return out, met


def batch_gyro_spectrum(index, intervals: list[Interval]) -> dict | None:
    """ISBH / ISBD 批次陀螺儀取樣 → 平均功率譜((deg/s)²/Hz,三軸相加)。沒有資料回傳 None。

    每筆 ISBD 歸給「批次編號 N 相同、時間在它之前的最近一筆 ISBH」;樣本 = 原始 int16 / mul
    (陀螺儀 rad/s),只取第一顆有資料的 IMU、飛行區間內的批次。
    """
    hdr = index.raw_table("ISBH") if index is not None else None
    dat = index.raw_table("ISBD") if index is not None else None
    if hdr is None or dat is None or not hdr["t"].size or not dat["t"].size:
        return None
    if not all(k in hdr for k in ("N", "type", "mul", "smp_cnt")) or             not all(k in dat for k in ("N", "seqno", "x", "y", "z")):
        return None
    ht = np.asarray(hdr["t"], float)
    hn = np.asarray(hdr["N"]).astype(np.int64)
    htype = np.asarray(hdr["type"]).astype(np.int64)
    hinst = np.asarray(hdr["instance"]).astype(np.int64) if "instance" in hdr else np.zeros(ht.size, np.int64)
    gyro = np.flatnonzero(htype == 1)
    if not gyro.size:
        return None
    inst0 = int(hinst[gyro].min())
    dt_ = np.asarray(dat["t"], float)
    dn = np.asarray(dat["N"]).astype(np.int64)
    dseq = np.asarray(dat["seqno"]).astype(np.int64)
    rate_col = hdr.get("smp_rate")
    groups: dict[tuple[int, float], list[np.ndarray]] = {}
    by_n: dict[int, np.ndarray] = {}
    order = np.argsort(dn, kind="stable")
    un, start = np.unique(dn[order], return_index=True)
    for i, nv in enumerate(un.tolist()):
        end = start[i + 1] if i + 1 < un.size else order.size
        rows = order[start[i]:end]
        by_n[nv] = rows[np.argsort(dt_[rows], kind="stable")]
    hsorted = np.argsort(ht, kind="stable")
    next_same = np.full(ht.size, math.inf)      # 下一筆同 N 的標頭時間(N 會繞回)
    last: dict[int, float] = {}
    for h in hsorted[::-1].tolist():
        next_same[h] = last.get(int(hn[h]), math.inf)
        last[int(hn[h])] = float(ht[h])
    for h in hsorted.tolist():
        if htype[h] != 1 or hinst[h] != inst0:
            continue
        if not _in_intervals([ht[h]], intervals)[0]:
            continue
        rows = by_n.get(int(hn[h]))
        if rows is None:
            continue
        rows = rows[(dt_[rows] >= ht[h] - 0.05) & (dt_[rows] < next_same[h])]
        if not rows.size:
            continue
        rows = rows[np.argsort(dseq[rows], kind="stable")]
        mul = float(hdr["mul"][h]) or 1.0
        n = int(hdr["smp_cnt"][h])
        xyz = np.stack([np.asarray(dat[k])[rows].reshape(-1)[:n] for k in ("x", "y", "z")]).astype(float)
        if xyz.shape[1] < 64:
            continue
        fs = float(rate_col[h]) if rate_col is not None else 1000.0
        groups.setdefault((xyz.shape[1], round(fs, 1)), []).append(np.degrees(xyz / mul))
    if not groups:
        return None
    (n, fs), batches = max(groups.items(), key=lambda kv: len(kv[1]))
    w = hann(n)
    scale = 2.0 / (fs * float(np.sum(w ** 2)))
    acc = 0.0
    for b in batches:
        acc = acc + (np.abs(np.fft.rfft((b - b.mean(axis=1, keepdims=True)) * w, axis=1)) ** 2).sum(axis=0)
    psd = acc / len(batches) * scale
    f = np.fft.rfftfreq(n, 1.0 / fs)
    pk = find_peaks(f, psd, fmin=NOTCH_FMIN_HZ, fmax=PEAK_FMAX_FRAC * fs / 2.0,
                    prom_ratio=NOTCH_PROM_RATIO, base_hz=max(10.0, fs / 50.0))
    top = max(pk, key=lambda p: p.prominence * p.rms) if pk else None
    return {"f": f, "psd": psd, "fs": fs, "n_batches": len(batches),
            "peak_hz": top.f if top else None, "peak_rms": top.rms if top else None,
            "peaks": [(p.f, p.rms) for p in pk]}


def _notch_findings(index, params: dict, intervals: list[Interval], hover: float | None,
                    data: dict) -> list[Finding]:
    out = []
    en = _num(params.get("INS_HNTCH_ENABLE"))
    bs = None
    try:
        bs = batch_gyro_spectrum(index, intervals)
    except Exception:  # noqa: BLE001 — 批次資料格式異常就當作沒有
        bs = None
    data["batch"] = bs is not None
    if bs is not None:
        data["batch_hz"] = bs["fs"]
        data["batch_peak_hz"] = bs["peak_hz"]
    bat_sug = _bat_mask_suggestion(params, 2)
    if bs is None:
        if en is not None and en == 0:
            out.append(Finding("notch_off", SEV_INFO, None, "諧波陷波濾波器未啟用,也沒有噪音頻譜可設定",
                               "INS_HNTCH_ENABLE = 0。陷波濾波可大幅降低馬達噪音進入 D 項,是 ArduPilot 調參流程的"
                               "第二步;但這份 log 沒有批次 IMU 取樣(ISBH / ISBD),量不到噪音頻率,先別盲設。",
                               [bat_sug, _check("INS_HNTCH_ENABLE", "量到噪音峰之後再依峰值設定陷波", 3)]))
        else:
            out.append(Finding("no_batch", SEV_INFO, None, "沒有批次 IMU 取樣(FFT)",
                               "無法檢查馬達噪音頻率與陷波濾波效果。", [bat_sug]))
        return out
    pk = bs["peak_hz"]
    if pk is None:
        out.append(Finding("noise_ok", SEV_OK, None, "陀螺儀噪音沒有明顯尖峰",
                           f"批次取樣 {bs['fs']:.0f} Hz、{bs['n_batches']} 批:≥ {NOTCH_FMIN_HZ:g} Hz 沒有明顯的馬達噪音峰。",
                           [], None, None, {}))
        return out
    met = {"noise_hz": pk, "noise_rms": bs["peak_rms"]}
    if en is None or en == 0:
        freq = _round_sig(min(max(pk, 10.0), 495.0), 3)
        bw = _round_sig(min(max(pk / 2.0, 5.0), 250.0), 3)
        sug = [Suggestion("INS_HNTCH_ENABLE", DIR_SET, "啟用諧波陷波濾波(設好後重開機才看得到其他參數)",
                          en, 1.0, None, 1),
               Suggestion("INS_HNTCH_MODE", DIR_SET, "1 = 依油門追蹤(有 ESC 轉速回報時用 3 更準)",
                          _num(params.get("INS_HNTCH_MODE")), 1.0, None, 2),
               Suggestion("INS_HNTCH_REF", DIR_SET, "油門追蹤的參考點 = 懸停油門",
                          _num(params.get("INS_HNTCH_REF")), _round_sig(hover, 3) if hover else None, None, 2),
               Suggestion("INS_HNTCH_FREQ", DIR_SET, "懸停時的噪音中心頻率", _num(params.get("INS_HNTCH_FREQ")),
                          freq, None, 2),
               Suggestion("INS_HNTCH_BW", DIR_SET, "頻寬約為中心頻率的一半", _num(params.get("INS_HNTCH_BW")),
                          bw, None, 2)]
        out.append(Finding("notch_setup", SEV_WARN, None, f"陀螺儀在 {pk:.0f} Hz 有馬達噪音,建議啟用陷波",
                           f"批次取樣 {bs['fs']:.0f} Hz、{bs['n_batches']} 批:{pk:.0f} Hz 有明顯噪音峰"
                           f"(RMS {bs['peak_rms']:.1f} °/s)。依 ArduPilot 文件設定油門追蹤的諧波陷波後,"
                           "再重新量一次確認峰值被壓下來。", sug, None, None, met))
    else:
        f0 = _num(params.get("INS_HNTCH_FREQ"))
        if f0 and abs(f0 - pk) / pk > 0.3:
            out.append(Finding("notch_mismatch", SEV_INFO, None, "陷波中心頻率可能不符",
                               f"陷波已啟用(INS_HNTCH_FREQ = {f0:g} Hz),但噪音峰在 {pk:.0f} Hz"
                               "(若批次資料是濾波後,代表殘留噪音)。", [_check("INS_HNTCH_FREQ / MODE / REF", "確認陷波追蹤到實際的噪音頻率", 2)],
                               None, None, met))
        else:
            out.append(Finding("notch_ok", SEV_OK, None, "陷波濾波已啟用",
                               f"噪音峰約 {pk:.0f} Hz,INS_HNTCH_FREQ = {_fmt_param(f0)}。", [], None, None, met))
    return out


def _pilot_share(logdata, intervals: list[Interval]) -> float | None:
    """飛行區間中「搖桿直接給姿態」模式所佔比例;沒有模式資訊回傳 None。"""
    from .timeline import mode_changes
    modes = mode_changes(logdata.events)
    tot = _total(intervals)
    if not modes or tot <= 0:
        return None
    iv = []
    for i, m in enumerate(modes):
        end = modes[i + 1].t if i + 1 < len(modes) else math.inf
        if (m.label or "").upper() in PILOT_MODES:
            iv.append((m.t, end))
    return _total(_intersect(intervals, iv)) / tot if iv else 0.0


def _bitmask_suggestion(params: dict, why: str, quadplane: bool = False) -> Suggestion | None:
    """提高姿態 / PID 記錄頻率的 LOG_BITMASK。

    Copter:加 ATTITUDE_FAST(bit 0)與 PID(bit 12)。Plane(QuadPlane):bit 0 只有 25 Hz、bit 12 是 CAMERA、
    沒有 PID bit → 加 ATTITUDE_FULLRATE(bit 20),VTOL 時 ATT / PIQR / PIQP / PIQY 隨姿態以迴圈頻率記錄。
    """
    if quadplane:
        bits, txt = LOG_BIT_PLANE_FULLRATE, "加上 ATTITUDE_FULLRATE bit 20(VTOL 時 ATT / PIQR / PIQP / PIQY 以迴圈頻率記錄)"
    else:
        bits, txt = LOG_BIT_ATT_FAST | LOG_BIT_PID, "加上 ATTITUDE_FAST bit 0 與 PID bit 12"
    cur = _num(params.get("LOG_BITMASK"))
    if cur is None:
        return Suggestion("LOG_BITMASK", DIR_SET, why + f"({txt})", None, None, None, 1)
    new = int(cur) | bits
    if new == int(cur):
        return None
    return Suggestion("LOG_BITMASK", DIR_SET, why + f"({txt};記錄檔會變大,調參飛行時再開)",
                      cur, float(new), None, 1)


def _fast_attitude(bm: float | None, quadplane: bool, att_hz: float | None) -> bool:
    if bm is None:
        return bool((att_hz or 0) > 50)
    return bool(int(bm) & (LOG_BIT_PLANE_FULLRATE if quadplane else LOG_BIT_ATT_FAST))


def _default_gain_finding(params: dict, prefix: str) -> Finding | None:
    if prefix != "ATC_":
        return None
    vals = [[_num(params.get(f"ATC_RAT_{a}_{k}")) for k in ("P", "I", "D")] for a in ("RLL", "PIT")]
    if any(v is None for row in vals for v in row):
        return None
    if all(all(abs(v - d) <= 1e-4 for v, d in zip(row, _DEFAULT_RP)) for row in vals):
        yp, yi = _num(params.get("ATC_RAT_YAW_P")), _num(params.get("ATC_RAT_YAW_I"))
        yaw_def = yp is not None and yi is not None and abs(yp - _DEFAULT_YAW[0]) < 1e-4 \
            and abs(yi - _DEFAULT_YAW[1]) < 1e-4
        # 角速度 PID 輸出是正規化的:迴路增益 ≈ 控制權(推力 / 慣量)× Kp。動力強、小型(推重比高)的機
        # 用預設值常會震盪;大型 / 反應慢的機常偏軟。
        return Finding("default_gains", SEV_INFO, None, "仍是出廠預設 PID",
                       "Roll / Pitch 角速度 PID 仍是預設 0.135 / 0.135 / 0.0036"
                       + ("、Yaw 0.18 / 0.018" if yaw_def else "")
                       + ":預設值只是起點 —— 動力強 / 小型機(推重比高)用預設值常會震盪,大型 / 反應慢的機常偏軟。"
                       "建議依 ArduPilot 調參流程:初始參數(MP「Initial Tune Parameters」)→ 量振動 / 設陷波 → AutoTune。",
                       [_check("AutoTune", "處理好振動與陷波後,在無風空曠處跑 AutoTune(一次一軸較安全)", 3)])
    return None


# =====================================================================
# 安全後處理
# =====================================================================

def apply_safety(findings: list[Finding], veh: VehicleContext,
                 axis_osc: dict[str, bool] | None = None) -> None:
    """就地移除不安全的「讓控制更積極」的建議(提高增益、減小 INPUT_TC、放寬 ACCEL 上限),並註明原因。

    振動過大 / 馬達飽和 / 推力不足 → 一律不給;振動偏高 → 不提高 D;該軸有震盪跡象 → 不提高該軸增益。
    是否「提高」以建議值 vs 目前值判斷(不只看方向標籤)。振動過大時其餘增益建議順位延後(先處理振動)。
    建議全被拿掉的 WARN / BAD 降為 INFO(只剩提醒)。
    """
    axis_osc = axis_osc or {}
    glob = veh.block_increase
    for f in findings:
        had = bool(f.suggestions)
        keep = []
        for s in f.suggestions:
            if _raises_aggressiveness(s):
                why = glob
                if why is None and veh.vibe == "warn" and s.param.endswith("_D"):
                    why = "振動偏高"
                if why is None and f.axis is not None and axis_osc.get(f.axis):
                    why = "該軸有震盪跡象"
                if why is not None:
                    verb = "減小" if s.param.endswith("INPUT_TC") else "提高"
                    f.detail += f"(因為{why},暫不建議{verb} {s.param})"
                    continue
            if veh.vibe == "bad" and _is_gain(s.param):
                s.priority += 1
            keep.append(s)
        f.suggestions = keep
        if had and not keep and f.severity in (SEV_WARN, SEV_BAD):
            f.severity = SEV_INFO          # 建議全被安全規則拿掉:只剩提醒


# =====================================================================
# 機型 / 資料偵測
# =====================================================================

def _frame_text(index, params: dict, quadplane: bool) -> tuple[str, int | None]:
    pfx = "Q_" if quadplane else ""
    fc = _num(params.get(pfx + "FRAME_CLASS"))
    ft = _num(params.get(pfx + "FRAME_TYPE"))
    fci = int(fc) if fc is not None else None
    txt = ""
    try:
        for m in index.log_messages():
            mm = re.match(r"^Frame:\s*([A-Za-z0-9_]+)(?:/([A-Za-z0-9_\-]+))?", m.text or "")
            if mm:
                txt = mm.group(1).lower() + (f" {mm.group(2)}" if mm.group(2) else "")
    except Exception:  # noqa: BLE001
        txt = ""
    if not txt and fci is not None:
        txt = FRAME_CLASS_NAMES.get(fci, f"class {fci}")
        if ft is not None:
            txt += " " + FRAME_TYPE_NAMES.get(int(ft), f"type {int(ft)}")
    return txt, fci


def _vehicle(index, params: dict) -> dict:
    v = getattr(index, "vehicle", "unknown") or "unknown"
    qp = False
    fc = _num(params.get("FRAME_CLASS"))
    if v == "copter" and fc is not None and int(fc) in _HELI_CLASSES:
        v = "heli"
    if v == "plane" and (_num(params.get("Q_ENABLE")) or 0) >= 1:
        qp = True
    if v == "unknown":
        if "ATC_RAT_RLL_P" in params:
            v = "copter"
        elif "Q_A_RAT_RLL_P" in params:
            v, qp = "plane", True
        elif index.has("ATT"):
            v = "copter"
    supported = v == "copter" or qp
    reason = ""
    if not supported:
        reason = {"heli": "傳統直升機的調參方式(FF 為主、旋翼頭動態)不同,這個工具只支援多旋翼。",
                  "plane": "固定翼的姿態控制(PTCH2SRV_ / RLL2SRV_)與多旋翼不同,這個工具只支援多旋翼與"
                           "QuadPlane 的 VTOL 部分(Q_ENABLE = 1)。"}.get(
            v, f"機型「{v}」不支援:這個工具只分析多旋翼(ArduCopter)與 QuadPlane 的 VTOL 姿態控制。")
    frame, fci = _frame_text(index, params, qp)
    return {"vehicle": v, "quadplane": qp, "supported": supported, "reason": reason,
            "prefix": "Q_A_" if qp else "ATC_", "frame": frame, "frame_class": fci,
            "pid_msg": _PIQ_MSG if qp else _PID_MSG}


def _pid_units_scale(index, msg: str, rate_tbl, axis: str, intervals: list[Interval]) -> float:
    """PIDx Tar / Act 的單位 → deg/s 的倍率(單位欄;沒有就和 RATE 比大小;再不行看量級)。"""
    u = (index.unit(f"{msg}.Act") or index.unit(f"{msg}.Tar") or "").lower()
    if u in ("rad/s", "rads"):
        return 180.0 / math.pi
    if u in ("deg/s", "degs"):
        return 1.0
    tbl = index.table(msg)
    act = np.asarray(tbl["Act"], float)
    m = _in_intervals(tbl["t"], intervals)
    a = act[m] if m.sum() > 20 else act
    if rate_tbl is not None and _RATE_FIELDS[axis][1] in rate_tbl:
        rr = np.asarray(rate_tbl[_RATE_FIELDS[axis][1]], float)
        mr = _in_intervals(rate_tbl["t"], intervals)
        r = rr[mr] if mr.sum() > 20 else rr
        sa, sr = float(np.nanstd(a)), float(np.nanstd(r))
        if sa > 0 and sr > 0:
            ratio = sr / sa
            return 180.0 / math.pi if abs(math.log(ratio / (180.0 / math.pi))) < abs(math.log(ratio)) else 1.0
    return 180.0 / math.pi if float(np.nanpercentile(np.abs(a), 95)) < 1.5 else 1.0


def _decimate(t, d, a, n_max: int = PLOT_MAX_POINTS):
    if t.size <= n_max:
        return t, d, a
    k = int(math.ceil(t.size / n_max))
    return t[::k], d[::k], a[::k]


def _plot_arrays(t, d, a, t0, t1, intervals):
    m = (t >= t0) & (t <= t1)
    tt, dd, aa = t[m].astype(float), np.asarray(d, float)[m].copy(), np.asarray(a, float)[m].copy()
    off = ~_in_intervals(tt, intervals)
    dd[off] = np.nan
    aa[off] = np.nan
    return _decimate(tt, dd, aa)


# =====================================================================
# 主程式
# =====================================================================

def _empty_report(t0, t1, vehicle, frame, prefix, findings, notes, data=None, params=None):
    return TuningReport(t0=float(t0), t1=float(t1), flight_s=0.0, vehicle=vehicle, frame=frame,
                        param_prefix=prefix, data=data or {}, axes={}, findings=findings,
                        params=params or {}, notes=notes)


def _relevant_params(params: dict, prefix: str) -> dict[str, float]:
    keys = ("LOG_BITMASK", "INS_GYRO_FILTER", "INS_LOG_BAT_MASK", "INS_LOG_BAT_OPT", "FRAME_CLASS",
            "FRAME_TYPE", "INS_HNTCH_ENABLE", "INS_HNTCH_MODE", "INS_HNTCH_FREQ", "INS_HNTCH_BW",
            "INS_HNTCH_REF", "INS_HNTCH_ATT", "MOT_THST_HOVER", "MOT_HOVER_LEARN", "MOT_PWM_MIN",
            "MOT_PWM_MAX", "MOT_SPIN_MIN", "MOT_SPIN_MAX", "MOT_YAW_HEADROOM", "Q_ENABLE",
            "Q_M_THST_HOVER", "Q_M_HOVER_LEARN", "Q_M_PWM_MIN", "Q_M_PWM_MAX", "Q_FRAME_CLASS",
            "Q_FRAME_TYPE", "SCHED_LOOP_RATE", "LOG_FILE_RATEMAX", "RCMAP_THROTTLE", "RC3_MIN", "RC3_MAX")
    out = {}
    for k, v in params.items():
        if k.startswith(prefix + "RAT_") or k.startswith(prefix + "ANG_") or k.startswith(prefix + "ACCEL_") \
                or k == prefix + "INPUT_TC" or k in keys:
            x = _num(v)
            if x is not None:
                out[k] = x
    return out


# 同嚴重度時的重要性(小 = 先列):震盪 > 阻尼 > 偏軟 > 飽和 > 偏差 > 配平 …
_CODE_RANK = {"thrust_loss": 0, "vibration": 1, "motor_saturation": 2, "motor_sat_low": 2,
              "rate_osc_high": 3, "rate_osc_mid": 4, "slew_limit": 5, "gyro_noise": 5, "rate_osc_suspect": 6,
              "angle_wobble": 7, "angle_overshoot": 8, "angle_freq_peak": 8, "d_dominant": 8,
              "rate_underdamped": 9, "saturation": 10, "rate_sluggish": 11,
              "angle_lag": 12, "i_saturated": 13, "bias": 14, "d_noise": 15, "trim_offset": 16,
              "low_rate": 20, "no_pid_log": 21, "no_rate_data": 22, "low_coverage": 22, "notch_setup": 23,
              "notch_off": 24, "no_batch": 25, "default_gains": 26, "hover_high": 27,
              "hover_low": 28, "hover_param": 29, "not_judged": 39, "ok": 40}


def _sev_key(f: Finding):
    pr = min((s.priority for s in f.suggestions), default=9)
    ai = AXES.index(f.axis) if f.axis in AXES else -1
    return (-SEV_ORDER.get(f.severity, 0), pr, _CODE_RANK.get(f.code, 30), ai, f.code)


def analyze(logdata, t0: float | None = None, t1: float | None = None) -> TuningReport:
    """分析 [t0, t1](log 絕對秒;None = 整份 log)內的飛行,產生 PID 調參建議報告。"""
    index = getattr(logdata, "index", None)
    notes = list(SAFETY_NOTES)
    if index is None:
        lo = float(t0) if t0 is not None else 0.0
        hi = float(t1) if t1 is not None else lo
        return _empty_report(lo, hi, "unknown", "", "ATC_",
                             [Finding("no_index", SEV_INFO, None, "沒有 log 欄位資料",
                                      "這份資料沒有 LogIndex(只有位置來源),無法做 PID 分析。")], notes)
    lo, hi = index.t_range()
    t0 = lo if t0 is None else float(t0)
    t1 = hi if t1 is None else float(t1)
    if t1 < t0:
        t0, t1 = t1, t0
    params = index.param_values()
    vi = _vehicle(index, params)
    prefix = vi["prefix"]
    rel = _relevant_params(params, prefix)
    if not vi["supported"]:
        return _empty_report(t0, t1, vi["vehicle"], vi["frame"], prefix,
                             [Finding("unsupported", SEV_INFO, None, "這個機型不支援 PID 調參建議",
                                      vi["reason"])], notes, params=rel)

    # ---- 資料可用性與取樣率 ----
    names = set(index.messages())
    data: dict = {}
    att = index.table("ATT") if "ATT" in names else None
    rate_tbl = index.table("RATE") if "RATE" in names else None
    data["ATT_hz"] = round(_fs_of(att["t"]), 1) if att is not None else None
    data["RATE_hz"] = round(_fs_of(rate_tbl["t"]), 1) if rate_tbl is not None else None
    pid_msgs = vi["pid_msg"]
    pid_tbls = {ax: index.table(pid_msgs[ax]) for ax in AXES if pid_msgs[ax] in names}
    pid_hz = [_fs_of(tb["t"]) for tb in pid_tbls.values() if tb is not None]
    data["PID"] = bool(pid_tbls)
    data["PID_hz"] = round(max(pid_hz), 1) if pid_hz else None
    imu = index.table("IMU") if "IMU" in names else None
    data["IMU_hz"] = round(_fs_of(imu["t"]), 1) if imu is not None else None
    data["CTRL"] = "CTRL" in names
    for nm_ in ("CTRL", "RCOU"):
        tb = index.table(nm_) if nm_ in names else None
        data[f"{nm_}_hz"] = round(_fs_of(tb["t"]), 1) if tb is not None else None
    bm = _num(params.get("LOG_BITMASK"))
    data["LOG_BITMASK"] = int(bm) if bm is not None else None
    data["fast_attitude"] = _fast_attitude(bm, vi["quadplane"], data["ATT_hz"])
    data["loop_hz"] = _loop_rate(params)

    # ---- 飛行遮罩 ----
    hover = _num(params.get(("Q_M_" if vi["quadplane"] else "MOT_") + "THST_HOVER"))
    intervals, mnotes = flight_intervals(logdata, t0, t1, vi["quadplane"], hover)
    notes += mnotes
    flight_s = _total(intervals)
    if vi["quadplane"]:
        notes.append("QuadPlane:只分析 VTOL(Q 開頭)模式,參數前綴 Q_A_;支援為初步。")
    for nm_ in ("ATC_RAT_RLL_P", "ATC_RAT_PIT_P", "ATC_RAT_YAW_P", "ATC_ANG_RLL_P", "ATC_ANG_PIT_P"):
        nm2 = nm_.replace("ATC_", prefix)
        hist = index.param_history(nm2)
        if any(t0 <= t <= t1 for t, _ in hist[1:]):
            notes.append(f"{nm2} 在分析範圍內有變動;建議只分析同一組參數的飛行段(目前用最後的值)。")

    findings: list[Finding] = []
    # ---- 資料品質 ----
    qp = vi["quadplane"]
    pid_names = "PIQR / PIQP / PIQY" if qp else "PIDR / PIDP / PIDY"
    best_rate = max([x for x in (data["RATE_hz"], data["PID_hz"]) if x], default=None)
    data["rate_source"] = None
    if best_rate is None and data["ATT_hz"] is None:
        findings.append(Finding("no_attitude_data", SEV_INFO, None, "沒有姿態 / 角速度記錄",
                                f"log 裡沒有 ATT、RATE 或 {pid_names},無法分析 PID。",
                                [s for s in [_bitmask_suggestion(params, "開啟姿態與 PID 記錄", qp)] if s]))
    else:
        eff = best_rate if best_rate is not None else data["ATT_hz"]
        if eff is not None and _low_fs(eff):
            what = (f"角速度(RATE{'/PID' if data['PID'] else ''})只有 {eff:g} Hz" if best_rate is not None
                    else f"沒有角速度記錄,姿態(ATT)只有 {eff:g} Hz")
            det = (f"{what}(Nyquist {eff / 2:g} Hz)。常見的 PID 震盪在 5–20 Hz,這個取樣率看不到,"
                   "甚至會混疊成假的低頻;步階響應、頻寬、過衝也量不準。只能判讀慢速行為"
                   "(角度追蹤、延遲、偏差、慢速擺動),震盪只能從 IMU / CTRL 間接推測。")
            if not data["PID"]:
                det += f"沒有 {pid_names},無法分辨 P / I / D 各自的貢獻。"
            findings.append(Finding("low_rate", SEV_INFO, None, "記錄取樣率太低:看不到 PID 震盪", det,
                                    [s for s in [_bitmask_suggestion(params, "提高姿態 / PID 記錄頻率", qp)] if s]))
        elif not data["PID"]:
            why = ("LOG_BITMASK 沒開 ATTITUDE_FULLRATE(bit 20;Plane 的 PIQx 隨姿態記錄)" if qp
                   else "LOG_BITMASK 沒開 PID(bit 12)")
            findings.append(Finding("no_pid_log", SEV_INFO, None, "沒有 PID 細項記錄",
                                    why + ":看不到 P / I / D / FF 各項輸出與 slew 限制器狀態,只能從整體追蹤推論。",
                                    [s for s in [_bitmask_suggestion(params, "開啟 PID 細項記錄", qp)] if s]))
        if best_rate is None:
            findings.append(Finding("no_rate_data", SEV_INFO, None, "沒有角速度記錄,只用姿態角分析",
                                    "log 沒有 RATE / PIDR…:只能分析角度層級(ATT),角速度環的震盪與阻尼無法直接判斷。"))

    if flight_s < MIN_FLIGHT_S:
        findings.append(Finding("no_flight", SEV_INFO, None, "飛行資料不足",
                                f"分析範圍內只找到 {flight_s:.1f} s 的飛行(至少需要 {MIN_FLIGHT_S:g} s;"
                                "已排除地面、起降前後各 2 s 與 AUTOTUNE)。請選擇有飛行的範圍。"))
        findings.sort(key=_sev_key)
        rep = _empty_report(t0, t1, vi["vehicle"], vi["frame"], prefix, findings, notes, data, rel)
        rep.flight_s = flight_s
        return rep
    notes.append(f"分析了 {len(intervals)} 段飛行,共 {flight_s:.0f} s(已裁掉起降前後各 2 s)。")

    # ---- 整機 ----
    veh = VehicleContext(prefix=prefix, params=params)
    veh.pilot_share = _pilot_share(logdata, intervals)
    data["pilot_share"] = veh.pilot_share
    try:
        fv, veh.vibe, _ = _vibe_findings(index, intervals)
        findings += fv
    except Exception as e:  # noqa: BLE001
        notes.append(f"振動分析失敗:{e}")
    mst: dict = {}
    try:
        mst = _motor_stats(index, params, intervals, vi["frame_class"], prefix)
        findings += mst["findings"]
        veh.motor_sat, veh.motor_hi, veh.motor_lo = mst["share"], mst["hi"], mst["lo"]
        veh.thrust_loss = mst["thrust_loss"]
        if mst.get("note"):
            notes.append(mst["note"])
    except Exception as e:  # noqa: BLE001
        notes.append(f"馬達輸出分析失敗:{e}")
    try:
        fh, hmet = _hover_findings(index, params, intervals, qp)
        findings += fh
        hover_meas = hmet.get("hover_thr")
        if qp and not hmet:
            notes.append("QuadPlane 沒有可用的 QTUN(VTOL 升力油門):跳過懸停油門檢查"
                         "(Plane 的 CTUN.ThO 是推進油門,不能用)。")
    except Exception as e:  # noqa: BLE001
        notes.append(f"懸停油門分析失敗:{e}")
        hover_meas = None
    veh.hover = hover_meas
    try:
        findings += _notch_findings(index, params, intervals, hover or hover_meas, data)
    except Exception as e:  # noqa: BLE001
        notes.append(f"陷波 / 批次 FFT 分析失敗:{e}")
    veh.batch = bool(data.get("batch"))
    dg = _default_gain_finding(params, prefix)
    if dg is not None:
        findings.append(dg)

    # ---- 各軸 ----
    axes: dict[str, AxisAnalysis] = {}
    axis_osc: dict[str, bool] = {}
    ctxs: dict[str, AxisContext] = {}
    for ax in AXES:
        try:
            aa, ctx, fl = _analyze_axis(index, ax, vi, veh, att, rate_tbl, pid_tbls.get(ax), imu,
                                        intervals, t0, t1, data)
        except Exception as e:  # noqa: BLE001 — 一軸失敗不影響其他軸
            notes.append(f"{_AXIS_SHORT[ax]} 分析失敗:{type(e).__name__}: {e}")
            continue
        if aa is None:
            continue
        axes[ax] = aa
        ctxs[ax] = ctx
        findings += fl
        # gyro_noise(高於控制頻寬的噪音峰)不算震盪跡象
        axis_osc[ax] = any(f.code in ("rate_osc_high", "rate_osc_mid", "rate_osc_suspect", "slew_limit",
                                      "d_dominant",
                                      "angle_wobble", "angle_overshoot", "angle_freq_peak",
                                      "rate_underdamped") for f in fl)
    veh.rate_fs = best_rate
    lims = [c.pid.limit_share for c in ctxs.values() if c.pid is not None and c.pid.limit_share is not None]
    veh.pid_limit = max(lims) if lims else None

    # ---- 馬達飽和(上 / 下限分開;下限要看各軸的雜訊證據)----
    noise_ev = [e for c in ctxs.values() for e in _axis_noise_evidence(c)]
    if data.get("batch_peak_hz"):
        noise_ev.append(f"批次 FFT 在 {data['batch_peak_hz']:.0f} Hz 有噪音峰")
    try:
        findings += _motor_sat_findings(mst, params, prefix, hover_meas, noise_ev)
    except Exception as e:  # noqa: BLE001
        notes.append(f"馬達飽和分析失敗:{e}")

    # ---- 實際納入分析的比例(掉訊息 / 資料空洞)----
    cov = sorted((data.get("coverage_axes") or {}).items(), key=lambda kv: AXES.index(kv[0]))
    if cov:
        data["coverage"] = min(c for _, c in cov)
        if data["coverage"] < COVERAGE_WARN:
            findings.append(Finding(
                "low_coverage", SEV_INFO, None, f"只有 {data['coverage'] * 100:.0f} % 的飛行資料能分析",
                "、".join(f"{_AXIS_SHORT[a]} {c * 100:.0f} %" for a, c in cov)
                + ":記錄有資料空洞或大量掉訊息(SD 卡太慢 / LOG_FILE_BUFSIZE 太小時常見,DSF 訊息的 Dp 會增加),"
                "頻譜與追蹤統計只根據剩下的片段,可信度較低。",
                [_check("SD 卡 / LOG_FILE_BUFSIZE", "換高速 SD 卡、加大記錄緩衝,或只在調參飛行時開高速記錄", 3)]))
    apply_safety(findings, veh, axis_osc)
    if veh.closed_loop_des and axes:
        notes.append(f"飛行中只有 {veh.pilot_share * 100:.0f} % 在 STABILIZE / ALT_HOLD 等「搖桿直接給姿態」的模式:"
                     "期望姿態主要由位置控制(LOITER / AUTO…)產生、含回授,角度環的頻率響應 / 過衝不採用;"
                     "要評估角度環與角速度環的反應,請在 STABILIZE / ALT_HOLD 下做幾次快速搖桿動作再分析。")

    # ---- 每軸 ok / 結論 ----
    # 振動過大 / 馬達飽和 / 推力不足時,規則刻意不判斷該軸(誤差反映的是裕度 / 量測問題):不能說「追蹤良好」
    blk = veh.block_increase
    for ax, aa in axes.items():
        mine = [f for f in findings if f.axis == ax]
        if not any(SEV_ORDER[f.severity] >= SEV_ORDER[SEV_INFO] for f in mine):
            low = _low_fs(aa.sample_hz)
            m = aa.metrics
            u = "°/s" if aa.level == "rate" else "°"
            det = (f"RMS 誤差 {_fmt(m.get('rms_err'))} {u}、延遲 {_fmt(m.get('lag_ms'), 0)} ms"
                   + (f"、過衝 {_fmt(m.get('overshoot_pct'), 0)} %" if m.get("overshoot_pct") is not None else "")
                   + "。")
            if blk is not None:
                findings.append(Finding(
                    "not_judged", SEV_INFO, ax, f"{_AXIS_SHORT[ax]} 暫不評估({blk},先處理後再分析)",
                    f"{blk}時姿態誤差主要反映控制裕度 / 量測問題,不能拿來判斷 PID 好壞(目前 " + det[:-1]
                    + ")。先處理後再飛一次分析。", [], None, None, dict(m)))
            else:
                if low:
                    det += f"({aa.sample_hz:g} Hz 資料只看得到 {aa.sample_hz / 2:g} Hz 以下,高頻震盪無法判斷。)"
                findings.append(Finding("ok", SEV_OK, ax,
                                        f"{_AXIS_SHORT[ax]} 追蹤良好" + ("(在可觀測範圍內)" if low else ""),
                                        det, [], None, None, dict(m)))
            mine = [f for f in findings if f.axis == ax]
        worst = max(mine, key=lambda f: (SEV_ORDER[f.severity], -_CODE_RANK.get(f.code, 50),
                                         -min((s.priority for s in f.suggestions), default=9)))
        aa.verdict = worst.severity
        aa.summary = _axis_summary(ax, aa, worst)
    findings.sort(key=_sev_key)
    return TuningReport(t0=float(t0), t1=float(t1), flight_s=flight_s, vehicle=vi["vehicle"],
                        frame=vi["frame"], param_prefix=prefix, data=data, axes=axes,
                        findings=findings, params=rel, notes=notes)


def _axis_summary(ax: str, aa: AxisAnalysis, worst: Finding) -> str:
    m = aa.metrics
    u = "°/s" if aa.level == "rate" else "°"
    tail = f"RMS 誤差 {_fmt(m.get('rms_err'))} {u}"
    if m.get("lag_ms") is not None:
        tail += f"、延遲 {_fmt(m.get('lag_ms'), 0)} ms"
    head = worst.title
    if worst.suggestions:
        s = min(worst.suggestions, key=lambda s: s.priority)
        if s.direction in (DIR_UP, DIR_DOWN) and s.proposed is not None:
            head += f";建議 {s.param} {_fmt_param(s.current)} → {_fmt_param(s.proposed)}"
    return f"{head}({tail})。"


def _analyze_axis(index, ax: str, vi: dict, veh: VehicleContext, att, rate_tbl, pid_tbl, imu,
                  intervals: list[Interval], t0: float, t1: float, data: dict):
    """單軸:組 LevelStats / IMU / CTRL / PID 統計 → 規則 → AxisAnalysis。"""
    nmap = gain_names(veh.prefix, ax)
    params = veh.params
    rate_ls = ang_ls = None
    plot_rate = plot_ang = None
    src = None
    target_txt = None
    cov_rate = cov_ang = None
    # 角速度層級:RATE(RDes = 未濾波的角速度目標)優先;RATE 沒記或慢很多時才用 PIDx(Tar 已經過 FLTT 低通,
    # 延遲 / |H| 換算回未濾波目標的基準,門檻才一致)。PIDx 另外用在 P / I / D / FF / Dmod / Flags 細項。
    rate_fs = _fs_of(rate_tbl["t"]) if rate_tbl is not None and _RATE_FIELDS[ax][0] in rate_tbl else 0.0
    pid_fs = 0.0
    tar = None
    if pid_tbl is not None and "Act" in pid_tbl:
        tar = "Tar" if "Tar" in pid_tbl else ("Des" if "Des" in pid_tbl else None)
        if tar is not None:
            pid_fs = _fs_of(pid_tbl["t"])
    fltt = None
    if rate_fs > 0 and rate_fs >= 0.9 * pid_fs:
        fd, fa, _ = _RATE_FIELDS[ax]
        t = rate_tbl["t"]
        d = np.asarray(rate_tbl[fd], float)
        a = np.asarray(rate_tbl[fa], float)
        fs, src = rate_fs, "RATE"
        target_txt = f"RATE.{fd}"
    elif tar is not None and pid_fs > 0:
        k = _pid_units_scale(index, vi["pid_msg"][ax], rate_tbl, ax, intervals)
        t = pid_tbl["t"]
        d = np.asarray(pid_tbl[tar], float) * k
        a = np.asarray(pid_tbl["Act"], float) * k
        fs, src = pid_fs, vi["pid_msg"][ax]
        fltt = _num(params.get(nmap["FLTT"]))
        fltt = fltt if fltt and fltt > 0 else None
        target_txt = f"{src}.{tar}" + (f"(+ FLTT {fltt:g} Hz 補償)" if fltt else "")
    else:
        t = d = a = None
        fs = 0.0
    if t is not None:
        pcs = _make_pieces(t, [d, a], intervals, fs)
        cov_rate = _coverage(pcs, intervals)
        rate_ls = level_stats(pcs, fs, "rate", ax, tar_fltt=fltt)
        plot_rate = _plot_arrays(t, d, a, t0, t1, intervals)
        if rate_ls is not None:
            data["rate_source"] = src
    # 輸出飽和 / 懸停配平(RATE.xOut)
    out_mean = None
    if rate_tbl is not None and _RATE_FIELDS[ax][2] in rate_tbl:
        m = _in_intervals(rate_tbl["t"], intervals)
        if m.sum() >= 10:
            o = np.asarray(rate_tbl[_RATE_FIELDS[ax][2]], float)[m]
            dd = np.asarray(rate_tbl[_RATE_FIELDS[ax][0]], float)[m]
            if rate_ls is not None:
                rate_ls.sat_share = float(np.mean(np.abs(o) >= OUT_SAT_LEVEL))
            q = np.abs(dd) < BIAS_RATE_DES_MAX
            if q.sum() >= 10:
                out_mean = float(np.mean(o[q]))
    # 角度層級:ATT
    if att is not None and _ATT_FIELDS[ax][0] in att:
        fd, fa = _ATT_FIELDS[ax]
        ta = att["t"]
        da = np.asarray(att[fd], float)
        aa_ = np.asarray(att[fa], float)
        if ax == "yaw":
            da, aa_ = _unwrap_yaw(da, aa_)
        fsa = _fs_of(ta)
        pcs = _make_pieces(ta, [da, aa_], intervals, fsa)
        cov_ang = _coverage(pcs, intervals)
        ang_ls = level_stats(pcs, fsa, "angle", ax)
        plot_ang = _plot_arrays(ta, da, aa_, t0, t1, intervals)
    cov_main = cov_rate if t is not None else cov_ang
    if cov_main is not None:
        data.setdefault("coverage_axes", {})[ax] = cov_main          # 就算這一軸整個分析不了也要記
    if rate_ls is None and ang_ls is None:
        return None, None, []
    # IMU 陀螺儀(取樣比角速度層級快時才有額外資訊)
    imu_fs, imu_peaks = None, []
    if imu is not None and _IMU_FIELDS[ax] in imu:
        f_imu = _fs_of(imu["t"])
        if f_imu > (rate_ls.fs if rate_ls is not None else 0.0) * 1.2:
            g = np.asarray(imu[_IMU_FIELDS[ax]], float)
            u = (index.unit(f"IMU.{_IMU_FIELDS[ax]}") or "rad/s").lower()
            if u.startswith("rad"):
                g = np.degrees(g)
            imu_fs, imu_peaks = _gyro_peaks(imu["t"], g, intervals)
    ctrl = _ctrl_stats(index, ax, intervals, params, nmap) if data.get("CTRL") else None
    pid = _pid_stats(index, vi["pid_msg"][ax], intervals, params, nmap) if pid_tbl is not None else None
    ctx = AxisContext(ax, veh, rate_ls, ang_ls, imu_peaks, imu_fs, ctrl, pid, out_mean)
    fl = axis_rules(ctx)

    # ---- AxisAnalysis ----
    prim = rate_ls if rate_ls is not None else ang_ls
    level = "rate" if rate_ls is not None else "angle"
    pr = plot_rate if rate_ls is not None else plot_ang
    met: dict = {"fs": prim.fs, "source": src if rate_ls is not None else "ATT",
                 "rate_target": target_txt if rate_ls is not None else None,
                 "coverage": cov_rate if rate_ls is not None else cov_ang,
                 "rate_coverage": cov_rate, "ang_coverage": cov_ang}

    def put(prefix_, ls: LevelStats | None):
        if ls is None:
            return
        met.update({f"{prefix_}rms_err": ls.rms_err, f"{prefix_}max_err": ls.max_err,
                    f"{prefix_}bias": ls.bias, f"{prefix_}lag_ms": ls.lag_ms,
                    f"{prefix_}tracking_pct": (None if ls.track_ratio is None
                                               else max(0.0, 100.0 * (1.0 - ls.track_ratio))),
                    f"{prefix_}fs": ls.fs})

    put("rate_", rate_ls)
    put("ang_", ang_ls)
    put("", prim)
    met["sat_pct"] = None if (rate_ls is None or rate_ls.sat_share is None) else rate_ls.sat_share * 100
    met["overshoot_pct"] = prim.overshoot if prim.step_usable() else None
    met["rise_ms"] = prim.rise_s * 1000 if (prim.step_usable() and prim.rise_s is not None) else None
    met["step_input_bw_hz"] = prim.input_bw
    met["bandwidth_hz"] = prim.bw_hz
    met["bandwidth_gt_hz"] = prim.bw_gt
    met["h_peak"] = prim.h_peak
    met["incoherent_pct"] = None if prim.incoherent is None else prim.incoherent * 100
    met["out_mean"] = out_mean
    # 沒有 CTRL 佐證的「疑似震盪」、高於控制頻寬的噪音峰、可能是混疊的峰都不寫進 osc_hz
    # (卡片會顯示成震盪頻率,那是過度宣稱);沒有就明確寫 None
    osc = [f for f in fl if f.metrics.get("osc_hz") is not None
           and not (f.code == "rate_osc_suspect" and not f.metrics.get("support"))]
    met["osc_hz"] = None
    if osc:
        met["osc_hz"] = osc[0].metrics["osc_hz"]
    elif prim.peaks and _full_rate(prim.fs, params):
        ok_pk = [p for p in prim.peaks if p.f <= ctrl_fmax(params)]
        if ok_pk:
            met["osc_hz"] = max(ok_pk, key=lambda p: p.rms).f
    noise = [f for f in fl if f.code == "gyro_noise"]
    if noise:
        met["noise_hz"] = noise[0].metrics["noise_hz"]
    if imu_peaks:
        best = max(imu_peaks, key=lambda p: p.rms)
        met.update({"imu_fs": imu_fs, "imu_peak_hz": best.f, "imu_peak_rms": best.rms})
    if ctrl is not None:
        met.update({"ctrl_d_over_p": ctrl.ratio, "ctrl_f_est_hz": ctrl.f_est, "ctrl_f_est_gt_hz": ctrl.f_est_gt})
    if pid is not None:
        met.update({"pid_dmod_pct": None if pid.dmod_share is None else pid.dmod_share * 100,
                    "pid_i_sat_pct": None if pid.i_sat_share is None else pid.i_sat_share * 100,
                    "pid_d_over_p": pid.d_rms / pid.p_rms if pid.p_rms > 0 else None})
    aa = AxisAnalysis(axis=ax, level=level, sample_hz=float(prim.fs),
                      t=pr[0] if pr else np.zeros(0), des=pr[1] if pr else np.zeros(0),
                      act=pr[2] if pr else np.zeros(0), metrics=met,
                      spectrum=prim.spectrum, freq=prim.freq,
                      step=prim.step if prim.step_usable() else None)
    if plot_ang is not None:
        aa.t_ang, aa.des_ang, aa.act_ang = plot_ang
    return aa, ctx, fl


# =====================================================================
# 文字報告
# =====================================================================

_DIR_TXT = {DIR_UP: "調高", DIR_DOWN: "調低", DIR_SET: "設為", DIR_CHECK: "檢查"}


def _sug_line(s: Suggestion) -> str:
    if s.direction == DIR_CHECK:
        return f"檢查 {s.param}:{s.reason}"
    cur = _fmt_param(s.current)
    if s.proposed is not None:
        pct = f"({s.change_pct:+.0f} %)" if s.change_pct is not None else ""
        return f"{s.param} {cur} → {_fmt_param(s.proposed)}{pct}:{s.reason}"
    pct = f" {s.change_pct:+.0f} %" if s.change_pct is not None else ""
    return f"{_DIR_TXT.get(s.direction, s.direction)} {s.param}{pct}(目前 {cur}):{s.reason}"


def format_report_text(report: TuningReport, t_base: float = 0.0) -> str:
    """純文字報告(CLI、「複製」用);時間以 t_base 為零點顯示。"""
    from .timeline import format_clock as _fc

    def format_clock(x: float) -> str:
        return _fc(0.0 if -0.05 < x < 0 else x)

    L: list[str] = ["PID 調參建議", "=" * 40]
    veh = report.vehicle + (f"({report.frame})" if report.frame else "")
    L.append(f"機型:{veh};參數前綴 {report.param_prefix or '—'}")
    L.append(f"分析範圍:{format_clock(report.t0 - t_base)} – {format_clock(report.t1 - t_base)}"
             f"(納入飛行 {report.flight_s:.0f} s)")
    d = report.data or {}
    parts = []
    for k, lab in (("RATE_hz", "RATE"), ("ATT_hz", "ATT"), ("PID_hz", "PID"), ("IMU_hz", "IMU")):
        if d.get(k):
            parts.append(f"{lab} {d[k]:g} Hz")
    if d.get("PID") is False:
        parts.append("無 PID 記錄")
    if "batch" in d:
        parts.append("有批次 FFT" if d["batch"] else "無批次 FFT")
    if parts:
        L.append("資料:" + "、".join(parts))
    L.append(f"整體評估:{SEV_LABEL.get(report.verdict, report.verdict)}")
    if report.axes:
        L.append("")
        L.append("各軸摘要")
        for ax in AXES:
            aa = report.axes.get(ax)
            if aa is None:
                continue
            lvl = "角速度" if aa.level == "rate" else "角度"
            L.append(f"  [{AXIS_LABEL[ax]}] {SEV_LABEL.get(aa.verdict, aa.verdict)}({lvl} {aa.sample_hz:g} Hz)")
            if aa.summary:
                L.append(f"      {aa.summary}")
    L.append("")
    L.append("診斷")
    if not report.findings:
        L.append("  (沒有發現)")
    for f in report.findings:
        axl = f"[{_AXIS_SHORT.get(f.axis, f.axis)}] " if f.axis else ""
        when = f" @ {format_clock(f.t_focus - t_base)}" if f.t_focus is not None else ""
        L.append(f"  [{SEV_LABEL.get(f.severity, f.severity)}] {axl}{f.title}{when}")
        if f.detail:
            L.append(f"      {f.detail}")
        for s in sorted(f.suggestions, key=lambda s: s.priority):
            L.append(f"      → [{s.priority}] {_sug_line(s)}")
    sug = report.suggestions()
    if sug:
        L.append("")
        L.append("建議清單(依優先順序;同一參數只列一次)")
        for i, s in enumerate(sug, 1):
            L.append(f"  {i}. [{s.priority}] {_sug_line(s)}")
    if report.notes:
        L.append("")
        L.append("注意")
        L += [f"  - {n}" for n in report.notes]
    return "\n".join(L) + "\n"
