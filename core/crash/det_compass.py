"""羅盤 / 航向偵測模組(domain = 'compass'):羅盤與 EKF 航向是不是失事原因?

六個偵測器(皆為 @guarded,任何情況至少回傳一筆 Finding):
  ekf_mag_reject              EKF 磁力計創新量被拒絕(3D 融合 SM / IM*;地面偏航融合 IYAW)
  compass_raw_consistency     原始羅盤一致性(磁場強度突變 / 雙羅盤方向 / EKF 航向 vs 羅盤航向;先扣除馬達電流的磁場)
  compass_motor_interference  馬達電流造成的羅盤干擾量(健康度指標,不是肇因)
  ekf_yaw_reset_lane_switch   飛行中 EKF 航向重置 / lane 切換 / 來源組切換造成航向來源改變
  compass_sensor_health       羅盤感測器掉線 / 不健康 / EKF 磁力計逾時
  ekf_attitude_consistency    EKF 姿態 vs 獨立姿態(AHR2 / DCM)是否一致(免責 / 排除用)

判讀原則(設計文件):羅盤警報要和「姿態失控」比先後才有意義。
  - 每個警報偵測器都把 t_onset(條件開始成立的相對秒)與 values['lead_s'](= t_warn − t_onset,
    正 = 早於姿態誤差 >10°)、values['verdict'](lead ≥ 1.0 s → possible_cause,否則 effect_not_cause)
    一併輸出,由 engine 貼 precursor / consequence 標籤;本模組自己不計分。
  - 00000124(唯一一次失事):所有羅盤警報都在 t_warn(1255.9)之後 2.1~4.0 s 才出現(翻機的連帶現象),
    事前羅盤 / 航向健康(SM ≤0.15、|IM| ≤28 mG、|B0| 穩定、無航向重置)→ 羅盤不是原因。
  - 依 XKFS.MAG_FUSION 選訊號:1 = 地面偏航融合(只有 XKF3.IYAW 有值),2 = 飛行中 3D 融合
    (只有 XKF4.SM、XKF3.IM* 有值)。
  - 只看解鎖區間 a.interval(上鎖之後 EKF 會掉回模式 1,IYAW 暴衝,不屬於飛行證據)。
  - 未解鎖(桌上 / 手持)log:全部 NA(context='not_armed')。

信心備註(honesty):唯一的失事樣本是 124,所有「警報」門檻的證據都是「正常飛行(飛行 1 + 飛行 2
失事前約 659 s)不會觸發」的邊際,沒有任何「真的羅盤故障」的正樣本。羅盤警報的 t_onset 順序(晚於姿態
失控)是實測事實;門檻本身(35 mG / 30° / 15° / 10°)是工程邊際,詳見各常數的註解。
馬達電流補償(RC_MOT_*)的「不誤報」證據:124 若 EKF 改用干擾 34-40% 的 MAG[1](假設情境,XKFS.MI 改 1)
飛行 1 / 飛行 2 前段都 CLEAR,合成資料 10-80% 干擾、爬升、快速轉向都 CLEAR;「仍抓得到」證據只有合成故障
(懸停中 |B| +15%、和油門變化同時的跳變),沒有真實的羅盤故障樣本。
"""
from __future__ import annotations

import re

import numpy as np

from .ctx import Ctx
from .guard import guarded
from .types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval, Finding
from .util import runs, wrap180

DOMAIN = "compass"

# ======================================================================================
# 共用常數
# ======================================================================================
MAX_LANES = 4                 # EKF lane / MAG instance 最多掃到編號幾
LANE_NEED = 2                 # lane 投票:至少 2 組 lane 同時超標才算(lane 數不足時取全部)
MIN_COVER_S = 10.0            # 沒警報時,要有這麼多秒的有效資料才可以說 CLEAR(否則 NA)
JITTER_TOL_S = 0.02           # 「持續 ≥ N 秒」判斷容許的時間戳抖動(10 Hz 取樣的 0.3 s = 4 個樣本)
LEAD_MIN_S = 1.0              # 羅盤警報早於 t_warn 至少這麼多秒才算 possible_cause(catalogue H_COMPASS_YAW)
TILT_SYMPTOM_DEG = 45.0       # 機身傾角(|Roll| 或 |Pitch|)超過它之後才出現的警報 = 翻機後遺症
MAG_MAX_DT_S = 0.25           # MAG / ATT(10 Hz,最大空洞 0.8 s)連續樣本時間差超過它 → run 斷開
XKF_MAX_DT_S = 0.25           # XKF(25 Hz)同上

# ======================================================================================
# ekf_mag_reject
# ======================================================================================
MR_SM_GATE = 1.0              # XKF4.SM > 1 = 磁力計被拒絕。飛行 1 / 2 的 SM 最大 0.15 → 6.7x 邊際;
                              # ≥0.1 在正常降落(616.8-619.1)也會誤報,所以不能降
MR_IM_GATE_MG = 100.0         # |XKF3.IMX/IMY/IMZ| > 100 mG。正常 3D 融合飛行最大 28 mG(3.6x);失事 530 mG
MR_IYAW_GATE_DEG = 15.0       # 地面偏航融合 |XKF3.IYAW| > 15°。解鎖區間內最大 2.4°(6.3x);baselines 0.7-9.7°
MR_MIN_DUR_S = 0.5            # 持續 ≥ 0.5 s
MR_BRIDGE_S = 0.15            # SM 約 8 Hz 更新後保持(25 Hz 內插三格):單次更新跌破 1 不應把一次事件切成兩段
                              # (124:SM 1259.70-1259.78 為 1.13,1259.82-1259.86 為 0.72,1259.90 起 >1)
MR_MIN_OVERLAP_S = 0.2        # 兩組 lane 同時超標的重疊至少這麼長
MR_T_MIN_S = 10.0             # 開機 10 s 內(初始對準)不看

# ======================================================================================
# compass_raw_consistency
# ======================================================================================
RC_MAG_JUMP_MG = 35.0         # |B| 偏離 30 s 中位數 > 35 mG(≈8% 地磁)。非飛行最大 29.4 mG(119 校正後),124 失事前 <25 mG
RC_MAG_REF_WIN_S = (30.0, 0.5)   # 參考值 = (t−30, t−0.5] 的中位數
RC_MAG_REF_MIN_N = 20
RC_MAG_T_MIN_S = 35.0
# ---- 馬達電流補償(compassmot 慣例;三項檢查共用)----
# 馬達電流在羅盤上疊加一個「固定在機體上、與油門成正比」的磁場。EKF 用的羅盤若干擾 10-40%(業餘機架常見),
# 起飛 / 降落的油門階躍會讓 |B| 跳 40-250 mG、懸停時的固定干擾會讓快速轉向時的羅盤航向偏 20°:
# 124 若 EKF 改用 MAG[1](34-40%):飛行 1 磁場強度 340.34 s HIT(偏離 223 mG)、飛行 2 起飛 878.64 s HIT、
# 1053.5 s 轉向 50° 時航向殘差 20°(油門不變)。因此先用解鎖區間(事件前、水平)的資料擬合每顆羅盤的模型:
#   純量 |B| = a + b·x(磁場強度檢查用;x = 油門)        124:MAG[0] b ≈ −71、MAG[1] b ≈ +690 mG / 單位油門
#   向量 B_body = Rᵀ·g + b0 + x·c(航向 / 雙羅盤檢查用,扣掉 x·c)
# 再在補償後的資料上做原本的檢查。
RC_MOT_SETTLE_S = 1.0         # 參考值只用解鎖 1 s 後(馬達已轉,MOT_SPOOL_TIME 0.5 s)的樣本:解鎖瞬間 idle 電流
                              # 讓 MAG[1] |B| 跳 +15 mG,而 ThrOut 仍 ≈0.003(模型看不到)→ 不能拿馬達未轉的樣本當參考
RC_MOT_FIT_TILT_DEG = 30.0    # 擬合只用 |Roll|、|Pitch| < 30° 的樣本(翻滾時的資料不納入)
RC_MOT_MIN_FIT_N = 200        # 擬合至少 20 s 樣本
RC_MOT_MIN_STEP = 0.05        # 油門要有「對比」:至少 RC_MOT_MIN_CONTRAST_N 個樣本離中位數 ≥ 0.05(地面 idle vs 懸停 0.15)
RC_MOT_MIN_CONTRAST_N = 10    # (1 s);沒有對比(油門幾乎不變)→ 斜率不可辨識,不補償
RC_MOT_THR_MAX_GAP_S = 0.3    # 油門樣本離 MAG 樣本超過 0.3 s → 該樣本不納入擬合
# 補償後仍超標的 run:若「模型預測的馬達磁場變化」至少是峰值偏離的一半 → 歸因馬達電流(不算警報)。
# 線性模型解釋不了的部分:電流 ∝ 推力^1.5(124 起飛 ThrOut 0.25 時 MAG[1] 比線性模型多 ~95 mG)、
# 馬達加速的瞬間電流、ThrOut 與 MAG 取樣時間差;實測 MAG[1] 起飛 / 降落殘差偏離 75-112 mG,
# 對應 |b|·Δx = 93-172 mG(比值 0.86-2.3)。124 MAG[0] 翻機的 run:|b|·Δx ≈ 21 mG、峰值 >50 mG(比值 <0.42)→ 照常警報。
RC_MAG_MOTOR_WIN_S = 1.0      # 油門變化看 run 前後 1 s(ThrOut 與 MAG 取樣時間差、電流落後油門)
RC_MAG_MOTOR_FRAC = 0.5
RC_MOT_VEC_GAIN = 0.8         # 磁場強度改用向量補償 |B − x·c| 的條件:擬合樣本上 |B| 殘差標準差 < 0.8 × 純量模型。
                              # 固定在機體上的懸停干擾有水平分量時,|B| 會隨航向變(純量模型看不到):124 飛行 2 若 EKF 用 MAG[1],
                              # 1053.5 s 轉向 50°(油門不變)純量殘差 48 mG;實測標準差 純量 / 向量:MAG[1] 16.8 / 8.3(→ 向量)、
                              # MAG[0] 1.8 / 3.5(→ 純量;MAG[0] 干擾小,向量係數 c 在兩次飛行間不穩定,只會加雜訊)
RC_PAIR_DEG = 30.0            # 兩顆羅盤方向夾角 > 30°。實測最大 21.3°(117,邊際 8.7°);124 飛行 12.9 / 17.5°
RC_PAIR_T_MIN_S = 10.0
RC_HEAD_DEG = 15.0            # EKF 航向 − 原始羅盤航向的殘差,相對 20 s 圓中位數偏離 > 15°。
                              # 非飛行最大 4.0°(119 未閘門 13.6°),124 失事前 ≤5.5°(邊際 2.7x)
RC_HEAD_REF_WIN_S = (20.0, 1.0)
RC_HEAD_REF_MIN_N = 20
RC_HEAD_T_MIN_S = 12.0
RC_MIN_DUR_S = 0.3
CAL_PRE_S = 120.0             # 羅盤校正窗 [t_OfsChange−120, t_OfsChange+5](119 的校正在 506.0)
CAL_POST_S = 5.0
CAL_SKIP_S = 5.0              # 開機 5 s 內的偏移變動是參數載入,不算校正
CAL_MAX_GATED_FRAC = 0.5      # 解鎖區間有超過一半被校正窗蓋住 → 這項檢查 NA

# ======================================================================================
# compass_motor_interference(compassmot 慣例)
# ======================================================================================
MI_RATIO_WARN = 0.30          # > 0.30 邊際;實測 124:主羅盤 0.128 / 0.074,備援 0.404 / 0.342
MI_RATIO_BAD = 0.60           # > 0.60 差
MI_GROUND_WIN_S = (63.0, 3.0)  # 地面參考 [t0−63, t0−3](馬達未轉)
MI_FLIGHT_T0_S = 10.0         # 飛行集 [t0+10, t1−5]
MI_FLIGHT_T1_S = 5.0
MI_THR_MIN = 0.10             # MOTB.ThrOut ≥ 0.10(124 懸停 ~0.15)
MI_LEVEL_DEG = 10.0           # |Roll|、|Pitch| < 10°
MI_MIN_SPAN_S = 20.0          # 地面 / 飛行各至少 20 s
MI_WARN_AT_WARN_S = 0.5       # 有 t_warn 時,飛行集在 t_warn−0.5 s 截止(翻機後不納入)

# ======================================================================================
# ekf_yaw_reset_lane_switch
# ======================================================================================
EV_YAW_RESET = 62
EV_SRC_SET = {85: 1, 86: 2, 87: 3}      # EK3_SOURCES_SET_TO_PRIMARY / SECONDARY / TERTIARY
ERR_SUBSYS_EKF_PRIMARY = 24
YR_T_MIN_S = 10.0             # 開機 10 s 內 = 初始對準(跳變可達 170°)
YR_ALIGN_AFTER_ARM_S = 15.0   # 解鎖後 15 s 內 = 空中偏航對準(MAG_FUSION 1→2,跳變 0.9-1.5°)
YR_JUMP_DEG = 10.0            # |跳變| > 10° 才算(正常 ≤1.5°,邊際 6.7x;沒有正樣本)
YR_JUMP_WIN_S = (0.1, 0.5)    # 轉動率取 (t−0.5, t−0.1] 的相鄰樣本;取樣範圍 [t−0.5, t+0.5]
YR_STEP_TOL_S = 0.06          # 跳變 = EV 62 時間 ±0.06 s 內相鄰 XKF1 樣本(25 Hz)的差 − 轉動率 × dt(EV 與 XKF 同一個 EKF 週期寫出)
YR_SPREAD_DEG = 10.0          # lane 間航向差 > 10°(時間對齊後 baselines ≤1.6°)
YR_SPREAD_MIN_DUR_S = 0.5
YR_SPREAD_T0_S = 15.0         # 解鎖後 15 s 才算(避開各 lane 各自對準的時間差)
YR_SPREAD_PRE_S = 60.0        # 有 t_warn 時,看 [t_warn−60, t_warn−1]

# ======================================================================================
# compass_sensor_health
# ======================================================================================
SH_HEALTH_MIN_DUR_S = 0.5     # MAG.Health == 0 持續 ≥ 0.5 s
SH_GAP_S = 2.0                # MAG 連續樣本間隔 > 2.0 s(正常 0.1 s,最大 0.8 s)
SH_ERR_SUBSYS_COMPASS = 3     # ERR COMPASS:ECode 1 初始化失敗、4 不健康;ECode 0 是開機雜訊
SH_ERR_BAD_CODES = (1, 4)
SH_ERR_T_MIN_S = 1.0
SH_TS_MAG_BIT = 8             # XKF4.TS bit3 = 磁力計逾時(bit4/5 永遠為 1,見 pitfalls)
SH_FS_T_MIN_S = 3.0           # XKF4.FS ≠ 0(開機 ~1 s 內的 128 = 狀態未初始化,正常)
SH_EKF_MIN_DUR_S = 0.1

# ======================================================================================
# ekf_attitude_consistency
# ======================================================================================
AC_RP_DEG = 15.0              # |ATT − AHR2| Roll / Pitch > 15°(失事過程最大 3.2 / 1.9°;飛行 1 / 2 失事前 pitch 7.6 / 7.7°)
AC_YAW_DEG = 20.0             # Yaw > 20°(失事過程 4.5°;baselines 手持旋轉 10.0°)
AC_DCM_YAW_DEG = 15.0         # |ATT.Yaw − DCM.Yaw| > 15°(常數 2-6° 偏移屬正常;飛行中最大 6.3°)
AC_MIN_DUR_S = 0.3            # AHR2 對照的持續時間(避免內插假象的單點)
AC_DCM_MIN_DUR_S = 0.5
AC_EVENT_PRE_S = 15.0         # 事件窗 [t_c − 15, t_c]:失事過程的報告數字
AC_T_MIN_S = 3.0              # 解鎖後 3 s 才算(AHR2 起始 ~2 s)
AC_BOOT_S = 10.0              # 開機 10 s 內不比:EKF 初始航向對準(2.5-3.8 s)之前 ATT.Yaw 停在 0/360,AHR2 已有航向
                              # → 拿掉解鎖閘門時 118/119/120/123 在 3.0-3.1 s 起 yaw 差 90-174°(假警報);10 s 後 ≤1.0°
AC_MAX_NEAR_S = 0.15          # 內插只用「附近 0.15 s 內有樣本」的點

CAUSALITY_END_TYPES = frozenset({"crash_check", "crash_check_replay", "hard_impact", "tumble",
                                 "abnormal_disarm", "log_ended_armed"})

SEV_MAG = "high"
SEV_RAW = "medium"
SEV_MOTOR = "medium"
SEV_RESET = "high"
SEV_HEALTH = "high"
SEV_ATT = "info"


# ======================================================================================
# 小工具
# ======================================================================================

def _num(x):
    """numpy 純量 → Python float / int;NaN / inf → None;容器遞迴處理(values 必須可 JSON 序列化)。"""
    if isinstance(x, dict):
        return {str(k): _num(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_num(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.ndarray):
        return _num(x.tolist())
    return x


def _r(x, nd=3):
    """四捨五入的 float(None / NaN → None)。"""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, nd) if np.isfinite(xf) else None


def _finding(det_id: str, status: str, severity: str, t_onset, t_alarm, context: str, values: dict,
             text: str) -> Finding:
    return Finding(id=det_id, domain=DOMAIN, status=status,
                   severity=severity if status in (HIT, SOFT) else "info",
                   t_onset=_r(t_onset, 4), t_alarm=_r(t_alarm, 4), context=context,
                   values=_num(values), text_zh=text)


def _na(det_id: str, context: str, text: str, **values) -> Finding:
    return _finding(det_id, NA, "info", None, None, context, values, text)


def _not_armed(det_id: str) -> Finding:
    return _na(det_id, "not_armed", "此 log 從未解鎖(桌上 / 手持測試),不做飛行中的羅盤 / 航向分析。")


def _lane_tables(ctx: Ctx, name: str):
    """該表所有 instance(EKF lane / MAG 編號)→ [(編號, 表)]。"""
    out = []
    for i in range(MAX_LANES):
        tb = ctx.table(name, i)
        if tb is not None and tb["t"].size:
            out.append((i, tb))
    return out


def _zoh(t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """零階保持:取 t_src ≤ t_dst 的最新值(各 EKF lane 時間戳不同,不能用索引對齊);之前沒有資料 → NaN。"""
    i = np.searchsorted(t_src, t_dst, side="right") - 1
    return np.where(i >= 0, v_src[np.clip(i, 0, None)], np.nan)


def _nearest_gap(t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """t_dst 每個點到最近 t_src 樣本的時間差(秒)。"""
    i = np.clip(np.searchsorted(t_src, t_dst), 1, t_src.size - 1)
    return np.minimum(np.abs(t_dst - t_src[i - 1]), np.abs(t_src[i] - t_dst))


def _runs_bridged(mask: np.ndarray, t: np.ndarray, min_dur: float, bridge: float, max_dt: float):
    """持續 ≥ min_dur 的連續為真區段 [(t_start, t_end, i0, i1)];間隔 ≤ bridge 秒的相鄰區段併成一段。"""
    merged: list[list] = []
    for s, e, i0, i1 in runs(mask, t, 0.0, max_dt):
        if merged and s - merged[-1][1] <= bridge:
            merged[-1][1], merged[-1][3] = e, i1
        else:
            merged.append([s, e, i0, i1])
    return [(s, e, i0, i1) for s, e, i0, i1 in merged if e - s >= min_dur - JITTER_TOL_S]


def _vote(lane_runs: list[list[tuple[float, float]]], need: int, min_overlap: float):
    """多組 lane 的區段投票:≥ need 組同時成立的區間 [(start, end)](start = 第 need 組加入的時間)。"""
    ev: list[tuple[float, int]] = []
    for lr in lane_runs:
        for s, e in lr:
            ev += [(s, 1), (e, -1)]
    ev.sort(key=lambda x: (x[0], -x[1]))           # 同時刻:先加入再離開
    out, cnt, start = [], 0, None
    for tt, d in ev:
        cnt += d
        if cnt >= need and start is None:
            start = tt
        elif cnt < need and start is not None:
            if tt - start >= min_overlap:
                out.append((start, tt))
            start = None
    return out


def _unwrap_interp(t_src: np.ndarray, deg: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """角度(度)先在弧度上 unwrap 再線性內插,回傳度(不折回,呼叫端自己 wrap180)。翻轉過 ±180° 不會跳。"""
    return np.degrees(np.interp(t_dst, t_src, np.unwrap(np.radians(deg))))


def _rolling_ref(t: np.ndarray, x: np.ndarray, idx: np.ndarray, win: tuple[float, float], min_n: int,
                 floor_t: np.ndarray | None = None, circular: bool = False) -> np.ndarray:
    """idx 每個樣本的參考值 = (t_i − win[0], t_i − win[1]] 內樣本的(圓)中位數;樣本數 < min_n → NaN。

    時間窗以「時間」計(資料空洞照常,不是以樣本數計)。floor_t:每個樣本的下限時間
    (羅盤校正窗結束後才開始累積參考)。窗長相同的樣本批次做(向量化),不逐樣本迴圈。
    """
    ti = t[idx]
    lo = np.searchsorted(t, ti - win[0], side="right")
    hi = np.searchsorted(t, ti - win[1], side="right")
    if floor_t is not None:
        fin = np.isfinite(floor_t)
        lo = np.maximum(lo, np.where(fin, np.searchsorted(t, np.where(fin, floor_t, 0.0), side="left"), 0))
    n = hi - lo
    out = np.full(idx.size, np.nan)
    xf = np.where(np.isfinite(x), x, np.nan)
    has_nan = bool(np.isnan(xf).any())
    for length in np.unique(n[n >= min_n]):
        rows = np.flatnonzero(n == length)
        for c0 in range(0, rows.size, 2000):               # 分批:控制記憶體
            r = rows[c0:c0 + 2000]
            seg = xf[lo[r][:, None] + np.arange(length)[None, :]]
            if has_nan:
                ok = np.isfinite(seg).sum(axis=1) >= min_n
            else:
                ok = np.ones(r.size, dtype=bool)
            med = np.nanmedian if has_nan else np.median
            if circular:
                rad = np.radians(seg)
                cen = np.degrees(np.arctan2(np.nanmean(np.sin(rad), axis=1), np.nanmean(np.cos(rad), axis=1)))
                out[r] = np.where(ok, wrap180(med(wrap180(seg - cen[:, None]), axis=1) + cen), np.nan)
            else:
                out[r] = np.where(ok, med(seg, axis=1), np.nan)
    return out


def _dt_nom(t: np.ndarray, default: float = 0.1) -> float:
    if t.size < 3:
        return default
    return float(np.median(np.diff(t)))


def _mag_tables(ctx: Ctx) -> dict[int, dict]:
    """MAG instance → 表(要有 MagX/Y/Z)。"""
    out = {}
    for i, tb in _lane_tables(ctx, "MAG"):
        if all(c in tb for c in ("MagX", "MagY", "MagZ")):
            out[i] = tb
    return out


def _vec(tb: dict) -> np.ndarray:
    return np.stack([tb["MagX"], tb["MagY"], tb["MagZ"]], axis=1)


def _used_compass(ctx: Ctx, iv: ArmedInterval, mags: dict[int, dict]) -> int | None:
    """EKF 實際使用的羅盤編號(XKFS.MI 在解鎖區間的中位數);沒有就取編號最小的 MAG。"""
    for _, ks in _lane_tables(ctx, "XKFS"):
        if "MI" in ks:
            m = (ks["t"] >= iv.t0) & (ks["t"] <= iv.t1) & np.isfinite(ks["MI"])
            if m.any():
                mi = int(round(float(np.median(ks["MI"][m]))))
                if mi in mags:
                    return mi
    return min(mags) if mags else None


def _cal_windows(mags: dict[int, dict]) -> list[tuple[float, float]]:
    """羅盤校正窗:任一羅盤的 OfsX/Y/Z 在開機 5 s 後有變動 → [t−120, t+5](合併重疊)。"""
    times: list[float] = []
    for tb in mags.values():
        t = tb["t"]
        for c in ("OfsX", "OfsY", "OfsZ"):
            if c in tb and tb[c].size > 1:
                ch = np.flatnonzero(np.abs(np.diff(tb[c])) > 0) + 1
                times += [float(t[k]) for k in ch if t[k] > CAL_SKIP_S]
    wins: list[list[float]] = []
    for tc in sorted(times):
        lo, hi = tc - CAL_PRE_S, tc + CAL_POST_S
        if wins and lo <= wins[-1][1]:
            wins[-1][1] = max(wins[-1][1], hi)
        else:
            wins.append([lo, hi])
    return [(a, b) for a, b in wins]


def _in_windows(t: np.ndarray, wins: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(t.size, dtype=bool)
    for a, b in wins:
        m |= (t >= a) & (t <= b)
    return m


def _att_tilt45(ctx: Ctx, iv: ArmedInterval) -> float | None:
    """解鎖區間內第一次 |Roll| 或 |Pitch| ≥ 45° 的時間(ATT 10 Hz);沒有 → None。"""
    at = ctx.table("ATT")
    if at is None or "Roll" not in at:
        return None
    m = (at["t"] >= iv.t0) & (at["t"] <= iv.t1) & ((np.abs(at["Roll"]) >= TILT_SYMPTOM_DEG)
                                                    | (np.abs(at["Pitch"]) >= TILT_SYMPTOM_DEG))
    return float(at["t"][m][0]) if m.any() else None


def _causality(a: Anchors, t_onset: float, t_tilt: float | None) -> dict:
    """警報相對姿態失控的先後:lead_s = t_warn − t_onset(正 = 早於);verdict 依 catalogue 的 1.0 s 規則。

    沒有 t_warn(姿態模組沒給錨點)時退而用機身傾角 ≥45° 的時間:警報不早於翻覆 1.0 s 以上
    (lead_tilt45_s < LEAD_MIN_S)→ effect_not_cause;早得多 → unknown(無法判定是不是原因)。
    """
    out: dict = {"t_tilt45": t_tilt}
    lead = (a.t_warn - t_onset) if a.t_warn is not None else None
    lead_tilt = (t_tilt - t_onset) if t_tilt is not None else None
    out["lead_s"] = lead
    out["lead_tilt45_s"] = lead_tilt
    if lead is not None:
        out["verdict"] = "possible_cause" if lead >= LEAD_MIN_S else "effect_not_cause"
    elif lead_tilt is not None and lead_tilt < LEAD_MIN_S:
        out["verdict"] = "effect_not_cause"
    else:
        out["verdict"] = "unknown"          # 沒有姿態失控錨點(例如正常降落):警報本身就是異常,但無先後可比
    out["after_tilt45"] = (t_tilt is not None and t_onset >= t_tilt)
    return out


def _cause_text(c: dict, a: Anchors) -> str:
    v = c["verdict"]
    if v == "effect_not_cause":
        if a.t_warn is not None and c["lead_s"] is not None:
            if c["lead_s"] < 0:
                return f"晚於姿態誤差 >10°({a.t_warn:.2f} s)達 {-c['lead_s']:.1f} s,屬翻機後的連帶現象,不是原因"
            # 早於 t_warn 但不到 1 s:不是「之後」,只能說前導時間不足以當成原因
            return (f"與姿態失控(姿態誤差 >10°,{a.t_warn:.2f} s)幾乎同時發生(只早 {c['lead_s']:.1f} s,"
                    f"未達 {LEAD_MIN_S:.0f} s 的前導),無法視為原因")
        tt, lt = c.get("t_tilt45"), c.get("lead_tilt45_s")
        if tt is not None and lt is not None and lt > 0:
            return (f"與機身翻覆(傾角超過 45°,{tt:.2f} s)幾乎同時(只早 {lt:.1f} s,未達 {LEAD_MIN_S:.0f} s 的前導),"
                    "無法視為原因,較可能是翻機的連帶現象")
        tail = f"({tt:.2f} s)" if tt is not None else ""
        return f"出現在機身傾角超過 45°{tail}之後,屬翻機後的連帶現象,不是原因"
    if v == "possible_cause":
        return f"早於姿態誤差 >10° 達 {c['lead_s']:.1f} s,可能是原因,請檢查羅盤校正與磁干擾(MAG.*)"
    return "沒有姿態失控錨點可比先後,請自行檢查羅盤校正與磁干擾"


def _event_start(a: Anchors, t_tilt: float | None = None) -> float | None:
    """「事前」統計的截止時間:事件起點 = 最早的 t_first_anomaly / t_touch / t_warn / 傾角 ≥45°;都沒有 → None。

    124:觸地 1255.225 之後 EKF 才開始受擾(SM 0.15 → 0.29),所以事前統計止於觸地而不是 t_warn。
    t_tilt(傾角 ≥45°)只在其他錨點缺席時才會是最早的:避免「警報前」統計把翻滾過程算進去。
    """
    c = [v for v in (a.t_first_anomaly, a.t_touch, a.t_warn, t_tilt) if v is not None]
    return min(c) if c else None


def _ctx_label(c: dict) -> str:
    """Finding.context:symptom = 晚於姿態失控的連帶現象;precursor_candidate = 早於 ≥1 s;unanchored = 沒有錨點可比。"""
    return {"effect_not_cause": "symptom", "possible_cause": "precursor_candidate"}.get(c["verdict"], "unanchored")


# ======================================================================================
# 1. ekf_mag_reject
# ======================================================================================

def _mag_lane_arrays(ctx: Ctx, iv: ArmedInterval, lane: int, k4: dict, ks: dict | None, k3: dict | None):
    """一條 EKF lane 在解鎖區間內、XKF4 時間軸上的 SM / MAG_FUSION / |IM*| / |IYAW|(XKF3、XKFS 零階保持對齊)。"""
    t4 = k4["t"]
    sel = (t4 >= max(iv.t0, MR_T_MIN_S)) & (t4 <= iv.t1)
    t = t4[sel]
    sm = k4["SM"][sel] if "SM" in k4 else np.full(t.size, np.nan)
    mf = _zoh(ks["t"], ks["MAG_FUSION"], t) if ks is not None and "MAG_FUSION" in ks else np.full(t.size, np.nan)
    if k3 is not None and all(c in k3 for c in ("IMX", "IMY", "IMZ")) and t.size:
        ims = np.stack([_zoh(k3["t"], np.abs(k3[c]), t) for c in ("IMX", "IMY", "IMZ")], axis=1)
        im = np.nanmax(np.where(np.isfinite(ims), ims, -1.0), axis=1)
        im = np.where(im < 0, np.nan, im)
        im_ax = np.argmax(np.where(np.isfinite(ims), ims, -1.0), axis=1)
    else:
        ims = np.full((t.size, 3), np.nan)
        im = np.full(t.size, np.nan)
        im_ax = np.zeros(t.size, dtype=int)
    iyaw = _zoh(k3["t"], np.abs(k3["IYAW"]), t) if k3 is not None and "IYAW" in k3 and t.size else np.full(t.size, np.nan)
    return dict(lane=lane, t=t, sm=sm, mf=mf, im=im, im_ax=im_ax, iyaw=iyaw)


@guarded("ekf_mag_reject", DOMAIN)
def ekf_mag_reject(ctx: Ctx, a: Anchors):
    """EKF 磁力計創新量被拒絕。依 XKFS.MAG_FUSION 選訊號:模式 2 看 SM>1 或 |IM*|>100 mG,模式 1 看 |IYAW|>15°。

    各 lane 的時間戳不同,逐 lane 以時間對齊;≥2 組 lane 的持續 ≥0.5 s 區段重疊 → HIT,只有 1 組 → SOFT。
    t_onset = 第 2 組 lane 加入的時間(SM 的單次更新跌破 1 不切斷事件,見 MR_BRIDGE_S)。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("ekf_mag_reject")
    k4s = dict(_lane_tables(ctx, "XKF4"))
    if not k4s:
        return _na("ekf_mag_reject", "no_data", "沒有 EKF 狀態紀錄(XKF4),無法檢查磁力計創新量。")
    k3s, kss = dict(_lane_tables(ctx, "XKF3")), dict(_lane_tables(ctx, "XKFS"))
    lanes = [_mag_lane_arrays(ctx, iv, l, k4, kss.get(l), k3s.get(l)) for l, k4 in k4s.items()]
    lanes = [L for L in lanes if L["t"].size]
    if not lanes:
        return _na("ekf_mag_reject", "no_data", "解鎖期間沒有 EKF 磁力計紀錄,無法檢查。")

    lane_runs, lane_first = [], []
    for L in lanes:
        m3 = (L["mf"] == 2) & ((L["sm"] > MR_SM_GATE) | (L["im"] > MR_IM_GATE_MG))
        m1 = (L["mf"] == 1) & (L["iyaw"] > MR_IYAW_GATE_DEG)
        L["runs"] = _runs_bridged(m3 | m1, L["t"], MR_MIN_DUR_S, MR_BRIDGE_S, XKF_MAX_DT_S)
        lane_runs.append([(s, e) for s, e, _, _ in L["runs"]])
        if L["runs"]:
            lane_first.append(L["runs"][0][0])

    need = min(LANE_NEED, len(lanes))
    voted = _vote(lane_runs, need, MR_MIN_OVERLAP_S)

    # ---- 有效資料長度 / 事前統計:只算「該模式的訊號真的有值」的樣本(缺 SM / IM* / IYAW 欄位不能算覆蓋)----
    cov3 = max(float(np.sum((L["mf"] == 2) & (np.isfinite(L["sm"]) | np.isfinite(L["im"]))))
               * _dt_nom(L["t"], 0.04) for L in lanes)
    cov1 = max(float(np.sum((L["mf"] == 1) & np.isfinite(L["iyaw"]))) * _dt_nom(L["t"], 0.04) for L in lanes)

    hit = bool(voted)
    soft = (not hit) and bool(lane_first)
    t_onset = voted[0][0] if hit else (min(lane_first) if soft else None)
    t_tilt = _att_tilt45(ctx, iv)
    # 事前 = 事件起點 / 任何一條 lane 開始超標之前(t_onset 是第 2 條 lane 加入的時間,第 1 條可能更早)
    pre_end = min([x for x in (_event_start(a, t_tilt), min(lane_first) if lane_first else None) if x is not None],
                  default=None)

    def _stat(sel_fn, key, mode):
        vals = []
        for L in lanes:
            m = (L["mf"] == mode) & sel_fn(L["t"]) & np.isfinite(L[key])
            if m.any():
                vals.append(float(np.max(L[key][m])))
        return max(vals) if vals else None

    pre_sel = (lambda t: t < pre_end) if pre_end is not None else (lambda t: np.ones(t.size, dtype=bool))
    vals: dict = {
        "n_lanes": len(lanes), "lanes_needed": need, "lanes_alarmed": len(lane_first),
        "coverage_3d_s": cov3, "coverage_yaw_only_s": cov1,
        "pre_sm_max": _stat(pre_sel, "sm", 2), "pre_im_max_mG": _stat(pre_sel, "im", 2),
        "pre_iyaw_max_deg": _stat(pre_sel, "iyaw", 1),
        "pre_end_t": pre_end,
        "thr_sm": MR_SM_GATE, "thr_im_mG": MR_IM_GATE_MG, "thr_iyaw_deg": MR_IYAW_GATE_DEG,
        "min_dur_s": MR_MIN_DUR_S,
    }

    if t_onset is None:
        if max(cov3, cov1) < MIN_COVER_S:
            return _na("ekf_mag_reject", "insufficient_data",
                       f"解鎖期間 EKF 磁力計有效資料只有 {max(cov3, cov1):.1f} s(< {MIN_COVER_S:.0f} s),不足以判定。", **vals)
        parts = []
        if vals["pre_sm_max"] is not None:
            parts.append(f"飛行中磁力計測試比 SM 最大 {vals['pre_sm_max']:.2f}(門檻 {MR_SM_GATE:.0f})")
        if vals["pre_im_max_mG"] is not None:
            parts.append(f"創新量最大 {vals['pre_im_max_mG']:.0f} mG(門檻 {MR_IM_GATE_MG:.0f})")
        if vals["pre_iyaw_max_deg"] is not None:
            parts.append(f"地面偏航融合 IYAW 最大 {vals['pre_iyaw_max_deg']:.1f}°(門檻 {MR_IYAW_GATE_DEG:.0f}°)")
        if not parts:            # 有覆蓋但事前窗內沒有任何可用數字(例如事件起點在解鎖後立刻發生)→ 不能說 CLEAR
            return _na("ekf_mag_reject", "no_data", "解鎖期間(事件前)沒有可用的 EKF 磁力計創新量數值,無法判定。", **vals)
        return _finding("ekf_mag_reject", CLEAR, SEV_MAG, max(iv.t0, MR_T_MIN_S), None,
                        "clear", vals, f"EKF 沒有拒絕磁力計:{'、'.join(parts)}。可排除磁力計融合失敗。")

    # ---- 警報:區間內的峰值 ----
    t_end = voted[0][1] if hit else max(e for L in lanes for s, e, _, _ in L["runs"][:1])
    alarm_lanes = [L for L in lanes if L["runs"]]
    peak_sm = max((float(np.max(L["sm"][(L["mf"] == 2) & (L["t"] >= t_onset) & np.isfinite(L["sm"])])) for L in alarm_lanes
                   if ((L["mf"] == 2) & (L["t"] >= t_onset) & np.isfinite(L["sm"])).any()), default=None)
    peak_im, peak_ax = None, None
    for L in alarm_lanes:
        m = (L["mf"] == 2) & (L["t"] >= t_onset) & np.isfinite(L["im"])
        if m.any():
            j = int(np.argmax(np.where(m, L["im"], -1.0)))
            if peak_im is None or L["im"][j] > peak_im:
                peak_im, peak_ax = float(L["im"][j]), "XYZ"[int(L["im_ax"][j])]
    peak_iyaw = max((float(np.max(L["iyaw"][(L["mf"] == 1) & (L["t"] >= t_onset) & np.isfinite(L["iyaw"])])) for L in alarm_lanes
                     if ((L["mf"] == 1) & (L["t"] >= t_onset) & np.isfinite(L["iyaw"])).any()), default=None)
    mode_at = "mag3d" if any(((L["mf"] == 2) & (abs(L["t"] - t_onset) < 0.05)).any() for L in alarm_lanes) else "yaw_only"
    c = _causality(a, t_onset, t_tilt)
    run_start_strict = min(L["runs"][0][0] for L in alarm_lanes)   # 未併接的原始 run 起點只供對照
    vals.update(peak_sm=peak_sm, peak_im_mG=peak_im, peak_im_axis=peak_ax, peak_iyaw_deg=peak_iyaw,
                mode_at_alarm=mode_at, t_end=t_end, t_first_lane=min(lane_first), **c)
    t_alarm = min(t_onset + MR_MIN_DUR_S, iv.t1)
    status = HIT if hit else SOFT
    head = f"EKF 磁力計被拒絕:{t_onset:.2f} s 起" + (f" {len(lane_first)}/{len(lanes)} 組 EKF lane 同時" if hit else " 僅 1 組 EKF lane(其餘正常)") + "超標,"
    if mode_at == "mag3d":
        nums = "、".join(x for x in (f"測試比 SM 峰值 {peak_sm:.1f}(>1 = 被拒絕)" if peak_sm is not None else "",
                                     f"創新量 |IM{peak_ax}| 峰值 {peak_im:.0f} mG" if peak_im is not None else "") if x)
    else:
        nums = f"地面偏航創新量 IYAW 峰值 {peak_iyaw:.1f}°" if peak_iyaw is not None else ""
    pre_bits = [x for x in (f"SM 最大 {vals['pre_sm_max']:.2f}(門檻 {MR_SM_GATE:.0f})" if vals["pre_sm_max"] is not None else "",
                            f"創新量最大 {vals['pre_im_max_mG']:.0f} mG(門檻 {MR_IM_GATE_MG:.0f})"
                            if vals["pre_im_max_mG"] is not None else "") if x]
    pre = f"警報前 {'、'.join(pre_bits)},都在正常範圍。" if pre_bits else ""
    text = f"{head}{nums};{_cause_text(c, a)}。{pre}"
    return _finding("ekf_mag_reject", status, SEV_MAG, t_onset, t_alarm,
                    _ctx_label(c) if status == HIT else "single_lane",
                    {**vals, "run_start_strict": run_start_strict}, text)


# ======================================================================================
# 2. compass_raw_consistency
# ======================================================================================

def _attitude_at(ctx: Ctx, tq: np.ndarray):
    """ATT 的 Roll / Pitch / Yaw(度,unwrap 後線性內插)到 tq;附「最近 ATT 樣本在 0.3 s 內」的有效遮罩。"""
    at = ctx.table("ATT")
    if at is None or not all(c in at for c in ("Roll", "Pitch", "Yaw")) or at["t"].size < 3:
        return None
    roll = _unwrap_interp(at["t"], at["Roll"], tq)
    pitch = _unwrap_interp(at["t"], at["Pitch"], tq)
    yaw = _unwrap_interp(at["t"], at["Yaw"], tq)
    ok = _nearest_gap(at["t"], tq) <= 0.3
    return roll, pitch, yaw, ok


def _tilt_comp_heading_deg(b: np.ndarray, roll_deg: np.ndarray, pitch_deg: np.ndarray) -> np.ndarray:
    """機體系磁場 → 傾角補償後的磁航向(度)。ArduPilot Compass::calculate_yaw 同公式。"""
    r, p = np.radians(roll_deg), np.radians(pitch_deg)
    xh = b[:, 0] * np.cos(p) + b[:, 1] * np.sin(r) * np.sin(p) + b[:, 2] * np.cos(r) * np.sin(p)
    yh = b[:, 1] * np.cos(r) - b[:, 2] * np.sin(r)
    return np.degrees(np.arctan2(-yh, xh))


def _throttle_at(ctx: Ctx, t: np.ndarray):
    """油門(MOTB.ThrOut,沒有就 CTUN.ThO)在 MAG 時間 t 的值 → (x, 有效遮罩, 來源);沒有油門 → None。

    MAG 每筆是前一個記錄週期 (t−dt, t] 內羅盤讀數的平均,所以 x 取同一段時間的油門平均
    (t 與 t−dt 兩點內插的平均),不用瞬時值:翻機時 ThrOut 有單點尖峰(124 1258.54 = 0.444)。
    """
    for name, col in (("MOTB", "ThrOut"), ("CTUN", "ThO")):
        tb = ctx.table(name)
        if tb is None or col not in tb:
            continue
        fin = np.isfinite(tb[col])
        if fin.sum() <= 3:
            continue
        tt, v = tb["t"][fin], np.clip(tb[col][fin], 0.0, None)
        dt = _dt_nom(t)
        x = 0.5 * (np.interp(t, tt, v) + np.interp(t - dt, tt, v))
        return x, _nearest_gap(tt, t) <= RC_MOT_THR_MAX_GAP_S, f"{name}.{col}"
    return None


def _robust_lstsq(A: np.ndarray, y: np.ndarray, groups: int, floor: float):
    """最小平方 + 一次離群剔除。A / y 依「分量優先」排列(groups 個分量 × n 個樣本);
    樣本殘差(各分量合成)> 中位數 + max(4·1.4826·MAD, floor) 的樣本第二次不用。回傳 (參數, 採用樣本數)。"""
    n = y.size // groups
    keep = np.ones(n, dtype=bool)
    p = np.zeros(A.shape[1])
    for _ in range(2):
        rows = np.tile(keep, groups)
        p = np.linalg.lstsq(A[rows], y[rows], rcond=None)[0]
        res = np.linalg.norm((y - A @ p).reshape(groups, n), axis=0)
        med = float(np.median(res))
        keep = res <= med + max(4.0 * 1.4826 * float(np.median(np.abs(res - med))), floor)
    return p, int(keep.sum())


def _motor_models(ctx: Ctx, a: Anchors, iv: ArmedInterval, mags: dict[int, dict], cal) -> dict[int, dict]:
    """每顆羅盤的馬達電流模型(見 RC_MOT_* 註解)。回傳 {編號: {'x', 'b', 'c', 'n_fit', ...}}:
    x = MAG 時間軸上的油門(沒有油門 → None);b = 純量斜率(mG / 單位油門,不可辨識 → 0);
    c = 機體系干擾向量(mG / 單位油門,沒有姿態或不可辨識 → None)。

    擬合樣本:解鎖 1 s 後到事件起點(觸地 / 姿態異常 / 傾角 45°;沒有就整段)、非校正窗、|Roll|,|Pitch| < 30°。
    含故障的少量樣本由離群剔除處理(剔除門檻至少 RC_MAG_JUMP_MG)。
    """
    ev0 = _event_start(a, _att_tilt45(ctx, iv))
    fit_end = min(iv.t1, ev0) if ev0 is not None else iv.t1
    out: dict[int, dict] = {}
    for i, tb in mags.items():
        t = tb["t"]
        B = _vec(tb)
        info: dict = {"x": None, "b": 0.0, "c": None, "n_fit": 0, "src": None, "fit_end": fit_end, "reason": "no_throttle"}
        out[i] = info
        th = _throttle_at(ctx, t)
        if th is None:
            continue
        x, ok, src = th
        info.update(x=x, src=src)
        att = _attitude_at(ctx, t)
        sel = ((t >= iv.t0 + RC_MOT_SETTLE_S) & (t <= fit_end) & ok & ~_in_windows(t, cal)
               & np.all(np.isfinite(B), axis=1))
        if att is not None:
            roll, pitch, yaw, ok_att = att
            sel &= ok_att & (np.abs(wrap180(roll)) < RC_MOT_FIT_TILT_DEG) & (np.abs(wrap180(pitch)) < RC_MOT_FIT_TILT_DEG)
        n = int(sel.sum())
        if n < RC_MOT_MIN_FIT_N:
            info["reason"] = "few_samples"
            continue
        xs = x[sel]
        if int(np.sum(np.abs(xs - np.median(xs)) >= RC_MOT_MIN_STEP)) < RC_MOT_MIN_CONTRAST_N:
            info["reason"] = "no_throttle_contrast"
            continue
        mag = np.linalg.norm(B[sel], axis=1)
        p, n_used = _robust_lstsq(np.stack([np.ones(n), xs], axis=1), mag, 1, RC_MAG_JUMP_MG)
        info.update(b=float(p[1]), n_fit=n_used, reason="", mag_mode="scalar")
        if att is not None:
            R = _rot_body_to_ned(roll[sel], pitch[sel], yaw[sel])
            one, zero = np.ones(n), np.zeros(n)
            A = np.concatenate([np.stack([R[0][j], R[1][j], R[2][j]]
                                         + [one if j == k else zero for k in range(3)]
                                         + [xs if j == k else zero for k in range(3)], axis=1) for j in range(3)], axis=0)
            pv, _ = _robust_lstsq(A, np.concatenate([B[sel, j] for j in range(3)]), 3, RC_MAG_JUMP_MG)
            c = pv[6:9].astype(float)
            info["c"] = c
            # 磁場強度用哪個模型:向量模型要讓 |B| 殘差的標準差明顯變小(固定在機體上的干擾,轉向時 |B| 也會變)才用
            sd_s = float(np.std(mag - p[0] - p[1] * xs))
            sd_v = float(np.std(np.linalg.norm(B[sel] - xs[:, None] * c[None, :], axis=1)))
            info.update(sd_scalar=sd_s, sd_vector=sd_v)
            if sd_v < RC_MOT_VEC_GAIN * sd_s:
                info["mag_mode"] = "vector"
    return out


def _compensated(tb: dict, model: dict | None) -> np.ndarray:
    """扣掉馬達磁場 x·c 後的機體系磁場(沒有向量模型 → 原值)。"""
    B = _vec(tb)
    if model is None or model.get("c") is None or model.get("x") is None:
        return B
    return B - model["x"][:, None] * model["c"][None, :]


def _check_runs(dev: np.ndarray, t: np.ndarray, thr: float, valid: np.ndarray):
    """dev > thr(且 valid)持續 ≥ RC_MIN_DUR_S 的 run。"""
    return runs((dev > thr) & valid & np.isfinite(dev), t, RC_MIN_DUR_S - JITTER_TOL_S, MAG_MAX_DT_S)


def _sustained_max(dev: np.ndarray, t: np.ndarray, valid: np.ndarray, dur: float = RC_MIN_DUR_S) -> float | None:
    """持續 ≥ dur 的最大偏離:每個起點取 [t_i, t_i + dur] 內(連續有效、沒有 > MAG_MAX_DT_S 空洞)樣本的最小值,再取最大。
    和警報判定是同一個量(門檻低於它才會有 run);單點尖峰不算。"""
    n = dev.size
    if n == 0:
        return None
    d = np.where(valid & np.isfinite(dev), dev, -np.inf)
    idx = np.arange(n)
    j = np.searchsorted(t, t + dur - JITTER_TOL_S, side="left")
    ok = j < n
    span = np.where(ok, j - idx, 0)
    out = d.copy()
    for k in range(1, min(int(span.max()), 50) + 1):
        need = span >= k
        sh = np.full(n, -np.inf)
        sh[:n - k] = d[k:]
        gap = np.full(n, np.inf)
        gap[:n - k] = t[k:] - t[k - 1:n - 1]
        out = np.where(need, np.minimum(out, sh), out)
        ok &= ~need | (gap <= MAG_MAX_DT_S)
    m = float(np.max(np.where(ok, out, -np.inf)))
    return m if np.isfinite(m) else None


def _finish_check(rr, valid: np.ndarray, dt: float, **extra) -> dict:
    """依 run 與有效樣本數決定這一項的狀態:有 run → alarm;有效資料 ≥ MIN_COVER_S → ok;否則 na。"""
    valid_s = float(valid.sum() * dt)
    return {"state": "alarm" if rr else ("ok" if valid_s >= MIN_COVER_S else "na"), "valid_s": valid_s, **extra}


def _raw_checks(ctx: Ctx, a: Anchors, iv: ArmedInterval):
    """三項原始羅盤檢查(都在扣掉馬達磁場之後做,見 RC_MOT_*)。
    回傳 (checks, used_idx, gated_frac, 馬達模型摘要);checks[k] = {'state': 'alarm'|'ok'|'na', ...}。"""
    mags = _mag_tables(ctx)
    if not mags:
        return {}, None, 0.0, {}
    used = _used_compass(ctx, iv, mags)
    cal = _cal_windows(mags)
    tb = mags[used]
    t = tb["t"]
    arm_idx = np.flatnonzero((t >= iv.t0) & (t <= iv.t1))
    if arm_idx.size == 0:
        return {}, used, 0.0, {}
    ta = t[arm_idx]
    dt = _dt_nom(t)
    gated = _in_windows(ta, cal)
    gated_frac = float(gated.mean())
    # 每個樣本之前最近一次校正窗結束的時間(參考值只能用校正之後的資料)
    floor_all = np.full(t.size, -np.inf)
    if cal:
        ends = np.array([b for _, b in cal])
        k = np.searchsorted(ends, t, side="right") - 1
        floor_all = np.where(k >= 0, ends[np.clip(k, 0, None)], -np.inf)
    # 參考值只用「校正窗之後」且「解鎖 RC_MOT_SETTLE_S 之後」的樣本(馬達未轉的樣本不能當參考)
    floor_a = np.maximum(floor_all[arm_idx], iv.t0 + RC_MOT_SETTLE_S)
    models = _motor_models(ctx, a, iv, mags, cal)
    mm = models.get(used)
    series: dict[str, tuple] = {}        # 各項的 (時間, 偏差量, 有效遮罩, |B|),最後統一算「警報前」統計
    out: dict = {}

    # ---- (a) 磁場強度突變(扣掉 b·油門 之後)----
    mag = np.linalg.norm(_vec(tb), axis=1)
    x = mm["x"] if mm is not None else None
    b = mm["b"] if mm is not None else 0.0          # 純量斜率:也用來判斷 run 是否可由油門變化解釋
    if mm is not None and mm.get("mag_mode") == "vector":
        resid = np.linalg.norm(_compensated(tb, mm), axis=1)
    else:
        resid = mag - b * x if x is not None else mag
    ref = _rolling_ref(t, resid, arm_idx, RC_MAG_REF_WIN_S, RC_MAG_REF_MIN_N, floor_a)
    dev = np.abs(resid[arm_idx] - ref)
    valid = (ta > RC_MAG_T_MIN_S) & ~gated & np.isfinite(ref)
    rr_all = _check_runs(dev, ta, RC_MAG_JUMP_MG, valid)
    xref = _rolling_ref(t, x, arm_idx, RC_MAG_REF_WIN_S, RC_MAG_REF_MIN_N, floor_a) if x is not None else None
    rr, explained = [], []
    excl = np.zeros(ta.size, dtype=bool)          # 歸因馬達電流的 run(前後 RC_MAG_MOTOR_WIN_S)不列入最大偏離統計
    for s, e, i0, i1 in rr_all:
        peak = float(np.max(dev[i0:i1 + 1]))
        motor = 0.0
        if xref is not None and b != 0.0 and np.isfinite(xref[i0]):
            w = (ta >= s - RC_MAG_MOTOR_WIN_S) & (ta <= e + RC_MAG_MOTOR_WIN_S)
            motor = abs(b) * float(np.max(np.abs(x[arm_idx][w] - xref[i0])))
        if motor >= RC_MAG_MOTOR_FRAC * peak:
            explained.append({"t": s, "dur_s": e - s, "peak_dev_mG": peak, "motor_mG": motor})
            excl |= (ta >= s - RC_MAG_MOTOR_WIN_S) & (ta <= e + RC_MAG_MOTOR_WIN_S)
        else:
            rr.append((s, e, i0, i1))
    vstat = valid & ~excl
    chk = _finish_check(rr, valid, dt, thr=RC_MAG_JUMP_MG, compass=used,
                        max_dev_mG=float(np.max(dev[vstat])) if vstat.any() else None,
                        max_sustained_dev_mG=_sustained_max(dev, ta, vstat),
                        motor_explained=explained[:10], n_motor_explained=len(explained),
                        motor_slope_mG=b if x is not None else None,
                        motor_model=(mm.get("mag_mode") if mm is not None and x is not None and b != 0.0 else None))
    if rr:
        i0, i1 = rr[0][2], rr[0][3]
        chk.update(t_first=rr[0][0], dur_s=rr[0][1] - rr[0][0], peak_dev_mG=float(np.max(dev[i0:i1 + 1])),
                   min_mag_mG=float(np.min(mag[arm_idx][i0:i1 + 1])))
    out["magnitude"] = chk
    series["magnitude"] = (ta, dev, vstat, mag[arm_idx])

    # ---- (b) 雙羅盤方向(編號最小的兩顆;各自的 COMPASS_USEn / ORIENTn 要成立且朝向相同)----
    ids = sorted(mags)[:2]
    sfx = {0: "", 1: "2", 2: "3"}
    if len(ids) < 2:
        out["pair"] = {"state": "na", "reason": "single_compass"}
    elif not ctx.param("COMPASS_USE" + sfx.get(ids[1], ""), 1.0):
        out["pair"] = {"state": "na", "reason": "backup_disabled"}
    elif ctx.param("COMPASS_ORIENT" + sfx.get(ids[0], ""), 0.0) != ctx.param("COMPASS_ORIENT" + sfx.get(ids[1], ""), 0.0):
        out["pair"] = {"state": "na", "reason": "orient_differs"}
    else:
        b0, b1 = _compensated(mags[ids[0]], models.get(ids[0])), _compensated(mags[ids[1]], models.get(ids[1]))
        t0m, t1m = mags[ids[0]]["t"], mags[ids[1]]["t"]
        if t0m.size == t1m.size and np.max(np.abs(t0m - t1m)) < 0.005:
            b1i, ok = b1, np.ones(t0m.size, dtype=bool)
        else:
            b1i = np.stack([np.interp(t0m, t1m, b1[:, k]) for k in range(3)], axis=1)
            ok = _nearest_gap(t1m, t0m) <= 0.15
        n0, n1 = np.linalg.norm(b0, axis=1), np.linalg.norm(b1i, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosang = np.sum(b0 * b1i, axis=1) / (n0 * n1)
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        pi = np.flatnonzero((t0m >= iv.t0) & (t0m <= iv.t1))
        tp = t0m[pi]
        valid = (tp > RC_PAIR_T_MIN_S) & ~_in_windows(tp, cal) & ok[pi] & np.isfinite(ang[pi])
        rr = _check_runs(ang[pi], tp, RC_PAIR_DEG, valid)
        chk = _finish_check(rr, valid, _dt_nom(t0m), thr=RC_PAIR_DEG, compasses=ids,
                            max_deg=float(np.max(ang[pi][valid])) if valid.any() else None,
                            max_sustained_deg=_sustained_max(ang[pi], tp, valid),
                            median_deg=float(np.median(ang[pi][valid])) if valid.any() else None,
                            motor_compensated=[i for i in ids if (models.get(i) or {}).get("c") is not None])
        if rr:
            i0, i1 = rr[0][2], rr[0][3]
            chk.update(t_first=rr[0][0], dur_s=rr[0][1] - rr[0][0], peak_deg=float(np.max(ang[pi][i0:i1 + 1])))
        out["pair"] = chk
        series["pair"] = (tp, ang[pi], valid, None)

    # ---- (c) EKF 航向 vs 原始羅盤航向 ----
    att = _attitude_at(ctx, t)
    if att is None:
        out["heading"] = {"state": "na", "reason": "no_att"}
    else:
        roll, pitch, yaw, ok = att
        res = wrap180(yaw - _tilt_comp_heading_deg(_compensated(tb, mm), roll, pitch))   # 常數 ≈ 磁偏角(此場地 −4.7°)
        refr = _rolling_ref(t, res, arm_idx, RC_HEAD_REF_WIN_S, RC_HEAD_REF_MIN_N, floor_a, circular=True)
        dev = np.abs(wrap180(res[arm_idx] - refr))
        valid = (ta > RC_HEAD_T_MIN_S) & ~gated & ok[arm_idx] & np.isfinite(refr)
        rr = _check_runs(dev, ta, RC_HEAD_DEG, valid)
        chk = _finish_check(rr, valid, dt, thr=RC_HEAD_DEG, max_dev_deg=float(np.max(dev[valid])) if valid.any() else None,
                            max_sustained_dev_deg=_sustained_max(dev, ta, valid),
                            motor_compensated=bool(mm is not None and mm.get("c") is not None))
        if rr:
            i0, i1 = rr[0][2], rr[0][3]
            chk.update(t_first=rr[0][0], dur_s=rr[0][1] - rr[0][0], peak_dev_deg=float(np.max(dev[i0:i1 + 1])))
        out["heading"] = chk
        series["heading"] = (ta, dev, valid, None)

    # ---- 「警報前」統計:事件起點(觸地 / 姿態異常)與第一個警報,取較早者 ----
    firsts = [c["t_first"] for c in out.values() if c.get("state") == "alarm"]
    ends = [x for x in (_event_start(a, _att_tilt45(ctx, iv)), min(firsts) if firsts else None) if x is not None]
    pre_end = min(ends) if ends else np.inf
    for k, (tt, dev_k, valid_k, mag_k) in series.items():
        m = valid_k & (tt < pre_end) & np.isfinite(dev_k)
        key = {"magnitude": "pre_max_dev_mG", "pair": "pre_max_deg", "heading": "pre_max_dev_deg"}[k]
        out[k][key] = float(np.max(dev_k[m])) if m.any() else None
        if k == "magnitude":
            out[k]["pre_mag_range_mG"] = [float(mag_k[m].min()), float(mag_k[m].max())] if m.any() else None
    motor = {str(i): {"slope_mG": md["b"] if md["x"] is not None else None,
                      "vector_mG": float(np.linalg.norm(md["c"])) if md["c"] is not None else None,
                      "mag_mode": md.get("mag_mode"), "sd_scalar_mG": md.get("sd_scalar"),
                      "sd_vector_mG": md.get("sd_vector"),
                      "n_fit": md["n_fit"], "throttle": md["src"], "reason": md["reason"]}
             for i, md in models.items()}
    return out, used, gated_frac, motor


@guarded("compass_raw_consistency", DOMAIN)
def compass_raw_consistency(ctx: Ctx, a: Anchors):
    """原始羅盤一致性:(a) |B| 磁場強度突變 (b) 兩顆羅盤方向夾角 (c) EKF 航向 vs 羅盤原始航向。

    任一項成立 → HIT(t_onset = 最早的那項的 run 起點);都沒有且至少一項可判 → CLEAR;都不可判 → NA。
    三項都先扣掉馬達電流的磁場(油門模型,見 RC_MOT_*);補償後仍超標、但油門變化足以解釋的磁場強度 run
    (起降 / 爬升的瞬間)歸因馬達電流,列在 checks['magnitude']['motor_explained'],不算警報。
    閘門:校正窗 [t_OfsChange−120, +5] 排除;開機 35 s(航向 12 s、雙羅盤 10 s)內不看;只看解鎖區間;
    參考值只用解鎖 1 s 之後的樣本。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("compass_raw_consistency")
    checks, used, gated_frac, motor = _raw_checks(ctx, a, iv)
    if not checks:
        return _na("compass_raw_consistency", "no_data", "沒有可用的羅盤(MAG)紀錄,無法檢查原始羅盤一致性。")
    if gated_frac > CAL_MAX_GATED_FRAC:
        return _na("compass_raw_consistency", "calibration_window",
                   f"解鎖期間有 {gated_frac * 100:.0f}% 落在羅盤校正窗內,不做一致性判定。", gated_frac=gated_frac)
    states = {k: v["state"] for k, v in checks.items()}
    alarms = sorted((v["t_first"], k) for k, v in checks.items() if v["state"] == "alarm")
    vals: dict = {"checks": checks, "used_compass": used, "gated_frac": gated_frac, "motor_model": motor,
                  "thr_mag_mG": RC_MAG_JUMP_MG, "thr_pair_deg": RC_PAIR_DEG, "thr_heading_deg": RC_HEAD_DEG,
                  "min_dur_s": RC_MIN_DUR_S, "states": states}
    if not alarms and not any(s == "ok" for s in states.values()):
        return _na("compass_raw_consistency", "insufficient_data",
                   "有效的羅盤樣本不足(開機初期 / 校正窗 / 缺姿態資料),無法判定一致性。", **vals)
    m, p, h = checks.get("magnitude", {}), checks.get("pair", {}), checks.get("heading", {})
    n_mot = m.get("n_motor_explained") or 0
    mot_txt = (f"(另有 {n_mot} 段磁場強度變化發生在油門變化時,大小與馬達電流模型相符,不算異常)"
               if n_mot else "")
    if not alarms:
        bits = []
        if m.get("max_sustained_dev_mG") is not None:
            bits.append(f"磁場強度偏離最大 {m['max_sustained_dev_mG']:.0f} mG(門檻 {RC_MAG_JUMP_MG:.0f}){mot_txt}")
        if p.get("max_sustained_deg") is not None:
            bits.append(f"雙羅盤夾角最大 {p['max_sustained_deg']:.1f}°(門檻 {RC_PAIR_DEG:.0f}°)")
        if h.get("max_sustained_dev_deg") is not None:
            bits.append(f"EKF 航向 vs 羅盤航向偏離最大 {h['max_sustained_dev_deg']:.1f}°(門檻 {RC_HEAD_DEG:.0f}°)")
        comp = "已扣除馬達電流的磁場、" if any(v.get("slope_mG") for v in motor.values()) else ""
        return _finding("compass_raw_consistency", CLEAR, SEV_RAW, iv.t0, None, "clear", vals,
                        f"原始羅盤一致({comp}持續 ≥{RC_MIN_DUR_S:.1f} s 的最大值):" + "、".join(bits)
                        + "。可排除羅盤干擾 / 航向被拉走。")
    t_first, k_first = alarms[0]
    c = _causality(a, t_first, _att_tilt45(ctx, iv))
    vals.update(c)
    vals["first_alarm"] = k_first
    vals["alarm_times"] = {k: checks[k]["t_first"] for _, k in alarms}
    parts = []
    for tt, k in alarms:
        v = checks[k]
        if k == "magnitude":
            parts.append(f"磁場強度 {tt:.2f} s 起偏離 {v['peak_dev_mG']:.0f} mG")
        elif k == "pair":
            parts.append(f"雙羅盤夾角 {tt:.2f} s 起達 {v['peak_deg']:.0f}°")
        else:
            parts.append(f"EKF 航向 {tt:.2f} s 起偏離羅盤航向 {v['peak_dev_deg']:.0f}°")
    pm = m.get("pre_mag_range_mG")
    pre = ""
    if pm and pm[1] - pm[0] <= 2.0 * RC_MAG_JUMP_MG:
        pre = f"警報前 |B| 穩定在 {pm[0]:.0f}~{pm[1]:.0f} mG。"
    elif pm and m.get("pre_max_dev_mG") is not None:
        pre = (f"警報前 |B| 隨油門(馬達電流)在 {pm[0]:.0f}~{pm[1]:.0f} mG 之間變化,"
               f"扣除馬達電流後最大偏離 {m['pre_max_dev_mG']:.0f} mG。")
    return _finding("compass_raw_consistency", HIT, SEV_RAW, t_first, t_first + RC_MIN_DUR_S, _ctx_label(c), vals,
                    f"羅盤原始資料警報:{';'.join(parts)};{_cause_text(c, a)}。{pre}")


# ======================================================================================
# 3. compass_motor_interference
# ======================================================================================

def _rot_body_to_ned(roll_deg, pitch_deg, yaw_deg):
    """R = Rz(yaw)·Ry(pitch)·Rx(roll) 的九個元素(機體 → NED)。"""
    r, p, y = np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return ((cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy),
            (cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy),
            (-sp, sr * cp, cr * cp))


def _to_ned(b: np.ndarray, roll, pitch, yaw) -> np.ndarray:
    R = _rot_body_to_ned(roll, pitch, yaw)
    return np.stack([R[i][0] * b[:, 0] + R[i][1] * b[:, 1] + R[i][2] * b[:, 2] for i in range(3)], axis=1)


def _interference_one(ctx: Ctx, iv: ArmedInterval, mags: dict[int, dict], t_cap: float | None):
    """單一解鎖區間的各羅盤馬達干擾量。回傳 (dict compass→結果, 說明) ;資料不足 → ({}, reason)。"""
    # 以編號最小的羅盤時間軸為準(MAG 各 instance 時間戳相同);其他羅盤內插
    base = mags[min(mags)]
    t = base["t"]
    att = _attitude_at(ctx, t)
    if att is None:
        return {}, "no_att"
    roll, pitch, yaw, ok_att = att
    # 油門:MOTB.ThrOut,沒有就 CTUN.ThO
    thr = None
    for name, col in (("MOTB", "ThrOut"), ("CTUN", "ThO")):
        tbl = ctx.table(name)
        if tbl is not None and col in tbl and tbl["t"].size > 3:
            thr = np.interp(t, tbl["t"], tbl[col])
            thr_ok = _nearest_gap(tbl["t"], t) <= 0.3
            break
    if thr is None:
        return {}, "no_throttle"
    other_armed = np.zeros(t.size, dtype=bool)
    for o in ctx.armed:
        if o.index != iv.index:
            other_armed |= (t >= o.t0) & (t <= o.t1)
    g_sel = (t >= iv.t0 - MI_GROUND_WIN_S[0]) & (t <= iv.t0 - MI_GROUND_WIN_S[1]) & ~other_armed & ok_att
    f_hi = iv.t1 - MI_FLIGHT_T1_S
    if t_cap is not None:
        f_hi = min(f_hi, t_cap - MI_WARN_AT_WARN_S)
    wl = wrap180(roll)
    wp = wrap180(pitch)
    f_sel = ((t >= iv.t0 + MI_FLIGHT_T0_S) & (t <= f_hi) & (thr >= MI_THR_MIN) & thr_ok & ok_att
             & (np.abs(wl) < MI_LEVEL_DEG) & (np.abs(wp) < MI_LEVEL_DEG))
    dt = _dt_nom(t)
    if g_sel.sum() * dt < MI_MIN_SPAN_S:
        return {}, "no_ground_reference"
    if f_sel.sum() * dt < MI_MIN_SPAN_S:
        return {}, "no_motor_flight"
    res = {}
    for i, tb in mags.items():
        if i == min(mags):
            b = _vec(tb)
        else:
            if tb["t"].size == t.size and np.max(np.abs(tb["t"] - t)) < 0.005:
                b = _vec(tb)
            else:
                bb = _vec(tb)
                b = np.stack([np.interp(t, tb["t"], bb[:, k]) for k in range(3)], axis=1)
        ned = _to_ned(b, roll, pitch, yaw)
        d = np.median(ned[f_sel], axis=0) - np.median(ned[g_sel], axis=0)
        mag = np.linalg.norm(b, axis=1)
        gm = float(np.median(mag[g_sel]))
        res[i] = {"delta_mG": float(np.linalg.norm(d)), "ratio": float(np.linalg.norm(d) / gm) if gm > 0 else None,
                  "dB_scalar_mG": float(np.median(mag[f_sel]) - gm),
                  "dB_scalar_pct": float((np.median(mag[f_sel]) - gm) / gm * 100.0) if gm > 0 else None,
                  "ground_mag_mG": gm}
    res["_n"] = {"ground_s": float(g_sel.sum() * dt), "flight_s": float(f_sel.sum() * dt),
                 "t_flight_first": float(t[f_sel][0]), "t_flight_last": float(t[f_sel][-1])}
    return res, ""


@guarded("compass_motor_interference", DOMAIN)
def compass_motor_interference(ctx: Ctx, a: Anchors):
    """馬達電流造成的羅盤干擾(健康度指標,不是肇因)。

    每個羅盤:delta = 飛行(ThrOut ≥0.1、水平)與地面(馬達未轉)在 NED 系的磁場中位數差(與航向無關);
    ratio = |delta| / 地面 |B|。status 只反映 EKF 實際使用的那顆:> 0.60 HIT、> 0.30 SOFT、否則 CLEAR;
    備援羅盤的干擾放在 values['backup'] / hygiene(不影響 status,避免被誤算成羅盤肇因)。
    不使用 BAT.Curr(此機型的電流計無效)。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("compass_motor_interference")
    mags = _mag_tables(ctx)
    if not mags:
        return _na("compass_motor_interference", "no_data", "沒有羅盤(MAG)紀錄,無法估計馬達干擾。")
    used = _used_compass(ctx, iv, mags)
    res, why = _interference_one(ctx, iv, mags, a.t_warn)
    if not res:
        txt = {"no_att": "缺姿態資料", "no_throttle": "缺油門資料", "no_ground_reference": "解鎖前沒有 ≥20 s 的地面(馬達未轉)參考",
               "no_motor_flight": "解鎖後沒有 ≥20 s 馬達運轉的水平飛行"}.get(why, why)
        return _na("compass_motor_interference", why, f"無法估計馬達對羅盤的干擾({txt})。", used_compass=used)
    n = res.pop("_n")
    # 其他解鎖區間(飛行 1 等)也算,放 segments 供報告對照
    segs = []
    for o in ctx.armed:
        if o.index == iv.index:
            continue
        r2, _ = _interference_one(ctx, o, mags, None)
        if r2:
            r2.pop("_n", None)
            segs.append({"index": o.index, "t0": o.t0, "t1": o.t1,
                         "ratios": {str(k): v["ratio"] for k, v in r2.items()}})
    u = res.get(used)
    if u is None or u["ratio"] is None:
        return _na("compass_motor_interference", "no_data", "地面磁場強度為 0,無法估計馬達干擾比例。", used_compass=used)
    backups = {i: v for i, v in res.items() if i != used and v["ratio"] is not None}
    worst_b = max(backups.items(), key=lambda kv: kv[1]["ratio"]) if backups else None
    ratio = u["ratio"]
    status = HIT if ratio > MI_RATIO_BAD else (SOFT if ratio > MI_RATIO_WARN else CLEAR)

    def _level(r: float) -> str:
        return "差" if r > MI_RATIO_BAD else ("邊際" if r > MI_RATIO_WARN else "良好")
    hygiene = bool(worst_b and worst_b[1]["ratio"] > MI_RATIO_WARN and status != HIT)
    vals = {"used_compass": used, "used_ratio": ratio, "used_delta_mG": u["delta_mG"],
            "used_dB_scalar_mG": u["dB_scalar_mG"], "used_dB_scalar_pct": u["dB_scalar_pct"], "level": _level(ratio),
            "compasses": {str(i): v for i, v in res.items()},
            "backup_compass": worst_b[0] if worst_b else None,
            "backup_ratio": worst_b[1]["ratio"] if worst_b else None,
            "backup_delta_mG": worst_b[1]["delta_mG"] if worst_b else None,
            "backup_dB_scalar_mG": worst_b[1]["dB_scalar_mG"] if worst_b else None,
            "backup_level": _level(worst_b[1]["ratio"]) if worst_b else None,
            "hygiene": hygiene, "segments": segs, "ground_s": n["ground_s"], "flight_s": n["flight_s"],
            "flight_window": [n["t_flight_first"], n["t_flight_last"]],
            "thr_warn": MI_RATIO_WARN, "thr_bad": MI_RATIO_BAD}
    text = (f"起飛後馬達電流使 EKF 使用的羅盤 MAG[{used}] 磁場改變 {u['delta_mG']:.0f} mG"
            f"(約 {ratio * 100:.0f}% 地磁),屬{_level(ratio)}")
    if worst_b:
        wb = worst_b[1]
        text += f";未被 EKF 使用的羅盤 MAG[{worst_b[0]}] 改變 {wb['delta_mG']:.0f} mG({wb['ratio'] * 100:.0f}%,{_level(wb['ratio'])})"
        if wb["ratio"] > MI_RATIO_WARN:
            text += ",主羅盤失效切到它時航向可能偏差,建議移離電源線與電調或執行 compassmot"
    if status == CLEAR:
        # 「不是失事原因」只在真的有失事(失事類結束 / 有姿態失控錨點)時才說:正常降落的健康觀察不得有失事字眼
        crash = a.end_type in CAUSALITY_END_TYPES or a.t_warn is not None
        text += "。這是健康度指標;EKF 使用的羅盤干擾小" + (",不是失事原因。" if crash else "。")
    else:
        text += "。EKF 使用的羅盤干擾偏高,可能影響航向估計,建議移離電源線與電調或執行 compassmot。"
    return _finding("compass_motor_interference", status, SEV_MOTOR, n["t_flight_first"], None,
                    "hygiene" if hygiene else "health", vals, text)


# ======================================================================================
# 4. ekf_yaw_reset_lane_switch
# ======================================================================================

def _yaw_jump(xk1: dict, t_ev: float) -> float | None:
    """EKF 航向重置的跳變(度):EV 62 時間附近(±YR_STEP_TOL_S)相鄰兩個 XKF1 樣本的差,扣掉正常轉動
    (轉動率 = 重置前 (t−0.5, t−0.1] 相鄰樣本轉動率的中位數),取絕對值最大的一對。航向先 unwrap(跨 0/360 不跳)。

    不用「前後兩窗平均相減」(catalogue 的寫法):飛手打舵時兩窗相隔 0.6 s,40°/s 就量出 24° 的假跳變(合成測試);
    124 的 882.49 重置後 0.1 s 開始轉向,兩窗平均得 +1.2°,實際階躍是 −0.14°(87.60 → 87.46);344.89 的 +1.50° 兩法一致。
    """
    t, y = xk1["t"], xk1["Yaw"]
    m = (t >= t_ev - YR_JUMP_WIN_S[1]) & (t <= t_ev + YR_JUMP_WIN_S[1]) & np.isfinite(y)
    tt = t[m]
    if tt.size < 4:
        return None
    yy = np.degrees(np.unwrap(np.radians(y[m])))
    d, dt = np.diff(yy), np.diff(tt)
    pre = (tt[:-1] >= t_ev - YR_JUMP_WIN_S[1]) & (tt[1:] <= t_ev - YR_JUMP_WIN_S[0]) & (dt > 0)
    cand = (tt[1:] >= t_ev - YR_STEP_TOL_S) & (tt[:-1] <= t_ev + YR_STEP_TOL_S)
    if pre.sum() < 2 or not cand.any():
        return None
    rate = float(np.median(d[pre] / dt[pre]))
    steps = d[cand] - rate * dt[cand]
    return float(wrap180(steps[int(np.argmax(np.abs(steps)))]))


def _lane_yaw_spread(ctx: Ctx, lo: float, hi: float):
    """XKF1 各 lane 的航向,時間對齊到第一條 lane 後的最大成對差 → (t, spread_deg);lane < 2 → None。"""
    lanes = _lane_tables(ctx, "XKF1")
    lanes = [(i, tb) for i, tb in lanes if "Yaw" in tb]
    if len(lanes) < 2:
        return None
    t0 = lanes[0][1]["t"]
    m = (t0 >= lo) & (t0 <= hi)
    t = t0[m]
    if t.size < 3:
        return None
    ys = [wrap180(_unwrap_interp(tb["t"], tb["Yaw"], t)) for _, tb in lanes]
    ok = np.ones(t.size, dtype=bool)
    for _, tb in lanes[1:]:
        ok &= _nearest_gap(tb["t"], t) <= 0.15
    sp = np.zeros(t.size)
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            sp = np.maximum(sp, np.abs(wrap180(ys[i] - ys[j])))
    return t, np.where(ok, sp, np.nan)


@guarded("ekf_yaw_reset_lane_switch", DOMAIN)
def ekf_yaw_reset_lane_switch(ctx: Ctx, a: Anchors):
    """飛行中 EKF 航向重置 / lane 切換 / 來源組切換。

    - 航向重置:EV 62;開機 10 s 內(初始對準)與解鎖後 15 s 內且跳變 ≤ 10°(空中對準)是良性
      (解鎖後 15 s 內但缺 XKF1 量不到跳變 → 仍視為空中對準,不當證據);
      其餘量跳變(所有 XKF1 lane 取 |跳變| 最大者,重置可能發生在非 lane 0 的 primary),|跳變| > 10° → HIT;
      跳變算不出來 → SOFT。
    - 沒有 EV 表(看不到航向重置與來源組切換)→ 不能說 CLEAR:沒有其他證據時回 NA。
    - lane 切換:MSG 'EKF3 lane switch n' 或 ERR Subsys 24,發生在解鎖期間 → SOFT。
    - 來源組切換:EV 85/86/87;切換前後 EK3_SRCn_YAW 不同 → HIT(本機三組都是羅盤,永遠相同)。
    - lane 間航向差 > 10° 持續 ≥ 0.5 s(時間對齊後)→ SOFT。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("ekf_yaw_reset_lane_switch")
    t_ev, ids = ctx.ev_rows()
    has_ev = t_ev.size > 0
    x1s = [(l, tb) for l, tb in _lane_tables(ctx, "XKF1") if "Yaw" in tb]
    x1 = x1s[0][1] if x1s else None
    at = ctx.table("ATT")
    cover = 0.0
    for tb in (x1, at):
        if tb is not None and "Yaw" in tb:
            m = (tb["t"] >= iv.t0) & (tb["t"] <= iv.t1) & np.isfinite(tb["Yaw"])
            cover = max(cover, float(m.sum()) * _dt_nom(tb["t"], 0.04))
    anchor = a.t_warn if a.t_warn is not None else iv.t1
    evid: list[dict] = []        # 證據:{kind, t, level, ...}
    resets: list[dict] = []
    # ---- 航向重置 ----
    for tt in [float(x) for x, i in zip(t_ev, ids) if i == EV_YAW_RESET]:
        r = {"t": tt}
        if tt < YR_T_MIN_S:
            r["kind"] = "initial_alignment"
        elif not (iv.t0 <= tt <= iv.t1):
            r["kind"] = "outside_interval"
        else:
            by_lane = {str(l): _yaw_jump(tb, tt) for l, tb in x1s}
            known = [(abs(v), v, l) for l, v in by_lane.items() if v is not None and np.isfinite(v)]
            jump = max(known)[1] if known else None
            r["jump_deg"] = jump
            r["jump_lane"] = max(known)[2] if known else None
            r["jump_by_lane"] = by_lane
            near_arm = tt - iv.t0 <= YR_ALIGN_AFTER_ARM_S
            if jump is None and near_arm:
                # 解鎖後 15 s 內的重置 = 每次起飛的空中偏航對準(MAG_FUSION 1→2,124 / baselines 跳變 0.9-1.6°),
                # 量不到跳變(缺 XKF1)也不該每次飛行都變成 SOFT 證據
                r["kind"] = "inflight_alignment_unmeasured"
            elif jump is None:
                r["kind"] = "inflight_unknown_jump"
                evid.append({"kind": "yaw_reset", "t": tt, "level": SOFT, "jump_deg": None})
            elif abs(jump) > YR_JUMP_DEG:
                r["kind"] = "inflight_large_jump_after_arm" if near_arm else "inflight_large_jump"
                evid.append({"kind": "yaw_reset", "t": tt, "level": SOFT if near_arm else HIT, "jump_deg": jump})
            else:
                r["kind"] = "inflight_alignment" if near_arm else "inflight_small_jump"
        resets.append(r)
    before = [r for r in resets if r["t"] < anchor]
    last_r = max(before, key=lambda r: r["t"]) if before else None
    last_reset = last_r["t"] if last_r else None
    # ---- lane 切換 ----
    lane_sw: list[dict] = []
    for tt, tx in ctx.messages:
        mm = re.search(r"EKF\d lane switch (\d)", tx)
        if mm and iv.t0 <= tt <= iv.t1:
            lane_sw.append({"t": float(tt), "lane": int(mm.group(1))})
    te, se, ce = ctx.err_rows()
    for tt, s, c in zip(te, se, ce):
        if s == ERR_SUBSYS_EKF_PRIMARY and iv.t0 <= tt <= iv.t1 and not any(abs(x["t"] - tt) < 0.5 for x in lane_sw):
            lane_sw.append({"t": float(tt), "lane": int(c)})
    for x in lane_sw:
        evid.append({"kind": "lane_switch", "t": x["t"], "level": SOFT, "lane": x["lane"]})
    # ---- 來源組切換是否改變航向來源 ----
    yaw_codes = [ctx.param(f"EK3_SRC{n}_YAW") for n in (1, 2, 3)]
    sw = sorted((float(x), EV_SRC_SET[int(i)]) for x, i in zip(t_ev, ids) if int(i) in EV_SRC_SET)
    prev_set = 1
    ks0 = next((tb for _, tb in _lane_tables(ctx, "XKFS")), None)
    if ks0 is not None and "SS" in ks0:
        j = int(np.searchsorted(ks0["t"], iv.t0, side="right")) - 1
        if j >= 0 and np.isfinite(ks0["SS"][j]) and 0 <= ks0["SS"][j] <= 2:
            prev_set = int(round(float(ks0["SS"][j]))) + 1
    srcsw = []
    for tt, s in sw:
        if tt < iv.t0:
            prev_set = s
            continue
        if tt > iv.t1:
            break
        ya, yb = yaw_codes[prev_set - 1], yaw_codes[s - 1]
        changed = (ya is not None and yb is not None and ya != yb)
        srcsw.append({"t": tt, "from_set": prev_set, "to_set": s, "yaw_changed": bool(changed)})
        if changed:
            evid.append({"kind": "yaw_source_changed", "t": tt, "level": HIT, "from_set": prev_set, "to_set": s})
        prev_set = s
    # ---- lane 間航向差 ----
    if a.t_warn is not None:
        lo, hi = max(iv.t0 + YR_SPREAD_T0_S, a.t_warn - YR_SPREAD_PRE_S), a.t_warn - 1.0
    else:
        lo, hi = iv.t0 + YR_SPREAD_T0_S, iv.t1 - 5.0
    spread_max, spread_t = None, None
    sp = _lane_yaw_spread(ctx, lo, hi) if hi > lo else None
    if sp is not None:
        ts, sv = sp
        if np.isfinite(sv).any():
            spread_max = float(np.nanmax(sv))
            spread_t = float(ts[int(np.nanargmax(sv))])
        rr = runs(np.nan_to_num(sv, nan=0.0) > YR_SPREAD_DEG, ts, YR_SPREAD_MIN_DUR_S - JITTER_TOL_S, XKF_MAX_DT_S)
        if rr:
            evid.append({"kind": "lane_yaw_spread", "t": rr[0][0], "level": SOFT, "spread_deg": float(np.max(sv[rr[0][2]:rr[0][3] + 1]))})
    pre30 = [e["kind"] for e in evid if anchor - 30.0 <= e["t"] <= anchor] if a.t_warn is not None else []
    vals: dict = {"resets": resets, "last_reset_t": last_reset, "last_reset_kind": last_r["kind"] if last_r else None,
                  "since_last_reset_s": (anchor - last_reset) if last_reset is not None else None,
                  "events_in_pre30s": pre30, "lane_yaw_spread_t": spread_t,
                  "lane_switches": lane_sw, "source_switches": srcsw, "yaw_source_codes": yaw_codes,
                  "lane_yaw_spread_max_deg": spread_max, "lane_yaw_spread_window": [lo, hi],
                  "evidence": evid, "thr_jump_deg": YR_JUMP_DEG, "thr_spread_deg": YR_SPREAD_DEG,
                  "anchor_t": anchor, "has_ev": has_ev}
    if not evid:
        if cover < MIN_COVER_S:
            return _na("ekf_yaw_reset_lane_switch", "insufficient_data",
                       "解鎖期間沒有足夠的 EKF 航向資料(XKF1 / ATT),無法判定航向重置。", **vals)
        if not has_ev:
            seen = []
            if spread_max is not None:
                seen.append(f"EKF 各 lane 航向差最大 {spread_max:.1f}°")
            seen.append("沒有 lane 切換訊息")
            return _na("ekf_yaw_reset_lane_switch", "no_ev",
                       f"log 沒有 EV 事件紀錄,看不到 EKF 航向重置與來源組切換,無法排除航向重置({'、'.join(seen)})。", **vals)
        kind_zh = {"initial_alignment": "開機初始對準", "inflight_alignment": "起飛後空中對準", "outside_interval": "地面",
                   "inflight_small_jump": "飛行中小幅調整",
                   "inflight_alignment_unmeasured": "起飛後空中對準,缺 XKF1 未量跳變"}.get(last_r["kind"] if last_r else "", "")
        ago = (f"最近一次航向重置在 {anchor - last_reset:.0f} s 前({last_reset:.1f} s,{kind_zh})"
               if last_reset is not None else "整段沒有航向重置")
        n_sw = len(srcsw)
        codes = "/".join(str(int(c)) if c is not None else "?" for c in yaw_codes)
        sw_txt = f";{n_sw} 次來源組切換,但切換前後的航向來源相同(EK3_SRC1/2/3_YAW={codes}),不會改變航向來源" if n_sw else ""
        sp_txt = f";EKF 各 lane 航向差最大 {spread_max:.1f}°" if spread_max is not None else ""
        return _finding("ekf_yaw_reset_lane_switch", CLEAR, SEV_RESET, iv.t0, None, "clear", vals,
                        f"飛行中沒有異常的 EKF 航向重置 / lane 切換:{ago}{sw_txt}{sp_txt}。可排除航向重置造成失控。")
    hits = [e for e in evid if e["level"] == HIT]
    use = hits if hits else evid
    first = min(use, key=lambda e: e["t"])
    status = HIT if hits else SOFT
    t_tilt = _att_tilt45(ctx, iv)
    c = _causality(a, first["t"], t_tilt)
    vals.update(c)
    vals["first_kind"] = first["kind"]
    names = {"yaw_reset": "EKF 航向重置", "lane_switch": "EKF lane 切換", "yaw_source_changed": "來源組切換改變了航向來源",
             "lane_yaw_spread": "EKF 各 lane 航向差過大"}
    desc = []
    for e in sorted(evid, key=lambda e: e["t"]):
        d = names[e["kind"]] + f"({e['t']:.2f} s"
        if e.get("jump_deg") is not None:
            d += f",跳變 {e['jump_deg']:+.1f}°"
        if e.get("spread_deg") is not None:
            d += f",差 {e['spread_deg']:.1f}°"
        desc.append(d + ")")
    text = f"飛行中偵測到:{'、'.join(desc)};{_cause_text(c, a)}。"
    return _finding("ekf_yaw_reset_lane_switch", status, SEV_RESET, first["t"], first["t"], _ctx_label(c), vals, text)


# ======================================================================================
# 5. compass_sensor_health
# ======================================================================================

@guarded("compass_sensor_health", DOMAIN)
def compass_sensor_health(ctx: Ctx, a: Anchors):
    """羅盤感測器健康:Health==0 ≥0.5 s;ERR COMPASS ECode 1/4;MAG 取樣空洞 >2 s;XKF4.TS 磁力計逾時位元;XKF4.FS≠0。

    EKF 使用的羅盤(XKFS.MI)出問題 → HIT;備援羅盤出問題 → SOFT。只看解鎖區間(地面事件放 values['ground_events'])。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("compass_sensor_health")
    mags = _mag_tables(ctx)
    k4s = _lane_tables(ctx, "XKF4")
    if not mags and not k4s:
        return _na("compass_sensor_health", "no_data", "沒有羅盤(MAG)或 EKF(XKF4)紀錄,無法檢查感測器健康。")
    used = _used_compass(ctx, iv, mags) if mags else None
    alarms: list[dict] = []
    cover = 0.0
    gaps_max = {}
    has_health = False
    # log 在解鎖前就已經在記錄(ATT / XKF4 有解鎖前 > SH_GAP_S 的樣本)?LOG_DISARMED=0 的 log 從解鎖才開始,
    # 啟動訊息先寫,MAG 晚幾秒才出現是正常的 → 那種 log 不把「解鎖到第一筆 MAG」當成資料中斷
    first_any = [tb["t"][0] for tb in (ctx.table("ATT"), *(k for _, k in k4s)) if tb is not None and tb["t"].size]
    log_before_arm = bool(first_any) and min(first_any) < iv.t0 - SH_GAP_S
    # 參考訊號(ATT,沒有就 XKF4 lane 0):MAG 空洞期間它也沒有記錄 → 是整個 log 暫停(SD 卡寫入卡住)或區間沒資料,
    # 不是羅盤掉線。空洞長度只算參考訊號有在記錄的部分
    ref_tb = ctx.table("ATT") or (k4s[0][1] if k4s else None)
    t_ref = ref_tb["t"] if ref_tb is not None else None
    log_pauses: list[dict] = []
    for i, tb in mags.items():
        t = tb["t"]
        m = (t >= iv.t0) & (t <= iv.t1)
        cover = max(cover, float(np.sum(m & np.isfinite(tb["MagX"]))) * _dt_nom(t))
        lvl = HIT if i == used else SOFT
        if "Health" in tb and (m & np.isfinite(tb["Health"])).any():
            has_health = True
            rr = runs((tb["Health"] == 0) & m, t, SH_HEALTH_MIN_DUR_S - JITTER_TOL_S, MAG_MAX_DT_S)
            for s, e, _, _ in rr[:1]:
                alarms.append({"kind": "health_zero", "compass": i, "t": s, "dur_s": e - s, "level": lvl})
        # 取樣空洞一律截在解鎖區間內(只算 [iv.t0, iv.t1] 裡的那一段):跨過解鎖 / 上鎖的記錄暫停,
        # 區間外的部分不是飛行證據(解鎖前最後一筆 → 解鎖後第一筆的空洞,從 iv.t0 起算)
        within = np.flatnonzero((t[1:] > iv.t0) & (t[:-1] < iv.t1))      # 與解鎖區間相交的取樣間隔
        gaps = [(max(float(t[k]), iv.t0), min(float(t[k + 1]), iv.t1)) for k in within]
        # 區間尾端:羅盤在解鎖期間停止記錄(之後再也沒有樣本 —— 感測器掉線最常見的樣子)
        if t[-1] < iv.t1:
            gaps.append((max(float(t[-1]), iv.t0), iv.t1))
        # 區間開頭:MAG 第一筆就在解鎖之後 → 只有 log 在解鎖前就在記錄時才算中斷
        if t[0] > iv.t0 and log_before_arm:
            gaps.append((iv.t0, min(float(t[0]), iv.t1)))
        eff = []
        for g0, g1 in gaps:
            if g1 <= g0:
                continue
            if g1 - g0 <= SH_GAP_S:                # 短間隔不可能超標,不必查參考訊號(效能:每個 0.1 s 間隔都會進來)
                eff.append((g0, g1 - g0))
                continue
            rec = _recording_s(t_ref, g0, g1)
            if g1 - g0 > SH_GAP_S and rec <= SH_GAP_S:
                log_pauses.append({"compass": i, "t": g0, "dur_s": g1 - g0, "recording_s": rec})
            eff.append((g0, rec))
        gaps = eff
        gaps_max[str(i)] = max(g for _, g in gaps) if gaps else None
        big = sorted(g for g in gaps if g[1] > SH_GAP_S)
        if big:
            alarms.append({"kind": "data_gap", "compass": i, "t": big[0][0], "dur_s": big[0][1], "level": lvl})
    te, se, ce = ctx.err_rows()
    ground_events = []
    for tt, s, c in zip(te, se, ce):
        if s == SH_ERR_SUBSYS_COMPASS and c in SH_ERR_BAD_CODES and tt > SH_ERR_T_MIN_S:
            if iv.t0 <= tt <= iv.t1:
                alarms.append({"kind": "err_compass", "t": float(tt), "ecode": int(c), "level": HIT})
            else:
                ground_events.append({"t": float(tt), "ecode": int(c)})
    ekf_cover = 0.0
    ts_max, fs_max = 0, 0
    for l, k4 in k4s:
        t = k4["t"]
        fin = np.zeros(t.size, dtype=bool)
        for col in ("TS", "FS"):
            if col in k4:
                fin |= np.isfinite(k4[col])
        m = (t >= max(iv.t0, SH_FS_T_MIN_S)) & (t <= iv.t1) & fin      # 覆蓋只算 TS / FS 真的有值的樣本
        if not m.any():
            continue
        ekf_cover = max(ekf_cover, float(m.sum()) * _dt_nom(t, 0.04))
        if "TS" in k4:
            ts = np.where(np.isfinite(k4["TS"]), k4["TS"], 0.0).astype(np.int64)
            ts_max = max(ts_max, int(ts[m].max()))
            rr = runs(((ts & SH_TS_MAG_BIT) != 0) & m, t, SH_EKF_MIN_DUR_S - JITTER_TOL_S, XKF_MAX_DT_S)
            if rr:
                alarms.append({"kind": "ekf_mag_timeout", "lane": l, "t": rr[0][0], "dur_s": rr[0][1] - rr[0][0], "level": HIT})
        if "FS" in k4:
            fs = np.where(np.isfinite(k4["FS"]), k4["FS"], 0.0).astype(np.int64)
            fs_max = max(fs_max, int(fs[m].max()))
            rr = runs((fs != 0) & m, t, SH_EKF_MIN_DUR_S - JITTER_TOL_S, XKF_MAX_DT_S)
            if rr:
                alarms.append({"kind": "ekf_fault", "lane": l, "t": rr[0][0], "dur_s": rr[0][1] - rr[0][0], "level": HIT,
                               "fs": int(fs[rr[0][2]])})
    vals = {"used_compass": used, "n_compasses": len(mags), "alarms": alarms, "max_gap_s": gaps_max,
            "health_checked": has_health, "log_before_arm": log_before_arm, "log_pauses": log_pauses[:10],
            "ts_max": ts_max, "fs_max": fs_max, "ground_events": ground_events,
            "mag_cover_s": cover, "ekf_cover_s": ekf_cover,
            "thr_gap_s": SH_GAP_S, "thr_health_dur_s": SH_HEALTH_MIN_DUR_S}
    if not alarms:
        if max(cover, ekf_cover) < MIN_COVER_S:
            return _na("compass_sensor_health", "insufficient_data", "解鎖期間羅盤 / EKF 紀錄太短,不足以判定感測器健康。", **vals)
        mg = max((v for v in gaps_max.values() if v is not None), default=None)
        gap_txt = f"MAG 取樣最大間隔 {mg:.2f} s(門檻 {SH_GAP_S:.0f} s)" if mg is not None else ""
        ekf_txt = "EKF 磁力計沒有逾時、無濾波器故障" if ekf_cover >= MIN_COVER_S else "缺 EKF 狀態紀錄(XKF4),未檢查 EKF 磁力計逾時"
        if mags:
            hb = (f"{len(mags)} 顆羅盤全程回報健康" if has_health
                  else f"{len(mags)} 顆羅盤(MAG 沒有 Health 欄位,健康旗標未檢查)")
            body = "、".join(x for x in (hb, "沒有 ERR COMPASS 錯誤", gap_txt) if x)
            return _finding("compass_sensor_health", CLEAR, SEV_HEALTH, iv.t0, None, "clear", vals,
                            f"羅盤感測器健康:{body};{ekf_txt}。可排除羅盤硬體故障或通訊中斷。")
        # 沒有 MAG:只有 EKF 正在融合磁力計(XKFS.MAG_FUSION ≥ 1)時,才能說羅盤資料持續送進 EKF
        fusing = False
        for _, ks in _lane_tables(ctx, "XKFS"):
            if "MAG_FUSION" in ks:
                m = (ks["t"] >= iv.t0) & (ks["t"] <= iv.t1) & np.isfinite(ks["MAG_FUSION"])
                fusing |= bool(m.any() and np.mean(ks["MAG_FUSION"][m] >= 1) > 0.5)
        vals["ekf_fusing_mag"] = fusing
        tail = "EKF 全程在融合磁力計,表示羅盤資料持續送進 EKF。" if fusing else "(無法確認 EKF 有在使用羅盤)。"
        return _finding("compass_sensor_health", CLEAR, SEV_HEALTH, iv.t0, None, "ekf_only", vals,
                        "沒有羅盤原始資料(MAG 未記錄),只能從 EKF 判斷:EKF 磁力計沒有逾時、無濾波器故障、沒有 ERR COMPASS 錯誤;"
                        + tail)
    hits = [e for e in alarms if e["level"] == HIT]
    use = hits if hits else alarms
    first = min(use, key=lambda e: e["t"])
    status = HIT if hits else SOFT
    t_tilt = _att_tilt45(ctx, iv)
    c = _causality(a, first["t"], t_tilt)
    vals.update(c)
    names = {"health_zero": "羅盤回報不健康", "data_gap": "羅盤資料中斷", "err_compass": "ERR COMPASS 錯誤",
             "ekf_mag_timeout": "EKF 磁力計逾時", "ekf_fault": "EKF 濾波器故障"}
    desc = "、".join(f"{names[e['kind']]}({e['t']:.2f} s 起" + (f" {e['dur_s']:.1f} s" if e.get("dur_s") else "")
                    + (f",MAG[{e['compass']}]" + ("" if e["compass"] == used else " 備援") if "compass" in e else "")
                    + (f",ECode {e['ecode']}" if "ecode" in e else "") + ")"
                    for e in sorted(alarms, key=lambda e: e["t"])[:4])
    text = f"羅盤感測器異常:{desc};{_cause_text(c, a)}。請檢查 CAN / I2C 線材與供電。"
    return _finding("compass_sensor_health", status, SEV_HEALTH, first["t"], first["t"], _ctx_label(c), vals, text)


# ======================================================================================
# 6. ekf_attitude_consistency
# ======================================================================================

def _recording_s(t_ref: np.ndarray | None, g0: float, g1: float) -> float:
    """[g0, g1] 之中參考訊號「有在記錄」的秒數(相鄰樣本間隔 ≤ SH_GAP_S 的區段與 [g0, g1] 的交集長度);
    沒有參考訊號 → 整段都算(無法區分)。"""
    if t_ref is None or t_ref.size < 2:
        return g1 - g0
    a0, a1 = t_ref[:-1], t_ref[1:]
    ok = (a1 - a0) <= SH_GAP_S
    return float(np.sum(np.clip(np.minimum(a1[ok], g1) - np.maximum(a0[ok], g0), 0.0, None)))


def _crash_ref(a: Anchors) -> float | None:
    """失事參考時間 t_c(事件窗用):判定失事的時間,沒有就上鎖時間;正常降落 → None。"""
    if a.end_type in CAUSALITY_END_TYPES:
        for v in (a.t_detect, a.t_disarm):
            if v is not None:
                return float(v)
        if a.interval is not None:
            return float(a.interval.t1)
    return None


@guarded("ekf_attitude_consistency", DOMAIN)
def ekf_attitude_consistency(ctx: Ctx, a: Anchors):
    """EKF 姿態(ATT)是否和獨立姿態(AHR2 / DCM)一致。

    d = wrap180(ATT − AHR2)(AHR2 先 unwrap 再內插到 ATT 時間);|dRoll|、|dPitch| > 15° 或 |dYaw| > 20°
    持續 ≥ 0.3 s → HIT('EKF 姿態與獨立姿態不一致',估計器故障候選)。次要:|ATT.Yaw − DCM.Yaw| > 15° 持續 ≥ 0.5 s
    → SOFT(低信心:DCM 偏航只是弱的獨立檢查)。判定看整個解鎖區間;失事時另外給事件窗 [t_c−15, t_c] 的數字。
    DCM.Yaw 是唯一獨立的偏航(EAHR 全 0;AHR2.Yaw 在本機型和 EKF 偏航很接近)。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("ekf_attitude_consistency")
    at = ctx.table("ATT")
    if at is None or not all(c in at for c in ("Roll", "Pitch", "Yaw")):
        return _na("ekf_attitude_consistency", "no_data", "沒有 ATT 姿態紀錄,無法比對獨立姿態。")
    ah = ctx.table("AHR2")
    dc = ctx.table("DCM")
    has_ahr = ah is not None and all(c in ah for c in ("Roll", "Pitch", "Yaw")) and ah["t"].size > 3
    has_dcm = dc is not None and "Yaw" in dc and dc["t"].size > 3
    if not has_ahr and not has_dcm:
        return _na("ekf_attitude_consistency", "no_data", "沒有 AHR2 / DCM 獨立姿態紀錄,無法比對。")
    sel = (at["t"] >= max(iv.t0 + AC_T_MIN_S, AC_BOOT_S)) & (at["t"] <= iv.t1)
    t = at["t"][sel]
    if t.size < 3 or t.size * _dt_nom(at["t"]) < MIN_COVER_S:
        return _na("ekf_attitude_consistency", "insufficient_data", "解鎖期間姿態紀錄太短,不足以比對。")
    roll, pitch, yaw = at["Roll"][sel], at["Pitch"][sel], at["Yaw"][sel]
    t_c = _crash_ref(a)
    vals: dict = {"window": [float(t[0]), float(t[-1])], "thr_rp_deg": AC_RP_DEG, "thr_yaw_deg": AC_YAW_DEG,
                  "thr_dcm_yaw_deg": AC_DCM_YAW_DEG, "event_ref_t": t_c}
    alarms: list[dict] = []
    dR = dP = dY = None
    dt_att = _dt_nom(at["t"])
    if has_ahr:
        near = _nearest_gap(ah["t"], t) <= AC_MAX_NEAR_S
        has_ahr = bool(near.any())              # AHR2 有紀錄但和解鎖區間沒有交集 → 當作沒有
    if has_ahr:
        dR = np.abs(wrap180(roll - _unwrap_interp(ah["t"], ah["Roll"], t)))
        dP = np.abs(wrap180(pitch - _unwrap_interp(ah["t"], ah["Pitch"], t)))
        dY = np.abs(wrap180(yaw - _unwrap_interp(ah["t"], ah["Yaw"], t)))
        dR, dP, dY = (np.where(near, x, np.nan) for x in (dR, dP, dY))
        # 三軸都要有足夠的有效(有限值)比對樣本才算可比(NaN 的 ATT / AHR2 不能變成 CLEAR)
        fin = np.isfinite(dR) & np.isfinite(dP) & np.isfinite(dY)
        has_ahr = float(fin.sum()) * dt_att >= MIN_COVER_S
    if has_dcm:
        near_dcm = _nearest_gap(dc["t"], t) <= 0.3
        dd = np.where(near_dcm, np.abs(wrap180(yaw - _unwrap_interp(dc["t"], dc["Yaw"], t))), np.nan)
        has_dcm = float(np.isfinite(dd).sum()) * dt_att >= MIN_COVER_S
    if not has_ahr and not has_dcm:
        return _na("ekf_attitude_consistency", "no_data", "解鎖期間沒有足夠的 AHR2 / DCM 獨立姿態有效樣本,無法比對。")
    if has_ahr:
        bad = (np.nan_to_num(dR) > AC_RP_DEG) | (np.nan_to_num(dP) > AC_RP_DEG) | (np.nan_to_num(dY) > AC_YAW_DEG)
        rr = runs(bad, t, AC_MIN_DUR_S - JITTER_TOL_S, MAG_MAX_DT_S)
        if rr:
            i0, i1 = rr[0][2], rr[0][3]
            ax = max((("roll", float(np.nanmax(dR[i0:i1 + 1]))), ("pitch", float(np.nanmax(dP[i0:i1 + 1]))),
                      ("yaw", float(np.nanmax(dY[i0:i1 + 1])))), key=lambda kv: kv[1] / (AC_YAW_DEG if kv[0] == "yaw" else AC_RP_DEG))
            alarms.append({"kind": "ahr2", "t": rr[0][0], "axis": ax[0], "peak_deg": ax[1], "level": HIT})
        vals.update(max_droll_deg=float(np.nanmax(dR)), max_dpitch_deg=float(np.nanmax(dP)), max_dyaw_deg=float(np.nanmax(dY)))
        if t_c is not None:
            ev = (t >= t_c - AC_EVENT_PRE_S) & (t <= t_c) & np.isfinite(dR)
            if ev.any():
                vals.update(event_window=[t_c - AC_EVENT_PRE_S, t_c], event_max_droll_deg=float(np.nanmax(dR[ev])),
                            event_max_dpitch_deg=float(np.nanmax(dP[ev])), event_max_dyaw_deg=float(np.nanmax(dY[ev])))
    if has_dcm:
        rr = runs(np.nan_to_num(dd) > AC_DCM_YAW_DEG, t, AC_DCM_MIN_DUR_S - JITTER_TOL_S, MAG_MAX_DT_S)
        if rr:
            alarms.append({"kind": "dcm_yaw", "t": rr[0][0], "axis": "dcm_yaw",
                           "peak_deg": float(np.nanmax(dd[rr[0][2]:rr[0][3] + 1])), "level": SOFT})
        vals.update(max_dcm_yaw_deg=float(np.nanmax(dd)), median_dcm_yaw_deg=float(np.nanmedian(dd)))
        if t_c is not None:
            ev = (t >= t_c - AC_EVENT_PRE_S) & (t <= t_c) & np.isfinite(dd)
            if ev.any():
                vals["event_max_dcm_yaw_deg"] = float(np.nanmax(dd[ev]))
    # EKF lane 間的 Roll / Pitch 差(僅供參考)
    sp_r = _lane_spread_rp(ctx, iv, a)
    if sp_r:
        vals.update(sp_r)
    vals["roll_pitch_checked"] = bool(has_ahr)
    vals["alarms"] = alarms
    if not alarms:
        if has_ahr:
            whole = f"{vals['max_droll_deg']:.1f}° / {vals['max_dpitch_deg']:.1f}° / {vals['max_dyaw_deg']:.1f}°"
            if "event_max_droll_deg" in vals:
                body = (f"失事前 {AC_EVENT_PRE_S:.0f} s 到上鎖 Roll / Pitch / Yaw 最大差 {vals['event_max_droll_deg']:.1f}° / "
                        f"{vals['event_max_dpitch_deg']:.1f}° / {vals['event_max_dyaw_deg']:.1f}°(整段解鎖期間 {whole})")
            else:
                body = f"整段解鎖期間 Roll / Pitch / Yaw 最大差 {whole}"
            text = (f"EKF 姿態與獨立姿態 AHR2 一致:{body},遠小於門檻 {AC_RP_DEG:.0f}° / {AC_YAW_DEG:.0f}°。"
                    "EKF 看到的姿態是真的,不是估計器出錯造成的假翻滾。")
            ctxs = "clear"
        else:
            text = (f"EKF 偏航與 DCM 偏航一致(最大差 {vals['max_dcm_yaw_deg']:.1f}°,門檻 {AC_DCM_YAW_DEG:.0f}°);"
                    "缺 AHR2,Roll / Pitch 未比對。")
            ctxs = "dcm_yaw_only"
        return _finding("ekf_attitude_consistency", CLEAR, SEV_ATT, float(t[0]), None, ctxs, vals, text)
    hits = [e for e in alarms if e["level"] == HIT]
    first = min(hits if hits else alarms, key=lambda e: e["t"])
    status = HIT if hits else SOFT
    t_tilt = _att_tilt45(ctx, iv)
    c = _causality(a, first["t"], t_tilt)
    vals.update(c)
    vals["axis"] = first["axis"]
    if first["kind"] == "ahr2":
        text = (f"EKF 姿態與獨立姿態 AHR2 不一致:{first['t']:.2f} s 起 {first['axis']} 差 {first['peak_deg']:.0f}°"
                f"(門檻 {AC_YAW_DEG if first['axis'] == 'yaw' else AC_RP_DEG:.0f}°),姿態估計器可能出錯;{_cause_text(c, a)}。")
    else:
        text = (f"EKF 偏航與 DCM 偏航相差 {first['peak_deg']:.0f}°({first['t']:.2f} s 起,門檻 {AC_DCM_YAW_DEG:.0f}°;"
                f"低信心,DCM 只是弱的獨立檢查);{_cause_text(c, a)}。")
    return _finding("ekf_attitude_consistency", status, SEV_ATT, first["t"], first["t"] + AC_MIN_DUR_S,
                    _ctx_label(c), vals, text)


def _lane_spread_rp(ctx: Ctx, iv: ArmedInterval, a: Anchors) -> dict:
    """XKF1 各 lane Roll / Pitch 的最大差(時間對齊,到 t_warn−1 或上鎖前 5 s)——只供參考,不參與 status。"""
    lanes = [(i, tb) for i, tb in _lane_tables(ctx, "XKF1") if "Roll" in tb and "Pitch" in tb]
    if len(lanes) < 2:
        return {}
    lo = iv.t0 + YR_SPREAD_T0_S
    hi = (a.t_warn - 1.0) if a.t_warn is not None else iv.t1 - 5.0
    t0 = lanes[0][1]["t"]
    m = (t0 >= lo) & (t0 <= hi)
    t = t0[m]
    if t.size < 3:
        return {}
    out = {}
    for key in ("Roll", "Pitch"):
        ys = [np.interp(t, tb["t"], tb[key]) for _, tb in lanes]
        sp = np.zeros(t.size)
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                sp = np.maximum(sp, np.abs(wrap180(ys[i] - ys[j])))
        out[f"lane_spread_{key.lower()}_deg"] = float(np.max(sp))
    return out


# ======================================================================================
# 進入點
# ======================================================================================

DETECTORS = (ekf_mag_reject, compass_raw_consistency, compass_motor_interference,
             ekf_yaw_reset_lane_switch, compass_sensor_health, ekf_attitude_consistency)
DETECTOR_IDS = ("ekf_mag_reject", "compass_raw_consistency", "compass_motor_interference",
                "ekf_yaw_reset_lane_switch", "compass_sensor_health", "ekf_attitude_consistency")


def run(ctx: Ctx, a: Anchors) -> list[Finding]:
    out: list[Finding] = []
    for det in DETECTORS:
        out.extend(det(ctx, a))
    return out
