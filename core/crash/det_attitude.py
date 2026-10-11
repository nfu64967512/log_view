"""姿態 / 控制偵測模組(DOMAIN = 'attitude'):姿態失控起點、角速度追蹤、控制輸出飽和、
搖桿置中卻被要求大傾角、駕駛操作背景、未下指令的偏航 / 下沉。

偵測器(每個 id 每次呼叫至少回傳一筆 Finding):
  attitude_loss_of_control    推力向量誤差(ANG 400 Hz,備援 ATT)> 10° / 30° → 設定 a.t_warn / a.t_loss / a.t_inverted
  rate_demand_not_achieved    角速度指令 vs 實際(RATE 滾動 0.5 s):要求轉但機身沒動(躺在地上 / 推力無法產生力矩)
  rate_output_saturation      RATE.ROut / POut 偏大或碰到上限(力矩用盡)
  lean_without_pilot          搖桿置中卻由飛控下大傾角指令(位置控制器推翻機身)
  pilot_input_context         駕駛大動作(滿舵 / 快速甩桿 / 來回切 EKF 來源開關)是原因還是事後反應
  uncommanded_yaw_or_sink     沒打舵的偏航自轉 / 下降比指令快很多(只作排除證據)

共通規則:
  - 只分析 a.interval(解鎖區間),t > a.interval.t1(上鎖後 / 失事後)的樣本一律不看;未解鎖 → NA('not_armed')。
    高度 / 升降率(CTUN.CRt / Alt)另外只取 a.t_first_hit 之前(撞擊後氣壓 / EKF 高度是垃圾)。
  - 時間一律相對秒;RCIN / RCOU 等 uint16 表由 Ctx 轉成 float64 後才做減法。
  - 特技模式(ACRO 1 / SPORT 13 / FLIP 14)與 EV21..EV22+3 s 是合法的大姿態,不做姿態 / 角速度 / 飽和判讀。
  - 持續 N 秒的判準一律在資料空洞處斷開(ANG / RATE:0.05 s;10 Hz 序列:0.35 s)。
  - 單一失事樣本:所有門檻都是「這一次失事 vs 同機兩段正常飛行」的間隔(見各常數註解),
    只有 124 一個正樣本的偵測器信心偏低,證據請連同實測數值一起呈現,不是機率。

【與規格的差異】
  - lean_without_pilot 的空中判讀只用「駕駛以搖桿定位」的位置模式(LOITER 5 / POSHOLD 16 / FLOWHOLD 22);
    規格列的 AUTO / GUIDED / RTL / CIRCLE / BRAKE / SMART_RTL / FOLLOW / AUTO_RTL 是「飛控自己規劃路徑」,
    搖桿置中本來就會有 12° 以上傾角(任務加速 / 逆風),照列會誤報。地面怠速判讀(GROUND)不限模式。
  - t_inverted = 傾角 tilt > 90° 持續 0.2 s(依目錄);倒扣深度 ≥ 120° 的時間另放在 values['t_tilt120']。
  - 規格的 t_onset(窗口型判準)取「第一個成立視窗的結束時間」:該時刻證據才完整,對因果判斷是保守的。
  - rate_demand_not_achieved 的 context 用 rate_output_saturation 的判準(窗內 ThO 中位數 < 0.65 × ThH → ground_idle),
    不是規格寫的 ThO ≤ 0.03:124 觸地後 CTUN.ThO(10 Hz)在 0 與 0.04–0.16 間彈跳,1255.12–1255.82 窗的中位數 0.05,
    照 ≤ 0.03 會把「躺在地上」誤標成 air(進而算進馬達假說)。
  - pilot_input_context 有多個大動作時,t_onset 取「最接近 t_warn 的前兆」(沒有前兆才取第一個反應),
    規格只說 delay = t_input − t_warn,沒有指定是哪一個動作。前兆只算 dataflow STEP 4(c) 的前兆窗
    P = [max(t_arm+3, t_warn−60), t_warn) 內的動作;更早的只列在 values['earlier_inputs'](資訊)。
  - 「一次失事只報一筆」:rate_demand_not_achieved / rate_output_saturation / lean_without_pilot 先把成立的 run
    依間隔 ≤ EPISODE_GAP_S 併成事件(episode),再挑「這次失事的那一段」:結束時間在失事參考時間
    (t_warn,沒有就 t_loss / 異常結束的 t_first_hit / t_detect)前 CRASH_LEAD_S 以內或之後的事件優先,
    其中依層級 / 與參考時間的距離挑;較早無關的事件只放 values['other_episodes'],不會蓋掉失事當下的證據。
    沒有失事參考(正常結束)時取最強、最早的一段。規格沒有規定多段時取哪一段(原寫法是整段最早一次)。
  - rate_output_saturation 同一次事件的分層起點改用上述 episode(原規格 / 原寫法:最高層級前 10 s 內最早的層級)。
  - attitude_loss_of_control:異常結束時 t_warn / t_loss 取「通往失事的那一段」— 在失事參考時間前已恢復
    (ae < 5° 持續 2 s)的較早越線不算,列在 values['earlier_warn_runs'];規格寫「第一次持續越線」。
    正常結束仍照規格取第一次。124 不受影響(段內 ae 最低 4.49° 只維持 0.42 s)。
  - lean_without_pilot 的 GROUND 判讀另要求離地 ≤ GROUND_AGL_M(有高度資料時;規格只看 ThO);
    AIR 判讀排除「搖桿回中 / 切入該模式後的煞車期」(LOITER:LOIT_BRK_DELAY + LOIT_SPEED / LOIT_BRK_ACCEL
    + LOIT_BRK_ACCEL / LOIT_BRK_JERK;POSHOLD:2·PHLD_BRAKE_ANGLE / PHLD_BRAKE_RATE + 1.5 s):124 兩段正常飛行
    搖桿置中的空中傾角 > 5.8° 全部出現在回中後 2 s 內(煞車;單點最大 8.5° / 9.7°,持續 0.5 s 最大 7.5° / 8.6°),
    煞車期(124 參數 4.6 s)之後持續 0.5 s 最大 5.7° / 3.0°;
    預設參數(LOIT_BRK_ACCEL 250 cm/s²)的煞車可達 atan(2.5/9.81) = 14.3°,會超過 12° 門檻。空中類 severity 為 high
    (規格整體 critical;只有地面 ≥ 15° 會擋住降落偵測,才是 critical)。
已知限制(n=1 校準):所有門檻都只用 00000124 一次失事 + 同一位溫和駕駛的兩段正常飛行校準;空中 lean 12°、
rate pitch 30°/s(1.33x)的間隔偏小,激烈操作的駕駛可能需要放寬。
"""
from __future__ import annotations

import numpy as np

from . import util
from .ctx import Ctx
from .guard import guarded
from .types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval, Finding

DOMAIN = "attitude"

# ============================ 共通常數 ============================
ARM_SKIP_S = 3.0              # 解鎖後前 3 s 不判讀(起轉 / 解鎖手勢:C4=1935、RATE.YDes ±120°/s)
GAP_FAST_S = 0.05             # ANG / RATE(~400 Hz)資料空洞超過這個就斷開持續判準(RATE 最長空洞 0.17 s)
GAP_SLOW_S = 0.35             # 10 Hz 序列(ATT / RCIN / CTUN)斷開門檻(124 有 13 個 0.5–0.8 s 空洞)
RC_MAX_AGE_S = 0.35           # RCIN / CTUN / RCOU 零階保持的最大資料年齡,超過視為沒有資料
MIN_ANALYSED_S = 5.0          # 可判讀秒數不足這個就回 NA(不能下「沒問題」的結論)
EXEMPT_MODES = (1, 13, 14)    # ACRO / SPORT / FLIP:合法的大姿態
GROUND_AGL_M = 1.5            # 離地高度 ≤ 1.5 m 視為「在地面附近」(沒有測距儀:AGL ≈ CTUN.Alt − 解鎖時 Alt)
EV_FLIP_START, EV_FLIP_END = 21, 22
FLIP_TAIL_S = 3.0             # FLIP 結束後再多 3 s 仍視為特技
FLIP_MAX_S = 10.0             # 找不到 EV22 時,EV21 後最多免判 10 s
THO_IDLE = 0.03               # CTUN.ThO ≤ 0.03 = 油門已壓到最低(124 觸地後為精確 0;不要用 == 0)
SPOOL_PWM_US = 1145.0         # 馬達已起轉(解鎖怠速 1190 µs,未解鎖 1100 µs)
IDLE_HOLD_S = 0.5             # 地面怠速的「保持」:0.5 s 內 ThO 碰過底就算(CTUN 10 Hz 在觸地彈跳時閃爍)
NORMAL_ENDS = ("not_armed", "normal_landing", "normal_end")
EPISODE_GAP_S = 3.0           # 同一事件:成立的 run 之間間隔 ≤ 3 s 併成一段(124 失事段內最大間隔 0.86 s,3.5x)
CRASH_LEAD_S = 10.0           # 事件結束在失事參考時間前 10 s 以內(或之後)= 這次失事的一部分(124 觸地→預警 0.7 s)
MAX_LISTED = 5                # values 裡最多列幾個其他事件

# ============================ attitude_loss_of_control ============================
# 推力向量誤差 ae(= 韌體 AngErr 同定義);124 正常飛行(659 武裝秒)最大 3.6°(p99.9 3.0)
AE_INFO_DEG = 5.0             # 資訊級:持續 0.3 s(正常最大 3.6° → 1.4x)
AE_WARN_DEG = 10.0            # 預警:持續 0.3 s → t_warn(124 實測 1255.915;正常 2.8x)
AE_LOSS_DEG = 30.0            # 失控:持續 0.3 s → t_loss(= ArduPilot CRASH_CHECK 角度;124 實測 1258.102)
AE_SUSTAIN_S = 0.3
TILT_INVERTED_DEG = 90.0      # 倒扣:tilt > 90° 持續 0.2 s → t_inverted(124:1258.97;正常飛行最大 tilt 12.1°)
TILT_INVERTED_S = 0.2
TILT_DEEP_DEG = 120.0         # 深度倒扣(124:1259.22),只放 values
ATTRIB_WINDOW_S = 1.0         # t_warn 後 1 s 比較「指令變化 vs 機身變化」
ATTRIB_RATIO = 3.0            # 指令變化 > 3x 機身變化 = demand 型(控制器把飛機推向一側)
ATTRIB_MIN_DEG = 1.0          # 兩者都 < 1° 時不下結論
H_WINDOW_END_S = 5.0          # 基準統計(ae_max_normal)取 t_warn − 5 s 以前
AE_RECOVER_S = 2.0            # ae < 5°(資訊級以下)持續 2 s = 已恢復;124 失事段內 ae 最低 4.49° 只維持 0.42 s

# ============================ rate_demand_not_achieved ============================
# 124 正常飛行 0.5 s 窗最大 rmsD:roll 13.6 / 17.5、pitch 15.8 / 22.5;失事 36–87(最薄的是 pitch 1.33x)
RATE_WINDOW_S = 0.5           # 窗長(400 Hz → 200 樣本)
RATE_WINDOW_SPAN_MAX = 1.2    # 窗內時間跨度 > 1.2 × 窗長視為跨過資料空洞,略過
RATE_RMSD_MIN = 30.0          # deg/s;低於 15 時 gain 檢定無意義(小指令是雜訊)
RATE_GAIN_MAX = 0.3           # 回歸斜率:正常機身 0.5–1;躺地面 ≈ 0 或負
RATE_SUSTAIN_S = 0.2

# ============================ rate_output_saturation ============================
# 124 正常:0.25 s 平均 |ROut| ≤ 0.118、|POut| ≤ 0.107;原始值 ≤ 0.149(飛行 1 觸地怠速秒 0.255)
SAT_MEAN_WINDOW_S = 0.25
SAT_MEAN_MIN = 0.20           # SOFT:1.7x
SAT_RAW_HIT = 0.5             # HIT:原始值 > 0.5 累計 ≥ 0.1 s(任一 2 s 窗內);正常最大 0.255 → 2x
SAT_HIT_TIME_S = 0.1
SAT_HIT_WINDOW_S = 2.0
SAT_RAW_CRIT = 0.9            # CRITICAL:原始值 ≥ 0.9 至少 2 個樣本在 0.1 s 內
SAT_CRIT_SAMPLES = 2
SAT_CRIT_WINDOW_S = 0.1
AIR_THO_RATIO = 0.65         # context:ThO ≥ 0.65 × ThH → 'air',否則 'ground_idle'

# ============================ lean_without_pilot ============================
# 124 正常:搖桿置中時空中傾角需求單點最大 8.5°(飛行 1)/ 9.7°(飛行 2)— 都在搖桿回中後 2 s 內(Loiter 煞車);
# 煞車期過後持續 0.5 s 最大 5.7° / 3.0°(12° 為 2.1x);地面怠速最大 2.2°(8° 為 3.6x)
STICK_CENTER_US = 30.0        # 搖桿置中死區(RC DZ 20 µs)
AIR_LEAN_DEG = 12.0           # 空中:持續 0.5 s(只判煞車期之後)
AIR_LEAN_S = 0.5
GROUND_LEAN_DEG = 8.0         # 地面怠速:持續 0.3 s
GROUND_LEAN_S = 0.3
GROUND_LEAN_CRIT_DEG = 15.0   # > 15° 會擋住降落偵測器(LDET bit8)
INFO_LEAN_DEG = 6.0           # 資訊級(只計數不告警):正常飛行 2 段共 6 次
FLIPPED_TILT_DEG = 45.0       # 機身實際傾角 > 45° = 已翻覆,後續的 lean 屬於翻覆本身(後果)
AIR_MODES = (5, 16, 22)       # LOITER / POSHOLD / FLOWHOLD(見模組 docstring:自主模式不判)
POS_AGREE_DEG = 4.0           # |lean_psc − lean| < 4° → 傾斜來自位置控制器
GRAVITY = 9.81
POS_ERR_REF_M = 1.5
# 煞車期(搖桿回中或切入位置模式後,飛控照設計自己傾斜減速):ArduCopter 預設值,參數有記錄時用 log 的
LOIT_DEFAULTS = {"LOIT_SPEED": 1250.0, "LOIT_BRK_ACCEL": 250.0, "LOIT_BRK_DELAY": 1.0, "LOIT_BRK_JERK": 500.0}
PHLD_DEFAULTS = {"PHLD_BRAKE_ANGLE": 3000.0, "PHLD_BRAKE_RATE": 8.0}
PHLD_LOITER_BLEND_S = 1.5     # POSHOLD 煞車結束後轉 Loiter 的混合時間(韌體 150 個 100 Hz 迴圈)
AIR_BRAKE_MIN_S = 2.0         # 煞車期下限(124 參數算出 4.6 s;實測回中後 2 s 內就降到 3.8° 以下)
AIR_BRAKE_MAX_S = 10.0        # 煞車期上限(預設參數:LOITER 6.5 s、POSHOLD 9.0 s)
POSHOLD_MODE = 16

# ============================ pilot_input_context ============================
STICK_FULL_TRAVEL_US = 420.0  # 半程(840 µs 全程);由 RCn_MIN/MAX 參數推算,沒有就用這個
PILOT_DEV_A = 0.8             # A:max(|dev1|,|dev2|) ≥ 0.8 持續 0.2 s(正常空中最大 0.40 / 0.52)
PILOT_A_S = 0.2
PILOT_DEV_B = 0.6             # B:0.5 s 內變化 ≥ 0.6 全程(正常最大 0.28 / 0.38)
PILOT_B_WINDOW_S = 0.5
PILOT_C_FLIP_S = 1.5          # C:EKF 來源開關 A→B→A 折返 ≤ 1.5 s(正常 0 次)
RC_OPTION_EKF_SRC = 90        # RCn_OPTION = 90:EKF 來源組選擇
RC_LEVEL_LOW, RC_LEVEL_HIGH = 1300.0, 1700.0
PILOT_PRE_WINDOW_S = 60.0     # 前兆窗 P = [max(t_arm+3, t_warn−60), t_warn):只有窗內的大動作算前兆(dataflow STEP 4c)
PILOT_CALM_DEV = 0.5 * PILOT_DEV_A   # 前兆窗內最大偏移 < 0.4 行程才寫「平穩」(124 失事前 ≤ 0.30;正常空中最大 0.40 / 0.52)
THR_MIN_MARGIN_US = 20.0

# ============================ uncommanded_yaw_or_sink ============================
# 無正樣本(124 兩者都沒觸發)→ 低信心,只當排除證據
YAW_MEAN_WINDOW_S = 0.5
YAW_RATE_MIN = 30.0           # deg/s(ATC_RATE_Y_MAX=60;正常偏航操作達 52.6 但 C4 一定有偏移)
YAW_STICK_HOLD_S = 1.0        # 偏航搖桿需置中至少 1 s
YAW_AE_MAX_DEG = 15.0         # ae ≥ 15° 時 Z 軸陀螺的投影是翻覆不是偏航
SINK_ERR_CMS = -60.0          # CRt − DCRt ≤ −60 cm/s(正常最差 −13)
SINK_SUSTAIN_S = 0.5
FREEFALL_ACC = 3.0            # 自由落體佐證:|a| < 3 m/s² ≥ 0.3 s,只在起點前 3 s 內看
FREEFALL_S = 0.3
SINK_PRE_S = 3.0


# =====================================================================
#                              通用小工具
# =====================================================================

def _f(x, nd: int = 3):
    """numpy 純量 → JSON 友善的 float;NaN / None → None。"""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _na(det_id: str, context: str, text: str, **values) -> Finding:
    return Finding(det_id, DOMAIN, NA, "info", None, None, context, dict(values), text)


def _not_armed(det_id: str) -> Finding:
    return _na(det_id, "not_armed", "未解鎖 / 非飛行 log(桌上或手持,馬達沒轉):不適用,不做飛行姿態 / 控制分析。")


def _runs(mask: np.ndarray, t: np.ndarray, min_dur: float = 0.0, max_gap: float | None = None):
    """連續為真的區段 [(t_start, t_end, i0, i1)],向量化版的 util.runs(400 Hz 長序列用)。

    相鄰樣本時間差 > max_gap 視為資料空洞而斷開;持續時間 t_end − t_start < min_dur 的區段丟掉。
    """
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    if n == 0:
        return []
    t = np.asarray(t, dtype=float)
    brk = np.zeros(n, dtype=bool)                 # brk[i]:i−1 與 i 之間有空洞
    if max_gap is not None and n > 1:
        brk[1:] = np.diff(t) > max_gap
    prev = np.concatenate(([False], mask[:-1]))
    start = mask & (~prev | brk)
    nxt = np.concatenate((mask[1:], [False]))
    brk_next = np.concatenate((brk[1:], [True]))
    end = mask & (~nxt | brk_next)
    i0 = np.flatnonzero(start)
    i1 = np.flatnonzero(end)
    return [(float(t[a]), float(t[b]), int(a), int(b)) for a, b in zip(i0, i1) if t[b] - t[a] >= min_dur]


def _dt_med(t: np.ndarray) -> float:
    d = np.diff(t)
    d = d[d > 0]
    return float(np.median(d)) if d.size else 0.0


def _slice(tb: dict, t0: float, t1: float, cols) -> dict:
    """依時間取表的 [t0, t1] 子集(表時間遞增);只回傳需要的欄位。"""
    t = tb["t"]
    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="right"))
    out = {"t": t[i0:i1]}
    for c in cols:
        out[c] = tb[c][i0:i1]
    return out


def _zoh(t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray, max_age: float = RC_MAX_AGE_S):
    """零階保持對時間;回傳 (值, 有效布林):最近的前一個樣本超過 max_age 就無效。"""
    if t_src.size == 0:
        return np.zeros(t_dst.size), np.zeros(t_dst.size, dtype=bool)
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    ok = idx >= 0
    idc = np.clip(idx, 0, t_src.size - 1)
    return v_src[idc], ok & ((t_dst - t_src[idc]) <= max_age)


def _analysed_s(t: np.ndarray, valid: np.ndarray, max_gap: float) -> float:
    """valid 為真且沒有跨空洞的累計秒數。"""
    if t.size < 2:
        return 0.0
    d = np.diff(t)
    return float(np.sum(d[(d <= max_gap) & valid[:-1] & valid[1:]]))


def _flip_windows(ctx: Ctx, iv: ArmedInterval) -> list[tuple[float, float]]:
    ends = ctx.ev_times(EV_FLIP_END, iv.t0 - 1.0, iv.t1 + FLIP_MAX_S)
    out = []
    for s in ctx.ev_times(EV_FLIP_START, iv.t0 - FLIP_MAX_S, iv.t1):
        e = next((x for x in ends if x >= s), s + FLIP_MAX_S)
        out.append((s, e + FLIP_TAIL_S))
    return out


def _mode_array(ctx: Ctx, t: np.ndarray) -> np.ndarray:
    """每個樣本當下的 ModeNum(整數;沒有 MODE 表 → −1)。"""
    tm, nums, _ = ctx.mode_rows()
    if not len(tm):
        return np.full(t.size, -1, dtype=int)
    idx = np.searchsorted(tm, t, side="right") - 1
    return np.where(idx >= 0, np.asarray(nums)[np.clip(idx, 0, None)], -1).astype(int)


def _exempt_mask(ctx: Ctx, iv: ArmedInterval, t: np.ndarray) -> np.ndarray:
    """True = 特技模式(ACRO / SPORT / FLIP)或 FLIP 事件窗內,姿態類判讀要略過。"""
    m = np.isin(_mode_array(ctx, t), EXEMPT_MODES)
    for a, b in _flip_windows(ctx, iv):
        m |= (t >= a) & (t <= b)
    return m


def _airborne_mask(ctx: Ctx, iv: ArmedInterval, t: np.ndarray):
    """解鎖區間內「不是落地狀態」且解鎖滿 ARM_SKIP_S 的樣本。回傳 (mask, 來源 'ev' | 'tho')。

    落地狀態來自 EV 28 / 18(Ctx.landed_steps,已排除上鎖當下寫出的 EV17/18);
    區間內從沒有 EV 28(沒記到起飛事件)時,退而用 CTUN.ThO > THO_IDLE。
    """
    m = (t >= iv.t0 + ARM_SKIP_S) & (t <= iv.t1)
    steps = ctx.landed_steps(iv)
    if any(not s for _, s in steps):
        st = np.array([x[0] for x in steps], dtype=float)
        sv = np.array([x[1] for x in steps], dtype=bool)
        idx = np.searchsorted(st, t, side="right") - 1
        landed = np.where(idx >= 0, sv[np.clip(idx, 0, None)], True)
        return m & ~landed, "ev"
    s = ctx.series("CTUN.ThO")
    if s is None:
        return np.zeros(t.size, dtype=bool), "none"
    v, ok = _zoh(s[0], s[1], t)
    return m & ok & (v > THO_IDLE), "tho"


def _att_source(ctx: Ctx, iv: ArmedInterval) -> dict | None:
    """姿態資料(解鎖區間內):優先 ANG(400 Hz),不夠用則 ATT(10 Hz)。"""
    cols = ("Roll", "Pitch", "Yaw", "DesRoll", "DesPitch", "DesYaw")
    for name, gap in (("ANG", GAP_FAST_S), ("ATT", GAP_SLOW_S)):
        tb = ctx.table(name)
        if tb is None or not all(c in tb for c in cols):
            continue
        s = _slice(tb, iv.t0, iv.t1, cols)
        if s["t"].size >= 20:
            s["src"] = name
            s["gap"] = gap
            return s
    return None


def _throttle_context(ctx: Ctx, ta: float, tb: float, iv: ArmedInterval | None = None):
    """[ta, tb] 的油門背景:('air' | 'ground_idle', ThO 中位數, ThH 中位數)。給 iv 時窗口裁在解鎖區間內(不看上鎖後)。"""
    so = ctx.series("CTUN.ThO")
    if so is None:
        return "unknown", None, None
    if iv is not None:
        ta, tb = max(ta, iv.t0), min(tb, iv.t1)
        tb = max(tb, ta)
    t, v = so
    m = (t >= ta) & (t <= tb)
    if not m.any():
        i = int(np.clip(np.searchsorted(t, ta) - 1, 0, t.size - 1))
        tho = float(v[i])
    else:
        tho = float(np.nanmedian(v[m]))
    sh = ctx.series("CTUN.ThH")
    thh = None
    if sh is not None:
        mm = (sh[0] >= ta) & (sh[0] <= tb)
        if mm.any():
            thh = float(np.nanmedian(sh[1][mm]))
    if thh is None or not np.isfinite(thh) or thh <= 0:
        thh = ctx.param("MOT_THST_HOVER", None)
    if thh is not None and thh > 0:
        return ("air" if tho >= AIR_THO_RATIO * thh else "ground_idle"), tho, float(thh)
    return ("air" if tho > THO_IDLE else "ground_idle"), tho, None


def _agl_at(ctx: Ctx, iv: ArmedInterval, t: float) -> float | None:
    """估計離地高度(m):CTUN.Alt(t) − 解鎖後 3 s 的 Alt 中位數;沒有測距儀,誤差約 ±0.15 m(氣壓漂移)。"""
    alt, g0 = ctx.series("CTUN.Alt"), ctx.ground_alt(iv)
    if alt is None or g0 is None:
        return None
    return float(np.interp(t, alt[0], alt[1]) - g0)


def _motor_cols(ctx: Ctx, tb: dict) -> list[str]:
    cols = [f"C{n}" for n in range(1, 17)
            if 33 <= ctx.param(f"SERVO{n}_FUNCTION", 0) <= 40 and f"C{n}" in tb]
    return cols or [c for c in ("C1", "C2", "C3", "C4") if c in tb]


def _rc_trim(ctx: Ctx, ch: int, rc: dict | None) -> float:
    """RCn_TRIM(校準值,不是 1500);沒有參數時取解鎖期間該通道的中位數。"""
    p = ctx.param(f"RC{ch}_TRIM", None)
    if p is not None and 900.0 < p < 2100.0:
        return float(p)
    if rc is not None and f"C{ch}" in rc:
        m = ctx.armed_mask(rc["t"])
        if m.any():
            return float(np.nanmedian(rc[f"C{ch}"][m]))
    return 1500.0


def _rc_half_travel(ctx: Ctx, ch: int) -> float:
    lo, hi = ctx.param(f"RC{ch}_MIN", None), ctx.param(f"RC{ch}_MAX", None)
    if lo is not None and hi is not None and hi - lo > 200.0:
        return float(hi - lo) / 2.0
    return STICK_FULL_TRAVEL_US


def _sustained_max(x: np.ndarray, n: int) -> float | None:
    """最大的 L 使得存在連續 n 個樣本都 ≥ L(= x 的 n 樣本滑動最小值的最大值);x 中不合格處請先設 −inf / 0。"""
    if x.size < n or n < 1:
        return None
    if n == 1:
        v = float(np.max(x))
    else:
        v = float(np.max(np.lib.stride_tricks.sliding_window_view(x, n).min(axis=1)))
    return v if np.isfinite(v) else None


def _fmt_t(t) -> str:
    return "—" if t is None else f"{t:.2f} s"


def _sentences(*parts: str) -> str:
    """把非空的句子用「。」串起來(每句去掉結尾的 。 / ;,避免「。。」)。"""
    out = [p.strip().rstrip("。;") for p in parts if p and p.strip().rstrip("。;")]
    return "。".join(out) + "。" if out else ""


def _unit(ctx: Ctx, key: str) -> str:
    """欄位單位(小寫;查不到回 '')。替身 / 沒有索引的 Ctx 不會因此丟例外。"""
    try:
        return str(ctx.ix.unit(key) or "").strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def _imu_table(ctx: Ctx):
    """IMU 第 0 個 instance(沒有 instance 就整張表);查不到回 None。"""
    try:
        return ctx.table("IMU", 0 if ctx.instances("IMU") else None)
    except Exception:  # noqa: BLE001
        return ctx.table("IMU")


# ---------------- 多段事件:挑「這次失事的那一段」 ----------------

def _crash_ref(a: Anchors) -> float | None:
    """失事參考時間:t_warn,沒有就 t_loss;異常結束再退回第一次硬撞擊 / 飛控判定失事。正常結束 → None。"""
    if a.t_warn is not None:
        return float(a.t_warn)
    if a.t_loss is not None:
        return float(a.t_loss)
    if a.end_type in NORMAL_ENDS:
        return None
    for v in (a.t_first_hit, a.t_detect):
        if v is not None:
            return float(v)
    return None


def group_episodes(items: list[dict], gap: float = EPISODE_GAP_S) -> list[list[dict]]:
    """把 {'t0','t1',...} 依時間排序後,間隔(前一段結束 → 下一段開始)≤ gap 的併成同一段事件。"""
    out: list[list[dict]] = []
    end = -np.inf
    for it in sorted(items, key=lambda x: (x["t0"], x["t1"])):
        if out and it["t0"] <= end + gap:
            out[-1].append(it)
            end = max(end, it["t1"])
        else:
            out.append([it])
            end = it["t1"]
    return out


def _dist(e: dict, ref: float | None) -> float:
    """事件 [t0, t1] 到參考時間的距離(包含參考時間 = 0)。"""
    if ref is None:
        return 0.0
    return max(0.0, e["t0"] - ref, ref - e["t1"])


def choose_episode(eps: list[dict], ref: float | None, key) -> dict | None:
    """挑代表事件。key(e, d) → 可比較的 tuple(大者優先);d = 與參考時間的距離。

    有參考時間:結束時間 ≥ ref − CRASH_LEAD_S 的事件(這次失事的一部分)優先,沒有才從較早的事件裡挑;
    沒有參考時間(正常結束):全部一起比。同分取較早的一段。
    """
    if not eps:
        return None
    pool = eps
    if ref is not None:
        near = [e for e in eps if e["t1"] >= ref - CRASH_LEAD_S]
        pool = near or eps
    return max(pool, key=lambda e: (*key(e, _dist(e, ref)), -e["t0"]))


def _in_spans(t: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(np.asarray(t).size, dtype=bool)
    for a0, a1 in spans:
        m |= (t >= a0) & (t <= a1)
    return m


# =====================================================================
#                      1. attitude_loss_of_control
# =====================================================================

def attitude_events(t: np.ndarray, ae: np.ndarray, tilt: np.ndarray, max_gap: float,
                    t_crash: float | None = None, hit_based: bool = False) -> dict:
    """推力向量誤差 / 傾角的持續越線(純數值,方便單元測試)。

    回傳 {'info','warn','loss','inv','deep'}:各為 (t_start, t_end) 或 None;另有 'earlier_warn'(不屬於失事段、
    已恢復的較早 warn run 清單)與 'recovered'(有 warn run 但全部在失事前恢復)。
    t_crash None(正常結束 / 沒有失事參考):各層級取第一次持續越線(規格)。
    t_crash 給定(異常結束):取「通往失事的那一段」— 在 t_crash 前已恢復(ae < 5° 持續 AE_RECOVER_S)的較早越線不算;
    同一段內 ae 暫時回落(124 在 Roll 穿過 DesRoll 時 1257.52–1257.94 掉到 4.49°)不算恢復,所以仍取段內第一次。
    全部都在 t_crash 前恢復時:hit_based(t_crash = 第一次硬撞擊)→ warn / loss = None(撞擊在先);
    否則取最後一段(例:失事後機身停在正立姿態超過 2 s 才上鎖)。
    """
    tol = 0.5 * _dt_med(t)

    def runs(mask, dur):
        return [(r[0], r[1]) for r in _runs(mask, t, max(dur - tol, 0.0), max_gap)]

    R = {"info": runs(ae > AE_INFO_DEG, AE_SUSTAIN_S),
         "warn": runs(ae > AE_WARN_DEG, AE_SUSTAIN_S),
         "loss": runs(ae > AE_LOSS_DEG, AE_SUSTAIN_S),
         "inv": runs(tilt > TILT_INVERTED_DEG, TILT_INVERTED_S),
         "deep": runs(tilt >= TILT_DEEP_DEG, TILT_INVERTED_S)}
    out = {k: (v[0] if v else None) for k, v in R.items()}
    out["earlier_warn"], out["recovered"] = [], False
    if t_crash is None:
        return out
    rec0 = [r[0] for r in runs(np.isfinite(ae) & (ae < AE_INFO_DEG), AE_RECOVER_S)]

    def recovered(t_from: float, t_to: float) -> bool:
        """t_from 之後、t_to 之前有一段完整的恢復(ae < 5° 持續 AE_RECOVER_S)。"""
        return any(t_from - tol <= s0 and s0 + AE_RECOVER_S - tol <= t_to for s0 in rec0)

    def pick(lst, t_to, fallback):
        for r in lst:
            if r[0] <= t_to + tol and not recovered(r[1], t_to) or r[0] > t_to + tol:
                return r
        return (lst[-1] if lst else None) if fallback else None

    warn = pick(R["warn"], t_crash, not hit_based)
    if warn is not None:
        loss = next((r for r in R["loss"] if r[0] >= warn[0] - tol), None)
        info = next((r for r in R["info"] if r[0] <= warn[0] + tol and not recovered(r[1], warn[0])), None)
    else:
        loss = None
        info = pick(R["info"], t_crash, not hit_based)
    ep0 = info[0] if info is not None else (warn[0] if warn is not None else (t_crash if hit_based else None))
    for k in ("inv", "deep"):
        out[k] = next((r for r in R[k] if r[0] >= ep0 - tol), None) if ep0 is not None else out[k]
    out.update(info=info, warn=warn, loss=loss)
    out["earlier_warn"] = [r for r in R["warn"] if warn is None or r[1] < warn[0]]
    out["recovered"] = warn is None and bool(R["warn"])
    return out


def _crash_end(a: Anchors, iv: ArmedInterval):
    """attitude_events 的 (t_crash, hit_based):正常結束 → (None, False);有區間內的第一次硬撞擊 → (撞擊, True);
    否則 → (上鎖 / 區間結束, False)。"""
    if a.end_type in NORMAL_ENDS:
        return None, False
    hit = a.t_first_hit
    if hit is not None and iv.t0 < hit <= iv.t1 + 0.5:
        return float(hit), True
    return float(iv.t1), False


def _attitude_core(ctx: Ctx, iv: ArmedInterval, a: Anchors | None = None):
    """算 ae / tilt 與事件。回傳 (series dict, ae, tilt, events dict, exempt mask) 或 None(沒有可用姿態資料)。"""
    s = _att_source(ctx, iv)
    if s is None:
        return None
    t = s["t"]
    exempt = _exempt_mask(ctx, iv, t)
    ae = util.thrust_vector_error_deg(s["Roll"], s["Pitch"], s["Yaw"], s["DesRoll"], s["DesPitch"], s["DesYaw"])
    tilt = util.tilt_deg(s["Roll"], s["Pitch"])
    ae = np.where(exempt, np.nan, ae)
    tilt = np.where(exempt, np.nan, tilt)
    t_crash, hit_based = _crash_end(a, iv) if a is not None else (None, False)
    ev = attitude_events(t, ae, tilt, s["gap"], t_crash, hit_based)
    return s, ae, tilt, ev, exempt


def _driver(s: dict, t_warn: float):
    """t_warn 後 1 s:指令傾角變化 vs 機身實際傾角變化。回傳 (driver, dDes, dAct)。"""
    t = s["t"]
    des = np.hypot(s["DesRoll"], s["DesPitch"])
    act = np.hypot(s["Roll"], s["Pitch"])
    i0 = int(np.searchsorted(t, t_warn, side="left"))
    i1 = int(np.searchsorted(t, t_warn + ATTRIB_WINDOW_S, side="right")) - 1
    if i0 >= t.size or i1 <= i0:
        return "indeterminate", None, None
    d_des = abs(float(des[i1] - des[i0]))
    d_act = abs(float(act[i1] - act[i0]))
    if d_des < ATTRIB_MIN_DEG and d_act < ATTRIB_MIN_DEG:
        return "indeterminate", d_des, d_act
    return ("demand" if d_des > ATTRIB_RATIO * d_act else "vehicle"), d_des, d_act


def _apply_attitude_anchors(a: Anchors, t_warn, t_loss, t_inv):
    """把算出的 t_warn / t_loss / t_inverted 寫進錨點(算不出來的不動既有值)。

    規格:ae 在撞擊前從未 > 10°(或姿態失控晚於第一次硬撞擊)→ t_warn = t_loss = t_first_hit
    (撞擊在先:外力撞擊型,不是姿態先失控)。需要 end_state 模組已填 t_first_hit 且 end_type 屬異常結束。
    """
    hit = a.t_first_hit
    abnormal = a.end_type not in ("not_armed", "normal_landing", "normal_end")
    if hit is not None and abnormal and (t_warn is None or t_warn > hit):
        a.t_warn = a.t_loss = float(hit)
        if t_inv is not None and t_inv > hit:
            a.t_inverted = float(t_inv)
        return "impact_first"
    if t_warn is not None:
        a.t_warn = float(t_warn)
    if t_loss is not None:
        a.t_loss = float(t_loss)
    if t_inv is not None:
        a.t_inverted = float(t_inv)
    return "attitude_first" if t_warn is not None else "none"


@guarded("attitude_loss_of_control", DOMAIN)
def attitude_loss_of_control(ctx: Ctx, a: Anchors):
    """姿態失控起點:ae > 10° 持續 0.3 s = t_warn(run 起點);> 30° = t_loss;tilt > 90° = t_inverted。

    HIT = 達預警(severity critical 表示也達 30° 失控);SOFT = 只到資訊級(> 5°,context 'minor'),或只有失事前
    就已恢復的較早越線(context 'recovered',不寫入錨點);CLEAR = 全程 < 5°。異常結束時取「通往失事的那一段」
    (見 attitude_events)。信心高(124 實測 + 韌體 AngErr 150.8 vs 150 重現),門檻與正常飛行最大 3.6° 有 ≥ 1.4x 間隔。
    """
    det = "attitude_loss_of_control"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    core = _attitude_core(ctx, iv, a)
    if core is None:
        return _na(det, "no_data", "缺少 ANG / ATT 姿態資料,無法計算姿態誤差。")
    s, ae, tilt, ev, exempt = core
    t = s["t"]
    analysed = _analysed_s(t, np.isfinite(ae), s["gap"])      # 特技段已是 NaN;NaN 資料也不算可判讀
    ev_warn, ev_loss, ev_info, ev_inv = ev["warn"], ev["loss"], ev["info"], ev["inv"]
    earlier = ev["earlier_warn"]
    t_warn = ev_warn[0] if ev_warn else None
    t_loss = ev_loss[0] if ev_loss else None
    t_inv = ev_inv[0] if ev_inv else None
    if not (ev_warn or ev_info or earlier) and analysed < MIN_ANALYSED_S:
        why = "特技模式 / FLIP 期間" if exempt.mean() > 0.5 else "可判讀時間太短"
        return _na(det, "commanded_mode" if exempt.mean() > 0.5 else "too_short",
                   f"姿態誤差未判讀({why},可判讀 {analysed:.1f} s)。", analysed_s=_f(analysed, 1))

    t_disarm = a.t_disarm if a.t_disarm is not None else iv.t1
    anchor_mode = _apply_attitude_anchors(a, t_warn, t_loss, t_inv)
    ctx_s, tho_w, agl_w = "normal", None, None
    if t_warn is not None:
        ctx_s, tho_w, _ = _throttle_context(ctx, t_warn - 0.5, t_warn + AE_SUSTAIN_S, iv)
        agl_w = _agl_at(ctx, iv, t_warn)
        if ctx_s == "ground_idle" and agl_w is not None and agl_w > GROUND_AGL_M:
            ctx_s = "air_low_throttle"          # 油門在底但離地很高:空中油門崩潰,不是躺在地上
    finite = np.isfinite(ae)
    # 基準:事件(t_warn / 資訊級 / 已恢復的越線)− 5 s 以前(沒有事件就整段)的最大誤差 = 這架飛機「正常」的追蹤誤差
    t_evt = t_warn if t_warn is not None else (ev_info[0] if ev_info else (earlier[-1][0] if earlier else None))
    lim = (t_evt - H_WINDOW_END_S) if t_evt is not None else iv.t1
    pre = finite & (t <= lim) & ~_in_spans(t, [(r[0] - 1.0, r[1] + 1.0) for r in earlier])   # 較早的越線不算「正常」
    ae_normal = float(np.max(ae[pre])) if pre.any() else None
    ae_max = float(np.max(ae[finite])) if finite.any() else None
    vals = {"source": s["src"], "n_samples": int(t.size), "analysed_s": _f(analysed, 1),
            "t_info": _f(ev_info[0] if ev_info else None), "t_warn": _f(t_warn), "t_loss": _f(t_loss),
            "t_inverted": _f(t_inv), "t_tilt120": _f(ev["deep"][0] if ev["deep"] else None),
            "ae_max_normal": _f(ae_normal, 2), "ae_max": _f(ae_max, 1),
            "thr": {"info": AE_INFO_DEG, "warn": AE_WARN_DEG, "loss": AE_LOSS_DEG, "sustain_s": AE_SUSTAIN_S,
                    "recover_s": AE_RECOVER_S},
            "anchor_mode": anchor_mode,
            "episode_rule": "first_crossing" if _crash_end(a, iv)[0] is None else "crash_episode",
            "earlier_warn_runs": [[_f(r[0]), _f(r[1])] for r in earlier[-MAX_LISTED:]]}
    earlier_zh = (f"另有 {len(earlier)} 次較早的姿態誤差 > 10°(最晚 {earlier[-1][0]:.2f} s)已恢復(ae < 5° 持續 "
                  f"{AE_RECOVER_S:.0f} s),不算這次失事的起點。") if earlier else ""
    if t_warn is not None:
        drv, d_des, d_act = _driver(s, t_warn)
        vals.update(driver=drv, d_des_deg=_f(d_des, 1), d_act_deg=_f(d_act, 1),
                    tho_at_warn=_f(tho_w, 3), agl_at_warn_m=_f(agl_w, 2),
                    lead_time_s=_f(t_disarm - t_warn, 2), t_warn_alarm=_f(t_warn + AE_SUSTAIN_S),
                    t_loss_alarm=_f(t_loss + AE_SUSTAIN_S if t_loss is not None else None))
        # 區間內的峰值(失控之後 ANG 仍會記到上鎖,恆 ~150°)
        ae_pk = float(np.nanmax(ae[(t >= t_warn) & finite])) if (finite & (t >= t_warn)).any() else None
        vals["ae_peak"] = _f(ae_pk, 1)
        drv_zh = {"demand": (f"起點是控制器的傾斜指令被拉高(指令變化 {d_des:.0f}° 而機身只動 {d_act:.0f}°),"
                             "不是機身先壞掉"),
                  "vehicle": (f"起點是機身實際姿態先偏離指令(機身變化 {d_act:.0f}° vs 指令 {d_des:.0f}°),"
                              "常見於推力不足 / 單邊失去推力 / 外力"),
                  "indeterminate": ""}[drv] if d_des is not None else ""
        inv_zh = f",{t_inv:.2f} s 傾角超過 90° 倒扣" if t_inv is not None else ""
        # 預警前 5 s 以前沒有資料(例:解鎖後幾秒內就翻覆)→ 沒有「正常」基準可比
        norm_zh = (f"此前正常飛行最大誤差 {ae_normal:.1f}°" if ae_normal is not None
                   else "預警前沒有足夠的正常飛行資料可當基準")
        if t_loss is not None:
            text = (f"姿態誤差(目標與實際機身朝向夾角)於 {t_warn:.2f} s 起超過 10°(預警),{t_loss:.2f} s 超過 30°(失控){inv_zh}。"
                    f"{drv_zh}。{norm_zh};預警比上鎖早 {t_disarm - t_warn:.1f} s。{earlier_zh}")
            sev = "critical"
        else:
            text = (f"姿態誤差於 {t_warn:.2f} s 起超過 10°(最大 {ae_pk:.1f}°),未達 30° 失控門檻。{drv_zh}。"
                    f"{norm_zh}。{earlier_zh}")
            sev = "high"
        return Finding(det, DOMAIN, HIT, sev, float(t_warn), float(t_warn + AE_SUSTAIN_S), ctx_s, vals,
                       text.replace("。。", "。"))
    # 門檻 5° / 10° 的校準來源:00000124 兩段正常飛行(最大 3.6°),只有一次失事樣本
    cal_zh = "門檻依 00000124 兩段正常飛行(最大 3.6°)校準,僅一次失事樣本"
    base_zh = f"此前追蹤誤差最大 {ae_normal:.1f}°;" if ae_normal is not None else ""
    if ev_info:
        return Finding(det, DOMAIN, SOFT, "medium", float(ev_info[0]), float(ev_info[0] + AE_SUSTAIN_S), "minor", vals,
                       f"姿態誤差於 {ev_info[0]:.2f} s 起超過 5°(最大 {ae_max:.1f}°)但未達 10° 預警;"
                       f"{base_zh}{cal_zh}。請留意追蹤誤差是否持續擴大。{earlier_zh}")
    if earlier:
        # 只有失事前就已恢復的越線(例:較早的陣風):弱證據,不是這次失事的起點,也不寫入 t_warn 錨點
        r0 = earlier[-1]
        ae_r = float(np.nanmax(ae[(t >= r0[0]) & (t <= r0[1])])) if ((t >= r0[0]) & (t <= r0[1])).any() else None
        pk = f"(最大 {ae_r:.1f}°)" if ae_r is not None and np.isfinite(ae_r) else ""
        return Finding(det, DOMAIN, SOFT, "medium", float(r0[0]), float(r0[0] + AE_SUSTAIN_S), "recovered", vals,
                       f"{r0[0]:.2f} s 起姿態誤差曾超過 10°{pk},但之後恢復(ae < 5° 持續 {AE_RECOVER_S:.0f} s),"
                       f"失事前沒有再越線:這不是失事的起點(撞擊在先 / 外力型),只列為參考;{cal_zh}。")
    peak = f"最大 {ae_max:.1f}°" if ae_max < AE_INFO_DEG else f"短暫尖峰最大 {ae_max:.1f}°,未持續超過 5°"
    return Finding(det, DOMAIN, CLEAR, "info", None, None, "normal", vals,
                   f"解鎖期間姿態誤差{peak},飛機有照飛控指令傾斜,沒有姿態失控跡象。")


# =====================================================================
#                      2. rate_demand_not_achieved
# =====================================================================

def rolling_window_stats(t: np.ndarray, d: np.ndarray, x: np.ndarray, n: int):
    """長度 n 的滾動窗:回傳 (t_start, t_end, rmsD, gain, n_bad),各長 N−n+1。

    rmsD = sqrt(Σd²/n);gain = Σd·x / max(Σd², 1e-9)(用「實際」對「指令」做迴歸的斜率:
    機身跟得上約 0.5–1,不動 ≈ 0,往反方向轉 < 0)。n_bad = 窗內非有限值個數(該窗無效)。
    """
    N = d.size
    if n < 2 or N < n:
        z = np.zeros(0)
        return z, z, z, z, z.astype(int)
    bad = ~(np.isfinite(d) & np.isfinite(x))
    d0 = np.where(bad, 0.0, d)
    x0 = np.where(bad, 0.0, x)
    c_dd = np.concatenate(([0.0], np.cumsum(d0 * d0)))
    c_dx = np.concatenate(([0.0], np.cumsum(d0 * x0)))
    c_bad = np.concatenate(([0], np.cumsum(bad.astype(int))))
    sdd = c_dd[n:] - c_dd[:-n]
    sdx = c_dx[n:] - c_dx[:-n]
    nbad = c_bad[n:] - c_bad[:-n]
    return t[: N - n + 1], t[n - 1:], np.sqrt(np.maximum(sdd, 0.0) / n), sdx / np.maximum(sdd, 1e-9), nbad


def _window_count(flag: np.ndarray, n: int) -> np.ndarray:
    """滾動窗 (長 n) 內 flag 為真的個數,長 N−n+1。"""
    c = np.concatenate(([0], np.cumsum(flag.astype(int))))
    return c[n:] - c[:-n]


@guarded("rate_demand_not_achieved", DOMAIN)
def rate_demand_not_achieved(ctx: Ctx, a: Anchors):
    """角速度指令未達成:roll / pitch 0.5 s 窗 rmsD ≥ 30°/s 且 gain < 0.3,持續 ≥ 0.2 s(兩軸 OR)。

    HIT = 持續成立;SOFT = 只成立不到 0.2 s;CLEAR = 資料足夠且沒有。context:ground_idle(窗內 ThO 中位數
    < 0.65 × ThH,躺在地上用不上力矩;見模組 docstring)vs air。信心中:只有 124 一個正樣本、pitch 軸間隔只有 1.33x(正常最大 22.5)。
    多段成立時:run 依 EPISODE_GAP_S 併成事件,挑「這次失事的那一段」(choose_episode:持續成立優先,再看與失事參考時間的距離);
    t_onset = 該段第一個持續成立 run 的起點(第一個成立視窗的結束時間)。其他段列在 values['other_episodes']。
    """
    det = "rate_demand_not_achieved"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    tb = ctx.table("RATE")
    need = ("RDes", "R", "PDes", "P")
    if tb is None or not all(c in tb for c in need):
        return _na(det, "no_data", "缺少 RATE 表(RDes / R / PDes / P),無法比對角速度指令與實際。")
    s = _slice(tb, iv.t0, iv.t1, need)
    t = s["t"]
    if t.size < 50:
        return _na(det, "too_short", "解鎖期間 RATE 資料太少,無法判讀。")
    dt = _dt_med(t)
    n = max(10, int(round(RATE_WINDOW_S / dt)))
    max_span = RATE_WINDOW_SPAN_MAX * RATE_WINDOW_S
    exempt = _exempt_mask(ctx, iv, t)
    ex_cnt = _window_count(exempt, n) if t.size >= n else np.zeros(0, dtype=int)
    tol = 0.5 * dt
    stats = {}           # axis -> (t_end, rmsD, 有效窗)
    analysed = 0.0
    runs_all = []        # 每個成立 run:{axis, t0, t1, sustained, rms_peak, gain_min, gain_at_peak, dur}
    for axis, (dk, ak) in (("roll", ("RDes", "R")), ("pitch", ("PDes", "P"))):
        ts, te, rms, gain, nbad = rolling_window_stats(t, s[dk], s[ak], n)
        if not te.size:
            continue
        ok = ((te - ts) <= max_span) & (nbad == 0) & (ex_cnt == 0) & (ts >= iv.t0) & (te >= iv.t0 + ARM_SKIP_S)
        analysed = max(analysed, _analysed_s(te, ok, GAP_FAST_S))
        stats[axis] = (te, rms, ok)
        cond = ok & (rms >= RATE_RMSD_MIN) & (gain < RATE_GAIN_MAX)
        for (t0r, t1r, i0, i1) in _runs(cond, te, 0.0, GAP_FAST_S):
            runs_all.append({"axis": axis, "t0": t0r, "t1": t1r, "dur": t1r - t0r,
                             "sustained": (t1r - t0r) >= RATE_SUSTAIN_S - tol,
                             "rms_peak": float(np.max(rms[i0:i1 + 1])), "gain_min": float(np.min(gain[i0:i1 + 1])),
                             "gain_at_peak": float(gain[i0 + int(np.argmax(rms[i0:i1 + 1]))])})
    # run → 事件(episode);代表 run = 段內第一個持續成立的 run(沒有就第一個 run)
    eps = []
    for grp in group_episodes(runs_all):
        sus = [r for r in grp if r["sustained"]]
        rep = sus[0] if sus else grp[0]
        eps.append({"t0": rep["t0"], "t1": max(r["t1"] for r in grp), "level": 1 if sus else 0, "run": rep,
                    "first": grp[0]["t0"], "rms_peak": max(r["rms_peak"] for r in grp)})
    ref = _crash_ref(a)
    chosen = choose_episode(eps, ref, lambda e, d: (e["level"], -d))
    others = [e for e in eps if e is not chosen]
    # 「正常」基準:觸發前 5 s 以前、且不在其他事件內的最大指令 rms(沒有觸發就整段)
    t_lim = (chosen["t0"] - H_WINDOW_END_S) if chosen is not None else np.inf
    other_spans = [(e["first"] - RATE_WINDOW_S, e["t1"] + RATE_WINDOW_S) for e in others]   # 含窗口拖尾
    max_rms = {}
    for ax, (te, r, o) in stats.items():
        mm = o & (te <= t_lim) & ~_in_spans(te, other_spans)
        max_rms[ax] = float(np.max(r[mm])) if mm.any() else None
    vals = {"window_s": RATE_WINDOW_S, "n_window": n, "analysed_s": _f(analysed, 1),
            "max_rmsd_normal_roll": _f(max_rms.get("roll"), 1), "max_rmsd_normal_pitch": _f(max_rms.get("pitch"), 1),
            "n_episodes": len(eps), "crash_ref": _f(ref),
            "other_episodes": [{"t": _f(e["t0"]), "t_end": _f(e["t1"]), "level": "HIT" if e["level"] else "SOFT",
                                "axis": e["run"]["axis"], "rmsd_peak": _f(e["rms_peak"], 1)} for e in others[:MAX_LISTED]],
            "thr": {"rmsd_min": RATE_RMSD_MIN, "gain_max": RATE_GAIN_MAX, "sustain_s": RATE_SUSTAIN_S,
                    "episode_gap_s": EPISODE_GAP_S}}
    if chosen is not None:
        info = chosen["run"]
        t_on = info["t0"]
        ctxs, tho, thh = _throttle_context(ctx, t_on - RATE_WINDOW_S, t_on + RATE_SUSTAIN_S, iv)
        vals.update(axis=info["axis"], rmsd_peak=_f(info["rms_peak"], 1), gain_min=_f(info["gain_min"], 2),
                    gain_at_peak=_f(info["gain_at_peak"], 2), duration_s=_f(info["dur"], 2),
                    tho=_f(tho, 3), thh=_f(thh, 3), t_run_end=_f(t_on + info["dur"]), t_episode_end=_f(chosen["t1"]))
        ax_zh = {"roll": "橫滾", "pitch": "俯仰"}[info["axis"]]
        normal = max([v for v in max_rms.values() if v is not None] + [0.0])
        oth = (f"另有 {len(others)} 段其他時間的未達成(最早 {others[0]['t0']:.2f} s,見 values.other_episodes)。"
               if others else "")
        if chosen["level"]:
            if ctxs == "ground_idle":
                where = "此時油門已在最低(飛機躺在地上,馬達沒有力矩餘裕),不是馬達失效的樣子"
            elif ctxs == "air":
                where = "發生在空中,請檢查馬達 / 槳 / 電調是否出力"
            else:
                where = ""
            text = (f"{t_on:.2f} s 起{ax_zh}軸角速度指令 rms {info['rms_peak']:.0f}°/s(門檻 30)但機身幾乎沒跟上"
                    f"(回歸斜率 {info['gain_at_peak']:.2f} < 0.3);{where}。此前正常飛行的指令 rms 最大 {normal:.0f}°/s。{oth}")
            return Finding(det, DOMAIN, HIT, "high", float(t_on), float(t_on + RATE_SUSTAIN_S), ctxs, vals,
                           text.replace(";。", "。"))
        return Finding(det, DOMAIN, SOFT, "medium", float(t_on), float(t_on), ctxs, vals,
                       f"{t_on:.2f} s 短暫(< 0.2 s){ax_zh}軸角速度指令未達成(rms {info['rms_peak']:.0f}°/s),證據偏弱。{oth}")
    if analysed < MIN_ANALYSED_S:
        return _na(det, "too_short", f"可判讀時間僅 {analysed:.1f} s,不足以判定角速度追蹤是否正常。", analysed_s=_f(analysed, 1))
    mr, mp = max_rms.get("roll"), max_rms.get("pitch")
    detail = f"0.5 s 窗指令 rms 最大 roll {mr:.0f} / pitch {mp:.0f} °/s," if mr is not None and mp is not None else ""
    return Finding(det, DOMAIN, CLEAR, "info", None, None, "normal", vals,
                   f"角速度指令都有被跟上:{detail}未出現「指令 ≥ 30°/s 但機身不動」。")


# =====================================================================
#                      3. rate_output_saturation
# =====================================================================

def moving_mean(t: np.ndarray, x: np.ndarray, n: int):
    """n 樣本滑動平均:回傳 (t_start, t_end, mean, n_bad)。"""
    N = x.size
    if n < 1 or N < n:
        z = np.zeros(0)
        return z, z, z, z.astype(int)
    bad = ~np.isfinite(x)
    x0 = np.where(bad, 0.0, x)
    c = np.concatenate(([0.0], np.cumsum(x0)))
    cb = np.concatenate(([0], np.cumsum(bad.astype(int))))
    return t[: N - n + 1], t[n - 1:], (c[n:] - c[:-n]) / n, cb[n:] - cb[:-n]


def _count_in_trailing(t: np.ndarray, flag: np.ndarray, span: float):
    """每個樣本往前 span 秒(含自己)內 flag 為真的個數,與該窗內第一個為真樣本的索引(沒有 = −1)。"""
    c = np.concatenate(([0], np.cumsum(flag.astype(int))))
    j0 = np.searchsorted(t, t - span, side="right")        # 窗從 j0 起算(不含 t − span 本身)
    cnt = c[1:] - c[j0]
    fl = np.flatnonzero(flag)
    if fl.size:
        p = np.searchsorted(fl, j0)
        first = np.where(p < fl.size, fl[np.clip(p, 0, fl.size - 1)], -1)
    else:
        first = np.full(t.size, -1)
    return cnt, first


@guarded("rate_output_saturation", DOMAIN)
def rate_output_saturation(ctx: Ctx, a: Anchors):
    """姿態控制輸出飽和:x = max(|ROut|, |POut|)(偏航 YOut 不算:正常偏航操作就到 0.5)。

    SOFT:0.25 s 滑動平均 ≥ 0.20(正常最大 0.118);HIT:x > 0.5 累計 ≥ 0.1 s(任一 2 s 窗;正常 ≤ 0.255);
    CRITICAL(仍是 HIT):x ≥ 0.9 至少 2 個樣本在 0.1 s 內。各層級成立的 run 依 EPISODE_GAP_S 併成同一次飽和事件,
    t_onset = 事件內最早的層級起點;多次事件時挑「這次失事的那一段」(choose_episode:層級優先,再看與失事參考時間的距離),
    其他事件列在 values['other_episodes'],不會把較早無關的層級起點混進來。
    context:'air'(ThO ≥ 0.65 ThH)/ 'ground_idle'(油門已低:是沒有力矩餘裕,不是推力不足)。信心中。
    """
    det = "rate_output_saturation"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    tb = ctx.table("RATE")
    if tb is None or "ROut" not in tb or "POut" not in tb:
        return _na(det, "no_data", "缺少 RATE.ROut / POut,無法判讀控制輸出是否飽和。")
    s = _slice(tb, iv.t0, iv.t1, ("ROut", "POut"))
    t = s["t"]
    if t.size < 50:
        return _na(det, "too_short", "解鎖期間 RATE 資料太少,無法判讀。")
    dt = _dt_med(t)
    exempt = _exempt_mask(ctx, iv, t)
    xr, xp = np.abs(s["ROut"]), np.abs(s["POut"])
    x = np.fmax(xr, xp)
    use = (t >= iv.t0 + ARM_SKIP_S) & ~exempt & np.isfinite(x)
    analysed = _analysed_s(t, use, GAP_FAST_S)
    n = max(2, int(round(SAT_MEAN_WINDOW_S / dt)))
    max_span = 1.4 * SAT_MEAN_WINDOW_S
    items = []                    # 各層級成立的 run:{tier 1 SOFT / 2 HIT / 3 CRIT, t0 起點, t1, t_alarm, ...}
    m25 = {}                      # axis -> (t_end, 0.25 s 平均, 有效窗)
    if t.size >= n:
        win_bad = _window_count(~use, n)
        for axis, xa in (("roll", xr), ("pitch", xp)):
            ts, te, mean, nbad = moving_mean(t, xa, n)
            ok = ((te - ts) <= max_span) & (nbad == 0) & (win_bad == 0)
            m25[axis] = (te, mean, ok)
            for (a0, a1, i0, _) in _runs(ok & (mean >= SAT_MEAN_MIN), te, 0.0, GAP_FAST_S):
                items.append({"tier": 1, "t0": a0, "t1": a1, "t_alarm": a0, "axis": axis, "mean": float(mean[i0])})
    # HIT 層:x > 0.5 累計 ≥ 0.1 s(取樣數換算)落在 2 s 窗內;CRIT:x ≥ 0.9 至少 2 個樣本在 0.1 s 內。
    # 起點 = 第一次確認時,尾窗內第一個越線樣本的時間;確認時間 = t_alarm。
    n_hit = max(2, int(round(SAT_HIT_TIME_S / dt)))
    for tier, thr_mask, span, need in ((2, use & (x > SAT_RAW_HIT), SAT_HIT_WINDOW_S, n_hit),
                                       (3, use & (x >= SAT_RAW_CRIT), SAT_CRIT_WINDOW_S, SAT_CRIT_SAMPLES)):
        cnt, first = _count_in_trailing(t, thr_mask, span)
        fl = np.flatnonzero(thr_mask)
        for (a0, _, i0, i1) in _runs(use & (cnt >= need), t, 0.0, None):
            # 結束 = run 內最後一個越線樣本(尾窗計數會讓 run 多拖 span 秒,不能拿來併段)
            last = fl[np.searchsorted(fl, i1, side="right") - 1]
            items.append({"tier": tier, "t0": float(t[first[i0]]), "t1": float(t[last]), "t_alarm": a0})
    # 層級 run → 同一次飽和事件(間隔 ≤ EPISODE_GAP_S);事件起點 = 段內最早的層級起點
    eps = []
    for grp in group_episodes(items):
        top = max(it["tier"] for it in grp)
        first_of = {k: min((it for it in grp if it["tier"] == k), key=lambda it: it["t0"], default=None) for k in (1, 2, 3)}
        eps.append({"t0": min(it["t0"] for it in grp), "t1": max(it["t1"] for it in grp), "tier": top,
                    "t_alarm": min(it["t_alarm"] for it in grp if it["tier"] == top), "first": first_of})
    ref = _crash_ref(a)
    chosen = choose_episode(eps, ref, lambda e, d: (e["tier"], -d))
    others = [e for e in eps if e is not chosen]
    t_on = chosen["t0"] if chosen is not None else None
    # 「正常」基準:事件前 5 s 以前、且不在其他飽和事件內的最大值(沒有事件就整段)
    t_lim = (t_on - H_WINDOW_END_S) if t_on is not None else np.inf
    other_spans = [(e["t0"] - SAT_MEAN_WINDOW_S, e["t1"] + SAT_MEAN_WINDOW_S) for e in others]   # 含窗口拖尾
    m25_norm = {}
    for ax, (te, mean, ok) in m25.items():
        mm_ = ok & (te <= t_lim) & ~_in_spans(te, other_spans)
        m25_norm[ax] = float(np.max(mean[mm_])) if mm_.any() else None
    base = use & (t <= t_lim) & ~_in_spans(t, other_spans)
    raw_norm = float(np.max(x[base])) if base.any() else None
    tier_zh = {1: "SOFT", 2: "HIT", 3: "CRIT"}
    vals = {"analysed_s": _f(analysed, 1), "m25_max_normal_roll": _f(m25_norm.get("roll"), 3),
            "m25_max_normal_pitch": _f(m25_norm.get("pitch"), 3), "raw_max_normal": _f(raw_norm, 3),
            "n_episodes": len(eps), "crash_ref": _f(ref),
            "other_episodes": [{"t": _f(e["t0"]), "t_end": _f(e["t1"]), "tier": tier_zh[e["tier"]]} for e in others[:MAX_LISTED]],
            "thr": {"mean_soft": SAT_MEAN_MIN, "raw_hit": SAT_RAW_HIT, "raw_crit": SAT_RAW_CRIT,
                    "episode_gap_s": EPISODE_GAP_S}}
    mm = max([v for v in m25_norm.values() if v is not None] + [0.0])
    if chosen is None:
        if analysed < MIN_ANALYSED_S:
            return _na(det, "too_short", f"可判讀時間僅 {analysed:.1f} s,不足以判定控制輸出是否飽和。",
                       analysed_s=_f(analysed, 1))
        return Finding(det, DOMAIN, CLEAR, "info", None, None, "normal", vals,
                       f"姿態控制輸出沒有飽和:0.25 s 平均 |ROut|/|POut| 最大 {mm:.2f}(門檻 0.20),"
                       f"原始值最大 {raw_norm:.2f}(門檻 0.5)。")
    t_al = float(chosen["t_alarm"])
    f1, f2, f3 = (chosen["first"][k] for k in (1, 2, 3))
    ctxs, tho, thh = _throttle_context(ctx, t_on - 0.5, t_on + 0.5, iv)
    seg = use & (t >= chosen["t0"] - SAT_MEAN_WINDOW_S) & (t <= chosen["t1"])
    raw_max = float(np.max(x[seg])) if seg.any() else None
    vals.update(t_soft=_f(f1["t0"] if f1 else None), t_hit=_f(f2["t0"] if f2 else None),
                t_crit=_f(f3["t0"] if f3 else None), t_episode_end=_f(chosen["t1"]), tho=_f(tho, 3), thh=_f(thh, 3),
                raw_max=_f(raw_max, 3), soft_axis=f1["axis"] if f1 else None, soft_mean=_f(f1["mean"] if f1 else None, 3))
    if ctxs == "air":
        # 飽和通常是失控後段的結果(124:SOFT 比姿態預警晚 0.13 s、碰上限晚 2.5 s);只有早於預警才指向推力不足
        rel = (f"比姿態預警({a.t_warn:.2f} s){'早' if t_on < a.t_warn else '晚'} {abs(t_on - a.t_warn):.1f} s;"
               if a.t_warn is not None else "")
        ground = (f"發生在空中;{rel}輸出飽和通常是失控後段的結果,只有早於姿態預警才指向馬達 / 槳 / 電調出力不足"
                  "(沒有電調遙測時無法直接確認)")
    else:
        ground = {"ground_idle": "發生在油門已壓到最低時(沒有力矩餘裕,不代表推力不足)", "unknown": ""}.get(ctxs, "")
    norm_txt = f"此前正常飛行 0.25 s 平均最大 {mm:.2f}" if m25_norm else ""
    oth = (f"另有 {len(others)} 段其他時間的輸出偏大(最早 {others[0]['t0']:.2f} s,見 values.other_episodes)"
           if others else "")
    if chosen["tier"] >= 2:
        # 各層級各自的起點(t_onset 是同一次事件最早的層級,不是碰到上限的時間)
        steps = []
        if f1 is not None:
            steps.append(f"{f1['t0']:.2f} s 起 0.25 s 平均 ≥ 0.20")
        if f2 is not None:
            steps.append(f"{f2['t0']:.2f} s 起超過 0.5")
        if f3 is not None:
            steps.append(f"{f3['t0']:.2f} s 碰到上限(≥ 0.9)")
        head = f"橫滾 / 俯仰控制輸出飽和:{'、'.join(steps)},原始值最大 {raw_max:.2f}"
        text = _sentences(f"{head};{ground}" if ground else head, norm_txt, oth)
        return Finding(det, DOMAIN, HIT, "critical" if f3 is not None else "high", t_on, t_al, ctxs, vals, text)
    head = f"{t_on:.2f} s 起{'橫滾' if f1['axis'] == 'roll' else '俯仰'}控制輸出 0.25 s 平均達 {f1['mean']:.2f}(門檻 0.20)"
    text = _sentences(f"{head};{ground}" if ground else head, norm_txt, oth)
    return Finding(det, DOMAIN, SOFT, "medium", t_on, t_al, ctxs, vals, text)


# =====================================================================
#                      4. lean_without_pilot
# =====================================================================

def _hover_throttle(ctx: Ctx) -> float | None:
    """懸停油門 ThH:解鎖期間 CTUN.ThH 中位數,沒有就用 MOT_THST_HOVER。"""
    sh = ctx.series("CTUN.ThH")
    if sh is not None:
        m = ctx.armed_mask(sh[0])
        v = sh[1][m] if m.any() else sh[1]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            return float(np.median(v))
    p = ctx.param("MOT_THST_HOVER", None)
    return float(p) if p else None


def _ground_idle_state(ctx: Ctx, t: np.ndarray, tho: np.ndarray) -> np.ndarray:
    """CTUN 格點上的「地面怠速」:ThO ≤ 0.03,或 IDLE_HOLD_S 內碰過 ≤ 0.03 且目前仍 < 0.65 ThH。

    CTUN 只有 10 Hz;觸地彈跳時 ThO 在 0.04–0.16 與 0 之間跳(124 的 1255.24–1255.84 s),
    只看當下一格會讓怠速狀態「閃爍」並把 lean 的起點延後。
    """
    idle = np.isfinite(tho) & (tho <= THO_IDLE)
    c = np.concatenate(([0], np.cumsum(idle.astype(int))))
    j0 = np.searchsorted(t, t - IDLE_HOLD_S, side="right")
    recent = (c[1:] - c[j0]) > 0
    thh = _hover_throttle(ctx)
    low = (tho < AIR_THO_RATIO * thh) if thh else idle
    return idle | (recent & low)


def _param(ctx: Ctx, name: str, defaults: dict) -> float:
    v = ctx.param(name, None)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = None
    return v if v is not None and np.isfinite(v) and v > 0 else float(defaults[name])


def brake_seconds(ctx: Ctx) -> tuple[float, float]:
    """(LOITER / FLOWHOLD, POSHOLD) 搖桿回中或切入模式後的煞車期(秒),依參數(沒有就用 ArduCopter 預設)。

    LOITER:LOIT_BRK_DELAY + LOIT_SPEED / LOIT_BRK_ACCEL(全速煞停)+ LOIT_BRK_ACCEL / LOIT_BRK_JERK(加速度爬升);
    POSHOLD:煞車角以 PHLD_BRAKE_RATE 升到 PHLD_BRAKE_ANGLE 再降回(2 × 角度 / 速率)+ 轉 Loiter 的混合 1.5 s。
    裁在 [AIR_BRAKE_MIN_S, AIR_BRAKE_MAX_S]。124:4.6 s / 9.0 s;預設參數:6.5 s / 9.0 s。
    """
    lo = (_param(ctx, "LOIT_BRK_DELAY", LOIT_DEFAULTS) + _param(ctx, "LOIT_SPEED", LOIT_DEFAULTS) / _param(ctx, "LOIT_BRK_ACCEL", LOIT_DEFAULTS)
          + _param(ctx, "LOIT_BRK_ACCEL", LOIT_DEFAULTS) / _param(ctx, "LOIT_BRK_JERK", LOIT_DEFAULTS))
    ph = 2.0 * (_param(ctx, "PHLD_BRAKE_ANGLE", PHLD_DEFAULTS) / 100.0) / _param(ctx, "PHLD_BRAKE_RATE", PHLD_DEFAULTS) \
        + PHLD_LOITER_BLEND_S
    return (float(np.clip(lo, AIR_BRAKE_MIN_S, AIR_BRAKE_MAX_S)), float(np.clip(ph, AIR_BRAKE_MIN_S, AIR_BRAKE_MAX_S)))


def _since_release(ctx: Ctx, rc: dict, t: np.ndarray, tr1: float, tr2: float) -> np.ndarray:
    """每個樣本距離「最後一次 roll / pitch 搖桿離開中立(或資料無效)/ 切換模式」的秒數(都沒有 → inf)。"""
    c1, c2 = rc["C1"], rc["C2"]
    off = ~(np.isfinite(c1) & np.isfinite(c2)) | (np.abs(c1 - tr1) > STICK_CENTER_US) | (np.abs(c2 - tr2) > STICK_CENTER_US)
    evs = [np.asarray(rc["t"], dtype=float)[off]]
    tm, nums, _ = ctx.mode_rows()
    if len(tm):
        nums = np.asarray(nums)
        chg = np.concatenate(([True], nums[1:] != nums[:-1]))
        evs.append(np.asarray(tm, dtype=float)[chg])
    e = np.sort(np.concatenate(evs))
    if not e.size:
        return np.full(t.size, np.inf)
    idx = np.searchsorted(e, t, side="right") - 1
    return np.where(idx >= 0, t - e[np.clip(idx, 0, None)], np.inf)


def _agl_series(ctx: Ctx, iv: ArmedInterval, t: np.ndarray, t_valid_end: float | None) -> np.ndarray:
    """每個樣本的估計離地高度(m;CTUN.Alt − 解鎖時地面高度);沒有資料、資料過舊或撞擊之後(高度是垃圾)→ NaN。"""
    alt, g0 = ctx.series("CTUN.Alt"), ctx.ground_alt(iv)
    if alt is None or g0 is None or not np.isfinite(g0):
        return np.full(t.size, np.nan)
    v, ok = _zoh(alt[0], alt[1], t)
    out = np.where(ok & np.isfinite(v), v - g0, np.nan)
    if t_valid_end is not None:
        out = np.where(t > t_valid_end, np.nan, out)
    return out


def lean_runs(t, lean, centered, idle, air_ok, tilt_act, exempt, t_arm, max_gap):
    """搖桿置中卻有大傾角指令的 run(純數值,方便單元測試)。

    GROUND:idle(油門已低、馬達起轉、離地 ≤ 1.5 m,由呼叫端算好):lean ≥ 8° 持續 0.3 s;
    AIR:air_ok(位置模式、不在地面怠速、已過煞車期,由呼叫端算好):lean ≥ 12° 持續 0.5 s。
    機身實際傾角 > 45°(已翻覆)與特技模式的樣本略過。回傳 (ground_runs, air_runs),格式同 _runs。
    """
    tol = 0.5 * _dt_med(t)
    base = centered & (tilt_act <= FLIPPED_TILT_DEG) & ~exempt
    g = base & idle & (lean >= GROUND_LEAN_DEG)
    ar = base & air_ok & (t >= t_arm + ARM_SKIP_S) & (lean >= AIR_LEAN_DEG)
    return (_runs(g, t, max(GROUND_LEAN_S - tol, 0.0), max_gap),
            _runs(ar, t, max(AIR_LEAN_S - tol, 0.0), max_gap))


@guarded("lean_without_pilot", DOMAIN)
def lean_without_pilot(ctx: Ctx, a: Anchors):
    """搖桿置中(±30 µs)卻由飛控下大傾角指令:地面怠速 ≥ 8° / 空中(LOITER 類位置模式、煞車期之後)≥ 12°。

    GROUND = ThO 在底(或 0.5 s 內碰過底且仍 < 0.65 ThH)、馬達起轉、離地 ≤ 1.5 m(有高度資料時;撞擊後高度不可信 → 不檢查);
    油門在底但離地 > 1.5 m 的樣本(空中油門崩潰 / 大幅下降)兩種都不判,只記在 values['idle_above_ground_s']。
    AIR = ThO > 0.03 且不在怠速、位置模式,並排除搖桿回中 / 切入模式後的煞車期(brake_seconds)。
    t_onset = 代表事件的起點(run 依 EPISODE_GAP_S 併段,choose_episode:與失事參考時間最近,同距離地面優先);
    其他段列在 values['other_episodes']。GROUND 且 ≥ 15° 為 critical(傾角需求 > 15° 會擋住降落偵測),其餘 high。
    context:'ground_idle'(地面怠速)/ 'air'(位置模式空中)。H_LAND_TIPOVER 的權重(規格:「GROUND 模式,或 AIR 且
    agl ≤ 1.5 m」)用 values['agl_at_onset_m'] / values['near_ground'](有高度資料時只看高度)判斷離地高度。
    來源歸因:|atan(√(TAN²+TAE²)/g) − lean| < 4° → 來自位置控制器(lean_from_pos)。信心高(地面):
    124 失事 24.4° vs 正常地面怠速 2.2°(11x);空中 12° vs 煞車期後 5.7°(2.1x;煞車期內單點 9.7°,不判)。
    """
    det = "lean_without_pilot"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    rc = ctx.table("RCIN")
    so = ctx.series("CTUN.ThO")
    s = _att_source(ctx, iv)
    if s is None or rc is None or so is None or "C1" not in rc or "C2" not in rc:
        return _na(det, "no_data", "缺少姿態指令(ANG / ATT)、遙控輸入(RCIN)或油門輸出(CTUN.ThO),無法判斷傾斜指令的來源。")
    t = s["t"]
    exempt = _exempt_mask(ctx, iv, t)
    lean = np.hypot(s["DesRoll"], s["DesPitch"])
    tilt_act = util.tilt_deg(s["Roll"], s["Pitch"])
    t1_, t2_ = _rc_trim(ctx, 1, rc), _rc_trim(ctx, 2, rc)
    c1, v1 = _zoh(rc["t"], rc["C1"], t)
    c2, v2 = _zoh(rc["t"], rc["C2"], t)
    centered = v1 & v2 & (np.abs(c1 - t1_) <= STICK_CENTER_US) & (np.abs(c2 - t2_) <= STICK_CENTER_US)
    tho, vt = _zoh(so[0], so[1], t)
    idle_state = _zoh(so[0], _ground_idle_state(ctx, so[0], so[1]).astype(float), t)[0] > 0.5
    ro = ctx.table("RCOU")
    if ro is not None:
        mc = _motor_cols(ctx, ro)
        pwm = np.max(np.stack([ro[c] for c in mc]), axis=0)
        pw, vp = _zoh(ro["t"], pwm, t)
        spooled = vp & (pw >= SPOOL_PWM_US)
    else:
        spooled = np.ones(t.size, dtype=bool)
    # 離地高度:有資料時,地面怠速判讀只算離地 ≤ 1.5 m 的樣本(撞擊後的高度不可信 → 視為未知)
    hit = a.t_first_hit if (a.t_first_hit is not None and iv.t0 < a.t_first_hit <= iv.t1) else None
    agl_s = _agl_series(ctx, iv, t, hit)
    low_s = ~np.isfinite(agl_s) | (agl_s <= GROUND_AGL_M)
    idle = vt & idle_state & spooled & low_s
    idle_high = vt & idle_state & spooled & ~low_s          # 油門在底但離地 > 1.5 m:不是躺在地上,兩種判讀都不做
    mode = _mode_array(ctx, t)
    air_pos = vt & ~idle_state & (tho > THO_IDLE) & np.isin(mode, AIR_MODES)
    # 煞車期:搖桿回中 / 切入模式後,飛控照設計自己傾斜減速(不是「搖桿置中卻被推翻」)
    brk_lo, brk_ph = brake_seconds(ctx)
    since = _since_release(ctx, rc, t, t1_, t2_)
    settled = since >= np.where(mode == POSHOLD_MODE, brk_ph, brk_lo)
    air_ok = air_pos & settled
    inwin = (t >= iv.t0) & (t <= iv.t1)
    gruns, aruns = lean_runs(t, lean, centered & inwin, idle, air_ok, tilt_act, exempt, iv.t0, s["gap"])
    dt = _dt_med(t)
    # 沒有任何可判讀的相位(沒有地面怠速、也沒有位置模式的空中)→ NA,不能說沒問題
    base = centered & inwin & ~exempt & (tilt_act <= FLIPPED_TILT_DEG)
    air_base = base & air_ok & (t >= iv.t0 + ARM_SKIP_S)
    brake_base = base & air_pos & ~settled & (t >= iv.t0 + ARM_SKIP_S)
    n_idle, n_air = int(np.sum(base & idle)), int(np.sum(air_base))
    n05 = max(1, int(round(AIR_LEAN_S / dt)))
    g_max = _sustained_max(np.where(base & idle, lean, 0.0), max(1, int(round(GROUND_LEAN_S / dt)))) if n_idle else None
    a_max = _sustained_max(np.where(air_base, lean, 0.0), n05) if n_air else None
    b_max = _sustained_max(np.where(brake_base, lean, 0.0), n05) if brake_base.any() else None
    # 資訊級:空中置中搖桿 ≥ 6° 持續 0.5 s 的次數(計數用,不告警)
    n_info = len(_runs(air_base & (lean >= INFO_LEAN_DEG), t, max(AIR_LEAN_S - 0.5 * dt, 0.0), s["gap"]))
    vals = {"source": s["src"], "stick_trim": [_f(t1_, 0), _f(t2_, 0)], "ground_lean_sustained_max": _f(g_max, 1),
            "air_lean_sustained_max": _f(a_max, 1), "air_brake_lean_max": _f(b_max, 1), "n_info_incidents": n_info,
            "idle_s": _f(n_idle * dt, 1), "air_pos_mode_s": _f(n_air * dt, 1),
            "air_brake_phase_s": _f(int(np.sum(brake_base)) * dt, 1),
            "idle_above_ground_s": _f(int(np.sum(base & idle_high)) * dt, 1),
            "brake_window_s": {"loiter": _f(brk_lo, 2), "poshold": _f(brk_ph, 2)},
            "agl_available": bool(np.isfinite(agl_s).any()),
            "thr": {"air": AIR_LEAN_DEG, "ground": GROUND_LEAN_DEG, "crit": GROUND_LEAN_CRIT_DEG,
                    "stick_us": STICK_CENTER_US, "air_modes": list(AIR_MODES), "ground_agl_m": GROUND_AGL_M,
                    "episode_gap_s": EPISODE_GAP_S}}
    # run → 事件(同類型、間隔 ≤ EPISODE_GAP_S 併段),再挑「這次失事的那一段」
    eps = []
    for kind, rr in (("ground_idle", gruns), ("air", aruns)):
        for grp in group_episodes([{"t0": r[0], "t1": r[1], "i0": r[2], "i1": r[3]} for r in rr]):
            idx = np.concatenate([np.arange(g["i0"], g["i1"] + 1) for g in grp])
            eps.append({"kind": kind, "t0": grp[0]["t0"], "t1": max(g["t1"] for g in grp), "idx": idx,
                        "peak": float(np.max(lean[idx]))})
    ref = _crash_ref(a)
    chosen = choose_episode(eps, ref, lambda e, d: (-d, e["kind"] == "ground_idle"))
    others = sorted((e for e in eps if e is not chosen), key=lambda e: e["t0"])
    vals.update(n_episodes=len(eps), crash_ref=_f(ref),
                other_episodes=[{"kind": e["kind"], "t": _f(e["t0"]), "t_end": _f(e["t1"]), "lean_peak": _f(e["peak"], 1)}
                                for e in others[:MAX_LISTED]])
    if chosen is not None:
        kind, t_on, t_end, idx = chosen["kind"], chosen["t0"], chosen["t1"], chosen["idx"]
        peak = chosen["peak"]
        crit_idx = idx[lean[idx] >= GROUND_LEAN_CRIT_DEG]
        t_crit = float(t[crit_idx[0]]) if crit_idx.size and kind == "ground_idle" else None
        # 來源歸因:飛控自己的位置控制器 vs 其他
        from_pos, lean_psc, pos_err = None, None, None
        pn, pe = ctx.table("PSCN"), ctx.table("PSCE")
        if pn is not None and pe is not None and all(k in pn for k in ("TAN", "PN", "TPN")) \
                and all(k in pe for k in ("TAE", "PE", "TPE")) and pn["t"].size >= 2 and pe["t"].size >= 2:
            sc = 0.01 if _unit(ctx, "PSCN.TAN").startswith("cm") else 1.0
            sc_p = 0.01 if _unit(ctx, "PSCN.PN").startswith("cm") else 1.0
            tp = pn["t"]
            # PSCE 依「時間」內插到 PSCN 的時間格(兩表列數可能差幾列,不能用索引對齊)
            tae = np.interp(tp, pe["t"], pe["TAE"])
            e_err = np.interp(tp, pe["t"], pe["PE"] - pe["TPE"])
            ac = np.hypot(pn["TAN"], tae) * sc
            lp = np.degrees(np.arctan(ac / GRAVITY))
            pe_err = np.hypot(pn["PN"] - pn["TPN"], e_err) * sc_p
            m = (tp >= t_on) & (tp <= min(t_end, t_on + 5.0)) & (tp >= pe["t"][0]) & (tp <= pe["t"][-1])
            if m.any():
                lean_i = np.interp(tp[m], t, lean)
                from_pos = bool(np.median(np.abs(lp[m] - lean_i)) < POS_AGREE_DEG)
                lean_psc = float(np.median(lp[m]))
                pos_err = float(np.max(pe_err[m]))
        # 對照:要求的滾轉角速度 vs 實際
        rt = ctx.table("RATE")
        rate_cmp = None
        if rt is not None and "RDes" in rt and "R" in rt:
            ss = _slice(rt, t_on, min(t_end, t_on + 2.0), ("RDes", "R"))
            if ss["t"].size:
                rate_cmp = [_f(np.sqrt(np.mean(ss["RDes"] ** 2)), 0), _f(np.sqrt(np.mean(ss["R"] ** 2)), 0)]
        agl = _agl_at(ctx, iv, t_on) if (hit is None or t_on <= hit) else None
        sev = "critical" if (kind == "ground_idle" and peak >= GROUND_LEAN_CRIT_DEG) else "high"
        # context = kind('ground_idle' / 'air');近地面:有高度資料時只看高度(≤ 1.5 m),沒有才用類型(地面怠速)
        near_ground = (agl <= GROUND_AGL_M) if agl is not None else (kind == "ground_idle")
        vals.update(kind=kind, lean_peak=_f(peak, 1), t_end=_f(t_end), t_crit=_f(t_crit), lean_from_pos=from_pos,
                    lean_psc=_f(lean_psc, 1), pos_err_max_m=_f(pos_err, 2), rate_rms_des_act=rate_cmp,
                    agl_at_onset_m=_f(agl, 2), near_ground=bool(near_ground), mode_at_onset=ctx.mode_at(t_on))
        who = ("且與位置控制器的目標加速度一致(飛控自己的 Loiter 在拉位置)" if from_pos
               else ("(與位置控制器的目標不一致,來源待查)" if from_pos is False else ""))
        oth = (f"另有 {len(others)} 段其他時間的搖桿置中大傾角(最早 {others[0]['t0']:.2f} s,見 values.other_episodes)"
               if others else "")
        if kind == "ground_idle":
            where = f"離地約 {agl:.1f} m、" if agl is not None else "沒有高度資料、只依油門判斷在地面,"
            text = _sentences(f"{t_on:.2f} s 起飛機在地面怠速({where}油門已壓到最低)、搖桿置中,飛控卻要求傾斜 {peak:.0f}°"
                              f"(門檻 8°{',超過 15° 會擋住降落偵測器' if peak >= GROUND_LEAN_CRIT_DEG else ''}){who}",
                              "常見原因是位置估測(VIO / GPS / EKF)在地上漂移;建議降落時用 LAND 模式或觸地立刻上鎖", oth)
        else:
            low = f"(離地僅約 {agl:.1f} m)" if near_ground and agl is not None else ""
            text = _sentences(f"{t_on:.2f} s 起空中{low}搖桿置中且已過煞車期,飛控卻自行要求傾斜 {peak:.0f}°"
                              f"(門檻 12°,僅以 00000124 一位駕駛的兩段正常飛行校準){who}",
                              "請檢查當時位置來源(VIO / GPS)是否跳動或漂移", oth)
        return Finding(det, DOMAIN, HIT, sev, float(t_on), float(t_on + (GROUND_LEAN_S if kind == "ground_idle" else AIR_LEAN_S)),
                       kind, vals, text)
    # 可判讀時間(地面怠速 + 位置模式空中,搖桿置中)不足 MIN_ANALYSED_S → NA,不能說沒問題
    if (n_idle + n_air) * dt < MIN_ANALYSED_S:
        if n_idle == 0 and n_air == 0:
            return _na(det, "no_applicable_phase",
                       "沒有可判讀的階段(沒有地面怠速、也沒有 LOITER / POSHOLD 類位置模式的空中搖桿置中時段)。", **vals)
        return _na(det, "too_short", f"可判讀的搖桿置中時間只有 {(n_idle + n_air) * dt:.1f} s,不足以判定。", **vals)
    g_txt = f"地面怠速最大 {g_max:.1f}°(門檻 8°)" if g_max is not None and n_idle * dt >= GROUND_LEAN_S else ""
    a_txt = f"空中位置模式最大 {a_max:.1f}°(門檻 12°)" if a_max is not None and n_air * dt >= AIR_LEAN_S else ""
    scope = (f"空中只判 LOITER / POSHOLD / FLOWHOLD 且搖桿回中 {brk_lo:.1f} s(POSHOLD {brk_ph:.1f} s)之後;"
             "煞車期間與 AUTO / GUIDED / RTL 等自主模式未判讀")
    return Finding(det, DOMAIN, CLEAR, "info", None, None, "normal", vals,
                   _sentences("搖桿置中時飛控沒有下異常大的傾斜指令:" + "、".join(x for x in (g_txt, a_txt) if x), scope))


# =====================================================================
#                      5. pilot_input_context
# =====================================================================

def level_flipbacks(t: np.ndarray, v: np.ndarray, max_s: float = PILOT_C_FLIP_S):
    """三段開關(RC9 = EKF 來源組選擇)的折返:A→B→A 在 max_s 內回到原位。

    t / v:RCIN 該通道樣本(已限定在空中)。回傳 [(t_first_change, t_back, from_level, to_level)]。
    單向掃過 1095→1515→1935 不算折返(正常的 set 1→3 切換)。
    """
    if t.size < 3:
        return []
    lvl = np.where(v < RC_LEVEL_LOW, 0, np.where(v <= RC_LEVEL_HIGH, 1, 2))
    ch = np.flatnonzero(np.diff(lvl) != 0) + 1            # 位準改變的樣本索引
    out = []
    for k in range(1, ch.size):
        i_prev, i_cur = ch[k - 1], ch[k]
        d1 = int(lvl[i_prev]) - int(lvl[i_prev - 1])
        d2 = int(lvl[i_cur]) - int(lvl[i_cur - 1])
        if d1 * d2 < 0 and (t[i_cur] - t[i_prev]) <= max_s:
            out.append((float(t[i_prev]), float(t[i_cur]), int(lvl[i_prev - 1]), int(lvl[i_prev])))
    return out


def stick_events(t: np.ndarray, p1: np.ndarray, p2: np.ndarray, h1: float, h2: float, air: np.ndarray):
    """A:滿舵(max(|p1|/h1, |p2|/h2) ≥ 0.8 持續 0.2 s);B:0.5 s 內變化 ≥ 0.6 全程(roll 或 pitch)。

    p1 / p2:roll / pitch 搖桿相對 trim 的偏移(µs);h1 / h2:半程(µs);air:空中遮罩。
    B 的比較樣本取 t − 0.5 s 最近的樣本(差 ≤ 0.06 s,跨空洞就不比)。回傳 (A_runs, B_runs, dev, move)。
    """
    tol = 0.5 * _dt_med(t)
    dev = np.fmax(np.abs(p1) / h1, np.abs(p2) / h2)
    ra = _runs(air & (dev >= PILOT_DEV_A), t, max(PILOT_A_S - tol, 0.0), GAP_SLOW_S)
    j = np.clip(np.searchsorted(t, t - PILOT_B_WINDOW_S), 1, max(t.size - 1, 1))
    prev_closer = np.abs(t - PILOT_B_WINDOW_S - t[j - 1]) < np.abs(t - PILOT_B_WINDOW_S - t[j])
    jj = np.where(prev_closer, j - 1, j)
    near = np.abs(t - PILOT_B_WINDOW_S - t[jj]) <= 0.06
    move = np.fmax(np.abs(p1 - p1[jj]) / h1, np.abs(p2 - p2[jj]) / h2)
    move_ok = air & air[jj] & near
    rb = _runs(move_ok & (move >= PILOT_DEV_B), t, 0.0, GAP_SLOW_S)
    return ra, rb, dev, np.where(move_ok, move, np.nan)


@guarded("pilot_input_context", DOMAIN)
def pilot_input_context(ctx: Ctx, a: Anchors):
    """駕駛操作背景:空中(解鎖滿 3 s、非落地)的滿舵 / 甩桿 / EKF 來源開關折返,與 t_warn 的先後。

    A:|roll 或 pitch 偏移| ≥ 0.8 全程持續 0.2 s;B:0.5 s 內變化 ≥ 0.6 全程;C:RCn_OPTION=90 的開關 A→B→A 折返 ≤ 1.5 s。
    前兆窗 P = [max(t_arm+3, t_warn−60), t_warn)(dataflow STEP 4c);窗之前的大動作只列在 values['earlier_inputs']。
    HIT = 窗內或 t_warn 之後有上述任一;context:'precursor'(前兆窗內有大動作 → 可能的駕駛因素;t_onset = 最接近 t_warn
    的那一個)/ 'reaction'(前兆窗內沒有、t_warn 之後才有 → 事後反應,不是原因;t_onset = 第一個)/
    'no_reference'(沒有 t_warn;t_onset = 最早的一個)。CLEAR = 空中有效搖桿資料 ≥ 5 s,且前兆窗內(沒有 t_warn 時:全程)
    平穩、t_warn 之後也沒有。RCIN 為 NaN 的樣本不算(全部無效 → NA)。
    信心中。注意:HIT+reaction 在「排除」邏輯上等同於「駕駛在失控前平穩」,請看 values['precursor_max_dev'] 與 context。
    """
    det = "pilot_input_context"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    rc = ctx.table("RCIN")
    if rc is None or "C1" not in rc or "C2" not in rc:
        return _na(det, "no_data", "缺少 RCIN 遙控輸入,無法判斷駕駛操作。")
    s = _slice(rc, iv.t0, iv.t1, [c for c in rc if c != "t"])
    t = s["t"]
    if t.size < 10:
        return _na(det, "too_short", "解鎖期間遙控輸入資料太少。")
    air_all, air_src = _airborne_mask(ctx, iv, t)
    if int(air_all.sum()) < 5:
        return _na(det, "no_airborne", "解鎖期間沒有空中(非落地)時段,不判讀駕駛的大動作(排除起飛 / 落地的搖桿手勢)。")
    valid = np.isfinite(s["C1"]) & np.isfinite(s["C2"])
    air = air_all & valid                     # NaN 搖桿樣本不算(不能把「沒有資料」當成「搖桿置中」)
    if int(air.sum()) < 5:
        return _na(det, "no_data", "空中時段的 roll / pitch 搖桿資料無效(NaN),無法判斷駕駛操作。",
                   airborne_source=air_src, n_invalid_airborne=int(np.sum(air_all & ~valid)))
    tr1, tr2 = _rc_trim(ctx, 1, rc), _rc_trim(ctx, 2, rc)
    h1, h2 = _rc_half_travel(ctx, 1), _rc_half_travel(ctx, 2)
    p1, p2 = s["C1"] - tr1, s["C2"] - tr2
    t_ref = a.t_warn
    ra, rb, dev, db = stick_events(t, p1, p2, h1, h2, air)
    events = []
    for (a0, a1, _, _) in ra:
        events.append({"kind": "A_full_stick", "t": a0, "t_end": a1})
    for (b0, b1, i0, i1) in rb:
        events.append({"kind": "B_fast_move", "t": b0, "t_end": b1, "delta": _f(float(np.nanmax(db[i0:i1 + 1])), 2)})
    # C:EKF 來源開關折返(只用該通道有效的空中樣本)
    chans = [n for n in range(1, 17) if ctx.param(f"RC{n}_OPTION", 0) == RC_OPTION_EKF_SRC and f"C{n}" in rc]
    for n in chans:
        mc = air_all & np.isfinite(s[f"C{n}"])
        for (t1c, t2c, lv0, lv1) in level_flipbacks(t[mc], s[f"C{n}"][mc]):
            events.append({"kind": "C_source_switch_flipback", "t": t1c, "t_end": t2c, "channel": n})
    events.sort(key=lambda e: e["t"])
    # 前兆窗 P = [max(t_arm+3, t_ref−60), t_ref)(dataflow STEP 4c);平穩度統計到 t_ref + 0.5 s(沒有 t_ref 就整個空中時段)
    win_lo = max(iv.t0 + ARM_SKIP_S, t_ref - PILOT_PRE_WINDOW_S) if t_ref is not None else None
    if t_ref is not None:
        pre = air & (t >= win_lo) & (t <= t_ref + 0.5)
    else:
        pre = air
    pre_dev = float(np.max(dev[pre])) if pre.any() else None
    pre_db = float(np.nanmax(db[pre])) if (pre & np.isfinite(db)).any() else None
    pre_ev = [e for e in events if t_ref is not None and win_lo <= e["t"] < t_ref]
    post_ev = [e for e in events if t_ref is not None and e["t"] >= t_ref]
    earlier = [e for e in events if t_ref is not None and e["t"] < win_lo]
    fmt = lambda e: {k: (_f(v, 2) if isinstance(v, float) else v) for k, v in e.items()}   # noqa: E731
    vals = {"airborne_source": air_src, "airborne_s": _f(int(air.sum()) * _dt_med(t), 1),
            "n_invalid_airborne": int(np.sum(air_all & ~valid)),
            "precursor_window": [_f(win_lo), _f(t_ref)] if t_ref is not None else None,
            "precursor_max_dev": _f(pre_dev, 2), "precursor_max_move_0p5s": _f(pre_db, 2),
            "n_events": len(events), "events": [fmt(e) for e in events if e not in earlier][:8],
            "n_earlier_events": len(earlier), "earlier_inputs": [fmt(e) for e in earlier[-MAX_LISTED:]],
            "trims": [_f(tr1, 0), _f(tr2, 0)], "rc_ekf_source_channels": chans,
            "thr": {"A_dev": PILOT_DEV_A, "B_move": PILOT_DEV_B, "C_flip_s": PILOT_C_FLIP_S,
                    "pre_window_s": PILOT_PRE_WINDOW_S, "calm_dev": PILOT_CALM_DEV},
            "t_ref": _f(t_ref)}
    # 輔助:油門搖桿(降落動作)與下降率
    if "C3" in s:
        c3ok = air & np.isfinite(s["C3"])
        lo = ctx.param("RC3_MIN", 1100.0) + THR_MIN_MARGIN_US
        lo_idx = np.flatnonzero(c3ok & (s["C3"] <= lo) & (t >= (t_ref - 10.0 if t_ref is not None else iv.t0)))
        vals["t_throttle_stick_min"] = _f(t[lo_idx[0]] if lo_idx.size else None)
        if t_ref is not None:
            pm = c3ok & (t >= t_ref - 30.0) & (t <= t_ref)
            vals["throttle_stick_median_pre_us"] = _f(np.median(s["C3"][pm]) if pm.any() else None, 0)
    crt = ctx.series("CTUN.CRt")
    if crt is not None and t_ref is not None:
        m = (crt[0] >= t_ref - 5.0) & (crt[0] <= t_ref - 0.5) & np.isfinite(crt[1])
        vals["climb_rate_pre_cms"] = _f(np.median(crt[1][m]) if m.any() else None, 0)
        vals["pilot_speed_dn_cms"] = _f(ctx.param("PILOT_SPEED_DN", None), 0)
    kinds = {"A_full_stick": "滿舵", "B_fast_move": "快速甩桿", "C_source_switch_flipback": "來回切換 EKF 來源開關"}
    early_zh = ""
    if earlier:
        e0 = earlier[-1]
        early_zh = (f"更早在 {e0['t']:.2f} s 有大動作({kinds[e0['kind']]},共 {len(earlier)} 次),"
                    f"比姿態預警早 {t_ref - e0['t']:.0f} s,在 {PILOT_PRE_WINDOW_S:.0f} s 前兆窗之外,只列為參考")
    calm_zh = ""
    if pre_dev is not None:
        calm_zh = (f"預警前 {PILOT_PRE_WINDOW_S:.0f} s 搖桿偏移最大 {pre_dev * 100:.0f}% 行程"
                   + ("(平穩)" if pre_dev < PILOT_CALM_DEV else "(有操作,但未達滿舵 / 甩桿門檻)"))
    if not (pre_ev or post_ev) and (t_ref is not None or not events):
        air_s = int(air.sum()) * _dt_med(t)
        if air_s < MIN_ANALYSED_S:
            return _na(det, "too_short", f"空中遙控輸入只有 {air_s:.1f} s,不足以判定駕駛操作是否平穩。", **vals)
        if t_ref is not None:
            head = f"姿態預警({t_ref:.2f} s)前 {PILOT_PRE_WINDOW_S:.0f} s 內與之後都沒有滿舵 / 快速甩桿 / 來回切換 EKF 來源開關的操作"
        else:
            head = "空中沒有滿舵 / 快速甩桿 / 來回切換 EKF 來源開關的操作"
        if pre_dev is not None:
            head += f"(搖桿偏移最大 {pre_dev * 100:.0f}% 行程)"
        return Finding(det, DOMAIN, CLEAR, "info", None, None, "calm", vals,
                       _sentences(head + ",駕駛不是肇因", early_zh))
    # 代表事件:有 t_ref 時取「前兆窗內最接近 t_ref 的前兆」(因果最相關),
    # 沒有前兆才取 t_ref 之後的第一個(反應);沒有 t_ref 取最早的一個。
    chosen = pre_ev[-1] if pre_ev else (post_ev[0] if post_ev else events[0])
    t_in = chosen["t"]
    delay = (t_in - t_ref) if t_ref is not None else None
    if delay is None:
        ctxs = "no_reference"
    else:
        ctxs = "reaction" if delay >= 0 else "precursor"
    vals.update(first_input_t=_f(events[0]["t"]), t_selected=_f(t_in), selected_kind=chosen["kind"],
                n_precursor_events=len(pre_ev), n_reaction_events=len(post_ev), delay_s=_f(delay, 2),
                verdict={"reaction": "pilot_reaction", "precursor": "possible_pilot_cause",
                         "no_reference": "no_failure_reference"}[ctxs])
    sel = pre_ev if ctxs == "precursor" else (post_ev if ctxs == "reaction" else events)
    desc = "、".join(v for k, v in kinds.items() if any(e["kind"] == k for e in sel))
    if ctxs == "reaction":
        text = _sentences(f"駕駛的大動作({desc})最早在 {t_in:.2f} s,晚於姿態預警 {t_ref:.2f} s 共 {delay:.1f} s,"
                          "屬於事後反應(後果),不是原因", calm_zh, early_zh)
        sev = "info"
    elif ctxs == "precursor":
        text = _sentences(f"駕駛在姿態預警({t_ref:.2f} s)之前 {-delay:.1f} s 就有大動作({desc},{t_in:.2f} s),"
                          "可能是肇因之一,請對照錄影確認操作意圖", early_zh)
        sev = "medium"
    else:
        text = f"空中有駕駛大動作({desc}),最早 {t_in:.2f} s;沒有姿態失控時間點可比較因果。"
        sev = "info"
    return Finding(det, DOMAIN, HIT, sev, float(t_in), float(t_in), ctxs, vals, text)


# =====================================================================
#                      6. uncommanded_yaw_or_sink
# =====================================================================

@guarded("uncommanded_yaw_or_sink", DOMAIN)
def uncommanded_yaw_or_sink(ctx: Ctx, a: Anchors):
    """沒有打舵的偏航自轉 / 下降速度比指令快很多(只作排除證據;124 沒有正樣本,信心低)。

    YAW:RATE.Y 0.5 s 平均 |mY| ≥ 30°/s、偏航搖桿置中 ≥ 1 s(RCIN.C4 為 NaN 視為不置中)、空中、ae < 15°(翻覆不算偏航);
    SINK:CRt − DCRt ≤ −60 cm/s 持續 ≥ 0.5 s、空中,只看 a.t_first_hit(第一次硬撞擊)之前。
    HIT = 任一成立;CLEAR 需兩項資料都齊。
    """
    det = "uncommanded_yaw_or_sink"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    events = []          # (t_onset, kind, info)
    ok_yaw = ok_sink = False
    vals: dict = {"thr": {"yaw_dps": YAW_RATE_MIN, "sink_cms": SINK_ERR_CMS, "sink_s": SINK_SUSTAIN_S}}
    # ---------- 偏航 ----------
    rt, rc = ctx.table("RATE"), ctx.table("RCIN")
    core = _attitude_core(ctx, iv)
    if rt is not None and "Y" in rt and rc is not None and "C4" in rc and core is not None:
        sr = _slice(rt, iv.t0, iv.t1, ("Y", "YOut"))
        t = sr["t"]
        if t.size >= 100:
            dt = _dt_med(t)
            n = max(10, int(round(YAW_MEAN_WINDOW_S / dt)))
            tr4 = _rc_trim(ctx, 4, rc)
            # 偏航搖桿「過去 1 s 全程置中」:在 RCIN 格點上算,再零階保持到 RATE 格點
            rs = _slice(rc, iv.t0 - YAW_STICK_HOLD_S, iv.t1, ("C4",))
            rtt = rs["t"]
            c4 = rs["C4"]
            bad = (~np.isfinite(c4) | (np.abs(c4 - tr4) > STICK_CENTER_US)).astype(int)   # NaN = 不知道 → 不算置中
            cs = np.concatenate(([0], np.cumsum(bad)))
            j0 = np.searchsorted(rtt, rtt - YAW_STICK_HOLD_S + 1e-6, side="left")
            held = ((cs[1:] - cs[j0]) == 0) & ((rtt - rtt[j0]) >= YAW_STICK_HOLD_S - 0.15)
            stick_ok, vs = _zoh(rtt, held.astype(float), t)
            stick_ok = (stick_ok > 0.5) & vs
            air, _ = _airborne_mask(ctx, iv, t)
            ts_att, ae_att = core[0]["t"], core[1]
            ae_i = np.interp(t, ts_att, np.nan_to_num(ae_att, nan=0.0))       # 豁免段 ae=NaN → 當 0(同時有 exempt 另外擋)
            exempt = _exempt_mask(ctx, iv, t)
            ts_, te_, mean, nbad = moving_mean(t, sr["Y"], n)
            if te_.size:
                bad_w = _window_count(~air | exempt | (ae_i >= YAW_AE_MAX_DEG) | ~stick_ok, n)
                span_ok = (te_ - ts_) <= 1.4 * YAW_MEAN_WINDOW_S
                ok = span_ok & (nbad == 0) & (bad_w == 0) & (ts_ >= iv.t0 + ARM_SKIP_S)
                ok_yaw = _analysed_s(te_, ok, GAP_FAST_S) >= MIN_ANALYSED_S
                cond = ok & (np.abs(mean) >= YAW_RATE_MIN)
                vals["yaw_abs_mean_max_dps"] = _f(float(np.max(np.abs(mean[ok]))) if ok.any() else None, 1)
                idx = np.flatnonzero(cond)
                if idx.size:
                    k = int(idx[0])
                    yo = np.abs(sr["YOut"][k:k + n]).mean() if "YOut" in sr else None
                    events.append((float(te_[k]), "yaw", {"mean_dps": float(mean[k]), "yout_mean": _f(yo, 2)}))
    # ---------- 下沉 ----------
    # 高度 / 升降率類特徵只取「第一次硬撞擊之前」(撞擊 / 翻滾時氣壓與 EKF 高度是垃圾,124 的 1259.44 單點 −126 cm/s)
    crt, dcrt = ctx.series("CTUN.CRt"), ctx.series("CTUN.DCRt")
    t_alt_end = iv.t1
    if a.t_first_hit is not None and iv.t0 < a.t_first_hit < iv.t1:
        t_alt_end = float(a.t_first_hit)
    if crt is not None and dcrt is not None:
        tc = crt[0]
        m = (tc >= iv.t0) & (tc <= t_alt_end)
        tc = tc[m]
        if tc.size >= 20:
            sc = 100.0 if _unit(ctx, "CTUN.CRt") == "m/s" else 1.0
            err = (crt[1][m] - np.interp(tc, dcrt[0], dcrt[1])) * sc
            air, _ = _airborne_mask(ctx, iv, tc)
            air &= np.isfinite(err)
            ok_sink = _analysed_s(tc, air, GAP_SLOW_S) >= MIN_ANALYSED_S
            dtc = _dt_med(tc)
            n_sus = int(round(SINK_SUSTAIN_S / dtc)) + 1 if dtc > 0 else 0
            worst_sus = _sustained_max(np.where(air, -err, -np.inf), n_sus) if n_sus else None
            vals.update(sink_err_min_cms=_f(float(np.min(err[air])) if air.any() else None, 0),
                        sink_err_sustained_min_cms=_f(-worst_sus if worst_sus is not None else None, 0),
                        sink_window_end=_f(t_alt_end), sink_clipped_at_impact=bool(t_alt_end < iv.t1))
            tol = 0.5 * dtc
            rs_ = _runs(air & (err <= SINK_ERR_CMS), tc, max(SINK_SUSTAIN_S - tol, 0.0), GAP_SLOW_S)
            if rs_:
                t0r, t1r, i0, i1 = rs_[0]
                alt = ctx.series("CTUN.Alt")
                info = {"err_min_cms": float(np.min(err[i0:i1 + 1])), "dur": t1r - t0r}
                mm = (tc >= t0r - SINK_PRE_S) & (tc <= t0r - 0.5)
                info["sink_rate_pre_cms"] = _f(np.median(crt[1][m][mm]) * sc if mm.any() else None, 0)
                if alt is not None:
                    info["alt_loss_m"] = _f(float(np.interp(t0r - SINK_PRE_S, alt[0], alt[1]) - np.interp(t0r, alt[0], alt[1])), 2)
                im = _imu_table(ctx)
                if im is not None and all(k in im for k in ("AccX", "AccY", "AccZ")):
                    si = _slice(im, t0r - SINK_PRE_S, t0r, ("AccX", "AccY", "AccZ"))
                    if si["t"].size:
                        an = np.sqrt(si["AccX"] ** 2 + si["AccY"] ** 2 + si["AccZ"] ** 2)
                        fr = _runs(an < FREEFALL_ACC, si["t"], FREEFALL_S, 0.05)
                        info["free_fall_before"] = bool(fr)
                events.append((float(t0r), "sink", info))
    if not events:
        if ok_yaw and ok_sink:
            sus, raw = vals.get("sink_err_sustained_min_cms"), vals.get("sink_err_min_cms")
            yaw_mx = vals.get("yaw_abs_mean_max_dps")
            spike = f";單點最低 {raw:.0f} cm/s 未持續" if raw is not None and raw <= SINK_ERR_CMS else ""
            clip = "(只看第一次撞擊之前)" if vals.get("sink_clipped_at_impact") else ""
            sus_txt = f"{sus:.0f}" if sus is not None else "—"
            yaw_txt = f"{yaw_mx:.0f}" if yaw_mx is not None else "—"
            return Finding(det, DOMAIN, CLEAR, "info", None, None, "yaw_and_sink", vals,
                           f"沒有未下指令的偏航自轉(搖桿置中時 0.5 s 平均最大 {yaw_txt}°/s,門檻 30);"
                           f"下降速度也沒有明顯超過指令{clip}:持續 0.5 s 的 CRt−DCRt 最差 {sus_txt} cm/s(門檻 −60){spike}。")
        return _na(det, "no_data" if not (ok_yaw or ok_sink) else "partial_data",
                   "偏航 / 下沉資料不足或沒有空中時段,無法判定。", **vals)
    events.sort(key=lambda e: e[0])
    t_on, kind, info = events[0]
    kinds = "+".join(sorted({e[1] for e in events}))
    vals.update(events=[{"kind": k, "t": _f(tt, 2), **{kk: (_f(vv, 2) if isinstance(vv, float) else vv) for kk, vv in inf.items()}}
                        for tt, k, inf in events[:4]])
    caveat = ("此偵測器沒有真實正樣本(00000124 兩者都沒觸發),信心低、主要當排除證據;"
              "沒有電調遙測(ESC telemetry / RPM)時無法直接證實是哪顆馬達")
    if kind == "yaw":
        text = _sentences(f"{t_on:.2f} s 起偏航角速度平均 {abs(info['mean_dps']):.0f}°/s,但偏航搖桿置中:機身在自己旋轉,"
                          "可能是單顆馬達 / 槳 / 電調效率不足(偏航力矩不平衡),建議檢查", caveat)
    else:
        text = _sentences(f"{t_on:.2f} s 起實際下降速度持續 {info['dur']:.1f} s 比指令快(CRt−DCRt 最低 {info['err_min_cms']:.0f} cm/s,"
                          "門檻 −60),可能推力不足 / 失去升力", caveat)
    return Finding(det, DOMAIN, HIT, "medium", float(t_on), float(t_on + (0.0 if kind == "yaw" else SINK_SUSTAIN_S)),
                   kinds, vals, text)


# =====================================================================
#                              入口
# =====================================================================

def run(ctx: Ctx, a: Anchors) -> list[Finding]:
    """依序執行本領域偵測器;attitude_loss_of_control 先跑(它設定 a.t_warn / a.t_loss / a.t_inverted,後面會用)。"""
    out: list[Finding] = []
    for fn in (attitude_loss_of_control, rate_demand_not_achieved, rate_output_saturation,
               lean_without_pilot, pilot_input_context, uncommanded_yaw_or_sink):
        out += fn(ctx, a)
    return out
