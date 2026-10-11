"""失事分析 · 末態偵測模組(end_state):這次飛行「怎麼結束的」、撞擊在何時、機身有沒有翻覆。

偵測器(每個至少回傳一筆 Finding;未解鎖 → NA):
  crash_check_anchor          飛控自己判定墜機並上鎖(ERR Subsys 12 / ARM.Method 12 / 'Crash: Disarming' 訊息)
  crash_check_replay          用 log 的姿態 + 加速度重算飛控的 CRASH_CHECK(FS_CRASH_CHECK=0 時仍能偵測)
  abnormal_disarm_classifier  上鎖分類:正常降落 / 墜機 / 失效保護上鎖 / 空中手動上鎖 / log 在解鎖中結束 / 沒起飛(ground_only)
  impact_shock_clusters       加速度衝擊群:觸地(ground_contact)與撞擊(hard_impact),時間、峰值、dV、方向;
                              飛行中連續 >0.2 s 的平台(油門推滿 / 急拉)= sustained_accel,不是撞擊
  free_fall_run               自由落體(合加速度 <3 m/s² 持續 ≥0.1 s)
  tumble_inversion            翻覆 / 倒扣(姿態傾角 + 加速度重力方向交叉驗證)
  vibe_clip_transient         撞擊造成的 VIBE 暴增 / Clip 躍升(必須對應到 IMU 衝擊群),或持續振動 / 削波

run() 依序執行後呼叫 classify_end() 把「結束型態」定案(dataflow STEP 2)並填錨點(STEP 3):
  a.end_type / t_detect / t_disarm / impacts / t_first_hit / t_impact_peak / t_rest。
  正常降落 / 正常結束不填撞擊錨點;t_detect − 2.6 s 的撞擊估計只用在 CRASH_CHECK(其 2 s 靜止條件造成的延遲)。
不設 t_warn / t_loss / t_inverted / t_touch(別的模組負責),所以翻覆偵測自己用 ANG/ATT 算傾角。

資料覆蓋:IMU / 姿態 / VIBE 只涵蓋解鎖區間的一部分(最長空洞 >1.5 s、覆蓋 <90%、或最後 10 s 覆蓋 <90%)時,
「沒有發現」只能回 NA('partial_coverage'),不能回 CLEAR;有資料的部分找到的證據照樣 HIT(文字附註)。

可信度與限制(誠實聲明):
  - 真實資料只有一次失事(00000124 飛行 2)與一次正常降落(飛行 1)。每個門檻都是「這次失事」與「兩次健康飛行」
    之間的量測間隔(見各常數註解),不是統計出來的;其餘分支(空中手動上鎖、失效保護上鎖、log 在解鎖中結束、
    ECode 2、沒起飛、ERR 遺失)沒有真實樣本,只靠 ArduPilot 語意與合成資料測試,可信度低。
  - 撞擊 = 一串短脈衝(124 每段 ≤0.09 s);連續 >0.2 s 的單一平滑鼓包(極軟的地面)會被當成平台而漏判(已知限制)。
  - RECOG_WIN_S(觸地後 2 s 內要宣告落地)只有 n=1 的正常降落當依據,單獨超時只給 SOFT。
  - IMU log 是 ~400 Hz 抽樣的串流(原始 1.6~3.2 kHz),撞擊峰值被低估;門檻以 log 的取樣率為準。
  - 全部時間為相對秒(= 絕對秒 − t_base)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .ctx import Ctx
from .guard import guarded
from .types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval, Finding
from .util import thrust_vector_error_deg, tilt_deg

DOMAIN = "end_state"

G = 9.80665

# ---------------------------------------------------------------- 門檻(皆附量測間隔:失事 vs 對照)
# --- CRASH_CHECK(ArduCopter crash_check.cpp 的常數;124 重算 1262.131 s vs ERR 1262.125 s,差 6 ms)
CRASH_SUBSYS = 12                 # ERR Subsys 12 = CRASH_CHECK;ECode 1 = 偵測到墜機,2 = 失去控制
CRASH_ANGLE_DEG = 30.0            # 姿態(推力向量)誤差
CRASH_ACCEL_MAX = 3.0             # 1 Hz 濾波、扣重力後的加速度大小 (m/s²)
CRASH_DWELL_S = 2.0               # 連續成立時間
CRASH_LPF_HZ = 1.0
CRASH_BLOCK = 1                   # IMU 不做區塊平均(全速率 ~400 Hz):124 條件起點 1260.1306 = 研究值 1260.130、
                                  # accf 0.131(MSG 0.1);舊版 8 筆平均(~50 Hz)起點晚 ~7 ms、accf 0.18
CRASH_LPF_WARMUP_S = 3.0          # 濾波只在 ae>30 的區段前 3 s 起算(時間常數 0.16 s,3 s 後初值影響 <1e-8)
CRASH_SKIP_MODES = (1, 14)        # ACRO / FLIP 不檢查
CRASH_GAP_S = 0.5                 # 資料空洞 > 0.5 s 就重新計時
ERR_MATCH_S = 0.2                 # ERR 與 ARM / EV11 / MSG 的對應視窗(量測偏移 < 0.1 ms)
ERR_TAIL_S = 0.5                  # 上鎖後多久內的 ERR 仍算這次解鎖
REPLAY_VALIDATE_S = 0.05          # 重算與 ERR 的最大容許差(124:+5 ms)
REPLAY_MIN_ARMED_S = 3.0          # 解鎖時間太短就無法判定
REST_ACCEL = 15.0                 # t_rest:最後一次 |a| > 15 m/s²(靜止前最後的動靜)
REST_LOOKBACK_S = 10.0
INVERTED_ACCZ = 5.0               # 倒扣旗標:靜止期 AccZ 中位數 > +5(正放 −9.8)且 |Roll| > 150°
INVERTED_ROLL_DEG = 150.0
IMPACT_LAG_TYPICAL_S = 2.6        # 撞擊 → CRASH_CHECK 的典型延遲(124:2.63 s);撞擊時間無法量測時的最後備援

# --- 衝擊群
IMPACT_ONSET = 20.0               # 非飛行最大 16.5;受控觸地 25.1~28.9 (m/s²)
HARD_PEAK = 40.0                  # 29 與 59 之間沒有任何資料 (m/s²)
HARD_DV = 3.0                     # 觸地 dV 0.66~1.22;失事 4.6~6.9 (m/s)
CLUSTER_MERGE_S = 0.3
DV_DT_CAP_S = 0.02                # IMU log 有掉資料(最長 0.45 s):dV 積分每步最多算 20 ms
SUSTAINED_RUN_S = 0.20            # 撞擊是一串短脈衝:124 每段連續 >20 m/s² 最長 0.028 / 0.03 / 0.036 / 0.09 s(觸地 / 觸地 /
                                  # 首撞 8 個脈衝 / 末撞 5 個脈衝);油門推滿 / 急拉是連續的平台(≥0.25 s)。連續 >0.2 s 的段落
                                  # 以該段中位數當基準(平台不算 dV,只算疊在平台上的脈衝),整群都是平台 → 'sustained_accel'(不是撞擊)
MEDIAN_WIN = 5                    # 5 點中位數預濾波(主撞擊峰值損失 ≤1.4%)
IMU_MIN_ROWS = 1000               # 取樣數太少的 IMU instance 不參與
RECOG_WIN_S = 2.0                 # 觸地後飛控宣告落地(EV17/EV18)的時限;正常 0.82 s,124 失事 6.9 s 仍沒有。
                                  # 只有 n=1 的正常降落當依據(手動降落收油門較慢時可能 >2 s)→ 單獨超時只給 SOFT(見 TIPOVER_FOLLOW_S)
TIPOVER_FOLLOW_S = 5.0            # 未宣告落地的觸地之後 5 s 內出現翻覆或硬撞擊,才稱為『觸地後翻覆的起點』(HIT);
                                  # 124:觸地 1255.225 → 翻覆 1258.14(2.9 s)、首撞 1258.18(2.96 s)
IMPACT_POST_S = 1.0               # 上鎖後多看 1 s(撞擊視窗 [t_d−6, t_d+1])
IMPACT_WINDOW_PRE_S = 6.0
IMPACT_POST_PEAK_S = 0.5          # t_impact_peak 的視窗上限:t_disarm + 0.5
IMPACT_EPISODE_GAP_S = 10.0       # 硬撞擊相隔 >10 s 算不同事件(124:首撞 → 末撞 0.65 s,觸地 → 首撞 2.96 s)
IMPACT_AE_PRE_S = 0.5             # ae_at_impact:群起點前 0.5 s 的最大姿態誤差(外力撞擊 = 撞擊前姿態正常 <10°)
MAX_IMPACT_FINDINGS = 8

# --- 自由落體
FF_ACC = 3.0                      # 失事 0.85 (med5);119 手揮最低 3.6 → 不可放寬
FF_MIN_S = 0.10                   # 失事 0.157 s
FF_GAP_S = 0.02                   # IMU 時間戳空洞 > 20 ms 就斷開
FF_THEN_IMPACT_S = 0.5            # 自由落體結束後 0.5 s 內開始硬撞擊 → 'free-fall then impact'
AE_WARN_DEG = 10.0                # 自由落體發生時姿態誤差 <10° → 空中先失效(先於姿態異常)

# --- 翻覆
TUMBLE_TILT_DEG = 45.0            # 正常飛行 / 觸地最大傾角 12.4°(飛行 2 失事前)、9.3°(飛行 1)
TUMBLE_MIN_S = 0.3
INVERT_TILT_DEG = 120.0
INVERT_MIN_S = 0.5
TUMBLE_END_INV_S = 1.0            # 無 ERR 時:倒扣 ≥1 s 視為失事終局
TILT_GAP_S = 0.25
ACC_TILT_WIN_S = 0.25             # 加速度重力方向的平均視窗
ACC_TILT_CONFIRM_DEG = 45.0       # 124:0.25 s 平均加速度傾角 >45° 於 1258.09(姿態 1258.06)
ACC_TILT_FALLEN_DEG = 60.0        # 1 s 平均 >60° 持續 2 s = 倒下
ACC_TILT_INVERT_DEG = 120.0
ACC_TILT_FALLEN_S = 2.0
COMMANDED_MODES = (1, 13, 14)     # ACRO / SPORT / FLIP:刻意翻轉
FLIP_EV_START, FLIP_EV_END = 21, 22
FLIP_TAIL_S = 3.0
FLIP_MAX_S = 10.0

# --- VIBE / Clip
VIBE_BAD = 30.0                   # ArduPilot 建議;解鎖飛行最大 11.24(2.7x 間隔)
VIBE_CRIT = 60.0
VIBE_TRANSIENT_MAX_S = 3.0
VIBE_SUSTAINED_S = 5.0
VIBE_FRAC_WIN_S = 30.0
VIBE_FRAC = 0.20
VIBE_RUN_MERGE_S = 1.0
VIBE_ASSOC_RUN_S = 1.5            # VIBE 超標與撞擊事件的對應視窗
VIBE_ASSOC_CLIP_S = 1.0
VIBE_CLIP_SAME_S = 0.2            # 多顆 IMU 的 Clip 同一筆內躍升
VIBE_CLIP_BIG = 5                 # 單顆 dc ≥ 5
VIBE_CLIP_REPEAT_N = 3            # 沒有撞擊卻累計 ≥3 次 Clip 增量 = 削波
VIBE_SKIP_S = 10.0                # 統計「撞擊前」p95 時略過起飛前 10 s
VIBE_TAIL_S = 0.1

# --- 上鎖分類
ARM_NORMAL_METHODS = frozenset({0, 1, 2, 3, 4, 13, 17})     # 正常上鎖原因;其餘為失效保護 / 自動(5~11、14+ 沒有樣本)
ARM_MANUAL_METHODS = frozenset({0, 1, 2, 4})                # 人為(方向舵 / MAVLink / AUX / Lua)
ARM_METHOD_CRASH = 12
EV_NOT_LANDED, EV_LAND_MAYBE, EV_LAND_COMPLETE, EV_EMERGENCY_STOP = 28, 17, 18, 54
DISARM_EV_EPS = 0.001             # 上鎖那一刻 1 ms 內補寫的 EV 是上鎖造成的
AIR_CRT_CMS = 50.0                # 空中手動上鎖:|CRt| 中位數 > 50 cm/s
AIR_AGL_M = 1.5                   # 或 agl > 1.5 m → critical
POWER_DROP_RATIO = 0.5            # log 在解鎖中結束:最後 2 s 電壓 < 先前中位數的 50% = 疑似斷電
POWER_TAIL_S = 2.0

# --- 資料覆蓋:IMU / 姿態 / VIBE 只覆蓋解鎖區間的一部分時,不能說「整段都沒問題」(CLEAR → NA)
COVER_MAX_GAP_S = 1.5             # 124 兩次解鎖期間最大空洞:IMU / ANG 0.39 s、ATT / VIBE 0.8 s(10 Hz 掉資料)→ 1.9x 餘裕
COVER_MIN_FRAC = 0.90             # 124 兩次解鎖的覆蓋率 ≥ 0.991
COVER_TAIL_S = 10.0               # 失事多半在最後幾秒:最後 10 s 也要 ≥ COVER_MIN_FRAC
ATT_INTERP_MAX_GAP_S = 1.0        # 重算 CRASH_CHECK 時姿態只內插跨越 ≤1 s 的空洞(10 Hz 掉資料 0.5~0.8 s);更長 / 範圍外 = 無資料

_ARM_METHOD_ZH = {0: "方向舵", 1: "MAVLink", 2: "AUX 開關", 3: "馬達測試", 4: "Lua 腳本",
                  12: "墜機偵測(CRASH_CHECK)", 13: "落地後自動", 17: "閒置自動"}
_DIR_ZH = {"bottom": "機底朝下(起落架)", "top": "機頂朝下(倒扣)", "lateral": "側向(機臂 / 槳葉)", "oblique": "斜向"}
_VIBE_TYPE = {"impact_transient": "impact clipping", "sustained_vibration": "sustained vibration",
              "vibration_clipping": "vibration clipping", "clip_increment": "clip increment",
              "short_burst": "short burst", "no_vibe_clip": "none"}
_CRASH_MSG_RX = re.compile(r"Crash:\s*Disarming:\s*AngErr=([0-9.]+)>([0-9.]+),\s*Accel=([0-9.]+)<([0-9.]+)", re.I)

HARD_CLASSES = ("hard_impact", "hard_landing")
IMPACT_CLASSES = HARD_CLASSES + ("ground_contact",)     # 'sustained_accel'(飛行中的持續加速)不是撞擊


# ================================================================ 小工具

def _r(x, nd: int = 3):
    """數值轉 JSON 友善的 float(NaN / None → None)。"""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _runs(mask, t, min_dur: float = 0.0, max_gap: float | None = None):
    """連續為真的區段 [(t_start, t_end, i0, i1)](向量化;相鄰真樣本時間差 > max_gap 斷開)。"""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    t = np.asarray(t, dtype=float)
    brk = np.diff(idx) > 1
    if max_gap is not None:
        brk = brk | ((t[idx[1:]] - t[idx[:-1]]) > max_gap)
    cut = np.flatnonzero(brk)
    s_idx = idx[np.r_[0, cut + 1]]
    e_idx = idx[np.r_[cut, idx.size - 1]]
    return [(float(t[i0]), float(t[i1]), int(i0), int(i1)) for i0, i1 in zip(s_idx, e_idx)
            if t[i1] - t[i0] >= min_dur]


def _na(det_id: str, context: str, text: str, values: dict | None = None) -> Finding:
    return Finding(det_id, DOMAIN, NA, "info", context=context, values=values or {}, text_zh=text)


def _not_armed(det_id: str) -> Finding:
    return _na(det_id, "not_armed", "未解鎖(桌上 / 手持)的 log,不做失事分析。")


def _med5(x: np.ndarray) -> np.ndarray:
    if x.size < MEDIAN_WIN:
        return x.copy()
    return np.median(sliding_window_view(np.pad(x, MEDIAN_WIN // 2, mode="edge"), MEDIAN_WIN), axis=1)


def _movmean(x: np.ndarray, k: int) -> np.ndarray:
    """置中移動平均(端點縮窗);x shape (N,) 或 (N, D)。"""
    n = x.shape[0]
    k = max(1, min(int(k), n))
    if k == 1:
        return x.copy()
    pad = [(1, 0)] + [(0, 0)] * (x.ndim - 1)
    c = np.cumsum(np.pad(x, pad), axis=0)
    h = k // 2
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    cnt = (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))
    return (c[hi] - c[lo]) / cnt


def _tilt_from_acc(m: np.ndarray) -> np.ndarray:
    """加速度計量到的比力方向 → 機身傾角(度):正放 AccZ=−9.8 → 0°,倒扣 +9.8 → 180°。"""
    nrm = np.linalg.norm(m, axis=1)
    nrm = np.where(nrm < 1e-6, 1e-6, nrm)
    return np.degrees(np.arccos(np.clip(-m[:, 2] / nrm, -1.0, 1.0)))


def _commanded_mask(ctx: Ctx, t: np.ndarray) -> np.ndarray:
    """ACRO / SPORT / FLIP 模式與 FLIP 事件(EV21~22 後 +3 s)期間為 True(刻意翻轉,不算失控)。"""
    t = np.asarray(t, dtype=float)
    m = np.zeros(t.size, dtype=bool)
    tm, nums, _ = ctx.mode_rows()
    if tm.size:
        i = np.searchsorted(tm, t, side="right") - 1
        mode = np.where(i >= 0, nums[np.clip(i, 0, None)], -1)
        m |= np.isin(mode, COMMANDED_MODES)
    ev_t, ev_id = ctx.ev_rows()
    ends = ev_t[ev_id == FLIP_EV_END]
    for s in ev_t[ev_id == FLIP_EV_START]:
        e = ends[ends >= s]
        e_t = (float(e[0]) if e.size else float(s) + FLIP_MAX_S) + FLIP_TAIL_S
        m |= (t >= s) & (t <= e_t)
    return m


def _mode_skip_mask(ctx: Ctx, t: np.ndarray, skip: tuple[int, ...]) -> np.ndarray:
    tm, nums, _ = ctx.mode_rows()
    if not tm.size:
        return np.zeros(np.asarray(t).size, dtype=bool)
    i = np.searchsorted(tm, t, side="right") - 1
    mode = np.where(i >= 0, nums[np.clip(i, 0, None)], -1)
    return np.isin(mode, skip)


def _landed_mask(ctx: Ctx, iv: ArmedInterval, t: np.ndarray) -> np.ndarray:
    """區間內每個時間點的「已落地」狀態(EV28 離地 / EV18 落地;上鎖當下補寫的 EV 不算)。"""
    steps = ctx.landed_steps(iv)
    st = np.array([s[0] for s in steps], dtype=float)
    sv = np.array([s[1] for s in steps], dtype=bool)
    i = np.searchsorted(st, t, side="right") - 1
    return sv[np.clip(i, 0, None)]


def coverage(t: np.ndarray | None, t0: float, t1: float) -> dict:
    """串流 t 對 [t0, t1] 的覆蓋:{frac, max_gap_s, gap_at, tail_frac, ok}(純函式)。

    兩端(t0 → 第一筆、最後一筆 → t1)也算空洞;每個空洞扣掉 max(5 個取樣間隔, 0.05 s) 的容許後才算「缺」。
    ok = 最大空洞 ≤ COVER_MAX_GAP_S 且整段與最後 COVER_TAIL_S 秒的覆蓋率都 ≥ COVER_MIN_FRAC。
    """
    span = float(t1 - t0)
    if span <= 0:
        return {"frac": 1.0, "max_gap_s": 0.0, "gap_at": None, "tail_frac": 1.0, "ok": True}
    t = np.asarray(t if t is not None else [], dtype=float)
    t = t[(t >= t0) & (t <= t1)]
    if t.size == 0:
        return {"frac": 0.0, "max_gap_s": _r(span), "gap_at": _r(t0), "tail_frac": 0.0, "ok": False}
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    tol = max(5.0 * dt, 0.05)
    e = np.r_[t0, t, t1]
    g = np.diff(e)
    miss = np.clip(g - tol, 0.0, None)
    k = int(np.argmax(g))
    tail0 = max(t0, t1 - COVER_TAIL_S)
    # 最後 COVER_TAIL_S 秒內的缺漏:每個空洞與 [tail0, t1] 的交集(扣掉容許)
    inter = np.clip(e[1:] - np.maximum(e[:-1], tail0), 0.0, None)
    tail_miss = float(np.sum(np.clip(inter - tol, 0.0, None)))
    frac = 1.0 - float(miss.sum()) / span
    tail_frac = 1.0 - tail_miss / max(t1 - tail0, 1e-9)
    ok = bool(g[k] <= COVER_MAX_GAP_S and frac >= COVER_MIN_FRAC and tail_frac >= COVER_MIN_FRAC)
    return {"frac": _r(frac, 3), "max_gap_s": _r(g[k]), "gap_at": _r(e[k]), "tail_frac": _r(tail_frac, 3), "ok": ok}


def _cover_txt(name: str, cov: dict) -> str:
    """覆蓋不足的說明(繁中)。"""
    return (f"{name} 資料只涵蓋解鎖期間的 {cov['frac'] * 100:.0f}%(最長空洞 {cov['max_gap_s']:.1f} s,自 {cov['gap_at']:.1f} s 起;"
            f"最後 {COVER_TAIL_S:.0f} s 涵蓋 {cov['tail_frac'] * 100:.0f}%)")


def _partial_na(det_id: str, name: str, cov: dict, clean_txt: str, values: dict) -> Finding:
    """資料覆蓋不足:有資料的部分看起來乾淨,但不能據此排除 → NA(不是 CLEAR)。"""
    return _na(det_id, "partial_coverage", f"{_cover_txt(name, cov)},無法確認整段解鎖期間;有資料的部分:{clean_txt}",
               {**values, "coverage": cov})


# ================================================================ 共用資料(同一次 run 內快取)

@dataclass
class _ImuWin:
    t: np.ndarray               # IMU[0] 時間(相對秒)
    acc: list                   # 每個 instance 的 (N,3) 比力(已內插到 t)
    gyr0: np.ndarray            # instance 0 角速度 (N,3) rad/s
    acons: np.ndarray           # 跨 instance 中位數的 5 點中位數合加速度 (N,)
    n0: np.ndarray              # instance 0 原始 |a| (N,)


def _imu_instances(ctx: Ctx) -> list:
    insts = ctx.instances("IMU") or [None]
    out = []
    for i in insts:
        tb = ctx.table("IMU", i)
        if tb is not None and all(k in tb for k in ("AccX", "AccY", "AccZ")) and len(tb["t"]) >= IMU_MIN_ROWS:
            out.append(tb)
    return out


def build_imu_window(tables: list, t0: float, t1: float) -> _ImuWin | None:
    """tables(各 IMU instance 的表)→ [t0, t1] 的合加速度共識(跨 instance 中位數 of 5 點中位數)。"""
    if not tables:
        return None
    base = tables[0]["t"]
    i0 = int(np.searchsorted(base, t0, side="left"))
    i1 = int(np.searchsorted(base, t1, side="right"))
    if i1 - i0 < 3:
        return None
    t = base[i0:i1]
    acc, a5 = [], []
    for k, tb in enumerate(tables):
        if k == 0:
            A = np.stack([tb[c][i0:i1] for c in ("AccX", "AccY", "AccZ")], axis=1)
        else:
            j0 = max(0, int(np.searchsorted(tb["t"], t[0], side="left")) - 2)
            j1 = min(len(tb["t"]), int(np.searchsorted(tb["t"], t[-1], side="right")) + 2)
            if j1 - j0 < 2:
                continue
            tt = tb["t"][j0:j1]
            A = np.stack([np.interp(t, tt, tb[c][j0:j1]) for c in ("AccX", "AccY", "AccZ")], axis=1)
        acc.append(A)
        a5.append(_med5(np.linalg.norm(A, axis=1)))
    acons = np.median(np.vstack(a5), axis=0)
    t0_tb = tables[0]
    if all(c in t0_tb for c in ("GyrX", "GyrY", "GyrZ")):
        gyr0 = np.stack([t0_tb[c][i0:i1] for c in ("GyrX", "GyrY", "GyrZ")], axis=1)
    else:
        gyr0 = np.zeros((t.size, 3))
    return _ImuWin(t, acc, gyr0, acons, np.linalg.norm(acc[0], axis=1))


def impact_clusters(t: np.ndarray, acons: np.ndarray, acc_xyz: np.ndarray | None = None,
                    gyr_dps: np.ndarray | None = None) -> list[dict]:
    """合加速度共識 > IMPACT_ONSET 的樣本合併(間隔 ≤0.3 s)成衝擊群。純函式(可直接餵合成資料)。

    每群:t_start / t_peak / t_end / peak (m/s²,原始峰值) / peak_impulse / dV (m/s) / cls / pulse_max_s /
    sustained_s / t_hard(第一個脈衝 ≥ HARD_PEAK 的樣本,沒有則 None)/ width40_s / gyro_peak_dps / u(力方向)/ dir。

    脈衝形狀:群內每段「連續 >20 m/s²」若超過 SUSTAINED_RUN_S 就是持續加速的平台(油門推滿 / 急拉),
    該段以自己的中位數當基準;其餘樣本以 1 g 當基準。dV = Σ max(a − 基準, 0)·dt(群前後各 0.1 s),
    peak_impulse = max(a − 基準) + 1 g。沒有平台時 peak_impulse = peak、dV 與舊算法相同(124 的四群皆如此)。
    cls:hard_impact(peak_impulse≥40 且 dV≥3)、hard_landing(兩者之一)、ground_contact(短脈衝 20~40 且 dV<3)、
    sustained_accel(含平台且平台上沒有硬撞擊等級的脈衝 = 飛行動作,不是撞擊)。
    """
    t = np.asarray(t, dtype=float)
    acons = np.asarray(acons, dtype=float)
    idx = np.flatnonzero(acons > IMPACT_ONSET)
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(t[idx]) > CLUSTER_MERGE_S)
    starts = idx[np.r_[0, cut + 1]]
    ends = idx[np.r_[cut, idx.size - 1]]
    dt_all = np.diff(t, append=t[-1] + (np.median(np.diff(t)) if t.size > 1 else 0.0025))
    dt_all = np.minimum(dt_all, DV_DT_CAP_S)
    out = []
    for i0, i1 in zip(starts, ends):
        seg = acons[i0:i1 + 1]
        k = i0 + int(np.argmax(seg))
        lo = int(np.searchsorted(t, t[i0] - 0.1, side="left"))
        hi = int(np.searchsorted(t, t[i1] + 0.1, side="right"))
        base = np.full(hi - lo, 9.81)
        pulse_max, sustained_s = 0.0, 0.0
        for rs, re_, r0, r1 in _runs(seg > IMPACT_ONSET, t[i0:i1 + 1], 0.0, DV_DT_CAP_S):
            dur = re_ - rs
            pulse_max = max(pulse_max, dur)
            if dur > SUSTAINED_RUN_S:
                sustained_s += dur
                base[i0 + r0 - lo:i0 + r1 + 1 - lo] = float(np.median(seg[r0:r1 + 1]))
        exc = np.clip(acons[lo:hi] - base, 0.0, None)
        dv = float(np.sum(exc * dt_all[lo:hi]))
        imp = exc[i0 - lo:i1 + 1 - lo] + 9.81           # 群內每個樣本的「脈衝」大小(平台已扣掉)
        peak = float(acons[k])
        peak_imp = float(np.max(imp))
        hard_idx = np.flatnonzero(imp >= HARD_PEAK)
        t_hard = float(t[i0 + hard_idx[0]]) if hard_idx.size else None
        width40 = float(t[i0 + hard_idx[-1]] - t[i0 + hard_idx[0]]) if hard_idx.size else 0.0
        if peak_imp >= HARD_PEAK and dv >= HARD_DV:
            cls = "hard_impact"
        elif peak_imp >= HARD_PEAK or dv >= HARD_DV:
            cls = "hard_landing"
        elif sustained_s > 0.0:
            cls = "sustained_accel"
        else:
            cls = "ground_contact"
        u, dr = None, None
        if acc_xyz is not None:
            a0, a1 = max(0, k - 3), min(len(t), k + 4)
            v = np.mean(acc_xyz[a0:a1], axis=0)
            nv = float(np.linalg.norm(v))
            if nv > 1e-6:
                u = (v / nv)
                uz = float(u[2])
                dr = "bottom" if uz <= -0.8 else "top" if uz >= 0.8 else "lateral" if abs(uz) < 0.5 else "oblique"
                u = [round(float(x), 2) for x in u]
        gp = None
        if gyr_dps is not None:
            g0 = int(np.searchsorted(t, t[i0] - 0.1, side="left"))
            g1 = int(np.searchsorted(t, t[i1] + 0.1, side="right"))
            gp = float(np.max(gyr_dps[g0:g1])) if g1 > g0 else None
        out.append({"t_start": _r(t[i0]), "t_peak": _r(t[k]), "t_end": _r(t[i1]), "peak": _r(peak, 2),
                    "peak_impulse": _r(peak_imp, 2), "dV": _r(dv, 2), "cls": cls, "pulse_max_s": _r(pulse_max),
                    "sustained_s": _r(sustained_s), "t_hard": _r(t_hard), "width40_s": _r(width40),
                    "gyro_peak_dps": _r(gp, 1), "u": u, "dir": dr})
    return out


def crash_dwell(t: np.ndarray, cond: np.ndarray, dwell: float = CRASH_DWELL_S,
                max_gap: float = CRASH_GAP_S):
    """條件連續成立 dwell 秒就觸發(同韌體計數器;資料空洞 > max_gap 重新計時)。

    回傳 (t_fire, t_start, longest):t_start = t_fire − dwell(= 條件成立的起點),longest = 最長連續成立秒數。
    純函式。第一個成立樣本也累計一個步長(同韌體:第一次迭代 counter=1)。
    """
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return None, None, 0.0
    dt0 = float(np.median(np.diff(t)))
    longest = 0.0
    for ts, te, i0, i1 in _runs(cond, t, 0.0, max_gap):
        longest = max(longest, te - ts + dt0)
        if te - ts + dt0 >= dwell:
            j = int(np.searchsorted(t[i0:i1 + 1], ts + dwell - dt0, side="left"))
            j = min(j, i1 - i0)
            t_fire = float(t[i0 + j])
            return t_fire, t_fire - dwell, longest
    return None, None, longest


class _Sh:
    """同一次 run 內共用的快取:IMU 視窗、姿態視窗、衝擊群。"""

    def __init__(self, ctx: Ctx, a: Anchors):
        self.ctx = ctx
        self.a = a
        self._tables = None
        self._imu: dict = {}
        self._att: dict = {}
        self._clusters: list | None = None

    def imu_tables(self) -> list:
        if self._tables is None:
            self._tables = _imu_instances(self.ctx)
        return self._tables

    def imu(self, t0: float, t1: float) -> _ImuWin | None:
        key = (round(t0, 3), round(t1, 3))
        if key not in self._imu:
            self._imu[key] = build_imu_window(self.imu_tables(), t0, t1)
        return self._imu[key]

    def att(self, t0: float, t1: float) -> dict | None:
        """ANG(~400 Hz)優先,沒有用 ATT:{src, t, roll, pitch, yaw, droll, dpitch, dyaw}(度,未 unwrap)。"""
        key = (round(t0, 3), round(t1, 3))
        if key not in self._att:
            self._att[key] = None
            for name in ("ANG", "ATT"):
                tb = self.ctx.table(name)
                if tb is None or not all(k in tb for k in ("Roll", "Pitch", "Yaw", "DesRoll", "DesPitch", "DesYaw")):
                    continue
                i0 = int(np.searchsorted(tb["t"], t0, side="left"))
                i1 = int(np.searchsorted(tb["t"], t1, side="right"))
                if i1 - i0 < 3:
                    continue
                sl = slice(i0, i1)
                self._att[key] = {"src": name, "t": tb["t"][sl], "roll": tb["Roll"][sl], "pitch": tb["Pitch"][sl],
                                  "yaw": tb["Yaw"][sl], "droll": tb["DesRoll"][sl], "dpitch": tb["DesPitch"][sl],
                                  "dyaw": tb["DesYaw"][sl]}
                break
        return self._att[key]

    def ae(self, t0: float, t1: float):
        """(t, ae, tilt):推力向量誤差(同韌體 AngErr)與機身傾角。沒有姿態資料回傳 None。"""
        w = self.att(t0, t1)
        if w is None:
            return None
        ae = thrust_vector_error_deg(w["roll"], w["pitch"], w["yaw"], w["droll"], w["dpitch"], w["dyaw"])
        return w["t"], ae, tilt_deg(w["roll"], w["pitch"])

    def intervals(self) -> list[ArmedInterval]:
        ivs = list(self.a.intervals) or list(self.ctx.armed)
        if self.a.interval is not None and all(self.a.interval is not x for x in ivs):
            ivs.append(self.a.interval)
        return sorted(ivs, key=lambda x: x.t0)

    def clusters(self) -> list[dict]:
        """所有解鎖區間的衝擊群(含上鎖後 1 s),每群帶 'interval' 編號;依時間排序。"""
        if self._clusters is None:
            out = []
            ivs = self.intervals()
            for n, iv in enumerate(ivs):
                nxt = ivs[n + 1].t0 if n + 1 < len(ivs) else np.inf
                w = self.imu(iv.t0, min(iv.t1 + IMPACT_POST_S, nxt))
                if w is None:
                    continue
                gdps = np.degrees(np.linalg.norm(w.gyr0, axis=1))
                acc_mean = np.mean(np.stack(w.acc, axis=0), axis=0)
                for c in impact_clusters(w.t, w.acons, acc_mean, gdps):
                    c["interval"] = int(iv.index)
                    out.append(c)
            self._clusters = sorted(out, key=lambda c: c["t_start"])
        return self._clusters

    def clusters_in(self, iv: ArmedInterval) -> list[dict]:
        return [c for c in self.clusters() if c["interval"] == iv.index]

    def imu_cover(self, iv: ArmedInterval) -> dict:
        """IMU[0](衝擊 / 自由落體 / 重算的基準時間軸)對解鎖區間的覆蓋。"""
        tb = self.imu_tables()
        return coverage(tb[0]["t"] if tb else None, iv.t0, iv.t1)

    def att_cover(self, iv: ArmedInterval) -> dict:
        w = self.att(iv.t0, iv.t1)
        return {**coverage(w["t"] if w is not None else None, iv.t0, iv.t1), "src": w["src"] if w is not None else None}

    def tumble_runs(self, iv: ArmedInterval) -> list:
        """解鎖期間非刻意(非 ACRO / SPORT / FLIP / EV21~22)的翻覆段 [(ts, te, i0, i1)]:傾角 ≥45° 持續 ≥0.3 s。"""
        key = ("tumble", iv.index, round(iv.t0, 3), round(iv.t1, 3))
        if key not in self._att:
            runs = []
            att = self.att(iv.t0, iv.t1)
            if att is not None:
                m = (att["t"] >= iv.t0) & (att["t"] <= iv.t1)
                t = att["t"][m]
                if t.size >= 3:
                    tilt = tilt_deg(att["roll"][m], att["pitch"][m])
                    runs = _runs(~_commanded_mask(self.ctx, t) & (tilt >= TUMBLE_TILT_DEG), t, TUMBLE_MIN_S, TILT_GAP_S)
            self._att[key] = runs
        return self._att[key]


def impact_clusters_for_window(ctx: Ctx, t0: float, t1: float) -> list[dict]:
    """公開工具:任意視窗(不限解鎖區間)的衝擊群,例如檢查 log 全程有沒有 >20 m/s² 的衝擊(基準 log 測試用)。"""
    w = build_imu_window(_imu_instances(ctx), t0, t1)
    if w is None:
        return []
    acc_mean = np.mean(np.stack(w.acc, axis=0), axis=0)
    return impact_clusters(w.t, w.acons, acc_mean, np.degrees(np.linalg.norm(w.gyr0, axis=1)))


def _t_rest(sh: _Sh, t_ref: float) -> float | None:
    """t_ref 之前最後一個 IMU[0] |a| > 15 m/s² 的樣本時間(機身靜止的起點);沒有資料回傳 None。"""
    w = sh.imu(t_ref - REST_LOOKBACK_S, t_ref)
    if w is None:
        return None
    m = (w.t <= t_ref) & (w.n0 > REST_ACCEL)
    idx = np.flatnonzero(m)
    return float(w.t[idx[-1]]) if idx.size else None


def _last_episode(hard: list[dict]) -> list[dict]:
    """依時間排序的硬撞擊 → 最後一串(相鄰間隔 ≤ IMPACT_EPISODE_GAP_S)。
    例:起飛時一次重落地、幾分鐘後才失事 → 只取失事那一串,早先的重落地不能變成 t_first_hit。"""
    if not hard:
        return []
    ep = [hard[-1]]
    for c in reversed(hard[:-1]):
        if ep[0]["t_start"] - c["t_end"] > IMPACT_EPISODE_GAP_S:
            break
        ep.insert(0, c)
    return ep


CRASH_CHECK_END_TYPES = ("crash_check", "crash_check_replay")
NO_IMPACT_END_TYPES = ("normal_landing", "normal_end", "not_armed", "unknown")


def _pick_impact_anchors(clusters: list[dict], iv: ArmedInterval, end_type: str, t_detect: float | None):
    """(t_first_hit, t_impact_peak, 來源):只看最後一串硬撞擊(_last_episode)。
    t_first_hit = 這串第一個硬撞擊中第一個 ≥40 m/s² 的樣本(群起點 20 m/s² 比它早 ~0.2 s);
    t_impact_peak = 這串最後一個硬撞擊的峰值時間(撞擊先於姿態失控的外力撞擊也要算,所以不以 t_loss 當下界;
    end_state 也比 attitude 模組先跑,拿不到 t_loss)。

    依結束型態:
      normal_landing / normal_end → 不填(正常飛行沒有「撞擊」錨點,報告不得出現撞擊時間);
      crash_check / crash_check_replay → 解鎖期間的硬撞擊 → 判定前最後一次 ground_contact → t_detect − 2.6 s
        (2.6 s 是 CRASH_CHECK「2 s 靜止」造成的延遲,只對這兩型成立);
      其他異常(空中上鎖、失效保護上鎖、log 在解鎖中結束、翻覆、hard_impact)→ 只看 [t_detect − 6, 上鎖 + 1] 的
        硬撞擊 / 觸地,找不到就不填(空中上鎖的撞擊在上鎖之後,不能用 2.6 s 倒推)。"""
    if end_type in NO_IMPACT_END_TYPES:
        return None, None, ""
    if end_type in CRASH_CHECK_END_TYPES:
        lo, hi = -np.inf, iv.t1 + IMPACT_POST_PEAK_S
    else:
        ref = t_detect if t_detect is not None else iv.t1
        lo, hi = min(ref, iv.t1) - IMPACT_WINDOW_PRE_S, iv.t1 + IMPACT_POST_S
    hard = _last_episode([c for c in clusters if c["cls"] in HARD_CLASSES and lo <= c["t_peak"] and c["t_start"] <= hi])
    if hard:
        t_first = hard[0]["t_hard"] if hard[0]["t_hard"] is not None else hard[0]["t_start"]
        return t_first, hard[-1]["t_peak"], "imu_hard_impact"
    t_cap = min(hi, t_detect) if (t_detect is not None and end_type in CRASH_CHECK_END_TYPES) else hi
    gc = [c for c in clusters if c["cls"] == "ground_contact" and lo <= c["t_peak"] <= t_cap]
    if gc:
        return None, gc[-1]["t_peak"], "ground_contact"
    if t_detect is not None and end_type in CRASH_CHECK_END_TYPES:
        return None, _r(t_detect - IMPACT_LAG_TYPICAL_S), "lag_estimate"
    return None, None, ""


# ================================================================ 1. crash_check_anchor

@guarded("crash_check_anchor", DOMAIN)
def crash_check_anchor(ctx: Ctx, a: Anchors, sh: _Sh | None = None):
    """飛控判定墜機並自動上鎖:ERR(Subsys 12, ECode 1/2) + ARM.Method 12 + 'Crash: Disarming' 訊息交叉確認。

    t_onset = 機身靜止的起點(t_rest:飛控的判定條件從這裡開始成立),t_alarm = ERR 時間;
    這是「結果」不是原因:真正的撞擊比 ERR 早 ~2.6 s(124:2.63 s)。
    FS_CRASH_CHECK=0 時 ERR 永遠不會出現 → NA,請看 crash_check_replay。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("crash_check_anchor")
    sh = sh or _Sh(ctx, a)
    te, sub, code = ctx.err_rows()
    sel = np.flatnonzero((sub == CRASH_SUBSYS) & np.isin(code, (1, 2)) & (te >= iv.t0) & (te <= iv.t1 + ERR_TAIL_S))
    fs = ctx.param("FS_CRASH_CHECK", None)
    arm = ctx.table("ARM")
    err_missing = sel.size == 0
    if err_missing:
        # ERR 列可能因 log 掉資料而遺失:ARM.Method=12 或 'Crash: Disarming' 訊息仍是飛控的墜機判定 → 不能回 CLEAR
        alt_t, alt_src = None, ""
        if arm is not None and "ArmState" in arm and "Method" in arm:
            m12 = np.flatnonzero((arm["ArmState"] == 0) & (arm["Method"] == ARM_METHOD_CRASH)
                                 & (arm["t"] >= iv.t0) & (arm["t"] <= iv.t1 + ERR_TAIL_S))
            if m12.size:
                alt_t, alt_src = float(arm["t"][m12[0]]), "ARM.Method=12"
        if alt_t is None and iv.method == ARM_METHOD_CRASH and not iv.open_end:
            alt_t, alt_src = float(iv.t1), "ARM.Method=12"
        if alt_t is None:
            for tm, txt in ctx.messages_between(iv.t0, iv.t1 + ERR_TAIL_S, r"Crash:\s*Disarming"):
                alt_t, alt_src = float(tm), "MSG"
                break
        if alt_t is None:
            if fs is not None and float(fs) == 0:
                return _na("crash_check_anchor", "fs_crash_check_off",
                           "FS_CRASH_CHECK=0:飛控不會自動判定墜機,這個偵測器無法使用;請看『墜機判定重算』的結果。",
                           {"FS_CRASH_CHECK": 0})
            return Finding("crash_check_anchor", DOMAIN, CLEAR, "info", context="no_crash_err",
                           values={"FS_CRASH_CHECK": None if fs is None else float(fs)},
                           text_zh="這次解鎖期間飛控沒有觸發 CRASH_CHECK(沒有 ERR 子系統 12、ARM.Method=12 或墜機上鎖訊息),沒有被判定為墜機。")
        t_err, ecode = alt_t, None
    else:
        k = int(sel[0])
        t_err = float(te[k])
        ecode = int(code[k])
    vals: dict = {"t_err": _r(t_err), "ecode": ecode, "FS_CRASH_CHECK": None if fs is None else float(fs),
                  "err_missing": bool(err_missing)}
    confirm = []
    # ARM 上鎖列(Method)
    if arm is not None and "ArmState" in arm:
        near = np.flatnonzero((arm["ArmState"] == 0) & (np.abs(arm["t"] - t_err) <= ERR_MATCH_S))
        if near.size and "Method" in arm:
            vals["arm_method"] = int(arm["Method"][near[0]])
            if vals["arm_method"] == ARM_METHOD_CRASH:
                confirm.append("ARM.Method=12")
    if "ARM.Method=12" not in confirm and iv.method == ARM_METHOD_CRASH and abs(iv.t1 - t_err) <= ERR_MATCH_S:
        vals["arm_method"] = ARM_METHOD_CRASH
        confirm.append("ARM.Method=12")
    ev_t, ev_id = ctx.ev_rows()
    if np.any((ev_id == 11) & (np.abs(ev_t - t_err) <= ERR_MATCH_S)):
        confirm.append("EV11")
    for tm, txt in ctx.messages_between(t_err - ERR_MATCH_S, t_err + ERR_MATCH_S, r"Crash:"):
        mm = _CRASH_MSG_RX.search(txt)
        if mm:
            vals.update({"angerr_deg": float(mm.group(1)), "angerr_limit": float(mm.group(2)),
                         "accel_ms2": float(mm.group(3)), "accel_limit": float(mm.group(4))})
            confirm.append("MSG")
            break
    vals["confirmed_by"] = confirm
    # 機身靜止起點與倒扣旗標
    t_rest = _t_rest(sh, t_err)
    vals["t_rest"] = _r(t_rest)
    lag_rest = None
    if t_rest is not None:
        lag_rest = t_err - t_rest
        vals["err_minus_rest_s"] = _r(lag_rest)
        w = sh.imu(t_rest, t_err)
        att = sh.att(t_rest, t_err)
        if w is not None and att is not None:
            accz = float(np.median(w.acc[0][:, 2]))
            roll = float(np.median(np.abs(((att["roll"] + 180.0) % 360.0) - 180.0)))
            vals["inverted"] = bool(accz > INVERTED_ACCZ and roll > INVERTED_ROLL_DEG)
            vals["rest_accz"] = _r(accz, 2)
            vals["rest_abs_roll_deg"] = _r(roll, 1)
    # 與 IMU 撞擊時間的差(用同一個 sh 的衝擊群)
    cl = sh.clusters_in(iv)
    _, t_imp, src = _pick_impact_anchors(cl, iv, "crash_check", t_err)
    if t_imp is not None and src == "imu_hard_impact":
        vals["t_impact_peak"] = _r(t_imp)
        vals["err_minus_impact_s"] = _r(t_err - t_imp)
    lag = vals.get("err_minus_impact_s")
    if err_missing:
        kind = f"log 沒有 ERR 子系統 12 紀錄(可能掉資料),依 {'、'.join(confirm) or alt_src} 判定"
    else:
        kind = "失去控制(ECode 2)" if ecode == 2 else "偵測到墜機"
    parts = [f"飛控於 {t_err:.2f} s 判定墜機並自動上鎖({kind}"]
    if "angerr_deg" in vals:
        parts.append(f",姿態誤差 {vals['angerr_deg']:.0f}°>{vals['angerr_limit']:.0f}°、加速度 {vals['accel_ms2']:.1f}<{vals['accel_limit']:.1f} m/s²")
    parts.append(")。")
    if lag is not None:
        parts.append(f"這只是結果:真正的撞擊約在 {t_imp:.2f} s,比此事件早 {lag:.1f} s")
    elif lag_rest is not None:
        parts.append(f"這只是結果:機身約在 {t_rest:.2f} s 就已靜止,比此事件早 {lag_rest:.1f} s")
    else:
        parts.append("這只是結果:真正的撞擊通常比這個時間點早約 2~3 s")
    if vals.get("inverted"):
        parts.append(",靜止時機身是倒扣的")
    parts.append("。原因請往前看事件鏈。")
    return Finding("crash_check_anchor", DOMAIN, HIT, "critical", t_onset=(t_rest if t_rest is not None else t_err),
                   t_alarm=t_err, context="loss_of_control" if ecode == 2 else "crash_check",
                   values=vals, text_zh="".join(parts))


# ================================================================ 2. crash_check_replay

def interp_valid(tq: np.ndarray, ts: np.ndarray, max_gap: float = ATT_INTERP_MAX_GAP_S) -> np.ndarray:
    """tq 的每個時間點能不能由 ts 內插(在 ts 範圍內、且兩側樣本相距 ≤ max_gap)。純函式。
    np.interp 在範圍外會沿用端點值、在長空洞裡會畫直線 —— 那些點都不是量測。"""
    tq = np.asarray(tq, dtype=float)
    ts = np.asarray(ts, dtype=float)
    if ts.size == 0:
        return np.zeros(tq.size, dtype=bool)
    j = np.searchsorted(ts, tq, side="left")
    jr = np.clip(j, 0, ts.size - 1)
    jl = np.clip(j - 1, 0, ts.size - 1)
    exact = ts[jr] == tq
    return (tq >= ts[0]) & (tq <= ts[-1]) & (exact | ((ts[jr] - ts[jl]) <= max_gap))


def _replay(sh: _Sh, ctx: Ctx, iv: ArmedInterval):
    """重算 CRASH_CHECK;回傳 dict 或 None(資料不足)。"""
    t0, t1 = iv.t0, iv.t1 + REPLAY_VALIDATE_S
    w = sh.imu(t0, t1)
    att = sh.att(t0 - 1.0, t1 + 1.0)
    if w is None or att is None:
        return None
    n = (w.t.size // CRASH_BLOCK) * CRASH_BLOCK
    if n < 8:
        return None
    tb = w.t[:n].reshape(-1, CRASH_BLOCK).mean(axis=1)
    acc = w.acc[0][:n].reshape(-1, CRASH_BLOCK, 3).mean(axis=1)
    ta = att["t"]
    valid = interp_valid(tb, ta)          # 姿態沒有覆蓋的 IMU 時間點不算(不沿用最後一筆姿態)
    if not valid.any():
        return None

    def uw(x):
        return np.interp(tb, ta, np.degrees(np.unwrap(np.radians(x))))

    r, p, y = uw(att["roll"]), uw(att["pitch"]), uw(att["yaw"])
    ae = thrust_vector_error_deg(r, p, y, uw(att["droll"]), uw(att["dpitch"]), uw(att["dyaw"]))
    ae = np.where(valid, ae, np.nan)      # NaN → 條件不成立 → dwell 重新計時
    out = {"src": att["src"], "ae_max": float(np.nanmax(ae)), "t": tb, "ae": ae, "att_valid_frac": float(valid.mean())}
    cond0 = ae > CRASH_ANGLE_DEG
    if not cond0.any():
        out.update(t_fire=None, t_start=None, longest=0.0, accf=None)
        return out
    # 機體 → NED 旋轉,扣重力,1 Hz 一階低通(每軸)。只有 ae>30 才需要濾波值:迴圈只跑 ae>30 區段與其前 3 s
    # (區段開頭濾波器從 0 起算 = 解鎖時機身靜止;正常飛行 ae 從未 >30 → 完全不跑迴圈)。
    need = np.zeros(tb.size, dtype=bool)
    for ts_, _te, _i0, i1_ in _runs(cond0, tb):
        need[int(np.searchsorted(tb, ts_ - CRASH_LPF_WARMUP_S, side="left")):i1_ + 1] = True
    cr, sr, cp, sp = np.cos(np.radians(r)), np.sin(np.radians(r)), np.cos(np.radians(p)), np.sin(np.radians(p))
    cy, sy = np.cos(np.radians(y)), np.sin(np.radians(y))
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    nx = cp * cy * ax + (sr * sp * cy - cr * sy) * ay + (cr * sp * cy + sr * sy) * az
    ny = cp * sy * ax + (sr * sp * sy + cr * cy) * ay + (cr * sp * sy - sr * cy) * az
    nz = -sp * ax + sr * cp * ay + cr * cp * az + G
    dts = np.minimum(np.diff(tb, prepend=tb[0] - float(np.median(np.diff(tb)))), 0.1)
    alpha = (dts / (dts + 1.0 / (2.0 * np.pi * CRASH_LPF_HZ))).tolist()
    nxl, nyl, nzl = nx.tolist(), ny.tolist(), nz.tolist()
    accf = np.full(tb.size, np.nan)            # 沒算的地方是 NaN → accf<3 為 False
    for _ts, _te, s0, s1 in _runs(need, tb):
        fx = fy = fz = 0.0
        seg = []
        for i in range(s0, s1 + 1):
            al = alpha[i]
            fx += al * (nxl[i] - fx)
            fy += al * (nyl[i] - fy)
            fz += al * (nzl[i] - fz)
            seg.append((fx * fx + fy * fy + fz * fz) ** 0.5)
        accf[s0:s1 + 1] = seg
    cond = cond0 & (accf < CRASH_ACCEL_MAX) & ~_mode_skip_mask(ctx, tb, CRASH_SKIP_MODES)
    ev_t, ev_id = ctx.ev_rows()
    if np.any(ev_id == EV_NOT_LANDED):      # log 有記離地事件才知道落地狀態(舊韌體沒有 EV28 → 不用 ~L 遮罩)
        cond &= ~_landed_mask(ctx, iv, tb)
    t_fire, t_start, longest = crash_dwell(tb, cond)
    out.update(t_fire=t_fire, t_start=t_start, longest=longest, accf=accf)
    return out


@guarded("crash_check_replay", DOMAIN)
def crash_check_replay(ctx: Ctx, a: Anchors, sh: _Sh | None = None):
    """用 log 的姿態 + 加速度重算飛控的 CRASH_CHECK(解鎖 & 未落地 & 非 ACRO/FLIP & 姿態誤差>30° & 1 Hz 濾波加速度<3,連續 2 s)。

    124:重算 1262.131 s(條件起點 1260.131,AngErr 149°、accf 0.13),飛控 ERR 1262.125 s(差 +5 ms);
    飛行 1 與 6 個基準 log(連閘門拿掉也一樣)零觸發:119 手持揮動 ae 49°,條件最長只連續 0.53 s。FS_CRASH_CHECK=0 時仍能偵測。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("crash_check_replay")
    if iv.duration < REPLAY_MIN_ARMED_S:
        return _na("crash_check_replay", "too_short", f"解鎖僅 {iv.duration:.1f} s,太短無法重算墜機判定。")
    sh = sh or _Sh(ctx, a)
    rp = _replay(sh, ctx, iv)
    if rp is None:
        return _na("crash_check_replay", "no_data", "缺少姿態(ANG/ATT)或 IMU 資料,無法重算墜機判定。")
    fs = ctx.param("FS_CRASH_CHECK", None)
    te, sub, code = ctx.err_rows()
    errs = te[(sub == CRASH_SUBSYS) & np.isin(code, (1, 2)) & (te >= iv.t0) & (te <= iv.t1 + ERR_TAIL_S)]
    vals = {"ae_max_deg": _r(rp["ae_max"], 1), "attitude_src": rp["src"], "FS_CRASH_CHECK": None if fs is None else float(fs),
            "longest_cond_s": _r(rp["longest"], 2), "limits": {"angle_deg": CRASH_ANGLE_DEG, "accel": CRASH_ACCEL_MAX,
                                                                 "dwell_s": CRASH_DWELL_S}}
    if rp["t_fire"] is None:
        if errs.size:
            vals["t_err"] = _r(errs[0])
            return _na("crash_check_replay", "not_reproduced", "飛控有 CRASH_CHECK 事件,但依 log 重算沒有重現(資料可能不足)。", vals)
        clean = (f"重算飛控的墜機判定從未成立:姿態誤差最大 {rp['ae_max']:.1f}°,條件(>30° 且加速度 <3 m/s² 連續 2 s)"
                 f"最長只連續 {rp['longest']:.2f} s。")
        for name, cov in (("IMU", sh.imu_cover(iv)), (f"姿態({rp['src']})", sh.att_cover(iv))):
            if not cov["ok"]:
                return _partial_na("crash_check_replay", name, cov, clean, vals)
        return Finding("crash_check_replay", DOMAIN, CLEAR, "info", context="no_fire", values=vals, text_zh=clean)
    t_fire, t_start = rp["t_fire"], rp["t_start"]
    i = int(np.searchsorted(rp["t"], t_fire))
    i = min(i, rp["t"].size - 1)
    vals.update({"t_fire": _r(t_fire), "t_start": _r(t_start), "ae_at_fire_deg": _r(rp["ae"][i], 1),
                 "accf_at_fire": _r(rp["accf"][i], 2)})
    txt = (f"依 log 重算飛控的墜機判定(姿態誤差 >30° 且濾波加速度 <3 m/s² 連續 2 s):{t_start:.2f} s 起成立,"
           f"{t_fire:.2f} s 觸發(此時姿態誤差 {rp['ae'][i]:.0f}°、加速度 {rp['accf'][i]:.2f} m/s²)")
    ctxt = "replay"
    if errs.size:
        d = t_fire - float(errs[0])
        vals["t_err"] = _r(errs[0])
        vals["dt_vs_err_s"] = _r(d, 4)
        vals["validated"] = bool(abs(d) < REPLAY_VALIDATE_S)
        txt += f",與飛控實際 ERR 相差 {abs(d) * 1000:.0f} ms" + ("(獨立驗證通過)" if vals["validated"] else "(偏差偏大,請人工確認)")
    else:
        ctxt = "replay_no_err"
        txt += ",但 log 裡沒有 CRASH_CHECK 事件" + ("(FS_CRASH_CHECK=0,飛控不會自動上鎖)" if fs is not None and float(fs) == 0 else "")
    return Finding("crash_check_replay", DOMAIN, HIT, "high", t_onset=t_start, t_alarm=t_fire, context=ctxt,
                   values=vals, text_zh=txt + "。")


# ================================================================ 3. abnormal_disarm_classifier

def _classify_interval(ctx: Ctx, iv: ArmedInterval) -> dict:
    """單一解鎖區間的結束分類(不依賴 IMU)。class ∈ crash_check / failsafe_disarm / manual_disarm_in_air /
    log_ended_armed / ground_only / normal_landing / normal_end / unknown。

    ground_only:log 有記離地事件(EV28),但這次解鎖期間一次都沒有 → 一直在地面(解鎖後沒有起飛就上鎖)。"""
    t_d = iv.t1
    m = iv.method
    ev_t, ev_id = ctx.ev_rows()
    has_land_info = bool(np.any(np.isin(ev_id, (EV_NOT_LANDED, EV_LAND_MAYBE, EV_LAND_COMPLETE))))
    landed = bool(ctx.landed_at(t_d - DISARM_EV_EPS, iv)) if not iv.open_end else bool(ctx.landed_at(t_d, iv))
    te, sub, code = ctx.err_rows()
    near_crash = bool(np.any((sub == CRASH_SUBSYS) & np.isin(code, (1, 2)) & (np.abs(te - t_d) <= ERR_MATCH_S)))
    lim = t_d - DISARM_EV_EPS
    in_iv = (ev_t >= iv.t0) & (ev_t < lim)
    t17 = ev_t[in_iv & (ev_id == EV_LAND_MAYBE)]
    t18 = ev_t[in_iv & (ev_id == EV_LAND_COMPLETE)]
    took_off = bool(np.any(in_iv & (ev_id == EV_NOT_LANDED)))
    no_takeoff = bool(np.any(ev_id == EV_NOT_LANDED)) and not took_off
    if iv.open_end:
        cls = "log_ended_armed"
    elif m == ARM_METHOD_CRASH or near_crash:
        cls = "crash_check"
    elif no_takeoff:
        cls = "ground_only"
    elif m is not None and m not in ARM_NORMAL_METHODS:
        cls = "failsafe_disarm"
    elif not has_land_info:
        cls = "unknown"
    elif landed:
        cls = "normal_landing"
    elif m is None or m in ARM_MANUAL_METHODS:
        cls = "manual_disarm_in_air"
    else:
        cls = "normal_end"
    return {"index": int(iv.index), "t0": _r(iv.t0), "t1": _r(iv.t1), "method": m, "class": cls,
            "landed_before": landed, "near_crash_err": near_crash, "has_land_info": has_land_info, "took_off": took_off,
            "dt17": _r(t_d - t17[-1]) if t17.size else None, "dt18": _r(t_d - t18[-1]) if t18.size else None,
            "abnormal": cls not in ("normal_landing", "normal_end", "ground_only", "unknown")}


def _disarm_metrics(ctx: Ctx, iv: ArmedInterval, t_cap: float) -> dict:
    """上鎖前 [t_d−6, t_d−5] 的 agl 與 |CRt| 中位數(絕不用撞擊之後的高度:t_cap 之後的視窗往前移)。"""
    out: dict = {}
    w0, w1 = iv.t1 - 6.0, iv.t1 - 5.0
    if w1 > t_cap:              # 視窗落在撞擊之後 → 往前移到撞擊前 1 s
        w0, w1 = t_cap - 1.0, t_cap
    if w0 < iv.t0:
        w0, w1 = iv.t0, min(iv.t0 + 1.0, iv.t1)
    alt = ctx.series("CTUN.Alt")
    g = ctx.ground_alt(iv)
    if alt is not None and g is not None:
        m = (alt[0] >= w0) & (alt[0] <= w1)
        if m.any():
            out["agl_5s_m"] = _r(float(np.median(alt[1][m])) - g, 2)
    crt = ctx.series("CTUN.CRt")
    if crt is not None:
        m = (crt[0] >= w0) & (crt[0] <= w1)
        if m.any():
            out["abs_crt_cms"] = _r(np.median(np.abs(crt[1][m])), 1)
    return out


def _tail_voltage(ctx: Ctx, t_end: float) -> dict:
    """log 在解鎖中結束:最後 2 s 電壓相對先前 10 s 中位數(BAT.Volt;沒有則 MOTB.BatVolt × MOT_BAT_VOLT_MAX)。"""
    s = ctx.series("BAT.Volt")
    scale = 1.0
    if s is None:
        s = ctx.series("MOTB.BatVolt")
        scale = float(ctx.param("MOT_BAT_VOLT_MAX", 0.0) or 0.0)
        if s is None or scale <= 0:
            return {}
    t, v = s[0], s[1] * scale
    pre = (t >= t_end - POWER_TAIL_S - 10.0) & (t < t_end - POWER_TAIL_S)
    tail = (t >= t_end - POWER_TAIL_S) & (t <= t_end)
    if pre.sum() < 3 or tail.sum() < 1:
        return {}
    med, vmin = float(np.median(v[pre])), float(np.min(v[tail]))
    if med <= 1.0:      # USB 供電 / 沒接電池:電壓無意義
        return {}
    return {"volt_before_med": _r(med, 1), "volt_tail_min": _r(vmin, 1), "volt_drop_ratio": _r(vmin / med, 2),
            "volt_collapsed": bool(vmin < POWER_DROP_RATIO * med)}


@guarded("abnormal_disarm_classifier", DOMAIN)
def abnormal_disarm_classifier(ctx: Ctx, a: Anchors, sh: _Sh | None = None):
    """上鎖分類:正常降落(先 EV17 → EV18 再上鎖)/ 墜機上鎖 / 失效保護上鎖 / 空中手動上鎖 / log 在解鎖中結束。

    上鎖那一刻 1 ms 內補寫的 EV17/18/11/56 是上鎖造成的,不算「落地」(ctx.landed_at 已處理)。
    只有 Method 0 與 12 有真實樣本;失效保護上鎖 / 空中手動上鎖 / log 中斷 / EV54 分支可信度低(ArduPilot 語意 + 合成測試)。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("abnormal_disarm_classifier")
    sh = sh or _Sh(ctx, a)
    infos = [_classify_interval(ctx, x) for x in sh.intervals()]
    me = next((i for i in infos if i["index"] == iv.index), None) or _classify_interval(ctx, iv)
    cls = me["class"]
    m = me["method"]
    mtxt = f"{_ARM_METHOD_ZH.get(m, '未知')}(ARM.Method={m})" if m is not None else "方式未知"
    vals = {"class": cls, "method": m, "landed_before": me["landed_before"], "dt17_s": me["dt17"], "dt18_s": me["dt18"],
            "near_crash_err": me["near_crash_err"], "intervals": [{k: i[k] for k in ("index", "t0", "t1", "method", "class", "abnormal")}
                                                                 for i in infos]}
    t_d = iv.t1
    # 失事後 EKF 高度是垃圾:高度視窗不可晚於第一個硬撞擊
    hard = [c for c in sh.clusters_in(iv) if c["cls"] in HARD_CLASSES]
    t_cap = hard[0]["t_start"] - 0.5 if hard else np.inf
    out: list[Finding] = []
    if cls == "crash_check":
        vals.update(_disarm_metrics(ctx, iv, t_cap))
        ev_t, ev_id = ctx.ev_rows()
        ev28 = ev_t[(ev_id == EV_NOT_LANDED) & (ev_t >= iv.t0) & (ev_t <= iv.t1)]
        written = ev_t[np.isin(ev_id, (EV_LAND_MAYBE, EV_LAND_COMPLETE)) & (ev_t >= t_d - DISARM_EV_EPS) & (ev_t <= t_d + DISARM_EV_EPS * 5)]
        vals["ev17_18_written_by_disarm"] = int(written.size)
        out.append(Finding("abnormal_disarm_classifier", DOMAIN, HIT, "critical", t_onset=t_d, t_alarm=t_d, context="crash_check",
                           values=vals, text_zh=(f"上鎖方式是 {mtxt},且上鎖前飛控一直認為飛機還在空中"
                                                 + (f"(最後一個『離地』事件 EV28 在 {ev28[-1]:.1f} s,之後沒有『落地完成』)" if ev28.size else "")
                                                 + "。上鎖瞬間補寫的 LAND_COMPLETE 事件是上鎖造成的,不算降落。這是墜機判定,不是正常降落。")))
    elif cls == "failsafe_disarm":
        out.append(Finding("abnormal_disarm_classifier", DOMAIN, HIT, "critical", t_onset=t_d, t_alarm=t_d, context="failsafe_disarm",
                           values=vals, text_zh=f"上鎖方式是 {mtxt},不屬於一般人為 / 落地上鎖,疑為失效保護或自動上鎖;請查上鎖前的 ERR / 模式變化。"))
    elif cls == "manual_disarm_in_air":
        vals.update(_disarm_metrics(ctx, iv, t_cap))
        crit = (vals.get("abs_crt_cms") or 0) > AIR_CRT_CMS or (vals.get("agl_5s_m") or 0) > AIR_AGL_M
        out.append(Finding("abnormal_disarm_classifier", DOMAIN, HIT, "critical" if crit else "high", t_onset=t_d, t_alarm=t_d,
                           context="manual_disarm_in_air", values=vals,
                           text_zh=(f"飛控還沒宣告落地時就以 {mtxt} 上鎖(上鎖前 5~6 s 高度約 {vals.get('agl_5s_m', float('nan')):.1f} m、"
                                    f"升降速度約 {vals.get('abs_crt_cms', float('nan')):.0f} cm/s)。空中上鎖會讓馬達立即停轉,請確認是否誤觸遙控器。"
                                    if vals.get("agl_5s_m") is not None and vals.get("abs_crt_cms") is not None
                                    else f"飛控還沒宣告落地就以 {mtxt} 上鎖。空中上鎖會讓馬達立即停轉,請確認是否誤觸遙控器。")))
    elif cls == "log_ended_armed":
        vals.update(_tail_voltage(ctx, iv.t1))
        landed = me["landed_before"]
        crit = bool(vals.get("volt_collapsed"))
        sev = "critical" if crit else ("info" if landed else "high")
        status = HIT if (crit or not landed) else SOFT
        out.append(Finding("abnormal_disarm_classifier", DOMAIN, status, sev, t_onset=iv.t1, t_alarm=iv.t1, context="log_ended_armed",
                           values=vals, text_zh=("log 在解鎖狀態下結束(最後資料 %.1f s)" % iv.t1)
                           + (f",且最後 2 s 電壓從 {vals['volt_before_med']:.1f} V 掉到 {vals['volt_tail_min']:.1f} V,疑似斷電 / 電源問題"
                              if crit else ",電壓沒有異常下降;可能是 SD 卡寫入中斷、斷電或直接拔電池")
                           + ("(結束時飛控認為已落地)" if landed else "") + "。"))
    elif cls == "ground_only":
        out.append(Finding("abnormal_disarm_classifier", DOMAIN, CLEAR, "info", context="ground_only", values=vals,
                           text_zh=f"解鎖後一直在地面:這次解鎖期間沒有離地事件(EV28 NOT_LANDED),之後以 {mtxt} 上鎖;沒有起飛,也就沒有降落。"))
    elif cls in ("normal_landing", "normal_end"):
        if cls == "normal_landing":
            if me["dt17"] is not None and me["dt18"] is not None:
                seq = (f"上鎖前 {me['dt17']:.1f} s 判定即將落地、{me['dt18']:.1f} s 判定落地,再以 {mtxt} 上鎖"
                       f"(順序:即將落地 → 落地 → 上鎖)")
            elif me["dt18"] is not None:
                seq = f"上鎖前 {me['dt18']:.1f} s 飛控已判定落地(LAND_COMPLETE),再以 {mtxt} 上鎖"
            else:
                seq = f"上鎖前飛控已處於落地狀態,以 {mtxt} 上鎖"
            out.append(Finding("abnormal_disarm_classifier", DOMAIN, CLEAR, "info", context="normal_landing", values=vals,
                               text_zh=f"正常降落:{seq}。"))
        else:
            out.append(Finding("abnormal_disarm_classifier", DOMAIN, SOFT, "info", context="normal_method_not_landed", values=vals,
                               text_zh=f"以 {mtxt} 上鎖,但上鎖前飛控沒有宣告落地(可能是馬達測試 / 自動上鎖);沒有其他異常。"))
    else:   # unknown:log 沒有任何落地 / 離地事件,無法判斷降落狀態
        out.append(_na("abnormal_disarm_classifier", "no_landing_info", "log 沒有 LAND_COMPLETE / NOT_LANDED 事件,無法判斷上鎖時是否已落地。", vals))
    # 空中緊急停止(EV54):與上鎖分類並列
    ev_t, ev_id = ctx.ev_rows()
    for te_ in ev_t[(ev_id == EV_EMERGENCY_STOP) & (ev_t >= iv.t0) & (ev_t <= iv.t1)]:
        if ctx.landed_at(float(te_), iv) is False:
            out.append(Finding("abnormal_disarm_classifier", DOMAIN, HIT, "critical", t_onset=float(te_), t_alarm=float(te_),
                               context="emergency_stop_in_air", values={"t": _r(te_)},
                               text_zh=f"{te_:.2f} s 在未落地時啟動了馬達緊急停止(EV54),飛行中馬達會立即停轉。"))
            break
    return out


# ================================================================ 4. impact_shock_clusters

def _recognised_after(ctx: Ctx, iv: ArmedInterval, t_start: float, win: float = RECOG_WIN_S) -> float | None:
    """觸地後 win 秒內飛控宣告落地(EV17 / EV18;上鎖當下補寫的不算)的延遲;沒有回傳 None。"""
    ev_t, ev_id = ctx.ev_rows()
    m = np.isin(ev_id, (EV_LAND_MAYBE, EV_LAND_COMPLETE)) & (ev_t >= t_start) & (ev_t <= t_start + win) \
        & (ev_t < iv.t1 - DISARM_EV_EPS)
    return float(ev_t[m][0] - t_start) if m.any() else None


LATE_RECOG_MAX_S = 10.0           # 「較晚才宣告落地」最多往後找 10 s(且不得越過下一次衝擊群):再晚就是另一次降落


def _vibe_tables(ctx: Ctx) -> dict:
    out = {}
    for i in (ctx.instances("VIBE") or [None]):
        tb = ctx.table("VIBE", i)
        if tb is not None and all(k in tb for k in ("VibeX", "VibeY", "VibeZ", "Clip")):
            out[i] = tb
    return out


def _clip_events(vt: dict, t0: float, t1: float) -> list[dict]:
    """Clip 是累計計數器(可能一開始就非 0):只看增量。回傳依時間群組(±0.2 s 跨 instance)的事件:
    {t, n_inst, dc_max, dc_sum}。"""
    raw = []
    for i, tb in vt.items():
        c = tb["Clip"].astype(np.int64)
        dc = np.diff(c)
        for k in np.flatnonzero(dc > 0):
            tt = float(tb["t"][k + 1])
            if t0 <= tt <= t1:
                raw.append((tt, int(dc[k]), i))
    raw.sort()
    groups: list[dict] = []
    for tt, d, i in raw:
        if groups and tt - groups[-1]["t"] <= VIBE_CLIP_SAME_S:
            g = groups[-1]
            g["insts"].add(i)
            g["dc_max"] = max(g["dc_max"], d)
            g["dc_sum"] += d
        else:
            groups.append({"t": tt, "insts": {i}, "dc_max": d, "dc_sum": d})
    for g in groups:
        g["n_inst"] = len(g["insts"])
        del g["insts"]
    return groups


@guarded("impact_shock_clusters", DOMAIN)
def impact_shock_clusters(ctx: Ctx, a: Anchors, sh: _Sh | None = None):
    """加速度衝擊群:撞擊(hard_impact:脈衝峰值≥40 m/s² 且 dV≥3 m/s)、重落地(hard_landing)、觸地(ground_contact,20~40)。

    Findings:每個硬撞擊 / 重落地一筆 HIT;觸地若飛控 2 s 內沒宣告落地:之後 5 s 內有翻覆或硬撞擊 → HIT
    (context 'ground_contact',『觸地後翻覆』的起點),較晚才宣告 → SOFT 'ground_contact_late',都沒有 → SOFT
    'ground_contact_unrecognised'(中性描述);宣告了 → CLEAR('normal_touchdown');完全沒有衝擊 → CLEAR('no_impact')。
    飛行中連續 >0.2 s 的 >20 m/s² 平台(油門推滿 / 急拉,cls 'sustained_accel')不是撞擊,只列在 values。
    IMU 沒有覆蓋整段解鎖區間時不給 CLEAR(NA 'partial_coverage')。
    time:t_onset = 群起點(>20 m/s²),t_alarm = 首次 ≥40 m/s²。完整群清單在 values['clusters'](與 a.impacts)。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("impact_shock_clusters")
    sh = sh or _Sh(ctx, a)
    w = sh.imu(iv.t0, iv.t1)
    if w is None:
        return _impact_vibe_fallback(ctx, a, iv)
    allc = [dict(c) for c in sh.clusters_in(iv)]
    mine = [c for c in allc if c["cls"] in IMPACT_CLASSES]
    sust = [c for c in allc if c["cls"] == "sustained_accel"]
    am = (w.t >= iv.t0) & (w.t <= iv.t1)
    amax = float(np.max(w.acons[am])) if am.any() else float(np.max(w.acons))
    cov = sh.imu_cover(iv)
    base = {"onset": IMPACT_ONSET, "hard_peak": HARD_PEAK, "hard_dv": HARD_DV, "n_clusters": len(mine),
            "n_sustained": len(sust), "acons_max_in_flight": _r(amax, 1), "coverage": cov}
    out: list[Finding] = []
    normal = []
    t_detect = a.t_detect
    hard_all = [c for c in mine if c["cls"] in HARD_CLASSES]
    tumbles = sh.tumble_runs(iv)
    for c in mine:
        c["recognised_dt"] = c["recognised_late_dt"] = c["followed_by"] = c["follow_dt_s"] = None
        if c["cls"] == "ground_contact":
            c["recognised_dt"] = _r(_recognised_after(ctx, iv, c["t_start"]), 2)
            if c["recognised_dt"] is None:
                nxt = [x["t_start"] for x in mine if x["t_start"] > c["t_end"]]
                lim = min([c["t_start"] + LATE_RECOG_MAX_S] + nxt)
                late = _recognised_after(ctx, iv, c["t_start"], lim - c["t_start"])
                c["recognised_late_dt"] = _r(late, 2)
                # 之後有沒有翻覆 / 硬撞擊(沒有的話不能說是「觸地後翻覆的起點」)
                fol = [(r[0] - c["t_start"], "tumble") for r in tumbles if c["t_start"] <= r[0] <= c["t_start"] + TIPOVER_FOLLOW_S]
                fol += [(h["t_start"] - c["t_start"], "hard_impact") for h in hard_all
                        if c["t_end"] < h["t_start"] <= c["t_start"] + TIPOVER_FOLLOW_S]
                if fol:
                    dt_f, what = min(fol)
                    c["followed_by"], c["follow_dt_s"] = what, _r(dt_f, 2)
        # 撞擊前的姿態(hypotheses 的外力撞擊閘門:撞擊前 ae<10° 才可能是「外力撞擊在先」)
        e = sh.ae(c["t_start"] - IMPACT_AE_PRE_S - 0.1, c["t_start"] + 0.1)
        c["ae_at_impact"] = c["tilt_at_impact"] = None
        if e is not None:
            m = (e[0] >= c["t_start"] - IMPACT_AE_PRE_S) & (e[0] <= c["t_start"])
            if m.any():
                c["ae_at_impact"] = _r(float(np.max(e[1][m])), 1)
                c["tilt_at_impact"] = _r(float(np.max(e[2][m])), 1)
    hard = [c for c in mine if c["cls"] in HARD_CLASSES]
    for c in hard[:MAX_IMPACT_FINDINGS]:
        sev = "critical" if c["cls"] == "hard_impact" else "high"
        peak, g = c["peak"], c["peak"] / 9.81
        lag = ""
        if t_detect is not None and t_detect > c["t_peak"] and c is hard[-1]:
            lag = f";飛控約晚 {t_detect - c['t_peak']:.1f} s 才判定失事"
        name = "撞擊等級衝擊" if c["cls"] == "hard_impact" else "重落地衝擊(峰值或 dV 其一超標)"
        gyro = f"、角速度峰值 {c['gyro_peak_dps']:.0f}°/s" if c.get("gyro_peak_dps") is not None else ""
        pre = ""
        if c["ae_at_impact"] is not None:
            pre = (f";撞擊前 {IMPACT_AE_PRE_S:.1f} s 姿態誤差最大 {c['ae_at_impact']:.0f}°"
                   + ("(姿態原本正常,撞擊在先)" if c["ae_at_impact"] < AE_WARN_DEG else "(撞擊前已失去姿態控制,撞擊是結果)"))
        out.append(Finding("impact_shock_clusters", DOMAIN, HIT, sev, t_onset=c["t_start"], t_alarm=c["t_hard"] or c["t_start"],
                           context=c["cls"], values={**base, **c},
                           text_zh=(f"{c['t_peak']:.2f} s {name}:合加速度峰值 {peak:.0f} m/s²(約 {g:.1f} g)、速度變化 {c['dV']:.1f} m/s{gyro},"
                                    f"力的方向{_DIR_ZH.get(c['dir'], '不明')}{pre}{lag}。正常觸地約 25~29 m/s²,這次明顯更強;請檢查機架、槳葉與起落架。")))
    for c in mine:
        if c["cls"] != "ground_contact":
            continue
        if c["recognised_dt"] is not None:
            normal.append(c)
            continue
        head = (f"{c['t_start']:.2f} s 觸地(合加速度 {c['peak']:.0f} m/s²,約 {c['peak'] / 9.81:.1f} g,速度變化 {c['dV']:.1f} m/s)")
        nodecl = f"飛控在 {RECOG_WIN_S:.0f} s 內沒有宣告落地(無 LAND_COMPLETE_MAYBE / LAND_COMPLETE)"
        if (iv.t1 - c["t_start"]) < RECOG_WIN_S:
            st, ctxt, txt = SOFT, "ground_contact_at_disarm", head + ",隨即上鎖,來不及判定。"
        elif c["followed_by"] is not None:
            what = "開始翻覆(傾角 ≥45°)" if c["followed_by"] == "tumble" else "出現硬撞擊"
            st, ctxt = HIT, "ground_contact"
            txt = head + f",但{nodecl},飛控仍以為在空中;{c['follow_dt_s']:.1f} s 後{what},這是『觸地後翻覆』的起點。"
        elif c["recognised_late_dt"] is not None:
            st, ctxt = SOFT, "ground_contact_late"
            txt = head + f",飛控 {c['recognised_late_dt']:.1f} s 後才宣告落地(比 {RECOG_WIN_S:.0f} s 慢;之後沒有翻覆或硬撞擊)。"
        else:
            st, ctxt = SOFT, "ground_contact_unrecognised"
            txt = head + f",{nodecl};之後 {TIPOVER_FOLLOW_S:.0f} s 內沒有翻覆或硬撞擊(可能是觸地彈起 / 重飛)。"
        out.append(Finding("impact_shock_clusters", DOMAIN, st, "info", t_onset=c["t_start"], t_alarm=c["t_start"],
                           context=ctxt, values={**base, **c}, text_zh=txt))
    sust_txt = ""
    if sust:
        s0 = max(sust, key=lambda c: c["sustained_s"] or 0.0)
        sust_txt = (f"另有 {len(sust)} 段持續 >{IMPACT_ONSET:.0f} m/s² 的加速度(最長 {s0['sustained_s']:.2f} s、峰值 {s0['peak']:.0f} m/s²,"
                    f"是油門推滿 / 急拉之類的平台,不是撞擊的短脈衝)。")
    clist = [{k: c[k] for k in ("t_start", "t_peak", "t_end", "peak", "peak_impulse", "dV", "cls")} for c in allc]
    if not out:
        if normal:
            c = normal[-1]
            clean = (f"{c['t_start']:.2f} s 正常觸地:{c['peak']:.0f} m/s²(約 {c['peak'] / 9.81:.1f} g)、速度變化 {c['dV']:.1f} m/s,"
                     f"飛控在 {c['recognised_dt']:.1f} s 內宣告落地;沒有撞擊等級衝擊。" + sust_txt)
            ctxt = "normal_touchdown"
            t_on = c["t_start"]
        else:
            clean = f"解鎖期間合加速度最大 {amax:.1f} m/s²(撞擊門檻 {IMPACT_ONSET:.0f}),沒有撞擊事件。" + sust_txt
            ctxt, t_on = "no_impact", None
        ci = _classify_interval(ctx, iv)
        if ci["class"] in ("manual_disarm_in_air", "failsafe_disarm") and not ci["landed_before"]:
            clean += (f"注意:上鎖時飛控仍認為在空中,上鎖後的墜落與落地撞擊在分析視窗(上鎖後 {IMPACT_POST_S:.0f} s)之外,"
                      "這裡只代表『上鎖前沒有撞擊』。")
        if not cov["ok"]:
            return _partial_na("impact_shock_clusters", "IMU", cov, clean, {**base, "clusters": clist})
        out.append(Finding("impact_shock_clusters", DOMAIN, CLEAR, "info", t_onset=t_on, context=ctxt,
                           values={**base, "clusters": clist}, text_zh=clean))
    else:
        for f in out:
            f.values["clusters"] = clist
            if not cov["ok"]:
                f.text_zh += f"(注意:{_cover_txt('IMU', cov)},其他時段的撞擊可能沒記到。)"
        out.sort(key=lambda f: f.t_onset if f.t_onset is not None else 0.0)
    return out


def _impact_vibe_fallback(ctx: Ctx, a: Anchors, iv: ArmedInterval):
    """沒有高取樣 IMU:退而用 VIBE(10 Hz)的 Clip 增量 / Vibe>30 找撞擊(只能給 SOFT)。"""
    vt = _vibe_tables(ctx)
    if not vt:
        return _na("impact_shock_clusters", "no_imu", "缺少 IMU 與 VIBE 資料,無法偵測撞擊。")
    ev = [g for g in _clip_events(vt, iv.t0, iv.t1 + VIBE_TAIL_S) if g["n_inst"] >= 2 or g["dc_max"] >= VIBE_CLIP_BIG]
    if ev:
        return Finding("impact_shock_clusters", DOMAIN, SOFT, "high", t_onset=ev[0]["t"], t_alarm=ev[0]["t"], context="vibe_fallback",
                       values={"clip_events": [{"t": _r(g["t"]), "n_inst": g["n_inst"], "dc_max": g["dc_max"]} for g in ev[:5]]},
                       text_zh=f"沒有高取樣 IMU;VIBE 的 Clip 在 {ev[0]['t']:.2f} s 大幅增加,疑似撞擊(僅供參考)。")
    return _na("impact_shock_clusters", "no_imu", "缺少高取樣 IMU,VIBE 也沒有削波增量;無法判斷有無撞擊。")


# ================================================================ 5. free_fall_run

@guarded("free_fall_run", DOMAIN)
def free_fall_run(ctx: Ctx, a: Anchors, sh: _Sh | None = None):
    """自由落體:合加速度共識 <3 m/s² 持續 ≥0.10 s(時間戳空洞 >20 ms 斷開)。

    先於姿態異常(ae<10°)→ 空中失效的證據(context 'before_attitude_error');翻覆途中才出現 → 後果('in_tumble')。
    124:單一 0.157 s(最低 0.85 m/s²,至少落差 0.12 m),在翻覆之中;119 手揮最低 3.6 → 門檻不能放寬到 <5。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("free_fall_run")
    sh = sh or _Sh(ctx, a)
    w = sh.imu(iv.t0, iv.t1)
    if w is None:
        return _na("free_fall_run", "no_imu", "缺少 IMU 加速度資料,無法偵測自由落體。")
    runs = _runs(w.acons < FF_ACC, w.t, FF_MIN_S, FF_GAP_S)
    cov = sh.imu_cover(iv)
    base = {"acc_limit": FF_ACC, "min_dur_s": FF_MIN_S, "acons_min": _r(float(np.min(w.acons)), 2), "coverage": cov}
    if not runs:
        clean = f"解鎖期間合加速度最低 {float(np.min(w.acons)):.1f} m/s²(自由落體門檻 <{FF_ACC:.0f} 持續 ≥{FF_MIN_S:.1f} s),沒有自由落體。"
        if not cov["ok"]:
            return _partial_na("free_fall_run", "IMU", cov, clean, base)
        return Finding("free_fall_run", DOMAIN, CLEAR, "info", context="no_free_fall", values=base, text_zh=clean)
    hard = [c for c in sh.clusters_in(iv) if c["cls"] in HARD_CLASSES]
    items = []
    for ts, te, i0, i1 in runs:
        dur = te - ts
        e = sh.ae(ts - 1.0, ts + 0.2)
        ae_before = None
        if e is not None:
            m = (e[0] >= ts - 0.5) & (e[0] <= ts)
            ae_before = float(np.max(e[1][m])) if m.any() else float(e[1][np.argmin(np.abs(e[0] - ts))])
        late = a.t_warn is not None and ts >= a.t_warn
        ctxt = "unknown_attitude" if ae_before is None else ("in_tumble" if (late or ae_before >= AE_WARN_DEG) else "before_attitude_error")
        then = any(0.0 <= c["t_start"] - te <= FF_THEN_IMPACT_S for c in hard)
        items.append({"t_start": _r(ts), "t_end": _r(te), "dur_s": _r(dur), "min_acc": _r(np.min(w.acons[i0:i1 + 1]), 2),
                      "raw_min_acc": _r(np.min(w.n0[i0:i1 + 1]), 2),
                      "drop_m": _r(0.5 * 9.81 * dur * dur, 2), "ae_before_deg": _r(ae_before, 1), "context": ctxt, "then_impact": then})
    # 代表那一筆:先於姿態異常的優先,其次最長
    first_air = [x for x in items if x["context"] == "before_attitude_error"]
    pick = first_air[0] if first_air else max(items, key=lambda x: x["dur_s"])
    ctxt = pick["context"]
    sev = "high" if ctxt == "before_attitude_error" else "medium"
    where = {"before_attitude_error": "發生在姿態誤差出現之前(空中先失去推力的跡象)", "in_tumble": "發生在翻覆途中,屬於後果(機身被翻起後落下)",
             "unknown_attitude": "(缺姿態資料,無法判斷前後關係)"}[ctxt]
    tail = ",緊接著硬撞擊(自由落體後撞地)" if pick["then_impact"] else ""
    return Finding("free_fall_run", DOMAIN, HIT, sev, t_onset=pick["t_start"], t_alarm=pick["t_start"] + FF_MIN_S, context=ctxt,
                   values={**base, **pick, "runs": items[:5], "n_runs": len(items)},
                   text_zh=f"{pick['t_start']:.2f} s 起合加速度 <{FF_ACC:.0f} m/s² 持續 {pick['dur_s']:.2f} s(最低 {pick['raw_min_acc']:.1f}),"
                           f"至少落下 {pick['drop_m']:.2f} m;{where}{tail}。")


# ================================================================ 6. tumble_inversion

def _acc_tilt_series(w: _ImuWin, win_s: float):
    """跨 instance 平均的比力 → win_s 平均 → 重力方向傾角(度)。"""
    acc = np.mean(np.stack(w.acc, axis=0), axis=0)
    dt = float(np.median(np.diff(w.t))) if w.t.size > 1 else 0.0025
    k = max(1, int(round(win_s / max(dt, 1e-4))))
    return _tilt_from_acc(_movmean(acc, k))


@guarded("tumble_inversion", DOMAIN)
def tumble_inversion(ctx: Ctx, a: Anchors, sh: _Sh | None = None):
    """翻覆 / 倒扣:傾角 = arccos(cosRoll·cosPitch)(跨 ±180° 不跳)。TUMBLE:≥45° 持續 ≥0.3 s;INVERTED:≥120° 持續 ≥0.5 s。

    只看解鎖期間且截止於上鎖(上鎖後 log 還記 ~28 s 倒扣畫面);ACRO / SPORT / FLIP 與 EV21~22 期間不算(刻意翻轉)。
    用加速度重力方向(0.25 s 平均)獨立驗證:姿態說翻覆、加速度也說傾斜 → HIT;只有姿態 → SOFT(可能是姿態估計壞掉)。
    自己從 ANG/ATT 算傾角,不依賴別的模組的 t_warn / t_inverted。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("tumble_inversion")
    sh = sh or _Sh(ctx, a)
    att = sh.att(iv.t0, iv.t1)
    if att is None:
        return _na("tumble_inversion", "no_attitude", "缺少姿態(ANG / ATT)資料,無法判斷翻覆。")
    m_iv = (att["t"] >= iv.t0) & (att["t"] <= iv.t1)
    t = att["t"][m_iv]
    if t.size < 3:
        return _na("tumble_inversion", "no_attitude", "解鎖期間姿態資料不足,無法判斷翻覆。")
    tilt = tilt_deg(att["roll"][m_iv], att["pitch"][m_iv])
    cmd = _commanded_mask(ctx, t)
    valid = ~cmd
    runs_t = _runs(valid & (tilt >= TUMBLE_TILT_DEG), t, TUMBLE_MIN_S, TILT_GAP_S)
    runs_i = _runs(valid & (tilt >= INVERT_TILT_DEG), t, INVERT_MIN_S, TILT_GAP_S)
    cov = sh.att_cover(iv)
    base = {"tumble_deg": TUMBLE_TILT_DEG, "tumble_min_s": TUMBLE_MIN_S, "invert_deg": INVERT_TILT_DEG, "invert_min_s": INVERT_MIN_S,
            "attitude_src": att["src"], "tilt_max_deg": _r(float(np.max(tilt[valid])) if valid.any() else float(np.max(tilt)), 1),
            "commanded_samples": int(cmd.sum()), "coverage": cov}
    if not runs_t:
        if cmd.any() and _runs(cmd & (tilt >= TUMBLE_TILT_DEG), t, TUMBLE_MIN_S, TILT_GAP_S):
            return _na("tumble_inversion", "commanded_flip", "有大角度傾斜,但發生在特技 / 翻滾模式(ACRO / SPORT / FLIP)或 FLIP 事件期間,屬刻意動作,不判定為失控翻覆。", base)
        clean = f"解鎖期間機身最大傾角 {base['tilt_max_deg']:.1f}°(翻覆門檻 {TUMBLE_TILT_DEG:.0f}°),沒有翻覆。"
        if not cov["ok"]:
            return _partial_na("tumble_inversion", f"姿態({att['src']})", cov, clean, base)
        return Finding("tumble_inversion", DOMAIN, CLEAR, "info", context="no_tumble", values=base, text_zh=clean)
    ts, te, i0, i1 = runs_t[0]
    inv = [r for r in runs_i if r[1] >= ts - TILT_GAP_S][:1]
    inverted = bool(inv)
    # 時間軸:最後一段 ≥10° 的起點 → ≥150° 的時間;繞哪一軸翻
    r10 = [r for r in _runs(valid & (tilt >= 10.0), t, 0.0, TILT_GAP_S) if r[0] <= ts <= r[1]]
    t10 = r10[0][0] if r10 else None
    r150 = [r for r in _runs(valid & (tilt >= 150.0), t, 0.0, TILT_GAP_S) if r[1] >= ts]
    t150 = r150[0][0] if r150 else None
    r0 = float(att["roll"][m_iv][i0])
    r0 = (r0 + 180.0) % 360.0 - 180.0
    axis = "roll" if abs(r0) >= abs(float(att["pitch"][m_iv][i0])) else "pitch"
    vals = {**base, "t_tumble": _r(ts), "t_tumble_end": _r(te), "axis": axis, "t_tilt10": _r(t10), "t_tilt150": _r(t150),
            "time_to_invert_s": _r(t150 - t10) if (t150 is not None and t10 is not None) else None}
    if inverted:
        its, ite = inv[0][0], inv[0][1]
        vals.update({"t_inverted": _r(its), "inverted_s": _r(ite - its)})
    # 加速度重力方向獨立驗證
    confirmed = None
    w = sh.imu(iv.t0, iv.t1)
    if w is not None:
        tacc = _acc_tilt_series(w, ACC_TILT_WIN_S)
        mm = (w.t >= ts - 0.5) & (w.t <= te + 1.0)
        # 翻覆前後沒有 IMU 樣本 = 無法驗證(None),不是「加速度計說沒傾斜」(False)
        confirmed = bool(np.max(tacc[mm]) >= ACC_TILT_CONFIRM_DEG) if mm.sum() >= 10 else None
        vals["accel_confirmed"] = confirmed
        vals["accel_tilt_max_deg"] = _r(float(np.max(tacc[mm])) if mm.any() else None, 1)
        f1 = _runs(tacc >= 15.0, w.t, 0.0, TILT_GAP_S)
        f1 = [r for r in f1 if r[1] >= ts - 5.0 and r[0] <= ts + 1.0]
        vals["t_acc_tilt15"] = _r(f1[-1][0]) if f1 else None
        t1s = _acc_tilt_series(w, 1.0)
        fallen = [r for r in _runs(t1s > ACC_TILT_FALLEN_DEG, w.t, ACC_TILT_FALLEN_S, TILT_GAP_S)]
        vals["accel_fallen_t"] = _r(fallen[0][0]) if fallen else None
    else:
        vals["accel_confirmed"] = None
    status = SOFT if confirmed is False else HIT
    sev = "critical" if inverted else "high"
    ctxt = "inverted" if inverted else "tumble"
    if status == SOFT:
        ctxt += "_attitude_only"
    ax = "滾轉(Roll)" if axis == "roll" else "俯仰(Pitch)"
    txt = f"機身傾角 ≥{TUMBLE_TILT_DEG:.0f}° 自 {ts:.2f} s 起(繞{ax}軸翻)"
    if inverted:
        txt += f",{vals['t_inverted']:.2f} s 起超過 {INVERT_TILT_DEG:.0f}° 倒扣"
    if vals["time_to_invert_s"] is not None:
        txt += f";從開始傾斜(>10°)到倒扣(>150°)約 {vals['time_to_invert_s']:.1f} s"
    if confirmed is True:
        txt += ";加速度計的重力方向也獨立證實(非姿態估計誤差)"
    elif confirmed is False:
        txt += ";但加速度計的重力方向沒有傾斜,姿態估計可能有問題"
    txt += "。不是刻意切到特技 / 翻滾模式的話,這就是失控翻覆。"
    return Finding("tumble_inversion", DOMAIN, status, sev, t_onset=ts, t_alarm=ts + TUMBLE_MIN_S, context=ctxt, values=vals, text_zh=txt)


# ================================================================ 7. vibe_clip_transient

def _merge_runs(runs: list, gap: float) -> list:
    """[(ts, te, peak)] 依時間排序後,間隔 ≤gap 的合併。"""
    out: list = []
    for ts, te, pk in sorted(runs):
        if out and ts - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], te), max(out[-1][2], pk))
        else:
            out.append((ts, te, pk))
    return out


@guarded("vibe_clip_transient", DOMAIN)
def vibe_clip_transient(ctx: Ctx, a: Anchors, sh: _Sh | None = None):
    """撞擊造成的 VIBE 暴增 / Clip 躍升(後果),或沒有撞擊的持續振動 / 削波(原因候選)。

    Clip 是累計計數器(119 一開始就是 2):只看增量。VIBE 10 Hz,Clip 躍升的時間戳可晚於 IMU 峰值 ~0.25 s。
    分類(撞擊 = IMU 衝擊群;飛行中的持續加速 'sustained_accel' 不算撞擊):
      impact_transient:<3 s 的 VIBE 超標且與硬撞擊 ±1.5 s 重疊;或 ≥2 顆 IMU 同一筆 Clip 躍升且在衝擊群(硬撞擊 / 觸地)±1 s 內;
                        或單顆 dc≥5 且在硬撞擊 ±1 s 內。沒有對應的衝擊群就不是撞擊暫態(多顆 IMU 同時削波正是振動削波的常態)。
      sustained_vibration:≥5 s 或 30 s 視窗內 ≥20% 超標,且不在撞擊附近。
      vibration_clipping:不在撞擊附近的 Clip 增量 ≥3 次(單顆或多顆);1~2 次 → SOFT clip_increment。
    124:Clip 0→12/12/1 於 1259.74(末撞 1259.463~1259.958 之內),首次 VIBE>30 於 1258.44;撞擊前 p95 9.3~9.5、最大 11.2 → 振動不是原因。
    """
    iv = a.interval
    if iv is None:
        return _not_armed("vibe_clip_transient")
    sh = sh or _Sh(ctx, a)
    vt = _vibe_tables(ctx)
    if not vt:
        return _na("vibe_clip_transient", "no_vibe", "缺少 VIBE 資料,無法判斷振動 / 削波。")
    t0, t1 = iv.t0, iv.t1 + VIBE_TAIL_S
    imu_ok = sh.imu(iv.t0, iv.t1) is not None
    cl = [c for c in sh.clusters_in(iv) if c["cls"] in IMPACT_CLASSES]
    hard = [c for c in cl if c["cls"] in HARD_CLASSES]

    def near(tt, pad, cs):
        return [c for c in cs if c["t_start"] - pad <= tt <= c["t_end"] + pad]

    clips = _clip_events(vt, t0, t1)
    imp_clips = [g for g in clips if (g["n_inst"] >= 2 and near(g["t"], VIBE_ASSOC_CLIP_S, cl))
                 or (g["dc_max"] >= VIBE_CLIP_BIG and near(g["t"], VIBE_ASSOC_CLIP_S, hard))]
    other_clips = [g for g in clips if g not in imp_clips and not near(g["t"], VIBE_ASSOC_CLIP_S, hard)]
    assoc_gc = [c for c in cl if c["cls"] == "ground_contact" and any(near(g["t"], VIBE_ASSOC_CLIP_S, [c]) for g in imp_clips)]
    starts = [c["t_start"] for c in hard + assoc_gc]
    t_cut = min(starts) if starts else None          # 「撞擊前」統計的截止點(沒有撞擊 = 整段解鎖期間)
    # 各 instance 的最大軸 → 超標區段;VIBE 覆蓋(取覆蓋最好的 instance)
    runs_all, pre = [], []
    n_samples = 0
    cov = None
    for i, tb in vt.items():
        ci = coverage(tb["t"], iv.t0, iv.t1)
        if cov is None or ci["frac"] > cov["frac"]:
            cov = ci
        m = (tb["t"] >= t0) & (tb["t"] <= t1)
        if m.sum() < 3:
            continue
        tt = tb["t"][m]
        mx = np.maximum.reduce([tb["VibeX"][m], tb["VibeY"][m], tb["VibeZ"][m]])
        n_samples = max(n_samples, int(m.sum()))
        for ts, te, i0, i1 in _runs(mx > VIBE_BAD, tt, 0.0, VIBE_RUN_MERGE_S):
            runs_all.append((ts, te, float(mx[i0:i1 + 1].max())))
        cut = t_cut if t_cut is not None else iv.t1
        mp = (tt >= iv.t0 + VIBE_SKIP_S) & (tt < cut)
        if mp.sum() >= 5:
            pre.append((float(np.percentile(mx[mp], 95)), float(mx[mp].max())))
    if n_samples == 0:
        return _na("vibe_clip_transient", "no_vibe", "解鎖期間沒有 VIBE 資料。")
    runs = _merge_runs(runs_all, VIBE_RUN_MERGE_S)
    pre_p95 = max((p for p, _ in pre), default=None)
    pre_max = max((m_ for _, m_ in pre), default=None)
    clip_before = sum(g["dc_sum"] for g in clips if t_cut is None or g["t"] < t_cut)
    base = {"vibe_limit": VIBE_BAD, "pre_impact_p95": _r(pre_p95, 1), "pre_impact_max": _r(pre_max, 1),
            "stats_window": "pre_impact" if t_cut is not None else "armed",
            "clip_events": [{"t": _r(g["t"]), "n_inst": g["n_inst"], "dc_max": g["dc_max"]} for g in clips[:5]],
            "n_clip_events": len(clips), "clip_before_first_hit": int(clip_before), "coverage": cov}
    no_imu_txt = "" if imu_ok else "(沒有 IMU 資料,無法確認是否為撞擊)"
    out: list[Finding] = []
    # --- 1. 撞擊暫態(必須真的對應到 IMU 衝擊群)
    imp_runs = [r for r in runs if (r[1] - r[0]) < VIBE_TRANSIENT_MAX_S
                and (near(r[0], VIBE_ASSOC_RUN_S, hard) or near(r[1], VIBE_ASSOC_RUN_S, hard))]
    if imp_runs or imp_clips:
        t_on = min([r[0] for r in imp_runs] + [g["t"] for g in imp_clips])
        pk = max([r[2] for r in imp_runs], default=None)
        dur = (max([r[1] for r in imp_runs] + [g["t"] for g in imp_clips]) - t_on)
        if imp_runs:
            rel = near(imp_runs[0][0], VIBE_ASSOC_RUN_S, hard) or near(imp_runs[0][1], VIBE_ASSOC_RUN_S, hard)
        else:
            rel = near(imp_clips[0]["t"], VIBE_ASSOC_CLIP_S, cl)
        rc = min(rel, key=lambda c: abs(c["t_peak"] - t_on))
        what = "硬撞擊" if rc["cls"] in HARD_CLASSES else "觸地衝擊"
        vals = {**base, "t_first_over": _r(imp_runs[0][0]) if imp_runs else None, "peak_vibe": _r(pk, 1), "transient_s": _r(dur, 2),
                "clip_total": int(sum(g["dc_sum"] for g in imp_clips)), "impact_t_peak": rc["t_peak"], "impact_cls": rc["cls"]}
        g0 = imp_clips[0] if imp_clips else None
        clip_txt = ""
        if g0 is not None:
            clip_txt = (f"Clip 於 {g0['t']:.2f} s 累計增加({g0['n_inst']} 顆 IMU 同時)" if g0["n_inst"] >= 2
                        else f"Clip 於 {g0['t']:.2f} s 單顆增加 {g0['dc_max']} 次")
        vib_txt = f"VIBE 自 {imp_runs[0][0]:.2f} s 起短暫衝到 {pk:.0f} m/s²(>30)" if imp_runs else ""
        pre_txt = (f"撞擊前正常飛行的振動 p95 只有 {pre_p95:.1f}、最大 {pre_max:.1f} m/s²,Clip 增量 {clip_before} 次"
                   if pre_p95 is not None else "")
        out.append(Finding("vibe_clip_transient", DOMAIN, HIT, "high", t_onset=t_on, t_alarm=t_on, context="impact_transient", values=vals,
                           text_zh="、".join(x for x in (vib_txt, clip_txt) if x)
                           + f",緊跟著 {rc['t_peak']:.2f} s 的{what}(IMU 衝擊群峰值 {rc['peak']:.0f} m/s²),是撞擊造成的結果,不是振動問題。"
                           + (pre_txt + "。" if pre_txt else "")))
    # --- 2. 持續振動(不在撞擊附近)
    sustained = []
    for ts, te, pk in runs:
        if (te - ts) >= VIBE_SUSTAINED_S and not near(ts, VIBE_ASSOC_RUN_S, hard):
            sustained.append((ts, te, pk))
    if not sustained:   # 30 s 視窗內 ≥20% 超標
        for i, tb in vt.items():
            m = (tb["t"] >= t0) & (tb["t"] <= t1)
            if m.sum() < 30:
                continue
            tt = tb["t"][m]
            mx = np.maximum.reduce([tb["VibeX"][m], tb["VibeY"][m], tb["VibeZ"][m]])
            over = (mx > VIBE_BAD) & ~np.array([bool(near(x, VIBE_ASSOC_RUN_S, hard)) for x in tt]) if hard else (mx > VIBE_BAD)
            dt = float(np.median(np.diff(tt)))
            k = max(1, int(round(VIBE_FRAC_WIN_S / dt)))
            if over.size >= k:
                c = np.cumsum(np.r_[0, over.astype(float)])
                frac = (c[k:] - c[:-k]) / k
                j = np.flatnonzero(frac >= VIBE_FRAC)
                if j.size:
                    sustained.append((float(tt[j[0]]), float(tt[min(j[0] + k, tt.size - 1)]), float(mx[over].max())))
                    break
    if sustained:
        ts, te, pk = sustained[0]
        sev = "critical" if pk >= VIBE_CRIT else "high"
        out.append(Finding("vibe_clip_transient", DOMAIN, HIT, sev, t_onset=ts, t_alarm=ts + VIBE_SUSTAINED_S, context="sustained_vibration",
                           values={**base, "t_start": _r(ts), "dur_s": _r(te - ts, 1), "peak_vibe": _r(pk, 1)},
                           text_zh=f"{ts:.2f} s 起 VIBE 持續超過 {VIBE_BAD:.0f} m/s²(約 {te - ts:.0f} s,峰值 {pk:.0f}),且不在撞擊附近,"
                                   f"是真正的振動問題;請查槳葉平衡、馬達軸承與飛控避震。"))
    # --- 3. 削波(不在撞擊附近的 Clip 增量,單顆或多顆 IMU 都算)
    n_multi = sum(1 for g in other_clips if g["n_inst"] >= 2)
    multi_txt = f",其中 {n_multi} 次多顆 IMU 同時" if n_multi else ""
    if len(other_clips) >= VIBE_CLIP_REPEAT_N:
        g0 = other_clips[0]
        out.append(Finding("vibe_clip_transient", DOMAIN, HIT, "high", t_onset=g0["t"], t_alarm=other_clips[VIBE_CLIP_REPEAT_N - 1]["t"],
                           context="vibration_clipping", values={**base, "n_other_clips": len(other_clips), "n_multi_imu_clips": n_multi},
                           text_zh=f"在沒有撞擊的時段,{g0['t']:.2f} s 起多次(共 {len(other_clips)} 次{multi_txt})加速度計削波{no_imu_txt},"
                                   f"振動讓感測器超出量程;請查機身振動與避震。"))
    elif other_clips and not out:
        g0 = other_clips[0]
        inst = f"({g0['n_inst']} 顆 IMU 同時)" if g0["n_inst"] >= 2 else ""
        out.append(Finding("vibe_clip_transient", DOMAIN, SOFT, "info", t_onset=g0["t"], t_alarm=g0["t"], context="clip_increment",
                           values={**base, "n_other_clips": len(other_clips), "n_multi_imu_clips": n_multi},
                           text_zh=f"{g0['t']:.2f} s 加速度計削波增加 {g0['dc_max']} 次{inst}(共 {len(other_clips)} 次{multi_txt}),"
                                   f"附近沒有 IMU 衝擊群{no_imu_txt};次數不多,先觀察。"))
    # --- 4. 短暫超標但無對應撞擊
    if not out:
        loose = [r for r in runs if (r[1] - r[0]) < VIBE_SUSTAINED_S]
        if loose:
            r = loose[0]
            out.append(Finding("vibe_clip_transient", DOMAIN, SOFT, "info", t_onset=r[0], t_alarm=r[0], context="short_burst",
                               values={**base, "peak_vibe": _r(r[2], 1)},
                               text_zh=f"{r[0]:.2f} s 起 VIBE 短暫超過 {VIBE_BAD:.0f} m/s²(峰值 {r[2]:.0f}),沒有對應的撞擊事件,可能是碰撞或槳葉擦到,先觀察。"))
    if not out:
        mxv = max((float(np.max(np.maximum.reduce([tb["VibeX"], tb["VibeY"], tb["VibeZ"]])[(tb["t"] >= t0) & (tb["t"] <= t1)]))
                   for tb in vt.values() if ((tb["t"] >= t0) & (tb["t"] <= t1)).any()), default=0.0)
        base["vibe_max"] = _r(mxv, 1)
        clean = f"解鎖期間 VIBE 最大 {mxv:.1f} m/s²(門檻 {VIBE_BAD:.0f}),加速度計沒有削波增量,沒有振動 / 削波問題。"
        if cov is not None and not cov["ok"]:
            out.append(_partial_na("vibe_clip_transient", "VIBE", cov, clean, base))
        else:
            out.append(Finding("vibe_clip_transient", DOMAIN, CLEAR, "info", context="no_vibe_clip", values=base, text_zh=clean))
    for f in out:       # 設計目錄的分類名稱(hypotheses 以 values['type'] 篩選)
        f.values["type"] = _VIBE_TYPE.get(f.context, f.context)
    return out


# ================================================================ 結束型態定案(dataflow STEP 2 / 3)

_DETECTOR_ORDER = (crash_check_anchor, crash_check_replay, impact_shock_clusters, free_fall_run, tumble_inversion,
                   vibe_clip_transient, abnormal_disarm_classifier)


def _first(findings: list[Finding], det_id: str, pred=None) -> Finding | None:
    for f in findings:
        if f.id == det_id and (pred is None or pred(f)):
            return f
    return None


def classify_end(ctx: Ctx, a: Anchors, findings: list[Finding] | None = None, sh: _Sh | None = None) -> str:
    """把這次解鎖的結束型態定案並填錨點(dataflow STEP 2 / 3);回傳 end_type。

    優先序:(1) CRASH_CHECK ERR / ARM.Method 12 → crash_check;(2) 重算觸發 → crash_check_replay;
    (3) 空中手動上鎖 / 失效保護上鎖 → abnormal_disarm;(4) 上鎖前 6 s~後 1 s 有 critical 硬撞擊 → hard_impact;
    (5) log 在解鎖中結束 → log_ended_armed;(6) 倒扣 ≥1 s → tumble;其餘 normal_landing / normal_end。
    填:end_type、t_detect、t_disarm、impacts(本次解鎖的撞擊 / 觸地群;不含 sustained_accel)、t_first_hit、t_impact_peak
    (依結束型態,見 _pick_impact_anchors;正常結束不填)、t_rest。ground_only(沒起飛)→ normal_end。
    不填 t_warn / t_loss / t_inverted / t_touch。
    """
    iv = a.interval
    if iv is None:
        a.end_type = "not_armed"
        return a.end_type
    sh = sh or _Sh(ctx, a)
    if findings is None:
        findings = []
        for fn in _DETECTOR_ORDER:
            findings += fn(ctx, a, sh)
    clusters = sh.clusters_in(iv)
    a.t_disarm = iv.t1
    anchor = _first(findings, "crash_check_anchor", lambda f: f.status == HIT)
    replay = _first(findings, "crash_check_replay", lambda f: f.status == HIT)
    disarm = _first(findings, "abnormal_disarm_classifier")
    cls = (disarm.values.get("class") if disarm is not None and disarm.status != NA else None)
    tumble = _first(findings, "tumble_inversion", lambda f: f.status == HIT and f.context == "inverted")
    crit_impacts = [c for c in clusters if c["cls"] == "hard_impact"
                    and iv.t1 - IMPACT_WINDOW_PRE_S <= c["t_peak"] <= iv.t1 + IMPACT_POST_S]
    t_detect = None
    why = ""
    if anchor is not None:
        end_type, t_detect, why = "crash_check", anchor.t_alarm, "CRASH_CHECK(ERR 子系統 12)"
    elif cls == "crash_check":
        end_type, t_detect, why = "crash_check", iv.t1, "ARM.Method=12(墜機上鎖)"
    elif replay is not None:
        end_type, t_detect, why = "crash_check_replay", replay.t_alarm, "重算的 CRASH_CHECK 條件成立(log 沒有該事件)"
    elif cls in ("manual_disarm_in_air", "failsafe_disarm"):
        end_type, t_detect, why = "abnormal_disarm", iv.t1, "異常上鎖:" + cls
    elif crit_impacts:
        end_type, t_detect, why = "hard_impact", iv.t1, "上鎖前後有 critical 硬撞擊但飛控沒有墜機判定"
    elif iv.open_end or cls == "log_ended_armed":
        end_type, t_detect, why = "log_ended_armed", iv.t1, "log 在解鎖中結束"
    elif tumble is not None and tumble.values.get("inverted_s", 0.0) >= TUMBLE_END_INV_S:
        end_type, t_detect, why = "tumble", (tumble.values.get("t_inverted", tumble.t_onset) + TUMBLE_END_INV_S), "倒扣 ≥1 s"
    elif cls == "normal_landing":
        end_type, why = "normal_landing", "先判定落地再上鎖"
    elif cls == "ground_only":
        end_type, why = "normal_end", "解鎖後一直在地面(沒有離地事件),沒有起飛"
    else:
        end_type, why = "normal_end", "沒有異常結束的證據"
    abnormal = end_type not in ("normal_landing", "normal_end")
    a.end_type = end_type
    a.t_detect = _r(t_detect) if t_detect is not None else None
    if end_type == "log_ended_armed":
        a.t_disarm = iv.t1
    a.impacts = [{k: c.get(k) for k in ("t_start", "t_peak", "t_end", "peak", "dV", "cls", "t_hard", "gyro_peak_dps", "dir", "u", "width40_s", "interval")}
                 for c in clusters if c["cls"] in IMPACT_CLASSES]
    t_first, t_imp, src = _pick_impact_anchors(clusters, iv, end_type, a.t_detect)
    a.t_first_hit, a.t_impact_peak = t_first, t_imp
    a.t_rest = None
    if abnormal and a.t_detect is not None:
        a.t_rest = _r(_t_rest(sh, a.t_detect))
        if a.t_rest is None and replay is not None and replay.t_onset is not None:
            a.t_rest = _r(replay.t_onset)
    a.extra["end_state"] = {"why": why, "impact_source": src, "class": cls,
                            "detect_lag_s": _r(a.t_detect - a.t_impact_peak) if (a.t_detect is not None and a.t_impact_peak is not None) else None,
                            "impacts_all": [{k: c.get(k) for k in ("t_start", "t_peak", "peak", "dV", "cls", "interval")} for c in sh.clusters()
                                            if c["cls"] in IMPACT_CLASSES],
                            "sustained_accel": [{k: c.get(k) for k in ("t_start", "t_end", "peak", "sustained_s", "interval")} for c in sh.clusters()
                                                if c["cls"] == "sustained_accel"]}
    a.notes.append(f"結束型態 {end_type}:{why}" + (f";撞擊 {a.t_impact_peak:.2f} s,飛控晚 {a.t_detect - a.t_impact_peak:.1f} s 判定"
                                                  if (a.t_detect is not None and a.t_impact_peak is not None and abnormal) else ""))
    return end_type


# ================================================================ 入口

def run(ctx: Ctx, a: Anchors) -> list[Finding]:
    """依序執行本領域的偵測器,最後把結束型態定案並填錨點。每個偵測器 id 至少回傳一筆 Finding;不丟例外。"""
    if a is None:           # 沒有錨點 = 沒有解鎖區間 → 每個偵測器都回 NA(not_armed)
        a = Anchors()
    if a.interval is None:
        a.end_type = "not_armed"
        return [f for fn in _DETECTOR_ORDER for f in fn(ctx, a, None)]    # 每個偵測器都回 NA(not_armed)
    sh = _Sh(ctx, a)
    findings: list[Finding] = []
    for fn in _DETECTOR_ORDER:
        findings += fn(ctx, a, sh)
    try:
        classify_end(ctx, a, findings, sh)
    except Exception as e:  # noqa: BLE001 — 分類失敗不拖垮報告
        a.notes.append(f"結束型態分類失敗:{type(e).__name__}: {e}")
    for f in findings:      # 時間統一到 ms(避免 1258.3622229999999 這種浮點尾巴)
        f.t_onset, f.t_alarm = _r(f.t_onset), _r(f.t_alarm)
    return findings
