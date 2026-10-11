"""馬達 / 推力偵測模組(domain = 'motor'):推力損失檢查、推力失衡、偏航 / 姿態積分項飽和。

三個偵測器都是「排除用」的硬體檢查 —— 00000124 的馬達端完全健康,墜機起點是降落觸地後的地面翻覆;
這裡的工作是用數字證明「不是馬達 / 電調 / 槳」(CLEAR),並在真的有推力問題時能正確 HIT。

重要限制(每則 CLEAR 的 text_zh / values['esc_telemetry'] 都會註明):
  * 這批 log 沒有 ESC / 轉速遙測(ESC.RPM 不存在),RCOU 只是「指令輸出」:控制器照常下指令、
    但實際已失效的馬達 / 槳無法從資料看出來,只能由姿態 / 油門的後果間接檢查。
  * 門檻是以「1 次失事 + 同機兩段健康飛行」的間距定的(n = 1)。
  * 韌體規則已對照 ArduPilot master 原始碼(ArduCopter/crash_check.cpp、Copter.h):
      thrust_loss_check():angle_target.xy() 長度 ≤ 15°、get_throttle_in() ≥ 0.9(或 limit.throttle_upper)且 ≥ 0.25、
        正在下降、姿態誤差 < 30°(「確保仍有姿態控制」)、~land_complete,連續 1 s → ERR 25 + 'Potential Thrust Loss' + thrust boost;
      yaw_imbalance_check():0.05 Hz 低通的 |yaw I| > 75% IMAX(或 I == IMAX)就立即警告;YAW_IMBALANCE_WARN_MS(10 s)
        只是訊息的最短間隔,不是持續時間;落地 / 未解鎖 / thrust boost 時濾波器歸零,濾波值不得大於 |I|。
    規格(catalogue.json)把 (a) 寫成「姿態誤差 > 30°」、油門用 MOTB.ThrOut、傾角用 max(|DesRoll|,|DesPitch|),
    並把 10 s 當成持續時間 —— 都和原始碼不符,本模組照原始碼實作(偏差見各偵測器說明)。
  * 6 個桌上 / 手持 baseline log 從未解鎖(RCOU 全 1100),不構成任何誤報驗證 → 一律 NA(不適用),不是「正常」。

情境(Finding.context):
  'air'             空中(CTUN.ThO 的 1 秒中位數 > 0.03):計入 H_MOTOR_THRUST;
  'ground_idle'     地面怠速(ThO ≤ 0.03 且貼地:離地 ≤ 1.5 m 或已過觸地錨點)的推力失衡:是翻覆的「症狀」,
                    只計入 H_LAND_TIPOVER,絕不拿來滿足空中偵測器;
  'idle_air'        怠速但無法確認貼地(例:空中收油門)的推力失衡:中性描述,不計入任何假說;
  'loss_of_control' 姿態失控(姿態誤差 > 30°)之後才出現的馬達滿載:翻覆 / 失控的後果,降為 SOFT,不證明推力損失;
  'ground'          尚未判定離地(例:起飛瞬間翻覆)時成立的推力損失。

信心(依資料能分離的程度):
  motor_thrust_loss_check  高(124 與健康飛行相差數倍,見常數註解)
  motor_thrust_imbalance   中(健康懸停的重算比值 p99 1.30-1.38、最大 1.447,離 1.5 門檻只有 ~3.5%;
                           靠「持續 2 s / 8 s + 已落地閘門」才穩)
  yaw_imbalance_iterm      高,但兩段健康飛行與墜機之間沒有分離(都遠低於門檻),純排除用
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ctx import Ctx
from .guard import guarded
from .types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval, Finding
from .util import thrust_vector_error_deg

DOMAIN = "motor"

# ======================================================================================
# 門檻常數(括號內:124 墜機 vs 124 兩段健康飛行的實測間距;n = 1,見模組說明)
# ======================================================================================
SPIN_PWM_MARGIN = 5.0          # spin = 任一馬達 PWM > MOT_PWM_MIN + 5(未解鎖輸出恰為 1100)
MIN_AIR_S = 5.0                # 判 CLEAR 至少要有的「轉動且未落地」秒數;不足 → NA(沒飛過就不能說沒推力問題)
ACRO_MODES = (1, 13, 14)       # ACRO / SPORT / FLIP:姿態誤差在刻意動作中沒有意義
EV_FLIP_START, EV_FLIP_END = 21, 22
FLIP_WINDOW_FALLBACK_S = 10.0  # 只有 FLIP_START 沒有 FLIP_END 時,往後排除的秒數
PWM_RANGE_MIN = 100.0          # MOT_PWM_MAX − MOT_PWM_MIN 至少 100 µs,否則參數無效 → NA 'bad_params'(本機 900 µs)

# ---- motor_thrust_loss_check ----
# (a) ArduCopter thrust_loss_check()(已對照原始碼;規格把姿態誤差條件寫反,見模組說明)
TL_DES_TILT_MAX = 15.0         # (a) hypot(DesRoll, DesPitch) ≤ 15°(韌體 angle_target.xy().length² ≤ radians(15)²)
TL_THR_MIN = 0.9               # (a) 油門需求 get_throttle_in()(= CTUN.ThI)≥ 0.9,或已達上限(124:受控下降中最高 0.25、全程 0.69)
TL_THROUT_MIN = TL_THR_MIN     # 舊名(相容)
TL_THR_FLOOR = 0.25            # (a) 且油門需求 ≥ 0.25(韌體:低油門時的推力限制不算)
TL_UPPER_ACT = 0.99            # (a)「已達上限」limit.throttle_upper 沒有記錄:以「任一馬達 act ≥ 0.99(油門已無法再加)」近似
TL_AE_DEG = 30.0               # (a) 姿態誤差 < 30°(仍有姿態控制);(b) 姿態誤差 > 30°(推力向量夾角)
TL_HOLD_S = 1.0                # 持續 ≥ 1.0 s(韌體計數器 1 s;墜機:(a) 0 個樣本;(b) 最長 0.1 s)
TL_SOFT_HOLD_S = 0.5           # 持續 ≥ 0.5 s → SOFT(接近門檻;無樣本)
# (b) 本模組補充規則(韌體沒有):任一馬達 act ≥ 0.95 且姿態誤差 > 30°,連續 ≥ 1.0 s。
#     只有「馬達先滿載、姿態才失控」才是推力損失的證據;失控之後才滿載是翻覆 / 失控的後果(降為 SOFT 'loss_of_control')。
TL_SAT_ACT = 0.95              # (b) 任一馬達 act ≥ 0.95(本機 PWM ≥ 1919);健康懸停 act ≤ 0.55(PWM 1566)
FF_BIT_BOOST = 1               # MOTB.FailFlags bit0 = thrust boost(韌體已判推力損失;124 全 7 個 log 從未設)
ERR_SUBSYS_THRUST_LOSS = 25    # ERR Subsys 25 = THRUST_LOSS_CHECK
TL_MSG_RX = r"potential thrust loss"

# ---- motor_thrust_imbalance(MOTB.FailFlags bit1 = thrust_balanced)----
IMB_MIN_MOTORS = 6             # 韌體的推力平衡閂鎖只有 N ≥ 6 才存在
FF_BIT_BALANCED = 2            # FailFlags & 2 == 0 → 推力失衡(124 所有健康 / 未解鎖取樣都是 2)
IMB_RATIO_LATCH = 1.5          # 重算:最大濾波推力 / 平均 ≥ 1.5 閂住
IMB_RATIO_RELEASE = 1.25       # ≤ 1.25 釋放(重現 12783 筆 FailFlags 的 99.96%)
IMB_RATIO_SEVERE = 1.7         # 嚴重(報告用,不影響判定)
IMB_LOOP_HZ = 400.0            # 韌體主迴圈
IMB_TAU_S = 0.5                # _thrust_rpyt_out 濾波時間常數
IMB_SUM_MIN = 0.1              # 濾波推力總和 ≤ 0.1 時比值視為 1.0
IMB_CUM_S = 2.0                # 8 s 窗內累計不平衡 ≥ 2.0 s 才算(健康飛行:空中 0 s;著陸瞬間最長 ~0.9 s 已被閘門濾掉)
IMB_WIN_S = 8.0
IMB_SOFT_CUM_S = 1.0           # 累計 ≥ 1.0 s → SOFT(無樣本)
IDLE_THO = 0.03                # CTUN.ThO ≤ 0.03 = 地面怠速(上緣與韌體 / 地面模組一致)
IDLE_MED_S = 1.0               # ThO 取 1 s 置中中位數,避免翻覆時零星的推力突波把症狀誤分成「空中」
IMB_NEAR_GROUND_M = 1.5        # 'ground_idle' 需貼地:離地 ≤ 1.5 m(和 hypotheses.NEAR_GROUND_AGL_M 一致;124 翻覆全程 −1.8~+0.2 m)
FLAGS_VALID_FRAC = 0.5         # MOTB.FailFlags 至少一半取樣有 bit1 才信它(否則改用重算)

# ---- yaw_imbalance_iterm(ArduCopter yaw_imbalance_check() + 本模組的持續時間規則)----
YI_RATIO = 0.75                # 低通後 |I| / IMAX > 0.75(韌體 YAW_IMBALANCE_IMAX_THRESHOLD)
YI_IMAX_EQ = 0.999             # |I| ≥ 0.999 IMAX 視為「I == IMAX」(韌體 is_equal(I_term, I_max) 分支;取絕對值)
YI_HOLD_S = 10.0               # 本模組:連續 ≥ 10 s 才 HIT(韌體沒有持續時間,10 s 是訊息間隔;124 最大 0.07,遠低於門檻)
YI_SOFT_HOLD_S = 5.0           # 滾轉 / 俯仰(韌體沒有這項檢查):連續 ≥ 5 s → SOFT(無樣本)
YI_FW_SOFT_S = 0.0             # 偏航:韌體條件成立任何時間(韌體此時已會發 'Yaw Imbalance',沒有持續時間)→ 至少 SOFT,不能 CLEAR
YI_LPF_HZ = 0.05               # 單極低通截止頻率(Copter.h:LowPassFilterFloat yaw_I_filt{0.05f})
YI_IMAX_DEFAULT = 0.5          # 參數缺值時的 IMAX(ATC_RAT_*_IMAX 預設)
YI_AXES = (("PIDY", "ATC_RAT_YAW_IMAX", "偏航"), ("PIDR", "ATC_RAT_RLL_IMAX", "滾轉"),
           ("PIDP", "ATC_RAT_PIT_IMAX", "俯仰"))
YI_MSG_RX = r"yaw imbalance"

_ESC_KEYS = ("ESC.RPM", "RPM.RPM", "ESC.RawRPM")
_EPS = 1e-6


# ======================================================================================
# 小工具
# ======================================================================================

def _na(det_id: str, context: str, text: str, **values) -> Finding:
    return Finding(det_id, DOMAIN, NA, "info", None, None, context, values, text)


def _na_reason(a: Anchors, det_id: str) -> Finding | None:
    """a(或 a.interval)缺 → NA 'not_armed'(桌上 / 手持 log:馬達從未運轉)。"""
    if a is None or getattr(a, "interval", None) is None:
        return _na(det_id, "not_armed", "未解鎖(桌上 / 手持)log:馬達從未運轉,馬達分析不適用,不做失事分析。")
    return None


def _has_esc(ctx: Ctx) -> bool:
    try:
        return any(ctx.has(k) for k in _ESC_KEYS)
    except Exception:  # noqa: BLE001
        return False


def _runs_v(mask: np.ndarray, t: np.ndarray, max_gap: float | None = None):
    """向量化的連續為真區段 [(i0, i1)](索引含);相鄰樣本時間差 > max_gap 視為斷開(資料空洞)。"""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    brk = np.diff(idx) > 1
    if max_gap is not None:
        brk |= np.diff(t[idx]) > max_gap
    starts = np.concatenate(([0], np.flatnonzero(brk) + 1))
    ends = np.concatenate((np.flatnonzero(brk), [idx.size - 1]))
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def _longest_run(mask: np.ndarray, t: np.ndarray, dt: float):
    """最長連續為真區段 → (秒數, 起點時間, 起點索引);秒數 = 樣本數 × 取樣間隔(1 個樣本 = 1 個 dt)。"""
    best = (0.0, None, None)
    for i0, i1 in _runs_v(mask, t, max_gap=2.5 * dt):
        dur = (t[i1] - t[i0]) + dt
        if dur > best[0]:
            best = (float(dur), float(t[i0]), i0)
    return best


def _run_start(mask: np.ndarray, i: int, t: np.ndarray, max_gap: float) -> int:
    """包含索引 i 的連續為真區段的起點索引(往回走;資料空洞 > max_gap 斷開)。mask[i] 必須為真。"""
    j = i
    while j > 0 and mask[j - 1] and t[j] - t[j - 1] <= max_gap:
        j -= 1
    return j


def _pnum(ctx: Ctx, name: str, default: float) -> float | None:
    """參數 → float。缺值 → default;有值但無效(None / NaN / 非數字)→ None(呼叫端回 NA 'bad_params')。"""
    v = ctx.param(name, default)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _lpf_yaw_fw(x: np.ndarray, gate: np.ndarray, alpha: float) -> np.ndarray:
    """韌體 yaw_I_filt 的重現:gate 為假(落地 / 未轉動 / thrust boost)時濾波器歸零;
    y += alpha (x − y);若 |y| > |x| 則 y = x(原始碼「never allow to be larger than I」)。NaN 樣本保持前值。

    有狀態重設,無法用封閉式向量化;只對 gate 為真的各段做 Python 迴圈(124 飛行 2 每軸 ~15 萬點,約 0.03 s)。
    """
    out = np.zeros(x.size)
    idx = np.flatnonzero(gate)
    if idx.size == 0:
        return out
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1          # gate 斷開處(中間有未 gate 的樣本)→ 每段從 0 重新開始
    for seg in np.split(idx, cuts):
        prev = 0.0
        ys = []
        app = ys.append
        for v in x[seg].tolist():
            if v == v:                                    # 非 NaN
                prev += alpha * (v - prev)
                if (prev if prev >= 0.0 else -prev) > (v if v >= 0.0 else -v):
                    prev = v
            app(prev)
        out[seg] = ys
    return out


def _cum_window(mask: np.ndarray, t: np.ndarray, dt: float, win_s: float) -> np.ndarray:
    """每個樣本 k 的窗 (t[k]-win_s, t[k]] 內 mask 為真的累計秒數(每個真樣本算 dt;資料空洞不灌水)。"""
    c = np.concatenate(([0.0], np.cumsum(mask.astype(float))))
    j0 = np.searchsorted(t, t - win_s, side="right")
    return (c[np.arange(t.size) + 1] - c[j0]) * dt


def _lpf1(x: np.ndarray, a: float, y0=0.0) -> np.ndarray:
    """單極低通 y[k] = y[k-1] + a (x[k] - y[k-1]),沿第 0 軸;分塊向量化(純 numpy,沒有 scipy)。

    塊內用 y[k] = q^k [ q y_prev + a Σ_{j≤k} x[j] q^-j ](q = 1-a)的封閉式;塊長限制到 q^-B ≤ 1e8,不會溢位。
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    q = 1.0 - a
    if n == 0:
        return x.copy()
    if q <= 0.0:
        return x.copy()
    blk = max(8, int(np.log(1e8) / -np.log(q)))
    y = np.empty_like(x)
    prev = np.broadcast_to(np.asarray(y0, dtype=float), x.shape[1:]).copy()
    for s in range(0, n, blk):
        xb = x[s:s + blk]
        k = np.arange(xb.shape[0], dtype=float).reshape((-1,) + (1,) * (x.ndim - 1))
        qk = q ** k
        y[s:s + blk] = qk * (q * prev + a * np.cumsum(xb / qk, axis=0))
        prev = y[s + xb.shape[0] - 1]
    return y


def _interp_finite(t: np.ndarray, t_src: np.ndarray, v: np.ndarray) -> np.ndarray | None:
    """np.interp,但先丟掉非有限值的來源樣本(NaN 不會擴散成 NaN 統計值 / JSON);全部無效 → None。"""
    ok = np.isfinite(v) & np.isfinite(t_src)
    if not ok.any():
        return None
    return np.interp(t, t_src[ok], v[ok])


def _nearest(t_src: np.ndarray, v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """最近鄰取值(離散旗標用;兩表時間戳差幾 ms 時不會錯位一格)。非有限值的來源樣本略過。"""
    ok = np.isfinite(v) & np.isfinite(t_src)
    t_src, v = t_src[ok], v[ok]
    if t_src.size == 0:
        return np.full(t.size, np.nan)
    if t_src.size == 1:
        return np.full(t.size, v[0])
    i = np.clip(np.searchsorted(t_src, t), 1, t_src.size - 1)
    left, right = t_src[i - 1], t_src[i]
    return np.where(np.abs(t - left) <= np.abs(right - t), v[i - 1], v[i])


def _rolling_median(x: np.ndarray, k: int) -> np.ndarray:
    """置中滾動中位數(邊緣補端點值);k 取奇數。"""
    if k <= 1 or x.size == 0:
        return x.copy()
    k |= 1
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=1)


def _motor_channels(ctx: Ctx) -> list[int]:
    """馬達輸出通道(RCOU Cn 的 n):SERVOn_FUNCTION 33..40,依馬達編號排序。沒有 SERVO 參數時回 []。"""
    pairs = []
    for k in range(1, 17):
        fn = ctx.param(f"SERVO{k}_FUNCTION")
        if fn is not None and np.isfinite(fn) and 33 <= int(round(fn)) <= 40:
            pairs.append((int(round(fn)), k))
    return [k for _, k in sorted(pairs)]


def _att_error(ctx: Ctx, t0: float, t1: float):
    """推力向量角度誤差 ae(度)(和韌體 AngErr 同定義,不含 yaw 的完整旋轉角)→ (t, ae) 或 None。

    優先 ANG(~400 Hz);沒有則 ATT(10 Hz)。只取 [t0-0.5, t1+0.5]。
    """
    for name in ("ANG", "ATT"):
        tb = ctx.table(name)
        if tb is None or not all(c in tb for c in ("Roll", "Pitch", "Yaw", "DesRoll", "DesPitch", "DesYaw")):
            continue
        t = tb["t"]
        i0, i1 = np.searchsorted(t, [t0 - 0.5, t1 + 0.5])
        if i1 - i0 < 3:
            continue
        sl = slice(int(i0), int(i1))
        ae = thrust_vector_error_deg(tb["Roll"][sl], tb["Pitch"][sl], tb["Yaw"][sl],
                                     tb["DesRoll"][sl], tb["DesPitch"][sl], tb["DesYaw"][sl])
        if not np.isfinite(ae).any():          # 整段無效(全 NaN)→ 換下一個來源;都沒有 → None(NA,不是 CLEAR)
            continue
        return t[sl], ae
    return None


def _excluded_mask(ctx: Ctx, t: np.ndarray) -> np.ndarray:
    """刻意動作(ACRO / SPORT / FLIP 模式、EV21..22 翻滾窗)的取樣:姿態誤差在這些時段沒有意義。"""
    ex = np.zeros(t.size, dtype=bool)
    tm, nums, _ = ctx.mode_rows()
    if tm.size:
        i = np.clip(np.searchsorted(tm, t, side="right") - 1, 0, tm.size - 1)
        ex |= np.isin(nums[i], ACRO_MODES) & (t >= tm[0])
    te, ids = ctx.ev_rows()
    for ts in te[ids == EV_FLIP_START]:
        ends = te[(ids == EV_FLIP_END) & (te >= ts)]
        ex |= (t >= ts) & (t <= (ends[0] if ends.size else ts + FLIP_WINDOW_FALLBACK_S))
    return ex


def _landed_on(ctx: Ctx, iv: ArmedInterval, t: np.ndarray) -> np.ndarray:
    """區間內每個取樣是否「已落地」(EV28 離地 / EV18 落地的階梯;EV17 不算,上鎖當下寫出的 EV17/18 已被 ctx 濾掉)。"""
    steps = ctx.landed_steps(iv)
    ts = np.array([s[0] for s in steps], dtype=float)
    vs = np.array([s[1] for s in steps], dtype=bool)
    i = np.searchsorted(ts, t, side="right") - 1
    return np.where(i >= 0, vs[np.clip(i, 0, vs.size - 1)], True)


def _thrust_ratio(act: np.ndarray, batv: np.ndarray, expo: float, dt: float):
    """由指令輸出重算韌體的推力平衡比值。act:(n, N) 0..1;batv:(n,) 正規化電壓。

    thr_i = e r² + (1-e) r(r = act × batv),對時間做 tau 0.5 s 的單極濾波(每 dt 的等效 alpha),
    ratio = max_i f_i × N / Σ f_i(Σ ≤ 0.1 → 1.0)。回傳 (ratio (n,), f (n, N))。
    """
    n_mot = act.shape[1]
    r = act * batv[:, None]
    thr = expo * r * r + (1.0 - expo) * r
    beta = 1.0 / (1.0 + IMB_TAU_S * IMB_LOOP_HZ)
    alpha = 1.0 - (1.0 - beta) ** (dt * IMB_LOOP_HZ)
    f = _lpf1(thr, alpha, y0=thr[0] if thr.shape[0] else 0.0)
    s = f.sum(axis=1)
    ratio = np.where(s > IMB_SUM_MIN, f.max(axis=1) * n_mot / np.where(s > 0, s, 1.0), 1.0)
    return ratio, f


def _latch(ratio: np.ndarray, hi: float = IMB_RATIO_LATCH, lo: float = IMB_RATIO_RELEASE) -> np.ndarray:
    """遲滯閂鎖(ratio ≥ hi 設、≤ lo 解),向量化。"""
    idx = np.arange(ratio.size)
    last_hi = np.maximum.accumulate(np.where(ratio >= hi, idx, -1))
    last_lo = np.maximum.accumulate(np.where(ratio <= lo, idx, -1))
    return last_hi > last_lo


# ======================================================================================
# 共用前處理(三個偵測器共用,依 (t0, t1) 快取在 ctx 上)
# ======================================================================================

@dataclass
class _Mot:
    iv: ArmedInterval
    t: np.ndarray                       # RCOU 時間(相對秒),截在解鎖區間內
    dt: float                           # 名目取樣間隔(RCOU 中位數,~0.1 s)
    ch: list[int]                       # 馬達 → RCOU 通道號
    pwm: np.ndarray                     # (n, N)
    pmin: float
    pmax: float
    spin_min: float
    spin_max: float
    expo: float
    spin: np.ndarray                    # bool(n)
    act: np.ndarray                     # (n, N) 0..1
    landed: np.ndarray                  # bool(n)
    batv: np.ndarray                    # MOTB.BatVolt(正規化;沒有 MOTB 就 1.0)
    throut: np.ndarray                  # MOTB.ThrOut(沒有 MOTB 就用各馬達推力平均)
    has_motb: bool
    flags: np.ndarray | None            # MOTB.FailFlags(整數值的 float)
    flags_valid: bool
    tho: np.ndarray | None              # CTUN.ThO
    idle: np.ndarray | None             # ThO 1 s 中位數 ≤ 0.03
    crt: np.ndarray | None              # CTUN.CRt(cm/s)
    des_tilt: np.ndarray | None         # hypot(DesRoll, DesPitch)(ATT;韌體 angle_target.xy() 的長度)
    ae: np.ndarray | None               # 推力向量角度誤差(度),在 RCOU 時間軸上
    ae_hf: tuple | None                 # (t, ae) 高頻原始序列(找 t_warn 用)
    excl: np.ndarray                    # 刻意動作時段
    n_motors: int
    thr_in: np.ndarray | None = None    # 韌體 get_throttle_in() = CTUN.ThI(沒有就用 throut)
    thr_in_src: str = ""
    agl: np.ndarray | None = None       # CTUN.Alt − 解鎖後 3 s 的地面高度(沒有測距儀;只在撞擊前可信)
    notes: list = field(default_factory=list)

    @property
    def spin_s(self) -> float:
        return float(self.spin.sum() * self.dt)

    @property
    def air(self) -> np.ndarray:
        return self.spin & ~self.landed

    @property
    def air_s(self) -> float:
        return float(self.air.sum() * self.dt)

    def pwm_cmd_max(self) -> float:
        return self.pmin + self.spin_max * (self.pmax - self.pmin)

    def pwm_sat(self) -> float:
        return self.pmin + (self.spin_min + TL_SAT_ACT * (self.spin_max - self.spin_min)) * (self.pmax - self.pmin)

    def pwm_idle(self) -> float:
        return self.pmin + self.spin_min * (self.pmax - self.pmin)


def _setup(ctx: Ctx, a: Anchors):
    """→ (_Mot, None) 或 (None, (context, text_zh))。"""
    iv = a.interval
    if iv is None:
        return None, ("not_armed", "未解鎖(桌上 / 手持)log:馬達從未運轉,馬達分析不適用,不做失事分析。")
    key = (round(iv.t0, 6), round(iv.t1, 6))
    cache = getattr(ctx, "_motor_prep", None)
    if cache is None:
        cache = {}
        try:
            ctx._motor_prep = cache
        except Exception:  # noqa: BLE001
            pass
    if key not in cache:
        cache[key] = _build(ctx, iv)
    return cache[key]


def _build(ctx: Ctx, iv: ArmedInterval):
    rc = ctx.table("RCOU")
    if rc is None:
        return None, ("no_rcou", "log 沒有 RCOU(馬達輸出)紀錄,無法做馬達檢查。")
    chans = _motor_channels(ctx)
    notes: list[str] = []
    pmin_mot = _pnum(ctx, "MOT_PWM_MIN", 0.0)           # 0 = 韌體預設「用 SERVOn_MIN/MAX」
    if not chans:
        # 沒有 SERVOn_FUNCTION:退而求其次 —— 區間內曾經轉動過的通道(不理想,註記)
        sel0 = (rc["t"] >= iv.t0) & (rc["t"] <= iv.t1)
        pmin0 = pmin_mot if pmin_mot else 1100.0
        chans = [k for k in range(1, 15) if f"C{k}" in rc and sel0.any() and np.nanmax(rc[f"C{k}"][sel0]) > pmin0 + SPIN_PWM_MARGIN]
        notes.append("沒有 SERVOn_FUNCTION 參數:馬達通道由 RCOU 曾轉動的通道推定")
    chans = [k for k in chans if f"C{k}" in rc]
    if len(chans) < 3:
        return None, ("no_motors", "找不到馬達輸出通道(SERVOn_FUNCTION 33..40)。")
    t_all = rc["t"]
    sl = slice(int(np.searchsorted(t_all, iv.t0, side="left")), int(np.searchsorted(t_all, iv.t1, side="right")))
    t = t_all[sl]
    dts = np.diff(t)
    dts = dts[np.isfinite(dts) & (dts > 0)]             # 重複時間戳(差 0)不算進取樣間隔
    if t.size < 5 or dts.size == 0:
        return None, ("insufficient_data", "解鎖區間內的 RCOU 取樣不足(或時間戳全部重複)。")
    dt = float(np.median(dts))
    pwm = np.stack([rc[f"C{k}"][sl] for k in chans], axis=1)
    pmax_mot = _pnum(ctx, "MOT_PWM_MAX", 0.0)
    # 0 = 韌體預設「改用 SERVOn_MIN/MAX」;None = 參數存在但無效 → 下面回 NA
    pmin = pmin_mot if pmin_mot != 0.0 else _pnum(ctx, f"SERVO{chans[0]}_MIN", 1100.0)
    pmax = pmax_mot if pmax_mot != 0.0 else _pnum(ctx, f"SERVO{chans[0]}_MAX", 1900.0)
    spin_min = _pnum(ctx, "MOT_SPIN_MIN", 0.15)
    spin_max = _pnum(ctx, "MOT_SPIN_MAX", 0.95)
    expo = _pnum(ctx, "MOT_THST_EXPO", 0.65)
    bad = []
    if pmin is None or pmax is None or pmax - pmin < PWM_RANGE_MIN:
        bad.append(f"MOT_PWM_MIN/MAX = {pmin}/{pmax}")
    if spin_min is None or spin_max is None or not (0.0 <= spin_min < spin_max <= 1.0):
        bad.append(f"MOT_SPIN_MIN/MAX = {spin_min}/{spin_max}")
    if expo is None or not (-1.0 <= expo <= 1.0):
        bad.append(f"MOT_THST_EXPO = {expo}")
    if bad:
        return None, ("bad_params", "馬達參數無效(" + "、".join(bad) + "),無法把 PWM 換算成輸出比例,馬達檢查不適用。")
    spin = pwm.max(axis=1) > pmin + SPIN_PWM_MARGIN
    act = np.clip(((pwm - pmin) / (pmax - pmin) - spin_min) / max(spin_max - spin_min, 1e-6), 0.0, 1.0)
    if not spin.any():
        return None, ("no_spin", "本次解鎖區間內馬達從未轉動(RCOU 全為最低輸出):馬達檢查不適用。")
    landed = _landed_on(ctx, iv, t)

    # 10 Hz 輔助序列一律依時間對到 RCOU 時間軸(列數不同,不能依索引);NaN 來源樣本略過(_interp_finite)
    mb = ctx.table("MOTB")
    thr_m = _interp_finite(t, mb["t"], mb["ThrOut"]) if mb is not None and "ThrOut" in mb else None
    has_motb = thr_m is not None
    batv = np.ones(t.size)
    flags = None
    flags_valid = False
    if has_motb:
        throut = thr_m
        if "BatVolt" in mb:
            bv = _interp_finite(t, mb["t"], mb["BatVolt"])
            if bv is not None:
                batv = bv
        if "FailFlags" in mb:
            ff = mb["FailFlags"]
            fin = ff[np.isfinite(ff)]
            if fin.size:
                flags = np.rint(_nearest(mb["t"], ff, t))
                flags_valid = bool(np.mean((np.rint(fin).astype(np.int64) & FF_BIT_BALANCED) != 0) >= FLAGS_VALID_FRAC)
    else:
        r = act * batv[:, None]
        throut = (expo * r * r + (1.0 - expo) * r).mean(axis=1)
        notes.append("沒有 MOTB:ThrOut 以各馬達推力平均代替,FailFlags 不可用")

    ct = ctx.table("CTUN")
    tho = crt = idle = thr_in = agl = None
    if ct is not None:
        if "ThO" in ct:
            tho = _interp_finite(t, ct["t"], ct["ThO"])
            if tho is not None:
                idle = _rolling_median(tho, int(round(IDLE_MED_S / dt)) | 1) <= IDLE_THO
        if "CRt" in ct:
            crt = _interp_finite(t, ct["t"], ct["CRt"])
        if "ThI" in ct:
            thr_in = _interp_finite(t, ct["t"], ct["ThI"])
        if "Alt" in ct:
            alt = _interp_finite(t, ct["t"], ct["Alt"])
            try:
                g = ctx.ground_alt(iv)
            except Exception:  # noqa: BLE001 — 沒有地面高度就不判貼地
                g = None
            if alt is not None and g is not None and np.isfinite(g):
                agl = alt - float(g)
    thr_in_src = "CTUN.ThI"
    if thr_in is None:
        thr_in = throut
        thr_in_src = "MOTB.ThrOut" if has_motb else "mean_thrust"
        notes.append(f"沒有 CTUN.ThI:推力損失 (a) 的油門需求改用 {thr_in_src}")
    at = ctx.table("ATT")
    des_tilt = None
    if at is not None and "DesRoll" in at and "DesPitch" in at:
        dr = _interp_finite(t, at["t"], at["DesRoll"])
        dp = _interp_finite(t, at["t"], at["DesPitch"])
        if dr is not None and dp is not None:
            des_tilt = np.hypot(dr, dp)
    ae_hf = _att_error(ctx, iv.t0, iv.t1)
    ae = np.interp(t, ae_hf[0], ae_hf[1]) if ae_hf is not None else None
    m = _Mot(iv, t, dt, chans, pwm, pmin, pmax, spin_min, spin_max, expo, spin, act, landed, batv, throut,
             has_motb, flags, flags_valid, tho, idle, crt, des_tilt, ae, ae_hf, _excluded_mask(ctx, t),
             len(chans), thr_in, thr_in_src, agl, notes)
    return m, None


def _local_t_warn(m: _Mot) -> float | None:
    """本地備援:姿態誤差 > 10° 持續 ≥ 0.3 s 的起點(排除刻意動作);a.t_warn 缺時才用。"""
    if m.ae_hf is None:
        return None
    t, ae = m.ae_hf
    ex = np.interp(t, m.t, m.excl.astype(float)) > 0.5
    sel = (t >= m.iv.t0 + 3.0) & (t <= m.iv.t1)
    ok = (ae > 10.0) & ~ex & sel
    for i0, i1 in _runs_v(ok, t, max_gap=0.25):
        if t[i1] - t[i0] >= 0.3:
            return float(t[i0])
    return None


def _pre_event_end(m: _Mot, a: Anchors) -> tuple[float, str]:
    """「事件前」窗的結束時間:最早的 a.t_touch / a.t_warn;沒有就用本地 ae > 10°;都沒有 → 區間終點。"""
    iv = m.iv
    c = [x for x in (a.t_touch, a.t_warn) if x is not None and iv.t0 < x <= iv.t1]
    if c:
        return float(min(c)), "anchor"
    lw = _local_t_warn(m)
    if lw is not None:
        return lw, "local_t_warn"
    return float(iv.t1), "interval_end"


# ======================================================================================
# 1. motor_thrust_loss_check
# ======================================================================================

def _b_runs(m: _Mot, a: Anchors, act_max: np.ndarray) -> tuple[list[dict], int]:
    """(b) 滿載 ∧ ae > 30° 的每個區段(依時間)→ ([{dur, t, sat_on, ae_on, after}], 取樣數)。

    sat_on = 包含此段起點的「滿載」連續區段起點;ae_on = 包含此段起點的「ae > 30°」連續區段起點。
    after(失控後才滿載)= 滿載起點不早於 ae > 30° 起點,或不早於 a.t_loss(10 Hz 同一格算「同時」→ 視為失控後)。
    """
    gap = 2.5 * m.dt
    sat = m.spin & (act_max >= TL_SAT_ACT) & ~m.excl
    ae30 = (m.ae > TL_AE_DEG) & ~m.excl
    mb_ = sat & ae30
    out = []
    for i0, i1 in _runs_v(mb_, m.t, max_gap=gap):
        s0 = float(m.t[_run_start(sat, i0, m.t, gap)])
        e0 = float(m.t[_run_start(ae30, i0, m.t, gap)])
        before = s0 < e0 - _EPS and (a.t_loss is None or s0 < a.t_loss - _EPS)
        out.append({"dur": float(m.t[i1] - m.t[i0] + m.dt), "t": float(m.t[i0]), "sat_on": s0, "ae_on": e0,
                    "after": not before})
    return out, int(mb_.sum())


@guarded("motor_thrust_loss_check", DOMAIN)
def motor_thrust_loss_check(ctx: Ctx, a: Anchors):
    """推力損失檢查,三項判準取聯集:
      (a) FIRMWARE   ArduCopter thrust_loss_check()(已對照原始碼):轉動 ∧ 未落地 ∧ hypot(DesRoll, DesPitch) ≤ 15°
                     ∧ (油門需求 CTUN.ThI ≥ 0.9 ∨ 已達上限[任一馬達 act ≥ 0.99]) ∧ ThI ≥ 0.25 ∧ CRt < 0 ∧ 姿態誤差 < 30°,
                     連續 ≥ 1.0 s。= 飛機想保持水平、姿態也還控得住、油門全開卻仍在掉 → 推力不夠。
                     規格偏差:catalogue.json 寫成 ae > 30°、MOTB.ThrOut、max(|DesRoll|,|DesPitch|),且沒有 0.25 下限 /
                     上限分支 / ~L —— 與原始碼不符;照規格做,教科書式的推力損失(姿態受控、全油門仍下沉)會得到 CLEAR。
      (b) SATURATION 本模組補充(韌體沒有):轉動 ∧ 任一馬達 act ≥ 0.95 ∧ 姿態誤差 > 30°,連續 ≥ 1.0 s。不看 ~L
                     (起飛時一顆馬達失效翻在地上正是要抓的)。只有「馬達先滿載、姿態才失控」(滿載起點早於該段 ae > 30°
                     的起點、也早於 a.t_loss)才是推力損失證據 → HIT,t_onset = 滿載起點。失控之後才滿載是翻覆 / 失控的
                     後果 → SOFT、context 'loss_of_control'、values['after_attitude_loss'] = True(不證明推力損失)。
      (c) FLAG       任一轉動取樣 FailFlags bit0(thrust boost)、訊息 'Potential Thrust Loss'、或 ERR Subsys 25。
                     韌體在條件持續 1 s 後才發出:t_alarm = 旗標 / 訊息時間,t_onset = t_alarm − 1 s(不早於解鎖)。
    SOFT(接近):(a) ≥ 0.5 s;失控前滿載的 (b) ≥ 0.5 s;受控下降中油門已滿(不論傾角需求)≥ 0.5 s;
               事件前已有馬達 ≥ 95% 滿載 ≥ 0.5 s。CLEAR 只在所有印出的餘裕都為正時才回。
    NA = 馬達從未轉 / 沒飛過 / 沒有姿態資料 / 參數無效 / 缺 CRt 或期望角(韌體條件 (a) 無法評估,不能下 CLEAR)。
    """
    det = "motor_thrust_loss_check"
    na = _na_reason(a, det)
    if na is not None:
        return na
    m, why = _setup(ctx, a)
    if m is None:
        return _na(det, *why)
    iv = m.iv
    esc = _has_esc(ctx)
    act_max = m.act.max(axis=1)

    # ---- (c) 旗標 / 訊息 / ERR(韌體在條件成立 1 s 後才發出)----
    c_alarm: list[tuple[float, str]] = []
    if m.flags is not None:
        bad = m.spin & ((m.flags.astype(np.int64) & FF_BIT_BOOST) != 0)
        if bad.any():
            c_alarm.append((float(m.t[np.argmax(bad)]), "FailFlags.bit0"))
    msgs = ctx.messages_between(iv.t0, iv.t1 + 1.0, TL_MSG_RX)
    if msgs:
        c_alarm.append((float(msgs[0][0]), "MSG"))
    te, sub, code = ctx.err_rows()
    em = (sub == ERR_SUBSYS_THRUST_LOSS) & (code != 0) & (te >= iv.t0) & (te <= iv.t1 + 0.5)
    if em.any():
        c_alarm.append((float(te[em][0]), "ERR25"))
    c_src = [s for _, s in c_alarm]

    # ---- (a) 韌體條件 ----
    ae_ok = m.ae is not None
    a_valid = ae_ok and m.des_tilt is not None and m.crt is not None
    a_dur = hd_dur = 0.0
    a_t = hd_t = None
    thr_desc_max = None
    if a_valid:
        ctrl_desc = m.air & ~m.excl & (m.crt < 0) & (m.ae < TL_AE_DEG)            # 受控下降(ae 為 NaN → 不算)
        full = ((m.thr_in >= TL_THR_MIN) | (act_max >= TL_UPPER_ACT)) & (m.thr_in >= TL_THR_FLOOR)
        a_dur, a_t, _ = _longest_run(ctrl_desc & full & (m.des_tilt <= TL_DES_TILT_MAX), m.t, m.dt)
        hd_dur, hd_t, _ = _longest_run(ctrl_desc & full, m.t, m.dt)                # 不論傾角需求
        if ctrl_desc.any():
            thr_desc_max = float(m.thr_in[ctrl_desc].max())

    # ---- (b) 滿載 + 失控 ----
    b_runs, n_sat = _b_runs(m, a, act_max) if ae_ok else ([], 0)
    b_best = max(b_runs, key=lambda r: r["dur"], default=None)
    b_dur = b_best["dur"] if b_best else 0.0
    b_pre = next((r for r in b_runs if not r["after"] and r["dur"] >= TL_HOLD_S - _EPS), None)
    b_post = max((r for r in b_runs if r["after"] and r["dur"] >= TL_HOLD_S - _EPS), key=lambda r: r["dur"], default=None)

    # ---- 餘裕數字(報告的「已排除」列)----
    pre_end, pre_src = _pre_event_end(m, a)
    pre = m.spin & (m.t <= pre_end)
    pre_pwm = float(m.pwm[pre].max()) if pre.any() else None
    pre_thr = float(m.throut[pre].max()) if pre.any() else None
    pre_sat_dur, pre_sat_t, _ = _longest_run(pre & (act_max >= TL_SAT_ACT) & ~m.excl, m.t, m.dt)   # 特技 / 翻滾中滿載是刻意的
    sp = m.spin
    thro_max = float(m.throut[sp].max())
    i_thr = int(np.argmax(np.where(sp, m.throut, -np.inf)))
    cmd_max = m.pwm_cmd_max()
    vals = {
        "n_motors": m.n_motors, "rcou_channels": m.ch, "spin_s": round(m.spin_s, 1), "air_s": round(m.air_s, 1),
        "max_throut": round(thro_max, 3), "max_throut_t": round(float(m.t[i_thr]), 2),
        "max_pwm": int(m.pwm[sp].max()),
        "pre_event_end": round(pre_end, 2), "pre_event_end_src": pre_src,
        "pre_event_max_pwm": None if pre_pwm is None else int(pre_pwm),
        "pre_event_max_throut": None if pre_thr is None else round(pre_thr, 3),
        "pwm_headroom": None if pre_pwm is None else int(round(cmd_max - pre_pwm)),
        "pre_event_sat_longest_s": round(pre_sat_dur, 2),
        "pwm_cmd_max": int(round(cmd_max)), "pwm_sat": int(round(m.pwm_sat())),
        "a_longest_s": round(a_dur, 2), "a_throttle_src": m.thr_in_src,
        "max_thr_in_ctrl_desc": None if thr_desc_max is None else round(thr_desc_max, 3),
        "full_thr_ctrl_desc_longest_s": round(hd_dur, 2),
        "b_longest_s": round(b_dur, 2), "b_sat_samples": n_sat,
        "b_runs": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in b_runs[:8]],
        "after_attitude_loss": None,
        "a_evaluated": bool(a_valid), "b_evaluated": bool(ae_ok),
        "flag_bit0_or_msg": c_src, "flag_alarm_t": round(min(t for t, _ in c_alarm), 3) if c_alarm else None,
        "has_motb": m.has_motb,
        "esc_telemetry": esc, "caveat": "no ESC telemetry: commanded-but-failed motor not visible" if not esc else "",
        "thresholds": {"thr_in": TL_THR_MIN, "thr_floor": TL_THR_FLOOR, "upper_act": TL_UPPER_ACT,
                       "des_tilt": TL_DES_TILT_MAX, "ae": TL_AE_DEG, "hold_s": TL_HOLD_S,
                       "sat_act": TL_SAT_ACT, "soft_hold_s": TL_SOFT_HOLD_S},
        "rule": "(a) ArduCopter crash_check.cpp thrust_loss_check (checked against master source); "
                "(b) module saturation rule; (c) firmware flag / ERR 25 / message",
    }
    if m.notes:
        vals["notes"] = list(m.notes)

    def margins() -> str:
        """各判準的餘裕(都是正的才會出現在 CLEAR;門檻只印在真正比較的量旁邊)。"""
        p = []
        if a_valid:
            p.append(f"韌體條件 (a)(期望傾角≤{TL_DES_TILT_MAX:.0f}°、姿態誤差<{TL_AE_DEG:.0f}°、油門需求≥{TL_THR_MIN:.1f} 或已達上限、"
                     f"仍在下降)最長 {a_dur:.1f} s(需 {TL_HOLD_S:.1f} s)")
            if thr_desc_max is not None and thr_desc_max < TL_THR_MIN:
                p.append(f"受控下降中最高油門需求 {thr_desc_max:.2f}(門檻 {TL_THR_MIN:.1f})")
            elif thr_desc_max is not None:
                p.append(f"受控下降中油門已滿最長 {hd_dur:.1f} s")
        else:
            p.append("韌體條件 (a) 無法評估(缺 CTUN.CRt / ATT.DesRoll)")
        p.append(f"全程最高油門輸出 {thro_max:.2f}" + (f"(門檻 {TL_THR_MIN:.1f})" if thro_max < TL_THR_MIN else ""))
        if pre_pwm is not None:
            head = cmd_max - pre_pwm
            if head > 0:
                p.append(f"事件前最高 PWM {pre_pwm:.0f}(離滿載 {cmd_max:.0f} 還有 {head:.0f} µs)")
            else:
                p.append(f"事件前曾有馬達到滿載指令 PWM {pre_pwm:.0f},≥{TL_SAT_ACT * 100:.0f}% 滿載最長 {pre_sat_dur:.1f} s")
        if ae_ok:
            p.append(f"滿載且姿態誤差>{TL_AE_DEG:.0f}° 最長 {b_dur:.1f} s")
        return "、".join(p)

    # ---- 判定 ----
    hits = []                                              # (t_onset, t_alarm, 說明)
    if a_dur >= TL_HOLD_S - _EPS:
        hits.append((a_t, a_t + TL_HOLD_S,
                     f"韌體推力損失條件連續 {a_dur:.1f} s:期望傾角≤{TL_DES_TILT_MAX:.0f}°、姿態誤差<{TL_AE_DEG:.0f}°(仍受控)、"
                     f"油門需求({m.thr_in_src})≥{TL_THR_MIN:.1f} 或已達上限,卻仍在下降"))
    if b_pre is not None:
        hits.append((b_pre["sat_on"], b_pre["t"] + TL_HOLD_S,
                     f"馬達先於 {b_pre['sat_on']:.2f} s 到 ≥{TL_SAT_ACT * 100:.0f}% 滿載(PWM≥{m.pwm_sat():.0f}),"
                     f"姿態誤差在 {b_pre['ae_on']:.2f} s 才超過 {TL_AE_DEG:.0f}°,滿載且失控持續 {b_pre['dur']:.1f} s"
                     "(控制器已用盡馬達推力仍壓不住姿態)"))
    if c_alarm:
        t_al_c = min(t for t, _ in c_alarm)
        hits.append((max(iv.t0, t_al_c - TL_HOLD_S), t_al_c,
                     f"韌體推力損失旗標 / 訊息({'、'.join(c_src)},{t_al_c:.2f} s;韌體需條件持續 {TL_HOLD_S:.0f} s,起點推估提前 {TL_HOLD_S:.0f} s)"))
    if hits:
        t_on, t_al = min(h[0] for h in hits), min(h[1] for h in hits)
        k_on = int(np.clip(np.searchsorted(m.t, t_on), 0, m.t.size - 1))
        ctx_name = "ground" if bool(m.landed[k_on]) else "air"      # 'ground':尚未判定離地(例:起飛瞬間翻覆)
        vals.update(onset_landed=bool(m.landed[k_on]), after_attitude_loss=False,
                    sub_a=a_dur >= TL_HOLD_S - _EPS, sub_b=b_pre is not None, sub_c=bool(c_alarm))
        return Finding(det, DOMAIN, HIT, "critical", float(t_on), float(t_al), ctx_name, vals,
                       "偵測到推力損失:" + ";".join(h[2] for h in hits) + "。請檢查槳葉、馬達座、電調線、電池內阻與電壓;"
                       "本資料沒有 ESC 遙測,無法分辨是哪一顆。")
    if b_post is not None:
        vals.update(after_attitude_loss=True, sub_a=False, sub_b=False, sub_c=False)
        return Finding(det, DOMAIN, SOFT, "medium", b_post["t"], b_post["t"] + TL_HOLD_S, "loss_of_control", vals,
                       f"姿態先失控、馬達才滿載:姿態誤差自 {b_post['ae_on']:.2f} s 起已 >{TL_AE_DEG:.0f}°,"
                       f"{b_post['sat_on']:.2f} s 起才有馬達 ≥{TL_SAT_ACT * 100:.0f}% 滿載(PWM≥{m.pwm_sat():.0f}),同時成立 {b_post['dur']:.1f} s。"
                       "失控 / 翻覆時控制器把一側馬達推到滿載,通常是翻覆的後果,不能證明推力損失。"
                       f"其餘:{margins()};韌體推力損失旗標與訊息皆無。注意:無 ESC 遙測。")
    if not ae_ok and not c_alarm:
        return _na(det, "no_attitude", "沒有 ANG / ATT 姿態資料,無法評估推力損失的姿態判準(旗標與訊息檢查:無異常)。", **vals)
    if m.air_s < MIN_AIR_S:
        return _na(det, "no_air_data", f"馬達雖有轉動,但未落地的飛行資料只有 {m.air_s:.1f} s(<{MIN_AIR_S:.0f} s),不足以排除推力損失。", **vals)
    softs = []                                             # (t_onset, t_alarm, severity, 說明)
    if a_dur >= TL_SOFT_HOLD_S - _EPS:
        softs.append((a_t, a_t + a_dur, "high", f"韌體推力損失條件連續 {a_dur:.1f} s(需 {TL_HOLD_S:.1f} s)"))
    b_pre_soft = next((r for r in b_runs if not r["after"] and r["dur"] >= TL_SOFT_HOLD_S - _EPS), None)
    if b_pre_soft is not None:
        softs.append((b_pre_soft["sat_on"], b_pre_soft["t"] + b_pre_soft["dur"], "high",
                      f"馬達先滿載({b_pre_soft['sat_on']:.2f} s)、姿態後失控,同時成立 {b_pre_soft['dur']:.1f} s(需 {TL_HOLD_S:.1f} s)"))
    if hd_dur >= TL_SOFT_HOLD_S - _EPS and hd_t is not None:
        softs.append((hd_t, hd_t + hd_dur, "medium",
                      f"受控下降中油門需求已滿(≥{TL_THR_MIN:.1f} 或已達上限)連續 {hd_dur:.1f} s(推力餘裕不足的跡象)"))
    if pre_sat_dur >= TL_SOFT_HOLD_S - _EPS and pre_sat_t is not None:
        softs.append((pre_sat_t, pre_sat_t + pre_sat_dur, "medium",
                      f"事件前已有馬達 ≥{TL_SAT_ACT * 100:.0f}% 滿載指令連續 {pre_sat_dur:.1f} s(推力餘裕已用盡)"))
    if softs:
        t_on = min(s[0] for s in softs)
        t_al = min(s[1] for s in softs)
        sev = "high" if any(s[2] == "high" for s in softs) else "medium"
        k_on = int(np.clip(np.searchsorted(m.t, t_on), 0, m.t.size - 1))
        vals.update(after_attitude_loss=False)
        return Finding(det, DOMAIN, SOFT, sev, float(t_on), float(t_al), "ground" if bool(m.landed[k_on]) else "air", vals,
                       "推力損失條件接近成立:" + ";".join(s[3] for s in softs) + f"。{margins()}。"
                       "建議檢查槳葉、馬達與電池;本資料沒有 ESC 遙測。")
    if not a_valid:
        return _na(det, "partial_data", f"缺 CTUN.CRt 或 ATT.DesRoll/DesPitch,韌體推力損失條件 (a) 無法評估,不能下「正常」;"
                   f"其餘檢查未見異常:{margins()};韌體旗標 bit0 與推力損失訊息皆無。", **vals)
    return Finding(det, DOMAIN, CLEAR, "info", None, None, "air", vals,
                   f"推力損失檢查未成立(ArduCopter thrust_loss_check 條件 + 滿載補充規則):{margins()};"
                   "韌體旗標 bit0 與推力損失訊息皆無。注意:這批資料沒有 ESC / 轉速遙測,指令正常但實際失效的馬達無法排除。")


# ======================================================================================
# 2. motor_thrust_imbalance
# ======================================================================================

def _near_ground(m: _Mot, a: Anchors) -> tuple[np.ndarray, str]:
    """每個取樣是否「貼地」:離地(CTUN.Alt − 地面高度)≤ 1.5 m,或已過觸地錨點 a.t_touch。→ (mask, 依據)。

    兩者都沒有 → 全部 False(依據 'unknown'):無法確認在地面,就不能把症狀歸給地面翻覆。
    """
    near = np.zeros(m.t.size, dtype=bool)
    src = []
    if m.agl is not None:
        near |= m.agl <= IMB_NEAR_GROUND_M
        src.append("agl")
    tt = a.t_touch
    if tt is not None and m.iv.t0 <= tt <= m.iv.t1:
        near |= m.t >= tt
        src.append("t_touch")
    return near, "+".join(src) or "unknown"


def _imbalance_runs(m: _Mot, unbalanced: np.ndarray, near: np.ndarray):
    """把不平衡取樣依情境拆成 air / ground_idle / idle_air 三條,各做 8 s 窗累計。→ {ctx: (bad, cum)}"""
    gate = m.spin & ~m.landed & unbalanced
    out = {}
    for name, sel in (("air", ~m.idle), ("ground_idle", m.idle & near), ("idle_air", m.idle & ~near)):
        bad = gate & sel
        out[name] = (bad, _cum_window(bad, m.t, m.dt, IMB_WIN_S))
    return out


def _first_trigger(bad: np.ndarray, cum: np.ndarray, m: _Mot, need: float):
    """第一個累計 ≥ need 的取樣 k → (t_onset, t_alarm);onset = 該窗內第一個不平衡樣本。沒有 → None。"""
    ks = np.flatnonzero(cum >= need - _EPS)
    if ks.size == 0:
        return None
    k = int(ks[0])
    j0 = int(np.searchsorted(m.t, m.t[k] - IMB_WIN_S, side="right"))
    first = j0 + int(np.argmax(bad[j0:k + 1]))
    return float(m.t[first]), float(m.t[k])


@guarded("motor_thrust_imbalance", DOMAIN)
def motor_thrust_imbalance(ctx: Ctx, a: Anchors):
    """推力失衡(MOTB.FailFlags bit1 = thrust_balanced 掉到 0;只有 N ≥ 6 才有)。

    bad = 轉動 ∧ 未落地 ∧ 失衡(主:FailFlags bit1 == 0;備:由 RCOU 重算的比值 ≥ 1.5 閂鎖、≤ 1.25 釋放);
    8 s 窗內累計 ≥ 2.0 s 才算。依 CTUN.ThO(1 s 中位數 ≤ 0.03 = 怠速)與是否貼地拆成三條:
      'air'         → 回一筆(HIT / SOFT / CLEAR):計入 H_MOTOR_THRUST;
      'ground_idle' → 怠速且貼地(離地 ≤ 1.5 m 或已過 a.t_touch),成立時另回一筆 HIT:翻覆的症狀,只計入 H_LAND_TIPOVER;
      'idle_air'    → 怠速但無法確認貼地(例:空中收油門),成立時另回一筆 HIT,中性描述,不計入任何假說。
    峰值比 / 嚴重秒數一律由 RCOU 重算(values['peak_ratio_src'] = 'recomputed');來源是旗標而重算比值未達閂鎖時,
    文字不印峰值比(兩者不一致,只留在 values)。
    """
    det = "motor_thrust_imbalance"
    na = _na_reason(a, det)
    if na is not None:
        return na
    m, why = _setup(ctx, a)
    if m is None:
        return _na(det, *why)
    esc = _has_esc(ctx)
    if m.n_motors < IMB_MIN_MOTORS:
        return _na(det, "n_motors<6", f"機型只有 {m.n_motors} 顆馬達:韌體的推力平衡旗標只存在於六軸以上,此項不適用。", n_motors=m.n_motors)
    if m.idle is None:
        return _na(det, "no_tho", "缺少 CTUN.ThO,無法區分空中 / 地面怠速,為避免把地面症狀當成空中證據,此項不適用。")
    ratio, f = _thrust_ratio(m.act, m.batv, m.expo, m.dt)
    latch = _latch(ratio)
    use_flag = m.flags is not None and m.flags_valid
    flag_bad = (m.flags.astype(np.int64) & FF_BIT_BALANCED) == 0 if m.flags is not None else None
    unbalanced = flag_bad if use_flag else latch
    near, near_src = _near_ground(m, a)
    runs_ = _imbalance_runs(m, unbalanced, near)
    air_mask = m.spin & ~m.landed & ~m.idle
    stat = {
        "source": "MOTB.FailFlags" if use_flag else "recomputed",
        "n_motors": m.n_motors, "rcou_channels": m.ch, "air_s": round(m.air_s, 1),
        "air_ratio_p99": round(float(np.percentile(ratio[air_mask], 99)), 3) if air_mask.sum() > 20 else None,
        "air_ratio_max": round(float(ratio[air_mask].max()), 3) if air_mask.any() else None,
        "flag_vs_recomputed_pct": round(100.0 * float(np.mean(flag_bad[m.spin] == latch[m.spin])), 2)
        if flag_bad is not None and m.spin.any() else None,
        "near_ground_src": near_src,
        "esc_telemetry": esc,
        "thresholds": {"latch": IMB_RATIO_LATCH, "release": IMB_RATIO_RELEASE, "severe": IMB_RATIO_SEVERE,
                       "cum_s": IMB_CUM_S, "win_s": IMB_WIN_S, "idle_tho": IDLE_THO, "near_ground_m": IMB_NEAR_GROUND_M},
    }
    out: list[Finding] = []
    src_zh = "推力平衡旗標(MOTB.FailFlags bit1)" if use_flag else "由 RCOU 重算的推力平衡閂鎖"

    def peak_info(bad):
        """(重算)峰值比 / 峰值時最高的馬達通道 / 剛開始 1 s 內推力最高的一半通道(哪一側先衝高)/ 嚴重(≥1.7)秒數。"""
        if not bad.any():
            return {}
        i = int(np.argmax(np.where(bad, ratio, -np.inf)))
        j = int(np.argmax(f[i]))
        i0 = int(np.argmax(bad))
        w = (m.t >= m.t[i0]) & (m.t <= m.t[i0] + 1.0)
        top0 = np.argsort(-f[w].mean(axis=0))[: m.n_motors // 2]
        info = {"peak_ratio": round(float(ratio[i]), 2), "peak_ratio_src": "recomputed", "peak_t": round(float(m.t[i]), 2),
                "peak_motor_channel": int(m.ch[j]),
                "onset_high_channels": sorted(int(m.ch[k]) for k in top0),
                "severe_s": round(float((bad & (ratio >= IMB_RATIO_SEVERE)).sum() * m.dt), 1)}
        if m.agl is not None:
            info["agl_at_onset_m"] = round(float(m.agl[i0]), 2)
        return info

    def peak_txt(v) -> str:
        """文字裡的峰值比:重算比值有達閂鎖(和判定一致)才印,否則註明不一致。"""
        pr = v.get("peak_ratio")
        if pr is None:
            return ""
        if pr >= IMB_RATIO_LATCH:
            return f"重算峰值比 {pr},嚴重(≥{IMB_RATIO_SEVERE:.1f}){v.get('severe_s', 0)} s"
        return f"由 RCOU 重算的比值最高只有 {pr}(未達閂鎖 {IMB_RATIO_LATCH:.1f},與旗標不一致)"

    # ---- air ----
    bad_a, cum_a = runs_["air"]
    cmax = float(cum_a.max()) if cum_a.size else 0.0
    va = dict(stat, bad_air_s=round(float(bad_a.sum() * m.dt), 2), max_cum_8s=round(cmax, 2), **peak_info(bad_a))
    if m.air_s < MIN_AIR_S:
        out.append(_na(det, "no_air_data", f"未落地的飛行資料只有 {m.air_s:.1f} s(<{MIN_AIR_S:.0f} s),不足以評估空中推力平衡。", **va))
    elif cmax >= IMB_CUM_S - _EPS:
        t_on, t_al = _first_trigger(bad_a, cum_a, m, IMB_CUM_S)
        out.append(Finding(det, DOMAIN, HIT, "high", t_on, t_al, "air", va,
                           f"空中推力失衡持續:最大輸出馬達超過平均 {IMB_RATIO_LATCH:.1f} 倍的狀態在 {IMB_WIN_S:.0f} s 內累計 {cmax:.1f} s"
                           f"(來源:{stat['source']};{peak_txt(va)};最高通道 C{va.get('peak_motor_channel', '-')})。"
                           "代表有馬達 / 電調 / 槳力量不足或機身重心偏移,請檢查各馬達與槳;本資料沒有 ESC 遙測。"))
    elif cmax >= IMB_SOFT_CUM_S - _EPS:
        t_on, t_al = _first_trigger(bad_a, cum_a, m, IMB_SOFT_CUM_S)
        out.append(Finding(det, DOMAIN, SOFT, "medium", t_on, t_al, "air", va,
                           f"空中推力平衡短暫失衡:{IMB_WIN_S:.0f} s 內累計 {cmax:.1f} s(需 {IMB_CUM_S:.1f} s)。持續出現請檢查重心與各馬達輸出。"))
    else:
        p99 = stat["air_ratio_p99"]
        out.append(Finding(det, DOMAIN, CLEAR, "info", None, None, "air", va,
                           f"空中推力平衡正常:{src_zh}判為失衡的空中時間累計 {float(bad_a.sum() * m.dt):.1f} s、任一 {IMB_WIN_S:.0f} s 窗最多 {cmax:.1f} s(門檻 {IMB_CUM_S:.1f} s);"
                           + (f"懸停重算比值 p99 {p99:.2f}(閂鎖 {IMB_RATIO_LATCH:.1f})。" if p99 is not None else "")
                           + "注意:無 ESC 遙測,只能由指令輸出判斷。"))

    # ---- ground_idle(症狀)/ idle_air(中性):成立才另外回一筆 ----
    bad_g, cum_g = runs_["ground_idle"]
    if bad_g.any() and float(cum_g.max()) >= IMB_CUM_S - _EPS:
        t_on, t_al = _first_trigger(bad_g, cum_g, m, IMB_CUM_S)
        vg = dict(stat, bad_idle_s=round(float(bad_g.sum() * m.dt), 2), max_cum_8s=round(float(cum_g.max()), 2),
                  symptom_of="ground_tipover", **peak_info(bad_g))
        out.append(Finding(det, DOMAIN, HIT, "medium", t_on, t_al, "ground_idle", vg,
                           f"油門已在怠速(CTUN.ThO≤{IDLE_THO:.2f})、貼地(離地≤{IMB_NEAR_GROUND_M:.1f} m 或已過觸地)且未判定落地時,"
                           f"{src_zh}判為失衡共 {float(bad_g.sum() * m.dt):.1f} s({peak_txt(vg)}):"
                           "控制器在地面硬要修正姿態,一側馬達衝高、另一側壓在怠速。這是地面翻覆的症狀而非馬達故障起點,不計入馬達 / 電調失效。"))
    bad_i, cum_i = runs_["idle_air"]
    if bad_i.any() and float(cum_i.max()) >= IMB_CUM_S - _EPS:
        t_on, t_al = _first_trigger(bad_i, cum_i, m, IMB_CUM_S)
        vi = dict(stat, bad_idle_s=round(float(bad_i.sum() * m.dt), 2), max_cum_8s=round(float(cum_i.max()), 2),
                  **peak_info(bad_i))
        agl_txt = (f"離地約 {vi['agl_at_onset_m']:.1f} m" if vi.get("agl_at_onset_m") is not None else "沒有可用的高度 / 觸地資料")
        out.append(Finding(det, DOMAIN, HIT, "medium", t_on, t_al, "idle_air", vi,
                           f"油門在怠速(CTUN.ThO≤{IDLE_THO:.2f})但無法確認在地面({agl_txt})時,{src_zh}判為失衡共 "
                           f"{float(bad_i.sum() * m.dt):.1f} s({peak_txt(vi)}):低油門時推力比值本來就容易放大(例如空中收油門),"
                           "不能單獨當成馬達故障或地面翻覆的證據,僅供對照。"))
    return out


# ======================================================================================
# 3. yaw_imbalance_iterm
# ======================================================================================

@guarded("yaw_imbalance_iterm", DOMAIN)
def yaw_imbalance_iterm(ctx: Ctx, a: Anchors):
    """偏航失衡 / 姿態積分項飽和。

    韌體 yaw_imbalance_check()(已對照原始碼):yaw I 項經 0.05 Hz 低通(落地 / 未轉動 / thrust boost 時歸零,
    且濾波值不得大於 |I|),|濾波值| > 75% IMAX 或 I == IMAX 就立即送 'Yaw Imbalance'(之後每 10 s 最多一次)。
    本模組每軸(PIDY / PIDR / PIDP;滾轉 / 俯仰是本模組延伸,稱 I-term wind-up)用 400 Hz 自己的時間軸重現上述濾波,
    over = 轉動 ∧ 未落地 ∧ 非 thrust boost ∧ (濾波比值 > 0.75 ∨ |I| ≥ 0.999 IMAX):
      HIT  = over 連續 ≥ 10 s(本模組的持續規則;規格寫的「10 s」其實是韌體訊息間隔),或有 'Yaw Imbalance' 訊息
             (t_onset = 訊息時間:韌體條件一成立就送,沒有持續時間);
      SOFT = 偏航 over 出現任何時間(韌體此時已會警告,不能說「可排除」);滾轉 / 俯仰 over 連續 ≥ 5 s;
      CLEAR = 有空中資料且以上皆無。
    永遠回報各軸原始 / 低通後的最大比值,讓 UI 顯示餘裕。主要是排除用:124 與健康飛行都遠低於門檻,沒有分離。
    """
    det = "yaw_imbalance_iterm"
    na = _na_reason(a, det)
    if na is not None:
        return na
    m, why = _setup(ctx, a)
    if m is None:
        return _na(det, *why)
    iv = m.iv
    esc = _has_esc(ctx)
    gate10 = m.air.copy()
    if m.flags is not None:
        gate10 &= (m.flags.astype(np.int64) & FF_BIT_BOOST) == 0          # thrust boost 時韌體把濾波器歸零
    gate_t = gate10.astype(float)
    per: dict[str, dict] = {}
    fires: list[tuple[float, float, str, float]] = []      # (t_onset, t_alarm, axis, dur)
    softs: list[tuple[float, float, str, float]] = []
    rc = 1.0 / (2.0 * np.pi * YI_LPF_HZ)
    for tab, imax_key, zh in YI_AXES:
        tb = ctx.table(tab)
        if tb is None or "I" not in tb:
            continue
        t = tb["t"]
        i0, i1 = int(np.searchsorted(t, iv.t0, side="left")), int(np.searchsorted(t, iv.t1, side="right"))
        if i1 - i0 < 100:
            continue
        ta, xi = t[i0:i1], np.asarray(tb["I"][i0:i1], dtype=float)
        imax_p = _pnum(ctx, imax_key, YI_IMAX_DEFAULT)
        imax = imax_p if imax_p is not None and imax_p > 0 else YI_IMAX_DEFAULT
        gate = np.interp(ta, m.t, gate_t) > 0.5
        dts = np.diff(ta)
        dts = dts[np.isfinite(dts) & (dts > 0)]
        dta = float(np.median(dts)) if dts.size else 1.0 / 400.0
        y = _lpf_yaw_fw(xi, gate, dta / (dta + rc))       # 韌體 LowPassFilter:alpha = dt / (dt + 1/(2π fc))
        raw_r = np.abs(xi) / imax
        lpf_r = np.abs(y) / imax
        rg = raw_r[gate]
        rg = rg[np.isfinite(rg)]
        info = {"imax": imax, "n_gate": int(rg.size),
                "raw_max_ratio": round(float(rg.max()), 4) if rg.size else None,
                "lpf_max_ratio": round(float(lpf_r[gate].max()), 4) if rg.size else None}
        if imax_p is None or imax_p <= 0:
            info["imax_param_invalid"] = True
        with np.errstate(invalid="ignore"):
            over = gate & ((lpf_r > YI_RATIO) | (raw_r >= YI_IMAX_EQ))
        info["over_s"] = round(float(over.sum() * dta), 2)
        per[tab] = info
        soft_need = YI_FW_SOFT_S if tab == "PIDY" else YI_SOFT_HOLD_S
        first_soft = None
        for j0, j1 in _runs_v(over, ta, max_gap=0.5):
            dur = float(ta[j1] - ta[j0] + dta)
            if dur >= YI_HOLD_S - _EPS:
                fires.append((float(ta[j0]), float(ta[j0]) + YI_HOLD_S, tab, dur))
                break
            if first_soft is None and dur >= soft_need - _EPS:
                first_soft = (float(ta[j0]), float(ta[j0]) + dur, tab, dur)   # 記下但繼續找:後面可能有 ≥ 10 s 的區段
        if first_soft is not None:
            softs.append(first_soft)
    msgs = ctx.messages_between(iv.t0, iv.t1 + 1.0, YI_MSG_RX)

    pidy = per.get("PIDY")
    worst = max(per.items(), key=lambda kv: kv[1]["raw_max_ratio"] or 0.0) if per else None
    vals = {
        "axes": per, "esc_telemetry": esc,
        "yaw_raw_max_ratio": None if pidy is None else pidy["raw_max_ratio"],
        "worst_axis": None if worst is None else worst[0],
        "worst_raw_max_ratio": None if worst is None else worst[1]["raw_max_ratio"],
        "msg_hits": len(msgs), "air_s": round(m.air_s, 1),
        "thresholds": {"ratio": YI_RATIO, "imax_eq": YI_IMAX_EQ, "hold_s": YI_HOLD_S, "lpf_hz": YI_LPF_HZ,
                       "yaw_soft_s": YI_FW_SOFT_S, "rp_soft_s": YI_SOFT_HOLD_S},
        "rule": "ArduCopter yaw_imbalance_check (checked against master source: 0.05 Hz LPF reset when landed / boost, "
                "capped at |I|, >75% IMAX or I==IMAX, warns immediately; 10 s = message interval) + module 10 s persistence "
                "for HIT; roll/pitch = module extension",
    }
    if fires or msgs:
        cands = [(f[0], f[1], f[2], f[3]) for f in fires]
        if msgs:
            # 韌體條件一成立就送訊息(YAW_IMBALANCE_WARN_MS 只是訊息間隔),所以訊息時間 = 條件起點
            cands.append((float(msgs[0][0]), float(msgs[0][0]), "MSG", 0.0))
        t_on, t_al, ax, dur = min(cands)
        label = {"PIDY": "偏航失衡(Yaw Imbalance)", "PIDR": "滾轉積分項飽和(I-term wind-up)", "PIDP": "俯仰積分項飽和(I-term wind-up)",
                 "MSG": "韌體 Yaw Imbalance 訊息"}[ax]
        vals["fired_axis"] = ax
        how = ("韌體在偏航積分項(0.05 Hz 低通)超過 IMAX 的 75% 時立即發出" if ax == "MSG" else
               f"積分項低通後超過 IMAX 的 {YI_RATIO * 100:.0f}%(或頂到 IMAX)並持續 {dur:.0f} s")
        return Finding(det, DOMAIN, HIT, "high", float(t_on), float(t_al), "air", vals,
                       f"{label}:{how}。"
                       "代表某幾顆馬達推力長期不一致(槳 / 馬達老化、安裝角度或電調校準不一致);請檢查轉向、槳型、電調行程校準與重心。")
    if pidy is None:
        return _na(det, "no_pidy", "沒有 PIDY(偏航 PID)紀錄,無法評估偏航失衡。", **vals)
    if m.air_s < MIN_AIR_S or not pidy["n_gate"]:
        return _na(det, "no_air_data", f"未落地的飛行資料只有 {m.air_s:.1f} s,不足以評估積分項。", **vals)
    if softs:
        t_on, t_al, ax, dur = min(softs)
        vals["fired_axis"] = ax
        if ax == "PIDY":
            txt = (f"偏航積分項低通後超過 IMAX 的 {YI_RATIO * 100:.0f}%(或頂到 IMAX)共 {dur:.1f} s:韌體這時已會發出 Yaw Imbalance 警告,"
                   f"但未達本模組持續 {YI_HOLD_S:.0f} s 的門檻。建議檢查馬達轉向、槳型與電調校準。")
        else:
            txt = (f"{dict((k, z) for k, _, z in YI_AXES)[ax]}軸積分項低通後超過 IMAX 的 {YI_RATIO * 100:.0f}% 已 {dur:.0f} s"
                   f"(需 {YI_HOLD_S:.0f} s):建議留意馬達 / 重心是否不平衡。")
        return Finding(det, DOMAIN, SOFT, "medium", float(t_on), float(t_al), "air", vals, txt)
    return Finding(det, DOMAIN, CLEAR, "info", None, None, "air", vals,
                   f"偏航積分項最大只有 IMAX 的 {pidy['raw_max_ratio'] * 100:.0f}%(低通後 {pidy['lpf_max_ratio'] * 100:.0f}%;"
                   f"韌體 Yaw Imbalance 條件:低通後 >{YI_RATIO * 100:.0f}% 或頂到 IMAX,從未成立);"
                   f"滾轉 / 俯仰最大 {max(per.get('PIDR', {}).get('raw_max_ratio') or 0.0, per.get('PIDP', {}).get('raw_max_ratio') or 0.0) * 100:.0f}%,"
                   "無『Yaw Imbalance』訊息 → 可排除偏航失衡與馬達長期不一致。注意:無 ESC 遙測。")


# ======================================================================================
# 入口
# ======================================================================================

def run(ctx: Ctx, a: Anchors) -> list[Finding]:
    out: list[Finding] = []
    for det in (motor_thrust_loss_check, motor_thrust_imbalance, yaw_imbalance_iterm):
        out.extend(det(ctx, a))
    return out
