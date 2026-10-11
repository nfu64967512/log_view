"""電源偵測模組(power):電池過低 / 電壓崩落 / 電源軌異常,加上電流計資料品質與保護參數稽核。

偵測器(各自獨立、各回傳 ≥ 1 筆 Finding;例外由 @guarded 降級為 NA):
  power_low_voltage         飛行中 1 秒中位數電壓 < MOT_BAT_VOLT_MIN,或持續低於 BATT_CRT_VOLT,或飛行中 ERR 子系統 6
                            (MOT_BAT_VOLT_* 沒設時門檻用估計串數 × 3.3 V/cell,串數只會少估不會多估,且最多給 SOFT)
  power_voltage_collapse    解鎖中 1 秒內掉壓 > 約 20% 滿電壓(12S ≈ 10 V)且持續 ≥ 0.3 s;單點跨線只給 SOFT;log 尾端拔電池不算
  power_rail_fault          飛控 5V / 伺服軌 / MCU 3.3V 掉電、電源模組失效、週邊過流(累計旗標解鎖前就有的只給 SOFT)
  power_data_quality        (一般健檢)BAT.Curr 讀值無效 → 電流衍生欄位不可用
  power_failsafe_config_audit (一般健檢)電池保護參數互相矛盾 / 失效保護形同虛設

重要資料特性(實測於 00000117..124):
  - BAT.Curr 在這批 log 都**無效**(懸停 0.004 A、與油門呈負相關),所以 CurrTot / EnrgTot / RemPct / Res / VoltR
    都是假值;本模組只用電壓,絕不輸出內阻 / 剩餘電量類結論。電流類故障(電調 / 接頭)因此無法被排除 →
    CLEAR 的文字只說「依電壓判斷」,H_POWER 的信心上限由 power_data_quality 的 HIT 決定(engine 端)。
  - USB 供電的 log(117/122/123)BAT.Volt ≈ 0 → NA(不是 CLEAR);log 尾端拔電池(118/120/124)的 -26..-33 V
    掉壓不是飛行中掉電:只分析解鎖區間,且尾端另外辨識並回報。
  - BAT 約 10 Hz、有最長 0.8 s 的空洞:所有「最近 1 秒」都用時間窗(searchsorted),不用樣本數。
  - 全部判斷只看電壓與電源軌;通過 / 未通過都以 a.interval(解鎖區間)內的樣本為準,上鎖之後不納入。

通用規則:
  - a.interval 為 None(從未解鎖)→ 五個 id 都回 NA、context='not_armed'(config_audit 另在 values 附上靜態稽核結果;
    其餘四項附上整份 log 的供電來源 power_source,USB 供電的文字寫「電池分析不適用」;collapse 另附尾端拔電池 log_tail)。
  - run() 在 power_data_quality 判定電流計無效時,於 power_low_voltage 的 CLEAR 文字補上「電流類故障無法排除」的保留(驗收 A7)。
  - HIT / SOFT 的嚴重度採目錄值;CLEAR / NA 一律 'info'。一般健檢的 HIT 帶 context='standing'(持續性狀態,不看時間先後)。
  - 這些偵測器本身不貼因果標籤;t_onset = 條件開始成立的時間,由 engine 對照 t_warn / t_loss 貼 tag。
    但 HIT / SOFT 的文字會照原樣顯示:條件在 a.t_loss 之後才開始時,文字寫成「屬後果」,不寫「可能導致失控」
    (values['after_loss'] 同步標記)。
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from .ctx import Ctx
from .guard import guarded
from .types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval, Finding
from .util import runs

DOMAIN = "power"

# ---------------------------------------------------------------------------
# 門檻(常數;括號內是 00000124 實測的「失事 vs 對照」間距)
# ---------------------------------------------------------------------------
V1_WIN_S = 1.0                  # 電壓中位數的時間窗(秒)
V1_MIN_SAMPLES = 3              # 窗內至少幾個樣本才算數(空洞太大就不下判斷)
CELL_V_MAX = 4.2                # 單顆滿電壓:cells = round(MOT_BAT_VOLT_MAX / 4.2)
CELL_V_CRIT = 3.3               # MOT_BAT_VOLT_MIN 沒設時的備援:3.3 V/cell(實測 124:MOT_BAT_VOLT_MIN=39.6 V=3.3 V/cell)
# MOT_BAT_VOLT_MAX 沒設(ArduPilot 預設 0)時估串數:⌈(到上鎖為止 1 秒中位數最高電壓) / 4.35⌉。
# 4.35 V = LiHV 滿電:只要讀值 ≤ 4.35 V/cell,估計值「不會多於」實際串數(多估一串會把 3.3 V/cell 門檻抬高 3.3 V,
# 健康飛行就變成 critical HIT;少估只會讓門檻變寬鬆)。估出來的門檻證據力較弱 → 低電壓最多給 SOFT(見 power_low_voltage)。
# 實測 124:到上鎖為止 1 秒中位數最高 50.66 V(4.22 V/cell)→ ⌈11.65⌉ = 12(與 MOT_BAT_VOLT_MAX 50.4 V 一致);
# 讀值偏高 3% 以上(> 52.2 V)才會多估成 13 串。
CELL_V_REST_MAX = 4.35
CELL_EST_EPS = 1e-3             # ⌈x⌉ 的浮點容差(17.4 / 4.35 = 4.000000000000001 不能進位成 5)
LOW_TIMER_MIN_S = 1.0           # BATT_LOW_TIMER 的下限(秒);實測 124:10 s
LAST_WINDOW_S = 30.0            # 回報「事件前 30 秒」統計的窗長
BATTERY_PRESENT_V = 5.0         # 解鎖期間 BAT.Volt 中位數低於此 → USB 供電 / 未接電池(實測:USB log ≤ 0.08 V;有電池 ≥ 47.8 V)
MIN_ARMED_SAMPLES = 20          # 解鎖區間內至少幾筆 BAT 才有資格下結論(10 Hz → 2 秒)
MIN_ARMED_S = 3.0               # 解鎖至少幾秒
RUN_MAX_GAP_S = 1.0             # 連續低電壓區段允許的資料空洞(BAT 實測空洞 ≤ 0.8 s)

# 電壓崩落:1 秒內掉壓 ≥ 滿電壓的 20%(12S/50.4 V → 10.08 V;規格寫 -10 V)
# 穩健版(審查後):規格的「單點對單點」比較會把一個壞樣本(0 V)或向上跳動後的回落當成崩落,因此
#   參考電壓 ref = min(中位數[t-1.5, t-0.5], 中位數[t-5, t-0.5])(1 秒前的水準;取較低者 → 向上跳動回落不算掉壓)
#   單點壓降 inst = v(t) - ref;持續壓降 persist = max(v 在 [t, t+0.3 s] 且至少含下一個樣本) - ref
#   persist ≤ -門檻 → HIT;只有 inst ≤ -門檻(沒持續 0.3 s 就恢復)→ SOFT 'transient';log 在掉壓上結束時免持續檢查。
# 實測 124:解鎖中最大單點壓降 3.57 V(飛行 1)/ 3.68 V(飛行 2,起飛下陷),持續壓降 2.97 / 3.12 V(規格的單點對單點
# 3.52 / 3.65 V);拔電池尾端 -26..-33 V(52..65%)→ 對 3.68 V 有 2.7 倍餘裕。
# 解鎖中單點最大偏離 1 秒中位數:124 飛行 1 2.22 V、飛行 2 2.83 V(含翻滾)→ 真實資料沒有會跨門檻的壞樣本。
COLLAPSE_LAG_S = 1.0
COLLAPSE_PRE_HALF_S = 0.5       # 1 秒前參考窗的半寬:[t-1.5, t-0.5]
COLLAPSE_PRE_MIN_N = 2          # 參考窗內至少幾個樣本(BAT 空洞最長 0.8 s;空洞太大就不比較)
COLLAPSE_LONG_S = 5.0           # 長參考窗 [t-5, t-0.5](至少 V1_MIN_SAMPLES 個樣本)
COLLAPSE_PERSIST_S = 0.3        # 掉壓要持續的時間(至少含掉壓樣本的下一筆)
COLLAPSE_DROP_FRAC = 0.20
COLLAPSE_ONSET_V = 1.0          # 起點 = 掉壓前最後一個「離參考電壓不到 1 V」的樣本
ON_GROUND_THO = 0.03            # 崩落發生前 2 秒 CTUN.ThO 中位數 ≤ 此值 = 在地面(落地後拔電池,不是飛行中掉電)
ON_GROUND_LOOKBACK_S = 2.0

# log 尾端拔電池:最後 TAIL_MAX_S 秒內電壓掉到參考值的 60% 以下(實測 118 0.6 s 50.2→16.9 V;120 0.6 s;124 0.5 s 47.5→21.2 V)
TAIL_MAX_S = 1.5
TAIL_END_FRAC = 0.6
TAIL_FALL_FRAC = 0.9            # 電壓掉到參考值 90% 以下才算尾端開始(平台結束)
TAIL_REF_FAR_S = 8.0            # 參考電壓取最後 8..2 秒的中位數
TAIL_REF_NEAR_S = 2.0
TAIL_TOL_V = 0.5                # 尾端起點往回找「離參考值 0.5 V 以內」的最後一個樣本
TAIL_MIN_SAMPLES = 30
TAIL_TRIM_S = 0.1               # 統計時再往前裁掉的秒數

# 電源軌(實測 124 全程:Vcc 最低 5.182 V、MCU 3.289..3.2937 V、解鎖中 VServo 最低 4.855 V;AccFlags 只有 1)
RAIL_VCC_MIN_V = 4.5            # 磚供電最低 5.182 V → 餘裕 0.68 V;USB 供電 4.65..4.9 V(不在此偵測範圍)
RAIL_MCU_MIN_V = 3.1            # 實測 3.289..3.296 V
RAIL_VSERVO_MIN_V = 4.5         # 解鎖中實測最低 4.855 V → 餘裕 0.36 V
RAIL_VSERVO_VALID_V = 4.5       # 解鎖中 VServo 中位數低於此 → 伺服軌沒接 / 沒量測,不評估
RAIL_OVERCURRENT_MASK = 8 | 16  # POWR.Flags/AccFlags:bit3 週邊過流、bit4 週邊高功率過流(實測從未出現)
RAIL_OVERCURRENT_BITS = (8, 16)  # AccFlags 是開機以來的累計值:逐位元找「整份 log 第一次出現」的時間,解鎖前就有的不算飛行中事件
RAIL_BRICK_PRESENT_FRAC = 0.5   # 解鎖期間至少這個比例有磚旗標,才把「磚旗標掉了」當成失效(沒接磚的機型不誤報)
FLAG_BRICK = 1
FLAG_USB = 4
RAIL_DIP_WIN_S = 10.0           # 回報 Vcc 相對自己前 10 秒中位數的下陷

# 電流計資料品質(實測:懸停中位數 0.0042 A 對 1.0 A 門檻 240 倍;起飛下陷 3.1..3.9 V 對 1.5 V 門檻 2 倍)
DQ_CURR_MAX_A = 1.0
DQ_SAG_MIN_V = 1.5
DQ_MIN_LOAD_S = 20.0
DQ_THR_FRAC = 0.5               # 「負載」= MOTB.ThrOut > 0.5 × MOT_THST_HOVER(油門被壓在 0.5×hover 的姿態優先混控不算)
DQ_HOVER_DEFAULT = 0.35         # MOT_THST_HOVER 與 CTUN.ThH 都沒有時的最後備援(ArduPilot 預設值)
DQ_REST_FAR_S = 10.0            # 靜止電壓 = 解鎖前 [t_arm-10, t_arm-0.5] 的中位數
DQ_REST_NEAR_S = 0.5
DQ_DO_NOT_USE = ["BAT.Curr", "BAT.CurrTot", "BAT.EnrgTot", "BAT.RemPct", "BAT.Res", "BAT.VoltR"]

SEV = {"power_low_voltage": "critical", "power_voltage_collapse": "critical", "power_rail_fault": "high",
       "power_data_quality": "medium", "power_failsafe_config_audit": "medium"}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _r(x, nd: int = 3):
    """數值 → 可 JSON 序列化的 float(NaN / None / inf → None)。"""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _na(det_id: str, context: str, text: str, **values) -> Finding:
    return Finding(det_id, DOMAIN, NA, severity="info", context=context, values=values, text_zh=text)


_NA_TEXT = {
    "not_armed": "未解鎖(桌上 / 手持 log),沒有飛行中的電源資料可分析,不適用。",
    "no_data": "缺少必要的電源資料表,無法判斷(不是正常)。",
    "too_short": "解鎖時間太短或電池資料太少,無法判斷。",
    "usb_or_no_battery": "這份 log 是 USB 供電 / 未接飛行電池(BAT.Volt 趨近 0 V),電池分析不適用。",
    "no_hover": "有負載的飛行(MOTB.ThrOut > 0.5 × 懸停油門)不到 20 秒,無法評估電流計是否有效。",
    "no_rest_voltage": "解鎖前沒有靜止電壓可比較(一開機就解鎖),無法評估電流計是否有效。",
}


def _na_reason(det_id: str, why: str, **values) -> Finding:
    return _na(det_id, why, _NA_TEXT.get(why, "資料不足,無法判斷。"), **values)


def _reduce_idx(v: np.ndarray, lo: np.ndarray, hi: np.ndarray, how: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """對每個 k 取 v[lo[k]:hi[k]] 的中位數(how='median')或最大值(how='max');空窗 → NaN。

    回傳 (結果, 窗內樣本數)。向量化,分塊避免記憶體爆掉。
    """
    n = v.size
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    out = np.full(lo.size, np.nan)
    cnt = np.clip(hi - lo, 0, None)
    if n == 0 or lo.size == 0:
        return out, cnt
    w = int(cnt.max())
    if w <= 0:
        return out, cnt
    base = np.arange(w)
    chunk = max(1, 1_500_000 // w)
    for s in range(0, lo.size, chunk):
        e = min(lo.size, s + chunk)
        idx = lo[s:e, None] + base[None, :]
        ok = idx < hi[s:e, None]
        vals = np.where(ok, v[np.clip(idx, 0, n - 1)], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[s:e] = np.nanmedian(vals, axis=1) if how == "median" else np.nanmax(vals, axis=1)
    return out, cnt


def _window_median(t: np.ndarray, v: np.ndarray, win: float, min_n: int = V1_MIN_SAMPLES) -> np.ndarray:
    """每個樣本取 ±win/2 秒「時間窗」內樣本的中位數(窗內樣本數 < min_n → NaN)。

    用 searchsorted 取窗,所以資料空洞(BAT 最長 0.8 s)時窗內樣本數自動變少,不會把「10 個樣本」
    誤當成「1 秒」。
    """
    if t.size == 0:
        return np.full(0, np.nan)
    lo = np.searchsorted(t, t - win / 2.0, side="left")
    hi = np.searchsorted(t, t + win / 2.0, side="right")
    med, cnt = _reduce_idx(v, lo, hi, "median")
    med[cnt < min_n] = np.nan
    return med


def unplug_tail(t: np.ndarray, v: np.ndarray) -> dict | None:
    """log 尾端「拔電池」辨識:最後 ≤ 1.5 s 內電壓從平台掉到參考值 60% 以下。

    參考值 = 最後 8..2 秒的中位數(需 > 5 V);起點 = 電壓離開 90% 參考值之前、最後一個離參考值 0.5 V 以內的樣本。
    回傳 {t_start, v_before, v_end, ref, tail_len_s, slope_v_s} 或 None。
    與「飛行中掉電」的差別只在「解鎖與否」:呼叫端要自己判斷尾端時有沒有在解鎖。
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.size < TAIL_MIN_SAMPLES:
        return None
    t_end = float(t[-1])
    mref = (t >= t_end - TAIL_REF_FAR_S) & (t <= t_end - TAIL_REF_NEAR_S)
    if int(mref.sum()) < 5:
        return None
    ref = float(np.median(v[mref]))
    if ref < BATTERY_PRESENT_V or v[-1] >= TAIL_END_FRAC * ref:
        return None
    above = np.flatnonzero(v >= TAIL_FALL_FRAC * ref)
    if above.size == 0:
        return None
    k0 = int(above[-1])
    if t_end - t[k0] > TAIL_MAX_S:
        return None
    near = np.flatnonzero((t[: k0 + 1] >= t[k0] - 1.0) & (v[: k0 + 1] >= ref - TAIL_TOL_V))
    k = int(near[-1]) if near.size else k0
    span = max(t_end - float(t[k]), 1e-6)
    return {"t_start": float(t[k]), "v_before": float(v[k]), "v_end": float(v[-1]), "ref": ref,
            "tail_len_s": float(t_end - t[k]), "slope_v_s": float((v[-1] - v[k]) / span)}


def _cells(ctx: Ctx, t_upto: np.ndarray, v_upto: np.ndarray) -> tuple[int, str, float | None]:
    """電池串數:round(MOT_BAT_VOLT_MAX / 4.2)(規格);沒設 → ⌈到上鎖為止 1 秒中位數最高電壓 / 4.35⌉(估計值只會少不會多)。

    用 1 秒中位數而不是單點最大值:一個向上跳動的壞樣本不能讓串數多一。回傳 (串數, 來源, 估計用的最高電壓)。
    """
    vmax = ctx.param("MOT_BAT_VOLT_MAX", 0.0) or 0.0
    if vmax > 1.0:
        return max(1, int(round(vmax / CELL_V_MAX))), "param", None
    med = _window_median(t_upto, v_upto, V1_WIN_S)
    if np.isfinite(med).any():
        top = float(np.nanmax(med))
    else:
        top = float(np.max(v_upto)) if v_upto.size else 0.0
    return max(1, int(math.ceil(top / CELL_V_REST_MAX - CELL_EST_EPS))), "estimated", top


def _t_ref(a: Anchors, iv: ArmedInterval) -> float:
    """事件參考時間:t_detect → t_disarm → 區間結束;並限制在解鎖區間內。"""
    for x in (a.t_detect, a.t_disarm):
        if x is not None:
            return float(min(max(x, iv.t0), iv.t1))
    return float(iv.t1)


def _after_loss(a: Anchors, t: float | None) -> bool | None:
    if t is None or a.t_loss is None:
        return None
    return bool(t >= a.t_loss)


def _late_zh(a: Anchors, t: float | None, why: str) -> str:
    """條件在姿態失控(t_loss)之後才開始 → 後果的說法(取代「可能導致失控」);否則回空字串。"""
    if not _after_loss(a, t):
        return ""
    return f"發生在姿態失控({a.t_loss:.1f} s)之後,屬後果({why}),不是失控原因。"


def _usb_fraction(ctx: Ctx, t0: float, t1: float) -> float | None:
    """[t0, t1] 內 POWR.Flags 顯示「USB 供電且沒有磚」的樣本比例;沒有 POWR → None。"""
    pw = ctx.table("POWR")
    if pw is None or "Flags" not in pw:
        return None
    m = (pw["t"] >= t0) & (pw["t"] <= t1)
    if not m.any():
        return None
    fl = pw["Flags"][m].astype(np.int64)
    return float(np.mean(((fl & FLAG_USB) > 0) & ((fl & FLAG_BRICK) == 0)))


def _not_armed(ctx: Ctx, det_id: str, with_tail: bool = False) -> Finding:
    """從未解鎖 → NA 'not_armed';另附整份 log 的供電來源(驗收 C:117/122/123 是 USB 供電 → 電池分析不適用)
    與尾端拔電池(118/120:50 → 17..21 V 是拔電池,不是電壓崩落)。只當說明,狀態一律 NA,不做任何判定。"""
    text = _NA_TEXT["not_armed"]
    tb = ctx.table("BAT")
    if tb is None or "Volt" not in tb:
        return _na(det_id, "not_armed", text)
    t = np.asarray(tb["t"], dtype=float)
    v = np.asarray(tb["Volt"], dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if t.size < MIN_ARMED_SAMPLES:
        return _na(det_id, "not_armed", text)
    vmed = float(np.median(v))
    usb = _usb_fraction(ctx, float(t[0]), float(t[-1]))
    no_batt = vmed < BATTERY_PRESENT_V or (usb is not None and usb > 0.5)
    vals: dict = {"power_source": "usb_or_no_battery" if no_batt else "battery",
                  "bat_volt_median": _r(vmed, 2), "usb_fraction": _r(usb, 2)}
    if no_batt:
        text = (f"未解鎖(桌上 / 手持 log),而且是 USB 供電 / 未接飛行電池(BAT.Volt 中位數 {vmed:.2f} V),"
                f"電池分析不適用。")
    elif with_tail:
        tail = unplug_tail(t, v)
        vals["log_tail"] = None
        if tail is not None:
            vals["log_tail"] = {"t_start": _r(tail["t_start"], 2), "v_before": _r(tail["v_before"], 2),
                                "v_end": _r(tail["v_end"], 2), "tail_len_s": _r(tail["tail_len_s"], 2),
                                "slope_v_s": _r(tail["slope_v_s"], 1), "armed_at_tail": False}
            text += (f"log 尾端 {tail['t_start']:.1f} s 電壓在 {tail['tail_len_s']:.1f} 秒內 "
                     f"{tail['v_before']:.1f}→{tail['v_end']:.1f} V 是拔電池,不是電壓崩落。")
    return _na(det_id, "not_armed", text, **vals)


@dataclass
class _Bat:
    """解鎖區間內的電池資料(已過閘門)。"""
    iv: ArmedInterval
    t_all: np.ndarray            # 整份 BAT 時間 / 電壓(相對秒、V;已去 NaN)
    v_all: np.ndarray
    m: np.ndarray                # t_all 中屬於解鎖區間的遮罩
    cells: int
    cells_src: str
    cells_top_v: float | None    # 估串數用的最高 1 秒中位數電壓(cells_src='estimated' 才有)
    v_full: float                # cells × 4.2:電壓比例判斷的參考滿電壓
    t_ref: float

    @property
    def t(self) -> np.ndarray:
        return self.t_all[self.m]

    @property
    def v(self) -> np.ndarray:
        return self.v_all[self.m]


def _prep_bat(ctx: Ctx, a: Anchors) -> tuple[_Bat | None, str, dict]:
    """閘門:有解鎖區間、有 BAT、夠長、不是 USB 供電。回傳 (_Bat | None, 失敗原因, 附帶數值)。"""
    iv = a.interval
    if iv is None:
        return None, "not_armed", {}
    tb = ctx.table("BAT")
    if tb is None or "Volt" not in tb:
        return None, "no_data", {}
    t_all = np.asarray(tb["t"], dtype=float)
    v_all = np.asarray(tb["Volt"], dtype=float)
    ok = np.isfinite(t_all) & np.isfinite(v_all)
    t_all, v_all = t_all[ok], v_all[ok]
    m = (t_all >= iv.t0) & (t_all <= iv.t1)
    n = int(m.sum())
    if n < MIN_ARMED_SAMPLES or (t_all[m][-1] - t_all[m][0]) < MIN_ARMED_S:
        return None, "too_short", {"armed_samples": n}
    vmed = float(np.median(v_all[m]))
    usb = _usb_fraction(ctx, iv.t0, iv.t1)
    if vmed < BATTERY_PRESENT_V or (usb is not None and usb > 0.5):
        return None, "usb_or_no_battery", {"bat_volt_median": _r(vmed), "usb_fraction": _r(usb, 2)}
    upto = t_all <= iv.t1
    cells, src, top = _cells(ctx, t_all[upto], v_all[upto])
    return _Bat(iv, t_all, v_all, m, cells, src, top, cells * CELL_V_MAX, _t_ref(a, iv)), "", {}


# ---------------------------------------------------------------------------
# 1. power_low_voltage
# ---------------------------------------------------------------------------
@guarded("power_low_voltage", DOMAIN)
def power_low_voltage(ctx: Ctx, a: Anchors):
    """飛行中電池過低。

    crit :1 秒中位數電壓 < MOT_BAT_VOLT_MIN(沒設 → 3.3 V/cell × 串數;串數也是估的 → 門檻是估計值,只給 SOFT)
    high :1 秒中位數持續 ≥ max(BATT_LOW_TIMER, 1) 秒 < BATT_CRT_VOLT(BATT_CRT_VOLT ≥ BATT_LOW_VOLT 的顛倒設定只給 SOFT)
    err6 :解鎖期間有 ERR 子系統 6(飛控電池失效保護)ECode ≠ 0
    只用電壓(BAT.Curr 無效);log 尾端拔電池的樣本先裁掉。條件在 t_loss 之後才開始 → 文字寫成後果,不說「可能導致失控」。
    實測 124:飛行 1 V1 最低 46.87 V、飛行 2 44.76 V(瞬間 42.38 V)→ 對 39.6 V 餘裕 5.2 V、對 42.0 V 餘裕 2.8 V。
    """
    det = "power_low_voltage"
    if a.interval is None:
        return _not_armed(ctx, det)
    fl, why, extra = _prep_bat(ctx, a)
    if fl is None:
        return _na_reason(det, why, **extra)
    iv = fl.iv
    t, v = fl.t, fl.v

    # 解鎖中才有的尾端(解鎖中拔電池):裁掉,由 power_voltage_collapse 負責
    tail = unplug_tail(fl.t_all, fl.v_all)
    trimmed = False
    if tail is not None and iv.t0 <= tail["t_start"] <= iv.t1 + 0.5:
        keep = t < tail["t_start"] - TAIL_TRIM_S
        trimmed = bool((~keep).any())
        t, v = t[keep], v[keep]
    if t.size < MIN_ARMED_SAMPLES:
        return _na_reason(det, "too_short", armed_samples=int(t.size))

    v1 = _window_median(t, v, V1_WIN_S)
    fin = np.isfinite(v1)
    if not fin.any():
        return _na_reason(det, "too_short", armed_samples=int(t.size))

    vmax_cfg = ctx.param("MOT_BAT_VOLT_MAX", 0.0) or 0.0
    vmin_cfg = ctx.param("MOT_BAT_VOLT_MIN", 0.0) or 0.0
    vmin_src = "param"
    if vmin_cfg <= 0.0:
        vmin_cfg, vmin_src = CELL_V_CRIT * fl.cells, "3.3V*cells"
    est = vmin_src != "param"                    # 門檻本身是估的(串數由電壓推估)→ crit 最多 SOFT
    cells_est = fl.cells_src != "param"
    if est:
        crit_name = f"估計門檻 {vmin_cfg:.1f} V(MOT_BAT_VOLT_MIN 未設定:估計 {fl.cells} 串 × {CELL_V_CRIT} V/cell)"
    else:
        crit_name = f"MOT_BAT_VOLT_MIN {vmin_cfg:.1f} V"
    per_cell_note = f"(估計 {fl.cells} 串)" if cells_est and not est else ""
    crt = ctx.param("BATT_CRT_VOLT", 0.0) or 0.0
    low = ctx.param("BATT_LOW_VOLT", 0.0) or 0.0
    timer = max(float(ctx.param("BATT_LOW_TIMER", 10.0) or 0.0), LOW_TIMER_MIN_S)
    cfg_inverted = bool(crt > 0 and low > 0 and crt >= low)

    crit_mask = fin & (v1 < vmin_cfg)
    crit_runs = runs(crit_mask, t, 0.0, RUN_MAX_GAP_S)
    crit = bool(crit_runs)
    high_runs = []
    if crt > 0:
        high_runs = runs(fin & (v1 < crt), t, timer, RUN_MAX_GAP_S)
    high = bool(high_runs)

    # 飛行中飛控自己的電池失效保護(ERR 子系統 6)
    et, es, ec = ctx.err_rows()
    e6 = (es == 6) & (ec != 0) & (et >= iv.t0) & (et <= iv.t1 + 0.1)
    err6_t = [float(x) for x in et[e6]]

    # 統計:整段解鎖 + 事件前 30 秒
    i_min = int(np.nanargmin(np.where(fin, v1, np.inf)))
    v1_min, t_v1_min = float(v1[i_min]), float(t[i_min])
    i_inst = int(np.argmin(v))
    vinst_min, t_vinst = float(v[i_inst]), float(t[i_inst])
    w0 = max(iv.t0, fl.t_ref - LAST_WINDOW_S)
    mw = (t >= w0) & (t <= fl.t_ref) & fin
    v1_min30 = float(np.min(v1[mw])) if mw.any() else None
    mw_raw = (t >= w0) & (t <= fl.t_ref)
    vinst30 = float(np.min(v[mw_raw])) if mw_raw.any() else None
    t_inst30 = float(t[mw_raw][np.argmin(v[mw_raw])]) if mw_raw.any() else None

    vals: dict = {
        "v1_min": _r(v1_min, 2), "v1_min_t": _r(t_v1_min, 2), "v1_min_per_cell": _r(v1_min / fl.cells, 3),
        "vinst_min": _r(vinst_min, 2), "vinst_min_t": _r(t_vinst, 2), "vinst_min_per_cell": _r(vinst_min / fl.cells, 3),
        "v1_min_30s": _r(v1_min30, 2), "vinst_min_30s": _r(vinst30, 2), "vinst_min_30s_t": _r(t_inst30, 2),
        "window_30s": [_r(w0, 2), _r(fl.t_ref, 2)],
        "v_crit_cfg": _r(vmin_cfg, 2), "v_crit_cfg_src": vmin_src, "margin_to_crit_v": _r(v1_min - vmin_cfg, 2),
        "batt_crt_volt": _r(crt, 2) if crt > 0 else None, "batt_low_volt": _r(low, 2) if low > 0 else None,
        "margin_to_crt_v": _r(v1_min - crt, 2) if crt > 0 else None,
        "batt_low_timer_s": _r(timer, 1), "config_inverted": cfg_inverted,
        "cells": fl.cells, "cells_src": fl.cells_src, "cells_top_v": _r(fl.cells_top_v, 2), "v_crit_estimated": est,
        "crit": crit, "high": high, "err6_count": len(err6_t),
        "tail_trimmed": trimmed, "armed_s": _r(iv.t1 - iv.t0, 1),
        "min_after_loss": _after_loss(a, t_vinst),
    }
    mb = ctx.table("MOTB")
    if mb is not None and "BatVolt" in mb and vmax_cfg > 1.0:
        mm = (mb["t"] >= w0) & (mb["t"] <= fl.t_ref)
        if mm.any():
            vals["motb_batvolt_min_v_30s"] = _r(float(np.min(mb["BatVolt"][mm])) * vmax_cfg, 2)
        if "LiftMax" in mb and mm.any():
            vals["lift_max_min_30s"] = _r(float(np.min(mb["LiftMax"][mm])), 3)

    cands: list[tuple[float, float, str, str]] = []      # (t_onset, t_alarm, 狀態, 說明)
    late_why = "翻滾 / 撞擊中油門與馬達劇烈變化造成的下陷"
    if crit:
        t_c = crit_runs[0][0]
        late = _late_zh(a, t_c, late_why)
        why = late or ("電量耗盡或大電流下陷使推力補償不足,可能導致失控,請檢查電池與放電狀態。" if not est else
                       "電量可能耗盡;但串數由電壓推估、門檻不是設定值,證據力較弱,請以實際電池串數核對。")
        cands.append((t_c, t_c, SOFT if est else HIT,
                      f"解鎖期間電池 1 秒中位數電壓在 {t_c:.1f} s 降到 {v1_min:.1f} V({v1_min / fl.cells:.2f} V/cell{per_cell_note}),"
                      f"低於 {crit_name};{why}"))
    if err6_t:
        late = _late_zh(a, err6_t[0], late_why)
        cands.append((err6_t[0], err6_t[0], HIT, f"飛行中 {err6_t[0]:.1f} s 飛控觸發電池失效保護(ERR 子系統 6)" + (f",{late}" if late else "。")))
    if high and (not crit or est):
        t_h, t_h_end = high_runs[0][0], high_runs[0][1]
        note = "(但 BATT_CRT_VOLT ≥ BATT_LOW_VOLT,參數順序顛倒,證據力較弱)" if cfg_inverted else ""
        late = _late_zh(a, t_h, late_why)
        cands.append((t_h, t_h + timer, SOFT if cfg_inverted else HIT,
                      f"解鎖期間電池電壓自 {t_h:.1f} s 起連續 {t_h_end - t_h:.0f} s 低於 BATT_CRT_VOLT {crt:.1f} V"
                      f"(1 秒中位數最低 {v1_min:.1f} V,{v1_min / fl.cells:.2f} V/cell{per_cell_note}){note},"
                      + (late or "電量接近耗盡。")))
    if cands:
        status = HIT if any(c[2] == HIT for c in cands) else SOFT
        basis = [c for c in cands if c[2] == status]     # 時間取自撐起這個狀態的條件(估計門檻的 SOFT 不能把 HIT 的起點提前)
        t_on = min(c[0] for c in basis)
        vals["conditions"] = [n for n, c in (("critical_low", crit), ("critical_timer", high), ("err6", bool(err6_t))) if c]
        vals["after_loss"] = _after_loss(a, t_on)
        return Finding(det, DOMAIN, status, SEV[det], t_onset=_r(t_on, 3), t_alarm=_r(min(c[1] for c in basis), 3),
                       context="air", values=vals, text_zh="".join(c[3] for c in cands))

    late = ""
    if vals["min_after_loss"]:
        late = f"(瞬間最低 {vinst_min:.1f} V 出現在姿態失控之後,屬後果)"
    text = (f"解鎖期間電池 1 秒中位數電壓最低 {v1_min:.1f} V({v1_min / fl.cells:.2f} V/cell{per_cell_note}),"
            f"高於 {crit_name} 達 {v1_min - vmin_cfg:.1f} V")
    if crt > 0:
        text += f"、高於 BATT_CRT_VOLT {crt:.1f} V 達 {v1_min - crt:.1f} V"
    text += f"{late},且沒有電池失效保護紀錄;依電壓判斷可排除電池耗盡"
    text += "(門檻由估計串數推得,串數若低估則門檻偏寬鬆)。" if est else "。"
    return Finding(det, DOMAIN, CLEAR, "info", context="air", values=vals, text_zh=text)


# ---------------------------------------------------------------------------
# 2. power_voltage_collapse
# ---------------------------------------------------------------------------
def _on_ground(ctx: Ctx, iv: ArmedInterval, t0: float, t1: float) -> bool | None:
    """[t0, t1] 這段飛機是否在地面:CTUN.ThO 中位數 ≤ 0.03;沒有 CTUN 才用 EV 落地狀態(且 log 有 EV 28);都沒有 → None。"""
    tb = ctx.table("CTUN")
    if tb is not None and "ThO" in tb:
        m = (tb["t"] >= t0) & (tb["t"] <= t1)
        if int(m.sum()) >= 3:
            return bool(np.median(tb["ThO"][m]) <= ON_GROUND_THO)
    try:
        if ctx.ev_times(28, iv.t0, t1):
            return ctx.landed_at(t1, iv) is True
    except Exception:  # noqa: BLE001
        pass
    return None


@guarded("power_voltage_collapse", DOMAIN)
def power_voltage_collapse(ctx: Ctx, a: Anchors):
    """解鎖中電源瞬間中斷:1 秒內掉壓 ≥ 20% 滿電壓(12S ≈ 10 V)。

    - 穩健的 1 秒壓降(見 COLLAPSE_* 常數):參考 = 1 秒前的中位數水準(取 1 秒窗與 5 秒窗較低者),
      掉壓要在之後 0.3 s(至少下一筆)都維持 → HIT;只有單點跨線、沒有持續 → SOFT 'transient'(壞樣本或瞬間接觸不良);
      參考窗內樣本太少(BAT 空洞)就不比較;
    - 解鎖中 log 就結束且尾端是拔電池曲線 = 「飛行中斷電導致 log 結束」(log_end_in_flight);
    - log 在解鎖中結束在掉壓上、且掉壓當下油門已在最低(CTUN.ThO ≤ 0.03)= 落地後解鎖狀態下拔電池,回 CLEAR
      (context='unplug_on_ground');log 之後還繼續記錄的掉壓(例如翻滾中電池脫落)不能這樣解釋 → HIT context='on_ground',
      由 engine 依時間貼先兆 / 後果;
    - 解鎖結束之後的尾端(124:上鎖後 27 秒)只記在 values['log_tail'],不算。
    實測:解鎖中最大單點壓降 3.68 V、持續壓降 3.12 V;拔電池尾端 -26..-33 V;沒有真實飛行中斷電樣本(召回未驗證)。
    """
    det = "power_voltage_collapse"
    if a.interval is None:
        return _not_armed(ctx, det, with_tail=True)
    fl, why, extra = _prep_bat(ctx, a)
    if fl is None:
        return _na_reason(det, why, **extra)
    iv = fl.iv
    t_all, v_all = fl.t_all, fl.v_all
    n = t_all.size
    ia = np.flatnonzero(fl.m)
    ta = t_all[ia]
    thr = COLLAPSE_DROP_FRAC * fl.v_full

    ref_hi = np.searchsorted(t_all, ta - COLLAPSE_LAG_S + COLLAPSE_PRE_HALF_S, side="right")
    pre, n_pre = _reduce_idx(v_all, np.searchsorted(t_all, ta - COLLAPSE_LAG_S - COLLAPSE_PRE_HALF_S, side="left"), ref_hi)
    pre[n_pre < COLLAPSE_PRE_MIN_N] = np.nan
    lng, n_lng = _reduce_idx(v_all, np.searchsorted(t_all, ta - COLLAPSE_LONG_S, side="left"), ref_hi)
    lng[n_lng < V1_MIN_SAMPLES] = np.nan
    ref = np.where(np.isfinite(pre), np.fmin(pre, lng), np.nan)     # 沒有 1 秒前的參考(空洞)就不比較
    post_hi = np.maximum(np.searchsorted(t_all, ta + COLLAPSE_PERSIST_S, side="right"), np.minimum(ia + 2, n))
    post, _ = _reduce_idx(v_all, ia, post_hi, "max")                 # 之後 0.3 s(至少下一筆)內的最高電壓
    inst = v_all[ia] - ref
    pers = post - ref
    if not np.isfinite(inst).any():
        return _na_reason(det, "too_short", armed_samples=int(ia.size))

    tail = unplug_tail(t_all, v_all)
    tail_in_armed = tail is not None and iv.t0 <= tail["t_start"] <= iv.t1 + 0.5
    k_worst = int(np.nanargmin(inst))
    worst = max(0.0, float(-inst[k_worst]))                         # 電壓只升不降時壓降記 0(不是負數)
    viol_p = np.flatnonzero(pers <= -thr)
    viol_i = np.flatnonzero(inst <= -thr)
    vals: dict = {
        "max_drop_1s_v": _r(worst, 2), "max_drop_t": _r(ta[k_worst], 2),
        "max_drop_persist_v": _r(max(0.0, float(-np.nanmin(pers))), 2) if np.isfinite(pers).any() else None,
        "threshold_v": _r(thr, 2), "margin_x": _r(thr / max(worst, 1e-6), 1),
        "drop_frac": COLLAPSE_DROP_FRAC, "lag_s": COLLAPSE_LAG_S, "persist_s": COLLAPSE_PERSIST_S,
        "transient_samples": int(viol_i.size - np.isin(viol_i, viol_p).sum()),
        "cells": fl.cells, "cells_src": fl.cells_src, "log_tail": None, "armed_s": _r(iv.t1 - iv.t0, 1),
    }
    if tail is not None:
        vals["log_tail"] = {"t_start": _r(tail["t_start"], 2), "v_before": _r(tail["v_before"], 2), "v_end": _r(tail["v_end"], 2),
                            "slope_v_s": _r(tail["slope_v_s"], 1), "armed_at_tail": bool(tail_in_armed),
                            "secs_after_disarm": _r(tail["t_start"] - iv.t1, 1),
                            "later_flight_between": any(o.t0 > iv.t1 and o.t0 < tail["t_start"] for o in ctx.armed)}

    def onset(q: int) -> float:
        """起點 = 掉壓樣本之前 1.5 s 內最後一個「離參考電壓不到 1 V」的樣本。"""
        i = int(ia[q])
        seg = np.arange(int(np.searchsorted(t_all, t_all[i] - COLLAPSE_LAG_S - COLLAPSE_PRE_HALF_S, side="left")), i + 1)
        near = seg[v_all[seg] >= ref[q] - COLLAPSE_ONSET_V]
        return float(t_all[int(near[-1]) if near.size else int(seg[0])])

    t_on = size = None
    kind = ""
    if viol_p.size:
        q = int(viol_p[0])
        t_on = onset(q)
        # 掉壓幅度 = 事件起點後 2 秒內最大的持續壓降(第一個超過門檻的樣本只是剛跨線)
        ev_win = (ta >= ta[q]) & (ta <= ta[q] + 2 * COLLAPSE_LAG_S) & np.isfinite(pers)
        size = float(-np.min(pers[ev_win]))
        kind = "log_end_in_flight" if (tail_in_armed and abs(tail["t_start"] - t_on) <= 2 * COLLAPSE_LAG_S) else "in_flight"
    elif tail_in_armed:
        # 取樣太稀疏、1 秒滯後比較沒抓到,但解鎖中 log 就結束在拔電池曲線上
        t_on, size = float(tail["t_start"]), float(tail["v_before"] - tail["v_end"])
        kind = "log_end_in_flight"
    elif viol_i.size:
        # 只有單點跨線、0.3 秒內就恢復:可能是壞樣本,也可能是瞬間接觸不良 → 弱證據
        q = int(viol_i[0])
        t_q = onset(q)
        late = _late_zh(a, t_q, "撞擊 / 翻滾造成的瞬間接觸不良")
        vals.update({"onset_t": _r(t_q, 2), "drop_v": _r(-inst[q], 2), "kind": "transient", "after_loss": _after_loss(a, t_q),
                     "on_ground": _on_ground(ctx, iv, t_q - ON_GROUND_LOOKBACK_S, t_q)})
        text = (f"解鎖中的 {t_q:.1f} s 有 {vals['transient_samples']} 個電壓樣本比 1 秒前低 {-inst[q]:.1f} V(門檻 {thr:.1f} V),"
                f"但 {COLLAPSE_PERSIST_S} 秒內就恢復,沒有持續的掉壓;可能是感測雜訊,也可能是瞬間接觸不良,證據力弱。"
                + (late or "若反覆出現請檢查電池接頭與電源模組。"))
        return Finding(det, DOMAIN, SOFT, SEV[det], t_onset=_r(t_q, 3), t_alarm=_r(t_q, 3), context="transient", values=vals, text_zh=text)

    if t_on is None:
        text = (f"解鎖期間電壓沒有突然崩落:最大 1 秒壓降 {worst:.2f} V(出現在 {ta[k_worst]:.1f} s,多為起飛 / 大油門的正常下陷),"
                f"遠小於判定門檻 {thr:.1f} V。")
        later_flight = tail is not None and any(o.t0 > iv.t1 and o.t0 < tail["t_start"] for o in ctx.armed)
        if tail is not None and not tail_in_armed and not later_flight:
            text += (f"log 尾端 {tail['t_start']:.0f} s 的大幅掉壓({tail['v_before']:.0f}→{tail['v_end']:.0f} V)發生在"
                     f"{'解鎖結束後 %.0f 秒' % (tail['t_start'] - iv.t1) if tail['t_start'] > iv.t1 else '解鎖區間之外'},是拔電池,不算飛行中掉電。")
        return Finding(det, DOMAIN, CLEAR, "info", context="armed", values=vals, text_zh=text)

    on_ground = _on_ground(ctx, iv, t_on - ON_GROUND_LOOKBACK_S, t_on)
    vals.update({"onset_t": _r(t_on, 2), "drop_v": _r(size, 2), "on_ground": on_ground, "kind": kind,
                 "after_loss": _after_loss(a, t_on)})
    if on_ground and kind == "log_end_in_flight":
        # 只有「log 在解鎖中結束在這個掉壓上」才能解釋成落地後拔電池(之後沒有任何資料可反證)
        text = (f"{t_on:.1f} s 電壓在 1 秒內掉了 {size:.1f} V,log 隨即結束,但當時油門已在最低(CTUN.ThO ≤ {ON_GROUND_THO}),"
                f"是落地後在解鎖狀態下拔電池,不是飛行中掉電。")
        return Finding(det, DOMAIN, CLEAR, "info", t_onset=_r(t_on, 3), context="unplug_on_ground", values=vals, text_zh=text)
    late = _late_zh(a, t_on, "撞擊 / 翻滾造成電池接頭鬆脫或電池脫落")
    if on_ground:
        kind = vals["kind"] = "on_ground"
        text = (f"解鎖中的 {t_on:.1f} s 電壓在 1 秒內崩落 {size:.1f} V(門檻 {thr:.1f} V),當時油門在最低(在地面),"
                f"之後 log 仍繼續記錄,不能解釋成人員拔電池;" + (late or "可能是接頭鬆脫、焊點斷裂或電源模組斷電,請檢查電池接頭與電源模組。"))
    elif kind == "log_end_in_flight":
        text = (f"解鎖中的 {t_on:.1f} s 電壓在 1 秒內崩落 {size:.1f} V(門檻 {thr:.1f} V),log 隨後在解鎖狀態下結束,油門非最低;"
                + (late or "判定為飛行中電源中斷(接頭鬆脫、焊點斷裂或電源模組斷電),請檢查電池接頭與電源模組。"))
    else:
        text = (f"解鎖中的 {t_on:.1f} s 電壓在 1 秒內崩落 {size:.1f} V(門檻 {thr:.1f} V),"
                + (late or "通常是電池接頭鬆脫、焊點斷裂或電源模組斷電,請檢查電池接頭與電源模組。"))
    return Finding(det, DOMAIN, HIT, SEV[det], t_onset=_r(t_on, 3), t_alarm=_r(t_on, 3), context=kind, values=vals, text_zh=text)


# ---------------------------------------------------------------------------
# 3. power_rail_fault
# ---------------------------------------------------------------------------
def _armed(t: np.ndarray, iv: ArmedInterval) -> np.ndarray:
    return (t >= iv.t0) & (t <= iv.t1)


@guarded("power_rail_fault", DOMAIN)
def power_rail_fault(ctx: Ctx, a: Anchors):
    """飛控 / 伺服 / MCU 電源軌異常(解鎖期間)。任一成立 → HIT(values['conditions'] 列出哪幾項):
      vcc_low            POWR.Vcc < 4.5 V
      brick_lost         解鎖期間電源模組(磚)有效旗標掉了(原本大多有效才算;沒接磚的機型不誤報)
      periph_overcurrent 解鎖期間即時旗標 POWR.Flags 的週邊過流位元(8 | 16),或累計旗標 AccFlags 的該位元「第一次出現」在解鎖期間
      mcu_low            MCU.MVolt < 3.1 V
      servo_rail_low     解鎖中 POWR.VServo < 4.5 V(僅在 VServo 平常 ≥ 4.5 V 的機型評估;未解鎖時 0.08..0.27 V 是正常的)
    AccFlags 是開機以來的累計值:解鎖前就已成立的位元不是這次飛行中發生的 → 沒有其他條件時回 SOFT context='pre_arm',
    t_onset = 真正第一次出現的時間(在解鎖之前 → engine 貼 ground_info,不計分)。
    非有限值(NaN)的列不參與判斷。條件在 t_loss 之後才開始 → 文字寫成後果。
    實測 124 全程:Vcc 最低 5.182 V、MCU 3.289..3.294 V、VServo 解鎖中最低 4.855 V,Flags/AccFlags 恆為 1。
    與規格差異:不要求 brick 旗標才檢查 Vcc(解鎖中電池供電本來就該有磚;USB 供電由閘門排除)。
    """
    det = "power_rail_fault"
    iv = a.interval
    if iv is None:
        return _not_armed(ctx, det)
    pw = ctx.table("POWR")
    if pw is None or "Vcc" not in pw:
        return _na_reason(det, "no_data")
    # 電池閘門:USB 供電 / 沒接電池 → NA(USB 供電時 Vcc 4.65..4.9 V 只高於門檻 0.15..0.4 V,不拿來判斷)
    bt = ctx.table("BAT")
    m_bat = _armed(bt["t"], iv) & np.isfinite(bt["Volt"]) if bt is not None and "Volt" in bt else None
    usb = _usb_fraction(ctx, iv.t0, iv.t1)
    if (m_bat is not None and m_bat.sum() >= 3 and float(np.median(bt["Volt"][m_bat])) < BATTERY_PRESENT_V) or (usb is not None and usb > 0.5):
        return _na_reason(det, "usb_or_no_battery", usb_fraction=_r(usb, 2))
    mp = _armed(pw["t"], iv)
    tp = pw["t"][mp]
    fv = np.isfinite(pw["Vcc"][mp])
    if int(fv.sum()) < MIN_ARMED_SAMPLES:
        return _na_reason(det, "too_short", armed_samples=int(fv.sum()))
    tv, vcc = tp[fv], pw["Vcc"][mp][fv]
    conds: dict[str, float] = {}      # 條件 → 最早成立時間

    def _first(mask: np.ndarray, tt: np.ndarray) -> float | None:
        idx = np.flatnonzero(mask)
        return float(tt[idx[0]]) if idx.size else None

    def _flags(name: str):
        """旗標欄位(解鎖區間內、只取有限值的列)→ (int64 陣列, 時間) 或 (None, None)。"""
        if name not in pw:
            return None, None
        x = pw[name][mp]
        ok = np.isfinite(x)
        return x[ok].astype(np.int64), tp[ok]

    x = _first(vcc < RAIL_VCC_MIN_V, tv)
    if x is not None:
        conds["vcc_low"] = x
    flags, tf = _flags("Flags")
    acc, _ = _flags("AccFlags")
    brick_frac = None
    if flags is not None and flags.size:
        brick = (flags & FLAG_BRICK) > 0
        brick_frac = float(brick.mean())
        if brick_frac >= RAIL_BRICK_PRESENT_FRAC:
            x = _first(~brick, tf)
            if x is not None:
                conds["brick_lost"] = x
    # 週邊過流:即時旗標看解鎖期間;累計旗標逐位元找整份 log(到上鎖為止)第一次出現的時間
    oc_first = []
    if flags is not None:
        x = _first((flags & RAIL_OVERCURRENT_MASK) > 0, tf)
        if x is not None:
            oc_first.append(x)
    acc_pre: list[dict] = []          # 解鎖前就已成立的累計位元
    if "AccFlags" in pw:
        fa = pw["AccFlags"]
        ok = np.isfinite(fa) & (pw["t"] <= iv.t1)
        ta_all, fa_all = pw["t"][ok], fa[ok].astype(np.int64)
        t_first_armed = float(tp[0]) if tp.size else iv.t0
        for bit in RAIL_OVERCURRENT_BITS:
            x = _first((fa_all & bit) > 0, ta_all)
            if x is None:
                continue
            if x < t_first_armed:
                acc_pre.append({"bit": bit, "t_first": _r(x, 3)})
            else:
                oc_first.append(x)
    if oc_first:
        conds["periph_overcurrent"] = min(oc_first)
    vs_min = vs_med = None
    if "VServo" in pw:
        vs = pw["VServo"][mp]
        fs = np.isfinite(vs)
        if fs.any():
            vs, tvs = vs[fs], tp[fs]
            vs_med = float(np.median(vs))
            vs_min = float(np.min(vs))
            if vs_med >= RAIL_VSERVO_VALID_V:
                x = _first(vs < RAIL_VSERVO_MIN_V, tvs)
                if x is not None:
                    conds["servo_rail_low"] = x
    mv_min = mv_max = None
    mc = ctx.table("MCU")
    if mc is not None and "MVolt" in mc:
        mm = _armed(mc["t"], iv) & np.isfinite(mc["MVolt"])
        if mm.any():
            mv = mc["MVolt"][mm]
            mv_min, mv_max = float(np.min(mv)), float(np.max(mv))
            x = _first(mv < RAIL_MCU_MIN_V, mc["t"][mm])
            if x is not None:
                conds["mcu_low"] = x

    i_vmin = int(np.argmin(vcc))
    t_vmin = float(tv[i_vmin])
    prior = (tv >= t_vmin - RAIL_DIP_WIN_S) & (tv <= t_vmin)
    vcc_dip = float(vcc[i_vmin] - np.median(vcc[prior])) if prior.any() else None
    vals: dict = {
        "vcc_min": _r(vcc[i_vmin], 3), "vcc_min_t": _r(t_vmin, 2), "vcc_median": _r(np.median(vcc), 3), "vcc_dip_v": _r(vcc_dip, 3),
        "vcc_threshold": RAIL_VCC_MIN_V, "vcc_margin_v": _r(vcc[i_vmin] - RAIL_VCC_MIN_V, 3),
        "vservo_min": _r(vs_min, 3), "vservo_median": _r(vs_med, 3), "vservo_evaluated": bool(vs_med is not None and vs_med >= RAIL_VSERVO_VALID_V),
        "vservo_threshold": RAIL_VSERVO_MIN_V,
        "mcu_volt_min": _r(mv_min, 3), "mcu_volt_max": _r(mv_max, 3), "mcu_threshold": RAIL_MCU_MIN_V,
        "brick_fraction": _r(brick_frac, 3), "acc_flags_max": int(acc.max()) if acc is not None and acc.size else None,
        "acc_overcurrent_pre_arm": acc_pre,
        "conditions": sorted(conds, key=conds.get), "vcc_min_after_loss": _after_loss(a, t_vmin),
    }
    pre_note = ""
    t_pre = None
    if acc_pre:
        t_pre = min(d["t_first"] for d in acc_pre)
        pre_note = (f"週邊過流累計旗標(POWR.AccFlags)在解鎖前的 {t_pre:.1f} s 就已成立(開機後累計、不會自動清除),"
                    f"不是這次飛行中發生的")
    if conds:
        t_on = min(conds.values())
        names = {"vcc_low": f"飛控 5V 軌 Vcc 低至 {vcc[i_vmin]:.2f} V(門檻 {RAIL_VCC_MIN_V} V)",
                 "brick_lost": "電源模組(磚)在解鎖中失效", "periph_overcurrent": "週邊過流旗標在解鎖期間成立",
                 "mcu_low": f"MCU 3.3V 軌低至 {mv_min:.2f} V" if mv_min is not None else "MCU 3.3V 軌掉電",
                 "servo_rail_low": f"伺服軌低至 {vs_min:.2f} V" if vs_min is not None else "伺服軌掉電"}
        what = "、".join(names[c] for c in vals["conditions"])
        vals["after_loss"] = _after_loss(a, t_on)
        late = _late_zh(a, t_on, "撞擊 / 翻滾造成的供電擾動")
        text = (f"解鎖期間電源軌異常({what});首次出現在 {t_on:.1f} s,"
                + (late or "可能造成飛控重啟或感測器 / 馬達供電不穩,請檢查電源模組、接頭與週邊供電。"))
        if pre_note:
            text += f"另外{pre_note}。"
        return Finding(det, DOMAIN, HIT, SEV[det], t_onset=_r(t_on, 3), t_alarm=_r(t_on, 3), context="armed", values=vals, text_zh=text)
    base = f"解鎖期間飛控 5V 軌 Vcc 最低 {vcc[i_vmin]:.2f} V(門檻 {RAIL_VCC_MIN_V} V)"
    if mv_min is not None:
        base += f"、MCU 3.3V 軌 {mv_min:.3f}–{mv_max:.3f} V"
    if vs_min is not None and vals["vservo_evaluated"]:
        base += f"、伺服軌最低 {vs_min:.2f} V"
    if t_pre is not None:
        # 解鎖前的累計過流:真實的硬體事件,但不在飛行中 → 弱證據,時間用真正第一次出現的時間(engine 會標成地面資訊)
        text = (f"{pre_note};解鎖期間即時過流旗標沒有成立,{base},沒有飛行中的電源軌掉電跡象。"
                "週邊曾經過流,建議檢查週邊供電與接線。")
        return Finding(det, DOMAIN, SOFT, SEV[det], t_onset=_r(t_pre, 3), t_alarm=_r(t_pre, 3), context="pre_arm", values=vals, text_zh=text)
    # 只說真的檢查過的項目:沒有磚旗標的機型不能說「電源模組一直有效」
    checked = []
    if brick_frac is not None:
        checked.append("電源模組(磚)一直有效" if brick_frac >= RAIL_BRICK_PRESENT_FRAC else "這架機沒有電源模組有效旗標(未評估磚失效)")
    if flags is not None or acc is not None:
        checked.append("無週邊過流旗標")
    text = base + ("," + "、".join(checked) if checked else "") + ",沒有電源軌掉電跡象。"
    return Finding(det, DOMAIN, CLEAR, "info", context="armed", values=vals, text_zh=text)


# ---------------------------------------------------------------------------
# 4. power_data_quality(一般健檢)
# ---------------------------------------------------------------------------
def _hover_throttle(ctx: Ctx, iv: ArmedInterval) -> tuple[float, str]:
    """懸停油門:MOT_THST_HOVER(規格);沒設 → 解鎖期間 CTUN.ThH(飛控即時學到的懸停油門)中位數;都沒有 → 0.35 預設。
    (ArduPilot 預設 0.35 對 124 這種 0.15 懸停的機型太高,會把整段懸停排除掉 → 只當最後備援。)"""
    p = ctx.param("MOT_THST_HOVER", None)
    if p is not None and float(p) > 0.0:
        return float(p), "param"
    ct = ctx.table("CTUN")
    if ct is not None and "ThH" in ct:
        m = _armed(ct["t"], iv) & np.isfinite(ct["ThH"]) & (ct["ThH"] > 0.0)
        if int(m.sum()) >= 10:
            return float(np.median(ct["ThH"][m])), "CTUN.ThH"
    return DQ_HOVER_DEFAULT, "default"


@guarded("power_data_quality", DOMAIN)
def power_data_quality(ctx: Ctx, a: Anchors):
    """電流計(BAT.Curr)資料品質。懸停(解鎖且 MOTB.ThrOut > 0.5×MOT_THST_HOVER)≥ 20 秒才評估:

      HIT :電流中位數 < 1.0 A 且 起飛後電壓比解鎖前靜止電壓下陷 ≥ 1.5 V(12S 六軸懸停不可能 < 1 A)
      SOFT:電流與油門呈負相關(corr < 0)但另一條件沒成立
      CLEAR:電流中位數 ≥ 1 A 且與油門不呈負相關
    HIT 時 values['do_not_use'] 列出不可使用的欄位:報告不得談內阻 / 剩餘電量 / 電池老化。這是資料品質問題,不是失事原因。
    實測 124:懸停中位數 0.0038..0.0051 A(對 1.0 A 門檻 200 倍)、corr(Curr, ThrOut) = -0.84..-0.96、下陷 3.1..3.3 V。
    """
    det = "power_data_quality"
    iv = a.interval
    if iv is None:
        return _not_armed(ctx, det)
    bt = ctx.table("BAT")
    mo = ctx.table("MOTB")
    if bt is None or "Curr" not in bt or "Volt" not in bt or mo is None or "ThrOut" not in mo or mo["t"].size == 0:
        return _na_reason(det, "no_data")
    t = np.asarray(bt["t"], dtype=float)
    volt = np.asarray(bt["Volt"], dtype=float)
    cur = np.asarray(bt["Curr"], dtype=float)
    m = _armed(t, iv) & np.isfinite(volt) & np.isfinite(cur)
    if int(m.sum()) < MIN_ARMED_SAMPLES:
        return _na_reason(det, "too_short", armed_samples=int(m.sum()))
    usb = _usb_fraction(ctx, iv.t0, iv.t1)
    if float(np.median(volt[m])) < BATTERY_PRESENT_V or (usb is not None and usb > 0.5):
        return _na_reason(det, "usb_or_no_battery")
    thr = np.interp(t, mo["t"], mo["ThrOut"])
    hover, hover_src = _hover_throttle(ctx, iv)
    sel = m & (thr > DQ_THR_FRAC * hover)
    dt = float(np.median(np.diff(t[m]))) if m.sum() > 1 else 0.1
    load_s = float(sel.sum()) * dt
    if load_s < DQ_MIN_LOAD_S:
        return _na_reason(det, "no_hover", load_s=_r(load_s, 1))
    i_med = float(np.median(cur[sel]))
    rest = (t >= iv.t0 - DQ_REST_FAR_S) & (t <= iv.t0 - DQ_REST_NEAR_S) & np.isfinite(volt)
    if int(rest.sum()) < 3:
        return _na_reason(det, "no_rest_voltage", i_med_a=_r(i_med, 4), load_s=_r(load_s, 1))
    v_rest = float(np.median(volt[rest]))
    sag = v_rest - float(np.median(volt[sel]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = np.corrcoef(cur[m], thr[m])[0, 1] if m.sum() > 2 and np.std(cur[m]) > 0 and np.std(thr[m]) > 0 else np.nan
    corr = _r(c, 2)
    t_sel = t[sel]
    t_on = float(t_sel[0])
    n_alarm = min(t_sel.size - 1, int(math.ceil(DQ_MIN_LOAD_S / dt)))
    vals = {"i_med_a": _r(i_med, 4), "i_max_a": _r(np.max(cur[m]), 3), "sag_v": _r(sag, 2), "v_rest": _r(v_rest, 2),
            "corr_curr_thr": corr, "load_s": _r(load_s, 1), "mot_thst_hover": _r(hover, 4), "hover_src": hover_src,
            "thresholds": {"curr_max_a": DQ_CURR_MAX_A, "sag_min_v": DQ_SAG_MIN_V, "min_load_s": DQ_MIN_LOAD_S}}
    sev = SEV[det]
    if i_med < DQ_CURR_MAX_A and sag >= DQ_SAG_MIN_V:
        vals.update({"curr_invalid": True, "do_not_use": list(DQ_DO_NOT_USE)})
        text = (f"BAT.Curr 讀值無效:有負載飛行 {load_s:.0f} 秒內電流中位數只有 {i_med:.3f} A,但電壓比解鎖前下陷 {sag:.1f} V,代表電流計沒有量到真實電流。"
                "因此由電流推算的欄位(BAT.CurrTot、EnrgTot、RemPct、Res、VoltR)都不可信,請勿用來判斷電池狀態;這是資料品質問題而非失事原因,建議檢查電流計接線與校正。")
        return Finding(det, DOMAIN, HIT, sev, t_onset=_r(t_on, 3), t_alarm=_r(t_sel[n_alarm], 3), context="standing", values=vals, text_zh=text)
    if corr is not None and corr < 0:
        vals["curr_invalid"] = None
        text = (f"BAT.Curr 與油門呈負相關(r = {corr:.2f}),不像真實電流;懸停中位數 {i_med:.2f} A、電壓下陷 {sag:.1f} V,"
                "證據不足以判定無效,但電流衍生欄位請謹慎使用。")
        return Finding(det, DOMAIN, SOFT, sev, t_onset=_r(t_on, 3), t_alarm=_r(t_sel[n_alarm], 3), context="standing", values=vals, text_zh=text)
    if i_med >= DQ_CURR_MAX_A:
        vals["curr_invalid"] = False
        # 相關係數算不出來(油門或電流是常數)就不談相關性,只說真的檢查過的項目
        rel = f",與油門不呈負相關(r = {corr:.2f})" if corr is not None else "(油門或電流沒有變化,無法計算與油門的相關性)"
        text = f"BAT.Curr 看起來合理:懸停中位數 {i_med:.1f} A{rel},電流計資料可用。"
        return Finding(det, DOMAIN, CLEAR, "info", context="standing", values=vals, text_zh=text)
    return _na(det, "indeterminate",
               f"懸停電流僅 {i_med:.2f} A 但電壓下陷只有 {sag:.1f} V,輕載機型 / 電流計皆有可能,無法判斷電流計是否有效。", **vals)


# ---------------------------------------------------------------------------
# 5. power_failsafe_config_audit(一般健檢,靜態參數稽核)
# ---------------------------------------------------------------------------
def _audit_params(ctx: Ctx) -> dict | None:
    """電池保護參數稽核;一個 BATT_* 保護參數都沒有 → None。"""
    names = ("BATT_LOW_VOLT", "BATT_CRT_VOLT", "BATT_FS_LOW_ACT", "BATT_FS_CRT_ACT", "BATT_LOW_MAH", "BATT_CRT_MAH",
             "BATT_ARM_VOLT", "ARMING_CHECK", "MOT_BAT_VOLT_MAX", "MOT_BAT_VOLT_MIN", "FS_CRASH_CHECK", "FS_EKF_ACTION")
    p = {n: ctx.param(n, None) for n in names}
    if all(p[n] is None for n in names[:7]):
        return None
    f = lambda n, d=0.0: float(p[n]) if p[n] is not None else d       # noqa: E731
    low, crt = f("BATT_LOW_VOLT"), f("BATT_CRT_VOLT")
    fs_low, fs_crt = f("BATT_FS_LOW_ACT"), f("BATT_FS_CRT_ACT")
    mah_low, mah_crt = f("BATT_LOW_MAH"), f("BATT_CRT_MAH")
    arm_v = f("BATT_ARM_VOLT")
    ac = int(f("ARMING_CHECK", 1.0))
    vmax, vmin = f("MOT_BAT_VOLT_MAX"), f("MOT_BAT_VOLT_MIN")
    cells = max(1, int(round(vmax / CELL_V_MAX))) if vmax > 1 else None
    issues: list[str] = []
    if crt > 0 and low > 0 and crt >= low:
        issues.append("crt_ge_low")
    if p["BATT_FS_LOW_ACT"] is not None and p["BATT_FS_CRT_ACT"] is not None and fs_low == 0 and fs_crt == 0:
        issues.append("no_failsafe_action")
    if p["BATT_LOW_MAH"] is not None and p["BATT_CRT_MAH"] is not None and mah_low == 0 and mah_crt == 0:
        issues.append("capacity_failsafe_off")
    if p["ARMING_CHECK"] is not None and arm_v > 0 and (ac & 1) == 0 and (ac & 256) == 0:
        issues.append("batt_arm_volt_not_enforced")
    notes: list[str] = []
    if p["FS_CRASH_CHECK"] is not None and f("FS_CRASH_CHECK") == 0:
        notes.append("crash_check_disabled")
    if p["FS_EKF_ACTION"] is not None and f("FS_EKF_ACTION") == 0:
        notes.append("ekf_failsafe_report_only")
    per = (lambda x: _r(x / cells, 2) if cells and x > 0 else None)      # noqa: E731
    et, es, ec = ctx.err_rows()
    return {"issues": issues, "notes": notes, "cells": cells,
            "values": {"batt_low_volt": _r(low, 2), "batt_crt_volt": _r(crt, 2), "batt_fs_low_act": int(fs_low), "batt_fs_crt_act": int(fs_crt),
                       "batt_low_mah": _r(mah_low, 0), "batt_crt_mah": _r(mah_crt, 0), "batt_arm_volt": _r(arm_v, 2),
                       "arming_check": ac, "mot_bat_volt_max": _r(vmax, 2), "mot_bat_volt_min": _r(vmin, 2),
                       "low_per_cell": per(low), "crt_per_cell": per(crt), "min_per_cell": per(vmin),
                       "fs_crash_check": None if p["FS_CRASH_CHECK"] is None else int(f("FS_CRASH_CHECK")),
                       "fs_ekf_action": None if p["FS_EKF_ACTION"] is None else int(f("FS_EKF_ACTION")),
                       "err_subsys6_count": int(np.sum(es == 6)), "issues": issues, "notes": notes}}


@guarded("power_failsafe_config_audit", DOMAIN)
def power_failsafe_config_audit(ctx: Ctx, a: Anchors):
    """電池保護參數稽核(靜態、每份 log 一次;這批 log 全部一樣 → 無法區分失事與否,只當飛行前衛生項目,不當原因):
      crt_ge_low                  BATT_CRT_VOLT ≥ BATT_LOW_VOLT(兩者都 > 0;順序顛倒)
      no_failsafe_action          BATT_FS_LOW_ACT = BATT_FS_CRT_ACT = 0(電壓過低不會有動作)
      capacity_failsafe_off       BATT_LOW_MAH = BATT_CRT_MAH = 0
      batt_arm_volt_not_enforced  BATT_ARM_VOLT > 0 但 ARMING_CHECK 沒勾 All(bit0)也沒勾電池電量(bit8)
    另記 FS_CRASH_CHECK = 0(關閉失事偵測)與 FS_EKF_ACTION = 0(EKF 失效保護只報告)於 values['notes']。
    從未解鎖的 log 依規格回 NA(靜態稽核結果仍附在 values['static_issues'])。
    """
    det = "power_failsafe_config_audit"
    iv = a.interval
    aud = _audit_params(ctx)
    if iv is None:
        extra = {} if aud is None else {"static_issues": aud["issues"], "static_notes": aud["notes"]}
        return _na_reason(det, "not_armed", **extra)
    if aud is None:
        return _na(det, "no_params", "log 內沒有電池保護參數(BATT_*),無法稽核。")
    vals = aud["values"]
    if aud["issues"]:
        zh = {"crt_ge_low": f"BATT_CRT_VOLT({vals['batt_crt_volt']:.1f} V)不低於 BATT_LOW_VOLT({vals['batt_low_volt']:.1f} V),順序顛倒",
              "no_failsafe_action": "低電壓與嚴重低電壓的失效保護動作皆為 0(電壓過低不會有任何動作)",
              "capacity_failsafe_off": "容量門檻 BATT_LOW_MAH / BATT_CRT_MAH 皆為 0",
              "batt_arm_volt_not_enforced": f"ARMING_CHECK={vals['arming_check']} 未含電池電量檢查,BATT_ARM_VOLT={vals['batt_arm_volt']:.1f} V 不會生效"}
        text = "電池保護參數設定有問題:" + ";".join(zh[k] for k in aud["issues"]) + "。"
        if aud["cells"]:
            text += (f"這屬飛行前應修正的設定問題,與失事原因無直接關係;建議 LOW 約 3.5 V/cell({3.5 * aud['cells']:.1f} V)、"
                     f"CRT 約 3.3 V/cell({3.3 * aud['cells']:.1f} V),失效保護動作設為 Land / RTL。")
        else:
            text += "這屬飛行前應修正的設定問題,與失事原因無直接關係;建議 LOW 約 3.5 V/cell、CRT 約 3.3 V/cell,失效保護動作設為 Land / RTL。"
        return Finding(det, DOMAIN, HIT, SEV[det], t_onset=_r(iv.t0, 3), t_alarm=_r(iv.t0, 3), context="standing", values=vals, text_zh=text)
    return Finding(det, DOMAIN, CLEAR, "info", context="standing", values=vals,
                   text_zh="電池保護參數(低 / 嚴重低電壓門檻、失效保護動作、容量門檻、解鎖電壓檢查)設定一致,沒有發現矛盾。")


# ---------------------------------------------------------------------------
# 模組進入點
# ---------------------------------------------------------------------------
CURR_CAVEAT_ZH = "但 BAT.Curr 電流計無效,電流類故障(電調 / 接頭過流)無法由電流資料排除。"


def run(ctx: Ctx, a: Anchors) -> list[Finding]:
    out: list[Finding] = []
    for fn in (power_low_voltage, power_voltage_collapse, power_rail_fault, power_data_quality, power_failsafe_config_audit):
        out.extend(fn(ctx, a))
    # 驗收 A7:電池「已排除」那一列要附上「電流計無效」的保留 —— 電壓正常只能排除電量耗盡,不能排除電流類故障
    d = {f.id: f for f in out}
    lv, dq = d.get("power_low_voltage"), d.get("power_data_quality")
    if lv is not None and dq is not None and lv.status != NA:
        lv.values["curr_meter_invalid"] = {HIT: True, CLEAR: False}.get(dq.status)
        if lv.status == CLEAR and dq.status == HIT:
            lv.text_zh += CURR_CAVEAT_ZH
    return out
