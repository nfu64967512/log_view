"""導航域(nav)偵測器:EKF 來源組切換 / VIO 資料中斷 / EKF 輔助逾時 / 測試比 / EKF–VIO 不一致 /
速度備援風險 / 切換位置跳變。

七個偵測器(每個每次呼叫至少回傳一筆 Finding;例外由 @guarded 降級成 NA):
  source_switch_audit            切到沒有 GPS 定位的 GPS 組、切換同時 GPS glitch、切到沒有新資料的 VIO 組
  vio_dropout                    VIO(VISP)資料中斷:觸地時中斷 / 使用 VIO 組時中斷 / 串流在上鎖前中止
  ekf_aiding_timeout             EKF 位置 / 速度輔助逾時(XKF4.TS):判斷是肇因還是失事後果
  ekf_testratio_gate             XKF4 SV/SP/SH 測試比是否失控(確認或排除 EKF 發散)
  ekf_vio_disagreement           (常態)EKF 位置與 VIO 位置長期不一致 / VIO 不確定度被鎖在高值
  nav_velocity_gps_fallback_risk (常態)使用中的來源組沒有水平速度來源、GPS 又沒定位 → 無備援
  source_switch_position_jump    來源切換時 EKF 位置跳變量(VIO 相對 GPS 座標的偏差;度量用,從不給 HIT)

在 00000124(唯一一次失事)量到的因果順序:
  觸地 1255.2 → VISP 中斷 1.16 s(1255.13)→ 姿態誤差 >10° 1255.9 → >30° 1258.1 → VISP 最後一筆 1258.146
  → RC9 撥開關切到 GPS 組 1258.86(GPS 已 324 s 無定位 → GPS glitch)→ 切回 VIO 組 1259.66(VISP 已停 1.5 s)
  → 上鎖 1262.1 → EKF 輔助逾時 1265.3 → EKFCHECK/FAILSAFE_EKFINAV 1266.2。
  其中來源切換 / GPS glitch / VIO 失效 / EKF 逾時全部發生在姿態失控「之後」,是後果;本模組只負責把
  「條件開始的時間」(t_onset)量準,因果標籤由 engine 依 t_onset 對 t_warn / t_loss 貼。
  EKF 自己沒有發散(SV/SP/SH ≤ 0.21),真正能在失控前看到的是常態風險:VIO 組沒有速度來源、GPS 無定位、
  VIO 位置不確定度被鎖在高值而 EKF 位置離 VIO 1.6 m,以及觸地時 VIO 中斷 1.16 s。

統計誠實聲明:log 裡只有一次失事(00000124);6 個基準 log(117/118/119/120/122/123)從未解鎖,
對「飛行中」偵測器沒有任何誤報驗證力。真正的陰性對照只有 124 飛行 1 與飛行 2 的 1255 s 之前。
所有門檻都是常數,待更多失事 log 後再調。

資料注意事項(pitfalls):
  - XKFS.SS 是 0-based 的「使用中來源組」(0 = 組 1),不是 XKF4.SS(那是 EKF 狀態位元遮罩);
  - 來源組內容一律讀 EK3_SRCn_*(用 core.ekfsrc 解碼;組 3 不一定是 VIO,117 的 VELXY=3);
  - GPS 沒定位時 Lat/Lng/Spd 是舊值:一律以 Status>=3 且 NSats>=6 遮罩;
  - XKF3.IPN/IPE = EKF 位置 − 量測;沒融合量測時新息「維持最後一個值」(IVN/IVE 凍結在 0.02/0.01);
  - XKF4.TS 的 bit4/5 恆為 1,只看 &3;t<~39 s(絕對輔助尚未建立)不算;
  - VISP / VISV 只存在於 119、120、124;三條 lane(XKF1~4、XKFS)的時間戳與列數都不同,一律依時間對齊。
"""
from __future__ import annotations

import re
import warnings
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from .. import ekfsrc
from .ctx import Ctx
from .guard import guarded
from .types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval, Finding
from .util import runs, thrust_vector_error_deg

DOMAIN = "nav"

DETECTOR_IDS = ("source_switch_audit", "vio_dropout", "ekf_aiding_timeout", "ekf_testratio_gate",
                "ekf_vio_disagreement", "nav_velocity_gps_fallback_risk", "source_switch_position_jump")

# ====================================================================================
# 門檻常數(括號內 = 124 實測的「失事 vs 對照」間距;對照 = 124 飛行 1 與飛行 2 的 1255 s 之前)
# ====================================================================================

# ---- 共通 ----
NAV_MIN_CLEAR_S = 5.0           # 解鎖不到 5 s(= 2.5 × 最長的持續門檻 TS_MIN_DUR_S 2 s)時,沒有證據只能回 NA,不能 CLEAR;
                                # HIT / SOFT 證據照常回報(短時間內的切換 / 中斷仍是真的)

# ---- GPS 可用判定(Status<3 時 Lat/Lng/Spd 是舊值,不能用) ----
GPS_MIN_STATUS = 3
GPS_MIN_SATS = 6
GPS_STALE_S = 1.0               # GPS 約 5 Hz;最後一筆樣本超過 1 s 前視為沒資料

# ---- 來源碼(EK3_SRCn_*) ----
SRC_GPS = 3                     # POSXY / VELXY = 3 → GPS
SRC_EXTNAV = 6                  # POSXY = 6 → ExternalNav(VIO)

# ---- source_switch_audit ----
SW_GPS_WINDOW_S = 1.0           # 切到 GPS 組前 1.0 s 內 GPS 必須全程可用(失事:GPS 已 324 s 無定位;對照 16 次切換中 15 次可用)
SW_GLITCH_WINDOW_S = 1.0        # 切到 GPS 組後 1.0 s 內出現 GPS glitch 邊緣 = 切換造成(失事:0.0 s;其餘 6 個邊緣離切換 ≥196 s)
SW_VIO_AGE_S = 0.5              # 切到 VIO 組時最後一筆 VISP 的年齡(健康切換 ≤0.04 s,失事 0.72 / 1.52 s)
SW_QUALITY_SV = 0.4             # glitch 邊緣前 3 s 的 SV 峰值 ≥0.4 或 SAcc ≥2 m/s → GPS 品質 glitch
SW_QUALITY_SACC = 2.0
SW_SV_LOOKBACK_S = 3.0
SW_RC_LOOKBACK_S = 0.6          # 切換前 0.6 s 內 RC 開關(RCx_OPTION=90)有變化 → 飛手手動切換(RC9 變化 → EV 0.2 s → SS 0.02 s)
GLITCH_BIT = 0x4000             # XKF4.SS bit14 = gps_glitching
RC_LEVEL_LO = 1300.0            # 三段開關:<1300 低(組 1)、<1700 中(組 2)、其餘高(組 3)
RC_LEVEL_HI = 1700.0
RC_OPTION_EKF_SOURCE = 90       # RCx_OPTION = 90 → EKF 來源組切換
GLITCH_ERR_SUBSYS = 11          # ERR Subsys 11 = GPS(ECode 2 = glitch,0 = 恢復)
SW_FOLLOWUP_S = 12.0            # 切換後 12 s 內起算的 XKF4.TS 逾時 = 切換的後續影響(124:1259.66 → 1265.26,5.6 s;只放 values)
_MISSING_FLAGS = ("vio_data_missing", "gps_data_missing")   # 只能說「無法確認」的旗標 → 最多 SOFT

# ---- vio_dropout ----
VIO_GAP_MIN_S = 1.0             # VISP 中斷 ≥1.0 s 才列為 dropout(飛行中正常的小中斷 0.52–1.09 s;失事觸地中斷 1.16 s)
VIO_MODE_AGE_S = 1.5            # 使用 VIO 組時資料年齡 ≥1.5 s(飛行 1/2 在 1258 前最大 1.16 s → 間距僅 0.34 s,信心中等)
VIO_TAIL_S = 1.5                # 最後一筆 VISP 距上鎖 ≥1.5 s 且之後沒恢復 = 串流提前結束(失事 3.98 s)
VIO_TD_WINDOW_S = 0.3           # 中斷起點與觸地撞擊群集起點相差 ≤0.3 s 視為「觸地時中斷」(失事 0.10 s)
VIO_SUBTHRESH_S = 0.5           # 低於 dropout 門檻但 ≥0.5 s 的中斷只放進 values(飛行 1 觸地 0.82 s,GPS 組、無害)
TD_SHOCK_MS2 = 20.0             # 觸地撞擊:IMU |a|>20 m/s²(失事 28.9、飛行 1 25.5;基準 log 最大 16.5)
TD_HARD_MS2 = 45.0              # ≥45 m/s² 是硬撞擊(不是觸地)
TD_MERGE_S = 0.3                # 撞擊群集合併間隔
VIO_MSG_POST_S = 30.0           # 伴隨電腦 VIO 訊息列到上鎖後 30 s(124:'stale' 1259.06 起每 5 s 一次)
VIO_MSG_MAX = 12
VIO_MSG_NOISE = re.compile(r"near-range", re.I)

# ---- ekf_aiding_timeout ----
HPOSABS_BIT = 0x10              # XKF4.SS bit4 = 絕對水平位置輔助;第一次成立(~39 s)之前 TS=50 是正常的
TS_MASK = 0x3                   # XKF4.TS bit0 posTimeout、bit1 velTimeout;bit4/5 恆為 1,要遮掉
TS_MIN_DUR_S = 2.0              # 逾時狀態持續 ≥2 s
TS_POST_WINDOW_S = 60.0         # 上鎖之後 60 s 內才起算的逾時列為「失事後果」
EKFCHECK_PAIR_S = 5.0           # 逾時起點後 5 s 內的 ERR(16,2) EKFCHECK / ERR(17,1) FAILSAFE_EKFINAV 配對
EKFCHECK_SUBSYS = 16
EKFINAV_SUBSYS = 17
MAX_GAP_XKF_S = 1.0             # XKF 資料空洞 >1 s 視為斷開

# ---- ekf_testratio_gate ----
GATE_RATIO = 1.0                # 測試比 ≥1 代表量測被拒絕(失事前 15 s 最大 SV 0.11 / SP 0.01 / SH 0.21)
GATE_MIN_DUR_S = 1.0            # 持續 ≥1.0 s 才算 gate fail(124 唯一一段:753.6–773.9 地面 GPS glitch,SV 3.27)
GATE_MIN_T = 40.0               # t<40 s 絕對輔助尚未建立
GATE_HEALTHY_MAX = 0.5          # 「EKF 健康」判定:SV/SP/SH 皆 <0.5(對照最大 0.34)
GATE_FS_IGNORE = 0x80           # XKF4.FS=128 只出現在每個 log 開頭 ~1 s
GATE_FS_SKIP_S = 5.0
GATE_PRE_WINDOW_S = 15.0        # 失事前 15 s 的峰值統計窗
GATE_SACC_QUALITY = 2.0         # GPA.SAcc ≥2 m/s → 歸因 GPS 品質

# ---- 常態窗(ekf_vio_disagreement / nav_velocity_gps_fallback_risk) ----
NAV_WINDOW_S = 60.0             # t_warn 之前 60 s
NAV_END_MARGIN_S = 5.0          # 沒有姿態異常的飛行:以上鎖前 5 s 為終點
NAV_ARM_SKIP_S = 3.0            # 解鎖後前 3 s 不算

# ---- ekf_vio_disagreement ----
VD_FRESH_AGE_S = 0.5            # VISP 資料年齡 <0.5 s 才比較 EKF 與 VIO
VD_HOLD_S = 30.0                # 來源切換 / VISP.Rst 之後 30 s 內不算(EKF 位置重置到新來源)
VD_MED_M = 1.0                  # |EKF−VIO| 中位數 >1.0 m(失事前窗 1.57 m;對照 0.17 m)
VD_P95_M = 3.0                  # 或 p95 >3 m(失事 3.47 m;對照 2.13 m)
VD_SIGMA_M = 3.0                # 或隱含新息 sigma >3 m(需要 XKF4.SP>0.01 的樣本;失事前窗沒有這種樣本 → 不採用)
VD_SP_MIN = 0.01
VD_SIGMA_MIN_N = 20
VD_PERR_M = 5.0                 # VISP.PErr ≥5 m 持續 ≥30 s = 不確定度被鎖在高值(失事 8.591 m 持續 209.7 s)
VD_PERR_MIN_S = 30.0
VD_PERR_MAX_GAP_S = 2.0         # PErr 鎖定段可跨過正常的 VISP 小中斷(124:1.09 / 1.16 s 前後都是 8.591 → 1048.4–1258.15 一段)
VD_MIN_VALID_S = 10.0           # 窗內有效樣本不足 10 s → NA

# ---- nav_velocity_gps_fallback_risk ----
NV_RISK_MIN_S = 10.0            # 窗內「無速度來源且 GPS 無定位」累積 ≥10 s
NV_LATCH_MIN_S = 30.0           # IVN/IVE 凍結 ≥30 s = 實際沒有水平速度量測被融合
NV_LATCH_EPS = 1e-4
NV_NOGPS_RUN_S = 30.0           # GPS 無定位連續 ≥30 s 且使用 VIO 位置 = 純 VIO 無備援
NV_VEL_FRESH_S = 0.2            # VISV 資料年齡 <0.2 s 才拿來比 EKF 速度

# ---- source_switch_position_jump ----
JP_PRE_S = (0.3, 0.04)          # 切換前窗 [t−0.3, t−0.04]
JP_POST_S = (0.3, 0.8)          # 切換後窗 [t+0.3, t+0.8](被下一次切換截短;不足 2 筆樣本則該次切換不量)
JP_MIN_SAMPLES = 2
JP_INFO_M = 1.0                 # ≥1 m 列為資訊(SOFT)
JP_MEDIUM_M = 3.0               # ≥3 m severity 升為 medium(124 正常切換 1.37 / 5.14 / 4.34 m,都沒事;失事那兩次反而最小 0.18 / 0.39 m)

# ---- 局部姿態異常(只在 engine 沒填 a.t_warn / a.t_loss 時當備援) ----
AE_WARN_DEG = 10.0
AE_LOSS_DEG = 30.0
AE_MIN_DUR_S = 0.3
ACRO_MODES = (1, 13, 14)        # ACRO / SPORT / FLIP:姿態誤差大是刻意的

# 近似值的「不確定」文字
_CONF = {"source_switch_audit": "medium", "vio_dropout": "medium", "ekf_aiding_timeout": "high",
         "ekf_testratio_gate": "high", "ekf_vio_disagreement": "low", "nav_velocity_gps_fallback_risk": "medium",
         "source_switch_position_jump": "high"}


# ====================================================================================
# 共用小工具
# ====================================================================================

def _r(x, nd: int = 3):
    """numpy / 一般數值 → JSON 友善的 float(NaN / inf / None → None)。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _na(det_id: str, context: str, text: str, **values) -> Finding:
    return Finding(det_id, DOMAIN, NA, context=context, values=dict(values, confidence=_CONF.get(det_id, "low")),
                   text_zh=text)


def _not_armed(det_id: str) -> Finding:
    return _na(det_id, "not_armed", "未解鎖(桌上 / 手持)的 log,沒有飛行可分析,導航偵測不適用。")


def _clear_or_short(det_id: str, iv: ArmedInterval, f: Finding) -> Finding:
    """解鎖太短(<NAV_MIN_CLEAR_S)時,「沒找到問題」不能當 CLEAR(資料不足以排除)→ NA insufficient_data。"""
    dur = float(iv.t1 - iv.t0)
    if f.status != CLEAR or dur >= NAV_MIN_CLEAR_S:
        return f
    return _na(det_id, "insufficient_data",
               f"解鎖只有 {dur:.1f}s(<{NAV_MIN_CLEAR_S:.0f}s),資料太短,沒有發現問題也不能據以排除。",
               armed_s=_r(dur))


def _int_bits(x) -> np.ndarray:
    """float 欄位 → int64 位元遮罩;NaN / inf 當 0(直接 astype 會出現 'invalid value encountered in cast' 與垃圾位元)。"""
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, 0.0).astype(np.int64)


def _nanmedian_rows(stack: np.ndarray) -> np.ndarray:
    """沿 axis 0 的 nanmedian;整欄都是 NaN 時回 NaN(不發 'All-NaN slice' 警告)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def _airborne(ctx: Ctx, iv: ArmedInterval, t) -> np.ndarray:
    """各時間點是否離地(ctx.landed_steps:EV28 離地、EV18 落地;解鎖當下視為在地面)。"""
    t = np.asarray(t, dtype=float)
    with warnings.catch_warnings():     # ctx.ev_rows() 直接把 EV.Id 轉 int;NaN 的 Id 只會變成不認得的事件,不影響落地階梯
        warnings.simplefilter("ignore", RuntimeWarning)
        st = ctx.landed_steps(iv)
    ts = np.array([s[0] for s in st], dtype=float)
    landed = np.array([bool(s[1]) for s in st], dtype=bool)
    i = np.searchsorted(ts, t, side="right") - 1
    return np.where(i >= 0, ~landed[np.clip(i, 0, None)], False)


def _air_runs(air: np.ndarray, t: np.ndarray, i0: int, i1: int, min_dur: float) -> list:
    """區段 [i0, i1] 內「離地」且持續 ≥min_dur 的子區段(索引已換回整體)。"""
    sub = runs(air[i0:i1 + 1], t[i0:i1 + 1], min_dur=min_dur, max_gap=MAX_GAP_XKF_S)
    return [(s, e, i0 + j0, i0 + j1) for (s, e, j0, j1) in sub]


def _hold_idx(t_src: np.ndarray, t_dst: np.ndarray):
    """各 t_dst 取 t_src 中「最後一筆 ≤ t_dst」的索引;(idx 已夾在 0..n-1, 有效遮罩 = t_dst 之前確有樣本)。"""
    i = np.searchsorted(t_src, t_dst, side="right") - 1
    return np.clip(i, 0, max(t_src.size - 1, 0)), i >= 0


def _hold(t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray, fill=np.nan) -> np.ndarray:
    """階梯保持式取值(樣本之前回 fill)。"""
    if t_src.size == 0:
        return np.full(np.shape(t_dst), fill, dtype=float)
    i, ok = _hold_idx(t_src, t_dst)
    out = np.asarray(v_src, dtype=float)[i]
    return np.where(ok, out, fill)


class _ParamLog:
    """讓 core.ekfsrc.source_sets() 能直接吃 ctx.params(它只讀 logdata.index.param_values())。"""

    def __init__(self, params: dict):
        self.index = SimpleNamespace(param_values=lambda: params)


def _sets(ctx: Ctx):
    """EK3_SRCn_* → ({1..3: ekfsrc.SourceSet}, EK3_SRC_OPTIONS);解碼重用 core.ekfsrc。"""
    return ekfsrc.source_sets(_ParamLog(getattr(ctx, "params", {}) or {}))


def _code(sets: dict, n: int, axis: str) -> int:
    ss = sets.get(int(n))
    return int(ss.codes.get(axis, 0)) if ss is not None else 0


def _uses_gps(sets: dict, n: int) -> bool:
    return _code(sets, n, "POSXY") == SRC_GPS or _code(sets, n, "VELXY") == SRC_GPS


def _uses_vio(sets: dict, n: int) -> bool:
    return _code(sets, n, "POSXY") == SRC_EXTNAV


def _flag_by_set(sets: dict, set_n: np.ndarray, pred) -> np.ndarray:
    """對一串組別(1-based)套 pred(sets, n) → 布林陣列。"""
    out = np.zeros(set_n.size, dtype=bool)
    for n in np.unique(set_n):
        out[set_n == n] = bool(pred(sets, int(n)))
    return out


def _kind(sets: dict, n: int) -> str:
    c = _code(sets, n, "POSXY")
    return "vio" if c == SRC_EXTNAV else "gps" if c == SRC_GPS else "other"


class _SetSteps:
    """使用中來源組的階梯(1-based 組別)。source: xkfs / ev / assumed。"""

    def __init__(self, t: np.ndarray, n: np.ndarray, source: str):
        self.t, self.n, self.source = np.asarray(t, float), np.asarray(n, int), source

    def at(self, t) -> np.ndarray:
        i = np.clip(np.searchsorted(self.t, np.asarray(t, float), side="right") - 1, 0, max(self.t.size - 1, 0))
        return self.n[i]

    def switches(self) -> list[tuple[float, int, int]]:
        if self.t.size < 2:
            return []
        d = np.flatnonzero(np.diff(self.n) != 0)
        return [(float(self.t[i + 1]), int(self.n[i]), int(self.n[i + 1])) for i in d]

    def seg_start(self, t) -> np.ndarray:
        """各時間點「目前這組」開始使用的時間(最後一次切換;之前沒有切換 → -inf)。"""
        t = np.asarray(t, float)
        sw = np.array([s[0] for s in self.switches()], float)
        if not sw.size:
            return np.full(t.shape, -np.inf)
        i = np.searchsorted(sw, t, side="right") - 1
        return np.where(i >= 0, sw[np.clip(i, 0, None)], -np.inf)


def _set_steps(ctx: Ctx) -> _SetSteps:
    """XKFS.SS(0-based → +1,與 core.ekfsrc 一致);沒有就用 EV 85/86/87;再沒有就假設整段組 1
    (source='assumed' = 使用中的組其實未知:依賴「當時是哪一組」的結論只能給 SOFT 或用最壞情況判斷)。"""
    s = ctx.series("XKFS.SS")
    ok = np.isfinite(s[1]) if s is not None else None
    if s is not None and ok.any():
        return _SetSteps(s[0][ok], np.rint(s[1][ok]).astype(int) + 1, "xkfs")
    te, ids = ctx.ev_rows()
    m = np.isin(ids, (85, 86, 87))
    if m.any():
        t = np.concatenate([[-1e9], te[m]])
        n = np.concatenate([[1], ids[m] - 84])
        return _SetSteps(t, n, "ev")
    return _SetSteps(np.array([-1e9]), np.array([1]), "assumed")


def _gps_ok_series(ctx: Ctx):
    """(t, ok) —— ok = Status>=3 且 NSats>=6;沒有 GPS 表回傳 None。"""
    tb = ctx.table("GPS")
    if tb is None or "Status" not in tb:
        return None
    ns = tb.get("NSats")
    ok = tb["Status"] >= GPS_MIN_STATUS
    if ns is not None:
        ok = ok & (ns >= GPS_MIN_SATS)
    return tb["t"], ok


def _gps_ok_on(grid: np.ndarray, gps) -> np.ndarray:
    """各 grid 時間點上 GPS 是否可用(用保持式;最後一筆樣本超過 GPS_STALE_S 前視為沒資料)。"""
    if gps is None:
        return np.zeros(grid.size, dtype=bool)
    tg, ok = gps
    i, has = _hold_idx(tg, grid)
    fresh = has & ((grid - tg[i]) <= GPS_STALE_S)
    return fresh & ok[i]


def _visp(ctx: Ctx):
    tb = ctx.table("VISP")
    if tb is None or tb["t"].size < 2:
        return None
    return tb


def _visp_age(tv: np.ndarray, t) -> np.ndarray:
    """t 時刻距「最後一筆 VISP」的秒數(之前沒有 → inf)。"""
    t = np.asarray(t, float)
    i, has = _hold_idx(tv, t)
    return np.where(has, t - tv[i], np.inf)


def _lanes(ctx: Ctx, name: str, need: tuple[str, ...]) -> list[dict]:
    """表的各 lane(沒有 instance 的表只有一條)。缺欄位的 lane 略過。"""
    out = []
    for ln in (ctx.instances(name) or [None]):
        tb = ctx.table(name, ln)
        if tb is not None and all(k in tb for k in need) and tb["t"].size:
            out.append(tb)
    return out


def _iv_mask(t: np.ndarray, iv: ArmedInterval, lo_skip: float = 0.0) -> np.ndarray:
    return (t >= iv.t0 + lo_skip) & (t <= iv.t1)


# ---------------- 姿態異常參考(優先用 engine 的 anchors,沒有就局部估算) ----------------

class _Ref(NamedTuple):
    t_warn: float | None
    t_loss: float | None
    local: bool          # True = anchors 沒有,這裡自己算的


def _local_att_events(ctx: Ctx, iv: ArmedInterval):
    """局部備援:推力向量誤差 ae(同韌體 AngErr)>10° / >30° 連續 0.3 s 的起點;不含 ACRO/SPORT/FLIP。"""
    names = ("Roll", "Pitch", "Yaw", "DesRoll", "DesPitch", "DesYaw")
    tb = ctx.table("ANG")
    if tb is None or not all(k in tb for k in names):
        tb = ctx.table("ATT")
    if tb is None or not all(k in tb for k in names):
        return None, None
    m = _iv_mask(tb["t"], iv)
    if m.sum() < 10:
        return None, None
    t = tb["t"][m]
    ae = thrust_vector_error_deg(*(tb[k][m] for k in names))
    ok = np.ones(t.size, dtype=bool)
    tm, nums, _ = ctx.mode_rows()
    if tm.size:
        i, has = _hold_idx(tm, t)
        ok = ~(has & np.isin(nums[i], ACRO_MODES))

    def first(th: float):
        rr = runs((ae > th) & ok, t, min_dur=AE_MIN_DUR_S, max_gap=0.2)
        return rr[0][0] if rr else None

    return first(AE_WARN_DEG), first(AE_LOSS_DEG)


def _iv_key(iv: ArmedInterval) -> tuple:
    """快取鍵:index 相同但起訖不同的區間(例如截短的分析區間)不能共用結果。"""
    return (iv.index, float(iv.t0), float(iv.t1))


def _crash_flight(a: Anchors, ref: "_Ref | None" = None) -> bool:
    """這次解鎖是否以失事 / 異常結束(決定文字用「失事前」還是「上鎖前」;正常降落不寫失事字眼)。"""
    if a.t_detect is not None or (ref is not None and ref.t_loss is not None):
        return True
    return a.end_type not in ("normal_landing", "normal_end", "not_armed", "unknown", "")


def _get_ref(ctx: Ctx, a: Anchors, iv: ArmedInterval) -> _Ref:
    key = ("ref", _iv_key(iv), a.t_warn, a.t_loss)
    cache = ctx.__dict__.setdefault("_nav_cache", {})
    if key not in cache:
        tw, tl, local = a.t_warn, a.t_loss, False
        if tw is None and tl is None:
            tw, tl = _local_att_events(ctx, iv)
            local = tw is not None or tl is not None
        cache[key] = _Ref(tw, tl, local)
    return cache[key]


def _phase(ref: _Ref, t: float) -> str:
    """與 engine 的標籤規則一致(只用 t_onset):precursor / development / consequence;沒有姿態異常 → no_attitude_event。"""
    if ref.t_warn is None and ref.t_loss is None:
        return "no_attitude_event"
    if ref.t_loss is not None and t >= ref.t_loss:
        return "consequence"
    if ref.t_warn is not None and t >= ref.t_warn:
        return "development"
    return "precursor"


def _after_text(ref: _Ref, t: float) -> str:
    """「這件事相對姿態異常的時間」的中文片語(已含句尾標點之前的部分);前面要接主詞(用 _rel_sentence)。"""
    ph = _phase(ref, t)
    if ph == "consequence" and ref.t_loss is not None:
        return f"發生在姿態失控(誤差 >30°,{ref.t_loss:.2f}s)之後 {t - ref.t_loss:.1f}s,屬於後果而非肇因"
    if ph == "development" and ref.t_warn is not None:
        return f"發生在姿態誤差 >10°({ref.t_warn:.2f}s)之後 {t - ref.t_warn:.1f}s,屬於姿態發散期間的反應而非起因"
    if ph == "precursor" and ref.t_warn is not None:
        return f"早於姿態誤差 >10°({ref.t_warn:.2f}s){ref.t_warn - t:.1f}s,是肇因候選"
    if ph == "precursor" and ref.t_loss is not None:
        return f"早於姿態失控(誤差 >30°,{ref.t_loss:.2f}s){ref.t_loss - t:.1f}s,是肇因候選"
    return "此次飛行沒有姿態異常"


def _rel_sentence(ref: _Ref, t: float, subject: str) -> str:
    """完整的因果短句(不含句號):有姿態異常 → 主詞 + _after_text;沒有 → 不接主詞(避免「這次切換此次飛行…」的斷句)。"""
    if _phase(ref, t) == "no_attitude_event":
        return "此次飛行沒有姿態異常(沒有失控時間可對照因果)"
    return subject + _after_text(ref, t)


def _std_window(iv: ArmedInterval, ref: _Ref):
    """常態窗 (t_ref, w0):窗 = [w0, t_ref];t_ref = t_warn(沒有姿態異常就取上鎖前 5 s),w0 = t_ref 前 60 s(不早於解鎖後 3 s)。"""
    if ref.t_warn is not None:
        t_ref = ref.t_warn
    else:
        t_ref = iv.t1 - NAV_END_MARGIN_S if iv.t1 - iv.t0 > 2 * NAV_END_MARGIN_S + NAV_ARM_SKIP_S else iv.t1
    t_ref = min(t_ref, iv.t1)
    return t_ref, max(iv.t0 + NAV_ARM_SKIP_S, t_ref - NAV_WINDOW_S)


def _touchdown_times(ctx: Ctx, a: Anchors, iv: ArmedInterval) -> list[float]:
    """觸地撞擊(ground_contact,20–45 m/s²)的群集起點。優先 a.impacts / a.t_touch,否則由 IMU[0] 局部算。"""
    out: list[float] = []
    for im in (a.impacts or []):
        if str(im.get("cls", "")) == "ground_contact":
            v = im.get("t_start", im.get("t_peak"))
            if v is not None and iv.t0 <= float(v) <= iv.t1:
                out.append(float(v))
    if a.t_touch is not None and iv.t0 <= a.t_touch <= iv.t1:
        out.append(float(a.t_touch))
    if out:
        return sorted(set(out))
    key = ("td", _iv_key(iv))
    cache = ctx.__dict__.setdefault("_nav_cache", {})
    if key not in cache:
        tb = ctx.table("IMU", 0) if ctx.instances("IMU") else ctx.table("IMU")
        res: list[float] = []
        if tb is not None and all(k in tb for k in ("AccX", "AccY", "AccZ")):
            m = _iv_mask(tb["t"], iv)
            t = tb["t"][m]
            mag = np.sqrt(tb["AccX"][m] ** 2 + tb["AccY"][m] ** 2 + tb["AccZ"][m] ** 2)
            hit = np.flatnonzero(mag > TD_SHOCK_MS2)
            if hit.size:
                start = 0
                for k in range(1, hit.size + 1):
                    if k == hit.size or t[hit[k]] - t[hit[k - 1]] > TD_MERGE_S:
                        seg = hit[start:k]
                        if float(mag[seg].max()) < TD_HARD_MS2:
                            res.append(float(t[seg[0]]))
                        start = k
        cache[key] = res
    return list(cache[key])


# ====================================================================================
# 1. source_switch_audit
# ====================================================================================

def _glitch_edges(ctx: Ctx, iv: ArmedInterval) -> list[float]:
    """GPS glitch 的上升邊緣時間(XKF4.SS bit14 lane 0;沒有就用 ERR(11,2))。只取解鎖期間。"""
    out: list[float] = []
    x4 = _lanes(ctx, "XKF4", ("SS",))
    if x4:
        b = (_int_bits(x4[0]["SS"]) & GLITCH_BIT) != 0
        up = np.flatnonzero(~b[:-1] & b[1:]) + 1
        out = [float(x4[0]["t"][i]) for i in up]
    else:
        te, sub, ec = ctx.err_rows()
        out = [float(t) for t, s, c in zip(te, sub, ec) if s == GLITCH_ERR_SUBSYS and c == 2]
    return [t for t in out if iv.t0 <= t <= iv.t1]


def _rc_switch_changes(ctx: Ctx):
    """RCx_OPTION=90 的開關通道 → [(通道, 層級變化時間 list)];用 RCIN(已轉 float)。"""
    tb = ctx.table("RCIN")
    out = []
    if tb is None:
        return out
    for ch in range(5, 15):
        if int(round(ctx.param(f"RC{ch}_OPTION", 0) or 0)) != RC_OPTION_EKF_SOURCE or f"C{ch}" not in tb:
            continue
        v = tb[f"C{ch}"]
        lvl = np.where(v < RC_LEVEL_LO, 0, np.where(v < RC_LEVEL_HI, 1, 2))
        ch_t = tb["t"][1:][np.diff(lvl) != 0]
        out.append((ch, [float(x) for x in ch_t]))
    return out


def _nearest_value(t: np.ndarray, v: np.ndarray, t0: float):
    if t.size == 0:
        return None
    return float(v[int(np.argmin(np.abs(t - t0)))])


def _timeout_runs(ctx: Ctx) -> list[tuple[float, float]]:
    """XKF4.TS&3 逾時區段(t > 首次絕對輔助、持續 ≥2 s;整份 log,不限解鎖)→ [(t_start, t_end)]。"""
    st = _aiding_state(ctx)
    if st is None or st[2] is None:
        return []
    t, ts, t_a = st
    return [(r[0], r[1]) for r in runs((ts != 0) & (t > t_a), t, min_dur=TS_MIN_DUR_S, max_gap=MAX_GAP_XKF_S)]


@guarded("source_switch_audit", DOMAIN)
def source_switch_audit(ctx: Ctx, a: Anchors):
    """稽核解鎖期間每一次 EKF 來源組切換:
      (a) 切到使用 GPS 的組(POSXY 或 VELXY = 3),但前 1 s GPS 沒有全程 Status>=3 且 NSats>=6;
      (b) 切到 GPS 組後 1 s 內出現 GPS glitch(XKF4.SS bit14 / ERR(11,2) 上升邊緣)= 切換造成;
      (c) 切到 VIO 組(POSXY=6),但最後一筆 VISP 已超過 0.5 s 前(vio_stale),或切換前根本還沒有任何 VISP
          (vio_never_received;VISP 表存在、第一筆在切換之後 → 切到一個還沒有資料的 VIO 組)。
    沒有 VISP 表(切到 VIO 組)或沒有 GPS 表(切到 GPS 組)→ data_missing,只給 SOFT(不能確認,也不能當 CLEAR)。
    HIT = (a)(b)(c) 任一成立(t_onset = 最早的有問題切換);SOFT = 只有 data_missing;
    CLEAR = 有切換或沒切換,但全部健康(解鎖 <5 s 時改回 NA)。切換相對 t_warn 的時間、切換當下的滾轉角、是否由 RC 開關
    手動切換,以及切換後 12 s 內的 EKF 輔助逾時 / EKFCHECK(後續影響,可能在上鎖後)都放進 values。"""
    det = "source_switch_audit"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    steps = _set_steps(ctx)
    if steps.source == "assumed":
        return _na(det, "no_xkfs", "log 沒有 XKFS / 來源組切換事件,無法稽核來源組切換。")
    ref = _get_ref(ctx, a, iv)
    sets, _ = _sets(ctx)
    gps = _gps_ok_series(ctx)
    visp = _visp(ctx)
    tv = visp["t"] if visp is not None else None
    sw_all = steps.switches()
    sw = [s for s in sw_all if iv.t0 <= s[0] <= iv.t1]
    edges = _glitch_edges(ctx, iv)
    te, sub, ec = ctx.err_rows()
    err_glitch = [float(t) for t, s, c in zip(te, sub, ec) if s == GLITCH_ERR_SUBSYS and c == 2]
    err_chk = [float(t) for t, s, c in zip(te, sub, ec) if s == EKFCHECK_SUBSYS and c == 2]
    rc_changes = _rc_switch_changes(ctx)
    att = ctx.table("ATT")
    x4 = _lanes(ctx, "XKF4", ("SV",))
    gpa = ctx.table("GPA")
    to_runs = _timeout_runs(ctx) if sw else []
    rows: list[dict] = []
    flagged: list[dict] = []
    for (ts, s_from, s_to) in sw:
        uses_gps, uses_vio = _uses_gps(sets, s_to), _uses_vio(sets, s_to)
        row: dict = {"t": _r(ts), "from_set": s_from, "to_set": s_to, "uses_gps": uses_gps, "uses_vio": uses_vio,
                     "phase": _phase(ref, ts)}
        flags: list[str] = []
        # (a) 切到 GPS 組,GPS 要在前 1 s 全程可用
        if uses_gps:
            if gps is not None:
                tg, ok = gps
                w = (tg >= ts - SW_GPS_WINDOW_S) & (tg <= ts)
                gps_ok = bool(w.any() and ok[w].all())
                last_ok = tg[(tg <= ts) & ok]
                row["gps_ok_1s"] = gps_ok
                row["gps_lost_s"] = None if gps_ok else (_r(ts - float(last_ok[-1])) if last_ok.size else None)
                if not gps_ok:
                    flags.append("gps_without_fix")
            else:
                row["gps_data_missing"] = True
                flags.append("gps_data_missing")
        # (b) glitch 邊緣在切換後 1 s 內
        if uses_gps:
            for e in edges:
                if 0.0 <= e - ts <= SW_GLITCH_WINDOW_S:
                    row["glitch_edge_t"] = _r(e)
                    nxt = [x for x in err_glitch if e - 0.05 <= x <= e + SW_GLITCH_WINDOW_S]
                    row["glitch_err_t"] = _r(nxt[0]) if nxt else None
                    flags.append("glitch_on_switch")
                    break
        # (c) 切到 VIO 組,最後一筆 VISP 的年齡;切換前根本沒有 VISP(年齡 = inf)→ 獨立旗標 vio_never_received
        if uses_vio:
            if tv is None:
                row["vio_data_missing"] = True
                flags.append("vio_data_missing")
            else:
                age = float(_visp_age(tv, ts))
                if np.isfinite(age):
                    row["vio_age_s"] = _r(age)
                    if age > SW_VIO_AGE_S:
                        flags.append("vio_stale")
                else:
                    # 與 vio_stale 同一把尺:切換後 0.5 s 內就收到第一筆 = 只是 VIO 剛好在切換時啟動,不算
                    nxt = tv[tv > ts]
                    row["vio_age_s"] = None
                    row["vio_first_t"] = _r(nxt[0]) if nxt.size else None
                    if not nxt.size or float(nxt[0]) - ts > SW_VIO_AGE_S:
                        flags.append("vio_never_received")
        # 脈絡:相對 t_warn、滾轉角、是否 RC 開關手動切
        if ref.t_warn is not None:
            row["dt_warn_s"] = _r(ts - ref.t_warn)
        if att is not None and "Roll" in att:
            row["roll_deg"] = _r(_nearest_value(att["t"], att["Roll"], ts), 1)
        manual = None
        for ch, chg in rc_changes:
            near = [c for c in chg if ts - SW_RC_LOOKBACK_S <= c <= ts + 0.05]
            if near:
                manual = {"channel": ch, "t": _r(near[-1])}
        row["pilot_rc_switch"] = manual
        # 後續影響:切換後 12 s 內(且在下一次切換之前)起算的 EKF 輔助逾時,及其後 5 s 內的 EKFCHECK(可能已在上鎖後)
        t_nxt = min([s[0] for s in sw_all if s[0] > ts] or [np.inf])
        fol = [r for r in to_runs if ts <= r[0] <= min(ts + SW_FOLLOWUP_S, t_nxt)]
        row["followup_timeout_t"] = _r(fol[0][0]) if fol else None
        chk = [x for x in err_chk if fol and fol[0][0] <= x <= fol[0][0] + EKFCHECK_PAIR_S]
        row["followup_ekfcheck_t"] = _r(chk[0]) if chk else None
        row["flags"] = flags
        rows.append(row)
        if flags:
            flagged.append(row)
    # 非切換造成的 glitch 邊緣:只分類(給報告看),不計入旗標
    other_glitches = []
    for e in edges:
        prev = [(t, n) for (t, f, n) in sw_all if t <= e]
        induced = bool(prev) and e - prev[-1][0] <= SW_GLITCH_WINDOW_S and _uses_gps(sets, prev[-1][1])
        if induced:
            continue
        gps_at = bool(_gps_ok_on(np.array([e]), gps)[0]) if gps is not None else False
        sv_win = [l["SV"][(l["t"] >= e - SW_SV_LOOKBACK_S) & (l["t"] <= e)] for l in x4]
        sv_win = [x for x in sv_win if x.size and np.isfinite(x).any()]
        sv_prev = float(max(np.nanmax(x) for x in sv_win)) if sv_win else None
        sacc = None
        if gpa is not None and "SAcc" in gpa:
            sacc = _nearest_value(gpa["t"], gpa["SAcc"], e)
        quality = gps_at and ((sv_prev is not None and sv_prev >= SW_QUALITY_SV) or (sacc is not None and sacc >= SW_QUALITY_SACC))
        other_glitches.append({"t": _r(e), "class": "gps_quality" if quality else "gps_reacquired",
                               "gps_ok": gps_at, "sv_prev3s": _r(sv_prev), "sacc": _r(sacc)})
    base = {"n_switches": len(rows), "switches": rows, "other_glitches": other_glitches,
            "set_source": steps.source, "gps_table": gps is not None, "visp_table": visp is not None,
            "thresholds": {"gps_window_s": SW_GPS_WINDOW_S, "glitch_window_s": SW_GLITCH_WINDOW_S,
                           "vio_age_s": SW_VIO_AGE_S},
            "confidence": _CONF[det], "t_warn": _r(ref.t_warn), "t_loss": _r(ref.t_loss),
            "anchors_local": ref.local}
    hard = [r for r in flagged if any(f not in _MISSING_FLAGS for f in r["flags"])]
    if hard:
        r0 = hard[0]
        ts0 = r0["t"]
        reason = []
        if "gps_without_fix" in r0["flags"]:
            lost = r0.get("gps_lost_s")
            reason.append("該組使用 GPS,而 GPS 已" + (f" {lost:.0f}s " if lost is not None else "") +
                          "沒有可用定位(Status<3 或衛星 <6)")
        if "glitch_on_switch" in r0["flags"]:
            reason.append(f"切換後 {r0['glitch_edge_t'] - ts0:.2f}s 內立刻出現 GPS glitch")
        if "vio_stale" in r0["flags"]:
            age0 = r0.get("vio_age_s")
            reason.append("該組使用 VIO,而最後一筆 VISP 已是" + (f" {age0:.2f}s " if age0 is not None else "很久") +
                          "前(健康切換 ≤0.04s)")
        if "vio_never_received" in r0["flags"]:
            ft = r0.get("vio_first_t")
            reason.append("該組使用 VIO,而切換時還沒收到任何 VIO 資料(VISP)" +
                          (f",第一筆在 {ft:.2f}s(切換後 {ft - ts0:.1f}s)才到" if ft is not None else ",之後也沒有收到"))
        rc = r0.get("pilot_rc_switch")
        roll = r0.get("roll_deg")
        more = f"(共 {len(hard)} 次有問題的切換)" if len(hard) > 1 else ""
        fol = next((r for r in hard if r.get("followup_timeout_t") is not None), None)
        fol_txt = (f"之後 EKF 位置輔助在 {fol['followup_timeout_t']:.2f}s 逾時"
                   + (f"、{fol['followup_ekfcheck_t']:.2f}s EKFCHECK" if fol.get("followup_ekfcheck_t") is not None else "")
                   + "。") if fol else ""
        txt = (f"解鎖期間 {ts0:.2f}s 把 EKF 來源組由第 {r0['from_set']} 組切到第 {r0['to_set']} 組,但" + ";".join(reason) +
               f"{more}。" + _rel_sentence(ref, ts0, "這次切換") +
               (f"(切換當下滾轉 {roll:.0f}°" + (",是飛手撥 RC 開關" if rc else "") + ")" if roll is not None else "") +
               "。" + fol_txt + "GPS 無定位時不要把開關當備援;切換前先確認 VIO 狀態為 valid。")
        return Finding(det, DOMAIN, HIT, "high", t_onset=ts0, t_alarm=r0.get("glitch_err_t") or ts0, context="switch",
                       values=dict(base, flagged_t=[r["t"] for r in hard], phase=r0["phase"],
                                   effect_of_tipover=bool(ref.t_loss is not None and ts0 >= ref.t_loss - 0.5)),
                       text_zh=txt)
    if flagged:      # 只有 data_missing
        r0 = flagged[0]
        what = ("VIO 來源組,但 log 沒有 VISP 資料,無法確認 VIO 資料是否新鮮" if "vio_data_missing" in r0["flags"]
                else "使用 GPS 的來源組,但 log 沒有 GPS 資料,無法確認切換時 GPS 是否有定位")
        return Finding(det, DOMAIN, SOFT, "info", t_onset=r0["t"], t_alarm=r0["t"], context="switch",
                       values=dict(base, flagged_t=[r["t"] for r in flagged], phase=r0["phase"], data_missing=True),
                       text_zh=f"{r0['t']:.2f}s 切到{what}(只能旁證:看 XKF4.TS 是否逾時)。")
    if not rows:
        return _clear_or_short(det, iv, Finding(
            det, DOMAIN, CLEAR, "info", context="no_switch", values=base,
            text_zh="解鎖期間沒有 EKF 來源組切換,沒有切換造成的 GPS glitch 或 VIO 失效。"))
    return _clear_or_short(det, iv, Finding(
        det, DOMAIN, CLEAR, "info", context="switches_ok", values=base,
        text_zh=f"解鎖期間共 {len(rows)} 次來源組切換,切換前 GPS 可用(或該組不用 GPS)、VIO 資料新鮮(≤{SW_VIO_AGE_S}s),"
                "切換後 1 s 內沒有 GPS glitch。"))


# ====================================================================================
# 2. vio_dropout
# ====================================================================================

def _fmt_gap(g: dict) -> dict:
    return {k: (_r(v) if isinstance(v, float) else v) for k, v in g.items()}


@guarded("vio_dropout", DOMAIN)
def vio_dropout(ctx: Ctx, a: Anchors):
    """VISP(VIO 位置)資料中斷。只適用有 VISP 表的 log(117/118/122/123 → NA)。事件三類(各自一筆 Finding):
      near_touchdown 中斷 ≥1.0 s,起點離觸地撞擊群集 ≤0.3 s;使用 VIO 組 → HIT(GPS 組 → SOFT,無害);
      in_vio_mode    使用 VIO 組(POSXY=6)而 VIO 資料年齡 ≥1.5 s → HIT。含:串流在 VIO 組中停掉(永久 / 之後恢復)、
                     切到(或解鎖時就在)VIO 組時資料已經舊了、VISP 表存在但切換 / 解鎖時還沒收到任何一筆(never_received;
                     年齡從開始使用這組 / 解鎖起算,不會是 inf);
      stream_end     最後一筆 VISP 距上鎖 ≥1.5 s 且沒恢復,但當時不是 VIO 組 → SOFT。
    in_vio_mode 的 t_onset:VIO 在「VIO 組使用中」停掉 → 最後一筆 VISP(中斷開始);停掉時沒在用 VIO 組(之後才切進來)、
    解鎖前就停、或從沒收到 → 開始使用觸發這段的 VIO 組(或解鎖)的時間(不會是 −inf / 解鎖前)。
    t_alarm = 年齡達 1.5 s 且正在用 VIO 組的時間;最後一筆 VISP 時間放 values.event.t_last。
    已知限制(審查 #10,未在本模組改):124 的 VISP 1258.146 在第 3 組使用中停掉 → onset 1258.146(驗收 A6 的
    「last VISP 1258.146」),與 engine 的 t_loss 1258.102 只差 0.044 s;t_loss 若落在驗收容許的上緣(>1258.146)會被貼成
    development。本模組自己的判斷 values.effect_of_tipover=True(最後一筆在 t_loss±0.5 s 內),需要 engine 端採用。
    來源組未知(沒有 XKFS / EV 85-87)時以「任何一組用 VIO 就可能在用」的最壞情況判斷,依賴使用中組別的結論降為 SOFT。
    飛行中正常的 0.5–1.1 s 小中斷(124 有 7 次)不觸發;只列在 values。信心中等:與門檻的間距僅 0.34 s。"""
    det = "vio_dropout"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    visp = _visp(ctx)
    if visp is None:
        return _na(det, "no_visp", "此 log 沒有 VISP(VIO 位置)資料,VIO 中斷偵測不適用。")
    tv = visp["t"]
    in_fl = _iv_mask(tv, iv)
    n_fl = int(in_fl.sum())
    ref = _get_ref(ctx, a, iv)
    sets, _ = _sets(ctx)
    steps = _set_steps(ctx)
    set_known = steps.source != "assumed"
    vio_possible = any(_uses_vio(sets, n) for n in (1, 2, 3))
    # ---- 使用 VIO 組且資料年齡 >= 1.5 s ----
    # 時間格:XKF4 / XKF3 lane 0(~25 Hz);都沒有就用 25 Hz 均勻格。不能用 VISP 自己的時間 —— 那樣資料年齡永遠 ≈0,
    # 串流在 VIO 組中停掉也看不到。
    x4 = _lanes(ctx, "XKF4", ("SS",)) or _lanes(ctx, "XKF3", ("IPN",))
    grid = x4[0]["t"] if x4 else np.arange(iv.t0, iv.t1 + 1e-9, 0.04)
    g_t = grid[_iv_mask(grid, iv)]
    set_n = steps.at(g_t)
    vio_on = _flag_by_set(sets, set_n, _uses_vio) if set_known else np.full(g_t.size, vio_possible)
    if n_fl < 2 and not vio_on.any():
        return _na(det, "no_visp_in_flight", "解鎖期間沒有 VISP 資料(此次飛行沒有收到 VIO),也沒有使用 VIO 組,"
                   "VIO 中斷偵測不適用。", set_source=steps.source)
    age_raw = _visp_age(tv, g_t)
    never = ~np.isfinite(age_raw)
    seg0 = np.maximum(steps.seg_start(g_t), iv.t0)      # 目前這組開始使用的時間(不早於解鎖)
    age = np.where(never, g_t - seg0, age_raw)          # 從沒收到 VISP:年齡從開始使用這組 / 解鎖起算
    rr = runs(vio_on & (age >= VIO_MODE_AGE_S), g_t, min_dur=0.0, max_gap=MAX_GAP_XKF_S)
    vio_runs = []
    for (t_s, t_e, i0, i1) in rr:
        t_last = None if never[i0] else float(t_s - age_raw[i0])     # 最後一筆正常 VISP(可能在解鎖 / 切換之前)
        seg_i = float(seg0[i0])                                      # 觸發這段的 VIO 組開始使用(不早於解鎖)
        # onset:VIO 在「VIO 組使用中」停掉 → 最後一筆 VISP;停掉時不在用 VIO 組(之後才切進來 / 解鎖前就停 / 從沒收到)
        # → 開始使用這段 VIO 組(或解鎖)的時間
        died_in_use = (t_last is not None and t_last >= iv.t0 and
                       (not set_known or _uses_vio(sets, int(steps.at(t_last)))))
        t_on = t_last if died_in_use else seg_i
        t_from = t_last if t_last is not None else t_on
        nxt = tv[tv > t_from + 1e-9]
        resume = float(nxt[0]) if nxt.size else None
        permanent = resume is None or resume > iv.t1
        end = min(resume, iv.t1) if resume is not None else iv.t1
        vio_runs.append({"t_onset": t_on, "t_last": t_last, "t_alarm": max(t_from + VIO_MODE_AGE_S, seg_i),
                         "t_run_start": float(t_s), "dur": end - t_on, "gap_s": end - t_from, "resume": resume,
                         "permanent": permanent, "never_received": bool(never[i0]),
                         "entered_stale": bool(t_last is not None and not died_in_use),
                         "reentered_t": seg_i if (died_in_use and seg_i > t_last + 1e-6) else None,
                         "age_at_reentry_s": (seg_i - t_last) if (died_in_use and seg_i > t_last + 1e-6) else None,
                         "set": int(set_n[i0]) if set_known else None})
    max_age_vio = float(np.max(age[vio_on])) if vio_on.any() else None
    q = visp.get("Q")

    def q_before(i_last: int) -> list:
        return [] if q is None else [int(x) if np.isfinite(x) else None for x in q[max(0, i_last - 4):i_last + 1]]

    # ---- 中斷 ≥ 門檻(起點與終點都在飛行中) ----
    td = _touchdown_times(ctx, a, iv)
    gaps = np.diff(tv)
    sel = np.flatnonzero((gaps >= VIO_SUBTHRESH_S) & (tv[:-1] >= iv.t0) & (tv[1:] <= iv.t1))
    dropouts, sub_gaps = [], []
    for i in sel:
        ts_, dur = float(tv[i]), float(gaps[i])
        near = [x for x in td if abs(ts_ - x) <= VIO_TD_WINDOW_S]
        g = {"t_start": ts_, "dur": dur, "resume": float(tv[i + 1]), "set": int(steps.at(ts_)) if set_known else None,
             "near_touchdown": bool(near), "q_before": q_before(int(i))}
        (dropouts if dur >= VIO_GAP_MIN_S else sub_gaps).append(g)
    # ---- 串流提前結束 ----
    t_last_fl = tail_open = None
    tail = False
    if n_fl:
        last_i = int(np.flatnonzero(in_fl)[-1])
        t_last_fl = float(tv[last_i])
        resume_after = float(tv[last_i + 1]) if last_i + 1 < tv.size else None
        tail_open = iv.t1 - t_last_fl
        tail = tail_open >= VIO_TAIL_S and (resume_after is None or resume_after > iv.t1)

    base = {"n_visp_in_flight": n_fl, "dropouts_ge_1s": [_fmt_gap(g) for g in dropouts[:12]],
            "subthreshold_gaps": [_fmt_gap(g) for g in sub_gaps[:12]], "max_age_in_vio_mode_s": _r(max_age_vio),
            "last_visp_t": _r(t_last_fl), "tail_open_s": _r(tail_open), "touchdown_t": [_r(x) for x in td],
            "set_source": steps.source, "vio_set_possible": vio_possible,
            "thresholds": {"gap_s": VIO_GAP_MIN_S, "vio_mode_age_s": VIO_MODE_AGE_S, "tail_s": VIO_TAIL_S,
                           "touchdown_window_s": VIO_TD_WINDOW_S},
            "confidence": _CONF[det], "t_warn": _r(ref.t_warn), "t_loss": _r(ref.t_loss), "anchors_local": ref.local}
    out: list[Finding] = []
    msgs = ctx.messages_between(iv.t0, iv.t1 + 40.0, r"^SRC=\d+/\d+:VIO (valid|degraded|invalid)") if hasattr(ctx, "messages_between") else []
    stale_msgs = [(t, s) for t, s in msgs if "stale" in s]
    # 伴隨電腦(sysid 1 / compid 197)的 VIO 狀態訊息:解鎖期間 + 上鎖後 30 s;'near-range' 降級每 5 s 重複一次,是雜訊
    # 連續重複的同一句(例:'stale' 每 5 s 一次)合併成一筆 + repeat 次數;只留最後 12 筆(離失事 / 上鎖最近的才有用)
    comp: list[dict] = []
    for t, s in msgs:
        if t > iv.t1 + VIO_MSG_POST_S or VIO_MSG_NOISE.search(s):
            continue
        if comp and comp[-1]["text"] == s[:80]:
            comp[-1]["repeat"] += 1
            continue
        comp.append({"t": _r(t), "text": s[:80], "repeat": 1,
                     "phase": ("post_disarm" if t > iv.t1 else _phase(ref, t))})
    base["companion_msgs"] = comp[-VIO_MSG_MAX:]
    base["n_companion_msgs"] = len(comp)
    unknown_txt = "(log 沒有來源組紀錄(XKFS / EV 85-87),無法確認當時是否使用 VIO 組,只能列為弱證據)"

    # near_touchdown
    others = [x["dur"] for x in dropouts + sub_gaps if not x["near_touchdown"]]
    for g in dropouts:
        if not g["near_touchdown"]:
            continue
        on_vio = _uses_vio(sets, g["set"]) if set_known else vio_possible
        no_vel = (_code(sets, g["set"], "VELXY") == 0) if set_known else None
        ph = _phase(ref, g["t_start"])
        txt = (f"觸地(撞擊 {min(td, key=lambda x: abs(x - g['t_start'])):.2f}s)時 VIO 資料中斷 {g['dur']:.2f}s"
               f"({g['t_start']:.2f}→{g['resume']:.2f}s,中斷前 Q={g['q_before']})。")
        if on_vio and set_known:
            txt += ("當時使用 VIO 組" + ("且沒有水平速度來源" if no_vel else "") + ",EKF 只能靠慣性硬撐,觸地震動易被積分成位置漂移。"
                    "請檢查相機視野 / 曝光、近地面特徵點與伴隨電腦 VIO 日誌。")
        elif on_vio:
            txt += "有來源組使用 VIO" + unknown_txt + "。"
        else:
            txt += (f"但當時使用第 {g['set']} 組(非 VIO),有 GPS 輔助時無害。" if set_known else
                    "但所有來源組都不用 VIO 位置,有 GPS 輔助時無害。")
        if others:
            mx = max(others)
            txt += (f"注意:本次飛行非觸地的中斷最長 {mx:.2f}s" +
                    (f"(已超過 {VIO_GAP_MIN_S}s 門檻,正常飛行也會出現)" if mx >= VIO_GAP_MIN_S else "") +
                    ",鑑別依據是「發生在觸地當下」;")
        txt += "門檻只用一次失事(124)校準,信心中等。"
        hit = on_vio and set_known
        out.append(Finding(det, DOMAIN, HIT if hit else SOFT, "medium" if on_vio else "info",
                           t_onset=g["t_start"], t_alarm=g["t_start"] + VIO_GAP_MIN_S, context="near_touchdown",
                           values=dict(base, event=_fmt_gap(g), active_set=g["set"], vio_set_active=on_vio,
                                       velocity_source=(None if no_vel is None else not no_vel), set_unknown=not set_known,
                                       max_other_gap_s=_r(max(others)) if others else None, phase=ph), text_zh=txt))
    # in_vio_mode(含永久中止 / 切到舊資料 / 從沒收到)
    tail_taken = False
    for vr in vio_runs:
        t_on, t_last = vr["t_onset"], vr["t_last"]
        t_from = t_last if t_last is not None else t_on
        ph = _phase(ref, t_on)
        effect = bool(ref.t_loss is not None and t_last is not None and t_last >= ref.t_loss - 0.5)
        tail_taken |= vr["permanent"]
        where = "解鎖時使用的是 VIO 組,但" if t_on <= iv.t0 + 1e-6 else f"{t_on:.2f}s 切到 VIO 組時"
        if vr["never_received"]:
            head = (f"{where}還沒收到任何 VIO 資料(VISP)" +
                    (f",直到上鎖({iv.t1:.2f}s)都沒有" if vr["permanent"] else
                     f",第一筆到 {vr['resume']:.2f}s 才出現(使用 VIO 組卻沒有 VIO {vr['dur']:.1f}s)"))
        elif vr["entered_stale"]:
            head = (f"{where} VIO 最後一筆資料已是 {t_last:.2f}s({t_on - t_last:.2f}s 前)" +
                    (f",之後再也沒有恢復(到上鎖前 {iv.t1 - t_last:.1f}s)" if vr["permanent"] else
                     f",{vr['resume']:.2f}s 才恢復(使用 VIO 組卻沒有新資料 {vr['dur']:.1f}s)"))
        elif vr["permanent"]:
            head = (f"VIO 在使用 VIO 組時({t_last:.2f}s)送出最後一筆資料後再也沒有恢復(到上鎖前 {iv.t1 - t_last:.1f}s),"
                    + (f"{vr['reentered_t']:.2f}s 切回 VIO 組時資料年齡已 {vr['age_at_reentry_s']:.2f}s" if vr["reentered_t"] is not None
                       else f"使用 VIO 組期間資料年齡已 ≥{VIO_MODE_AGE_S}s"))
        else:
            head = (f"使用 VIO 組期間 VIO 中斷 {vr['dur']:.2f}s({t_last:.2f}s 起,{vr['resume']:.2f}s 恢復"
                    + (f";{vr['reentered_t']:.2f}s 切回 VIO 組時資料年齡已 {vr['age_at_reentry_s']:.2f}s" if vr["reentered_t"] is not None
                       else "") + ")")
        rel = (f"最後一筆 VISP({t_last:.2f}s)在姿態失控前後 0.5 s 內,是翻覆的結果(相機傾斜 / 失去特徵)" if effect else
               _rel_sentence(ref, t_on, "這次中斷"))
        stale = [t for t, s in stale_msgs if t >= t_from]
        out.append(Finding(det, DOMAIN, HIT if set_known else SOFT, "medium", t_onset=t_on, t_alarm=vr["t_alarm"],
                           context="in_vio_mode",
                           values=dict(base, event={k: (_r(v) if isinstance(v, float) else v) for k, v in vr.items()},
                                       permanent=vr["permanent"], never_received=vr["never_received"],
                                       effect_of_tipover=effect, phase=ph, set_unknown=not set_known,
                                       stale_msg_t=_r(stale[0]) if stale else None),
                           text_zh=f"{head}" + ("" if set_known else unknown_txt) +
                                   f";{rel}。請檢查相機視野 / 曝光、伴隨電腦 VIO 程式的狀態,以及切換前是否確認 VIO 為 valid。"))
    if tail and not tail_taken:
        on_vio_end = _uses_vio(sets, int(steps.at(iv.t1))) if set_known else vio_possible
        ph = _phase(ref, t_last_fl)
        effect = bool(ref.t_loss is not None and t_last_fl >= ref.t_loss - 0.5)
        if on_vio_end and set_known:
            who = "上鎖時使用的是 VIO 組(資料空洞讓使用期間的年齡無法逐點確認),EKF 可能已失去位置來源。"
        elif on_vio_end:
            who = "有來源組使用 VIO" + unknown_txt + "。"
        else:
            who = "但當時使用的不是 VIO 組,對 EKF 沒有直接影響。" if set_known else "但所有來源組都不用 VIO 位置,對 EKF 沒有直接影響。"
        out.append(Finding(det, DOMAIN, SOFT, "info", t_onset=t_last_fl, t_alarm=t_last_fl + VIO_TAIL_S, context="stream_end",
                           values=dict(base, permanent=True, effect_of_tipover=effect, phase=ph, vio_set_active_at_end=on_vio_end,
                                       set_unknown=not set_known),
                           text_zh=f"VIO 資料在 {t_last_fl:.2f}s 後就沒有了(距上鎖 {tail_open:.1f}s)," + who))
    out.sort(key=lambda f: (f.t_onset if f.t_onset is not None else 1e18))
    if out:
        return out
    if vio_on.any():
        ax = f"使用 VIO 組期間沒有 ≥{VIO_MODE_AGE_S}s 的資料中斷(最長 {max_age_vio:.2f}s)" + \
             ("(來源組未知,以「一直在用 VIO 組」的最壞情況檢查)" if not set_known else "")
    else:
        ax = "解鎖期間沒有使用 VIO 組"
    big = f";飛行中另有 {len(dropouts)} 次 ≥{VIO_GAP_MIN_S}s 的小中斷(最長 {max(g['dur'] for g in dropouts):.2f}s,非觸地、非 VIO 組久斷)" if dropouts else ""
    return _clear_or_short(det, iv, Finding(
        det, DOMAIN, CLEAR, "info", context="none", values=base,
        text_zh=f"{ax},觸地時沒有 ≥{VIO_GAP_MIN_S}s 的 VIO 中斷,VIO 串流直到上鎖都在{big}。"))


# ====================================================================================
# 3. ekf_aiding_timeout
# ====================================================================================

def _aiding_state(ctx: Ctx):
    """XKF4 各 lane 的 (TS&3) 取最大,放到 lane 0 的時間格上;回傳 (t_grid, ts, t_abs_aiding) 或 None。"""
    lanes = _lanes(ctx, "XKF4", ("TS", "SS"))
    if not lanes:
        return None
    t = lanes[0]["t"]
    ts = np.zeros(t.size)
    for l in lanes:
        i, has = _hold_idx(l["t"], t)
        ts = np.maximum(ts, np.where(has, (_int_bits(l["TS"]) & TS_MASK)[i], 0))
    hp = np.flatnonzero((_int_bits(lanes[0]["SS"]) & HPOSABS_BIT) != 0)
    t_a = float(t[hp[0]]) if hp.size else None
    return t, ts, t_a


def _last_position_data(ctx: Ctx, t: float):
    """時間 t 之前 EKF 最後一筆「可用」的水平位置資料:(時間, 來源)。
    只算當時使用中的來源組真的有用到的感測器:GPS(Status>=3 且 NSats>=6,且該組 POSXY/VELXY=3)、
    VISP(該組 POSXY=6)。例:118 切到 VIO 組但沒有 VISP,GPS 雖然健康卻沒被使用 → 從切走 GPS 的時刻起算(約 7 s 後逾時)。"""
    sets, _ = _sets(ctx)
    steps = _set_steps(ctx)
    cand: list[tuple[float, str]] = []
    gps = _gps_ok_series(ctx)
    if gps is not None:
        tg, ok = gps
        m = (tg <= t) & ok
        if m.any():
            m[m] = _flag_by_set(sets, steps.at(tg[m]), _uses_gps)
        if m.any():
            cand.append((float(tg[m][-1]), "GPS"))
    v = _visp(ctx)
    if v is not None:
        tv = v["t"]
        m = tv <= t
        if m.any():
            m[m] = _flag_by_set(sets, steps.at(tv[m]), _uses_vio)
        if m.any():
            cand.append((float(tv[m][-1]), "VISP"))
    return max(cand) if cand else (None, None)


@guarded("ekf_aiding_timeout", DOMAIN)
def ekf_aiding_timeout(ctx: Ctx, a: Anchors):
    """EKF 位置 / 速度輔助逾時(XKF4.TS 取三條 lane 最大並遮 &3;t>首次絕對輔助,持續 ≥2 s),並配對 5 s 內的
    ERR(16,2) EKFCHECK / ERR(17,1) FAILSAFE_EKFINAV。
      解鎖期間內的部分本身也要 ≥2 s(解鎖前就開始的逾時只看解鎖後的那一段;解鎖後不到 2 s 就恢復的只記在
      values.pre_arm_carryover,不算);其中離地(EV28 之後)≥2 s → HIT(context in_flight;若已在 t_loss 之後則
      in_flight_after_loss);全部在解鎖後的地面 → SOFT armed_ground;沒有 → CLEAR(解鎖 <5 s → NA)。
      上鎖後 60 s 內起算(且在下次解鎖之前)→ 另外一筆 HIT,context post_disarm = 失事後果標記。
    lag_s(距最後一筆位置資料)與 EKFCHECK 配對都以「逾時真正開始的時間」計算(解鎖前開始的不用夾過的起點)。
    典型機制:最後一筆位置資料(GPS 或 VISP)之後約 7 s 逾時(118、123×2、124×2 共 5 次皆成立);
    FS_EKF_ACTION=0 時只記錄不採取動作。作為「肇因」偵測器在 124 為 0 TP / 0 FP,作為「後果」標記 5/5。"""
    det = "ekf_aiding_timeout"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    st = _aiding_state(ctx)
    if st is None:
        return _na(det, "no_xkf4", "log 沒有 XKF4.TS / SS,無法判斷 EKF 輔助逾時。")
    t, ts, t_a = st
    if t_a is None:
        return _na(det, "no_abs_aiding", "EKF 從未建立絕對水平位置輔助(XKF4.SS bit4),輔助逾時偵測不適用。")
    if int(((t >= iv.t0) & (t <= iv.t1)).sum()) < 10:
        return _na(det, "insufficient_data", "解鎖期間沒有足夠的 XKF4 資料。")
    ref = _get_ref(ctx, a, iv)
    rr = runs((ts != 0) & (t > t_a), t, min_dur=TS_MIN_DUR_S, max_gap=MAX_GAP_XKF_S)
    te, sub, ec = ctx.err_rows()
    next_arm = min([x.t0 for x in ctx.armed if x.t0 > iv.t1] or [np.inf])
    fs_action = ctx.param("FS_EKF_ACTION", None)

    def make(run, t_cond: float | None = None) -> dict:
        """run = (起點, 終點, i0, i1)(可能已夾到解鎖區間);t_cond = 逾時真正開始的時間(算延遲 / 配對 EKFCHECK 用)。"""
        t_s, t_e, i0, i1 = run
        tc = t_s if t_cond is None else t_cond
        chk = [float(x) for x, s, c in zip(te, sub, ec) if s == EKFCHECK_SUBSYS and c == 2 and tc <= x <= tc + EKFCHECK_PAIR_S]
        inav = [float(x) for x, s, c in zip(te, sub, ec) if s == EKFINAV_SUBSYS and c == 1 and tc <= x <= tc + EKFCHECK_PAIR_S]
        t_pos, src = _last_position_data(ctx, tc)
        kinds = int(ts[i0:i1 + 1].max())
        return {"t_start": t_s, "t_end": t_e, "dur": t_e - t_s, "vel_timeout": bool(kinds & 2), "pos_timeout": bool(kinds & 1),
                "ekfcheck_t": chk[0] if chk else None, "ekfinav_t": inav[0] if inav else None,
                "last_pos_t": t_pos, "last_pos_src": src, "lag_s": (tc - t_pos) if t_pos is not None else None}

    # 解鎖區間內的部分(解鎖前就開始的逾時夾到解鎖時刻);夾過之後仍要 ≥2 s,否則只是解鎖前的地面逾時剛好延續了一下
    air = _airborne(ctx, iv, t)
    inflight, carry = [], []
    for (t_s, t_e, i0, i1) in rr:
        if t_s > iv.t1 or t_e < iv.t0:
            continue
        j0 = max(i0, int(np.searchsorted(t, iv.t0, side="left")))
        j1 = min(i1, int(np.searchsorted(t, iv.t1, side="right")) - 1)
        if j1 < j0:
            continue
        if float(t[j1] - t[j0]) < TS_MIN_DUR_S:
            carry.append({"run_start": _r(t_s), "run_end": _r(t_e), "overlap_s": _r(float(t[j1] - t[j0])),
                          "started_before_arm": bool(t_s < iv.t0)})
            continue
        d = make((float(t[j0]), float(t[j1]), j0, j1), t_cond=t_s)
        d["run_start"] = t_s
        d["started_before_arm"] = bool(t_s < iv.t0)
        ar = _air_runs(air, t, j0, j1, TS_MIN_DUR_S)
        d["airborne"] = bool(ar)
        d["air_start"] = ar[0][0] if ar else None
        inflight.append(d)
    post = [make(r) for r in rr if iv.t1 < r[0] <= min(iv.t1 + TS_POST_WINDOW_S, next_arm)]
    base = {"t_abs_aiding": _r(t_a), "fs_ekf_action": fs_action, "ts_max_in_flight": int(ts[_iv_mask(t, iv) & (t > t_a)].max()
            if (_iv_mask(t, iv) & (t > t_a)).any() else 0), "pre_arm_carryover": carry,
            "thresholds": {"min_dur_s": TS_MIN_DUR_S, "ts_mask": TS_MASK, "pair_s": EKFCHECK_PAIR_S},
            "confidence": _CONF[det], "t_warn": _r(ref.t_warn), "t_loss": _r(ref.t_loss), "anchors_local": ref.local}

    def lag_txt(r: dict) -> str:
        if r["lag_s"] is None:
            return ""
        return f"距最後一筆位置資料({r['last_pos_src']} {r['last_pos_t']:.2f}s)約 {r['lag_s']:.1f}s(EKF 約 7 s 沒位置就逾時)"

    out: list[Finding] = []
    crash = _crash_flight(a, ref)
    if inflight:
        air_runs = [r for r in inflight if r["airborne"]]
        in_air = bool(air_runs)
        r0 = air_runs[0] if in_air else inflight[0]
        t0r = r0["t_start"]
        after = ref.t_loss is not None and t0r >= ref.t_loss
        chk = (f";{r0['ekfcheck_t'] - r0['t_start']:.1f}s 後出現 EKFCHECK / FAILSAFE_EKFINAV" if r0["ekfcheck_t"] is not None else "")
        act = "(FS_EKF_ACTION=0 只記錄)" if fs_action == 0 else ""
        pre_arm = (f"(解鎖前 {r0['run_start']:.2f}s 就已逾時、解鎖時仍未恢復)" if r0.get("started_before_arm") else "")
        where = "飛行中" if in_air else "解鎖後在地面(尚未離地)"
        ph = _phase(ref, t0r)
        if after:
            rel = "但已在姿態失控之後,屬後果。"
        elif ph == "no_attitude_event":
            rel = ("此次飛行沒有姿態異常,但飛行中失去位置輔助本身就有風險:請檢查位置來源為何斷掉。" if in_air else
                   "此次飛行沒有姿態異常;起飛前應確認 EKF 已恢復位置輔助。")
        else:
            rel = _rel_sentence(ref, t0r, "這次逾時") + ("。請檢查位置來源為何斷掉。" if ph == "precursor" else "。")
        n_gnd = len(inflight) - len(air_runs)
        more = f"(另有 {n_gnd} 段在解鎖後的地面)" if in_air and n_gnd else ""
        txt = (f"{where} {t0r:.2f}s 起 EKF 位置/速度輔助逾時 {r0['dur']:.1f}s{pre_arm}{more}," + lag_txt(r0) + chk + act +
               "。" + rel)
        if in_air:
            status, sev, cx = HIT, ("info" if after else "medium"), ("in_flight_after_loss" if after else "in_flight")
        else:
            status, sev, cx = SOFT, "info", "armed_ground"
        out.append(Finding(det, DOMAIN, status, sev, t_onset=t0r, t_alarm=t0r + TS_MIN_DUR_S, context=cx,
                           values=dict(base, runs=[{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()} for r in inflight],
                                       n_air_runs=len(air_runs), phase=ph), text_zh=txt))
    else:
        co = (f"(解鎖前就開始的逾時在解鎖後 {max(c['overlap_s'] for c in carry):.2f}s 內就恢復,未達 {TS_MIN_DUR_S:.0f}s,不算)"
              if carry else "")
        out.append(_clear_or_short(det, iv, Finding(
            det, DOMAIN, CLEAR, "info", context="none", values=base,
            text_zh=f"解鎖期間 EKF 位置 / 速度輔助沒有 ≥{TS_MIN_DUR_S:.0f}s 的逾時(XKF4.TS&3;絕對輔助自 {t_a:.1f}s 建立){co}。")))
    for r in post[:1]:
        dt = r["t_start"] - iv.t1
        chk = (f";{r['ekfcheck_t'] - r['t_start']:.1f}s 後 EKFCHECK / FAILSAFE_EKFINAV" if r["ekfcheck_t"] is not None else "")
        out.append(Finding(det, DOMAIN, HIT, "info", t_onset=r["t_start"], t_alarm=r["t_start"] + TS_MIN_DUR_S, context="post_disarm",
                           values=dict(base, runs=[{k: (_r(v) if isinstance(v, float) else v) for k, v in x.items()} for x in post],
                                       after_disarm_s=_r(dt), phase="post_crash" if crash else "post_disarm"),
                           text_zh=f"EKF 位置/速度輔助在上鎖之後 {dt:.1f}s({r['t_start']:.2f}s)才逾時 {r['dur']:.1f}s," + lag_txt(r) + chk +
                                   ("。這是位置來源中斷後的連鎖反應(失事後果),不是失事原因。" if crash else
                                    "。這是上鎖後位置來源停止的正常連鎖反應,與飛行無關。")))
    return out


# ====================================================================================
# 4. ekf_testratio_gate
# ====================================================================================

@guarded("ekf_testratio_gate", DOMAIN)
def ekf_testratio_gate(ctx: Ctx, a: Anchors):
    """XKF4 SV/SP/SH(速度 / 位置 / 高度新息測試比)是否失控。
    gate-fail = 三條 lane 的 max(SV,SP,SH) 取中位數 ≥1.0 持續 ≥1.0 s(t>40 s,解鎖期間)。每一段都分類:段內離地 ≥1.0 s
    → 空中;只要有空中段 → HIT(t_onset = 第一個空中段;地面段留在 values.runs)/ 全部在解鎖後的地面 → SOFT armed_ground。
    沒有 gate-fail 時看峰值:失事前 15 s 與 t_warn 前 60 s 兩個窗,三個比值都 <0.5 且 XKF4.FS(遮掉開機 1 s 的 128)=0
    → CLEAR「EKF 失事前健康」(解鎖 <5 s → NA);否則 SOFT。attribution:run 內「GPS 有定位時」的 GPA.SAcc ≥2 m/s → GPS 品質,
    否則 VIO / 其他(GPS 沒定位時 SAcc 是舊值,不採用,文字改說 GPS 無定位)。
    注意 XKF4.SP 在 VISP.PErr 很大時幾乎恆為 0,抓不到 VIO 位置不一致(那是 ekf_vio_disagreement 的工作)。"""
    det = "ekf_testratio_gate"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    lanes = _lanes(ctx, "XKF4", ("SV", "SP", "SH"))
    if not lanes:
        return _na(det, "no_xkf4", "log 沒有 XKF4 測試比(SV/SP/SH),無法判斷 EKF 一致性。")
    t = lanes[0]["t"]
    if int(((t >= iv.t0) & (t <= iv.t1) & (t > GATE_MIN_T)).sum()) < 25:
        return _na(det, "insufficient_data", "解鎖期間(t>40 s)沒有足夠的 XKF4 資料。")
    ref = _get_ref(ctx, a, iv)
    mx, comp = [], {"SV": [], "SP": [], "SH": []}
    for l in lanes:
        i, has = _hold_idx(l["t"], t)
        per = {k: np.where(has, l[k][i], np.nan) for k in ("SV", "SP", "SH")}
        for k in comp:
            comp[k].append(per[k])
        mx.append(np.fmax(np.fmax(per["SV"], per["SP"]), per["SH"]))
    m3 = _nanmedian_rows(np.stack(mx))
    arm = _iv_mask(t, iv) & (t > GATE_MIN_T)
    rr = runs(arm & (m3 >= GATE_RATIO), t, min_dur=GATE_MIN_DUR_S, max_gap=MAX_GAP_XKF_S)

    def win(lo: float, hi: float) -> np.ndarray:
        return (t >= max(lo, iv.t0)) & (t <= min(hi, iv.t1)) & (t > GATE_MIN_T)

    def peaks(lo: float, hi: float) -> dict:
        w = win(lo, hi)
        if not w.any():
            return {}
        out = {}
        for k in comp:
            vals = [float(np.nanmax(x[w])) for x in comp[k] if np.isfinite(x[w]).any()]
            out[k] = _r(max(vals)) if vals else None
        return out

    t_c = a.t_detect if a.t_detect is not None else iv.t1
    t_c = min(t_c, iv.t1)
    p15 = peaks(t_c - GATE_PRE_WINDOW_S, t_c)
    t_ref, w0 = _std_window(iv, ref)
    pP = peaks(w0, t_ref + 0.5)
    fs_bad, fs_t = False, None
    for l in lanes:
        if "FS" not in l:
            continue
        fsv = _int_bits(l["FS"]) & ~GATE_FS_IGNORE
        bad = (fsv != 0) & (l["t"] >= max(iv.t0, GATE_FS_SKIP_S)) & (l["t"] <= iv.t1)
        if bad.any():
            fs_bad = True
            t1st = float(l["t"][np.flatnonzero(bad)[0]])
            fs_t = t1st if fs_t is None else min(fs_t, t1st)
    gpa = ctx.table("GPA")
    ipn = ctx.table("XKF3")
    base = {"pre_crash_max": p15, "pre_crash_window": [_r(t_c - GATE_PRE_WINDOW_S), _r(t_c)],
            "precursor_max": pP, "precursor_window": [_r(w0), _r(t_ref + 0.5)], "fs_nonzero": fs_bad,
            "n_lanes": len(lanes), "thresholds": {"gate": GATE_RATIO, "min_dur_s": GATE_MIN_DUR_S, "healthy_max": GATE_HEALTHY_MAX},
            "confidence": _CONF[det], "t_warn": _r(ref.t_warn), "t_loss": _r(ref.t_loss), "anchors_local": ref.local}
    if rr:
        # 每一段都分類:段內離地(EV28 之後)≥1.0 s → 空中;否則 = 解鎖後的地面。只要有空中段就以第一個空中段為 HIT
        # (起飛前地面上的短暫 GPS / EKF gate fail 很常見,不能讓它蓋掉之後真正的空中發散);全部在地面才回 SOFT。
        air = _airborne(ctx, iv, t)
        is_air = [bool(_air_runs(air, t, r[2], r[3], GATE_MIN_DUR_S)) for r in rr]
        air_rr = [r for r, f in zip(rr, is_air) if f]
        gnd_rr = [r for r, f in zip(rr, is_air) if not f]
        in_air = bool(air_rr)
        t_s, t_e, i0, i1 = air_rr[0] if in_air else rr[0]
        # 歸因:GPA.SAcc 只在 GPS 有定位(Status>=3 且 NSats>=6)時才有意義 —— 沒定位時 SAcc 是舊值(124:934.6–1262 s 恆為 0.0)
        sacc, gps_fix, sacc_avail = None, None, False
        if gpa is not None and "SAcc" in gpa:
            w = (gpa["t"] >= t_s) & (gpa["t"] <= t_e) & np.isfinite(gpa["SAcc"])
            sacc_avail = bool(w.any())
            gps = _gps_ok_series(ctx)
            if gps is not None:
                w = w & _gps_ok_on(gpa["t"], gps)
                gps_fix = bool(w.any())
            sacc = float(np.max(gpa["SAcc"][w])) if w.any() else None
        gap = None
        if ipn is not None and "IPN" in ipn and "IPE" in ipn:
            w = (ipn["t"] >= t_s) & (ipn["t"] <= t_e)
            hv = np.hypot(ipn["IPN"][w], ipn["IPE"][w])
            hv = hv[np.isfinite(hv)]
            gap = float(np.max(hv)) if hv.size else None
        cause = "gps_quality" if (sacc is not None and sacc >= GATE_SACC_QUALITY) else "vio_or_other"
        if cause == "gps_quality":
            att = f",同時 GPS 速度精度 SAcc 達 {sacc:.1f} m/s,歸因 GPS 品質差"
        elif sacc is not None:
            att = ",GPS 有定位且速度精度正常,比較像 VIO / 其他來源的問題"
        elif gps_fix is False and sacc_avail:
            att = ",當時 GPS 沒有定位(SAcc 是舊值,不採用),比較像 VIO / 其他來源的問題"
        else:
            att = ",沒有可用的 GPS 精度資料(GPA.SAcc),無法歸因"
        peak = float(np.nanmax(m3[i0:i1 + 1]))
        ph = _phase(ref, t_s)
        more = ""
        if in_air and gnd_rr:
            more = f"(此前解鎖後在地面另有 {len(gnd_rr)} 段,最早 {gnd_rr[0][0]:.2f}s)" if gnd_rr[0][0] < t_s else \
                   f"(另有 {len(gnd_rr)} 段在解鎖後的地面)"
        txt = (f"EKF 測試比失控:{t_s:.2f}–{t_e:.2f}s 三條 lane 的 max(SV,SP,SH) 中位數最高 {peak:.2f}(≥{GATE_RATIO} 代表量測被拒絕)"
               + att + ("" if in_air else "(解鎖在地面,尚未離地)") + more + "。" + _rel_sentence(ref, t_s, "這段測試比失控") + "。")
        return Finding(det, DOMAIN, HIT if in_air else SOFT, "high" if in_air else "medium", t_onset=t_s, t_alarm=t_s + GATE_MIN_DUR_S,
                       context="air" if in_air else "armed_ground",
                       values=dict(base, runs=[{"t_start": _r(x[0]), "t_end": _r(x[1]), "air": f} for x, f in list(zip(rr, is_air))[:6]],
                                   n_runs=len(rr), n_air_runs=len(air_rr), peak_median=_r(peak),
                                   sacc_max=_r(sacc), gps_fix_in_run=gps_fix, ipn_ipe_max_m=_r(gap), attribution=cause, phase=ph),
                       text_zh=txt)
    allp = [v for d in (p15, pP) for v in d.values() if v is not None]
    worst = max(allp) if allp else None
    if worst is None:
        return _na(det, "insufficient_data", "解鎖期間(t>40 s)沒有足夠的 XKF4 資料。")
    if worst >= GATE_HEALTHY_MAX or fs_bad:
        # t_onset = 兩個評估窗內 m3 第一次 ≥ 健康上限的時間(或第一個非零 FS);各 lane 取最大以免漏掉單一 lane 的峰值
        both = win(t_c - GATE_PRE_WINDOW_S, t_c) | win(w0, t_ref + 0.5)
        lane_max = np.fmax.reduce(np.stack(mx), axis=0)
        hi = np.flatnonzero(both & (lane_max >= GATE_HEALTHY_MAX))
        cands = [float(t[hi[0]])] if hi.size else []
        if fs_t is not None:
            cands.append(fs_t)
        t_on = min(cands) if cands else None
        if t_on is None:      # 只有 fs_bad 但不在窗內不會發生;保險起見退回窗起點
            t_on = float(t[both][0]) if both.any() else float(iv.t0)
        return Finding(det, DOMAIN, SOFT, "medium", t_onset=t_on, t_alarm=t_on, context="near_gate",
                       values=dict(base, phase=_phase(ref, t_on)),
                       text_zh=f"EKF 測試比沒有持續失控,但峰值已達 {worst:.2f}(健康 <{GATE_HEALTHY_MAX},{t_on:.2f}s 起)"
                               + (",且 XKF4.FS 有非零故障位元" if fs_bad else "") + ",EKF 並非完全健康。")
    pre = "失事前" if _crash_flight(a, ref) else "上鎖前"
    return _clear_or_short(det, iv, Finding(
        det, DOMAIN, CLEAR, "info", context="healthy", values=base,
        text_zh=f"EKF 一致性正常:{pre} 15 s 的 SV/SP/SH 峰值 = {p15.get('SV')}/{p15.get('SP')}/{p15.get('SH')}"
                f"(門檻 {GATE_RATIO},健康 <{GATE_HEALTHY_MAX}),解鎖期間沒有 gate-fail,XKF4.FS=0。"
                "EKF 自己覺得一切正常 → 不是 EKF 數值發散。"))


# ====================================================================================
# 5. ekf_vio_disagreement(常態)
# ====================================================================================

def _perr_latched(visp: dict, iv: ArmedInterval):
    """VISP.PErr ≥5 m 連續 ≥30 s 的區段 [(t0, t1, 中位數)]。"""
    if "PErr" not in visp:
        return []
    m = _iv_mask(visp["t"], iv)
    t, p = visp["t"][m], visp["PErr"][m]
    out = []
    for (t_s, t_e, i0, i1) in runs(p >= VD_PERR_M, t, min_dur=VD_PERR_MIN_S, max_gap=VD_PERR_MAX_GAP_S):
        out.append((t_s, t_e, float(np.median(p[i0:i1 + 1]))))
    return out


@guarded("ekf_vio_disagreement", DOMAIN)
def ekf_vio_disagreement(ctx: Ctx, a: Anchors):
    """(常態)EKF 位置與 VIO 位置長期不一致。gap = (XKF3.IPN, XKF3.IPE) = EKF 位置 − VIO 位置(已驗證與 XKF1.PN − VISP.PX 相關 0.9999)。
    只看「使用 VIO 組(POSXY=6)、VISP 新鮮(年齡 <0.5 s)、離上次來源切換 / VISP.Rst ≥30 s」的樣本(使用中的組未知 → NA);
    窗 = 這些有效樣本的最後 60 s
    (終點 ≤ t_warn,沒有姿態異常則 ≤ 上鎖前 5 s)。旗標:median|gap| >1.0 m 或 p95 >3 m 或隱含新息 sigma >3 m。
    另報 VISP.PErr 是否被鎖在 ≥5 m 超過 30 s(失事:8.591 m 持續 209.7 s,EKF 幾乎不採信 VIO 位置)。
    信心低:它分不出「失事」與「VIO 很重的健康飛行」,只能當貢獻因素;機制(漂移是否導致傾斜指令)未證實。"""
    det = "ekf_vio_disagreement"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    visp = _visp(ctx)
    if visp is None:
        return _na(det, "no_visp", "此 log 沒有 VISP(VIO 位置)資料,EKF 與 VIO 位置比較不適用。")
    x3s = _lanes(ctx, "XKF3", ("IPN", "IPE"))
    if not x3s:
        return _na(det, "no_xkf3", "log 沒有 XKF3 位置新息(IPN/IPE)。")
    steps = _set_steps(ctx)
    if steps.source == "assumed":
        return _na(det, "set_unknown", "log 沒有來源組紀錄(XKFS / EV 85-87),無法確認 XKF3 位置新息是不是對 VIO 算的,"
                   "EKF 與 VIO 位置比較不適用。", set_source=steps.source)
    ref = _get_ref(ctx, a, iv)
    sets, _ = _sets(ctx)
    x3 = x3s[0]
    t = x3["t"]
    t_ref, _ = _std_window(iv, ref)
    cand = _iv_mask(t, iv, NAV_ARM_SKIP_S) & (t <= t_ref)
    set_n = steps.at(t)
    vio_on = _flag_by_set(sets, set_n, _uses_vio)
    age = _visp_age(visp["t"], t)
    sw_t = np.array([s[0] for s in steps.switches()], float)
    since_sw = np.full(t.size, np.inf)
    if sw_t.size:
        i, has = _hold_idx(sw_t, t)
        since_sw = np.where(has, t - sw_t[i], np.inf)
    rst_t = np.zeros(0)
    if "Rst" in visp:
        rst_t = visp["t"][1:][np.diff(visp["Rst"]) != 0]
    since_rst = np.full(t.size, np.inf)
    if rst_t.size:
        i, has = _hold_idx(rst_t, t)
        since_rst = np.where(has, t - rst_t[i], np.inf)
    valid = cand & vio_on & (age < VD_FRESH_AGE_S) & (since_sw >= VD_HOLD_S) & (since_rst >= VD_HOLD_S)         & np.isfinite(x3["IPN"]) & np.isfinite(x3["IPE"])
    perr = _perr_latched(visp, iv)
    if not valid.any():
        return _na(det, "vio_not_active", "解鎖期間沒有「使用 VIO 組且 VISP 新鮮(離切換 / VIO 重置 ≥30 s)」的樣本,EKF 與 VIO 位置比較不適用。",
                   perr_latched=[[_r(p[0]), _r(p[1]), _r(p[2])] for p in perr])
    t_end = float(t[valid][-1])
    win = valid & (t >= t_end - NAV_WINDOW_S)
    dt = float(np.median(np.diff(t))) if t.size > 2 else 0.04
    valid_s = float(win.sum() * dt)
    if valid_s < VD_MIN_VALID_S:
        return _na(det, "insufficient_data", f"有效的 VIO 比較樣本只有 {valid_s:.1f}s(<{VD_MIN_VALID_S:.0f}s),不足以判斷。")
    gx, gy = x3["IPN"][win], x3["IPE"][win]
    g = np.hypot(gx, gy)
    med, p95, frac = float(np.median(g)), float(np.percentile(g, 95)), float(np.mean(g > 1.0))
    # 隱含新息 sigma = sqrt(median((IPN²+IPE²)/(25·SP))),SP>0.01 的樣本
    sigma = None
    x4s = _lanes(ctx, "XKF4", ("SP",))
    if x4s:
        i, has = _hold_idx(x4s[0]["t"], t[win])
        sp = np.where(has, x4s[0]["SP"][i], 0.0)
        ok = sp > VD_SP_MIN
        if ok.sum() >= VD_SIGMA_MIN_N:
            sigma = float(np.sqrt(np.median((g[ok] ** 2) / (25.0 * sp[ok]))))
    perr_in = [p for p in perr if p[0] <= t_end and p[1] >= t_end - NAV_WINDOW_S]
    pe_samples = visp["PErr"][_iv_mask(visp["t"], iv) & (visp["t"] >= t_end - NAV_WINDOW_S) & (visp["t"] <= t_end)] if "PErr" in visp else np.zeros(0)
    flagged = med > VD_MED_M or p95 > VD_P95_M or (sigma is not None and sigma > VD_SIGMA_M)
    t_win0 = float(t[win][0])
    base = {"median_gap_m": _r(med), "p95_gap_m": _r(p95), "frac_gt_1m": _r(frac), "median_abs_ipn_m": _r(np.median(np.abs(gx))),
            "median_abs_ipe_m": _r(np.median(np.abs(gy))), "implied_sigma_m": _r(sigma), "window": [_r(t_win0), _r(t_end)],
            "valid_s": _r(valid_s, 1), "perr_median_m": _r(np.median(pe_samples)) if pe_samples.size else None,
            "perr_latched": [[_r(p[0]), _r(p[1]), _r(p[2])] for p in perr], "perr_latched_in_window": bool(perr_in),
            "thresholds": {"median_m": VD_MED_M, "p95_m": VD_P95_M, "sigma_m": VD_SIGMA_M, "perr_m": VD_PERR_M},
            "confidence": _CONF[det], "standing": True, "t_warn": _r(ref.t_warn), "anchors_local": ref.local}
    if flagged:
        on = np.flatnonzero(g > VD_MED_M)
        t_on = float(t[win][on[0]]) if on.size else t_win0
        pe_txt = (f"VIO 位置不確定度 PErr 被鎖在 {perr_in[0][2]:.2f} m({perr_in[0][1] - perr_in[0][0]:.0f}s),EKF 幾乎不採信 VIO 位置;" if perr_in else "")
        return Finding(det, DOMAIN, HIT, "medium", t_onset=t_on, t_alarm=t_end, context="standing", values=base,
                       text_zh=(f"{t_win0:.0f}–{t_end:.0f}s 使用 VIO 組期間,EKF 位置與 VIO 位置相差中位數 {med:.2f} m(p95 {p95:.1f} m,"
                                f"{frac * 100:.0f}% 樣本 >1 m)。" + pe_txt +
                                "定位模式依 EKF 位置定點,實際位置會與 VIO 不同,觸地擾動易放大成漂移。這是常態風險(貢獻因素,信心低),"
                                "不是單獨的失事證據;請檢查伴隨電腦輸出的共變異數 / PErr 是否過度保守。"))
    return Finding(det, DOMAIN, CLEAR, "info", context="standing", values=base,
                   text_zh=f"{t_win0:.0f}–{t_end:.0f}s 使用 VIO 組期間 EKF 位置與 VIO 位置一致:相差中位數 {med:.2f} m、p95 {p95:.1f} m"
                           f"(門檻 {VD_MED_M} / {VD_P95_M} m)。")


# ====================================================================================
# 6. nav_velocity_gps_fallback_risk(常態)
# ====================================================================================

def _latched_mask(ctx: Ctx, grid: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, list]:
    """XKF3.IVN / IVE 連續 ≥30 s 不變(沒有水平速度量測被融合)的區段,並標到 grid 上。"""
    x3s = _lanes(ctx, "XKF3", ("IVN", "IVE"))
    if not x3s:
        return np.zeros(grid.size, bool), []
    x3 = x3s[0]
    m = (x3["t"] >= lo) & (x3["t"] <= hi)
    t, vn, ve = x3["t"][m], x3["IVN"][m], x3["IVE"][m]
    if t.size < 3:
        return np.zeros(grid.size, bool), []
    const = np.concatenate([[False], (np.abs(np.diff(vn)) < NV_LATCH_EPS) & (np.abs(np.diff(ve)) < NV_LATCH_EPS)])
    spans = runs(const, t, min_dur=NV_LATCH_MIN_S, max_gap=MAX_GAP_XKF_S)
    out = np.zeros(grid.size, bool)
    for (t_s, t_e, _, _) in spans:
        out |= (grid >= t_s) & (grid <= t_e)
    return out, [(float(s[0]), float(s[1])) for s in spans]


@guarded("nav_velocity_gps_fallback_risk", DOMAIN)
def nav_velocity_gps_fallback_risk(ctx: Ctx, a: Anchors):
    """(常態)導航設定風險:使用中的來源組沒有水平速度來源(EK3_SRCn_VELXY=0,或 XKF3.IVN/IVE 凍結 ≥30 s)而 GPS 又沒有定位 → 無備援。
    窗 = t_warn 前 60 s(沒有姿態異常:上鎖前 5 s 之前的 60 s)。窗內累積:
      HIT  無速度來源且 GPS 無定位 ≥10 s;
      SOFT 無速度來源 ≥10 s 但 GPS 仍可用(有備援);
      CLEAR 其餘(有速度來源)。
    這是風險指標不是鑑別器:124 飛行 1 的 VIO 組期間也有 67 s 無 GPS 沒出事(窗內不成立 → CLEAR)。
    使用中的組未知(沒有 XKFS / EV 85-87,set_source='assumed'):以「任一個有設定的組」的最壞情況計算;只靠這個假設
    才成立的風險最多 SOFT(IVN/IVE 凍結是實測,仍可 HIT);CLEAR 只在最壞情況也沒有風險時才給。
    EK3_SRC_OPTIONS bit0(FuseAllVelocities)=1 時 GPS 可用仍可能融合速度,值內另註。"""
    det = "nav_velocity_gps_fallback_risk"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    steps = _set_steps(ctx)
    ref = _get_ref(ctx, a, iv)
    sets, opt = _sets(ctx)
    w1, w0 = _std_window(iv, ref)
    if w1 - w0 < NV_RISK_MIN_S:
        return _na(det, "insufficient_data", "解鎖時間太短,無法評估導航備援風險。")
    s_all = ctx.series("XKFS.SS")
    gt = s_all[0] if s_all is not None else np.arange(iv.t0, w1, 0.04)
    grid = gt[(gt >= iv.t0 + NAV_ARM_SKIP_S) & (gt <= w1)]
    if grid.size < 25:
        return _na(det, "insufficient_data", "解鎖期間沒有足夠的 XKFS / 來源組資料。")
    dt = np.minimum(np.diff(grid, append=grid[-1] + 0.04), 0.5)
    set_n = steps.at(grid)
    set_known = steps.source != "assumed"
    if set_known:
        vel0 = _flag_by_set(sets, set_n, lambda ss, n: _code(ss, n, "VELXY") == 0)
        vio_pos = _flag_by_set(sets, set_n, _uses_vio)
    else:
        # 使用中的組未知(沒有 XKFS / EV 85-87):以「任一個有設定的組」的最壞情況判斷;只靠這個假設成立的風險最多 SOFT
        cand_sets = [n for n in (1, 2, 3) if _code(sets, n, "POSXY") != 0 or _code(sets, n, "VELXY") != 0] or [1]
        vel0 = np.full(grid.size, any(_code(sets, n, "VELXY") == 0 for n in cand_sets))
        vio_pos = np.full(grid.size, any(_uses_vio(sets, n) for n in cand_sets))
    lat, spans = _latched_mask(ctx, grid, iv.t0, w1)
    gps = _gps_ok_series(ctx)
    gps_ok = _gps_ok_on(grid, gps)
    unaided = vel0 | lat
    risk = unaided & ~gps_ok
    inw = grid >= w0
    risk_s, unaided_s = float(dt[inw & risk].sum()), float(dt[inw & unaided].sum())
    # 末端連續的風險段起點(窗終點仍在風險中)
    rr = runs(risk, grid, min_dur=0.0, max_gap=MAX_GAP_XKF_S)
    cur = rr[-1] if rr and rr[-1][1] >= grid[-1] - 0.5 else None
    # GPS 無定位連續段(整個解鎖到窗終點)
    nog = runs(~gps_ok, grid, min_dur=NV_NOGPS_RUN_S, max_gap=MAX_GAP_XKF_S)
    pure_vio = [(s, e) for (s, e, i0, i1) in nog if vio_pos[i0:i1 + 1].mean() > 0.5]
    # EKF 與 VIO 速度差(VISV 新鮮才比)
    dv = None
    x1s = _lanes(ctx, "XKF1", ("VN", "VE"))
    vv = ctx.table("VISV")
    if x1s and vv is not None and "VX" in vv and vv["t"].size > 2:
        x1 = x1s[0]
        w = (x1["t"] >= w0) & (x1["t"] <= w1)
        if w.any():
            age = _visp_age(vv["t"], x1["t"][w])
            f = age < NV_VEL_FRESH_S
            if f.sum() >= 25:
                i, _ = _hold_idx(vv["t"], x1["t"][w][f])
                dn, de = x1["VN"][w][f] - vv["VX"][i], x1["VE"][w][f] - vv["VY"][i]
                dv = {"mean_n": _r(dn.mean()), "mean_e": _r(de.mean()), "rms": _r(np.sqrt(np.mean(dn ** 2 + de ** 2)))}
    if set_known:
        codes = {str(int(n)): _code(sets, int(n), "VELXY") for n in np.unique(set_n)}
        active = sorted(int(n) for n in np.unique(set_n[inw])) if inw.any() else []
    else:
        codes = {str(n): _code(sets, n, "VELXY") for n in cand_sets}
        active = []
    # GPS 失去時間(窗終點時 GPS 仍不可用才有):unusable = 最後一筆可用樣本之後的第一筆;no_fix = 其後第一筆 Status<3
    gps_unusable_t = gps_nofix_t = None
    if gps is not None and not gps_ok[-1]:
        tg, ok = gps
        st = ctx.table("GPS")["Status"]
        okb = np.flatnonzero(ok & (tg <= w1))
        k0 = int(okb[-1]) + 1 if okb.size else 0
        if k0 < tg.size and tg[k0] <= w1:
            gps_unusable_t = float(tg[k0])
            nf = np.flatnonzero((st[k0:] < GPS_MIN_STATUS) & (tg[k0:] <= w1))
            gps_nofix_t = float(tg[k0 + nf[0]]) if nf.size else None
    t_det = a.t_detect if a.t_detect is not None else (ref.t_loss if ref.t_loss is not None else None)
    base = {"window": [_r(w0), _r(w1)], "risk_s": _r(risk_s, 1), "unaided_s": _r(unaided_s, 1),
            "active_sets": active, "velxy_codes": codes, "latched_spans": [[_r(s), _r(e)] for s, e in spans[-4:]],
            "risk_run_start_t": _r(cur[0]) if cur else None,
            "gps_lost_t": _r(gps_unusable_t), "gps_nofix_t": _r(gps_nofix_t),
            "gps_lost_to_detect_s": _r(t_det - gps_unusable_t) if (t_det is not None and gps_unusable_t is not None) else None,
            "pure_vio_no_gps": [[_r(s), _r(e)] for s, e in pure_vio[-3:]],
            "gps_table": gps is not None, "fuse_all_velocities": bool(int(opt) & 1), "ekf_minus_vio_vel": dv,
            "set_source": steps.source, "set_unknown": not set_known,
            "thresholds": {"risk_min_s": NV_RISK_MIN_S, "latch_min_s": NV_LATCH_MIN_S, "nogps_run_s": NV_NOGPS_RUN_S},
            "confidence": _CONF[det], "standing": True, "t_warn": _r(ref.t_warn), "anchors_local": ref.local}
    no_vel_sets = [n for n in (active if set_known else cand_sets) if _code(sets, n, "VELXY") == 0]
    unk_txt = ("log 沒有來源組紀錄(XKFS / EV 85-87),無法確認實際使用哪一組,以最壞情況(" +
               "、".join(f"第 {n} 組 VELXY={_code(sets, n, 'VELXY')}" for n in cand_sets) + ")估計") if not set_known else ""
    fix_txt = ("建議把 " + "、".join(f"EK3_SRC{n}_VELXY" for n in no_vel_sets) + " 設為 6(ExternalNav 速度)"
               if no_vel_sets else "建議確認 VIO 速度有被融合")
    if risk_s >= NV_RISK_MIN_S and gps is not None:
        t_on = float(cur[0]) if cur else float(grid[inw & risk][0])
        t_on = max(t_on, iv.t0)
        if gps_nofix_t is not None and gps_unusable_t is not None and gps_nofix_t - gps_unusable_t > 0.05:
            gps_lost = f"GPS 自 {gps_nofix_t:.1f}s 起沒有 3D 定位({gps_unusable_t:.1f}s 起衛星已 <{GPS_MIN_SATS})"
        elif gps_unusable_t is not None:
            gps_lost = f"GPS 自 {gps_unusable_t:.1f}s 起沒有可用定位"
        else:
            gps_lost = "GPS 在窗內沒有可用定位"
        vio_txt = ("使用 VIO 位置、" if any(_uses_vio(sets, n) for n in active) else "")
        dv_txt = f"EKF 與 VIO 速度差 rms {dv['rms']:.2f} m/s," if dv and dv["rms"] is not None else ""
        risk_lat_s = float(dt[inw & lat & ~gps_ok].sum())
        if not set_known and risk_lat_s < NV_RISK_MIN_S:     # 只有「假設在用 VELXY=0 的組」才成立 → 弱證據
            return Finding(det, DOMAIN, SOFT, "medium", t_onset=t_on, t_alarm=t_on + NV_RISK_MIN_S, context="standing",
                           values=dict(base, risk_latched_s=_r(risk_lat_s, 1)),
                           text_zh=(f"{w0:.0f}–{w1:.0f}s 內若使用的是沒有水平速度來源的來源組,就有 {risk_s:.0f}s 處於"
                                    f"「無水平速度來源、GPS 又沒定位」的狀態({gps_lost});但{unk_txt},只能列為弱證據。"
                                    "速度新息 IVN/IVE 沒有凍結 ≥30 s。" + fix_txt + "。"))
        return Finding(det, DOMAIN, HIT, "medium", t_onset=t_on, t_alarm=t_on + NV_RISK_MIN_S, context="standing", values=base,
                       text_zh=(f"{w0:.0f}–{w1:.0f}s 內有 {risk_s:.0f}s 的導航處於「無水平速度來源、GPS 又沒定位」的狀態:" + vio_txt +
                                f"來源組的 EK3_SRCn_VELXY=0(或速度新息 IVN/IVE 凍結),{gps_lost},{dv_txt}"
                                "水平速度只能靠慣性推算,VIO 一變差就完全沒有備援。這是常態風險(貢獻因素),不是單獨的失事證據;"
                                + fix_txt + ",並確認 VIO 狀態再切換。"))
    if unaided_s >= NV_RISK_MIN_S:
        uu = runs(unaided, grid, min_dur=0.0, max_gap=MAX_GAP_XKF_S)
        t_on = float(uu[-1][0]) if uu and uu[-1][1] >= grid[-1] - 0.5 else float(grid[inw & unaided][0])
        t_on = max(t_on, iv.t0)
        if gps is None:     # 沒有 GPS 資料:不能確認有沒有 GPS 備援 → 最多 SOFT
            return Finding(det, DOMAIN, SOFT, "medium", t_onset=t_on, t_alarm=t_on + NV_RISK_MIN_S, context="standing",
                           values=dict(base, data_missing="GPS"),
                           text_zh=f"{w0:.0f}–{w1:.0f}s 內有 {unaided_s:.0f}s 使用沒有水平速度來源的來源組(EK3_SRCn_VELXY=0 或 IVN/IVE 凍結),"
                                   f"而 log 沒有 GPS 資料,無法確認是否有 GPS 備援" + (f";{unk_txt}" if unk_txt else "") + f";{fix_txt}。")
        return Finding(det, DOMAIN, SOFT, "info", t_onset=t_on, t_alarm=t_on + NV_RISK_MIN_S, context="standing", values=base,
                       text_zh=f"{w0:.0f}–{w1:.0f}s 內有 {unaided_s:.0f}s 使用沒有水平速度來源的來源組,但 GPS 仍可用,有備援;風險較低"
                               + (f"({unk_txt})" if unk_txt else "") + "。")
    return Finding(det, DOMAIN, CLEAR, "info", context="standing", values=base,
                   text_zh=f"{w0:.0f}–{w1:.0f}s 內使用中的來源組都有水平速度來源(或 GPS 可用),沒有「無速度來源且無 GPS」的備援風險"
                           + ("(來源組未知,以所有有設定的組的最壞情況檢查)" if not set_known else "") + "。")


# ====================================================================================
# 7. source_switch_position_jump
# ====================================================================================

def _window_mean(tb: dict, cols: tuple[str, ...], lo: float, hi: float):
    """[lo, hi] 內各欄平均與平均時間;樣本不足回傳 None。"""
    w = (tb["t"] >= lo) & (tb["t"] <= hi)
    for c in cols:
        w &= np.isfinite(tb[c])
    if int(w.sum()) < JP_MIN_SAMPLES:
        return None
    return float(tb["t"][w].mean()), [float(tb[c][w].mean()) for c in cols]


@guarded("source_switch_position_jump", DOMAIN)
def source_switch_position_jump(ctx: Ctx, a: Anchors):
    """來源切換時 EKF 位置跳變量 = VIO 與 GPS 座標系的差(度量用,拿來驗證 VIO 精度;不是失事鑑別器,從不給 HIT)。
    每次切換:P0 = 切換前 [−0.3,−0.04] s 的 XKF1.PN/PE 平均,P1 = 切換後 [+0.3,+0.8] s(被下一次切換 / 上鎖截短;
    上鎖後的位置是失事後的垃圾值,不採用;剩不到 2 筆樣本則該次不量)的平均;
    jump = |(P1−P0) − 0.5·(V0+V1)·Δt|,三條 lane 取中位數。另以 XKF4.OFN/OFE 的切換前後差交叉驗證。
    分類 GPS↔GPS / VIO↔GPS。最大跳變 ≥1 m → SOFT(info)、≥3 m → SOFT(medium);<1 m → CLEAR;沒有切換 → NA。
    124 正常切換 1.37 / 5.14 / 4.34 m 都沒事;失事那兩次因為 GPS 不存在、VISP 已停,反而只有 0.18 / 0.39 m。"""
    det = "source_switch_position_jump"
    iv = a.interval
    if iv is None:
        return _not_armed(det)
    steps = _set_steps(ctx)
    if steps.source == "assumed":
        return _na(det, "no_xkfs", "log 沒有 XKFS / 來源組切換事件,無法量測切換跳變。")
    lanes = _lanes(ctx, "XKF1", ("PN", "PE", "VN", "VE"))
    if not lanes:
        return _na(det, "no_xkf1", "log 沒有 XKF1(EKF 位置 / 速度)。")
    sets, _ = _sets(ctx)
    sw_all = steps.switches()
    sw = [(i, s) for i, s in enumerate(sw_all) if iv.t0 <= s[0] <= iv.t1]
    if not sw:
        return _na(det, "no_switch", "解鎖期間沒有 EKF 來源組切換,沒有跳變量可量。")
    ref = _get_ref(ctx, a, iv)
    of4 = _lanes(ctx, "XKF4", ("OFN", "OFE"))
    rows = []
    for i, (ts, s_from, s_to) in sw:
        t_prev = sw_all[i - 1][0] if i > 0 else -np.inf
        t_next = sw_all[i + 1][0] if i + 1 < len(sw_all) else np.inf
        # 窗一律夾在解鎖區間內:上鎖後的 EKF 位置是失事後的垃圾值(pitfalls),不能混進跳變量
        pre = (max(ts - JP_PRE_S[0], t_prev + 0.04, iv.t0), ts - JP_PRE_S[1])
        post = (ts + JP_POST_S[0], min(ts + JP_POST_S[1], t_next - 0.04, iv.t1))
        short = post[1] - post[0] < 0.1
        if short:      # 停留太短(或離上鎖太近):改用「切換後立刻到下一次切換 / 上鎖前」;樣本不足 2 筆則該次不量
            post = (ts + 0.04, min(t_next - 0.04 if np.isfinite(t_next) else ts + JP_POST_S[1], iv.t1))
        jumps = []
        for l in lanes:
            p0, p1 = _window_mean(l, ("PN", "PE", "VN", "VE"), *pre), _window_mean(l, ("PN", "PE", "VN", "VE"), *post)
            if p0 is None or p1 is None:
                continue
            (t0m, v0), (t1m, v1) = p0, p1
            dtm = t1m - t0m
            dn = (v1[0] - v0[0]) - 0.5 * (v0[2] + v1[2]) * dtm
            de = (v1[1] - v0[1]) - 0.5 * (v0[3] + v1[3]) * dtm
            jumps.append(float(np.hypot(dn, de)))
        row = {"t": _r(ts), "from_set": s_from, "to_set": s_to,
               "pair": "{}->{}".format(_kind(sets, s_from), _kind(sets, s_to)),
               "dwell_s": _r(t_next - ts) if np.isfinite(t_next) else None, "short_dwell": bool(short),
               "clipped_at_disarm": bool(ts + JP_POST_S[1] > iv.t1),
               "jump_m": _r(float(np.median(jumps))) if jumps else None, "n_lanes": len(jumps)}
        if of4:
            o0, o1 = _window_mean(of4[0], ("OFN", "OFE"), *pre), _window_mean(of4[0], ("OFN", "OFE"), *post)
            if o0 is not None and o1 is not None:
                row["of_after_m"] = _r(np.hypot(*o1[1]))
                row["of_step_m"] = _r(np.hypot(o1[1][0] - o0[1][0], o1[1][1] - o0[1][1]))
        row["phase"] = _phase(ref, ts)
        rows.append(row)
    meas = [r for r in rows if r["jump_m"] is not None]
    base = {"switches": rows, "thresholds": {"info_m": JP_INFO_M, "medium_m": JP_MEDIUM_M},
            "confidence": _CONF[det], "t_warn": _r(ref.t_warn), "anchors_local": ref.local}
    if not meas:
        return _na(det, "insufficient_data", "來源切換前後的 XKF1 樣本不足(停留太短),無法量測跳變。", **base)
    top = max(meas, key=lambda r: r["jump_m"])
    jm = top["jump_m"]
    pair_txt = {"vio->gps": "VIO→GPS", "gps->vio": "GPS→VIO", "gps->gps": "GPS→GPS", "vio->vio": "VIO→VIO"}.get(top["pair"], top["pair"])
    listing = "、".join(f"{r['t']:.1f}s {r['jump_m']:.2f} m" for r in meas[:6])
    note = ""
    if top["pair"] == "gps->gps" and jm >= JP_INFO_M:
        note = "同為 GPS 的兩組之間不該有跳變,請檢查來源組參數是否真的相同。"
    if jm >= JP_INFO_M:
        tail = "本身不是失事原因。" if _crash_flight(a, ref) else "本身不是故障。"
        return Finding(det, DOMAIN, SOFT, "medium" if jm >= JP_MEDIUM_M else "info", t_onset=top["t"], t_alarm=top["t"], context="switch",
                       values=dict(base, max_jump_m=jm, max_jump_pair=top["pair"]),
                       text_zh=f"共 {len(meas)} 次來源切換,EKF 位置最大跳變 {jm:.2f} m({top['t']:.1f}s,{pair_txt});各次:{listing}。"
                               "跳變量就是 VIO 相對 GPS 座標的累積偏差,可用來驗證 VIO 精度;位置控制器會跟著重設目標,"
                               + tail + note)
    return Finding(det, DOMAIN, CLEAR, "info", context="switch", values=dict(base, max_jump_m=jm, max_jump_pair=top["pair"]),
                   text_zh=f"共 {len(meas)} 次來源切換,EKF 位置跳變都 <{JP_INFO_M} m(最大 {jm:.2f} m,{top['t']:.1f}s),座標一致。")


# ====================================================================================
# 模組進入點
# ====================================================================================

_DETECTORS = (source_switch_audit, vio_dropout, ekf_aiding_timeout, ekf_testratio_gate, ekf_vio_disagreement,
              nav_velocity_gps_fallback_risk, source_switch_position_jump)


def run(ctx: Ctx, a: Anchors) -> list[Finding]:
    """依序執行七個偵測器。每個 id 至少回傳一筆 Finding(保險:任何偵測器若沒回傳就補 NA)。"""
    out: list[Finding] = []
    for fn, det_id in zip(_DETECTORS, DETECTOR_IDS):
        got = fn(ctx, a)
        if not got:
            got = [_na(det_id, "no_result", "偵測器沒有產生結果。")]
        out.extend(got)
    # 給 engine 的導航脈絡(JSON 友善;engine 要畫時間軸時可用)
    try:
        if a.interval is not None:
            sw = next((f for f in out if f.id == "source_switch_audit" and f.values.get("switches") is not None), None)
            a.extra["nav"] = {"switches": [{"t": s["t"], "from": s["from_set"], "to": s["to_set"], "flags": s["flags"]}
                                           for s in (sw.values["switches"] if sw else [])]}
    except Exception:  # noqa: BLE001 — 附加資訊,失敗不影響偵測結果
        pass
    return out
