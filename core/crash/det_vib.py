"""振動 / IMU 偵測模組(domain = 'vibration'):排除「振動過大」與「IMU 感測器故障」為失事原因。

偵測器(每個 id 每次呼叫至少回傳一筆 Finding):
  vibration_flight_level  飛行中 VIBE 振動等級(p50 / p95 / max、Clip 增量、與同 log 較早飛行的比值)
  imu_consistency_health  三顆 IMU 去固定偏差後的加速度 / 角速度一致性 + 健康旗標(GH / AH / EG / EA / GHz / AHz / T)

只用「事件之前」的資料下結論:撞擊 / 翻覆之後 VIBE 暴增、Clip 躍升、IMU 之間差異變大,
都是撞擊的後果,只放在 values['post_event'] / values['post_impact'] 當說明,不影響 status。

事件時間(統計窗的結束)依序取:
  1. 別的模組填的錨點(只採用落在解鎖區間內的有限值,見 _anchor_times);
  2. 本模組自己從 IMU 偵測到的第一個硬撞擊(跨 IMU |a| 峰值 ≥ HARD_PEAK,含 a.impacts)。1 與 2 取最早者。
     不依賴錨點:只有 basic 錨點、甚至完全沒有錨點(log 結束時仍解鎖)時,撞擊後的資料也不會混進「事件前」;
  3. 只有 VIBE、沒有可用的 IMU,且這段解鎖不是確認的正常降落:VIBE 的撞擊特徵
     (平靜之後 VIBE>30 不到 3 s 的短暫突波,或 ≥ 2 個實例同時 Clip 躍升,見 _vibe_impact_signature);
  4. 以上都沒有:CRASH_CHECK(t_detect)前 5 s;再沒有就是上鎖前 5 s(VIBE)/ 上鎖時間(IMU)。

資料覆蓋(契約:資料缺就不能 CLEAR):統計窗內相鄰有效樣本相隔 > HOLE_S 視為「資料空洞」(含窗頭 / 窗尾),
寫進 values['coverage'] 與文字。統計窗最後 LAST_WIN_S 有一半以上是空洞,或最後一筆有效資料距窗尾 > TAIL_GAP_MAX_S
→ 原本會是 CLEAR 的改回 NA(context 'data_gap');HIT / SOFT 是已有資料裡的證據,保留但加註。
IMU 的覆蓋以「兩顆 IMU 都有有效樣本」的共同時間軸計算(比較基準那顆自己的空洞 / 提早結束也算在內)。
非有限值(NaN / inf)一律當成沒有資料,不會讓比較默默失效。

ctx 只用到:armed、instances()、table()、err_rows(),所以測試可以用假 ctx(見 tests/test_crash_vibration.py)。

可信度(誠實說明)
  整個資料集只有 00000124 一次失事,且 VIBE 與 IMU 從沒壞過:兩個偵測器的「壞」分支(bad / IMU 故障)
  沒有真實正例,門檻來自 ArduPilot 官方建議(VIBE 30 / 60)與實測的飛行基準,不是用失效 log 校準的。
  'ok' / 'consistent' 的結論則有兩趟獨立飛行(124 飛行 1、飛行 2 撞擊前)互相印證,可靠。
  沒有 IMU 時的 VIBE 撞擊特徵(上面第 3 點)沒有真實樣本可驗證(7 份 log 都有 IMU);它可能把「平靜飛行中
  突然出現、之後才惡化」的振動當成撞擊而截掉,所以只在沒有 IMU、且不是正常降落時才使用。
  6 個 baseline log(117/118/119/120/122/123)從未解鎖,不是飛行對照,全部回 NA。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx
from .guard import guarded
from .types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval, Finding

DOMAIN = "vibration"

# ===================================================================== vibration_flight_level
# VIBE 為 10 Hz;m = max(VibeX, VibeY, VibeZ),每個 IMU 實例各自統計(IMU2 的 X 軸比 IMU0/1 低約 5 倍,不跨實例比較)
VIBE_MARGINAL = 15.0     # p95 ≥ 此值 → marginal。124 兩趟飛行 p95 6.9–10.1(30 s 窗)→ 約 1.5 倍餘裕
VIBE_BAD = 30.0          # p95 ≥ 此值 → bad。ArduPilot 建議上限 30(>60 一定有問題);124 飛行最大 11.24 → 2.7 倍
FRAC_BAD = 0.05          # 飛行中 m > 30 的樣本比例 > 5% → bad(124:0.000)
SUSTAINED_BAD_S = 5.0    # m > 30 連續 ≥ 5 s → bad(對應假說的 sustained 條件;124:0 s)
RATIO_MARGINAL = 2.0     # 最後 60 s 的 p95 ÷ 同 log 較早飛行的 p95 > 2 → marginal(124 飛行 2 對飛行 1:1.0)
RATIO_MIN_ABS = 10.0     # 比值只在最後 60 s 的 p95 ≥ 此值才算(避免 p95 從 2 升到 5 這種仍然很安靜的情況被標成劣化)
LAST_WIN_S = 60.0        # 「最近」窗長:統計窗結束(事件)前的 60 s(不是「最後 60 s 有資料的地方」)
BASE_MIN_S = 60.0        # 較早飛行至少要有這麼長的有效資料才當比較基準(否則不比,只用絕對門檻)
SKIP_START_S = 10.0      # 解鎖後前 10 s 不算(起飛前怠速 / 起飛瞬間)
EVENT_LEAD_S = 1.0       # 事件(錨點 / 本地硬撞擊 / VIBE 撞擊特徵)前 1 s 結束統計窗
CRASH_LEAD_S = 5.0       # 只有 CRASH_CHECK(t_detect)可用時:t_detect − 5 s(韌體比實際撞擊晚約 2.6 s,再留餘裕)
TAIL_EXCL_S = 5.0        # 正常降落:排除上鎖前最後 5 s(觸地 / 降落怠速);比較基準飛行也一樣
MIN_WINDOW_S = 20.0      # 統計窗內有效 VIBE 至少 20 s,否則 NA
RUN_GAP_S = 0.25         # VIBE 10 Hz、log 偶有 0.5–0.8 s 空洞:相鄰樣本時間差超過此值就斷開連續段
ERR_VIBE_SUBSYS = 29     # ERR Subsys 29 = FAILSAFE_VIBE(韌體振動失效保護)
NORMAL_ENDS = ("normal_landing", "normal_end")

# 沒有 IMU 時的 VIBE 撞擊特徵(124 實測:兩次硬撞擊時三個實例都在同一筆 VIBE 超過 30、各持續 0.4–0.6 s,
# 第二次撞擊時 Clip 三個實例在同一筆躍升 +12 / +12 / +1;撞擊前 VIBE p95 < 10)
IMPACT_BURST_MAX_S = 3.0 # VIBE>30 的連續段 < 3 s 才算撞擊突波(124:0.4–0.6 s → 5 倍餘裕;持續振動 bad 要 ≥ 5 s,兩者不重疊)
SIG_CALM_S = 10.0        # 突波 / Clip 躍升之前 10 s 該實例 VIBE p95 < VIBE_MARGINAL(平靜)才算撞擊;
                         # 持續在 30 上下的振動會有很多短暫超標,但之前不平靜 → 不會被誤當成撞擊而截掉
SIG_CALM_MIN_FRAC = 0.5  # 那 10 s 至少一半時間有(解鎖區間內的)資料才能判定平靜;剛解鎖、資料不夠 → 不當撞擊
SIG_SYNC_S = 0.5        # 多個實例的特徵相距 ≤ 0.5 s 視為同一事件(Clip 最多比 IMU 峰值晚約 0.25 s)

# ===================================================================== 資料覆蓋(兩個偵測器共用)
HOLE_S = 1.0             # 相鄰有效樣本相隔 > 1 s = 資料空洞(124:VIBE 掉樣最長 0.8 s、IMU 最長 0.45 s,都不算空洞)
LAST_COV_MIN = 0.5       # 統計窗最後 60 s 的覆蓋率 < 50% → 無法排除(CLEAR 改回 NA)
TAIL_GAP_MAX_S = 5.0     # 最後一筆有效資料距窗尾 > 5 s → 無法排除(124 實測 ≤ 0.8 s,約 6 倍餘裕)

# ===================================================================== imu_consistency_health
DA_THR = 0.5             # 加速度差(去固定偏差、0.25 s 平均)m/s²;飛行最大 0.112(4.5 倍餘裕),手持最大 0.251
DG_THR = 0.15            # 角速度差 rad/s;飛行最大 0.022(6.8 倍餘裕),手持最大 0.064
SOFT_FRAC = 0.6          # 達門檻 60% 但未達 → SOFT(0.3 m/s² 約為飛行最大的 2.7 倍;0.09 rad/s 約 4 倍)
SMOOTH_S = 0.25          # 平滑窗(置中、以時間計,遇資料空洞不失真)
MIN_RUN_S = 0.10         # 超標至少連續 0.1 s 才算(避免平滑後剛好擦過門檻的單點)
RUN_GAP_IMU_S = 0.1      # IMU 約 400 Hz、log 偶有最長 0.45 s 空洞:超標段在空洞處斷開
MIN_IMU_SAMPLES = 1000   # IMU 實例至少這麼多筆才採用
OFFSET_REF_S = 120.0     # 固定偏差的參考區間:log 開頭(開機後、解鎖前)最多 120 s 的中位數。
OFFSET_REF_GUARD_S = 5.0 # 參考區間要在解鎖前 5 s 結束(不混入起飛 / 飛行資料);
OFFSET_REF_MIN_S = 5.0   # 不足 5 s 才退回「整份 log 的中位數」(spec 原做法;若故障佔 log 一半以上,中位數會被故障帶走)
OFFSET_STRIDE = 4        # 整份 log 中位數用每 4 筆估計,速度夠、精度足
SHOCK_ONSET = 20.0       # |a| 跨 IMU 中位數 > 20 m/s² 視為撞擊 / 觸地(非撞擊最大 16.5;正常觸地 25–29)
HARD_PEAK = 40.0         # 峰值 ≥ 40 m/s² 視為硬撞擊(124:62 與 90;正常觸地最大 29);其後的 VIBE / IMU 差異一律算後果
SHOCK_MERGE_S = 0.5      # 相距 < 0.5 s 的衝擊合併成一段
SHOCK_PRE_S = 0.3        # 衝擊窗往前延 0.3 s(0.25 s 平滑會把差異往前抹開約 0.125 s)
SHOCK_POST_S = 2.0       # 衝擊窗往後延 2 s(撞擊後機體還在彈跳 / 翻動,IMU 位置不同 → 離心 / 切向加速度不同)
EVENT_SMEAR_S = 0.15     # 超標段起點 ≥ 事件時間 − 0.15 s 視為事件的一部分(平滑抹開 + 餘裕)
GHZ_FRAC = 0.9           # GHz / AHz < 該實例整份 log 中位數的 90% → 取樣率掉落(124:全程 3200–3276,不分板子)
T_JUMP_C = 5.0           # 10 s 內 IMU 溫度變化 > 5 °C(124:最大 1 °C)
T_JUMP_WIN_S = 10.0
GAP_NOTE_S = 0.05        # IMU 相鄰樣本間隔 50 ms–1 s = 短暫記錄空洞(資料品質註記,不是感測器故障;124 解鎖期間 40 處、最長 0.39 s)
PAIR_MAX_DIST_S = 0.02   # 實例 j 在基準樣本 ±20 ms 內要有真實樣本才比較(400 Hz;7 份 log 實測最近樣本距離最大 2.8 ms)
MIN_CHECK_S = 10.0       # 統計窗(與實際可比較的資料)至少 10 s 才下結論
REF_MIN_FRAC = 0.9       # 比較基準:優先最小編號的 IMU;但它在解鎖區間的有效筆數要 ≥ 最多者的 90%、
                         # 且不比別人早結束 > HOLE_S,否則改用資料最完整的那顆(124 三顆筆數差 ≤ 2 筆)

_AXES = ("X", "Y", "Z")
_IMU_COLS = ("AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ")
_KIND_ZH = {"gyro_unhealthy": "陀螺儀健康旗標 GH=0", "accel_unhealthy": "加速度計健康旗標 AH=0",
            "gyro_error_count": "陀螺儀錯誤計數 EG 增加", "accel_error_count": "加速度計錯誤計數 EA 增加",
            "gyro_rate_drop": "陀螺儀取樣率掉落", "accel_rate_drop": "加速度計取樣率掉落", "temp_jump": "溫度突變"}


# ===================================================================== 共用小工具

def _f(x, nd: int = 3):
    """numpy 純量 → JSON 安全的 float(NaN / inf → None)。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _fnum(x) -> float | None:
    """→ 有限的 float;None / NaN / inf / 非數字 → None。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _fmt(x, nd: int = 2) -> str:
    """文字用的數字格式;沒有值 / 非有限值 → '—'(文字裡不會出現 nan)。"""
    v = _fnum(x)
    return "—" if v is None else f"{v:.{nd}f}"


def _runs_idx(mask: np.ndarray, t: np.ndarray, max_gap: float | None = None) -> list[tuple[int, int]]:
    """連續為真的區段 [(i0, i1)](含);相鄰樣本時間差 > max_gap 視為斷開。向量化。"""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1))
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size - 1)
    out: list[tuple[int, int]] = []
    for s, e in zip(starts, ends):
        if max_gap is None or e == s:
            out.append((int(s), int(e)))
            continue
        cut = np.flatnonzero(np.diff(t[s:e + 1]) > max_gap)
        prev = int(s)
        for b in cut:
            out.append((prev, int(s + b)))
            prev = int(s + b + 1)
        out.append((prev, int(e)))
    return out


def _tmean(t: np.ndarray, v: np.ndarray, win_s: float) -> np.ndarray:
    """以時間計的置中移動平均(v 形狀 (k, N) 或 (N,));窗口在端點縮小;資料空洞不拉長窗。v 必須全為有限值。"""
    t = np.asarray(t, dtype=float)
    v2 = np.atleast_2d(np.asarray(v, dtype=float))
    lo = np.searchsorted(t, t - win_s / 2.0, side="left")
    hi = np.searchsorted(t, t + win_s / 2.0, side="right")
    c = np.concatenate([np.zeros((v2.shape[0], 1)), np.cumsum(v2, axis=1)], axis=1)
    out = (c[:, hi] - c[:, lo]) / np.maximum(hi - lo, 1)
    return out[0] if np.ndim(v) == 1 else out


def _tmean_valid(t: np.ndarray, v: np.ndarray, valid: np.ndarray, win_s: float) -> np.ndarray:
    """_tmean,但只平均 valid 且有限的樣本(窗內沒有這種樣本 → 0)。

    非有限值一律當成無效:一個 NaN 進了 cumsum 會讓之後所有平均都變 NaN、比較永遠為 False(故障被默默遮掉)。
    valid 全真且沒有非有限值時與 _tmean 完全相同。
    """
    v = np.asarray(v, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(v)
    if valid.all():
        return _tmean(t, v, win_s)
    num = _tmean(t, np.where(valid, v, 0.0), win_s)
    den = _tmean(t, valid.astype(float), win_s)
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def _paired(tj: np.ndarray, t: np.ndarray, max_dist: float = PAIR_MAX_DIST_S) -> np.ndarray:
    """t 的每個時間點,實例 j(時間 tj,已排序)在 max_dist 內是否有真實樣本。"""
    if tj.size == 0:
        return np.zeros(t.size, dtype=bool)
    k = np.clip(np.searchsorted(tj, t), 1, max(tj.size - 1, 1))
    d = np.minimum(np.abs(tj[np.minimum(k, tj.size - 1)] - t), np.abs(tj[k - 1] - t))
    return d <= max_dist


def _nanmax(x: np.ndarray):
    """有限值的最大值;沒有有限值回傳 None(避免 NaN 讓整段比較默默變成 False)。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.max()) if x.size else None


def _nanmed_rows(x: np.ndarray) -> np.ndarray:
    """(k, N) 每一列有限值的中位數 → (k, 1);某列沒有有限值 → NaN(呼叫端把整組當成無法比較)。"""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    out = np.full((x.shape[0], 1), np.nan)
    for r in range(x.shape[0]):
        v = x[r][np.isfinite(x[r])]
        if v.size:
            out[r, 0] = float(np.median(v))
    return out


def _na(det_id: str, ctx_name: str, text: str, **values) -> Finding:
    return Finding(det_id, DOMAIN, NA, context=ctx_name, values=values, text_zh=text)


def _anchor_times(a: Anchors, iv: ArmedInterval, names: tuple[str, ...]) -> list[tuple[float, str]]:
    """[(時間, 錨點名)]:只取有限值、且落在解鎖區間內(iv.t0 < v ≤ iv.t1 + 1 s)的錨點。

    兩個偵測器共用同一條規則:過期 / 屬於別段解鎖 / NaN 的錨點一律忽略,兩者的統計窗才會一致。
    """
    out = []
    for name in names:
        v = _fnum(getattr(a, name, None))
        if v is not None and iv.t0 < v <= iv.t1 + 1.0:
            out.append((v, name))
    return out


def _impacts_in(a: Anchors, iv: ArmedInterval) -> list[dict]:
    """a.impacts 中 t_start 為有限值、且落在解鎖區間內的衝擊(其餘欄位的檢查在 _shock_events)。"""
    out = []
    for im in getattr(a, "impacts", None) or []:
        try:
            ts = _fnum(im.get("t_start"))
        except AttributeError:
            continue
        if ts is not None and iv.t0 <= ts <= iv.t1 + 1.0:
            out.append(im)
    return out


def _coverage(ts: np.ndarray, a: float, b: float) -> dict:
    """有效樣本時間 ts(已排序)對 [a, b] 的覆蓋。

    空洞 = 相鄰有效樣本(含窗頭 a、窗尾 b)相隔 > HOLE_S 的區段;frac / last_frac = 扣掉空洞後的比例
    (整窗 / 窗尾 LAST_WIN_S);tail_gap_s = 窗尾 − 最後一筆有效樣本。
    """
    ts = np.asarray(ts, dtype=float)
    ts = ts[np.isfinite(ts) & (ts >= a) & (ts <= b)]
    span = max(b - a, 1e-9)
    edges = np.concatenate([[a], ts, [b]])
    d = np.diff(edges)
    k = np.flatnonzero(d > HOLE_S)
    holes = [(float(edges[i]), float(edges[i + 1])) for i in k]
    la = max(a, b - LAST_WIN_S)
    miss_last = sum(h1 - max(h0, la) for h0, h1 in holes if h1 > la)
    return {"a": float(a), "b": float(b), "n": int(ts.size), "holes": holes,
            "frac": max(0.0, 1.0 - float(d[k].sum()) / span),
            "last_frac": max(0.0, 1.0 - miss_last / max(b - la, 1e-9)),
            "tail_gap_s": float(b - ts[-1]) if ts.size else float(b - a),
            "t_last": float(ts[-1]) if ts.size else None}


def _cov_ok(c: dict) -> bool:
    """窗尾(事件前)的資料夠不夠下「排除」結論。"""
    return c["last_frac"] >= LAST_COV_MIN and c["tail_gap_s"] <= TAIL_GAP_MAX_S


def _cov_values(c: dict) -> dict:
    holes = c["holes"]
    lh = max(holes, key=lambda h: h[1] - h[0]) if holes else None
    return {"frac": _f(c["frac"], 3), "last60_frac": _f(c["last_frac"], 3), "n_holes": len(holes),
            "holes_s": _f(sum(h1 - h0 for h0, h1 in holes), 2),
            "longest_hole": None if lh is None else [_f(lh[0], 2), _f(lh[1], 2)],
            "tail_gap_s": _f(c["tail_gap_s"], 2), "t_last": _f(c["t_last"], 3)}


def _cov_best(covs: dict) -> tuple[str | None, dict | None]:
    """覆蓋最好的一個(先看窗尾夠不夠,再看窗尾覆蓋率、整窗覆蓋率)。"""
    if not covs:
        return None, None
    return max(covs.items(), key=lambda kv: (_cov_ok(kv[1]), kv[1]["last_frac"], kv[1]["frac"]))


def _holes_phrase(c: dict) -> str:
    """'共 X 秒(a–b s 等 N 處)'。"""
    holes = c["holes"]
    tot = sum(h1 - h0 for h0, h1 in holes)
    h0, h1 = max(holes, key=lambda h: h[1] - h[0])
    more = f" 等 {len(holes)} 處" if len(holes) > 1 else ""
    return f"共 {tot:.1f} 秒({h0:.1f}–{h1:.1f} s{more})"


def _pre_event_end(a: Anchors, iv: ArmedInterval, shock: tuple[float, str] | None = None) -> tuple[float, str]:
    """VIBE 統計窗的結束時間(相對秒)與依據。

    量到的事件:錨點(t_touch / t_warn / t_loss / t_first_hit / t_impact_peak,只取解鎖區間內的有限值)
    與本地偵測(shock = (時間, 'hard_shock' | 'vibe_signature')),取最早者 − 1 s;
    都沒有:有 CRASH_CHECK 的 t_detect → t_detect − 5 s;否則(正常降落)解鎖區間結束 − 5 s。
    """
    cands = [(v - EVENT_LEAD_S, n) for v, n in
             _anchor_times(a, iv, ("t_touch", "t_warn", "t_loss", "t_first_hit", "t_impact_peak"))]
    if shock is not None and _fnum(shock[0]) is not None:
        cands.append((float(shock[0]) - EVENT_LEAD_S, shock[1]))
    if cands:
        hi, why = min(cands)
        return min(hi, iv.t1), why
    det = _anchor_times(a, iv, ("t_detect",))
    if det:
        return min(det[0][0] - CRASH_LEAD_S, iv.t1), "t_detect"
    return iv.t1 - TAIL_EXCL_S, "disarm"


# ===================================================================== vibration_flight_level

def _vibe_tables(ctx: Ctx) -> list[tuple[int | None, dict]]:
    """[(instance, table)];table 需有 t / VibeX / VibeY / VibeZ。"""
    out = []
    for inst in (ctx.instances("VIBE") or [None]):
        tb = ctx.table("VIBE", inst)
        if tb is not None and all(k in tb for k in ("t", "VibeX", "VibeY", "VibeZ")) and tb["t"].size:
            out.append((inst, tb))
    return out


def _vibe_ok(tb: dict) -> np.ndarray:
    return np.isfinite(tb["t"]) & np.isfinite(tb["VibeX"]) & np.isfinite(tb["VibeY"]) & np.isfinite(tb["VibeZ"])


def _vibe_m(tb: dict) -> np.ndarray:
    return np.maximum.reduce([tb["VibeX"], tb["VibeY"], tb["VibeZ"]])


def _vibe_stats(tb: dict, lo: float, hi: float) -> dict | None:
    """單一 IMU 實例在 [lo, hi] 的 VIBE 統計;沒有樣本回傳 None。內部欄位(以 _ 開頭)不進 values。"""
    t = tb["t"]
    msk = (t >= lo) & (t <= hi) & _vibe_ok(tb)
    n = int(msk.sum())
    if n == 0:
        return None
    ts = t[msk]
    ax = {k: tb[f"Vibe{k}"][msk] for k in _AXES}
    m = np.maximum.reduce([ax["X"], ax["Y"], ax["Z"]])
    dt = float(np.median(np.diff(ts))) if n > 2 else 0.1
    st: dict = {
        "n": n, "dur_s": n * dt,
        "p50": float(np.percentile(m, 50)), "p95": float(np.percentile(m, 95)), "max": float(m.max()),
        "axes": {k.lower(): [float(np.percentile(v, 50)), float(np.percentile(v, 95)), float(v.max())]
                 for k, v in ax.items()},
        "frac15": float(np.mean(m > VIBE_MARGINAL)), "frac30": float(np.mean(m > VIBE_BAD)),
    }
    # 事件前最後 60 s:以統計窗結束 hi 為準(資料若在窗尾中斷,這一段就是空的,不會悄悄改看更早的時段)
    ml = ts >= hi - LAST_WIN_S
    st["n_last"] = int(ml.sum())
    st["p95_last"] = float(np.percentile(m[ml], 95)) if ml.sum() >= 5 else None
    # 連續超標
    r30 = _runs_idx(m > VIBE_BAD, ts, RUN_GAP_S)
    st["longest30_s"] = max((ts[j] - ts[i] + dt for i, j in r30), default=0.0)
    for thr, key in ((VIBE_MARGINAL, "15"), (VIBE_BAD, "30")):
        i_all = np.flatnonzero(m > thr)
        i_last = np.flatnonzero((m > thr) & ml)
        st["_t" + key] = float(ts[i_all[0]]) if i_all.size else None
        # 只有「最近 60 s」的 p95 達標時,起點取最近 60 s 窗內第一次超標(較早的單點突波不是這段振動的起點)
        st["_t" + key + "_last"] = float(ts[i_last[0]]) if i_last.size else None
    # Clip:累計計數器 → 只看增量(可能從非 0 起算,甚至歸零);取窗內「正增量」的總和
    st["clip_delta"], st["_t_clip"] = _clip_increments(tb.get("Clip"), msk, t)
    return st


def _clip_series(clip, msk: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """msk 範圍內有限的 Clip 值(int64)與時間;沒有可用欄位 → None。

    NaN 樣本先丟掉(轉 int64 會變成極大的負數,差分後變成假的巨大增量)。
    """
    if clip is None or getattr(clip, "dtype", None) is None or clip.dtype.kind not in "biuf":
        return None
    c = np.asarray(clip[msk], dtype=float)
    tc = t[msk]
    ok = np.isfinite(c)
    return c[ok].astype(np.int64), tc[ok]


def _clip_increments(clip, msk: np.ndarray, t: np.ndarray) -> tuple[int | None, float | None]:
    """累計計數器 Clip 在 msk 範圍內的正增量總和與第一次增加的時間;沒有欄位回傳 (None, None)。

    歸零(負增量)不抵銷、也不算。
    """
    cs = _clip_series(clip, msk, t)
    if cs is None:
        return None, None
    c, tc = cs
    dc = np.diff(c)
    jj = np.flatnonzero(dc > 0)
    return (int(dc[jj].sum()) if jj.size else 0), (float(tc[jj[0] + 1]) if jj.size else None)


def _clip_jump_times(clip, msk: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Clip 正增量發生的時間(已排序)。"""
    cs = _clip_series(clip, msk, t)
    if cs is None:
        return np.zeros(0)
    c, tc = cs
    jj = np.flatnonzero(np.diff(c) > 0)
    return tc[jj + 1]


def _has_near(arr: np.ndarray, x: float, tol: float) -> bool:
    """已排序的 arr 在 x ± tol 內是否有值。"""
    if arr.size == 0:
        return False
    k = int(np.searchsorted(arr, x))
    return any(0 <= i < arr.size and abs(arr[i] - x) <= tol for i in (k - 1, k))


def _synced_first(lists: list[np.ndarray], need: int, tol: float) -> float | None:
    """至少 need 個實例在 ±tol 內都有事件的最早時間;沒有 → None。lists 各自已排序。"""
    if need <= 0 or sum(1 for L in lists if L.size) < need:
        return None
    best = None
    for L in lists:
        for x in L:
            if best is not None and x >= best:
                break
            if sum(1 for M in lists if _has_near(M, float(x), tol)) >= need:
                best = float(x)
                break
    return best


def _calm_before(t: np.ndarray, m: np.ndarray, x: float, dt: float) -> bool:
    """x 之前 SIG_CALM_S 內(解鎖區間內)該實例 VIBE 平靜:至少一半時間有資料,且 p95 < VIBE_MARGINAL。

    資料不夠(例如剛解鎖)→ 不算平靜:無法確認是「平靜中突然一下」,就不當撞擊截窗
    (否則一解鎖就持續在 30 上下的壞振動,第一個短暫超標會被當成撞擊,整段振動被截掉)。
    """
    sel = (t >= x - SIG_CALM_S) & (t < x)
    n = int(sel.sum())
    return n * max(dt, 1e-3) >= SIG_CALM_MIN_FRAC * SIG_CALM_S and float(np.percentile(m[sel], 95)) < VIBE_MARGINAL


def _vibe_impact_signature(tabs: list[tuple[int | None, dict]], iv: ArmedInterval) -> float | None:
    """沒有 IMU 時,從 VIBE 找撞擊時間(解鎖區間內最早者);找不到 → None。

    (a) 平靜之後 VIBE>30 的連續段 < IMPACT_BURST_MAX_S:≥ 2 個實例(只有一個實例時就是它)在 ±0.5 s 內同時出現;
    (b) 平靜之後的 Clip 躍升:≥ 2 個實例在 ±0.5 s 內同時增加(只有一個實例無法區分「撞擊」與「振動削波」,不用)。
    持續的高振動不平靜、或超標段 ≥ 3 s,都不算撞擊特徵(留在統計窗內當證據)。
    """
    bursts: list[np.ndarray] = []
    clips: list[np.ndarray] = []
    for _, tb in tabs:
        msk = _vibe_ok(tb) & (tb["t"] >= iv.t0) & (tb["t"] <= iv.t1)       # 只看這段解鎖區間
        ts, m = tb["t"][msk], _vibe_m(tb)[msk]
        if ts.size < 3:
            continue
        dt = float(np.median(np.diff(ts)))
        b = [float(ts[i]) for i, j in _runs_idx(m > VIBE_BAD, ts, HOLE_S)
             if ts[j] - ts[i] + dt < IMPACT_BURST_MAX_S and _calm_before(ts, m, float(ts[i]), dt)]
        bursts.append(np.asarray(b, dtype=float))
        cj = [float(x) for x in _clip_jump_times(tb.get("Clip"), msk, tb["t"]) if _calm_before(ts, m, float(x), dt)]
        clips.append(np.asarray(cj, dtype=float))
    n = len(bursts)
    cands = [_synced_first(bursts, min(2, n), SIG_SYNC_S)]
    if n >= 2:
        cands.append(_synced_first(clips, 2, SIG_SYNC_S))
    cands = [x for x in cands if x is not None]
    return min(cands) if cands else None


def _first_hard_shock(ctx: Ctx, a: Anchors, iv: ArmedInterval) -> tuple[float | None, bool]:
    """(解鎖區間內第一個硬撞擊的開始時間或 None, 有沒有可用的 IMU)。與 imu_consistency_health 同一套偵測。"""
    tabs = _imu_tables(ctx)
    if not tabs:
        return None, False
    t, acc = _window_data(tabs, iv)
    ev = _shock_events(t, acc, _impacts_in(a, iv))
    hard = [e["t_start"] for e in ev if e["peak"] >= HARD_PEAK and iv.t0 <= e["t_start"] <= iv.t1 + 1.0]
    return (min(hard) if hard else None), True


def _post_event_info(tabs, hi: float, t_end: float) -> dict:
    """事件之後(hi → 解鎖區間結束)的 VIBE:只當說明(後果),不進 status。"""
    t30, mx, tclip, cd = None, 0.0, None, 0
    for _, tb in tabs:
        t = tb["t"]
        msk = (t > hi) & (t <= t_end) & _vibe_ok(tb)
        if not msk.any():
            continue
        ts = t[msk]
        m = np.maximum.reduce([tb["VibeX"][msk], tb["VibeY"][msk], tb["VibeZ"][msk]])
        mx = max(mx, float(m.max()))
        k = np.flatnonzero(m > VIBE_BAD)
        if k.size and (t30 is None or ts[k[0]] < t30):
            t30 = float(ts[k[0]])
        d, tc = _clip_increments(tb.get("Clip"), msk, t)
        if d:
            cd = max(cd, d)
            tclip = tc if tclip is None else min(tclip, tc)
    return {"t_first_gt30": _f(t30, 2), "max": _f(mx, 1), "clip_delta": cd, "t_first_clip": _f(tclip, 2)}


def _classify_vibe(stats: dict[str, dict], base: dict[str, dict], err29_t: float | None) -> dict:
    """純函式:各實例統計(+ 較早飛行基準)→ 判定。

    stats / base:{實例名: _vibe_stats 結果};回傳 {verdict, clipping, reasons, ratio, worst, p95, t_onset}。
    verdict:'bad' p95≥30 / m>30 比例>5% / 連續 ≥5 s / ERR 29;'marginal' p95≥15 或比值>2;否則 'ok'。
    p95 取「整段」與「最後 60 s」較大者(後者抓得到飛行最後才開始的振動)。
    """
    reasons: list[str] = []
    onset: list[float] = []
    verdict = "ok"
    worst, p95_worst = None, -1.0
    ratio_best, ratio_inst = None, None
    clipping = False
    def first_exceed(s: dict, key: str):
        # 整段 p95 已達標 → 整段第一次超標;只有最近 60 s 達標 → 最近 60 s 窗內第一次超標
        whole = s["p95"] >= (VIBE_BAD if key == "30" else VIBE_MARGINAL)
        return s.get("_t" + key) if whole else s.get("_t" + key + "_last", s.get("_t" + key))

    for name, s in stats.items():
        p95 = max(s["p95"], s["p95_last"] or 0.0)
        if p95 > p95_worst:
            worst, p95_worst = name, p95
        if p95 >= VIBE_BAD:
            verdict = "bad"
            reasons.append(f"{name} p95 {p95:.1f} ≥ {VIBE_BAD:.0f}")
            t30 = first_exceed(s, "30")
            onset.append(t30 if t30 is not None else first_exceed(s, "15"))
        if s["frac30"] > FRAC_BAD:
            verdict = "bad"
            reasons.append(f"{name} 超過 30 的樣本 {s['frac30']*100:.1f}% > {FRAC_BAD*100:.0f}%")
            onset.append(s["_t30"])
        if s["longest30_s"] >= SUSTAINED_BAD_S:
            verdict = "bad"
            reasons.append(f"{name} 連續超過 30 達 {s['longest30_s']:.1f} s")
            onset.append(s["_t30"])
        if s.get("clip_delta"):
            clipping = True
            onset.append(s["_t_clip"])
        b = base.get(name)
        if b is not None and s["p95_last"] is not None and b["p95"] > 0:
            r = s["p95_last"] / b["p95"]
            if ratio_best is None or r > ratio_best:
                ratio_best, ratio_inst = r, name
    if verdict != "bad":
        for name, s in stats.items():
            p95 = max(s["p95"], s["p95_last"] or 0.0)
            if p95 >= VIBE_MARGINAL:
                verdict = "marginal"
                reasons.append(f"{name} p95 {p95:.1f} ≥ {VIBE_MARGINAL:.0f}")
                onset.append(first_exceed(s, "15"))
        if ratio_best is not None and ratio_best > RATIO_MARGINAL and stats[ratio_inst]["p95_last"] >= RATIO_MIN_ABS:
            verdict = "marginal"
            reasons.append(f"{ratio_inst} 最近 60 s p95 是較早飛行的 {ratio_best:.1f} 倍 > {RATIO_MARGINAL:.0f}")
    if err29_t is not None:
        verdict = "bad"
        reasons.append("韌體振動失效保護(ERR 29)")
        onset.append(err29_t)
    onset = [x for x in onset if x is not None]
    return {"verdict": verdict, "clipping": clipping, "reasons": reasons, "ratio": ratio_best,
            "ratio_inst": ratio_inst, "worst": worst, "p95": p95_worst, "t_onset": min(onset) if onset else None}


def _vib_cov_text(c: dict | None) -> str:
    if not c or not c["holes"]:
        return ""
    return f"(VIBE 在統計窗內有超過 {HOLE_S:.0f} 秒的資料空洞,{_holes_phrase(c)},該段無法評估)"


def _vib_text(verdict: str, clipping: bool, s_w: dict, p95: float, hi_dur: float, ratio, base_p95, post: dict,
              crash_like: bool, reasons: list[str], basis: str = "", t_event: float | None = None,
              cov_txt: str = "") -> str:
    ax = s_w["axes"]
    lvl = (f"解鎖飛行 {hi_dur:.0f} 秒內 VIBE 最大軸 95 百分位 {p95:.1f} m/s²(注意值 {VIBE_MARGINAL:.0f}、上限 {VIBE_BAD:.0f}),"
           f"X/Y/Z 中位數約 {ax['x'][0]:.1f}/{ax['y'][0]:.1f}/{ax['z'][0]:.1f}、最大 {s_w['max']:.1f}")
    cmp_ = f";與同一份 log 較早的飛行相比為 {ratio:.1f} 倍(較早飛行 p95 {base_p95:.1f})" if ratio is not None and base_p95 else ""
    clip_ = "" if not clipping else f"。統計窗內 Clip 計數增加 {s_w.get('clip_delta')} 次(加速度計曾超出量程)"
    bits = []
    if post.get("t_first_gt30") is not None:
        bits.append(f"VIBE 到 {post['t_first_gt30']:.2f} s 才第一次超過 30(最大 {post['max']:.0f})")
    if post.get("t_first_clip") is not None:
        bits.append(f"Clip 到 {post['t_first_clip']:.2f} s 才增加(+{post['clip_delta']})")
    post_ = ""
    if bits:
        if crash_like:
            where = ",屬撞擊 / 翻覆之後的後果"
        elif basis == "hard_shock" and t_event is not None:
            where = f",發生在 {t_event:.2f} s 的強烈衝擊(≥ {HARD_PEAK:.0f} m/s²)之後(統計窗之外)"
        elif basis == "vibe_signature" and t_event is not None:
            where = f",發生在 {t_event:.2f} s 的 VIBE 突波 / Clip 躍升之後(統計窗之外)"
        else:
            where = ",發生在觸地 / 降落階段(統計窗之外)"
        post_ = "。" + "、".join(bits) + where
    if verdict == "ok" and not clipping:
        who = "振動在正常範圍" + ("" if not crash_like else ",可排除振動為失事原因")
        return f"{who}:{lvl}{cmp_};Clip 無增加{post_}。{cov_txt}"
    if verdict == "ok":
        return f"振動等級在正常範圍,但有削波:{lvl}{cmp_}{clip_}{post_}。請檢查是否有短暫的強烈顛簸或槳葉 / 避震問題。{cov_txt}"
    why = "、".join(reasons[:3])
    if verdict == "marginal":
        return (f"飛行中振動偏高({why}):{lvl}{cmp_}{clip_}{post_}。建議檢查槳葉平衡、馬達軸承與飛控避震;"
                f"若振動在事件前就存在,可能是影響姿態估計的因素。{cov_txt}")
    return (f"飛行中振動過大({why}):{lvl}{cmp_}{clip_}{post_}。請檢查槳葉是否損傷 / 不平衡、馬達軸承、機架共振與飛控避震;"
            f"振動過大會讓 EKF 與姿態控制失準。{cov_txt}")


def _crash_like(a: Anchors) -> bool:
    """這段解鎖是否以失事收尾(決定文字用「可排除為失事原因 / 撞擊後果」還是一般健康描述)。"""
    if a.end_type in ("normal_landing", "normal_end", "not_armed"):
        return False
    if a.end_type not in ("unknown", "", None):
        return True
    names = ("t_detect", "t_loss", "t_first_hit", "t_impact_peak")
    if a.interval is None:
        return any(_fnum(getattr(a, k, None)) is not None for k in names)
    return bool(_anchor_times(a, a.interval, names))          # 同一條錨點規則:區間外 / NaN 的錨點不算


@guarded("vibration_flight_level", DOMAIN)
def vibration_flight_level(ctx: Ctx, a: Anchors):
    """飛行中 VIBE 等級:'ok' → CLEAR、'marginal' / 單獨削波 → SOFT、'bad' / 'marginal'+削波 → HIT。

    統計窗 = 解鎖區間 [t0+10 s, 事件前 1 s](見模組說明與 _pre_event_end)。
    資料不足 / 未解鎖 → NA;事件前最後 60 s 大多沒有資料 → 不能 CLEAR,改回 NA(data_gap)。
    """
    did = "vibration_flight_level"
    iv = a.interval
    if iv is None:
        return _na(did, "not_armed", "未解鎖(桌上 / 手持)的 log:沒有飛行振動可評估(不是『正常』,是不適用)。")
    tabs = _vibe_tables(ctx)
    if not tabs:
        return _na(did, "no_data", "log 沒有 VIBE 訊息,無法評估飛行振動。")
    lo = iv.t0 + SKIP_START_S
    # 撞擊時間不靠錨點也要找得到:IMU 的硬撞擊;沒有 IMU 才用 VIBE 自己的撞擊特徵(正常降落不用,見模組說明)
    t_shock, has_imu = _first_hard_shock(ctx, a, iv)
    shock = (t_shock, "hard_shock") if t_shock is not None else None
    if not has_imu and a.end_type not in NORMAL_ENDS:
        t_sig = _vibe_impact_signature(tabs, iv)
        if t_sig is not None:
            shock = (t_sig, "vibe_signature")
    hi, basis = _pre_event_end(a, iv, shock)
    t_event = hi + EVENT_LEAD_S if basis in ("hard_shock", "vibe_signature") else None
    names = {inst: (f"IMU{inst}" if inst is not None else "IMU0") for inst, _ in tabs}
    stats, covs = {}, {}
    for inst, tb in tabs:
        s = _vibe_stats(tb, lo, hi)
        if s is not None and s["dur_s"] >= MIN_WINDOW_S:
            stats[names[inst]] = s
        if hi > lo:
            covs[names[inst]] = _coverage(tb["t"][_vibe_ok(tb)], lo, hi)
    if not stats:
        return _na(did, "insufficient_window", f"事件前可用的飛行資料不足 {MIN_WINDOW_S:.0f} 秒,無法評估振動。",
                   window=[_f(lo, 2), _f(hi, 2)], basis=basis, window_basis=basis)

    # 較早的飛行(同 log):當比較基準
    base: dict[str, dict] = {}
    for prev in ctx.armed:
        if prev.t1 <= iv.t0 + 1e-6:
            cand = {}
            for inst, tb in tabs:
                s = _vibe_stats(tb, prev.t0 + SKIP_START_S, prev.t1 - TAIL_EXCL_S)
                if s is not None and s["dur_s"] >= BASE_MIN_S:
                    cand[names[inst]] = s
            if cand:
                base = cand
                break          # 最早的一趟

    t_err, sub, _code = ctx.err_rows()
    e29 = [float(x) for x, s_ in zip(t_err, sub) if s_ == ERR_VIBE_SUBSYS and iv.t0 <= x <= hi]
    cl = _classify_vibe(stats, base, min(e29) if e29 else None)
    verdict, clipping = cl["verdict"], cl["clipping"]
    post = _post_event_info(tabs, hi, iv.t1)

    w = stats[cl["worst"]]
    dur = w["dur_s"]
    base_p95 = base[cl["worst"]]["p95"] if cl["worst"] in base else None
    ratio = cl["ratio"]
    ratio_w = (w["p95_last"] / base_p95) if (cl["worst"] in base and base_p95 and w["p95_last"] is not None) else None
    crash_like = _crash_like(a)
    cov_name, cov = _cov_best(covs)
    cov_ok = cov is not None and _cov_ok(cov)

    if verdict == "bad" or (verdict == "marginal" and clipping):
        status, sev = HIT, "high" if verdict == "bad" else "medium"
    elif verdict == "marginal" or clipping:
        status, sev = SOFT, "medium"
    else:
        status, sev = CLEAR, "medium"
    t_on = cl["t_onset"] if status in (HIT, SOFT) else None
    if status in (HIT, SOFT) and t_on is None:
        t_on = max(lo, hi - LAST_WIN_S)         # 只靠比值 / 統計量成立:起點取最近 60 s 窗的開頭
    ctx_name = verdict + ("+clipping" if clipping else "")

    per = {}
    for nm, s in stats.items():
        per[nm] = {"p50": _f(s["p50"], 2), "p95": _f(s["p95"], 2), "max": _f(s["max"], 2),
                   "x": [_f(v, 2) for v in s["axes"]["x"]], "y": [_f(v, 2) for v in s["axes"]["y"]],
                   "z": [_f(v, 2) for v in s["axes"]["z"]],
                   "p95_last60": _f(s["p95_last"], 2), "frac_gt15": _f(s["frac15"], 4), "frac_gt30": _f(s["frac30"], 4),
                   "longest_gt30_s": _f(s["longest30_s"], 2), "clip_delta": s["clip_delta"],
                   "coverage": _f(covs[nm]["frac"], 3) if nm in covs else None}
        if nm in base:
            per[nm]["p95_base"] = _f(base[nm]["p95"], 2)
    values = {
        "verdict": verdict, "clipping": clipping, "reasons": cl["reasons"],
        "worst_instance": cl["worst"], "p95": _f(w["p95"], 2), "p95_last60": _f(w["p95_last"], 2),
        "p50": _f(w["p50"], 2), "max": _f(w["max"], 2), "p95_effective": _f(cl["p95"], 2),
        "frac_gt15": _f(w["frac15"], 4), "frac_gt30": _f(w["frac30"], 4),
        "longest_gt30_s": _f(max(x["longest30_s"] for x in stats.values()), 2),
        "clip_delta": int(max((s.get("clip_delta") or 0) for s in stats.values())),
        "ratio_vs_earlier": _f(ratio_w, 2), "ratio_max": _f(ratio, 2), "p95_earlier": _f(base_p95, 2),
        "window": [_f(lo, 2), _f(hi, 2)], "window_basis": basis, "t_event_local": _f(t_event, 3),
        "dur_s": _f(dur, 1), "per_instance": per, "post_event": post,
        "coverage": None if cov is None else _cov_values(cov), "coverage_instance": cov_name,
        "thresholds": {"marginal_p95": VIBE_MARGINAL, "bad_p95": VIBE_BAD, "bad_frac": FRAC_BAD,
                       "sustained_s": SUSTAINED_BAD_S, "ratio": RATIO_MARGINAL, "ratio_min_abs": RATIO_MIN_ABS,
                       "hole_s": HOLE_S, "last_cov_min": LAST_COV_MIN, "tail_gap_max_s": TAIL_GAP_MAX_S},
        "confidence": "medium" if (cov is not None and not cov["holes"]) else "low",
    }

    if status == CLEAR and not cov_ok:
        # 事件前最後一段沒有資料:振動可能就在那段出現,不能拿來排除
        when = "事件前" if crash_like else "飛行最後階段"
        lf = cov["last_frac"] if cov else 0.0
        tl = cov["t_last"] if cov else None
        tail = cov["tail_gap_s"] if cov else hi - lo
        values["confidence"] = "low"
        return Finding(did, DOMAIN, NA, context="data_gap", values=values, text_zh=(
            f"統計窗最後 {LAST_WIN_S:.0f} 秒只有 {lf*100:.0f}% 有 VIBE 資料(最後一筆在 {_fmt(tl)} s,"
            f"距統計窗結束 {hi:.2f} s 還有 {tail:.1f} s),無法確認{when}的振動是否正常(資料不足,不是『正常』)。"
            f"已有資料的部分:VIBE 最大軸 95 百分位 {cl['p95']:.1f} m/s²、最大 {w['max']:.1f},Clip 無增加。"
            f"{_vib_cov_text(cov)}"))

    cov_txt = _vib_cov_text(cov)
    if status in (HIT, SOFT) and not cov_ok and cov is not None:
        cov_txt += f"(注意:統計窗最後 {LAST_WIN_S:.0f} 秒只有 {cov['last_frac']*100:.0f}% 有資料)"
    text = _vib_text(verdict, clipping, w, cl["p95"], dur, ratio_w, base_p95, post, crash_like, cl["reasons"],
                     basis, t_event, cov_txt)
    return Finding(did, DOMAIN, status, sev, t_onset=t_on, t_alarm=hi if t_on is not None else None,
                   context=ctx_name, values=values, text_zh=text)


# ===================================================================== imu_consistency_health

def _imu_tables(ctx: Ctx) -> dict[int, dict]:
    """可用的 IMU 實例 → table(需 > 1000 筆且有 Acc / Gyr 六欄)。"""
    need = ("t",) + _IMU_COLS
    out: dict[int, dict] = {}
    for inst in (ctx.instances("IMU") or [0]):
        tb = ctx.table("IMU", inst if ctx.instances("IMU") else None)
        if tb is not None and tb["t"].size > MIN_IMU_SAMPLES and all(k in tb for k in need):
            out[int(inst)] = tb
    return out


def _imu_valid_t(tb: dict, t0: float, t1: float) -> np.ndarray:
    """[t0, t1] 內六軸都是有限值的樣本時間。"""
    t = tb["t"]
    i0, i1 = int(np.searchsorted(t, t0)), int(np.searchsorted(t, t1, side="right"))
    ok = np.isfinite(t[i0:i1])
    for c in _IMU_COLS:
        ok &= np.isfinite(tb[c][i0:i1])
    return t[i0:i1][ok]


def _ref_inst(tabs: dict[int, dict], iv: ArmedInterval) -> int:
    """比較基準 IMU:最小編號、但資料要夠完整(見 REF_MIN_FRAC);否則用資料最完整的那顆。

    基準那顆的時間就是比較與衝擊偵測的時間軸:它提早結束或有大空洞時,其他 IMU 的資料再完整也比不到。
    """
    cnt, last = {}, {}
    for i, tb in tabs.items():
        ts = _imu_valid_t(tb, iv.t0, iv.t1)
        cnt[i], last[i] = int(ts.size), (float(ts[-1]) if ts.size else -np.inf)
    n_max, l_max = max(cnt.values()), max(last.values())
    for i in sorted(tabs):
        if cnt[i] > 0 and cnt[i] >= REF_MIN_FRAC * n_max and last[i] >= l_max - HOLE_S:
            return i
    return max(sorted(tabs), key=lambda i: (cnt[i], last[i]))


def _stack(tb: dict, cols: tuple[str, ...], t_dst: np.ndarray) -> np.ndarray:
    """欄位內插到 t_dst(不同實例的取樣時間略有差異)→ shape (len(cols), N)。非有限值會帶到相鄰點,由呼叫端遮掉。"""
    return np.vstack([np.interp(t_dst, tb["t"], tb[c]) for c in cols])


def _window_data(tabs: dict[int, dict], iv: ArmedInterval, ref: int | None = None) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """解鎖區間內、以基準 IMU(_ref_inst)的時間為基準的 (t, {實例: 加速度 (3, N)})。"""
    ref = _ref_inst(tabs, iv) if ref is None else ref
    t_all = tabs[ref]["t"]
    t = t_all[int(np.searchsorted(t_all, iv.t0)):int(np.searchsorted(t_all, iv.t1, side="right"))]
    return t, {i: _stack(tb, ("AccX", "AccY", "AccZ"), t) for i, tb in tabs.items()}


def _shock_events(t: np.ndarray, acc: dict[int, np.ndarray], extra: list[dict] | None) -> list[dict]:
    """衝擊段 [{t_start, t_end, peak}]:|a| 跨 IMU 合成值 > 20 m/s² 的樣本,相距 < 0.5 s 合併;a.impacts 也併入。

    合成值:≥ 3 顆 IMU 取中位數;2 顆取較小者(兩顆都要超標,單顆毛刺不算);1 顆就是它自己。
    非有限值的樣本不會超標;a.impacts 裡 t_start / peak 不是有限值的項目丟掉。
    """
    if t.size == 0:
        return []
    norms = np.vstack([np.sqrt((v ** 2).sum(axis=0)) for v in acc.values()])
    acons = np.median(norms, axis=0) if norms.shape[0] >= 3 else norms.min(axis=0)
    idx = np.flatnonzero(np.isfinite(acons) & (acons > SHOCK_ONSET))
    ev: list[dict] = []
    if idx.size:
        for g in np.split(idx, np.flatnonzero(np.diff(t[idx]) > SHOCK_MERGE_S) + 1):
            ev.append({"t_start": float(t[g[0]]), "t_end": float(t[g[-1]]), "peak": float(acons[g].max())})
    for im in extra or []:
        try:
            ts = _fnum(im["t_start"])
            pk = _fnum(im.get("peak", SHOCK_ONSET))
            te = _fnum(im.get("t_end", im["t_start"]))
        except (AttributeError, KeyError, TypeError):
            continue
        if ts is None or pk is None or pk < SHOCK_ONSET:
            continue
        ev.append({"t_start": ts, "t_end": te if te is not None and te >= ts else ts, "peak": pk})
    # 本地偵測與 a.impacts 通常是同一批衝擊:重疊 / 相距 < 0.5 s 的合併,shock_windows 才是真正的衝擊次數
    merged: list[dict] = []
    for e in sorted(ev, key=lambda e: e["t_start"]):
        if merged and e["t_start"] <= merged[-1]["t_end"] + SHOCK_MERGE_S:
            m = merged[-1]
            m["t_end"] = max(m["t_end"], e["t_end"])
            m["peak"] = max(m["peak"], e["peak"])
        else:
            merged.append(dict(e))
    return merged


def _in_windows(t: np.ndarray, wins: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(t.size, dtype=bool)
    for w0, w1 in wins:
        m |= (t >= w0) & (t <= w1)
    return m


def _offset_reference(t_all: np.ndarray, iv: ArmedInterval) -> tuple[np.ndarray, str]:
    """IMU 之間「固定偏差」的估計樣本:開機後、解鎖前的靜止段(最多 OFFSET_REF_S);不足才用整份 log。

    spec 原做法是整份 log 的中位數;但若故障從飛行中途開始、佔了 log 一半以上,中位數會把故障狀態當成『正常偏差』
    而漏報(合成測試證實)。開機時三顆 IMU 通常都健康,所以優先用解鎖前的區段。124 兩種做法的結果相同(差 < 0.01 m/s²)。
    """
    lo = float(t_all[0])
    hi = min(lo + OFFSET_REF_S, iv.t0 - OFFSET_REF_GUARD_S)
    if hi - lo >= OFFSET_REF_MIN_S:
        sel = t_all[(t_all >= lo) & (t_all <= hi)]
        return sel[::2], "pre_arm"
    return t_all[::OFFSET_STRIDE], "whole_log"


def _consistency(tabs: dict[int, dict], iv: ArmedInterval, t: np.ndarray, acc: dict[int, np.ndarray],
                 events: list[dict], t_event: float, ref: int | None = None) -> dict | None:
    """IMU 之間的一致性(每顆 IMU 與基準 ref 比較)。< 2 個實例回傳 None。

    超標段(dA > 0.5 或 dG > 0.15,連續 ≥ 0.1 s)依「起點」分兩類:起點落在衝擊窗內 / 在事件之後 → 'impact'(後果);
    其餘 → 'free'(事件前、沒有撞擊解釋的不一致 = 疑似 IMU 故障)。
    每組的覆蓋(_coverage)在兩顆都有有效樣本的共同時間軸上計算,放在 'pair_cov'(內部用,不進 values)。
    """
    if len(tabs) < 2 or t.size < 50:
        return None
    i0 = min(tabs) if ref is None else ref
    others = [j for j in sorted(tabs) if j != i0]
    wins = [(e["t_start"] - SHOCK_PRE_S, e["t_end"] + SHOCK_POST_S) for e in events]
    shock = _in_windows(t, wins)
    t_off, off_basis = _offset_reference(tabs[i0]["t"], iv)    # 固定偏差的估計樣本時間
    gyr0 = _stack(tabs[i0], ("GyrX", "GyrY", "GyrZ"), t)
    clean0 = (t <= t_event - EVENT_SMEAR_S) & ~shock
    pairs, pair_cov = {}, {}
    free_hit, free_soft, impact_runs = [], [], []
    for j in others:
        key = f"{i0}-{j}"
        # 只比較「實例 j 在 20 ms 內有真實樣本」的時間點:np.interp 會把 j 自己的記錄空洞線性補起來、
        # 在資料結束後夾住最後一筆,與基準的真實資料相減會變成假的「不一致」(124 / baseline 的空洞三顆同步,實測無影響)
        ok_off = _paired(tabs[j]["t"], t_off)
        t_off_j = t_off[ok_off] if ok_off.sum() >= 50 else t_off
        # 固定偏差:有限值的中位數(一個 NaN 不能讓整組永遠不比較)
        da_off = _nanmed_rows(_stack(tabs[i0], ("AccX", "AccY", "AccZ"), t_off_j)
                              - _stack(tabs[j], ("AccX", "AccY", "AccZ"), t_off_j))
        dg_off = _nanmed_rows(_stack(tabs[i0], ("GyrX", "GyrY", "GyrZ"), t_off_j)
                              - _stack(tabs[j], ("GyrX", "GyrY", "GyrZ"), t_off_j))
        da = acc[i0] - acc[j] - da_off
        dg = gyr0 - _stack(tabs[j], ("GyrX", "GyrY", "GyrZ"), t) - dg_off
        na_ = np.sqrt((da ** 2).sum(axis=0))
        ng_ = np.sqrt((dg ** 2).sum(axis=0))
        # 非有限值(任一顆的 NaN、偏差無法估計)= 這個時間點無法比較
        valid = _paired(tabs[j]["t"], t) & np.isfinite(na_) & np.isfinite(ng_)
        dA = _tmean_valid(t, na_, valid, SMOOTH_S)
        dG = _tmean_valid(t, ng_, valid, SMOOTH_S)
        clean = clean0 & valid
        pair_cov[key] = _coverage(t[valid], iv.t0, t_event)
        pairs[key] = {
            "dA_max": _f(_nanmax(dA[clean]), 3) if clean.any() else None,
            "dG_max": _f(_nanmax(dG[clean]), 4) if clean.any() else None,
            "offset_acc": [_f(x, 3) for x in da_off.ravel()], "offset_gyr": [_f(x, 4) for x in dg_off.ravel()],
            "unpaired_n": int((~valid).sum()),
            "unpaired_s": _f(float((~valid).sum()) * (float(np.median(np.diff(t))) if t.size > 2 else 0.0), 2),
            "coverage": _cov_values(pair_cov[key])}
        for level, fa, fg, bucket in (("hit", DA_THR, DG_THR, free_hit),
                                      ("soft", SOFT_FRAC * DA_THR, SOFT_FRAC * DG_THR, free_soft)):
            flag = ((dA > fa) | (dG > fg)) & valid
            for r0, r1 in _runs_idx(flag, t, RUN_GAP_IMU_S):
                if float(t[r1] - t[r0]) < MIN_RUN_S:
                    continue
                onset = float(t[r0])
                rec = {"pair": key, "t0": onset, "t1": float(t[r1]), "dA": _nanmax(dA[r0:r1 + 1]),
                       "dG": _nanmax(dG[r0:r1 + 1]), "level": level}
                if bool(shock[r0]) or onset >= t_event - EVENT_SMEAR_S:        # 以「起點」判斷:故障先於衝擊、持續到衝擊之後仍是故障
                    if level == "hit":
                        impact_runs.append(rec)
                else:
                    bucket.append(rec)
    return {"pairs": pairs, "pair_cov": pair_cov, "free_hit": free_hit, "free_soft": free_soft,
            "impact_runs": impact_runs, "n_shock_windows": len(wins), "offset_basis": off_basis, "ref": i0}


def _finite_median(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None


def _health_flags(tabs: dict[int, dict], iv: ArmedInterval, t_event: float) -> dict:
    """GH / AH / EG / EA / GHz / AHz / T:事件前(含)與事件後分開記。

    gaps:50 ms–1 s 的短暫記錄空洞(紀錄緩衝問題);> 1 s 的空洞另記在 n_long / longest_long_s
    (那是「沒有資料」,由覆蓋檢查處理,不能說成緩衝問題)。
    """
    pre_hit, pre_soft, post = [], [], []
    gaps = {"n": 0, "longest_s": 0.0, "n_pre": 0, "n_long": 0, "longest_long_s": 0.0}
    checked: set[str] = set()          # 實際檢查過的欄位(文字只能宣稱有檢查過的旗標)
    for i, tb in sorted(tabs.items()):
        t = tb["t"]
        arm = (t >= iv.t0) & (t <= iv.t1)
        if arm.sum() < 3:
            continue
        ta = t[arm]
        pre = ta <= t_event
        checked.update(c for c in ("GH", "AH", "EG", "EA", "GHz", "AHz", "T") if c in tb)

        def put(kind: str, level: str, mask: np.ndarray, detail: str):
            if not mask.any():
                return
            k_pre = np.flatnonzero(mask & pre)
            if k_pre.size:
                (pre_hit if level == "hit" else pre_soft).append(
                    {"inst": i, "kind": kind, "t0": float(ta[k_pre[0]]), "n": int(k_pre.size), "detail": detail})
            k_post = np.flatnonzero(mask & ~pre)
            if k_post.size:
                post.append({"inst": i, "kind": kind, "t0": float(ta[k_post[0]]), "n": int(k_post.size)})

        for col, kind in (("GH", "gyro_unhealthy"), ("AH", "accel_unhealthy")):
            if col in tb:
                put(kind, "hit", tb[col][arm] == 0, col + "==0")
        for col, kind in (("EG", "gyro_error_count"), ("EA", "accel_error_count")):
            if col in tb:
                inc = np.concatenate([[False], np.diff(tb[col][arm]) > 0])
                put(kind, "hit", inc, col + " 增加")
        for col, kind in (("GHz", "gyro_rate_drop"), ("AHz", "accel_rate_drop")):
            if col in tb:
                nominal = _finite_median(tb[col])
                if nominal is not None and nominal > 0:
                    put(kind, "soft", tb[col][arm] < GHZ_FRAC * nominal, f"{col}<{GHZ_FRAC:.0%}×{nominal:.0f}")
        if "T" in tb:
            T = tb["T"][arm]
            j = np.searchsorted(ta, ta - T_JUMP_WIN_S)
            put("temp_jump", "soft", np.abs(T - T[j]) > T_JUMP_C, f"{T_JUMP_WIN_S:.0f}s 內溫度變化>{T_JUMP_C:.0f}°C")
        dt = np.diff(ta)
        g = (dt > GAP_NOTE_S) & (dt <= HOLE_S)
        big = dt > HOLE_S
        gaps["n"] = max(gaps["n"], int(g.sum()))
        gaps["longest_s"] = max(gaps["longest_s"], float(dt[g].max()) if g.any() else 0.0)
        gaps["n_pre"] = max(gaps["n_pre"], int((g & (ta[1:] <= t_event)).sum()))
        gaps["n_long"] = max(gaps["n_long"], int(big.sum()))
        gaps["longest_long_s"] = max(gaps["longest_long_s"], float(dt[big].max()) if big.any() else 0.0)
    return {"hit": pre_hit, "soft": pre_soft, "post": post, "gaps": gaps, "checked": sorted(checked)}


def _imu_holes_text(inst_cov: dict[int, dict]) -> str:
    """各 IMU 在統計窗內 > 1 s 沒有資料的區段(含提早結束);相同的空洞合併成一句。"""
    groups: dict[tuple, list[int]] = {}
    for i, c in sorted(inst_cov.items()):
        if c["holes"]:
            tot = sum(h1 - h0 for h0, h1 in c["holes"])
            lh = max(c["holes"], key=lambda h: h[1] - h[0])
            groups.setdefault((round(tot, 1), round(lh[0], 1), round(lh[1], 1), len(c["holes"])), []).append(i)
    bits = []
    for (tot, h0, h1, n), insts in groups.items():
        who = "/".join(f"IMU{i}" for i in insts)
        more = f" 等 {n} 處" if n > 1 else ""
        bits.append(f"{who} 有 {tot:.1f} 秒沒有記錄({h0:.1f}–{h1:.1f} s{more})、該段無法比較")
    return ("(" + ";".join(bits) + ")") if bits else ""


def _imu_text(status: str, cons: dict | None, health: dict, dA: float | None, dG: float | None, post: dict | None,
              n_imu: int = 3, crash_like: bool = True, holes_txt: str = "") -> str:
    gaps = health["gaps"]
    gap_txt = (f"(log 有 {gaps['n']} 處 IMU 短暫記錄空洞、最長 {gaps['longest_s']:.2f} 秒,屬紀錄緩衝問題,不是感測器故障)"
               if gaps["n"] else "")
    if status == HIT:
        bits = []
        for h in health["hit"][:3]:
            bits.append(f"IMU{h['inst']} {h['detail']}(自 {h['t0']:.2f} s)")
        if cons and cons["free_hit"]:
            r = cons["free_hit"][0]
            bits.append(f"IMU {r['pair']} 在 {r['t0']:.2f} s 起不一致(加速度差 {_fmt(r['dA'])} m/s²、角速度差 {_fmt(r['dG'], 3)} rad/s,"
                        f"門檻 {DA_THR} / {DG_THR})且當時沒有撞擊")
        head = "IMU 在事件之前就出現異常:" if crash_like else "IMU 在解鎖飛行期間出現異常:"
        return head + ";".join(bits) + f"。請檢查 IMU 安裝與避震、線路 / 供電,必要時更換感測器。{holes_txt}"
    if status == SOFT:
        bits = [f"IMU{h['inst']} {h['detail']}(自 {h['t0']:.2f} s)" for h in (health["soft"] + health["hit"])[:3]]
        if cons and cons["free_soft"]:
            r = cons["free_soft"][0]
            bits.append(f"IMU {r['pair']} 在 {r['t0']:.2f} s 起略有差異(加速度差 {_fmt(r['dA'])} m/s²、角速度差 {_fmt(r['dG'], 3)} rad/s)")
        return "IMU 有輕微異常跡象:" + ";".join(bits) + f"。尚未達故障門檻,建議留意。{holes_txt}{gap_txt}"
    if status == CLEAR:
        post_txt = ""
        if post:
            post_txt = (f"差異只在撞擊 / 翻覆時才出現(自 {post['t0']:.2f} s,IMU {post['pair']} 加速度差最大 {_fmt(post['dA'], 1)} m/s²),"
                        f"是撞擊的後果。")
        if health["post"]:
            h = min(health["post"], key=lambda x: x["t0"])
            post_txt += f"IMU{h['inst']} {_KIND_ZH.get(h['kind'], h['kind'])}到 {h['t0']:.2f} s(事件之後)才出現,屬後果。"
        num = "、".join(x for x in (f"加速度差最大 {dA:.3f} m/s²" if dA is not None else "",
                                    f"角速度差最大 {dG:.4f} rad/s" if dG is not None else "") if x)
        when = "事件前" if crash_like else "解鎖飛行期間"
        ck = set(health.get("checked", ()))
        flags = []
        if ck & {"GH", "AH"}:
            flags.append("健康旗標 " + " / ".join(c for c in ("GH", "AH") if c in ck) + " 全程為 1")
        if ck & {"EG", "EA"}:
            flags.append("錯誤計數 " + " / ".join(c for c in ("EG", "EA") if c in ck) + " 沒有增加")
        flag_txt = ("、".join(flags) + "," if flags else "(log 沒有 GH / AH / EG / EA 健康欄位,只比較了感測值),")
        return (f"{n_imu} 顆 IMU {when}彼此一致(去除固定偏差後{num or '差異在門檻內'},門檻 {DA_THR} m/s² / {DG_THR} rad/s),"
                f"{flag_txt}沒有 IMU 感測器故障的跡象{holes_txt}。{post_txt}{gap_txt}")
    return ""


@guarded("imu_consistency_health", DOMAIN)
def imu_consistency_health(ctx: Ctx, a: Anchors):
    """三顆 IMU 一致性 + 健康旗標。

    HIT:事件前健康旗標失效(GH / AH = 0、EG / EA 增加),或 IMU 之間差異超標而且不在撞擊 / 衝擊窗內。
    SOFT:差異達門檻 60%、取樣率掉落、溫度突變。CLEAR:兩顆以上 IMU、事件前全程一致且健康、事件前最後一段有資料可比。
    只有一顆 IMU(無法互相比較)、資料太短、沒有任何可比較的樣本、或事件前最後 60 s 大多無法比較 → 乾淨時回 NA
    (不拿單一感測器的自我回報、或缺資料的時段當排除證據)。撞擊後差異變大是後果:只放 values['post_impact'],不改 status。
    """
    did = "imu_consistency_health"
    iv = a.interval
    if iv is None:
        return _na(did, "not_armed", "未解鎖(桌上 / 手持)的 log:沒有飛行中的 IMU 一致性可評估(不適用)。")
    tabs = _imu_tables(ctx)
    if not tabs:
        return _na(did, "no_data", "log 沒有可用的 IMU 訊息(或筆數太少),無法檢查 IMU 一致性。")

    ref = _ref_inst(tabs, iv)
    t, acc = _window_data(tabs, iv, ref)
    events = _shock_events(t, acc, _impacts_in(a, iv))
    hard = [e["t_start"] for e in events if e["peak"] >= HARD_PEAK and iv.t0 <= e["t_start"] <= iv.t1 + 1.0]
    # 事件時間:別的模組填的錨點(t_loss / t_first_hit / t_impact_peak,只取解鎖區間內的有限值)與本地偵測的第一個硬撞擊,
    # 取最早;兩者都沒有才退回 CRASH_CHECK 前 5 s(粗估);都沒有(正常降落)就是上鎖時間
    anc = _anchor_times(a, iv, ("t_loss", "t_first_hit", "t_impact_peak"))
    cands = {}
    if anc:
        cands["anchor"] = min(v for v, _ in anc)
    if hard:
        cands["hard_shock"] = min(hard) - SHOCK_PRE_S
    det = _anchor_times(a, iv, ("t_detect",))
    if cands:
        basis, t_event = min(cands.items(), key=lambda kv: kv[1])
    elif det:
        basis, t_event = "t_detect", det[0][0] - CRASH_LEAD_S
    else:
        basis, t_event = "disarm", iv.t1
    t_event = min(t_event, iv.t1)
    if t_event - iv.t0 < MIN_CHECK_S:
        return _na(did, "insufficient_window", f"事件前的解鎖飛行不足 {MIN_CHECK_S:.0f} 秒,無法檢查 IMU 一致性。",
                   window=[_f(iv.t0, 2), _f(t_event, 2)], window_basis=basis)

    health = _health_flags(tabs, iv, t_event)
    cons = _consistency(tabs, iv, t, acc, events, t_event, ref)
    n_imu = len(tabs)
    crash_like = _crash_like(a)
    inst_cov = {i: _coverage(_imu_valid_t(tb, iv.t0, t_event), iv.t0, t_event) for i, tb in tabs.items()}
    holes_txt = _imu_holes_text(inst_cov)

    dA_max = dG_max = None
    post = None
    pairs = {}
    best_pair, best_cov = None, None
    if cons is not None:
        pairs = cons["pairs"]
        das = [p["dA_max"] for p in pairs.values() if p["dA_max"] is not None]
        dgs = [p["dG_max"] for p in pairs.values() if p["dG_max"] is not None]
        dA_max = max(das) if das else None
        dG_max = max(dgs) if dgs else None
        best_pair, best_cov = _cov_best(cons["pair_cov"])
        if cons["impact_runs"]:
            r0 = min(cons["impact_runs"], key=lambda r: r["t0"])
            post = {"t0": r0["t0"], "pair": r0["pair"],
                    "dA": _nanmax([r["dA"] for r in cons["impact_runs"] if r["dA"] is not None]),
                    "dG": _nanmax([r["dG"] for r in cons["impact_runs"] if r["dG"] is not None])}

    onsets, status, sev = [], CLEAR, "info"
    if health["hit"] or (cons and cons["free_hit"]):
        status, sev = HIT, "high"
        onsets += [h["t0"] for h in health["hit"]] + [r["t0"] for r in (cons["free_hit"] if cons else [])]
    elif health["soft"] or (cons and cons["free_soft"]):
        status, sev = SOFT, "info"
        onsets += [h["t0"] for h in health["soft"]] + [r["t0"] for r in (cons["free_soft"] if cons else [])]
    t_on = min(onsets) if onsets else None

    cov_vals = {"instances": {f"IMU{i}": _cov_values(c) for i, c in sorted(inst_cov.items())},
                "best_pair": best_pair, "ref": None if cons is None else cons["ref"]}
    if status == CLEAR:
        na_ctx, na_txt = None, ""
        if cons is None:
            why = "只有一顆 IMU(或資料太短)" if n_imu < 2 else "資料太短"
            na_ctx = "single_imu" if n_imu < 2 else "insufficient_window"
            na_txt = f"{why},無法做 IMU 之間的一致性檢查;只靠單一 IMU 自己回報的健康旗標不足以排除 IMU 故障。"
        elif dA_max is None and dG_max is None:
            na_ctx = "no_comparison"
            na_txt = ("IMU 之間在統計窗內沒有任何可比較的樣本(記錄缺漏、時間對不上或數值無效),"
                      f"無法做一致性檢查(資料不足,不是『正常』)。{holes_txt}")
        elif best_cov is None or not _cov_ok(best_cov) or best_cov["frac"] * (t_event - iv.t0) < MIN_CHECK_S:
            na_ctx = "data_gap"
            when = "事件前" if crash_like else "上鎖前"
            lf = best_cov["last_frac"] if best_cov else 0.0
            tl = best_cov["t_last"] if best_cov else None
            tail = best_cov["tail_gap_s"] if best_cov else t_event - iv.t0
            na_txt = (f"IMU 之間可比較的資料在{when}最後 {LAST_WIN_S:.0f} 秒只有 {lf*100:.0f}%(最後一筆可比較的資料在 {_fmt(tl)} s,"
                      f"距 {t_event:.2f} s 還有 {tail:.1f} s),無法確認這段期間 IMU 是否一致(資料不足,不是『正常』)。"
                      f"已有資料的部分加速度差最大 {_fmt(dA_max, 3)} m/s²、角速度差最大 {_fmt(dG_max, 4)} rad/s。{holes_txt}")
        if na_ctx is not None:
            return Finding(did, DOMAIN, NA, context=na_ctx,
                           values={"n_imu": n_imu, "window": [_f(iv.t0, 2), _f(t_event, 2)], "window_basis": basis,
                                   "dA_max": _f(dA_max, 3), "dG_max": _f(dG_max, 4), "pairs": pairs,
                                   "coverage": cov_vals, "health": {"gaps": health["gaps"]}, "confidence": "low"},
                           text_zh=na_txt)

    values = {
        "n_imu": n_imu, "window": [_f(iv.t0, 2), _f(t_event, 2)], "window_basis": basis,
        "dA_max": _f(dA_max, 3), "dG_max": _f(dG_max, 4), "pairs": pairs,
        "free_runs": [{k: (_f(v, 3) if isinstance(v, float) or v is None else v) for k, v in r.items()}
                      for r in ((cons["free_hit"] or cons["free_soft"]) if cons else [])[:5]],
        "health_flags": [{**h, "t0": _f(h["t0"], 2)} for h in (health["hit"] + health["soft"])[:8]],
        "post_impact": None if post is None else {k: (_f(v, 3) if isinstance(v, float) or v is None else v)
                                                  for k, v in post.items()},
        "post_event_health": [{**h, "t0": _f(h["t0"], 2)} for h in health["post"][:8]],
        "log_gaps": {"n": health["gaps"]["n"], "n_pre_event": health["gaps"]["n_pre"],
                     "longest_s": _f(health["gaps"]["longest_s"], 3), "n_long": health["gaps"]["n_long"],
                     "longest_long_s": _f(health["gaps"]["longest_long_s"], 2)},
        "coverage": cov_vals,
        "health_checked": health["checked"],
        "thresholds": {"dA": DA_THR, "dG": DG_THR, "soft_frac": SOFT_FRAC, "smooth_s": SMOOTH_S,
                       "ghz_frac": GHZ_FRAC, "t_jump_c": T_JUMP_C, "hole_s": HOLE_S, "last_cov_min": LAST_COV_MIN,
                       "tail_gap_max_s": TAIL_GAP_MAX_S},
        "confidence": "low",
    }
    if cons is not None:
        values["shock_windows"] = cons["n_shock_windows"]
        values["offset_basis"] = cons["offset_basis"]
    ctx_name = {HIT: "fault_before_event", SOFT: "marginal", CLEAR: "consistent"}[status]
    return Finding(did, DOMAIN, status, sev, t_onset=t_on,
                   t_alarm=None if t_on is None else t_on + MIN_RUN_S, context=ctx_name, values=values,
                   text_zh=_imu_text(status, cons, health, dA_max, dG_max, post, n_imu, crash_like, holes_txt))


# ===================================================================== 模組入口

def run(ctx: Ctx, a: Anchors) -> list[Finding]:
    out: list[Finding] = []
    out += vibration_flight_level(ctx, a)
    out += imu_consistency_health(ctx, a)
    return out
