"""地面階段偵測器(領域 ground):解釋 00000124 的失事 ——『降落觸地後飛控沒判定落地 → 在地面被撬翻』。

這個模組有 5 個偵測器,每個一律至少回一筆 Finding(HIT / SOFT 證據、CLEAR 排除、NA 不適用):

  touchdown_not_recognised      觸地衝擊(20 m/s² 以上、dV<3、姿態仍正常)後 2.0 s 內沒有 EV17 / EV18。
                                信心:中 —— 只有 1 個正例(124 飛行 2)與 1 個負例(飛行 1),邊界很薄(0.82 s vs 6.9 s)。
                                沒有 EV28(從未起飛)→ NA no_takeoff;空中(>1.5 m)油門崩潰 → NA airborne_throttle_collapse(spec:不是本偵測器)。
  idle_motor_asymmetry          油門已最低(ThO≤0.03)但馬達 PWM 全距 ≥120 µs(或左右力矩 ≥250 / 油門補償 ≥0.04)持續 ≥0.5 s。
                                信心:高 —— 健康怠速 spread ≤57 µs,失事 218 µs(約 4 倍)。
  gravity_only_tipover          翻倒過程(傾斜 5° → 50°)IMU AccZ 殘差 RMS<0.5 m/s²、無油門、PWM 低於懸停值、角速度單調增:
                                只剩重力在翻,不是推力失效。信心:高(但只有 1 個正例)。
  touchdown_lateral_speed       觸地前 1 s 水平速度 ≥0.35 m/s 或 EKF / VIO 速度差 ≥0.3 m/s。信心:中(衝擊大小本身分不開)。
  ekf_phantom_motion_on_ground  IMU 靜止 + 怠速(≥0.5 s)時 EKF 速度中位數 ≥0.15 m/s 或位移 ≥0.3 m。
                                信心:高 —— 健康怠速 ≤0.05 m/s、0.03 m,失事 0.28 m/s、0.41 m。

資料前提(見 spec/pitfalls):
  * 時間一律是相對秒;ctx 已把 uint8/uint16/int16 轉成 float64。EV 表上鎖當下 1 ms 內寫出的 EV17/18 是『上鎖造成的』,不算落地。
  * IMU[0] 約 400 Hz(有 >50 ms 空洞),CTUN / RCOU / PSCN 約 10 Hz,XKF1 lane 0 約 25 Hz,VISP/VISV 約 27 Hz(有 0.5–1.2 s 空洞)。
    10 Hz 序列的『持續 0.5 s』= 6 個樣本 = 跨度 0.4999 s,所以持續時間比較都留 DUR_TOL 容差。
  * GPS 無定位時欄位是舊值,速度只在 Status≥3 且 NSats≥6 才採用。沒有測距儀:agl = CTUN.Alt − 解鎖後 3 s 的 Alt 中位數。
  * 分析區間是 a.interval;撞擊 / 失控之後(t_stop)的樣本不進特徵(倒扣怠速、EKF 垃圾值都不算地面階段證據)。

錨點:本模組設定 a.t_touch = 『和這次事件相關』的觸地時間,只在下列情況設定:
  * 有事件參考點(t_warn,沒有則 t_loss / t_first_hit / t_detect)時:觸地在 [參考點 − 10 s, 參考點 + 1 s] 內,
    且觸地到參考點之間沒有重新起飛(EV28,或 agl > 1.5 m 持續 0.5 s);
  * 沒有任何參考點時:只有 touchdown_not_recognised HIT 才設定(SOFT 的彈跳 / 輕觸地不設)。
  更早的觸地(之後又起飛,或離事件太遠)不是這次事件的觸地:engine 會拿 t_touch 當 t_first_anomaly 與情境判定時間點,
  用錯會把空中失控標成 GROUND_TIPOVER。
不依賴其他模組先跑過:衝擊簇由 IMU 本地計算(a.impacts 的 cls 欄位 schema 尚未固定,不使用);a.t_loss / a.t_first_hit 有就用來截止,
沒有就用本地硬撞擊 / 『貼地時』傾斜 60° 截止(空中急轉 / FLIP 的大角度不算,否則會把之後的地面階段證據整段丟掉)。
『貼地』= 飛控認為已落地,或 agl ≤ 1.5 m;怠速不對稱、重力翻倒都只看貼地的樣本(空中收油門不是地面翻覆機制)。

所有門檻都是模組常數(UPPER_CASE),註解標了 124 的實測邊界;唯一的失事樣本代表它們只是『在這台機體上有餘裕』的啟發值。
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np

from . import util
from .ctx import Ctx
from .guard import guarded
from .types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval, Finding

DOMAIN = "ground"

ID_TOUCHDOWN = "touchdown_not_recognised"
ID_IDLE = "idle_motor_asymmetry"
ID_TIPOVER = "gravity_only_tipover"
ID_LATERAL = "touchdown_lateral_speed"
ID_PHANTOM = "ekf_phantom_motion_on_ground"

G0 = 9.80665
RAD2DEG = 57.29577951308232
DUR_TOL = 0.01                  # 10 Hz 序列的樣本間隔 0.0999 s,N 個間隔的跨度略小於 N×0.1

# ------------------------------------------------------------------ 觸地衝擊(IMU[0] |a|)
SHOCK_MIN = 20.0                # m/s²:基準 log 最大 10.1–16.5(手持);124 兩次受控觸地 25.5 / 28.9
SHOCK_MERGE_GAP = 0.3           # s:彈跳合併成同一簇(124 觸地簇 1255.225–1255.513 含 3 次彈跳)
GROUND_DV_MAX = 3.0             # m/s:dV=∫(|a|−g)₊dt;觸地 0.64 / 1.20,失事撞擊 4.61 / 6.74(<3 = 觸地型)
HARD_PEAK = 40.0                # m/s²:硬撞擊 = 峰值 ≥40 且 dV ≥3(29 到 59 之間沒有任何資料)
IMU_DV_DT_MAX = 0.02            # s:積分時單一間隔上限(IMU 有 >50 ms 空洞,不能把空洞積成衝量)
DV_PAD_S = 0.1                  # s:dV 積分範圍 = [簇起點−0.1, 簇終點+0.1](spec;只積 >20 的樣本會少算上升 / 下降段:
                                #    124 觸地 0.45→0.66 / 0.98→1.23、硬撞擊 4.47→4.69 / 6.62→6.94,和研究值 0.64 / 1.20 / 4.6 / 6.9 一致)
STOP_TILT_DEG = 60.0            # 傾斜 ≥60° 之後視為已翻覆,不再當地面階段特徵(沒有 t_loss 時的備援截止)

# ------------------------------------------------------------------ 落地辨識
RECOGNISE_WIN = 2.0             # s:觸地後這段時間內要出現 EV17 / EV18(正常降落 EV17 +0.82 s,EV18 +1.62 s;失事 6.9 s 都沒有)
RECOGNISE_LOOKBACK = 0.5        # s:EV17 比衝擊簇起點早最多 0.5 s 也算已辨識(落地偵測可能先於衝擊成立)
DISARM_EV_WINDOW = 0.001        # s:上鎖當下 1 ms 內的 EV17/18 是上鎖寫的(與 Ctx 一致)
AE_NORMAL_MAX = 10.0            # deg:觸地瞬間姿態仍正常(124 觸地前 ae 最大 4.7,飛行 1 為 2.0)
NEAR_GROUND_AGL = 1.5           # m:agl>1.5 m 的衝擊不是觸地(空中碰撞 / 其他),交給撞擊分析
THO_IDLE = 0.03                 # 油門輸出≤0.03 視為怠速(崩到 0 的樣本是 0.000;MOTB.ThrOut 被混控夾在 0.5×hover,不能用 ==0)
THO_DROP_FRAC = 0.65            # ThO ≤ 0.65×ThH 視為油門開始收(健康飛行最低 0.78×ThH)
THO_SUSTAIN_MIN_S = 0.19        # ≥3 個 CTUN 樣本(跨度 0.2 s)才算『持續』怠速
IDLE_AGL_MAX = 1.0              # m:無衝擊備援路徑的離地上限
IDLE_CRT_MAX = 30.0             # cm/s:無衝擊備援路徑的垂直速度上限
WINDUP_WIN = 3.0                # s:觸地後看位置控制器是否在地面纏繞(windup)
WINDUP_ACCEL = 2.0              # m/s²:hypot(PSCN.TAN, PSCE.TAE) > 2(巡航 / 懸停 <1)
WINDUP_MIN_S = 0.5
WINDUP_CRT = 15.0               # cm/s
WINDUP_THO_FRAC = 0.6
WINDUP_GYRO_DEG = 30.0          # deg/s:纏繞判定用的『靜止』較寬鬆(只要不是在翻)
WINDUP_ACC_DEV = 0.5            # m/s²
AIR_COLLAPSE_MIN_S = 3.0        # s:空中(agl>1.5 m、未落地)油門 ≤0.03 累計 ≥3 s …
AIR_COLLAPSE_WIN_S = 8.0        # … 於 8 s 內 → 『空中油門崩潰』,不是本偵測器(124 空中 ThO 最低 0.11,從未成立)
# 觸地和事件的關係(a.t_touch 只給『這次事件的』觸地)
TOUCH_REF_LOOKBACK = 10.0       # s:觸地最早可比事件參考點早 10 s(124:觸地 1255.225 → t_warn 1255.91 只差 0.69 s、
                                #    → CRASH_CHECK 1262.125 差 6.9 s;觸地後位置控制器纏繞到翻倒需要數秒)
TOUCH_REF_LEAD = 1.0            # s:觸地最晚可比參考點晚 1 s(和 touchdown_lateral_speed 的 [t_warn−5, t_warn+1] 一致)
REFLY_AGL = NEAR_GROUND_AGL     # m:觸地後 agl > 1.5 m 持續 … (124 觸地後到翻倒前 agl 最高 +0.16 m;翻倒後氣壓讀到 −1.9 m)
REFLY_MIN_S = 0.5               # … 0.5 s(5 個 CTUN 樣本,不吃單點氣壓雜訊)= 已重新起飛,那次觸地和之後的事件無關

# ------------------------------------------------------------------ 怠速馬達不對稱
IDLE_SPREAD_FRAC = 120.0 / 900.0   # 全距 / (PWM_MAX−PWM_MIN) ≥ 0.133(=120 µs):健康怠速 ≤57 µs,失事 218 µs
IDLE_ROLLMOM_US = 250.0            # µs:hexa-X 左 3 顆總和 − 右 3 顆總和(健康怠速 ≤97,失事中位數 406、峰值 565)
IDLE_BOOST = 0.04                  # MOTB.ThrOut − CTUN.ThO(健康怠速 ≤0.021,失事 0.077)
IDLE_MIN_S = 0.5                   # 持續 ≥0.5 s 才 HIT(飛行 1 觸地有 1 個孤立樣本 189 µs,只有 0.1 s)
IDLE_SOFT_MIN_S = 0.2              # 0.2–0.5 s 為 SOFT
SPOOL_FRAC = 0.05                  # 馬達已轉起來:最小 PWM ≥ PWM_MIN + 5% 全距(=1145)
# hexa-X(FRAME_CLASS 2 / TYPE 1;以 RCOU 對 RATE.ROut/POut 迴歸驗證):馬達編號 → 側邊
HEXA_X = (2, 1)
HEXA_LEFT = frozenset({2, 3, 6})
HEXA_RIGHT = frozenset({1, 4, 5})
HEXA_FRONT = frozenset({3, 5})
HEXA_REAR = frozenset({4, 6})
FRAME_MOTORS = {1: 4, 2: 6, 3: 8, 4: 8, 5: 6, 7: 3, 10: 8}     # FRAME_CLASS → 馬達數(沒有 SERVOn_FUNCTION 時備援)

# ------------------------------------------------------------------ 重力翻倒
TIP_START_DEG = 5.0                # 傾斜離開 5° 以內算翻倒開始
TIP_END_DEG = 50.0                 # 到 50° 結束
TIP_BIN_S = 0.05                   # IMU 分箱
TIP_MIN_WIN_S = 0.3                # 視窗太短(快速翻轉)不評估
TIP_RMS_MAX = 0.5                  # m/s²:AccZ − (−g·cosφ·cosθ) 的 RMS;失事 0.28(最大 0.65)
TIP_THO_MAX = 0.02                 # 視窗內 ThO 的 90 百分位;失事 0.000
TIP_GYRO_BIN_S = 0.1
TIP_GYRO_RHO = 0.8                 # 角速度對時間的 Spearman 相關(單調成長);失事 ≈1.0
TIP_GYRO_GROWTH = 1.5              # 後 3 箱 / 前 3 箱 平均;失事 ≈5
HOVER_MIN_SAMPLES = 50             # 估懸停 PWM 至少 5 s 的穩定懸停樣本
HOVER_THO_BAND = 0.25              # |ThO−ThH| ≤25%·ThH
HOVER_CRT_MAX = 30.0               # cm/s
HOVER_GUARD_S = 5.0                # 懸停取樣截止在翻倒視窗前 5 s

# ------------------------------------------------------------------ 觸地水平速度
LAT_WIN = 1.0                      # s:觸地前 1 s
LAT_SPEED = 0.35                   # m/s:正常著陸 0.12–0.17,失事 VISV 1.14 / EKF 0.54
LAT_MISMATCH = 0.30                # m/s:|v_ekf − v_vio|;正常 0.03,失事 0.61
LAT_SOFT_FRAC = 0.7                # SOFT = 兩個門檻的 70%(沒有資料支持更精細的分界)
GPS_MIN_STATUS = 3
GPS_MIN_SATS = 6
LAT_IMPACT_WIN = 0.6               # s:觸地後側向衝擊視窗(AccY 峰值:失事 12.5 vs 飛行 1 7.4)

# ------------------------------------------------------------------ EKF 地面幻影運動
PH_HOP = 0.05                      # s:靜止判定視窗的滑動步長(對齊敏感度:固定 0.2 s 分箱會把 0.6 s 的靜止切成 0.4 s)
PH_WIN = 0.2                       # s:視窗長
PH_GYRO_DEG = 6.0                  # deg/s:視窗平均角速度(怠速 2.5–5.2;開始翻時 ≥9.6)
PH_ACC_DEV = 0.8                   # m/s²:|a_mean − 9.8|
PH_ACC_SD = 0.6                    # m/s²:a 的標準差(失事靜止段 0.08–0.55)
PH_MIN_S = 0.5                     # 靜止 + 怠速至少 0.5 s
PH_SPEED = 0.15                    # m/s:EKF 速度中位數(健康怠速最大 0.05,失事 0.28)
PH_DISP = 0.30                     # m:EKF 位移(健康 ≤0.03,失事 0.41)
PH_VIO_DISP = 0.15                 # m:同一段 VIO 位移 ≥0.15 表示機身真的動了 → 降為 SOFT
PH_VIO_MIN_SPAN = 0.3              # s:VISP 資料跨度不足(例如觸地時 VIO 中斷)就不做交叉驗證
PH_TILT_END_DEG = 10.0             # 地面階段結束 = 傾斜 ≥10°(算『觸地後 EKF 漂移』用)
PH_VIO_GAP_S = 0.5                 # s:VISP 相鄰兩筆間隔 ≥0.5 s 視為 VIO 中斷(正常 ~27 Hz、間隔 ≤0.06 s;124 觸地時 1255.128–1256.287 中斷 1.16 s)


# =================================================================== 小工具

def _f(x, nd: int = 4):
    """numpy 純量 → JSON 友善 float(nan / None → None)。"""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _fmt(x, spec: str = ".2f", default: str = "—") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return default
    return format(x, spec)


def _vz_txt(v: float | None) -> str:
    """CTUN.CRt(cm/s,負 = 下降)→ 『下降 30 cm/s』。"""
    if v is None:
        return "垂直速度不明"
    return f"下降 {abs(v):.0f} cm/s" if v < 0 else f"上升 {v:.0f} cm/s"


def _fd(det_id: str, status: str, severity: str = "info", **kw) -> Finding:
    return Finding(det_id, DOMAIN, status, severity, **kw)


def _na(det_id: str, context: str, text: str, **values) -> Finding:
    return _fd(det_id, NA, "info", context=context, values=values, text_zh=text)


def _not_armed(det_id: str) -> Finding:
    return _na(det_id, "not_armed", "未解鎖(桌上 / 手持 log),不做地面階段失事分析。")


def _hold(t_src: np.ndarray, v: np.ndarray, t_q) -> np.ndarray:
    """前一筆樣本保持(10 Hz 序列對齊到別的時間軸;第一筆之前取第一筆)。"""
    idx = np.clip(np.searchsorted(t_src, np.asarray(t_q, dtype=float), side="right") - 1, 0, None)
    return v[idx]


def _span(t: np.ndarray, t0: float, t1: float) -> tuple[int, int]:
    return int(np.searchsorted(t, t0, side="left")), int(np.searchsorted(t, t1, side="right"))


def _cols(tb: dict | None, *names: str) -> bool:
    return tb is not None and all(n in tb for n in names)


def _nanmax(x) -> float | None:
    """有限值的最大值;沒有任何有限值回 None(不產生 All-NaN 警告,也不把 NaN 帶進 values / 文字)。"""
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    return float(x.max()) if x.size else None


def _nanmed(x) -> float | None:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None


def _disp(t: np.ndarray, x: np.ndarray, y: np.ndarray, t0: float, t1: float):
    """[t0, t1] 內第一筆到最後一筆『有限值』樣本的水平位移:(位移, 起點時間, 終點時間);有限樣本 <2 回 None。"""
    m = (t >= t0) & (t <= t1) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return None
    k = np.flatnonzero(m)
    return float(np.hypot(x[k[-1]] - x[k[0]], y[k[-1]] - y[k[0]])), float(t[k[0]]), float(t[k[-1]])


def _spearman(y: np.ndarray) -> float:
    """y 對時間順序的 Spearman 相關(無併列處理,連續量夠用)。"""
    n = y.size
    if n < 3:
        return float("nan")
    r = np.argsort(np.argsort(y)).astype(float)
    x = np.arange(n, dtype=float)
    if np.std(r) == 0:
        return float("nan")
    return float(np.corrcoef(x, r)[0, 1])


# =================================================================== IMU / 衝擊簇

@dataclass
class Shock:
    """一個加速度衝擊簇(|a|>SHOCK_MIN,間隔 ≤0.3 s 合併)。"""
    t_start: float
    t_end: float
    t_peak: float
    peak: float          # m/s²
    dv: float            # m/s:∫(|a|−g)₊dt

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def cls(self) -> str:
        if self.peak >= HARD_PEAK and self.dv >= GROUND_DV_MAX:
            return "hard_impact"
        if self.dv < GROUND_DV_MAX:
            return "ground_contact"
        return "long_shock"


def _imu_acc(ctx: Ctx, t0: float, t1: float):
    """IMU[0] 加速度:(t, acc(N,3), |a|) 在 [t0, t1];缺任一軸回 None。"""
    s = [ctx.series(f"IMU.Acc{c}") for c in "XYZ"]
    if any(x is None for x in s):
        return None
    t = s[0][0]
    n = min(x[0].size for x in s)
    i0, i1 = _span(t[:n], t0, t1)
    if i1 - i0 < 3:
        return None
    acc = np.stack([x[1][i0:i1] for x in s], axis=1)
    return t[i0:i1], acc, np.sqrt(np.sum(acc * acc, axis=1))


def _imu_gyro_deg(ctx: Ctx, t0: float, t1: float):
    """IMU[0] 角速度範數 (t, deg/s);缺軸回 None。"""
    s = [ctx.series(f"IMU.Gyr{c}") for c in "XYZ"]
    if any(x is None for x in s):
        return None
    t = s[0][0]
    n = min(x[0].size for x in s)
    i0, i1 = _span(t[:n], t0, t1)
    if i1 - i0 < 3:
        return None
    g = np.stack([x[1][i0:i1] for x in s], axis=1)
    return t[i0:i1], np.sqrt(np.sum(g * g, axis=1)) * RAD2DEG


def find_shocks(t: np.ndarray, anorm: np.ndarray) -> list[Shock]:
    """|a|>SHOCK_MIN 的樣本依間隔 ≤SHOCK_MERGE_GAP 合併成簇,算峰值與 dV(積分範圍前後各加 DV_PAD_S)。向量化。"""
    idx = np.flatnonzero(anorm > SHOCK_MIN)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(t[idx]) > SHOCK_MERGE_GAP)
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk, [idx.size - 1]])
    out: list[Shock] = []
    for s, e in zip(starts, ends):
        ia, ib = int(idx[s]), int(idx[e])
        seg_t, seg_a = t[ia:ib + 1], anorm[ia:ib + 1]
        ja = int(np.searchsorted(t, t[ia] - DV_PAD_S, side="left"))
        jb = int(np.searchsorted(t, t[ib] + DV_PAD_S, side="right"))
        it, ia_ = t[ja:jb], anorm[ja:jb]
        f = np.maximum(np.nan_to_num(ia_, nan=G0) - G0, 0.0)
        dt = np.clip(np.diff(it), 0.0, IMU_DV_DT_MAX)
        dv = float(np.sum(0.5 * (f[1:] + f[:-1]) * dt)) if it.size > 1 else 0.0
        k = int(np.nanargmax(seg_a))          # 簇內至少有一個 >20 的有限樣本;argmax 會把 NaN 當最大值
        out.append(Shock(float(seg_t[0]), float(seg_t[-1]), float(seg_t[k]), float(seg_a[k]), dv))
    return out


_SHOCK_CACHE: "weakref.WeakKeyDictionary[Ctx, dict]" = weakref.WeakKeyDictionary()


def _shocks(ctx: Ctx, iv: ArmedInterval) -> list[Shock] | None:
    """解鎖區間內的衝擊簇(同一個 ctx + 區間只算一次);沒有 IMU(或 |a| 全是 NaN,無法判斷有沒有衝擊)回 None。"""
    cache = _SHOCK_CACHE.setdefault(ctx, {})
    key = (iv.t0, iv.t1)
    if key not in cache:
        imu = _imu_acc(ctx, iv.t0 - 1.0, iv.t1)
        cache[key] = None if imu is None or not np.isfinite(imu[2]).any() else \
            [s for s in find_shocks(imu[0], imu[2]) if iv.t0 <= s.t_start <= iv.t1]
    return cache[key]


# =================================================================== 姿態 / 高度 / 落地狀態

def _att_arrays(ctx: Ctx, t0: float, t1: float):
    """ANG(400 Hz)優先、ATT(10 Hz)備援:dict(t, roll, pitch, yaw, droll, dpitch, dyaw, src)。"""
    for name in ("ANG", "ATT"):
        tb = ctx.table(name)
        if _cols(tb, "Roll", "Pitch", "Yaw", "DesRoll", "DesPitch", "DesYaw"):
            i0, i1 = _span(tb["t"], t0, t1)
            if i1 - i0 >= 2:
                return {"t": tb["t"][i0:i1], "roll": tb["Roll"][i0:i1], "pitch": tb["Pitch"][i0:i1],
                        "yaw": tb["Yaw"][i0:i1], "droll": tb["DesRoll"][i0:i1], "dpitch": tb["DesPitch"][i0:i1],
                        "dyaw": tb["DesYaw"][i0:i1], "src": name}
    return None


def _ae_max(ctx: Ctx, t0: float, t1: float) -> float | None:
    """[t0, t1] 內推力向量誤差 ae 的最大值(deg);沒有姿態資料回 None。"""
    att = _att_arrays(ctx, t0, t1)
    if att is None:
        return None
    ae = util.thrust_vector_error_deg(att["roll"], att["pitch"], att["yaw"], att["droll"], att["dpitch"], att["dyaw"])
    return _nanmax(ae)                    # 全是 NaN = 沒有姿態資料(None),不能當『姿態正常』


def _ground_ref(ctx: Ctx, iv: ArmedInterval) -> float | None:
    """解鎖後地面高度基準(ctx.ground_alt);NaN(前 3 s 的 Alt 全是 NaN)當成沒有。"""
    g = ctx.ground_alt(iv)
    return float(g) if g is not None and np.isfinite(g) else None


def _agl(ctx: Ctx, iv: ArmedInterval, t: float) -> float | None:
    """t 當下(最後一筆 ≤t 的 CTUN)Alt − 解鎖後地面基準(沒有測距儀;氣壓漂移約 0.15 m,所以可能略小於 0)。"""
    tb = ctx.table("CTUN")
    g = _ground_ref(ctx, iv)
    if not _cols(tb, "Alt") or g is None or not tb["t"].size or tb["t"][0] > t:
        return None
    v = float(_hold(tb["t"], tb["Alt"], [t])[0]) - g
    return v if np.isfinite(v) else None


def _near_ground(ctx: Ctx, iv: ArmedInterval, t) -> tuple[np.ndarray, bool]:
    """t 每個樣本是否『貼地』:飛控認為已落地,或 agl ≤ NEAR_GROUND_AGL。回傳 (mask, gated)。
    沒有高度資料時 gated=False、mask 全 True(分不出空中 / 地面,不排除任何樣本 = 舊行為);
    個別 agl 樣本是 NaN 也當貼地(不排除)。"""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    tb, g = ctx.table("CTUN"), _ground_ref(ctx, iv)
    if not _cols(tb, "Alt") or g is None or not tb["t"].size:
        return np.ones(t.size, dtype=bool), False
    agl = _hold(tb["t"], tb["Alt"], t) - g
    with np.errstate(invalid="ignore"):
        return _landed_series(ctx, iv, t) | ~(agl > NEAR_GROUND_AGL), True


def _vz_pre(ctx: Ctx, t: float) -> float | None:
    """觸地前 [t−5, t−0.5] 的 CTUN.CRt 中位數(cm/s,負=下降)。"""
    tb = ctx.table("CTUN")
    if not _cols(tb, "CRt"):
        return None
    v = util.median_in(tb["t"], tb["CRt"], t - 5.0, t - 0.5)
    return None if not np.isfinite(v) else float(v)


def _landed_series(ctx: Ctx, iv: ArmedInterval, t: np.ndarray) -> np.ndarray:
    """t 每個樣本的『已落地』狀態(EV28 離地 / EV18 落地;與 ctx.landed_at 同義,向量化)。"""
    steps = ctx.landed_steps(iv)
    st = np.array([s[0] for s in steps], dtype=float)
    sv = np.array([s[1] for s in steps], dtype=bool)
    return sv[np.clip(np.searchsorted(st, t, side="right") - 1, 0, None)]


def _first_tilt(ctx: Ctx, t0: float, t1: float, deg: float, after: float | None = None) -> float | None:
    """ANG / ATT 傾斜角首次 ≥deg 的時間(after 之後);沒有姿態資料或沒達到回 None。"""
    att = _att_arrays(ctx, t0 if after is None else max(t0, after), t1)
    if att is None:
        return None
    tilt = util.tilt_deg(att["roll"], att["pitch"])
    k = np.flatnonzero(tilt >= deg)
    return float(att["t"][k[0]]) if k.size else None


def _first_ground_tilt(ctx: Ctx, iv: ArmedInterval, deg: float) -> float | None:
    """解鎖區間內第一次『貼地時』(已落地或 agl ≤1.5 m)傾斜 ≥deg 的時間;沒有回 None。
    空中的大角度(急轉、FLIP)不是地面翻覆,不能拿來截掉之後的地面階段證據。"""
    att = _att_arrays(ctx, iv.t0, iv.t1)
    if att is None:
        return None
    with np.errstate(invalid="ignore"):
        kb = np.flatnonzero(util.tilt_deg(att["roll"], att["pitch"]) >= deg)
    if not kb.size:
        return None
    near, _ = _near_ground(ctx, iv, att["t"][kb])
    k = kb[near]
    return float(att["t"][k[0]]) if k.size else None


def _t_stop(ctx: Ctx, a: Anchors, iv: ArmedInterval, shocks: list[Shock] | None) -> float:
    """地面階段特徵的截止時間:失控 / 第一次硬撞擊 / 貼地時傾斜 ≥60° 的最早者,否則上鎖時間。
    (撞擊後的倒扣怠速、EKF 垃圾值都不是地面階段證據。)"""
    cands = [iv.t1]
    for v in (a.t_loss, a.t_first_hit):
        if v is not None and iv.t0 <= v <= iv.t1:
            cands.append(float(v))
    if shocks:
        hard = [s.t_start for s in shocks if s.cls == "hard_impact"]
        if hard:
            cands.append(min(hard))
    t_tilt = _first_ground_tilt(ctx, iv, STOP_TILT_DEG)
    if t_tilt is not None:
        cands.append(t_tilt)
    return float(min(cands))


def _event_ref(a: Anchors) -> tuple[float | None, str]:
    """a.t_touch 要對應的『事件』時間:t_warn,沒有則 t_loss / t_first_hit / t_detect;都沒有回 (None, '')。"""
    for name in ("t_warn", "t_loss", "t_first_hit", "t_detect"):
        v = getattr(a, name)
        if v is not None and np.isfinite(v):
            return float(v), name
    return None, ""


def _refly(ctx: Ctx, iv: ArmedInterval, t_td: float, t_end: float) -> dict | None:
    """(t_td, t_end] 之間是否重新起飛:EV28(飛控宣告離地),或 agl > REFLY_AGL 持續 ≥REFLY_MIN_S。
    回傳最早的 {t, how: 'EV28' | 'agl', agl_m} 或 None。"""
    if t_end <= t_td:
        return None
    out = None
    ev_t, ev_id = ctx.ev_rows()
    k = np.flatnonzero((ev_id == 28) & (ev_t > t_td) & (ev_t <= t_end))
    if k.size:
        out = {"t": float(ev_t[k[0]]), "how": "EV28", "agl_m": None}
    tb, g = ctx.table("CTUN"), _ground_ref(ctx, iv)
    if _cols(tb, "Alt") and g is not None:
        i0, i1 = int(np.searchsorted(tb["t"], t_td, side="right")), int(np.searchsorted(tb["t"], t_end, side="right"))
        tt, agl = tb["t"][i0:i1], tb["Alt"][i0:i1] - g
        with np.errstate(invalid="ignore"):
            r = util.runs(agl > REFLY_AGL, tt, REFLY_MIN_S - DUR_TOL, max_gap=0.25)
        if r and (out is None or r[0][0] < out["t"]):
            out = {"t": r[0][0], "how": "agl", "agl_m": _nanmax(agl[r[0][2]:r[0][3] + 1])}
    return out


def _relation(ctx: Ctx, iv: ArmedInterval, t_td: float, agl: float | None, ref_t: float | None, t_end: float) -> dict | None:
    """觸地候選和事件的關係:None = 相關;否則 {'why': 'far_from_event' | 'took_off_again', ...}。
    有參考點:觸地要在 [ref − 10 s, ref + 1 s] 內;不論有沒有參考點,觸地到 min(ref, t_end) 之間都不能重新起飛。
    離地 >1.5 m 的衝擊沒有『落地後再起飛』可言,只看時間窗(交給 airborne_shock 分支)。"""
    if ref_t is not None and not (ref_t - TOUCH_REF_LOOKBACK <= t_td <= ref_t + TOUCH_REF_LEAD):
        return {"why": "far_from_event", "lead_s": _f(ref_t - t_td, 2)}
    if agl is not None and agl > NEAR_GROUND_AGL:
        return None
    rf = _refly(ctx, iv, t_td, t_end if ref_t is None else min(ref_t, t_end))
    if rf is not None:
        return {"why": "took_off_again", "t_refly": _f(rf["t"], 3), "how": rf["how"], "refly_agl_m": _f(rf["agl_m"], 2)}
    return None


# =================================================================== 觸地候選

@dataclass
class Touchdown:
    shock: Shock
    ae: float | None          # 衝擊前 0.5 s 內姿態誤差最大值(deg);None = 沒有姿態資料
    agl: float | None
    vz_pre: float | None      # cm/s


def _touchdowns(ctx: Ctx, iv: ArmedInterval, shocks: list[Shock]) -> list[Touchdown]:
    """觸地型衝擊候選:dV<3、尚未落地(L==False)、衝擊前姿態仍正常(ae<10°)。"""
    out: list[Touchdown] = []
    for s in shocks:
        if s.dv >= GROUND_DV_MAX:
            continue
        if ctx.landed_at(s.t_start, iv):
            continue                     # 已落地狀態下的碰撞(搬動 / 地面彈跳),不是觸地
        ae = _ae_max(ctx, s.t_start - 0.5, s.t_start)
        if ae is not None and ae >= AE_NORMAL_MAX:
            continue                     # 姿態已失控:這是後果撞擊
        out.append(Touchdown(s, ae, _agl(ctx, iv, s.t_start), _vz_pre(ctx, s.t_start)))
    return out


def _recognition(ctx: Ctx, iv: ArmedInterval, t_c: float) -> dict:
    """觸地後的落地辨識事件:EV17 / EV18 的時間差(排除上鎖寫的)、是否在窗內。
    open_end(log 在解鎖中結束)沒有上鎖,區間終點的 EV17/18 不是上鎖寫的,照常算。"""
    ev_t, ev_id = ctx.ev_rows()
    t17 = t18 = None
    ignored: list[int] = []
    for tt, i in zip(ev_t, ev_id):
        if i not in (17, 18):
            continue
        if not iv.open_end and tt >= iv.t1 - DISARM_EV_WINDOW:
            if tt <= iv.t1 + 0.01:
                ignored.append(int(i))        # 上鎖那一刻寫出來的,不算
            continue
        if tt < t_c - RECOGNISE_LOOKBACK or tt > iv.t1:
            continue
        if i == 17 and t17 is None:
            t17 = float(tt)
        if i == 18 and t18 is None:
            t18 = float(tt)
    firsts = [x for x in (t17, t18) if x is not None]
    first = min(firsts) if firsts else None
    return {"t17": t17, "t18": t18, "dt17": None if t17 is None else t17 - t_c,
            "dt18": None if t18 is None else t18 - t_c,
            "recognised": first is not None and first <= t_c + RECOGNISE_WIN,
            "ignored_at_disarm": sorted(set(ignored))}


def _tho_collapse(ctx: Ctx, iv: ArmedInterval, t_c: float) -> dict:
    """觸地後油門收到怠速的時間:t_drop(首次 ≤0.65·ThH)、t_zero(首次 ≤0.03)、t_sustained(持續怠速段起點)。"""
    out = {"t_drop": None, "t_zero": None, "t_sustained": None}
    tb = ctx.table("CTUN")
    if not _cols(tb, "ThO"):
        return out
    t, tho = tb["t"], tb["ThO"]
    thh = tb["ThH"] if "ThH" in tb else np.full_like(tho, np.nan)
    m = (t >= t_c - 0.1) & (t <= min(t_c + RECOGNISE_WIN, iv.t1))
    drop = m & (tho <= THO_DROP_FRAC * np.where(np.isfinite(thh), thh, np.inf))
    zero = m & (tho <= THO_IDLE)
    if drop.any():
        out["t_drop"] = float(t[np.flatnonzero(drop)[0]])
    if zero.any():
        out["t_zero"] = float(t[np.flatnonzero(zero)[0]])
    in_iv = (t >= iv.t0) & (t <= iv.t1)
    for s, e, _, _ in util.runs(in_iv & (tho <= THO_IDLE), t, THO_SUSTAIN_MIN_S, max_gap=0.25):
        if e >= t_c and s <= t_c + RECOGNISE_WIN:
            out["t_sustained"] = float(s)
            break
    return out


def _ldet_info(ctx: Ctx, iv: ArmedInterval, t_c: float) -> dict:
    """落地偵測器內部計數(LDET 只在旗標變化時記錄;bit 意義是由行為推定的,報告標『推定』)。"""
    tb = ctx.table("LDET")
    if not _cols(tb, "Flags", "Count"):
        return {}
    m = (tb["t"] >= t_c) & (tb["t"] <= min(t_c + 6.0, iv.t1))
    if not m.any():
        return {"count_max": None}
    fl = tb["Flags"][m].astype(np.int64)
    k = np.flatnonzero((fl & 256) != 0)
    return {"count_max": int(np.max(tb["Count"][m])),
            "bit8_t": None if not k.size else float(tb["t"][m][k[0]])}


def _windup(ctx: Ctx, iv: ArmedInterval, t_c: float, t_end: float) -> dict:
    """觸地後 3 s 位置控制器是否在地面纏繞:IMU 靜止 + 油門低 + 無垂直運動 + 控制器加速度需求 >2 m/s²(≥0.5 s)。
    另外回報最大傾斜指令、最大姿態誤差、EKF 水平漂移(只作證據,不影響狀態)。"""
    out: dict = {}
    t1 = min(t_c + WINDUP_WIN, t_end, iv.t1)
    att = _att_arrays(ctx, t_c, t1)
    if att is not None:
        out["lean_max_deg"] = _f(_nanmax(np.hypot(att["droll"], att["dpitch"])), 2)
        ae = util.thrust_vector_error_deg(att["roll"], att["pitch"], att["yaw"], att["droll"], att["dpitch"], att["dyaw"])
        out["ae_max_deg"] = _f(_nanmax(ae), 2)
    xk = ctx.table("XKF1")
    if _cols(xk, "PN", "PE"):
        d = _disp(xk["t"], xk["PN"], xk["PE"], t_c, t1)
        if d is not None:
            out["ekf_drift_m"] = _f(d[0], 2)
    pn, pe, ct = ctx.table("PSCN"), ctx.table("PSCE"), ctx.table("CTUN")
    if not (_cols(pn, "TAN") and _cols(pe, "TAE") and _cols(ct, "ThO", "ThH", "CRt")):
        out["windup"] = None
        return out
    tg = pn["t"]
    m = (tg >= t_c) & (tg <= t1)
    if m.sum() < 3:
        out["windup"] = None
        return out
    tg = tg[m]
    acc_cmd = np.hypot(pn["TAN"][m], _hold(pe["t"], pe["TAE"], tg))
    out["pos_acc_cmd_max"] = _f(_nanmax(acc_cmd), 2)
    cond = (acc_cmd > WINDUP_ACCEL) \
        & (_hold(ct["t"], ct["ThO"], tg) < WINDUP_THO_FRAC * _hold(ct["t"], ct["ThH"], tg)) \
        & (np.abs(_hold(ct["t"], ct["CRt"], tg)) < WINDUP_CRT)
    gy, ac = _imu_gyro_deg(ctx, t_c, t1), _imu_acc(ctx, t_c, t1)
    if gy is not None and ac is not None:
        gb = _hold(gy[0], gy[1], tg)
        ab = _hold(ac[0], ac[2], tg)
        cond &= (gb < WINDUP_GYRO_DEG) & (np.abs(ab - G0) < WINDUP_ACC_DEV)
    runs = util.runs(cond, tg, WINDUP_MIN_S - DUR_TOL, max_gap=0.25)
    out["windup"] = bool(runs)
    if runs:
        out["windup_t"] = _f(runs[0][0], 3)
    return out


# =================================================================== 1. touchdown_not_recognised

def _idle_not_landed(ctx: Ctx, iv: ArmedInterval, t_stop: float, related=None) -> dict | None:
    """沒有衝擊簇時的備援:未落地、離地 ≤1 m、垂直速度小、油門持續怠速的區段(攔截很輕的觸地)。
    回傳第一段『和事件相關』的(related(t, agl) 回 None 才算;沒給就全部算):{t_start, t_end, rec, agl}。"""
    tb = ctx.table("CTUN")
    if not _cols(tb, "ThO", "Alt", "CRt"):
        return None
    g = _ground_ref(ctx, iv)
    if g is None:
        return None
    t = tb["t"]
    with np.errstate(invalid="ignore"):
        cond = (t >= iv.t0) & (t <= min(iv.t1, t_stop)) & ~_landed_series(ctx, iv, t) \
            & (tb["ThO"] <= THO_IDLE) & ((tb["Alt"] - g) <= IDLE_AGL_MAX) & (np.abs(tb["CRt"]) <= IDLE_CRT_MAX)
    for s, e, _, _ in util.runs(cond, t, THO_SUSTAIN_MIN_S, max_gap=0.25):
        agl = _agl(ctx, iv, s)
        if related is not None and related(s, agl) is not None:
            continue
        return {"t_start": s, "t_end": e, "rec": _recognition(ctx, iv, s), "agl": agl}
    return None


def _airborne_collapse(ctx: Ctx, iv: ArmedInterval, t_stop: float) -> dict | None:
    """空中油門崩潰(spec:agl>1.5 m、未落地、ThO≤0.03 在 8 s 內累計 ≥3 s):不是觸地未辨識,交給導航 / 失效保護分析。
    回傳 {t_start, idle_s, agl} 或 None。"""
    tb = ctx.table("CTUN")
    g = _ground_ref(ctx, iv)
    if not _cols(tb, "ThO", "Alt") or g is None:
        return None
    t = tb["t"]
    m = (t >= iv.t0) & (t <= min(iv.t1, t_stop))
    t = t[m]
    if t.size < 2:
        return None
    cond = ~_landed_series(ctx, iv, t) & (tb["ThO"][m] <= THO_IDLE) & ((tb["Alt"][m] - g) > NEAR_GROUND_AGL)
    if not cond.any():
        return None
    dt = np.minimum(np.diff(t, append=t[-1] + np.median(np.diff(t))), 0.25)     # 每個樣本代表的時間(空洞不灌水)
    cs = np.concatenate([[0.0], np.cumsum(np.where(cond, dt, 0.0))])
    j = np.searchsorted(t, t + AIR_COLLAPSE_WIN_S, side="right")
    acc = cs[j] - cs[np.arange(t.size)]
    ok = np.flatnonzero(cond & (acc >= AIR_COLLAPSE_MIN_S - DUR_TOL))
    if not ok.size:
        return None
    i = int(ok[0])
    return {"t_start": float(t[i]), "idle_s": float(acc[i]), "agl": float(tb["Alt"][m][i] - g)}


def _touch_anchor(a: Anchors, t: float | None, ref_t: float | None, status: str) -> None:
    """設定 a.t_touch(只給和這次事件相關的觸地)。
    呼叫端只傳『相關』的觸地(有參考點時:在 [ref−10 s, ref+1 s] 內且之間沒有重新起飛)。
    沒有任何事件參考點時,只有 HIT 才設定:SOFT 的彈跳 / 輕觸地不能當 t_first_anomaly。"""
    if t is not None and (ref_t is not None or status == HIT):
        a.t_touch = float(t)


def _unrel_txt(why: dict, ref_t: float | None, ref_name: str) -> str:
    """『這次觸地為什麼和事件無關』的一句話。"""
    if why["why"] == "far_from_event":
        lead = why["lead_s"]
        if lead >= 0:
            return f"它比事件參考點({ref_name} {ref_t:.2f} s)早 {lead:.1f} s(超過 {TOUCH_REF_LOOKBACK:.0f} s)"
        return f"它比事件參考點({ref_name} {ref_t:.2f} s)晚 {-lead:.1f} s"
    how = "EV28 飛控宣告離地" if why["how"] == "EV28" else f"離地升到 {_fmt(why['refly_agl_m'], '.1f')} m"
    return f"之後 {why['t_refly']:.2f} s 已重新起飛({how})"


@guarded(ID_TOUCHDOWN, DOMAIN)
def touchdown_not_recognised(ctx: Ctx, a: Anchors):
    """觸地(衝擊 20 m/s² 以上、dV<3、姿態正常)後 2.0 s 內沒有 EV17 / EV18 → HIT(t_onset = 衝擊簇起點)。

    * 已落地(L==True)、姿態已失控(ae≥10°)、dV≥3(硬撞擊)的衝擊不算觸地。
    * 只評估和事件相關的觸地:有參考點(t_warn / t_loss / t_first_hit / t_detect)時要在 [ref−10 s, ref+1 s] 內;
      觸地之後又重新起飛(EV28 或 agl>1.5 m 持續 0.5 s)的不算。只剩無關的未辨識觸地 → NA('touchdown_unrelated');
      無關的觸地都正常辨識 → CLEAR('recognised')。
    * 離地 >1.5 m 的衝擊不是觸地 → NA('airborne_shock'),交給撞擊 / 失效保護分析。
    * 觸地後油門沒有收到怠速(彈跳 / 繼續飛)→ SOFT;沒有姿態資料 → SOFT。
    * 上鎖當下寫出的 EV17/18 不算;觀察窗被上鎖截短(<2.0 s)→ NA。超過 2 s 才出現的 EV17/18 仍是 HIT(values 記延遲)。
    * 整段解鎖都沒有 EV28(飛控從未宣告離地)→ NA('no_takeoff'):沒有降落可評估,不能 CLEAR。
    * 沒有任何(相關)衝擊簇時走備援路徑(未落地 + 離地 ≤1 m + 油門持續怠速):未辨識 → SOFT;
      離地 >1.5 m 時油門 8 s 內累計 ≥3 s 怠速 → NA('airborne_throttle_collapse',spec:不是本偵測器)。
    設定錨點 a.t_touch(規則見 _touch_anchor)。"""
    iv = a.interval
    if iv is None:
        return _not_armed(ID_TOUCHDOWN)
    shocks = _shocks(ctx, iv)
    ev_t, _ = ctx.ev_rows()
    if shocks is None:
        return _na(ID_TOUCHDOWN, "no_imu", "沒有 IMU 加速度資料,無法判斷觸地衝擊。")
    if ev_t.size == 0:
        return _na(ID_TOUCHDOWN, "no_ev", "log 沒有 EV 事件表,無法判斷飛控是否宣告落地。")
    if not any(not landed for _, landed in ctx.landed_steps(iv)):
        # 整段都是『已落地』(沒有 EV28 NOT_LANDED):飛控從未認為起飛,沒有降落可評估(也不能說『落地辨識正常』)
        return _na(ID_TOUCHDOWN, "no_takeoff",
                   "這次解鎖期間飛控從未宣告離地(沒有 EV28 NOT_LANDED 起飛事件),沒有降落過程可評估落地辨識。",
                   n_shocks=len(shocks))
    t_stop = _t_stop(ctx, a, iv, shocks)
    ref_t, ref_name = _event_ref(a)
    tds = _touchdowns(ctx, iv, shocks)

    def related(t: float, agl: float | None) -> dict | None:
        return _relation(ctx, iv, t, agl, ref_t, t_stop)

    rel: list[Touchdown] = []
    unrel: list[tuple[Touchdown, dict]] = []
    for d in tds:
        why = related(d.shock.t_start, d.agl)
        if why is None:
            rel.append(d)
        else:
            unrel.append((d, why))

    unrec_near: list[tuple[Touchdown, dict]] = []
    unrec_trunc: list[tuple[Touchdown, dict]] = []
    unrec_air: list[tuple[Touchdown, dict]] = []
    rec_ok: list[tuple[Touchdown, dict]] = []
    for d in rel:
        rec = _recognition(ctx, iv, d.shock.t_start)
        if rec["recognised"]:
            rec_ok.append((d, rec))
        elif d.agl is not None and d.agl > NEAR_GROUND_AGL:
            unrec_air.append((d, rec))
        elif iv.t1 - d.shock.t_start < RECOGNISE_WIN:
            unrec_trunc.append((d, rec))
        else:
            unrec_near.append((d, rec))

    thr = {"shock_min": SHOCK_MIN, "dv_max": GROUND_DV_MAX, "recognise_win_s": RECOGNISE_WIN,
           "ae_gate_deg": AE_NORMAL_MAX, "near_ground_m": NEAR_GROUND_AGL, "touch_ref_lookback_s": TOUCH_REF_LOOKBACK,
           "touch_ref_lead_s": TOUCH_REF_LEAD, "refly_agl_m": REFLY_AGL, "refly_min_s": REFLY_MIN_S}
    ref_vals = {"event_ref": ref_name or None, "t_event_ref": _f(ref_t, 3), "n_unrelated_touchdowns": len(unrel)}
    end_txt = "直到 log 結束,期間沒有上鎖" if iv.open_end else "直到上鎖"

    # ---- HIT / SOFT
    if unrec_near:
        d, rec = unrec_near[0]
        s = d.shock
        col = _tho_collapse(ctx, iv, s.t_start)
        ld = _ldet_info(ctx, iv, s.t_start)
        wu = _windup(ctx, iv, s.t_start, t_stop)
        late = [x for x in (rec["t17"], rec["t18"]) if x is not None]       # 超過 2 s 才出現的落地事件(上鎖寫的已排除)
        t_first_ev = min(late) if late else None
        no_ev_s = (t_first_ev if t_first_ev is not None else iv.t1) - s.t_start
        land_class = "GROUND_TIPOVER_like" if (d.agl is None or d.agl <= 1.0) and (d.vz_pre is None or d.vz_pre >= -100.0) \
            else "HARD_LANDING_like"
        collapse = col["t_sustained"] is not None
        reasons = []
        if d.ae is None:
            reasons.append("ae_unavailable")
        if not collapse:
            reasons.append("no_throttle_collapse")
        status = HIT if not reasons else SOFT
        _touch_anchor(a, s.t_start, ref_t, status)
        vals = {"t_shock": _f(s.t_start, 3), "t_peak": _f(s.t_peak, 3), "peak_ms2": _f(s.peak, 1), "dv_ms": _f(s.dv, 2),
                "duration_s": _f(s.duration, 3), "ae_deg": _f(d.ae, 2), "agl_m": _f(d.agl, 2), "vz_pre_cms": _f(d.vz_pre, 1),
                "near_ground": True, "landing_class": land_class, "recognised": False,
                "dt_ev17_s": _f(rec["dt17"], 3), "dt_ev18_s": _f(rec["dt18"], 3), "no_ev_s": _f(no_ev_s, 2),
                "late_recognition": t_first_ev is not None,
                "ev_ignored_at_disarm": rec["ignored_at_disarm"], "open_end": bool(iv.open_end),
                "t_tho_drop": _f(col["t_drop"], 3), "t_tho_zero": _f(col["t_zero"], 3),
                "t_tho_sustained": _f(col["t_sustained"], 3),
                "ldet_count_max": ld.get("count_max"), "ldet_bit8_t": _f(ld.get("bit8_t"), 3),
                "n_unrecognised": len(unrec_near), "downgrade": reasons, "confidence": "medium",
                "thresholds": thr, **ref_vals}
        vals.update({k: v for k, v in wu.items()})
        windup_txt = ""
        if wu.get("windup"):
            windup_txt = (f"觸地後位置控制器在地面纏繞(加速度需求最大 {_fmt(wu.get('pos_acc_cmd_max'), '.1f')} m/s²、"
                          f"傾斜指令最大 {_fmt(wu.get('lean_max_deg'), '.1f')}°)。")
        tho_parts = []
        if col["t_drop"] is not None:
            tho_parts.append(f"油門 {col['t_drop']:.2f} s 起收小")
        if col["t_zero"] is not None:
            tho_parts.append(f"{col['t_zero']:.2f} s 首次到怠速")
        if col["t_sustained"] is not None:
            tho_parts.append(f"{col['t_sustained']:.2f} s 起持續怠速")
        tho_txt = ("、".join(tho_parts) + ",") if tho_parts else ""
        if t_first_ev is None:
            ev_txt = (f"之後 {no_ev_s:.1f} s({end_txt})都沒有 LAND_COMPLETE_MAYBE / LAND_COMPLETE(正常降落 2 s 內會出現):"
                      f"{tho_txt}飛控卻一直認為還在空中。")
        else:
            ev_txt = (f"直到 {no_ev_s:.1f} s 後才出現 LAND_COMPLETE_MAYBE / LAND_COMPLETE(超過 {RECOGNISE_WIN:.1f} s 門檻,"
                      f"正常降落 2 s 內會出現):{tho_txt}這段時間飛控仍認為在空中。")
        txt = (f"{s.t_start:.2f} s 觸地(衝擊 {s.peak:.1f} m/s²、高度相對解鎖地面 {_fmt(d.agl, '+.2f')} m、{_vz_txt(d.vz_pre)}),"
               f"{ev_txt}"
               f"{windup_txt}"
               f"建議:觸地後立刻把油門拉到底並上鎖,或改用 LAND 模式降落;檢查降落偵測為何沒成立(傾角需求超過 15° 會擋住)。")
        if reasons:
            txt += "(證據降級為 SOFT:" + ("油門未持續收到怠速,可能只是彈跳 / 繼續飛" if "no_throttle_collapse" in reasons
                                        else "缺姿態資料無法確認觸地時姿態正常") + ")"
        return _fd(ID_TOUCHDOWN, status, "high", t_onset=float(s.t_start), t_alarm=float(s.t_start + RECOGNISE_WIN),
                   context="near_ground", values=vals, text_zh=txt)

    if unrec_trunc:
        d, rec = unrec_trunc[0]
        return _na(ID_TOUCHDOWN, "window_truncated",
                   f"{d.shock.t_start:.2f} s 觸地後不到 {RECOGNISE_WIN:.0f} s 就"
                   f"{'結束 log' if iv.open_end else '上鎖'},觀察窗不足,無法判斷是否辨識落地。",
                   t_shock=_f(d.shock.t_start, 3), observed_s=_f(iv.t1 - d.shock.t_start, 2), thresholds=thr)

    if rec_ok:
        d, rec = rec_ok[0]             # 第一個相關觸地(之後的簇是同一次降落的彈跳)
        s = d.shock
        _touch_anchor(a, s.t_start, ref_t, CLEAR)
        first = min(x for x in (rec["dt17"], rec["dt18"]) if x is not None)
        vals = {"t_shock": _f(s.t_start, 3), "peak_ms2": _f(s.peak, 1), "dv_ms": _f(s.dv, 2), "ae_deg": _f(d.ae, 2),
                "agl_m": _f(d.agl, 2), "vz_pre_cms": _f(d.vz_pre, 1), "recognised": True, "related_to_event": True,
                "dt_ev17_s": _f(rec["dt17"], 3), "dt_ev18_s": _f(rec["dt18"], 3), "n_touchdowns": len(rec_ok),
                "confidence": "medium", "thresholds": thr, **ref_vals}
        when = f"後 {first:.2f} s 即出現" if first >= 0 else f"前 {-first:.2f} s 已出現"
        txt = (f"{s.t_start:.2f} s 觸地(衝擊 {s.peak:.1f} m/s²){when} LAND_COMPLETE_MAYBE / LAND_COMPLETE,"
               f"落地辨識正常(門檻 {RECOGNISE_WIN:.1f} s)。")
        return _fd(ID_TOUCHDOWN, CLEAR, "info", t_onset=float(s.t_start), context="recognised", values=vals, text_zh=txt)

    if unrec_air:
        d, _ = unrec_air[0]
        return _na(ID_TOUCHDOWN, "airborne_shock",
                   f"{d.shock.t_start:.2f} s 的衝擊({d.shock.peak:.1f} m/s²)發生在離地約 {d.agl:.1f} m,不是觸地,"
                   f"本偵測器不適用(請看撞擊 / 失效保護分析)。",
                   t_shock=_f(d.shock.t_start, 3), agl_m=_f(d.agl, 2), thresholds=thr)

    # ---- 沒有(相關的)觸地型衝擊簇:備援路徑
    fb = _idle_not_landed(ctx, iv, t_stop, related)
    if fb is None:
        ctun = ctx.table("CTUN")
        if not _cols(ctun, "ThO", "Alt", "CRt") or _ground_ref(ctx, iv) is None:
            return _na(ID_TOUCHDOWN, "no_ctun", "缺 CTUN 油門 / 高度資料,又沒有觸地衝擊,無法判斷落地辨識。", thresholds=thr)
        air = _airborne_collapse(ctx, iv, t_stop)
        if air is not None:
            # spec:空中油門崩潰不是本偵測器(不能用來排除也不能當觸地證據)→ NA,交給導航 / 失效保護分析
            return _na(ID_TOUCHDOWN, "airborne_throttle_collapse",
                       f"{air['t_start']:.2f} s 起飛控仍認為在空中、離地約 {air['agl']:.1f} m,油門卻在 {AIR_COLLAPSE_WIN_S:.0f} s 內"
                       f"累計 {air['idle_s']:.1f} s 收到怠速:這是空中的油門崩潰,不是觸地未辨識,本偵測器不適用(請看導航 / 失效保護分析)。",
                       t_start=_f(air["t_start"], 3), idle_s=_f(air["idle_s"], 2), agl_m=_f(air["agl"], 2),
                       n_shocks=len(shocks), thresholds=thr)
        if unrel:
            # 只有和事件無關的觸地(之後又起飛 / 離事件太遠):不設 a.t_touch
            unrec_u = [(d, w) for d, w in unrel if not _recognition(ctx, iv, d.shock.t_start)["recognised"]]
            d, w = (unrec_u or unrel)[-1]                   # 最接近事件的那一次
            ev_txt = f"{ref_name} {ref_t:.2f} s 的事件" if ref_t is not None else "之後的飛行"
            uvals = {"t_shock": _f(d.shock.t_start, 3), "peak_ms2": _f(d.shock.peak, 1), "agl_m": _f(d.agl, 2),
                     "related_to_event": False, "n_unrecognised_unrelated": len(unrec_u), "thresholds": thr,
                     **{k: v for k, v in w.items()}, **ref_vals}
            if unrec_u:
                return _na(ID_TOUCHDOWN, "touchdown_unrelated",
                           f"{d.shock.t_start:.2f} s 有一次觸地沒有被判定落地,但{_unrel_txt(w, ref_t, ref_name)},和{ev_txt}無關;"
                           f"{'事件前' if ref_t is not None else '這次解鎖最後'}沒有和它相關的觸地,本偵測器不適用"
                           f"(不能用來排除觸地翻覆,也不是它的證據)。", **uvals)
            return _fd(ID_TOUCHDOWN, CLEAR, "info", t_onset=float(d.shock.t_start), context="recognised",
                       values={**uvals, "recognised": True, "n_touchdowns": len(unrel), "confidence": "medium"},
                       text_zh=(f"這次解鎖的 {len(unrel)} 次觸地都在 {RECOGNISE_WIN:.1f} s 內出現 LAND_COMPLETE_MAYBE / LAND_COMPLETE,"
                                f"落地辨識正常;最後一次 {d.shock.t_start:.2f} s 觸地{_unrel_txt(w, ref_t, ref_name)},和{ev_txt}無關。"))
        n_skip = len(shocks) - len(tds)
        return _fd(ID_TOUCHDOWN, CLEAR, "info", context="no_ground_contact",
                   values={"n_shocks": len(shocks), "n_shocks_excluded": n_skip, "confidence": "medium", "thresholds": thr,
                           **ref_vals},
                   text_zh="這次解鎖期間沒有觸地型衝擊(>20 m/s²、dV<3、姿態正常),也沒有『油門怠速卻未落地』的區段:未見落地辨識失敗。")
    rec = fb["rec"]
    if rec["recognised"]:
        _touch_anchor(a, fb["t_start"], ref_t, CLEAR)
        return _fd(ID_TOUCHDOWN, CLEAR, "info", t_onset=float(fb["t_start"]), context="recognised_no_shock",
                   values={"t_idle": _f(fb["t_start"], 3), "dt_ev17_s": _f(rec["dt17"], 3), "dt_ev18_s": _f(rec["dt18"], 3),
                           "confidence": "medium", "thresholds": thr, **ref_vals},
                   text_zh=f"{fb['t_start']:.2f} s 油門收到怠速(沒有明顯衝擊),之後很快出現落地事件,落地辨識正常。")
    if iv.t1 - fb["t_start"] < RECOGNISE_WIN:
        return _na(ID_TOUCHDOWN, "window_truncated",
                   f"{fb['t_start']:.2f} s 油門收到怠速後不到 {RECOGNISE_WIN:.0f} s 就"
                   f"{'結束 log' if iv.open_end else '上鎖'},觀察窗不足。", thresholds=thr)
    _touch_anchor(a, fb["t_start"], ref_t, SOFT)
    col = _tho_collapse(ctx, iv, fb["t_start"])
    ld = _ldet_info(ctx, iv, fb["t_start"])
    vals = {"t_idle": _f(fb["t_start"], 3), "agl_m": _f(fb["agl"], 2), "recognised": False, "no_ev_s": _f(iv.t1 - fb["t_start"], 2),
            "ev_ignored_at_disarm": rec["ignored_at_disarm"], "open_end": bool(iv.open_end),
            "t_tho_sustained": _f(col["t_sustained"], 3),
            "ldet_count_max": ld.get("count_max"), "ldet_bit8_t": _f(ld.get("bit8_t"), 3), "n_shocks": len(shocks),
            "confidence": "low", "thresholds": thr, **ref_vals}
    return _fd(ID_TOUCHDOWN, SOFT, "high", t_onset=float(fb["t_start"]), t_alarm=float(fb["t_start"] + RECOGNISE_WIN),
               context="idle_no_shock", values=vals,
               text_zh=(f"{fb['t_start']:.2f} s 起油門收到怠速、離地約 {_fmt(fb['agl'], '+.2f')} m、幾乎沒有垂直運動,"
                        f"但 {iv.t1 - fb['t_start']:.1f} s 內({end_txt})都沒有 LAND_COMPLETE 事件(沒有明顯觸地衝擊,證據較弱)。"
                        f"建議檢查降落偵測條件,觸地後立刻上鎖或改用 LAND 模式。"))


# =================================================================== 2. idle_motor_asymmetry

def _motor_channels(ctx: Ctx, rcou: dict) -> tuple[list[int], list[int]]:
    """馬達輸出通道 [n...] 與對應馬達編號:SERVOn_FUNCTION 33..40;沒有就用 FRAME_CLASS 推馬達數。"""
    pairs = []
    for n in range(1, 17):
        f = ctx.param(f"SERVO{n}_FUNCTION")
        if f is not None and 33 <= int(f) <= 40 and f"C{n}" in rcou:
            pairs.append((int(f) - 32, n))
    if not pairs:
        k = FRAME_MOTORS.get(int(ctx.param("FRAME_CLASS", 0) or 0))
        if k:
            pairs = [(i, i) for i in range(1, k + 1) if f"C{i}" in rcou]
    pairs.sort()
    return [n for _, n in pairs], [m for m, _ in pairs]


def _pwm_range(ctx: Ctx) -> tuple[float, float]:
    lo = float(ctx.param("MOT_PWM_MIN", 0) or 0)
    hi = float(ctx.param("MOT_PWM_MAX", 0) or 0)
    lo = lo if lo > 0 else 1100.0
    hi = hi if hi > lo else 2000.0
    return lo, hi


def _frame_groups(ctx: Ctx, motor_nos: list[int]):
    """hexa-X 的左 / 右 / 前 / 後欄位索引;其他機型回 None(只用全距)。"""
    fc, ft = int(ctx.param("FRAME_CLASS", 0) or 0), int(ctx.param("FRAME_TYPE", 0) or 0)
    if (fc, ft) != HEXA_X or len(motor_nos) != 6:
        return None
    pos = {m: i for i, m in enumerate(motor_nos)}
    pick = lambda s: [pos[m] for m in sorted(s) if m in pos]     # noqa: E731
    g = {"left": pick(HEXA_LEFT), "right": pick(HEXA_RIGHT), "front": pick(HEXA_FRONT), "rear": pick(HEXA_REAR)}
    return g if len(g["left"]) == len(g["right"]) == 3 else None


@guarded(ID_IDLE, DOMAIN)
def idle_motor_asymmetry(ctx: Ctx, a: Anchors):
    """油門已在怠速(ThO≤0.03、馬達已轉起來)且『貼地』(已落地或 agl≤1.5 m)時左右馬達明顯不對稱 → HIT(≥0.5 s)/ SOFT(0.2–0.5 s)。

    flag = 全距 ≥120 µs(0.133 全距)或 |左右力矩| ≥250 µs 或 油門補償 ≥0.04;hexa-X 才有力矩,其他機型只用全距。
    空中(agl>1.5 m 且飛控認為未落地)的怠速不算:空中收油門不是地面翻覆機制(spec 把它歸給 airborne_throttle_collapse),
    而且未落地時 MOTB.ThrOut 本來就被夾在約 0.5×懸停值,油門補償條件在空中必然成立。只有空中怠速 → NA('air_idle_only')。
    撞擊 / 失控之後(t_stop)的樣本不算(倒扣怠速不是地面階段證據)。"""
    iv = a.interval
    if iv is None:
        return _not_armed(ID_IDLE)
    rc, ct = ctx.table("RCOU"), ctx.table("CTUN")
    if rc is None or not _cols(ct, "ThO"):
        return _na(ID_IDLE, "no_data", "缺 RCOU 或 CTUN.ThO,無法判斷怠速時的馬達對稱性。")
    chs, mno = _motor_channels(ctx, rc)
    if len(chs) < 4:
        return _na(ID_IDLE, "no_motor_channels", "找不到馬達輸出通道(SERVOn_FUNCTION 33..40),無法判斷馬達對稱性。")
    pmin, pmax = _pwm_range(ctx)
    span = pmax - pmin
    shocks = _shocks(ctx, iv)
    t_stop = _t_stop(ctx, a, iv, shocks)
    t = rc["t"]
    m_iv = (t >= iv.t0) & (t <= min(iv.t1, t_stop))
    pwm = np.stack([rc[f"C{n}"] for n in chs], axis=1).astype(np.float64)   # uint16 相減 / 加總會回繞(Ctx 已轉,這裡再保險)
    tho = _hold(ct["t"], ct["ThO"], t)
    mb = ctx.table("MOTB")
    boost = _hold(mb["t"], mb["ThrOut"], t) - tho if _cols(mb, "ThrOut") else np.full(t.size, np.nan)
    with np.errstate(invalid="ignore"):
        idle_any = m_iv & (tho <= THO_IDLE) & (pwm.min(axis=1) >= pmin + SPOOL_FRAC * span)
    near, gated = _near_ground(ctx, iv, t)
    idle = idle_any & near                  # 貼地怠速:本偵測器評估的樣本
    idle_air = idle_any & ~near             # 空中怠速:不評估,只記錄
    n_idle, n_air = int(idle.sum()), int(idle_air.sum())
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.1
    thr = {"spread_us": _f(IDLE_SPREAD_FRAC * span, 1), "roll_mom_us": IDLE_ROLLMOM_US, "boost": IDLE_BOOST,
           "min_s": IDLE_MIN_S, "tho_idle": THO_IDLE, "near_ground_m": NEAR_GROUND_AGL}

    spread = pwm.max(axis=1) - pwm.min(axis=1)
    floor_pwm = pmin + float(ctx.param("MOT_SPIN_MIN", 0.15) or 0.15) * span
    grp = _frame_groups(ctx, mno)
    if grp is not None:
        roll_mom = pwm[:, grp["left"]].sum(axis=1) - pwm[:, grp["right"]].sum(axis=1)
        pitch_mom = pwm[:, grp["front"]].sum(axis=1) - pwm[:, grp["rear"]].sum(axis=1)
    else:
        roll_mom = np.full(t.size, np.nan)
        pitch_mom = np.full(t.size, np.nan)
    with np.errstate(invalid="ignore"):
        f_sp = spread >= IDLE_SPREAD_FRAC * span
        f_rm = np.abs(roll_mom) >= IDLE_ROLLMOM_US
        f_bo = boost >= IDLE_BOOST
    f_any = f_sp | f_rm | f_bo
    flag = idle & f_any

    air_vals: dict = {"air_idle_excluded_s": _f(n_air * dt, 1), "near_ground_gate": bool(gated)}
    air_agl = None
    if n_air:
        air_runs = util.runs(idle_air & f_any, t, 0.0, max_gap=0.25)
        air_vals["air_idle_flag_longest_s"] = _f(max((r[1] - r[0] for r in air_runs), default=0.0), 2)
        g = _ground_ref(ctx, iv)
        if _cols(ct, "Alt") and g is not None:
            air_agl = _nanmax(_hold(ct["t"], ct["Alt"], t[idle_air]) - g)
        air_vals["air_idle_agl_max_m"] = _f(air_agl, 2)
    if n_idle * dt < IDLE_MIN_S - DUR_TOL:
        if n_air * dt >= IDLE_MIN_S - DUR_TOL:
            note = ""
            if (air_vals.get("air_idle_flag_longest_s") or 0.0) >= IDLE_MIN_S - DUR_TOL:
                note = f"(空中怠速期間 PWM 不對稱 / 油門補償超標最長 {air_vals['air_idle_flag_longest_s']:.1f} s,僅供參考)"
            return _na(ID_IDLE, "air_idle_only",
                       f"油門怠速(ThO≤{THO_IDLE})的 {n_air * dt:.1f} s 都在空中(離地最高約 {_fmt(air_agl, '.1f')} m、飛控認為未落地),"
                       f"沒有貼地的怠速段:空中收油門不是地面翻覆機制,本偵測器不適用(請看空中油門崩潰 / 馬達分析)。{note}",
                       idle_samples=n_idle, thresholds=thr, **air_vals)
        return _na(ID_IDLE, "no_idle_window", "這次解鎖沒有足夠長的貼地怠速段(油門≤0.03 且馬達已轉起來),無法判斷怠速時的馬達對稱性。",
                   idle_samples=n_idle, thresholds=thr, **air_vals)
    air_txt = f"(另有 {n_air * dt:.1f} s 空中怠速不納入)" if n_air * dt >= dt / 2 else ""

    runs = util.runs(flag, t, 0.0, max_gap=0.25)
    hard = [r for r in runs if r[1] - r[0] >= IDLE_MIN_S - DUR_TOL]
    soft = [r for r in runs if r[1] - r[0] >= IDLE_SOFT_MIN_S - DUR_TOL]

    def stats(i0: int, i1: int) -> dict:
        sl = slice(i0, i1 + 1)
        ii = np.flatnonzero(idle[sl]) + i0
        med = np.median(pwm[ii], axis=0)
        d = {"spread_med_us": _f(np.median(spread[ii]), 0), "spread_max_us": _f(np.max(spread[ii]), 0),
             "roll_mom_med": _f(_nanmed(roll_mom[ii]), 0) if grp is not None else None,
             "roll_mom_peak": _f(_nanmax(np.abs(roll_mom[ii])), 0) if grp is not None else None,
             "pitch_mom_med": _f(_nanmed(pitch_mom[ii]), 0) if grp is not None else None,
             "boost_med": _f(_nanmed(boost[ii]), 3), "boost_max": _f(_nanmax(boost[ii]), 3),
             "n_floor": int(np.sum(med <= floor_pwm + 5)), "n_ceiling": int(np.sum(med >= pmax - 50)),
             "pwm_med": {f"C{n}": int(v) for n, v in zip(chs, med)}}
        if grp is not None:
            d["left_pwm_range"] = [int(med[grp["left"]].min()), int(med[grp["left"]].max())]
            d["right_pwm_range"] = [int(med[grp["right"]].min()), int(med[grp["right"]].max())]
        return d

    if soft:
        s, e, i0, i1 = (hard or soft)[0]
        st = stats(i0, i1)
        status = HIT if hard else SOFT
        run_s = e - s
        # 這段實際是哪個條件觸發的(佔超標樣本 ≥50% 的條件;都不到就取最多的那個)
        fl = np.flatnonzero(flag[i0:i1 + 1]) + i0
        frac = {"spread": float(np.mean(f_sp[fl])), "roll_mom": float(np.mean(f_rm[fl])), "boost": float(np.mean(f_bo[fl]))}
        crit = [k for k, v in frac.items() if v >= 0.5] or [max(frac, key=frac.get)]
        parts = []
        if "spread" in crit:
            parts.append(f"馬達 PWM 全距 {_fmt(st['spread_med_us'], '.0f')} µs(門檻 {IDLE_SPREAD_FRAC * span:.0f} µs;"
                         f"124 健康怠速 ≤57 µs)")
        if "roll_mom" in crit:
            parts.append(f"左右三顆 PWM 總和差 {_fmt(st['roll_mom_med'], '+.0f')} µs(門檻 ±{IDLE_ROLLMOM_US:.0f} µs)")
        if "boost" in crit:
            parts.append(f"MOTB 油門輸出比 ThO 高 {_fmt(st['boost_med'], '.3f')}(門檻 {IDLE_BOOST})")
        if "spread" in crit or "roll_mom" in crit:
            side = ""
            if grp is not None:
                side = (f"左側 {st['left_pwm_range'][0]}–{st['left_pwm_range'][1]} µs、右側 "
                        f"{st['right_pwm_range'][0]}–{st['right_pwm_range'][1]} µs:")
            mech = f"{side}控制器命令的 PWM 型態是在地面上硬要把機身側傾,單側推力可能把機身撬起翻倒(翻覆的機制 / 症狀)。"
        else:
            landed_frac = float(np.mean(_landed_series(ctx, iv, t[fl])))
            why = ("飛控仍認為在空中(未判定落地),混控會把油門輸出夾在約 0.5×懸停值,馬達在地面持續出力。" if landed_frac < 0.5
                   else "飛控已判定落地,輸出卻仍被抬高(原因不明,請對照混控 / 怠速設定)。")
            mech = f"PWM 左右大致對稱(全距 {_fmt(st['spread_med_us'], '.0f')} µs),但輸出沒有降到地面怠速:{why}"
            st["landed_frac"] = _f(landed_frac, 2)
        esc = ctx.table("ESC") is not None
        caveat = ("這裡看的是控制器命令的 PWM" + ("" if esc else "(log 沒有 ESC 轉速遙測,無法直接確認馬達本身)")
                  + ",型態與控制器需求一致,不像單顆馬達故障。")
        vals = {"t_run_start": _f(s, 3), "t_run_end": _f(e, 3), "run_s": _f(run_s, 2), "n_runs": len(runs),
                "idle_total_s": _f(n_idle * dt, 1), "criteria": crit, "criteria_frac": {k: _f(v, 2) for k, v in frac.items()},
                "esc_telemetry": esc, "confidence": "high", "thresholds": thr, **air_vals}
        vals.update(st)
        txt = (f"{s:.2f} s 起油門已在怠速(ThO≤{THO_IDLE})且貼地(離地 ≤{NEAR_GROUND_AGL} m 或已落地),持續 {run_s:.1f} s 超標:"
               + "、".join(parts) + "。" + mech + caveat + air_txt)
        if not hard:
            txt += f"(持續僅 {run_s:.1f} s,不足 {IDLE_MIN_S:.1f} s,列為 SOFT)"
        return _fd(ID_IDLE, status, "high", t_onset=float(s), t_alarm=float(s + IDLE_MIN_S) if hard else float(e),
                   context="ground_idle", values=vals, text_zh=txt)

    ii = np.flatnonzero(idle)
    sp95 = float(np.percentile(spread[ii], 95))
    longest = max((r[1] - r[0] for r in runs), default=None)          # 超標樣本最長連續跨度(都 <0.2 s)
    vals = {"idle_total_s": _f(n_idle * dt, 1), "spread_p95_us": _f(sp95, 0), "spread_max_us": _f(spread[ii].max(), 0),
            "roll_mom_max": _f(_nanmax(np.abs(roll_mom[ii])), 0) if grp is not None else None,
            "boost_max": _f(_nanmax(boost[ii]), 3),
            "n_flagged_samples": int(flag.sum()), "flag_longest_s": _f(longest, 2), "confidence": "high", "thresholds": thr,
            **air_vals}
    if longest is None:
        flag_txt = "沒有任何樣本超標"
    elif longest < dt / 2:
        flag_txt = f"超標的只有 {int(flag.sum())} 個孤立樣本"
    else:
        flag_txt = f"超標最長只連續 {longest:.1f} s"
    return _fd(ID_IDLE, CLEAR, "info", t_onset=float(t[ii[0]]), context="ground_idle", values=vals,
               text_zh=(f"貼地怠速段共 {n_idle * dt:.1f} s,馬達 PWM 全距 95 百分位 {sp95:.0f} µs、最大 {spread[ii].max():.0f} µs"
                        f"({flag_txt};門檻 {IDLE_SPREAD_FRAC * span:.0f} µs 持續 ≥{IDLE_MIN_S:.1f} s),"
                        f"左右馬達對稱,沒有怠速翻覆力矩。{air_txt}"))


# =================================================================== 3. gravity_only_tipover

def _hover_pwm(ctx: Ctx, iv: ArmedInterval, t_end: float) -> float | None:
    """穩定懸停時各馬達 PWM 中位數的最大值(翻倒視窗前 5 s 以前、已離地、ThO 在 ThH ±25%、爬升率小);樣本不足回 None。"""
    ct, rc = ctx.table("CTUN"), ctx.table("RCOU")
    if rc is None or not _cols(ct, "ThO", "ThH", "CRt"):
        return None
    chs, _ = _motor_channels(ctx, rc)
    if not chs:
        return None
    t = ct["t"]
    with np.errstate(invalid="ignore"):
        ok = (t >= iv.t0 + 10.0) & (t <= t_end - HOVER_GUARD_S) & ~_landed_series(ctx, iv, t) \
            & (np.abs(ct["CRt"]) <= HOVER_CRT_MAX) & (np.abs(ct["ThO"] - ct["ThH"]) <= HOVER_THO_BAND * ct["ThH"])
    if ok.sum() < HOVER_MIN_SAMPLES:
        return None
    idx = np.clip(np.searchsorted(rc["t"], t[ok], side="right") - 1, 0, None)
    meds = [_nanmed(rc[f"C{n}"][idx]) for n in chs]
    meds = [m for m in meds if m is not None]
    return float(max(meds)) if meds else None


@guarded(ID_TIPOVER, DOMAIN)
def gravity_only_tipover(ctx: Ctx, a: Anchors):
    """貼地的翻倒過程(傾斜 5° → 50°)只剩重力在作用 → HIT。

    HIT:AccZ 殘差 RMS<0.5 m/s² 且 ThO(p90)≤0.02、視窗內最大 PWM 低於懸停 PWM、角速度單調成長(能評估的全部成立);
    SOFT:RMS 成立但其他條件有一個不成立;CLEAR:貼地翻倒但 RMS≥0.5(有推力 / 外力);
    NA:沒有翻倒到 50°、只在空中傾斜 ≥50°(急轉 / FLIP / 空中失控,'airborne_tilt')、視窗太短或有效 IMU 樣本不足。
    視窗 = 第一次『貼地時』(已落地或 agl≤1.5 m)傾斜到 50° 之前『最後一次傾斜≤5°』到該時刻(只看一次連續的翻出);
    視窗起點也必須貼地,否則是從空中就開始傾斜,不是地面翻倒。NaN 樣本先剔除,不會變成 CLEAR。"""
    iv = a.interval
    if iv is None:
        return _not_armed(ID_TIPOVER)
    att = _att_arrays(ctx, iv.t0, iv.t1)
    if att is None:
        return _na(ID_TIPOVER, "no_attitude", "缺姿態資料(ANG / ATT),無法判斷翻倒過程。")
    tilt = util.tilt_deg(att["roll"], att["pitch"])
    thr = {"start_deg": TIP_START_DEG, "end_deg": TIP_END_DEG, "rms_max": TIP_RMS_MAX, "tho_max": TIP_THO_MAX,
           "near_ground_m": NEAR_GROUND_AGL}
    with np.errstate(invalid="ignore"):
        kb = np.flatnonzero(tilt >= TIP_END_DEG)
    if not kb.size:
        return _na(ID_TIPOVER, "no_tipover", "這次解鎖傾斜角未超過 50°,沒有翻倒過程可分析。",
                   max_tilt_deg=_f(_nanmax(tilt), 1), thresholds=thr)
    near_b, _ = _near_ground(ctx, iv, att["t"][kb])
    k50 = kb[near_b]
    if not k50.size:
        t_air = float(att["t"][kb[0]])
        agl = _agl(ctx, iv, t_air)
        return _na(ID_TIPOVER, "airborne_tilt",
                   f"傾斜 ≥50° 只發生在空中(首次 {t_air:.2f} s、離地約 {_fmt(agl, '.1f')} m),沒有貼地的翻倒過程可分析"
                   f"(空中大角度 / 翻滾不是本偵測器)。",
                   t_first_50_air=_f(t_air, 3), agl_m=_f(agl, 2), max_tilt_deg=_f(_nanmax(tilt), 1), thresholds=thr)
    j50 = int(k50[0])
    with np.errstate(invalid="ignore"):
        low = np.flatnonzero(tilt[:j50] <= TIP_START_DEG)
    j5 = int(low[-1]) + 1 if low.size else 0
    t5, t50 = float(att["t"][min(j5, j50)]), float(att["t"][j50])
    n_air50 = int(np.sum(att["t"][kb] < t50))          # 之前空中的 ≥50° 樣本(不分析,只記錄)
    near5, _ = _near_ground(ctx, iv, [t5])
    if not near5[0]:
        agl5 = _agl(ctx, iv, t5)
        return _na(ID_TIPOVER, "airborne_tilt",
                   f"傾斜從空中就開始({t5:.2f} s 離地約 {_fmt(agl5, '.1f')} m),{t50:.2f} s 貼地時已 ≥50°:不是貼地翻倒,"
                   f"本偵測器不適用(請看空中失控分析)。",
                   t_start=_f(t5, 3), t_end=_f(t50, 3), agl_start_m=_f(agl5, 2), thresholds=thr)
    if t50 - t5 < TIP_MIN_WIN_S:
        return _na(ID_TIPOVER, "window_too_short", f"傾斜 5°→50° 只花 {t50 - t5:.2f} s(<{TIP_MIN_WIN_S} s),太快無法用加速度分辨。",
                   t_start=_f(t5, 3), t_end=_f(t50, 3), thresholds=thr)
    imu = _imu_acc(ctx, t5, t50)
    gy = _imu_gyro_deg(ctx, t5, t50)
    if imu is None:
        return _na(ID_TIPOVER, "no_imu", "缺 IMU 加速度資料,無法判斷翻倒過程。")
    ti, accz = imu[0], imu[1][:, 2]
    roll = np.interp(ti, att["t"], att["roll"])
    pitch = np.interp(ti, att["t"], att["pitch"])
    pred = -G0 * np.cos(np.radians(roll)) * np.cos(np.radians(pitch))
    ok = np.isfinite(accz) & np.isfinite(pred)
    n_bad = int(ti.size - ok.sum())
    ti, accz, pred = ti[ok], accz[ok], pred[ok]
    if ti.size < 3:
        return _na(ID_TIPOVER, "no_imu", "翻倒視窗內沒有足夠的有效 IMU / 姿態樣本(NaN)。",
                   t_start=_f(t5, 3), t_end=_f(t50, 3), n_invalid=n_bad, thresholds=thr)
    b = np.floor((ti - t5) / TIP_BIN_S).astype(int)
    nb = int(b.max()) + 1
    cnt = np.bincount(b, minlength=nb)
    okb = cnt > 0
    res = (np.bincount(b, weights=accz, minlength=nb)[okb] / cnt[okb]) - (np.bincount(b, weights=pred, minlength=nb)[okb] / cnt[okb])
    if res.size < 4:
        return _na(ID_TIPOVER, "window_too_short", "翻倒視窗內有效 IMU 樣本不足。", t_start=_f(t5, 3), t_end=_f(t50, 3),
                   n_invalid=n_bad, thresholds=thr)
    rms = float(np.sqrt(np.mean(res * res)))

    ct = ctx.table("CTUN")
    tho_p90 = None
    if _cols(ct, "ThO"):
        m = (ct["t"] >= t5 - 0.1) & (ct["t"] <= t50) & np.isfinite(ct["ThO"])
        if m.any():
            tho_p90 = float(np.percentile(ct["ThO"][m], 90))
        else:
            v = float(_hold(ct["t"], ct["ThO"], [t5])[0])
            tho_p90 = v if np.isfinite(v) else None
    rc = ctx.table("RCOU")
    pwm_max = hover = None
    if rc is not None:
        chs, _ = _motor_channels(ctx, rc)
        m = (rc["t"] >= t5 - 0.1) & (rc["t"] <= t50)
        if chs and m.any():
            pwm_max = _nanmax(np.concatenate([rc[f"C{n}"][m] for n in chs]))
            if pwm_max is not None:
                hover = _hover_pwm(ctx, iv, t5)
    rho = growth = None
    if gy is not None:
        gt, gv = gy[0][np.isfinite(gy[1])], gy[1][np.isfinite(gy[1])]
        if gt.size:
            gb = np.floor((gt - t5) / TIP_GYRO_BIN_S).astype(int)
            gc = np.bincount(gb)
            gm = np.bincount(gb, weights=gv)[gc > 0] / gc[gc > 0]
            if gm.size >= 5:
                rho = _spearman(gm)
                growth = float(np.mean(gm[-3:]) / max(np.mean(gm[:3]), 1.0))

    checks = {"rms": rms < TIP_RMS_MAX}
    if tho_p90 is not None:
        checks["throttle"] = tho_p90 <= TIP_THO_MAX
    if pwm_max is not None and hover is not None:
        checks["pwm_below_hover"] = pwm_max < hover
    if rho is not None and np.isfinite(rho):
        checks["gyro_monotone"] = bool(rho >= TIP_GYRO_RHO and growth >= TIP_GYRO_GROWTH)
    vals = {"t_start": _f(t5, 3), "t_end": _f(t50, 3), "window_s": _f(t50 - t5, 2), "rms_ms2": _f(rms, 3),
            "max_resid_ms2": _f(np.max(np.abs(res)), 2), "tho_p90": _f(tho_p90, 3), "max_pwm": _f(pwm_max, 0),
            "hover_pwm": _f(hover, 0), "gyro_rho": _f(rho, 3), "gyro_growth": _f(growth, 2), "checks": checks,
            "n_invalid_imu": n_bad, "n_air_tilt50_before": n_air50, "confidence": "high", "thresholds": thr}
    if not checks["rms"]:
        return _fd(ID_TIPOVER, CLEAR, "info", t_onset=float(t5), context="not_gravity_only", values=vals,
                   text_zh=(f"貼地傾斜 5°→50°({t5:.2f}–{t50:.2f} s)期間加速度計不只是重力(AccZ 殘差 RMS {rms:.2f} m/s² ≥ {TIP_RMS_MAX}),"
                            f"有推力或外力參與,不是單純被重力翻倒。"))
    failed = [k for k, v in checks.items() if not v]
    status = HIT if not failed else SOFT
    vals["failed"] = failed
    hover_txt = "" if hover is None else (f"(低於懸停 {hover:.0f} µs)" if checks.get("pwm_below_hover") else f"(懸停 {hover:.0f} µs)")
    gyro_txt = {True: "、角速度單調增大", False: "、角速度沒有單調增大"}.get(checks.get("gyro_monotone"), "")
    txt = (f"翻倒過程({t5:.2f}–{t50:.2f} s,貼地傾斜 5°→50°)加速度計只看到重力(AccZ 殘差 RMS {rms:.2f} m/s²)、"
           f"油門 {_fmt(tho_p90, '.2f')}、最大 PWM {_fmt(pwm_max, '.0f')} µs{hover_txt}{gyro_txt}:"
           f"機身是被地面撬起後靠重力翻過去,不是馬達失去推力或空中失控。")
    if failed:
        txt += f"(部分條件未成立:{'、'.join(failed)},列為 SOFT)"
    return _fd(ID_TIPOVER, status, "medium", t_onset=float(t5), t_alarm=float(t50), context="ground_tipover",
               values=vals, text_zh=txt)


# =================================================================== 4. touchdown_lateral_speed

def _vio_speed_series(ctx: Ctx, w0: float, w1: float):
    """[w0, w1) 的 VIO 水平速度 (t, 速度):VISV 優先,沒有 VISV 用 VISP 位置的 0.5 s 差分;沒資料回 None。"""
    vv = ctx.table("VISV")
    if _cols(vv, "VX", "VY"):
        m = (vv["t"] >= w0) & (vv["t"] < w1) & np.isfinite(vv["VX"]) & np.isfinite(vv["VY"])
        if m.any():
            return vv["t"][m], np.hypot(vv["VX"][m], vv["VY"][m])
    vp = ctx.table("VISP")
    if _cols(vp, "PX", "PY"):
        i0, i1 = _span(vp["t"], w0 - 0.5, w1)
        t, x, y = vp["t"][i0:i1], vp["PX"][i0:i1], vp["PY"][i0:i1]
        fin = np.isfinite(x) & np.isfinite(y)
        t, x, y = t[fin], x[fin], y[fin]
        if t.size >= 3:
            j = np.searchsorted(t, t - 0.5, side="left")
            ok = (t - t[j] > 0.3) & (t >= w0)
            if ok.any():
                return t[ok], np.hypot(x[ok] - x[j][ok], y[ok] - y[j][ok]) / (t[ok] - t[j][ok])
    return None


def _lateral_sources(ctx: Ctx, w0: float, w1: float) -> dict:
    """觸地前視窗 [w0, w1) 的各來源水平速度序列:{'ekf': (t, v), 'vio': (t, v), 'gps': (t, v)}(沒有有限值資料的來源不列;
    NaN 樣本剔除,不會讓 max 變 NaN)。GPS 只在 Status≥3 且 NSats≥6 才採用(無定位時欄位是舊值)。"""
    out: dict = {}
    xk, gp = ctx.table("XKF1"), ctx.table("GPS")
    if _cols(xk, "VN", "VE"):
        m = (xk["t"] >= w0) & (xk["t"] < w1) & np.isfinite(xk["VN"]) & np.isfinite(xk["VE"])
        if m.any():
            out["ekf"] = (xk["t"][m], np.hypot(xk["VN"][m], xk["VE"][m]))
    vio = _vio_speed_series(ctx, w0, w1)
    if vio is not None:
        out["vio"] = vio
    if _cols(gp, "Status", "NSats", "Spd"):
        with np.errstate(invalid="ignore"):
            m = (gp["t"] >= w0) & (gp["t"] < w1) & (gp["Status"] >= GPS_MIN_STATUS) & (gp["NSats"] >= GPS_MIN_SATS)                 & np.isfinite(gp["Spd"])
        if m.any():
            out["gps"] = (gp["t"][m], gp["Spd"][m])
    return out


@guarded(ID_LATERAL, DOMAIN)
def touchdown_lateral_speed(ctx: Ctx, a: Anchors):
    """觸地前 1 s 水平速度 ≥0.35 m/s,或 EKF 與 VIO 速度差 ≥0.3 m/s → HIT(門檻 70% → SOFT)。

    觸地 = t_warn 已知時取 [t_warn−5, t_warn+1] 內、之後到 t_warn 沒有重新起飛的第一個觸地衝擊,否則取最後一個觸地衝擊(降落)。
    速度來源:EKF XKF1 lane 0、VIO(VISV / VISP 差分)、GPS(Status≥3 且 NSats≥6 才採用;無定位時欄位是舊值)。
    衝擊大小本身分不開正常與失事觸地(25.5 vs 28.9 m/s²),所以只看速度。t_onset = 速度首次達門檻的時間(沒有則為觸地時間)。"""
    iv = a.interval
    if iv is None:
        return _not_armed(ID_LATERAL)
    shocks = _shocks(ctx, iv)
    if shocks is None:
        return _na(ID_LATERAL, "no_imu", "沒有 IMU 加速度資料,無法定位觸地時間。")
    tds = _touchdowns(ctx, iv, shocks)
    if not tds:
        return _na(ID_LATERAL, "no_touchdown", "這次解鎖沒有偵測到觸地衝擊,無法評估觸地水平速度。")
    if a.t_warn is not None:
        near = [d for d in tds if a.t_warn - 5.0 <= d.shock.t_start <= a.t_warn + 1.0
                and _refly(ctx, iv, d.shock.t_start, a.t_warn) is None]
        if not near:
            return _na(ID_LATERAL, "no_touchdown_near_warn", "姿態異常前後沒有觸地衝擊,觸地水平速度與這次事件無關。")
        td = near[0]          # spec:窗內『第一個』觸地(之後的簇是彈跳,彈跳前的速度不是觸地速度)
    else:
        td = tds[-1]
    t_td = td.shock.t_start
    src = _lateral_sources(ctx, t_td - LAT_WIN, t_td)
    if not src:
        return _na(ID_LATERAL, "no_velocity", f"{t_td:.2f} s 觸地前 1 s 沒有可用的速度資料(EKF / VIO / GPS)。")
    vmax_by = {k: float(v.max()) for k, (_, v) in src.items()}
    v_ekf, v_vio, v_gps = vmax_by.get("ekf"), vmax_by.get("vio"), vmax_by.get("gps")
    vmax = max(vmax_by.values())
    diff = abs(v_ekf - v_vio) if (v_ekf is not None and v_vio is not None) else None
    ay = ctx.series("IMU.AccY")
    lat_imp = None
    if ay is not None:
        m = (ay[0] >= t_td) & (ay[0] <= t_td + LAT_IMPACT_WIN)
        lat_imp = _nanmax(np.abs(ay[1][m])) if m.any() else None
    crossed = [float(t[v >= LAT_SPEED][0]) for t, v in src.values() if (v >= LAT_SPEED).any()]
    t_cross = min(crossed) if crossed else float(t_td)
    hit = vmax >= LAT_SPEED or (diff is not None and diff >= LAT_MISMATCH)
    soft = vmax >= LAT_SOFT_FRAC * LAT_SPEED or (diff is not None and diff >= LAT_SOFT_FRAC * LAT_MISMATCH)
    vals = {"t_touch": _f(t_td, 3), "v_ekf": _f(v_ekf, 3), "v_vio": _f(v_vio, 3), "v_gps": _f(v_gps, 3), "v_max": _f(vmax, 3),
            "ekf_vio_diff": _f(diff, 3), "lateral_impact_ms2": _f(lat_imp, 1), "peak_ms2": _f(td.shock.peak, 1),
            "confidence": "medium",
            "thresholds": {"speed": LAT_SPEED, "mismatch": LAT_MISMATCH, "window_s": LAT_WIN, "soft_frac": LAT_SOFT_FRAC}}
    names = {"vio": "VIO", "ekf": "EKF", "gps": "GPS"}
    sp_txt = " / ".join(f"{names[k]} {vmax_by[k]:.2f}" for k in ("vio", "ekf", "gps") if k in vmax_by) + " m/s"
    if hit or soft:
        mismatch = diff is not None and diff >= LAT_SOFT_FRAC * LAT_MISMATCH
        txt = (f"{t_td:.2f} s 觸地前 1 s 水平速度 {sp_txt}(正常著陸約 0.15 m/s)"
               f"{'' if diff is None else f',EKF 與 VIO 速度差 {diff:.2f} m/s'}:"
               f"{'觸地帶著橫向速度,起落架容易卡住或滑動' if vmax >= LAT_SOFT_FRAC * LAT_SPEED else '水平速度本身不大'}"
               f"{',且導航估計與 VIO 不一致' if mismatch else ''}。建議降落最後階段降低水平速度,並確認 VIO 速度與 EKF 一致。")
        if not hit:
            txt += "(接近門檻,列為 SOFT)"
        return _fd(ID_LATERAL, HIT if hit else SOFT, "medium", t_onset=float(t_cross), t_alarm=float(t_td),
                   context="touchdown", values=vals, text_zh=txt)
    return _fd(ID_LATERAL, CLEAR, "info", t_onset=float(t_td), context="touchdown", values=vals,
               text_zh=(f"{t_td:.2f} s 觸地前 1 s 水平速度 {sp_txt},低於 {LAT_SPEED} m/s"
                        f"{'' if diff is None else ',EKF 與 VIO 之間也一致'}:觸地水平速度正常。"))


# =================================================================== 5. ekf_phantom_motion_on_ground

def _static_windows(t: np.ndarray, anorm: np.ndarray, gdeg: np.ndarray, t0: float):
    """以 PH_HOP 為步長、PH_WIN 為長度的滑動視窗(cumsum 向量化):回傳 (視窗起點, 終點, 樣本數, 角速度均值, |a| 均值, |a| 標準差)。
    比固定分箱穩:靜止段不會因為分箱邊界被切短。"""
    k = int(round(PH_WIN / PH_HOP))
    b = np.floor((t - t0) / PH_HOP).astype(int)
    nb = int(b.max()) + 1
    if nb < k:
        return None
    cs = lambda v: np.concatenate([[0.0], np.cumsum(v)])      # noqa: E731
    n = cs(np.bincount(b, minlength=nb).astype(float))
    g = cs(np.bincount(b, weights=gdeg, minlength=nb))
    s1 = cs(np.bincount(b, weights=anorm, minlength=nb))
    s2 = cs(np.bincount(b, weights=anorm * anorm, minlength=nb))
    nn = n[k:] - n[:-k]
    den = np.maximum(nn, 1.0)
    gm = (g[k:] - g[:-k]) / den
    am = (s1[k:] - s1[:-k]) / den
    asd = np.sqrt(np.maximum((s2[k:] - s2[:-k]) / den - am * am, 0.0))
    ts = t0 + np.arange(nn.size) * PH_HOP
    return ts, ts + k * PH_HOP, nn, gm, am, asd


@guarded(ID_PHANTOM, DOMAIN)
def ekf_phantom_motion_on_ground(ctx: Ctx, a: Anchors):
    """機身在地面靜止(IMU 靜止 + 怠速 ≥0.5 s)但 EKF 速度中位數 ≥0.15 m/s 或位移 ≥0.3 m → HIT。

    同一段 VIO 位移 ≥0.15 m(機身真的被推動)→ 降為 SOFT;VIO 中斷(資料跨度<0.3 s)則不做交叉驗證。
    靜止 = 視窗(0.2 s)角速度 <6°/s、|a|−9.8 <0.8、|a| 標準差 <0.6。撞擊 / 失控(t_stop)之後的樣本不算。
    NaN 樣本先剔除(IMU 的 NaN 會經 cumsum 污染之後所有視窗;EKF / VIO 的 NaN 會讓中位數 / 位移變 NaN)。
    『觸地後累計漂移』另外回報 VIO 實際涵蓋的時段與中斷(VIO 中斷時兩者比較的時段不同)。"""
    iv = a.interval
    if iv is None:
        return _not_armed(ID_PHANTOM)
    xk, ct, rc = ctx.table("XKF1"), ctx.table("CTUN"), ctx.table("RCOU")
    if not _cols(xk, "VN", "VE", "PN", "PE") or not _cols(ct, "ThO") or rc is None:
        return _na(ID_PHANTOM, "no_data", "缺 EKF(XKF1)、CTUN 或 RCOU 資料,無法判斷地面上的 EKF 幻影運動。")
    shocks = _shocks(ctx, iv)
    t_stop = _t_stop(ctx, a, iv, shocks)
    ac, gy = _imu_acc(ctx, iv.t0, min(iv.t1, t_stop)), _imu_gyro_deg(ctx, iv.t0, min(iv.t1, t_stop))
    if ac is None or gy is None or ac[0].size != gy[0].size:
        return _na(ID_PHANTOM, "no_imu", "缺 IMU 資料,無法判斷機身是否靜止。")
    fin = np.isfinite(ac[2]) & np.isfinite(gy[1])
    if fin.sum() < 3:
        return _na(ID_PHANTOM, "no_imu", "IMU 資料全是無效值(NaN),無法判斷機身是否靜止。")
    imu_t, imu_a, imu_g = ac[0][fin], ac[2][fin], gy[1][fin]
    chs, _ = _motor_channels(ctx, rc)
    pmin, pmax = _pwm_range(ctx)
    thr = {"speed": PH_SPEED, "disp_m": PH_DISP, "min_s": PH_MIN_S, "gyro_deg_s": PH_GYRO_DEG, "vio_disp_m": PH_VIO_DISP}
    w = _static_windows(imu_t, imu_a, imu_g, iv.t0)
    if w is None:
        return _na(ID_PHANTOM, "no_static_idle", "解鎖期間太短,沒有可評估的靜止段。", thresholds=thr)
    ts, te, nn, gm, am, asd = w
    exp_n = PH_WIN / max(float(np.median(np.diff(ac[0]))), 1e-6)       # 名目取樣率(NaN 剔除前),缺樣本的視窗不算靜止
    tc = 0.5 * (ts + te)
    static = (nn >= 0.5 * exp_n) & (gm < PH_GYRO_DEG) & (np.abs(am - G0) < PH_ACC_DEV) & (asd < PH_ACC_SD)
    idle = _hold(ct["t"], ct["ThO"], tc) <= THO_IDLE
    if chs:
        pw = np.stack([_hold(rc["t"], rc[f"C{n}"], tc) for n in chs], axis=1)
        idle &= pw.min(axis=1) >= pmin + SPOOL_FRAC * (pmax - pmin)
    runs = [r for r in util.runs(static & idle, ts, 0.0, max_gap=PH_HOP * 1.5) if te[r[3]] - ts[r[2]] >= PH_MIN_S - DUR_TOL]
    if not runs:
        return _na(ID_PHANTOM, "no_static_idle", "解鎖期間沒有『IMU 靜止 + 怠速』持續 0.5 s 的區段,無法判斷地面上的 EKF 幻影運動。",
                   thresholds=thr)
    vp = ctx.table("VISP")
    res = []
    for (_, _, i0, i1) in runs:
        a0, a1 = float(ts[i0]), float(te[i1])
        m = (xk["t"] >= a0) & (xk["t"] <= a1) & np.isfinite(xk["VN"]) & np.isfinite(xk["VE"])
        dp = _disp(xk["t"], xk["PN"], xk["PE"], a0, a1)
        if m.sum() < 2 or dp is None:
            continue
        sp = np.hypot(xk["VN"][m], xk["VE"][m])
        disp = dp[0]
        vio = None
        if _cols(vp, "PX", "PY"):
            dv = _disp(vp["t"], vp["PX"], vp["PY"], a0, a1)
            if dv is not None and dv[2] - dv[1] >= PH_VIO_MIN_SPAN:
                vio = dv[0]
        moved = float(np.median(sp)) >= PH_SPEED or disp >= PH_DISP
        res.append({"t0": a0, "t1": a1, "speed_med": float(np.median(sp)), "speed_max": float(sp.max()), "disp": disp,
                    "vio": vio, "moved": moved, "ekf_wrong": moved and (vio is None or vio < PH_VIO_DISP)})
    if not res:
        return _na(ID_PHANTOM, "no_ekf_rows", "靜止段內沒有足夠的 EKF 樣本。", thresholds=thr)
    flagged = [r for r in res if r["moved"]]
    if not flagged:
        r = max(res, key=lambda x: x["speed_med"])
        return _fd(ID_PHANTOM, CLEAR, "info", t_onset=res[0]["t0"], context="static_idle",
                   values={"n_runs": len(res), "static_idle_s": _f(sum(x["t1"] - x["t0"] for x in res), 1),
                           "speed_med_max": _f(max(x["speed_med"] for x in res), 3), "speed_max": _f(max(x["speed_max"] for x in res), 3),
                           "disp_max": _f(max(x["disp"] for x in res), 3), "confidence": "high", "thresholds": thr},
                   text_zh=(f"機身靜止在地面怠速共 {sum(x['t1'] - x['t0'] for x in res):.1f} s,EKF 速度中位數最大 "
                            f"{max(x['speed_med'] for x in res):.2f} m/s、位移最大 {max(x['disp'] for x in res):.2f} m"
                            f"(門檻 {PH_SPEED} m/s、{PH_DISP} m):沒有幻影運動。"))
    best = [r for r in flagged if r["ekf_wrong"]]
    r = (best or flagged)[0]
    status = HIT if best else SOFT
    # 觸地後到地面階段結束的累計漂移(給報告敘述;不影響狀態)
    tds = _touchdowns(ctx, iv, shocks) if shocks is not None else []
    t_td = a.t_touch if a.t_touch is not None and r["t0"] - 4.0 <= a.t_touch <= r["t0"] else None
    if t_td is None:
        c = [d.shock.t_start for d in tds if r["t0"] - 4.0 <= d.shock.t_start <= r["t0"]]
        t_td = c[-1] if c else None
    drift = vio_drift = ekf_vio_span = None
    vio_span = vio_gap = None                   # VIO 實際涵蓋的時段;[t_td, t_end] 內最長的 VISP 中斷(≥0.5 s 才記)
    t_end = None
    if t_td is not None:
        t_end = _first_tilt(ctx, r["t0"], t_stop, PH_TILT_END_DEG) or min(r["t1"], t_stop)
        de = _disp(xk["t"], xk["PN"], xk["PE"], t_td, t_end)
        drift = None if de is None else de[0]
        if _cols(vp, "PX", "PY"):
            dv = _disp(vp["t"], vp["PX"], vp["PY"], t_td, t_end)
            if dv is not None:
                vio_drift, vio_span = dv[0], (dv[1], dv[2])
                de2 = _disp(xk["t"], xk["PN"], xk["PE"], dv[1], dv[2])           # 同一段時間的 EKF(對等比較)
                ekf_vio_span = None if de2 is None else de2[0]
            tv = vp["t"][np.isfinite(vp["PX"]) & np.isfinite(vp["PY"])]
            i0 = max(int(np.searchsorted(tv, t_td, side="left")) - 1, 0)        # 含觸地前最後一筆:中斷可能從觸地前就開始
            i1 = int(np.searchsorted(tv, t_end, side="right"))
            seg = np.concatenate([tv[i0:i1], [t_end]]) if tv.size else np.array([t_td, t_end])
            if seg.size >= 2:
                k = int(np.argmax(np.diff(seg)))
                if seg[k + 1] - seg[k] >= PH_VIO_GAP_S:
                    vio_gap = (float(seg[k]), float(seg[k + 1]))
    vals = {"t_run_start": _f(r["t0"], 3), "t_run_end": _f(r["t1"], 3), "run_s": _f(r["t1"] - r["t0"], 2),
            "ekf_speed_med": _f(r["speed_med"], 3), "ekf_speed_max": _f(r["speed_max"], 3), "ekf_disp_m": _f(r["disp"], 3),
            "vio_disp_m": _f(r["vio"], 3), "ekf_wrong": bool(r["ekf_wrong"]), "n_flagged": len(flagged),
            "t_touch": _f(t_td, 3), "t_drift_end": _f(t_end, 3), "ekf_drift_since_touch_m": _f(drift, 2),
            "vio_drift_since_touch_m": _f(vio_drift, 2),
            "vio_span_t0": _f(vio_span[0], 3) if vio_span else None, "vio_span_t1": _f(vio_span[1], 3) if vio_span else None,
            "ekf_drift_vio_span_m": _f(ekf_vio_span, 2),
            "vio_gap_t0": _f(vio_gap[0], 3) if vio_gap else None, "vio_gap_t1": _f(vio_gap[1], 3) if vio_gap else None,
            "vio_gap_s": _f(vio_gap[1] - vio_gap[0], 2) if vio_gap else None,
            "confidence": "high", "thresholds": {**thr, "vio_gap_s": PH_VIO_GAP_S}}
    vio_txt = "" if r["vio"] is None else f"同時 VIO 位置只動 {'<0.01' if r['vio'] < 0.01 else format(r['vio'], '.2f')} m。"
    drift_txt = ""
    if drift is not None:
        drift_txt = f"從觸地({t_td:.2f} s)到 {t_end:.2f} s EKF 累計漂移約 {drift:.1f} m"
        if vio_drift is None:
            drift_txt += "(這段沒有 VIO 位置可對照)。"
        elif vio_gap is None:
            drift_txt += f"(同段 VIO 只動 {vio_drift:.2f} m)。"
        else:
            drift_txt += (f";VIO 在 {vio_gap[0]:.2f}–{vio_gap[1]:.2f} s 中斷 {vio_gap[1] - vio_gap[0]:.2f} s,"
                          f"只能比較有 VIO 的 {vio_span[0]:.2f}–{vio_span[1]:.2f} s:VIO 動 {vio_drift:.2f} m、"
                          f"同段 EKF 動 {_fmt(ekf_vio_span, '.2f')} m。")
    txt = (f"機身在地面靜止(IMU 靜止、油門怠速)的 {r['t0']:.2f}–{r['t1']:.2f} s,EKF 卻認為在移動:"
           f"速度中位數 {r['speed_med']:.2f} m/s、位移 {r['disp']:.2f} m(健康怠速 ≤0.05 m/s、0.03 m)。{vio_txt}{drift_txt}"
           f"這是 Loiter 往錯方向用力的來源;可能原因:撞擊加速度被積分成速度、VIO 中斷、使用中的來源組沒有速度量測。")
    if not best:
        txt += "(同段 VIO 也有位移,列為 SOFT)"
    return _fd(ID_PHANTOM, status, "high", t_onset=float(r["t0"]), t_alarm=float(r["t0"] + PH_MIN_S), context="static_idle",
               values=vals, text_zh=txt)


# =================================================================== 入口

def run(ctx: Ctx, a: Anchors) -> list[Finding]:
    """依序執行 5 個偵測器;touchdown_not_recognised 會設定 a.t_touch。"""
    out: list[Finding] = []
    for fn in (touchdown_not_recognised, idle_motor_asymmetry, gravity_only_tipover,
               touchdown_lateral_speed, ekf_phantom_motion_on_ground):
        out.extend(fn(ctx, a))
    return out
