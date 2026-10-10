"""EKF 來源檢查:EKF 是否「偷參考」GPS(想看純 VIO 飛行時)。

兩層證據(單看新息欄位 XKF3 / XKF4 無法分辨資料來自 GPS 還是 VIO,所以不用):
 1. 設定層:使用中的來源組(XKFS.SS)對照 EK3_SRCn_POSXY / POSZ / VELXY / VELZ / YAW,
    列出每一軸宣告用哪個感測器、哪幾軸含 GPS。這是「設定允許」,不等於實際有融合。
 2. 比對層:EKF 輸出實際跟著誰走。取 a = EKF − VIO、b = GPS − VIO(同一時刻、VIO 已對齊
    到 GPS 座標框),對 b 做迴歸得洩漏係數 α:α≈0 表示 EKF 只跟 VIO,α≈1 表示被 GPS 帶著走。
    只有 b 在視窗內有足夠變化(GPS 與 VIO 看起來不同)時才判得出來,否則 α = NaN。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import align as align_mod
from . import pipeline, sync
from .types import EV_EVENT, LogData

AXES = [("POSXY", "水平位置"), ("POSZ", "高度"), ("VELXY", "水平速度"),
        ("VELZ", "垂直速度"), ("YAW", "航向")]
AXIS_ZH = dict(AXES)

# EK3_SRCn_* 值 → 名稱(ArduPilot EKF3 Source)
CODE_NAMES = {
    "POSXY": {0: "無", 3: "GPS", 4: "Beacon", 6: "ExternalNav(VIO)", 7: "輪編碼器"},
    "POSZ": {0: "無", 1: "氣壓計", 2: "測距儀", 3: "GPS", 4: "Beacon", 6: "ExternalNav(VIO)"},
    "VELXY": {0: "無", 3: "GPS", 4: "Beacon", 5: "光流", 6: "ExternalNav(VIO)", 7: "輪編碼器"},
    "VELZ": {0: "無", 3: "GPS", 4: "Beacon", 6: "ExternalNav(VIO)"},
    "YAW": {0: "無", 1: "羅盤", 2: "GPS 航向", 3: "GPS+羅盤", 6: "ExternalNav(VIO)", 8: "GSF"},
}
_GPS = {"POSXY": {3}, "POSZ": {3}, "VELXY": {3}, "VELZ": {3}, "YAW": {2, 3}}
_VIO = 6
_DEFAULT_SET = {1: {"POSXY": 3, "POSZ": 1, "VELXY": 3, "VELZ": 3, "YAW": 1}}   # 來源組 2、3 預設全 0
_EVENT_SET = {"EK3_SOURCES_SET_TO_PRIMARY": 1, "EK3_SOURCES_SET_TO_SECONDARY": 2,
              "EK3_SOURCES_SET_TO_TERTIARY": 3}


@dataclass
class SourceSet:
    n: int                              # 1 ~ 3
    codes: dict[str, int]

    def name(self, axis: str) -> str:
        c = self.codes.get(axis, 0)
        return CODE_NAMES[axis].get(c, f"代碼 {c}")

    @property
    def gps_axes(self) -> list[str]:
        return [a for a, _ in AXES if self.codes.get(a, 0) in _GPS[a]]

    @property
    def vio_axes(self) -> list[str]:
        return [a for a, _ in AXES if self.codes.get(a, 0) == _VIO]

    @property
    def configured(self) -> bool:
        return any(v for v in self.codes.values())

    @property
    def verdict(self) -> str:
        g, v = self.gps_axes, self.vio_axes
        if not self.configured:
            return "未設定"
        zh = "、".join(AXIS_ZH[a] for a in g)
        if v and not g:
            return "純 VIO(沒有任何軸用 GPS)"
        if v and g:
            return f"VIO + GPS({zh}用 GPS)"
        if g:
            return "GPS(沒有 VIO)"
        return "無 GPS、無 VIO"


@dataclass
class Segment:
    t0: float                           # log 絕對秒
    t1: float
    set_n: int
    gps_fix: float | None = None        # 區間內 GPS 有 3D 定位(Status ≥ 3)的樣本比例
    alpha: float | None = None          # 洩漏係數(算得出來才有)
    alpha_std_b: float | None = None    # 支撐 α 的 GPS−VIO 變化量(m);小 → α 不可靠


@dataclass
class EkfSourceInfo:
    sets: dict[int, SourceSet]
    segments: list[Segment]
    options: int = 0                    # EK3_SRC_OPTIONS
    assumed: bool = False               # 沒有 XKFS / 切換事件:整段假設用來源組 1
    notes: list[str] = field(default_factory=list)


def _param(pv: dict, key: str, default):
    return int(round(pv[key])) if key in pv else default


def source_sets(logdata: LogData) -> tuple[dict[int, SourceSet], int]:
    pv = {}
    if logdata.index is not None:
        try:
            pv = logdata.index.param_values()
        except Exception:  # noqa: BLE001
            pv = {}
    sets = {}
    for n in (1, 2, 3):
        base = _DEFAULT_SET.get(n, {a: 0 for a, _ in AXES})
        sets[n] = SourceSet(n, {a: _param(pv, f"EK3_SRC{n}_{a}", base[a]) for a, _ in AXES})
    return sets, _param(pv, "EK3_SRC_OPTIONS", 0)


def active_segments(logdata: LogData) -> tuple[list[Segment], bool]:
    """使用中的來源組隨時間的分段 [(t0, t1, 組別)];第二個回傳值 = 是否為假設(沒有切換紀錄)。"""
    lo, hi = float(logdata.t_base), float(logdata.t_last)
    pts: list[tuple[float, int]] = []
    idx = logdata.index
    s = None
    if idx is not None and idx.has("XKFS.SS"):
        s = idx.series("XKFS.SS")
    if s is not None and len(s[0]):
        pts = [(float(t), int(v) + 1) for t, v in zip(*s)]
    else:
        pts = sorted((e.t, _EVENT_SET[e.label]) for e in logdata.events
                     if e.kind == EV_EVENT and e.label in _EVENT_SET)
    assumed = not pts
    cur = 1                              # 第一筆切換之前:EKF 預設用來源組 1
    runs: list[list] = [[lo, cur]]
    for t, n in pts:
        if n == runs[-1][1]:
            continue
        if t <= runs[-1][0]:
            runs[-1][1] = n
        else:
            runs.append([t, n])
    segs = []
    for i, (t0, n) in enumerate(runs):
        t1 = runs[i + 1][0] if i + 1 < len(runs) else hi
        if t1 > t0:
            segs.append(Segment(float(t0), float(t1), int(n)))
    return segs, assumed


def _gps_fix_fraction(logdata: LogData, t0: float, t1: float) -> float | None:
    idx = logdata.index
    if idx is None or not idx.has("GPS.Status"):
        return None
    t, v = idx.series("GPS.Status")
    m = (t >= t0) & (t <= t1)
    return float(np.mean(v[m] >= 3)) if m.any() else None


def inspect(logdata: LogData) -> EkfSourceInfo:
    sets, opt = source_sets(logdata)
    segs, assumed = active_segments(logdata)
    for sg in segs:
        sg.gps_fix = _gps_fix_fraction(logdata, sg.t0, sg.t1)
    notes = []
    if assumed:
        notes.append("log 沒有 XKFS / 來源切換事件:整段假設使用來源組 1")
    if opt & 1:
        notes.append("EK3_SRC_OPTIONS 已啟用 FuseAllVelocities(同時融合多個速度來源),"
                     "「純 VIO」的判定要連同比對層的洩漏係數一起看")
    if logdata.index is not None and not logdata.index.param_values():
        notes.append("log 沒有參數表:來源組設定用 ArduPilot 預設值")
    return EkfSourceInfo(sets, segs, opt, assumed, notes)


# ---------------- 比對層:洩漏係數 ----------------

def leak_coefficient(ekf: np.ndarray, vio: np.ndarray, gps: np.ndarray,
                     min_std: float = 0.5) -> tuple[float, float]:
    """(α, std_b):a = EKF − VIO、b = GPS − VIO(水平 E、N,各去平均),α = Σa·b / Σb·b。

    b 的標準差(每軸 RMS)< min_std 時 GPS 與 VIO 太接近,分不出 EKF 跟誰 → α = NaN。
    """
    a = np.asarray(ekf, dtype=float)[:, :2] - np.asarray(vio, dtype=float)[:, :2]
    b = np.asarray(gps, dtype=float)[:, :2] - np.asarray(vio, dtype=float)[:, :2]
    if len(a) < 5:
        return float("nan"), 0.0
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    den = float(np.sum(b * b))
    std_b = float(np.sqrt(den / b.size))
    if std_b < min_std:
        return float("nan"), std_b
    return float(np.sum(a * b) / den), std_b


@dataclass
class LeakResult:
    vio_name: str
    gps_name: str
    ekf_name: str
    t: np.ndarray                       # 視窗中心(log 絕對秒)
    alpha: np.ndarray                   # 每個視窗的 α(NaN = 不可判定)
    std_b: np.ndarray
    overall: float                      # 整段 α
    overall_std_b: float
    t_all: np.ndarray                   # 配對樣本時間
    ekf_enu: np.ndarray
    vio_enu: np.ndarray                 # 已對齊到 GPS
    gps_enu: np.ndarray
    min_std: float
    win_s: float


def pick_ekf_source(logdata: LogData) -> str | None:
    for n in ("POS", "XKF1[0]", "XKF1"):
        if n in logdata.sources:
            return n
    return None


def leak_analysis(logdata: LogData, vio_name: str, gps_name: str, ekf_name: str | None = None,
                  t0: float | None = None, t1: float | None = None, win_s: float = 30.0,
                  step_s: float = 10.0, min_std: float = 0.5, max_gap_s: float = 0.5) -> LeakResult:
    """EKF 輸出相對 GPS / VIO 的洩漏係數(整段 + 滑動視窗)。以 GPS 時間戳為基準。"""
    ekf_name = ekf_name or pick_ekf_source(logdata)
    if ekf_name is None:
        raise ValueError("log 沒有 EKF 位置來源(POS / XKF1)")
    for nm in (vio_name, gps_name, ekf_name):
        if nm not in logdata.sources:
            raise ValueError(f"找不到來源 {nm}")
    if len({vio_name, gps_name, ekf_name}) < 3:
        raise ValueError("VIO、GPS、EKF 必須是三個不同的來源")
    gps, vio, ekf = (logdata.sources[n] for n in (gps_name, vio_name, ekf_name))
    origin, _ = pipeline.resolve_origin(logdata, gps, vio)
    if origin is None and "global" in (gps.kind, vio.kind, ekf.kind):
        raise ValueError("沒有 ENU 原點")
    g_enu, v_enu, e_enu = gps.enu(origin), vio.enu(origin), ekf.enu(origin)
    t_a, gp_a, vp = sync.sync_pair(gps.t, g_enu, vio.t, v_enu, t0=t0, t1=t1, max_gap_s=max_gap_s)
    t_b, _, ep = sync.sync_pair(gps.t, g_enu, ekf.t, e_enu, t0=t0, t1=t1, max_gap_s=max_gap_s)
    t, ia, ib = np.intersect1d(t_a, t_b, return_indices=True)
    if t.size < 20:
        raise ValueError(f"三個來源同時有資料的樣本不足({t.size} 筆)")
    gp, vp, ep = gp_a[ia], vp[ia], ep[ib]
    vp_al, _ = align_mod.align(vp, gp, "se3")       # VIO 對齊到 GPS 座標框(整段一次)

    alpha, std_b, mids = [], [], []
    lo, hi = float(t[0]), float(t[-1])
    s = lo
    while s + win_s <= hi + 1e-9:
        m = (t >= s) & (t < s + win_s)
        a, sb = leak_coefficient(ep[m], vp_al[m], gp[m], min_std)
        alpha.append(a); std_b.append(sb); mids.append(s + win_s / 2)
        s += step_s
    oa, osb = leak_coefficient(ep, vp_al, gp, min_std)
    return LeakResult(vio_name, gps_name, ekf_name, np.array(mids), np.array(alpha), np.array(std_b),
                      oa, osb, t, ep, vp_al, gp, min_std, win_s)


def segment_alpha(res: LeakResult, sg: Segment) -> tuple[float, float]:
    m = (res.t_all >= sg.t0) & (res.t_all <= sg.t1)
    if m.sum() < 20:
        return float("nan"), 0.0
    return leak_coefficient(res.ekf_enu[m], res.vio_enu[m], res.gps_enu[m], res.min_std)


def apply_leak(info: EkfSourceInfo, res: LeakResult | None) -> None:
    for sg in info.segments:
        if res is None:
            sg.alpha = sg.alpha_std_b = None
        else:
            a, sb = segment_alpha(res, sg)
            sg.alpha = None if not np.isfinite(a) else a
            sg.alpha_std_b = sb


def judge(info: EkfSourceInfo, sg: Segment) -> str:
    """一段的綜合判讀:設定層 + 比對層。"""
    ss = info.sets.get(sg.set_n)
    cfg = ss.verdict if ss else "未知"
    if sg.alpha is None:
        return cfg + " — 未驗證實際融合(沒有比對資料,或 GPS 與 VIO 太接近)"
    a = sg.alpha
    if ss is not None and ss.vio_axes and not ss.gps_axes:
        if a > 0.5:
            return f"⚠ 設定是純 VIO,但 EKF 輸出明顯跟著 GPS(α={a:.2f})"
        if a > 0.2:
            return f"設定是純 VIO;EKF 輸出略受 GPS 影響(α={a:.2f})"
        return f"純 VIO,EKF 輸出只跟 VIO(α={a:.2f})"
    if a < 0.2:
        return f"{cfg};EKF 輸出實際只跟 VIO(α={a:.2f})"
    if a > 0.8:
        return f"{cfg};EKF 輸出跟 GPS 一致(α={a:.2f})"
    return f"{cfg};EKF 輸出介於 VIO 與 GPS 之間(α={a:.2f})"


def format_report(logdata: LogData, info: EkfSourceInfo, res: LeakResult | None) -> str:
    tb = logdata.t_base
    L = ["EKF 來源檢查 — vio_ate_studio", f"資料來源:{logdata.path or '(模擬)'}", "",
         "來源組設定(EK3_SRCn_*):"]
    head = f"  {'':<10}" + "".join(f"{'來源組 ' + str(n):<22}" for n in (1, 2, 3))
    L.append(head)
    for a, zh in AXES:
        row = "".join(f"{('★' if info.sets[n].codes.get(a, 0) in _GPS[a] else ' ') + info.sets[n].name(a):<22}"
                      for n in (1, 2, 3))
        L.append(f"  {zh:<8}" + row)
    L.append("  (★ = 該軸使用 GPS)")
    for n in (1, 2, 3):
        L.append(f"  來源組 {n}:{info.sets[n].verdict}")
    L += ["", "使用中的來源組(依 XKFS.SS / 切換事件):"]
    for sg in info.segments:
        fix = "" if sg.gps_fix is None else f",GPS 定位 {sg.gps_fix * 100:.0f}%"
        L.append(f"  {sg.t0 - tb:8.1f} ~ {sg.t1 - tb:8.1f} s  來源組 {sg.set_n}{fix}  → {judge(info, sg)}")
    if res is not None:
        L += ["", f"洩漏係數(EKF 輸出 {res.ekf_name};GPS = {res.gps_name}、VIO = {res.vio_name}):"
              f"整段 α = " + ("不可判定" if not np.isfinite(res.overall) else f"{res.overall:.2f}")
              + f"(GPS−VIO 變化 {res.overall_std_b:.2f} m)",
              "  α≈0:EKF 只跟 VIO;α≈1:EKF 被 GPS 帶著走。"]
    for n in info.notes:
        L.append(f"註:{n}")
    return "\n".join(L)
