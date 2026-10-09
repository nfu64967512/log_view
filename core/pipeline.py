"""ATE 計算管線:來源選擇 → ENU → 時間同步 → 空間對齊 → 統計;含匯出。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import align as align_mod
from . import metrics, sync, timeline
from .types import KIND_GLOBAL, KIND_LABEL, KIND_NED, LogData, TrajSource


@dataclass
class AteResult:
    ref_name: str
    est_name: str
    t: np.ndarray                 # 同步後時間戳(log 秒)
    ref_enu: np.ndarray           # Nx3
    est_enu: np.ndarray           # Nx3(已對齊)
    err: np.ndarray               # Nx3(估測 − 參考)
    e3: np.ndarray
    e2: np.ndarray
    stats: dict
    align_info: dict
    origin: tuple | None
    origin_desc: str
    cfg: dict = field(default_factory=dict)
    # 移動距離(參考 = GPS、估測 = VIO 時即「GPS 移動距離 / VIO 移動距離」):
    # 沿同步後的樣本累加 3D 位移;估測取「對齊前」的原始軌跡 —
    # Sim(3) 對齊會把尺度修正掉,若用對齊後的軌跡就看不出 VIO 的尺度誤差
    path_len_ref: float = 0.0
    path_len_est: float = 0.0
    cum_ref: np.ndarray = field(default_factory=lambda: np.zeros(0))   # 逐點累積距離(m)
    cum_est: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def dist_ratio(self) -> float | None:
        """估測移動距離 / 參考移動距離;參考沒動(距離 0)時為 None。"""
        return self.path_len_est / self.path_len_ref if self.path_len_ref > 1e-9 else None

    @property
    def n(self) -> int:
        return int(self.t.size)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n else 0.0


def resolve_origin(logdata: LogData, ref: TrajSource, est: TrajSource):
    """決定 ENU 原點:ORGN 優先,否則用第一個 global 來源的第一點。"""
    if ref.kind == KIND_NED and est.kind == KIND_NED:
        return None, "(皆為局部 NED,毋須原點)"
    if logdata.origin is not None:
        return logdata.origin, logdata.origin_desc or "ORGN"
    for s, which in ((ref, "參考"), (est, "估測")):
        if s.kind == KIND_GLOBAL:
            o = (float(s.lat[0]), float(s.lng[0]), float(s.alt[0]))
            return o, f"{which}軌跡第一點(log 無 ORGN,備援)"
    return None, ""


def compute(logdata: LogData, ref_name: str, est_name: str,
            t0: float | None = None, t1: float | None = None,
            offset_s: float = 0.0, method: str = "se3",
            max_gap_s: float = 0.5) -> AteResult:
    """t0 / t1 為 log 絕對秒(呼叫端自行從相對秒換算)。"""
    if ref_name == est_name:
        raise ValueError("參考與估測不能選同一個來源")
    ref = logdata.sources[ref_name]
    est = logdata.sources[est_name]

    origin, origin_desc = resolve_origin(logdata, ref, est)
    ref_enu = ref.enu(origin)
    est_enu = est.enu(origin)

    t, rp, ep = sync.sync_pair(ref.t, ref_enu, est.t, est_enu,
                               t0=t0, t1=t1, offset_s=offset_s, max_gap_s=max_gap_s)
    if t.size < 10:
        raise ValueError(f"同步後樣本不足({t.size} 筆);請放寬時間區間或內插間隙上限")

    ep_al, info = align_mod.align(ep, rp, method)
    err = ep_al - rp
    stats = metrics.ate_stats(err)
    cum_ref = metrics.path_cumulative(rp)
    cum_est = metrics.path_cumulative(ep)

    return AteResult(
        ref_name=ref_name, est_name=est_name,
        t=t, ref_enu=rp, est_enu=ep_al, err=err,
        e3=np.linalg.norm(err, axis=1), e2=np.linalg.norm(err[:, :2], axis=1),
        stats=stats, align_info=info, origin=origin, origin_desc=origin_desc,
        cfg={"t0": t0, "t1": t1, "offset_s": offset_s,
             "method": method, "max_gap_s": max_gap_s},
        path_len_ref=float(cum_ref[-1]), path_len_est=float(cum_est[-1]),
        cum_ref=cum_ref, cum_est=cum_est,
    )


def get_enu_pair(logdata: LogData, ref_name: str, est_name: str,
                 t0: float | None = None, t1: float | None = None):
    """供時間偏移自動搜尋等用途:回傳 (ref_t, ref_enu, est_t, est_enu)。

    給 t0 / t1(log 絕對秒)時只取 [t0, t1] 內的樣本 — 多航段 log 只看所選飛行段;
    不往外多取:VIO 重置常緊貼 ARM,跳點的速度尖峰會毀掉偏移搜尋。
    """
    ref = logdata.sources[ref_name]
    est = logdata.sources[est_name]
    origin, _ = resolve_origin(logdata, ref, est)
    out = [ref.t, ref.enu(origin), est.t, est.enu(origin)]
    if t0 is not None or t1 is not None:
        lo = -np.inf if t0 is None else float(t0)
        hi = np.inf if t1 is None else float(t1)
        for k in (0, 2):
            m = (out[k] >= lo) & (out[k] <= hi)
            out[k], out[k + 1] = out[k][m], out[k + 1][m]
    return tuple(out)


# ---------- 匯出 ----------

_ROWS = [("rmse", "RMSE"), ("mean", "平均"), ("median", "中位數"),
         ("std", "標準差"), ("max", "最大"), ("p90", "P90"), ("p95", "P95")]


def _disp_width(text: str) -> int:
    """顯示寬度:全形字算兩格(等寬字型下才會對齊)。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _ljust(text: str, width: int) -> str:
    return text + " " * max(0, width - _disp_width(text))


def _rjust(text: str, width: int) -> str:
    return " " * max(0, width - _disp_width(text)) + text


def format_report(res: AteResult, logdata: LogData) -> str:
    ref = logdata.sources[res.ref_name]
    est = logdata.sources[res.est_name]
    a = res.align_info
    s = res.stats
    tb = logdata.t_base
    lines = [
        "VIO / GPS ATE 分析報告 — vio_ate_studio",
        f"產生時間:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"資料來源:{logdata.path}",
        "",
        f"參考(基準):{res.ref_name}({KIND_LABEL[ref.kind]},{ref.n} 筆,≈{ref.rate_hz:.1f} Hz)",
        f"估測:{res.est_name}({KIND_LABEL[est.kind]},{est.n} 筆,≈{est.rate_hz:.1f} Hz)",
        f"分析區間:{res.t[0] - tb:.1f} ~ {res.t[-1] - tb:.1f} s(相對 log 起點),"
        f"同步樣本 {res.n} 筆,時長 {res.duration:.1f} s",
        *_segment_lines(res, logdata),
        f"時間偏移:{res.cfg['offset_s'] * 1000:.0f} ms;內插間隙上限:{res.cfg['max_gap_s']:.2f} s",
        f"ENU 原點:{_fmt_origin(res)}",
        f"空間對齊:{a['label']};yaw {a['yaw_deg']:+.2f}°,"
        f"平移 [{a['translation'][0]:+.2f}, {a['translation'][1]:+.2f}, {a['translation'][2]:+.2f}] m,"
        f"尺度 {a['scale']:.4f}",
        f"移動距離:{res.ref_name} {res.path_len_ref:.1f} m / {res.est_name} {res.path_len_est:.1f} m"
        + _ratio_text(res),
        "",
        f"ATE RMSE:3D {s['3d']['rmse']:.3f} m|水平 {s['2d']['rmse']:.3f} m",
        "",
        _ljust("統計量(公尺)", 14)
        + "".join(_rjust(h, 10) for h in ("3D", "水平", "東 E", "北 N", "天 U")),
        "-" * (14 + 10 * 5),
    ]
    for key, zh in _ROWS:
        row = [s["3d"][key], s["2d"][key],
               s["axes"]["E"][key], s["axes"]["N"][key], s["axes"]["U"][key]]
        signed = key in ("mean", "median")   # 軸向的平均/中位數帶正負號
        cells = [f"{v:+.3f}" if signed and i >= 2 else f"{v:.3f}"
                 for i, v in enumerate(row)]
        lines.append(_ljust(zh, 14) + "".join(_rjust(c, 10) for c in cells))
    lines += [
        "",
        "註:軸向之「平均 / 中位數」為帶號偏差;「最大 / P90 / P95」取絕對值。",
        "ATE = 時間同步(+空間對齊)後成對位置差的統計;3D 為誤差向量長度,水平只取東北分量。",
    ]
    return "\n".join(lines)


def _ratio_text(res: AteResult) -> str:
    r = res.dist_ratio
    if r is None:
        return ""
    return f"(估測為參考的 {r * 100:.1f}%,{(r - 1) * 100:+.1f}%;取對齊前的原始軌跡)"


def _segment_lines(res: AteResult, logdata: LogData) -> list[str]:
    """log 有 ARM 飛行段時,說明分析區間涵蓋哪幾段(沒有段則不輸出)。"""
    segs = logdata.segments
    if not segs:
        return []
    tb = logdata.t_base
    lo, hi = float(res.t[0]), float(res.t[-1])
    tol = 0.5   # 同步樣本落在參考時間戳上,與段起訖差不到一個取樣間隔
    parts = []
    for i in timeline.overlapping_segments(segs, lo, hi):
        sg = segs[i]
        part = "" if lo <= sg.t0 + tol and hi >= sg.t1 - tol else ",部分"
        parts.append(f"{sg.label}({sg.t0 - tb:.1f}~{sg.t1 - tb:.1f} s{part})")
    covered = "、".join(parts) if parts else "無(全在未 ARM 的地面段)"
    return [f"ARM 飛行段:共 {len(segs)} 段;分析區間涵蓋 {covered}"]


def _fmt_origin(res: AteResult) -> str:
    if res.origin is None:
        return res.origin_desc or "無"
    return (f"lat {res.origin[0]:.7f}°, lng {res.origin[1]:.7f}°, "
            f"alt {res.origin[2]:.1f} m — {res.origin_desc}")


def save_csv(res: AteResult, logdata: LogData, path: str) -> None:
    tb = logdata.t_base
    header = (
        f"# vio_ate_studio ATE 匯出|參考={res.ref_name} 估測={res.est_name} "
        f"對齊={res.align_info['method']} 偏移={res.cfg['offset_s'] * 1000:.0f}ms "
        f"移動距離 {res.ref_name}={res.path_len_ref:.2f}m {res.est_name}={res.path_len_est:.2f}m\n"
        f"# ENU 原點:{_fmt_origin(res)}\n"
        "t_rel_s,ref_E,ref_N,ref_U,est_E,est_N,est_U,err_E,err_N,err_U,err_2d,err_3d,"
        "ref_dist_m,est_dist_m\n"
    )
    data = np.column_stack([res.t - tb, res.ref_enu, res.est_enu, res.err, res.e2, res.e3,
                            res.cum_ref, res.cum_est])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.4f", delimiter=",")
