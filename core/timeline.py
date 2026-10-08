"""時間軸:ARM 飛行段偵測、模式查詢、軌跡逐時內插、播放時鐘(不依賴 Qt)。

ArduPilot 預設「每次 ARM 開一份 log」,但若設了 LOG_DISARMED 或沒依 ARM 切檔,
一份 log 會涵蓋整次開機:前後一長段地面待機,中間可能有好幾次起降。這裡把
ARM/DISARM 事件整理成飛行段,讓 UI 能挑段分析、並在時間軸上播放。
"""
from __future__ import annotations

from bisect import bisect_right

import numpy as np

from .types import EV_ARM, EV_DISARM, EV_MODE, LogEvent, TimeSegment

_EPS = 1e-6


# ---------------- 事件 → 飛行段 ----------------

_ARM_GAP_S = 5.0   # 推定 ARM 起點時,視為「log 中斷」的資料空洞長度


def first_state_is_disarm(events: list[LogEvent]) -> bool:
    evs = [e for e in events if e.kind in (EV_ARM, EV_DISARM)]
    return bool(evs) and min(evs, key=lambda e: e.t).kind == EV_DISARM


def arm_segments(events: list[LogEvent], t_start: float, t_end: float,
                 data_t: np.ndarray | None = None) -> list[TimeSegment]:
    """把 ARM / DISARM 事件配對成飛行段,裁在 [t_start, t_end] 內。

    - 第一個狀態事件是 DISARM → 沒記到 ARM(open_start):起點推定為 log 開頭;
      但若給了 data_t(所有來源的時間戳,已排序)且 DISARM 之前有 > 5 s 的資料空洞,
      起點改為最後一個空洞之後的第一筆資料。ArduPilot 4.1+ 在 LOG_DISARMED=0 時,
      ARM 前一次解鎖失敗會留下約 15 s 的未解鎖資料,之後真正 ARM 的事件卻寫不進 log。
    - 最後仍是 ARM → 飛行段延伸到 log 結尾(open_end,例如斷電、墜機)。
    - 連續重複的 ARM 或 DISARM(ARM 與 EV 重複記錄、或狀態重送)只取第一個。
    """
    evs = sorted((e for e in events if e.kind in (EV_ARM, EV_DISARM)), key=lambda e: e.t)
    raw: list[tuple[float, float, bool, bool]] = []
    armed_at: float | None = None
    open_start = False
    if evs and evs[0].kind == EV_DISARM:
        armed_at, open_start = t_start, True
        if data_t is not None:
            dt = np.asarray(data_t, dtype=float)
            dt = dt[dt < evs[0].t]
            gaps = np.flatnonzero(np.diff(dt) > _ARM_GAP_S)
            if gaps.size:
                armed_at = float(dt[gaps[-1] + 1])
    for e in evs:
        if e.kind == EV_ARM:
            if armed_at is None:
                armed_at, open_start = e.t, False
        elif armed_at is not None:
            raw.append((armed_at, e.t, open_start, False))
            armed_at, open_start = None, False
    if armed_at is not None:
        raw.append((armed_at, t_end, open_start, True))

    segs: list[TimeSegment] = []
    for a, b, os_, oe in raw:
        a, b = max(a, t_start), min(b, t_end)
        if b - a <= _EPS:
            continue
        segs.append(TimeSegment(a, b, f"ARM {len(segs) + 1}", open_start=os_, open_end=oe))
    return segs


def mode_changes(events: list[LogEvent]) -> list[LogEvent]:
    """依時間排序的模式切換事件(相鄰同名者合併)。"""
    out: list[LogEvent] = []
    for e in sorted((e for e in events if e.kind == EV_MODE), key=lambda e: e.t):
        if out and out[-1].label == e.label:
            continue
        out.append(e)
    return out


def mode_at(modes: list[LogEvent], t: float) -> str | None:
    """t 時刻的飛行模式;modes 需為 mode_changes() 的輸出。"""
    i = bisect_right([m.t for m in modes], t) - 1
    return modes[i].label if i >= 0 else None


def segment_index_at(segments: list[TimeSegment], t: float) -> int | None:
    for i, s in enumerate(segments):
        if s.contains(t):
            return i
    return None


def segment_matching(segments: list[TimeSegment], t0: float, t1: float,
                     tol: float = 0.05) -> int | None:
    """分析區間 [t0, t1] 若(在 tol 秒內)恰好等於某飛行段,回傳其索引。"""
    for i, s in enumerate(segments):
        if abs(s.t0 - t0) <= tol and abs(s.t1 - t1) <= tol:
            return i
    return None


def overlapping_segments(segments: list[TimeSegment], t0: float, t1: float) -> list[int]:
    return [i for i, s in enumerate(segments) if s.t0 < t1 and s.t1 > t0]


def boundaries(segments: list[TimeSegment], events: list[LogEvent] | None = None) -> list[float]:
    """可「跳到上/下一個」的時間點:飛行段起訖 + 事件時刻(已排序、去重)。"""
    pts = [x for s in segments for x in (s.t0, s.t1)]
    if events:
        pts += [e.t for e in events]
    pts.sort()
    out: list[float] = []
    for p in pts:
        if not out or p - out[-1] > 1e-3:
            out.append(p)
    return out


# ---------------- 軌跡逐時查詢 ----------------

class Track:
    """單一軌跡(時間遞增)的逐時內插,供播放游標使用。

    超出時間範圍、或落在大於 max_gap 秒的資料空洞內時回傳 None,
    避免在沒有資料的地方畫出假位置。
    """

    def __init__(self, t: np.ndarray, p: np.ndarray, max_gap: float = 1.0, edge: float = 0.0):
        self.t = np.asarray(t, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.max_gap = float(max_gap)
        self.edge = float(edge)     # 超出首末樣本 edge 秒內仍回傳首末值(區間邊緣不讓游標消失)

    @property
    def t0(self) -> float:
        return float(self.t[0]) if self.t.size else np.inf

    @property
    def t1(self) -> float:
        return float(self.t[-1]) if self.t.size else -np.inf

    def at(self, t: float) -> np.ndarray | None:
        n = self.t.size
        if n == 0 or not np.isfinite(t):
            return None
        if t < self.t[0] - _EPS:
            return self.p[0].copy() if t >= self.t[0] - self.edge else None
        if t > self.t[-1] + _EPS:
            return self.p[-1].copy() if t <= self.t[-1] + self.edge else None
        j = int(np.searchsorted(self.t, t, side="right"))
        if j <= 0:
            return self.p[0].copy()
        if j >= n:
            return self.p[-1].copy()
        ta, tb = self.t[j - 1], self.t[j]
        if tb - ta > self.max_gap:
            # 空洞內:只在貼近任一端樣本時才給值
            if t - ta <= self.max_gap / 2:
                return self.p[j - 1].copy()
            if tb - t <= self.max_gap / 2:
                return self.p[j].copy()
            return None
        w = (t - ta) / (tb - ta) if tb > ta else 0.0
        return self.p[j - 1] * (1.0 - w) + self.p[j] * w

    def index_at(self, t: float) -> int | None:
        """最接近 t 的樣本索引(超出範圍回傳 None)。"""
        n = self.t.size
        if n == 0 or t < self.t[0] - _EPS or t > self.t[-1] + _EPS:
            return None
        j = int(np.searchsorted(self.t, t))
        if j <= 0:
            return 0
        if j >= n:
            return n - 1
        return j if (self.t[j] - t) < (t - self.t[j - 1]) else j - 1

    def trail(self, t_from: float, t_to: float) -> np.ndarray:
        """[t_from, t_to] 內的樣本,尾端補上 t_to 的內插點(讓尾巴貼住游標)。Mx3。"""
        a = int(np.searchsorted(self.t, t_from, side="left"))
        b = int(np.searchsorted(self.t, t_to, side="right"))
        seg = self.p[a:b]
        cur = self.at(t_to)
        if cur is not None and (seg.shape[0] == 0 or self.t[b - 1] < t_to - _EPS):
            seg = np.vstack([seg, cur[None, :]]) if seg.size else cur[None, :]
        return seg


def horizontal_speed(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """水平速度(m/s),取樣於相鄰樣本中點。"""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if t.size < 2:
        return np.empty(0), np.empty(0)
    dt = np.diff(t)
    d = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    ok = dt > 1e-6
    return 0.5 * (t[1:] + t[:-1])[ok], d[ok] / dt[ok]


def binned(t: np.ndarray, v: np.ndarray, lo: float, hi: float, nbins: int,
           reducer: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """把時間序列壓成 nbins 格(時間軸迷你圖用);空格為 NaN。

    reducer: "mean" | "max" | "min"。回傳 (各格中心時間, 各格值)。
    """
    nbins = max(1, int(nbins))
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    out = np.full(nbins, np.nan)
    if t.size == 0 or hi <= lo:
        return centers, out
    m = (t >= lo) & (t <= hi) & np.isfinite(v)
    if not m.any():
        return centers, out
    idx = np.clip(((t[m] - lo) / (hi - lo) * nbins).astype(int), 0, nbins - 1)
    vv = v[m]
    if reducer == "mean":
        s = np.bincount(idx, weights=vv, minlength=nbins)
        c = np.bincount(idx, minlength=nbins)
        ok = c > 0
        out[ok] = s[ok] / c[ok]
    elif reducer in ("max", "min"):
        fill = -np.inf if reducer == "max" else np.inf
        acc = np.full(nbins, fill)
        (np.maximum if reducer == "max" else np.minimum).at(acc, idx, vv)
        ok = np.isfinite(acc)
        out[ok] = acc[ok]
    else:
        raise ValueError(f"未知 reducer: {reducer}")
    return centers, out


# ---------------- 播放時鐘 ----------------

class PlaybackClock:
    """與 GUI 無關的播放狀態:播放頭、速度、可拖曳範圍與播放範圍。

    - bounds(lo, hi):播放頭可到的範圍(整份 log)。
    - play range:按播放時實際走的範圍(可設成分析區間);播放頭在範圍外按播放
      會先跳到範圍起點,走到終點自動停。
    """

    def __init__(self, lo: float = 0.0, hi: float = 0.0):
        self.lo, self.hi = float(lo), float(max(lo, hi))
        self.play_lo, self.play_hi = self.lo, self.hi
        self.t = self.lo
        self.speed = 1.0
        self.playing = False

    def set_bounds(self, lo: float, hi: float) -> None:
        self.lo, self.hi = float(lo), float(max(lo, hi))
        self.set_play_range(self.play_lo, self.play_hi)
        self.t = self._clamp(self.t)

    def set_play_range(self, lo: float, hi: float) -> None:
        lo, hi = sorted((float(lo), float(hi)))
        self.play_lo = min(max(lo, self.lo), self.hi)
        self.play_hi = min(max(hi, self.lo), self.hi)

    def _clamp(self, t: float) -> float:
        return min(max(float(t), self.lo), self.hi)

    def seek(self, t: float) -> float:
        self.t = self._clamp(t)
        return self.t

    def step(self, dt: float) -> float:
        return self.seek(self.t + dt)

    def play(self) -> None:
        if self.play_hi - self.play_lo <= _EPS:
            self.playing = False
            return
        if self.t < self.play_lo - _EPS or self.t >= self.play_hi - _EPS:
            self.t = self.play_lo
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle(self) -> bool:
        self.pause() if self.playing else self.play()
        return self.playing

    def advance(self, wall_dt: float) -> float:
        """經過 wall_dt 秒(真實時間)後的播放頭;到播放範圍終點自動停。"""
        if not self.playing:
            return self.t
        self.t += max(0.0, float(wall_dt)) * self.speed
        if self.t >= self.play_hi:
            self.t = self.play_hi
            self.playing = False
        return self.t


def format_clock(sec: float, decimals: int = 1) -> str:
    """秒 → 「m:ss.s」;超過一小時為「h:mm:ss」。"""
    if not np.isfinite(sec):
        return "—"
    sign = "-" if sec < 0 else ""
    sec = abs(float(sec))
    if sec >= 3600:
        h = int(sec // 3600)
        m = int(sec % 3600 // 60)
        s = int(sec % 60)
        return f"{sign}{h}:{m:02d}:{s:02d}"
    m = int(sec // 60)
    s = sec - m * 60
    if decimals <= 0:
        return f"{sign}{m}:{int(s):02d}"
    w = 3 + decimals
    txt = f"{s:0{w}.{decimals}f}"
    if txt.startswith("60"):          # 四捨五入進位
        m += 1
        txt = f"{0.0:0{w}.{decimals}f}"
    return f"{sign}{m}:{txt}"
