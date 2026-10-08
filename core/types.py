"""資料模型:軌跡來源與載入結果。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import frames

KIND_GLOBAL = "global"   # 經緯高 (GPS / POS / AHR2 ...)
KIND_NED = "ned"         # 局部 NED (VISP / XKF1 ...,相對 EKF 原點)

KIND_LABEL = {KIND_GLOBAL: "經緯度", KIND_NED: "局部 NED"}

# 時間軸事件種類
EV_ARM = "arm"           # 解鎖(ARM / EV 10)
EV_DISARM = "disarm"     # 上鎖(ARM / EV 11)
EV_MODE = "mode"         # 飛行模式切換(MODE),label = 模式名稱
EV_RESET = "reset"       # VIO 重置(VISP.ResetCnt 改變),label = 來源名稱
EV_EVENT = "ev"          # 其他 EV 事件(EKF 來源切換、降落完成…),label = 事件名稱
EV_ERROR = "err"         # ERR 錯誤 / 失效保護,label = 「子系統 代碼」說明


@dataclass
class LogEvent:
    """log 中的離散事件(時間為 log 時基秒)。"""
    t: float
    kind: str
    label: str = ""


@dataclass
class TimeSegment:
    """一段連續時間區間 — 目前用於「一次 ARM 到 DISARM」的飛行段。"""
    t0: float
    t1: float
    label: str
    open_start: bool = False   # 沒記到 ARM 事件、起點是推定的(log 開頭,或開頭那段資料空洞之後)
    open_end: bool = False     # 到 log 結尾都沒有 DISARM

    @property
    def duration(self) -> float:
        return max(0.0, self.t1 - self.t0)

    def contains(self, t: float) -> bool:
        return self.t0 <= t <= self.t1


@dataclass
class TrajSource:
    """單一軌跡來源(已按時間排序、去除無效樣本)。"""
    name: str
    kind: str
    t: np.ndarray                      # 秒(log 時基,TimeUS / 1e6)
    lat: np.ndarray | None = None      # kind == global
    lng: np.ndarray | None = None
    alt: np.ndarray | None = None
    ned: np.ndarray | None = None      # kind == ned, Nx3 (N, E, D) 公尺
    meta: str = ""

    @property
    def n(self) -> int:
        return int(len(self.t))

    @property
    def rate_hz(self) -> float:
        if self.n < 2:
            return 0.0
        dt = float(np.median(np.diff(self.t)))
        return 1.0 / dt if dt > 1e-9 else 0.0

    def enu(self, origin: tuple[float, float, float] | None) -> np.ndarray:
        """轉為局部 ENU (Nx3)。global 來源需要 origin。"""
        if self.kind == KIND_GLOBAL:
            if origin is None:
                raise ValueError(f"來源 {self.name} 為經緯度,需要 ENU 原點(log 無 ORGN 訊息)")
            return frames.geodetic_to_enu(self.lat, self.lng, self.alt, origin)
        return frames.ned_to_enu(self.ned)


@dataclass
class LogData:
    """一份 log(或模擬資料)解析後的全部軌跡來源。"""
    sources: dict[str, TrajSource]
    origin: tuple[float, float, float] | None = None   # EKF 原點 (lat, lng, alt)
    origin_desc: str = ""
    path: str = ""
    notes: list[str] = field(default_factory=list)
    events: list[LogEvent] = field(default_factory=list)   # 依時間排序
    index: object | None = None   # core.dflog.LogIndex:任意訊息 / 欄位的查詢(繪圖、參數、訊息面板)

    @property
    def segments(self) -> list[TimeSegment]:
        """ARM→DISARM 飛行段(一份「開機到關機」的 log 可能有好幾段)。"""
        from .timeline import arm_segments, first_state_is_disarm
        data_t = None
        if first_state_is_disarm(self.events):     # 少了第一個 ARM:要靠資料空洞推定起點
            ts = [s.t for s in self.sources.values() if s.n]
            data_t = np.sort(np.concatenate(ts)) if ts else None
        return arm_segments(self.events, self.t_base, self.t_last, data_t)

    @property
    def t_base(self) -> float:
        return min((float(s.t[0]) for s in self.sources.values() if s.n), default=0.0)

    @property
    def t_end(self) -> float:
        return max((float(s.t[-1]) for s in self.sources.values() if s.n), default=0.0)

    @property
    def t_last(self) -> float:
        """時間軸 / 播放的終點:位置來源、事件與 log 其他訊息(MSG / ERR / PARM…)的最後時刻。

        LOG_DISARMED=0 時位置資料在 DISARM 後就停,但文字訊息、ERR 還會繼續記;
        t_end 只看位置來源(ATE 用),時間軸要涵蓋到這裡才看得到、跳得到那些事件。
        """
        hi = self.t_end
        if self.events:
            hi = max(hi, max(e.t for e in self.events))
        if self.index is not None:
            # 只看文字訊息 / 事件(不看全部訊息的 t_range):單筆壞掉的 TimeUS 不該把時間軸拉長
            try:
                msgs = self.index.log_messages()
                if msgs:
                    hi = max(hi, max(m.t for m in msgs if np.isfinite(m.t)))
            except Exception:  # noqa: BLE001 — 索引壞掉就只用位置與事件
                pass
        return hi

    @property
    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_base)
