"""PID 調參分析的資料模型(core.tuning 產生、ui.tuning 顯示、cli 列印)。

不依賴 Qt。所有時間為 log 絕對秒(TimeUS/1e6);角度 deg、角速度 deg/s。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 嚴重度(由輕到重)
SEV_OK = "ok"          # 良好
SEV_INFO = "info"      # 資訊 / 可改善
SEV_WARN = "warn"      # 注意:建議調整
SEV_BAD = "bad"        # 明顯問題:先處理
SEV_ORDER = {SEV_OK: 0, SEV_INFO: 1, SEV_WARN: 2, SEV_BAD: 3}
SEV_LABEL = {SEV_OK: "良好", SEV_INFO: "可改善", SEV_WARN: "建議調整", SEV_BAD: "需要處理"}

AXES = ("roll", "pitch", "yaw")
AXIS_LABEL = {"roll": "Roll(橫滾)", "pitch": "Pitch(俯仰)", "yaw": "Yaw(偏航)"}
AXIS_PARAM = {"roll": "RLL", "pitch": "PIT", "yaw": "YAW"}   # ATC_RAT_<RLL|PIT|YAW>_P

# 建議方向
DIR_UP = "increase"
DIR_DOWN = "decrease"
DIR_SET = "set"        # 設成某個值(例如 LOG_BITMASK、INS_HNTCH_ENABLE)
DIR_CHECK = "check"    # 檢查(機構、重心、螺旋槳…),不是改參數


@dataclass
class Suggestion:
    """一條具體的參數建議。"""
    param: str                       # 例:"ATC_RAT_RLL_P";DIR_CHECK 時可為描述性名稱
    direction: str                   # DIR_UP / DIR_DOWN / DIR_SET / DIR_CHECK
    reason: str                      # 為什麼(zh-TW,一兩句)
    current: float | None = None     # 目前值(PARM;None = log 沒記)
    proposed: float | None = None    # 建議值(已夾在合理範圍、適度四捨五入)
    change_pct: float | None = None  # 建議變動百分比(+20 / -25)
    priority: int = 2                # 1 = 先改這個;數字越大越後面


@dataclass
class Finding:
    """一個診斷結果(症狀 + 證據 + 對應的調整方向)。"""
    code: str                        # 穩定代碼,例:"rate_osc_mid"、"angle_lag"、"bias"、"saturation"
    severity: str                    # SEV_*
    axis: str | None                 # "roll" / "pitch" / "yaw";None = 整機
    title: str                       # 一行標題(zh-TW)
    detail: str                      # 說明 + 證據數字(zh-TW)
    suggestions: list[Suggestion] = field(default_factory=list)
    t_focus: float | None = None     # 最能看出問題的時刻(log 絕對秒;UI 點了跳過去)
    window: tuple[float, float] | None = None   # 相關時間範圍
    metrics: dict = field(default_factory=dict)


@dataclass
class Spectrum:
    """頻譜:誤差與實際值的功率譜密度,以及偵測到的峰值。"""
    f: np.ndarray                    # Hz
    psd_err: np.ndarray              # (deg/s)²/Hz(或角度層級時 deg²/Hz)
    psd_act: np.ndarray
    peaks: list[tuple[float, float]] = field(default_factory=list)   # (頻率 Hz, 該峰 RMS 振幅)
    nyquist: float = 0.0


@dataclass
class FreqResponse:
    """期望 → 實際的頻率響應估計(Welch 交叉譜)。"""
    f: np.ndarray
    mag: np.ndarray                  # |H|(1 = 完全跟上)
    phase_deg: np.ndarray
    coherence: np.ndarray
    bandwidth_hz: float | None = None   # |H| 掉到 -3 dB(0.707)的頻率
    peak_mag: float | None = None       # 共振峰 |H| 最大值(> 1 表示會過衝)
    peak_hz: float | None = None


@dataclass
class StepResponse:
    """由期望 / 實際反卷積估出的步階響應(多個時間窗平均)。"""
    t: np.ndarray                    # 秒(從 0 起)
    y: np.ndarray                    # 正規化步階響應(穩態 ≈ 1)
    overshoot_pct: float | None = None
    rise_s: float | None = None      # 10 % → 90 %
    settle_s: float | None = None    # 進入 ±5 % 不再出來
    n_windows: int = 0


@dataclass
class AxisAnalysis:
    """單一軸的分析結果。"""
    axis: str
    level: str                       # 使用的訊號層級:"rate"(RATE / PIDx)或 "angle"(ATT)
    sample_hz: float                 # 該層級資料的取樣率
    t: np.ndarray                    # 角速度層級:時間(飛行段以外為 NaN 斷開)
    des: np.ndarray                  # 期望角速度 deg/s
    act: np.ndarray                  # 實際角速度 deg/s
    t_ang: np.ndarray | None = None  # 角度層級(ATT):時間
    des_ang: np.ndarray | None = None
    act_ang: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)   # 例:rms_err, max_err, bias, lag_ms, tracking_pct
    spectrum: Spectrum | None = None
    freq: FreqResponse | None = None
    step: StepResponse | None = None
    verdict: str = SEV_OK            # 這一軸最嚴重的 finding
    summary: str = ""                # 一句話結論(zh-TW)


@dataclass
class TuningReport:
    """整份分析報告。"""
    t0: float                        # 分析範圍(log 絕對秒)
    t1: float
    flight_s: float                  # 範圍內實際納入分析的飛行秒數
    vehicle: str                     # "copter" / "plane" / …
    frame: str                       # 例:"hexa X"
    param_prefix: str                # "ATC_"(Copter)或 "Q_A_"(QuadPlane)
    data: dict                       # 資料品質,例:{"RATE_hz": 10.0, "ATT_hz": 10.0, "PID": False, "fast_attitude": False}
    axes: dict[str, AxisAnalysis]
    findings: list[Finding]          # 依嚴重度(重→輕)、優先順序排序
    params: dict[str, float]         # 相關參數目前值
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return max((f.severity for f in self.findings), key=SEV_ORDER.get, default=SEV_OK)

    def suggestions(self) -> list[Suggestion]:
        """去重後的建議(同一參數只留優先順序最高、其次嚴重度最高的那條)。"""
        best: dict[tuple[str, str], tuple[tuple, Suggestion]] = {}
        for f in self.findings:
            for s in f.suggestions:
                key = (s.param, s.direction if s.direction == DIR_CHECK else "param")
                rank = (s.priority, -SEV_ORDER.get(f.severity, 0))
                if key not in best or rank < best[key][0]:
                    best[key] = (rank, s)
        return [s for _, s in sorted(best.values(), key=lambda kv: kv[0])]

    def param_file(self) -> str:
        """只含「有建議值」的參數,Mission Planner 相容的 .param 內容。"""
        lines = ["# vio_ate_studio PID 調參建議 — 僅供參考:一次只改一項、小幅修改、在安全環境試飛"]
        for s in self.suggestions():
            if s.direction in (DIR_UP, DIR_DOWN, DIR_SET) and s.proposed is not None:
                lines.append(f"{s.param},{s.proposed:g}")
        return "\n".join(lines) + "\n"
