"""失事原因分析的共用資料型別。

時間一律是「相對秒」(= log 絕對秒 − ld.t_base),和 UI 顯示的相對 log 起點時間一致;
要轉回絕對秒:abs = rel + ctx.t_base。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 偵測結果狀態
HIT = "HIT"        # 條件成立(證據)
SOFT = "SOFT"      # 弱證據(接近門檻 / 較寬鬆的判準成立)
CLEAR = "CLEAR"    # 資料足夠、確認沒有這個問題(可用來「排除」假說)
NA = "NA"          # 不適用 / 資料不足(不能拿來排除,也不計分)

STATUSES = (HIT, SOFT, CLEAR, NA)
SEVERITIES = ("critical", "high", "medium", "info")

# 事件因果標籤(由 engine 依 t_onset 對照 t_warn / t_loss 貼上)
TAG_PRECURSOR = "precursor"          # 先於姿態異常:可能是原因
TAG_DEVELOPMENT = "development"      # 姿態異常發展期間
TAG_CONSEQUENCE = "consequence"      # 姿態失控之後:後果,不計分
TAG_POST = "post_crash"              # 上鎖 / 撞擊之後
TAG_GROUND = "ground_info"           # 未解鎖(地面)事件:只顯示資訊
TAG_STANDING = "standing"            # 持續性狀態(設定、長期統計):不看時間先後


@dataclass
class Finding:
    """一個偵測器的一項結論。"""
    id: str                              # 偵測器 id(snake_case,見設計目錄)
    domain: str                          # end_state / attitude / ground / motor / power / vibration / nav / compass / failsafe / infra
    status: str                          # HIT / SOFT / CLEAR / NA
    severity: str = "info"
    t_onset: float | None = None         # 條件開始成立的時間(相對秒);因果判斷用這個
    t_alarm: float | None = None         # 條件被確認的時間(相對秒;持續 N 秒才成立的偵測器 > t_onset)
    context: str = ""                    # 判讀情境(例如 'air' / 'ground_idle' / 'near_touchdown')
    values: dict = field(default_factory=dict)   # 量測數值(JSON 可序列化:float / int / str / list)
    text_zh: str = ""                    # 給使用者看的繁體中文說明(含關鍵數字)
    tag: str = ""                        # engine 事後貼的因果標籤

    @property
    def is_evidence(self) -> bool:
        return self.status in (HIT, SOFT)

    def level(self) -> float:
        return {HIT: 1.0, SOFT: 0.5}.get(self.status, 0.0)


@dataclass
class ArmedInterval:
    """一次解鎖到上鎖。"""
    index: int
    t0: float                            # 解鎖時間(相對秒)
    t1: float                            # 上鎖時間(相對秒);open_end 時為 log 最後一筆資料的時間
    method: int | None = None            # ARM.Method(上鎖原因;12 = CRASH_CHECK)
    forced: bool = False
    open_end: bool = False               # log 結束時還在解鎖狀態

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass
class Anchors:
    """分析用的關鍵時間點(相對秒)與分析區間;由各模組依序填入。

    填入者:  armed_gate / engine → interval, end_type(初判)、t_detect、t_disarm
             end_state 模組      → end_type(定案)、t_impact_peak、t_first_hit、t_rest、impacts
             attitude 模組       → t_warn、t_loss、t_inverted
             ground 模組         → t_touch
             engine              → t_first_anomaly、scenario
    """
    interval: ArmedInterval | None = None
    intervals: list[ArmedInterval] = field(default_factory=list)
    end_type: str = "unknown"            # not_armed / normal_landing / crash_check / crash_check_replay /
                                         # abnormal_disarm / hard_impact / log_ended_armed / tumble / normal_end
    t_detect: float | None = None        # 飛控 / 偵測器「判定失事」的時間(例:CRASH_CHECK ERR)
    t_disarm: float | None = None
    t_impact_peak: float | None = None   # 最後一次硬撞擊的峰值時間
    t_first_hit: float | None = None     # 第一次硬撞擊
    t_rest: float | None = None          # 機身靜止(最後一次 |a| > 15 m/s²)的時間
    t_warn: float | None = None          # 姿態誤差 > 10° 開始
    t_loss: float | None = None          # 姿態誤差 > 30° 開始
    t_inverted: float | None = None
    t_touch: float | None = None         # 觸地(ground_contact)時間
    t_first_anomaly: float | None = None
    scenario: str = ""
    impacts: list[dict] = field(default_factory=list)   # {t_start, t_peak, t_end, peak, dV, cls}
    notes: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def t_ref(self) -> float | None:
        """「事件起點」參考時間:t_warn,沒有就 t_loss / t_first_hit / t_detect。"""
        for v in (self.t_warn, self.t_loss, self.t_first_hit, self.t_detect, self.t_disarm):
            if v is not None:
                return v
        return None
