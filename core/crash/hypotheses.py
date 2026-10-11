"""根因假說評分:Finding(偵測結果)→ 各假說分數 → 排名 / 貢獻因素 / 已排除 / 證據不足。

規則(確定性、可解釋,不是機率):
  S_h = Σ  W[h][detector] × level(HIT 1.0 / SOFT 0.5) × prox(因果標籤、離 t_warn 的時間)
  prox:precursor(t_onset < t_warn)距 t_warn ≤ 60 s → 1.0、≤ 300 s → 0.6、更早 0.3;
        development(t_warn ≤ t < t_loss)→ 0.8;consequence / post_crash / ground_info → 0;
        standing(持續性狀態)→ 固定 0.6
  每個假說有「必要證據」閘門:不滿足 → 分數 × 0.3 並標記。
判定:第一名信心 高 / 中 / 低;分數 ≥ max(S_MIN, 0.35·S1) 的其他假說 = 貢獻因素(信心最多「中」:吻合但未證實);
證據只來自沒有實測樣本的結束分類(空中上鎖 / 失效保護上鎖 / log 中斷 / 緊急停止)→ 信心封頂為「低」;
證據不足 / 無法區分 / 資料不足 → 不硬給原因:有分數的假說一律是 candidate(候選),沒有第一名與貢獻因素;
主要偵測器 CLEAR(或證據只在失控之後 / 情境不符)→ 已排除,附排除依據與排除信心(資料品質受限時由 engine 封頂為中)。
必要證據閘門遇到缺資料(例如觸地沒有 agl / 姿態誤差)→ 視為「未驗證」不通過,不預設成通過。
Weight.when 依各偵測器實際輸出的 context / values 篩選(例:lean_without_pilot 的 'ground_idle';vibe_clip_transient 的
'sustained_vibration' / 'vibration_clipping' / 'clip_increment' / 'impact_transient')。
權重只用一次真實失事(00000124)校準,報告永遠附這個免責聲明。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .types import (CLEAR, HIT, NA, SOFT, TAG_CONSEQUENCE, TAG_DEVELOPMENT, TAG_GROUND, TAG_POST,
                    TAG_PRECURSOR, TAG_STANDING, Anchors, Finding)

S_MIN = 3.0                  # 最佳假說低於此分數 → 證據不足
GATE_PENALTY = 0.3
CONTRIB_SHARE = 0.35         # 非第一名但分數 ≥ max(S_MIN, 0.35·S1) → 貢獻因素
AMBIG_RATIO = 0.8            # S2 ≥ 0.8·S1 且 S1−S2 < 1.5 → 無法區分
AMBIG_MARGIN = 1.5
HIGH_SCORE = 8.0             # 信心「高」:S1 ≥ 8、占比 ≥ 0.5、領先 ≥ 3、必要證據成立、≥ 2 個領域
HIGH_SHARE = 0.5
HIGH_MARGIN = 3.0
MID_SHARE = 0.35
NEAR_GROUND_AGL_M = 1.5      # 觸地翻覆的必要證據:觸地時 agl ≤ 1.5 m(124:-0.13 m)
HARD_VZ_CMS = -100.0         # 觸地前 5 s 下降中位數 < −100 cm/s → 重落地,不是觸地翻覆(124:−30 cm/s)
TOUCH_AE_MAX_DEG = 10.0      # 觸地當下姿態誤差需 < 10°(124:觸地時姿態正常)
DISCLAIMER = ("門檻與權重只用一次真實失事(00000124)校準,是可解釋的經驗規則,不是機率;"
              "請搭配時間軸與證據圖人工確認。")


def prox(f: Finding, a: Anchors) -> float:
    if f.tag in (TAG_CONSEQUENCE, TAG_POST, TAG_GROUND):
        return 0.0
    if f.tag == TAG_STANDING:
        return 0.6
    if f.tag == TAG_DEVELOPMENT:
        return 0.8
    if f.tag == TAG_PRECURSOR:
        ref = a.t_ref()
        if ref is None or f.t_onset is None:
            return 0.6
        lead = ref - f.t_onset
        return 1.0 if lead <= 60 else 0.6 if lead <= 300 else 0.3
    return 0.0


@dataclass
class Weight:
    det: str
    w: float
    when: Callable[[Finding, "Ctx2"], bool] | None = None     # 額外條件(例:context 篩選)
    only_precursor: bool = False                              # 觸發型證據:只有 precursor 標籤才算
    note: str = ""


@dataclass
class Hypothesis:
    id: str
    title_zh: str
    weights: list[Weight]
    primary: list[str]                    # 全部 CLEAR(且資料有效)→ 已排除
    advice_zh: str
    gate: Callable[["Ctx2"], tuple[bool, str]] | None = None
    bonus: Callable[["Ctx2"], list[tuple[str, float, str]]] | None = None   # 非偵測器證據:(標籤, 權重, 說明)


def _bonus_flip(c: "Ctx2") -> list[tuple[str, float, str]]:
    """失控起點在 ACRO / SPORT / FLIP 模式或 FLIP 事件期間 → 刻意動作(權重 4)。"""
    if c.a.scenario == "COMMANDED_FLIP":
        return [("scenario:COMMANDED_FLIP", 4.0, "姿態異常起點在特技 / 翻滾模式(ACRO / SPORT / FLIP)或 FLIP 事件期間,屬刻意動作。")]
    return []


@dataclass
class Ctx2:
    """評分用的查詢介面(findings 索引 + 錨點)。"""
    findings: list[Finding]
    a: Anchors
    by_id: dict[str, list[Finding]] = field(default_factory=dict)

    def __post_init__(self):
        for f in self.findings:
            self.by_id.setdefault(f.id, []).append(f)

    def get(self, det: str) -> list[Finding]:
        return self.by_id.get(det, [])

    def best(self, det: str) -> Finding | None:
        """同 id 多筆時取證據最強的一筆。"""
        fs = self.get(det)
        return max(fs, key=lambda f: (f.level(), -(f.t_onset if f.t_onset is not None else 0.0)), default=None)

    def hit(self, det: str, **kw) -> bool:
        return any(f.status == HIT and all(f.context == v for v in kw.values()) for f in self.get(det))

    def hit_causal(self, det: str) -> bool:
        """HIT 且可能是原因(先兆 / 發展 / 持續狀態);失控之後才出現的後果不算。"""
        return any(f.status == HIT and f.tag in (TAG_PRECURSOR, TAG_DEVELOPMENT, TAG_STANDING) for f in self.get(det))

    def status(self, det: str) -> str:
        f = self.best(det)
        return f.status if f is not None else NA


def _ctx_in(*contexts: str):
    return lambda f, c: (not contexts) or f.context in contexts


def _num(v, default: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if x == x else default


def _cls(f: Finding) -> str:
    """abnormal_disarm_classifier 的分類(values['class'];緊急停止那筆只有 context)。"""
    return str((f.values or {}).get("class") or f.context or "")


def _fs_trigger(f: Finding, subsys: frozenset = frozenset(), rsn: frozenset = frozenset()) -> bool:
    """inflight_failsafe_or_mode_change 的觸發清單裡有指定的 ERR 子系統 / 模式切換原因。"""
    for tr in f.values.get("triggers") or []:
        if not isinstance(tr, dict):
            continue
        if subsys and any(int(_num(s, -1)) in subsys for s in (tr.get("subsys") or [])):
            return True
        if rsn and tr.get("rsn") is not None and int(_num(tr.get("rsn"), -1)) in rsn:
            return True
    return False


def _opt(v) -> float | None:
    """數值或 None(缺值 / NaN / 非數字)。閘門用:缺值 = 未驗證,不能預設成「通過」。"""
    x = _num(v, float("nan"))
    return None if x != x else x


def _attitude_ok_at(c: "Ctx2", t: float | None) -> bool:
    """t 時姿態誤差 < 10° 能否由姿態模組證實:attitude_loss_of_control 有有效結果(非 NA)且 t 早於 t_warn
    (t_warn = ae > 10° 持續 0.3 s 的起點;撞擊在先時姿態模組把 t_warn 設成第一次硬撞擊)。"""
    if t is None:
        return False
    if not any(f.status != NA for f in c.get("attitude_loss_of_control")):
        return False
    return c.a.t_warn is None or t < c.a.t_warn


def _lean_near_ground(f: Finding, c: "Ctx2") -> bool:
    """lean_without_pilot:地面怠速(context ground_idle),或空中但離地 ≤ 1.5 m。"""
    if f.context in ("ground_idle", "ground"):
        return True
    return f.context in ("air", "air_low") and _num(f.values.get("agl_at_onset_m"), 1e9) <= NEAR_GROUND_AGL_M


_VIBE_SUBSYS = frozenset({29})            # ERR 子系統 29 = FAILSAFE_VIBE
_VIB_CLIP_CONTEXTS = ("sustained_vibration", "vibration_clipping", "clip_increment")

# ------------------------------------------------------------------ 必要證據閘門

def _gate_land(c: Ctx2):
    f = c.best("touchdown_not_recognised")
    if f is None or f.status != HIT:
        return False, "未偵測到「觸地後未判定落地」"
    if c.hit_causal("motor_thrust_loss_check"):
        return False, "同時有推力損失證據,不能單獨歸因於觸地後翻覆"
    v = f.values or {}
    # 觸地偵測沒附數字時,改用 engine 的型態特徵(同一個觸地時間取樣);兩邊都沒有 = 必要證據未驗證(不預設通過)
    feats = c.a.extra.get("scenario", {}) if isinstance(c.a.extra, dict) else {}
    same_t = (f.t_onset is not None and feats.get("t_feat") is not None and abs(feats["t_feat"] - f.t_onset) <= 0.5)
    agl = _opt(v.get("agl_m"))
    if agl is None and same_t:
        agl = _opt(feats.get("agl_m"))
    vz = _opt(v.get("vz_pre_cms"))
    if vz is None and same_t:
        vz = _opt(feats.get("vz_pre_cms"))
    if agl is None:
        return False, "必要證據未驗證:沒有觸地當下的離地高度(agl)"
    if agl > NEAR_GROUND_AGL_M:
        return False, f"觸地時離地高度 {agl:.1f} m(> {NEAR_GROUND_AGL_M} m),不是貼地翻覆"
    if vz is None:
        return False, "必要證據未驗證:沒有觸地前 5 s 的垂直速度"
    if vz < HARD_VZ_CMS:
        return False, f"觸地前下降 {vz:.0f} cm/s(< {HARD_VZ_CMS:.0f}),屬重落地而非觸地後翻覆"
    ae = _opt(v.get("ae_deg"))
    if ae is None and not _attitude_ok_at(c, f.t_onset):
        return False, "必要證據未驗證:沒有觸地當下的姿態誤差(姿態資料不足或觸地晚於姿態預警)"
    if ae is not None and ae >= TOUCH_AE_MAX_DEG:
        return False, f"觸地當下姿態誤差已 {ae:.0f}°(≥ {TOUCH_AE_MAX_DEG:.0f}°),觸地前就已異常"
    if c.a.t_loss is not None and f.t_onset is not None and f.t_onset >= c.a.t_loss:
        return False, "觸地發生在姿態失控之後(是結果,不是起點)"
    return True, ""


def _gate_nav(c: Ctx2):
    for det in ("ekf_testratio_gate", "source_switch_audit", "ekf_aiding_timeout",
                "ekf_phantom_motion_on_ground", "ekf_attitude_consistency"):
        for f in c.get(det):
            if f.status == HIT and f.tag in (TAG_PRECURSOR, TAG_DEVELOPMENT):
                return True, ""
    return False, "沒有「先於姿態失控」的非持續性導航異常(只有持續性狀態不足以當根因)"


def _gate_motor(c: Ctx2):
    a = c.a
    lim = (a.t_warn + 1.0) if a.t_warn is not None else None
    if c.hit_causal("motor_thrust_loss_check") or c.hit_causal("yaw_imbalance_iterm"):
        return True, ""
    imb = [f for f in c.get("motor_thrust_imbalance") if f.status == HIT and f.context == "air"]
    sat = [f for f in c.get("rate_output_saturation") if f.status == HIT and f.context == "air"]
    if imb and sat and all(lim is None or (f.t_onset is not None and f.t_onset <= lim) for f in imb + sat):
        return True, ""
    return False, "沒有空中的推力損失 / 偏航失衡 / (推力失衡 + 輸出飽和)證據(地面怠速的症狀不算)"


def _gate_power(c: Ctx2):
    for det in ("power_voltage_collapse", "power_low_voltage", "power_rail_fault"):
        for f in c.get(det):
            if f.status == HIT and f.tag in (TAG_PRECURSOR, TAG_DEVELOPMENT, TAG_STANDING):
                return True, ""
    # log 在解鎖中結束:必須同時有最後 2 s 的電壓崩落,才算電源證據(否則也可能是 SD 卡 / 記錄中斷)
    for f in c.get("abnormal_disarm_classifier"):
        if f.status == HIT and _cls(f) == "log_ended_armed" and f.values.get("volt_collapsed"):
            return True, ""
    return False, "沒有先於姿態失控的電壓 / 電源異常"


def _gate_vib(c: Ctx2):
    f = c.best("vibration_flight_level")
    if f is not None and f.status == HIT and f.tag not in (TAG_CONSEQUENCE, TAG_POST, TAG_GROUND):
        return True, ""
    for g in c.get("imu_consistency_health"):
        if g.status == HIT and g.tag in (TAG_PRECURSOR, TAG_STANDING):
            return True, ""
    for g in c.get("vibe_clip_transient"):
        if g.status == HIT and g.context in _VIB_CLIP_CONTEXTS and g.tag in (TAG_PRECURSOR, TAG_DEVELOPMENT):
            return True, ""
    for g in c.get("inflight_failsafe_or_mode_change"):
        if g.status == HIT and g.tag in (TAG_PRECURSOR, TAG_DEVELOPMENT) and _fs_trigger(g, subsys=_VIBE_SUBSYS):
            return True, ""
    return False, "撞擊之前沒有振動 / IMU 異常(撞擊後的削波是後果)"


def _gate_compass(c: Ctx2):
    a = c.a
    for det in ("ekf_mag_reject", "compass_raw_consistency", "ekf_yaw_reset_lane_switch",
                "compass_sensor_health", "ekf_attitude_consistency"):
        for f in c.get(det):
            if f.status == HIT and f.tag == TAG_PRECURSOR and (
                    a.t_warn is None or f.t_onset is None or f.t_onset <= a.t_warn - 1.0):
                return True, ""
    return False, "羅盤 / 航向異常出現得比姿態失控晚(是結果,不是原因)或根本沒有"


def _gate_pilot(c: Ctx2):
    for f in c.get("pilot_input_context"):
        if f.status == HIT and f.tag == TAG_PRECURSOR:
            return True, ""
    for f in c.get("abnormal_disarm_classifier"):
        if f.status == HIT and _cls(f) == "manual_disarm_in_air" and f.tag in (TAG_PRECURSOR, TAG_DEVELOPMENT):
            return True, ""
    if c.a.scenario == "COMMANDED_FLIP":
        return True, ""
    return False, "沒有先於異常的飛手操作 / 刻意動作(失控後的搖桿是反應)"


def _gate_failsafe(c: Ctx2):
    for f in c.get("inflight_failsafe_or_mode_change"):
        if f.status == HIT and f.tag in (TAG_PRECURSOR, TAG_DEVELOPMENT):
            return True, ""
    for f in c.get("abnormal_disarm_classifier"):
        if f.status == HIT and _cls(f) == "failsafe_disarm" and f.tag in (TAG_PRECURSOR, TAG_DEVELOPMENT):
            return True, ""
    return False, "沒有先於姿態失控的失效保護 / 模式切換(之後或上鎖後的是後果)"


IMPACT_AE_MAX_DEG = 10.0     # 外力撞擊的必要證據:撞擊當下姿態誤差 < 10°(撞擊前姿態原本正常)


def _hard_impact(f: Finding) -> bool:
    """impact_shock_clusters 的硬撞擊群(cls hard_impact;沒有 IMU 時的振動備援 vibe_fallback)。觸地(ground_contact)不算。"""
    return (f.values or {}).get("cls") == "hard_impact" or f.context == "vibe_fallback"


def _gate_impact(c: Ctx2):
    unverified = False
    for f in c.get("impact_shock_clusters"):
        if f.status != HIT or f.tag != TAG_PRECURSOR or not _hard_impact(f):
            continue
        v = f.values or {}
        ae = _opt(v.get("ae_at_impact", v.get("ae_deg")))
        if ae is not None:
            if ae < IMPACT_AE_MAX_DEG:
                return True, ""
        elif _attitude_ok_at(c, f.t_onset):
            return True, ""
        else:
            unverified = True
    if unverified:
        return False, "必要證據未驗證:有先於姿態異常的硬撞擊,但沒有撞擊當下的姿態誤差資料"
    return False, "沒有「姿態正常時」的硬撞擊(硬撞擊在姿態失控之後是後果;一般觸地不算)"


def _no_motor_nav_evidence(f: Finding, c: Ctx2) -> bool:
    """rate_output_saturation 只在「沒有任何馬達 / 導航證據」時才算飛手操作的旁證。"""
    for det in ("motor_thrust_loss_check", "yaw_imbalance_iterm", "ekf_testratio_gate", "ekf_aiding_timeout",
                "ekf_phantom_motion_on_ground", "source_switch_audit"):
        if any(g.status == HIT and g.tag in (TAG_PRECURSOR, TAG_DEVELOPMENT) for g in c.get(det)):
            return False
    return not any(g.status == HIT and g.context == "air" for g in c.get("motor_thrust_imbalance"))


HYPOTHESES: list[Hypothesis] = [
    Hypothesis("H_LAND_TIPOVER", "落地觸地後未判定落地,於地面翻覆(近地面翻覆)", [
        Weight("touchdown_not_recognised", 4.0),
        Weight("lean_without_pilot", 3.0, _lean_near_ground),
        Weight("idle_motor_asymmetry", 3.0),
        Weight("impact_shock_clusters", 2.0, lambda f, c: f.values.get("cls") == "ground_contact"
               or f.context == "ground_contact"),
        Weight("gravity_only_tipover", 2.0),
        Weight("rate_demand_not_achieved", 2.0, _ctx_in("ground_idle")),
        Weight("motor_thrust_imbalance", 1.0, _ctx_in("ground_idle")),
        Weight("ekf_phantom_motion_on_ground", 1.0),
        Weight("touchdown_lateral_speed", 1.0),
        Weight("vio_dropout", 1.0, _ctx_in("near_touchdown")),
        Weight("attitude_loss_of_control", 1.0),
        Weight("tumble_inversion", 1.0, lambda f, c: f.tag != TAG_CONSEQUENCE and f.values.get("axis", "roll") == "roll"),
    ], ["touchdown_not_recognised", "idle_motor_asymmetry", "lean_without_pilot"],
        "降落最後階段改用 LAND 模式,或觸地後立刻把油門拉到底並上鎖(不要停在 Loiter 待機);"
        "降落前確認 VIO/EKF 位置與速度一致(尤其 VIO-only 來源組,考慮設定 EK3_SRC3_VELXY=6 取得速度來源);"
        "檢查降落偵測為何沒成立(傾角需求超過 15 度會擋住 LAND_COMPLETE_MAYBE);"
        "檢查起落架、地面平整度與觸地水平速度;維持 FS_CRASH_CHECK=1。", _gate_land),
    Hypothesis("H_NAV_FAILURE", "導航估計失效或漂移(EKF/GPS/VIO),導致位置控制器下錯誤指令", [
        Weight("ekf_testratio_gate", 4.0),
        Weight("ekf_attitude_consistency", 3.0),
        Weight("source_switch_audit", 3.0, only_precursor=True),
        Weight("ekf_aiding_timeout", 3.0, only_precursor=True),
        Weight("ekf_phantom_motion_on_ground", 3.0),
        Weight("lean_without_pilot", 2.0, lambda f, c: bool(f.values.get("lean_from_pos"))),
        Weight("vio_dropout", 2.0, _ctx_in("in_vio_mode", "near_touchdown")),
        Weight("ekf_vio_disagreement", 2.0),
        Weight("nav_velocity_gps_fallback_risk", 1.0),
        Weight("touchdown_lateral_speed", 1.0),
    ], ["ekf_testratio_gate", "ekf_aiding_timeout", "ekf_phantom_motion_on_ground"],
        "確認使用中的來源組是否有水平速度來源;GPS 無定位時不要把開關切到 GPS 組;"
        "檢查 VIO 的 PErr 為何卡在高值、觸地時 VIO 為何中斷;飛行中即時監看 EKF 與 VIO 位置差"
        "(而不是只看 EKF 測試比);降落階段避免依賴漂移的位置估測(改用 LAND 或高度/姿態模式)。", _gate_nav),
    Hypothesis("H_MOTOR_THRUST", "馬達/電調/螺旋槳推力損失或推力失衡(含偏航失衡)", [
        Weight("motor_thrust_loss_check", 4.0),
        Weight("yaw_imbalance_iterm", 3.0),
        Weight("motor_thrust_imbalance", 3.0, _ctx_in("air")),
        Weight("rate_output_saturation", 2.0, _ctx_in("air")),
        Weight("rate_demand_not_achieved", 2.0, _ctx_in("air")),
        Weight("uncommanded_yaw_or_sink", 2.0),
        Weight("free_fall_run", 2.0, lambda f, c: c.a.t_warn is None or f.t_onset is None or f.t_onset < c.a.t_warn),
        Weight("abnormal_disarm_classifier", 3.0, lambda f, c: _cls(f) == "emergency_stop_in_air"),
    ], ["motor_thrust_loss_check", "yaw_imbalance_iterm", "motor_thrust_imbalance"],
        "檢查各馬達/槳/電調外觀與安裝,確認轉向與槳型;加裝電調遙測(ESC telemetry/RPM)與電流計以便下次排除;"
        "若推力損失條件成立,檢查電池內阻與電壓、電調校準與馬達座。", _gate_motor),
    Hypothesis("H_POWER", "電池/電源系統失效(電壓過低、瞬間斷電、電源軌異常)", [
        Weight("power_voltage_collapse", 4.0),
        Weight("power_low_voltage", 4.0),
        Weight("power_rail_fault", 3.0),
        # 規格另列 log_ended_armed 3、電池原因的失效保護 3:det_power 已把「解鎖中 log 結束在掉壓曲線上」併入
        # power_voltage_collapse、把飛行中 ERR 6 併入 power_low_voltage,再加權會把同一個物理事件算兩次 → 不另加。
    ], ["power_voltage_collapse", "power_low_voltage", "power_rail_fault"],
        "落地後檢查電池接頭、焊點與電源模組;修好電流計(BATT_CURR_PIN、BATT_AMP_PERVLT 並用鉗表校正);"
        "修正電池保護參數(LOW≈3.5 V/cell、CRT≈3.3 V/cell、失效保護動作設為 Land/RTL)。", _gate_power),
    Hypothesis("H_VIBRATION_IMU", "振動過大或 IMU 感測器異常", [
        Weight("vibration_flight_level", 4.0),
        Weight("vibe_clip_transient", 3.0, lambda f, c: f.context in _VIB_CLIP_CONTEXTS),
        Weight("imu_consistency_health", 3.0),
        Weight("inflight_failsafe_or_mode_change", 4.0, lambda f, c: _fs_trigger(f, subsys=_VIBE_SUBSYS),
               only_precursor=True),
    ], ["vibration_flight_level", "imu_consistency_health"],
        "若 p95 超過 15,檢查槳葉平衡、馬達軸承、飛控避震與機架共振;確認 IMU 取樣率與濾波參數(INS_ 與 notch)。", _gate_vib),
    Hypothesis("H_COMPASS_YAW", "羅盤干擾或航向估計錯誤", [
        Weight("ekf_mag_reject", 4.0, only_precursor=True),
        Weight("compass_raw_consistency", 3.0, only_precursor=True),
        Weight("ekf_yaw_reset_lane_switch", 3.0, only_precursor=True),
        Weight("compass_sensor_health", 3.0, only_precursor=True),
        Weight("ekf_attitude_consistency", 2.0, lambda f, c: f.values.get("axis") in ("yaw", "dcm_yaw"), only_precursor=True),
        Weight("compass_motor_interference", 2.0),
    ], ["ekf_mag_reject", "compass_sensor_health", "ekf_yaw_reset_lane_switch"],
        "重新校正羅盤,將羅盤遠離電源線與電調;備援羅盤干擾偏高時移位或執行 compassmot;確認 COMPASS_ORIENT 與 EK3_SRCn_YAW 設定。",
        _gate_compass),
    Hypothesis("H_CONTROL_PILOT", "飛手操作或刻意飛行動作(含特技/翻滾模式、空中手動上鎖)", [
        Weight("pilot_input_context", 3.0, only_precursor=True),
        Weight("abnormal_disarm_classifier", 4.0, lambda f, c: _cls(f) == "manual_disarm_in_air"),
        Weight("rate_output_saturation", 1.0, _no_motor_nav_evidence, only_precursor=True),
    ], ["pilot_input_context"],
        "此項成立時檢查遙控器操作與模式設定;若無法排除,建議以錄影確認操作過程。", _gate_pilot, _bonus_flip),
    Hypothesis("H_FAILSAFE_ACTION", "失效保護動作觸發(遙控/電池/GCS/GPS/EKF failsafe 造成模式或動力改變)", [
        Weight("inflight_failsafe_or_mode_change", 4.0, only_precursor=True),
        Weight("abnormal_disarm_classifier", 3.0, lambda f, c: _cls(f) == "failsafe_disarm"),
        Weight("ekf_aiding_timeout", 2.0, only_precursor=True),
        Weight("power_low_voltage", 2.0, lambda f, c: _num(f.values.get("err6_count"), 0) > 0),
    ], ["inflight_failsafe_or_mode_change"],
        "檢查被觸發的失效保護來源(訊號/電池/GCS/GPS/EKF)與其參數,確認失效保護動作與門檻合理。", _gate_failsafe),
    Hypothesis("H_EXTERNAL_IMPACT", "外力撞擊或重落地造成的機械損傷(撞擊前姿態原本正常)", [
        Weight("impact_shock_clusters", 4.0, lambda f, c: _hard_impact(f), only_precursor=True),
        Weight("vibe_clip_transient", 2.0, lambda f, c: f.context == "impact_transient", only_precursor=True),
        Weight("free_fall_run", 2.0, only_precursor=True),
        Weight("attitude_loss_of_control", 2.0, lambda f, c: c.a.t_first_hit is not None and f.t_onset is not None
               and abs(f.t_onset - c.a.t_first_hit) <= 0.5),
    ], ["impact_shock_clusters"],
        "此項成立時檢查機架、起落架、槳葉與飛控固定是否受損,並回溯撞擊前的環境(障礙物、降落速度)。", _gate_impact),
]
HYP_BY_ID = {h.id: h for h in HYPOTHESES}


@dataclass
class HypScore:
    id: str
    title_zh: str
    score: float = 0.0
    raw: float = 0.0
    gate_ok: bool = True
    gate_note: str = ""
    evidence: list[dict] = field(default_factory=list)      # {det, domain, status, tag, t_onset, w, level, prox, contrib, text_zh}
    status: str = "ranked"        # ranked / contributing / minor / ruled_out / not_applicable / no_evidence
    confidence: str = ""          # 高 / 中 / 低(第一名與貢獻因素)
    share: float = 0.0
    advice_zh: str = ""
    ruled_out_basis: list[str] = field(default_factory=list)   # 已排除的依據(CLEAR 的數字 / 不計分的後果)
    checked_ok: list[str] = field(default_factory=list)        # 此假說相關偵測器中「已檢查正常」(CLEAR)的說明
    rule_out_conf: str = ""       # 已排除的信心(高 / 中;資料品質受限時封頂為中)
    rule_out_note: str = ""       # 封頂原因
    confidence_note: str = ""     # 信心被封頂的原因(例:證據只來自沒有實測樣本的結束分類)


_LATE_TAGS = (TAG_CONSEQUENCE, TAG_POST)

# abnormal_disarm_classifier 中沒有任何實測樣本的分類(規格 catalogue:只觀察過 Method 0 與 12;
# 空中上鎖 / 失效保護上鎖 / log 在解鎖中結束 / EV54 緊急停止 都只依 ArduPilot 語意寫成)→ 只靠它們的假說信心封頂為「低」
UNTESTED_END_CLASSES = frozenset({"manual_disarm_in_air", "failsafe_disarm", "log_ended_armed", "emergency_stop_in_air"})
UNTESTED_CONF_NOTE = "證據只來自沒有實測樣本的結束分類(空中上鎖 / 失效保護上鎖 / log 中斷 / 緊急停止),信心封頂為低"


def _untested(f: Finding) -> bool:
    return f.id == "abnormal_disarm_classifier" and _cls(f) in UNTESTED_END_CLASSES


def _excluded_label(f: Finding) -> str:
    if f.tag in _LATE_TAGS:
        return "(後果,不計分)"
    if f.tag == TAG_GROUND:
        return "(地面 / 未解鎖事件,不計分)"
    return "(不符合此假說的條件,不計分)"


def _safe_when(w: Weight, f: Finding, c: "Ctx2") -> bool:
    try:
        return bool(w.when(f, c))
    except Exception:  # noqa: BLE001 — 條件函式只是篩選,出錯就當不符
        return False


def _safe_gate(h: Hypothesis, c: "Ctx2") -> tuple[bool, str]:
    if h.gate is None:
        return True, ""
    try:
        ok, note = h.gate(c)
        return bool(ok), str(note or "")
    except Exception as e:  # noqa: BLE001 — 閘門檢查本身出錯 = 必要證據未驗證(不讓整個分析中斷)
        return False, f"必要證據檢查失敗({type(e).__name__}),視為未驗證"


def _safe_bonus(h: Hypothesis, c: "Ctx2") -> list[tuple[str, float, str]]:
    if h.bonus is None:
        return []
    try:
        return list(h.bonus(c) or [])
    except Exception:  # noqa: BLE001
        return []


def score_all(findings: list[Finding], a: Anchors) -> list[HypScore]:
    for f in findings:                      # 防呆:偵測器回傳 values=None 時當成空 dict
        if not isinstance(f.values, dict):
            f.values = {} if f.values is None else {"value": f.values}
    c = Ctx2(findings, a)
    out: list[HypScore] = []
    for h in HYPOTHESES:
        hs = HypScore(h.id, h.title_zh, advice_zh=h.advice_zh)
        used: set[int] = set()
        for w in h.weights:
            for f in c.get(w.det):
                lvl = f.level()
                if lvl <= 0:
                    continue
                if w.when is not None and not _safe_when(w, f, c):
                    continue
                if w.only_precursor and f.tag not in (TAG_PRECURSOR, TAG_STANDING):
                    continue
                p = prox(f, a)
                contrib = w.w * lvl * p
                hs.raw += contrib
                if contrib > 0:
                    used.add(id(f))
                hs.evidence.append({"det": f.id, "domain": f.domain, "status": f.status, "tag": f.tag,
                                    "t_onset": f.t_onset, "w": w.w, "level": lvl, "prox": p, "contrib": contrib,
                                    "context": f.context, "text_zh": f.text_zh, "untested": _untested(f)})
        for label, wt, text in _safe_bonus(h, c):
            hs.raw += wt
            hs.evidence.append({"det": label, "domain": "scenario", "status": HIT, "tag": TAG_PRECURSOR,
                                "t_onset": a.t_ref(), "w": wt, "level": 1.0, "prox": 1.0, "contrib": wt,
                                "context": "scenario", "text_zh": text, "untested": False})
        hs.gate_ok, hs.gate_note = _safe_gate(h, c)
        hs.score = hs.raw if hs.gate_ok or hs.raw == 0 else hs.raw * GATE_PENALTY
        # 已檢查正常:權重表內、情境條件相符(例如 context)的 CLEAR 結果;主要偵測器在前
        order = {d: i for i, d in enumerate(h.primary)}
        ok = []
        for w in sorted(h.weights, key=lambda w: order.get(w.det, len(order))):
            for f in c.get(w.det):
                if f.status == CLEAR and f.text_zh and (w.when is None or _safe_when(w, f, c)):
                    ok.append(f.text_zh)
        hs.checked_ok = list(dict.fromkeys(ok))[:4]
        # 已排除 / 不適用:主要偵測器的狀態
        stats = [c.status(d) for d in h.primary]
        if hs.raw == 0:
            excluded = [f for d in h.primary for f in c.get(d) if f.status in (HIT, SOFT) and id(f) not in used]
            prim_clear = [f.text_zh for d in h.primary for f in c.get(d) if f.status == CLEAR and f.text_zh]
            if excluded or any(s == CLEAR for s in stats):
                hs.status = "ruled_out"           # 主要偵測器正常,或異常只在姿態失控之後 / 不符情境出現
                basis = list(dict.fromkeys(prim_clear)) + [_excluded_label(f) + f.text_zh for f in excluded if f.text_zh]
                hs.ruled_out_basis = basis[:3]
                hs.rule_out_conf = "高" if all(s != NA for s in stats) else "中"
                if hs.rule_out_conf == "中":
                    hs.rule_out_note = "部分主要偵測器資料不足(不適用),排除依據不完整"
            elif stats and all(s == NA for s in stats):
                hs.status = "not_applicable"
            else:
                hs.status = "no_evidence"
        out.append(hs)
    return out


def _untested_only(s: HypScore) -> bool:
    pos = [e for e in s.evidence if e["contrib"] > 0]
    return bool(pos) and all(e.get("untested") for e in pos)


def _insufficient(live: list[HypScore], note: str, scores: list[HypScore]) -> tuple[list[HypScore], str, str]:
    """證據不足 / 無法區分:不硬給原因 —— 所有有分數的假說都只是「候選」(沒有第一名、沒有貢獻因素)。"""
    for s in live:
        s.status = "candidate"
    return scores, "insufficient", note


def rank(scores: list[HypScore], a: Anchors, data_ok: bool = True) -> tuple[list[HypScore], str, str]:
    """排名 + 信心 + 總判定。回傳 (排序後的 scores, verdict, note);verdict ∈ ranked / insufficient / no_failure。
    insufficient 時有分數的假說狀態一律是 candidate(候選,無法確定),不是 ranked。"""
    live = sorted([s for s in scores if s.score > 0], key=lambda s: -s.score)
    total = sum(s.score for s in live) or 1.0
    for s in live:
        s.share = s.score / total
    if a.end_type in ("not_armed", "normal_landing", "normal_end"):
        return scores, "no_failure", "沒有需要解釋的失事:只列出健康觀察"
    if not data_ok:
        return _insufficient(live, "資料不足(核心資料表不到 3 張可用,或解鎖後不到 20 秒)", scores)
    if not live or live[0].score < S_MIN:
        return _insufficient(live, (f"最高分假說僅 {live[0].score:.1f}(< {S_MIN}):證據不足" if live
                                    else "沒有任何假說有證據"), scores)
    s1 = live[0]
    s2 = live[1] if len(live) > 1 else None
    if s2 is not None and s2.score >= AMBIG_RATIO * s1.score and s1.score - s2.score < AMBIG_MARGIN:
        s1.confidence = s2.confidence = "低"
        return _insufficient(live, f"無法區分 {s1.title_zh} 與 {s2.title_zh}(分數接近)", scores)
    if not s1.gate_ok and not any(s.gate_ok for s in live[1:]):
        return _insufficient(live, f"第一名「{s1.title_zh}」缺少必要證據:{s1.gate_note}", scores)
    margin = s1.score - (s2.score if s2 else 0.0)
    domains = len({e.get("domain", e["det"]) for e in s1.evidence if e["contrib"] > 0})
    if s1.score >= HIGH_SCORE and s1.share >= HIGH_SHARE and margin >= HIGH_MARGIN and s1.gate_ok:
        s1.confidence = "高"
    elif s1.gate_ok and (s1.score >= S_MIN or MID_SHARE <= s1.share < HIGH_SHARE):
        s1.confidence = "中"
    else:
        s1.confidence = "低"
    if domains < 2 and s1.confidence == "高":
        s1.confidence = "中"
    s1.status = "ranked"
    for s in live[1:]:
        if s.score >= max(S_MIN, CONTRIB_SHARE * s1.score):
            s.status = "contributing"
            # 貢獻因素:與事件吻合但無法證明是觸發原因 → 信心最多「中」
            s.confidence = "中" if s.gate_ok else "低"
        else:
            s.status = "minor"
    for s in live:
        if s.status in ("ranked", "contributing") and _untested_only(s):
            s.confidence = "低"
            s.confidence_note = UNTESTED_CONF_NOTE
    return scores, "ranked", ""
