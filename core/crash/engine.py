"""失事原因分析引擎:Ctx → 閘門 → 各領域偵測 → 因果標記 → 假說評分 → 報告。

入口:
  analyze(logdata, interval_index=None)   LogData → CrashReport(預設分析「最後一次異常結束」的解鎖區間)
  analyze_ctx(ctx, ...)                   已建好的 Ctx(測試可用合成 Ctx / 假偵測模組)
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field

import numpy as np

from core.dflog import ARM_METHODS, err_text, ev_text

from . import hypotheses as hyp
from .anchors import CRASH_CHECK_SUBSYS, NORMAL_DISARM_METHODS, basic_anchors
from .ctx import Ctx
from .types import (HIT, NA, SOFT, TAG_CONSEQUENCE, TAG_DEVELOPMENT, TAG_GROUND, TAG_POST,
                    TAG_PRECURSOR, TAG_STANDING, Anchors, Finding)

# 偵測模組(執行順序:end_state 先定案終點 → 姿態 → 地面 → 其他)
MODULES = ["det_endstate", "det_attitude", "det_ground", "det_motor", "det_power", "det_vib",
           "det_nav", "det_compass", "det_failsafe"]

# 持續性狀態 / 健康檢查偵測器:不依時間先後歸因(因果標籤 = standing)
STANDING_IDS = frozenset({"ekf_vio_disagreement", "nav_velocity_gps_fallback_risk", "power_failsafe_config_audit",
                          "power_data_quality", "compass_motor_interference", "vibration_flight_level",
                          "imu_consistency_health", "source_switch_position_jump", "ekf_testratio_gate_summary"})
# 已列在【資料品質與限制】的偵測器(不重複列在其他觀察)
LIMITATION_IDS = frozenset({"power_data_quality"})

NO_FAILURE_END_TYPES = frozenset({"not_armed", "normal_landing", "normal_end"})
_EV_NOT_LANDED = 28
POST_WINDOW_S = 60.0          # 上鎖後 60 s 內的事件仍列在時間軸(標「事後」)
PRE_ANOMALY_S = 30.0          # 時間軸從最早異常前 30 s 開始
NORMAL_TAIL_S = 60.0          # 正常結束的飛行:時間軸只列上鎖前 60 s ~ 上鎖後 10 s
NORMAL_POST_S = 10.0
CORE_TABLES = (("ANG", "ATT"), ("IMU",), ("CTUN",), ("RCOU",), ("XKF4",))
MIN_CORE_TABLES = 3
MIN_ARMED_S = 20.0            # 異常結束前的解鎖時間不到 20 s → 資料不足

# ---- 失事型態(dataflow STEP 3)門檻 ----
SCN_AGL_GROUND_M = 1.0        # 觸地(或姿態預警)時 agl ≤ 1.0 m 視為貼地(124:-0.13 m;無測距儀,氣壓漂移約 ±0.15 m)
SCN_VZ_PRE_S = 5.0            # vz_pre = 觸地前 5 s 的 CTUN.CRt 中位數
SCN_VZ_HARD_CMS = -100.0      # 貼地且 vz_pre < −100 cm/s → 重落地(124:−30 cm/s)
SCN_VZ_FREEFALL_CMS = -400.0  # vz_pre ≤ −400 cm/s → 墜落
SCN_SINK_ERR_CMS = -100.0     # 預警前 1 s CRt − DCRt 中位數 ≤ −100 cm/s → 失去升力(下沉超過指令)
SCN_LEAN_AUTOPILOT_DEG = 12.0  # 搖桿置中時飛控自己要求的傾斜 ≥ 12° → autopilot_lean(124 峰值 25°;正常空中最大 9.7°)
SCN_THO_IDLE = 0.03           # 預警時油門 ThO ≤ 0.03 且飛控仍認為在空中 → land_not_declared(124:0.00)
SCN_STICK_WIN_S = 1.0         # 搖桿偏移特徵:預警前 1 s 的 C1/C2/C4 相對 trim 最大偏移(只列數字,判定交給 pilot_input_context)
FLIP_MODES = ("ACRO", "SPORT", "FLIP")
FLIP_MODE_NUMS = (1, 13, 14)  # ArduCopter ModeNum:ACRO / SPORT / FLIP(模式名稱不可用時備援)
EV_FLIP_START = 21
FLIP_EV_WIN_S = 10.0
EDGE_TOL_S = 0.01             # 偵測器把 t_onset 四捨五入到 ms / 0.1 ms:解鎖區間邊界容許 10 ms

# 時間軸要列出的 STATUSTEXT(其餘視為雜訊)
_MSG_KEEP = re.compile(r"^(Crash|EKF variance|GPS Glitch|Glitch cleared|Using EKF Source Set|SRC=|Arm:|PreArm: (AHRS|Need|GPS)|"
                       r"EKF\d IMU\d stopped aiding|Potential Thrust Loss|Yaw Imbalance|.*[Ff]ailsafe|Land|Disarm|Mode)", re.I)
_MSG_DROP = re.compile(r"(ArduCopter V|ChibiOS|IOMCU|RCOut|IMU\d: fast|Frame:|GCS:Mission Planner|PreArm: Hardware safety|"
                       r"PreArm: Scripting|^Lua:|VIO degraded: near-range|EKF_YAW_RESET|"
                       r"^(GPS Glitch or Compass error|Glitch cleared|Using EKF Source Set \d|EKF variance)$)", re.I)

SCENARIO_ZH = {
    "GROUND_TIPOVER": "降落觸地後在地面翻覆(不是空中失控,也不是墜落)",
    "HARD_LANDING": "過猛的降落(落地速度過大)",
    "FREE_FALL": "空中失去升力後墜落",
    "AIR_LOSS_OF_CONTROL": "空中失去控制",
    "COMMANDED_FLIP": "刻意的翻滾 / 特技動作(非失控)",
    "UNKNOWN": "型態不明",
}
SUBLABEL_ZH = {
    "autopilot_lean": "搖桿置中時飛控自己要求傾斜(位置控制器)",
    "pilot_stick": "異常前有飛手大動作",
    "land_not_declared": "油門已在怠速但飛控沒有宣告落地",
    "motor_saturation": "馬達輸出飽和",
    "attitude_cmd_runaway": "姿態指令失控(搖桿置中仍要求大傾角)",
    "unknown": "原因型態不明",
}
END_TYPE_ZH = {
    "crash_check": "飛控判定墜機並自動上鎖(CRASH_CHECK)",
    "crash_check_replay": "依 crash_check 條件重算判定為墜機(log 內沒有 CRASH_CHECK 紀錄)",
    "abnormal_disarm": "非正常落地的上鎖",
    "hard_impact": "偵測到硬撞擊(沒有 CRASH_CHECK)",
    "log_ended_armed": "log 在解鎖狀態下結束(疑似斷電 / 記錄中斷)",
    "tumble": "機身翻覆",
    "normal_landing": "正常降落後上鎖",
    "normal_end": "正常結束",
    "not_armed": "未解鎖",
    "unknown": "未知",
}
DOMAIN_ZH = {"end_state": "終點 / 撞擊", "attitude": "姿態 / 控制", "ground": "地面 / 降落", "motor": "馬達",
             "power": "電池 / 電源", "vibration": "振動 / IMU", "nav": "導航(EKF / VIO / GPS)",
             "compass": "羅盤 / 航向", "failsafe": "失效保護"}

# 未解鎖 log 的供電判斷(只當資訊):BAT.Volt 中位數 < 5 V = USB 供電 / 沒接電池(實測 USB log ≤ 0.08 V;有電池 ≥ 47.8 V)
BENCH_BATTERY_PRESENT_V = 5.0


@dataclass
class TimelineEvent:
    t: float                      # 相對秒
    category: str                 # arm / mode / ev / err / msg / derived / pilot / vio
    label: str
    tag: str = ""
    source: str = ""
    t_abs: float | None = None    # log 絕對秒(= t + t_base)

    def to_dict(self):
        return {"t": self.t, "t_abs": self.t_abs, "category": self.category, "label": self.label, "tag": self.tag,
                "source": self.source}


@dataclass
class CrashReport:
    path: str = ""
    t_base: float = 0.0
    end_type: str = "unknown"
    scenario: str = ""
    scenario_sub: list[str] = field(default_factory=list)       # 子型態(autopilot_lean / land_not_declared …)
    scenario_features: dict = field(default_factory=dict)       # agl / vz_pre / ThO / 模式 …(判定型態用的數字)
    verdict: str = "no_failure"          # ranked / insufficient / no_failure
    verdict_note: str = ""
    anchors: Anchors | None = None
    interval: dict | None = None
    intervals: list[dict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    hypotheses: list = field(default_factory=list)      # hypotheses.HypScore(未解鎖 / 正常結束 → 空:不排名)
    chain: list[dict] = field(default_factory=list)     # 事件鏈(機制):第一名與貢獻因素的先兆 / 發展證據,依時間
    observations: list[dict] = field(default_factory=list)   # 其他觀察(非本次失事原因)/ 正常飛行的健康觀察
    na_domains: list[dict] = field(default_factory=list)     # 不適用的分析領域(未解鎖 log)
    timeline: list[TimelineEvent] = field(default_factory=list)
    ground_events: list[TimelineEvent] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    summary_zh: str = ""
    missing_modules: list[str] = field(default_factory=list)   # 不存在的偵測模組
    failed_modules: list[str] = field(default_factory=list)    # 載入 / 執行失敗的偵測模組(該領域沒有結果)
    fs_crash_check: float | None = None                        # PARM FS_CRASH_CHECK(None = log 沒記錄)

    # --- 方便查詢 ---
    def finding(self, det_id: str) -> Finding | None:
        fs = [f for f in self.findings if f.id == det_id]
        return max(fs, key=lambda f: f.level(), default=None)

    def hypothesis(self, hid: str):
        return next((h for h in self.hypotheses if h.id == hid), None)

    def top(self):
        r = [h for h in self.hypotheses if h.status == "ranked" and h.score > 0]
        return max(r, key=lambda h: h.score, default=None)

    def to_dict(self) -> dict:
        a = self.anchors
        return {
            "path": self.path, "t_base": self.t_base, "end_type": self.end_type, "scenario": self.scenario,
            "scenario_sub": self.scenario_sub, "scenario_features": self.scenario_features,
            "verdict": self.verdict, "verdict_note": self.verdict_note, "summary_zh": self.summary_zh,
            "interval": self.interval, "intervals": self.intervals,
            "anchors": {k: getattr(a, k) for k in ("t_first_anomaly", "t_touch", "t_warn", "t_loss", "t_first_hit",
                                                   "t_impact_peak", "t_rest", "t_detect", "t_disarm", "t_inverted")} if a else {},
            "anchor_notes": list(a.notes) if a else [],
            "findings": [dict(id=f.id, domain=f.domain, status=f.status, severity=f.severity, t_onset=f.t_onset,
                              t_alarm=f.t_alarm, context=f.context, values=f.values, text_zh=f.text_zh, tag=f.tag)
                         for f in self.findings],
            "hypotheses": [dict(id=h.id, title_zh=h.title_zh, score=h.score, raw=h.raw, share=h.share, status=h.status,
                                confidence=h.confidence, confidence_note=h.confidence_note, gate_ok=h.gate_ok,
                                gate_note=h.gate_note, evidence=h.evidence,
                                ruled_out_basis=h.ruled_out_basis, checked_ok=h.checked_ok,
                                rule_out_conf=h.rule_out_conf, rule_out_note=h.rule_out_note) for h in self.hypotheses],
            "chain": self.chain, "observations": self.observations, "na_domains": self.na_domains,
            "timeline": [e.to_dict() for e in self.timeline],
            "ground_events": [e.to_dict() for e in self.ground_events],
            "recommendations": self.recommendations, "notes": self.notes, "limitations": self.limitations,
            "missing_modules": self.missing_modules, "failed_modules": self.failed_modules,
            "fs_crash_check": self.fs_crash_check,
        }

    def format_text(self) -> str:
        from .report import format_report
        return format_report(self)

    def format_markdown(self) -> str:
        from .report import format_markdown
        return format_markdown(self)


# ------------------------------------------------------------------ 主流程

def analyze(logdata, interval_index: int | None = None) -> CrashReport:
    """LogData → CrashReport。interval_index=None:自動選「最後一次異常結束」的解鎖區間(沒有就最後一次),
    異常的判準見 interval_abnormal_reasons()。"""
    return analyze_ctx(Ctx(logdata), path=getattr(logdata, "path", "") or "", interval_index=interval_index)


def analyze_ctx(ctx: Ctx, path: str = "", interval_index: int | None = None, modules=None) -> CrashReport:
    """modules:模組名稱(core.crash 底下)或有 run(ctx, a) 的物件清單;None = MODULES。"""
    a = select_anchors(ctx, interval_index)
    rep = CrashReport(path=path, t_base=ctx.t_base, anchors=a)
    fs = ctx.param("FS_CRASH_CHECK", None)
    rep.fs_crash_check = None if fs is None else float(fs)
    rep.intervals = _interval_summaries(ctx)
    findings = _run_modules(ctx, a, rep, modules)

    if a.interval is None:
        a.end_type = rep.end_type = "not_armed"
        for f in findings:
            f.tag = TAG_GROUND
        rep.findings = findings
        rep.verdict = "no_failure"
        rep.verdict_note = "未解鎖 / 非飛行 log(桌上或手持):不做失事分析"
        rep.na_domains = _na_domains(findings)
        bad = sorted({f.id for f in findings if f.status != NA})
        if bad:
            rep.notes.append("(未解鎖 log 卻有偵測器回傳非「不適用」結果,已忽略:" + "、".join(bad) + ")")
        rep.ground_events = _ground_events(ctx)
        rep.notes.append("這份 log 從未解鎖(馬達沒轉):故障碼 / 失效保護 / GPS glitch 只可能是地面事件,不列入原因分析。")
        power_txt = " ".join(f.text_zh for f in findings if f.domain == "power")
        rep.notes.extend(n for n in _bench_power_notes(ctx)
                         if not (("USB" in n and "USB" in power_txt) or ("拔電池" in n and "拔電池" in power_txt)))
        _finish_text(rep, ctx, findings)
        return rep

    rep.interval = {"t0": a.interval.t0, "t1": a.interval.t1, "method": a.interval.method,
                    "index": a.interval.index, "open_end": a.interval.open_end}
    if a.end_type in ("unknown", ""):
        a.end_type = _fallback_end_type(ctx, a)
    _finalize_anchors(ctx, a, findings)
    _tag_findings(ctx, a, findings)
    rep.findings = findings
    rep.end_type = a.end_type
    rep.scenario = a.scenario
    rep.scenario_sub = list(a.extra.get("scenario", {}).get("sub", []))
    rep.scenario_features = {k: v for k, v in a.extra.get("scenario", {}).items() if k != "sub"}

    if undetermined_end(a):
        # 結束型態不明且沒有任何事件錨點(姿態預警 / 撞擊 / 判定):不知道有沒有失事 → 不排名原因,只列觀察
        rep.verdict = "insufficient"
        rep.verdict_note = ("無法判定結束型態(沒有上鎖原因、落地事件、撞擊或姿態異常可判斷是否失事),"
                            "不排名原因;下方只列出觀察到的異常(未計分)")
        rep.hypotheses = []
        rep.observations = _observations(findings, [], health=True)
    else:
        scores = hyp.score_all(findings, a)
        scores, rep.verdict, rep.verdict_note = hyp.rank(scores, a, data_ok=_data_ok(ctx, a))
        if rep.verdict == "no_failure":
            rep.hypotheses = []                    # 正常結束:沒有原因排名,只列健康觀察
            rep.observations = _observations(findings, scores, health=True)
        else:
            _cap_rule_out(ctx, findings, scores)
            rep.hypotheses = sorted(scores, key=lambda s: -s.score)
            rep.chain = _chain(rep.hypotheses)
            rep.observations = _observations(findings, scores, health=False)
    rep.timeline = _timeline(ctx, a, findings)
    if _no_causality(a):
        rep.ground_events = _ground_events(ctx)
    else:
        post_lo = a.t_disarm if a.t_disarm is not None else a.interval.t1
        rep.ground_events = [e for e in _ground_events(ctx) if not (post_lo < e.t <= post_lo + POST_WINDOW_S)]   # 已在時間軸
    _finish_text(rep, ctx, findings)
    return rep


CRASH_ERR_TAIL_S = 0.5        # CRASH_CHECK ERR 最多比上鎖晚 0.5 s 寫出,仍算這次解鎖
LANDED_QUERY_BEFORE_S = 0.002  # 「上鎖前是否已落地」在上鎖前 2 ms 查(上鎖 1 ms 內補寫的 EV17/18 不算落地)


def _crash_err_times(ctx: Ctx) -> list[float]:
    t, sub, code = ctx.err_rows()
    return [float(x) for x, s, c in zip(t, sub, code) if s == CRASH_CHECK_SUBSYS and c in (1, 2)]


def interval_abnormal_reasons(ctx: Ctx, iv, crash_t: list[float] | None = None) -> list[str]:
    """dataflow STEP 2 的便宜判準(不跑偵測器):這次解鎖是否「異常結束」,回傳理由(空 = 正常)。
      log 在解鎖中結束;ARM.Method 12 或不在一般上鎖原因內(失效保護 / 自動上鎖);區間內有 CRASH_CHECK ERR;
      曾起飛(EV28)卻到上鎖前都沒有 LAND_COMPLETE(上鎖時飛控仍認為在空中:空中上鎖 / 撞擊後手動上鎖 / FS_CRASH_CHECK=0)。"""
    crash_t = _crash_err_times(ctx) if crash_t is None else crash_t
    out: list[str] = []
    if iv.open_end:
        out.append("log 在解鎖中結束")
    if iv.method == 12:
        out.append("CRASH_CHECK 上鎖")
    elif iv.method is not None and iv.method not in NORMAL_DISARM_METHODS:
        out.append(f"非一般上鎖(Method {iv.method})")
    if any(iv.t0 <= x <= iv.t1 + CRASH_ERR_TAIL_S for x in crash_t) and iv.method != 12:
        out.append("CRASH_CHECK 事件")
    if not iv.open_end and ctx.ev_times(_EV_NOT_LANDED, iv.t0, iv.t1) \
            and ctx.landed_at(iv.t1 - LANDED_QUERY_BEFORE_S, iv) is False:
        out.append("上鎖時飛控仍認為在空中(沒有 LAND_COMPLETE)")
    return out


def auto_interval_index(ctx: Ctx) -> int | None:
    """自動選要分析的解鎖區間:最後一次異常結束的那次(interval_abnormal_reasons 非空);都正常就最後一次。"""
    if not ctx.armed:
        return None
    crash_t = _crash_err_times(ctx)
    abn = [i for i, iv in enumerate(ctx.armed) if interval_abnormal_reasons(ctx, iv, crash_t)]
    return abn[-1] if abn else len(ctx.armed) - 1


def select_anchors(ctx: Ctx, interval_index: int | None = None) -> Anchors:
    """初始錨點。interval_index 指定要分析第幾次解鎖(0 起算,可負數);None = auto_interval_index 的自動選擇。"""
    if not ctx.armed:
        return basic_anchors(ctx)           # 未解鎖:end_type not_armed
    if interval_index is None:
        interval_index = auto_interval_index(ctx)
    iv = ctx.armed[interval_index]          # 超出範圍 → IndexError(呼叫端的錯)
    a = Anchors(intervals=list(ctx.armed))
    a.interval = iv
    a.t_disarm = iv.t1
    inside = [x for x in _crash_err_times(ctx) if iv.t0 <= x <= iv.t1 + CRASH_ERR_TAIL_S]
    if inside:
        a.t_detect = inside[0]
        a.end_type = "crash_check"
    return a


# 模組 → 領域(模組載入失敗、拿不到 DOMAIN 時用)
MODULE_DOMAIN = {"det_endstate": "end_state", "det_attitude": "attitude", "det_ground": "ground", "det_motor": "motor",
                 "det_power": "power", "det_vib": "vibration", "det_nav": "nav", "det_compass": "compass",
                 "det_failsafe": "failsafe"}
MODULE_ERROR_PREFIX = "module_error:"


def _module_error(name: str, domain: str | None, e: Exception, stage: str) -> Finding:
    """整個模組壞掉時的佔位結果:該領域 NA(不適用 / 資料缺失),絕不當成 CLEAR。"""
    dom = domain or MODULE_DOMAIN.get(name, "infra")
    where = f"「{DOMAIN_ZH[dom]}」領域" if dom in DOMAIN_ZH else "這個模組的偵測器"
    return Finding(MODULE_ERROR_PREFIX + name, dom, NA, "info", context="error",
                   values={"error": f"{type(e).__name__}: {e}", "stage": stage},
                   text_zh=f"偵測模組 {name} {stage}失敗,{where}沒有分析結果(不代表正常)。")


def _normalise(f: Finding) -> Finding:
    """偵測器輸出的防呆:values 一定是 dict;t_onset / t_alarm 非有限值 → None。"""
    if not isinstance(f.values, dict):
        f.values = {} if f.values is None else {"value": f.values}
    for k in ("t_onset", "t_alarm"):
        v = getattr(f, k)
        if v is not None:
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float("nan")
            setattr(f, k, v if np.isfinite(v) else None)
    return f


def _run_modules(ctx: Ctx, a: Anchors, rep: CrashReport, modules) -> list[Finding]:
    findings: list[Finding] = []
    for m in (MODULES if modules is None else modules):
        name = m if isinstance(m, str) else getattr(m, "__name__", type(m).__name__)
        if isinstance(m, str):
            try:
                m = importlib.import_module(f"core.crash.{m}")
            except ModuleNotFoundError as e:
                if e.name == f"core.crash.{name}":
                    rep.missing_modules.append(name)
                    continue
                rep.failed_modules.append(name)          # 模組在,但它 import 的東西不在
                rep.notes.append(f"偵測模組 {name} 載入失敗:{type(e).__name__}: {e}")
                findings.append(_module_error(name, None, e, "載入"))
                continue
            except Exception as e:  # noqa: BLE001 — SyntaxError / NameError 等:記一筆,其餘照跑
                rep.failed_modules.append(name)
                rep.notes.append(f"偵測模組 {name} 載入失敗:{type(e).__name__}: {e}")
                findings.append(_module_error(name, None, e, "載入"))
                continue
        try:
            out = m.run(ctx, a)
            if isinstance(out, Finding):
                out = [out]
            findings.extend(_normalise(f) for f in (out or []) if isinstance(f, Finding))
        except Exception as e:  # noqa: BLE001 — 模組整個壞掉:記一筆,其餘照跑
            rep.failed_modules.append(name)
            rep.notes.append(f"偵測模組 {name} 執行失敗:{type(e).__name__}: {e}")
            findings.append(_module_error(name, getattr(m, "DOMAIN", None), e, "執行"))
    return findings


def _fallback_end_type(ctx: Ctx, a: Anchors) -> str:
    """end_state 模組沒有定案(缺模組 / 失敗)時,只用 ARM / EV 表推結束型態。"""
    iv = a.interval
    if a.t_detect is not None or iv.method == 12:
        return "crash_check"
    if iv.open_end:
        return "log_ended_armed"
    if iv.method is not None and iv.method not in NORMAL_DISARM_METHODS:
        return "abnormal_disarm"
    if ctx.landed_at(iv.t1 - 0.002, iv):
        return "normal_landing"
    return "unknown"


def _data_ok(ctx: Ctx, a: Anchors) -> bool:
    have = sum(1 for alts in CORE_TABLES if any(ctx.table(n) is not None for n in alts))
    t_end = a.t_disarm if a.t_disarm is not None else a.interval.t1
    return have >= MIN_CORE_TABLES and (t_end - a.interval.t0) >= MIN_ARMED_S


def _interval_summaries(ctx: Ctx) -> list[dict]:
    """每次解鎖一行;abnormal 與自動選區間用同一套判準(interval_abnormal_reasons)。"""
    out = []
    crash_t = _crash_err_times(ctx)
    for iv in ctx.armed:
        why = interval_abnormal_reasons(ctx, iv, crash_t)
        airborne = any("仍認為在空中" in w for w in why)
        if iv.open_end:
            end = "log 在解鎖中結束"
        elif iv.method == 12:
            end = "CRASH_CHECK 自動上鎖(墜機)"
        elif iv.method is None:
            end = "上鎖(原因未記錄)" + (",上鎖時飛控仍認為在空中" if airborne else "")
        elif iv.method in NORMAL_DISARM_METHODS:
            meth = ARM_METHODS.get(iv.method, f"Method {iv.method}")
            end = f"上鎖({meth}),但上鎖時飛控仍認為在空中(沒有 LAND_COMPLETE)" if airborne else f"正常上鎖({meth})"
        else:
            end = f"非一般上鎖(Method {iv.method})"
        out.append({"index": iv.index, "t0": iv.t0, "t1": iv.t1, "method": iv.method, "end": end,
                    "abnormal": bool(why), "abnormal_why": why})
    return out


# ------------------------------------------------------------------ 錨點收尾 / 情境

def consequence_boundary(a: Anchors) -> float | None:
    """「後果」從何時開始(相對秒)。
    有姿態預警:t_loss 與第一次硬撞擊中「不早於 t_warn」的最早者(低空失控可能在 ae 到 30° 之前就先撞地;
               撞擊後的振動 / EKF 垃圾 / 電壓擺盪都是後果);兩者都沒有 → None(預警後到上鎖都算發展)。
    沒有預警:第一次硬撞擊,沒有就飛控 / 偵測器判定失事的時間(t_detect)。"""
    if a.t_warn is not None:
        c = [x for x in (a.t_loss, a.t_first_hit) if x is not None and x >= a.t_warn]
        return min(c) if c else None
    return a.t_first_hit if a.t_first_hit is not None else a.t_detect


def undetermined_end(a: Anchors) -> bool:
    """結束型態不明(沒有上鎖原因 / 落地事件可判斷)且沒有任何事件錨點:無從判斷有沒有失事。"""
    return (a.interval is not None and a.end_type in ("unknown", "")
            and a.t_warn is None and a.t_first_hit is None and a.t_detect is None)


def _no_causality(a: Anchors) -> bool:
    """不貼先兆 / 發展 / 後果標籤的情況:正常結束、未解鎖、結束型態不明且沒有事件錨點。"""
    return a.end_type in NO_FAILURE_END_TYPES or undetermined_end(a)


def _finalize_anchors(ctx: Ctx, a: Anchors, findings: list[Finding]) -> None:
    if _no_causality(a):
        a.t_first_anomaly = None
        a.scenario = ""
        return
    cands = [v for v in (a.t_touch, a.t_warn) if v is not None]
    b = consequence_boundary(a)
    hi = b if b is not None else (a.t_disarm if a.t_disarm is not None else 1e18)
    for f in findings:
        if f.status == HIT and f.t_onset is not None and f.id not in STANDING_IDS and f.domain in (
                "motor", "power", "failsafe", "vibration", "compass") and a.interval.t0 <= f.t_onset < hi:
            cands.append(f.t_onset)
    a.t_first_anomaly = min(cands) if cands else (a.t_first_hit or a.t_detect)
    label, subs, feats = _scenario(ctx, a, findings)
    a.scenario = label
    a.extra["scenario"] = {"sub": subs, **feats}


def _r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _median_in(s, lo: float, hi: float) -> float | None:
    if s is None:
        return None
    m = (s[0] >= lo) & (s[0] <= hi) & np.isfinite(s[1])
    return float(np.median(s[1][m])) if m.any() else None


def _scenario_features(ctx: Ctx, a: Anchors, ref: float) -> dict:
    """型態判定用的數字(全部在撞擊之前取樣):觸地 / 預警時的 agl、觸地前 5 s 垂直速度、下沉誤差、油門、落地狀態、搖桿。"""
    iv = a.interval
    t_feat = a.t_touch if (a.t_touch is not None and a.t_touch <= ref) else ref
    f: dict = {"t_ref": _r(ref), "t_feat": _r(t_feat), "mode": ctx.mode_at(ref) or ""}
    tm, nums, _ = ctx.mode_rows()
    i = int(np.searchsorted(tm, ref, side="right")) - 1 if tm.size else -1
    f["mode_num"] = int(nums[i]) if i >= 0 else None
    alt, g = ctx.series("CTUN.Alt"), ctx.ground_alt(iv)
    if alt is not None and g is not None:
        i = int(np.searchsorted(alt[0], t_feat, side="right")) - 1
        if i >= 0 and alt[0][i] >= iv.t0:
            f["agl_m"] = _r(alt[1][i] - g, 2)
    crt = ctx.series("CTUN.CRt")
    f["vz_pre_cms"] = _r(_median_in(crt, t_feat - SCN_VZ_PRE_S, t_feat), 1)
    dcrt = ctx.series("CTUN.DCRt")
    if crt is not None and dcrt is not None:
        m = (crt[0] >= ref - 1.0) & (crt[0] <= ref)
        if m.any():
            f["sink_err_cms"] = _r(float(np.median(crt[1][m] - np.interp(crt[0][m], dcrt[0], dcrt[1]))), 1)
    f["tho"] = _r(_median_in(ctx.series("CTUN.ThO"), ref - 0.25, ref + 0.5), 3)
    landed = ctx.landed_at(ref, iv)
    f["landed"] = None if landed is None else bool(landed)
    rc = ctx.table("RCIN")
    if rc is not None:
        m = (rc["t"] >= ref - SCN_STICK_WIN_S) & (rc["t"] <= ref)
        best = None
        for ch in (1, 2, 4):
            if f"C{ch}" not in rc or not m.any():
                continue
            d = np.abs(np.asarray(rc[f"C{ch}"][m], dtype=float) - ctx.rc_trim(ch))
            if not np.isfinite(d).any():
                continue
            i = int(np.nanargmax(d))
            if best is None or d[i] > best[0]:
                best = (float(d[i]), float(rc["t"][m][i]), ch)
        if best is not None:
            f["stick_dev_us"] = _r(best[0], 0)
            f["stick_dev_t"] = _r(best[1], 2)
            f["stick_dev_ch"] = best[2]
    return f


def _scenario(ctx: Ctx, a: Anchors, findings: list[Finding]) -> tuple[str, list[str], dict]:
    """dataflow STEP 3:COMMANDED_FLIP > FREE_FALL > GROUND_TIPOVER > HARD_LANDING > AIR_LOSS_OF_CONTROL。
    參考時間 = t_warn,沒有就第一次硬撞擊(不晚於判定);兩者都沒有(例:log 在空中正常飛行時中斷、空中上鎖)時
    只在有刻意動作 / 墜落證據時給型態,否則 UNKNOWN(型態不明)—— 沒有證據不說「空中失去控制」。"""
    anchored = True
    if a.t_warn is not None:
        ref = a.t_warn
    elif a.t_first_hit is not None and (a.t_detect is None or a.t_first_hit <= a.t_detect + EDGE_TOL_S):
        ref = a.t_first_hit
    else:
        ref, anchored = a.t_detect, False
    if ref is None:
        return "UNKNOWN", [], {}
    feats = _scenario_features(ctx, a, ref)
    b = consequence_boundary(a)
    t_loss = b if b is not None else 1e18

    def hits(det, pred=lambda f: True):
        return [f for f in findings if f.id == det and f.status == HIT and pred(f)]

    agl, vz, sink = feats.get("agl_m"), feats.get("vz_pre_cms"), feats.get("sink_err_cms")
    name = feats["mode"].upper()
    named = bool(name) and not name.startswith("MODE ")
    flip = (name in FLIP_MODES if named else feats.get("mode_num") in FLIP_MODE_NUMS)         or bool(ctx.ev_times(EV_FLIP_START, ref - FLIP_EV_WIN_S, ref + FLIP_EV_WIN_S))
    fall = bool(hits("free_fall_run", lambda f: f.t_onset is not None and f.t_onset < ref))
    touch_hit = bool(hits("touchdown_not_recognised"))
    near_ground = (agl is not None and agl <= SCN_AGL_GROUND_M) or (agl is None and touch_hit)
    if flip:
        label = "COMMANDED_FLIP"
    elif fall or (vz is not None and vz <= SCN_VZ_FREEFALL_CMS) or (sink is not None and sink <= SCN_SINK_ERR_CMS):
        label = "FREE_FALL"
    elif not anchored:
        label = "UNKNOWN"
    elif near_ground:
        label = "GROUND_TIPOVER" if vz is None or vz >= SCN_VZ_HARD_CMS else "HARD_LANDING"
    else:
        label = "AIR_LOSS_OF_CONTROL"

    subs: list[str] = []
    def before_loss(f):
        return f.t_onset is not None and f.t_onset < t_loss

    lean = hits("lean_without_pilot", before_loss)
    pilot = hits("pilot_input_context", lambda f: f.t_onset is not None and f.t_onset < ref)
    if label in ("GROUND_TIPOVER", "HARD_LANDING"):
        auto = [f for f in lean if hyp._num(f.values.get("lean_peak"), 0.0) >= SCN_LEAN_AUTOPILOT_DEG]
        if auto:
            subs.append("autopilot_lean")
            thr = auto[0].values.get("thr")
            tol = hyp._opt(thr.get("stick_us")) if isinstance(thr, dict) else None
            if tol is not None:
                feats["lean_stick_tol_us"] = tol        # lean 偵測器的「搖桿置中」容許偏移(傾斜期間都在這以內)
        if pilot:
            subs.append("pilot_stick")
        if feats.get("landed") is False and feats.get("tho") is not None and feats["tho"] <= SCN_THO_IDLE:
            subs.append("land_not_declared")
    elif label == "AIR_LOSS_OF_CONTROL":
        if hits("motor_thrust_loss_check") or hits("rate_output_saturation", lambda f: f.context == "air" and before_loss(f)):
            subs.append("motor_saturation")
        if any(f.context == "air" for f in lean):
            subs.append("attitude_cmd_runaway")
        if pilot:
            subs.append("pilot_stick")
        if not subs:
            subs.append("unknown")
    return label, subs, feats


# ------------------------------------------------------------------ 因果標記

def tag_time(t: float | None, a: Anchors, standing: bool = False) -> str:
    """依時間貼因果標籤:t < t_warn 先兆;t_warn ≤ t < 後果邊界(consequence_boundary:失控或第一次硬撞擊,取早者)
    發展;之後後果;上鎖後 60 s 內事後。正常結束的飛行不貼「先兆 / 後果」(沒有失事可言):飛行中 ''、上鎖後 ground_info。"""
    if standing or t is None:
        return TAG_STANDING
    iv = a.interval
    if iv is None:
        return TAG_GROUND
    normal = _no_causality(a)
    if a.t_disarm is not None and t > a.t_disarm:
        if normal:
            return TAG_GROUND
        return TAG_POST if t <= a.t_disarm + POST_WINDOW_S else TAG_GROUND
    if t < iv.t0 - EDGE_TOL_S or t > iv.t1 + EDGE_TOL_S:
        return TAG_GROUND
    if normal:
        return ""
    b = consequence_boundary(a)
    if a.t_warn is not None:
        if t < a.t_warn:
            return TAG_PRECURSOR
        return TAG_DEVELOPMENT if b is None or t < b else TAG_CONSEQUENCE
    if b is not None:
        return TAG_PRECURSOR if t < b else TAG_CONSEQUENCE
    return TAG_PRECURSOR


# 觸發 / 反應型證據:只有「先於姿態異常」才可能是原因;在姿態異常發展中才出現的是反應或連帶現象
TRIGGER_IDS = frozenset({"pilot_input_context", "source_switch_audit", "inflight_failsafe_or_mode_change",
                         "ekf_mag_reject", "compass_raw_consistency", "compass_sensor_health",
                         "ekf_yaw_reset_lane_switch", "vibe_clip_transient"})


# 「結束事件本身」型的上鎖分類:沒有姿態預警時,參考時間(t_detect)就是這次上鎖 / log 結束,事件本身是起點不是後果
END_EVENT_CLASSES = frozenset({"manual_disarm_in_air", "failsafe_disarm", "log_ended_armed", "emergency_stop_in_air"})


def _late_tag(t: float | None, a: Anchors) -> str | None:
    """持續狀態型偵測器的 t_onset 落在後果邊界之後 / 上鎖之後 → 後果 / 事後(撞擊後才出現的「持續狀態」不是風險);
    否則 None(維持 standing)。正常結束不適用。"""
    if t is None or _no_causality(a) or a.interval is None:
        return None
    if a.t_disarm is not None and t > a.t_disarm + EDGE_TOL_S:
        return tag_time(t, a)
    b = consequence_boundary(a)
    if b is not None and t >= b and t >= a.interval.t0:
        return TAG_CONSEQUENCE
    return None


def _tag_findings(ctx: Ctx | None, a: Anchors, findings: list[Finding]) -> None:
    ref = consequence_boundary(a) if a.t_warn is None else None
    for f in findings:
        if f.id in STANDING_IDS or f.context == "standing":
            f.tag = _late_tag(f.t_onset, a) or TAG_STANDING
            continue
        f.tag = tag_time(f.t_onset, a)
        if f.tag == TAG_DEVELOPMENT and f.id in TRIGGER_IDS:
            f.tag = TAG_CONSEQUENCE
        elif (f.tag == TAG_CONSEQUENCE and a.t_warn is None and f.id == "abnormal_disarm_classifier"
              and f.status in (HIT, SOFT) and hyp._cls(f) in END_EVENT_CLASSES
              and f.t_onset is not None and ref is not None and abs(f.t_onset - ref) <= EDGE_TOL_S):
            f.tag = TAG_PRECURSOR


# ------------------------------------------------------------------ 已排除的信心 / 事件鏈 / 其他觀察

def _cap_rule_out(ctx: Ctx, findings: list[Finding], scores: list) -> None:
    """資料品質限制「排除」的力度:電流計無效 → 電源排除最多「中」;沒有電調遙測 → 馬達排除最多「中」。"""
    caps: dict[str, str] = {}
    if any(f.id == "power_data_quality" and f.status in (HIT, SOFT) for f in findings):
        caps["H_POWER"] = "電流計讀值無效(BAT.Curr),電流類故障(接頭 / 電調過流)無法排除"
    if not (ctx.has("ESC.RPM") or ctx.table("ESC") is not None):
        caps["H_MOTOR_THRUST"] = "沒有電調遙測(ESC RPM / 電流),指令正常但實際失效的馬達無法排除"
    for h in scores:
        if h.status == "ruled_out" and h.id in caps:
            h.rule_out_conf = "中"
            h.rule_out_note = ";".join(x for x in (h.rule_out_note, caps[h.id]) if x)


def _first_sentence(s: str) -> str:
    """說明文字的第一句(到第一個「。」為止;過長時截斷)。"""
    s = s.split("。")[0]
    return s if len(s) <= 140 else s[:138] + "…"


_CHAIN_ROLE = {"ranked": "top", "contributing": "contributing", "candidate": "candidate"}


def _chain(scores: list) -> list[dict]:
    """事件鏈:第一名與貢獻因素中「先兆 / 發展」的證據,依 t_onset 排序;持續狀態另列(風險,不是觸發)。
    證據不足時沒有第一名:列候選假說的證據(role candidate),只是「已知證據」不是機制結論。"""
    out, seen = [], set()
    for h in scores:
        if h.status not in _CHAIN_ROLE or h.score <= 0:
            continue
        role = _CHAIN_ROLE[h.status]
        for e in h.evidence:
            if e["contrib"] <= 0 or not e.get("text_zh"):
                continue
            k = (e["det"], e["t_onset"])
            if k in seen:
                continue
            seen.add(k)
            out.append({"t": e["t_onset"], "det": e["det"], "tag": e["tag"], "role": role, "hyp": h.id,
                        "text_zh": _first_sentence(e["text_zh"])})
    timed = sorted((c for c in out if c["tag"] != TAG_STANDING and c["t"] is not None), key=lambda c: c["t"])
    return timed + [c for c in out if c["tag"] == TAG_STANDING or c["t"] is None]


def _observations(findings: list[Finding], scores: list, health: bool) -> list[dict]:
    """失事:沒有計入任何假說分數的持續狀態 / 保養項目(非本次失事原因);正常飛行:所有 HIT / SOFT(健康觀察)。"""
    used = {(e["det"], e["t_onset"]) for h in scores if h.score > 0 for e in h.evidence if e["contrib"] > 0}
    out, seen = [], set()
    for f in findings:
        if f.id in LIMITATION_IDS or not f.text_zh or f.id.startswith(MODULE_ERROR_PREFIX):
            continue
        if health:
            keep = f.status in (HIT, SOFT) or f.context == "hygiene"
        else:
            keep = ((f.status in (HIT, SOFT) and f.tag == TAG_STANDING and (f.id, f.t_onset) not in used)
                    or (f.context == "hygiene" and f.status != NA))
        if keep and f.text_zh not in seen:
            seen.add(f.text_zh)
            out.append({"id": f.id, "domain": f.domain, "status": f.status, "tag": f.tag, "t_onset": f.t_onset,
                        "text_zh": f.text_zh})
    return out


def _na_domains(findings: list[Finding]) -> list[dict]:
    """每個領域一行:第一筆 NA 說明,加上同領域其他 NA 說明多出來的部分(例如 log 尾端拔電池)。"""
    out: dict[str, dict] = {}
    for f in findings:
        if f.status != NA or not f.text_zh:
            continue
        d = out.get(f.domain)
        if d is None:
            out[f.domain] = {"domain": f.domain, "domain_zh": DOMAIN_ZH.get(f.domain, f.domain), "text_zh": f.text_zh,
                             "_base": f.text_zh}
            continue
        extra = f.text_zh[len(d["_base"]):] if f.text_zh.startswith(d["_base"]) else f.text_zh
        if extra and extra not in d["text_zh"]:
            d["text_zh"] += extra
    return [{k: v for k, v in d.items() if k != "_base"} for d in out.values()]


def _bench_power_notes(ctx: Ctx) -> list[str]:
    """未解鎖 log 的供電資訊:USB 供電(BAT.Volt≈0)→ 電池分析不適用;log 尾端拔電池 → 不是掉電。"""
    s = ctx.series("BAT.Volt")
    if s is None or s[0].size < 10:
        return []
    med = float(np.nanmedian(s[1]))
    if med < BENCH_BATTERY_PRESENT_V:
        return [f"電池分析不適用:這份 log 是 USB 供電 / 沒接飛行電池(BAT.Volt 中位數 {med:.2f} V)。"]
    try:
        from .det_power import unplug_tail
        tail = unplug_tail(s[0], s[1])
    except Exception:  # noqa: BLE001 — 只是資訊,電源模組有問題就不寫
        tail = None
    if tail:
        return [f"log 尾端 {tail['t_start']:.1f} s 起電壓 {tail['v_before']:.1f} → {tail['v_end']:.1f} V 是拔電池"
                f"(未解鎖、{tail['tail_len_s']:.1f} s 內),不是電壓崩落。"]
    return []


# ------------------------------------------------------------------ 時間軸

def _msg_rows(ctx: Ctx) -> list[tuple[float, str]]:
    """真正的 STATUSTEXT(MSG 表);ctx.messages 混有 EV / ERR / MODE / ARM 的文字,只在沒有 MSG 表時備援。"""
    tb = ctx.table("MSG")
    if tb is not None and "Message" in tb:
        return [(float(t), str(m)) for t, m in zip(tb["t"], tb["Message"])]
    return list(ctx.messages)


def _timeline(ctx: Ctx, a: Anchors, findings: list[Finding]) -> list[TimelineEvent]:
    iv = a.interval
    if _no_causality(a):
        lo, hi = max(iv.t0, iv.t1 - NORMAL_TAIL_S), iv.t1 + NORMAL_POST_S
    else:
        lo = max(iv.t0, (a.t_first_anomaly if a.t_first_anomaly is not None else iv.t0) - PRE_ANOMALY_S)
        hi = (a.t_disarm if a.t_disarm is not None else iv.t1) + POST_WINDOW_S
    t_dis = a.t_disarm if a.t_disarm is not None else 1e18
    ev: list[TimelineEvent] = []

    def add(t, cat, label, src):
        if lo <= t <= hi:
            ev.append(TimelineEvent(float(t), cat, label, tag_time(t, a), src))

    t, sub, code = ctx.err_rows()
    for tt, s, c in zip(t, sub, code):
        add(tt, "err", err_text(int(s), int(c)), "ERR")
    t, ids = ctx.ev_rows()
    for tt, i in zip(t, ids):
        # 10/11 由 ARM 表表示;上鎖那一刻 1 ms 內補寫的 EV17/18/56 是上鎖造成的,不是落地
        if i in (10, 11) or (tt >= t_dis - 0.001 and tt <= t_dis + 0.01 and i in (17, 18, 56)):
            continue
        add(tt, "ev", ev_text(int(i)), "EV")
    for tt, text in _msg_rows(ctx):
        if _MSG_KEEP.search(text) and not _MSG_DROP.search(text):
            add(tt, "msg", text, "MSG")
    tm, nums, names = ctx.mode_rows()
    for tt, n in zip(tm, names):
        add(tt, "mode", f"模式 → {n}", "MODE")
    add(iv.t0, "arm", "解鎖", "ARM")
    if not iv.open_end:
        meth = ARM_METHODS.get(iv.method, f"Method {iv.method}") if iv.method is not None else "原因未記錄"
        add(iv.t1, "arm", f"上鎖({'CRASH_CHECK' if iv.method == 12 else meth})", "ARM")
    for name, label in (("t_touch", "觸地"), ("t_warn", "姿態誤差 > 10°"), ("t_loss", "姿態誤差 > 30°(失控)"),
                        ("t_first_hit", "第一次硬撞擊"), ("t_impact_peak", "最後一次撞擊(峰值)"), ("t_rest", "機身靜止"),
                        ("t_detect", "飛控 / 偵測器判定失事")):
        v = getattr(a, name)
        if v is not None:
            ev.append(TimelineEvent(float(v), "derived", label, tag_time(v, a), "analysis"))
    for f in findings:
        # 持續狀態不列(沒有時間點);持續狀態型偵測器若起點落在後果之後已被改標後果 / 事後 → 照常列出
        if f.status == HIT and f.t_onset is not None and f.tag != TAG_STANDING \
                and f.text_zh and lo <= f.t_onset <= hi:
            ev.append(TimelineEvent(float(f.t_onset), "derived", _first_sentence(f.text_zh), f.tag, f.id))
    ev.extend(_pilot_rows(ctx, a, findings, lo, hi))
    ev.extend(_vio_rows(ctx, a, findings, lo, hi))
    ev.sort(key=lambda e: (e.t, e.category))
    out, seen = [], set()
    for e in ev:                      # 去重(同 10 ms 同標籤)
        k = (round(e.t, 2), e.label)
        if k not in seen:
            seen.add(k)
            e.t_abs = round(e.t + ctx.t_base, 6)
            out.append(e)
    return out


# ---- 時間軸的飛手 / VIO 衍生事件(規格 event_causality_tagging 的 derived events) ----
PILOT_EVENT_ZH = {"A_full_stick": "飛手把搖桿打到底", "B_fast_move": "飛手快速撥動搖桿",
                  "C_source_switch_flipback": "飛手來回撥動 EKF 來源組開關"}
RC_SRC_OPTION = 90             # RCn_OPTION 90 = EKF 來源組開關(124:RC9 1095 / 1515 / 1935 µs = 第 1 / 2 / 3 組)
AUX_LOW_US, AUX_HIGH_US = 1200.0, 1800.0   # ArduPilot 三段開關門檻(< 1200 低、> 1800 高;124 的三檔在 1095 / 1515 / 1935)
RC_VALID_MIN_US = 800.0        # RCIN < 800 µs = 沒有訊號(0),不算切換
THR_MIN_TOL_US = 20.0          # 油門搖桿 ≤ RC3_MIN + 20 µs = 收到最低(規格的 throttle-min 容許值)
THR_MIN_REARM_S = 1.0          # 離開最低不到 1 s 又回到最低 → 不重複列(124:1257.04 短暫推到 1138 又收回)
VIO_GAP_S = 1.0                # VISP 間隔 ≥ 1.0 s 列為 VIO 中斷(規格的 VIO gap marker;正常串流 ~27 Hz)
VIO_DUP_S = 0.3                # 與 vio_dropout 偵測結果起點相差 ≤ 0.3 s → 已在時間軸,不重複列


def _row(ctx_t: float, cat: str, label: str, tag: str, src: str) -> TimelineEvent:
    return TimelineEvent(float(ctx_t), cat, label, tag, src)


def _pilot_rows(ctx: Ctx, a: Anchors, findings: list[Finding], lo: float, hi: float) -> list[TimelineEvent]:
    """飛手相關:EKF 來源組開關(RCn_OPTION=90)的每次切換、油門搖桿收到最低、pilot_input_context 的各段動作。
    觸發型(搖桿 / 開關)在發展期出現的標「後果」(飛手反應),和 pilot_input_context 本身的標籤規則一致。"""
    out: list[TimelineEvent] = []

    def trig_tag(t):
        tg = tag_time(t, a)
        return TAG_CONSEQUENCE if tg == TAG_DEVELOPMENT else tg

    rc = ctx.table("RCIN")
    src_chans: set[int] = set()
    if rc is not None and rc["t"].size >= 2:
        t = rc["t"]
        for n in range(1, 17):
            opt = ctx.param(f"RC{n}_OPTION", None)
            if opt is None or int(opt) != RC_SRC_OPTION or f"C{n}" not in rc:
                continue
            src_chans.add(n)
            v = np.asarray(rc[f"C{n}"], dtype=float)
            ok = np.isfinite(v) & (v >= RC_VALID_MIN_US)
            pos = np.where(v < AUX_LOW_US, 1, np.where(v > AUX_HIGH_US, 3, 2))
            for i in np.flatnonzero((pos[1:] != pos[:-1]) & ok[1:] & ok[:-1]) + 1:
                if lo <= t[i] <= hi:
                    out.append(_row(t[i], "pilot", f"RC{n} 來源組開關 → 第 {int(pos[i])} 組({v[i]:.0f} µs)",
                                    trig_tag(float(t[i])), f"RCIN.C{n}"))
        if "C3" in rc:
            v = np.asarray(rc["C3"], dtype=float)
            low = np.isfinite(v) & (v >= RC_VALID_MIN_US) & (v <= float(ctx.param("RC3_MIN", 1100.0)) + THR_MIN_TOL_US)
            starts = np.flatnonzero(low & ~np.r_[False, low[:-1]])
            ends = np.flatnonzero(low & ~np.r_[low[1:], False])
            for k, s in enumerate(starts):
                if k > 0 and t[s] - t[ends[k - 1]] < THR_MIN_REARM_S:
                    continue
                if lo <= t[s] <= hi:
                    out.append(_row(t[s], "pilot", f"油門搖桿收到最低(C3 {v[s]:.0f} µs)", tag_time(float(t[s]), a), "RCIN.C3"))
    for f in findings:
        if f.id != "pilot_input_context" or f.status not in (HIT, SOFT):
            continue
        for e in f.values.get("events") or []:
            if not isinstance(e, dict):
                continue
            t0, t1, kind = hyp._opt(e.get("t")), hyp._opt(e.get("t_end")), str(e.get("kind", ""))
            if t0 is None or not lo <= t0 <= hi:
                continue
            if f.t_onset is not None and abs(t0 - f.t_onset) <= EDGE_TOL_S:
                continue              # 和偵測結果本身那一列同一時刻
            if kind.startswith("C_") and src_chans:
                continue              # 開關的每次切換已逐筆列出
            span = f"({t0:.2f}–{t1:.2f} s)" if t1 is not None and t1 > t0 else ""
            out.append(_row(t0, "pilot", PILOT_EVENT_ZH.get(kind, kind) + span, trig_tag(t0), "pilot_input_context"))
    return out


def _vio_rows(ctx: Ctx, a: Anchors, findings: list[Finding], lo: float, hi: float) -> list[TimelineEvent]:
    """VIO 串流:≥ 1 s 的中斷、上鎖前就停止的最後一筆;已由 vio_dropout 偵測結果列出的不重複。"""
    vp = ctx.table("VISP")
    if vp is None or vp["t"].size < 2:
        return []
    t = vp["t"]
    dups = [f.t_onset for f in findings if f.id == "vio_dropout" and f.status in (HIT, SOFT) and f.t_onset is not None]

    def dup(x):
        return any(abs(x - d) <= VIO_DUP_S for d in dups)

    out: list[TimelineEvent] = []
    g = np.diff(t)
    for i in np.flatnonzero(g >= VIO_GAP_S):
        if lo <= t[i] <= hi and not dup(float(t[i])):
            out.append(_row(t[i], "vio", f"VIO 資料中斷 {g[i]:.2f} s({t[i]:.2f}→{t[i + 1]:.2f} s)",
                            tag_time(float(t[i]), a), "VISP"))
    t_last = float(t[-1])
    t_end = a.t_disarm if a.t_disarm is not None else a.interval.t1
    if lo <= t_last <= hi and t_last < t_end - VIO_GAP_S and not dup(t_last):
        out.append(_row(t_last, "vio", "VIO 最後一筆資料(之後到上鎖 / log 結束都沒有 VIO)", tag_time(t_last, a), "VISP"))
    return out


def _ground_events(ctx: Ctx) -> list[TimelineEvent]:
    """未解鎖期間(或未解鎖 log)的故障 / 失效保護 / 來源切換:只當資訊列出,不參與原因分析。"""
    out = []
    t, sub, code = ctx.err_rows()
    for tt, s, c in zip(t, sub, code):
        if ctx.armed_at(float(tt)) is None:
            out.append(TimelineEvent(float(tt), "err", err_text(int(s), int(c)), TAG_GROUND, "ERR"))
    t, ids = ctx.ev_rows()
    for tt, i in zip(t, ids):
        if i not in (10, 11, 17, 18, 28, 56) and ctx.armed_at(float(tt)) is None:
            out.append(TimelineEvent(float(tt), "ev", ev_text(int(i)), TAG_GROUND, "EV"))
    out.sort(key=lambda e: e.t)
    out = out[:200]
    for e in out:
        e.t_abs = round(e.t + ctx.t_base, 6)
    return out


# ------------------------------------------------------------------ 文字摘要

def _finish_text(rep: CrashReport, ctx: Ctx | None = None, findings: list[Finding] | None = None) -> None:
    from .report import build_limitations, build_recommendations, build_summary
    rep.summary_zh = build_summary(rep)
    rep.recommendations = build_recommendations(rep)
    rep.limitations = build_limitations(rep, ctx, findings or [])
