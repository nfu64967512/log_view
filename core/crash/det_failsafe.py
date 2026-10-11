"""失效保護偵測模組(領域 failsafe):解鎖飛行中有沒有失效保護被觸發、或飛控被動切換了模式。

偵測器(1 個):
  inflight_failsafe_or_mode_change
      只看「解鎖且未落地(armed & ~L)」的時間:解鎖列 t0 ≤ t ≤ 上鎖列 t1(上鎖列之後的列一律不算飛行中,
      放 values['post_disarm'];落地狀態 L 由 EV 28 / 18 階梯決定,上鎖當下 1 ms 內補寫的 EV 17 / 18 不算):
        (a) MODE 表真的換了模式(ModeNum 變了;解鎖時重記的同模式列不算),依 MODE.Rsn(ModeReason)分類:
            失效保護原因(3 遙控 / 4 電池 / 5 GCS / 6 EKF / 7 GPS glitch / 10 圍欄 / 11 地形 / 19 墜機 / 25 一般失效保護
            / 29 漏水 / 50 航位推算 / 51 起飛失效保護)→ HIT;
            飛手 / 指令(1 遙控 / 2 GCS / 9 推油門脫離 LAND / 32 腳本 / 42 FrSky / 52 DDS / 53 AUX 開關)與一般自動切換
            (任務結束、煞車逾時、翻滾結束…)→ 不算證據;
            其他有名稱的自動原因(避障、TERMINATE…)→ SOFT「被動切換模式」;Rsn 0 / 表外編號 → SOFT「原因不明的模式切換」;
            MODE 表沒有 Rsn 欄 → 前後 1 s 內有 ERR 失效保護就併入,否則 SOFT「原因未記錄的模式切換」(不能當 CLEAR);
        (b) ERR 子系統 5/6/7/8/9/16/17/20/23/25/26/27/29/31 且 ECode != 0(0 = 解除)→ HIT;
            觸發 / 解除成對算持續時間;EKFCHECK(16)與 FAILSAFE_EKFINAV(17)合併成一段;
            ERR 10(想切模式但被拒)→ SOFT;
        (c) EV 54 MOTORS_EMERGENCY_STOPPED → HIT(EV 55 解除);
        (d) 只有 MSG 文字出現 failsafe、附近(±2 s)沒有上述 ERR / 模式切換 → SOFT。
      因果標籤不自己算:每個觸發的 phase 直接呼叫 engine.tag_time(t, a)(延遲 import,engine 只用 importlib 載入
      det_*,沒有循環),所以說明文字與 engine 的計分永遠一致;lead_to_warn_s = 參考時間 − t(參考時間與 tag_time 相同:
      t_warn,沒有就 t_first_hit / t_detect;values['lead_ref'] 記用了哪一個,文字寫出對應的「姿態異常 / 第一次撞擊 /
      飛控判定失事」)。engine 會把觸發型證據的 development 改標 consequence(只有早於姿態異常才算原因)。
      代表觸發(決定 Finding.status / t_onset):先挑 precursor 的,再挑 HIT 層,最後挑最早的 ——「早的 SOFT
      + 失控後才出現的 HIT」以 SOFT 代表,不會讓失控後的失效保護蓋掉真正的前兆。
      values['triggers'] 與 Finding 的因果標籤同一層:有 precursor 證據時只列 precursor 的觸發,較晚的(development /
      consequence)移到 values['late_triggers'] —— 假說評分會掃 triggers 找 ERR 6 / ERR 29 等子系統,後果不能混進去。
      與規格的差異:規格 (a) 只看 t_loss 前 10 s 內的模式切換;這裡飛行中全部列出(每筆附 lead_to_loss_s),
      離 t_warn 多遠由 engine 的 prox 降權(≤ 60 s 1.0、≤ 300 s 0.6、更早 0.3)—— 例如電池失效保護 2 分鐘前切 RTL
      也是失事脈絡,不該因為早於 10 s 視窗就變成 CLEAR。
      落地期間(解鎖但未離地 / 降落後)與未解鎖期間(地面)的事件不算證據:只放進 values['landed_events'] /
      values['ground_events'];上鎖後 60 s 內的事件放 values['post_disarm'](墜機的後果)。
      GPS glitch(ERR 11)不是失效保護:放 values['info_events'](說明用)。
      MODE 表缺失:仍檢查 ERR / EV / MSG(CLEAR),但文字寫「無法檢查模式切換」,values['mode_check'] = 'no_mode_table'。

信心:low(HIT / SOFT)—— 這批 log 只有 1 次失事且失事前沒有任何失效保護 / 模式切換,所有「有事件」的分支沒有真實飛行樣本
(只有合成資料測試與 118 / 123 / 124 的地面事件);CLEAR 的信心 medium(看的是「ERR / MODE / EV 表裡沒有列」,
表本身完整;MODE 表缺失時 low)。電池失效保護動作參數 = 0 時 ArduPilot 可能不留 ERR 6,所以「沒有失效保護」不等於
「電壓沒問題」,電壓請看電源領域的偵測器。ModeReason 編號依 ArduPilot libraries/AP_Vehicle/ModeReason.h(0–55;
開機列 Rsn = 26 INITIALISED 與 log 吻合),Rsn 9 / 25 / 32 / 50 / 53 等沒有真實樣本。
"""
from __future__ import annotations

import re

import numpy as np

from core.dflog import err_text

from .ctx import Ctx
from .guard import guarded
from .types import (CLEAR, HIT, NA, SOFT, TAG_CONSEQUENCE, TAG_DEVELOPMENT, TAG_POST, TAG_PRECURSOR, Anchors,
                    ArmedInterval, Finding)

DOMAIN = "failsafe"
DET_ID = "inflight_failsafe_or_mode_change"

# ---------------------------------------------------------------- 常數(規格值,不是從資料調出來的)

# 失效保護類 ERR 子系統 → (名稱, 中文)。ECode != 0 = 發生,0 = 解除(ArduPilot LogErrorCode FAILSAFE_OCCURRED / RESOLVED)。
# 規格:5/6/7/8/9/16/17/25/26/29/31(目錄)+ 27(crash_chain 的最早異常集合);20 / 23 是同語意的 ADS-B / 地形失效保護(額外加,無樣本)。
FAILSAFE_SUBSYS: dict[int, tuple[str, str]] = {
    5: ("FAILSAFE_RADIO", "遙控訊號失效保護"),
    6: ("FAILSAFE_BATT", "電池失效保護"),
    7: ("FAILSAFE_GPS", "GPS 失效保護"),
    8: ("FAILSAFE_GCS", "地面站(GCS)失效保護"),
    9: ("FAILSAFE_FENCE", "地理圍欄失效保護"),
    16: ("EKFCHECK", "EKF 失效保護"),
    17: ("FAILSAFE_EKFINAV", "EKF 失效保護"),
    20: ("FAILSAFE_ADSB", "ADS-B 失效保護"),
    23: ("FAILSAFE_TERRAIN", "地形失效保護"),
    25: ("THRUST_LOSS_CHECK", "推力損失檢查"),
    26: ("FAILSAFE_SENSORS", "感測器失效保護"),
    27: ("FAILSAFE_LEAK", "漏水失效保護"),
    29: ("FAILSAFE_VIBE", "振動失效保護"),
    31: ("FAILSAFE_DEADRECKON", "航位推算失效保護"),
}
ERR_FAMILY = {17: 16}              # EKFCHECK(16)+ FAILSAFE_EKFINAV(17)同一次 EKF 失效保護,合併成一段
ERR_GPS_GLITCH = 11                # GPS glitch:不是失效保護,只當說明資訊
ERR_MODE_REFUSED = 10              # FLIGHT_MODE:想切換模式但被拒(ECode = 目標模式)→ SOFT
EV_EMERGENCY_STOP, EV_EMERGENCY_CLEAR = 54, 55

# MODE.Rsn(ModeReason;ArduPilot libraries/AP_Vehicle/ModeReason.h,各機型共用、只往後加)
MODE_REASON_NAMES: dict[int, str] = {
    0: "UNKNOWN", 1: "RC_COMMAND", 2: "GCS_COMMAND", 3: "RADIO_FAILSAFE", 4: "BATTERY_FAILSAFE", 5: "GCS_FAILSAFE",
    6: "EKF_FAILSAFE", 7: "GPS_GLITCH", 8: "MISSION_END", 9: "THROTTLE_LAND_ESCAPE", 10: "FENCE_BREACHED",
    11: "TERRAIN_FAILSAFE", 12: "BRAKE_TIMEOUT", 13: "FLIP_COMPLETE", 14: "AVOIDANCE", 15: "AVOIDANCE_RECOVERY",
    16: "THROW_COMPLETE", 17: "TERMINATE", 18: "TOY_MODE", 19: "CRASH_FAILSAFE",
    20: "SOARING_FBW_B_WITH_MOTOR_RUNNING", 21: "SOARING_THERMAL_DETECTED", 22: "SOARING_THERMAL_ESTIMATE_DETERIORATED",
    23: "VTOL_FAILED_TRANSITION", 24: "VTOL_FAILED_TAKEOFF", 25: "FAILSAFE", 26: "INITIALISED", 27: "SURFACE_COMPLETE",
    28: "BAD_DEPTH", 29: "LEAK_FAILSAFE", 30: "SERVOTEST", 31: "STARTUP", 32: "SCRIPTING", 33: "UNAVAILABLE",
    34: "AUTOROTATION_START", 35: "AUTOROTATION_BAILOUT", 36: "SOARING_ALT_TOO_HIGH", 37: "SOARING_ALT_TOO_LOW",
    38: "SOARING_DRIFT_EXCEEDED", 39: "RTL_COMPLETE_SWITCHING_TO_VTOL_LAND_RTL",
    40: "RTL_COMPLETE_SWITCHING_TO_FIXEDWING_AUTOLAND", 41: "MISSION_CMD", 42: "FRSKY_COMMAND",
    43: "FENCE_RETURN_PREVIOUS_MODE", 44: "QRTL_INSTEAD_OF_RTL", 45: "AUTO_RTL_EXIT", 46: "LOITER_ALT_REACHED_QLAND",
    47: "LOITER_ALT_IN_VTOL", 48: "RADIO_FAILSAFE_RECOVERY", 49: "QLAND_INSTEAD_OF_RTL", 50: "DEADRECKON_FAILSAFE",
    51: "MODE_TAKEOFF_FAILSAFE", 52: "DDS_COMMAND", 53: "AUX_FUNCTION", 54: "FIXED_WING_AUTOLAND", 55: "FENCE_REENABLE",
}
# 規格 {3,4,5,6,7,11,19};10 圍欄是同類(ERR 9 在規格內);25 FAILSAFE(一般失效保護)、29 漏水、50 航位推算、51 起飛失效保護同義
FAILSAFE_MODE_REASONS = frozenset({3, 4, 5, 6, 7, 10, 11, 19, 25, 29, 50, 51})
# 飛手 / 指令造成的切換(正常操作,不是被動切換):遙控 / GCS / 推油門脫離 LAND / 腳本 / FrSky / DDS / AUX 開關
PILOT_MODE_REASONS = frozenset({1, 2, 9, 32, 42, 52, 53})
# 一般自動切換:任務結束 / 煞車逾時 / 翻滾結束 / 拋飛結束 / 玩具模式 / 開機初始化 / 開機 / 任務指令 /
# 圍欄解除後回原模式 / AUTO_RTL 結束 / 遙控恢復後回原模式(恢復前的失效保護本身另有 Rsn 3 / ERR 5)
BENIGN_AUTO_REASONS = frozenset({8, 12, 13, 16, 18, 26, 31, 41, 43, 45, 48})
# 失效保護模式切換對應的 ERR 子系統(用來把「ERR + 隨後的 MODE 列」合併成同一個觸發)
REASON_TO_SUBSYS: dict[int, frozenset] = {
    3: frozenset({5}), 4: frozenset({6}), 5: frozenset({8}), 6: frozenset({16, 17}), 7: frozenset({7}),
    10: frozenset({9}), 11: frozenset({23}), 19: frozenset(), 25: frozenset(FAILSAFE_SUBSYS), 29: frozenset({27}),
    50: frozenset({31}), 51: frozenset(),
}

MERGE_S = 1.0              # ERR 觸發與失效保護 MODE 列相差 ≤ 此值 → 同一個觸發(實測 ERR 與 MODE 列相差 < 0.1 s)
MSG_MERGE_S = 2.0          # 失效保護 MSG 與 ERR / MODE 觸發相差 ≤ 此值 → 同一個觸發(MSG 只當補充文字)
COMPANION_S = 0.3          # ERR 觸發 ±0.3 s 內的相關 MSG 原文附在觸發上(只是說明文字)
POST_WINDOW_S = 60.0       # 上鎖後仍列為「後果」的視窗(規格:t_disarm + 60 s)
MIN_FLIGHT_S = 5.0         # 離地飛行少於此秒數 → 沒有飛行可評估(NA);實測 124:飛行 1 約 277 s、飛行 2 約 383 s
THO_AIRBORNE = 0.05        # 沒有 EV28 時,CTUN.ThO > 此值才算在飛(實測 124:飛行中 98% 的樣本 > 0.05、中位數 0.15;降落後 / 6 個 baseline 全是 0;
                           # 起飛前的怠速 → 爬升段有 6% 的樣本 > 0.05,約 0.3-0.5 s,遠小於 MIN_FLIGHT_S)
MAX_LISTED = 30            # values 裡每個清單最多列幾筆(其餘只記總數)
# 飛行中 = 解鎖列 t0 ≤ t ≤ 上鎖列 t1,不加容差:CRASH_CHECK 的 ERR 比 ARM 上鎖列早 45 µs(124 實測),失效保護的 ERR / MODE
# 列也都在 disarm() 之前寫出;晚於上鎖列的列(即使只晚幾 ms)是上鎖之後的事,放 post_disarm。

_RX_FS_MSG = re.compile(r"fail-?safe", re.I)
_RX_FS_MSG_SKIP = re.compile(r"^(PreArm|Arm)\b|initiali[sz]ed|clear|recover|resolv|disabled", re.I)
_RX_MSG_COMPANION = re.compile(r"fail-?safe|variance|glitch|battery|radio|fence|terrain", re.I)

# 與失效保護有關的參數(報告用:動作設成 0 時,「沒有失效保護」不代表安全)
FS_PARAMS = ("FS_THR_ENABLE", "FS_GCS_ENABLE", "FS_EKF_ACTION", "FS_DR_ENABLE", "FS_VIBE_ENABLE", "FS_CRASH_CHECK",
             "BATT_FS_LOW_ACT", "BATT_FS_CRT_ACT", "FENCE_ENABLE")

_CRASH_END_TYPES = frozenset({"crash_check", "crash_check_replay", "abnormal_disarm", "hard_impact", "tumble",
                              "log_ended_armed"})
# engine.tag_time 的參考時間順序(t_warn → t_first_hit → t_detect)與文字用的名稱
_REF_NAMES = (("t_warn", "姿態異常"), ("t_first_hit", "第一次撞擊"), ("t_detect", "飛控判定失事"))
_CMD_CLASSES = ("pilot", "benign")
_LATE_TAGS = (TAG_DEVELOPMENT, TAG_CONSEQUENCE, TAG_POST)     # engine 對觸發型證據:這些都不計分


# ---------------------------------------------------------------- 小工具

def _f(x, nd: int = 3):
    return None if x is None else round(float(x), nd)


def _ts(t) -> str:
    return f"{float(t):.2f} s"


def _clean(d: dict) -> dict:
    """丟掉 None 值,讓 values 精簡。"""
    return {k: v for k, v in d.items() if v is not None}


def _txt(x) -> str:
    """MSG / 模式名稱 → str(bytes / np.bytes_ 先解碼,去掉 C 字串尾端的 NUL)。"""
    if isinstance(x, (bytes, bytearray)):
        x = bytes(x).decode("utf-8", "replace")
    return str(x).rstrip("\x00").strip()


# ---------------------------------------------------------------- 事件收集(純函式,吃陣列,方便合成測試)

def err_episodes(t, sub, code, subsys=None, family: dict | None = None) -> list[dict]:
    """ERR 列 → 事件段:ECode != 0 觸發、0 解除;同家族(EKFCHECK + FAILSAFE_EKFINAV)合併。

    回傳 [{'t', 't_end'(None = 到 log 尾都沒解除), 'subsys': [...], 'ecode': {subsys: code}, 'n': 觸發列數}],依 t 排序。
    """
    fam = family or {}
    openeps: dict[int, dict] = {}
    pending: dict[int, set] = {}
    out: list[dict] = []
    for ti, s, c in zip(t, sub, code):
        s, c = int(s), int(c)
        if subsys is not None and s not in subsys:
            continue
        key = fam.get(s, s)
        ep = openeps.get(key)
        if c != 0:
            if ep is None:
                ep = {"t": float(ti), "t_end": None, "subsys": [], "ecode": {}, "n": 0}
                openeps[key] = ep
                pending[key] = set()
                out.append(ep)
            if s not in ep["subsys"]:
                ep["subsys"].append(s)
            ep["ecode"][s] = c
            ep["n"] += 1
            pending[key].add(s)
        elif ep is not None:
            pending[key].discard(s)
            if not pending[key]:
                ep["t_end"] = float(ti)
                del openeps[key], pending[key]
    out.sort(key=lambda e: e["t"])
    return out


def mode_changes(t, nums, rsn, names) -> list[dict]:
    """MODE 表 → 真的換了模式的列(第一列是開機初始化、同模式重記不算)。

    回傳 [{'t', 'from', 'to', 'to_num', 'rsn'(None = 表裡沒有 Rsn 欄)}]。
    """
    out = []
    n = len(t)

    def nm(k):
        return _txt(names[k]) if k < len(names) else f"MODE {int(nums[k])}"

    for i in range(1, n):
        if int(nums[i]) == int(nums[i - 1]):
            continue
        out.append({"t": float(t[i]), "from": nm(i - 1), "to": nm(i), "to_num": int(nums[i]),
                    "rsn": None if rsn is None else int(rsn[i])})
    return out


def ev_stop_episodes(t, ids) -> list[dict]:
    """EV 54(馬達緊急停止)→ 事件段(EV 55 解除)。"""
    out: list[dict] = []
    cur = None
    for ti, i in zip(t, ids):
        i = int(i)
        if i == EV_EMERGENCY_STOP and cur is None:
            cur = {"t": float(ti), "t_end": None}
            out.append(cur)
        elif i == EV_EMERGENCY_CLEAR and cur is not None:
            cur["t_end"] = float(ti)
            cur = None
    return out


def msg_failsafes(t, texts) -> list[dict]:
    """MSG 文字含 failsafe(排除 PreArm / 解除 / 初始化訊息)→ [{'t', 'text'}](bytes 先解碼)。"""
    out = []
    for ti, s in zip(t, texts):
        s = _txt(s)
        if _RX_FS_MSG.search(s) and not _RX_FS_MSG_SKIP.search(s):
            out.append({"t": float(ti), "text": s})
    return out


def reason_class(rsn) -> str:
    """MODE.Rsn → failsafe / pilot / benign / passive / unexplained(Rsn 0 或表外編號)/ unrecorded(表裡沒有 Rsn 欄)。"""
    if rsn is None:
        return "unrecorded"
    if rsn in FAILSAFE_MODE_REASONS:
        return "failsafe"
    if rsn in PILOT_MODE_REASONS:
        return "pilot"
    if rsn in BENIGN_AUTO_REASONS:
        return "benign"
    if rsn == 0 or rsn not in MODE_REASON_NAMES:
        return "unexplained"
    return "passive"


def _mode_label(m: dict) -> str:
    arrow = f"{m['from']}→{m['to']}"
    cls = m["cls"]
    if cls == "failsafe":
        return f"失效保護切換模式 {arrow}(原因 {m['reason']})"
    if cls == "unexplained":
        return f"原因不明的模式切換 {arrow}(Rsn {m['rsn']}{' UNKNOWN' if m['rsn'] == 0 else ''})"
    if cls == "unrecorded":
        return f"原因未記錄的模式切換 {arrow}(MODE 表沒有 Rsn 欄)"
    return f"被動切換模式 {arrow}(原因 {m['reason']})"


def _err_label(ep: dict) -> str:
    names = "/".join(FAILSAFE_SUBSYS[s][0] for s in ep["subsys"] if s in FAILSAFE_SUBSYS)
    zh = FAILSAFE_SUBSYS[ep["subsys"][0]][1] if ep["subsys"] and ep["subsys"][0] in FAILSAFE_SUBSYS else "失效保護"
    return f"{zh} {names}"


def _mode_name_table(ctx: Ctx) -> dict[int, str]:
    """機型的「模式編號 → 名稱」表(ERR 10 的目標模式可能從沒出現在 MODE 表裡);認不出機型回 {}。"""
    mt = getattr(ctx.ix, "mav_type", None)
    if mt is None:
        return {}
    try:
        from core.dflog import mode_mapping
        return {int(k): str(v) for k, v in (mode_mapping(mt) or {}).items()}
    except Exception:  # noqa: BLE001 — 只影響說明文字
        return {}


def collect_events(ctx: Ctx) -> dict:
    """整份 log 的失效保護相關事件(不分是否解鎖),之後再依解鎖區間 / 落地狀態分流。

    回傳 {'triggers': [...], 'glitches': [...], 'mode_changes': [...]};
    每個 trigger: {t, t_end, kind(err/mode/ev/msg), tier(hit/soft), label, ...補充欄位}。
    MSG / 伴隨訊息 / 模式列與 ERR 觸發的配對都用排序後的 np.searchsorted(O((n+m) log n)):上萬筆 ERR + MSG 仍在 1 s 內。
    """
    trig: list[dict] = []

    # (b) ERR 失效保護
    et, esub, ecode = ctx.err_rows()
    err_hits: list[dict] = []
    for ep in err_episodes(et, esub, ecode, subsys=set(FAILSAFE_SUBSYS), family=ERR_FAMILY):
        tr = {"t": ep["t"], "t_end": ep["t_end"], "kind": "err", "tier": "hit", "label": _err_label(ep),
              "subsys": list(ep["subsys"]), "ecode": {str(k): v for k, v in ep["ecode"].items()},
              "_subsys": set(ep["subsys"])}
        trig.append(tr)
        err_hits.append(tr)
    err_t = np.array([tr["t"] for tr in err_hits], dtype=float)          # 已依 t 排序(err_episodes)
    glitches = [{"t": g["t"], "t_end": g["t_end"], "kind": "err", "label": "GPS glitch(ERR 11)"}
                for g in err_episodes(et, esub, ecode, subsys={ERR_GPS_GLITCH})]

    def err_near(t: float, subs: frozenset | None):
        """t ± MERGE_S 內第一個 ERR 失效保護觸發(subs = None:任何子系統)。"""
        lo = int(np.searchsorted(err_t, t - MERGE_S, side="left"))
        hi = int(np.searchsorted(err_t, t + MERGE_S, side="right"))
        for k in range(lo, hi):
            if subs is None or (err_hits[k]["_subsys"] & subs):
                return err_hits[k]
        return None

    # (a) MODE 切換
    tm, nums, names = ctx.mode_rows()
    tb = ctx.table("MODE")
    rsn = tb["Rsn"] if tb is not None and "Rsn" in tb else None
    mchg = mode_changes(tm, nums, rsn, names) if tm.size else []
    name_of = _mode_name_table(ctx)
    if tm.size and len(names) == len(nums):
        name_of.update({int(n): _txt(nm) for n, nm in zip(nums, names)})
    for m in mchg:
        m["reason"] = MODE_REASON_NAMES.get(m["rsn"], f"Rsn {m['rsn']}") if m["rsn"] is not None else None
        m["cls"] = reason_class(m["rsn"])
    # ERR 10:想切模式被拒
    for k in np.flatnonzero(np.asarray(esub) == ERR_MODE_REFUSED):
        tgt = name_of.get(int(ecode[k]), f"模式 {int(ecode[k])}")
        trig.append({"t": float(et[k]), "t_end": None, "kind": "err", "tier": "soft", "subsys": [ERR_MODE_REFUSED],
                     "label": f"模式切換被拒(想切到 {tgt},ERR FLIGHT_MODE)", "_subsys": set()})
    for m in mchg:
        cls = m["cls"]
        if cls in _CMD_CLASSES:
            continue
        if cls in ("failsafe", "unrecorded"):
            # 同一個失效保護:ERR 在前、MODE 列在後 → 補上「它造成的模式」(沒有 Rsn 欄時只能依時間配對)
            merged = err_near(m["t"], REASON_TO_SUBSYS.get(m["rsn"], frozenset()) if cls == "failsafe" else None)
            if merged is not None:
                merged.update({"mode_to": m["to"], "mode_from": m["from"], "mode_reason": m["reason"],
                               "t_mode": m["t"], "rsn": m["rsn"]})
                if cls == "unrecorded":
                    merged["mode_reason_note"] = "MODE 表沒有 Rsn 欄,依時間(±1 s)併入"
                continue
        trig.append({"t": m["t"], "t_end": None, "kind": "mode", "tier": "hit" if cls == "failsafe" else "soft",
                     "label": _mode_label(m), "mode_from": m["from"], "mode_to": m["to"], "mode_reason": m["reason"],
                     "rsn": m["rsn"], "mode_cls": cls, "_subsys": set()})

    # (c) EV 54 / 55
    vt, vid = ctx.ev_rows()
    for ep in ev_stop_episodes(vt, vid):
        trig.append({"t": ep["t"], "t_end": ep["t_end"], "kind": "ev", "tier": "hit",
                     "label": "馬達緊急停止 EV54(MOTORS_EMERGENCY_STOPPED)", "_subsys": set()})

    # (d) MSG 文字
    mt = ctx.table("MSG")
    if mt is not None and "Message" in mt and len(mt["t"]):
        mtt = np.asarray(mt["t"], dtype=float)
        order = np.argsort(mtt, kind="stable")
        mtt = mtt[order]
        texts = [_txt(mt["Message"][i]) for i in order]
        msgs = msg_failsafes(mtt, texts)
        comp = [i for i, s in enumerate(texts) if _RX_MSG_COMPANION.search(s)]
        comp_t = mtt[comp] if comp else np.zeros(0)
        comp_s = [texts[i] for i in comp]
    else:
        msgs, comp_t, comp_s = [], np.zeros(0), []
    for tr in err_hits:                              # 附上同時間的 MSG 原文(窗內最早的一筆)
        k = int(np.searchsorted(comp_t, tr["t"] - COMPANION_S, side="left"))
        if k < comp_t.size and comp_t[k] <= tr["t"] + COMPANION_S:
            tr["msg"] = comp_s[k]
    base = sorted(trig, key=lambda x: x["t"])
    base_t = np.array([x["t"] for x in base], dtype=float)
    last_msg = None                                  # 最近一筆 MSG-only 觸發(連續的 failsafe 訊息併成一筆)
    for g in msgs:
        j = int(np.searchsorted(base_t, g["t"]))
        cand = [k for k in (j - 1, j) if 0 <= k < base_t.size and abs(base_t[k] - g["t"]) <= MSG_MERGE_S]
        if cand:
            k = min(cand, key=lambda k: abs(base_t[k] - g["t"]))
            base[k].setdefault("msg", g["text"])
        elif last_msg is not None and g["t"] - last_msg["t"] <= MSG_MERGE_S:
            continue
        else:
            last_msg = {"t": g["t"], "t_end": None, "kind": "msg", "tier": "soft",
                        "label": f"訊息出現 failsafe:{g['text'][:40]}", "msg": g["text"], "_subsys": set()}
            trig.append(last_msg)
    trig.sort(key=lambda x: x["t"])
    return {"triggers": trig, "glitches": glitches, "mode_changes": mchg}


# ---------------------------------------------------------------- 落地狀態 / 分流

def flight_state(ctx: Ctx, iv: ArmedInterval):
    """回傳 (landed_at(t)→bool, 是否有 NOT_LANDED 紀錄, 離地秒數, 離地秒數的依據)。

    依據 'ev28':EV 28 / 18 階梯;沒有 EV 28(不知道何時離地)時,known = False,離地秒數改用
    'tho'(CTUN.ThO > THO_AIRBORNE 的取樣時間;10 Hz),連 CTUN 都沒有就用 'armed_time'(整段解鎖時間)。
    """
    steps = ctx.landed_steps(iv)
    known = any(not s for _, s in steps[1:])
    ts = np.array([x[0] for x in steps], dtype=float)
    st = np.array([x[1] for x in steps], dtype=bool)

    def landed(t: float) -> bool:
        i = int(np.searchsorted(ts, t, side="right")) - 1
        return bool(st[max(i, 0)])

    if known:
        edges = list(ts) + [iv.t1]
        return landed, True, float(sum(edges[k + 1] - edges[k] for k in range(len(st)) if not st[k])), "ev28"
    tho = ctx.series("CTUN.ThO")
    if tho is not None:
        t, v = tho
        m = (t >= iv.t0) & (t <= iv.t1)
        if m.sum() >= 2:
            dt = float(np.median(np.diff(t[m])))
            return landed, False, float(np.count_nonzero(v[m] > THO_AIRBORNE) * dt), "tho"
    return landed, False, float(iv.t1 - iv.t0), "armed_time"


# ---------------------------------------------------------------- 因果標籤(直接用 engine.tag_time,不另抄規則)

def _ref(a: Anchors) -> tuple[float | None, str | None, str | None]:
    """engine.tag_time 用的參考時間:t_warn,沒有就 t_first_hit / t_detect。回傳 (時間, 錨點名, 中文名)。"""
    for key, zh in _REF_NAMES:
        v = getattr(a, key, None)
        if v is not None:
            return float(v), key, zh
    return None, None, None


def _tag(t: float, a: Anchors) -> str | None:
    """engine 會貼給這個時間的因果標籤(= engine.tag_time(t, a),同一個函式)。

    延遲 import:engine 只在執行時用 importlib 載入 det_*,這裡 import engine 不會循環。engine 載入失敗(只會發生在
    單獨使用本模組時)→ None = 不做因果排序,代表觸發改成「HIT 優先、最早」。'' = 正常結束的飛行(沒有先兆 / 後果可言)。
    """
    try:
        from .engine import tag_time
    except Exception:  # noqa: BLE001
        return None
    return tag_time(float(t), a)


def _phase(t: float, a: Anchors):
    """回傳 (phase, lead_s):phase = engine 的標籤(precursor / development / consequence / post_crash …;
    沒有因果標籤時 None);lead_s = 參考時間 − t(正 = 早於參考時間;沒有參考時間 → None)。"""
    tag = _tag(t, a)
    ref, _, _ = _ref(a)
    return (tag or None), (None if ref is None else ref - float(t))


def _is_precursor(t: float, a: Anchors) -> bool:
    """engine 會把這個時間標成 precursor 嗎(engine 不可用 = 沒有因果資訊,一律當 precursor 不降權)。"""
    tag = _tag(t, a)
    return tag is None or tag == TAG_PRECURSOR


def _public(tr: dict, a: Anchors, state: str | None = None, with_phase: bool = True) -> dict:
    """內部 trigger → values 裡的精簡 dict(tier 只在證據清單裡保留;地面 / 上鎖後事件不附因果階段)。"""
    d = {k: v for k, v in tr.items() if not k.startswith("_")}
    if not with_phase:
        d.pop("tier", None)
    d["t"] = _f(d["t"])
    d["t_end"] = _f(d.get("t_end"))
    if d.get("t_end") is not None:
        d["duration"] = _f(d["t_end"] - d["t"])
    if "t_mode" in d:
        d["t_mode"] = _f(d["t_mode"])
    if state is not None:
        d["state"] = state
    if with_phase:
        ph, lead = _phase(tr["t"], a)
        if ph is not None:
            d["phase"] = ph
            if lead is not None:
                d["lead_to_warn_s"] = _f(lead)
            if a.t_loss is not None:
                d["lead_to_loss_s"] = _f(a.t_loss - tr["t"])
    return _clean(d)


def _cap(lst: list, n: int = MAX_LISTED):
    return lst[:n]


def _fs_context(ctx: Ctx) -> tuple[dict, list[str]]:
    params = {k: float(ctx.param(k)) for k in FS_PARAMS if ctx.param(k) is not None}
    cav = []
    if params.get("FS_EKF_ACTION") == 0:
        cav.append("FS_EKF_ACTION=0:EKF 失效保護只回報不動作(仍會留下 ERR 16/17)")
    if params.get("BATT_FS_LOW_ACT") == 0 and params.get("BATT_FS_CRT_ACT") == 0:
        cav.append("電池失效保護動作皆為 0:低電壓不會觸發動作,也可能不留 ERR 6(電壓請看電源偵測結果)")
    if params.get("FS_THR_ENABLE") == 0:
        cav.append("FS_THR_ENABLE=0:遙控訊號失效保護未啟用")
    return params, cav


# ---------------------------------------------------------------- 偵測器

@guarded(DET_ID, DOMAIN)
def inflight_failsafe_or_mode_change(ctx: Ctx, a: Anchors) -> Finding:
    """解鎖飛行中的失效保護 / 被動切換模式(見模組說明)。"""
    ev = collect_events(ctx)
    iv = a.interval
    params, caveats = _fs_context(ctx)

    # ---- 未解鎖:地面事件只當資訊 -------------------------------------------------
    if iv is None:
        ground = [_public(tr, Anchors(), with_phase=False) for tr in ev["triggers"] if tr["tier"] == "hit"] + \
                 [_clean({"t": _f(g["t"]), "t_end": _f(g["t_end"]), "kind": g["kind"], "label": g["label"]})
                  for g in ev["glitches"]]
        ground.sort(key=lambda d: d["t"])
        n = len(ground)
        txt = "未解鎖(桌上 / 手持)的 log:不分析飛行中的失效保護。"
        if n:
            txt += f"log 內 {n} 筆失效保護 / GPS glitch 事件都發生在地面,只當資訊,不列入原因。"
        return Finding(DET_ID, DOMAIN, NA, "info", context="not_armed",
                       values={"reason": "not_armed", "n_ground_events": n, "ground_events": _cap(ground),
                               "fs_params": params}, text_zh=txt)

    t0, t1 = iv.t0, iv.t1
    mode_tb = ctx.table("MODE")
    if mode_tb is None and ctx.table("EV") is None and ctx.table("ERR") is None:
        return Finding(DET_ID, DOMAIN, NA, "info", context="no_data",
                       values={"reason": "no_event_tables"}, text_zh="log 沒有 MODE / EV / ERR 事件表,無法判斷飛行中有沒有失效保護。")
    mode_check = "no_mode_table" if mode_tb is None else ("ok" if "Rsn" in mode_tb else "no_rsn")
    if mode_check == "no_mode_table":
        caveats.append("MODE 表缺失:無法檢查飛行中的模式切換,只檢查了 ERR / EV / MSG")
    elif mode_check == "no_rsn":
        caveats.append("MODE 表沒有 Rsn 欄:模式切換的原因未記錄,飛行中的切換一律列為弱證據(SOFT)")

    landed, known, air_s, air_basis = flight_state(ctx, iv)
    never_flew = air_basis == "tho" and air_s < MIN_FLIGHT_S      # 沒有 EV28、油門輸出也證明沒離地
    later = [x.t0 for x in ctx.armed if x.t0 > t1]
    next_t0 = min(later) if later else float("inf")
    post_hi = min(t1 + POST_WINDOW_S, next_t0)

    def other_flight(t: float) -> bool:
        return any(o.t0 <= t <= o.t1 for o in ctx.armed if o is not iv and o.index != iv.index)

    counted: list[dict] = []          # 飛行中的證據(hit / soft 兩層)
    counted_state: list[str] = []
    landed_ev: list[dict] = []        # 解鎖但已落地(起飛前 / 降落後)
    post: list[dict] = []             # 上鎖後 60 s 內
    ground: list[dict] = []           # 未解鎖期間(含上鎖 60 s 以後)
    for tr in ev["triggers"]:
        t = tr["t"]
        if t0 <= t <= t1:
            if (known and landed(t)) or never_flew:
                landed_ev.append(_public(tr, a, "landed", with_phase=False))
            else:
                counted.append(dict(tr))
                counted_state.append("flying" if known else "unknown")
        elif t1 < t <= post_hi:
            d = _public(tr, a, with_phase=False)
            d["dt_after_disarm"] = _f(t - t1)
            post.append(d)
        elif other_flight(t):
            continue                                               # 另一次飛行的事件:不是這次分析的對象
        else:
            ground.append(_public(tr, a, with_phase=False))
    info = []
    for g in ev["glitches"]:
        tt = g["t"]
        d = _clean({"t": _f(tt), "t_end": _f(g["t_end"]), "kind": g["kind"], "label": g["label"]})
        if t0 <= tt <= t1:
            ph, lead = _phase(tt, a)
            if ph is not None:
                d["phase"] = ph
                if lead is not None:
                    d["lead_to_warn_s"] = _f(lead)
            info.append(d)
        elif t1 < tt <= post_hi:
            d["dt_after_disarm"] = _f(tt - t1)
            post.append(d)
        elif not any(o.t0 <= tt <= o.t1 for o in ctx.armed):
            ground.append(d)
    post.sort(key=lambda d: d["t"])
    ground.sort(key=lambda d: d["t"])

    # ---- 飛行中的模式(說明用):全程模式 + 所有切換 --------------------------------
    mode0 = ctx.mode_at(t0)
    mchg_in = [m for m in ev["mode_changes"] if t0 < m["t"] <= t1]
    modes = ([mode0] if mode0 else []) + [m["to"] for m in mchg_in]
    modes = list(dict.fromkeys(modes))
    mode_list = [_clean({"t": _f(m["t"]), "from": m["from"], "to": m["to"], "rsn": m["rsn"], "reason": m.get("reason"),
                         "cls": m["cls"]}) for m in mchg_in]

    # 解鎖區間內第一筆 ERR(任何子系統,含非失效保護):「某時間之前沒有任何 ERR」的數字依據(124 飛行 2 = 1258.94 GPS glitch)
    et, esub, ecode = ctx.err_rows()
    k_in = np.flatnonzero((et >= t0) & (et <= t1))
    first_err = None
    if k_in.size:
        k = int(k_in[0])
        first_err = {"t": _f(et[k]), "subsys": int(esub[k]), "ecode": int(ecode[k]),
                     "label": err_text(int(esub[k]), int(ecode[k]))}

    ref_t, ref_key, ref_zh = _ref(a)
    base_values = {
        "interval": [_f(t0), _f(t1)], "airborne_s": _f(air_s, 1), "airborne_basis": air_basis,
        "landed_state_known": bool(known),
        "modes": modes, "mode_changes": _cap(mode_list), "info_events": _cap(info),
        "landed_events": _cap(landed_ev), "post_disarm": _cap(post), "ground_events": _cap(ground),
        "n_ground_events": len(ground), "fs_params": params, "caveats": caveats, "first_err_in_interval": first_err,
        "err_subsys_checked": sorted(FAILSAFE_SUBSYS), "mode_reasons_failsafe": sorted(FAILSAFE_MODE_REASONS),
        "rsn_available": mode_check == "ok", "mode_check": mode_check, "lead_ref": ref_key,
    }

    # ---- 有證據 ----------------------------------------------------------------
    evidence: list[tuple[dict, str]] = []       # (trigger, tier)  unknown 狀態一律降為 soft
    for tr, st in zip(counted, counted_state):
        evidence.append((tr, "soft" if st == "unknown" else tr["tier"]))
    if evidence:
        n_ev = len(evidence)
        tags = [_tag(tr["t"], a) for tr, _ in evidence]
        pre = [tg is None or tg == TAG_PRECURSOR for tg in tags]      # None = engine 不可用:不做因果排序
        late = [tg in _LATE_TAGS for tg in tags]                       # '' = 正常結束的飛行:既不是先兆也不是後果
        # 代表觸發:precursor 優先 → HIT 層優先 → 最早(見模組說明)
        i_first = min(range(n_ev), key=lambda i: (0 if pre[i] else 1, 0 if evidence[i][1] == "hit" else 1,
                                                  evidence[i][0]["t"]))
        first, first_tier = evidence[i_first]
        status = HIT if first_tier == "hit" else SOFT
        # triggers 與 Finding 的因果標籤同一層:有 precursor 證據時,較晚的觸發移到 late_triggers(不得被假說計分)
        split = any(pre) and any(late)
        pub, late_pub = [], []
        for i, ((tr, tier), st) in enumerate(zip(evidence, counted_state)):
            d = _public(tr, a, st)
            d["tier"] = tier
            (late_pub if (split and late[i]) else pub).append(d)
        pub.sort(key=lambda d: d["t"])
        late_pub.sort(key=lambda d: d["t"])
        later_tr = sorted((evidence[i][0] for i in range(n_ev) if late[i] and i != i_first), key=lambda x: x["t"])
        first_pub = _public(first, a)
        ph, lead = _phase(first["t"], a)
        values = dict(base_values)
        values.update({
            "triggers": _cap(pub), "n_triggers": n_ev, "n_hit": sum(1 for _, tier in evidence if tier == "hit"),
            "n_precursor": sum(pre), "t_first_trigger": _f(min(tr["t"] for tr, _ in evidence)),
            "t_selected": _f(first["t"]), "first_kind": first["kind"], "first_label": first["label"],
            "phase": ph, "lead_to_warn_s": _f(lead) if ph is not None else None,
            "t_first_err": _f(min((x["t"] for x, _ in evidence if x["kind"] == "err" and x["tier"] == "hit"),
                                  default=None)),
            "confidence": "low",
        })
        if split:
            values["late_triggers"] = _cap(late_pub)
            values["n_late"] = len(late_pub)
        values = _clean(values)
        text = _evidence_text(first, first_pub, ph, lead, ref_zh, n_ev, status, known, later_tr, bool(pre[i_first]))
        return Finding(DET_ID, DOMAIN, status, "high" if status == HIT else "medium", t_onset=float(first["t"]),
                       t_alarm=float(first["t"]), context="air" if known else "landed_state_unknown",
                       values=values, text_zh=text)

    # ---- 沒有證據:CLEAR 或 NA ------------------------------------------------------
    if air_s < MIN_FLIGHT_S:
        v = dict(base_values)
        v.update({"reason": "no_flight", "min_flight_s": MIN_FLIGHT_S, "triggers": [], "n_triggers": 0})
        return Finding(DET_ID, DOMAIN, NA, "info", context="no_flight", values=v,
                       text_zh=f"這次解鎖離地飛行只有 {air_s:.1f} s(不足 {MIN_FLIGHT_S:.0f} s):不評估飛行中的失效保護。")
    values = dict(base_values)
    values.update({"triggers": [], "n_triggers": 0, "n_hit": 0, "min_flight_s": MIN_FLIGHT_S,
                   "confidence": "low" if mode_check == "no_mode_table" else "medium"})
    text = _clear_text(t0, t1, air_s, air_basis, modes, mchg_in, mode_check, info, landed_ev, post, a, params,
                       first_err)
    return Finding(DET_ID, DOMAIN, CLEAR, "info", context="air", values=values, text_zh=text)


# ---------------------------------------------------------------- 說明文字

def _phase_txt(ph, lead, ref_zh) -> str:
    """因果階段的說明(數字一律取絕對值,不會出現「-0.0 s」;與參考時間幾乎同時另寫)。"""
    if ph is None or lead is None or ref_zh is None:
        return "發生在上鎖之後,是後果而非原因" if ph == "post_crash" else ""
    d = abs(lead)
    same = d < 0.05
    if ph == "precursor":
        if same:
            return f"與{ref_zh}幾乎同時(相差 {d:.2f} s),依規則視為起點,可能是原因之一"
        return f"早於{ref_zh} {d:.1f} s,可能是原因之一"
    if ph == "development":
        # engine 對觸發型證據:development 一律改標 consequence(只有早於姿態異常才算原因)
        return f"晚於{ref_zh}開始 {d:.1f} s、尚未失控,屬反應 / 連帶現象,不算原因"
    if ph == "consequence":
        if same:
            return f"與{ref_zh}幾乎同時(相差 {d:.2f} s),依規則算在其後,是後果而非原因"
        return f"晚於{ref_zh} {d:.1f} s,是後果而非原因"
    if ph == "post_crash":
        return "發生在上鎖之後,是後果而非原因"
    return ""


def _evidence_text(first: dict, first_pub: dict, ph, lead, ref_zh, n: int, status: str, known: bool,
                   later: list[dict], first_is_pre: bool) -> str:
    dur = first_pub.get("duration")
    head = ("飛行中觸發失效保護 / 被動切換模式:" if status == HIT else "飛行中疑似失效保護 / 被動切換模式:") + first["label"]
    head += f"({_ts(first['t'])}" + (f",持續 {dur:.1f} s" if dur is not None else "") + ")"
    if first.get("mode_to") and first["kind"] == "err":
        head += f",隨後切換為 {first['mode_to']}"
    if n > 1:
        head += f",共 {n} 項"
    s = head
    p = _phase_txt(ph, lead, ref_zh)
    if p:
        s += f"({p})"
    s += "。"
    if later:
        lh = later[0]
        where = f"{ref_zh}之後" if ref_zh else "較晚"
        s += ((f"之後另有 {len(later)} 項發生在{where}" if first_is_pre else f"另有 {len(later)} 項也發生在{where}")
              + f"(最早 {_ts(lh['t'])} {lh['label']},後果,不算原因)。")
    if status == HIT:
        s += "失效保護通常只是中間動作,請往上查真正的觸發原因(GPS / EKF / 電池 / 遙控訊號)與當時的飛行模式。"
    else:
        s += "屬非典型失效保護訊號,證據偏弱,請對照模式與 STATUSTEXT 確認。"
    if not known:
        s += "(log 沒有 NOT_LANDED 事件,無法確認已離地,故降為 SOFT。)"
    return s


def _air_txt(air_s: float, basis: str) -> str:
    if basis == "armed_time":
        return f"解鎖約 {air_s:.0f} s,無離地紀錄"
    if basis == "tho":
        return f"依油門輸出估計離地約 {air_s:.0f} s"
    return f"離地約 {air_s:.0f} s"


def _clear_text(t0: float, t1: float, air_s: float, air_basis: str, modes: list[str], mchg_in: list[dict],
                mode_check: str, info: list[dict], landed_ev: list[dict], post: list[dict], a: Anchors, params: dict,
                first_err: dict | None = None) -> str:
    if mode_check == "no_mode_table":
        mode_part = "MODE 表缺失,無法檢查模式切換"
        what = "沒有失效保護 ERR 或馬達緊急停止"
    else:
        what = "沒有失效保護 ERR、失效保護原因的模式切換或馬達緊急停止"
        if not mchg_in:
            mode_part = f"模式全程 {modes[0]} 無切換" if modes else "飛行中沒有模式切換(解鎖時的模式未記錄)"
        else:
            n_cmd = sum(1 for m in mchg_in if m["cls"] in _CMD_CLASSES)
            rest = len(mchg_in) - n_cmd
            mode_part = (f"模式 {'、'.join(modes)}(共 {len(mchg_in)} 次切換,"
                         + ("皆為飛手指令 / 一般自動切換" if rest == 0
                            else f"{n_cmd} 次為飛手指令 / 一般自動切換,其餘 {rest} 次在落地期間") + ")")
    err_part = (f";解鎖後第一筆 ERR 在 {first_err['t']:.2f} s" if first_err is not None
                else ";整段解鎖期間沒有任何 ERR 列")
    s = f"解鎖飛行段 {t0:.1f}–{t1:.1f} s({_air_txt(air_s, air_basis)}){what},{mode_part}{err_part}。"
    extra = []
    _, _, ref_zh = _ref(a)
    if info:
        g = info[0]
        end = f"–{g['t_end']:.2f}" if g.get("t_end") is not None else ""
        late = ""
        if g.get("phase") == "consequence" and g.get("lead_to_warn_s") is not None and ref_zh:
            late = f",晚於{ref_zh} {abs(g['lead_to_warn_s']):.1f} s,是後果"
        extra.append(f"另有 GPS glitch {g['t']:.2f}{end} s(ERR 11,不屬失效保護{late})")
    crash_like = a.end_type in _CRASH_END_TYPES
    if post:
        p = post[0]
        lab = p.get("label", "失效保護")
        dt = float(p["dt_after_disarm"])
        dt_txt = f"{dt:.1f} s" if dt >= 0.1 else f"{dt * 1000:.0f} ms"
        extra.append(f"上鎖後 {dt_txt}({p['t']:.3f} s)才出現 {lab}"
                     + ("(墜機後的連鎖反應,是後果" if crash_like else "(地面事件")
                     + (",FS_EKF_ACTION=0 只回報不動作)" if params.get("FS_EKF_ACTION") == 0 and "EKF" in lab else ")"))
    if landed_ev:
        extra.append(f"落地期間有 {len(landed_ev)} 筆事件(解鎖未離地 / 降落後,不算飛行中)")
    if extra:
        s += ";".join(extra) + "。"
    return s


# ---------------------------------------------------------------- 模組入口

def run(ctx: Ctx, a: Anchors) -> list[Finding]:
    """所有偵測器依序執行並串接結果(每個偵測器至少回一筆 Finding,例外已被 @guarded 轉成 NA)。"""
    out: list[Finding] = []
    out += inflight_failsafe_or_mode_change(ctx, a)
    return out
