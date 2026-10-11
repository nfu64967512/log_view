"""導航偵測模組測試(core/crash/det_nav.py):python tests/test_crash_nav.py 或 pytest tests/test_crash_nav.py

七個偵測器:source_switch_audit / vio_dropout / ekf_aiding_timeout / ekf_testratio_gate /
ekf_vio_disagreement / nav_velocity_gps_fallback_risk / source_switch_position_jump。

涵蓋:
  1. 00000124 飛行 2(墜機)+ 真相錨點 / 只有 basic_anchors(局部姿態備援):驗收 A5 / A6 / A7 的導航部分 ——
     觸地 VIO 中斷 1255.13–1256.29(precursor);RC9 → 切到沒有定位的 GPS 組 1258.86 + GPS glitch 1258.94、
     切回 VIO 組 1259.66(VISP 已停 1.52 s)、最後一筆 VISP 1258.146、VIO 'stale' 1259.06 都是後果;
     EKF 輔助逾時 1265.26 / EKFCHECK 1266.24 在上鎖後(post_crash);SV/SP/SH ≤0.21(EKF 健康);
     VIO 組 VELXY=0 + GPS 934.5 起無定位、EKF–VIO 位置差 1.57 m / PErr 8.591 m(常態風險,standing)。
  2. 00000124 飛行 1(正常降落):沒有任何 HIT、沒有失事字眼;切換跳變 1.37 / 5.14 m 只是 SOFT(度量)。
  3. 00000124 飛行 2 截到 1255.0 s:事件型偵測器在 1255 s 之前沒有 HIT(常態型設定風險照實 HIT、標成 standing)。
  4. 6 個 baseline(從未解鎖):全部 NA 'not_armed'。另把整段 log「強制當成解鎖」驗證真實地面事件的分類
     (117/118 切到沒有 VISP 的 VIO 組 → 只能 SOFT;118 / 123 約 7 s 逾時機制;122 / 123 glitch 屬 GPS 品質;
     119 / 120 的 VIO 切換資料年齡 ≤0.04 s)—— 這些是 124 以外唯一的真實資料驗證。
  5. 合成資料(數字刻意與 124 不同):每個分支、門檻兩側、資料缺漏 → NA / SOFT、lane 中位數 / 最大值、
     來源組對應不寫死(第 2 組 = VIO)、uint8 / uint16 / int8 欄位、時間平移不變。
  5b. 審查修正的回歸測試(test_fix*):地面 gate fail 不蓋掉空中段、切到還沒有 VISP 的 VIO 組(不再 TypeError / −inf)、
     解鎖前逾時的重疊 <2 s 不算、VIO 組 + 解鎖前就沒 VISP、跳變窗夾在上鎖前、發散期間的文字、GPS 無定位時不用舊 SAcc、
     來源組未知不下肯定結論、NaN 欄位不發警告、斷句、解鎖 <5 s 不給 CLEAR。
  6. 契約:每個 id 每種情境至少一筆、values 可 JSON 序列化、例外 → NA、決定性、每個偵測器 < 1.5 s。

已知與設計規格的出入(以實測為準):124 的 XKF4.TS 三條 lane 都在 1265.262 才變 51(規格寫 1265.0);
最後位置資料到逾時的延遲實測 7.04–7.12 s(118 / 123 / 124,規格寫 6.7–6.9 s)。
"""
from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import crashkit  # noqa: E402
from core.crash import det_nav as dn  # noqa: E402
from core.crash import engine  # noqa: E402
from core.crash.anchors import basic_anchors  # noqa: E402
from core.crash.ctx import Ctx  # noqa: E402
from core.crash.types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval  # noqa: E402

OWNED = dn.DETECTOR_IDS
EVENT_IDS = ("source_switch_audit", "vio_dropout", "ekf_aiding_timeout", "ekf_testratio_gate",
             "source_switch_position_jump")
STANDING_IDS = ("ekf_vio_disagreement", "nav_velocity_gps_fallback_risk")
T_BASE = 100.0                     # 合成 log 的 t_base(證明相對 / 絕對時間轉換有做)
INF = 1e9

# 124 的來源組設定(1、2 = GPS;3 = VIO 位置、沒有速度來源),RC9 = 來源組開關
SRC_124 = {"EK3_SRC1_POSXY": 3, "EK3_SRC1_VELXY": 3, "EK3_SRC1_POSZ": 1, "EK3_SRC1_YAW": 1,
           "EK3_SRC2_POSXY": 3, "EK3_SRC2_VELXY": 3, "EK3_SRC2_POSZ": 1, "EK3_SRC2_YAW": 1,
           "EK3_SRC3_POSXY": 6, "EK3_SRC3_VELXY": 0, "EK3_SRC3_POSZ": 1, "EK3_SRC3_YAW": 1,
           "EK3_SRC_OPTIONS": 1, "RC9_OPTION": 90, "FS_EKF_ACTION": 0}


# ======================================================================================
# 合成 log:假的 LogData / LogIndex,讓真正的 Ctx 做 dtype 轉型、相對時間、解鎖區間、落地階梯
# ======================================================================================

class _Ix:
    """tables:名稱 → [lane0 欄位 dict, lane1, ...](欄位時間為絕對秒);一條 lane 的表沒有 instance。"""

    def __init__(self, tables, params, msgs):
        self.tables, self.params, self.msgs = tables, params, msgs

    def param_values(self):
        return dict(self.params)

    def log_messages(self):
        return [SimpleNamespace(t=t, text=s) for t, s in self.msgs]

    def messages(self):
        return {n: SimpleNamespace(instances=list(range(len(l))) if len(l) > 1 else []) for n, l in self.tables.items()}

    def table(self, name, instance=None):
        lanes = self.tables.get(name)
        i = 0 if instance is None else int(instance)
        if not lanes or i >= len(lanes):
            return None
        return {k: np.asarray(v) for k, v in lanes[i].items()}

    def raw_table(self, name):
        return self.table(name)

    def mode_labels(self, raw):
        return ["LOITER" for _ in raw["ModeNum"]]

    def has(self, key):
        name, _, fld = key.partition(".")
        lanes = self.tables.get(name)
        return bool(lanes) and fld in lanes[0]

    def series(self, key):
        name, _, fld = key.partition(".")
        lanes = self.tables.get(name)
        if not lanes or fld not in lanes[0]:
            return None
        return np.asarray(lanes[0]["t"], dtype=float), np.asarray(lanes[0][fld], dtype=float)


class _LD:
    def __init__(self, ix, t_base, t_last):
        self.index, self.t_base, self.t_last, self.events = ix, t_base, t_last, []


def _steps(t, pts, default):
    out = np.full(t.size, float(default))
    for ti, v in sorted(pts):
        out[t >= ti] = v
    return out


def _in(t, spans):
    m = np.zeros(t.size, dtype=bool)
    for sp in spans:
        m |= (t >= sp[0]) & (t < sp[1])
    return m


def make_log(*, arm=((10.0, 210.0, 0),), t_end=None, shift=0.0, t_base=T_BASE, takeoff=5.0, params=None,
             sets=((0.0, 1),), xkfs=True, xkf4=True, xkf3=True, ev=(), hposabs_t=5.0,
             gps=True, gps_bad=(), gps_lowsats=(), sacc=(),
             visp=True, visp_start=0.0, visp_stop=None, visp_gaps=(), perr=(), rst=(),
             gap=None, ivn_frozen=(), vel=(0.0, 0.0), jumps=(), lane_jumps=None,
             ts_runs=(), ratio=(), fs=(), glitch=(), err=(), msgs=(), rc9=None, att=None):
    """合成一次(或多次)解鎖的導航資料,時間用「情境秒」,最後整體加 shift + t_base 變成絕對秒。

    sets:[(t, 組別 1..3)] → XKFS.SS(0-based,三條 lane 時間錯開 13 ms);ev:[(t, EV id)];
    gps_bad / gps_lowsats:[(ta, tb)] → Status=1,NSats=0 / Status=3,NSats=4;sacc:[(ta, tb, m/s)];
    visp_gaps:[(ta, tb)] 這段沒有 VISP;perr:[(ta, tb, m)];rst:[t] VISP.Rst 遞增;
    gap(t) → (IPN, IPE);ivn_frozen:[(ta, tb)] IVN/IVE 凍結 0.02/0.01;vel:真實水平速度 (N, E);
    jumps:[(t, dN, dE)] EKF 位置跳變(三條 lane);lane_jumps:{lane: (t, dN, dE)} 只有該 lane;
    ts_runs:[(ta, tb, bits[, lanes])] XKF4.TS |= bits;ratio:[(ta, tb, 'SV'|'SP'|'SH', 值[, lanes])];
    fs:[(ta, tb, 值)];glitch:[(ta, tb)] XKF4.SS bit14;err:[(t, subsys, ecode)];msgs:[(t, 文字)];
    rc9:[(t, PWM)];att:[(ta, tb, 滾轉°)](目標 0°)。"""
    t_end = t_end if t_end is not None else max(b for _, b, _ in arm) + 40.0
    P = dict(SRC_124 if params is None else params)
    T: dict[str, list[dict]] = {}

    def set_at(t):
        return _steps(t, sets, sets[0][1]).astype(int)

    lanes_t = [np.arange(0.0, t_end, 0.04) + 0.013 * k for k in range(3)]
    if xkfs:
        T["XKFS"] = [{"t": tl, "SS": (set_at(tl) - 1).astype(np.uint8)} for tl in lanes_t]
    if xkf4:
        L = []
        for k, tl in enumerate(lanes_t):
            ss = np.zeros(tl.size, dtype=np.int64)
            tsb = np.full(tl.size, 50, dtype=np.int64)          # 絕對輔助建立前 TS=50(正常)
            if hposabs_t is not None:
                ss[tl >= hposabs_t] |= 0x10
                tsb[tl >= hposabs_t] = 48                       # bit4/5 恆為 1
            ss[_in(tl, glitch)] |= dn.GLITCH_BIT
            for r in ts_runs:
                if k in (r[3] if len(r) > 3 else (0, 1, 2)):
                    tsb[(tl >= r[0]) & (tl < r[1])] |= r[2]
            comp = {"SV": 0.05 + 0.02 * np.sin(tl / 3.0 + k), "SP": np.full(tl.size, 0.01),
                    "SH": 0.05 + 0.02 * np.cos(tl / 5.0 + k)}
            for r in ratio:
                if k in (r[4] if len(r) > 4 else (0, 1, 2)):
                    comp[r[2]][(tl >= r[0]) & (tl < r[1])] = r[3]
            fsv = np.where(tl < 1.0, 128, 0)
            for (ta, tb, val) in fs:
                fsv[(tl >= ta) & (tl < tb)] = val
            L.append({"t": tl, "SS": ss.astype(np.uint32), "TS": tsb.astype(np.uint8), "SV": comp["SV"],
                      "SP": comp["SP"], "SH": comp["SH"], "FS": fsv.astype(np.uint16),
                      "OFN": np.zeros(tl.size), "OFE": np.zeros(tl.size)})
        T["XKF4"] = L
    if xkf3:
        L = []
        for k, tl in enumerate(lanes_t):
            gn, ge = gap(tl) if gap is not None else (0.05 * np.sin(tl / 2.0), 0.05 * np.cos(tl / 2.0))
            fr = _in(tl, ivn_frozen)
            ivn = np.where(fr, 0.02, 0.05 * np.sin(1.7 * tl + k))
            ive = np.where(fr, 0.01, 0.05 * np.cos(1.3 * tl + k))
            L.append({"t": tl, "IPN": np.asarray(gn, float), "IPE": np.asarray(ge, float), "IVN": ivn, "IVE": ive})
        T["XKF3"] = L
    L = []
    for k, tl in enumerate(lanes_t):
        pn, pe = vel[0] * tl, vel[1] * tl
        js = list(jumps) + ([lane_jumps[k]] if lane_jumps and k in lane_jumps else [])
        for (tj, dnn, dee) in js:
            pn = pn + np.where(tl >= tj, dnn, 0.0)
            pe = pe + np.where(tl >= tj, dee, 0.0)
        L.append({"t": tl, "PN": pn, "PE": pe, "VN": np.full(tl.size, float(vel[0])), "VE": np.full(tl.size, float(vel[1]))})
    T["XKF1"] = L
    if gps:
        tg = np.arange(0.0, t_end, 0.2) + 0.01
        st, ns = np.full(tg.size, 3), np.full(tg.size, 12)
        ns[_in(tg, gps_lowsats)] = 4
        bad = _in(tg, gps_bad)
        st[bad], ns[bad] = 1, 0
        T["GPS"] = [{"t": tg, "Status": st.astype(np.uint8), "NSats": ns.astype(np.uint8)}]
        sa = np.full(tg.size, 0.3)
        for (ta, tb, v) in sacc:
            sa[(tg >= ta) & (tg < tb)] = v
        T["GPA"] = [{"t": tg, "SAcc": sa}]
    if visp:
        tv = np.arange(visp_start, t_end, 0.037)
        keep = ~_in(tv, visp_gaps)
        if visp_stop is not None:
            keep &= tv < visp_stop
        tv = tv[keep]
        pe = np.full(tv.size, 1.443)
        for (ta, tb, v) in perr:
            pe[(tv >= ta) & (tv < tb)] = v
        rs = np.zeros(tv.size)
        for tr in rst:
            rs[tv >= tr] += 1
        T["VISP"] = [{"t": tv, "PX": vel[0] * tv, "PY": vel[1] * tv, "Q": np.full(tv.size, 80, dtype=np.int8),
                      "PErr": pe, "Rst": rs.astype(np.uint8)}]
        T["VISV"] = [{"t": tv, "VX": np.full(tv.size, float(vel[0])), "VY": np.full(tv.size, float(vel[1]))}]
    if rc9 is not None:
        tr = np.arange(0.0, t_end, 0.1)
        T["RCIN"] = [{"t": tr, "C9": _steps(tr, rc9, rc9[0][1]).astype(np.uint16), "C1": np.full(tr.size, 1513, np.uint16)}]
    if att is not None:
        ta_ = np.arange(0.0, t_end, 0.1)
        z = np.zeros(ta_.size)
        roll = z.copy()
        for (a0, a1, r) in att:
            roll[(ta_ >= a0) & (ta_ < a1)] = r
        T["ATT"] = [{"t": ta_, "Roll": roll, "Pitch": z, "Yaw": z + 90.0, "DesRoll": z, "DesPitch": z, "DesYaw": z + 90.0}]
    arm_t, arm_s, arm_m = [], [], []
    for t0, t1, meth in arm:
        arm_t += [t0, t1]
        arm_s += [1, 0]
        arm_m += [0, meth]
    T["ARM"] = [{"t": np.array(arm_t, float), "ArmState": np.array(arm_s, np.uint8), "Method": np.array(arm_m, np.uint8),
                 "Forced": np.zeros(len(arm_t), np.uint8)}]
    evs = sorted(list(ev) + ([(t0 + takeoff, 28) for t0, _, _ in arm] if takeoff is not None else []))
    if evs:
        T["EV"] = [{"t": np.array([e[0] for e in evs], float), "Id": np.array([e[1] for e in evs], np.uint8)}]
    if err:
        e = sorted(err)
        T["ERR"] = [{"t": np.array([x[0] for x in e], float), "Subsys": np.array([x[1] for x in e], np.uint8),
                     "ECode": np.array([x[2] for x in e], np.uint8)}]
    T["MODE"] = [{"t": np.array([0.0]), "Mode": np.array([5], np.uint8), "ModeNum": np.array([5], np.uint8),
                  "Rsn": np.array([0], np.uint8)}]
    off = shift + t_base
    for lanes in T.values():
        for ln in lanes:
            ln["t"] = np.asarray(ln["t"], float) + off
    ix = _Ix(T, P, [(t + off, s) for t, s in msgs])
    return Ctx(_LD(ix, t_base, t_end + off))


def anc(ctx, idx=-1, shift=0.0, **kw):
    """合成 log 的錨點:interval = 第 idx 次解鎖;kw 的時間(t_warn / t_loss / t_touch / t_detect)以情境秒給,自動加 shift。"""
    a = Anchors(intervals=list(ctx.armed))
    a.interval = ctx.armed[idx] if ctx.armed else None
    a.end_type = kw.pop("end_type", "unknown")
    for k, v in kw.items():
        setattr(a, k, v + shift if isinstance(v, (int, float)) and k.startswith("t_") else v)
    return a


# ======================================================================================
# 共用檢查
# ======================================================================================

def check_contract(fs, label, a=None):
    ids = [f.id for f in fs]
    missing = set(OWNED) - set(ids)
    assert not missing, f"{label}:缺少 {missing}"
    for f in fs:
        assert f.domain == dn.DOMAIN == "nav" and f.id in OWNED, (label, f.id, f.domain)
        assert f.status in (HIT, SOFT, CLEAR, NA), (label, f.status)
        assert f.context != "error", (label, f.id, f.values)
        json.dumps(f.values)                                           # values 必須可 JSON 序列化
        assert f.text_zh, (label, f.id, "text_zh 空")
        if f.status in (HIT, SOFT):
            assert f.t_onset is not None and f.t_alarm is not None and f.t_onset <= f.t_alarm + 1e-9, \
                (label, f.id, f.t_onset, f.t_alarm)
            if a is not None and a.interval is not None and f.context != "post_disarm":
                assert f.t_onset <= a.interval.t1 + 1e-6, (label, f.id, f.t_onset, a.interval)
        else:
            assert f.t_onset is None, (label, f.id, f.status, f.t_onset)


def run(ctx, a, label):
    fs = dn.run(ctx, a)
    check_contract(fs, label, a)
    return fs


def pick(fs, det, context=None):
    c = [f for f in fs if f.id == det and (context is None or f.context == context)]
    assert c, f"{det}/{context} 沒有輸出:{statuses(fs)}"
    return c[0]


def every(fs, det):
    return [f for f in fs if f.id == det]


def statuses(fs):
    return [(f.id, f.status, f.context) for f in fs]


def near(x, ref, tol):
    return x is not None and abs(float(x) - ref) <= tol


def no_crash_words(fs, label):
    for f in fs:
        for w in ("失事", "墜機", "翻覆"):
            assert w not in f.text_zh, (label, f.id, w, f.text_zh)


# ======================================================================================
# 1. 00000124 飛行 2(墜機)
# ======================================================================================

def _check_124_crash(fs, label, local=False):
    """真相錨點與 basic_anchors(局部姿態備援)兩種都要成立的驗收項目。"""
    # ---- source_switch_audit:兩次「有問題的切換」都在姿態失控之後(後果) ----
    f = pick(fs, "source_switch_audit")
    assert f.status == HIT and f.context == "switch" and f.severity == "high", statuses(fs)
    assert near(f.t_onset, 1258.86, 0.05) and near(f.t_alarm, 1258.94, 0.05), (f.t_onset, f.t_alarm)
    v = f.values
    sw = v["switches"]
    assert v["n_switches"] == 3 and [round(r["t"], 1) for r in sw] == [919.1, 1258.9, 1259.7], sw
    assert sw[0]["to_set"] == 3 and sw[0]["flags"] == [] and sw[0]["vio_age_s"] <= 0.04, sw[0]   # 健康的 VIO 切換
    s1, s2 = sw[1], sw[2]
    assert (s1["from_set"], s1["to_set"]) == (3, 2) and s1["flags"] == ["gps_without_fix", "glitch_on_switch"], s1
    assert 320.0 < s1["gps_lost_s"] < 330.0 and near(s1["glitch_err_t"], 1258.94, 0.05), s1   # GPS 934.x 起無定位
    assert s1["pilot_rc_switch"]["channel"] == 9 and near(s1["pilot_rc_switch"]["t"], 1258.64, 0.06), s1   # RC9 1258.64
    assert 60.0 <= abs(s1["roll_deg"]) <= 90.0 and s1["phase"] == "consequence", s1                    # 切換時滾轉 65–87°
    assert (s2["from_set"], s2["to_set"]) == (2, 3) and s2["flags"] == ["vio_stale"] and near(s2["vio_age_s"], 1.52, 0.03), s2
    assert s2["pilot_rc_switch"]["channel"] == 9 and s2["phase"] == "consequence", s2
    # 後續:EKF 輔助逾時 1265.26 / EKFCHECK 1266.24(都在上鎖後)只歸給最後一次切換
    assert near(s2["followup_timeout_t"], 1265.26, 0.05) and near(s2["followup_ekfcheck_t"], 1266.24, 0.05), s2
    assert s1["followup_timeout_t"] is None, s1
    assert [round(x, 1) for x in v["flagged_t"]] == [1258.9, 1259.7] and v["effect_of_tipover"] is True, v["flagged_t"]
    assert v["phase"] == "consequence" and v["other_glitches"] == [] and v["anchors_local"] is local, v
    assert "後果" in f.text_zh and "1265.26" in f.text_zh
    # ---- vio_dropout:觸地中斷(precursor)+ 翻覆後串流中止(後果),飛行中其餘小中斷不觸發 ----
    vd = every(fs, "vio_dropout")
    assert sorted(f.context for f in vd) == ["in_vio_mode", "near_touchdown"] and all(f.status == HIT for f in vd), statuses(vd)
    td = pick(fs, "vio_dropout", "near_touchdown")
    e = td.values["event"]
    assert near(td.t_onset, 1255.13, 0.03) and near(e["resume"], 1256.29, 0.03) and near(e["dur"], 1.16, 0.03), e
    assert near(td.t_alarm, td.t_onset + dn.VIO_GAP_MIN_S, 1e-6)
    assert e["set"] == 3 and td.values["vio_set_active"] is True and td.values["velocity_source"] is False, td.values
    assert e["q_before"] == [70, 99, 72, 48, 85] and td.values["phase"] == "precursor", e
    # 審查 #11b:門檻 1.0 s 低於同一趟飛行中正常的 1.06 / 1.09 s 中斷 → 文字要講明鑑別依據是「觸地當下」、只用一次失事校準
    assert near(td.values["max_other_gap_s"], 1.09, 0.02) and "1.09" in td.text_zh and "信心中等" in td.text_zh, td.text_zh
    vm = pick(fs, "vio_dropout", "in_vio_mode")
    # VISP 1258.146 在第 3 組「使用中」停掉 → onset = 最後一筆(驗收 A6 的 last VISP 1258.146);0.71 s 後切到 GPS 組、
    # 1259.66 切回第 3 組時年齡 1.52 s ≥1.5 → alarm = 1259.66(真正「在 VIO 組且年齡 ≥1.5 s」的時間)
    ev = vm.values["event"]
    assert near(vm.t_onset, 1258.146, 0.01) and near(vm.t_alarm, 1259.66, 0.02), (vm.t_onset, vm.t_alarm)
    assert ev["entered_stale"] is False and ev["never_received"] is False and near(ev["t_last"], 1258.146, 0.01), ev
    assert near(ev["reentered_t"], 1259.66, 0.02) and near(ev["age_at_reentry_s"], 1.52, 0.03), ev
    assert "1258.15" in vm.text_zh and "切回 VIO 組" in vm.text_zh and "翻覆的結果" in vm.text_zh, vm.text_zh
    assert vm.values["permanent"] is True and vm.values["effect_of_tipover"] is True and vm.values["phase"] == "consequence"
    assert near(vm.values["stale_msg_t"], 1259.06, 0.05), vm.values["stale_msg_t"]
    gaps = td.values["dropouts_ge_1s"] + td.values["subthreshold_gaps"]
    assert {round(g["t_start"]) for g in gaps} >= {891, 956, 1049, 1085, 1141, 1255}, gaps    # 規格列的飛行中小中斷
    assert all(not g["near_touchdown"] for g in gaps if g["t_start"] < 1255.0)
    cm = td.values["companion_msgs"]                 # 伴隨電腦 VIO 訊息:'requalifying' 1256.05、'valid' 1257.06、'stale' 1259.06(重複)
    rq = [m for m in cm if "position_source_requalifying" in m["text"] and near(m["t"], 1256.05, 0.05)]
    sl = [m for m in cm if "stale" in m["text"] and near(m["t"], 1259.06, 0.05)]
    assert rq and rq[0]["phase"] == "development", cm
    assert sl and sl[0]["phase"] == "consequence" and sl[0]["repeat"] >= 5 and cm[-1] is sl[0], cm
    # ---- ekf_aiding_timeout:飛行中沒有逾時;上鎖後 3.1 s 起逾時 = 後果 ----
    at = every(fs, "ekf_aiding_timeout")
    assert sorted((f.status, f.context) for f in at) == [(CLEAR, "none"), (HIT, "post_disarm")], statuses(at)
    p = pick(fs, "ekf_aiding_timeout", "post_disarm")
    r0 = p.values["runs"][0]
    assert near(p.t_onset, 1265.26, 0.05) and near(p.values["after_disarm_s"], 3.14, 0.05), (p.t_onset, p.values["after_disarm_s"])
    assert near(r0["ekfcheck_t"], 1266.24, 0.05) and near(r0["ekfinav_t"], 1266.24, 0.05), r0
    assert r0["last_pos_src"] == "VISP" and near(r0["last_pos_t"], 1258.146, 0.01) and 6.5 <= r0["lag_s"] <= 7.5, r0
    assert p.values["phase"] == "post_crash" and p.severity == "info" and 38.0 <= p.values["t_abs_aiding"] <= 40.5
    assert p.values["fs_ekf_action"] == 0 and "後果" in p.text_zh
    # ---- ekf_testratio_gate:失事前 15 s SV/SP/SH ≤0.21、FS=0 → EKF 健康(驗收 A7) ----
    g = pick(fs, "ekf_testratio_gate")
    assert g.status == CLEAR and g.context == "healthy", statuses(fs)
    pm = g.values["pre_crash_max"]
    assert pm["SV"] <= 0.15 and pm["SP"] <= 0.05 and pm["SH"] <= 0.25 and g.values["fs_nonzero"] is False, pm
    assert near(g.values["pre_crash_window"][1], 1262.125, 0.01) and "失事前" in g.text_zh
    # ---- ekf_vio_disagreement(常態):1.57 m / p95 3.47 m、PErr 8.591 m 鎖 1048.4–1258.15 ----
    d = pick(fs, "ekf_vio_disagreement")
    dv = d.values
    assert d.status == HIT and d.context == "standing" and dv["standing"] is True, statuses(fs)
    assert 1.4 <= dv["median_gap_m"] <= 1.8 and 3.0 <= dv["p95_gap_m"] <= 4.0, dv
    assert 1250.0 <= dv["window"][1] <= 1255.9 + 0.01, dv["window"]                    # 窗終點 ≤ t_warn
    pl = dv["perr_latched"]
    assert len(pl) == 1 and near(pl[0][0], 1048.4, 0.3) and near(pl[0][1], 1258.15, 0.05) and pl[0][2] == 8.591, pl
    assert dv["perr_latched_in_window"] is True and "8.59" in d.text_zh
    # ---- nav_velocity_gps_fallback_risk(常態):VIO 組 VELXY=0、GPS 934.5 起無定位 ----
    n = pick(fs, "nav_velocity_gps_fallback_risk")
    nv = n.values
    assert n.status == HIT and n.context == "standing" and n.severity == "medium", statuses(fs)
    assert near(nv["gps_nofix_t"], 934.52, 0.05) and near(nv["gps_lost_t"], 934.16, 0.05), nv   # 934.16 起衛星 <6
    assert nv["risk_s"] >= 55.0 and nv["active_sets"] == [3] and nv["velxy_codes"]["3"] == 0, nv
    assert nv["latched_spans"] and nv["pure_vio_no_gps"] and nv["ekf_minus_vio_vel"]["rms"] is not None, nv
    assert near(nv["gps_lost_to_detect_s"], 1262.125 - 934.16, 0.1), nv["gps_lost_to_detect_s"]
    assert "934.5" in n.text_zh and "EK3_SRC3_VELXY" in n.text_zh
    # ---- source_switch_position_jump:度量(永遠不是 HIT):4.34 / 0.18 / 0.39 m ----
    j = pick(fs, "source_switch_position_jump")
    assert j.status == SOFT and j.severity == "medium" and j.values["max_jump_pair"] == "gps->vio", statuses(fs)
    jm = [r["jump_m"] for r in j.values["switches"]]
    assert near(jm[0], 4.34, 0.15) and near(jm[1], 0.18, 0.1) and near(jm[2], 0.39, 0.1), jm


def test_124_crash_flight_truth_anchors():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.truth_anchors_124(ctx)
    t0 = time.perf_counter()
    fs = run(ctx, a, "124 飛行2")
    dt = time.perf_counter() - t0
    _check_124_crash(fs, "124 飛行2")
    # engine 依 t_onset 貼的因果標籤:只有觸地 VIO 中斷是 precursor;切換 / 串流中止 = 後果;逾時 = 上鎖後
    tags = {(f.id, f.context): engine.tag_time(f.t_onset, a, standing=f.id in engine.STANDING_IDS or f.context == "standing")
            for f in fs}
    assert tags[("source_switch_audit", "switch")] == "consequence", tags
    assert tags[("vio_dropout", "near_touchdown")] == "precursor", tags
    assert tags[("vio_dropout", "in_vio_mode")] == "consequence", tags
    assert tags[("ekf_aiding_timeout", "post_disarm")] == "post_crash", tags
    assert all(tags[(d, "standing")] == "standing" for d in STANDING_IDS), tags
    assert tags[("source_switch_position_jump", "switch")] == "standing", tags
    pre = [f for f in fs if f.status == HIT and f.id in EVENT_IDS and f.t_onset < a.t_warn]
    assert [(f.id, f.context) for f in pre] == [("vio_dropout", "near_touchdown")], statuses(pre)
    assert dt < 1.5, dt
    # 審查 #10(已知限制,記錄在案):VIO 串流中止的 onset 1258.146 只比 t_loss 晚 0.046 s(engine 算出 1258.102),
    # 「後果」標籤靠這點間距;模組自己的 effect_of_tipover 在 t_loss 落在驗收容許範圍(1258.10 ±0.1)兩端都成立。
    vm = pick(fs, "vio_dropout", "in_vio_mode")
    assert 0.0 < vm.t_onset - a.t_loss < 0.1, vm.t_onset - a.t_loss
    for tl in (1258.00, 1258.20):
        a2 = crashkit.truth_anchors_124(ctx)
        a2.t_loss = tl
        vm2 = pick(run(ctx, a2, f"124 t_loss={tl}"), "vio_dropout", "in_vio_mode")
        assert vm2.values["effect_of_tipover"] is True and "翻覆的結果" in vm2.text_zh, (tl, vm2.values["effect_of_tipover"])
    print(f"  切換 HIT {fs[0].t_onset:.2f}(後果)、觸地 VIO 中斷 {pick(fs, 'vio_dropout', 'near_touchdown').t_onset:.2f}"
          f"(precursor)、逾時 {pick(fs, 'ekf_aiding_timeout', 'post_disarm').t_onset:.2f}(上鎖後);{dt * 1000:.0f} ms")


def test_124_crash_flight_bare_anchors_local_fallback():
    """engine 還沒填 t_warn / t_loss 時,模組自己用推力向量誤差估(10° / 30°),結論不變。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = basic_anchors(ctx)
    assert a.t_warn is None and a.t_loss is None
    fs = run(ctx, a, "124 飛行2 bare")
    _check_124_crash(fs, "124 飛行2 bare", local=True)
    v = pick(fs, "source_switch_audit").values
    assert near(v["t_warn"], 1255.9, 0.15) and near(v["t_loss"], 1258.10, 0.15), (v["t_warn"], v["t_loss"])
    print(f"  局部估算 t_warn {v['t_warn']:.2f} / t_loss {v['t_loss']:.2f},結論與真相錨點相同")


def test_124_engine_integration():
    """完整 engine:導航的後果 / 上鎖後證據不計分;H_NAV 是貢獻因素不是第一名(驗收 A3 / A6 / D4)。"""
    ld = crashkit.load_ctx(crashkit.CRASH_LOG).ld
    rep = engine.analyze(ld)
    nav = [f for f in rep.findings if f.domain == "nav"]
    check_contract(nav, "engine 124")
    tag = {(f.id, f.context): f.tag for f in nav}
    assert tag[("source_switch_audit", "switch")] == "consequence" and tag[("vio_dropout", "in_vio_mode")] == "consequence", tag
    assert tag[("vio_dropout", "near_touchdown")] == "precursor" and tag[("ekf_aiding_timeout", "post_disarm")] == "post_crash", tag
    hs = {h.id: h for h in rep.hypotheses}
    top = rep.top()
    assert top is not None and top.id != "H_NAV_FAILURE", [(h.id, h.score) for h in rep.hypotheses]
    hn = hs["H_NAV_FAILURE"]
    assert hn.status == "contributing" and hn.score > 0, (hn.status, hn.score)
    for e in hn.evidence:
        if e["det"] in OWNED and e["tag"] in ("consequence", "post_crash", "ground_info"):
            assert e["contrib"] == 0, e                                       # 後果不能計分
    contrib = {(e["det"], e["context"]) for e in hn.evidence if e["det"] in OWNED and e["contrib"] > 0}
    assert ("vio_dropout", "near_touchdown") in contrib and ("source_switch_audit", "switch") not in contrib, contrib
    print(f"  第一名 {top.id} {top.score:.1f};H_NAV {hn.score:.1f}(貢獻因素);導航計分項:{sorted(contrib)}")


# ======================================================================================
# 2. 00000124 飛行 1(正常降落)/ 3. 飛行 2 的 1255 s 之前
# ======================================================================================

def test_124_flight1_control_no_hit():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.flight1_anchors_124(ctx)
    fs = run(ctx, a, "124 飛行1")
    assert not [f for f in fs if f.status == HIT], statuses(fs)
    st = {f.id: (f.status, f.context) for f in fs}
    for d in ("source_switch_audit", "vio_dropout", "ekf_aiding_timeout", "ekf_testratio_gate", "ekf_vio_disagreement",
              "nav_velocity_gps_fallback_risk"):
        assert st[d][0] == CLEAR, (d, st[d])
    assert len(every(fs, "ekf_aiding_timeout")) == 1                       # 上鎖後 60 s 內沒有逾時(765.8 那段是地面)
    sw = pick(fs, "source_switch_audit").values["switches"]
    assert [round(r["t"], 1) for r in sw] == [391.9, 557.1] and sw[0]["vio_age_s"] <= 0.04, sw
    assert pick(fs, "vio_dropout").values["max_age_in_vio_mode_s"] < 0.5
    g = pick(fs, "ekf_testratio_gate").values
    assert max(g["pre_crash_max"].values()) <= 0.34 and max(g["precursor_max"].values()) <= 0.34, g
    assert pick(fs, "ekf_vio_disagreement").values["median_gap_m"] <= 0.2
    # 切換跳變 1.37 / 5.14 m:正常切換也會跳,只是 SOFT 度量
    j = pick(fs, "source_switch_position_jump")
    jm = [r["jump_m"] for r in j.values["switches"]]
    assert j.status == SOFT and near(jm[0], 1.37, 0.1) and near(jm[1], 5.14, 0.15), jm
    no_crash_words(fs, "124 飛行1")                                        # 正常降落:不得有失事字眼
    print(f"  全部 CLEAR(跳變 SOFT {jm[0]:.2f} / {jm[1]:.2f} m);沒有失事字眼")


def test_124_flight2_before_1255_no_event_hit():
    """驗收 MUST NOT:飛行 2 在 1255.0 s 之前不得有事件型 HIT。常態型(standing)的導航設定風險在這段已經存在,
    照實 HIT 並標成 standing(它們本來就不是鑑別器,只能當貢獻因素)。"""
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    iv = ArmedInterval(1, ctx.armed[1].t0, 1255.0, None)
    a = Anchors(interval=iv, intervals=list(ctx.armed), end_type="unknown")
    fs = run(ctx, a, "124 飛行2 <1255")
    bad = [f for f in fs if f.status == HIT and f.id in EVENT_IDS and f.t_onset < 1255.0]
    assert not bad, statuses(bad)
    st = {(f.id, f.context): f.status for f in fs}
    assert st[("source_switch_audit", "switches_ok")] == CLEAR and st[("vio_dropout", "none")] == CLEAR, st
    assert st[("ekf_aiding_timeout", "none")] == CLEAR and st[("ekf_testratio_gate", "healthy")] == CLEAR, st
    assert pick(fs, "vio_dropout").values["max_age_in_vio_mode_s"] < dn.VIO_MODE_AGE_S   # 1255 前最長 1.07 s
    for d in STANDING_IDS:
        f = pick(fs, d)
        assert f.status == HIT and f.context == "standing", (d, f.status, f.context)
    assert pick(fs, "source_switch_audit").values["t_warn"] is None                     # 1255 前沒有姿態異常
    print("  事件型偵測器 1255 s 前沒有 HIT;常態風險(VIO 無速度源 + GPS 無定位、EKF–VIO 差)照實標 standing")


# ======================================================================================
# 4. baseline(從未解鎖)+ 強制當成解鎖的真實地面事件
# ======================================================================================

def test_baselines_not_armed_all_na():
    for name in crashkit.BASELINES:
        ctx = crashkit.load_ctx(name)
        a = crashkit.not_armed_anchors(ctx)
        assert a.interval is None, name
        fs = run(ctx, a, name)
        assert len(fs) == len(OWNED) and all(f.status == NA and f.context == "not_armed" for f in fs), (name, statuses(fs))
        assert "未解鎖" in fs[0].text_zh
    print(f"  {len(crashkit.BASELINES)} 個 baseline × {len(OWNED)} 個 id 全部 NA 'not_armed'")


def _forced(name):
    ctx = crashkit.load_ctx(name)
    iv = ArmedInterval(0, 0.0, ctx.t_end, None)
    a = Anchors(interval=iv, intervals=[iv], end_type="unknown")
    return ctx, a, run(ctx, a, name + " 強制")


def test_baselines_forced_ground_events_real_data():
    """沒有閘門時(把整份 log 當成一次解鎖)用真實地面事件驗證分類邏輯 —— 124 以外唯一的真實資料:
    117/118 切到 VIO 組但 log 沒有 VISP → 只能 SOFT(data_missing);118 逾時 = 切走 GPS 後約 7 s;
    119/120 的 VIO 切換資料年齡 ≤0.04 s;122/123 的 glitch 都是 GPS 品質(SAcc ≥2);123 測試比失控歸因 GPS 品質。"""
    _, _, fs = _forced("00000117")
    f = pick(fs, "source_switch_audit")
    assert f.status == SOFT and f.values["data_missing"] is True and [round(x, 2) for x in f.values["flagged_t"]] == [437.36], f.values
    jm = [r["jump_m"] for r in pick(fs, "source_switch_position_jump").values["switches"]]
    assert max(jm) < 0.1, jm                                         # GPS 組之間 / 沒有 VIO 資料時跳變 ≤0.06 m
    _, _, fs = _forced("00000118")
    f = pick(fs, "source_switch_audit")
    rows = {round(r["t"], 2): r for r in f.values["switches"]}
    assert f.status == SOFT and rows[86.48]["flags"] == ["vio_data_missing"] and near(rows[86.48]["followup_timeout_t"], 93.44, 0.05), rows
    assert rows[85.08]["followup_timeout_t"] is None                 # 逾時只歸給最後一次切換
    t = pick(fs, "ekf_aiding_timeout")
    r0 = t.values["runs"][0]
    # bench log 從未離地(沒有 EV28)→ 解鎖後的地面逾時 = SOFT armed_ground,不是「飛行中」HIT
    assert (t.status, t.context) == (SOFT, "armed_ground") and r0["airborne"] is False, (t.status, t.context)
    assert near(t.t_onset, 93.44, 0.05) and near(r0["ekfcheck_t"], 94.36, 0.05) and "飛行中" not in t.text_zh, r0
    # 使用中的第 3 組不用 GPS:最後可用位置 = 86.4 切走 GPS 組之前(不是 GPS 本身停掉)→ 延遲約 7 s
    assert r0["last_pos_src"] == "GPS" and near(r0["last_pos_t"], 86.4, 0.1) and 6.5 <= r0["lag_s"] <= 7.5, r0
    for name, want in (("00000119", [182.96, 190.04, 197.56]), ("00000120", [42.97, 50.69])):
        _, _, fs = _forced(name)
        f = pick(fs, "source_switch_audit")
        ages = [(round(r["t"], 2), r["vio_age_s"]) for r in f.values["switches"] if r["uses_vio"]]
        assert f.status == CLEAR and [x[0] for x in ages] == want and all(x[1] <= 0.04 for x in ages), (name, ages)
    for name, edges in (("00000122", [398.56, 519.76]), ("00000123", [155.0, 346.4])):
        _, _, fs = _forced(name)
        og = pick(fs, "source_switch_audit").values["other_glitches"]
        assert [round(g["t"], 2) for g in og] == edges and all(g["class"] == "gps_quality" and g["sacc"] >= 2.0 for g in og), (name, og)
    _, _, fs = _forced("00000123")
    g = pick(fs, "ekf_testratio_gate")
    assert g.status == SOFT and g.context == "armed_ground" and g.values["attribution"] == "gps_quality", g.values
    assert g.values["n_runs"] == 9 and near(g.t_onset, 153.8, 0.1), (g.values["n_runs"], g.t_onset)
    t = pick(fs, "ekf_aiding_timeout")
    r0 = t.values["runs"][0]
    assert (t.status, t.context) == (SOFT, "armed_ground") and near(r0["t_start"], 212.0, 0.05) and 6.5 <= r0["lag_s"] <= 7.5, r0
    print("  117/118 只有 SOFT(沒有 VISP);118 / 123 逾時延遲 ~7 s(從未離地 → SOFT 地面);119/120 VIO 切換年齡 ≤0.04 s;"
          "122/123 glitch = GPS 品質;123 測試比 9 段歸因 GPS 品質")


# ======================================================================================
# 5. 合成資料
# ======================================================================================

# ---------------- source_switch_audit ----------------

def _gps_switch_log(**kw):
    """VIO 組(3)飛行,150.3 s 飛手把開關撥到 GPS 組(2),但 GPS 60 s 起就沒有定位;151.1 s 撥回 VIO 組。"""
    return make_log(sets=((0.0, 3), (150.3, 2), (151.1, 3)), gps_bad=((60.0, INF),), glitch=((150.3, 151.1),),
                    err=((150.42, 11, 2), (151.2, 11, 0)), rc9=((0.0, 1935), (150.05, 1515), (150.9, 1935)), **kw)


def test_syn_switch_into_gps_without_fix():
    ctx = _gps_switch_log()
    fs = run(ctx, anc(ctx), "切到無定位 GPS 組")
    f = pick(fs, "source_switch_audit")
    v = f.values
    r = v["switches"][0]
    assert f.status == HIT and near(f.t_onset, 150.3, 0.05) and near(f.t_alarm, 150.42, 0.01), (f.status, f.t_onset, f.t_alarm)
    assert r["flags"] == ["gps_without_fix", "glitch_on_switch"] and near(r["gps_lost_s"], 90.5, 0.1), r
    assert r["pilot_rc_switch"]["channel"] == 9 and v["switches"][1]["flags"] == [] and v["n_switches"] == 2, v["switches"]
    assert v["phase"] == "no_attitude_event" and "沒有姿態異常" in f.text_zh
    # 同一筆資料,姿態異常在切換之後 → 肇因候選;在切換之前 → 後果
    f = pick(run(ctx, anc(ctx, t_warn=160.0, t_loss=165.0), "precursor"), "source_switch_audit")
    assert f.values["phase"] == "precursor" and f.values["effect_of_tipover"] is False and "肇因候選" in f.text_zh
    f = pick(run(ctx, anc(ctx, t_warn=140.0, t_loss=145.0), "consequence"), "source_switch_audit")
    assert f.values["phase"] == "consequence" and f.values["effect_of_tipover"] is True and "後果" in f.text_zh
    print(f"  {r['t']:.2f}s 切到 GPS 組:GPS 已 {r['gps_lost_s']:.1f}s 無定位 + glitch → HIT;前/後因果標籤正確")


def test_syn_switch_into_gps_with_fix_is_clear():
    ctx = make_log(sets=((0.0, 3), (150.3, 2), (151.1, 3)))
    f = pick(run(ctx, anc(ctx), "GPS 正常"), "source_switch_audit")
    assert f.status == CLEAR and f.context == "switches_ok" and f.values["switches"][0]["gps_ok_1s"] is True, f.values
    f = pick(run(make_log(), anc(make_log()), "沒切換"), "source_switch_audit")
    assert f.status == CLEAR and f.context == "no_switch"
    print("  GPS 有定位時的切換 / 沒有切換 → CLEAR")


def test_syn_switch_into_stale_vio():
    ctx = make_log(sets=((0.0, 1), (120.0, 3)), visp_gaps=((119.2, 121.0),))
    fs = run(ctx, anc(ctx), "切到停掉的 VIO")
    f = pick(fs, "source_switch_audit")
    r = f.values["switches"][0]
    assert f.status == HIT and r["flags"] == ["vio_stale"] and 0.5 < r["vio_age_s"] < 1.0, r
    vm = pick(fs, "vio_dropout", "in_vio_mode")           # 切過去之後資料年齡 ≥1.5 s
    ev = vm.values["event"]
    # onset = 切到 VIO 組的時間(之前用的是 GPS 組,VIO 停掉還不是問題);alarm = 最後一筆 + 1.5 s
    assert vm.status == HIT and vm.values["permanent"] is False and near(vm.t_onset, 120.0, 0.05), ev
    assert ev["entered_stale"] is True and near(vm.t_alarm, ev["t_last"] + 1.5, 1e-3) and ev["t_last"] < 119.3, ev
    # 對照:VISP 正常時年齡 ≤0.04 s
    ctx = make_log(sets=((0.0, 1), (120.0, 3)))
    f = pick(run(ctx, anc(ctx), "VIO 正常"), "source_switch_audit")
    assert f.status == CLEAR and f.values["switches"][0]["vio_age_s"] <= 0.04, f.values
    print(f"  切到 VIO 組時 VISP 已 {r['vio_age_s']:.2f}s 沒資料 → HIT;正常時 ≤0.04 s → CLEAR")


def test_syn_switch_data_missing_is_soft_not_clear():
    ctx = make_log(sets=((0.0, 1), (120.0, 3)), visp=False)
    fs = run(ctx, anc(ctx), "沒有 VISP")
    f = pick(fs, "source_switch_audit")
    assert f.status == SOFT and f.values["data_missing"] is True and "VISP" in f.text_zh, (f.status, f.values)
    assert pick(fs, "vio_dropout").status == NA and pick(fs, "ekf_vio_disagreement").status == NA
    ctx = make_log(sets=((0.0, 3), (120.0, 1)), gps=False)
    f = pick(run(ctx, anc(ctx), "沒有 GPS"), "source_switch_audit")
    assert f.status == SOFT and f.values["switches"][0]["flags"] == ["gps_data_missing"] and "GPS" in f.text_zh, f.values
    print("  切到 VIO 組但沒有 VISP / 切到 GPS 組但沒有 GPS → SOFT(無法確認),不是 CLEAR 也不是 HIT")


def test_syn_glitch_classification_not_switch_induced():
    ctx = make_log(glitch=((100.0, 103.0), (180.0, 181.0)), ratio=((97.5, 100.0, "SV", 0.6),), sacc=((99.0, 101.0, 3.0),))
    f = pick(run(ctx, anc(ctx), "glitch"), "source_switch_audit")
    og = f.values["other_glitches"]
    assert f.status == CLEAR and [round(g["t"]) for g in og] == [100, 180], og
    assert og[0]["class"] == "gps_quality" and og[0]["sv_prev3s"] >= 0.6 and og[0]["sacc"] == 3.0, og[0]
    assert og[1]["class"] == "gps_reacquired", og[1]
    print("  非切換造成的 glitch:SV / SAcc 高 → GPS 品質;其餘 → 重新取得 GPS(只分類、不觸發)")


def test_syn_set_mapping_not_hardcoded():
    """不能假設第 3 組 = VIO:這裡第 2 組是 VIO、第 3 組是 GPS。"""
    P = dict(SRC_124, EK3_SRC2_POSXY=6, EK3_SRC2_VELXY=0, EK3_SRC3_POSXY=3, EK3_SRC3_VELXY=3)
    ctx = make_log(params=P, sets=((0.0, 1), (100.0, 2), (150.0, 3)), gps_bad=((60.0, INF),))
    fs = run(ctx, anc(ctx), "第 2 組 = VIO")
    f = pick(fs, "source_switch_audit")
    rows = f.values["switches"]
    assert rows[0]["uses_vio"] and not rows[0]["uses_gps"] and rows[0]["flags"] == [], rows[0]
    assert rows[1]["uses_gps"] and rows[1]["flags"] == ["gps_without_fix"] and near(f.t_onset, 150.0, 0.05), rows[1]
    j = pick(fs, "source_switch_position_jump")
    assert [r["pair"] for r in j.values["switches"]] == ["gps->vio", "vio->gps"], j.values["switches"]
    print("  來源組內容一律讀 EK3_SRCn_*:第 2 組 VIO 切換正常、第 3 組 GPS 無定位 → HIT")


def test_syn_set_steps_from_ev_and_missing():
    ctx = make_log(xkfs=False, ev=((150.0, 86),), gps_bad=((60.0, INF),))
    f = pick(run(ctx, anc(ctx), "EV 86"), "source_switch_audit")
    assert f.status == HIT and f.values["set_source"] == "ev" and near(f.t_onset, 150.0, 1e-6), (f.status, f.values["set_source"])
    ctx = make_log(xkfs=False)
    fs = run(ctx, anc(ctx), "沒有 XKFS / EV")
    assert pick(fs, "source_switch_audit").status == NA and pick(fs, "source_switch_audit").context == "no_xkfs"
    assert pick(fs, "source_switch_position_jump").status == NA
    print("  沒有 XKFS 改用 EV 85/86/87;都沒有 → NA")


# ---------------- vio_dropout ----------------

def test_syn_vio_dropout_near_touchdown():
    ctx = make_log(sets=((0.0, 3),), visp_gaps=((180.0, 181.25),))
    fs = run(ctx, anc(ctx, t_touch=180.1), "觸地中斷 VIO 組")
    f = pick(fs, "vio_dropout", "near_touchdown")
    e = f.values["event"]
    assert f.status == HIT and e["t_start"] < 180.0 and 1.2 < e["dur"] < 1.4 and f.values["velocity_source"] is False, e
    assert near(f.t_alarm, f.t_onset + 1.0, 1e-6) and len(every(fs, "vio_dropout")) == 1, statuses(fs)   # 1.3 s < 1.5 → 不是 in_vio_mode
    ctx = make_log(sets=((0.0, 1),), visp_gaps=((180.0, 181.25),))
    f = pick(run(ctx, anc(ctx, t_touch=180.1), "觸地中斷 GPS 組"), "vio_dropout")
    assert f.status == SOFT and f.context == "near_touchdown" and f.values["vio_set_active"] is False, (f.status, f.context)
    ctx = make_log(sets=((0.0, 3),), visp_gaps=((180.0, 181.25),))
    for label, a in (("沒有觸地", anc(ctx)), ("觸地在 0.6 s 後", anc(ctx, t_touch=180.6))):
        f = pick(run(ctx, a, label), "vio_dropout")
        assert f.status == CLEAR and len(f.values["dropouts_ge_1s"]) == 1, (label, f.status, f.context)
    # 撞擊群 a.impacts 的 ground_contact 也算觸地;hard_impact 不算
    a = anc(ctx, impacts=[{"t_start": 180.05, "t_peak": 180.06, "cls": "ground_contact"}])
    assert pick(run(ctx, a, "impacts"), "vio_dropout").context == "near_touchdown"
    a = anc(ctx, impacts=[{"t_start": 180.05, "t_peak": 180.06, "cls": "hard_impact"}])
    assert pick(run(ctx, a, "hard"), "vio_dropout").status == CLEAR
    print(f"  觸地 ±0.3 s 內中斷 {e['dur']:.2f}s:VIO 組 HIT / GPS 組 SOFT;不在觸地 → CLEAR")


def test_syn_vio_dropout_in_vio_mode_and_stream_end():
    ctx = make_log(sets=((0.0, 3),), visp_gaps=((150.0, 152.0),))
    f = pick(run(ctx, anc(ctx), "VIO 組中斷 2 s"), "vio_dropout")
    e = f.values["event"]
    assert f.status == HIT and f.context == "in_vio_mode" and e["permanent"] is False and 1.9 < e["dur"] < 2.1, e
    assert near(f.t_alarm, f.t_onset + dn.VIO_MODE_AGE_S, 1e-6)
    ctx = make_log(sets=((0.0, 3),), visp_stop=150.0)
    fs = run(ctx, anc(ctx), "VIO 組串流中止")
    vd = every(fs, "vio_dropout")
    assert [(f.status, f.context) for f in vd] == [(HIT, "in_vio_mode")] and vd[0].values["permanent"] is True, statuses(vd)
    ctx = make_log(sets=((0.0, 1),), visp_stop=150.0)
    f = pick(run(ctx, anc(ctx), "GPS 組串流中止"), "vio_dropout")
    assert f.status == SOFT and f.context == "stream_end" and f.values["vio_set_active_at_end"] is False, (f.status, f.context)
    assert near(f.t_alarm, f.t_onset + dn.VIO_TAIL_S, 1e-6) and "不是 VIO 組" in f.text_zh
    # 姿態失控前後 0.5 s 內停掉 = 翻覆的結果
    ctx = make_log(sets=((0.0, 3),), visp_stop=150.0)
    f = pick(run(ctx, anc(ctx, t_warn=148.0, t_loss=150.2), "翻覆後停"), "vio_dropout")
    assert f.values["effect_of_tipover"] is True and f.values["phase"] == "development" and "翻覆的結果" in f.text_zh
    print("  VIO 組中斷 ≥1.5 s → HIT(永久 / 恢復);GPS 組串流提前結束 → SOFT;失控前後停 → 標為結果")


def test_syn_vio_dropout_grid_fallback_without_xkf():
    """沒有 XKF4 / XKF3 時改用均勻時間格;不能用 VISP 自己的時間(年齡永遠 ≈0,會把 VIO 組中的串流中止誤標成無害)。"""
    ctx = make_log(sets=((0.0, 3),), visp_stop=150.0, xkf4=False, xkf3=False)
    f = pick(run(ctx, anc(ctx), "沒有 XKF"), "vio_dropout")
    assert f.status == HIT and f.context == "in_vio_mode" and f.values["permanent"] is True, (f.status, f.context)
    print("  沒有 XKF 時間格仍抓得到 VIO 組中的串流中止")


def test_syn_vio_dropout_small_gaps_clear_and_na():
    ctx = make_log(sets=((0.0, 3),), visp_gaps=((100.0, 100.6), (140.0, 140.7)),
                   msgs=((120.0, "SRC=1/197:VIO invalid: position_source_requalifying"),
                         (121.0, "SRC=1/197:VIO degraded: near-range height"),
                         (215.0, "SRC=1/197:VIO invalid: stale_last_vo_accepted_timestamp_ns")))
    f = pick(run(ctx, anc(ctx), "小中斷"), "vio_dropout")
    assert f.status == CLEAR and len(f.values["subthreshold_gaps"]) == 2 and f.values["max_age_in_vio_mode_s"] < 1.0, f.values
    cm = f.values["companion_msgs"]
    assert [round(m["t"]) for m in cm] == [120, 215] and cm[1]["phase"] == "post_disarm", cm   # near-range 是雜訊
    ctx = make_log(visp_stop=5.0)
    assert pick(run(ctx, anc(ctx), "VISP 只在解鎖前"), "vio_dropout").context == "no_visp_in_flight"
    ctx = make_log(visp=False)
    assert pick(run(ctx, anc(ctx), "沒有 VISP"), "vio_dropout").context == "no_visp"
    print("  0.6–0.7 s 小中斷 → CLEAR(列在 values);沒有 VISP → NA")


# ---------------- ekf_aiding_timeout ----------------

def test_syn_aiding_timeout_in_flight():
    ctx = make_log(gps_bad=((93.0, INF),), ts_runs=((100.0, 103.5, 3),), err=((101.0, 16, 2), (101.0, 17, 1)))
    fs = run(ctx, anc(ctx), "飛行中逾時")
    f = pick(fs, "ekf_aiding_timeout")
    r = f.values["runs"][0]
    assert f.status == HIT and f.context == "in_flight" and f.severity == "medium" and near(f.t_onset, 100.0, 0.05), statuses(fs)
    assert r["pos_timeout"] and r["vel_timeout"] and near(r["ekfcheck_t"], 101.0, 1e-6) and near(r["ekfinav_t"], 101.0, 1e-6), r
    assert r["last_pos_src"] == "GPS" and near(r["lag_s"], 7.2, 0.1) and near(f.t_alarm, f.t_onset + 2.0, 1e-6), r
    # 失控之後才逾時 → info / in_flight_after_loss
    f = pick(run(ctx, anc(ctx, t_warn=95.0, t_loss=98.0), "失控後"), "ekf_aiding_timeout")
    assert f.context == "in_flight_after_loss" and f.severity == "info" and "後果" in f.text_zh
    # 只有一條 lane 逾時也算(三條 lane 取最大)
    ctx = make_log(ts_runs=((100.0, 103.5, 1, (2,)),))
    assert pick(run(ctx, anc(ctx), "lane 2"), "ekf_aiding_timeout").status == HIT
    print(f"  {f.t_onset:.2f}s 起逾時,最後 GPS 位置 {r['last_pos_t']:.2f}s(延遲 {r['lag_s']:.1f}s)、EKFCHECK 配對")


def test_syn_aiding_timeout_thresholds_and_masks():
    ctx = make_log(ts_runs=((100.0, 101.5, 1),))
    assert pick(run(ctx, anc(ctx), "1.5 s"), "ekf_aiding_timeout").status == CLEAR            # < 2 s
    ctx = make_log(arm=((2.0, 150.0, 0),), hposabs_t=8.0)                                       # TS=50 在絕對輔助前是正常的
    f = pick(run(ctx, anc(ctx), "絕對輔助前"), "ekf_aiding_timeout")
    assert f.status == CLEAR and near(f.values["t_abs_aiding"], 8.0, 0.05), f.values
    ctx = make_log(ts_runs=((8.0, 14.0, 3),), takeoff=1.0)                                      # 解鎖前就開始、延續到起飛(11 s)後
    f = pick(run(ctx, anc(ctx), "解鎖前開始"), "ekf_aiding_timeout")
    assert f.status == HIT and f.context == "in_flight" and near(f.t_onset, 10.0, 0.05), (f.status, f.context, f.values)
    assert f.values["runs"][0]["started_before_arm"] is True and near(f.values["runs"][0]["run_start"], 8.0, 0.05)
    assert "就已逾時" in f.text_zh and "飛行中" in f.text_zh
    ctx = make_log(xkf4=False)
    assert pick(run(ctx, anc(ctx), "沒有 XKF4"), "ekf_aiding_timeout").context == "no_xkf4"
    ctx = make_log(hposabs_t=None)
    assert pick(run(ctx, anc(ctx), "沒有絕對輔助"), "ekf_aiding_timeout").context == "no_abs_aiding"
    print("  <2 s / 絕對輔助前 / bit4-5 → 不算;解鎖前就逾時 → 算;缺資料 → NA")


def test_syn_aiding_timeout_post_disarm():
    ctx = make_log(ts_runs=((215.0, 220.0, 3),), err=((216.0, 16, 2),))
    fs = run(ctx, anc(ctx, end_type="normal_landing"), "上鎖後(正常降落)")
    at = every(fs, "ekf_aiding_timeout")
    assert sorted((f.status, f.context) for f in at) == [(CLEAR, "none"), (HIT, "post_disarm")], statuses(at)
    p = pick(fs, "ekf_aiding_timeout", "post_disarm")
    assert near(p.values["after_disarm_s"], 5.0, 0.05) and p.values["phase"] == "post_disarm" and "失事" not in p.text_zh
    p = pick(run(ctx, anc(ctx, end_type="crash_check", t_detect=210.0), "上鎖後(失事)"), "ekf_aiding_timeout", "post_disarm")
    assert p.values["phase"] == "post_crash" and "失事後果" in p.text_zh
    # 下一次解鎖之後的逾時屬於下一次飛行,不是上一次的「上鎖後」
    ctx = make_log(arm=((10.0, 100.0, 0), (105.0, 200.0, 0)), ts_runs=((120.0, 125.0, 3),))
    f1 = every(run(ctx, anc(ctx, idx=0), "飛行1"), "ekf_aiding_timeout")
    assert [(f.status, f.context) for f in f1] == [(CLEAR, "none")], statuses(f1)
    assert pick(run(ctx, anc(ctx, idx=1), "飛行2"), "ekf_aiding_timeout").status == HIT
    print("  上鎖後 60 s 內的逾時另列 post_disarm(正常降落不寫失事);下次解鎖後的不算")


# ---------------- ekf_testratio_gate ----------------

def test_syn_testratio_gate():
    ctx = make_log()
    f = pick(run(ctx, anc(ctx, end_type="normal_landing"), "健康"), "ekf_testratio_gate")
    assert f.status == CLEAR and f.context == "healthy" and "上鎖前" in f.text_zh and "失事" not in f.text_zh, f.text_zh
    f = pick(run(ctx, anc(ctx, t_detect=200.0), "健康 + t_detect"), "ekf_testratio_gate")
    assert f.values["pre_crash_window"] == [185.0, 200.0] and "失事前" in f.text_zh
    # 三條 lane 都 ≥1 持續 2 s(空中)→ HIT;SAcc 正常 → 歸因 VIO / 其他;SAcc 3 m/s → GPS 品質
    ctx = make_log(ratio=((180.0, 182.0, "SV", 1.5),))
    f = pick(run(ctx, anc(ctx), "gate fail"), "ekf_testratio_gate")
    assert f.status == HIT and f.context == "air" and near(f.t_onset, 180.0, 0.05) and f.values["attribution"] == "vio_or_other", f.values
    assert near(f.values["peak_median"], 1.5, 1e-6) and f.values["n_runs"] == 1
    ctx = make_log(ratio=((180.0, 182.0, "SV", 1.5),), sacc=((179.0, 183.0, 3.0),))
    f = pick(run(ctx, anc(ctx), "gate fail GPS"), "ekf_testratio_gate")
    assert f.values["attribution"] == "gps_quality" and f.values["sacc_max"] == 3.0
    # 只有一條 lane 高 → 中位數沒過門檻,但峰值 ≥0.5 → SOFT near_gate(t_onset = 峰值時間)
    ctx = make_log(ratio=((180.0, 182.0, "SV", 1.5, (1,)),))
    f = pick(run(ctx, anc(ctx), "單一 lane"), "ekf_testratio_gate")
    assert f.status == SOFT and f.context == "near_gate" and near(f.t_onset, 180.0, 0.05), (f.status, f.context, f.t_onset)
    ctx = make_log(ratio=((180.0, 180.6, "SH", 1.4),))                                           # 0.6 s < 1.0 s
    assert pick(run(ctx, anc(ctx), "太短"), "ekf_testratio_gate").context == "near_gate"
    ctx = make_log(fs=((150.0, 150.5, 4),))                                                       # 非零 FS(開頭 128 不算)
    f = pick(run(ctx, anc(ctx), "FS"), "ekf_testratio_gate")
    assert f.status == SOFT and f.values["fs_nonzero"] is True and near(f.t_onset, 150.0, 0.05), f.values
    # 解鎖但還在地面(EV28 之前)→ SOFT armed_ground
    ctx = make_log(arm=((50.0, 250.0, 0),), takeoff=20.0, ratio=((55.0, 57.5, "SP", 2.0),))
    f = pick(run(ctx, anc(ctx), "地面"), "ekf_testratio_gate")
    assert f.status == SOFT and f.context == "armed_ground" and near(f.t_onset, 55.0, 0.05), (f.status, f.context)
    ctx = make_log(xkf4=False)
    assert pick(run(ctx, anc(ctx), "沒有 XKF4"), "ekf_testratio_gate").context == "no_xkf4"
    ctx = make_log(arm=((10.0, 30.0, 0),))
    assert pick(run(ctx, anc(ctx), "t<40"), "ekf_testratio_gate").context == "insufficient_data"
    print("  三 lane 中位數 ≥1 持續 1 s → HIT(空中)/ SOFT(地面);單一 lane / 太短 / FS → SOFT;健康 → CLEAR")


# ---------------- ekf_vio_disagreement ----------------

def _const_gap(val):
    return lambda t: (np.full(t.size, val) + 0.05 * np.sin(t), 0.05 * np.cos(t))


def test_syn_vio_disagreement():
    ctx = make_log(sets=((0.0, 3),))
    f = pick(run(ctx, anc(ctx), "一致"), "ekf_vio_disagreement")
    assert f.status == CLEAR and f.values["median_gap_m"] < 0.1 and near(f.values["window"][1], 205.0, 0.05), f.values
    ctx = make_log(sets=((0.0, 3),), gap=_const_gap(2.0), perr=((40.0, INF, 8.591),))
    f = pick(run(ctx, anc(ctx), "差 2 m"), "ekf_vio_disagreement")
    v = f.values
    assert f.status == HIT and f.context == "standing" and near(v["median_gap_m"], 2.0, 0.1), v
    assert v["perr_latched_in_window"] is True and near(v["perr_latched"][0][0], 40.0, 0.05) and "PErr" in f.text_zh, v
    # 中位數小但 10% 樣本 5 m → p95 >3 m → HIT
    spiky = (lambda t: (np.where(np.arange(t.size) % 10 == 0, 5.0, 0.5), np.zeros(t.size)))
    ctx = make_log(sets=((0.0, 3),), gap=spiky)
    f = pick(run(ctx, anc(ctx), "尖峰"), "ekf_vio_disagreement")
    assert f.status == HIT and f.values["median_gap_m"] < 1.0 and f.values["p95_gap_m"] > 3.0, f.values
    print("  |EKF−VIO| 中位數 / p95 門檻、PErr 鎖定都依實際數字判斷")


def test_syn_vio_disagreement_holds_and_window():
    big = (lambda lo, hi: (lambda t: (np.where((t >= lo) & (t < hi), 3.0, 0.1), np.zeros(t.size))))
    # 切換到 VIO 組後 30 s 內不算(EKF 位置重置)
    ctx = make_log(sets=((0.0, 1), (150.0, 3)), gap=big(150.0, 178.0))
    f = pick(run(ctx, anc(ctx), "切換後 30 s"), "ekf_vio_disagreement")
    assert f.status == CLEAR and f.values["window"][0] >= 180.0 - 0.05, f.values
    # VISP.Rst 改變之後 30 s 內不算
    ctx = make_log(sets=((0.0, 3),), rst=(170.0,), gap=big(170.0, 199.0))
    f = pick(run(ctx, anc(ctx), "Rst"), "ekf_vio_disagreement")
    assert f.status == CLEAR, f.values
    # 窗終點 = t_warn:只在 t_warn 之後才變大 → CLEAR;沒有姿態異常(窗到上鎖前 5 s)→ HIT
    ctx = make_log(sets=((0.0, 3),), gap=big(160.0, INF))
    f = pick(run(ctx, anc(ctx, t_warn=158.0, t_loss=165.0), "t_warn 前"), "ekf_vio_disagreement")
    assert f.status == CLEAR and f.values["window"][1] <= 158.0, f.values["window"]
    assert pick(run(ctx, anc(ctx), "沒有姿態異常"), "ekf_vio_disagreement").status == HIT
    ctx = make_log(sets=((0.0, 1),))
    assert pick(run(ctx, anc(ctx), "沒用 VIO"), "ekf_vio_disagreement").context == "vio_not_active"
    ctx = make_log(sets=((0.0, 3),), visp_start=198.0)
    assert pick(run(ctx, anc(ctx), "有效 7 s"), "ekf_vio_disagreement").context == "insufficient_data"
    print("  切換 / VISP 重置後 30 s 不算、窗終點在 t_warn;沒用 VIO 或有效樣本不足 → NA")


# ---------------- nav_velocity_gps_fallback_risk ----------------

def test_syn_nav_velocity_risk():
    ctx = make_log(sets=((0.0, 3),), gps_bad=((30.0, INF),))
    f = pick(run(ctx, anc(ctx), "VIO 無速度 + GPS 無定位"), "nav_velocity_gps_fallback_risk")
    v = f.values
    assert f.status == HIT and f.context == "standing" and near(f.t_onset, 30.0, 0.1) and v["risk_s"] >= 59.0, (f.status, f.t_onset, v)
    assert near(v["gps_lost_t"], 30.0, 0.05) and v["active_sets"] == [3] and "EK3_SRC3_VELXY" in f.text_zh, v
    # 晚切到 VIO 組:風險段從切換開始,但文字的 GPS 失去時間仍是 30 s(不是切換時間)
    ctx = make_log(sets=((0.0, 1), (100.0, 3)), gps_bad=((30.0, INF),))
    f = pick(run(ctx, anc(ctx), "晚切 VIO"), "nav_velocity_gps_fallback_risk")
    assert f.status == HIT and near(f.t_onset, 100.0, 0.05) and near(f.values["gps_lost_t"], 30.0, 0.05) and "30.0s" in f.text_zh, f.text_zh
    # 先衛星不足、再完全沒定位
    ctx = make_log(sets=((0.0, 3),), gps_lowsats=((30.0, 31.0),), gps_bad=((31.0, INF),))
    f = pick(run(ctx, anc(ctx), "衛星不足"), "nav_velocity_gps_fallback_risk")
    assert near(f.values["gps_lost_t"], 30.0, 0.05) and near(f.values["gps_nofix_t"], 31.0, 0.05) and "31.0s" in f.text_zh, f.values
    # GPS 組但 IVN/IVE 凍結(速度量測實際沒被融合)+ GPS 無定位 → 也是 HIT
    ctx = make_log(sets=((0.0, 1),), gps_bad=((30.0, INF),), ivn_frozen=((30.0, INF),))
    f = pick(run(ctx, anc(ctx), "IVN 凍結"), "nav_velocity_gps_fallback_risk")
    assert f.status == HIT and f.values["latched_spans"] and f.values["active_sets"] == [1], f.values
    print("  VELXY=0 或速度新息凍結 + GPS 無定位 ≥10 s → HIT;GPS 失去時間獨立計算")


def test_syn_nav_velocity_soft_clear_na():
    ctx = make_log(sets=((0.0, 3),))
    f = pick(run(ctx, anc(ctx), "GPS 可用"), "nav_velocity_gps_fallback_risk")
    assert f.status == SOFT and f.severity == "info" and "有備援" in f.text_zh
    P = dict(SRC_124, EK3_SRC3_VELXY=6)
    ctx = make_log(params=P, sets=((0.0, 3),), gps_bad=((30.0, INF),))
    f = pick(run(ctx, anc(ctx), "VELXY=6"), "nav_velocity_gps_fallback_risk")
    assert f.status == CLEAR and f.values["velxy_codes"]["3"] == 6, f.values
    ctx = make_log(sets=((0.0, 3),), gps=False)
    f = pick(run(ctx, anc(ctx), "沒有 GPS 表"), "nav_velocity_gps_fallback_risk")
    assert f.status == SOFT and f.values["data_missing"] == "GPS", (f.status, f.values.get("data_missing"))
    ctx = make_log(arm=((10.0, 25.0, 0),))
    assert pick(run(ctx, anc(ctx), "太短"), "nav_velocity_gps_fallback_risk").context == "insufficient_data"
    print("  GPS 可用 → SOFT;VIO 有速度源 → CLEAR;沒有 GPS 資料 → SOFT(無法確認);太短 → NA")


# ---------------- source_switch_position_jump ----------------

def test_syn_position_jump():
    base = dict(vel=(3.0, 0.0), sets=((0.0, 1), (150.0, 3)))
    for dj, want in ((2.5, (SOFT, "info")), (4.0, (SOFT, "medium")), (0.0, (CLEAR, "info"))):
        ctx = make_log(jumps=((150.0, dj, 0.0),), **base)
        f = pick(run(ctx, anc(ctx), f"跳 {dj}"), "source_switch_position_jump")
        assert (f.status, f.severity) == want and near(f.values["max_jump_m"], dj, 0.1), (dj, f.status, f.values["max_jump_m"])
        assert f.values["max_jump_pair"] == "gps->vio" and f.status != HIT
    # 單一 lane 異常 → 三條 lane 取中位數
    ctx = make_log(lane_jumps={2: (150.0, 10.0, 0.0)}, **base)
    f = pick(run(ctx, anc(ctx), "單 lane"), "source_switch_position_jump")
    assert f.status == CLEAR and f.values["switches"][0]["n_lanes"] == 3 and f.values["max_jump_m"] < 0.1, f.values
    # GPS 組之間不該跳 → 註記
    ctx = make_log(vel=(3.0, 0.0), sets=((0.0, 1), (150.0, 2)), jumps=((150.0, 1.5, 0.0),))
    f = pick(run(ctx, anc(ctx), "GPS→GPS"), "source_switch_position_jump")
    assert f.status == SOFT and f.values["max_jump_pair"] == "gps->gps" and "同為 GPS" in f.text_zh
    # 停留太短:改用「切換後到下一次切換前」
    ctx = make_log(vel=(3.0, 0.0), sets=((0.0, 1), (150.0, 3), (150.2, 1)), jumps=((150.0, 2.5, 0.0), (150.2, -2.5, 0.0)))
    f = pick(run(ctx, anc(ctx), "停留短"), "source_switch_position_jump")
    r0 = f.values["switches"][0]
    assert r0["short_dwell"] is True and near(r0["jump_m"], 2.5, 0.2), r0
    for label, ctx in (("沒切換", make_log()), ("解鎖前切換", make_log(sets=((0.0, 1), (5.0, 3))))):
        f = pick(run(ctx, anc(ctx), label), "source_switch_position_jump")
        assert f.status == NA and f.context == "no_switch", (label, f.status, f.context)
    print("  跳變 = 扣掉速度積分後的位置差(三 lane 中位數);≥1 m SOFT、≥3 m medium,永遠不是 HIT")


# ---------------- 局部姿態備援 / 時間平移 / 例外 ----------------

def test_syn_local_attitude_fallback():
    """沒有 anchors 的 t_warn / t_loss 時由 ATT 推力向量誤差自行估算(>10° / >30° 持續 0.3 s)。"""
    ctx = _gps_switch_log(att=((148.0, 149.5, 15.0), (149.5, 160.0, 40.0)))
    f = pick(run(ctx, anc(ctx), "局部"), "source_switch_audit")
    v = f.values
    assert v["anchors_local"] is True and near(v["t_warn"], 148.0, 0.15) and near(v["t_loss"], 149.5, 0.15), v
    assert v["phase"] == "consequence" and "後果" in f.text_zh
    print(f"  局部 t_warn {v['t_warn']:.2f} / t_loss {v['t_loss']:.2f} → 150.3 s 的切換判為後果")


def _scenario(shift=0.0, t_base=T_BASE):
    return make_log(shift=shift, t_base=t_base, sets=((0.0, 3), (150.3, 2), (151.1, 3)), gps_bad=((60.0, INF),),
                    glitch=((150.3, 151.1),), err=((150.42, 11, 2),), visp_gaps=((120.0, 122.2),),
                    ts_runs=((180.0, 184.0, 3),), gap=_const_gap(1.6), vel=(1.0, 0.5), jumps=((150.3, 2.0, 0.0),))


def test_syn_time_shift_invariance():
    """整份 log 平移 37.5 s(t_base 也換)→ 狀態 / 情境 / 數值相同,t_onset 剛好差 37.5 s。"""
    c0, c1 = _scenario(), _scenario(shift=37.5, t_base=-3.0)
    f0 = run(c0, anc(c0), "平移 0")
    f1 = run(c1, anc(c1), "平移 37.5")
    assert statuses(f0) == statuses(f1), (statuses(f0), statuses(f1))
    for a, b in zip(f0, f1):
        if a.t_onset is not None:
            assert near(b.t_onset - a.t_onset, 37.5, 1e-6), (a.id, a.t_onset, b.t_onset)
    for det, key in (("ekf_vio_disagreement", "median_gap_m"), ("source_switch_position_jump", "max_jump_m"),
                     ("nav_velocity_gps_fallback_risk", "risk_s")):
        assert pick(f0, det).values[key] == pick(f1, det).values[key], (det, key)
    print(f"  {len(f0)} 筆 Finding 平移後一致:{[s[1] for s in statuses(f0)]}")


def test_guard_errors_become_na():
    ctx = make_log(sets=((0.0, 3),))

    def boom(*a, **k):
        raise RuntimeError("boom")

    ctx.table = boom
    ctx.series = boom
    fs = dn.run(ctx, anc(ctx))
    assert {f.id for f in fs} == set(OWNED) and all(f.status == NA and f.context == "error" for f in fs), statuses(fs)
    assert all("RuntimeError" in f.values["error"] for f in fs)
    print("  每個偵測器的例外都被包成 NA(context=error),仍然每個 id 一筆")


def test_not_armed_synthetic():
    ctx = make_log()
    a = Anchors(intervals=[])
    fs = run(ctx, a, "未解鎖")
    assert len(fs) == len(OWNED) and all(f.status == NA and f.context == "not_armed" for f in fs)
    print("  interval=None → 每個 id 一筆 NA")


# ======================================================================================
# 5b. 審查修正的回歸測試(nav reviewer #1–#12)
# ======================================================================================

def _strict(fs, label):
    """values 與時間都要是嚴格 JSON(沒有 NaN / ±inf);文字不能出現 None / inf / nan。"""
    for f in fs:
        json.dumps(f.values, allow_nan=False)
        for k in ("t_onset", "t_alarm"):
            v = getattr(f, k)
            assert v is None or np.isfinite(v), (label, f.id, f.context, k, v)
        for bad in ("None", "inf", "nan"):
            assert bad not in f.text_zh, (label, f.id, bad, f.text_zh)


def _crash_anc(ctx, **kw):
    a = anc(ctx, end_type="crash_check", **kw)
    a.t_disarm = a.interval.t1
    return a


def test_fix1_gate_ground_run_then_air_run():
    """#1:起飛前地面上的 gate fail 不能蓋掉之後真正的空中發散(之前只看第一段 → SOFT armed_ground、H_NAV 只剩 1.2 分)。"""
    from core.crash import hypotheses as hyp
    kw = dict(arm=((50.0, 250.0, 0),), takeoff=70.0)                    # EV28 = 120 s
    res = {}
    for label, ratio in (("只有空中", ((150.0, 155.0, "SV", 3.0),)),
                         ("地面再空中", ((55.0, 57.5, "SP", 2.0), (150.0, 155.0, "SV", 3.0)))):
        ctx = make_log(ratio=ratio, **kw)
        a = _crash_anc(ctx, t_warn=160.0, t_loss=162.0, t_detect=165.0)
        fs = run(ctx, a, label)
        for f in fs:
            f.tag = engine.tag_time(f.t_onset, a, standing=f.id in engine.STANDING_IDS or f.context == "standing")
        g = pick(fs, "ekf_testratio_gate")
        hn = {h.id: h for h in hyp.score_all(fs, a)}["H_NAV_FAILURE"]
        contrib = sum(e["contrib"] for e in hn.evidence if e["det"] == "ekf_testratio_gate")
        res[label] = (g, hn, contrib)
        assert (g.status, g.context, g.severity) == (HIT, "air", "high") and near(g.t_onset, 150.0, 0.06), \
            (label, g.status, g.context, g.t_onset)
        assert hn.gate_ok is True and near(contrib, 4.0, 1e-6), (label, hn.gate_ok, contrib)
    g = res["地面再空中"][0]
    assert g.values["n_runs"] == 2 and g.values["n_air_runs"] == 1, g.values
    assert [r["air"] for r in g.values["runs"]] == [False, True] and near(g.values["runs"][0]["t_start"], 55.0, 0.06)
    assert "地面另有 1 段" in g.text_zh and "55.0" in g.text_zh and "(解鎖在地面" not in g.text_zh, g.text_zh
    # 跨過起飛的 gate fail:離地部分 ≥1.0 s 才算空中(0.5 s → 地面 SOFT;2 s → 空中 HIT,onset = 段起點)
    for tb, want in ((120.5, (SOFT, "armed_ground")), (122.0, (HIT, "air"))):
        ctx = make_log(ratio=((118.0, tb, "SV", 2.0),), **kw)
        f = pick(run(ctx, anc(ctx), f"跨起飛 {tb}"), "ekf_testratio_gate")
        assert (f.status, f.context) == want and near(f.t_onset, 118.0, 0.06), (tb, f.status, f.context, f.t_onset)
    print(f"  地面 55 s + 空中 150 s:HIT air onset {g.t_onset:.2f}、H_NAV 測試比貢獻 {res['地面再空中'][2]:.1f}(之前 SOFT / 1.2)")


def test_fix2_switch_into_vio_set_before_any_visp():
    """#2:切到 VIO 組時 VISP 還沒有任何一筆(年齡 inf)—— 之前 source_switch_audit 會 TypeError → NA,
    vio_dropout 給 t_onset = −inf(engine 貼成 ground_info、嚴格 JSON 失敗)。"""
    ctx = make_log(sets=((0.0, 1), (100.0, 3)), visp_start=120.0)
    a = _crash_anc(ctx, t_warn=150.0, t_loss=152.0, t_detect=155.0)
    fs = run(ctx, a, "切到還沒有 VISP 的 VIO 組")
    _strict(fs, "切到還沒有 VISP 的 VIO 組")
    f = pick(fs, "source_switch_audit")
    r = f.values["switches"][0]
    assert f.status == HIT and f.context == "switch" and near(f.t_onset, 100.0, 0.05), (f.status, f.context, f.values.get("error"))
    assert r["flags"] == ["vio_never_received"] and r["vio_age_s"] is None and near(r["vio_first_t"], 120.0, 0.05), r
    assert "還沒收到任何 VIO 資料" in f.text_zh and "120.00s" in f.text_zh, f.text_zh
    vm = pick(fs, "vio_dropout", "in_vio_mode")
    e = vm.values["event"]
    assert vm.status == HIT and near(vm.t_onset, 100.0, 0.05) and near(vm.t_alarm, 101.5, 0.05), (vm.t_onset, vm.t_alarm)
    assert e["never_received"] is True and e["t_last"] is None and near(e["resume"], 120.0, 0.05) and near(e["dur"], 20.0, 0.1), e
    assert engine.tag_time(vm.t_onset, a) == "precursor" and engine.tag_time(f.t_onset, a) == "precursor"
    # VIO 組從解鎖起就在用,VISP 解鎖 30 s 後才開始 → onset = 解鎖,不是 −inf
    ctx = make_log(sets=((0.0, 3),), visp_start=40.0)
    a = _crash_anc(ctx, t_warn=150.0, t_loss=152.0, t_detect=155.0)
    fs = run(ctx, a, "解鎖時 VIO 還沒來")
    _strict(fs, "解鎖時 VIO 還沒來")
    vm = pick(fs, "vio_dropout", "in_vio_mode")
    assert near(vm.t_onset, 10.0, 1e-6) and near(vm.t_alarm, 11.5, 1e-6) and "解鎖時" in vm.text_zh, (vm.t_onset, vm.text_zh)
    assert engine.tag_time(vm.t_onset, a) == "precursor", engine.tag_time(vm.t_onset, a)
    # 第一筆 VISP 在切換後 0.3 s 就到(≤ SW_VIO_AGE_S)= VIO 剛好在切換時啟動 → 不算
    ctx = make_log(sets=((0.0, 1), (100.0, 3)), visp_start=100.3)
    fs = run(ctx, anc(ctx), "切換後 0.3 s 就有 VISP")
    f = pick(fs, "source_switch_audit")
    sw0 = f.values["switches"][0]
    assert f.status == CLEAR and sw0["flags"] == [] and near(sw0["vio_first_t"], 100.3, 0.05), f.values
    assert all(x.status != HIT for x in every(fs, "vio_dropout")), statuses(fs)
    print("  切到 / 解鎖時還沒有 VISP:source_switch_audit HIT vio_never_received;vio_dropout onset = 切換 / 解鎖(有限值、嚴格 JSON)")


def test_fix3_pre_arm_timeout_overlap_and_ground():
    """#3:解鎖前就開始的逾時,解鎖後的部分也要 ≥2 s;0.12 s 的重疊之前會變成「飛行中」HIT(H_NAV 3.0 分、通過 _gate_nav)。"""
    ctx = make_log(ts_runs=((6.0, 10.1, 3),))
    f = pick(run(ctx, anc(ctx, end_type="normal_landing"), "重疊 0.1 s"), "ekf_aiding_timeout")
    co = f.values["pre_arm_carryover"]
    assert f.status == CLEAR and f.context == "none" and len(co) == 1 and co[0]["overlap_s"] < dn.TS_MIN_DUR_S, f.values
    assert near(co[0]["run_start"], 6.0, 0.05) and "不算" in f.text_zh and "飛行中" not in f.text_zh, f.text_zh
    # 解鎖後 4 s 都在地面(15 s 才起飛)→ SOFT armed_ground,不叫「飛行中」
    ctx = make_log(ts_runs=((8.0, 14.0, 3),))
    f = pick(run(ctx, anc(ctx), "地面逾時"), "ekf_aiding_timeout")
    assert (f.status, f.context, f.severity) == (SOFT, "armed_ground", "info") and "解鎖後在地面" in f.text_zh, \
        (f.status, f.context, f.text_zh)
    assert "飛行中" not in f.text_zh and f.values["n_air_runs"] == 0
    # 延遲與 EKFCHECK 配對用「逾時真正開始」(6.0 s),不是夾過的 10.0 s:GPS 1.81 s 後就沒定位 → 延遲 4.19 s(夾過的話是 8.19 s)
    ctx = make_log(ts_runs=((6.0, 14.0, 3),), takeoff=1.0, gps_bad=((2.0, INF),), err=((7.0, 16, 2),))
    f = pick(run(ctx, anc(ctx), "延遲從原起點"), "ekf_aiding_timeout")
    r0 = f.values["runs"][0]
    assert f.status == HIT and f.context == "in_flight" and near(f.t_onset, 10.0, 0.05), (f.status, f.context)
    assert near(r0["lag_s"], 4.19, 0.05) and near(r0["run_start"], 6.0, 0.05) and near(r0["ekfcheck_t"], 7.0, 1e-6), r0
    print(f"  重疊 {co[0]['overlap_s']:.2f}s → CLEAR(只記 carryover);地面 4 s → SOFT;延遲以原起點計 {r0['lag_s']:.2f}s")


def test_fix4_vio_set_active_but_visp_stopped_before_arm():
    """#4:VIO 組整趟都在用、VISP 在解鎖前就停了 —— 這是最嚴重的 VIO 中斷,之前回 NA 'no_visp_in_flight'。"""
    ctx = make_log(sets=((0.0, 3),), visp_stop=8.0)
    fs = run(ctx, anc(ctx), "解鎖前 VISP 已停")
    _strict(fs, "解鎖前 VISP 已停")
    vm = pick(fs, "vio_dropout", "in_vio_mode")
    e = vm.values["event"]
    assert vm.status == HIT and vm.values["permanent"] is True and near(vm.t_onset, 10.0, 1e-6), (vm.status, vm.t_onset)
    assert near(e["t_last"], 7.99, 0.05) and e["entered_stale"] is True and vm.values["n_visp_in_flight"] == 0, e
    assert "解鎖時使用的是 VIO 組" in vm.text_zh, vm.text_zh
    # GPS 組 + VISP 在解鎖前停 → 仍然 NA(VIO 沒在用)
    ctx = make_log(sets=((0.0, 1),), visp_stop=8.0)
    f = pick(run(ctx, anc(ctx), "GPS 組"), "vio_dropout")
    assert f.status == NA and f.context == "no_visp_in_flight", (f.status, f.context)
    print(f"  VIO 組 + 解鎖前就沒 VISP → HIT in_vio_mode onset = 解鎖 {vm.t_onset:.1f}s;GPS 組 → NA")


def test_fix5_position_jump_post_window_clipped_at_disarm():
    """#5:切換後窗不能伸到上鎖之後(失事後 EKF 位置是垃圾值)。209.9 s 切換、210 s 上鎖、210.2 s 位置跳 20 m。"""
    ctx = make_log(sets=((0.0, 1), (209.9, 3)), vel=(2.0, 0.0), jumps=((209.9, 3.0, 0.0), (210.2, 20.0, 0.0)))
    f = pick(run(ctx, anc(ctx), "上鎖前 0.1 s 切換"), "source_switch_position_jump")
    r0 = f.values["switches"][0]
    assert near(f.values["max_jump_m"], 3.0, 0.15) and r0["clipped_at_disarm"] is True and r0["short_dwell"] is True, f.values
    # 太靠近上鎖、剩不到 2 筆樣本 → 該次不量(NA),不是拿上鎖後的資料湊
    ctx = make_log(sets=((0.0, 1), (209.99, 3)), vel=(2.0, 0.0), jumps=((210.2, 20.0, 0.0),))
    f = pick(run(ctx, anc(ctx), "上鎖前 0.01 s 切換"), "source_switch_position_jump")
    assert f.status == NA and f.context == "insufficient_data", (f.status, f.context, f.values.get("switches"))
    print(f"  上鎖後的 20 m 跳變不再混入:量到 {r0['jump_m']:.2f} m(之前 23.00 m)")


def test_fix6_timeout_text_by_phase():
    """#6:t_warn 之後、t_loss 之前才逾時 = 發散期間的反應,不能寫「肇因候選」。"""
    ctx = make_log(gps_bad=((93.0, INF),), ts_runs=((100.0, 103.5, 3),))
    f = pick(run(ctx, _crash_anc(ctx, t_warn=99.0, t_loss=110.0, t_detect=115.0), "發散期間"), "ekf_aiding_timeout")
    assert f.values["phase"] == "development" and "姿態發散期間的反應而非起因" in f.text_zh and "肇因候選" not in f.text_zh, f.text_zh
    f = pick(run(ctx, _crash_anc(ctx, t_warn=105.0, t_loss=110.0, t_detect=115.0), "先兆"), "ekf_aiding_timeout")
    assert f.values["phase"] == "precursor" and "肇因候選" in f.text_zh and "請檢查位置來源" in f.text_zh, f.text_zh
    print("  發散期間 → 「反應而非起因」;t_warn 之前 → 「肇因候選」")


def test_fix7_gate_attribution_ignores_stale_sacc():
    """#7:GPS 沒定位時 GPA.SAcc 是舊值(124 整段 0.0),不能說「GPS 速度精度正常」。"""
    ctx = make_log(sets=((0.0, 3),), gps_bad=((30.0, INF),), ratio=((150.0, 153.0, "SP", 2.0),), sacc=((30.0, INF, 0.0),))
    f = pick(run(ctx, anc(ctx), "GPS 無定位"), "ekf_testratio_gate")
    assert f.status == HIT and f.values["gps_fix_in_run"] is False and f.values["sacc_max"] is None, f.values
    assert "GPS 沒有定位" in f.text_zh and "速度精度正常" not in f.text_zh and f.values["attribution"] == "vio_or_other", f.text_zh
    # 舊的 SAcc 很大也不能歸因 GPS 品質(沒定位時根本沒在用 GPS)
    ctx = make_log(sets=((0.0, 3),), gps_bad=((30.0, INF),), ratio=((150.0, 153.0, "SP", 2.0),), sacc=((30.0, INF, 5.0),))
    assert pick(run(ctx, anc(ctx), "舊 SAcc 5"), "ekf_testratio_gate").values["attribution"] == "vio_or_other"
    # 有定位時照舊:正常 → 「有定位且速度精度正常」;≥2 m/s → GPS 品質(test_syn_testratio_gate)
    ctx = make_log(ratio=((150.0, 153.0, "SP", 2.0),))
    f = pick(run(ctx, anc(ctx), "GPS 正常"), "ekf_testratio_gate")
    assert f.values["gps_fix_in_run"] is True and "速度精度正常" in f.text_zh
    print("  GPS 無定位 → 不採用舊 SAcc、文字說 GPS 沒有定位")


def test_fix8_unknown_active_set_no_confident_claims():
    """#8:沒有 XKFS / EV 85-87 時使用中的組未知;之前 nav_velocity 回 CLEAR、vio_dropout 說「不是 VIO 組」。"""
    ctx = make_log(sets=((0.0, 3),), xkfs=False, gps_bad=((30.0, INF),), visp_stop=150.0)
    fs = run(ctx, anc(ctx), "組別未知")
    vd = every(fs, "vio_dropout")
    assert [(f.status, f.context) for f in vd] == [(SOFT, "in_vio_mode")] and vd[0].values["set_unknown"] is True, statuses(vd)
    assert vd[0].values["set_source"] == "assumed" and "無法確認當時是否使用 VIO 組" in vd[0].text_zh, vd[0].text_zh
    assert "不是 VIO 組" not in vd[0].text_zh
    nv = pick(fs, "nav_velocity_gps_fallback_risk")
    assert nv.status == SOFT and nv.values["set_unknown"] is True and nv.values["set_source"] == "assumed", (nv.status, nv.values)
    assert "無法確認實際使用哪一組" in nv.text_zh, nv.text_zh
    assert pick(fs, "ekf_vio_disagreement").context == "set_unknown"
    # 速度新息凍結是實測 → 組別未知也可以 HIT
    ctx = make_log(sets=((0.0, 1),), xkfs=False, gps_bad=((30.0, INF),), ivn_frozen=((30.0, INF),))
    assert pick(run(ctx, anc(ctx), "組別未知 + IVN 凍結"), "nav_velocity_gps_fallback_risk").status == HIT
    # 沒有任何組用 VIO、每組都有速度來源 → 不管用哪組都成立 → CLEAR / 「所有來源組都不用 VIO」
    P = dict(SRC_124, EK3_SRC3_POSXY=3, EK3_SRC3_VELXY=3)
    ctx = make_log(params=P, xkfs=False, gps_bad=((30.0, INF),), visp_stop=150.0)
    fs = run(ctx, anc(ctx), "組別未知、都不用 VIO")
    assert pick(fs, "nav_velocity_gps_fallback_risk").status == CLEAR
    f = pick(fs, "vio_dropout")
    assert (f.status, f.context) == (SOFT, "stream_end") and "所有來源組都不用 VIO" in f.text_zh, (f.status, f.context, f.text_zh)
    print("  組別未知:依賴組別的結論降為 SOFT(並寫明無法確認);最壞情況也沒風險才 CLEAR")


def test_fix9_nan_fields_no_warnings():
    """#9:XKF4.SS/TS/FS、GPA.SAcc 等欄位出現 NaN 時不能出現 'invalid value encountered in cast' / 'All-NaN slice'
    (在 -W error 下會變 NA/error)。每 7 筆塞一個 NaN(含 XKFS.SS 序列)。"""
    import warnings
    ctx = make_log(sets=((0.0, 3), (150.3, 2), (151.1, 3)), gps_bad=((60.0, INF),), glitch=((150.3, 151.1),),
                   ts_runs=((180.0, 184.0, 3),), ratio=((190.0, 193.0, "SV", 2.0),), fs=((195.0, 195.5, 4),))
    t_orig, s_orig = ctx.table, ctx.series

    def nanify(v):
        v = np.array(v, dtype=float)
        v[3::7] = np.nan
        return v

    def table(name, inst=None):
        tb = t_orig(name, inst)
        if tb is None or name in ("ARM", "EV", "MODE", "ERR"):
            return tb
        return {k: (nanify(v) if k != "t" and isinstance(v, np.ndarray) and v.dtype.kind == "f" else v) for k, v in tb.items()}

    def series(key):
        s = s_orig(key)
        return None if s is None else (s[0], nanify(s[1]))

    ctx.table, ctx.series = table, series
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fs = dn.run(ctx, anc(ctx))
    check_contract(fs, "NaN")
    errs = [(f.id, f.values.get("error")) for f in fs if f.context == "error"]
    assert not errs, errs
    assert pick(fs, "ekf_testratio_gate").status == HIT and pick(fs, "ekf_aiding_timeout").status == HIT, statuses(fs)
    print(f"  NaN 欄位 + RuntimeWarning→error:{len(fs)} 筆都沒有 error")


def test_fix11_text_wording():
    """#11a:沒有姿態異常時不能出現「這次切換此次飛行…」斷句;解鎖期間的切換不一律叫「飛行中」。(#11b 在 124 測試)"""
    ctx = _gps_switch_log()
    f = pick(run(ctx, anc(ctx), "沒有姿態異常"), "source_switch_audit")
    assert "這次切換此次飛行" not in f.text_zh and "沒有失控時間可對照因果" in f.text_zh, f.text_zh
    assert f.text_zh.startswith("解鎖期間") and "飛行中" not in f.text_zh, f.text_zh
    ctx = make_log(sets=((0.0, 3),), visp_gaps=((150.0, 152.0),))
    f = pick(run(ctx, anc(ctx), "VIO 中斷沒有姿態異常"), "vio_dropout")
    assert "這次中斷此次飛行" not in f.text_zh and "沒有失控時間可對照因果" in f.text_zh, f.text_zh
    print("  沒有姿態異常:「此次飛行沒有姿態異常(沒有失控時間可對照因果)」;切換文字用「解鎖期間」")


def test_fix12_short_interval_never_clear():
    """#12:解鎖 0 / 0.5 / 3 s 時「沒找到問題」不能當 CLEAR(資料不足以排除)→ NA;但短時間內的真實證據照常回報。"""
    ctx = make_log()
    for dur in (0.0, 0.5, 3.0):
        iv = ArmedInterval(0, 100.0, 100.0 + dur, 0)
        fs = run(ctx, Anchors(interval=iv, intervals=[iv]), f"解鎖 {dur}s")
        assert not [f for f in fs if f.status == CLEAR], (dur, statuses(fs))
    # 3.5 s 的解鎖裡切到沒有定位的 GPS 組 → 仍是 HIT
    ctx = _gps_switch_log(arm=((148.0, 151.5, 0),))
    f = pick(run(ctx, anc(ctx), "短解鎖內的切換"), "source_switch_audit")
    assert f.status == HIT and near(f.t_onset, 150.3, 0.05), (f.status, f.context)
    # ≥5 s 才可以 CLEAR
    ctx = make_log(arm=((100.0, 106.0, 0),))
    st = {f.id: f.status for f in run(ctx, anc(ctx), "6 s")}
    assert st["source_switch_audit"] == CLEAR and st["vio_dropout"] == CLEAR, st
    print("  解鎖 <5 s:沒有 CLEAR(只有 NA / 證據);短解鎖內的切換問題照樣 HIT")


# ======================================================================================
# 6. 決定性 / 效能
# ======================================================================================

def test_deterministic_and_json():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    a = crashkit.truth_anchors_124(ctx)

    def sig():
        return [(f.id, f.status, f.context, f.t_onset, json.dumps(f.values, sort_keys=True), f.text_zh) for f in dn.run(ctx, a)]

    assert sig() == sig()
    print("  決定性:同輸入同輸出;values 可 JSON 序列化")


def test_performance_per_detector():
    ctx = crashkit.load_ctx(crashkit.CRASH_LOG)
    for a in (crashkit.truth_anchors_124(ctx), crashkit.flight1_anchors_124(ctx)):
        worst = 0.0
        for fn in dn._DETECTORS:
            t0 = time.perf_counter()
            fn(ctx, a)
            dt = time.perf_counter() - t0
            worst = max(worst, dt)
            assert dt < 1.5, (fn.__name__, dt)
        t0 = time.perf_counter()
        dn.run(ctx, a)
        total = time.perf_counter() - t0
        assert total < 1.5, total
    print(f"  每個偵測器最慢 {worst * 1000:.0f} ms,整個模組 {total * 1000:.0f} ms(< 1500 ms)")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  !! 失敗:{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 60}\n{len(tests) - failed}/{len(tests)} 通過")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
