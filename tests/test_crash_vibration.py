"""振動 / IMU 偵測模組(core/crash/det_vib.py)驗證:python -m tests.test_crash_vibration

分四層:
  1. 真實 log 00000124:飛行 2(truth 錨點、只給 CRASH_CHECK 的 basic 錨點)與飛行 1(正常降落對照);
  2. 六個 baseline(從未解鎖的桌上 / 手持 log)→ 只能回 NA;
  3. 合成資料(假 ctx):證明偵測器沒有寫死 124 的數字 —— 124 只有一次失事,『壞』分支只能靠合成資料驗證;
  4. 純函式(runs / 時間平均 / Clip 增量 / 判定)與介面規則(id 必出現、values 可 JSON 序列化、效能)。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import crashkit as ck  # noqa: E402
from core.crash import det_vib  # noqa: E402
from core.crash.anchors import basic_anchors  # noqa: E402
from core.crash.types import CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval  # noqa: E402

IDS = ("vibration_flight_level", "imu_consistency_health")
_runs_cache: dict[tuple, dict] = {}


def check(cond, msg):
    assert cond, msg


def by_id(fs):
    out = {}
    for f in fs:
        out.setdefault(f.id, []).append(f)
    return out


def run_checked(ctx, a):
    """跑模組,檢查通用介面規則:兩個 id 都有、沒有 error、values 可 JSON 序列化、text_zh 非空。"""
    fs = det_vib.run(ctx, a)
    d = by_id(fs)
    for i in IDS:
        check(i in d and len(d[i]) >= 1, f"缺少偵測器 {i} 的 Finding")
    for f in fs:
        check(f.domain == "vibration", f"{f.id} domain={f.domain}")
        check(f.context != "error", f"{f.id} 執行失敗:{f.values}")
        check(f.status in (HIT, SOFT, CLEAR, NA), f.status)
        check(bool(f.text_zh), f"{f.id} 沒有 text_zh")
        json.dumps(f.values, allow_nan=False)          # 不可有 numpy 純量 / NaN
        if f.status in (HIT, SOFT):
            check(f.t_onset is not None, f"{f.id} {f.status} 沒有 t_onset")
    return {i: d[i][0] for i in IDS}


def real(name):
    return ck.load_ctx(name)


# ===================================================================== 假 ctx / 合成資料

class FakeCtx:
    """只實作 det_vib 用到的 ctx 介面:armed / instances() / table() / err_rows()。"""

    def __init__(self, armed, vibe=None, imu=None, errs=()):
        self.armed = armed
        self._tb = {"VIBE": vibe or {}, "IMU": imu or {}}
        self._errs = list(errs)          # [(t, subsys, ecode)]

    def instances(self, name):
        return sorted(self._tb.get(name, {}))

    def table(self, name, instance=None):
        d = self._tb.get(name, {})
        if not d:
            return None
        if instance is None:
            instance = min(d)
        return d.get(instance)

    def err_rows(self):
        if not self._errs:
            return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        e = np.array(self._errs, dtype=float)
        return e[:, 0], e[:, 1].astype(int), e[:, 2].astype(int)


def make_vibe(T, base=(7.0, 2.5, 2.0), sd=0.8, scale=1.0, bursts=(), clip_at=(), clip_start=0, seed=0, dt=0.1, t_off=0.0):
    """10 Hz VIBE 表。bursts:[(ta, tb, level)] 把 X 軸改成 level(±sd);clip_at:[(t, 增量)] 的累計計數。"""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T, dt)
    ax = [np.maximum(rng.normal(b * scale, sd * min(1.0, b / 3.0) * scale, t.size), 0.0) for b in base]
    for ta, tb, lvl in bursts:
        m = (t >= ta) & (t < tb)
        ax[0][m] = np.maximum(rng.normal(lvl, sd, int(m.sum())), 0.0)
    clip = np.full(t.size, float(clip_start))
    for tc, inc in clip_at:
        clip[t >= tc] += inc
    return {"t": t + t_off, "VibeX": ax[0], "VibeY": ax[1], "VibeZ": ax[2], "Clip": clip}


def make_imu(T, n=3, seed=1, hz=400.0, bias_acc=None, bias_gyr=None, impact=None, gh_zero=None, eg_inc=None,
             ghz_drop=None, temp_jump=None, gaps=(), t_off=0.0):
    """400 Hz IMU 表(各實例時間略有錯開、有固定偏差)。

    bias_acc / bias_gyr:{實例: (t_start, 向量)} 從 t_start 起加偏差;impact:(t, 幅度)共模衝擊 + IMU2 額外差異;
    gh_zero:{實例: t};eg_inc:{實例: [t,...]};ghz_drop:{實例: t};temp_jump:{實例: t};gaps:[(ta, tb)] 全實例資料空洞。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T, 1.0 / hz)
    g = 9.81
    roll = 0.15 * np.sin(2 * np.pi * t / 7.0)
    pitch = 0.10 * np.sin(2 * np.pi * t / 5.0 + 1.0)
    acc_c = np.stack([-g * np.sin(pitch), g * np.sin(roll) * np.cos(pitch), -g * np.cos(roll) * np.cos(pitch)])
    gyr_c = np.stack([0.15 * 2 * np.pi / 7 * np.cos(2 * np.pi * t / 7.0), 0.10 * 2 * np.pi / 5 * np.cos(2 * np.pi * t / 5.0 + 1),
                      0.05 * np.sin(2 * np.pi * t / 3.0)])
    spike = np.zeros(t.size)
    if impact is not None:
        ti, amp = impact
        spike = np.where((t >= ti) & (t < ti + 0.05), amp, 0.0)
    offs = [np.array([0.0, 0.0, 0.0]), np.array([-0.006, 0.03, -0.042]), np.array([0.052, -0.162, -0.047])]
    out = {}
    for i in range(n):
        A = acc_c + offs[i][:, None] + rng.normal(0, 0.015, (3, t.size))
        G = gyr_c + rng.normal(0, 0.002, (3, t.size))
        A[2] += -spike                                   # 共模衝擊(向下的力)
        if impact is not None and i == 2:                # IMU2 裝在別處:衝擊後看到不同的東西
            ti = impact[0]
            m = (t >= ti) & (t < ti + 0.4)
            A[0, m] += 18.0
            A[2] += -0.4 * spike
            G[1, m] += 1.2
        for dct, arr in ((bias_acc, A), (bias_gyr, G)):
            if dct and i in dct:
                ts, vec = dct[i]
                arr[:, t >= ts] += np.asarray(vec, dtype=float)[:, None]
        GH = np.ones(t.size)
        AH = np.ones(t.size)
        EG = np.zeros(t.size)
        EA = np.zeros(t.size)
        GHz = np.full(t.size, 3200.0)
        AHz = np.full(t.size, 3200.0)
        Tm = np.full(t.size, 45.0)
        if gh_zero and i in gh_zero:
            GH[t >= gh_zero[i]] = 0.0
        for te in (eg_inc or {}).get(i, []):
            EG[t >= te] += 1.0
        if ghz_drop and i in ghz_drop:
            m = t >= ghz_drop[i]
            GHz[m] = 1500.0
            AHz[m] = 1500.0
        if temp_jump and i in temp_jump:
            Tm[t >= temp_jump[i]] += 8.0
        tt = t + i * 0.0003
        keep = np.ones(t.size, dtype=bool)
        for ta, tb in gaps:
            keep &= ~((t >= ta) & (t < tb))
        out[i] = {"t": tt[keep] + t_off, "AccX": A[0][keep], "AccY": A[1][keep], "AccZ": A[2][keep],
                  "GyrX": G[0][keep], "GyrY": G[1][keep], "GyrZ": G[2][keep], "GH": GH[keep], "AH": AH[keep],
                  "EG": EG[keep], "EA": EA[keep], "GHz": GHz[keep], "AHz": AHz[keep], "T": Tm[keep]}
    return out


def anchors_for(armed, **kw):
    a = Anchors(intervals=list(armed))
    a.interval = armed[-1] if armed else None
    a.end_type = kw.pop("end_type", "normal_landing")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def one_flight(t0=10.0, t1=310.0):
    return [ArmedInterval(0, t0, t1, 0)]


# ===================================================================== 1. 真實 log:00000124

def test_real_124_crash_flight_truth_anchors():
    ctx = real("00000124")
    a = ck.truth_anchors_124(ctx)
    r = run_checked(ctx, a)
    v, im = r["vibration_flight_level"], r["imu_consistency_health"]
    check(v.status == CLEAR and v.context == "ok", f"振動應為 CLEAR/ok:{v.status} {v.context}")
    check(v.t_onset is None, "CLEAR 不該有 t_onset")
    vv = v.values
    check(vv["p95_effective"] <= 10.2, f"事件前 p95 應 ≤ 10.1(spec 6.9-10.1):{vv['p95_effective']}")
    check(8.0 <= vv["p95"] <= 10.2, vv["p95"])
    check(vv["max"] <= 11.3 and vv["max"] >= 10.5, f"事件前最大軸 max 應約 11.2:{vv['max']}")
    check(vv["frac_gt15"] == 0 and vv["frac_gt30"] == 0, "飛行中不該有 >15 的樣本")
    check(vv["clip_delta"] == 0, "撞擊前 Clip 增量應為 0")
    check(vv["ratio_vs_earlier"] is not None and 0.9 <= vv["ratio_vs_earlier"] <= 1.1,
          f"飛行 2 對飛行 1 的 p95 比值應約 1.0:{vv['ratio_vs_earlier']}")
    # 撞擊之後(後果):第一次 VIBE>30 在 1258.44、Clip 躍升在 1259.74(10 Hz 容許 ±0.15 s)
    pe = vv["post_event"]
    check(abs(pe["t_first_gt30"] - 1258.44) <= 0.15, f"第一次 VIBE>30 應在 1258.44:{pe}")
    check(abs(pe["t_first_clip"] - 1259.74) <= 0.15 and pe["clip_delta"] == 12, f"Clip 0→12 應在 1259.74:{pe}")
    check(pe["max"] >= 70, f"撞擊後 VibeZ 峰值約 78:{pe}")
    check(vv["window"][1] < 1255.0, f"統計窗必須在觸地 1255.2 之前結束:{vv['window']}")
    for nm, p in vv["per_instance"].items():           # IMU0/1 約 X 7、IMU2 的 X 約低 5 倍
        check(p["frac_gt30"] == 0, nm)
    check(vv["per_instance"]["IMU0"]["x"][0] > 6.5 and vv["per_instance"]["IMU2"]["x"][0] < 2.0, "三顆 IMU 的 X 軸水準不符")
    check("可排除振動" in v.text_zh, v.text_zh)

    check(im.status == CLEAR and im.context == "consistent", f"IMU 一致性應為 CLEAR:{im.status} {im.context}")
    iv = im.values
    check(iv["dA_max"] <= 0.12, f"事件前加速度差最大應 ≤ 0.112:{iv['dA_max']}")
    check(iv["dG_max"] <= 0.03, f"事件前角速度差最大應 ≤ 0.022:{iv['dG_max']}")
    check(iv["health_flags"] == [] and iv["free_runs"] == [], "不該有健康旗標 / 事件前不一致")
    po = iv["post_impact"]
    check(po is not None and 1258.0 <= po["t0"] <= 1258.4, f"IMU 差異應從撞擊 / 翻覆(1258.1-1258.4)才開始:{po}")
    check(po["pair"] == "0-2" and 4.0 <= po["dA"] <= 5.0, f"IMU0-IMU2 撞擊時加速度差約 4.6:{po}")
    check(iv["window"][1] < 1258.1, f"事件時間應早於 t_loss:{iv['window']}")
    check(iv["pairs"]["0-2"]["offset_acc"][1] < -0.1, "IMU2 的 Y 軸固定偏差應約 -0.16")
    check(iv["pairs"]["0-1"]["unpaired_n"] == 0 and iv["pairs"]["0-2"]["unpaired_n"] == 0, "124 的 IMU 空洞三顆同步")
    check(set(iv["health_checked"]) >= {"GH", "AH", "EG", "EA"}, iv["health_checked"])
    # 資料覆蓋:124 的 VIBE 掉樣 ≤ 0.8 s、IMU ≤ 0.45 s,都不算空洞;窗尾完整
    check(vv["coverage"]["n_holes"] == 0 and vv["coverage"]["last60_frac"] == 1.0 and vv["coverage"]["tail_gap_s"] < 0.5,
          vv["coverage"])
    check(all(c["n_holes"] == 0 for c in iv["coverage"]["instances"].values()), iv["coverage"])
    check(iv["pairs"]["0-2"]["coverage"]["last60_frac"] == 1.0 and iv["coverage"]["ref"] == 0, iv["coverage"])
    check(iv["log_gaps"]["n_long"] == 0, iv["log_gaps"])
    check("後果" in im.text_zh and "沒有 IMU 感測器故障" in im.text_zh and "3 顆 IMU" in im.text_zh, im.text_zh)
    # A7 已排除列:振動要帶數字(p95 ≤ 10.1 vs 30、Clip 到 1259.74 才增加)
    check("上限 30" in v.text_zh and "1259.74" in v.text_zh and "Clip" in v.text_zh, v.text_zh)
    print(f"  振動 CLEAR p95={vv['p95_effective']:.2f} max={vv['max']:.2f} 比值={vv['ratio_vs_earlier']};"
          f"IMU CLEAR dA={iv['dA_max']} dG={iv['dG_max']} 撞擊後差異自 {po['t0']:.2f}s")


def test_real_124_basic_anchors_only():
    """只有 CRASH_CHECK(basic_anchors,沒有 end_state 模組的衝擊錨點)也要得到同樣結論。

    VIBE 統計窗由本地偵測到的第一個硬撞擊(1258.18,62 m/s²)決定,不是 t_detect − 5 s 的粗估。"""
    ctx = real("00000124")
    a = basic_anchors(ctx)
    check(a.t_impact_peak is None and a.t_warn is None, "basic 錨點不該有衝擊 / 姿態錨點")
    r = run_checked(ctx, a)
    v, im = r["vibration_flight_level"], r["imu_consistency_health"]
    check(v.status == CLEAR and v.values["p95_effective"] <= 10.2, f"{v.status} {v.values.get('p95_effective')}")
    check(v.values["window_basis"] == "hard_shock" and 1257.0 <= v.values["window"][1] < 1258.2, v.values["window"])
    check(v.values["clip_delta"] == 0 and abs(v.values["post_event"]["t_first_clip"] - 1259.74) <= 0.15, v.values)
    check(im.status == CLEAR, f"{im.status} {im.text_zh}")
    check(im.values["window_basis"] == "hard_shock", "沒有錨點時應由本地偵測到的第一個硬撞擊決定事件時間")
    check(1257.5 <= im.values["window"][1] <= 1258.1, im.values["window"])
    check(im.values["post_impact"] is not None, "撞擊後的 IMU 差異應記在 post_impact")


def test_real_124_flight1_control():
    """飛行 1(正常降落)是陰性對照:不能有任何 HIT / SOFT。"""
    ctx = real("00000124")
    a = ck.flight1_anchors_124(ctx)
    r = run_checked(ctx, a)
    v, im = r["vibration_flight_level"], r["imu_consistency_health"]
    check(v.status == CLEAR and im.status == CLEAR, f"{v.status} / {im.status}")
    check(v.values["ratio_vs_earlier"] is None, "第一趟飛行沒有較早飛行可比較")
    check(v.values["p95_effective"] <= 10.2 and v.values["max"] <= 11.3, v.values)
    check(v.values["clip_delta"] == 0 and v.values["window_basis"] == "disarm", v.values)
    check(v.values["window"][1] <= 614.2, "正常降落應排除上鎖前最後 5 s")
    check(im.values["dA_max"] <= 0.12 and im.values["dG_max"] <= 0.03, im.values)
    check(im.values["post_impact"] is None, "正常降落不該有撞擊後差異")
    check(im.values["shock_windows"] == 1, "飛行 1 只有一次觸地衝擊(615.6 s,25 m/s²)")
    # 觸地時 IMU0-IMU2 差異最大 ~0.21(norm→mean),仍遠低於 0.5
    # B:正常降落不可有失事字眼
    for f in (v, im):
        for w in ("失事", "撞擊", "後果", "事件前", "翻覆"):
            check(w not in f.text_zh, f"飛行 1(正常降落)的 {f.id} 文字不該出現「{w}」:{f.text_zh}")
    check(v.values["post_event"]["t_first_gt30"] is None and v.values["post_event"]["clip_delta"] == 0, v.values["post_event"])
    print(f"  飛行 1:p95={v.values['p95_effective']:.2f}, dA={im.values['dA_max']}, dG={im.values['dG_max']}")


def test_real_124_flight2_before_1255_no_hit():
    """acceptance B:飛行 2 在 1255.0 之前(觸地前、沒有任何錨點)兩個偵測器都不能 HIT / SOFT。"""
    ctx = real("00000124")
    iv = ctx.armed[1]
    a = Anchors(intervals=list(ctx.armed))
    a.interval = ArmedInterval(iv.index, iv.t0, 1255.0, iv.method)
    a.end_type = "unknown"
    r = run_checked(ctx, a)
    v, im = r["vibration_flight_level"], r["imu_consistency_health"]
    check(v.status == CLEAR and im.status == CLEAR, f"{v.status} {v.context} / {im.status} {im.context}")
    check(v.values["p95_effective"] <= 10.2 and v.values["clip_delta"] == 0, v.values)
    check(v.values["ratio_vs_earlier"] is not None and 0.9 <= v.values["ratio_vs_earlier"] <= 1.1, v.values["ratio_vs_earlier"])
    check(im.values["dA_max"] <= 0.12 and im.values["dG_max"] <= 0.03 and im.values["shock_windows"] == 0, im.values)
    check(im.values["post_impact"] is None and im.values["window_basis"] == "disarm", im.values)


def test_real_124_engine_like_anchors():
    """引擎實際的情境:end_state / attitude / ground 模組已填好錨點與 a.impacts(和本地偵測是同一批衝擊)。
    衝擊窗要合併(3 次,不是 6 次);結論與 truth 錨點相同。"""
    ctx = real("00000124")
    a = ck.truth_anchors_124(ctx)
    a.impacts = [
        {"t_start": 1255.225, "t_peak": 1255.493, "t_end": 1255.513, "peak": 28.26, "dV": 1.22, "cls": "ground_contact"},
        {"t_start": 1258.183, "t_peak": 1258.411, "t_end": 1258.808, "peak": 59.32, "dV": 4.6, "cls": "hard_impact"},
        {"t_start": 1259.463, "t_peak": 1259.498, "t_end": 1259.958, "peak": 88.65, "dV": 6.89, "cls": "hard_impact"},
    ]
    r = run_checked(ctx, a)
    v, im = r["vibration_flight_level"], r["imu_consistency_health"]
    check(v.status == CLEAR and v.values["window_basis"] == "t_touch" and abs(v.values["window"][1] - 1254.225) < 0.01,
          v.values["window"])
    check(im.status == CLEAR and im.values["shock_windows"] == 3, f"{im.status} shock_windows={im.values.get('shock_windows')}")
    check(im.values["window_basis"] == "hard_shock" and 1257.8 <= im.values["window"][1] <= 1257.95, im.values["window"])
    check(1258.0 <= im.values["post_impact"]["t0"] <= 1258.4, im.values["post_impact"])
    # 本地偵測的衝擊段 = acceptance D(3) 的 124 衝擊列表(飛行 2 部分)
    tabs = det_vib._imu_tables(ctx)
    t, acc = det_vib._window_data(tabs, ctx.armed[1])
    ev = det_vib._shock_events(t, acc, None)
    check(len(ev) == 3, ev)
    for e, (ts, te, pk) in zip(ev, ((1255.225, 1255.513, 28.9), (1258.183, 1258.808, 62.1), (1259.463, 1259.958, 90.1))):
        check(abs(e["t_start"] - ts) <= 0.05 and abs(e["t_end"] - te) <= 0.05 and abs(e["peak"] - pk) <= 1.0, e)
    t1, acc1 = det_vib._window_data(tabs, ctx.armed[0])
    ev1 = det_vib._shock_events(t1, acc1, None)
    check(len(ev1) == 1 and abs(ev1[0]["t_start"] - 615.57) <= 0.05 and ev1[0]["peak"] < HARD(), ev1)


def HARD():
    return det_vib.HARD_PEAK


def test_real_124_no_anchor_fallback():
    """錨點全空(沒有 t_detect 也沒有 end_state)時,仍用本地硬撞擊偵測把撞擊後的差異歸為後果,不誤報 IMU 故障。"""
    ctx = real("00000124")
    a = Anchors(intervals=list(ctx.armed))
    a.interval = ctx.armed[1]
    a.end_type = "unknown"
    r = run_checked(ctx, a)
    check(r["imu_consistency_health"].status == CLEAR, r["imu_consistency_health"].text_zh)
    v = r["vibration_flight_level"]
    # 窗到本地硬撞擊 − 1 s(≈1257.18);含觸地後怠速,p95 與 max 仍低
    check(v.status == CLEAR and v.values["p95_effective"] < 12, v.values)
    check(v.values["window_basis"] == "hard_shock" and v.values["window"][1] < 1258.2, v.values["window"])


def _open_end_anchors(ctx, end_type):
    """124 飛行 2,但解鎖區間延長到 log 結束(像 log 結束時仍解鎖),而且沒有任何錨點。"""
    iv = ctx.armed[1]
    a = Anchors(intervals=list(ctx.armed))
    a.interval = ArmedInterval(iv.index, iv.t0, ctx.t_end, None, open_end=True)
    a.end_type = end_type
    return a


def test_real_124_open_end_no_anchors():
    """審查缺陷 1:撞擊比上鎖 / log 結束早很多(這裡 31 s)且沒有錨點時,撞擊後的 Clip 躍升與 VIBE 暴增不能混進
    「事件前」統計(舊版:HIT marginal+clipping、『事件前 Clip 計數增加 12 次』、窗到 1284.78)。"""
    ctx = real("00000124")
    for et in ("log_ended_armed", "unknown"):
        a = _open_end_anchors(ctx, et)
        r = run_checked(ctx, a)
        v, im = r["vibration_flight_level"], r["imu_consistency_health"]
        check(v.status == CLEAR and v.context == "ok", f"{et}:{v.status} {v.context} {v.text_zh}")
        vv = v.values
        check(vv["window_basis"] == "hard_shock" and vv["window"][1] < 1258.2, f"{et}:{vv['window']} {vv['window_basis']}")
        check(vv["clip_delta"] == 0 and vv["p95_effective"] <= 10.2 and vv["max"] <= 11.3, vv)
        check(abs(vv["post_event"]["t_first_clip"] - 1259.74) <= 0.15 and vv["post_event"]["clip_delta"] == 12, vv["post_event"])
        check(abs(vv["post_event"]["t_first_gt30"] - 1258.44) <= 0.15, vv["post_event"])
        check("事件前 Clip" not in v.text_zh and "統計窗內 Clip" not in v.text_zh and "Clip 無增加" in v.text_zh, v.text_zh)
        # 兩個偵測器用同一個本地硬撞擊
        check(im.status == CLEAR and im.values["window_basis"] == "hard_shock", f"{im.status} {im.values.get('window_basis')}")
        check(abs((vv["window"][1] + det_vib.EVENT_LEAD_S) - (im.values["window"][1] + det_vib.SHOCK_PRE_S)) < 0.01,
              (vv["window"], im.values["window"]))
    # log_ended_armed 算失事收尾 → 「可排除 / 後果」;unknown 且沒有失事錨點 → 只說衝擊之後(不用失事字眼)
    v = det_vib.vibration_flight_level(ctx, _open_end_anchors(ctx, "log_ended_armed"))[0]
    check("可排除振動" in v.text_zh and "後果" in v.text_zh, v.text_zh)
    u = det_vib.vibration_flight_level(ctx, _open_end_anchors(ctx, "unknown"))[0]
    check("強烈衝擊" in u.text_zh and "後果" not in u.text_zh and "可排除振動" not in u.text_zh, u.text_zh)


class CutCtx:
    """真實 ctx 的包裝:把 names 訊息在 [c0, c1) 的資料拿掉(模擬記錄中斷)。"""

    def __init__(self, ctx, c0, c1, names=("VIBE", "IMU")):
        self._c, self.c0, self.c1, self.names = ctx, c0, c1, names
        self.armed = ctx.armed
        self.t_end = ctx.t_end

    def instances(self, n):
        return self._c.instances(n)

    def err_rows(self):
        return self._c.err_rows()

    def table(self, n, inst=None):
        tb = self._c.table(n, inst)
        if tb is None or n not in self.names:
            return tb
        keep = ~((tb["t"] >= self.c0) & (tb["t"] < self.c1))
        return {k: (v[keep] if getattr(v, "shape", ())[:1] == tb["t"].shape else v) for k, v in tb.items()}


def test_real_124_missing_tail_is_not_clear():
    """審查缺陷 2:事件前 93 s 沒有 VIBE / IMU 資料(1165 s 起中斷)不能 CLEAR、不能說『可排除』;
    舊版兩個都 CLEAR,而且 log_gaps 完全沒提到這 93 s。"""
    ctx = real("00000124")
    a = ck.truth_anchors_124(ctx)
    r = run_checked(CutCtx(ctx, 1165.0, 1262.2), a)
    v, im = r["vibration_flight_level"], r["imu_consistency_health"]
    check(v.status == NA and v.context == "data_gap", f"{v.status} {v.context} {v.text_zh}")
    check(v.values["coverage"]["last60_frac"] == 0.0 and v.values["coverage"]["tail_gap_s"] > 80, v.values["coverage"])
    check("可排除" not in v.text_zh and "資料不足" in v.text_zh and "1164.9" in v.text_zh, v.text_zh)
    check(im.status == NA and im.context == "data_gap", f"{im.status} {im.context} {im.text_zh}")
    check("93.1 秒沒有記錄" in im.text_zh and "資料不足" in im.text_zh and "沒有 IMU 感測器故障" not in im.text_zh, im.text_zh)
    # 只缺 3 s(短於 TAIL_GAP_MAX_S、窗尾覆蓋 95%)→ 仍可 CLEAR,但要寫出空洞
    r = run_checked(CutCtx(ctx, 1251.0, 1254.0), a)
    v = r["vibration_flight_level"]
    check(v.status == CLEAR and v.values["coverage"]["n_holes"] == 1 and v.values["confidence"] == "low", v.values["coverage"])
    check("資料空洞" in v.text_zh, v.text_zh)


# ===================================================================== 2. baseline:從未解鎖 → 只能 NA

def test_baselines_not_armed_only_na():
    for name in ck.BASELINES:
        ctx = real(name)
        a = ck.not_armed_anchors(ctx)
        check(a.interval is None, f"{name} 應該沒有解鎖區間")
        fs = det_vib.run(ctx, a)
        d = by_id(fs)
        for i in IDS:
            check(i in d, f"{name} 缺 {i}")
        check(all(f.status == NA for f in fs), f"{name}:未解鎖 log 只能 NA:{[(f.id, f.status) for f in fs]}")
        check(all(f.context == "not_armed" for f in fs), [(f.id, f.context) for f in fs])
        check(all("未解鎖" in f.text_zh and "不適用" in f.text_zh for f in fs), name)     # C:印「不適用」,不是 OK
        check(all(f.t_onset is None for f in fs), name)
        for f in fs:
            json.dumps(f.values, allow_nan=False)
    print(f"  {len(ck.BASELINES)} 個 baseline 全部 NA/not_armed")


# ===================================================================== 3. 合成資料:振動

def vib_run(vibe, armed=None, errs=(), **anchor_kw):
    armed = armed or one_flight()
    ctx = FakeCtx(armed, vibe={i: v for i, v in enumerate(vibe)} if isinstance(vibe, (list, tuple)) else {0: vibe}, errs=errs)
    a = anchors_for(armed, **anchor_kw)
    return det_vib.vibration_flight_level(ctx, a)[0]


def test_vibe_synthetic_quiet_is_clear():
    f = vib_run(make_vibe(320.0))
    check(f.status == CLEAR and f.context == "ok", f"{f.status} {f.context} {f.values.get('p95')}")
    check(f.values["window_basis"] == "disarm" and f.values["window"] == [20.0, 305.0], f.values["window"])
    check(5 < f.values["p95"] < 11, f.values["p95"])


def test_vibe_synthetic_sustained_bad():
    f = vib_run(make_vibe(320.0, bursts=[(100.0, 200.0, 40.0)]))     # 100 s 的 40 m/s²(約 35% 的飛行時間)
    check(f.status == HIT and f.context == "bad" and f.severity == "high", f"{f.status} {f.context}")
    check(99.0 <= f.t_onset <= 101.0, f"t_onset 應是第一個 >30 的樣本(~100 s):{f.t_onset}")
    check(f.values["frac_gt30"] > 0.3 and f.values["longest_gt30_s"] >= 90, f.values)
    check("振動過大" in f.text_zh, f.text_zh)


def test_vibe_synthetic_short_burst_5s_is_bad():
    """一段 7 s 的 >30(只占 2.5% < 5%)仍要因為『連續 ≥ 5 s』判 bad;3 s 的就不算。"""
    f = vib_run(make_vibe(320.0, bursts=[(150.0, 157.0, 35.0)]))
    check(f.status == HIT and f.values["frac_gt30"] < FRAC(), f"{f.status} {f.values['frac_gt30']}")
    check(f.values["longest_gt30_s"] >= 5.0, f.values["longest_gt30_s"])
    g = vib_run(make_vibe(320.0, bursts=[(150.0, 153.0, 35.0)]))
    check(g.status == CLEAR, f"3 s 的短暫突波不該判 bad:{g.status} {g.context}")


def FRAC():
    return det_vib.FRAC_BAD


def test_vibe_synthetic_marginal():
    f = vib_run(make_vibe(320.0, base=(18.0, 2.5, 2.0), sd=1.5))
    check(f.status == SOFT and f.context == "marginal", f"{f.status} {f.context} p95={f.values['p95']}")
    check(15 <= f.values["p95"] < 30, f.values["p95"])
    check("偏高" in f.text_zh, f.text_zh)


def test_vibe_synthetic_late_onset_caught_by_last60():
    """整段 p95 看起來正常,但最後 60 s 才開始偏高(槳葉剛損傷):要被最近 60 s 的 p95 抓到。"""
    f = vib_run(make_vibe(320.0, bursts=[(290.0, 300.0, 20.0)]))      # 統計窗 20–305 s;10 s 的偏高 < 5% 的窗、約 17% 的最後 60 s
    check(f.values["p95"] < 15 <= f.values["p95_last60"], f"p95={f.values['p95']} last60={f.values['p95_last60']}")
    check(f.status == SOFT and f.context == "marginal", f"{f.status} {f.context}")
    check(289.0 <= f.t_onset <= 291.0, f"t_onset 應是 290 s 左右,而不是整段起點:{f.t_onset}")


def test_vibe_synthetic_late_onset_ignores_earlier_spike():
    """較早的單點突波(1 個樣本 > 15)不能變成最後 60 s 才開始的振動的起點。"""
    vb = make_vibe(320.0, bursts=[(290.0, 300.0, 20.0)])
    k = int(np.searchsorted(vb["t"], 50.0))
    vb["VibeX"][k] = 22.0
    f = vib_run(vb)
    check(f.status == SOFT and f.context == "marginal", f"{f.status} {f.context}")
    check(289.0 <= f.t_onset <= 291.0, f"t_onset 應是最近 60 s 窗內的 290 s,不是 50 s 的單點:{f.t_onset}")
    # 整段 p95 就已達標時,起點仍是整段第一次超標
    g = vib_run(make_vibe(320.0, bursts=[(60.0, 300.0, 20.0)]))
    check(g.status == SOFT and 59.0 <= g.t_onset <= 61.0, f"{g.status} {g.t_onset}")


def test_vibe_synthetic_post_event_wording():
    """事件後的 VIBE>30 / Clip:失事收尾 → 「後果」;正常降落(觸地在統計窗外)→ 不用失事字眼。"""
    vb = make_vibe(320.0, bursts=[(306.0, 307.5, 45.0)], clip_at=[(306.5, 3)])
    f = vib_run(vb)                                                       # 正常降落:窗到 305 s
    check(f.status == CLEAR and f.values["post_event"]["t_first_gt30"] is not None, f.values["post_event"])
    check("後果" not in f.text_zh and "可排除振動" not in f.text_zh and "降落階段" in f.text_zh, f.text_zh)
    g = vib_run(vb, end_type="crash_check", t_detect=309.0)
    check(g.status == CLEAR and "後果" in g.text_zh and "可排除振動" in g.text_zh, g.text_zh)
    check("Clip 到 306.5" in g.text_zh or "Clip 到 306.6" in g.text_zh, g.text_zh)
    # end_type 不明但有失事錨點 → 也算失事收尾
    h = vib_run(vb, end_type="unknown", t_first_hit=306.0)
    check("後果" in h.text_zh, h.text_zh)


def test_vibe_synthetic_clip_nan_is_not_clipping():
    """Clip 欄位有 NaN(訊息缺欄 / 轉換失敗)不能被轉成巨大的假增量。"""
    vb = make_vibe(320.0, clip_start=2)
    vb["Clip"][1000:1003] = np.nan
    f = vib_run(vb)
    check(f.status == CLEAR and f.values["clip_delta"] == 0, f"{f.status} {f.values['clip_delta']}")
    vb2 = make_vibe(320.0, clip_at=[(400.0, 0)])
    vb2["Clip"][3000:3002] = np.nan                                      # 在 t_warn 之後(事件後統計)
    g = vib_run(vb2, t_warn=250.0, end_type="crash_check")
    check(g.values["post_event"]["clip_delta"] == 0, g.values["post_event"])


def test_vibe_synthetic_ratio_vs_earlier_flight():
    armed = [ArmedInterval(0, 10.0, 260.0, 0), ArmedInterval(1, 400.0, 700.0, 0)]
    base = make_vibe(260.0, base=(4.0, 1.5, 1.5), sd=0.4, seed=1)
    later = make_vibe(700.0 - 260.0, base=(11.0, 1.5, 1.5), sd=0.8, seed=2, t_off=260.0)
    t = np.concatenate([base["t"], later["t"]])
    tab = {k: np.concatenate([base[k], later[k]]) for k in ("VibeX", "VibeY", "VibeZ", "Clip")}
    tab["t"] = t
    ctx = FakeCtx(armed, vibe={0: tab})
    a = anchors_for(armed)
    f = det_vib.vibration_flight_level(ctx, a)[0]
    check(f.status == SOFT and f.context == "marginal", f"{f.status} {f.context} {f.values.get('reasons')}")
    check(f.values["ratio_vs_earlier"] is not None and f.values["ratio_vs_earlier"] > 2.0, f.values["ratio_vs_earlier"])
    check(f.values["p95"] < 15, "絕對水準未達 15,是靠『比先前飛行高 2 倍』判 marginal")
    check("倍" in f.text_zh, f.text_zh)
    # 同樣 2.5 倍但絕對水準很低(2 → 5):不該標劣化
    base2 = make_vibe(260.0, base=(2.0, 1.0, 1.0), sd=0.2, seed=3)
    later2 = make_vibe(440.0, base=(5.0, 1.0, 1.0), sd=0.3, seed=4, t_off=260.0)
    tab2 = {k: np.concatenate([base2[k], later2[k]]) for k in ("t", "VibeX", "VibeY", "VibeZ", "Clip")}
    g = det_vib.vibration_flight_level(FakeCtx(armed, vibe={0: tab2}), a)[0]
    check(g.status == CLEAR, f"絕對水準仍很低時比值不該觸發:{g.status} {g.context} {g.values.get('ratio_max')}")


def test_vibe_synthetic_clip_logic():
    # 累計計數器從 2 起算、全程不動(像 119):不是削波
    f = vib_run(make_vibe(320.0, clip_start=2))
    check(f.status == CLEAR and f.values["clip_delta"] == 0, f"{f.status} {f.values['clip_delta']}")
    # 事件前 Clip 增加 → 正常振動 + 削波 = SOFT
    g = vib_run(make_vibe(320.0, clip_start=2, clip_at=[(120.0, 3), (180.0, 1)]))
    check(g.status == SOFT and g.context == "ok+clipping" and g.values["clip_delta"] == 4, f"{g.status} {g.context} {g.values['clip_delta']}")
    check(119.5 <= g.t_onset <= 121.0, f"t_onset 應是第一次 Clip 增加:{g.t_onset}")
    # 振動偏高 + 削波 = HIT
    h = vib_run(make_vibe(320.0, base=(18.0, 2.5, 2.0), sd=1.5, clip_at=[(120.0, 5)]))
    check(h.status == HIT and h.context == "marginal+clipping", f"{h.status} {h.context}")
    # Clip 在事件(t_warn)之後才增加 = 後果:不影響 status,記在 post_event
    k = vib_run(make_vibe(320.0, clip_at=[(250.0, 12)]), t_warn=200.0)
    check(k.status == CLEAR and k.values["clip_delta"] == 0, f"{k.status} {k.values['clip_delta']}")
    check(k.values["post_event"]["clip_delta"] == 12 and abs(k.values["post_event"]["t_first_clip"] - 250.0) < 0.2, k.values["post_event"])
    # 計數器歸零(重開機 / 溢位):負增量不能算成 Clip
    m = vib_run(make_vibe(320.0, clip_at=[(100.0, 8), (200.0, -8)]))
    check(m.values["clip_delta"] == 8, f"歸零的負增量不該抵銷或被算成削波:{m.values['clip_delta']}")


def test_vibe_synthetic_post_event_is_consequence():
    """t_warn 之後振動暴增(撞擊後果)不能讓 status 變 HIT;窗必須在 t_warn − 1 s 結束。"""
    f = vib_run(make_vibe(320.0, bursts=[(250.0, 320.0, 60.0)]), t_warn=250.0, end_type="crash_check")
    check(f.status == CLEAR, f"{f.status} {f.context}")
    check(f.values["window"][1] == 249.0 and f.values["window_basis"] == "t_warn", f.values["window"])
    check(abs(f.values["post_event"]["t_first_gt30"] - 250.0) < 0.2 and f.values["post_event"]["max"] > 50, f.values["post_event"])
    check("後果" in f.text_zh, f.text_zh)
    # 但事件『之前』就有同樣的振動 → HIT
    g = vib_run(make_vibe(320.0, bursts=[(150.0, 230.0, 60.0)]), t_warn=250.0, end_type="crash_check")
    check(g.status == HIT and 149.0 <= g.t_onset <= 151.0, f"{g.status} {g.t_onset}")
    # 錨點的優先順序:最早的錨點決定窗(t_first_hit/t_loss/t_impact_peak/t_touch)
    h = vib_run(make_vibe(320.0), t_loss=200.0, t_first_hit=210.0, t_impact_peak=215.0, t_touch=190.0)
    check(h.values["window"][1] == 189.0 and h.values["window_basis"] == "t_touch", h.values["window"])


def test_vibe_synthetic_err29_failsafe():
    f = vib_run(make_vibe(320.0), errs=[(120.0, 29, 1)])
    check(f.status == HIT and f.context == "bad" and abs(f.t_onset - 120.0) < 1e-6, f"{f.status} {f.context} {f.t_onset}")
    # 事件之後的 ERR 29 是後果
    g = vib_run(make_vibe(320.0), errs=[(280.0, 29, 1)], t_warn=250.0)
    check(g.status == CLEAR, f"{g.status}")
    # 其它 Subsys 不算
    h = vib_run(make_vibe(320.0), errs=[(120.0, 12, 1), (130.0, 6, 1)])
    check(h.status == CLEAR, h.status)


def test_vibe_synthetic_instances_independent():
    """一顆 IMU 高、其餘正常:用各自的水準判斷,不跨實例比較;IMU2 本來就低很多不能被拿來『稀釋』。"""
    good = make_vibe(320.0, seed=1)
    low = make_vibe(320.0, scale=0.2, seed=2)
    bad = make_vibe(320.0, base=(35.0, 3.0, 3.0), sd=2.0, seed=3)
    f = vib_run([good, good, low])
    check(f.status == CLEAR, f.status)
    g = vib_run([good, low, bad])
    check(g.status == HIT and g.values["worst_instance"] == "IMU2", f"{g.status} {g.values['worst_instance']}")
    check(g.values["per_instance"]["IMU0"]["p95"] < 15 and g.values["per_instance"]["IMU2"]["p95"] > 30, g.values["per_instance"])


def test_vibe_synthetic_na_cases():
    # 未解鎖
    ctx = FakeCtx([], vibe={0: make_vibe(100.0)})
    f = det_vib.vibration_flight_level(ctx, Anchors())[0]
    check(f.status == NA and f.context == "not_armed", f"{f.status} {f.context}")
    # 沒有 VIBE
    armed = one_flight()
    f = det_vib.vibration_flight_level(FakeCtx(armed), anchors_for(armed))[0]
    check(f.status == NA and f.context == "no_data", f"{f.status} {f.context}")
    # 事件前資料太短(解鎖後 10 s 略過,剩 < 20 s)
    armed = [ArmedInterval(0, 10.0, 40.0, 0)]
    f = det_vib.vibration_flight_level(FakeCtx(armed, vibe={0: make_vibe(60.0)}), anchors_for(armed))[0]
    check(f.status == NA and f.context == "insufficient_window", f"{f.status} {f.context}")
    # 事件發生得太早(t_warn 在解鎖後 15 s)
    armed = one_flight()
    f = det_vib.vibration_flight_level(FakeCtx(armed, vibe={0: make_vibe(320.0)}), anchors_for(armed, t_warn=25.0))[0]
    check(f.status == NA, f"{f.status} {f.context}")


def test_vibe_synthetic_impact_signature_no_imu():
    """審查缺陷 1(合成):只有 t_detect、撞擊比 CRASH_CHECK 早 5.5 / 7 s(超過 CRASH_LEAD_S)、沒有 IMU。
    舊版:SOFT ok+clipping、『事件前 Clip 計數增加 12 次』,同一段文字又說 VIBE>30 是後果。"""
    for lead in (5.5, 7.0):
        t_imp = 309.0 - lead
        vb = make_vibe(320.0, bursts=[(t_imp, t_imp + 1.5, 45.0)], clip_at=[(t_imp + 0.2, 12)])
        f = vib_run(vb, end_type="crash_check", t_detect=309.0)
        check(f.status == CLEAR and f.context == "ok", f"lead {lead}:{f.status} {f.context} {f.text_zh}")
        check(f.values["window_basis"] == "vibe_signature" and f.values["window"][1] < t_imp, f.values["window"])
        check(f.values["clip_delta"] == 0 and f.values["post_event"]["clip_delta"] == 12, f.values)
        check("後果" in f.text_zh and "統計窗內 Clip" not in f.text_zh and "事件前 Clip" not in f.text_zh, f.text_zh)
    # 三個實例:只有 Clip 同時躍升(沒有 VIBE 突波)也算撞擊特徵
    vbs = [make_vibe(320.0, seed=s, clip_at=[(200.0, 5)] if s < 2 else ()) for s in range(3)]
    g = vib_run(vbs, end_type="crash_check", t_detect=230.0)
    check(g.status == CLEAR and g.values["window_basis"] == "vibe_signature" and abs(g.values["window"][1] - 199.0) < 0.2,
          f"{g.status} {g.values['window']} {g.values['window_basis']}")
    # 只有一個實例時 Clip 躍升無法和「振動削波」區分 → 不當撞擊,留在統計窗內當證據
    h = vib_run(make_vibe(320.0, clip_at=[(200.0, 5)]), end_type="crash_check", t_detect=230.0)
    check(h.status == SOFT and h.context == "ok+clipping" and h.values["window_basis"] == "t_detect", f"{h.status} {h.context}")
    # 正常降落不用 VIBE 撞擊特徵:飛行中的短暫突波 + 削波仍是證據(SOFT),不會被截掉
    k = vib_run(make_vibe(320.0, bursts=[(150.0, 151.5, 45.0)], clip_at=[(150.2, 3)]))
    check(k.status == SOFT and k.values["window_basis"] == "disarm" and 150.0 <= k.t_onset <= 150.5, f"{k.status} {k.t_onset}")
    # 持續在 30 上下的壞振動有很多短暫超標,但之前不平靜 → 不能被當成撞擊而截掉(否則壞振動會被說成『可排除』)。
    # 真實情況:解鎖前馬達沒轉、VIBE 很低;一解鎖就開始振動 → 第一個短暫超標之前沒有足夠的「平靜」資料,也不能當撞擊
    vb = make_vibe(320.0, base=(29.0, 2.5, 2.0), sd=2.0)
    quiet = make_vibe(320.0, seed=5)
    pre = vb["t"] < 10.0
    for k_ in ("VibeX", "VibeY", "VibeZ"):
        vb[k_][pre] = quiet[k_][pre]
    m = vib_run(vb, end_type="crash_check", t_detect=309.0)
    check(m.status == HIT and m.context.startswith("bad") and m.values["window_basis"] == "t_detect",
          f"{m.status} {m.context} {m.values['window_basis']}")
    # ≥ 3 s 的超標段不是撞擊突波
    n = vib_run(make_vibe(320.0, bursts=[(250.0, 254.0, 35.0)]), end_type="crash_check", t_detect=309.0)
    check(n.values["window_basis"] == "t_detect", n.values["window_basis"])


def test_vibe_synthetic_hard_shock_from_imu():
    """有 IMU 時,VIBE 統計窗結束在 IMU 偵測到的第一個硬撞擊(與 imu_consistency_health 同一個),不依賴錨點。"""
    armed = one_flight()
    vb = make_vibe(320.0, bursts=[(302.0, 303.5, 45.0)], clip_at=[(302.2, 12)])
    ctx = FakeCtx(armed, vibe={0: vb}, imu=make_imu(320.0, impact=(302.0, 80.0)))
    for kw in ({"end_type": "crash_check", "t_detect": 309.0}, {"end_type": "unknown"}, {"end_type": "log_ended_armed"}):
        r = run_checked(ctx, anchors_for(armed, **kw))
        v, im = r["vibration_flight_level"], r["imu_consistency_health"]
        check(v.status == CLEAR and v.values["window_basis"] == "hard_shock" and abs(v.values["window"][1] - 301.0) < 0.05,
              f"{kw}:{v.status} {v.values['window']} {v.values['window_basis']}")
        check(im.status == CLEAR and im.values["window_basis"] == "hard_shock" and abs(im.values["window"][1] - 301.7) < 0.05,
              f"{kw}:{im.status} {im.values['window']}")
        check(v.values["post_event"]["clip_delta"] == 12, v.values["post_event"])
    # 錨點比本地撞擊早 → 用錨點
    r = run_checked(ctx, anchors_for(armed, end_type="crash_check", t_warn=280.0))
    check(r["vibration_flight_level"].values["window_basis"] == "t_warn", r["vibration_flight_level"].values["window_basis"])


def test_vibe_synthetic_coverage():
    """審查缺陷 2(合成):資料缺就不能 CLEAR;窗尾(事件前最後 60 s)缺一大段 → NA,中間的空洞 → CLEAR + 低信心 + 註明。"""
    def cut(vb, a0, a1):
        keep = ~((vb["t"] >= a0) & (vb["t"] < a1))
        return {k: v[keep] for k, v in vb.items()}
    # 資料在 60 s 停止,撞擊在 125 s(舊版:CLEAR『可排除振動為失事原因』)
    armed = [ArmedInterval(0, 20.0, 130.0, 0)]
    f = vib_run(cut(make_vibe(140.0), 60.0, 999.0), armed=armed, end_type="crash_check", t_first_hit=125.0)
    check(f.status == NA and f.context == "data_gap", f"{f.status} {f.context} {f.text_zh}")
    check(f.values["coverage"]["last60_frac"] == 0.0 and 63.0 < f.values["coverage"]["tail_gap_s"] < 65.0, f.values["coverage"])
    check("可排除" not in f.text_zh and "資料不足" in f.text_zh, f.text_zh)
    # 正常降落也一樣(上鎖前最後一段沒有資料):NA,而且不用失事字眼
    g = vib_run(cut(make_vibe(320.0), 200.0, 999.0))
    check(g.status == NA and g.context == "data_gap", f"{g.status} {g.context}")
    for w in ("失事", "撞擊", "後果", "事件前", "翻覆"):
        check(w not in g.text_zh, f"正常降落的 data_gap 文字不該出現「{w}」:{g.text_zh}")
    # 窗尾只缺 8 s(> TAIL_GAP_MAX_S 5 s)→ NA;缺 3 s → CLEAR 但註明空洞
    h = vib_run(cut(make_vibe(320.0), 297.0, 999.0))
    check(h.status == NA and h.context == "data_gap" and 7.9 < h.values["coverage"]["tail_gap_s"] < 8.2, h.values["coverage"])
    k = vib_run(cut(make_vibe(320.0), 302.0, 999.0))
    check(k.status == CLEAR and k.values["coverage"]["tail_gap_s"] < 3.2, f"{k.status} {k.values['coverage']}")
    # 中間 60 s 空洞(窗尾完整)→ CLEAR、confidence low、文字寫出空洞
    m = vib_run(cut(make_vibe(320.0), 100.0, 160.0))
    check(m.status == CLEAR and m.values["confidence"] == "low", f"{m.status} {m.values['confidence']}")
    cv = m.values["coverage"]
    check(cv["n_holes"] == 1 and 59.9 <= cv["holes_s"] <= 60.3 and cv["last60_frac"] == 1.0, cv)
    check("資料空洞" in m.text_zh and f"{cv['holes_s']:.1f} 秒" in m.text_zh, m.text_zh)
    # 完整資料:沒有空洞、confidence medium
    q = vib_run(make_vibe(320.0))
    check(q.values["coverage"]["n_holes"] == 0 and q.values["confidence"] == "medium" and "空洞" not in q.text_zh, q.values)
    # 證據(HIT)在資料不完整時仍保留,但要加註
    s = vib_run(cut(make_vibe(320.0, bursts=[(100.0, 200.0, 40.0)]), 230.0, 999.0))
    check(s.status == HIT and "注意" in s.text_zh, f"{s.status} {s.text_zh}")


def test_vibe_stats_last60_anchored_at_window_end():
    """「最近 60 s」是統計窗結束前的 60 s,不是最後 60 s 有資料的地方(舊版用 ts[-1] − 60,資料中斷時悄悄改看更早的時段)。"""
    t = np.arange(0, 200, 0.1)
    tb = {"t": t, "VibeX": np.full(t.size, 5.0), "VibeY": np.full(t.size, 2.0), "VibeZ": np.full(t.size, 1.0)}
    s = det_vib._vibe_stats(tb, 10.0, 300.0)            # 資料在 200 s 結束,窗到 300 s
    check(s["n_last"] == 0 and s["p95_last"] is None, (s["n_last"], s["p95_last"]))
    s = det_vib._vibe_stats(tb, 10.0, 199.95)
    check(s["n_last"] == 600 and s["p95_last"] == 5.0, (s["n_last"], s["p95_last"]))


def test_coverage_helper():
    ts = np.concatenate([np.arange(0.0, 50.0, 0.1), np.arange(70.0, 100.0, 0.1)])
    c = det_vib._coverage(ts, 0.0, 100.0)                  # 空洞 49.9–70.0(20.1 s),全落在最後 60 s(40–100)內
    check(len(c["holes"]) == 1 and abs(c["holes"][0][0] - 49.9) < 1e-6 and abs(c["holes"][0][1] - 70.0) < 1e-6, c["holes"])
    check(abs(c["frac"] - 0.799) < 1e-3 and abs(c["last_frac"] - (1 - 20.1 / 60.0)) < 1e-3, c)
    check(c["tail_gap_s"] < 0.2 and det_vib._cov_ok(c), c)
    c2 = det_vib._coverage(np.concatenate([np.arange(0.0, 45.0, 0.1), np.arange(80.0, 100.0, 0.1)]), 0.0, 100.0)
    check(c2["last_frac"] < det_vib.LAST_COV_MIN and not det_vib._cov_ok(c2), c2)   # 最後 60 s 缺 40 s → 不夠
    e = det_vib._coverage(np.zeros(0), 0.0, 100.0)
    check(e["frac"] == 0.0 and e["last_frac"] == 0.0 and e["t_last"] is None and not det_vib._cov_ok(e), e)
    # 短於 HOLE_S 的掉樣不算空洞;開頭 / 結尾缺資料算
    d = det_vib._coverage(np.array([0.5, 1.2, 2.0, 2.9, 3.7]), 0.0, 10.0)
    check(len(d["holes"]) == 1 and d["holes"][0] == (3.7, 10.0) and abs(d["tail_gap_s"] - 6.3) < 1e-9 and not det_vib._cov_ok(d), d)
    json.dumps(det_vib._cov_values(c), allow_nan=False)


def test_vibe_synthetic_time_shift_invariance():
    """整份資料(含解鎖區間)平移 5000 s,結論與數值不變 → 沒有寫死絕對時間。"""
    f0 = vib_run(make_vibe(320.0, bursts=[(100.0, 120.0, 40.0)]))
    armed = [ArmedInterval(0, 5010.0, 5310.0, 0)]
    f1 = vib_run(make_vibe(320.0, bursts=[(100.0, 120.0, 40.0)], t_off=5000.0), armed=armed)
    check(f0.status == f1.status == HIT, f"{f0.status} {f1.status}")
    check(abs((f1.t_onset - 5000.0) - f0.t_onset) < 1e-6 and f0.values["p95"] == f1.values["p95"], (f0.t_onset, f1.t_onset))


# ===================================================================== 3b. 合成資料:IMU 一致性

def imu_run(imu, armed=None, **anchor_kw):
    armed = armed or [ArmedInterval(0, 20.0, 130.0, 0)]
    ctx = FakeCtx(armed, imu=imu)
    a = anchors_for(armed, **anchor_kw)
    return det_vib.imu_consistency_health(ctx, a)[0]


def test_imu_synthetic_consistent_clear():
    f = imu_run(make_imu(140.0))
    check(f.status == CLEAR and f.context == "consistent", f"{f.status} {f.text_zh}")
    check(f.values["dA_max"] < 0.1 and f.values["dG_max"] < 0.01, f"{f.values['dA_max']} {f.values['dG_max']}")
    off = f.values["pairs"]["0-2"]["offset_acc"]
    check(abs(off[0] + 0.052) < 0.01 and abs(off[1] - 0.162) < 0.01, f"固定偏差(IMU0 − IMU2)應被估出並扣掉:{off}")
    check(f.values["window_basis"] == "disarm" and f.values["shock_windows"] == 0, f.values)


def test_imu_synthetic_offset_reference():
    """固定偏差優先用解鎖前的靜止段估計:故障佔 log 一半以上時,整份 log 的中位數會被故障帶走,
    解鎖前的參考段不會(故障 onset 仍落在偏差出現的時間)。"""
    f = imu_run(make_imu(140.0, bias_acc={1: (50.0, [1.0, 0.0, 0.0])}))      # 50 s 起偏 → 佔 log 64%
    check(f.values["offset_basis"] == "pre_arm", f.values["offset_basis"])
    check(f.status == HIT and 49.5 <= f.t_onset <= 50.5, f"{f.status} {f.t_onset}")
    # 解鎖前不足 5 s(log 一開頭就解鎖)→ 退回整份 log 的中位數,小偏差仍找得到
    g = imu_run(make_imu(140.0, bias_acc={1: (100.0, [1.0, 0.0, 0.0])}), armed=[ArmedInterval(0, 2.0, 130.0, 0)])
    check(g.status == HIT and 99.5 <= g.t_onset <= 100.5, f"{g.status} {g.t_onset}")
    h = imu_run(make_imu(140.0), armed=[ArmedInterval(0, 2.0, 130.0, 0)])
    check(h.status == CLEAR and h.values["offset_basis"] == "whole_log", f"{h.status} {h.values.get('offset_basis')}")


def test_imu_synthetic_accel_bias_fault_is_hit():
    f = imu_run(make_imu(140.0, bias_acc={1: (60.0, [1.0, 0.0, 0.0])}))
    check(f.status == HIT and f.context == "fault_before_event" and f.severity == "high", f"{f.status} {f.context}")
    check(59.5 <= f.t_onset <= 60.5, f"t_onset 應在偏差出現的 60 s:{f.t_onset}")
    check(f.values["free_runs"] and f.values["free_runs"][0]["pair"] == "0-1", f.values["free_runs"])
    check("解鎖飛行期間" in f.text_zh and "沒有撞擊" in f.text_zh, f.text_zh)        # 正常降落:不用「事件」字眼
    fc = imu_run(make_imu(140.0, bias_acc={1: (60.0, [1.0, 0.0, 0.0])}), end_type="crash_check", t_detect=129.0)
    check(fc.status == HIT and "事件之前" in fc.text_zh and "沒有撞擊" in fc.text_zh, fc.text_zh)
    # IMU2 壞也找得到(IMU0-IMU2 一組);IMU0 壞則兩組都超標
    g = imu_run(make_imu(140.0, bias_acc={2: (40.0, [0.0, 0.8, 0.0])}))
    check(g.status == HIT and g.values["free_runs"][0]["pair"] == "0-2", f"{g.status}")
    h = imu_run(make_imu(140.0, bias_acc={0: (40.0, [0.0, 0.0, 1.5])}))
    check(h.status == HIT and {r["pair"] for r in h.values["free_runs"]} >= {"0-1", "0-2"}, h.values["free_runs"])


def test_imu_synthetic_gyro_bias_fault_is_hit():
    f = imu_run(make_imu(140.0, bias_gyr={2: (70.0, [0.0, 0.0, 0.3])}))
    check(f.status == HIT and 69.5 <= f.t_onset <= 70.5, f"{f.status} {f.t_onset}")
    check(f.values["free_runs"][0]["dG"] > 0.15, f.values["free_runs"])


def test_imu_synthetic_soft_band():
    f = imu_run(make_imu(140.0, bias_acc={1: (60.0, [0.35, 0.0, 0.0])}))      # 0.3 ≤ dA < 0.5
    check(f.status == SOFT and f.context == "marginal", f"{f.status} {f.context} {f.values['free_runs']}")
    g = imu_run(make_imu(140.0, bias_acc={1: (60.0, [0.2, 0.0, 0.0])}))        # 0.2 < 0.3:雜訊以上但仍屬正常
    check(g.status == CLEAR, f"{g.status}")


def test_imu_synthetic_impact_differences_are_consequence():
    """共模衝擊(三顆都 > 20 m/s²,IMU2 另外看到不同的東西)造成的差異是後果:CLEAR,記在 post_impact。"""
    imu = make_imu(140.0, impact=(100.0, 80.0))
    f = imu_run(imu)
    check(f.status == CLEAR, f"{f.status} {f.context} {f.values['free_runs']}")
    po = f.values["post_impact"]
    check(po is not None and 99.0 <= po["t0"] <= 101.0 and po["dA"] > 1.0, po)
    check(f.values["window_basis"] == "hard_shock" and 99.0 <= f.values["window"][1] <= 100.0, f.values["window"])
    check(f.values["shock_windows"] == 1, f.values["shock_windows"])
    # 同樣的 IMU2 差異但沒有共模衝擊(只有 IMU2 自己壞掉)→ 是故障
    imu2 = make_imu(140.0)
    m = (imu2[2]["t"] >= 100.0) & (imu2[2]["t"] < 100.4)
    imu2[2]["AccX"][m] += 18.0
    g = imu_run(imu2)
    check(g.status == HIT and 99.0 <= g.t_onset <= 101.0, f"單顆 IMU 突波不是撞擊:{g.status}")


def test_imu_synthetic_fault_before_impact_still_hit():
    """先壞、後撞:事件前的不一致仍要回報(t_onset 在撞擊之前),不被後面的撞擊洗掉。"""
    f = imu_run(make_imu(140.0, impact=(110.0, 80.0), bias_acc={1: (50.0, [1.0, 0.0, 0.0])}))
    check(f.status == HIT and 49.5 <= f.t_onset <= 50.5, f"{f.status} {f.t_onset}")


def test_imu_synthetic_anchor_boundary():
    """錨點(t_loss)之後才出現的差異是後果;之前的是故障。沒有任何本地衝擊也成立。"""
    imu = make_imu(140.0, bias_acc={1: (95.0, [1.0, 0.0, 0.0])})
    f = imu_run(imu, t_loss=90.0)
    check(f.status == CLEAR and f.values["post_impact"] is not None, f"{f.status} {f.context}")
    check(f.values["window_basis"] == "anchor" and f.values["window"][1] == 90.0, f.values["window"])
    g = imu_run(imu, t_loss=100.0)
    check(g.status == HIT, f"{g.status}")
    # a.impacts(end_state 模組填的衝擊)也併入衝擊窗
    imu3 = make_imu(140.0)
    m = (imu3[1]["t"] >= 80.0) & (imu3[1]["t"] < 80.5)
    imu3[1]["AccZ"][m] += 5.0
    h = imu_run(imu3, impacts=[{"t_start": 79.9, "t_end": 80.2, "t_peak": 80.0, "peak": 25.0, "dV": 0.5, "cls": "ground_contact"}])
    check(h.status == CLEAR, f"a.impacts 的衝擊窗內的差異是後果:{h.status}")
    k = imu_run(imu3)
    check(k.status == HIT, f"沒有衝擊資訊時同樣的差異是故障:{k.status}")


def test_imu_synthetic_health_flags():
    f = imu_run(make_imu(140.0, gh_zero={1: 60.0}))
    check(f.status == HIT and any(h["kind"] == "gyro_unhealthy" for h in f.values["health_flags"]), f"{f.status} {f.values['health_flags']}")
    check(59.5 <= f.t_onset <= 60.5, f.t_onset)
    g = imu_run(make_imu(140.0, eg_inc={2: [50.0, 80.0]}))
    check(g.status == HIT and g.values["health_flags"][0]["kind"] == "gyro_error_count" and g.values["health_flags"][0]["n"] == 2,
          g.values["health_flags"])
    h = imu_run(make_imu(140.0, ghz_drop={0: 70.0}))
    check(h.status == SOFT and h.values["health_flags"][0]["kind"].endswith("rate_drop"), f"{h.status} {h.values['health_flags']}")
    k = imu_run(make_imu(140.0, temp_jump={1: 70.0}))
    check(k.status == SOFT and k.values["health_flags"][0]["kind"] == "temp_jump", f"{k.status} {k.values['health_flags']}")
    # 事件之後才失效的健康旗標:後果
    m = imu_run(make_imu(140.0, gh_zero={1: 100.0}), t_loss=90.0)
    check(m.status == CLEAR and m.values["post_event_health"], f"{m.status} {m.values['post_event_health']}")


def test_imu_synthetic_gaps_are_data_quality_only():
    f = imu_run(make_imu(140.0, gaps=[(50.0, 50.4), (90.0, 90.15)]))
    check(f.status == CLEAR, f"記錄空洞不是感測器故障:{f.status} {f.text_zh}")
    check(f.values["log_gaps"]["n"] == 2 and 0.39 < f.values["log_gaps"]["longest_s"] < 0.42, f.values["log_gaps"])
    check("記錄空洞" in f.text_zh, f.text_zh)


def test_imu_synthetic_unpaired_samples_not_fault():
    """實例 j 自己的記錄空洞 / 提早結束:np.interp 會補直線或夾住最後一筆,不能當成 IMU 不一致。"""
    imu = make_imu(140.0)
    keep = imu[2]["t"] < 80.0                                           # IMU2 在 80 s 停止記錄(IMU0/1 照常)
    imu[2] = {k: v[keep] for k, v in imu[2].items()}
    f = imu_run(imu)
    check(f.status == CLEAR, f"{f.status} {f.values.get('free_runs')}")
    check(f.values["pairs"]["0-2"]["unpaired_n"] > 10000 and f.values["pairs"]["0-1"]["unpaired_n"] == 0, f.values["pairs"])
    check(49.0 <= f.values["pairs"]["0-2"]["unpaired_s"] <= 51.0 and "IMU2 有 50" in f.text_zh, f"{f.values['pairs']} {f.text_zh}")
    imu = make_imu(140.0)
    gap = (imu[2]["t"] >= 60.0) & (imu[2]["t"] < 62.5)                  # 只有 IMU2 有 2.5 s 空洞(機體在擺動)
    imu[2] = {k: v[~gap] for k, v in imu[2].items()}
    g = imu_run(imu)
    check(g.status == CLEAR, f"{g.status} {g.values.get('free_runs')}")
    # 遮罩不能把真正的故障遮掉:空洞以外的偏差照樣抓到
    imu = make_imu(140.0, bias_acc={2: (90.0, [0.0, 1.0, 0.0])})
    gap = (imu[2]["t"] >= 60.0) & (imu[2]["t"] < 62.5)
    imu[2] = {k: v[~gap] for k, v in imu[2].items()}
    h = imu_run(imu)
    check(h.status == HIT and 89.5 <= h.t_onset <= 90.5, f"{h.status} {h.t_onset}")


def _imu_cut(imu, a0, a1, insts=None):
    out = {}
    for i, tb in imu.items():
        if insts is None or i in insts:
            keep = ~((tb["t"] >= a0) & (tb["t"] < a1))
            out[i] = {k: v[keep] for k, v in tb.items()}
        else:
            out[i] = tb
    return out


def test_imu_synthetic_coverage():
    """審查缺陷 2(IMU):事件前最後一段沒有可比較的資料 → 不能 CLEAR;基準 IMU 自己的空洞 / 提早結束也要算。"""
    # 三顆都在 60 s 停止記錄、撞擊在 125 s(舊版:CLEAR『3 顆 IMU 事件前彼此一致』)
    f = imu_run(_imu_cut(make_imu(140.0), 60.0, 999.0), end_type="crash_check", t_first_hit=125.0)
    check(f.status == NA and f.context == "data_gap", f"{f.status} {f.context} {f.text_zh}")
    check("沒有記錄" in f.text_zh and "資料不足" in f.text_zh and "沒有 IMU 感測器故障" not in f.text_zh, f.text_zh)
    check(f.values["coverage"]["instances"]["IMU0"]["tail_gap_s"] > 60, f.values["coverage"])
    # 基準 IMU0 在 60 s 停止、IMU1 從 100 s 起偏 1.5 m/s²(舊版:CLEAR、unpaired_n 0 —— IMU0 是時間軸,空洞看不到)
    imu = _imu_cut(make_imu(140.0, bias_acc={1: (100.0, [1.5, 0.0, 0.0])}), 60.0, 999.0, insts={0})
    g = imu_run(imu)
    check(g.status == HIT and 99.5 <= g.t_onset <= 100.5, f"{g.status} {g.t_onset} {g.text_zh}")
    check(g.values["coverage"]["ref"] == 1 and g.values["free_runs"][0]["pair"] == "1-2", g.values["coverage"])
    check("IMU0 有 70.0 秒沒有記錄" in g.text_zh, g.text_zh)
    # 同樣但沒有故障:IMU1 / IMU2 仍可比到最後 → CLEAR,但寫明 IMU0 那段無法比較
    h = imu_run(_imu_cut(make_imu(140.0), 60.0, 999.0, insts={0}))
    check(h.status == CLEAR and "IMU0 有 70.0 秒沒有記錄" in h.text_zh, f"{h.status} {h.text_zh}")
    check(h.values["pairs"]["1-0"]["coverage"]["last60_frac"] == 0.0 and h.values["pairs"]["1-2"]["coverage"]["frac"] == 1.0,
          h.values["pairs"])
    # 中間 60 s 空洞(三顆同時):窗尾完整 → CLEAR,但不可把 60 s 的空洞說成「紀錄緩衝問題」
    k = imu_run(make_imu(200.0, gaps=[(60.0, 120.0)]), armed=[ArmedInterval(0, 20.0, 190.0, 0)])
    check(k.status == CLEAR, f"{k.status} {k.text_zh}")
    check("IMU0/IMU1/IMU2 有 60.0 秒沒有記錄" in k.text_zh and "緩衝" not in k.text_zh, k.text_zh)
    check(k.values["log_gaps"]["n"] == 0 and k.values["log_gaps"]["n_long"] == 1 and k.values["log_gaps"]["longest_long_s"] > 59,
          k.values["log_gaps"])
    # 窗尾缺 8 s(> 5 s)→ NA;缺 3 s → CLEAR
    m = imu_run(_imu_cut(make_imu(140.0), 122.0, 999.0))
    check(m.status == NA and m.context == "data_gap", f"{m.status} {m.context}")
    n = imu_run(_imu_cut(make_imu(140.0), 127.0, 999.0))
    check(n.status == CLEAR and "IMU0/IMU1/IMU2 有 3.0 秒沒有記錄" in n.text_zh, f"{n.status} {n.text_zh}")


def test_imu_synthetic_nonfinite_values():
    """審查缺陷 3:NaN 不能讓一致性檢查默默失效(舊版:一個 NaN 毒化 cumsum → 之後的故障全部看不到,仍 CLEAR)。"""
    def with_nan(imu, inst, col, t_nan):
        k = int(np.searchsorted(imu[inst]["t"], t_nan))
        imu[inst][col][k] = np.nan
        return imu
    # IMU1 AccX 在 40 s 有一個 NaN、80 s 起真的偏 1.0 m/s²
    f = imu_run(with_nan(make_imu(140.0, bias_acc={1: (80.0, [1.0, 0.0, 0.0])}), 1, "AccX", 40.0))
    check(f.status == HIT and 79.5 <= f.t_onset <= 80.5, f"{f.status} {f.t_onset} dA={f.values['dA_max']}")
    # NaN 落在固定偏差的參考區段(解鎖前)
    g = imu_run(with_nan(make_imu(140.0, bias_acc={1: (80.0, [1.0, 0.0, 0.0])}), 1, "AccY", 5.0))
    check(g.status == HIT and 79.5 <= g.t_onset <= 80.5, f"{g.status} {g.t_onset}")
    check(None not in g.values["pairs"]["0-1"]["offset_acc"], g.values["pairs"]["0-1"]["offset_acc"])
    # 基準 IMU0 的 GyrZ 有 NaN + 加速度故障:HIT,文字不可出現 nan
    h = imu_run(with_nan(make_imu(140.0, bias_acc={1: (80.0, [1.0, 0.0, 0.0])}), 0, "GyrZ", 40.0))
    check(h.status == HIT and "nan" not in h.text_zh.lower(), h.text_zh)
    check(all(r["dG"] is not None for r in h.values["free_runs"]), h.values["free_runs"])
    # 沒有故障時,NaN 只是少一個樣本:CLEAR、數字照常
    k = imu_run(with_nan(make_imu(140.0), 2, "AccZ", 70.0))
    check(k.status == CLEAR and k.values["dA_max"] is not None and k.values["dA_max"] < 0.1, k.values["dA_max"])
    # a.impacts 的 t_start 是 NaN:丟掉,不能讓事件時間變 NaN、所有比較變空(舊版 CLEAR 且 dA_max None)
    m = imu_run(make_imu(140.0), impacts=[{"t_start": float("nan"), "t_end": 1.0, "peak": 60.0}])
    check(m.status == CLEAR and m.values["dA_max"] is not None and m.values["window_basis"] == "disarm", m.values)
    check(m.values["shock_windows"] == 0, m.values["shock_windows"])
    # 完全沒有可比較的樣本(另外兩顆加速度全是 NaN)→ NA,不是 CLEAR「差異在門檻內」
    imu = make_imu(140.0)
    imu[1]["AccX"][:] = np.nan
    imu[2]["AccX"][:] = np.nan
    n = imu_run(imu)
    check(n.status == NA and n.context == "no_comparison", f"{n.status} {n.context} {n.text_zh}")
    # 振動偵測器的 VIBE 浮點欄位也一樣(已由 _vibe_stats 遮掉)
    vb = make_vibe(320.0)
    vb["VibeY"][500:510] = np.nan
    v = vib_run(vb)
    check(v.status == CLEAR and v.values["coverage"]["n_holes"] == 1, v.values["coverage"])


def test_anchor_rules_shared_by_both_detectors():
    """審查缺陷 4:兩個偵測器用同一條錨點規則(只採用解鎖區間內的有限值)。
    舊版:t_loss=5.0(在解鎖之前)→ IMU NA insufficient_window,同樣錨點下 VIBE 卻是 CLEAR。"""
    armed = [ArmedInterval(0, 20.0, 130.0, 0)]
    ctx = FakeCtx(armed, vibe={0: make_vibe(140.0)}, imu=make_imu(140.0))
    for kw in ({"t_loss": 5.0}, {"t_loss": 500.0}, {"t_loss": float("nan")}, {"t_first_hit": float("inf")},
               {"t_impact_peak": 19.0}, {"t_detect": float("nan")}):
        r = run_checked(ctx, anchors_for(armed, end_type="unknown", **kw))
        v, im = r["vibration_flight_level"], r["imu_consistency_health"]
        check(im.status == CLEAR and im.values["window_basis"] == "disarm", f"{kw}:IMU {im.status} {im.values.get('window_basis')}")
        check(v.status == CLEAR and v.values["window_basis"] == "disarm", f"{kw}:VIBE {v.status} {v.values.get('window_basis')}")
    # 區間內的錨點兩邊都採用
    r = run_checked(ctx, anchors_for(armed, end_type="crash_check", t_loss=100.0))
    check(r["imu_consistency_health"].values["window"][1] == 100.0 and r["vibration_flight_level"].values["window"][1] == 99.0,
          (r["imu_consistency_health"].values["window"], r["vibration_flight_level"].values["window"]))


def test_imu_synthetic_text_matches_data():
    """文字只能宣稱實際檢查過的東西:IMU 數量、有沒有健康欄位。"""
    f = imu_run(make_imu(140.0, n=2))
    check(f.status == CLEAR and "2 顆 IMU" in f.text_zh and "三顆" not in f.text_zh, f.text_zh)
    imu = make_imu(140.0)
    for tb in imu.values():
        for c in ("GH", "AH", "EG", "EA"):
            tb.pop(c)
    g = imu_run(imu)
    check(g.status == CLEAR and "GH / AH 全程為 1" not in g.text_zh and "沒有 GH / AH / EG / EA" in g.text_zh, g.text_zh)
    check(set(g.values["health_checked"]) == {"GHz", "AHz", "T"}, g.values["health_checked"])
    # 事件後才失效的健康旗標寫進文字(屬後果)
    h = imu_run(make_imu(140.0, gh_zero={1: 100.0}), t_loss=90.0, end_type="crash_check")
    check(h.status == CLEAR and "陀螺儀健康旗標 GH=0" in h.text_zh and "後果" in h.text_zh, h.text_zh)


def test_imu_synthetic_shock_windows_merged():
    """本地偵測與 a.impacts 的同一次衝擊只算一次;不同衝擊各算一次。"""
    imu = make_imu(140.0, impact=(100.0, 80.0))
    f = imu_run(imu, impacts=[{"t_start": 100.0, "t_end": 100.05, "peak": 78.0}, {"t_start": 70.0, "t_end": 70.1, "peak": 25.0}])
    check(f.status == CLEAR and f.values["shock_windows"] == 2, f"{f.status} {f.values.get('shock_windows')}")
    ev = det_vib._shock_events(np.arange(0.0, 1.0, 0.01), {0: np.zeros((3, 100))},
                               [{"t_start": 0.1, "t_end": 0.2, "peak": 30.0}, {"t_start": 0.5, "t_end": 0.6, "peak": 50.0},
                                {"t_start": 5.0, "t_end": 5.1, "peak": 22.0}, {"t_start": 9.0, "peak": 10.0}, {"bad": 1}])
    check([(e["t_start"], e["t_end"], e["peak"]) for e in ev] == [(0.1, 0.6, 50.0), (5.0, 5.1, 22.0)], ev)


def test_imu_synthetic_single_imu_is_na():
    f = imu_run(make_imu(140.0, n=1))
    check(f.status == NA and f.context == "single_imu", f"{f.status} {f.context}")
    # 但單顆 IMU 回報自己不健康 → 仍是證據
    g = imu_run(make_imu(140.0, n=1, gh_zero={0: 60.0}))
    check(g.status == HIT, g.status)
    # 兩顆 IMU:可比較;兩顆都超標才算衝擊(單顆毛刺不當衝擊)
    h = imu_run(make_imu(140.0, n=2, bias_acc={1: (60.0, [1.0, 0.0, 0.0])}))
    check(h.status == HIT, h.status)


def test_imu_synthetic_na_cases():
    f = det_vib.imu_consistency_health(FakeCtx([], imu=make_imu(30.0)), Anchors())[0]
    check(f.status == NA and f.context == "not_armed", f"{f.status} {f.context}")
    armed = one_flight()
    f = det_vib.imu_consistency_health(FakeCtx(armed), anchors_for(armed))[0]
    check(f.status == NA and f.context == "no_data", f"{f.status} {f.context}")
    armed = [ArmedInterval(0, 10.0, 15.0, 0)]
    f = det_vib.imu_consistency_health(FakeCtx(armed, imu=make_imu(30.0)), anchors_for(armed))[0]
    check(f.status == NA and f.context == "insufficient_window", f"{f.status} {f.context}")
    # 筆數太少(< 1000)的實例不採用
    f = det_vib.imu_consistency_health(FakeCtx(one_flight(), imu=make_imu(2.0)), anchors_for(one_flight()))[0]
    check(f.status == NA, f.status)


def test_imu_synthetic_time_shift_invariance():
    f0 = imu_run(make_imu(140.0, bias_acc={1: (60.0, [1.0, 0.0, 0.0])}))
    f1 = imu_run(make_imu(140.0, bias_acc={1: (60.0, [1.0, 0.0, 0.0])}, t_off=7000.0), armed=[ArmedInterval(0, 7020.0, 7130.0, 0)])
    check(f0.status == f1.status == HIT and abs((f1.t_onset - 7000.0) - f0.t_onset) < 1e-6, (f0.t_onset, f1.t_onset))


# ===================================================================== 4. 純函式與介面規則

def test_runs_idx_and_gaps():
    t = np.array([0, .1, .2, .3, 1.0, 1.1, 1.2, 2.0, 2.1])
    m = np.array([0, 1, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    check(det_vib._runs_idx(m, t) == [(1, 5), (7, 8)], det_vib._runs_idx(m, t))
    check(det_vib._runs_idx(m, t, max_gap=0.25) == [(1, 3), (4, 5), (7, 8)], det_vib._runs_idx(m, t, 0.25))
    check(det_vib._runs_idx(np.zeros(5, bool), np.arange(5.0)) == [], "全 False 應無區段")
    check(det_vib._runs_idx(np.ones(3, bool), np.arange(3.0)) == [(0, 2)], "全 True")
    check(det_vib._runs_idx(np.array([True]), np.array([0.0]), 0.1) == [(0, 0)], "單點")


def test_tmean_is_time_based_across_gap():
    t = np.concatenate([np.arange(0, 1, 0.01), np.arange(3, 4, 0.01)])       # 1~3 s 之間有 2 s 空洞
    v = np.where(t < 2, 0.0, 1.0)
    out = det_vib._tmean(t, v, 0.25)
    # 空洞後第一個樣本(t=3.00)的窗只含 3.00±0.125 的資料,不會把空洞前的 0 平均進來
    i = int(np.searchsorted(t, 3.0))
    check(abs(out[i] - 1.0) < 1e-9, f"跨空洞被拉長的窗:{out[i]}")
    check(abs(out[0]) < 1e-12 and out.shape == t.shape, "端點窗縮小")
    # 一維與二維輸入
    check(det_vib._tmean(t, np.vstack([v, 2 * v]), 0.25).shape == (2, t.size), "二維輸入形狀")


def test_vibe_stats_clip_uses_increments():
    t = np.arange(0, 100, 0.1)
    tb = {"t": t, "VibeX": np.full(t.size, 5.0), "VibeY": np.full(t.size, 2.0), "VibeZ": np.full(t.size, 1.0),
          "Clip": np.full(t.size, 2.0)}
    s = det_vib._vibe_stats(tb, 10.0, 90.0)
    check(s["clip_delta"] == 0 and s["p95"] == 5.0 and s["axes"]["x"] == [5.0, 5.0, 5.0], s)
    tb["Clip"] = np.where(t >= 50, 14.0, 2.0)
    s = det_vib._vibe_stats(tb, 10.0, 90.0)
    check(s["clip_delta"] == 12 and abs(s["_t_clip"] - 50.0) < 0.11, s["clip_delta"])
    check(det_vib._vibe_stats(tb, 200.0, 300.0) is None, "窗內無樣本 → None")
    tb["VibeX"][:] = np.nan
    check(det_vib._vibe_stats(tb, 10.0, 90.0) is None, "NaN 樣本不計入")


def _st(p95, p95_last=None, frac30=0.0, longest=0.0, clip=0, t15=None, t30=None, tclip=None):
    return {"p95": p95, "p95_last": p95 if p95_last is None else p95_last, "frac30": frac30, "longest30_s": longest,
            "clip_delta": clip, "_t15": t15, "_t30": t30, "_t_clip": tclip}


def test_classify_vibe_thresholds():
    c = det_vib._classify_vibe({"IMU0": _st(9.0)}, {}, None)
    check(c["verdict"] == "ok" and not c["clipping"], c)
    c = det_vib._classify_vibe({"IMU0": _st(14.99)}, {}, None)
    check(c["verdict"] == "ok", "p95 < 15 為 ok")
    c = det_vib._classify_vibe({"IMU0": _st(15.0, t15=3.0)}, {}, None)
    check(c["verdict"] == "marginal" and c["t_onset"] == 3.0, c)
    c = det_vib._classify_vibe({"IMU0": _st(30.0, t30=4.0, t15=2.0)}, {}, None)
    check(c["verdict"] == "bad" and c["t_onset"] == 4.0, c)
    c = det_vib._classify_vibe({"IMU0": _st(8.0, frac30=0.051, t30=9.0)}, {}, None)
    check(c["verdict"] == "bad", "m>30 的比例 > 5% 為 bad")
    c = det_vib._classify_vibe({"IMU0": _st(8.0, frac30=0.05, t30=9.0)}, {}, None)
    check(c["verdict"] == "ok", "剛好 5% 不算(嚴格大於)")
    c = det_vib._classify_vibe({"IMU0": _st(8.0, longest=5.0, t30=9.0)}, {}, None)
    check(c["verdict"] == "bad", "連續 5 s 為 bad")
    c = det_vib._classify_vibe({"IMU0": _st(8.0)}, {}, 12.0)
    check(c["verdict"] == "bad" and c["t_onset"] == 12.0, "ERR 29")
    # 比值:要 > 2 且最近 60 s 的 p95 ≥ 10
    base = {"IMU0": _st(4.0)}
    check(det_vib._classify_vibe({"IMU0": _st(9.0, 8.0)}, base, None)["verdict"] == "ok", "絕對水準 < 10:不算")
    check(det_vib._classify_vibe({"IMU0": _st(9.0, 10.5)}, base, None)["verdict"] == "marginal", "10.5/4 > 2 且 ≥ 10")
    check(det_vib._classify_vibe({"IMU0": _st(9.0, 10.5)}, {"IMU0": _st(5.25)}, None)["verdict"] == "ok", "剛好 2.0 倍不算")
    # 最近 60 s 取較大者
    c = det_vib._classify_vibe({"IMU0": _st(9.0, 31.0, t30=1.0)}, {}, None)
    check(c["verdict"] == "bad" and c["p95"] == 31.0, c)


def test_every_scenario_returns_both_ids():
    """各情境(真實 + 合成)兩個 id 都至少一筆;沒有一筆 context=='error'。"""
    scen = []
    ctx = real("00000124")
    scen.append((ctx, ck.truth_anchors_124(ctx)))
    scen.append((ctx, ck.flight1_anchors_124(ctx)))
    scen.append((ctx, basic_anchors(ctx)))
    for name in ck.BASELINES:
        c = real(name)
        scen.append((c, ck.not_armed_anchors(c)))
    armed = one_flight()
    scen.append((FakeCtx(armed, vibe={0: make_vibe(320.0)}, imu=make_imu(140.0)), anchors_for(armed)))
    scen.append((FakeCtx(armed), anchors_for(armed)))                       # 完全沒有資料
    scen.append((FakeCtx([]), Anchors()))
    for c, a in scen:
        fs = det_vib.run(c, a)
        ids = [f.id for f in fs]
        check(sorted(set(ids)) == sorted(IDS), f"id 清單不符:{ids}")
        check(not any(f.context == "error" for f in fs), [(f.id, f.values) for f in fs if f.context == "error"])
        for f in fs:
            json.dumps(f.values, allow_nan=False)


def test_guard_converts_errors_to_na():
    class Boom(FakeCtx):
        def table(self, *a, **k):
            raise RuntimeError("boom")
    armed = one_flight()
    fs = det_vib.run(Boom(armed, vibe={0: make_vibe(100.0)}, imu=make_imu(30.0)), anchors_for(armed))
    check(len(fs) == 2 and all(f.status == NA and f.context == "error" for f in fs), [(f.id, f.status) for f in fs])


def test_no_hardcoded_124_numbers():
    """模組內不可出現 124 的時間常數 / log 名稱。"""
    src = open(det_vib.__file__, encoding="utf-8").read()
    for bad in ("1255.2", "1258.4", "1259.5", "1262.1", "8.1593", "873.1", "335.2", "1257.", "1254.9"):
        check(bad not in src, f"det_vib.py 寫死了 124 的數字 {bad}")


def test_performance_on_124():
    ctx = real("00000124")
    a = ck.truth_anchors_124(ctx)
    det_vib.run(ctx, a)                       # 暖機:ctx 的 table 轉換 / 快取
    worst = 0.0
    for fn in (det_vib.vibration_flight_level, det_vib.imu_consistency_health):
        best = min(_timed(fn, ctx, a) for _ in range(3))
        worst = max(worst, best)
        print(f"  {fn.__name__}: {best:.3f} s")
    check(worst < 1.5, f"偵測器應 < 1.5 s(400 Hz × 1290 s):{worst:.2f}")


def _timed(fn, ctx, a):
    t0 = time.perf_counter()
    fn(ctx, a)
    return time.perf_counter() - t0


def test_deterministic():
    ctx = real("00000124")
    a = ck.truth_anchors_124(ctx)
    r1 = [(f.id, f.status, f.t_onset, json.dumps(f.values, sort_keys=True), f.text_zh) for f in det_vib.run(ctx, a)]
    r2 = [(f.id, f.status, f.t_onset, json.dumps(f.values, sort_keys=True), f.text_zh) for f in det_vib.run(ctx, a)]
    check(r1 == r2, "同樣輸入應有同樣輸出")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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
