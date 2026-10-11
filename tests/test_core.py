"""核心驗證(不需 PyQt6):python -m tests.test_core 或 pytest tests/"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import align, clock, dflog, ekfsrc, fakebin, frames, logio, metrics, mission, pipeline, sim, sync, timeline  # noqa: E402
from core.types import EV_ARM, EV_DISARM, EV_MODE, EV_RESET, LogData, LogEvent  # noqa: E402


def test_frames_roundtrip():
    origin = (23.702, 120.431, 38.0)
    enu = np.array([[0.0, 0.0, 0.0], [100.0, -250.0, 12.5], [-3000.0, 4000.0, -80.0]])
    lat, lng, alt = frames.enu_to_geodetic(enu, origin)
    back = frames.geodetic_to_enu(lat, lng, alt, origin)
    err = np.max(np.abs(back - enu))
    assert err < 1e-6, f"ENU↔WGS84 往返誤差 {err:.3e} m 過大"
    print(f"  ENU↔WGS84 往返誤差 {err:.2e} m")


def test_frames_scale():
    """1e-5° 緯度 ≈ 1.11 m;經度需乘 cos(lat)。"""
    origin = (23.702, 120.431, 0.0)
    enu = frames.geodetic_to_enu([23.70201], [120.43101], [0.0], origin)[0]
    assert abs(enu[1] - 1.1057) < 0.01, f"北向尺度異常: {enu[1]:.4f}"
    expected_e = 1.1132 * np.cos(np.radians(23.702))
    assert abs(enu[0] - expected_e) < 0.01, f"東向尺度異常: {enu[0]:.4f}"
    print(f"  1e-5° → E {enu[0]:.4f} m / N {enu[1]:.4f} m")


def test_ned_enu():
    ned = np.array([[1.0, 2.0, 3.0]])
    enu = frames.ned_to_enu(ned)
    assert np.allclose(enu, [[2.0, 1.0, -3.0]])
    assert np.allclose(frames.enu_to_ned(enu), ned)
    print("  NED↔ENU 互換正確")


def test_align_recovers_transform():
    """對已知 yaw+平移+尺度變換的軌跡,對齊後應幾乎零誤差且參數可還原。"""
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 20, (500, 3))
    yaw = np.radians(7.5)
    c, s = np.cos(yaw), np.sin(yaw)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    scale, t = 1.02, np.array([13.0, -4.0, 2.5])
    est = (ref @ r.T) * scale + t          # est = scale·R·ref + t

    out, info = align.align(est, ref, "sim3")
    assert np.max(np.abs(out - ref)) < 1e-8
    assert abs(info["scale"] - 1 / scale) < 1e-6, info["scale"]
    assert abs(info["yaw_deg"] + 7.5) < 1e-6, info["yaw_deg"]
    print(f"  Sim(3) 還原:殘差 {np.max(np.abs(out - ref)):.2e} m, "
          f"yaw {info['yaw_deg']:+.4f}°, scale {info['scale']:.6f}")

    est_rigid = (ref @ r.T) + t
    out, info = align.align(est_rigid, ref, "se3")
    assert np.max(np.abs(out - ref)) < 1e-8
    out, info = align.align(est_rigid, ref, "yaw")
    assert np.max(np.abs(out - ref)) < 1e-8, "純 yaw 變換應被 4DOF 完全吸收"
    print("  SE(3) / 4DOF 還原:殘差 < 1e-8 m")

    out, _ = align.align(est, ref, "none")
    assert np.allclose(out, est), "不對齊應原封不動"


def test_align_translation_only():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 10, (200, 3))
    est = ref + np.array([5.0, -2.0, 1.0])
    out, info = align.align(est, ref, "translation")
    assert np.max(np.abs(out - ref)) < 1e-9
    assert np.allclose(info["translation"], [-5.0, 2.0, -1.0])
    print("  僅平移對齊正確")


def test_metrics_known():
    """已知誤差:E 恆 +3 m → 3D RMSE = 3,E 平均 = +3(帶號),U 全 0。"""
    err = np.zeros((100, 3))
    err[:, 0] = 3.0
    s = metrics.ate_stats(err)
    assert abs(s["3d"]["rmse"] - 3.0) < 1e-12
    assert abs(s["2d"]["rmse"] - 3.0) < 1e-12
    assert abs(s["axes"]["E"]["mean"] - 3.0) < 1e-12
    assert abs(s["axes"]["U"]["max"]) < 1e-12
    assert s["n"] == 100

    err2 = np.zeros((4, 3))
    err2[:, 0] = [3.0, -4.0, 0.0, 0.0]
    err2[:, 1] = [4.0, 3.0, 0.0, 0.0]
    s2 = metrics.ate_stats(err2)
    assert abs(s2["3d"]["max"] - 5.0) < 1e-12, "3D 最大應為 5 (3-4-5)"
    assert abs(s2["axes"]["E"]["mean"] + 0.25) < 1e-12, "帶號平均 = (3-4)/4"
    assert abs(s2["axes"]["E"]["max"] - 4.0) < 1e-12, "軸向最大取絕對值"
    print("  ATE 統計:3-4-5 三角、帶號平均、絕對最大值皆正確")

    line = np.column_stack([np.arange(11.0), np.zeros(11), np.zeros(11)])
    assert abs(metrics.path_length(line) - 10.0) < 1e-12
    print("  路徑長度正確")


def test_sync_interp_and_gap():
    t_ref = np.arange(0, 10, 0.1)
    p_ref = np.column_stack([t_ref, 2 * t_ref, np.zeros_like(t_ref)])
    t_est = np.arange(0, 10, 0.037) + 0.4          # 不同取樣率 + 起訖不同
    p_est = np.column_stack([t_est, 2 * t_est, np.zeros_like(t_est)])
    t, rp, ep = sync.sync_pair(t_ref, p_ref, t_est, p_est)
    assert np.max(np.abs(rp - ep)) < 1e-9, "同一條線內插後應完全吻合"
    assert t[0] >= 0.4 - 1e-9 and t[-1] <= t_est[-1] + 1e-9, "應裁到重疊區間"
    print(f"  時間同步:{t.size} 對樣本,內插殘差 < 1e-9 m")

    # 資料空洞:est 缺 4~6 s,max_gap 應剔除該段
    keep = (t_est < 4.0) | (t_est > 6.0)
    t2, _, _ = sync.sync_pair(t_ref, p_ref, t_est[keep], p_est[keep], max_gap_s=0.2)
    assert not np.any((t2 > 4.3) & (t2 < 5.7)), "空洞區間應被剔除"
    print(f"  空洞保護:剔除後剩 {t2.size} 對(原 {t.size})")

    try:
        sync.sync_pair(t_ref, p_ref, t_est + 100, p_est)
    except ValueError:
        print("  無重疊時正確拋出例外")
    else:
        raise AssertionError("無重疊應拋出 ValueError")


def test_time_offset_search():
    logdata, _ = sim.make_sim_with_truth(seed=3)
    rt, rp, et, ep = pipeline.get_enu_pair(logdata, "GPS", "VISP")
    true_off = 0.35
    off, corr = sync.find_time_offset(rt, rp, et - true_off, ep, search_s=1.5)
    assert abs(off - true_off) < 0.08, f"估到 {off:.3f} s,應接近 {true_off} s"
    print(f"  時間偏移搜尋:注入 {true_off:.2f} s → 估得 {off:.3f} s (corr {corr:.3f})")


def _ref_find_time_offset(ref_t, ref_p, est_t, est_p, search_s=2.0, step_s=0.02):
    """PERF-3 之前的逐一偏移實作(對照組,原樣保留)。"""
    rt, rv = sync._speed_profile(ref_t, ref_p)
    et, ev = sync._speed_profile(est_t, est_p)
    if rt.size < 20 or et.size < 20:
        raise ValueError("樣本太少")
    lo = max(rt[0], et[0] - search_s)
    hi = min(rt[-1], et[-1] + search_s)
    if hi - lo < 5.0:
        raise ValueError("重疊時間太短")
    grid = np.arange(lo, hi, 0.05)
    rg = np.interp(grid, rt, rv)
    best_off, best_corr = 0.0, -np.inf
    for off in np.arange(-search_s, search_s + 1e-9, step_s):
        eg = np.interp(grid, et + off, ev, left=np.nan, right=np.nan)
        m = np.isfinite(eg)
        if int(m.sum()) < 50:
            continue
        a = rg[m] - rg[m].mean()
        b = eg[m] - eg[m].mean()
        den = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
        if den <= 1e-12:
            continue
        corr = float((a * b).sum()) / den
        if corr > best_corr:
            best_off, best_corr = float(off), corr
    if not np.isfinite(best_corr):
        raise ValueError("速度剖面無變化")
    return best_off, best_corr


def _offset_parts(rt, rp, et, ep, search_s, step_s):
    rt, rv = sync._speed_profile(rt, rp)
    et, ev = sync._speed_profile(et, ep)
    lo, hi = max(rt[0], et[0] - search_s), min(rt[-1], et[-1] + search_s)
    grid = np.arange(lo, hi, sync._GRID_DT)
    return grid, np.interp(grid, rt, rv), et, ev, np.arange(-search_s, search_s + 1e-9, step_s)


def test_time_offset_fast_matches_reference():
    """PERF-3:共用內插的快速搜尋與逐一偏移的原做法結果相同((offset, corr) 差 ≤ 1e-12),
    逐偏移的相關係數與「跳過」位置也相同;含 VIO 重置尖峰、部分重疊、大時間值、各種步長。"""
    cases = []
    for seed in (3, 7):
        ld, _ = sim.make_sim_with_truth(seed=seed)
        rt, rp, et, ep = pipeline.get_enu_pair(ld, "GPS", "VISP")
        cases += [(f"sim{seed} GPS→VISP", rt, rp, et - 0.35, ep),
                  (f"sim{seed} VISP→GPS", et, ep, rt, rp)]
    mf, _ = sim.make_multi_flight_sim()
    rt, rp, et, ep = pipeline.get_enu_pair(mf, "GPS", "VISP")         # 有 VIO 重置跳點
    cases += [("多航段 GPS→VISP", rt, rp, et, ep), ("多航段 VISP→GPS", et, ep, rt, rp)]
    s0 = mf.segments[0]
    cases.append(("多航段第 1 段", *pipeline.get_enu_pair(mf, "POS", "VISP", t0=s0.t0, t1=s0.t1)))
    h = et.size // 3
    cases.append(("估測只有一部分", rt, rp, et[h:2 * h], ep[h:2 * h]))
    m = et > rt[-1] - 9.0                                              # 重疊約 9 s:部分偏移 < 50 點
    cases.append(("重疊很短", rt, rp, et[m], ep[m]))
    cases.append(("時間值大(+5000 s)", rt + 5000.0, rp, et + 4999.7, ep))
    n_cmp, worst = 0, 0.0
    for name, rt, rp, et, ep in cases:
        for search_s, step_s in ((5.0, 0.02), (1.5, 0.02), (2.0, 0.01), (2.0, 0.05), (2.0, 0.03),
                                 (2.0, 0.1), (1.0, 0.013), (1.0, 0.0123456789)):
            grid, rg, t_e, v_e, offs = _offset_parts(rt, rp, et, ep, search_s, step_s)
            direct = sync._corrs_direct(grid, rg, t_e, v_e, offs)
            shared = sync._corrs_shared(grid, rg, t_e, v_e, offs, step_s)
            if step_s == 0.0123456789:
                assert shared is None, "步長不是整數 μs → 應交給逐一偏移做法"
            else:
                assert np.array_equal(np.isnan(shared), np.isnan(direct)), (name, step_s)
                d = np.nanmax(np.abs(shared - direct)) if np.isfinite(direct).any() else 0.0
                assert d <= 1e-12, f"{name} step={step_s}:逐偏移相關係數差 {d:.2e}"
            try:
                want = _ref_find_time_offset(rt, rp, et, ep, search_s, step_s)
            except ValueError:
                try:
                    sync.find_time_offset(rt, rp, et, ep, search_s, step_s)
                except ValueError:
                    continue
                raise AssertionError(f"{name}:原做法拋例外,新做法沒有")
            got = sync.find_time_offset(rt, rp, et, ep, search_s, step_s)
            assert got[0] == want[0], f"{name} S={search_s} step={step_s}:偏移 {got[0]} ≠ {want[0]}"
            assert abs(got[1] - want[1]) <= 1e-12, f"{name}:corr {got[1]!r} vs {want[1]!r}"
            worst = max(worst, abs(got[1] - want[1]))
            n_cmp += 1
    # 時間非遞增(不該發生,但結果要與原做法相同)→ 走逐一偏移
    grid, rg, t_e, v_e, offs = _offset_parts(rt, rp, et, ep, 2.0, 0.02)
    assert sync._corrs_shared(grid, rg, t_e[::-1].copy(), v_e, offs, 0.02) is None
    print(f"  {n_cmp} 組(含重置尖峰 / 部分重疊 / 短重疊 / +5000 s / 8 種步長):"
          f"偏移全同,corr 最大差 {worst:.1e}")


def _synthetic_pair(minutes: float, t0: float = 300.0, lag: float = 0.3, seed: int = 5):
    """長時間合成軌跡:參考 5 Hz、估測 30 Hz(時鐘慢 lag 秒 → 正確偏移 +lag;帶雜訊與重置跳點)。"""
    rng = np.random.default_rng(seed)
    dur = minutes * 60.0

    def pos(t):
        return np.column_stack([40 * np.sin(t / 23.0) + 15 * np.sin(t / 3.1),
                                30 * np.cos(t / 17.0) + 10 * np.sin(t / 4.3),
                                5 * np.sin(t / 50.0)])
    rt = t0 + np.arange(0.0, dur, 0.2)
    et = t0 + 0.07 + np.arange(0.0, dur, 1 / 30)
    rp = pos(rt) + rng.normal(0, 0.05, (rt.size, 3))
    ep = pos(et) + rng.normal(0, 0.01, (et.size, 3))
    for tr in t0 + dur * np.array([0.3, 0.6]):                          # VIO 重置跳點
        ep[et > tr] += [3.0, -2.0, 0.0]
    return rt, rp, et - lag, ep


def test_time_offset_speed_large():
    """PERF-3:120 分鐘 log 的「自動估」(±5 s / 0.02 s)應在 0.4 s 左右內完成(原做法約 3 s)。"""
    rt, rp, et, ep = _synthetic_pair(30)
    want = _ref_find_time_offset(rt, rp, et, ep, search_s=5.0)
    got = sync.find_time_offset(rt, rp, et, ep, search_s=5.0)
    assert got[0] == want[0] and abs(got[1] - want[1]) <= 1e-12, (got, want)
    rt, rp, et, ep = _synthetic_pair(120)
    sync.find_time_offset(rt[:500], rp[:500], et[:3000], ep[:3000], search_s=5.0)   # 暖機
    t = time.perf_counter()
    off, corr = sync.find_time_offset(rt, rp, et, ep, search_s=5.0)
    dt = time.perf_counter() - t
    assert abs(off - 0.3) < 0.08 and corr > 0.9, (off, corr)
    assert dt < 1.5, f"120 分鐘自動估 {dt:.2f} s 太慢"
    print(f"  30 分鐘與原做法相同;120 分鐘(參考 {rt.size:,} 點、估測 {et.size:,} 點)"
          f"自動估 {dt * 1000:.0f} ms → 偏移 {off:+.2f} s(corr {corr:.3f})")


def test_expr_heading_on_sim_log():
    """F1:真實 LogIndex(有 FMTU 單位、經 .bin 往返)上,跨北的航向差不再出現 ±180 假尖峰。"""
    from core import expr
    logdata = sim.make_sim_logdata()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sim.bin")
        fakebin.write_logdata_to_bin(logdata, p)
        loaded = logio.load_bin(p)
        for idx, tag in ((logdata.index, "記憶體"), (loaded.index, ".bin")):
            assert idx.unit("ATT.Yaw") == "degheading"
            for e in ("wrap_180(VISP.Yaw - ATT.Yaw)", "wrap_180(ATT.Yaw - XKF1.Yaw)",
                      "wrap_180(XKF1.Yaw - ATT.Yaw)"):
                _, v = expr.evaluate(e, idx)
                mx = float(np.nanmax(np.abs(v)))
                assert mx < 20.0, f"{tag} {e}:最大 {mx:.1f}°(跨 0/360 被線性內插)"
            _, v = expr.evaluate("ATT.Roll * 0 + GPS.GCrs", idx)
            assert np.nanmin(v) >= 0.0 and np.nanmax(v) <= 360.0, "應維持 0~360 範圍"
    print("  模擬 log(記憶體 / .bin 往返):VISP−ATT、ATT−XKF1 航向差皆 < 20°,GCrs 維持 0~360")


def test_pipeline_on_sim():
    logdata, truth = sim.make_sim_with_truth(seed=7)
    for name in ("GPS", "VISP", "POS"):
        assert name in logdata.sources
    print(f"  模擬來源:{ {k: f'{v.n}筆@{v.rate_hz:.0f}Hz' for k, v in logdata.sources.items()} }")

    res_none = pipeline.compute(logdata, "GPS", "VISP", method="none")
    res_se3 = pipeline.compute(logdata, "GPS", "VISP", method="se3")
    res_sim3 = pipeline.compute(logdata, "GPS", "VISP", method="sim3")
    r = [res_none.stats["3d"]["rmse"], res_se3.stats["3d"]["rmse"], res_sim3.stats["3d"]["rmse"]]
    print(f"  VISP vs GPS 3D RMSE:不對齊 {r[0]:.3f} → SE(3) {r[1]:.3f} → Sim(3) {r[2]:.3f} m")
    assert r[1] <= r[0] + 1e-9, "SE(3) 對齊不該讓誤差變大"
    assert r[2] <= r[1] + 1e-9, "Sim(3) 自由度更高,誤差不該比 SE(3) 大"

    # Sim(3) 應還原注入的尺度誤差(VISP 被放大 SCALE_ERR,對齊回 GPS 要縮回)
    got = res_sim3.align_info["scale"]
    assert abs(got - 1 / truth["scale"]) < 0.01, f"尺度還原 {got:.4f} vs 期望 {1 / truth['scale']:.4f}"
    print(f"  Sim(3) 尺度還原:{got:.4f}(注入 {truth['scale']},期望 ≈{1 / truth['scale']:.4f})")

    # VIO 在 NED 內被轉 +yaw_mis(等於 ENU 的 −yaw_mis),對齊回 GPS 需轉 +yaw_mis
    yaw = res_se3.align_info["yaw_deg"]
    exp = truth["yaw_mis_deg"]
    assert abs(yaw - exp) < 0.6, f"yaw 還原 {yaw:+.2f}° vs 期望 ≈{exp:+.1f}°"
    print(f"  SE(3) 偏航還原:{yaw:+.3f}°(注入 {exp:+.1f}°,誤差來自漂移)")

    # POS(EKF)應比 VISP(VIO)貼近 GPS
    res_pos = pipeline.compute(logdata, "GPS", "POS", method="se3")
    assert res_pos.stats["3d"]["rmse"] < res_se3.stats["3d"]["rmse"]
    print(f"  POS vs GPS 3D RMSE {res_pos.stats['3d']['rmse']:.3f} m < VISP 的 {r[1]:.3f} m ✓")

    # 時間裁切
    tb = logdata.t_base
    sub = pipeline.compute(logdata, "GPS", "VISP", t0=tb + 30, t1=tb + 60, method="se3")
    assert sub.n < res_se3.n and 29 < sub.duration < 31
    print(f"  時間裁切 30~60 s:{sub.n} 對樣本,時長 {sub.duration:.1f} s")

    try:
        pipeline.compute(logdata, "GPS", "GPS")
    except ValueError:
        print("  參考=估測時正確拋出例外")
    else:
        raise AssertionError("參考=估測應拋出 ValueError")


def test_report_and_csv():
    logdata = sim.make_sim_logdata()
    res = pipeline.compute(logdata, "GPS", "VISP", method="se3")
    txt = pipeline.format_report(res, logdata)
    for token in ("ATE RMSE", "RMSE", "P95", "空間對齊", "ENU 原點", "移動距離:GPS", "VISP"):
        assert token in txt, f"報告缺少 {token}"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ate.csv")
        pipeline.save_csv(res, logdata, p)
        rows = np.loadtxt(p, delimiter=",", skiprows=3, encoding="utf-8-sig")
        assert rows.shape == (res.n, 14), rows.shape
        # err_3d 欄位應等於重算的誤差長度
        assert np.max(np.abs(rows[:, 11] - res.e3)) < 1e-3
        # 移動距離欄:逐點累積、從 0 起算、最後一列 = 總移動距離
        assert rows[0, 12] == 0.0 and rows[0, 13] == 0.0
        assert np.all(np.diff(rows[:, 12]) >= 0) and np.all(np.diff(rows[:, 13]) >= 0)
        assert abs(rows[-1, 12] - res.path_len_ref) < 1e-3
        assert abs(rows[-1, 13] - res.path_len_est) < 1e-3
        with open(p, encoding="utf-8-sig") as f:
            head = [next(f) for _ in range(3)]
        assert "ref_dist_m,est_dist_m" in head[2] and "移動距離 GPS=" in head[0]
    print(f"  報告 {len(txt)} 字元、CSV {res.n} 列 × 14 欄,數值一致")


def test_travel_distance():
    """GPS / VIO 移動距離:逐點累積、不受對齊方式影響(尺度誤差要看得出來)。"""
    logdata, truth = sim.make_sim_with_truth(seed=7)
    res = {m: pipeline.compute(logdata, "GPS", "VISP", method=m) for m in ("none", "se3", "sim3")}
    r = res["se3"]
    assert r.path_len_ref > 100 and r.path_len_est > 100
    assert r.cum_ref.shape == r.cum_est.shape == r.t.shape
    assert r.cum_ref[0] == 0.0 and r.cum_est[0] == 0.0
    assert abs(r.cum_ref[-1] - r.path_len_ref) < 1e-9 and abs(r.cum_est[-1] - r.path_len_est) < 1e-9
    assert abs(r.path_len_ref - metrics.path_length(r.ref_enu)) < 1e-6
    # VIO 距離取對齊前的軌跡:三種對齊方式(含會縮放的 Sim3)結果一致
    for m in ("none", "sim3"):
        assert abs(res[m].path_len_est - r.path_len_est) < 1e-6, m
    # Sim3 對齊後的軌跡被縮放過(長度 = 原始 × 尺度),所以移動距離不能取對齊後的
    s3 = res["sim3"]
    scale = s3.align_info["scale"]
    assert abs(scale - 1) > 1e-3, "模擬資料應有可辨識的尺度誤差"
    assert abs(metrics.path_length(s3.est_enu) - s3.path_len_est * scale) < 1e-6 * s3.path_len_est
    assert 0.9 < r.dist_ratio < 1.1, r.dist_ratio
    print(f"  GPS {r.path_len_ref:.1f} m / VISP {r.path_len_est:.1f} m(比 {r.dist_ratio:.3f};"
          f"Sim3 對齊後軌跡長 {metrics.path_length(s3.est_enu):.1f} m,不採用)")


def test_mission_waypoints():
    """CMD → 預定航點:只留帶位置的導航指令(排除 home / RTL),轉成 ENU,.bin 往返不失真。"""
    logdata = sim.make_sim_logdata()
    wps = mission.load_mission(logdata)
    assert len(wps) == 16 and [w.n for w in wps] == list(range(1, 17)), [w.n for w in wps]
    assert all(w.cmd == 16 and w.has_position for w in wps)
    enu = mission.waypoints_enu(wps, logdata.origin)
    assert enu.shape == (16, 3) and abs(np.median(enu[:, 2]) - wps[0].alt) < 1e-3   # 相對高度 → 以原點為 home
    # 預定轉角應落在模擬的 30 m 正方形上
    assert np.all(np.minimum(np.abs(enu[:, 0]), np.abs(enu[:, 0] - 30)) < 1.5)
    assert np.all(np.minimum(np.abs(enu[:, 1]), np.abs(enu[:, 1] - 30)) < 1.5)
    # 絕對高度 frame:高度為海拔,要扣掉原點高度
    w_abs = mission.Waypoint(1, 16, "WAYPOINT", wps[0].lat, wps[0].lng, logdata.origin[2] + 12.0, 0, 0.0)
    assert abs(mission.waypoints_enu([w_abs], logdata.origin)[0, 2] - 12.0) < 1e-6
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.bin")
        fakebin.write_logdata_to_bin(logdata, p)
        back = mission.load_mission(logio.load_bin(p))
        assert [w.n for w in back] == [w.n for w in wps]
        assert max(abs(a.lat - b.lat) + abs(a.lng - b.lng) for a, b in zip(wps, back)) < 1e-6
    class _NoIdx:
        index = None
    assert mission.load_mission(_NoIdx()) == []
    print(f"  {len(wps)} 個航點,ENU 落在預期轉角;.bin 往返一致;無 CMD → 空清單")


def test_mission_progress():
    """推估目前飛往的航點:只在 AUTO 推進、進入半徑才算到達、離開 AUTO 保留進度、可跳過漏掉的航點。"""
    wp = np.array([[10.0, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]])
    t = np.arange(0, 60.0, 0.5)
    x = np.clip(t - 5.0, 0, 40.0)                       # 5 s 起飛,以 1 m/s 沿 x 軸前進
    enu = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    auto = np.ones(t.size, bool)
    ct, ck = mission.estimate_progress(t, enu, wp, auto, radius=2.0)
    assert list(ck) == [0, 1, 2, 3, 4], list(ck)         # 4 = 全部到達
    assert abs(ct[1] - (5.0 + 10.0 - 2.0)) < 1.0         # 進入第一個航點半徑 2 m 的時刻

    auto2 = (t < 20) | (t > 40)                          # 20~40 s 暫離 AUTO(手動接管)
    ct2, ck2 = mission.estimate_progress(t, enu, wp, auto2, radius=2.0)
    assert -1 in ck2 and ck2[0] == 0
    k_before = ck2[list(ck2).index(-1) - 1]
    k_after = ck2[list(ck2).index(-1) + 1]
    assert k_after >= k_before and ck2[-1] == 4, list(ck2)   # 進度保留,回到 AUTO 續飛

    jump = np.column_stack([np.where(t < 30, 0.0, 30.0), np.zeros(t.size), np.zeros(t.size)])
    _, ck3 = mission.estimate_progress(t, jump, wp, np.ones(t.size, bool), radius=2.0, look_ahead=2)
    assert ck3[-1] == 3, list(ck3)                       # 直接出現在 WP3 附近:一次跳過 WP1、WP2
    _, ck4 = mission.estimate_progress(t, enu, np.zeros((0, 3)), auto, radius=2.0)
    assert list(ck4) == [-1]                             # 沒有航點 → 永遠沒有目標

    logdata = sim.make_sim_logdata()
    wps = mission.load_mission(logdata)
    ref = logdata.sources["GPS"]
    mask = mission.auto_mask(timeline.mode_changes(logdata.events), ref.t)
    assert 0 < mask.sum() < ref.n
    ct5, ck5 = mission.estimate_progress(ref.t, ref.enu(logdata.origin),
                                         mission.waypoints_enu(wps, logdata.origin), mask,
                                         mission.reach_radius(logdata))
    reached = [k for k in ck5 if k >= 0]
    assert reached == list(range(len(wps) + 1)), reached  # 模擬飛行依序飛完全部 16 個航點
    print(f"  合成路徑依序到達、AUTO 外不推進、可續飛 / 跳過;模擬飛行 {len(wps)} 個航點依序到達")


def test_ekf_leak_coefficient():
    """洩漏係數:EKF 只跟 VIO → 0、被 GPS 帶著走 → 1、混合 → 混合比;GPS 與 VIO 太接近 → 不可判定。"""
    rng = np.random.default_rng(3)
    n = 600
    vio = np.cumsum(rng.normal(0, 0.05, (n, 3)), axis=0)
    drift = np.column_stack([np.linspace(0, 6, n) * np.sin(np.linspace(0, 6, n)),
                             np.linspace(0, 4, n), np.zeros(n)])
    gps = vio + drift + rng.normal(0, 0.1, (n, 3))
    a0, s0 = ekfsrc.leak_coefficient(vio, vio, gps)
    a1, _ = ekfsrc.leak_coefficient(gps, vio, gps)
    ah, _ = ekfsrc.leak_coefficient(0.3 * gps + 0.7 * vio, vio, gps)
    assert abs(a0) < 0.05 and abs(a1 - 1) < 0.05 and abs(ah - 0.3) < 0.05, (a0, a1, ah)
    assert s0 > 0.5
    near, sb = ekfsrc.leak_coefficient(vio, vio, vio + rng.normal(0, 0.05, (n, 3)))
    assert np.isnan(near) and sb < 0.5
    assert np.isnan(ekfsrc.leak_coefficient(vio[:3], vio[:3], gps[:3])[0])
    print(f"  α:只跟 VIO {a0:.3f}、GPS {a1:.3f}、30% GPS {ah:.3f};GPS≈VIO → 不可判定")


def test_ekf_source_sets_and_segments():
    """來源組設定判讀(純 VIO / VIO+GPS / GPS)、使用中來源組的分段(切換事件、假設值)。"""
    pure = ekfsrc.SourceSet(2, {"POSXY": 6, "POSZ": 1, "VELXY": 6, "VELZ": 0, "YAW": 6})
    assert pure.gps_axes == [] and pure.verdict.startswith("純 VIO")
    mixed = ekfsrc.SourceSet(3, {"POSXY": 6, "POSZ": 1, "VELXY": 3, "VELZ": 0, "YAW": 1})
    assert mixed.gps_axes == ["VELXY"] and "水平速度" in mixed.verdict and mixed.verdict.startswith("VIO + GPS")
    assert ekfsrc.SourceSet(1, {"POSXY": 3, "POSZ": 1, "VELXY": 3, "VELZ": 3, "YAW": 1}).verdict.startswith("GPS")
    assert ekfsrc.SourceSet(3, {a: 0 for a, _ in ekfsrc.AXES}).verdict == "未設定"
    assert ekfsrc.SourceSet(1, {"POSXY": 0, "POSZ": 3, "VELXY": 0, "VELZ": 0, "YAW": 0}).gps_axes == ["POSZ"]
    assert ekfsrc.SourceSet(1, {"POSXY": 0, "POSZ": 0, "VELXY": 0, "VELZ": 0, "YAW": 2}).gps_axes == ["YAW"]

    logdata = sim.make_sim_logdata()
    sets, opt = ekfsrc.source_sets(logdata)
    assert sets[1].verdict.startswith("GPS") and sets[2].verdict.startswith("純 VIO"), (sets[1].verdict, sets[2].verdict)
    info = ekfsrc.inspect(logdata)
    assert info.assumed and [s.set_n for s in info.segments] == [1]        # 沒有切換紀錄 → 假設來源組 1
    tb = logdata.t_base
    logdata.events += [LogEvent(tb + 50, "ev", "EK3_SOURCES_SET_TO_SECONDARY"),
                       LogEvent(tb + 80, "ev", "EK3_SOURCES_SET_TO_PRIMARY")]
    segs = ekfsrc.inspect(logdata).segments
    assert [s.set_n for s in segs] == [1, 2, 1], [s.set_n for s in segs]
    assert abs(segs[1].t0 - (tb + 50)) < 1e-6 and abs(segs[1].t1 - (tb + 80)) < 1e-6
    assert abs(segs[0].t0 - tb) < 1e-6 and abs(segs[-1].t1 - logdata.t_last) < 1e-6
    assert all(0.99 <= (s.gps_fix or 1.0) <= 1.0 for s in segs)            # 模擬資料 GPS 全程 3D 定位
    print("  來源組判讀正確;切換事件 → 分段 [1, 2, 1];沒紀錄 → 假設來源組 1")


def test_ekf_leak_analysis_on_sim():
    """整條管線:EKF 輸出 = 對齊後的 VIO → α≈0;EKF 輸出 = GPS → α≈1;來源不足 → 例外。"""
    from core.types import KIND_GLOBAL, TrajSource
    logdata = sim.make_sim_logdata()
    res = pipeline.compute(logdata, "GPS", "VISP", method="se3")
    lat, lng, alt = frames.enu_to_geodetic(res.est_enu, res.origin)
    srcs = dict(logdata.sources)
    srcs["EKF_V"] = TrajSource("EKF_V", KIND_GLOBAL, res.t, lat=lat, lng=lng, alt=alt)
    g = logdata.sources["GPS"]
    srcs["EKF_G"] = TrajSource("EKF_G", KIND_GLOBAL, g.t, lat=g.lat, lng=g.lng, alt=g.alt)
    ld2 = LogData(sources=srcs, origin=logdata.origin, origin_desc=logdata.origin_desc, path="sim")
    rv = ekfsrc.leak_analysis(ld2, "VISP", "GPS", "EKF_V", min_std=0.1)
    rg = ekfsrc.leak_analysis(ld2, "VISP", "GPS", "EKF_G", min_std=0.1)
    assert abs(rv.overall) < 0.1, rv.overall
    assert abs(rg.overall - 1.0) < 0.1, rg.overall
    ok = np.isfinite(rv.alpha)
    assert ok.sum() >= 3 and abs(np.nanmedian(rv.alpha)) < 0.2 and abs(np.nanmedian(rg.alpha) - 1) < 0.2
    info = ekfsrc.inspect(logdata)
    ekfsrc.apply_leak(info, rg)
    assert info.segments[0].alpha is not None and abs(info.segments[0].alpha - 1) < 0.1
    assert "跟 GPS 一致" in ekfsrc.judge(info, info.segments[0]) or "GPS" in ekfsrc.judge(info, info.segments[0])
    for bad in (("VISP", "GPS", "GPS"), ("VISP", "NOPE", "EKF_V")):
        try:
            ekfsrc.leak_analysis(ld2, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"應拋出 ValueError:{bad}")
    txt = ekfsrc.format_report(logdata, info, rg)
    assert "來源組設定" in txt and "洩漏係數" in txt
    print(f"  EKF=VIO α={rv.overall:+.3f}、EKF=GPS α={rg.overall:.3f};報告含設定表與洩漏係數")


def test_wall_clock():
    """GPS 週 / 毫秒 → UTC:閏秒表、跨 2017-01-01 閏秒、由 log 求真實時間(無 GPS 時間 → None)。"""
    from datetime import datetime, timezone

    def utc(*a):
        return datetime(*a, tzinfo=timezone.utc).timestamp()

    # 已知對照:2020-01-01 00:00:00 UTC(週三)= GPS 週 2086、週內 3 天 + 18 秒 = 259218 s
    assert abs(clock.gps_to_utc_unix(2086, 259218 * 1000) - utc(2020, 1, 1)) < 1e-6
    assert clock.gps_minus_utc(utc(2020, 1, 1)) == 18 and clock.gps_minus_utc(utc(2016, 6, 1)) == 17
    assert clock.gps_minus_utc(utc(1981, 6, 30)) == 0 and clock.gps_minus_utc(utc(1981, 7, 1)) == 1
    # 同一個 GPS 時刻在閏秒前 / 後換算出不同的 UTC 偏移
    g0 = clock.GPS_EPOCH_UNIX + 1900 * 604800          # 2016-06-12 附近(閏秒數 17)
    assert abs((g0 - clock.gps_to_utc_unix(1900, 0)) - 17) < 1e-6
    g1 = clock.GPS_EPOCH_UNIX + 2100 * 604800          # 2020-03 附近(閏秒數 18)
    assert abs((g1 - clock.gps_to_utc_unix(2100, 0)) - 18) < 1e-6

    logdata = sim.make_sim_logdata()
    wc = clock.wall_clock(logdata)
    assert wc is not None
    g = logdata.index.series("GPS.GMS")
    wk = logdata.index.series("GPS.GWk")[1]
    for i in (0, len(g[0]) // 2, len(g[0]) - 1):       # 任一 GPS 樣本:wall(t) = 該樣本的 GPS 時間換算
        want = clock.gps_to_utc_unix(wk[i], g[1][i])
        assert abs(wc.unix_at(g[0][i]) - want) < 0.01, (i, wc.unix_at(g[0][i]) - want)
    assert wc.utc_at(logdata.t_base).tzinfo is not None and wc.local_at(logdata.t_base).tzinfo is not None
    assert clock.format_local(wc.local_at(logdata.t_base)).count(":") == 2

    class _NoIdx:
        index = None
    assert clock.wall_clock(_NoIdx()) is None
    t = np.arange(10.0)                                 # 有 GPS 但沒有 3D 定位 → 不可信
    bad = dflog.LogIndex.from_tables({"GPS": {"t": t, "Status": np.full(10, 2, np.uint8),
                                              "GWk": np.full(10, 2300, np.uint16),
                                              "GMS": (t * 1000 + 1000).astype(np.uint32),
                                              "Lat": np.zeros(10), "Lng": np.zeros(10), "Alt": np.zeros(10)}})
    class _L:
        index = bad
    assert clock.wall_clock(_L()) is None
    print(f"  GPS 週 / 毫秒換算與閏秒表正確;模擬 log 起點 {clock.format_local(wc.utc_at(logdata.t_base))} UTC")


def test_stat_armed_fallback():
    """ARM、EV 都沒有時用 STAT.Armed 推 ARM / DISARM;有 ARM 或 EV 時不用 STAT。"""
    from core.types import EV_ARM, EV_DISARM
    t = np.arange(0, 60.0, 1.0)
    gps = {"t": t, "Status": np.full(t.size, 3, np.uint8), "Lat": 23.7 + 1e-6 * t, "Lng": 120.4 + 0 * t,
           "Alt": np.full(t.size, 30.0), "GWk": np.full(t.size, 2300, np.uint16),
           "GMS": (t * 1000 + 1000).astype(np.uint32)}
    armed = ((t >= 10) & (t < 30)) | (t >= 45)         # 10 s 解鎖、30 s 上鎖、45 s 再解鎖(之後沒上鎖)
    stat = {"t": t, "Armed": armed.astype(np.uint8), "isFlying": np.zeros(t.size, np.uint8)}
    ld = logio.logdata_from_index(dflog.LogIndex.from_tables({"GPS": gps, "STAT": stat}))
    arm = [(round(e.t), e.kind) for e in ld.events if e.kind in (EV_ARM, EV_DISARM)]
    assert arm == [(10, EV_ARM), (30, EV_DISARM), (45, EV_ARM)], arm
    # 開頭就已解鎖 → 第一筆算 ARM
    stat2 = dict(stat, Armed=np.ones(t.size, np.uint8))
    ld2 = logio.logdata_from_index(dflog.LogIndex.from_tables({"GPS": gps, "STAT": stat2}))
    assert [(round(e.t), e.kind) for e in ld2.events if e.kind in (EV_ARM, EV_DISARM)] == [(0, EV_ARM)]
    # 有 EV 10/11 時以 EV 為準(STAT 不參與)
    ev = {"t": np.array([12.0, 28.0]), "Id": np.array([10, 11], np.uint8)}
    ld3 = logio.logdata_from_index(dflog.LogIndex.from_tables({"GPS": gps, "STAT": stat, "EV": ev}))
    assert [(round(e.t), e.kind) for e in ld3.events if e.kind in (EV_ARM, EV_DISARM)] == [(12, EV_ARM), (28, EV_DISARM)]
    ld4 = logio.logdata_from_index(dflog.LogIndex.from_tables({"GPS": gps}))
    assert not [e for e in ld4.events if e.kind in (EV_ARM, EV_DISARM)]
    print("  STAT.Armed → ARM / DISARM [10 s, 30 s, 45 s];已解鎖開頭、EV 優先、無紀錄皆正確")


def test_mission_int_latlng():
    """CMD 經緯度存成 1e7 倍整數時要還原成度。"""
    n = 4
    cmd = {"t": 3.0 + 0.001 * np.arange(n), "CTot": np.full(n, n, np.uint16), "CNum": np.arange(n, dtype=np.uint16),
           "CId": np.full(n, 16, np.uint16), "Lat": np.array([0, 237020000, 237020500, 237021000], np.int64),
           "Lng": np.array([0, 1204310000, 1204310500, 1204311000], np.int64),
           "Alt": np.full(n, 20.0), "Frame": np.full(n, 3, np.uint8)}
    class _L:
        index = dflog.LogIndex.from_tables({"CMD": cmd})
    w = mission.load_mission(_L())
    assert [x.n for x in w] == [1, 2, 3]
    assert abs(w[0].lat - 23.702) < 1e-9 and abs(w[0].lng - 120.431) < 1e-9, (w[0].lat, w[0].lng)
    cmd2 = dict(cmd, Lat=cmd["Lat"] * 1e-7, Lng=cmd["Lng"] * 1e-7)          # 本來就是度:不縮放
    class _L2:
        index = dflog.LogIndex.from_tables({"CMD": cmd2})
    w2 = mission.load_mission(_L2())
    assert abs(w2[0].lat - 23.702) < 1e-9 and abs(w2[2].lng - 120.4311) < 1e-9
    print("  整數 1e7 倍經緯度還原成度;原本是度的不再縮放")


def test_bin_roundtrip():
    """寫出 .bin → 用 pymavlink 讀回 → ATE 結果應與原始模擬資料一致。"""
    logdata, _ = sim.make_sim_with_truth(seed=7)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sim.bin")
        fakebin.write_logdata_to_bin(logdata, p)
        size = os.path.getsize(p)
        loaded = logio.load_bin(p)
        print(f"  寫出 {size / 1024:.0f} KB .bin → 解析出來源 {sorted(loaded.sources)}")
        for name in ("GPS", "POS", "VISP"):
            assert name in loaded.sources, f"缺少 {name}(實得 {sorted(loaded.sources)})"
        assert loaded.origin is not None, "ORGN 未被解析"
        assert abs(loaded.origin[0] - sim.ORIGIN[0]) < 1e-6

        for name in ("GPS", "VISP", "POS"):
            a, b = logdata.sources[name], loaded.sources[name]
            assert a.n == b.n, f"{name} 樣本數 {a.n} → {b.n}"
            assert np.max(np.abs(a.t - b.t)) < 1e-5

        res_a = pipeline.compute(logdata, "GPS", "VISP", method="se3")
        res_b = pipeline.compute(loaded, "GPS", "VISP", method="se3")
        d_rmse = abs(res_a.stats["3d"]["rmse"] - res_b.stats["3d"]["rmse"])
        assert d_rmse < 0.02, f"RMSE 差異 {d_rmse:.4f} m 過大"
        print(f"  .bin 往返 RMSE:{res_a.stats['3d']['rmse']:.4f} → "
              f"{res_b.stats['3d']['rmse']:.4f} m(差 {d_rmse * 100:.2f} cm,經緯度量化所致)")


# ---------------- 時間軸 / 多航段 ----------------

def _ev(t, kind, label=""):
    return LogEvent(float(t), kind, label)


def test_arm_segments_edge_cases():
    A, D = EV_ARM, EV_DISARM
    assert timeline.arm_segments([], 0.0, 100.0) == []
    assert timeline.arm_segments([_ev(1, EV_MODE, "AUTO")], 0.0, 10.0) == []

    segs = timeline.arm_segments([_ev(10, A), _ev(20, D), _ev(40, A), _ev(55, D)], 0.0, 100.0)
    assert [(s.t0, s.t1, s.label) for s in segs] == [(10, 20, "ARM 1"), (40, 55, "ARM 2")]
    assert not any(s.open_start or s.open_end for s in segs)
    assert segs[0].duration == 10 and segs[0].contains(15) and not segs[0].contains(25)

    # log 開頭就已 ARM(第一個狀態事件是 DISARM)
    segs = timeline.arm_segments([_ev(5, D), _ev(30, A), _ev(50, D)], 0.0, 100.0)
    assert (segs[0].t0, segs[0].t1, segs[0].open_start) == (0.0, 5.0, True)
    assert not segs[1].open_start and segs[1].label == "ARM 2"

    # 到 log 結尾都沒有 DISARM
    segs = timeline.arm_segments([_ev(10, A), _ev(20, D), _ev(70, A)], 0.0, 100.0)
    assert (segs[-1].t0, segs[-1].t1, segs[-1].open_end) == (70, 100, True)

    # 重複 ARM / DISARM(ARM 與 EV 同時記錄、狀態重送)+ 輸入未排序
    segs = timeline.arm_segments([_ev(20.002, D), _ev(10, A), _ev(10.001, A), _ev(20, D)], 0.0, 100.0)
    assert [(s.t0, s.t1) for s in segs] == [(10, 20)], segs

    # 裁切到 [t_start, t_end];完全在範圍外的段捨棄;模式事件不影響
    segs = timeline.arm_segments([_ev(-5, A), _ev(10, D), _ev(50, EV_MODE, "AUTO"), _ev(90, A),
                                  _ev(120, D), _ev(200, A), _ev(210, D)], 0.0, 100.0)
    assert [(s.t0, s.t1) for s in segs] == [(0, 10), (90, 100)], segs
    assert [s.label for s in segs] == ["ARM 1", "ARM 2"], "編號應連續(捨棄的段不佔號)"
    print("  ARM 段:空 / 開頭已 ARM / 未 DISARM / 重複事件 / 裁切 皆正確")

    assert timeline.segment_index_at(segs, 95) == 1 and timeline.segment_index_at(segs, 50) is None
    assert timeline.overlapping_segments(segs, 5, 95) == [0, 1]
    assert timeline.overlapping_segments(segs, 20, 80) == []
    assert timeline.segment_matching(segs, 90.02, 99.97) == 1
    assert timeline.segment_matching(segs, 80, 100) is None


def test_mode_changes_and_mode_at():
    evs = [_ev(30, EV_MODE, "AUTO"), _ev(10, EV_MODE, "LOITER"), _ev(20, EV_MODE, "LOITER"),
           _ev(15, EV_ARM), _ev(50, EV_MODE, "LAND")]
    modes = timeline.mode_changes(evs)
    assert [(m.t, m.label) for m in modes] == [(10, "LOITER"), (30, "AUTO"), (50, "LAND")]
    assert timeline.mode_at(modes, 5) is None
    assert timeline.mode_at(modes, 10) == "LOITER"
    assert timeline.mode_at(modes, 29.9) == "LOITER"
    assert timeline.mode_at(modes, 30) == "AUTO"
    assert timeline.mode_at(modes, 1e9) == "LAND"
    assert timeline.mode_at([], 1.0) is None
    print("  模式切換:排序、相鄰同名合併、逐時查詢正確")


def test_track_interp_gap_trail():
    t = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
    p = np.column_stack([t, 2 * t, np.zeros_like(t)])
    tr = timeline.Track(t, p, max_gap=1.5)
    assert np.allclose(tr.at(0.5), [0.5, 1.0, 0.0])
    assert np.allclose(tr.at(10.25), [10.25, 20.5, 0.0])
    assert tr.at(-0.1) is None and tr.at(11.1) is None, "超出時間範圍應為 None"
    assert tr.at(6.0) is None, "資料空洞中間不該內插"
    assert np.allclose(tr.at(2.5), p[2]) and np.allclose(tr.at(9.5), p[3]), "貼近空洞端點取該樣本"
    assert tr.index_at(0.4) == 0 and tr.index_at(0.6) == 1 and tr.index_at(50) is None

    tail = tr.trail(0.0, 1.5)
    assert tail.shape == (3, 3) and np.allclose(tail[-1], tr.at(1.5)), "尾巴末端應貼住游標"
    tail = tr.trail(0.0, 2.0)
    assert tail.shape == (3, 3) and np.allclose(tail[-1], p[2]), "游標正好在樣本上不重複補點"
    assert timeline.Track(np.empty(0), np.empty((0, 3))).at(0.0) is None

    tv = np.arange(0.0, 5.0, 0.2)
    ts, v = timeline.horizontal_speed(tv, np.column_stack([3 * tv, 4 * tv, tv]))
    assert ts.size == tv.size - 1 and np.allclose(v, 5.0), "水平速度只看東北分量"
    print("  Track:內插、範圍外 / 空洞回傳 None、尾跡末端正確;水平速度正確")


def test_binned():
    t = np.array([0.1, 0.2, 0.3, 2.5, 2.6])
    v = np.array([1.0, 2.0, 6.0, 5.0, np.nan])
    c, m = timeline.binned(t, v, 0.0, 3.0, 3, "mean")
    assert np.allclose(c, [0.5, 1.5, 2.5])
    assert np.isclose(m[0], 3.0) and np.isnan(m[1]) and np.isclose(m[2], 5.0), m
    _, mx = timeline.binned(t, v, 0.0, 3.0, 3, "max")
    assert mx[0] == 6.0 and np.isnan(mx[1]) and mx[2] == 5.0, mx
    _, mn = timeline.binned(t, v, 0.0, 3.0, 3, "min")
    assert mn[0] == 1.0 and np.isnan(mn[1])
    _, e = timeline.binned(np.empty(0), np.empty(0), 0.0, 1.0, 4)
    assert e.shape == (4,) and np.all(np.isnan(e))
    _, e = timeline.binned(t, v, 5.0, 5.0, 2)
    assert np.all(np.isnan(e)), "hi <= lo 應全為 NaN"
    try:
        timeline.binned(t, v, 0.0, 3.0, 3, "median")
    except ValueError:
        pass
    else:
        raise AssertionError("未知 reducer 應拋出 ValueError")
    print("  分格壓縮:平均 / 最大 / 最小、空格 NaN 正確")


def test_playback_clock():
    clk = timeline.PlaybackClock(100.0, 300.0)
    assert clk.t == 100.0 and not clk.playing
    assert clk.seek(50.0) == 100.0 and clk.seek(1e9) == 300.0, "seek 應夾在範圍內"
    assert clk.step(-20.0) == 280.0

    clk.set_play_range(200.0, 150.0)            # 順序顛倒也接受
    assert (clk.play_lo, clk.play_hi) == (150.0, 200.0)
    clk.seek(120.0)
    clk.play()
    assert clk.playing and clk.t == 150.0, "播放頭在播放範圍外 → 跳到範圍起點"
    clk.speed = 4.0
    assert np.isclose(clk.advance(2.0), 158.0), "速度 4× → 2 s 走 8 s"
    clk.pause()
    assert clk.advance(5.0) == 158.0 and not clk.playing, "暫停時不前進"
    assert clk.toggle() and np.isclose(clk.t, 158.0), "範圍內續播不跳回起點"
    clk.advance(100.0)
    assert clk.t == 200.0 and not clk.playing, "走到播放範圍終點自動停"
    clk.play()
    assert clk.t == 150.0 and clk.playing, "停在終點再按播放 → 從頭"

    clk.set_play_range(250.0, 999.0)
    assert clk.play_hi == 300.0, "播放範圍也夾在 bounds 內"
    clk.set_bounds(0.0, 50.0)
    assert clk.t <= 50.0 and clk.play_hi <= 50.0
    z = timeline.PlaybackClock(5.0, 5.0)
    z.play()
    assert not z.playing, "長度為零的範圍不能播放"
    print("  播放時鐘:跳到範圍起點、終點自動停、seek 夾範圍、速度倍率正確")


def test_format_clock():
    fc = timeline.format_clock
    assert fc(0) == "0:00.0"
    assert fc(65.04) == "1:05.0"
    assert fc(59.96) == "1:00.0", fc(59.96)
    assert fc(3725) == "1:02:05"
    assert fc(-3.5) == "-0:03.5"
    assert fc(float("nan")) == "—"
    assert fc(125.7, decimals=0) == "2:05"
    assert fc(7.123, decimals=2) == "0:07.12"
    print("  時鐘格式:m:ss.s、進位、h:mm:ss、負值、NaN 正確")


def test_apply_alignment_reproduces_result():
    """apply_alignment 套到同步後的原始估測點,應與 compute() 的 est_enu 完全相同。"""
    logdata = sim.make_sim_logdata()
    rt, rp, et, ep = pipeline.get_enu_pair(logdata, "GPS", "VISP")
    _, _, ep_raw = sync.sync_pair(rt, rp, et, ep)
    worst = 0.0
    for method, _ in align.ALIGN_METHODS:
        res = pipeline.compute(logdata, "GPS", "VISP", method=method)
        assert np.asarray(res.align_info["rotation"]).shape == (3, 3)
        out = align.apply_alignment(ep_raw, res.align_info)
        worst = max(worst, float(np.max(np.abs(out - res.est_enu))))
    assert worst < 1e-9, worst
    full = align.apply_alignment(ep, res.align_info)
    assert full.shape == ep.shape
    print(f"  apply_alignment 重現各對齊方式的 est_enu:最大差 {worst:.1e} m")


def test_sim_single_flight_events():
    logdata, truth = sim.make_sim_with_truth(seed=7)
    ts = [e.t for e in logdata.events]
    assert ts == sorted(ts), "events 應依時間排序"
    segs = logdata.segments
    assert len(segs) == 1, segs
    sg = segs[0]
    assert not sg.open_start and not sg.open_end
    starts = truth["phases"]
    t_climb = next(t for n, t in starts if n == "climb")
    t_cruise = next(t for n, t in starts if n == "cruise")
    t_cruise_end = [t for n, t in starts if n == "hover"][1]
    t_touch = starts[-1][1]
    assert logdata.t_base < sg.t0 < t_climb, "ARM 應在起飛前的地面段"
    assert t_touch < sg.t1 < logdata.t_end, "DISARM 應在著地後的地面段"
    assert sg.contains(t_cruise) and sg.contains(t_cruise_end), "飛行段應涵蓋整段巡航"
    modes = timeline.mode_changes(logdata.events)
    assert [m.label for m in modes] == ["STABILIZE", "LOITER", "AUTO", "LAND"], modes
    assert timeline.mode_at(modes, 0.5 * (t_cruise + t_cruise_end)) == "AUTO"
    assert not any(e.kind == EV_RESET for e in logdata.events)
    tb = logdata.t_base
    print(f"  單航段:ARM {sg.t0 - tb:.1f} ~ {sg.t1 - tb:.1f} s,模式 {[m.label for m in modes]}")


def test_multi_flight_sim():
    logdata, info = sim.make_multi_flight_sim()
    tb = logdata.t_base
    segs = logdata.segments
    assert len(segs) == 2, segs
    for sg, (a, d) in zip(segs, info["flights"]):
        assert abs(sg.t0 - a) < 1e-9 and abs(sg.t1 - d) < 1e-9
    assert segs[0].t0 - tb > 15.0, "第一次 ARM 前應有一長段地面待機"
    resets = [e for e in logdata.events if e.kind == EV_RESET]
    assert len(resets) == 1 and resets[0].label == "VISP"
    assert segs[0].t1 < info["reset_t"] <= segs[1].t0, "VIO 重置應在第二次 ARM 時(之前)"
    labels = [m.label for m in timeline.mode_changes(logdata.events)]
    assert labels == ["STABILIZE"] + ["LOITER", "AUTO", "LAND", "STABILIZE"] * 2, labels

    # VISP 在重置點跳回新座標框原點附近
    v = logdata.sources["VISP"]
    k = int(np.searchsorted(v.t, info["reset_t"]))
    assert np.linalg.norm(v.ned[k] - v.ned[k - 1]) > 5.0, "重置點應有明顯跳點"

    # 上鎖時被手提移動:兩段之間的水平速度仍有活動
    pos = logdata.sources["POS"]
    ts, sp = timeline.horizontal_speed(pos.t, pos.enu(logdata.origin))
    _, sp_bin = timeline.binned(ts, sp, segs[0].t1, segs[1].t0, 20, "mean")
    assert np.nanmax(sp_bin) > 0.8, f"上鎖期間應看得到步行速度(得 {np.nanmax(sp_bin):.2f} m/s)"

    whole = pipeline.compute(logdata, "GPS", "VISP", method="se3").stats["3d"]["rmse"]
    per = [pipeline.compute(logdata, "GPS", "VISP", t0=sg.t0, t1=sg.t1, method="se3")
           .stats["3d"]["rmse"] for sg in segs]
    print(f"  多航段 SE(3) 3D RMSE:整份 {whole:.3f} m vs 逐段 "
          + " / ".join(f"{x:.3f}" for x in per) + " m")
    assert whole >= 2.0 * max(per), "整份一起對齊應明顯比逐段差(兩個 VIO 座標框)"
    assert max(per) < 2.0, per


def test_report_segment_line():
    logdata, _ = sim.make_multi_flight_sim()
    tb = logdata.t_base
    segs = logdata.segments

    def seg_line(res, ld):
        lines = pipeline.format_report(res, ld).splitlines()
        i = next(i for i, ln in enumerate(lines) if ln.startswith("分析區間"))
        return lines[i + 1]

    res = pipeline.compute(logdata, "GPS", "VISP", t0=segs[1].t0, t1=segs[1].t1)
    ln = seg_line(res, logdata)
    assert ln.startswith("ARM 飛行段:共 2 段;分析區間涵蓋 ARM 2(") and "部分" not in ln, ln
    assert "ARM 1" not in ln

    ln = seg_line(pipeline.compute(logdata, "GPS", "VISP"), logdata)
    assert "ARM 1(" in ln and "ARM 2(" in ln, ln

    ln = seg_line(pipeline.compute(logdata, "GPS", "VISP", t0=tb + 1, t1=tb + 15), logdata)
    assert ln.endswith("無(全在未 ARM 的地面段)"), ln

    ln = seg_line(pipeline.compute(logdata, "GPS", "VISP", t0=tb + 100, t1=tb + 200), logdata)
    assert "部分" in ln, ln

    # 沒有事件的 log(或模擬)不輸出此行,其餘格式不變
    bare = LogData(sources=logdata.sources, origin=logdata.origin,
                   origin_desc=logdata.origin_desc, path="bare")
    txt = pipeline.format_report(pipeline.compute(bare, "GPS", "VISP"), bare)
    assert "ARM 飛行段" not in txt
    for token in ("ATE RMSE", "RMSE", "P95", "空間對齊", "ENU 原點", "時間偏移"):
        assert token in txt
    print(f"  報告 ARM 段說明:{seg_line(res, logdata)}")


def test_bin_events_roundtrip():
    """多航段模擬 → .bin(有 / 無 ARM 訊息)→ 讀回:飛行段、模式名稱、VIO 重置一致。"""
    logdata, info = sim.make_multi_flight_sim()
    want_modes = timeline.mode_changes(logdata.events)
    with tempfile.TemporaryDirectory() as d:
        for arm_msg in (True, False):
            p = os.path.join(d, f"multi_{int(arm_msg)}.bin")
            fakebin.write_logdata_to_bin(logdata, p, arm_msg=arm_msg)
            loaded = logio.load_bin(p)
            ts = [e.t for e in loaded.events]
            assert ts == sorted(ts), "events 應依時間排序"

            a_segs, b_segs = logdata.segments, loaded.segments
            assert len(b_segs) == len(a_segs) == 2, b_segs
            dmax = max(max(abs(a.t0 - b.t0), abs(a.t1 - b.t1)) for a, b in zip(a_segs, b_segs))
            assert dmax < 1e-3, f"飛行段邊界差 {dmax:.2e} s"
            n_arm = sum(e.kind == EV_ARM for e in loaded.events)
            n_dis = sum(e.kind == EV_DISARM for e in loaded.events)
            assert (n_arm, n_dis) == (2, 2), f"ARM/EV 不可重複計算(得 {n_arm} / {n_dis})"

            got_modes = timeline.mode_changes(loaded.events)
            assert [m.label for m in got_modes] == [m.label for m in want_modes], \
                [m.label for m in got_modes]
            assert {"STABILIZE", "LOITER", "AUTO", "LAND"} <= {m.label for m in got_modes}
            assert max(abs(a.t - b.t) for a, b in zip(want_modes, got_modes)) < 1e-3

            resets = [e for e in loaded.events if e.kind == EV_RESET]
            assert len(resets) == 1 and resets[0].label == "VISP", resets
            lag = resets[0].t - info["reset_t"]
            assert -1e-3 < lag < 1.0 / 30 + 1e-3, f"重置時間差 {lag:.4f} s"
            assert any("2 個 ARM 飛行段" in n for n in loaded.notes), loaded.notes

            per = [pipeline.compute(loaded, "GPS", "VISP", t0=s.t0, t1=s.t1).stats["3d"]["rmse"]
                   for s in b_segs]
            ref = [pipeline.compute(logdata, "GPS", "VISP", t0=s.t0, t1=s.t1).stats["3d"]["rmse"]
                   for s in a_segs]
            assert max(abs(x - y) for x, y in zip(per, ref)) < 0.02
            src = "ARM 訊息(EV 重複記錄已去除)" if arm_msg else "只有 EV(舊韌體)"
            print(f"  {src}:{len(b_segs)} 段、模式 {len(got_modes)} 筆、重置 1 次 @ +{lag * 1000:.0f} ms")


def test_bin_reset_and_mode_edges():
    """重置計數一開始就非零不算重置;未知來源 / 未知模式名稱不寫入;單航段無多段註記。"""
    logdata = sim.make_sim_logdata()
    logdata.index = None          # 走舊寫法:由 events 產生 ResetCnt / MODE(索引寫法只寫索引表格)
    tb = logdata.t_base
    logdata.events += [_ev(tb - 1.0, EV_RESET, "VISP"),        # 開頭 ResetCnt 就是 1
                       _ev(tb + 50.0, EV_RESET, "VISP"),
                       _ev(tb + 60.0, EV_RESET, "XKF1"),        # 不存在的來源
                       _ev(tb + 70.0, EV_MODE, "NOT_A_MODE")]
    logdata.events.sort(key=lambda e: e.t)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "edge.bin")
        fakebin.write_logdata_to_bin(logdata, p)
        loaded = logio.load_bin(p)
    resets = [e for e in loaded.events if e.kind == EV_RESET]
    assert len(resets) == 1 and abs(resets[0].t - (tb + 50.0)) < 0.05, resets
    labels = [m.label for m in timeline.mode_changes(loaded.events)]
    assert labels == ["STABILIZE", "LOITER", "AUTO", "LAND"], labels
    assert len(loaded.segments) == 1
    assert not any("ARM 飛行段" in n for n in loaded.notes), "單一飛行段不需提示"
    print("  第一筆不算重置、未知模式略過、單航段不提示多段")


def test_cli_segments():
    import contextlib
    import io

    import cli

    def run(*argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    rc, out, _ = run("info", "--sim-multi")
    assert rc == 0 and "ARM 飛行段:2 段" in out and "VIO 重置:1 次" in out, out
    assert "LOITER" in out and "AUTO" in out
    rc, out, _ = run("ate", "--sim-multi", "--segment", "2", "--auto-offset")
    assert rc == 0 and "分析區間涵蓋 ARM 2(" in out, out
    rc, _, err = run("ate", "--sim-multi", "--segment", "3")
    assert rc == 2 and "可用" in err and "ARM 2" in err, err
    rc, _, err = run("ate", "--sim-multi", "--segment", "1", "--t0", "5")
    assert rc == 2 and "--segment" in err, err
    print("  CLI:info 列出段 / 模式 / 重置;ate --segment 正常與錯誤路徑正確")


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
