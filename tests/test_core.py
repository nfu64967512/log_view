"""核心驗證(不需 PyQt6):python -m tests.test_core 或 pytest tests/"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import align, fakebin, frames, logio, metrics, pipeline, sim, sync, timeline  # noqa: E402
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
