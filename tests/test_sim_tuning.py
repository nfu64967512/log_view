"""PID 調參閉迴路模擬驗證(不需 PyQt6):python tests/test_sim_tuning.py

每個情境的「已知答案」都要從 log 本身看得出來(不只看模擬器內部真值):
good 響應俐落無震盪、p_high 中頻持續震盪、d_high 高頻 / D 項雜訊、p_low 遲滯、i_low_bias 偏差、
ang_p_high 角度層過衝、yaw_p_low 偏航遲滯、saturation 馬達頂到上限、vibration VIBE / Clip、
noisy_notch batch 頻譜峰;另驗證格式 / 單位、10 Hz 記錄(取樣抽取 → 混疊)、.bin 往返、
決定性與執行時間、「套用預期修正後症狀消失」,以及壓力測試選項(只懸停、窄頻陀螺儀噪音、掉訊息、MOT_PWM = 0)。
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import fakebin, logio  # noqa: E402
from core import sim_tuning as st  # noqa: E402
from core.timeline import mode_changes  # noqa: E402
from core.tuning_model import DIR_CHECK, DIR_DOWN, DIR_SET, DIR_UP  # noqa: E402

_CACHE: dict = {}
_TIMES: dict = {}


def run(scenario: str, **kw):
    """make_tuning_sim 的快取版(同參數只跑一次)。"""
    key = (scenario, repr(sorted(kw.items())))
    if key not in _CACHE:
        t = time.perf_counter()
        _CACHE[key] = st.make_tuning_sim(scenario, **kw)
        _TIMES[key] = time.perf_counter() - t
    return _CACHE[key]


# ---------------- 小工具(numpy only,獨立於模擬器內部) ----------------

def welch(x: np.ndarray, fs: float, nper: int):
    x = np.asarray(x, dtype=float)
    w = np.hanning(nper)
    acc, n = 0.0, 0
    for i in range(0, x.size - nper + 1, nper // 2):
        s = x[i:i + nper]
        acc = acc + np.abs(np.fft.rfft((s - s.mean()) * w)) ** 2
        n += 1
    if n == 0:
        return None, None
    p = acc / (n * fs * np.sum(w ** 2))
    p[1:-1] *= 2
    return np.fft.rfftfreq(nper, 1.0 / fs), p


def band_rms(f, p, lo, hi) -> float:
    m = (f >= lo) & (f <= hi)
    return float(np.sqrt(np.trapz(p[m], f[m]))) if m.sum() > 1 else 0.0


def windows_psd(t, x, wins, fs, nper):
    """各窗各自 Welch 再以長度加權平均;另回傳每窗的 (f, p)。"""
    tot, cnt, per = None, 0, []
    f = None
    for a, b in wins:
        m = (t >= a) & (t < b)
        seg = x[m]
        if seg.size < nper:
            continue
        f, p = welch(seg, fs, nper)
        per.append(p)
        tot = p * seg.size if tot is None else tot + p * seg.size
        cnt += seg.size
    return f, (tot / cnt if cnt else None), per


def series(ld, key):
    t, v = ld.index.series(key)
    return np.asarray(t), np.asarray(v, dtype=float)


def mask_windows(t, wins):
    m = np.zeros(t.size, bool)
    for a, b in wins:
        m |= (t >= a) & (t < b)
    return m


def wrap180(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def flight_mask(t, info, margin=1.0):
    a, b = info["windows"]["flight"]
    return (t > a + margin) & (t < b - margin)


def tf_peak(x, y, fs, nper=512, lo=0.5, hi=12.0):
    """x → y 的 |H| 峰值(Welch 交叉譜;只看 coherence > 0.5)。"""
    w = np.hanning(nper)
    sxx = syy = 0.0
    sxy = 0.0
    for i in range(0, x.size - nper + 1, nper // 2):
        a = np.fft.rfft((x[i:i + nper] - x[i:i + nper].mean()) * w)
        b = np.fft.rfft((y[i:i + nper] - y[i:i + nper].mean()) * w)
        sxx = sxx + np.abs(a) ** 2
        syy = syy + np.abs(b) ** 2
        sxy = sxy + np.conj(a) * b
    f = np.fft.rfftfreq(nper, 1.0 / fs)
    h = np.abs(sxy) / sxx
    coh = np.abs(sxy) ** 2 / (sxx * syy)
    m = (f >= lo) & (f <= hi) & (coh > 0.5)
    i = int(np.argmax(np.where(m, h, 0.0)))
    return float(h[i]), float(f[i])


# ======================================================================

def test_api_and_scenarios():
    names = ["good", "p_high", "d_high", "p_low", "i_low_bias", "ang_p_high", "yaw_p_low",
             "saturation", "vibration", "noisy_notch"]
    assert list(st.SCENARIOS) == names, list(st.SCENARIOS)
    assert all(isinstance(v, str) and v for v in st.SCENARIOS.values())
    for bad in (lambda: st.make_tuning_sim("nope"), lambda: st.make_tuning_sim("good", duration_s=20),
                lambda: st.make_tuning_sim("good", frame="octa")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("應拋 ValueError")
    dflt = st.default_params()
    for name in names:
        _, info = run(name)
        for k in ("scenario", "expected", "forbid", "expected_codes", "osc_hz", "step", "angle_step",
                  "metrics", "params", "changed", "truth", "windows", "times", "plant", "log",
                  "saturation", "vibe", "runtime_s", "expect_ok", "expected_logging", "slow_log"):
            assert k in info, (name, k)
        assert (name == "good") == (not info["expected"]) == info["expect_ok"], name
        for prm, d in info["expected"]:
            assert d in (DIR_UP, DIR_DOWN, DIR_SET, DIR_CHECK), (name, d)
            if d in (DIR_UP, DIR_DOWN):     # 飛的值相對預設值偏向「需要反向修正」的那一邊
                v, v0 = info["params"][prm], dflt[prm]
                assert (v < v0) if d == DIR_UP else (v > v0), (name, prm, v, v0)
        # 禁止的建議與預期建議不衝突
        assert not set(info["expected"]) & set(info["forbid"]), name
    print("  10 個情境、info 欄位齊全;預期修正方向與飛行參數 / 預設值一致;錯誤輸入拋 ValueError")


def test_good_calibration():
    ld, info = run("good")
    s = info["step"]
    for ax in ("roll", "pitch"):
        assert 5.0 <= s[ax]["overshoot_pct"] <= 20.0, (ax, s[ax])
        assert s[ax]["rise_s"] < 0.08, s[ax]
    assert s["yaw"]["overshoot_pct"] <= 20.0, s["yaw"]
    a = info["angle_step"]["roll"]
    assert a["overshoot_vs_target_pct"] < 8.0, a
    m = info["metrics"]
    for ax in ("roll", "pitch"):
        assert m[ax]["rate_rms_hover_dps"] < 5.0, m[ax]
        assert abs(m[ax]["angle_bias_deg"]) < 0.3, m[ax]
        assert m[ax]["angle_err_rms_deg"] < 1.5, m[ax]
    assert info["saturation"]["at_max_pct"] < 0.5 and info["saturation"]["at_min_pct"] < 0.5
    assert max(info["vibe"]["mean"]) < 15.0 and info["vibe"]["clip_total"] == 0, info["vibe"]
    print(f"  預設增益:角速度步階過衝 Roll {s['roll']['overshoot_pct']:.1f}% / Pitch "
          f"{s['pitch']['overshoot_pct']:.1f}% / Yaw {s['yaw']['overshoot_pct']:.1f}%,上升 "
          f"{s['roll']['rise_s'] * 1000:.0f} ms;角度步階超出目標 {a['overshoot_vs_target_pct']:.1f}%;"
          f"懸停角速度 RMS {m['roll']['rate_rms_hover_dps']:.1f} deg/s;追蹤延遲 "
          f"{m['roll']['angle_lag_ms']:.0f} ms;VIBE {np.round(info['vibe']['mean'], 1)}")


def _hover_band(ld, info, key, lo, hi, fs=100.0, nper=128):
    t, v = series(ld, key)
    f, p, per = windows_psd(t, v, info["windows"]["hover"], fs, nper)
    return f, p, per


def test_p_high_oscillation():
    ld, info = run("p_high")
    gld, ginfo = run("good")
    hz = info["osc_hz"]
    assert 6.0 <= hz <= 15.0, hz
    for ax, key in (("roll", "RATE.R"), ("pitch", "RATE.P")):
        f, p, per = _hover_band(ld, info, key, 0, 50)
        gf, gp, gper = _hover_band(gld, ginfo, key, 0, 50)
        fpk = float(f[np.argmax(np.where((f > 3) & (f < 20), p, 0))])
        assert abs(fpk - hz) <= 1.2, (ax, fpk, hz)
        amp = band_rms(f, p, hz - 1.5, hz + 1.5)
        gamp = band_rms(gf, gp, hz - 1.5, hz + 1.5)
        assert amp >= 3.0 * gamp, (ax, amp, gamp)
        # 持續:幾乎每個懸停窗都看得到這個峰
        hits = [band_rms(f, q, hz - 1.5, hz + 1.5) >= 2.0 * gamp for q in per]
        assert np.mean(hits) >= 0.8, (ax, hits)
        print(f"  {ax}:震盪 {fpk:.1f} Hz(真值 {hz:.2f} Hz),頻帶 RMS {amp:.1f} vs good {gamp:.1f} deg/s,"
              f"{np.mean(hits) * 100:.0f}% 懸停窗可見")
    # P 項主導(PIDR.P 振幅 > D 項)
    tp, pv = series(ld, "PIDR.P")
    _, dv = series(ld, "PIDR.D")
    hm = mask_windows(tp, info["windows"]["hover"])
    assert np.std(pv[hm]) > 1.5 * np.std(dv[hm]), (np.std(pv[hm]), np.std(dv[hm]))
    assert info["step"]["roll"]["overshoot_pct"] > 40.0
    print(f"  PIDR.P / PIDR.D 標準差 {np.std(pv[hm]):.3f} / {np.std(dv[hm]):.3f};步階過衝 "
          f"{info['step']['roll']['overshoot_pct']:.0f}%")


def test_d_high_noise():
    ld, info = run("d_high")
    _, pinfo = run("p_high")
    gld, ginfo = run("good")
    hz = info["osc_hz"]
    assert pinfo["osc_hz"] < hz <= 20.0, (hz, pinfo["osc_hz"])
    # 控制輸出的高頻能量(D 項放大雜訊):100 Hz log 的 20~50 Hz 與真值 20~150 Hz
    f, p, _ = _hover_band(ld, info, "RATE.ROut", 0, 50)
    gf, gp, _ = _hover_band(gld, ginfo, "RATE.ROut", 0, 50)
    hf, ghf = band_rms(f, p, 20, 50), band_rms(gf, gp, 20, 50)
    assert hf >= 3.0 * ghf, (hf, ghf)
    for ax in ("roll", "pitch"):
        assert info["metrics"][ax]["out_hf_rms"] >= 3.0 * ginfo["metrics"][ax]["out_hf_rms"], ax
    tp, pv = series(ld, "PIDR.P")
    _, dv = series(ld, "PIDR.D")
    hm = mask_windows(tp, info["windows"]["hover"])
    assert np.std(dv[hm]) > 2.0 * np.std(pv[hm]), (np.std(dv[hm]), np.std(pv[hm]))
    tc, rd = series(ld, "CTRL.RMSRollD")
    gtc, grd = series(gld, "CTRL.RMSRollD")
    ratio = np.mean(rd[flight_mask(tc, info)]) / np.mean(grd[flight_mask(gtc, ginfo)])
    assert ratio >= 3.0, ratio
    print(f"  震盪 {hz:.2f} Hz(高於 p_high 的 {pinfo['osc_hz']:.2f} Hz);ROut 20~50 Hz RMS "
          f"{hf:.4f} vs good {ghf:.4f};D/P 標準差 {np.std(dv[hm]) / np.std(pv[hm]):.1f}×;"
          f"CTRL.RMSRollD {ratio:.1f}× good")


def test_p_low_sluggish():
    ld, info = run("p_low")
    gld, ginfo = run("good")
    for ax in ("roll", "pitch"):
        lag, glag = info["metrics"][ax]["angle_lag_ms"], ginfo["metrics"][ax]["angle_lag_ms"]
        assert lag >= 1.8 * glag, (ax, lag, glag)
        assert info["step"][ax]["rise_s"] >= 2.0 * ginfo["step"][ax]["rise_s"], ax
    # 沒有震盪:6~15 Hz 頻帶不比 good 大
    f, p, _ = _hover_band(ld, info, "RATE.R", 0, 50)
    gf, gp, _ = _hover_band(gld, ginfo, "RATE.R", 0, 50)
    assert band_rms(f, p, 6, 15) <= 1.5 * band_rms(gf, gp, 6, 15)
    # log 上的角度追蹤誤差(操作期間)
    def err(l, i):
        t, d = series(l, "ATT.DesRoll")
        _, a = series(l, "ATT.Roll")
        m = mask_windows(t, [(x, y + 1.0) for x, y in i["windows"]["manoeuvre"]])
        return float(np.sqrt(np.mean((d[m] - a[m]) ** 2)))
    e, ge = err(ld, info), err(gld, ginfo)
    assert e >= 1.5 * ge, (e, ge)
    print(f"  延遲 {info['metrics']['roll']['angle_lag_ms']:.0f} vs {ginfo['metrics']['roll']['angle_lag_ms']:.0f} ms、"
          f"上升 {info['step']['roll']['rise_s'] * 1000:.0f} vs {ginfo['step']['roll']['rise_s'] * 1000:.0f} ms、"
          f"操作時角度誤差 RMS {e:.2f} vs {ge:.2f}°,無中頻震盪")


def test_i_low_bias():
    ld, info = run("i_low_bias")
    gld, ginfo = run("good")

    def bias(l, i, a, b):
        t, d = series(l, f"ATT.{a}")
        _, x = series(l, f"ATT.{b}")
        m = mask_windows(t, i["windows"]["hover"])
        return float(np.mean(d[m] - x[m]))
    br, bp = bias(ld, info, "DesRoll", "Roll"), bias(ld, info, "DesPitch", "Pitch")
    # 重心力矩 roll 正、pitch 負 → 實際 Roll 大於期望、Pitch 小於期望
    assert br <= -1.0 and bp >= 1.0, (br, bp)
    gbr, gbp = bias(gld, ginfo, "DesRoll", "Roll"), bias(gld, ginfo, "DesPitch", "Pitch")
    assert abs(gbr) < 0.3 and abs(gbp) < 0.3, (gbr, gbp)
    # I 項還在慢慢爬(沒到穩態)
    t, iv = series(ld, "PIDR.I")
    fm = flight_mask(t, info, 5.0)
    assert abs(iv[fm][-1]) > abs(iv[fm][0]) + 0.005
    print(f"  懸停平均 Des−實際:Roll {br:+.2f}°、Pitch {bp:+.2f}°(good {gbr:+.2f}° / {gbp:+.2f}°);"
          f"PIDR.I {iv[fm][0]:+.4f} → {iv[fm][-1]:+.4f}")


def test_ang_p_high():
    ld, info = run("ang_p_high")
    gld, ginfo = run("good")

    def ang_h(l, i):
        t, d = series(l, "ATT.DesRoll")
        _, a = series(l, "ATT.Roll")
        fm = flight_mask(t, i, 3.0)
        return tf_peak(d[fm], a[fm], 100.0)
    h, fh = ang_h(ld, info)
    gh, gfh = ang_h(gld, ginfo)
    assert h >= gh + 0.3, (h, gh)
    # 角速度層級的步階不變(只改了 ANG_P)
    assert abs(info["step"]["roll"]["overshoot_pct"] - ginfo["step"]["roll"]["overshoot_pct"]) < 0.5
    a, ga = info["angle_step"]["roll"], ginfo["angle_step"]["roll"]
    assert a["overshoot_vs_target_pct"] >= ga["overshoot_vs_target_pct"] + 2.0, (a, ga)
    print(f"  角度層 |H| 峰 {h:.2f} @ {fh:.1f} Hz vs good {gh:.2f} @ {gfh:.1f} Hz;角度步階超出目標 "
          f"{a['overshoot_vs_target_pct']:.1f}% vs {ga['overshoot_vs_target_pct']:.1f}%;角速度步階相同")


def test_yaw_p_low():
    ld, info = run("yaw_p_low")
    gld, ginfo = run("good")

    def yaw_err(l, i):
        t, d = series(l, "ATT.DesYaw")
        _, a = series(l, "ATT.Yaw")
        wins = [(x, y + 1.0) for x, y, kind, axis, _ in i["manoeuvres"] if axis == "yaw"]
        m = mask_windows(t, wins)
        return float(np.sqrt(np.mean(wrap180(d[m] - a[m]) ** 2)))
    e, ge = yaw_err(ld, info), yaw_err(gld, ginfo)
    assert e >= 2.5 * ge, (e, ge)
    assert info["metrics"]["yaw"]["angle_err_rms_deg"] >= 3.0 * ginfo["metrics"]["yaw"]["angle_err_rms_deg"]
    s, gs = info["step"]["yaw"], ginfo["step"]["yaw"]
    assert math.isnan(s["rise_s"]) or s["rise_s"] >= 2.0 * gs["rise_s"], (s, gs)
    # roll / pitch 不受影響
    assert abs(info["step"]["roll"]["overshoot_pct"] - ginfo["step"]["roll"]["overshoot_pct"]) < 0.5
    rise = ">0.8 s" if math.isnan(s["rise_s"]) else f"{s['rise_s'] * 1000:.0f} ms"
    print(f"  偏航操作時 DesYaw−Yaw RMS {e:.1f}° vs good {ge:.1f}°;偏航步階上升 "
          f"{rise} vs {gs['rise_s'] * 1000:.0f} ms")


def _top_share(ld, info):
    rc = ld.index.table("RCOU")
    t = rc["t"]
    fm = flight_mask(t, info)
    pwm = np.column_stack([rc[f"C{i}"] for i in range(1, 5)]).astype(float)[fm]
    top = info["saturation"]["pwm_top"]
    return float(np.mean((pwm >= top - 1).any(axis=1)))


def test_saturation():
    ld, info = run("saturation")
    gld, ginfo = run("good")
    share, gshare = _top_share(ld, info), _top_share(gld, ginfo)
    assert share >= 0.05 and gshare <= 0.005, (share, gshare)
    assert ld.index.param_values()["MOT_THST_HOVER"] == np.float32(0.65)
    # 混控器先「砍油門保姿態」(limit.throttle_upper):CTUN.ThO(要求)> MOTB.ThrOut(實際)
    def cut_share(l, i):
        t, tho = series(l, "CTUN.ThO")
        _, out = series(l, "MOTB.ThrOut")
        fm = flight_mask(t, i)
        return float(np.mean((tho - out)[fm] > 0.02))
    cut, gcut = cut_share(ld, info), cut_share(gld, ginfo)
    assert cut >= 0.03 and gcut <= 0.002, (cut, gcut)
    for ax in ("roll", "pitch", "yaw"):
        assert info["metrics"][ax]["rate_err_rms_dps"] >= 2.0 * ginfo["metrics"][ax]["rate_err_rms_dps"], ax
    print(f"  RCOU 頂到上限(10 Hz、空中):{share * 100:.1f}% vs good {gshare * 100:.1f}%;油門被砍"
          f"(ThO−ThrOut>0.02){cut * 100:.1f}% vs {gcut * 100:.1f}%;角速度誤差 RMS "
          f"{info['metrics']['roll']['rate_err_rms_dps']:.0f} vs {ginfo['metrics']['roll']['rate_err_rms_dps']:.0f} deg/s")


def test_vibration():
    ld, info = run("vibration")
    gld, ginfo = run("good")
    v = ld.index.table("VIBE", 0)
    fm = flight_mask(v["t"], info)
    mx, my, mz = (float(np.mean(v[k][fm])) for k in ("VibeX", "VibeY", "VibeZ"))
    assert 30 <= mx <= 50 and 30 <= my <= 50 and 40 <= mz <= 65, (mx, my, mz)
    clip = v["Clip"].astype(np.int64)
    assert clip[-1] > 100 and np.all(np.diff(clip) >= 0), clip[-1]
    gv = gld.index.table("VIBE", 0)
    assert np.max(gv["VibeZ"]) < 20 and int(gv["Clip"][-1]) == 0
    assert info["metrics"]["roll"]["out_hf_rms"] >= 3.0 * ginfo["metrics"]["roll"]["out_hf_rms"]
    print(f"  VIBE X/Y/Z {mx:.0f}/{my:.0f}/{mz:.0f} m/s²、Clip {clip[-1]}(good VibeZ ≤ "
          f"{np.max(gv['VibeZ']):.0f}、Clip 0);控制輸出高頻 "
          f"{info['metrics']['roll']['out_hf_rms'] / ginfo['metrics']['roll']['out_hf_rms']:.1f}× good")


def _batch_spectrum(ld, info, typ=1, axis="x"):
    h = ld.index.table("ISBH")
    d = ld.index.table("ISBD")
    fl = info["windows"]["flight"]
    acc, n, fs = 0.0, 0, None
    for i in np.flatnonzero(h["type"] == typ):
        if not (fl[0] + 3 < h["t"][i] < fl[1] - 3):
            continue
        sel = d["N"] == h["N"][i]
        order = np.argsort(d["seqno"][sel])
        x = d[axis][sel][order].reshape(-1).astype(float) / float(h["mul"][i])
        assert x.size == int(h["smp_cnt"][i])
        fs = float(h["smp_rate"][i])
        acc = acc + np.abs(np.fft.rfft((x - x.mean()) * np.hanning(x.size))) ** 2
        n += 1
    return np.fft.rfftfreq(x.size, 1.0 / fs), acc / n, n


def test_noisy_notch_batch():
    ld, info = run("noisy_notch")
    pv = ld.index.param_values()
    assert pv["INS_LOG_BAT_MASK"] == 1 and pv["INS_HNTCH_ENABLE"] == 0
    assert "INS_HNTCH_FREQ" not in pv          # ENABLE=0 時子參數不記錄(同 ArduPilot)
    h = ld.index.table("ISBH")
    d = ld.index.table("ISBD")
    assert d["x"].shape[1] == 32 and d["x"].dtype == np.int16
    assert set(h["type"].tolist()) == {0, 1} and np.all(h["smp_cnt"] == 1024)
    assert np.all(h["smp_rate"] == np.float32(st.SENSOR_HZ))
    gy = h["mul"][h["type"] == 1]
    assert np.all(gy == int(32767 / math.radians(2000.0))), gy[:3]
    noise_hz = info["noise_hz"]
    for ax in ("x", "y"):
        f, p, n = _batch_spectrum(ld, info, 1, ax)
        m = (f > 40) & (f < 0.45 * st.SENSOR_HZ)
        fpk = float(f[np.argmax(np.where(m, p, 0))])
        assert abs(fpk - noise_hz) <= 0.1 * noise_hz, (ax, fpk, noise_hz)
        m2 = (f > 1.8 * noise_hz) & (f < 2.2 * noise_hz)
        floor = float(np.median(p[m]))
        assert np.max(p[m2]) > 20 * floor                 # 二次諧波也看得到
        print(f"  batch 陀螺儀 {ax} 軸:{n} 批、峰值 {fpk:.0f} Hz(馬達 {noise_hz:.0f} Hz),"
              f"峰 / 底 {np.max(p[m]) / floor:.0f}×,二次諧波 {np.max(p[m2]) / floor:.0f}×")
    # 對照:good 開 batch 時峰值在較低的馬達頻率、振幅小得多
    gld, ginfo = run("good", batch=True)
    gf, gp, _ = _batch_spectrum(gld, ginfo, 1, "x")
    f, p, _ = _batch_spectrum(ld, info, 1, "x")
    assert np.max(p) > 50 * np.max(gp[gf > 40])
    assert "ISBH" not in run("good")[0].index.messages()
    print(f"  good(batch=True)峰值 {gf[np.argmax(np.where(gf > 40, gp, 0))]:.0f} Hz、功率小 "
          f"{np.max(p) / np.max(gp[gf > 40]):.0f}×;預設無 batch 時沒有 ISBH")


def test_units_and_formats():
    ld, info = run("good")
    idx = ld.index
    units = {"ATT.Roll": "deg", "ATT.Yaw": "degheading", "RATE.R": "deg/s", "PIDR.Tar": "rad/s",
             "PIDY.Act": "rad/s", "IMU.GyrX": "rad/s", "IMU.AccZ": "m/s/s", "VIBE.VibeZ": "m/s/s",
             "RCOU.C1": "us", "CTUN.Alt": "m", "CTUN.CRt": "cm/s", "GPS.Lat": "deglatitude"}
    for k, u in units.items():
        assert idx.unit(k) == u, (k, idx.unit(k))
    msgs = idx.messages()
    expect = {
        "ATT": ["DesRoll", "Roll", "DesPitch", "Pitch", "DesYaw", "Yaw", "ErrRP", "ErrYaw", "AEKF"],
        "RATE": ["RDes", "R", "ROut", "PDes", "P", "POut", "YDes", "Y", "YOut", "ADes", "A", "AOut",
                 "AOutSlew"],
        "PIDR": ["Tar", "Act", "Err", "P", "I", "D", "FF", "DFF", "Dmod", "SRate", "Flags"],
        "CTRL": ["RMSRollP", "RMSRollD", "RMSPitchP", "RMSPitchD", "RMSYaw"],
        "CTUN": ["ThI", "ABst", "ThO", "ThH", "DAlt", "Alt", "BAlt", "DSAlt", "SAlt", "TAlt", "DCRt", "CRt"],
        "MOTB": ["LiftMax", "BatVolt", "ThLimit", "ThrAvMx", "ThrOut", "FailFlags"],
        "VIBE": ["IMU", "VibeX", "VibeY", "VibeZ", "Clip"],
        "IMU": ["I", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ", "EG", "EA", "T", "GH", "AH", "GHz", "AHz"],
        "RCOU": [f"C{i}" for i in range(1, 15)],
    }
    for name, flds in expect.items():
        assert msgs[name].fields == flds, (name, msgs[name].fields)
    assert msgs["PIDP"].fields == msgs["PIDY"].fields == expect["PIDR"]
    # 單位一致性:degrees(PIDR.Act) = RATE.R、ROut = P + I + D + FF(同一迴圈樣本)
    r, p = idx.table("RATE"), idx.table("PIDR")
    j = np.searchsorted(p["t"], r["t"])
    j = np.clip(j, 0, p["t"].size - 1)
    ok = np.abs(p["t"][j] - r["t"]) < 0.001
    assert ok.mean() > 0.99
    assert np.max(np.abs(np.degrees(p["Act"][j][ok]) - r["R"][ok])) < 0.01
    tot = (p["P"] + p["I"] + p["D"] + p["FF"])[j][ok]
    assert np.max(np.abs(tot - r["ROut"][ok])) < 1e-5
    # 其他:版本訊息在最前、機型、參數預設值、四軸只有 C1~C4 有輸出
    lm = idx.log_messages()
    assert lm[0].text == "ArduCopter V4.5.7 (sim)" and idx.vehicle == "copter"
    prm = {q.name: q for q in idx.params()}
    assert prm["ATC_RAT_RLL_P"].default == np.float32(0.135) and prm["ATC_RAT_YAW_FLTE"].value == 2.0
    assert prm["FRAME_CLASS"].value == 1 and prm["LOG_DISARMED"].value == 1
    rc = idx.table("RCOU")
    assert rc["C1"].max() > 1500 and rc["C5"].max() == 0 and rc["C14"].max() == 0
    assert rc["C1"].min() == 1000                            # 上鎖時 MOT_PWM_MIN
    print("  欄位 / 單位同 ArduPilot 4.5(PIDx Tar/Act/Err 為 rad/s、RATE 為 deg/s,"
          "已核對 degrees(PIDR.Act)=RATE.R、ROut=P+I+D+FF)")


def test_slow_logging_like_user():
    ld, info = run("good", fast_log=False)
    idx = ld.index
    names = set(idx.messages())
    assert not names & {"PIDR", "PIDP", "PIDY", "ISBH", "ISBD"}, names
    pv = idx.param_values()
    assert pv["LOG_BITMASK"] == st.LOG_BITMASK_USER == 176126 and pv["INS_LOG_BAT_MASK"] == 0
    for name in ("ATT", "RATE", "CTRL", "RCOU", "CTUN", "MOTB"):
        dt = np.diff(idx.table(name)["t"])
        assert abs(np.median(dt) - 0.1) < 0.003, (name, np.median(dt))
        assert np.ptp(dt) > 0.001                        # 排程抖動(±1 個迴圈)
    assert abs(np.median(np.diff(idx.table("IMU", 0)["t"])) - 0.04) < 0.002
    # 取樣抽取(不是平均):10 Hz 的 RATE.R 等於該迴圈樣本的濾波陀螺儀
    tr = info["truth"]
    r = idx.table("RATE")
    k = np.rint((r["t"] - tr["t"][0]) * st.LOOP_HZ).astype(int)
    assert np.max(np.abs(r["R"] - tr["gyro"][k, 0])) < 1e-3
    # 混疊:p_high 的 8 Hz 震盪在 10 Hz log 裡仍然保留變異量(平均的話會被濾掉)
    sl, sinfo = run("p_high", fast_log=False)
    fl, finfo = run("p_high")
    hw = finfo["windows"]["hover"]
    t1, v1 = series(sl, "RATE.R")
    t2, v2 = series(fl, "RATE.R")
    s1, s2 = np.std(v1[mask_windows(t1, hw)]), np.std(v2[mask_windows(t2, hw)])
    assert s1 >= 0.6 * s2, (s1, s2)
    # 10 Hz 時仍可靠的線索:CTRL(迴圈頻率算的 RMS)
    def ctrl(l, i, k):
        t, v = series(l, f"CTRL.{k}")
        return float(np.mean(v[flight_mask(t, i)]))
    dl, dinfo = run("d_high", fast_log=False)
    rp = ctrl(sl, sinfo, "RMSRollP") / ctrl(ld, info, "RMSRollP")
    rd = ctrl(dl, dinfo, "RMSRollD") / ctrl(ld, info, "RMSRollD")
    assert rp >= 3.0 and rd >= 3.0, (rp, rd)
    assert sinfo["slow_log"]["detectable"] == "partial" and info["slow_log"]["detectable"] == "yes"
    assert info["expected_logging"] == [("LOG_BITMASK", DIR_SET, 180223.0), ("INS_LOG_BAT_MASK", DIR_SET, 1.0)]
    # 10 Hz 但開 PID 記錄(ArduPilot 預設 180222)/ 快速記錄的速率與 LOG_BITMASK
    pl, pinfo = run("good", fast_log=False, pid_log=True)
    assert pl.index.param_values()["LOG_BITMASK"] == 180222
    assert abs(np.median(np.diff(pl.index.table("PIDR")["t"])) - 0.1) < 0.003
    _, i100 = run("good")
    assert i100["log"]["LOG_BITMASK"] == 180223 and abs(i100["log"]["rates_hz"]["ATT"] - 100) < 0.5
    assert abs(i100["log"]["rates_hz"]["PIDR"] - 100) < 0.5
    assert i100["params"]["LOG_FILE_RATEMAX"] == 100
    _, i400 = run("good", log_hz=400.0)
    assert abs(i400["log"]["rates_hz"]["RATE"] - 400) < 1 and i400["params"]["LOG_FILE_RATEMAX"] == 0
    assert i100["expected_logging"] == [("INS_LOG_BAT_MASK", DIR_SET, 1.0)]
    print(f"  仿使用者:ATT/RATE/CTRL/RCOU/CTUN/MOTB 10 Hz、IMU 25 Hz、無 PIDx、LOG_BITMASK=176126;"
          f"取樣抽取(p_high 10 Hz 的 RATE.R 標準差 {s1:.1f} vs 100 Hz {s2:.1f} deg/s → 混疊而非平均);"
          f"10 Hz 下 CTRL.RMSRollP {rp:.1f}× / RMSRollD {rd:.1f}× good;pid_log → 180222 / fast → 180223;"
          f"應建議 LOG_BITMASK=180223、INS_LOG_BAT_MASK=1")


def _compare_index(a, b, formats):
    bad = []
    for name, info_m in a.messages().items():
        for inst in (info_m.instances or [None]):
            ta, tb = a.table(name, inst), b.table(name, inst)
            if tb is None:
                bad.append((name, inst, "缺"))
                continue
            for c in ta:
                va, vb = ta[c], tb[c]
                if va.dtype.kind in "US":
                    ok = np.array_equal(va.astype(str), vb.astype(str))
                elif c == "t":
                    ok = np.max(np.abs(va - vb)) < 1.5e-6
                else:
                    x, y = va.astype(float), vb.astype(float)
                    tol = 1e-6 * np.maximum(1.0, np.abs(x)) + (0.0051 if (name, c) in formats else 0.0)
                    ok = bool(np.all((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= tol)))
                if not ok:
                    bad.append((name, inst, c))
                if c != "t" and a.unit(f"{name}.{c}") != b.unit(f"{name}.{c}"):
                    bad.append((name, c, "單位"))
    return bad


def test_bin_roundtrip():
    ld, info = run("noisy_notch", duration_s=60.0, seed=3)
    with tempfile.TemporaryDirectory() as d:
        for label, writer in (("ArduPilot 格式字元", lambda p: st.write_tuning_bin(ld, p)),
                              ("fakebin 預設", lambda p: fakebin.write_logdata_to_bin(ld, p))):
            p = os.path.join(d, "tune.bin")
            writer(p)
            back = logio.load_bin(p)
            bad = _compare_index(ld.index, back.index, st.SIM_FORMATS)
            assert not bad, (label, bad[:10])
            assert [(round(e.t, 4), e.kind, e.label) for e in back.events] == \
                   [(round(e.t, 4), e.kind, e.label) for e in ld.events]
            assert sorted(back.sources) == sorted(ld.sources) == ["GPS", "POS"]
            assert len(back.segments) == 1
            if label.startswith("ArduPilot"):
                bi = back.index
                assert "".join(bi.field_format("ATT", f) for f in bi.messages()["ATT"].fields) == "ccccCCCCB"
                assert "".join(bi.field_format("PIDR", f) for f in bi.messages()["PIDR"].fields) == "ffffffffffB"
                assert "".join(bi.field_format("ISBD", f) for f in bi.messages()["ISBD"].fields) == "HHaaa"
                assert bi.field_format("CTUN", "BAlt") == "e"
            print(f"  {label}:{os.path.getsize(p) / 1e6:.1f} MB,load_bin 讀回全部表格 / 欄位 / 單位 / 事件相同")


def test_logdata_for_app():
    ld, info = run("good")
    assert sorted(ld.sources) == ["GPS", "POS"] and ld.origin is not None
    segs = ld.segments
    assert len(segs) == 1
    tm = info["times"]
    assert segs[0].t0 == tm["arm"] and segs[0].t1 == tm["disarm"]
    assert segs[0].t0 < tm["liftoff"] < tm["touch"] < segs[0].t1
    assert abs(ld.t_base - st.T_BOOT) < 0.01 and ld.t_last >= tm["disarm"] + 4.0
    assert [m.label for m in mode_changes(ld.events)] == ["STABILIZE", "ALT_HOLD"]
    labels = {e.label for e in ld.events if e.kind == "ev"}
    assert {"NOT_LANDED", "LAND_COMPLETE", "AUTO_ARMED"} <= labels, labels
    enu = ld.sources["GPS"].enu(ld.origin)
    assert np.max(np.hypot(enu[:, 0], enu[:, 1])) < 60.0
    fm = (ld.sources["POS"].t > tm["liftoff"] + 8) & (ld.sources["POS"].t < tm["touch"] - 8)
    alt = ld.sources["POS"].enu(ld.origin)[fm, 2]
    assert 3.0 < np.median(alt) < 7.0, np.median(alt)
    assert any("PID 調參模擬" in n for n in ld.notes)
    print(f"  GPS / POS / ORGN、1 個 ARM 段({segs[0].t0:.1f}~{segs[0].t1:.1f} s)、STABILIZE→ALT_HOLD、"
          f"EV 起降事件;離原點最遠 {np.max(np.hypot(enu[:, 0], enu[:, 1])):.1f} m、懸停高度 "
          f"{np.median(alt):.1f} m")


def test_hexa_frame():
    ld, info = run("p_high", frame="hexa")
    rc = ld.index.table("RCOU")
    assert all(rc[f"C{i}"].max() > 1500 for i in range(1, 7)) and rc["C7"].max() == 0
    assert ld.index.param_values()["FRAME_CLASS"] == 2
    assert any(m.text == "Frame: HEXA/X" for m in ld.index.log_messages())
    assert 6.0 <= info["osc_hz"] <= 15.0
    print(f"  六軸 X:RCOU C1~C6 有輸出、FRAME_CLASS=2;p_high 震盪 {info['osc_hz']:.2f} Hz")


def test_corrections_cure_symptoms():
    """套用預期修正(回到預設值 / 開陷波)後重飛,症狀應消失 —— 給調參建議做閉迴路驗證用。"""
    dflt = st.default_params()
    rows = []
    for sc in ("p_high", "d_high", "yaw_p_low", "i_low_bias"):
        _, bad = run(sc, seed=2)
        fix = {p: dflt[p] for p, _ in bad["expected"]}
        _, good = run(sc, seed=2, overrides=fix)
        if sc == "p_high":
            hz = bad["osc_hz"]
            assert bad["osc"]["roll"]["band_rms"] >= 2.5 * good["osc"]["roll"]["band_rms"]
            assert good["step"]["roll"]["overshoot_pct"] <= 20.0
            rows.append(f"p_high {hz:.1f} Hz 頻帶 RMS {bad['osc']['roll']['band_rms']:.1f}→"
                        f"{good['osc']['roll']['band_rms']:.1f}")
        elif sc == "d_high":
            a, b = bad["metrics"]["roll"]["out_hf_rms"], good["metrics"]["roll"]["out_hf_rms"]
            assert a >= 2.5 * b, (a, b)
            rows.append(f"d_high 輸出高頻 {a:.4f}→{b:.4f}")
        elif sc == "yaw_p_low":
            a, b = bad["metrics"]["yaw"]["angle_err_rms_deg"], good["metrics"]["yaw"]["angle_err_rms_deg"]
            assert a >= 2.0 * b, (a, b)
            rows.append(f"yaw_p_low 偏航誤差 {a:.1f}→{b:.1f}°")
        else:
            a, b = bad["metrics"]["roll"]["angle_bias_deg"], good["metrics"]["roll"]["angle_bias_deg"]
            assert abs(a) >= 1.0 and abs(b) < 0.3, (a, b)
            rows.append(f"i_low_bias 偏差 {a:+.2f}→{b:+.2f}°")
    _, nn = run("noisy_notch", seed=2)
    notch = {"INS_HNTCH_ENABLE": 1, "INS_HNTCH_MODE": 1, "INS_HNTCH_FREQ": nn["noise_hz"],
             "INS_HNTCH_REF": nn["params"]["MOT_THST_HOVER"], "INS_HNTCH_BW": nn["noise_hz"] / 2,
             "INS_HNTCH_ATT": 40, "INS_HNTCH_HMNCS": 3}
    nld, nf = run("noisy_notch", seed=2, overrides=notch)
    a, b = nn["metrics"]["roll"]["out_hf_rms"], nf["metrics"]["roll"]["out_hf_rms"]
    assert a >= 2.0 * b, (a, b)
    assert nld.index.param_values()["INS_HNTCH_FREQ"] == nn["noise_hz"]
    rows.append(f"noisy_notch 開陷波 輸出高頻 {a:.4f}→{b:.4f}")
    print("  " + ";".join(rows))


def test_stress_options():
    """壓力測試選項:只懸停、濾波後陀螺儀窄頻噪音(混疊)、掉訊息、MOT_PWM = 0 的舊韌體。"""
    kw = dict(seed=4, duration_s=60.0)
    base, bi = run("good", log_hz=400.0, **kw)
    assert bi["options"] == {"pilot": "normal", "tone": None, "drop_frac": 0.0, "legacy_pwm": False}
    # 只懸停:沒有操作、整段都是懸停窗,期望角速度只剩手抖
    hov, hi_ = run("good", log_hz=400.0, pilot="hover", **kw)
    assert not hi_["manoeuvres"] and len(hi_["windows"]["hover"]) == 1
    t_b, rd_b = series(base, "RATE.RDes")
    fm = flight_mask(t_b, bi)
    t_h, rd_h = series(hov, "RATE.RDes")
    fmh = flight_mask(t_h, hi_)
    assert np.std(rd_h[fmh]) < 0.3 * np.std(rd_b[fm]), (np.std(rd_h[fmh]), np.std(rd_b[fm]))
    # 窄頻噪音 108 Hz、4.5 °/s:400 Hz 記錄看到 108 Hz;100 Hz 記錄混疊到 8 Hz;姿態角沒有對應擺動
    tone = (108.0, 4.5)
    ld4, i4 = run("good", log_hz=400.0, pilot="hover", tone=tone, **kw)
    t, r = series(ld4, "RATE.R")
    f, p, _ = windows_psd(t, r, i4["windows"]["hover"], 400.0, 1024)
    amp = band_rms(f, p, 104, 112)
    assert 3.5 < amp < 5.5, amp
    def band8(ld, info, key):
        t, r = series(ld, key)
        f, p, _ = windows_psd(t, r, info["windows"]["hover"], 100.0, 512)
        return band_rms(f, p, 6.5, 9.5)

    ld1, i1 = run("good", log_hz=100.0, pilot="hover", tone=tone, **kw)
    ld0, i0 = run("good", log_hz=100.0, pilot="hover", **kw)
    a8, a8_0 = band8(ld1, i1, "RATE.R"), band8(ld0, i0, "RATE.R")
    g8, g8_0 = band8(ld1, i1, "ATT.Roll"), band8(ld0, i0, "ATT.Roll")
    assert a8 > 3.0 and a8 > 2.0 * a8_0, (a8, a8_0)          # 角速度在 8 Hz 多了混疊下來的噪音
    assert abs(g8 - g8_0) < 0.2 * g8_0, (g8, g8_0)           # 姿態角沒有(真實運動才有)
    assert any("陀螺儀噪音 108 Hz" in n for n in ld1.notes)
    for bad_kw in (dict(tone=(250.0, 1.0)), dict(pilot="crazy")):
        try:
            st.make_tuning_sim("good", **bad_kw, **kw)
        except ValueError:
            continue
        raise AssertionError(f"{bad_kw} 應該報錯")
    # 掉 1 % 訊息:只影響 ATT / RATE / PIDx
    ldd, _ = run("good", log_hz=400.0, drop_frac=0.01, **kw)
    n0, n1 = base.index.table("RATE")["t"].size, ldd.index.table("RATE")["t"].size
    assert 0.98 * n0 < n1 < 0.995 * n0, (n0, n1)
    assert ldd.index.table("CTRL")["t"].size == base.index.table("CTRL")["t"].size
    # 舊韌體:MOT_PWM_MIN / MAX = 0、RC3_MIN / MAX = 實際範圍;馬達輸出不變
    ldl, _ = run("good", log_hz=400.0, legacy_pwm=True, **kw)
    pv = ldl.index.param_values()
    assert pv["MOT_PWM_MIN"] == 0 and pv["MOT_PWM_MAX"] == 0 and pv["RC3_MIN"] == 1000 and pv["RC3_MAX"] == 2000
    assert np.array_equal(ldl.index.table("RCOU")["C1"], base.index.table("RCOU")["C1"])
    print(f"  只懸停 RDes 標準差 {np.std(rd_h[fmh]):.1f} vs {np.std(rd_b[fm]):.1f};108 Hz 噪音 400 Hz 看到 {amp:.1f} °/s、"
          f"100 Hz 混疊 8 Hz {a8:.1f} vs 無噪音 {a8_0:.1f} °/s(姿態角 {g8:.3f} vs {g8_0:.3f}°,不變);掉訊息 {n0}→{n1} 列;MOT_PWM = 0 + RC3")


def test_determinism_and_runtime():
    a, _ = st.make_tuning_sim("d_high", seed=5, duration_s=50.0)
    b, _ = st.make_tuning_sim("d_high", seed=5, duration_s=50.0)
    c, _ = st.make_tuning_sim("d_high", seed=6, duration_s=50.0)
    for name in ("ATT", "RATE", "PIDR", "RCOU", "VIBE", "GPS", "CTUN"):
        ta, tb = a.index.table(name), b.index.table(name)
        assert all(np.array_equal(ta[k], tb[k], equal_nan=True) for k in ta), name
    assert not np.array_equal(a.index.table("RATE")["R"], c.index.table("RATE")["R"])
    full = {k[0]: v for k, v in _TIMES.items() if k[1] == repr([])}
    for name in st.SCENARIOS:
        if name not in full:
            run(name)
            full[name] = _TIMES[(name, repr([]))]
    slow = max(full.values())
    assert slow <= 3.0, full
    print("  同 seed 逐位元相同、不同 seed 不同;120 s 模擬耗時:"
          + "、".join(f"{k} {v:.2f}s" for k, v in full.items()))


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    t0 = time.perf_counter()
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  !! 失敗:{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 60}\n{len(tests) - failed}/{len(tests)} 通過({time.perf_counter() - t0:.1f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
