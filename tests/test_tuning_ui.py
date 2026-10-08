r"""PID 調參建議面板(ui/tuning.py)測試:離線、不需人工點擊。

只依賴 core.tuning_model:用合成資料做出幾種假報告(良好 / 震盪 / 低取樣率 / 資料不足 / None),
檢查卡片、診斷清單、建議表、訊號(seekRequested / analyzeRequested)、.param 匯出、
TSV / 報告文字、四張圖的繪製(artist 數量、標註文字)、播放游標(set_time 便宜且安全)、
藏在分頁後的延後繪製,並在 1100×700 與 900×650 截圖。

  python tests\test_tuning_ui.py [截圖輸出資料夾]
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32" and os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PyQt6.QtCore import QEventLoop, QPoint, Qt, QTimer  # noqa: E402
from PyQt6.QtGui import QGuiApplication  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QStyleOptionViewItem, QTabWidget  # noqa: E402

from core.tuning_model import (DIR_CHECK, DIR_DOWN, DIR_SET, DIR_UP, SEV_BAD,  # noqa: E402
                               SEV_INFO, SEV_OK, SEV_WARN, AxisAnalysis, Finding, FreqResponse,
                               Spectrum, StepResponse, Suggestion, TuningReport)
from ui import theme  # noqa: E402
from ui import tuning as tui  # noqa: E402

T0 = 5000.0            # log 絕對秒(t_base = T0 - 100)
T_BASE = T0 - 100.0
FS_SIM = 400.0


def _pump(ms: int = 60):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


# =====================================================================
# 合成資料
# =====================================================================

def _welch(x, y, fs, nper):
    """簡易 Welch:回傳 f, Pxx, Pyy, Pxy(單邊)。"""
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    nper = min(nper, x.size)
    step = nper // 2
    w = np.hanning(nper)
    scale = fs * (w ** 2).sum()
    pxx = pyy = 0.0
    pxy = 0.0j
    nseg = 0
    for s in range(0, x.size - nper + 1, step):
        X = np.fft.rfft(w * x[s:s + nper])
        Y = np.fft.rfft(w * y[s:s + nper])
        pxx = pxx + np.abs(X) ** 2
        pyy = pyy + np.abs(Y) ** 2
        pxy = pxy + np.conj(X) * Y
        nseg += 1
    f = np.fft.rfftfreq(nper, 1 / fs)
    k = 2.0 / (nseg * scale)
    return f, pxx * k, pyy * k, pxy * k


def _peaks(f, p, nyq):
    df = f[1] - f[0]
    med = np.median(p[f > 0.5])
    out = []
    for i in range(2, f.size - 2):
        if 1.0 < f[i] < 0.95 * nyq and p[i] > 12 * med and p[i] == p[i - 2:i + 3].max():
            out.append((float(f[i]), float(np.sqrt(p[i - 2:i + 3].sum() * df))))
    return sorted(out, key=lambda q: -q[1])[:3]


def _step(zeta, wn_hz, lag, fs):
    t = np.arange(0, 0.6, 1 / fs)
    wn = 2 * np.pi * wn_hz
    tt = np.clip(t - lag, 0, None)
    wd = wn * np.sqrt(1 - zeta ** 2)
    y = 1 - np.exp(-zeta * wn * tt) * (np.cos(wd * tt) + zeta / np.sqrt(1 - zeta ** 2) * np.sin(wd * tt))
    os_pct = float(max(y.max() - 1.0, 0) * 100)
    t10 = t[np.argmax(y >= 0.1)]
    t90 = t[np.argmax(y >= 0.9)]
    out = np.abs(y - 1) > 0.05
    last = int(np.flatnonzero(out)[-1]) + 1 if out.any() else 0
    settle = float(t[last]) if last < t.size else None
    return StepResponse(t=t, y=y, overshoot_pct=os_pct, rise_s=float(t90 - t10), settle_s=settle,
                        n_windows=14)


def sim_axis(axis, fs, *, dur=180.0, zeta=0.7, wn_hz=6.0, lag=0.02, osc_hz=None, osc_amp=0.0,
             noise=0.4, amp=60.0, seed=0, gap=(70.0, 95.0), angle=True, bias=0.0) -> AxisAnalysis:
    rng = np.random.default_rng(seed)
    n = int(dur * FS_SIM)
    ts = T0 + np.arange(n) / FS_SIM
    des = np.zeros(n)
    i = 0
    while i < n:
        L = int(rng.uniform(0.4, 1.6) * FS_SIM)
        des[i:i + L] = rng.choice([-1, 0, 1]) * rng.uniform(0.3, 1.0) * amp
        i += L
    a = 1 - np.exp(-1 / (FS_SIM * 0.06))
    for k in range(1, n):                       # 一階平滑(像 ATC_INPUT_TC)
        des[k] = des[k - 1] + a * (des[k] - des[k - 1])
    u = np.interp(ts - lag, ts, des)
    wn = 2 * np.pi * wn_hz
    dt = 1 / FS_SIM
    x = np.zeros(n)
    v = 0.0
    for k in range(1, n):
        acc = wn * wn * (u[k - 1] - x[k - 1]) - 2 * zeta * wn * v
        v += acc * dt
        x[k] = x[k - 1] + v * dt
    act = x + bias + noise * rng.standard_normal(n)
    if osc_hz:
        env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * (ts - T0))
        act += osc_amp * env * np.sin(2 * np.pi * osc_hz * (ts - T0))
    k = int(round(FS_SIM / fs))
    t, d, y = ts[::k].copy(), des[::k].copy(), act[::k].copy()

    # 頻譜 / 頻率響應 / 步階(用飛行段 1 計算)
    seg = (t - T0) < gap[0]
    nper = 256 if fs >= 100 else 64
    f, pdd, pyy, pdy = _welch(d[seg], y[seg], fs, nper)
    _, pee, _, _ = _welch(y[seg] - d[seg], y[seg] - d[seg], fs, nper)
    nyq = fs / 2
    peaks = _peaks(f, pee, nyq)
    sp = Spectrum(f=f, psd_err=pee, psd_act=pyy, peaks=peaks, nyquist=nyq)
    H = pdy / np.maximum(pdd, 1e-12)
    coh = np.abs(pdy) ** 2 / np.maximum(pdd * pyy, 1e-12)
    mag = np.abs(H)
    ph = np.degrees(np.unwrap(np.angle(H)))
    m = (f > 0.3) & (mag < 0.707)
    bw = float(f[np.argmax(m)]) if m.any() else None
    j = int(np.argmax(np.where((f > 0.2) & (coh > 0.6) & (f < 0.5 * nyq), mag, 0)))
    fr = FreqResponse(f=f, mag=mag, phase_deg=ph, coherence=np.clip(coh, 0, 1), bandwidth_hz=bw,
                      peak_mag=float(mag[j]), peak_hz=float(f[j]))
    st = _step(zeta, wn_hz, lag, fs)

    # 飛行段之間的空洞:刪掉樣本並插入 NaN 斷點
    gm = ((t - T0) > gap[0]) & ((t - T0) < gap[1])
    cut = int(np.argmax(gm))
    keep = ~gm
    t, d, y = t[keep], d[keep], y[keep]
    t = np.insert(t, cut, np.nan)
    d = np.insert(d, cut, np.nan)
    y = np.insert(y, cut, np.nan)

    err = y - d
    metrics = dict(rms_err=float(np.sqrt(np.nanmean(err ** 2))), max_err=float(np.nanmax(np.abs(err))),
                   bias=float(np.nanmean(err)), lag_ms=float(lag * 1000 + 1000 / (2 * np.pi * wn_hz)),
                   overshoot_pct=st.overshoot_pct, bandwidth_hz=bw)
    if peaks:
        metrics["osc_hz"] = peaks[0][0]
    aa = AxisAnalysis(axis=axis, level="rate", sample_hz=float(fs), t=t, des=d, act=y,
                      metrics=metrics, spectrum=sp, freq=fr, step=st)
    if angle:
        ka = max(1, int(round(fs / 10)))
        ta = t[::ka]
        good = np.isfinite(ta)
        tg = ta[good]
        ang_des = np.cumsum(np.nan_to_num(d[::ka][good])) * (ka / fs) * 0.25
        ang_act = np.interp(tg - 0.08, tg, ang_des) + 0.3 * rng.standard_normal(tg.size)
        aa.t_ang = np.insert(tg, int(np.argmax(np.diff(tg) > 5)) + 1, np.nan)
        aa.des_ang = np.insert(ang_des, int(np.argmax(np.diff(tg) > 5)) + 1, np.nan)
        aa.act_ang = np.insert(ang_act, int(np.argmax(np.diff(tg) > 5)) + 1, np.nan)
    return aa


PARAMS = {"ATC_RAT_RLL_P": 0.135, "ATC_RAT_RLL_I": 0.135, "ATC_RAT_RLL_D": 0.0036,
          "ATC_RAT_PIT_P": 0.135, "ATC_RAT_PIT_I": 0.135, "ATC_RAT_PIT_D": 0.0036,
          "ATC_RAT_YAW_P": 0.18, "ATC_RAT_YAW_I": 0.018, "ATC_ANG_RLL_P": 4.5,
          "ATC_ANG_PIT_P": 4.5, "ATC_INPUT_TC": 0.15, "LOG_BITMASK": 176126.0,
          "INS_LOG_BAT_MASK": 0.0, "MOT_THST_HOVER": 0.25}


def report_good() -> TuningReport:
    axes = {"roll": sim_axis("roll", 400, zeta=0.75, wn_hz=7, seed=1),
            "pitch": sim_axis("pitch", 400, zeta=0.72, wn_hz=6.5, seed=2),
            "yaw": sim_axis("yaw", 400, zeta=0.9, wn_hz=2.2, lag=0.04, amp=40, seed=3, angle=False)}
    for ax in ("roll", "pitch"):
        axes[ax].summary = "追蹤良好:延遲短、幾乎沒有過衝,不需要調整。"
    axes["yaw"].verdict = SEV_INFO
    axes["yaw"].summary = "偏航反應略慢(頻寬約 2 Hz),可小幅調高 ATC_RAT_YAW_P。"
    findings = [
        Finding("yaw_slow", SEV_INFO, "yaw", "Yaw 反應略慢",
                "偏航角速度頻寬約 2.1 Hz、延遲約 110 ms;期望變化快時實際值跟不上。",
                [Suggestion("ATC_RAT_YAW_P", DIR_UP, "提高偏航角速度環比例增益,加快反應",
                            current=0.18, proposed=0.2, change_pct=11, priority=2)],
                t_focus=T0 + 40.0, window=(T0 + 38.0, T0 + 44.0)),
        Finding("track_ok", SEV_OK, "roll", "Roll 追蹤良好", "RMS 誤差 2.1 °/s,過衝 3 %。",
                t_focus=T0 + 20.0),
        Finding("track_ok", SEV_OK, "pitch", "Pitch 追蹤良好", "RMS 誤差 2.3 °/s,過衝 4 %。"),
    ]
    return TuningReport(t0=T0, t1=T0 + 180, flight_s=155.0, vehicle="copter", frame="quad X",
                        param_prefix="ATC_",
                        data={"RATE_hz": 400.0, "ATT_hz": 400.0, "PID": True, "PID_hz": 400.0,
                              "fast_attitude": True, "batch": True},
                        axes=axes, findings=findings, params=dict(PARAMS, LOG_BITMASK=180223.0),
                        notes=["分析了 2 段飛行(共 155 s)。"])


def report_osc() -> TuningReport:
    axes = {"roll": sim_axis("roll", 400, zeta=0.32, wn_hz=8, osc_hz=14.0, osc_amp=9.0, seed=4),
            "pitch": sim_axis("pitch", 400, zeta=0.5, wn_hz=7, seed=5),
            "yaw": sim_axis("yaw", 400, zeta=0.85, wn_hz=3, lag=0.03, amp=40, seed=6, angle=False)}
    axes["roll"].verdict = SEV_BAD
    axes["roll"].summary = "角速度環 14 Hz 持續震盪、過衝約 35 %:先降 D,再降 P。"
    axes["pitch"].verdict = SEV_WARN
    axes["pitch"].summary = "過衝約 16 %,阻尼偏低;可小幅降 P 或提高 D。"
    axes["yaw"].summary = "追蹤正常。"
    findings = [
        Finding("rate_osc_mid", SEV_BAD, "roll", "Roll 角速度環 14 Hz 震盪",
                "誤差頻譜在 14.0 Hz 有明顯尖峰(RMS 約 6 °/s),整段飛行都存在;"
                "這個頻段的持續震盪多半是 D(或 P)增益過高。",
                [Suggestion("ATC_RAT_RLL_D", DIR_DOWN, "D 過高會放大高頻、造成 10–20 Hz 震盪",
                            current=0.0036, proposed=0.0029, change_pct=-20, priority=1),
                 Suggestion("ATC_RAT_RLL_P", DIR_DOWN, "降 D 後仍震盪再降 P;P 與 I 一起調",
                            current=0.135, proposed=0.115, change_pct=-15, priority=2)],
                t_focus=T0 + 31.5, window=(T0 + 29.0, T0 + 34.0)),
        Finding("overshoot", SEV_WARN, "roll", "Roll 步階響應過衝 35 %",
                "由 14 個時間窗反卷積:過衝約 35 %,上升時間 40 ms,約 0.3 s 才穩定。",
                [Suggestion("ATC_RAT_RLL_P", DIR_DOWN, "過衝大、阻尼低", current=0.135,
                            proposed=0.115, change_pct=-15, priority=2)],
                t_focus=T0 + 52.0),
        Finding("overshoot", SEV_WARN, "pitch", "Pitch 過衝 16 %",
                "阻尼偏低,步階響應過衝約 16 %。",
                [Suggestion("ATC_RAT_PIT_P", DIR_DOWN, "小幅降低 P 增加阻尼", current=0.135,
                            proposed=0.12, change_pct=-11, priority=3)],
                t_focus=T0 + 60.0),
        Finding("vibration", SEV_INFO, None, "高頻振動偏高(建議檢查機構)",
                "陀螺儀在 120 Hz 附近有能量,可能來自螺旋槳不平衡;調 PID 前先處理。",
                [Suggestion("螺旋槳平衡 / 馬達座鎖緊", DIR_CHECK, "機構振動會讓 D 項放大雜訊",
                            priority=1),
                 Suggestion("INS_HNTCH_ENABLE", DIR_SET, "啟用油門追蹤陷波濾波器", current=0,
                            proposed=1, priority=3)]),
        Finding("track_ok", SEV_OK, "yaw", "Yaw 追蹤正常", "RMS 誤差 3 °/s。"),
    ]
    return TuningReport(t0=T0, t1=T0 + 180, flight_s=155.0, vehicle="copter", frame="hexa X",
                        param_prefix="ATC_",
                        data={"RATE_hz": 400.0, "PID": True, "PID_hz": 400.0, "fast_attitude": True,
                              "batch": False},
                        axes=axes, findings=findings, params=dict(PARAMS, LOG_BITMASK=180223.0),
                        notes=["建議依序:先處理振動 → 降 Roll D → 試飛 → 再看是否要降 P。",
                               "Roll 的 P 與 I 通常保持相同。"])


def report_lowrate() -> TuningReport:
    axes = {"roll": sim_axis("roll", 10, zeta=0.6, wn_hz=3, lag=0.08, seed=7),
            "pitch": sim_axis("pitch", 10, zeta=0.6, wn_hz=3.2, lag=0.08, seed=8),
            "yaw": sim_axis("yaw", 10, zeta=0.9, wn_hz=1.5, lag=0.06, amp=40, seed=9, angle=False,
                            bias=2.5)}
    for ax in ("roll", "pitch"):
        axes[ax].verdict = SEV_WARN
        axes[ax].summary = "延遲約 130 ms(10 Hz 資料,震盪無法判斷);外環略慢。"
    axes["yaw"].verdict = SEV_INFO
    axes["yaw"].summary = "穩態偏差 +2.5 °/s,I 可能偏低。"
    findings = [
        Finding("logging_low_rate", SEV_WARN, None, "記錄取樣率太低:無法判斷 PID 震盪",
                "RATE / ATT 只有 10 Hz(Nyquist 5 Hz),常見的 5–20 Hz PID 震盪看不到,"
                "甚至會混疊成假的低頻;沒有 PIDR/PIDP/PIDY,無法分辨 P / D 的貢獻。",
                [Suggestion("LOG_BITMASK", DIR_SET, "加上 ATTITUDE_FAST(bit 0)與 PID(bit 12)",
                            current=176126, proposed=180223, priority=1),
                 Suggestion("INS_LOG_BAT_MASK", DIR_SET, "開啟陀螺儀批次取樣(FFT)",
                            current=0, proposed=1, priority=1)]),
        Finding("angle_lag", SEV_WARN, "roll", "Roll 角度追蹤延遲 130 ms",
                "期望角與實際角之間延遲約 130 ms(互相關),RMS 誤差 4.2°。",
                [Suggestion("ATC_ANG_RLL_P", DIR_UP, "外環略慢;10 Hz 資料下保守地小幅調高",
                            current=4.5, proposed=5.0, change_pct=11, priority=2)],
                t_focus=T0 + 45.0, window=(T0 + 42.0, T0 + 50.0)),
        Finding("angle_lag", SEV_WARN, "pitch", "Pitch 角度追蹤延遲 125 ms",
                "期望角與實際角之間延遲約 125 ms。",
                [Suggestion("ATC_ANG_PIT_P", DIR_UP, "外環略慢", current=4.5, proposed=5.0,
                            change_pct=11, priority=2)],
                t_focus=T0 + 47.0),
        Finding("bias", SEV_INFO, "yaw", "Yaw 穩態偏差 +2.5 °/s",
                "實際偏航角速度長時間偏離期望約 +2.5 °/s,可能是 I 偏低或馬達 / 機架扭轉。",
                [Suggestion("ATC_RAT_YAW_I", DIR_UP, "提高 I 消除穩態偏差", current=0.018,
                            proposed=0.022, change_pct=22, priority=3),
                 Suggestion("馬達是否歪斜 / 機架扭轉", DIR_CHECK, "偏航偏差常見的機構原因", priority=3)],
                t_focus=T0 + 120.0),
    ]
    return TuningReport(t0=T0, t1=T0 + 180, flight_s=155.0, vehicle="copter", frame="hexa X",
                        param_prefix="ATC_",
                        data={"RATE_hz": 10.0, "ATT_hz": 10.0, "PID": False, "fast_attitude": False,
                              "batch": False},
                        axes=axes, findings=findings, params=dict(PARAMS),
                        notes=["10 Hz 取樣:Nyquist 5 Hz,只能判斷慢的追蹤問題(延遲、偏差、外環)。",
                               "改 LOG_BITMASK 後重新飛一次,就能分析震盪與 P / D 的貢獻。"])


def report_insufficient() -> TuningReport:
    e = np.empty(0)
    axes = {"roll": AxisAnalysis(axis="roll", level="rate", sample_hz=10.0, t=e, des=e, act=e,
                                 verdict=SEV_INFO, summary="飛行時間太短(3 s),無法分析。"),
            "yaw": AxisAnalysis(axis="yaw", level="angle", sample_hz=10.0,
                                t=T0 + np.arange(30) / 10.0, des=np.zeros(30), act=np.ones(30) * 0.4,
                                summary="只有 3 s 資料。")}
    findings = [Finding("insufficient_data", SEV_INFO, None, "資料不足",
                        "分析範圍內只有 3 s 的飛行資料,至少需要 20 s 以上、有操控輸入的飛行。")]
    return TuningReport(t0=T0, t1=T0 + 3, flight_s=3.0, vehicle="copter", frame="quad X",
                        param_prefix="ATC_", data={"RATE_hz": 10.0, "PID": False}, axes=axes,
                        findings=findings, params={}, notes=[])


def report_quadplane() -> TuningReport:
    r = report_good()
    r.param_prefix = "Q_A_"
    r.vehicle = "plane"
    return r


# =====================================================================
# 測試
# =====================================================================

def _texts(plot) -> list[str]:
    out = []
    for ax in plot.fig.axes:
        out += [t.get_text() for t in ax.texts]
        out.append(ax.get_title(loc="left"))
        out.append(ax.get_title())
    return out


def _labels(widget) -> list[str]:
    return [w.text() for w in widget.findChildren(QLabel)]


def _canvas_point(plot, ax, x, y) -> QPoint:
    xd, yd = ax.transData.transform((x, y))
    dpr = plot.canvas.device_pixel_ratio or 1.0
    return QPoint(int(round(xd / dpr)), int(round(plot.canvas.height() - yd / dpr)))


def main(outdir: str | None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)
    failed = 0

    def check(label, cond, extra=""):
        nonlocal failed
        print(f"  [{'OK' if cond else 'FAIL'}]  {label} {extra}")
        if not cond:
            failed += 1

    shots = outdir or tempfile.mkdtemp(prefix="tuning_shots_")
    os.makedirs(shots, exist_ok=True)

    def shot(w, name, size):
        w.resize(*size)
        _pump(80)
        for p in w._plots():
            if p.isVisible():
                p.flush_draw()
        _pump(80)
        path = os.path.join(shots, f"{name}_{size[0]}x{size[1]}.png")
        w.grab().save(path)
        return path

    print("[純函式]")
    check("fmt_param 整數", tui.fmt_param(180223.0) == "180223")
    check("fmt_param 小數", tui.fmt_param(0.0036) == "0.0036" and tui.fmt_param(0.135) == "0.135")
    check("fmt_param None", tui.fmt_param(None) == "—")
    check("fmt_pct", tui.fmt_pct(-20) == "-20 %" and tui.fmt_pct(11.4) == "+11 %"
          and tui.fmt_pct(None) == "—")
    x = np.arange(100000, dtype=float)
    y = np.sin(x * 0.7)
    x[50000] = np.nan
    xs, ys = tui._envelope(x, y, 4000)
    check("包絡抽稀點數", xs.size <= 4100, f"({xs.size})")
    check("包絡保留斷點", np.isnan(xs).any())
    check("包絡保留振幅", np.nanmax(ys) > 0.99 and np.nanmin(ys) < -0.99)
    ticks = tui._ClockLocator().tick_values(3.0, 289.0)
    check("時間刻度落在整分鐘", len(ticks) >= 3 and all(abs(v % 60) < 1e-6 for v in ticks), str(ticks))
    check("_hz", tui._hz(None) == "" and tui._hz(10.0) == " 10 Hz")
    page = tui.chart_page_for
    check("chart_page_for", page(Finding("rate_osc_mid", SEV_BAD, "roll", "", "")) == tui.PAGE_SPEC
          and page(Finding("overshoot", SEV_WARN, "roll", "", "")) == tui.PAGE_STEP
          and page(Finding("angle_lag", SEV_WARN, "roll", "", "")) == tui.PAGE_TRACK
          and page(Finding("gyro_noise", SEV_WARN, "roll", "", "")) == tui.PAGE_SPEC)
    # 引擎明確寫 osc_hz = None(峰可能是噪音混疊)時,卡片不能拿頻譜峰補成「震盪 X Hz」
    aa_n = sim_axis("roll", 400, osc_hz=14.0, osc_amp=9.0, seed=4)
    aa_n.metrics["osc_hz"] = None
    aa_n.metrics["noise_hz"] = 108.0
    osc_row = next(r for r in tui.axis_metrics(aa_n) if r[0] == "震盪")
    check("osc_hz = None 不用頻譜峰補", osc_row[1] == "—" and "噪音" in osc_row[3], str(osc_row))
    # QuadPlane 的記錄建議:ATTITUDE_FULLRATE bit 20,不是 Copter 的 bit 0 / 12
    rq = report_quadplane()
    rq.params = dict(rq.params or {}, LOG_BITMASK=65470.0)
    rq.data = dict(rq.data, RATE_hz=10.0, ATT_hz=10.0, PID=False, PID_hz=None)
    tips = " ".join(c[2] for c in tui.data_chips(rq))
    check("QuadPlane 記錄建議 bit 20", "bit 20" in tips and "bit 12" not in tips and str(65470 | (1 << 20)) in tips,
          tips[:200])

    print("[None / 忙碌]")
    w = tui.TuningPanel()
    w.resize(1100, 700)
    w.show()
    _pump(150)
    check("初始為訊息頁", w.stack.currentIndex() == 0)
    w.set_report(None, message="載入 log 後分析")
    check("訊息文字", w.lbl_msg.text() == "載入 log 後分析")
    check("無報告時按鈕停用", not w.btn_export.isEnabled() and not w.btn_copy.isEnabled())
    check("無報告時資料標籤隱藏", not w.row_chips.isVisible())
    shot(w, "none", (1100, 700))
    w.set_busy(True)
    check("忙碌:按鈕停用、顯示分析中", not w.btn_analyze.isEnabled() and w.lbl_msg.text() == "分析中…"
          and w.busy_bar.isVisible())
    w.set_busy(False)
    check("忙碌結束:訊息還原", w.btn_analyze.isEnabled() and w.lbl_msg.text() == "載入 log 後分析")
    w.set_report(None, message="這份 log 是定翼機:PID 調參建議只支援多旋翼(Copter / QuadPlane)")
    check("錯誤訊息", "定翼機" in w.lbl_msg.text())

    hits = []
    w.analyzeRequested.connect(lambda: hits.append(1))
    QTest.mouseClick(w.btn_analyze, Qt.MouseButton.LeftButton)
    check("analyzeRequested", hits == [1])

    print("[震盪報告]")
    seeks: list[float] = []
    w.seekRequested.connect(seeks.append)
    rep = report_osc()
    t_rep = time.perf_counter()
    w.set_report(rep, t_base=T_BASE, scope_text="分析範圍:ARM 1 · 1:40–4:40 · 飛行 155 s")
    _pump(100)
    print(f"    set_report {1000 * (time.perf_counter() - t_rep):.0f} ms")
    check("主頁", w.stack.currentIndex() == 1)
    check("範圍文字", "ARM 1" in w.lbl_scope.full_text())
    check("選最嚴重的軸", w.selected_axis() == "roll")
    check("Roll 卡片選取", w.cards["roll"].is_selected() and not w.cards["pitch"].is_selected())
    check("Roll 卡片結論", any("需要處理" in t for t in _labels(w.cards["roll"])))
    check("Roll 卡片震盪 Hz", any(m.startswith("震盪 14") for m in w.cards["roll"].metric_texts()),
          str(w.cards["roll"].metric_texts()))
    check("整體結論標籤", any("整體:需要處理" in t for t in _labels(w)))
    chips = [t for t in _labels(w.row_chips)]
    check("資料標籤 PID 記錄 400 Hz", any(c == "PID 記錄 400 Hz" for c in chips), str(chips))
    check("資料標籤 無 FFT", any("無 FFT 批次資料" in c for c in chips))
    check("診斷卡片數", len(w.findings.cards) == len(rep.findings))
    check("嚴重的診斷預設展開", w.findings.cards[0].is_expanded()
          and not w.findings.cards[-1].is_expanded())
    sugg = rep.suggestions()
    check("建議表列數", w.table.rowCount() == len(sugg), f"({w.table.rowCount()})")
    row0 = [w.table.item(0, c).text() for c in range(w.table.columnCount())]
    check("建議表第一列", row0[1] in ("ATC_RAT_RLL_D", "螺旋槳平衡 / 馬達座鎖緊"), str(row0))
    rows = {w.table.item(r, 1).text(): [w.table.item(r, c).text() for c in range(7)]
            for r in range(w.table.rowCount())}
    d = rows.get("ATC_RAT_RLL_D")
    check("D 列內容", d is not None and d[2] == "0.0036" and d[3] == "↓ 調低" and d[4] == "0.0029"
          and d[5] == "-20 %", str(d))
    chk = rows.get("螺旋槳平衡 / 馬達座鎖緊")
    check("檢查列", chk is not None and chk[3] == "◎ 檢查" and chk[4] == "—")
    check("預設分頁 = 建議調整", w.tabs.currentIndex() == tui.PAGE_TABLE)
    check("建議表分頁標題含數量", f"({len(sugg)})" in w.tabs.tabText(tui.PAGE_TABLE))
    check("匯出按鈕可用", w.btn_export.isEnabled())
    shot(w, "osc_table", (1100, 700))
    shot(w, "osc_table", (900, 650))
    check("窄:建議表精簡版(原因併入參數欄)", w.table.compact and w.table.isColumnHidden(tui.SuggestionTable.C_REASON))
    w.resize(1100, 700)
    _pump(120)
    check("寬:建議表完整版", not w.table.compact and not w.table.isColumnHidden(tui.SuggestionTable.C_REASON))
    dlg = w.table.itemDelegateForColumn(tui.SuggestionTable.C_REASON)
    txt = "加上 ATTITUDE_FAST(bit 0)與 PID(bit 12)"
    h_narrow = dlg._doc(txt, False, w.table.font(), 60).size().height()
    h_wide = dlg._doc(txt, False, w.table.font(), 400).size().height()
    check("原因欄窄時換行(長英文字也會斷行)", h_narrow > 2 * h_wide, f"({h_narrow:.0f} > 2×{h_wide:.0f})")
    opt = QStyleOptionViewItem()
    opt.font = w.table.font()
    hint = dlg.sizeHint(opt, w.table.model().index(0, tui.SuggestionTable.C_REASON))
    check("原因欄 sizeHint", hint.height() >= h_wide, f"({hint.height()})")
    w.resize(1100, 700)
    _pump(120)

    print("[TSV / 報告 / .param]")
    tsv = w.copy_tsv()
    lines = tsv.strip().split("\n")
    check("TSV 列數", len(lines) == len(sugg) + 1)
    check("TSV 欄位", lines[0].split("\t") == tui._COLS)
    check("剪貼簿 = TSV", QGuiApplication.clipboard().text() == tsv)
    txt = w.copy_report()
    check("報告文字", "ATC_RAT_RLL_D" in txt and ("Roll" in txt) and len(txt) > 200)
    check("剪貼簿 = 報告", QGuiApplication.clipboard().text() == txt)
    esc = tui.FindingCard._sugg_html(Suggestion("A<B", DIR_UP, "相干性 < 0.5 & 低", current=1, proposed=2,
                                                change_pct=10), True)
    check("建議文字 HTML 跳脫", "A&lt;B" in esc and "&lt; 0.5 &amp;" in esc)
    fb = tui.build_report_text(rep, T_BASE, "")
    check("備用報告含安全須知與建議", "安全須知" in fb and "0.0036 → 0.0029" in fb and "建議調整" in fb)
    tmp = tempfile.mkdtemp()
    pth = os.path.join(tmp, "s.param")
    n = w.export_param(pth)
    with open(pth, encoding="utf-8") as fh:
        content = fh.read()
    check(".param 內容 = report.param_file()", content == rep.param_file())
    check(".param 筆數", n == len([s for s in sugg if s.direction != DIR_CHECK and s.proposed is not None]),
          f"({n})")
    check(".param 不含 CHECK", "螺旋槳" not in content and "ATC_RAT_RLL_D,0.0029" in content)

    print("[訊號]")
    seeks.clear()
    card0 = w.findings.cards[0]
    card0.btn_seek.click()
    check("「跳到」→ seekRequested(t_focus)", seeks == [rep.findings[0].t_focus], str(seeks))
    seeks.clear()
    QTest.mouseDClick(card0.title, Qt.MouseButton.LeftButton)
    _pump(30)
    check("雙擊卡片 → seekRequested", seeks and abs(seeks[-1] - rep.findings[0].t_focus) < 1e-9, str(seeks))

    # 點診斷 → 切到對應圖 + 標出範圍
    QTest.mouseClick(card0.title, Qt.MouseButton.LeftButton)
    _pump(60)
    check("點震盪診斷 → 頻譜分頁", w.tabs.currentIndex() == tui.PAGE_SPEC)
    check("卡片選取", card0._sel)
    w.p_track.flush_draw()
    check("追蹤圖標出所選診斷", any("所選診斷" in s for s in _texts(w.p_track)))
    w.p_spec.flush_draw()
    st = _texts(w.p_spec)
    check("頻譜標出 14 Hz 峰值", any(s.startswith("14.") and " Hz" in s for s in st),
          str([s for s in st if "Hz" in s][:4]))
    check("頻譜 Nyquist 200 Hz", any("Nyquist 200 Hz" in s for s in st))
    check("高取樣率不畫無法判斷區", not any("無法判斷" in s for s in st))
    shot(w, "osc_spectrum", (1100, 700))

    # 過衝診斷 → 步階分頁
    QTest.mouseClick(w.findings.cards[1].title, Qt.MouseButton.LeftButton)
    _pump(60)
    check("點過衝診斷 → 步階分頁", w.tabs.currentIndex() == tui.PAGE_STEP)
    w.p_step.flush_draw()
    stt = _texts(w.p_step)
    check("步階標出過衝", any(s.startswith("過衝 3") for s in stt), str(stt))
    check("步階標出上升時間", any(s.startswith("上升") for s in stt))
    check("步階有 1.0 參考線與響應", len(w.p_step.fig.axes[0].lines) >= 3)
    shot(w, "osc_step", (1100, 700))

    # 點 Pitch 診斷 → 切軸
    QTest.mouseClick(w.findings.cards[2].title, Qt.MouseButton.LeftButton)
    _pump(60)
    check("點 Pitch 診斷 → 選 Pitch", w.selected_axis() == "pitch" and w.cards["pitch"].is_selected())

    # 點卡片選軸
    QTest.mouseClick(w.cards["yaw"], Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    _pump(60)
    check("點 Yaw 卡片 → 選 Yaw", w.selected_axis() == "yaw" and w.seg_btns["yaw"].isChecked())
    w.seg_btns["roll"].click()
    _pump(60)
    check("分段按鈕選 Roll", w.selected_axis() == "roll" and w.cards["roll"].is_selected())

    print("[追蹤圖 / 播放游標]")
    w.tabs.setCurrentIndex(tui.PAGE_TRACK)
    _pump(150)
    p = w.p_track
    p.flush_draw()
    check("追蹤圖 3 個子圖(角速度 / 誤差 / 角度)", len(p.fig.axes) == 3, f"({len(p.fig.axes)})")
    check("追蹤圖軸為 Roll", p.axis_name == "roll" and "Roll" in p.fig.axes[0].get_title(loc="left"))
    nl = sum(len(a.lines) for a in p.fig.axes)
    check("追蹤圖線條數", nl >= 5 + len(p._anim) - 1, f"({nl})")
    check("包絡抽稀:每條線 ≤ 5100 點", all(ln.get_xdata().size <= 5100 for ln, *_ in p._lines),
          str([ln.get_xdata().size for ln, *_ in p._lines]))
    check("x 軸格式 m:ss", ":" in p.fig.axes[-1].xaxis.get_major_formatter()(60.0))
    # 放大後重新取樣
    p.fig.axes[0].set_xlim(10.0 + 100, 12.0 + 100)
    xs0 = p._lines[1][0].get_xdata()
    check("放大後看得到原始取樣", 700 <= xs0.size <= 900, f"({xs0.size})")
    p.fig.axes[0].set_xlim(*p.fig.axes[0].dataLim.intervalx)
    _pump(60)

    t_mid = T0 + 30.0
    w.set_time(t_mid)
    _pump(60)
    vis = [a for a in p._anim if a.get_visible()]
    check("游標可見(3 條線 + 4 點 + 讀數)", len(vis) == 8, f"({len(vis)})")
    check("讀數文字", "期望" in p._lbl.get_text() and "實際" in p._lbl.get_text())
    w.set_time(T0 + 82.0)            # 空洞內
    check("空洞內:點隱藏、線保留", not p._dots[0].get_visible() and p._vlines[0].get_visible())
    w.set_time(T0 - 50.0)
    check("範圍外:游標全隱藏", not any(a.get_visible() for a in p._anim))
    w.set_time(None)
    check("None:游標全隱藏", not any(a.get_visible() for a in p._anim))
    _pump(80)
    t1 = time.perf_counter()
    N = 300
    for i in range(N):
        w.set_time(T0 + 5 + i * 0.1)
    dt_ms = (time.perf_counter() - t1) * 1000 / N
    check("set_time 便宜(blit)", dt_ms < 15, f"({dt_ms:.2f} ms/次)")
    w.set_time(t_mid)
    shot(w, "osc_track", (1100, 700))
    shot(w, "osc_track", (900, 650))

    # 點追蹤圖 → seekRequested
    seeks.clear()
    _pump(60)
    ax0 = p.fig.axes[0]
    xv = 40.0 + 100.0
    pt = _canvas_point(p, ax0, xv, float(np.mean(ax0.get_ylim())))
    QTest.mouseClick(p.canvas, Qt.MouseButton.LeftButton, pos=pt)
    _pump(60)
    check("點追蹤圖 → seekRequested(log 絕對秒)", seeks and abs(seeks[-1] - (xv + T_BASE)) < 1.0,
          str(seeks))

    # 頻率響應
    w.tabs.setCurrentIndex(tui.PAGE_FREQ)
    _pump(120)
    fq = w.p_freq
    fq.flush_draw()
    ft = _texts(fq)
    check("頻率響應:-3 dB / 頻寬 / 共振峰", any(s == "-3 dB" for s in ft) and any(s.startswith("頻寬") for s in ft)
          and any(s.startswith("共振峰") for s in ft), str([s for s in ft if s][:8]))
    check("頻率響應:相位子圖 + 相干性雙軸", len(fq.fig.axes) == 3)
    shot(w, "osc_freq", (1100, 700))

    print("[低取樣率報告]")
    rl = report_lowrate()
    w.set_report(rl, t_base=T_BASE)
    _pump(80)
    check("預設範圍文字", w.lbl_scope.full_text().startswith("分析範圍:1:40–4:40"), w.lbl_scope.full_text())
    chips = _labels(w.row_chips)
    check("資料標籤 RATE 10 Hz / 無 PID / 無 FFT",
          any(c == "RATE 10 Hz" for c in chips) and any(c == "無 PID 記錄" for c in chips)
          and any(c == "無 FFT 批次資料" for c in chips), str(chips))
    tip = next(c for c in w.row_chips.findChildren(QLabel) if c.text() == "RATE 10 Hz").toolTip()
    check("RATE 標籤說明含 LOG_BITMASK 建議值", "180223" in tip, tip.replace("\n", " / "))
    rows = {w.table.item(r, 1).text(): [w.table.item(r, c).text() for c in range(7)]
            for r in range(w.table.rowCount())}
    lb = rows.get("LOG_BITMASK")
    check("LOG_BITMASK 列", lb is not None and lb[2] == "176126" and lb[4] == "180223" and lb[3] == "→ 設定",
          str(lb))
    check("Roll 震盪欄 = 只見 ≤5 Hz", any("≤5 Hz" in m for m in w.cards["roll"].metric_texts()),
          str(w.cards["roll"].metric_texts()))
    w.select_axis("roll")
    w.tabs.setCurrentIndex(tui.PAGE_SPEC)
    _pump(120)
    w.p_spec.flush_draw()
    st = _texts(w.p_spec)
    check("低取樣率:無法判斷區", any("無法判斷" in s for s in st))
    check("低取樣率:Nyquist 5 Hz", any("Nyquist 5 Hz" in s for s in st))
    check("x 軸延伸到 25 Hz", w.p_spec.fig.axes[0].get_xlim()[1] >= 25 - 1e-6)
    shot(w, "low_spectrum", (1100, 700))
    shot(w, "low_spectrum", (900, 650))
    w.tabs.setCurrentIndex(tui.PAGE_STEP)
    _pump(100)
    w.p_step.flush_draw()
    check("低取樣率:步階時間解析度提示", any("時間解析度 100 ms" in s for s in _texts(w.p_step)))
    w.tabs.setCurrentIndex(tui.PAGE_TRACK)
    _pump(100)
    w.set_time(T0 + 45.0)
    shot(w, "low_track", (1100, 700))
    w.banner.set_expanded(True)
    shot(w, "low_table_banner", (1100, 700))
    check("安全須知展開", w.banner.more.isVisible())
    w.banner.set_expanded(False)
    w.tabs.setCurrentIndex(tui.PAGE_TABLE)
    shot(w, "low_table", (1100, 700))
    shot(w, "low_table", (900, 650))

    print("[良好報告]")
    rg = report_good()
    w.set_report(rg, t_base=T_BASE, scope_text="分析範圍:ARM 2 · 2:31–3:49 · 飛行 72 s")
    _pump(80)
    check("整體 = 可改善", any("整體:可改善" in t for t in _labels(w)))
    check("選 Yaw(唯一有建議的軸)", w.selected_axis() == "yaw")
    check("Yaw 無角度子圖 → 2 個子圖", (w.p_track.flush_draw() or True) and len(w.p_track.fig.axes) == 2)
    w.select_axis("roll")
    w.tabs.setCurrentIndex(tui.PAGE_FREQ)
    _pump(100)
    shot(w, "good_freq", (1100, 700))
    w.tabs.setCurrentIndex(tui.PAGE_SPEC)
    _pump(100)
    w.p_spec.flush_draw()
    check("良好:沒有峰值", any("沒有明顯峰值" in s for s in _texts(w.p_spec)))
    shot(w, "good_spec", (900, 650))

    print("[QuadPlane / 資料不足]")
    w.set_report(report_quadplane(), t_base=T_BASE)
    check("QuadPlane 標籤", any("QuadPlane" in c for c in _labels(w.row_chips)))
    ri = report_insufficient()
    w.set_report(ri, t_base=T_BASE)
    _pump(60)
    check("Pitch 卡片 = 無資料", any("無資料" in t for t in _labels(w.cards["pitch"])))
    w.select_axis("roll")
    w.tabs.setCurrentIndex(tui.PAGE_TRACK)
    _pump(80)
    w.p_track.flush_draw()
    check("Roll 追蹤 = 資料不足", any("資料不足" in s for s in _texts(w.p_track)))
    w.set_time(T0 + 1.0)            # 沒有游標也不出錯
    w.select_axis("pitch")
    _pump(60)
    check("Pitch 無資料佔位", any(tui.NO_AXIS_MSG in s for s in _texts(w.p_track)))
    w.select_axis("yaw")
    w.tabs.setCurrentIndex(tui.PAGE_SPEC)
    _pump(80)
    w.p_spec.flush_draw()
    check("Yaw 頻譜資料不足", any("無法計算頻譜" in s for s in _texts(w.p_spec)))
    check("建議表為空、按鈕停用", w.table.rowCount() == 0 and not w.btn_export.isEnabled())
    shot(w, "insufficient", (1100, 700))

    print("[藏在分頁後:延後繪製]")
    host = QTabWidget()
    other = QLabel("其他分頁")
    host.addTab(other, "地圖")
    w2 = tui.TuningPanel()
    host.addTab(w2, "PID 調參")
    host.resize(1100, 740)
    host.show()
    _pump(120)
    w2.set_report(report_osc(), t_base=T_BASE)
    check("藏著時圖表延後繪製", w2.p_track.has_pending_draw() or w2.p_spec.has_pending_draw()
          or w2.p_freq.has_pending_draw())
    for i in range(50):
        w2.set_time(T0 + i)          # 藏著時也安全
    host.setCurrentIndex(1)
    _pump(150)
    w2.tabs.setCurrentIndex(tui.PAGE_TRACK)
    _pump(150)
    check("顯示後畫完", not w2.p_track.has_pending_draw() and len(w2.p_track.fig.axes) == 3)
    w2.set_time(T0 + 31.0)
    check("顯示後游標可見", any(a.get_visible() for a in w2.p_track._anim))
    w2.set_report(None, message="分析失敗:ValueError")
    check("報告清空後回訊息頁", w2.stack.currentIndex() == 0 and "分析失敗" in w2.lbl_msg.text())
    w2.set_time(T0 + 31.0)
    host.close()

    w.set_report(None)
    shot(w, "none_default", (900, 650))
    w.close()
    print(f"\n截圖:{shots}")
    print("全部通過" if not failed else f"{failed} 項失敗")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
