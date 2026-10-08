"""matplotlib 圖表元件測試(離線、不需人工點擊)。

- 延後繪製(PERF-1):視窗開著、圖藏在別的分頁時,重繪只清空、不畫;顯示時只畫最後一次;
  藏著時 set_time 不出錯、畫完後套用;佔位訊息取代尚未畫的繪製;save_png / 工具列存檔
  先把延後的繪製做完;從沒顯示過的獨立元件(tests/shot.py)與沒顯示的視窗照常立即畫。
- 軌跡播放(PERF-5):淡色整份 log / 分析區間線抽稀(保留 NaN 斷點、形狀誤差 < eps);
  跟隨模式只在飛機離開視野中央 50% 方框時置中重畫,其餘 tick 走 blit;
  藏著時連續重畫兩次仍保留跟隨縮放。並量測 2 小時合成 log 的重繪 / tick 成本。

  python tests\\test_plots_ui.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32" and os.path.isdir(r"C:\Windows\Fonts"):
    # offscreen 平台預設沒有字型資料庫(中文會變方塊),指向系統字型目錄
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QTabWidget  # noqa: E402

from core import frames, pipeline, sim  # noqa: E402
from core.types import (EV_ARM, EV_DISARM, EV_MODE, KIND_GLOBAL, KIND_NED,  # noqa: E402
                        LogData, LogEvent, TrajSource)
from ui import plots  # noqa: E402

ORIGIN = (23.702, 120.431, 38.0)


def _pump(ms: int = 80):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


class _DrawCounter:
    """數畫布的完整重繪(draw_event;savefig 的不算)。"""

    def __init__(self, w):
        self.n = 0
        self._w = w
        w.canvas.mpl_connect("draw_event", self._on)

    def _on(self, _ev):
        if not self._w.canvas.is_saving():
            self.n += 1


def _spy_draw_result(w):
    """包住 draw_result:記下每次真的被執行時的 res。"""
    calls = []
    orig = w.draw_result

    def spy(res, logdata):
        calls.append(res)
        return orig(res, logdata)

    w.draw_result = spy
    return calls


def _texts(w) -> list[str]:
    return [t.get_text() for ax in w.fig.axes for t in ax.texts]


def make_long_logdata(hours: float = 2.0) -> LogData:
    """長時間合成 log:半徑 30 m 圓周 3 m/s 一直繞(GPS 5 Hz / VISP 30 Hz),中間一段 VISP 空洞。"""
    rng = np.random.default_rng(4)
    dur = hours * 3600.0
    t0 = 1000.0

    def enu_at(t):
        ph = t * 3.0 / 30.0
        return np.column_stack([30 * np.sin(ph), 30 * (1 - np.cos(ph)), 10 + 0 * t])

    tg = np.arange(0.0, dur, 0.2)
    lat, lng, alt = frames.enu_to_geodetic(enu_at(tg) + rng.normal(0, 0.15, (tg.size, 3)), ORIGIN)
    gps = TrajSource("GPS", KIND_GLOBAL, t0 + tg, lat=lat, lng=lng, alt=alt)
    tv = np.arange(0.0, dur, 1 / 30)
    tv = tv[(tv < 0.4 * dur) | (tv > 0.4 * dur + 30)]          # 30 秒資料空洞
    ev = enu_at(tv) + rng.normal(0, 0.05, (tv.size, 3))
    visp = TrajSource("VISP", KIND_NED, t0 + tv, ned=frames.enu_to_ned(ev))
    events = [LogEvent(t0 + 1.0, EV_MODE, "AUTO"), LogEvent(t0 + 5.0, EV_ARM),
              LogEvent(t0 + dur - 5.0, EV_DISARM)]
    return LogData(sources={"GPS": gps, "VISP": visp}, origin=ORIGIN, origin_desc="合成",
                   path="long_synth.bin", events=events)


def _static_vertices(pp) -> int:
    return sum(len(ln.get_xdata()) for ln in pp._ax.lines if not ln.get_animated())


def _center(ax):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    failed = 0

    def check(label, cond, extra=""):
        nonlocal failed
        if cond:
            print(f"  [OK]   {label} {extra}")
        else:
            failed += 1
            print(f"  [FAIL] {label} {extra}")

    ld = sim.make_sim_logdata()
    res = pipeline.compute(ld, "GPS", "VISP", method="se3")
    res2 = pipeline.compute(ld, "GPS", "VISP", method="sim3")
    t_mid = float(res.t[res.n // 2])
    out_dir = tempfile.mkdtemp(prefix="plots_ui_")

    # =================================================================
    print("[延後繪製:視窗開著、圖藏在別的分頁]")
    tabs = QTabWidget()
    p_top = plots.TopViewPlot()
    p_err = plots.ErrorTimePlot()
    p_play = plots.PlaybackPlot()
    for w, name in ((p_top, "平面"), (p_err, "誤差"), (p_play, "播放")):
        tabs.addTab(w, name)
    tabs.resize(1000, 700)
    tabs.show()
    _pump(150)
    n_err = _DrawCounter(p_err)
    calls = _spy_draw_result(p_err)

    p_top.render(res, ld)
    check("目前分頁:立即畫", not p_top.has_pending_draw() and p_top.fig.axes and p_top._anim)

    _pump(80)
    d0 = n_err.n
    p_err.set_time(t_mid)                      # 先有播放頭:延後的繪製畫完要套用
    p_err.render(res, ld)
    p_err.render(res2, ld)
    check("藏著:只清空、不執行 draw_fn", calls == [] and p_err.has_pending_draw(), f"({len(calls)} 次)")
    check("藏著:清空後狀態反映新內容(沒有舊 axes / 游標)", p_err.fig.axes == [] and p_err._anim == [])
    _pump(120)
    check("藏著的畫布不重繪", n_err.n == d0, f"({n_err.n - d0} 次)")
    try:
        p_err.set_time(t_mid + 1.0)
        ok = True
    except Exception as e:  # noqa: BLE001
        ok, _ = False, print(e)
    check("繪製延後中 set_time 不出錯、記下時間", ok and p_err._t == t_mid + 1.0)

    tabs.setCurrentWidget(p_err)
    _pump(150)
    check("顯示時只畫最後一次(1 次、res2)", len(calls) == 1 and calls[0] is res2, f"({len(calls)} 次)")
    check("顯示後沒有待畫的繪製、畫布有重繪", not p_err.has_pending_draw() and n_err.n > d0)
    vis = [a for a in p_err._anim if a.get_visible()]
    x_line = p_err._cur_line.get_xdata()
    check("延後期間設的播放頭在畫完後套用(游標在 t)", vis and abs(x_line[0] - (t_mid + 1.0 - ld.t_base)) < 1e-9,
          f"({len(vis)} 個游標可見)")
    d1 = n_err.n
    p_err.set_time(t_mid + 2.0)
    _pump(60)
    check("顯示中 set_time 走 blit(不整張重繪)", n_err.n == d1, f"({n_err.n - d1} 次)")

    print("\n[延後繪製:佔位訊息取代尚未畫的繪製]")
    tabs.setCurrentWidget(p_top)
    _pump(60)
    n_calls = len(calls)
    p_err.render(res, ld)
    p_err.show_placeholder("無法計算 ATE:測試")
    check("佔位訊息取消待畫的繪製", not p_err.has_pending_draw())
    tabs.setCurrentWidget(p_err)
    _pump(120)
    check("顯示時是佔位訊息、沒執行舊的 draw_fn",
          len(calls) == n_calls and "無法計算 ATE:測試" in _texts(p_err), f"({_texts(p_err)})")

    print("\n[延後繪製:存圖前先畫完]")
    tabs.setCurrentWidget(p_top)
    _pump(60)
    p_err.render(res, ld)
    png = os.path.join(out_dir, "hidden_err.png")
    p_err.save_png(png, dpi=80)
    check("save_png 先把延後的繪製做完", not p_err.has_pending_draw() and len(calls) == n_calls + 1
          and p_err.fig.axes and os.path.getsize(png) > 15000, f"({os.path.getsize(png)} bytes)")
    p_err.render(res2, ld)
    seen = []
    real_save = plots.NavigationToolbar2QT.save_figure
    plots.NavigationToolbar2QT.save_figure = lambda tb, *a: seen.append(len(tb.canvas.figure.axes))
    try:
        p_err.toolbar.save_figure()
    finally:
        plots.NavigationToolbar2QT.save_figure = real_save
    check("工具列「存檔」也先畫完", seen and seen[0] > 0 and not p_err.has_pending_draw(), f"({seen})")

    print("\n[延後繪製:從沒顯示過的元件照常立即畫(tests/shot.py)]")
    solo = plots.TopViewPlot()
    solo.resize(800, 600)
    solo.render(res, ld)
    solo.set_time(t_mid)
    check("獨立元件:立即畫、有游標", not solo.has_pending_draw() and solo.fig.axes
          and any(a.get_visible() for a in solo._anim))
    png = os.path.join(out_dir, "solo.png")
    solo.fig.savefig(png, dpi=80)
    check("獨立元件:直接 savefig 有內容", os.path.getsize(png) > 15000, f"({os.path.getsize(png)} bytes)")
    hidden_win = QTabWidget()                  # 還沒 show 的視窗(例如建好主視窗才 show)
    a, b = plots.ErrorTimePlot(), plots.AxisErrorPlot()
    hidden_win.addTab(a, "a")
    hidden_win.addTab(b, "b")
    b.render(res, ld)
    check("視窗還沒顯示:非目前分頁也立即畫", not b.has_pending_draw() and b.fig.axes)

    print("\n[延後繪製:軌跡播放]")
    tabs.setCurrentWidget(p_top)
    _pump(60)
    p_play.draw_playback(ld, "GPS", "VISP", None)
    check("藏著:延後、_ax / 游標已清空", p_play.has_pending_draw() and p_play._ax is None and not p_play._anim)
    check("藏著:來源名稱 / 對齊狀態已是新內容",
          (p_play.ref_name, p_play.est_name, p_play.aligned) == ("GPS", "VISP", False))
    p_play.render(res, ld)
    check("換成 ATE 結果:aligned 立即為真", p_play.aligned and p_play.has_pending_draw())
    p_play.set_time(t_mid)
    tabs.setCurrentWidget(p_play)
    _pump(150)
    hud = [a.get_text() for a in p_play._anim if hasattr(a, "get_text")]
    check("顯示後畫好、HUD 有誤差讀數", p_play._ax is not None and not p_play.has_pending_draw()
          and any("誤差 3D" in h for h in hud), f"({hud[:1]})")
    p_play.chk_follow.setChecked(True)
    ax = p_play._ax
    cx, cy, _, _ = _center(ax)
    ax.set_xlim(cx - 7.0, cx + 7.0)
    ax.set_ylim(cy - 5.0, cy + 5.0)
    tabs.setCurrentWidget(p_top)
    _pump(60)
    p_play.render(res2, ld)                    # 藏著時連續重畫兩次
    p_play.render(res, ld)
    tabs.setCurrentWidget(p_play)
    _pump(150)
    _, _, hx, hy = _center(p_play._ax)
    check("藏著時連續重畫兩次:跟隨縮放仍保留", abs(hx - 7.0) < 1e-6 and abs(hy - 5.0) < 1e-6,
          f"(半寬 {hx:.2f} / 半高 {hy:.2f})")
    r = p_play._trk_ref.at(t_mid)
    cx, cy, _, _ = _center(p_play._ax)
    check("重畫後視野中心對準飛機", r is not None and abs(cx - r[0]) < 1e-6 and abs(cy - r[1]) < 1e-6)
    p_play.chk_follow.setChecked(False)
    tabs.close()

    # =================================================================
    print("\n[抽稀 _decimate_path]")
    small = np.random.default_rng(1).normal(size=(500, 3))
    check("點數不多時原樣回傳", np.array_equal(plots._decimate_path(small, 1000), small[:, :2]))
    rng = np.random.default_rng(2)
    n = 120_000
    s = np.linspace(0, 600, n)
    xy = np.column_stack([30 * np.sin(s / 10), 30 * np.cos(s / 7)]) + rng.normal(0, 0.02, (n, 2))
    xy[: n // 3] = xy[n // 3] + rng.normal(0, 0.01, (n // 3, 2))   # 前 1/3 停在地上(只有雜訊)
    nan_at = [50_000, 50_001, 80_000, 110_000]
    xy[nan_at] = np.nan
    n_max = 4000
    keep = plots._decimate_keep(xy, n_max)
    d = plots._decimate_path(xy, n_max)
    check("抽稀後點數 ≈ 上限", d.shape[0] <= n_max + 3 * len(nan_at) + 2, f"({n} → {d.shape[0]})")
    check("NaN 斷點全部保留", int((~np.isfinite(d).all(axis=1)).sum()) == len(nan_at))
    check("斷點前後端點、首末點保留",
          keep[0] and keep[-1] and all(keep[k - 1] and keep[k + 1] for k in nan_at))
    check("回傳的是原始點的子集(依序)", np.array_equal(d, xy[keep], equal_nan=True))
    step = np.hypot(*np.diff(xy, axis=0).T)
    eps = np.nansum(step) / n_max
    idx = np.flatnonzero(keep)
    last = idx[np.searchsorted(idx, np.arange(n), side="right") - 1]
    ok = np.isfinite(xy).all(axis=1)
    dev = np.hypot(*(xy[ok] - xy[last[ok]]).T)
    check("每個原始點離前一個留下的點 < eps(形狀誤差有上限)", float(dev.max()) <= eps + 1e-9,
          f"(最大 {dev.max():.3f} m,eps {eps:.3f} m)")
    still = np.zeros((50_000, 2))
    check("完全沒移動:只留首末點", plots._decimate_path(still, 1000).shape[0] == 2)

    # =================================================================
    print("\n[軌跡播放:2 小時合成 log(抽稀 + 跟隨只在離開中央方框時重畫)]")
    t0 = time.perf_counter()
    big = make_long_logdata(2.0)
    bres = pipeline.compute(big, "GPS", "VISP", method="se3")
    print(f"  (合成 + ATE {time.perf_counter() - t0:.1f} s;VISP {big.sources['VISP'].n} 筆)")
    pp = plots.PlaybackPlot()
    pp.resize(1100, 720)
    pp.show()
    _pump(100)
    draws = _DrawCounter(pp)
    t0 = time.perf_counter()
    pp.render(bres, big)
    t_render = (time.perf_counter() - t0) * 1000
    raw_n = pp._brk_ref[1].shape[0] + pp._brk_est[1].shape[0]
    nv = _static_vertices(pp)
    check("淡色線抽稀到 3 × 8000 點以內", nv <= 3 * (plots._FAINT_MAX + 20), f"(原始 {raw_n} → {nv})")
    pr = pp._brk_ref[1]
    n_nan = int((~np.isfinite(pp._ax.lines[1].get_xydata()).all(axis=1)).sum())
    check("估測淡線保留資料空洞的斷點", n_nan == int((~np.isfinite(pp._brk_est[1]).all(axis=1)).sum()) >= 1,
          f"({n_nan} 個)")
    t0 = time.perf_counter()
    pp.canvas.draw()
    t_full = (time.perf_counter() - t0) * 1000
    print(f"  (render {t_render:.0f} ms,完整重繪 {t_full:.0f} ms;參考原始 {pr.shape[0]} 點)")

    pp.cmb_tail.setCurrentIndex(len(pp.TAILS) - 1)       # 尾巴「全部」
    t = big.t_base + 1800.0
    pp.set_time(t)
    pp.chk_follow.setChecked(True)
    _pump(100)
    r = pp._trk_ref.at(t)
    cx, cy, hx, hy = _center(pp._ax)
    check("勾「跟隨」:視野立即對準飛機", abs(cx - r[0]) < 1e-6 and abs(cy - r[1]) < 1e-6,
          f"(半寬 {hx:.1f} m)")
    # 直接檢查置中規則
    ax = pp._ax
    lim = (ax.get_xlim(), ax.get_ylim())
    check("在中央方框內:不重新置中", pp._recenter(cx + 0.4 * hx, cy - 0.4 * hy) is False
          and (ax.get_xlim(), ax.get_ylim()) == lim)
    check("離開中央方框:置中、要求重畫", pp._recenter(cx + 0.6 * hx, cy) is True
          and abs(_center(ax)[0] - (cx + 0.6 * hx)) < 1e-9)
    check("force:一律置中", pp._recenter(cx, cy, True) is True and abs(_center(ax)[0] - cx) < 1e-9)

    _pump(100)
    d0 = draws.n
    recenters, outside, dts = 0, 0, []
    for _ in range(300):                       # 1× 播放 10 秒
        t += 1 / 30
        before = (ax.get_xlim(), ax.get_ylim())
        a = time.perf_counter()
        pp.set_time(t)
        app.processEvents()
        dts.append((time.perf_counter() - a) * 1000)
        if (ax.get_xlim(), ax.get_ylim()) != before:
            recenters += 1
        p = pp._trk_ref.at(t)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if p is not None and not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
            outside += 1
    dts = np.array(dts)
    full = draws.n - d0
    check("跟隨播放 10 秒:飛機始終在視野內", outside == 0, f"({outside} 次在外)")
    check("只在離開中央方框時置中(其餘 tick 走 blit)", 1 <= recenters <= 15 and full <= recenters + 2,
          f"(300 tick:置中 {recenters} 次、完整重繪 {full} 次)")
    print(f"  (每 tick 平均 {dts.mean():.1f} ms、中位數 {np.median(dts):.1f}、最大 {dts.max():.1f} ms)")
    t += 900.0
    pp.set_time(t)
    r = pp._trk_ref.at(t)
    cx, cy, _, _ = _center(ax)
    check("跳到遠處:視野置中到飛機", abs(cx - r[0]) < 1e-6 and abs(cy - r[1]) < 1e-6)
    pp.chk_follow.setChecked(False)
    check("取消跟隨:回到全圖", (ax.get_xlim(), ax.get_ylim()) == pp._home)
    pp.close()

    print(f"\n{'=' * 60}\n{'全部通過' if not failed else f'{failed} 項失敗'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
