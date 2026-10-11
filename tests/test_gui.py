"""GUI 煙霧測試:不需人工點擊,涵蓋主視窗建立、背景執行緒載入 .bin、
切換來源/對齊、每張圖繪製、時間軸播放與選段、匯出。

  python tests\test_gui.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QEventLoop, QPoint, QPointF, Qt, QTimer  # noqa: E402
from PyQt6.QtGui import QWheelEvent  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from core import align as align_mod  # noqa: E402
from core import fakebin, sim  # noqa: E402
from ui import plots, theme  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402


def _pump(ms: int = 120):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _wait_until(pred, timeout_ms: int = 30000) -> bool:
    waited = 0
    while waited < timeout_ms:
        if pred():
            return True
        _pump(100)
        waited += 100
    return False


def _visible_cursors(page) -> int:
    return sum(1 for a in page._anim if a.get_visible())


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)
    w = MainWindow()
    w.resize(1500, 980)
    w.show()
    _pump(300)
    failed = 0

    def check(label, cond, extra=""):
        nonlocal failed
        if cond:
            print(f"  [OK]   {label} {extra}")
        else:
            failed += 1
            print(f"  [FAIL] {label} {extra}")

    print("[視窗建立]")
    check("主視窗顯示", w.isVisible())
    check("計算鈕在無資料時停用", not w.btn_run.isEnabled())
    check("分頁數 = 13(含 EKF 來源)", w.tabs.count() == 13, f"(實得 {w.tabs.count()})")
    check("第二個分頁是 PID 調參", w.tabs.tabText(1) == "PID 調參")
    check("第一個分頁是 3D 地圖", w.tabs.tabText(0) == "3D 地圖")
    check("左側四個分頁", [w.side.tabText(i) for i in range(w.side.count())]
          == ["ATE 分析", "欄位", "參數", "訊息"])
    check("欄位圖一開始收起", w.vsplit.sizes()[0] == 0)
    check("無資料時播放鈕停用", not w.playback.btn_play.isEnabled())

    with tempfile.TemporaryDirectory() as d:
        binpath = os.path.join(d, "gui_sim.bin")
        fakebin.write_logdata_to_bin(sim.make_sim_logdata(), binpath)

        print("\n[背景執行緒載入 .bin]")
        w.load_bin(binpath)
        ok = _wait_until(lambda: w.logdata is not None and w.result is not None)
        check("載入 + 自動計算完成", ok)
        if not ok:
            print("  !! 逾時,終止")
            return 1
        check("解析出 GPS / VISP / POS", {"GPS", "VISP", "POS"} <= set(w.logdata.sources),
              f"({sorted(w.logdata.sources)})")
        check("進度條已隱藏", not w.progress.isVisible())
        check("計算鈕已啟用", w.btn_run.isEnabled())
        check("預設參考 = GPS", w.cmb_ref.currentData() == "GPS",
              f"({w.cmb_ref.currentData()})")
        check("預設估測 = VISP", w.cmb_est.currentData() == "VISP",
              f"({w.cmb_est.currentData()})")
        rmse = w.result.stats["3d"]["rmse"]
        check("RMSE 落在合理範圍", 0.5 < rmse < 2.0, f"(3D RMSE {rmse:.3f} m)")
        check("統計磚已填值", w.tile_3d.value.text() != "—", f"({w.tile_3d.value.text()})")
        check("檔案標籤含記錄時間(GPS 週 / 毫秒換算)", "記錄時間:" in w.lbl_file.text()
              and "~" in w.lbl_file.text(), f"({w.lbl_file.text()[:90]})")
        pe = w.p_ekf
        sets_txt = " ".join(pe.tbl_sets.item(r, c).text() for r in range(pe.tbl_sets.rowCount())
                            for c in range(pe.tbl_sets.columnCount()) if pe.tbl_sets.item(r, c))
        check("EKF 來源分頁:來源組設定表有內容(★ = 用 GPS、來源組 2 = 純 VIO)",
              "★ GPS" in sets_txt and "純 VIO" in sets_txt, f"({sets_txt[:60]})")
        check("EKF 來源分頁:使用中來源組分段表有資料",
              pe.tbl_seg.rowCount() >= 1 and "來源組" not in (pe.tbl_seg.item(0, 1).text() or "x")
              and pe.tbl_seg.item(0, 4).text() != "", f"({pe.tbl_seg.rowCount()} 段)")
        pe.plot.flush_draw()                    # 分頁沒顯示時圖是延後繪製的
        check("EKF 來源分頁:GPS(參考)+ VIO(估測)→ 畫出洩漏係數圖",
              pe.plot.fig.axes and "洩漏係數" in pe.plot.fig.axes[0].get_ylabel(),
              f"({[a.get_ylabel() for a in pe.plot.fig.axes]})")
        tr = w.p_att._track
        check("姿態儀:載入 log 後有 ATT 序列", tr is not None and len(tr.t) > 100)
        i = len(tr.t) // 3
        w.p_att.set_time(float(tr.t[i]))
        st = w.p_att._state
        check("姿態儀:播放頭位置取到對應的 Roll / Pitch / Yaw",
              st is not None and abs(st["roll"] - tr.roll[i]) < 1e-9
              and abs(st["pitch"] - tr.pitch[i]) < 1e-9 and 0 <= st["yaw"] < 360)
        w.p_att.set_time(float(tr.t[-1]) + 1000)
        check("姿態儀:離 ATT 資料太遠 → 無資料而非沿用舊值", w.p_att._state is None)
        check("姿態儀浮動視窗:預設隱藏、可從檢視選單打開", w.dock_att.isFloating() and not w.dock_att.isVisible())
        w.dock_att.toggleViewAction().trigger()
        _pump(100)
        check("姿態儀浮動視窗:打開後顯示", w.dock_att.isVisible())
        w.tabs.setCurrentWidget(w.p_top)          # 停在別的分頁,姿態儀仍要跟著播放頭
        w.playback.seek(float(tr.t[i]))
        _pump(100)
        a1 = dict(w.p_att._state or {})
        w.playback.seek(float(tr.t[2 * i]))
        _pump(100)
        a2 = w.p_att._state
        check("姿態儀:切到其他分頁後仍隨播放頭更新",
              a1 and a2 and abs(a2["roll"] - tr.roll[2 * i]) < 1e-9 and a1["roll"] != a2["roll"])
        w.dock_att.hide()
        w.playback.seek(float(tr.t[i]))
        w.dock_att.show()
        _pump(100)
        check("姿態儀:關掉再開 → 立刻補上目前播放頭的姿態",
              w.p_att._state is not None and abs(w.p_att._state["roll"] - tr.roll[i]) < 1e-9)
        check("移動距離磚:標籤用來源名、數值 = 結果的距離",
              w.tile_dist_ref.cap.text() == f"{w.result.ref_name} 移動距離"
              and w.tile_dist_est.cap.text() == f"{w.result.est_name} 移動距離"
              and float(w.tile_dist_ref.value.text()) == round(w.result.path_len_ref, 1)
              and float(w.tile_dist_est.value.text()) == round(w.result.path_len_est, 1),
              f"({w.tile_dist_ref.cap.text()} {w.tile_dist_ref.value.text()} / "
              f"{w.tile_dist_est.cap.text()} {w.tile_dist_est.value.text()})")
        check("統計表已填值", w.table.item(0, 1).text() != "—",
              f"({w.table.item(0, 1).text()})")
        check("報告已產生", "ATE RMSE" in w.txt_report.toPlainText())
        check("解析出 1 個 ARM 段", len(w.logdata.segments) == 1,
              f"({[(s.label, round(s.t0 - w.logdata.t_base, 1)) for s in w.logdata.segments]})")

        print("\n[每張圖都畫得出來]")
        for i in range(w.tabs.count()):
            page = w.tabs.widget(i)
            if not isinstance(page, plots.PlotWidget):
                continue
            w.tabs.setCurrentIndex(i)
            _pump(150)
            name = w.tabs.tabText(i)
            # 佔位訊息的 axes 只有文字、沒有資料 artist(Axes3D 的 axison 語意
            # 與 2D 不同,不能拿來判斷)
            n_art = sum(len(a.lines) + len(a.collections) + len(a.patches)
                        for a in page.fig.axes)
            check(f"分頁「{name}」", n_art > 0,
                  f"({len(page.fig.axes)} 個 axes, {n_art} 個 artist)")

        print("\n[切換對齊方式,結果要跟著變]")
        base = w.result.stats["3d"]["rmse"]
        seen = {}
        for idx in range(w.cmb_align.count()):
            w.cmb_align.setCurrentIndex(idx)
            _pump(250)
            key = w.cmb_align.itemData(idx)
            seen[key] = w.result.stats["3d"]["rmse"]
        check("Sim(3) ≤ SE(3) ≤ 不對齊",
              seen["sim3"] <= seen["se3"] + 1e-9 <= seen["none"] + 1e-9,
              f"(sim3 {seen['sim3']:.3f} / se3 {seen['se3']:.3f} / none {seen['none']:.3f})")
        check("涵蓋全部 5 種對齊", len(seen) == len(align_mod.ALIGN_METHODS))
        w.cmb_align.setCurrentIndex(0)
        _pump(250)
        check("切回 SE(3) 數值一致", abs(w.result.stats["3d"]["rmse"] - base) < 1e-9)

        print("\n[時間裁切]")
        w.spn_t0.setValue(30.0)
        w.spn_t1.setValue(90.0)
        _pump(400)
        check("裁切後時長 ≈ 60 s", 59 < w.result.duration < 61,
              f"({w.result.duration:.1f} s, {w.result.n} 對樣本)")
        check("飛行段選單顯示「自訂區間」",
              w.playback.cmb_segment.currentIndex() == w.playback.cmb_segment.count() - 1,
              f"({w.playback.cmb_segment.currentText()})")

        print("\n[時間軸游標]")
        tb = w.logdata.t_base
        pb = w.playback
        w.tabs.setCurrentWidget(w.p_top)
        _pump(200)
        pb.seek(tb + 60.0)
        _pump(80)
        check("播放頭在分析區間內 → 平面軌跡游標可見", _visible_cursors(w.p_top) > 0,
              f"({_visible_cursors(w.p_top)} 個)")
        check("讀數顯示誤差", "誤差" in pb.lbl_readout.text() and "m" in pb.lbl_readout.text(),
              f"({pb.lbl_readout.text()[:60]}…)")
        pb.seek(tb + 10.0)
        _pump(80)
        check("播放頭在分析區間外 → 平面軌跡游標隱藏", _visible_cursors(w.p_top) == 0)
        check("讀數標示在分析區間外", "區間外" in pb.lbl_readout.text())
        w.tabs.setCurrentWidget(w.p_err)
        _pump(200)
        pb.seek(tb + 45.0)
        _pump(80)
        check("切分頁後誤差時序也有游標", _visible_cursors(w.p_err) > 0)
        pb.seek(tb - 500.0)
        check("播放頭不會跑出 log 範圍", abs(pb.time() - tb) < 1e-9, f"({pb.time() - tb:+.3f} s)")

        print("\n[播放]")
        pb.chk_window_only.setChecked(False)
        pb.seek(tb + 20.0)
        pb.set_speed(10.0)
        t_before = pb.time()
        pb.play()
        _pump(700)
        pb.pause()
        adv = pb.time() - t_before
        check("10× 播放 0.7 s ≈ 前進 7 s", 3.0 < adv < 10.0, f"(前進 {adv:.2f} s)")
        check("暫停後不再前進", (lambda t: (_pump(200), abs(pb.time() - t) < 1e-9)[1])(pb.time()))

        pb.chk_window_only.setChecked(True)
        pb.seek(tb + 88.0)
        pb.play()
        ok = _wait_until(lambda: not pb.is_playing(), 3000)
        check("只播分析區間:播到終點自動停", ok and abs(pb.time() - (tb + 90.0)) < 0.05,
              f"(停在 {pb.time() - tb:.2f} s)")
        pb.play()
        _pump(50)
        check("在終點再按播放 → 從區間起點重播", pb.time() - tb < 35.0,
              f"({pb.time() - tb:.2f} s)")
        pb.pause()

        acts = {a.text(): a for a in w._play_actions}
        a_play = next((a for k, a in acts.items() if "播放" in k), None)
        check("播放選單有「播放/暫停」動作", a_play is not None, f"({list(acts)})")
        if a_play is not None:
            a_play.trigger()
            _pump(80)
            on = pb.is_playing()
            a_play.trigger()
            _pump(50)
            check("動作可切換播放 / 暫停", on and not pb.is_playing())

        print("\n[時間軸拖曳分析區間]")
        pb.timeline.windowEdited.emit(tb + 10.0, tb + 50.0)
        _pump(400)
        check("拖曳後起訖寫回左側欄位",
              abs(w.spn_t0.value() - 10.0) < 0.01 and abs(w.spn_t1.value() - 50.0) < 0.01,
              f"({w.spn_t0.value():.2f} ~ {w.spn_t1.value():.2f} s)")
        check("結果跟著重算(時長 ≈ 40 s)", 39 < w.result.duration < 41,
              f"({w.result.duration:.1f} s)")
        pb.select_segment(-1)
        _pump(400)
        check("選「整份 log」恢復全區間",
              w.spn_t0.value() < 0.01 and abs(w.spn_t1.value() - w.logdata.duration) < 0.01)

        print("\n[自動估時間偏移]")
        w.on_auto_offset()
        _pump(600)
        check("偏移估值接近 0", abs(w.spn_offset.value()) < 0.3,
              f"({w.spn_offset.value() * 1000:+.0f} ms)")
        w.spn_offset.setValue(0.0)
        _pump(300)

        print("\n[換成 POS 當估測]")
        i_pos = next(i for i in range(w.cmb_est.count()) if w.cmb_est.itemData(i) == "POS")
        w.cmb_est.setCurrentIndex(i_pos)
        _pump(500)
        check("POS 比 VISP 更貼近 GPS",
              w.result.stats["3d"]["rmse"] < base,
              f"(POS {w.result.stats['3d']['rmse']:.3f} < VISP {base:.3f})")
        check("圖例文字已更新", "POS" in w.chip_est.lab.text(), f"({w.chip_est.lab.text()})")

        print("\n[參考 = 估測 時不當機]")
        i_gps = next(i for i in range(w.cmb_est.count()) if w.cmb_est.itemData(i) == "GPS")
        w.cmb_est.setCurrentIndex(i_gps)
        _pump(400)
        check("視窗仍存活", w.isVisible())
        check("舊結果已清掉(不讓舊數字冒充新設定)",
              w.result is None and w.tile_3d.value.text() == "—"
              and w.tile_dist_ref.value.text() == "—" and w.tile_dist_est.value.text() == "—"
              and "無法計算" in w.txt_report.toPlainText(), f"({w.tile_3d.value.text()})")
        check("讀數不再顯示舊誤差", "誤差" not in w.playback.lbl_readout.text(),
              f"({w.playback.lbl_readout.text()[:40]})")
        check("軌跡播放改為提示", w.p_play._ax is None)
        w.cmb_est.setCurrentIndex(i_pos)
        _pump(400)
        check("換回不同來源後重新算出結果", w.result is not None)

        print("\n[匯出]")
        csv_p = os.path.join(d, "out.csv")
        rep_p = os.path.join(d, "out.txt")
        png_p = os.path.join(d, "out.png")
        from core import pipeline
        pipeline.save_csv(w.result, w.logdata, csv_p)
        with open(rep_p, "w", encoding="utf-8") as f:
            f.write(pipeline.format_report(w.result, w.logdata))
        w.tabs.setCurrentWidget(w.p_top)
        _pump(150)
        pb.seek(0.5 * (w.result.t[0] + w.result.t[-1]))
        _pump(80)
        n_vis = _visible_cursors(w.p_top)
        w.tabs.currentWidget().save_png(png_p, dpi=120)
        for p, lbl, mn in ((csv_p, "CSV", 10000), (rep_p, "報告", 500), (png_p, "PNG", 20000)):
            sz = os.path.getsize(p)
            check(f"{lbl} 寫出", sz > mn, f"({sz // 1024} KB)")
        check("存圖後游標恢復顯示", n_vis > 0 and _visible_cursors(w.p_top) == n_vis,
              f"({n_vis} → {_visible_cursors(w.p_top)})")

        print("\n[開機到關機一份 log:兩次 ARM、中間 VIO 重置]")
        w.on_load_sim_multi()
        _pump(500)
        segs = w.logdata.segments
        tb = w.logdata.t_base
        check("偵測到 2 個 ARM 段", len(segs) == 2,
              f"({[(s.label, round(s.t0 - tb, 1), round(s.t1 - tb, 1)) for s in segs]})")
        check("飛行段選單 = 整份 + 2 段 + 自訂", pb.cmb_segment.count() == 4,
              f"({[pb.cmb_segment.itemText(i) for i in range(pb.cmb_segment.count())]})")
        whole = w.result.stats["3d"]["rmse"] if w.result is not None else float("nan")
        per = []
        for i in range(len(segs)):
            pb.select_segment(i)
            _pump(500)
            per.append(w.result.stats["3d"]["rmse"])
            check(f"選 {segs[i].label} → 區間 = 該段",
                  abs(w.spn_t0.value() - (segs[i].t0 - tb)) < 0.01
                  and abs(w.spn_t1.value() - (segs[i].t1 - tb)) < 0.01,
                  f"({w.spn_t0.value():.2f} ~ {w.spn_t1.value():.2f} s)")
            check(f"選 {segs[i].label} → 選單對到該段", pb.cmb_segment.currentIndex() == i + 1)
            check(f"選 {segs[i].label} → 播放頭跳到段首", abs(pb.time() - segs[i].t0) < 0.05)
        check("逐段分析遠比整份 log 準(VIO 重置後不同框)",
              len(per) == 2 and max(per) * 2 < whole,
              f"(整份 {whole:.2f} m / 逐段 {', '.join(f'{v:.2f}' for v in per)} m)")
        w.on_auto_offset()
        _pump(400)
        check("自動估偏移只看分析區間(不被 VIO 重置跳點帶偏)", abs(w.spn_offset.value()) < 0.15,
              f"({w.spn_offset.value() * 1000:+.0f} ms)")
        w.spn_offset.setValue(0.0)
        _pump(300)
        w.tabs.setCurrentWidget(w.p_play)
        _pump(250)
        pb.seek(segs[1].t0 + 20.0)
        _pump(100)
        check("讀數顯示目前飛行段", "ARM 2" in pb.lbl_readout.text(),
              f"({pb.lbl_readout.text()[:50]}…)")
        hud = [a.get_text() for a in w.p_play._anim if hasattr(a, "get_text")]
        check("軌跡播放 HUD 顯示飛行段", any("ARM 2" in s for s in hud), f"({hud[:1]})")
        check("軌跡播放游標可見", _visible_cursors(w.p_play) > 0)
        pb.seek((segs[0].t1 + segs[1].t0) / 2)
        _pump(100)
        check("兩段之間標示為地面", "地面" in pb.lbl_readout.text(),
              f"({pb.lbl_readout.text()[:50]}…)")
        resets = [e.t for e in w.logdata.events if e.kind == "reset"]
        pb.jump(+1)
        check("下一個事件 → 先停在 VIO 重置", len(resets) == 1 and abs(pb.time() - resets[0]) < 0.05,
              f"({pb.time() - tb:.2f} s)")
        pb.jump(+1)
        check("再下一個 → ARM 2 起點", abs(pb.time() - segs[1].t0) < 0.05,
              f"({pb.time() - tb:.2f} s)")
        pb.jump(-1)
        check("上一個 → 回到 VIO 重置", abs(pb.time() - resets[0]) < 0.05)

        print("\n[計算失敗時清掉舊結果]")
        w.playback.select_segment(1)
        _pump(500)
        ok_before = w.result is not None
        w.spn_t1.setValue(w.spn_t0.value() + 0.3)       # 區間太短 → 同步樣本不足
        _pump(400)
        check("區間太短 → 結果清空、統計磚歸零", ok_before and w.result is None
              and w.tile_3d.value.text() == "—", f"({w.statusBar().currentMessage()[:40]})")
        check("結果圖改為提示(不殘留舊游標)",
              all(not p._anim for p in w._result_plots()))
        w.playback.select_segment(1)
        _pump(500)
        check("再選回 ARM 2 → 重新有結果", w.result is not None)

        print("\n[欄位圖(UAV Log Viewer 式)]")
        ok = w.browser.plot_field("ATT.Roll")
        _pump(300)
        check("勾選 ATT.Roll → 欄位圖展開", ok and w.vsplit.sizes()[0] > 0
              and w.fplot.has_content(), f"(sizes {w.vsplit.sizes()})")
        t_mid = 0.5 * (segs[1].t0 + segs[1].t1)
        pb.seek(t_mid)
        _pump(100)
        check("播放頭同步到欄位圖游標", any(a.get_visible() for a in w.fplot._anim))
        w.fplot.seeked.emit(segs[0].t0 + 10.0)
        _pump(50)
        check("點欄位圖 → 播放頭跳過去", abs(pb.time() - (segs[0].t0 + 10.0)) < 1e-6)
        pb.timeline.viewChanged.emit(segs[1].t0, segs[1].t1)
        _pump(300)
        vr = w.fplot.view_range()
        check("時間軸縮放 → 欄位圖時間範圍同步",
              vr is not None and abs(vr[0] - segs[1].t0) < 0.5 and abs(vr[1] - segs[1].t1) < 0.5,
              f"({vr})")
        n_before = len(w.fplot.fig.axes)
        w.act_err_extra.setChecked(True)
        _pump(300)
        check("檢視 → 欄位圖加上 ATE 3D 誤差(另一條軸)", len(w.fplot.fig.axes) > n_before,
              f"({n_before} → {len(w.fplot.fig.axes)} 個 axes)")
        w.act_err_extra.setChecked(False)
        w.fplot.clear_series()
        _pump(300)
        check("清除欄位圖 → 收起", w.vsplit.sizes()[0] == 0, f"(sizes {w.vsplit.sizes()})")

        print("\n[參數 / 訊息面板]")
        check("參數面板有內容", len(w.params.visible_names()) > 5,
              f"({len(w.params.visible_names())} 個)")
        w.messages.seekRequested.emit(segs[1].t0 + 5.0)
        _pump(50)
        check("訊息雙擊 → 播放頭跳過去", abs(pb.time() - (segs[1].t0 + 5.0)) < 1e-6)

        print("\n[3D 地圖]")
        check("3D 地圖是第一個分頁", w.tabs.indexOf(w.p_map) == 0)
        w.tabs.setCurrentWidget(w.p_map)
        _pump(200)
        w.p_map.seeked.emit(segs[0].t0 + 3.0)
        _pump(50)
        check("點地圖航跡 → 播放頭跳過去", abs(pb.time() - (segs[0].t0 + 3.0)) < 1e-6)

        print("\n[焦點與快捷鍵]")
        w.activateWindow()                         # 姿態儀浮動視窗可能搶走啟用狀態
        _pump(50)
        w.spn_gap.setFocus()
        _pump(50)
        w.tabs.setCurrentWidget(w.p_top)
        _pump(150)
        QTest.mouseClick(w.p_top.canvas, Qt.MouseButton.LeftButton, pos=QPoint(200, 200))
        _pump(50)
        fw = QApplication.focusWidget()
        right = w.vsplit.parentWidget()
        check("點圖表 → 焦點移到右側", fw is not None and (fw is right or right.isAncestorOf(fw)),
              f"({type(fw).__name__ if fw else None})")
        was = pb.is_playing()
        QTest.keyClick(fw or right, Qt.Key.Key_Space)
        _pump(80)
        check("空白鍵切換播放(不會觸發左側按鈕)", pb.is_playing() != was
              and len(w.logdata.segments) == len(segs))
        pb.pause()

        print("\n[匯出檔名]")
        name = MainWindow._default_name(w.result, w.logdata, ".csv")
        check("預設檔名沒有 Windows 不允許的字元", not any(c in name for c in '<>:"/\\|?*'),
              f"({name})")

        print("\n[載入競態與失敗]")
        a_bin = os.path.join(d, "race_a.bin")
        b_bin = os.path.join(d, "race_b.bin")
        fakebin.write_logdata_to_bin(sim.make_multi_flight_sim()[0], a_bin)
        fakebin.write_logdata_to_bin(sim.make_sim_logdata(), b_bin)
        w.load_bin(a_bin)
        w.load_bin(b_bin)
        ok = _wait_until(lambda: w.logdata is not None and w.logdata.path == b_bin, 20000)
        _pump(800)
        check("連續開兩個檔 → 最後選的勝出", ok and w.logdata.path == b_bin,
              f"({os.path.basename(w.logdata.path)})")
        w.load_bin(a_bin)
        w.on_load_sim_multi()
        _pump(1500)
        check("載入中改按模擬資料 → 不會被背景載入蓋掉", w.logdata.path.startswith("("),
              f"({w.logdata.path})")
        bad = os.path.join(d, "bad.bin")
        with open(bad, "wb") as f:
            f.write(os.urandom(5000))
        orig_crit = QMessageBox.critical
        QMessageBox.critical = staticmethod(lambda *a, **k: None)
        try:
            prev = w.logdata
            w.load_bin(bad)
            _wait_until(lambda: not w.progress.isVisible(), 15000)
            _pump(200)
        finally:
            QMessageBox.critical = orig_crit
        check("壞檔載入失敗 → 保留原 log、計算鈕仍可用",
              w.logdata is prev and w.btn_run.isEnabled(), f"({w.lbl_file.text()[-30:]})")

        print("\n[數值欄位:停手才重算、沒焦點不吃滾輪]")
        calls = []
        orig_show = w._show_result
        w._show_result = lambda res: (calls.append(1), orig_show(res))[1]
        try:
            for v in (10.0, 11.0, 12.0, 13.0):
                w.spn_t0.setValue(v)
                _pump(40)
            _pump(500)
        finally:
            w._show_result = orig_show
        check("連續改 4 次只重算 1 次", len(calls) == 1, f"({len(calls)} 次)")
        w.side.setCurrentWidget(w._ate_page)
        _pump(100)
        w.cmb_ref.setFocus()
        _pump(50)
        before = w.spn_offset.value()
        pos = QPointF(10, 10)
        ev = QWheelEvent(pos, w.spn_offset.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 120),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(w.spn_offset, ev)
        _pump(100)
        check("沒焦點的時間偏移欄位不吃滾輪", w.spn_offset.value() == before,
              f"({before} → {w.spn_offset.value()})")
        w.spn_t0.setValue(0.0)
        _pump(500)

        print("\n[欄位圖收起時換 log / 重新拉開]")
        w.browser.plot_field("ATT.Roll")
        _pump(300)
        w.vsplit.setSizes([0, sum(w.vsplit.sizes())])
        _pump(100)
        w.on_load_sim()
        _pump(600)
        vr = w.fplot.view_range()
        ld2 = w.logdata
        check("換 log 後欄位圖範圍在新 log 內(不沿用舊 log 的時間)",
              vr is not None and vr[0] >= ld2.t_base - 1 and vr[1] <= ld2.t_last + 1,
              f"({vr[0] - ld2.t_base:.1f} ~ {vr[1] - ld2.t_base:.1f} s)" if vr else "(None)")
        sz = w.vsplit.sizes()
        w.vsplit.setSizes([0, sum(sz)])
        _pump(50)
        pb.seek(ld2.t_base + 77.0)
        _pump(50)
        w.vsplit.setSizes([300, max(1, sum(sz) - 300)])
        w._on_vsplit_moved()
        _pump(80)
        check("欄位圖重新拉開 → 游標跟上播放頭",
              w.fplot._t is not None and abs(w.fplot._t - pb.time()) < 1e-6)
        w.fplot.clear_series()
        _pump(200)

        print("\n[沒有地理原點的 log]")
        from core.types import KIND_NED, LogData, TrajSource
        import numpy as np
        tt = np.arange(0.0, 60.0, 0.1) + 50.0
        ned = np.column_stack([np.sin(tt / 5) * 5, np.cos(tt / 5) * 5, -np.ones_like(tt)])
        ned_only = LogData(sources={
            "VISP": TrajSource("VISP", KIND_NED, tt, ned=ned + 0.1),
            "XKF1": TrajSource("XKF1", KIND_NED, tt, ned=ned)}, path="(室內)")
        w.set_logdata(ned_only)
        _pump(400)
        check("只有局部 NED、沒有 ORGN → 不切到 3D 地圖", w.tabs.currentWidget() is not w.p_map)
        check("ATE 照常算得出來", w.result is not None)
        w.tabs.setCurrentWidget(w.p_map)
        _pump(150)
        check("3D 地圖顯示「無法放上地圖」而非空白", not w.p_map.has_scene)

        real = r"C:\Users\admin\Downloads\203 2025-3-7 上午 10-53-08.bin"
        if os.path.isfile(real):
            print("\n[真實 log:時間軸涵蓋到最後一筆訊息]")
            w.load_bin(real)
            ok = _wait_until(lambda: w.logdata is not None and w.logdata.path == real, 20000)
            _pump(600)
            ld3 = w.logdata
            check("播放範圍到 t_last(位置資料之後的 MSG / ERR 也到得了)",
                  ok and abs(pb.clock.hi - ld3.t_last) < 1e-6 and ld3.t_last > ld3.t_end + 60,
                  f"(t_end {ld3.t_end - ld3.t_base:.1f} s / t_last {ld3.t_last - ld3.t_base:.1f} s)")
            errs = [e.t for e in ld3.events if e.kind == "err"]
            if errs:
                w.messages.seekRequested.emit(errs[0])
                _pump(50)
                check("雙擊 t_end 之後的 ERR 訊息 → 播放頭真的跳過去",
                      abs(pb.time() - errs[0]) < 1e-6, f"({pb.time() - ld3.t_base:.1f} s)")

        print("\n[PID 調參建議]")
        w.on_load_tuning_sim("p_high")
        ok = _wait_until(lambda: w.p_tune.report() is not None, 30000)
        rep = w.p_tune.report()
        check("載入調參模擬情境 → 切到 PID 調參分頁並完成分析",
              ok and w.tabs.currentWidget() is w.p_tune and rep is not None)
        if rep is not None:
            sugg = [(s.param, s.direction) for s in rep.suggestions()]
            check("P 太高的情境 → 建議降低 Roll 的 rate P",
                  ("ATC_RAT_RLL_P", "decrease") in sugg, f"({sugg[:4]})")
            tf = next((f.t_focus for f in rep.findings if f.t_focus is not None), None)
            if tf is not None:
                w.p_tune.seekRequested.emit(tf)
                _pump(50)
                check("點診斷項目 → 播放頭跳過去", abs(pb.time() - tf) < 1e-6)
        n_before = w._tune_gen
        w.playback.select_segment(0)
        _pump(900)
        check("改分析區間 → 自動重新分析", w._tune_gen > n_before)
        w.tabs.setCurrentWidget(w.p_map)
        _pump(100)
        g = w._tune_gen
        w.spn_t1.setValue(w.spn_t1.value() - 5.0)
        _pump(900)
        check("分頁沒開時不分析(省時間)", w._tune_gen == g)

        print("\n[關閉視窗]")
        pb.play()
        w.close()
        _pump(300)
        check("關閉後載入執行緒已停", w._thread is None or not w._thread.is_alive())
        check("關閉後播放計時器已停", not pb.timer.isActive())

    print(f"\n{'=' * 60}\n{'全部通過' if not failed else f'{failed} 項失敗'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
