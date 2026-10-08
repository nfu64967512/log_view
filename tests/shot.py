r"""離線截圖:把每張圖用模擬資料算一次並存成 PNG,不需要開 GUI。

  python tests\shot.py [輸出資料夾]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from core import pipeline, sim  # noqa: E402
from ui import plots  # noqa: E402

CHARTS = [("top", plots.TopViewPlot), ("heat", plots.HeatmapPlot),
          ("3d", plots.Plot3D), ("err", plots.ErrorTimePlot),
          ("axis", plots.AxisErrorPlot), ("dist", plots.DistributionPlot)]


def _save(w, path: str) -> None:
    w.fig.set_size_inches(12.4, 8.2)
    w.fig.savefig(path, dpi=110)   # savefig 會畫出播放頭游標(animated artist)
    print("寫出", path)


def main(outdir: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    logdata = sim.make_sim_logdata()
    res = pipeline.compute(logdata, "GPS", "VISP", method="se3")

    ov = plots.OverviewPlot()
    ov.resize(1240, 820)
    ov.draw_logdata(logdata, logdata.origin)
    _save(ov, os.path.join(outdir, "00_overview.png"))

    for i, (name, cls) in enumerate(CHARTS, start=1):
        w = cls()
        w.resize(1240, 820)
        w.render(res, logdata)
        _save(w, os.path.join(outdir, f"{i:02d}_{name}.png"))

    # 開機到關機一份 log(兩次 ARM、中間 VIO 重置):總覽 + 第二段的軌跡播放
    multi, _ = sim.make_multi_flight_sim()
    seg = multi.segments[-1]
    ov = plots.OverviewPlot()
    ov.resize(1240, 820)
    ov.draw_logdata(multi, multi.origin)
    ov.set_time(seg.t0 + 0.4 * seg.duration)
    _save(ov, os.path.join(outdir, f"{len(CHARTS) + 1:02d}_overview_multi.png"))

    res2 = pipeline.compute(multi, "GPS", "VISP", t0=seg.t0, t1=seg.t1, method="se3")
    pp = plots.PlaybackPlot()
    pp.resize(1240, 820)
    pp.render(res2, multi)
    pp.set_time(seg.t0 + 0.4 * seg.duration)
    _save(pp, os.path.join(outdir, f"{len(CHARTS) + 2:02d}_play.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "output/shots"))
