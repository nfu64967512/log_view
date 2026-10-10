"""命令列介面(headless):批次算 ATE、匯出 CSV / 報告。

  python main.py --cli info <log.bin>
  python main.py --cli ate <log.bin> --ref GPS --est VISP --align se3
  python main.py --cli ate <log.bin> --ref GPS --est VISP --csv out.csv --report out.txt
  python main.py --cli ate <log.bin> --segment 2          # 只算第 2 個 ARM 飛行段
  python main.py --cli ate --sim --align sim3
  python main.py --cli info --sim-multi                   # 多航段模擬(開機到關機一份 log)
  python main.py --cli ekfsrc <log.bin>                   # EKF 來源檢查:有沒有偷參考 GPS
  python main.py --cli tune <log.bin> --segment 1         # PID 調參建議
  python main.py --cli tune --scenario p_high --slow      # PID 調參模擬情境(10 Hz 記錄)
"""
from __future__ import annotations

import argparse
import sys

from core import align as align_mod
from core import ekfsrc, logio, pipeline, sim, sim_tuning, sync, timeline
from core.types import EV_RESET, KIND_LABEL

_MAX_MODE_LINES = 40


def _load(args):
    scenario = getattr(args, "scenario", None)
    if scenario:
        slow = getattr(args, "slow", False)
        return sim_tuning.make_tuning_sim(scenario, fast_log=not slow, pid_log=not slow)[0]
    if args.sim_multi:
        return sim.make_multi_flight_sim()[0]
    if args.sim:
        return sim.make_sim_logdata()
    if not args.log:
        raise SystemExit("請指定 .bin 檔,或用 --sim / --sim-multi 跑模擬資料")
    return logio.load_bin(args.log)


def _print_sources(logdata):
    print(f"資料來源:{logdata.path or '(模擬)'}")
    print(f"時長 {logdata.duration:.1f} s"
          + (f",ENU 原點 {logdata.origin_desc}" if logdata.origin_desc else ""))
    print(f"{'名稱':<18}{'座標框':<12}{'樣本':>8}{'速率':>9}")
    for name, s in logdata.sources.items():
        print(f"{name:<18}{KIND_LABEL[s.kind]:<12}{s.n:>8}{s.rate_hz:>7.1f} Hz")
    for n in logdata.notes:
        print(f"  註:{n}")


def _seg_text(seg, tb: float) -> str:
    flags = [f for f, on in (("ARM 事件遺失、起點推定", seg.open_start), ("未 DISARM", seg.open_end)) if on]
    return (f"{seg.label:<7}{seg.t0 - tb:>9.1f} ~ {seg.t1 - tb:>8.1f} s"
            f"(時長 {seg.duration:.1f} s)" + (f" [{'、'.join(flags)}]" if flags else ""))


def _print_timeline(logdata):
    """ARM 飛行段、模式切換、VIO 重置(時間皆相對 log 起點)。"""
    tb = logdata.t_base
    segs = logdata.segments
    print()
    if segs:
        print(f"ARM 飛行段:{len(segs)} 段(秒,相對 log 起點)")
        for sg in segs:
            print("  " + _seg_text(sg, tb))
    else:
        print("ARM 飛行段:無(log 中沒有 ARM / DISARM 事件)")

    modes = timeline.mode_changes(logdata.events)
    print(f"飛行模式切換:{len(modes)} 筆" if modes else "飛行模式切換:無")
    for e in modes[:_MAX_MODE_LINES]:
        print(f"  {e.t - tb:>9.1f} s  {e.label}")
    if len(modes) > _MAX_MODE_LINES:
        print(f"  …還有 {len(modes) - _MAX_MODE_LINES} 筆")

    resets = [e for e in logdata.events if e.kind == EV_RESET]
    print(f"VIO 重置:{len(resets)} 次" if resets else "VIO 重置:無")
    for e in resets:
        print(f"  {e.t - tb:>9.1f} s  {e.label}")


def _add_source_args(p):
    p.add_argument("log", nargs="?")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true", help="改用模擬資料(單次 ARM)")
    g.add_argument("--sim-multi", action="store_true",
                   help="改用多航段模擬資料(開機到關機一份 log,兩次起降 + VIO 重置)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vio_ate_studio --cli", description="VIO/GPS ATE 分析")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="列出 log 中的位置來源、ARM 飛行段與事件")
    _add_source_args(pi)

    pa = sub.add_parser("ate", help="計算 ATE")
    _add_source_args(pa)
    pa.add_argument("--ref", default="GPS", help="參考來源名稱(預設 GPS)")
    pa.add_argument("--est", default="VISP", help="估測來源名稱(預設 VISP)")
    pa.add_argument("--align", default="se3",
                    choices=[k for k, _ in align_mod.ALIGN_METHODS], help="空間對齊方式")
    pa.add_argument("--offset", type=float, default=0.0, help="時間偏移(秒)")
    pa.add_argument("--auto-offset", action="store_true", help="自動估測時間偏移")
    pa.add_argument("--t0", type=float, help="分析起點(秒,相對 log 起點)")
    pa.add_argument("--t1", type=float, help="分析終點(秒,相對 log 起點)")
    pa.add_argument("--segment", type=int, metavar="N",
                    help="只分析第 N 個 ARM 飛行段(1 起算;不可與 --t0/--t1 併用)")
    pa.add_argument("--max-gap", type=float, default=0.5, help="內插間隙上限(秒)")
    pa.add_argument("--csv", help="匯出逐點誤差 CSV")
    pa.add_argument("--report", help="匯出文字報告")

    pe = sub.add_parser("ekfsrc", help="EKF 來源檢查:來源組設定、使用中的來源組、GPS 洩漏係數")
    _add_source_args(pe)
    pe.add_argument("--vio", default="VISP", help="VIO 位置來源名稱(預設 VISP)")
    pe.add_argument("--gps", default="GPS", help="GPS 來源名稱(預設 GPS)")
    pe.add_argument("--ekf", help="EKF 輸出來源名稱(預設 POS,其次 XKF1)")
    pe.add_argument("--report", help="把文字報告寫到檔案")

    pt = sub.add_parser("tune", help="PID 調參建議(追蹤誤差、震盪 → 參數調整方向)")
    _add_source_args(pt)
    pt.add_argument("--scenario", choices=list(sim_tuning.SCENARIOS),
                    help="改用 PID 調參模擬情境:" + "、".join(sim_tuning.SCENARIOS))
    pt.add_argument("--slow", action="store_true", help="模擬情境用一般 log 的 10 Hz 記錄(無 PID 記錄)")
    pt.add_argument("--t0", type=float, help="分析起點(秒,相對 log 起點)")
    pt.add_argument("--t1", type=float, help="分析終點(秒,相對 log 起點)")
    pt.add_argument("--segment", type=int, metavar="N", help="只分析第 N 個 ARM 飛行段(1 起算)")
    pt.add_argument("--param", help="把有建議值的參數寫成 .param 檔")
    pt.add_argument("--report", help="把文字報告寫到檔案")

    args = p.parse_args(argv)
    if args.cmd in ("ate", "tune") and args.segment is not None and (args.t0 is not None or args.t1 is not None):
        print("--segment 不能與 --t0 / --t1 同時使用", file=sys.stderr)
        return 2
    try:
        logdata = _load(args)
    except (OSError, ValueError) as e:
        print(f"無法讀取 log:{e}", file=sys.stderr)
        return 2

    if args.cmd == "info":
        _print_sources(logdata)
        _print_timeline(logdata)
        return 0
    if args.cmd == "tune":
        return _tune(args, logdata)
    if args.cmd == "ekfsrc":
        return _ekfsrc(args, logdata)

    for which, name in (("--ref", args.ref), ("--est", args.est)):
        if name not in logdata.sources:
            print(f"找不到來源 {name}({which});可用:{', '.join(logdata.sources)}", file=sys.stderr)
            return 2

    tb = logdata.t_base
    t0 = tb + args.t0 if args.t0 is not None else None
    t1 = tb + args.t1 if args.t1 is not None else None
    if args.segment is not None:
        segs = logdata.segments
        if not 1 <= args.segment <= len(segs):
            if segs:
                avail = "\n".join("  " + _seg_text(sg, tb) for sg in segs)
                print(f"沒有第 {args.segment} 個 ARM 飛行段;可用(1~{len(segs)}):\n{avail}",
                      file=sys.stderr)
            else:
                print(f"沒有第 {args.segment} 個 ARM 飛行段:這份 log 沒有 ARM / DISARM 事件",
                      file=sys.stderr)
            return 2
        seg = segs[args.segment - 1]
        t0, t1 = seg.t0, seg.t1
        print(f"分析 {_seg_text(seg, tb)}")

    where = f"第 {args.segment} 個 ARM 飛行段" if args.segment is not None else "所選區間"
    offset = args.offset
    try:
        if args.auto_offset:
            rt, rp, et, ep = pipeline.get_enu_pair(logdata, args.ref, args.est, t0, t1)
            offset, corr = sync.find_time_offset(rt, rp, et, ep, search_s=5.0)
            print(f"自動時間偏移:{offset * 1000:+.0f} ms(相關係數 {corr:.3f})")
        res = pipeline.compute(
            logdata, args.ref, args.est, t0=t0, t1=t1,
            offset_s=offset, method=args.align, max_gap_s=args.max_gap)
    except ValueError as e:
        print(f"無法計算 ATE({where}):{e}", file=sys.stderr)
        return 2

    print()
    print(pipeline.format_report(res, logdata))

    if args.csv:
        pipeline.save_csv(res, logdata, args.csv)
        print(f"\n已寫出 {args.csv}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(pipeline.format_report(res, logdata))
        print(f"已寫出 {args.report}")
    return 0


def _ekfsrc(args, logdata) -> int:
    info = ekfsrc.inspect(logdata)
    res, why = None, ""
    try:
        res = ekfsrc.leak_analysis(logdata, args.vio, args.gps, args.ekf)
    except ValueError as e:
        why = str(e)
    ekfsrc.apply_leak(info, res)
    text = ekfsrc.format_report(logdata, info, res)
    if res is None:
        text += f"\n註:沒有算洩漏係數({why})"
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n已寫出 {args.report}")
    return 0


def _tune(args, logdata) -> int:
    """PID 調參建議:分析範圍 = --segment / --t0 --t1 / 整份 log 的飛行部分。"""
    from core import tuning
    tb = logdata.t_base
    t0 = tb + args.t0 if args.t0 is not None else None
    t1 = tb + args.t1 if args.t1 is not None else None
    if args.segment is not None:
        segs = logdata.segments
        if not 1 <= args.segment <= len(segs):
            print(f"沒有第 {args.segment} 個 ARM 飛行段(共 {len(segs)} 段)", file=sys.stderr)
            return 2
        seg = segs[args.segment - 1]
        t0, t1 = seg.t0, seg.t1
        print(f"分析 {_seg_text(seg, tb)}")
    try:
        rep = tuning.analyze(logdata, t0, t1)
    except ValueError as e:
        print(f"無法分析:{e}", file=sys.stderr)
        return 2
    text = tuning.format_report_text(rep, tb)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n已寫出 {args.report}")
    if args.param:
        with open(args.param, "w", encoding="utf-8") as f:
            f.write(rep.param_file())
        print(f"已寫出 {args.param}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
