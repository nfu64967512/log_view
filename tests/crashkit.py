"""失事分析測試共用工具:載入真實 log(有快取)、124 的「已知真相」錨點。

時間皆為相對秒。00000124 是唯一有飛行(解鎖)的 log:
  飛行 1 335.24–619.14 s(正常降落,對照組);飛行 2 873.14–1262.13 s(降落觸地後在地面翻覆 → CRASH_CHECK)。
其餘 117/118/119/120/122/123 從未解鎖(桌上 / 手持),是「不得誤報」的對照。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import logio  # noqa: E402
from core.crash.anchors import basic_anchors  # noqa: E402
from core.crash.ctx import Ctx  # noqa: E402
from core.crash.types import Anchors  # noqa: E402

OUT = os.path.join(ROOT, "output")
BASELINES = ["00000117", "00000118", "00000119", "00000120", "00000122", "00000123"]
CRASH_LOG = "00000124"

_cache: dict[str, Ctx] = {}


def real_path(name: str) -> str:
    return os.path.join(OUT, f"{name}.BIN")


def load_ctx(name: str) -> Ctx:
    """真實 log → Ctx(同一個行程內快取;大檔解析要數十秒)。"""
    if name not in _cache:
        _cache[name] = Ctx(logio.load_bin(real_path(name)))
    return _cache[name]


def truth_anchors_124(ctx: Ctx | None = None) -> Anchors:
    """00000124 飛行 2 的已知關鍵時間(相對秒;來自研究階段的人工鑑識,容許誤差見各偵測器測試)。"""
    ctx = ctx or load_ctx(CRASH_LOG)
    a = basic_anchors(ctx)
    a.end_type = "crash_check"
    a.t_detect = 1262.125
    a.t_disarm = 1262.125
    a.t_touch = 1255.225          # 觸地 28.9 m/s²(未被判定為落地)
    a.t_warn = 1255.9             # 姿態誤差 > 10°
    a.t_loss = 1258.10            # 姿態誤差 > 30°
    a.t_first_hit = 1258.403      # 第一次硬撞擊 62 m/s²
    a.t_impact_peak = 1259.50     # 最後一次硬撞擊峰值 ~90 m/s²(倒扣)
    a.t_rest = 1260.04            # 靜止
    a.t_inverted = 1259.2
    a.scenario = "GROUND_TIPOVER"
    return a


def flight1_anchors_124(ctx: Ctx | None = None) -> Anchors:
    """00000124 飛行 1(335.24–619.14 s):正常降落(ARM.Method 0、EV17 616.39 / EV18 617.19,觸地約 615.58 s)。
    沒有失事:t_warn / t_loss / t_impact_peak 都是 None,偵測器要能處理。"""
    ctx = ctx or load_ctx(CRASH_LOG)
    a = Anchors(intervals=list(ctx.armed))
    a.interval = ctx.armed[0]
    a.t_disarm = a.interval.t1
    a.end_type = "normal_landing"
    return a


def not_armed_anchors(ctx: Ctx) -> Anchors:
    """從未解鎖的 log(117/118/119/120/122/123):和 engine 的閘門行為一致。"""
    return basic_anchors(ctx)
