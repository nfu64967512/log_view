"""初步錨點:分析哪一次解鎖、飛控何時判定失事(CRASH_CHECK)。由 engine 在偵測模組之前呼叫。

end_state 模組會再用衝擊 / 翻覆等偵測器把 end_type 定案並補上 t_impact_peak 等。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx
from .types import Anchors

CRASH_CHECK_SUBSYS = 12
ARM_METHOD_CRASH = 12
# 正常上鎖原因(方向舵 / MAVLink / AUX 開關 / 馬達測試 / Lua / 落地後自動 / 閒置自動);其餘為失效保護 / 自動上鎖
NORMAL_DISARM_METHODS = frozenset({0, 1, 2, 3, 4, 13, 17})


def basic_anchors(ctx: Ctx) -> Anchors:
    a = Anchors(intervals=list(ctx.armed))
    if not ctx.armed:
        a.end_type = "not_armed"
        return a
    t, sub, code = ctx.err_rows()
    crash_t = [float(x) for x, s, c in zip(t, sub, code) if s == CRASH_CHECK_SUBSYS and c in (1, 2)]
    # 分析「最後一次異常結束」的解鎖區間:有 CRASH_CHECK 的那次;否則最後一次
    pick = None
    for iv in ctx.armed:
        if any(iv.t0 <= x <= iv.t1 + 0.5 for x in crash_t):
            pick = iv
    a.interval = pick or ctx.armed[-1]
    a.t_disarm = a.interval.t1
    if crash_t:
        inside = [x for x in crash_t if a.interval.t0 <= x <= a.interval.t1 + 0.5]
        if inside:
            a.t_detect = inside[0]
            a.end_type = "crash_check"
    return a
