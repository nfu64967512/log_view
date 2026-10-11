"""偵測器例外保護:單一偵測器出錯不能拖垮整份報告。"""
from __future__ import annotations

import functools

from .types import NA, Finding


def guarded(det_id: str, domain: str):
    """裝飾偵測函式 fn(ctx, a, ...) -> list[Finding] | Finding | None。

    - 回傳 Finding 自動包成 list;None 變成空 list。
    - 丟出例外 → 一筆 status=NA、context='error' 的 Finding(values['error'] 放例外說明),不往外丟。
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                r = fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 — 偵測器是輔助分析,出錯只降級為 NA
                return [Finding(det_id, domain, NA, context="error",
                                values={"error": f"{type(e).__name__}: {e}"},
                                text_zh=f"偵測器執行失敗({type(e).__name__}),結果不納入")]
            if r is None:
                return []
            return [r] if isinstance(r, Finding) else list(r)
        return wrapper
    return deco
