"""失事 / 失效原因分析。

流程(core.crash.engine.analyze):
  1. Ctx 包住 LogData(相對秒、float64、快取);
  2. 解鎖閘門:沒有解鎖就不做失事分析(桌上 / 手持 log);
  3. 各領域偵測模組(det_*.py)依序執行,每個回傳 Finding 清單並更新 Anchors;
  4. 事件因果標記 → 假說評分 → 繁體中文報告。
"""
from .types import (CLEAR, HIT, NA, SOFT, Anchors, ArmedInterval, Finding)  # noqa: F401
