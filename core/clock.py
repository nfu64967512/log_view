"""log 的真實時間:用 GPS 的週數(GWk)與週內毫秒(GMS)把 log 時基(TimeUS)對回 UTC。

做法同 UAV Log Viewer:取 3D 定位(Status ≥ 3)、週數 > 1000、毫秒 > 0 的 GPS 樣本,
GPS 時間 = GPS 紀元(1980-01-06)+ 週 × 7 天 + 毫秒,再扣掉 GPS − UTC 的閏秒數。
和它不同的是:不只看第一筆,而是取前面若干筆有效樣本的「UTC − log 時間」中位數(單筆有 GPS 輸出延遲)。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

GPS_EPOCH_UNIX = 315964800          # 1980-01-06 00:00:00 UTC
_WEEK_S = 7 * 24 * 3600
_MIN_WEEK = 1000                    # 週數低於此視為無效(粗略驗證)
_N_SAMPLES = 100                    # 取前 N 筆有效樣本的中位數

# 閏秒插入日(UTC 當日 00:00 起生效;1980 起共 18 個),GPS − UTC = 已插入的閏秒數
_LEAP_DATES = [(1981, 7, 1), (1982, 7, 1), (1983, 7, 1), (1985, 7, 1), (1988, 1, 1), (1990, 1, 1),
               (1991, 1, 1), (1992, 7, 1), (1993, 7, 1), (1994, 7, 1), (1996, 1, 1), (1997, 7, 1),
               (1999, 1, 1), (2006, 1, 1), (2009, 1, 1), (2012, 7, 1), (2015, 7, 1), (2017, 1, 1)]
_LEAP_UNIX = [datetime(y, m, d, tzinfo=timezone.utc).timestamp() for y, m, d in _LEAP_DATES]


def gps_minus_utc(utc_unix: float) -> int:
    """UTC 時刻 utc_unix(unix 秒)的 GPS − UTC 閏秒數(2017-01-01 起為 18)。"""
    return sum(1 for d in _LEAP_UNIX if d <= utc_unix)


def gps_to_utc_unix(week: float, ms: float) -> float:
    """GPS 週數 + 週內毫秒 → UTC unix 秒(已扣閏秒)。"""
    gps_s = GPS_EPOCH_UNIX + float(week) * _WEEK_S + float(ms) / 1000.0
    leap = 0
    for _ in range(3):                 # 閏秒數取決於 UTC 日期,迭代到穩定(跨閏秒當天最多差 1 秒)
        leap = gps_minus_utc(gps_s - leap)
    return gps_s - leap


@dataclass
class WallClock:
    unix0: float                      # log 時基 t = 0 對應的 UTC unix 秒;t 為絕對 log 秒(TimeUS / 1e6)
    source: str = "GPS"

    def unix_at(self, t: float) -> float:
        return self.unix0 + float(t)

    def utc_at(self, t: float) -> datetime:
        return datetime.fromtimestamp(self.unix_at(t), tz=timezone.utc)

    def local_at(self, t: float) -> datetime:
        """本機時區的時間(aware datetime)。"""
        return self.utc_at(t).astimezone()


def wall_clock(logdata) -> WallClock | None:
    """由 GPS 的 GWk / GMS 求 log 的真實時間;沒有 GPS、沒有 3D 定位或週數無效時回傳 None。"""
    idx = getattr(logdata, "index", None)
    if idx is None:
        return None
    try:
        info = idx.messages().get("GPS")
        if info is None or not {"GWk", "GMS", "Status"} <= set(info.fields):
            return None
        keys = ["GPS"] + [f"GPS[{i}]" for i in (info.instances or [])][1:]   # 第一個 instance 就是 "GPS"
        best: tuple[float, float] | None = None      # (最早有效樣本的 log 時間, 該來源的 unix0)
        for k in keys:
            ser = {f: idx.series(f"{k}.{f}") for f in ("Status", "GWk", "GMS")}
            if any(v is None for v in ser.values()):
                continue
            t, st = ser["Status"]
            wk, ms = ser["GWk"][1], ser["GMS"][1]
            ok = np.flatnonzero((st >= 3) & (wk > _MIN_WEEK) & (ms > 0))
            if ok.size == 0:
                continue
            ok = ok[:_N_SAMPLES]
            off = [gps_to_utc_unix(wk[i], ms[i]) - float(t[i]) for i in ok]
            cand = (float(t[ok[0]]), float(np.median(off)))
            if best is None or cand[0] < best[0]:
                best = cand
        return None if best is None else WallClock(best[1])
    except Exception:  # noqa: BLE001 — 真實時間只是輔助資訊,索引壞掉就當作沒有
        return None


def format_local(dt: datetime, with_date: bool = True) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")
