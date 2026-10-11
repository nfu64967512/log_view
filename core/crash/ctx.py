"""失事分析的資料脈絡:包住 LogData,提供「相對秒、已轉 float64」的存取,與解鎖區間 / 落地狀態。

所有偵測器只透過這個類別讀資料,這樣:
  - 時間一律是相對秒(abs − t_base);
  - ix.table() 回傳的 uint8 / uint16 / int16 欄位先轉 float64(避免減法回繞 / 平方溢位);
  - 查詢結果有快取,多個偵測器重複讀同一欄位不會重算。
"""
from __future__ import annotations

import numpy as np

from .types import ArmedInterval

_EV_NOT_LANDED, _EV_LAND_COMPLETE_MAYBE, _EV_LAND_COMPLETE = 28, 17, 18
_DISARM_EV_WINDOW = 0.001      # 上鎖那一刻同時寫出的 EV 17/18/11/56(1 ms 內)是「上鎖造成的」,不是真的落地


class Ctx:
    def __init__(self, logdata):
        self.ld = logdata
        self.ix = logdata.index
        self.t_base = float(logdata.t_base)
        self.t_end = float(logdata.t_last) - self.t_base
        self._series: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        self._tables: dict[tuple[str, int | None], dict | None] = {}
        self.params: dict[str, float] = {}
        if self.ix is not None:
            try:
                self.params = dict(self.ix.param_values())
            except Exception:  # noqa: BLE001
                self.params = {}
        self.messages: list[tuple[float, str]] = []
        if self.ix is not None:
            try:
                self.messages = sorted((float(m.t) - self.t_base, str(m.text)) for m in self.ix.log_messages()
                                       if np.isfinite(m.t))
            except Exception:  # noqa: BLE001
                self.messages = []
        # 事件:(相對秒, kind, label);kind ∈ arm / disarm / mode / reset / ev / err
        self.events = sorted(((float(e.t) - self.t_base, e.kind, e.label) for e in logdata.events),
                             key=lambda x: x[0])
        self.armed: list[ArmedInterval] = self._armed_intervals()

    # ---------------- 基本存取 ----------------

    def has(self, key: str) -> bool:
        return self.ix is not None and bool(self.ix.has(key))

    def series(self, key: str):
        """('ATT.Roll') → (t_rel, v float64) 或 None。key 同 LogIndex.series(沒寫 instance 取最小的)。"""
        if key not in self._series:
            r = None
            if self.ix is not None:
                try:
                    s = self.ix.series(key)
                except Exception:  # noqa: BLE001
                    s = None
                if s is not None and len(s[0]):
                    r = (np.asarray(s[0], dtype=float) - self.t_base, np.asarray(s[1], dtype=float))
            self._series[key] = r
        return self._series[key]

    def table(self, name: str, instance: int | None = None):
        """整張表:{'t': 相對秒, 欄位: float64 陣列(字串欄原樣)};沒有回傳 None。"""
        k = (name, instance)
        if k not in self._tables:
            out = None
            if self.ix is not None:
                try:
                    tb = self.ix.table(name, instance)
                except Exception:  # noqa: BLE001
                    tb = None
                if tb is not None and len(tb.get("t", [])):
                    out = {}
                    for c, v in tb.items():
                        v = np.asarray(v)
                        if c == "t":
                            out[c] = v.astype(float) - self.t_base
                        elif v.dtype.kind in "biuf":
                            out[c] = v.astype(np.float64)
                        else:
                            out[c] = v
            self._tables[k] = out
        return self._tables[k]

    def instances(self, name: str) -> list[int]:
        """有 instance 的訊息的 instance 編號(沒有則 [])。"""
        if self.ix is None:
            return []
        info = self.ix.messages().get(name)
        return list(info.instances) if info is not None and info.instances else []

    def param(self, name: str, default=None):
        return self.params.get(name, default)

    def ev_rows(self):
        """EV 表 → (t_rel, ids int 陣列),含 10 / 11(解鎖 / 上鎖);沒有回傳 (空, 空)。"""
        tb = self.table("EV")
        if tb is None or "Id" not in tb:
            return np.zeros(0), np.zeros(0, dtype=int)
        return tb["t"], tb["Id"].astype(int)

    def err_rows(self):
        """ERR 表 → (t_rel, subsys int, ecode int)。"""
        tb = self.table("ERR")
        if tb is None or "Subsys" not in tb:
            return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        return tb["t"], tb["Subsys"].astype(int), tb["ECode"].astype(int)

    def mode_rows(self):
        """MODE 表 → (t_rel, mode_num int, 名稱 list[str])。"""
        tb = self.table("MODE")
        if tb is None:
            return np.zeros(0), np.zeros(0, dtype=int), []
        num = tb.get("ModeNum", tb.get("Mode"))
        nums = num.astype(int) if num is not None and getattr(num, "dtype", None) is not None and num.dtype.kind in "biuf" \
            else np.zeros(tb["t"].size, dtype=int)
        try:
            names = list(self.ix.mode_labels(self.ix.raw_table("MODE")))
        except Exception:  # noqa: BLE001
            names = [f"MODE {int(n)}" for n in nums]
        return tb["t"], nums, names

    def mode_at(self, t: float) -> str | None:
        tm, _, names = self.mode_rows()
        if not tm.size:
            return None
        i = int(np.searchsorted(tm, t, side="right")) - 1
        return names[i] if i >= 0 else None

    def messages_between(self, a: float, b: float, pattern: str | None = None):
        import re
        rx = re.compile(pattern, re.I) if pattern else None
        return [(t, s) for t, s in self.messages if a <= t <= b and (rx is None or rx.search(s))]

    # ---------------- 解鎖區間 ----------------

    def _armed_intervals(self) -> list[ArmedInterval]:
        """依 ARM 表(有上鎖原因 Method)建解鎖區間;沒有 ARM 表就用 ld.events 的 arm / disarm(含 EV 10/11、STAT 備援)。"""
        rows: list[tuple[float, bool, int | None, bool]] = []     # (t, armed, method, forced)
        tb = self.table("ARM")
        if tb is not None and "ArmState" in tb:
            meth = tb.get("Method")
            forced = tb.get("Forced")
            for i, (t, s) in enumerate(zip(tb["t"], tb["ArmState"])):
                rows.append((float(t), bool(s), int(meth[i]) if meth is not None else None,
                             bool(forced[i]) if forced is not None else False))
        else:
            rows = [(t, k == "arm", None, False) for t, k, _ in self.events if k in ("arm", "disarm")]
        out: list[ArmedInterval] = []
        cur: float | None = None
        for t, armed, method, forced in sorted(rows, key=lambda r: r[0]):
            if armed and cur is None:
                cur = t
            elif not armed and cur is not None:
                out.append(ArmedInterval(len(out), cur, t, method, forced))
                cur = None
            elif not armed and cur is None and not out and t > 0:
                # 開頭就已解鎖(沒記到 ARM):log 起點到這次上鎖
                out.append(ArmedInterval(0, 0.0, t, method, forced))
        if cur is not None:
            out.append(ArmedInterval(len(out), cur, self.t_end, None, False, open_end=True))
        return out

    def armed_at(self, t: float) -> ArmedInterval | None:
        for iv in self.armed:
            if iv.t0 <= t <= iv.t1:
                return iv
        return None

    def armed_mask(self, t: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """t(相對秒陣列)每個樣本是否落在某個解鎖區間內。"""
        t = np.asarray(t, dtype=float)
        m = np.zeros(t.size, dtype=bool)
        for iv in self.armed:
            m |= (t >= iv.t0 - margin) & (t <= iv.t1 + margin)
        return m

    # ---------------- 落地狀態 ----------------

    def landed_steps(self, iv: ArmedInterval):
        """區間內的「已落地」階梯:[(t, landed bool)],起點 = 解鎖時 landed。

        EV 28 NOT_LANDED → 離地;EV 18 LAND_COMPLETE → 落地。EV 17(LAND_COMPLETE_MAYBE)不算落地確認。
        上鎖當下 1 ms 內寫出的 EV 17 / 18(上鎖造成的,不是真的落地)略過。
        """
        t, ids = self.ev_rows()
        steps = [(iv.t0, True)]
        for tt, i in zip(t, ids):
            if tt < iv.t0 or tt > iv.t1:
                continue
            if i == _EV_NOT_LANDED:
                steps.append((float(tt), False))
            elif i == _EV_LAND_COMPLETE and tt < iv.t1 - _DISARM_EV_WINDOW:
                steps.append((float(tt), True))
        return steps

    def landed_at(self, t: float, iv: ArmedInterval | None = None) -> bool | None:
        iv = iv or self.armed_at(t)
        if iv is None:
            return None
        state = True
        for tt, s in self.landed_steps(iv):
            if tt <= t:
                state = s
            else:
                break
        return state

    def ev_times(self, ev_id: int, a: float | None = None, b: float | None = None) -> list[float]:
        t, ids = self.ev_rows()
        m = ids == ev_id
        if a is not None:
            m &= t >= a
        if b is not None:
            m &= t <= b
        return [float(x) for x in t[m]]

    def ground_alt(self, iv: ArmedInterval, win: float = 3.0) -> float | None:
        """解鎖後前 win 秒 CTUN.Alt 的中位數(= 地面高度基準;沒有測距儀,AGL ≈ CTUN.Alt − 這個值)。"""
        s = self.series("CTUN.Alt")
        if s is None:
            return None
        m = (s[0] >= iv.t0) & (s[0] <= iv.t0 + win)
        return float(np.nanmedian(s[1][m])) if m.any() else None

    def rc_trim(self, ch: int, default: float = 1500.0) -> float:
        return float(self.params.get(f"RC{ch}_TRIM", default))
