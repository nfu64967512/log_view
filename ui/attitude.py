"""姿態儀(人工地平儀):隨播放頭顯示 log 的 ATT(Roll / Pitch / Yaw)。

資料取自 LogData.index 的 ATT 訊息;沒有 ATT 就顯示提示。
慣例:Roll 右傾為正、Pitch 抬頭為正、Yaw 為羅盤航向(0 = 北,順時針)。
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from . import theme

_SKY = QColor("#2f6fb5")
_GROUND = QColor("#8a5a2b")
_WHITE = QColor("#ffffff")
_MAX_GAP_S = 1.0          # 播放頭離最近的 ATT 樣本超過這麼久 → 視為沒資料
_PITCH_SPAN = 25.0        # 圓內可見的俯仰範圍(±度)


class AttitudeTrack:
    """ATT 的 Roll / Pitch / Yaw 時間序列;at(t) 取最近樣本(不內插,避免 Yaw 跨 360° 出錯)。"""

    def __init__(self, t, roll, pitch, yaw, des_roll=None, des_pitch=None):
        self.t, self.roll, self.pitch, self.yaw = t, roll, pitch, yaw
        self.des_roll, self.des_pitch = des_roll, des_pitch

    @classmethod
    def from_logdata(cls, logdata) -> "AttitudeTrack | None":
        idx = getattr(logdata, "index", None)
        if idx is None or not idx.has("ATT.Roll"):
            return None
        cols = {k: idx.series(f"ATT.{k}") for k in ("Roll", "Pitch", "Yaw", "DesRoll", "DesPitch")}
        if any(cols[k] is None for k in ("Roll", "Pitch", "Yaw")) or len(cols["Roll"][0]) == 0:
            return None
        t = cols["Roll"][0]
        pick = lambda k: cols[k][1] if cols[k] is not None and len(cols[k][0]) == len(t) else None  # noqa: E731
        return cls(t, pick("Roll"), pick("Pitch"), pick("Yaw"), pick("DesRoll"), pick("DesPitch"))

    def at(self, t: float) -> dict | None:
        n = len(self.t)
        if n == 0 or not np.isfinite(t):
            return None
        i = int(np.searchsorted(self.t, t))
        i = min(max(i, 0), n - 1)
        if i > 0 and abs(self.t[i - 1] - t) < abs(self.t[i] - t):
            i -= 1
        if abs(self.t[i] - t) > _MAX_GAP_S:
            return None
        g = lambda a: None if a is None else float(a[i])  # noqa: E731
        return {"roll": g(self.roll), "pitch": g(self.pitch), "yaw": g(self.yaw) % 360.0,
                "des_roll": g(self.des_roll), "des_pitch": g(self.des_pitch)}


class AttitudeIndicator(QWidget):
    """人工地平儀 + 羅盤航向 + 數值讀數。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self._track: AttitudeTrack | None = None
        self._state: dict | None = None
        self._msg = "載入 log 後顯示姿態"

    # --- 資料 ---
    def set_logdata(self, logdata):
        self._track = AttitudeTrack.from_logdata(logdata) if logdata is not None else None
        self._state = None
        self._msg = ("這份 log 沒有 ATT 訊息(Roll / Pitch / Yaw)" if logdata is not None
                     and self._track is None else "載入 log 後顯示姿態")
        self.update()

    def set_time(self, t: float | None):
        self._state = None if self._track is None or t is None else self._track.at(t)
        if self._track is not None and self._state is None:
            self._msg = "此刻沒有姿態資料"
        self.update()

    # --- 繪製 ---
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.SURFACE))
        if self._state is None:
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._msg)
            return
        s = self._state
        side = min(self.width() * 0.62, self.height() - 24)
        r = side / 2
        cx, cy = self.width() * 0.34, self.height() / 2
        self._draw_horizon(p, cx, cy, r, s["roll"], s["pitch"])
        self._draw_compass(p, self.width() * 0.80, self.height() * 0.30, min(r * 0.62, self.width() * 0.14),
                           s["yaw"])
        self._draw_readout(p, self.width() * 0.62, self.height() * 0.60, s)

    def _draw_horizon(self, p: QPainter, cx, cy, r, roll, pitch):
        p.save()
        clip = QPainterPath()
        clip.addEllipse(QPointF(cx, cy), r, r)
        p.setClipPath(clip)
        p.translate(cx, cy)
        p.rotate(-roll)                       # 機體右傾 → 地平線相對機身逆時針轉
        k = r / _PITCH_SPAN                   # 每度對應像素
        off = pitch * k                       # 抬頭 → 地平線下移
        p.fillRect(QRectF(-3 * r, -3 * r + off, 6 * r, 3 * r), _SKY)
        p.fillRect(QRectF(-3 * r, off, 6 * r, 3 * r), _GROUND)
        p.setPen(QPen(_WHITE, 2))
        p.drawLine(QPointF(-3 * r, off), QPointF(3 * r, off))
        font = QFont(self.font()); font.setPointSizeF(max(7.0, r / 16)); p.setFont(font)
        for deg in range(-20, 21, 5):
            if deg == 0:
                continue
            y = off - deg * k
            w = r * (0.34 if deg % 10 == 0 else 0.18)
            p.drawLine(QPointF(-w, y), QPointF(w, y))
            if deg % 10 == 0:
                p.drawText(QPointF(w + 4, y + 4), str(abs(deg)))
                p.drawText(QPointF(-w - 4 - r * 0.12, y + 4), str(abs(deg)))
        p.restore()

        # 外框與滾轉刻度(隨機身不動)
        p.save()
        p.translate(cx, cy)
        p.setPen(QPen(QColor(theme.BORDER), 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(0, 0), r, r)
        p.setPen(QPen(_WHITE, 2))
        for a in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
            rad = math.radians(a)
            l = r * (0.10 if a % 30 == 0 else 0.06)
            p.drawLine(QPointF(math.sin(rad) * (r - l), -math.cos(rad) * (r - l)),
                       QPointF(math.sin(rad) * r, -math.cos(rad) * r))
        # 滾轉指標:隨地平線轉
        p.rotate(-roll)
        p.setBrush(QBrush(QColor(theme.S_WARN)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF([QPointF(0, -r + 2), QPointF(-r * 0.05, -r + r * 0.12),
                                 QPointF(r * 0.05, -r + r * 0.12)]))
        p.restore()

        # 機身符號(固定在中心)
        p.save()
        p.translate(cx, cy)
        p.setPen(QPen(QColor(theme.S_WARN), 4))
        p.drawLine(QPointF(-r * 0.45, 0), QPointF(-r * 0.15, 0))
        p.drawLine(QPointF(r * 0.15, 0), QPointF(r * 0.45, 0))
        p.drawLine(QPointF(-r * 0.15, 0), QPointF(-r * 0.15, r * 0.07))
        p.drawLine(QPointF(r * 0.15, 0), QPointF(r * 0.15, r * 0.07))
        p.setBrush(QBrush(QColor(theme.S_WARN)))
        p.drawEllipse(QPointF(0, 0), 3, 3)
        p.restore()

    def _draw_compass(self, p: QPainter, cx, cy, r, yaw):
        p.save()
        p.translate(cx, cy)
        p.setPen(QPen(QColor(theme.BORDER), 2))
        p.setBrush(QBrush(QColor(theme.PLANE)))
        p.drawEllipse(QPointF(0, 0), r, r)
        font = QFont(self.font()); font.setPointSizeF(max(7.0, r / 5)); p.setFont(font)
        p.save()
        p.rotate(-yaw)                         # 羅盤盤面轉動,機頭永遠朝上
        for i, lab in enumerate(("N", "E", "S", "W")):
            p.save()
            p.rotate(90 * i)
            p.setPen(QColor(theme.S_CRIT if lab == "N" else theme.INK_2))
            p.drawText(QRectF(-r * 0.2, -r * 0.92, r * 0.4, r * 0.3), Qt.AlignmentFlag.AlignCenter, lab)
            p.restore()
        p.setPen(QPen(QColor(theme.MUTED), 1))
        for a in range(0, 360, 30):
            rad = math.radians(a)
            p.drawLine(QPointF(math.sin(rad) * r * 0.78, -math.cos(rad) * r * 0.78),
                       QPointF(math.sin(rad) * r * 0.9, -math.cos(rad) * r * 0.9))
        p.restore()
        p.setBrush(QBrush(QColor(theme.S_WARN)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF([QPointF(0, -r * 0.55), QPointF(-r * 0.14, r * 0.3), QPointF(0, r * 0.18),
                                 QPointF(r * 0.14, r * 0.3)]))
        p.restore()

    def _draw_readout(self, p: QPainter, x, y, s):
        font = QFont(self.font()); font.setPointSizeF(11); p.setFont(font)

        def line(i, name, val, des=None):
            txt = f"{name}  {val:+7.1f}°" if name != "航向" else f"{name}  {val:6.1f}°"
            if des is not None:
                txt += f"   (目標 {des:+.1f}°)"
            p.setPen(QColor(theme.INK))
            p.drawText(QPointF(x, y + i * 24), txt)

        line(0, "滾轉", s["roll"], s["des_roll"])
        line(1, "俯仰", s["pitch"], s["des_pitch"])
        line(2, "航向", s["yaw"])
