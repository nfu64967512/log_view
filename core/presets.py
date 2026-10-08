"""常用圖表預設組(仿 UAV Log Viewer / MAVExplorer 的 preset 分組)。

欄位名稱依 ArduPilot 4.x 的 LogStructure(新舊韌體欄位名不同時兩種都列,例如
VISP 的 ``Rst`` / ``ResetCnt``);執行時用 ``available_presets(index)`` 依 log 實際內容
過濾 —— 缺欄位的序列隱藏、整組都沒有的預設丟掉。

座標軸:``axis=1`` 左軸、``axis=2`` 右軸;單位不同的序列分開放(例如電壓 / 電流)。
沒指定 instance 的訊息(``GPS.Spd``)若 log 中有多個 instance,取最小的 instance。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .expr import ExprError, referenced_keys, resolve_key

__all__ = ["SeriesSpec", "Preset", "PRESETS", "GROUPS", "available_presets", "group_presets"]

# 角色色(與 ui/theme.py 的 C_REF / C_EST 相同;core 不依賴 ui,這裡複寫一份,
# tests/test_expr.py 會檢查兩邊一致)。VIO 比較圖:VIO = 估測橘、EKF / GPS = 參考藍。
C_REF = "#3987e5"
C_EST = "#d95926"


@dataclass
class SeriesSpec:
    expr: str
    axis: int = 1
    color: str | None = None


@dataclass
class Preset:
    group: str
    name: str
    series: list[SeriesSpec]
    min_series: int = 1          # 過濾後少於此數就整組不顯示(例如「雙 GPS 比較」需 2 條)


def _p(group: str, name: str, *series, min_series: int = 1) -> Preset:
    specs = []
    for s in series:
        if isinstance(s, SeriesSpec):
            specs.append(s)
        elif isinstance(s, tuple):
            specs.append(SeriesSpec(*s))
        else:
            specs.append(SeriesSpec(s))
    return Preset(group, name, specs, min_series)


def _ch(msg: str, lo: int, hi: int) -> list[str]:
    return [f"{msg}.C{i}" for i in range(lo, hi + 1)]


def _inst(fmt: str, n: int) -> list[str]:
    return [fmt.format(i) for i in range(n)]


G_ATT, G_ALT, G_SPD, G_VIO = "姿態", "高度", "速度", "VIO / 外部導航"
G_GPS, G_EKF, G_VIB, G_PWR = "GPS", "EKF", "振動", "電力"
G_RCIN, G_OUT, G_CTRL, G_MAG, G_BRD = "RC 輸入", "馬達輸出", "控制器", "羅盤", "板子"
GROUPS = [G_ATT, G_ALT, G_SPD, G_VIO, G_GPS, G_EKF, G_VIB, G_PWR,
          G_RCIN, G_OUT, G_CTRL, G_MAG, G_BRD]

PRESETS: list[Preset] = [
    # --- 姿態 ---
    _p(G_ATT, "Roll:目標 / 實際", "ATT.DesRoll", "ATT.Roll"),
    _p(G_ATT, "Pitch:目標 / 實際", "ATT.DesPitch", "ATT.Pitch"),
    _p(G_ATT, "Yaw:目標 / 實際", "ATT.DesYaw", "ATT.Yaw"),
    _p(G_ATT, "Roll / Pitch", "ATT.Roll", "ATT.Pitch"),
    _p(G_ATT, "姿態估計誤差", "ATT.ErrRP", "ATT.ErrYaw"),
    _p(G_ATT, "AHRS 比較(ATT vs AHR2)", "ATT.Roll", "AHR2.Roll", "ATT.Pitch", "AHR2.Pitch"),
    _p(G_ATT, "偏航 vs GPS 航跡角", "ATT.Yaw", "GPS.GCrs"),
    _p(G_ATT, "GSF 偏航估計(XKY0)", "XKY0.YC", "ATT.Yaw"),

    # --- 高度 ---
    _p(G_ALT, "高度:目標 / 實際 / 氣壓(CTUN)", "CTUN.DAlt", "CTUN.Alt", "CTUN.BAlt"),
    _p(G_ALT, "高度來源比較", "CTUN.Alt", "BARO.Alt", "POS.RelHomeAlt", "rel(GPS.Alt)",
       "-XKF1.PD"),
    _p(G_ALT, "測距 / 地形高度", "CTUN.SAlt", "CTUN.DSAlt", "CTUN.TAlt", "RFND.Dist",
       "RFND.Dist1"),
    _p(G_ALT, "爬升率:目標 / 實際(CTUN)", "CTUN.DCRt", "CTUN.CRt"),
    _p(G_ALT, "氣壓計高度 / 溫度", ("BARO.Alt", 1), ("BARO.Temp", 2)),
    _p(G_ALT, "海拔高度", "GPS.Alt", "POS.Alt", "AHR2.Alt"),
    _p(G_ALT, "多氣壓計比較", *_inst("BARO[{}].Alt", 3), min_series=2),

    # --- 速度 ---
    _p(G_SPD, "地速(GPS)", "GPS.Spd"),
    _p(G_SPD, "水平速度:GPS / EKF", "GPS.Spd", "mag(XKF1.VN, XKF1.VE)",
       "mag(NKF1.VN, NKF1.VE)"),
    _p(G_SPD, "EKF 速度 N / E / D(XKF1)", "XKF1.VN", "XKF1.VE", "XKF1.VD"),
    _p(G_SPD, "垂直速度(向下為正)", "XKF1.VD", "GPS.VZ"),
    _p(G_SPD, "空速 / 地速", "ARSP.Airspeed", "GPS.Spd"),
    _p(G_SPD, "風速估計(XKF2)", "XKF2.VWN", "XKF2.VWE"),

    # --- VIO / 外部導航(VISION_POSITION_ESTIMATE 等) ---
    _p(G_VIO, "北向位置:VIO vs EKF", ("VISP.PX", 1, C_EST), ("XKF1.PN", 1, C_REF)),
    _p(G_VIO, "東向位置:VIO vs EKF", ("VISP.PY", 1, C_EST), ("XKF1.PE", 1, C_REF)),
    _p(G_VIO, "垂直位置(向下):VIO vs EKF", ("VISP.PZ", 1, C_EST), ("XKF1.PD", 1, C_REF)),
    _p(G_VIO, "位置差 VIO − EKF(N / E / D)", "VISP.PX - XKF1.PN", "VISP.PY - XKF1.PE",
       "VISP.PZ - XKF1.PD"),
    _p(G_VIO, "水平位置差 VIO − EKF", "mag(VISP.PX - XKF1.PN, VISP.PY - XKF1.PE)"),
    _p(G_VIO, "偏航:VIO vs AHRS", ("VISP.Y", 1, C_EST), ("VISP.Yaw", 1, C_EST),
       ("ATT.Yaw", 1, C_REF)),
    _p(G_VIO, "偏航差 VIO − AHRS", "wrap_180(VISP.Y - ATT.Yaw)",
       "wrap_180(VISP.Yaw - ATT.Yaw)"),
    _p(G_VIO, "Roll / Pitch:VIO vs AHRS", "VISP.R", "VISP.Roll", "ATT.Roll",
       "VISP.P", "VISP.Pitch", "ATT.Pitch"),
    _p(G_VIO, "VIO 誤差估計(位置 / 姿態)", ("VISP.PErr", 1), ("VISP.AErr", 2)),
    _p(G_VIO, "VIO 品質 / 重置 / 忽略", ("VISP.Q", 1), ("VISP.Rst", 2), ("VISP.ResetCnt", 2),
       ("VISP.Ign", 2), ("VISP.Ignored", 2)),
    _p(G_VIO, "VIO 速度(VISV)", "VISV.VX", "VISV.VY", "VISV.VZ"),
    _p(G_VIO, "水平速度:VIO vs EKF", ("mag(VISV.VX, VISV.VY)", 1, C_EST),
       ("mag(XKF1.VN, XKF1.VE)", 1, C_REF)),
    _p(G_VIO, "VIO 位置微分 vs EKF 水平速度", ("mag(ddt(VISP.PX), ddt(VISP.PY))", 1, C_EST),
       ("mag(XKF1.VN, XKF1.VE)", 1, C_REF)),
    _p(G_VIO, "VIO 速度誤差估計(VISV)", "VISV.VErr"),
    _p(G_VIO, "外部導航位置創新(XKF3)", "XKF3.IPN", "XKF3.IPE", "XKF3.IPD"),
    _p(G_VIO, "EKF 來源組切換(XKFS.SS)", "XKFS.SS"),
    _p(G_VIO, "視覺里程計增量(VISO)", ("VISO.PosDX", 1), ("VISO.PosDY", 1), ("VISO.PosDZ", 1),
       ("VISO.conf", 2)),
    _p(G_VIO, "里程計創新(XKFD)", "XKFD.IX", "XKFD.IY", "XKFD.IZ"),
    _p(G_VIO, "光流(OF)", ("OF.flowX", 1), ("OF.flowY", 1), ("OF.bodyX", 1), ("OF.bodyY", 1),
       ("OF.Qual", 2)),
    _p(G_VIO, "地速:GPS[0] vs GPS[1](GPS_INPUT)", ("GPS[0].Spd", 1, C_REF),
       ("GPS[1].Spd", 1, C_EST), min_series=2),
    _p(G_VIO, "海拔:GPS[0] vs GPS[1](GPS_INPUT)", ("GPS[0].Alt", 1, C_REF),
       ("GPS[1].Alt", 1, C_EST), min_series=2),

    # --- GPS ---
    _p(G_GPS, "衛星數 / HDop", ("GPS.NSats", 1), ("GPS.HDop", 2)),
    _p(G_GPS, "定位狀態(3 = 3D fix)", "GPS.Status"),
    _p(G_GPS, "精度估計(GPA)", ("GPA.HAcc", 1), ("GPA.VAcc", 1), ("GPA.SAcc", 2)),
    _p(G_GPS, "HDop / VDop", "GPS.HDop", "GPA.VDop"),
    _p(G_GPS, "地速 / 垂直速度", "GPS.Spd", "GPS.VZ"),
    _p(G_GPS, "航跡角", "GPS.GCrs"),
    _p(G_GPS, "雙天線偏航", "GPS.Yaw"),
    _p(G_GPS, "更新間隔(GPA.Delta)", "GPA.Delta"),
    _p(G_GPS, "雙 GPS 衛星數", *_inst("GPS[{}].NSats", 2), min_series=2),

    # --- EKF ---
    _p(G_EKF, "檢驗比(XKF4)", "XKF4.SV", "XKF4.SP", "XKF4.SH", "XKF4.SM", "XKF4.SVT"),
    _p(G_EKF, "速度創新(XKF3)", "XKF3.IVN", "XKF3.IVE", "XKF3.IVD"),
    _p(G_EKF, "位置創新(XKF3)", "XKF3.IPN", "XKF3.IPE", "XKF3.IPD"),
    _p(G_EKF, "磁力創新(XKF3)", "XKF3.IMX", "XKF3.IMY", "XKF3.IMZ"),
    _p(G_EKF, "偏航創新 / 錯誤分數(XKF3)", ("XKF3.IYAW", 1), ("XKF3.ErSc", 2)),
    _p(G_EKF, "主要核心(XKF4.PI)", "XKF4.PI"),
    _p(G_EKF, "EKF 位置 N / E / D(XKF1)", "XKF1.PN", "XKF1.PE", "XKF1.PD"),
    _p(G_EKF, "陀螺偏差(XKF1)", "XKF1.GX", "XKF1.GY", "XKF1.GZ"),
    _p(G_EKF, "EKF2 檢驗比(NKF4)", "NKF4.SV", "NKF4.SP", "NKF4.SH", "NKF4.SM"),
    _p(G_EKF, "EKF2 位置創新(NKF3)", "NKF3.IPN", "NKF3.IPE", "NKF3.IPD"),

    # --- 振動 ---
    _p(G_VIB, "振動 X / Y / Z(VIBE)", "VIBE.VibeX", "VIBE.VibeY", "VIBE.VibeZ"),
    _p(G_VIB, "削波次數(Clip)", "VIBE.Clip", "VIBE.Clip0", "VIBE.Clip1", "VIBE.Clip2"),
    _p(G_VIB, "振動強度", "mag(VIBE.VibeX, VIBE.VibeY, VIBE.VibeZ)"),
    _p(G_VIB, "各 IMU Z 軸振動", *_inst("VIBE[{}].VibeZ", 3), min_series=2),
    _p(G_VIB, "加速度計(IMU)", "IMU.AccX", "IMU.AccY", "IMU.AccZ"),
    _p(G_VIB, "陀螺儀(IMU)", "IMU.GyrX", "IMU.GyrY", "IMU.GyrZ"),
    _p(G_VIB, "各 IMU Z 加速度", *_inst("IMU[{}].AccZ", 3), min_series=2),

    # --- 電力 ---
    _p(G_PWR, "電池電壓 / 電流", ("BAT.Volt", 1), ("BAT.Curr", 2)),
    _p(G_PWR, "電壓(含內阻補償)", "BAT.Volt", "BAT.VoltR"),
    _p(G_PWR, "累積耗電 mAh", "BAT.CurrTot"),
    _p(G_PWR, "功率 W", "BAT.Volt * BAT.Curr"),
    _p(G_PWR, "剩餘電量 %", "BAT.RemPct"),
    _p(G_PWR, "電池內阻", "BAT.Res"),
    _p(G_PWR, "飛控 / 伺服供電(POWR)", "POWR.Vcc", "POWR.VServo"),

    # --- RC 輸入 ---
    _p(G_RCIN, "搖桿 C1–C4", *_ch("RCIN", 1, 4)),
    _p(G_RCIN, "通道 C5–C8", *_ch("RCIN", 5, 8)),
    _p(G_RCIN, "通道 C9–C14", *_ch("RCIN", 9, 14)),

    # --- 馬達輸出 ---
    _p(G_OUT, "輸出 C1–C4", *_ch("RCOU", 1, 4)),
    _p(G_OUT, "輸出 C5–C8", *_ch("RCOU", 5, 8)),
    _p(G_OUT, "油門(CTUN)", "CTUN.ThI", "CTUN.ThO", "CTUN.ThH"),
    _p(G_OUT, "推力餘裕(MOTB)", "MOTB.LiftMax", "MOTB.ThLimit", "MOTB.ThrAvMx", "MOTB.ThrOut"),
    _p(G_OUT, "ESC 轉速", *_inst("ESC[{}].RPM", 8)),
    _p(G_OUT, "ESC 電流", *_inst("ESC[{}].Curr", 8)),
    _p(G_OUT, "ESC 溫度", *_inst("ESC[{}].Temp", 8)),

    # --- 控制器 ---
    _p(G_CTRL, "Roll 角速率:目標 / 實際", "RATE.RDes", "RATE.R"),
    _p(G_CTRL, "Pitch 角速率:目標 / 實際", "RATE.PDes", "RATE.P"),
    _p(G_CTRL, "Yaw 角速率:目標 / 實際", "RATE.YDes", "RATE.Y"),
    _p(G_CTRL, "姿態控制輸出", "RATE.ROut", "RATE.POut", "RATE.YOut"),
    _p(G_CTRL, "垂直加速度:目標 / 實際", "RATE.ADes", "RATE.A"),
    _p(G_CTRL, "水平位置追蹤(PSCN / PSCE)", "PSCN.TPN", "PSCN.PN", "PSCE.TPE", "PSCE.PE"),
    _p(G_CTRL, "水平速度追蹤(PSCN / PSCE)", "PSCN.TVN", "PSCN.VN", "PSCE.TVE", "PSCE.VE"),
    _p(G_CTRL, "水平加速度追蹤(PSCN / PSCE)", "PSCN.TAN", "PSCN.AN", "PSCE.TAE", "PSCE.AE"),
    _p(G_CTRL, "垂直位置 / 速度追蹤(PSCD)", ("PSCD.TPD", 1), ("PSCD.PD", 1),
       ("PSCD.TVD", 2), ("PSCD.VD", 2)),
    _p(G_CTRL, "Roll PID 分量", "PIDR.P", "PIDR.I", "PIDR.D", "PIDR.FF"),
    _p(G_CTRL, "Pitch PID 分量", "PIDP.P", "PIDP.I", "PIDP.D", "PIDP.FF"),
    _p(G_CTRL, "Yaw PID 分量", "PIDY.P", "PIDY.I", "PIDY.D", "PIDY.FF"),

    # --- 羅盤 ---
    _p(G_MAG, "磁場 X / Y / Z(MAG)", "MAG.MagX", "MAG.MagY", "MAG.MagZ"),
    _p(G_MAG, "磁場強度", "mag(MAG.MagX, MAG.MagY, MAG.MagZ)"),
    _p(G_MAG, "各羅盤磁場強度",
       *[f"mag(MAG[{i}].MagX, MAG[{i}].MagY, MAG[{i}].MagZ)" for i in range(3)], min_series=2),
    _p(G_MAG, "磁力計偏移", "MAG.OfsX", "MAG.OfsY", "MAG.OfsZ"),

    # --- 板子 ---
    _p(G_BRD, "CPU 負載 / 慢迴圈數", ("PM.Load", 1), ("PM.NLon", 2)),
    _p(G_BRD, "最長迴圈時間", "PM.MaxT"),
    _p(G_BRD, "主迴圈頻率", "PM.LR"),
    _p(G_BRD, "剩餘記憶體", "PM.Mem"),
    _p(G_BRD, "飛控供電(POWR.Vcc)", "POWR.Vcc"),
    _p(G_BRD, "MCU 溫度", "MCU.MTemp", "POWR.MTemp"),
    _p(G_BRD, "MCU 電壓", "MCU.MVolt", "MCU.MVmin", "MCU.MVmax", "POWR.MVolt"),
    _p(G_BRD, "IMU / 氣壓計溫度", "IMU.T", "BARO.Temp"),
]


def _spec_ok(spec: SeriesSpec, index, memo: dict[str, bool], msgs: dict) -> bool:
    keys = referenced_keys(spec.expr)
    if not keys:
        return False
    for k in keys:
        if k not in memo:
            memo[k] = resolve_key(index, k, messages=msgs) is not None
        if not memo[k]:
            return False
    return True


def available_presets(index) -> list[Preset]:
    """依 log 內容過濾:去掉缺欄位的序列,序列數不足的預設整組丟掉(順序保留)。
    回傳新的 Preset 物件,不會改動 PRESETS。只查欄位是否存在,不載入資料。"""
    if index is None:
        return []
    memo: dict[str, bool] = {}
    try:
        msgs = index.messages() or {}
    except Exception:  # noqa: BLE001 — 只影響 instance 退路
        msgs = {}
    out: list[Preset] = []
    for p in PRESETS:
        try:
            kept = [s for s in p.series if _spec_ok(s, index, memo, msgs)]
        except ExprError:
            kept = []
        if kept and len(kept) >= max(1, p.min_series):
            out.append(replace(p, series=[replace(s) for s in kept]))
    return out


def group_presets(presets: list[Preset]) -> dict[str, list[Preset]]:
    """依 group 分組;群組與組內順序都依第一次出現的順序。"""
    groups: dict[str, list[Preset]] = {}
    for p in presets:
        groups.setdefault(p.group, []).append(p)
    return groups
