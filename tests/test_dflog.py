"""core.dflog(任意欄位索引)與新版 load_bin 驗證(不需 PyQt6):python tests/test_dflog.py

- 向量化解碼 == pymavlink DFReader 逐筆讀出的值(兩份範例 log 的每個型別、每個欄位)
- instance 拆分、字串解碼、格式乘數、key 解析、series / has / field_keys
- 參數、訊息面板(排序、EV / ERR 名稱)、機型判斷
- from_tables ↔ .bin 往返、新版 load_bin == 舊版逐筆掃描(含各種邊界情況)
- 模擬資料:既有軌跡逐位元不變、姿態符號、events 與 load_bin(fakebin(模擬)) 一致
- 大檔載入時間:舊版 vs 新版
"""
from __future__ import annotations

import array
import contextlib
import hashlib
import io
import math
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dflog, fakebin, logio, sim  # noqa: E402
from core.dflog import LogIndex, parse_key  # noqa: E402
from core.types import EV_ARM, EV_DISARM, EV_ERROR, EV_EVENT, EV_MODE, EV_RESET  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = [os.path.join(ROOT, "output", "sample_sim.bin"),
           os.path.join(ROOT, "output", "sample_multi_flight.bin")]
_LEGACY_KINDS = (EV_ARM, EV_DISARM, EV_MODE, EV_RESET)


# ---------------- 共用 ----------------

def _dfreader_all(path: str) -> dict[str, list]:
    """DFReader 逐筆讀整份 log:{型別: [DFMessage, ...]}(檔案順序)。"""
    from pymavlink import mavutil
    mlog = mavutil.mavlink_connection(path)
    out: dict[str, list] = {}
    try:
        while True:
            m = mlog.recv_msg()
            if m is None:
                break
            out.setdefault(m.get_type(), []).append(m)
    finally:
        mlog.close()
    return out


def _same_values(name: str, col: str, ours: np.ndarray, ref: list) -> None:
    assert len(ours) == len(ref), f"{name}.{col}:筆數 {len(ours)} vs DFReader {len(ref)}"
    if not ref:
        return
    if isinstance(ref[0], bytes):        # FILE.Data:DFReader 保留原始 bytes(numpy S 型別去掉尾端 \0)
        assert ours.dtype.kind == "S" and ours.tolist() == [v.rstrip(b"\0") for v in ref], f"{name}.{col}"
    elif isinstance(ref[0], str):
        assert ours.dtype.kind == "U", f"{name}.{col} 應為字串"
        assert ours.tolist() == ref, f"{name}.{col}:字串不同"
    elif isinstance(ref[0], array.array):
        assert np.array_equal(ours, np.array([list(a) for a in ref])), f"{name}.{col}:陣列不同"
    elif isinstance(ref[0], float):
        r = np.array(ref, dtype=np.float64)
        assert ours.dtype == np.float64, f"{name}.{col} dtype {ours.dtype}"
        assert np.array_equal(ours, r, equal_nan=True), \
            f"{name}.{col}:最大差 {np.nanmax(np.abs(ours - r)):.3e}"
    else:
        assert ours.dtype.kind in "iu", f"{name}.{col} dtype {ours.dtype}"
        assert ours.tolist() == [int(v) for v in ref], f"{name}.{col}:整數不同"


def _check_against_dfreader(path: str) -> tuple[int, int]:
    """每個型別、每個欄位都和 DFReader 逐筆讀出的值相同。回傳 (型別數, 欄位數)。"""
    ref = _dfreader_all(path)
    ix = LogIndex.open(path)
    n_types = n_cols = 0
    for name, msgs in ref.items():
        tbl = ix.raw_table(name)
        assert tbl is not None, f"索引缺少 {name}"
        cols = msgs[0].get_fieldnames()
        if "TimeUS" in cols:
            t_ref = [float(m.TimeUS) * 1e-6 for m in msgs]
            assert tbl["t"].tolist() == t_ref, f"{name}.t 與 TimeUS×1e-6 不同"
        for c in cols:
            if c == "TimeUS":
                continue
            _same_values(name, c, tbl[c], [getattr(m, c) for m in msgs])
            n_cols += 1
        n_types += 1
    assert set(ref) == {n for n in ref if ix.raw_table(n) is not None}
    assert sorted(set(ix.messages()) | {"FMT", "FMTU", "UNIT", "MULT"}) == \
        sorted(set(ref) | {"FMT", "FMTU", "UNIT", "MULT"}), "messages() 型別不符"
    return n_types, n_cols


def _same_logdata(a, b, what: str) -> None:
    """新舊 load_bin 結果相同:來源(順序、陣列逐位元)、原點、註記、舊有事件種類。"""
    assert list(a.sources) == list(b.sources), f"{what}:來源 {list(a.sources)} vs {list(b.sources)}"
    for k in a.sources:
        sa, sb = a.sources[k], b.sources[k]
        assert sa.kind == sb.kind and sa.name == sb.name, f"{what}:{k} 種類不同"
        for f in ("t", "lat", "lng", "alt", "ned"):
            x, y = getattr(sa, f), getattr(sb, f)
            assert (x is None) == (y is None), f"{what}:{k}.{f} 有無不同"
            if x is not None:
                assert x.shape == y.shape and np.array_equal(x, y), f"{what}:{k}.{f} 值不同"
    assert a.origin == b.origin and a.origin_desc == b.origin_desc, f"{what}:原點不同"
    assert a.notes == b.notes, f"{what}:註記 {a.notes} vs {b.notes}"
    ea = [(e.t, e.kind, e.label) for e in a.events]
    eb = [(e.t, e.kind, e.label) for e in b.events if e.kind in _LEGACY_KINDS]
    assert ea == eb, f"{what}:事件不同\n{ea}\n{eb}"


@contextlib.contextmanager
def _quiet_stderr():
    """連 C 層(dfindexer)寫到 fd 2 的訊息也收起來;結束後 out["text"] 為收到的文字。"""
    out = {"text": ""}
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            with contextlib.redirect_stderr(io.StringIO()) as py_err:
                yield out
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            out["text"] = tmp.read().decode("utf-8", "replace") + py_err.getvalue()


def _bin(d: str, name: str) -> str:
    return os.path.join(d, name)


# ---------------- 範例 log 對 DFReader ----------------

def test_values_match_dfreader():
    """兩份範例 log:每個型別、每個欄位都與 DFReader recv_msg 逐筆讀出的值相同(含 FMT / UNIT…)。"""
    for p in SAMPLES:
        nt, nc = _check_against_dfreader(p)
        print(f"  {os.path.basename(p)}:{nt} 個型別、{nc} 個欄位逐筆一致")


def test_slow_path_matches_fast():
    """慢路徑(逐筆 recv_msg)建出的索引與快路徑相同。"""
    from pymavlink import mavutil
    p = SAMPLES[1]
    fast = LogIndex.open(p)
    mlog = mavutil.mavlink_connection(p)
    try:
        slow = LogIndex(dflog._TableStore.from_recv(mlog), p)
    finally:
        mlog.close()
    assert slow.vehicle == fast.vehicle == "copter"
    assert {k: (v.fields, v.count, v.instances, v.units) for k, v in slow.messages().items()} == \
        {k: (v.fields, v.count, v.instances, v.units) for k, v in fast.messages().items()}
    for name in fast.messages():
        a, b = fast.raw_table(name), slow.raw_table(name)
        for c in a:
            assert a[c].tolist() == b[c].tolist(), f"{name}.{c}"
    assert slow.field_keys() == fast.field_keys()
    assert [(m.t, m.kind, m.text) for m in slow.log_messages()] == \
        [(m.t, m.kind, m.text) for m in fast.log_messages()]
    print(f"  慢路徑 {len(fast.messages())} 種訊息、{len(fast.field_keys())} 個欄位與快路徑相同")


# ---------------- instance / 字串 / 乘數 ----------------

def _two_gps_tables(n: int = 40) -> dict:
    t = 100.0 + np.arange(n) * 0.2
    g = {}
    for i in (0, 1):
        g[f"GPS[{i}]"] = {"t": t + 0.01 * i, "I": np.full(n, i, np.uint8),
                          "Status": np.full(n, 3 + 3 * i, np.uint8),
                          "GMS": (300000 + np.arange(n) * 200).astype(np.uint32),
                          "GWk": np.full(n, 2320, np.uint16),
                          "Lat": 23.702 + 1e-5 * np.arange(n) + 1e-6 * i,
                          "Lng": 120.431 + 2e-5 * np.arange(n), "Alt": 38.0 + 0.1 * np.arange(n),
                          "Spd": np.linspace(0, 5, n) + i}
    tb = 100.0 + np.arange(2 * n) * 0.1
    for i in (0, 1):   # 沒給 instance 欄 → from_tables 自動補 "I"
        g[f"BAT[{i}]"] = {"t": tb, "Volt": 16.0 - 0.01 * np.arange(2 * n) - i, "Curr": np.full(2 * n, 5.0 + i)}
    g["MSG"] = {"t": np.array([99.0]), "Message": np.array(["ArduCopter V4.5.7 (test)"])}
    return g


def test_instance_splitting():
    tables = _two_gps_tables()
    ix = LogIndex.from_tables(tables)
    info = ix.messages()
    assert info["GPS"].instances == [0, 1] and info["GPS"].count == 80
    assert info["BAT"].instances == [0, 1] and info["BAT"].fields[0] == "I", info["BAT"].fields
    assert ix.instance_field("BAT") == "I" and ix.unit("BAT.I") == "instance"
    with tempfile.TemporaryDirectory() as d:
        p = _bin(d, "inst.bin")
        fakebin.write_index_to_bin(ix, p)
        lx = LogIndex.open(p)
        assert lx.messages()["GPS"].instances == [0, 1]
        assert lx.instance_field("GPS") == "I" and lx.instance_field("BAT") == "I"
        for i in (0, 1):
            a, b = ix.table("GPS", i), lx.table("GPS", i)
            assert np.allclose(a["t"], b["t"], atol=1e-6, rtol=0)
            assert np.allclose(a["Spd"], b["Spd"], rtol=1e-6) and np.all(b["I"] == i)
            assert np.allclose(a["Lat"], b["Lat"], atol=1e-7, rtol=0)
        # DFReader 也依 FMTU '#' 拆 instance
        ref = _dfreader_all(p)
        assert ref["GPS"][0].fmt.instance_field == "I"
        by_i = {i: [m.Spd for m in ref["GPS"] if m.I == i] for i in (0, 1)}
        assert all(lx.table("GPS", i)["Spd"].tolist() == by_i[i] for i in (0, 1))
        assert "GPS[1].Spd" in lx.field_keys() and "GPS.Spd" not in lx.field_keys()
        assert "GPS[0].I" not in lx.field_keys(), "instance 欄不列入可繪欄位"
        assert np.array_equal(lx.series("GPS.Spd")[1], lx.series("GPS[0].Spd")[1]), \
            "不寫 instance 時取最小的 instance"
        assert lx.table("GPS", 7) is None and lx.table("MSG", 0) is None
        merged = lx.raw_table("GPS")
        assert merged["t"].size == 80 and set(merged["I"].tolist()) == {0, 1}
        # load_bin:GPS 兩個 instance → GPS[0] / GPS[1] (GPS2),新舊一致
        a, b = logio._legacy_load_bin(p), logio.load_bin(p)
        assert list(b.sources) == ["GPS[0]", "GPS[1] (GPS2)"], list(b.sources)
        _same_logdata(a, b, "兩顆 GPS")
    print("  GPS[0]/GPS[1]、BAT 自動補 instance 欄:拆分與 DFReader 一致;load_bin 顯示 GPS[1] (GPS2)")


def test_strings_and_multipliers():
    """所有格式字元(含 L / e / c / C / E / g / a / n / N / Z)與 DFReader 值逐位元相同。"""
    n = 25
    rng = np.random.default_rng(5)
    t = 50.0 + np.arange(n) * 0.1
    tbl = {"t": t,
           "Lat": rng.uniform(-80, 80, n), "Lng": rng.uniform(-179, 179, n),
           "AltE": rng.uniform(-100, 3000, n), "Cc": rng.uniform(-300, 300, n),
           "CC": rng.uniform(0, 600, n), "EE": rng.uniform(0, 4e6, n), "Hf": rng.uniform(-60, 60, n),
           "Dd": rng.normal(0, 1e6, n), "Bb": rng.integers(-128, 127, n).astype(np.int8),
           "Hh": rng.integers(-3e4, 3e4, n).astype(np.int16), "Uu": rng.integers(0, 6e4, n).astype(np.uint16),
           "Ii": rng.integers(-2e9, 2e9, n).astype(np.int32), "Qq": rng.integers(0, 2 ** 62, n).astype(np.uint64),
           "Mm": rng.integers(0, 27, n).astype(np.int8),
           "Arr": rng.integers(-3e4, 3e4, (n, 32)).astype(np.int16)}
    fmt = {("TST", "Lat"): "L", ("TST", "Lng"): "L", ("TST", "AltE"): "e", ("TST", "Cc"): "c",
           ("TST", "CC"): "C", ("TST", "EE"): "E", ("TST", "Hf"): "g", ("TST", "Dd"): "d",
           ("TST", "Mm"): "M"}
    quantum = {"Lat": 1e-7, "Lng": 1e-7, "AltE": 0.01, "Cc": 0.01, "CC": 0.01, "EE": 0.01}
    strs = {"t": t[:6], "Short": np.array(["ab", "", "WXYZ", "Q", "中", "x"]),
            "Mid": np.array(["EK3_SRC1_POSXY", "a", "", "LOG_DISARMED", "Ω", "16 chars exactly"]),
            "Long": np.array(["ArduCopter V4.5.7 (2a3dc4b7)", "中文訊息 OK", "", "x" * 64,
                              "EKF3 IMU0 is using external nav data", "é"])}
    sfmt = {("STR", "Short"): "n", ("STR", "Mid"): "N", ("STR", "Long"): "Z"}
    ix = LogIndex.from_tables({"TST": tbl, "STR": strs})
    with tempfile.TemporaryDirectory() as d:
        p = _bin(d, "fmt.bin")
        fakebin.write_index_to_bin(ix, p, fmt_override={**fmt, **sfmt})
        # 再補幾筆 DFReader 解碼的刁鑽情況:非 UTF-8(退回 latin-1)、\0 後的垃圾
        with open(p, "ab") as f:
            w = fakebin.DFBinWriter()
            w.buf = bytearray()
            w.defs["STR"] = (ix_type_id(p, "STR"), "QnNZ")
            w.write("STR", [int(60e6), b"\xe9t\xe9", b"ABC\x00\xff\xfeJUNK", b"caf\xc3\xa9\x00\x80\x81"])
            f.write(bytes(w.buf))
        nt, nc = _check_against_dfreader(p)
        lx = LogIndex.open(p)
        a = lx.table("TST")
        for c, q in quantum.items():
            err = np.max(np.abs(a[c] - tbl[c]))
            assert err <= q * 0.5 + 1e-9 * max(1.0, np.max(np.abs(tbl[c]))), f"{c} 量化誤差 {err}"
        assert np.allclose(a["Hf"], tbl["Hf"], rtol=1e-3)                     # float16
        assert np.array_equal(a["Dd"], tbl["Dd"]) and np.array_equal(a["Arr"], tbl["Arr"])
        for c in ("Bb", "Hh", "Uu", "Ii", "Qq", "Mm"):
            assert np.array_equal(a[c], tbl[c]) and a[c].dtype == tbl[c].dtype, c
        s = lx.raw_table("STR")
        assert s["Short"].tolist()[:4] == ["ab", "", "WXYZ", "Q"]
        assert s["Mid"].tolist()[3] == "LOG_DISARMED" and s["Long"].tolist()[1] == "中文訊息 OK"
        assert s["Long"].tolist()[3] == "x" * 64
        assert s["Short"].tolist()[-1] == "été" and s["Mid"].tolist()[-1] == "ABC"
        assert s["Long"].tolist()[-1] == "cafÃ©", s["Long"].tolist()[-1]
        assert lx.series("STR.Long") is None and lx.series("TST.Arr") is None
        assert "TST.Arr" not in lx.field_keys() and "STR.Short" not in lx.field_keys()
    print(f"  {nc} 個欄位(L/e/c/C/E/g/d/a/b/h/H/i/Q/M/n/N/Z)與 DFReader 逐位元一致;"
          "UTF-8 / latin-1 退回 / \\0 截斷正確")


def ix_type_id(path: str, name: str) -> int:
    from pymavlink import mavutil
    mlog = mavutil.mavlink_connection(path)
    try:
        return mlog.name_to_id[name]
    finally:
        mlog.close()


# ---------------- key / series / has ----------------

def test_parse_key():
    assert parse_key("GPS[1].Spd") == ("GPS", 1, "Spd")
    assert parse_key("ATT") == ("ATT", None, None)
    assert parse_key("ATT.Roll") == ("ATT", None, "Roll")
    assert parse_key("XKF1[0]") == ("XKF1", 0, None)
    assert parse_key(" VIBE[2].Clip0 ") == ("VIBE", 2, "Clip0")
    assert dflog.make_key("GPS", 1, "Spd") == "GPS[1].Spd" and dflog.make_key("ATT") == "ATT"
    for bad in ("", "GPS[x].Spd", "GPS.Spd.X", "A B", "GPS[1]Spd", "ATT.", "[1].Spd"):
        try:
            parse_key(bad)
        except ValueError:
            continue
        raise AssertionError(f"應拒絕 {bad!r}")
    print("  parse_key:一般 / instance / 只有訊息名 / 錯誤格式 皆正確")


def test_series_has_field_keys():
    ld, _ = sim.make_multi_flight_sim()
    ix = ld.index
    assert ix.has("ATT") and ix.has("ATT.Roll") and ix.has("GPS[0].Spd") and ix.has("GPS.Spd")
    assert not ix.has("GPS[1].Spd") and not ix.has("ATT[0].Roll") and not ix.has("ATT.Nope")
    assert not ix.has("NOPE") and not ix.has("bad key!") and ix.has("MSG.Message")
    t, v = ix.series("ATT.Roll")
    assert t.dtype == v.dtype == np.float64 and t.size == v.size > 1000 and np.all(np.diff(t) > 0)
    t2, v2 = ix.series("GPS.Spd")
    assert np.array_equal(v2, ix.series("GPS[0].Spd")[1])
    tn, vn = ix.series("GPS[0].NSats")
    assert vn.dtype == np.float64, "整數欄位轉成 float64"
    for bad in ("MSG.Message", "ATT.Nope", "ATT", "GPS[3].Spd", "???", "ATT.t"):
        assert ix.series(bad) is None, bad
    keys = ix.field_keys()
    assert keys == sorted(keys) and len(keys) == len(set(keys))
    for k in ("ATT.Roll", "ATT.DesYaw", "GPS[0].Spd", "XKF1[0].PN", "BAT[0].Volt", "CTUN.Alt",
              "VIBE[0].VibeZ", "RCOU.C1", "VISP.PX", "VISP.ResetCnt", "EV.Id"):
        assert k in keys, k
    for k in ("GPS.Spd", "GPS[0].I", "PARM.Value", "MSG.Message", "ATT.TimeUS", "BAT[0].Inst"):
        assert k not in keys, k
    assert all(ix.has(k) and ix.series(k) is not None for k in keys), "每個 field_key 都畫得出來"
    assert ix.unit("ATT.Roll") == "deg" and ix.unit("CTUN.CRt") == "cm/s" and ix.unit("GPS[0].Spd") == "m/s"
    assert ix.unit("ATT.AEKF") == "" and ix.unit("nope") == ""
    tbl = ix.table("ATT")
    try:
        tbl["Roll"][0] = 1.0
    except ValueError:
        pass
    else:
        raise AssertionError("快取陣列應為唯讀")
    assert ld.sources["GPS"].t.flags.writeable, "from_tables 不可把呼叫端的陣列設成唯讀"
    assert ix.table("ATT", 0) is None and ix.table("NOPE") is None
    lo, hi = ix.t_range()
    assert abs(lo - ld.t_base) < 1e-9 and abs(hi - ld.t_end) < 1e-9, (lo, hi)
    print(f"  {len(keys)} 個可繪欄位;has / series / unit / 唯讀快取 / t_range 正確")


# ---------------- 參數 / 訊息 / 名稱 ----------------

def test_params():
    ld = sim.make_sim_logdata()
    ix = ld.index
    pv = ix.param_values()
    assert pv["VISO_TYPE"] == 1 and pv["EK3_SRC2_POSXY"] == 6 and pv["LOG_DISARMED"] == 1
    assert len(pv) == len(sim.SIM_PARAMS) and abs(pv["VISO_POS_M_NSE"] - 0.2) < 1e-7
    ps = ix.params()
    assert [p.name for p in ps] == sorted(pv)
    p = next(p for p in ps if p.name == "WPNAV_SPEED")
    assert (p.value, p.default) == (300.0, 1000.0) and abs(p.t - (ld.t_base + 0.01)) < 0.05
    # 飛行中改參數:值取最後一次、t = 改成該值的時間;重複寫同值不改 t
    tbl = {"PARM": {"t": np.array([1.0, 1.1, 50.0, 80.0, 90.0]),
                    "Name": np.array(["WPNAV_SPEED", "RTL_ALT", "WPNAV_SPEED", "WPNAV_SPEED", "RTL_ALT"]),
                    "Value": np.array([500.0, 1500.0, 800.0, 800.0, 1500.0])},
           "GPS": {"t": np.arange(12.0), "Lat": np.zeros(12), "Lng": np.zeros(12), "Alt": np.zeros(12)}}
    jx = LogIndex.from_tables(tbl)
    q = {p.name: p for p in jx.params()}
    assert q["WPNAV_SPEED"].value == 800.0 and q["WPNAV_SPEED"].t == 50.0
    assert q["RTL_ALT"].t == 1.1 and math.isnan(q["RTL_ALT"].default), "沒有 Default 欄 → nan"
    assert jx.param_history("WPNAV_SPEED") == [(1.0, 500.0), (50.0, 800.0)]
    # .bin:與 DFReader 的 params / param_defaults 相同
    from pymavlink import mavutil
    mlog = mavutil.mavlink_connection(SAMPLES[0])
    try:
        while mlog.recv_match(type="PARM") is not None:
            pass
        ref, ref_d = dict(mlog.params), dict(getattr(mlog, "param_defaults", {}))
    finally:
        mlog.close()
    lx = LogIndex.open(SAMPLES[0])
    assert lx.param_values() == ref
    assert {p.name: p.default for p in lx.params()} == ref_d
    print(f"  {len(pv)} 個參數:值 / 預設值 / 設定時間 / 變更歷史 與 DFReader 一致")


def test_names():
    assert dflog.err_text(6, 1) == "FAILSAFE_BATT:觸發 (1)"
    assert dflog.err_text(6, 0) == "FAILSAFE_BATT:解除 (0)"
    assert dflog.err_text(11, 2) == "GPS:Glitch (2)"
    assert dflog.err_text(11, 0) == "GPS:已恢復 (0)"
    assert dflog.err_text(16, 2) == "EKFCHECK:變異數過大 (2)"
    assert dflog.err_text(16, 0) == "EKFCHECK:變異數恢復 (0)"
    assert dflog.err_text(12, 1) == "CRASH_CHECK:偵測到墜機 (1)"
    assert dflog.err_text(10, 5) == "FLIGHT_MODE:模式切換失敗 (5)"
    assert dflog.err_text(11, 7) == "GPS 代碼 7"
    assert dflog.err_text(99, 3) == "ERR 99 代碼 3"
    assert dflog.err_text(9, 4) == "FAILSAFE_FENCE:觸發 (4)"
    assert dflog.ERR_SUBSYS[1] == "MAIN" and dflog.ERR_SUBSYS[31] == "FAILSAFE_DEADRECKON"
    assert len(dflog.ERR_SUBSYS) == 31
    assert dflog.ev_text(18) == "LAND_COMPLETE" and dflog.ev_text(86) == "EK3_SOURCES_SET_TO_SECONDARY"
    assert dflog.ev_text(67) == "GPS_PRIMARY_CHANGED" and dflog.ev_text(62) == "EKF_YAW_RESET"
    assert dflog.ev_text(200) == "EV 200" and dflog.EV_NAMES[166] == "NOT_BOTTOMED"
    assert dflog.arm_text(True, 1) == "解鎖 ARMED(MAVLink)"
    assert dflog.arm_text(False, 99, forced=True) == "上鎖 DISARMED(強制)"
    print("  ERR 子系統 / 代碼說明、EV 名稱、ARM 方式 正確")


def test_log_messages():
    ld, info = sim.make_multi_flight_sim()
    with tempfile.TemporaryDirectory() as d:
        for arm_msg in (True, False):
            p = _bin(d, f"m{int(arm_msg)}.bin")
            fakebin.write_logdata_to_bin(ld, p, arm_msg=arm_msg)
            lm = LogIndex.open(p).log_messages()
            ts = [m.t for m in lm]
            assert ts == sorted(ts), "依時間排序"
            assert (lm[0].kind, lm[0].text) == ("msg", "ArduCopter V4.5.7 (2a3dc4b7)"), lm[0]
            assert (lm[1].kind, lm[1].text) == ("mode", "STABILIZE"), "同時刻依檔案順序"
            kinds = [m.kind for m in lm]
            assert kinds.count("arm") == 2 and kinds.count("disarm") == 2, "EV 10/11 不重複"
            assert not any(m.kind == "ev" and m.text in ("ARMED", "DISARMED") for m in lm)
            texts = [m.text for m in lm]
            for s in ("LAND_COMPLETE", "EK3_SOURCES_SET_TO_SECONDARY", "NOT_LANDED",
                      "EKFCHECK:變異數過大 (2)", "EKFCHECK:變異數恢復 (0)", "EKF3 IMU0 is using GPS",
                      "EKF3 IMU0 switching to ExternalNav", "Mission: 8 WP", "LOITER", "AUTO", "LAND"):
                assert s in texts, s
            arm_txt = [m.text for m in lm if m.kind == "arm"]
            assert arm_txt[0] == ("解鎖 ARMED(MAVLink)" if arm_msg else "解鎖 ARMED"), arm_txt
            modes = [m.text for m in lm if m.kind == "mode"]
            assert modes == ["STABILIZE"] + ["LOITER", "AUTO", "LAND", "STABILIZE"] * 2
            i_sw = texts.index("EK3_SOURCES_SET_TO_SECONDARY")
            assert lm[i_sw].kind == "ev" and lm[i_sw + 1].text.startswith("EKF3 IMU0 switching")
            if arm_msg:
                mem = ld.index.log_messages()          # 記憶體索引(模擬)與 .bin 讀回的一致
                assert [(m.kind, m.text) for m in mem] == [(m.kind, m.text) for m in lm]
    print(f"  {len(lm)} 則訊息:排序、同時刻順序、ARM/EV 去重、EV / ERR / MODE 文字正確")


def _vehicle_log(path: str, banner: str | None, mode_nums: list[int], ver_bu: int | None = None) -> None:
    w = fakebin.DFBinWriter()
    w.add_message_type("MSG", "QZ", "TimeUS,Message")
    w.add_message_type("VER", "QBB", "TimeUS,BT,BU")
    w.add_message_type("GPS", "QBBHILLe", "TimeUS,I,Status,GWk,GMS,Lat,Lng,Alt")
    w.add_message_type("MODE", "QMBB", "TimeUS,Mode,ModeNum,Rsn")
    if banner is not None:
        w.write("MSG", [1_000_000, banner])
    if ver_bu is not None:
        w.write("VER", [1_000_100, 0, ver_bu])
    for k in range(20):
        us = 1_100_000 + k * 200_000
        w.write("GPS", [us, 0, 3, 2320, 300000 + k * 200, 23.7 + k * 1e-5, 120.4, 40.0])
        if k < len(mode_nums):
            w.write("MODE", [us + 1, mode_nums[k], mode_nums[k], 1])
    w.save(path)


def test_vehicle_and_mode_names():
    cases = [("ArduCopter V4.5.7 (abc)", None, "copter", [5, 3, 9, 99], ["LOITER", "AUTO", "LAND", "MODE 99"]),
             ("ArduPlane V4.5.7 (abc)", None, "plane", [10, 12, 21], ["AUTO", "LOITER", "QRTL"]),
             ("ArduRover V4.5.7 (abc)", None, "rover", [0, 10, 11], ["MANUAL", "AUTO", "RTL"]),
             ("ArduSub V4.1.0", None, "sub", [0, 19], ["STABILIZE", "MANUAL"]),
             ("ArduCopter V4.5.7 heli", 13, "heli", [5, 6], ["LOITER", "RTL"]),
             ("Blimp V4.4.0", None, "blimp", [0, 1], ["LAND", "MANUAL"]),
             (None, None, "unknown", [10, 5], ["MODE 10", "MODE 5"])]   # 無機型 → 只顯示編號(不猜固定翼)
    with tempfile.TemporaryDirectory() as d:
        for i, (banner, bu, veh, nums, want) in enumerate(cases):
            p = _bin(d, f"v{i}.bin")
            _vehicle_log(p, banner, nums, bu)
            ix = LogIndex.open(p)
            assert ix.vehicle == veh, (banner, ix.vehicle)
            got = [m.text for m in ix.log_messages() if m.kind == "mode"]
            assert got == want, (banner, got)
            b = logio.load_bin(p)
            if veh == "unknown":     # 舊版沿用 DFReader 的固定翼表,這裡刻意不同
                assert any("無法判斷機型" in n for n in b.notes), b.notes
                assert [e.label for e in b.events if e.kind == "mode"] == want
                continue
            _same_logdata(logio._legacy_load_bin(p), b, f"機型 {veh}")
    assert LogIndex.from_tables({"MODE": {"t": [1.0], "Mode": [12], "ModeNum": [12]}},
                                vehicle="plane").mode_labels() == ["LOITER"]
    print("  機型(MSG / VER.BU)判斷與 MODE 名稱:copter / plane / rover / sub / heli / blimp 與 DFReader 一致;未知機型只顯示編號")


# ---------------- 往返 / 新舊等價 ----------------

def _tol(ch: str, ref: np.ndarray) -> float:
    if ch in ("L",):
        return 0.51e-7 + 1e-12
    if ch in ("e", "c", "C", "E"):
        return 0.0051
    if ch in ("f",):
        return float(np.max(np.abs(ref), initial=0.0)) * 1.2e-7 + 1e-12
    return 0.0


def test_from_tables_bin_roundtrip():
    """模擬索引 → .bin → 讀回:欄位、單位、instance、數值(格式量化內)、參數、訊息全部一致。"""
    for ld in (sim.make_sim_logdata(), sim.make_multi_flight_sim()[0]):
        ix = ld.index
        with tempfile.TemporaryDirectory() as d:
            p = _bin(d, "rt.bin")
            fakebin.write_logdata_to_bin(ld, p)
            lx = LogIndex.open(p)
            assert lx.vehicle == ix.vehicle == "copter"
            ma, mb = ix.messages(), lx.messages()
            assert list(ma) == list(mb), (list(ma), list(mb))
            for name in ma:
                a, b = ma[name], mb[name]
                assert (a.fields, a.count, a.instances) == (b.fields, b.count, b.instances), name
                assert a.units == b.units, (name, a.units, b.units)
                for inst in (a.instances or [None]):
                    ta, tb = ix.table(name, inst), lx.table(name, inst)
                    assert np.max(np.abs(ta["t"] - tb["t"]), initial=0) < 1e-6, name
                    for c in a.fields:
                        x, y = ta[c], tb[c]
                        if x.dtype.kind == "U":
                            assert x.tolist() == y.tolist(), (name, c)
                            continue
                        ch = lx.field_format(name, c)
                        err = float(np.max(np.abs(x.astype(float) - y.astype(float)), initial=0))
                        assert err <= _tol(ch, x), f"{name}.{c}({ch}):誤差 {err:.3e}"
                        if ch in "bBhHiIqQM":
                            assert x.dtype == y.dtype, (name, c, x.dtype, y.dtype)
            assert ix.field_keys() == lx.field_keys()
            assert [(q.name, q.value, q.default) for q in ix.params()] == \
                [(q.name, q.value, q.default) for q in lx.params()]
            la, lb = ix.log_messages(), lx.log_messages()
            assert [(m.kind, m.text) for m in la] == [(m.kind, m.text) for m in lb]
            assert max(abs(x.t - y.t) for x, y in zip(la, lb)) < 1e-6
        print(f"  {ld.path}:{len(ma)} 種訊息往返一致(格式量化內)")


def _messy_log(path: str) -> None:
    """各種邊界:非 3D-fix、重複 / 倒退時間戳、NaN、兩顆 GPS、XKF1 雙核心、Rst 欄、樣本不足、
    未除 1e7 的經緯度、ORGN Type 1/0、未知模式、只有 EV、ERR、垃圾位元組、截斷的最後一筆。"""
    w = fakebin.DFBinWriter()
    w.add_message_type("MSG", "QZ", "TimeUS,Message")
    w.add_message_type("GPS", "QBBHILLe", "TimeUS,I,Status,GWk,GMS,Lat,Lng,Alt")
    w.add_message_type("GPS2", "QBHIiif", "TimeUS,Status,GWk,GMS,Lat,Lng,Alt")
    w.add_message_type("AHR2", "QccCfLL", "TimeUS,Roll,Pitch,Yaw,Alt,Lat,Lng")
    w.add_message_type("XKF1", "QBccCfff", "TimeUS,C,Roll,Pitch,Yaw,PN,PE,PD")
    w.add_message_type("NKF1", "Qfff", "TimeUS,PN,PE,PD")
    w.add_message_type("VISP", "QfffB", "TimeUS,PX,PY,PZ,Rst")
    w.add_message_type("POS", "QLLf", "TimeUS,Lat,Lng,Alt")
    w.add_message_type("ORGN", "QBLLe", "TimeUS,Type,Lat,Lng,Alt")
    w.add_message_type("MODE", "QMBB", "TimeUS,Mode,ModeNum,Rsn")
    w.add_message_type("EV", "QB", "TimeUS,Id")
    w.add_message_type("ERR", "QBB", "TimeUS,Subsys,ECode")
    w.write("MSG", [900_000, "ArduCopter V4.5.7 (messy)"])
    w.write("ORGN", [950_000, 1, 23.70, 120.43, 39.0])
    us0 = 1_000_000
    for k in range(60):
        us = us0 + k * 100_000
        if k % 2 == 0:
            st = 1 if k < 10 else 3
            uu = us if k != 30 else us - 200_000          # 重複時間戳(與 k=28 同時)
            w.write("GPS", [uu, 0, st, 2320, 300000 + k * 100, 23.7 + k * 1e-5, 120.43, 40.0 + k * 0.1])
            w.write("GPS", [us + 3, 1, 3, 2320, 300000 + k * 100, 23.7 + k * 1.1e-5, 120.43, 41.0])
        if k % 3 == 0:
            uu = us if k != 21 else us - 450_000          # 倒退的時間戳
            alt = float("nan") if k == 33 else 42.0
            w.write("GPS2", [uu, 3, 2320, 300000 + k * 100, int((23.7 + k * 1e-5) * 1e7),
                             int(120.43 * 1e7), alt])
            w.write("AHR2", [us + 7, 1.5, -2.0, 90.0, 40.0, 23.7 + k * 1e-5, 120.43])
        w.write("XKF1", [us + 1, k % 2, 1.0, -1.0, 45.0, k * 0.1, -k * 0.05, -3.0])
        if k % 4 == 0:
            pn = float("nan") if k == 8 else k * 0.1
            w.write("NKF1", [us + 2, pn, 0.0, -2.0])
        rst = 0 if k < 25 else (1 if k < 45 else 2)
        px = float("nan") if k == 26 else k * 0.11
        w.write("VISP", [us + 4, px, 0.5, -1.0, rst])
        if k < 5:
            w.write("POS", [us + 5, 23.7, 120.43, 40.0])
        if k == 7:
            w.raw(b"\x00\xa3\x00garbage\xa3")               # 垃圾位元組
        if k in (3, 20, 40, 50):
            w.write("MODE", [us + 6, {3: 5, 20: 3, 40: 99, 50: 9}[k], {3: 5, 20: 3, 40: 99, 50: 9}[k], 1])
        if k in (4, 10, 30, 45, 52):
            w.write("EV", [us + 8, {4: 10, 10: 28, 30: 62, 45: 18, 52: 11}[k]])
        if k in (15, 18):
            w.write("ERR", [us + 9, 11, 2 if k == 15 else 0])
        if k == 12:
            w.write("ORGN", [us, 0, 23.71, 120.44, 38.0])
        if k == 14:
            w.write("ORGN", [us, 1, 23.72, 120.45, 37.0])
    w.write("VISP", [us0 + 7_000_000, 1.0, 2.0, 3.0, 2])
    w.buf = w.buf[:-5]                                       # 最後一筆被截斷
    w.save(path)


def test_load_bin_equivalence():
    """新版 load_bin == 舊版逐筆掃描:範例 log、舊寫法 log(有 / 無 ARM)、各種邊界。"""
    with tempfile.TemporaryDirectory() as d:
        paths = list(SAMPLES)
        legacy = sim.make_multi_flight_sim()[0]
        legacy.index = None
        for arm_msg in (True, False):
            p = _bin(d, f"legacy_{int(arm_msg)}.bin")
            fakebin.write_logdata_to_bin(legacy, p, arm_msg=arm_msg)
            paths.append(p)
        p = _bin(d, "messy.bin")
        _messy_log(p)
        paths.append(p)
        for p in paths:
            with _quiet_stderr():           # 垃圾位元組:pymavlink 會印 bad header
                a = logio._legacy_load_bin(p)
                b = logio.load_bin(p)
            _same_logdata(a, b, os.path.basename(p))
            assert b.index is not None and a.index is None
            new_kinds = sorted({e.kind for e in b.events} - set(_LEGACY_KINDS))
            print(f"  {os.path.basename(p)}:來源 {list(b.sources)}、{len(a.events)} 個事件一致"
                  + (f"(另有 {new_kinds})" if new_kinds else ""))
        with _quiet_stderr() as err:
            b = logio.load_bin(paths[-1])
        assert "bad header" in err["text"], "垃圾位元組應被略過(pymavlink 會回報)"
        assert list(b.sources) == ["AHR2", "GPS[0]", "GPS[1] (GPS2)", "GPS2", "NKF1", "VISP",
                                   "XKF1[0]", "XKF1[1]"], list(b.sources)
        assert any("GPS[0]:濾除 5 筆非 3D-fix" in n for n in b.notes), b.notes
        assert any("POS:樣本不足" in n for n in b.notes), b.notes
        assert abs(b.origin[0] - 23.71) < 1e-9 and b.origin_desc == "ORGN(EKF 原點)"
        assert abs(b.sources["GPS2"].lat[0] - 23.7) < 1e-6, "未除 1e7 的經緯度自動換算"
        ev = [(e.kind, e.label) for e in b.events]
        assert ("mode", "MODE 99") in ev and ev.count(("reset", "VISP")) == 2
        assert ("ev", "NOT_LANDED") in ev and ("ev", "EKF_YAW_RESET") in ev and ("ev", "LAND_COMPLETE") in ev
        assert ("err", "GPS:Glitch (2)") in ev and ("err", "GPS:已恢復 (0)") in ev
        assert sum(k == "arm" for k, _ in ev) == 1 and sum(k == "disarm" for k, _ in ev) == 1
        # 找不到位置訊息:錯誤訊息相同
        p = _bin(d, "nopos.bin")
        w = fakebin.DFBinWriter()
        w.add_message_type("MSG", "QZ", "TimeUS,Message")
        w.add_message_type("BARO", "Qff", "TimeUS,Alt,Press")
        w.write("MSG", [1, "ArduCopter V4.5.7"])
        for k in range(5):
            w.write("BARO", [10 + k, 1.0, 1000.0])
        w.save(p)
        errs = []
        for fn in (logio._legacy_load_bin, logio.load_bin):
            try:
                fn(p)
            except ValueError as e:
                errs.append(str(e))
        assert len(errs) == 2 and errs[0] == errs[1] and "BARO" in errs[0], errs
    print("  沒有位置訊息時錯誤訊息相同")


def test_extra_logs():
    """選用:DFLOG_EXTRA_LOGS="a.bin;b.bin" 指定真實 log,逐欄比對 DFReader 並比對新舊 load_bin。"""
    paths = [p for p in os.environ.get("DFLOG_EXTRA_LOGS", "").split(";") if p.strip()]
    if not paths:
        print("  略過(未設定 DFLOG_EXTRA_LOGS)")
        return
    for p in paths:
        with _quiet_stderr():
            nt, nc = _check_against_dfreader(p)
            t = time.perf_counter()
            b = logio.load_bin(p)
            t_new = time.perf_counter() - t
            t = time.perf_counter()
            a = logio._legacy_load_bin(p)
            t_old = time.perf_counter() - t
        _same_logdata(a, b, os.path.basename(p))
        print(f"  {os.path.basename(p)}({os.path.getsize(p) / 1e6:.1f} MB,{b.index.vehicle}):"
              f"{nt} 型別 / {nc} 欄位與 DFReader 一致;來源 {list(b.sources)};"
              f"載入 {t_old:.2f} s → {t_new:.2f} s")


def test_progress_and_file_lock():
    with tempfile.TemporaryDirectory() as d:
        p = _bin(d, "lock.bin")
        fakebin.write_logdata_to_bin(sim.make_sim_logdata(), p)
        seen = []
        ld = logio.load_bin(p, progress_cb=seen.append)
        assert seen and seen[-1] == 100 and seen == sorted(seen) and all(0 <= x <= 100 for x in seen)
        assert all(isinstance(x, int) for x in seen)
        ld.index.table("ATT")                 # 延遲解碼不需要檔案
        os.remove(p)                          # Windows:檔案沒被鎖住
        assert ld.index.table("CTUN")["t"].size > 0
    print(f"  進度回報 {seen}(遞增、以 100 結束);載入後檔案即可刪除,延遲解碼仍可用")


# ---------------- 模擬資料 ----------------

_SIM_FP = {   # 改版前的軌跡指紋(sha256 前 16 碼):輔助表格用另一組亂數,既有軌跡不能變
    ("single", 7): {"GPS": "2c545e7b1e436169", "VISP": "96421a4b85683818", "POS": "3fbe39ef8e82b20b"},
    ("single", 3): {"GPS": "f1286e6611fb8ba8", "VISP": "f7bae8c16d7e1fe5", "POS": "dfe71e082d1113dd"},
    ("multi", 11): {"GPS": "07f5945ecc1c98f1", "VISP": "3a253e8b44980ecc", "POS": "4fae3f0bee22f941"},
}


def _fp(src) -> str:
    h = hashlib.sha256()
    for a in (src.t, src.lat, src.lng, src.alt, src.ned):
        if a is not None:
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def test_sim_sources_unchanged():
    for (kind, seed), want in _SIM_FP.items():
        ld = sim.make_sim_with_truth(seed=seed)[0] if kind == "single" else sim.make_multi_flight_sim(seed)[0]
        got = {k: _fp(ld.sources[k]) for k in want}
        assert got == want, (kind, seed, got)
        assert list(ld.sources) == ["GPS", "VISP", "POS", "XKF1"], list(ld.sources)
        assert ld.index is not None and ld.index.vehicle == "copter"
        base = {k: v for k, v in ld.sources.items() if k in want}
        assert ld.t_base == min(s.t[0] for s in base.values()) and             ld.t_end == max(s.t[-1] for s in base.values()), "XKF1 / 輔助表格不可改變 log 時間範圍"
    assert (sim.make_sim_logdata().t_end, sim.make_multi_flight_sim()[0].t_end) ==         (291.9666666666667, 337.29999999999995), "與改版前相同"
    print("  單航段 seed 7 / 3、多航段 seed 11:GPS / VISP / POS 與改版前逐位元相同(另加 XKF1),"
          "時間範圍不變")


def test_sim_attitude_signs():
    ld, truth = sim.make_sim_with_truth(seed=7)
    ix = ld.index
    att = ix.table("ATT")
    t, roll, pitch, yaw = att["t"], att["Roll"], att["Pitch"], att["Yaw"]
    t_hover1, t_hover2 = [tt for n, tt in truth["phases"] if n == "hover"][:2]
    t_cruise = [tt for n, tt in truth["phases"] if n == "cruise"][0]
    cr = (t > t_cruise + 2) & (t < t_hover2 - 2)
    # 轉彎:航向變化率與 roll 同號(右轉 = 航向增加 = 正 roll);方形逆時針繞 → 左轉、roll 為負
    yr = np.gradient(np.degrees(np.unwrap(np.radians(yaw))), t)
    turning = cr & (np.abs(yr) > 30)
    assert turning.sum() > 20
    assert np.mean(np.sign(roll[turning]) == np.sign(yr[turning])) > 0.95
    assert np.median(roll[turning]) < -5, "逆時針方形 → 左轉 → 負 roll"
    # 開始巡航(向前加速)→ 低頭;結束巡航(減速)→ 抬頭
    acc_win = (t > t_cruise) & (t < t_cruise + 0.8)
    dec_win = (t > t_hover2 - 0.8) & (t < t_hover2)
    assert np.mean(pitch[acc_win]) < -3 and np.mean(pitch[dec_win]) > 3, \
        (np.mean(pitch[acc_win]), np.mean(pitch[dec_win]))
    # 直線段:Yaw ≈ GPS 對地航跡;懸停:航向保持
    g = ix.table("GPS")
    straight = cr & (np.abs(yr) < 2) & (np.abs(roll) < 2)
    gc = np.interp(t[straight], g["t"], np.degrees(np.unwrap(np.radians(g["GCrs"])))) % 360
    d = (yaw[straight] - gc + 180) % 360 - 180
    assert np.median(np.abs(d)) < 3, np.median(np.abs(d))
    hov = (t > t_hover1 + 0.5) & (t < t_cruise - 0.5)
    assert np.ptp(yaw[hov]) < 5 and abs(np.mean(yaw[hov]) - 90) < 5, "起飛後懸停:保持初始航向(東)"
    assert np.all((yaw >= 0) & (yaw < 360))
    # 上鎖時期望姿態為 0,解鎖後跟著實際姿態
    arm = np.zeros(t.size, bool)
    for s in ld.segments:
        arm |= (t >= s.t0) & (t < s.t1)
    assert np.all(att["DesRoll"][~arm] == 0) and np.corrcoef(att["DesRoll"][cr], roll[cr])[0, 1] > 0.9
    ctun = ix.table("CTUN")
    air = ctun["Alt"] > 1
    assert 0.3 < np.median(ctun["ThO"][air]) < 0.5 and np.all(ctun["ThO"][ctun["t"] < ld.segments[0].t0] == 0)
    bat = ix.table("BAT")
    assert np.all(np.diff(bat["CurrTot"]) >= 0) and bat["Volt"][-1] < bat["Volt"][0]
    print(f"  轉彎 roll 與航向率同號、加速低頭 / 減速抬頭、直線段 Yaw≈GCrs(中位差 "
          f"{np.median(np.abs(d)):.2f}°)、懸停保持航向;油門 / 電池合理")


def test_sim_events_match_load_bin():
    for ld in (sim.make_sim_logdata(), sim.make_multi_flight_sim()[0]):
        kinds = {e.kind for e in ld.events}
        assert EV_EVENT in kinds and EV_ERROR in kinds, kinds
        with tempfile.TemporaryDirectory() as d:
            p = _bin(d, "ev.bin")
            fakebin.write_logdata_to_bin(ld, p)
            got = logio.load_bin(p)
        a = [(e.kind, e.label) for e in ld.events]
        b = [(e.kind, e.label) for e in got.events]
        assert a == b, f"\n{a}\n{b}"
        for x, y in zip(ld.events, got.events):
            if x.kind == EV_RESET:      # 讀回的是重置後第一筆 VISP 的時間
                assert 0 <= y.t - x.t <= 1.0 / 30 + 1e-6, (x, y)
            else:
                assert abs(y.t - x.t) <= 1e-6, (x, y)
        assert list(got.sources) == ["GPS", "POS", "VISP", "XKF1"], list(got.sources)
        x0, x1 = ld.sources["XKF1"], got.sources["XKF1"]
        assert np.array_equal(x0.ned, x1.ned) and np.max(np.abs(x0.t - x1.t)) < 1e-6
        evs = [e.label for e in ld.events if e.kind in (EV_EVENT, EV_ERROR)]
        print(f"  {ld.path}:{len(ld.events)} 個事件與 load_bin(fakebin) 一致,含 {evs}")


def test_sample_bins_current():
    """output/ 的範例 .bin 是用目前的模擬資料產生的(讀回的索引與記憶體中的相同)。"""
    for p, ld in zip(SAMPLES, (sim.make_sim_logdata(), sim.make_multi_flight_sim()[0])):
        lx = LogIndex.open(p)
        assert list(lx.messages()) == list(ld.index.messages()), os.path.basename(p)
        a, b = lx.table("ATT"), ld.index.table("ATT")
        assert np.array_equal(a["Roll"], b["Roll"]), f"{os.path.basename(p)} 過期,請重新產生"
        print(f"  {os.path.basename(p)}:{len(lx.messages())} 種訊息,與目前模擬一致")


# ---------------- 大檔載入時間 ----------------

def _big_tables(minutes: float = 30.0, seed: int = 0) -> dict:
    """約 30 分鐘、多次起降的「開機到關機」log:ATT 25 Hz、IMU×2 50 Hz、XKF1×2 / CTUN / BAT /
    VIBE×2 / RCIN / RCOU / BARO / MAG 10 Hz、GPS 5 Hz、POS 10 Hz、VISP 30 Hz。"""
    rng = np.random.default_rng(seed)
    T = minutes * 60.0
    t0 = 100.0

    def grid(hz):
        return t0 + np.arange(int(T * hz)) / hz

    def walk(n, s=0.05):
        return np.cumsum(rng.normal(0, s, n))

    tb = {"MSG": {"t": np.array([t0]), "Message": np.array(["ArduCopter V4.5.7 (big)"])}}
    arms = t0 + np.arange(60.0, T - 120, 300.0)
    tb["ARM"] = {"t": np.sort(np.concatenate([arms, arms + 200])),
                 "ArmState": np.tile([1, 0], arms.size).astype(np.uint8)}
    tb["EV"] = {"t": tb["ARM"]["t"].copy(), "Id": np.tile([10, 11], arms.size).astype(np.uint8)}
    mt = np.sort(np.concatenate([arms - 1, arms + 20, arms + 190]))
    tb["MODE"] = {"t": mt, "Mode": np.tile([5, 3, 9], arms.size).astype(np.int8),
                  "ModeNum": np.tile([5, 3, 9], arms.size).astype(np.uint8)}
    tb["PARM"] = {"t": t0 + 0.001 * np.arange(800),
                  "Name": np.array([f"P_{i:04d}" for i in range(800)]), "Value": rng.normal(0, 100, 800)}
    t = grid(25)
    tb["ATT"] = {"t": t, **{c: walk(t.size) for c in ("DesRoll", "Roll", "DesPitch", "Pitch", "DesYaw", "Yaw")}}
    tb["RATE"] = {"t": t, **{c: walk(t.size) for c in ("RDes", "R", "ROut", "PDes", "P", "POut", "YDes", "Y", "YOut")}}
    t = grid(50)
    for i in (0, 1):
        tb[f"IMU[{i}]"] = {"t": t, "I": np.full(t.size, i, np.uint8),
                           **{c: rng.normal(0, 1, t.size) for c in ("GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ")},
                           "EG": np.zeros(t.size, np.uint32), "EA": np.zeros(t.size, np.uint32),
                           "T": np.full(t.size, 40.0)}
    t = grid(10)
    for i in (0, 1):
        tb[f"XKF1[{i}]"] = {"t": t, "C": np.full(t.size, i, np.uint8), "Roll": walk(t.size), "Pitch": walk(t.size),
                            "Yaw": walk(t.size) % 360, "VN": walk(t.size), "VE": walk(t.size), "VD": walk(t.size),
                            "PN": walk(t.size), "PE": walk(t.size), "PD": walk(t.size)}
        tb[f"VIBE[{i}]"] = {"t": t, "IMU": np.full(t.size, i, np.uint8),
                            **{c: np.abs(walk(t.size)) for c in ("VibeX", "VibeY", "VibeZ")},
                            "Clip": np.zeros(t.size, np.uint32)}
    for name, cols in (("CTUN", ("ThI", "ABst", "ThO", "ThH", "DAlt", "Alt", "BAlt")),
                       ("BARO", ("Alt", "Press", "Temp", "CRt")), ("MAG", ("MagX", "MagY", "MagZ")),
                       ("PSCN", ("TPN", "PN", "DVN", "TVN", "VN", "DAN", "TAN", "AN")),
                       ("PSCE", ("TPE", "PE", "DVE", "TVE", "VE", "DAE", "TAE", "AE"))):
        tb[name] = {"t": t, **{c: walk(t.size) for c in cols}}
    tb["BAT[0]"] = {"t": t, "Inst": np.zeros(t.size, np.uint8), "Volt": 16 - np.abs(walk(t.size, 0.001)),
                    "Curr": np.abs(walk(t.size)), "CurrTot": np.cumsum(np.full(t.size, 0.3))}
    for name in ("RCIN", "RCOU"):
        tb[name] = {"t": t, **{f"C{k}": (1500 + rng.integers(-400, 400, t.size)).astype(np.uint16) for k in range(1, 15)}}
    tb["POS"] = {"t": t, "Lat": 23.70 + walk(t.size, 1e-6), "Lng": 120.43 + walk(t.size, 1e-6),
                 "Alt": 40 + walk(t.size)}
    t = grid(5)
    tb["GPS[0]"] = {"t": t, "I": np.zeros(t.size, np.uint8), "Status": np.full(t.size, 3, np.uint8),
                    "GMS": (300000 + np.arange(t.size) * 200).astype(np.uint32),
                    "GWk": np.full(t.size, 2320, np.uint16), "NSats": np.full(t.size, 15, np.uint8),
                    "HDop": np.full(t.size, 0.8), "Lat": 23.70 + walk(t.size, 1e-6),
                    "Lng": 120.43 + walk(t.size, 1e-6), "Alt": 40 + walk(t.size), "Spd": np.abs(walk(t.size))}
    t = grid(30)
    tb["VISP"] = {"t": t, "PX": walk(t.size), "PY": walk(t.size), "PZ": walk(t.size),
                  "ResetCnt": (np.searchsorted(arms, t) % 256).astype(np.uint8)}
    return tb


def test_timing_large_log():
    minutes = float(os.environ.get("DFLOG_BIG_MINUTES", "30"))
    with tempfile.TemporaryDirectory() as d:
        p = _bin(d, "big.bin")
        t = time.perf_counter()
        ix = LogIndex.from_tables(_big_tables(minutes))
        fakebin.write_index_to_bin(ix, p)
        t_gen = time.perf_counter() - t
        mb = os.path.getsize(p) / 1e6
        n_msgs = sum(m.count for m in ix.messages().values())

        t = time.perf_counter()
        new = logio.load_bin(p)
        t_new = time.perf_counter() - t
        t = time.perf_counter()
        for name, info in new.index.messages().items():
            for inst in (info.instances or [None]):
                new.index.table(name, inst)
        t_all = time.perf_counter() - t
        t = time.perf_counter()
        old = logio._legacy_load_bin(p)
        t_old = time.perf_counter() - t
        _same_logdata(old, new, "大檔")
    print(f"  {minutes:.0f} 分鐘合成 log:{mb:.1f} MB、{n_msgs:,} 筆、{len(ix.messages())} 種訊息"
          f"(產生 {t_gen:.1f} s)")
    print(f"  載入時間:舊版逐筆 {t_old:.2f} s → 新版 {t_new:.2f} s(快 {t_old / t_new:.0f} 倍);"
          f"再把全部 {sum(len(i.instances or [None]) for i in new.index.messages().values())} 張表解碼 "
          f"{t_all:.2f} s;來源 / 事件與舊版逐位元相同")
    assert t_new < t_old


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  !! 失敗:{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 60}\n{len(tests) - failed}/{len(tests)} 通過")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
