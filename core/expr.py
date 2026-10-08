"""欄位運算式:把 ``ATT.Roll``、``degrees(atan2(VISP.PY, VISP.PX))``、``GPS[1].Spd*3.6``
這類字串安全地算成時間序列(t, v)。

做法:先用正規式找出欄位引用(訊息名大寫字母開頭、可帶 ``[instance]``),換成
使用者打不出來的內部變數名,再用 ``ast`` 解析並嚴格白名單檢查 —— 只允許數值常數、
``+ - * / ** % //``、正負號、比較(結果 0/1)、白名單函式呼叫。屬性、索引、lambda、
推導式、字串、關鍵字參數… 一律拒絕;只解析成語法樹自行求值,不執行任何 Python 程式碼。

時間基準 = 第一個引用欄位的取樣時間;其他欄位線性內插到這個時間上,
超出該欄位自身時間範圍、或落在它的資料空洞(> max(10×中位取樣間隔, 1 s))裡為 NaN,
不內插出假資料。全部引用都來自同一訊息時直接用原始樣本。航向欄位(單位 degheading,
或沒有單位時的 Yaw / GCrs / YC… 等欄位名)跨 0/360 時走最短弧內插,維持欄位自己的
0~360 或 ±180 範圍。

``index`` 為 duck type:需有 ``series(key)``、``has(key)``、``field_keys()``、``messages()``
(見 core/dflog.py 的 LogIndex)。
"""
from __future__ import annotations

import ast
import difflib
import functools
import re
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["ExprError", "evaluate", "referenced_keys", "validate", "resolve_key",
           "FUNCTION_HELP", "CONSTANTS", "MAX_LEN"]

MAX_LEN = 1000          # 運算式字元上限
MAX_NODES = 400         # AST 節點上限(防止病態巢狀 / 超長運算式)
GAP_FACTOR = 10.0       # 內插:間隔 > GAP_FACTOR × 中位取樣間隔 視為資料空洞
GAP_MIN_S = 1.0         # …但至少 1 秒才算空洞


class ExprError(ValueError):
    """運算式錯誤(訊息為給使用者看的繁中說明)。"""


# --- 欄位引用 ---------------------------------------------------------------
# 訊息名:大寫字母開頭的大寫 / 數字 / 底線;可帶 [instance];欄位:英數底線。
# 前面不能緊接英數、底線或 '.',避免吃到 1E5.x、foo.ATT.Roll 這類片段。
_REF_RE = re.compile(r"(?<![A-Za-z0-9_.])([A-Z][A-Z0-9_]*)(?:\[\s*(\d+)\s*\])?"
                     r"\.([A-Za-z_][A-Za-z0-9_]*)")
_KEY_RE = re.compile(r"^([A-Z][A-Z0-9_]*)(?:\[(\d+)\])?(?:\.([A-Za-z_][A-Za-z0-9_]*))?$")
_UNDERSCORE_NAME_RE = re.compile(r"(?<![A-Za-z0-9_])_[A-Za-z0-9_]*")
_VAR_FMT = "_F{}_"      # 內部變數名(以底線開頭,使用者輸入的底線名稱一律拒絕)


def _norm_key(msg: str, inst: str | int | None, field: str) -> str:
    return f"{msg}[{int(inst)}].{field}" if inst not in (None, "") else f"{msg}.{field}"


def _split_key(key: str) -> tuple[str, int | None, str | None]:
    m = _KEY_RE.match(key)
    if not m:
        return key, None, None
    return m.group(1), (int(m.group(2)) if m.group(2) is not None else None), m.group(3)


# --- 函式表 -----------------------------------------------------------------
def _wrap_360(x):
    r = np.mod(x, 360.0)
    return np.where(r >= 360.0, r - 360.0, r)      # np.mod(-1e-20, 360) 會給 360.0


def _wrap_180(x):
    r = _wrap_360(x)
    return np.where(r > 180.0, r - 360.0, r)        # 與 ArduPilot wrap_180 相同:(-180, 180]


def _mag(*xs):
    acc = np.square(xs[0])
    for x in xs[1:]:
        acc = acc + np.square(x)
    return np.sqrt(acc)


def _rel(x):
    """減去第一個有效值(相對起點的變化量)。"""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        return a * 0.0
    ok = np.flatnonzero(np.isfinite(a))
    return a - a[ok[0]] if ok.size else a


def ddt(t: np.ndarray, x) -> np.ndarray:
    """對時間(秒)微分。內部點用非等距二階中央差分(對二次函數精確),
    端點與鄰點缺值時退回單側差分;重複時間戳不產生 inf;缺值只影響自己。"""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        return np.float64(0.0) if np.isfinite(x) else np.float64(np.nan)
    n = x.size
    out = np.full(n, np.nan)
    if n < 2:
        return out
    with np.errstate(all="ignore"):
        h = np.diff(t)
        s = np.diff(x) / h                       # 區間 i→i+1 的斜率
        s[~(h > 0)] = np.nan
        if n > 2:
            hb, hf, sb, sf = h[:-1], h[1:], s[:-1], s[1:]
            c = (hf * sb + hb * sf) / (hb + hf)
            one = np.where(np.isfinite(sb), sb, sf)
            out[1:-1] = np.where(np.isfinite(c), c, one)
    out[0] = s[0]
    out[-1] = s[-1]
    out[~np.isfinite(x)] = np.nan
    return out


_LP_SPAN = 300.0        # lowpass 區塊的時間跨度上限(以 τ 計);exp(300) ≈ 2e130 仍遠低於溢位


def _lowpass_loop(tt: np.ndarray, xx: np.ndarray, tau: float) -> np.ndarray:
    k = (-np.expm1(-np.diff(tt) / tau)).tolist()
    y = float(xx[0])
    out = [y]
    for ki, xi in zip(k, xx[1:].tolist()):
        y += ki * (xi - y)
        out.append(y)
    return np.asarray(out, dtype=np.float64)


def lowpass(t: np.ndarray, x, tau: float) -> np.ndarray:
    """一階低通(時間常數 tau 秒),依實際取樣間隔:y += (1 − e^(−dt/τ))·(x − y)。

    以第一個有效值起始;NaN 樣本輸出 NaN 並保持狀態(下一個有效樣本以距上一個
    有效樣本的 dt 遞推)。向量化:每段 ≤ 300τ 的區塊內把遞迴改寫成
    z = y·e^((t−T)/τ) 的累加和,區塊間只傳一個狀態值。"""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        return x
    out = np.full(x.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(x) & np.isfinite(t))
    if idx.size == 0:
        return out
    tt, xx = t[idx], x[idx]
    if tau < 1e-9:                     # τ → 0:濾波器等於直通
        out[idx] = xx
        return out
    with np.errstate(all="ignore"):
        bid = np.floor((tt - tt[0]) / (tau * _LP_SPAN))
        starts = np.flatnonzero(np.r_[True, bid[1:] != bid[:-1]])
        if starts.size > 64 + idx.size // 32:            # 區塊太碎(τ ≪ 取樣間隔):逐點遞推
            out[idx] = _lowpass_loop(tt, xx, tau)
            return out
        y = np.empty_like(xx)
        ends = np.r_[starts[1:], xx.size]
        prev_y, prev_t = float(xx[0]), float(tt[0])
        for s, e in zip(starts.tolist(), ends.tolist()):
            tb, xb = tt[s:e], xx[s:e]
            y0 = prev_y + (-np.expm1(-(tb[0] - prev_t) / tau)) * (xb[0] - prev_y)
            if e - s == 1:
                y[s] = y0
            else:
                ex = np.exp((tb - tb[0]) / tau)
                k = -np.expm1(-np.diff(tb) / tau)        # 1 − a_i
                z = np.empty(e - s)
                z[0] = y0
                np.cumsum(k * ex[1:] * xb[1:], out=z[1:])
                z[1:] += y0
                y[s:e] = z / ex
            prev_y, prev_t = float(y[e - 1]), float(tb[-1])
    out[idx] = y
    return out


def _reduce(fn):
    return lambda *xs: functools.reduce(fn, xs)


@dataclass(frozen=True)
class _Func:
    fn: Callable
    nmin: int
    nmax: int | None            # None = 不限
    timed: bool = False         # 第一個參數需要時間基準 t
    sig: str = ""
    doc: str = ""


_FUNCS: dict[str, _Func] = {
    "abs": _Func(np.abs, 1, 1, sig="abs(x)", doc="絕對值"),
    "sqrt": _Func(np.sqrt, 1, 1, sig="sqrt(x)", doc="平方根(負數為 NaN)"),
    "sin": _Func(np.sin, 1, 1, sig="sin(x)", doc="正弦(弧度)"),
    "cos": _Func(np.cos, 1, 1, sig="cos(x)", doc="餘弦(弧度)"),
    "tan": _Func(np.tan, 1, 1, sig="tan(x)", doc="正切(弧度)"),
    "asin": _Func(np.arcsin, 1, 1, sig="asin(x)", doc="反正弦(弧度)"),
    "acos": _Func(np.arccos, 1, 1, sig="acos(x)", doc="反餘弦(弧度)"),
    "atan": _Func(np.arctan, 1, 1, sig="atan(x)", doc="反正切(弧度)"),
    "atan2": _Func(np.arctan2, 2, 2, sig="atan2(y, x)", doc="四象限反正切(弧度)"),
    "degrees": _Func(np.degrees, 1, 1, sig="degrees(x)", doc="弧度 → 度"),
    "radians": _Func(np.radians, 1, 1, sig="radians(x)", doc="度 → 弧度"),
    "hypot": _Func(np.hypot, 2, 2, sig="hypot(x, y)", doc="√(x² + y²)"),
    "mag": _Func(_mag, 1, None, sig="mag(x, y, …)", doc="向量長度 √(Σx²)"),
    "min": _Func(_reduce(np.minimum), 2, None, sig="min(a, b, …)", doc="逐點最小值"),
    "max": _Func(_reduce(np.maximum), 2, None, sig="max(a, b, …)", doc="逐點最大值"),
    "log": _Func(np.log, 1, 1, sig="log(x)", doc="自然對數"),
    "log10": _Func(np.log10, 1, 1, sig="log10(x)", doc="常用對數"),
    "exp": _Func(np.exp, 1, 1, sig="exp(x)", doc="e 的次方"),
    "floor": _Func(np.floor, 1, 1, sig="floor(x)", doc="無條件捨去"),
    "ceil": _Func(np.ceil, 1, 1, sig="ceil(x)", doc="無條件進位"),
    "sign": _Func(np.sign, 1, 1, sig="sign(x)", doc="正負號(−1 / 0 / 1)"),
    "wrap_180": _Func(_wrap_180, 1, 1, sig="wrap_180(deg)", doc="角度包到 (−180, 180]"),
    "wrap_360": _Func(_wrap_360, 1, 1, sig="wrap_360(deg)", doc="角度包到 [0, 360)"),
    "clip": _Func(np.clip, 3, 3, sig="clip(x, lo, hi)", doc="限制在 lo~hi 之間"),
    "rel": _Func(_rel, 1, 1, sig="rel(x)", doc="減去第一個有效值(相對起點的變化)"),
    "ddt": _Func(ddt, 1, 1, timed=True, sig="ddt(x)", doc="對時間微分(每秒變化量)"),
    "lowpass": _Func(lowpass, 2, 2, timed=True, sig="lowpass(x, τ秒)",
                     doc="一階低通濾波,τ 為時間常數(秒)"),
}

CONSTANTS: dict[str, float] = {"pi": float(np.pi), "e": float(np.e), "g": 9.80665}

# UI 說明用:(簽名, 說明)
FUNCTION_HELP: list[tuple[str, str]] = (
    [(f.sig, f.doc) for f in _FUNCS.values()]
    + [("pi", "圓周率"), ("e", "自然常數"), ("g", "重力加速度 9.80665 m/s²")])

_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.true_divide, ast.Pow: np.power, ast.Mod: np.mod,
           ast.FloorDiv: np.floor_divide}
_CMPOPS = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
           ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}
_OP_SYMBOL = {ast.MatMult: "@", ast.LShift: "<<", ast.RShift: ">>", ast.BitOr: "|",
              ast.BitAnd: "&", ast.BitXor: "^", ast.Invert: "~", ast.Not: "not",
              ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in"}
_NODE_DESC = {
    ast.Attribute: "屬性存取「.」", ast.Subscript: "索引「[ ]」(instance 請寫成 GPS[1].Spd)",
    ast.Lambda: "lambda", ast.ListComp: "推導式", ast.SetComp: "推導式",
    ast.DictComp: "推導式", ast.GeneratorExp: "推導式", ast.List: "串列「[ ]」",
    ast.Tuple: "多個值 / 元組(函式參數請放在括號內)", ast.Set: "集合「{ }」",
    ast.Dict: "字典「{ }」", ast.IfExp: "條件運算式 if / else", ast.BoolOp: "and / or",
    ast.NamedExpr: "指派「:=」", ast.JoinedStr: "f 字串", ast.FormattedValue: "f 字串",
    ast.Starred: "「*」展開", ast.Slice: "切片", ast.Await: "await",
    ast.Yield: "yield", ast.YieldFrom: "yield",
}


# --- 編譯(解析 + 白名單) ----------------------------------------------------
@dataclass(frozen=True)
class _Program:
    source: str
    tree: ast.Expression
    keys: tuple[str, ...]               # 實際用到的欄位,依文字出現順序、不重複
    var_key: tuple[tuple[str, str], ...]  # (內部變數名, 欄位 key)
    consts: dict                        # id(Constant 節點) → np.float64


def _syntax_reason(msg: str) -> str:
    m = msg.lower()
    if "never closed" in m or "was not closed" in m:
        return "括號沒有閉合"
    if "unmatched" in m or "does not match" in m:
        return "括號不成對"
    if "unexpected eof" in m or "unexpected end" in m:
        return "運算式不完整"
    if "nested" in m:
        return "括號巢狀太深"
    if "null" in m:
        return "含有無效字元"
    if "invalid character" in m or "invalid decimal" in m or "invalid digit" in m:
        return "含有無效字元或數字"
    if "perhaps you forgot a comma" in m:
        return "兩個值之間缺少運算子或逗號"
    return "語法不正確"


def _substitute(src: str):
    """欄位引用 → 內部變數名。回傳 (新字串, [(變數名, key)], 位置對照)。"""
    parts: list[str] = []
    var_of: dict[str, str] = {}
    order: list[tuple[str, str]] = []
    spans: list[tuple[int, int, int]] = []    # (新字串起點, 新字串終點, 原字串起點)
    pos = 0
    out_len = 0
    for m in _REF_RE.finditer(src):
        gap = src[pos:m.start()]
        bad = _UNDERSCORE_NAME_RE.search(gap)
        if bad:
            raise ExprError(f"不允許以底線開頭的名稱「{bad.group(0)}」")
        parts.append(gap)
        out_len += len(gap)
        key = _norm_key(m.group(1), m.group(2), m.group(3))
        if key not in var_of:
            var_of[key] = _VAR_FMT.format(len(var_of))
            order.append((var_of[key], key))
        name = var_of[key]
        spans.append((out_len, out_len + len(name), m.start()))
        parts.append(name)
        out_len += len(name)
        pos = m.end()
    tail = src[pos:]
    bad = _UNDERSCORE_NAME_RE.search(tail)
    if bad:
        raise ExprError(f"不允許以底線開頭的名稱「{bad.group(0)}」")
    parts.append(tail)
    return "".join(parts), order, spans


def _close(word: str, choices, n: int = 3, cutoff: float = 0.6) -> list[str]:
    return difflib.get_close_matches(word, list(choices), n=n, cutoff=cutoff)


def _fmt_list(items) -> str:
    return "、".join(items)


class _Checker:
    def __init__(self, vars_: set[str]):
        self.vars = vars_
        self.used: set[str] = set()
        self.consts: dict = {}

    def check(self, node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            self.check(node.body)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOPS:
                raise ExprError(f"不支援的運算子「{_OP_SYMBOL.get(type(node.op), '?')}」")
            self.check(node.left)
            self.check(node.right)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise ExprError(f"不支援的運算子「{_OP_SYMBOL.get(type(node.op), '?')}」")
            self.check(node.operand)
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                if type(op) not in _CMPOPS:
                    raise ExprError(f"不支援的比較「{_OP_SYMBOL.get(type(op), '?')}」"
                                    "(可用 < <= > >= == !=)")
            self.check(node.left)
            for c in node.comparators:
                self.check(c)
        elif isinstance(node, ast.Call):
            self._check_call(node)
        elif isinstance(node, ast.Name):
            self._check_name(node)
        elif isinstance(node, ast.Constant):
            self._check_const(node)
        elif isinstance(node, ast.Attribute):
            raise ExprError(self._attr_msg(node))
        else:
            desc = _NODE_DESC.get(type(node), type(node).__name__)
            raise ExprError(f"不允許的語法:{desc}")

    def _check_call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            if isinstance(node.func, ast.Attribute):
                raise ExprError(self._attr_msg(node.func))
            if type(node.func) in _NODE_DESC:
                raise ExprError(f"不允許的語法:{_NODE_DESC[type(node.func)]}")
            raise ExprError("只能呼叫內建函式(例如 sqrt(x)、atan2(y, x))")
        name = node.func.id
        if name in self.vars:
            raise ExprError("欄位不能當函式呼叫")
        if name in CONSTANTS:
            raise ExprError(f"常數「{name}」不能當函式呼叫")
        f = _FUNCS.get(name)
        if f is None:
            sugg = _close(name, _FUNCS, cutoff=0.5)
            hint = f",是不是要用 {_fmt_list(s + '()' for s in sugg)}?" if sugg else ""
            raise ExprError(f"不支援的函式「{name}」{hint}")
        if node.keywords:
            raise ExprError(f"函式 {name}() 不接受具名參數")
        for a in node.args:
            if isinstance(a, ast.Starred):
                raise ExprError("不允許的語法:「*」展開")
        n = len(node.args)
        if n < f.nmin or (f.nmax is not None and n > f.nmax):
            if f.nmax is None:
                need = f"至少需要 {f.nmin} 個"
            elif f.nmin == f.nmax:
                need = f"需要 {f.nmin} 個"
            else:
                need = f"需要 {f.nmin}~{f.nmax} 個"
            raise ExprError(f"函式 {f.sig} {need}參數,收到 {n} 個")
        for a in node.args:
            self.check(a)

    def _check_name(self, node: ast.Name) -> None:
        name = node.id
        if name in self.vars:
            self.used.add(name)
            return
        if name in CONSTANTS:
            return
        if name in _FUNCS:
            raise ExprError(f"函式「{name}」要加括號呼叫,例如 {_FUNCS[name].sig}")
        if name.isupper():
            raise ExprError(f"「{name}」少了欄位名稱,請寫成 {name}.欄位(例如 ATT.Roll)")
        raise ExprError(f"不認得的名稱「{name}」。欄位請寫成 訊息.欄位(例如 ATT.Roll)")

    def _check_const(self, node: ast.Constant) -> None:
        v = node.value
        if isinstance(v, bool) or v is None or v is Ellipsis:
            raise ExprError(f"不允許的常數「{v}」(只能用數字)")
        if isinstance(v, (str, bytes)):
            raise ExprError("不允許字串")
        if isinstance(v, complex):
            raise ExprError("不支援複數")
        if not isinstance(v, (int, float)):
            raise ExprError("只能用數字常數")
        try:
            fv = float(v)
        except (OverflowError, ValueError):
            raise ExprError("數值常數超出範圍") from None
        self.consts[id(node)] = np.float64(fv)

    @staticmethod
    def _attr_msg(node: ast.Attribute) -> str:
        base = node.value
        if isinstance(base, ast.Name) and not base.id.startswith("_"):
            name = base.id
            if name.upper() != name:
                return (f"訊息名稱要全大寫:「{name}.{node.attr}」應寫成 "
                        f"{name.upper()}.{node.attr}")
            return f"欄位引用中間不能有空白,請寫成 {name}.{node.attr}"
        return "不允許的語法:屬性存取「.」(欄位請寫成 訊息.欄位,例如 ATT.Roll)"


@functools.lru_cache(maxsize=512)
def _compile(expr: str) -> _Program:
    if not isinstance(expr, str):
        raise ExprError("運算式必須是文字")
    src = expr.strip()
    if not src:
        raise ExprError("運算式是空的")
    if len(src) > MAX_LEN:
        raise ExprError(f"運算式太長({len(src)} 字元,上限 {MAX_LEN})")
    if "\x00" in src:
        raise ExprError("運算式含有無效字元")
    src = re.sub(r"[\r\n\t]", " ", src)
    sub, order, spans = _substitute(src)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # 例如字串裡的無效跳脫字元
            tree = ast.parse(sub, mode="eval")
    except SyntaxError as e:
        col = _orig_col(e.offset, spans, src)
        where = f"(第 {col} 個字元附近)" if col else ""
        raise ExprError(f"語法錯誤:{_syntax_reason(str(e.msg))}{where}") from None
    except (RecursionError, MemoryError, ValueError, OverflowError):
        raise ExprError("運算式太複雜") from None
    n_nodes = sum(1 for _ in ast.walk(tree))
    if n_nodes > MAX_NODES:
        raise ExprError(f"運算式太複雜({n_nodes} 個節點,上限 {MAX_NODES})")
    chk = _Checker({v for v, _ in order})
    try:
        chk.check(tree)
    except RecursionError:
        raise ExprError("運算式太複雜") from None
    keys = tuple(k for v, k in order if v in chk.used)
    if not keys:
        raise ExprError("運算式至少要引用一個欄位(例如 ATT.Roll)")
    var_key = tuple((v, k) for v, k in order if v in chk.used)
    return _Program(src, tree, keys, var_key, chk.consts)


def _orig_col(offset: int | None, spans, src: str) -> int | None:
    """SyntaxError.offset(1 起,替換後字串)→ 原字串的欄位(1 起)。"""
    if not offset:
        return None
    off = offset - 1
    shift = 0
    for s, e, o in spans:
        if off < s:
            break
        if off < e:
            return o + 1
        m = _REF_RE.match(src, o)
        orig_len = (m.end() - m.start()) if m else (e - s)
        shift = (o + orig_len) - e
    return min(off + shift, len(src)) + 1


# --- 取資料 -----------------------------------------------------------------
def _messages(index) -> dict:
    try:
        return index.messages() or {}
    except Exception:  # noqa: BLE001 — 只用來給提示,失敗就當沒有
        return {}


def _alt_keys(index, key: str, msgs: dict | None = None) -> list[str]:
    """沒有指定 instance 的訊息若有多個 instance → 退到最小的 instance;
    反之 [0] 對沒有 instance 的訊息 → 退到無 instance 寫法。"""
    msg, inst, field = _split_key(key)
    if field is None:
        return []
    info = (_messages(index) if msgs is None else msgs).get(msg)
    insts = list(getattr(info, "instances", None) or []) if info is not None else []
    if inst is None and insts:
        return [f"{msg}[{min(insts)}].{field}"]
    if inst == 0 and info is not None and not insts:
        return [f"{msg}.{field}"]
    return []


def resolve_key(index, key: str, *, messages: dict | None = None) -> str | None:
    """欄位 key → index 實際可查的 key(含 instance 退路);沒有則 None。不載入資料。
    大量查詢時可傳入預先取好的 ``index.messages()``。"""
    if index is None:
        return None
    try:
        if index.has(key):
            return key
        for alt in _alt_keys(index, key, messages):
            if index.has(alt):
                return alt
    except Exception:  # noqa: BLE001
        return None
    return None


def _unknown_field_msg(index, key: str) -> str:
    msg, inst, field = _split_key(key)
    msgs = _messages(index)
    try:
        keys = list(index.field_keys())
    except Exception:  # noqa: BLE001
        keys = []
    info = msgs.get(msg)
    if info is not None and field is not None:
        fields = list(getattr(info, "fields", []) or [])
        insts = list(getattr(info, "instances", None) or [])
        if inst is not None and insts and inst not in insts:
            return (f"訊息 {msg} 沒有 instance {inst}(可用:"
                    f"{_fmt_list(f'{msg}[{i}]' for i in insts)})")
        if field in fields:
            return f"欄位「{key}」不是數值資料,無法運算"
        lower = {f.lower(): f for f in fields}
        sugg = [lower[field.lower()]] if field.lower() in lower else _close(field, fields, cutoff=0.5)
        if sugg:
            hint = f",是不是要用 {_fmt_list(_norm_key(msg, inst, s) for s in sugg)}?"
        else:
            shown = fields[:10]
            more = " …" if len(fields) > len(shown) else ""
            hint = f"(可用欄位:{_fmt_list(shown)}{more})" if shown else ""
        return f"訊息 {msg} 沒有欄位「{field}」{hint}"
    lower = {k.lower(): k for k in keys}
    sugg = [lower[key.lower()]] if key.lower() in lower else _close(key, keys)
    hint = f",是不是要用 {_fmt_list(sugg)}?" if sugg else ""
    if info is None and msgs:
        return f"log 裡沒有訊息「{msg}」,找不到欄位「{key}」{hint}"
    return f"找不到欄位「{key}」{hint}"


def _series(index, key: str):
    try:
        return index.series(key)
    except (KeyError, ValueError, IndexError, TypeError):
        return None


def _fetch(index, key: str) -> tuple[np.ndarray, np.ndarray]:
    res = _series(index, key)
    if res is None:
        for alt in _alt_keys(index, key):
            res = _series(index, alt)
            if res is not None:
                break
    if res is None:
        raise ExprError(_unknown_field_msg(index, key))
    t, v = res
    t = np.asarray(t, dtype=np.float64)
    try:
        v = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError):
        raise ExprError(f"欄位「{key}」不是數值資料,無法運算") from None
    if t.shape != v.shape or t.ndim != 1:
        raise ExprError(f"欄位「{key}」資料格式不正確")
    return t, v


# 航向欄位:單位 degheading;沒有單位(或只標 deg)時依常見欄位名判斷
_HEADING_NAMES = frozenset({"yaw", "desyaw", "gcrs", "yc", "hdg", "heading", "crs", "cog"})


def _heading_wrap(index, key: str, vi: np.ndarray) -> float | None:
    """航向欄位回傳包角方式(360.0 → [0, 360)、180.0 → (−180, 180]),否則 None。
    包角方式依欄位自己的資料範圍;超出這兩種範圍(例如已展開的累積角)當一般欄位。"""
    msg, _, field = _split_key(key)
    unit = ""
    try:
        unit = index.unit(key) or ""
    except Exception:  # noqa: BLE001 — duck type 不一定有 unit()
        info = _messages(index).get(msg)
        unit = (getattr(info, "units", None) or {}).get(field, "") if info is not None else ""
    if unit != "degheading" and not (unit in ("", "deg") and (field or "").lower() in _HEADING_NAMES):
        return None
    v = vi[np.isfinite(vi)]
    if v.size == 0:
        return None
    lo, hi = float(v.min()), float(v.max())
    tol = 0.01                                    # float32 / 整數化的捨入(例如 360.00002)
    if lo >= -tol and hi <= 360.0 + tol:
        return 360.0
    if lo >= -180.0 - tol and hi <= 180.0 + tol:
        return 180.0
    return None


def _fix_heading(out: np.ndarray, tb: np.ndarray, ti: np.ndarray, vi: np.ndarray,
                 wrap: float) -> None:
    """跨 0/360(或 ±180)的相鄰樣本之間改走最短弧內插(就地修改 out),結果包回原範圍;
    其餘區間維持一般線性內插的值不變。"""
    with np.errstate(invalid="ignore"):
        dv = np.diff(vi)
        cross = np.abs(dv) > 180.0
    if not cross.any():
        return
    j = np.searchsorted(ti, tb, side="right")               # ti[j-1] <= tb < ti[j]
    k = np.clip(j, 1, ti.size - 1) - 1
    sel = (j >= 1) & (j < ti.size) & cross[k] & (tb > ti[k])
    if not sel.any():
        return
    k = k[sel]
    d = dv[k] - 360.0 * np.round(dv[k] / 360.0)             # 最短弧,|d| <= 180
    r = vi[k] + (tb[sel] - ti[k]) / (ti[k + 1] - ti[k]) * d
    out[sel] = _wrap_360(r) if wrap == 360.0 else _wrap_180(r)


def _interp_onto(tb: np.ndarray, ti: np.ndarray, vi: np.ndarray,
                 wrap: float | None = None) -> np.ndarray:
    """vi(ti) 線性內插到 tb;超出 ti 範圍或落在資料空洞內為 NaN。
    wrap(360 / 180)= 航向欄位:跨 0/360 的區間走最短弧,不會在中間內插出 ≈180。"""
    ok = np.isfinite(ti)
    if not ok.all():
        ti, vi = ti[ok], vi[ok]
    if ti.size == 0:
        return np.full(tb.shape, np.nan)
    if ti.size > 1 and np.any(ti[1:] < ti[:-1]):
        order = np.argsort(ti, kind="stable")
        ti, vi = ti[order], vi[order]
    if ti.size == 1:
        return np.where(tb == ti[0], vi[0], np.nan)
    out = np.interp(tb, ti, vi, left=np.nan, right=np.nan)
    if wrap is not None:
        _fix_heading(out, tb, ti, vi, wrap)
    d = np.diff(ti)
    pos = d[d > 0]
    max_gap = max(GAP_FACTOR * float(np.median(pos)), GAP_MIN_S) if pos.size else GAP_MIN_S
    big = d > max_gap
    if big.any():
        j = np.searchsorted(ti, tb, side="right")           # ti[j-1] <= tb < ti[j]
        inside = (j >= 1) & (j < ti.size)
        jj = np.clip(j, 1, ti.size - 1)
        out[inside & big[jj - 1] & (tb > ti[jj - 1])] = np.nan
    return out


def _load(prog: _Program, index):
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, key in prog.var_key:
        cache[key] = _fetch(index, key)
    tb, _ = cache[prog.keys[0]]
    env: dict[str, np.ndarray] = {}
    for var, key in prog.var_key:
        ti, vi = cache[key]
        if ti is tb or (ti.shape == tb.shape and np.array_equal(ti, tb)):
            env[var] = vi                     # 同一訊息:直接用原始樣本
        else:
            env[var] = _interp_onto(tb, ti, vi, _heading_wrap(index, key, vi))
    return tb, env


# --- 求值 -------------------------------------------------------------------
class _Eval:
    def __init__(self, prog: _Program, env: dict, t: np.ndarray):
        self.prog, self.env, self.t = prog, env, t

    def run(self, node: ast.AST):
        if isinstance(node, ast.Expression):
            return self.run(node.body)
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](self.run(node.left), self.run(node.right))
        if isinstance(node, ast.UnaryOp):
            v = self.run(node.operand)
            return np.negative(v) if isinstance(node.op, ast.USub) else np.positive(v)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Call):
            f = _FUNCS[node.func.id]
            args = [self.run(a) for a in node.args]
            if f.fn is lowpass:
                tau = np.asarray(args[1], dtype=np.float64)
                if tau.ndim != 0 or not np.isfinite(tau) or tau <= 0:
                    raise ExprError("lowpass(x, τ) 的 τ 必須是大於 0 的常數(秒)")
                return lowpass(self.t, args[0], float(tau))
            if f.timed:
                return f.fn(self.t, *args)
            return f.fn(*args)
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            return np.float64(CONSTANTS[node.id])
        if isinstance(node, ast.Constant):
            return self.prog.consts[id(node)]
        raise ExprError("不允許的語法")          # 白名單已擋,不應到這裡

    def _compare(self, node: ast.Compare):
        left = self.run(node.left)
        result = None
        nan_mask = np.isnan(left)
        for op, comp in zip(node.ops, node.comparators):
            right = self.run(comp)
            r = _CMPOPS[type(op)](left, right)
            nan_mask = nan_mask | np.isnan(right)
            result = r if result is None else np.logical_and(result, r)
            left = right
        return np.where(nan_mask, np.nan, np.asarray(result, dtype=np.float64))


def evaluate(expr: str, index) -> tuple[np.ndarray, np.ndarray]:
    """運算式 → (t 秒, v),皆為 float64 新陣列(可自由修改)。錯誤時拋 ExprError。"""
    if index is None:
        raise ExprError("尚未載入 log")
    prog = _compile(expr)
    t, env = _load(prog, index)
    if t.size == 0:
        raise ExprError(f"運算結果沒有資料(欄位 {prog.keys[0]} 沒有任何樣本)")
    with np.errstate(all="ignore"):
        v = _Eval(prog, env, t).run(prog.tree)
        v = np.array(np.broadcast_to(np.asarray(v, dtype=np.float64), t.shape),
                     dtype=np.float64)
    if not np.isfinite(v).any():
        raise ExprError("運算結果沒有任何有效數值(可能除以零、超出定義域,"
                        "或各欄位的時間範圍沒有重疊)")
    return np.array(t, dtype=np.float64), v


def referenced_keys(expr: str) -> list[str]:
    """運算式引用到的欄位 key(依出現順序、不重複,正規化成 "GPS[1].Spd")。
    語法有誤時仍盡量以字面掃描回傳。"""
    try:
        return list(_compile(expr).keys)
    except ExprError:
        pass
    out: list[str] = []
    for m in _REF_RE.finditer(expr if isinstance(expr, str) else ""):
        k = _norm_key(m.group(1), m.group(2), m.group(3))
        if k not in out:
            out.append(k)
    return out


def validate(expr: str, index, *, deep: bool = True) -> str | None:
    """檢查運算式;正確回傳 None,否則回傳繁中錯誤說明。
    deep=False 只檢查語法與欄位是否存在(不載入資料、不求值)。"""
    try:
        if not deep:
            prog = _compile(expr)
            if index is None:
                return "尚未載入 log"
            for key in prog.keys:
                if resolve_key(index, key) is None:
                    return _unknown_field_msg(index, key)
            return None
        evaluate(expr, index)
    except ExprError as e:
        return str(e)
    return None
