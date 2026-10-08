"""時間同步:重疊區間裁切、線性內插配對、時間偏移自動搜尋。"""
from __future__ import annotations

import math

import numpy as np


def sync_pair(ref_t, ref_p, est_t, est_p,
              t0: float | None = None, t1: float | None = None,
              offset_s: float = 0.0, max_gap_s: float = 0.5):
    """把估測軌跡內插到參考軌跡的時間戳上。

    - offset_s: 加在估測時間戳上的偏移(秒)。
    - max_gap_s: 參考時間戳離最近估測樣本超過此值時捨棄(避免跨資料空洞內插)。
    回傳 (t, ref_xyz, est_xyz),皆已裁切到重疊區間。
    """
    ref_t = np.asarray(ref_t, dtype=float)
    est_t = np.asarray(est_t, dtype=float) + float(offset_s)
    ref_p = np.asarray(ref_p, dtype=float)
    est_p = np.asarray(est_p, dtype=float)

    lo = max(ref_t[0], est_t[0])
    hi = min(ref_t[-1], est_t[-1])
    if t0 is not None:
        lo = max(lo, float(t0))
    if t1 is not None:
        hi = min(hi, float(t1))
    if hi <= lo:
        raise ValueError("兩軌跡(或所選時間區間)沒有重疊,請檢查時間範圍與偏移設定")

    m = (ref_t >= lo) & (ref_t <= hi)
    rt = ref_t[m]
    rp = ref_p[m]
    if rt.size == 0:
        raise ValueError("時間區間內沒有參考軌跡樣本")

    est_i = np.column_stack([np.interp(rt, est_t, est_p[:, k]) for k in range(est_p.shape[1])])

    # 空洞保護:到最近估測樣本的時間差
    idx = np.searchsorted(est_t, rt)
    left = np.clip(idx - 1, 0, est_t.size - 1)
    right = np.clip(idx, 0, est_t.size - 1)
    nearest = np.minimum(np.abs(rt - est_t[left]), np.abs(est_t[right] - rt))
    keep = nearest <= max_gap_s
    return rt[keep], rp[keep], est_i[keep]


def _speed_profile(t: np.ndarray, p: np.ndarray):
    """水平速度剖面(取樣於區段中點),供時間偏移相關性搜尋。"""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    dt = np.diff(t)
    ok = dt > 1e-6
    v = np.full(dt.shape, np.nan)
    d = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    v[ok] = d[ok] / dt[ok]
    tm = 0.5 * (t[1:] + t[:-1])
    m = np.isfinite(v)
    return tm[m], v[m]


_GRID_DT = 0.05          # 速度剖面共同網格間隔(秒)
_MIN_PTS = 50            # 每個偏移至少要有這麼多重疊網格點才計算相關係數


def find_time_offset(ref_t, ref_p, est_t, est_p,
                     search_s: float = 2.0, step_s: float = 0.02):
    """掃描 offset ∈ [-search, +search],找讓兩軌跡水平速度剖面相關係數最大的偏移。

    回傳 (offset_s, corr)。offset 意義同 sync_pair(加在估測時間戳上)。
    """
    rt, rv = _speed_profile(ref_t, ref_p)
    et, ev = _speed_profile(est_t, est_p)
    if rt.size < 20 or et.size < 20:
        raise ValueError("樣本太少,無法自動估測時間偏移")

    lo = max(rt[0], et[0] - search_s)
    hi = min(rt[-1], et[-1] + search_s)
    if hi - lo < 5.0:
        raise ValueError("重疊時間太短(< 5 s),無法自動估測時間偏移")

    grid = np.arange(lo, hi, _GRID_DT)
    rg = np.interp(grid, rt, rv)
    offs = np.arange(-search_s, search_s + 1e-9, step_s)

    corrs = _corrs_shared(grid, rg, et, ev, offs, step_s)
    if corrs is None:
        corrs = _corrs_direct(grid, rg, et, ev, offs)
    ok = np.flatnonzero(np.isfinite(corrs))
    if ok.size == 0:
        raise ValueError("速度剖面無變化,無法自動估測時間偏移")
    k = int(ok[np.argmax(corrs[ok])])          # 同分取第一個(與逐一比較 > 相同)
    return float(offs[k]), float(corrs[k])


def _corrs_direct(grid, rg, et, ev, offs) -> np.ndarray:
    """逐一偏移重新內插估測速度(通用做法);跳過的偏移為 NaN。"""
    out = np.full(offs.size, np.nan)
    for k, off in enumerate(offs):
        eg = np.interp(grid, et + off, ev, left=np.nan, right=np.nan)
        m = np.isfinite(eg)
        if int(m.sum()) < _MIN_PTS:
            continue
        a = rg[m] - rg[m].mean()
        b = eg[m] - eg[m].mean()
        den = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
        if den <= 1e-12:
            continue
        out[k] = float((a * b).sum()) / den
    return out


def _corrs_shared(grid, rg, et, ev, offs, step_s) -> np.ndarray | None:
    """與 _corrs_direct 同結果(相關係數差 ~1e-14)但快十倍以上;不適用時回傳 None。

    偏移步長與網格間隔成整數比(step/0.05 = r_s/r_g)時,編號同餘 r_g 的偏移彼此只差
    整數個網格點,網格點落在估測序列的哪一段可以共用:每類只做一次二分搜尋(以類中間的
    偏移為基準,其餘平移 sh 格取用),各偏移再以與 np.interp 相同的算式
    v[j] + 斜率[j] · (grid − (t[j] + off)) 逐點求值 —— 連 t + off 的浮點捨入都相同,
    不會因「平移」與「偏移差」在浮點下差 ~1e-13 s 而產生系統誤差。
    有效範圍與 np.interp(left/right=NaN) 的邊界判斷完全相同(兩端連續一段),
    相關係數用預先去均值的切片 + 內積計算,不做遮罩複製。
    """
    n = grid.size
    su = round(float(step_s) * 1e6)                  # 步長以 μs 整數求公因數
    if n < 2 or su <= 0 or abs(float(step_s) * 1e6 - su) > 1e-6:
        return None
    if not np.all(np.diff(et) > 0):                 # 時間非遞增:交給通用做法
        return None
    gu = round(_GRID_DT * 1e6)
    g = math.gcd(su, gu)
    r_g, r_s = gu // g, su // g

    # 每個偏移的有效網格範圍 [i0, i1):grid 落在 [et[0]+off, et[-1]+off]
    i0s = np.searchsorted(grid, et[0] + offs, side="left")
    i1s = np.searchsorted(grid, et[-1] + offs, side="right")
    lo, step = float(grid[0]), float(grid[1] - grid[0])   # np.arange 以 lo + i·step 填值
    a_all = rg - rg.mean()
    ev_c = ev - ev.mean()
    slope = np.diff(ev) / np.diff(et)
    buf = np.empty(n)
    a_stats: dict[tuple[int, int], tuple[float, float]] = {}
    out = np.full(offs.size, np.nan)
    for cls in range(min(r_g, offs.size)):
        ks = np.arange(cls, offs.size, r_g)
        ks = ks[i1s[ks] - i0s[ks] >= _MIN_PTS]
        if ks.size == 0:
            continue
        kr = int(ks[ks.size // 2])
        sh = -((ks - kr) // r_g) * r_s                 # 偏移 k 的第 i 點 ≈ 基準的第 i+sh 點
        e_lo = int((i0s[ks] + sh).min())
        x = lo + np.arange(e_lo, int((i1s[ks] + sh).max()), dtype=np.float64) * step
        seg = np.clip(np.searchsorted(et + offs[kr], x, side="right") - 1, 0, slope.size - 1)
        t_seg, v_seg, s_seg = et[seg], ev_c[seg], slope[seg]
        for k, s in zip(ks.tolist(), sh.tolist()):
            i0, i1 = int(i0s[k]), int(i1s[k])
            nw = i1 - i0
            st = a_stats.get((i0, i1))
            if st is None:
                a = a_all[i0:i1]
                st = a_stats[(i0, i1)] = (float(a.sum()), float(np.dot(a, a)))
            sa, saa = st
            a = a_all[i0:i1]
            j0, j1 = i0 + s - e_lo, i1 + s - e_lo
            b = np.add(t_seg[j0:j1], offs[k], out=buf[:nw])      # 同 np.interp(grid, et + off, …)
            np.subtract(grid[i0:i1], b, out=b)
            b *= s_seg[j0:j1]
            b += v_seg[j0:j1]
            sb = float(b.sum())
            va = saa - sa * sa / nw
            vb = float(np.dot(b, b)) - sb * sb / nw
            if not (va > 0.0 and vb > 0.0):
                continue
            den = math.sqrt(va * vb)
            if den <= 1e-12:
                continue
            out[k] = (float(np.dot(a, b)) - sa * sb / nw) / den
    return out
