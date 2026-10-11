"""失事分析報告的繁體中文文字:結論、型態、假說排名、事件鏈、時間軸、已排除項目、建議、其他觀察、限制。
format_report → 純文字;format_markdown → Markdown(同樣的段落,標題 / 清單格式)。"""
from __future__ import annotations

import re

from core.timeline import format_clock

from . import hypotheses as hyp
from .types import HIT, NA, SOFT

_TAG_ZH = {"precursor": "先兆", "development": "發展", "consequence": "後果", "post_crash": "事後",
           "ground_info": "地面資訊", "standing": "持續狀態", "": ""}
_CONF_ZH = {"高": "高", "中": "中", "低": "低", "": ""}
_STATUS_ZH = {"HIT": "成立", "SOFT": "弱", "CLEAR": "正常", "NA": "不適用"}
_HYP_STATUS_ZH = {"contributing": "貢獻因素", "minor": "次要", "candidate": "候選(無法確定)"}
STICK_PILOT_US = 100.0        # 規格 STEP 3:搖桿偏移 ≥ 100 µs 才算飛手動作(pilot_stick);低於此只是小幅修正

# 正常結束 / 結束型態不明的報告不談「失事」:偵測器說明裡的失事字眼換成中性說法(依序套用,長的在前)
_NEUTRAL = ((re.compile(r"[,,;;]?\s*與失事原因無直接關係"), ",屬保養 / 設定項目"),
            (re.compile(r"不是失事原因"), "不影響這次飛行"),
            (re.compile(r"非本次失事原因"), "保養參考"),
            (re.compile(r"失事原因"), "異常原因"),
            (re.compile(r"失事前"), "結束前"),
            (re.compile(r"失事"), "異常"))


def neutral_text(s: str) -> str:
    for rx, rep in _NEUTRAL:
        s = rx.sub(rep, s)
    return s


def _undetermined(rep) -> bool:
    """結束型態不明、沒有事件錨點:不排名原因(engine.undetermined_end)。"""
    return rep.verdict == "insufficient" and rep.end_type in ("unknown", "") and not rep.hypotheses


def _clk(t) -> str:
    """相對秒(和偵測器說明文字一致)。"""
    return "—" if t is None else f"{float(t):.1f} s"


def _clk2(t) -> str:
    """相對秒 + 分:秒(關鍵時間表用)。"""
    return "—" if t is None else f"{float(t):.2f} s({format_clock(float(t), 1)})"


def _scenario_zh(rep) -> str:
    from .engine import SCENARIO_ZH
    return SCENARIO_ZH.get(rep.scenario, rep.scenario or "—")


def _sub_zh(rep) -> str:
    from .engine import SUBLABEL_ZH
    return "、".join(SUBLABEL_ZH.get(s, s) for s in rep.scenario_sub)


def _end_zh(rep) -> str:
    from .engine import END_TYPE_ZH
    return END_TYPE_ZH.get(rep.end_type, rep.end_type)


def _first(s: str) -> str:
    return s.split("。")[0]


def build_summary(rep) -> str:
    if rep.end_type == "not_armed":
        return "這是未解鎖(桌上或手持)的非飛行 log:馬達沒有轉,不做失事原因分析(各項飛行分析皆不適用)。"
    iv = rep.interval or {}
    when = f"解鎖 {_clk(iv.get('t0'))} ~ {_clk(iv.get('t1'))}"
    if rep.verdict == "no_failure":
        n = len(rep.observations)
        return (f"這次飛行({when})以正常方式結束({_end_zh(rep)}),沒有需要解釋的異常結束;"
                + (f"下方列出 {n} 項健康觀察。" if n else "各項檢查沒有異常。"))
    if _undetermined(rep):
        n = len(rep.observations)
        return (f"結論:無法判定這次飛行({when})是否失事。{rep.verdict_note}"
                + (f";共 {n} 項。" if n else ";沒有觀察到異常。"))
    if rep.verdict == "insufficient":
        return (f"結論:無法確定原因。{rep.verdict_note}。終點判定:{_end_zh(rep)};"
                f"型態:{_scenario_zh(rep)}。下方列出已有的證據與缺少的資料。")
    top = rep.top()
    if top is None:
        return "沒有足夠的證據排出最可能的原因。"
    bits = [f"結論:{_scenario_zh(rep)}。", f"最可能原因:{top.title_zh}(信心:{_CONF_ZH.get(top.confidence, top.confidence)})。"]
    ev = sorted([e for e in top.evidence if e["contrib"] > 0 and e.get("text_zh")], key=lambda e: -e["contrib"])
    seen = set()
    for e in ev:
        s = _first(e["text_zh"])
        if s not in seen:
            seen.add(s)
            bits.append(s + "。")
        if len(seen) >= 3:
            break
    contrib = [h for h in rep.hypotheses if h.status == "contributing"]
    if contrib:
        bits.append("貢獻因素:" + "、".join(f"{h.title_zh}(信心:{h.confidence or '中'})" for h in contrib)
                    + ";與事件經過吻合,但無法證明是觸發原因。")
    return "".join(bits)


CANDIDATE_ADVICE_MAX = 2      # 證據不足時最多列兩個候選的「驗證用」檢查


def build_recommendations(rep) -> list[str]:
    if rep.end_type == "not_armed" or rep.verdict == "no_failure":
        return []
    out: list[str] = []
    if rep.verdict != "ranked":
        # 不硬給原因:先說要補什麼資料;候選假說的建議只當「若要驗證」的條件式檢查
        out.append("補齊缺少的資料(電調遙測 / 電流計 / 測距儀 / 飛行影片 / VIO 伴隨電腦日誌)後再分析;"
                   "目前請勿把任何單一假說當成結論。")
        cands = sorted([h for h in rep.hypotheses if h.status == "candidate" and h.advice_zh], key=lambda h: -h.score)
        for h in cands[:CANDIDATE_ADVICE_MAX]:
            out.append(f"(候選,未證實)若要驗證「{h.title_zh}」:{h.advice_zh}")
        return out
    top = rep.top()
    if top is not None:
        out.append(top.advice_zh)
    for h in rep.hypotheses:
        if h.status == "contributing" and h.advice_zh and h.advice_zh not in out:
            out.append(h.advice_zh)
    return out


def build_limitations(rep, ctx, findings) -> list[str]:
    if rep.end_type == "not_armed":
        return [hyp.DISCLAIMER]
    out = [hyp.DISCLAIMER]
    if ctx is not None:
        if not ctx.has("ESC.RPM"):
            out.append("沒有電調遙測(ESC RPM / 電流),無法直接證明馬達或電調沒有故障;馬達結論只依指令輸出與姿態反應。")
        out.append("沒有測距儀:離地高度(AGL)由氣壓 / EKF 高度減去解鎖時高度估算,只在撞擊前有效。")
    from .engine import MODULE_ERROR_PREFIX
    for f in findings:
        if f.id == "power_data_quality" and f.status in (HIT, SOFT) and f.text_zh:
            out.append(f.text_zh if rep.verdict != "no_failure" else neutral_text(f.text_zh))
        if f.status == NA and f.context == "error" and not f.id.startswith(MODULE_ERROR_PREFIX):
            out.append(f"偵測器 {f.id} 執行失敗,結果未納入:{(f.values or {}).get('error', '')}")
    if rep.missing_modules:
        out.append("下列偵測模組不存在,未參與分析:" + "、".join(rep.missing_modules))
    if getattr(rep, "failed_modules", None):
        out.append("下列偵測模組載入或執行失敗,所屬領域沒有結果(不代表正常):" + "、".join(rep.failed_modules))
    return out


def _features_line(rep) -> str:
    f = rep.scenario_features or {}
    bits = []
    if f.get("agl_m") is not None:
        bits.append(f"{'觸地' if f.get('t_feat') != f.get('t_ref') else '姿態預警'}時離地高度約 {f['agl_m']:+.2f} m")
    if f.get("vz_pre_cms") is not None:
        bits.append(f"之前 5 s 垂直速度中位數 {f['vz_pre_cms']:.0f} cm/s")
    if f.get("tho") is not None:
        bits.append(f"預警時油門 ThO {f['tho']:.2f}")
    if f.get("landed") is not None:
        bits.append("飛控認為已落地" if f["landed"] else "飛控仍認為在空中")
    if f.get("stick_dev_us") is not None:
        dev = float(f["stick_dev_us"])
        when = f"{f['stick_dev_t']:.2f} s" if f.get("stick_dev_t") is not None else ""
        if dev < STICK_PILOT_US:
            note = "、".join(x for x in (when, f"小幅修正,未達飛手動作的 {STICK_PILOT_US:.0f} µs") if x)
        else:
            note = "、".join(x for x in (when, f"≥ {STICK_PILOT_US:.0f} µs,飛手有操作") if x)
        s = f"預警前 1 s 搖桿最大偏移 {dev:.0f} µs({note})"
        if "autopilot_lean" in (rep.scenario_sub or []) and f.get("lean_stick_tol_us") is not None:
            s += f";飛控要求傾斜的期間搖桿都在置中 ±{f['lean_stick_tol_us']:.0f} µs 內"
        bits.append(s)
    if f.get("mode"):
        bits.append(f"模式 {f['mode']}")
    return "、".join(bits)


def _fs_crash_zh(v) -> str:
    if v is None:
        return "FS_CRASH_CHECK 未記錄(無法確認飛控墜機偵測是否開啟)"
    if int(round(v)) == 0:
        return "FS_CRASH_CHECK = 0(飛控墜機偵測關閉:log 不會有 CRASH_CHECK 事件,終點改用重算 / 上鎖分類判斷)"
    return f"FS_CRASH_CHECK = {int(round(v))}(飛控墜機偵測開啟)"


def format_report(rep) -> str:
    L: list[str] = []
    tb = rep.t_base
    failure = rep.end_type != "not_armed" and rep.verdict != "no_failure" and not _undetermined(rep)
    if rep.end_type == "not_armed":
        L.append("失事原因分析 — vio_ate_studio")
    elif failure:
        L.append("失事原因分析 — vio_ate_studio")
    else:
        L.append("飛行結束分析 — vio_ate_studio")
    L.append(f"資料來源:{rep.path or '(模擬)'}")
    L.append(f"時間基準:報告時間為相對 log 起點的秒數;log 絕對秒 = 相對秒 + t_base({tb:.4f} s)")
    if failure:
        L.append(_fs_crash_zh(getattr(rep, "fs_crash_check", None)))
    L.append("")
    L.append("【結論】")
    L.append(rep.summary_zh or "—")
    if rep.intervals:
        L.append("")
        L.append("解鎖區間:")
        for iv in rep.intervals:
            mark = " ← 分析對象" if rep.interval and iv["index"] == rep.interval["index"] else ""
            L.append(f"  第 {iv['index'] + 1} 次  {_clk(iv['t0'])} ~ {_clk(iv['t1'])}  {iv['end']}"
                     + ("(異常)" if iv["abnormal"] else "") + mark)
    if rep.end_type == "not_armed":
        if rep.na_domains:
            L.append("")
            L.append("【不適用】(馬達從未解鎖運轉:不是「正常」,是沒有飛行資料可判斷)")
            for d in rep.na_domains:
                L.append(f"  {d['domain_zh']}:不適用 — {d['text_zh']}")
        _ground_section(L, rep)
        _tail(L, rep)
        return "\n".join(L)

    if rep.verdict == "no_failure":
        L.append("")
        L.append("【飛行結束方式】")
        L.append(f"終點判定:{_end_zh(rep)}")
        _observation_section(L, rep, "【健康觀察】(正常結束;只供保養參考)", neutral=True)
        _key_times(L, rep, tb)
        _timeline_section(L, rep, "【結束前時間軸】")
        _ground_section(L, rep)
        _tail(L, rep)
        return "\n".join(L)

    if _undetermined(rep):
        L.append("")
        L.append("【飛行結束方式】")
        L.append(f"終點判定:{_end_zh(rep)}(無法判斷是正常降落還是異常結束)")
        L.append(f"  ({rep.verdict_note})")
        _observation_section(L, rep, "【觀察到的異常】(結束型態不明:沒有計分、不是原因結論)", neutral=True)
        _key_times(L, rep, tb)
        _timeline_section(L, rep, "【結束前時間軸】")
        if rep.recommendations:
            L.append("")
            L.append("【建議】")
            for i, r in enumerate(rep.recommendations, 1):
                L.append(f"  {i}. {r}")
        _ground_section(L, rep)
        _tail(L, rep)
        return "\n".join(L)

    L.append("")
    L.append("【失事型態】")
    L.append(f"終點判定:{_end_zh(rep)}")
    if rep.scenario:
        L.append(f"型態:{_scenario_zh(rep)}" + (f"(子型態:{_sub_zh(rep)})" if rep.scenario_sub else ""))
        fl = _features_line(rep)
        if fl:
            L.append(f"判定依據:{fl}")

    ranked = [h for h in rep.hypotheses if h.score > 0]
    decided = rep.verdict == "ranked"
    if ranked:
        L.append("")
        L.append("【假說排名】(分數 = Σ 權重 × 證據強度 × 時間接近度;缺必要證據者 ×0.3)" if decided else
                 "【候選假說】(無法確定原因:以下只是有證據的候選,不是結論;分數 = Σ 權重 × 證據強度 × 時間接近度,"
                 "缺必要證據者 ×0.3)")
        for i, h in enumerate(ranked, 1):
            tag = "第一名" if h.status == "ranked" else _HYP_STATUS_ZH.get(h.status, h.status)
            conf = f" 信心:{h.confidence}" if h.confidence else ""
            L.append(f"  {i}. {h.title_zh}  分數 {h.score:.1f}(占 {h.share * 100:.0f}%){conf}  {tag}"
                     + ("" if h.gate_ok else f"  ⚠ 缺必要證據:{h.gate_note}"))
            if getattr(h, "confidence_note", ""):
                L.append(f"       ⚠ {h.confidence_note}")
            for e in sorted(h.evidence, key=lambda e: -e["contrib"])[:6]:
                if e["contrib"] <= 0:
                    continue
                L.append(f"       · {e['det']}({_STATUS_ZH.get(e['status'], e['status'])},{_TAG_ZH.get(e['tag'], e['tag'])},"
                         f"{_clk(e['t_onset'])},+{e['contrib']:.1f}):{e.get('text_zh', '')}")
            if h.status in ("ranked", "contributing") and h.checked_ok:
                for c in h.checked_ok[:3]:
                    L.append(f"       ✓ 已檢查正常:{c}")
    if rep.verdict == "insufficient":
        L.append(f"  (無法確定:{rep.verdict_note})")

    if rep.chain:
        L.append("")
        L.append("【事件鏈(機制)】(第一名的證據依時間排列;貢獻因素的項目只表示「與經過吻合」,未證實為觸發原因)" if decided
                 else "【已知證據(依時間)】(無法確定原因:以下是候選假說的證據,不是機制結論)")
        for c in rep.chain:
            role = {"top": "", "contributing": "(貢獻因素,吻合但未證實)", "candidate": "(候選,未證實)"}.get(c["role"], "")
            when = "持續狀態" if c["tag"] == "standing" or c["t"] is None else _clk(c["t"])
            L.append(f"  {when:>9}  [{_TAG_ZH.get(c['tag'], c['tag'])}]{role} {c['text_zh']}")

    _key_times(L, rep, tb)
    _timeline_section(L, rep, "【關鍵時間軸】(先兆 / 發展 / 後果 依與姿態失控的先後標示;後果不是原因)")

    out_h = [h for h in rep.hypotheses if h.status == "ruled_out"]
    na_h = [h for h in rep.hypotheses if h.status in ("not_applicable", "no_evidence")]
    if out_h or na_h:
        L.append("")
        L.append("【已排除 / 不適用】")
        for h in out_h:
            conf = f"(排除信心:{h.rule_out_conf}" + (f";{h.rule_out_note}" if h.rule_out_note else "") + ")" \
                if h.rule_out_conf else ""
            L.append(f"  已排除 {h.title_zh}{conf}")
            for b in h.ruled_out_basis:
                L.append(f"       · {b}")
        for h in na_h:
            word = "不適用" if h.status == "not_applicable" else "無證據"
            L.append(f"  {word} {h.title_zh}(資料不足或條件不符,不代表正常)")

    if rep.recommendations:
        L.append("")
        L.append("【建議】")
        for i, r in enumerate(rep.recommendations, 1):
            L.append(f"  {i}. {r}")
    _observation_section(L, rep, "【其他觀察(非本次失事原因)】(設定 / 保養項目,沒有計入原因分數)")
    _ground_section(L, rep)
    _tail(L, rep)
    return "\n".join(L)


def _key_times(L, rep, tb):
    a = rep.anchors
    rows = []
    for name, label in (("t_first_anomaly", "最早異常"), ("t_touch", "觸地"), ("t_warn", "姿態誤差 > 10°"),
                        ("t_loss", "姿態誤差 > 30°"), ("t_first_hit", "第一次硬撞擊"), ("t_impact_peak", "最後一次撞擊"),
                        ("t_rest", "機身靜止"), ("t_detect", "飛控 / 偵測器判定失事"), ("t_disarm", "上鎖")):
        v = getattr(a, name)
        if v is not None:
            rows.append(f"  {label:<14}{_clk2(v)}  log 絕對 {v + tb:.3f} s")
    if not rows:
        return
    L.append("")
    L.append("【關鍵時間】(相對 log 起點的秒數;偵測器說明裡的時間也是這個)")
    L.extend(rows)
    if rep.verdict != "no_failure" and a.t_detect is not None and a.t_impact_peak is not None:
        L.append(f"  → 飛控判定比實際撞擊晚 {a.t_detect - a.t_impact_peak:.1f} 秒;請往前看事件鏈,不要只看判定當下。")


def _timeline_section(L, rep, title):
    if rep.timeline:
        L.append("")
        L.append(title)
        for e in rep.timeline:
            tag = _TAG_ZH.get(e.tag, e.tag)
            L.append(f"  {_clk(e.t):>9}  " + (f"[{tag}] " if tag else "") + e.label)


def _observation_section(L, rep, title, neutral: bool = False):
    if rep.observations:
        L.append("")
        L.append(title)
        for o in rep.observations:
            L.append(f"  · {neutral_text(o['text_zh']) if neutral else o['text_zh']}")


def _ground_section(L, rep):
    if rep.ground_events:
        L.append("")
        L.append("【地面 / 未解鎖期間的事件】(只當資訊,不列入原因)")
        for e in rep.ground_events[:40]:
            L.append(f"  {_clk(e.t):>9}  {e.label}")
        if len(rep.ground_events) > 40:
            L.append(f"  …還有 {len(rep.ground_events) - 40} 筆")


def _tail(L, rep):
    a = rep.anchors
    anchor_notes = [n for n in (a.notes if a is not None else []) if n]
    if rep.verdict == "no_failure" or _undetermined(rep):
        anchor_notes = [neutral_text(n) for n in anchor_notes]
    if anchor_notes:
        L.append("")
        L.append("【分析備註】(終點 / 錨點判定過程)")
        for n in anchor_notes:
            L.append(f"  · {n}")
    if rep.notes or rep.limitations:
        L.append("")
        L.append("【資料品質與限制】")
        for n in rep.notes + rep.limitations:
            L.append(f"  · {n}")


_SEC = re.compile(r"^【(.+?)】(.*)$")


def format_markdown(rep) -> str:
    """Markdown 匯出:和 format_report 同樣的內容;【段落】→ ## 標題,縮排的行 → 清單(更深的縮排 → 子清單)。"""
    out: list[str] = []
    for i, ln in enumerate(format_report(rep).splitlines()):
        if i == 0:
            out.append(f"# {ln}")
            continue
        if not ln.strip():
            out.append("")
            continue
        m = _SEC.match(ln)
        if m:
            out += ["", f"## {m.group(1)}"] + ([f"*{m.group(2).strip()}*"] if m.group(2).strip() else []) + [""]
            continue
        body = ln.strip()
        if body.startswith("· "):
            body = body[2:]
        indent = len(ln) - len(ln.lstrip(" "))
        if indent == 0:
            out.append(body + "  ")
        elif indent <= 4:
            out.append(f"- {body}")
        else:
            out.append(f"    - {body}")
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
