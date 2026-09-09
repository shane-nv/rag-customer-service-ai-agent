# -*- coding: utf-8 -*-
"""Agent 版評估：把題庫跑過 plan A 管線，衡量「規劃層 + 執行判定」是否可靠。

複用 Block 3 的精神（evaluate.py）：多數指標是確定性的、不靠人評。
指標：
  - 工具選擇正確率   ：規劃的工具集是否符合該類別的期望
  - 過度規劃         ：policy_only / off_topic 卻叫了訂單工具
  - 訂單編號抽取     ：expected_order_id 是否正確進入計劃
  - 政策檢索 Recall  ：expected_policy_docs（憑 ground truth 標的正解，跑之前就定死）是否都被政策步撈到；
                       單一問題也只標一份 canonical 正解，撈不到就照實掛，不放寬標準遷就輸出（dispute 不看）。
  - 退貨判定         ：expected_eligible 是否等於 check_return_window 算出的
  - dispute 判定     ：送達爭議是否被正確標記
  - 離題             ：off_topic 是否得到空計劃 plan==[]

per-question CSV 另記 sub_queries / retrieved_docs，方便定位。

Run:
    python src/evaluate_agent.py                       # dev 集 agent_gold.csv（可調 prompt）
    python src/evaluate_agent.py data/agent_test.csv   # held-out 測試集（調 prompt 時勿看，最後跑一次）
輸出檔名依輸入衍生：agent_gold.csv → agent_gold_results.csv、agent_test.csv → agent_test_results.csv。
"""
import sys, pathlib, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import config
from retriever import get_store, retrieve
from agent import plan, validate_plan, execute, compose, _fallback

GOLD = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (config.REPO_ROOT / "data" / "agent_gold.csv")
OUT = GOLD.with_name(GOLD.stem + "_results.csv")


def run_instrumented(query, store):
    """鏡像 agent.handle 的三路，但額外回傳 facts（handle 本身不暴露內部）。
    若日後 handle 分支邏輯改了，記得同步這裡。"""
    steps = validate_plan(plan(query), query)
    if steps is None:                       # planner 壞掉 → 降級
        out = _fallback(query, store)
        return out, {}
    if not steps:                           # 合法空計劃（離題）
        out = _fallback(query, store)
        out["plan"] = []
        return out, {}
    facts = execute(steps, query, store)
    out = compose(query, facts)
    out["plan"] = steps
    return out, facts


def split(cell):
    return [x.strip() for x in (cell or "").split("|") if x.strip()]


def to_bool(cell):
    c = (cell or "").strip().lower()
    return True if c == "true" else (False if c == "false" else None)


def planned_tools(plan_steps):
    """plan 可能是 None / [] / list-of-steps。回傳工具名稱集合。"""
    if not plan_steps:
        return set()
    return {s.get("tool") for s in plan_steps}


def retrieved_docs_for(sub_queries, store):
    """對每個政策子問題重跑 retrieve，收集撈到的 doc_id 聯集（複用 Recall 機制）。"""
    docs = set()
    for q in sub_queries:
        results, _ = retrieve(q, store)
        docs |= {d.metadata.get("doc_id") for d, _ in results}
    return docs


def main():
    store = get_store()
    with open(GOLD, encoding="utf-8-sig") as f:
        gold = list(csv.DictReader(f))

    rows = []
    for g in gold:
        q = g["query"].strip()
        cat = g["category"].strip()
        exp_oids = set(o.upper() for o in split(g["expected_order_id"]))  # 支援多張(pipe)
        exp_docs = set(split(g["expected_policy_docs"]))
        exp_elig = to_bool(g["expected_eligible"])
        exp_disp = to_bool(g["expected_dispute"])

        out, facts = run_instrumented(q, store)
        plan_steps = out.get("plan")
        ptools = planned_tools(plan_steps)
        via = out.get("via")

        # 訂單編號抽取（set 比對：expected 有幾張就要抓到哪幾張，多訂單題才測得出限制）
        got_oids = set()
        for s in (plan_steps or []):
            if s.get("tool") == "get_order_status":
                oid = (s.get("args", {}).get("order_id") or "").upper()
                if oid:
                    got_oids.add(oid)
        order_id_ok = (got_oids == exp_oids) if exp_oids else None

        # 政策檢索 Recall + 記錄實際撈到的 doc（僅供定位，不用來事後改 gold）
        sub_qs = [s.get("args", {}).get("sub_query") for s in (plan_steps or [])
                  if s.get("tool") == "lookup_policy"]
        sub_qs = [x for x in sub_qs if x]
        retrieved = retrieved_docs_for(sub_qs, store) if sub_qs else set()
        if cat == "dispute" or not exp_docs:
            policy_recall = None                      # dispute 政策段被壓下、不看檢索
        else:
            policy_recall = (exp_docs <= retrieved)   # 期望的正解文件都要被撈到（不放寬）

        # 退貨判定
        rw = (facts or {}).get("return_window")
        got_elig = rw.get("eligible") if rw else None
        eligible_ok = (got_elig == exp_elig) if exp_elig is not None else None

        # dispute 判定
        got_disp = bool((facts or {}).get("dispute"))
        dispute_ok = (got_disp == exp_disp) if exp_disp is not None else None

        # 過度規劃：不該叫訂單工具卻叫了
        over_plan = ("get_order_status" in ptools) if cat in ("policy_only", "off_topic") else False

        # 各類別的「工具選擇」判準
        if cat == "single_tool":
            tool_ok = (ptools == {"get_order_status"})
        elif cat == "policy_only":
            tool_ok = (ptools == {"lookup_policy"})
        elif cat == "multi_tool":
            tool_ok = ({"get_order_status", "lookup_policy"} <= ptools)
        elif cat == "off_topic":
            tool_ok = (plan_steps == [])          # 正確判空
        elif cat == "dispute":
            tool_ok = (via == "dispute")          # dispute 的成功訊號是 via
        else:
            tool_ok = None

        rows.append({
            "id": g["id"], "category": cat, "query": q,
            "planned_tools": "|".join(sorted(ptools)) or "(empty)",
            "via": via,
            "sub_queries": "|".join(sub_qs),
            "retrieved_docs": "|".join(sorted(d for d in retrieved if d)),
            "tool_ok": tool_ok, "over_plan": over_plan,
            "order_id_ok": order_id_ok, "policy_recall": policy_recall,
            "eligible_ok": eligible_ok, "dispute_ok": dispute_ok,
        })

    # ---- per-row CSV ----
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 印出摘要 ----
    def rate(key, subset=None):
        xs = [r[key] for r in rows if r[key] is not None and (subset is None or r["category"] in subset)]
        return f"{sum(xs)}/{len(xs)}" if xs else "n/a"

    print("\n===== AGENT 評估摘要 [{}] (n={}) =====".format(GOLD.name, len(rows)))
    print("工具選擇正確      :", rate("tool_ok"))
    print("  single_tool     :", rate("tool_ok", {"single_tool"}))
    print("  multi_tool      :", rate("tool_ok", {"multi_tool"}))
    print("  policy_only     :", rate("tool_ok", {"policy_only"}))
    print("  off_topic(判空) :", rate("tool_ok", {"off_topic"}))
    print("  dispute         :", rate("tool_ok", {"dispute"}))
    print("訂單編號抽取正確  :", rate("order_id_ok"))
    print("政策檢索 Recall   :", rate("policy_recall"))
    print("退貨判定正確      :", rate("eligible_ok"))
    print("dispute 判定正確  :", rate("dispute_ok"))
    over = [r["id"] for r in rows if r["over_plan"]]
    print("過度規劃(不該叫訂單卻叫了):", (", ".join(over) if over else "0"))

    # ---- 逐題失敗清單（方便定位）----
    def failed(r):
        checks = [r["tool_ok"], r["order_id_ok"], r["policy_recall"], r["eligible_ok"], r["dispute_ok"]]
        return any(c is False for c in checks) or r["over_plan"]
    bad = [r for r in rows if failed(r)]
    if bad:
        print("\n----- 未過的題（逐項）-----")
        for r in bad:
            flags = [k for k in ("tool_ok", "order_id_ok", "policy_recall", "eligible_ok", "dispute_ok")
                     if r[k] is False]
            if r["over_plan"]:
                flags.append("over_plan")
            print(f"  {r['id']} [{r['category']}] {r['query'][:28]}  → 失敗: {', '.join(flags)}"
                  f"；plan={r['planned_tools']}, via={r['via']}, 撈到={r['retrieved_docs'] or '-'}")
    else:
        print("\n全部通過 ✅")

    print(f"\nper-question detail -> {OUT}")


if __name__ == "__main__":
    main()
