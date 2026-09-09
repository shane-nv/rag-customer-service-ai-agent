# -*- coding: utf-8 -*-
"""Evaluation harness for the RAG pipeline.

Runs the gold set through the real retrieval path (retriever.retrieve, which applies
the language filter + refuse gate), then reports:
  - Retrieval  : Recall@k, Recall@1  (only for questions that have an expected doc)
  - Refusal    : refuse accuracy, false-refuse rate (answerable wrongly refused),
                 catch rate (out-of-scope correctly refused)
  - Threshold  : a sweep over candidate SCORE_THRESHOLD values using the recorded
                 best-distance per question, to pick the threshold from data.

Run (LLM not needed — this is retrieval-only, fast):
    python3 src/evaluate.py

Outputs data/eval_results.csv (per-question) and prints a summary.
"""
import sys, pathlib, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import config
from retriever import get_store, retrieve

GOLD = config.REPO_ROOT / "data" / "gold_set.csv"
OUT = config.REPO_ROOT / "data" / "eval_results.csv"


def load_gold():
    with open(GOLD, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pct(xs):
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):.1f}%  (n={len(xs)})" if xs else "n/a"


def main():
    store = get_store()
    gold = load_gold()

    rows = []
    for g in gold:
        q = g["question"].strip()
        expected = g["expected_doc_id"].strip()
        expected_set = [e.strip() for e in expected.split("|") if e.strip()]
        should_refuse = (g["label"] == "out_of_scope")

        results, refused = retrieve(q, store)
        retrieved_ids = [d.metadata.get("doc_id") for d, _ in results]
        best = results[0][1] if results else None

        rows.append({
            "id": g["id"], "label": g["label"], "question": q,
            "expected_doc_id": expected,
            "retrieved_top_k": "|".join(retrieved_ids),
            "best_distance": round(best, 4) if best is not None else "",
            "refused": refused,
            "should_refuse": should_refuse,
            "refuse_correct": (refused == should_refuse),
            # 系統找出的前 K 篇文件中，有沒有包含正確答案？（只要有中就算及格）
            "hit@k": (any(e in retrieved_ids for e in expected_set)) if expected_set else None,
            # 系統排名第 1 的文件就是正確答案嗎？（衡量檢索精準度的最高標準）
            "hit@1": (bool(retrieved_ids) and retrieved_ids[0] in expected_set) if expected_set else None,
        })

    # ---- per-question CSV ----
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    scored = [r for r in rows if r["expected_doc_id"]]           # has an expected doc
    answerable = [r for r in rows if not r["should_refuse"]]     # must NOT refuse
    oos = [r for r in rows if r["should_refuse"]]                # must refuse

    print("\n===== RETRIEVAL =====")
    print("Recall@k :", pct([r["hit@k"] for r in scored]))
    print("Recall@1 :", pct([r["hit@1"] for r in scored]))
    misses = [r["id"] for r in scored if not r["hit@k"]]
    if misses:
        print("  missed@k:", ", ".join(misses))

    print("\n===== REFUSAL (with current SCORE_THRESHOLD =", config.SCORE_THRESHOLD, ") =====")
    print("Refuse accuracy   :", pct([r["refuse_correct"] for r in rows]))
    print("False-refuse rate :", pct([r["refused"] for r in answerable]), "(answerable wrongly refused — want LOW)")
    print("Catch rate        :", pct([r["refused"] for r in oos]), "(out-of-scope correctly refused — want HIGH)")

    # ---- threshold sweep (data-driven calibration) ----
    print("\n===== THRESHOLD SWEEP (pick the one that maximizes refuse accuracy) =====")
    def dist(r):
        return r["best_distance"] if isinstance(r["best_distance"], float) else None
    print(f"{'thr':>5} | {'refuse_acc':>10} | {'false_refuse':>12} | {'catch':>6}")
    best_thr, best_acc = None, -1
    t = 0.30
    while t <= 0.60001:
        def refuse_at(r):
            d = dist(r)
            return (d is None) or (d > t)
        acc = sum((refuse_at(r) == r["should_refuse"]) for r in rows) / len(rows)
        fr = sum(refuse_at(r) for r in answerable) / len(answerable)
        catch = sum(refuse_at(r) for r in oos) / len(oos)
        print(f"{t:>5.2f} | {acc*100:>9.1f}% | {fr*100:>11.1f}% | {catch*100:>5.1f}%")
        if acc > best_acc:
            best_acc, best_thr = acc, t
        t += 0.02
    print(f"-> best threshold by refuse accuracy: {best_thr:.2f} ({best_acc*100:.1f}%)")

    # ---- distance distributions (sanity for the threshold) ----
    def dstats(group):
        ds = sorted(dist(r) for r in group if dist(r) is not None)
        return f"min={ds[0]:.3f} median={ds[len(ds)//2]:.3f} max={ds[-1]:.3f}" if ds else "n/a"
    print("\nbest-distance by group:")
    print("  answerable         :", dstats([r for r in rows if r["label"] == "answerable"]))
    print("  answerable_negative:", dstats([r for r in rows if r["label"] == "answerable_negative"]))
    print("  out_of_scope       :", dstats(oos))

    print(f"\nper-question detail -> {OUT}")

    # 後續可加：answer-quality 檢查 —— 對每個 answerable 題呼叫 answer()、比對回覆是否含預期字串
    #   （需在 gold_set.csv 加 expected_substring 欄）。因逐題呼叫 LLM 較慢，預設不執行。


if __name__ == "__main__":
    main()
