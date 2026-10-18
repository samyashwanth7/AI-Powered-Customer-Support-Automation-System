"""
Evaluation harness for the AI-Powered Customer Support Automation System.

Run from the REPO ROOT (so `src` and `docs/` resolve):
    python eval/run_eval.py
    python eval/run_eval.py --limit 10 --delay 1.0     # quick smoke run
    python eval/run_eval.py --out eval/results.json

Measures, over eval/queries.jsonl:
  1. Intent routing accuracy (overall + per class + misses)
  2. HITL approval precision / recall (does the graph pause when it should?)
  3. Answer correctness on answerable queries (expected fact appears in answer)
  4. Safe-failure rate on unanswerable queries (admits it doesn't know / escalates)
  5. Latency (mean / p50 / p95) and tokens per turn (mean / p95)

Notes:
  - Every query runs in its own fresh thread, so Memory queries are routing-only.
  - Approval queries pause before `human_approval` (interrupt_before), so their
    latency is time-to-pause and the checked answer is the department draft.
  - Check 4 is a keyword heuristic, so treat it as approximate.
"""
import sys
# Global compatibility shim for langchain_core 0.3+/1.0+ pydantic_v1 deprecation
try:
    import pydantic.v1 as pydantic_v1
    sys.modules['langchain_core.pydantic_v1'] = pydantic_v1
except ImportError:
    pass

import argparse
import json
import os
import statistics
import time
import uuid
from collections import Counter, defaultdict

sys.path.insert(0, os.getcwd())

REFUSAL_HINTS = [
    "don't have", "do not have", "don't know", "do not know", "not sure",
    "unable to", "escalat", "no information", "not specified", "cannot find",
    "can't find", "not covered", "isn't covered", "not available in",
    "reach out", "contact support", "follow up", "let you know", "apologize",
    "consult", "contact our"
]


def pct(x, n):
    return f"{(100.0 * x / n):.1f}%" if n else "n/a"


def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def norm(s):
    return " ".join(str(s).lower().split())


def last_answer(values):
    """Prefer the polished final response, else the last AI message (draft)."""
    if values.get("final_response"):
        return values["final_response"]
    for msg in reversed(values.get("messages", [])):
        if msg.__class__.__name__ == "AIMessage":
            return str(msg.content)
    return ""


def run_one(app, query, retries=3, backoff=4.0):
    from langchain_core.messages import HumanMessage

    cfg = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:10]}"}}
    err = None
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            app.invoke({"messages": [HumanMessage(content=query)]}, cfg)
            latency = time.perf_counter() - t0
            values = app.get_state(cfg).values
            return values, latency, None
        except Exception as e:  # rate limits, transient API errors
            err = f"{type(e).__name__}: {e}"
            time.sleep(backoff * (attempt + 1))
    return {}, float("nan"), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/queries.jsonl")
    ap.add_argument("--out", default="eval/results.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between queries")
    args = ap.parse_args()

    from src.graph import app  # imported late so the path tweak above applies

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    results, errors = [], 0
    for i, r in enumerate(rows, 1):
        values, latency, err = run_one(app, r["query"])
        if err:
            errors += 1
        res = {
            "id": r["id"],
            "query": r["query"],
            "expected_intent": r["intent"],
            "pred_intent": values.get("intent"),
            "expected_approval": r["approval"],
            "pred_approval": bool(values.get("approval_required", False)),
            "sentiment": values.get("sentiment"),
            "priority": values.get("priority"),
            "latency_s": None if err else round(latency, 3),
            "tokens": values.get("total_tokens"),
            "answer": last_answer(values),
            "error": err,
            "expect_any": r["expect_any"],
            "answerable": r["answerable"],
        }
        results.append(res)
        flag = "ERR" if err else ("ok " if res["pred_intent"] == r["intent"] else "MISS")
        print(f"[{i:02d}/{len(rows)}] {flag} {r['intent']:<9} -> {res['pred_intent']!s:<9} {r['query'][:60]}", flush=True)
        time.sleep(args.delay)

    ok = [x for x in results if not x["error"]]

    # 1. intent routing
    hits = sum(1 for x in ok if x["pred_intent"] == x["expected_intent"])
    per = defaultdict(lambda: [0, 0])
    for x in ok:
        per[x["expected_intent"]][1] += 1
        per[x["expected_intent"]][0] += x["pred_intent"] == x["expected_intent"]

    # 2. HITL
    tp = sum(1 for x in ok if x["expected_approval"] and x["pred_approval"])
    fp = sum(1 for x in ok if not x["expected_approval"] and x["pred_approval"])
    fn = sum(1 for x in ok if x["expected_approval"] and not x["pred_approval"])
    tn = sum(1 for x in ok if not x["expected_approval"] and not x["pred_approval"])
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")

    # 3. answer correctness (answerable, with an expected fact, non-memory)
    graded = [x for x in ok if x["answerable"] and x["expect_any"] and x["expected_intent"] != "Memory"]
    correct = sum(1 for x in graded if any(norm(k) in norm(x["answer"]) for k in x["expect_any"]))

    # 4. safe failure (unanswerable)
    unans = [x for x in ok if not x["answerable"]]
    safe = sum(1 for x in unans if any(h in norm(x["answer"]) for h in REFUSAL_HINTS))

    # 5. latency / tokens
    lats = [x["latency_s"] for x in ok if x["latency_s"] is not None]
    toks = [x["tokens"] for x in ok if isinstance(x["tokens"], (int, float))]

    print("\n" + "=" * 62, flush=True)
    print(f"Queries run: {len(rows)}   errors: {errors}", flush=True)
    print(f"Intent routing accuracy : {hits}/{len(ok)} = {pct(hits, len(ok))}", flush=True)
    for k, (h, n) in sorted(per.items()):
        print(f"   {k:<10} {h}/{n} = {pct(h, n)}", flush=True)
    print(f"HITL approval  precision={prec:.2f}  recall={rec:.2f}   (TP={tp} FP={fp} FN={fn} TN={tn})", flush=True)
    print(f"Answer correctness      : {correct}/{len(graded)} = {pct(correct, len(graded))}", flush=True)
    print(f"Safe failure (unanswerable): {safe}/{len(unans)} = {pct(safe, len(unans))}  [heuristic]", flush=True)
    if lats:
        print(f"Latency  mean={statistics.mean(lats):.2f}s  p50={percentile(lats, 50):.2f}s  p95={percentile(lats, 95):.2f}s", flush=True)
    if toks:
        print(f"Tokens/turn  mean={statistics.mean(toks):.0f}  p95={percentile(toks, 95):.0f}", flush=True)

    misses = [x for x in ok if x["pred_intent"] != x["expected_intent"]]
    if misses:
        print("\nIntent misses:", flush=True)
        for x in misses:
            print(f"  #{x['id']:>2} expected={x['expected_intent']} got={x['pred_intent']} | {x['query']}", flush=True)
    bad_h = [x for x in ok if x["pred_approval"] != x["expected_approval"]]
    if bad_h:
        print("\nHITL errors:", flush=True)
        for x in bad_h:
            kind = "FALSE POSITIVE" if x["pred_approval"] else "MISSED"
            print(f"  #{x['id']:>2} {kind} | {x['query']}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved per-query results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
