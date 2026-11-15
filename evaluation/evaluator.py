"""Reproducible evaluation of the QA engine against stored expected answers.

Metric (clearly defined): a question is CORRECT iff
  (1) the engine's status equals expected_status, AND
  (2) the STRUCTURED result (SQL result rows) matches the expected value
      - scalar: first row's `column` equals value (floats compared to 1e-6)
      - set:    the set of `column` values over all rows equals the expected set
      - row:    the first row contains every expected column=value pair
      - empty:  zero rows
      - status: only the status is checked (refusals, invalid plans, clarifications)
Accuracy = correct / total * 100. Answer WORDING is never compared. A separate 'grounded' flag records
that answered questions carry SQL and rows (and, for scalars, that the number appears in the answer text)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from assistant.qa_engine import QAEngine, QAResult


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    return a == b


def check(expected: Dict[str, Any], res: QAResult) -> Tuple[bool, str]:
    rows = res.as_dicts()
    t = expected["type"]
    if t == "status":
        return True, ""
    if t == "empty":
        return (not rows, "" if not rows else f"expected no rows, got {len(rows)}")
    if not rows:
        return False, "no rows returned"
    if t == "scalar":
        got = rows[0].get(expected["column"])
        return _eq(got, expected["value"]), f"expected {expected['value']!r}, got {got!r}"
    if t == "set":
        got = {r.get(expected["column"]) for r in rows}
        want = set(expected["values"])
        return got == want, f"expected {sorted(map(str, want))}, got {sorted(map(str, got))}"
    if t == "row":
        first = rows[0]
        bad = {k: (v, first.get(k)) for k, v in expected["values"].items() if not _eq(first.get(k), v)}
        return not bad, f"mismatched columns (expected, got): {bad}"
    return False, f"unknown expectation type {t!r}"


def grounded(expected: Dict[str, Any], res: QAResult) -> bool:
    if res.status != "answered":
        return True
    if not res.sql or not res.rows:
        return False
    if expected["type"] == "scalar":
        v = res.as_dicts()[0].get(expected["column"])
        return (f"{v:,}" if isinstance(v, int) else f"{v:,.2f}".rstrip("0").rstrip(".")) in res.answer or str(v) in res.answer
    return True


def run_evaluation(engine: QAEngine, questions_path: Path | str) -> Dict[str, Any]:
    doc = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    results: List[Dict[str, Any]] = []
    for q in doc["questions"]:
        res = engine.ask(q["question"])
        status_ok = res.status == q["expected_status"]
        match_ok, why = check(q["expected"], res) if status_ok else (False, "")
        ok = status_ok and match_ok
        reason = "" if ok else (f"status {res.status!r} != expected {q['expected_status']!r}" if not status_ok else why)
        results.append({"id": q["id"], "category": q["category"], "question": q["question"], "correct": ok,
                        "expected_status": q["expected_status"], "actual_status": res.status,
                        "expected": q["expected"], "actual_rows": [list(r) for r in res.rows[:10]],
                        "actual_columns": res.columns, "sql": res.sql, "answer": res.answer,
                        "grounded": grounded(q["expected"], res), "reason": reason})
    total = len(results)
    correct = sum(r["correct"] for r in results)
    by_cat: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += int(r["correct"])
    return {
        "dataset": doc.get("dataset"), "note": doc.get("note"),
        "summary": {"total": total, "correct": correct, "incorrect": total - correct,
                    "accuracy_pct": round(100 * correct / total, 1) if total else 0.0,
                    "grounded_answers": sum(r["grounded"] for r in results)},
        "by_category": {k: {"correct": v[0], "total": v[1]} for k, v in sorted(by_cat.items())},
        "failures": [r for r in results if not r["correct"]], "results": results,
    }


def write_report(report: Dict[str, Any], out_dir: Path | str) -> Tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp, mp = out / "evaluation_results.json", out / "evaluation_report.md"
    jp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    s = report["summary"]
    lines = [f"# Evaluation report ({report.get('dataset')})", "", f"> {report.get('note') or ''}", "",
             f"- Total questions: **{s['total']}**", f"- Correct: **{s['correct']}**",
             f"- Incorrect: **{s['incorrect']}**", f"- Accuracy: **{s['accuracy_pct']}%**",
             f"- Answered questions traceable to SQL + rows: {s['grounded_answers']}/{s['total']}", "",
             "| Category | Correct | Total |", "|---|---|---|"]
    lines += [f"| {k} | {v['correct']} | {v['total']} |" for k, v in report["by_category"].items()]
    lines += ["", "## Failure cases", ""]
    lines += [f"- #{f['id']} `{f['question']}` — {f['reason']}" for f in report["failures"]] or ["None."]
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jp, mp
