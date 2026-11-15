"""Generate evaluation/test_questions_synthetic.json from the generator's INDEPENDENT ground truth
(data/synthetic/ground_truth.json), never from the database or the QA engine.
Run once after changing the synthetic fixtures:  python -m evaluation.build_synthetic_expectations"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import config
from processing.synthetic import generate_fixtures

OUT = config.ROOT / "evaluation" / "test_questions_synthetic.json"


def r2(num: int, den: int) -> float:
    """Half-up rounding to 2 dp (matches SQLite ROUND; Python's round() is banker's/float-based)."""
    return float((Decimal(num) / Decimal(den)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def build(gt: dict) -> dict:
    plans = {i + 1: p for i, p in enumerate(sorted(gt["plans"], key=lambda p: p["name"]))}
    nrooms = {i: len(p["rooms"]) for i, p in plans.items()}
    ndoors = {i: len(p["doors"]) for i, p in plans.items()}
    nwin = {i: len(p["windows"]) for i, p in plans.items()}
    def rname(p, k): return f"{p['rooms'][k]['label']} #{k + 1}"   # grid cells are already in reading order
    def largest(i): r = max(plans[i]["rooms"], key=lambda r: r["area_px"]); return rname(plans[i], r["cell_index"]), r["area_px"]
    def smallest(i): r = min(plans[i]["rooms"], key=lambda r: r["area_px"]); return rname(plans[i], r["cell_index"]), r["area_px"]
    def nowin(i):
        withw = {w["room"] for w in plans[i]["windows"]}
        return [rname(plans[i], r["cell_index"]) for r in plans[i]["rooms"] if r["cell_index"] not in withw]
    def count_label(i, lab): return sum(1 for r in plans[i]["rooms"] if r["label"] == lab)
    gt_over3 = sorted(i for i in plans if ndoors[i] > 3)
    gt_win_ge4 = sorted(i for i in plans if nwin[i] >= 4)
    gt_rooms_lt4 = sorted(i for i in plans if nrooms[i] < 4)
    maxr = max(nrooms.values()); maxd = max(ndoors.values())
    all_nowin = sorted((i, n) for i in plans for n in nowin(i))
    l7, s3 = largest(7), smallest(3)
    Q = []
    def add(cat, q, expected, status="answered"):
        Q.append({"id": len(Q) + 1, "category": cat, "question": q, "expected_status": status, "expected": expected})
    add("count", "How many rooms does plan 3 have?", {"type": "scalar", "column": "value", "value": nrooms[3]})
    add("count", "How many rooms are in floor plan 6?", {"type": "scalar", "column": "value", "value": nrooms[6]})
    add("count", "How many doors does plan 8 have?", {"type": "scalar", "column": "value", "value": ndoors[8]})
    add("count", "How many windows does plan 2 have?", {"type": "scalar", "column": "value", "value": nwin[2]})
    add("count", "How many plans are there?", {"type": "scalar", "column": "value", "value": len(plans)})
    add("count", "How many bedrooms are in plan 6?", {"type": "scalar", "column": "value", "value": count_label(6, "bedroom")})
    add("count", "How many bathrooms does plan 1 have?", {"type": "scalar", "column": "value", "value": count_label(1, "bathroom")})
    add("largest_smallest", "What is the largest room in plan 7?", {"type": "row", "values": {"room_name": l7[0], "area_px": l7[1]}})
    add("largest_smallest", "What is the smallest room in plan 3?", {"type": "row", "values": {"room_name": s3[0], "area_px": s3[1]}})
    add("filtering", "Which plans have more than 3 doors?", {"type": "set", "column": "plan_id", "values": gt_over3})
    add("filtering", "Which floor plans contain at least four windows?", {"type": "set", "column": "plan_id", "values": gt_win_ge4})
    add("filtering", "Which plans have fewer than 4 rooms?", {"type": "set", "column": "plan_id", "values": gt_rooms_lt4})
    add("windows", "Which room in plan 4 has no windows?", {"type": "set", "column": "room_name", "values": nowin(4)})
    add("windows", "Which rooms have no windows?", {"type": "set", "column": "room_name",
                                                    "values": [f"{n}" for _, n in all_nowin]})
    add("windows", "Which rooms in plan 1 have no windows?", {"type": "set", "column": "room_name", "values": nowin(1)})
    add("averages", "What is the average number of rooms across all plans?", {"type": "scalar", "column": "value",
                                                                              "value": r2(sum(nrooms.values()), len(plans))})
    add("averages", "What is the average number of doors per plan?", {"type": "scalar", "column": "value",
                                                                       "value": r2(sum(ndoors.values()), len(plans))})
    add("averages", "What is the average number of windows across all plans?", {"type": "scalar", "column": "value",
                                                                                "value": r2(sum(nwin.values()), len(plans))})
    add("extremes", "Which plan has the most rooms?", {"type": "set", "column": "plan_id", "values": sorted(i for i in plans if nrooms[i] == maxr)})
    add("extremes", "Which plan has the most doors?", {"type": "set", "column": "plan_id", "values": sorted(i for i in plans if ndoors[i] == maxd)})
    add("empty_result", "Which plans have more than 20 doors?", {"type": "empty"}, "no_results")
    add("invalid_plan", "How many rooms does plan 12 have?", {"type": "status"}, "invalid_plan")
    add("invalid_plan", "What is the largest room in plan 99?", {"type": "status"}, "invalid_plan")
    add("ambiguous", "How many doors are there?", {"type": "status"}, "clarification")
    add("unsupported", "What is the area of the living room in plan 2 in square metres?", {"type": "status"}, "unsupported")
    add("unsupported", "Where is the main entrance in plan 5?", {"type": "status"}, "unsupported")
    add("unsupported", "What colour are the walls in plan 1?", {"type": "status"}, "unsupported")
    add("unsupported", "How much furniture is in the living room of plan 3?", {"type": "status"}, "unsupported")
    return {"dataset": "synthetic", "note": "SYNTHETIC TEST DATA expectations derived from the generator ground truth, "
            "not from the database or the QA engine.", "questions": Q}


def main() -> None:
    gt = generate_fixtures(config.SYNTHETIC_DIR)
    doc = build(gt)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {len(doc['questions'])} questions to {OUT}")


if __name__ == "__main__":
    main()
