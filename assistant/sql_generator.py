"""Deterministic natural-language -> intent -> parameterised SQL (no LLM needed).

Every supported question maps to a fixed, parameterised query. Anything else is declined rather than guessed."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "twenty": 20}
METRIC_COLUMN = {"room": "room_count", "door": "door_count", "window": "window_count", "wall": "wall_count"}
OPS = [(r"no more than|at most|up to|a maximum of|maximum of", "<="),
       (r"more than|greater than|over|above|exceed(?:s|ing)?", ">"),
       (r"at least|no fewer than|no less than|a minimum of|minimum of", ">="),
       (r"fewer than|less than|under|below", "<"),
       (r"exactly|equal to", "=")]
_OP_RE = "|".join(f"(?P<op{i}>{p})" for i, (p, _) in enumerate(OPS))
_NUM_RE = r"(?P<n>\d+|" + "|".join(NUMBER_WORDS) + r")"
FILTER_RE = re.compile(rf"\b(?:{_OP_RE})\s+{_NUM_RE}\s+(?P<metric>rooms?|doors?|windows?|walls?)\b")
PLAN_ID_RE = re.compile(r"\b(?:floor\s*)?plans?\s*(?:#|no\.?|number|id)?\s*(\d+)\b")

UNIT_RE = re.compile(r"square\s*(?:met(?:er|re)s?|feet|foot|ft|yards?)|\bsq\.?\s*(?:m|ft|metres?|meters?|feet)\b|"
                     r"m²|\bm2\b|\bsqft\b|\bmet(?:er|re)s?\b|\bfeet\b|\bfoot\b|\bcentimet(?:er|re)s?\b|\binch(?:es)?\b")
NAV_RE = re.compile(r"how (?:do|can|would|should) (?:i|we) (?:get|go|walk)|\broute\b|\bpath (?:from|to)\b|"
                    r"\bdirections?\b|\bnavigate\b|\bentrance\b|\bexit\b|\bfront door\b|\bmain door\b")
OTHER_UNSUPPORTED_RE = re.compile(
    r"\bfurniture\b|\bsofa\b|\bbed\b|\btables?\b|\bcolou?rs?\b|\bprice\b|\bcost\b|\bceiling\b|\bheight\b|"
    r"\bmaterials?\b|\bflooring\b|\bpaint\b|\baddress\b|\bowner\b|\barchitect\b|\borientation\b|\bfacing\b|"
    r"\bnorth\b|\bsouth\b|\beast\b|\bwest\b|\bsunlight\b|\belectrical\b|\bplumbing\b|\boccupan|\bpeople\b|\bbuilt in\b")

SUPPORTED_HELP = [
    "How many rooms/doors/windows does plan 12 have?",
    "What is the largest (or smallest) room in plan 7?",
    "Which plans have more than 3 doors?",
    "Which rooms have no windows? (optionally: in plan 4)",
    "What is the average number of rooms across all plans?",
    "Which plan has the most rooms?  /  List the rooms in plan 3.",
    "How many bedrooms are in plan 2?  /  What is the total room area (pixels) of plan 5?",
]


@dataclass
class Intent:
    name: str
    plan_id: Optional[int] = None
    metric: Optional[str] = None      # room | door | window | wall
    op: Optional[str] = None
    number: Optional[int] = None
    label: Optional[str] = None
    extreme: Optional[str] = None     # max | min
    scope_all: bool = False
    plan_ids_mentioned: List[int] = field(default_factory=list)


@dataclass
class Unanswerable:
    status: str       # unsupported | clarification | invalid_input
    message: str


def extract_plan_ids(question: str) -> List[int]:
    seen: List[int] = []
    for m in PLAN_ID_RE.findall(question.lower()):
        v = int(m)
        if v not in seen:
            seen.append(v)
    return seen


def _normalise(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().replace("’", "'")).strip(" ?!.")


def _metric(text: str) -> Optional[str]:
    for word, key in (("room", "room"), ("door", "door"), ("window", "window"), ("wall", "wall")):
        if re.search(rf"\b{word}s?\b", text):
            return key
    return None


def parse_intent(question: str, known_labels: Sequence[str] = (), selected_plan_id: Optional[int] = None
                 ) -> Union[Intent, Unanswerable]:
    q = _normalise(question)
    if not q:
        return Unanswerable("invalid_input", "Please type a question about the floor plans.")
    if len(q) > 500:
        return Unanswerable("invalid_input", "That question is too long. Please keep it under 500 characters.")

    if UNIT_RE.search(q):
        return Unanswerable("unsupported",
                            "This can't be determined from the available data: no real-world scale is stored for "
                            "these floor plans, so areas and lengths exist only in pixels, not metres or feet. "
                            "Ask for the area in pixels instead.")
    if NAV_RE.search(q):
        return Unanswerable("unsupported",
                            "This can't be determined from the available data: the data does not identify an "
                            "entrance, and route/navigation answers are not implemented in this build.")
    if OTHER_UNSUPPORTED_RE.search(q):
        return Unanswerable("unsupported",
                            "This can't be determined from the available data: the database only holds rooms, "
                            "doors, windows and walls (counts, pixel areas and positions).")

    ids = extract_plan_ids(q)
    if re.search(r"\b(?:this|selected|current) (?:floor )?plan\b", q) and selected_plan_id is not None and not ids:
        ids = [selected_plan_id]
    all_scope = bool(re.search(r"\b(?:all|every|each|any)\b.*\bplans?\b|\bacross\b|\boverall\b|\bin total\b|"
                               r"\bdataset\b|\btotal number\b", q)) or bool(re.search(r"\bplans\b", q) and not ids)
    base = dict(plan_ids_mentioned=ids)
    if len(ids) > 1 and not re.search(r"\b(?:compare|both|and)\b", q):
        return Unanswerable("clarification", f"Your question mentions several plans ({', '.join(map(str, ids))}). "
                                             "Please ask about one plan at a time.")
    if len(ids) > 1:
        return Unanswerable("clarification", "Please ask about one plan at a time (comparisons are not supported).")
    pid = ids[0] if ids else None

    m = FILTER_RE.search(q)
    if m and not pid and re.search(r"\bplans?\b", q):
        op = next(OPS[i][1] for i in range(len(OPS)) if m.group(f"op{i}"))
        n = m.group("n")
        return Intent("plans_filter", metric=_metric(m.group("metric")), op=op,
                      number=int(n) if n.isdigit() else NUMBER_WORDS[n], **base)

    if re.search(r"\b(?:average|mean|avg)\b", q):
        if re.search(r"\barea\b", q):
            return Intent("avg_room_area", plan_id=pid, **{k: v for k, v in base.items()})
        met = _metric(q)
        if met in ("room", "door", "window", "wall") and not pid:
            return Intent("avg_count", metric=met, **base)
        return Unanswerable("unsupported", "I can compute the average number of rooms/doors/windows per plan or "
                                           "the average room area (pixels); please rephrase.")

    m = re.search(r"\bwhich (?:floor )?plans? (?:has|have|contains?|had) the (most|fewest|least|highest number of|"
                  r"lowest number of) (rooms?|doors?|windows?|walls?)\b", q)
    if m:
        ext = "max" if m.group(1) in ("most", "highest number of") else "min"
        return Intent("plan_extreme", metric=_metric(m.group(2)), extreme=ext, **base)

    m = re.search(r"\b(largest|biggest|smallest|tiniest)\b(?: \w+)? room\b", q)
    if m:
        ext = "max" if m.group(1) in ("largest", "biggest") else "min"
        if pid is None and not all_scope:
            return Unanswerable("clarification", "Which floor plan do you mean? Please include a plan number, "
                                                 "e.g. 'largest room in plan 7' (or say 'across all plans').")
        return Intent("room_extreme", plan_id=pid, extreme=ext, scope_all=pid is None, **{k: v for k, v in base.items()})

    m = re.search(r"\b(?:no|without|lack(?:s|ing)?|missing|zero) (windows?|doors?)\b|"
                  r"\b(?:do not|don't|does not|doesn't) have (?:any )?(windows?|doors?)\b", q)
    if m and re.search(r"\brooms?\b", q):
        what = (m.group(1) or m.group(2))
        return Intent("rooms_without", plan_id=pid, metric="window" if what.startswith("window") else "door", **base)

    if re.search(r"\btotal (?:room )?area\b|\bsum of (?:the )?(?:room )?areas?\b", q):
        if pid is None:
            return Unanswerable("clarification", "Which floor plan do you mean? Please include a plan number.")
        return Intent("total_area", plan_id=pid, **base)

    if re.search(r"\bhow many (?:floor )?plans\b|\bnumber of (?:floor )?plans\b", q) and not pid:
        return Intent("total_plans", **base)

    if re.search(r"\b(?:list|show|name|what|which)\b.*\brooms\b", q) and pid and not re.search(r"\bhow many\b", q):
        return Intent("list_rooms", plan_id=pid, **base)

    count = re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q)
    if count:
        for lab in sorted(known_labels, key=len, reverse=True):
            if lab and re.search(rf"\b{re.escape(lab.lower())}(?:s|es)?\b", q):
                if pid is None:
                    return Unanswerable("clarification", "Which floor plan do you mean? Please include a plan number.")
                return Intent("count_label", plan_id=pid, label=lab, **base)
        met = _metric(q)
        if met:
            if pid is None and all_scope:
                return Intent("count_total", metric=met, **base)
            if pid is None:
                return Unanswerable("clarification", "Which floor plan do you mean? Please include a plan number, "
                                                     "e.g. 'plan 12' (or say 'across all plans').")
            return Intent("count", plan_id=pid, metric=met, **base)

    return Unanswerable("unsupported", "I can't map that question to the data I have. Questions I can answer "
                                       "include: " + " ".join(f"({i + 1}) {s}" for i, s in enumerate(SUPPORTED_HELP)))


def build_query(intent: Intent) -> Tuple[str, Tuple]:
    """Return (parameterised SQL, params) for an Intent. Column names come from fixed whitelists only."""
    n = intent.name
    col = METRIC_COLUMN.get(intent.metric or "", None)
    if n == "count":
        return f"SELECT plan_id, {col} AS value FROM plan_summary WHERE plan_id = ?", (intent.plan_id,)
    if n == "count_total":
        return f"SELECT SUM({col}) AS value, COUNT(*) AS plan_count FROM plan_summary", ()
    if n == "count_label":
        return ("SELECT plan_id, COUNT(*) AS value FROM rooms WHERE plan_id = ? AND LOWER(label) = LOWER(?) "
                "GROUP BY plan_id", (intent.plan_id, intent.label))
    if n == "room_extreme":
        fn, order = ("MAX", "DESC") if intent.extreme == "max" else ("MIN", "ASC")
        if intent.plan_id is None:
            return (f"SELECT plan_id, room_name, label, area_px FROM room_details WHERE area_px = "
                    f"(SELECT {fn}(area_px) FROM room_details) ORDER BY plan_id, room_index", ())
        return (f"SELECT plan_id, room_name, label, area_px FROM room_details WHERE plan_id = ? AND area_px = "
                f"(SELECT {fn}(area_px) FROM room_details WHERE plan_id = ?) ORDER BY room_index",
                (intent.plan_id, intent.plan_id))
    if n == "plans_filter":
        return (f"SELECT plan_id, name, {col} AS value FROM plan_summary WHERE {col} {intent.op} ? "
                f"ORDER BY plan_id", (intent.number,))
    if n == "rooms_without":
        cc = "window_count" if intent.metric == "window" else "door_count"
        if intent.plan_id is None:
            return (f"SELECT plan_id, room_name, label, area_px FROM room_details WHERE {cc} = 0 "
                    f"ORDER BY plan_id, room_index", ())
        return (f"SELECT plan_id, room_name, label, area_px FROM room_details WHERE {cc} = 0 AND plan_id = ? "
                f"ORDER BY room_index", (intent.plan_id,))
    if n == "avg_count":
        return f"SELECT ROUND(AVG({col}), 2) AS value, COUNT(*) AS plan_count FROM plan_summary", ()
    if n == "avg_room_area":
        if intent.plan_id is None:
            return "SELECT ROUND(AVG(area_px), 1) AS value, COUNT(*) AS room_count FROM rooms", ()
        return ("SELECT plan_id, ROUND(AVG(area_px), 1) AS value, COUNT(*) AS room_count FROM rooms "
                "WHERE plan_id = ? GROUP BY plan_id", (intent.plan_id,))
    if n == "plan_extreme":
        fn = "MAX" if intent.extreme == "max" else "MIN"
        return (f"SELECT plan_id, name, {col} AS value FROM plan_summary WHERE {col} = "
                f"(SELECT {fn}({col}) FROM plan_summary) ORDER BY plan_id", ())
    if n == "total_plans":
        return "SELECT COUNT(*) AS value FROM floor_plans", ()
    if n == "list_rooms":
        return ("SELECT plan_id, room_name, label, area_px FROM room_details WHERE plan_id = ? "
                "ORDER BY room_index", (intent.plan_id,))
    if n == "total_area":
        return "SELECT plan_id, total_room_area_px AS value FROM plan_summary WHERE plan_id = ?", (intent.plan_id,)
    raise ValueError(f"unknown intent {n!r}")
