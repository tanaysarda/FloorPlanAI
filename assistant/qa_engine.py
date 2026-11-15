"""Question-answering engine shared by the CLI, the Streamlit app and the evaluator.

Every answer is built from rows returned by a validated, read-only SQL query; the SQL, its parameters and
the evidence rows are always returned alongside the answer text."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from assistant.executor import QueryExecutionError, QueryResult, QueryTimeoutError, execute_readonly
from assistant.llm import LLMError, LLMProvider
from assistant.query_validator import QueryValidationError
from assistant.sql_generator import Intent, Unanswerable, build_query, extract_plan_ids, parse_intent
from database import db

log = logging.getLogger(__name__)
NOUN = {"room": "room", "door": "door", "window": "window", "wall": "wall region"}


@dataclass
class QAResult:
    question: str
    status: str            # answered | no_results | unsupported | invalid_plan | clarification | invalid_input | error
    answer: str
    sql: Optional[str] = None
    params: tuple = ()
    columns: List[str] = field(default_factory=list)
    rows: List[tuple] = field(default_factory=list)
    source: str = "none"   # deterministic | llm | none
    notes: List[str] = field(default_factory=list)
    truncated: bool = False

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [dict(zip(self.columns, r)) for r in self.rows]

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "status": self.status, "answer": self.answer, "sql": self.sql,
                "params": list(self.params), "columns": self.columns, "rows": [list(r) for r in self.rows],
                "source": self.source, "notes": self.notes, "truncated": self.truncated}


def _n(v) -> str:
    return f"{v:,}" if isinstance(v, int) else (f"{v:,.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))


def _plural(count, word: str) -> str:
    return f"{_n(count)} {word}{'' if count == 1 else 's'}"


def format_answer(intent: Intent, res: QueryResult) -> tuple[str, str]:
    """Return (status, answer text) built strictly from the result rows."""
    d, n = res.as_dicts(), intent.name
    noun = NOUN.get(intent.metric or "", "")
    pid = intent.plan_id
    if n == "count":
        if not d:
            return "no_results", f"No data found for plan {pid}."
        return "answered", f"Plan {pid} has {_plural(d[0]['value'], noun)}." + (
            " (Walls are counted as connected wall regions, not individual straight segments.)" if noun.startswith("wall") else "")
    if n == "count_total":
        if not d or d[0]["plan_count"] == 0:
            return "no_results", "There are no floor plans in the database."
        return "answered", f"Across all {d[0]['plan_count']} plans there are {_plural(d[0]['value'] or 0, noun)}."
    if n == "count_label":
        cnt = d[0]["value"] if d else 0
        return "answered", f"Plan {pid} has {_plural(cnt, intent.label)}."
    if n in ("room_extreme",):
        kind = "largest" if intent.extreme == "max" else "smallest"
        if not d:
            return "no_results", f"No rooms found{'' if pid is None else f' in plan {pid}'}."
        names = [f"{r['room_name']}" + ("" if pid is not None else f" (plan {r['plan_id']})") for r in d]
        where = f"in plan {pid}" if pid is not None else "across all plans"
        tie = " (tied)" if len(d) > 1 else ""
        return "answered", (f"The {kind} room {where} is {' and '.join(names)}{tie}, with an area of "
                            f"{_n(d[0]['area_px'])} pixels (pixel area; no real-world scale is available).")
    if n == "plans_filter":
        word = {">": "more than", ">=": "at least", "<": "fewer than", "<=": "at most", "=": "exactly"}[intent.op]
        crit = f"{word} {intent.number} {intent.metric}s"
        if not d:
            return "no_results", f"No plans have {crit}."
        items = ", ".join(f"plan {r['plan_id']} ({r['value']})" for r in d)
        return "answered", f"{_plural(len(d), 'plan')} with {crit}: {items}."
    if n == "rooms_without":
        what = f"{intent.metric}s"
        where = f"in plan {pid}" if pid is not None else "in the database"
        if not d:
            return "no_results", f"No rooms {where} are without a detected {intent.metric}."
        items = ", ".join(r["room_name"] + ("" if pid is not None else f" (plan {r['plan_id']})") for r in d)
        return "answered", (f"{_plural(len(d), 'room')} {where} with no detected {what}: {items}. "
                            f"(Based on {what} detected in the masks.)")
    if n == "avg_count":
        if not d or d[0]["plan_count"] == 0 or d[0]["value"] is None:
            return "no_results", "There are no floor plans in the database, so no average can be computed."
        return "answered", f"The average number of {intent.metric}s per plan across {d[0]['plan_count']} plans is {_n(d[0]['value'])}."
    if n == "avg_room_area":
        if not d or d[0]["value"] is None:
            return "no_results", "No rooms found, so no average area can be computed."
        where = f"in plan {pid}" if pid is not None else "across all plans"
        return "answered", f"The average room area {where} is {_n(d[0]['value'])} pixels (pixel area; no real-world scale is available)."
    if n == "plan_extreme":
        kind = "most" if intent.extreme == "max" else "fewest"
        if not d:
            return "no_results", "There are no floor plans in the database."
        plans = " and ".join(f"plan {r['plan_id']}" for r in d)
        return "answered", f"{plans} {'has' if len(d) == 1 else 'have'} the {kind} {intent.metric}s ({d[0]['value']})."
    if n == "total_plans":
        return "answered", f"The database contains {_plural(d[0]['value'], 'floor plan')}."
    if n == "list_rooms":
        if not d:
            return "no_results", f"No rooms found in plan {pid}."
        items = "; ".join(f"{r['room_name']} ({_n(r['area_px'])} px)" for r in d)
        return "answered", f"Plan {pid} has {_plural(len(d), 'room')}: {items}."
    if n == "total_area":
        if not d:
            return "no_results", f"No data found for plan {pid}."
        return "answered", f"The total room area in plan {pid} is {_n(d[0]['value'])} pixels (pixel area; no real-world scale is available)."
    return "error", "Unhandled question type."


def generic_answer(res: QueryResult) -> tuple[str, str]:
    """Answer text for free-form (LLM-generated) SQL: states only what the rows contain."""
    if not res.rows:
        return "no_results", "The query returned no rows, so there is nothing in the data matching that question."
    if len(res.rows) == 1 and len(res.columns) == 1:
        return "answered", f"Result: {_n(res.rows[0][0])}."
    shown = "; ".join(", ".join(f"{c}={_n(v)}" for c, v in zip(res.columns, r)) for r in res.rows[:5])
    more = f" (showing 5 of {len(res.rows)})" if len(res.rows) > 5 else ""
    return "answered", f"The query returned {_plural(len(res.rows), 'row')}{more}: {shown}."


class QAEngine:
    """mode: 'auto' (LLM if configured, else deterministic), 'deterministic', or 'llm'."""

    def __init__(self, db_path: Path | str, provider: Optional[LLMProvider] = None, mode: str = "auto"):
        self.db_path, self.provider, self.mode = Path(db_path), provider, mode

    # ---- helpers
    def _conn(self):
        return db.connect(self.db_path, read_only=True)

    def schema_text(self, conn) -> str:
        rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table','view') "
                            "AND name IN ('floor_plans','rooms','doors','windows','walls','door_rooms',"
                            "'window_rooms','room_connections','plan_summary','room_details') ORDER BY name").fetchall()
        return "\n".join(f"{s};" for _, s in rows)

    def _plan_ids(self, conn) -> List[int]:
        return db.list_plan_ids(conn)

    def _run(self, conn, question, sql, params, formatter, source) -> QAResult:
        res = execute_readonly(conn, sql, params)
        status, text = formatter(res)
        return QAResult(question, status, text, sql, tuple(params), res.columns, res.rows, source,
                        truncated=res.truncated)

    # ---- public API
    def ask(self, question: str, selected_plan_id: Optional[int] = None) -> QAResult:
        question = question if isinstance(question, str) else ""
        try:
            conn = self._conn()
        except db.DatabaseError:
            return QAResult(question, "error", "The database is not available yet. Build it first "
                            "(python cli.py build-synthetic, or ingest the real dataset).")
        try:
            plan_ids = self._plan_ids(conn)
            requested = extract_plan_ids(question)
            if len(requested) == 1 and requested[0] not in plan_ids:
                return self._invalid_plan(conn, question, requested[0], plan_ids)
            use_llm = self.provider is not None and self.mode in ("auto", "llm")
            notes: List[str] = []
            if use_llm:
                try:
                    out = self._ask_llm(conn, question)
                    if out is not None:
                        return out
                except (LLMError, QueryValidationError, QueryExecutionError, QueryTimeoutError) as exc:
                    log.warning("LLM path failed (%s); using deterministic fallback", type(exc).__name__)
                    notes.append("The AI model was unavailable or returned an unusable query; "
                                 "answered with the built-in deterministic engine instead.")
            out = self._ask_deterministic(conn, question, selected_plan_id, plan_ids)
            out.notes.extend(notes)
            return out
        except (QueryValidationError, QueryExecutionError, QueryTimeoutError) as exc:
            log.warning("query failed: %r", exc)
            return QAResult(question, "error", "Sorry, that question could not be answered because the query failed.")
        except Exception:  # noqa: BLE001 - never leak internals to the user
            log.exception("unexpected error answering question")
            return QAResult(question, "error", "Sorry, something went wrong while answering. Please try again.")
        finally:
            conn.close()

    def _invalid_plan(self, conn, question, pid, plan_ids) -> QAResult:
        res = execute_readonly(conn, "SELECT id FROM floor_plans WHERE id = ?", (pid,))
        avail = (f"Available plan numbers: {plan_ids[0]}–{plan_ids[-1]}" if plan_ids else "The database has no floor plans")
        return QAResult(question, "invalid_plan", f"Plan {pid} does not exist. {avail}.",
                        "SELECT id FROM floor_plans WHERE id = ?", (pid,), res.columns, res.rows, "deterministic")

    def _ask_deterministic(self, conn, question, selected_plan_id, plan_ids) -> QAResult:
        labels = [r[0] for r in conn.execute("SELECT DISTINCT label FROM rooms WHERE label IS NOT NULL")]
        intent = parse_intent(question, labels, selected_plan_id)
        if isinstance(intent, Unanswerable):
            msg = intent.message
            if intent.status == "unsupported" and "can't map" in msg and labels:
                msg += f" Room types in the data: {', '.join(sorted(labels))}."
            return QAResult(question, intent.status, msg, source="deterministic")
        if intent.plan_id is not None and intent.plan_id not in plan_ids:
            return self._invalid_plan(conn, question, intent.plan_id, plan_ids)
        sql, params = build_query(intent)
        return self._run(conn, question, sql, params, lambda r: format_answer(intent, r), "deterministic")

    def _ask_llm(self, conn, question) -> Optional[QAResult]:
        decision = self.provider.generate(question, self.schema_text(conn))
        if decision.kind == "decline":
            return QAResult(question, "unsupported",
                            "This can't be determined from the available data. " + (decision.reason or "").strip()[:300],
                            source="llm")
        return self._run(conn, question, decision.sql, (), generic_answer, "llm")
