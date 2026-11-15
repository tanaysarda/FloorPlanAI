"""Static validation of SQL before execution. Defence in depth: executor.py adds an authorizer,
a read-only connection, a timeout and a row cap, so this layer is not the only safeguard."""
from __future__ import annotations

import re
from typing import Set

ALLOWED_RELATIONS = frozenset({
    "floor_plans", "rooms", "doors", "windows", "walls", "door_rooms", "window_rooms",
    "room_connections", "plan_summary", "room_details",
})
FORBIDDEN_KEYWORDS = frozenset({
    "insert", "update", "delete", "drop", "alter", "create", "replace", "truncate", "attach", "detach",
    "pragma", "vacuum", "reindex", "analyze", "begin", "commit", "rollback", "savepoint", "release",
    "explain", "load_extension", "upsert", "merge", "grant", "revoke",
})
MAX_SQL_LENGTH = 4000


class QueryValidationError(ValueError):
    """The SQL is not an allowed read-only query. The message is safe to show to users."""


_STRING = re.compile(r"'(?:[^']|'')*'")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FROM_JOIN = re.compile(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)", re.I)
_CTE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^()]*\))?\s+as\s*\(", re.I)


def validate_sql(sql: str) -> str:
    """Return the cleaned single SELECT statement or raise QueryValidationError."""
    if not isinstance(sql, str) or not sql.strip():
        raise QueryValidationError("The query is empty.")
    if len(sql) > MAX_SQL_LENGTH:
        raise QueryValidationError("The query is too long.")
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise QueryValidationError("SQL comments are not allowed.")
    if "\x00" in sql:
        raise QueryValidationError("The query contains invalid characters.")
    text = sql.strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    stripped = _STRING.sub(" 0 ", text)         # ignore keywords that appear inside string literals
    if "'" in stripped:
        raise QueryValidationError("The query has an unterminated string.")
    if ";" in stripped:
        raise QueryValidationError("Only a single statement is allowed.")
    words = [w.lower() for w in _WORD.findall(stripped)]
    if not words or words[0] not in ("select", "with"):
        raise QueryValidationError("Only read-only SELECT queries are allowed.")
    bad = sorted({w for w in words if w in FORBIDDEN_KEYWORDS})
    if bad:
        raise QueryValidationError(f"Forbidden SQL keyword(s): {', '.join(bad).upper()}.")
    if any(w.startswith("sqlite_") for w in words):
        raise QueryValidationError("Access to SQLite internals is not allowed.")
    ctes: Set[str] = {m.group(1).lower() for m in _CTE.finditer(stripped)}
    for m in _FROM_JOIN.finditer(stripped):
        name = m.group(1).lower()
        if name not in ALLOWED_RELATIONS and name not in ctes:
            raise QueryValidationError(f"Table or view '{name}' is not available for querying.")
    return text
