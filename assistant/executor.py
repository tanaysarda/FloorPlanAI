"""Guarded read-only SQL execution: validator + query_only + authorizer + timeout + row cap."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any, List, Sequence

from assistant.query_validator import ALLOWED_RELATIONS, QueryValidationError, validate_sql
from config import QUERY_MAX_ROWS, QUERY_TIMEOUT_S

BLOCKED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


class QueryTimeoutError(Exception):
    pass


class QueryExecutionError(Exception):
    """Execution failed; the message is generic and safe to show."""


@dataclass
class QueryResult:
    columns: List[str]
    rows: List[tuple]
    truncated: bool = False

    def as_dicts(self) -> List[dict]:
        return [dict(zip(self.columns, r)) for r in self.rows]


def _authorizer(action, arg1, arg2, dbname, source):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        if dbname is None:                 # CTE / ephemeral relation: not a real table, nothing to protect
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_OK if (arg1 or "").lower() in ALLOWED_RELATIONS else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if (arg2 or arg1 or "").lower() in BLOCKED_FUNCTIONS else sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY   # every write, DDL, PRAGMA, ATTACH, transaction statement...


def execute_readonly(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = (), *,
                     timeout_s: float = QUERY_TIMEOUT_S, max_rows: int = QUERY_MAX_ROWS) -> QueryResult:
    """Validate then run a query that cannot modify the database."""
    clean = validate_sql(sql)                      # raises QueryValidationError
    prev_qo = conn.execute("PRAGMA query_only").fetchone()[0]
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_authorizer)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    try:
        cur = conn.execute(clean, tuple(params))
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows + 1)
        return QueryResult(cols, [tuple(r) for r in rows[:max_rows]], truncated=len(rows) > max_rows)
    except sqlite3.OperationalError as exc:
        if "interrupt" in str(exc).lower():
            raise QueryTimeoutError("The query took too long and was stopped.") from exc
        raise QueryExecutionError("The query could not be executed.") from exc
    except sqlite3.Error as exc:
        raise QueryExecutionError("The query could not be executed.") from exc
    finally:
        conn.set_progress_handler(None, 0)
        conn.set_authorizer(None)
        conn.execute(f"PRAGMA query_only = {'ON' if prev_qo else 'OFF'}")
