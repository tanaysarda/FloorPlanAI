import sqlite3
import unittest

from assistant import executor
from assistant.executor import QueryTimeoutError, execute_readonly
from assistant.query_validator import QueryValidationError, validate_sql
from tests.helpers import Workspace, table_counts

BAD = [
    "DELETE FROM rooms", "delete from rooms where 1=1", "UPDATE rooms SET area_px = 0",
    "INSERT INTO rooms(plan_id) VALUES (1)", "DROP TABLE rooms", "ALTER TABLE rooms ADD COLUMN x INT",
    "CREATE TABLE t(x)", "REPLACE INTO meta VALUES ('a','b')", "PRAGMA writable_schema = ON",
    "ATTACH DATABASE 'x.db' AS x", "VACUUM", "SELECT 1; DROP TABLE rooms", "SELECT 1; SELECT 2",
    "WITH x AS (SELECT 1) DELETE FROM rooms", "SELECT * FROM sqlite_master", "SELECT * FROM meta",
    "SELECT * FROM rooms -- comment", "SELECT /* c */ 1", "", "   ", "EXPLAIN SELECT 1", "BEGIN",
    "SELECT load_extension('x')", "SELECT * FROM rooms JOIN secret ON 1=1", "'; DROP TABLE rooms; --",
    "SELECT 'unterminated", "SELECT * FROM pragma_table_info('rooms')",
]
GOOD = [
    "SELECT COUNT(*) FROM rooms", "select plan_id from plan_summary where room_count > ?;",
    "SELECT r.label, d.id FROM rooms r JOIN door_rooms d ON d.room_id = r.id",
    "SELECT * FROM rooms WHERE label = 'drop table'",           # keyword inside a string literal is fine
    "WITH big AS (SELECT * FROM rooms WHERE area_px > 5) SELECT COUNT(*) FROM big",
]


class ValidatorTests(unittest.TestCase):
    def test_bad_queries_rejected(self):
        for sql in BAD:
            with self.subTest(sql=sql), self.assertRaises(QueryValidationError):
                validate_sql(sql)

    def test_non_string_rejected(self):
        for v in (None, 5, ["SELECT 1"]):
            with self.assertRaises(QueryValidationError):
                validate_sql(v)

    def test_good_queries_accepted(self):
        for sql in GOOD:
            with self.subTest(sql=sql):
                self.assertTrue(validate_sql(sql).lower().startswith(("select", "with")))

    def test_length_limit(self):
        with self.assertRaises(QueryValidationError):
            validate_sql("SELECT " + "1," * 3000 + "1")


class ExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def test_select_works_with_params(self):
        r = execute_readonly(self.ws.conn, "SELECT room_count FROM plan_summary WHERE plan_id = ?", (3,))
        self.assertEqual(r.rows, [(6,)])

    def test_validation_runs_before_execution_and_db_unchanged(self):
        before = table_counts(self.ws.conn)
        for sql in BAD:
            try:
                execute_readonly(self.ws.conn, sql)
            except QueryValidationError:
                pass
        self.assertEqual(table_counts(self.ws.conn), before)

    def test_authorizer_blocks_writes_even_if_validator_is_bypassed(self):
        before = table_counts(self.ws.conn)
        self.ws.conn.set_authorizer(executor._authorizer)
        try:
            for sql in ("DELETE FROM rooms", "INSERT INTO meta VALUES ('x','y')", "DROP TABLE rooms",
                        "SELECT * FROM sqlite_master", "PRAGMA query_only = OFF", "SELECT * FROM meta"):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.DatabaseError):
                    self.ws.conn.execute(sql)
        finally:
            self.ws.conn.set_authorizer(None)
        self.assertEqual(table_counts(self.ws.conn), before)

    def test_connection_state_restored_after_query(self):
        execute_readonly(self.ws.conn, "SELECT 1")
        self.assertEqual(self.ws.conn.execute("PRAGMA query_only").fetchone()[0], 0)
        with self.ws.conn:
            self.ws.conn.execute("INSERT INTO meta(key, value) VALUES ('probe', '1') ON CONFLICT(key) DO NOTHING")

    def test_timeout_stops_runaway_query(self):
        sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
        with self.assertRaises(QueryTimeoutError):
            execute_readonly(self.ws.conn, sql, timeout_s=0.3)

    def test_row_cap(self):
        r = execute_readonly(self.ws.conn, "SELECT * FROM rooms", max_rows=5)
        self.assertEqual(len(r.rows), 5)
        self.assertTrue(r.truncated)


if __name__ == "__main__":
    unittest.main()
