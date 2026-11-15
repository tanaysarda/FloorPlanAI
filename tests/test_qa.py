import os
import unittest
from unittest import mock

from assistant.executor import execute_readonly
from assistant.llm import LLMDecision, LLMError, LLMProvider, get_provider
from assistant.qa_engine import QAEngine
from database import db
from tests.helpers import Workspace, table_counts


class FakeProvider(LLMProvider):
    name = "fake"
    def __init__(self, decision=None, exc=None):
        self.decision, self.exc, self.calls = decision, exc, 0
    def generate(self, question, schema_text):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.decision


class DeterministicQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()
        cls.engine = QAEngine(cls.ws.db_path, None)
        cls.gt = {i + 1: p for i, p in enumerate(sorted(cls.ws.gt["plans"], key=lambda p: p["name"]))}

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def test_counts_match_ground_truth_for_every_plan(self):
        for pid, p in self.gt.items():
            for word, n in (("rooms", len(p["rooms"])), ("doors", len(p["doors"])), ("windows", len(p["windows"]))):
                r = self.engine.ask(f"How many {word} does plan {pid} have?")
                self.assertEqual(r.status, "answered")
                self.assertEqual(r.as_dicts()[0]["value"], n, (pid, word))
                self.assertIn(str(n), r.answer)

    def test_largest_and_smallest(self):
        r = self.engine.ask("What is the largest room in plan 7?")
        want = max(self.gt[7]["rooms"], key=lambda x: x["area_px"])["area_px"]
        self.assertEqual(r.as_dicts()[0]["area_px"], want)
        self.assertIn("pixels", r.answer)
        r = self.engine.ask("what's the smallest room in plan 3")
        self.assertEqual(r.as_dicts()[0]["area_px"], min(x["area_px"] for x in self.gt[3]["rooms"]))

    def test_global_largest_room(self):
        r = self.engine.ask("What is the largest room across all plans?")
        self.assertEqual(r.status, "answered")
        self.assertEqual(r.rows[0][3], max(x["area_px"] for p in self.gt.values() for x in p["rooms"]))

    def test_filter_and_number_words(self):
        a = self.engine.ask("Which plans have more than 3 doors?")
        b = self.engine.ask("Which plans have more than three doors?")
        want = sorted(i for i, p in self.gt.items() if len(p["doors"]) > 3)
        self.assertEqual([r["plan_id"] for r in a.as_dicts()], want)
        self.assertEqual(a.rows, b.rows)

    def test_rooms_without_windows(self):
        r = self.engine.ask("Which room in plan 4 has no windows?")
        self.assertEqual([x["room_name"] for x in r.as_dicts()], ["hallway #2"])
        self.assertIn("SELECT", r.sql)

    def test_average(self):
        r = self.engine.ask("What is the average number of rooms across all plans?")
        self.assertEqual(r.as_dicts()[0]["value"], 4.75)

    def test_invalid_plan(self):
        r = self.engine.ask("How many rooms does plan 12 have?")
        self.assertEqual(r.status, "invalid_plan")
        self.assertIn("does not exist", r.answer)

    def test_ambiguous_and_multiple_plans(self):
        self.assertEqual(self.engine.ask("How many doors are there?").status, "clarification")
        self.assertEqual(self.engine.ask("What is the largest room?").status, "clarification")
        self.assertEqual(self.engine.ask("How many rooms in plan 1 and plan 2?").status, "clarification")

    def test_selected_plan_context(self):
        r = self.engine.ask("How many rooms does this plan have?", selected_plan_id=2)
        self.assertEqual(r.as_dicts()[0]["value"], 4)

    def test_unsupported_questions_are_declined_not_invented(self):
        for q in ("What is the area of plan 2 in square metres?", "How long is the kitchen wall in feet in plan 1?",
                  "Where is the entrance in plan 3?", "What colour is the bedroom in plan 2?",
                  "How do I get from the entrance to room 3 in plan 3?", "Tell me a joke"):
            with self.subTest(q=q):
                r = self.engine.ask(q)
                self.assertEqual(r.status, "unsupported")
                self.assertEqual(r.rows, [])
                self.assertIsNone(r.sql)

    def test_empty_result(self):
        r = self.engine.ask("Which plans have more than 20 doors?")
        self.assertEqual(r.status, "no_results")
        self.assertIn("No plans", r.answer)

    def test_bad_input(self):
        for q in ("", "   ", None, "x" * 600):
            self.assertEqual(self.engine.ask(q).status, "invalid_input", repr(q)[:20])

    def test_every_answer_traceable_to_sql_and_rows(self):
        qs = ["How many rooms does plan 3 have?", "Which plans have more than 3 doors?",
              "What is the average number of rooms across all plans?", "Which rooms have no windows?",
              "List the rooms in plan 2", "Which plan has the most rooms?", "How many bedrooms are in plan 6?"]
        for q in qs:
            r = self.engine.ask(q)
            self.assertEqual(r.status, "answered", q)
            again = execute_readonly(self.ws.conn, r.sql, r.params)
            self.assertEqual(again.rows, r.rows, q)
            self.assertEqual(again.columns, r.columns, q)

    def test_questions_never_modify_database(self):
        before = table_counts(self.ws.conn)
        for q in ("DELETE FROM rooms", "drop table rooms; how many rooms", "How many rooms in plan 1?"):
            self.engine.ask(q)
        self.assertEqual(table_counts(self.ws.conn), before)


class LLMPathTests(unittest.TestCase):
    """The LLM path is tested with fake providers only; no live API was available."""

    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def test_llm_sql_is_validated_executed_and_grounded(self):
        prov = FakeProvider(LLMDecision("sql", sql="SELECT COUNT(*) AS n FROM rooms WHERE plan_id = 3"))
        r = QAEngine(self.ws.db_path, prov).ask("how many rooms in three?")
        self.assertEqual((r.status, r.source, r.rows), ("answered", "llm", [(6,)]))
        self.assertEqual(r.answer, "Result: 6.")

    def test_malicious_llm_sql_blocked_and_fallback_used(self):
        before = table_counts(self.ws.conn)
        prov = FakeProvider(LLMDecision("sql", sql="DELETE FROM rooms"))
        r = QAEngine(self.ws.db_path, prov).ask("How many rooms does plan 3 have?")
        self.assertEqual(r.source, "deterministic")
        self.assertEqual(r.as_dicts()[0]["value"], 6)
        self.assertTrue(r.notes)
        self.assertEqual(table_counts(self.ws.conn), before)

    def test_api_failure_falls_back_without_leaking_details(self):
        prov = FakeProvider(exc=LLMError("secret-key-abc123 https://internal.example"))
        r = QAEngine(self.ws.db_path, prov).ask("How many rooms does plan 3 have?")
        self.assertEqual(r.status, "answered")
        self.assertNotIn("secret", str(r.to_dict()))

    def test_llm_decline_is_reported_as_unsupported(self):
        prov = FakeProvider(LLMDecision("decline", reason="No scale information."))
        r = QAEngine(self.ws.db_path, prov).ask("Size in metres?")
        self.assertEqual((r.status, r.source), ("unsupported", "llm"))

    def test_llm_empty_result_is_stated_plainly(self):
        prov = FakeProvider(LLMDecision("sql", sql="SELECT * FROM rooms WHERE plan_id = 3 AND area_px < 0"))
        self.assertEqual(QAEngine(self.ws.db_path, prov).ask("tiny rooms?").status, "no_results")

    def test_invalid_plan_short_circuits_before_llm_call(self):
        prov = FakeProvider(LLMDecision("sql", sql="SELECT 1"))
        r = QAEngine(self.ws.db_path, prov).ask("How many rooms does plan 99 have?")
        self.assertEqual(r.status, "invalid_plan")
        self.assertEqual(prov.calls, 0)

    def test_deterministic_mode_ignores_provider(self):
        prov = FakeProvider(LLMDecision("sql", sql="SELECT 1"))
        r = QAEngine(self.ws.db_path, prov, mode="deterministic").ask("How many rooms does plan 3 have?")
        self.assertEqual((r.source, prov.calls), ("deterministic", 0))


class RobustnessTests(unittest.TestCase):
    def test_no_api_key_means_no_provider_and_engine_still_works(self):
        ws = Workspace()
        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(get_provider())
                self.assertEqual(QAEngine(ws.db_path, get_provider()).ask(
                    "How many rooms does plan 1 have?").status, "answered")
        finally:
            ws.close()

    def test_missing_database_does_not_crash_or_leak_paths(self):
        r = QAEngine("/nonexistent/dir/x.db").ask("How many rooms does plan 1 have?")
        self.assertEqual(r.status, "error")
        self.assertNotIn("/nonexistent", r.answer)

    def test_empty_database(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.db"
            c = db.connect(p); db.init_db(c); c.close()
            e = QAEngine(p)
            self.assertEqual(e.ask("What is the average number of rooms across all plans?").status, "no_results")
            self.assertEqual(e.ask("How many rooms does plan 1 have?").status, "invalid_plan")
            self.assertEqual(e.ask("Which rooms have no windows?").status, "no_results")
            self.assertEqual(e.ask("How many plans are there?").as_dicts()[0]["value"], 0)


if __name__ == "__main__":
    unittest.main()
