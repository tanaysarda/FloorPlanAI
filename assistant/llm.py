"""Optional LLM integration (tool/function calling). NOT required: without a key the deterministic engine is used.

NOTE: these provider classes have not been exercised against a live API in this repository's test
environment (no network/key). They are unit-tested only through a fake provider; see README."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about building floor plans using ONLY a read-only SQLite database.
Always call exactly one tool: run_sql_query (a single SELECT) or decline (when the data cannot answer).
Rules: use only the tables/views in the schema; never invent data; all areas are PIXEL areas and no real-world
scale exists, so decline questions needing metres/feet, furniture, colours, costs, or the entrance location.
'Plan N' means floor_plans.id = N. Prefer the views plan_summary and room_details. SELECT only."""

TOOLS = [
    {"name": "run_sql_query", "description": "Run one read-only SQLite SELECT query against the floor plan database.",
     "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}},
    {"name": "decline", "description": "Use when the database cannot answer the question.",
     "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
]


class LLMError(Exception):
    """Any LLM failure. Details are logged, never shown to end users."""


@dataclass
class LLMDecision:
    kind: str                 # "sql" | "decline"
    sql: str = ""
    reason: str = ""


class LLMProvider:
    name = "base"

    def generate(self, question: str, schema_text: str) -> LLMDecision:  # pragma: no cover - interface
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("anthropic package not installed") from exc
        self._client, self._model = anthropic.Anthropic(api_key=api_key), model

    def generate(self, question: str, schema_text: str) -> LLMDecision:
        try:
            resp = self._client.messages.create(
                model=self._model, max_tokens=600, system=SYSTEM_PROMPT + "\n\nSCHEMA:\n" + schema_text,
                tools=[{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                       for t in TOOLS],
                tool_choice={"type": "any"}, messages=[{"role": "user", "content": question}])
            for block in resp.content:
                if getattr(block, "type", "") == "tool_use":
                    if block.name == "run_sql_query":
                        return LLMDecision("sql", sql=str(block.input.get("sql", "")))
                    return LLMDecision("decline", reason=str(block.input.get("reason", "")))
        except Exception as exc:  # noqa: BLE001 - deliberately broad; details only logged
            log.warning("LLM call failed: %r", exc)
            raise LLMError("LLM call failed") from exc
        raise LLMError("LLM returned no tool call")


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str):
        try:
            import openai
        except ImportError as exc:
            raise LLMError("openai package not installed") from exc
        self._client, self._model = openai.OpenAI(api_key=api_key), model

    def generate(self, question: str, schema_text: str) -> LLMDecision:
        import json
        try:
            resp = self._client.chat.completions.create(
                model=self._model, tool_choice="required",
                messages=[{"role": "system", "content": SYSTEM_PROMPT + "\n\nSCHEMA:\n" + schema_text},
                          {"role": "user", "content": question}],
                tools=[{"type": "function", "function": t} for t in TOOLS])
            calls = resp.choices[0].message.tool_calls or []
            if calls:
                args = json.loads(calls[0].function.arguments or "{}")
                if calls[0].function.name == "run_sql_query":
                    return LLMDecision("sql", sql=str(args.get("sql", "")))
                return LLMDecision("decline", reason=str(args.get("reason", "")))
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM call failed: %r", exc)
            raise LLMError("LLM call failed") from exc
        raise LLMError("LLM returned no tool call")


def get_provider() -> Optional[LLMProvider]:
    """Return a provider if an API key (and SDK) is configured, else None. Never raises."""
    choice = os.environ.get("LLM_PROVIDER", "").strip().lower()
    try:
        if choice in ("", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicProvider(os.environ["ANTHROPIC_API_KEY"], os.environ.get("LLM_MODEL", "claude-sonnet-4-6"))
        if choice in ("", "openai") and os.environ.get("OPENAI_API_KEY"):
            return OpenAIProvider(os.environ["OPENAI_API_KEY"], os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    except LLMError as exc:
        log.warning("LLM provider unavailable: %s", exc)
    return None
