"""The conversational agent: multi-turn memory + tool loop.

State lives in Postgres (sessions + messages tables), so any serverless
instance can continue a conversation. The loop caps tool rounds so a chat turn
always finishes well inside the function time limit.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from . import config, llm, tools

MAX_TOOL_ROUNDS = 4


async def load_history(conn, session_id: UUID) -> list[dict[str, str]]:
    rows = await conn.fetch(
        "SELECT role, content FROM messages WHERE session_id = $1 "
        "ORDER BY id DESC LIMIT $2",
        session_id,
        config.HISTORY_LIMIT,
    )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


async def save_message(conn, session_id: UUID, role: str, content: str) -> None:
    await conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES ($1, $2, $3)",
        session_id,
        role,
        content,
    )


def _trim_jobs_for_context(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool results going back to the model stay compact (saves tokens)."""
    return [
        {
            "job_id": j["job_id"],
            "position": j.get("position"),
            "title": j["title"],
            "company": j["company"],
            "location": j["location"],
            "work_mode": j["work_mode"],
            "salary_min_lpa": j["salary_min_lpa"],
            "salary_max_lpa": j["salary_max_lpa"],
            "skills": j["skills"][:8],
            "match_score": j.get("match_score"),
            "match_reasons": j.get("match_reasons"),
            "description": j["description"][:280],
        }
        for j in jobs
    ]


async def handle_message(conn, session_id: UUID, text: str) -> dict[str, Any]:
    """Run one chat turn. Returns reply text plus any jobs to render as cards."""
    prefs_row = await conn.fetchrow("SELECT prefs FROM sessions WHERE id = $1", session_id)
    prefs = dict(prefs_row["prefs"]) if prefs_row else {}

    history = await load_history(conn, session_id)

    # Gemini content format: roles user/model, text parts.
    contents: list[dict[str, Any]] = [
        {"role": "user" if h["role"] == "user" else "model",
         "parts": [{"text": h["content"]}]}
        for h in history
    ]
    contents.append({"role": "user", "parts": [{"text": text}]})

    collected_jobs: list[dict[str, Any]] = []
    reply_text = ""
    error: str | None = None

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            if config.LLM_PROVIDER == "openai":
                reply_text = await _openai_loop(conn, session_id, prefs, history, text)
                break
            content = await llm.gemini_step(contents)
        except llm.LLMError as exc:
            error = str(exc)
            break

        text_out, calls = llm.parse_gemini_parts(content)
        if text_out:
            reply_text = text_out

        if not calls:
            break

        contents.append(content)
        results = []
        for call in calls:
            result = await tools.execute_tool(
                call["name"], call["args"], conn, session_id, prefs
            )
            if call["name"] == "search_jobs" and result.get("jobs"):
                collected_jobs = result["jobs"]
                result = {**result, "jobs": _trim_jobs_for_context(result["jobs"])}
            if call["name"] == "set_preferences" and result.get("preferences"):
                prefs = result["preferences"]
            results.append({"name": call["name"], "result": result})
        contents.append(llm.gemini_function_response_parts(results))

    if not reply_text:
        reply_text = (
            "My model endpoint is busy right now - try again in a moment."
            if error
            else "I couldn't put together a reply - try rephrasing?"
        )

    await save_message(conn, session_id, "user", text)
    await save_message(conn, session_id, "model", reply_text)

    return {
        "reply": reply_text,
        "jobs": collected_jobs,
        "error": error,
    }


async def _openai_loop(conn, session_id, prefs, history, text) -> str:
    """Single OpenAI tool loop (local-dev path, exercised without CI keys)."""
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": text})
    for _ in range(MAX_TOOL_ROUNDS):
        message = await llm.openai_step(messages)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content") or ""
        messages.append(message)
        for tc in tool_calls:
            args = json.loads(tc["function"].get("arguments") or "{}")
            result = await tools.execute_tool(
                tc["function"]["name"], args, conn, session_id, prefs
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": llm.dumps_compact(result),
                }
            )
    return "I hit my reasoning limit on that one - ask it in a smaller step?"
