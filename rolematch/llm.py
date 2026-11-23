"""LLM chat clients with function calling.

Gemini is the default (free tier). An OpenAI-compatible path exists for local
runs with your own key; it follows the same tool-loop contract. Both return a
list of "events": text deltas and tool calls, so the agent loop is
provider-agnostic.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from . import config
from .tools import to_gemini_declarations, to_openai_tools

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_BASE = "https://api.openai.com/v1"
_TIMEOUT = httpx.Timeout(45.0, connect=10.0)

SYSTEM_PROMPT = """You are RoleMatch AI, a sharp conversational job-discovery assistant.

How you work:
- When the user describes what they want (even vaguely), call search_jobs with a
  faithful natural-language query plus any structured filters they stated.
- Remember durable preferences with set_preferences when the user states them
  ("I'm looking for remote backend work in Bengaluru, 8 LPA minimum").
- Resolve references like "the second one" or "that Stripe role" using the
  position numbers / job ids from earlier tool results in this conversation.
- Offer to save interesting roles with save_job, and answer "what did I save?"
  with list_saved_jobs.
- Keep answers short and human. Surface title, company, location, salary band
  and why each role fits. Never invent listings: only talk about jobs the
  tools returned.
- If nothing matches, say so and suggest relaxing a filter instead of
  pretending.
- The dataset is a sample catalog for demonstration; if asked, be upfront
  that listings are synthetic demo data.
"""


class LLMError(RuntimeError):
    pass


async def _gemini_generate(model: str, contents: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "tools": [{"function_declarations": to_gemini_declarations()}],
        "tool_config": {"function_calling_config": {"mode": "AUTO"}},
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{GEMINI_BASE}/models/{model}:generateContent",
            params={"key": config.GEMINI_API_KEY},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


async def gemini_step(contents: list[dict[str, Any]]) -> dict[str, Any]:
    """One model turn; returns the candidate's content dict (parts list)."""
    models = [config.CHAT_MODEL] + [
        m for m in config.GEMINI_FALLBACK_MODELS if m != config.CHAT_MODEL
    ]
    last_error: Exception | None = None
    for model in models:
        try:
            data = await _gemini_generate(model, contents)
            candidates = data.get("candidates") or []
            if not candidates:
                raise LLMError(f"gemini returned no candidates on {model}")
            return candidates[0].get("content") or {"role": "model", "parts": []}
        except (httpx.HTTPError, LLMError) as exc:  # try the next model
            last_error = exc
    raise LLMError(f"all gemini models failed: {last_error}")


def gemini_function_response_parts(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "user",
        "parts": [
            {"functionResponse": {"name": r["name"], "response": {"result": r["result"]}}}
            for r in results
        ],
    }


async def openai_step(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """OpenAI tool-calling step (used when LLM_PROVIDER=openai)."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{OPENAI_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={
                "model": config.OPENAI_CHAT_MODEL,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                "tools": to_openai_tools(),
                "tool_choice": "auto",
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]


def parse_gemini_parts(content: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Split a model content into (text, [function calls])."""
    text_parts: list[str] = []
    calls: list[dict[str, Any]] = []
    for part in content.get("parts", []):
        if "text" in part and part["text"]:
            text_parts.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            calls.append({"name": fc.get("name", ""), "args": fc.get("args") or {}})
    return "\n".join(text_parts).strip(), calls


def dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
